"""Ultralight IPC client and protocol for the Dynamic IPC Bridge (FEA Studio).

Pure Python (standard library) – NO import of PySide6 or other heavy libraries!
Allows the MCP server and other clients to find out whether the FEA Studio GUI is running,
and to redirect commands transparently into it for live manipulation of the model, mesh and results.

On Windows: named pipe r"\\\\\\\\.\\\\pipe\\\\feastudio_live_ipc_<user>"
On Linux / macOS: UNIX domain socket in $XDG_RUNTIME_DIR, otherwise in a private folder (0700) in the temporary directory.

Security (17. 10. 2026, protocol 2 since 9. 11. 2026): at start the GUI creates a secret session key (file `<name>.token`
in the user data folder, readable only by the owner, 0600). One JSON line per message, HMAC-SHA256 = `mac(key, label, …)`:

  1. client → {"hello": cnonce, "v": 2}
  2. server → {"v": 2, "snonce": snonce, "proof": mac("srv", cnonce, snonce)}     (snonce = new random, one-time)
  3. client verifies the proof (a fake server with the pipe name taken gets NOTHING – not even the arguments) →
     {"body": json, "mac": mac("req", snonce, body)}
  4. server verifies (snonce is consumed – a replayed request is refused) → {"ack": true, "proof": mac("ack", snonce)}
  5. server → result {"status": …, "rmac": mac("res", snonce, canonical result)}

A v1 request ({"nonce", "body", "mac"} without the hello) is refused with a version message.
"""
from __future__ import annotations

import errno
import getpass
import hashlib
import hmac
import io
import json
import logging
import os
import queue
import secrets
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("fea_core.ipc")

PIPE_NAME = "feastudio_live_ipc"
ACTIVE_SESSION_FILENAME = "active_session.json"
MAX_LINE = 256 << 20            # largest protocol line (a reply with images in base64)
ACK_TIMEOUT = 30.0              # the GUI must confirm the request (prove the key) within this time
PROTOCOL_VERSION = 2
V1_REFUSED = ("IPC protocol version 1 is no longer supported by this FEA Studio window – update the client "
              "(MCP server / CAD add-in) to the same FEA Studio version")


def _user_tag() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 – without the user's environment variables
        user = str(os.getuid()) if hasattr(os, "getuid") else "user"
    return hashlib.sha1(user.encode("utf-8", "replace")).hexdigest()[:8]


def default_pipe_name() -> str:
    """Name of the pipe / socket for the current user (another user of the same PC has another one)."""
    return f"{PIPE_NAME}_{_user_tag()}"


def _private_runtime_dir() -> Path:
    """Private folder for the unix socket: $XDG_RUNTIME_DIR, otherwise <tmp>/feastudio-<uid> with rights 0700."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        return Path(xdg)
    d = Path(tempfile.gettempdir()) / f"feastudio-{os.getuid() if hasattr(os, 'getuid') else _user_tag()}"
    d.mkdir(mode=0o700, exist_ok=True)
    st = d.stat()
    if hasattr(os, "getuid") and (st.st_uid != os.getuid() or st.st_mode & 0o077):
        raise PermissionError(f"The folder for IPC socket {d} does not belong to the user or is not private")
    return d


def get_pipe_path(name: str | None = None) -> str:
    """Returns the system path of the named pipe (Windows) or of the unix socket (Linux)."""
    name = name or default_pipe_name()
    if sys.platform == "win32":
        return rf"\\.\pipe\{name}"
    return str(_private_runtime_dir() / name)


def server_name(name: str | None = None) -> str:
    """What to pass to QLocalServer.listen: Windows the pipe name, unix the full path of the socket."""
    name = name or default_pipe_name()
    return name if sys.platform == "win32" else get_pipe_path(name)


def _name_of(pipe_path: str) -> str:
    return pipe_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def token_path(name: str | None = None) -> Path:
    """File with the secret session key of the GUI (readable only by the owner)."""
    from .config import user_data_dir
    return user_data_dir() / "mcp" / "ipc" / f"{name or default_pipe_name()}.token"


def _write_private(p: Path, text: str) -> None:
    """Writes a secret file readable only by the owner (0600; atomically through a temporary file)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(text)
    os.replace(tmp, p)


def write_token(name: str | None = None) -> str:
    """Creates a new session key and writes it (the GUI calls it after a successful listen)."""
    tok = secrets.token_hex(32)
    _write_private(token_path(name), tok)
    return tok


# ---------------------------------------------------------------- tokens of the HTTP servers (REST API, MCP over HTTP)
SERVER_TOKEN_KINDS = ("rest", "mcp-http")


def server_token_path(kind: str = "rest") -> Path:
    """Persistent per-user token of an HTTP server (`rest` = REST API, `mcp-http` = MCP sse / streamable-http)."""
    if kind not in SERVER_TOKEN_KINDS:
        raise ValueError(f"unknown server token kind {kind!r}")
    from .config import user_data_dir
    return user_data_dir() / "tokens" / f"{kind}.token"


def read_server_token(kind: str = "rest") -> str | None:
    """Token of the HTTP server from the user file, None when it was not created yet (clients: CAD adapters, scripts)."""
    try:
        tok = server_token_path(kind).read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return None
    return tok if len(tok) >= 32 else None


def ensure_server_token(kind: str = "rest") -> str:
    """Token of the HTTP server: the existing one, otherwise a new random one written to the user file (0600).
    On unix a file readable by others is replaced by a new token.

    The first token is created exclusively (O_EXCL): two servers starting at the same moment get the SAME token (the one
    that loses the race reads the winner's) – otherwise the last writer would win and the first server would refuse the
    clients that read the file afterwards."""
    p = server_token_path(kind)
    for attempt in range(50):
        tok = read_server_token(kind)
        if tok is not None:
            try:
                private = os.name == "nt" or not (p.stat().st_mode & 0o077)
            except OSError:
                private = False
            if private:
                return tok
            tok = secrets.token_urlsafe(32)            # readable by others: a new token replaces it
            _write_private(p, tok)
            return tok
        if p.exists():
            if attempt < 49:                             # another server is writing it right now (empty / partial)
                time.sleep(0.02)
                continue
            tok = secrets.token_urlsafe(32)            # a damaged file (too short) that nobody completes – replace it
            _write_private(p, tok)
            return tok
        tok = secrets.token_urlsafe(32)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:                          # the other server was faster – take its token
            continue
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(tok)
        return tok
    raise OSError(f"Cannot create the server token {p}")


def read_token(name: str | None = None) -> str | None:
    try:
        tok = token_path(name).read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return None
    return tok or None


def remove_token(name: str | None = None, token: str | None = None) -> None:
    """Deletes the key file (only if it contains our `token` – a second window does not delete the key of the first one)."""
    try:
        if token is None or read_token(name) == token:
            token_path(name).unlink()
    except OSError:
        pass


def result_body(resp: dict) -> bytes:
    """Canonical JSON of the result for signing (without the field rmac; json dumps→loads→dumps is stable in Python)."""
    return json.dumps({k: v for k, v in resp.items() if k != "rmac"}, ensure_ascii=False, sort_keys=True).encode("utf-8")


def mac(token: str, *parts: bytes) -> str:
    return hmac.new(token.encode("ascii"), b"\n".join(parts), hashlib.sha256).hexdigest()


ERROR_FILE_NOT_FOUND, ERROR_SEM_TIMEOUT, ERROR_PIPE_BUSY = 2, 121, 231
_wait_named_pipe = None


def _wait_pipe(path: str, timeout_ms: int) -> tuple[bool, int]:
    """WaitNamedPipeW: (an instance is free, Windows error code). It does NOT connect to the pipe (os.path.exists on
    \\\\.\\pipe\\… opens it = a phantom connection to the server per call, and returns False while all instances are busy)."""
    global _wait_named_pipe
    if _wait_named_pipe is None:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.WaitNamedPipeW
        fn.argtypes, fn.restype = [wintypes.LPCWSTR, wintypes.DWORD], wintypes.BOOL
        _wait_named_pipe = (fn, ctypes.get_last_error)
    fn, last = _wait_named_pipe
    ok = fn(path, max(1, int(timeout_ms)))
    return bool(ok), (0 if ok else int(last()))


def pipe_exists(pipe_path: str) -> bool:
    """Does a server listen on the pipe / socket? Windows: without connecting; busy instances count as listening."""
    if sys.platform != "win32":
        return os.path.exists(pipe_path)
    try:
        ok, err = _wait_pipe(pipe_path, 1)
    except (OSError, AttributeError, ImportError):       # no ctypes / kernel32: listing of the pipe namespace
        try:
            return _name_of(pipe_path).lower() in {n.lower() for n in os.listdir("\\\\.\\pipe\\\\")}
        except OSError:
            return False
    return ok or err in (ERROR_SEM_TIMEOUT, ERROR_PIPE_BUSY)


def is_gui_running(pipe_path: str | None = None) -> bool:
    """Checks whether the GUI application is open and listening on the IPC pipe / socket (no connection is made).

    FEA_MCP_NO_GUI=1 = the user's GUI (default pipe) is never used – an MCP server that must stay headless (tests: otherwise every
    tool call of the test suite would run in whatever FEA Studio window is open on the machine). An explicit pipe_path is checked.
    """
    if pipe_path is None and os.environ.get("FEA_MCP_NO_GUI", "").strip() not in ("", "0"):
        return False
    try:
        p = pipe_path or get_pipe_path()
        if not (token_path(_name_of(p)).is_file() and pipe_exists(p)):
            return False
        return True
    except OSError:
        return False


def acquire_server_lock(name: str | None = None):
    """Exclusive OS lock of the pipe name for the lifetime of the GUI IPC server → an object to keep (release with
    `release_server_lock`), None = another FEA Studio window owns the name. The lock disappears with the process (crash
    included), so a leftover key file never blocks a new window. Windows allows a second listener on the same pipe
    name – without this lock two windows would overwrite each other's session key."""
    p = token_path(name).with_suffix(".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def release_server_lock(lock) -> None:
    if lock is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        lock.close()


class IpcError(RuntimeError):
    """General error of the IPC communication."""


class GuiNotRunningError(IpcError):
    """The GUI window is not running or does not answer on IPC."""


def _open(path: str):
    """Opens the connection → (file for writing and reading line by line, closing function)."""
    if sys.platform == "win32":
        f = None
        for _ in range(2):
            try:
                f = open(path, "r+b", buffering=0)
                break
            except FileNotFoundError:
                raise GuiNotRunningError(f"GUI pipe not found: {path}") from None
            except OSError as e:
                # Python's open() reports ERROR_PIPE_BUSY as EINVAL without winerror
                busy = getattr(e, "winerror", None) == ERROR_PIPE_BUSY or (e.errno == errno.EINVAL and pipe_exists(path))
                if not busy:
                    raise GuiNotRunningError(f"Cannot connect to the GUI: {e}") from e
                # all instances are busy (the GUI thread has not accepted the previous clients yet): the window LIVES –
                # wait for a free instance; it must not look like "GUI not running" (the tool would run headless)
                try:
                    _wait_pipe(path, 5000)
                except (OSError, AttributeError, ImportError):
                    time.sleep(1.0)
        if f is None:
            raise IpcError("The FEA Studio window does not accept the connection (all IPC pipe instances are busy) – "
                           "the command was not sent")

        def write(data: bytes) -> None:
            view = memoryview(data)
            while view:
                view = view[f.write(view):]
        # reading through a buffer (readline on raw FileIO reads byte by byte = a system call for every byte)
        return io.BufferedReader(f, 1 << 16), write, f.close
    if not os.path.exists(path):
        raise GuiNotRunningError(f"GUI socket not found: {path}")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(10.0)
        s.connect(path)
        s.settimeout(None)
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout) as e:
        s.close()
        raise GuiNotRunningError(f"Cannot connect to the GUI: {e}") from e
    rf = s.makefile("rb")

    def close():
        for c in (rf.close, s.close):
            try:
                c()
            except OSError:
                pass
    return rf, s.sendall, close


def send_command(
    tool: str,
    args: dict[str, Any] | None = None,
    timeout: float | None = None,
    pipe_path: str | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Sends a command over IPC to the running GUI window and waits for the result (JSON lines).

    `timeout` = the longest time to wait for the result (None = no limit – a calculation in the GUI may take long);
    the confirmation from the GUI must arrive within ACK_TIMEOUT. `cancel` (a set event) ends the waiting.

    Returns a dictionary of the answer:
      - on success: {"status": "ok", "result": ...}
      - on error:   {"status": "error", "error": "error description"}

    GuiNotRunningError = nothing reached the GUI (no key, no pipe, the connection closed during the handshake) – the caller
    may safely run the command elsewhere. Any other IpcError = the request may have been executed by the GUI.
    """
    path = pipe_path or get_pipe_path()
    name = _name_of(path)
    token = read_token(name)
    if token is None:
        raise GuiNotRunningError(f"GUI is not running (session key {token_path(name)} missing)")
    body = json.dumps({"tool": tool, "args": args or {}}, ensure_ascii=False)
    cnonce = secrets.token_hex(16)
    hello = json.dumps({"hello": cnonce, "v": PROTOCOL_VERSION}).encode("ascii") + b"\n"

    f, write, close = _open(path)
    lines: queue.Queue = queue.Queue()
    rt: threading.Thread | None = None

    def reader(n: int):                 # n lines, then the end (do not block in another read)
        try:
            for _ in range(n):
                line = f.readline(MAX_LINE)
                lines.put(line)
                if not line or not line.endswith(b"\n"):
                    return
        except Exception as e:  # noqa: BLE001 – pass to the waiting thread
            lines.put(e)

    def next_line(limit: float | None, before_request: bool = False) -> bytes:
        end = None if limit is None else time.monotonic() + limit
        while True:
            if cancel is not None and cancel.is_set():
                raise IpcError("Call cancelled")
            wait = 0.25 if end is None else min(0.25, end - time.monotonic())
            if wait <= 0:
                raise IpcError(f"GUI response timed out ({limit} s)")
            try:
                item = lines.get(timeout=wait)
            except queue.Empty:
                continue
            if isinstance(item, Exception):
                if before_request:      # nothing was sent to the GUI yet → the caller may work without it
                    raise GuiNotRunningError(f"GUI connection error before the request: {item}") from item
                raise IpcError(f"GUI connection error: {item}") from item
            if not item:
                if before_request:
                    raise GuiNotRunningError("GUI closed the connection before the request was sent")
                raise IpcError("GUI closed the connection without a response")
            if not item.endswith(b"\n"):
                raise IpcError("GUI response is too long or incomplete")
            return item

    first = ACK_TIMEOUT if timeout is None else min(ACK_TIMEOUT, timeout)
    try:
        try:
            write(hello)
        except OSError as e:            # the pipe closed at once – nothing was sent
            raise GuiNotRunningError(f"Cannot talk to the GUI: {e}") from e
        # one reader per phase: on Windows a synchronous pipe handle serializes I/O – a write while another thread is
        # blocked in a read on the same handle would wait for that read (deadlock)
        rt = threading.Thread(target=reader, args=(1,), daemon=True, name="fea-ipc-reader")
        rt.start()

        # phase 1: the server proves the session key over our nonce and its own (one-time) nonce
        srv = _parse(next_line(first, before_request=True))
        if "proof" not in srv:
            hint = "" if srv.get("v") else " (the FEA Studio window may be an older version with IPC protocol 1 – update it)"
            raise IpcError(f"GUI rejected the connection: {srv.get('error', 'no proof')}{hint}")
        snonce = str(srv.get("snonce", ""))
        if not (16 <= len(snonce) <= 128) or not hmac.compare_digest(
                str(srv.get("proof", "")), mac(token, b"srv", cnonce.encode(), snonce.encode())):
            # a foreign process (or a stale key file) on the pipe: a hard error, never a silent run without the GUI – the
            # request body was not sent
            raise IpcError("The process on the IPC pipe did not prove the GUI session key – nothing was sent to it")
        # phase 2: the request signed over the server nonce (replay of a recorded request is useless)
        req = {"body": body, "mac": mac(token, b"req", snonce.encode(), body.encode("utf-8"))}
        rt.join(5.0)                    # it has delivered its line and ends at once
        try:
            write(json.dumps(req, ensure_ascii=False).encode("utf-8") + b"\n")
        except OSError as e:
            raise IpcError(f"GUI connection error while sending the request: {e}") from e
        rt = threading.Thread(target=reader, args=(2,), daemon=True, name="fea-ipc-reader")
        rt.start()
        ack = _parse(next_line(first))
        if ack.get("status") == "error" and "proof" not in ack:
            raise IpcError(f"GUI rejected the request: {ack.get('error')}")
        if not hmac.compare_digest(str(ack.get("proof", "")), mac(token, b"ack", snonce.encode())):
            raise IpcError("The GUI did not confirm the request with the session key – the response is discarded")
        res = _parse(next_line(timeout))
        if not hmac.compare_digest(str(res.get("rmac", "")), mac(token, b"res", snonce.encode(), result_body(res))):
            raise IpcError("The result from the IPC pipe is not signed with the GUI session key – discarded")
        res.pop("rmac", None)
        return res
    finally:
        if rt is not None and rt.is_alive():
            # the reading thread hangs in a read (cancel, time limit): closing the pipe on Windows may block
            threading.Thread(target=close, daemon=True).start()
        else:
            close()


def _parse(line: bytes) -> dict[str, Any]:
    try:
        data = json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise IpcError(f"Invalid JSON response from the GUI: {e}") from e
    if not isinstance(data, dict):
        raise IpcError("GUI response is not a JSON object")
    return data


def active_session_path() -> Path:
    """Path to the file with information about the active background session."""
    from .config import user_data_dir
    return user_data_dir() / "mcp" / ACTIVE_SESSION_FILENAME


def save_active_session_info(
    study_name: str,
    step_path: str | Path | None = None,
    study_path: str | Path | None = None,
    workdir: str | Path | None = None,
) -> None:
    """Saves metadata of a study in progress in headless mode for later takeover in the GUI."""
    try:
        p = active_session_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "study_name": study_name,
            "step_path": str(step_path) if step_path else None,
            "study_path": str(study_path) if study_path else None,
            "workdir": str(workdir) if workdir else None,
            "timestamp": time.time(),
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        log.warning("Cannot write active session metadata: %s", e)


def load_active_session_info(max_age_seconds: float = 7200.0) -> dict[str, Any] | None:
    """Loads metadata of a recent running assistant session (if it is not older than max_age_seconds)."""
    p = active_session_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        age = time.time() - float(data.get("timestamp", 0))
        if age <= max_age_seconds:
            return data
    except Exception as e:  # noqa: BLE001
        log.warning("Cannot read active session metadata: %s", e)
    return None


def clear_active_session_info() -> None:
    """Deletes the record of the active session."""
    p = active_session_path()
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass

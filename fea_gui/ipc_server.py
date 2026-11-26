"""IPC server for the FEA Studio GUI (Dynamic IPC Bridge).

Runs in the main process of the GUI (PySide6) and accepts commands from the MCP server or other
client processes through a local Named Pipe (Windows) or a UNIX socket (Linux).
Commands run directly in the context of the main window MainWindow (in the Qt main thread),
so the 3D scene, the study tree and the control panels are updated live in front of the user.
"""
from __future__ import annotations

import hmac
import json
import logging
import secrets
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from fea_core import ipc

if TYPE_CHECKING:
    from .main import MainWindow

log = logging.getLogger("fea_gui.ipc_server")


class IpcServer(QObject):
    """Local IPC server attached to the main window MainWindow."""

    MAX_BUFFER = 64 << 20               # a request without a line end longer than this = the connection is closed

    def __init__(self, main_window: MainWindow, pipe_name: str | None = None):
        parent = main_window if isinstance(main_window, QObject) else None
        super().__init__(parent)
        self.main_window = main_window
        self.pipe_name = pipe_name or ipc.default_pipe_name()
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)   # this user only
        self.clients: list[QLocalSocket] = []
        self._buffers: dict[QLocalSocket, bytearray] = {}
        self._token: str | None = None
        self._pending: dict[QLocalSocket, str] = {}            # server nonce of a finished hello (one-time)
        self._lock = None                                       # OS lock of the pipe name (one owner per user)
        self.warning: str | None = None                         # why this window is NOT controlled by MCP (None = it is)

        # Windows lets a second server listen on the same pipe name: clients would reach either window at random and
        # the second key would replace the first one. The name is owned through an OS lock (released with the process).
        try:
            self._lock = ipc.acquire_server_lock(self.pipe_name)
        except OSError as e:
            log.warning("IPC: the lock of the pipe name could not be created: %s", e)
        if self._lock is None or (ipc.is_gui_running(ipc.get_pipe_path(self.pipe_name)) and self._other_alive()):
            self._warn("Another FEA Studio window already accepts commands from the AI assistant (MCP) – "
                       "this window is not controlled by MCP")
            ipc.release_server_lock(self._lock)
            self._lock = None
            return
        name = ipc.server_name(self.pipe_name)
        # cleanup of a possibly hanging socket after an earlier crash (unix; the name is ours through the lock)
        QLocalServer.removeServer(name)
        if not self.server.listen(name):
            log.warning("IPC server could not start listening on '%s': %s",
                        self.pipe_name, self.server.errorString())
            self._warn(f"The AI assistant (MCP) cannot control this window – the IPC server did not start: "
                       f"{self.server.errorString()}")
            ipc.release_server_lock(self._lock)
            self._lock = None
        else:
            self._token = ipc.write_token(self.pipe_name)
            log.info("IPC server is listening on '%s' (serverName: %s)",
                     self.pipe_name, self.server.fullServerName())

        self.server.newConnection.connect(self._on_new_connection)

    def _warn(self, text: str) -> None:
        """Visible warning: log of the program, the Messages table and the status bar of the main window (after the window
        is built – the server is created at the end of MainWindow.__init__)."""
        self.warning = text
        log.warning("IPC: %s", text)
        mw = self.main_window

        def show():
            try:
                if hasattr(mw, "log_msg"):
                    mw.log_msg(f"Warning: {text}")
                msgs = getattr(mw, "messages", None)
                if msgs is not None and hasattr(msgs, "add"):
                    msgs.add("warning", "mcp", text)
                if hasattr(mw, "statusBar"):
                    mw.statusBar().showMessage(text, 15000)
            except Exception as e:  # noqa: BLE001 – a warning must never break the start of the window
                log.debug("IPC warning not shown: %s", e)
        QTimer.singleShot(0, show)

    def _other_alive(self) -> bool:
        """Does another window with a valid key answer on the pipe? (otherwise it is a leftover after a crash)."""
        try:
            ipc.send_command("fea_gui_status", pipe_path=ipc.get_pipe_path(self.pipe_name), timeout=3.0)
            return True
        except ipc.GuiNotRunningError:          # nobody listens / key missing = leftover after a crash
            return False
        except ipc.IpcError:                    # timeout, foreign answer: the window is alive (e.g. computing) – do not take over
            return True

    def _on_new_connection(self) -> None:
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            if not sock:
                continue
            self.clients.append(sock)
            self._buffers[sock] = bytearray()
            sock.readyRead.connect(lambda s=sock: self._on_ready_read(s))
            sock.disconnected.connect(lambda s=sock: self._on_disconnected(s))

    def _on_disconnected(self, sock: QLocalSocket) -> None:
        if sock in self.clients:
            self.clients.remove(sock)
        self._buffers.pop(sock, None)
        self._pending.pop(sock, None)
        try:
            sock.readyRead.disconnect()          # no further handler on a socket that is being deleted
            sock.deleteLater()
        except (RuntimeError, ReferenceError, TypeError):
            pass

    def _close_later(self, sock: QLocalSocket) -> None:
        """Disconnects the client ONLY after returning from the readyRead handler – a synchronous disconnect inside the handler triggers
        disconnected → deleteLater and further work with the socket ended in a crash (access violation).
        """
        self._buffers.pop(sock, None)
        try:
            sock.readyRead.disconnect()
        except (RuntimeError, TypeError):
            pass
        QTimer.singleShot(0, lambda s=sock: self._safe_disconnect(s))

    @staticmethod
    def _safe_disconnect(sock: QLocalSocket) -> None:
        try:
            sock.disconnectFromServer()
        except RuntimeError:                     # object already deleted
            pass

    def _on_ready_read(self, sock: QLocalSocket) -> None:
        buf = self._buffers.get(sock)
        if buf is None:
            buf = bytearray()
            self._buffers[sock] = buf

        chunk = sock.readAll().data()
        buf.extend(chunk)
        if len(buf) > self.MAX_BUFFER and b"\n" not in chunk:
            log.warning("IPC: request too long – connection closed")
            buf.clear()
            self._close_later(sock)
            return
        while b"\n" in buf:
            line, _, rest = buf.partition(b"\n")
            buf[:] = rest
            if not line.strip():
                continue
            if not self._dispatch_line(sock, line):  # rejected → discard the rest of the buffer, disconnect later
                buf.clear()
                self._close_later(sock)
                return

    def _reject(self, sock: QLocalSocket, why: str, public: str = "Request rejected") -> bool:
        log.warning("IPC: request rejected (%s)", why)
        self._send(sock, {"status": "error", "error": public, "v": ipc.PROTOCOL_VERSION})
        return False

    def _hello(self, sock: QLocalSocket, req: dict) -> bool:
        """Phase 1: proof of the session key over the client nonce and a new server nonce (one per request)."""
        cnonce = str(req.get("hello", ""))
        if req.get("v") != ipc.PROTOCOL_VERSION:
            return self._reject(sock, f"protocol version {req.get('v')!r}",
                                f"Unsupported IPC protocol version {req.get('v')!r} (this window speaks "
                                f"{ipc.PROTOCOL_VERSION}) – use the same FEA Studio version")
        if not (16 <= len(cnonce) <= 128) or sock in self._pending:
            return self._reject(sock, "invalid hello")
        snonce = secrets.token_hex(16)
        self._pending[sock] = snonce
        return self._send(sock, {"v": ipc.PROTOCOL_VERSION, "snonce": snonce,
                                 "proof": ipc.mac(self._token, b"srv", cnonce.encode(), snonce.encode())})

    def _authenticate(self, sock: QLocalSocket, req: dict) -> tuple[str, dict, str]:
        """Phase 2: verifies the HMAC of the request over the server nonce → (tool, arguments, nonce); otherwise an exception."""
        snonce = self._pending.pop(sock, None)          # one-time: a replayed request has no nonce to match
        if snonce is None:
            raise PermissionError("Request without the handshake")
        body = req.get("body")
        if not isinstance(body, str):
            raise PermissionError("Request has no body")
        expect = ipc.mac(self._token, b"req", snonce.encode(), body.encode("utf-8"))
        if not hmac.compare_digest(str(req.get("mac", "")), expect):
            raise PermissionError("Invalid request signature")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("Request body is not a JSON object")
        tool, args = data.get("tool"), data.get("args", {})
        if not tool or not isinstance(tool, str):
            raise ValueError("Missing tool name ('tool') in the JSON request")
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return tool, args, snonce

    def _send(self, sock: QLocalSocket, resp: dict) -> bool:
        try:
            sock.write(json.dumps(resp, ensure_ascii=False).encode("utf-8") + b"\n")
            sock.flush()
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to send IPC response: %s", e)
            return False

    def _dispatch_line(self, sock: QLocalSocket, line: bytes) -> bool:
        """Handles one protocol line; False = rejected (the calling connection is closed)."""
        if self._token is None:
            return self._reject(sock, "IPC server has no session key")
        try:
            req = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            return self._reject(sock, f"invalid JSON: {e}")
        if not isinstance(req, dict):
            return self._reject(sock, "request is not a JSON object")
        if "hello" in req:
            return self._hello(sock, req)
        if "nonce" in req and sock not in self._pending:      # protocol 1 client (request before the server proof)
            return self._reject(sock, "protocol 1 client", ipc.V1_REFUSED)
        try:
            tool, args, nonce = self._authenticate(sock, req)
        except Exception as e:  # noqa: BLE001 – rejection without proof of the key
            return self._reject(sock, str(e))
        # acknowledgement with proof of the key, then the result
        if not self._send(sock, {"ack": True, "proof": ipc.mac(self._token, b"ack", nonce.encode())}):
            return False
        try:
            # Execution in the context of the main window
            result = self.main_window.execute_mcp_action(tool, args)
            resp = {"status": "ok", "result": result}
        except Exception as e:  # noqa: BLE001 – pass errors to the client as an understandable status
            log.exception("Error while executing IPC command")
            resp = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        try:
            resp["rmac"] = ipc.mac(self._token, b"res", nonce.encode(), ipc.result_body(resp))
        except (TypeError, ValueError) as e:            # the result cannot be converted to JSON – send an error (signed)
            resp = {"status": "error", "error": f"The result cannot be sent: {e}"}
            resp["rmac"] = ipc.mac(self._token, b"res", nonce.encode(), ipc.result_body(resp))
        self._send(sock, resp)
        return True

    def close(self) -> None:
        """Stops listening and cleans up the system named pipe / socket."""
        for c in list(self.clients):
            try:                                 # first disconnect the handlers (signals after deletion = crash), then close
                c.readyRead.disconnect()
                c.disconnected.disconnect()
                c.abort()
                c.deleteLater()
            except (RuntimeError, TypeError):
                pass
        self.clients.clear()
        self._buffers.clear()
        self._pending.clear()
        try:
            self.server.newConnection.disconnect()
        except (RuntimeError, TypeError):
            pass
        if self.server.isListening():
            self.server.close()
            QLocalServer.removeServer(ipc.server_name(self.pipe_name))
        if self._token is not None:
            ipc.remove_token(self.pipe_name, self._token)
            self._token = None
        ipc.release_server_lock(self._lock)
        self._lock = None

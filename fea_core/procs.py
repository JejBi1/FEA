"""Helper processes of the solvers (NGSolve worker, mpiexec + hypre workers): cancellable, output into a log file.

`run_tree` is like `config.run_logged` but on cancel / timeout it ends the WHOLE process tree (mpiexec starts the MPI ranks
as its children – killing only mpiexec would leave them computing) and it can pass the output lines on (`on_line`).
The output goes to a file, not to a pipe (a full pipe would block the child); the file is followed by a reader thread.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from . import config, memory


class Canceled(RuntimeError):
    """The process was ended because the calculation was cancelled."""


def kill_tree(pid: int) -> None:
    """Ends the process and all its descendants (children first – the parent could otherwise restart them)."""
    import psutil
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        kids = root.children(recursive=True)
    except psutil.Error:
        kids = []
    for p in reversed(kids):
        try:
            p.kill()
        except psutil.Error:
            pass
    try:
        root.kill()
    except psutil.Error:
        pass
    psutil.wait_procs([*kids, root], timeout=10)


def tail(path: Path, n: int = 3000) -> str:
    """The last n characters of a log file ("" when missing)."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def run_tree(cmd: list, log_path: Path, cancel: threading.Event | None = None, env: dict | None = None,
             on_line: Callable[[str], None] | None = None, timeout: float | None = None,
             cwd: Path | None = None) -> int:
    """Starts cmd (cwd = project root), output (stdout + stderr) into log_path, waits for the end and returns the return code.
    cancel set → the process tree is killed and `Canceled` raised; after timeout [s] → killed + subprocess.TimeoutExpired.
    on_line(line) gets the output lines (from a reader thread of the log file, also the rest after the end).
    env = complete environment (None = config.child_env()).
    """
    log_path = Path(log_path)
    memory.release_unused()                  # the helper gets the memory that this process does not use (fea_core.memory)
    with open(log_path, "w", encoding="utf-8", errors="replace") as out:
        p = subprocess.Popen(cmd, cwd=cwd or config.PROJECT_ROOT, env=env if env is not None else config.child_env(),
                             stdout=out, stderr=subprocess.STDOUT, creationflags=config.NO_WINDOW)
    done = threading.Event()
    reader = None
    if on_line is not None:
        def follow():
            with open(log_path, encoding="utf-8", errors="replace") as f:
                buf = ""
                while True:
                    chunk = f.read()
                    if chunk:
                        buf += chunk
                        *lines, buf = buf.split("\n")
                        for ln in lines:
                            on_line(ln.rstrip("\r"))
                    elif done.is_set():
                        if buf:
                            on_line(buf.rstrip("\r"))
                        return
                    else:
                        time.sleep(0.1)
        reader = threading.Thread(target=follow, daemon=True, name="fea-proc-log")
        reader.start()
    t0 = time.monotonic()
    try:
        while p.poll() is None:
            if cancel is not None and cancel.wait(0.1):
                kill_tree(p.pid)
                p.wait()
                raise Canceled("Analysis canceled by user")
            if cancel is None:
                time.sleep(0.1)
            if timeout is not None and time.monotonic() - t0 > timeout:
                kill_tree(p.pid)
                p.wait()
                raise subprocess.TimeoutExpired(p.args, timeout)
    finally:
        if p.poll() is None:                   # exception in this thread (KeyboardInterrupt) – no orphaned solver
            kill_tree(p.pid)
            p.wait()
        done.set()
        if reader is not None:
            reader.join(timeout=5)
    return p.returncode

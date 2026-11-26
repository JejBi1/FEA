"""Tests of the IPC server (Qt) run in a SEPARATE process from tests/test_ipc.py.

In the full pytest set processEvents would process the leftovers of objects from the earlier GUI tests in the same process (an occasional
crash access violation outside the IPC). Run: python -m tests.ipc_server_cases <function name>.
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import fea_core.ipc as ipc_mod  # noqa: E402
from fea_core.ipc import get_pipe_path, is_gui_running, send_command  # noqa: E402


def test_ipc_server_communication():
    """Check of the two-way communication through QLocalServer and an IPC client."""
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])

    test_pipe = "test_feastudio_ipc_unit"
    pipe_path = get_pipe_path(test_pipe)

    # Mock MainWindow
    mock_win = MagicMock()
    mock_win.execute_mcp_action.side_effect = lambda tool, args: {
        "executed_tool": tool,
        "received_args": args,
        "gui_ok": True,
    }

    server = IpcServer(mock_win, pipe_name=test_pipe)
    try:
        assert is_gui_running(pipe_path)

        res_container = []
        err_container = []

        def client_thread():
            try:
                resp = send_command(
                    "fea_gui_toast",
                    {"message": "Ahoj z testu", "timeout_ms": 3000},
                    pipe_path=pipe_path,
                    timeout=5.0,
                )
                res_container.append(resp)
            except Exception as e:
                err_container.append(e)

        t = threading.Thread(target=client_thread)
        t.start()

        start = time.time()
        while t.is_alive() and (time.time() - start) < 5.0:
            app.processEvents()
            time.sleep(0.01)

        t.join(timeout=1.0)
        assert not err_container, f"Client error: {err_container}"
        assert res_container, "No response received"

        resp = res_container[0]
        assert resp["status"] == "ok"
        res = resp["result"]
        assert res["executed_tool"] == "fea_gui_toast"
        assert res["received_args"]["message"] == "Ahoj z testu"
        assert res["gui_ok"] is True
        assert mock_win.execute_mcp_action.called
    finally:
        server.close()
        assert not is_gui_running(pipe_path)


def test_ipc_rejects_unsigned_and_forged_server():
    """A request without a valid HMAC is refused by the GUI; a server without the key (a planted pipe) is not accepted by the client."""
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])
    test_pipe = "test_feastudio_ipc_auth"
    pipe_path = get_pipe_path(test_pipe)
    mock_win = MagicMock()
    mock_win.execute_mcp_action.side_effect = lambda tool, args: {"ok": True}
    server = IpcServer(mock_win, pipe_name=test_pipe)
    try:
        good = ipc_mod.read_token(test_pipe)
        assert good
        # a client with a wrong key (the key file overwritten) → rejected, the tool is not run
        ipc_mod.token_path(test_pipe).write_text("0" * 64, encoding="ascii")
        errs = []

        def client():
            try:
                send_command("fea_gui_toast", {"message": "x"}, pipe_path=pipe_path, timeout=5.0)
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        t = threading.Thread(target=client)
        t.start()
        start = time.time()
        while t.is_alive() and time.time() - start < 6.0:
            app.processEvents()
            time.sleep(0.01)
        t.join(timeout=1.0)
        assert errs and isinstance(errs[0], ipc_mod.IpcError)
        assert not mock_win.execute_mcp_action.called
    finally:
        ipc_mod.token_path(test_pipe).write_text(good, encoding="ascii")
        server.close()
    assert not ipc_mod.token_path(test_pipe).exists()


def _pump(app, t, limit=6.0):
    start = time.time()
    while t.is_alive() and time.time() - start < limit:
        app.processEvents()
        time.sleep(0.01)
    t.join(timeout=1.0)


def _raw_client(pipe_path, lines_out, read_n, results):
    """Raw protocol client in a thread: sends `lines_out` (bytes, a callable(previous replies) -> bytes, or None = stop),
    reads one reply after each line."""
    import json
    try:
        f, write, close = ipc_mod._open(pipe_path)
        try:
            for item in lines_out:
                data = item(results) if callable(item) else item
                if data is None:
                    break
                write(data)
                for _ in range(read_n.pop(0) if read_n else 1):
                    line = f.readline(1 << 20)
                    results.append(json.loads(line) if line.strip() else None)
                    if not line:
                        return
        finally:
            close()
    except Exception as e:  # noqa: BLE001
        results.append(e)


def test_ipc_protocol_v2_rules():
    """Protocol 2: a v1 request gets a version message; a replayed request (same server nonce) and a request without hello
    are refused; a server that does not prove the key receives no request body."""
    import json
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])
    test_pipe = "test_feastudio_ipc_v2"
    pipe_path = get_pipe_path(test_pipe)
    calls = []
    mock_win = MagicMock()
    mock_win.execute_mcp_action.side_effect = lambda tool, args: calls.append(tool) or {"ok": True}
    server = IpcServer(mock_win, pipe_name=test_pipe)
    try:
        token = ipc_mod.read_token(test_pipe)
        # (1) protocol 1 client: request with its own nonce, without the hello → clear version error, nothing runs
        body = json.dumps({"tool": "fea_gui_toast", "args": {}})
        nonce = "a" * 32
        v1 = json.dumps({"nonce": nonce, "body": body, "mac": ipc_mod.mac(token, b"req", nonce.encode(), body.encode())})
        res = []
        t = threading.Thread(target=_raw_client, args=(pipe_path, [v1.encode() + b"\n"], [1], res))
        t.start()
        _pump(app, t)
        assert res and isinstance(res[0], dict) and "version 1" in res[0]["error"], res
        assert not calls

        # (2) a valid exchange recorded and replayed on a new connection → the old server nonce is unknown → refused
        cn = "b" * 32
        hello = json.dumps({"hello": cn, "v": 2}).encode() + b"\n"
        rec = {}

        def req(replies):
            srv = replies[-1]
            assert ipc_mod.mac(token, b"srv", cn.encode(), srv["snonce"].encode()) == srv["proof"]
            line = json.dumps({"body": body, "mac": ipc_mod.mac(token, b"req", srv["snonce"].encode(), body.encode())})
            rec["line"] = line.encode() + b"\n"
            return rec["line"]
        res = []
        t = threading.Thread(target=_raw_client, args=(pipe_path, [hello, req], [1, 2], res))
        t.start()
        _pump(app, t)
        assert res[1].get("ack") and res[2]["status"] == "ok", res
        assert calls == ["fea_gui_toast"]
        res = []
        t = threading.Thread(target=_raw_client, args=(pipe_path, [hello, lambda r: rec["line"]], [1, 1], res))
        t.start()
        _pump(app, t)
        assert res[1]["status"] == "error" and "proof" not in res[1], res      # new snonce → the old MAC does not match
        res = []
        t = threading.Thread(target=_raw_client, args=(pipe_path, [rec["line"]], [1], res))   # request without hello
        t.start()
        _pump(app, t)
        assert res[0]["status"] == "error", res
        assert calls == ["fea_gui_toast"]

        # (3) a wrong protocol version in the hello → version message
        res = []
        t = threading.Thread(target=_raw_client, args=(pipe_path, [json.dumps({"hello": cn, "v": 3}).encode() + b"\n"], [1], res))
        t.start()
        _pump(app, t)
        assert "version" in res[0]["error"], res
    finally:
        server.close()

    # (4) a fake server on the pipe (the real key in the file, but the server does not know it) never receives the body
    from PySide6.QtNetwork import QLocalServer
    fake_pipe = "test_feastudio_ipc_fake"
    fake = QLocalServer()
    QLocalServer.removeServer(ipc_mod.server_name(fake_pipe))
    assert fake.listen(ipc_mod.server_name(fake_pipe))
    ipc_mod.write_token(fake_pipe)
    got = []

    def on_conn():
        s = fake.nextPendingConnection()

        def rd():
            data = s.readAll().data()
            got.append(data)
            if b"hello" in data:
                s.write(json.dumps({"v": 2, "snonce": "c" * 32, "proof": "0" * 64}).encode() + b"\n")
                s.flush()
        s.readyRead.connect(rd)
        fake._keep = s
    fake.newConnection.connect(on_conn)
    errs = []

    def client():
        try:
            send_command("fea_gui_toast", {"message": "secret-arg"}, pipe_path=get_pipe_path(fake_pipe), timeout=5.0)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    t = threading.Thread(target=client)
    t.start()
    _pump(app, t)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    try:
        assert errs and isinstance(errs[0], ipc_mod.IpcError) and not isinstance(errs[0], ipc_mod.GuiNotRunningError), errs
        assert "did not prove" in str(errs[0])
        assert not any(b"secret-arg" in g for g in got), got
    finally:
        fake.close()
        ipc_mod.remove_token(fake_pipe)


def test_ipc_probe_does_not_connect():
    """is_gui_running must not connect to the pipe (os.path.exists did: one phantom connection per call, and False once
    all instances were busy → MCP ran the tool headless next to a live window)."""
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])
    test_pipe = "test_feastudio_ipc_probe"
    pipe_path = get_pipe_path(test_pipe)
    server = IpcServer(MagicMock(), pipe_name=test_pipe)
    handles = []
    try:
        assert server.server.isListening() and server.warning is None
        assert all(is_gui_running(pipe_path) for _ in range(80))   # no event loop in between = nothing replenished
        app.processEvents()
        assert not server.server.hasPendingConnections()
        if sys.platform == "win32":                                   # all instances busy = the window still runs
            try:
                while len(handles) < 200:
                    handles.append(open(pipe_path, "r+b", buffering=0))
            except OSError:
                pass
            assert is_gui_running(pipe_path)
    finally:
        for h in handles:
            h.close()
        server.close()
    assert not is_gui_running(pipe_path)


def test_ipc_second_server_is_refused_visibly():
    """A second window on the same pipe name does not listen, keeps the key of the first one and says why."""
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])
    test_pipe = "test_feastudio_ipc_two"
    first = IpcServer(MagicMock(), pipe_name=test_pipe)
    win2 = MagicMock()
    second = None
    try:
        tok = ipc_mod.read_token(test_pipe)
        assert first.server.isListening() and tok
        second = IpcServer(win2, pipe_name=test_pipe)
        assert not second.server.isListening()
        assert second.warning and "Another FEA Studio window" in second.warning
        assert ipc_mod.read_token(test_pipe) == tok                   # the key of the first window was not replaced
        for _ in range(5):
            app.processEvents()
        assert win2.messages.add.called and win2.messages.add.call_args[0][0] == "warning"
        second.close()                                                # does not delete the key of the first one
        assert ipc_mod.read_token(test_pipe) == tok
    finally:
        first.close()
    third = IpcServer(MagicMock(), pipe_name=test_pipe)             # the lock was released with the first window
    try:
        assert third.server.isListening() and third.warning is None
    finally:
        third.close()


if __name__ == "__main__":
    globals()[sys.argv[1]]()
    print("OK")

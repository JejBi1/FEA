"""IPC client (protocol 2): a process on the pipe that does not prove the session key (stale key file, foreign process) never
receives the request – the call fails with an IpcError (it is NOT run silently without the GUI), and FEA_MCP_NO_GUI keeps
a headless server (tests) away from a window that is open on the machine."""
from __future__ import annotations

import io
import json

import pytest

from fea_core import ipc


class _Pipe(io.RawIOBase):
    """Fake GUI side of one protocol-2 exchange (as fea_gui/ipc_server.py): hello → {snonce, proof}; request → ack + result."""

    SNONCE = "n" * 32

    def __init__(self, token, sign=True, reject=None):
        self.token, self.sign, self.reject = token, sign, reject
        self.inbuf, self.out, self.lines = b"", io.BytesIO(), []

    def readable(self):
        return True

    def writable(self):
        return True

    def write(self, b):
        self.inbuf += bytes(b)
        while b"\n" in self.inbuf:
            line, self.inbuf = self.inbuf.split(b"\n", 1)
            self.lines.append(line)
            msg = json.loads(line)
            n = self.SNONCE.encode()
            if "hello" in msg:
                if self.reject:
                    self.out = io.BytesIO((json.dumps({"status": "error", "error": self.reject}) + "\n").encode())
                    continue
                proof = ipc.mac(self.token, b"srv", msg["hello"].encode(), n) if self.sign else "0" * 64
                self.out = io.BytesIO((json.dumps({"v": 2, "snonce": self.SNONCE, "proof": proof}) + "\n").encode())
            else:
                assert ipc.mac(self.token, b"req", n, msg["body"].encode("utf-8")) == msg["mac"]
                res = {"status": "ok", "result": "fine"}
                res["rmac"] = ipc.mac(self.token, b"res", n, ipc.result_body(res))
                ack = {"ack": True, "proof": ipc.mac(self.token, b"ack", n)}
                self.out = io.BytesIO((json.dumps(ack) + "\n" + json.dumps(res) + "\n").encode())
        return len(b)

    def readinto(self, b):
        data = self.out.read(len(b))
        b[:len(data)] = data
        return len(data)


@pytest.fixture
def fake_pipe(tmp_path, monkeypatch):
    pipe_file = tmp_path / "feastudio_test_pipe"
    pipe_file.write_text("")                                      # "the pipe exists"
    monkeypatch.setattr(ipc, "token_path", lambda name=None: tmp_path / f"{name or 'x'}.token")
    monkeypatch.setattr(ipc, "pipe_exists", lambda p: True)
    ipc.write_token(pipe_file.name)
    name = pipe_file.name
    token = ipc.read_token(name)
    opened = []

    def serve(pipe):
        def _open(path):
            opened.append(path)
            return io.BufferedReader(pipe, 1 << 16), (lambda data: pipe.write(data)), (lambda: None)
        monkeypatch.setattr(ipc, "_open", _open)
    yield str(pipe_file), token, serve, opened


def test_unproven_server_gets_no_request_and_is_a_hard_error(fake_pipe):
    path, token, serve, _ = fake_pipe
    pipe = _Pipe(token, sign=False)                               # a server that does not know the key
    serve(pipe)
    assert ipc.is_gui_running(path)
    with pytest.raises(ipc.IpcError, match="did not prove") as e:
        ipc.send_command("fea_gui_status", {"secret": "arg"}, pipe_path=path, timeout=2.0)
    assert not isinstance(e.value, ipc.GuiNotRunningError)        # never a silent run without the GUI
    assert len(pipe.lines) == 1 and b"secret" not in pipe.lines[0]    # only the hello was sent, never the body


def test_old_protocol_server_error_is_reported(fake_pipe):
    path, token, serve, _ = fake_pipe
    serve(_Pipe(token, reject="invalid signature"))
    with pytest.raises(ipc.IpcError, match="invalid signature"):
        ipc.send_command("fea_gui_status", pipe_path=path, timeout=2.0)


def test_valid_server_works(fake_pipe):
    path, token, serve, _ = fake_pipe
    serve(_Pipe(token))
    assert ipc.send_command("fea_gui_status", pipe_path=path, timeout=5.0) == {"status": "ok", "result": "fine"}
    assert ipc.is_gui_running(path)


def test_no_gui_env_keeps_the_default_pipe_away(monkeypatch):
    monkeypatch.setenv("FEA_MCP_NO_GUI", "1")
    monkeypatch.setattr(ipc, "pipe_exists", lambda p: True)
    monkeypatch.setattr(ipc, "token_path", lambda name=None: __file__ and __import__("pathlib").Path(__file__))
    assert not ipc.is_gui_running()                              # the default pipe is never used
    assert ipc.is_gui_running(pipe_path="anything")              # an explicit pipe is still checked

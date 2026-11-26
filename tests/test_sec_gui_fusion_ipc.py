"""Fusion add-in: the client copy of the IPC protocol stays in sync with fea_core/ipc.py, and the REST client sends nothing to a
port owner it cannot trust. Pure Python (the add-in is imported without Fusion's `adsk`)."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from fea_core import ipc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def addin():
    p = ROOT / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.py"
    spec = importlib.util.spec_from_file_location("fea_fusion_addin_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.adsk is None
    return mod


def test_protocol_primitives_match_core(addin, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    tok = "ab" * 32
    nonce = "0123456789abcdef" * 2
    assert addin._pipe_name() == ipc.default_pipe_name()
    assert addin.IPC_PROTOCOL == ipc.PROTOCOL_VERSION
    if sys.platform == "win32":
        assert Path(addin._ipc_token_path()) == ipc.token_path()
        assert addin.PIPE_PREFIX + addin._pipe_name() == ipc.get_pipe_path()
    for parts in ((b"req", nonce.encode(), "č{}".encode()), (b"ack", nonce.encode()), (b"res",),
                  (b"srv", nonce.encode(), nonce[::-1].encode())):
        assert addin._ipc_mac(tok, *parts) == ipc.mac(tok, *parts)
    res = {"status": "ok", "result": {"b": [1, 2.5, "ž"], "a": None}, "rmac": "x"}
    assert addin._ipc_result_body(res) == ipc.result_body(res)


class _FakePipe(io.RawIOBase):
    """The GUI side of one protocol-2 exchange, built only from fea_core.ipc primitives (as fea_gui/ipc_server.py does):
    hello → {snonce, proof}; request {body, mac over the server nonce} → ack proof + signed result."""

    SNONCE = "s" * 32

    def __init__(self, token, result, sign=True):
        self.token, self.result, self.sign = token, result, sign
        self.inbuf, self.out, self.cnonce, self.lines = b"", io.BytesIO(), None, []

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
            if "hello" in msg:
                self.cnonce = msg["hello"]
                assert msg["v"] == ipc.PROTOCOL_VERSION
                proof = ipc.mac(self.token, b"srv", self.cnonce.encode(), self.SNONCE.encode()) if self.sign else "0" * 64
                self.out = io.BytesIO((json.dumps({"v": 2, "snonce": self.SNONCE, "proof": proof}) + "\n").encode())
            else:
                n = self.SNONCE.encode()
                assert ipc.mac(self.token, b"req", n, msg["body"].encode("utf-8")) == msg["mac"]
                ack = {"ack": True, "proof": ipc.mac(self.token, b"ack", n)}
                res = {"status": "ok", "result": self.result}
                res["rmac"] = ipc.mac(self.token, b"res", n, ipc.result_body(res))
                self.out = io.BytesIO((json.dumps(ack) + "\n" + json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
        return len(b)

    def readinto(self, b):
        data = self.out.read(len(b))
        b[:len(data)] = data
        return len(data)


def test_ipc_call_against_core_protocol(addin, monkeypatch):
    tok = "cd" * 32
    monkeypatch.setattr(addin, "_ipc_token", lambda: tok)
    monkeypatch.setattr(addin, "_ipc_open", lambda: _FakePipe(tok, {"mode": "update", "x": "ř"}))
    assert addin.ipc_call("fea_cad_update", {"step_path": "a.step"}, timeout=10) == {"mode": "update", "x": "ř"}
    pipe = _FakePipe(tok, {}, sign=False)                      # a server that does not know the key
    monkeypatch.setattr(addin, "_ipc_open", lambda: pipe)
    with pytest.raises(RuntimeError, match="did not prove"):
        addin.ipc_call("fea_cad_update", {"secret": "arg"}, timeout=10)
    assert len(pipe.lines) == 1 and b"secret" not in b"".join(pipe.lines)   # only the hello was sent, never the request body


def test_rest_check_accepts_own_listener(addin):
    """A listener of this process (= our user) passes; a remote server is not checked; no listener = nothing to refuse."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        port = s.getsockname()[1]
        settings = {"server": f"http://127.0.0.1:{port}"}
        addin.check_server(settings)                         # no exception
        if os.name == "nt":
            assert os.getpid() in addin._listener_pids(port)
    finally:
        s.close()
    addin.check_server({"server": "http://192.0.2.1:8765"})


@pytest.mark.skipif(os.name != "nt", reason="Windows owner check")
def test_rest_check_refuses_foreign_owner(addin, monkeypatch):
    sent = []
    monkeypatch.setattr(addin, "_listener_pids", lambda port: {4})            # PID 4 = System: not this user
    monkeypatch.setattr(addin.urllib.request, "urlopen", lambda *a, **k: sent.append(a))
    st = addin.server_status({"server": "http://127.0.0.1:8765", "token": "secret"})
    assert st is None and not sent                                           # neither the token nor a request went out
    assert any("another user" in n for n in addin.NOTES)
    with pytest.raises(addin.UntrustedServer):
        addin._call({"server": "http://localhost:8765", "token": "secret"}, "POST", "/api/import", b"x")
    assert not sent
    assert addin._process_is_mine(os.getpid())
    assert not addin._process_is_mine(4)

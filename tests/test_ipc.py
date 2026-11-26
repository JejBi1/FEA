"""Tests of the Dynamic IPC Bridge (control from an LLM / MCP and detection of the GUI)."""
import json
import threading  # noqa: F401

import pytest

pytest.importorskip("mcp")                       # module mcp (the IPC bridge is exercised through the MCP server)
import anyio  # noqa: E402

import fea_core.ipc as ipc_mod
from fea_core.ipc import (
    GuiNotRunningError,

    get_pipe_path,
    is_gui_running,
    load_active_session_info,
    save_active_session_info,
    clear_active_session_info,
    send_command,
)
from fea_core.mcp.session import Session
from fea_core.mcp.server import build_server


def test_gui_detection_when_not_running():
    """When the GUI is not running, is_gui_running() returns False and send_command raises GuiNotRunningError."""
    fake_pipe = get_pipe_path("non_existent_pipe_for_test_12345")
    assert not is_gui_running(fake_pipe)
    with pytest.raises(GuiNotRunningError):
        send_command("fea_gui_status", pipe_path=fake_pipe, timeout=1.0)


def test_active_session_handoff_info():
    """Writing and reading of the metadata about the active session of the assistant."""
    clear_active_session_info()
    try:
        assert load_active_session_info() is None
        save_active_session_info(
            study_name="Nosnik_L",
            step_path="/path/to/step.stp",
            study_path="/path/to/study.fea",
            workdir="/path/to/workdir",
        )
        info = load_active_session_info()
        assert info is not None
        assert info["study_name"] == "Nosnik_L"
        assert info["step_path"] == "/path/to/step.stp"
        assert info["study_path"] == "/path/to/study.fea"
    finally:
        clear_active_session_info()


def test_headless_tools_gui():
    """The fea_gui_* tools in headless mode safely return gui_running=False without an error."""
    srv = build_server(Session())
    tool_mgr = srv._tool_manager

    # fea_gui_status
    status_tool = tool_mgr.get_tool("fea_gui_status")
    assert status_tool is not None

    # fea_gui_toast
    toast_tool = tool_mgr.get_tool("fea_gui_toast")
    assert toast_tool is not None

    # fea_gui_camera_focus
    cam_tool = tool_mgr.get_tool("fea_gui_camera_focus")
    assert cam_tool is not None


def test_mcp_routing_through_ipc(monkeypatch):
    """When the GUI is active, a call of an MCP tool is automatically redirected to the GUI IPC."""
    monkeypatch.setattr(ipc_mod, "is_gui_running", lambda *_: True)

    calls = []
    def mock_send(tool, args, **kw):
        calls.append((tool, args))
        return {"status": "ok", "result": {"routed_to_gui": True, "tool": tool}}

    monkeypatch.setattr(ipc_mod, "send_command", mock_send)

    srv = build_server(Session())
    tool = srv._tool_manager.get_tool("fea_gui_toast")
    assert tool is not None

    async def call_it():
        return await tool.fn(message="Test IPC bridge")

    res = anyio.run(call_it)
    assert len(calls) == 1
    assert calls[0][0] == "fea_gui_toast"
    assert calls[0][1]["message"] == "Test IPC bridge"
    assert res == {"routed_to_gui": True, "tool": "fea_gui_toast"}


def _run_case(name: str) -> None:
    import subprocess
    import sys
    from pathlib import Path
    p = subprocess.run([sys.executable, "-m", "tests.ipc_server_cases", name], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[1]), timeout=120)
    assert p.returncode == 0 and "OK" in p.stdout, (p.stdout + p.stderr)[-3000:]


def test_ipc_server_communication():
    """Two-way communication through QLocalServer and an IPC client (a separate process)."""
    _run_case("test_ipc_server_communication")


def test_ipc_rejects_unsigned_and_forged_server():
    """A request without a valid HMAC is refused by the GUI and the tool is not run (a separate process)."""
    _run_case("test_ipc_rejects_unsigned_and_forged_server")


def test_ipc_protocol_v2_rules():
    """Protocol 2: v1 refused with a version message, replay refused, a fake server gets no request body (separate process)."""
    _run_case("test_ipc_protocol_v2_rules")


def test_ipc_probe_does_not_connect():
    """is_gui_running neither connects to the pipe nor reports a window with busy pipe instances as closed (separate process)."""
    _run_case("test_ipc_probe_does_not_connect")


def test_ipc_second_server_is_refused_visibly():
    """A second window on the same pipe name is refused through the OS lock and shows a warning (separate process)."""
    _run_case("test_ipc_second_server_is_refused_visibly")


def test_server_lock_is_exclusive(tmp_path, monkeypatch):
    import fea_core.config as cfg
    monkeypatch.setattr(cfg, "user_data_dir", lambda: tmp_path)
    a = ipc_mod.acquire_server_lock("lock_test")
    assert a is not None
    assert ipc_mod.acquire_server_lock("lock_test") is None
    ipc_mod.release_server_lock(a)
    b = ipc_mod.acquire_server_lock("lock_test")
    assert b is not None
    ipc_mod.release_server_lock(b)


def test_server_token_first_creation_is_exclusive(tmp_path, monkeypatch):
    """Servers starting at the same time end with ONE token (previously the last writer won)."""
    import fea_core.config as cfg
    monkeypatch.setattr(cfg, "user_data_dir", lambda: tmp_path)
    start = threading.Barrier(8)
    got = []

    def run():
        start.wait()
        got.append(ipc_mod.ensure_server_token("rest"))
    ts = [threading.Thread(target=run) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert len(got) == 8 and len(set(got)) == 1, got
    assert ipc_mod.read_server_token("rest") == got[0]
    assert ipc_mod.ensure_server_token("rest") == got[0]


def _toast(monkeypatch, send):
    monkeypatch.setattr(ipc_mod, "is_gui_running", lambda *_: True)
    monkeypatch.setattr(ipc_mod, "send_command", send)
    srv = build_server(Session())
    return srv._tool_manager.get_tool("fea_gui_toast")


def test_mcp_falls_back_to_headless_only_before_the_request(monkeypatch):
    """GuiNotRunningError (nothing reached the GUI) → the local session runs the tool."""
    def send(tool, args, **kw):
        raise GuiNotRunningError("closed before the request")
    tool = _toast(monkeypatch, send)
    res = anyio.run(lambda: tool.fn(message="hi"))
    assert json.loads(res)["gui_running"] is False


@pytest.mark.parametrize("behaviour", ["ipc_error", "unexpected"])
def test_mcp_does_not_repeat_a_gui_call_headless(monkeypatch, behaviour):
    """An error after the request was sent, or a reply without status, is a ToolError – NOT a second run in the headless session."""
    from mcp.server.mcpserver.exceptions import ToolError
    import fea_core.mcp.tools_gui as tg
    ran = []
    orig = tg.out
    monkeypatch.setattr(tg, "out", lambda d: ran.append(d) or orig(d))

    def send(tool, args, **kw):
        if behaviour == "ipc_error":
            raise ipc_mod.IpcError("GUI response timed out")
        return {"something": "else"}
    tool = _toast(monkeypatch, send)
    with pytest.raises(ToolError):
        anyio.run(lambda: tool.fn(message="hi"))
    assert not ran


def test_stage_switch_accepts_english_and_czech(monkeypatch):
    """fea_gui_stage_switch: English names (the GUI knows only them), Czech aliases translated before forwarding."""
    from fea_core.mcp.tools_gui import normalize_stage
    from mcp.server.mcpserver.exceptions import ToolError
    assert normalize_stage("results") == "Results" and normalize_stage("Výsledky") == "Results"
    assert normalize_stage("3d TISK") == "3D printing" and normalize_stage("Síť") == "Mesh" and normalize_stage("zatizeni") == "Loads"
    sent = []

    def send(tool, args, **kw):
        sent.append(args)
        return {"status": "ok", "result": {"stage": args["stage"]}}
    monkeypatch.setattr(ipc_mod, "is_gui_running", lambda *_: True)
    monkeypatch.setattr(ipc_mod, "send_command", send)
    tool = build_server(Session())._tool_manager.get_tool("fea_gui_stage_switch")
    assert anyio.run(lambda: tool.fn(stage="Únava")) == {"stage": "Fatigue"}
    with pytest.raises(ToolError):
        anyio.run(lambda: tool.fn(stage="nonsense"))
    assert sent == [{"stage": "Fatigue"}]
    # the list of the tool matches the GUI (read from the source – importing fea_gui.main_defs would pull in Qt and OCP)
    import ast
    from pathlib import Path
    from fea_core.mcp.tools_gui import STAGE_NAMES
    src = (Path(__file__).resolve().parents[1] / "fea_gui" / "main_defs.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "STAGES" for t in n.targets))
    assert list(STAGE_NAMES) == ast.literal_eval(node.value)

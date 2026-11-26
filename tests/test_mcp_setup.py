"""Registration of the MCP server in the clients (fea_core.mcp.setup) on imitated configurations in a temporary folder."""
import json
import os
import subprocess
import sys

import pytest

from fea_core.mcp import setup as ms

# the imitated client configurations use the Windows layout (APPDATA, USERPROFILE)
WINDOWS_LAYOUT = pytest.mark.skipif(sys.platform != "win32", reason="Windows client configuration paths")


def _env(tmp_path):
    home = tmp_path / "home"
    return {"USERPROFILE": str(home), "APPDATA": str(home / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(home / "AppData" / "Local")}, home


@WINDOWS_LAYOUT
def test_register_update_remove(tmp_path):
    env, home = _env(tmp_path)
    appdata = home / "AppData" / "Roaming"
    (appdata / "Claude").mkdir(parents=True)
    (appdata / "Claude" / "claude_desktop_config.json").write_text(
        json.dumps({"mcpServers": {"jiny": {"command": "x"}}, "jineNastaveni": 1}), encoding="utf-8")
    (home / ".cursor").mkdir(parents=True)                       # a client without a configuration → it is created
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "config.toml").write_text('model = "o3"\n\n[mcp_servers.jiny]\ncommand = "y"\n', encoding="utf-8")
    (appdata / "Code" / "User").mkdir(parents=True)
    (appdata / "Code" / "User" / "mcp.json").write_text('{ // komentář\n "servers": {} }', encoding="utf-8")
    spec = ms.server_spec(python=r"C:\FEA\.venv\Scripts\python.exe")

    rows = {r[0]: r[2] for r in ms.apply(env=env, spec=spec, use_cli=False)}
    assert rows["Claude Desktop"] == "added" and rows["Cursor"] == "added" and rows["Codex CLI"] == "added"
    assert rows["VS Code (Copilot)"].startswith("ERROR")         # JSON with comments is not overwritten
    assert "Windsurf" not in rows                                # a client that is not installed is not created

    cd = json.loads((appdata / "Claude" / "claude_desktop_config.json").read_text(encoding="utf-8"))
    assert cd["mcpServers"]["jiny"] == {"command": "x"} and cd["jineNastaveni"] == 1
    assert cd["mcpServers"]["fea-studio"]["args"] == ["-P", "-m", "fea_core.mcp"]
    assert (appdata / "Claude" / "claude_desktop_config.json.fea-zaloha").exists()
    cur = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert cur["mcpServers"]["fea-studio"]["env"]["PYTHONIOENCODING"] == "utf-8"
    toml = (home / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert 'model = "o3"' in toml and "[mcp_servers.jiny]" in toml and "[mcp_servers.fea-studio]" in toml
    assert r"command = 'C:\FEA\.venv\Scripts\python.exe'" in toml
    import tomllib
    t = tomllib.loads(toml)
    assert t["mcp_servers"]["fea-studio"]["env"]["PYTHONPATH"] == spec["env"]["PYTHONPATH"]
    assert (appdata / "Code" / "User" / "mcp.json").read_text(encoding="utf-8").startswith("{ // komentář")

    again = {r[0]: r[2] for r in ms.apply(env=env, spec=spec, use_cli=False)}
    assert again["Claude Desktop"] == again["Cursor"] == again["Codex CLI"] == "no changes"

    gone = {r[0]: r[2] for r in ms.apply(env=env, spec=spec, remove=True, use_cli=False)}
    assert gone["Claude Desktop"] == gone["Codex CLI"] == "removed"
    toml = (home / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "fea-studio" not in toml and "[mcp_servers.jiny]" in toml
    cd = json.loads((appdata / "Claude" / "claude_desktop_config.json").read_text(encoding="utf-8"))
    assert "fea-studio" not in cd["mcpServers"] and "jiny" in cd["mcpServers"]


@WINDOWS_LAYOUT
def test_vscode_format_and_dry_run(tmp_path):
    env, home = _env(tmp_path)
    user = home / "AppData" / "Roaming" / "Code" / "User"
    user.mkdir(parents=True)
    spec = ms.server_spec(python="py")
    rows = ms.apply(env=env, spec=spec, dry=True, use_cli=False)
    assert rows and rows[0][2].endswith("(dry-run)") and not (user / "mcp.json").exists()
    ms.apply(env=env, spec=spec, use_cli=False)
    data = json.loads((user / "mcp.json").read_text(encoding="utf-8"))
    assert data["servers"]["fea-studio"]["type"] == "stdio"


def test_spec_starts_from_any_folder(tmp_path):
    """The client entry works from any working folder (PYTHONPATH = the root of the program)."""
    pytest.importorskip("mcp")
    spec = ms.server_spec()
    env = dict(os.environ, **spec["env"])
    env.pop("PYTHONHOME", None)
    r = subprocess.run([spec["command"], "-c", "import fea_core.mcp.server as s; print(s.__name__)"], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "fea_core.mcp.server" in r.stdout, r.stderr
    assert spec["command"] == sys.executable or spec["command"].lower().endswith("python.exe")


@WINDOWS_LAYOUT
def test_vscode_trailing_commas_and_comments(tmp_path):
    """VS Code reads mcp.json as JSONC: trailing commas do not matter (the file is written, the keys stay), comments do
    (the file is not overwritten – they would be lost).
    """
    env, home = _env(tmp_path)
    user = home / "AppData" / "Roaming" / "Code" / "User"
    user.mkdir(parents=True)
    cfg = user / "mcp.json"
    cfg.write_text('{\n  "servers": {\n    "jiny": {"command": "x, y", "args": ["a,]"],},\n  },\n  "inputs": [],\n}\n',
                   encoding="utf-8")
    spec = ms.server_spec(python="py")
    rows = {r[0]: r[2] for r in ms.apply(env=env, spec=spec, use_cli=False, only=["vscode"])}
    assert rows["VS Code (Copilot)"] == "added"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["servers"]["jiny"] == {"command": "x, y", "args": ["a,]"]} and data["inputs"] == []
    assert data["servers"]["fea-studio"]["type"] == "stdio"
    cfg.write_text('{ "servers": { "jiny": {"command": "x"}, }, // komentář\n}', encoding="utf-8")
    rows = {r[0]: r[2] for r in ms.apply(env=env, spec=spec, use_cli=False, only=["vscode"])}
    assert rows["VS Code (Copilot)"].startswith("ERROR") and "komentář" in cfg.read_text(encoding="utf-8")


class _FakeClaude:
    """An imitation of the CLI `claude mcp` (add / get / remove) including the variadic option -e/--env: the server name right after the
    -e values would be taken by the real CLI as another variable and rejected ("Invalid environment variable format").
    """

    def __init__(self):
        self.calls, self.entry = [], None

    def __call__(self, args, **kw):
        a = list(args[1:])
        self.calls.append(a)
        if a[:2] == ["mcp", "get"]:
            if self.entry is None:
                return subprocess.CompletedProcess(args, 1, "", f'No MCP server named "{a[2]}"')
            e = self.entry
            text = (f"{a[2]}:\n  Scope: User config (available in all your projects)\n  Type: stdio\n"
                    f"  Command: {e['command']}\n  Args: {' '.join(e['args'])}\n  Environment:\n"
                    + "".join(f"    {k}={v}\n" for k, v in e["env"].items()))
            return subprocess.CompletedProcess(args, 0, text, "")
        if a[:2] == ["mcp", "remove"]:
            had, self.entry = self.entry, None
            return subprocess.CompletedProcess(args, 0 if had else 1, "", "")
        if a[:2] == ["mcp", "add"]:
            rest, name, env, i = a[2:], None, {}, 0
            while i < len(rest) and rest[i] != "--":
                if rest[i] in ("--scope", "-s"):
                    i += 2
                    continue
                if rest[i] in ("-e", "--env"):
                    i += 1
                    while i < len(rest) and rest[i] != "--" and not rest[i].startswith("-"):
                        if "=" not in rest[i]:
                            return subprocess.CompletedProcess(args, 1, "",
                                                               f"Invalid environment variable format: {rest[i]}")
                        k, v = rest[i].split("=", 1)
                        env[k] = v
                        i += 1
                    continue
                name = rest[i]
                i += 1
            if name is None or "--" not in rest:
                return subprocess.CompletedProcess(args, 1, "", "missing name")
            if self.entry is not None:
                return subprocess.CompletedProcess(args, 1, "", f"MCP server {name} already exists in user config")
            cmd = rest[rest.index("--") + 1:]
            self.entry = {"command": cmd[0], "args": cmd[1:], "env": env}
            return subprocess.CompletedProcess(args, 0, f"Added stdio MCP server {name}", "")
        return subprocess.CompletedProcess(args, 1, "", "?")


def test_claude_cli_order_and_no_needless_remove(monkeypatch):
    """`claude mcp add --scope user fea-studio -e … -- …`: the name BEFORE -e (previously after -e → the CLI rejected it and the earlier
    `remove` deleted the working entry); an identical entry is not deleted, a different one is replaced, removal only of an existing one.
    """
    fake = _FakeClaude()
    monkeypatch.setattr(ms.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(ms.subprocess, "run", fake)
    spec = ms.server_spec(python=r"C:\FEA\.venv\Scripts\python.exe")
    assert ms._claude_cli(spec, remove=False, dry=False).startswith("added")
    add = next(c for c in fake.calls if c[:2] == ["mcp", "add"])
    assert add.index(ms.NAME) < add.index("-e") and add[add.index("--") + 1] == spec["command"]
    assert fake.entry["env"] == spec["env"]
    fake.calls.clear()
    assert ms._claude_cli(spec, remove=False, dry=False) == "no changes"
    assert not any(c[:2] in (["mcp", "remove"], ["mcp", "add"]) for c in fake.calls)
    other = dict(spec, env=dict(spec["env"], PYTHONPATH=r"D:\jinde"))
    assert ms._claude_cli(other, remove=False, dry=False).startswith("updated")
    assert fake.entry["env"]["PYTHONPATH"] == r"D:\jinde"
    dry = ms._claude_cli(spec, remove=False, dry=True)
    assert "dry-run" in dry and dry.index("fea-studio") < dry.index("-e")
    assert ms._claude_cli(spec, remove=True, dry=False).startswith("removed") and fake.entry is None
    assert ms._claude_cli(spec, remove=True, dry=False) == "not written"


# ------------------------------------------------------------------ the uninstaller: only the records of its own installation
@WINDOWS_LAYOUT
def test_remove_owned_by_keeps_the_record_of_another_installation(tmp_path):
    env, home = _env(tmp_path)
    appdata = home / "AppData" / "Roaming"
    (appdata / "Claude").mkdir(parents=True)
    (home / ".cursor").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    mine, other = tmp_path / "FEA mine", tmp_path / "FEA mine other"          # a prefix of the name is not "inside"
    spec_mine = ms.server_spec(root=mine, python=str(mine / ".venv" / "Scripts" / "python.exe"))
    spec_other = ms.server_spec(root=other, python=str(other / ".venv" / "Scripts" / "python.exe"))
    ms.apply(env=env, spec=spec_mine, use_cli=False)
    ms.apply(only=["cursor"], env=env, spec=spec_other, use_cli=False)         # Cursor belongs to the other installation
    rows = {r[0]: r[2] for r in ms.apply(env=env, spec=spec_mine, remove=True, use_cli=False, owner=mine)}
    assert rows["Claude Desktop"] == "removed" and rows["Codex CLI"] == "removed" and rows["Cursor"] == ms.KEPT
    assert "fea-studio" in (home / ".cursor" / "mcp.json").read_text(encoding="utf-8")
    assert "fea-studio" not in (home / ".codex" / "config.toml").read_text(encoding="utf-8")
    again = {r[0]: r[2] for r in ms.apply(env=env, spec=spec_mine, remove=True, use_cli=False, owner=mine)}
    assert again["Claude Desktop"] == "not written" and again["Codex CLI"] == "not written"   # safe when nothing is registered


def test_claude_cli_remove_respects_the_owner(monkeypatch, tmp_path):
    ran = []
    record = {"command": str(tmp_path / "other" / ".venv" / "python"), "args": ["-m", "fea_core.mcp"], "env": {}}
    monkeypatch.setattr(ms.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(ms, "_claude_get", lambda exe: record)
    monkeypatch.setattr(ms, "_run_claude", lambda args: ran.append(args) or subprocess.CompletedProcess(args, 0, "", ""))
    assert ms._claude_cli(ms.server_spec(), True, False, tmp_path / "mine") == ms.KEPT and not ran
    assert ms._claude_cli(ms.server_spec(), True, False, tmp_path / "other").startswith("removed") and ran


def test_the_uninstaller_removes_the_registration_of_its_installation(tmp_path, monkeypatch):
    """fea_setup uninstall calls `python -m fea_core.mcp.setup --remove --owned-by <program folder>` before .venv is deleted."""
    import importlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "installer"))
    fs = importlib.import_module("fea_setup")
    root = tmp_path / "app"
    (root / "fea_core" / "mcp").mkdir(parents=True)
    (root / "fea_core" / "mcp" / "setup.py").write_text("", encoding="utf-8")
    calls = []
    monkeypatch.setattr(fs, "ROOT", root)
    monkeypatch.setattr(fs, "VENV", root / ".venv")
    monkeypatch.setattr(fs, "TOOLS", root / "tools")
    monkeypatch.setattr(fs, "VPY", root / ".venv" / "python")
    monkeypatch.setattr(fs, "PROBLEMS", [])
    for name in ("remove_shortcuts", "remove_fusion_addin"):
        monkeypatch.setattr(fs, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(fs, "run", lambda cmd, **kw: calls.append([str(c) for c in cmd]) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(fs.slim, "rmtree", lambda d: calls.append("rmtree"))
    assert fs.cmd_uninstall(type("A", (), {"all": True, "pycache": False})()) == 0
    cmd = next(c for c in calls if isinstance(c, list))
    assert cmd[1:] == ["-m", "fea_core.mcp.setup", "--remove", "--owned-by", str(root)]
    assert cmd[0] == sys.executable                                    # no .venv python: the interpreter of the installer
    assert calls.index(cmd) < calls.index("rmtree")
    assert not fs.PROBLEMS
    monkeypatch.setattr(fs, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    fs.unregister_mcp()
    assert fs.PROBLEMS and "fea-studio" in fs.PROBLEMS[-1]

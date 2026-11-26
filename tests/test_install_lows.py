"""Smaller findings of the review of the 1.5.1 installers: folder checks of the zip installation, the installer's language, the launchers, an
installation without the GUI, the Linux uninstall, the check calculation of the installer, the README of the package."""
from __future__ import annotations

import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

fs = importlib.import_module("fea_setup")
setup_lang = importlib.import_module("setup_lang")
mp = importlib.import_module("make_package")


# ------------------------------------------------------------------ the folder checks of setup.exe for the zip installation (L9)
def test_network_and_too_long_folders_are_recognised():
    assert fs.location_problem("\\\\server\\share\\FEA Studio", True)[0] == "network"
    long = "C:\\" + "x" * 120
    assert fs.location_problem(long, False)[0] == "long" and fs.location_problem(long, True) == ("", "")
    assert fs.location_problem("C:\\" + "x" * (fs.MAX_ROOT_LENGTH - 3), False) == ("", "")
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    assert f"Length(WizardDirValue) > {fs.MAX_ROOT_LENGTH}" in iss                          # the same limit as setup.exe


@pytest.mark.parametrize("root, installed, force, stops", [
    ("\\\\srv\\x", False, False, True), ("\\\\srv\\x", True, False, True), ("\\\\srv\\x", True, True, False),
    ("C:\\" + "y" * 150, False, False, True), ("C:\\" + "y" * 150, True, False, False), ("C:\\" + "y" * 150, False, True, False),
    ("C:\\FEA", False, False, False),
])
def test_a_new_installation_into_a_bad_folder_is_refused(monkeypatch, root, installed, force, stops):
    warned = []
    monkeypatch.setattr(fs, "IS_WIN", True)
    monkeypatch.setattr(fs, "ROOT", root)
    monkeypatch.setattr(fs, "long_paths_enabled", lambda: False)
    monkeypatch.setattr(fs, "warn", warned.append)
    if stops:
        with pytest.raises(SystemExit) as e:
            fs.check_location(installed, force)
        assert e.value.code == 2
    else:
        fs.check_location(installed, force)
        assert bool(warned) == (root != "C:\\FEA")
    body = (ROOT / "installer" / "fea_setup.py").read_text(encoding="utf-8")
    body = body[body.index("def cmd_install("):body.index("def check_install(")]
    assert body.index("check_location(") < body.index("install_solver_first(keys)")           # before anything is written


# ------------------------------------------------------------------ the language of the installer (L6)
def test_the_lang_switch_decides_the_language_of_the_installer(monkeypatch):
    assert setup_lang.argv_language(["update", "--lang", "en", "--quiet"]) == "en"
    assert setup_lang.argv_language(["install", "--lang=cs"]) == "cs"
    assert setup_lang.argv_language(["install", "--lang"]) == setup_lang.argv_language(["--language", "en"]) == ""
    for argv, env, want in ((["--lang", "en"], "cs_CZ.UTF-8", False), (["--lang=cs"], "en_US.UTF-8", True)):
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import setup_lang; print(setup_lang.CS)",
                            str(ROOT / "installer"), *argv], env={**__import__("os").environ, "FEA_LANG": env}, capture_output=True, text=True,
                           check=True)
        assert r.stdout.strip() == str(want), (argv, env)


def test_the_check_calculation_prints_its_own_lines_and_the_log_only_on_failure(monkeypatch, capsys):
    from fea_core import warmup
    from verification import run_all
    monkeypatch.setattr(run_all, "run", lambda keys: print("=== V1: Konzola …\nVÝSLEDEK: …") or True)
    assert warmup.quiet_run(["V1"])
    out = capsys.readouterr().out
    assert "All checks of V1 passed." in out and "Konzola" not in out
    monkeypatch.setattr(run_all, "run", lambda keys: print("the detail of the failed check") or False)
    assert not warmup.quiet_run(["V1"])
    assert "the detail of the failed check" in capsys.readouterr().out


# ------------------------------------------------------------------ launchers (PYTHONHOME / PYTHONPATH of another program)
@pytest.mark.parametrize("name", ["FEA Studio.bat", "Install FEA Studio.bat", "Register MCP.bat"])
def test_windows_launchers_clear_foreign_python_variables(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert 'set "PYTHONHOME="' in text and 'set "PYTHONPATH="' in text


@pytest.mark.parametrize("name", ["fea-studio", "install.sh"])
def test_linux_launchers_clear_foreign_python_variables(name):
    data = (ROOT / name).read_bytes()
    assert b"\nunset PYTHONHOME PYTHONPATH" in data and b"\r" not in data


# ------------------------------------------------------------------ an installation without the GUI (L8)
def test_starting_the_gui_without_the_gui_module_says_what_to_do():
    code = ("import importlib.util, runpy, sys\n"
            "real = importlib.util.find_spec\n"
            "importlib.util.find_spec = lambda n, *a: None if n == 'PySide6' else real(n, *a)\n"
            "runpy.run_module('fea_gui', run_name='__main__')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "modules install gui" in r.stderr and "Traceback" not in r.stderr, r.stderr


# ------------------------------------------------------------------ Linux uninstall (L7)
def test_linux_uninstall_removes_the_generated_mime_file_and_rebuilds_the_databases(tmp_path, monkeypatch):
    apps = tmp_path / "share"
    files = [apps / "applications" / "fea-studio.desktop", apps / "mime" / "packages" / "fea-studio.xml",
             apps / "mime" / "application" / "x-fea-studio.xml"]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"Exec={tmp_path / 'app'}/fea-studio", encoding="utf-8")
    ran = []
    monkeypatch.setattr(fs, "IS_WIN", False)
    monkeypatch.setattr(fs, "ROOT", tmp_path / "app")
    monkeypatch.setenv("XDG_DATA_HOME", str(apps))
    monkeypatch.setattr(fs, "remove_locator", lambda: None)
    monkeypatch.setattr(fs, "ok", lambda *_: None)
    monkeypatch.setattr(fs.shutil, "which", lambda name: name)
    monkeypatch.setattr(fs, "run", lambda cmd, **k: ran.append(cmd[0]))
    fs.remove_shortcuts()
    assert not any(f.exists() for f in files)
    assert ran == ["update-mime-database", "update-desktop-database"]


# ------------------------------------------------------------------ the content of the package (L4, internal notes)
def test_the_readme_links_only_files_that_the_package_has():
    for name in ("README.md", "README_cs.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(((?!https?:|#|mailto:)[^)#]+)\)", text):
            rel = urllib.parse.unquote(target.rstrip("/"))
            assert (ROOT / rel).exists(), (name, target)
            if (ROOT / rel).is_file():
                assert mp.packaged(rel), (name, target)
        n = len({re.match(r"example(\d+)_", p.name).group(1) for p in (ROOT / "examples").glob("example*.fea")})
        assert f"1–{n}" in text, name


def test_the_internal_note_of_the_help_texts_is_not_packaged():
    assert not mp.packaged("docs/Přehled změn textů v1.3.md")


# ------------------------------------------------------------------ the .run: nothing is extracted before every option is known
def test_the_run_knows_exactly_the_switches_of_fea_setup():
    r = subprocess.run([sys.executable, str(ROOT / "installer" / "fea_setup.py"), "install", "--help"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", check=True)
    switches, valued = set(), set()
    for line in r.stdout.splitlines():
        m = re.match(r"\s+(--[a-z-]+)(\s+(\{[a-z,]+\}|[A-Z_]+))?", line)
        if m and m.group(1) != "--help":
            (valued if m.group(2) else switches).add(m.group(1))
    assert valued == {"--lang", "--profile"}
    assert switches | {"--quiet", "--yes", "-y"} == set(mp.RUN_SETUP_FLAGS)
    head = mp.run_header("9.9.9", "0" * 64)
    assert "@" + "FLAGS@" not in head and "@MODULES@" not in head and "|".join(mp.RUN_SETUP_FLAGS) in head
    retired = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))["retired"]
    assert retired and all(name in head for name in retired) and "|sketch|" in head


@pytest.fixture
def run_file(tmp_path):
    sh = shutil.which("sh")
    if not sh or sys.platform == "win32" and "system32" in sh.lower():
        pytest.skip("no POSIX shell")
    tgz = tmp_path / "p.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for name, data in (("FEAStudio-9.9.9/install.sh", b"#!/bin/sh\necho INSTALL \"$@\"\n"),):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), 0o755
            t.addfile(ti, io.BytesIO(data))
    run = mp.write_run(tgz, tmp_path / "x.run", "9.9.9")

    def go(*args, lang="en_US.UTF-8"):
        env = {**os.environ, "LANG": lang, "LC_ALL": lang, "FEA_LANG": ""}
        env.pop("LC_MESSAGES", None)
        return subprocess.run([sh, run.as_posix(), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                              cwd=tmp_path, timeout=120)
    return tmp_path, go


def _sh_path(p: Path) -> str:
    """An absolute path as the POSIX shell sees it (the sh of Git on Windows takes "C:/x" as a relative one)."""
    s = p.as_posix()
    return f"/{s[0].lower()}{s[2:]}" if sys.platform == "win32" and s[1:2] == ":" else s


@pytest.mark.parametrize("args", [["--targett", "x"], ["--profil", "standard"], ["--quiet", "--bogus"], ["--profile", "huge"], ["--lang"],
                                  ["nonsense_module"], ["--profile=big"]])
def test_the_run_refuses_an_unknown_option_before_extracting(run_file, args):
    tmp, go = run_file
    target = tmp / "app with space"
    r = go("--target", _sh_path(target), *args)
    assert r.returncode == 2 and not target.exists(), (r.stdout, r.stderr)
    assert "ERROR" in r.stderr and "CHYBA" not in r.stderr


def test_the_run_takes_its_options_anywhere_and_quotes_the_path_hint(run_file):
    tmp, go = run_file
    target = tmp / "app with space"
    r = go("--quiet", "--profile", "minimal", "sketch", "--target", _sh_path(target), "--no-shortcuts", "--extract-only")
    assert r.returncode == 0 and (target / "install.sh").is_file(), (r.stdout, r.stderr)
    assert f'sh "{_sh_path(target)}/install.sh" install' in r.stdout
    r = go("--target", _sh_path(target), "--lang", "cs", "--extract-only", "--bogus")
    assert r.returncode == 2 and "CHYBA" in r.stderr
    r = go("--target", _sh_path(tmp / "second"), "--quiet", "--lang=en", "--yes")
    assert r.returncode == 0 and "INSTALL install --quiet --lang=en --yes" in r.stdout, (r.stdout, r.stderr)

"""The solver of the package at an update (review of the 1.5.1 installers): Linux replaces an older `ccx_spooles` and its libraries (it used to keep
the binary of the previous version for ever) and never touches a `ccx_pastix` the user built; an update whose package download fails (no internet)
has the new solver in place already and removes nothing of the previous installation. Fast – synthetic archives (tests/elf_fake.py), no network.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from elf_fake import build_archive

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

fs = importlib.import_module("fea_setup")
migrations = importlib.import_module("migrations")

ARCHIVE = "ccx_2.23_linux_x86_64.tar.gz"


@pytest.fixture
def linux(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "IS_WIN", False)
    monkeypatch.setattr(fs, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    (tmp_path / "tools" / "calculix").mkdir(parents=True)
    said: list[str] = []
    monkeypatch.setattr(fs, "ok", said.append)
    monkeypatch.setattr(fs, "run", lambda *a, **k: pytest.fail("the installer must not start a process for ccx"))
    build_archive(tmp_path / "tools" / "calculix" / ARCHIVE, pastix=False)
    return tmp_path, tmp_path / "tools" / "calculix" / "ccx_2.23_linux", said


def _members(archive: Path) -> dict[str, str]:
    return fs.ccx_linux_digests(archive)


def _same_as_package(root: Path, d: Path) -> bool:
    want = _members(root / "tools" / "calculix" / ARCHIVE)
    return all((d / n).is_file() and fs.upgrade.sha256_file(d / n) == h for n, h in want.items())


def test_linux_update_replaces_the_solver_of_the_previous_version(linux):
    root, d, said = linux
    (d / "lib").mkdir(parents=True)
    (d / "ccx_spooles").write_bytes(b"OLD ccx_spooles of 1.4.1 (stand-in)\n")
    (d / "lib" / "libgomp.so.1").write_bytes(b"OLD libgomp (stand-in)\n")
    (d / "lib" / "libmine.so.9").write_bytes(b"not in the archive")
    fs.install_ext_ccx("ccx_static")
    assert _same_as_package(root, d)
    assert (d / "lib" / "libmine.so.9").read_bytes() == b"not in the archive"       # files the archive does not have stay
    assert said and "ccx_spooles" in said[-1] and "libgomp.so.1" in said[-1]       # the replacement is reported
    assert not (d.parent / ".ccx_linux_new").exists()
    assert (d / "ccx_spooles").stat().st_mode & 0o111 or sys.platform == "win32"
    said.clear()
    fs.install_ext_ccx("ccx_static")                                               # a second run: matches, nothing is written
    assert said and ("matches the package" in said[-1] or "odpovídá balíčku" in said[-1])


def test_linux_update_repairs_a_truncated_binary_and_a_missing_library(linux):
    root, d, _said = linux
    fs.install_ext_ccx("ccx_static")
    data = (d / "ccx_spooles").read_bytes()
    (d / "ccx_spooles").write_bytes(data[: len(data) // 2])
    (d / "lib" / "libquadmath.so.0").unlink()
    fs.install_ext_ccx("ccx_static")
    assert _same_as_package(root, d)


def test_a_user_built_ccx_pastix_is_never_touched(linux):
    """Also when the archive itself was built with PaStiX (--with-ccx-pastix): an existing ccx_pastix may be the user's own build."""
    root, d, said = linux
    d.mkdir(parents=True)
    (d / "ccx_pastix").write_bytes(b"the user's own build")
    (d / "ccx_spooles").write_bytes(b"old")
    fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_pastix").read_bytes() == b"the user's own build" and _same_as_package(root, d)
    assert "ccx_pastix" in said[-1]
    build_archive(root / "tools" / "calculix" / ARCHIVE, pastix=True)
    (d / "ccx_spooles").write_bytes(b"old again")
    fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_pastix").read_bytes() == b"the user's own build" and _same_as_package(root, d)


def test_only_a_user_built_pastix_gets_the_package_solver_beside_it(linux):
    root, d, _said = linux
    d.mkdir(parents=True)
    (d / "ccx_pastix").write_bytes(b"mine")
    fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_pastix").read_bytes() == b"mine" and (d / "ccx_spooles").is_file()


def test_a_damaged_archive_keeps_the_old_solver_and_says_so(linux):
    root, d, _said = linux
    d.mkdir(parents=True)
    (d / "ccx_spooles").write_bytes(b"old")
    (root / "tools" / "calculix" / ARCHIVE).write_bytes(b"not a tar file")
    with pytest.raises(RuntimeError, match="damaged|poškozený"):
        fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_spooles").read_bytes() == b"old"


def test_without_an_archive_an_existing_solver_stays(linux):
    root, d, said = linux
    (root / "tools" / "calculix" / ARCHIVE).unlink()
    d.mkdir(parents=True)
    (d / "ccx_spooles").write_bytes(b"old")
    fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_spooles").read_bytes() == b"old" and said


# ------------------------------------------------------------------ an update without internet
@pytest.mark.parametrize("win", [True, False])
def test_an_offline_update_has_the_new_solver_and_removes_nothing_old(tmp_path, monkeypatch, win):
    """The package download fails (no internet, a dead proxy): the exit is an error, the state stays the old version – but the solver of the new
    package is already unpacked (it needs no network) and the Intel CalculiX / MKL of the previous installation are still there."""
    root = tmp_path / "app"
    venv = root / ".venv"
    (root / "fea_core").mkdir(parents=True)
    (root / "fea_core" / "__init__.py").write_text('__version__ = "1.5.1"\n', encoding="utf-8")
    venv.mkdir()
    state = {"modules": ["core", "pardiso"], "version": "1.4.1"}
    (venv / "fea_modules.json").write_text(json.dumps(state), encoding="utf-8")
    old_ccx = root / "tools" / "calculix" / "calculix_2.23_4win" / "ccx_i4.exe"
    old_ccx.parent.mkdir(parents=True)
    old_ccx.write_bytes(b"ccx_i4 of 1.4.1")
    mkl = venv / "Library" / "bin" / "mkl_rt.2.dll"
    mkl.parent.mkdir(parents=True)
    mkl.write_bytes(b"M")
    di = venv / "Lib" / "site-packages" / "mkl-2023.2.0.dist-info"
    di.mkdir(parents=True)
    (di / "RECORD").write_text("../../Library/bin/mkl_rt.2.dll,,\nmkl-2023.2.0.dist-info/RECORD,,\n", encoding="utf-8")
    for name, value in (("ROOT", root), ("VENV", venv), ("VPY", venv / "Scripts" / "python.exe"), ("STATE", venv / "fea_modules.json"),
                        ("TOOLS", root / "tools"), ("DL", root / "tools" / "_download"), ("PROBLEMS", []), ("LOG", None), ("IS_WIN", win)):
        monkeypatch.setattr(fs, name, value)
    calls: list[str] = []
    for name in ("ensure_venv", "write_pth", "vc_runtime_local", "restore_pruned_qt", "launchers", "ensure_sys_libs", "write_bytecode", "pick_language",
                 "check_location"):
        monkeypatch.setattr(fs, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(fs, "EXT", {"ccx_static": lambda: calls.append("ccx")})

    def offline(keys):
        calls.append("sync")
        raise RuntimeError("download failed (offline)")
    monkeypatch.setattr(fs, "sync_packages", offline)
    args = type("A", (), {"modules": [], "profile": None, "exact": False, "force": False, "no_slim": True, "no_shortcuts": True, "no_mcp": True,
                          "no_fusion": True, "no_check": True, "keep_downloads": True, "lang": None, "no_hashes": False})()
    with pytest.raises(RuntimeError, match="offline"):
        fs.cmd_install(args, update=True)
    assert calls.index("ccx") < calls.index("sync") and calls.count("ccx") == 1
    assert old_ccx.read_bytes() == b"ccx_i4 of 1.4.1" and mkl.is_file() and (di / "RECORD").is_file()
    assert json.loads((venv / "fea_modules.json").read_text(encoding="utf-8")) == state


def test_the_solver_is_unpacked_once_in_a_successful_installation():
    src = (ROOT / "installer" / "fea_setup.py").read_text(encoding="utf-8")
    body = src[src.index("def cmd_install("):src.index("def check_install(")]
    assert body.index("install_solver_first(keys)") < body.index("upgrade_files(old_ver") < body.index("sync_packages(keys)")
    assert "e == SOLVER_EXT and solver_done" in body
    assert not any(m.fn is migrations.remove_intel_components for m in migrations.pending("1.4.1", "1.5.1", "pre"))

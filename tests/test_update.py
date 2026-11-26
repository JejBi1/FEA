"""Update of the installation: version, manifest of the program files, migrations, deleting obsolete files (synthetic trees,
without network, uv and CalculiX).
"""
from __future__ import annotations

import gzip
import importlib
import io
import json
import os
import re
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

upgrade = importlib.import_module("upgrade")
migrations = importlib.import_module("migrations")
REAL_MIGRATIONS = list(migrations.MIGRATIONS)          # the steps of the installer (the `fs` fixture replaces them by test steps)


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ verze a manifest
def test_parse_version_and_program_version(tmp_path):
    assert upgrade.parse_version("1.0.1") == (1, 0, 1)
    assert upgrade.parse_version("1.2") == (1, 2, 0)
    assert upgrade.parse_version("1.1.0rc1") == (1, 1, 0)
    assert upgrade.parse_version("1.0.10") > upgrade.parse_version("1.0.9")
    _tree(tmp_path, {"fea_core/__init__.py": '"""x"""\n__version__ = "2.3.4"\n'})
    assert upgrade.program_version(tmp_path) == "2.3.4"
    assert upgrade.program_version(ROOT).count(".") == 2


def test_manifest_of_dir_skips_tools_cache_and_itself(tmp_path):
    _tree(tmp_path, {"fea_core/a.py": "a", "fea_core/__pycache__/a.cpython-311.pyc": "b", "tools/uv/uv.exe": "c",
                     "installer/manifest.json": "{}", "README.md": "r"})
    m = upgrade.manifest_of_dir(tmp_path, "1.2.3")
    assert m["version"] == "1.2.3"
    assert set(m["files"]) == {"fea_core/a.py", "README.md"}
    assert m["files"]["README.md"] == upgrade.bytes_hash(b"r")
    upgrade.write_manifest(tmp_path / "m.json", m)
    assert upgrade.read_manifest(tmp_path / "m.json") == m
    assert upgrade.read_manifest(tmp_path / "chybi.json") is None


def test_history_manifests_are_relative_and_anonymous():
    """The bundled manifests of older releases: only relative paths and digests (nothing personal, no absolute paths)."""
    hist = upgrade.history_manifests(ROOT)
    assert {"0.1.0", "1.0.0"} <= set(hist)
    for ver, m in hist.items():
        for rel, digest in m["files"].items():
            assert not rel.startswith(("/", "\\")) and ":" not in rel and ".." not in rel.split("/"), (ver, rel)
            assert re.fullmatch(r"[0-9a-f]{16}", digest), (ver, rel)
    assert "fea_core/contacts.py" in hist["0.1.0"]["files"]          # replaced by the package contacts/ in version 1.0.0


# ------------------------------------------------------------------ deleting obsolete files
def test_remove_obsolete_keeps_user_data_and_modified_files(tmp_path):
    root = tmp_path / "app"
    files = {"fea_core/stary.py": "old", "fea_core/upraveny.py": "old2", "fea_core/novy.py": "n",
             "stara_slozka/x.txt": "x", "examples/ukazka1.fea": "fea", "moje.fea": "uziv", ".venv/lib/a.py": "v",
             "tools/uv/uv.exe": "u", "scratch/pokus.py": "s", "dist/a.zip": "z"}
    _tree(root, files)
    (root / "fea_core/__pycache__").mkdir()
    (root / "fea_core/__pycache__/stary.cpython-311.pyc").write_bytes(b"pyc")
    old = {"version": "1.0.0", "files": {k: upgrade.bytes_hash(v.encode()) for k, v in files.items()}}
    old["files"]["../mimo.txt"] = upgrade.bytes_hash(b"m")
    (tmp_path / "mimo.txt").write_text("m", encoding="utf-8")
    (root / "fea_core/upraveny.py").write_text("upraveno uživatelem", encoding="utf-8")
    new = {"version": "1.0.1", "files": {"fea_core/novy.py": upgrade.bytes_hash(b"n")}}
    obs = upgrade.obsolete_files(old, new)
    assert "fea_core/novy.py" not in obs
    removed, kept = upgrade.remove_obsolete(root, obs, say=lambda *_: None)
    assert sorted(removed) == ["fea_core/stary.py", "stara_slozka/x.txt"]
    assert kept == ["fea_core/upraveny.py"]
    assert not (root / "fea_core/__pycache__/stary.cpython-311.pyc").exists()
    assert not (root / "stara_slozka").exists()                           # empty folder after the deletion
    for rel in ("fea_core/upraveny.py", "fea_core/novy.py", "examples/ukazka1.fea", "moje.fea", ".venv/lib/a.py",
                "tools/uv/uv.exe", "scratch/pokus.py", "dist/a.zip"):
        assert (root / rel).exists(), rel
    assert (tmp_path / "mimo.txt").exists()                               # never outside the program folder


def test_is_protected():
    for rel in (".venv/x", "tools/a", "scratch/b", "dist/c", ".git/HEAD", "a/b.fea", "../x", "/abs", "C:/x", ""):
        assert upgrade.is_protected(rel), rel
    for rel in ("fea_core/a.py", "docs/x.pdf", "examples/geometry.py"):
        assert not upgrade.is_protected(rel), rel


def test_orphan_bytecode(tmp_path):
    _tree(tmp_path, {"fea_core/a.py": "a"})
    c = tmp_path / "fea_core" / "__pycache__"
    c.mkdir()
    (c / "a.cpython-311.pyc").write_bytes(b"1")
    (c / "smazany.cpython-311.pyc").write_bytes(b"2")
    assert [p.name for p in upgrade.orphan_bytecode(tmp_path)] == ["smazany.cpython-311.pyc"]


# ------------------------------------------------------------------ migrace
def test_pending_range_and_order():
    calls = []
    ms = [migrations.Migration("1.0.3", "post", "c", lambda c: calls.append("c")),
          migrations.Migration("1.0.1", "post", "a2", lambda c: calls.append("a2")),
          migrations.Migration("1.0.1", "pre", "a1", lambda c: calls.append("a1")),
          migrations.Migration("1.0.2", "pre", "b", lambda c: (_ for _ in ()).throw(RuntimeError("chyba")))]
    assert [m.title for m in migrations.pending("1.0.0", "1.0.3", migrations=ms)] == ["a1", "a2", "b", "c"]
    assert [m.title for m in migrations.pending("1.0.1", "1.0.2", migrations=ms)] == ["b"]
    assert migrations.pending("1.0.3", "1.0.3", migrations=ms) == []
    warned = []
    ctx = migrations.Context(root=Path("."), venv=Path("."), vpy=Path("."), modules=["core"], say=lambda *_: None,
                             warn=warned.append)
    done = migrations.run(ctx, "1.0.0", "1.0.3", "pre", migrations=ms)
    assert calls == ["a1"] and done == ["1.0.1: a1"] and len(warned) == 1          # an error of a step does not stop the next ones
    migrations.run(ctx, "1.0.0", "1.0.3", "post", migrations=ms)
    assert calls == ["a1", "a2", "c"]


def test_builtin_migrations_cover_101():
    assert {m.phase for m in migrations.pending("1.0.0", "1.0.1")} == {"pre", "post"}


def test_newest_migration_is_not_newer_than_the_program_version():
    """A migration of a version that the program does not have yet never runs (the update of a package of this version is judged by
    `fea_core.__version__`: `pending(old, new)` takes the steps with old < step <= new) – the version has to be raised together with the step."""
    program = upgrade.parse_version(upgrade.program_version(ROOT))
    newest = max(migrations.MIGRATIONS, key=lambda m: upgrade.parse_version(m.version))
    assert upgrade.parse_version(newest.version) <= program, (
        f"the migration {newest.version} ({newest.title}) is newer than the program version {upgrade.program_version(ROOT)} – "
        "a package of this version would update an older installation without running it")
    # and the step is really reached by an update from the previous release to the program version
    assert newest in migrations.pending("1.4.1", upgrade.program_version(ROOT)) or upgrade.parse_version(newest.version) <= (1, 4, 1)


def test_migration_15_moves_the_direct_solver_to_its_own_module(tmp_path):
    """1.4.2: the module ngsolve no longer carries rslab. An installation that had ngsolve gets the module direct (and accel for numba); ngsolve itself
    stays only when the module motion is installed (flexible bodies), otherwise it is replaced – the closure would add Exudyn for it. One without
    ngsolve is left alone, the step is idempotent and runs only for an update from an older version."""
    assert [m.title for m in migrations.pending("1.4.1", "1.4.2", "pre")
            if "direct" in m.title] and not migrations.pending("1.4.2", "1.4.2", "pre")
    # the installed list (the state file) → the planned list
    mods, notes = migrations.plan_modules(["core", "gui", "ngsolve", "hypre"], "1.4.1", "1.4.2")
    assert mods == ["core", "gui", "hypre", "accel", "direct"] and len(notes) == 1 and "ngsolve" in notes[0]
    mods, notes = migrations.plan_modules(["core", "gui", "ngsolve", "motion"], "1.4.1", "1.4.2")
    assert mods == ["core", "gui", "ngsolve", "motion", "accel", "direct"] and len(notes) == 1 and "170" in notes[0]
    assert migrations.plan_modules(["core", "gui", "accel"], "1.4.1", "1.4.2") == (["core", "gui", "accel"], [])
    assert migrations.plan_modules(["core", "gui", "ngsolve"], "1.4.2", "1.4.2")[0] == ["core", "gui", "ngsolve"]     # not an update of the version
    for installed in (["core", "ngsolve"], ["core", "ngsolve", "motion"]):                                               # idempotent
        once, _ = migrations.plan_modules(installed, "1.4.1", "1.4.2")
        assert migrations.plan_modules(once, "1.4.1", "1.4.2") == (once, [])
    # the safety net on the module list of the update (for a caller that did not plan): same rule, the closure's motion keeps ngsolve there
    base = dict(root=tmp_path, venv=tmp_path / ".venv", vpy=tmp_path / "nic", say=lambda *_: None)
    ctx = migrations.Context(modules=["core", "gui", "ngsolve", "motion", "hypre"], **base)
    for _ in range(2):
        migrations.add_direct_solver_module(ctx)
        assert ctx.modules == ["core", "gui", "ngsolve", "motion", "hypre", "accel", "direct"]
    other = migrations.Context(modules=["core", "gui", "accel"], **base)
    migrations.add_direct_solver_module(other)
    assert other.modules == ["core", "gui", "accel"]


def test_update_from_141_shows_and_installs_the_final_module_list(fs, monkeypatch, capsys):
    """An update of an installation [core, plots, gui, ngsolve] (the old solver module, no Exudyn): the list printed BEFORE the work, the packages
    synchronised and the state written are the same final list – direct + accel instead of ngsolve, no motion (Exudyn) the user never had."""
    monkeypatch.setattr(migrations, "MIGRATIONS", REAL_MIGRATIONS)           # the fixture replaced them by two test steps
    _tree(fs.root, {"fea_core/__init__.py": '__version__ = "1.4.2"\n'})
    upgrade.write_manifest(fs.root / upgrade.PACKAGE_MANIFEST, {"version": "1.4.2", "files": {}})
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "plots", "gui", "ngsolve"], "version": "1.4.1"}), encoding="utf-8")
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    final = ["core", "plots", "gui", "accel", "direct"]
    assert f"sync:{' '.join(final)}" in fs.calls
    st = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))
    assert st["version"] == "1.4.2" and st["modules"] == final
    listed = [ln for ln in out.splitlines() if ln.startswith(("Modules:", "Moduly:"))]
    assert len(listed) == 1 and "ngsolve" not in listed[0] and "motion" not in listed[0] and "direct" in listed[0], out
    assert out.index(listed[0]) < out.index(fs.mod.T("migrace 1.4.2", "migration 1.4.2"))      # shown before the migration runs
    # the same installation with Exudyn keeps ngsolve (flexible bodies) and gains direct
    fs.calls.clear()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "plots", "gui", "motion", "ngsolve"], "version": "1.4.1"}), encoding="utf-8")
    assert fs.mod.cmd_update(_args()) == 0
    st = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))
    assert set(st["modules"]) == {"core", "plots", "gui", "motion", "ngsolve", "accel", "direct"}
    # an explicit selection (--exact) is not rewritten by the step
    fs.calls.clear()
    assert fs.mod.cmd_install(_args(modules=["gui", "ngsolve"], exact=True)) == 0
    st = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))
    assert "ngsolve" in st["modules"] and "motion" in st["modules"] and "direct" not in st["modules"]


def test_remove_old_release_files_is_idempotent(tmp_path):
    """Step 1.0.1: files of the releases 0.1.0 / 1.0.0 without a saved manifest (according to installer/manifests) – only the unchanged ones."""
    root = tmp_path
    _tree(root, {"fea_core/contacts.py": "stary modul", "tests/gui_smoke.py": "upraveno", "fea_core/novy.py": "n"})
    (root / "installer" / "manifests").mkdir(parents=True)
    upgrade.write_manifest(root / "installer/manifests/0.1.0.json", {"version": "0.1.0", "files": {
        "fea_core/contacts.py": upgrade.bytes_hash(b"stary modul"), "tests/gui_smoke.py": upgrade.bytes_hash(b"puvodni"),
        "fea_core/novy.py": upgrade.bytes_hash(b"n")}})
    upgrade.write_manifest(root / upgrade.PACKAGE_MANIFEST, {"version": "1.0.1", "files": {
        "fea_core/novy.py": upgrade.bytes_hash(b"n")}})
    ctx = migrations.Context(root=root, venv=root / ".venv", vpy=root / "nic", modules=["core"], say=lambda *_: None)
    for _ in range(2):
        migrations.remove_old_release_files(ctx)
        assert not (root / "fea_core/contacts.py").exists()
        assert (root / "tests/gui_smoke.py").exists() and (root / "fea_core/novy.py").exists()


def test_broken_packages(tmp_path):
    site = tmp_path
    _tree(site, {"PySide6_Essentials-6.8.0.dist-info/RECORD": "PySide6/__init__.py,sha256=x,1\nPySide6/QtCore.pyd,,\n",
                 "numpy-2.1.0.dist-info/RECORD": "numpy/__init__.py,,\n", "numpy/__init__.py": "",
                 "vtk-9.3.0.dist-info/RECORD": "vtkmodules/__init__.py,,\n"})
    assert migrations.broken_packages(site) == ["PySide6_Essentials==6.8.0"]      # vtkmodules is deleted by slim on purpose
    _tree(site, {"PySide6/__init__.py": ""})
    assert migrations.broken_packages(site) == []


# ------------------------------------------------------------------ fea_setup: posloupnost aktualizace
@pytest.fixture
def fs(tmp_path, monkeypatch):
    """fea_setup over a synthetic installation in tmp_path (all heavy steps replaced by a record of the calls)."""
    fs = importlib.import_module("fea_setup")
    root = tmp_path / "app"
    venv = root / ".venv"
    _tree(root, {"fea_core/__init__.py": '__version__ = "1.0.1"\n', "fea_core/novy.py": "n", "fea_core/stary.py": "s",
                 "moje_studie.fea": "uziv"})
    monkeypatch.setattr(fs, "ROOT", root)
    monkeypatch.setattr(fs, "VENV", venv)
    monkeypatch.setattr(fs, "VPY", venv / "Scripts" / "python.exe")
    monkeypatch.setattr(fs, "STATE", venv / "fea_modules.json")
    monkeypatch.setattr(fs, "TOOLS", root / "tools")                      # uninstall --all deletes TOOLS – never the real ones
    monkeypatch.setattr(fs, "DL", root / "tools" / "_download")
    monkeypatch.setattr(fs, "PROBLEMS", [])
    monkeypatch.setattr(fs, "LOG", None)
    calls: list[str] = []
    for name in ("ensure_venv", "write_pth", "vc_runtime_local", "launchers", "ensure_sys_libs", "write_bytecode",
                 "check_install", "make_shortcuts", "install_fusion_addin", "check_location"):
        monkeypatch.setattr(fs, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(fs, "solver_available", lambda: True)             # the synthetic installation has no CalculiX (own tests below)
    monkeypatch.setattr(fs, "sync_packages", lambda keys: calls.append("sync:" + " ".join(keys)))
    monkeypatch.setattr(fs, "remove_ext", lambda k: calls.append("remove_ext:" + k))
    monkeypatch.setattr(fs, "fusion_dir", lambda: None)
    monkeypatch.setattr(fs, "site_packages", lambda: venv / "Lib" / "site-packages")
    monkeypatch.setattr(fs.slim, "slim_all", lambda *a, **k: calls.append("slim"))
    monkeypatch.setattr(fs, "EXT", {})
    ran: list[str] = []
    ms = [migrations.Migration("1.0.1", "pre", "pre", lambda c: ran.append("pre:" + ",".join(c.modules))),
          migrations.Migration("1.0.1", "post", "post", lambda c: ran.append("post"))]
    monkeypatch.setattr(migrations, "MIGRATIONS", ms)
    upgrade.write_manifest(root / upgrade.PACKAGE_MANIFEST, {"version": "1.0.1", "files": {
        "fea_core/__init__.py": upgrade.bytes_hash(b'__version__ = "1.0.1"\n'), "fea_core/novy.py": upgrade.bytes_hash(b"n")}})
    ns = types.SimpleNamespace(root=root, venv=venv, calls=calls, ran=ran, mod=fs)
    return ns


def _args(**kw):
    a = dict(modules=[], profile=None, exact=False, force=False, no_slim=False, no_shortcuts=True, no_mcp=True,
             no_fusion=True, no_check=True, quick_check=False, keep_downloads=False)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_update_from_legacy_100(fs):
    """A state without a version (release 1.0.0) → it is taken as 1.0.0: obsolete files according to the manifest 1.0.0, migrations pre/post,
    modules as last time (without a question), the new version and the manifest saved.
    """
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui"], "platform": "x"}), encoding="utf-8")
    hist = fs.root / "installer" / "manifests"
    upgrade.write_manifest(hist / "1.0.0.json", {"version": "1.0.0", "files": {
        "fea_core/stary.py": upgrade.bytes_hash(b"s"), "fea_core/novy.py": upgrade.bytes_hash(b"n"),
        "moje_studie.fea": upgrade.bytes_hash(b"uziv")}})
    assert fs.mod.installed_version() == "1.0.0"
    assert fs.mod.cmd_update(_args()) == 0
    assert not (fs.root / "fea_core/stary.py").exists()
    assert (fs.root / "fea_core/novy.py").exists() and (fs.root / "moje_studie.fea").exists()
    assert fs.ran == ["pre:core,plots,gui", "post"]
    assert "sync:core plots gui" in fs.calls
    assert fs.calls.index("sync:core plots gui") < fs.calls.index("slim")
    st = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))
    assert st["version"] == "1.0.1" and st["modules"] == ["core", "plots", "gui"]
    saved = upgrade.read_manifest(fs.venv / upgrade.MANIFEST_NAME)
    assert saved and saved["version"] == "1.0.1"
    # a second run of the same version: no migrations, nothing is deleted
    fs.ran.clear()
    assert fs.mod.cmd_update(_args()) == 0
    assert fs.ran == [] and (fs.root / "fea_core/novy.py").exists()


def test_update_uses_stored_manifest_and_exact_removes_modules(fs):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui", "report"], "version": "1.0.0"}),
                                              encoding="utf-8")
    upgrade.write_manifest(fs.venv / upgrade.MANIFEST_NAME, {"version": "1.0.0", "files": {
        "fea_core/stary.py": upgrade.bytes_hash(b"s")}})
    assert fs.mod.cmd_install(_args(modules=["gui"], exact=True)) == 0
    assert not (fs.root / "fea_core/stary.py").exists()
    assert "sync:core plots gui" in fs.calls and "remove_ext:report" in fs.calls


def test_downgrade_needs_force(fs):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "9.0.0"}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        fs.mod.cmd_update(_args())
    assert e.value.code == 2 and fs.calls == []                         # nothing has changed
    assert fs.mod.cmd_update(_args(force=True)) == 0
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["version"] == "1.0.1"


def test_fresh_install_without_state(fs):
    assert fs.mod.installed_version() is None
    assert fs.mod.cmd_install(_args(modules=["core"])) == 0
    assert fs.ran == [] and (fs.root / "fea_core/stary.py").exists()      # without a previous version nothing is deleted
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["version"] == "1.0.1"


# ------------------------------------------------------------------ exit codes: a missing solver / a failed check are not "success"
def _state(fs):
    p = fs.venv / "fea_modules.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def test_install_without_a_solver_fails_and_records_nothing(fs, monkeypatch, capsys):
    """cmd_install level: no usable CalculiX after the external components = exit code 4, no state, no manifest, no shortcuts / check."""
    monkeypatch.setattr(fs.mod, "solver_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        fs.mod.cmd_install(_args(modules=["core"]))
    assert e.value.code == fs.mod.EXIT_NO_SOLVER and e.value.code not in (0, None)
    assert _state(fs) is None and not (fs.venv / upgrade.MANIFEST_NAME).exists()
    assert "check_install" not in fs.calls and "make_shortcuts" not in fs.calls and "write_bytecode" not in fs.calls
    assert fs.mod.installed_version() is None                           # the next run is a fresh installation again
    assert "CalculiX" in capsys.readouterr().err


def test_update_without_a_solver_keeps_the_old_state(fs, monkeypatch):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.0"}), encoding="utf-8")
    monkeypatch.setattr(fs.mod, "solver_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        fs.mod.cmd_update(_args())
    assert e.value.code == fs.mod.EXIT_NO_SOLVER
    assert _state(fs)["version"] == "1.0.0"                             # not marked as updated


def test_failed_provider_warns_and_the_solver_check_decides(fs, monkeypatch):
    """A failing CalculiX step is a warning; what makes it fatal is that no CalculiX is usable afterwards – FEA_CCX / PATH count."""
    def fail():
        raise RuntimeError("no archive")
    monkeypatch.setattr(fs.mod, "EXT", {"ccx_static": fail})
    monkeypatch.setattr(fs.mod, "solver_available", lambda: True)       # e.g. FEA_CCX points at the user's own ccx
    assert fs.mod.cmd_install(_args(modules=["core"])) == 0
    assert any("no archive" in p for p in fs.mod.PROBLEMS)
    assert _state(fs)["modules"] == ["core"]


def test_a_failed_check_calculation_is_exit_code_5_but_the_state_stays(fs, monkeypatch):
    monkeypatch.setattr(fs.mod, "check_install", lambda keys: False)
    assert fs.mod.cmd_install(_args(modules=["core"], no_check=False)) == fs.mod.EXIT_CHECK_FAILED == 5
    assert _state(fs)["version"] == "1.0.1"                             # the files ARE installed
    monkeypatch.setattr(fs.mod, "check_install", lambda keys: True)
    assert fs.mod.cmd_install(_args(modules=["core"], no_check=False)) == 0


def test_solver_available_looks_where_the_program_looks(tmp_path, monkeypatch):
    fs = importlib.import_module("fea_setup")
    monkeypatch.setattr(fs, "TOOLS", tmp_path / "tools")
    monkeypatch.delenv("FEA_CCX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))                 # no ccx in PATH
    for win, folder, name in ((True, "ccx_2.23_windows", "ccx_spooles.exe"), (False, "ccx_2.23_linux", "ccx_spooles")):
        monkeypatch.setattr(fs, "IS_WIN", win)
        assert not fs.solver_available()
        exe = tmp_path / "tools" / "calculix" / folder / name
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"x")
        assert fs.solver_available()
        exe.unlink()
    monkeypatch.setattr(fs, "IS_WIN", False)
    own = tmp_path / "my_ccx"
    own.write_bytes(b"x")
    monkeypatch.setenv("FEA_CCX", str(own))
    assert fs.solver_available()
    monkeypatch.setenv("FEA_CCX", str(tmp_path / "nowhere"))
    assert not fs.solver_available()
    monkeypatch.delenv("FEA_CCX")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ccx").write_bytes(b"x")
    (tmp_path / "bin" / "ccx").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert fs.solver_available()


# ------------------------------------------------------------------ the retired module pardiso (Intel MKL, removed in 1.4.2)
@pytest.mark.parametrize("win", [True, False])
def test_a_retired_module_named_on_the_command_line_is_skipped_with_a_note(fs, monkeypatch, capsys, win):
    monkeypatch.setattr(fs.mod, "IS_WIN", win)
    fs.mod._RETIRED_SAID.clear()                                        # the note is printed once per process
    assert fs.mod.cmd_install(_args(modules=["pardiso"])) == 0           # never an error: an old script keeps working
    assert _state(fs)["modules"] == ["core"]
    assert "pardiso" in capsys.readouterr().out
    assert not any("mkl" in c.lower() for c in fs.calls)


@pytest.mark.parametrize("win", [True, False])
def test_exact_selection_gets_the_replacement_of_the_retired_module_named_by_the_state(fs, monkeypatch, capsys, win):
    """setup.exe over a setup.exe installation passes --exact: plan_modules does not run there, a custom selection of 1.4.x (core + gui + pardiso) used to end
    without any fast solver and without a note. Now the state naming pardiso adds direct + accel (the retired module meant "the fast solver")."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui", "pardiso"], "version": "1.0.0"}), encoding="utf-8")
    monkeypatch.setattr(fs.mod, "IS_WIN", win)
    monkeypatch.setattr(fs.mod, "EXT", {"ccx_static": lambda: None})
    assert fs.mod.cmd_install(_args(modules=["gui"], exact=True)) == 0           # what the wizard passes: the components that are ticked
    assert fs.mod.PROBLEMS == []
    assert _state(fs)["modules"] == ["core", "plots", "gui", "accel", "direct"]
    out = capsys.readouterr().out
    assert "pardiso" in out and "core, plots, gui, accel, direct" in out
    assert any(c.startswith("sync:") and "direct" in c and "pardiso" not in c for c in fs.calls)
    assert fs.mod.cmd_install(_args(modules=["gui"], exact=True)) == 0           # a second run: the state no longer names pardiso – the selection is as chosen
    assert _state(fs)["modules"] == ["core", "plots", "gui"]


def test_exact_selection_without_the_retired_module_stays_as_chosen(fs, monkeypatch):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui", "accel"], "version": "1.0.0"}), encoding="utf-8")
    monkeypatch.setattr(fs.mod, "EXT", {"ccx_static": lambda: None})
    assert fs.mod.cmd_install(_args(modules=["gui"], exact=True)) == 0
    assert _state(fs)["modules"] == ["core", "plots", "gui"]                    # accel unticked by the user: removed, no direct added behind their back


def test_exact_selection_with_pardiso_on_the_command_line(fs, monkeypatch):
    fs.mod._RETIRED_SAID.clear()
    monkeypatch.setattr(fs.mod, "EXT", {"ccx_static": lambda: None})
    assert fs.mod.cmd_install(_args(modules=["gui", "pardiso"], exact=True)) == 0
    assert _state(fs)["modules"] == ["core", "plots", "gui", "accel", "direct"]


def test_exact_retired_modules_is_idempotent_and_silent_when_the_wizard_already_ticked_the_replacement():
    f = migrations.exact_retired_modules
    mods, note = f(["core", "gui"], ["core", "pardiso"], [])
    assert mods == ["core", "gui", "accel", "direct"] and "pardiso" in note and "direct" in note
    assert f(["core", "gui", "accel", "direct"], ["core", "pardiso"], []) == (["core", "gui", "accel", "direct"], "")      # Standard type: nothing to add, no noise
    assert f(["core", "direct"], ["core", "pardiso"], [])[0] == ["core", "direct", "accel"]                               # direct without accel: accel is added
    assert f(["core", "gui"], ["core", "gui"], []) == (["core", "gui"], "")                                                # the state does not name it: untouched
    assert f(mods, ["core", "pardiso"], []) == (mods, "")                                                                  # idempotent
    assert f(["core"], [], ["pardiso"])[0] == ["core", "accel", "direct"]                                                  # the command line names it


def test_the_wizard_preselects_the_replacement_of_the_retired_module():
    """FEAStudio.iss is not compiled here: the text test pins what the Pascal code does – the state naming pardiso preselects exactly the modules that
    migrations.drop_pardiso_module adds, also after a setup.exe installation (PreviousInno, the choice Inno restored), once (RetiredMapped)."""
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    added = set(migrations.drop_pardiso_module(["core", "pardiso"])[0]) - {"core"}
    assert added == {"accel", "direct"}
    wants = iss.split("function StateWants")[1].split("\nend;")[0]
    assert "Pos('\"pardiso\"', S) > 0" in wants
    for name in added:
        assert f"(Name = '{name}')" in wants, name
        assert name in iss.split("MODULES = '")[1].split("'")[0].split()                  # and it is a component the wizard has
    page = iss.split("procedure CurPageChanged")[1].split("\nend;")[0]
    # the components follow what is installed (the state), also after a setup.exe installation: modules added through the program are kept;
    # only a module that the old version could not have keeps the wizard's own choice
    assert "StateWants(S, Name)" in page and "WizardIsComponentSelected(Name)" in page and "NEW_MODULES" in page
    assert "PreviousInno" not in page and "StateApplied := WizardDirValue" in page
    assert "HasParamPrefix('/COMPONENTS=')" in page and "HasParamPrefix('/TYPE=')" in page
    assert "StateApplied := ''" in iss.split("function InitializeSetup")[1].split("\nend;")[0]
    new = iss.split("NEW_MODULES = '")[1].split("'")[0].split()
    assert set(new) <= set(iss.split("MODULES = '")[1].split("'")[0].split()) and set(migrations.NEW_STANDARD_151) <= set(new)
    assert "RetiredMapped" not in iss


def test_the_profiles_do_not_name_the_retired_module(fs):
    for name, mods in fs.mod.PROFILES.items():
        assert "pardiso" not in mods, name
    assert "pardiso" not in fs.mod.MODS and "pardiso" in fs.mod.RETIRED


@pytest.mark.parametrize("win", [True, False])
def test_an_installation_state_that_lists_pardiso_updates_and_forgets_it(fs, monkeypatch, win):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui", "pardiso"], "version": "1.0.0"}), encoding="utf-8")
    monkeypatch.setattr(fs.mod, "IS_WIN", win)
    monkeypatch.setattr(fs.mod, "EXT", {"ccx_static": lambda: None})
    assert fs.mod.cmd_update(_args()) == 0
    assert fs.mod.PROBLEMS == []
    assert _state(fs)["modules"] == ["core", "plots", "gui"]            # the retired module is not kept (plots = what gui requires)
    assert not any("pardiso" in c for c in fs.calls if c.startswith("sync:"))


def test_list_remove_and_slim_read_a_state_with_the_retired_module(fs, monkeypatch, capsys):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "accel", "plots", "pardiso"], "version": "1.0.0"}), encoding="utf-8")
    assert fs.mod.cmd_list(_args()) == 0
    assert "pardiso" not in capsys.readouterr().out
    assert fs.mod.cmd_slim(_args()) == 0
    assert fs.mod.cmd_remove(_args(modules=["plots"])) == 0
    assert _state(fs)["modules"] == ["core", "accel"]


def test_a_repair_of_the_same_version_finishes_the_intel_cleanup(fs):
    """The state already names the current version although the cleanup of 1.4.2 left a file behind (a DLL held by a running program): the next run of
    the same version (repair / update) finishes it – the cleanup does not depend on the version step."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.1"}), encoding="utf-8")
    dll = fs.venv / "Library" / "bin" / "mkl_rt.2.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"M")
    di = fs.venv / "Lib" / "site-packages" / "mkl-2023.2.0.dist-info"
    di.mkdir(parents=True)
    (di / "RECORD").write_text("../../Library/bin/mkl_rt.2.dll,,\nmkl-2023.2.0.dist-info/RECORD,,\n", encoding="utf-8")
    ccx = fs.root / "tools" / "calculix" / "calculix_2.23_4win" / "ccx_i4.exe"
    ccx.parent.mkdir(parents=True)
    ccx.write_text("x", encoding="utf-8")
    assert fs.mod.installed_version() == upgrade.program_version(fs.root) == "1.0.1"
    assert fs.mod.cmd_update(_args()) == 0
    assert not dll.exists() and not di.exists() and not ccx.exists()
    assert fs.ran == []                                                 # no version step ran for the same version: the cleanup alone did the work


# ------------------------------------------------------------------ bytecode of the program (best effort)
def test_write_bytecode_is_a_best_effort_step(tmp_path, monkeypatch):
    fs = importlib.import_module("fea_setup")
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    monkeypatch.setattr(fs, "PROBLEMS", [])
    monkeypatch.setattr(fs, "ok", lambda *_: None)
    monkeypatch.setattr(fs, "info", lambda *_: None)
    seen = []
    code = {"rc": 0}
    monkeypatch.setattr(fs, "run", lambda cmd, **k: seen.append([str(c) for c in cmd]) or type("R", (), {"returncode": code["rc"]})())
    monkeypatch.setattr(fs, "VPY", tmp_path / "no_python")
    fs.write_bytecode()
    assert seen == []                                                   # no environment, nothing to do
    py = tmp_path / "python"
    py.write_bytes(b"x")
    monkeypatch.setattr(fs, "VPY", py)
    fs.write_bytecode()
    assert seen[-1][1:5] == ["-m", "compileall", "-q", "-j0"] and seen[-1][5:] == [str(tmp_path / "fea_core"), str(tmp_path / "fea_gui")]
    code["rc"] = 1                                                      # a read-only folder: a note, never a warning or an error
    fs.write_bytecode()
    assert fs.PROBLEMS == []


def test_install_kind():
    fs = importlib.import_module("fea_setup")
    assert fs.install_kind(None, "1.0.1") == "install"
    assert fs.install_kind("1.0.1", "1.0.1") == "repair"
    assert fs.install_kind("1.0.0", "1.0.1") == "update"
    assert fs.install_kind("1.0.2", "1.0.1") == "downgrade"


def test_repair_install_same_version_message(fs, capsys):
    """The same version again (setup.exe over a finished installation) = "repair installation of version X", not "update X → X"."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.1"}), encoding="utf-8")
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert "1.0.1 → 1.0.1" not in out
    assert fs.mod.T("opravná instalace verze 1.0.1", "repair installation of version 1.0.1") in out
    assert fs.mod.T("Opravná instalace verze 1.0.1 dokončena.", "Repair installation of version 1.0.1 finished.") in out


# ------------------------------------------------------------------ the pre-release name 1.5.0 of the release 1.4.2
def test_state_version_reads_a_prerelease_name_as_the_release_it_became():
    """The builds of the branch next-1.5 (the test installations on Linux and Windows) were called 1.5.0 before the release got the number 1.4.2: the state
    says 1.5.0. The program reads it as 1.4.2 – only while it is older than 1.5.0 (a real 1.5.0 later is not mistaken for 1.4.2), other versions never change."""
    assert upgrade.state_version("1.5.0", "1.4.2") == "1.4.2"
    assert upgrade.state_version("1.5.0", "1.4.3") == "1.4.2"
    assert upgrade.state_version("1.5.0", "1.5.0") == "1.5.0"
    assert upgrade.state_version("1.5.0", "1.6.0") == "1.5.0"
    for v in ("1.4.1", "1.4.2", "1.5.1", "1.0.0", "9.0.0"):
        assert upgrade.state_version(v, "1.4.2") == v
    assert upgrade.parse_version(upgrade.state_version("1.5.0", "1.4.2")) == upgrade.parse_version("1.4.2")


def test_the_history_manifests_load_and_the_last_release_is_among_them():
    """installer/manifests/<version>.json = the program files of every released version (the update deletes what an old release had and the new one has not).
    The release 1.4.2 needs the one of 1.4.1 (made from its package by the functions of make_package.py: `scratch/make_manifest_141.py`-style, the PragTic data file
    left out)."""
    hist = upgrade.history_manifests(ROOT)
    assert {"1.4.0", "1.4.1"} <= set(hist)
    for version, m in hist.items():
        assert m["version"] == version and m["files"], version
        assert all(isinstance(k, str) and re.fullmatch(r"[0-9a-f]{16}", v) for k, v in m["files"].items()), version
        if upgrade.parse_version(version) >= (1, 1, 0):                                         # 0.1.0 – 1.0.1 shipped it, before the licence decision
            assert not set(upgrade.PROTECTED_FILES) & set(m["files"]), version                   # the PragTic data are not distributed
    assert upgrade.read_manifest(ROOT / "installer" / "manifests" / "1.4.1.json")["files"]["fea_core/__init__.py"]
    assert upgrade.parse_version("1.4.1") < upgrade.parse_version(upgrade.program_version(ROOT))     # older than the program it ships with


def test_the_program_is_never_version_1_5_0():
    """The tolerance above makes an installation state "1.5.0" mean 1.4.2 while the program is older than 1.5.0. A real release called 1.5.0 would be taken for
    1.4.2 by those installations (no step would run). So the number is reserved: the next feature release after 1.4.2 is 1.5.1 (installer/README.md, "Version
    numbers"). This fails the moment `fea_core.__version__` is 1.5.0 (or any other pre-release name), when the changelog gets such a heading, or when the rule is
    dropped from the README."""
    import fea_core
    names = set(upgrade.PRERELEASE_NAMES)
    assert "1.5.0" in names                                                  # the tolerance is kept (the test installations of the pre-release builds)
    assert fea_core.__version__ != "1.5.0" and fea_core.__version__ not in names, "the next release is 1.5.1, never 1.5.0 – see installer/README.md"
    assert upgrade.program_version(ROOT) == fea_core.__version__
    for name in names:
        # the program is older than every pre-release name (the mapping is inert only once the program reaches it – which must never be a release)
        assert upgrade.parse_version(fea_core.__version__) != upgrade.parse_version(name)
    headings = re.findall(r"^##\s+v?(\d+\.\d+\.\d+)", (ROOT / "installer" / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    assert headings and not names & set(headings), f"CHANGELOG.md has a heading of a reserved pre-release name: {sorted(names & set(headings))}"
    readme = (ROOT / "installer" / "README.md").read_text(encoding="utf-8")
    assert "never 1.5.0" in readme and "1.5.1" in readme


def test_the_state_of_a_prerelease_150_build_is_not_a_downgrade_and_runs_no_step_again(fs, monkeypatch, capsys):
    """Program 1.4.2 over an installation whose state was written by a 1.5.0 pre-release build: the same version – `install_kind` says repair, `pending` has no
    step (the steps of 1.4.2 ran in that build), `update` does not stop with the exit code 2 and writes the state as 1.4.2. A really newer state (1.5.1) is still a
    downgrade that needs --force."""
    (fs.root / "fea_core" / "__init__.py").write_text('__version__ = "1.4.2"\n', encoding="utf-8")
    steps = [migrations.Migration("1.4.2", "pre", "step of 1.4.2", lambda c: fs.ran.append("1.4.2"))]
    monkeypatch.setattr(migrations, "MIGRATIONS", steps)
    fs.venv.mkdir()
    state = fs.venv / "fea_modules.json"
    state.write_text(json.dumps({"modules": ["core"], "version": "1.5.0"}), encoding="utf-8")
    assert fs.mod.installed_version() == "1.4.2"
    assert fs.mod.install_kind(fs.mod.installed_version(), "1.4.2") == "repair"
    assert migrations.pending(fs.mod.installed_version(), "1.4.2") == [] and migrations.plan_modules(["core"], fs.mod.installed_version(), "1.4.2") == (["core"], [])
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert fs.mod.T("opravná instalace verze 1.4.2", "repair installation of version 1.4.2") in out and "1.5.0" not in out
    assert fs.ran == []
    assert json.loads(state.read_text(encoding="utf-8"))["version"] == "1.4.2"
    # an older release over it is refused as before, a state that is really newer needs --force
    state.write_text(json.dumps({"modules": ["core"], "version": "1.5.1"}), encoding="utf-8")
    assert fs.mod.installed_version() == "1.5.1"
    with pytest.raises(SystemExit) as e:
        fs.mod.cmd_update(_args())
    assert e.value.code == 2 and fs.ran == []
    # an older installation (1.4.1) runs the step of 1.4.2 as ever
    state.write_text(json.dumps({"modules": ["core"], "version": "1.4.1"}), encoding="utf-8")
    assert fs.mod.cmd_update(_args()) == 0 and fs.ran == ["1.4.2"]


def test_the_run_installer_and_setup_exe_read_a_prerelease_150_state_as_142(tmp_path):
    """The two installers that decide before fea_setup.py runs: the shell header of the Linux .run (the part that compares the installed version with its own,
    run with a real shell) and the Inno script (InstalledVersion)."""
    import shutil
    import subprocess
    mp = importlib.import_module("make_package")
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    inst = iss.split("function InstalledVersion")[1].split("function InitializeSetup")[0]
    assert "(Result = '1.5.0') and (VersionNumber('{#AppVersion}') < VersionNumber('1.5.0'))" in inst and "Result := '1.4.2'" in inst
    sh = shutil.which("sh")
    if not sh:
        pytest.skip("no POSIX shell")
    head = mp.RUN_HEADER
    block = head[head.index('state="$target/.venv/fea_modules.json"'):head.index('[ -n "$mode" ] || mode=install')]

    def run(installed: str, version: str, force: int = 0) -> tuple[int, str]:
        target = tmp_path / f"t_{installed}_{version}_{force}"
        (target / ".venv").mkdir(parents=True)
        (target / ".venv" / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": installed}), encoding="utf-8")
        script = f'target="{target.as_posix()}"\nVERSION="{version}"\nforce={force}\nmode=""\n' + block + 'echo "mode=$mode"\n'
        r = subprocess.run([sh, "-c", script], capture_output=True, text=True, encoding="utf-8")
        return r.returncode, r.stdout.strip()
    assert run("1.5.0", "1.4.2") == (0, "mode=update")             # the pre-release state: the same version, an update / repair
    assert run("1.4.1", "1.4.2") == (0, "mode=update")
    assert run("1.5.1", "1.4.2")[0] == 2                            # really newer: refused without --force
    assert run("1.5.1", "1.4.2", 1) == (0, "mode=update")
    assert run("1.5.0", "1.5.0") == (0, "mode=update")             # the mapping is inert once the installer is 1.5.0
    assert run("1.5.0", "1.4.1")[0] == 2                            # an installer older than the release is still a downgrade


def test_kept_modified_file_reported_once(fs, capsys, monkeypatch):
    """A modified obsolete file is in the saved manifest and in the manifest of an older release (migration 1.0.1) – it is listed once."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.0"}), encoding="utf-8")
    (fs.root / "fea_core/stary.py").write_text("upraveno uživatelem", encoding="utf-8")
    digest = {"fea_core/stary.py": upgrade.bytes_hash(b"s")}
    upgrade.write_manifest(fs.venv / upgrade.MANIFEST_NAME, {"version": "1.0.0", "files": digest})
    upgrade.write_manifest(fs.root / "installer/manifests/1.0.0.json", {"version": "1.0.0", "files": digest})
    monkeypatch.setattr(migrations, "MIGRATIONS", [migrations.Migration("1.0.1", "pre", "staré soubory",
                                                                        migrations.remove_old_release_files)])
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert (fs.root / "fea_core/stary.py").exists()
    assert out.count("fea_core/stary.py") == 1, out
    kept = fs.mod.T("upravené zastaralé soubory ponechány", "modified obsolete files kept")
    assert out.count(kept) == 1 and fs.mod.T("žádné zastaralé soubory", "no obsolete files") not in out


def test_remove_obsolete_reported_set(tmp_path):
    _tree(tmp_path, {"a.py": "upraveno"})
    files = {"a.py": upgrade.bytes_hash(b"puvodni")}
    said, seen = [], set()
    for _ in range(2):
        assert upgrade.remove_obsolete(tmp_path, files, say=said.append, reported=seen) == ([], ["a.py"])
    assert len(said) == 1 and seen == {"a.py"}
    upgrade.remove_obsolete(tmp_path, files, say=said.append)            # without a set it always prints
    assert len(said) == 2


def test_installer_messages_follow_language(tmp_path, monkeypatch):
    """Messages of upgrade / migrations / slim through T(cs, en) (setup_lang) like the rest of the installer."""
    lang = importlib.import_module("setup_lang")
    for cs, start in ((True, "zastaralé soubory programu smazány"), (False, "obsolete program files deleted")):
        monkeypatch.setattr(lang, "CS", cs)
        _tree(tmp_path, {"a.py": "x"})
        said = []
        upgrade.remove_obsolete(tmp_path, {"a.py": upgrade.bytes_hash(b"x")}, say=said.append)
        assert said and said[0].startswith(start), said
    # no Czech text of a message outside T(): say/warn with the literal text with diacritics
    import ast
    czech = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
    for mod in ("upgrade", "migrations", "slim"):
        tree = ast.parse((ROOT / "installer" / f"{mod}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in ("say", "warn", "_say"):
                for arg in node.args[:1]:
                    bare = [n for n in ast.walk(arg) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                            and not any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "T" and n in list(ast.walk(c))
                                        for c in ast.walk(arg))]
                    assert not any(czech & set(n.value) for n in bare), (mod, node.lineno)


# ------------------------------------------------------------------ uv and micromamba only into the program folder
def test_uv_and_mamba_write_only_into_program_folder(monkeypatch, tmp_path):
    fs = importlib.import_module("fea_setup")
    e = fs.uv_env()
    assert e["UV_PYTHON_INSTALL_BIN"] == "0" and e["UV_PYTHON_INSTALL_REGISTRY"] == "0"     # without .local/bin and the registry
    assert Path(e["UV_PYTHON_INSTALL_DIR"]) == fs.TOOLS / "python"
    ps1 = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8-sig")
    assert '$env:UV_PYTHON_INSTALL_BIN = "0"' in ps1 and '$env:UV_PYTHON_INSTALL_REGISTRY = "0"' in ps1
    assert '"--no-bin", "--no-registry"' in ps1 and "$env:TEMP" not in ps1
    sh = (ROOT / "install.sh").read_text(encoding="utf-8")
    assert "UV_PYTHON_INSTALL_BIN=0" in sh and "UV_CACHE_DIR=" in sh
    monkeypatch.setattr(fs, "DL", tmp_path / "dl")
    m = fs.mamba_env()
    for k in ("MAMBA_ROOT_PREFIX", "CONDA_PKGS_DIRS", "XDG_CACHE_HOME", "HOME") + (("USERPROFILE",) if fs.IS_WIN else ()):
        assert Path(m[k]).is_relative_to(tmp_path / "dl"), k                # cache, environments.txt… into _download
    assert set(fs.MAMBA_DIRS) >= {"mamba-root", "mamba-home", "mamba-cache"}
    fs.mamba_cleanup()
    assert not (tmp_path / "dl" / "mamba-home").exists()


# ------------------------------------------------------------------ uninstall: empty program folders
def _junction(link: Path, target: Path) -> bool:
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return True
    except (ImportError, AttributeError, OSError):
        try:
            link.symlink_to(target, target_is_directory=True)
            return True
        except OSError:
            return False


def test_uninstall_removes_empty_program_dirs(fs, monkeypatch, tmp_path):
    """uninstall --pycache: the bytecode and the empty program folders bottom-up; folders with files, .venv / tools (without --all),
    development folders and the targets of links / junctions stay. --all deletes .venv and tools too.
    """
    monkeypatch.setattr(fs.mod, "remove_shortcuts", lambda: None)
    monkeypatch.setattr(fs.mod, "remove_fusion_addin", lambda: None)
    r = fs.root
    _tree(r, {"fea_core/__pycache__/novy.cpython-311.pyc": "b", "stary_balicek/__pycache__/x.cpython-311.pyc": "b",
              "examples/moje.fea": "uziv", "examples/ukazka1.fea.bak": "zaloha", ".venv/lib/a.py": "v", "tools/uv/uv.exe": "u",
              "scratch/pokus.py": "s"})
    for d in ("stara_slozka/prazdna/hloubeji", "docs/prazdna", ".venv/prazdna", "tools/prazdna", ".git/refs/prazdna",
              "scratch/prazdna"):
        (r / d).mkdir(parents=True)
    outside = tmp_path / "mimo" / "prazdna"
    outside.mkdir(parents=True)
    linked = _junction(r / "odkaz", tmp_path / "mimo")
    args = types.SimpleNamespace(pycache=True, all=False)
    assert fs.mod.cmd_uninstall(args) == 0
    for gone in ("stara_slozka", "stary_balicek", "docs", "fea_core/__pycache__"):
        assert not (r / gone).exists(), gone
    for kept in ("fea_core/novy.py", "examples/moje.fea", "examples/ukazka1.fea.bak", ".venv/prazdna", "tools/prazdna",
                 ".git/refs/prazdna", "scratch/prazdna"):
        assert (r / kept).exists(), kept
    assert outside.is_dir()                                                # a junction is not entered
    if linked:
        assert os.path.lexists(r / "odkaz")
    assert fs.mod.cmd_uninstall(types.SimpleNamespace(pycache=False, all=True)) == 0
    assert not (r / ".venv").exists() and not (r / "tools").exists()
    assert (r / "examples/moje.fea").exists() and (r / ".git/refs/prazdna").exists() and outside.is_dir()


# ------------------------------------------------------------------ 1.5.1: an update removes the MYSTRAN program that earlier packages put into tools/mystran
CALCULIX_FILES = {"tools/calculix/ccx_2.23_windows/ccx_spooles.exe": "ccx", "tools/calculix/ccx_2.23_windows/manifest.json": "{}",
                  "tools/calculix/notes-about-mystran.txt": "mine"}


def _install_151(fs, monkeypatch, state_version: str | None, extra: dict[str, str]) -> None:
    """A synthetic installation that the program 1.5.1 is unpacked over: the real migration steps, the files of `extra` on disk, the state of
    `state_version` (None = no installation yet)."""
    monkeypatch.setattr(migrations, "MIGRATIONS", REAL_MIGRATIONS)           # the fixture replaced them by two test steps
    _tree(fs.root, {"fea_core/__init__.py": '__version__ = "1.5.1"\n', **CALCULIX_FILES, **extra})
    upgrade.write_manifest(fs.root / upgrade.PACKAGE_MANIFEST, {"version": "1.5.1", "files": {}})
    if state_version is not None:
        fs.venv.mkdir()
        (fs.venv / "fea_modules.json").write_text(
            json.dumps({"modules": ["core", "plots", "gui", "accel", "direct", "verify"], "version": state_version}), encoding="utf-8")


def _removal_message(fs) -> str:
    return fs.mod.T("MYSTRAN odstraněn", "MYSTRAN removed")


@pytest.mark.parametrize("name", ["mystran.exe", "mystran"])                # Windows and Linux
def test_update_142_to_151_removes_the_bundled_mystran(fs, monkeypatch, capsys, name):
    _install_151(fs, monkeypatch, "1.4.2", {f"tools/mystran/{name}": "bundled program"})
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert not (fs.root / "tools" / "mystran" / name).exists() and not (fs.root / "tools" / "mystran").exists()
    for rel, text in CALCULIX_FILES.items():                                # nothing else under tools is touched
        assert (fs.root / rel).read_text(encoding="utf-8") == text, rel
    assert out.count(_removal_message(fs)) == 1 and migrations.MYSTRAN_URL in out and "tools/mystran" in out, out
    assert fs.mod.T("migrace 1.5.1", "migration 1.5.1") in out
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["version"] == "1.5.1"


def test_update_to_151_keeps_a_mystran_folder_that_holds_another_file(fs, monkeypatch, capsys):
    _install_151(fs, monkeypatch, "1.4.2", {"tools/mystran/mystran.exe": "bundled program", "tools/mystran/MYSTRAN.ini": "mine"})
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert not (fs.root / "tools" / "mystran" / "mystran.exe").exists()
    assert (fs.root / "tools" / "mystran" / "MYSTRAN.ini").read_text(encoding="utf-8") == "mine"
    assert _removal_message(fs) in out


@pytest.mark.parametrize("old", ["1.5.1", None])
def test_a_repair_or_a_first_installation_of_151_keeps_the_users_own_mystran(fs, monkeypatch, capsys, old):
    """The same version again (repair, a new build of 1.5.1) and an installation without a state: a MYSTRAN in tools/mystran is the user's own download."""
    _install_151(fs, monkeypatch, old, {"tools/mystran/mystran.exe": "downloaded by the user"})
    assert (fs.mod.cmd_update(_args()) if old else fs.mod.cmd_install(_args(modules=["gui", "verify"]))) == 0
    out = capsys.readouterr().out
    assert (fs.root / "tools" / "mystran" / "mystran.exe").read_text(encoding="utf-8") == "downloaded by the user"
    assert _removal_message(fs) not in out and fs.mod.T("migrace 1.5.1", "migration 1.5.1") not in out


def _run_mystran_step(root: Path) -> tuple[list[str], list[str]]:
    said: list[str] = []
    warned: list[str] = []
    ctx = migrations.Context(root=root, venv=root / ".venv", vpy=root / "nic", modules=["core"], say=said.append, warn=warned.append)
    only = [m for m in REAL_MIGRATIONS if m.fn is migrations.remove_bundled_mystran]
    assert [(m.version, m.phase) for m in only] == [("1.5.1", "final")]
    migrations.run(ctx, "1.4.2", "1.5.1", "final", migrations=only)
    return said, warned


@pytest.mark.parametrize("linked", ["tools/mystran", "tools"])             # tools = the git worktree whose tools folder is the main checkout's
def test_the_mystran_step_never_deletes_through_a_link(tmp_path, linked):
    root, other = tmp_path / "app", tmp_path / "other"
    _tree(other, {"tools/mystran/mystran.exe": "owned by the other installation"})
    link = root / linked
    link.parent.mkdir(parents=True)
    if not _junction(link, other / linked):
        pytest.skip("no junction / symbolic link possible here")
    try:
        said, warned = _run_mystran_step(root)
        assert (other / "tools" / "mystran" / "mystran.exe").read_text(encoding="utf-8") == "owned by the other installation"
        assert os.path.lexists(link) and warned == []
        assert any(str(link) in s and ("link" in s or "odkaz" in s) for s in said), said
        assert not any("MYSTRAN removed" in s or "MYSTRAN odstraněn" in s for s in said), said
    finally:
        try:
            os.rmdir(link)                                                  # the link only, never what it points to
        except OSError:
            os.unlink(link)


def test_the_mystran_step_is_idempotent_and_silent_without_a_program(tmp_path):
    _tree(tmp_path, {"tools/mystran/mystran.exe": "x"})
    first, _ = _run_mystran_step(tmp_path)
    assert not (tmp_path / "tools" / "mystran").exists() and len(first) == 2          # the step's title and what it did
    again, warned = _run_mystran_step(tmp_path)
    assert len(again) == 1 and warned == []                                          # only the title
    (tmp_path / "tools" / "mystran").mkdir()                                         # an empty folder without a program is left alone
    _run_mystran_step(tmp_path)
    assert (tmp_path / "tools" / "mystran").is_dir()


# ------------------------------------------------------------------ packages: .run and setup.exe
def test_run_header_and_payload(tmp_path):
    mp = importlib.import_module("make_package")
    tgz = tmp_path / "p.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        data = b"hello"
        ti = tarfile.TarInfo("FEAStudio-9.9.9/a.txt")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    run = mp.write_run(tgz, tmp_path / "x.run", "9.9.9")
    raw = run.read_bytes()
    marker = b"\n__FEA_STUDIO_PAYLOAD__\n"
    head, payload = raw.split(marker, 1)
    head.decode("ascii")                                                  # the header purely ASCII, LF
    assert b"\r" not in head and head.startswith(b"#!/bin/sh\n")
    assert payload == tgz.read_bytes()
    assert mp.sha(tgz).encode() in head and b'VERSION="9.9.9"' in head
    assert gzip.decompress(payload)


def test_normalize_eol_follows_gitattributes(tmp_path):
    """Line ends in the package: .ps1 / .bat CRLF (PowerShell 5.1, cmd), .sh and executables LF, the others unchanged; the extensions
    with eol=crlf / eol=lf in .gitattributes must be known to make_package.
    """
    mp = importlib.import_module("make_package")
    st = tmp_path / "stage"
    files = {"installer/install.ps1": b"\xef\xbb\xbfa\nb\r\nc\n", "x.bat": b"a\nb\r\n", "install.sh": b"a\r\nb\r\n",
             "installer/fea_setup.py": b"#!/usr/bin/env python3\r\nx\r\n", "pkg/a.py": b"a\r\nb\n", "fea-studio": b"#!/bin/sh\r\n"}
    for rel, data in files.items():
        (st / rel).parent.mkdir(parents=True, exist_ok=True)
        (st / rel).write_bytes(data)
    mp.normalize_eol(st)
    assert (st / "installer/install.ps1").read_bytes() == b"\xef\xbb\xbfa\r\nb\r\nc\r\n"     # the BOM stays
    assert (st / "x.bat").read_bytes() == b"a\r\nb\r\n"
    assert (st / "install.sh").read_bytes() == b"a\nb\n"
    assert (st / "installer/fea_setup.py").read_bytes() == b"#!/usr/bin/env python3\nx\n"
    assert (st / "fea-studio").read_bytes() == b"#!/bin/sh\n"
    assert (st / "pkg/a.py").read_bytes() == b"a\r\nb\n"
    for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+)\s.*\beol=(crlf|lf)\b", line)
        if not m:
            continue
        pat, eol = m.groups()
        if pat.startswith("*."):
            assert pat[1:] in (mp.CRLF if eol == "crlf" else mp.LF), line
        else:
            assert eol == "lf" and pat in mp.EXEC, line


def test_sha256sums_have_lf(tmp_path):
    """SHA256SUMS.txt for `sha256sum -c`: LF even on Windows, the format "digest␣␣name"."""
    mp = importlib.import_module("make_package")
    a, b = tmp_path / "a.zip", tmp_path / "b c.tar.gz"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    raw = mp.write_sums(tmp_path / "SHA256SUMS.txt", [a, b]).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert lines == [f"{mp.sha(a)}  a.zip", f"{mp.sha(b)}  b c.tar.gz"]
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S.*", l) for l in lines)


def test_iss_consistent_with_modules():
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    assert "AppId={{6F1E7C2A-6E0B-4B7B-9C55-FEA5D1000001}" in iss            # the same AppId = update over the old one
    assert "{6F1E7C2A-6E0B-4B7B-9C55-FEA5D1000001}_is1" in iss
    assert "PrivilegesRequired=lowest" in iss and r"{localappdata}\Programs\FEA Studio" in iss
    mods = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))["modules"]
    listed = re.search(r"MODULES = '([^']+)'", iss).group(1).split()
    comps = re.findall(r'^Name: "(\w+)"; Description: "[^"]*"; Types:', iss.split("[Components]")[1].split("[")[0], re.M)
    optional = {k for k, m in mods.items() if k not in ("core", "docs")}
    assert set(listed) == optional and set(comps) == optional | {"core"}
    # examples: a modified one is backed up before being overwritten (not onlyifdoesntexist – new versions of the examples would not be updated);
    # uninstall: empty program folders (also from the zip) – verified with a test setup.exe (scratch/fix_install/iss_test.py)
    files = iss.split("[Files]")[1].split("\n[")[0]
    assert r'Excludes: "\examples\*.fea,\tools\python\*"' in files and "BeforeInstall: BackupModifiedExample" in files
    # onlyifdoesntexist only for the bundled Python (the same-named folder = the same content; during an update it is held by
    # the running MCP server / GUI) – not for the examples
    only = [ln for ln in files.splitlines() if "onlyifdoesntexist" in ln]
    assert len(only) == 1 and r'DestDir: "{app}\tools\python"' in files.split("onlyifdoesntexist")[0].splitlines()[-2]
    assert 'DestName: "fea_new_manifest.json"; Flags: dontcopy' in files
    assert "function RunningFromDir" in iss and "CustomMessage('Running')" in iss
    assert "usPostUninstall" in iss and "RemoveEmptyDirs(ExpandConstant('{app}'))" in iss
    assert "skipifdoesntexist" in iss.split("[UninstallRun]")[1].split("\n[")[0]
    assert not (ROOT / "installer" / "uninstall.ps1").exists()          # obsolete – uninstall only through fea_setup.py


# ------------------------------------------------------------------ build id (same version, another build = update)
def test_install_kind_with_build_ids():
    fs = importlib.import_module("fea_setup")
    assert fs.install_kind("1.4.2", "1.4.2", "20261101T000000Z-aaaa1111", "20261109T000000Z-bbbb2222") == "update"
    assert fs.install_kind("1.4.2", "1.4.2", "b1", "b1") == "repair"
    assert fs.install_kind("1.4.2", "1.4.2", None, "b2") == "repair"          # unknown old build (git / older package)
    assert fs.install_kind("1.4.2", "1.4.2", "b1", None) == "repair"
    assert fs.install_kind("1.4.0", "1.4.2", "b1", "b1") == "update"
    assert fs.install_kind("1.6.0", "1.4.2", "b1", "b2") == "downgrade"


def test_make_build_id_format(tmp_path):
    b = upgrade.make_build_id(ROOT, when=0)
    assert re.fullmatch(r"19700101T000000Z(-[0-9a-f]{8})?", b), b
    assert upgrade.make_build_id(tmp_path, when=0) == "19700101T000000Z"           # outside git: only the time


def test_make_build_id_names_the_packed_ref():
    """make_package --ref X: the build id carries the hash of X, not of HEAD."""
    import subprocess
    if subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("no git history")
    short = subprocess.run(["git", "rev-parse", "--short=8", "HEAD~1"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert upgrade.make_build_id(ROOT, when=0, ref="HEAD~1") == f"19700101T000000Z-{short}"
    assert upgrade.make_build_id(ROOT, when=0, ref="--output=x") == "19700101T000000Z"      # never an option for git


def test_manifest_build_roundtrip(tmp_path):
    _tree(tmp_path, {"fea_core/a.py": "a"})
    m = upgrade.manifest_of_dir(tmp_path, "1.2.3", build="20261109T120000Z-0123abcd")
    assert m["build"] == "20261109T120000Z-0123abcd"
    assert "build" not in upgrade.manifest_of_dir(tmp_path, "1.2.3")
    assert upgrade.manifest_build(tmp_path) is None
    upgrade.write_manifest(tmp_path / upgrade.PACKAGE_MANIFEST, m)
    assert upgrade.manifest_build(tmp_path) == "20261109T120000Z-0123abcd"


def _build_manifest(root, build):
    upgrade.write_manifest(root / upgrade.PACKAGE_MANIFEST, {"version": "1.0.1", "build": build, "files": {
        "fea_core/__init__.py": upgrade.bytes_hash(b'__version__ = "1.0.1"\n'), "fea_core/novy.py": upgrade.bytes_hash(b"n")}})


def test_update_same_version_new_build_is_update(fs, capsys):
    """The same version with another build id = a normal update (obsolete files by the saved manifest, logged), not "repair"."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.1", "build": "B-old"}),
                                              encoding="utf-8")
    upgrade.write_manifest(fs.venv / upgrade.MANIFEST_NAME, {"version": "1.0.1", "build": "B-old", "files": {
        "fea_core/stary.py": upgrade.bytes_hash(b"s")}})
    _build_manifest(fs.root, "B-new")
    fs.mod.LOG = fs.root / "install.log"
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert "B-old → B-new" in out
    assert fs.mod.T("opravná instalace", "repair installation") not in out
    assert not (fs.root / "fea_core/stary.py").exists()                     # obsolete file of the old build removed
    st = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))
    assert st["version"] == "1.0.1" and st["build"] == "B-new"
    assert "build B-old → B-new, update" in (fs.root / "install.log").read_text(encoding="utf-8")
    fs.mod.LOG = None


def test_update_same_build_is_repair(fs, capsys):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.1", "build": "B1"}),
                                              encoding="utf-8")
    _build_manifest(fs.root, "B1")
    assert fs.mod.cmd_update(_args()) == 0
    assert fs.mod.T("opravná instalace verze 1.0.1", "repair installation of version 1.0.1") in capsys.readouterr().out
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["build"] == "B1"


def test_write_state_keeps_build_on_remove(fs):
    fs.venv.mkdir()
    fs.mod.write_state(["core"], "1.0.1", "B9")
    fs.mod.write_state(["core", "gui"])                                       # e.g. `remove` / `slim` – no new package
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["build"] == "B9"


# ------------------------------------------------------------------ the list printed is the list installed (setup.exe --exact), and new standard modules
def _listed(out: str) -> list[str]:
    lines = [ln for ln in out.splitlines() if ln.startswith(("Modules:", "Moduly:"))]
    assert len(lines) == 1, out
    return [k.strip() for k in lines[0].split(":", 1)[1].split("(~")[0].split(",")]


def _installed_from(fs, monkeypatch, state: list[str]) -> None:
    monkeypatch.setattr(migrations, "MIGRATIONS", REAL_MIGRATIONS)
    monkeypatch.setattr(fs.mod, "EXT", {"ccx_static": lambda: None})
    _tree(fs.root, {"fea_core/__init__.py": '__version__ = "1.5.1"\n'})
    upgrade.write_manifest(fs.root / upgrade.PACKAGE_MANIFEST, {"version": "1.5.1", "files": {}})
    fs.venv.mkdir(exist_ok=True)
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": state, "version": "1.4.1"}), encoding="utf-8")


@pytest.mark.parametrize("state, selection, expect_motion", [
    (["core", "plots", "gui", "ngsolve"], ["gui", "plots", "ngsolve"], False),                 # the old solver: no Exudyn unasked
    (["core", "plots", "gui", "pardiso", "ngsolve"], ["gui", "plots", "ngsolve"], False),
    (["core", "plots", "gui", "motion", "ngsolve"], ["gui", "plots", "motion", "ngsolve"], True),
    (["core", "plots", "gui", "ngsolve"], ["gui", "plots", "ngsolve", "motion"], True),        # motion ticked in the wizard: ngsolve stays
])
def test_exact_upgrade_from_141_prints_the_list_it_installs(fs, monkeypatch, capsys, state, selection, expect_motion):
    _installed_from(fs, monkeypatch, state)
    assert fs.mod.cmd_install(_args(modules=list(selection), exact=True)) == 0
    out = capsys.readouterr().out
    installed = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["modules"]
    assert _listed(out) == installed, out
    assert f"sync:{' '.join(installed)}" in fs.calls
    assert ("motion" in installed) is expect_motion and ("ngsolve" in installed) is expect_motion and "direct" in installed
    assert "sketch" not in installed                                  # --exact: the selection decides, nothing is added behind it


def test_adding_ngsolve_says_that_it_brings_motion(fs, monkeypatch, capsys):
    _installed_from(fs, monkeypatch, ["core", "plots", "gui"])
    assert fs.mod.cmd_install(_args(modules=["ngsolve"])) == 0
    out = capsys.readouterr().out
    installed = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["modules"]
    assert _listed(out) == installed and "motion" in installed
    assert ("requires the module motion" in out) or ("vyžaduje modul motion" in out), out


def test_an_update_of_a_standard_installation_gets_the_new_standard_modules(fs, monkeypatch, capsys):
    """The zip / .run update keeps the modules of last time; a standard installation of 1.4.1 then lacked sketch, which a fresh one has."""
    standard_141 = ["core", "plots", "gui", "accel", "pardiso", "beams", "report", "motion"]
    _installed_from(fs, monkeypatch, standard_141)
    assert fs.mod.cmd_update(_args()) == 0
    installed = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["modules"]
    assert set(fs.mod.PROFILES["standard"]) <= set(installed) and "sketch" in installed
    assert _listed(capsys.readouterr().out) == installed
    _installed_from(fs, monkeypatch, ["core", "plots", "gui", "accel", "beams"])               # hand-picked: left alone
    assert fs.mod.cmd_update(_args()) == 0
    assert "sketch" not in json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["modules"]


def test_the_new_standard_modules_are_really_new_and_the_step_is_idempotent():
    old = json.loads((ROOT / "installer" / "manifests" / "1.4.1.json").read_text(encoding="utf-8"))["files"]
    assert old                                                                                 # 1.4.1 is a released version
    profiles = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))["profiles"]
    assert set(migrations.NEW_STANDARD_151) | set(migrations.STANDARD_BEFORE_151) <= set(profiles["standard"])
    once, note = migrations.new_standard_modules(list(migrations.STANDARD_BEFORE_151))
    assert note and migrations.new_standard_modules(once) == (once, "")
    assert not migrations.pending("1.5.1", "1.5.1")

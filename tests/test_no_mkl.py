"""No Intel MKL anywhere (F13 step 2, user decision): nothing of Intel is shipped, downloaded or optional – no module, no package in the version
locks, no CalculiX build with PARDISO, no MKL variable in the environment of ccx; an older installation is cleaned by the migration 1.4.2, and the
stored words of older studies (`solver = "PARDISO"`, backend `ngs_pardiso`) keep loading. Fast, without Gmsh and ccx."""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from fea_core import config, modules
from fea_core.study import Study

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

INTEL_DISTS = ("mkl", "intel-openmp", "intel_openmp", "tbb")


# ------------------------------------------------------------------ nothing of Intel in the definitions
def test_no_module_profile_or_pip_pin_names_intel():
    raw = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))
    assert "pardiso" not in raw["modules"] and set(raw["retired"]) == {"pardiso"}
    for name, keys in raw["profiles"].items():
        assert "pardiso" not in keys, name
    for k, m in raw["modules"].items():
        pins = [p.split("==")[0].lower().replace("_", "-") for p in m.get("pip", []) + m.get("pip_nodeps", [])]
        assert not set(pins) & {"mkl", "intel-openmp", "tbb"}, k
        assert "ccx_i4" not in m.get("ext", []) and m.get("check") != "pardiso", k


@pytest.mark.parametrize("plat", ["windows-x86_64", "linux-x86_64"])
def test_the_version_locks_have_no_intel_package(plat):
    names = set()
    for line in (ROOT / "installer" / "locks" / f"{plat}.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", " ")):
            names.add(line.split("==")[0].strip().lower().replace("_", "-"))
    assert names and not names & {"mkl", "intel-openmp", "tbb"}
    # the extras of pyproject.toml are the modules
    import tomllib
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert "pardiso" not in extras


def test_the_package_of_the_worker_is_gone_and_config_has_no_mkl():
    assert not (ROOT / "fea_core" / "pardiso").exists()
    with pytest.raises(ImportError):
        importlib.import_module("fea_core.pardiso")
    for attr in ("MKL_DIR", "MKL_RT_NAME", "mkl_rt", "CCX_PARDISO"):
        assert not hasattr(config, attr), attr
    assert set(config.ccx_solvers()) == {"PASTIX", "SPOOLES"}
    assert modules.get("pardiso") is None and "pardiso" in modules.retired()


def test_the_environment_of_ccx_has_no_mkl_variable(monkeypatch):
    for det in (False, True):
        env = config.ccx_env(deterministic=det)
        assert not [k for k in env if k.upper().startswith(("MKL_", "KMP_")) and k not in os.environ], det


def test_ccx_i4_is_never_selected(tmp_path, monkeypatch):
    d = tmp_path / "calculix_2.23_4win"
    d.mkdir()
    (d / f"ccx_i4{config.EXE}").write_text("x")
    monkeypatch.delenv("FEA_CCX", raising=False)
    monkeypatch.setattr(config, "CCX_TOOLS", tmp_path)
    monkeypatch.setattr(config, "which", lambda name: None)
    assert config.find_ccx().name == f"ccx_spooles{config.EXE}" and not config.find_ccx().exists()      # the missing default, not the Intel build
    (d / f"ccx_spooles{config.EXE}").write_text("x")
    assert config.find_ccx() == d / f"ccx_spooles{config.EXE}"


@pytest.mark.parametrize("name", ["ccx_i4.exe", "CCX_I4.EXE", "ccx_i4"])
def test_a_leftover_fea_ccx_naming_ccx_i4_is_skipped_with_a_clear_message(tmp_path, monkeypatch, name):
    """An installation that used the MKL build often has FEA_CCX=...ccx_i4.exe in the user's environment: the program must neither start that file (it is
    deleted by the update, or it would fail to load the Intel libraries) nor fail without a word."""
    d = tmp_path / "calculix_2.23_4win"
    d.mkdir()
    (d / name).write_text("x")
    (d / f"ccx_spooles{config.EXE}").write_text("x")
    monkeypatch.setattr(config, "CCX_TOOLS", tmp_path)
    monkeypatch.setattr(config, "which", lambda n: None)
    monkeypatch.setenv("FEA_CCX", str(d / name))
    assert config.find_ccx() == d / f"ccx_spooles{config.EXE}"                    # the variable is skipped
    note = config.ccx_env_note()
    assert "FEA_CCX" in note and "ignored" in note and name in note
    from fea_core import solvers
    from fea_core.study import Study
    assert note in solvers.solver_warnings(Study("t", ""))                       # the readiness check says it before a calculation starts
    assert solvers.summary()["note"] == note and any(note in ln for ln in solvers.table())      # MCP / REST status and `fea_setup.py doctor`
    monkeypatch.setenv("FEA_CCX", str(tmp_path / "my_ccx"))                      # a ccx of the user's own is still honoured
    assert config.find_ccx() == tmp_path / "my_ccx" and config.ccx_env_note() == ""
    assert "note" not in solvers.summary()
    monkeypatch.delenv("FEA_CCX")
    assert config.ccx_env_note() == ""


def test_a_missing_ccx_with_a_leftover_variable_says_why(tmp_path, monkeypatch):
    from fea_core.ccx.runner import CcxError, run_ccx
    from fea_core import solvers
    monkeypatch.setenv("FEA_CCX", str(tmp_path / "ccx_i4.exe"))
    monkeypatch.setattr(config, "CCX_EXE", tmp_path / f"ccx_spooles{config.EXE}")          # nothing installed
    with pytest.raises(CcxError, match="FEA_CCX"):
        run_ccx(tmp_path, "job")
    assert "FEA_CCX" in (solvers.backend_error("ccx") or "")


# ------------------------------------------------------------------ the words of older studies
def test_the_stored_words_of_older_studies_keep_loading():
    st = Study.from_dict({"name": "old", "geometry": "", "solver": {"solver": "PARDISO", "backend": "ngs_pardiso"}})
    assert st.solver.solver == config.AUTO_SOLVER == "PARDISO"                 # the word of "automatic"
    assert st.solver.backend == "ngs_direct"
    assert Study("new", "").solver.solver == config.AUTO_SOLVER


def test_the_status_of_a_study_does_not_report_the_stored_word_of_automatic():
    """MCP / REST `fea_study_status`: an agent must not read the stored word as a request for a solver that no longer exists."""
    from fea_core.mcp.tools_study import solver_setting
    assert solver_setting(Study("new", ""))["solver"] == "automatic"
    old = Study.from_dict({"name": "old", "geometry": "", "solver": {"solver": "PARDISO", "backend": "ngs_pardiso"}})
    assert "PARDISO" not in json.dumps(solver_setting(old)).upper() and solver_setting(old)["backend"] == "ngs_direct"
    st = Study("x", "")
    st.solver.solver = "SPOOLES"
    assert solver_setting(st)["solver"] == "SPOOLES" and st.solver.solver == "SPOOLES"       # an explicit choice as it is; the study is untouched
    assert Study("new", "").solver.solver == config.AUTO_SOLVER                                 # … the stored word itself does not change (files stay readable)


def test_the_stored_word_maps_by_the_procedure(monkeypatch):
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_static.exe"))
    assert [config.solver_name("PARDISO", p) for p in ("static", "heat", "modal", "shell")] == ["PASTIX", "PASTIX", "SPOOLES", "SPOOLES"]
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles"))
    assert {config.solver_name("PARDISO", p) for p in ("static", "heat", "modal", "shell")} == {"SPOOLES"}


# ------------------------------------------------------------------ the migration of an older installation
migrations = importlib.import_module("migrations")


def _dist(site: Path, venv: Path, name: str, version: str, files: dict[str, bytes], data_files: dict[str, bytes] | None = None) -> None:
    """A fake installed distribution: files in site-packages and (like the wheel of mkl) files in the environment root, listed in RECORD."""
    rows = []
    for rel, data in files.items():
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append(f"{rel},sha256=x,{len(data)}")
    for rel, data in (data_files or {}).items():
        p = venv / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append(f"{os.path.relpath(p, site).replace(chr(92), '/')},sha256=x,{len(data)}")
    di = site / f"{name}-{version}.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    (di / "RECORD").write_text("\n".join(rows + [f"{di.name}/METADATA,,", f"{di.name}/RECORD,,"]) + "\n")


def _old_install(tmp_path: Path, win: bool = True):
    root = tmp_path / "FEA Studio"
    venv = root / ".venv"
    site = venv / "Lib" / "site-packages" if win else venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    libdir = "Library/bin" if win else "lib"

    def dll(n: str) -> str:
        return f"{libdir}/{n}"
    _dist(site, venv, "mkl", "2023.2.0", {"mkl/__init__.py": b"x"}, {dll("mkl_rt.2.dll" if win else "libmkl_rt.so.2"): b"M", dll("mkl_core.2.dll" if win else "libmkl_core.so.2"): b"M"})
    _dist(site, venv, "intel_openmp", "2023.2.4", {}, {dll("libiomp5md.dll" if win else "libiomp5.so"): b"M"})
    _dist(site, venv, "tbb", "2021.13.1", {"tbb/__init__.py": b"x"}, {dll("tbb12.dll" if win else "libtbb.so.12"): b"M"})
    _dist(site, venv, "numpy", "1.26.4", {"numpy/__init__.py": b"n"})
    _dist(site, venv, "mkl_service", "2.4.0", {"mkl_service/__init__.py": b"s"})            # another package whose name starts with mkl: stays
    calc = root / "tools" / "calculix" / ("calculix_2.23_4win" if win else "ccx_2.23_linux")
    calc.mkdir(parents=True)
    for n in (("ccx_i4.exe", "ccx_static.exe") if win else ("ccx_i4", "ccx_spooles")) + ("ccx_i4.log", "README_Install"):      # ccx_i4.log: not a file of the old installer
        (calc / n).write_text("x")
    return root, venv, site, calc


def _ctx(root: Path, venv: Path, said: list, warned: list):
    return migrations.Context(root=root, venv=venv, vpy=venv / "Scripts" / "python.exe", modules=["core", "pardiso"], say=said.append, warn=warned.append)


@pytest.mark.parametrize("win", [True, False])
def test_the_migration_removes_ccx_i4_and_the_intel_packages_and_is_idempotent(tmp_path, win):
    root, venv, site, calc = _old_install(tmp_path, win)
    said, warned = [], []
    migrations.remove_intel_components(_ctx(root, venv, said, warned))
    assert not warned and said and "Intel" in said[0]
    keep = "ccx_static.exe" if win else "ccx_spooles"
    assert sorted(p.name for p in calc.iterdir()) == sorted(["README_Install", "ccx_i4.log", keep])        # the CalculiX without MKL stays, and a foreign file with the prefix
    left = sorted(p.name for p in site.iterdir())
    assert left == ["mkl_service", "mkl_service-2.4.0.dist-info", "numpy", "numpy-1.26.4.dist-info"], left
    libdir = venv / ("Library" if win else "lib")
    assert not [p for p in libdir.rglob("*") if p.is_file() and any(t in p.name.lower() for t in ("mkl", "iomp", "tbb"))]
    said.clear()
    migrations.remove_intel_components(_ctx(root, venv, said, warned))                                   # a second run finds nothing
    assert said == [] and warned == []
    assert sorted(p.name for p in site.iterdir()) == left


def test_the_migration_never_deletes_outside_the_environment(tmp_path):
    root, venv, site, calc = _old_install(tmp_path)
    outside = tmp_path / "keep.txt"
    outside.write_text("mine")
    rec = site / "mkl-2023.2.0.dist-info" / "RECORD"
    rec.write_text(rec.read_text() + "../../../../keep.txt,sha256=x,4\n../../../tools/calculix/calculix_2.23_4win/ccx_static.exe,,\n")
    migrations.remove_intel_components(_ctx(root, venv, [], []))
    assert outside.read_text() == "mine"                                  # a RECORD is data: a path outside the environment is ignored
    assert (calc / "ccx_static.exe").is_file()                            # … and ccx_static is no Intel file whatever a RECORD says


def test_the_migration_is_registered_for_1_4_2_and_drops_the_module(tmp_path):
    # the final phase: nothing of the previous installation goes before the packages and the new CalculiX are in place (an offline update)
    steps = migrations.pending("1.4.1", "1.4.2", "final")
    assert any(m.fn is migrations.remove_intel_components for m in steps)
    assert not any(m.fn is migrations.remove_intel_components for m in migrations.pending("1.4.1", "1.4.2", "pre"))
    assert [m.phase for m in migrations.HOUSEKEEPING if m.fn is migrations.remove_intel_components] == ["final"]
    assert not any(m.fn is migrations.remove_intel_components for m in migrations.pending("1.4.2", "1.4.2"))
    mods, notes = migrations.plan_modules(["core", "gui", "pardiso", "accel"], "1.4.1", "1.4.2")
    assert mods == ["core", "gui", "accel", "direct"] and any("pardiso" in n and "direct" in n for n in notes)       # the fast solver module is replaced, not just dropped
    mods2, notes2 = migrations.plan_modules(mods, "1.4.1", "1.4.2")          # idempotent
    assert mods2 == mods and not any("pardiso" in n for n in notes2)
    # without accel in the old list (it comes with direct), and a list that already has direct is not doubled
    assert migrations.plan_modules(["core", "pardiso"], "1.4.1", "1.4.2")[0] == ["core", "accel", "direct"]
    assert migrations.plan_modules(["core", "direct", "pardiso", "accel"], "1.4.1", "1.4.2")[0] == ["core", "direct", "accel"]
    # the unfinished list of an interrupted update: the step runs again without harm
    root, venv, site, calc = _old_install(tmp_path)
    ran = migrations.run(_ctx(root, venv, [], []), "1.4.1", "1.4.2", "final")
    assert any("Intel MKL removed" in r or "Intel MKL odstraněn" in r for r in ran)
    assert not (calc / "ccx_i4.exe").exists()


def test_a_missing_environment_is_not_an_error(tmp_path):
    ctx = _ctx(tmp_path / "nothing", tmp_path / "nothing" / ".venv", [], [])
    migrations.remove_intel_components(ctx)                                  # no tools, no .venv: nothing to do
    assert ctx.say is not None


def test_a_file_held_by_a_running_program_keeps_the_package_registered_until_the_next_run(tmp_path, monkeypatch):
    root, venv, site, calc = _old_install(tmp_path)
    real_unlink = Path.unlink
    held = "mkl_rt.2.dll"

    def locked(self, *a, **k):
        if self.name == held:
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", locked)
    said, warned = [], []
    migrations.remove_intel_components(_ctx(root, venv, said, warned))
    assert len(warned) == 1 and "1" in warned[0]
    assert (site / "mkl-2023.2.0.dist-info").is_dir()                          # still registered: the next run finishes it
    assert not (site / "tbb-2021.13.1.dist-info").exists()                     # the others are done
    monkeypatch.setattr(Path, "unlink", real_unlink)                           # the program was closed
    warned.clear()
    migrations.remove_intel_components(_ctx(root, venv, said, warned))
    assert not warned and not (site / "mkl-2023.2.0.dist-info").exists()
    assert not list(venv.rglob(held))


# ------------------------------------------------------------------ review: only the known files, never through a link, a cleanup that is retried
def test_only_the_files_of_the_old_installer_are_deleted(tmp_path):
    root, venv, site, calc = _old_install(tmp_path)
    deep = calc / "sub" / "deeper"
    deep.mkdir(parents=True)
    (deep / "ccx_i4.exe").write_text("x")                          # two levels below the archive folder: not a place of the installer, nor of find_ccx
    (calc / "ccx_i4_notes.txt").write_text("mine")                  # a foreign file with the prefix
    own = root / "tools" / "calculix" / "own"
    own.mkdir()
    (own / "ccx_i4.exe").write_text("x")                           # a direct subfolder of tools/calculix: the place the installer extracted into
    migrations.remove_intel_components(_ctx(root, venv, [], []))
    assert not (calc / "ccx_i4.exe").exists() and not (own / "ccx_i4.exe").exists()
    assert (deep / "ccx_i4.exe").is_file() and (calc / "ccx_i4_notes.txt").is_file() and (calc / "ccx_i4.log").is_file()


def _link(target: Path, link: Path) -> bool:
    """A directory link: a junction on Windows (what the git worktrees of the project use), a symbolic link elsewhere. False = not possible here."""
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, ImportError, AttributeError):
        return False
    return True


def _drop_link(link: Path) -> None:
    try:
        os.rmdir(link)                                              # a junction (and a directory symlink on Windows): the link only
    except OSError:
        os.unlink(link)


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("linked", [(".venv",), ("tools",), (".venv", "tools")])
def test_the_migration_does_not_delete_through_a_junction_or_a_symlink(tmp_path, linked):
    """A git worktree: `.venv` and `tools` are junctions to the main checkout – the update of the worktree must not clean the main installation."""
    main_root, _, main_site, main_calc = _old_install(tmp_path / "main")
    wt_root, wt_venv, wt_site, wt_calc = _old_install(tmp_path / "worktree")
    made = []
    try:
        for name in linked:
            shutil.rmtree(wt_root / name)
            if not _link(main_root / name, wt_root / name):
                pytest.skip("cannot create a directory link here")
            made.append(wt_root / name)
        before = _files(main_root)
        said, warned = [], []
        migrations.remove_intel_components(_ctx(wt_root, wt_venv, said, warned))
        assert _files(main_root) == before                          # nothing of the main installation was touched
        assert any("junction" in t for t in said) and not warned
        # what is a real folder in the worktree is cleaned as usual
        if ".venv" not in linked:
            assert not list(wt_site.glob("mkl-*.dist-info")) and (wt_site / "numpy").is_dir()
        if "tools" not in linked:
            assert not (wt_calc / "ccx_i4.exe").exists() and (wt_calc / "ccx_static.exe").is_file()
        # the second run is the same (the notes come again, nothing is deleted)
        migrations.remove_intel_components(_ctx(wt_root, wt_venv, [], []))
        assert _files(main_root) == before
    finally:
        for link in made:
            _drop_link(link)
    assert main_calc.is_dir() and main_site.is_dir()


def test_the_link_predicate_and_a_skip_with_a_patched_predicate(tmp_path, monkeypatch):
    """Where a link cannot be created (Windows without the right to make symlinks) the logic is checked with a patched predicate."""
    root, venv, site, calc = _old_install(tmp_path)
    assert not migrations.is_link(venv) and not migrations.is_link(root / "tools") and not migrations.is_link(root / "missing")
    real = migrations.is_link
    monkeypatch.setattr(migrations, "is_link", lambda p: p == root / "tools" or real(p))
    said, warned = [], []
    migrations.remove_intel_components(_ctx(root, venv, said, warned))
    assert (calc / "ccx_i4.exe").is_file() and any("junction" in t for t in said)          # tools: skipped
    assert not list(site.glob("mkl-*.dist-info"))                                         # .venv: cleaned
    root2, venv2, site2, calc2 = _old_install(tmp_path / "second")
    monkeypatch.setattr(migrations, "is_link", lambda p: p == venv2)
    migrations.remove_intel_components(_ctx(root2, venv2, [], []))
    assert list(site2.glob("mkl-*.dist-info")) and not (calc2 / "ccx_i4.exe").exists()    # .venv skipped, tools cleaned


def test_a_cloud_placeholder_is_not_a_link(tmp_path, monkeypatch):
    """Only the reparse tags that redirect (junction, symlink) count: the reparse point of a sync folder (OneDrive and other sync clients) is an ordinary folder."""
    d = tmp_path / "folder"
    d.mkdir()
    real_lstat = os.lstat

    class _St:
        st_reparse_tag = 0x9000001A                                 # a cloud-files tag
        st_file_attributes = 0x400 | 0x10

    monkeypatch.setattr(migrations.os, "lstat", lambda p, *a, **k: _St() if Path(p) == d else real_lstat(p, *a, **k))
    assert not migrations.is_link(d)


def test_a_held_file_is_finished_by_the_next_run_of_the_same_version(tmp_path, monkeypatch):
    """The state is written as 1.4.2 although a DLL was held: the repair / update run of the SAME version retries the cleanup (HOUSEKEEPING)."""
    root, venv, site, calc = _old_install(tmp_path)
    real_unlink = Path.unlink

    def locked(self, *a, **k):
        if self.name == "mkl_rt.2.dll":
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", locked)
    said, warned = [], []
    ctx = _ctx(root, venv, said, warned)
    done = migrations.run(ctx, "1.4.1", "1.4.2", "final")
    intel = [w for w in warned if "Intel" in w]       # the final phase also reports on the old third-party CalculiX
    assert done and len(intel) == 1 and (site / "mkl-2023.2.0.dist-info").is_dir()
    assert "again" in intel[0] or "znovu" in intel[0]
    monkeypatch.setattr(Path, "unlink", real_unlink)                # the program was closed
    warned.clear()
    said.clear()
    assert migrations.run(ctx, "1.4.2", "1.4.2", "final") == []      # no migration step of this version …
    assert not [w for w in warned if "Intel" in w] and not (site / "mkl-2023.2.0.dist-info").exists() and any("Intel" in t for t in said)   # gone
    assert not list(venv.rglob("mkl_rt.2.dll"))
    said.clear()
    warned.clear()
    assert migrations.run(ctx, "1.4.2", "1.4.2", "final") == [] and said == [] and not [w for w in warned if "Intel" in w]   # silent
    assert migrations.run(ctx, "1.4.2", "1.4.2", "post") == [] and said == []
    # the cleanup runs once per call also when the update step of the version ran in it
    root2, venv2, site2, calc2 = _old_install(tmp_path / "second")
    said2 = []
    migrations.run(_ctx(root2, venv2, said2, []), "1.4.1", "1.4.2", "final")
    assert sum(("Intel removed" in t or "Intel odstraněn" in t) for t in said2) == 1


def test_run_with_its_own_list_of_steps_does_not_add_the_housekeeping(tmp_path):
    root, venv, site, calc = _old_install(tmp_path)
    ran = []
    ms = [migrations.Migration("1.0.1", "pre", "x", lambda c: ran.append(1))]
    migrations.run(_ctx(root, venv, [], []), "1.0.0", "1.0.1", "pre", migrations=ms)
    assert ran == [1] and (calc / "ccx_i4.exe").is_file()


def test_a_folder_that_cannot_be_removed_does_not_stop_the_other_packages(tmp_path, monkeypatch):
    root, venv, site, calc = _old_install(tmp_path)
    real_rmdir = Path.rmdir

    def rmdir(self):
        if self.name == "mkl":                                      # the emptied package folder of the first distribution is held
            raise PermissionError("in use")
        return real_rmdir(self)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    migrations.remove_intel_components(_ctx(root, venv, [], []))
    assert not list(site.glob("intel_openmp-*.dist-info")) and not list(site.glob("tbb-*.dist-info"))      # the distributions after it are removed
    assert not list(site.glob("mkl-*.dist-info"))                    # the failed rmdir is only a folder left behind, the files are gone


def test_a_dist_info_that_cannot_be_removed_is_reported_and_retried(tmp_path, monkeypatch):
    root, venv, site, calc = _old_install(tmp_path)
    real_rmtree = shutil.rmtree
    monkeypatch.setattr(migrations.shutil, "rmtree", lambda p, *a, **k: None if Path(p).name.startswith("tbb-") else real_rmtree(p, *a, **k))
    warned = []
    migrations.remove_intel_components(_ctx(root, venv, [], warned))
    assert len(warned) == 1 and (site / "tbb-2021.13.1.dist-info").is_dir()
    monkeypatch.undo()
    warned.clear()
    migrations.remove_intel_components(_ctx(root, venv, [], warned))                       # the retry finishes it
    assert not warned and not (site / "tbb-2021.13.1.dist-info").exists()


def test_a_site_packages_outside_the_environment_is_left_alone(tmp_path):
    import types
    root, venv, site, calc = _old_install(tmp_path)
    (venv / "Scripts").mkdir(exist_ok=True)
    (venv / "Scripts" / "python.exe").write_text("x")                 # a "working" environment: the setup module is asked for its site-packages
    elsewhere = tmp_path / "elsewhere" / "site-packages"
    elsewhere.mkdir(parents=True)
    said, warned = [], []
    ctx = _ctx(root, venv, said, warned)
    ctx.setup = types.SimpleNamespace(site_packages=lambda: str(elsewhere))
    migrations.remove_intel_components(ctx)
    assert list(site.glob("mkl-*.dist-info")) and not warned           # nothing deleted in the environment, no misleading "link" note
    assert not any("junction" in t for t in said)


def test_a_leftover_variable_pointing_at_ccx_i4_is_reported(tmp_path, monkeypatch):
    root, venv, site, calc = _old_install(tmp_path)
    monkeypatch.setenv("FEA_CCX", str(calc / "ccx_i4.exe"))
    warned = []
    migrations.remove_intel_components(_ctx(root, venv, [], warned))
    assert len(warned) == 1 and "FEA_CCX" in warned[0]
    monkeypatch.setenv("FEA_CCX", str(calc / "ccx_static.exe"))
    warned.clear()
    migrations.remove_intel_components(_ctx(root, venv, [], warned))
    assert not warned
    monkeypatch.delenv("FEA_CCX")
    migrations.remove_intel_components(_ctx(root, venv, [], warned))
    assert not warned


# ------------------------------------------------------------------ the names stay out of the product: every remaining mention is justified AND counted here
import re  # noqa: E402
import subprocess  # noqa: E402
import zipfile  # noqa: E402

PATTERN = re.compile(r"mkl|pardiso|ccx_i4|intel[-_]openmp|\btbb\b", re.IGNORECASE)
SELF = "tests/test_no_mkl.py"                       # defines the pattern and tests the migration: its own mentions are the guard, not a finding
# files that are not text (their bytes say nothing) – every OTHER file that does not decode as UTF-8 must be listed in NON_UTF8_TEXT or the test fails
BINARY_SUFFIX = (".png", ".bmp", ".ico", ".icns", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".whl", ".exe", ".dll", ".so", ".npz",
                 ".npy", ".fea", ".ttf", ".otf", ".woff", ".woff2")
ZIP_TEXT_SUFFIX = (".docx",)                        # a zip of XML parts: the text of its parts is scanned
NON_UTF8_TEXT = {                                   # licence texts of third parties stored byte for byte in another encoding (.gitattributes: -text)
    "installer/license_texts/pastix/LICENSE-PaStiX4CalculiX-CeCILL-2.0": "CeCILL text (French, Latin-1)",
    "installer/license_texts/scotch/CeCILL-C_V1-en.txt": "CeCILL-C text (Latin-1)",
    "installer/license_texts/taucs/index.html": "the TAUCS licence page (Latin-1)",
}
BASE64 = re.compile(r"base64,[A-Za-z0-9+/=\s]{200,}")      # the images embedded in the generated HTML manuals (random letters would count as mentions)

# file -> (number of mentions, why they are justified). A NEW mention in a file listed here changes its count and fails the test just like a new file does.
ALLOWED: dict[str, tuple[int, str]] = {
    ".github/workflows/tests.yml": (7, "the CI step that checks that no Intel package or ccx_i4 is installed"),
    "fea_core/config.py": (11, "AUTO_SOLVER = \"PARDISO\": the stored word of \"automatic\" in studies of earlier versions; retired_ccx / ccx_env_note: a leftover "
                               "FEA_CCX naming the removed ccx_i4 build is skipped with a message"),
    "fea_core/direct/__init__.py": (1, "comment: the stored backend key ngs_pardiso loads as an alias"),
    "fea_core/i18n/cs.json": (12, "Czech labels of results of earlier versions (\"NGSolve + MKL PARDISO (earlier version)\"), the skip entries of the stored words and the "
                                  "message about a leftover FEA_CCX"),
    "fea_core/load_kinds.py": (2, "ngs_pardiso in ALL_BACKENDS: the stored name of ngs_direct"),
    "fea_core/mcp/tools_jobs.py": (1, "comment on the alias ngs_pardiso"),
    "fea_core/mcp/tools_results.py": (1, "comment on the alias ngs_pardiso"),
    "fea_core/modules.json": (4, "the record of the retired module pardiso"),
    "fea_core/printing/variants.py": (1, "numba's own threading layer called tbb"),
    "fea_core/report/word.py": (6, "label of a result of an earlier version (backend ngs_pardiso)"),
    "fea_core/results.py": (2, "the info key \"pardiso\" of results saved by an earlier version"),
    "fea_core/solvers.py": (11, "alias ngs_pardiso and the label of results of an earlier version; the note about a leftover FEA_CCX naming ccx_i4"),
    "fea_core/study.py": (3, "the stored word PARDISO and the alias ngs_pardiso of old study files"),
    "fea_gui/properties.py": (1, "comment: a result of an earlier version says ngs_pardiso"),
    "installer/CHANGELOG.md": (47, "history of the release notes"),
    "installer/README.md": (2, "says that nothing of Intel is shipped and that the module was removed"),
    "installer/FEAStudio.iss": (4, "the wizard preselects direct + accel when the installation state still names the retired module (as migrations.exact_retired_modules does)"),
    "installer/fea_setup.py": (3, "comment on the migration that drops the retired module; solver_available() skips a FEA_CCX naming ccx_i4"),
    "installer/migrations.py": (72, "the migration 1.4.2 that removes ccx_i4, the Intel packages and the module (and its housekeeping, and its replacement under --exact); the list of files of the old CalculiX zip names ccx_i4.exe"),
    "installer/license_texts/SOURCES.md": (1, "statement that the third-party ccx_static.exe contains no MKL"),
    "installer/license_texts/rslab/NOTICE": (4, "text of the rslab licence notice (it names reference solvers used in its benchmarks)"),
    "installer/linux/patches/ccx_2.23_pastix_upstream.patch": (1, "a context line of the CalculiX source (a comment of the author)"),
    "installer/windows/build_ccx.sh": (2, "the audit of the finished ccx_spooles.exe: its symbols must contain no MKL library ('mkl_', 'Intel(R) MKL')"),
    "tests/test_win_ccx.py": (2, "the same audit of the real binary (symbols of removed libraries) and of the build script"),
    "tests/test_install_solver_update.py": (13, "an offline update of an old installation: its ccx_i4 / mkl stay until the new solver and packages are in place"),
    "tests/gui/gui_direct_review_smoke.py": (7, "checks that a stored result of an earlier version is labelled by what ran"),
    "tests/gui/gui_solver_menu_smoke.py": (10, "checks that the solver menu and the Properties card never show the stored word of \"automatic\""),
    "tests/test_direct.py": (3, "the stored backend name ngs_pardiso loads as ngs_direct"),
    "tests/test_direct_review.py": (10, "a result of an earlier version is labelled by what ran, not as the direct solver"),
    "tests/test_modules.py": (13, "AUTO_SOLVER = \"PARDISO\" maps to PaStiX / SPOOLES; no Intel package in the lock"),
    "tests/test_security.py": (1, "a study file that stores the word PARDISO"),
    "tests/test_solver_rule.py": (3, "the ccx reference of the modal superposition no longer writes the stored word"),
    "tests/test_solvers.py": (4, "the alias ngs_pardiso; no MKL in the error text"),
    "tests/test_topo_feedback.py": (2, "the stored backend name ngs_pardiso is kept in the original study"),
    "tests/test_update.py": (39, "the retired module pardiso in the state of an old installation (also under --exact and in the wizard); the cleanup of leftovers (ccx_i4, mkl)"),
}
# generated documents that were made before the removal and still describe the removed solver: regenerated at the release 1.4.2 (19. 11. 2026, the docs module
# in a separate environment: `fea_gui.help_pdf` + `fea_gui.help_export`) - the manuals no longer name PARDISO / MKL, the list is empty; a manual generated
# later from an older checkout would put an entry here again (the scan above reports every mention)
STALE_DOCS: dict[str, tuple[int, str]] = {}
ALLOWED_PREFIX = {"installer/manifests/": "file lists of old releases: the update deletes the files they name that the new version no longer has (any count: a release adds one)"}


def _shipped() -> list[str]:
    """What a release package carries (`installer/make_package.py`): the versioned and the new unignored files without scratch, tools, development notes, PragTic
    and the private comparison – i.e. tests/, verification/, docs/ and the top level too, not only the code. Without git (a source export) the folder is walked
    with the same rule."""
    make_package = importlib.import_module("make_package")
    try:
        return make_package.file_list(None)
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", "tools", "scratch", ".claude", "dist", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
        out = []
        for dp, dns, fns in os.walk(ROOT):
            dns[:] = [d for d in dns if d not in skip]
            out += [(Path(dp) / fn).relative_to(ROOT).as_posix() for fn in fns]
        return sorted(rel for rel in out if make_package.packaged(rel))


def _text_of(rel: str) -> str | None:
    """The text of a shipped file for the scan; None = a known binary. Raises UnicodeDecodeError for a file that is neither."""
    p = ROOT / rel
    low = rel.lower()
    if low.endswith(BINARY_SUFFIX):
        return None
    if low.endswith(ZIP_TEXT_SUFFIX):
        with zipfile.ZipFile(p) as z:
            return "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith((".xml", ".rels")))
    data = p.read_bytes()
    return data.decode("latin-1" if rel in NON_UTF8_TEXT else "utf-8")


def _scan() -> tuple[dict[str, int], list[str]]:
    """({file: number of mentions}, files that are not binary and not UTF-8 text) over everything that is shipped, except this test."""
    found: dict[str, int] = {}
    undecodable: list[str] = []
    for rel in _shipped():
        if rel == SELF:
            continue
        try:
            text = _text_of(rel)
        except UnicodeDecodeError:
            undecodable.append(rel)
            continue
        except (OSError, zipfile.BadZipFile):
            undecodable.append(rel)
            continue
        if text is None:
            continue
        n = len(PATTERN.findall(BASE64.sub("base64,", text)))
        if n:
            found[rel] = n
    return found, undecodable


def test_every_shipped_file_is_text_or_a_known_binary():
    """A file that cannot be read as UTF-8 and is no known binary must not be skipped silently – it could carry the names unseen."""
    _, undecodable = _scan()
    assert not undecodable, f"shipped files that are neither UTF-8 text nor a known binary type (add to BINARY_SUFFIX / NON_UTF8_TEXT or fix the encoding): {undecodable}"


def test_the_product_names_nothing_of_intel_except_the_justified_places():
    found, _ = _scan()
    expected = {**{f: n for f, (n, _) in ALLOWED.items()}, **{f: n for f, (n, _) in STALE_DOCS.items()}}
    unknown = {f: n for f, n in sorted(found.items()) if f not in expected and not f.startswith(tuple(ALLOWED_PREFIX))}
    assert not unknown, f"new mention of MKL / PARDISO / ccx_i4 / Intel OpenMP / TBB in {unknown} (remove it, or justify it in ALLOWED with its count)"
    changed = {f: (found[f], n) for f, n in sorted(expected.items()) if f in found and found[f] != n}
    assert not changed, f"the number of mentions changed {{file: (now, pinned)}}: {changed} – a new mention must be justified, a removed one lowers the pin"
    stale = sorted(f for f in expected if f not in found)
    assert not stale, f"ALLOWED / STALE_DOCS entries without a mention (remove them): {stale}"
    assert not [f for f in ALLOWED if f in STALE_DOCS]
    assert all((ROOT / f).is_file() for f in STALE_DOCS)


def test_the_stale_documents_are_still_listed_for_the_release():
    """The generated manuals of the earlier version: the list is the release checklist (docs/projekt/F13_plan…) – regenerate them, then empty STALE_DOCS."""
    plan = (ROOT / "docs" / "projekt" / "F13_plan_speed_memory_licences_native.md")
    if not plan.is_file():                          # the plan is a development note, not part of a package
        pytest.skip("development notes are not part of this checkout")
    text = plan.read_text(encoding="utf-8")
    for f in STALE_DOCS:
        assert Path(f).stem in text, f"{f} is stale but not named in the release checklist of the F13 plan"

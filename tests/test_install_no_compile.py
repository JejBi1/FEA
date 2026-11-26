"""Nothing is compiled at installation (F13 step 3): the installer only unpacks and downloads. Package build refuses a Linux package
without the prebuilt CalculiX; PragTic data and the private Abaqus files never get into a package.
"""
from __future__ import annotations

import importlib
import inspect
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from elf_fake import build_archive

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

fs = importlib.import_module("fea_setup")
mp = importlib.import_module("make_package")
upgrade = importlib.import_module("upgrade")


# ------------------------------------------------------------------ the install path has no build step
def test_installer_has_no_compile_steps():
    """No native build and no numba warm-up. The ONLY byte-compile is `write_bytecode` of the program's own .py files (best effort,
    seconds): a read-only installation never gets the bytecode that Python would write at the first start (+1–2 s at every start)."""
    assert not hasattr(fs, "compile_project") and not hasattr(fs, "build_ccx_linux")
    assert list(inspect.signature(fs.check_install).parameters) == ["keys"]            # no "full" = no numba warm-up
    src = inspect.getsource(fs)
    assert [m.start() for m in re.finditer(r"compileall", src)] and src.count('"compileall"') == 1
    assert '"compileall"' in inspect.getsource(fs.write_bytecode)                      # nowhere else
    assert "check=False" in inspect.getsource(fs.write_bytecode)                       # a failure never stops the installation


def test_check_install_runs_only_the_quick_check(monkeypatch):
    seen = []
    monkeypatch.setattr(fs, "run", lambda cmd, **k: seen.append([str(c) for c in cmd]) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(fs, "info", lambda *_: None)
    fs.check_install(["core", "accel"])                    # accel = the module that used to trigger the 1–2 minute warm-up
    assert seen[-1][-3:] == ["-m", "fea_core.warmup", "--quick"], seen


def test_install_path_sources_name_no_compiler():
    """The files that run at installation (installer/fea_setup.py, install.sh, install.ps1, the .bat, the Inno script) never name a
    compiler or a native build tool (`compileall` only in fea_setup.write_bytecode – see above); `build_ccx.sh` is the release tool
    and the installer never starts it."""
    forbidden = re.compile(r"\b(gcc|g\+\+|gfortran|cmake|ninja|clang|nvcc|msbuild|vcvarsall|bdist_wheel)\b"
                           r"|(?<![\w])setup\.py\s+(build|install)|--no-binary|pip\s+wheel", re.I)
    for rel in ("installer/fea_setup.py", "install.sh", "installer/install.ps1", "Install FEA Studio.bat", "installer/FEAStudio.iss"):
        text = (ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            m = forbidden.search(line)
            assert not m, f"{rel}:{n} names a build tool ({m.group(0)}): {line.strip()}"
            if rel != "installer/fea_setup.py":
                assert "compileall" not in line, f"{rel}:{n}: the byte-compile step belongs to fea_setup.write_bytecode only"
    src = (ROOT / "installer" / "fea_setup.py").read_text(encoding="utf-8")
    runs = [l for l in src.splitlines() if re.search(r"\brun\(|subprocess\.(run|call|Popen)", l)]
    assert not [l for l in runs if "build_ccx" in l], "build_ccx.sh must not be started by the installer"
    assert '"bash"' not in "".join(runs), "no shell script is run by the installer"


def test_uv_pip_installs_wheels_only(monkeypatch):
    got = []
    monkeypatch.setattr(fs, "uv", lambda *a, **k: got.append(list(a)) or type("R", (), {"returncode": 0})())
    for sub in ("sync", "install", "compile"):
        fs.uv_pip(sub, "x.txt")
    assert all("--no-build" in c for c in got) and len(got) == 3
    src = inspect.getsource(fs)
    assert re.findall(r'uv\("pip"', src) and all("NO_BUILD" in l for l in src.splitlines() if 'uv("pip"' in l)


# ------------------------------------------------------------------ Linux CalculiX: unpacked from the package, never built
@pytest.fixture
def linux(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "IS_WIN", False)
    monkeypatch.setattr(fs, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    (tmp_path / "tools" / "calculix").mkdir(parents=True)
    monkeypatch.setattr(fs, "run", lambda *a, **k: pytest.fail("the installer must not run any build"))
    monkeypatch.setattr(fs.subprocess, "run", lambda *a, **k: pytest.fail("the installer must not run any build"))
    monkeypatch.setattr(fs, "ok", lambda *_: None)
    monkeypatch.setattr(fs, "info", lambda *_: None)
    return tmp_path


def _prebuilt(path: Path) -> None:
    """An archive like the output of build_ccx.sh for the package: ccx_spooles + the libraries it needs (synthetic ELF files – the package build
    reads DT_NEEDED, tests/elf_fake.py)."""
    build_archive(path, pastix=False)


def test_linux_ccx_comes_from_the_package(linux):
    _prebuilt(linux / "tools" / "calculix" / "ccx_2.23_linux_x86_64.tar.gz")
    fs.install_ext_ccx("ccx_static")
    assert (linux / "tools" / "calculix" / "ccx_2.23_linux" / "ccx_spooles").is_file()
    assert (linux / "tools" / "calculix" / "ccx_2.23_linux" / "lib" / "libgomp.so.1").is_file()
    fs.install_ext_ccx("ccx_static")                                       # repeated install: nothing to do


def test_linux_without_the_prebuilt_ccx_is_an_error_not_a_build(linux):
    with pytest.raises(RuntimeError) as e:
        fs.install_ext_ccx("ccx_static")
    msg = str(e.value)
    assert "ccx_2.23_linux_x86_64.tar.gz" in msg and "FEA_CCX" in msg
    assert "nepřekládá" in msg or "not compiled" in msg
    # the way out for a developer in a source checkout: the script and the place where the package builder expects the archive
    assert "installer/linux/build_ccx.sh" in msg and "installer/linux/prebuilt/ccx_2.23_linux_x86_64.tar.gz" in msg


def test_linux_ccx_comes_from_the_checkout_prebuilt_folder(linux):
    """`make_package.py` reads installer/linux/prebuilt/, build_ccx.sh writes there: a source checkout installs from the same place."""
    folder = linux / "installer" / "linux" / "prebuilt"
    folder.mkdir(parents=True)
    _prebuilt(folder / "ccx_2.23_linux_x86_64.tar.gz")
    assert fs.prebuilt_ccx_candidates()[-1] == folder / "ccx_2.23_linux_x86_64.tar.gz"
    fs.install_ext_ccx("ccx_static")
    assert (linux / "tools" / "calculix" / "ccx_2.23_linux" / "ccx_spooles").is_file()
    assert (linux / "tools" / "calculix" / "ccx_2.23_linux" / "ccx_spooles").stat().st_mode & 0o111 or sys.platform == "win32"


def test_ci_builds_the_prebuilt_ccx_on_a_cache_miss():
    """The Linux jobs of the CI workflow get the archive that the installer unpacks: restored from the cache or built by
    build_ccx.sh in an explicit step (the installer itself never builds)."""
    path = ROOT / ".github" / "workflows" / "tests.yml"
    if not path.is_file():
        pytest.skip("no CI workflow in this tree")
    yml = path.read_text(encoding="utf-8")
    assert "bash installer/linux/build_ccx.sh" in yml and "installer/linux/prebuilt/ccx_2.23_linux_x86_64.tar.gz" in yml
    assert "steps.ccx-cache.outputs.cache-hit != 'true'" in yml and "id: ccx-cache" in yml
    assert yml.index("bash installer/linux/build_ccx.sh") < yml.index("installer/fea_setup.py install")   # built / restored BEFORE the install
    assert "built from source by installer/linux/build_ccx.sh when no prebuilt archive exists" not in yml  # the old, wrong comment
    key = next(line for line in yml.splitlines() if "key: ccx-linux-archive-" in line)
    for dep in ("build_ccx.sh", "ccx_toolchain.lock", "ccx_deps.lock", "patches/*.patch"):       # everything the archive depends on
        assert dep in key, f"the cache key of the prebuilt CalculiX does not cover {dep}"


# ------------------------------------------------------------------ package build: the Linux CalculiX is mandatory
def test_package_build_requires_the_prebuilt_linux_ccx(tmp_path):
    with pytest.raises(SystemExit) as e:
        mp.require_linux_ccx(tmp_path / "ccx_2.23_linux_x86_64.tar.gz")
    assert "build_ccx.sh" in str(e.value) and "--no-linux-ccx" in str(e.value)
    ok = tmp_path / "pre.tar.gz"
    _prebuilt(ok)
    assert mp.require_linux_ccx(ok) == ok
    bad = tmp_path / "bad.tar.gz"                                           # an archive without ccx_spooles is unusable too
    with tarfile.open(bad, "w:gz") as t:
        ti = tarfile.TarInfo("lib/x")
        ti.size = 1
        t.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(SystemExit, match="ccx_spooles is missing"):
        mp.require_linux_ccx(bad)


def test_make_package_stops_before_staging_without_the_prebuilt_ccx(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mp, "LINUX_CCX", tmp_path / "missing.tar.gz")
    monkeypatch.setattr(mp, "dirty_files", lambda: [])
    monkeypatch.setattr(mp, "file_list", lambda ref: pytest.fail("nothing must be staged"))
    monkeypatch.setattr(sys, "argv", ["make_package.py", "--out", str(tmp_path / "dist")])
    with pytest.raises(SystemExit) as e:
        mp.main()
    assert "The prebuilt CalculiX for Linux is missing" in str(e.value)
    assert not (tmp_path / "dist").exists()


def test_check_only_and_the_opt_out_do_not_need_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "LINUX_CCX", tmp_path / "missing.tar.gz")
    monkeypatch.setattr(mp, "dirty_files", lambda: [])
    staged = []
    monkeypatch.setattr(mp, "file_list", lambda ref: staged.append(ref) or (_ for _ in ()).throw(KeyboardInterrupt()))
    for extra in (["--check-only"], ["--no-linux-ccx", "--no-windows-ccx", "--out", str(tmp_path / "dist")]):   # (the Windows archive is not
                                                                     # on a CI runner / a fresh checkout either: this test is about the Linux one)
        monkeypatch.setattr(sys, "argv", ["make_package.py", *extra])
        with pytest.raises(KeyboardInterrupt):                               # got past the check: staging started
            mp.main()
    assert len(staged) == 2


# ------------------------------------------------------------------ L5 / L6: what must never be in a package
PRAGTIC = "fea_core/fatigue/data/pragtic_vmat.json"


def test_pragtic_data_are_not_packaged_and_survive_an_update(monkeypatch):
    assert mp.PRAGTIC == PRAGTIC
    assert not mp.packaged(PRAGTIC)
    assert PRAGTIC in mp.EXCLUDE_FILES
    assert upgrade.is_protected(PRAGTIC)                                    # an update never deletes the user's copy
    listed = mp.file_list(None)
    assert PRAGTIC not in listed
    leaked = [f for f in listed if "pragtic_vmat" in f.lower()]
    assert not leaked, leaked
    # the exclusion does real work when git lists the file (tracked in the repository): without it the file WOULD be packaged
    raw = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", PRAGTIC], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    if PRAGTIC in raw and (ROOT / PRAGTIC).is_file():
        monkeypatch.setattr(mp, "EXCLUDE_FILES", set(mp.EXCLUDE_FILES) - {PRAGTIC})
        assert PRAGTIC in mp.file_list(None), "the PragTic file is only kept out by EXCLUDE_FILES"


def test_no_other_data_file_carries_the_pragtic_records():
    """A copy of the data under another name or format would get past the name check: no packaged data file (json / csv / npz / txt)
    mentions the database."""
    bad = []
    for f in mp.file_list(None):
        if f.lower().endswith((".json", ".csv", ".npz", ".txt", ".dat")) and not f.startswith(("installer/license_texts/", "installer/manifests/")):
            head = (ROOT / f).read_bytes()[:200_000].lower()
            if b"pragtic" in head and f not in ("fea_core/i18n/cs.json",):
                bad.append(f)
    assert not bad, bad


def test_with_pragtic_is_the_only_way_in_and_it_is_private(monkeypatch):
    monkeypatch.setattr(mp, "EXCLUDE_FILES", set(mp.EXCLUDE_FILES))
    assert PRAGTIC in mp.EXCLUDE_FILES
    src = inspect.getsource(mp.main)
    assert "--with-pragtic" in src and "PRIVATE build" in src


def test_private_abaqus_files_stay_out_of_every_listing():
    listed = set(mp.file_list(None))
    assert not [f for f in listed if f.startswith(mp.PRIVATE_PREFIX) or f in mp.PRIVATE_FILES]
    for name in ("verification/abaqus_check.py", "verification/out_abaqus/report.json", "tests/test_abaqus_all.py"):
        assert not mp.packaged(name)
    # the exclusion is by name, so the same must hold for a worktree listing that includes new (untracked) files
    assert not mp.packaged("verification/abaqus_new_future_module.py")

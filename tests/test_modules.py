"""Program modules, the installer and the platform layer (without network, without Gmsh and CalculiX)."""
from __future__ import annotations

import importlib
import re
import sys
import tomllib
import types
from pathlib import Path

import pytest

from fea_core import config, modules

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(lines):
    out = {}
    for l in lines:
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", l.strip())
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


# ------------------------------------------------------------------ module definitions
def test_modules_json_consistent():
    mods = modules.all_modules()
    assert "core" in mods and mods["core"].required
    for k, m in mods.items():
        assert m.title("cs") and m.title("en") and m.desc("cs") and m.desc("en"), k
        for r in m.requires:
            assert r in mods, f"{k} vyžaduje neznámý modul {r}"
        assert m.size() > 0, k
    for name, keys in modules.profiles().items():
        assert set(keys) <= set(mods), name
        assert modules.closure(keys)[0] == "core"


def test_closure_order_and_unknown():
    c = modules.closure(["hypre"])
    assert c.index("core") < c.index("accel") < c.index("hypre")      # the own matrix assembly needs numba
    assert "ngsolve" not in c                           # the iterative solver does not need NGSolve
    d = modules.closure(["direct"])
    assert d.index("core") < d.index("accel") < d.index("direct") and "ngsolve" not in d and "hypre" not in d
    assert "motion" in modules.closure(["ngsolve"]) and "direct" not in modules.closure(["ngsolve"])    # NGSolve = flexible bodies of mechanisms
    with pytest.raises(KeyError):
        modules.closure(["neexistuje"])


@pytest.mark.parametrize("plat", ["windows-x86_64", "linux-x86_64"])
def test_locks_cover_all_pins(plat):
    """Every package from modules.json has the same version in the lock of the platform (otherwise the installer would install other versions)."""
    lock = _pins((ROOT / "installer" / "locks" / f"{plat}.txt").read_text(encoding="utf-8").splitlines())
    assert not set(lock) & {"mkl", "intel-openmp", "tbb"}, "nothing of Intel in the lock (tests/test_no_mkl.py)"
    for k, m in modules.all_modules().items():
        for k2, v in _pins(list(m.pip) + list(m.pip_nodeps)).items():
            assert lock.get(k2) == v, f"{plat}: {k2}=={v} (modul {k}) nesedí se zámkem ({lock.get(k2)})"


def test_pyproject_extras_match_modules():
    d = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = d["project"]["optional-dependencies"]
    for k, m in modules.all_modules().items():
        if k == "core":
            continue
        assert _pins(extras[k]) == _pins(list(m.pip) + list(m.pip_nodeps)), k


def test_imports_of_core_are_light():
    """The core must be importable without numba, matplotlib, PySide6, mcp, docx (optional modules)."""
    import subprocess
    code = ("import sys\n"
            "for n in ('numba', 'PySide6', 'mcp', 'docx', 'fastapi', 'matplotlib'):\n"
            "    sys.modules[n] = None\n"      # the import fails → ImportError
            "import fea_core.solve, fea_core.cli, fea_core.study, fea_core.thermal.run, fea_core.shell.run\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_need_raises_module_missing():
    with pytest.raises(modules.ModuleMissing) as e:
        modules.need("fea_neexistujici_balicek", "accel")
    assert e.value.key == "accel" and "accel" in str(e.value)
    assert isinstance(e.value, ImportError)


def test_numba_users_raise_friendly_error(monkeypatch):
    """Without numba the fatigue / print / rasteriser report ModuleMissing (not a bare ImportError without instructions)."""
    import subprocess
    code = ("import sys\nsys.modules['numba'] = None\n"
            "import fea_core.modules as m\n"
            "try:\n    import fea_core.fatigue.kernel\nexcept m.ModuleMissing as e:\n    print('MISSING', e.key)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert "MISSING accel" in r.stdout, r.stdout + r.stderr[-500:]


def test_status_core_available():
    assert modules.available("core")
    st = modules.status()
    assert set(st) == set(modules.all_modules())


# ------------------------------------------------------------------ platform layer
def test_solver_name(monkeypatch):
    """"PARDISO" is only the stored word of "automatic" (config.AUTO_SOLVER): PaStiX where ccx has it, otherwise SPOOLES."""
    assert config.AUTO_SOLVER == "PARDISO"
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_static.exe"))
    assert config.solver_name("PARDISO") == "PASTIX" == config.solver_name(config.AUTO_SOLVER)
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles"))
    assert config.solver_name("PARDISO") == "SPOOLES"
    assert config.solver_name("SPOOLES") == "SPOOLES"        # the explicit choice does not change
    assert config.solver_name("nonsense from a foreign file") == "SPOOLES"


def test_solver_name_by_procedure(monkeypatch):
    """The automatic solver is PaStiX except for the procedures where it is slower than SPOOLES (config.SPOOLES_PROCEDURES)."""
    for exe in ("ccx_static.exe", "ccx_pastix"):
        monkeypatch.setattr(config, "CCX_EXE", Path("x") / exe)
        assert config.ccx_solvers()["PASTIX"] is True
        for proc in ("static", "buckle", "heat"):
            assert config.solver_name("PARDISO", proc) == "PASTIX", (exe, proc)
        for proc in ("modal", "shell", "twod", "beam", "mortar"):
            assert config.solver_name("PARDISO", proc) == "SPOOLES", (exe, proc)
        assert config.solver_name("PASTIX", "modal") == "PASTIX"            # an explicit choice is not overridden by the rule
        assert config.solver_name("SPOOLES", "static") == "SPOOLES"
        assert config.solver_name("PARDISO") == "PASTIX"                      # default procedure = static
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles"))               # Linux fallback binary: no PaStiX
    assert config.ccx_solvers()["PASTIX"] is False
    assert {config.solver_name("PARDISO", p) for p in ("static", "buckle", "modal", "shell")} == {"SPOOLES"}
    assert config.solver_name("PASTIX", "static") == "SPOOLES"


def test_mortar_supported(monkeypatch):
    monkeypatch.delenv("FEA_MORTAR", raising=False)
    monkeypatch.setattr(config, "IS_WIN", True)
    for exe in ("ccx_static.exe", "ccx_spooles.exe"):                           # the retired third-party build and the own build
        monkeypatch.setattr(config, "CCX_EXE", Path("x") / exe)
        assert config.mortar_supported() is False                                # the NaN of the mortar matrix on Windows (K2, also in the own build)
    monkeypatch.setattr(config, "IS_WIN", False)
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_pastix"))
    assert config.mortar_supported() is False                                    # Linux too: K2 "singular" with 1 and 2 equation threads (27. 11. 2026)
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles"))
    assert config.mortar_supported() is False
    monkeypatch.setattr(config, "IS_WIN", True)
    monkeypatch.setenv("FEA_MORTAR", "1")
    assert config.mortar_supported() is True                                     # a ccx of your own can say so
    monkeypatch.setenv("FEA_MORTAR", "0")
    assert config.mortar_supported() is False


def test_spooles_runs_with_one_equation_solver_thread_unless_the_binary_carries_the_checked_factorization(tmp_path, monkeypatch):
    """The multi-threaded solve of an unpatched SPOOLES gives wrong results without any message (data race: B1 buckling factor 1.67 instead of 21.65
    with ccx_static.exe; the unpatched own build: natural frequencies and buckling wrong with 3, 4 and 8 threads, different in every run) → one thread –
    unless the binary says (marker FEA_SPOOLES_FACTOR_CHECK=2 of `ccx -v`) that its multi-threaded factorization is CHECKED (and its solve serial, in the order of the
    sums that does not depend on the timing; patches ccx_2.23_spooles_serial_solve.patch and spooles_2.2_ordered_assembly.patch): then the factorization may use threads.
    The marker of the serial solve alone (FEA_SPOOLES_SERIAL_SOLVE=1, the first candidate without the check) is not enough."""
    from fea_core.ccx.runner import FACTOR_CHECK_MARKER, ORDERED_MARKER, SERIAL_SOLVE_MARKER, one_equation_thread, spooles_threads_safe, uses_spooles
    monkeypatch.delenv("FEA_SPOOLES_THREADS", raising=False)
    (tmp_path / "a.inp").write_text("*STEP\n*STATIC, SOLVER=SPOOLES\n*END STEP\n")
    (tmp_path / "b.inp").write_text("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")
    (tmp_path / "c.inp").write_text("*STEP\n*STATIC\n*END STEP\n")                                 # no SOLVER=: ccx takes SPOOLES (the own build has no other)
    (tmp_path / "m.inp").write_text("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n*STEP\n*FREQUENCY, SOLVER=SPOOLES\n*END STEP\n")   # prestressed modal job
    assert uses_spooles(tmp_path / "a.inp") and not uses_spooles(tmp_path / "b.inp")
    assert not uses_spooles(tmp_path / "missing.inp")
    (tmp_path / "empty.inp").write_text("")
    assert not uses_spooles(tmp_path / "empty.inp")
    plain, marked = tmp_path / "ccx_plain.exe", tmp_path / "ccx_marked.exe"
    plain.write_bytes(b"MZ" + b"CalculiX 2.23" * 100)
    marked.write_bytes(b"MZ" + b"This is Version 2.23\n" + FACTOR_CHECK_MARKER + b" (multi-threaded SPOOLES factorization)\n" + ORDERED_MARKER + b"\n")
    serial_only, check_only = tmp_path / "ccx_serial_only.exe", tmp_path / "ccx_check_only.exe"
    serial_only.write_bytes(b"MZ" + b"This is Version 2.23\n" + SERIAL_SOLVE_MARKER + b" (multi-threaded SPOOLES factorization, serial triangular solve)\n")
    check_only.write_bytes(b"MZ" + b"This is Version 2.23\n" + SERIAL_SOLVE_MARKER + b"\n" + FACTOR_CHECK_MARKER + b"\n")      # the check without the ordered SPOOLES (an earlier candidate)
    assert not spooles_threads_safe(plain) and spooles_threads_safe(marked) and not spooles_threads_safe(serial_only) and not spooles_threads_safe(check_only)
    for exe in (serial_only, check_only):                                                                                       # a marker alone keeps ONE thread
        assert one_equation_thread(tmp_path / "a.inp", exe) and one_equation_thread(tmp_path / "c.inp", exe)
    assert not spooles_threads_safe(None) and not spooles_threads_safe(tmp_path / "nowhere.exe")        # an unknown binary: one thread
    for exe in (None, plain, tmp_path / "nowhere.exe"):                                                  # without the marker: one thread, as before
        assert one_equation_thread(tmp_path / "a.inp", exe) and one_equation_thread(tmp_path / "c.inp", exe) and one_equation_thread(tmp_path / "m.inp", exe)
        assert one_equation_thread(tmp_path / "empty.inp", exe) and one_equation_thread(tmp_path / "missing.inp", exe)
        assert not one_equation_thread(tmp_path / "b.inp", exe)                  # only a pure PaStiX deck (a developer's ccx_pastix) keeps its threads
    for name in ("a", "c", "m", "empty", "missing"):                             # the marked binary: SPOOLES decks may use threads
        assert not one_equation_thread(tmp_path / f"{name}.inp", marked), name
    monkeypatch.setenv("FEA_SPOOLES_THREADS", "1")                               # the switch back to the single-thread rule
    assert one_equation_thread(tmp_path / "a.inp", marked) and not one_equation_thread(tmp_path / "b.inp", marked)
    monkeypatch.delenv("FEA_SPOOLES_THREADS")
    marked.write_bytes(b"MZ" + b"nothing at all")                                # a replaced file is looked at again (the cache is keyed by time and size)
    assert not spooles_threads_safe(marked)


def test_run_ccx_sets_the_equation_solver_threads_by_the_binary(tmp_path, monkeypatch):
    """The variable that limits SPOOLES is set for every input: 1 for a binary without the serial-solve marker (the environment of the user cannot bring
    the racy parallel solve back), the threads of the job for a binary that has it (never above FEA_SPOOLES_THREADS), nothing for a PaStiX deck."""
    import subprocess
    from fea_core.ccx import runner
    seen = {}

    class Out:
        def __iter__(self):
            return iter(["Job finished\n"])

        def close(self):
            pass

    class P:
        stdout = Out()
        returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    def popen(cmd, **kw):
        seen["env"] = kw["env"]
        return P()
    exe = tmp_path / "ccx_spooles.exe"
    exe.write_text("x")
    marked = tmp_path / "ccx_marked.exe"
    marked.write_bytes(b"x" + runner.FACTOR_CHECK_MARKER + runner.ORDERED_MARKER)
    (tmp_path / "job.inp").write_text("*STEP\n*STATIC\n*END STEP\n")
    (tmp_path / "px.inp").write_text("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.delenv("FEA_SPOOLES_THREADS", raising=False)
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "8")                       # a value from the user's environment is replaced
    monkeypatch.setattr(runner.dllguard, "check_workdir", lambda *a, **k: None)
    runner.run_ccx(tmp_path, "job", exe=exe, threads=4)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "1" and seen["env"]["OMP_NUM_THREADS"] == "4"
    runner.run_ccx(tmp_path, "job", exe=marked, threads=4)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "4" and seen["env"]["OMP_NUM_THREADS"] == "4"
    runner.run_ccx(tmp_path, "job", exe=marked, threads=1)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "1"
    monkeypatch.setenv("FEA_SPOOLES_THREADS", "2")
    runner.run_ccx(tmp_path, "job", exe=marked, threads=8)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "2" and seen["env"]["OMP_NUM_THREADS"] == "8"      # the cap limits the solver, not the assembly
    monkeypatch.setenv("FEA_SPOOLES_THREADS", "1")
    runner.run_ccx(tmp_path, "job", exe=marked, threads=8)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "1"
    monkeypatch.delenv("FEA_SPOOLES_THREADS")
    runner.run_ccx(tmp_path, "px", exe=marked, threads=4)
    assert "CCX_NPROC_EQUATION_SOLVER" not in seen["env"]



def test_equation_threads_survive_odd_environment_values_and_mixed_decks(tmp_path, monkeypatch):
    """OMP_NUM_THREADS of the user may be an OpenMP list or junk (`config.ccx_env` keeps it): the equation threads of a marked binary follow its first number or the
    default of the program, never an exception. The cap FEA_SPOOLES_THREADS holds for a mixed deck (PaStiX + SPOOLES step) too; 0 reads as 1, a negative or
    non-numeric value is not set. An empty or unreadable binary has no marker: one thread."""
    import subprocess
    from fea_core import config
    from fea_core.ccx import runner
    seen = {}

    class Out:
        def __iter__(self):
            return iter(["Job finished\n"])

        def close(self):
            pass

    class P:
        stdout = Out()
        returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    def popen(cmd, **kw):
        seen["env"] = kw["env"]
        return P()
    marked = tmp_path / "ccx_marked.exe"
    marked.write_bytes(b"x" + runner.FACTOR_CHECK_MARKER + runner.ORDERED_MARKER)
    empty = tmp_path / "ccx_empty.exe"
    empty.write_bytes(b"")
    (tmp_path / "job.inp").write_text("*STEP\n*STATIC\n*END STEP\n")
    (tmp_path / "mixed.inp").write_text("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n*STEP\n*FREQUENCY, SOLVER=SPOOLES\n*END STEP\n")
    (tmp_path / "px.inp").write_text("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")
    monkeypatch.delenv("FEA_THREADS", raising=False)               # with FEA_THREADS the program sets OMP_NUM_THREADS itself (test_threads_cap.py)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(runner.dllguard, "check_workdir", lambda *a, **k: None)
    monkeypatch.setattr(config, "NUM_THREADS", 5)
    monkeypatch.delenv("FEA_SPOOLES_THREADS", raising=False)

    def run(job, exe, **kw):
        runner.run_ccx(tmp_path, job, exe=exe, **kw)
        return seen["env"].get("CCX_NPROC_EQUATION_SOLVER")
    for odd, expect in (("4,2", "4"), (" 3 ", "3"), ("junk", "5"), ("", "5"), ("0", "5"), ("-2", "5"), ("2.5", "5")):
        monkeypatch.setenv("OMP_NUM_THREADS", odd)
        assert run("job", marked) == expect, odd                       # threads=None: the value of the user's environment
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert run("job", marked) == "5"                                    # no value at all: the default of the program
    assert run("job", marked, threads=3) == "3"                         # the threads of the job win over the environment
    assert run("mixed", marked, threads=6) == "6"                       # a mixed deck has a SPOOLES step: the marked build threads it
    assert run("px", marked, threads=6) is None                         # only a pure PaStiX deck is left alone
    monkeypatch.setenv("FEA_SPOOLES_THREADS", "2")
    assert run("job", marked, threads=6) == "2" and run("mixed", marked, threads=6) == "2"
    assert run("job", marked, threads=1) == "1"
    for zero in ("0", "1"):                                             # 0 = the safe reading of "no threads"
        monkeypatch.setenv("FEA_SPOOLES_THREADS", zero)
        assert run("job", marked, threads=6) == "1" and run("mixed", marked, threads=6) == "1", zero
    for bad in ("-1", "abc", "", "2.5"):                                # not a usable cap: not set
        monkeypatch.setenv("FEA_SPOOLES_THREADS", bad)
        assert run("job", marked, threads=6) == "6" and run("mixed", marked, threads=6) == "6", bad
    monkeypatch.delenv("FEA_SPOOLES_THREADS")
    assert not runner.spooles_threads_safe(empty)                       # a zero-byte file cannot be mapped: no marker
    assert run("job", empty, threads=6) == "1" and run("mixed", empty, threads=6) == "1"


def test_find_tool_depth(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "ccx_static").write_text("x")
    assert config._find_tool(("ccx_static",), tmp_path) == tmp_path / "a" / "ccx_static"
    assert config._find_tool(("nic",), tmp_path) is None
    assert config._find_tool(("x",), tmp_path / "neexistuje") is None


def test_find_ccx_env_override(monkeypatch):
    monkeypatch.setenv("FEA_CCX", "/opt/ccx_2.23")
    assert config.find_ccx() == Path("/opt/ccx_2.23")


def test_user_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IS_WIN", False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert config.user_config_dir() == tmp_path / "cfg" / "FEAStudio"
    assert config.user_data_dir() == tmp_path / "data" / "FEAStudio"
    monkeypatch.setattr(config, "IS_WIN", True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert config.user_config_dir() == tmp_path / "roaming" / "FEAStudio"
    assert config.user_data_dir() == tmp_path / "local" / "FEAStudio"


def test_threads_env(monkeypatch):
    monkeypatch.setenv("FEA_THREADS", "3")
    assert config._physical_cores() == 3
    monkeypatch.delenv("FEA_THREADS")
    assert 1 <= config._physical_cores() <= (config.os.cpu_count() or 1)


def test_child_env_adds_sys_libs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IS_WIN", False)
    sysl = tmp_path / "lib"
    sysl.mkdir()
    monkeypatch.setattr(config, "SYS_LIBS", sysl)
    e = config.child_env({"LD_LIBRARY_PATH": "/x"})
    assert e["LD_LIBRARY_PATH"].split(config.os.pathsep) == [str(sysl), "/x"]


def test_mcp_clients_linux_paths(monkeypatch, tmp_path):
    from fea_core.mcp import setup
    monkeypatch.setattr(setup, "os", types.SimpleNamespace(name="posix", environ={}))   # not os.name globally (it would break pathlib)
    cl = {c.key: c for c in setup.clients({"HOME": str(tmp_path)})}
    assert cl["vscode"].path == tmp_path / ".config" / "Code" / "User" / "mcp.json"
    assert cl["cursor"].path == tmp_path / ".cursor" / "mcp.json"
    assert "claude-desktop-store" not in cl


# ------------------------------------------------------------------ installer and pruning
def test_slim_prune_python_refuses_foreign(tmp_path):
    import slim
    venv = tmp_path / ".venv"
    venv.mkdir()
    other = tmp_path / "system_python"
    (other / "Lib" / "test").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {other}\n", encoding="utf-8")
    assert slim.prune_python(venv, say=lambda *_: None) == 0
    assert (other / "Lib" / "test").exists()                 # a foreign Python is never deleted
    ours = tmp_path / "tools" / "python" / "cpython-3.11-x" / "bin"
    (ours.parent / "lib" / "python3.11" / "test").mkdir(parents=True)
    (ours.parent / "lib" / "python3.11" / "os.py").write_text("x")
    ours.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {ours}\n", encoding="utf-8")
    slim.prune_python(venv, say=lambda *_: None)
    assert not (ours.parent / "lib" / "python3.11" / "test").exists()
    assert (ours.parent / "lib" / "python3.11" / "os.py").exists()


def test_fea_setup_profiles_and_quiet_flags(monkeypatch):
    fs = importlib.import_module("fea_setup")
    ns = type("A", (), {"modules": [], "profile": "standard"})()
    keys = fs.pick_modules(ns)
    assert keys[0] == "core" and "gui" in keys and "direct" in keys and "accel" in keys and "mcp" not in keys
    assert "pardiso" not in keys
    assert "ngsolve" not in keys                      # the standard profile has the fast linear path without NGSolve
    ns2 = type("A", (), {"modules": ["hypre"], "profile": None})()
    assert {"core", "accel", "hypre"} <= set(fs.pick_modules(ns2))


def test_make_package_file_list_excludes():
    mp = importlib.import_module("make_package")
    files = mp.file_list(None)
    assert "fea_core/modules.json" in files and "installer/fea_setup.py" in files and "install.sh" in files
    assert not any(f.startswith(("scratch/", "tools/", ".venv/", "dist/")) for f in files)
    assert not any(f.endswith(".fea") for f in files)
    assert mp.version().count(".") == 2


def test_make_package_linux_ccx_neutral(tmp_path):
    """The bundled CalculiX for Linux: the owner of the building user is repacked to 0:0 without names, a path from a home folder = an error."""
    import io
    import tarfile
    from elf_fake import fake_elf
    mp = importlib.import_module("make_package")

    def archive(path, payload):
        with tarfile.open(path, "w:gz") as t:
            for name, data, mode in (("ccx_spooles", fake_elf(["libgomp.so.1"], payload), 0o755),
                                     ("lib/libgomp.so.1", fake_elf([], b"/home/conda/x/"), 0o755)):
                ti = tarfile.TarInfo(name)
                ti.size, ti.mode, ti.mtime = len(data), mode, 1_700_000_000
                ti.uid = ti.gid = 1000
                ti.uname = ti.gname = "builder"
                t.addfile(ti, io.BytesIO(data))

    archive(tmp_path / "ok.tar.gz", b"\x7fELF ... /build/arpack-src/UTIL/dvout.f")
    mp.linux_ccx(tmp_path / "ok.tar.gz", tmp_path / "out.tar.gz")
    with tarfile.open(tmp_path / "out.tar.gz") as t:
        ms = t.getmembers()
        assert [m.name for m in ms] == ["ccx_spooles", "lib/libgomp.so.1"]
        assert all((m.uid, m.gid, m.uname, m.gname) == (0, 0, "", "") for m in ms)
        assert all(m.mode == 0o755 and m.mtime == 1_700_000_000 for m in ms)
        assert t.extractfile("ccx_spooles").read().endswith(b"dvout.f")
    assert (tmp_path / "out.tar.gz").read_bytes()[3] == 0              # gzip without the file name
    archive(tmp_path / "bad.tar.gz", b"\x7fELF ... /home/builder/fea/ccx_build/arpack-src/UTIL/dvout.f")
    with pytest.raises(SystemExit, match="home folder"):
        mp.linux_ccx(tmp_path / "bad.tar.gz", tmp_path / "out2.tar.gz")


def test_verification_imports_without_numba():
    """The check calculation of the installation (fea_core.warmup → verification.run_all) must also run without the accel module (Numba)."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['numba'] = None\n"
            "import verification.run_all as r, verification.cases as C\n"
            "assert 'V1' in C.CASES\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert p.returncode == 0, p.stderr[-2000:]

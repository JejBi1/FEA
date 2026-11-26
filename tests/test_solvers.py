"""Availability of the solvers (fea_core.solvers): standard installation without hypre, without the direct solver library or numba, hypre without MS-MPI.
Without Qt, without a calculation – the components are simulated (find_spec, the HYPRE library, mpiexec, ccx).
"""
from pathlib import Path

import pytest

from fea_core import config, solvers
from fea_core.selectors import FaceSelector
from fea_core.study import Fixed, Force, LoadCase, Material, Study


def _install(monkeypatch, tmp_path, *, ngsolve=True, mpi4py=True, hypre_lib=True, mpiexec=True,
             ccx=True, ccx_name="ccx_static.exe", rslab=True, numba=True):
    """Simulated installation: Python modules (find_spec), the HYPRE library, mpiexec (PATH and the MS-MPI folder), ccx."""
    mods = {"ngsolve": ngsolve, "mpi4py": mpi4py, "rslab": rslab, "numba": numba}
    monkeypatch.delenv("FEA_ASSEMBLY", raising=False)
    monkeypatch.setattr(solvers, "_has", lambda name: mods.get(name, True))
    hb = tmp_path / "hypre_bin"
    hb.mkdir(parents=True)
    if hypre_lib:
        (hb / config.HYPRE_LIB_NAME).write_text("x")
    monkeypatch.setattr(config, "HYPRE_BIN", hb)
    monkeypatch.setattr(config, "MPIEXEC_DIR", tmp_path / "hypre_env_bin")       # own mpiexec of the environment: none
    mpi = tmp_path / "mpi" / ("mpiexec" + config.EXE)
    if mpiexec:
        mpi.parent.mkdir()
        mpi.write_text("x")
    monkeypatch.setattr(solvers.config, "which", lambda name: str(mpi) if mpiexec and name == "mpiexec" else None)
    monkeypatch.setenv("MSMPI_BIN", str(tmp_path / "msmpi_neni"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    exe = tmp_path / "ccx" / ccx_name
    if ccx:
        exe.parent.mkdir()
        exe.write_text("x")
    monkeypatch.setattr(config, "CCX_EXE", exe)


def _study(backend="auto", n_cases=1) -> Study:
    st = Study("t", "m.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9))
    st.add_selection("dno", [FaceSelector("Dil", "plane", 1.0, [0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, 0, 1, 1, 0])])
    st.add_selection("vrch", [FaceSelector("Dil", "plane", 1.0, [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1, 1, 1, 1])])
    st.constraints.append(Fixed("dno"))
    st.load_cases = [LoadCase(f"LC{i + 1}", [Force("vrch", [0, 0, -100.0])]) for i in range(n_cases)]
    st.solver.backend = backend
    return st


# ------------------------------------------------------------------ state of the backends
def test_standard_install_without_hypre_and_the_direct_solver(monkeypatch, tmp_path):
    """Minimal profile (without the modules accel, direct and hypre): only CalculiX is available."""
    _install(monkeypatch, tmp_path, ngsolve=False, mpi4py=False, hypre_lib=False, mpiexec=False, rslab=False, numba=False)
    st = solvers.backend_status()
    assert st["ccx"]["ok"] and st["ccx"]["missing"] == []
    assert not st["ngs_direct"]["ok"]
    assert st["ngs_direct"]["missing"] == ["numba (module accel)", "the direct solver library rslab (module direct)"]
    assert not st["ngs_hypre"]["ok"]
    assert st["ngs_hypre"]["missing"][:2] == ["numba (module accel)", "module hypre"]
    err = solvers.backend_error("ngs_direct")
    assert "is not installed" in err and "module direct" in err and "Direct solver (rslab)" in err and "NGSolve" not in err
    assert solvers.backend_error("ngs_pardiso") == err                 # the earlier name of the same backend
    assert solvers.backend_error("direct") == err and solvers.canonical("hypre") == "ngs_hypre"      # the short names
    assert "Tools → Program modules" in err and "python -m fea_core.modules install accel direct" in err
    assert "Automatic" in err
    assert solvers.backend_error("ccx") is None and solvers.backend_error("auto") is None
    assert "install accel hypre" in solvers.backend_error("ngs_hypre")


def test_standard_profile_without_ngsolve_has_the_direct_solver(monkeypatch, tmp_path):
    """NGSolve is only for the flexible bodies of mechanisms: without it the direct and the iterative solver work."""
    _install(monkeypatch, tmp_path, ngsolve=False)
    st = solvers.backend_status()
    assert st["ngs_direct"]["ok"] and st["ngs_hypre"]["ok"] and st["ccx"]["ok"]


def test_the_ngsolve_fallback_assembly_needs_ngsolve(monkeypatch, tmp_path):
    """FEA_ASSEMBLY=ngsolve (the earlier assembly, a debugging fallback) is the only case where the linear path needs the module ngsolve."""
    _install(monkeypatch, tmp_path, ngsolve=False)
    monkeypatch.setenv("FEA_ASSEMBLY", "ngsolve")
    assert solvers.backend_status()["ngs_direct"]["missing"] == ["module ngsolve"]
    assert "install ngsolve" in solvers.backend_error("ngs_direct")


@pytest.mark.skipif(not config.IS_WIN, reason="Microsoft MPI jen ve Windows")
def test_hypre_without_msmpi(monkeypatch, tmp_path):
    """The hypre module installed (mpi4py, HYPRE.dll) but MS-MPI missing → a report about MPI, not "hypre is not installed"."""
    _install(monkeypatch, tmp_path, mpiexec=False)
    st = solvers.backend_status()
    assert st["ngs_direct"]["ok"]
    assert st["ngs_hypre"]["missing"] == ["Microsoft MPI (mpiexec)"]
    err = solvers.backend_error("ngs_hypre")
    assert "missing Microsoft MPI (mpiexec)" in err and "module hypre" not in err.split("Doinstalujte")[0]
    assert "python installer/fea_setup.py mpi" in err
    assert solvers.hypre_missing() == ["Microsoft MPI (mpiexec)"]
    from fea_core import hypre
    assert not hypre.available()


@pytest.mark.skipif(not config.IS_WIN, reason="složka MS-MPI jen ve Windows")
def test_mpiexec_from_msmpi_folder(monkeypatch, tmp_path):
    """mpiexec outside PATH (a program started before the installation of MS-MPI) – found in the MSMPI_BIN folder / Program Files."""
    _install(monkeypatch, tmp_path, mpiexec=False)
    pf = tmp_path / "pf" / "Microsoft MPI" / "Bin"
    pf.mkdir(parents=True)
    (pf / "mpiexec.exe").write_text("x")
    assert solvers.find_mpiexec() == str(pf / "mpiexec.exe")
    assert solvers.backend_status()["ngs_hypre"]["ok"]
    msb = tmp_path / "msmpi"
    msb.mkdir()
    (msb / "mpiexec.exe").write_text("x")
    monkeypatch.setenv("MSMPI_BIN", str(msb))
    assert solvers.find_mpiexec() == str(msb / "mpiexec.exe")


def test_full_install(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    st = solvers.backend_status()
    assert all(v["ok"] and not v["missing"] and not v["fix"] for v in st.values())
    assert all(solvers.backend_error(k) is None for k in solvers.KEYS)
    s = solvers.summary()
    assert s["ccx"] == s["ngs_direct"] == s["ngs_hypre"] == "ok" and "fix" not in s
    assert s["ccx_linear_solvers"] == ["PASTIX", "SPOOLES"]
    from fea_core import hypre
    assert hypre.available()


def test_missing_ccx_and_direct_library(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, ccx=False, rslab=False)
    st = solvers.backend_status()
    assert st["ccx"]["missing"] == ["the CalculiX program (ccx)"] and "FEA_CCX" in st["ccx"]["fix"]
    assert st["ngs_direct"]["missing"] == ["the direct solver library rslab (module direct)"]
    assert "install direct" in solvers.backend_error("ngs_direct")
    assert "MKL" not in solvers.backend_error("ngs_direct")           # nothing of Intel anywhere


def test_worker_module_error():
    msg = solvers.module_error("ngs_direct", "ngsolve.comp")
    assert "cannot run" in msg and "ngsolve" in msg
    assert "install ngsolve" in msg
    assert "direct solver library" in solvers.module_error("ngs_direct", "rslab")
    assert "module accel" in solvers.module_error("ngs_direct", "numba")
    assert "repair installation" in solvers.module_error("ngs_hypre", "neznamy")


def test_the_ngsolve_fallback_assembly_does_not_ask_for_numba(monkeypatch, tmp_path):
    """FEA_ASSEMBLY=ngsolve assembles with NGSolve (no numba in the process): missing numba is not reported, missing NGSolve is."""
    _install(monkeypatch, tmp_path / "a", numba=False, ngsolve=True)
    monkeypatch.setenv("FEA_ASSEMBLY", "ngsolve")
    assert solvers.backend_status()["ngs_direct"]["missing"] == [] and solvers.backend_error("ngs_direct") is None
    assert solvers.direct_missing() == []
    _install(monkeypatch, tmp_path / "b", numba=True, ngsolve=False)
    monkeypatch.setenv("FEA_ASSEMBLY", "ngsolve")
    assert solvers.direct_missing() == ["module ngsolve"]
    monkeypatch.delenv("FEA_ASSEMBLY")                                     # the own assembly: numba, not NGSolve
    assert solvers.direct_missing() == []
    _install(monkeypatch, tmp_path / "c", numba=False)
    assert solvers.direct_missing() == ["numba (module accel)"]


def test_import_failure_of_a_helper_process():
    """The end of the log of a helper process → the clear error: a missing module, an installed but broken one (DLL, version) – and None for anything else."""
    miss = solvers.import_failure("ngs_direct", "Traceback (most recent call last):\n  File \"w.py\", line 1\nModuleNotFoundError: No module named 'llvmlite'\n")
    assert "missing numba (module accel)" in miss and "install accel" in miss
    broken = ('Traceback (most recent call last):\n  File "/app/.venv/lib/python3.11/site-packages/rslab/__init__.py", line 3, in <module>\n'
              'OSError: libgomp.so.1: cannot open shared object file')
    msg = solvers.import_failure("ngs_direct", broken)
    assert "rslab" in msg and "installed but cannot be loaded" in msg and "libgomp" in msg and "install --force" in msg
    assert "Direct solver (rslab)" in msg
    mine = 'Traceback (most recent call last):\n  File "C:\\app\\fea_core\\assembly\\worker.py", line 3\nImportError: cannot import name x from fea_core'
    assert solvers.import_failure("ngs_direct", mine) is None                           # an error of our own code is shown as it is
    assert solvers.import_failure("ngs_direct", "ZeroDivisionError: division by zero") is None
    assert solvers.import_failure("ngs_direct", "") is None and solvers.import_failure("ngs_direct", None) is None


# ------------------------------------------------------------------ CalculiX linear solver
def test_linear_solvers(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, ccx_name="ccx_static.exe")
    ls = solvers.linear_solvers()
    assert ls == {"PASTIX": True, "SPOOLES": True}
    assert config.solver_name(config.AUTO_SOLVER) == "PASTIX"
    assert solvers.linear_solver_note(config.AUTO_SOLVER) == "" and solvers.linear_solver_note("SPOOLES") == ""
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles"))          # Linux build without PaStiX
    assert not solvers.linear_solvers()["PASTIX"]
    assert config.solver_name("PASTIX") == "SPOOLES" and config.solver_name(config.AUTO_SOLVER) == "SPOOLES"
    assert "does not contain" in solvers.linear_solver_note("PASTIX") and solvers.linear_solver_note(config.AUTO_SOLVER) == ""


def test_ccx_solver_note(monkeypatch, tmp_path):
    """The automatic choice (the stored word "PARDISO") only a line of the log (the user chose nothing – otherwise the warning would be in every
    result); an explicitly chosen unavailable solver warns with the name of the one actually used (previously silently swapped).
    """
    from fea_core.solve import ccx_solver_note
    _install(monkeypatch, tmp_path, ccx_name="ccx_static.exe")
    st = _study()
    warnings, logs = [], []
    assert ccx_solver_note(st, warnings, logs.append) == "PASTIX"
    assert warnings == [] and len(logs) == 1 and "automatic choice" in logs[0] and "PASTIX" in logs[0]
    st.solver.solver = "SPOOLES"
    assert ccx_solver_note(st, [], logs.append) == "SPOOLES" and len(logs) == 1
    monkeypatch.setattr(config, "ccx_solvers", lambda: {"PASTIX": False, "SPOOLES": True})
    monkeypatch.setattr(config, "solver_name", lambda req, procedure="static": "SPOOLES")
    st.solver.solver = "PASTIX"
    warnings = []
    assert ccx_solver_note(st, warnings, logs.append) == "SPOOLES"
    assert ccx_solver_note(st, warnings, logs.append) == "SPOOLES"
    assert len(warnings) == 1 and "uses the solver SPOOLES" in warnings[0] and len(logs) == 2


# ------------------------------------------------------------------ readiness check (before meshing)
def test_check_study_reports_missing_backend_before_meshing(monkeypatch, tmp_path):
    from fea_core.solve import check_study
    _install(monkeypatch, tmp_path, mpi4py=False, hypre_lib=False, mpiexec=False, rslab=False)
    errors, _ = check_study(_study("ngs_direct"), ["Dil"])
    assert any("is not installed" in e and "direct solver" in e for e in errors)
    errors, _ = check_study(_study("auto", n_cases=3), ["Dil"])               # auto: without an error, CalculiX calculates
    assert not any("Řešič" in e for e in errors), errors
    assert solvers.check_backend(_study("ccx")) == []


def test_check_backend_ignored_and_unsupported(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, mpi4py=False)
    st = _study("ngs_hypre")
    assert solvers.check_backend(st)
    st.analysis.kind = "nonlinear"                     # a nonlinear calculation always goes through ccx – the choice is not used
    assert solvers.backend_ignored(st)
    assert solvers.check_backend(st) == []
    st = _study("ngs_direct")
    st.load_cases = []                                 # without load states (heat, frequencies) – always ccx
    assert solvers.check_backend(st) == []
    st = _study("ngs_direct")
    st.twod.enabled = True
    assert solvers.check_backend(st) == []
    _install(monkeypatch, tmp_path / "plna")
    st = _study("ngs_pardiso")                         # the stored name of the earlier versions is still accepted
    st.constraints.append(Fixed("vrch"))
    assert solvers.check_backend(st) == []
    from fea_core.joints.model import BoltSpec
    from fea_core.study import CylindricalSupport, RemoteSupport
    st.constraints.append(CylindricalSupport("vrch"))                  # the own assembly eliminates the equations of the support (F13 step 1b)
    assert solvers.check_backend(st) == []
    st.bolts.append(BoltSpec("B1"))                                    # bolts: beam elements between rigid faces on the own assembly – the direct solver only (F13 step 1b, part 2)
    assert solvers.check_backend(st) == []
    errs = solvers.check_backend(st, "ngs_hypre")
    assert len(errs) == 1 and "cannot handle this study" in errs[0] and "bolts" in errs[0] and "direct solver only" in errs[0]
    assert solvers.check_backend(st, "auto") == []                     # override of the choice (MCP backend=)
    st.bolts.clear()
    st.constraints.append(RemoteSupport("vrch", [0, 0, 5.0]))          # a constraint that couples nodes: the direct solver only
    assert solvers.check_backend(st) == []
    hy = solvers.check_backend(st, "ngs_hypre")
    assert len(hy) == 1 and "direct solver only" in hy[0]


def test_run_study_rejects_missing_backend_without_meshing(monkeypatch, tmp_path):
    """run_study with a taken-over mesh (without the study check) still refuses an uninstalled solver – before meshing."""
    from fea_core import solve
    _install(monkeypatch, tmp_path, rslab=False)

    def no_mesh(*a, **k):
        raise AssertionError("síťovat se nemělo")
    monkeypatch.setattr(solve, "prepare_mesh", no_mesh)
    with pytest.raises(solve.StudyError, match="is not installed – missing"):
        solve.run_study(_study("ngs_direct"), tmp_path / "wd")


def test_modules_problem_reason(monkeypatch, tmp_path):
    """Window Program modules: hypre installed without MPI → the reason in the Status column."""
    from fea_core import modules
    _install(monkeypatch, tmp_path, mpiexec=False)
    monkeypatch.setattr(modules, "_has_import", lambda name: True)
    why = modules.problem("hypre")
    assert why.startswith("missing") and ("Microsoft MPI (mpiexec)" in why or "MPI (mpiexec" in why)
    assert modules.problem("gui") == ""


def test_mcp_and_rest_summary(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, rslab=False)
    s = solvers.summary()
    assert s["ccx"] == "ok" and s["ngs_direct"] == "missing the direct solver library rslab (module direct)"
    assert "install direct" in s["fix"]
    lines = solvers.table()
    assert any("ngs_direct" in x and "rslab" in x for x in lines)


def test_ngs_worker_missing_module_message(monkeypatch, tmp_path):
    """Helper process of NGSolve without the ngsolve module: an understandable message instead of a traceback printout."""
    from fea_core import solve_large

    from fea_core import procs
    out = {"text": "Traceback (most recent call last):\n  File \"worker.py\", line 1\n"
                   "ModuleNotFoundError: No module named 'ngsolve'\n"}

    def fake_run(cmd, log_path, cancel=None, **kw):   # the worker output goes to a log file (procs.run_tree)
        log_path.write_text(out["text"], encoding="utf-8")
        return 1
    monkeypatch.setattr(procs, "run_tree", fake_run)
    with pytest.raises(solve_large.LargeSolverError) as e:
        solve_large._worker("assemble", tmp_path, print, "ngs_direct")
    assert "cannot run" in str(e.value) and "ngsolve" in str(e.value)
    assert "Traceback" not in str(e.value)
    out["text"] = "RuntimeError: něco jiného"
    with pytest.raises(solve_large.LargeSolverError, match=r"NGSolve \(assemble\) failed"):
        solve_large._worker("assemble", tmp_path, print)


def test_run_ngs_reports_missing_component(monkeypatch, tmp_path):
    """run_ngs (a manual choice passed without a check) reports the missing component before it builds anything."""
    from fea_core import solve_large
    _install(monkeypatch, tmp_path, mpiexec=False)
    with pytest.raises(solve_large.LargeSolverError, match="mpiexec"):
        solve_large.run_ngs(_study("ngs_hypre"), None, tmp_path, [], print, solver="hypre")

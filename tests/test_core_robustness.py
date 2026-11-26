"""Robustness of the solver chain: progress markers, cancellable helper processes, ccx runner / parallel jobs,
.frd reading, saving of results and projects, cache of the nodal loads (bit-identical .inp and reactions)."""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _perf_helpers():
    spec = importlib.util.spec_from_file_location("_perf_eq", ROOT / "tests" / "test_perf_equivalence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================================================================== progress markers
def _string_starts() -> set[str]:
    """Starts of all string literals in fea_core (plain strings and the constant start of f-strings)."""
    out = set()
    for p in (ROOT / "fea_core").rglob("*.py"):
        if p.name == "progress.py":                  # the markers themselves
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.add(node.value)
            elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
                out.add(str(node.values[0].value))
    return out


def test_log_marks_match_producers():
    """Every LOG_MARKS prefix starts some message literal (the English rewording once broke 5 of them)."""
    from fea_core.progress import LOG_MARKS
    starts = _string_starts()
    missing = [p for p, *_ in LOG_MARKS if not any(s.startswith(p) for s in starts)]
    assert not missing, f"LOG_MARKS prefixes without a producer: {missing}"


def test_ccx_marks_in_real_log():
    """CCX_MARKS follow the output of ccx (a log of a verification run, when present)."""
    from fea_core.progress import CCX_MARKS, AFTER_FACTOR, FACTOR_MARKS
    logs = [p for p in (ROOT / "verification" / "out" / "V1" / "run").glob("job*.log")] if \
        (ROOT / "verification" / "out" / "V1" / "run").exists() else []
    if not logs:
        pytest.skip("no ccx log of V1 (python -m verification.run_all V1)")
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    for mark in [m for m, _ in CCX_MARKS if m not in FACTOR_MARKS] + [AFTER_FACTOR[0]]:
        assert mark in text, mark
    assert any(m in text for m in FACTOR_MARKS), FACTOR_MARKS      # the log of the SPOOLES / PaStiX solver


def test_progress_moves_on_real_messages():
    from fea_core.progress import Progress
    seen = []
    prog = Progress(lambda v, b, t: seen.append(t), ["check", "mesh", "mech", "post"])
    log = prog.log(lambda s: None)
    for msg in ("Solving (CalculiX)", "Assembling stiffness matrix",
                "Solving the system of 3 000 equations directly (rslab, one factorization for 2 states, memory estimate 0.1 GB before the analysis)",
                "Solving a system of 3 000 equations (hypre BoomerAMG, 8 processes, fast mode)",
                "3D printing: assigning shell and infill to the mesh", "3D printing: failure criterion"):
        log(msg)
    assert seen == ["Load cases (CalculiX)", "Assembling stiffness matrix", "Solving system of equations",
                    "Solving system of equations", "3D printing: shell and infill", "3D printing: failure criterion"]


# ===================================================================== ccx warnings
def test_relevant_warnings_allowlist():
    from fea_core.ccx.runner import relevant_warnings
    w = ["*WARNING in calinput: no elastic constants", "*WARNING in gentiedmpc:",
         "*WARNING reading *BOUNDARY: parameter X is ignored",
         "*WARNING reading *HEAT TRANSFER: a nonlinear geometric analysis is requested"]
    assert relevant_warnings(w) == ["*WARNING in gentiedmpc:", "*WARNING reading *BOUNDARY: parameter X is ignored"]


# ===================================================================== helper processes
_TREE = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""


def test_run_tree_cancel_kills_children(tmp_path):
    import psutil
    from fea_core.procs import run_tree, Canceled
    cancel = threading.Event()
    lines = []
    script = tmp_path / "tree.py"
    script.write_text(_TREE, encoding="utf-8")

    def later():
        t0 = time.time()
        while not lines and time.time() - t0 < 20:
            time.sleep(0.05)
        cancel.set()
    threading.Thread(target=later, daemon=True).start()
    t0 = time.time()
    with pytest.raises(Canceled):
        run_tree([sys.executable, str(script)], tmp_path / "tree.log", cancel, on_line=lines.append)
    assert time.time() - t0 < 30
    child = int(lines[0])
    time.sleep(0.3)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_run_tree_output_and_timeout(tmp_path):
    from fea_core.procs import run_tree, tail
    got = []
    rc = run_tree([sys.executable, "-c", "print('a'); print('b'); raise SystemExit(3)"], tmp_path / "x.log",
                  on_line=got.append)
    assert rc == 3 and got[:2] == ["a", "b"] and "b" in tail(tmp_path / "x.log")
    with pytest.raises(subprocess.TimeoutExpired):
        run_tree([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "y.log", timeout=0.5)


def test_ngs_worker_cancel(tmp_path, monkeypatch):
    """The NGSolve worker is cancellable: the process is killed, LargeSolverError("… canceled …")."""
    from fea_core import solve_large
    cancel = threading.Event()
    real = solve_large._worker
    from fea_core import procs
    orig = procs.run_tree

    def slow(cmd, log_path, cancel_=None, **kw):     # instead of the NGSolve worker a sleeping Python
        return orig([sys.executable, "-c", "import time; time.sleep(30)"], log_path, cancel_, **kw)
    monkeypatch.setattr(procs, "run_tree", slow)
    threading.Timer(0.5, cancel.set).start()
    t0 = time.time()
    with pytest.raises(solve_large.LargeSolverError, match="canceled"):
        real("assemble", tmp_path, print, "ngs_direct", cancel)
    assert time.time() - t0 < 10


def test_ngs_files_removed_after_error(tmp_path, monkeypatch):
    from fea_core import solve_large
    job = tmp_path / "ngs"

    def boom(*a, **k):
        (job / "hypre").mkdir(parents=True, exist_ok=True)
        np.save(job / "F.npy", np.zeros(3))
        np.save(job / "hypre" / "data.npy", np.zeros(3))
        raise solve_large.LargeSolverError("boom")
    monkeypatch.setattr(solve_large, "_run_ngs_job", boom)
    monkeypatch.setattr(solve_large, "unsupported", lambda s, solver="direct": [])
    import fea_core.solvers as solvers
    monkeypatch.setattr(solvers, "backend_error", lambda b: None)
    monkeypatch.delenv(solve_large.KEEP_FILES_ENV, raising=False)
    with pytest.raises(solve_large.LargeSolverError):
        solve_large.run_ngs(None, None, tmp_path, [], print)
    assert not list(job.rglob("*.npy"))
    monkeypatch.setenv(solve_large.KEEP_FILES_ENV, "1")
    logged = []
    with pytest.raises(solve_large.LargeSolverError):
        solve_large.run_ngs(None, None, tmp_path, [], logged.append)
    assert (job / "F.npy").exists() and any("FEA_KEEP_SOLVER_FILES" in s for s in logged)


# ===================================================================== parallel ccx jobs, .frd
def test_parallel_sibling_cancel_not_reported(tmp_path, monkeypatch):
    from fea_core.ccx import parallel
    from fea_core.ccx.runner import CcxError, CcxRun
    (tmp_path / "job.inp").write_text("*NODE\n*STEP\n*STATIC\n*END STEP\n*STEP\n*STATIC\n*END STEP\n", encoding="latin-1")

    def fake(workdir, name, exe=None, cancel=None, threads=None, on_line=None):
        if name.endswith("s0"):
            time.sleep(0.2)
            return CcxRun(1, False, ["*ERROR in step 1"], [], None)
        while not cancel.wait(0.05):
            pass
        raise CcxError("Analysis canceled by user")
    monkeypatch.setattr(parallel, "run_ccx", fake)
    with pytest.raises(CcxError) as e:
        parallel.run_steps(tmp_path, "job", 100, workers=2, threads=3)
    assert "*ERROR in step 1" in str(e.value) and "canceled" not in str(e.value)


def test_parallel_user_cancel(tmp_path, monkeypatch):
    from fea_core.ccx import parallel
    from fea_core.ccx.runner import CcxError
    (tmp_path / "job.inp").write_text("*NODE\n*STEP\n*END STEP\n*STEP\n*END STEP\n", encoding="latin-1")
    cancel = threading.Event()

    def fake(workdir, name, exe=None, cancel=None, threads=None, on_line=None):
        while not cancel.wait(0.05):
            pass
        raise CcxError("Analysis canceled by user")
    monkeypatch.setattr(parallel, "run_ccx", fake)
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(CcxError, match="canceled by user"):
        parallel.run_steps(tmp_path, "job", 100, cancel=cancel, workers=2, threads=3)


def test_read_frd_steps_empty_job(tmp_path):
    from fea_core.ccx import parallel
    from fea_core.ccx.runner import CcxError
    (tmp_path / "job_s0.frd").write_bytes(b"")
    with pytest.raises(CcxError, match="empty .frd"):
        parallel.read_frd_steps(tmp_path, ["job_s0"])


def test_frd_truncated_binary_block(tmp_path):
    from fea_core.ccx.frd import read_frd
    from fea_core.ccx.runner import CcxError
    n = 50
    head = ("    1C\n" f"    1PSTEP{1:26d}{1:12d}{1:12d}\n"
            f"  100CL  101 1.000000000{n:12d}                     0    1           2\n"
            f" -4  DISP    {3:4d}    1\n" + "".join(f" -5  {c:8s}    1    2    1    0\n" for c in ("D1", "D2", "D3")))
    data = np.zeros(10, dtype=np.dtype([("id", "<i4"), ("vals", "<f4", (3,))])).tobytes()   # 10 of 50 records
    p = tmp_path / "t.frd"
    p.write_bytes(head.encode() + data)
    with pytest.raises(CcxError, match="truncated"):
        read_frd(p)


# ===================================================================== results and project files
def test_results_info_roundtrip(tmp_path):
    from fea_core.beams.run import empty_mesh
    from fea_core.results import Results, INFO_KEYS
    info = {"backend": "ccx", "ccx_parallel": {"jobs": 3, "workers": 3, "threads": 3},
            "backend_fallback": {"from": "ngs_direct", "reason": "x"}, "twod": {"kind": "axisym", "n": np.int64(3)},
            "shell": {"parts": ["A"]}, "inp_map": object(), "log": "job.log"}
    Results(empty_mesh(), [], [], info).save(tmp_path / "results.npz")
    back = Results.load(tmp_path / "results.npz").info
    assert back["ccx_parallel"] == info["ccx_parallel"] and back["backend_fallback"] == info["backend_fallback"]
    assert back["twod"] == {"kind": "axisym", "n": 3} and back["shell"] == {"parts": ["A"]}
    assert "inp_map" not in back and set(back) <= set(INFO_KEYS)


def test_result_files_follow_results():
    from fea_core.project import RESULT_FILES
    assert RESULT_FILES == ("results.npz", "results_mesh.npz", "results_2d.npz", "results_shell.npz",
                            "results_shell.json", "results_dyn.npz", "results_sub.npz")


def test_save_project_removes_tmp_on_failure(tmp_path, monkeypatch):
    import zipfile
    from fea_core.project import save_project
    from fea_core.study import Study
    step = tmp_path / "m.step"
    step.write_text("ISO-10303-21;", encoding="ascii")

    def bad(self, *a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(zipfile.ZipFile, "write", bad)
    with pytest.raises(OSError):
        save_project(tmp_path / "p.fea", Study("s", str(step)), step)
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "p.fea").exists()


# ===================================================================== cache of the nodal loads
def test_nodal_cache_bit_identical(tmp_path):
    """write_inp and nodal_loads inside nodal_cache = without the cache, byte for byte / bit for bit."""
    H = _perf_helpers()
    from fea_core.ccx.inp import write_inp
    from fea_core.loads import nodal_cache, nodal_loads, load_resultant, force_nodal
    from fea_core.study import Force, Pressure, Gravity, LoadCase
    m = H.box_mesh()
    loads = [Force("x1", [10.0, -3.0, 0.0]), Pressure("z0", 0.7), Gravity([0.0, 0.0, -9810.0])]
    st = H.study_for(m, loads)
    st.load_cases.append(LoadCase("L2", [Force("x1", [0.0, 5.0, 1.0]), loads[1]]))
    write_inp(st, m, tmp_path / "a.inp")
    ref = [nodal_loads(st, m, lc) for lc in st.load_cases]
    res = [load_resultant(st, m, lc) for lc in st.load_cases]
    with nodal_cache():
        write_inp(st, m, tmp_path / "b.inp")
        got = [nodal_loads(st, m, lc) for lc in st.load_cases]
        got2 = [nodal_loads(st, m, lc) for lc in st.load_cases]
        res2 = [load_resultant(st, m, lc) for lc in st.load_cases]
        f = force_nodal(st, m, loads[0])
        f[:] = 1e9                                   # a caller writing into the result does not spoil the cache
        assert H.same(force_nodal(st, m, loads[0]), force_nodal.uncached(st, m, loads[0]))
    assert (tmp_path / "a.inp").read_bytes() == (tmp_path / "b.inp").read_bytes()
    for a, b, c in zip(ref, got, got2):
        for x, y, z in zip(a, b, c):
            assert H.same(x, y) and H.same(x, z)
    for a, b in zip(res, res2):
        assert H.same(a, b)

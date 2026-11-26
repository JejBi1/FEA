"""Direct solver, review fixes: a failing float32 factor is repeated in float64 (or declines for the iterative solver when float64 does not
fit), acceptance by the backward error, out-of-memory recognised from the output only, one direct run at a time, no paths in the saved
results, status checks without importing scipy.sparse, solver names of stored results, the end-to-end chain against CalculiX.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from fea_core import direct
from fea_core.direct import engines
from fea_core.direct import worker as W

HAS_RSLAB = importlib.util.find_spec("rslab") is not None
needs_rslab = pytest.mark.skipif(not HAS_RSLAB, reason="the library rslab is not installed (module direct)")
needs_direct_chain = pytest.mark.skipif(not (HAS_RSLAB and importlib.util.find_spec("numba") is not None),
                                        reason="numba + rslab (modules accel, direct) are not installed")


@pytest.fixture(autouse=True)
def _rslab_on_the_path_of_the_worker(monkeypatch):
    """The worker is a separate process: where rslab is found by this process (a development copy outside the site-packages), it must find
    it too (other tests import the installer, which clears PYTHONPATH of the process)."""
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))


def _biharmonic(n: int):
    """The 1D biharmonic operator (clamped): positive definite with the condition number ~ n^4 (n = 1000: 1.6e11, n = 3000: 1.3e13)."""
    D = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")
    K = (D @ D).tocsr()
    K.sort_indices()
    B = np.column_stack([np.ones(n), np.sin(np.linspace(0, 3, n))])
    return K, B


def _write(path: Path, K, B):
    U = sp.triu(K, format="csr")
    U.sort_indices()
    np.save(path / "indptr.npy", U.indptr.astype(np.int64))
    np.save(path / "indices.npy", U.indices.astype(np.int32))
    np.save(path / "data.npy", U.data)
    np.save(path / "rhs.npy", B)
    (path / "matrix.json").write_text(json.dumps({"format": "upper"}), encoding="utf-8")


# ------------------------------------------------------------------ 1: a float32 factor that fails is repeated in float64
@needs_rslab
def test_float32_negative_pivots_of_an_ill_conditioned_matrix_are_repeated_in_float64(tmp_path):
    """The matrix is positive definite (float64 has no negative pivot), but its condition number 1e11 is beyond float32: the float32 factor
    has negative pivots. It is NOT a free body – the worker repeats the factorization in float64 instead of exit code 5."""
    K, B = _biharmonic(1000)
    _write(tmp_path, K, B)
    X, info = direct.run_prepared(tmp_path, threads=2, precision="float32")
    assert info["precision"] == "float64" and "float32 factor not accepted" in info["precision_reason"]
    assert "negative" in info["float32_failure"]
    assert max(info["backward_error"]) < 1e-14 and info["inertia"] == [1000, 0, 0]
    assert np.abs(K @ X - B).max() < 1e-3                      # condition 1e11 · eps = the forward accuracy; the backward error is what counts


@needs_rslab
def test_automatic_precision_keeps_float32_for_a_well_conditioned_matrix(tmp_path):
    rng = np.random.default_rng(5)
    A = sp.random(700, 700, density=0.01, random_state=5, format="csr")
    K = (A @ A.T + sp.identity(700) * 5.0).tocsr()
    _write(tmp_path, K, rng.standard_normal((700, 2)))
    _, info = direct.run_prepared(tmp_path, threads=2, precision="float32")
    assert info["precision"] == "float32" and "float32_failure" not in info


@needs_rslab
def test_a_float64_failure_is_a_free_body(tmp_path):
    """Only the float64 factor may say 'free body': a matrix with a zero row is singular in every precision."""
    K, B = _biharmonic(200)
    K = K.tolil()
    K[7, :] = 0
    K[:, 7] = 0
    K = K.tocsr()
    K.eliminate_zeros()
    K.sort_indices()
    _write(tmp_path, K, B)
    for precision in ("float32", "float64"):
        with pytest.raises(direct.DirectError, match="singular|not positive definite|pivot") as e:
            direct.run_prepared(tmp_path, threads=1, precision=precision)
        assert not isinstance(e.value, direct.DirectMemoryError)       # the repeated float32 case ends with the float64 diagnosis too


class _FakeEngine(engines.Engine):
    """An engine whose float32 factor fails and whose float64 factor needs a given memory – the decisions of the worker without a library."""

    name, version = "fake", "0"
    plans = {"float32": 0.01, "float64": 0.02}
    fail = {"float32": True, "float64": False}

    def __init__(self, n, indptr, indices, data, threads):
        self.n = n
        self.factor_nnz = 3 * n
        self._prec = None
        self.factored: list[str] = []

    def analyze(self):
        pass

    def plan_gb(self, precision, nrhs):
        return self.plans[precision]

    def factor(self, precision):
        self.factored.append(precision)
        if self.fail[precision]:
            raise engines.EngineError("fake: zero pivot")
        self.inertia, self.n_perturbed, self._prec = (self.n, 0, 0), 0, precision

    def solve(self, B):
        return np.linalg.solve(self.K.toarray(), B)


def _fake_system(monkeypatch, plans=None, fail=None, n=40):
    K = sp.diags([-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    U = sp.triu(K, format="csr")
    U.sort_indices()
    made = []

    class Fake(_FakeEngine):
        pass
    Fake.plans = {**_FakeEngine.plans, **(plans or {})}
    Fake.fail = {**_FakeEngine.fail, **(fail or {})}
    Fake.K = K

    def make(name, n_, indptr, indices, data, threads):
        e = Fake(n_, indptr, indices, data, threads)
        made.append(e)
        return e
    monkeypatch.setattr(engines, "make", make)
    return U, K, made


def test_float32_failure_and_float64_fits_gives_float64(monkeypatch):
    U, K, made = _fake_system(monkeypatch)
    B = np.ones((40, 1))
    info = {}
    X = W.solve_system(U, B, {"engine": "fake", "precision": "float32", "reserve_gb": 0.0}, info)
    assert info["precision"] == "float64" and [e.factored for e in made] == [["float32"], ["float64"]]
    assert np.abs(K @ X - B).max() < 1e-10


def test_float32_failure_and_float64_does_not_fit_declines_like_a_missing_memory(monkeypatch):
    """The caller (auto) continues with the iterative solver (exit code 3 = DirectMemoryError), not with CalculiX."""
    U, K, made = _fake_system(monkeypatch, plans={"float64": 1e6})
    info = {}
    with pytest.raises(W.WorkerFail) as e:
        W.solve_system(U, np.ones((40, 1)), {"engine": "fake", "precision": "float32", "reserve_gb": 0.0}, info)
    assert e.value.code == W.EXIT_NOFIT and e.value.kind == "memory"
    assert "float32 factor was not accepted" in e.value.message and "float64 factor needs" in e.value.message
    assert [x.factored for x in made] == [["float32"]]                   # float64 was never started


def test_float64_failure_after_a_float32_failure_reports_the_free_body(monkeypatch):
    U, K, made = _fake_system(monkeypatch, fail={"float32": True, "float64": True})
    with pytest.raises(W.WorkerFail) as e:
        W.solve_system(U, np.ones((40, 1)), {"engine": "fake", "precision": "auto", "reserve_gb": 0.0}, {})
    assert e.value.code == W.EXIT_NUMERIC and "singular" in e.value.message


def test_worker_exit_code_3_becomes_direct_memory_error(tmp_path, monkeypatch):
    """… and run_prepared turns that exit code into DirectMemoryError (the caller's fallback to hypre)."""
    monkeypatch.setattr(direct, "available", lambda: True)

    def fake_run_tree(cmd, log_path, cancel=None, env=None, on_line=None, **kw):
        Path(log_path).write_text("direct solver: float32 factor not accepted\n", encoding="utf-8")
        (tmp_path / "direct_info.json").write_text(json.dumps({"error": {"kind": "memory", "message": "float64 needs 9 GB"}}),
                                                   encoding="utf-8")
        return direct.EXIT_NOFIT
    monkeypatch.setattr("fea_core.procs.run_tree", fake_run_tree)
    with pytest.raises(direct.DirectMemoryError, match="float64 needs 9 GB"):
        direct.run_prepared(tmp_path, threads=1)


# ------------------------------------------------------------------ 6: acceptance by the backward error
@needs_rslab
def test_float64_solution_of_an_ill_conditioned_system_is_accepted_by_the_backward_error(tmp_path):
    """Condition 1e13: backward error 5e-17 (as good as it gets) but the relative residual 7e-4 – more than the former limit 1e-4."""
    K, B = _biharmonic(3000)
    _write(tmp_path, K, B)
    _, info = direct.run_prepared(tmp_path, threads=2, precision="float64")
    assert max(info["backward_error"]) < 1e-15 and max(info["rel_residual"]) > 1e-4
    assert max(info["rel_residual"]) < W.RESIDUAL_GUARD


@needs_rslab
def test_nearly_singular_system_is_still_rejected_by_the_sanity_guard(tmp_path):
    """Condition 1.6e15 (≈ 1/eps): the relative residual 0.09 is beyond the loose sanity limit – no meaningful solution."""
    K, B = _biharmonic(10000)
    _write(tmp_path, K, B)
    with pytest.raises(direct.DirectError, match="accuracy"):
        direct.run_prepared(tmp_path, threads=2, precision="float64")


# ------------------------------------------------------------------ 5: out of memory only from what the process says
@pytest.mark.parametrize("rc, text, oom", [
    (0xC0000409, "", False),                                                # Rust abort / fast-fail: any abort, not memory
    (0xC0000409, "memory allocation of 8589934592 bytes failed", True),     # … unless it says so
    (0xC00000FD, "thread 'main' has overflowed its stack", False),          # stack overflow
    (0x80000003, "", False),                                                # breakpoint
    (-6, "", False),                                                        # Linux SIGABRT without the text
    (-6, "memory allocation of 123456 bytes failed\nAborted", True),
    (134, "memory allocation of 123456 bytes failed", True),
    (1, "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 3.2 GiB for an array", True),
    (1, "Traceback ...\nMemoryError", True),
    (1, "thread panicked at src/lib.rs: index out of bounds", False),
    (1, "", False),
])
def test_out_of_memory_is_recognised_from_the_output(rc, text, oom):
    assert direct.is_out_of_memory(rc, text) is oom


def test_out_of_memory_exit_codes_without_output(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert direct.is_out_of_memory(0xC0000017, "") and direct.is_out_of_memory(-1073741801, "")   # STATUS_NO_MEMORY
    monkeypatch.setattr(sys, "platform", "linux")
    assert direct.is_out_of_memory(-9, "") and direct.is_out_of_memory(137, "")                    # the OOM killer (nobody else kills it)
    assert not direct.is_out_of_memory(-11, "")                                                    # SIGSEGV


@pytest.mark.parametrize("rc, text, kind", [
    (0xC0000409, "", direct.DirectError),
    (0xC0000409, "memory allocation of 8589934592 bytes failed", direct.DirectMemoryError),
    (-1073740791, "thread panicked: assertion failed", direct.DirectError),
])
def test_run_prepared_classifies_a_crashed_worker(tmp_path, monkeypatch, rc, text, kind):
    monkeypatch.setattr(direct, "available", lambda: True)

    def fake_run_tree(cmd, log_path, cancel=None, env=None, on_line=None, **kw):
        Path(log_path).write_text(text, encoding="utf-8")
        return rc
    monkeypatch.setattr("fea_core.procs.run_tree", fake_run_tree)
    with pytest.raises(direct.DirectError) as e:
        direct.run_prepared(tmp_path, threads=1)
    assert isinstance(e.value, direct.DirectMemoryError) is (kind is direct.DirectMemoryError)
    if kind is direct.DirectError:
        first = str(e.value).splitlines()[0]
        assert "failed (code" in first and "memory" not in first.lower()          # the real reason is in the first line (the fallback logs it)


# ------------------------------------------------------------------ 3: one direct run at a time in the program
def test_direct_runs_exclude_each_other_and_waiting_is_cancellable():
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def first():
        with direct.exclusive():
            order.append("first in")
            inside.set()
            release.wait(5)
            order.append("first out")

    t = threading.Thread(target=first)
    t.start()
    assert inside.wait(5)
    logs: list[str] = []

    def second():
        with direct.exclusive(log=logs.append):
            order.append("second in")

    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.5)
    assert order == ["first in"] and logs and "Waiting" in logs[0]           # the second one waits and says so
    cancel = threading.Event()
    box: list[BaseException] = []

    def third():
        try:
            with direct.exclusive(cancel):
                order.append("third in")
        except direct.DirectError as e:
            box.append(e)

    t3 = threading.Thread(target=third)
    t3.start()
    time.sleep(0.4)
    cancel.set()
    t3.join(5)
    assert box and "canceled" in str(box[0]) and "third in" not in order      # a cancelled waiting run leaves without running
    release.set()
    t.join(5)
    t2.join(5)
    assert order == ["first in", "first out", "second in"]
    with direct.exclusive():                                                  # re-entrant in the same thread (assembly + run_prepared)
        with direct.exclusive():
            pass


# ------------------------------------------------------------------ 4: status checks import nothing heavy
def test_solver_status_does_not_import_scipy_sparse():
    code = ("import sys\nimport fea_core.solvers as s\ns.backend_status()\ns.summary()\n"
            "import fea_core.direct\nfea_core.direct.available()\nfea_core.direct.library_name()\n"
            "print('SCIPY_SPARSE' if 'scipy.sparse' in sys.modules else 'LIGHT')")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=root, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "LIGHT"


# ------------------------------------------------------------------ 7: results are labelled by what ran
def test_a_result_of_the_earlier_mkl_version_is_not_called_the_direct_solver():
    from fea_core.solvers import canonical, result_title
    assert canonical("ngs_pardiso") == "ngs_direct"                          # the SETTING of a study: the alias
    assert "MKL PARDISO" in result_title("ngs_pardiso") and "rslab" not in result_title("ngs_pardiso")
    assert result_title("ngs_direct") == "Direct solver (rslab)" and result_title("ccx") == "CalculiX"


def test_report_labels_the_mkl_results_by_what_ran():
    from types import SimpleNamespace

    from fea_core.report import word
    from fea_core.study import Study
    st = Study("t", "m.step")
    old = SimpleNamespace(info={"backend": "ngs_pardiso"})
    new = SimpleNamespace(info={"backend": "ngs_direct"})
    assert "MKL PARDISO" in word._program_row(st, old) and "direct solver" not in word._program_row(st, old)
    assert "Direct solver (rslab)" in word._program_row(st, new)


# ------------------------------------------------------------------ 2 + 9: the whole chain on a real mesh, and what is saved
@needs_direct_chain
def test_ngs_direct_chain_matches_ccx_and_saves_no_path(tmp_path):
    """Gmsh mesh → the own assembly writes the upper triangle → the direct worker (rslab) → recovery of the stresses → the same results as
    CalculiX on the same mesh; the saved results carry no absolute path (the .fea files are shared)."""
    from fea_core.results import Results
    from fea_core.solve import run_study
    from fea_core.solve_large import run_ngs
    from verification import cases as C
    st, _ = C.v1_build(tmp_path)
    st.solver.backend = "ccx"
    res_c = run_study(st, tmp_path / "ccx")
    st.solver.backend = "ngs_direct"
    logs: list[str] = []
    res_d = run_ngs(st, res_c.mesh, tmp_path / "ngs", [], logs.append, solver="direct")
    a, b = res_c.step(0), res_d.step(0)
    assert float(np.abs(a.U - b.U).max() / np.abs(a.U).max()) < 1e-5
    assert abs(float(a.mises.max()) - float(b.mises.max())) / float(a.mises.max()) < 1e-3
    assert float(np.abs(a.reactions["vetknuti"] - b.reactions["vetknuti"]).max()) < 1e-3
    assert b.equilibrium["imbalance"] < 1e-6
    p = res_d.info["solver"]
    assert p["backend"] == "ngs_direct" and p["engine"].startswith("rslab") and p["matrix_format"] == "upper"
    assert max(c["backward_error"] for c in p["cases"]) < 1e-12 and p["memory"]["peak_gb"] > 0
    assert any("directly" in ln for ln in logs)
    # the ccx error estimate: in the ccx result, and an explicit note (not a silent gap) in the direct solver result
    from fea_core.mcp import diagnostics
    from fea_core.report.word import session_from
    h_c = diagnostics.hotspots(session_from(st, tmp_path / "model.step", res_c), 0, 3)
    h_d = diagnostics.hotspots(session_from(st, tmp_path / "model.step", res_d), 0, 3)
    assert "stress_error_pct" in h_c["hotspots"][0] and "stress_error_note" not in h_c
    assert "stress_error_pct" not in h_d["hotspots"][0] and "backend='ccx'" in h_d["stress_error_note"]
    # nothing in the saved meta is an absolute path
    out = tmp_path / "saved" / "results.npz"
    out.parent.mkdir()
    res_d.save(out)
    with np.load(out, allow_pickle=False) as z:
        meta_text = bytes(z["meta"]).decode("utf-8")
    meta = json.loads(meta_text)
    assert "log" not in meta["direct"] and "log" not in meta["solver"]
    for needle in (str(tmp_path), str(tmp_path).replace("\\", "/"), "direct.log", "ngs_assemble.log"):
        assert needle not in meta_text and needle.replace("\\", "\\\\") not in meta_text
    import re
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", meta_text.replace("\\\\", "\\"))      # a drive path
    back = Results.load(out)
    assert back.info["solver"]["backend"] == "ngs_direct"


# ------------------------------------------------------------------ 8c: the backend of a run is decided once
class _Mesh:
    def __init__(self, nodes: int):
        self.nodes = np.zeros((nodes, 3))
        self.part_names = ["P"]


def _study_with_modal():
    from fea_core.selectors import FaceSelector
    from fea_core.study import Fixed, Force, LoadCase, Material, Study
    st = Study("t", "m.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9))
    st.add_selection("dno", [FaceSelector("P", "plane", 1.0, [0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, 0, 1, 1, 0])])
    st.constraints.append(Fixed("dno"))
    st.load_cases = [LoadCase("LC1", [Force("dno", [0, 0, -100.0])])]
    st.modal.enabled = True
    return st


def test_side_jobs_take_the_decision_of_the_run_and_do_not_ask_again(monkeypatch):
    from fea_core import solve
    from fea_core.side_jobs import SideJobs

    def boom(*a, **k):
        raise AssertionError("the backend must not be decided again")
    monkeypatch.setattr(solve, "decide_backend", boom)
    monkeypatch.setattr(solve, "choose_backend_reason", boom)
    side = SideJobs(_study_with_modal(), _Mesh(1000), Path("."), print, None, None, [], backend="ngs_hypre")
    assert not side.active                                       # beside the iterative solver (MPI on all cores) the side jobs run after it


def test_side_jobs_beside_the_direct_solver_are_planned_from_the_memory():
    from fea_core.side_jobs import beside_direct
    ndof = 200_000
    assert beside_direct(ndof, 1, free_gb=16.0, cores=8) == (2, 4)               # the direct solver + one ccx job, 4 threads each
    assert beside_direct(ndof, 3, free_gb=64.0, cores=8) == (4, 3)               # + three ccx jobs, never fewer than 3 threads
    assert beside_direct(ndof, 1, free_gb=3.0, cores=8) == (1, 8)                # no memory for both: the side jobs follow the statics
    assert beside_direct(ndof, 1, free_gb=16.0, cores=4) == (1, 4)               # too few cores
    assert beside_direct(ndof, 0, free_gb=16.0, cores=8) == (1, 8)
    assert beside_direct(1_500_000, 2, free_gb=24.0, cores=8) == (1, 8)          # a large model: the direct solver alone has the memory


def test_side_jobs_start_beside_the_direct_solver_with_its_share_of_threads(monkeypatch, tmp_path):
    from fea_core import side_jobs, solve
    from fea_core.ccx import parallel
    started = []

    class FakeBackground:
        def __init__(self, fn, cancel, label):
            started.append(label)
            self.stop = threading.Event()

        def abort(self, log=None):
            pass
    monkeypatch.setattr(solve, "Background", FakeBackground)
    monkeypatch.setenv("FEA_MODAL_ENGINE", "ccx")                  # the natural frequencies of the direct path are no side job (they follow the mechanics)
    monkeypatch.setattr(side_jobs, "beside_direct", lambda ndof, n_side, **k: (2, 4))
    side = side_jobs.SideJobs(_study_with_modal(), _Mesh(70_000), tmp_path, print, None, None, [], backend="ngs_direct")
    assert side.active and started == ["eigen"] and side.mech_share == (1, 4)
    with side.mech_budget():
        assert parallel.budget_threads() == 4                      # the direct worker takes these threads (solve_large)


def test_side_jobs_decide_once_when_not_told(monkeypatch):
    from fea_core import solve
    from fea_core.ccx import parallel
    from fea_core.side_jobs import SideJobs
    calls = []
    monkeypatch.setenv("FEA_MODAL_ENGINE", "ccx")                  # the natural frequencies of the direct path are no side job (they follow the mechanics)
    monkeypatch.setattr(solve, "decide_backend", lambda st, mesh, log=None: calls.append(1) or ("ngs_hypre", "x"))
    side = SideJobs(_study_with_modal(), _Mesh(1000), Path("."), print, None, None, [])
    assert calls == [1] and not side.active
    monkeypatch.setattr(solve, "decide_backend", lambda st, mesh, log=None: ("ccx", "x"))
    monkeypatch.setattr(parallel, "plan", lambda n_steps, ndof, *a, **k: (1, 8))     # one job: nothing runs beside the load cases
    assert not SideJobs(_study_with_modal(), _Mesh(1000), Path("."), print, None, None, []).active


def test_run_mechanics_uses_the_decision_it_is_given(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fea_core import solve, solve_large
    seen = []

    def fake_run_ngs(study, mesh, workdir, warnings, log, cancel, thermal, solver):
        seen.append(solver)
        return SimpleNamespace(info={"solver": {}}, thermal=None)
    monkeypatch.setattr(solve_large, "run_ngs", fake_run_ngs)
    monkeypatch.setattr(solve, "choose_backend_reason", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decided again")))
    st = _study_with_modal()
    res = solve._run_mechanics(st, _Mesh(100), tmp_path, [], print, None, None, [], None, None,
                               decision=("ngs_direct", "decided at the start of the run"))
    assert seen == ["direct"] and res.info["solver"]["reason"] == "decided at the start of the run"


def test_decide_backend_nonlinear_is_ccx_without_asking(monkeypatch):
    from fea_core import solve
    monkeypatch.setattr(solve, "choose_backend_reason", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked")))
    st = _study_with_modal()
    st.contact_settings.nlgeom = True
    assert solve.decide_backend(st, _Mesh(100)) == ("ccx", solve.NONLINEAR_REASON)


# ------------------------------------------------------------------ 8a: the missing ccx error estimate is stated, not an empty column
def test_convergence_says_why_the_ccx_error_estimate_is_missing(monkeypatch, tmp_path):
    import types

    from fea_core import convergence
    from fea_core.study import Study
    nodes = np.array([[0.0, 0, 0], [10.0, 0, 0]])

    def fake_run_study(st, workdir, **kw):
        stp = types.SimpleNamespace(mises=np.array([0.0, 200.0 - 2.0 * st.mesh.size ** 2]), umag=np.zeros(2), fields={})   # no stress_error
        return types.SimpleNamespace(mesh=types.SimpleNamespace(nodes=nodes, tets=np.zeros((3, 10), int)), modes=[], steps=[stp],
                                     step=lambda i: stp, info={"backend": "ngs_direct"})
    monkeypatch.setattr("fea_core.solve.run_study", fake_run_study)
    st = Study("t", "a.step")
    st.mesh.size = 4.0
    out = convergence.run_convergence(st, tmp_path, convergence.ConvergenceSpec(levels=3, ratio=2.0))
    assert all(r["stress_error_pct"] is None and r["backend"] == "ngs_direct" for r in out["levels"])
    assert "ngs_direct" in out["stress_error_note"] and "ccx" in out["stress_error_note"] and "verdict" in out["stress_error_note"]
    assert out["converged"] in (True, False) and out["verdict"]            # the verdict does not depend on the estimate


def test_mcp_hotspots_say_why_the_error_estimate_is_missing():
    from types import SimpleNamespace

    from fea_core.mcp.diagnostics import no_error_estimate_note
    note = no_error_estimate_note(SimpleNamespace(info={"backend": "ngs_direct"}))
    assert "ngs_direct" in note and "backend='ccx'" in note

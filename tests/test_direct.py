"""Direct solver (fea_core.direct): the worker over synthetic SPD systems (needs the library rslab – skipped without it), the choice of the
precision from the memory plan, the acceptance check, failure kinds, the memory estimate and the solver protocol rows. No Gmsh, no ccx, no NGSolve.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import scipy.sparse as sp

from fea_core import direct
from fea_core.direct import worker as W

needs_rslab = pytest.mark.skipif(importlib.util.find_spec("rslab") is None, reason="the library rslab is not installed (module ngsolve)")


@pytest.fixture(autouse=True)
def _rslab_on_the_path_of_the_worker(monkeypatch):
    """The worker is a separate process: where rslab is found by this process (a development copy outside the site-packages), it must find
    it too (other tests import the installer, which clears PYTHONPATH of the process)."""
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        import os
        from pathlib import Path
        here = str(Path(spec.origin).parent.parent)
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))


def _system(n: int = 600, nrhs: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.01, random_state=seed, format="csr")
    K = (A @ A.T + sp.identity(n) * 5.0).tocsr()
    K.sort_indices()
    return K, rng.standard_normal((n, nrhs))


def _write(tmp_path, K, B, fmt="upper"):
    M = sp.triu(K, format="csr") if fmt == "upper" else K
    M.sort_indices()
    np.save(tmp_path / "indptr.npy", M.indptr.astype(np.int64))
    np.save(tmp_path / "indices.npy", M.indices.astype(np.int32))
    np.save(tmp_path / "data.npy", M.data)
    np.save(tmp_path / "rhs.npy", B)
    (tmp_path / "matrix.json").write_text(json.dumps({"format": fmt}), encoding="utf-8")


# ------------------------------------------------------------------ the worker over files
@needs_rslab
@pytest.mark.parametrize("fmt", ["upper", "full"])
def test_worker_float64_upper_and_full(tmp_path, fmt):
    K, B = _system()
    _write(tmp_path, K, B, fmt)
    X, info = direct.run_prepared(tmp_path, threads=2)
    assert X.shape == B.shape
    assert np.abs(K @ X - B).max() < 1e-10
    assert info["precision"] == "float64" and info["engine"] == "rslab" and info["format"] == fmt
    assert max(info["backward_error"]) < 1e-14 and info["inertia"] == [K.shape[0], 0, 0]
    assert info["nnz"] == (K.nnz + K.shape[0]) // 2 or fmt == "full"
    assert {"t_analyze", "t_factor", "t_solve", "plan_gb", "peak_gb", "free_gb", "factor_nnz", "fill_per_dof"} <= set(info)


@needs_rslab
def test_worker_float32_with_refinement(tmp_path):
    K, B = _system(800)
    _write(tmp_path, K, B)
    X, info = direct.run_prepared(tmp_path, threads=2, precision="float32")
    assert info["precision"] == "float32" and max(info["iterations"]) >= 1
    assert np.abs(K @ X - B).max() < 1e-8
    assert max(info["backward_error"]) < 1e-13


@needs_rslab
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_worker_load_state_without_load(tmp_path, precision):
    """A zero right-hand side (a load state without load, free thermal expansion T4) has the solution 0 – no NaN from 0/0 in the backward error
    or the refinement (T4 went to CalculiX through the fallback because of it)."""
    K, B = _system(500)
    B[:, 1] = 0.0
    _write(tmp_path, K, B)
    X, info = direct.run_prepared(tmp_path, threads=2, precision=precision)
    assert np.all(X[:, 1] == 0.0) and np.abs(K @ X - B).max() < 1e-8
    assert np.all(np.isfinite(info["backward_error"])) and info["backward_error"][1] == 0.0


@needs_rslab
def test_worker_singular_matrix_is_a_clear_error(tmp_path):
    K, B = _system(300)
    K = K.tolil()
    K[7, :] = 0
    K[:, 7] = 0
    K = K.tocsr()
    K.eliminate_zeros()
    K.sort_indices()
    _write(tmp_path, K, B)
    with pytest.raises(direct.DirectError, match="singular|not positive definite|pivot") as e:
        direct.run_prepared(tmp_path, threads=1)
    assert not isinstance(e.value, direct.DirectMemoryError)


@needs_rslab
def test_worker_not_positive_definite_is_an_error(tmp_path):
    n = 300
    K = sp.diags([np.full(n, 2.0)], [0], format="csr")
    K = K.tolil()
    K[5, 5] = -3.0
    K = K.tocsr()
    _write(tmp_path, K, np.ones((n, 2)))
    with pytest.raises(direct.DirectError, match="positive definite"):
        direct.run_prepared(tmp_path, threads=1)


@needs_rslab
def test_worker_that_does_not_fit_declines_before_factoring(tmp_path):
    """A system that does not fit into the free memory is declined after the analysis (exit code 3), nothing is factored."""
    K, B = _system(400)
    _write(tmp_path, K, B)
    (tmp_path / "direct_options.json").write_text(json.dumps({"engine": "rslab", "threads": 1, "reserve_gb": 1e6}), encoding="utf-8")
    assert W.main(tmp_path) == W.EXIT_NOFIT
    info = json.loads((tmp_path / "direct_info.json").read_text(encoding="utf-8"))
    assert info["error"]["kind"] == "memory" and "t_factor" not in info and not (tmp_path / "x.npy").exists()
    assert info["plan_gb"]["float32"] > 0 and info["factor_nnz"] > 0


@needs_rslab
def test_memory_guard_stops_the_worker_when_the_free_ram_drops(tmp_path, monkeypatch):
    """The caller watches the free RAM while the worker runs: below the reserve the worker is killed (no swapping) – DirectMemoryError."""
    K, B = _system(400)
    _write(tmp_path, K, B)
    monkeypatch.setattr(direct, "reserve_gb", lambda: 1e6)
    with pytest.raises(direct.DirectMemoryError, match="free memory of the computer fell") as e:
        direct.run_prepared(tmp_path, threads=1)
    # the message blames the free system memory (other programs may use it), not the solver
    assert "Other programs" in str(e.value) and "ran out of memory" not in str(e.value)


# ------------------------------------------------------------------ pieces without the library
def test_symmetric_operator_and_norm():
    K, _ = _system(400)
    op = W.SymmetricOperator(sp.triu(K, format="csr"))
    X = np.random.default_rng(3).standard_normal((400, 3))
    assert np.abs(op.matmat(X) - K @ X).max() < 1e-12
    assert op.norm1(block=90) == pytest.approx(float(abs(K).sum(axis=1).max()), rel=1e-12)


def test_backward_error_is_small_for_a_good_solution_and_large_for_a_bad_one():
    K, B = _system(300)
    op = W.SymmetricOperator(sp.triu(K, format="csr"))
    X = np.linalg.solve(K.toarray(), B)
    bwd, rel = W.accuracy(op, X, B, op.norm1())
    assert bwd.max() < 1e-14 and rel.max() < 1e-12
    bwd, rel = W.accuracy(op, X * 1.001, B, op.norm1())
    assert bwd.max() > 1e-6 and rel.max() > 1e-4


PLANS = {"float64": 4.0, "float32": 2.2}


def test_precision_float64_when_it_fits_and_the_factor_is_small():
    p, why = W.choose_precision("auto", 500.0, PLANS, avail_gb=10.0, reserve_gb=1.0)
    assert p == "float64" and "fits" in why


def test_precision_float32_for_a_large_factor_per_dof():
    p, why = W.choose_precision("auto", W.FILL_FLOAT32 + 1, PLANS, avail_gb=10.0, reserve_gb=1.0)
    assert p == "float32" and "faster" in why


def test_precision_float32_when_float64_does_not_fit():
    p, why = W.choose_precision("auto", 500.0, PLANS, avail_gb=5.0, reserve_gb=1.0)
    assert p == "float32" and "float64 needs" in why


def test_precision_nothing_fits_declines():
    with pytest.raises(W.WorkerFail) as e:
        W.choose_precision("auto", 500.0, PLANS, avail_gb=2.5, reserve_gb=1.0)
    assert e.value.code == W.EXIT_NOFIT and e.value.kind == "memory"
    with pytest.raises(W.WorkerFail):                                  # a requested precision that does not fit is not changed silently
        W.choose_precision("float64", 500.0, PLANS, avail_gb=5.0, reserve_gb=1.0)


def test_memory_estimate_matches_the_measurements_of_the_lab():
    """Float32 growth of the process measured 17. 11. 2026: plate 198 k DOF 0.92 GB, compact block 252 k 1.77 GB, 658 k plate 3.97 GB.
    The estimate before the analysis is conservative for the thin and plate-like models, close for the compact block."""
    assert 0.9 <= direct.estimate_gb(197_817) <= 1.6
    assert 1.2 <= direct.estimate_gb(252_207) <= 2.2
    assert 3.9 <= direct.estimate_gb(657_744) <= 6.5
    assert direct.estimate_gb(100_000, "float64") > direct.estimate_gb(100_000, "float32")


def test_fits_follows_the_free_memory(monkeypatch):
    monkeypatch.setattr(direct, "reserve_gb", lambda: 1.0)
    assert direct.fits(200_000, free_gb=8.0)
    assert not direct.fits(1_000_000, free_gb=4.0)


def test_engine_registry_rejects_an_unknown_engine():
    from fea_core.direct.engines import EngineError, make
    with pytest.raises(EngineError, match="Unknown direct solver engine"):
        make("mumps2", 3, np.zeros(4, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0), 1)


def test_protocol_rows_describe_the_solve():
    from fea_core.solver_protocol import brief, rows
    p = {"backend": "ngs_direct", "engine": "rslab 1.14.0", "reason": "linear study", "ndof": 197817, "nnz": 7_700_000,
         "matrix_format": "upper", "factor_nnz": 110_000_000, "fill_per_dof": 556.0, "precision": "float32",
         "precision_reason": "float64 needs 4.0 GB", "memory": {"estimate_gb": 1.3, "growth_gb": 1.25, "peak_gb": 1.6, "free_gb": 9.1},
         "times": {"assembly": 0.7, "analysis": 0.9, "factorization": 2.0, "solution": 1.0},
         "cases": [{"name": "A", "backward_error": 2e-16, "rel_residual": 1e-9, "iterations": 7},
                   {"name": "B", "backward_error": 3e-16, "rel_residual": 1e-9, "iterations": 9}]}
    r = dict(rows(p))
    assert r["Solver"] == "rslab 1.14.0" and r["Why this solver"] == "linear study"
    assert "197 817 DOF" in r["Size"] and "110 M entries" in r["Size"]
    assert r["Precision"] == "float32 factor, refined in float64 (7–9 iterations)" and r["Why this precision"] == "float64 needs 4.0 GB"
    assert "plan 1.3 GB" in r["Memory of the solver"] and "measured 1.2 GB" in r["Memory of the solver"]
    assert "peak of the process 1.6 GB" in r["Memory of the solver"]
    assert "factorization 2.0 s" in r["Time per phase"] and "3.0e-16" in r["Accuracy"]
    assert rows(None) == [] and brief(None) == {}
    assert brief(p)["backward_error_max"] == 3e-16 and "cases" not in brief(p)


def test_stored_name_ngs_pardiso_loads_as_ngs_direct():
    from fea_core.study import Study
    st = Study.from_dict({"name": "x", "geometry": "", "solver": {"backend": "ngs_pardiso"}})
    assert st.solver.backend == "ngs_direct"
    assert Study.from_dict({"name": "x", "geometry": "", "solver": {"backend": "ngs_hypre"}}).solver.backend == "ngs_hypre"


# ------------------------------------------------------------------ the automatic choice of the backend
class _Mesh:
    def __init__(self, nodes: int):
        self.nodes = np.zeros((nodes, 3))


def _study(backend="auto", n_cases=1):
    from fea_core.selectors import FaceSelector
    from fea_core.study import Fixed, Force, LoadCase, Material, Study
    st = Study("t", "m.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9))
    st.add_selection("dno", [FaceSelector("Dil", "plane", 1.0, [0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, 0, 1, 1, 0])])
    st.add_selection("vrch", [FaceSelector("Dil", "plane", 1.0, [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1, 1, 1, 1])])
    st.constraints.append(Fixed("dno"))
    st.load_cases = [LoadCase(f"LC{i + 1}", [Force("vrch", [0, 0, -100.0])]) for i in range(n_cases)]
    st.solver.backend = backend
    return st


def test_auto_choice_direct_hypre_ccx(monkeypatch):
    import fea_core.hypre as hypre
    from fea_core import solve
    mesh = _Mesh(70_000)                                       # 210 000 DOF
    monkeypatch.setattr(solve, "_direct_ready", lambda: True)
    monkeypatch.setattr(hypre, "available", lambda: True)
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: True)
    be, why = solve.choose_backend_reason(_study(n_cases=1), mesh)         # a single load case goes to the direct solver too
    assert be == "ngs_direct" and "one factorization" in why
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: False)    # does not fit into the free RAM → iterative
    be, why = solve.choose_backend_reason(_study(n_cases=3), mesh)
    assert be == "ngs_hypre" and "iterative" in why
    monkeypatch.setattr(hypre, "available", lambda: False)                 # … and without hypre → CalculiX
    be, why = solve.choose_backend_reason(_study(), mesh)
    assert be == "ccx" and "hypre is not installed" in why
    monkeypatch.setattr(solve, "_direct_ready", lambda: False)             # direct solver not installed: small → ccx, large → hypre
    assert solve.choose_backend_reason(_study(), mesh)[0] == "ccx"
    monkeypatch.setattr(hypre, "available", lambda: True)
    assert solve.choose_backend_reason(_study(), mesh)[0] == "ccx"
    assert solve.choose_backend_reason(_study(), _Mesh(250_000))[0] == "ngs_hypre"


def test_auto_choice_names_what_is_really_missing(monkeypatch):
    """The reason in the protocol says which component is missing – numba (module accel) is not "module direct"."""
    from fea_core import solve, solvers
    monkeypatch.delenv("FEA_ASSEMBLY", raising=False)
    mesh = _Mesh(1000)
    monkeypatch.setattr(solvers, "_has", lambda name: name != "numba")
    be, why = solve.choose_backend_reason(_study(), mesh)
    assert be == "ccx" and "missing numba (module accel)" in why and "module direct" not in why
    monkeypatch.setattr(solvers, "_has", lambda name: name != "rslab")
    be, why = solve.choose_backend_reason(_study(), mesh)
    assert be == "ccx" and "missing the direct solver library rslab (module direct)" in why
    monkeypatch.setattr(solvers, "_has", lambda name: True)
    assert solve._direct_ready()


def test_auto_choice_ccx_only_features_and_manual_choice(monkeypatch):
    from fea_core import hypre, solve
    from fea_core.joints.model import BoltSpec
    from fea_core.study import CylindricalSupport, RemoteSupport
    monkeypatch.setattr(solve, "_direct_ready", lambda: True)
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: True)
    # the rule is tested here, not the mesh: the pre-check of the constraints (`linear_constraints.check`, tests/test_constraint_check.py) needs a real mesh
    monkeypatch.setattr("fea_core.linear_constraints.check", lambda study, mesh, solver="direct": [])
    st = _study()
    st.analysis.kind = "nonlinear"                                         # contact / nonlinear analysis is ccx only
    be, why = solve.choose_backend_reason(st, _Mesh(1000))
    assert be == "ccx" and why.startswith("CalculiX only:") and "nonlinear" in why
    st = _study()
    st.bolts.append(BoltSpec("B1"))                                        # F13 step 1b part 2: bolts are beam elements of the own assembly (direct solver)
    assert solve.choose_backend_reason(st, _Mesh(1000))[0] == "ngs_direct"
    monkeypatch.setattr(hypre, "available", lambda: True)
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: False)    # … which the iterative solver cannot do: not hypre, ccx with the reason
    be, why = solve.choose_backend_reason(st, _Mesh(1000))
    assert be == "ccx" and "iterative solver cannot do" in why and "bolts" in why
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: True)
    st = _study()
    st.constraints.append(CylindricalSupport("vrch"))                      # F13 step 1b: the own assembly eliminates the equations of a cylindrical support
    assert solve.choose_backend_reason(st, _Mesh(1000))[0] == "ngs_direct"
    # a constraint that couples nodes is for the direct solver only: when it does not fit, hypre is NOT the next step – ccx is
    st.constraints.append(RemoteSupport("vrch", [0, 0, 5.0]))
    monkeypatch.setattr(hypre, "available", lambda: True)
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: False)
    be, why = solve.choose_backend_reason(st, _Mesh(1000))
    assert be == "ccx" and "iterative solver cannot do" in why
    monkeypatch.setattr(direct, "fits", lambda ndof, free=None: True)
    assert solve.choose_backend_reason(_study("ccx"), _Mesh(1000)) == ("ccx", "chosen by the user")
    assert solve.choose_backend_reason(_study("ngs_pardiso"), _Mesh(1000))[0] == "ngs_direct"       # the stored name of the earlier versions

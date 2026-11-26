"""Own matrix assembly (fea_core.assembly): the stiffness matrix of C3D10 against the stress recovery (the same 4 Gauss points of ccx), the thermal load,
the elimination of the constraints, determinism over the number of threads, the worker process. No Gmsh, no ccx, no solver library: the meshes are
structured tet10 grids with random distortion and curved edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("numba")

from fea_core.assembly import kernels as K                    # noqa: E402
from fea_core.assembly.system import (BlockMatrix, build_system, rigid_body_residual, thermal_load, AssemblyError)  # noqa: E402
from fea_core.recovery import MID_EDGES, _element_pass       # noqa: E402

CCX_MIDS = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]     # the 6 tetrahedra of a cube (conforming)


def grid_tet10(nx=3, ny=2, nz=2, h=1.0, jitter=0.15, curve=0.05, seed=1):
    """A structured grid of cubes split into 6 tet10 each; the corners are jittered, the midpoints of the edges are moved off the chord (curved edges)."""
    rng = np.random.default_rng(seed)
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    X = np.array([[i, j, k] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)], float) * h
    X += rng.uniform(-jitter * h, jitter * h, X.shape)
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i + (v & 1), j + ((v >> 1) & 1), k + ((v >> 2) & 1)) for v in range(8)]
                for t in KUHN:
                    tets.append([c[v] for v in t])
    tets = np.array(tets)
    vol = np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]), X[tets[:, 3]] - X[tets[:, 0]])
    flip = vol < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    mids, extra, conn = {}, [], np.empty((len(tets), 10), np.int64)
    conn[:, :4] = tets
    for e, t in enumerate(tets):
        for m, (a, b) in enumerate(CCX_MIDS):
            key = (min(t[a], t[b]), max(t[a], t[b]))
            if key not in mids:
                mids[key] = len(X) + len(extra)
                extra.append(0.5 * (X[key[0]] + X[key[1]]) + rng.uniform(-curve * h, curve * h, 3))
            conn[e, 4 + m] = mids[key]
    return np.vstack([X, np.array(extra)]), conn


def tables(ne, seed=2, parts=2):
    """Per-element material: part → an isotropic matrix, element → index."""
    Cm = np.stack([K.iso_stiffness(210000.0, 0.3), K.iso_stiffness(3500.0, 0.36)][:parts])
    return Cm, (np.arange(ne) % parts).astype(np.int32)


def recovery_forces(nodes, conn, U, Cm, mat_of, alpha_e=None, dT=None):
    """Nodal forces ∫Bᵀσ from the stress recovery (the independent implementation, numpy) – scattered to the nodes (N, 3)."""
    Ce = Cm[mat_of]
    dTe = None if dT is None else dT[conn]
    _, fe = _element_pass(nodes, conn, U[conn], Ce, alpha_e, dTe)
    f = np.zeros((len(nodes), 3))
    for c in range(3):
        f[:, c] = np.bincount(conn.ravel(), weights=fe[:, :, c].ravel(), minlength=len(nodes))
    return f


def test_gauss_points_and_shape_functions_are_those_of_the_core():
    from fea_core.loads import GAUSS4, W_GAUSS4, _tet10_dN
    from fea_core.results import tet10_shape
    assert np.array_equal(K.GAUSS4, GAUSS4) and K.W4 == W_GAUSS4
    for g in range(4):
        assert np.array_equal(K.DN4[g], _tet10_dN(*GAUSS4[g]))
        assert np.allclose(K.N4[g], tet10_shape(GAUSS4[g]), atol=1e-15, rtol=0)
    assert set(map(tuple, MID_EDGES.tolist())) == set(CCX_MIDS)


@pytest.mark.parametrize("upper", [True, False])
def test_k_times_u_equals_recovery_nodal_forces_on_curved_elements(upper):
    nodes, conn = grid_tet10()
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    s = build_system(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=upper)
    Ksys = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n, n))
    if upper:
        Ksys = Ksys + sp.triu(Ksys, 1).T
    U = np.random.default_rng(5).standard_normal((len(nodes), 3))
    f = recovery_forces(nodes, conn, U, Cm, mat_of)
    ref = (Ksys @ U.ravel()).reshape(-1, 3)
    assert np.abs(ref - f).max() / np.abs(f).max() < 1e-12
    assert s.info["rigid_body_residual"] < 1e-14


def test_upper_triangle_is_the_triangle_of_the_whole_matrix():
    nodes, conn = grid_tet10(2, 2, 2)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    args = (nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 2)))
    U = build_system(*args, upper=True)
    F = build_system(*args, upper=False)
    Ku = sp.csr_matrix((U.data, U.indices, U.indptr), shape=(n, n))
    Kf = sp.csr_matrix((F.data, F.indices, F.indptr), shape=(n, n))
    assert abs(Kf - Kf.T).max() < 1e-12 * abs(Kf).max()
    assert abs(sp.triu(Kf).tocsr() - Ku).max() < 1e-12 * abs(Kf).max()
    for M in (Ku, Kf):
        for r in range(0, n, 17):
            row = M.indices[M.indptr[r]:M.indptr[r + 1]]
            assert np.all(np.diff(row) > 0)
    assert U.indptr.dtype == np.int64 and U.indices.dtype == np.int32 and U.data.dtype == np.float64 and U.rhs.shape == (n, 2)


def test_matrix_is_bitwise_identical_for_any_number_of_threads():
    import numba
    nodes, conn = grid_tet10(4, 3, 3)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    rng = np.random.default_rng(3)
    fixed = rng.random(n) < 0.05
    g = np.where(fixed, rng.standard_normal(n), 0.0)
    dT = rng.standard_normal((len(nodes), 2))
    alpha = np.full(len(conn), 1.2e-5)
    keep = numba.get_num_threads()
    hashes = {}
    try:
        for th in (1, 2, 3, numba.config.NUMBA_NUM_THREADS):
            if th > numba.config.NUMBA_NUM_THREADS:
                continue
            numba.set_num_threads(th)
            for upper in (True, False):
                s = build_system(nodes, conn, mat_of, Cm, fixed, g, np.ones((n, 2)), upper=upper, dT=dT, alpha_e=alpha)
                hashes.setdefault(upper, set()).add((s.data.tobytes(), s.indices.tobytes(), s.indptr.tobytes(), s.rhs.tobytes()))
    finally:
        numba.set_num_threads(keep)
    assert all(len(v) == 1 for v in hashes.values())


@pytest.mark.parametrize("prune", [False, True])
def test_constraints_are_eliminated_like_hypre_apply_dirichlet(prune):
    from fea_core.hypre import apply_dirichlet
    nodes, conn = grid_tet10(3, 2, 2)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    rng = np.random.default_rng(7)
    fixed = rng.random(n) < 0.08
    g = np.where(fixed, rng.standard_normal(n) * 0.01, 0.0)
    F = rng.standard_normal((n, 3))
    free0 = build_system(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=False)
    K0 = sp.csr_matrix((free0.data, free0.indices, free0.indptr), shape=(n, n))
    Kref, Fref = apply_dirichlet(K0, F, fixed, g)
    for upper in (False, True):
        s = build_system(nodes, conn, mat_of, Cm, fixed, g, F, upper=upper, prune=prune)
        Ks = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n, n))
        if upper:
            Ks = Ks + sp.triu(Ks, 1).T
        assert abs(Ks - Kref).max() < 1e-12 * abs(Kref).max()
        assert np.abs(s.rhs - Fref).max() < 1e-11 * np.abs(Fref).max()
        # a constrained row has the diagonal d and zeros (the pattern of the matrix stays) – or, pruned, nothing else
        for r in np.flatnonzero(fixed)[:10]:
            cols = list(s.indices[s.indptr[r]:s.indptr[r + 1]])                     # the arrays of the system, not the sums above (they drop zeros)
            vals = s.data[s.indptr[r]:s.indptr[r + 1]]
            assert r in cols and vals[cols.index(r)] == s.d and np.count_nonzero(vals) == 1
            ref = [c for c in K0.indices[K0.indptr[r]:K0.indptr[r + 1]] if c >= r or not upper]
            assert cols == ([r] if prune else sorted(ref))
        assert s.info["pruned"] is prune
        assert s.d == pytest.approx(float(K0.diagonal()[~fixed].mean()), rel=1e-14)


def test_thermal_load_matches_the_recovery_kernel():
    nodes, conn = grid_tet10(3, 2, 2)
    Cm, mat_of = tables(len(conn))
    rng = np.random.default_rng(11)
    dT = rng.uniform(-30, 80, len(nodes))
    alpha = np.where(mat_of == 0, 1.2e-5, 7.0e-5)
    f = thermal_load(nodes, conn.astype(np.int32), mat_of, Cm, alpha, dT)
    # the recovery gives the internal forces ∫Bᵀ C (ε − α ΔT) for ε = 0 → minus the thermal load
    ref = -recovery_forces(nodes, conn, np.zeros((len(nodes), 3)), Cm, mat_of, alpha, dT)
    assert np.abs(f - ref).max() / np.abs(ref).max() < 1e-12
    # the free thermal expansion of a uniform ΔT with equal α is a state without stress: K·(α ΔT x) = f
    n = 3 * len(nodes)
    s = build_system(nodes, conn, np.zeros(len(conn), np.int32), Cm[:1], np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True,
                     dT=np.full((len(nodes), 1), 25.0), alpha_e=np.full(len(conn), 1.2e-5))
    u = (1.2e-5 * 25.0 * nodes).ravel()
    Kb = BlockMatrix(len(nodes), *_block(nodes, conn, np.zeros(len(conn), np.int32), Cm[:1]))
    assert np.abs(Kb.matmat(u) - s.rhs[:, 0]).max() < 1e-9 * np.abs(s.rhs).max()


def _block(nodes, conn, mat_of, Cm):
    conn = conn.astype(np.int32)
    ptr, el, loc = K.node_elements(len(nodes), conn)
    indptr, indices = K.pattern(len(nodes), conn, ptr, el, True)
    data = np.zeros((len(indices), 3, 3))
    K.stiffness(np.ascontiguousarray(nodes), conn, mat_of, Cm, ptr, el, loc, indptr, indices, K.DN4, True, data)
    return indptr, indices, data, True


def test_per_element_material_matrices_of_printed_parts():
    """The anisotropic C of every element (3D printing: shell and core with the orientation of the lines) = an index into a table of 6×6 matrices."""
    nodes, conn = grid_tet10(3, 2, 2)
    rng = np.random.default_rng(13)
    tab = []
    for _ in range(5):
        A = rng.standard_normal((6, 6))
        tab.append(A @ A.T + 6 * np.eye(6))
    Cm = np.stack(tab) * 1000.0
    mat_of = rng.integers(0, 5, len(conn)).astype(np.int32)
    n = 3 * len(nodes)
    s = build_system(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)
    Ks = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n, n))
    Ks = Ks + sp.triu(Ks, 1).T
    U = rng.standard_normal((len(nodes), 3))
    f = recovery_forces(nodes, conn, U, Cm, mat_of)
    assert np.abs((Ks @ U.ravel()).reshape(-1, 3) - f).max() / np.abs(f).max() < 1e-12


def test_node_without_element_is_constrained_not_a_singular_row():
    nodes, conn = grid_tet10(2, 1, 1)
    nodes = np.vstack([nodes, [[9.0, 9.0, 9.0]]])                 # an orphan node
    Cm, mat_of = tables(len(conn), parts=1)
    n = 3 * len(nodes)
    fixed = np.zeros(n, bool)
    fixed[:3] = True
    s = build_system(nodes, conn, mat_of, Cm, fixed, np.zeros(n), np.ones((n, 1)), upper=True)
    assert s.info["orphan_nodes"] == 1
    K_ = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n, n))
    assert K_.diagonal()[-3:].min() == pytest.approx(s.d) and s.rhs[-3:].max() == 0.0


def test_invalid_inputs_are_clear_errors():
    nodes, conn = grid_tet10(1, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    with pytest.raises(AssemblyError, match="tet10"):
        build_system(nodes, conn[:, :4], mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)
    with pytest.raises(AssemblyError, match="material"):
        build_system(nodes, conn, mat_of + 5, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)
    with pytest.raises(AssemblyError, match="constrained"):
        build_system(nodes, conn, mat_of, Cm, np.ones(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)
    with pytest.raises(AssemblyError, match="expansion"):
        build_system(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True, dT=np.ones((len(nodes), 1)))


def test_rigid_body_residual_detects_a_wrong_matrix():
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn), parts=1)
    indptr, indices, data, upper = _block(nodes, conn, mat_of, Cm)
    Kb = BlockMatrix(len(nodes), indptr, indices, data, upper)
    assert rigid_body_residual(Kb, nodes) < 1e-14
    data[3, 0, 1] *= 1.01                                       # a wrong entry of the matrix
    assert rigid_body_residual(Kb, nodes) > 1e-9          # the working limit of the solver is 1e-8; a good matrix gives 1e-18


def test_block_matmat_symmetric_product_equals_the_full_matrix():
    nodes, conn = grid_tet10(3, 2, 2)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    sf = build_system(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=False)
    Kf = sp.csr_matrix((sf.data, sf.indices, sf.indptr), shape=(n, n))
    Ku = BlockMatrix(len(nodes), *_block(nodes, conn, mat_of, Cm))
    M = np.random.default_rng(1).standard_normal((n, 4))
    assert np.abs(Ku.matmat(M) - Kf @ M).max() < 1e-10 * np.abs(Kf @ M).max()
    assert Ku.matmat(M[:, 0]).shape == (n,)


# ------------------------------------------------------------------ the worker process
def _job(tmp_path, nodes, conn, mat_of, Cm, fixed, g, F, dT=None, alpha=None, printing=False):
    np.save(tmp_path / "nodes.npy", nodes)
    np.save(tmp_path / "tets.npy", conn.astype(np.int32))
    parts = (mat_of if not printing else np.zeros(len(conn))).astype(np.int32)
    np.save(tmp_path / "tet_part.npy", parts)
    mats = [{"E": 210000.0, "nu": 0.3, "alpha": 1.2e-5}, {"E": 3500.0, "nu": 0.36, "alpha": 7.0e-5}]
    (tmp_path / "materials.json").write_text(json.dumps(mats), encoding="utf-8")
    if printing:
        np.save(tmp_path / "C_tab.npy", Cm)
        np.save(tmp_path / "C_idx.npy", mat_of)
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "fixed.npy", fixed)
    np.save(tmp_path / "g.npy", g)
    if dT is not None:
        np.save(tmp_path / "dT.npy", dT)


@pytest.mark.parametrize("solver", ["direct", "hypre"])
def test_worker_process_writes_the_files_of_the_solver(tmp_path, solver):
    from fea_core import assembly
    nodes, conn = grid_tet10(3, 2, 2)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    rng = np.random.default_rng(17)
    fixed = rng.random(n) < 0.05
    g = np.where(fixed, 0.001 * rng.standard_normal(n), 0.0)
    F = rng.standard_normal((n, 2))
    dT = rng.uniform(0, 50, (len(nodes), 2))
    dT[:, 1] = 0.0
    _job(tmp_path, nodes, conn, mat_of, Cm, fixed, g, F, dT)
    notes = []
    info = assembly.run_worker(tmp_path, solver, threads=2, on_notice=notes.append)
    assert info["assembler"] == "own" and info["format"] == ("upper" if solver == "direct" else "full") and info["rigid_body_residual"] < 1e-12
    assert {"t_import", "t_compile", "t_read", "t_assemble", "t_write", "t_total", "times"} <= set(info)
    alpha = np.where(mat_of == 0, 1.2e-5, 7.0e-5)
    ref = build_system(nodes, conn, mat_of, Cm, fixed, g, F, upper=(solver == "direct"), dT=dT, alpha_e=alpha)
    d = tmp_path / solver
    assert np.array_equal(np.load(d / "indptr.npy"), ref.indptr) and np.array_equal(np.load(d / "indices.npy"), ref.indices)
    assert np.array_equal(np.load(d / "data.npy"), ref.data) and np.array_equal(np.load(d / "rhs.npy"), ref.rhs)
    if solver == "direct":
        assert json.loads((d / "matrix.json").read_text(encoding="utf-8"))["format"] == "upper"
    else:
        assert np.load(d / "rbm.npy").shape == (n, 3)
    assert info["nnz"] == len(ref.data) and all(not x.startswith(">>") for x in notes)


def test_worker_with_a_table_of_matrices_for_printing(tmp_path):
    from fea_core import assembly
    nodes, conn = grid_tet10(2, 2, 1)
    rng = np.random.default_rng(19)
    tab = np.stack([K.iso_stiffness(2000.0 + 100 * i, 0.35) for i in range(4)])
    mat_of = rng.integers(0, 4, len(conn)).astype(np.int32)
    n = 3 * len(nodes)
    _job(tmp_path, nodes, conn, mat_of, tab, np.zeros(n, bool), np.zeros(n), np.ones((n, 1)), printing=True)
    assembly.run_worker(tmp_path, "direct", threads=1)
    ref = build_system(nodes, conn, mat_of, tab, np.zeros(n, bool), np.zeros(n), np.ones((n, 1)), upper=True)
    assert np.array_equal(np.load(tmp_path / "direct" / "data.npy"), ref.data)


def test_worker_failure_is_reported_with_the_log(tmp_path):
    from fea_core import assembly
    with pytest.raises(RuntimeError, match="assembly of the stiffness matrix failed"):
        assembly.run_worker(tmp_path, "direct", threads=1)           # no input files


def test_worker_can_be_cancelled(tmp_path):
    import threading
    from fea_core import assembly
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    _job(tmp_path, nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.ones((n, 1)))
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="canceled"):
        assembly.run_worker(tmp_path, "direct", threads=1, cancel=stop)


def test_assembler_choice_and_availability(monkeypatch):
    from fea_core import assembly
    monkeypatch.delenv(assembly.ASSEMBLER_ENV, raising=False)
    assert assembly.assembler() == "own" and assembly.available()
    monkeypatch.setenv(assembly.ASSEMBLER_ENV, "NGSolve")
    assert assembly.assembler() == "ngsolve"
    monkeypatch.setenv(assembly.ASSEMBLER_ENV, "whatever")
    assert assembly.assembler() == "own"


def test_the_package_imports_nothing_heavy():
    import subprocess
    import sys
    from fea_core import config
    code = ("import sys, fea_core.assembly; "
            "bad = [m for m in ('numba', 'scipy', 'llvmlite', 'ngsolve') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=config.PROJECT_ROOT, capture_output=True, text=True, env=config.child_env())
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.slow
def test_first_use_compilation_is_announced_to_the_log(tmp_path, monkeypatch):
    """With an empty numba cache the worker compiles its kernels (about 20 s): the caller gets a notice early, so that the progress bar of the
    calculation does not look stuck; with a warm cache there is none (the other worker tests)."""
    from fea_core import assembly
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "cold_cache"))
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    _job(tmp_path, nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.ones((n, 1)))
    notes = []
    info = assembly.run_worker(tmp_path, "direct", threads=2, on_notice=notes.append)
    assert len(notes) == 1 and notes[0].startswith("Compiling the assembly kernel") and "first use" in notes[0]
    assert info["t_compile"] > 2.0
    notes.clear()                                       # the second run loads the kernels from the cache
    info2 = assembly.run_worker(tmp_path, "direct", threads=2, on_notice=notes.append)
    assert notes == [] and info2["t_compile"] < info["t_compile"] / 2


@pytest.mark.parametrize("curve, thermal", [(0.10, False), (0.0, True)])
def test_system_equals_the_ngsolve_assembly(tmp_path, curve, thermal):
    """The reference of the earlier versions: the NGSolve worker (FEA_ASSEMBLY=ngsolve) assembles the same system from the same job files. The
    stiffness matrix agrees to rounding errors also on tet10 with strongly curved edges (up to 16 % of the edge length off the chord; with 12 % the random test mesh
    contains an inverted element, which the assembly now rejects like ccx):
    NGSolve integrates the elasticity of the P2 element with the same 4-point rule as ccx. The thermal load is compared on straight-edged elements:
    on curved ones NGSolve integrates it with a higher order than ccx (and than `recovery`, which gives the stress), the difference is ∝ the
    curvature (midpoints moved by 10 % of the element size, the most one up to 16 % off the chord: below 0.4 %) – the own assembly uses the rule of ccx (thermal case T5: 2.6e-6 / 1.1e-5 against ccx, NGSolve 2.6e-5 / 3.8e-4)."""
    import importlib.util
    import subprocess
    import sys

    from fea_core import config
    from fea_core.mesh import Mesh
    if importlib.util.find_spec("ngsolve") is None:
        pytest.skip("NGSolve (module ngsolve) is not installed")
    nodes, conn = grid_tet10(3, 2, 2, curve=curve)
    part = (np.arange(len(conn)) % 2).astype(np.int64)
    Cm, _ = tables(len(conn))
    n = 3 * len(nodes)
    rng = np.random.default_rng(23)
    fixed = rng.random(n) < 0.06
    g = np.where(fixed, 0.002 * rng.standard_normal(n), 0.0)
    F = rng.standard_normal((n, 2))
    # a linear and a constant temperature change: with the linear one the integrand has the degree 2, which the 4-point rule integrates exactly on
    # straight elements (a random field would show the difference of the quadrature rules, also on straight elements)
    dT = np.column_stack([10 + 0.5 * nodes[:, 0] - 0.3 * nodes[:, 1] + 0.2 * nodes[:, 2], np.full(len(nodes), 25.0)]) if thermal else None
    alpha = np.where(part == 0, 1.2e-5, 7.0e-5)
    own = build_system(nodes, conn, part, Cm, fixed, g, F, upper=True, dT=dT, alpha_e=alpha if thermal else None)
    Mesh(nodes, conn.astype(np.int64), part, ["A", "B"], {}, {}).save(tmp_path / "mesh.npz")
    (tmp_path / "materials.json").write_text(json.dumps([{"E": 210000.0, "nu": 0.3, "alpha": 1.2e-5}, {"E": 3500.0, "nu": 0.36, "alpha": 7.0e-5}]),
                                             encoding="utf-8")
    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "fixed.npy", fixed)
    np.save(tmp_path / "g.npy", g)
    if thermal:
        np.save(tmp_path / "dT.npy", dT)
    r = subprocess.run([sys.executable, "-m", "fea_core.ngs.worker", "assemble", str(tmp_path), "direct"], cwd=config.PROJECT_ROOT,
                       capture_output=True, text=True, env=config.child_env())
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-1500:]
    d = tmp_path / "direct"
    Kn = sp.csr_matrix((np.load(d / "data.npy"), np.load(d / "indices.npy"), np.load(d / "indptr.npy")), shape=(n, n))
    Ko = sp.csr_matrix((own.data, own.indices, own.indptr), shape=(n, n))
    assert abs(Kn - Ko).max() < 1e-12 * abs(Kn).max()
    rn = np.load(d / "rhs.npy")
    assert np.abs(rn - own.rhs).max() < 1e-10 * np.abs(rn).max()      # the loads and the prescribed displacements (and the thermal load, straight elements)


# ------------------------------------------------------------------ the glue in solve_large
def test_assemble_glue_maps_failures_to_solver_errors(tmp_path, monkeypatch):
    """`solve_large._assemble`: a cancelled / failed worker and a bad rigid-body check become LargeSolverError (what `solve._run_mechanics` expects);
    the notices of the worker go to the log; the NGSolve fallback is used only with FEA_ASSEMBLY=ngsolve."""
    from fea_core import assembly, solve_large
    monkeypatch.delenv(assembly.ASSEMBLER_ENV, raising=False)
    logs = []

    def cancelled(job, solver, threads=None, cancel=None, on_notice=None, hom_cols=0):
        raise RuntimeError("Analysis canceled by user")
    monkeypatch.setattr(assembly, "run_worker", cancelled)
    with pytest.raises(solve_large.LargeSolverError, match="canceled by user"):
        solve_large._assemble(tmp_path, "direct", logs.append, None, 2)
    assert logs == ["Assembling stiffness matrix"]

    def fine(job, solver, threads=None, cancel=None, on_notice=None, hom_cols=0):
        on_notice("Compiling the assembly kernel (first use after the installation or an update, about 20 s – later runs load it from the cache)")
        return {"rigid_body_residual": 1e-18, "ndof": 3, "nnz": 6, "format": "upper", "assembler": "own", "threads": 2}
    monkeypatch.setattr(assembly, "run_worker", fine)
    logs.clear()
    info = solve_large._assemble(tmp_path, "direct", logs.append, None, 2)
    assert info["assembler"] == "own" and logs[1].startswith("Compiling the assembly kernel")
    assert solve_large._assembler_text(info) == "own (numba, 2 threads)" and solve_large._times_of_assembly(info)["compile"] is None

    monkeypatch.setattr(assembly, "run_worker", lambda *a, **k: {**fine(None, None, on_notice=lambda s: None), "rigid_body_residual": 1e-3})
    with pytest.raises(solve_large.LargeSolverError, match="rigid-body"):
        solve_large._assemble(tmp_path, "hypre", logs.append, None)

    calls = []
    monkeypatch.setenv(assembly.ASSEMBLER_ENV, "ngsolve")
    monkeypatch.setattr(solve_large, "_worker", lambda stage, job, log, backend, cancel, extra=(): calls.append((stage, extra)))
    (tmp_path / "assemble.json").write_text(json.dumps({"rigid_body_residual": 1e-18, "ndof": 3, "nnz": 6}), encoding="utf-8")
    info = solve_large._assemble(tmp_path, "direct", logs.append, None)
    assert calls == [("assemble", ("direct",))] and info["assembler"] == "ngsolve" and solve_large._assembler_text(info) == "NGSolve"


# ------------------------------------------------------------------ the check of the elements (the rule of ccx: Jacobian > 0 at every integration point)
def _one_element():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5], [.5, 0, .5], [0, .5, .5]], float)
    return nodes, np.arange(10, dtype=np.int32)[None, :]


def _free_system(nodes, tets, **kw):
    n = 3 * len(nodes)
    return build_system(nodes, tets, np.zeros(len(tets), np.int32), K.iso_stiffness(1000.0, 0.3)[None], np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)),
                        upper=True, **kw)


def test_invalid_elements_are_rejected_like_ccx_does():
    """An inverted element (reversed node order), a flat one (zero volume) and one with a NaN coordinate are `ElementError`s that name the element –
    the earlier code took |det| for the first, and the zero-volume / NaN matrix passed the rigid-body check (`scale > 0` is False for NaN → 0.0).
    A sliver with a tiny positive determinant passes, as in ccx (experiment: scratch/review_asm/ccx_inverted.py)."""
    from fea_core.assembly import ElementError
    nodes, tets = _one_element()
    assert _free_system(nodes, tets).info["rigid_body_residual"] < 1e-14
    nan_nodes = nodes.copy()
    nan_nodes[3, 2] = np.nan
    cases = {"inverted": (nodes, tets[:, [0, 1, 3, 2, 4, 8, 7, 6, 5, 9]]),
             "flat": (nodes * [1.0, 1.0, 0.0], tets),
             "nan": (nan_nodes, tets)}
    for name, (x, t) in cases.items():
        with pytest.raises(ElementError, match="not positive and finite") as e:
            _free_system(x, np.ascontiguousarray(t))
        assert e.value.elements == [0] and e.value.count == 1 and " 1" in str(e.value), name
    assert _free_system(nodes * [1.0, 1.0, 1e-9], tets).info["rigid_body_residual"] < 1e-9              # a sliver: ccx accepts it too


def test_many_invalid_elements_are_counted_and_the_list_is_capped():
    from fea_core.assembly import ElementError
    nodes, conn = grid_tet10(4, 3, 3)
    Cm, mat_of = tables(len(conn))
    bad = conn.copy()
    bad[:30] = bad[:30][:, [0, 1, 3, 2, 4, 8, 7, 6, 5, 9]]
    n = 3 * len(nodes)
    with pytest.raises(ElementError) as e:
        build_system(nodes, bad, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)
    assert e.value.count == 30 and e.value.elements == list(range(20))
    assert "30 element(s)" in str(e.value) and "1, 2, 3, 4, 5, 6, 7, 8" in str(e.value)


def test_a_valid_curved_grid_passes_the_check_and_the_determinants_are_positive():
    nodes, conn = grid_tet10(3, 2, 2, curve=0.10)               # strongly curved edges (the midpoints up to 16 % of the edge off the chord)
    j = K.min_jacobian(np.ascontiguousarray(nodes), conn.astype(np.int32), K.DN4)
    assert (j > 0).all() and j.min() == pytest.approx(0.126, abs=0.01)


def test_the_rigid_body_check_cannot_be_fooled_by_nan_or_a_zero_matrix():
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn), parts=1)
    indptr, indices, data, upper = _block(nodes, conn, mat_of, Cm)
    good = BlockMatrix(len(nodes), indptr, indices, data.copy(), upper)
    assert rigid_body_residual(good, nodes) < 1e-14
    nan = data.copy()
    nan[2, 0, 0] = np.nan
    assert rigid_body_residual(BlockMatrix(len(nodes), indptr, indices, nan, upper), nodes) == float("inf")
    assert rigid_body_residual(BlockMatrix(len(nodes), indptr, indices, np.zeros_like(data), upper), nodes) == float("inf")      # no stiffness at all
    n = 3 * len(nodes)
    with pytest.raises(AssemblyError, match="zero or contains NaN"):
        build_system(nodes, conn, mat_of, Cm * 0.0, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)


def test_other_invalid_input_is_a_clear_error_not_a_crash():
    nodes, conn = grid_tet10(1, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    rest = (np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)))
    with pytest.raises(AssemblyError, match="does not exist"):                       # the kernels do not check the indices: an index outside would crash
        build_system(nodes, conn + len(nodes), mat_of, Cm, *rest, upper=True)
    with pytest.raises(AssemblyError, match="does not exist"):
        build_system(nodes, conn - 1, mat_of, Cm, *rest, upper=True)
    bad_c = Cm.copy()
    bad_c[0, 0, 0] = np.nan
    with pytest.raises(AssemblyError, match="material matrices"):
        build_system(nodes, conn, mat_of, bad_c, *rest, upper=True)
    orphan = np.vstack([nodes, [[np.nan, 0, 0]]])                                    # a node outside the elements with damaged coordinates
    n2 = 3 * len(orphan)
    with pytest.raises(AssemblyError, match="coordinates"):
        build_system(orphan, conn, mat_of, Cm, np.zeros(n2, bool), np.zeros(n2), np.zeros((n2, 1)), upper=True)


def test_worker_reports_invalid_elements_as_an_element_error(tmp_path):
    """A mesh problem is not a crash of the process: the worker exits with its own code and a file, `run_worker` raises ElementError (the parts are named by
    solve_large), no traceback in the message."""
    from fea_core import assembly
    nodes, conn = grid_tet10(3, 2, 2)
    Cm, mat_of = tables(len(conn))
    conn = conn.copy()
    conn[[7, 40]] = conn[[7, 40]][:, [0, 1, 3, 2, 4, 8, 7, 6, 5, 9]]
    n = 3 * len(nodes)
    _job(tmp_path, nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.ones((n, 1)))
    with pytest.raises(assembly.ElementError) as e:
        assembly.run_worker(tmp_path, "direct", threads=1)
    assert e.value.elements == [7, 40] and e.value.count == 2 and "Traceback" not in str(e.value)
    assert (tmp_path / assembly.ERROR_FILE).exists() and not (tmp_path / "assemble.json").exists()
    assert not (tmp_path / "direct" / "data.npy").exists()                         # nothing half-written is left for the solver


def test_worker_reports_an_unusable_input_as_a_message_not_a_traceback(tmp_path):
    from fea_core import assembly
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    _job(tmp_path, nodes, conn, mat_of, Cm, np.ones(n, bool), np.zeros(n), np.ones((n, 1)))          # every degree of freedom constrained
    with pytest.raises(assembly.AssemblyError, match="All degrees of freedom are constrained") as e:
        assembly.run_worker(tmp_path, "direct", threads=1)
    assert not isinstance(e.value, assembly.ElementError) and "Traceback" not in str(e.value)


def test_a_failure_while_writing_is_not_masked_by_the_count_check(tmp_path, monkeypatch):
    from fea_core.assembly.system import Assembled, assemble
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    asm = assemble(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=True)

    def boom(self, a0, a1):
        raise MemoryError("the chunk does not fit")
    monkeypatch.setattr(Assembled, "fill", boom)
    with pytest.raises(MemoryError, match="does not fit"):
        asm.write(tmp_path / "x")


def test_streamed_system_equals_the_system_in_memory(tmp_path):
    """The worker writes the scalar CSR chunk by chunk (the block matrix is the only large array): the files are the same as the arrays of the system,
    for the upper triangle and the whole matrix, with and without pruning, with chunks of a few hundred entries."""
    from fea_core.assembly.system import assemble
    nodes, conn = grid_tet10(3, 2, 2)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    rng = np.random.default_rng(31)
    fixed = rng.random(n) < 0.07
    g = np.where(fixed, 0.01 * rng.standard_normal(n), 0.0)
    F = rng.standard_normal((n, 2))
    dT = rng.uniform(0, 40, (len(nodes), 2))
    alpha = np.full(len(conn), 1.2e-5)
    for upper in (True, False):
        for prune in (False, True):
            args = (nodes, conn, mat_of, Cm, fixed, g, F)
            ref = build_system(*args, upper=upper, dT=dT, alpha_e=alpha, prune=prune)
            asm = assemble(*args, upper=upper, dT=dT, alpha_e=alpha, prune=prune)
            ranges = asm.node_ranges(300)
            assert ranges[0][0] == 0 and ranges[-1][1] == len(nodes) and all(a1 == b0 for (_, a1), (b0, _) in zip(ranges, ranges[1:])) and len(ranges) > 5
            d = tmp_path / f"u{int(upper)}p{int(prune)}"
            t = asm.write(d, chunk_entries=300)
            assert set(t) == {"convert", "write"} and asm.info["times"]["convert"] >= t["convert"] * 0.99
            assert np.array_equal(np.load(d / "indptr.npy"), ref.indptr) and np.load(d / "indptr.npy").dtype == np.int64
            assert np.array_equal(np.load(d / "indices.npy"), ref.indices) and np.load(d / "indices.npy").dtype == np.int32
            assert np.array_equal(np.load(d / "data.npy"), ref.data) and np.load(d / "data.npy").dtype == np.float64
            assert np.array_equal(np.load(d / "rhs.npy"), ref.rhs)
            assert np.load(d / "data.npy", mmap_mode="r").shape == (ref.info["nnz"],)          # the header is the one of np.save (a memmap reads it)


def test_worker_records_the_peak_memory_and_a_stale_error_file_is_removed(tmp_path):
    from fea_core import assembly
    nodes, conn = grid_tet10(2, 1, 1)
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    _job(tmp_path, nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.ones((n, 1)))
    (tmp_path / assembly.ERROR_FILE).write_text("{}", encoding="utf-8")                  # a leftover of an earlier run
    info = assembly.run_worker(tmp_path, "hypre", threads=1)
    assert info["peak_gb"] > 0 and not (tmp_path / assembly.ERROR_FILE).exists()


def test_a_broken_or_missing_numba_gets_the_install_hint_not_the_log_tail(tmp_path, monkeypatch):
    from fea_core import assembly, procs
    broken = ('Traceback (most recent call last):\n  File "C:\\app\\.venv\\Lib\\site-packages\\numba\\core\\typing\\templates.py", line 5, in <module>\n'
              '    from numba import _dynfunc\nImportError: DLL load failed while importing _dynfunc: The specified module could not be found.\n')
    missing = "Traceback (most recent call last):\n  File \"x.py\", line 3\nModuleNotFoundError: No module named 'numba'\n"
    other = "Traceback (most recent call last):\n  File \"x.py\", line 3\nZeroDivisionError: division by zero\n"
    for log, needles in ((broken, ("installed but cannot be loaded", "DLL load failed", "numba (module accel)", "--force")),
                         (missing, ("missing numba (module accel)", "install accel")),
                         (other, ("assembly of the stiffness matrix failed", "ZeroDivisionError"))):
        def fake(cmd, log_path, cancel=None, env=None, on_line=None, _log=log, **k):
            Path(log_path).write_text(_log, encoding="utf-8")
            return 1
        monkeypatch.setattr(procs, "run_tree", fake)
        with pytest.raises(RuntimeError) as e:
            assembly.run_worker(tmp_path, "direct", threads=1)
        assert all(x in str(e.value) for x in needles), str(e.value)
        assert log is other or "Traceback" not in str(e.value)


def test_availability_follows_the_chosen_assembly(monkeypatch):
    """numba is needed by the own assembly only; the fallback FEA_ASSEMBLY=ngsolve needs NGSolve (and no numba)."""
    import importlib.util
    from fea_core import assembly
    monkeypatch.delenv(assembly.ASSEMBLER_ENV, raising=False)
    present = {"numba": True, "ngsolve": False}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object() if present.get(name) else None)
    assert assembly.available()
    present["numba"] = False
    assert not assembly.available()
    monkeypatch.setenv(assembly.ASSEMBLER_ENV, "ngsolve")
    assert not assembly.available()
    present["ngsolve"] = True
    assert assembly.available()

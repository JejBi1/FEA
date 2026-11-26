"""The constraints of the own assembly (F13 step 1b): the elimination of dependent DOFs (u_D = A·u + c, TᵀKT), extra nodes of rigid couplings, springs, orthotropic
materials rotated at the Gauss points, the *TIE projection. Against DENSE reference computations (the transformation matrices written out), against the
independent numpy implementation of the stress recovery, against tensor rotations. No Gmsh, no ccx, no solver library: structured tet10 grids.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

pytest.importorskip("numba")

from fea_core import ortho                                           # noqa: E402
from fea_core.assembly import kernels as K                           # noqa: E402
from fea_core.assembly.reduction import Reduction, ReductionError    # noqa: E402
from fea_core.assembly.system import Aniso, AssemblyError, assemble, build_system     # noqa: E402
from fea_core.recovery import _element_pass                          # noqa: E402
from tests.test_assembly import grid_tet10, tables                   # noqa: E402


def full_matrix(nodes, conn, Cm, mat_of):
    """The whole matrix K of the elements (no constraints) as a dense array."""
    n = 3 * len(nodes)
    s = build_system(nodes, conn, mat_of, Cm, np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=False)
    return sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n, n)).toarray()


def solve_system(system, upper):
    """The solution of an assembled system (scipy; the diagonal blocks of the eliminated DOFs are decoupled)."""
    n = len(system.indptr) - 1
    Kr = sp.csr_matrix((system.data, system.indices, system.indptr), shape=(n, n))
    if upper:
        Kr = Kr + sp.triu(Kr, 1).T
    return spl.spsolve(Kr.tocsc(), system.rhs).reshape(n, -1)


def dense_solution(Kd, F, fixed, g, dep, A, c, extra_fixed=None):
    """The reduced problem written out: u = P·w + u_p, w = the free DOFs, (PᵀKP)·w = Pᵀ(F − K·u_p). Kd is the whole matrix incl. the extra DOFs."""
    n = len(Kd)
    isfix = fixed.copy()
    Fr = np.setdiff1d(np.arange(n), np.concatenate([np.nonzero(isfix)[0], dep]))
    P = np.zeros((n, len(Fr)))
    P[Fr, np.arange(len(Fr))] = 1.0
    P[dep] = A[:, Fr]
    u_p = np.zeros(n)
    u_p[isfix] = g[isfix]
    u_p[dep] = A[:, isfix] @ g[isfix] + c
    w = np.linalg.solve(P.T @ Kd @ P, P.T @ (F - (Kd @ u_p)[:, None]))
    return P @ w + u_p[:, None]


@pytest.mark.parametrize("upper", [True, False])
def test_the_elimination_equals_the_dense_transformation(upper):
    """Local equations of one node, a tie-like equation of a node from three others, an offset from a prescribed value, a spring: the reduced system solved
    and expanded = the dense TᵀKT solution, and the constraint equations hold."""
    nodes, conn = grid_tet10(3, 3, 3, seed=4)
    Cm, mat_of = tables(len(conn))
    nn = len(nodes)
    n = 3 * nn
    Kd = full_matrix(nodes, conn, Cm, mat_of)
    rng = np.random.default_rng(1)
    fixed = np.zeros(n, bool)
    fx = np.nonzero(nodes[:, 0] < 0.01)[0]
    for a in fx:
        fixed[3 * a:3 * a + 3] = True
    g = np.zeros(n)
    g[3 * fx[0]] = 0.01
    F = rng.standard_normal((n, 2))
    free_nodes = [a for a in range(nn) if not fixed[3 * a]]
    s1, s2, s3 = free_nodes[5], free_nodes[17], free_nodes[8]
    masters = [free_nodes[30], free_nodes[40], free_nodes[50]]
    w = np.array([0.5, 0.3, 0.2])
    rows = [(3 * s1 + 1, [(3 * s1, 0.3), (3 * s1 + 2, -0.2)], 0.0)]
    rows += [(3 * s2 + k, [(3 * m + k, wi) for m, wi in zip(masters, w)], 0.0) for k in range(3)]
    rows.append((3 * s3, [(3 * fx[0], 2.0), (3 * masters[0], 0.1)], 0.05))            # a constrained master with a prescribed value + a constant
    sn = np.array([free_nodes[2], free_nodes[60]])
    blocks = np.stack([np.diag([5.0, 7.0, 9.0]), 4.0 * np.outer([0, 0, 1.0], [0, 0, 1.0])])
    red = Reduction.from_rows(nn, rows, spring_nodes=sn, spring_blocks=blocks)
    Ks = Kd.copy()
    for a, b in zip(sn, blocks):
        Ks[3 * a:3 * a + 3, 3 * a:3 * a + 3] += b
    asm = assemble(nodes, conn, mat_of, Cm, fixed.copy(), g.copy(), F.copy(), upper=upper, red=red)
    X = solve_system(asm.to_system(), upper)
    U = red.expand(X)
    A = red.matrix().toarray()
    ref = dense_solution(Ks, F, fixed, g, red.dep, A, red.c)
    assert np.abs(U - ref).max() / np.abs(ref).max() < 1e-12
    assert np.abs(U[red.dep] - (A @ U + red.c[:, None])).max() < 1e-13 * np.abs(U).max()
    assert asm.info["dependent_dofs"] == red.n_dep and asm.info["springs"] == 2 and asm.info["correction_entries"] > 0


def rigid_rows(nodes, face, point, n_mesh):
    """The rows of a rigid body: u_i = u_ref + θ × (x_i − p); the extra nodes n_mesh (displacement) and n_mesh + 1 (rotation)."""
    e0, e1 = 3 * n_mesh, 3 * (n_mesh + 1)
    rows = []
    for i in face:
        rx, ry, rz = nodes[i] - point
        rows += [(3 * i, [(e0, 1.0), (e1 + 1, rz), (e1 + 2, -ry)], 0.0), (3 * i + 1, [(e0 + 1, 1.0), (e1 + 2, rx), (e1, -rz)], 0.0),
                 (3 * i + 2, [(e0 + 2, 1.0), (e1, ry), (e1 + 1, -rx)], 0.0)]
    return rows


def test_a_rigid_coupling_with_extra_nodes_equals_the_dense_solution_and_moves_the_face_as_a_rigid_body():
    nodes, conn = grid_tet10(3, 3, 3, seed=5)
    Cm, mat_of = tables(len(conn))
    nn = len(nodes)
    n = 3 * nn
    Kd = np.zeros((n + 6, n + 6))
    Kd[:n, :n] = full_matrix(nodes, conn, Cm, mat_of)
    p = np.array([3.5, 1.0, 1.0])
    face = np.nonzero(nodes[:, 0] > 2.99)[0]
    fixed = np.zeros(n + 6, bool)
    for a in np.nonzero(nodes[:, 0] < 0.01)[0]:
        fixed[3 * a:3 * a + 3] = True
    red = Reduction.from_rows(nn, rigid_rows(nodes, face, p, nn), extra_xyz=np.array([p, p]), fixed_extra=np.array([False, True, False, False, False, True]))
    fixed[n:] = red.fixed_extra
    F = np.zeros((n + 6, 2))
    F[n:n + 6, 0] = [100.0, 0.0, -50.0, 0.0, 800.0, 0.0]                           # the load of the rigid load on the extra DOFs (the free components)
    F[n:n + 6, 1] = [0.0, 20.0, 0.0, 300.0, 0.0, 500.0]
    for upper in (True, False):
        asm = assemble(nodes, conn, mat_of, Cm, np.concatenate([fixed[:n], np.zeros(6, bool)]), np.zeros(n + 6), F.copy(), upper=upper, red=red)
        X = solve_system(asm.to_system(), upper)
        U = red.expand(X)
        A = red.matrix().toarray()
        ref = dense_solution(Kd, F, fixed, np.zeros(n + 6), red.dep, A, red.c)
        assert np.abs(U - ref).max() / np.abs(ref).max() < 1e-11
        q = U[n:]                                                                    # displacement and rotation of the point (per state)
        for k in range(2):
            rigid = q[:3, k] + np.cross(q[3:, k], nodes[face] - p)                   # u_i = u_ref + θ × (x_i − p)
            assert np.abs(U[:n, k].reshape(-1, 3)[face] - rigid).max() < 1e-12 * np.abs(U).max()
        assert np.abs(q[[1, 5]]).max() == 0.0 and np.abs(q[0]).max() > 0              # the constrained components of the point stay zero
    assert asm.info["extra_nodes"] == 2 and asm.info["constrained"] >= 2


def rotated_C_reference(Cl, R):
    """C in the global axes from the local one by the tensor rotation of the stress and strain written out (not the strain transformation)."""
    def tensor(v):
        return np.array([[v[0], v[3] / 2, v[4] / 2], [v[3] / 2, v[1], v[5] / 2], [v[4] / 2, v[5] / 2, v[2]]])      # engineering shear → the strain tensor

    def stress_voigt(t):
        return np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]])
    C = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        eg = tensor(e)
        el = R @ eg @ R.T
        v = np.array([el[0, 0], el[1, 1], el[2, 2], 2 * el[0, 1], 2 * el[0, 2], 2 * el[1, 2]])
        sl = stress_voigt_from(Cl @ v)
        sg = R.T @ sl @ R
        C[:, j] = stress_voigt(sg)
    return C


def stress_voigt_from(s):
    return np.array([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]])


def random_rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def test_the_strain_transformation_and_the_rotated_matrix_equal_the_tensor_rotation():
    o = {"E1": 100000.0, "E2": 20000.0, "E3": 10000.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.4, "G12": 5000.0, "G13": 4000.0, "G23": 3000.0}
    Cl = ortho.stiffness_local(o)
    assert np.allclose(Cl @ ortho.compliance(o), np.eye(6))
    for seed in (1, 2, 3):
        R = random_rotation(seed)
        T = ortho.strain_transform(R)
        assert np.abs(T.T @ Cl @ T - rotated_C_reference(Cl, R)).max() < 1e-9 * np.abs(Cl).max()
        # the kernel and the numpy function are the same formulas
        Tk = np.empty((6, 6))
        K.strain_transform(R, Tk)
        assert np.array_equal(Tk, T)
    Ciso = K.iso_stiffness(210000.0, 0.3)
    R = random_rotation(9)
    T = ortho.strain_transform(R)
    assert np.abs(T.T @ Ciso @ T - Ciso).max() < 1e-9 * Ciso.max()                       # an isotropic matrix does not change with the axes


def cylindrical_C_at_gauss_points(nodes, conn, Cl, ax):
    """C (E, 4, 6, 6) in the global axes of a cylindrically orthotropic material: numpy, the tensor rotation (`rotated_C_reference`) per Gauss point."""
    from fea_core.loads import GAUSS4
    from fea_core.results import tet10_shape
    N = np.stack([tet10_shape(np.asarray(p)) for p in GAUSS4])
    xg = np.einsum("gk,ekc->egc", N, nodes[conn])
    R = ortho.cylindrical_axes_at(ax, xg)
    out = np.empty((len(conn), 4, 6, 6))
    for e in range(len(conn)):
        for g in range(4):
            out[e, g] = rotated_C_reference(Cl, R[e, g])
    return out, R


def test_cylindrical_orthotropy_in_the_kernel_equals_the_numpy_recovery_at_the_gauss_points():
    o = {"E1": 10000.0, "E2": 100000.0, "E3": 10000.0, "nu12": 0.03, "nu13": 0.01, "nu23": 0.02, "G12": 5000.0, "G13": 4000.0, "G23": 4000.0}
    Cl = ortho.stiffness_local(o)
    nodes, conn = grid_tet10(3, 3, 2, seed=7)
    nodes = nodes + np.array([4.0, 4.0, 0.0])                       # the axis outside the body
    ax = np.array([0.3, 0.2, -1.0, 0.0, 0.0, 1.0])
    Cg, _ = cylindrical_C_at_gauss_points(nodes, conn, Cl, ax)
    ne, n = len(conn), 3 * len(nodes)
    alpha = np.array([2e-6, 11e-6, 5e-6])
    Am = np.concatenate([alpha, np.zeros(3)])[None]
    an = Aniso(np.array([1], np.int8), ax[None], Am)
    dT = np.random.default_rng(3).uniform(-20, 80, (len(nodes), 1))
    s = build_system(nodes, conn, np.zeros(ne, np.int32), Cl[None], np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=False, dT=dT, aniso=an)
    Ksys = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n, n))
    U = np.random.default_rng(5).standard_normal((len(nodes), 3))
    # the stiffness: K·u = the nodal forces of the recovery with C at the Gauss points
    _, fe = _element_pass(nodes, conn, U[conn], Cg)
    f = np.zeros((len(nodes), 3))
    for c in range(3):
        f[:, c] = np.bincount(conn.ravel(), weights=fe[:, :, c].ravel(), minlength=len(nodes))
    ref = (Ksys @ U.ravel()).reshape(-1, 3)
    assert np.abs(ref - f).max() / np.abs(f).max() < 1e-12
    # the thermal load: u = 0 with the thermal strain ε_th(x) = a(x)·ΔT gives σ = −C·ε_th and the nodal forces −f_th (the recovery is the independent numpy code)
    R = cylindrical_axes_at_gauss(nodes, conn, ax)
    Ag = np.einsum("egki,k,egkj->egij", R, alpha, R)
    A6 = np.stack([Ag[..., 0, 0], Ag[..., 1, 1], Ag[..., 2, 2], 2 * Ag[..., 0, 1], 2 * Ag[..., 0, 2], 2 * Ag[..., 1, 2]], axis=-1)
    _, fe0 = _element_pass(nodes, conn, np.zeros_like(U[conn]), Cg, None, dT[conn][:, :, 0], A6)
    f_th = np.zeros((len(nodes), 3))
    for c in range(3):
        f_th[:, c] = np.bincount(conn.ravel(), weights=fe0[:, :, c].ravel(), minlength=len(nodes))
    assert np.abs(s.rhs[:, 0].reshape(-1, 3) + f_th).max() / np.abs(f_th).max() < 1e-12


def cylindrical_axes_at_gauss(nodes, conn, ax):
    from fea_core.loads import GAUSS4
    from fea_core.results import tet10_shape
    N = np.stack([tet10_shape(np.asarray(p)) for p in GAUSS4])
    return ortho.cylindrical_axes_at(ax, np.einsum("gk,ekc->egc", N, nodes[conn]))


def test_a_cartesian_orthotropic_material_with_thermal_expansion_equals_the_recovery():
    o = {"E1": 100000.0, "E2": 20000.0, "E3": 10000.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.4, "G12": 5000.0, "G13": 4000.0, "G23": 3000.0}
    R = random_rotation(4)
    T = ortho.strain_transform(R)
    Cg = T.T @ ortho.stiffness_local(o) @ T
    alpha = np.array([3e-6, 12e-6, 7e-6])
    a6 = ortho.thermal_voigt(R, alpha)
    nodes, conn = grid_tet10(2, 2, 2, seed=8)
    ne, n = len(conn), 3 * len(nodes)
    dT = np.random.default_rng(2).uniform(-10, 50, (len(nodes), 1))
    an = Aniso(np.array([0], np.int8), np.zeros((1, 6)), a6[None])
    s = build_system(nodes, conn, np.zeros(ne, np.int32), Cg[None], np.zeros(n, bool), np.zeros(n), np.zeros((n, 1)), upper=False, dT=dT, aniso=an)
    _, fe0 = _element_pass(nodes, conn, np.zeros((ne, 10, 3)), np.broadcast_to(Cg, (ne, 6, 6)).copy(), None, dT[conn][:, :, 0],
                           np.broadcast_to(a6, (ne, 4, 6)).copy())
    f_th = np.zeros((len(nodes), 3))
    for c in range(3):
        f_th[:, c] = np.bincount(conn.ravel(), weights=fe0[:, :, c].ravel(), minlength=len(nodes))
    assert np.abs(s.rhs[:, 0].reshape(-1, 3) + f_th).max() / np.abs(f_th).max() < 1e-12
    # a free uniform expansion of a body without constraints carries no stress: the thermal strain ε = a6·ΔT is compatible → σ = C(ε − ε_th) = 0 for u = x·a6·ΔT
    dT1 = np.full((len(nodes), 1), 30.0)
    E = np.array([[a6[0], a6[3] / 2, a6[4] / 2], [a6[3] / 2, a6[1], a6[5] / 2], [a6[4] / 2, a6[5] / 2, a6[2]]]) * 30.0
    U = nodes @ E.T
    sig, fe = _element_pass(nodes, conn, U[conn], np.broadcast_to(Cg, (ne, 6, 6)).copy(), None, dT1[conn][:, :, 0], np.broadcast_to(a6, (ne, 4, 6)).copy())
    assert np.abs(sig).max() < 1e-8 and np.abs(fe).max() < 1e-8


def test_extra_couplings_enter_the_pattern_and_the_pattern_without_them_is_unchanged():
    nodes, conn = grid_tet10(2, 2, 2)
    nn = len(nodes)
    ptr, el, loc = K.node_elements(nn, conn)
    for upper in (True, False):
        base = K.pattern(nn, conn, ptr, el, upper)
        again = K.pattern(nn, conn, ptr, el, upper, K.extra_csr(nn, np.zeros(0, np.int64), np.zeros(0, np.int64), upper))
        assert np.array_equal(base[0], again[0]) and np.array_equal(base[1], again[1])
        a, b = np.array([0, 5, 7]), np.array([nn - 1, 2, 7])
        ip, ix = K.pattern(nn, conn, ptr, el, upper, K.extra_csr(nn, a, b, upper))
        for x, y in zip(a, b):
            lo, hi = min(x, y), max(x, y)
            assert hi in ix[ip[lo]:ip[lo + 1]]
            if not upper:
                assert lo in ix[ip[hi]:ip[hi + 1]]
        for r in range(nn):
            row = ix[ip[r]:ip[r + 1]]
            assert np.all(np.diff(row) > 0) and (not upper or row.min() >= r)


def test_the_coupling_pairs_cover_every_entry_of_the_correction():
    """The pattern made from the dependent nodes holds every entry of Aᵀ·K·... that the elimination writes (add_scalar_entries reports a missing block)."""
    nodes, conn = grid_tet10(3, 3, 3, seed=2)
    Cm, mat_of = tables(len(conn))
    nn, n = len(nodes), 3 * len(nodes)
    rng = np.random.default_rng(0)
    pick = rng.choice(nn, 6, replace=False)
    masters = rng.choice(nn, 12, replace=False)
    rows = []
    for k, s in enumerate(pick):
        ms = masters[2 * k:2 * k + 2]
        rows += [(3 * s + d, [(3 * m + d, 0.5) for m in ms], 0.0) for d in range(3)]
    red = Reduction.from_rows(nn, rows)
    fixed = np.zeros(n, bool)
    fixed[:3] = True
    for upper in (True, False):
        asm = assemble(nodes, conn, mat_of, Cm, fixed.copy(), np.zeros(n), np.zeros((n, 1)), upper=upper, red=red)      # raises AssemblyError when a block is missing
        assert asm.info["correction_entries"] > 0


def test_invalid_constraints_are_rejected():
    nodes, conn = grid_tet10(2, 2, 2)
    Cm, mat_of = tables(len(conn))
    nn, n = len(nodes), 3 * len(nodes)
    z = np.zeros(n, bool)
    z[0] = True
    base = dict(nodes=nodes, tets=conn, mat_of=mat_of, Cm=Cm, g=np.zeros(n), F=np.zeros((n, 1)), upper=True)

    def run(rows, fixed=z):
        return assemble(fixed=fixed.copy(), red=Reduction.from_rows(nn, rows), **base)
    with pytest.raises(AssemblyError, match="dependent in two"):
        run([(10, [(20, 1.0)], 0.0), (10, [(21, 1.0)], 0.0)])
    with pytest.raises(AssemblyError, match="master of another"):
        run([(10, [(20, 1.0)], 0.0), (20, [(21, 1.0)], 0.0)])
    with pytest.raises(AssemblyError, match="also constrained"):
        run([(0, [(20, 1.0)], 0.0)])
    with pytest.raises(AssemblyError, match="does not exist"):
        run([(10, [(n + 5, 1.0)], 0.0)])
    with pytest.raises(AssemblyError, match="node of the mesh"):
        assemble(fixed=np.concatenate([z, np.zeros(6, bool)]), red=Reduction.from_rows(nn, [(n + 1, [(20, 1.0)], 0.0)], extra_xyz=np.zeros((2, 3))),
                 **{**base, "g": np.zeros(n + 6), "F": np.zeros((n + 6, 1))})
    with pytest.raises(ReductionError):
        Reduction.from_rows(nn, [(10, [(20, 1.0)], 0.0)]).merge(Reduction(nn + 1))


def test_the_reduction_survives_the_files_and_merging_shifts_the_extra_nodes():
    a = Reduction.from_rows(5, [(3, [(4, 0.5), (15, 2.0)], 0.1)], extra_xyz=np.array([[1.0, 2, 3], [1.0, 2, 3]]), fixed_extra=np.array([1, 0, 0, 0, 0, 1], bool),
                            spring_nodes=np.array([1]), spring_blocks=np.eye(3)[None], labels={"rigid": 1})
    b = Reduction.from_rows(5, [(6, [(5, 1.0), (15 + 6, 3.0)], 0.0)], extra_xyz=np.array([[0.0, 0, 0], [0.0, 0, 0]]), fixed_extra=np.zeros(6, bool), labels={"rigid": 1})
    m = a.merge(b)
    assert m.n_extra == 4 and m.labels == {"rigid": 2} and list(m.dep) == [3, 6]
    assert list(m.col) == [4, 15, 5, 15 + 6 + 6]                           # the master of b in its extra node 0 (DOF 15) → the extra node 2 (DOF 21)
    assert list(m.ptr) == [0, 2, 4] and m.matrix().shape == (2, 3 * 9)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        r = Reduction.load(d)
    assert r.n_mesh == 5 and np.array_equal(r.dep, m.dep) and np.array_equal(r.col, m.col) and np.array_equal(r.val, m.val) and np.array_equal(r.fixed_extra, m.fixed_extra)
    assert Reduction.load("no_such_folder") is None
    X = np.zeros((3 * 9, 1))
    X[4], X[15] = 2.0, 10.0
    assert m.expand(X)[3, 0] == pytest.approx(0.5 * 2.0 + 2.0 * 10.0 + 0.1)


def curved_face():
    """A quadratic triangle with the midpoints moved off the chords (a bulge in z): the corners (0,0,0) (4,0,0) (0,4,0), the midpoints by `bulge`."""
    c = np.array([[0.0, 0, 0], [4, 0, 0], [0, 4, 0]])
    mid = np.array([(c[0] + c[1]) / 2, (c[1] + c[2]) / 2, (c[2] + c[0]) / 2]) + np.array([[0, 0, 0.3], [0.1, 0.1, 0.5], [0, 0, 0.2]])
    return np.vstack([c, mid])


def test_the_projection_on_a_curved_quadratic_face_finds_the_point_and_the_distance():
    from fea_core.ties import _tri6_shape, project_on_faces
    X = curved_face()
    rng = np.random.default_rng(0)
    xi = rng.uniform(0.05, 0.4, 20)
    eta = rng.uniform(0.05, 0.4, 20)
    N, dxi, deta = _tri6_shape(xi, eta)
    pts = N @ X
    nrm = np.cross(dxi @ X, deta @ X)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    off = 0.05                                                                  # a point beside the surface along the normal: the closest point is the foot
    xs, es, dist = project_on_faces(pts + off * nrm, np.repeat(X[None], 20, 0))
    assert np.abs(xs - xi).max() < 1e-9 and np.abs(es - eta).max() < 1e-9 and np.abs(dist - off).max() < 1e-9
    # a point beyond the edge: the closest point is on the boundary of the triangle
    far = np.array([[6.0, -2.0, 0.0]])
    xs, es, dist = project_on_faces(far, X[None])
    assert -1e-12 <= xs[0] <= 1 + 1e-12 and es[0] == pytest.approx(0.0, abs=1e-9) and dist[0] > 1.0
    # the shape functions reproduce the linear field exactly at the projected point (u = a·x): the tie of a slave node to a rotated rigid body
    a = rng.standard_normal((3, 3))
    xs, es, dist = project_on_faces(pts[:3] + 0.01 * nrm[:3], np.repeat(X[None], 3, 0))
    N2, _, _ = _tri6_shape(xs, es)
    assert np.abs((N2 @ X) @ a - (N2 @ (X @ a))).max() < 1e-12


def test_tie_weights_tie_the_nodes_of_a_flat_interface_and_leave_the_far_ones():
    """Two flat meshes of one square, the slave finer and shifted by half an element: every slave node within the tolerance is tied to the master face that
    contains it with weights that sum to 1 and reproduce a linear field; a node farther than the tolerance stays free."""
    from fea_core.ties import tie_weights

    class M:
        pass

    def tri6_grid(n, z, size=10.0):
        pts, tris, idx = [], [], {}

        def node(i, j):                                               # i, j on the grid of half steps (the midpoints)
            if (i, j) not in idx:
                idx[(i, j)] = len(pts)
                pts.append([i * size / (2 * n), j * size / (2 * n), z])
            return idx[(i, j)]
        for i in range(n):
            for j in range(n):
                a, b, c, d = (2 * i, 2 * j), (2 * i + 2, 2 * j), (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)
                for t in ((a, b, c), (a, c, d)):
                    m = [((t[0][0] + t[1][0]) // 2, (t[0][1] + t[1][1]) // 2), ((t[1][0] + t[2][0]) // 2, (t[1][1] + t[2][1]) // 2),
                         ((t[2][0] + t[0][0]) // 2, (t[2][1] + t[0][1]) // 2)]
                    tris.append([node(*p) for p in (*t, *m)])
        return np.array(pts), np.array(tris)
    Xm, tm = tri6_grid(3, 0.0)
    Xs, ts = tri6_grid(5, 0.002)
    Xs = np.vstack([Xs, [[5.0, 5.0, 0.5]]])                          # a node of the slave side far from the master surface (no face of the slave refers to it)
    mesh = M()
    mesh.nodes = np.vstack([Xm, Xs])
    mesh.surfaces = {"m": tm, "s": ts + len(Xm)}
    w = tie_weights(mesh, "s", "m", 0.01)
    assert len(w["untied"]) == 0 and len(w["nodes"]) == len(np.unique(ts))
    assert np.abs(w["weights"].sum(axis=1) - 1).max() < 1e-12
    lin = np.array([1.5, -0.7, 0.3])
    tied = np.einsum("nk,nk->n", w["weights"], mesh.nodes[w["masters"]] @ lin)
    assert np.abs(tied - mesh.nodes[w["nodes"]] @ lin + 0.3 * 0).max() < 0.003 * abs(lin[2]) + 1e-12      # the field of the foot of the projection (z differs by 0.002)
    mesh.surfaces["s2"] = np.vstack([ts + len(Xm), [[len(mesh.nodes) - 1] * 6]])
    w2 = tie_weights(mesh, "s2", "m", 0.01)
    assert list(w2["untied"]) == [len(mesh.nodes) - 1]


def test_worker_with_a_reduction_and_orthotropic_materials_writes_the_same_system(tmp_path):
    """The job files of the worker: reduction.npz + aniso.npz + the padded F / fixed / g → the files of the solver = `build_system` in memory."""
    import subprocess
    import sys
    o = {"E1": 10000.0, "E2": 100000.0, "E3": 10000.0, "nu12": 0.03, "nu13": 0.01, "nu23": 0.02, "G12": 5000.0, "G13": 4000.0, "G23": 4000.0}
    Cl = ortho.stiffness_local(o)
    nodes, conn = grid_tet10(3, 2, 2, seed=11)
    nodes = nodes + np.array([4.0, 4.0, 0.0])
    ne, nn = len(conn), len(nodes)
    n = 3 * (nn + 2)
    ax = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0])
    an = Aniso(np.array([1], np.int8), ax[None], np.concatenate([[2e-6, 11e-6, 5e-6], np.zeros(3)])[None])
    face = np.nonzero(nodes[:, 0] > nodes[:, 0].max() - 0.2)[0]
    p = np.array([nodes[:, 0].max() + 1, 1.0, 1.0])
    red = Reduction.from_rows(nn, rigid_rows(nodes, face, p, nn), extra_xyz=np.array([p, p]), fixed_extra=np.zeros(6, bool), spring_nodes=np.array([3]),
                              spring_blocks=3.0 * np.eye(3)[None])
    fixed = np.zeros(n, bool)
    fixed[:3 * 3] = True
    F = np.random.default_rng(1).standard_normal((n, 2))
    dT = np.random.default_rng(2).uniform(0, 50, (nn, 2))
    job = tmp_path / "job"
    job.mkdir()
    np.save(job / "nodes.npy", nodes)
    np.save(job / "tets.npy", conn.astype(np.int32))
    np.save(job / "tet_part.npy", np.zeros(ne, np.int32))
    (job / "materials.json").write_text(json.dumps([{"E": 1.0, "nu": 0.3, "alpha": 0.0}]), encoding="utf-8")
    np.save(job / "C_tab.npy", Cl[None])
    np.save(job / "C_idx.npy", np.zeros(ne, np.int32))
    np.savez(job / "aniso.npz", kind=an.kind, ax=an.ax, A=an.A)
    red.save(job)
    np.save(job / "F.npy", F)
    np.save(job / "fixed.npy", fixed)
    np.save(job / "g.npy", np.zeros(n))
    np.save(job / "dT.npy", dT)
    (job / "assembly.json").write_text(json.dumps({"threads": 2}), encoding="utf-8")
    rc = subprocess.run([sys.executable, "-m", "fea_core.assembly.worker", str(job), "direct"], capture_output=True, text=True, encoding="utf-8")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    ref = build_system(nodes, conn, np.zeros(ne, np.int32), Cl[None], fixed, np.zeros(n), F, upper=True, dT=dT, red=red, aniso=an)
    d = job / "direct"
    assert np.array_equal(np.load(d / "indptr.npy"), ref.indptr) and np.array_equal(np.load(d / "indices.npy"), ref.indices)
    assert np.array_equal(np.load(d / "data.npy"), ref.data) and np.array_equal(np.load(d / "rhs.npy"), ref.rhs)
    info = json.loads((job / "assemble.json").read_text(encoding="utf-8"))
    assert info["extra_nodes"] == 2 and info["dependent_dofs"] == 3 * len(face) and info["springs"] == 1 and info["ndof"] == n
    # the iterative solver cannot take constraints that couple several nodes: the worker says so (an input problem, not a crash)
    rc = subprocess.run([sys.executable, "-m", "fea_core.assembly.worker", str(job), "hypre"], capture_output=True, text=True, encoding="utf-8")
    assert rc.returncode == 4 and "several nodes" in rc.stdout


def test_a_spring_on_a_node_without_elements_is_an_error():
    nodes, conn = grid_tet10(2, 2, 2)
    nodes = np.vstack([nodes, [[9.0, 9, 9]]])                              # an orphan node
    Cm, mat_of = tables(len(conn))
    n = 3 * len(nodes)
    red = Reduction(len(nodes), spring_nodes=np.array([len(nodes) - 1]), spring_blocks=np.eye(3)[None])
    fixed = np.zeros(n, bool)
    fixed[:3] = True
    with pytest.raises(AssemblyError, match="spring"):
        assemble(nodes, conn, mat_of, Cm, fixed, np.zeros(n), np.zeros((n, 1)), upper=True, red=red)


# ------------------------------------------------------------------ review of F13 step 1b: the labels of the reduction reach the worker
def _cross_node_reduction(nn, labels=None):
    """Two dependent nodes tied to others (a *TIE / distributing like equation: the masters belong to ANOTHER node) – no extra nodes."""
    rows = []
    for dep, ms in ((0, (5, 7)), (1, (8, 9))):
        for k in range(3):
            rows.append((3 * dep + k, [(3 * ms[0] + k, 0.5), (3 * ms[1] + k, 0.5)], 0.0))
    return Reduction.from_rows(nn, rows, labels=labels or {})


def test_the_labels_of_a_reduction_survive_the_job_file_and_multi_node_is_decided_by_the_equations(tmp_path):
    red = _cross_node_reduction(12, {"tie": 6, "springs": 2})
    red.save(tmp_path)
    back = Reduction.load(tmp_path)
    assert back.labels == {"tie": 6, "springs": 2} and back.multi_node
    # an older job file without the labels: the equations themselves tell that they couple nodes
    with np.load(tmp_path / "reduction.npz") as z:
        np.savez(tmp_path / "reduction.npz", **{k: z[k] for k in z.files if k != "labels"})
    old = Reduction.load(tmp_path)
    assert old.labels == {} and old.multi_node
    local = Reduction.from_rows(12, [(3 * 4 + 1, [(3 * 4, 0.3), (3 * 4 + 2, -0.2)], 0.0)])          # a cylindrical-support-like equation of one node
    assert not local.multi_node and not Reduction(12, spring_nodes=np.array([2]), spring_blocks=np.eye(3)[None]).multi_node
    assert Reduction.from_rows(12, [(0, [(3 * 12, 1.0)], 0.0)], extra_xyz=np.zeros((1, 3))).multi_node


def test_the_worker_reports_the_constraints_and_refuses_ties_for_the_iterative_solver(tmp_path):
    """The labels reach assemble.json (they were lost on the way to the worker: `info["constraints"]` stayed empty) and the guard of the iterative solver works
    for a *TIE / distributing support, which has no extra nodes."""
    import subprocess
    import sys
    nodes, conn = grid_tet10(3, 2, 2, seed=4)
    nn, ne = len(nodes), len(conn)
    Cm, mat_of = tables(ne)
    n = 3 * nn
    red = _cross_node_reduction(nn, {"tie": 6})
    fixed = np.zeros(n, bool)
    fixed[3 * 10:3 * 10 + 3] = True
    job = tmp_path / "job"
    job.mkdir()
    np.save(job / "nodes.npy", nodes)
    np.save(job / "tets.npy", conn.astype(np.int32))
    np.save(job / "tet_part.npy", np.zeros(ne, np.int32))
    (job / "materials.json").write_text(json.dumps([{"E": 1000.0, "nu": 0.3, "alpha": 0.0}]), encoding="utf-8")
    red.save(job)
    np.save(job / "F.npy", np.random.default_rng(3).standard_normal((n, 1)))
    np.save(job / "fixed.npy", fixed)
    np.save(job / "g.npy", np.zeros(n))
    (job / "assembly.json").write_text(json.dumps({"threads": 2}), encoding="utf-8")
    rc = subprocess.run([sys.executable, "-m", "fea_core.assembly.worker", str(job), "direct"], capture_output=True, text=True, encoding="utf-8")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert json.loads((job / "assemble.json").read_text(encoding="utf-8"))["constraints"] == {"tie": 6}
    rc = subprocess.run([sys.executable, "-m", "fea_core.assembly.worker", str(job), "hypre"], capture_output=True, text=True, encoding="utf-8")
    assert rc.returncode == 4 and "several nodes" in rc.stdout


@pytest.mark.parametrize("upper", [True, False])
def test_one_column_of_prescribed_values_per_state_equals_separate_assemblies(upper):
    """`g` with one column per state (check 3: six unit states on one factorization): the matrix is the one of every single-column assembly and the right-hand side
    columns are theirs – with a constrained master of a constraint equation (the offset of the dependent DOF) and prescribed values of the extra DOFs of a rigid support."""
    nodes, conn = grid_tet10(3, 3, 3, seed=6)
    Cm, mat_of = tables(len(conn))
    nn = len(nodes)
    n = 3 * nn
    p = np.array([3.5, 1.0, 1.0])
    face = np.nonzero(nodes[:, 0] > 2.99)[0]
    left = np.nonzero(nodes[:, 0] < 0.01)[0]
    free_nodes = [a for a in range(nn) if a not in set(left) and a not in set(face)]
    rows = rigid_rows(nodes, face, p, nn)
    s3 = free_nodes[8]
    rows.append((3 * s3, [(3 * left[0], 2.0), (3 * free_nodes[50], 0.1)], 0.05))     # a dependent DOF whose master is constrained and prescribed
    fx = np.ones(6, bool)
    red0 = Reduction.from_rows(nn, rows, extra_xyz=np.array([p, p]), fixed_extra=fx)
    fixed = np.zeros(n + 6, bool)
    for a in left:
        fixed[3 * a:3 * a + 3] = True
    rng = np.random.default_rng(3)
    k = 3
    G = np.zeros((n + 6, k))
    G[:n][fixed[:n]] = rng.standard_normal((int(fixed[:n].sum()), k)) * 0.01
    G[n:] = rng.standard_normal((6, k)) * 0.01
    F = rng.standard_normal((n + 6, k))
    multi = assemble(nodes, conn, mat_of, Cm, fixed.copy(), G.copy(), F.copy(), upper=upper, red=red0)
    for j in range(k):
        red1 = Reduction.from_rows(nn, rows, extra_xyz=np.array([p, p]), fixed_extra=fx, g_extra=G[n:, j].copy())    # the old route: the values of the extra DOFs in the reduction
        g1 = G[:, j].copy()
        g1[n:] = 0.0
        one = assemble(nodes, conn, mat_of, Cm, fixed.copy(), g1, F[:, j:j + 1].copy(), upper=upper, red=red1)
        assert np.array_equal(multi.data, one.data) and np.array_equal(multi.indices, one.indices)
        assert np.array_equal(multi.rhs[:, j], one.rhs[:, 0]), j
    with pytest.raises(AssemblyError):
        assemble(nodes, conn, mat_of, Cm, fixed.copy(), G[:, :2].copy(), F.copy(), upper=upper, red=red0)


def test_the_value_of_a_constrained_extra_dof_is_red_g_extra_unless_the_rows_of_g_hold_values():
    """`red.g_extra` replaces the rows of g of the constrained extra DOFs (as it always did); a row of g that carries a value (the per-state values of check 3) is kept, and a DOF
    that is not constrained takes nothing from either."""
    nodes, conn = grid_tet10(2, 2, 2, seed=1)
    Cm, mat_of = tables(len(conn))
    nn = len(nodes)
    n = 3 * nn
    p = np.array([2.5, 0.5, 0.5])
    face = np.nonzero(nodes[:, 0] > 1.99)[0]
    left = np.nonzero(nodes[:, 0] < 0.01)[0]
    rows = rigid_rows(nodes, face, p, nn)
    fixed = np.zeros(n + 6, bool)
    for a in left:
        fixed[3 * a:3 * a + 3] = True
    F = np.zeros((n + 6, 1))
    vals = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    other = np.array([0.4, 0.0, -0.2, 0.05, 0.0, 0.0])

    def rhs(fx, g_extra, g_rows):
        red = Reduction.from_rows(nn, rows, extra_xyz=np.array([p, p]), fixed_extra=fx, g_extra=g_extra)
        g = np.zeros(n + 6)
        g[n:] = g_rows
        return assemble(nodes, conn, mat_of, Cm, fixed.copy(), g, F.copy(), upper=True, red=red).rhs

    all_fixed = np.ones(6, bool)
    by_reduction = rhs(all_fixed, vals, np.zeros(6))                                 # the route of every caller but check 3
    assert np.array_equal(by_reduction, rhs(all_fixed, np.zeros(6), vals))            # the same values given in the rows of g (check 3)
    mixed = np.where(other != 0.0, other, vals)                                       # a row of g with a value wins, red.g_extra stays for the DOFs whose row of g is zero
    assert np.array_equal(rhs(all_fixed, vals, other), rhs(all_fixed, mixed, np.zeros(6)))
    part = np.array([True, True, True, True, True, False])
    assert np.array_equal(rhs(part, vals, np.zeros(6)), rhs(part, np.where(part, vals, 0.0), np.zeros(6)))    # a DOF that is not constrained: its g_extra is not a value

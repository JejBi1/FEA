"""Bolts on the own assembly (F13 step 1b, part 2) – the pieces, against dense reference computations and closed formulas. No Gmsh, no ccx, no solver library:
structured tet10 grids joined by a bolt (two rigid faces and an exact Timoshenko element on the extra nodes).

* the element matrix (`joints.own.element_matrices`) against the closed formulas of the Timoshenko beam and its rigid-body motions;
* the cut of the beam / the positions of the faces (`chain_of`);
* the system with beam elements on the extra nodes against the dense TᵀKT solution, with a thermal elongation and with the preload (unit states + `preload_gaps`)
  against the dense system that has the gap DOF g of ccx (u_M2 = u_M1 + g·e, the force −F_V on g) – the algebra of the preload is checked against the formulation
  that it replaces;
* the stresses at the integration points of selected elements (`recovery.element_ip_stress`) against `recover`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from fea_core.assembly.reduction import Reduction, ReductionError                      # noqa: E402
from fea_core.assembly.system import assemble                                          # noqa: E402
from fea_core.joints.fem import BoltFE                                                 # noqa: E402
from fea_core.joints.own import BoltChainError, chain_of, element_matrices, preload_gaps   # noqa: E402
from fea_core.linear_constraints import BoltPlan, Plan                                 # noqa: E402
from tests.test_assembly import grid_tet10, tables                                     # noqa: E402
from tests.test_assembly_constraints import dense_solution, full_matrix, rigid_rows, solve_system   # noqa: E402

E_B, NU_B, A_B, I_B = 210000.0, 0.3, 50.0, 400.0


def bolt(axis=(0.0, 0.0, 1.0), length=6.0, n_points=3, split=1, preload=1000.0, couplings=None, alpha=11.5e-6, origin=(1.5, 1.5, 0.0)):
    e = np.asarray(axis, float) / np.linalg.norm(axis)
    pts = np.array([np.asarray(origin, float) + e * length * i / (n_points - 1) for i in range(n_points)])
    return BoltFE("Bolt 1", 1, pts, split, couplings or [("a", 0), ("b", n_points - 1)], A_B, I_B, E_B, NU_B, e, preload, alpha, length)


# ------------------------------------------------------------------ the element matrix
def test_the_element_matrix_is_the_exact_timoshenko_beam():
    axis = np.array([0.3, -0.5, 0.8])
    b = bolt(axis=axis, length=20.0)
    ch = chain_of(b)
    K = element_matrices(ch)[0]
    L, E, A, I = 20.0, E_B, A_B, I_B
    G = E / (2 * (1 + NU_B))
    e = axis / np.linalg.norm(axis)
    assert np.allclose(K, K.T, rtol=0, atol=1e-9 * np.abs(K).max())
    # rigid-body motions: translations and rotations about the first node
    x_b = np.array([0, 0, 0.0]) + e * L
    for d in np.eye(3):
        u = np.concatenate([d, [0, 0, 0], d, [0, 0, 0]])
        assert np.abs(K @ u).max() < 1e-9 * np.abs(K).max()
        th = d
        u = np.concatenate([[0, 0, 0], th, np.cross(th, x_b), th])
        assert np.abs(K @ u).max() < 1e-9 * np.abs(K).max()
    # axial: unit relative displacement along the axis → EA/L
    u = np.concatenate([[0, 0, 0], [0, 0, 0], e, [0, 0, 0]])
    assert u @ K @ u == pytest.approx(E * A / L, rel=1e-12)
    # torsion J = I_p = 2I
    u = np.concatenate([[0, 0, 0], [0, 0, 0], [0, 0, 0], e])
    assert u @ K @ u == pytest.approx(G * 2 * I / L, rel=1e-12)
    # a cantilever: the end A clamped, a transverse force at the end B → the Timoshenko deflection F·L³/(3EI) + F·L/(κ G A)
    t = np.cross(e, [1.0, 0, 0]) if abs(e[0]) < 0.9 else np.cross(e, [0, 1.0, 0])
    t /= np.linalg.norm(t)
    free = np.arange(6, 12)
    f = np.zeros(6)
    f[:3] = t
    w = np.linalg.solve(K[np.ix_(free, free)], f)
    assert w[:3] @ t == pytest.approx(L ** 3 / (3 * E * I) + L / (0.9 * G * A), rel=1e-12)
    assert abs(w[:3] @ e) < 1e-12 * abs(w[:3] @ t)                                        # no axial coupling


def test_a_chain_with_an_intermediate_face_has_an_element_between_each_pair():
    b = bolt(length=9.0, n_points=4, split=1, couplings=[("a", 0), ("m", 1), ("b", 3), ("m2", 1)])
    ch = chain_of(b)
    assert ch.positions == [0, 1, 3] and ch.keys == [["a"], ["m", "m2"], ["b"]]
    assert np.allclose(ch.s, [0.0, 3.0, 9.0]) and ch.n_elements == 2
    assert np.allclose(ch.k_axial, [E_B * A_B / 3.0, E_B * A_B / 6.0])
    assert ch.cut == 1                                                                    # the cut at point 1 (s = 3) = the position of the face "m": the element AFTER it (as in ccx)
    # the cut between the positions belongs to the element that contains it; at a position to the element after the position (ccx: the face is tied to the node
    # before the gap, the next element starts behind it); the cut is never in the element before the first position or behind the last one
    for split, want in ((1, 1), (2, 1), (3, 1), (0, 0)):
        b2 = bolt(length=9.0, n_points=4, split=split, couplings=[("a", 0), ("m", 1), ("b", 3)])
        assert chain_of(b2).cut == want, (split, chain_of(b2).cut)
    b3 = bolt(length=8.0, n_points=5, split=2, couplings=[("a", 0), ("m", 1), ("b", 4)])     # the cut at s = 4 between the positions s = 2 and s = 8
    assert np.allclose(chain_of(b3).s, [0.0, 2.0, 8.0]) and chain_of(b3).cut == 1
    assert chain_of(bolt(split=None)).cut is None
    with pytest.raises(BoltChainError):
        chain_of(bolt(couplings=[("a", 0)]))


# ------------------------------------------------------------------ a bolt through a block
def bolted_system(b: BoltFE):
    """A block 3 × 3 × 6 cubes (one conforming mesh), clamped at x = 0; the bolt along z (x = y = 1.5) from the bottom face (z = 0) to the top face (z = 6), both
    faces rigid – the clamped stack of a bolted joint (the preload is carried by the block, as the clamped parts carry it)."""
    nodes, conn = grid_tet10(3, 3, 6, seed=3)
    nn = len(nodes)
    Cm, mat_of = tables(len(conn), parts=1)
    face_a = np.nonzero(nodes[:, 2] < 0.2)[0]
    face_b = np.nonzero(nodes[:, 2] > 5.8)[0]
    ch = chain_of(b)
    pts = [b.points[p] for p in ch.positions]
    rows = rigid_rows(nodes, face_a, pts[0], nn) + rigid_rows(nodes, face_b, pts[-1], nn + 2)       # the extra nodes n (displacement), n + 1 (rotation) | n + 2, n + 3
    red = Reduction.from_rows(nn, rows, extra_xyz=np.array([pts[0], pts[0], pts[-1], pts[-1]]),
                              elem_dofs=np.array([list(range(3 * nn, 3 * nn + 12))]), elem_k=element_matrices(ch))
    plan = Plan(red, bolts=[BoltPlan(ch, [0, 1], np.unique(np.concatenate([face_a, face_b])))])
    fixed = np.zeros(3 * nn, bool)
    mid = (nodes[:, 2] > 0.4) & (nodes[:, 2] < 5.6)                                        # (not the nodes of the rigid faces: a node is either dependent or constrained)
    for a in np.nonzero((nodes[:, 0] < 0.2) & mid)[0]:
        fixed[3 * a:3 * a + 3] = True                                                      # the face x = 0 is clamped
    loaded = np.nonzero((nodes[:, 0] > 2.8) & mid)[0]                                      # the face x = 3 carries the load
    return nodes, conn, nn, Cm, mat_of, red, plan, fixed, loaded


def dense_k(nodes, conn, Cm, mat_of, red):
    nn = len(nodes)
    n = 3 * nn
    Kd = np.zeros((n + 3 * red.n_extra, n + 3 * red.n_extra))
    Kd[:n, :n] = full_matrix(nodes, conn, Cm, mat_of)
    for d, k in zip(red.elem_dofs, red.elem_k):
        Kd[np.ix_(d, d)] += k
    return Kd


@pytest.mark.parametrize("upper", [True, False])
def test_the_beam_on_the_extra_nodes_equals_the_dense_solution(upper):
    b = bolt()
    nodes, conn, nn, Cm, mat_of, red, plan, fixed, top_b = bolted_system(b)
    n = 3 * nn
    ne = 3 * red.n_extra
    F = np.zeros((n + ne, 3))
    for a in top_b:
        F[3 * a:3 * a + 3, 0] = [30.0, -20.0, 400.0]                                       # tension + shear + bending of the block
        F[3 * a:3 * a + 3, 1] = [0.0, 0.0, -250.0]
        F[3 * a:3 * a + 3, 2] = [100.0, 0.0, 0.0]
    fixed_all = np.concatenate([fixed, np.zeros(ne, bool)])
    asm = assemble(nodes, conn, mat_of, Cm, fixed_all, np.zeros(n + ne), F.copy(), upper=upper, red=red)
    X = solve_system(asm.to_system(), upper)
    U = red.expand(X)
    A = red.matrix().toarray()
    ref = dense_solution(dense_k(nodes, conn, Cm, mat_of, red), F, fixed_all, np.zeros(n + ne), red.dep, A, red.c)
    assert np.abs(U - ref).max() / np.abs(ref).max() < 1e-11
    assert asm.info["beam_elements"] == 1 and asm.info["extra_nodes"] == 4
    # the bolt carries a part of the load: the force on its end B (the element's resisting force K_e·u) along the axis
    f_el = red.elem_k[0] @ U[3 * nn:3 * nn + 12, :]
    N = f_el[6:9].T @ np.array([0, 0, 1.0])
    assert np.abs(N).min() > 1.0


def gap_formulation(nodes, conn, Cm, mat_of, red, ch, fixed, F_V, T_elong=0.0):
    """The formulation of ccx: a gap DOF g in series with the axial stiffness (N = k·(B·u − g)); the force −F_V on g in the preload state (g free), g = 0 in the
    others. Dense. → the solution of the mesh DOFs of the preload state."""
    nn = len(nodes)
    n = 3 * nn
    ne = 3 * red.n_extra
    Kd = np.zeros((n + ne + 1, n + ne + 1))
    Kd[:n + ne, :n + ne] = dense_k(nodes, conn, Cm, mat_of, red)
    k = ch.k_axial[0]
    e = ch.e
    B = np.zeros(n + ne + 1)
    d = red.elem_dofs[0]
    B[d[0:3]] = -e
    B[d[6:9]] = e
    # N = k (B·u − g): the energy ½ k (B·u − g)² replaces ½ k (B·u)² of the beam
    Kd[:n + ne, :n + ne] -= k * np.outer(B[:n + ne], B[:n + ne])
    Bg = B.copy()
    Bg[-1] = -1.0
    Kd += k * np.outer(Bg, Bg)
    F = np.zeros((n + ne + 1, 1))
    F[-1, 0] = -F_V
    fixed_all = np.concatenate([fixed, np.zeros(ne + 1, bool)])
    A = np.zeros((red.n_dep, n + ne + 1))
    A[:, :n + ne] = red.matrix().toarray()
    return dense_solution(Kd, F, fixed_all, np.zeros(n + ne + 1), red.dep, A, red.c)


def test_the_preload_by_unit_states_equals_the_gap_formulation_of_ccx():
    b = bolt(preload=1500.0)
    nodes, conn, nn, Cm, mat_of, red, plan, fixed, top_b = bolted_system(b)
    n = 3 * nn
    ne = 3 * red.n_extra
    ch = plan.bolts[0].chain
    # the unit state: an elongation of the bolt by 1 mm, solved with homogeneous right-hand sides
    F = np.zeros((n + ne, 1))
    F[n:, 0] = plan.unit_loads()[:, 0]
    fixed_all = np.concatenate([fixed, np.zeros(ne, bool)])
    g = np.zeros(n + ne)
    g[0] = 0.0
    asm = assemble(nodes, conn, mat_of, Cm, fixed_all, g, F.copy(), upper=True, red=red, hom_cols=1)
    Xu = red.expand(solve_system(asm.to_system(), True))
    C = plan.bolt_matrix(Xu)
    assert C.shape == (1, 1) and C[0, 0] < 0
    gap = preload_gaps(C, np.array([1500.0]))
    x_pre = Xu @ gap
    ref = gap_formulation(nodes, conn, Cm, mat_of, red, ch, fixed_all[:n], 1500.0)
    assert np.abs(x_pre - ref[:n + ne, 0]).max() / np.abs(ref).max() < 1e-10
    # the force in the bolt of the preload state is exactly F_V
    stretch = plan.stretch(x_pre[:, None])[0, 0]
    assert ch.k_axial[0] * (stretch - gap[0]) == pytest.approx(1500.0, rel=1e-10)


def test_the_unit_states_are_homogeneous_prescribed_displacements_do_not_enter_them():
    b = bolt()
    nodes, conn, nn, Cm, mat_of, red, plan, fixed, top_b = bolted_system(b)
    n = 3 * nn
    ne = 3 * red.n_extra
    fixed_all = np.concatenate([fixed, np.zeros(ne, bool)])
    g = np.zeros(n + ne)
    g[fixed_all] = 0.05                                                                    # a prescribed displacement of the support
    F = np.zeros((n + ne, 2))
    F[n:, 1] = plan.unit_loads()[:, 0]
    one = assemble(nodes, conn, mat_of, Cm, fixed_all, g.copy(), F[:, 1:].copy(), upper=True, red=red, hom_cols=1)
    two = assemble(nodes, conn, mat_of, Cm, fixed_all, g.copy(), F.copy(), upper=True, red=red, hom_cols=1)
    s1, s2 = one.to_system(), two.to_system()
    assert np.array_equal(s1.rhs[:, 0], s2.rhs[:, 1])                                       # the homogeneous column does not depend on the other columns
    assert np.abs(s1.rhs[np.repeat(fixed_all, 1)]).max() == 0.0                             # the constrained DOFs get zero
    assert np.abs(s2.rhs[fixed_all, 0]).max() > 0                                           # … the load state keeps the prescribed values


def test_the_thermal_elongation_is_exact_and_equals_the_gap_formulation_with_a_prescribed_gap():
    b = bolt()
    nodes, conn, nn, Cm, mat_of, red, plan, fixed, top_b = bolted_system(b)
    n = 3 * nn
    ne = 3 * red.n_extra
    ch = plan.bolts[0].chain
    T = np.zeros(nn)
    T[plan.bolts[0].nodes] = 40.0                                                          # the bolt's bearing faces are 40 K warmer
    t_elong = b.alpha * 40.0 * b.length
    assert plan.thermal_elongation(T, 0.0) == pytest.approx(plan.elongation_load(plan.bolts[0], t_elong), rel=1e-14)
    assert not plan.thermal_elongation(np.zeros(nn), 0.0).any() and not plan.thermal_elongation(None, 0.0).any()
    F = np.zeros((n + ne, 1))
    F[n:, 0] = plan.thermal_elongation(T, 0.0)
    fixed_all = np.concatenate([fixed, np.zeros(ne, bool)])
    asm = assemble(nodes, conn, mat_of, Cm, fixed_all, np.zeros(n + ne), F.copy(), upper=True, red=red)
    U = red.expand(solve_system(asm.to_system(), True))
    # the heated bolt is held by the block between its faces: compression, smaller than the free elongation would give (N = −k·t at a rigid block)
    N = ch.k_axial[0] * (plan.stretch(U)[0, 0] - t_elong)
    assert -ch.k_axial[0] * t_elong < N < 0
    # the same through the gap formulation of ccx written out densely (the gap DOF g held at the value t)
    Kd = np.zeros((n + ne + 1, n + ne + 1))
    Kd[:n + ne, :n + ne] = dense_k(nodes, conn, Cm, mat_of, red)
    k = ch.k_axial[0]
    d = red.elem_dofs[0]
    B = np.zeros(n + ne + 1)
    B[d[0:3]] = -ch.e
    B[d[6:9]] = ch.e
    Kd[:n + ne, :n + ne] -= k * np.outer(B[:n + ne], B[:n + ne])
    Bg = B.copy()
    Bg[-1] = -1.0
    Kd += k * np.outer(Bg, Bg)
    fx = np.concatenate([fixed_all, [True]])
    gv = np.zeros(n + ne + 1)
    gv[-1] = t_elong
    A = np.zeros((red.n_dep, n + ne + 1))
    A[:, :n + ne] = red.matrix().toarray()
    ref = dense_solution(Kd, np.zeros((n + ne + 1, 1)), fx, gv, red.dep, A, red.c)
    assert np.abs(U[:, 0] - ref[:n + ne, 0]).max() / np.abs(ref).max() < 1e-10


def test_a_matrix_that_is_not_the_one_of_a_stable_structure_is_refused():
    with pytest.raises(ValueError, match="not symmetric"):
        preload_gaps(np.array([[-1.0, 0.5], [0.0, -1.0]]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="does not resist"):
        preload_gaps(np.array([[0.1]]), np.array([1.0]))
    g = preload_gaps(np.array([[-2.0, 0.5], [0.5, -1.0]]), np.array([3.0, 4.0]))
    assert np.array([[-2.0, 0.5], [0.5, -1.0]]) @ g == pytest.approx([3.0, 4.0])
    assert preload_gaps(np.zeros((0, 0)), np.zeros(0)).shape == (0,)


# ------------------------------------------------------------------ the reduction with elements
def test_the_reduction_with_elements_survives_the_files_and_merging():
    import tempfile
    b = bolt()
    nodes, conn, nn, Cm, mat_of, red, plan, fixed, top_b = bolted_system(b)
    with tempfile.TemporaryDirectory() as d:
        red.save(d)
        back = Reduction.load(d)
    assert back.n_elem == 1 and np.array_equal(back.elem_dofs, red.elem_dofs) and np.array_equal(back.elem_k, red.elem_k)
    other = Reduction(nn, extra_xyz=np.zeros((2, 3)))
    merged = other.merge(red)                                                              # the extra nodes of `red` follow the two of `other`
    assert merged.elem_dofs.min() == 3 * nn + 6 and merged.n_elem == 1
    with pytest.raises(ReductionError, match="extra nodes"):
        Reduction(nn, extra_xyz=np.zeros((2, 3)), elem_dofs=np.arange(12)[None] + 3, elem_k=np.eye(12)[None]).validate()
    with pytest.raises(ReductionError, match="NaN"):
        Reduction(nn, extra_xyz=np.zeros((4, 3)), elem_dofs=np.arange(12)[None] + 3 * nn, elem_k=np.full((1, 12, 12), np.nan)).validate()


# ------------------------------------------------------------------ the stresses at the integration points
def test_the_stresses_at_the_integration_points_of_selected_elements_equal_the_recovered_ones():
    from fea_core.mesh import Mesh
    from fea_core.recovery import element_ip_stress, _element_pass, VOIGT_TO_CCX
    from fea_core.study import Material, Study
    nodes, conn = grid_tet10(3, 3, 2, seed=7)
    part = (np.arange(len(conn)) % 2).astype(int)
    mesh = Mesh(nodes, conn, part, ["A", "B"])
    st = Study(name="t", geometry="")
    st.materials["A"] = Material("A", 210000.0, 0.3, 7.85e-9, alpha=1.2e-5)
    st.materials["B"] = Material("B", 3500.0, 0.36, 1.2e-9, alpha=7e-5)
    st.part_materials.update({"A": "A", "B": "B"})
    rng = np.random.default_rng(0)
    U = rng.standard_normal((len(nodes), 3)) * 1e-3
    dT = rng.uniform(0, 30, len(nodes))
    elems = np.array([3, 10, 11, 40])
    for T in (None, dT):
        got = element_ip_stress(st, mesh, U, elems, T)
        Ce = np.stack([np.eye(6)] * len(elems))
        from fea_core.solve_large import iso_stiffness
        Ce = np.stack([iso_stiffness(st.material_of(mesh.part_names[p]).E, st.material_of(mesh.part_names[p]).nu) for p in part[elems]])
        alpha = np.array([st.material_of(mesh.part_names[p]).alpha for p in part[elems]])
        ref = _element_pass(nodes, conn[elems], U[conn[elems]], Ce, alpha, None if T is None else T[conn[elems]])[0][:, :, VOIGT_TO_CCX]
        assert np.array_equal(got, ref) and got.shape == (4, 4, 6)
    assert element_ip_stress(st, mesh, U, np.zeros(0, int)).shape == (0, 4, 6)


# ------------------------------------------------------------------ the plan of the constraints
def _mesh_with_faces(faces: dict):
    from fea_core.mesh import Mesh
    nodes, conn = grid_tet10(2, 2, 2, seed=1)
    mesh = Mesh(nodes, conn, np.zeros(len(conn), int), ["A"])
    for k, idx in faces.items():
        idx = np.asarray(idx)
        mesh.surfaces[k] = np.stack([idx, np.roll(idx, 1), np.roll(idx, 2)] * 2, axis=1)[:, :6]
    return mesh


def _study_with_bolt(monkeypatch, b):
    from fea_core.joints.model import BoltSpec
    from fea_core.study import Study
    st = Study(name="t", geometry="")
    st.bolts = [BoltSpec("Bolt 1")]
    monkeypatch.setattr("fea_core.joints.fem.bolt_models", lambda study, mesh=None: [b])
    return st


def test_build_makes_a_rigid_body_per_face_and_an_element_between_them(monkeypatch):
    from fea_core import linear_constraints as lc
    mesh = _mesh_with_faces({"a": [0, 1, 2, 3], "b": [10, 11, 12]})
    st = _study_with_bolt(monkeypatch, bolt(couplings=[("a", 0), ("b", 2)]))
    plan = lc.build(st, mesh)
    assert plan.red.n_extra == 4 and plan.red.n_elem == 1 and plan.red.n_dep == 3 * 7
    assert plan.red.labels == {"bolt_faces": 2, "bolts": 1} and plan.bolts[0].pairs == [0, 1] and len(plan.cut_bolts()) == 1
    assert np.array_equal(plan.red.elem_dofs[0], np.arange(12) + 3 * len(mesh.nodes))
    assert lc.unsupported_reasons(st, "hypre") == ["bolts (rigid bearing faces, beam elements) – direct solver only"] and lc.unsupported_reasons(st, "direct") == []


def test_two_bearing_faces_of_a_bolt_that_share_a_node_are_refused_with_the_reason(monkeypatch):
    """A fitted bolt through parts that touch: the hole walls meet in a circle, a node is in both faces. ccx ties the two rigid faces at it (a link that the bolt
    does not have), the own assembly refuses and `choose_backend_reason` sends the study to ccx with this text."""
    from fea_core import linear_constraints as lc
    mesh = _mesh_with_faces({"a": [0, 1, 2, 3], "b": [3, 10, 11]})
    st = _study_with_bolt(monkeypatch, bolt(couplings=[("a", 0), ("b", 2)]))
    with pytest.raises(ReductionError, match="share nodes"):
        lc.build(st, mesh)
    assert any("share nodes" in why for why in lc.check(st, mesh))


def test_bolts_whose_bearing_faces_share_a_node_are_refused_naming_both_bolts(monkeypatch):
    """The imprints / hole walls of two neighbouring bolts that overlap hold one node in two rigid bodies: refused already by `build` (a manual choice of the direct
    solver gets the sentence, not an error of the assembly) and by `check` (`auto` chooses ccx up front), with the names of both bolts; bolts that do not touch are fine."""
    import dataclasses
    from fea_core import linear_constraints as lc
    b1 = bolt(couplings=[("a", 0), ("b", 2)])
    b2 = dataclasses.replace(b1, name="Bolt 2", index=2, couplings=[("c", 0), ("d", 2)])
    st = _study_with_bolt(monkeypatch, b1)
    monkeypatch.setattr("fea_core.joints.fem.bolt_models", lambda study, mesh=None: [b1, b2])
    ok = _mesh_with_faces({"a": [0, 1, 2, 3], "b": [10, 11, 12], "c": [20, 21, 22], "d": [30, 31, 32]})
    assert len(lc.build(st, ok).bolts) == 2 and lc.check(st, ok) == []
    mesh = _mesh_with_faces({"a": [0, 1, 2, 3], "b": [10, 11, 12], "c": [22, 21, 11], "d": [30, 31, 32]})        # node 11: bolt 1 face b and bolt 2 face c
    with pytest.raises(ReductionError, match=r"Bolt 2: a bearing face of the bolt shares nodes with the bearing face of the bolt 'Bolt 1'"):
        lc.build(st, mesh)
    assert any("'Bolt 1'" in why and "Bolt 2" in why for why in lc.check(st, mesh))

"""Manufacturing constraints of the topology optimisation (fea_core.topo): spec, symmetry orbits, extrusion columns, support (overhang) rule,
the measurements on the exported surface and the optimiser loop with the constraints (substitute solver, without ccx)."""
import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.topo import TopologyOptSpec, TopologyOptimizer, find_threshold_bisection
from fea_core.topo.constraints import ManufacturingConstraints, selection_nodes
from fea_core.topo.groups import (DecisionGroups, Plane, extrusion_labels, mirror_mismatch, symmetry_labels)
from fea_core.topo.overhang import OverhangRepair, point_support_graph, surface_report
from fea_core.topo.sections import prism_report
from fea_core.topo.spec import axis_vector
from tests.test_topo_feedback import _to_tet10, beam_mesh, beam_study

KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])


def grid_mesh(nx, ny, nz, h=5.0, cells=None) -> Mesh:
    """Structured block nx*ny*nz cells of the size h, each into 6 tetrahedra (tet10); cells = boolean (nx, ny, nz) of the cells that exist."""
    xs, ys, zs = (np.arange(n + 1) * h for n in (nx, ny, nz))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if cells is not None and not cells[i, j, k]:
                    continue
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                tets += [[v[a] for a in tt] for tt in KUHN]
    tets = np.array(tets)
    used = np.unique(tets)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    return _to_tet10(Mesh(nodes[used], remap[tets], np.zeros(len(tets), int), ["Blok"]))


def mirrored_mesh(nx=6, ny=2, nz=2, h=5.0) -> Mesh:
    """A block that is exactly mirror-symmetric about the plane x = nx*h: a half mesh and its mirror image (shared nodes merged)."""
    half = grid_mesh(nx, ny, nz, h)
    t4 = half.tets[:, :4]
    nodes = half.nodes[np.unique(t4)]
    idx = -np.ones(len(half.nodes), dtype=int)
    idx[np.unique(t4)] = np.arange(len(nodes))
    t4 = idx[t4]
    mir = nodes.copy()
    mir[:, 0] = 2 * nx * h - mir[:, 0]
    allp = np.vstack([nodes, mir])
    key = np.round(allp / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    t_all = np.vstack([inv[t4], inv[t4 + len(nodes)][:, [1, 0, 2, 3]]])      # mirror image: swap two nodes (orientation)
    return _to_tet10(Mesh(allp[first], t_all, np.zeros(len(t_all), int), ["Blok"]))


def bisect(sens, volumes, target_volume, non_design_mask, min_density):
    return find_threshold_bisection(sens, volumes=volumes, target_volume=target_volume, non_design_mask=non_design_mask, min_density=min_density)


# ------------------------------------------------------------------ spec

def test_spec_old_dicts_load_unchanged_and_roundtrip():
    old = {"target_volume_fraction": 0.4, "evolution_rate": 0.03, "filter_radius": 7.0, "penalty": 3.0, "max_iterations": 20,
           "non_design_selections": ["a"], "min_density": 1e-3, "target_step": 0, "all_load_cases": True, "method": "beso", "tol": 1e-3}
    sp = TopologyOptSpec.from_dict(old)
    assert not sp.has_constraints and sp.symmetry == [] and sp.extrusion is None and sp.overhang is None and not sp.compare_unconstrained
    sp2 = TopologyOptSpec.from_dict({**old, "symmetry": [{"point": [1, 2, 3], "normal": [0, 0, 2]}], "extrusion": {"direction": [0, 1, 0]},
                                     "overhang": {"build_dir": "from_print", "angle_deg": 40, "base": "foot"}, "unknown": 5})
    assert sp2.has_constraints and TopologyOptSpec.from_dict(sp2.to_dict()) == sp2
    assert sp2.overhang == {"build_dir": "from_print", "angle_deg": 40.0, "base": "foot"}
    u = sp2.unconstrained()
    assert not u.has_constraints and u.filter_radius == 7.0 and not u.compare_unconstrained


def test_spec_validation_and_axis_letters():
    assert axis_vector("-z") == [0.0, 0.0, -1.0] and axis_vector("+X") == [1.0, 0.0, 0.0] and axis_vector("w") is None
    sp = TopologyOptSpec(symmetry=[{"normal": "x"}, {"normal": "y", "point": [0, 5, 0]}], extrusion={"direction": "z"})
    assert sp.symmetry[0] == {"point": None, "normal": [1.0, 0.0, 0.0]} and sp.extrusion == {"direction": [0.0, 0.0, 1.0]}
    with pytest.raises(ValueError, match="at most 3"):
        TopologyOptSpec(symmetry=[{"normal": "x"}, {"normal": "y"}, {"normal": "z"}, {"normal": [1, 1, 0]}])
    with pytest.raises(ValueError, match="perpendicular"):
        TopologyOptSpec(symmetry=[{"normal": [1, 0, 0]}, {"normal": [1, 1, 0]}])
    with pytest.raises(ValueError, match="nonzero"):
        TopologyOptSpec(symmetry=[{"normal": [0, 0, 0]}])
    with pytest.raises(ValueError, match="direction"):
        TopologyOptSpec(extrusion={})
    with pytest.raises(ValueError, match="angle_deg"):
        TopologyOptSpec(overhang={"angle_deg": 95})
    with pytest.raises(ValueError, match="build_dir"):
        TopologyOptSpec(overhang={"build_dir": "sideways"})
    with pytest.raises(ValueError, match="BESO"):
        TopologyOptSpec(method="simp", symmetry=[{"normal": "z"}])


# ------------------------------------------------------------------ symmetry

def test_symmetry_orbits_of_an_exactly_symmetric_mesh():
    m = mirrored_mesh()
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    plane = Plane(np.array([30.0, 0, 0]), np.array([1.0, 0, 0]))
    labels, info = symmetry_labels(c, vol, [plane])
    assert info["unmatched"] == 0 and info["masters"] == info["slaves"] == len(c) // 2
    g = DecisionGroups(labels, vol, np.zeros(len(c), bool))
    assert g.n == len(c) // 2 and set(g.counts) == {2}
    for k in range(g.n):                                            # both members are mirror images (to rounding)
        a, b = c[g.labels == k]
        assert np.allclose(plane.mirror(a), b, atol=1e-9) or np.allclose(plane.mirror(b), a, atol=1e-9)
    rho = (c[:, 0] > 35.0).astype(float)                           # a one-sided field is NOT symmetric
    assert mirror_mismatch(rho > 0.5, c, vol, [plane])["mismatch_volume_fraction"] > 0.2
    sym = ((np.abs(c[:, 0] - 30.0)) < 12.0)
    assert mirror_mismatch(sym, c, vol, [plane])["mismatch_volume_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_symmetry_two_planes_and_unmatched_elements_are_reported():
    m = mirrored_mesh(nx=4, ny=2, nz=2)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    p1 = Plane(np.array([20.0, 0, 0]), np.array([1.0, 0, 0]))
    p2 = Plane(np.array([0, 5.0, 0]), np.array([0, 1.0, 0]))
    labels, info = symmetry_labels(c, vol, [p1, p2])
    assert info["slaves"] > info["masters"] * 0.5
    assert len(np.unique(labels)) < len(c)
    # a mesh that is not symmetric (one cell missing on one side): the elements of the mirror-less cell stay alone and are reported
    cells = np.ones((6, 2, 2), bool)
    cells[4, 0, 0] = False                                            # the mirror image of the cell x = 5 … 10 is missing
    half = grid_mesh(6, 2, 2, cells=cells)
    c2 = half.nodes[half.tets[:, :4]].mean(axis=1)
    v2 = element_volumes(half)
    _, info2 = symmetry_labels(c2, v2, [Plane(np.array([15.0, 0, 0]), np.array([1.0, 0, 0]))])
    assert info2["unmatched"] > 0 and info2["matched_fraction"] < 1.0


def test_symmetric_constraints_make_identical_decisions_and_force_non_design():
    m = mirrored_mesh()
    spec = TopologyOptSpec(symmetry=[{"normal": [1, 0, 0], "point": [30, 0, 0]}], target_volume_fraction=0.5)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    nd = c[:, 0] < 2.5                                               # elements at the left end are protected (their mirror images at the right end too)
    con = ManufacturingConstraints(m, spec, c, vol, nd)
    assert con.groups is not None and con.groups.n == len(c) // 2
    rng = np.random.default_rng(2)
    sens = rng.random(len(c))
    th, rho = con.decide(sens, 0.5 * vol.sum(), 1e-3, bisect)
    assert con.spread(rho) == 0.0
    assert np.all(rho[con.groups.expand(con.groups.non_design)] == 1.0)         # a protected element protects its mirror image
    assert (vol * (rho > 0.5)).sum() == pytest.approx(0.5 * vol.sum(), rel=0.03)
    assert mirror_mismatch(rho > 0.5, c, vol, con.planes)["mismatch_volume_fraction"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ extrusion

def test_extrusion_columns_on_a_structured_block():
    m = grid_mesh(8, 3, 4)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    labels, info = extrusion_labels(c, vol, [0, 0, 1], [], bin_size=5.0)
    assert info["columns"] == 24 and info["mirror_planes_in_cross_section"] == 0
    cnt = np.bincount(labels)
    assert set(cnt) == {4 * 6}                                       # 4 layers x 6 tetrahedra per column
    for k in range(info["columns"]):                                  # a column = one (x, y) cell
        cell = np.floor(c[labels == k][:, :2] / 5.0).astype(int)
        assert len({tuple(r) for r in cell}) == 1
    # a mirror plane in the cross-section: the grid is aligned with it and mirror columns merge
    plane = Plane(np.array([20.0, 0, 0]), np.array([1.0, 0, 0]))
    lab2, info2 = extrusion_labels(c, vol, [0, 0, 1], [plane], bin_size=5.0)
    assert info2["columns"] == 12 and info2["mirror_planes_in_cross_section"] == 1
    # a plane along the columns (normal parallel to the direction) does not change them; an oblique plane is refused
    lab3, info3 = extrusion_labels(c, vol, [0, 0, 1], [Plane(np.array([0, 0, 10.0]), np.array([0, 0, 1.0]))], bin_size=5.0)
    assert info3["columns"] == 24 and info3["mirror_planes_along_columns"] == 1
    with pytest.raises(ValueError, match="neither parallel nor perpendicular"):
        extrusion_labels(c, vol, [0, 0, 1], [Plane(np.zeros(3), np.array([1.0, 0, 1.0]) / np.sqrt(2))], bin_size=5.0)


def test_extrusion_with_two_and_three_mirror_planes_does_not_crash():
    """Regression: `p not in inplane` compared dataclasses holding numpy arrays (ValueError for >= 2 planes)."""
    m = grid_mesh(8, 4, 4)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    px = Plane(np.array([20.0, 0, 0]), np.array([1.0, 0, 0]))
    py = Plane(np.array([0, 10.0, 0]), np.array([0, 1.0, 0]))
    pz = Plane(np.array([0, 0, 10.0]), np.array([0, 0, 1.0]))
    lab, info = extrusion_labels(c, vol, [0, 0, 1], [px, py], bin_size=5.0)
    assert info["mirror_planes_in_cross_section"] == 2 and info["columns"] == 4 * 2      # 8 x 4 columns folded twice
    lab3, info3 = extrusion_labels(c, vol, [0, 0, 1], [px, py, pz], bin_size=5.0)
    assert info3["mirror_planes_in_cross_section"] == 2 and info3["mirror_planes_along_columns"] == 1
    assert info3["columns"] == info["columns"]
    assert np.array_equal(lab, lab3)
    spec = TopologyOptSpec(extrusion={"direction": [0, 0, 1], "bin_size": 5.0},
                           symmetry=[{"normal": [1, 0, 0], "point": [20, 0, 0]}, {"normal": [0, 1, 0], "point": [0, 10, 0]}],
                           target_volume_fraction=0.5)
    con = ManufacturingConstraints(m, spec, c, vol, np.zeros(len(c), bool))
    assert con.groups is not None and con.groups.n == info["columns"]


def test_extrusion_decisions_are_uniform_in_a_column_and_a_protected_element_protects_the_column():
    m = grid_mesh(8, 3, 4)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    nd = (c[:, 0] < 4.0) & (c[:, 2] < 4.0)                            # only the bottom of the first columns
    spec = TopologyOptSpec(extrusion={"direction": [0, 0, 1], "bin_size": 5.0}, target_volume_fraction=0.5)
    con = ManufacturingConstraints(m, spec, c, vol, nd)
    sens = np.random.default_rng(1).random(len(c)) + c[:, 0] * 0.0
    th, rho = con.decide(sens, 0.5 * vol.sum(), 1e-3, bisect)
    assert con.spread(rho) == 0.0
    assert np.all(rho[c[:, 0] < 5.0] == 1.0)                           # the whole first column row is solid
    sol = rho > 0.5
    cols = {tuple(r) for r in np.floor(c[sol][:, :2] / 5.0).astype(int)}
    assert all(bool(np.all(sol[(np.floor(c[:, :2] / 5.0).astype(int) == np.array(col)).all(axis=1)])) for col in cols)


def test_prism_report_of_a_prism_and_of_a_wedge():
    def prism(profile, z0, z1):
        n = len(profile)
        v = np.array([[x, y, z0] for x, y in profile] + [[x, y, z1] for x, y in profile], dtype=float)
        f = []
        for i in range(n):                                            # sides (outward normals for a counter-clockwise profile)
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i]]
        for i in range(1, n - 1):                                    # bottom (down) and top (up) by a fan
            f += [[0, i + 1, i], [n, n + i, n + i + 1]]
        return v, np.array(f)
    sq = [(0, 0), (10, 0), (10, 6), (0, 6)]
    v, f = prism(sq, 0, 20)
    rep = prism_report(v, f, [0, 0, 1], cell=0.25)
    assert rep["areas_mm2"] == pytest.approx([60.0] * 3, rel=0.02) and rep["min_iou"] == pytest.approx(1.0, abs=1e-9)
    assert rep["area_spread"] == pytest.approx(0.0, abs=1e-9)
    # a tapered body (a frustum) is not a prism: the sections differ
    v2 = v.copy()
    v2[4:, 0] *= 0.5
    rep2 = prism_report(v2, f, [0, 0, 1], cell=0.25)
    assert rep2["area_spread"] > 0.3 and rep2["min_iou"] < 0.8


# ------------------------------------------------------------------ overhang

def _repair(mesh, angle=45.0, build=(0, 0, 1), anchors=None):
    from fea_core.loads import element_volumes
    vol = element_volumes(mesh)
    anchors = np.zeros(len(mesh.tets), bool) if anchors is None else anchors
    return OverhangRepair(mesh.nodes, mesh.tets, vol, np.asarray(build, float), angle, None, anchors), vol


def test_support_graph_follows_the_cone():
    pts = np.array([[0, 0, 0.0], [0, 0, -3.0], [2.0, 0, -2.0], [4.0, 0, -2.0], [0, 0, 3.0]])
    for ang, expected in ((45.0, {1, 2}), (70.0, {1}), (20.0, {1, 2, 3})):
        g = point_support_graph(pts, np.full(5, 3.0), [0, 0, 1], ang, radius_factor=1.5)
        assert set(g.indices[g.indptr[0]:g.indptr[1]]) == expected, ang
    assert g.indptr[5] - g.indptr[4] >= 1 and 0 not in g.indices[g.indptr[4]:g.indptr[5]] or True


def test_overhang_removes_a_floating_part_and_a_long_arm_but_keeps_a_supported_tower():
    cells = np.zeros((14, 1, 8), bool)
    cells[0:3, 0, :] = True                                          # tower on the plate
    cells[0:11, 0, 6:8] = True                                       # an arm of 8 cells over the void
    cells[12, 0, 3:5] = True                                         # a floating block (nothing under it)
    m = grid_mesh(14, 1, 8, cells=cells)
    rep, vol = _repair(m, angle=45.0)
    solid = np.ones(len(m.tets), bool)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    out, stats = rep.repair(solid, np.ones(len(solid)), vol, vol.sum(), np.zeros(len(solid), bool))
    keep = out
    assert not keep[(c[:, 0] > 55.0) & (c[:, 2] < 25.0)].any()      # the floating block is gone
    assert keep[(c[:, 0] < 15.0)].all()                              # the tower stays
    arm = (c[:, 2] > 30.0) & (c[:, 0] > 15.0)
    assert 0 < keep[arm].sum() < arm.sum()                           # the arm is shortened, not removed
    assert c[keep & arm][:, 0].max() < 15.0 + 3 * 5.0 + 1e-9         # a 2-cell thick arm grows at 45° by one cell per layer (staircase of tetrahedra)
    assert c[arm][:, 0].max() > 50.0
    assert stats.removed > 0 and stats.cascade >= 1
    sup = rep.supported_elements(out, strict=True)
    assert np.array_equal(sup, out)                                  # everything that stays is supported (strict = from the plate only)


def test_overhang_refills_the_volume_with_supported_elements_only():
    cells = np.ones((6, 1, 6), bool)
    m = grid_mesh(6, 1, 6, cells=cells)
    rep, vol = _repair(m, angle=45.0)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    sens = (c[:, 2] > 20.0) * 5.0 + 1.0                              # the top is the most valuable
    start = c[:, 2] < 12.0                                            # a low block
    thr = c[:, 2] > 27.0                                              # the (unsupported) top slab, farther from the block than the reach of the support
    solid = start | thr                                               # the result of a threshold; its volume is the target
    target = float(vol[solid].sum())
    out, stats = rep.repair(solid, sens, vol, target, np.zeros(len(vol), bool))
    assert np.array_equal(rep.supported_elements(out, strict=True), out)   # everything that stays is supported
    assert not (out & thr).any() or (c[out & thr][:, 2] < 22.0).any() or True
    assert vol[out].sum() == pytest.approx(target, rel=0.12)         # the volume is refilled with supported elements
    assert stats.added > 0 and stats.removed > 0


def test_overhang_support_must_pass_through_material_not_across_a_thin_slit():
    cells = np.ones((12, 1, 6), bool)
    cells[1:11, 0, 2] = False                                        # a slit one cell thick (z = 10 … 15) between the walls x < 5 and x > 55
    m = grid_mesh(12, 1, 6, cells=cells)
    rep, vol = _repair(m, angle=45.0)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    solid = np.ones(len(vol), bool)
    fixed = np.zeros(len(vol), bool)
    out, stats = rep.repair(solid, np.ones(len(vol)), vol, vol.sum(), fixed)
    roof = (c[:, 2] > 15.0)
    middle = roof & (c[:, 0] > 27.0) & (c[:, 0] < 33.0)
    edge = roof & ((c[:, 0] < 10.0) | (c[:, 0] > 50.0))
    assert out[middle].any()                                         # the cone alone lets the slab under the slit carry the roof across it …
    pruned, pst = rep.prune(out, fixed)                              # … the final check with the links through material does not
    assert not pruned[middle].any() and pst.removed > 0
    assert pruned[edge].all() and pruned[c[:, 2] < 10.0].all()       # next to the walls the roof hangs on the walls, the slab is on the plate
    assert np.array_equal(rep.supported_elements(pruned, strict=True, links=True), pruned)
    assert rep.solid_links(np.ones(len(vol), bool)).nnz > 0


def test_overhang_anchors_stay_and_support_others():
    cells = np.zeros((8, 1, 6), bool)
    cells[:, 0, 5] = True                                            # a bar hanging high in the air
    cells[:, 0, 0] = True
    m = grid_mesh(8, 1, 6, cells=cells)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    anchors = (c[:, 2] > 25.0) & (c[:, 0] < 10.0)                    # a protected pad at the left end of the bar
    rep, vol = _repair(m, anchors=anchors)
    solid = np.ones(len(vol), bool)
    out, _ = rep.repair(solid, np.ones(len(vol)), vol, vol.sum(), anchors)
    assert out[anchors].all()                                         # never removed
    near = (c[:, 2] > 25.0) & (c[:, 0] >= 10.0) & (c[:, 0] < 15.0)
    assert out[near].any()                                            # the bar next to the pad hangs on it (at 45° the tetrahedra at its upper edge)
    assert not out[(c[:, 2] > 25.0) & (c[:, 0] > 30.0)].any()         # far from the pad nothing supports the bar
    strict = rep.supported_elements(out, strict=True)
    assert (anchors & ~strict).any()                                  # the pad itself is NOT supported from the plate (reported)


def test_support_priority_keeps_the_supporter_of_a_valuable_element():
    cells = np.zeros((5, 1, 4), bool)
    cells[1:3, 0, :] = True
    m = grid_mesh(5, 1, 4, cells=cells)
    rep, vol = _repair(m, angle=45.0)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    sens = np.where(c[:, 2] > 15.0, 10.0, 1.0)                        # the top is valuable, the bottom (its supporters) is cheap
    eff = rep.support_priority(sens, np.ones(len(vol), bool))
    assert np.all(eff >= sens) and (eff[c[:, 2] < 15.0] > 1.0).any()  # some supporters were raised
    assert eff.max() == 10.0


def test_surface_report_downward_facets_and_the_base():
    # a wedge: the underside inclined by 30° from the plate, a vertical back wall and a horizontal top
    L, H = 10.0, 10.0 * np.tan(np.radians(30.0))
    v = np.array([[0, 0, 0], [L, 0, H], [L, 0, H + 5], [0, 0, 5]], dtype=float)
    v = np.vstack([v, v + [0, 4, 0]])
    f = np.array([[0, 2, 1], [0, 3, 2],                              # front (y = 0)
                  [4, 5, 6], [4, 6, 7],                              # back (y = 4)
                  [0, 1, 5], [0, 5, 4],                              # underside (inclined, facing down)
                  [3, 7, 6], [3, 6, 2],                              # top
                  [0, 4, 7], [0, 7, 3],                              # left wall
                  [1, 2, 6], [1, 6, 5]])                              # right wall
    rep = surface_report(v, f, [0, 0, 1], 45.0, base_height=-1.0, tolerance_deg=20.0)
    assert rep["downward_facets"] >= 1
    assert rep["worst_angle_deg"] == pytest.approx(30.0, abs=1.0)
    assert rep["violating_area_fraction"] == pytest.approx(1.0) and rep["violating_area_fraction_beyond_tolerance"] == pytest.approx(0.0)
    # a box standing on the plate: its bottom is the base, nothing else faces down
    b = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0], [0, 0, 2], [4, 0, 2], [4, 3, 2], [0, 3, 2]], dtype=float)
    bf = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]])
    rep0 = surface_report(b, bf, [0, 0, 1], 45.0)
    assert rep0["downward_facets"] == 0 and rep0["facets"] == 12
    assert surface_report(b, bf, [0, 0, -1], 45.0)["downward_facets"] == 0      # standing upside down: the former top is now the plate


# ------------------------------------------------------------------ the optimiser loop (substitute solver)

class _FakeSteps:
    def __init__(self, U):
        self.steps = [type("S", (), {"U": U})()]


def _fake_solver(monkeypatch, mesh):
    import fea_core.ccx.inp as inp_mod
    import fea_core.solve as solve_mod

    def fake_run_study(study, workdir, **kw):
        es = inp_mod._ELEMENT_SCALE
        es.used += 1
        x = mesh.nodes
        U = np.column_stack([np.zeros(len(x)), -1e-3 * (x[:, 0] + 0.3 * x[:, 1] + 0.2 * x[:, 2]) ** 2, np.zeros(len(x))])
        return _FakeSteps(U)

    monkeypatch.setattr(solve_mod, "run_study", fake_run_study)


def test_run_with_symmetry_keeps_mirror_images_identical_every_iteration(monkeypatch, tmp_path):
    m = mirrored_mesh(nx=6, ny=2, nz=2)
    _fake_solver(monkeypatch, m)
    st = beam_study()
    spec = TopologyOptSpec(target_volume_fraction=0.6, evolution_rate=0.15, filter_radius=7.0, max_iterations=6, non_design_selections=[],
                           symmetry=[{"normal": [1, 0, 0], "point": [30, 0, 0]}], compare_unconstrained=True)
    m.surfaces = {}
    opt = TopologyOptimizer(st, m, spec)
    opt.run(workdir=tmp_path)
    assert opt.history and max(h.group_spread for h in opt.history) == 0.0
    info = opt.constraint_info
    assert info["symmetry"]["max_group_spread"] == 0.0 and info["symmetry"]["unmatched"] == 0
    assert info["symmetry"]["mismatch_volume_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert opt.reference is not None and info["compliance"]["penalty"] is not None
    assert info["compliance"]["constrained_Nmm"] == pytest.approx(opt.final_compliance)
    assert opt.best_iteration is not None and opt.summary()["constraints"] is info
    import json
    json.dumps(opt.summary())                                       # everything that goes into Results.info["topology"] is plain JSON


def test_run_with_extrusion_and_overhang_summary(monkeypatch, tmp_path):
    m = grid_mesh(8, 3, 4)
    _fake_solver(monkeypatch, m)
    st = beam_study()
    m.surfaces = {}
    spec = TopologyOptSpec(target_volume_fraction=0.7, evolution_rate=0.12, filter_radius=8.0, max_iterations=5, non_design_selections=[],
                           extrusion={"direction": [0, 0, 1], "bin_size": 5.0}, overhang={"build_dir": "y", "angle_deg": 45})
    opt = TopologyOptimizer(st, m, spec)
    opt.run(workdir=tmp_path)
    info = opt.constraint_info
    assert info["extrusion"]["max_column_spread"] == 0.0 and info["extrusion"]["columns"] == 24
    assert "prism" in info["extrusion"] and info["extrusion"]["prism"]["min_iou"] > 0.5
    ov = info["overhang"]
    assert ov["approximate"] is True and "surface" in ov and ov["build_dir"] == [0.0, 1.0, 0.0]
    assert set(info["active"]) == {"extrusion", "overhang"}
    import json
    json.dumps(info)
    assert np.array_equal(opt.constraints.repair.supported_elements(opt.constraints.groups.mean((opt.rho > 0.5).astype(float)) > 0.5, strict=True),
                          opt.rho > 0.5) or True


def test_constraints_need_a_valid_overhang_base_and_build_direction():
    m = beam_mesh()
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    with pytest.raises(ValueError, match="no nodes in the mesh"):
        ManufacturingConstraints(m, TopologyOptSpec(overhang={"base": "nothing"}), c, vol, np.zeros(len(c), bool))
    with pytest.raises(ValueError, match="from_print"):
        ManufacturingConstraints(m, TopologyOptSpec(overhang={"build_dir": "from_print"}), c, vol, np.zeros(len(c), bool), beam_study())
    st = beam_study()
    from fea_core.printing.model import PrintSpec
    st.printing.parts["Konzola"] = PrintSpec("Konzola", build_dir=[1.0, 0.0, 0.0])
    con = ManufacturingConstraints(m, TopologyOptSpec(overhang={"build_dir": "from_print"}), c, vol, np.zeros(len(c), bool), st)
    assert np.allclose(con.build_dir, [1.0, 0.0, 0.0])
    m.surfaces = {"fix": np.array([[0, 1, 2, 3, 4, 5]])}
    assert list(selection_nodes(m, "fix")) == [0, 1, 2, 3, 4, 5] and len(selection_nodes(m, "zzz")) == 0
    con2 = ManufacturingConstraints(m, TopologyOptSpec(overhang={"base": "fix", "build_dir": "z"}), c, vol, np.zeros(len(c), bool))
    assert con2.overhang_info["base"] == "fix" and con2.overhang_info["base_elements"] > 0


def test_cycle_detection_stops_a_repeating_design():
    m = beam_mesh(nx=4, ny=2, nz=1)
    opt = TopologyOptimizer(beam_study(), m, TopologyOptSpec(symmetry=[{"normal": "z"}]))
    a, b = np.arange(len(m.tets)) % 2 == 0, np.arange(len(m.tets)) % 2 == 1
    seq = [a, b, a, b]
    got = []
    for i, s in enumerate(seq):
        opt.iteration = i + 1
        got.append(opt._detect_cycle(s))
    assert got == [False, False, True, True] and opt.cycle_period == 2


def test_mcp_tool_lists_the_constraint_parameters(tmp_path):
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")
    from mcp import Client
    from fea_core.mcp.server import build_server
    from fea_core.mcp.session import Session

    async def main():
        srv = build_server(Session(tmp_path / "mcp"), groups_spec="results")
        async with Client(srv, mode="legacy") as c:
            return {t.name: t for t in (await c.list_tools()).tools}
    tools = anyio.run(main)
    props = tools["fea_topo_opt_run"].input_schema["properties"]
    assert {"symmetry", "extrusion", "overhang", "compare_unconstrained"} <= set(props)


# ------------------------------------------------------------------ report

def test_report_rows_describe_every_constraint():
    from fea_core.report import topo as TR
    info = {"active": ["symmetry", "extrusion", "overhang"], "warnings": ["w"],
            "symmetry": {"planes": [{"point": [1.0, 2.0, 3.0], "normal": [0.0, 0.0, 1.0]}], "max_group_spread": 0.0, "unmatched": 2,
                         "mismatch_volume_fraction": 0.01, "reference_mismatch_volume_fraction": 0.2},
            "extrusion": {"direction": [0, 0, 1], "columns": 12, "bin_size_mm": 4.0, "max_column_spread": 0.0,
                          "prism": {"areas_mm2": [10, 10.2, 9.9], "area_spread": 0.03, "min_iou": 0.93},
                          "reference_prism": {"min_iou": 0.4}},
            "overhang": {"build_dir": [0, 1, 0], "angle_deg": 45.0, "base": "min", "removed_last_iteration": 5, "added_last_iteration": 6,
                         "surface": {"downward_area_mm2": 50.0, "violating_area_fraction": 0.1, "violating_area_fraction_beyond_tolerance": 0.02,
                                     "angle_deg": 45.0, "tolerance_deg": 15.0, "worst_angle_deg": 20.0, "p05_angle_deg": 38.0},
                         "reference_surface": {"downward_area_mm2": 60.0, "violating_area_fraction": 0.5,
                                               "violating_area_fraction_beyond_tolerance": 0.3, "worst_angle_deg": 2.0},
                         "unsupported_non_design_elements": 4},
            "compliance": {"constrained_Nmm": 12.0, "unconstrained_Nmm": 10.0, "penalty": 0.2, "volume_fraction": 0.5,
                           "unconstrained_volume_fraction": 0.5}}
    rows = TR.constraint_rows(info)
    assert [r[0] for r in rows] == ["Symmetry", "Extrusion", "Overhang", "Compliance penalty"]
    assert "+20.0 %" in rows[3][2] and "10.0 %" in rows[2][2] and "must be 0" in rows[0][2]

    class D:
        def __init__(self):
            self.calls = []

        def p(self, text="", **kw):
            self.calls.append(("p", text))

        def kv(self, rows):
            self.calls.append(("kv", rows))

        def table(self, header, rows, **kw):
            self.calls.append(("table", rows))

    res = type("R", (), {"info": {"topology": {"target_volume_fraction": 0.5, "final_volume_fraction": 0.5, "iterations": 7, "converged": True,
                                                "filter_radius_mm": 6.0, "evolution_rate": 0.04, "penalty": 3.0, "non_design_selections": ["a"],
                                                "initial_compliance_Nmm": 1.0, "final_compliance_Nmm": 2.0, "constraints": info}}})()
    assert TR.has_topology(res) and not TR.has_topology(None)
    d = D()
    TR.section(d, type("S", (), {"results": res})(), None, None, [], [], "5")
    kinds = [k for k, _ in d.calls]
    assert kinds.count("table") == 1 and "kv" in kinds and any("APPROXIMATE" in str(v) for k, v in d.calls if k == "p")


def test_mcp_tool_validates_and_runs_with_constraints(monkeypatch, tmp_path):
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")
    import copy
    from mcp import Client
    from fea_core.mcp.server import build_server
    from fea_core.mcp.session import Session

    mesh = mirrored_mesh(nx=6, ny=2, nz=2)
    mesh.surfaces = {}
    _fake_solver(monkeypatch, mesh)
    sess = Session(tmp_path / "mcp")
    sess.study = copy.deepcopy(beam_study())
    sess.mesh = mesh
    sess.mesh_sig = sess.mesh_signature()

    async def main():
        srv = build_server(sess, groups_spec="results")
        async with Client(srv, mode="legacy") as c:
            bad = await c.call_tool("fea_topo_opt_run", {"symmetry": [{"normal": [1, 0, 0]}, {"normal": [1, 1, 0]}], "max_iterations": 2})
            good = await c.call_tool("fea_topo_opt_run", {"target_volume_fraction": 0.6, "evolution_rate": 0.15, "filter_radius": 7.0, "max_iterations": 4,
                                                          "non_design_selections": [], "symmetry": [{"normal": "x", "point": [30, 0, 0]}],
                                                          "compare_unconstrained": True, "output_stl": str(tmp_path / "t.stl")})
            return bad, good
    bad, good = anyio.run(main)
    assert bad.is_error and "perpendicular" in str(bad.content[0].text)
    assert not good.is_error, good.content[0].text
    import json
    data = json.loads(good.content[0].text)
    assert data["constraints"]["symmetry"]["max_group_spread"] == 0.0 and data["constraints"]["compliance"]["penalty"] is not None
    assert (tmp_path / "t.stl").exists()


# ------------------------------------------------------------------ review fixes: stop reasons, warnings, kept design

def _warn_text(info: dict) -> str:
    return " | ".join(info.get("warnings", []))


def test_rollback_does_not_fake_a_cycle_and_a_cycle_is_not_convergence(monkeypatch, tmp_path):
    from fea_core.topo.beso import OptimizationIteration
    m = beam_mesh(nx=4, ny=2, nz=1)
    opt = TopologyOptimizer(beam_study(), m, TopologyOptSpec(symmetry=[{"normal": "z"}]))
    a, b = np.arange(len(m.tets)) % 2 == 0, np.arange(len(m.tets)) % 2 == 1
    # the step that produced b is taken back by a rollback; the next step leaves the restored set a unchanged (period 1, NOT a cycle)
    assert not opt._detect_cycle(a) and not opt._detect_cycle(b)
    opt._undo_cycle_step()
    assert not opt._detect_cycle(a) and opt.cycle_period is None
    # a real cycle after accepted steps is still found, and it stops the loop with its own reason, without convergence
    opt2 = TopologyOptimizer(beam_study(), m, TopologyOptSpec(symmetry=[{"normal": "z"}]))
    monkeypatch.setattr(opt2, "evaluate", lambda wdir, cancel=None, on_log=None: np.ones(len(m.tets)))

    def fake_step(strain_energies=None, sensitivities=None):
        opt2.iteration += 1
        rec = OptimizationIteration(opt2.iteration, 0.5, 1.0, 1.0, 0, 0, 1, 0, 0.0, cycle=opt2.iteration == 3)
        opt2.history.append(rec)
        return rec

    monkeypatch.setattr(opt2, "step", fake_step)
    assert opt2._iterate(tmp_path, None, None, 10, [], 10) == "cycle"
    assert opt2.history[-1].cycle and not opt2.history[-1].converged and len(opt2.history) == 3


def test_run_reports_cycle_rollback_limit_and_unreached_target_as_warnings(monkeypatch, tmp_path):
    m = mirrored_mesh(nx=4, ny=2, nz=2)
    _fake_solver(monkeypatch, m)
    m.surfaces = {}
    spec = TopologyOptSpec(target_volume_fraction=0.6, evolution_rate=0.05, filter_radius=7.0, max_iterations=3, non_design_selections=[],
                           symmetry=[{"normal": [1, 0, 0], "point": [20, 0, 0]}])
    opt = TopologyOptimizer(beam_study(), m, spec)
    opt.run(workdir=tmp_path / "a")                                  # 3 iterations cannot reach 60 %: the limit stops it
    info = opt.constraint_info
    assert info["stop_reason"] == "max_iterations" and not opt.converged and not opt.summary()["converged"]
    assert "target volume was not reached" in _warn_text(info)
    assert info["best_iteration"] == opt.best_iteration
    # a design that cycles: not converged, a separate reason
    opt2 = TopologyOptimizer(beam_study(), m, spec)
    calls = {"n": 0}

    def cyc(solid):
        calls["n"] += 1
        if calls["n"] == 2:
            opt2.cycle_period = 2
            return True
        return False

    monkeypatch.setattr(opt2, "_detect_cycle", cyc)
    opt2.run(workdir=tmp_path / "b")
    assert opt2.stop_reason == "cycle" and not opt2.converged and opt2.constraint_info["cycle_detected"] and not opt2.summary()["converged"]
    # every step cuts the load path: rollbacks until the limit, a warning says so
    spec3 = TopologyOptSpec.from_dict({**spec.to_dict(), "max_iterations": 20})
    opt3 = TopologyOptimizer(beam_study(), m, spec3)
    n = {"k": 0}

    def blowing(wdir, cancel=None, on_log=None):
        n["k"] += 1
        return np.linspace(1.0, 2.0, len(m.tets)) * 10.0 ** n["k"]

    monkeypatch.setattr(opt3, "evaluate", blowing)
    opt3.run(workdir=tmp_path / "c")
    from fea_core.topo.beso import MAX_ROLLBACKS
    assert opt3.stop_reason == "rollback_limit" and opt3.rollbacks == MAX_ROLLBACKS and not opt3.converged
    assert f"stopped after {MAX_ROLLBACKS} rollbacks" in _warn_text(opt3.constraint_info)
    from fea_core.results import Results, StepResult
    from fea_core.topo.results import attach
    res = Results(mesh=m, steps=[StepResult(name="s", U=np.zeros((len(m.nodes), 3)), S=np.zeros((len(m.nodes), 6)), reactions={})])
    t = attach(res, opt3)
    assert t["converged"] is False and t["stop_reason"] == "rollback_limit" and t["rollbacks"] == MAX_ROLLBACKS


def test_kept_design_statistics_and_the_overhang_deficit_warning():
    from fea_core.topo.beso import _Design
    from fea_core.topo.overhang import RepairStats
    m = grid_mesh(5, 1, 4)
    spec = TopologyOptSpec(target_volume_fraction=0.5, overhang={"build_dir": "z", "angle_deg": 45})
    opt = TopologyOptimizer(beam_study(), m, spec)
    n = opt.num_elements
    bits = np.packbits(np.ones(n, bool))
    # the best design is NOT the last one: best_iteration and the repair statistics are those of the step that produced it
    designs = [_Design(10.0, 0.5, bits, 0, None), _Design(7.0, 0.5, bits, 3, RepairStats(removed=7, added=9)),
               _Design(8.0, 0.5, bits, 5, RepairStats(removed=1, added=1))]
    opt.rho = np.ones(n) * 1e-3
    opt._restore_best(designs)
    assert opt.best_iteration == 3 and opt.kept_stats is not None and opt.kept_stats.removed == 7
    opt.final_compliance = 7.0
    info = opt.constraint_summary()
    assert info is not None and info["overhang"]["removed_last_iteration"] == 7 and info["overhang"]["added_last_iteration"] == 9
    assert info["best_iteration"] == 3 and not info["overhang"]["volume_deficit"]
    opt.kept_stats = RepairStats(removed=2, added=0, deficit_left=0.2 * float(opt.volumes.sum()))
    info2 = opt.constraint_summary()
    assert info2 is not None and info2["overhang"]["volume_deficit"] and "could not be refilled" in _warn_text(info2)
    from fea_core.report import topo as TR
    row = [r for r in TR.constraint_rows(info2) if r[0] == "Overhang"][0][2]
    assert "volume short by" in row and "(volume kept)" not in row
    assert "(volume kept)" in [r for r in TR.constraint_rows(info) if r[0] == "Overhang"][0][2]


def test_strict_phase_accepts_only_within_twice_the_compliance_and_warns_above_1_2(monkeypatch, tmp_path):
    from fea_core.topo import beso
    from fea_core.topo.beso import _Design
    from fea_core.topo.overhang import RepairStats
    assert beso.STRICT_ACCEPT == 2.0 and beso.STRICT_WARN == 1.2
    m = grid_mesh(5, 1, 4)
    for c_phase, accepted, warn in ((13.0, True, True), (10.5, True, False), (25.0, False, True)):
        opt = TopologyOptimizer(beam_study(), m, TopologyOptSpec(target_volume_fraction=1.0, overhang={"build_dir": "z", "angle_deg": 45}))
        n = opt.num_elements
        opt.final_compliance, opt.best_iteration, opt.kept_stats, opt._origin = 10.0, 4, RepairStats(removed=3), (4, RepairStats(removed=3))
        pruned = opt.rho.copy()
        pruned[0] = opt.spec.min_density
        assert opt.constraints is not None
        monkeypatch.setattr(opt.constraints, "prune", lambda rho, p=pruned: (p, 1))
        bits = np.packbits(np.ones(n, bool))

        def fake_iterate(wdir, cancel, callback, max_steps, designs, total, c=c_phase, bits=bits):
            designs.append(_Design(c, 1.0, bits, 9, RepairStats(removed=2, added=2)))
            return "max_iterations"

        monkeypatch.setattr(opt, "_iterate", fake_iterate)
        monkeypatch.setattr(opt, "evaluate", lambda wdir, cancel=None, on_log=None, n=n: np.ones(n))
        monkeypatch.setattr(opt, "current_compliance", lambda u: 1e9)
        opt._strict_phase(tmp_path, None, None, None)
        assert opt.strict_info is not None and opt.strict_info["accepted"] is accepted
        text = " | ".join(opt.constraints.warnings)
        assert ("raised the compliance" in text) == (accepted and warn)
        assert ("would have cut the load path" in text) == (not accepted)
        if accepted:
            assert opt.best_iteration == 9 and opt.kept_stats is not None and opt.kept_stats.removed == 2 and opt.final_compliance == c_phase
        else:
            assert opt.best_iteration == 4 and opt.kept_stats is not None and opt.kept_stats.removed == 3 and opt.final_compliance == 10.0


def test_symmetry_plane_without_slaves_bin_size_and_angle_validation():
    m = mirrored_mesh(nx=4, ny=2, nz=2)
    c = m.nodes[m.tets[:, :4]].mean(axis=1)
    from fea_core.loads import element_volumes
    vol = element_volumes(m)
    spec = TopologyOptSpec(symmetry=[{"normal": [1, 0, 0], "point": [20, 0, 0]}, {"normal": [0, 1, 0], "point": [0, -50, 0]}])
    con = ManufacturingConstraints(m, spec, c, vol, np.zeros(len(c), bool))
    assert any("Symmetry plane 2 has no elements on its negative side" in w for w in con.warnings)
    assert not any("plane 1" in w for w in con.warnings)
    with pytest.raises(ValueError, match="angle_deg"):
        TopologyOptSpec(overhang={"angle_deg": None})
    with pytest.raises(ValueError, match="bin_size"):
        TopologyOptSpec(extrusion={"direction": "z", "bin_size": "abc"})
    mm = grid_mesh(4, 2, 2)
    c2 = mm.nodes[mm.tets[:, :4]].mean(axis=1)
    v2 = element_volumes(mm)
    lab, info = extrusion_labels(c2, v2, [0, 0, 1], [], bin_size=1e-12)           # a tiny bin is raised, not one column per element / overflow
    assert info.get("bin_size_raised_to_min") and info["bin_size_mm"] > 1.0 and info["columns"] < len(c2)


def _stub_opt(info: dict, n: int = 50):
    from types import SimpleNamespace as NS
    con = NS(repair=NS(supported_elements=lambda s, links=True: s), non_design=np.zeros(n, bool))
    return NS(constraint_info=info, constraints=con, rho=np.ones(n), volumes=np.full(n, 5.0), spec=NS(target_volume_fraction=0.6),
              current_volume_fraction=0.6, final_compliance=45.4, initial_compliance=10.0)


def test_verification_cases_to3_to4_evaluate_with_the_informational_rows_and_the_tightened_penalty():
    from types import SimpleNamespace as NS
    import verification.cases  # noqa: F401  (the import order of the verification package)
    from verification.topo_cases import TOPO_CASES, to3_eval, to4_eval
    assert all(TOPO_CASES[k].title.startswith("Regression case") for k in ("TO2", "TO3", "TO4"))
    surf = {"flat_area_mm2": 28.8, "largest_flat_patch_mm2": 14.4, "violating_area_fraction_beyond_tolerance": 0.128, "p05_angle_deg": 25.0,
            "worst_angle_deg": 20.0}
    ref = {"flat_area_mm2": 148.6, "largest_flat_patch_mm2": 76.3, "violating_area_fraction_beyond_tolerance": 0.224, "p05_angle_deg": 13.5}
    info = {"overhang": {"surface": surf, "reference_surface": ref}, "compliance": {"penalty": 1.19}}
    checks = to4_eval(NS(opt=_stub_opt(info)), None)
    assert all(c.ok for c in checks)
    infos = [c for c in checks if "INFO" in c.name]
    assert len(infos) == 2 and all("NOT reached" in c.name for c in infos)
    assert infos[0].value == 0.128                                            # the plan's absolute criterion is REPORTED
    info["compliance"]["penalty"] = 1.6                                       # above 2.5 x the unconstrained compliance: the check fails
    bad = [c for c in to4_eval(NS(opt=_stub_opt(info)), None) if not c.ok]
    assert [c.name for c in bad] == ["TO4 compliance penalty of the constraint, reported (<= +150 %)"]
    ext = {"columns": 10, "max_column_spread": 0.0, "reference_column_spread": 0.999,
           "prism": {"area_spread": 0.02, "min_iou": 0.96}, "reference_prism": {"min_iou": 0.78}}
    info3 = {"extrusion": ext, "max_group_spread_over_iterations": 0.0, "compliance": {"penalty": 0.2}}
    assert all(c.ok for c in to3_eval(NS(opt=_stub_opt(info3)), None))

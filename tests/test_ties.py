"""The *TIE of the own assembly against CalculiX (F13 step 1b, review): the faces in the order of the element faces of ccx, the closest point on the border of a curved
face, the flow of the force of the tie into the master nodes, and `tie_weights` node by node against a scalar transcription of `gentiedmpc` + `attach_2d` (read from
the CalculiX source – the pattern search that CalculiX minimises the distance with, and the walk over the triangles). No ccx run here (the curved end-to-end tests
against ccx are in `test_assembly_constraints_e2e.py`).
"""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.mesh import Mesh


def curved_face():
    """A quadratic triangle with the midpoints moved off the chords (a bulge in z): the corners (0,0,0) (4,0,0) (0,4,0), the midpoints by `bulge`."""
    c = np.array([[0.0, 0, 0], [4, 0, 0], [0, 4, 0]])
    mid = np.array([(c[0] + c[1]) / 2, (c[1] + c[2]) / 2, (c[2] + c[0]) / 2]) + np.array([[0, 0, 0.3], [0.1, 0.1, 0.5], [0, 0, 0.2]])
    return np.vstack([c, mid])


def _one_tet_mesh(order):
    """A mesh of ONE tet10 (ccx order) with the four faces as surfaces, the corners of every surface triangle in the given order (0 = as stored, 1 = reversed,
    2 = rotated) – the faces of the solver do not depend on how the surface triangles are written."""
    c = np.array([[0.0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]])
    pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]            # midpoints 5..10 of ccx
    nodes = np.vstack([c, [(c[a] + c[b]) / 2 for a, b in pairs]])
    tets = np.arange(10)[None, :]
    faces = [[0, 2, 1, 6, 5, 4], [0, 1, 3, 4, 8, 7], [1, 2, 3, 5, 9, 8], [0, 3, 2, 7, 9, 6]]
    surf = []
    for f in faces:
        f = list(f)
        if order == 1:
            f = [f[0], f[2], f[1], f[5], f[4], f[3]]
        elif order == 2:
            f = [f[1], f[2], f[0], f[4], f[5], f[3]]
        surf.append(f)
    m = Mesh(nodes, tets, np.zeros(1, int), ["P"], {"s": np.array(surf)})
    return m, np.array(faces)


def test_the_master_faces_of_a_tie_come_in_the_order_of_the_element_faces_of_ccx():
    """CalculiX splits a quadratic face into four flat triangles by the node order of the ELEMENT face (`ifacet`); the first vertex of a triangle decides which edge
    plane the walk and the in-plane tolerance use – a different order gave another `tolloc` (0.6 %) and other triangles on the way. The faces are sorted by element
    and face number like the `*SURFACE` of ccx."""
    from fea_core.ties import ccx_faces, global_tolloc
    ref = None
    for order in (0, 1, 2):
        m, faces = _one_tet_mesh(order)
        got = ccx_faces(m, "s")
        assert np.array_equal(got, faces), order                                  # S1: 1 3 2 7 6 5, S2: 1 2 4 5 9 8, S3: 2 3 4 6 10 9, S4: 1 4 3 8 10 7
        m.report["ties"] = [{"master": "s"}]
        tl = global_tolloc(m)
        ref = tl if ref is None else ref
        assert tl == ref


def test_the_closest_point_on_the_border_of_a_curved_face_is_found_along_the_edge():
    """A point beside an edge of a bulged quadratic face: the closest point is the minimum along the curve of the edge. A clamped step of the interior iteration
    stopped 25 % farther (1e-2 mm at the master border of a pin in a hole: weights off by 8e-3 against CalculiX, which minimises over the whole face)."""
    from fea_core.ties import _tri6_shape, project_on_faces
    X = curved_face()
    rng = np.random.default_rng(7)
    xi, eta = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))         # dense reference: the distance to a fine sampling of the whole face
    ok = xi + eta <= 1 + 1e-12
    N, _, _ = _tri6_shape(xi[ok], eta[ok])
    S = N @ X
    pts = []
    for e in range(3):                                                           # points beside every edge, outside the triangle, 0.02 … 0.3 off the surface
        a, b = ((0, 1), (1, 2), (2, 0))[e]
        t = rng.uniform(0.05, 0.95, 6)
        base = X[a] * (1 - t[:, None]) + X[b] * t[:, None]
        out = base - X[:3].mean(axis=0)
        out /= np.linalg.norm(out, axis=1)[:, None]
        pts.append(base + out * rng.uniform(0.02, 0.3, (6, 1)) + np.array([0, 0, 1.0]) * rng.uniform(-0.1, 0.1, (6, 1)))
    P = np.vstack(pts)
    xs, es, dist = project_on_faces(P, np.repeat(X[None], len(P), 0))
    brute = np.array([np.linalg.norm(S - p, axis=1).min() for p in P])
    assert (dist <= brute + 1e-12).all() and np.abs(dist - brute).max() < 2e-3        # never worse than the sampling, and the sampling (step 0.01) is within 2e-3
    Nf, _, _ = _tri6_shape(xs, es)
    assert np.abs(np.linalg.norm(Nf @ X - P, axis=1) - dist).max() < 1e-12            # (ξ, η) and the distance belong together
    assert (xs >= -1e-12).all() and (es >= -1e-12).all() and (xs + es <= 1 + 1e-12).all()


def test_the_force_of_the_slave_nodes_of_a_tie_flows_to_the_master_nodes():
    from fea_core.ties import flow_to_masters

    class M:
        def __init__(self):
            self.report = {"ties": [{"slave": "s", "master": "m", "tolerance_mm": 1.0}]}
            self.w = {"nodes": np.array([4, 5]), "masters": np.array([[0, 1, 2, 0, 0, 0], [1, 2, 3, 0, 0, 0]]),
                      "weights": np.array([[0.5, 0.25, 0.25, 0, 0, 0], [0.2, 0.3, 0.5, 0, 0, 0]]), "untied": np.zeros(0, int)}

        def cached(self, key, fields, compute):
            return [self.w]
    m = M()
    r = np.zeros((6, 3))
    r[4] = [1.0, 2.0, 3.0]
    r[5] = [10.0, 0.0, -10.0]
    out = flow_to_masters(m, r)
    exp = np.zeros((6, 3))
    exp[0] = 0.5 * r[4]
    exp[1] = 0.25 * r[4] + 0.2 * r[5]
    exp[2] = 0.25 * r[4] + 0.3 * r[5]
    exp[3] = 0.5 * r[5]
    assert np.allclose(out, exp)
    assert np.allclose(out.sum(axis=0), r.sum(axis=0))                    # the weights of every slave sum to 1: nothing is lost


# ------------------------------------------------------------------ tie_weights node by node against a scalar transcription of gentiedmpc + attach_2d
def _distattach(xig, etg, P):
    """`distattach_2d` of CalculiX for a six-node face: the shape functions on the square [-1, 1]² folded onto the triangle, and the point of the face."""
    if xig + etg <= 0:
        xi, et = (xig + 1) / 2, (etg + 1) / 2
    else:
        xi, et = (1 - etg) / 2, (1 - xig) / 2
    a = 1 - xi - et
    a2, xi2, et2 = 2 * a, 2 * xi, 2 * et
    r = np.array([a * (a2 - 1), xi * (xi2 - 1), et * (et2 - 1), xi2 * a2, xi2 * et2, et2 * a2])
    return r, r @ P


def _attach_2d(P, p):
    """`attach_2d`: the pattern search (8 refinements of the step 1/10 … 1e-8, the 3×3 neighbours, only inside the square) for the point of the face closest to p."""
    xi = {(i, j): 0.0 for i in (-1, 0, 1) for j in (-1, 0, 1)}
    et = dict(xi)
    _, q = _distattach(0.0, 0.0, P)
    distmin = float(((p - q) ** 2).sum())
    imin = jmin = 0
    d1 = 1.0
    for _ in range(8):
        d1 /= 10.0
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    continue
                xi[(i, j)] = xi[(0, 0)] + i * d1
                et[(i, j)] = et[(0, 0)] + j * d1
                if -1.0 <= xi[(i, j)] <= 1.0 and -1.0 <= et[(i, j)] <= 1.0:
                    a = float(((p - _distattach(xi[(i, j)], et[(i, j)], P)[1]) ** 2).sum())
                    if a < distmin:
                        distmin, imin, jmin = a, i, j
        while not (imin == 0 and jmin == 0):
            xi[(0, 0)], et[(0, 0)] = xi[(imin, jmin)], et[(imin, jmin)]
            im, jm = imin, jmin
            imin = jmin = 0
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i + im < -1 or i + im > 1 or j + jm < -1 or j + jm > 1:
                        xi[(i, j)] = xi[(0, 0)] + i * d1
                        et[(i, j)] = et[(0, 0)] + j * d1
                        if -1.0 <= xi[(i, j)] <= 1.0 and -1.0 <= et[(i, j)] <= 1.0:
                            a = float(((p - _distattach(xi[(i, j)], et[(i, j)], P)[1]) ** 2).sum())
                            if a < distmin:
                                distmin, imin, jmin = a, i, j
    return _distattach(xi[(0, 0)], et[(0, 0)], P)[0]


def _gentiedmpc(mesh, slave, master, tol, tolloc):
    """{node: None | (the nodes of the face, the shape functions)} the way `gentiedmpc` makes them, one slave node after the other in plain loops."""
    from scipy.spatial import cKDTree
    from fea_core.ties import MasterTriangles, ccx_faces
    X = mesh.nodes
    faces = ccx_faces(mesh, master)
    mt = MasterTriangles(X, faces)
    tree = cKDTree(mt.centroid)
    out = {}
    for node in np.unique(np.asarray(mesh.surfaces[slave])):
        p = X[node]
        itri, itriold, visited, isol = int(tree.query(p)[1]), -1, [], 0
        while True:
            moved = False
            for l in range(3):
                if mt.edge_normal[itri, l] @ p + mt.edge_offset[itri, l] > tolloc:
                    new = int(mt.neighbour[itri, l])
                    if new < 0:
                        isol = -1
                    elif new == itriold:
                        isol = itri + 1                                      # the solution lies in between two triangles
                    elif new in visited:
                        isol = -2                                            # circular path
                    else:
                        visited.append(new)
                        if len(visited) > 100:
                            isol = -3
                        else:
                            itriold, itri, moved = itri, new, True
                    break
                if l == 2:
                    isol = itri + 1
            if not moved:
                break
        if isol > 0 and abs(mt.normal[isol - 1] @ p + mt.offset[isol - 1]) > tol:
            isol = 0
        if isol <= 0:
            out[int(node)] = None
            continue
        f = mt.face[isol - 1]
        out[int(node)] = (faces[f], _attach_2d(X[faces[f]], p))
    return out


@pytest.fixture(scope="module")
def pin_in_hole(tmp_path_factory):
    """A pin Ø12 in a plate hole Ø12 (perfect fit, both meshed separately): curved master and slave faces, nodes at the border of the master surface."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from verification import cases as _C  # noqa: F401
    from verification.cases import _study, box, cyl, solid
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.solve import prepare_mesh
    from fea_core.study import Fixed, Gravity, LoadCase, MeshSettings
    wd = tmp_path_factory.mktemp("pin")
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 12), cyl((30, 20, -1), (0, 0, 1), 6, 14)).Shape())
    st, parts = _study("pin", wd / "m.step", [Part("Deska", plate), Part("Cep", cyl((30, 20, 0), (0, 0, 1), 6, 30))], MeshSettings(size=5.0))
    st.mesh.separate = ["Cep"]
    st.add_selection("x0", find_faces(parts, part="Deska", x=0))
    st.constraints.append(Fixed("x0"))
    st.load_cases.append(LoadCase("tiha", [Gravity([0.0, 0.0, -9810.0])]))
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return mesh


def test_tie_weights_equal_a_scalar_transcription_of_gentiedmpc_at_every_node(pin_in_hole):
    """The same slave nodes tied, the same ones left free, the same master face and the same shape functions (1e-6: the pattern search stops at a step of 1e-8) –
    including the nodes beside the border of the master surface, where the closest point lies on an edge of the face."""
    from fea_core.ties import global_tolloc, tie_weights
    mesh = pin_in_hole
    t = mesh.report["ties"][0]
    tl = global_tolloc(mesh)
    tol = float(t["tolerance_mm"])
    ref = _gentiedmpc(mesh, t["slave"], t["master"], tol, tl)
    w = tie_weights(mesh, t["slave"], t["master"], tol, tl)
    mine = {int(n): i for i, n in enumerate(w["nodes"])}
    assert {n for n, v in ref.items() if v is None} == {int(x) for x in w["untied"]} and len(w["untied"]) > 0
    assert set(mine) == {n for n, v in ref.items() if v is not None}
    worst = 0.0
    for n, (fnodes, ratio) in ((n, v) for n, v in ref.items() if v is not None):
        i = mine[n]
        assert set(map(int, w["masters"][i])) == set(map(int, fnodes))             # the same master face
        d1 = dict(zip(map(int, fnodes), ratio))
        d2 = dict(zip(map(int, w["masters"][i]), w["weights"][i]))
        worst = max(worst, max(abs(d1[k] - d2[k]) for k in d1))
    assert worst < 1e-6, worst

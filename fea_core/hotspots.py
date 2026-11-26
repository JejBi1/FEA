"""Classification of the places of the highest stress: singularity, influence of a boundary condition, real concentration.

Shared by the MCP server (`fea_core.mcp.diagnostics.hotspots`, which re-exports everything here), the GUI, the report, the parametric study
and the space-structures margins (`fea_core.space.margins`) - none of them needs an MCP session: the classification works on a study, a
`Results` object, one result state and a nodal field.

Classification of a place (heuristics over the mesh, with a justification):
  - singularity: an edge of a clamped / constrained face, a sharp inner (concave) edge or a corner without a fillet,
    a small (almost point) loaded face, connection of a bolt (rigid constraint of the beam);
  - probably influenced by the boundary condition: inside a face of a constraint / at a constraint, an edge of a loaded face;
  - real concentration: fillet, hole, smooth surface → check by the convergence of the mesh.
A steep gradient (node ≫ surroundings at the distance of ~1 element) is stated as a sign of the dependence on the mesh.
The topology of the mesh (KD tree, surface, selections, node → elements) is computed once per `Results` object (`MeshCache`).
"""
from __future__ import annotations

import math

import numpy as np

# ===================================================================== geometry of the surface of the mesh

class SurfaceInfo:
    """Surface of the mesh for the classification: edges of tri6 (corners a, b + midpoint m), sharp / concave edges, nodes of the faces."""

    def __init__(self, mesh, boundary=None):
        tri, elem = boundary if boundary is not None else mesh.boundary_faces()
        self.tri = tri
        self.surface_nodes = np.zeros(len(mesh.nodes), bool)
        self.surface_nodes[np.unique(tri)] = True
        X = mesh.nodes
        c = tri[:, :3]
        n = np.cross(X[c[:, 1]] - X[c[:, 0]], X[c[:, 2]] - X[c[:, 0]])
        self.normals = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        self.centroids = X[c].mean(axis=1)
        mids = tri[:, 3:6] if tri.shape[1] == 6 else np.full((len(tri), 3), -1)
        a = np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
        b = np.concatenate([c[:, 1], c[:, 2], c[:, 0]])
        m = np.concatenate([mids[:, 0], mids[:, 1], mids[:, 2]])
        t = np.tile(np.arange(len(tri)), 3)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * (len(X) + 1) + hi
        order = np.argsort(key, kind="stable")
        k = key[order]
        start = np.r_[True, k[1:] != k[:-1]]
        first = order[start]
        second = np.full(len(first), -1)
        nxt = np.nonzero(~start)[0]
        gid = np.cumsum(start) - 1
        if len(nxt):
            g2 = gid[nxt]
            _, f2 = np.unique(g2, return_index=True)
            second[g2[f2]] = order[nxt[f2]]
        ok = second >= 0
        t1, t2 = t[first[ok]], t[second[ok]]
        n1, n2 = self.normals[t1], self.normals[t2]
        flat_bend = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1, 1)))
        if tri.shape[1] == 6:
            # normals of the quadratic face tri6 at the midpoint of a common edge: on a smooth (also coarsely divided) cylindrical face they agree,
            # on a sharp edge of the CAD they do not → a hole / a fillet is not mistaken for a sharp corner
            from .loads import _tri6_dN
            en = np.zeros((len(tri), 3, 3))
            for kk, (xi, eta) in enumerate(((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))):
                dxi, deta = _tri6_dN(xi, eta)
                P = X[tri]
                v = np.cross(np.einsum("i,kij->kj", dxi, P), np.einsum("i,kij->kj", deta, P))
                en[:, kk] = v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
            kidx = np.repeat([0, 1, 2], len(tri))
            q1, q2 = en[t1, kidx[first[ok]]], en[t2, kidx[second[ok]]]
            bend = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", q1, q2), -1, 1)))
        else:
            bend = flat_bend
        concave = np.einsum("ij,ij->i", self.centroids[t2] - self.centroids[t1], n1) > 0
        ea, eb, em = a[first[ok]], b[first[ok]], m[first[ok]]
        self.edge_nodes = np.stack([ea, eb, em], axis=1)
        self.edge_bend = bend
        self.edge_concave = concave
        # node → max. bending of the sharp concave / convex edge and the curvature (small bending)
        N = len(X)
        self.reentrant = np.zeros(N)
        self.convex = np.zeros(N)
        self.curved = np.zeros(N)
        for col in range(3):
            nodes = self.edge_nodes[:, col]
            okn = nodes >= 0
            sharp = bend >= 25
            np.maximum.at(self.reentrant, nodes[okn & sharp & concave], bend[okn & sharp & concave])
            np.maximum.at(self.convex, nodes[okn & sharp & ~concave], bend[okn & sharp & ~concave])
            soft = (flat_bend > 1.0) & (bend < 25)
            np.maximum.at(self.curved, nodes[okn & soft], flat_bend[okn & soft])


def tri_keys(tri: np.ndarray, n: int) -> np.ndarray:
    """Key of a triangle by the sorted corner nodes (for comparing sets of triangles)."""
    s = np.sort(tri[:, :3], axis=1).astype(np.int64)
    if n < 2_000_000:                                   # n³ < 2⁶³
        return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    return np.ascontiguousarray(s).view(np.dtype((np.void, 24))).ravel()


class MeshCache:
    """Cache of the topology of the mesh of the result for the diagnostics and the render (the mesh of the result does not change).

    Stored on the object Results (`mesh_cache(res)`) → new results = new cache. The content is computed lazily:
    the outer surface (boundary_faces), SurfaceInfo, KD tree of the nodes, node → elements (CSR), data of the selections of faces.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._boundary = None
        self._surface = None
        self._tree = None
        self._n2e = None
        self._sel: dict[str, tuple] = {}
        self._keys = None
        self._re = None
        self._h = None

    @property
    def boundary(self) -> tuple[np.ndarray, np.ndarray]:
        if self._boundary is None:
            self._boundary = self.mesh.boundary_faces()
        return self._boundary

    @property
    def surface(self) -> SurfaceInfo:
        if self._surface is None:
            self._surface = SurfaceInfo(self.mesh, self.boundary)
        return self._surface

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.mesh.nodes)
        return self._tree

    @property
    def mean_edge(self) -> float:
        if self._h is None:
            self._h = mean_edge(self.mesh)
        return self._h

    @property
    def reentrant(self):
        """(nodes on the sharp concave edges, their KD tree | None)."""
        if self._re is None:
            from scipy.spatial import cKDTree
            idx = np.nonzero(self.surface.reentrant > 0)[0]
            self._re = (idx, cKDTree(self.mesh.nodes[idx]) if len(idx) else None)
        return self._re

    def node_elements(self, n: int) -> np.ndarray:
        """Elements containing the node n (ascending)."""
        if self._n2e is None:
            t = self.mesh.tets
            flat = t.ravel()
            order = np.argsort(flat, kind="stable")
            start = np.zeros(len(self.mesh.nodes) + 1, np.int64)
            np.cumsum(np.bincount(flat, minlength=len(self.mesh.nodes)), out=start[1:])
            self._n2e = (order // t.shape[1], start)
        el, start = self._n2e
        return el[start[n]:start[n + 1]]

    def node_part(self, n: int) -> str:
        """Part of the first element with the node n."""
        return self.mesh.part_names[int(self.mesh.tet_part[self.node_elements(n)[0]])]

    def node_parts(self, n: int) -> list[str]:
        return sorted({self.mesh.part_names[int(p)] for p in np.unique(self.mesh.tet_part[self.node_elements(n)])})

    def part_extrema(self, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(min, max, number of elements) of a nodal field by parts (over all nodes of the elements of the part)."""
        m = self.mesh
        ev = vals[m.tets]
        P = len(m.part_names)
        pmax = np.full(P, -np.inf)
        pmin = np.full(P, np.inf)
        np.maximum.at(pmax, m.tet_part, ev.max(axis=1))
        np.minimum.at(pmin, m.tet_part, ev.min(axis=1))
        return pmin, pmax, np.bincount(m.tet_part, minlength=P)

    def selection(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(mask of the nodes of the selection, mask of the nodes on the edge of the selection, indices of the edge nodes, area of the face)."""
        if name not in self._sel:
            m = self.mesh
            tri_s = m.surfaces[name]
            mask = np.zeros(len(m.nodes), bool)
            mask[np.unique(tri_s)] = True
            edge = self._sel_boundary(tri_s, mask)
            self._sel[name] = (mask, edge, np.nonzero(edge)[0], m.surface_area(name))
        return self._sel[name]

    def _sel_boundary(self, tri_sel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Nodes on the edge of the selection (shared with a surface triangle outside the selection)."""
        n = len(self.mesh.nodes)
        tri = self.surface.tri
        if self._keys is None:
            self._keys = tri_keys(tri, n)
        touching = mask[tri].any(axis=1)
        outside = touching & ~np.isin(self._keys, tri_keys(tri_sel, n))
        edge = np.zeros(n, bool)
        if outside.any():
            nodes = np.unique(tri[outside])
            edge[nodes[mask[nodes]]] = True
        return edge


def mesh_cache(res) -> MeshCache:
    c = getattr(res, "_mcp_cache", None)
    if c is None or c.mesh is not res.mesh:
        c = MeshCache(res.mesh)
        res._mcp_cache = c
    return c


def is_sliding(mesh, c) -> bool:
    """A prescribed ZERO displacement only perpendicular to a planar face (symmetry, sliding support) – it does not prevent the
    lateral contraction, so the edge of such a face is not a singularity (unlike a clamping).
    """
    if c.kind != "displacement" or c.selection not in mesh.surfaces:
        return False
    vals = [(k, v) for k, v in enumerate((c.ux, c.uy, c.uz)) if v is not None]
    if not vals or any(abs(v) > 0 for _, v in vals):
        return False
    tri = mesh.surfaces[c.selection][:, :3]
    if not len(tri):
        return False
    X = mesh.nodes
    n = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    n0 = n[0]
    if np.min(np.abs(n @ n0)) < 0.999:                       # not a plane
        return False
    return all(abs(n0[k]) > 0.999 for k, _ in vals)


def mean_edge(mesh) -> float:
    t = mesh.tets[: min(len(mesh.tets), 5000), :4]
    X = mesh.nodes
    e = [np.linalg.norm(X[t[:, i]] - X[t[:, j]], axis=1) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    return float(np.mean(e))


def pick(X: np.ndarray, vals: np.ndarray, top_n: int, r_cluster: float) -> list[int]:
    """Nodes with the highest values in separate places (farther from each other than r_cluster)."""
    order = np.argsort(-vals)
    cand = order[: max(2000, 50 * top_n)]
    P = X[cand]
    alive = np.ones(len(cand), bool)
    picked: list[int] = []
    while len(picked) < top_n:
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        j = idx[0]
        picked.append(int(cand[j]))
        alive &= np.linalg.norm(P - P[j], axis=1) > r_cluster
    return picked


# ===================================================================== classification of single places

class Classifier:
    """Classifies nodes of one result state: `classify(n)` → the row of `hotspots` (rank, node, xyz, value, parts, classification,
    reasons, recommendation). `vals` = the nodal field that is ranked (already absolute for signed quantities); it is also what the
    "steep gradient" test looks at. Built once per (result, state, field); the mesh topology comes from the cache of the result.
    """

    def __init__(self, study, res, st_res, vals: np.ndarray, memo: dict | None = None):
        self.study, self.res, self.st_res, self.vals = study, res, st_res, vals
        # `memo` = the geometric part of `classify` per node and the cluster balls: valid for ONE study and result (the roles of the selections, the mesh), so the caller
        # may share it between the classifiers of its own evaluation (the margins: one per case and rule) - never between studies
        self.memo: dict = {} if memo is None else memo
        mesh = res.mesh
        self.mesh = mesh
        self.cache = c = mesh_cache(res)
        self.surf = c.surface
        self.h = c.mean_edge
        diag = float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)))
        self.r_cluster = max(3.0 * self.h, 0.02 * diag)
        # role of the selections
        roles: dict[str, tuple[str, str]] = {}
        for con in study.constraints:
            roles[con.selection] = ("sliding" if is_sliding(mesh, con) else "constraint", con.kind)
        for lc in study.load_cases:
            for load in lc.loads:
                if getattr(load, "selection", None):
                    roles.setdefault(load.selection, ("load", load.kind))
        for name in mesh.surfaces:
            if name.startswith("__b"):
                roles[name] = ("bolt", "bolt")
        self.roles = roles
        self.line_loads = {name: np.unique(mesh.edges[name]) if name in mesh.edges else np.unique(mesh.points[name])
                           for name in {getattr(load, "selection", None) for lc in study.load_cases for load in lc.loads}
                           if name in mesh.edges or name in mesh.points}
        self.err_field = st_res.fields.get("stress_error")          # estimate of the error of ccx (STR %, informative)
        self.tree = c.tree
        self.re_idx, self.re_tree = c.reentrant
        self._p_all = getattr(st_res, "_principal", None)          # principal stresses: all nodes only when already cached,
        self._p_some: dict[int, np.ndarray] = {}                    # otherwise just the asked ones (eigvalsh per matrix = the same values)

    def prepare(self, nodes) -> None:
        """Principal stresses of the given nodes in one call (the same values as one by one)."""
        if self._p_all is None and len(nodes):
            from .results import principal
            idx = np.asarray(nodes, int)
            self._p_some.update(zip((int(i) for i in idx), principal(self.st_res.S[idx])))

    def _principal(self, n: int) -> np.ndarray:
        if self._p_all is not None:
            return self._p_all[n]
        if n not in self._p_some:
            self.prepare([n])
        return self._p_some[n]

    def _near_set(self, idx: np.ndarray, x) -> float:
        """Distance of a point from the nearest node of the set (inf = empty)."""
        if not len(idx):
            return math.inf
        return float(np.min(np.linalg.norm(self.mesh.nodes[idx] - x, axis=1)))

    def ball(self, n: int) -> np.ndarray:
        """The nodes within the cluster radius of node `n` (memoised: a singular place removes its surroundings from the walk of every case and rule)."""
        key = ("ball", n)
        b = self.memo.get(key)
        if b is None:
            b = self.memo[key] = np.asarray(self.tree.query_ball_point(self.mesh.nodes[n], self.r_cluster), np.int64)
        return b

    def _geometry(self, n: int) -> tuple:
        """What `classify(n)` says about a node that does not depend on the stress field: (reasons, classification, recommendation, parts, on_surface, neighbours within
        1.2 elements, tags of the GUI), memoised per node."""
        g = self.memo.get(n)
        if g is not None:
            return g
        mesh, c, surf, h, tree = self.mesh, self.cache, self.surf, self.h, self.tree
        x = mesh.nodes[n]
        reasons, sing, bc = [], False, False
        tag_s, tag_b = [], []                        # short labels of the singularities / influences of the boundary conditions (tags=True)
        for name, (role, kind) in self.roles.items():
            if name not in mesh.surfaces or len(mesh.surfaces[name]) == 0:
                continue
            mask, edge, edge_idx, area = c.selection(name)
            if role == "sliding":
                if mask[n]:
                    reasons.append(f"on symmetry plane / sliding support '{name}' (displacement normal to the face only) – no singularity")
            elif role in ("constraint", "bolt"):
                de = self._near_set(edge_idx, x)
                if de <= 1.0 * h:
                    where = "on edge" if de < 1e-9 else f"{de:.2g} mm (≤ 1 element) from the edge"
                    reasons.append(f"{where} of {('constraint' if role == 'constraint' else 'bolt connection')} face '{name}' ({kind}) – stiffness discontinuity, stress grows with mesh refinement")
                    sing = True
                    tag_s.append("bolt connection" if role == "bolt" else
                                 "fixed support edge" if kind == "fixed" else "constraint edge")
                elif mask[n]:
                    reasons.append(f"on support face '{name}' ({kind}) – the support restrains lateral contraction")
                    bc = True
                    tag_b.append("on support face")
                else:
                    near = tree.query_ball_point(x, 1.5 * h)
                    if len(near) and mask[np.asarray(near, np.int64)].any():
                        reasons.append(f"within 1.5 elements of support '{name}' – affected by the boundary condition")
                        bc = True
                        tag_b.append("at the constraint")
            elif role == "load" and mask[n]:
                if area < (3 * h) ** 2:
                    reasons.append(f"small loaded face '{name}' ({area:.3g} mm² < (3·element)²) – close to a point force")
                    sing = True
                    tag_s.append("point load")
                elif edge[n]:
                    reasons.append(f"on the edge of loaded face '{name}' – load discontinuity")
                    bc = True
                    tag_b.append("edge of loaded face")
        for name, idx in self.line_loads.items():           # force on an edge / at a point: singularity in its surroundings
            d = self._near_set(idx, x)
            if d <= 2.0 * h:
                what = "an edge" if name in mesh.edges else "a point"
                reasons.append(f"{('at the location' if d < 1e-09 else f'{d:.2g} mm (≤ 2 elements) from')} of the force into {what} '{name}' – a point / line force is singular, stress grows with mesh refinement")
                sing = True
                tag_s.append("edge force" if what == "an edge" else "point force")
        dre, jre = (self.re_tree.query(x) if self.re_tree is not None else (math.inf, -1))
        if dre <= 1.0 * h:
            bend = surf.reentrant[self.re_idx[jre]]
            where = ("on a sharp inner (concave) edge/corner" if dre < 1e-9 else
                     f"{dre:.2g} mm (≤ 1 element) from a sharp inner (concave) edge/corner")
            reasons.append(f"{where}, angle change {bend:.0f}° without a fillet – theoretically infinite stress, the value depends on the mesh")
            sing = True
            tag_s.append("sharp inner edge")
        elif surf.convex[n] > 0 and not reasons:
            reasons.append(f"outer (convex) edge, angle change {surf.convex[n]:.0f}° – not a singularity by itself")
        if surf.curved[n] > 0 and not sing:
            reasons.append("curved surface (fillet/hole) – typical location of a real stress concentration")
        parts = c.node_parts(n)
        if len(parts) > 1:
            reasons.append(f"at the interface of parts {', '.join(parts)}")
        if not surf.surface_nodes[n]:
            reasons.append("inside the material")
        near = np.asarray([j for j in tree.query_ball_point(x, 1.2 * h) if j != n], np.int64)
        if sing:
            cls = "singularity"
            rec = ("Do not evaluate from the nodal value: evaluate the stress farther from the location (≥ 2–3 elements), in a section, or the nominal stress; for a sharp corner add a fillet in CAD, for a support/force model the real attachment.")
        elif bc:
            cls = "affected by boundary condition"
            rec = "Verify that the support/load matches reality (support stiffness, contact area); evaluate away from the support."
        else:
            cls = "probably real concentration"
            rec = "Verify by mesh convergence (refine locally 2× – change < 5 %); assess against Re / fatigue."
        g = self.memo[n] = (reasons, cls, rec, list(parts), bool(surf.surface_nodes[n]), near, list(dict.fromkeys(tag_s if sing else tag_b if bc else [])))
        return g

    def classify(self, n: int, rank: int = 1, tags: bool = False) -> dict:
        """The row of the node `n` (0-based); tags=True adds "tags" = short labels for the GUI (see `classify_hotspots`)."""
        vals = self.vals
        x = self.mesh.nodes[n]
        base_reasons, cls, rec, parts, on_surface, near, tag_list = self._geometry(n)
        reasons = list(base_reasons)
        grad = float(vals[n] / max(np.mean(vals[near]), 1e-30)) if len(near) else 1.0
        if grad > 1.4:
            reasons.append(f"steep gradient: {grad:.2f}× the neighborhood average within 1.2 elements – mesh dependent")
        p = self._principal(n)
        row = {
            "rank": rank, "node": n + 1, "xyz_mm": x, "value": float(vals[n]),
            "mises_MPa": float(self.st_res.mises[n]), "s1_MPa": float(p[0]), "s3_MPa": float(p[2]),
            "parts": list(parts), "on_surface": on_surface,
            "classification": cls, "reasons": reasons, "recommendation": rec,
        }
        if "T" in self.st_res.fields:
            row["T_C"] = float(self.st_res.fields["T"][n])
        if self.err_field is not None:
            row["stress_error_pct"] = float(self.err_field[n])
        if tags:
            row["tags"] = list(tag_list)
        return row


def classify_hotspots(study, res, st_res, vals: np.ndarray, top_n: int = 5, tags: bool = False) -> tuple[list[dict], float, float]:
    """The `top_n` places of the highest `vals` (separate places) with the classification → (rows, mean element edge, cluster radius).
    tags=True adds to the places "tags" = short labels of the kind of the singularity / influence of the boundary condition for the GUI
    (e.g. "clamping edge", "sharp inner edge", "point load")."""
    cl = Classifier(study, res, st_res, vals)
    picked = pick(res.mesh.nodes, vals, top_n, cl.r_cluster)
    cl.prepare(picked)
    return [cl.classify(n, rank, tags) for rank, n in enumerate(picked, 1)], cl.h, cl.r_cluster


def scan_maxima(cl: Classifier, allowed: np.ndarray | None = None, max_places: int = 16,
                tags: bool = False, nodes: np.ndarray | None = None) -> tuple[list[dict], dict | None]:
    """Walks down the places of the highest values of `cl.vals` (restricted to the nodes where `allowed` is True) until the first one that is
    not a singularity. A singular place removes its surroundings (one cluster radius) from the walk.
    Returns (places, governing): `places` = the classified places in descending order (the singular ones and, last, the governing one),
    `governing` = the first place that is not a singularity (a real concentration or influenced by a boundary condition), None when
    `max_places` places were all singular or no node is left.
    nodes = the allowed nodes as an ASCENDING array of unique indices (instead of the mask `allowed`): the walk then costs O(nodes) instead of O(all nodes) per place,
    with the same order of the places (an equal maximum goes to the lowest node index in both)."""
    if nodes is not None:
        if allowed is not None:
            raise ValueError("scan_maxima: pass either `allowed` (a mask) or `nodes` (ascending unique indices), not both")
        nodes = np.asarray(nodes, np.int64)
        if nodes.ndim != 1 or (len(nodes) and (nodes[0] < 0 or nodes[-1] >= len(cl.mesh.nodes))) or (len(nodes) > 1 and not (nodes[1:] > nodes[:-1]).all()):
            raise ValueError("scan_maxima: `nodes` must be a one-dimensional ascending array of unique node indices inside the mesh")
        return _scan_nodes(cl, nodes, max_places, tags)
    v = np.array(cl.vals, float)
    if allowed is not None:
        v[~allowed] = -np.inf
    v[~np.isfinite(v)] = -np.inf
    places: list[dict] = []
    while len(places) < max_places:
        n = int(np.argmax(v))
        if not np.isfinite(v[n]):
            break
        row = cl.classify(n, len(places) + 1, tags)
        places.append(row)
        if row["classification"] != "singularity":
            return places, row
        v[cl.ball(n)] = -np.inf
        v[n] = -np.inf
    return places, None


def _scan_nodes(cl: Classifier, nodes: np.ndarray, max_places: int, tags: bool) -> tuple[list[dict], dict | None]:
    """`scan_maxima` over the given (ascending) nodes only."""
    v = np.array(cl.vals[nodes], float)
    v[~np.isfinite(v)] = -np.inf
    pos = np.full(len(cl.mesh.nodes), -1, np.int64)               # node -> position in `nodes`
    pos[nodes] = np.arange(len(nodes))
    places: list[dict] = []
    while len(places) < max_places:
        k = int(np.argmax(v)) if len(v) else 0
        if not len(v) or not np.isfinite(v[k]):
            break
        n = int(nodes[k])
        row = cl.classify(n, len(places) + 1, tags)
        places.append(row)
        if row["classification"] != "singularity":
            return places, row
        near = pos[cl.ball(n)]
        v[near[near >= 0]] = -np.inf
        v[k] = -np.inf
    return places, None

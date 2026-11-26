"""Parts meshed separately and connected by a *TIE constraint (chapter 5.5 of the specification: "shared mesh, or *TIE").

`MeshSettings.separate` = parts that are not fragmented in Gmsh together with the others (own faces, independent mesh) – for
parts that Gmsh does not join with the assembly (tangential contact: "PLC Error: segment and facet intersect"), or for another
fineness of the mesh. After meshing `detect` finds the faces of the surface of a separate part that lie on the surface of another part
(distance of the centroid of the face from the surface ≤ tolerance) → faces `__tie{i}_s` (slave = the separate part) and
`__tie{i}_m` (master = the other parts) in Mesh.surfaces; ccx: *TIE, POSITION TOLERANCE (element faces of both surfaces).
Nodes of the slave surface must not be in *BOUNDARY or dependent in another equation (ccx: "dependent side of a MPC and a
SPC") → such faces are left out of the tie with a warning. The NGSolve backend and MYSTRAN cannot (→ ccx / error).
"""
from __future__ import annotations

import numpy as np

PREFIX = "__tie"


def tolerance(study, mesh) -> float:
    t = study.mesh.tie_tolerance
    if t:
        return float(t)
    return max(0.05 * float(study.mesh.size), 1e-3)


def _point_tri_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance of the points p (n, 3) from the triangles (a, b, c) (n, 3) – Ericson, Real-Time Collision Detection 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) > 1e-300, den, 1e-300)
    v, w = vb / den, vc / den
    q = a + ab * v[:, None] + ac * w[:, None]                   # inside
    # regions of the vertices and edges (a simplified approach is enough: limit the barycentric coordinates and project onto the edges)
    out = (v < 0) | (w < 0) | (v + w > 1)
    if out.any():
        best = np.full(len(p), np.inf)
        for s0, s1 in ((a, b), (b, c), (c, a)):
            e = s1 - s0
            t = np.clip(np.einsum("ij,ij->i", p - s0, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0, 1)
            d = np.linalg.norm(p - (s0 + e * t[:, None]), axis=1)
            best = np.minimum(best, d)
        dist = np.linalg.norm(p - q, axis=1)
        return np.where(out, best, dist)
    return np.linalg.norm(p - q, axis=1)


def detect(study, mesh) -> list[str]:
    """Finds the contact faces of separately meshed parts → mesh.surfaces __tie{i}_s / _m. Returns warnings."""
    from .contacts.detect import contact_parts, contact_part_pairs
    cparts = contact_parts(study) if study.active_contacts() else set()
    cpairs = contact_part_pairs(study) if cparts else set()
    sep = [p for p in dict.fromkeys([*study.mesh.separate, *sorted(cparts)]) if p in mesh.part_names]
    if not sep:
        return []
    from scipy.spatial import cKDTree
    warnings = []
    tri, elem = mesh.boundary_faces()
    part = mesh.tet_part[elem]
    X = mesh.nodes
    cen = X[tri[:, :3]].mean(axis=1)
    tol = tolerance(study, mesh)
    from .solve import constraint_masks
    bc = np.zeros(len(X), bool)
    for m in constraint_masks(study, mesh).values():
        bc |= m.any(axis=1)
    k = 0
    for pn in sep:
        pi = mesh.part_names.index(pn)
        mine = np.nonzero(part == pi)[0]
        # a contact pair (F8.5) is not tied by *TIE – between the parts of the pair there is contact
        skip = [mesh.part_names.index(q) for q in mesh.part_names if frozenset((pn, q)) in cpairs]
        other = np.nonzero((part != pi) & ~np.isin(part, skip))[0]
        if not len(mine) or not len(other):
            continue

        def touching(src, dst):
            """Faces src whose centroid lies within tol of some face dst."""
            tree = cKDTree(cen[dst])
            span = np.linalg.norm(X[tri[dst, 0]] - X[tri[dst, 1]], axis=1).max() if len(dst) else 0.0
            cand = tree.query_ball_point(cen[src], r=span + tol)
            ok = np.zeros(len(src), bool)
            for i, cs in enumerate(cand):
                if not cs:
                    continue
                cs = dst[np.asarray(cs)]
                d = _point_tri_dist(np.repeat(cen[src[i]][None], len(cs), 0), X[tri[cs, 0]], X[tri[cs, 1]],
                                    X[tri[cs, 2]])
                ok[i] = d.min() <= tol
            return src[ok]
        slave = touching(mine, other)
        master = touching(other, mine)
        if not len(slave) or not len(master):
            if pn in cparts and pn not in study.mesh.separate:
                continue                          # a part only in contact – without *TIE it is fine
            warnings.append(f"Part '{pn}' is meshed separately but does not touch another part (tolerance {tol:.3g} mm) – it is not connected to anything")
            continue
        fixed = bc[tri[slave, :6]].any(axis=1)
        if fixed.any():
            warnings.append(f"*TIE of part '{pn}': {int(fixed.sum())} contact faces have nodes in a support – omitted from *TIE (ccx does not allow a *BOUNDARY node to also be dependent in *TIE)")
            slave = slave[~fixed]
        k += 1
        mesh.surfaces[f"{PREFIX}{k}_s"] = tri[slave]
        mesh.surfaces[f"{PREFIX}{k}_m"] = tri[master]
        area = float(np.linalg.norm(np.cross(X[tri[slave, 1]] - X[tri[slave, 0]],
                                             X[tri[slave, 2]] - X[tri[slave, 0]]), axis=1).sum() / 2)
        mesh.report.setdefault("ties", []).append({"part": pn, "slave": f"{PREFIX}{k}_s", "master": f"{PREFIX}{k}_m",
                                                   "area_mm2": area, "tolerance_mm": tol})
    return warnings


def pairs(mesh) -> list[tuple[str, str]]:
    return [(t["slave"], t["master"]) for t in mesh.report.get("ties", [])]


def write(f, mesh, m, sel_ids_surface) -> None:
    """*TIE for the pairs from detect (the faces already written as *SURFACE)."""
    tl = mesh.report.get("ties", [])
    for i, t in enumerate(tl, 1):
        f.write(f"*TIE, NAME=T{i}, POSITION TOLERANCE={t['tolerance_mm']:.6g}, ADJUST=NO\n"
                f"{sel_ids_surface(t['slave'])}, {sel_ids_surface(t['master'])}\n")


def tied_part_pairs(mesh) -> list[tuple[int, int]]:
    """Pairs of part indices connected by *TIE (for the check of free bodies)."""
    out = []
    for t in mesh.report.get("ties", []):
        s = np.unique(mesh.surfaces[t["slave"]])
        mst = np.unique(mesh.surfaces[t["master"]])
        owner = np.full(len(mesh.nodes), -1)
        for pi in range(len(mesh.part_names)):
            owner[np.unique(mesh.tets[mesh.tet_part == pi])] = pi
        ps = set(np.unique(owner[s]).tolist()) - {-1}
        pm = set(np.unique(owner[mst]).tolist()) - {-1}
        out += [(a, b) for a in ps for b in pm]
    return out


# ------------------------------------------------------------------ the constraint equations of a *TIE (own assembly)

def _tri6_shape(xi: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N (k, 6), dN/dξ, dN/dη of the quadratic triangle (Gmsh / ccx order: corners, midpoints 01, 12, 20) at the points (ξ, η) (k,)."""
    l0 = 1 - xi - eta
    N = np.stack([l0 * (2 * l0 - 1), xi * (2 * xi - 1), eta * (2 * eta - 1), 4 * l0 * xi, 4 * xi * eta, 4 * eta * l0], axis=1)
    z = np.zeros_like(xi)
    dxi = np.stack([-(4 * l0 - 1), 4 * xi - 1, z, 4 * (l0 - xi), 4 * eta, -4 * eta], axis=1)
    deta = np.stack([-(4 * l0 - 1), z, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (l0 - eta)], axis=1)
    return N, dxi, deta


def _closest_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ξ, η) = the barycentric coordinates (of b and c) of the point of the flat triangle (a, b, c) closest to p (all arrays (k, 3))."""
    ab, ac, ap = b - a, c - a, p - a
    d11, d12, d22 = np.einsum("ij,ij->i", ab, ab), np.einsum("ij,ij->i", ab, ac), np.einsum("ij,ij->i", ac, ac)
    r1, r2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    den = d11 * d22 - d12 * d12
    den = np.where(np.abs(den) > 1e-300, den, 1e-300)
    xi, eta = (d22 * r1 - d12 * r2) / den, (d11 * r2 - d12 * r1) / den
    inside = (xi >= 0) & (eta >= 0) & (xi + eta <= 1)
    if inside.all():
        return xi, eta
    best = np.full(len(p), np.inf)
    bx, be = xi.copy(), eta.copy()
    for (u0, v0), (u1, v1) in (((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0))):      # the three edges in (ξ, η)
        q0 = a + ab * u0 + ac * v0
        q1 = a + ab * u1 + ac * v1
        e = q1 - q0
        t = np.clip(np.einsum("ij,ij->i", p - q0, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0, 1)
        d = np.linalg.norm(p - (q0 + e * t[:, None]), axis=1)
        better = d < best
        best = np.where(better, d, best)
        bx = np.where(better, u0 + (u1 - u0) * t, bx)
        be = np.where(better, v0 + (v1 - v0) * t, be)
    return np.where(inside, xi, bx), np.where(inside, eta, be)


EDGES = ((0, 1, 3), (1, 2, 4), (2, 0, 5))        # the edges of a quadratic face: (first corner, second corner, midpoint) in the order of the nodes


def _edge_minimum(p: np.ndarray, X6: np.ndarray, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """The point of the edge `edge` of the quadratic faces X6 (k, 6, 3) closest to p (k, 3): (t in [0, 1] from the first to the second corner, squared distance).
    A coarse scan finds the valley, Gauss–Newton steps on the quadratic curve polish it (a step that does not lower the distance is halved)."""
    ia, ib, im = EDGES[edge]
    Xa, Xb, Xm = X6[:, ia], X6[:, ib], X6[:, im]

    def point(t):                                 # the point of the curve and its tangent
        n = np.stack([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)], axis=-1)
        d = np.stack([4 * t - 3, 4 * t - 1, 4 - 8 * t], axis=-1)
        return (n[..., 0:1] * Xa + n[..., 1:2] * Xb + n[..., 2:3] * Xm, d[..., 0:1] * Xa + d[..., 1:2] * Xb + d[..., 2:3] * Xm)

    def dist2(x):
        r = x - p
        return np.einsum("kc,kc->k", r, r)

    scan = np.linspace(0.0, 1.0, 17)
    xs = np.stack([point(np.full(len(p), v))[0] for v in scan], axis=1)               # (k, 17, 3)
    d = ((xs - p[:, None]) ** 2).sum(axis=2)
    t = scan[np.argmin(d, axis=1)]
    x, x1 = point(t)
    f = dist2(x)
    for _ in range(40):
        step = -np.einsum("kc,kc->k", x - p, x1) / np.maximum(np.einsum("kc,kc->k", x1, x1), 1e-300)
        tn = np.clip(t + step, 0.0, 1.0)
        xn, x1n = point(tn)
        fn = dist2(xn)
        for _half in range(10):
            bad = fn > f
            if not bad.any():
                break
            tn = np.where(bad, 0.5 * (t + tn), tn)
            xn, x1n = point(tn)
            fn = dist2(xn)
        keep = fn <= f                              # (still worse after the halvings: the point stays)
        moved = float(np.abs(np.where(keep, tn - t, 0.0)).max())
        t, x, x1, f = np.where(keep, tn, t), np.where(keep[:, None], xn, x), np.where(keep[:, None], x1n, x1), np.where(keep, fn, f)
        if moved < 1e-13:
            break
    return t, f


def project_on_faces(p: np.ndarray, X6: np.ndarray, iterations: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The point of the (quadratic, curved) triangle X6 (k, 6, 3) closest to p (k, 3): (ξ, η, distance). The closest point is the stationary point inside the
    face (Gauss–Newton from the closest point of the flat triangle of the corners) or, where that lies outside the triangle (or does not converge), the minimum along
    one of the three edges – a minimisation along the curve of the edge: a clamped step of the interior iteration is NOT the closest point of the edge (1e-2 mm
    farther at the nodes beside the border of the master surface), which CalculiX, minimising over the whole face, does not make."""
    k = len(p)
    xi, eta = _closest_on_triangle(p, X6[:, 0], X6[:, 1], X6[:, 2])
    last = np.full(k, np.inf)
    for _ in range(iterations):
        N, dxi, deta = _tri6_shape(xi, eta)
        x = np.einsum("kj,kjc->kc", N, X6)
        r = x - p
        J1, J2 = np.einsum("kj,kjc->kc", dxi, X6), np.einsum("kj,kjc->kc", deta, X6)
        g11, g12, g22 = np.einsum("kc,kc->k", J1, J1), np.einsum("kc,kc->k", J1, J2), np.einsum("kc,kc->k", J2, J2)
        b1, b2 = np.einsum("kc,kc->k", J1, r), np.einsum("kc,kc->k", J2, r)
        den = g11 * g22 - g12 * g12
        den = np.where(np.abs(den) > 1e-300, den, 1e-300)
        dx, de = -(g22 * b1 - g12 * b2) / den, -(g11 * b2 - g12 * b1) / den
        xi, eta = xi + dx, eta + de
        last = np.maximum(np.abs(dx), np.abs(de))
        if float(last.max()) < 1e-13:
            break
    eps = 1e-10
    inside = (xi >= -eps) & (eta >= -eps) & (xi + eta <= 1 + eps) & (last < 1e-8)
    N, _, _ = _tri6_shape(xi, eta)
    best = np.where(inside, np.linalg.norm(np.einsum("kj,kjc->kc", N, X6) - p, axis=1) ** 2, np.inf)
    bxi, beta = np.where(inside, xi, 0.0), np.where(inside, eta, 0.0)
    out = np.nonzero(~inside)[0]                       # only these can have the closest point on the border (the others: the stationary point inside the face)
    if len(out):
        po, Xo = p[out], X6[out]
        for edge in range(3):
            t, f = _edge_minimum(po, Xo, edge)
            e_xi, e_eta = ((t, np.zeros(len(out))), (1 - t, t), (np.zeros(len(out)), 1 - t))[edge]
            better = f < best[out] * (1 - 1e-12)
            best[out] = np.where(better, f, best[out])
            bxi[out], beta[out] = np.where(better, e_xi, bxi[out]), np.where(better, e_eta, beta[out])
    return bxi, beta, np.sqrt(np.maximum(best, 0.0))


SUBTRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])      # the four triangles of a quadratic face (corners 0 1 2, midpoints 01 12 20) – ccx `triangucont`
WALK_LIMIT = 100                                                      # ccx: more triangles on the way = no solution
# the nodes of the faces S1..S4 of a C3D10 in the order ccx takes them (`ifacet` of `triangucont` / `gentiedmpc`, 0-based in the element): three corners, then the midpoints
# 12, 23, 31 of THAT corner order. The four flat triangles are labelled by it (the first vertex of a triangle decides which edge plane the walk and the in-plane
# tolerance use), so the faces of the surface must come in this order and not in the order of the surface triangles of the mesh.
CCX_FACE_NODES = np.array([[0, 2, 1, 6, 5, 4], [0, 1, 3, 4, 8, 7], [1, 2, 3, 5, 9, 8], [0, 3, 2, 7, 9, 6]])


def ccx_faces(mesh, name: str) -> np.ndarray:
    """The faces of the surface `name` as CalculiX numbers them: the nodes of the element face in the order of `ifacet`, the faces sorted by element and face
    (ccx sorts the `*SURFACE` entries). A mesh without elements (a unit test) or a face that is not the face of an element: the faces as stored."""
    tri = np.asarray(mesh.surfaces[name])
    tets = getattr(mesh, "tets", None)
    if tets is None or tri.shape[1] != 6 or np.asarray(tets).shape[1] != 10:
        return tri
    from .mesh import MeshError
    try:
        elem, fno = mesh.element_faces(name)
    except MeshError:
        return tri
    order = np.lexsort((fno, elem))
    return np.asarray(tets)[elem[order][:, None], CCX_FACE_NODES[fno[order] - 1]]


class MasterTriangles:
    """The master surface of a *TIE as CalculiX sees it (`triangucont`, `straighteq3d`, `gentiedmpc`): every quadratic face is split into four flat triangles;
    each has the outward planes of its three edges (edge l is opposite the vertex l), the plane of the triangle and the neighbours across the edges."""

    def __init__(self, X: np.ndarray, faces: np.ndarray):
        self.faces = np.asarray(faces)
        if self.faces.shape[1] != 6:
            raise ValueError("The own assembly ties only quadratic (tri6) faces")
        self.sub = self.faces[:, SUBTRI].reshape(-1, 3)
        self.face = np.repeat(np.arange(len(self.faces)), 4)
        P = X[self.sub]
        self.centroid = P.mean(axis=1)
        xn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 1])
        xn /= np.maximum(np.linalg.norm(xn, axis=1), 1e-300)[:, None]
        self.normal = xn
        self.offset = -np.einsum("tc,tc->t", xn, P[:, 0])
        self.edge_normal = np.empty((len(P), 3, 3))
        self.edge_offset = np.empty((len(P), 3))
        for l in range(3):                                            # the edge opposite the vertex l
            a, b = P[:, (l + 1) % 3], P[:, (l + 2) % 3]
            nrm = np.cross(b - a, xn)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
            nrm = np.where((np.einsum("tc,tc->t", nrm, P[:, l] - a) > 0)[:, None], -nrm, nrm)      # pointing out of the triangle
            self.edge_normal[:, l] = nrm
            self.edge_offset[:, l] = -np.einsum("tc,tc->t", nrm, a)
        # tolerance of the position in the plane: 0.025 × the mean distance of the centroid from the edge opposite the first vertex (ccx: "a typical element size")
        self.tolloc = 0.025 * float(np.abs(np.einsum("tc,tc->t", self.edge_normal[:, 0], self.centroid) + self.edge_offset[:, 0]).mean())
        # the neighbour across every edge (the triangle with the same two nodes), −1 = the border of the surface
        nt = len(self.sub)
        keys, owner, slot = [], [], []
        for l in range(3):
            a, b = self.sub[:, (l + 1) % 3], self.sub[:, (l + 2) % 3]
            keys.append(np.minimum(a, b).astype(np.int64) * (int(self.sub.max()) + 1) + np.maximum(a, b))
            owner.append(np.arange(nt))
            slot.append(np.full(nt, l))
        keys, owner, slot = np.concatenate(keys), np.concatenate(owner), np.concatenate(slot)
        order = np.argsort(keys, kind="stable")
        ks, os_, ss = keys[order], owner[order], slot[order]
        self.neighbour = np.full((nt, 3), -1, np.int64)
        same = np.nonzero(ks[1:] == ks[:-1])[0]                       # a pair of consecutive equal keys = two triangles sharing the edge
        self.neighbour[os_[same], ss[same]] = os_[same + 1]
        self.neighbour[os_[same + 1], ss[same + 1]] = os_[same]


def tie_weights(mesh, slave: str, master: str, tol: float, tolloc: float | None = None) -> dict:
    """The tie of the nodes of the slave surface to the master surface exactly as CalculiX makes it (`gentiedmpc`, `*TIE, POSITION TOLERANCE=tol, ADJUST=NO`):
    the master surface is split into flat triangles (four per quadratic face); a slave node starts at the triangle with the nearest centroid and walks over the
    edges beyond which it lies farther than `tolloc` (the border of the surface, a circular path or more than 100 triangles = the node stays free); the
    triangle it ends in must be within `tol` perpendicular to its plane; the node is then attached to the quadratic face of that triangle at the point of the face
    closest to it (`attach_2d`) with the shape functions of the face: u_slave = Σ N_k·u_master,k.
    → {"nodes" (n,), "masters" (n, 6), "weights" (n, 6), "distance" (n,), "untied" (nodes without a master face), "tolloc"}."""
    from scipy.spatial import cKDTree
    X = mesh.nodes
    mt = MasterTriangles(X, ccx_faces(mesh, master))
    tl = mt.tolloc if tolloc is None else float(tolloc)
    nodes = np.unique(np.asarray(mesh.surfaces[slave]))
    p = X[nodes]
    n = len(nodes)
    cur = cKDTree(mt.centroid).query(p)[1].astype(np.int64)
    old = np.full(n, -1, np.int64)
    result = np.full(n, -1, np.int64)                                 # the triangle of the solution, −1 = none
    state = np.zeros(n, np.int8)                                      # 0 walking, 1 solution, 2 no solution
    visited = np.full((n, WALK_LIMIT + 2), -1, np.int64)
    steps = np.zeros(n, np.int64)
    active = np.arange(n)
    while len(active):
        c = cur[active]
        d = np.einsum("akc,ac->ak", mt.edge_normal[c], p[active]) + mt.edge_offset[c]
        viol = d > tl
        has = viol.any(axis=1)
        done = ~has
        state[active[done]] = 1
        result[active[done]] = c[done]
        mv = active[has]
        if not len(mv):
            break
        l = np.argmax(viol[has], axis=1)
        new = mt.neighbour[cur[mv], l]
        border = new < 0
        state[mv[border]] = 2
        between = (~border) & (new == old[mv])                        # the solution lies in between two triangles
        state[mv[between]] = 1
        result[mv[between]] = cur[mv[between]]
        go = (~border) & (~between)
        seen = go & (visited[mv] == new[:, None]).any(axis=1)         # a circular path
        state[mv[seen]] = 2
        go &= ~seen
        steps[mv[go]] += 1
        long_ = go & (steps[mv] > WALK_LIMIT)
        state[mv[long_]] = 2
        go &= ~long_
        g = mv[go]
        visited[g, steps[g]] = new[go]
        old[g] = cur[g]
        cur[g] = new[go]
        active = g
    ok = state == 1
    plane = np.abs(np.einsum("nc,nc->n", mt.normal[np.where(ok, result, 0)], p) + mt.offset[np.where(ok, result, 0)])
    ok &= plane <= tol                                                # farther from the plane of the triangle than the position tolerance: no MPC
    idx = np.nonzero(ok)[0]
    face = mt.face[result[idx]]
    xi, eta, dist = project_on_faces(p[idx], X[mt.faces[face]])
    N, _, _ = _tri6_shape(xi, eta)
    return {"nodes": nodes[idx], "masters": mt.faces[face], "weights": N, "distance": dist, "untied": nodes[~ok], "tolloc": tl}


def global_tolloc(mesh) -> float:
    """ccx takes the in-plane tolerance as the mean over ALL master triangles of all ties of the model (`gentiedmpc`)."""
    ties = mesh.report.get("ties", [])
    faces = np.vstack([ccx_faces(mesh, t["master"]) for t in ties])
    return MasterTriangles(mesh.nodes, faces).tolloc


def mesh_ties(mesh) -> list[dict]:
    """`tie_weights` of every *TIE of the mesh (with the in-plane tolerance of the whole model), computed once per mesh (cache `Mesh.cached`)."""
    ties = mesh.report.get("ties", [])
    if not ties:
        return []

    def compute():
        tolloc = global_tolloc(mesh)
        return [tie_weights(mesh, t["slave"], t["master"], float(t["tolerance_mm"]), tolloc) for t in ties]
    key = "tie_weights:" + "|".join(f"{t['slave']}:{t['master']}:{t['tolerance_mm']}" for t in ties)
    return mesh.cached(key, ("nodes", "tets"), compute)


def flow_to_masters(mesh, r: np.ndarray) -> np.ndarray:
    """The force that the constraint of a *TIE passes from its slave nodes to the master nodes: (N, 3), Σ_slave w·r at the master nodes. r = K·u − F (the nodal forces
    without the constraint, at the slave nodes they ARE the force of the tie; the sign is that of the equation u_s = Σ w·u_m). Zero for a mesh whose faces are not quadratic."""
    out = np.zeros_like(r)
    try:
        for w in mesh_ties(mesh):
            if len(w["nodes"]):
                np.add.at(out, w["masters"].ravel(), (w["weights"][:, :, None] * r[w["nodes"]][:, None, :]).reshape(-1, 3))
    except ValueError:                              # linear faces (MasterTriangles): no correction
        return np.zeros_like(r)
    return out

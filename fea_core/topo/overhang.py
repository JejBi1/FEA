"""Overhang (support-region) rule of the BESO for additive manufacturing without supports – APPROXIMATE, element level.

The rule works on the corner nodes of the elements. A node is SUPPORTED when it lies on the build plate (base) or when a node of a solid element
lies in its support cone: below it (against the build direction `b`), within `RADIUS_FACTOR` x the local element size and with the angle from
−b not larger than 90° − angle (angle = the smallest angle between a surface and the build plate, 45° = the usual self-supporting limit);
the supporter has to be supported itself, so a chain of supports leads from every node down to the base (breadth-first search over the graph
of the active nodes, scipy.sparse.csgraph). An element may stay solid only when ALL its corner nodes are supported; the others are removed
(cascade until the set is stable) and the lost volume is replaced by the most sensitive VOID elements whose nodes can be supported (the target
volume stays). With decision groups (symmetry / extrusion) a group is removed when any of its elements fails.

The nodes of the underside of a solid lie ON its surface, so a surface that is flatter than the angle has no supporter in the cone and is not accepted –
this is what makes the rule follow the real surface and not only the position of the element centroids (a first version with centroids bridged
holes of 1 – 2 elements and left a third of the downward area flatter than the limit). Still APPROXIMATE: unstructured mesh, discrete elements, a distance
cut-off – the surface of the result is measured afterwards on the exported triangles (`surface_report`) and reported with a tolerance.
Cf. the layer-wise AM filter of Langelaar (2016), adapted to an unstructured mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import KDTree

from .groups import element_sizes, unit_vector

RADIUS_FACTOR = 1.5        # support distance in multiples of the local element size
MARGIN_FACTOR = 0.0        # the supporter must lie this much (x element size) deeper than the cone needs (0 = the cone itself)
NEIGHBOURS = 64            # nearest nodes examined per node (the support cone is a part of this neighbourhood)
MAX_CASCADE = 400          # passes of the removal cascade
MAX_FILL_ROUNDS = 60
MAX_FILL_PASSES = 6        # refill → check → refill …
LINK_HOPS = 2              # a support is linked to the supported node through solid material within this many element hops


def point_support_graph(points: np.ndarray, h: np.ndarray, build_dir, angle_deg: float, radius_factor: float = RADIUS_FACTOR,
                        neighbours: int = NEIGHBOURS, margin_factor: float = MARGIN_FACTOR) -> sparse.csr_matrix:
    """Sparse matrix P (n x n): P[i, j] = 1 when the point j lies in the support cone of the point i (below it, within radius_factor x h[i])."""
    c = np.ascontiguousarray(points, dtype=float)
    n = len(c)
    b = unit_vector(build_dir)
    sin_a = float(np.sin(np.radians(angle_deg)))          # cos(90° − angle)
    k = int(min(neighbours, n))
    tree = KDTree(c)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    chunk = 20000
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        dist, idx = (np.asarray(a) for a in tree.query(c[lo:hi], k=k))
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        rel = c[idx] - c[lo:hi, None, :]                    # (m, k, 3): supporter minus the point
        down = np.negative(rel @ b)                               # distance BELOW the point along b
        hh = h[lo:hi, None]
        ok = (dist > 1e-12) & (dist <= radius_factor * hh) & (down >= sin_a * dist + margin_factor * hh - 1e-9 * hh)
        r, q = np.nonzero(ok)
        rows.append(r + lo)
        cols.append(idx[r, q])
    r_all = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c_all = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(r_all), dtype=np.int32), (r_all, c_all)), shape=(n, n))


def base_node_mask(points: np.ndarray, build_dir, h_mean: float) -> np.ndarray:
    """Nodes on the build plate: the lowest height along the build direction in the mesh, tolerance 0.2 x the mean element size."""
    z = np.asarray(points, dtype=float) @ unit_vector(build_dir)
    return z <= float(z.min()) + 0.2 * h_mean + 1e-6 * float(np.ptp(z))


@dataclass
class RepairStats:
    removed: int = 0               # elements removed because they had no support
    added: int = 0                 # elements added instead (the most sensitive supported voids)
    deficit_left: float = 0.0      # volume that could not be refilled [mm3]
    rounds: int = 0
    cascade: int = 0               # passes of the removal cascade


class OverhangRepair:
    """The support rule for a mesh (corner nodes) on the level of decision units (groups of elements, or single elements when `labels` is None)."""

    def __init__(self, nodes: np.ndarray, tets: np.ndarray, volumes: np.ndarray, build_dir, angle_deg: float,
                 base_nodes: np.ndarray | None, anchors: np.ndarray, labels: np.ndarray | None = None) -> None:
        """nodes, tets: the mesh (corner nodes = the first 4 columns of tets); base_nodes: (len(nodes),) bool on the build plate (None = the lowest nodes);
        anchors: (M,) bool elements that count as supported without a chain and are never removed (non-design elements)."""
        self.build_dir = unit_vector(build_dir)
        self.angle_deg = float(angle_deg)
        conn = np.asarray(tets)[:, :4]
        n_el = len(conn)
        uniq, inv = np.unique(conn, return_inverse=True)
        self.conn = inv.reshape(n_el, 4).astype(np.int64)                  # corner nodes numbered 0 … Nc-1
        pts = np.asarray(nodes, dtype=float)[uniq]
        self.n_nodes = len(uniq)
        self.inc = sparse.csr_matrix((np.ones(n_el * 4, dtype=np.int32), (np.repeat(np.arange(n_el), 4), self.conn.ravel())),
                                     shape=(n_el, self.n_nodes))
        self.inc_t = self.inc.T.tocsr()
        h_el = element_sizes(volumes)
        valence = np.maximum(np.asarray(self.inc_t.sum(axis=1)).ravel(), 1)
        h_node = (self.inc_t @ h_el) / valence
        self.node_height = pts @ self.build_dir
        self.parents = point_support_graph(pts, h_node, self.build_dir, self.angle_deg)      # parents[n, m] = 1: node m supports node n
        self.children = self.parents.T.tocsr()
        self.base = base_node_mask(pts, self.build_dir, float(np.mean(h_el))) if base_nodes is None else np.asarray(base_nodes, dtype=bool)[uniq]
        self.use_links = False                                              # the optimiser loop works with the cone only; `prune` adds the links
        self.anchor_el = np.asarray(anchors, dtype=bool)
        self.anchor_nodes = (self.inc_t @ self.anchor_el.astype(np.int32)) > 0
        # decision units
        if labels is None:
            self.labels = np.arange(n_el)
            self.n = n_el
        else:
            self.labels = np.asarray(labels, dtype=np.int64)
            self.n = int(self.labels.max()) + 1 if n_el else 0
        self._low_node = self.conn[np.arange(n_el), np.argmin(self.node_height[self.conn], axis=1)]
        self._low_height = self.node_height[self._low_node]

    # --------------------------------------------------------------- nodes and elements
    def _expand(self, solid_u: np.ndarray) -> np.ndarray:
        return np.asarray(solid_u, dtype=bool)[self.labels]

    def active_nodes(self, solid_el: np.ndarray) -> np.ndarray:
        return (self.inc_t @ solid_el.astype(np.int32)) > 0

    def solid_links(self, solid_el: np.ndarray) -> sparse.csr_matrix:
        """Pairs of nodes that are connected through SOLID material within two element hops (they share a solid element, or both share one
        with a third node) – a support has to be transmitted through material, not across a thin void (a slit one element thick)."""
        d = sparse.diags(solid_el.astype(np.int32), format="csr", dtype=np.int32)
        s1 = (self.inc_t @ d @ self.inc).tocsr()
        out = s1
        for _ in range(LINK_HOPS - 1):
            out = (out @ s1).tocsr()
        return out

    def supported_nodes(self, solid_el: np.ndarray, strict: bool = False, links: bool | None = None) -> np.ndarray:
        """Nodes of the solid elements connected to the build plate (and to the anchors unless strict) through a chain of supporters in the cone;
        links = every supporter also has to be linked to the supported node through solid material (`solid_links` – a support does not jump
        across a thin slit). The iterations of the optimiser use the cone only (the rule with the links is so brittle that one removed element
        takes a quarter of the part with it); the result is checked with the links at the end (`prune`)."""
        n = self.n_nodes
        active = self.active_nodes(solid_el)
        use_links = self.use_links if links is None else links
        if use_links:
            edges = self.children.multiply(self.solid_links(solid_el))
        else:
            d = sparse.diags(active.astype(np.int32), format="csr", dtype=np.int32)
            edges = d @ self.children @ d                                   # only between nodes of solid elements
        sub = edges.tocoo()                                                # edges supporter → supported
        roots = np.nonzero(active & (self.base if strict else (self.base | self.anchor_nodes)))[0]
        rows = np.concatenate([sub.row, np.full(len(roots), n)])
        cols = np.concatenate([sub.col, roots])
        g = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n + 1, n + 1))
        order = csgraph.breadth_first_order(g, n, directed=True, return_predecessors=False)
        out = np.zeros(n + 1, dtype=bool)
        out[order] = True
        return out[:n]

    def _elements_ok(self, nodes_ok: np.ndarray) -> np.ndarray:
        """Elements whose corner nodes are all in `nodes_ok`."""
        return (self.inc @ (~nodes_ok).astype(np.int32)) == 0

    def _units_failing(self, element_bad: np.ndarray) -> np.ndarray:
        return np.bincount(self.labels, weights=element_bad.astype(float), minlength=self.n) > 0

    def supported_elements(self, solid_u: np.ndarray, strict: bool = False, links: bool | None = None) -> np.ndarray:
        """(M,) bool: solid elements that are supported (all corner nodes in the chain to the plate)."""
        solid_el = self._expand(solid_u)
        s = self.supported_nodes(solid_el, strict, links)
        return solid_el & self._elements_ok(s)

    # --------------------------------------------------------------- support-aware ranking
    def support_priority(self, sensitivity: np.ndarray, solid_u: np.ndarray) -> np.ndarray:
        """Sensitivity (per unit) for the threshold in which the supporters of valuable elements are not cheaper than what they carry.

        The lowest node of every solid element that needs support designates its supporter = the active node of its support cone whose elements have
        the highest sensitivity; the best element at that node then carries at least the sensitivity of the element it supports, from the top downwards
        (a chain of single elements down to the plate). Without this the threshold removes the lightly loaded supporters under a highly loaded
        roof, the repair then deletes the ROOF (it is unsupported) and the structure loses its load path.
        """
        sens = np.asarray(sensitivity, dtype=float)
        solid_el = self._expand(solid_u)
        eff = sens.copy()
        el_val = np.where(solid_el, sens[self.labels], -np.inf)
        # per node: the best solid element (value + index)
        pair_el = np.repeat(np.arange(len(self.conn)), 4)
        pair_nd = self.conn.ravel()
        keep = solid_el[pair_el]
        pe, pn = pair_el[keep], pair_nd[keep]
        if len(pe) == 0:
            return eff
        order = np.lexsort((-el_val[pe], pn))
        pe, pn = pe[order], pn[order]
        first = np.concatenate([[True], pn[1:] != pn[:-1]])
        best_el = np.full(self.n_nodes, -1, dtype=np.int64)
        best_val = np.full(self.n_nodes, -np.inf)
        best_el[pn[first]] = pe[first]
        best_val[pn[first]] = el_val[pe[first]]
        # designated supporter node of every active, non-base node: the supporter with the best value
        ptr = self.parents.indptr
        rows = np.repeat(np.arange(self.n_nodes), np.diff(ptr))
        cols = self.parents.indices
        ok = (best_el[rows] >= 0) & (best_el[cols] >= 0) & ~self.base[rows]
        rows, cols = rows[ok], cols[ok]
        designated = np.full(self.n_nodes, -1, dtype=np.int64)
        if len(rows):
            o2 = np.lexsort((-best_val[cols], rows))
            rows, cols = rows[o2], cols[o2]
            f2 = np.concatenate([[True], rows[1:] != rows[:-1]])
            designated[rows[f2]] = cols[f2]
        # elements that need support: solid, not anchors, lowest node not on the plate
        need = np.nonzero(solid_el & ~self.anchor_el & ~self.base[self._low_node] & (designated[self._low_node] >= 0))[0]
        if len(need) == 0:
            return eff
        parent_el = best_el[designated[self._low_node[need]]]
        top_down = np.argsort(np.negative(self._low_height[need]), kind="stable")
        e = eff.tolist()
        lab = self.labels
        for el, p in zip(need[top_down].tolist(), parent_el[top_down].tolist()):
            u, v = int(lab[el]), int(lab[p])
            if e[u] > e[v]:
                e[v] = e[u]
        return np.asarray(e, dtype=float)

    # --------------------------------------------------------------- repair
    def _cascade(self, solid: np.ndarray, fixed: np.ndarray, cnt: np.ndarray, stats: RepairStats) -> np.ndarray:
        """Removes the units with an unsupported element until the set is stable (removing a unit can deprive others of their support)."""
        solid = solid.copy()
        for _ in range(MAX_CASCADE):
            solid_el = self._expand(solid)
            bad = solid_el & ~self._elements_ok(self.supported_nodes(solid_el)) & ~self.anchor_el
            fail = self._units_failing(bad) & ~fixed & solid
            if not fail.any():
                break
            stats.removed += int(cnt[fail].sum())
            stats.cascade += 1
            solid[fail] = False
        return solid

    def prune(self, solid: np.ndarray, fixed: np.ndarray, counts: np.ndarray | None = None) -> tuple[np.ndarray, RepairStats]:
        """Final check of a design with the rule INCLUDING the links through solid material: removes (cascade) the units whose support would have to
        cross a thin slit. No refill – the volume ends lower; the caller reports it."""
        cnt = np.ones(self.n, dtype=np.int64) if counts is None else np.asarray(counts)
        stats = RepairStats()
        old, self.use_links = self.use_links, True
        try:
            out = self._cascade(np.asarray(solid, dtype=bool) | np.asarray(fixed, dtype=bool), np.asarray(fixed, dtype=bool), cnt, stats)
        finally:
            self.use_links = old
        return out, stats

    def repair(self, solid: np.ndarray, sensitivity: np.ndarray, volumes: np.ndarray, target_volume: float,
               fixed: np.ndarray, counts: np.ndarray | None = None) -> tuple[np.ndarray, RepairStats]:
        """Removes the unsupported solids (cascade) and refills the volume with the most sensitive supported voids.

        solid: (n,) bool per unit after the threshold; sensitivity, volumes per unit; fixed: units that stay solid (non-design); counts: elements per unit
        (for the statistics only). The refill is checked by the same rule afterwards (a unit that does not pass is removed and not tried again).
        """
        fixed = np.asarray(fixed, dtype=bool)
        solid = np.asarray(solid, dtype=bool) | fixed
        cnt = np.ones(self.n, dtype=np.int64) if counts is None else np.asarray(counts)
        stats = RepairStats()
        solid = self._cascade(solid, fixed, cnt, stats)
        kept = solid.copy()                                               # what the threshold wanted and the rule accepts
        blocked = np.zeros(self.n, dtype=bool)
        vmin = float(volumes.min()) if len(volumes) else 0.0
        for _ in range(MAX_FILL_PASSES):
            deficit = float(target_volume - volumes[solid].sum())
            if deficit <= 0.5 * vmin:
                break
            added = np.zeros(self.n, dtype=bool)
            for _ in range(MAX_FILL_ROUNDS):
                deficit = float(target_volume - volumes[solid].sum())
                if deficit <= 0.5 * vmin:
                    break
                solid_el = self._expand(solid)
                s = self.supported_nodes(solid_el)
                can = s | self.base | ((self.parents @ s.astype(np.int32)) > 0)    # nodes that a supported node can carry
                cand = np.nonzero(~self._units_failing(~self._elements_ok(can)) & ~solid & ~blocked)[0]
                if len(cand) == 0:
                    break
                order = cand[np.argsort(-sensitivity[cand], kind="stable")]
                v = volumes[order]
                take = (np.cumsum(v) - v) + 0.5 * v <= deficit
                if not take.any():
                    break
                solid[order[take]] = True
                added[order[take]] = True
                stats.rounds += 1
            if not added.any():
                break
            checked = self._cascade(solid, fixed, cnt, stats)              # the rule with the solid links decides about the added units
            blocked |= added & ~checked
            solid = checked
        stats.added = int(cnt[solid & ~kept].sum())
        stats.deficit_left = max(float(target_volume - volumes[solid].sum()), 0.0)
        return solid, stats


# ------------------------------------------------------------------ measurement on the exported surface

def surface_report(verts: np.ndarray, faces: np.ndarray, build_dir, angle_deg: float, base_height: float | None = None,
                   base_tol: float = 1e-6, tolerance_deg: float = 15.0) -> dict:
    """Independent check of the overhang on the triangles of the exported STL (outward normals).

    A facet is DOWNWARD when its normal has a component against the build direction; its inclination from the build plate is the angle
    between its normal and −b (0° = a flat ceiling, 90° = a vertical wall). Facets on the base (all vertices at the height of the plate) are
    excluded. Reported: area of the downward facets, the share (by area) flatter than `angle_deg` and flatter than `angle_deg − tolerance_deg`
    (the mesh tolerance), the worst facet and the area-weighted 5th percentile of the inclination, and the connected patches of the facets
    flatter than `angle_deg` (their total area, number and the largest one – a flat ceiling of several elements is what makes a part unprintable,
    single flat facets of the size of an element are the roughness of the mesh).
    """
    out = {"build_dir": [float(x) for x in unit_vector(build_dir)], "angle_deg": float(angle_deg), "tolerance_deg": float(tolerance_deg),
           "facets": int(len(faces)), "downward_facets": 0, "downward_area_mm2": 0.0, "violating_area_fraction": 0.0,
           "violating_area_fraction_beyond_tolerance": 0.0, "worst_angle_deg": None, "p05_angle_deg": None,
           "flat_area_mm2": 0.0, "flat_patches": 0, "largest_flat_patch_mm2": 0.0}
    if len(faces) == 0:
        return out
    b = unit_vector(build_dir)
    tri = np.asarray(verts, dtype=float)[np.asarray(faces)]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    good = area > 1e-12
    unit_n = np.zeros_like(nrm)
    unit_n[good] = nrm[good] / (2.0 * area[good, None])
    z = tri @ b
    z0 = float(np.min(np.asarray(verts, dtype=float) @ b)) if base_height is None else float(base_height)
    on_base = z.max(axis=1) <= z0 + base_tol
    down = good & (unit_n @ b < -1e-9) & ~on_base
    out["downward_facets"] = int(down.sum())
    if not down.any():
        return out
    incl = np.degrees(np.arccos(np.clip(np.negative(unit_n[down] @ b), -1.0, 1.0)))
    a = area[down]
    total = float(a.sum())
    out["downward_area_mm2"] = total
    out["violating_area_fraction"] = float(a[incl < angle_deg].sum() / total)
    out["violating_area_fraction_beyond_tolerance"] = float(a[incl < angle_deg - tolerance_deg].sum() / total)
    out["worst_angle_deg"] = float(incl.min())
    srt = np.argsort(incl)
    cum = np.cumsum(a[srt]) / total
    out["p05_angle_deg"] = float(incl[srt][min(int(np.searchsorted(cum, 0.05)), len(srt) - 1)])
    flat = np.nonzero(down)[0][incl < angle_deg]
    if len(flat):
        f = np.asarray(faces)[flat]
        inc = sparse.csr_matrix((np.ones(f.size), (np.repeat(np.arange(len(flat)), 3), f.ravel())), shape=(len(flat), len(verts)))
        n_patch, lab = csgraph.connected_components(inc @ inc.T, directed=False)
        patch_area = np.bincount(lab, weights=area[flat], minlength=n_patch)
        out.update(flat_area_mm2=float(area[flat].sum()), flat_patches=int(n_patch), largest_flat_patch_mm2=float(patch_area.max()))
    return out

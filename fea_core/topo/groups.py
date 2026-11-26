"""Decision groups of the BESO: elements that must get the same density (mirror images, columns of an extrusion).

Manufacturing constraints on the element level of an UNSTRUCTURED tet10 mesh (pre/post-processing around ccx, not a solver):

- Symmetry: the element centroid is folded into the positive half-space of every mirror plane; an element on the negative side
  ("slave") joins the nearest element of the positive side ("master") to its mirrored centroid when it lies within
  `SYM_TOL` x the element size – otherwise it stays alone and is reported (the mesh is not symmetric there). The planes are mutually
  perpendicular, so folding is a plain reflection of the coordinates in the normals.
- Extrusion: the centroids are projected onto the plane perpendicular to the direction and binned into columns of the size of an element
  (the grid is aligned with the mirror planes so that a mirror maps a column onto a column); slivers are merged into the nearest column (2D kd-tree).

A group has ONE decision: its sensitivity is the volume-weighted mean of its elements, it is solid or void as a whole, a non-design element
makes the whole group solid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

SYM_TOL = 0.5              # tolerance of the mirror matching in multiples of the element size
MIN_BIN_IN_H = 0.25        # the smallest bin of an extrusion in multiples of the mean element size (smaller values are raised to it)
SLIVER_FRACTION = 0.25     # a column with less than this share of the median column volume is merged into the nearest column
_SQRT2 = float(np.sqrt(2.0))


def element_sizes(volumes: np.ndarray) -> np.ndarray:
    """Edge length of the regular tetrahedron of the same volume (local element size h)."""
    return np.cbrt(6.0 * _SQRT2 * np.maximum(np.asarray(volumes, dtype=float), 1e-300))


def unit_vector(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("zero vector")
    return a / n


@dataclass(eq=False)       # identity semantics: `in` / `==` on numpy fields would raise "truth value of an array is ambiguous"
class Plane:
    point: np.ndarray        # (3,) a point of the plane [mm]
    normal: np.ndarray       # (3,) unit normal

    def distance(self, x: np.ndarray) -> np.ndarray:
        """Signed distance of the points x (…, 3) from the plane [mm]."""
        return (np.asarray(x, dtype=float) - self.point) @ self.normal

    def mirror(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x - 2.0 * self.distance(x)[..., None] * self.normal

    def to_dict(self) -> dict:
        return {"point": [float(c) for c in self.point], "normal": [float(c) for c in self.normal]}


def planes_from_spec(items: list[dict], centre: np.ndarray) -> list[Plane]:
    """Planes of `TopologyOptSpec.symmetry`; a missing point = the centre of the design domain."""
    return [Plane(np.asarray(centre if it.get("point") is None else it["point"], dtype=float), unit_vector(it["normal"]))
            for it in items]


class DecisionGroups:
    """Partition of the elements into groups with one decision each (labels 0 … G-1)."""

    def __init__(self, labels: np.ndarray, volumes: np.ndarray, non_design: np.ndarray) -> None:
        uniq, inv = np.unique(np.asarray(labels), return_inverse=True)
        self.n = len(uniq)
        self.labels = inv.astype(np.int64).ravel()
        self.element_volumes = np.asarray(volumes, dtype=float)
        self.volumes = np.bincount(self.labels, weights=self.element_volumes, minlength=self.n)
        self.counts = np.bincount(self.labels, minlength=self.n)
        self.non_design = np.bincount(self.labels, weights=np.asarray(non_design, dtype=float), minlength=self.n) > 0
        self._order = np.argsort(self.labels, kind="stable")
        self._starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)

    def mean(self, values: np.ndarray) -> np.ndarray:
        """Volume-weighted mean of the element values per group (G,)."""
        return np.bincount(self.labels, weights=np.asarray(values, dtype=float) * self.element_volumes,
                           minlength=self.n) / np.maximum(self.volumes, 1e-300)

    def expand(self, group_values: np.ndarray) -> np.ndarray:
        """Group values → element values (M,)."""
        return np.asarray(group_values)[self.labels]

    def spread(self, values: np.ndarray) -> float:
        """Largest difference max − min of the element values inside one group (0 = identical decisions)."""
        v = np.asarray(values, dtype=float)[self._order]
        return float(np.max(np.maximum.reduceat(v, self._starts) - np.minimum.reduceat(v, self._starts))) if len(v) else 0.0


# ------------------------------------------------------------------ symmetry

def symmetry_labels(centroids: np.ndarray, volumes: np.ndarray, planes: list[Plane],
                    tol: float = SYM_TOL) -> tuple[np.ndarray, dict]:
    """Group labels (the index of the master element) of the mirror images + statistics of the matching."""
    c = np.asarray(centroids, dtype=float)
    n_el = len(c)
    labels = np.arange(n_el)
    if n_el == 0 or not planes:
        return labels, {"masters": n_el, "slaves": 0, "unmatched": 0, "matched_fraction": 1.0, "max_distance_in_h": 0.0, "tolerance_in_h": tol}
    h = element_sizes(volumes)
    d = np.stack([p.distance(c) for p in planes], axis=1)                 # (M, K)
    eps = 1e-9 * max(float(np.ptp(c, axis=0).max()), 1.0)
    master = np.all(d >= -eps, axis=1)
    fold = c - 2.0 * np.minimum(d, 0.0) @ np.stack([p.normal for p in planes])   # reflect every negative coordinate
    slaves = np.nonzero(~master)[0]
    masters = np.nonzero(master)[0]
    unmatched = np.zeros(0, dtype=np.int64)
    max_dist = 0.0
    if len(slaves) and len(masters):
        dist, j = (np.asarray(a) for a in KDTree(c[masters]).query(fold[slaves]))
        m = masters[j]
        ok = dist <= tol * 0.5 * (h[slaves] + h[m])
        labels[slaves[ok]] = m[ok]
        unmatched = slaves[~ok]
        if ok.any():
            max_dist = float(np.max(dist[ok] / (0.5 * (h[slaves[ok]] + h[m[ok]]))))
    elif len(slaves):
        unmatched = slaves
    info = {"masters": int(len(masters)), "slaves": int(len(slaves)), "unmatched": int(len(unmatched)),
            "matched_fraction": float(1.0 - len(unmatched) / n_el), "max_distance_in_h": max_dist, "tolerance_in_h": tol}
    return labels, info


def mirror_mismatch(solid: np.ndarray, centroids: np.ndarray, volumes: np.ndarray, planes: list[Plane],
                    tol: float = SYM_TOL) -> dict:
    """Independent check of the symmetry of the result: share of the volume whose mirror image (the nearest element to the mirrored
    centroid) has the opposite state. Zero for a symmetric mesh; a small number > 0 at the boundary of a non-symmetric mesh.
    """
    c = np.asarray(centroids, dtype=float)
    v = np.asarray(volumes, dtype=float)
    s = np.asarray(solid, dtype=bool)
    if len(c) == 0 or not planes:
        return {"mismatch_volume_fraction": 0.0, "unmatched_fraction": 0.0}
    h = element_sizes(v)
    tree = KDTree(c)
    worst, unmatched = 0.0, 0.0
    for p in planes:
        dist, j = (np.asarray(a) for a in tree.query(p.mirror(c)))
        ok = dist <= tol * 0.5 * (h + h[j])
        denom = float(v[ok].sum())
        bad = float(v[ok & (s != s[j])].sum())
        worst = max(worst, bad / denom if denom > 0 else 0.0)
        unmatched = max(unmatched, float(v[~ok].sum() / max(v.sum(), 1e-300)))
    return {"mismatch_volume_fraction": worst, "unmatched_fraction": unmatched}


# ------------------------------------------------------------------ extrusion

def _grid_axes(d: np.ndarray, planes: list[Plane]) -> tuple[np.ndarray, np.ndarray, list[Plane], list[Plane]]:
    """Axes (u, v) of the grid of columns in the plane perpendicular to d; the mirror planes whose normal lies in that plane define them."""
    inplane = [p for p in planes if abs(float(p.normal @ d)) < 1e-4]
    along = [p for p in planes if abs(abs(float(p.normal @ d)) - 1.0) < 1e-4]
    other = [p for p in planes if p not in inplane and p not in along]
    if other:
        raise ValueError("a symmetry plane is neither parallel nor perpendicular to the extrusion direction – the mirror image of a column "
                         "would not be a column; use planes whose normal is parallel or perpendicular to the direction")
    if inplane:
        u = unit_vector(inplane[0].normal - (inplane[0].normal @ d) * d)
    else:
        e = np.zeros(3)
        e[int(np.argmin(np.abs(d)))] = 1.0
        u = unit_vector(e - (e @ d) * d)
    v = unit_vector(np.cross(d, u))
    if len(inplane) > 1:
        v = unit_vector(inplane[1].normal - (inplane[1].normal @ d) * d)
    return u, v, inplane, along


def extrusion_labels(centroids: np.ndarray, volumes: np.ndarray, direction, planes: list[Plane],
                     bin_size: float | None = None) -> tuple[np.ndarray, dict]:
    """Column labels of an extrusion along `direction` (+ the mirror images of the columns for planes perpendicular to the columns' cross-section)."""
    c = np.asarray(centroids, dtype=float)
    n_el = len(c)
    d = unit_vector(direction)
    h = element_sizes(volumes)
    h_mean = float(np.mean(h)) if n_el else 1.0
    bs = float(bin_size) if bin_size else h_mean
    clamped = False
    if bs < MIN_BIN_IN_H * h_mean:             # a bin far below the element size = (almost) one column per element, and the column key could overflow
        bs, clamped = MIN_BIN_IN_H * h_mean, True
    u, v, inplane, along = _grid_axes(d, planes)
    a = c @ u - (inplane[0].point @ u if inplane else 0.0)
    b = c @ v - (inplane[1].point @ v if len(inplane) > 1 else 0.0)
    ia = np.floor(a / bs).astype(np.int64)
    ib = np.floor(b / bs).astype(np.int64)
    if inplane:                                   # mirror: bin k ↔ bin −k−1 (the grid starts at the plane)
        ia = np.where(ia < 0, -ia - 1, ia)
    if len(inplane) > 1:
        ib = np.where(ib < 0, -ib - 1, ib)
    ia -= ia.min() if n_el else 0
    ib -= ib.min() if n_el else 0
    key = ia * (int(ib.max()) + 1 if n_el else 1) + ib
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    ncol = len(uniq)
    vol = np.bincount(inv, weights=volumes, minlength=ncol)
    ca = np.bincount(inv, weights=ia, minlength=ncol) / np.maximum(np.bincount(inv, minlength=ncol), 1)
    cb = np.bincount(inv, weights=ib, minlength=ncol) / np.maximum(np.bincount(inv, minlength=ncol), 1)
    merged = 0
    if ncol > 3:
        tiny = vol < SLIVER_FRACTION * float(np.median(vol))
        if tiny.any() and (~tiny).any():
            centres = np.stack([ca, cb], axis=1)
            dist, j = (np.asarray(a) for a in KDTree(centres[~tiny]).query(centres[tiny]))
            near = dist <= 2.0                       # only slivers next to a column (not separate islands)
            target = np.arange(ncol)
            big = np.nonzero(~tiny)[0]
            tiny_ids = np.nonzero(tiny)[0]
            target[tiny_ids[near]] = big[j[near]]
            merged = int(near.sum())
            inv = target[inv]
            _, inv = np.unique(inv, return_inverse=True)
            inv = inv.ravel()
            ncol = int(inv.max()) + 1 if n_el else 0
    info = {"direction": [float(x) for x in d], "columns": int(ncol), "bin_size_mm": bs, "merged_slivers": merged,
            "mirror_planes_in_cross_section": len(inplane), "mirror_planes_along_columns": len(along)}
    if clamped:
        info["bin_size_raised_to_min"] = True
    return inv.astype(np.int64), info

"""Cross-sections of a closed triangle surface by planes perpendicular to a direction – the independent check that an extruded result is a prism.

The section of the exported STL at several heights along the extrusion direction is rasterised (winding number of the oriented section
segments) and compared: area at every height and the overlap (intersection over union) of every pair of sections. For a prism the
sections are identical; on an unstructured mesh they differ by the roughness of the mesh (an element is larger than the cell of the raster).
"""
from __future__ import annotations

import numpy as np

from .groups import unit_vector


def frame(direction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal frame (u, v, d) with u × v = d."""
    d = unit_vector(direction)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(d)))] = 1.0
    u = unit_vector(e - (e @ d) * d)
    v = np.cross(d, u)
    return u, v, d


def section_segments(verts: np.ndarray, faces: np.ndarray, direction, height: float) -> np.ndarray:
    """Oriented segments (m, 2, 2) of the section by the plane `height` (measured along `direction`) in the coordinates (u, v);
    the material lies on the left of every segment (counter-clockwise outer boundaries)."""
    u, v, d = frame(direction)
    xyz = np.asarray(verts, dtype=float) @ np.stack([u, v, d], axis=1)
    tri = xyz[np.asarray(faces)]                                   # (k, 3, 3) in (u, v, w)
    h = tri[..., 2] - height
    pos = h >= 0.0
    cross = (pos.sum(axis=1) > 0) & (pos.sum(axis=1) < 3)
    tri, h, pos = tri[cross], h[cross], pos[cross]
    if len(tri) == 0:
        return np.zeros((0, 2, 2))
    pts = np.zeros((len(tri), 2, 2))
    cnt = np.zeros(len(tri), dtype=int)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        c = pos[:, i] != pos[:, j]
        t = h[c, i] / (h[c, i] - h[c, j])
        p = tri[c, i, :2] + t[:, None] * (tri[c, j, :2] - tri[c, i, :2])
        idx = np.nonzero(c)[0]
        pts[idx, cnt[idx]] = p
        cnt[idx] += 1
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, :2]
    direction_ccw = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1)       # outward normal turned by +90°: the material is on the left
    flip = np.einsum("ij,ij->i", pts[:, 1] - pts[:, 0], direction_ccw) < 0
    pts[flip] = pts[flip][:, ::-1]
    return pts


def rasterise(segments: np.ndarray, lo: np.ndarray, shape: tuple[int, int], cell: float) -> np.ndarray:
    """Boolean mask (nu, nv) of the cell centres inside the region bounded by the oriented segments (winding number ≠ 0)."""
    nu, nv = shape
    gu = lo[0] + (np.arange(nu) + 0.5) * cell
    gv = lo[1] + (np.arange(nv) + 0.5) * cell
    qu, qv = np.meshgrid(gu, gv, indexing="ij")
    q = np.stack([qu.ravel(), qv.ravel()], axis=1)
    wn = np.zeros(len(q), dtype=np.int32)
    if len(segments) == 0:
        return np.zeros(shape, dtype=bool)
    p1, p2 = segments[:, 0], segments[:, 1]
    chunk = max(1, 4_000_000 // max(len(segments), 1))
    for lo_i in range(0, len(q), chunk):
        qq = q[lo_i:lo_i + chunk]
        qv_ = qq[:, 1][:, None]
        qu_ = qq[:, 0][:, None]
        left = (p2[:, 0] - p1[:, 0]) * (qv_ - p1[:, 1]) - (qu_ - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
        up = (p1[:, 1] <= qv_) & (p2[:, 1] > qv_)
        down = (p1[:, 1] > qv_) & (p2[:, 1] <= qv_)
        wn[lo_i:lo_i + chunk] = ((up & (left > 0)).sum(axis=1) - (down & (left < 0)).sum(axis=1)).astype(np.int32)
    return (wn != 0).reshape(shape)


def prism_report(verts: np.ndarray, faces: np.ndarray, direction, fractions=(0.15, 0.5, 0.85), cell: float | None = None) -> dict:
    """Sections of a closed surface at the given fractions of its extent along `direction`: areas and the overlap of the sections.

    Returns {heights, areas_mm2, area_spread (max / min − 1), min_iou (smallest overlap of a pair), cell_mm}.
    """
    u, v, d = frame(direction)
    xyz = np.asarray(verts, dtype=float) @ np.stack([u, v, d], axis=1)
    out: dict = {"direction": [float(x) for x in d], "heights": [], "areas_mm2": [], "area_spread": None, "min_iou": None, "cell_mm": None}
    if len(faces) == 0:
        return out
    w0, w1 = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    cell = float(cell) if cell else float(max(hi - lo)) / 200.0
    shape = (int(np.ceil((hi[0] - lo[0]) / cell)) + 1, int(np.ceil((hi[1] - lo[1]) / cell)) + 1)
    masks = []
    for f in fractions:
        z = w0 + f * (w1 - w0)
        masks.append(rasterise(section_segments(verts, faces, d, z), lo, shape, cell))
        out["heights"].append(float(z))
        out["areas_mm2"].append(float(masks[-1].sum() * cell * cell))
    out["cell_mm"] = cell
    a = np.asarray(out["areas_mm2"])
    out["area_spread"] = float(a.max() / a.min() - 1.0) if a.min() > 0 else None
    ious = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = float(np.logical_or(masks[i], masks[j]).sum())
            ious.append(float(np.logical_and(masks[i], masks[j]).sum()) / union if union > 0 else 1.0)
    out["min_iou"] = float(min(ious)) if ious else 1.0
    return out

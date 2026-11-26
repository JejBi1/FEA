"""Where a surrogate may answer: the DATA SUPPORT of one fitted model (F15-S4).

The conformal band holds for queries that look like the calibration points. A query that lies in a region without analyses – beyond
the range of the usable (converged) points, outside their convex hull, in a gap or hole of the design, between the few levels of a
gridded parameter, off the lines of a one-at-a-time design or next to failed analyses – is not covered by it, however narrow the band
looks. `Support` decides that, from the USABLE training and calibration points only (the requested range of the study, `bounds`, is
irrelevant: a failed region shrinks the box):

  1. every parameter inside [min, max] of the usable points;
  2. two or more parameters: inside the convex hull of the usable points (Qhull, an exact subset construction for many points; linear
     programming only when Qhull fails – never an unchecked answer);
  3. a parameter with fewer than `MIN_LEVELS` distinct values (a coarse grid: 2, 3, 4, 5 levels) is queried only AT one of its levels:
     what the response does between three levels is not known, whatever the other parameters do;
  4. a design that varies one parameter at a time (a "star": every point differs from the base point in at most one parameter) answers
     only ON those lines – the interaction of the parameters was never analysed;
  5. otherwise the point is not deeper than `HOLE_RIM` × the median spacing inside an EMPTY BALL of radius larger than the gap scale of
     the design – `HOLE_K` × the median nearest-neighbour spacing of the usable points, or `HOLE_MAX` × the largest one when the design is
     irregular (a hole, a gap between clusters, the region between the coarse rows of an anisotropic grid) – and its nearest usable point
     is at most `SUPPORT_K` median spacings (`REACH_MAX` largest ones) away;
     (a point within `HOLE_RIM` spacings of the rim of such a hole is answered but not marked reliable: `at_hole_rim`);
  6. the point is not closer to a failed analysis (or to an analysis without a value of this KPI) than to a successful one.

`SUPPORT_K`, `HOLE_K`, `HOLE_MAX`, `REACH_MAX`, `HOLE_RIM` are numerical settings, set with simulated designs (F15-S4 re-check; the tests
are in `tests/test_surrogate_design.py`): over 200 Latin hypercubes (1–5 parameters, 19 to 112 points) none of the 1 500 queries per design
inside the hull is refused; random designs and the points of one category of a split design, whose natural gaps are larger, lose 0–7 % of
the hull on average (the core of their largest gap); an empty disc of radius 0.2 in a 75-point Latin hypercube is found in 29 of 30 designs
and refused in 96 % of its region, one of radius 0.25 in all of them and entirely. A hole smaller than that cannot be told
from the gaps of a random design: no data-based rule can see it, and a feature of the response narrower than the spacing of the design
is invisible to every surrogate.
"""
from __future__ import annotations

import numpy as np

SUPPORT_K = 2.5                 # numerical setting: the largest accepted distance to the nearest usable point in units of the median spacing
HOLE_K = 2.0                    # numerical setting: an empty ball with a radius above this many median spacings is a hole of the design ...
HOLE_MAX = 1.25                 # ... and above this multiple of the LARGEST nearest-neighbour distance of the points: an irregular design (random
                                # points, the points of one category of a split design) has larger natural gaps than a regular one
REACH_MAX = 1.6                 # the same for the distance to the nearest point: at most SUPPORT_K median spacings or this multiple of the largest one
HOLE_RIM = 0.25                 # numerical setting: a point nearer to the rim of a hole than this (in median spacings) is as close to data as
                                # an ordinary query is – only the deeper interior of a hole is refused
MIN_LEVELS = 6                  # a parameter with fewer distinct values is answered at its levels only (numerical setting)
LEVEL_MERGE = 1e-6              # values closer than this (unit box) are one level
LEVEL_TOL = 1e-4                # a query this close to a level (share of the analysed range) is AT the level
LINE_SHARE = 0.9                # share of the points that differ from the base point in at most one parameter: a one-at-a-time design
BOX_TOL = 1e-9                  # relative tolerance of the range check
HULL_TOL = 1e-7                 # tolerance of the convex-hull test (unit coordinates)
HULL_MAX_POINTS = 600           # up to this many points the hull is built from all of them, above: from a subset that is grown until it holds the rest
HULL_ROUNDS = 40                # rounds of that growth (it ends after a few); Qhull or the growth failing → linear programs
HOLE_SAMPLES = 8000             # random centres searched for empty balls (deterministic: seed 0)
HOLE_KEEP = 3000                # the largest empty balls kept
CHUNK = 2_000_000               # elements of a temporary distance / facet matrix (the memory of one block, 16 MB)
RANK_TOL = 1e-6                 # relative singular value below which the design points do not span a direction
EDGE_MARGIN = 0.05              # share of the analysed range at each end where the coverage of the band can be lower (answers there are NOT reliable)


def spanned_rank(U: np.ndarray) -> int:
    """Number of directions of the unit-scaled design points (rank of the centred matrix): d = the points span the whole space."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if len(U) < 2:
        return 0
    s = np.linalg.svd(U - U.mean(axis=0), compute_uv=False)
    return int(np.sum(s > RANK_TOL * max(float(s[0]), 1e-300))) if s[0] > 0 else 0


def _clusters(values: np.ndarray, merge: float) -> tuple[np.ndarray, np.ndarray]:
    """(start index of every cluster in the sorted values, sort order): values within `merge` of their neighbour are one cluster."""
    order = np.argsort(values, kind="stable")
    s = values[order]
    return np.r_[0, np.flatnonzero(np.diff(s) > merge) + 1], order


def _blocks(m: int, width: int) -> list[slice]:
    """Slices of the rows of an (m × width) temporary matrix that keep one block at CHUNK elements."""
    step = max(1, CHUNK // max(1, width))
    return [slice(i, min(m, i + step)) for i in range(0, m, step)]


class Support:
    """Support of one (category, KPI) model. U = unit-scaled usable points (n, d), X = the same in the units of the parameters,
    failed = unit-scaled analyses of the same category that failed or have no value of this KPI (m, d)."""

    def __init__(self, U: np.ndarray, X: np.ndarray, failed: np.ndarray | None = None):
        self.U = np.atleast_2d(np.asarray(U, dtype=float))
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        n, d = self.U.shape
        self.d = d
        self.lo_u, self.hi_u = self.U.min(axis=0), self.U.max(axis=0)
        self.lo_x, self.hi_x = self.X.min(axis=0), self.X.max(axis=0)
        from .model import nn_distances
        nn = nn_distances(self.U, self.U, skip_self=True) if n > 1 else np.ones(1)
        self.h = max(float(np.median(nn)), 1e-9) if n > 1 else 1.0
        h_max = max(float(np.max(nn)), self.h) if n > 1 else 1.0
        self.hole_radius = max(HOLE_K * self.h, HOLE_MAX * h_max)       # the radius of an empty ball above which it is a hole
        self.reach = max(SUPPORT_K * self.h, REACH_MAX * h_max)         # the farthest accepted distance to the nearest usable point
        self.failed = np.zeros((0, d)) if failed is None or len(failed) == 0 else np.atleast_2d(np.asarray(failed, dtype=float))
        self._eq: np.ndarray | bool | None = None            # Qhull facet equations, False = Qhull failed (linear programs then)
        self._balls: tuple[np.ndarray, np.ndarray] | None = None   # centres and radii of the large empty balls
        self._levels()

    # ------------------------------------------------------------------ the structure of the design (decided from the points only)
    def _levels(self) -> None:
        """Distinct values (levels) of every parameter, the coarse ones, and whether the design is one-at-a-time."""
        self.levels_u: list[np.ndarray] = []
        self.levels_x: list[np.ndarray] = []
        self.base = np.zeros(self.d)                          # the most frequent level of every parameter
        for j in range(self.d):
            idx, order = _clusters(self.U[:, j], LEVEL_MERGE)
            counts = np.diff(np.r_[idx, len(order)])
            su, sx = self.U[order, j], self.X[order, j]
            self.levels_u.append(np.add.reduceat(su, idx) / counts)
            self.levels_x.append(np.add.reduceat(sx, idx) / counts)
            self.base[j] = self.levels_u[j][int(np.argmax(counts))]
        self.coarse = [len(lv) < MIN_LEVELS for lv in self.levels_u]
        off = np.sum(np.abs(self.U - self.base) > LEVEL_TOL, axis=1)
        self.lines = self.d >= 2 and float(np.mean(off <= 1)) >= LINE_SHARE
        self.kind = "lines" if self.lines else ("levels" if any(self.coarse) else "space-filling")

    def design_notes(self, keys: list[str]) -> list[str]:
        """What the design means for the answers (sentences for the warnings of the fit)."""
        out: list[str] = []
        if self.lines:
            out.append("The design points vary one parameter at a time (a star / one-at-a-time design): the model answers only ON those lines "
                       "– the interaction of the parameters was never analysed; use mode lhs to fit a surrogate of the whole region")
        for j, k in enumerate(keys):
            if self.coarse[j]:
                lv = ", ".join(f"{v:g}" for v in self.levels_x[j][:MIN_LEVELS])
                out.append(f"{k} has only {len(self.levels_x[j])} analysed levels ({lv}): the model answers at those levels only, "
                           f"not between them (at least {MIN_LEVELS} distinct values are needed for that)")
        if not self.lines and self._ball_arrays()[1].size:
            r = float(self._ball_arrays()[1].max())
            out.append(f"The analysed points leave a region without analyses (an empty ball of radius {r / self.h:.1f} × the usual spacing of the "
                       f"design, limit {self.hole_radius / self.h:.1f} ×): the model refuses queries in its inner part")
        return out

    # ------------------------------------------------------------------ tests (vectorised over the rows of Q, unit coordinates)
    def in_box(self, Q: np.ndarray) -> np.ndarray:
        tol = BOX_TOL * np.maximum(self.hi_u - self.lo_u, 1e-12)
        return np.all((Q >= self.lo_u - tol) & (Q <= self.hi_u + tol), axis=1)

    @staticmethod
    def _violation(P: np.ndarray, eq: np.ndarray) -> np.ndarray:
        """Largest value of the facet equations over the facets for every row of P (> HULL_TOL = outside the hull); block by block."""
        out = np.empty(len(P))
        for s in _blocks(len(P), len(eq)):
            out[s] = np.max(P[s] @ eq[:, :-1].T + eq[:, -1], axis=1)
        return out

    def _grown_hull(self) -> np.ndarray:
        """Facets of the EXACT hull of many points: the hull of a subset (the extremes in many directions), then the points outside it
        are added until none is outside – the hull of the final subset is the hull of all of them."""
        from scipy.spatial import ConvexHull
        U, n, d = self.U, len(self.U), self.d
        rng = np.random.default_rng(0)
        dirs = np.vstack([np.eye(d), -np.eye(d), rng.normal(size=(16 * d, d))])
        pick = set(np.argmax(U @ dirs.T, axis=0).tolist()) | set(rng.choice(n, size=min(n, 5 * d + 5), replace=False).tolist())
        for _ in range(HULL_ROUNDS):
            sub = np.array(sorted(pick))
            eq = np.asarray(ConvexHull(U[sub]).equations)
            viol = self._violation(U, eq)
            out = np.flatnonzero(viol > HULL_TOL)
            if out.size == 0:
                return eq
            pick |= set(out[np.argsort(-viol[out])[:100]].tolist())
        raise RuntimeError("the hull did not settle")

    def _facets(self):
        if self._eq is None:
            self._eq = False
            try:
                from scipy.spatial import ConvexHull
                self._eq = np.asarray(ConvexHull(self.U).equations) if len(self.U) <= HULL_MAX_POINTS else self._grown_hull()
            except Exception:  # noqa: BLE001 – QhullError (degenerate set), a hull that does not settle: linear programs decide
                self._eq = False
        return self._eq

    def _in_hull_lp(self, q: np.ndarray) -> bool:
        """Is q a convex combination of the points? (a linear program: weights ≥ 0, summing to 1, reproducing q)"""
        from scipy.optimize import linprog
        A = np.vstack([self.U.T, np.ones(len(self.U))])
        res = linprog(np.zeros(len(self.U)), A_eq=A, b_eq=np.append(q, 1.0), bounds=(0, None), method="highs")
        return bool(res.status == 0)

    def in_hull(self, Q: np.ndarray) -> np.ndarray:
        """Inside the convex hull of the usable points (for one parameter: inside [min, max])."""
        Q = np.atleast_2d(Q)
        ok = self.in_box(Q)
        if self.d < 2 or not ok.any():
            return ok
        eq = self._facets()
        if eq is not False:
            out = ok.copy()
            idx = np.flatnonzero(ok)
            out[idx] = self._violation(Q[idx], np.asarray(eq)) <= HULL_TOL
            return out
        out = ok.copy()
        for i in np.flatnonzero(ok):
            out[i] = self._in_hull_lp(Q[i])
        return out

    def at_levels(self, Q: np.ndarray) -> np.ndarray:
        """Every coarse parameter (fewer than MIN_LEVELS levels) is AT one of its levels."""
        Q = np.atleast_2d(Q)
        ok = np.ones(len(Q), dtype=bool)
        for j in range(self.d):
            if self.coarse[j]:
                ok &= np.min(np.abs(Q[:, j:j + 1] - self.levels_u[j][None, :]), axis=1) <= LEVEL_TOL
        return ok

    def on_lines(self, Q: np.ndarray) -> np.ndarray:
        """A one-at-a-time design: at most one parameter differs from the base point."""
        Q = np.atleast_2d(Q)
        return np.sum(np.abs(Q - self.base) > LEVEL_TOL, axis=1) <= 1

    def _ball_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Centres and radii of the empty balls (radius = distance of the centre to the nearest usable point) that are larger than
        SUPPORT_K × the median spacing; the centres are drawn inside the hull. Deterministic."""
        if self._balls is None:
            self._balls = (np.zeros((0, self.d)), np.zeros(0))
            if not self.lines and len(self.U) > 1:
                from .model import nn_distances
                rng = np.random.default_rng(0)
                C = self.lo_u + rng.random((HOLE_SAMPLES, self.d)) * np.maximum(self.hi_u - self.lo_u, 1e-12)
                C = C[self.in_hull(C)]
                if len(C):
                    R = nn_distances(C, self.U)
                    big = np.flatnonzero(R > self.hole_radius)
                    big = big[np.argsort(-R[big])[:HOLE_KEEP]]
                    self._balls = (C[big], R[big])
        return self._balls

    def in_hole(self, Q: np.ndarray) -> np.ndarray:
        """Inside an empty ball larger than SUPPORT_K × the median spacing (a hole of the design)."""
        Q = np.atleast_2d(Q)
        C, R = self._ball_arrays()
        out = np.zeros(len(Q), dtype=bool)
        if not len(C):
            return out
        core = np.maximum(R - HOLE_RIM * self.h, 0.0)
        for s in _blocks(len(Q), len(C) * max(1, self.d)):
            D2 = np.sum((Q[s, None, :] - C[None, :, :]) ** 2, axis=2)
            out[s] = np.any(D2 < core ** 2, axis=1)
        return out

    def at_hole_rim(self, Q: np.ndarray) -> np.ndarray:
        """Inside a hole of the design, but within HOLE_RIM spacings of its rim: answered (an ordinary query is as close to data), but the
        data lie on one side only and the response may change fast towards the hole, so the answer is not `reliable`."""
        Q = np.atleast_2d(Q)
        C, R = self._ball_arrays()
        out = np.zeros(len(Q), dtype=bool)
        if self.lines or not len(C):
            return out
        for s in _blocks(len(Q), len(C) * max(1, self.d)):
            D2 = np.sum((Q[s, None, :] - C[None, :, :]) ** 2, axis=2)
            out[s] = np.any(D2 < R ** 2, axis=1)
        return out

    def distance_ratio(self, Q: np.ndarray) -> np.ndarray:
        """Distance to the nearest usable point in units of the median nearest-neighbour spacing."""
        from .model import nn_distances
        return nn_distances(np.atleast_2d(Q), self.U) / self.h

    def near_failed(self, Q: np.ndarray) -> np.ndarray:
        """Closer to a failed analysis than to any usable one?"""
        Q = np.atleast_2d(Q)
        if len(self.failed) == 0:
            return np.zeros(len(Q), dtype=bool)
        from .model import nn_distances
        return nn_distances(Q, self.failed) < nn_distances(Q, self.U)

    def supported(self, Q: np.ndarray) -> np.ndarray:
        """All tests at once for the rows of Q."""
        Q = np.atleast_2d(Q)
        ok = self.in_hull(Q) & self.at_levels(Q)
        ok &= self.on_lines(Q) if self.lines else ~self.in_hole(Q)
        if ok.any():
            ok[ok] = (self.distance_ratio(Q[ok]) <= self.reach / self.h) & ~self.near_failed(Q[ok])
        return ok

    def reasons(self, u: np.ndarray, x: np.ndarray, keys: list[str]) -> list[str]:
        """Why one query is outside the support (empty = inside). u = unit coordinates, x = in the units of the parameters."""
        out: list[str] = []
        q = np.atleast_2d(u)
        tol = BOX_TOL * np.maximum(self.hi_u - self.lo_u, 1e-12)
        for j, k in enumerate(keys):
            if q[0, j] < self.lo_u[j] - tol[j] or q[0, j] > self.hi_u[j] + tol[j]:
                out.append(f"{k} = {x[j]:g} is outside the range of the analysed points [{self.lo_x[j]:g}; {self.hi_x[j]:g}]")
        if not out and self.d >= 2 and not bool(self.in_hull(q)[0]):
            out.append("the point lies outside the convex hull of the analysed points (a region without analyses)")
        for j, k in enumerate(keys):
            if self.coarse[j] and np.min(np.abs(q[0, j] - self.levels_u[j])) > LEVEL_TOL:
                lv = ", ".join(f"{v:g}" for v in self.levels_x[j])
                out.append(f"{k} = {x[j]:g} lies between the analysed levels of this parameter ({lv}): only {len(self.levels_x[j])} levels were "
                           "analysed, what the response does between them is unknown")
        if self.lines and not bool(self.on_lines(q)[0]):
            out.append("the point lies off the lines of the one-at-a-time design (more than one parameter differs from the base point): "
                       "the interaction of the parameters was never analysed")
        if not self.lines and bool(self.in_hole(q)[0]):
            out.append(f"the point lies in an empty region of the design (larger than {self.hole_radius / self.h:.1f} × the usual spacing): a gap in the design")
        r = float(self.distance_ratio(q)[0])
        if r > self.reach / self.h and not any("gap in the design" in o for o in out):
            out.append(f"the nearest analysed point is {r:.1f} × the usual spacing of the design away (limit {self.reach / self.h:.1f} ×): a gap in the design")
        if bool(self.near_failed(q)[0]):
            out.append("the point is closer to a failed analysis than to a successful one")
        return out

    def edge_notes(self, u: np.ndarray, x: np.ndarray, keys: list[str]) -> list[str]:
        """Parameters whose value lies in the outer EDGE_MARGIN of the analysed range: the conformal coverage is an AVERAGE over the region
        and is lower near its faces (measured: 83-89 % against 95 % overall), so the answer there carries a warning and is not `reliable`.
        A coarse parameter (answered at its levels only) has no edge zone: its lowest and highest level are analysed values."""
        q = np.atleast_2d(u)[0]
        span = np.maximum(self.hi_u - self.lo_u, 1e-12)
        rim = ([f"the point lies at the rim of an empty region of the design (a hole of more than {self.hole_radius / self.h:.1f} × the usual spacing): "
                "the analyses lie on one side of it only, the response may change fast towards the hole, so the answer is not marked reliable"]
               if bool(self.at_hole_rim(q[None, :])[0]) else [])
        return rim + [f"{k} = {x[j]:g} is within {EDGE_MARGIN:.0%} of the edge of the analysed range: the coverage of the band can be lower here "
                "than stated (it is an average over the whole region), so the answer is not marked reliable"
                for j, k in enumerate(keys)
                if not self.coarse[j] and (q[j] < self.lo_u[j] + EDGE_MARGIN * span[j] or q[j] > self.hi_u[j] - EDGE_MARGIN * span[j])]


def describe_failed(failed: list[dict], keys: list[str], n_total: int) -> str:
    """One sentence for the warning at fit time: how many analyses failed and in which region of the parameters (range of each
    parameter over the failed points)."""
    parts = []
    for k in keys:
        vals = [f[k] for f in failed if k in f]
        nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums and len(nums) == len(vals):
            lo, hi = min(nums), max(nums)
            parts.append(f"{k} = {lo:g}" if lo == hi else f"{k} in [{lo:g}; {hi:g}]")
        elif vals:
            parts.append(f"{k} in {{{', '.join(sorted({str(v) for v in vals})[:6])}}}")
    return (f"{len(failed)} of {n_total} analyses failed and are not used; they lie at " + ", ".join(parts)
            + " – the surrogate answers nowhere near them and nowhere beyond the analysed points")

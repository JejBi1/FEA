"""Surrogate model of the KPI of a parametric study with conformal error bars (F15-S4): fit, predict, validate, save, load.

The training data are the design points of a parametric study (`sweep.ParametricStudyResult.points`, any mode; the Latin-hypercube
mode is the one made for it). Continuous parameters (numbers, up to 5) span the input space, categorical ones (material names, print
patterns) split the points into groups – one model per category combination. One model per (group, KPI): the KPI are fitted in the
log space when strictly positive in the TRAINING points (stress, displacement, mass, frequency), inputs are scaled to the unit box,
a seeded share of the points (default 20 %, at least ⌈level / (1 − level)⌉ = 9) is held out and used ONLY to calibrate the band
(calibrate.py). The model answers only where analyses exist (support.py): the box is that of the USABLE points, not the requested range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import LIMITATIONS
from .calibrate import (LEVEL, Q_MAX, Calibration, conformal_quantile, coverage, judge, r_squared, required_calibration, scores_of)
from .design import MAX_CONTINUOUS, MIN_VALIDATION, VALIDATION_FRACTION, split_indices, validation_size
from .model import BackendError, make_backend, resolve_backend, restore_backend
from .support import Support, describe_failed, spanned_rank

SURROGATE_KPIS = ("max_stress_clean", "max_stress", "max_disp", "min_safety", "mass", "freq1", "print_safety")
MIN_TRAIN_PER_DIM = 4           # hard minimum of training points: max(6, 4·d + 2)
WARN_TRAIN_PER_DIM = 10         # fewer training points per dimension: a warning
MAX_ROWS = 2000                 # design points of one surrogate (fit and file)
MAX_PARAMS = 8
MAX_MEMBERS = 256               # (category, KPI) models of one file
MAX_CATEGORIES = 200
MAX_FAILED = 2000               # failed analyses kept with the model
FORMAT = 2                      # format 1 (no fingerprint, no failed analyses) is still read
CV_FOLDS = 10
CV_LOO_MAX = 100                # up to this many training points the cross-validation leaves ONE point out at a time (as the band is calibrated)
NO_FINGERPRINT = ("No study fingerprint was recorded with these design points: a later change of the study cannot be detected – "
                  "fit the surrogate from a parametric study run in this version")
GAP_WARN = 0.02                 # the analysed points reach less than this share short of the sampled range: no warning


class SurrogateError(ValueError):
    """The surrogate cannot be fitted or queried (too few points, outside the analysed region, unknown parameter …)."""


def min_train(d: int) -> int:
    return max(6, MIN_TRAIN_PER_DIM * d + 2)


def holdout_minimum(level: float = LEVEL, min_holdout: int | None = None) -> int:
    """Smallest hold-out set of a fit: ⌈level / (1 − level)⌉ (9 at 90 %) unless the caller asks for fewer – the band then guarantees
    only n / (n + 1) and says so."""
    return max(1, int(min_holdout)) if min_holdout is not None else max(MIN_VALIDATION, required_calibration(level))


def min_points(d: int, fraction: float = VALIDATION_FRACTION, level: float = LEVEL, min_holdout: int | None = None) -> int:
    """Smallest group size that can be fitted: the training minimum plus the hold-out set."""
    mv = holdout_minimum(level, min_holdout)
    n = min_train(d) + mv
    while n - validation_size(n, fraction, mv) < min_train(d):
        n += 1
    return n


@dataclass
class ParamSpec:
    """One input of the surrogate: continuous (lo, hi = the range of the analysed points) or categorical (the categories seen in the data)."""
    key: str
    kind: str = "continuous"                     # continuous | categorical
    lo: float = 0.0
    hi: float = 1.0
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind, "lo": self.lo, "hi": self.hi, "categories": list(self.categories)}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        return cls(key=str(d["key"]), kind=str(d.get("kind", "continuous")), lo=float(d.get("lo", 0.0)), hi=float(d.get("hi", 1.0)),
                   categories=[str(c) for c in d.get("categories", [])])


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _split_points(points) -> tuple[list[dict], list[dict]]:
    """(rows, failed): rows {"params", "kpi"[, "adaptive"]} of the converged points, the parameters of the failed analyses."""
    rows: list[dict] = []
    failed: list[dict] = []
    for p in points:
        if isinstance(p, dict):
            ok, params, src = p.get("converged", True), p.get("params") or {}, p.get("kpi", p)
            get = src.get
            adaptive = bool(p.get("adaptive", False))
        else:
            ok, params = getattr(p, "converged", True), getattr(p, "params", {}) or {}
            get = lambda k, p=p: getattr(p, k, None)      # noqa: E731
            adaptive = bool(getattr(p, "adaptive", False))
        if not ok:
            failed.append(dict(params))
            continue
        row = {"params": dict(params), "kpi": {k: _finite(get(k)) for k in SURROGATE_KPIS}}
        if adaptive:
            row["adaptive"] = True
        rows.append(row)
    return rows, failed


def training_set(points, keys: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Rows {"params": {...}, "kpi": {...}} from design points (sweep.DesignPoint or dicts with params / KPI), failed points left out.
    Returns (rows, keys); keys = the parameter keys in order of first appearance unless given."""
    rows, _ = _split_points(points)
    if keys is None:
        keys = []
        for r in rows:
            keys += [k for k in r["params"] if k not in keys]
    return rows, list(keys)


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _specs(rows: list[dict], keys: list[str], warnings: list[str]) -> list[ParamSpec]:
    """The inputs of the surrogate. The range of a continuous parameter is the range of the ANALYSED (converged) points."""
    specs: list[ParamSpec] = []
    for k in keys:
        vals = [r["params"].get(k) for r in rows]
        if any(v is None for v in vals):
            raise SurrogateError(f"Parameter '{k}' is missing at some design points")
        if all(_is_number(v) for v in vals):
            if not all(math.isfinite(float(v)) for v in vals):
                raise SurrogateError(f"Parameter '{k}' has a value that is not a finite number")
            lo, hi = float(min(vals)), float(max(vals))
            if not hi > lo:
                warnings.append(f"Parameter '{k}' has the same value ({lo:g}) at every design point – left out of the model")
                continue
            specs.append(ParamSpec(k, "continuous", lo, hi))
        elif all(isinstance(v, str) for v in vals):
            specs.append(ParamSpec(k, "categorical", categories=sorted({str(v) for v in vals})))
        else:
            raise SurrogateError(f"Parameter '{k}' mixes numbers and names")
    return specs


class _Member:
    """One fitted model: a (category group, KPI) pair, its data split, its calibration and its data support."""

    def __init__(self, group: tuple, kpi: str):
        self.group, self.kpi = group, kpi
        self.transform = "id"                    # log | id (decided by the training values only)
        self.ymean, self.ystd = 0.0, 1.0
        self.train_ids: list[int] = []
        self.cal_ids: list[int] = []
        self.backend: Any = None                 # model.RBFBackend | model.GPBackend
        self.cal = Calibration()
        self.support: Support | None = None      # where this model may answer (support.py)

    # transformed standardised output ↔ KPI
    def to_std(self, y):
        y = np.asarray(y, dtype=float)
        if self.transform == "log":
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.log(np.where(y > 0, y, np.nan))        # a value the log cannot take is NaN → an infinite score, never a silent one
        else:
            t = y
        return (t - self.ymean) / self.ystd

    def from_std(self, s):
        t = np.asarray(s, dtype=float) * self.ystd + self.ymean
        return np.exp(np.minimum(t, 700.0)) if self.transform == "log" else t


class SurrogateModel:
    """Fitted surrogate of the KPI of a parametric study. Build it with `fit_surrogate`; `predict(params)` answers in milliseconds."""

    def __init__(self, params: list[ParamSpec], kpis: list[str], rows: list[dict], level: float, seed: int, backend: str,
                 validation_fraction: float):
        self.params, self.kpis, self.rows = params, kpis, rows
        self.level, self.seed, self.backend, self.validation_fraction = level, seed, backend, validation_fraction
        self.members: dict[tuple, dict[str, _Member]] = {}
        self._supports: dict[tuple, Support] = {}      # the data support of members that rest on the same points (see _fit_member)
        self.warnings: list[str] = []
        self.skipped_groups: dict[tuple, str] = {}
        self.failed: list[dict] = []                  # parameters of the analyses that failed (not used, but their region is known)
        self.fingerprint = ""                         # stale.study_fingerprint of the study the points were analysed on
        self.fingerprint_keys: list[str] = []         # the swept keys that fingerprint neutralised (a constant one is no model parameter)
        self.sweep: dict = {}                         # the specification of the parametric study (ParametricStudySpec.to_dict)
        self.stale = ""                               # not empty = the study changed since (the sentence says what to do)
        self.min_holdout: int | None = None           # only when the caller asked for a smaller hold-out set than ⌈level / (1 − level)⌉

    # ------------------------------------------------------------------ helpers
    @property
    def continuous(self) -> list[ParamSpec]:
        return [p for p in self.params if p.kind == "continuous"]

    @property
    def categorical(self) -> list[ParamSpec]:
        return [p for p in self.params if p.kind == "categorical"]

    def _group_of(self, params: dict) -> tuple:
        return tuple(str(params[p.key]) for p in self.categorical)

    def _x(self, params: dict) -> np.ndarray:
        return np.array([float(params[p.key]) for p in self.continuous])

    def _unit(self, X: np.ndarray) -> np.ndarray:
        lo = np.array([p.lo for p in self.continuous])
        hi = np.array([p.hi for p in self.continuous])
        return (np.atleast_2d(X) - lo) / (hi - lo)

    def _rows_xy(self, ids: list[int], kpi: str) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([self._x(self.rows[i]["params"]) for i in ids]).reshape(len(ids), len(self.continuous))
        y = np.array([self.rows[i]["kpi"][kpi] for i in ids], dtype=float)
        return X, y

    def groups(self) -> list[tuple]:
        return list(self.members)

    @property
    def reliable(self) -> bool:
        return bool(self.members) and not self.stale and all(m.cal.reliable for g in self.members.values() for m in g.values())

    @property
    def guaranteed_level(self) -> float:
        """The coverage that the bands of ALL models guarantee: the smallest `Calibration.guaranteed` (= `level` when every band is exact)."""
        return min((m.cal.guaranteed for g in self.members.values() for m in g.values()), default=self.level)

    def check_study(self, study, geometry_path=None) -> str:
        """Compares the study with the one the surrogate was fitted for; sets and returns `stale` ("" = the same study)."""
        from .stale import stale_reason
        self.stale = stale_reason(self, study, geometry_path)
        return self.stale

    def _member(self, group: tuple, kpi: str) -> _Member:
        g = self.members.get(group)
        if g is None:
            why = self.skipped_groups.get(group)
            raise SurrogateError(f"No model for the category {', '.join(group) or '–'}" + (f": {why}" if why else "")
                                 + f" (trained: {', '.join('/'.join(t) or '–' for t in self.members)})")
        if kpi not in g:
            raise SurrogateError(f"KPI '{kpi}' is not in the surrogate (available: {', '.join(g)})")
        return g[kpi]

    # ------------------------------------------------------------------ prediction
    def predict_arrays(self, kpi: str, group: tuple, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(value, lower, upper) in the units of the KPI for the rows of X (continuous parameters, NO check of the analysed region – see
        `supported`)."""
        m = self._member(group, kpi)
        mu, sd = m.backend.predict(self._unit(X))
        half = m.cal.q * sd
        return m.from_std(mu), m.from_std(mu - half), m.from_std(mu + half)

    def supported(self, kpi: str, group: tuple, X: np.ndarray) -> np.ndarray:
        """Boolean per row of X (continuous parameters): inside the analysed region of this model (support.py)."""
        m = self._member(group, kpi)
        if m.support is None:
            return np.ones(len(np.atleast_2d(X)), dtype=bool)
        return m.support.supported(self._unit(X))

    def in_hull(self, group: tuple, x: np.ndarray) -> bool | None:
        """Is x inside the convex hull of the analysed points of the group? None = not asked (one parameter: the range decides)."""
        if len(self.continuous) < 2:
            return None
        mems = self.members.get(group)
        if not mems:
            return None
        sup = next(iter(mems.values())).support
        return None if sup is None else bool(sup.in_hull(self._unit(x))[0])

    def predict(self, params: dict, kpis: list[str] | None = None, extrapolate: bool = False) -> dict:
        """Value ± band for every KPI at `params` {key: value}. Refuses (SurrogateError) a query outside the analysed region – beyond the
        range of the converged points, outside their convex hull, in a gap or hole of the design, between the few levels of a gridded
        parameter, off the lines of a one-at-a-time design, next to failed analyses – and an unknown category unless extrapolate=True
        (a warning and NO guarantee then: `reliable` is False for every result). In the outer 5 % of a continuous parameter's analysed
        range the answer carries a warning and `reliable` is False too: the conformal coverage is an average over the region and lower
        at its faces."""
        if not self.members:
            raise SurrogateError("The surrogate has no fitted model")
        unknown = [k for k in params if k not in {p.key for p in self.params}]
        if unknown:
            raise SurrogateError(f"Unknown parameter(s): {', '.join(unknown)} (the surrogate has: {', '.join(p.key for p in self.params)})")
        missing = [p.key for p in self.params if p.key not in params]
        if missing:
            raise SurrogateError(f"Missing parameter(s): {', '.join(missing)}")
        for p in self.continuous:
            try:
                v = float(params[p.key])
            except (TypeError, ValueError):
                raise SurrogateError(f"Parameter '{p.key}' must be a number") from None
            if not math.isfinite(v):
                raise SurrogateError(f"Parameter '{p.key}' must be finite")
        outside: list[str] = []
        for p in self.categorical:
            if str(params[p.key]) not in p.categories:
                outside.append(f"{p.key} = {params[p.key]} was not in the study (categories: {', '.join(p.categories)})")
        if outside and not extrapolate:
            raise SurrogateError("Extrapolation refused: " + "; ".join(outside) + ". The error bars hold only where analyses exist.")
        if outside:
            raise SurrogateError("The category was not in the study – there is no model for it")
        group = self._group_of(params)
        if group not in self.members:
            raise SurrogateError(f"No model for the category {', '.join(group)}: " + self.skipped_groups.get(group, "no data"))
        x = self._x(params)
        unit = self._unit(x[None, :])
        keys = [p.key for p in self.continuous]
        use = list(kpis or [k for k in self.kpis if k in self.members[group]])
        why: dict[str, list[str]] = {}
        for k in use:
            m = self._member(group, k)
            why[k] = m.support.reasons(unit[0], x, keys) if m.support is not None else []
        every = list(dict.fromkeys(r for rs in why.values() for r in rs))
        if every and not extrapolate:
            raise SurrogateError("Extrapolation refused: " + "; ".join(every) + ". The error bars hold only where analyses exist "
                                 "(extrapolate=true predicts anyway, WITHOUT a guarantee).")
        warns: list[str] = []
        if every:
            warns.append("EXTRAPOLATION – " + "; ".join(every) + ": the band is NOT valid here")
        if self.stale:
            warns.append("STALE – " + self.stale)
        edge: dict[str, list[str]] = {k: [] for k in use}        # per KPI: the answer in the outer EDGE_MARGIN of the range is not `reliable`
        if not every:
            for k in use:
                sup = self._member(group, k).support
                if sup is not None:
                    edge[k] = sup.edge_notes(unit[0], x, keys)
            warns += list(dict.fromkeys(n for ns in edge.values() for n in ns))
        out: dict[str, dict] = {}
        for k in use:
            v, lo, hi = (a[0] for a in self.predict_arrays(k, group, x[None, :]))
            m = self._member(group, k)
            rel = float((hi - lo) / 2.0 / abs(v)) if v else float("inf")
            ok = m.cal.reliable and not why[k] and not self.stale and not edge[k]
            out[k] = {"value": float(v), "lower": float(lo), "upper": float(hi), "half_width_rel": rel if math.isfinite(rel) else None,
                      "level": m.cal.guaranteed, "reliable": ok}
            if not m.cal.reliable:
                warns.append(f"{k}: model not reliable – " + "; ".join(m.cal.reasons or ["see the validation"]))
            elif rel > 0.25:
                warns.append(f"{k}: the band is wide (±{rel:.0%})")
            if m.cal.guaranteed < self.level:
                warns.append(f"{k}: the band holds for {m.cal.guaranteed:.0%} of the new points, not {self.level:.0%} (too few calibration points)")
        hull = self.in_hull(group, x)
        return {"params": {p.key: params[p.key] for p in self.params}, "kpis": out,
                "level": min((r["level"] for r in out.values()), default=self.level), "requested_level": self.level,
                "extrapolated": bool(every), "inside_support": not every, "inside_hull": hull, "near_edge": any(edge.values()), "stale": bool(self.stale),
                "warnings": list(dict.fromkeys(warns)), "limitations": LIMITATIONS}

    def curve(self, kpi: str, key: str, fixed: dict, n: int = 60, group: tuple | None = None) -> dict:
        """1D response: KPI and band along the continuous parameter `key`, the other parameters at `fixed`. `supported` marks the
        points inside the analysed region."""
        cont = self.continuous
        p = next((c for c in cont if c.key == key), None)
        if p is None:
            raise SurrogateError(f"'{key}' is not a continuous parameter of the surrogate")
        grp = group if group is not None else self._group_of(fixed)
        xs = np.linspace(p.lo, p.hi, int(n))
        X = np.tile([float(fixed.get(c.key, 0.5 * (c.lo + c.hi))) for c in cont], (len(xs), 1))
        X[:, cont.index(p)] = xs
        v, lo, hi = self.predict_arrays(kpi, grp, X)
        return {"x": xs, "value": v, "lower": lo, "upper": hi, "supported": self.supported(kpi, grp, X)}

    def grid(self, kpi: str, kx: str, ky: str, fixed: dict, n: int = 40, group: tuple | None = None) -> dict:
        """2D response surface of the KPI over two continuous parameters. `supported` marks the cells inside the analysed region."""
        cont = self.continuous
        keys = [c.key for c in cont]
        if kx not in keys or ky not in keys or kx == ky:
            raise SurrogateError("Two different continuous parameters are needed for the response surface")
        px, py = cont[keys.index(kx)], cont[keys.index(ky)]
        gx, gy = np.meshgrid(np.linspace(px.lo, px.hi, n), np.linspace(py.lo, py.hi, n))
        X = np.tile([float(fixed.get(c.key, 0.5 * (c.lo + c.hi))) for c in cont], (gx.size, 1))
        X[:, keys.index(kx)] = gx.ravel()
        X[:, keys.index(ky)] = gy.ravel()
        grp = group if group is not None else self._group_of(fixed)
        v, lo, hi = self.predict_arrays(kpi, grp, X)
        return {"x": gx, "y": gy, "value": v.reshape(gx.shape), "lower": lo.reshape(gx.shape), "upper": hi.reshape(gx.shape),
                "supported": self.supported(kpi, grp, X).reshape(gx.shape)}

    def validation_table(self, kpi: str | None = None, group: tuple | None = None) -> list[dict]:
        """The held-out points: parameters, analysed value, predicted value and band, inside the band?"""
        kpi = kpi or self.kpis[0]
        out = []
        for grp, mems in self.members.items():
            if group is not None and grp != group or kpi not in mems:
                continue
            m = mems[kpi]
            if not m.cal_ids:
                continue
            X, y = self._rows_xy(m.cal_ids, kpi)
            v, lo, hi = self.predict_arrays(kpi, grp, X)
            for i, rid in enumerate(m.cal_ids):
                out.append({"row": rid, "params": dict(self.rows[rid]["params"]), "analysed": float(y[i]), "predicted": float(v[i]),
                            "lower": float(lo[i]), "upper": float(hi[i]), "inside": bool(lo[i] <= y[i] <= hi[i]),
                            "rel_error": float(abs(v[i] - y[i]) / abs(y[i])) if y[i] else None})
        return out

    def suggest(self, n: int = 3, seed: int = 0, group: tuple | None = None) -> list[dict]:
        """Where to analyse next: the points INSIDE the analysed region where the model is least informed (the predictive σ of the Gaussian
        process, or the distance to the nearest analysed point for the radial-basis fallback), well apart from each other. Adaptive
        refinement – run them and refit."""
        from .design import candidate_pool
        grp = group if group is not None else next(iter(self.members), ())
        cont = self.continuous
        if not cont or grp not in self.members:
            return []
        pool = candidate_pool([(p.lo, p.hi) for p in cont], 512, seed)
        score = np.zeros(len(pool))
        keep = np.ones(len(pool), dtype=bool)
        for k, m in self.members[grp].items():
            sp = m.backend.spread(self._unit(pool))
            score = np.maximum(score, sp / max(float(np.max(sp)), 1e-12))
            if m.support is not None:
                keep &= m.support.supported(self._unit(pool))
        chosen: list[int] = []
        unit = self._unit(pool)
        radius = 0.35 / max(1, len(cont)) ** 0.5
        for i in np.argsort(-score, kind="stable"):
            if not keep[i]:
                continue
            if all(np.linalg.norm(unit[i] - unit[j]) >= radius for j in chosen):
                chosen.append(int(i))
            if len(chosen) >= n:
                break
        cat = {p.key: grp[i] for i, p in enumerate(self.categorical)}
        return [{**{p.key: float(pool[i][j]) for j, p in enumerate(cont)}, **cat} for i in chosen]

    # ------------------------------------------------------------------ report
    def describe(self) -> dict:
        """Everything the GUI / MCP / report shows: parameters and ranges, per (category, KPI) the data split, accuracy and band."""
        members = []
        for grp, mems in self.members.items():
            for k, m in mems.items():
                members.append({"category": list(grp), "kpi": k, "backend": self.backend, "transform": m.transform,
                                "n_train": m.cal.n_train, "n_validation": m.cal.n_cal,
                                "design": m.support.kind if m.support is not None else "", **m.cal.to_dict()})
        return {"parameters": [p.to_dict() for p in self.params], "kpis": list(self.kpis), "level": self.level,
                "guaranteed_level": self.guaranteed_level, "backend": self.backend,
                "seed": self.seed, "validation_fraction": self.validation_fraction, "n_points": len(self.rows), "n_failed": len(self.failed),
                "reliable": self.reliable, "stale": self.stale, "members": members, "warnings": list(self.warnings),
                "sweep": dict(self.sweep), "skipped": [{"category": list(g), "reason": r} for g, r in self.skipped_groups.items()],
                "limitations": LIMITATIONS}

    # ------------------------------------------------------------------ refit with more points
    def refit_with(self, extra_rows: list[dict], study_fingerprint: str | None = None) -> "SurrogateModel":
        """A new surrogate over the same points plus `extra_rows` (rows as `training_set` makes them; the answer of "verify this point
        with FEM"). The earlier hold-out points stay held out; the new points are chosen by the model, not drawn at random, so they join the
        TRAINING set only (never the calibration). Refuses rows from a changed study (`study_fingerprint` is the current one) and a
        point outside the analysed region: it would widen the box of the model beyond what its calibration covers – a new parametric
        study over the larger range and a new fit do that."""
        if self.stale or (study_fingerprint is not None and self.fingerprint and study_fingerprint != self.fingerprint):
            raise SurrogateError("The study was changed after the surrogate was fitted – its points and the new ones do not belong together; "
                                 "run the parametric study again and fit a new surrogate")
        keys = [p.key for p in self.params]
        extra: list[dict] = []
        for r in extra_rows:
            r = {"params": dict(r["params"]), "kpi": dict(r["kpi"]), "adaptive": True}
            if any(k not in r["params"] for k in keys):
                raise SurrogateError("A new point lacks a parameter of the surrogate")
            grp = self._group_of(r["params"])
            mems = self.members.get(grp)
            if mems is None:
                raise SurrogateError(f"No model for the category {', '.join(grp) or '–'} – the new point cannot be added")
            u = self._unit(self._x(r["params"])[None, :])
            for k, m in mems.items():
                if r["kpi"].get(k) is not None and m.support is not None:
                    bad = m.support.reasons(u[0], self._x(r["params"]), [p.key for p in self.continuous])
                    if bad:
                        raise SurrogateError("The new point lies outside the analysed region of the surrogate (" + "; ".join(bad) + "): it would "
                                             "widen the model beyond its calibration – run a parametric study over the larger range and fit again")
            extra.append(r)
        n0 = len(self.rows)
        splits: dict[tuple, tuple[list[int], list[int]]] = {}
        for grp, mems in self.members.items():
            for k, m in mems.items():
                add = [n0 + j for j, r in enumerate(extra) if self._group_of(r["params"]) == grp and r["kpi"].get(k) is not None]
                splits[(grp, k)] = (m.train_ids + add, m.cal_ids)
        rows = self.rows + extra
        backend = self.backend
        note = ""
        if backend == "gp" and not resolve_ok("gp"):
            backend, note = "rbf", "scikit-learn is not installed here: refitted with the radial-basis backend"
        sm = _fit_rows(rows, keys, self.kpis, self.validation_fraction, self.seed, backend, self.level, None, splits=splits,
                       failed=self.failed, fingerprint=self.fingerprint, sweep=self.sweep, min_holdout=self.min_holdout,
                       fingerprint_keys=self.fingerprint_keys)
        if note:
            sm.warnings.insert(0, note)
        return sm

    # ------------------------------------------------------------------ files (no pickle)
    def save(self, path: str | Path) -> None:
        """One .npz: the design points (arrays + a JSON header with the settings, the split, the fitted hyperparameters, the study
        fingerprint and the sweep). No pickle; `load_surrogate` rebuilds the very same model without optimisation and recomputes every
        accuracy figure and band from the stored split."""
        from .store import write_npz
        write_npz(self, path)


def resolve_ok(name: str) -> bool:
    try:
        resolve_backend(name)
        return True
    except BackendError:
        return False


# ---------------------------------------------------------------------- fitting
def _ids_by_group(rows: list[dict], cats: list[ParamSpec]) -> dict[tuple, list[int]]:
    out: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        out.setdefault(tuple(str(r["params"][p.key]) for p in cats), []).append(i)
    return out


def _failed_unit(sm: SurrogateModel, grp: tuple) -> np.ndarray:
    """Unit coordinates of the failed analyses of the category `grp`."""
    cats = sm.categorical
    out = []
    for f in sm.failed:
        try:
            if tuple(str(f[p.key]) for p in cats) != grp:
                continue
            out.append([float(f[p.key]) for p in sm.continuous])
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        return np.zeros((0, len(sm.continuous)))
    X = np.array(out)
    return sm._unit(X[np.all(np.isfinite(X), axis=1)])


def _fit_member(sm: SurrogateModel, group: tuple, kpi: str, train_ids: list[int], cal_ids: list[int], hyper: dict | None = None) -> _Member:
    """Fits one (category, KPI) model on `train_ids` and calibrates the band on `cal_ids`. hyper = saved hyperparameters (a loaded
    file: no optimisation). EVERYTHING is computed here from the points and the split – a loaded file supplies nothing but the split and
    the hyperparameters, which are range-checked by the backend."""
    m = _Member(group, kpi)
    Xt, yt = sm._rows_xy(train_ids, kpi)
    Xc, yc = sm._rows_xy(cal_ids, kpi)
    m.train_ids, m.cal_ids = list(train_ids), list(cal_ids)
    m.transform = "log" if np.all(yt > 0) else "id"            # the TRAINING values decide: a hold-out value never changes the fit
    t = np.log(yt) if m.transform == "log" else yt
    m.ymean, m.ystd = float(np.mean(t)), float(np.std(t))
    if m.ystd <= 1e-12 * max(1.0, abs(m.ymean)):
        m.ystd = 1.0                                            # a constant KPI
    ys_t, ys_c = m.to_std(yt), m.to_std(yc)
    Ut, Uc = sm._unit(Xt), sm._unit(Xc)
    try:
        be = make_backend(sm.backend).fit(Ut, ys_t, seed=sm.seed) if hyper is None else restore_backend(sm.backend, Ut, ys_t, hyper)
    except BackendError as e:
        raise SurrogateError(f"{kpi}: {e}") from None
    m.backend = be
    mu_c, sd_c = be.predict(Uc)
    sc = scores_of(ys_c, mu_c, sd_c)
    q, exact = conformal_quantile(sc, sm.level)
    if not math.isfinite(q):
        q = Q_MAX
    pred = m.from_std(mu_c)
    rel = np.abs(pred - yc) / np.maximum(np.abs(yc), 1e-300)
    with np.errstate(over="ignore", invalid="ignore"):
        half = 0.5 * (m.from_std(mu_c + q * sd_c) - m.from_std(mu_c - q * sd_c))
    yall = np.concatenate([yt, yc])
    c = Calibration(level=sm.level, n_train=len(train_ids), n_cal=len(cal_ids), q=q, exact=exact, coverage_cal=coverage(sc, q),
                    r2=r_squared(yc, pred), spread=float(np.std(yall)), scale=float(np.median(np.abs(yall))), half_band=float(np.median(half)) if np.all(np.isfinite(half)) else float("inf"),
                    median_rel_err=float(np.median(rel)), max_rel_err=float(np.max(rel)))
    # check of the calibrated band: cross-validation over the training points – leave-one-out up to CV_LOO_MAX points (that is how the
    # band is calibrated: a point the fit has not seen), else 10-fold – with the hyperparameters of the fit to ALL training points, so it
    # is slightly optimistic (the trigger of `judge` is tighter than a plain test for that); no optimiser is run, a loaded file gives the
    # very same figures
    hp = be.params()
    k = len(train_ids) if len(train_ids) <= CV_LOO_MAX else CV_FOLDS
    folds = np.arange(len(train_ids)) % k
    cv_scores: list[float] = []
    closed = be.cv_predict(Ut, ys_t, folds)                  # the left-out predictions from ONE matrix inverse (model.py), None where that cannot be trusted
    if closed is not None:
        cv_scores = list(scores_of(ys_t, closed[0], closed[1]))
    else:
        for f in range(k):
            te = folds == f
            try:
                b2 = restore_backend(sm.backend, Ut[~te], ys_t[~te], hp)
                mu, sd = b2.predict(Ut[te])
            except BackendError:
                continue
            cv_scores += list(scores_of(ys_t[te], mu, sd))
    c.coverage_cv = coverage(cv_scores, q) if cv_scores else float("nan")
    m.cal = judge(c)
    ids = list(train_ids) + list(cal_ids)
    # the region of the failed analyses AND of the analyses without a value of this KPI (they are as unknown as a failure)
    gaps = [i for i, r in enumerate(sm.rows) if r["kpi"].get(kpi) is None and tuple(str(r["params"][p.key]) for p in sm.categorical) == group]
    # the support is decided by the usable points and the failed ones only: KPIs that rest on the same points (the usual case: every analysis gave every KPI) share one
    # Support, and with it its hull and its search for holes, which are the expensive part
    key = (group, tuple(ids), tuple(gaps))
    sup = sm._supports.get(key)
    if sup is None:
        Xa, _ = sm._rows_xy(ids, kpi)
        Fa = _failed_unit(sm, group)
        if gaps:
            Fa = np.vstack([Fa, sm._unit(np.array([sm._x(sm.rows[i]["params"]) for i in gaps]).reshape(len(gaps), len(sm.continuous)))])
        sup = sm._supports[key] = Support(sm._unit(Xa), Xa, Fa)
    m.support = sup
    return m


def _fit_rows(rows: list[dict], keys: list[str], kpis: list[str] | None, fraction: float, seed: int, backend: str, level: float,
              bounds: dict | None, *, splits: dict[tuple, tuple[list[int], list[int]]] | None = None, failed: list[dict] | None = None,
              fingerprint: str = "", sweep: dict | None = None, min_holdout: int | None = None,
              fingerprint_keys: list[str] | None = None) -> SurrogateModel:
    if not rows:
        raise SurrogateError("No design points to fit (all failed or none were run)")
    if len(rows) > MAX_ROWS:
        raise SurrogateError(f"At most {MAX_ROWS} design points")
    warns: list[str] = []
    try:
        backend = resolve_backend(backend)
    except BackendError as e:
        raise SurrogateError(str(e)) from None
    specs = _specs(rows, keys, warns)
    cont = [p for p in specs if p.kind == "continuous"]
    cats = [p for p in specs if p.kind == "categorical"]
    if not cont:
        raise SurrogateError("The surrogate needs at least one continuous (numeric) parameter that changes between the design points")
    if len(cont) > MAX_CONTINUOUS:
        raise SurrogateError(f"At most {MAX_CONTINUOUS} continuous parameters (got {len(cont)}: {', '.join(p.key for p in cont)}) – "
                             "a Gaussian process is not reliable with so few points per dimension")
    if not 0.05 <= fraction <= 0.5:
        raise SurrogateError("validation_fraction must be between 0.05 and 0.5")
    if not 0.5 <= level < 1.0:
        raise SurrogateError("level must be in [0.5; 1)")
    mv = holdout_minimum(level, min_holdout)
    wanted = list(kpis) if kpis else None
    if wanted:
        bad = [k for k in wanted if k not in SURROGATE_KPIS]
        if bad:
            raise SurrogateError(f"Unknown KPI {', '.join(bad)} (available: {', '.join(SURROGATE_KPIS)})")
    need = min_points(len(cont), fraction, level, min_holdout)
    names = wanted or [k for k in SURROGATE_KPIS if sum(r["kpi"].get(k) is not None for r in rows) >= need]
    if not names:
        have = max((sum(r["kpi"].get(k) is not None for r in rows) for k in SURROGATE_KPIS), default=0)
        raise SurrogateError(f"Training set too small: {have} usable design points; {len(cont)} continuous parameter(s) need at least "
                             f"{need} per KPI (training ≥ {min_train(len(cont))} + hold-out ≥ {mv})")
    sm = SurrogateModel(specs, names, rows, level, seed, backend, fraction)
    sm.warnings = warns
    sm.failed = [dict(f) for f in (failed or [])][:MAX_FAILED]
    sm.fingerprint, sm.sweep, sm.min_holdout = str(fingerprint or ""), dict(sweep or {}), min_holdout
    sm.fingerprint_keys = list(fingerprint_keys if fingerprint_keys is not None else keys) if sm.fingerprint else []
    if not sm.fingerprint:
        sm.warnings.append(NO_FINGERPRINT)
    if failed:
        sm.warnings.append(describe_failed(sm.failed, [p.key for p in specs], len(rows) + len(sm.failed)))
    for p in cont:                                              # the analysed points reach less far than the sampled range
        if bounds and p.key in bounds:
            blo, bhi = float(bounds[p.key][0]), float(bounds[p.key][1])
            span = max(bhi - blo, 1e-300)
            if p.lo > blo + GAP_WARN * span or p.hi < bhi - GAP_WARN * span:
                sm.warnings.append(f"The analysed points cover {p.key} in [{p.lo:g}; {p.hi:g}] of the sampled range [{blo:g}; {bhi:g}]: "
                                   "the surrogate answers only inside the analysed region")
    d = len(cont)
    groups = _ids_by_group(rows, cats)
    fitted_any = False
    refusals: list[str] = []
    for grp, ids in groups.items():
        label = "/".join(grp) or "all points"
        for k in names:
            ok_ids = [i for i in ids if rows[i]["kpi"].get(k) is not None]
            if len(ok_ids) < need:
                msg = (f"{label}: {k} has {len(ok_ids)} usable design points, at least {need} are needed for {d} parameter(s) "
                       f"(training ≥ {min_train(d)} + hold-out ≥ {mv})")
                if wanted:
                    raise SurrogateError("Training set too small – " + msg)
                sm.warnings.append("Left out: " + msg)
                refusals.append(msg)
                sm.skipped_groups.setdefault(grp, "too few design points")
                continue
            U = sm._unit(np.array([sm._x(rows[i]["params"]) for i in ok_ids]))
            rank = spanned_rank(U)
            if rank < d:
                msg = (f"{label}: the analysed points of {k} span only {rank} of the {d} parameters (they lie on a line or in a plane of the "
                       "parameter space): a surrogate of all of them cannot be fitted – vary the parameters independently (mode lhs)")
                if wanted:
                    raise SurrogateError(msg)
                sm.warnings.append("Left out: " + msg)
                refusals.append(msg)
                sm.skipped_groups.setdefault(grp, "the design points do not span the parameters")
                continue
            if splits is not None and (grp, k) in splits:
                tr, ca = list(splits[(grp, k)][0]), list(splits[(grp, k)][1])
            else:
                free = [i for i in ok_ids if not rows[i].get("adaptive")]           # a point chosen by the model never calibrates it
                fixed = [i for i in ok_ids if rows[i].get("adaptive")]
                n_val = min(validation_size(len(ok_ids), fraction, mv), len(free) - 1)
                a, b = split_indices(len(free), fraction, mv, seed, n_val=n_val)
                tr, ca = [free[i] for i in a] + fixed, [free[i] for i in b]
            if len(tr) < min_train(d):
                raise SurrogateError(f"Training set too small – {label}: {k} has {len(tr)} training points, at least {min_train(d)} are needed")
            m = _fit_member(sm, grp, k, tr, ca)
            sm.members.setdefault(grp, {})[k] = m
            fitted_any = True
            if m.support is not None:
                sm.warnings += [f"{label}: {note}" if grp else note for note in m.support.design_notes([p.key for p in cont])]
            if len(tr) < WARN_TRAIN_PER_DIM * d:
                sm.warnings.append(f"{label}: {k}: only {len(tr)} training points for {d} parameter(s) (≥ {WARN_TRAIN_PER_DIM * d} recommended) – "
                                   "check the validation table")
    if not fitted_any:
        if refusals:
            raise SurrogateError("No surrogate could be fitted: " + "; ".join(dict.fromkeys(refusals)))
        raise SurrogateError(f"Training set too small: {len(rows)} design points; {d} continuous parameter(s) need at least {need} per category")
    sm.kpis = [k for k in names if any(k in mems for mems in sm.members.values())]
    sm.warnings = list(dict.fromkeys(sm.warnings))
    return sm


def fit_surrogate(points, keys: list[str] | None = None, kpis: list[str] | None = None, *, validation_fraction: float = VALIDATION_FRACTION,
                  seed: int = 0, backend: str = "auto", level: float = LEVEL, bounds: dict | None = None, fingerprint: str = "",
                  sweep: dict | None = None, min_holdout: int | None = None, fingerprint_keys: list[str] | None = None) -> SurrogateModel:
    """Fits the surrogate to the design points of a parametric study (`ParametricStudyResult.points` or rows from `training_set`).

    keys = the parameters to use (default all); kpis = the KPI to model (default: every KPI with enough values); validation_fraction =
    the hold-out share that calibrates the band (default 20 %, at least ⌈level / (1 − level)⌉ = 9 points: fewer cannot guarantee the
    level; `min_holdout` lets a caller ask for fewer – the band then states n / (n + 1) and the model is not "reliable"); seed fixes the
    split and the optimiser, so the same data give the same model; bounds = {key: (lo, hi)} the range that was SAMPLED – used only to
    warn that the converged points reach less far (the model answers inside the analysed points, never inside the requested box);
    fingerprint / sweep = the study fingerprint and specification of the parametric study (stale.py); fingerprint_keys = the swept keys
    that fingerprint neutralised (default: `keys` – a swept key that is constant over the points is no parameter of the model, but it
    stays neutralised when the model is compared with the study).
    Failed points (`converged` False) are left out and their region is reported in a warning.
    Raises SurrogateError when the training set is too small or does not span the parameters."""
    if points and isinstance(points[0], dict) and "params" in points[0] and "kpi" in points[0] and isinstance(points[0]["kpi"], dict) \
            and set(points[0]["kpi"]) >= set(SURROGATE_KPIS) and all("converged" not in p for p in points):
        rows = [{"params": dict(r["params"]), "kpi": dict(r["kpi"]), **({"adaptive": True} if r.get("adaptive") else {})} for r in points]
        failed: list[dict] = []
        ks = keys if keys is not None else list(dict.fromkeys(k for r in rows for k in r["params"]))
    else:
        rows, failed = _split_points(points)
        ks = keys if keys is not None else list(dict.fromkeys(k for r in rows for k in r["params"]))
    return _fit_rows(rows, ks, kpis, float(validation_fraction), int(seed), backend, float(level), bounds, failed=failed,
                     fingerprint=fingerprint, sweep=sweep, min_holdout=min_holdout, fingerprint_keys=fingerprint_keys)


def load_surrogate(path: str | Path) -> SurrogateModel:
    """Reads a file written by `SurrogateModel.save`; the model is rebuilt from the stored split and hyperparameters (no pickle) and
    every figure is recomputed. A damaged, crafted or foreign file is a SurrogateError (sizes, types, ranges and indices are checked
    before anything is read or fitted)."""
    from .store import read_model
    return read_model(path)


def describe_text(sm: SurrogateModel) -> list[str]:
    """Short lines for logs and the GUI summary."""
    out = [f"Surrogate ({sm.backend}), {len(sm.rows)} design points, band {sm.guaranteed_level:.0%}, seed {sm.seed}"]
    for grp, mems in sm.members.items():
        for k, m in mems.items():
            c = m.cal
            out.append(f"{'/'.join(grp) + ': ' if grp else ''}{k}: R² {c.r2:.3f}, median error {c.median_rel_err:.1%}, q = {c.q:.2f}, coverage "
                       f"{c.coverage_cv:.0%} (CV) – {'reliable' if c.reliable else 'NOT reliable'}")
    return out


__all__ = ["SurrogateModel", "SurrogateError", "ParamSpec", "fit_surrogate", "load_surrogate", "training_set", "SURROGATE_KPIS",
           "MIN_TRAIN_PER_DIM", "min_train", "min_points", "holdout_minimum", "describe_text"]

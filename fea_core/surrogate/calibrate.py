"""Split-conformal calibration of the error bars (F15-S4).

Nonconformity score of a held-out point: s = |y − μ| / σ (in the transformed output space). With n calibration scores the band
μ ± q·σ, q = the ⌈(n + 1)·level⌉-th smallest score, covers a new exchangeable point with probability ≥ level (Vovk, Lei et al.).
The guarantee is marginal over the draw of the calibration set (an AVERAGE over the sampled region, not a statement about one
corner of it) and model-agnostic: the same code calibrates the Gaussian process and the radial-basis fallback.
A finite quantile exists only from ⌈level / (1 − level)⌉ calibration points (9 at 90 %); with fewer the band is the largest score
and guarantees only n / (n + 1) – that level, not the requested one, is what every report states (`Calibration.guaranteed`).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict, dataclass, field

import numpy as np

LEVEL = 0.90
MIN_R2 = 0.9                    # the plan: validation R² below this = "not reliable"
COVERAGE_SLACK = 0.10           # the cross-validated coverage may be at most this far below the level (80 % for a 90 % band) …
CV_Z = 1.645                    # … and, for a large training set, not further below than this many standard errors (one-sided 5 %; the old rule used 2.33)
FLAT_REL = 0.01                 # a result whose band is within ±1 % of its value and wider than its spread is "constant": R² is not tested
COARSE_CAL = 20                 # fewer calibration points: the band is correct but coarse (and the coverage estimate noisy)
Q_MAX = 1e6                     # a band factor beyond this is "unbounded" (a held-out value that the log transform cannot score)
EPS = 1e-12


def required_calibration(level: float = LEVEL) -> int:
    """Smallest hold-out set for which the conformal band holds at `level`: ⌈level / (1 − level)⌉ (9 at 90 %)."""
    return max(1, int(math.ceil(level / (1.0 - level) - 1e-9)))


def conformal_quantile(scores, level: float = LEVEL) -> tuple[float, bool]:
    """(q, exact): the ⌈(n + 1)·level⌉-th smallest score. exact = False when n is too small for a finite quantile
    (n < ⌈level / (1 − level)⌉, i.e. fewer than 9 points at 90 %): the largest score is returned then and the guarantee is lost."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("No calibration points")
    k = int(math.ceil((n + 1) * level - 1e-9))
    if k > n:
        return float(s[-1]), False
    return float(s[k - 1]), True


def coverage(scores, q: float) -> float:
    s = np.asarray(scores, dtype=float)
    return float(np.mean(s <= q + EPS)) if len(s) else float("nan")


def scores_of(y, mu, sigma) -> np.ndarray:
    """|y − μ| / σ with σ floored (σ = 0 would give an infinite score for a non-zero error); a value that cannot be scored (NaN) is an
    infinite score – it never hides in the quantile."""
    sg = np.maximum(np.asarray(sigma, dtype=float), EPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.abs(np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sg
    return np.where(np.isnan(s), np.inf, s)


def r_squared(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= EPS else 0.0
    return 1.0 - ss_res / ss_tot


@dataclass
class Calibration:
    """Quality of one fitted (group, KPI) model: the conformal factor, the hold-out accuracy and the verdict. Everything here is
    COMPUTED from the data and the stored split (never read from a file)."""
    level: float = LEVEL
    n_train: int = 0
    n_cal: int = 0
    q: float = 1.0                         # band half-width in units of σ (transformed space)
    exact: bool = True                     # finite-sample conformal guarantee holds at `level`
    guaranteed: float = LEVEL              # coverage that the conformal guarantee gives: `level` when exact, else n / (n + 1)
    coverage_cal: float = float("nan")     # fraction of calibration points inside the band (≥ level when exact, by construction)
    coverage_cv: float = float("nan")      # fraction of the TRAINING points inside the band in a cross-validation (leave-one-out up to 100
    #                                        points, else 10-fold; a check – the hyperparameters are those fitted to all training points, so
    #                                        it is slightly optimistic; training points at the edge of the design are the hard ones)
    r2: float = float("nan")               # R² on the held-out points (original scale)
    r2_tested: bool = True                 # False: the result hardly varies (spread below the band) – R² says nothing then
    spread: float = float("nan")           # standard deviation of the result over the analysed points
    half_band: float = float("nan")        # median half-width of the band on the held-out points (units of the result)
    scale: float = float("nan")            # median |value| of the result over the analysed points
    median_rel_err: float = float("nan")   # median |error| / |value| on the held-out points
    max_rel_err: float = float("nan")
    reliable: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        """Only the declared fields, with type checks; anything else in `d` is ignored. (A saved calibration is informational: a loaded
        model recomputes its calibration from the stored split.)"""
        c = cls()
        kinds = {f.name: str(f.type) for f in dataclasses.fields(cls)}
        for k, v in d.items():
            t = kinds.get(k)
            if t is None:
                continue
            if t == "list[str]":
                if isinstance(v, list):
                    c.reasons = [str(x)[:300] for x in v[:20]]
            elif t == "bool":
                if isinstance(v, bool):
                    setattr(c, k, v)
            elif t == "int":
                if isinstance(v, int) and not isinstance(v, bool):
                    setattr(c, k, v)
            elif v is None:
                setattr(c, k, float("nan"))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                setattr(c, k, float(v))
        return c


def coverage_floor(level: float, n_train: int) -> float:
    """Lowest cross-validated coverage that is still accepted: the larger of `level − 10 %` and `level − 1.645 σ` of the binomial error of
    the estimate (the first version used `level − 2.33 σ`: 0.78 for 31 points, 0.83 for 100; this one is 0.81 and 0.85 – tighter at every
    size, because the cross-validation is optimistic: its hyperparameters are those of the fit to ALL training points). The coverage of
    a calibration with n points is itself uncertain (Beta distributed), so a correct model fails this test now and then (~15 % with
    9 calibration points, less with more)."""
    return max(level - COVERAGE_SLACK, level - CV_Z * math.sqrt(level * (1.0 - level) / max(n_train, 1)))


def judge(c: Calibration) -> Calibration:
    """Fills `guaranteed`, `reliable` and `reasons`. A model is reliable only when
      * the band is exact (⌈level / (1 − level)⌉ calibration points, 9 at 90 %): fewer points guarantee only n / (n + 1), which is
        then the level of the band everywhere, and the model is never called reliable;
      * the validation R² is at least 0.9 – unless the result is practically constant (its spread is below the band and the band is
        within ±1 % of the value): R² only measures noise then, and a nearly constant result must not make the model "unreliable";
      * the cross-validated coverage is not below `coverage_floor`."""
    why: list[str] = []
    c.guaranteed = c.level if c.exact else c.n_cal / (c.n_cal + 1.0)
    reliable = bool(c.exact)
    if not c.exact:
        why.append(f"only {c.n_cal} calibration points: the band is guaranteed for {c.guaranteed:.0%}, not {c.level:.0%} "
                   f"(≥ {required_calibration(c.level)} points needed)")
    elif c.n_cal < COARSE_CAL:
        why.append(f"{c.n_cal} calibration points – the band is coarse (≥ {COARSE_CAL} recommended)")
    flat = (math.isfinite(c.spread) and math.isfinite(c.half_band) and math.isfinite(c.scale)
            and c.spread <= c.half_band <= FLAT_REL * c.scale)
    c.r2_tested = not flat
    if flat:
        why.append("the result is practically constant (its spread is below the band, and the band is within "
                   f"±{FLAT_REL:.0%} of the value) – R² is not tested")
    elif math.isfinite(c.r2) and c.r2 < MIN_R2:
        why.append(f"validation R² = {c.r2:.3f} < {MIN_R2}")
        reliable = False
    if not math.isfinite(c.q) or c.q >= Q_MAX:
        why.append("a held-out value cannot be scored (not positive although the fit is in the log space): the band is unbounded")
        reliable = False
    if math.isfinite(c.coverage_cv) and c.n_train > 0:
        floor = coverage_floor(c.level, c.n_train)
        if c.coverage_cv < floor:
            why.append(f"cross-validated coverage {c.coverage_cv:.0%} is below {floor:.0%} (target {c.level:.0%})")
            reliable = False
    c.reliable, c.reasons = reliable, why
    return c

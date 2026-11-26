"""Primary notching / force limiting of a sine or random test profile (F14 stage S4).

The response of the structure to a base acceleration is linear, so the quantity that a limit applies to (the interface force or moment, the acceleration at a
probe, an equivalent stress) is the product of the input profile and a transfer function T(f) per unit input:
  * sine:   X(f) = T(f)·a(f)      (amplitudes; a in g)        → the notched amplitude  a_n = a·min(1, L / (J·X))
  * random: S_X(f) = T²(f)·ASD(f) (PSD of the quantity)       → the notched ASD        ASD_n = ASD·min(1, L_psd / (J·S_X))
J = the design factor applied to the predicted AMPLITUDE of the quantity (force, moment, acceleration, stress), L = the limit. A PSD is the square of an amplitude, so on a
PSD limit (random profile) the factor acts as J²: S_X·J² ≤ L_psd. The rules of the requirement set shape the notch:
  * `notch_floor_ratio`    – the notched level is never below this fraction of the unnotched specification level (a deeper notch is clamped; if the limit is still
                              exceeded the notch is reported as floor-limited and the verdict is FAIL);
  * `notch_min_width_oct`  – every notch is widened (symmetrically in log f, flat bottom at its deepest level) to at least this width in octaves.
Nothing here is a standard number: the limit, the factor and both rules come from the confirmed requirement set (`s4_limits`). A value passed by the caller is
used but never counts as confirmed; the verdict is then UNCONFIRMED_REQUIREMENTS (never PASS), the same rule as for the margins of safety.

A non-finite level, response or limit, a design factor ≤ 0 or an absurd minimum width is an error (`NotchError`), never a verdict. A limit and the quantity that it limits
are checked at the points of the analysis only: for a stress or an acceleration at a probe the verdict says so ("at probe X only").

Outputs: the notched profile on the analysis grid, a simplified breakpoint table for the test house, the notch justification table (unnotched vs notched, reduction
in dB, remaining margin) and CSV text for both (`export_csv.text_cell` on every text cell).
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .s4_limits import S4Limits

PASS, FAIL, UNCONFIRMED = "PASS", "FAIL", "UNCONFIRMED_REQUIREMENTS"
KINDS = ("sine", "random")
LIMIT_KINDS = ("force", "moment", "probe_acc", "stress", "force_psd")
TABLE_TOL = 1e-3                     # numerical tolerance of the check of the exported breakpoint table against the limit (0.1 %, the resolution of the check)
MAX_FINE_POINTS = 400_000            # the grid on which the exported breakpoint table is re-evaluated: at most this many frequencies (the evaluation is chunked, see `interface_transfer`)
MAX_WIDTH_OCT = 40.0                # numerical guard (2**40 is a frequency ratio of 1e12): a minimum notch width above it is an input error, not a notch rule
LIMIT_KEY_OF = {"force": "notch_force_limit_N", "moment": "notch_moment_limit_Nmm", "probe_acc": "notch_probe_acc_limit_g",
                "stress": "notch_stress_limit_MPa", "force_psd": "notch_force_psd_limit_N2_Hz"}


class NotchError(ValueError):
    """The notching inputs are inconsistent."""


@dataclass
class Notch:
    """One notch of the result (a band where the input is reduced)."""
    f_lo: float
    f_hi: float
    f_center: float
    width_oct: float
    f_deepest: float
    scale_min: float                 # smallest notched/unnotched ratio in the notch
    reduction_db: float              # 20 log10 (sine) / 10 log10 (random) of scale_min (negative)
    input_unnotched: float           # input level at the deepest point (g or g²/Hz)
    input_notched: float
    quantity_unnotched_peak: float   # predicted limit quantity: peak in the notch band before the notch
    quantity_notched_peak: float     # … after the notch
    limit: float
    margin_remaining: float | None   # limit / (J · notched peak) − 1
    widened: bool
    floor_limited: bool
    modes_near_Hz: list[float] = field(default_factory=list)
    status: str = "ok"               # ok | exceeds_limit (floor-limited and still above the limit)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class NotchResult:
    kind: str
    limit_kind: str
    f: np.ndarray
    level: np.ndarray                # unnotched input (g or g²/Hz)
    notched: np.ndarray
    quantity: np.ndarray             # predicted limit quantity of the unnotched input (amplitude, or PSD)
    quantity_notched: np.ndarray
    limit: float
    design_factor: float
    notches: list[Notch]
    verdict: str
    verdict_text: str
    inputs: list[dict]               # where every number came from: name, value, source, confirmed
    warnings: list[str] = field(default_factory=list)
    breakpoints: list[tuple[float, float]] = field(default_factory=list)
    design_factor_applied: float = 1.0       # J (amplitude quantities) or J² (a PSD)
    scope: str = ""
    table_ratio: float | None = None         # J·(response of the exported breakpoint table, on a grid 10x finer) / limit: <= 1 + TABLE_TOL when the table keeps the limit

    @property
    def reduction_db(self) -> np.ndarray:
        s = np.where(self.level > 0, self.notched / np.where(self.level > 0, self.level, 1.0), 1.0)
        return (20.0 if self.kind == "sine" else 10.0) * np.log10(np.maximum(s, 1e-300))

    def summary(self) -> dict:
        return {"kind": self.kind, "limit_kind": self.limit_kind, "limit": self.limit, "design_factor": self.design_factor,
                "design_factor_applied": self.design_factor_applied, "scope": self.scope, "table_ratio": self.table_ratio, "table_tolerance": TABLE_TOL, "verdict": self.verdict,
                "verdict_text": self.verdict_text, "notches": [n.to_dict() for n in self.notches], "inputs": self.inputs, "warnings": self.warnings,
                "max_reduction_db": float(self.reduction_db.min()) if len(self.f) else 0.0,
                "unnotched_peak": float(self.quantity.max()) if len(self.f) else 0.0, "notched_peak": float(self.quantity_notched.max()) if len(self.f) else 0.0}


# ------------------------------------------------------------------ the numerical core
def _regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i0, i1] of consecutive True values."""
    m = np.asarray(mask, bool)
    if not m.size:
        return []
    d = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))      # +1 where a run starts, -1 one after its last element
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _edge(f: np.ndarray, i: int, j: int) -> float:
    """Edge of a notch between the first sample inside (i) and the first sample outside (j): their geometric mean (the grid step is the resolution); the grid edge itself."""
    if j < 0 or j >= len(f):
        return float(f[i])
    return float(math.sqrt(f[i] * f[j]))


def notch_scale(f: np.ndarray, response: np.ndarray, limit: float | np.ndarray, design_factor: float = 1.0, floor_ratio: float | None = None,
                min_width_oct: float | None = None) -> tuple[np.ndarray, list[dict]]:
    """The notch scale s(f) in (0, 1] (notched level / unnotched level) and the description of the notch bands.

    response = the predicted limit quantity of the UNnotched input (amplitude for a sine profile, PSD for a random one); limit may be a table value per f.
    The same formula serves a sine amplitude and a PSD (both are proportional to the input level); only the reduction in dB differs (20 / 10 log10)."""
    f = np.asarray(f, float)
    X = np.asarray(response, float) * float(design_factor)
    L = np.broadcast_to(np.asarray(limit, float), f.shape)
    if not len(f) or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0) or f[0] <= 0:
        raise NotchError("The frequencies must be finite, positive and increasing")
    if not np.all(np.isfinite(X)) or np.any(X < 0):
        raise NotchError("The predicted quantity has a non-finite or negative value (a transfer function with no input at some frequency?) - no notch is computed")
    if not np.all(np.isfinite(L)) or np.any(L <= 0):
        raise NotchError("The limit must be finite and positive at every frequency")
    if not (math.isfinite(float(design_factor)) and float(design_factor) > 0):
        raise NotchError("The design factor must be finite and positive")
    if min_width_oct is not None and not (math.isfinite(float(min_width_oct)) and 0.0 < float(min_width_oct) <= MAX_WIDTH_OCT):
        raise NotchError(f"The minimum notch width must be between 0 and {MAX_WIDTH_OCT:g} octaves")
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(X > 0, np.minimum(1.0, L / np.where(X > 0, X, 1.0)), 1.0)
    s = np.clip(s, 0.0, 1.0)
    floor = None if floor_ratio is None else float(floor_ratio)
    if floor is not None:
        if not 0.0 <= floor <= 1.0:
            raise NotchError("notch_floor_ratio must be between 0 and 1")
        s = np.maximum(s, floor)
    bands: list[dict] = []
    widened_any = np.zeros(len(f), bool)
    for _ in range(8):                                   # widening can merge neighbouring notches → repeat until nothing changes
        regs = _regions(s < 1.0 - 1e-12)
        changed = False
        for i0, i1 in regs:
            lo = _edge(f, i0, i0 - 1) if i0 > 0 else float(f[i0])
            hi = _edge(f, i1, i1 + 1) if i1 < len(f) - 1 else float(f[i1])
            width = math.log2(hi / lo) if hi > lo > 0 else 0.0
            if min_width_oct and width < float(min_width_oct) - 1e-9:
                c = math.sqrt(lo * hi)
                half = 0.5 * float(min_width_oct)
                new_lo, new_hi = c / 2.0 ** half, c * 2.0 ** half
                m = (f >= new_lo) & (f <= new_hi)
                if m.any():                                  # one more sample on each side: the edges (midpoints to the outside samples) then enclose the minimum width
                    idx = np.nonzero(m)[0]
                    m[max(int(idx[0]) - 1, 0):min(int(idx[-1]) + 2, len(f))] = True
                smin = float(s[i0:i1 + 1].min())
                new = np.minimum(s[m], smin)
                if not np.array_equal(new, s[m]):
                    s[m] = new
                    widened_any |= m
                    changed = True
        if not changed:
            break
    for i0, i1 in _regions(s < 1.0 - 1e-12):
        lo = _edge(f, i0, i0 - 1) if i0 > 0 else float(f[i0])
        hi = _edge(f, i1, i1 + 1) if i1 < len(f) - 1 else float(f[i1])
        seg = slice(i0, i1 + 1)
        k = i0 + int(np.argmin(s[seg]))
        bands.append({"i0": i0, "i1": i1, "f_lo": lo, "f_hi": hi, "k": k, "widened": bool(widened_any[seg].any()),
                      "floor_limited": floor is not None and abs(float(s[k]) - floor) < 1e-12 and floor > 0.0})
    return s, bands


def _rdp(x: np.ndarray, y: np.ndarray, tol: float) -> list[int]:
    """Ramer–Douglas–Peucker simplification of the polyline (x, y); the kept indices."""
    n = len(x)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        xs, ys = x[a + 1:b], y[a + 1:b]
        t = (xs - x[a]) / (x[b] - x[a]) if x[b] != x[a] else np.zeros_like(xs)
        d = np.abs(ys - (y[a] + t * (y[b] - y[a])))
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return [int(i) for i in np.nonzero(keep)[0]]


def _rdp_below(x: np.ndarray, y: np.ndarray, tol: float) -> list[int]:
    """Simplification of the polyline (x, y) that NEVER lies above it at a vertex of the polyline (one-sided: the simplified line may be lower by `tol`, not higher).
    Every kept index is a vertex; between two kept vertices the simplified segment is a straight line, so it is also below the dense polyline between its vertices."""
    n = len(x)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        xs, ys = x[a + 1:b], y[a + 1:b]
        t = (xs - x[a]) / (x[b] - x[a]) if x[b] != x[a] else np.zeros_like(xs)
        dev = (y[a] + t * (y[b] - y[a])) - ys                      # > 0: the simplified line is ABOVE the dense profile (not allowed)
        i = int(np.argmax(dev))
        if dev[i] > 1e-12:
            split = i
        else:
            j = int(np.argmin(dev))
            if -dev[j] <= tol:
                continue
            split = j
        keep[a + 1 + split] = True
        stack += [(a, a + 1 + split), (a + 1 + split, b)]
    return [int(i) for i in np.nonzero(keep)[0]]


def fine_grid(f: np.ndarray) -> np.ndarray:
    """The frequencies for the re-evaluation of the exported table: every interval of the analysis grid is divided (in log f) into k parts, k = 10 for a small grid, fewer when the
    total would exceed `MAX_FINE_POINTS` (never fewer than 2: the midpoint of every interval is where an interpolated notch departs most from the exact one)."""
    f = np.asarray(f, float)
    n = len(f)
    k = int(min(10, max(2, MAX_FINE_POINTS // max(n, 1))))
    idx = np.linspace(0.0, n - 1.0, (n - 1) * k + 1)
    return np.exp(np.interp(idx, np.arange(n), np.log(f)))


def _rdp_band(x: np.ndarray, y: np.ndarray, c: np.ndarray, tol: float) -> list[int]:
    """Simplification of the polyline (x, y) whose vertices are the points (x_i, c_i) with y_i − tol ≤ c_i ≤ y_i, so that the simplified line P satisfies y_i − tol ≤ P(x_i) ≤ y_i at
    EVERY point (the polyline never lies above the dense polyline - at its own vertices and, being straight between them, in between - and at most `tol` below it).
    Points with c_i < y_i (lowered by up to tol/2) are what lets a concave part of the profile (a notch flank) be simplified at all: a chord through points ON a concave curve
    is above it, a chord through points lowered by tol/2 is allowed to deviate by tol/2 either way."""
    n = len(x)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    eps = 1e-12
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        xs = x[a + 1:b]
        t = (xs - x[a]) / (x[b] - x[a]) if x[b] != x[a] else np.zeros_like(xs)
        P = c[a] + t * (c[b] - c[a])
        viol = np.maximum(P - y[a + 1:b], (y[a + 1:b] - tol) - P)             # > 0: outside [y − tol, y]
        i = int(np.argmax(viol))
        if viol[i] <= eps:
            continue
        keep[a + 1 + i] = True
        stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return [int(i) for i in np.nonzero(keep)[0]]


def breakpoints(f: np.ndarray, level: np.ndarray, tol_db: float = 0.1, power: bool = False, unnotched: np.ndarray | None = None) -> list[tuple[float, float]]:
    """A simplified breakpoint table of the notched profile: a log-log polyline that is never ABOVE the dense profile and at most `tol_db` below it (a presentation
    tolerance, not a standard value) - the table handed to a test house therefore cannot exceed the limit that the dense profile keeps. With `unnotched` (the specification
    level) the vertices inside a notch lie tol_db/2 below the profile (see `_rdp_band`: the table is several times shorter), those outside it exactly on the specification."""
    f = np.asarray(f, float)
    lv = np.maximum(np.asarray(level, float), 1e-300)
    k = 10.0 if power else 20.0
    db = k * np.log10(lv)
    tol = float(tol_db)
    c = db.copy()
    if unnotched is not None and tol > 0:
        un = np.maximum(np.asarray(unnotched, float), 1e-300)
        c[lv < un * (1.0 - 1e-9)] -= 0.5 * tol
    idx = _rdp_band(np.log10(f), db, c, tol)
    return [(float(f[i]), float(lv[i]) if c[i] == db[i] else float(10.0 ** (c[i] / k))) for i in idx]


# ------------------------------------------------------------------ rules and the high-level function
@dataclass
class NotchSpec:
    """What to notch: `kind` sine | random, the quantity that is limited (`limit_kind`) and where the limit comes from. Numbers given here (`limit`, `design_factor`,
    `min_width_oct`, `floor_ratio`) are used but never confirmed; leave them None to take them from the confirmed requirement set."""
    kind: str = "sine"
    limit_kind: str = "force"
    limit: float | None = None
    limit_table: list | None = None                    # [[f Hz, value], …] limit per frequency (log-log), instead of a constant
    design_factor: float | None = None
    min_width_oct: float | None = None
    floor_ratio: float | None = None
    breakpoint_tol_db: float = 0.1
    scope: str = ""                                    # where the limited quantity was evaluated ("probe P"): printed in the verdict, the quantity elsewhere is not checked
    note: str = ""


def _resolve(lim: S4Limits, key: str, override: float | None, name: str, inputs: list[dict], required: bool = True) -> tuple[float | None, bool]:
    v, ok, src = lim.get(key, override)
    inputs.append({"name": name, "key": key, "value": v, "source": src or "not available", "confirmed": ok})
    return v, ok if v is not None else (not required)


def limit_from_table(table: Any, f: np.ndarray, warnings: list[str]) -> np.ndarray:
    """The limit per frequency from a table [[f Hz, value], ...] (log-log between the points). OUTSIDE the range of the table the SMALLEST value of the table applies (the most
    restrictive one - never an extrapolation or the largest value); a note says so."""
    from ..dynamics.psd import loglog_interp
    tab = np.asarray(table, float)
    if tab.ndim != 2 or tab.shape[1] != 2 or len(tab) < 1 or not np.all(np.isfinite(tab)) or np.any(tab <= 0) or np.any(np.diff(tab[:, 0]) <= 0):
        raise NotchError("limit_table must be [[f Hz, value], ...]: finite positive numbers, increasing frequencies")
    out = np.full(f.shape, float(tab[:, 1].min()))
    inside = (f >= tab[0, 0]) & (f <= tab[-1, 0])
    if len(tab) > 1 and inside.any():
        out[inside] = loglog_interp(tab, f[inside])
    elif len(tab) == 1:
        out[:] = float(tab[0, 1])
        inside[:] = True
    if not inside.all():
        warnings.append(f"The limit table covers {tab[0, 0]:g} - {tab[-1, 0]:g} Hz; outside it the smallest value of the table ({float(tab[:, 1].min()):.4g}) is used (most restrictive)")
    return out


def make_notch(f: np.ndarray, level: np.ndarray, response: np.ndarray, spec: NotchSpec, limits: S4Limits | None = None,
               mode_freqs: np.ndarray | None = None, notes: list[str] | None = None,
               table_check: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, require_table_check: bool = False) -> NotchResult:
    """Notched profile for the input `level(f)` (amplitude in g for a sine profile, ASD in g²/Hz for a random one) whose limit quantity is predicted by `response(f)`
    (the amplitude, resp. the PSD of the limited quantity for that very input). The limit, the factor and the rules come from `limits` unless `spec` overrides them.
    `notes` = remarks of the caller about the response (frequencies dropped ...), added to the warnings.
    `table_check(f, level)` = the response (same meaning as `response`) to a candidate input `level` at the frequencies `f`: the exported breakpoint table is re-evaluated with it
    on a grid up to ten times finer than the analysis grid (`fine_grid`), and a table that exceeds the limit by more than `TABLE_TOL` turns a PASS into a FAIL (the deliverable
    would break the limit between the points of the analysis). `require_table_check`: without `table_check` (the response is known at the analysis frequencies only) a PASS
    becomes UNCONFIRMED - the table is not verified."""
    if spec.kind not in KINDS:
        raise NotchError(f"kind must be one of {', '.join(KINDS)}")
    if spec.limit_kind not in LIMIT_KINDS:
        raise NotchError(f"limit_kind must be one of {', '.join(LIMIT_KINDS)}")
    if spec.kind == "sine" and spec.limit_kind == "force_psd":
        raise NotchError("A PSD limit applies to a random profile")
    if spec.kind == "random" and spec.limit_kind != "force_psd":
        raise NotchError("A random profile is notched against a PSD limit (limit_kind 'force_psd'); use a sine profile for amplitude limits")
    lim = limits or S4Limits()
    inputs: list[dict] = []
    f = np.asarray(f, float)
    level = np.asarray(level, float)
    response = np.asarray(response, float)
    if not (len(f) == len(level) == len(response)):
        raise NotchError("f, level and response must have the same length")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(level)) and np.all(np.isfinite(response))) or np.any(level < 0) or np.any(response < 0):
        raise NotchError("The input profile or the predicted quantity has a non-finite or negative value - no notch is computed (no verdict is given for it)")
    if len(f) < 2:
        raise NotchError("A notch needs at least two frequencies - a single point has no profile to notch (no verdict is given)")
    if not (np.any(level > 0) and np.any(response > 0)):
        raise NotchError("The input profile or the predicted quantity is zero at every frequency - there is nothing to notch (no verdict is given)")
    warnings: list[str] = list(notes or [])
    if spec.limit_table is not None:
        L = limit_from_table(spec.limit_table, f, warnings)
        limit_ok = False
        inputs.append({"name": "limit table", "key": LIMIT_KEY_OF[spec.limit_kind], "value": f"{len(spec.limit_table)} points", "source": "entered directly (not confirmed)",
                       "confirmed": False})
        limit_val = float(np.min(L))
    else:
        limit_val, limit_ok = _resolve(lim, LIMIT_KEY_OF[spec.limit_kind], spec.limit, f"limit ({spec.limit_kind})", inputs)
        if limit_val is None or not (math.isfinite(limit_val) and limit_val > 0):
            raise NotchError(f"No limit: give `limit` (finite, positive) or enter {LIMIT_KEY_OF[spec.limit_kind]} in the requirement set")
        L = limit_val
    j_name = "design factor J" + (" (applied as J^2 to the PSD)" if spec.kind == "random" else "")
    j_val, j_ok = _resolve(lim, "notch_design_factor", spec.design_factor, j_name, inputs, required=False)
    if j_val is None:
        j_val, j_ok = 1.0, False
        inputs[-1].update(value=1.0, source="no factor (1.0) - the requirement set has none", confirmed=False)
        warnings.append("No design factor: J = 1 is assumed (the verdict stays unconfirmed)")
    if not (math.isfinite(j_val) and j_val > 0):
        raise NotchError("The design factor must be finite and positive")
    w_val, w_ok = _resolve(lim, "notch_min_width_oct", spec.min_width_oct, "minimum notch width [oct]", inputs, required=False)
    fl_val, fl_ok = _resolve(lim, "notch_floor_ratio", spec.floor_ratio, "notch floor ratio", inputs, required=False)
    if w_val is None:
        warnings.append("No minimum notch width: notches are not widened (the verdict stays unconfirmed)")
    else:
        if not (math.isfinite(w_val) and 0.0 < w_val <= MAX_WIDTH_OCT):
            raise NotchError(f"The minimum notch width must be between 0 and {MAX_WIDTH_OCT:g} octaves")
        span = math.log2(f[-1] / f[0]) if len(f) > 1 and f[0] > 0 else 0.0
        if w_val > span > 0:
            warnings.append(f"The minimum notch width ({w_val:g} oct) exceeds the analysed band ({span:.3g} oct): a notch can only cover the band")
            w_val = span
    if fl_val is None:
        warnings.append("No notch floor: the notch depth is not limited (the verdict stays unconfirmed)")
    j_eff = j_val * j_val if spec.kind == "random" else j_val          # a PSD is the square of an amplitude
    s, bands = notch_scale(f, response, L, j_eff, fl_val, w_val)
    notched = level * s
    q_n = response * s
    notches: list[Notch] = []
    db_factor = 20.0 if spec.kind == "sine" else 10.0
    Lb = np.broadcast_to(np.asarray(L, float), f.shape)
    for b in bands:
        seg = slice(b["i0"], b["i1"] + 1)
        k = b["k"]
        lo, hi = b["f_lo"], b["f_hi"]
        modes = [] if mode_freqs is None else [float(m) for m in np.asarray(mode_freqs, float) if lo <= m <= hi]
        qn = float(q_n[seg].max())
        lim_here = float(Lb[seg].min())
        margin = (lim_here / (j_eff * qn) - 1.0) if qn > 0 else None
        exceeds = margin is not None and margin < -1e-9
        notches.append(Notch(lo, hi, math.sqrt(lo * hi), math.log2(hi / lo) if hi > lo else 0.0, float(f[k]), float(s[k]), db_factor * math.log10(max(float(s[k]), 1e-300)),
                             float(level[k]), float(notched[k]), float(response[seg].max()), qn, lim_here, margin, bool(b["widened"]), bool(b["floor_limited"]), modes,
                             "exceeds_limit" if exceeds else "ok"))
    if w_val is not None:                                      # a notch that touches the edge of the profile cannot be widened beyond it
        for n in notches:
            at_edge = n.f_lo <= f[0] * (1 + 1e-9) or n.f_hi >= f[-1] * (1 - 1e-9)
            if at_edge and n.width_oct < w_val - 1e-9:
                warnings.append(f"The notch {n.f_lo:.4g} - {n.f_hi:.4g} Hz touches the edge of the profile ({f[0]:.4g} - {f[-1]:.4g} Hz): its width {n.width_oct:.3g} oct "
                                f"cannot reach the minimum width {w_val:g} oct inside the band")
    over = float((j_eff * q_n / np.maximum(Lb, 1e-300)).max()) if len(f) else 0.0
    exceeds_any = over > 1.0 + 1e-9
    confirmed = bool(limit_ok and j_ok and w_ok and fl_ok and w_val is not None and fl_val is not None and j_val is not None)
    scope = str(spec.scope or "")
    if not confirmed:
        verdict, text = UNCONFIRMED, "the limit, the design factor or a notch rule is not confirmed (or not defined) in the requirement set - indicative only"
    elif exceeds_any:
        verdict, text = FAIL, "the notched profile still exceeds the limit (the floor or the minimum width prevents a deeper notch)"
    else:
        verdict, text = PASS, "the notched profile keeps the limit with the confirmed rules"
    if scope:
        text += f" - {scope} ONLY: the quantity at other points of the structure is not checked"
    if exceeds_any:
        warnings.append("The notched profile still exceeds the limit at "
                        + ", ".join(f"{n.f_deepest:.4g} Hz" for n in notches if n.status == "exceeds_limit")[:200])
    res = NotchResult(spec.kind, spec.limit_kind, f, level, notched, response, q_n, float(limit_val), float(j_val), notches, verdict, text, inputs, warnings,
                      design_factor_applied=float(j_eff), scope=scope)
    res.breakpoints = breakpoints(f, notched, spec.breakpoint_tol_db, power=spec.kind == "random", unnotched=level)
    if table_check is not None and len(res.breakpoints) >= 2 and len(f) > 1:
        fb = np.array([b[0] for b in res.breakpoints])
        ab = np.maximum(np.array([b[1] for b in res.breakpoints]), 1e-300)
        fine = fine_grid(f)
        a_fine = np.exp(np.interp(np.log(fine), np.log(fb), np.log(ab)))
        q_tab = np.asarray(table_check(fine, a_fine), float)
        if not np.all(np.isfinite(q_tab)):
            raise NotchError("The response of the breakpoint table is not finite - no notch is computed")
        L_fine = limit_from_table(spec.limit_table, fine, []) if spec.limit_table is not None else np.full(fine.shape, float(L))
        res.table_ratio = float((j_eff * q_tab / np.maximum(L_fine, 1e-300)).max())
        if res.table_ratio > 1.0 + TABLE_TOL:
            res.warnings.append(f"The exported breakpoint table exceeds the limit by {100.0 * (res.table_ratio - 1.0):.2f} % between the points of the analysis grid")
            if res.verdict == PASS:
                res.verdict = FAIL
                res.verdict_text = "the dense profile keeps the limit, but the exported breakpoint table exceeds it between the points of the analysis grid"
    elif table_check is None:
        res.warnings.append("The breakpoint table was not re-evaluated between the analysis frequencies (no transfer function available for it)")
        if require_table_check and res.verdict == PASS:
            res.verdict = UNCONFIRMED
            res.verdict_text = ("the limit is kept at the analysis frequencies, but the exported breakpoint table could not be checked between them (the quantity is known at the frequencies "
                                "of the harmonic run only) - no PASS; run the harmonic analysis on a finer frequency grid (more points per resonance) and repeat")
            if scope:
                res.verdict_text += f" - {scope} ONLY"
    return res


# ------------------------------------------------------------------ transfers from other results
def interface_response(si: Any, quantity: str = "force") -> np.ndarray:
    """Predicted peak force [N] / moment [N·mm] of a `dynamics.interface.SineInterface` (for exactly its input)."""
    return np.asarray(si.F_peak if quantity == "force" else si.M_peak, float)


def interface_psd_response(ri: Any) -> np.ndarray:
    """PSD of the resultant interface force [N²/Hz] of a `RandomInterface` (sum of the component PSDs)."""
    return np.asarray(ri.F_psd, float).sum(axis=1)


BASE_KINDS = ("base_acc", "base_disp")


def check_base_only(excitation: Any) -> str | None:
    """The harmonic result of a notching must come from a BASE excitation in ONE direction only (the transfer is then response per unit base acceleration). `excitation` =
    the list of `Excitation` of the harmonic settings of the study; raises `NotchError` for a force excitation, for none, or for several base directions. Returns the unit
    direction as text."""
    items = [e for e in (excitation or []) if getattr(e, "enabled", True)]
    if not items:
        raise NotchError("The harmonic settings have no excitation - run the sine dynamics with a base acceleration first")
    bad = [str(getattr(e, "kind", "?")) for e in items if getattr(e, "kind", None) not in BASE_KINDS]
    if bad:
        raise NotchError("The harmonic result is not a pure base-acceleration run (it has a " + ", ".join(sorted(set(bad))) + " excitation): a notch of a base profile needs the response to the base alone")
    dirs = set()
    for e in items:
        d = getattr(e, "direction", None)
        v = np.array([1.0, 0.0, 0.0]) if d is None else np.asarray(d, float).ravel()
        if v.shape == (1,):
            u = np.zeros(3)
            u[int(v[0]) - 1] = 1.0
            v = u
        n = float(np.linalg.norm(v))
        dirs.add(tuple(np.round(v / n, 9)) if n > 0 else (0.0, 0.0, 0.0))
    if len(dirs) > 1:
        raise NotchError("The harmonic result has base excitations in several directions: notch one direction at a time (run the harmonic analysis with one)")
    return ", ".join(f"{x:g}" for x in next(iter(dirs)))


def harmonic_response_ex(dyn: Any, quantity: str, probe: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """(f, response per unit base acceleration of 1 g, notes, probe) from the harmonic result of a base excitation: quantity 'probe_acc' (absolute acceleration at the probe, g/g)
    or 'stress' (equivalent stress at the probe, MPa/g). The probe is a name of `dyn.harmonic['probes']` (stress: the automatic peak probe if omitted). The response is that
    at the PROBE only. Frequencies at which the base acceleration is zero (the transfer is undefined) are dropped, with a note."""
    from ..dynamics.model import G_MM
    from ..dynamics.response import P_VM, max_over_phase
    h = getattr(dyn, "harmonic", None)
    if h is None:
        raise NotchError("The study has no harmonic result - run the sine dynamics with a base acceleration first")
    f = np.asarray(h["f"], float)
    ab = np.asarray(h["ab"])
    ab_mag = np.sqrt((np.abs(ab) ** 2).sum(axis=1))
    if not np.any(ab_mag > 0):
        raise NotchError("The harmonic result has no base acceleration (force excitation)")
    probes = h.get("probes", {})
    if probe is None:
        probe = next((k for k in probes if probes[k].get("S") is not None and k.startswith("max")), next(iter(probes), None))
    if probe is None or probe not in probes:
        raise NotchError(f"Unknown probe '{probe}' (probes: {', '.join(probes) or 'none'})")
    ser = probes[probe]
    ok = ab_mag > 1e-9 * float(ab_mag.max())
    notes: list[str] = []
    if not ok.all():
        notes.append(f"{int((~ok).sum())} of {len(f)} frequencies of the harmonic result have no base acceleration (the transfer is undefined there): they are left out")
    if quantity == "probe_acc":
        U = np.asarray(ser["U"])
        w = 2.0 * np.pi * f
        A = -(w ** 2)[:, None] * U + ab
        mag = max_over_phase(A.real, -A.imag)
        t = mag[ok] / ab_mag[ok]                                         # dimensionless: acceleration per base acceleration (g per g)
    elif quantity == "stress":
        S = np.asarray(ser["S"])
        sm = max_over_phase(S.real, -S.imag, P_VM)
        t = sm[ok] / (ab_mag[ok] / G_MM)                                 # MPa per g of base acceleration
    else:
        raise NotchError("quantity must be 'probe_acc' or 'stress'")
    if not np.all(np.isfinite(t)):
        raise NotchError("The harmonic response is not finite at some frequencies - no notch is computed")
    return f[ok], t, notes, str(probe)


def harmonic_response(dyn: Any, quantity: str, probe: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(f, response per unit base acceleration of 1 g) - see `harmonic_response_ex`."""
    f, t, _notes, _probe = harmonic_response_ex(dyn, quantity, probe)
    return f, t


def sine_transfer_response(f: np.ndarray, level_g: np.ndarray, transfer_per_g: np.ndarray) -> np.ndarray:
    return np.asarray(level_g, float) * np.asarray(transfer_per_g, float)


# ------------------------------------------------------------------ CSV
def profile_csv(res: NotchResult, name: str = "", dense: bool = False) -> str:
    """CSV text of the notched profile: the simplified breakpoints (default) or the dense analysis grid. Text cells are protected against spreadsheet formulas."""
    from ..export_csv import text_cell
    unit = "g" if res.kind == "sine" else "g2/Hz"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([text_cell("profile"), text_cell(name or f"{res.kind} notched profile"), text_cell("verdict"), text_cell(res.verdict)])
    w.writerow([text_cell("f_Hz"), text_cell(f"notched_{unit}"), text_cell(f"unnotched_{unit}"), text_cell("reduction_dB")])
    if dense:
        red = res.reduction_db
        for fi, a, b, r in zip(res.f, res.notched, res.level, red):
            w.writerow([repr(float(fi)), repr(float(a)), repr(float(b)), f"{float(r):.4f}"])
    else:
        un = np.interp(np.log([p[0] for p in res.breakpoints]), np.log(res.f), res.level) if res.breakpoints else []
        for (fi, a), b in zip(res.breakpoints, un):
            r = (20.0 if res.kind == "sine" else 10.0) * math.log10(max(a, 1e-300) / max(float(b), 1e-300))
            w.writerow([repr(fi), repr(a), repr(float(b)), f"{r:.4f}"])
    return buf.getvalue()


JUSTIFICATION_COLUMNS = ("notch", "f_lo_Hz", "f_hi_Hz", "width_oct", "f_deepest_Hz", "input_unnotched", "input_notched", "reduction_dB", "quantity_unnotched_peak",
                         "quantity_notched_peak", "limit", "margin_remaining", "widened_to_min_width", "floor_limited", "modes_in_band_Hz", "status")


def justification_rows(res: NotchResult) -> list[list[Any]]:
    rows = []
    for i, n in enumerate(res.notches, 1):
        rows.append([i, n.f_lo, n.f_hi, n.width_oct, n.f_deepest, n.input_unnotched, n.input_notched, n.reduction_db, n.quantity_unnotched_peak,
                     n.quantity_notched_peak, n.limit, n.margin_remaining, "yes" if n.widened else "no", "yes" if n.floor_limited else "no",
                     ", ".join(f"{m:.4g}" for m in n.modes_near_Hz), n.status])
    return rows


def justification_csv(res: NotchResult) -> str:
    """The notch justification table as CSV (text cells protected)."""
    from ..export_csv import text_cell
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([text_cell(c) for c in JUSTIFICATION_COLUMNS])
    for r in justification_rows(res):
        w.writerow([text_cell(v) if isinstance(v, str) else ("" if v is None else repr(float(v)) if isinstance(v, float) else v) for v in r])
    return buf.getvalue()

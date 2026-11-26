"""Budgets of the space-structures verification that stage S2 checks (mass, centre of mass, minimum natural frequency) and the quasi-static levels it uses.

Standing rule of the F14 plan: the program ships NO number of a standard or of a launcher. Every budget / level here is a NAMED LIMIT of the requirement set
(`RequirementSet.limits`, the shared table of stages S2 / S3 / S5) that the user enters with document / revision / clause and confirms together with the set.
`install_limit_rows` appends the empty rows of this stage to a requirement set; a check whose limit is missing, has no provenance, or belongs to a set that is not
confirmed still reports its numbers, but its status is `UNCONFIRMED` - never `PASS`. `FAIL` is only given with confirmed inputs.
No Qt and no `fea_core.mcp` import.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .model import CLAUSE_PLACEHOLDER, LimitEntry, RequirementSet, SpaceSettings

PASS, FAIL, UNCONFIRMED, SKIPPED = "PASS", "FAIL", "UNCONFIRMED", "SKIPPED"

# name of the limit -> (unit, what it limits). The names are identifiers; there is no value anywhere in the program.
S2_LIMITS: dict[str, tuple[str, str]] = {
    "qsl_axial_g": ("g", "quasi-static load factor along the launch axis"),
    "qsl_lateral_g": ("g", "quasi-static load factor of the first lateral axis (the second one too unless qsl_lateral_2_g is given)"),
    "qsl_lateral_2_g": ("g", "quasi-static load factor of the second lateral axis (empty = the same as qsl_lateral_g)"),
    "mass_max_kg": ("kg", "largest total mass of the structure"),
    "mass_min_kg": ("kg", "smallest total mass of the structure (empty = no lower bound)"),
    "cog_min_x_mm": ("mm", "smallest x of the centre of mass in the model frame"),
    "cog_max_x_mm": ("mm", "largest x of the centre of mass in the model frame"),
    "cog_min_y_mm": ("mm", "smallest y of the centre of mass in the model frame"),
    "cog_max_y_mm": ("mm", "largest y of the centre of mass in the model frame"),
    "cog_min_z_mm": ("mm", "smallest z of the centre of mass in the model frame"),
    "cog_max_z_mm": ("mm", "largest z of the centre of mass in the model frame"),
    "f_min_axial_hz": ("Hz", "smallest allowed first natural frequency along the launch axis"),
    "f_min_lateral_hz": ("Hz", "smallest allowed first natural frequency along the lateral axes"),
    "f_min_eff_mass_fraction": ("-", "effective-mass fraction (0-1) from which a SINGLE mode counts as significant in a direction: a computed mode below f_min with at least this share fails (empty = every mode counts, the lowest mode of the model decides)"),
}


# value rule of a named limit: "positive" (> 0), "nonnegative" (>= 0) or "fraction" (0 < x <= 1). A value outside its rule is refused by the validation of the
# requirement set and is never usable (`limit_of` gives it no value): a QSL factor of 0 or below, a minimum frequency of 0 or an effective-mass fraction of 0 are
# not requirements but typing errors, and a limit of 0 Hz would let every model PASS.
LIMIT_RULES: dict[str, str] = {
    "qsl_axial_g": "positive", "qsl_lateral_g": "positive", "qsl_lateral_2_g": "positive",
    "mass_max_kg": "positive", "mass_min_kg": "nonnegative",
    "f_min_axial_hz": "positive", "f_min_lateral_hz": "positive", "f_min_eff_mass_fraction": "fraction",
}


def limit_value_problem(key: str, value: float | None) -> str:
    """Why `value` is not acceptable for the named limit `key` (empty = fine or not a limit of this stage)."""
    rule = LIMIT_RULES.get(key)
    if rule is None or value is None:
        return ""
    if not math.isfinite(value):
        return "the value must be a finite number"
    if rule == "positive" and value <= 0:
        return f"the value must be greater than 0 (entered {value:g})"
    if rule == "nonnegative" and value < 0:
        return f"the value must not be negative (entered {value:g})"
    if rule == "fraction" and not 0 < value <= 1:
        return f"the fraction must be in (0, 1] (entered {value:g})"
    return ""


def install_limit_rows(rs: RequirementSet) -> list[str]:
    """Appends the empty rows of the S2 limits that the set does not have yet (value None, clause placeholder). Returns the names added."""
    have = {x.key for x in rs.limits}
    added: list[str] = []
    for key, (unit, what) in S2_LIMITS.items():
        if key not in have:
            rs.limits.append(LimitEntry(key=key, value=None, unit=unit, clause=CLAUSE_PLACEHOLDER, note=what))
            added.append(key)
    return added


def _provenance(x: LimitEntry) -> bool:
    return bool(x.document.strip() and x.revision.strip() and x.clause.strip() and x.clause.strip() != CLAUSE_PLACEHOLDER)


@dataclass
class LimitValue:
    """A limit as the checks see it: its value (None when missing), the provenance, and whether it may decide (the set is confirmed, the row has a value
    and a source)."""
    key: str
    value: float | None
    unit: str = ""
    document: str = ""
    revision: str = ""
    clause: str = ""
    usable: bool = False
    why_not: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "unit": self.unit, "document": self.document, "revision": self.revision, "clause": self.clause,
                "usable": self.usable, "why_not": self.why_not}


def limit_of(space: SpaceSettings | None, key: str) -> LimitValue:
    """The limit `key` of the requirement set of the study. `usable` only when the set (and the structure classes) are confirmed and the row has a finite value
    with document / revision / clause."""
    rs = space.requirement_set if space is not None else None
    if rs is None:
        return LimitValue(key, None, why_not="no requirement set")
    x = next((e for e in rs.limits if e.key == key), None)
    if x is None or x.value is None or not math.isfinite(x.value):
        return LimitValue(key, None, x.unit if x else "", why_not="the value is not entered")
    bad = limit_value_problem(key, float(x.value))
    if bad:
        return LimitValue(key, None, x.unit, x.document, x.revision, x.clause, why_not=f"invalid value: {bad}")
    lv = LimitValue(key, float(x.value), x.unit, x.document, x.revision, x.clause)
    if not _provenance(x):
        lv.why_not = "document / revision / clause are not filled"
    elif space is None or not space.requirement_confirmed():
        lv.why_not = "the requirement set is not confirmed"
    else:
        lv.usable = True
    return lv


@dataclass
class BudgetItem:
    """One checked quantity against one limit."""
    name: str
    value: float | None
    limit: float | None
    unit: str
    kind: str                           # "max" (value <= limit) | "min" (value >= limit)
    status: str
    limit_key: str = ""
    source: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "limit": self.limit, "unit": self.unit, "kind": self.kind, "status": self.status,
                "limit_key": self.limit_key, "source": self.source, "note": self.note}


@dataclass
class BudgetReport:
    items: list[BudgetItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """FAIL when a usable limit is violated; PASS only when every item has a usable limit and holds; UNCONFIRMED otherwise. A SKIPPED item (no limit entered,
        nothing to check) does not count; an item that has a limit but no value is UNCONFIRMED (`judge`), never skipped, so it cannot hide behind a PASS."""
        st = [i.status for i in self.items if i.status != SKIPPED]
        if FAIL in st:
            return FAIL
        if st and all(s == PASS for s in st):
            return PASS
        return UNCONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "items": [i.to_dict() for i in self.items], "notes": list(self.notes)}


def judge(name: str, value: float | None, key: str, space: SpaceSettings | None, kind: str, note: str = "") -> BudgetItem:
    """Compares `value` with the limit `key`: `kind` "max" - value must not exceed it, "min" - must not fall below it."""
    lv = limit_of(space, key)
    unit = lv.unit or S2_LIMITS.get(key, ("", ""))[0]
    src = " / ".join(x for x in (lv.document, lv.revision, lv.clause) if x.strip())
    if value is None or not math.isfinite(value):
        # a limit that is entered but has no value to compare with is UNCONFIRMED (it must stop a PASS); with no limit there is nothing to check
        return BudgetItem(name, None, lv.value, unit, kind, UNCONFIRMED if lv.value is not None else SKIPPED, key, src, note or "no value")
    if lv.value is None:
        return BudgetItem(name, value, None, unit, kind, UNCONFIRMED, key, "", lv.why_not)
    ok = value <= lv.value if kind == "max" else value >= lv.value
    if not lv.usable:
        return BudgetItem(name, value, lv.value, unit, kind, UNCONFIRMED, key, src, lv.why_not or note)
    return BudgetItem(name, value, lv.value, unit, kind, PASS if ok else FAIL, key, src, note)


# ------------------------------------------------------------------ mass, centre of mass
def mass_budget(props: Any, space: SpaceSettings | None) -> BudgetReport:
    """Total mass and centre of mass of `props` (`mass_props.MassProperties`) against the budgets of the requirement set. Only the limits that exist in the
    set are judged (an absent bound is not a check); with no mass limit at all the report says so."""
    rep = BudgetReport()
    rs = space.requirement_set if space is not None else None
    present = {x.key for x in rs.limits if x.value is not None} if rs is not None else set()
    m_kg = props.mass_t * 1000.0
    if "mass_max_kg" in present or "mass_min_kg" not in present:
        rep.items.append(judge("total mass", m_kg, "mass_max_kg", space, "max"))
    if "mass_min_kg" in present:
        rep.items.append(judge("total mass (lower bound)", m_kg, "mass_min_kg", space, "min"))
    for i, ax in enumerate("xyz"):
        for bound, kind in (("min", "min"), ("max", "max")):
            key = f"cog_{bound}_{ax}_mm"
            if key in present:
                rep.items.append(judge(f"centre of mass {ax} ({bound})", float(props.cog[i]), key, space, kind))
    if not any(k.startswith("cog_") for k in present):
        rep.notes.append("no centre-of-mass bound is entered in the requirement set - the centre of mass is only reported")
    if not present & {"mass_max_kg", "mass_min_kg"}:
        rep.notes.append("no mass budget is entered in the requirement set - the mass is only reported")
    return rep


# ------------------------------------------------------------------ minimum natural frequency
# Analysis constant of the frequency check (NOT a requirement of a standard): the share of the effective mass of a direction that the computed
# modes must account for. It equals the usual criterion for reduced mathematical models of ECSS-E-ST-32-11C Table 5-4 (total effective mass above 90 % of the rigid-body mass;
# the table is informative, "usual criteria"); 5.7.3 says that the sum of the effective masses indicates the completeness of a mode set. Below it some mass sits in modes that were not computed, so the check says UNCONFIRMED instead of PASS. The same level limits the mass
# that may sit in modes below f_min which are each below the significance threshold (1 - level): many small modes below f_min must not add up unnoticed.
MODAL_COMPLETENESS = 0.9


def first_mode(freq: Any, frac: Any, threshold: float | None) -> tuple[float | None, int | None]:
    """The first SIGNIFICANT mode of a direction: the lowest-frequency mode whose effective-mass fraction `frac` (per mode, in that direction) reaches `threshold`;
    without a threshold the lowest mode of the model. (frequency [Hz], 0-based index) or (None, None)."""
    n = len(freq)
    if not n:
        return None, None
    order = sorted(range(n), key=lambda i: float(freq[i]))
    for i in order:
        if float(freq[i]) <= 0:
            continue
        if threshold is None or float(frac[i]) >= threshold:
            return float(freq[i]), i
    return None, None


@dataclass
class DirectionModes:
    """What the computed basis says about ONE direction (axis) against ONE minimum frequency `f_min`.
    `lowest`: the lowest significant mode of the basis (frequency, 0-based index, share) or None; `below`: the significant modes under f_min; `cumulative`: the share
    of the effective mass of the direction that the computed modes account for; `mass_below`: the share carried by ALL modes under f_min (significant or not);
    `unknown`: the effective-mass fractions of the direction are not known (no total mass)."""
    axis: str
    lowest: tuple[float, int, float] | None = None
    below: list[tuple[float, int, float]] = field(default_factory=list)
    cumulative: float = 0.0
    mass_below: float = 0.0
    unknown: bool = False


def assess_direction(axis: str, freq: Any, frac: Any, f_min: float | None, thr: float | None) -> DirectionModes:
    """The significant modes of one direction. `thr` is the user-confirmed effective-mass fraction from which a SINGLE mode counts as significant (None = every mode
    with a positive frequency counts: the lowest mode of the model decides). A fraction that is not a finite number (the total mass is unknown) makes the direction
    `unknown`: it is never read as zero."""
    d = DirectionModes(axis)
    pairs = [(float(f), k, float(frac[k])) for k, f in enumerate(freq) if float(f) > 0]
    if any(not math.isfinite(sh) for _f, _k, sh in pairs):
        d.unknown = True
        return d
    pairs.sort()
    d.cumulative = float(sum(sh for _f, _k, sh in pairs))
    for f, k, sh in pairs:
        significant = thr is None or sh >= thr
        if f_min is not None and f < f_min:
            d.mass_below += sh
            if significant:
                d.below.append((f, k, sh))
        if significant and d.lowest is None:
            d.lowest = (f, k, sh)
    return d


def _describe(m: tuple[float, int, float], axis: str) -> str:
    return f"{axis}: mode {m[1] + 1} ({m[0]:.4g} Hz, {m[2] * 100:.1f} % of the effective mass)"


def _frequency_item(name: str, dirs: list[DirectionModes], key: str, space: SpaceSettings | None, f_top: float | None, thr: float | None, env_ok: bool) -> BudgetItem:
    """One frequency item (axial, or lateral over both lateral axes) against the minimum frequency `key`.

    The significance threshold `thr` (the confirmed `f_min_eff_mass_fraction`) decides whether a SINGLE mode counts. Then:
    * FAIL    - a computed mode under f_min is significant in a direction (usable limit, confirmed environment); the item value is the lowest significant mode.
    * UNCONFIRMED (with the reason) - the basis does not reach f_min (a mode under it may not have been computed), OR the computed modes of a direction account for
                less than `MODAL_COMPLETENESS` of its effective mass, OR the modes under f_min that are not significant one by one carry more than
                1 - MODAL_COMPLETENESS of it, OR the effective masses are unknown, OR the limit / environment is not usable.
    * PASS    - only when none of the above holds: the basis covers f_min and no significant mode lies under it.
    A bound inferred from the highest computed frequency (the earlier rule) is NOT used: with split mass the first mode may lie far below it."""
    lv = limit_of(space, key)
    src = " / ".join(x for x in (lv.document, lv.revision, lv.clause) if x.strip())
    unit = lv.unit or S2_LIMITS.get(key, ("", ""))[0]
    cands = [(d.lowest, d.axis) for d in dirs if d.lowest is not None]
    low: tuple[float, int, float] | None = None
    low_axis = ""
    for m, ax in cands:
        if m is not None and (low is None or m[0] < low[0]):
            low, low_axis = m, ax
    shown = f"{name}, mode {low[1] + 1} ({low_axis})" if low is not None else f"{name} (no significant mode computed)"
    value = low[0] if low is not None else None
    fmin = lv.value
    notes: list[str] = []
    if low is not None:
        notes.append("lowest significant mode: " + "; ".join(_describe(d.lowest, d.axis) for d in dirs if d.lowest is not None))
    elif thr is not None:
        notes.append(f"no computed mode reaches the effective-mass fraction {thr:g} in this direction")
    if fmin is None:
        return BudgetItem(shown, value, None, unit, "min", UNCONFIRMED if value is not None else SKIPPED, key, src, "; ".join(notes + [lv.why_not]))
    below = [(m, d.axis) for d in dirs for m in d.below]
    if below:
        notes.append(f"significant mode(s) BELOW f_min {fmin:g} Hz: " + "; ".join(_describe(m, ax) for m, ax in sorted(below)[:6]))
        status = FAIL if lv.usable else UNCONFIRMED
        if not lv.usable:
            notes.append(lv.why_not)
    else:
        problems: list[str] = []
        if any(d.unknown for d in dirs):
            problems.append("the effective-mass fractions are unknown (no total mass) - the modes cannot be classified")
        if f_top is None or f_top < fmin:
            problems.append((f"the computed basis ends at {f_top:g} Hz" if f_top is not None else "no mode with a positive frequency was computed")
                            + f", BELOW f_min {fmin:g} Hz - a mode under f_min may not have been computed")
        for d in dirs:
            if d.unknown:
                continue
            if d.cumulative < MODAL_COMPLETENESS:
                problems.append(f"{d.axis}: the computed modes account for {d.cumulative * 100:.1f} % of the effective mass (< {MODAL_COMPLETENESS * 100:g} %) - "
                                f"compute more modes")
            if d.mass_below > 1.0 - MODAL_COMPLETENESS:
                problems.append(f"{d.axis}: modes under f_min {fmin:g} Hz carry {d.mass_below * 100:.1f} % of the effective mass without one of them reaching "
                                + (f"the fraction {thr:g}" if thr is not None else "significance") + " - the split mass cannot be judged by one mode")
        if problems:
            status = UNCONFIRMED
            notes += problems
        elif not lv.usable:
            status = UNCONFIRMED
            notes.append(lv.why_not)
        else:
            status = PASS
            notes.append(f"the computed basis reaches {f_top:g} Hz >= f_min {fmin:g} Hz and no significant mode lies under f_min")
    it = BudgetItem(shown, value, fmin, unit, "min", status, key, src, "; ".join(n for n in notes if n))
    if not env_ok and it.status in (PASS, FAIL):
        it.status = UNCONFIRMED
        it.note = (it.note + "; " if it.note else "") + "the launch environment (launch axis) is not confirmed - the axial / lateral split is not a verified input"
    return it


def frequency_budget(freq: Any, eff_frac: Any, launch_axis: str, space: SpaceSettings | None, env_confirmed: bool | None = None) -> BudgetReport:
    """f_1 >= f_min along the launch axis (axial) and along the lateral axes. freq (K,) [Hz], eff_frac (K, 3) effective-mass fractions of the modes in x, y, z (NaN =
    unknown). The confirmed `f_min_eff_mass_fraction` is the significance threshold of a SINGLE mode; the rule of PASS / FAIL / UNCONFIRMED is at `_frequency_item`.
    The launch axis comes from the environments, so PASS / FAIL need the environments to be confirmed (`env_confirmed`; None = read from `space`). The basis is taken to
    hold the LOWEST modes of the model (a window that skips the low modes is not a basis for this check)."""
    from .environments import environment_confirmed
    rep = BudgetReport()
    if launch_axis not in ("x", "y", "z"):
        rep.notes.append("the launch axis is not set (environments) - the axial and lateral frequencies cannot be separated")
        return rep
    env_ok = environment_confirmed(space) if env_confirmed is None else bool(env_confirmed)
    if not env_ok:
        rep.notes.append("the launch environments are not confirmed: the launch axis decides which limit applies, so no frequency item can be PASS or FAIL")
    pos = [float(f) for f in freq if float(f) > 0]
    f_top = max(pos) if pos else None
    thr_lv = limit_of(space, "f_min_eff_mass_fraction")
    thr = thr_lv.value if thr_lv.usable else None
    if thr is None:
        rep.notes.append("every mode counts as significant (no confirmed effective-mass fraction): a computed mode under f_min fails the check"
                         if thr_lv.value is None else
                         f"the effective-mass fraction {thr_lv.value:g} is not usable ({thr_lv.why_not}) - every mode counts as significant")
    ia = "xyz".index(launch_axis)
    lat = [i for i in range(3) if i != ia]

    def assess(i: int, key: str) -> DirectionModes:
        col = [float(row[i]) for row in eff_frac] if len(eff_frac) else []
        return assess_direction("xyz"[i], freq, col, limit_of(space, key).value, thr)

    rep.items.append(_frequency_item(f"first axial natural frequency ({launch_axis})", [assess(ia, "f_min_axial_hz")], "f_min_axial_hz", space, f_top, thr, env_ok))
    rep.items.append(_frequency_item("first lateral natural frequency", [assess(i, "f_min_lateral_hz") for i in lat], "f_min_lateral_hz", space, f_top, thr, env_ok))
    return rep


def modal_data_of(results: Any) -> tuple[Any, Any, str] | None:
    """(frequencies [Hz], effective-mass fractions (K, 3), source) from the results: the modal result set with effective masses, else the dynamics basis."""
    import numpy as np
    for ms in getattr(results, "modes", None) or []:
        if getattr(ms, "kind", "") == "modal" and ms.eff_mass and ms.total_mass:
            tot = np.asarray(ms.total_mass, float)[:3]
            frac = np.asarray(ms.eff_mass, float)[:, :3] / np.where(tot > 0, tot, np.nan)         # an axis without a total mass is UNKNOWN (NaN), never divided by 1
            return np.asarray(ms.values, float), frac, f"natural frequencies '{ms.name}' ({ms.engine})"
    dy = getattr(results, "dynamics", None)
    if dy is not None and getattr(dy, "K", 0):
        tot = float(getattr(dy, "model_mass", 0.0)) or 0.0
        em = np.asarray(dy.eff_mass, float)[:, :3]
        return np.asarray(dy.freq, float), em / (tot if tot > 0 else np.nan), "modal basis of the dynamics"
    return None

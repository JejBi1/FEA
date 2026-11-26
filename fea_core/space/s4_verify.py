"""Is a stored S4 result still the answer for the present study? (review finding C2, re-check findings 5 and 6)

A verdict in `Study.space.extra["s4"]` was true for the study, the requirement set and the allowables of the time of the calculation (`s4_store.basis_of`). A project file can be
edited by hand or the limits changed afterwards, so the report never prints a stored verdict as if it were current:

  * no basis in the record (an older or forged file)            -> UNCONFIRMED_REQUIREMENTS, never PASS / FAIL
  * the basis differs from the present study                    -> OUTDATED ("outdated - recompute (what changed)")
  * the basis matches                                           -> bolts and PCB are RECOMPUTED from the stored specification against the present requirement set, rails: the
                                                                   margins are recomputed from the stored forces against the present limits. `Shown.data` is the recomputed
                                                                   record: THE REPORT PRINTS ITS ROWS, never the stored numbers under a recomputed verdict. When the recomputation
                                                                   differs from the stored record (a hand edit, or a program update that changed a formula) `reproduced` is False and
                                                                   the text says "the stored result was not reproduced - recomputed values shown". A record whose stored inputs are
                                                                   missing (spec deleted, rails without `effective_N` or without rows) gives NO verdict: it cannot be recomputed.
  * notching and interface records depend on a modal analysis of the study and cannot be recomputed here: they are judged by the digests alone, and `Shown.note` says so -
    "stored result, not recomputed - tamper-evident only (staleness), not a signature". The digests are not a signature: whoever edits a stored number AND the basis of a
    notch or an interface record (the basis is a plain digest of the present study, computable by anybody) gets the stored verdict printed with that note.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

PASS, FAIL, UNCONFIRMED, OUTDATED = "PASS", "FAIL", "UNCONFIRMED_REQUIREMENTS", "OUTDATED"
OUTDATED_TEXT = "outdated - recompute"


RECOMPUTED = ("bolts", "pcb", "rails")
NOT_RECOMPUTED_NOTE = "stored result, not recomputed - tamper-evident only (staleness), not a signature"
RAILS_NOTE = "the forces of the supports are stored values (they need the solved results); only the margins are recomputed against the present limits"
PCB_NOTE = "the RMS displacement of the board is a stored input (the modal analysis is not repeated); everything else is recomputed from the stored inputs and the present requirement set"
CANNOT_RECOMPUTE = "no verdict: the stored inputs of this result are missing or damaged - it cannot be recomputed"
NOT_REPRODUCED = "the stored result was not reproduced - recomputed values shown"


@dataclass
class Shown:
    verdict: str          # PASS | FAIL | UNCONFIRMED_REQUIREMENTS | OUTDATED
    text: str             # the verdict text (the recomputed one where the record was recomputed), or why no verdict is shown
    fresh: bool           # the basis matches the present study (the numbers of the record are those of the present state)
    data: dict | None = None        # the RECOMPUTED record (bolts, PCB, rails): the numbers to print instead of the stored ones; None = print the stored record
    reproduced: bool | None = None  # True / False = the recomputation gave the stored numbers / did not; None = the record is not recomputed
    note: str = ""                  # what the reader needs to know about how far the verdict is checked

    @property
    def label(self) -> str:
        """The text that a report prints in place of the verdict."""
        return OUTDATED_TEXT if self.verdict == OUTDATED else self.verdict


def _norm(v: Any) -> str:
    return v if v in (PASS, FAIL) else UNCONFIRMED


_CTX: ContextVar[dict | None] = ContextVar("fea_s4_judging", default=None)    # per thread / task: two blocks that overlap in time (two reports at once) never see each other


@contextmanager
def judging(study: Any) -> Iterator[None]:
    """Inside the block the present basis of `study` is computed once (a report judges many records of the same study), and `shown` judges a record once (the recomputation
    of a bolt or a PCB record is the cost of a verdict; a report prints it and then summarises it); `study` and its records must not change inside. A block for the study that
    is being judged already is the same block: the basis and the verdicts stay."""
    from .s4_store import basis_of
    cur = _CTX.get()
    if cur is not None and cur["study"] is study:
        yield
        return
    token = _CTX.set({"study": study, "basis": basis_of(study), "shown": {}})
    try:
        yield
    finally:
        _CTX.reset(token)                                                  # back to exactly what this thread / task had before the block (nesting, any order of exit)


def mismatch(study: Any, rec: dict) -> list[str]:
    """What differs between the basis of the record and the present study (['no basis'] when the record has none)."""
    from .s4_store import basis_of
    b = rec.get("basis") if isinstance(rec, dict) else None
    if not isinstance(b, dict) or not b.get("study") or "req" not in b:
        return ["no basis stored"]
    ctx = _CTX.get()
    cur = ctx["basis"] if ctx is not None and ctx["study"] is study else basis_of(study)
    out = []
    if b.get("req") != cur["req"]:
        out.append("the requirement set changed")
    if bool(b.get("confirmed")) != cur["confirmed"]:
        out.append("the confirmation of the requirement set changed")
    if b.get("allow") != cur["allow"]:
        out.append("the material allowables changed")
    if b.get("study") != cur["study"]:
        out.append("the study changed")
    return out


def _recompute(study: Any, section: str, rec: dict) -> dict | None:
    """The record recomputed from the stored inputs against the present requirement set (a dictionary with `verdict`, `verdict_text` and the numbers), or None when the stored
    inputs are not there (the record cannot be recomputed). Exceptions of the recomputation are the caller's business."""
    from . import s4_run
    if section == "bolts":
        sp = rec.get("spec")
        if not isinstance(sp, dict) or not isinstance(sp.get("raw"), dict):
            return None
        return s4_run.bolts_report(study, [sp["raw"]], sp.get("bolt_material"), sp.get("part_material"), sp.get("bearing_allowable") or "Fu")["bolts"][0]
    if section == "pcb":
        sp = rec.get("spec")
        if not isinstance(sp, dict) or not sp:
            return None
        return s4_run.pcb_report(study, [dict(sp)])["components"][0]
    if section == "rails":
        return _rails_recompute(study, rec)
    return None


def _rails_recompute(study: Any, rec: dict) -> dict | None:
    """The rail record with the margins and the verdict recomputed from the stored forces (effective force of every support, deflections) against the PRESENT limits; None
    when a support row has no effective force or the record has no rows at all (no verdict can be given)."""
    from .s4_limits import S4Limits
    sup, defl = rec.get("supports"), rec.get("deflection") or []
    if not isinstance(sup, list) or not isinstance(defl, list) or (not sup and not defl):
        return None
    try:
        sup = [dict(x) for x in sup]
        defl = [dict(x) for x in defl]
        for x in sup:
            float(x["effective_N"])
            x["kind"] = str(x["kind"])
        for x in defl:
            float(x["max_lateral_deflection_mm"])
    except (KeyError, TypeError, ValueError):
        return None
    lim = S4Limits.from_requirement_set(getattr(study, "space", None))
    rl, rl_ok, _ = lim.get("rail_reaction_limit_N")
    al, al_ok, _ = lim.get("rail_axial_reaction_limit_N")
    dl, dl_ok, _ = lim.get("rail_lateral_deflection_limit_mm")
    out = dict(rec)
    mos: list[float] = []
    unconfirmed = False
    for x in sup:
        limit, ok = (rl, rl_ok) if x["kind"] == "wall" else (al, al_ok)
        e = abs(float(x["effective_N"]))
        x["limit_N"] = limit
        x["mos"] = None if limit is None else (math.inf if e <= 0 else limit / e - 1.0)
        if limit is None or not ok:
            unconfirmed = True
        elif x["mos"] is not None:
            mos.append(x["mos"])
    for x in defl:
        d = float(x["max_lateral_deflection_mm"])
        x["limit_mm"] = dl
        x["mos"] = None if dl is None else (math.inf if d <= 0 else dl / d - 1.0)
        if dl is None or not dl_ok:
            unconfirmed = True
        elif x["mos"] is not None:
            mos.append(x["mos"])
    out["supports"], out["deflection"] = sup, defl
    if rec.get("verdict") not in (PASS, FAIL) or unconfirmed or not mos:
        out["verdict"], out["verdict_text"] = UNCONFIRMED, "a limit is not confirmed in the present requirement set (or the stored result had no verdict) - indicative only"
    elif min(mos) < 0:
        out["verdict"], out["verdict_text"] = FAIL, "a limit is exceeded"
    else:
        out["verdict"], out["verdict_text"] = PASS, "all rail reactions and deflections are within the confirmed limits"
    return out


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def _close(a: Any, b: Any, rel: float = 1e-9) -> bool:
    """Two stored numbers equal (None / "inf" / "nan" strings compared as values)."""
    if a is None or b is None:
        return a is None and b is None
    fa, fb = _f(a), _f(b)
    if fa is None or fb is None:
        return str(a) == str(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return abs(fa - fb) <= rel * max(abs(fa), abs(fb), 1e-300)


def _reproduced(section: str, stored: dict, again: dict) -> bool:
    """Do the recomputed numbers equal the stored ones (also after a program update that changed a formula)?"""
    if _norm(stored.get("verdict")) != _norm(again.get("verdict")):
        return False
    if section == "bolts":
        a, b = stored.get("margins"), again.get("margins")
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not isinstance(x, dict) or x.get("check") != y.get("check") or x.get("status") != y.get("status") or not all(_close(x.get(k), y.get(k)) for k in ("mos", "demand", "capacity")):
                return False
        return all(_close(stored.get(k), again.get(k)) for k in ("F_pmin_N", "F_pmax_N", "F_bolt_max_N", "F_clamp_min_N"))
    if section == "pcb":
        keys = ("MoS", "Z_allow_mm", "Z_sigma_mm", "Z_nsigma_mm", "damage_narrow_band", "damage_limit")
        da, db = stored.get("damage_three_band") or {}, again.get("damage_three_band") or {}
        return all(_close(stored.get(k), again.get(k)) for k in keys) and _close(da.get("damage"), db.get("damage"))
    if section == "rails":
        for key, fields in (("supports", ("effective_N", "limit_N", "mos")), ("deflection", ("max_lateral_deflection_mm", "limit_mm", "mos"))):
            a, b = stored.get(key) or [], again.get(key) or []
            if len(a) != len(b) or not all(isinstance(x, dict) and all(_close(x.get(k), y.get(k)) for k in fields) for x, y in zip(a, b)):
                return False
        return True
    return True


def shown(study: Any, section: str, rec: Any) -> Shown:
    """The verdict of a stored result as it may be printed now (see the module docstring). Inside `judging(study)` the verdict of one record is computed once."""
    ctx = _CTX.get()
    memo = ctx["shown"] if ctx is not None and ctx["study"] is study else None
    if memo is None:
        return _shown(study, section, rec)
    hit = memo.get((section, id(rec)))
    if hit is not None and hit[0] is rec:                                    # the record itself is kept in the memo: its id cannot be reused by another object
        return hit[1]
    sh = _shown(study, section, rec)
    memo[(section, id(rec))] = (rec, sh)
    return sh


def _shown(study: Any, section: str, rec: Any) -> Shown:
    if not isinstance(rec, dict):
        return Shown(UNCONFIRMED, "no stored result", False)
    stored = _norm(rec.get("verdict"))
    text = str(rec.get("verdict_text", ""))
    why = mismatch(study, rec)
    if why == ["no basis stored"]:
        return Shown(UNCONFIRMED, "not verified: the stored result carries no basis (an older or hand-edited project file) - recompute", False)
    if why:
        return Shown(OUTDATED, f"{OUTDATED_TEXT} ({'; '.join(why)})", False)
    cur_confirmed = bool(getattr(getattr(study, "space", None), "requirement_confirmed", lambda: False)())
    if section not in RECOMPUTED:                                          # notching, interface: they depend on a modal analysis - judged by the digests alone
        v = stored
        if v == PASS and not cur_confirmed:
            v, text = UNCONFIRMED, "the requirement set is not confirmed - no PASS"
        return Shown(v, text, True, None, None, NOT_RECOMPUTED_NOTE)
    try:
        again = _recompute(study, section, rec)
    except Exception as e:  # noqa: BLE001
        return Shown(OUTDATED, f"{OUTDATED_TEXT} (the stored inputs cannot be recomputed: {e})", False)
    if again is None:
        return Shown(UNCONFIRMED, CANNOT_RECOMPUTE, False)
    v = _norm(again.get("verdict"))
    same = _reproduced(section, rec, again)
    text = str(again.get("verdict_text", "")) if not same else text
    if not same:
        text = f"{NOT_REPRODUCED}: recomputed from the stored inputs: {v} (the stored result said {stored}); {text}"
    if v == PASS and not cur_confirmed:
        v, text = UNCONFIRMED, "the requirement set is not confirmed - no PASS"
    note = {"rails": RAILS_NOTE, "pcb": PCB_NOTE}.get(section, "")
    return Shown(v, text, True, again, same, note)

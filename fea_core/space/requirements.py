"""Requirement set: the empty ECSS template, validation, confirmation, JSON / CSV import and export, material allowables.

The template names the fields, units and clause placeholders; every number stays `None` until the user enters it, and the set counts as
confirmed only after the user confirmed it explicitly (`confirm`). Nothing in this module is a standard value.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
from dataclasses import dataclass

from .model import (ALLOWABLE_KEYS, BUCKLING_ALLOWABLE, CHECKS, CLAUSE_PLACEHOLDER, DEFAULT_SCOPE_DIGEST, FACTOR_KEYS, STRESS_MEASURES,
                    STRUCTURE_CLASSES, THERMAL_HANDLING, VERIFICATION_APPROACHES, FactorEntry, MarginRule, MaterialAllowable,
                    RequirementError, RequirementSet, SpaceSettings)

TEMPLATE_STANDARD = "ECSS-E-ST-32-10C"          # identifiers of the document of the template (edition and clauses are confirmed by the user)
TEMPLATE_REVISION = ""                          # the edition of the document is entered (and checked) by the user, never taken from the template
MAX_JSON_BYTES = 2 << 20                        # bounds of the importers (a requirement set is a few kB)
MAX_FACTORS, MAX_RULES, MAX_LIMITS = 2000, 200, 500
CSV_COLUMNS = ("key", "structure_class", "approach", "value", "document", "revision", "clause", "note")


@dataclass
class Issue:
    level: str                    # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


# ------------------------------------------------------------------ template
def ecss_32_10c_template() -> RequirementSet:
    """Empty requirement set of ECSS-E-ST-32-10C: every factor row (factor × structure class × verification approach) with value `None`
    and the clause placeholder, one margin rule per check of each structure class with the allowable it is compared with (the user
    chooses the stress measure, the factors that form the design factor and the thermal handling). Composite structures have no rule:
    the stress-based check of composites is not implemented."""
    rs = RequirementSet(name="", standard=TEMPLATE_STANDARD, revision=TEMPLATE_REVISION)
    for cls in STRUCTURE_CLASSES:
        for key in FACTOR_KEYS:
            for approach in VERIFICATION_APPROACHES:
                rs.factors.append(FactorEntry(key=key, structure_class=cls, approach=approach, value=None, document=TEMPLATE_STANDARD,
                                              revision=TEMPLATE_REVISION, clause=CLAUSE_PLACEHOLDER))
    for cls, check, allowable in (("metallic", "yield", "Fy"), ("metallic", "ultimate", "Fu"),
                                  ("joints_inserts", "yield", "Fy"), ("joints_inserts", "ultimate", "Fu"),
                                  ("buckling", "buckling", BUCKLING_ALLOWABLE)):
        rs.rules.append(MarginRule(check=check, structure_class=cls, allowable=allowable, document=TEMPLATE_STANDARD,
                                   revision=TEMPLATE_REVISION, clause=CLAUSE_PLACEHOLDER))
    return rs


# ------------------------------------------------------------------ resolving a rule
@dataclass
class ResolvedRule:
    """The design factors of a rule for the approach of the set: `factor` (mechanical), `thermal_factor` (the thermal part of a
    combined case; None = not decided / not resolvable), the factor rows used and what is missing."""
    rule: MarginRule
    factor: float | None
    thermal_factor: float | None
    used: list[FactorEntry]
    missing: list[str]

    @property
    def complete(self) -> bool:
        return not self.missing


def _product(rs: RequirementSet, rule: MarginRule, keys: list[str], used: list[FactorEntry], missing: list[str], label: str) -> float | None:
    if not keys:
        missing.append(f"{label}: no factor is listed")
        return None
    out = 1.0
    ok = True
    for k in keys:
        e = rs.entry(k, rule.structure_class)
        v = None if e is None else e.value
        if e is None or v is None or not math.isfinite(v):
            missing.append(f"{label}: value of {k} ({rule.structure_class}, {rs.verification_approach or 'no approach'}) is not entered")
            ok = False
            continue
        used.append(e)
        out *= v
    return out if ok else None


def resolve_rule(rs: RequirementSet, rule: MarginRule) -> ResolvedRule:
    """Design factor(s) of a rule: product of the values of `rule.factors` (taken for the structure class and the approach of the set).
    A rule is incomplete while its allowable, stress measure, factors, thermal handling or provenance is missing."""
    missing: list[str] = []
    used: list[FactorEntry] = []
    if not rs.verification_approach:
        missing.append("the verification approach is not selected")
    label = f"{rule.check} / {rule.structure_class}"
    if rule.check == "buckling":
        if rule.allowable != BUCKLING_ALLOWABLE:
            missing.append(f"{label}: the allowable of a buckling check is the buckling load factor")
    else:
        if rule.allowable not in ALLOWABLE_KEYS:
            missing.append(f"{label}: the allowable is not chosen")
        if rule.stress_measure not in STRESS_MEASURES:
            missing.append(f"{label}: the stress measure is not chosen")
        if rule.thermal_handling not in THERMAL_HANDLING[1:]:
            missing.append(f"{label}: the thermal handling is not chosen")
    factor = _product(rs, rule, rule.factors, used, missing, label)
    thermal: float | None = None
    if rule.check != "buckling":
        if rule.thermal_handling == "same_as_mechanical":
            thermal = factor
        elif rule.thermal_handling == "none":
            thermal = 1.0
        elif rule.thermal_handling == "separate":
            thermal = _product(rs, rule, rule.thermal_factors, used, missing, f"{label} (thermal)")
    if not rule.provenance_ok:
        missing.append(f"{label}: document / revision / clause of the rule are not filled")
    for e in used:
        if not e.provenance_ok:
            missing.append(f"{e.key} ({e.structure_class}, {e.approach or 'any approach'}): document / revision / clause are not filled")
    return ResolvedRule(rule, factor, thermal, used, list(dict.fromkeys(missing)))


def rule_in_use(rule: MarginRule) -> bool:
    """A rule that the user has started to define (any factor listed) - it must then be complete to confirm the set."""
    return bool(rule.factors or rule.thermal_factors)


# ------------------------------------------------------------------ validation
def validate(rs: RequirementSet) -> list[Issue]:
    """Structural problems of a requirement set (errors) and things still to fill (warnings)."""
    out: list[Issue] = []

    def err(where: str, msg: str) -> None:
        out.append(Issue("error", where, msg))

    def warn(where: str, msg: str) -> None:
        out.append(Issue("warning", where, msg))

    if rs.verification_approach and rs.verification_approach not in VERIFICATION_APPROACHES:
        err("verification_approach", f"must be one of {', '.join(VERIFICATION_APPROACHES)}")
    if not rs.verification_approach:
        warn("verification_approach", "not selected")
    if not rs.standard.strip():
        warn("standard", "the standard is not named")
    seen: set[tuple[str, str, str]] = set()
    for f in rs.factors:
        where = f"factor {f.key}/{f.structure_class}/{f.approach or 'any'}"
        if not f.key.strip():
            err(where, "the factor name is empty")
        if f.structure_class not in STRUCTURE_CLASSES:
            err(where, f"structure class must be one of {', '.join(STRUCTURE_CLASSES)}")
        if f.approach and f.approach not in VERIFICATION_APPROACHES:
            err(where, f"approach must be empty or one of {', '.join(VERIFICATION_APPROACHES)}")
        if f.value is not None and (not math.isfinite(f.value) or f.value <= 0):
            err(where, "the value must be a positive number")
        k = (f.key, f.structure_class, f.approach)
        if k in seen:
            err(where, "duplicate row")
        seen.add(k)
        if f.value is not None and not f.provenance_ok:
            warn(where, "document / revision / clause are not filled")
    seen_rules: set[tuple[str, str]] = set()
    for r in rs.rules:
        where = f"rule {r.check}/{r.structure_class}"
        if r.check not in CHECKS:
            err(where, f"the check must be one of {', '.join(CHECKS)}")
        if r.structure_class not in STRUCTURE_CLASSES:
            err(where, f"structure class must be one of {', '.join(STRUCTURE_CLASSES)}")
        if r.allowable and r.allowable not in ALLOWABLE_KEYS and r.allowable != BUCKLING_ALLOWABLE:
            err(where, f"unknown allowable '{r.allowable}'")
        if r.stress_measure and r.stress_measure not in STRESS_MEASURES:
            err(where, f"unknown stress measure '{r.stress_measure}'")
        if r.thermal_handling not in THERMAL_HANDLING:
            err(where, f"unknown thermal handling '{r.thermal_handling}'")
        if r.thermal_handling == "separate" and not r.thermal_factors:
            warn(where, "thermal handling 'separate' needs thermal factors")
        for k in list(r.factors) + list(r.thermal_factors):
            if not any(f.key == k and f.structure_class == r.structure_class for f in rs.factors):
                err(where, f"factor '{k}' is not in the factor table for the class {r.structure_class}")
        key = (r.check, r.structure_class)
        if key in seen_rules:
            err(where, "duplicate rule")
        seen_rules.add(key)
    keys: set[str] = set()
    for x in rs.limits:
        where = f"limit {x.key}"
        if not x.key.strip():
            err(where, "the name is empty")
        if x.key in keys:
            err(where, "duplicate limit")
        keys.add(x.key)
        if x.value is not None and not math.isfinite(x.value):
            err(where, "the value must be a number")
        elif x.not_applicable and x.value is not None:
            err(where, "a limit declared not applicable must not carry a value")
        else:
            from .budgets import limit_value_problem        # the value rule of a limit of stage S2 (a QSL factor / f_min of 0 or below, a fraction outside (0, 1])
            bad = limit_value_problem(x.key, x.value)
            if bad:
                err(where, bad)
    if len(rs.factors) > MAX_FACTORS or len(rs.rules) > MAX_RULES or len(rs.limits) > MAX_LIMITS:
        err("size", "the requirement set is too large")
    return out


def confirmation_problems(rs: RequirementSet, by: str) -> list[str]:
    """What stops the set from being confirmed now (empty = it can be confirmed)."""
    msgs = [str(i) for i in validate(rs) if i.level == "error"]
    if not by.strip():
        msgs.append("the name of the person who confirms is missing")
    if not rs.verification_approach:
        msgs.append("the verification approach is not selected")
    used = [r for r in rs.rules if rule_in_use(r)]
    if not used:
        msgs.append("no margin rule is defined (list the factors of at least one rule)")
    for r in used:
        msgs += resolve_rule(rs, r).missing
    for r in rs.rules:
        if r.not_applicable and not r.provenance_ok:
            msgs.append(f"rule {r.check}/{r.structure_class}: declared not applicable, but document / revision / clause are not filled")
    for f in rs.factors:
        if f.value is not None and not f.provenance_ok:
            msgs.append(f"factor {f.key}/{f.structure_class}/{f.approach or 'any'}: document / revision / clause are not filled")
    for x in rs.limits:
        if x.not_applicable and not x.provenance_ok:
            msgs.append(f"limit {x.key}: declared not applicable, but document / revision / clause are not filled")
    return list(dict.fromkeys(msgs))


def confirm(rs: RequirementSet, by: str, date: str | None = None, via: str = "", scope: SpaceSettings | None = None) -> None:
    """The user confirms the requirement set (after checking every value against the source document). Raises `RequirementError`
    with the reasons when it cannot be confirmed. Any later edit of the content withdraws the confirmation, and so does a change of the
    structure classes of the parts (`scope` = the space settings whose classes are confirmed together with the set; without it the
    default classes - every part metallic - are confirmed)."""
    problems = confirmation_problems(rs, by)
    if problems:
        raise RequirementError("The requirement set cannot be confirmed: " + "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else ""),
                               problems)
    rs.confirmed, rs.confirmed_by = True, by.strip()
    rs.date = date or _dt.date.today().isoformat()
    rs.confirmed_via = via
    rs.scope_digest = scope.scope_digest() if scope is not None else DEFAULT_SCOPE_DIGEST
    rs.confirmed_digest = rs.content_digest()


# ------------------------------------------------------------------ import / export
def to_json(rs: RequirementSet) -> str:
    return json.dumps({"format": "fea-studio-requirement-set", "version": 1, **rs.to_dict()}, indent=2, ensure_ascii=False)


def from_json(text: str) -> RequirementSet:
    """Requirement set from JSON text (bounded). An imported set is NOT confirmed: whoever uses it confirms it again."""
    if len(text.encode("utf-8", "replace")) > MAX_JSON_BYTES:
        raise RequirementError("The requirement set file is too large")
    try:
        d = json.loads(text)
    except ValueError as e:
        raise RequirementError(f"Not a JSON file: {e}") from None
    if not isinstance(d, dict):
        raise RequirementError("A requirement set is a JSON object")
    for k, cap in (("factors", MAX_FACTORS), ("rules", MAX_RULES), ("limits", MAX_LIMITS)):
        v = d.get(k)
        if v is not None and (not isinstance(v, list) or len(v) > cap or not all(isinstance(x, dict) for x in v)):
            raise RequirementError(f"'{k}' must be a list of at most {cap} objects")
    try:
        rs = RequirementSet.from_dict(d)
    except (ValueError, TypeError, AttributeError) as e:
        raise RequirementError(f"The requirement set could not be read: {e}") from None
    for x in rs.limits:
        x.set_via = "import" if x.not_applicable else ""           # a file cannot claim that a flag was entered in the GUI
    rs.withdraw_confirmation()
    return rs


def factors_to_csv(rs: RequirementSet) -> str:
    """The factor table as CSV (text cells protected against spreadsheet formulas)."""
    from ..export_csv import text_cell
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for f in rs.factors:
        w.writerow([text_cell(f.key), text_cell(f.structure_class), text_cell(f.approach), "" if f.value is None else repr(f.value),
                    text_cell(f.document), text_cell(f.revision), text_cell(f.clause), text_cell(f.note)])
    return buf.getvalue()


def factors_from_csv(text: str) -> list[FactorEntry]:
    """Factor rows from CSV (the columns of `factors_to_csv`; bounded). The caller replaces `RequirementSet.factors` by the result."""
    if len(text.encode("utf-8", "replace")) > MAX_JSON_BYTES:
        raise RequirementError("The CSV file is too large")
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) > MAX_FACTORS:
        raise RequirementError("The CSV file has too many rows")
    out: list[FactorEntry] = []
    for i, r in enumerate(rows, 2):
        if not (r.get("key") or "").strip():
            continue
        try:
            out.append(FactorEntry.from_dict({k: (v or "") for k, v in r.items() if k in CSV_COLUMNS}))
        except ValueError as e:
            raise RequirementError(f"CSV line {i}: {e}") from None
    return out


# ------------------------------------------------------------------ material allowables
def allowable_of(material) -> MaterialAllowable | None:
    """The allowables record of a `Material` (None when none was entered)."""
    raw = getattr(material, "allowables", None)
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return MaterialAllowable.from_dict(raw)
    except ValueError:
        return None


def set_allowable(material, record: MaterialAllowable) -> None:
    """Stores the record on the material; a confirmation that does not match the content of the record is withdrawn."""
    if record.confirmed and record.confirmed_digest != record.content_digest():
        record.withdraw_confirmation()
    material.allowables = record.to_dict()


def confirm_allowable(material, by: str, date: str | None = None, via: str = "") -> MaterialAllowable:
    """The user confirms the allowables of a material (A or B basis, source named). Raises `RequirementError` otherwise."""
    rec = allowable_of(material)
    if rec is None:
        raise RequirementError(f"Material '{material.name}' has no allowables entered")
    problems = rec.problems()
    if not by.strip():
        problems.append("the name of the person who confirms is missing")
    if problems:
        raise RequirementError(f"The allowables of '{material.name}' cannot be confirmed: " + "; ".join(problems), problems)
    rec.confirmed, rec.confirmed_by = True, by.strip()
    rec.date = date or _dt.date.today().isoformat()
    rec.confirmed_via = via
    rec.confirmed_digest = rec.content_digest()
    material.allowables = rec.to_dict()
    return rec

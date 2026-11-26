"""Data model of the space-structures verification (stored in `Study.space` and `Material.allowables`).

All classes are plain dataclasses with `to_dict` / `from_dict` (JSON round trip; unknown keys are ignored on loading so that a file of a
newer version still opens). Nothing here carries a standard number: factors, limits and allowables are `None` until the user enters them,
and a requirement set counts as confirmed only while its content is the one that was confirmed (`RequirementSet.is_confirmed`).
No Qt and no `fea_core.mcp` import.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------ vocabulary (identifiers only – no numbers)
STRUCTURE_CLASSES = ("metallic", "joints_inserts", "buckling", "composite")        # classes of the factor table
PART_CLASSES = ("metallic", "joints_inserts", "composite")                          # classes a part can have ("buckling" = the global buckling check)
VERIFICATION_APPROACHES = ("qualification_model", "protoflight", "analysis_only")
# names of the factors of the requirement-set table = the factors of ECSS-E-ST-32-10C Rev.2 Corr.1 that enter the design factor J of a margin (clause 3.2 / 4.1.4 / 4.1.5, Table 4-1):
# FOSY, FOSU (4.3.2), KQ (test factor of satellites, 4.1.4.4; used only at system level, 4.2.1d), KP, KM (4.1.4.3, 4.1.4.2), KLD (4.1.5.2), KMP (margin policy of launch vehicles,
# 4.1.5.3). The acceptance factor KA applies to test loads, not to a margin, and is not part of J; a row of any other name can be added by the user.
FACTOR_KEYS = ("FOSY", "FOSU", "KQ", "KP", "KM", "KLD", "KMP")
CHECKS = ("yield", "ultimate", "buckling")
STRESS_MEASURES = ("von_mises", "max_principal", "max_shear")
ALLOWABLE_KEYS = ("Fy", "Fu", "Fsy", "Fsu")
BUCKLING_ALLOWABLE = "lambda"                                         # the "allowable" of a buckling check = the buckling load factor
THERMAL_HANDLING = ("", "same_as_mechanical", "separate", "none")      # how thermal stress is factored in a combined case ("" = not decided)
COMPONENT_KINDS = ("mechanical", "thermal", "mixed")   # "mixed" = a state with mechanical AND thermal loads that cannot be separated
VERIFIED_BASES = ("A", "B")                                           # allowable bases that count as verified (the others stay "unconfirmed")
CLAUSE_PLACEHOLDER = "<clause>"                                       # shipped in the empty template; the user replaces it by the real clause


class RequirementError(ValueError):
    """A requirement set / allowable cannot be accepted; `issues` lists the reasons."""

    def __init__(self, message: str, issues: list[str] | None = None):
        super().__init__(message)
        self.issues = list(issues or [message])


def _num(v: Any, what: str) -> float | None:
    """A number from JSON (None / "" = not set). Anything else that is not a number is an error, never a silent None."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        raise ValueError(f"{what}: a number is expected, not {v!r}")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: a number is expected, not {v!r}") from None


def _text(v: Any) -> str:
    return "" if v is None else str(v)


def _strs(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


# ------------------------------------------------------------------ requirement set
@dataclass
class FactorEntry:
    """One factor of the requirement-set table: `key` for a structure class and a verification approach ("" = any approach).
    document / revision / clause = where the user took the value from (printed in the report)."""
    key: str
    structure_class: str
    approach: str = ""
    value: float | None = None
    document: str = ""
    revision: str = ""
    clause: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "structure_class": self.structure_class, "approach": self.approach, "value": self.value,
                "document": self.document, "revision": self.revision, "clause": self.clause, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "FactorEntry":
        return cls(key=_text(d.get("key")), structure_class=_text(d.get("structure_class")), approach=_text(d.get("approach")),
                   value=_num(d.get("value"), f"factor {d.get('key')}"), document=_text(d.get("document")),
                   revision=_text(d.get("revision")), clause=_text(d.get("clause")), note=_text(d.get("note")))

    @property
    def provenance_ok(self) -> bool:
        return bool(self.document.strip() and self.revision.strip() and self.clause.strip()
                    and self.clause.strip() != CLAUSE_PLACEHOLDER)


@dataclass
class MarginRule:
    """How one margin of safety is formed (the formula chain of the standard, entered by the user):

    MoS = allowable / (J · σ_eq) − 1, with the design factor J = product of the values of `factors` (keys of the factor table, taken for
    the structure class and the verification approach of the set). `allowable` = a key of `MaterialAllowable` (or "lambda": the buckling
    load factor), `stress_measure` = the equivalent stress compared with it. For a case that combines a mechanical and a thermal load,
    `thermal_handling` says how the thermal part is factored ("same_as_mechanical" | "separate" with `thermal_factors` | "none").
    """
    check: str
    structure_class: str
    allowable: str = ""
    stress_measure: str = ""
    factors: list[str] = field(default_factory=list)
    thermal_handling: str = ""
    thermal_factors: list[str] = field(default_factory=list)
    document: str = ""
    revision: str = ""
    clause: str = ""
    note: str = ""
    not_applicable: bool = False      # the user declares the check not applicable to the class (needs document / revision / clause)

    def to_dict(self) -> dict:
        d = {"check": self.check, "structure_class": self.structure_class, "allowable": self.allowable,
             "stress_measure": self.stress_measure, "factors": list(self.factors), "thermal_handling": self.thermal_handling,
             "thermal_factors": list(self.thermal_factors), "document": self.document, "revision": self.revision,
             "clause": self.clause, "note": self.note}
        if self.not_applicable:                                     # only when set: the digest of the other rules does not change
            d["not_applicable"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MarginRule":
        return cls(check=_text(d.get("check")), structure_class=_text(d.get("structure_class")), allowable=_text(d.get("allowable")),
                   stress_measure=_text(d.get("stress_measure")), factors=_strs(d.get("factors")),
                   thermal_handling=_text(d.get("thermal_handling")), thermal_factors=_strs(d.get("thermal_factors")),
                   document=_text(d.get("document")), revision=_text(d.get("revision")), clause=_text(d.get("clause")),
                   note=_text(d.get("note")), not_applicable=d.get("not_applicable") is True)

    @property
    def provenance_ok(self) -> bool:
        return bool(self.document.strip() and self.revision.strip() and self.clause.strip()
                    and self.clause.strip() != CLAUSE_PLACEHOLDER)


@dataclass
class LimitEntry:
    """A named limit of a later stage (mass tolerance, minimum frequency, mode-pairing limits ...) with its provenance. The keys are
    defined by the stage that uses them; this table only stores and confirms them with the rest of the requirement set."""
    key: str
    value: float | None = None
    unit: str = ""
    document: str = ""
    revision: str = ""
    clause: str = ""
    note: str = ""
    not_applicable: bool = False        # the user declares the criterion not applicable to the project (needs document / revision / clause, no value)
    set_via: str = ""                   # through which interface the not-applicable flag was entered: "mcp" (an agent) | "import" (a file) | "gui" | "" = unknown

    def to_dict(self) -> dict:
        d = {"key": self.key, "value": self.value, "unit": self.unit, "document": self.document, "revision": self.revision,
             "clause": self.clause, "note": self.note}
        if self.not_applicable:                                     # only when set: the digest of the other limits does not change
            d["not_applicable"] = True
            if self.set_via:
                d["set_via"] = self.set_via
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LimitEntry":
        return cls(key=_text(d.get("key")), value=_num(d.get("value"), f"limit {d.get('key')}"), unit=_text(d.get("unit")),
                   document=_text(d.get("document")), revision=_text(d.get("revision")), clause=_text(d.get("clause")),
                   note=_text(d.get("note")), not_applicable=d.get("not_applicable") is True,
                   set_via=_text(d.get("set_via"))[:12] if d.get("not_applicable") is True else "")

    @property
    def provenance_ok(self) -> bool:
        return bool(self.document.strip() and self.revision.strip() and self.clause.strip() and self.clause.strip() != CLAUSE_PLACEHOLDER)


@dataclass
class RequirementSet:
    """The requirement set of a project: metadata, factor table, margin rules and limits, plus who confirmed it and when.

    `confirmed` is only the last state the user set; the set counts as confirmed (`is_confirmed`) only while its content still hashes to
    `confirmed_digest`, so any later edit of a value, a rule or the approach withdraws the confirmation.
    """
    name: str = ""
    standard: str = ""
    revision: str = ""
    verification_approach: str = ""
    factors: list[FactorEntry] = field(default_factory=list)
    rules: list[MarginRule] = field(default_factory=list)
    limits: list[LimitEntry] = field(default_factory=list)
    notes: str = ""
    confirmed: bool = False
    confirmed_by: str = ""
    date: str = ""
    confirmed_digest: str = ""
    confirmed_via: str = ""             # "gui" | "mcp": through which interface the confirmation was made ("" = unknown / other code)
    scope_digest: str = ""              # the structure classes of the parts at the confirmation (`SpaceSettings.scope_digest`)

    # ---------------------------------------------------------------- JSON
    def content_dict(self) -> dict:
        return {"name": self.name, "standard": self.standard, "revision": self.revision,
                "verification_approach": self.verification_approach, "notes": self.notes,
                "factors": [f.to_dict() for f in self.factors], "rules": [r.to_dict() for r in self.rules],
                "limits": [x.to_dict() for x in self.limits]}

    def to_dict(self) -> dict:
        d = self.content_dict()
        d.update({"confirmed": self.confirmed, "confirmed_by": self.confirmed_by, "date": self.date,
                  "confirmed_digest": self.confirmed_digest, "confirmed_via": self.confirmed_via,
                  "scope_digest": self.scope_digest})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RequirementSet":
        return cls(name=_text(d.get("name")), standard=_text(d.get("standard")), revision=_text(d.get("revision")),
                   verification_approach=_text(d.get("verification_approach")), notes=_text(d.get("notes")),
                   factors=[FactorEntry.from_dict(x) for x in d.get("factors") or []],
                   rules=[MarginRule.from_dict(x) for x in d.get("rules") or []],
                   limits=[LimitEntry.from_dict(x) for x in d.get("limits") or []],
                   confirmed=bool(d.get("confirmed", False)), confirmed_by=_text(d.get("confirmed_by")),
                   date=_text(d.get("date")), confirmed_digest=_text(d.get("confirmed_digest")),
                   confirmed_via=_text(d.get("confirmed_via")), scope_digest=_text(d.get("scope_digest")))

    # ---------------------------------------------------------------- confirmation
    def content_digest(self) -> str:
        """SHA-256 of the content (everything except the confirmation fields)."""
        text = json.dumps(self.content_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def is_confirmed(self) -> bool:
        return bool(self.confirmed and self.confirmed_by.strip() and self.confirmed_digest
                    and self.confirmed_digest == self.content_digest())

    def withdraw_confirmation(self) -> None:
        self.confirmed, self.confirmed_by, self.date, self.confirmed_digest, self.confirmed_via = False, "", "", "", ""
        self.scope_digest = ""

    def entry(self, key: str, structure_class: str, approach: str | None = None) -> FactorEntry | None:
        """The factor row for (key, class): the row of the given approach (default: the approach of the set) if it has a value, otherwise
        the row valid for any approach ("") if it has one; without a value anywhere the row of the approach (or the "any" row)."""
        ap = self.verification_approach if approach is None else approach
        exact = any_row = None
        for f in self.factors:
            if f.key == key and f.structure_class == structure_class:
                if f.approach == ap and ap:
                    exact = f
                elif f.approach == "":
                    any_row = f
        for cand in (exact, any_row):
            if cand is not None and cand.value is not None:
                return cand
        return exact or any_row

    def rule(self, check: str, structure_class: str) -> MarginRule | None:
        for r in self.rules:
            if r.check == check and r.structure_class == structure_class:
                return r
        return None


# ------------------------------------------------------------------ material allowables (Material.allowables)
@dataclass
class MaterialAllowable:
    """Allowables of a material entered by the user (A/B basis from the material handbook): tension yield / ultimate `Fy`, `Fu`, shear
    yield / ultimate `Fsy`, `Fsu` [MPa] at the temperature `temp_C`, with the source. `confirmed` is withdrawn by any later edit
    (digest), as for the requirement set."""
    basis: str = ""
    Fy: float | None = None
    Fu: float | None = None
    Fsy: float | None = None
    Fsu: float | None = None
    temp_C: float | None = None
    source: str = ""
    confirmed: bool = False
    confirmed_by: str = ""
    date: str = ""
    confirmed_digest: str = ""
    confirmed_via: str = ""             # "gui" | "mcp" (see RequirementSet)

    def content_dict(self) -> dict:
        return {"basis": self.basis, "Fy": self.Fy, "Fu": self.Fu, "Fsy": self.Fsy, "Fsu": self.Fsu, "temp_C": self.temp_C,
                "source": self.source}

    def to_dict(self) -> dict:
        d = self.content_dict()
        d.update({"confirmed": self.confirmed, "confirmed_by": self.confirmed_by, "date": self.date,
                  "confirmed_digest": self.confirmed_digest, "confirmed_via": self.confirmed_via})
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "MaterialAllowable":
        d = d or {}
        basis = _text(d.get("basis")).strip()
        return cls(basis=basis.upper() if basis.lower() in ("a", "b", "s") else basis,
                   Fy=_num(d.get("Fy"), "Fy"), Fu=_num(d.get("Fu"), "Fu"), Fsy=_num(d.get("Fsy"), "Fsy"), Fsu=_num(d.get("Fsu"), "Fsu"),
                   temp_C=_num(d.get("temp_C"), "temp_C"), source=_text(d.get("source")), confirmed=bool(d.get("confirmed", False)),
                   confirmed_by=_text(d.get("confirmed_by")), date=_text(d.get("date")), confirmed_digest=_text(d.get("confirmed_digest")),
                   confirmed_via=_text(d.get("confirmed_via")))

    def content_digest(self) -> str:
        text = json.dumps(self.content_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def value(self, key: str) -> float | None:
        return getattr(self, key, None) if key in ALLOWABLE_KEYS else None

    def problems(self) -> list[str]:
        """Why this record cannot be confirmed / used (empty = fine)."""
        out = []
        for k in ALLOWABLE_KEYS:
            v = self.value(k)
            if v is not None and (not math.isfinite(v) or v <= 0):
                out.append(f"{k} must be a positive number")
        if all(self.value(k) is None for k in ALLOWABLE_KEYS):
            out.append("no allowable value entered")
        if self.temp_C is not None and not math.isfinite(self.temp_C):
            out.append("temp_C must be a number")
        if not self.basis.strip():
            out.append("the basis (A / B) is missing")
        if not self.source.strip():
            out.append("the source of the values is missing")
        return out

    def is_confirmed(self) -> bool:
        return bool(self.confirmed and self.confirmed_by.strip() and self.confirmed_digest
                    and self.confirmed_digest == self.content_digest())

    def withdraw_confirmation(self) -> None:
        self.confirmed, self.confirmed_by, self.date, self.confirmed_digest, self.confirmed_via = False, "", "", "", ""

    def is_verified(self) -> bool:
        """Confirmed by the user and of a verified basis (A or B)."""
        return self.is_confirmed() and self.basis.strip().upper() in VERIFIED_BASES and not self.problems()


def scope_digest_of(default_class: str, part_classes: dict[str, str]) -> str:
    text = json.dumps({"default_class": default_class or "metallic", "part_classes": {k: v for k, v in sorted(part_classes.items()) if v}},
                      sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


DEFAULT_SCOPE_DIGEST = scope_digest_of("metallic", {})             # no per-part class, every part metallic


# ------------------------------------------------------------------ margin cases, justifications, settings
@dataclass
class CaseComponent:
    """One term of a margin case: scale · (result state `source`), kind = how its stress is factored."""
    source: str
    scale: float = 1.0
    kind: str = "mechanical"

    def to_dict(self) -> dict:
        return {"source": self.source, "scale": self.scale, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict) -> "CaseComponent":
        scale = _num(d.get("scale", 1.0), "scale")
        return cls(source=_text(d.get("source")), scale=1.0 if scale is None else scale, kind=_text(d.get("kind")) or "mechanical")


@dataclass
class MarginCase:
    """A case of the margin evaluation made of result states: e.g. quasi-static load + thermal state with separate factor handling."""
    name: str
    components: list[CaseComponent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "components": [c.to_dict() for c in self.components]}

    @classmethod
    def from_dict(cls, d: dict) -> "MarginCase":
        return cls(name=_text(d.get("name")), components=[CaseComponent.from_dict(c) for c in d.get("components") or []])


@dataclass
class Justification:
    """The user's note for a singular place (the singularity justification table): where it is and why it is accepted. A detected
    singularity matches the note of the same part within `tolerance_mm` of `xyz`."""
    part: str
    xyz: list[float]
    note: str
    tolerance_mm: float = 5.0
    cause: str = ""
    author: str = ""
    date: str = ""

    def to_dict(self) -> dict:
        return {"part": self.part, "xyz": [float(x) for x in self.xyz], "note": self.note, "tolerance_mm": self.tolerance_mm,
                "cause": self.cause, "author": self.author, "date": self.date}

    @classmethod
    def from_dict(cls, d: dict) -> "Justification":
        xyz = [float(_num(x, "xyz") or 0.0) for x in (d.get("xyz") or [0.0, 0.0, 0.0])]
        tol = _num(d.get("tolerance_mm", 5.0), "tolerance_mm")
        return cls(part=_text(d.get("part")), xyz=(xyz + [0.0, 0.0, 0.0])[:3], note=_text(d.get("note")),
                   tolerance_mm=5.0 if tol is None else tol, cause=_text(d.get("cause")), author=_text(d.get("author")),
                   date=_text(d.get("date")))


@dataclass
class SpaceSettings:
    """`Study.space`: the requirement set, the structure class of the parts, extra margin cases and the singularity justifications.
    `extra` keeps keys of later stages / newer versions untouched (round trip)."""
    requirement_set: RequirementSet | None = None
    default_class: str = "metallic"
    part_classes: dict[str, str] = field(default_factory=dict)
    cases: list[MarginCase] = field(default_factory=list)
    justifications: list[Justification] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def class_of(self, part: str) -> str:
        return self.part_classes.get(part) or self.default_class

    def scope_digest(self) -> str:
        """SHA-256 of the structure classes (default class and per-part classes): which rules apply to which part is part of what was confirmed."""
        return scope_digest_of(self.default_class, self.part_classes)

    def requirement_confirmed(self) -> bool:
        """The requirement set is confirmed AND the structure classes are those it was confirmed for (an edit of either withdraws it)."""
        rs = self.requirement_set
        return bool(rs is not None and rs.is_confirmed() and rs.scope_digest == self.scope_digest())

    def to_dict(self) -> dict:
        d: dict = dict(self.extra)
        d["default_class"] = self.default_class
        if self.requirement_set is not None:
            d["requirement_set"] = self.requirement_set.to_dict()
        if self.part_classes:
            d["part_classes"] = dict(self.part_classes)
        if self.cases:
            d["cases"] = [c.to_dict() for c in self.cases]
        if self.justifications:
            d["justifications"] = [j.to_dict() for j in self.justifications]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "SpaceSettings":
        d = d or {}
        known = {"requirement_set", "default_class", "part_classes", "cases", "justifications"}
        rs = d.get("requirement_set")
        return cls(requirement_set=RequirementSet.from_dict(rs) if isinstance(rs, dict) else None,
                   default_class=_text(d.get("default_class")) or "metallic",
                   part_classes={str(k): str(v) for k, v in (d.get("part_classes") or {}).items()},
                   cases=[MarginCase.from_dict(c) for c in d.get("cases") or []],
                   justifications=[Justification.from_dict(j) for j in d.get("justifications") or []],
                   extra={k: v for k, v in d.items() if k not in known})

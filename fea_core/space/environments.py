"""Launch environments of the space-structures verification (F14 stage S2): the named levels (`qualification`, `protoflight`, `acceptance`) with their sine
table, random ASD, shock response spectrum, duration and sweep rate, plus the launch axis and the way the quasi-static cases are combined.

The program ships NO environment: every table is empty until the user enters it from the launcher user's guide / the project requirements, with the source
(document, revision, clause) as text. An environment counts as confirmed only while its content still hashes to the digest that was confirmed (the same
mechanism as the requirement set, `fea_core.space.model.RequirementSet`): any later edit withdraws the confirmation, an import never carries it over, and an
agent (MCP) may confirm only in a call of its own. Results computed with an unconfirmed environment are labelled so and can never produce a PASS.

The set is stored in `Study.space.extra["environments"]` (round trip of the space settings keeps it); `environments_of` / `store_environments` are the accessors.
No Qt and no `fea_core.mcp` import.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from .model import CLAUSE_PLACEHOLDER, RequirementError, SpaceSettings

ENV_LEVELS = ("qualification", "protoflight", "acceptance")
ENV_KINDS = ("sine", "random", "shock")
LAUNCH_AXES = ("x", "y", "z")
QSL_COMBINATIONS = ("none", "axial_lateral", "axial_both_laterals")          # how the signed quasi-static cases are combined (a choice of the user)
MAX_POINTS = 500
MAX_LEVELS = 12
MAX_TEXT = 500                          # document / revision / clause / names / the source of the damping
MAX_NOTE = 5000                         # free notes (table note, environment notes)
KIND_UNITS = {"sine": ("Hz", "g"), "random": ("Hz", "g2/Hz"), "shock": ("Hz", "g")}      # (abscissa, ordinate) of the tables
ENV_KEY = "environments"


def parse_number(v: Any, what: str) -> float | None:
    """A finite float from a user value (None / empty text = None); ValueError with `what` otherwise (also for inf / NaN)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        raise ValueError(f"{what}: a number is expected, not {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: a number is expected, not {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{what}: a finite number is expected, not {v!r}")
    return x


def bounded_text(v: Any, what: str = "text", limit: int = MAX_TEXT) -> str:
    """`v` as text; a text longer than `limit` characters is refused (not truncated)."""
    s = "" if v is None else str(v)
    if len(s) > limit:
        raise ValueError(f"{what}: at most {limit} characters (it has {len(s)})")
    return s


def _points(v: Any, what: str) -> list[list[float]]:
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{what}: a list of [frequency, value] pairs is expected")
    if len(v) > MAX_POINTS:
        raise ValueError(f"{what}: at most {MAX_POINTS} points")
    out: list[list[float]] = []
    for p in v:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{what}: every point is a pair [frequency, value]")
        a, b = parse_number(p[0], what), parse_number(p[1], what)
        if a is None or b is None:
            raise ValueError(f"{what}: a point has an empty number")
        out.append([a, b])
    return out


# ------------------------------------------------------------------ one table (sine / random / shock) of a level
@dataclass
class EnvTable:
    """The table of one kind at one level, with its source. sine: [[f Hz, amplitude g]] (zero-to-peak), random: [[f Hz, ASD g²/Hz]], shock: [[f Hz, SRS g]].
    `duration_s` (random: time per axis; used for the damage), `sweep_rate_oct_min` / `sweeps` (sine), `sigma` (random: the multiple of the RMS stress that
    is compared with the allowable; None = the RMS stress is only reported) are entered by the user with the table."""
    kind: str
    points: list[list[float]] = field(default_factory=list)
    duration_s: float | None = None
    sweep_rate_oct_min: float | None = None
    sweeps: int | None = None
    sigma: float | None = None
    document: str = ""
    revision: str = ""
    clause: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "points": [list(p) for p in self.points], "duration_s": self.duration_s,
                "sweep_rate_oct_min": self.sweep_rate_oct_min, "sweeps": self.sweeps, "sigma": self.sigma, "document": self.document,
                "revision": self.revision, "clause": self.clause, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvTable":
        sweeps = parse_number(d.get("sweeps"), "sweeps")
        if sweeps is not None and sweeps != int(sweeps):
            raise ValueError(f"sweeps: a whole number is expected, not {d.get('sweeps')!r}")
        return cls(kind=bounded_text(d.get("kind"), "kind", 50), points=_points(d.get("points"), f"{d.get('kind')} table"),
                   duration_s=parse_number(d.get("duration_s"), "duration_s"), sweep_rate_oct_min=parse_number(d.get("sweep_rate_oct_min"), "sweep_rate_oct_min"),
                   sweeps=None if sweeps is None else int(sweeps), sigma=parse_number(d.get("sigma"), "sigma"), document=bounded_text(d.get("document"), "document"),
                   revision=bounded_text(d.get("revision"), "revision"), clause=bounded_text(d.get("clause"), "clause"), note=bounded_text(d.get("note"), "note", MAX_NOTE))

    @property
    def provenance_ok(self) -> bool:
        return bool(self.document.strip() and self.revision.strip() and self.clause.strip() and self.clause.strip() != CLAUSE_PLACEHOLDER)

    def problems(self, include_source: bool = True) -> list[str]:
        """Why the table cannot be used / confirmed (empty = fine). `include_source=False` leaves out the missing document / revision / clause: such a table still
        computes (its result stays UNCONFIRMED), it only cannot be confirmed."""
        out: list[str] = []
        if self.kind not in ENV_KINDS:
            out.append(f"kind must be one of {', '.join(ENV_KINDS)}")
        if len(self.points) < 2:
            out.append("at least two points are needed")
        last = 0.0
        for i, (f, v) in enumerate(self.points):
            if not (math.isfinite(f) and math.isfinite(v)):
                out.append(f"point {i + 1} is not finite")
            elif f <= 0 or v <= 0:
                out.append(f"point {i + 1}: frequency and value must be positive")
            elif f <= last:
                out.append(f"point {i + 1}: the frequencies must increase")
            last = f if f > last else last
        for name in ("duration_s", "sweep_rate_oct_min", "sigma"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v <= 0):
                out.append(f"{name} must be a positive number")
        if self.sweeps is not None and self.sweeps < 1:
            out.append("sweeps must be at least 1")
        if include_source and not self.provenance_ok:
            out.append("document / revision / clause of the source are not filled")
        return out


@dataclass
class EnvLevel:
    """A named level (qualification | protoflight | acceptance): the tables the user entered for it. `combine` / `missing_mass` are analysis options of the shock."""
    name: str
    sine: EnvTable | None = None
    random: EnvTable | None = None
    shock: EnvTable | None = None
    combine: str = "cqc"
    missing_mass: bool = True

    def table(self, kind: str) -> EnvTable | None:
        return getattr(self, kind, None) if kind in ENV_KINDS else None

    def kinds(self) -> list[str]:
        out: list[str] = []
        for k in ENV_KINDS:
            t = self.table(k)
            if t is not None and t.points:
                out.append(k)
        return out

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "combine": self.combine, "missing_mass": self.missing_mass}
        for k in ENV_KINDS:
            t = self.table(k)
            if t is not None:
                d[k] = t.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvLevel":
        lv = cls(name=bounded_text(d.get("name"), "level name", 50), combine=bounded_text(d.get("combine"), "combine", 50) or "cqc", missing_mass=d.get("missing_mass", True) is not False)
        for k in ENV_KINDS:
            raw = d.get(k)
            if isinstance(raw, dict):
                t = EnvTable.from_dict({**raw, "kind": k})
                setattr(lv, k, t)
        return lv


@dataclass
class EnvironmentSet:
    """All environments of the project: the levels, the launch axis and the way the QSL cases are combined, the modal damping of the launch analysis with its source
    text (`damping_zeta`: part of the content, so part of the confirmation), plus who confirmed it and when."""
    levels: list[EnvLevel] = field(default_factory=list)
    launch_axis: str = ""
    qsl_combination: str = "axial_lateral"
    notes: str = ""
    damping_zeta: float | None = None
    damping_source: str = ""
    confirmed: bool = False
    confirmed_by: str = ""
    date: str = ""
    confirmed_digest: str = ""
    confirmed_via: str = ""             # "gui" | "mcp"

    # ---------------------------------------------------------------- JSON
    def content_dict(self) -> dict[str, Any]:
        return {"levels": [lv.to_dict() for lv in self.levels], "launch_axis": self.launch_axis, "qsl_combination": self.qsl_combination,
                "notes": self.notes, "damping_zeta": self.damping_zeta, "damping_source": self.damping_source}

    def to_dict(self) -> dict[str, Any]:
        d = self.content_dict()
        d.update({"confirmed": self.confirmed, "confirmed_by": self.confirmed_by, "date": self.date, "confirmed_digest": self.confirmed_digest,
                  "confirmed_via": self.confirmed_via})
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvironmentSet":
        levels = d.get("levels") or []
        if not isinstance(levels, list) or len(levels) > MAX_LEVELS or not all(isinstance(x, dict) for x in levels):
            raise ValueError(f"'levels' must be a list of at most {MAX_LEVELS} objects")
        return cls(levels=[EnvLevel.from_dict(x) for x in levels], launch_axis=bounded_text(d.get("launch_axis"), "launch_axis", 50),
                   qsl_combination=bounded_text(d.get("qsl_combination"), "qsl_combination", 50) or "axial_lateral", notes=bounded_text(d.get("notes"), "notes", MAX_NOTE),
                   damping_zeta=parse_number(d.get("damping_zeta"), "damping_zeta"), damping_source=bounded_text(d.get("damping_source"), "damping_source"),
                   confirmed=bool(d.get("confirmed", False)), confirmed_by=bounded_text(d.get("confirmed_by"), "confirmed_by"), date=bounded_text(d.get("date"), "date", 50),
                   confirmed_digest=bounded_text(d.get("confirmed_digest"), "confirmed_digest", 200), confirmed_via=bounded_text(d.get("confirmed_via"), "confirmed_via", 50))

    # ---------------------------------------------------------------- confirmation
    def content_digest(self) -> str:
        text = json.dumps(self.content_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def is_confirmed(self) -> bool:
        return bool(self.confirmed and self.confirmed_by.strip() and self.confirmed_digest and self.confirmed_digest == self.content_digest())

    def withdraw_confirmation(self) -> None:
        self.confirmed, self.confirmed_by, self.date, self.confirmed_digest, self.confirmed_via = False, "", "", "", ""

    def level(self, name: str) -> EnvLevel | None:
        for lv in self.levels:
            if lv.name == name:
                return lv
        return None

    def ensure_level(self, name: str) -> EnvLevel:
        lv = self.level(name)
        if lv is None:
            if len(self.levels) >= MAX_LEVELS:
                raise ValueError(f"at most {MAX_LEVELS} levels")
            lv = EnvLevel(name)
            self.levels.append(lv)
        return lv


# ------------------------------------------------------------------ template, validation, confirmation
def environment_template() -> EnvironmentSet:
    """The empty template: the three named levels with a source placeholder per kind, no number anywhere."""
    es = EnvironmentSet()
    for name in ENV_LEVELS:
        lv = EnvLevel(name)
        for k in ENV_KINDS:
            setattr(lv, k, EnvTable(kind=k, clause=CLAUSE_PLACEHOLDER))
        es.levels.append(lv)
    return es


def validate_environment(es: EnvironmentSet) -> list[str]:
    """Structural problems (an error text per problem); used tables must be complete, the unused (no points) are ignored."""
    out: list[str] = []
    if es.launch_axis and es.launch_axis not in LAUNCH_AXES:
        out.append(f"launch_axis must be one of {', '.join(LAUNCH_AXES)}")
    if es.qsl_combination not in QSL_COMBINATIONS:
        out.append(f"qsl_combination must be one of {', '.join(QSL_COMBINATIONS)}")
    if es.damping_zeta is not None and not (math.isfinite(es.damping_zeta) and 0 < es.damping_zeta < 1):
        out.append("damping_zeta must be a damping ratio between 0 and 1 (exclusive)")
    if es.damping_zeta is not None and not es.damping_source.strip():
        out.append("damping_source: the source of the damping ratio (a test, the launcher guide, a document) is missing")
    names: set[str] = set()
    for lv in es.levels:
        if lv.name not in ENV_LEVELS:
            out.append(f"level '{lv.name}': the name must be one of {', '.join(ENV_LEVELS)}")
        if lv.name in names:
            out.append(f"level '{lv.name}': duplicate")
        names.add(lv.name)
        if lv.combine not in ("srss", "cqc"):
            out.append(f"level {lv.name}: shock combination must be srss or cqc")
        for k in lv.kinds():
            t = lv.table(k)
            if t is not None:
                out += [f"{lv.name} / {k}: {p}" for p in t.problems()]
    return out


def confirmation_problems(es: EnvironmentSet, by: str) -> list[str]:
    msgs = validate_environment(es)
    if not by.strip():
        msgs.append("the name of the person who confirms is missing")
    if not any(lv.kinds() for lv in es.levels):
        msgs.append("no environment table is entered")
    return list(dict.fromkeys(msgs))


def confirm_environment(es: EnvironmentSet, by: str, date: str | None = None, via: str = "") -> None:
    """The user confirms the environments (after checking them against the launcher guide / the requirements). Raises `RequirementError` with the reasons."""
    problems = confirmation_problems(es, by)
    if problems:
        raise RequirementError("The environments cannot be confirmed: " + "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else ""), problems)
    es.confirmed, es.confirmed_by = True, by.strip()
    es.date = date or _dt.date.today().isoformat()
    es.confirmed_via = via
    es.confirmed_digest = es.content_digest()


# ------------------------------------------------------------------ storage in Study.space.extra
def environments_of(space: SpaceSettings | None) -> EnvironmentSet | None:
    """The environments stored in the space settings (None when there are none / the record cannot be read)."""
    if space is None:
        return None
    raw = space.extra.get(ENV_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        return EnvironmentSet.from_dict(raw)
    except (ValueError, TypeError, OverflowError):
        return None


def store_environments(space: SpaceSettings, es: EnvironmentSet | None) -> None:
    """Stores the set; a confirmation that does not match the content is withdrawn."""
    if es is None:
        space.extra.pop(ENV_KEY, None)
        return
    if es.confirmed and not es.is_confirmed():
        es.withdraw_confirmation()
    space.extra[ENV_KEY] = es.to_dict()


def environment_confirmed(space: SpaceSettings | None) -> bool:
    es = environments_of(space)
    return bool(es is not None and es.is_confirmed())


def to_json(es: EnvironmentSet) -> str:
    return json.dumps({"format": "fea-studio-environments", "version": 1, **es.to_dict()}, indent=2, ensure_ascii=False)


def from_json(text: str) -> EnvironmentSet:
    """Environments from JSON text (bounded). An imported set is NOT confirmed."""
    if len(text.encode("utf-8", "replace")) > (2 << 20):
        raise RequirementError("The environment file is too large")
    try:
        d = json.loads(text)
    except ValueError as e:
        raise RequirementError(f"Not a JSON file: {e}") from None
    if not isinstance(d, dict):
        raise RequirementError("An environment set is a JSON object")
    try:
        es = EnvironmentSet.from_dict(d)
    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        raise RequirementError(f"The environments could not be read: {e}") from None
    es.withdraw_confirmation()
    return es

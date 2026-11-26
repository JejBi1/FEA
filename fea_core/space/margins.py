"""Margins of safety of a solved study against the user-confirmed requirement set and material allowables (ECSS-E-ST-32-10C style).

    MoS = allowable / (J · σ_eq) − 1        J = product of the factors of the margin rule (`requirements.resolve_rule`)

The engine knows the form of the margin and nothing else: which factors make up J, which stress measure and which allowable belong to a
check is entered in the requirement set by the user and confirmed (no factor, load level or clause is built in). Per part, load case
(a result state, an EN 1990 combination, a user-defined case, an externally supplied stress envelope) and check the governing place is
the highest-ranking one that the singularity classifier (`fea_core.hotspots`) does NOT call a singularity; the singular places go to the
singularity justification table and need a note of the user before sign-off.

Verdict: `FAIL` when a margin with confirmed inputs is negative, `UNCONFIRMED_REQUIREMENTS` while any input (requirement set, rule,
allowable, structure class, linear superposition ...) is not confirmed or complete - the numbers are then indicative only - and `PASS`
only when everything is confirmed and every margin is non-negative. Linear studies only: factors scale the stress, which is invalid for
nonlinear analyses (the verdict is then never PASS).
No Qt and no `fea_core.mcp` import.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..hotspots import Classifier, scan_maxima
from ..joints.model import is_preload
from ..results import StepResult, principal, von_mises
from .model import (BUCKLING_ALLOWABLE, CaseComponent, Justification, MarginCase, MarginRule, MaterialAllowable, RequirementSet,
                    SpaceSettings)
from .requirements import ResolvedRule, allowable_of, ecss_32_10c_template, resolve_rule

if TYPE_CHECKING:
    from ..results import Results
    from ..study import Study

PASS = "PASS"
FAIL = "FAIL"
UNCONFIRMED = "UNCONFIRMED_REQUIREMENTS"
MAX_PLACES = 16                     # singular places that are walked down before the margin outside singularities is declared undetermined
PRIMARY_CHECKS = ("yield", "ultimate")
DEFAULT_ALLOWABLE = {"yield": "Fy", "ultimate": "Fu"}                   # when a rule does not name its allowable (indicative)
FALLBACK_ALLOWABLE = {"Fy": "Re", "Fu": "Rm"}          # indicative values from the material properties when no allowables were entered


@dataclass
class ScalarSource:
    """An equivalent-stress field (N,) supplied from outside, e.g. a dynamic envelope: it can only be added with another scalar of the
    same case, so it forms a case of its own (`kind` = how its stress is factored)."""
    name: str
    eq: np.ndarray
    kind: str = "mechanical"


@dataclass
class MarginRow:
    """The governing margin of one (part, case, check)."""
    part: str
    case: str
    check: str
    structure_class: str
    material: str
    allowable_name: str
    allowable: float | None
    design_factor: float | None
    thermal_factor: float | None
    design_stress: float | None            # equivalent stress of the factored load at the governing place [MPa]
    mos: float | None                      # outside singularities
    mos_raw: float | None                  # at the highest place including singularities
    node: int | None
    location: list[float] | None
    classification: str
    status: str                            # "pass" | "fail" | "unconfirmed"
    negative: bool = False                 # margin < 0 although the inputs are not confirmed (indicative)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"part": self.part, "case": self.case, "check": self.check, "structure_class": self.structure_class,
                "material": self.material, "allowable_name": self.allowable_name, "allowable": self.allowable,
                "design_factor": self.design_factor, "thermal_factor": self.thermal_factor, "design_stress_MPa": self.design_stress,
                "mos": self.mos, "mos_incl_singularities": self.mos_raw, "node": self.node, "location_mm": self.location,
                "classification": self.classification, "status": self.status, "negative": self.negative,
                "reasons": list(self.reasons)}


@dataclass
class SingularityRow:
    """A detected singular place for the justification table (one per place, merged over cases and checks)."""
    part: str
    location: list[float]
    cause: str
    mos_raw: float | None
    cases: list[str]
    checks: list[str]
    justification: Justification | None = None

    @property
    def justified(self) -> bool:
        return self.justification is not None and bool(self.justification.note.strip())

    def to_dict(self) -> dict:
        return {"part": self.part, "location_mm": self.location, "cause": self.cause, "mos_incl_singularities": self.mos_raw,
                "cases": list(self.cases), "checks": list(self.checks), "justified": self.justified,
                "note": self.justification.note if self.justification else ""}


@dataclass
class MarginSummary:
    verdict: str
    rows: list[MarginRow]
    singularities: list[SingularityRow]
    reasons: list[str]                     # what keeps the verdict from being PASS (global)
    warnings: list[str]
    requirement_set: str = ""
    standard: str = ""
    revision: str = ""
    approach: str = ""
    confirmed: bool = False
    confirmed_by: str = ""
    date: str = ""
    cases: list[str] = field(default_factory=list)
    confirmed_via: str = ""                # "gui" | "mcp" | "": how the requirement set was confirmed
    allowables_via: dict[str, str] = field(default_factory=dict)   # material -> how its allowables were confirmed (confirmed ones only)

    @property
    def worst(self) -> MarginRow | None:
        rows = [r for r in self.rows if r.mos is not None]
        return min(rows, key=lambda r: r.mos if r.mos is not None else math.inf) if rows else None

    @property
    def signoff_ready(self) -> bool:
        return self.verdict == PASS and all(s.justified for s in self.singularities)

    @property
    def verdict_text(self) -> str:
        """The verdict as it is shown to a person: a PASS with singularities nobody justified yet is not a sign-off."""
        if self.verdict == PASS and not self.signoff_ready:
            n = sum(1 for s in self.singularities if not s.justified)
            return f"PASS – not ready for sign-off (open singularities: {n})"
        return self.verdict

    @property
    def open_items(self) -> list[str]:
        out = list(self.reasons)
        if self.verdict == FAIL:
            out.insert(0, f"{sum(1 for r in self.rows if r.status == 'fail')} margin(s) of safety negative (FAIL)")
        out += [f"singularity not justified: {s.part} at {_xyz(s.location)} ({s.cause})" for s in self.singularities if not s.justified]
        return out

    def lowest(self, n: int = 10) -> list[MarginRow]:
        rows = [r for r in self.rows if r.mos is not None]
        return sorted(rows, key=lambda r: r.mos if r.mos is not None else math.inf)[:n]

    def to_dict(self, max_rows: int = 40) -> dict:
        w = self.worst
        return {"verdict": self.verdict, "verdict_text": self.verdict_text, "signoff_ready": self.signoff_ready,
                "requirement_set": {"name": self.requirement_set, "standard": self.standard, "revision": self.revision,
                                    "approach": self.approach, "confirmed": self.confirmed, "confirmed_by": self.confirmed_by,
                                    "confirmed_via": self.confirmed_via, "date": self.date},
                "allowables_confirmed_via": dict(self.allowables_via),
                "worst": w.to_dict() if w else None, "lowest": [r.to_dict() for r in self.lowest(max_rows)],
                "rows": len(self.rows), "cases": list(self.cases), "singularities": [s.to_dict() for s in self.singularities],
                "reasons": list(self.reasons), "open_items": self.open_items, "warnings": list(self.warnings)}


def _xyz(p: list[float] | None) -> str:
    return "–" if not p else "[" + "; ".join(f"{v:.4g}" for v in p) + "] mm"


# ------------------------------------------------------------------ cases
KNOWN_KINDS = ("mechanical", "thermal", "mixed")


def kind_of_step(study: Study, name: str, steps: dict[str, StepResult] | None = None) -> str:
    """"thermal" / "mechanical" / "mixed" (both kinds of loads in one state) by the loads of the load case of that name (a result
    reference such as "0" / "LoadCase_1" is resolved through `steps`). "unknown" when no load case of that name exists: the kind of the
    loads is then not known and the state is never factored as a purely mechanical one."""
    from ..study import ThermalLoad
    if steps is not None and name in steps:
        name = steps[name].name
    for lc in study.load_cases:
        if lc.name == name:
            th = any(isinstance(x, ThermalLoad) for x in lc.loads)
            mech = any(not isinstance(x, ThermalLoad) for x in lc.loads)
            return "mixed" if th and mech else "thermal" if th else "mechanical"
    return "unknown"


def _step_map(results: Results) -> dict[str, StepResult]:
    m: dict[str, StepResult] = {s.name: s for s in results.steps}
    for i, s in enumerate(results.steps):                  # the same references as combinations.combine_results
        m.setdefault(str(i), s)
        m.setdefault(f"LoadCase_{i + 1}", s)
    return m


def default_cases(study: Study, results: Results) -> tuple[list[MarginCase], list[str]]:
    """One case per load state (the preload state only inside combinations / user cases) and per non-envelope EN 1990 combination.
    The second item lists the cases that could not be formed (each one keeps the verdict from being PASS)."""
    warns: list[str] = []
    cases: list[MarginCase] = []
    names = _step_map(results)
    for s in results.steps:
        if is_preload(s.name):
            continue
        cases.append(MarginCase(s.name, [CaseComponent(s.name, 1.0, kind_of_step(study, s.name))]))
    if getattr(study, "combinations", None):
        for c in study.combinations:
            if c.envelope:
                continue                                    # the members of an envelope are cases of their own
            comps = [CaseComponent(k, float(v), kind_of_step(study, k, names)) for k, v in c.factors.items() if abs(float(v)) > 1e-15]
            bad = [x.source for x in comps if x.source not in names]
            if bad:
                warns.append(f"Combination '{c.name}' refers to unknown load states ({', '.join(bad)}) - not evaluated")
                continue
            if comps:
                cases.append(MarginCase(c.name, comps))
    return cases, warns


# ------------------------------------------------------------------ equivalent stress
def equivalent(S: np.ndarray, measure: str, p: np.ndarray | None = None) -> np.ndarray:
    """Equivalent stress of a design tensor field (N, 6) [MPa], never negative. `p` = principal(S) when the caller already has it (two measures of one tensor)."""
    if measure == "von_mises":
        return von_mises(S)
    if p is None:
        p = principal(S)
    if measure == "max_principal":
        return np.maximum(p[:, 0], 0.0)
    if measure == "max_shear":
        return 0.5 * (p[:, 0] - p[:, 2])
    raise ValueError(f"Unknown stress measure '{measure}'")


def _allowable_value(mat, rec: MaterialAllowable | None, name: str) -> tuple[float | None, list[str]]:
    """(value, reasons that make the margin indicative). The material strength Re / Rm replaces a missing Fy / Fu (indicative only)."""
    reasons: list[str] = []
    if rec is not None and rec.value(name) is not None:
        if not rec.is_confirmed():
            reasons.append(f"the allowables of '{mat.name}' are not confirmed")
        elif not rec.is_verified():
            reasons.append(f"the allowables of '{mat.name}' are not of a verified basis (A or B) or are incomplete")
        return rec.value(name), reasons
    fb = FALLBACK_ALLOWABLE.get(name)
    v = getattr(mat, fb, None) if fb else None
    if v is not None and v > 0:
        reasons.append(f"material '{mat.name}' has no confirmed {name} allowable - {fb} of the material used (indicative)")
        return float(v), reasons
    reasons.append(f"material '{mat.name}' has no {name} allowable")
    return None, reasons


def _part_nodes(mesh, pi: int) -> np.ndarray:
    from ..assessment import part_nodes
    return part_nodes(mesh, pi)


# ------------------------------------------------------------------ engine
class _Engine:
    def __init__(self, study: Study, results: Results, extra: tuple[ScalarSource, ...], max_places: int):
        self.study, self.res = study, results
        self.space: SpaceSettings = study.space or SpaceSettings()
        self.rs: RequirementSet | None = self.space.requirement_set
        self.rs_confirmed = self.space.requirement_confirmed()           # content AND structure classes as confirmed
        self._memo: dict = {}          # `Classifier.memo` of every classifier of this evaluation: one study, one result
        self.mesh = results.mesh
        self.n = len(self.mesh.nodes)
        self.steps = _step_map(results)
        self.extra = extra
        self.max_places = max_places
        self.reasons: list[str] = []
        self.warnings: list[str] = []
        self.rows: list[MarginRow] = []
        self.sing: list[SingularityRow] = []
        self.linear = True

    # ---- gate
    def gate(self) -> None:
        rs = self.rs
        if rs is None:
            self.reasons.append("no requirement set: margins are computed without design factors (indicative only)")
        elif not self.rs_confirmed:
            self.reasons.append("the requirement set is not confirmed (or it, or the structure classes of the parts, were edited after the "
                                "confirmation)")
        try:
            if self.study.is_nonlinear(mesh=self.mesh):
                self.linear = False
                self.reasons.append("nonlinear analysis: scaling the stress by the design factors is not valid - apply the factors to the "
                                    "loads and solve again")
        except Exception as e:  # noqa: BLE001 - a failed check must not stop the margins, it blocks the verdict
            self.linear = False
            self.reasons.append(f"could not decide whether the analysis is linear ({e})")
        if not self.res.steps:
            self.reasons.append("the results contain no load state")
        st = self.study
        if getattr(getattr(st, "frame", None), "active", False):
            self.reasons.append("beam / frame members are not assessed (the margins cover solid parts only)")
        if getattr(getattr(st, "shell", None), "enabled", False):
            self.reasons.append("shell model: plate and sheet parts are not assessed by the solid-stress margins")
        bolts = st.active_bolts()
        if bolts:
            self.reasons.append("bolts are not assessed (" + ", ".join(b.name for b in bolts[:6]) + ("..." if len(bolts) > 6 else "") + ")")

    # ---- rules
    def rule_for(self, cls: str, check: str) -> MarginRule | None:
        """The rule of (class, check) of the requirement set; None = the set has none (the check is then not decided)."""
        rs = self.rs if self.rs is not None else ecss_32_10c_template()
        r = rs.rule(check, cls)
        if r is not None and self.rs is None:             # indicative: von Mises, factors 1 (the template object is a fresh copy)
            r.stress_measure = "von_mises"
        return r

    def _declared_na(self, rule: MarginRule, plist: list[tuple[str, np.ndarray]], case_name: str, cls: str) -> None:
        """A rule the user declared not applicable counts only with its provenance in a confirmed set (otherwise the parts stay unassessed)."""
        rs = self.rs
        why = ""
        if not rule.provenance_ok:
            why = f"the '{rule.check}' rule of '{cls}' is declared not applicable without document / revision / clause"
        elif rs is None or not self.rs_confirmed:
            why = f"the '{rule.check}' rule of '{cls}' is declared not applicable, but the requirement set is not confirmed"
        if why:
            for pname, _nodes in plist:
                self._not_assessed(pname, case_name, cls, self._material_name(pname), [why], (rule.check,))
            return
        msg = f"'{rule.check}' check of the structure class '{cls}' declared not applicable ({rule.document}, {rule.revision}, {rule.clause})"
        if msg not in self.warnings:
            self.warnings.append(msg)

    def resolve(self, rule: MarginRule) -> tuple[ResolvedRule | None, list[str]]:
        """(resolved rule, reasons). Without a requirement set J = 1 (indicative)."""
        if self.rs is None:
            return None, ["no requirement set"]
        rr = resolve_rule(self.rs, rule)
        reasons = list(rr.missing)
        if not self.rs_confirmed:
            reasons.append("requirement set not confirmed")
        return rr, reasons

    # ---- stress of a case
    def case_tensors(self, case: MarginCase) -> tuple[dict[str, np.ndarray], list[str]] | None:
        """(tensor sums per kind of the components of a case, notes); None when a component cannot be found (a reason is recorded).
        A component of an unknown kind is treated as mixed (the larger factor) and noted - never as a purely mechanical one."""
        out: dict[str, np.ndarray] = {}
        notes: list[str] = []
        if not case.components:
            self._cannot(f"Case '{case.name}' has no component - not evaluated")
            return None
        for c in case.components:
            st = self.steps.get(c.source)
            if st is None or len(st.S) != self.n:
                self._cannot(f"Case '{case.name}': load state '{c.source}' is not in the results - not evaluated")
                return None
            if not math.isfinite(float(c.scale)):
                self._cannot(f"Case '{case.name}': the scale of '{c.source}' is not a finite number - not evaluated")
                return None
            kind = c.kind
            if kind not in KNOWN_KINDS:
                notes.append(f"the kind of the loads of '{c.source}' is not known ('{c.kind}') - treated as mechanical + thermal (larger factor)")
                kind = "mixed"
            out[kind] = out.get(kind, 0.0) + float(c.scale) * np.asarray(st.S, float)
        return out, notes

    def _cannot(self, msg: str) -> None:
        """A case / source that cannot be evaluated: a warning AND a reason (the margins of that case are missing, so no PASS)."""
        self.warnings.append(msg)
        if msg not in self.reasons:
            self.reasons.append(msg)

    def design_tensor(self, tens: dict[str, np.ndarray], jm: float | None, jt: float | None, why: list[str]) -> np.ndarray:
        """Σ J_kind · S_kind. Missing factors count as 1 (indicative) and give reasons; mixed states use the larger of J_m, J_t."""
        j_m = 1.0 if jm is None else jm
        S = np.zeros((self.n, 6))
        if "mechanical" in tens:
            S += j_m * tens["mechanical"]
        if "thermal" in tens:
            if jt is None:
                why.append("thermal factor not defined (thermal handling of the rule)")
            S += (j_m if jt is None else jt) * tens["thermal"]
        if "mixed" in tens:
            why.append("a state with mechanical and thermal loads (or of an unknown kind) cannot be factored separately (the larger factor "
                       "is used) - split it into separate load cases")
            S += max(j_m, j_m if jt is None else jt) * tens["mixed"]
        return S

    @staticmethod
    def relieving_tensors(tens: dict[str, np.ndarray], rule: MarginRule, jm: float | None, jt: float | None = None) -> list[np.ndarray]:
        """The other combinations of ECSS-E-ST-32-10C Rev.2 Corr.1 4.3.2.1 c for a case with a mechanical and a thermal part (every thermal handling: "same_as_mechanical",
        "separate" and "none"; J_t = the thermal factor the handling implies): J_m S_mech + 1,0 S_thermal (a relieving thermal load) where J_m > 1, and 1,0 S_mech + J_t S_thermal
        (a relieving mechanical load against a dominant thermal one) where J_t > 1; the combination J_m S_mech + J_t S_thermal itself is the main one. Empty when the rule or the
        case does not call for it. Whether a load relieves or adds is not known per place, so the margin takes the largest of all combinations."""
        if rule.thermal_handling not in ("same_as_mechanical", "separate", "none") or jm is None or jt is None or "mechanical" not in tens or "thermal" not in tens:
            return []
        out: list[np.ndarray] = []
        if jt > 1.0:                                                               # (J_m, 1,0); with J_t = 1,0 it is the main combination
            out.append(jm * tens["mechanical"] + tens["thermal"])
        if jm > 1.0:                                                               # (1,0, J_t); with J_m = 1,0 it is the main combination
            out.append(tens["mechanical"] + jt * tens["thermal"])
        return out

    # ---- main loop
    def run(self) -> None:
        self.gate()
        cases, warns = default_cases(self.study, self.res)
        for w in warns:
            self._cannot(w)
        names = {c.name for c in cases}
        for c in self.space.cases:
            if c.name in names:
                self._cannot(f"User case '{c.name}' has the name of another case - not evaluated")
            else:
                cases.append(c)
                names.add(c.name)
        if self.res.steps and not cases:
            self.reasons.append("no load state to evaluate")
        parts = self._parts()
        for case in cases:
            got = self.case_tensors(case)
            if got is None:
                continue
            self._eval_case(case.name, got[0], None, parts, got[1])
        for src in self.extra:
            if len(src.eq) != self.n:
                self._cannot(f"Source '{src.name}' does not match the mesh ({len(src.eq)} values for {self.n} nodes) - not evaluated")
                continue
            if src.kind not in KNOWN_KINDS:
                self._cannot(f"Source '{src.name}': the kind '{src.kind}' is not known - not evaluated")
                continue
            self._eval_case(src.name, {}, src, parts, [])
        self._buckling()
        if not self.rows and not any("no load state" in r for r in self.reasons):
            self.reasons.append("no margin could be computed (check the materials, parts and rules)")

    def _parts(self) -> list[tuple[int, str, np.ndarray]]:
        out = []
        for pi, pname in enumerate(self.mesh.part_names):
            nodes = _part_nodes(self.mesh, pi)
            if len(nodes):
                out.append((pi, pname, nodes))
            else:
                self.reasons.append(f"part '{pname}' has no solid elements in the mesh - not assessed")
        return out

    def _eval_case(self, case_name: str, tens: dict[str, np.ndarray], scalar: ScalarSource | None,
                   parts: list[tuple[int, str, np.ndarray]], notes: list[str]) -> None:
        by_class: dict[str, list[tuple[str, np.ndarray]]] = {}
        for _pi, pname, nodes in parts:
            by_class.setdefault(self.space.class_of(pname), []).append((pname, nodes))
        eq_cache: dict[str, np.ndarray] = {}                  # measure -> equivalent stress of the purely mechanical tensor (J = 1) of this case
        for cls, plist in by_class.items():
            rules: list[tuple[MarginRule, ResolvedRule | None, list[str]]] = []
            for chk in PRIMARY_CHECKS:                     # both checks must be decided for every class present
                rule = self.rule_for(cls, chk)
                if rule is None:
                    for pname, _nodes in plist:
                        self._not_assessed(pname, case_name, cls, self._material_name(pname),
                                           [f"the requirement set has no '{chk}' rule for the structure class '{cls}' (not assessed)"], (chk,))
                elif rule.not_applicable:
                    self._declared_na(rule, plist, case_name, cls)
                else:
                    rules.append((rule, *self.resolve(rule)))
            for rule, rr, rreasons in rules:
                jm = rr.factor if rr is not None else None
                jt = rr.thermal_factor if rr is not None else None
                why: list[str] = list(notes)
                measure = rule.stress_measure or "von_mises"
                if scalar is not None:
                    if measure != "von_mises":
                        for pname, _nodes in plist:
                            self._not_assessed(pname, case_name, cls, self._material_name(pname),
                                               [f"source '{scalar.name}' is an equivalent (von Mises) stress, the rule '{rule.check}' uses "
                                                f"'{measure}' (not assessed)"], (rule.check,))
                        continue
                    j = jt if scalar.kind == "thermal" else jm
                    if j is None:
                        why.append("design factor not defined")
                    vals_eq = (1.0 if j is None else j) * np.asarray(scalar.eq, float)
                    # the state has no tensor: a zero field without the (N, 6) memory (only the principal stresses of the few classified nodes are read from it)
                    st_res = StepResult(name=scalar.name, U=np.zeros((0, 3)), S=np.broadcast_to(np.zeros((1, 6)), (self.n, 6)))
                    st_res._mises = vals_eq                              # the classifier reads .mises of the state
                else:
                    S = self.design_tensor(tens, jm, jt, why)
                    vals_eq = self._equivalent(S, tens, jm, measure, eq_cache)
                    st_mises = vals_eq if measure == "von_mises" else None            # = von_mises(S): the classifier needs it, do not compute it again
                    for alt in self.relieving_tensors(tens, rule, jm, jt):        # a relieving load counts with 1,0 (ECSS-E-ST-32-10C 4.3.2.1 c): the largest stress governs
                        vals_eq = np.maximum(vals_eq, equivalent(alt, measure))
                    st_res = StepResult(name=case_name, U=np.zeros((0, 3)), S=S)
                    st_res._mises = st_mises
                cl = Classifier(self.study, self.res, st_res, np.zeros(self.n), self._memo)
                for pname, nodes in plist:
                    self._eval_part(pname, nodes, cls, case_name, rule, jm, jt, list(rreasons) + why, vals_eq, cl)

    @staticmethod
    def _equivalent(S: np.ndarray, tens: dict[str, np.ndarray], jm: float | None, measure: str, cache: dict[str, np.ndarray]) -> np.ndarray:
        """equivalent(S, measure) of the design tensor S. A purely mechanical state is S = J_m·S_mech and every measure of `equivalent` is positively homogeneous
        (von Mises, max principal, max shear), so the equivalent stress of S_mech is computed once per case and measure and scaled by J_m (the eigenvalues of the principal
        stresses dominated the margins; the result differs from the direct evaluation by round-off only, exactly for J_m = 1)."""
        j = 1.0 if jm is None else float(jm)
        if tuple(tens) != ("mechanical",) or not (math.isfinite(j) and j > 0.0):
            return equivalent(S, measure)
        base = cache.get(measure)
        if base is None:
            p = None
            if measure != "von_mises":                      # max principal and max shear of one case share the eigenvalues (once per case)
                p = cache.get("principal")
                if p is None:
                    p = cache["principal"] = principal(tens["mechanical"])
            base = cache[measure] = equivalent(tens["mechanical"], measure, p)
        return base if j == 1.0 else j * base

    def _material_name(self, part: str) -> str:
        try:
            return self.study.material_of(part).name
        except KeyError:
            return ""

    def _not_assessed(self, part: str, case: str, cls: str, material: str, reasons: list[str],
                      checks: tuple[str, ...] = PRIMARY_CHECKS) -> None:
        for chk in checks:
            self.rows.append(MarginRow(part, case, chk, cls, material, "", None, None, None, None, None, None, None, None, "", "unconfirmed",
                                       False, list(reasons)))

    def _eval_part(self, pname: str, nodes: np.ndarray, cls: str, case_name: str, rule: MarginRule, jm: float | None, jt: float | None,
                   reasons: list[str], vals_eq: np.ndarray, cl: Classifier) -> None:
        try:
            mat = self.study.material_of(pname)
        except KeyError:
            self._not_assessed(pname, case_name, cls, "", ["the part has no material"])
            return
        allow, a_reasons = _allowable_value(mat, allowable_of(mat), rule.allowable or DEFAULT_ALLOWABLE.get(rule.check, ""))
        reasons = reasons + a_reasons
        if getattr(mat, "is_ortho", False) or getattr(mat, "is_print", False):
            reasons.append(f"material '{mat.name}' is orthotropic / printed: the isotropic equivalent stress does not apply")
        if not self.linear:
            reasons.append("nonlinear analysis")
        if allow is None:
            self.rows.append(MarginRow(pname, case_name, rule.check, cls, mat.name, rule.allowable, None, jm, jt, None, None, None, None,
                                       None, "", "unconfirmed", False, list(dict.fromkeys(reasons))))
            return
        bad = int(np.count_nonzero(~np.isfinite(vals_eq[nodes])))
        if bad:                                           # a NaN / inf must never vanish in a maximum
            reasons.append(f"the stress of part '{pname}' is not finite at {bad} node(s) - the margin cannot be determined")
            self.rows.append(MarginRow(pname, case_name, rule.check, cls, mat.name, rule.allowable, allow, jm, jt, None, None, None, None,
                                       None, "", "unconfirmed", False, list(dict.fromkeys(reasons))))
            return
        ratio = np.zeros(self.n)
        ratio[nodes] = vals_eq[nodes] / allow
        raw_max = float(ratio[nodes].max())
        mos_raw = (1.0 / raw_max - 1.0) if raw_max > 0 else math.inf
        cl.vals = ratio
        places, gov = scan_maxima(cl, max_places=self.max_places, tags=True, nodes=nodes)
        for pl in places:
            if pl["classification"] == "singularity":
                self._add_singularity(pname, rule.check, case_name, pl, cl.r_cluster, ratio)
        if gov is None:
            reasons.append(f"the {len(places)} highest places are all singularities - the margin outside singularities is undetermined")
            self.rows.append(MarginRow(pname, case_name, rule.check, cls, mat.name, rule.allowable, allow, jm, jt, None, None, mos_raw, None,
                                       None, "singularity", "unconfirmed", False, list(dict.fromkeys(reasons))))
            return
        n = int(gov["node"]) - 1
        r = float(ratio[n])
        mos = (1.0 / r - 1.0) if r > 0 else math.inf
        reasons = list(dict.fromkeys(reasons))
        status = "unconfirmed" if reasons else ("fail" if mos < 0 else "pass")
        self.rows.append(MarginRow(pname, case_name, rule.check, cls, mat.name, rule.allowable, allow, jm, jt, float(vals_eq[n]), mos, mos_raw,
                                   n + 1, [float(v) for v in self.mesh.nodes[n]], gov["classification"], status,
                                   bool(reasons and mos < 0), reasons))

    def _add_singularity(self, part: str, check: str, case: str, place: dict, radius: float, ratio: np.ndarray) -> None:
        xyz = [float(v) for v in place["xyz_mm"]]
        r = float(place["value"])
        mos_raw = (1.0 / r - 1.0) if r > 0 else math.inf
        tags = place.get("tags") or []
        cause = ", ".join(tags) if tags else (place.get("reasons") or ["singularity"])[0]
        for s in self.sing:
            if s.part == part and math.dist(s.location, xyz) <= radius:
                if case not in s.cases:
                    s.cases.append(case)
                if check not in s.checks:
                    s.checks.append(check)
                if s.mos_raw is None or mos_raw < s.mos_raw:
                    s.mos_raw, s.location = mos_raw, xyz
                return
        self.sing.append(SingularityRow(part, xyz, cause, mos_raw, [case], [check]))

    # ---- buckling
    def _buckling(self) -> None:
        modes = [m for m in getattr(self.res, "modes", []) if getattr(m, "kind", "") == "buckling"]
        if not modes:
            return
        rs = self.rs if self.rs is not None else ecss_32_10c_template()
        rules = [r for r in rs.rules if r.check == "buckling"]
        if not rules:                                     # buckling results exist, the set does not decide them: never silently PASS
            self.reasons.append("the results contain buckling modes, but the requirement set has no 'buckling' rule (not assessed)")
        for rule in rules:
            if rule.not_applicable:
                if rule.provenance_ok and self.rs_confirmed:
                    self.warnings.append(f"'buckling' check declared not applicable ({rule.document}, {rule.revision}, {rule.clause})")
                else:
                    self.reasons.append("the 'buckling' rule is declared not applicable, but without provenance or in an unconfirmed set")
                continue
            rr, reasons = self.resolve(rule)
            reasons = list(reasons)
            if not self.linear:
                reasons.append("nonlinear analysis")
            j = rr.factor if rr is not None else None
            for m in modes:
                lam = [float(v) for v in m.values if math.isfinite(float(v)) and float(v) > 0]
                if not lam:
                    self._cannot(f"Buckling case '{m.case or m.name}' has no positive load factor - not evaluated")
                    continue
                lam1 = min(lam)
                mos = lam1 / (1.0 if j is None else j) - 1.0
                rs_reasons = list(dict.fromkeys(reasons + (["design factor not defined"] if j is None else [])))
                status = "unconfirmed" if rs_reasons else ("fail" if mos < 0 else "pass")
                self.rows.append(MarginRow("(structure)", f"buckling: {m.case or m.name}", "buckling", rule.structure_class, "",
                                           BUCKLING_ALLOWABLE, lam1, j, None, None, mos, mos, None, None, "", status,
                                           bool(rs_reasons and mos < 0), rs_reasons))

    # ---- result
    def summary(self) -> MarginSummary:
        sp = self.space
        for s in self.sing:
            s.justification = match_justification(sp, s.part, s.location)
        fail = any(r.status == "fail" for r in self.rows)
        unconf = bool(self.reasons) or any(r.status == "unconfirmed" for r in self.rows) or not self.rows
        verdict = FAIL if fail else UNCONFIRMED if unconf else PASS
        rs = self.rs
        via = rs.confirmed_via if rs is not None and self.rs_confirmed else ""
        av = {}
        for m in self.study.materials.values():
            rec = allowable_of(m)
            if rec is not None and rec.is_confirmed():
                av[m.name] = rec.confirmed_via
        if via == "mcp":
            self.warnings.append("the requirement set was confirmed through the MCP interface (an agent, not the GUI) - the user should check it "
                                 "and confirm it in the GUI")
        for k, x in av.items():
            if x == "mcp":
                self.warnings.append(f"the allowables of '{k}' were confirmed through the MCP interface (an agent, not the GUI) - the user should "
                                     f"check them and confirm them in the GUI")
        reasons = list(self.reasons)
        for r in self.rows:
            for x in r.reasons:
                if x not in reasons and r.status == "unconfirmed":
                    reasons.append(x)
        return MarginSummary(verdict=verdict, rows=self.rows, singularities=self.sing, reasons=reasons[:30], warnings=self.warnings,
                             requirement_set=rs.name if rs else "", standard=rs.standard if rs else "",
                             revision=rs.revision if rs else "", approach=rs.verification_approach if rs else "",
                             confirmed=bool(rs and self.rs_confirmed), confirmed_by=rs.confirmed_by if rs else "",
                             date=rs.date if rs else "", cases=sorted({r.case for r in self.rows}), confirmed_via=via, allowables_via=av)


def compute_margins(study: Study, results: Results, *, extra_sources: tuple[ScalarSource, ...] = (),
                    max_places: int = MAX_PLACES) -> MarginSummary:
    """Margins of safety of `results` for the requirement set in `study.space` (see the module docstring).
    `extra_sources` = stress envelopes from outside (dynamics); each forms a case of its own."""
    eng = _Engine(study, results, tuple(extra_sources), max_places)
    eng.run()
    return eng.summary()


# ------------------------------------------------------------------ justifications
def match_justification(space: SpaceSettings, part: str, xyz: list[float]) -> Justification | None:
    """The justification note of the same part within its tolerance of the place (the nearest one)."""
    best: tuple[float, Justification] | None = None
    for j in space.justifications:
        if j.part != part:
            continue
        d = math.dist(j.xyz, xyz)
        if d <= max(j.tolerance_mm, 0.0) and (best is None or d < best[0]):
            best = (d, j)
    return best[1] if best else None


def add_justification(space: SpaceSettings, part: str, xyz: list[float], note: str, *, cause: str = "", author: str = "",
                      tolerance_mm: float = 5.0, date: str | None = None) -> Justification:
    """Stores / replaces the note of a singular place (an existing note of the same place is replaced)."""
    if not note.strip():
        raise ValueError("The justification note is empty")
    if not all(math.isfinite(float(v)) for v in xyz) or len(xyz) != 3:
        raise ValueError("The location must be [x, y, z]")
    old = match_justification(space, part, [float(v) for v in xyz])
    if old is not None:
        space.justifications.remove(old)
    j = Justification(part=part, xyz=[float(v) for v in xyz], note=note.strip(), tolerance_mm=float(tolerance_mm), cause=cause,
                      author=author, date=date or _dt.date.today().isoformat())
    space.justifications.append(j)
    return j

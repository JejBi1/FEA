"""Shared types of the FEM model quality checks (F14 stage S3): the result of one check, the limits with their provenance, small numerical helpers.

Standing rule of the stage (F14 plan section 1): the program ships NO numeric limit of a standard. Every tolerance that turns a computed number into a verdict
is a named entry of `ModelCheckLimits`; an entry has a value and a `confirmed` flag (+ document / revision / clause / who / when, the same fields as the
requirement set of stage S1) and is empty in the template. A check whose limit is missing or unconfirmed still computes and reports its numbers, but its status
is `UNCONFIRMED` – never `PASS`. What the checks decide WITHOUT a limit are physical invariants only (a free body has exactly six rigid-body modes, a result must
be in equilibrium to the solver's own acceptance tolerance `config.EQUILIBRIUM_TOL`, the mass of a part is positive), and those can only produce `FAIL`.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

PASS, WARN, FAIL, UNCONFIRMED, SKIPPED = "PASS", "WARN", "FAIL", "UNCONFIRMED", "SKIPPED"
STATUSES = (PASS, WARN, FAIL, UNCONFIRMED, SKIPPED)

# the seven checks of ECSS-E-ST-32-11C 6.4.1 and common FEM practice model quality (order = number of the check)
CHECK_IDS = ("mass", "free_free", "unit_displacement", "isothermal", "effective_mass", "mesh_convergence", "equilibrium")
CHECK_TITLES = {
    "mass": "Mass and centre of mass",
    "free_free": "Free-free modes and rigid-body behaviour",
    "unit_displacement": "Unit enforced displacement (ground check)",
    "isothermal": "Isothermal thermal expansion",
    "effective_mass": "Effective modal mass completeness",
    "mesh_convergence": "Mesh convergence summary",
    "equilibrium": "Global force and moment equilibrium",
}

# every tolerance of the checks: name -> what it limits (no value anywhere in the program)
LIMIT_DESCRIPTIONS: dict[str, str] = {
    "mass_tol_rel": "relative tolerance of the FE total mass against the mass budget (check 1)",
    "cog_tol_mm": "tolerance of the FE centre of mass against the budget position [mm] (check 1)",
    "cad_mass_tol_rel": "relative tolerance of the FE mass of a part against its CAD volume x density (check 1)",
    "part_mass_tol_rel": "relative tolerance of the FE mass of a part against its mass budget (check 1)",
    "solver_mass_tol_rel": "relative tolerance of the mass seen by the solver (unit-gravity reaction) against the integrated mass (check 1)",
    "rigid_energy_ratio_max": "largest strain energy of a rigid-body mode relative to the first elastic mode (check 2)",
    "rigid_fit_tol": "largest share of a rigid-body mode that is not a rigid-body motion (check 2)",
    "unit_disp_energy_ratio_max": "largest strain energy of a unit enforced displacement relative to E*Lc*delta^2/2 (check 3)",
    "unit_disp_reaction_ratio_max": "largest net interface reaction (force / moment) of a unit enforced displacement relative to E*Lc*delta (check 3)",
    "isothermal_stress_ratio_max": "largest von Mises stress of the uniform temperature change relative to E*alpha*dT (check 4)",
    "thermal_support_reaction_ratio_max": "largest net support reaction under a uniform temperature change relative to E*alpha*dT*A (check 4)",
    "eff_mass_min_fraction": "smallest cumulative effective mass fraction per translation axis up to the cutoff (check 5)",
    "eff_mass_cutoff_hz": "cutoff frequency up to which the effective mass is summed [Hz] (check 5)",
    "freq_convergence_tol": "largest relative change of a natural frequency with the last mesh refinement (check 6)",
    "stress_convergence_tol": "largest relative change of the governing stress with the last mesh refinement (check 6)",
    "equilibrium_tol": ("largest relative imbalance of forces and moments (check 7): |applied + reactions| divided by the RESULTANT max(|applied|, |reactions|) when a net force "
                        "(moment) is applied, by the sum of the nodal force (moment) norms of the solver record for a self-equilibrated state (pure moment, temperature change)"),
    "thermal_support_stress_ratio_max": "largest von Mises stress of the supported model under a uniform temperature change relative to E*alpha*dT, judged when the study has one support group (check 4)",
}


@dataclass
class LimitEntry:
    """One limit with its provenance. `value` is only used when `confirmed` is True."""
    value: float | None = None
    confirmed: bool = False
    document: str = ""
    revision: str = ""
    clause: str = ""
    confirmed_by: str = ""
    date: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.confirmed) and self.value is not None


@dataclass
class ModelCheckLimits:
    """The tolerances of the model checks. The empty template has no value; fill it from the confirmed requirement set (`from_entries`)."""
    entries: dict[str, LimitEntry] = field(default_factory=lambda: {k: LimitEntry() for k in LIMIT_DESCRIPTIONS})

    @classmethod
    def template(cls) -> "ModelCheckLimits":
        """The empty template: every name, no value, nothing confirmed."""
        return cls()

    @classmethod
    def from_entries(cls, data: dict[str, Any] | None, *, trusted: bool = False) -> "ModelCheckLimits":
        """{name: number | {"value", "confirmed", "document", "revision", "clause", "confirmed_by", "date"}}. A value from a caller is only a value: it is never
        confirmed (a bare number or a dict that says `confirmed: true` does not make a limit usable), so the check that uses it stays UNCONFIRMED. A limit is
        confirmed only through the confirmed requirement set (`from_requirement_set`). `trusted=True` takes the caller's `confirmed` flag as it is and is meant for
        tests and verification cases that enter example limits. An unknown name is an error (a typo must not silently leave a check without its limit)."""
        out = cls()
        for name, raw in (data or {}).items():
            if name not in LIMIT_DESCRIPTIONS:
                raise ValueError(f"Unknown model-check limit '{name}' (known: {', '.join(LIMIT_DESCRIPTIONS)})")
            if raw is None:
                continue
            if isinstance(raw, dict):
                val = raw.get("value")
                out.entries[name] = LimitEntry(None if val is None else float(val), trusted and bool(raw.get("confirmed", False)), str(raw.get("document", "")),
                                               str(raw.get("revision", "")), str(raw.get("clause", "")), str(raw.get("confirmed_by", "")),
                                               str(raw.get("date", "")))
            else:
                out.entries[name] = LimitEntry(float(raw), trusted, document="entered directly")
        return out

    @classmethod
    def from_requirement_set(cls, rs: Any, confirmed: bool | None = None) -> "ModelCheckLimits":
        """The limits of a CONFIRMED requirement set (stage S1, `RequirementSet.limits`): an entry is usable when it has a value and its document / revision / clause.
        An unconfirmed (or edited-after-confirmation) set, or None, gives the empty template: every check is then UNCONFIRMED. Names of other stages are ignored.
        `confirmed` = the verdict of `SpaceSettings.requirement_confirmed()` (the set is confirmed AND the structure classes are those it was confirmed for); without it only the
        set itself is asked."""
        out = cls()
        if rs is None or not (rs.is_confirmed() if confirmed is None else confirmed):
            return out
        for x in rs.limits:
            if x.key not in LIMIT_DESCRIPTIONS or x.value is None or not math.isfinite(x.value):
                continue
            if not (x.document.strip() and x.revision.strip() and x.clause.strip() and x.clause.strip() != "<clause>"):
                continue
            out.entries[x.key] = LimitEntry(float(x.value), True, x.document, x.revision, x.clause, rs.confirmed_by, rs.date)
        return out

    def value(self, name: str) -> float | None:
        """The confirmed value of a limit, None when it is missing or not confirmed."""
        e = self.entries.get(name)
        return e.value if e is not None and e.usable else None

    def provenance(self, name: str) -> dict[str, Any]:
        e = self.entries.get(name, LimitEntry())
        return {"name": name, "description": LIMIT_DESCRIPTIONS.get(name, ""), **asdict(e)}

    def to_dict(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in self.entries.items()}


@dataclass
class CheckResult:
    """The outcome of one check: a status, the numbers behind it, the limits it used and what to do about a problem."""
    id: str
    number: int
    title: str
    status: str
    summary: str
    values: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, dict[str, Any]] = field(default_factory=dict)      # limits that were consulted: provenance + whether they decided
    findings: list[str] = field(default_factory=list)                     # why FAIL / WARN: actionable diagnostics
    notes: list[str] = field(default_factory=list)                        # what was done, what was not covered
    elapsed_s: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)                 # the settings of THIS check that differ from the defaults (budgets included), as they were when it ran
    finished: str = ""                                                    # when this check finished ("YYYY-MM-DD HH:MM"): a kept run may merge checks of several runs

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


# a local path in a message (an exception text of a solver run, a file the geometry reader could not open): the answer and the report must not carry it
_SEG = r"[^\\/\s:*?\"<>|;',)]"                            # a character of a path part that is not a separator, a space or a punctuation of the text around
# a directory part may hold single spaces inside ("work dir") but neither starts nor ends with one: " / " or " - x /" in a sentence is not a separator of a path
_PATH_RE = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\|(?<![\w)\].:/])/(?=[\w.\-]+/[\w.\-]))(?:" + _SEG + r"+(?: +" + _SEG + r"+)*[\\/])*" + _SEG + "*")


def scrub_paths(text: str) -> str:
    """The text with every local path (Windows drive, UNC, absolute POSIX path of two or more parts) replaced by `<path>/` and its last part, e.g. a file name; a path that ends
    with a folder separator (a folder only: its name may be the name of a person) by `<path>` alone."""
    def one(m: "re.Match[str]") -> str:
        if m.group(0)[-1] in "\\/":
            return "<path>"
        last = re.split(r"[\\/]", m.group(0))[-1]
        return f"<path>/{last}" if last else "<path>"
    return _PATH_RE.sub(one, text)


def jsonable(x: Any) -> Any:
    """Plain JSON types from numpy scalars / arrays (nan / inf become strings: JSON has none)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else ("nan" if np.isnan(v) else ("inf" if v > 0 else "-inf"))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Judge:
    """Collects the reasons of a verdict. `violate` = a physical invariant or a confirmed limit is broken (FAIL); `warn` = suspicious (WARN); `need(limit)` records a
    limit that the verdict would need – when it is missing or unconfirmed the status cannot be PASS."""

    def __init__(self, limits: ModelCheckLimits) -> None:
        self.limits = limits
        self.violations: list[str] = []
        self.warnings: list[str] = []
        self.missing: list[str] = []
        self.undecided: list[str] = []                 # things that keep the check from a verdict although no limit is missing (an input of the check is not decided)
        self.used: dict[str, dict[str, Any]] = {}

    def need(self, name: str) -> float | None:
        """The confirmed value of the limit, or None (and the limit is listed as missing)."""
        v = self.limits.value(name)
        prov = self.limits.provenance(name)
        prov["decided"] = v is not None
        self.used[name] = prov
        if v is None:
            self.missing.append(name)
        return v

    def violate(self, text: str) -> None:
        self.violations.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def cannot_decide(self, text: str) -> None:
        """The check ran, but something it needs to be conclusive is not decided (an interface that is not named, a state that was not examined): never PASS."""
        self.undecided.append(text)

    def compare_max(self, value: float, name: str, what: str) -> bool | None:
        """value <= limit? Records a violation when not; None when the limit is missing."""
        lim = self.need(name)
        if lim is None:
            return None
        ok = bool(np.isfinite(value) and value <= lim)
        if not ok:
            self.violate(f"{what} = {value:.3g} exceeds the limit {lim:.3g}")
        return ok

    def compare_min(self, value: float, name: str, what: str) -> bool | None:
        lim = self.need(name)
        if lim is None:
            return None
        ok = bool(np.isfinite(value) and value >= lim)
        if not ok:
            self.violate(f"{what} = {value:.3g} is below the limit {lim:.3g}")
        return ok

    def status(self) -> str:
        if self.violations:
            return FAIL
        if self.missing or self.undecided or not self.used:          # a check that compared nothing with a limit has decided nothing
            return UNCONFIRMED
        if self.warnings:
            return WARN
        return PASS

    def finish(self, number: int, summary_ok: str, values: dict[str, Any], notes: list[str], elapsed: float) -> CheckResult:
        st = self.status()
        cid = CHECK_IDS[number - 1]
        if st == FAIL:
            summary = "; ".join(self.violations[:3]) + (f" (+{len(self.violations) - 3} more)" if len(self.violations) > 3 else "")
        elif st == UNCONFIRMED:
            why = []
            if self.missing:
                why.append(f"limit(s) not confirmed: {', '.join(dict.fromkeys(self.missing))}")
            why += self.undecided
            if not self.used:
                why.append("nothing was compared with a limit")
            summary = f"{summary_ok} – no verdict: " + "; ".join(why)
        elif st == WARN:
            summary = f"{summary_ok} – {self.warnings[0]}"
        else:
            summary = summary_ok
        return CheckResult(cid, number, CHECK_TITLES[cid], st, scrub_paths(summary), jsonable(values), self.used,
                           [scrub_paths(t) for t in self.violations + self.warnings + (self.undecided if st == UNCONFIRMED else [])],
                           [scrub_paths(t) for t in dict.fromkeys(notes)], elapsed)


def skipped(number: int, reason: str, elapsed: float = 0.0, notes: list[str] | None = None) -> CheckResult:
    """A check that could not be run (not applicable to the study, a solver path that is not available) – never counted as passed."""
    cid = CHECK_IDS[number - 1]
    return CheckResult(cid, number, CHECK_TITLES[cid], SKIPPED, scrub_paths(reason), {}, {}, [scrub_paths(reason)], [scrub_paths(n) for n in (notes or [])], elapsed)


def overall_verdict(results: list[CheckResult]) -> str:
    """FAIL > INCOMPLETE (a check was skipped or not requested: fewer than the seven) > UNCONFIRMED_REQUIREMENTS > WARN > PASS."""
    st = [r.status for r in results]
    if FAIL in st:
        return "FAIL"
    if not results or SKIPPED in st or {r.id for r in results} != set(CHECK_IDS):       # a subset of the seven checks is not a model verification
        return "INCOMPLETE"
    if UNCONFIRMED in st:
        return "UNCONFIRMED_REQUIREMENTS"
    if WARN in st:
        return "WARN"
    return "PASS"


# ------------------------------------------------------------------------------------------------------------------ inputs of a run (not limits)
DEFAULT_DENSITY_RANGE = (1.0e-11, 2.3e-8)         # plausible density of a solid / core material [t/mm3]; a unit error shows up as 1000x outside
DEFAULT_CTE_RANGE = (1.0e-8, 1.0e-3)              # plausible coefficient of thermal expansion [1/K]


@dataclass
class ModelCheckOptions:
    """What the checks compare against and how they are run. None of these is a limit of a standard: budgets are the user's data, the rest are numerical settings."""
    # --- check 1: the budgets the FE model is compared with (the tolerance is `mass_tol_rel` / `cog_tol_mm` of the limits)
    mass_budget_t: float | None = None
    cog_budget_mm: tuple[float, float, float] | None = None
    part_mass_budget_t: dict[str, float] = field(default_factory=dict)
    solver_mass_check: bool = True                    # also the unit-gravity statics (the mass and the centre of mass as the SOLVER sees them); off = check 1 is not complete (never PASS)
    density_range_t_mm3: tuple[float, float] = DEFAULT_DENSITY_RANGE      # a density outside it is a warning; a WIDER range weakens the check (never a full PASS)
    cte_range_per_K: tuple[float, float] = DEFAULT_CTE_RANGE           # an expansion coefficient outside it is a warning; a WIDER range weakens the check (never a full PASS)
    # --- checks 3, 4
    interface_selections: list[str] | None = None    # the selections of the launch interface (None = every Fixed / Displacement support of the study)
    unit_states: tuple[str, ...] = ("UX", "UY", "UZ", "RX", "RY", "RZ")
    thermal_dT_K: float = 1.0                         # the uniform temperature change of the isothermal check (linear: the value only scales the numbers)
    thermal_alpha_per_K: float = 1.0e-5               # the one expansion coefficient given to every material in the isothermal check (a test value, not material data)
    # --- checks 2, 5, 6
    free_free_modes: int = 12                         # modes of the free-free run (six rigid-body modes + the first elastic ones)
    modes: int | None = None                          # effective mass: modes of the constrained run (None = the study's own setting, else 30)
    max_modes: int = 200                              # with a cutoff frequency: the most modes computed below it
    convergence_frequencies: int = 5
    convergence_levels: int = 3
    convergence_ratio: float = 1.5
    convergence_case: str | int | None = None         # the load case of the stress convergence (None = the one with the highest stress)
    threads: int | None = None                        # threads of the solver processes (None = the program's budget)
    include_convergence: bool = True
    convergence_max_dof: float = 3.0e6                # the finest level may have this many DOF (a guard against a run that cannot finish)


# ------------------------------------------------------------------------------------------------------------------ validation of the options
# Numerical settings, NOT limits of a standard. The smallest element-size ratio of the convergence check: Richardson extrapolation / GCI of two meshes that differ by less than
# 30 % in element size has no usable order of convergence (the common recommendation for the grid convergence index is r >= 1.3).
MIN_CONVERGENCE_RATIO = 1.3
MAX_CONVERGENCE_RATIO = 4.0
UNIT_STATE_NAMES = ("UX", "UY", "UZ", "RX", "RY", "RZ")                 # the six unit enforced displacements of check 3 (three translations, three rotations)
INT_OPTION_BOUNDS = {"threads": (1, 64), "free_free_modes": (7, 200), "modes": (1, 500), "max_modes": (1, 500), "convergence_frequencies": (1, 20), "convergence_levels": (2, 6)}
FLOAT_OPTION_BOUNDS = {"convergence_ratio": (MIN_CONVERGENCE_RATIO, MAX_CONVERGENCE_RATIO), "convergence_max_dof": (1.0e4, 2.0e7), "thermal_alpha_per_K": (1.0e-7, 1.0e-3)}
MAX_THERMAL_STRAIN = 1.0e-2                                             # |alpha * dT| of the isothermal check: the check is linear, a large strain is not
# the checks (numbers 1..7) that an option influences: what is printed next to the verdict of each check
OPTION_CHECKS: dict[str, tuple[int, ...]] = {
    "mass_budget_t": (1,), "cog_budget_mm": (1,), "part_mass_budget_t": (1,), "solver_mass_check": (1, 7), "density_range_t_mm3": (1,), "cte_range_per_K": (4,),
    "interface_selections": (3, 4), "unit_states": (3,), "thermal_dT_K": (4,), "thermal_alpha_per_K": (4,), "free_free_modes": (2,), "modes": (5,), "max_modes": (5,),
    "convergence_frequencies": (6,), "convergence_levels": (6,), "convergence_ratio": (6,), "convergence_case": (6,), "include_convergence": (6,), "convergence_max_dof": (6,),
    "threads": (1, 2, 3, 4, 5, 6, 7)}


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _seq(v: Any) -> bool:
    """A list or a tuple (None, a string or a number is not a sequence of values)."""
    return isinstance(v, (list, tuple))


def option_problems(o: "ModelCheckOptions") -> list[str]:
    """What is wrong with the options (empty = usable): every number is finite and inside the bounds that keep a run from being meaningless or unable to finish. These are bounds of
    the NUMERICAL settings; a tolerance of a standard is never an option."""
    bad: list[str] = []
    for k, (lo, hi) in INT_OPTION_BOUNDS.items():
        v = getattr(o, k)
        if v is None and k in ("modes", "threads"):               # None = the study's own setting / the program's thread budget
            continue
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            bad.append(f"option {k} must be an integer from {lo} to {hi}")
    for k, (flo, fhi) in FLOAT_OPTION_BOUNDS.items():
        v = getattr(o, k)
        if not _finite(v) or not flo <= float(v) <= fhi:
            bad.append(f"option {k} must be a number from {flo:g} to {fhi:g}" + (f" (the smallest element-size ratio that a convergence study can judge is {MIN_CONVERGENCE_RATIO:g})" if k == "convergence_ratio" else ""))
    if not _finite(o.thermal_dT_K) or not 0.01 <= abs(float(o.thermal_dT_K)) <= 1000.0:
        bad.append("option thermal_dT_K must be a number with 0.01 <= |value| <= 1000 [K]")
    elif _finite(o.thermal_alpha_per_K) and abs(float(o.thermal_dT_K) * float(o.thermal_alpha_per_K)) > MAX_THERMAL_STRAIN:
        bad.append(f"thermal_dT_K x thermal_alpha_per_K must not exceed the strain {MAX_THERMAL_STRAIN:g} (the isothermal check is linear)")
    if o.mass_budget_t is not None and (not _finite(o.mass_budget_t) or not 0.0 < float(o.mass_budget_t) <= 1.0e7):
        bad.append("option mass_budget_t must be a positive number of tonnes (at most 1e7)")
    if o.cog_budget_mm is not None and (not _seq(o.cog_budget_mm) or len(o.cog_budget_mm) != 3 or not all(_finite(c) and abs(float(c)) <= 1.0e7 for c in o.cog_budget_mm)):
        bad.append("option cog_budget_mm must be three finite coordinates [x, y, z] in mm")
    if not isinstance(o.part_mass_budget_t, dict) or not all(isinstance(k, str) and k and _finite(v) and 0.0 < float(v) <= 1.0e7 for k, v in o.part_mass_budget_t.items()):
        bad.append("option part_mass_budget_t must be {part name: positive mass in t}")
    for k in ("density_range_t_mm3", "cte_range_per_K"):
        v = getattr(o, k)
        if not (_seq(v) and len(v) == 2 and all(_finite(c) for c in v) and 0.0 < float(v[0]) < float(v[1]) <= 1.0):
            bad.append(f"option {k} must be (low, high) with 0 < low < high <= 1")
    if not _seq(o.unit_states) or not all(isinstance(n, str) and n in UNIT_STATE_NAMES for n in o.unit_states) or len(set(o.unit_states)) != len(o.unit_states):
        bad.append(f"option unit_states must be a list of distinct names from {', '.join(UNIT_STATE_NAMES)}")
    if o.interface_selections is not None and not (isinstance(o.interface_selections, (list, tuple)) and all(isinstance(n, str) and n for n in o.interface_selections)):
        bad.append("option interface_selections must be a list of selection names")
    if o.convergence_case is not None and (isinstance(o.convergence_case, bool) or not isinstance(o.convergence_case, (str, int))):
        bad.append("option convergence_case must be a load case name or its index")
    for k in ("solver_mass_check", "include_convergence"):
        if not isinstance(getattr(o, k), bool):
            bad.append(f"option {k} must be true or false")
    return bad


def range_widened(rng: Any, default: tuple[float, float]) -> bool:
    """True when a plausibility range reaches beyond the default one at either end (the screening of the values is then weaker than the default)."""
    return bool(float(rng[0]) < default[0] or float(rng[1]) > default[1])


def _plain(v: Any) -> Any:
    return jsonable(list(v) if isinstance(v, tuple) else v)


def nondefault_options(o: "ModelCheckOptions", number: int | None = None) -> dict[str, Any]:
    """The options that differ from the defaults as plain data; with `number` only those that influence that check (1..7)."""
    default = ModelCheckOptions()
    out: dict[str, Any] = {}
    for f in fields(ModelCheckOptions):
        v, d = _plain(getattr(o, f.name)), _plain(getattr(default, f.name))
        if v != d and (number is None or number in OPTION_CHECKS.get(f.name, ())):
            out[f.name] = v
    return out


def describe_option(name: str, value: Any) -> str:
    """One option as text for the report and the panel (a mass in kg, a list joined)."""
    if name == "mass_budget_t" and _finite(value):
        return f"{float(value) * 1000.0:.6g} kg"
    if name == "part_mass_budget_t" and isinstance(value, dict):
        return ", ".join(f"{k}: {float(v) * 1000.0:.5g} kg" for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in value) or "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


@dataclass
class CheckContext:
    """Everything a check needs; `shared` carries results from one check to a later one (the mass for the effective mass, the statics for the equilibrium)."""
    study: Any
    mesh: Any
    limits: ModelCheckLimits
    options: ModelCheckOptions
    workdir: Any
    log: Any
    cancel: Any = None
    parts: Any = None
    study_dir: Any = None
    results: Any = None
    shared: dict[str, Any] = field(default_factory=dict)
    on_check: Any = None

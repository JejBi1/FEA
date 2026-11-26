"""Contacts of assemblies and nonlinear stepping (stages F8.5a/b, F8.7): Study.contacts, Study.contact_settings,
Study.analysis.

Realisation in ccx 2.23 (specification F8, 4.1–4.5; verified trials scratch/f8_contact, scratch/ccx_manual):
  - contact = *CONTACT PAIR, TYPE=SURFACE TO SURFACE (penalty, default) or MORTAR; N2S only as a backup;
  - the normal stiffness K and the slope of the sticking λ are ALWAYS written explicitly (the default of ccx takes the E of the first material in the input);
  - friction *FRICTION μ > 0 (μ = 0 is refused by ccx) – "without friction" in the history with a later friction = μ 1e-4;
  - interference = *CLEARANCE, VALUE = −δ_r (radially, overwrites the geometric gap of the whole pair);
  - nonlinear calculation: branches ([preload] + k-th state, in parallel) or a sequence (history, one job).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

CONTACT_TYPES = ("bonded", "no_separation", "frictionless", "frictional", "rough", "interference",
                 "compression_only", "none")
CONTACT_LABELS = {"bonded": "Bonded", "no_separation": "No separation (sliding)",
                  "frictionless": "Frictionless contact", "frictional": "Frictional contact", "rough": "No-slip contact",
                  "interference": "Interference (press fit)", "compression_only": "Compression only (support)",
                  "none": "No interaction"}
NONLINEAR_TYPES = ("no_separation", "frictionless", "frictional", "rough", "interference", "compression_only")
MU_NONE = 1e-4                    # "without friction" in a study where the friction is switched on later (*CHANGE FRICTION)
MU_ROUGH = 10.0                   # contact without slip = large μ
K_FACTOR = 10.0                   # default penalty stiffness K = K_FACTOR · min(E of the pair) (50 before 1.5: forces differ < 0.03 %, K6 25 % faster)
# an interference (press fit) keeps 50·E: the stresses and displacements of K4 are 0.03 % / 0.11 % from Lamé with 50·E against 0.24 % / 0.41 % with 10·E
# (the penetration p/K = 3.75e-5 mm is 0.4 % of the 0.01 mm interference); the nodal-pressure deviation of K4 (1.7 % against 0.05 %) is an averaging artefact
K_FACTOR_INTERFERENCE = 50.0
# Increment control of the steps without their own LoadStep (every branch, the automatic preload step): initial, minimum, maximum increment (share of the
# step) and the maximum number of increments. dt0 and dt_max are AUTOMATIC (None) by default – `branch_increments` chooses them by the formulation.
# Measured 17. 11. 2026 (ccx_static, K = 10·E): 0.25 / 0.5 gives on K6 17 % fewer Newton iterations (-12 % time) and +0.2 % in the forces of the
# slipping state (0.5 / 0.5: +1.7 %), K1 K1c K4 K6 K7 K7n R1 G1 KK2 pass – BUT the Hertz case K2 (mortar, hard contact) moves from 2.5 % to 5.7 %
# (limit 3 %): the mortar contact result depends on the number of increments (0.1 / 0.5 gives the same 5.7 %, so the maximum increment decides).
# Hence surface-to-surface and node-to-surface 0.25 / 0.5, mortar 0.1 / 0.25; a study without a contact (plasticity, NLGEOM, an always nonlinear
# run) keeps the former 0.1 / 0.25 (not measured with larger ones). LoadStep has its own values (a sequence has friction history, prescribed motion).
BRANCH_DT0_AUTO, BRANCH_DT_MAX_AUTO = 0.25, 0.5                # surface-to-surface, node-to-surface
BRANCH_DT0_SAFE, BRANCH_DT_MAX_SAFE = 0.1, 0.25                # mortar, no contact
BRANCH_DT_MIN, BRANCH_MAX_INC = 1e-5, 200


def _known(klass, d):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class ContactPair:
    """Contact pair between the parts or the face selections `a`, `b` (name of a selection, otherwise the name of a part = all its faces
    in the contact). slave = the finer mesh, on equality the softer material (None = automatically, otherwise "a" | "b").

    mu = coefficient of friction (None = from the library by the purpose: slip = μ_min for the transmission of the force by friction,
    resistance = μ_max for the resistance / the force of a drive); stiffness K [N/mm³] (None = 10·min(E) of the pair);
    stick_slope λ [N/mm³] (None = K/100); clearance [mm] RADIAL (−Δd/2 = interference) overwrites the geometric gap of the
    pair by one value; adjust = bring the slave nodes into contact (change of the coordinates of the mesh, never for an interference);
    initial = initial state from "Find touches" (contact | gap | penetration).
    """
    name: str
    a: str
    b: str
    kind: str = "frictionless"
    mu: float | None = None
    purpose: str = "slip"
    materials: str | None = None      # pair in the library of friction (e.g. "steel–steel dry")
    stiffness: float | None = None
    stick_slope: float | None = None
    clearance: float | None = None
    adjust: bool = False
    slave: str | None = None
    initial: str | None = None
    gap: float | None = None          # min. gap / penetration from the detection [mm] (negative = penetration)
    area: float | None = None         # contact area from detection [mm²]
    source: str = "user"              # auto | user
    enabled: bool = True
    conductance: float | None = None  # W/(m²·K) thermal contact conductance (stage F9.6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "ContactPair":
        return _known(cls, d)


@dataclass
class ContactSettings:
    """method: surface_to_surface (default, pressure ±0.1 %) | mortar (hard contact, interference) | node_to_surface
    (only for the convergence of the friction – pressure with tet10 is not usable). One formulation per study (ccx does not allow mixing).
    split: separate = the parts of the pair meshed separately without *TIE (non-conforming mesh) | conformal_split = conforming mesh
    + split of the interface (nodes opposite each other, bolted joints). stabilize = weak springs against free bodies.
    nlgeom = large deformations (snap-fit, cam); pressure = dead | following pressure (only with NLGEOM).
    """
    method: str = "surface_to_surface"
    detect_tol: float | None = None
    split: str = "separate"
    stabilize: str = "none"
    stabilize_k: float | None = None
    nlgeom: bool = False
    pressure: str = "dead"

    @classmethod
    def from_dict(cls, d) -> "ContactSettings":
        return _known(cls, d)


@dataclass
class Motion:
    """Prescribed motion of a rigid body / selection in the sequence (C2): target = a selection with a rigid constraint (remote
    constraint rigid) or a selection of faces; kind disp [mm] | rot [rad]; dof 1–3 in global axes (or csys);
    amplitude [[t, value], …] in the time of the step (0–1).
    """
    target: str
    kind: str = "disp"
    dof: int = 1
    amplitude: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.0]])
    csys: str | None = None

    @classmethod
    def from_dict(cls, d) -> "Motion":
        return _known(cls, d)


@dataclass
class LoadStep:
    """Step of the load history (nonlinear sequence). case = load state (None = only the prescribed motion / a change of contact),
    scale = multiple of the load of the state, ramp linear | step; bolts preload (preload of the bolts) | lock
    (length locked by *BOUNDARY, FIXED) | free; contacts_off = pairs switched off (*MODEL CHANGE); mu_change = {pair: μ}
    (*CHANGE FRICTION); prescribed = prescribed motions; control of the increments dt0, dt_min, dt_max (share of the step),
    max_inc; output_every = a frame every n increments; cycles ≥ 2 for fatigue with friction (repetition of the block).
    """
    name: str
    case: str | None = None
    scale: float = 1.0
    ramp: str = "linear"
    bolts: str = "lock"
    contacts_off: list[str] = field(default_factory=list)
    mu_change: dict = field(default_factory=dict)
    prescribed: list[Motion] = field(default_factory=list)
    dt0: float = 0.1
    dt_min: float = 1e-5
    dt_max: float = 0.25
    max_inc: int = 200
    direct: bool = False
    output_every: int = 1
    cycles: int = 1
    # implicit dynamics with contact (C3/D7, ccx *DYNAMIC HHT-α): time = duration of the step [s], dt0/dt_min/dt_max then
    # in seconds; alpha ∈ [−1/3, 0] (0 for impacts – specification KK4)
    dynamic: bool = False
    time: float = 1.0
    alpha: float = -0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "LoadStep":
        s = _known(cls, d)
        s.prescribed = [Motion.from_dict(m) for m in (d or {}).get("prescribed", [])]
        return s


@dataclass
class AnalysisSettings:
    """kind linear (today's superposition) | nonlinear. Nonlinear: branches = True → each load state separately
    after [bolt preload] (in parallel, without history); False → the sequence `sequence` (one job, history).
    """
    kind: str = "linear"
    branches: bool = True
    sequence: list[LoadStep] = field(default_factory=list)
    stop_keep: bool = True            # on non-convergence keep the last converted increment (partial result)
    initial_velocity: dict = field(default_factory=dict)   # {part: [vx, vy, vz] mm/s} – initial velocity (*DYNAMIC)
    # increment control of the steps WITHOUT their own LoadStep: every branch (load state after [bolt preload]) and the automatic
    # preload step of a sequence – initial / minimum / maximum increment as a share of the step, maximum number of increments.
    # dt0 and dt_max None = automatic by the formulation (see `branch_increments`). A step of the sequence has its own values
    # (LoadStep.dt0 …). Larger increments = fewer Newton iterations (the time of a nonlinear run is driven by their number), but
    # a friction slip path (and a mortar contact) is then followed more coarsely.
    dt0: float | None = None
    dt_min: float = BRANCH_DT_MIN
    dt_max: float | None = None
    max_inc: int = BRANCH_MAX_INC

    @property
    def nonlinear(self) -> bool:
        return self.kind == "nonlinear"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in (("dt0", None), ("dt_min", BRANCH_DT_MIN), ("dt_max", None), ("max_inc", BRANCH_MAX_INC)):
            if d[k] == v:                            # written only when changed / explicit (automatic and default values stay out of the file)
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d) -> "AnalysisSettings":
        a = _known(cls, d)
        a.sequence = [LoadStep.from_dict(s) for s in (d or {}).get("sequence", [])]
        return a


def effective_method(study) -> str:
    """The contact formulation that is really written to the ccx input: the chosen one, except mortar when this ccx cannot solve the
    mortar matrix (config.mortar_supported – the Windows CalculiX) → surface-to-surface with K = 50·E (check.py warns)."""
    from .. import config
    m = study.contact_settings.method
    if m == "mortar" and not config.mortar_supported():
        return "surface_to_surface"
    return m


def mortar_substituted(study) -> bool:
    """Mortar was chosen but surface-to-surface is written (see effective_method)."""
    return study.contact_settings.method == "mortar" and effective_method(study) != "mortar"


# Warning of the readiness check AND of the nonlinear run (the run alone is not checked when a finished mesh is taken over without the geometry):
# one text, so that the two do not produce duplicates in Results.warnings.
MORTAR_SUBSTITUTED_WARNING = (
    "Contacts: mortar is not available with this CalculiX build (in the Hertz test case K2 the mortar matrix assembled by the "
    "Windows CalculiX contained NaN entries and SPOOLES stopped with \"singular\") – surface-to-surface "
    "with K = 50·E is used instead (K2: peak pressure −1.2 %); check the penetration, or use the Linux version for mortar")


def method_record(study) -> dict:
    """{"chosen", "effective", "substituted"} – what the study asks for and what ccx computed (Results.info["nonlinear"]["contact_method"])."""
    eff = effective_method(study)
    return {"chosen": study.contact_settings.method, "effective": eff, "substituted": eff != study.contact_settings.method}


def auto_increments(study) -> tuple[float, float]:
    """(dt0, dt_max) of the automatic increment control: 0.25 / 0.5 for surface-to-surface and node-to-surface contact, 0.1 / 0.25 for mortar
    and for a study without an active contact (see the comment at BRANCH_DT0_AUTO)."""
    if study.active_contacts() and effective_method(study) != "mortar":
        return BRANCH_DT0_AUTO, BRANCH_DT_MAX_AUTO
    return BRANCH_DT0_SAFE, BRANCH_DT_MAX_SAFE


def branch_increments(study) -> dict:
    """EFFECTIVE increment control of the steps without their own LoadStep: {"dt0", "dt_min", "dt_max", "max_inc", "auto": [names of
    the automatic ones]}. An automatic value is clamped to an explicit neighbour (dt0 ≤ dt_max), so entering only one of them never
    gives an invalid pair."""
    an = study.analysis
    a0, am = auto_increments(study)
    auto = [k for k in ("dt0", "dt_max") if getattr(an, k) is None]
    dt0 = a0 if an.dt0 is None else an.dt0
    dtm = am if an.dt_max is None else an.dt_max
    try:
        if an.dt0 is None and an.dt_max is not None:
            dt0 = min(dt0, float(an.dt_max))
        if an.dt_max is None and an.dt0 is not None:
            dtm = max(dtm, float(an.dt0))
    except (TypeError, ValueError):
        pass                                          # a non-numeric value from a foreign file: check() reports it
    return {"dt0": dt0, "dt_min": an.dt_min, "dt_max": dtm, "max_inc": an.max_inc, "auto": auto}

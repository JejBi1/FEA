"""Bolted joints of a spacecraft structure: preload range, thermal preload change, gapping, slip, bolt tensile / shear / combined, thread pull-out and bearing
margins (F14 stage S4), built beside the VDI 2230 evaluation of `fea_core.joints`.

The chain follows ECSS-E-HB-32-23A Rev.1 (6 February 2023); the clauses are quoted per formula below (paraphrased, no text of the handbook is reproduced). The
handbook's safety factor sf multiplies the EXTERNAL load only, never the bolt preload ("the uncertainty in preload is already factored into F_V,max", 7.9.2; ECSS-E-ST-32-10C
Table 4-4 note a: the factors are not applied on the bolt preload) - this program's design factor J does the same. Every coefficient that a standard or a project requirement
prescribes - the tightening scatter, the other preload losses, the minimum friction coefficient, every design factor J - is read from the confirmed requirement set
(`s4_limits`, keys `bolt_*`) with its document / revision / clause; the program ships no standard value (the typical values of the handbook, Table 6-1 and Table 6-3, are
quoted in the help page for orientation only). Every direct input is validated once (`validate_bolt`): finite numbers, J > 0, 0 <= u, 0 <= rho < 1, 0 < mu <= 1, 0 < n <= 1,
forces >= 0, embedding >= 0. A missing embedding (settlement) or missing thermal data (T_min, T_max, alpha) blocks a PASS: the result is UNCONFIRMED (give 0 um to state that there
is none). A margin has the form of the handbook, MoS = capacity / demand - 1 (5.3.2, eq. 5.3.1); a check whose J, allowable or chain constant is not confirmed is
UNCONFIRMED_REQUIREMENTS and is never PASS.

Chain (per bolt; N, mm, MPa; compliances in mm/N). Clause = ECSS-E-HB-32-23A Rev.1:
  Phi_n    = n k_b / (k_b + k_c)                                  force ratio with the loading-plane factor n (7.4, eq. 7.4.6)
  F_Z      = f_Z / (1/k_b + 1/k_c)                                preload loss by embedding, f_Z = sum of the settlements [mm] (6.4.2.1, eq. 6.4.2)
  F_V,max0 = F_nom (1 + u),   F_V,min0 = F_nom,min (1 - u) - F_Z - rho F_nom        u = uncertainty of the tightening (6.3.2.3, eq. 6.3.17 / 6.3.18; the typical coefficient
             method with the uncertainty factor of Table 6-1; u may also be the bound of the experimental coefficient method, 6.3.2.2), rho = loss by the other
             relaxation mechanisms (6.4.1: group interaction, gasket creep, metallic creep, torsional relaxation) as a fraction of the nominal preload (the form of eq. 6.4.3);
             F_nom,min = the lowest nominal preload (`F_nom_min`, optional, default F_nom): with a prevailing torque (self-locking nut / insert, 6.3.15, 6.3.18) the preload
             from the lowest installation torque is lower than the nominal one
  dF_T(T)  = (alpha_c - alpha_b)(T - T_ref) L_j / (1/k_b + 1/k_c)  thermally induced load for a Young's modulus that does not change with temperature (6.3.5.1, eq. 6.3.22);
             stack alpha_c = thickness-weighted mean of the layers (6.3.5.2: the sign printed in eq. 6.3.28 of Rev.1 is opposite to eq. 6.3.22; the code follows 6.3.22 and physics:
             clamped parts that expand more than the bolt raise the preload)
  F_V,min  = F_V,min0 + J_T min(0, dF_T(T_min), dF_T(T_max)),  F_V,max = F_V,max0 + J_T max(0, dF_T(T_min), dF_T(T_max))     (6.3.2.2, eq. 6.3.14 / 6.3.15; the handbook adds the
             lower / higher of the two increments without the clipping at 0: the program also keeps the assembly temperature in the range, which is conservative).
             J_T = the factor of the thermally induced bolt load: ECSS-E-ST-32-10C 4.3.2.1 b (with 4.3.2.2 a) applies the factors of safety to additive loads, thermal ones included,
             and 4.3.2.2 b with Table 4-4 notes a / e keeps them on the thermally induced load too (the preload itself carries none; no factor below 1,0 on an additive load, 4.3.2.1 d). Default J_T = the factor J of the check being computed (a loss of the preload by a cold case is amplified in gapping / slip,
             an increase by a hot case in the tensile / combined / pull-out / crushing checks); J_T = max(J, 1,0); the requirement-set row `bolt_thermal_factor` replaces it (a value below 1,0 is UNCONFIRMED; 1,0 is the user's
             decision, e.g. for an expendable launch vehicle, Table 4-4 note e)
  gapping  : MoS = (F_V,min - F_K,req) / ((1 - Phi_n) J F_ax) - 1                                          (7.8.2, eq. 7.8.1; F_K,req = required clamp force, default 0;
             without a separating force the margin is F_V,min / F_K,req - 1: the joint fails whenever F_V,min < F_K,req, and with F_K,req = 0 whenever F_V,min <= 0: the clamp force is gone)
  slip     : MoS = (F_V,min - (1 - Phi_n) F_ax) mu_min n_s / (J F_sh) - 1                                  (9.2.3, eq. 9.2.6; n_s = number of faying surfaces x; F_c = max(.,0))
  tensile  : MoS = F_allow A_s / (F_V,max + J Phi_n F_ax) - 1, yield and ultimate allowables                (7.9.2 / 7.9.3, eq. 7.9.2 and 7.9.3; overall load, J on the external part only)
  shear    : MoS = F_su A_s / (J F_sh) - 1 (shear plane in the thread, A_s; with F_sy also the yield margin)  (9.3.2.2, eq. 9.3.1 / 9.3.2; no F_sy of the bolt = UNCONFIRMED)
  combined : R_A = (F_V,max + J Phi_n F_ax) / (F_tu A_s),  R_S = J F_sh / (F_su A_s),  R_comb = sqrt(R_A^2 + R_S^p),  MoS = 1/R_comb - 1          (9.3.2.3, eq. 9.3.3 - 9.3.14)
             p = 2 for the thread in the shear plane (default, conservative), p = 3 for the shank (`shear_plane`); the yield form uses F_ty, F_sy and the yield factor
  pull-out : F_th,crit = min(F_th,n, F_th,b) of the Alexander shear areas of L_eng,eff = L_eng - 2 p with the thread-bending factors c1 c2 c3 (7.10.2 - 7.10.3, eq. 7.10.1 - 7.10.15;
             computed by `joints.vdi2230.stripping`, whose thread constants are the same)
                                                                         MoS = F_th,crit / (F_V,max + J Phi_n F_ax) - 1                          (7.10.4, eq. 7.10.17)
  crushing : MoS = sigma_br A_uh / (F_V,max0 + J_T dF_T,gain + J Phi_n F_ax) - 1, A_uh = pi/4 (d_w^2 - d_h^2) under the head / nut         (7.11, eq. 7.11.1 / 7.11.2; ultimate allowable with the
             ultimate factor and the yield margin with the yield strength F_y of the part and the yield factor)
  hole bearing : sigma_br = F_sh / (d t),  MoS = F_br / (J sigma_br) - 1                                  (9.3.4, eq. 9.3.22 - 9.3.24; ultimate and yield margins as for crushing)

Deviations and limits of the program (all on the conservative side or a declared limit; listed in `DEVIATIONS`, in the answer and in the report):
  * the increments of the thermal preload are clipped at 0 (see above);
  * shear, combined and the hole bearing are computed for every joint, although the handbook assigns them to bearing joints (Table 5-1) and the slip check to friction-grip joints;
  * the shear area is the stress area A_s of the thread (the shank area A_sha, if the shank is in the shear plane, is larger: conservative);
  * a joint is assessed as one that must not gap: the checks of the external load alone (7.9.2 eq. 7.9.4 / 7.9.5, 9.3.2.3 eq. 9.3.11 / 9.3.12) that apply only when gapping is
    tolerated are not implemented;
  * more than two faying surfaces per bolt (the handbook allows any number x) need the confirmed requirement-set row `bolt_slip_planes_max`: a project decision of the program;
  * concentric joints only (no eccentric load introduction, chapter 8), no Young's modulus that changes with temperature (6.3.5.3), no bending of the fastener (Table 5-1 note 1);
  * the 5-fastener relaxation of the slip check with the nominal preload (9.2.2.2, eq. 9.2.3), the tightening margin of 6.5 (von Mises of preload and torque), the torque
    relation of 6.3.1 and the fastener-group clearance strength of 9.3.2.4 are not implemented; the net-section and shear-out checks of the flanges (9.3.3, 9.3.5) are not either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .notching import FAIL, PASS, UNCONFIRMED
from .s4_limits import S4Limits

HANDBOOK = "ECSS-E-HB-32-23A Rev.1 (6 February 2023)"
FORMULA_NOTE = f"formula chain after {HANDBOOK}, clause per formula; the deviations of the program are listed with the result"
CLAUSE = "<clause>"                      # the clause of a factor / coefficient is entered by the user in the requirement set
DEVIATIONS = (
    "the increments of the thermal preload are clipped at zero (the assembly temperature stays in the range): conservative against 6.3.5.1, eq. 6.3.23 / 6.3.24",
    "shear, combined and hole bearing are computed for every joint, the slip check too (the handbook assigns them to bearing / friction-grip joints, Table 5-1): conservative",
    "the shear area is the thread stress area A_s (the shank area is larger): conservative against 9.3.2.2",
    "a joint is assessed as one that must not gap: the external-load-only checks of 7.9.2 (eq. 7.9.4, 7.9.5) and 9.3.2.3 (eq. 9.3.11, 9.3.12) for tolerated gapping are not implemented",
    "more than two faying surfaces per bolt need the confirmed requirement-set row bolt_slip_planes_max (a project decision of the program, not in the handbook)",
    "concentric joints only: no eccentric load introduction (chapter 8, Phi_e,n), no Young's modulus that changes with temperature (6.3.5.3), no bending of the fastener (Table 5-1 note 1)",
    "the thermally induced bolt load carries the factor of the check (ECSS-E-ST-32-10C 4.3.2.1 b with 4.3.2.2 a, 4.3.2.2 b, Table 4-4 notes a / e; never below 1,0) unless the requirement-set row bolt_thermal_factor says otherwise; the handbook (6.3.14 - 6.3.18) adds it without a factor",
    "a prevailing torque (self-locking nut) lowers the minimum preload only through the optional input F_nom_min (6.3.15, 6.3.18); without it the preload range of the tightening method alone is used",
    "a ratio s_w/d of the nut outside 1,4 - 1,9 (7.10.2) leaves the pull-out margin UNCONFIRMED; the yield bearing margins use the yield strength F_y of the clamped material",
    "not implemented: tightening margin 6.5, torque relation 6.3.1, slip with the nominal preload of 5 or more fasteners 9.2.2.2, fastener-group clearance 9.3.2.4, net-section 9.3.3 and shear-out 9.3.5 of the flanges",
)
TAN30 = math.tan(math.radians(30.0))
MAX_SLIP_PLANES = 10                     # geometry sanity: the number of friction interfaces of one bolt
FREE_SLIP_PLANES = 2                     # a joint of up to this many friction interfaces per bolt (single / double shear) needs no further confirmation; more is a requirement-set row
GRIP_TOL = 0.05                          # numerical tolerance of the comparison of L_grip with the sum of the layers (5 %): a larger difference is only a warning
CHECK_ORDER = ("gapping", "slip", "tensile_yield", "tensile_ultimate", "shear_yield", "shear", "combined_yield", "combined", "pullout", "bearing_head_yield", "bearing_head",
               "bearing_hole_yield", "bearing_hole")
NUT_RATIO_RANGE = (1.4, 1.9)             # the range of s_w/d for which the nut coefficient c1 of 7.10.2 is given
SHEAR_PLANES = {"thread": 2.0, "shank": 3.0}          # exponent p of the shear term R_S^p of the combined interaction (ECSS-E-HB-32-23A Rev.1, 9.3.2.3, eq. 9.3.3 - 9.3.6)


class BoltsError(ValueError):
    """The bolt inputs are inconsistent."""


@dataclass
class Layer:
    """A layer of the clamped stack: thickness [mm] and the expansion coefficient [1/K]."""
    t_mm: float
    alpha: float
    name: str = ""


@dataclass
class Allowable:
    """An allowable [MPa] with its source and whether it is a verified (confirmed, A/B basis) value of the program's material record."""
    value: float
    verified: bool = False
    source: str = ""


@dataclass
class EcssBolt:
    """One bolt of the joint with its stiffness, thermal data, limit loads and allowables. Lengths mm, forces N, stresses MPa, stiffness N/mm."""
    name: str = "bolt"
    d_mm: float = 0.0
    pitch_mm: float | None = None
    A_s_mm2: float = 0.0                       # tensile stress area
    k_b: float = 0.0                           # stiffness of the bolt [N/mm]
    k_c: float = 0.0                           # stiffness of the clamped parts [N/mm]
    n_load_plane: float = 1.0                  # loading-plane factor (1 = load introduced at the head / nut planes)
    F_nom: float = 0.0                         # nominal (target) preload [N]
    F_nom_min: float | None = None             # lowest nominal preload [N] for F_V,min (prevailing torque, 6.3.15 / 6.3.18; None = F_nom)
    F_K_req: float = 0.0                       # required clamp force for the separation margin (7.8.1; 0 = the joint only must not open) [N]
    embedding_um: float | None = None          # sum of the settlements [µm] (None = not accounted)
    layers: list[Layer] = field(default_factory=list)
    alpha_bolt: float | None = None
    L_grip_mm: float | None = None             # grip length (default: the sum of the layers)
    T_ref: float = 20.0                        # assembly temperature [°C]
    T_min: float | None = None
    T_max: float | None = None
    F_ax: float = 0.0                          # limit separating (tensile) external force per bolt [N]
    F_sh: float = 0.0                          # limit shear force per bolt [N]
    n_slip_planes: int = 1                     # number of faying surfaces x (9.2.2.2)
    shear_plane: str = "thread"                # "thread" | "shank": where the shear plane of the bolt is (exponent of the combined interaction, 9.3.2.3)
    engaged_mm: float | None = None            # engaged thread length L_eng (pull-out; the effective length is L_eng - 2 p, 7.10.2)
    wrench_mm: float | None = None             # width across flats of a NUT (coefficient c1 of 7.10.2; None = a tapped hole, c1 = 1)
    d_w_mm: float | None = None                # bearing face diameter under head / nut
    d_h_mm: float | None = None                # hole diameter
    t_bearing_mm: float | None = None          # thickness of the plate for the hole bearing
    allow: dict[str, Allowable] = field(default_factory=dict)   # bolt_Fty, bolt_Ftu, bolt_Fsy, bolt_Fsu, part_Fsu (internal thread), part_Fbr, hole_Fbr, part_Fy (yield bearing)
    # direct (never confirmed) overrides of requirement-set values
    scatter: float | None = None
    relaxation: float | None = None
    mu_min: float | None = None
    factors: dict[str, float] = field(default_factory=dict)     # gapping, slip, tensile_yield, tensile, shear, combined, pullout, bearing, thermal


FACTOR_KEY = {"gapping": "bolt_factor_gapping", "slip": "bolt_factor_slip", "tensile_yield": "bolt_factor_tensile_yield", "tensile": "bolt_factor_tensile",
              "shear": "bolt_factor_shear", "combined": "bolt_factor_combined", "pullout": "bolt_factor_pullout", "bearing": "bolt_factor_bearing",
              "thermal": "bolt_thermal_factor"}


@dataclass
class Margin:
    check: str
    mos: float | None
    demand: float | None
    capacity: float | None
    design_factor: float | None
    formula: str
    status: str                       # PASS | FAIL | UNCONFIRMED_REQUIREMENTS | n/a
    unconfirmed: list[str] = field(default_factory=list)
    note: str = ""
    thermal_factor: float | None = None   # J_T applied to the thermally induced increment of the check (None = the check has none)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class BoltResult:
    bolt: str
    phi: float
    F_emb: float
    F_pmin0: float
    F_pmax0: float
    dF_th: dict[str, float]
    F_pmin: float
    F_pmax: float
    F_c_min: float
    F_b_max: float
    margins: list[Margin]
    verdict: str
    verdict_text: str
    inputs: list[dict]
    warnings: list[str]
    formula_note: str = FORMULA_NOTE

    def worst(self) -> Margin | None:
        rows = [m for m in self.margins if m.mos is not None]
        return min(rows, key=lambda m: m.mos if m.mos is not None else math.inf) if rows else None

    def to_dict(self) -> dict:
        w = self.worst()
        return {"bolt": self.bolt, "phi": self.phi, "F_emb_N": self.F_emb, "F_pmin0_N": self.F_pmin0, "F_pmax0_N": self.F_pmax0, "dF_th_N": self.dF_th,
                "F_pmin_N": self.F_pmin, "F_pmax_N": self.F_pmax, "F_clamp_min_N": self.F_c_min, "F_bolt_max_N": self.F_b_max,
                "margins": [m.to_dict() for m in self.margins], "worst": w.check if w else None, "worst_MoS": w.mos if w else None, "verdict": self.verdict,
                "verdict_text": self.verdict_text, "inputs": self.inputs, "warnings": self.warnings, "formula_note": self.formula_note}


# ------------------------------------------------------------------ pieces of the chain (pure functions)
def load_factor(k_b: float, k_c: float, n: float = 1.0) -> float:
    """Φ = n k_b / (k_b + k_c): the share of the external axial force that goes into the bolt."""
    if not (k_b > 0 and k_c > 0 and n > 0):
        raise BoltsError("k_b, k_c and n must be positive")
    return n * k_b / (k_b + k_c)


def embedding_loss(f_emb_mm: float, k_b: float, k_c: float) -> float:
    """F_emb = f_emb / (1/k_b + 1/k_c) [N]."""
    return f_emb_mm / (1.0 / k_b + 1.0 / k_c)


def stack_alpha(layers: list[Layer]) -> tuple[float, float]:
    """(thickness-weighted mean expansion coefficient, total thickness) of the clamped stack."""
    t = sum(x.t_mm for x in layers)
    if not layers or t <= 0:
        raise BoltsError("The clamped stack needs at least one layer with a positive thickness")
    return sum(x.alpha * x.t_mm for x in layers) / t, t


def thermal_preload_change(k_b: float, k_c: float, alpha_c: float, alpha_b: float, L_grip_mm: float, T: float, T_ref: float) -> float:
    """ΔF_th [N] = (α_c − α_b)(T − T_ref) L_grip / (1/k_b + 1/k_c): positive when the clamped parts expand more than the bolt (the preload rises)."""
    return (alpha_c - alpha_b) * (T - T_ref) * L_grip_mm / (1.0 / k_b + 1.0 / k_c)


def thread_shear_areas(d: float, P: float, engaged_mm: float) -> dict:
    """Shear areas of the engaged thread (Alexander, as in VDI 2230): internal thread at the major diameter, bolt thread at the minor diameter of the nut."""
    from ..joints import tables
    th = tables.Thread(d, P)
    D1, D2 = th.d1, th.d2
    aM = math.pi * d / P * (P / 2 + (d - D2) * TAN30)
    aS = math.pi * D1 / P * (P / 2 + (th.d2 - D1) * TAN30)
    return {"A_internal": aM * engaged_mm, "A_bolt": aS * engaged_mm}


def _finite(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise BoltsError(f"{name} must be a finite number (got {v!r})")
    return float(v)


def validate_bolt(b: EcssBolt) -> list[str]:
    """Checks the direct inputs of a bolt once; raises `BoltsError`; returns the notes (a shear force given with a sign is used by its magnitude)."""
    notes: list[str] = []
    if not isinstance(b.factors, dict):
        raise BoltsError("factors must be a dictionary {gapping: J, slip: J, ...}")
    for k, v in b.factors.items():
        if k not in FACTOR_KEY:
            raise BoltsError(f"Unknown design factor '{k}' (known: {', '.join(FACTOR_KEY)})")
        if _finite(f"design factor J ({k})", v) <= 0:
            raise BoltsError(f"design factor J ({k}) must be positive")
    for name, v in (("d", b.d_mm), ("A_s", b.A_s_mm2), ("k_b", b.k_b), ("k_c", b.k_c), ("F_nom", b.F_nom)):
        if _finite(name, v) <= 0:
            raise BoltsError(f"{name} must be positive")
    if b.F_nom_min is not None and not 0 < _finite("F_nom_min", b.F_nom_min) <= b.F_nom:
        raise BoltsError("F_nom_min (the lowest nominal preload) must satisfy 0 < F_nom_min <= F_nom")
    if _finite("F_K_req", b.F_K_req) < 0:
        raise BoltsError("F_K_req (the required clamp force) must be >= 0")
    if b.shear_plane not in SHEAR_PLANES:
        raise BoltsError(f"shear_plane must be one of {', '.join(SHEAR_PLANES)}")
    if not 0 < _finite("n_load_plane", b.n_load_plane) <= 1:
        raise BoltsError("n_load_plane must satisfy 0 < n <= 1")
    if b.pitch_mm is not None and _finite("pitch", b.pitch_mm) <= 0:
        raise BoltsError("pitch must be positive")
    if b.embedding_um is not None and _finite("embedding", b.embedding_um) < 0:
        raise BoltsError("embedding (settlement) must be >= 0")
    if _finite("F_ax", b.F_ax) < 0:
        raise BoltsError("F_ax is the separating (tensile) external force per bolt: it must be >= 0 (give 0 for a compressed joint)")
    if _finite("F_sh", b.F_sh) < 0:
        b.F_sh = abs(b.F_sh)
        notes.append("F_sh was given negative: its magnitude is used")
    if isinstance(b.n_slip_planes, bool) or not isinstance(b.n_slip_planes, int) or not 1 <= b.n_slip_planes <= MAX_SLIP_PLANES:
        raise BoltsError(f"n_slip_planes must be an integer from 1 to {MAX_SLIP_PLANES} (the number of friction interfaces of one bolt)")
    for name, v in (("T_ref", b.T_ref), ("T_min", b.T_min), ("T_max", b.T_max), ("alpha_bolt", b.alpha_bolt), ("L_grip", b.L_grip_mm), ("engaged length", b.engaged_mm),
                    ("d_w", b.d_w_mm), ("d_h", b.d_h_mm), ("bearing thickness", b.t_bearing_mm), ("wrench size", b.wrench_mm)):
        if v is not None:
            _finite(name, v)
    for name, v in (("L_grip (the grip length)", b.L_grip_mm), ("The engaged thread length", b.engaged_mm), ("The wrench size", b.wrench_mm)):
        if v is not None and not v > 0:                             # 0 is not "none": leave the input out to say that there is no grip length / pull-out check
            raise BoltsError(f"{name} must be positive (leave it out to skip it)")
    if b.T_min is not None and b.T_max is not None and b.T_min > b.T_max:
        raise BoltsError("T_min must not exceed T_max")
    for x in b.layers:
        if not (_finite("layer thickness", x.t_mm) > 0 and math.isfinite(x.alpha)):
            raise BoltsError("every layer needs a positive thickness and a finite expansion coefficient")
    for k, a in b.allow.items():
        if not (math.isfinite(a.value) and a.value > 0):
            raise BoltsError(f"allowable {k} must be a finite positive stress")
    for name, v in (("tightening scatter", b.scatter), ("relaxation loss", b.relaxation), ("minimum friction coefficient", b.mu_min)):
        if v is not None:
            _finite(name, v)
    return notes


def _mos(capacity: float, demand: float, J: float, base: float = 0.0) -> float:
    """MoS = capacity / (base + J (demand - base)) - 1: J multiplies the part of the demand above `base` (the unfactored preload F_V,max in the overall-load checks of the
    handbook, 7.9.2: the safety factor acts on the external load only); base = 0 factors the whole demand."""
    den = base + J * (demand - base)
    return math.inf if den <= 0 else capacity / den - 1.0


def _get(lim: S4Limits, key: str, override: float | None, name: str, inputs: list[dict]) -> tuple[float | None, bool]:
    v, ok, src = lim.get(key, override)
    inputs.append({"name": name, "key": key, "value": v, "source": src or "not available", "confirmed": ok})
    return v, ok


# ------------------------------------------------------------------ the assessment of one bolt
def assess_bolt(b: EcssBolt, limits: S4Limits | None = None) -> BoltResult:
    """The margins of one bolt (see the module docstring for the chain and the clauses). The coefficients come from `limits` (the requirement set) unless `b` overrides them."""
    lim = limits or S4Limits()
    inputs: list[dict] = []
    warnings: list[str] = []
    warnings.extend(validate_bolt(b))
    phi = load_factor(b.k_b, b.k_c, b.n_load_plane)
    inputs.append({"name": "loading-plane factor n", "key": "", "value": b.n_load_plane, "source": "entered directly (geometry of the joint)", "confirmed": None})
    if b.F_sh > 0:
        inputs.append({"name": "number of friction interfaces n_s", "key": "", "value": b.n_slip_planes, "source": "entered directly (geometry of the joint)", "confirmed": None})
    u, u_ok = _get(lim, "bolt_tightening_scatter", b.scatter, "tightening scatter u", inputs)
    rho, rho_ok = _get(lim, "bolt_relaxation_loss", b.relaxation, "relaxation loss rho", inputs)
    mu, mu_ok = _get(lim, "bolt_slip_mu_min", b.mu_min, "minimum friction coefficient", inputs)
    if u is not None and not (math.isfinite(u) and u >= 0):
        raise BoltsError("The tightening scatter u must be >= 0")
    if rho is not None and not (math.isfinite(rho) and 0 <= rho < 1):
        raise BoltsError("The relaxation loss rho must satisfy 0 <= rho < 1")
    if mu is not None and not (math.isfinite(mu) and 0 < mu <= 1):
        raise BoltsError("The minimum friction coefficient must satisfy 0 < mu <= 1")
    for tag, key in FACTOR_KEY.items():                    # every design factor of the requirement set is checked, used or not (a J <= 0 anywhere is an input error)
        e = lim.entries.get(key)
        if e is not None and e.value is not None and not (math.isfinite(e.value) and e.value > 0):
            raise BoltsError(f"design factor J ({tag}) of the requirement set must be a finite positive number (got {e.value})")
    data_unconf: list[str] = []
    if b.n_load_plane != 1.0:
        data_unconf.append("loading-plane factor n < 1 entered directly (it lowers the load share of the bolt: not confirmed)")
    if b.embedding_um is None:
        warnings.append("Embedding (settlement) is not given: the preload loss by embedding is not accounted and no PASS is possible (give 0 um to state that there is none)")
        data_unconf.append("embedding (settlement) not given")
    F_emb = embedding_loss((b.embedding_um or 0.0) * 1e-3, b.k_b, b.k_c)
    u_v = 0.0 if u is None else u
    r_v = 0.0 if rho is None else rho
    F_lo = b.F_nom if b.F_nom_min is None else b.F_nom_min
    if b.F_nom_min is not None:
        inputs.append({"name": "lowest nominal preload F_nom,min", "key": "", "value": b.F_nom_min, "source": "entered directly (prevailing torque, 6.3.15 / 6.3.18)", "confirmed": None})
    F_pmax0 = b.F_nom * (1.0 + u_v)
    F_pmin0_raw = F_lo * (1.0 - u_v) - F_emb - r_v * b.F_nom                  # 6.3.2.3 eq. 6.3.18 with F_Z = embedding + the other relaxation losses (6.4)
    F_pmin0 = max(F_pmin0_raw, 0.0)
    if F_pmin0_raw < 0.0:
        warnings.append(f"The minimum preload before the thermal change is negative ({F_pmin0_raw:.1f} N: the tightening scatter, the embedding and the other losses exceed the lowest "
                        "nominal preload): it is taken as 0 N, the clamp force is lost and the gapping check fails")
    dF: dict[str, float] = {}
    if b.L_grip_mm and b.layers:                                    # the grip length of the thermal change against the layers of the stack
        t_sum = sum(x.t_mm for x in b.layers)
        if t_sum > 0 and abs(b.L_grip_mm - t_sum) > GRIP_TOL * t_sum:
            warnings.append(f"L_grip = {b.L_grip_mm:g} mm differs from the sum of the layers ({t_sum:g} mm) by {100.0 * abs(b.L_grip_mm - t_sum) / t_sum:.1f} % "
                            f"(more than {100.0 * GRIP_TOL:g} %): the thermal preload change uses L_grip - check the input")
    if b.T_min is not None and b.T_max is not None and b.alpha_bolt is not None and b.layers:
        a_c, t_tot = stack_alpha(b.layers)
        L = b.L_grip_mm if b.L_grip_mm else t_tot
        for tag, T in (("T_min", b.T_min), ("T_max", b.T_max)):
            dF[tag] = thermal_preload_change(b.k_b, b.k_c, a_c, b.alpha_bolt, L, T, b.T_ref)
    else:
        warnings.append("No thermal data (T_min, T_max, alpha of the bolt and of the layers): the thermal preload change is not accounted and no PASS is possible")
        data_unconf.append("thermal data (T_min, T_max, alpha of bolt and layers) not given")
    d_loss = min([0.0, *dF.values()])                               # the loss of the preload (<= 0) and the gain (>= 0) of the thermal cases; the assembly temperature stays in the range
    d_gain = max([0.0, *dF.values()])
    F_pmin = F_pmin0 + d_loss                                       # J_T = 1 (the values shown); the checks use their own J_T
    F_pmax = F_pmax0 + d_gain
    F_sep = (1.0 - phi) * b.F_ax
    F_c = max(F_pmin - F_sep, 0.0)
    F_b = F_pmax + phi * b.F_ax
    chain_unconf = [n for n, ok, v in (("tightening scatter", u_ok, u), ("relaxation loss", rho_ok, rho)) if not ok or v is None] + data_unconf
    margins: list[Margin] = []
    fcache: dict[str, tuple[float | None, bool]] = {}

    def factor(name: str) -> tuple[float | None, bool]:
        if name not in fcache:
            fcache[name] = _get(lim, FACTOR_KEY[name], b.factors.get(name), f"design factor J ({name})", inputs)
        return fcache[name]

    th_v, th_ok, th_src = lim.get(FACTOR_KEY["thermal"], b.factors.get("thermal"))          # the optional factor of the thermally induced bolt load (default: J of the check)
    if th_v is not None:
        if not (math.isfinite(th_v) and th_v > 0):
            raise BoltsError(f"design factor J (thermal) must be a finite positive number (got {th_v})")
        inputs.append({"name": "design factor J (thermal)", "key": FACTOR_KEY["thermal"], "value": th_v, "source": th_src or "not available", "confirmed": th_ok})

    def j_thermal(name: str) -> tuple[float, list[str]]:
        """(J_T, what is unconfirmed about it) for the check `name`: the requirement-set row `bolt_thermal_factor` when present, else the factor J of the check itself."""
        if th_v is not None:
            un_j = ([] if th_ok else ["design factor J (thermal) not confirmed"]) + (
                [f"design factor J (thermal) = {th_v:g} is below 1,0: the standard applies no factor below 1,0 to an additive load (ECSS-E-ST-32-10C 4.3.2.1 b, 4.3.2.1 d, 4.3.2.2 a, 4.3.2.2 b)"]
                if th_v < 1.0 else [])
            return th_v, un_j
        j = factor(name)[0]
        return (1.0 if j is None else max(j, 1.0)), []                  # a factor of the check below 1,0 never lowers an additive thermal load (4.3.2.1 b, 4.3.2.2 a)

    def add(check: str, cap: float | None, dem: float | None, J_name: str, formula: str, extra_unconf: list[str], allow_names: list[str] | None = None,
            note: str = "", ext: float | None = None, thermal: float | None = None) -> None:
        """One margin: MoS = cap / (J dem) - 1; with `ext` (the external part of the overall bolt load, Phi_n F_ax) the overall-load form of 7.9.2:
        MoS = cap / (F_V,max0 + J_T dF_gain + J ext) - 1 (the preload carries no factor, the thermally induced increment carries J_T)."""
        J, J_ok = factor(J_name)
        if J is not None and not (math.isfinite(J) and J > 0):
            raise BoltsError(f"design factor J ({J_name}) must be a finite positive number (got {J})")
        un = list(extra_unconf)
        if J is None:
            un.append(f"design factor J ({J_name}) missing")
        elif not J_ok:
            un.append(f"design factor J ({J_name}) not confirmed")
        for a in allow_names or []:
            al = b.allow.get(a)
            if al is None:
                un.append(f"allowable {a} missing")
            elif not al.verified:
                un.append(f"allowable {a} not verified (confirmed A/B basis)")
        if cap is None or dem is None or J is None:
            margins.append(Margin(check, None, dem, cap, J, formula, UNCONFIRMED, un or ["input missing"], note, thermal))
            return
        if ext is None:
            mos = _mos(cap, dem, J)
        else:
            jt, un_t = j_thermal(J_name)
            un += un_t
            den = F_pmax0 + jt * d_gain + J * ext
            mos = math.inf if den <= 0 else cap / den - 1.0
            thermal = jt
            note = (note + "; " if note else "") + f"factored demand {den:.1f} N = {F_pmax0:.1f} + {jt:.4g} x {d_gain:.1f} (thermal) + {J:.4g} x {ext:.1f}"
        status = UNCONFIRMED if un else (PASS if mos >= 0 else FAIL)
        margins.append(Margin(check, mos, dem, cap, J, formula, status, un, note, thermal))

    # gapping (7.8.2, eq. 7.8.1): the loss of the preload by the cold case carries J_T (the same factor as the thermally induced load of the other checks)
    Jt_g, un_tg = j_thermal("gapping")
    F_pmin_g = F_pmin0 + Jt_g * d_loss
    if b.F_ax > 0:
        add("gapping", F_pmin_g - b.F_K_req, F_sep, "gapping", "MoS = (F_V,min - F_K,req) / ((1 - Phi_n) J F_ax) - 1, F_V,min = F_V,min0 + J_T dF_T,loss [7.8.1]", chain_unconf + un_tg,
            note=f"Phi_n = {phi:.4f}, F_V,min = {F_pmin_g:.1f} N (J_T = {Jt_g:.4g})" + (f", F_K,req = {b.F_K_req:g} N" if b.F_K_req else ""), thermal=Jt_g)
    elif b.F_K_req > 0 or F_pmin_g <= 0.0:                          # no separating force, but a required clamp force (or a clamp force that is lost altogether): the preload against it, never PASS below it
        un = chain_unconf + un_tg
        if th_v is None:
            j_g, j_g_ok = factor("gapping")
            if j_g is None:
                un.append("design factor J (gapping) missing")
            elif not j_g_ok:
                un.append("design factor J (gapping) not confirmed")
        mos = F_pmin_g / b.F_K_req - 1.0 if b.F_K_req > 0 else (-1.0 if F_pmin_g <= 0.0 else math.inf)    # F_K,req = 0: the clamp force must stay above zero (separation when it reaches F_K,req, 7.8.1)
        margins.append(Margin("gapping", mos, b.F_K_req, F_pmin_g, None,
                              "MoS = F_V,min / F_K,req - 1, F_V,min = F_V,min0 + J_T dF_T,loss (no separating force: the joint keeps the required clamp force; F_K,req = 0: F_V,min > 0) [7.8.1]",
                              UNCONFIRMED if un else (PASS if mos >= 0 else FAIL), un, f"F_V,min = {F_pmin_g:.1f} N (J_T = {Jt_g:.4g}), F_K,req = {b.F_K_req:g} N, no axial force", Jt_g))
        if mos < 0:
            warnings.append(f"The minimum preload F_V,min = {F_pmin_g:.1f} N is below the required clamp force F_K,req = {b.F_K_req:g} N although no axial force acts" if b.F_K_req > 0
                            else f"The minimum preload F_V,min = {F_pmin_g:.1f} N is not above zero: the clamp force is lost and the joint separates although no axial force acts")
    else:
        margins.append(Margin("gapping", math.inf, 0.0, F_pmin, None, "no separating force", "n/a", [], "no axial external force"))
    # slip (9.2.3, eq. 9.2.6)
    if b.F_sh > 0:
        if mu is None:
            margins.append(Margin("slip", None, b.F_sh, None, None, "MoS = (F_V,min - (1 - Phi_n) F_ax) mu_min n_s / (J F_sh) - 1 [9.2.6]", UNCONFIRMED, ["minimum friction coefficient missing"]))
        else:
            ns_unconf: list[str] = []
            if b.n_slip_planes > FREE_SLIP_PLANES:                  # more than single / double shear multiplies the slip capacity: only with a confirmed project row
                ns_max, ns_ok = _get(lim, "bolt_slip_planes_max", None, "largest number of friction interfaces per bolt", inputs)
                if ns_max is None or not ns_ok or b.n_slip_planes > ns_max:
                    ns_unconf.append(f"n_slip_planes = {b.n_slip_planes} (> {FREE_SLIP_PLANES}) is not covered by a confirmed requirement-set row bolt_slip_planes_max")
            Jt_s, un_ts = j_thermal("slip")
            F_c_s = max(F_pmin0 + Jt_s * d_loss - F_sep, 0.0)
            add("slip", mu * b.n_slip_planes * F_c_s, b.F_sh, "slip", "MoS = (F_V,min - (1 - Phi_n) F_ax) mu_min n_s / (J F_sh) - 1, F_V,min = F_V,min0 + J_T dF_T,loss [9.2.6]",
                chain_unconf + un_ts + ([] if mu_ok else ["friction coefficient not confirmed"]) + ns_unconf, note=f"F_clamp = {F_c_s:.1f} N (J_T = {Jt_s:.4g})", thermal=Jt_s)
    else:
        margins.append(Margin("slip", math.inf, 0.0, None, None, "no shear force", "n/a", [], "no shear external force"))
    # bolt tensile, overall load (7.9.2, eq. 7.9.2 / 7.9.3): J on the external part, J_T on the thermal increment, none on the preload
    for tag, key, jn in (("tensile_yield", "bolt_Fty", "tensile_yield"), ("tensile_ultimate", "bolt_Ftu", "tensile")):
        al = b.allow.get(key)
        add(tag, None if al is None else al.value * b.A_s_mm2, F_b, jn, f"MoS = {key} A_s / (F_V,max0 + J_T dF_T,gain + J Phi_n F_ax) - 1; F_b,max = F_V,max + Phi_n F_ax [7.9.2, 7.9.3]", chain_unconf, [key],
            note=f"F_b,max = {F_b:.1f} N, sigma = {F_b / b.A_s_mm2:.1f} MPa", ext=phi * b.F_ax)
    # bolt shear (9.3.2.2, eq. 9.3.1 / 9.3.2); the yield form needs the shear yield allowable of the bolt: without it the result stays UNCONFIRMED
    if b.F_sh > 0:
        al = b.allow.get("bolt_Fsy")
        if al is None:
            warnings.append("Yield margins of the bolt in shear and combined are not assessed: the shear yield allowable bolt_Fsy (Fsy of the bolt material) is missing - no PASS is possible")
        add("shear_yield", None if al is None else al.value * b.A_s_mm2, b.F_sh, "tensile_yield", "MoS = F_sy A_s / (J F_sh) - 1 (shear of the bolt if the joint slips) [9.3.1]", [], ["bolt_Fsy"])
        al = b.allow.get("bolt_Fsu")
        add("shear", None if al is None else al.value * b.A_s_mm2, b.F_sh, "shear", "MoS = F_su A_s / (J F_sh) - 1 (shear of the bolt if the joint slips) [9.3.2]", [], ["bolt_Fsu"])
    # combined tension and shear (9.3.2.3, eq. 9.3.3 - 9.3.14)
    p_exp = SHEAR_PLANES[b.shear_plane]
    for tag, jn, ft, fs in (("combined_yield", "tensile_yield", "bolt_Fty", "bolt_Fsy"), ("combined", "combined", "bolt_Ftu", "bolt_Fsu")):
        if b.F_sh <= 0:
            continue
        al_t, al_s = b.allow.get(ft), b.allow.get(fs)
        Jc, Jc_ok = factor(jn)
        if Jc is not None and not (math.isfinite(Jc) and Jc > 0):
            raise BoltsError(f"design factor J ({jn}) must be a finite positive number (got {Jc})")
        form = (f"MoS = 1/R_comb - 1, R_comb = sqrt(R_A^2 + R_S^{p_exp:g}), R_A = (F_V,max0 + J_T dF_T,gain + J Phi_n F_ax)/({ft} A_s), R_S = J F_sh/({fs} A_s) "
                f"(shear plane in the {b.shear_plane}) [9.3.{3 if b.shear_plane == 'thread' else 5} - 9.3.14]")
        miss = [f"allowable {k} missing" for k, a in ((ft, al_t), (fs, al_s)) if a is None]
        if miss or Jc is None:
            margins.append(Margin(tag, None, None, None, Jc, form, UNCONFIRMED, chain_unconf + miss + ([f"design factor J ({jn}) missing"] if Jc is None else [])))
            continue
        assert al_t is not None and al_s is not None
        jt, un_t = j_thermal(jn)
        Ra = (F_pmax0 + jt * d_gain + Jc * phi * b.F_ax) / (al_t.value * b.A_s_mm2)
        Rs = Jc * b.F_sh / (al_s.value * b.A_s_mm2)
        try:
            r_comb = math.sqrt(Ra ** 2 + Rs ** p_exp)
        except OverflowError:                                          # a utilisation far above 1 with a cubic term: beyond any float
            r_comb = math.inf
        un = list(chain_unconf) + un_t + ([] if Jc_ok else [f"design factor J ({jn}) not confirmed"])
        for k, a in ((ft, al_t), (fs, al_s)):
            if not a.verified:
                un.append(f"allowable {k} not verified (confirmed A/B basis)")
        mos = math.inf if r_comb <= 0 else (-1.0 if math.isinf(r_comb) else 1.0 / r_comb - 1.0)
        margins.append(Margin(tag, mos, r_comb, 1.0, Jc, form, UNCONFIRMED if un else (PASS if mos >= 0 else FAIL), un,
                              f"R_A = {Ra:.4f}, R_S = {Rs:.4f}, p = {p_exp:g}, J_T = {jt:.4g}", jt))
    # pull-out (7.10, eq. 7.10.1 - 7.10.17)
    if b.engaged_mm and b.d_mm > 0 and b.pitch_mm:
        al_p, al_b = b.allow.get("part_Fsu"), b.allow.get("bolt_Fsu")
        formula = ("MoS = F_th,crit / (F_V,max0 + J_T dF_T,gain + J Phi_n F_ax) - 1 [7.10.17]; F_th,crit = min(tau_part A_th,n c1 c3, tau_bolt A_th,b c1 c2) with L_eng,eff = L_eng - 2 p [7.10.1, 7.10.3, 7.10.12]; "
                   "tau = the shear allowables")
        missing = [f"allowable {k} missing" for k, a in (("part_Fsu", al_p), ("bolt_Fsu", al_b)) if a is None]
        if missing:
            margins.append(Margin("pullout", None, F_b, None, None, formula, UNCONFIRMED, missing, "both the bolt and the part shear allowables are needed"))
        else:
            from ..joints import vdi2230
            assert al_p is not None and al_b is not None
            l_eff = max(b.engaged_mm - 2.0 * b.pitch_mm, 0.0)                   # the first and the last thread turn carry no load (7.10.2, eq. 7.10.3)
            sv = vdi2230.stripping(b.d_mm, b.pitch_mm, l_eff, 1.0, al_b.value, 1.0, al_p.value, b.wrench_mm, b.A_s_mm2)
            cap = min(sv["F_mGM"], sv["F_mGS"])
            nut_unconf: list[str] = []
            if b.wrench_mm and not NUT_RATIO_RANGE[0] - 1e-9 <= b.wrench_mm / b.d_mm <= NUT_RATIO_RANGE[1] + 1e-9:
                nut_unconf.append(f"s_w/d = {b.wrench_mm / b.d_mm:.2f} of the nut is outside {NUT_RATIO_RANGE[0]:g} - {NUT_RATIO_RANGE[1]:g}, the range of the coefficient c1 (7.10.2): "
                                  "the pull-out margin is not confirmed")
            add("pullout", cap, F_b, "pullout", formula, chain_unconf + nut_unconf, ["part_Fsu", "bolt_Fsu"], ext=phi * b.F_ax,
                note=f"L_eng,eff = {l_eff:.2f} mm; governing: {sv['governs']}; A_th,n = {sv['A_SGM']:.1f} mm2, A_th,b = {sv['A_SGS']:.1f} mm2, c1 c2 c3 = {sv['C1']:.3f} {sv['C2']:.3f} {sv['C3']:.3f}, "
                     f"R_S = {sv['R_S']:.3f}" + (f"; {sv['note']}" if sv["note"] else ""))
    else:
        warnings.append("Pull-out is not assessed: give the engaged length, the diameter and the pitch")
    # crushing of the flange under the head / nut (7.11, eq. 7.11.1 / 7.11.2): ultimate and yield
    if b.d_w_mm and b.d_h_mm and b.d_w_mm > b.d_h_mm:
        A_br = math.pi / 4 * (b.d_w_mm ** 2 - b.d_h_mm ** 2)
        for tag, key, jn, what in (("bearing_head_yield", "part_Fy", "tensile_yield", "yield"), ("bearing_head", "part_Fbr", "bearing", "ultimate")):
            al = b.allow.get(key)
            add(tag, None if al is None else al.value * A_br, F_b, jn, f"MoS = sigma_br({what}) A_uh / (F_V,max0 + J_T dF_T,gain + J Phi_n F_ax) - 1, A_uh = pi/4 (d_w^2 - d_h^2) [7.11.1, 7.11.2]",
                chain_unconf, [key], note=f"sigma_br = {F_b / A_br:.1f} MPa", ext=phi * b.F_ax)
    else:
        warnings.append("Head bearing is not assessed: give d_w and d_h")
    # hole bearing (9.3.4, eq. 9.3.22 - 9.3.24; d = the fastener diameter): ultimate and yield
    if b.F_sh > 0 and b.t_bearing_mm and b.d_mm > 0:
        for tag, key, jn, what in (("bearing_hole_yield", "part_Fy", "tensile_yield", "yield"), ("bearing_hole", "hole_Fbr", "bearing", "ultimate")):
            al = b.allow.get(key)
            add(tag, None if al is None else al.value * b.d_mm * b.t_bearing_mm, b.F_sh, jn, f"MoS = sigma_br,allow({what}) d t / (J F_sh) - 1 [9.3.22 - 9.3.24]", [], [key])
    elif b.F_sh > 0:
        warnings.append("Hole bearing is not assessed: give the thickness of the plate (t_bearing)")
    rows = [m for m in margins if m.status != "n/a"]
    if any(m.status == UNCONFIRMED for m in rows) or not rows:
        verdict, text = UNCONFIRMED, "a design factor, a chain constant or an allowable is not confirmed (or an input is missing) - indicative only"
    elif any(m.status == FAIL for m in rows):
        verdict, text = FAIL, "at least one margin of safety with confirmed inputs is negative"
    else:
        verdict, text = PASS, "all margins are non-negative with confirmed inputs"
    warnings.append(FORMULA_NOTE)
    return BoltResult(b.name, phi, F_emb, F_pmin0, F_pmax0, dF, F_pmin, F_pmax, F_c, F_b, margins, verdict, text, inputs, warnings)


# ------------------------------------------------------------------ adapters to fea_core.joints (VDI 2230)
def from_vdi_basics(bas: dict, name: str = "bolt", **kw: Any) -> EcssBolt:
    """EcssBolt from the result of `joints.vdi2230.basics` / `joints.fem.bolt_basics`: stiffness from the compliances (k = 1/δ), stress area, bearing face and hole,
    embedding f_Z, nominal preload = the mean of the tightening scatter of the VDI assembly (F_M,max + F_M,min)/2. Further fields (loads, thermal data, allowables)
    are given as keywords."""
    b = EcssBolt(name=name, d_mm=float(bas["d"]), pitch_mm=float(bas["P"]), A_s_mm2=float(bas["As"]), k_b=1.0 / float(bas["delta_S"]), k_c=1.0 / float(bas["delta_P"]),
                 F_nom=0.5 * (float(bas["F_Mmax"]) + float(bas["F_Mmin"])), embedding_um=float(bas["f_Z"]) * 1e3, d_w_mm=float(bas["dW"]), d_h_mm=float(bas["dh"]),
                 L_grip_mm=float(bas["lK"]))
    for k, v in kw.items():
        if not hasattr(b, k):
            raise BoltsError(f"Unknown bolt field '{k}'")
        setattr(b, k, v)
    return b


def loads_from_vdi(w: dict) -> dict:
    """{"F_ax": F_A, "F_sh": F_Q} from the operating forces of a VDI assessment (`joints.vdi2230.assess` input `w`)."""
    return {"F_ax": max(float(w.get("F_A", 0.0)), 0.0), "F_sh": abs(float(w.get("F_Q", 0.0)))}

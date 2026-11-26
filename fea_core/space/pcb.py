"""PCB vibration check by the Steinberg method (F14 stage S4): allowable relative displacement of a board in random vibration, margin of safety and the
three-band fatigue damage cross-checked against the spectral fatigue of `fea_core.fatigue.spectral`.

    Z_allow = c · B / (C · h · r · √L)                                   (the allowable relative displacement of the board; B, h, L, Z_allow in the unit of c)
    Z_3σ    = f_amp · n_σ · σ_z                                          (σ_z = RMS of the displacement of the board relative to its supports)
    MoS     = Z_allow / (J · Z_3σ) − 1

B = length of the board edge parallel to the component, h = board thickness, L = component length, C = component constant, r = relative position factor
(1 at the place of the largest displacement; `position_factor` gives the fundamental mode of a simply supported plate, an explicit value overrides it).
The constant c, the amplitude convention f_amp (1 = single, 2 = double amplitude), the number of sigma n_σ, the component constant C, the lowest position factor r_min,
the design factor J, and the fatigue law of the three-band damage (N = N_ref (Z_allow / Z)^b, damage limit) are NOT built in: they come from the confirmed requirement set
(`s4_limits`, keys `pcb_*`) – the program cannot know which convention the user's source uses. A value passed directly (also C and n_σ typed by a caller or the GUI) is used
but never confirmed: the verdict is then UNCONFIRMED_REQUIREMENTS, never PASS. The relative position factor r is bounded from below by r_min (a position near a support would
otherwise give an unbounded allowable): r_used = max(r, r_min). r and the RMS displacement are DATA of the check (listed with their source, not requirement-set rows); a zero
RMS displacement gives no verdict.

Three-band damage: the response is Gaussian; it spends the fractions p_i of the time with an amplitude of i·σ (i = 1, 2, 3; the fractions are the Gaussian
probabilities of the bands, computed here), at the rate of the board's natural frequency: D = Σ_i p_i · f_n · T / N(Z_i). The narrow-band (Rayleigh) damage of the
same process and the same displacement-life law from `fatigue.spectral` is reported next to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .notching import FAIL, PASS, UNCONFIRMED
from .s4_limits import S4Limits

MM_PER_IN = 25.4
AGREE_TOL = 0.01                         # numerical tolerance (1 %) of the agreement of a typed r / RMS displacement with the position-derived / model-derived value
R_NUMERIC_MIN = 1e-6                     # numerical guard only (a position factor below it divides by ~0); the usable lower bound is the requirement-set row pcb_position_factor_min


class PcbError(ValueError):
    """The PCB check inputs are inconsistent."""


@dataclass
class PcbSpec:
    """One component on one board. Lengths in mm. `r` explicit, or the position (x, y from the board corner; a × b board) for the simply supported plate."""
    name: str = "component"
    B_mm: float = 0.0                      # board edge parallel to the component
    h_mm: float = 0.0                      # board thickness
    L_mm: float = 0.0                      # component length
    C: float | None = None                 # component constant typed for THIS component (unconfirmed); None = the requirement-set row pcb_component_const
    r: float | None = None                 # relative position factor
    x_mm: float | None = None
    y_mm: float | None = None
    a_mm: float | None = None              # board size for the plate-mode position factor
    b_mm: float | None = None
    z_rms_mm: float | None = None          # RMS relative displacement of the board (1 sigma)
    f_n_Hz: float | None = None            # natural frequency of the board mode (cycle rate of the damage)
    duration_s: float | None = None        # exposure time of the damage
    n_sigma: float | None = None           # the "3 sigma" of the method: None = the requirement-set row pcb_n_sigma (a typed value stays unconfirmed)
    r_min: float | None = None             # typed lowest position factor (unconfirmed); None = the requirement-set row pcb_position_factor_min
    z_source: str = ""                     # where z_rms_mm comes from (modal analysis of the study | a report ...): a typed value WITHOUT a source is never confirmed
    r_source: str = ""                     # where a typed r comes from (a position-derived r needs none): a typed r without a source is never confirmed
    z_model_mm: float | None = None        # the RMS displacement derived from the model of the study, when there is one: a typed z_rms_mm that differs from it is never confirmed
    # direct (unconfirmed) overrides of the requirement-set values
    const: float | None = None
    const_unit: str | None = None          # "in" | "mm"
    amplitude_factor: float | None = None
    design_factor: float | None = None
    fatigue_exponent: float | None = None
    ref_cycles: float | None = None
    damage_limit: float | None = None


def position_factor(x_mm: float, y_mm: float, a_mm: float, b_mm: float) -> float:
    """r = sin(πx/a)·sin(πy/b): the fundamental mode of a simply supported rectangular plate, 1 at the centre."""
    if not (0.0 <= x_mm <= a_mm and 0.0 <= y_mm <= b_mm and a_mm > 0 and b_mm > 0):
        raise PcbError("The component position must lie on the board (0 <= x <= a, 0 <= y <= b)")
    return math.sin(math.pi * x_mm / a_mm) * math.sin(math.pi * y_mm / b_mm)


def norm_unit(unit: Any) -> str:
    """'in' | 'mm' from the text the user typed ('in', 'IN', 'inch', ' Inches ', 'mm', 'MM'; surrounding spaces ignored); '' for anything else."""
    u = str(unit or "").strip().lower().replace(" ", "")
    return {"in": "in", "inch": "in", "inches": "in", '"': "in", "mm": "mm", "millimeter": "mm", "millimetre": "mm", "millimeters": "mm", "millimetres": "mm"}.get(u, "")


def steinberg_allowable(B_mm: float, h_mm: float, L_mm: float, C: float, r: float, const: float, unit: str) -> float:
    """Z_allow [mm] = const·B/(C·h·r·√L) evaluated in the length unit of `const` ('in' or 'mm'; case and spaces are ignored, 'inch' = 'in')."""
    unit = norm_unit(unit)
    if unit not in ("in", "mm"):
        raise PcbError("The unit of the Steinberg constant must be 'in' or 'mm'")
    vals = (B_mm, h_mm, L_mm, C, r, const)
    if not (all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals) and B_mm > 0 and h_mm > 0 and L_mm > 0 and C > 0 and r >= R_NUMERIC_MIN and const > 0):
        raise PcbError("B, h, L, C, r and the constant must be finite and positive")
    k = 1.0 / MM_PER_IN if unit == "in" else 1.0
    return const * (B_mm * k) / (C * (h_mm * k) * r * math.sqrt(L_mm * k)) / k


def band_fractions() -> list[float]:
    """Time fractions of the amplitude bands 0–1σ, 1–2σ, 2–3σ of a Gaussian process (the band is represented by its upper level)."""
    e = [math.erf(i / math.sqrt(2.0)) for i in (1, 2, 3)]
    return [e[0], e[1] - e[0], e[2] - e[1]]


def three_band_damage(z_sigma_mm: float, amp_factor: float, z_allow_mm: float, exponent: float, n_ref: float, f_n: float, duration_s: float) -> dict:
    """Three-band damage of the displacement-life law N(Z) = n_ref (z_allow / Z)^exponent; Z_i = amp_factor·i·σ."""
    vals = (z_sigma_mm, amp_factor, z_allow_mm, exponent, n_ref, f_n, duration_s)
    if not (all(math.isfinite(v) for v in vals) and z_sigma_mm >= 0 and amp_factor > 0 and z_allow_mm > 0 and exponent > 0 and n_ref > 0 and f_n > 0 and duration_s >= 0):
        raise PcbError("The damage inputs must be finite and positive (sigma >= 0)")
    rows = []
    D = 0.0
    for i, p in enumerate(band_fractions(), 1):
        z = amp_factor * i * z_sigma_mm
        n = p * f_n * duration_s
        N = math.inf if z <= 0 else n_ref * (z_allow_mm / z) ** exponent
        d = 0.0 if math.isinf(N) else n / N
        D += d
        rows.append({"band_sigma": i, "fraction": p, "Z_mm": z, "cycles": n, "N_to_failure": N, "damage": d})
    return {"damage": D, "bands": rows}


def narrow_band_damage(z_sigma_mm: float, amp_factor: float, z_allow_mm: float, exponent: float, n_ref: float, f_n: float, duration_s: float) -> float:
    """The same displacement-life law as a Rayleigh (narrow band) damage by `fatigue.spectral` (S-N curve in displacement amplitudes, no knee)."""
    from ..fatigue.sn import SNCurve
    from ..fatigue.spectral import damage
    s2 = (amp_factor * z_sigma_mm) ** 2
    m = np.array([[s2, f_n * s2, f_n ** 2 * s2, f_n ** 4 * s2]])
    sn = SNCurve(SD=float(z_allow_mm), ND=float(n_ref), k=float(exponent), k2=float(exponent), source="displacement-life law of the PCB check")
    return float(damage(m, float(duration_s), sn, "nb")[0])


@dataclass
class PcbResult:
    spec: PcbSpec
    r: float | None
    z_allow_mm: float | None
    z_sigma_mm: float | None
    z_3sigma_mm: float | None
    design_factor: float | None
    mos: float | None
    damage: dict | None
    damage_nb: float | None
    damage_limit: float | None
    damage_ok: bool | None
    verdict: str
    verdict_text: str
    inputs: list[dict]
    warnings: list[str] = field(default_factory=list)
    n_sigma: float | None = None
    r_used: float | None = None
    C_used: float | None = None

    def to_dict(self) -> dict:
        return {"component": self.spec.name, "r": self.r, "r_used": self.r_used, "C": self.C_used, "Z_allow_mm": self.z_allow_mm, "Z_sigma_mm": self.z_sigma_mm,
                "Z_nsigma_mm": self.z_3sigma_mm, "n_sigma": self.n_sigma, "design_factor": self.design_factor, "MoS": self.mos, "damage_three_band": self.damage,
                "damage_narrow_band": self.damage_nb, "damage_limit": self.damage_limit, "damage_ok": self.damage_ok, "verdict": self.verdict,
                "verdict_text": self.verdict_text, "inputs": self.inputs, "warnings": self.warnings,
                "formula": "Z_allow = c*B/(C*h*r*sqrt(L)); MoS = Z_allow/(J*Z_nsigma) - 1 (constants and conventions entered by the user)"}


def _get(lim: S4Limits, key: str, override: float | None, name: str, inputs: list[dict]) -> tuple[float | None, bool]:
    v, ok, src = lim.get(key, override)
    inputs.append({"name": name, "key": key, "value": v, "source": src or "not available", "confirmed": ok})
    return v, ok


def _data(inputs: list[dict], name: str, value: Any, source: str, confirmed: bool | None = None) -> None:
    """A DATA input of the check (geometry, the response of the board): listed with its source; it is not a requirement-set row. confirmed = True when it is derived (from the
    position / the model of the study) or typed with a stated source, False when typed without a source or in conflict with the derived value (no PASS), None for plain geometry."""
    inputs.append({"name": name, "key": "", "value": value, "source": source, "confirmed": confirmed})


def _positive(name: str, v: float | None, strict: bool = True) -> None:
    if v is None:
        return
    if not isinstance(v, (int, float)) or not math.isfinite(v) or (v <= 0 if strict else v < 0):
        raise PcbError(f"{name} must be a finite {'positive' if strict else 'non-negative'} number")


def check_pcb(spec: PcbSpec, limits: S4Limits | None = None) -> PcbResult:
    """Steinberg check of one component. Every coefficient comes from the requirement set (or from `spec` overrides, never confirmed)."""
    lim = limits or S4Limits()
    inputs: list[dict] = []
    warnings: list[str] = []
    for name, v in (("B", spec.B_mm), ("h", spec.h_mm), ("L", spec.L_mm)):
        _positive(name, v)
    _positive("f_n", spec.f_n_Hz)
    _positive("duration", spec.duration_s, strict=False)
    _positive("z_rms_mm", spec.z_rms_mm, strict=False)
    data_issues: list[str] = []
    if spec.r is not None:
        r = float(spec.r)
        _positive("r", r)
        r_pos = None
        if None not in (spec.x_mm, spec.y_mm, spec.a_mm, spec.b_mm):
            try:
                r_pos = position_factor(float(spec.x_mm), float(spec.y_mm), float(spec.a_mm), float(spec.b_mm))  # pyright: ignore[reportArgumentType] – checked above
            except PcbError:
                r_pos = None
        if r_pos is not None and abs(r - r_pos) > AGREE_TOL * max(r_pos, 1e-12):
            _data(inputs, "relative position factor r", r, f"entered directly - DIFFERS from the position-derived value {r_pos:.4g}", False)
            data_issues.append(f"the typed r = {r:.4g} differs from the value derived from the position ({r_pos:.4g})")
        elif spec.r_source.strip():
            _data(inputs, "relative position factor r", r, spec.r_source.strip(), True)
        elif r_pos is not None:
            _data(inputs, "relative position factor r", r, "entered directly, agrees with the position-derived value", True)
        else:
            _data(inputs, "relative position factor r", r, "entered directly (no source stated)", False)
            data_issues.append("r was typed without a stated source (r_source)")
    elif spec.x_mm is not None and spec.y_mm is not None and spec.a_mm is not None and spec.b_mm is not None:
        r = position_factor(float(spec.x_mm), float(spec.y_mm), float(spec.a_mm), float(spec.b_mm))
        warnings.append("r = sin(pi x/a) sin(pi y/b) (simply supported plate); give r for another support condition")
        _data(inputs, "relative position factor r", r, "position on the board, r = sin(pi x/a) sin(pi y/b) (simply supported plate)", True)
    else:
        r = None
    const, c_ok = _get(lim, "pcb_steinberg_const", spec.const, "Steinberg constant c", inputs)
    unit_raw = spec.const_unit if spec.const_unit else lim.unit_of("pcb_steinberg_const")
    unit = norm_unit(unit_raw)
    if spec.const_unit:
        c_ok = False                                           # a unit typed by a caller is not the unit of the confirmed value
    if const is not None and not unit:
        warnings.append("The unit of the Steinberg constant ('in' or 'mm') is not given or not understood - Z_allow is not computed")
        const = None
    if const is not None:
        inputs[-1]["unit"] = unit
    amp, a_ok = _get(lim, "pcb_amplitude_factor", spec.amplitude_factor, "amplitude convention factor", inputs)
    J, j_ok = _get(lim, "pcb_design_factor", spec.design_factor, "design factor J", inputs)
    n_sig, ns_ok = _get(lim, "pcb_n_sigma", spec.n_sigma, "number of sigma n_sigma", inputs)
    C, C_ok = _get(lim, "pcb_component_const", spec.C, "component constant C", inputs)
    rmin, rmin_ok = _get(lim, "pcb_position_factor_min", spec.r_min, "lowest position factor r_min", inputs)
    for name, v in (("the Steinberg constant", const), ("the amplitude factor", amp), ("the design factor J", J), ("n_sigma", n_sig), ("C", C), ("r_min", rmin)):
        if v is not None and not (math.isfinite(v) and v > 0):
            raise PcbError(f"{name} must be a finite positive number (got {v})")
    r_used = None
    if r is not None:
        if rmin is not None:
            r_used = max(r, float(rmin))
            if r < rmin:
                warnings.append(f"r = {r:.4g} is below the lowest position factor {rmin:.4g}: r_min is used (conservative)")
        else:
            r_used = r
            warnings.append("No lowest position factor (pcb_position_factor_min) is available: r is used as given and the verdict stays unconfirmed")
        if r_used < R_NUMERIC_MIN:
            raise PcbError(f"r = {r:.3g} is too small (the position is at a support of the board: the allowable displacement is unbounded)")
    z_allow = None
    if const is not None and r_used is not None and C is not None:
        z_allow = steinberg_allowable(spec.B_mm, spec.h_mm, spec.L_mm, float(C), r_used, const, unit)
    else:
        missing = [n for n, v in (("the Steinberg constant", const), ("r (or the position and the board size)", r), ("the component constant C", C)) if v is None]
        warnings.append("Z_allow is not computed: missing " + ", ".join(missing))
    z_sigma = None if spec.z_rms_mm is None else float(spec.z_rms_mm)
    if z_sigma is None:
        warnings.append("The RMS relative displacement of the board is not given (z_rms_mm)")
    else:
        zm = spec.z_model_mm
        if zm is not None and abs(z_sigma - float(zm)) > AGREE_TOL * max(float(zm), 1e-300):
            _data(inputs, "RMS relative displacement of the board [mm]", z_sigma, f"entered directly - DIFFERS from the model-derived value {float(zm):.4g} mm", False)
            data_issues.append(f"the typed RMS displacement {z_sigma:.4g} mm differs from the value derived from the model ({float(zm):.4g} mm)")
        elif zm is not None or spec.z_source.strip():
            _data(inputs, "RMS relative displacement of the board [mm]", z_sigma, spec.z_source.strip() or "modal analysis of the study", True)
        else:
            _data(inputs, "RMS relative displacement of the board [mm]", z_sigma, "entered directly (no source stated)", False)
            data_issues.append("the RMS displacement was typed without a stated source (z_source)")
    zero_response = z_sigma is not None and z_sigma <= 0.0
    if zero_response:
        warnings.append("The RMS displacement of the board is zero: no displacement response, no verdict")
    z_n = None if z_sigma is None or amp is None or n_sig is None else float(amp) * float(n_sig) * z_sigma
    mos = None
    if z_allow is not None and z_n is not None and J is not None and z_n > 0:
        mos = z_allow / (J * z_n) - 1.0
    # three-band damage
    b, b_ok = _get(lim, "pcb_fatigue_exponent", spec.fatigue_exponent, "fatigue exponent b", inputs)
    nref, n_ok = _get(lim, "pcb_ref_cycles", spec.ref_cycles, "reference cycles N_ref", inputs)
    dlim, d_ok = _get(lim, "pcb_damage_limit", spec.damage_limit, "damage limit", inputs)
    for name, v in (("the fatigue exponent", b), ("N_ref", nref), ("the damage limit", dlim)):
        if v is not None and not (math.isfinite(v) and v > 0):
            raise PcbError(f"{name} must be a finite positive number (got {v})")
    damage = dnb = None
    damage_ok = None
    if (b is not None and nref is not None and z_allow is not None and z_sigma is not None and amp is not None and spec.f_n_Hz is not None
            and spec.duration_s is not None and not zero_response):
        damage = three_band_damage(z_sigma, amp, z_allow, b, nref, float(spec.f_n_Hz), float(spec.duration_s))
        dnb = narrow_band_damage(z_sigma, amp, z_allow, b, nref, float(spec.f_n_Hz), float(spec.duration_s))
        if dlim is not None:
            damage_ok = damage["damage"] <= dlim
    used_ok = [c_ok, a_ok, j_ok, ns_ok, C_ok, rmin_ok]
    if damage is not None:
        used_ok += [b_ok, n_ok, d_ok and dlim is not None]
    fatigue_missing = False
    if damage is None and spec.f_n_Hz is not None and spec.duration_s is not None and not zero_response:         # asked for, but not assessable: never a silent PASS
        fatigue_missing = True
        lacking = [n for n, v in (("the fatigue exponent (pcb_fatigue_exponent)", b), ("the reference cycles (pcb_ref_cycles)", nref), ("Z_allow", z_allow), ("the RMS displacement", z_sigma),
                                  ("the amplitude factor", amp)) if v is None]
        warnings.append("The fatigue damage is NOT assessed although f_n and the duration are given (missing: " + ", ".join(lacking or ["an input"]) + ") - there is no fatigue verdict")
    confirmed = all(used_ok) and z_allow is not None and mos is not None and not zero_response and not data_issues and not fatigue_missing
    if zero_response:
        verdict, text = UNCONFIRMED, "no verdict: the RMS displacement of the board is zero"
    elif data_issues:
        verdict, text = UNCONFIRMED, "no PASS: " + "; ".join(data_issues) + " - state the source of the input or take it from the model"
    elif fatigue_missing:
        verdict, text = UNCONFIRMED, "the fatigue damage was asked for (f_n and the duration are given) but cannot be assessed with the requirement set - no fatigue verdict, no PASS"
    elif not confirmed:
        verdict, text = UNCONFIRMED, "a coefficient or convention is not confirmed in the requirement set (or an input is missing) - indicative only"
    elif mos is not None and mos < 0 or damage_ok is False:
        verdict, text = FAIL, "the displacement margin is negative or the three-band damage exceeds its limit"
    else:
        verdict, text = PASS, "the displacement margin is non-negative with the confirmed coefficients"
    return PcbResult(spec, r, z_allow, z_sigma, z_n, J, mos, damage, dnb, dlim, damage_ok, verdict, text, inputs, warnings, n_sigma=n_sig, r_used=r_used, C_used=C)


# ------------------------------------------------------------------ the board displacement from a modal analysis
def relative_response(freq: np.ndarray, zeta: Any, gamma_d: np.ndarray, coef: np.ndarray, asd: Any, eta: float | None = None) -> dict:
    """RMS [mm] and the dominant frequency of Σ_j coef_j·q_j for a base ASD table [[f Hz, g²/Hz], …]: coef_j = the modal relative displacement (board centre minus supports,
    along the board normal) per unit mode; gamma_d = Γ_j in the base direction (CQC in the full form, `psd.modal_moments`). The dominant frequency is the rate of zero
    up-crossings of the Gaussian response, √(m2/m0), over the band of the ASD (m_p = ∫ f^p S df of the response PSD)."""
    from ..dynamics import psd
    t = np.asarray(asd, float)
    z = np.broadcast_to(np.asarray(zeta, float), np.shape(freq))
    f = psd.freq_grid(float(t[:, 0].min()), float(t[:, 0].max()), freq, z)  # pyright: ignore[reportArgumentType] – freq_grid takes a scalar or a per-mode array
    W = psd.asd_to_mm(psd.asd_interp(t, f))
    M = psd.modal_moments(f, freq, z, np.asarray(gamma_d, float), W, powers=(0, 2), eta=eta, kind="disp")
    c = np.asarray(coef, float)
    m0 = max(float(c @ M[0].real @ c), 0.0)
    m2 = max(float(c @ M[1].real @ c), 0.0)
    return {"z_rms_mm": math.sqrt(m0), "f_dominant_Hz": math.sqrt(m2 / m0) if m0 > 0 else 0.0}


def relative_rms(freq: np.ndarray, zeta: Any, gamma_d: np.ndarray, coef: np.ndarray, asd: Any, eta: float | None = None) -> float:
    """RMS [mm] of the relative displacement (see `relative_response`)."""
    return float(relative_response(freq, zeta, gamma_d, coef, asd, eta)["z_rms_mm"])


def board_modal_coefficients(study: Any, mesh: Any, basis: Any, centre: Any, supports: Any, normal: Any) -> np.ndarray:
    """coef_j (K,) of the relative out-of-plane displacement: the shape at the board centre minus the mean shape at the supports, projected on the normal.
    `supports` = a list of points [mm] or the name of a face selection (mean of its nodes)."""
    from ..dynamics.model import Probe
    from ..dynamics.response import probe_operators
    n = np.asarray(normal, float)
    if n.shape != (3,) or not np.linalg.norm(n) > 0:
        raise PcbError("normal must be a vector of three numbers")
    n = n / np.linalg.norm(n)
    probes = [Probe("centre", point=[float(v) for v in centre])]
    if isinstance(supports, str):
        probes.append(Probe("s0", selection=supports))
    else:
        probes += [Probe(f"s{i}", point=[float(v) for v in p]) for i, p in enumerate(supports)]
    if len(probes) < 2:
        raise PcbError("Give at least one support point or a support selection")
    ops = probe_operators(study, mesh, probes)
    phi = [np.einsum("p,kpi->ki", op["w"], np.asarray(basis.shapes)[:, op["nodes"]].astype(float)) @ n for op in ops]
    return phi[0] - np.mean(phi[1:], axis=0)


def pcb_basis(study: Any, mesh: Any, workdir: str | Path, *, modes: int | None = None, log: Callable[[str], None] | None = None, cancel: Any = None) -> Any:
    """The modal basis the board displacement of `z_rms_from_study` is taken from (it does not depend on the component: compute it once for a list of components)."""
    from ..dynamics.modal import run_basis
    return run_basis(study, mesh, Path(workdir), log or (lambda _s: None), None, cancel, modes=modes, job="pcb_modal", with_stress=False, with_reactions=False)       # shapes, Γ and frequencies only


def z_rms_from_study(study: Any, mesh: Any, workdir: str | Path, *, centre: Any, supports: Any, normal: Any, direction: Any, asd: Any, zeta: float | None,
                     modes: int | None = None, log: Callable[[str], None] | None = None, cancel: Any = None, basis: Any = None) -> dict:
    """RMS relative displacement of a board from the modal basis of the study (random base excitation `asd` in `direction`). zeta = a modal damping ratio, or None = the
    damping of the study (`dynamics.modal.zetas`, per mode). basis = the modal basis of this study / mesh / `modes` when the caller has it already (the components of one
    report share it: `pcb_basis`); None = it is computed here."""
    from ..dynamics.interface import unit_direction
    from ..dynamics.modal import zetas
    log = log or (lambda _s: None)
    if basis is None:
        basis = pcb_basis(study, mesh, workdir, modes=modes, log=log, cancel=cancel)
    coef = board_modal_coefficients(study, mesh, basis, centre, supports, normal)
    d = unit_direction(direction)
    gam = np.asarray(basis.gamma)[:, :3] @ d
    if zeta is None:
        z = zetas(study.dynamics.damping, basis.freq)
        z_src = f"damping of the study ({study.dynamics.damping.kind})"
    else:
        z = np.full(basis.K, float(zeta))
        z_src = f"zeta = {float(zeta):g} given"
    resp = relative_response(basis.freq, z, gam, coef, asd)
    t = np.asarray(asd, float)
    in_band = (basis.freq >= float(t[:, 0].min())) & (basis.freq <= float(t[:, 0].max()))
    score = np.abs(coef * gam) * np.where(in_band, 1.0, 0.0)
    j = int(np.argmax(score)) if score.max() > 0 else int(np.argmin(np.abs(basis.freq - resp["f_dominant_Hz"])))
    return {"z_rms_mm": resp["z_rms_mm"], "dominant_mode": j + 1, "dominant_mode_f_Hz": float(basis.freq[j]), "dominant_f_Hz": resp["f_dominant_Hz"], "modes": int(basis.K),
            "damping": z_src, "engine": getattr(basis, "engine", ""),
            "mass_fraction": np.asarray(basis.checks.get("mass_fraction_xyz", [])).tolist() if basis.checks else []}

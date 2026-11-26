"""The named limits and factors of stage S4 (interface forces / notching, ECSS bolts, PCB, rails) and where their values come from.

Standing rule of the F14 plan: the program ships NO standard number. Every number that turns a computed value into a verdict, and every coefficient of a formula
chain that belongs to a standard or a project requirement, is an entry of `RequirementSet.limits` (key, value, unit, document / revision / clause) and counts only
while the requirement set is confirmed (`SpaceSettings.requirement_confirmed()` – the same rule as for the S3 limits). A value that the caller passes directly is
used for the computation but is never confirmed: the result then carries the status `UNCONFIRMED_REQUIREMENTS` and never PASS.

This file names the keys and what they limit. It contains no value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ._checks_common import LimitEntry

# key -> (unit, what it is)
LIMIT_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # --- interface forces and notching
    "notch_min_width_oct": ("oct", "minimum width of a notch of the input profile (full width at the notch bottom)"),
    "notch_floor_ratio": ("1", "floor of a notch: the notched input is not lower than this fraction of the unnotched specification level"),
    "notch_force_limit_N": ("N", "limit of the interface force (launcher / project requirement) that a notched sine profile keeps"),
    "notch_moment_limit_Nmm": ("N*mm", "limit of the interface moment that a notched profile keeps"),
    "notch_force_psd_limit_N2_Hz": ("N^2/Hz", "limit of the interface force PSD that a notched ASD profile keeps (flat level; a table can be passed directly)"),
    "notch_probe_acc_limit_g": ("g", "limit of the response acceleration at the probe point that a notched profile keeps"),
    "notch_stress_limit_MPa": ("MPa", "limit of the equivalent stress at the governing point that a notched profile keeps (allowable / design factor, entered by the user)"),
    "notch_design_factor": ("1", "factor J applied to the predicted AMPLITUDE of the limited quantity (force, moment, acceleration, stress) before it is compared with the limit; "
                                 "on a PSD limit of a random profile it acts as J^2 (a PSD is an amplitude squared); 1 = none"),
    # --- ECSS bolted joints: the formula chain of ECSS-E-HB-32-23A Rev.1 is built in (`bolts_ecss`); every coefficient and factor is entered
    "bolt_tightening_scatter": ("1", "uncertainty u of the preload from the tightening method (ECSS-E-HB-32-23A Rev.1, 6.3.2.3 / Table 6-1): F_V,max = F_nom (1 + u), F_V,min = F_nom (1 - u) - losses"),
    "bolt_relaxation_loss": ("1", "fraction of the nominal preload lost by the other relaxation mechanisms (group interaction, gasket / metallic creep, torsional relaxation; 6.4); embedding is computed separately"),
    "bolt_slip_mu_min": ("1", "minimum slip (friction) coefficient of the faying surfaces used for the slip check (9.2.2.4: the minimum of its range)"),
    "bolt_factor_gapping": ("1", "safety factor J of the gapping (separation) margin (the factor of Table 5-4 / ECSS-E-ST-32-10C Table 4-4 for gapping / sliding)"),
    "bolt_factor_slip": ("1", "safety factor J of the slip margin (ultimate factor, 9.2.3)"),
    "bolt_slip_planes_max": ("1", "largest number of friction interfaces per bolt that the slip check may count (more than 2 needs this confirmed row)"),
    "bolt_factor_tensile_yield": ("1", "safety factor J (yield) applied to the EXTERNAL load (and, as J_T, to the thermally induced increment; never to the preload) in the bolt tensile, combined (yield), shear (yield) and the yield margins of crushing and hole bearing"),
    "bolt_factor_tensile": ("1", "safety factor J (ultimate) applied to the EXTERNAL load and, as J_T, to the thermally induced increment in the bolt tensile margin (never to the preload, 7.9.2)"),
    "bolt_factor_shear": ("1", "safety factor J (ultimate) of the bolt shear margin"),
    "bolt_factor_combined": ("1", "safety factor J (ultimate) of the combined tension + shear margin (applied to the external loads and, as J_T, to the thermally induced increment; not to the preload)"),
    "bolt_factor_pullout": ("1", "safety factor J (ultimate) of the thread pull-out margin (applied to the external load and, as J_T, to the thermally induced increment; not to the preload, 7.10.4)"),
    "bolt_factor_bearing": ("1", "safety factor J of the crushing (under the head / nut, applied to the external load and, as J_T, to the thermally induced increment; not to the preload) and hole bearing margins"),
    "bolt_thermal_factor": ("1", "optional factor J_T of the thermally induced bolt load (ECSS-E-ST-32-10C 4.3.2.1 b with 4.3.2.2 a, 4.3.2.2 b, Table 4-4 notes a / e); without a value each check applies its own factor J "
                                 "(not below 1) to the thermal increment (conservative); 1 = none (e.g. expendable launch vehicles, note e); a value below 1 is not accepted as confirmed"),
    # --- PCB (Steinberg)
    "pcb_steinberg_const": ("in", "constant of the allowable relative displacement Z_allow = c * B / (C * h * r * sqrt(L)); the unit says in which length unit B, h, L, Z are"),
    "pcb_amplitude_factor": ("1", "factor from the 3-sigma RMS displacement to the displacement measure of the allowable (1 = single amplitude, 2 = double amplitude)"),
    "pcb_design_factor": ("1", "design factor J of the PCB displacement margin"),
    "pcb_n_sigma": ("1", "number of sigma of the RMS relative displacement that is compared with the allowable (the '3 sigma' of the method)"),
    "pcb_component_const": ("1", "component constant C of the Steinberg formula for the component type being checked (a value typed per component stays unconfirmed)"),
    "pcb_position_factor_min": ("1", "smallest relative position factor r that may be used (r -> 0 near a board support gives an unbounded allowable displacement)"),
    "pcb_fatigue_exponent": ("1", "exponent b of the displacement-life law N = N_ref (Z_allow / Z)^b"),
    "pcb_ref_cycles": ("1", "number of cycles N_ref to failure at the amplitude Z_allow"),
    "pcb_damage_limit": ("1", "largest cumulative damage ratio of the three-band check"),
    # --- CubeSat deployer rails
    "rail_reaction_limit_N": ("N", "limit of the reaction of one rail support (deployer interface)"),
    "rail_lateral_deflection_limit_mm": ("mm", "limit of the lateral deflection of a rail (deployer clearance)"),
    "rail_axial_reaction_limit_N": ("N", "limit of the axial reaction at the rail ends"),
}

S4_KEYS = tuple(LIMIT_DESCRIPTIONS)
# rows that an older version of the program wrote and that are no longer read: key -> why (shown as obsolete, never used, never deleted: removing a row would change the
# digest of the requirement set and withdraw its confirmation)
OBSOLETE_LIMITS: dict[str, str] = {
    "bolt_combined_exponent": "obsolete, ignored: the tension + shear interaction follows ECSS-E-HB-32-23A Rev.1 9.3.2.3 (exponent from the bolt input shear_plane); the row has no effect",
}
PLACEHOLDER = "<clause>"


@dataclass
class S4Limits:
    """The S4 limits that a calculation may use: `entries[key]` has a value and a `confirmed` flag (+ provenance)."""
    entries: dict[str, LimitEntry] = field(default_factory=dict)
    set_confirmed: bool = False
    units: dict[str, str] = field(default_factory=dict)          # the unit text the user wrote with the value (the Steinberg constant needs it)
    obsolete: list[str] = field(default_factory=list)            # keys of rows of the requirement set that this version does not read any more (`OBSOLETE_LIMITS`)

    @classmethod
    def from_requirement_set(cls, space: Any) -> "S4Limits":
        """The limits of the requirement set in `SpaceSettings` `space` (None allowed). An entry is usable when the set is confirmed (content + structure classes) and the
        entry has a value and document / revision / clause."""
        out = cls()
        rs = getattr(space, "requirement_set", None)
        if rs is None:
            return out
        out.set_confirmed = bool(space.requirement_confirmed())
        for x in rs.limits:
            if x.key in OBSOLETE_LIMITS and x.key not in out.obsolete:
                out.obsolete.append(x.key)
            if x.key not in LIMIT_DESCRIPTIONS or x.value is None or not math.isfinite(x.value):
                continue
            prov = bool(x.document.strip() and x.revision.strip() and x.clause.strip() and x.clause.strip() != PLACEHOLDER)
            out.entries[x.key] = LimitEntry(float(x.value), out.set_confirmed and prov, x.document, x.revision, x.clause,
                                            rs.confirmed_by if out.set_confirmed else "", rs.date if out.set_confirmed else "")
            out.units[x.key] = x.unit.strip()
        return out

    def get(self, key: str, override: float | None = None) -> tuple[float | None, bool, str]:
        """(value, confirmed, source) of a limit: a caller's `override` wins and is never confirmed; otherwise the entry of the requirement set (confirmed only when
        the set and the entry are); (None, False, "") when the limit is not available at all."""
        if override is not None:
            return float(override), False, "entered directly (not confirmed)"
        e = self.entries.get(key)
        if e is None or e.value is None:
            return None, False, ""
        src = f"{e.document} {e.revision}, {e.clause}".strip(" ,")
        return float(e.value), bool(e.confirmed), src if e.confirmed else (src + " (requirement set not confirmed)").strip()

    def unit_of(self, key: str) -> str:
        """The unit the user entered with the value of a requirement-set limit ("" = none given)."""
        return self.units.get(key, "")


def template() -> list[dict]:
    """The empty template of the S4 limits (key, unit, meaning; no value): what a requirement set has to list to let the checks reach a verdict."""
    return [{"key": k, "unit": u, "meaning": m, "value": None} for k, (u, m) in LIMIT_DESCRIPTIONS.items()]


def install_template(rs: Any) -> list[str]:
    """Adds the S4 limit keys (value None, clause placeholder) that the requirement set does not have yet; returns the keys added. Never overwrites a value."""
    from .model import LimitEntry as RsLimit
    have = {x.key for x in rs.limits}
    added = []
    for k, (u, _m) in LIMIT_DESCRIPTIONS.items():
        if k not in have:
            rs.limits.append(RsLimit(key=k, value=None, unit=u, clause=PLACEHOLDER))
            added.append(k)
    return added

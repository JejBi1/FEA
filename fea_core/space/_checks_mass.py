"""Check 1 – mass and centre of mass of the FE model against the budget / the CAD volumes, and as the SOLVER sees them (unit gravity).

The FE mass is integrated here on the mesh with the same 14-point rule and the same per-element density as the dynamics (`dynamics.mass`, `printing.element_density`),
plus the added masses of the study (point masses at their point, surface masses at the weights of their face). The solver's view = three unit-gravity statics runs
(`solve.run_study`): the sum of the reactions is M·g and the moment of the reactions gives the centre of mass – this catches a density / added mass that the solver
takes differently from the integration. The statics steps are kept for the equilibrium check (7).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._checks_common import DEFAULT_DENSITY_RANGE, CheckContext, CheckResult, Judge, range_widened, skipped
from ._checks_run import run_statics

G_MM_S2 = 9810.0                                  # the program's convention (CLAUDE.md): g = 9810 mm/s²


@dataclass
class MassProps:
    mass_t: float
    bodies_t: float
    added_t: float
    cog: np.ndarray
    volume_mm3: float
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)
    added: list[dict[str, Any]] = field(default_factory=list)


def mass_properties(study: Any, mesh: Any) -> MassProps:
    """Total mass [t], centre of mass [mm], per-part mass / volume / centre and the added masses of the model check (check 1).

    ONE implementation of the mass: this is `fea_core.space.mass_props.mass_properties` (stage S2: the budgets, the report and the MCP tool use it) in the form the
    S3 checks read (tet10 meshes only, per-part density and element count, added masses in tonnes), so a report cannot show two masses that disagree. The mass
    of the model check and the mass budget of S2 are therefore the same number; the S2 budget limits (`mass_max_kg` ...) judge it, the check-1 budget options
    (`mass_budget_t`, `cog_budget_mm`) compare it with the user's budget of the check. A rejected added mass (<= 0, a non-physical inertia tensor) is a `ValueError`."""
    from .mass_props import mass_properties as _mass
    if mesh.tets.shape[1] != 10:
        raise ValueError("The mass integration is available for tet10 meshes")
    p = _mass(study, mesh)
    parts: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(mesh.part_names):
        v = p.parts[str(name)]
        vol = float(v["volume_mm3"])
        parts[str(name)] = {"mass_t": float(v["mass_t"]), "volume_mm3": vol, "rho_t_mm3": float(v["mass_t"] / vol) if vol > 0 else 0.0, "cog_mm": list(v["cog_mm"]),
                            "elements": int((mesh.tet_part == i).sum())}
    added = [{"name": a["name"], "kind": a["kind"], "mass_t": float(a["mass_kg"]) / 1000.0, "point_mm": list(a["point_mm"])} for a in p.added]
    return MassProps(p.mass_t, p.bodies_t, p.added_t, np.asarray(p.cog, float), p.volume_mm3, parts, added)


def cad_volumes(parts: Any) -> dict[str, float]:
    """{part name: CAD volume [mm³]} of the geometry parts (OCP, `geometry.Part.volume`)."""
    return {str(p.name): float(p.volume()) for p in parts or []}


def unit_gravity_cases() -> list[Any]:
    from ..study import Gravity, LoadCase
    return [LoadCase(f"Unit gravity {ax}", [Gravity(vector=[G_MM_S2 * (i == k) for i in range(3)])]) for k, ax in enumerate(("X", "Y", "Z"))]


def gravity_mass_and_cog(res: Any) -> dict[str, Any]:
    """The mass and the centre of mass from the reactions of the three unit-gravity steps: the reaction total is -M·g·e_k, the moment of the reactions about the
    moment point c0 is -(r_c - c0) x (M·g·e_k)."""
    steps = res.steps[:3]
    m_est: list[float] = []
    lateral: list[float] = []
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for k, st in enumerate(steps):
        e = st.equilibrium
        R = np.asarray(e["reaction_total"], float)
        MR = np.asarray(e["reaction_moment_total"], float)
        m_est.append(float(-R[k] / G_MM_S2))
        lateral.append(float(np.linalg.norm(np.delete(R, k)) / max(abs(R[k]), 1e-300)))
        ek = np.zeros(3)
        ek[k] = 1.0
        skew = np.array([[0.0, -ek[2], ek[1]], [ek[2], 0.0, -ek[0]], [-ek[1], ek[0], 0.0]])      # [e]x
        m_ref = max(m_est[-1], 1e-300)
        rows.append(skew)                                        # [e]x r' = M_R / (M g)
        rhs.extend((MR / (m_ref * G_MM_S2)).tolist())
    c0 = np.asarray(steps[0].moment_point if steps[0].moment_point is not None else np.zeros(3), float)
    sol, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(rhs), rcond=None)
    return {"mass_t": float(np.mean(m_est)), "mass_t_per_axis": m_est, "lateral_reaction_share": lateral, "cog_mm": (sol + c0).tolist()}


def run_mass_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    opt = ctx.options
    notes: list[str] = []
    try:
        props = mass_properties(ctx.study, ctx.mesh)
    except ValueError as e:
        if "tet10" in str(e):
            return skipped(1, str(e), time.time() - t0)
        judge.violate(f"the added masses are not valid: {e}")             # an added mass of 0 or less / a non-physical inertia is an error of the model, not a skipped check
        return judge.finish(1, "the mass cannot be integrated", {}, notes, time.time() - t0)
    ctx.shared["mass"] = props
    values: dict[str, Any] = {
        "fe_mass_t": props.mass_t, "fe_mass_kg": props.mass_t * 1000.0, "bodies_t": props.bodies_t, "added_t": props.added_t,
        "cog_mm": props.cog.tolist(), "volume_mm3": props.volume_mm3,
        "parts": {k: {"mass_kg": v["mass_t"] * 1000.0, "share": v["mass_t"] / props.mass_t if props.mass_t > 0 else 0.0, "rho_t_mm3": v["rho_t_mm3"],
                      "cog_mm": v["cog_mm"], "elements": v["elements"]} for k, v in props.parts.items()},
        "added_masses": props.added,
    }
    if not (np.isfinite(props.mass_t) and props.mass_t > 0):
        judge.violate(f"the FE mass is {props.mass_t:.3g} t – not a positive number")
    # --- the plausibility of the densities (a unit error is a factor 1000): a warning, the budget below decides
    lo, hi = opt.density_range_t_mm3
    if range_widened(opt.density_range_t_mm3, DEFAULT_DENSITY_RANGE):
        judge.cannot_decide(f"the plausible density range {lo:.2g}–{hi:.2g} t/mm3 is wider than the default {DEFAULT_DENSITY_RANGE[0]:.2g}–{DEFAULT_DENSITY_RANGE[1]:.2g}: the screening "
                            "of the densities for a unit error is weakened, so the check is not complete")
    for name, v in props.parts.items():
        if v["rho_t_mm3"] > 0 and not lo <= v["rho_t_mm3"] <= hi:
            judge.warn(f"part '{name}': density {v['rho_t_mm3']:.3g} t/mm3 is outside the plausible range {lo:.2g}–{hi:.2g} t/mm3 (a unit error? steel is 7.85e-9)")
    if ctx.study.active_bolts():
        notes.append(f"{len(ctx.study.active_bolts())} bolt(s) are replaced by beam elements and are not part of this mass – add their mass as point masses or compare with the budget WITHOUT fasteners")
    if getattr(ctx.study.frame, "active", False):
        notes.append("the members of the beam system are not part of this mass")
    # --- the budget
    if opt.mass_budget_t is not None:
        rel = (props.mass_t - float(opt.mass_budget_t)) / float(opt.mass_budget_t)
        values["mass_budget_t"], values["mass_rel_error"] = float(opt.mass_budget_t), rel
        judge.compare_max(abs(rel), "mass_tol_rel", f"|FE mass - budget| / budget (FE {props.mass_t * 1000:.4g} kg, budget {opt.mass_budget_t * 1000:.4g} kg)")
    else:
        notes.append("no mass budget given – the FE mass is only reported")
    if opt.cog_budget_mm is not None:
        d = float(np.linalg.norm(props.cog - np.asarray(opt.cog_budget_mm, float)))
        values["cog_budget_mm"], values["cog_error_mm"] = list(map(float, opt.cog_budget_mm)), d
        judge.compare_max(d, "cog_tol_mm", f"distance of the FE centre of mass from the budget position (FE {np.round(props.cog, 2).tolist()} mm)")
    for name, b in opt.part_mass_budget_t.items():
        if name not in props.parts:
            judge.violate(f"the mass budget names a part '{name}' that is not in the mesh")
            continue
        rel = (props.parts[name]["mass_t"] - float(b)) / float(b)
        values.setdefault("part_budget_errors", {})[name] = rel
        judge.compare_max(abs(rel), "part_mass_tol_rel", f"|FE mass - budget| / budget of part '{name}'")
    # --- the CAD volumes
    if ctx.parts:
        cad = cad_volumes(ctx.parts)
        rows: dict[str, Any] = {}
        for name, v in props.parts.items():
            if name in cad and cad[name] > 0 and v["rho_t_mm3"] > 0:
                if ctx.study.material_of(name).is_print:
                    notes.append(f"part '{name}' is 3D printed (shell + infill): its mass is not compared with the CAD volume")
                    continue
                m_cad = cad[name] * v["rho_t_mm3"]
                rel = (v["mass_t"] - m_cad) / m_cad
                rows[name] = {"cad_volume_mm3": cad[name], "fe_volume_mm3": v["volume_mm3"], "cad_mass_kg": m_cad * 1000.0, "fe_mass_kg": v["mass_t"] * 1000.0,
                              "rel_error": rel}
                judge.compare_max(abs(rel), "cad_mass_tol_rel", f"|FE mass - CAD volume x density| / CAD mass of part '{name}'")
        values["cad"] = rows
        missing = [n for n in props.parts if n not in cad]
        if missing:
            notes.append(f"no CAD volume for the part(s) {', '.join(missing)} (names differ from the geometry)")
    else:
        notes.append("the geometry parts were not given – the FE volumes are not compared with the CAD volumes")
    # --- the solver's view: unit gravity
    if not opt.solver_mass_check:
        judge.cannot_decide("the solver cross-check of the mass (the unit-gravity statics, solver_mass_check) was switched off: the mass and the centre of mass as the solver sees them "
                            "were not verified, so the check is not complete")
    if opt.solver_mass_check:
        if not ctx.study.constraints:
            notes.append("unit-gravity run skipped: the study has no support (a free body has no reaction)")
        else:
            try:
                res = run_statics(ctx, unit_gravity_cases(), "check_gravity")
                ctx.shared["gravity_results"] = res
                g = gravity_mass_and_cog(res)
                rel = (g["mass_t"] - props.mass_t) / props.mass_t
                dcog = float(np.linalg.norm(np.asarray(g["cog_mm"]) - props.cog))
                values["solver"] = {**g, "mass_rel_diff_to_integration": rel, "cog_diff_to_integration_mm": dcog}
                judge.compare_max(abs(rel), "solver_mass_tol_rel", "|mass of the solver (unit-gravity reactions) - integrated mass| / integrated mass")
                if opt.mass_budget_t is not None:
                    values["solver"]["mass_rel_error_to_budget"] = (g["mass_t"] - float(opt.mass_budget_t)) / float(opt.mass_budget_t)
            except Exception as e:  # noqa: BLE001 – the unit-gravity run is a cross-check: its failure is reported, the integrated mass stands
                notes.append(f"unit-gravity run failed: {str(e).splitlines()[0][:200] if str(e) else type(e).__name__}")
                judge.warn("the unit-gravity statics could not be run – the mass seen by the solver is not verified")
    return judge.finish(1, f"FE mass {props.mass_t * 1000:.4g} kg, centre of mass {np.round(props.cog, 2).tolist()} mm", values, notes, time.time() - t0)

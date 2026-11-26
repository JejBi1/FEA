"""MCP group "space", stage S4: interface forces, notching, ECSS bolts, PCB check, CubeSat rails (`fea_core.space`, `fea_core.dynamics.interface`).

No standard number is built in: limits, factors and conventions come from the requirement set (`fea_space_requirement_set`, data.limits with document / revision /
clause) and count only while it is confirmed by the USER. A number passed to these tools is used for the calculation but never confirmed (verdict
UNCONFIRMED_REQUIREMENTS). The bolt formula chain follows ECSS-E-HB-32-23A Rev.1 (clause per formula; the deviations of the program are returned as `deviations`). The calculations are in
`fea_core.space.s4_run` (shared with the GUI); this module only maps the arguments and the errors, and records the summaries in the study (`s4_store`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import out
from .session import FeaError, Session, user_path


def _write(path: str, text: str) -> str:
    p = user_path(path)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="")
    return str(p)


def register(mcp: Any, sess: Session) -> None:
    from ..space import s4_run, s4_store

    def cache() -> dict:
        return sess.__dict__.setdefault("space_s4_cache", {})

    def interface(modes: int | None, ref_point: list[float] | None = None) -> tuple[Any, bool]:
        return s4_run.get_interface(sess.require_study(), sess.valid_mesh(), lambda: sess.new_workdir("space_interface"), cache(), modes, ref_point, log=sess.log.append)

    @mcp.tool(structured_output=False)
    def fea_space_interface_forces(directions: str = "x,y,z", sine: list | None = None, asd: list | None = None, zeta: float | None = None, modes: int | None = None,
                                   group: str | None = None, ref_point: list[float] | None = None) -> str:
        """Interface force / moment of the base excitation (modal base reactions per mode, sine spectrum, random RMS). Needs a mesh and supports; runs the modal basis (cached until the study changes).
        directions = x,y,z; sine = [[f Hz, a g], ...] (default: base_acc table of the harmonic settings); asd = [[f Hz, g2/Hz], ...] (default: random settings); zeta (default: damping of the study; uncertain, ±50 %); modes; group = a constraint selection or 'all'; ref_point [x,y,z] mm = moment reference (default: centroid of the held nodes). The force is the support force of Newton's law (elastic reaction + the damping force of the modes), only the HELD components of the held nodes (sliding supports, cylindrical supports and elastic-only models are handled). Returns the static check (f→0 reaction vs total mass·a), the modal table, per direction the peak |F|, |M| of the sine and the RMS / 3-sigma of the random. Force in N, moment in N·mm."""
        with sess.use(op="interface forces"):
            st = sess.require_study()
            try:
                s4_run.directions_list(directions)                       # cheap validation before the modal analysis
                mi, cached = interface(modes, ref_point)
                d = s4_run.interface_report(st, mi, directions, sine, asd, zeta, group, cached)
            except s4_run.S4Error as e:
                raise FeaError(str(e)) from None
            if "directions" in d and any(d["directions"].values()):
                s4_store.record(st, "interface", d)
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_space_notch(direction: str = "x", kind: str = "sine", limit_kind: str = "force", limit: float | None = None, limit_table: list | None = None,
                        profile: list | None = None, design_factor: float | None = None, min_width_oct: float | None = None, floor_ratio: float | None = None,
                        group: str | None = None, probe: str | None = None, zeta: float | None = None, modes: int | None = None, export_path: str | None = None) -> str:
        """Primary notching of a sine (kind sine) or random (kind random) input profile so that a limit holds: limit_kind force N | moment N·mm | probe_acc g | stress MPa (sine; probe_acc/stress use the harmonic base-acceleration result, probe = its probe name) | force_psd N²/Hz (random; flat `limit` or limit_table [[f, N²/Hz]]).
        The limit, design factor, minimum notch width [oct] and floor ratio come from the confirmed requirement set (limits notch_*); numbers passed here are used but never confirmed (verdict UNCONFIRMED_REQUIREMENTS, never PASS). profile = [[f Hz, g]] or [[f Hz, g2/Hz]] (default: study dynamics settings). A non-finite input or J <= 0 is an error; on a PSD limit the design factor acts as J². probe_acc / stress limits need a harmonic result of ONE base direction (a force excitation is refused) and hold at the probe only (the verdict says so). The breakpoint table never exceeds the limit (re-evaluated between the grid points, grid refined automatically); table_ratio = J·response of the table / limit, accepted up to 1 + table_tolerance. A probe_acc / stress notch gets no PASS (no re-evaluation between its frequencies). Only the first 80 breakpoints are listed here (breakpoints_truncated), export_path = CSV of the full profile (+ `<name>_justification.csv`). Returns the notch justification table (unnotched vs notched, dB reduction, remaining margin), breakpoints and the verdict."""
        from ..space import notching as NT
        with sess.use(op="notching"):
            st = sess.require_study()
            try:
                dyn = None
                if limit_kind in ("probe_acc", "stress"):
                    dyn = getattr(sess.require_results(), "dynamics", None)
                d, res = s4_run.notch_report(st, kind, direction, limit_kind, limit=limit, limit_table=limit_table, profile=profile, design_factor=design_factor,
                                             min_width_oct=min_width_oct, floor_ratio=floor_ratio, group=group, probe=probe, zeta=zeta,
                                             interface=lambda: interface(modes)[0], dynamics=dyn)
            except s4_run.S4Error as e:
                raise FeaError(str(e)) from None
            if export_path:
                p = _write(export_path, NT.profile_csv(res, name=f"{kind} notched profile, {limit_kind} limit"))
                q = _write(str(Path(p).with_suffix("")) + "_justification.csv", NT.justification_csv(res))
                d["exported"] = [p, q]
            s4_store.record(st, "notch", {k: v for k, v in d.items() if k not in ("exported", "limits_needed")}, f"{kind} {direction} {limit_kind}")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_space_bolts_ecss(bolts: list[dict], bolt_material: str | None = None, part_material: str | None = None, bearing_allowable: str = "Fu") -> str:
        """Bolt margins after ECSS-E-HB-32-23A Rev.1 (2023): preload range, thermal preload change, gapping, slip, tensile, shear, combined, thread pull-out, crushing, hole bearing; `formula` of each margin gives the clause, `deviations` what differs. J multiplies the EXTERNAL load (not the preload, 7.9.2) and the thermal increment (J_T = J of the check, or the row bolt_thermal_factor). Factors / coefficients come from the confirmed requirement set (limits bolt_*); numbers passed here are never confirmed. Pull-out needs both shear allowables; missing embedding_um (0 = none), thermal data, bolt Fsy (with F_sh), part Fy or a nut s_w/d outside 1.4-1.9 blocks a PASS; n_slip_planes > 2 needs the row bolt_slip_planes_max.
        bolts = [{name, joint (bolt of the study: stiffness, A_s, preload from VDI 2230) or d_mm, pitch_mm, A_s_mm2, k_b, k_c [N/mm], F_nom N, embedding_um, layers [{t_mm, alpha}], alpha_bolt, T_ref, T_min, T_max °C, F_ax, F_sh N, n_slip_planes, engaged_mm, d_w_mm, d_h_mm, t_bearing_mm, n_load_plane, F_K_req N, F_nom_min N (prevailing torque), wrench_mm (nut; empty = tapped hole), shear_plane thread|shank}]. Allowables [MPa] from the study materials (confirmed A/B only): bolt_material → Fy, Fu, Fsy, Fsu; part_material → Fsu (thread), Fy (yield bearing), bearing_allowable key (Fu | Fy) for ultimate bearing."""
        with sess.use(op="ECSS bolts"):
            st = sess.require_study()
            try:
                d = s4_run.bolts_report(st, bolts, bolt_material, part_material, bearing_allowable)
            except s4_run.S4Error as e:
                raise FeaError(str(e)) from None
            for r in d["bolts"]:
                s4_store.record(st, "bolts", r, str(r["bolt"]))
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_space_pcb_check(components: list[dict], direction: str = "z", asd: list | None = None, zeta: float | None = None, modes: int | None = None) -> str:
        """PCB check by the Steinberg method: allowable relative displacement Z_allow = c·B/(C·h·r·√L), margin MoS = Z_allow/(J·Z_nσ) − 1, three-band damage against the narrow-band damage of fea_core.fatigue.spectral. The constant c (with its unit in/mm), the amplitude convention, J and the fatigue law (b, N_ref, damage limit) come from the confirmed requirement set (limits pcb_*); numbers passed here are never confirmed.
        C, n_sigma and the lowest position factor are requirement-set rows (pcb_component_const, pcb_n_sigma, pcb_position_factor_min): a value typed here is unconfirmed and never gives PASS; zero RMS displacement gives no verdict. A typed r needs r_source, a typed z_rms_mm needs z_source; either one differing from the position-derived / model-derived value is UNCONFIRMED. f_n_Hz + duration_h without the fatigue rows of the requirement set give no fatigue verdict and no PASS.
        components = [{name, B_mm, h_mm, L_mm, C (optional, unconfirmed), r (+ r_source) | x_mm,y_mm,a_mm,b_mm, z_rms_mm (+ z_source) | geometry {centre [x,y,z], supports [[x,y,z],...] | support_selection, normal [nx,ny,nz]} (RMS relative displacement from the modal basis of the study under the ASD `asd` in `direction`, damping zeta), f_n_Hz, duration_h}]."""
        with sess.use(op="PCB check"):
            st = sess.require_study()
            try:
                d = s4_run.pcb_report(st, components, mesh=sess.valid_mesh(), workdir=lambda: sess.new_workdir("space_pcb"), direction=direction, asd=asd, zeta=zeta,
                                      modes=modes, log=sess.log.append)
            except s4_run.S4Error as e:
                raise FeaError(str(e)) from None
            for r in d["components"]:
                s4_store.record(st, "pcb", r, str(r["component"]))
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_space_rails(action: str = "check", rail_parts: list[str] | None = None, axis: str = "x", axial: str = "none", reaction_limit_N: float | None = None,
                        axial_reaction_limit_N: float | None = None, lateral_deflection_limit_mm: float | None = None) -> str:
        """CubeSat deployer rails. action 'apply' = write the rail supports into the study (wall-normal Displacement on the outer faces of the rails, axial support at the rail end(s) axial = none|min|max|both along the deployment axis); 'check' = rail reactions and lateral deflection of the solved study against the limits of the requirement set (rail_reaction_limit_N, rail_axial_reaction_limit_N, rail_lateral_deflection_limit_mm; numbers passed here are never confirmed). A wall or an end plate that would have to pull (in ANY state) is flagged; the margin uses the conservative force |push| + |pull of the opposite wall / end plate|. 'apply' removes EVERY earlier rail support and stores the setup (part, kind, axis, sign) in the study; rail_parts empty on 'check' = every rail of the stored setup; a rail without a support row gives no verdict."""
        from ..space import rails as RL
        with sess.use(op="rails"):
            st = sess.require_study()
            spec = RL.RailSpec(list(rail_parts or []), axis, axial, reaction_limit_N=reaction_limit_N, axial_reaction_limit_N=axial_reaction_limit_N,
                               lateral_deflection_limit_mm=lateral_deflection_limit_mm)
            try:
                if action == "apply":
                    r = RL.apply_rail_supports(st, sess.parts, spec)
                    sess.changed()
                    return out(r)
                if action == "check":
                    rr = RL.check_rails(st, sess.require_results(), spec, s4_run.limits_of(st)).to_dict()
                    s4_store.record(st, "rails", rr)
                    return out(rr)
            except (RL.RailError, ArithmeticError) as e:
                raise FeaError(str(e)) from None
            raise FeaError("action must be apply | check")

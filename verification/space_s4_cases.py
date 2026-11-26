"""Verification of the interface forces, notching, ECSS bolts and PCB check of the spacecraft verification (F14 S4): hand-calculated reference cases.

S4A  one-DOF structure: the interface force at f -> 0 is M_total * a, at the resonance m a sqrt(1 + 4 zeta^2) / (2 zeta) (Newton, with the damping force); the notch holds the limit exactly
S4B  PCB, Steinberg: Z_allow, the margin and the three-band damage by hand arithmetic (the constants are TEST INPUTS of the hand calculation, not program values)
S4C  bolted joint after ECSS-E-HB-32-23A Rev.1: preload range, thermal preload change (6.3.5), gapping (7.8.1), slip (9.2.6), tensile with J on the external load only (7.9.2 / 7.9.3),
     shear and combined (9.3.1 - 9.3.14), pull-out with L_eng - 2 p (7.10) of an aluminium plate with a steel and a titanium bolt by hand (arbitrary test coefficients)
S4D  a real bar (Gmsh mesh, CalculiX modal basis): the quasi-static limit of the interface reaction = M_total * a_base, the moment = M a * lever arm

No standard value is used anywhere: every coefficient is an arbitrary test number entered through a (test) requirement set or as a direct, unconfirmed input.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.dynamics import interface as IF
from fea_core.space import bolts_ecss as BE
from fea_core.space import notching as NT
from fea_core.space import pcb as PB
from fea_core.space.model import LimitEntry, SpaceSettings
from fea_core.space.requirements import confirm
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Fixed, Force, LoadCase, MeshSettings

from .cases import Case, Check
from .space_cases import block_study, make_requirement_set

SPACE_S4_CASES: dict[str, Case] = {}
G_MM = 9810.0


class _Out(dict):
    @property
    def warnings(self) -> list[str]:
        return []


def _build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return block_study(), []


def _confirmed_limits(limits: dict[str, float], units: dict[str, str] | None = None) -> S4Limits:
    rs = make_requirement_set("protoflight", confirmed=False)
    for k, v in limits.items():
        rs.limits.append(LimitEntry(key=k, value=v, unit=(units or {}).get(k, ""), document="test document", revision="test revision", clause="test clause"))
    sp = SpaceSettings(requirement_set=rs)
    confirm(rs, "test engineer", "2026-01-01", scope=sp)
    return S4Limits.from_requirement_set(sp)


# ------------------------------------------------------------------ S4A: one-DOF interface force and notch
def _sdof(m: float, k: float) -> IF.ModalInterface:
    phi = 1.0 / math.sqrt(m)
    fs = np.zeros((3, 3))
    fs[0, 0] = m
    grp = IF.InterfaceGroup(IF.ALL_GROUP, np.array([[-k * phi, 0.0, 0.0]]), np.zeros((1, 3)), fs, np.zeros((3, 3)), 1)
    gam = np.array([[math.sqrt(m), 0.0, 0.0]])
    return IF.ModalInterface(np.array([math.sqrt(k / m) / (2 * math.pi)]), gam, gam ** 2, m, np.zeros(3), {IF.ALL_GROUP: grp})


def _two_dof_chain(zeta: float = 0.03):
    """Base - k1 - m1 - k2 - m2 (test numbers): the modal interface data of the complete basis, the matrices and the exact (direct frequency-domain, modal damping) support force."""
    m1, m2, k1, k2 = 2.0e-3, 1.0e-3, 4.0e6, 2.5e6
    M = np.diag([m1, m2])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    w2, V = np.linalg.eig(np.linalg.solve(M, K))
    o = np.argsort(w2.real)
    w2, V = w2.real[o], V.real[:, o]
    for j in range(2):
        V[:, j] /= math.sqrt(V[:, j] @ M @ V[:, j])
    gam = np.zeros((2, 3))
    gam[:, 0] = [V[:, j] @ M @ np.ones(2) for j in range(2)]
    F = np.zeros((2, 3))
    F[:, 0] = -k1 * V[0, :]
    fs = np.zeros((3, 3))
    fs[0, 0] = m1 + m2
    grp = IF.InterfaceGroup(IF.ALL_GROUP, F, np.zeros((2, 3)), fs, np.zeros((3, 3)), 1)
    mi = IF.ModalInterface(np.sqrt(w2) / (2 * math.pi), gam, gam ** 2, m1 + m2, np.zeros(3), {IF.ALL_GROUP: grp})
    C = sum(2 * zeta * math.sqrt(w2[j]) * np.outer(M @ V[:, j], M @ V[:, j]) for j in range(2))

    def exact(f: float) -> complex:
        w = 2 * math.pi * f
        u = np.linalg.solve(K - w ** 2 * M + 1j * w * C, -M @ np.ones(2))
        return complex(np.diag(M) @ (1.0 - w ** 2 * u))               # Newton: m1 (a1 + a_b) + m2 (a2 + a_b)
    return mi, exact


def s4a_run(study, workdir: Path, parts) -> _Out:
    m, k, zeta = 1.0e-3, 4.0e3, 0.02                              # 1 kg, 4 N/mm -> 318.3 Hz
    mi = _sdof(m, k)
    f = IF.sine_grid(50.0, 2000.0, mi.freq, np.array([zeta]))
    sp = IF.sine_interface(mi, "x", 1.0, zeta, f=f)
    low = IF.sine_interface(mi, "x", 1.0, zeta, f=np.array([1e-3]))
    limit = 0.4 * m * G_MM * math.sqrt(1 + 4 * zeta ** 2) / (2 * zeta)
    res = NT.make_notch(sp.f, sp.a_g, NT.interface_response(sp), NT.NotchSpec("sine", "force", limit=limit, design_factor=1.0), S4Limits(), mi.freq)
    # the two-DOF chain: the interface force against the direct solution, two notches that keep the limit of the EXACT response
    z2 = 0.03
    mi2, exact = _two_dof_chain(z2)
    f2 = IF.sine_grid(2000.0, 30000.0, mi2.freq, np.array([z2, z2]))
    sp2 = IF.sine_interface(mi2, "x", 1.0, z2, f=f2)
    ex = np.array([abs(exact(x)) for x in f2]) * G_MM
    lim2 = 0.5 * min(float(ex[(f2 > 0.9 * fj) & (f2 < 1.1 * fj)].max()) for fj in mi2.freq)
    res2 = NT.make_notch(sp2.f, sp2.a_g, NT.interface_response(sp2), NT.NotchSpec("sine", "force", limit=lim2, design_factor=1.0), S4Limits(), mi2.freq)
    return _Out(m=m, zeta=zeta, sp=sp, low=low, res=res, limit=limit, f2=f2, sp2=sp2, ex=ex, res2=res2, lim2=lim2, exact=exact)


def s4a_eval(out: _Out, study) -> list[Check]:
    m, zeta = out["m"], out["zeta"]
    peak = m * G_MM * math.sqrt(1 + 4 * zeta ** 2) / (2 * zeta)
    return [Check("S4A interface reaction at f -> 0 = M_total * a_base", float(out["low"].F_peak[0]), m * G_MM, 1e-6),
            Check("S4A interface force at the resonance = m a sqrt(1 + 4 zeta^2) / (2 zeta)", float(out["sp"].F_peak.max()), peak, 1e-9),
            Check("S4A the notched peak equals the limit exactly", float(out["res"].quantity_notched.max()), out["limit"], 1e-12),
            Check("S4A the notch depth = 20 log10(limit / peak) [dB]", out["res"].notches[0].reduction_db, 20 * math.log10(0.4), 1e-9, "dB"),
            Check("S4A an unconfirmed limit never gives PASS", float(out["res"].verdict == NT.UNCONFIRMED), 1.0, 0.0),
            Check("S4A two-DOF: interface force = the direct frequency-domain solution (relative to its maximum)", float(np.max(np.abs(out["sp2"].F_peak - out["ex"])) / np.max(out["ex"])),
                  0.0, 1e-9),
            Check("S4A two-DOF: two notches, one per resonance", float(len(out["res2"].notches)), 2.0, 0.0),
            Check("S4A two-DOF: the notched profile applied to the EXACT response keeps the limit (peak = limit)",
                  float(np.max(out["ex"] * out["res2"].notched / out["res2"].level)), out["lim2"], 1e-9, "N"),
            Check("S4A two-DOF: the unnotched profile between the notches is untouched",
                  float(np.max(np.abs(out["res2"].notched / out["res2"].level - 1)[(out["f2"] > out["res2"].notches[0].f_hi * 1.01) & (out["f2"] < out["res2"].notches[1].f_lo * 0.99)])),
                  0.0, 1e-12)]


SPACE_S4_CASES["S4A"] = Case("S4A", "Space S4: interface force of a one-DOF structure, quasi-static limit M a, resonance m a/(2 zeta), notch holds the limit exactly",
                             _build, s4a_eval, run=s4a_run)


# ------------------------------------------------------------------ S4B: PCB, Steinberg by hand
def s4b_run(study, workdir: Path, parts) -> _Out:
    lim = _confirmed_limits({"pcb_steinberg_const": 0.00022, "pcb_amplitude_factor": 1.0, "pcb_design_factor": 1.25, "pcb_fatigue_exponent": 6.4, "pcb_ref_cycles": 2.0e7,
                             "pcb_damage_limit": 1.0, "pcb_n_sigma": 3.0, "pcb_component_const": 1.0,
                             "pcb_position_factor_min": 0.25}, {"pcb_steinberg_const": "in"})
    spec = PB.PcbSpec(name="U1", B_mm=5.0 * 25.4, h_mm=0.062 * 25.4, L_mm=25.4, r=1.0, z_rms_mm=0.05, f_n_Hz=100.0, duration_s=3600.0,
                     r_source="test input of the verification case", z_source="test input of the verification case")
    return _Out(res=PB.check_pcb(spec, lim))


def s4b_eval(out: _Out, study) -> list[Check]:
    r = out["res"]
    z_allow = 0.00022 * 5.0 / (1.0 * 0.062 * 1.0 * math.sqrt(1.0)) * 25.4                # mm, by hand
    z3 = 3.0 * 0.05
    p = [math.erf(1 / math.sqrt(2)), math.erf(2 / math.sqrt(2)) - math.erf(1 / math.sqrt(2)), math.erf(3 / math.sqrt(2)) - math.erf(2 / math.sqrt(2))]
    d3 = sum(pi * 100.0 * 3600.0 / (2.0e7 * (z_allow / (i * 0.05)) ** 6.4) for i, pi in zip((1, 2, 3), p))
    return [Check("S4B Z_allow = c B / (C h r sqrt(L)) [mm]", r.z_allow_mm or float("nan"), z_allow, 1e-12, "mm"),
            Check("S4B MoS = Z_allow / (J Z_3sigma) - 1", r.mos or float("nan"), z_allow / (1.25 * z3) - 1.0, 1e-12),
            Check("S4B three-band damage", (r.damage or {}).get("damage", float("nan")), d3, 1e-12),
            Check("S4B three-band damage against the narrow-band (Rayleigh) damage of fatigue.spectral", (r.damage or {}).get("damage", float("nan")) / (r.damage_nb or float("nan")),
                  1.0, 0.10),
            Check("S4B all inputs confirmed -> a verdict (PASS)", float(r.verdict == PB.PASS), 1.0, 0.0)]


SPACE_S4_CASES["S4B"] = Case("S4B", "Space S4: PCB Steinberg check, allowable displacement, margin and three-band damage by hand (test inputs)", _build, s4b_eval, run=s4b_run)


# ------------------------------------------------------------------ S4C: bolted joint by hand
KB, KC, AS, L = 3.5e5, 4.0e5, 20.1, 12.0
A_AL, A_ST, A_TI = 23.4e-6, 11.7e-6, 8.6e-6


def _bolt(alpha_b: float, **kw) -> BE.EcssBolt:
    al = {k: BE.Allowable(v, True, "test") for k, v in (("bolt_Fty", 640.0), ("bolt_Ftu", 800.0), ("bolt_Fsy", 370.0), ("bolt_Fsu", 480.0), ("part_Fsu", 200.0), ("part_Fbr", 420.0), ("part_Fy", 300.0))}
    d = dict(name="B", d_mm=6.0, pitch_mm=1.0, A_s_mm2=AS, k_b=KB, k_c=KC, F_nom=6000.0, embedding_um=6.0, layers=[BE.Layer(L, A_AL)], alpha_bolt=alpha_b,
             T_ref=20.0, T_min=-40.0, T_max=80.0, F_ax=3000.0, F_sh=800.0, d_w_mm=10.0, d_h_mm=6.6, engaged_mm=9.0, allow=al)
    d.update(kw)
    return BE.EcssBolt(**d)


def s4c_run(study, workdir: Path, parts) -> _Out:
    lim = _confirmed_limits({"bolt_tightening_scatter": 0.1, "bolt_relaxation_loss": 0.05, "bolt_slip_mu_min": 0.2, "bolt_factor_gapping": 1.2, "bolt_factor_slip": 1.25,
                             "bolt_factor_tensile_yield": 1.1, "bolt_factor_tensile": 1.4, "bolt_factor_shear": 1.4, "bolt_factor_combined": 1.4, "bolt_factor_pullout": 1.5,
                             "bolt_factor_bearing": 1.3})
    lim1 = _confirmed_limits({**{k: e.value for k, e in lim.entries.items()}, "bolt_thermal_factor": 1.0})            # the factor of the thermally induced load set to 1,0 (Table 4-4 note e)
    return _Out(steel=BE.assess_bolt(_bolt(A_ST), lim), ti=BE.assess_bolt(_bolt(A_TI), lim), steel_jt1=BE.assess_bolt(_bolt(A_ST), lim1),
                clamp=BE.assess_bolt(_bolt(A_ST, F_ax=0.0, F_sh=0.0, F_K_req=3000.0), lim), lost=BE.assess_bolt(_bolt(A_ST, F_nom=1500.0, F_ax=0.0, F_sh=0.0), lim))


def s4c_eval(out: _Out, study) -> list[Check]:
    kser = KB * KC / (KB + KC)
    phi = KB / (KB + KC)
    checks: list[Check] = []
    for tag, r, ab in (("steel bolt", out["steel"], A_ST), ("titanium bolt", out["ti"], A_TI)):
        hot = (A_AL - ab) * 60.0 * L * kser
        cold = (A_AL - ab) * (-60.0) * L * kser
        pmin0 = 6000.0 * 0.9 - 0.006 * kser - 0.05 * 6000.0                          # eq. 6.3.18 with F_Z = embedding + the other losses
        pmax0 = 6000.0 * 1.1
        ext = phi * 3000.0

        def fpmax(j: float) -> float:                         # F_V,max0 + J_T dF_hot, J_T = the factor J of the check (ECSS-E-ST-32-10C 4.3.2.2 b)
            return pmax0 + j * hot
        f_sep = (1 - phi) * 3000.0
        m = {x.check: x for x in r.margins}
        checks += [Check(f"S4C {tag}: dF_th at T_max = (alpha_c - alpha_b) dT L / (1/k_b + 1/k_c) [N]", r.dF_th["T_max"], hot, 1e-12, "N"),
                   Check(f"S4C {tag}: dF_th at T_min [N]", r.dF_th["T_min"], cold, 1e-12, "N"),
                   Check(f"S4C {tag}: gapping MoS = F_p,min / ((1 - Phi) J F_ax) - 1", m["gapping"].mos or float("nan"), (pmin0 + 1.2 * cold) / (1.2 * f_sep) - 1.0, 1e-12),
                   Check(f"S4C {tag}: slip MoS = mu F_c / (J F_sh) - 1", m["slip"].mos or float("nan"), 0.2 * max(pmin0 + 1.25 * cold - f_sep, 0.0) / (1.25 * 800.0) - 1.0, 1e-12),
                   Check(f"S4C {tag}: tensile ultimate MoS = F_tu A_s / (F_V,max + J Phi F_ax) - 1 (J on the external load only)", m["tensile_ultimate"].mos or float("nan"),
                         800.0 * AS / (fpmax(1.4) + 1.4 * ext) - 1.0, 1e-12),
                   Check(f"S4C {tag}: crushing under the head, yield MoS = F_y A_uh / (F_V,max + J_y Phi F_ax) - 1", m["bearing_head_yield"].mos or float("nan"),
                         300.0 * 3.141592653589793 / 4 * (10.0 ** 2 - 6.6 ** 2) / (fpmax(1.1) + 1.1 * ext) - 1.0, 1e-12),
                   Check(f"S4C {tag}: combined MoS = 1 / sqrt(R_A^2 + R_S^2) - 1", m["combined"].mos or float("nan"),
                         1.0 / math.sqrt(((fpmax(1.4) + 1.4 * ext) / (800.0 * AS)) ** 2 + (1.4 * 800.0 / (480.0 * AS)) ** 2) - 1.0, 1e-12)]
    r1 = {x.check: x for x in out["steel_jt1"].margins}
    hot_s, cold_s = (A_AL - A_ST) * 60.0 * L * kser, -(A_AL - A_ST) * 60.0 * L * kser
    pmin0_s = 6000.0 * 0.9 - 0.006 * kser - 0.05 * 6000.0
    checks.append(Check("S4C bolt_thermal_factor = 1: gapping MoS with the unfactored cold loss", r1["gapping"].mos or float("nan"),
                        (pmin0_s + cold_s) / (1.2 * (1 - phi) * 3000.0) - 1.0, 1e-12))
    checks.append(Check("S4C bolt_thermal_factor = 1: tensile ultimate MoS with the unfactored hot increment", r1["tensile_ultimate"].mos or float("nan"),
                        800.0 * AS / (6600.0 + hot_s + 1.4 * phi * 3000.0) - 1.0, 1e-12))
    g = {x.check: x for x in out["clamp"].margins}["gapping"]
    checks.append(Check("S4C no axial force, F_K,req = 3000 N: MoS = F_V,min / F_K,req - 1", g.mos or float("nan"), (pmin0_s + 1.2 * cold_s) / 3000.0 - 1.0, 1e-12))
    checks.append(Check("S4C no axial force and F_V,min below F_K,req: the joint fails", float(g.status == "FAIL"), 1.0, 0.0))
    lm = {x.check: x for x in out["lost"].margins}["gapping"]
    checks.append(Check("S4C no axial force, F_K,req = 0, clamp force lost by the cold case: F_V,min = F_V,min0 + J_T dF_T,loss [N]", lm.capacity or float("nan"),
                        1500.0 * 0.9 - 0.006 * kser - 0.05 * 1500.0 + 1.2 * cold_s, 1e-12))
    checks.append(Check("S4C no axial force, F_K,req = 0, F_V,min <= 0: the joint fails (MoS = -1)", float(lm.status == "FAIL" and lm.mos == -1.0), 1.0, 0.0))
    checks.append(Check("S4C the titanium bolt has the larger thermal preload change", float(out["ti"].dF_th["T_max"] > out["steel"].dF_th["T_max"]), 1.0, 0.0))
    checks.append(Check("S4C the effective engaged length of the pull-out is L_eng - 2 p", float("L_eng,eff = 7.00" in {x.check: x for x in out["steel"].margins}["pullout"].note), 1.0, 0.0))
    return checks


SPACE_S4_CASES["S4C"] = Case("S4C", "Space S4: bolted joint, gapping, slip and the differential-CTE preload change (aluminium plate, steel / titanium bolt) by hand (test coefficients)",
                             _build, s4c_eval, run=s4c_run)


# ------------------------------------------------------------------ S4D: a real bar
def s4d_build(wd: Path):
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces

    from .cases import _study, box
    wd.mkdir(parents=True, exist_ok=True)
    st, parts = _study("S4D interface forces of a bar", wd / "model.step", [Part("Bar", box(0, 0, 0, 80, 10, 10))], MeshSettings(size=4.0))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.add_selection("end", find_faces(parts, x=80))
    st.constraints.append(Fixed("clamp"))
    st.load_cases.append(LoadCase("Load", [Force("end", [100.0, 0.0, 0.0])]))
    st.dynamics.modes = 12
    st.solver.modal_engine = "ccx"
    return st, parts


def s4d_eval(res, study) -> list[Check]:
    wd = Path(study.geometry).parent / "s4d_interface"
    mi = IF.run_interface(study, res.mesh, wd, modes=12)
    mass = 80.0 * 10.0 * 10.0 * 7.85e-9
    checks = [Check("S4D total mass of the model [t]", mi.total_mass, mass, 1e-3, "t")]
    for ax in "xyz":
        sp = IF.sine_interface(mi, ax, 1.0, 0.02, f=np.array([1e-3]))
        checks.append(Check(f"S4D interface reaction at f -> 0, base axis {ax} = M_total * a_base (gate 0.5 %)", float(sp.F_peak[0]), mi.total_mass * G_MM, 5e-3, "N"))
    sy = IF.sine_interface(mi, "y", 1.0, 0.02, f=np.array([1e-3]))
    checks.append(Check("S4D moment about the reference point = M a * lever arm (L/2 = 40 mm)", float(sy.M_peak[0]), mi.total_mass * G_MM * 40.0, 5e-3, "N*mm"))
    return checks


SPACE_S4_CASES["S4D"] = Case("S4D", "Space S4: interface reaction of a real bar (CalculiX modal basis), quasi-static limit M a and the moment M a * lever arm", s4d_build, s4d_eval)

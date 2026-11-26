"""Verification of the FEM model quality checks of the spacecraft structural verification (F14 stage S3, `fea_core.space.model_checks`) against hand calculations.

SC1 mass and centre of mass of two bars + a point mass + a surface mass by hand; the mass the solver sees (three unit-gravity statics) equals it.
SC2 free-free steel beam: six rigid-body modes of zero strain energy, the first elastic mode against the Euler–Bernoulli free-free beam (4.730041).
SC3 unit enforced displacement of a bar (nu = 0) held at the other end: the net interface reaction is E·A·delta / L (axial stiffness); a bar that is not held has zero energy.
SC4 isothermal expansion of a bar (nu = 0) fixed at both ends: the support force is E·alpha·dT·A (the model of the check has the single expansion coefficient alpha); a free model has no stress.
SC5 effective mass of an axial rod (nu = 0) fixed at one end: the first two axial modes carry 8/pi²·(1 + 1/9) of the mass (the sum rule of a fixed-free rod).
SC6 mesh convergence of the first bending frequency of a cantilever on three meshes against the Euler–Bernoulli frequency (Timoshenko correction ≈ −1 % at L/h = 10).
SC7 equilibrium of a clamped bar under a tip force: the reaction is the load, the recomputed imbalance is rounding.
The limits that the checks are run with are example values of THESE cases – the program ships none.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fea_core.dynamics.model import PointMass, SurfaceMass
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh, run_study
from fea_core.space import model_checks as mc
from fea_core.study import Fixed, Force, LoadCase, Material, MeshSettings
from verification.cases import AtMost, Case, Check, E, RHO, _study, box

L, B, H = 200.0, 20.0, 10.0
ALPHA = 1.0e-5

# example limits of the cases (not values of a standard)
_LIM = {"mass_tol_rel": 1e-6, "cog_tol_mm": 1e-3, "solver_mass_tol_rel": 1e-6, "rigid_energy_ratio_max": 1e-6, "rigid_fit_tol": 1e-3, "unit_disp_energy_ratio_max": 1e-8,
       "unit_disp_reaction_ratio_max": 1e-8, "isothermal_stress_ratio_max": 1e-6, "thermal_support_reaction_ratio_max": 1e-3, "thermal_support_stress_ratio_max": 5.0, "eff_mass_min_fraction": 0.5,
       "eff_mass_cutoff_hz": 25000.0, "freq_convergence_tol": 0.02, "stress_convergence_tol": 0.5, "equilibrium_tol": 1e-6}
LIM = mc.ModelCheckLimits.from_entries(_LIM, trusted=True)       # confirmed for these cases only: a caller's dict is never confirmed


def _run_checks(study, wd: Path, parts, ids: list[str], options: dict, results=None, mesh=None):
    if mesh is None:
        st = study
        st.modal.enabled = st.modal.enabled or not st.load_cases
        mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    out = {r.id: r for r in mc.run_model_checks(study, mesh, results, LIM, {"threads": 4, **options}, workdir=wd / "checks", checks=ids, parts=parts, study_dir=wd)}
    return SimpleNamespace(warnings=[], mesh=mesh, checks=out)


def _got(res, cid: str):
    """The result of a check; a check that could not run (the direct solver is not installed, ...) stops the case with its reason instead of a KeyError."""
    r = res.checks[cid]
    if r.status == "SKIPPED":
        raise RuntimeError(f"model check '{cid}' was skipped: {r.summary}")
    return r


def _bar(wd: Path, name: str, nu: float = 0.3, size: float = 5.0):
    st, parts = _study(name, wd / "model.step", [Part("Nosnik", box(0, 0, 0, L, B, H))], MeshSettings(size=size))
    if nu != 0.3:
        st.materials["Ocel S235"] = Material("Ocel S235", E, nu, RHO, Re=235.0, Rm=360.0, alpha=ALPHA)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L))
    return st, parts


# ---------------------------------------------------------------------------------------------------------------- SC1
def _sc1_masses():
    m_body = L * B * H * RHO
    m_pt = 1.0e-4
    m_sf = 2.0 * 1e-9 * (L / 2 * B)
    pts = [(m_body, np.array([L / 2, B / 2, H / 2])), (m_pt, np.array([0.75 * L, B / 2, H])), (m_sf, np.array([0.25 * L, B / 2, H]))]
    m = sum(p[0] for p in pts)
    return m, sum(p[0] * p[1] for p in pts) / m


def sc1_build(wd: Path):
    geo = [Part("A", box(0, 0, 0, L / 2, B, H)), Part("B", box(L / 2, 0, 0, L / 2, B, H))]
    st, parts = _study("SC1 hmota a teziste", wd / "model.step", geo, MeshSettings(size=5.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("top_B", find_faces(parts, part="B", normal=[0, 0, 1]))
    st.add_selection("top_A", find_faces(parts, part="A", normal=[0, 0, 1]))
    st.constraints.append(Fixed("vetknuti"))
    st.masses.append(PointMass("top_B", 1.0e-4, coupling="distributing", name="box"))
    st.masses.append(SurfaceMass("top_A", 2.0, name="pcb"))
    st.modal.enabled = True
    st.modal.modes = 4
    return st, parts


def sc1_run(st, wd, parts):
    m, c = _sc1_masses()
    return _run_checks(st, wd, parts, ["mass"], {"mass_budget_t": m, "cog_budget_mm": list(c)})


def sc1_eval(res, st):
    r = _got(res, "mass")
    m, c = _sc1_masses()
    s = r.values["solver"]
    return [Check("FE mass (integration) / hand calculation", r.values["fe_mass_t"], m, 1e-9, "t"),
            Check("FE centre of mass x", r.values["cog_mm"][0], c[0], 1e-9, "mm"), Check("FE centre of mass y", r.values["cog_mm"][1], c[1], 1e-9, "mm"),
            Check("FE centre of mass z", r.values["cog_mm"][2], c[2], 1e-9, "mm"),
            Check("mass of the solver (unit-gravity reaction) / hand calculation", s["mass_t"], m, 1e-6, "t"),
            Check("centre of mass of the solver x", s["cog_mm"][0], c[0], 1e-6, "mm"), Check("centre of mass of the solver z", s["cog_mm"][2], c[2], 1e-6, "mm")]


# ---------------------------------------------------------------------------------------------------------------- SC2
def sc2_build(wd: Path):
    st, parts = _bar(wd, "SC2 volny nosnik, kontrola 2", size=5.0)
    st.constraints.append(Fixed("vetknuti"))                    # the check removes the support itself
    st.modal.enabled = True
    st.modal.modes = 4
    return st, parts


def sc2_run(st, wd, parts):
    return _run_checks(st, wd, parts, ["free_free"], {"free_free_modes": 12})


def sc2_eval(res, st):
    r = _got(res, "free_free")
    f_weak = 4.730041 ** 2 / (2 * math.pi * L ** 2) * math.sqrt(E * (B * H ** 3 / 12) / (RHO * B * H))
    return [Check("number of rigid-body modes of the free body", float(r.values["rigid_body_modes"]), 6.0, 0.0, ""),
            Check("first elastic mode (bending about the weak axis) / E–B free-free beam", r.values["first_elastic_Hz"], f_weak, 0.02, "Hz"),
            AtMost("strain energy of the rigid-body modes / first elastic mode", max(r.values["rigid_energy_ratio_max"], 1e-300), 1e-6, 0.0, ""),
            AtMost("share of a rigid-body mode that is not a rigid-body motion", r.values["rigid_fit_residual_max"], 1e-3, 0.0, "")]


# ---------------------------------------------------------------------------------------------------------------- SC3
def sc3_build(wd: Path):
    st, parts = _bar(wd, "SC3 jednotkovy vynuceny posuv", nu=0.0)
    st.constraints.append(Fixed("vetknuti"))
    st.constraints.append(Fixed("konec"))
    st.modal.enabled = True
    st.modal.modes = 4
    return st, parts


def sc3_run(st, wd, parts):
    return _run_checks(st, wd, parts, ["unit_displacement"], {"interface_selections": ["vetknuti"], "unit_states": ["UX"]})


def sc3_eval(res, st):
    r = _got(res, "unit_displacement")
    ref = r.values["reference"]
    f = r.values["states"]["UX"]["net_force_ratio"] * ref["E_MPa"] * ref["Lc_mm"] * 1.0
    return [Check("net interface reaction of the unit displacement / (E·A·delta/L)", f, E * B * H / L, 2e-3, "N"),
            Check("the check fails (FAIL: the other support holds the body)", 1.0 if r.status == "FAIL" else 0.0, 1.0, 0.0, "")]


# ---------------------------------------------------------------------------------------------------------------- SC4
def sc4_build(wd: Path):
    st, parts = _bar(wd, "SC4 izotermicka roztaznost", nu=0.0)
    st.constraints.append(Fixed("vetknuti"))
    st.constraints.append(Fixed("konec"))
    st.modal.enabled = True
    st.modal.modes = 4
    return st, parts


def sc4_run(st, wd, parts):
    return _run_checks(st, wd, parts, ["isothermal"], {"thermal_alpha_per_K": ALPHA, "thermal_dT_K": 1.0})


def sc4_eval(res, st):
    r = _got(res, "isothermal")
    sup = r.values["support"]
    force = max(abs(v) for v in sup["per_selection"]["vetknuti"]["net_force_N"])
    return [Check("support force of the doubly held bar / (E·alpha·dT·A)", force, E * ALPHA * 1.0 * B * H, 2e-3, "N"),
            AtMost("stress of the free expansion / (E·alpha·dT)", max(r.values["free_expansion"]["stress_ratio"], 1e-300), 1e-9, 0.0, "")]


# ---------------------------------------------------------------------------------------------------------------- SC5
def sc5_build(wd: Path):
    st, parts = _bar(wd, "SC5 efektivni hmota tyce", nu=0.0, size=4.0)
    st.constraints.append(Fixed("vetknuti"))
    st.modal.enabled = True
    st.modal.modes = 6
    return st, parts


def sc5_run(st, wd, parts):
    return _run_checks(st, wd, parts, ["effective_mass"], {})


def sc5_eval(res, st):
    r = _got(res, "effective_mass")
    ref = 8.0 / math.pi ** 2 * (1.0 + 1.0 / 9.0)
    return [Check("cumulative effective mass in x (2 axial modes) / sum rule of the rod", r.values["cumulative_fraction_xyz"][0], ref, 0.02, ""),
            Check("first axial mode: m_eff / m = 8/π²", max(row[0] for row in r.values["eff_mass_fraction_per_mode"]), 8.0 / math.pi ** 2, 0.02, "")]


# ---------------------------------------------------------------------------------------------------------------- SC6
def sc6_build(wd: Path):
    st, parts = _bar(wd, "SC6 konvergence site", size=8.0)
    st.constraints.append(Fixed("vetknuti"))
    st.modal.enabled = True
    st.modal.modes = 3
    return st, parts


def sc6_run(st, wd, parts):
    return _run_checks(st, wd, parts, ["mesh_convergence"], {"convergence_levels": 3, "convergence_ratio": 1.5, "convergence_frequencies": 5})


def sc6_eval(res, st):
    r = _got(res, "mesh_convergence")
    f1 = 1.875104 ** 2 / (2 * math.pi * L ** 2) * math.sqrt(E * (B * H ** 3 / 12) / (RHO * B * H))
    fc = r.values["frequency_convergence"][0]
    return [Check("f1 on the finest mesh / E–B cantilever (Timoshenko about -1 %)", fc["frequencies_Hz"][-1], f1, 0.02, "Hz"),
            AtMost("change of f1 with the last refinement", max(fc["last_change"], 1e-300), 0.01, 0.0, "")]


# ---------------------------------------------------------------------------------------------------------------- SC7
def sc7_build(wd: Path):
    st, parts = _bar(wd, "SC7 rovnovaha", size=5.0)
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Sila 100 N", [Force("konec", [0, 0, -100.0])]))
    return st, parts


def sc7_run(st, wd, parts):
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    res = run_study(st, wd / "solved", mesh=mesh)
    out = _run_checks(st, wd, parts, ["equilibrium"], {}, results=res, mesh=mesh)
    out.steps = res.steps
    return out


def sc7_eval(res, st):
    r = _got(res, "equilibrium")
    row = r.values["states"][0]
    return [Check("reaction in z / load", row["reaction_force_N"][2], 100.0, 1e-5, "N"),
            AtMost("recomputed force and moment imbalance", max(row["imbalance"], 1e-300), 1e-4, 0.0, "")]


SPACE_CHECK_CASES = {
    "SC1": Case("SC1", "Model check 1: mass and centre of mass (two bars + a point mass + a surface mass)", sc1_build, sc1_eval, run=sc1_run),
    "SC2": Case("SC2", "Model check 2: free-free modes – 6 rigid-body modes, the first elastic mode (E–B free-free beam)", sc2_build, sc2_eval, run=sc2_run),
    "SC3": Case("SC3", "Model check 3: unit enforced displacement – the reaction E·A·delta/L of a support off the interface", sc3_build, sc3_eval, run=sc3_run),
    "SC4": Case("SC4", "Model check 4: isothermal expansion – the force E·alpha·dT·A of a bar held at both ends", sc4_build, sc4_eval, run=sc4_run),
    "SC5": Case("SC5", "Model check 5: effective mass of an axial rod (8/pi²·(1 + 1/9))", sc5_build, sc5_eval, run=sc5_run),
    "SC6": Case("SC6", "Model check 6: mesh convergence of the first frequency of a cantilever (E–B)", sc6_build, sc6_eval, run=sc6_run),
    "SC7": Case("SC7", "Model check 7: equilibrium – reaction = load, imbalance at rounding level", sc7_build, sc7_eval, run=sc7_run),
}

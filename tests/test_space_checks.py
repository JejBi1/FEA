"""F14 stage S3 – the FEM model quality checks (`fea_core.space.model_checks`).

Gate of the stage: the clean reference model passes all seven checks, and every one of the injected faults is detected and reported as FAIL with a diagnostic that names the cause:
  1 a part that is not connected              -> free-free modes: 12 rigid-body modes, the groups of parts that move independently
  2 a stray grounded support / spring         -> unit enforced displacement: the net interface reaction, the retained constraint that takes the force
  3 a wrong (1000x) density                   -> mass against the budget
  4 an over-constrained thermal support, a rigid coupling over an expanding face -> isothermal expansion
  5 a truncated modal basis                   -> effective modal mass
  6 an inconsistent reaction                  -> equilibrium recomputed from the stored reactions
The limits used in these tests are EXAMPLE values chosen here (the program ships none): with the empty template no check may ever say PASS.
Needs the direct path (numba + rslab) for the prescribed-displacement checks; the other checks run on CalculiX or the direct modes.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.dynamics.model import PointMass, SurfaceMass
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh, run_study
from fea_core.space import _checks_common as cc
from fea_core.space import _checks_equil as eq
from fea_core.space import _checks_mass as cmass
from fea_core.space import _checks_modal as cmodal
from fea_core.space import model_checks as mc
from fea_core.study import ElasticSupport, Fixed, Force, LoadCase, Material, MeshSettings
from verification.cases import RHO, _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
slow = pytest.mark.slow

# the example limits of the tests – NOT values of a standard
LIMITS_DICT = {"mass_tol_rel": 0.01, "cog_tol_mm": 0.5, "cad_mass_tol_rel": 0.01, "part_mass_tol_rel": 0.01, "solver_mass_tol_rel": 1e-6,
          "rigid_energy_ratio_max": 1e-6, "rigid_fit_tol": 1e-3, "unit_disp_energy_ratio_max": 1e-8, "unit_disp_reaction_ratio_max": 1e-8,
          "isothermal_stress_ratio_max": 1e-6, "thermal_support_reaction_ratio_max": 1e-3, "thermal_support_stress_ratio_max": 5.0, "eff_mass_min_fraction": 0.5, "eff_mass_cutoff_hz": 12000.0,
          "freq_convergence_tol": 0.05, "stress_convergence_tol": 0.25, "equilibrium_tol": 1e-6}
LIMITS = cc.ModelCheckLimits.from_entries(LIMITS_DICT, trusted=True)         # confirmed for the tests only: a caller's dict is never confirmed

L_BAR, W_BAR, H_BAR = 120.0, 20.0, 10.0
M_POINT = 5.0e-5                                 # t = 50 g, distributing coupling (does not stiffen)
SURF = 1.5                                       # kg/m²


@pytest.fixture(scope="module", autouse=True)
def _rslab_on_the_path_of_the_worker():
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it too."""
    mp = pytest.MonkeyPatch()
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        mp.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    yield
    mp.undo()


def build_sat(wd: Path, name: str = "sat", gap: float = 0.0, nu: float = 0.3):
    """Two bars A | B in a row (bonded by the common mesh, or `gap` apart), the launch face at x = 0, the other faces as selections for the faults; a distributing point
    mass on the top of B and a surface mass on the top of A. → (study, parts, mesh)."""
    xa, xb = L_BAR / 2 - gap / 2, L_BAR / 2 + gap / 2
    geo = [Part("A", box(0, 0, 0, xa, W_BAR, H_BAR)), Part("B", box(xb, 0, 0, L_BAR - xb, W_BAR, H_BAR))]
    st, parts = _study(name, wd / f"{name}.step", geo, MeshSettings(size=5.0))
    if nu != 0.3:
        st.materials["Ocel S235"] = Material("Ocel S235", 210000.0, nu, RHO, Re=235.0, Rm=360.0, alpha=1.2e-5)
    st.add_selection("launch", find_faces(parts, x=0))
    st.add_selection("end", find_faces(parts, x=L_BAR))
    st.add_selection("top_B", find_faces(parts, part="B", normal=[0, 0, 1]))
    st.add_selection("top_A", find_faces(parts, part="A", normal=[0, 0, 1]))
    st.constraints.append(Fixed("launch"))
    if gap:                                       # B is not connected to A but has a support of its own (so the study meshes: the program refuses a free body at the meshing)
        st.constraints.append(Fixed("end"))
    st.masses.append(PointMass("top_B", M_POINT, coupling="distributing", name="box"))
    st.masses.append(SurfaceMass("top_A", SURF, name="pcb"))
    st.load_cases.append(LoadCase("Tip load", [Force("end", [0, 0, -100.0])]))
    st.modal.enabled, st.modal.modes = True, 8
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return st, parts, mesh


def analytic_mass() -> tuple[float, np.ndarray]:
    """Mass [t] and centre of mass [mm] of build_sat by hand: two bars of the same steel + the point mass at the centroid of the top of B + the surface mass on the top of A."""
    m_body = L_BAR * W_BAR * H_BAR * RHO
    m_surf = SURF * 1e-9 * (L_BAR / 2 * W_BAR)
    pts = [(m_body, np.array([L_BAR / 2, W_BAR / 2, H_BAR / 2])), (M_POINT, np.array([0.75 * L_BAR, W_BAR / 2, H_BAR])), (m_surf, np.array([0.25 * L_BAR, W_BAR / 2, H_BAR]))]
    m = sum(p[0] for p in pts)
    return m, sum(p[0] * p[1] for p in pts) / m


@pytest.fixture(scope="module")
def sat(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sat")
    st, parts, mesh = build_sat(wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd)


def run(sat, checks, limits=LIMITS, options=None, study=None, mesh=None, results=None, sub="w", **kw):
    opt = options if options is not None else {}
    if isinstance(opt, dict):
        opt = {"threads": 4, **opt}
    res = mc.run_model_checks(study or sat.study, mesh or sat.mesh, results, limits, opt, workdir=sat.wd / sub, checks=checks, parts=sat.parts, **kw)
    return {r.id: r for r in res}


# ---------------------------------------------------------------------------------------------------------------- limits and verdicts (no solver)
def test_the_template_has_no_value_and_nothing_is_confirmed():
    lim = cc.ModelCheckLimits.template()
    assert set(lim.entries) == set(cc.LIMIT_DESCRIPTIONS)
    assert all(e.value is None and not e.confirmed for e in lim.entries.values())
    assert all(lim.value(n) is None for n in lim.entries)
    assert all(e["value"] is None and e["confirmed"] is False for e in mc.template_limits().values())


def test_a_limit_counts_only_when_confirmed_and_a_typo_is_an_error():
    lim = cc.ModelCheckLimits.from_entries({"mass_tol_rel": {"value": 0.02, "confirmed": False, "document": "d", "clause": "c"},
                                            "cog_tol_mm": {"value": 1.5, "confirmed": True, "document": "doc", "revision": "A", "clause": "5.1",
                                                           "confirmed_by": "x", "date": "2026-01-01"}, "equilibrium_tol": 1e-4}, trusted=True)
    assert lim.value("mass_tol_rel") is None                         # entered but not confirmed
    assert lim.value("cog_tol_mm") == 1.5 and lim.provenance("cog_tol_mm")["clause"] == "5.1"
    assert lim.value("equilibrium_tol") == 1e-4 and lim.provenance("equilibrium_tol")["document"] == "entered directly"
    with pytest.raises(ValueError, match="Unknown model-check limit"):
        cc.ModelCheckLimits.from_entries({"mass_tolerance": 0.1})


def test_the_judge_never_passes_without_its_limit_and_a_violation_wins():
    j = cc.Judge(cc.ModelCheckLimits.template())
    assert j.compare_max(1e-12, "mass_tol_rel", "x") is None and j.status() == cc.UNCONFIRMED
    j = cc.Judge(cc.ModelCheckLimits.from_entries({"mass_tol_rel": 0.01}, trusted=True))
    assert j.compare_max(0.001, "mass_tol_rel", "x") is True and j.status() == cc.PASS
    assert j.compare_max(0.5, "mass_tol_rel", "x") is False and j.status() == cc.FAIL
    j = cc.Judge(cc.ModelCheckLimits.from_entries({"mass_tol_rel": 0.01}, trusted=True))
    j.warn("odd")
    assert j.compare_max(float("nan"), "mass_tol_rel", "x") is False          # a NaN is never within a limit
    assert j.status() == cc.FAIL
    j = cc.Judge(cc.ModelCheckLimits.template())
    j.violate("invariant")
    j.need("equilibrium_tol")
    assert j.status() == cc.FAIL                                           # an invariant fails even without a limit


def test_the_overall_verdict_order():
    def seven(*first: str, rest: str = cc.PASS):
        """The seven checks of the standard with the given statuses of the first ones."""
        st = list(first) + [rest] * (len(cc.CHECK_IDS) - len(first))
        return [cc.CheckResult(cid, i + 1, "t", s, "") for i, (cid, s) in enumerate(zip(cc.CHECK_IDS, st))]
    assert cc.overall_verdict(seven()) == "PASS"
    assert cc.overall_verdict(seven(cc.WARN)) == "WARN"
    assert cc.overall_verdict(seven(cc.UNCONFIRMED)) == "UNCONFIRMED_REQUIREMENTS"
    assert cc.overall_verdict(seven(cc.UNCONFIRMED, cc.SKIPPED)) == "INCOMPLETE"
    assert cc.overall_verdict(seven(cc.SKIPPED, cc.FAIL)) == "FAIL"
    assert cc.overall_verdict([]) == "INCOMPLETE"
    assert cc.overall_verdict(seven()[:-1]) == "INCOMPLETE"                                 # a subset of the seven checks is never a PASS of the model
    assert cc.overall_verdict(seven()[:1]) == "INCOMPLETE"


def test_no_check_module_contains_a_standard_number():
    """Static guard: the names of the ECSS factors and any limit defaults must not appear in the code of the checks; the options dataclass has no field named like a limit."""
    import re
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path(cc.__file__).parent.glob("*checks*.py")) + [Path(mc.__file__)])
    for sym in ("FOSY", "FOSU", "KQ", "KP", "KLD"):
        assert not re.search(rf"\b{sym}\b", src), sym
    assert not (set(cc.LIMIT_DESCRIPTIONS) & {f for f in cc.ModelCheckOptions.__dataclass_fields__})
    assert all(e.value is None for e in cc.ModelCheckLimits().entries.values())


def test_options_from_dict_rejects_unknown_keys():
    o = mc.options_from_dict({"mass_budget_t": 1.0, "cog_budget_mm": [1, 2, 3], "unit_states": ["UX"]})
    assert o.cog_budget_mm == (1, 2, 3) and o.unit_states == ("UX",)
    with pytest.raises(ValueError, match="Unknown model-check option"):
        mc.options_from_dict({"mass_budjet": 1})
    with pytest.raises(ValueError, match="Unknown check"):
        mc.run_model_checks(object(), object(), None, None, None, workdir=".", checks=["nonsense"])


# ---------------------------------------------------------------------------------------------------------------- check 1: mass (the mesh only)
def test_the_integrated_mass_and_centre_of_mass_match_the_hand_calculation(sat):
    p = cmass.mass_properties(sat.study, sat.mesh)
    m, c = analytic_mass()
    assert p.mass_t == pytest.approx(m, rel=1e-12)
    assert np.allclose(p.cog, c, atol=1e-9)
    assert p.parts["A"]["mass_t"] == pytest.approx(L_BAR / 2 * W_BAR * H_BAR * RHO, rel=1e-12)
    assert p.added_t == pytest.approx(M_POINT + SURF * 1e-9 * L_BAR / 2 * W_BAR, rel=1e-12)


def test_the_mass_check_judges_against_the_budget_and_the_cad_volumes(sat):
    m, c = analytic_mass()
    r = run(sat, ["mass"], options={"mass_budget_t": m, "cog_budget_mm": list(c), "solver_mass_check": False,
                                    "part_mass_budget_t": {"A": L_BAR / 2 * W_BAR * H_BAR * RHO}})["mass"]
    assert r.status == cc.UNCONFIRMED and "solver_mass_check" in r.summary, r.findings       # every limit decided and held - but the solver cross-check was switched off: never a full PASS
    assert all(p["decided"] for p in r.limits.values()) and r.limits
    assert r.values["mass_rel_error"] == pytest.approx(0.0, abs=1e-12) and r.values["cad"]["A"]["rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert r.values["fe_mass_kg"] == pytest.approx(m * 1000.0)
    r = run(sat, ["mass"], options={"mass_budget_t": m * 1.2, "solver_mass_check": False})["mass"]               # a budget that is 20 % off
    assert r.status == cc.FAIL and "budget" in r.summary
    r = run(sat, ["mass"], limits={}, options={"mass_budget_t": m, "solver_mass_check": False})["mass"]         # no limit: the number is reported, no verdict
    assert r.status == cc.UNCONFIRMED and r.values["mass_rel_error"] == pytest.approx(0.0, abs=1e-12)


def test_fault_wrong_density_is_detected_by_the_mass_budget(sat):
    """Fault 3: the density of the steel typed in kg/m3 (7850) instead of t/mm3 – a factor 1e12; typed 1000x off here: the mass is far from the budget and the density is implausible."""
    m, _ = analytic_mass()
    bad = copy.deepcopy(sat.study)
    bad.materials["Ocel S235"].rho = RHO * 1000.0
    r = run(sat, ["mass"], study=bad, options={"mass_budget_t": m, "solver_mass_check": False})["mass"]
    assert r.status == cc.FAIL
    assert r.values["mass_rel_error"] > 100
    assert any("density" in f and "plausible" in f for f in r.findings)
    assert any("budget" in f for f in r.findings)


@needs_direct
@needs_ccx
@slow
def test_the_solver_sees_the_same_mass_and_centre_of_mass(sat):
    r = run(sat, ["mass"], options={"mass_budget_t": analytic_mass()[0]})["mass"]
    assert r.status == cc.PASS, r.findings
    s = r.values["solver"]
    assert abs(s["mass_rel_diff_to_integration"]) < 1e-9 and s["cog_diff_to_integration_mm"] < 1e-6
    assert max(s["lateral_reaction_share"]) < 1e-9


# ---------------------------------------------------------------------------------------------------------------- check 2: free-free
def test_rigid_groups_names_the_parts_that_move_independently():
    """Synthetic shapes: part A and B rigid and the same -> one group; B moving by itself -> two groups; a deforming part is named."""
    from fea_core.mesh import Mesh
    nodes = np.array([[x, y, z] for x in (0.0, 1.0, 2.0, 3.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    tets = np.zeros((2, 10), np.int64)
    tets[0, :4] = [0, 1, 2, 4]
    tets[0, 4:] = [0, 1, 2, 4, 5, 6]
    tets[1, :4] = [16 - 8 + 0, 9, 10, 12]
    tets[1, 4:] = [8, 9, 10, 12, 13, 14]
    mesh = Mesh(nodes, tets, np.array([0, 1]), ["A", "B"])
    n = len(nodes)
    t = np.zeros((6, n, 3))
    for k in range(3):                                          # three translations of the whole body
        t[k, :, k] = 1.0
    b_nodes = np.unique(tets[1])
    t[3, b_nodes, 0] = 1.0                                      # B alone moves in x: an extra rigid mode of a free part
    g = cmodal.rigid_groups(mesh, t[:4])
    assert sorted(map(sorted, g["groups"])) == [["A"], ["B"]]
    g1 = cmodal.rigid_groups(mesh, t[:3])
    assert g1["groups"] == [["A", "B"]] or g1["groups"] == [["B", "A"]]
    t[4, b_nodes[0], 2] = 1.0                                   # one node of B moves alone: B deforms with zero energy
    g2 = cmodal.rigid_groups(mesh, t[[0, 1, 2, 4]])
    assert "B" in g2["deforming_parts"]


@needs_direct
@slow
def test_the_clean_model_has_six_rigid_body_modes_of_zero_energy(sat):
    r = run(sat, ["free_free"])["free_free"]
    assert r.status == cc.PASS, r.findings
    assert r.values["rigid_body_modes"] == 6 and r.values["first_elastic_Hz"] > 100
    assert r.values["rigid_energy_ratio_max"] < 1e-6 and r.values["rigid_fit_residual_max"] < 1e-3
    assert len(r.values["rigid_groups"]) == 1                     # one rigid group: A and B are connected


@needs_direct
@slow
def test_fault_a_disconnected_part_gives_twelve_rigid_modes_and_names_the_groups(tmp_path):
    """Fault 1: B is 20 mm away from A – no connection (it is held by a support of its own, so the study is solvable). The free-free run finds 12 rigid-body modes and says
    which parts move independently."""
    st, parts, mesh = build_sat(tmp_path, "gap", gap=20.0)
    sat = SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=tmp_path)
    r = run(sat, ["free_free"])["free_free"]
    assert r.status == cc.FAIL
    assert r.values["rigid_body_modes"] == 12
    assert sorted(map(sorted, r.values["rigid_groups"])) == [["A"], ["B"]]
    assert any("independently" in f and "A" in f and "B" in f for f in r.findings)


# ---------------------------------------------------------------------------------------------------------------- check 3: unit enforced displacement
@needs_direct
@slow
def test_prescribed_nodes_agree_with_the_standard_static_solution(sat):
    """The machinery of checks 3 / 4 (own assembly + rslab with per-node prescribed displacements, K·u by `recover`) against `run_study`: the bar is clamped at x = 0 and the end
    face is pulled by 0.1 mm – the displacements and the reaction must be those of the study with a Displacement constraint."""
    from fea_core.space import _checks_static as cs
    from fea_core.study import Displacement
    st = copy.deepcopy(sat.study)
    st.masses = []
    st.constraints.append(Displacement("end", ux=0.1, uy=0.0, uz=0.0))        # all three components: the prescribed nodes of the machinery hold all three
    st.modal.enabled = False
    st.load_cases = [LoadCase("pull", [])]
    ref = run_study(st, sat.wd / "ref_pull", mesh=sat.mesh)
    st2 = copy.deepcopy(sat.study)
    st2.masses = []
    ctx = cc.CheckContext(st2, sat.mesh, cc.ModelCheckLimits.template(), cc.ModelCheckOptions(threads=4), sat.wd / "prescribed", lambda s: None)
    end = np.unique(sat.mesh.surface_nodes("end"))
    out = cs.solve_prescribed(ctx, st2, [cs.PrescribedState("pull", end, np.tile([0.1, 0.0, 0.0], (len(end), 1)))], "pull")[0]
    assert np.abs(out.U - ref.steps[0].U).max() < 1e-9
    assert np.abs(out.S - ref.steps[0].S).max() < 1e-3 * np.abs(ref.steps[0].S).max()
    fx = out.RF[sat.mesh.surface_nodes("launch"), 0].sum()
    assert fx == pytest.approx(ref.steps[0].reactions["launch"][0], rel=1e-6)


@needs_direct
@slow
def test_the_clean_model_moves_as_a_rigid_body_under_all_six_unit_states(sat):
    r = run(sat, ["unit_displacement"])["unit_displacement"]
    assert r.status == cc.PASS, r.findings
    assert set(r.values["states"]) == {"UX", "UY", "UZ", "RX", "RY", "RZ"}
    assert r.values["worst_strain_energy_ratio"] < 1e-10 and r.values["worst_reaction_ratio"] < 1e-10
    assert all(s["displacement_deviation"] < 1e-8 for s in r.values["states"].values())


@needs_direct
@slow
@pytest.mark.parametrize("kind", ["fixed", "spring"])
def test_fault_a_stray_support_takes_a_reaction_of_the_unit_displacement(sat, kind):
    """Fault 2: the launch interface is named; another support – a fixed face or a spring to the ground – holds the model: the unit translation is resisted (energy, net reaction)
    and the diagnostic names the support that holds it."""
    bad = copy.deepcopy(sat.study)
    bad.constraints.append(Fixed("end") if kind == "fixed" else ElasticSupport("end", 1.0e3, direction="all", per_area=False))
    r = run(sat, ["unit_displacement"], study=bad, options={"interface_selections": ["launch"], "unit_states": ["UX"]})["unit_displacement"]
    assert r.status == cc.FAIL
    assert r.values["worst_reaction_ratio"] > 1e-5 and r.values["worst_strain_energy_ratio"] > 1e-8
    assert any("'end'" in f and "not on the interface" in f for f in r.findings), r.findings


@needs_direct
@slow
def test_a_rigid_remote_support_can_be_the_interface(sat):
    """The launch interface modelled as a face rigidly tied to a point (all six components held there): the unit states are enforced on the point; the clean model passes,
    a stray fixed face makes it fail."""
    from fea_core.study import RemoteSupport
    st = copy.deepcopy(sat.study)
    st.constraints = [RemoteSupport("launch", [0.0, W_BAR / 2, H_BAR / 2], coupling="rigid")]
    ok = run(sat, ["unit_displacement"], study=st)["unit_displacement"]
    assert ok.status == cc.PASS, ok.findings
    assert ok.values["worst_strain_energy_ratio"] < 1e-10 and ok.values["worst_reaction_ratio"] < 1e-10
    assert all(v["displacement_deviation"] < 1e-8 for v in ok.values["states"].values())
    bad = copy.deepcopy(st)
    bad.constraints.append(Fixed("end"))
    r = run(sat, ["unit_displacement"], study=bad, options={"unit_states": ["UX"], "interface_selections": ["launch"]})["unit_displacement"]
    assert r.status == cc.FAIL and any("'end'" in f for f in r.findings), r.findings


@needs_direct
@slow
def test_the_unit_displacement_reaction_is_the_axial_stiffness_of_a_bar_held_at_both_ends(tmp_path):
    """An analytic reference of the machinery: a bar (nu = 0) held at both ends, the unit translation enforced at one end (the other end is NOT the interface): the net interface
    reaction is E·A·delta / L exactly."""
    st, parts, mesh = build_sat(tmp_path, "bar0", nu=0.0)
    st.masses = []
    st.constraints.append(Fixed("end"))
    sat = SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=tmp_path)
    r = run(sat, ["unit_displacement"], limits={}, options={"interface_selections": ["launch"], "unit_states": ["UX"]})["unit_displacement"]
    ratio = r.values["states"]["UX"]["net_force_ratio"]
    e_lc_delta = r.values["reference"]["E_MPa"] * r.values["reference"]["Lc_mm"] * 1.0
    assert ratio * e_lc_delta == pytest.approx(210000.0 * W_BAR * H_BAR * 1.0 / L_BAR, rel=2e-3)


# ---------------------------------------------------------------------------------------------------------------- check 4: isothermal
@needs_direct
@slow
def test_the_clean_model_expands_freely_without_stress(sat):
    r = run(sat, ["isothermal"])["isothermal"]
    assert r.status == cc.PASS, r.findings
    assert r.values["free_expansion"]["stress_ratio"] < 1e-9 and r.values["free_expansion"]["displacement_deviation"] < 1e-9
    assert r.values["support"]["worst_ratio"] < 1e-9 and len(r.values["support"]["per_selection"]) == 1


@needs_direct
@slow
def test_fault_an_over_constrained_thermal_support_is_detected(tmp_path):
    """Fault 4a: the bar is fixed at both ends – a uniform temperature change cannot expand it: each support pushes on the other with E·alpha·dT·A (nu = 0: exactly)."""
    st, parts, mesh = build_sat(tmp_path, "bar0", nu=0.0)
    st.masses = []
    st.constraints.append(Fixed("end"))
    sat = SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=tmp_path)
    r = run(sat, ["isothermal"])["isothermal"]
    assert r.status == cc.FAIL
    assert r.values["support"]["worst_ratio"] == pytest.approx(1.0, rel=2e-3)
    assert any("statically indeterminate" in f for f in r.findings)
    assert r.values["free_expansion"]["stress_ratio"] < 1e-9              # the model itself is fine: it is the support that over-constrains


@needs_direct
@slow
def test_fault_a_rigid_coupling_over_an_expanding_face_is_a_hidden_over_constraint(sat):
    """Fault 4b: a point mass coupled RIGIDLY to the top face of B – the face cannot expand with the rest: a stress under the free thermal expansion that no support causes."""
    bad = copy.deepcopy(sat.study)
    bad.masses = [PointMass("top_B", M_POINT, coupling="rigid", name="box")]
    r = run(sat, ["isothermal"], study=bad)["isothermal"]
    assert r.status == cc.FAIL
    assert r.values["free_expansion"]["stress_ratio"] > 1e-3
    assert any("resists a uniform expansion" in f for f in r.findings)


# ---------------------------------------------------------------------------------------------------------------- check 5: effective mass
@needs_direct
@slow
def test_effective_mass_of_the_clean_model_and_a_truncated_basis(sat):
    """Fault 5: only the lowest modes were computed – the cumulative effective mass is far below the limit and the check says the basis is truncated."""
    from fea_core.eigen import run_eigen
    from fea_core.results import Results
    full = run(sat, ["effective_mass"])["effective_mass"]
    assert full.status == cc.PASS, full.findings
    assert min(full.values["cumulative_fraction_xyz"]) > 0.5
    st = copy.deepcopy(sat.study)
    st.load_cases = []
    st.modal.enabled, st.modal.modes = True, 2
    ms = run_eigen(st, sat.mesh, sat.wd / "two_modes", lambda s: None)[0]
    res = Results(sat.mesh, [], [], {}, modes=[ms])
    bad = run(sat, ["effective_mass"], results=res)["effective_mass"]
    assert bad.status == cc.FAIL
    assert bad.values["basis_reaches_cutoff"] is False and min(bad.values["cumulative_fraction_xyz"]) < 0.5
    assert any("truncated" in f for f in bad.findings)


# ---------------------------------------------------------------------------------------------------------------- check 7: equilibrium
@needs_direct
@needs_ccx
@slow
def test_equilibrium_of_the_results_and_the_fault_of_an_inconsistent_reaction(sat):
    """Fault 6: a reaction of the stored results was changed after the solution: the balance recomputed from the stored reactions is far off, and differs from the record."""
    st = copy.deepcopy(sat.study)
    st.modal.enabled = False
    res = run_study(st, sat.wd / "solved", mesh=sat.mesh)
    ok = run(sat, ["equilibrium"], results=res)["equilibrium"]
    assert ok.status == cc.PASS, ok.findings
    assert ok.values["worst_imbalance"] < 1e-9 and ok.values["states"][0]["consistent_with_record"] is True
    bad_res = copy.deepcopy(res)
    bad_res.steps[0].reactions["launch"] = bad_res.steps[0].reactions["launch"] * 1.05
    bad = run(sat, ["equilibrium"], results=bad_res)["equilibrium"]
    assert bad.status == cc.FAIL
    assert bad.values["states"][0]["imbalance"] > 1e-3
    assert any("recomputed" in f and "recorded" in f for f in bad.findings) and any("not in equilibrium" in f for f in bad.findings)


def test_recompute_balance_of_a_synthetic_state():
    """No solver: a state without loads and without reactions balances; a reaction without a load does not (the imbalance is 1 = the whole reaction)."""
    from fea_core.mesh import Mesh
    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    mesh = Mesh(nodes, np.zeros((0, 10), np.int64), np.zeros(0, np.int64), ["P"])
    lc = LoadCase("none", [])
    study = SimpleNamespace(load_cases=[lc], masses=[])

    def state(r):
        return SimpleNamespace(equilibrium={"imbalance": 0.0}, frame={}, moment_point=np.zeros(3), reactions={"s": np.asarray(r, float)}, reaction_moments={"s": np.zeros(3)})
    ok = eq.recompute_balance(study, mesh, lc, state([0.0, 0.0, 0.0]))
    assert ok is not None and ok["imbalance"] == 0.0
    bad = eq.recompute_balance(study, mesh, lc, state([0.0, 0.0, 5.0]))
    assert bad is not None and bad["force_imbalance"] is None and bad["force_residual_N"] == [0.0, 0.0, 5.0]           # self-equilibrated state and no scale of the solver: no verdict
    assert bad["scale_missing"] == ["force"]
    withrec = eq.recompute_balance(study, mesh, lc, SimpleNamespace(equilibrium={"force_scale": 5.0, "moment_scale": 50.0}, frame={}, moment_point=np.zeros(3),
                                                                      reactions={"s": np.array([0.0, 0.0, 5.0])}, reaction_moments={"s": np.zeros(3)}))
    assert withrec["imbalance"] == pytest.approx(1.0) and withrec["force_divisor"] == "nodal norms"          # the whole reaction without a load: 100 % of the scale of the record
    assert eq.recompute_balance(study, mesh, None, state([0, 0, 0])) is None                     # no load case behind the state: not recomputable
    # a self-equilibrated state (thermal, preload): two supports with opposite reactions and no load balance at the scale of the reactions, and a state with rounding noise only
    # (1e-9 N and no load) is below the force floor of the model, not a 100 % imbalance
    two = SimpleNamespace(equilibrium={"imbalance": 0.0}, frame={}, moment_point=np.zeros(3), reactions={"a": np.array([0.0, 0.0, 800.0]), "b": np.array([0.0, 0.0, -800.0])},
                          reaction_moments={"a": np.zeros(3), "b": np.zeros(3)})
    assert eq.recompute_balance(study, mesh, lc, two)["force_imbalance"] == 0.0


# ---------------------------------------------------------------------------------------------------------------- the whole run
@needs_direct
@needs_ccx
@slow
def test_the_clean_reference_model_passes_all_seven_checks_and_the_empty_template_passes_none(sat):
    m, c = analytic_mass()
    opts = {"mass_budget_t": m, "cog_budget_mm": list(c), "interface_selections": ["launch"]}          # the default convergence settings: a weaker one is never a full PASS
    done: list[str] = []
    res = mc.run_model_checks(sat.study, sat.mesh, None, LIMITS, {"threads": 4, **opts}, workdir=sat.wd / "all", parts=sat.parts, on_check=lambda r: done.append(r.id))
    by = {r.id: r for r in res}
    assert done == list(cc.CHECK_IDS)
    assert [r.number for r in res] == [1, 2, 3, 4, 5, 6, 7]
    assert all(r.status in (cc.PASS, cc.WARN) for r in res), {k: (v.status, v.findings) for k, v in by.items()}
    assert by["mesh_convergence"].values["frequency_worst_last_change"] < 0.05
    json.dumps(mc.to_dict(res))                                     # JSON-ready
    assert mc.overall_verdict(res) in ("PASS", "WARN")
    assert len(mc.table_rows(res)) == 7 and "Model quality checks" in mc.format_text(res)
    # the same run without limits: the numbers are there, no check says PASS
    nolim = mc.run_model_checks(sat.study, sat.mesh, None, None, {"threads": 4, **opts, "include_convergence": False}, workdir=sat.wd / "nolim", parts=sat.parts)
    assert all(r.status != cc.PASS for r in nolim), [(r.id, r.status) for r in nolim]
    assert {r.id: r.status for r in nolim}["mass"] == cc.UNCONFIRMED
    assert mc.overall_verdict(nolim) in ("UNCONFIRMED_REQUIREMENTS", "INCOMPLETE")
    assert nolim[0].values["fe_mass_kg"] == pytest.approx(m * 1000.0)


@needs_ccx
def test_the_study_of_the_caller_is_not_changed_and_cancel_stops(sat):
    before = json.dumps(sat.study.to_dict(), sort_keys=True)
    ev = threading.Event()
    ev.set()
    with pytest.raises(Exception, match="canceled"):
        mc.run_model_checks(sat.study, sat.mesh, None, LIMITS, {"threads": 4}, workdir=sat.wd / "cancelled", checks=["free_free"], cancel=ev)
    run(sat, ["mass"], options={"solver_mass_check": False})
    assert json.dumps(sat.study.to_dict(), sort_keys=True) == before


def test_a_check_that_cannot_run_is_skipped_and_never_counts_as_passed(sat):
    from fea_core.space import _checks_conv as cconv
    ctx = cc.CheckContext(sat.study, sat.mesh, cc.ModelCheckLimits.template(), cc.ModelCheckOptions(include_convergence=False), sat.wd / "skip", lambda s: None)
    r = cconv.run_convergence_check(ctx)
    assert r.status == cc.SKIPPED
    assert mc.overall_verdict([r]) == "INCOMPLETE"
    ctx = cc.CheckContext(sat.study, sat.mesh, cc.ModelCheckLimits.template(), cc.ModelCheckOptions(), sat.wd / "skip", lambda s: None, results=None)
    no_support = copy.deepcopy(sat.study)
    no_support.constraints = []
    ctx.study = no_support
    assert cmodal.run_effective_mass_check(ctx).status == cc.SKIPPED                            # a free body has no constrained modal masses
    from fea_core.space import _checks_static as cs
    assert cs.run_unit_displacement_check(ctx).status == cc.SKIPPED                             # no interface to enforce the motion on


def test_run_for_session_uses_the_session_objects(sat):
    sess = SimpleNamespace(require_study=lambda: sat.study, parts=sat.parts, valid_mesh=lambda: sat.mesh, results=None, step_path=sat.wd / "sat.step",
                           new_workdir=lambda name: sat.wd / "sess", add_log=lambda s: None)
    opts = {"mass_budget_t": analytic_mass()[0], "solver_mass_check": False}
    out = mc.run_for_session(sess, options=opts, checks=["mass"])                       # no requirement set: no confirmed limit
    assert out["checks"][0]["status"] == "UNCONFIRMED" and out["verdict"] == "INCOMPLETE"        # (one check of the seven: INCOMPLETE at best)
    from fea_core.space.model import LimitEntry, SpaceSettings
    from fea_core.space.requirements import confirm
    from verification import space_cases as SC
    st = copy.deepcopy(sat.study)
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    rs.limits = [LimitEntry(k, v, "", "doc", "rev", "clause") for k, v in LIMITS_DICT.items()]
    st.space = SpaceSettings(requirement_set=rs)
    sess.require_study = lambda: st
    assert mc.run_for_session(sess, options=opts, checks=["mass"])["checks"][0]["status"] == "UNCONFIRMED"       # the set is not confirmed
    confirm(rs, "test engineer", "2026-01-01")
    out = mc.run_for_session(sess, options=opts, checks=["mass"])
    assert out["checks"][0]["status"] == "UNCONFIRMED" and out["checks"][0]["id"] == "mass" and "Model quality checks" in out["text"]       # decided, but the solver check is off
    assert out["checks"][0]["limits"]["mass_tol_rel"]["decided"] is True and "solver_mass_check" in out["checks"][0]["summary"]
    assert out["verdict"] == "INCOMPLETE"                                              # a subset of the seven checks is never a PASS of the model
    rs.limits[0].value = 0.5                                                           # edited after the confirmation: not confirmed any more
    assert mc.run_for_session(sess, options=opts, checks=["mass"])["checks"][0]["limits"]["mass_tol_rel"]["decided"] is False
    json.dumps(out)


def test_limits_entered_by_a_caller_are_never_confirmed():
    lim = cc.ModelCheckLimits.from_entries({"mass_tol_rel": 0.01, "cog_tol_mm": {"value": 1.0, "confirmed": True, "document": "d", "revision": "r", "clause": "c"}})
    assert lim.value("mass_tol_rel") is None and lim.value("cog_tol_mm") is None and lim.entries["mass_tol_rel"].value == 0.01
    j = cc.Judge(lim)
    assert j.compare_max(0.0, "mass_tol_rel", "x") is None and j.finish(1, "ok", {}, [], 0.0).status == cc.UNCONFIRMED
    assert cc.ModelCheckLimits.from_requirement_set(None).value("mass_tol_rel") is None
    from fea_core.space.model import LimitEntry, RequirementSet
    rs = RequirementSet(limits=[LimitEntry("mass_tol_rel", 0.01, "", "d", "r", "c"), LimitEntry("cog_tol_mm", 1.0, "", "d", "r", ""),
                                LimitEntry("other_stage", 1.0, "", "d", "r", "c")], confirmed=True, confirmed_by="x", confirmed_digest="bad")
    assert cc.ModelCheckLimits.from_requirement_set(rs).value("mass_tol_rel") is None              # the confirmation does not match the content
    rs.confirmed_digest = rs.content_digest()
    got = cc.ModelCheckLimits.from_requirement_set(rs)
    assert got.value("mass_tol_rel") == 0.01 and got.value("cog_tol_mm") is None and got.provenance("mass_tol_rel")["confirmed_by"] == "x"


# ---------------------------------------------------------------------------------------------------------------- review fixes: a check that decided nothing is never PASS
def test_a_check_that_compared_nothing_with_a_limit_is_never_pass():
    r = cc.Judge(LIMITS).finish(6, "ok", {}, [], 0.0)
    assert r.status == cc.UNCONFIRMED and "nothing was compared" in r.summary
    j = cc.Judge(LIMITS)
    assert j.compare_max(0.0, "mass_tol_rel", "x") is True and j.status() == cc.PASS
    j.cannot_decide("the launch interface is not named")
    r = j.finish(3, "ok", {}, [], 0.0)
    assert r.status == cc.UNCONFIRMED and "interface is not named" in r.summary and "interface is not named" in r.findings[0]
    j = cc.Judge(LIMITS)
    j.compare_max(1.0, "mass_tol_rel", "x")                                  # a violation still wins
    j.cannot_decide("y")
    assert j.status() == cc.FAIL


def test_the_mass_check_with_nothing_to_compare_is_unconfirmed(sat):
    r = mc.run_model_checks(sat.study, sat.mesh, None, LIMITS, {"threads": 4, "solver_mass_check": False}, workdir=sat.wd / "nothing", checks=["mass"])[0]
    assert r.status == cc.UNCONFIRMED and "nothing was compared" in r.summary and r.values["fe_mass_kg"] > 0


def test_effective_mass_truncation_is_judged_by_the_highest_computed_mode(sat):
    """A basis with a mode above the cutoff covers the window: an incomplete mass is then NOT reported as a truncated basis."""
    from fea_core.eigen import ModeSet
    from fea_core.results import Results
    n = len(sat.mesh.nodes)
    m = analytic_mass()[0]
    em = [[0.2 * m, 0.2 * m, 0.2 * m, 0.0, 0.0, 0.0], [0.1 * m, 0.1 * m, 0.1 * m, 0.0, 0.0, 0.0]]
    ms = ModeSet("modal", "Modes", [1000.0, 20000.0], np.zeros((2, n, 3), np.float32), eff_mass=em, total_mass=[m, m, m, 0.0, 0.0, 0.0], engine="direct")
    res = Results(sat.mesh, [], [], {}, modes=[ms])
    r = run(sat, ["effective_mass"], results=res)["effective_mass"]                     # the cutoff of LIMITS is 12000 Hz: the mode at 20000 Hz is above it
    assert r.status == cc.FAIL and r.values["basis_reaches_cutoff"] is True
    assert not any("truncated" in x for x in r.findings) and any("not complete" in x for x in r.findings)
    ms.values = [1000.0, 2000.0]                                               # no mode reaches the cutoff: truncated
    r = run(sat, ["effective_mass"], results=res)["effective_mass"]
    assert r.values["basis_reaches_cutoff"] is False and any("truncated" in x for x in r.findings)


def test_equilibrium_examines_the_load_cases_of_the_study_not_only_unit_gravity(sat, monkeypatch):
    called: list[list[str]] = []

    def fake_statics(ctx, lcs, name):
        called.append([lc.name for lc in lcs])
        return SimpleNamespace(steps=[])
    monkeypatch.setattr(eq, "run_statics", fake_statics)
    ctx = cc.CheckContext(sat.study, sat.mesh, LIMITS, cc.ModelCheckOptions(), sat.wd / "eq", lambda s: None)
    ctx.shared["gravity_results"] = SimpleNamespace(steps=[])                  # the unit-gravity states of check 1 exist, the load cases of the study were not examined
    eq.run_equilibrium_check(ctx)
    assert called == [["Tip load"]]
    called.clear()
    st = copy.deepcopy(sat.study)
    st.constraints = []                                                         # nothing to run them with: said, never a PASS
    ctx = cc.CheckContext(st, sat.mesh, LIMITS, cc.ModelCheckOptions(), sat.wd / "eq2", lambda s: None)
    ctx.shared["gravity_results"] = SimpleNamespace(steps=[])
    r = eq.run_equilibrium_check(ctx)
    assert not called and r.status == cc.UNCONFIRMED and "not examined" in r.summary
    ctx.results = SimpleNamespace(steps=[SimpleNamespace(name="Tip load", equilibrium={"imbalance": 0.0}, frame=False, moment_point=None, reactions={})])
    r = eq.run_equilibrium_check(ctx)
    assert not called and r.status == cc.UNCONFIRMED and "stored scalar" in " ".join(r.findings)      # covered by the results (not run again); a bare scalar record decides nothing


@needs_direct
@slow
def test_several_supports_without_a_named_interface_give_no_verdict(sat):
    two = copy.deepcopy(sat.study)
    two.constraints.append(Fixed("end"))                                          # both ends are enforced as the interface: a rigid motion costs nothing - and proves nothing
    r = run(sat, ["unit_displacement"], study=two, options={"unit_states": ["UX"]})["unit_displacement"]
    assert r.status == cc.UNCONFIRMED and "not named" in r.summary
    r = run(sat, ["unit_displacement"], study=two, options={"unit_states": ["UX"], "interface_selections": ["launch", "end"]})["unit_displacement"]
    assert r.status == cc.UNCONFIRMED and "not named" not in r.summary and "of the 6 unit states" in r.summary, r.findings        # named: no more undecided interface, one state of six


@needs_direct
@slow
def test_the_isothermal_check_judges_the_stress_of_a_single_support(sat):
    ok = run(sat, ["isothermal"])["isothermal"]
    assert ok.status == cc.PASS, ok.findings
    ratio = ok.values["support"]["max_stress_ratio"]
    assert 1e-6 < ratio < 5.0                                                    # the clamp restrains the lateral expansion at its face: a stress that is judged, not ignored
    no = {k: v for k, v in LIMITS_DICT.items() if k != "thermal_support_stress_ratio_max"}
    r = run(sat, ["isothermal"], limits=cc.ModelCheckLimits.from_entries(no, trusted=True))["isothermal"]
    assert r.status == cc.UNCONFIRMED and "thermal_support_stress_ratio_max" in r.summary
    tight = cc.ModelCheckLimits.from_entries(dict(LIMITS_DICT, thermal_support_stress_ratio_max=ratio / 10), trusted=True)
    r = run(sat, ["isothermal"], limits=tight)["isothermal"]
    assert r.status == cc.FAIL and "supported model" in r.summary

"""Bolts and welds on the own assembly end to end (F13 step 1b, part 2), nothing mocked: a real Gmsh mesh, the job files of `solve_large._run_ngs_job`
(reduction.npz with the beam elements, the unit states of the preload as homogeneous right-hand sides), the assembly worker, the direct solver, the preload by the
unit states, the stress recovery, `joints.evaluate.step_forces` with the stresses of `recovery.element_ip_stress` – against CalculiX on the SAME mesh (the U1 beams
of ccx: a stiffer shear than the exact Timoshenko element, the gap DOF of the preload, a penalty spring for the thermal elongation) and against the closed formulas of
the verification cases.

Studies: (1) a lap joint M10 8.8 with the preload, shear, weight and a temperature change of aluminium plates (the thermal force of the bolt); (2) two bolts M12 with
the preload under a rigid remote load and a rigid remote support with a cylindrical support on a pin hole; (3) a bolt screwed into a thread (ISO 4762) and a countersunk
bolt through a clamped bundle with a clearance (no cut, an intermediate face); (4) a T-joint with two fillet welds under tension and longitudinal shear; (5) a lap joint
with a 3D-printed plate (the stresses at the integration points of an element with an *ORIENTATION: `*EL PRINT S` needs GLOBAL=YES); (6) a lap joint turned by 30° about the x
axis (a tilted bolt) with a PRESCRIBED displacement of the support: the load states keep it, the preload state (homogeneous right-hand side) does not.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.joints.assess import assess_joints, bolt_case_forces
from fea_core.joints.detect import detect
from fea_core.joints.model import PRELOAD_CASE
from fea_core.selectors import find_faces
from fea_core.solve import choose_backend_reason, prepare_mesh, run_study
from fea_core.study import (CylindricalSupport, Fixed, Force, Gravity, LoadCase, Material, MeshSettings, RemoteLoad, RemoteSupport, ThermalLoad)
from verification import joint_models as jm
from verification.cases import _study

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_chain = pytest.mark.skipif(not (HAS_DIRECT and config.CCX_EXE.exists()), reason="numba + rslab (modules accel, direct) and CalculiX are needed")


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


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _both(st, parts, wd):
    """(ccx result, own result, the log of the own run, mesh) on one mesh."""
    from fea_core.solve_large import run_ngs
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.solver.backend = "ccx"
    ref = run_study(st, wd / "ccx", mesh=mesh, parts=parts)
    logs: list[str] = []
    res = run_ngs(st, mesh, wd / "own", [], logs.append, solver="direct")
    return ref, res, logs, mesh


def _compare(ref, res, tol_u=1e-3, tol_s=1e-4, tol_r=1e-4, tol_imbalance=1e-6):
    """The own path against ccx: displacements, stresses, reactions, the equilibrium – the same states in the same order (the preload state last)."""
    assert [s.name for s in ref.steps] == [s.name for s in res.steps]
    f_scale = max(np.abs(v).max() for st in ref.steps for v in st.reactions.values())
    for a, b in zip(ref.steps, res.steps):
        assert _rel(b.U, a.U) < tol_u, (a.name, "U", _rel(b.U, a.U))
        assert _rel(b.S, a.S) < tol_s, (a.name, "S", _rel(b.S, a.S))
        assert b.equilibrium["imbalance"] < tol_imbalance, (a.name, b.equilibrium["imbalance"])
        for k in a.reactions:
            assert np.abs(a.reactions[k] - b.reactions[k]).max() / f_scale < tol_r, (a.name, k, a.reactions[k], b.reactions[k])
    assert res.info["backend"] == "ngs_direct" and ref.info["backend"] == "ccx"


def _bolt_table(st, ref, res):
    """{(bolt, state, quantity): (ccx, own)} of the forces of the bolts and the largest value of every quantity."""
    out, scale = {}, {}
    for b in st.bolts:
        for a, o in zip(ref.steps, res.steps):
            fa = bolt_case_forces(b, a.joints["bolts"][b.name])
            fo = bolt_case_forces(b, o.joints["bolts"][b.name])
            for q in ("F_SA", "M_b"):
                out[(b.name, a.name, q)] = (fa[q], fo[q])
            for k, (pa, po) in enumerate(zip(fa["planes"], fo["planes"])):
                for q in ("F_A", "F_SA", "F_PA", "F_Q", "Q_bolt", "M_b"):
                    out[(b.name, a.name, f"{k}.{q}")] = (pa[q], po[q])
    for (_, _, q), (x, y) in out.items():
        scale[q.split(".")[-1]] = max(scale.get(q.split(".")[-1], 0.0), abs(x))
    return out, scale


def _compare_bolts(st, ref, res, tol: dict | None = None):
    """The forces of the bolts in all states: the difference of each quantity relative to the largest value of that quantity in the study. The axial force
    and the force in the joint are the same to the digits of the .frd of ccx; the shear and the bending of the bolt differ by what the exact Timoshenko element
    differs from the U1 element of ccx (see `tol`)."""
    tol = {"F_SA": 5e-5, "F_A": 5e-5, "F_PA": 5e-5, "F_Q": 5e-5, "Q_bolt": 3e-3, "M_b": 3e-3, **(tol or {})}
    out, scale = _bolt_table(st, ref, res)
    for (n, s, q), (x, y) in out.items():
        k = q.split(".")[-1]
        assert abs(x - y) <= tol[k] * max(scale[k], 1e-12), (n, s, q, x, y, scale[k])
    ua, uo = assess_joints(st, ref), assess_joints(st, res)
    for ca, co in zip(ua.all(), uo.all()):
        assert co.utilization == pytest.approx(ca.utilization, rel=5e-3, abs=1e-4), (ca.name, ca.governing, co.governing)


# ------------------------------------------------------------------ (1) lap joint: preload, shear, weight, temperature (aluminium plates)
def _study_lap_thermal(wd: Path):
    st, parts = _study("lap thermal", wd / "model.step", jm.lap_joint(), MeshSettings(size=5.0))
    al = Material("Al", 70000.0, 0.33, 2.7e-9, Re=250.0, Rm=300.0, alpha=23e-6)
    st.materials["Al"] = al
    for p in ("Deska A", "Deska B"):
        st.part_materials[p] = "Al"
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases += [LoadCase("Smyk", [Force("konecA", [-8000.0, 0, 0])]), LoadCase("Tiha", [Gravity([0.0, 0.0, -9810.0])]),
                      LoadCase("Teplo", [ThermalLoad(value=60.0)]), LoadCase("Smyk + teplo", [Force("konecA", [-4000.0, 0, 500.0]), ThermalLoad(value=-30.0)])]
    return st, parts


@pytest.fixture(scope="module")
def lap_thermal(tmp_path_factory):
    wd = tmp_path_factory.mktemp("lap_thermal")
    st, parts = _study_lap_thermal(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_a_bolted_lap_joint_with_preload_and_temperature_equals_ccx(lap_thermal):
    st, parts, wd, ref, res, logs, mesh = lap_thermal
    assert ref.steps[-1].name == PRELOAD_CASE and res.steps[-1].name == PRELOAD_CASE and len(res.steps) == 5
    _compare(ref, res)
    _compare_bolts(st, ref, res, {"F_SA": 3e-4, "F_A": 3e-4, "F_PA": 3e-4, "F_Q": 3e-4})       # ccx holds the thermal elongation by a penalty spring: 1e-4 of the force
    proto = res.info["solver"]
    assert proto["backend"] == "ngs_direct" and proto["bolts"]["count"] == 1 and proto["bolts"]["preload_states"] == 1 and proto["constraints"]["bolts"] == 1
    assert res.info["joints"]["preload_case"] is True and res.info["joints"]["bolts"].keys() == ref.info["joints"]["bolts"].keys()


@needs_chain
def test_the_preload_is_exact_and_the_thermal_force_has_the_sign_and_the_size_of_vdi(lap_thermal):
    st, parts, wd, ref, res, logs, mesh = lap_thermal
    from fea_core.joints.fem import bolt_basics
    b = st.bolts[0]
    bas = bolt_basics(st, b)
    pre = bolt_case_forces(b, res.step(PRELOAD_CASE).joints["bolts"][b.name])
    assert pre["F_SA"] == pytest.approx(bas["F_Vnom"], rel=1e-9)                              # N = F_V to the accuracy of the solution (not of the .frd)
    assert -pre["planes"][0]["F_PA"] == pytest.approx(bas["F_Vnom"], rel=1e-6)                # the clamped parts carry the same force
    # the plates expand more than the steel bolt (α 23 against 11.5e-6): heating tightens the bolt. VDI 2230: ΔF = ΔT·L_K·(α_P − α_S)/(δ_S + δ_P) – an estimate with the
    # cone method (2.07 kN in the FEM against 2.30 kN here: the plates are bonded over the whole overlap); the exact check is the next test
    th = bolt_case_forces(b, res.step("Teplo").joints["bolts"][b.name])["F_SA"]
    L_K = 2 * 10.0
    vdi = (60.0 - st.thermal.reference_temperature) * L_K * (23e-6 - 11.5e-6) / (bas["delta_S"] + bas["delta_P"])
    assert th > 0 and 0.8 * vdi < th < 1.05 * vdi
    cold = bolt_case_forces(b, res.step("Smyk + teplo").joints["bolts"][b.name])["F_SA"]
    assert cold < 0                                                                            # cooling (−30 K) relaxes the bolt
    # the thermal states keep their equilibrium (the elongation is an internal load of the bolt)
    assert max(s.equilibrium["imbalance"] for s in res.steps) < 1e-9


@pytest.fixture(scope="module")
def lap_bolt_heat(tmp_path_factory):
    """The lap joint with plates that do not expand (α = 1e-12): only the bolt is heated – its force is the elongation of the bolt held by the clamped stack."""
    from fea_core.solve_large import run_ngs
    wd = tmp_path_factory.mktemp("lap_bolt_heat")
    st, parts = _study("lap bolt heat", wd / "model.step", jm.lap_joint(), MeshSettings(size=5.0))
    st.materials[st.default_material].alpha = 1e-12
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases += [LoadCase("Teplo", [ThermalLoad(value=60.0)]), LoadCase("Chlad", [ThermalLoad(value=-25.0)])]
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.solver.backend = "ngs_direct"
    return st, run_ngs(st, mesh, wd / "own", [], lambda s: None, solver="direct")


@needs_chain
def test_the_thermal_elongation_of_a_bolt_is_the_elongation_of_the_preload_states_scaled_exactly(lap_bolt_heat):
    """Heating the bolt by ΔT with plates that do not expand is an initial elongation t = α·ΔT·L of the bolt – the same load that the preload state applies
    (the elongation g that gives N = F_V): N(t) = F_V·t/g, linear and exact (ccx holds the thermal elongation by a penalty spring: 1e-4)."""
    st, res = lap_bolt_heat
    from fea_core.joints.fem import bolt_models
    b = st.bolts[0]
    bm = bolt_models(st, res.mesh)[0]
    g = res.info["solver"]["bolts"]["preload_elongation_mm"][0]
    t_ref = st.thermal.reference_temperature
    for name, value in (("Teplo", 60.0), ("Chlad", -25.0)):             # the temperatures of the states; the bolt is heated by value − t_ref
        n = bolt_case_forces(b, res.step(name).joints["bolts"][b.name])["F_SA"]
        t = bm.alpha * (value - t_ref) * bm.length
        assert n == pytest.approx(bm.preload * t / g, rel=2e-5), (name, n, bm.preload * t / g)
    assert bolt_case_forces(b, res.step("Teplo").joints["bolts"][b.name])["F_SA"] < 0 < bolt_case_forces(b, res.step("Chlad").joints["bolts"][b.name])["F_SA"]


@needs_chain
def test_auto_chooses_the_direct_solver_for_bolts_and_the_iterative_one_refuses_them(lap_thermal):
    from fea_core.solve_large import LargeSolverError, run_ngs, unsupported
    st, parts, wd, ref, res, logs, mesh = lap_thermal
    st.solver.backend = "auto"
    be, why = choose_backend_reason(st, mesh)
    assert be == "ngs_direct" and "CalculiX only" not in why
    assert unsupported(st) == [] and any("bolts" in w and "direct solver only" in w for w in unsupported(st, "hypre"))
    with pytest.raises(LargeSolverError, match="bolts"):
        run_ngs(st, mesh, wd / "hy", [], lambda s: None, solver="hypre")


@needs_chain
def test_the_log_tells_that_the_preload_is_solved_on_the_same_factorization(lap_thermal):
    st, parts, wd, ref, res, logs, mesh = lap_thermal
    text = "\n".join(logs)
    assert "one factorization for 4 states" in text and "Computing stresses" in text
    assert res.info["solver"]["bolts"]["preload_elongation_mm"][0] < 0                         # the bolt was made shorter than the distance between its faces


# ------------------------------------------------------------------ (2) two bolts with the preload + rigid remote support + rigid remote load + cylindrical support
def _study_two_bolts_remote(wd: Path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from verification.cases import cyl, solid
    parts = jm.lap_joint(d=12.0, nbolts=2, pitch=40.0)
    # a pin hole Ø 10 in the far end of plate B (x = 190) for a cylindrical support
    B = next(p for p in parts if p.name == "Deska B")
    B.shape = solid(BRepAlgoAPI_Cut(B.shape, cyl((190.0, 20.0, -13), (0, 0, 1), 5.0, 16)).Shape())
    st, parts = _study("two bolts remote", wd / "model.step", parts, MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.add_selection("kolik", find_faces(parts, part="Deska B", radius=5.0))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints += [CylindricalSupport("kolik", radial=True, tangential=False, axial=False),
                       RemoteSupport("konecA", [-30.0, 20.0, 5.0], [True, True, True, False, False, False], coupling="rigid")]
    st.add_selection("konecB", find_faces(parts, part="Deska B", x=200))
    st.load_cases += [LoadCase("Tah 15 kN", [RemoteLoad("konecB", [250.0, 20.0, -5.0], force=[15000.0, 0, 0], coupling="rigid")]),
                      LoadCase("Ohyb", [RemoteLoad("konecB", [250.0, 20.0, -5.0], force=[0, 0, -1500.0], coupling="rigid")])]
    return st, parts


@pytest.fixture(scope="module")
def two_bolts_remote(tmp_path_factory):
    wd = tmp_path_factory.mktemp("two_bolts_remote")
    st, parts = _study_two_bolts_remote(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_two_preloaded_bolts_under_remote_constraints_equal_ccx(two_bolts_remote):
    st, parts, wd, ref, res, logs, mesh = two_bolts_remote
    assert len(st.bolts) == 2 and res.steps[-1].name == PRELOAD_CASE
    _compare(ref, res)
    _compare_bolts(st, ref, res)
    for a, b in zip(ref.steps, res.steps):
        for sel, info in (a.remote or {}).items():
            o = b.remote[sel]
            assert np.abs(np.subtract(o["force"], info["force"])).max() <= 1e-4 * max(np.abs(info["force"]).max(), 1.0) + 1e-3
    pro = res.info["solver"]
    assert pro["bolts"]["count"] == 2 and pro["bolts"]["preload_states"] == 2
    # the preload of both bolts is exact, each in its own state of the matrix C
    for b in st.bolts:
        pre = bolt_case_forces(b, res.step(PRELOAD_CASE).joints["bolts"][b.name])
        from fea_core.joints.fem import bolt_basics
        assert pre["F_SA"] == pytest.approx(bolt_basics(st, b)["F_Vnom"], rel=1e-9)


# ------------------------------------------------------------------ (3) a thread and a clamped bundle with a clearance
def _study_tapped(wd: Path):
    st, parts = _study("tapped", wd / "model.step", jm.tapped_joint(), MeshSettings(size=4.0))
    st.bolts = detect(parts).bolts
    st.add_selection("blok", find_faces(parts, part="Blok", z=-30))
    st.add_selection("deska", find_faces(parts, part="Deska", x=0))
    st.constraints.append(Fixed("blok"))
    st.load_cases += [LoadCase("Tah", [Force("deska", [0, 0, 3000.0])]), LoadCase("Smyk", [Force("deska", [-2000.0, 0, 0])])]
    return st, parts


def _study_countersunk(wd: Path):
    st, parts = _study("countersunk", wd / "model.step", jm.countersunk_joint(), MeshSettings(size=2.5))
    st.bolts = detect(parts).bolts
    st.add_selection("B", find_faces(parts, part="Deska B", x=0))
    st.add_selection("A", find_faces(parts, part="Deska A", x=60))
    st.constraints.append(Fixed("B"))
    st.load_cases += [LoadCase("Smyk", [Force("A", [1000.0, 0, 0])]), LoadCase("Tah", [Force("A", [0, 0, 600.0])])]
    return st, parts


@pytest.fixture(scope="module", params=["tapped", "countersunk"])
def other_bolts(request, tmp_path_factory):
    wd = tmp_path_factory.mktemp(request.param)
    st, parts = (_study_tapped if request.param == "tapped" else _study_countersunk)(wd)
    return (request.param, st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_a_threaded_and_a_countersunk_bolt_equal_ccx(other_bolts):
    kind, st, parts, wd, ref, res, logs, mesh = other_bolts
    from fea_core.joints.fem import has_gap
    assert st.bolts and (kind == "tapped") == (st.bolts[0].connection == "tapped")
    _compare(ref, res, tol_u=2e-3, tol_s=2e-3)
    # the bolt of the countersunk joint has a clearance between the plates: no cut, no preload state (as in ccx)
    assert has_gap(st.bolts[0]) == (kind == "countersunk")
    assert (res.steps[-1].name == PRELOAD_CASE) == (kind == "tapped")
    _compare_bolts(st, ref, res, {"Q_bolt": 2e-2, "M_b": 2e-2})


# ------------------------------------------------------------------ (5) a joint with a printed plate
@pytest.fixture(scope="module")
def printed_joint(tmp_path_factory):
    from fea_core import materials as matlib
    wd = tmp_path_factory.mktemp("printed_joint")
    st, parts = _study("printed joint", wd / "model.step", jm.lap_joint(), MeshSettings(size=5.0))
    pla = matlib.by_name("Tisk PLA Bambu Basic")
    st.add_material(pla)
    st.part_materials["Deska A"] = pla.name
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases += [LoadCase("Smyk", [Force("konecA", [-2000.0, 0, 0])]), LoadCase("Tah", [Force("konecA", [0, 0, 800.0])])]
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_the_force_through_the_joint_plane_of_a_printed_plate_is_the_load_in_both_paths(printed_joint):
    """The clamped plate has elements with an *ORIENTATION: ccx printed their stresses at the integration points in the material axes (`*EL PRINT S` without GLOBAL=YES), so the
    forces of the joint from them were wrong in the ccx path (793 N through the plane for a load of 800 N, a transverse force of 300 N for none). The force transferred by the
    plane (bolt + the clamped interface) is the load; the preload state transfers nothing."""
    st, parts, wd, ref, res, logs, mesh = printed_joint
    b = st.bolts[0]
    for r in (ref, res):
        smyk = bolt_case_forces(b, r.step("Smyk").joints["bolts"][b.name])["planes"][0]
        tah = bolt_case_forces(b, r.step("Tah").joints["bolts"][b.name])["planes"][0]
        pre = bolt_case_forces(b, r.step(PRELOAD_CASE).joints["bolts"][b.name])["planes"][0]
        assert smyk["F_Q"] == pytest.approx(2000.0, rel=2e-4) and abs(smyk["F_A"]) < 1e-3 * 2000.0
        assert tah["F_A"] == pytest.approx(800.0, rel=2e-4) and tah["F_Q"] < 2e-3 * 800.0
        assert abs(pre["F_A"]) < 1e-3 * abs(pre["F_SA"]) and pre["F_Q"] < 1e-2 * abs(pre["F_SA"])
    # the two paths agree on the forces of the bolt (the printed plate itself differs by ~1e-3 between ccx, which writes the orthotropic constants of the groups,
    # and the own assembly, which takes the full matrix C – independent of the bolt)
    _compare_bolts(st, ref, res, {"F_Q": 5e-4})
    assert _rel(res.step("Smyk").U, ref.step("Smyk").U) < 2e-3


# ------------------------------------------------------------------ (6) a tilted bolt, a prescribed displacement
def _rotated_lap_joint(angle_deg: float):
    """The parts of the lap joint turned about the x axis (the faces x = const stay x = const, the bolt axis is tilted)."""
    import math
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
    from fea_core.geometry import Part
    from verification.cases import solid
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(angle_deg))
    return [Part(p.name, solid(BRepBuilderAPI_Transform(p.shape, t, True).Shape())) for p in jm.lap_joint()]


@pytest.fixture(scope="module")
def tilted_joint(tmp_path_factory):
    from fea_core.study import Displacement
    wd = tmp_path_factory.mktemp("tilted_joint")
    st, parts = _study("tilted", wd / "model.step", _rotated_lap_joint(30.0), MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Displacement("vetknuti", ux=0.01, uy=0.0, uz=0.0))
    st.load_cases += [LoadCase("Tah", [Force("konecA", [-3000.0, 0, 0])]), LoadCase("Bok", [Force("konecA", [0, 1500.0, 700.0])])]
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_a_tilted_bolt_with_a_prescribed_displacement_equals_ccx(tilted_joint):
    st, parts, wd, ref, res, logs, mesh = tilted_joint
    b = st.bolts[0]
    assert abs(abs(b.e[0]) - 0.0) < 1e-6 and 0.4 < abs(b.e[1]) < 0.6 and res.steps[-1].name == PRELOAD_CASE      # the axis is tilted 30° out of z
    _compare(ref, res)
    _compare_bolts(st, ref, res)
    # the load states keep the prescribed displacement of the support, the preload state (a homogeneous system) has none
    idx = mesh.surface_nodes("vetknuti")
    for s in res.steps[:-1]:
        assert np.allclose(s.U[idx, 0], 0.01, atol=1e-12)
    assert np.abs(res.step(PRELOAD_CASE).U[idx]).max() == 0.0
    assert bolt_case_forces(b, res.step(PRELOAD_CASE).joints["bolts"][b.name])["F_SA"] == pytest.approx(bolt_case_forces(b, ref.step(PRELOAD_CASE).joints["bolts"][b.name])["F_SA"], rel=1e-5)


# ------------------------------------------------------------------ (4) welds
def _study_welds(wd: Path):
    st, parts = _study("welds", wd / "model.step", jm.t_joint_fillet(), MeshSettings(size=5.0))
    st.welds = detect(parts).welds
    st.add_selection("pata", find_faces(parts, part="Pasnice", z=-10))
    st.add_selection("hlava", find_faces(parts, part="Stojina", z=100))
    st.constraints.append(Fixed("pata"))
    st.load_cases += [LoadCase("Tah", [Force("hlava", [0, 0, 20000.0])]),
                      LoadCase("Smyk", [RemoteLoad("hlava", [0, 50, 0], force=[0, 10000.0, 0], coupling="distributing")]),
                      LoadCase("Teplo", [ThermalLoad(value=40.0)])]
    st.materials[st.default_material].alpha = 1.2e-5
    return st, parts


@pytest.fixture(scope="module")
def welds(tmp_path_factory):
    wd = tmp_path_factory.mktemp("welds")
    st, parts = _study_welds(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_welds_equal_ccx_and_the_forces_of_the_segments_are_the_load(welds):
    st, parts, wd, ref, res, logs, mesh = welds
    _compare(ref, res, tol_u=2e-4, tol_s=5e-4)                  # measured: dU 4.9e-6, dS 1.0e-5 (the weld toe is a singularity: the digits of the .frd of ccx)
    wa = {s.name: s.joints["welds"] for s in ref.steps}
    wo = {s.name: s.joints["welds"] for s in res.steps}
    for state in wa:
        for name in wa[state]:
            Ta, To = np.asarray(wa[state][name]["T"]), np.asarray(wo[state][name]["T"])
            assert np.abs(Ta - To).max() <= 1e-4 * max(np.abs(Ta).max(), 1.0), (state, name)
            sa, so = np.asarray(wa[state][name]["stress"]), np.asarray(wo[state][name]["stress"])
            assert np.abs(sa - so).max() <= 1e-4 * max(np.abs(sa).max(), 1e-9), (state, name)
    Tz = sum(np.asarray(w["T"]).sum(axis=0)[2] for w in wo["Tah"].values())
    assert Tz == pytest.approx(20000.0, rel=1e-5)                                         # the forces of the segments of both welds = the load
    ua, uo = assess_joints(st, ref), assess_joints(st, res)
    for ca, co in zip(ua.welds, uo.welds):
        assert co.utilization == pytest.approx(ca.utilization, rel=1e-3, abs=1e-4)
    assert res.info["joints"]["welds"].keys() == ref.info["joints"]["welds"].keys()
    assert "Weld" in next(iter(res.info["joints"]["welds"])) or res.info["joints"]["welds"]


@needs_chain
def test_welds_go_to_the_iterative_solver_too(welds):
    """The weld needs no constraint of the own assembly: only the mesh (opened roots) and the stresses at the integration points – hypre can solve it as well."""
    from fea_core import hypre
    from fea_core.solve_large import unsupported
    st, parts, wd, ref, res, logs, mesh = welds
    assert unsupported(st) == [] and unsupported(st, "hypre") == []
    if not hypre.available():
        pytest.skip("the iterative solver (module hypre, MPI) is not installed")
    from fea_core.solve_large import run_ngs
    st.solver.nproc = 2
    hy = run_ngs(st, mesh, wd / "hypre", [], lambda s: None, solver="hypre")
    for a, b in zip(res.steps, hy.steps):
        Ta, Tb = np.asarray(a.joints["welds"][next(iter(a.joints["welds"]))]["T"]), np.asarray(b.joints["welds"][next(iter(b.joints["welds"]))]["T"])
        assert np.abs(Ta - Tb).max() <= 1e-3 * max(np.abs(Ta).max(), 1.0)


# ------------------------------------------------------------------ (5) the settlement consumes the preload of one bolt (F_V clamped at 0), the other bolt is preloaded
def _study_consumed_preload(wd: Path):
    """Two M12 in a lap joint; bolt 1 is tightened with 4 kN (the settlement of ~5.7 kN consumes it: F_V = ½(F_M,max + F_M,min) − F_Z < 0 → 0), bolt 2 with 40 kN."""
    st, parts = _study("consumed preload", wd / "model.step", jm.lap_joint(d=12.0, nbolts=2, pitch=40.0), MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.joints.preload_case = True
    st.bolts[0].preload_mode, st.bolts[0].preload = "force", 4000.0
    st.bolts[1].preload_mode, st.bolts[1].preload = "force", 40000.0
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=200))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Smyk 20 kN", [Force("konecA", [-20000.0, 0, 0])]))
    return st, parts


@pytest.fixture(scope="module")
def consumed_preload(tmp_path_factory):
    wd = tmp_path_factory.mktemp("consumed_preload")
    st, parts = _study_consumed_preload(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_a_preload_consumed_by_the_settlement_is_zero_not_compression_in_both_paths(consumed_preload):
    """The bolt that keeps no preload is cut in the preload state (carries no force, as a bolt of zero preload does in ccx: a free gap with a zero load), the other bolt
    is preloaded exactly; ccx and the own assembly agree, and the readiness check and the assessment say what happened."""
    st, parts, wd, ref, res, logs, mesh = consumed_preload
    from fea_core.joints.fem import bolt_basics
    from fea_core.solve import check_study
    bas = [bolt_basics(st, b) for b in st.bolts]
    assert bas[0]["preload_consumed"] and bas[0]["F_Vraw"] < -1000.0 and bas[0]["F_Vnom"] == 0.0        # the review: −2 448 N without the clamp
    assert not bas[1]["preload_consumed"] and bas[1]["F_Vnom"] == bas[1]["F_Vraw"] > 20000.0
    _compare(ref, res)
    _compare_bolts(st, ref, res)
    for run, rel in ((res, 1e-9), (ref, 1e-5)):
        n1 = bolt_case_forces(st.bolts[0], run.step(PRELOAD_CASE).joints["bolts"][st.bolts[0].name])["F_SA"]
        n2 = bolt_case_forces(st.bolts[1], run.step(PRELOAD_CASE).joints["bolts"][st.bolts[1].name])["F_SA"]
        assert abs(n1) < 1e-3 and n2 == pytest.approx(bas[1]["F_Vnom"], rel=rel), (n1, n2)            # not −2 448 N (compression)
    assert res.info["solver"]["bolts"]["preload_states"] == 2
    errors, warns = check_study(st, [p.name for p in parts])
    mine = [w for w in warns if "settlement" in w]
    assert not errors and len(mine) == 1 and mine[0].startswith(f"{st.bolts[0].name}:") and "F_V = 0" in mine[0]
    for run in (ref, res):
        a = assess_joints(st, run)
        c0, c1 = a.bolts
        assert any("consumes the given preload" in w for w in c0.warnings) and not any("consumes" in w for w in c1.warnings)
        assert any("consumes the preload" in r[2] for r in c0.rows if r[0].startswith("VDI 2230: Residual clamp force"))


# ------------------------------------------------------------------ (6) three plates on a fitted bolt: the cut of the beam lies AT the bearing position of the middle plate
def _study_three_plates_fitted(wd: Path):
    """Plates A / B / C (10 mm each, one conforming mesh), a DIN 7968 fitted bolt through them: the bearing positions are the middle planes of the plates, so the middle of
    the beam (the preload cut) is the position of plate B. The hole of B is wider (12.4 against 11 mm) so that the hole walls of two plates do not share the circle
    of nodes at the interface (the own assembly would refuse it, `test_two_bearing_faces_of_a_bolt_that_share_a_node_are_refused_with_the_reason`)."""
    from fea_core.geometry import Part
    t = 10.0
    plates = [Part(f"Deska {n}", jm.plate_with_holes(0, 0, -k * t, 100, 40, t, [(50, 20)], d)) for k, (n, d) in enumerate((("A", 11.0), ("B", 12.4), ("C", 11.0)))]
    parts = plates + [Part("DIN 7968 M10x50 5.6", jm.hex_bolt(50, 20, t, 10.0, 40.0))]
    st, parts = _study("three plates fitted", wd / "model.step", parts, MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    assert len(st.bolts) == 1 and st.bolts[0].connection == "fitted" and len(st.bolts[0].plates) == 3
    st.joints.preload_case = True
    st.bolts[0].preload_mode, st.bolts[0].preload = "force", 20000.0
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=100))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.add_selection("konecC", find_faces(parts, part="Deska C", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases += [LoadCase("Smyk A", [Force("konecA", [-5000.0, 0, 0])]),
                      LoadCase("Smyk A + C", [Force("konecA", [-3000.0, 0, 0]), Force("konecC", [2000.0, 0, 0])])]
    return st, parts


@pytest.fixture(scope="module")
def three_plates_fitted(tmp_path_factory):
    wd = tmp_path_factory.mktemp("three_plates_fitted")
    st, parts = _study_three_plates_fitted(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_the_preload_cut_at_a_bearing_position_is_in_the_element_after_it_as_in_ccx(three_plates_fitted):
    """With the elongation in the element BEFORE the position (the first version of the own path) the preload state pulled plates A and B together, ccx pulls B and C: dU 0.79,
    the bolt force of the preload state 12.7 kN against 54 N in the plane. Now the same element as ccx: the states agree to the digits of the .frd."""
    st, parts, wd, ref, res, logs, mesh = three_plates_fitted
    from fea_core.joints import own
    from fea_core.joints.fem import bolt_models
    bm = bolt_models(st, mesh)[0]
    ch = own.chain_of(bm)
    assert np.allclose(ch.s, [0.0, 10.0, 20.0]) and float((bm.points[bm.split] - bm.points[0]) @ ch.e) == pytest.approx(10.0) and ch.cut == 1
    assert res.steps[-1].name == PRELOAD_CASE and len(res.steps) == 3
    _compare(ref, res, tol_u=1e-5, tol_s=5e-5)
    _compare_bolts(st, ref, res)
    pre_a = bolt_case_forces(st.bolts[0], ref.step(PRELOAD_CASE).joints["bolts"][st.bolts[0].name])
    pre_o = bolt_case_forces(st.bolts[0], res.step(PRELOAD_CASE).joints["bolts"][st.bolts[0].name])
    assert abs(pre_o["F_SA"]) > 1.0 and pre_o["F_SA"] == pytest.approx(pre_a["F_SA"], rel=1e-4)
    assert [p["F_A"] for p in pre_o["planes"]] == pytest.approx([p["F_A"] for p in pre_a["planes"]], abs=0.05)


# ------------------------------------------------------------------ (7) two bolts so close that their washer imprints share nodes
@pytest.fixture(scope="module")
def close_bolts(tmp_path_factory):
    wd = tmp_path_factory.mktemp("close_bolts")
    st, parts = _study("close bolts", wd / "model.step", jm.lap_joint(d=12.0, nbolts=2, pitch=24.0, washers=True), MeshSettings(size=4.0))   # washers Ø24 at a pitch of 24 mm
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=168.0))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Smyk", [Force("konecA", [-5000.0, 0, 0])]))
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return st, parts, wd, mesh


@needs_chain
def test_two_bolts_with_overlapping_imprints_go_to_ccx_up_front_and_a_manual_choice_gets_the_reason(close_bolts):
    from fea_core.solve_large import LargeSolverError
    st, parts, wd, mesh = close_bolts
    assert len(st.bolts) == 2
    backend, why = choose_backend_reason(st, mesh)
    assert backend == "ccx" and "Bolt 2: a bearing face of the bolt shares nodes with the bearing face of the bolt 'Bolt 1'" in why
    st.solver.backend = "ngs_direct"                              # a manual choice: the error names both bolts before any assembly
    with pytest.raises(LargeSolverError, match=r"shares nodes with the bearing face of the bolt 'Bolt 1'"):
        run_study(st, wd / "manual", mesh=mesh, parts=parts)

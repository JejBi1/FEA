"""What the review of the bolts and welds on the own assembly could not cover (F13 step 1b, part 2), nothing mocked – the own path against CalculiX on the SAME mesh:

(1) welds on printed and orthotropic parts: the stresses at the integration points of the weld bodies and the nodal stresses of the sheets (`recovery`) are in the global axes,
    the weld toe hot-spot (IIW) and its fatigue, the weld fatigue (EN 1993-1-9);
(2) intermittent welds (the groups and the gap rows of EN 1993-1-8 4.3.2.1);
(3) fatigue (`run_fatigue`, the VDI fatigue of the bolt) and combinations (`combine_results`) built on the results of the own path with a bolt;
(4) a bolt together with a *TIE of a separately meshed block and an elastic support (all three in one reduced system).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import materials as matlib
from fea_core.combinations import Combination, combine_results
from fea_core.fatigue.analysis import run_fatigue
from fea_core.fatigue.history import Channel, LoadHistory
from fea_core.fatigue.settings import FatigueSettings
from fea_core.geometry import Part
from fea_core.joints.assess import assess_joints, intermittent_groups
from fea_core.joints.detect import detect
from fea_core.joints.model import PRELOAD_CASE
from fea_core.printing import PrintSpec
from fea_core.selectors import find_faces
from fea_core.study import ElasticSupport, Fixed, Force, LoadCase, Material, MeshSettings
from tests.test_bolts_own_e2e import _both, _compare, _compare_bolts, _rslab_on_the_path_of_the_worker, needs_chain   # noqa: F401  (the fixture is autouse here too)
from verification import joint_models as jm
from verification.cases import _study, box
from verification.joint_cases import w1_build, w2_build

PLA = matlib.by_name("Tisk PLA Bambu Basic")
ORTHO = {"E1": 150000.0, "E2": 210000.0, "E3": 90000.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.28, "G12": 70000.0, "G13": 40000.0, "G23": 50000.0}


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)) if a.size else 0.0


def _compare_welds(st, ref, res, parts, tol_seg=1e-5, tol_hs=5e-5, tol_d=1e-3):
    """Segment forces and stresses of every weld and state, the toe hot-spot, the fatigue of the toe and of the weld, the utilization – ccx against the own path."""
    for a, b in zip(ref.steps, res.steps):
        for w in st.welds:
            wa, wo = (a.joints or {}).get("welds", {}).get(w.name), (b.joints or {}).get("welds", {}).get(w.name)
            if wa is None:
                continue
            assert _rel(wo["T"], wa["T"]) < tol_seg and _rel(wo["stress"], wa["stress"]) < tol_seg, (a.name, w.name)
    ua, uo = assess_joints(st, ref, parts), assess_joints(st, res, parts)
    assert ua.warnings == uo.warnings and ua.fatigue == uo.fatigue
    seen = {"toe": 0, "toe_fatigue": 0, "weld_fatigue": 0}
    for ca, co in zip(ua.welds, uo.welds):
        assert co.utilization == pytest.approx(ca.utilization, rel=tol_d, abs=1e-6) and co.governing == ca.governing, (ca.name, ca.governing, co.governing)
        assert [r[0] for r in ca.rows] == [r[0] for r in co.rows]
        for ra, ro in zip(ca.rows, co.rows):
            if np.isfinite(ra[1]):
                assert ro[1] == pytest.approx(ra[1], rel=tol_d, abs=1e-6), (ca.name, ra[0])
        ta, to = (ca.data or {}).get("toe"), (co.data or {}).get("toe")
        if ta and to:
            assert _rel(to["hotspot"], ta["hotspot"]) < tol_hs and ta["t"] == pytest.approx(to["t"])
            seen["toe"] += 1
            if ta.get("fatigue") and to.get("fatigue"):
                assert to["fatigue"]["damage"] == pytest.approx(ta["fatigue"]["damage"], rel=tol_d)
                assert (to["fatigue"]["segment"], to["fatigue"]["side"]) == (ta["fatigue"]["segment"], ta["fatigue"]["side"])
                seen["toe_fatigue"] += 1
        fa, fo = (ca.data or {}).get("fatigue"), (co.data or {}).get("fatigue")
        if fa and fo:
            assert fo["damage"] == pytest.approx(fa["damage"], rel=tol_d)
            seen["weld_fatigue"] += 1
    return ua, uo, seen


# ------------------------------------------------------------------ (1) welds on printed / orthotropic parts, the toe hot-spot, the fatigue of the toe
@pytest.fixture(scope="module")
def toe_on_printed_plate(tmp_path_factory):
    """W2 (a stub on a plate, two fillet welds of compliant material, a bending load, sine ±1) with the PLATE printed (grid 20 %, perimeters along the walls: ccx ANISO groups)."""
    wd = tmp_path_factory.mktemp("toe_printed")
    st, parts = w2_build(wd)
    st.mesh.size = 4.0
    st.materials[PLA.name] = PLA
    st.part_materials["Plech"] = PLA.name
    st.printing.parts["Plech"] = PrintSpec("Plech", infill=0.2, pattern="grid")
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_the_toe_hotspot_and_its_fatigue_on_a_printed_plate_equal_ccx(toe_on_printed_plate):
    """The nodal stress of the printed sheet at the toe (IIW 0.4t / 1.0t extrapolation) comes from the recovery with the full C of the groups in the global axes (own) or from the .frd of ccx
    (ANISO groups after the fix of `elastic_card`): measured dU 6e-7, dS 1e-5, hot-spot 1e-5 of its maximum, the fatigue damage of the toe 1e-5."""
    st, parts, wd, ref, res, logs, mesh = toe_on_printed_plate
    assert res.info["backend"] == "ngs_direct" and ref.info["backend"] == "ccx"
    _compare(ref, res, tol_u=1e-5, tol_s=5e-5, tol_r=2e-5)
    ua, uo, seen = _compare_welds(st, ref, res, parts)
    assert seen["toe"] == 2 and seen["toe_fatigue"] == 2


@pytest.fixture(scope="module")
def weld_orthotropic(tmp_path_factory):
    """W1 (a T-joint, two fillet welds, tension + longitudinal shear, fatigue) with an orthotropic web and orthotropic weld bodies (a general orthotropic material, Cartesian axes)."""
    wd = tmp_path_factory.mktemp("weld_ortho")
    st, parts = w1_build(wd)
    o = Material("Ortho web", 210000.0, 0.3, 7.85e-9, Re=235.0, Rm=360.0, ortho=ORTHO)
    ow = Material("Ortho weld", 200000.0, 0.3, 7.85e-9, Re=235.0, Rm=360.0, ortho=ORTHO)
    st.materials[o.name] = o
    st.materials[ow.name] = ow
    st.part_materials["Stojina"] = o.name
    for w in st.welds:
        st.part_materials[w.part] = ow.name
    return (st, parts, wd) + _both(st, parts, wd)


@pytest.fixture(scope="module")
def weld_printed(tmp_path_factory):
    """The same T-joint with a printed flange (grid 20 %, ANISO groups in ccx) and printed weld bodies (solid PLA extrusion with the print axes) – the stresses of the weld bodies at the
    integration points come from the full C of the printed element in the global axes (own) or from `*EL PRINT S, GLOBAL=YES` (ccx). (Orthotropy together with printing stays in ccx.)"""
    wd = tmp_path_factory.mktemp("weld_printed")
    st, parts = w1_build(wd)
    st.materials[PLA.name] = PLA
    st.part_materials["Pasnice"] = PLA.name
    st.printing.parts["Pasnice"] = PrintSpec("Pasnice", infill=0.2, pattern="grid")
    for w in st.welds:
        st.part_materials[w.part] = PLA.name
        st.printing.parts[w.part] = PrintSpec(w.part, enabled=False)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
@pytest.mark.parametrize("fixture", ["weld_orthotropic", "weld_printed"])
def test_welds_with_orthotropic_or_printed_parts_and_bodies_equal_ccx(request, fixture):
    """The weld forces come from the stresses at the integration points of the weld bodies (`recovery.element_ip_stress`, global axes) and the toe hot-spot from the nodal stresses of the
    sheet. Measured: segments 6e-8, toe hot-spot 1.2e-5, the fatigue of the weld identical, dU ≤ 5e-6, dS ≤ 1e-5."""
    st, parts, wd, ref, res, logs, mesh = request.getfixturevalue(fixture)
    _compare(ref, res, tol_u=1e-5, tol_s=5e-5, tol_r=2e-5)
    ua, uo, seen = _compare_welds(st, ref, res, parts)
    assert seen["toe"] == 2 and seen["toe_fatigue"] == 2 and seen["weld_fatigue"] == 2
    assert res.info["joints"]["welds"].keys() == ref.info["joints"]["welds"].keys()


# ------------------------------------------------------------------ (2) intermittent welds
@pytest.fixture(scope="module")
def intermittent(tmp_path_factory):
    """A T-joint with two fillet weld segments per side (y 0–40 and 60–100): four weld bodies, the gap of 20 mm."""
    wd = tmp_path_factory.mktemp("intermittent")
    z, t = 5.0 * 2 ** 0.5, 10.0

    def prism(pts, y0, length):
        return jm.prism([(x, y0, zz) for x, zz in pts], length, (0, 1, 0))
    parts = [Part("Pasnice", jm.solid(box(-50, 0, -t, 100, 100.0, t))), Part("Stojina", jm.solid(box(-t / 2, 0, 0, t, 100.0, 100.0)))]
    for k, (y0, length) in enumerate(((0.0, 40.0), (60.0, 40.0)), 1):
        parts.append(Part(f"Svar a5 vpravo {k}", jm.solid(prism([(t / 2, 0), (t / 2 + z, 0), (t / 2, z)], y0, length))))
        parts.append(Part(f"Svar a5 vlevo {k}", jm.solid(prism([(-t / 2, 0), (-t / 2, z), (-t / 2 - z, 0)], y0, length))))
    st, parts = _study("intermittent", wd / "model.step", parts, MeshSettings(size=5.0))
    st.welds = detect(parts).welds
    st.add_selection("pata", find_faces(parts, part="Pasnice", z=-10))
    st.add_selection("hlava", find_faces(parts, part="Stojina", z=100))
    st.constraints.append(Fixed("pata"))
    st.load_cases += [LoadCase("Tah 20 kN", [Force("hlava", [0, 0, 20000.0])]), LoadCase("Ohyb", [Force("hlava", [0, 6000.0, 0])])]
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_intermittent_welds_are_grouped_and_assessed_the_same_in_both_paths(intermittent):
    st, parts, wd, ref, res, logs, mesh = intermittent
    assert len(st.welds) == 4
    _compare(ref, res, tol_u=1e-5, tol_s=5e-5, tol_r=2e-5)
    ga, go = intermittent_groups(st, ref), intermittent_groups(st, res)
    assert [g["welds"] for g in ga] == [g["welds"] for g in go] and len(ga) == 2
    for a, b in zip(ga, go):
        assert b["gaps"] == pytest.approx(a["gaps"], abs=1e-6) and b["gaps"] == pytest.approx([20.0], abs=1e-6) and b["t_min"] == pytest.approx(a["t_min"])
    ua, uo, seen = _compare_welds(st, ref, res, parts)
    assert seen["toe"] == 4
    assert all(any("4.3.2.1" in r[0] or "gap" in r[0].lower() for r in c.rows) for c in uo.welds)         # the gap rows of the group are in every weld of the group


# ------------------------------------------------------------------ (3) fatigue and combinations on the results with a bolt
@pytest.fixture(scope="module")
def fatigue_lap(tmp_path_factory):
    wd = tmp_path_factory.mktemp("fatigue_lap")
    st, parts = _study("fatigue lap", wd / "model.step", jm.lap_joint(), MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases += [LoadCase("Smyk", [Force("konecA", [-6000.0, 0, 0])]), LoadCase("Tah", [Force("konecA", [0, 0, 900.0])])]
    ch = [Channel("Smyk", "sine", amplitude=0.5, mean=0.5, cycles=1000), Channel("Tah", "sine", amplitude=1.0, cycles=1000, phase_deg=40.0)]
    st.fatigue = FatigueSettings(LoadHistory(ch), repeats=500.0)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_fatigue_of_the_structure_and_of_the_bolt_on_the_own_results_equal_ccx(fatigue_lap):
    """`run_fatigue` superposes the unit states of the own path (the state "Bolt preload" is among them with a zero factor); the VDI fatigue of the bolt (σ_ASV, the force and the bending of the bolt per
    state) differs by what the exact Timoshenko element differs from U1 in the bending (≤ 0.1 %); the signed equivalent stress is ambiguous at points with a pure shear (one node of 11 540 flips)."""
    st, parts, wd, ref, res, logs, mesh = fatigue_lap
    fa, fo = run_fatigue(st, ref), run_fatigue(st, res)
    assert fa.load_cases == fo.load_cases and PRELOAD_CASE in fo.load_cases
    assert float(fo.damage.max()) == pytest.approx(float(fa.damage.max()), rel=5e-4) and float(fo.safety.min()) == pytest.approx(float(fa.safety.min()), rel=5e-4)
    d = np.abs(fa.amplitude - fo.amplitude) / fa.amplitude.max()
    assert np.quantile(d, 0.999) < 1e-4 and (d > 1e-3).sum() <= 2 and np.median(d) < 1e-5
    ua, uo = assess_joints(st, ref, parts), assess_joints(st, res, parts)
    assert ua.fatigue and uo.fatigue
    va, vo = ua.bolts[0].data["vdi"]["fatigue"], uo.bolts[0].data["vdi"]["fatigue"]
    assert vo["sigma_a"] == pytest.approx(va["sigma_a"], rel=2e-3) and vo["damage"] == pytest.approx(va["damage"], rel=1e-2)
    assert uo.bolts[0].utilization == pytest.approx(ua.bolts[0].utilization, rel=5e-3)
    _compare_bolts(st, ref, res)


@needs_chain
def test_a_combination_with_the_preload_state_of_the_own_path_equals_ccx(fatigue_lap):
    st, parts, wd, ref, res, logs, mesh = fatigue_lap
    comb = Combination("K1", {"Smyk": 1.35, "Tah": 1.5, PRELOAD_CASE: 1.0})
    ca, co = combine_results(st, ref, comb), combine_results(st, res, comb)
    fs = max(np.abs(v).max() for v in ca.reactions.values())
    assert _rel(co.U, ca.U) < 2e-5 and _rel(co.S, ca.S) < 5e-5
    assert max(float(np.abs(co.reactions[k] - ca.reactions[k]).max()) for k in ca.reactions) / fs < 2e-5
    assert co.equilibrium["reaction_total"] == pytest.approx(ca.equilibrium["reaction_total"], abs=2e-5 * fs)


# ------------------------------------------------------------------ (4) a bolt + *TIE of a separately meshed block + an elastic support
def _study_bolt_tie_elastic(wd: Path):
    """The lap joint plus a block on plate A (away from the bolt) meshed separately and tied by *TIE (the plates carry the bolt imprints, the mesher refuses to mesh them separately),
    the end of plate B on an elastic support (50 N/mm³ in all directions) – the bolt, the *TIE and the spring in one reduced system; the loads act on the block."""
    st, parts = _study("bolt tie elastic", wd / "model.step", jm.lap_joint() + [Part("Blok", box(8, 0, 10, 30, 40, 20))], MeshSettings(size=5.0))
    st.mesh.separate = ["Blok"]
    st.bolts = detect(parts).bolts
    st.add_selection("opora", find_faces(parts, part="Deska B", x=160))
    st.add_selection("horni", find_faces(parts, part="Blok", z=30))
    st.constraints.append(ElasticSupport("opora", 50.0, "all"))
    st.load_cases += [LoadCase("Smyk", [Force("horni", [-6000.0, 0, 0])]), LoadCase("Tah", [Force("horni", [0, 0, 900.0])])]
    return st, parts


@pytest.fixture(scope="module")
def bolt_tie_elastic(tmp_path_factory):
    wd = tmp_path_factory.mktemp("bolt_tie_elastic")
    st, parts = _study_bolt_tie_elastic(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_a_bolt_with_a_tie_and_an_elastic_support_equals_ccx(bolt_tie_elastic):
    st, parts, wd, ref, res, logs, mesh = bolt_tie_elastic
    assert mesh.report.get("ties") and res.steps[-1].name == PRELOAD_CASE
    _compare(ref, res, tol_u=1e-5, tol_s=5e-5, tol_r=2e-5, tol_imbalance=5e-5)             # the imbalance of the load states, 2.0e-5, is the same in ccx (the elastic support with the block)
    assert [s.equilibrium["imbalance"] for s in res.steps] == pytest.approx([s.equilibrium["imbalance"] for s in ref.steps], rel=0.1, abs=1e-6)
    _compare_bolts(st, ref, res)
    c = res.info["solver"]["constraints"]
    assert c.get("bolts") == 1 and c.get("tie") and c.get("springs")

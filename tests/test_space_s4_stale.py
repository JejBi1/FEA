"""F14 S4 review finding C2: a stored verdict is not a fact. The report prints it only while the basis (requirement-set digest + confirmation, allowable digests, study digest)
matches the present study; otherwise 'outdated - recompute'; a record without a basis never prints PASS; bolts, PCB and rails are recomputed from their stored inputs, so a
forged PASS does not survive."""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (first the cases)
from fea_core.report.space_s4 import s4_sections  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
from fea_core.space import s4_run, s4_store, s4_verify  # noqa: E402
from fea_core.space.model import MaterialAllowable  # noqa: E402
from fea_core.space.requirements import confirm_allowable, set_allowable  # noqa: E402
from fea_core.study import Material, Study  # noqa: E402
from tests.s4_helpers import confirmed_space  # noqa: E402
from tests.test_space_s4_bolts import COEF  # noqa: E402

PCB = {"pcb_steinberg_const": 0.00022, "pcb_amplitude_factor": 1.0, "pcb_design_factor": 1.0, "pcb_fatigue_exponent": 6.4, "pcb_ref_cycles": 2.0e7, "pcb_damage_limit": 1.0,
       "pcb_n_sigma": 3.0, "pcb_component_const": 1.0, "pcb_position_factor_min": 0.25}
RAIL = {"rail_reaction_limit_N": 800.0, "rail_axial_reaction_limit_N": 1500.0, "rail_lateral_deflection_limit_mm": 0.5}
BOLT = dict(name="B1", d_mm=6.0, pitch_mm=1.0, A_s_mm2=20.1, k_b=3.5e5, k_c=4.0e5, F_nom=6000.0, embedding_um=6.0, layers=[{"t_mm": 12.0, "alpha": 23.4e-6}], alpha_bolt=8.6e-6,
            T_min=-40.0, T_max=80.0, F_ax=500.0, F_sh=100.0, engaged_mm=9.0, d_w_mm=10.0, d_h_mm=6.6, t_bearing_mm=4.0)


def _text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def _study(limits=None, units=None) -> Study:
    st = Study("review", "none.step")
    for name, fy, fu, fsu, rho in (("Ti6Al4V", 880.0, 950.0, 570.0, 4.4e-9), ("Al7075", 480.0, 540.0, 320.0, 2.8e-9)):
        st.add_material(Material(name, E=110000.0, nu=0.34, rho=rho, Re=fy, Rm=fu))
        set_allowable(st.materials[name], MaterialAllowable(basis="B", Fy=fy, Fu=fu, Fsy=0.6 * fy, Fsu=fsu, source="test data sheet"))
        confirm_allowable(st.materials[name], "test engineer")
    st.space = confirmed_space({**COEF, **PCB, **RAIL, **(limits or {})}, units={"pcb_steinberg_const": "in", **(units or {})})
    return st


def _bolt_pass(st: Study) -> dict:
    d = s4_run.bolts_report(st, [dict(BOLT)], "Ti6Al4V", "Al7075")
    assert d["bolts"][0]["verdict"] == "PASS", d["bolts"][0]["margins"]
    s4_store.record(st, "bolts", d["bolts"][0], "B1")
    return s4_store.get(st)["bolts"]["B1"]


def _doc_text(st) -> str:
    D = _Doc()
    s4_sections(D, st, "9")
    return _text(D)


def test_a_fresh_record_prints_its_verdict_and_the_basis_is_stable_over_a_save():
    st = _study()
    _bolt_pass(st)
    sh = s4_verify.shown(st, "bolts", s4_store.get(st)["bolts"]["B1"])
    assert sh.verdict == "PASS" and sh.fresh
    assert "Bolt B1: verdict PASS" in _doc_text(st)
    back = Study.from_dict(json.loads(json.dumps(st.to_dict())))                     # saved and loaded: the digests are those of the same state
    assert s4_verify.shown(back, "bolts", s4_store.get(back)["bolts"]["B1"]).verdict == "PASS"
    assert s4_store.basis_of(back) == s4_store.basis_of(st)


def test_a_changed_limit_makes_the_stored_verdict_outdated_not_pass():
    st = _study()
    _bolt_pass(st)
    for x in st.space.requirement_set.limits:
        if x.key == "bolt_factor_gapping":
            x.value = 99.0
    assert not st.space.requirement_confirmed()
    sh = s4_verify.shown(st, "bolts", s4_store.get(st)["bolts"]["B1"])
    assert sh.verdict == "OUTDATED" and "requirement set changed" in sh.text and not sh.fresh
    text = _doc_text(st)
    assert "outdated - recompute" in text and "verdict PASS" not in text and "PASS" not in "\n".join(l for l in text.splitlines() if "Bolt B1" in l or "gapping" in l)


def test_a_limit_changed_back_to_the_same_value_is_fresh_again_only_when_it_is_confirmed():
    st = _study()
    _bolt_pass(st)
    rs = st.space.requirement_set
    x = next(y for y in rs.limits if y.key == "bolt_factor_gapping")
    old = x.value
    x.value = 5.0
    x.value = old                                                                       # same content, but the confirmation was withdrawn by the edit? (digest again equal)
    sh = s4_verify.shown(st, "bolts", s4_store.get(st)["bolts"]["B1"])
    assert sh.verdict in ("PASS", "OUTDATED")
    if st.space.requirement_confirmed():
        assert sh.verdict == "PASS"
    else:
        assert sh.verdict == "OUTDATED"                                                  # the confirmation is part of the basis


def test_a_changed_study_or_allowable_makes_it_outdated():
    st = _study()
    _bolt_pass(st)
    rec = s4_store.get(st)["bolts"]["B1"]
    st2 = copy.deepcopy(st)
    from fea_core.study import Force, LoadCase
    st2.load_cases.append(LoadCase("new", [Force("none", [1.0, 0.0, 0.0])]))
    sh = s4_verify.shown(st2, "bolts", rec)
    assert sh.verdict == "OUTDATED" and "study changed" in sh.text
    st3 = copy.deepcopy(st)
    st3.materials["Al7075"].allowables["Fsu"] = 100.0                                    # an allowable edited after the calculation
    sh3 = s4_verify.shown(st3, "bolts", rec)
    assert sh3.verdict == "OUTDATED" and "allowables" in sh3.text


def test_a_forged_record_without_a_basis_is_never_pass():
    st = _study()
    d = Study("forged", "none.step").to_dict()
    d["space"] = {"s4": {"bolts": {"M6 flange": {"verdict": "PASS", "margins": [{"check": "gapping", "demand": 1, "capacity": 9, "design_factor": 1.0, "mos": 8.0, "status": "PASS",
                                                                               "formula": "x", "unconfirmed": []}], "inputs": []}},
                         "pcb": {"U1": {"verdict": "PASS", "MoS": 3.0}}, "rails": {"verdict": "PASS", "verdict_text": "all good", "supports": [], "deflection": []},
                         "notch": {"sine x force": {"verdict": "PASS", "verdict_text": "ok", "notches": [], "inputs": []}}}}
    forged = Study.from_dict(json.loads(json.dumps(d)))
    text = _doc_text(forged)
    assert "PASS" not in text and "UNCONFIRMED_REQUIREMENTS" in text and "no basis" in text
    for sec, key in (("bolts", "M6 flange"), ("pcb", "U1"), ("notch", "sine x force")):
        sh = s4_verify.shown(forged, sec, s4_store.get(forged)[sec][key])
        assert sh.verdict == "UNCONFIRMED_REQUIREMENTS" and not sh.fresh
    assert s4_verify.shown(forged, "rails", s4_store.get(forged)["rails"]).verdict == "UNCONFIRMED_REQUIREMENTS"
    # the same record on a study with a CONFIRMED set: still no PASS without a basis
    st.space.extra["s4"] = copy.deepcopy(forged.space.extra["s4"])
    assert "PASS" not in _doc_text(st)


def test_a_forged_record_with_a_copied_basis_is_recomputed_from_its_stored_inputs():
    st = _study()
    good = _bolt_pass(st)
    # a bolt that really FAILS (the separating force is far too large) but carries a PASS verdict and the copied basis of the present state
    forged = copy.deepcopy(good)
    forged["spec"]["raw"] = {**BOLT, "F_ax": 20000.0}
    assert forged["verdict"] == "PASS"
    st.space.extra["s4"]["bolts"]["B1"] = forged
    sh = s4_verify.shown(st, "bolts", forged)
    assert sh.fresh and sh.verdict == "FAIL" and "recomputed from the stored inputs" in sh.text
    assert "verdict PASS" not in _doc_text(st) and "verdict FAIL" in _doc_text(st)
    # a forged PCB record the same way
    pr = s4_run.pcb_report(st, [dict(name="U1", B_mm=127.0, h_mm=1.575, L_mm=25.4, r=1.0, z_rms_mm=0.001, r_source="test drawing", z_source="test analysis")])["components"][0]
    assert pr["verdict"] == "PASS"
    s4_store.record(st, "pcb", pr, "U1")
    fp = copy.deepcopy(s4_store.get(st)["pcb"]["U1"])
    fp["spec"]["z_rms_mm"] = 5.0
    st.space.extra["s4"]["pcb"]["U1"] = fp
    assert s4_verify.shown(st, "pcb", fp).verdict == "FAIL"


def test_a_rails_record_is_rechecked_against_the_present_limits():
    st = _study()
    rec = {"verdict": "PASS", "verdict_text": "x", "inputs": [], "warnings": [], "deflection": [{"part": "R1", "max_lateral_deflection_mm": 0.25, "state": "L", "limit_mm": 0.5, "mos": 1.0}],
           "supports": [{"support": "Rail R1 +y", "part": "R1", "kind": "wall", "state": "L", "reaction_N": 400.0, "effective_N": 400.0, "pulls": False, "limit_N": 800.0, "mos": 1.0,
                         "pull_states": []}]}
    s4_store.record(st, "rails", rec)
    assert s4_verify.shown(st, "rails", s4_store.get(st)["rails"]).verdict == "PASS"
    forged = copy.deepcopy(s4_store.get(st)["rails"])
    forged["supports"][0]["effective_N"] = 5000.0                                         # the stored force is above the limit, the stored verdict says PASS
    assert s4_verify.shown(st, "rails", forged).verdict == "FAIL"
    forged2 = copy.deepcopy(s4_store.get(st)["rails"])
    st.space.requirement_set.limits = [x for x in st.space.requirement_set.limits if x.key != "rail_reaction_limit_N"]
    assert s4_verify.shown(st, "rails", forged2).verdict == "OUTDATED"                    # the set changed


def test_the_report_computes_the_present_basis_once(monkeypatch):
    st = _study()
    _bolt_pass(st)
    s4_store.record(st, "bolts", {**s4_store.get(st)["bolts"]["B1"]}, "B2")
    calls = []
    real = s4_store.basis_of
    monkeypatch.setattr(s4_store, "basis_of", lambda study: calls.append(1) or real(study))
    text = _doc_text(st)
    assert "Bolt B1: verdict PASS" in text and "Bolt B2: verdict PASS" in text
    assert len(calls) == 1                                                            # one basis for the whole report, not one per record
    assert s4_verify._CTX.get() is None                                                       # and the context is left clean
    calls.clear()
    s4_verify.shown(st, "bolts", s4_store.get(st)["bolts"]["B1"])
    assert len(calls) == 1

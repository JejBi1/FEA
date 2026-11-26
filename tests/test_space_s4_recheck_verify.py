"""F14 S4 re-check regressions: the report prints the RECOMPUTED rows (and says when the stored result was not reproduced), a record whose stored inputs are gone gets no verdict,
notch / interface records say they are not recomputed, bolt input checks, clean errors for malformed items, structural damping, the interface record without a basis."""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (first the cases)
from fea_core.report.space_s4 import s4_sections  # noqa: E402
from fea_core.report.word import _Doc, num  # noqa: E402
from fea_core.space import bolts_ecss as BE  # noqa: E402
from fea_core.space import s4_run, s4_store, s4_verify  # noqa: E402
from fea_core.space.s4_limits import S4Limits  # noqa: E402
from fea_core.study import Study  # noqa: E402
from tests.s4_helpers import confirmed_space, sdof_interface  # noqa: E402
from tests.test_space_s4_bolts import COEF, _bolt, _lim  # noqa: E402
from tests.test_space_s4_stale import BOLT, PCB, _bolt_pass, _doc_text, _study, _text  # noqa: E402

NOT_RECOMPUTED = "stored result, not recomputed - tamper-evident only (staleness), not a signature"


def _doc(st):
    D = _Doc()
    s4_sections(D, st, "9")
    return D


# ------------------------------------------------------------------ 5: the report prints the recomputed rows
def test_a_forged_bolt_prints_the_recomputed_rows_not_the_stored_numbers():
    st = _study()
    good = _bolt_pass(st)
    stored_gap = next(m for m in good["margins"] if m["check"] == "gapping")
    forged = copy.deepcopy(good)
    forged["spec"]["raw"] = {**BOLT, "F_ax": 20000.0}                          # the inputs say FAIL, the stored margins still say PASS
    st.space.extra["s4"]["bolts"]["B1"] = forged
    sh = s4_verify.shown(st, "bolts", forged)
    assert sh.verdict == "FAIL" and sh.reproduced is False and sh.data is not None and sh.fresh
    again_gap = next(m for m in sh.data["margins"] if m["check"] == "gapping")
    text = _doc_text(st)
    assert "verdict FAIL" in text and "the stored result was not reproduced - recomputed values shown" in text
    assert num(again_gap["mos"], 3) in text and num(stored_gap["mos"], 3) not in text          # the recomputed margin is printed, the stored one is not
    assert any(num(again_gap["demand"]) == c for c in text.splitlines())


def test_a_stored_margin_edited_by_hand_is_replaced_by_the_recomputed_one():
    st = _study()
    good = _bolt_pass(st)
    forged = copy.deepcopy(good)
    for m in forged["margins"]:
        if m["mos"] is not None:
            m["mos"] = 8123.0                                                  # real inputs, a hand-edited margin
    st.space.extra["s4"]["bolts"]["B1"] = forged
    sh = s4_verify.shown(st, "bolts", forged)
    assert sh.verdict == "PASS" and sh.reproduced is False                      # the verdict is the recomputed one (the same), the numbers are not those stored
    text = _doc_text(st)
    assert num(8123.0) not in text and "not reproduced" in text


def test_a_program_update_that_changes_a_number_is_reported_not_hidden(monkeypatch):
    st = _study()
    _bolt_pass(st)
    rec = s4_store.get(st)["bolts"]["B1"]
    assert s4_verify.shown(st, "bolts", rec).reproduced is True
    assert "not reproduced" not in _doc_text(st)
    monkeypatch.setattr(BE, "embedding_loss", lambda f_emb_mm, k_b, k_c: 0.0)    # "a later version of the program computes the embedding loss differently"
    sh = s4_verify.shown(st, "bolts", rec)
    assert sh.reproduced is False and sh.verdict == "PASS"                      # the verdict happens to be the same, the numbers are not
    text = _doc_text(st)
    assert "the stored result was not reproduced - recomputed values shown" in text
    assert num(sh.data["F_pmin0_N"]) != num(rec["F_pmin0_N"]) or num(sh.data["F_pmin_N"]) != num(rec["F_pmin_N"])


def test_a_pcb_record_is_recomputed_and_its_rows_are_the_recomputed_ones():
    st = _study()
    pr = s4_run.pcb_report(st, [dict(name="U1", B_mm=127.0, h_mm=1.575, L_mm=25.4, r=1.0, z_rms_mm=0.001, r_source="drawing", z_source="analysis")])["components"][0]
    assert pr["verdict"] == "PASS"
    s4_store.record(st, "pcb", pr, "U1")
    fp = copy.deepcopy(s4_store.get(st)["pcb"]["U1"])
    fp["spec"]["z_rms_mm"] = 5.0
    st.space.extra["s4"]["pcb"]["U1"] = fp
    sh = s4_verify.shown(st, "pcb", fp)
    assert sh.verdict == "FAIL" and sh.reproduced is False and sh.data["Z_sigma_mm"] == pytest.approx(5.0)
    text = _doc_text(st)
    assert "the stored result was not reproduced" in text and num(5.0) in text and num(0.001) not in text.replace("0,001", "")  # the stored sigma is not printed any more


def test_no_confirmed_no_row_under_a_pass():
    st = _study()
    _bolt_pass(st)
    D = _doc(st)
    flat = [c.text for t in D.doc.tables for row in t.rows for c in row.cells]
    assert "NO" not in flat and "PASS" in flat                                  # a PASS bolt lists no "Confirmed = NO" input; plain geometry data say "data"
    assert "data" in flat


# ------------------------------------------------------------------ 6: tamper evidence
def test_a_record_with_its_spec_deleted_gets_no_verdict():
    st = _study()
    good = _bolt_pass(st)
    forged = copy.deepcopy(good)
    forged.pop("spec")
    st.space.extra["s4"]["bolts"]["B1"] = forged
    sh = s4_verify.shown(st, "bolts", forged)
    assert sh.verdict == "UNCONFIRMED_REQUIREMENTS" and "cannot be recomputed" in sh.text and not sh.fresh
    text = _doc_text(st)
    assert "verdict PASS" not in text and "cannot be recomputed" in text
    forged2 = copy.deepcopy(good)
    forged2["spec"] = {"bolt_material": "Ti6Al4V"}                              # a spec without the raw inputs
    assert s4_verify.shown(st, "bolts", forged2).verdict == "UNCONFIRMED_REQUIREMENTS"
    # the same for a PCB record
    pr = s4_run.pcb_report(st, [dict(name="U1", B_mm=127.0, h_mm=1.575, L_mm=25.4, r=1.0, z_rms_mm=0.001, r_source="drawing", z_source="analysis")])["components"][0]
    s4_store.record(st, "pcb", pr, "U1")
    fp = copy.deepcopy(s4_store.get(st)["pcb"]["U1"])
    fp.pop("spec")
    sh = s4_verify.shown(st, "pcb", fp)
    assert sh.verdict == "UNCONFIRMED_REQUIREMENTS" and "cannot be recomputed" in sh.text


def _rails_rec(**kw):
    rec = {"verdict": "PASS", "verdict_text": "x", "inputs": [], "warnings": [], "deflection": [{"part": "R1", "max_lateral_deflection_mm": 0.25, "state": "L", "limit_mm": 0.5, "mos": 1.0}],
           "supports": [{"support": "Rail R1 +y", "part": "R1", "kind": "wall", "state": "L", "reaction_N": 400.0, "effective_N": 400.0, "pulls": False, "limit_N": 800.0, "mos": 1.0,
                         "pull_states": []}]}
    rec.update(kw)
    return rec


def test_a_rails_record_without_effective_force_or_without_rows_gets_no_verdict():
    st = _study()
    s4_store.record(st, "rails", _rails_rec())
    assert s4_verify.shown(st, "rails", s4_store.get(st)["rails"]).verdict == "PASS"
    rec = _rails_rec()
    del rec["supports"][0]["effective_N"]                                      # the forced PASS of the re-check: a stored reaction of 5000 N, no effective force
    rec["supports"][0]["reaction_N"] = 5000.0
    s4_store.record(st, "rails", rec)
    sh = s4_verify.shown(st, "rails", s4_store.get(st)["rails"])
    assert sh.verdict == "UNCONFIRMED_REQUIREMENTS" and "cannot be recomputed" in sh.text
    s4_store.record(st, "rails", _rails_rec(supports=[], deflection=[]))
    assert s4_verify.shown(st, "rails", s4_store.get(st)["rails"]).verdict == "UNCONFIRMED_REQUIREMENTS"
    assert "verdict PASS" not in _doc_text(st)
    bad = _rails_rec()
    bad["supports"][0]["effective_N"] = "x"
    s4_store.record(st, "rails", bad)
    assert s4_verify.shown(st, "rails", s4_store.get(st)["rails"]).verdict == "UNCONFIRMED_REQUIREMENTS"


def test_rails_margins_are_recomputed_and_printed_against_the_present_limits():
    st = _study()
    rec = _rails_rec()
    rec["supports"][0].update(effective_N=1000.0, mos=0.5, limit_N=1500.0)       # stored: limit 1500, margin +0.5 -> PASS; the present limit is 800 N
    s4_store.record(st, "rails", rec)
    sh = s4_verify.shown(st, "rails", s4_store.get(st)["rails"])
    assert sh.verdict == "FAIL" and sh.reproduced is False and sh.data["supports"][0]["limit_N"] == 800.0 and sh.data["supports"][0]["mos"] == pytest.approx(-0.2)
    text = _doc_text(st)
    assert "verdict FAIL" not in text and "Verdict: FAIL" in text and "1500" not in text and "not reproduced" in text
    assert "The forces of the supports are stored values" in text


def test_notch_and_interface_records_say_that_they_are_not_recomputed():
    st = _study()
    s4_store.record(st, "notch", {"verdict": "PASS", "verdict_text": "the notched profile keeps the limit with the confirmed rules", "notches": [], "inputs": [], "warnings": []},
                    "sine x force")
    s4_store.record(st, "interface", {"total_mass_kg": 1.0, "ref_point_mm": [0, 0, 0], "modes": 1, "engine": "x", "damping": "x", "static_mass_check": {}, "modal_table": [],
                                      "directions": {}, "warnings": []})
    sh = s4_verify.shown(st, "notch", s4_store.get(st)["notch"]["sine x force"])
    assert sh.verdict == "PASS" and sh.note == NOT_RECOMPUTED and sh.data is None and sh.reproduced is None
    text = _doc_text(st)
    assert text.count("Stored result, not recomputed - tamper-evident only (staleness), not a signature.") == 2             # next to the notch verdict AND in the interface section
    assert "forged PASS does not survive" not in (s4_verify.__doc__ or "") and "tamper-evident" in (s4_verify.__doc__ or "")


def test_the_report_discloses_the_tolerance_of_the_table_check_and_the_achieved_ratio():
    st = _study()
    s4_store.record(st, "notch", {"verdict": "PASS", "verdict_text": "ok", "notches": [], "inputs": [], "warnings": [], "table_ratio": 0.99987, "table_tolerance": 1e-3},
                    "sine x force")
    text = _doc_text(st)
    assert "Check of the exported breakpoint table between the analysis frequencies: J x response / limit = 0,99987" in text.replace(".", ",")
    assert "accepted up to 1 + 0,001" in text.replace(".", ",")


def test_an_interface_record_without_a_basis_has_a_remark_in_the_report():
    st = _study()
    st.space.extra["s4"] = {"interface": {"total_mass_kg": 1.0, "ref_point_mm": [0, 0, 0], "modes": 1, "engine": "x", "damping": "x", "static_mass_check": {}, "modal_table": [],
                                          "directions": {}, "warnings": []}}
    assert "carries no basis" in _doc_text(st)


# ------------------------------------------------------------------ minor: bolt inputs
def test_grip_length_and_engaged_length_must_be_positive_and_a_grip_mismatch_is_a_warning():
    for kw in ({"L_grip_mm": 0.0}, {"L_grip_mm": -3.0}, {"engaged_mm": 0.0}, {"engaged_mm": -3.0}):
        with pytest.raises(BE.BoltsError):
            BE.assess_bolt(_bolt(**kw), _lim())
    ok = BE.assess_bolt(_bolt(), _lim())                                       # L_grip = 12 = 8 + 4
    assert not any("L_grip" in w for w in ok.warnings)
    off = BE.assess_bolt(_bolt(L_grip_mm=15.0), _lim())
    assert any("L_grip = 15 mm differs from the sum of the layers (12 mm) by 25.0 %" in w for w in off.warnings)
    near = BE.assess_bolt(_bolt(L_grip_mm=12.3), _lim())                       # 2.5 %: inside the tolerance
    assert not any("L_grip" in w for w in near.warnings)
    with pytest.raises(s4_run.S4Error, match="positive"):
        s4_run.bolts_report(Study("t", "x.step"), [dict(name="x", d_mm=6, A_s_mm2=20.1, k_b=1e5, k_c=1e5, F_nom=1000.0, L_grip_mm=-1.0)])


def test_more_than_two_friction_interfaces_need_a_confirmed_requirement_row():
    r2 = BE.assess_bolt(_bolt(n_slip_planes=2), _lim())
    assert next(m for m in r2.margins if m.check == "slip").status in (BE.PASS, BE.FAIL)
    r3 = BE.assess_bolt(_bolt(n_slip_planes=3), _lim())
    slip = next(m for m in r3.margins if m.check == "slip")
    assert slip.status == BE.UNCONFIRMED and any("bolt_slip_planes_max" in u for u in slip.unconfirmed) and r3.verdict == BE.UNCONFIRMED
    with_row = S4Limits.from_requirement_set(confirmed_space({**COEF, "bolt_slip_planes_max": 3.0}))
    assert next(m for m in BE.assess_bolt(_bolt(n_slip_planes=3), with_row).margins if m.check == "slip").status in (BE.PASS, BE.FAIL)
    too_many = BE.assess_bolt(_bolt(n_slip_planes=4), with_row)                  # the row allows 3
    assert next(m for m in too_many.margins if m.check == "slip").status == BE.UNCONFIRMED
    unconfirmed_row = S4Limits.from_requirement_set(confirmed_space({**COEF, "bolt_slip_planes_max": 3.0}, confirm_it=False))
    assert next(m for m in BE.assess_bolt(_bolt(n_slip_planes=3), unconfirmed_row).margins if m.check == "slip").status == BE.UNCONFIRMED
    from fea_core.space import s4_limits
    assert "bolt_slip_planes_max" in s4_limits.LIMIT_DESCRIPTIONS


# ------------------------------------------------------------------ minor: raw exceptions
@pytest.mark.parametrize("bad", [[1, 2], "abc", [None], [[1, 2]], [{"name": "x"}, 3]])
def test_a_malformed_list_item_is_a_clean_error(bad):
    st = Study("t", "x.step")
    with pytest.raises(s4_run.S4Error):
        s4_run.bolts_report(st, bad)
    with pytest.raises(s4_run.S4Error):
        s4_run.pcb_report(st, bad)


def test_a_single_dictionary_is_accepted_as_a_list_of_one():
    st = _study()
    d = s4_run.bolts_report(st, dict(BOLT), "Ti6Al4V", "Al7075")
    assert len(d["bolts"]) == 1
    p = s4_run.pcb_report(st, dict(name="U1", B_mm=127.0, h_mm=1.575, L_mm=25.4, r=1.0, z_rms_mm=0.001, r_source="d", z_source="a"))
    assert len(p["components"]) == 1


# ------------------------------------------------------------------ minor: structural damping
def test_structural_damping_is_documented_as_the_equivalent_viscous_ratio():
    st = Study("t", "x.step")
    st.dynamics.damping.kind = "structural"
    st.dynamics.damping.eta = 0.04
    z, src = s4_run.damping_of(st, sdof_interface(), None)
    assert z[0] == pytest.approx(0.02) and "structural" in src and "eta/2" in src and "0.04" in src
    st.dynamics.damping.kind = "constant"
    z2, src2 = s4_run.damping_of(st, sdof_interface(), None)
    assert "eta/2" not in src2 and z2[0] == pytest.approx(st.dynamics.damping.zeta)


def test_the_pcb_fixture_limits_are_the_ones_of_the_stale_tests():
    assert "pcb_steinberg_const" in PCB and _text is not None

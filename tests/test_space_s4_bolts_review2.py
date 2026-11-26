"""F14 S4, second ECSS review of the bolt margins (findings 1 - 10 of the review of ECSS-E-HB-32-23A Rev.1 / ECSS-E-ST-32-10C):
F_K,req without a separating force fails, the thermally induced bolt load carries a factor (`bolt_thermal_factor`, default = the J of the check), yield margins of crushing and
hole bearing, a missing bolt Fsy is UNCONFIRMED, the lowest nominal preload `F_nom_min`, the nut ratio s_w/d, the deviations are shown, stored records of an older bolt model.
All numbers are ARBITRARY TEST VALUES."""
from __future__ import annotations

import pytest

from fea_core.report import space_s4 as RS
from fea_core.space import bolts_ecss as BE
from fea_core.space import s4_run
from fea_core.space.s4_limits import OBSOLETE_LIMITS, S4Limits
from fea_core.study import Study
from tests.s4_helpers import confirmed_space
from tests.test_space_s4_bolts import A_AL, A_STEEL, ALLOW, COEF, KB, KC, KSER, L, T_MAX, T_MIN, T_REF, _bolt, _lim

PHI = KB / (KB + KC)
DIFF = (A_AL - A_STEEL) * L * KSER                    # dF_T per kelvin of the aluminium stack on a steel bolt
F_EMB = 0.006 * KSER


def _lim_with(**kw) -> S4Limits:
    return S4Limits.from_requirement_set(confirmed_space({**COEF, **kw}))


def _m(r, name):
    return next(x for x in r.margins if x.check == name)


# ------------------------------------------------------------------ 1. F_K,req without a separating force
def test_required_clamp_force_without_an_axial_force_fails_when_the_preload_is_too_low():
    ok = BE.assess_bolt(_bolt(F_ax=0.0, F_K_req=1000.0, F_sh=0.0), _lim())
    g = _m(ok, "gapping")
    f_pmin_j = 6000.0 * 0.9 - F_EMB - 0.05 * 6000.0 + 1.2 * DIFF * (T_MIN - T_REF)     # F_V,min with the cold loss carrying J_T = 1,2
    assert g.status == BE.PASS and g.capacity == pytest.approx(f_pmin_j, rel=1e-12) and g.mos == pytest.approx(f_pmin_j / 1000.0 - 1.0, rel=1e-12)
    bad = BE.assess_bolt(_bolt(F_ax=0.0, F_K_req=f_pmin_j + 1.0, F_sh=0.0), _lim())
    gb = _m(bad, "gapping")
    assert gb.status == BE.FAIL and gb.mos < 0 and bad.verdict == BE.FAIL
    assert gb.mos == pytest.approx(f_pmin_j / (f_pmin_j + 1.0) - 1.0, rel=1e-12)
    assert any("below the required clamp force" in w for w in bad.warnings)
    # with a separating force the same bolt fails through eq. 7.8.1 as before; without a required clamp force there is no gapping margin at all
    assert _m(BE.assess_bolt(_bolt(F_ax=0.0, F_sh=0.0), _lim()), "gapping").status == "n/a"
    # unconfirmed inputs never give a FAIL verdict out of this margin (the status follows the rule of all margins)
    unc = BE.assess_bolt(_bolt(F_ax=0.0, F_K_req=1e5, F_sh=0.0), S4Limits.from_requirement_set(confirmed_space(COEF, confirm_it=False)))
    assert _m(unc, "gapping").status == BE.UNCONFIRMED


class _Doc:
    """A minimal report writer: it records what a section prints."""
    def __init__(self):
        self.out: list[str] = []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.out.append(" ".join(str(x) for x in a if isinstance(x, str)))
        return rec


def test_report_prints_the_warnings_and_the_deviations_of_the_bolt():
    st = Study("review", "none.step")
    st.space = confirmed_space(COEF)
    raw = dict(name="B", d_mm=6, pitch_mm=1.0, A_s_mm2=20.1, k_b=3.5e5, k_c=4.0e5, F_nom=3000, F_ax=0.0, F_K_req=5000.0, embedding_um=6.0, T_min=-40.0, T_max=80.0, alpha_bolt=A_STEEL,
               layers=[{"t_mm": 12.0, "alpha": A_AL}], allowables={"bolt_Ftu": 800.0})
    d = s4_run.bolts_report(st, [raw])
    r = d["bolts"][0]
    assert r["verdict"] != "PASS" and any("below the required clamp force" in w for w in r["warnings"])
    doc = _Doc()
    RS.bolts_section(doc, {"B": r}, "6.1", st)
    text = "\n".join(doc.out)
    assert "below the required clamp force" in text and "Notes on bolt B" in text
    assert d["deviations"] and all(x in text for x in d["deviations"][:2])


# ------------------------------------------------------------------ 2. the thermally induced bolt load carries a factor
def test_thermal_increment_carries_the_factor_of_the_check_by_default():
    r = BE.assess_bolt(_bolt(), _lim())
    cold, hot = DIFF * (T_MIN - T_REF), DIFF * (T_MAX - T_REF)
    assert r.F_pmin == pytest.approx(r.F_pmin0 + cold) and r.F_pmax == pytest.approx(r.F_pmax0 + hot)         # the shown range has J_T = 1
    ext = PHI * 3000.0
    assert _m(r, "gapping").capacity == pytest.approx(r.F_pmin0 + 1.2 * cold, rel=1e-12)
    assert _m(r, "tensile_yield").mos == pytest.approx(_m(r, "tensile_yield").capacity / (r.F_pmax0 + 1.1 * hot + 1.1 * ext) - 1.0, rel=1e-12)
    assert _m(r, "tensile_ultimate").mos == pytest.approx(_m(r, "tensile_ultimate").capacity / (r.F_pmax0 + 1.4 * hot + 1.4 * ext) - 1.0, rel=1e-12)
    # with J_T = 1 (the row of the requirement set, e.g. an expendable launch vehicle, Table 4-4 note e) the increment is not factored
    one = BE.assess_bolt(_bolt(), _lim_with(bolt_thermal_factor=1.0))
    assert _m(one, "gapping").capacity == pytest.approx(one.F_pmin0 + cold, rel=1e-12)
    assert _m(one, "tensile_ultimate").mos == pytest.approx(_m(one, "tensile_ultimate").capacity / (one.F_pmax0 + hot + 1.4 * ext) - 1.0, rel=1e-12)
    assert _m(one, "tensile_ultimate").mos > _m(r, "tensile_ultimate").mos and _m(one, "gapping").mos > _m(r, "gapping").mos
    two = BE.assess_bolt(_bolt(), _lim_with(bolt_thermal_factor=2.0))
    assert _m(two, "tensile_yield").mos == pytest.approx(_m(two, "tensile_yield").capacity / (two.F_pmax0 + 2.0 * hot + 1.1 * ext) - 1.0, rel=1e-12)
    # slip: the cold loss carries J_T too; combined and pull-out likewise
    assert _m(r, "slip").mos == pytest.approx(0.2 * (r.F_pmin0 + 1.25 * cold - (1 - PHI) * 3000.0) / (1.25 * 800.0) - 1.0, rel=1e-12)
    assert "J (thermal)" in " ".join(i["name"] for i in two.inputs)


@pytest.mark.parametrize("t_min,t_max", [(-40.0, -10.0), (40.0, 80.0), (20.0, 80.0), (-40.0, 20.0)])
def test_both_operating_temperatures_on_one_side_of_the_assembly_temperature(t_min, t_max):
    r = BE.assess_bolt(_bolt(T_min=t_min, T_max=t_max), _lim())
    d_lo, d_hi = DIFF * (t_min - T_REF), DIFF * (t_max - T_REF)
    loss, gain = min(0.0, d_lo, d_hi), max(0.0, d_lo, d_hi)
    assert r.F_pmin == pytest.approx(r.F_pmin0 + loss) and r.F_pmax == pytest.approx(r.F_pmax0 + gain)
    assert _m(r, "gapping").capacity == pytest.approx(r.F_pmin0 + 1.2 * loss, rel=1e-12)
    assert _m(r, "tensile_ultimate").mos == pytest.approx(_m(r, "tensile_ultimate").capacity / (r.F_pmax0 + 1.4 * gain + 1.4 * PHI * 3000.0) - 1.0, rel=1e-12)
    if t_min >= T_REF:                                   # both hot: nothing lost, the cold-side margin does not move with J_T
        assert loss == 0.0 and _m(r, "gapping").capacity == pytest.approx(r.F_pmin0)
    if t_max <= T_REF:                                   # both cold: no gain, the tensile margin has no thermal term
        assert gain == 0.0 and r.F_pmax == pytest.approx(r.F_pmax0)


def test_thermal_factor_validation_confirmation_and_direct_input():
    with pytest.raises(BE.BoltsError):
        BE.assess_bolt(_bolt(), _lim_with(bolt_thermal_factor=0.0))
    with pytest.raises(BE.BoltsError):
        BE.assess_bolt(_bolt(factors={"thermal": -1.0}), _lim())
    direct = BE.assess_bolt(_bolt(factors={"thermal": 1.0}), _lim())                         # a typed J_T is used but never confirmed
    assert direct.verdict == BE.UNCONFIRMED and any("thermal" in u for m in direct.margins for u in m.unconfirmed)
    assert _m(direct, "gapping").capacity == pytest.approx(direct.F_pmin, rel=1e-12)
    unc = BE.assess_bolt(_bolt(), S4Limits.from_requirement_set(confirmed_space({**COEF, "bolt_thermal_factor": 1.0}, confirm_it=False)))
    assert unc.verdict == BE.UNCONFIRMED
    # a requirement set without the row stays complete: the row is optional
    assert "bolt_thermal_factor" not in s4_run.BOLT_LIMIT_KEYS and BE.assess_bolt(_bolt(), _lim()).verdict in (BE.PASS, BE.FAIL)


# ------------------------------------------------------------------ 4. yield margins of crushing and hole bearing
def test_yield_margins_of_crushing_and_hole_bearing():
    r = BE.assess_bolt(_bolt(), _lim())
    hot = DIFF * (T_MAX - T_REF)
    a_br = 3.141592653589793 / 4 * (10.0 ** 2 - 6.6 ** 2)
    h = _m(r, "bearing_head_yield")
    assert h.mos == pytest.approx(300.0 * a_br / (r.F_pmax0 + 1.1 * hot + 1.1 * PHI * 3000.0) - 1.0, rel=1e-12) and h.status in (BE.PASS, BE.FAIL)
    o = _m(r, "bearing_hole_yield")
    assert o.mos == pytest.approx(300.0 * 6.0 * 4.0 / (1.1 * 800.0) - 1.0, rel=1e-12)
    # a weak part: the yield margin fails although the ultimate one holds
    al = dict(ALLOW)
    al["part_Fy"] = BE.Allowable(60.0, True, "test")
    weak = BE.assess_bolt(_bolt(allow=al), _lim())
    assert _m(weak, "bearing_head_yield").mos < 0 and _m(weak, "bearing_head").mos > 0 and weak.verdict == BE.FAIL
    # no yield strength of the part: UNCONFIRMED, never PASS
    del al["part_Fy"]
    no = BE.assess_bolt(_bolt(allow=al), _lim())
    assert _m(no, "bearing_head_yield").status == BE.UNCONFIRMED and _m(no, "bearing_hole_yield").status == BE.UNCONFIRMED and no.verdict == BE.UNCONFIRMED


def test_material_allowables_give_the_yield_strength_of_the_part():
    from verification import space_cases as SC
    st = SC.block_study()
    SC.confirm_test_allowables(st, fy=250.0, fu=330.0)
    al = s4_run.material_allowables(st, None, st.default_material)
    assert al["part_Fy"].value == 250.0 and al["part_Fy"].verified and al["part_Fbr"].value == 330.0
    assert s4_run.material_allowables(st, None, st.default_material, "Fy")["part_Fbr"].value == 250.0


# ------------------------------------------------------------------ 5. no shear yield allowable of the bolt
def test_missing_bolt_shear_yield_allowable_is_unconfirmed_not_pass():
    al = dict(ALLOW)
    del al["bolt_Fsy"]
    r = BE.assess_bolt(_bolt(allow=al), _lim())
    assert _m(r, "shear_yield").status == BE.UNCONFIRMED and _m(r, "combined_yield").status == BE.UNCONFIRMED
    assert r.verdict == BE.UNCONFIRMED and any("allowable bolt_Fsy missing" in u for u in _m(r, "shear_yield").unconfirmed)
    assert any("bolt_Fsy" in w and "no PASS" in w for w in r.warnings)
    # without a shear force there is nothing to assess: the missing Fsy is of no consequence
    none = BE.assess_bolt(_bolt(allow=al, F_sh=0.0), _lim())
    assert none.verdict != BE.UNCONFIRMED and not [m for m in none.margins if m.check in ("shear_yield", "combined_yield")]


# ------------------------------------------------------------------ 6. prevailing torque: the lowest nominal preload
def test_lowest_nominal_preload_lowers_the_minimum_preload_only():
    ref = BE.assess_bolt(_bolt(), _lim())
    low = BE.assess_bolt(_bolt(F_nom_min=5000.0), _lim())
    assert low.F_pmax0 == pytest.approx(ref.F_pmax0)
    assert low.F_pmin0 == pytest.approx(5000.0 * 0.9 - F_EMB - 0.05 * 6000.0, rel=1e-12)           # F_nom,min (1 - u) - F_Z - rho F_nom (6.3.15, 6.3.18)
    assert _m(low, "gapping").mos < _m(ref, "gapping").mos and _m(low, "tensile_ultimate").mos == pytest.approx(_m(ref, "tensile_ultimate").mos, rel=1e-12)
    assert any(i["name"].startswith("lowest nominal preload") for i in low.inputs)
    assert BE.assess_bolt(_bolt(F_nom_min=6000.0), _lim()).F_pmin0 == pytest.approx(ref.F_pmin0)
    for bad in (0.0, -10.0, 6001.0, float("nan")):
        with pytest.raises(BE.BoltsError):
            BE.assess_bolt(_bolt(F_nom_min=bad), _lim())
    st = Study("review", "none.step")
    st.space = confirmed_space(COEF)
    out = s4_run.bolts_report(st, [dict(name="B", d_mm=6, A_s_mm2=20.1, k_b=3.5e5, k_c=1.2e6, F_nom=6000, F_nom_min=5000, F_ax=1000)])
    assert out["bolts"][0]["F_pmin0_N"] < 6000.0 * 0.9


# ------------------------------------------------------------------ 7. the ratio s_w/d of the nut (7.10.2)
@pytest.mark.parametrize("wrench,ok", [(10.2, True), (8.4, True), (11.4, True), (7.0, False), (8.0, False), (12.0, False)])
def test_nut_ratio_outside_the_range_of_c1_leaves_the_pull_out_unconfirmed(wrench, ok):
    r = BE.assess_bolt(_bolt(wrench_mm=wrench), _lim())
    m = _m(r, "pullout")
    assert m.mos is not None
    if ok:
        assert m.status in (BE.PASS, BE.FAIL) and not any("s_w/d" in u for u in m.unconfirmed)
    else:
        assert m.status == BE.UNCONFIRMED and any("s_w/d" in u and "1.4 - 1.9" in u for u in m.unconfirmed) and r.verdict == BE.UNCONFIRMED
    assert _m(BE.assess_bolt(_bolt(), _lim()), "pullout").status in (BE.PASS, BE.FAIL)           # a tapped hole has no ratio


def test_vdi2230_stripping_is_not_changed_for_its_other_callers():
    from fea_core.joints import vdi2230
    r = vdi2230.stripping(10.0, 1.5, 7.2, 800.0, 0.65, 800.0, 0.65, 12.0, 58.0)                # s/d = 1,2 < 1,4: the VDI evaluation still evaluates c1 for 1,4 with its note
    assert "1.4" in r["note"] and r["C1"] == pytest.approx(-1.4 ** 2 + 3.8 * 1.4 - 2.61)


# ------------------------------------------------------------------ 8 / 9. deviations
def test_the_deviations_are_declared_and_shown():
    text = " ".join(BE.DEVIATIONS)
    for needle in ("Phi_e,n", "6.3.5.3", "Table 5-1 note 1", "bolt_thermal_factor", "F_nom_min", "s_w/d"):
        assert needle in text
    assert "DEVIATIONS" not in BE.FORMULA_NOTE and "see " not in BE.FORMULA_NOTE
    st = Study("review", "none.step")
    st.space = confirmed_space(COEF)
    d = s4_run.bolts_report(st, [dict(name="B", d_mm=6, A_s_mm2=20.1, k_b=3.5e5, k_c=1.2e6, F_nom=6000, F_ax=1000)])
    assert d["deviations"] == list(BE.DEVIATIONS) and "DEVIATIONS" not in d["formula_note"]


# ------------------------------------------------------------------ 10. records of an older bolt model
def test_a_record_stored_with_the_old_combined_exponent_gives_a_clear_message():
    st = Study("review", "none.step")
    st.space = confirmed_space(COEF)
    raw = dict(name="B", d_mm=6, A_s_mm2=20.1, k_b=3.5e5, k_c=1.2e6, F_nom=6000, F_ax=1000, combined_exponent=2.0)
    with pytest.raises(s4_run.S4Error) as e:
        s4_run.bolts_report(st, [raw])
    assert "older bolt model" in str(e.value) and "combined_exponent" in str(e.value) and "Traceback" not in str(e.value) and "__init__" not in str(e.value)
    with pytest.raises(s4_run.S4Error) as e2:
        s4_run.bolts_report(st, [{**raw, "combined_exponent": None, "bogus_field": 1}])
    assert "older bolt model" in str(e2.value) or "bogus_field" in str(e2.value)
    from fea_core.space import s4_verify
    rec = {"bolt": "B", "spec": {"raw": raw, "bolt_material": None, "part_material": None, "bearing_allowable": "Fu"}, "verdict": "PASS", "margins": []}
    sh = s4_verify.shown(st, "bolts", rec)
    assert sh.verdict != "PASS" and sh.text


def test_a_stale_combined_exponent_row_of_the_requirement_set_is_marked_obsolete():
    assert "bolt_combined_exponent" in OBSOLETE_LIMITS
    st = Study("review", "none.step")
    st.space = confirmed_space({**COEF, "bolt_combined_exponent": 2.0})
    lim = S4Limits.from_requirement_set(st.space)
    assert lim.obsolete == ["bolt_combined_exponent"] and "bolt_combined_exponent" not in lim.entries                    # never used
    d = s4_run.bolts_report(st, [dict(name="B", d_mm=6, A_s_mm2=20.1, k_b=3.5e5, k_c=1.2e6, F_nom=6000, F_ax=1000)])
    assert any("bolt_combined_exponent" in w and "obsolete" in w for w in d["bolts"][0]["warnings"])
    plain = S4Limits.from_requirement_set(confirmed_space(COEF))
    assert plain.obsolete == []

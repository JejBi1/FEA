"""F14 S4 review regressions of the ECSS bolt margins: the pull-out needs BOTH shear allowables and is the program's own VDI 2230 stripping capacity (I2, I3), every direct input
is validated once (I9), a missing embedding or thermal data blocks a PASS, the J convention (J on the external load only, ECSS-E-HB-32-23A Rev.1 7.9.2) is stated in the formula text."""
from __future__ import annotations

import pytest

from fea_core.joints import vdi2230
from fea_core.space import bolts_ecss as BE
from fea_core.space import s4_run
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Study
from tests.s4_helpers import confirmed_space
from tests.test_space_s4_bolts import AS, COEF, D, P, _bolt, _lim


def _m(r, name):
    return next(x for x in r.margins if x.check == name)


def test_baseline_has_a_verdict():
    r = BE.assess_bolt(_bolt(), _lim())
    assert r.verdict in (BE.PASS, BE.FAIL) and not any(m.status == BE.UNCONFIRMED for m in r.margins)


def test_pull_out_needs_both_the_bolt_and_the_part_allowable():                                   # I2
    for missing in ("part_Fsu", "bolt_Fsu"):
        al = {k: v for k, v in _bolt().allow.items() if k != missing}
        r = BE.assess_bolt(_bolt(allow=al, F_sh=0.0), _lim())
        m = _m(r, "pullout")
        assert m.status == BE.UNCONFIRMED and m.mos is None and f"allowable {missing} missing" in m.unconfirmed
        assert r.verdict == BE.UNCONFIRMED
    # an unverified allowable is not a PASS either
    al = dict(_bolt().allow)
    al["part_Fsu"] = BE.Allowable(200.0, False, "unverified")
    assert _m(BE.assess_bolt(_bolt(allow=al), _lim()), "pullout").status == BE.UNCONFIRMED


@pytest.mark.parametrize("tau_part", [60.0, 120.0, 200.0, 300.0, 600.0])
def test_pull_out_capacity_is_the_vdi_2230_stripping_capacity(tau_part):                           # I3
    al = dict(_bolt().allow)
    al["part_Fsu"] = BE.Allowable(tau_part, True, "test")
    r = BE.assess_bolt(_bolt(allow=al), _lim())
    m = _m(r, "pullout")
    sv = vdi2230.stripping(D, P, 9.0 - 2.0 * P, 1.0, 480.0, 1.0, tau_part, None, AS)               # L_eng,eff = L_eng - 2 p (ECSS-E-HB-32-23A Rev.1 7.10.2, eq. 7.10.3)
    assert m.capacity == pytest.approx(min(sv["F_mGM"], sv["F_mGS"]), rel=1e-12)                  # the thread constants c1 c2 c3 are those of VDI 2230
    assert "7.10" in m.formula and "c1 c3" in m.formula
    hot, ext = r.F_pmax - r.F_pmax0, r.F_b_max - r.F_pmax
    assert m.mos == pytest.approx(m.capacity / (r.F_pmax0 + 1.5 * (hot + ext)) - 1, rel=1e-12)       # eq. 7.10.17: J on the external and the thermal load, none on the preload
    # the old capacity without the thread-bending factors would differ from the VDI one for a weak internal thread: it must not be what is reported
    ar = BE.thread_shear_areas(D, P, 9.0)
    old = min(tau_part * ar["A_internal"], 480.0 * ar["A_bolt"])
    if sv["R_S"] < 0.9 or sv["R_S"] > 1.1:
        assert m.capacity != pytest.approx(old, rel=1e-3)


def test_the_j_convention_is_stated_in_the_formula_text():
    r = BE.assess_bolt(_bolt(), _lim())
    assert "F_V,max0 + J_T dF_T,gain + J Phi_n F_ax" in _m(r, "tensile_ultimate").formula and "[9.3." in _m(r, "combined").formula
    assert "to be confirmed" not in BE.__doc__.lower() and "EXTERNAL load only" in BE.__doc__ and "7.9.2" in BE.__doc__
    assert all("to be confirmed" not in m.formula for m in r.margins)


def test_missing_embedding_or_thermal_data_blocks_the_pass():
    base = BE.assess_bolt(_bolt(), _lim())
    assert base.verdict != BE.UNCONFIRMED
    no_emb = BE.assess_bolt(_bolt(embedding_um=None), _lim())
    assert no_emb.verdict == BE.UNCONFIRMED and any("embedding" in u for m in no_emb.margins for u in m.unconfirmed)
    assert any("no PASS is possible" in w for w in no_emb.warnings)
    zero = BE.assess_bolt(_bolt(embedding_um=0.0), _lim())                                           # "there is none" is a statement, not a gap
    assert zero.verdict != BE.UNCONFIRMED
    no_th = BE.assess_bolt(_bolt(T_min=None, T_max=None), _lim())
    assert no_th.verdict == BE.UNCONFIRMED and any("thermal" in u for m in no_th.margins for u in m.unconfirmed)
    no_alpha = BE.assess_bolt(_bolt(alpha_bolt=None), _lim())
    assert no_alpha.verdict == BE.UNCONFIRMED


def _lim_with(**kw):
    return S4Limits.from_requirement_set(confirmed_space({**COEF, **kw}))


@pytest.mark.parametrize("key,val", [("bolt_factor_gapping", 0.0), ("bolt_factor_tensile", -1.0), ("bolt_factor_shear", -0.5), ("bolt_factor_combined", 0.0),
                                     ("bolt_factor_pullout", -2.0), ("bolt_factor_bearing", 0.0), ("bolt_tightening_scatter", -0.1), ("bolt_relaxation_loss", 1.0),
                                     ("bolt_relaxation_loss", -0.5), ("bolt_slip_mu_min", 0.0), ("bolt_slip_mu_min", 50.0)])
def test_invalid_coefficients_of_the_requirement_set_are_errors(key, val):                       # I9
    with pytest.raises(BE.BoltsError):
        BE.assess_bolt(_bolt(), _lim_with(**{key: val}))
    st = Study("review", "none.step")
    st.space = confirmed_space({**COEF, key: val})
    with pytest.raises(s4_run.S4Error):
        s4_run.bolts_report(st, [dict(name="B", d_mm=6, A_s_mm2=20.1, k_b=3.5e5, k_c=1.2e6, F_nom=6000, F_ax=1000)])


@pytest.mark.parametrize("kw", [{"n_load_plane": 3.0}, {"n_load_plane": 0.0}, {"n_load_plane": -1.0}, {"F_ax": -5.0}, {"F_ax": float("nan")}, {"F_sh": float("inf")},
                                {"embedding_um": -500.0}, {"n_slip_planes": 500}, {"n_slip_planes": 0}, {"n_slip_planes": 1.5}, {"T_min": 90.0}, {"k_c": float("nan")},
                                {"factors": [1.2]}, {"factors": {"gapping": 0.0}}, {"factors": {"nonsense": 1.0}}, {"scatter": float("nan")}, {"F_nom": -1.0}, {"A_s_mm2": 0.0},
                                {"d_mm": float("inf")}])
def test_invalid_direct_inputs_are_errors(kw):                                                    # I9
    with pytest.raises(BE.BoltsError):
        BE.assess_bolt(_bolt(**kw), _lim())


def test_a_negative_shear_force_is_used_by_its_magnitude_with_a_note():
    r = BE.assess_bolt(_bolt(F_sh=-800.0), _lim())
    ref = BE.assess_bolt(_bolt(F_sh=800.0), _lim())
    assert any("magnitude" in w for w in r.warnings) and _m(r, "slip").mos == pytest.approx(_m(ref, "slip").mos, rel=1e-12)


def test_arithmetic_overflow_is_a_clean_result_or_error():
    st = Study("review", "none.step")
    st.space = confirmed_space(COEF)
    allow = {"bolt_Ftu": 800.0, "bolt_Fsu": 480.0, "bolt_Fty": 640.0, "part_Fsu": 200.0}
    raw = dict(name="B", d_mm=6, pitch_mm=1.0, A_s_mm2=20.1, k_b=3.5e5, k_c=1.2e6, F_nom=6000, F_ax=1e5, F_sh=1e120, shear_plane="shank", allowables=allow)
    out = s4_run.bolts_report(st, [raw])                                  # R_S ~ 1e117, R_S^3 overflows a float: the combined margin is -1, not a traceback
    comb = next(m for m in out["bolts"][0]["margins"] if m["check"] == "combined")
    assert comb["mos"] == -1.0
    # an ArithmeticError anywhere in a calculation reaches the caller as a clean S4Error
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        def boom(*_a, **_k):
            raise OverflowError("(34, 'Result too large')")
        mp.setattr(BE, "assess_bolt", boom)
        with pytest.raises(s4_run.S4Error):
            s4_run.bolts_report(st, [raw])


def test_the_listed_inputs_carry_the_geometry_data():
    r = BE.assess_bolt(_bolt(), _lim())
    names = {i["name"]: i for i in r.inputs}
    assert names["loading-plane factor n"]["confirmed"] is None and "entered directly" in names["loading-plane factor n"]["source"]
    assert "number of friction interfaces n_s" in names


def test_a_loading_plane_factor_below_one_is_used_but_never_confirmed():
    """Reviewer counterexample B5: n = 1e-6 makes every bolt pass (the load share of the bolt vanishes); it is a direct input, so it blocks the PASS."""
    bad = BE.assess_bolt(_bolt(F_ax=9000.0), _lim())
    assert bad.verdict == BE.FAIL
    rescued = BE.assess_bolt(_bolt(F_ax=9000.0, n_load_plane=1e-6), _lim())
    assert rescued.verdict == BE.UNCONFIRMED and any("loading-plane factor n < 1" in u for m in rescued.margins for u in m.unconfirmed)
    assert BE.assess_bolt(_bolt(n_load_plane=1.0), _lim()).verdict != BE.UNCONFIRMED

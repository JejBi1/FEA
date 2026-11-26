"""F14 S4: ECSS bolted-joint margins – hand-calculated gapping, slip, differential-CTE preload change (aluminium clamped parts, steel and titanium bolts), tensile /
shear / combined / pull-out / bearing, and the verdict rules. Every coefficient below is an ARBITRARY TEST VALUE entered through the requirement set; the formulas are
those of ECSS-E-HB-32-23A Rev.1 (6 February 2023), the equation numbers are quoted in the tests."""
from __future__ import annotations

import math

import pytest

from fea_core.space import bolts_ecss as BE
from fea_core.space.s4_limits import S4Limits
from tests.s4_helpers import confirmed_space

# the joint: M6, aluminium plate clamped by a steel bolt (and the same with a titanium bolt)
AS, D, P = 20.1, 6.0, 1.0
KB, KC = 3.5e5, 4.0e5                         # N/mm
A_AL, A_STEEL, A_TI = 23.4e-6, 11.7e-6, 8.6e-6
L = 12.0
T_REF, T_MIN, T_MAX = 20.0, -40.0, 80.0
KSER = KB * KC / (KB + KC)

COEF = {"bolt_tightening_scatter": 0.1, "bolt_relaxation_loss": 0.05, "bolt_slip_mu_min": 0.2, "bolt_factor_gapping": 1.2, "bolt_factor_slip": 1.25,
        "bolt_factor_tensile_yield": 1.1, "bolt_factor_tensile": 1.4, "bolt_factor_shear": 1.4, "bolt_factor_combined": 1.4, "bolt_factor_pullout": 1.5,
        "bolt_factor_bearing": 1.3}
ALLOW = {"bolt_Fty": BE.Allowable(640.0, True, "test"), "bolt_Ftu": BE.Allowable(800.0, True, "test"), "bolt_Fsu": BE.Allowable(480.0, True, "test"),
         "bolt_Fsy": BE.Allowable(370.0, True, "test"), "part_Fsu": BE.Allowable(200.0, True, "test"), "part_Fbr": BE.Allowable(420.0, True, "test"), "hole_Fbr": BE.Allowable(420.0, True, "test"),
         "part_Fy": BE.Allowable(300.0, True, "test")}


def _bolt(alpha_b=A_STEEL, **kw):
    d = dict(name="B1", d_mm=D, pitch_mm=P, A_s_mm2=AS, k_b=KB, k_c=KC, F_nom=6000.0, embedding_um=6.0, layers=[BE.Layer(8.0, A_AL), BE.Layer(4.0, A_AL)],
             alpha_bolt=alpha_b, L_grip_mm=L, T_ref=T_REF, T_min=T_MIN, T_max=T_MAX, F_ax=3000.0, F_sh=800.0, n_slip_planes=1, engaged_mm=9.0, d_w_mm=10.0, d_h_mm=6.6,
             t_bearing_mm=4.0, allow=dict(ALLOW))
    d.update(kw)
    return BE.EcssBolt(**d)


def _lim():
    return S4Limits.from_requirement_set(confirmed_space(COEF))


def test_load_factor_and_embedding():
    assert BE.load_factor(KB, KC) == pytest.approx(KB / (KB + KC), rel=1e-14)
    assert BE.load_factor(KB, KC, 0.5) == pytest.approx(0.5 * KB / (KB + KC), rel=1e-14)
    assert BE.embedding_loss(0.006, KB, KC) == pytest.approx(0.006 * KSER, rel=1e-14)
    with pytest.raises(BE.BoltsError):
        BE.load_factor(0.0, KC)


def test_differential_cte_preload_change_by_hand():
    # aluminium (23.4e-6) clamped by a steel bolt (11.7e-6): hot -> the plate grows more than the bolt -> the preload rises; cold -> it falls
    hot = (A_AL - A_STEEL) * (T_MAX - T_REF) * L * KSER
    cold = (A_AL - A_STEEL) * (T_MIN - T_REF) * L * KSER
    assert hot == pytest.approx(11.7e-6 * 60 * 12 * 186666.6667, rel=1e-6) and hot > 0 > cold
    assert BE.thermal_preload_change(KB, KC, A_AL, A_STEEL, L, T_MAX, T_REF) == pytest.approx(hot, rel=1e-14)
    assert BE.thermal_preload_change(KB, KC, A_AL, A_STEEL, L, T_MIN, T_REF) == pytest.approx(cold, rel=1e-14)
    r = BE.assess_bolt(_bolt(), _lim())
    assert r.dF_th["T_max"] == pytest.approx(hot) and r.dF_th["T_min"] == pytest.approx(cold)
    # titanium bolt: smaller bolt expansion -> larger differential, larger dF_th
    ti = BE.assess_bolt(_bolt(A_TI), _lim())
    assert ti.dF_th["T_max"] == pytest.approx((A_AL - A_TI) * 60 * L * KSER) and ti.dF_th["T_max"] > r.dF_th["T_max"]
    # a stack with the layers weighted by thickness
    a_c, t = BE.stack_alpha([BE.Layer(8.0, A_AL), BE.Layer(4.0, 10e-6)])
    assert t == 12.0 and a_c == pytest.approx((8 * A_AL + 4 * 10e-6) / 12)


def test_preload_gapping_and_slip_by_hand():
    r = BE.assess_bolt(_bolt(), _lim())
    phi = KB / (KB + KC)
    f_emb = 0.006 * KSER
    pmax0 = 6000.0 * 1.1
    pmin0 = 6000.0 * 0.9 - f_emb - 0.05 * 6000.0                        # eq. 6.3.18 with F_Z = embedding + the other losses (additive, eq. 6.4.3 form)
    cold = (A_AL - A_STEEL) * (T_MIN - T_REF) * L * KSER
    hot = (A_AL - A_STEEL) * (T_MAX - T_REF) * L * KSER
    assert r.phi == pytest.approx(phi) and r.F_emb == pytest.approx(f_emb)
    assert r.F_pmax0 == pytest.approx(pmax0) and r.F_pmin0 == pytest.approx(pmin0)
    assert r.F_pmin == pytest.approx(pmin0 + cold) and r.F_pmax == pytest.approx(pmax0 + hot)
    f_sep = (1 - phi) * 3000.0
    gap = {m.check: m for m in r.margins}["gapping"]
    assert gap.mos == pytest.approx((pmin0 + 1.2 * cold) / (1.2 * f_sep) - 1.0, rel=1e-12)           # eq. 7.8.1 with F_K,req = 0; the cold loss carries J_T = J (gapping)
    f_c = pmin0 + 1.25 * cold - f_sep                                                                # J_T = J (slip) = 1,25 (ECSS-E-ST-32-10C 4.3.2.2 b)
    slip = {m.check: m for m in r.margins}["slip"]
    assert slip.mos == pytest.approx(0.2 * f_c / (1.25 * 800.0) - 1.0, rel=1e-12)                    # eq. 9.2.6 (x = 1)
    # an axial force that opens the joint gives a negative gapping margin and a clamp force of zero
    big = BE.assess_bolt(_bolt(F_ax=12000.0), _lim())
    gm = {m.check: m for m in big.margins}
    assert gm["gapping"].mos < 0 and big.F_c_min == 0.0 and gm["slip"].mos == pytest.approx(-1.0)


def test_tensile_shear_combined_pullout_bearing_by_hand():
    r = BE.assess_bolt(_bolt(), _lim())
    m = {x.check: x for x in r.margins}
    phi = KB / (KB + KC)
    hot = (A_AL - A_STEEL) * (T_MAX - T_REF) * L * KSER
    fb = 6000.0 * 1.1 + hot + phi * 3000.0
    assert r.F_b_max == pytest.approx(fb)
    pmax0 = 6000.0 * 1.1
    ext = phi * 3000.0

    def fpmax(j):                                           # F_V,max0 + J_T dF_hot with J_T = J of the check
        return pmax0 + j * hot
    # eq. 7.9.2 / 7.9.3: the safety factor multiplies the EXTERNAL load and the thermally induced increment, never the preload
    assert m["tensile_yield"].mos == pytest.approx(640.0 * AS / (fpmax(1.1) + 1.1 * ext) - 1, rel=1e-12)
    assert m["tensile_ultimate"].mos == pytest.approx(800.0 * AS / (fpmax(1.4) + 1.4 * ext) - 1, rel=1e-12)
    assert m["shear"].mos == pytest.approx(480.0 * AS / (1.4 * 800.0) - 1, rel=1e-12)                 # eq. 9.3.2
    assert m["shear_yield"].mos == pytest.approx(370.0 * AS / (1.1 * 800.0) - 1, rel=1e-12)           # eq. 9.3.1
    # eq. 9.3.3 / 9.3.4 / 9.3.13 / 9.3.14: thread in the shear plane, R_comb = sqrt(R_A^2 + R_S^2), MoS = 1/R_comb - 1
    ra, rs = (fpmax(1.4) + 1.4 * ext) / (800.0 * AS), 1.4 * 800.0 / (480.0 * AS)
    assert m["combined"].mos == pytest.approx(1 / math.sqrt(ra ** 2 + rs ** 2) - 1, rel=1e-12)
    ray, rsy = (fpmax(1.1) + 1.1 * ext) / (640.0 * AS), 1.1 * 800.0 / (370.0 * AS)
    assert m["combined_yield"].mos == pytest.approx(1 / math.sqrt(ray ** 2 + rsy ** 2) - 1, rel=1e-12)
    # thread shear areas (ISO metric geometry): internal thread at d, bolt thread at D1
    D1, D2 = D - 1.082532 * P, D - 0.649519 * P
    a_int = math.pi * D / P * (P / 2 + (D - D2) * math.tan(math.radians(30))) * 9.0
    a_bolt = math.pi * D1 / P * (P / 2 + (D2 - D1) * math.tan(math.radians(30))) * 9.0
    # the thread-bending factors of VDI 2230 (Alexander; C1 = 1 for a tapped hole): R_S = tau_M A_SGM / (tau_S A_SGS), C2 for the bolt thread, C3 for the internal one
    rs_ = 200.0 * a_int / (480.0 * a_bolt)
    c2 = 0.897 if rs_ <= 1 else 5.594 - 13.682 * rs_ + 14.107 * rs_ ** 2 - 6.057 * rs_ ** 3 + 0.9353 * rs_ ** 4
    c3 = 0.897 if rs_ >= 1 else 0.728 + 1.769 * rs_ - 2.896 * rs_ ** 2 + 1.296 * rs_ ** 3
    cap = min(200.0 * a_int * c3, 480.0 * a_bolt * c2)
    from fea_core.joints import vdi2230
    sv = vdi2230.stripping(D, P, 9.0, 1.0, 480.0, 1.0, 200.0, None, AS)               # the VDI evaluation of the program on the full engaged length: the same constants
    assert cap == pytest.approx(min(sv["F_mGM"], sv["F_mGS"]), rel=1e-9)
    # 7.10.2 eq. 7.10.3: the effective engaged length is L_eng - 2 p (7 mm here), the capacity is proportional to it; 7.10.4 eq. 7.10.17: J on the external load only
    cap_eff = cap * (9.0 - 2.0 * P) / 9.0
    assert m["pullout"].mos == pytest.approx(cap_eff / (fpmax(1.5) + 1.5 * ext) - 1, rel=1e-9)
    a_br = math.pi / 4 * (10.0 ** 2 - 6.6 ** 2)
    assert m["bearing_head"].mos == pytest.approx(420.0 * a_br / (fpmax(1.3) + 1.3 * ext) - 1, rel=1e-12)    # eq. 7.11.2
    assert m["bearing_head_yield"].mos == pytest.approx(300.0 * a_br / (fpmax(1.1) + 1.1 * ext) - 1, rel=1e-12)   # yield strength of the part with the yield factor
    assert m["bearing_hole_yield"].mos == pytest.approx(300.0 * 6.0 * 4.0 / (1.1 * 800.0) - 1, rel=1e-12)
    assert m["bearing_hole"].mos == pytest.approx(420.0 * 6.0 * 4.0 / (1.3 * 800.0) - 1, rel=1e-12)    # eq. 9.3.22 - 9.3.24 (d = the fastener diameter)
    assert r.verdict in (BE.PASS, BE.FAIL)                                     # everything confirmed: a verdict is allowed
    assert BE.FORMULA_NOTE in r.warnings and "ECSS-E-HB-32-23A Rev.1" in r.formula_note and "to be confirmed" not in r.formula_note


def test_safety_factor_acts_on_the_external_load_only():
    """ECSS-E-HB-32-23A Rev.1 7.9.2: the uncertainty of the preload is already in F_V,max, so J must not multiply it (ECSS-E-ST-32-10C Table 4-4 note a). The thermally induced
    increment carries the factor of the check (4.3.2.2 b) unless the row bolt_thermal_factor says 1,0."""
    f1 = {**COEF, "bolt_factor_tensile": 1.0, "bolt_factor_pullout": 1.0, "bolt_factor_bearing": 1.0, "bolt_factor_combined": 1.0}
    f2 = {**COEF, "bolt_factor_tensile": 2.0, "bolt_factor_pullout": 2.0, "bolt_factor_bearing": 2.0, "bolt_factor_combined": 2.0}
    m1 = {m.check: m for m in BE.assess_bolt(_bolt(), S4Limits.from_requirement_set(confirmed_space(f1))).margins}
    m2 = {m.check: m for m in BE.assess_bolt(_bolt(), S4Limits.from_requirement_set(confirmed_space(f2))).margins}
    r = BE.assess_bolt(_bolt(), S4Limits.from_requirement_set(confirmed_space(f1)))
    ext = r.F_b_max - r.F_pmax
    hot = r.F_pmax - r.F_pmax0
    cap = m1["tensile_ultimate"].capacity
    assert m1["tensile_ultimate"].mos == pytest.approx(cap / r.F_b_max - 1, rel=1e-12)
    assert m2["tensile_ultimate"].mos == pytest.approx(cap / (r.F_pmax0 + 2.0 * (hot + ext)) - 1, rel=1e-12)
    assert m2["tensile_ultimate"].mos > cap / (2.0 * r.F_b_max) - 1                 # the old whole-force factor (also on the preload) was more conservative
    # without an external axial force and with J_T = 1 (row bolt_thermal_factor) J has no effect on the overall-load margins
    n1 = {m.check: m for m in BE.assess_bolt(_bolt(F_ax=0.0), S4Limits.from_requirement_set(confirmed_space({**f1, "bolt_thermal_factor": 1.0}))).margins}
    n2 = {m.check: m for m in BE.assess_bolt(_bolt(F_ax=0.0), S4Limits.from_requirement_set(confirmed_space({**f2, "bolt_thermal_factor": 1.0}))).margins}
    for k in ("tensile_ultimate", "pullout", "bearing_head"):
        assert n1[k].mos == pytest.approx(n2[k].mos, rel=1e-12)


def test_handbook_options_required_clamp_force_shank_and_wrench_size():
    base = {m.check: m for m in BE.assess_bolt(_bolt(), _lim()).margins}
    # 7.8.1: F_K,req lowers the numerator of the separation margin
    kreq = {m.check: m for m in BE.assess_bolt(_bolt(F_K_req=500.0), _lim()).margins}
    assert kreq["gapping"].mos == pytest.approx((base["gapping"].capacity - 500.0) / (1.2 * base["gapping"].demand) - 1, rel=1e-12)
    # 9.3.2.3 eq. 9.3.5 / 9.3.6: a shank in the shear plane has the cubic shear term
    sh = {m.check: m for m in BE.assess_bolt(_bolt(shear_plane="shank"), _lim()).margins}
    ra, rs = base["combined"].demand ** 2 - (1.4 * 800.0 / (480.0 * AS)) ** 2, 1.4 * 800.0 / (480.0 * AS)
    assert sh["combined"].mos == pytest.approx(1 / math.sqrt(ra + rs ** 3) - 1, rel=1e-12) and sh["combined"].mos > base["combined"].mos
    with pytest.raises(BE.BoltsError):
        BE.assess_bolt(_bolt(shear_plane="middle"), _lim())
    # 7.10.2 eq. 7.10.4: a nut (wrench size s) has c1 = -(s/d)^2 + 3.8 s/d - 2.61 (1.4 <= s/d <= 1.9), c1 = 1 for a tapped hole
    nut = {m.check: m for m in BE.assess_bolt(_bolt(wrench_mm=10.2), _lim()).margins}
    c1 = -1.7 ** 2 + 3.8 * 1.7 - 2.61
    assert nut["pullout"].capacity == pytest.approx(c1 * base["pullout"].capacity, rel=1e-12)
    # no thread engagement beyond the two unloaded turns: no capacity
    short = {m.check: m for m in BE.assess_bolt(_bolt(engaged_mm=2.0 * P), _lim()).margins}
    assert short["pullout"].capacity == 0.0 and short["pullout"].mos == pytest.approx(-1.0)
    # no shear yield allowable: the yield margins of shear and combined are UNCONFIRMED (never PASS), with a warning
    al = dict(ALLOW)
    del al["bolt_Fsy"]
    r = BE.assess_bolt(_bolt(allow=al), _lim())
    mm = {m.check: m for m in r.margins}
    assert mm["shear_yield"].status == BE.UNCONFIRMED and mm["combined_yield"].status == BE.UNCONFIRMED and r.verdict == BE.UNCONFIRMED
    assert any("bolt_Fsy" in w for w in r.warnings)


def test_nothing_unconfirmed_ever_passes():
    unc = BE.assess_bolt(_bolt(), S4Limits.from_requirement_set(confirmed_space(COEF, confirm_it=False)))
    assert unc.verdict == BE.UNCONFIRMED and all(m.status != BE.PASS for m in unc.margins)
    # the same numbers entered directly are used but not confirmed
    direct = BE.assess_bolt(_bolt(scatter=0.1, relaxation=0.05, mu_min=0.2,
                                  factors={"gapping": 1.2, "slip": 1.25, "tensile_yield": 1.1, "tensile": 1.4, "shear": 1.4, "combined": 1.4, "pullout": 1.5, "bearing": 1.3}))
    ref = BE.assess_bolt(_bolt(), _lim())
    assert [round(m.mos or 0, 9) for m in direct.margins] == [round(m.mos or 0, 9) for m in ref.margins]
    assert direct.verdict == BE.UNCONFIRMED
    # an unverified allowable keeps its check unconfirmed even when every coefficient is confirmed
    al = dict(ALLOW)
    al["bolt_Ftu"] = BE.Allowable(800.0, False, "test")
    part = BE.assess_bolt(_bolt(allow=al), _lim())
    assert {m.check: m for m in part.margins}["tensile_ultimate"].status == BE.UNCONFIRMED and part.verdict == BE.UNCONFIRMED
    # a missing J: that margin is not computed
    lim = dict(COEF)
    del lim["bolt_factor_gapping"]
    nj = BE.assess_bolt(_bolt(), S4Limits.from_requirement_set(confirmed_space(lim)))
    g = {m.check: m for m in nj.margins}["gapping"]
    assert g.mos is None and g.status == BE.UNCONFIRMED


def test_failing_margin_with_confirmed_inputs_is_fail():
    r = BE.assess_bolt(_bolt(F_sh=20000.0), _lim())
    assert {m.check: m for m in r.margins}["slip"].mos < 0 and r.verdict == BE.FAIL


def test_adapter_from_the_vdi_evaluation():
    bas = {"d": 6.0, "P": 1.0, "As": AS, "delta_S": 1 / KB, "delta_P": 1 / KC, "F_Mmax": 8000.0, "F_Mmin": 5000.0, "f_Z": 0.006, "dW": 10.0, "dh": 6.6, "lK": 12.0}
    b = BE.from_vdi_basics(bas, "B7", F_ax=100.0, T_min=-40.0)
    assert b.k_b == pytest.approx(KB) and b.k_c == pytest.approx(KC) and b.F_nom == 6500.0 and b.embedding_um == pytest.approx(6.0) and b.F_ax == 100.0
    assert BE.loads_from_vdi({"F_A": -5.0, "F_Q": -300.0}) == {"F_ax": 0.0, "F_sh": 300.0}
    with pytest.raises(BE.BoltsError):
        BE.from_vdi_basics(bas, nonsense=1)


def test_bolt_of_the_study_gives_its_stiffness_to_the_ecss_chain():
    """`joint` = a bolt of the study: stiffness, stress area, preload and embedding come from its VDI 2230 evaluation (`joints.fem.bolt_basics`)."""
    from fea_core.joints.fem import bolt_basics
    from fea_core.joints.model import BoltSpec
    from fea_core.space import s4_run
    from verification import space_cases as SC
    st = SC.block_study()
    st.bolts = [BoltSpec("M10", plates=[{"part": "Block", "s0": 0.0, "s1": 10.0, "hole": 11.0}, {"part": "Block", "s0": 10.0, "s1": 20.0, "hole": 11.0}], head_part="Block",
                         nut_part="Block", nut_s=20.0, head_dw=16.0, nut_dw=16.0, preload_mode="force", preload=20000.0)]
    bas = bolt_basics(st, st.bolts[0])
    b = s4_run.bolt_from_dict(st, {"joint": "M10", "F_ax": 1000.0, "F_sh": 500.0, "T_min": -40.0}, {})
    assert b.name == "M10" and b.k_b == pytest.approx(1.0 / bas["delta_S"]) and b.k_c == pytest.approx(1.0 / bas["delta_P"]) and b.A_s_mm2 == pytest.approx(bas["As"])
    assert b.F_nom == pytest.approx(0.5 * (bas["F_Mmax"] + bas["F_Mmin"])) and b.embedding_um == pytest.approx(bas["f_Z"] * 1e3) and b.F_ax == 1000.0
    r = BE.assess_bolt(b, S4Limits())
    assert 0.0 < r.phi < 1.0 and r.F_pmin0 > 0 and r.verdict == BE.UNCONFIRMED
    with pytest.raises(s4_run.S4Error):
        s4_run.bolt_from_dict(st, {"joint": "M99"}, {})
    with pytest.raises(s4_run.S4Error):
        s4_run.bolt_from_dict(st, {"joint": "M10", "bogus": 1}, {})

"""F14 S4, third pass on the ECSS bolt margins (round-2 review): a joint whose clamp force is lost by the cold case fails even without F_K,req, a thermal factor below 1,0 is not
confirmed, J_T is traceable (column of the margin, formulas, report), the missing plate thickness of the hole bearing is warned about. ARBITRARY TEST VALUES."""
from __future__ import annotations

import pytest

from fea_core.report import space_s4 as RS
from fea_core.space import bolts_ecss as BE
from fea_core.space import s4_run
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Study
from tests.s4_helpers import confirmed_space
from tests.test_space_s4_bolts import COEF, KSER, L, T_MIN, T_REF, A_AL, A_STEEL, _bolt, _lim

DIFF = (A_AL - A_STEEL) * L * KSER
F_EMB = 0.006 * KSER


def _lim_with(**kw) -> S4Limits:
    return S4Limits.from_requirement_set(confirmed_space({**COEF, **kw}))


def _m(r, name):
    return next(x for x in r.margins if x.check == name)


# ------------------------------------------------------------------ 1. the clamp force is gone although F_K,req = 0 and no axial force acts
def test_cold_loss_of_the_whole_preload_fails_without_a_required_clamp_force():
    """F_nom 1500 N, T_min = -40 C: F_V,min0 = 155 N, the cold case takes 1,2 x 1572 N: F_V,min = -1732 N. It used to be PASS (n/a margin) with no warning."""
    r = BE.assess_bolt(_bolt(F_nom=1500.0, F_ax=0.0, F_sh=0.0), _lim())
    g = _m(r, "gapping")
    f_min = 1500.0 * 0.9 - F_EMB - 0.05 * 1500.0 + 1.2 * DIFF * (T_MIN - T_REF)
    assert f_min < -1700.0 and g.capacity == pytest.approx(f_min, rel=1e-12)
    assert g.status == BE.FAIL and g.mos == -1.0 and r.verdict == BE.FAIL and g.thermal_factor == pytest.approx(1.2)
    assert any("not above zero" in w and "separates" in w for w in r.warnings)


def test_a_negative_minimum_preload_before_the_thermal_change_is_warned_about_and_fails():
    r = BE.assess_bolt(_bolt(F_nom=400.0, F_ax=0.0, F_sh=0.0, T_min=20.0, T_max=20.0), _lim())              # no thermal change at all: only the embedding loss remains
    assert r.F_pmin0 == 0.0 and any("negative" in w and "taken as 0 N" in w for w in r.warnings)
    g = _m(r, "gapping")
    assert g.status == BE.FAIL and g.mos == -1.0
    # a joint that keeps its clamp force stays n/a / PASS and carries no such warning
    ok = BE.assess_bolt(_bolt(F_ax=0.0, F_sh=0.0), _lim())
    assert _m(ok, "gapping").status == "n/a" and not any("negative" in w or "separates" in w for w in ok.warnings)


def test_a_required_clamp_force_still_gives_the_ratio_margin():
    r = BE.assess_bolt(_bolt(F_nom=1500.0, F_ax=0.0, F_sh=0.0, F_K_req=100.0), _lim())
    g = _m(r, "gapping")
    assert g.mos == pytest.approx(g.capacity / 100.0 - 1.0, rel=1e-12) and g.status == BE.FAIL and any("below the required clamp force" in w for w in r.warnings)


def test_gapping_with_an_axial_force_is_unchanged_for_a_lost_clamp_force():
    r = BE.assess_bolt(_bolt(F_nom=1500.0, F_ax=3000.0), _lim())
    assert _m(r, "gapping").status == BE.FAIL and _m(r, "gapping").mos < 0


# ------------------------------------------------------------------ 3. J_T below 1,0
def test_a_thermal_factor_row_below_one_is_unconfirmed():
    r = BE.assess_bolt(_bolt(), _lim_with(bolt_thermal_factor=0.8))
    assert r.verdict == BE.UNCONFIRMED
    assert any("below 1,0" in u and "4.3.2.1" in u for m in r.margins for u in m.unconfirmed)
    ok = BE.assess_bolt(_bolt(), _lim_with(bolt_thermal_factor=1.0))
    assert not any("below 1,0" in u for m in ok.margins for u in m.unconfirmed)


def test_the_default_thermal_factor_is_never_below_one():
    lim = S4Limits.from_requirement_set(confirmed_space({**COEF, "bolt_factor_gapping": 0.9, "bolt_factor_tensile": 0.8, "bolt_factor_tensile_yield": 0.7, "bolt_factor_slip": 0.95}))
    r = BE.assess_bolt(_bolt(), lim)
    cold, hot = DIFF * (T_MIN - T_REF), DIFF * (80.0 - T_REF)
    assert _m(r, "gapping").thermal_factor == 1.0 and _m(r, "gapping").capacity == pytest.approx(r.F_pmin0 + cold, rel=1e-12)       # not 0,9 x the loss
    assert _m(r, "slip").thermal_factor == 1.0
    tu = _m(r, "tensile_ultimate")
    assert tu.thermal_factor == 1.0 and tu.mos == pytest.approx(tu.capacity / (r.F_pmax0 + hot + 0.8 * 3000.0 * (3.5e5 / 7.5e5)) - 1.0, rel=1e-12)
    # a check factor above one is used as it is
    hi = BE.assess_bolt(_bolt(), _lim())
    assert _m(hi, "tensile_ultimate").thermal_factor == pytest.approx(1.4) and _m(hi, "gapping").thermal_factor == pytest.approx(1.2)


# ------------------------------------------------------------------ 4. J_T is traceable
def test_jt_is_in_the_margins_the_formulas_and_the_report():
    r = BE.assess_bolt(_bolt(), _lim())
    d = r.to_dict()
    thermal_checks = {m["check"]: m["thermal_factor"] for m in d["margins"]}
    for check in ("gapping", "slip", "tensile_yield", "tensile_ultimate", "combined", "combined_yield", "pullout", "bearing_head", "bearing_head_yield"):
        assert thermal_checks[check] is not None, check
        assert "J_T" in next(m["formula"] for m in d["margins"] if m["check"] == check) or check.startswith("combined")
    for m in d["margins"]:
        if m["check"] in ("tensile_ultimate", "pullout", "bearing_head", "combined"):
            assert "J_T dF_T,gain" in m["formula"], m["check"]
        if m["check"] in ("gapping", "slip"):
            assert "J_T dF_T,loss" in m["formula"], m["check"]
    assert thermal_checks["shear"] is None                                      # no thermal increment in the shear of the bolt
    # the report prints the column and says which factor the preload row carries
    st = Study("round3", "none.step")
    st.space = confirmed_space(COEF)
    raw = dict(name="B", d_mm=6, pitch_mm=1.0, A_s_mm2=20.1, k_b=3.5e5, k_c=4.0e5, F_nom=3000, F_ax=1000.0, embedding_um=6.0, T_min=-40.0, T_max=80.0, alpha_bolt=11.7e-6,
               layers=[{"t_mm": 12.0, "alpha": 23.4e-6}], allowables={"bolt_Ftu": 800.0})
    rep = s4_run.bolts_report(st, [raw])["bolts"][0]
    assert all("thermal_factor" in m for m in rep["margins"])

    class Rec:
        def __init__(self):
            self.tables, self.kv_rows, self.text = [], [], []

        def table(self, header, rows, **kw):
            self.tables.append((header, rows))

        def kv(self, rows, **kw):
            self.kv_rows += rows

        def __getattr__(self, name):
            def rec(*a, **k):
                self.text += [x for x in a if isinstance(x, str)]
            return rec

    doc = Rec()
    RS.bolts_section(doc, {"B": rep}, "6.1", st)
    header, rows = next((h, r) for h, r in doc.tables if "J_T" in h)
    col = header.index("J_T")
    assert header.index("Status") == col + 2 and any(row[col] != "–" for row in rows)
    assert any("J_T" in k for k, _v in doc.kv_rows if k.startswith("Preload after thermal change"))
    assert any("4.3.2.1 b" in t and "4.3.2.2 a" in t for t in doc.text)


def test_limit_meanings_name_the_thermal_increment():
    from fea_core.space.s4_limits import LIMIT_DESCRIPTIONS
    for key in ("bolt_factor_tensile_yield", "bolt_factor_tensile", "bolt_factor_combined", "bolt_factor_pullout", "bolt_factor_bearing"):
        meaning = LIMIT_DESCRIPTIONS[key][1]
        assert "thermally induced increment" in meaning and "external" in meaning.lower(), key
    assert "4.3.2.1 b" in LIMIT_DESCRIPTIONS["bolt_thermal_factor"][1]


# ------------------------------------------------------------------ 5. citations
def test_the_positive_rule_4_3_2_1_b_is_cited_with_the_clauses_of_the_thermal_factor():
    assert "4.3.2.1 b" in BE.__doc__ and "4.3.2.2 a" in BE.__doc__
    assert any("4.3.2.1 b" in d and "4.3.2.2 a" in d and "4.3.2.2 b" in d for d in BE.DEVIATIONS)


# ------------------------------------------------------------------ 6. nits
def test_hole_bearing_without_a_plate_thickness_is_warned_about():
    r = BE.assess_bolt(_bolt(t_bearing_mm=None), _lim())
    assert any("Hole bearing is not assessed" in w and "t_bearing" in w for w in r.warnings)
    assert not any(m.check.startswith("bearing_hole") for m in r.margins)
    ok = BE.assess_bolt(_bolt(), _lim())
    assert not any("Hole bearing is not assessed" in w for w in ok.warnings)
    none_shear = BE.assess_bolt(_bolt(t_bearing_mm=None, F_sh=0.0), _lim())               # no shear force: nothing to warn about
    assert not any("Hole bearing is not assessed" in w for w in none_shear.warnings)

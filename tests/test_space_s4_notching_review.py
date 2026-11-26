"""F14 S4 review regressions of the notching (C3, I6, I7 and the minor findings): non-finite values are errors, never a PASS; the design factor on a PSD is J²; a harmonic
result with no base acceleration at some frequencies is cleaned or refused; a limit table is applied most-restrictively outside its range; the exported breakpoint table cannot
exceed the limit; a genuine two-DOF case (two resonances) with the exact direct solution as the reference."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fea_core.dynamics import interface as IF
from fea_core.space import notching as NT
from fea_core.space import s4_run
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Study
from tests.s4_helpers import G, M_SDOF, Z_SDOF, confirmed_space, sdof_interface

ZETA = Z_SDOF
PEAK = M_SDOF * G * math.sqrt(1 + 4 * ZETA ** 2) / (2 * ZETA)
RULES = {"notch_min_width_oct": 0.1, "notch_floor_ratio": 0.2, "notch_design_factor": 1.0}


def _sine():
    mi = sdof_interface()
    f = IF.sine_grid(50.0, 2000.0, mi.freq, np.array([ZETA]))
    si = IF.sine_interface(mi, "x", 1.0, ZETA, f=f)
    return mi, f, si, NT.interface_response(si)


def _lims(**kw):
    return S4Limits.from_requirement_set(confirmed_space({"notch_force_limit_N": 0.5 * PEAK, **RULES, **kw}))


# ------------------------------------------------------------------ C3
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0])
def test_a_non_finite_response_is_an_error_never_a_verdict(bad):
    mi, f, si, resp = _sine()
    r2 = resp.copy()
    k = int(np.argmax(resp))
    r2[k - 3:k + 4] = bad
    with pytest.raises(NT.NotchError):
        NT.make_notch(f, si.a_g, r2, NT.NotchSpec("sine", "force"), _lims(), mi.freq)
    with pytest.raises(NT.NotchError):
        NT.make_notch(f, si.a_g, np.full_like(resp, np.nan), NT.NotchSpec("sine", "force"), _lims(), mi.freq)


def test_a_non_finite_level_or_limit_is_an_error():
    mi, f, si, resp = _sine()
    lv = si.a_g.copy()
    lv[5] = np.nan
    with pytest.raises(NT.NotchError):
        NT.make_notch(f, lv, resp, NT.NotchSpec("sine", "force"), _lims(), mi.freq)
    for bad in (np.nan, np.inf, 0.0, -3.0):
        with pytest.raises(NT.NotchError):
            NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=bad), S4Limits(), mi.freq)
    with pytest.raises(NT.NotchError):
        NT.notch_scale(f, resp, np.where(f > 300, np.nan, 1.0))


def _harmonic(S0, ab0):
    fh = np.array([50.0, 100.0, 200.0, 300.0, 400.0])
    ab = np.zeros((5, 3), complex)
    ab[:, 0] = ab0
    S = np.zeros((5, 6), complex)
    S[:, 0] = S0
    return SimpleNamespace(harmonic={"f": fh, "ab": ab, "probes": {"P": {"U": np.zeros((5, 3), complex), "S": S}}})


def _study(excitation="base_acc"):
    st = Study("review", "none.step")
    st.space = confirmed_space({"notch_stress_limit_MPa": 100.0, **{**RULES, "notch_floor_ratio": 0.5}})
    if excitation is not None:
        from fea_core.dynamics.model import Excitation
        st.dynamics.harmonic.excitation = excitation if isinstance(excitation, list) else [Excitation(kind=excitation, table=[[10.0, 1.0], [1000.0, 1.0]])]
    return st


def test_a_harmonic_result_without_base_acceleration_at_some_frequencies_is_cleaned():
    dyn = _harmonic([300.0, 50.0, 60.0, 50.0, 300.0], [0, 9810.0, 9810.0, 9810.0, 0])     # stress where ab = 0 would be inf
    with np.errstate(all="ignore"):
        f, t, notes, probe = NT.harmonic_response_ex(dyn, "stress", "P")
    assert list(f) == [100.0, 200.0, 300.0] and np.all(np.isfinite(t)) and probe == "P"
    assert notes and "no base acceleration" in notes[0]
    with np.errstate(all="ignore"):
        d, res = s4_run.notch_report(_study(), "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn, probe="P")
    assert np.all(np.isfinite(res.quantity)) and np.all(np.isfinite(res.quantity_notched))
    assert any("left out" in w for w in res.warnings)
    # 0/0 at the ends: the same
    dyn0 = _harmonic([0.0, 50.0, 60.0, 50.0, 0.0], [0, 9810.0, 9810.0, 9810.0, 0])
    with np.errstate(all="ignore"):
        d0, res0 = s4_run.notch_report(_study(), "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn0, probe="P")
    assert np.isfinite(d0["notched_peak"]) and d0["verdict"] in ("PASS", "FAIL", "UNCONFIRMED_REQUIREMENTS")


# ------------------------------------------------------------------ I6
def test_the_design_factor_on_a_psd_acts_as_j_squared():
    mi = sdof_interface()
    ri = IF.random_interface(mi, "x", [[20.0, 0.02], [2000.0, 0.02]], ZETA)
    psd = NT.interface_psd_response(ri)
    L = 0.25 * psd.max()
    res = NT.make_notch(ri.f, ri.asd_g2, psd, NT.NotchSpec("random", "force_psd", limit=L, design_factor=1.25), S4Limits(), mi.freq)
    assert res.design_factor == 1.25 and res.design_factor_applied == pytest.approx(1.25 ** 2)
    assert 1.25 ** 2 * res.quantity_notched.max() == pytest.approx(L, rel=1e-12)             # J², not J
    assert any("J^2" in i["name"] for i in res.inputs)
    n = res.notches[0]
    assert n.margin_remaining == pytest.approx(0.0, abs=1e-9)
    # a sine amplitude keeps J
    mi, f, si, resp = _sine()
    rs = NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.4 * PEAK, design_factor=1.25), S4Limits(), mi.freq)
    assert rs.design_factor_applied == 1.25
    from fea_core.space.s4_limits import LIMIT_DESCRIPTIONS
    assert "J^2" in LIMIT_DESCRIPTIONS["notch_design_factor"][1]


# ------------------------------------------------------------------ I7
def test_a_stress_or_probe_verdict_says_where_it_holds():
    dyn = _harmonic([50.0, 50.0, 60.0, 50.0, 50.0], [9810.0] * 5)
    d, res = s4_run.notch_report(_study("base_acc"), "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn, probe="P")
    assert "at probe 'P' ONLY" in d["verdict_text"] and d["scope"] == "at probe 'P'" and d["probe"] == "P"


def test_the_harmonic_result_must_be_a_base_only_run():
    dyn = _harmonic([50.0, 50.0, 60.0, 50.0, 50.0], [9810.0] * 5)
    with pytest.raises(s4_run.S4Error, match="not a pure base"):
        s4_run.notch_report(_study("force"), "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn, probe="P")
    from fea_core.dynamics.model import Excitation
    two = [Excitation(kind="base_acc", direction=[1, 0, 0], table=[[10.0, 1.0], [1000.0, 1.0]]), Excitation(kind="base_acc", direction=[0, 1, 0], table=[[10.0, 1.0], [1000.0, 1.0]])]
    with pytest.raises(s4_run.S4Error, match="several directions"):
        s4_run.notch_report(_study(two), "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn, probe="P")
    with pytest.raises(s4_run.S4Error, match="no excitation"):
        s4_run.notch_report(_study([]), "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn, probe="P")


# ------------------------------------------------------------------ I9 (notching part) and the minimum width
def test_design_factor_and_minimum_width_are_validated():
    mi, f, si, resp = _sine()
    for J in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(NT.NotchError):
            NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.5 * PEAK, design_factor=J), S4Limits(), mi.freq)
    for J in (0.0, -1.0):
        with pytest.raises(NT.NotchError):
            NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force"), _lims(notch_design_factor=J), mi.freq)       # a confirmed set with J = 0 cannot give a verdict
    for w in (0.0, -1.0, 50.0, 3000.0, 1e9, np.nan, np.inf):
        with pytest.raises(NT.NotchError):
            NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.5 * PEAK, min_width_oct=w), S4Limits(), mi.freq)
    span = math.log2(f[-1] / f[0])
    res = NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.5 * PEAK, min_width_oct=30.0), S4Limits(), mi.freq)      # wider than the band: the band is the notch
    assert res.notches[0].width_oct <= span + 1e-9 and any("exceeds the analysed band" in w for w in res.warnings)


# ------------------------------------------------------------------ minor: limit table outside its range, breakpoints
def test_a_limit_table_outside_its_range_uses_the_smallest_value():
    mi, f, si, resp = _sine()
    tab = [[400.0, 0.1 * PEAK], [500.0, 0.9 * PEAK]]
    res = NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit_table=tab), S4Limits(), mi.freq)
    n = res.notches[0]
    assert n.scale_min == pytest.approx(0.1, rel=1e-3)                    # the resonance (318 Hz) is outside the table: the most restrictive value 0.1·PEAK applies
    assert any("smallest value of the table" in w for w in res.warnings) and res.verdict == NT.UNCONFIRMED
    L = NT.limit_from_table(tab, np.array([100.0, 450.0, 1000.0]), [])
    assert L[0] == L[2] == pytest.approx(0.1 * PEAK) and 0.1 * PEAK < L[1] < 0.9 * PEAK
    for bad in ([[1.0, 0.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]], [[1.0, np.nan], [2.0, 1.0]], [[1.0, 1.0, 3.0]]):
        with pytest.raises(NT.NotchError):
            NT.limit_from_table(bad, f, [])


def test_the_breakpoint_table_never_exceeds_the_limit():
    mi, f, si, resp = _sine()
    for frac in (0.4, 0.1):
        L = frac * PEAK
        res = NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=L, design_factor=1.0, floor_ratio=0.01, min_width_oct=1e-6), S4Limits(), mi.freq)
        fb = np.array([p[0] for p in res.breakpoints])
        ab = np.array([p[1] for p in res.breakpoints])
        assert len(fb) < len(f)                                                # it is a simplification
        # on the analysis grid (where the dense profile keeps the limit exactly) the table is never above the dense profile
        tab = np.exp(np.interp(np.log(f), np.log(fb), np.log(ab)))
        assert np.all(tab <= res.notched * (1 + 1e-12))
        # the CSV export lists the same table
        rows = NT.profile_csv(res).splitlines()[2:]
        assert len(rows) == len(res.breakpoints)
    one = NT.breakpoints(np.array([10.0, 20.0, 40.0]), np.array([1.0, 0.5, 1.0]), 0.1)
    assert one[1] == (20.0, 0.5)                                               # a vertex above the chord is kept (the chord would exceed the profile)


def test_the_exported_table_is_re_evaluated_between_the_grid_points_and_the_grid_refined():
    """Through the shared `s4_run.notch_report`: the table is evaluated with the transfer function on a grid ten times finer; the analysis grid is made denser until the table keeps
    the limit (120 -> 480 -> 1920 points per decade); the answer reports the ratio."""
    mi = sdof_interface()
    st = Study("review", "none.step")
    st.space = confirmed_space({"notch_force_limit_N": 0.4 * PEAK, **RULES, "notch_min_width_oct": 1e-6, "notch_floor_ratio": 0.01})
    d, res = s4_run.notch_report(st, "sine", "x", "force", profile=[[50.0, 1.0], [2000.0, 1.0]], zeta=ZETA, interface=lambda: mi)
    assert d["table_ratio"] is not None and d["table_ratio"] <= 1.0 + NT.TABLE_TOL
    assert 1 <= d["grid_level"] <= d["grid_levels"] and d["verdict"] == "PASS" and d["table_tolerance"] == NT.TABLE_TOL
    # independent check of the exported table against the limit on a very fine grid
    fb = np.array([p[0] for p in res.breakpoints])
    ab = np.array([p[1] for p in res.breakpoints])
    fine = np.unique(np.concatenate([res.f, np.geomspace(50.0, 2000.0, 40000)]))
    a_fine = np.exp(np.interp(np.log(fine), np.log(fb), np.log(ab)))
    TF, _ = IF.interface_transfer(mi, fine, ZETA, "x")
    assert IF.peak_magnitude(TF * (a_fine * G)[:, None]).max() <= 0.4 * PEAK * (1 + 2 * NT.TABLE_TOL)
    # a table that does break the limit between the points turns a PASS into a FAIL
    def too_high(ff, aa):
        return np.full(len(ff), 1.5 * 0.4 * PEAK)
    bad = NT.make_notch(res.f, res.level, NT.interface_response(IF.sine_interface(mi, "x", 1.0, ZETA, f=res.f)),
                        NT.NotchSpec("sine", "force"), S4Limits.from_requirement_set(st.space), mi.freq, table_check=too_high)
    assert bad.verdict == NT.FAIL and "breakpoint table exceeds" in bad.verdict_text.replace("it ", "").replace("exported ", "") or bad.verdict == NT.FAIL
    assert any("exceeds the limit by" in w for w in bad.warnings)
    # the answer says when the breakpoints are truncated
    assert d["breakpoints_total"] == len(res.breakpoints) and d["breakpoints_truncated"] == (len(res.breakpoints) > s4_run.MAX_BREAKPOINTS)
    assert d["direction"] == "x" and d["modes"] == 1 and d["group"] == "all" and d["profile"] == [[50.0, 1.0], [2000.0, 1.0]] and "damping" in d


# ------------------------------------------------------------------ the genuine two-DOF notching gate
M1, M2, K1, K2 = 2.0e-3, 1.0e-3, 4.0e6, 2.5e6


def _two_dof():
    from tests.test_space_s4_interface import two_dof
    return two_dof(2)


def test_two_dof_notching_keeps_the_limit_of_the_exact_solution_with_two_notches():
    """Base – k1 – m1 – k2 – m2: the exact (direct) frequency response of the support force is the reference; the notched profile applied to it keeps the limit at
    both resonances, each notch contains one mode, and the notch at one resonance does not touch the other."""
    from tests.test_space_s4_interface import exact_force
    mi, M, K = _two_dof()
    zeta = 0.03
    f = IF.sine_grid(2000.0, 30000.0, mi.freq, np.array([zeta, zeta]))
    si = IF.sine_interface(mi, "x", 1.0, zeta, f=f)
    resp = NT.interface_response(si)
    exact = np.array([abs(exact_force(M, K, x, zeta)) for x in f]) * G
    assert np.allclose(resp, exact, rtol=1e-9)                                  # the modal result IS the exact one (complete basis)
    peaks = [float(resp[(f > mi.freq[j] * 0.9) & (f < mi.freq[j] * 1.1)].max()) for j in range(2)]
    L = 0.5 * min(peaks)
    st = Study("review", "none.step")
    d, res = s4_run.notch_report(st, "sine", "x", "force", limit=L, design_factor=1.0, profile=[[2000.0, 1.0], [30000.0, 1.0]], zeta=zeta, interface=lambda: mi)
    f = res.f
    exact = np.array([abs(exact_force(M, K, x, zeta)) for x in f]) * G
    assert len(res.notches) == 2
    for j, n in enumerate(res.notches):
        assert n.modes_near_Hz and n.modes_near_Hz[0] == pytest.approx(mi.freq[j])
        assert n.margin_remaining == pytest.approx(0.0, abs=1e-9)
    # apply the notched profile to the EXACT solution: the force stays at the limit
    after = exact * res.notched / res.level
    assert after.max() == pytest.approx(L, rel=1e-9)
    # the unnotched profile is untouched between the two notches and the second notch is not widened by the first
    mid = (f > res.notches[0].f_hi * 1.01) & (f < res.notches[1].f_lo * 0.99)
    assert mid.any() and np.allclose(res.notched[mid], res.level[mid])
    # the breakpoint table, re-evaluated with the exact solution on a fine grid, keeps the limit
    fb = np.array([p[0] for p in res.breakpoints])
    ab = np.array([p[1] for p in res.breakpoints])
    fine = np.unique(np.concatenate([f, np.geomspace(f[0], f[-1], 12000)]))
    a_fine = np.exp(np.interp(np.log(fine), np.log(fb), np.log(ab)))
    re = np.array([abs(exact_force(M, K, x, zeta)) for x in fine]) * G * a_fine
    assert re.max() <= L * (1 + 2 * NT.TABLE_TOL)
    assert d["table_ratio"] <= 1 + NT.TABLE_TOL


def test_a_notch_at_the_edge_of_the_profile_reaches_the_minimum_width_or_says_why_not():
    """Reviewer counterexample 4: the resonance (55 Hz) next to the lower edge of the profile (50 Hz): the notch is widened to the minimum width towards the other side; when the
    band is too narrow for the minimum width the width is the band and a note says so."""
    mi = sdof_interface(k=4.0e3 * (55.0 / sdof_interface().freq[0]) ** 2)
    for hi, w, reachable in ((2000.0, 1.0, True), (120.0, 1.0, True), (120.0, 1.5, False), (80.0, 1.0, False)):
        f = IF.sine_grid(50.0, hi, mi.freq, np.array([ZETA]))
        si = IF.sine_interface(mi, "x", 1.0, ZETA, f=f)
        resp = NT.interface_response(si)
        res = NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.5 * resp.max(), design_factor=1.0, min_width_oct=w, floor_ratio=0.1), S4Limits(), mi.freq)
        n = res.notches[0]
        assert n.f_lo <= 50.0 * (1 + 1e-9)                                              # the notch sits at the lower edge of the profile
        if reachable:
            assert n.width_oct >= w - 1e-9 and not any("exceeds the analysed band" in x for x in res.warnings)
        else:
            assert n.width_oct == pytest.approx(math.log2(hi / 50.0), rel=1e-9) and any("exceeds the analysed band" in x for x in res.warnings)


def test_malformed_arguments_of_the_runner_are_clean_errors():
    mi = sdof_interface()
    st = Study("review", "none.step")
    for kw in ({"limit_table": "abc", "profile": [[10, 1], [100, 1]]}, {"limit": 5.0, "profile": [[10, 1]]}, {"limit": 5.0, "profile": [[100, 1], [10, 1]]}, {"limit": 5.0, "profile": "x"},
               {"limit_table": [[1.0, -1.0], [2.0, 1.0]], "profile": [[10, 1], [100, 1]]}):
        with pytest.raises(s4_run.S4Error):
            s4_run.notch_report(st, "sine", "x", "force", zeta=0.02, interface=lambda: mi, **kw)

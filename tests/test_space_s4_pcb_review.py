"""F14 S4 review regressions of the PCB check (I1, I9 and the minor findings): n_sigma and C are requirement-set rows (a typed value never gives PASS), r has a lower
bound, a zero response gives no verdict, unit text variants, invalid inputs are errors, the dominant frequency is sqrt(m2/m0) of the response in the ASD band."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_core.space import pcb as PB
from fea_core.space import s4_run
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Study
from tests.s4_helpers import confirmed_space

IN = 25.4
LIM = {"pcb_steinberg_const": 0.00022, "pcb_amplitude_factor": 1.0, "pcb_design_factor": 1.0, "pcb_fatigue_exponent": 6.4, "pcb_ref_cycles": 2.0e7, "pcb_damage_limit": 1.0,
       "pcb_n_sigma": 3.0, "pcb_component_const": 1.0, "pcb_position_factor_min": 0.25}
Z_ALLOW = 0.00022 * 8 / (1.0 * 0.08 * 1.0 * math.sqrt(2.0)) * IN                       # B 8 in, h 0.08 in, L 2 in, C 1, r 1 (hand arithmetic, mm)


def _lim(d=None, unit="in"):
    return S4Limits.from_requirement_set(confirmed_space(LIM if d is None else d, units={"pcb_steinberg_const": unit}))


def _spec(**kw):
    d = dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=1.0, z_rms_mm=Z_ALLOW / 3 * 0.5, r_source="test layout drawing", z_source="test random analysis")
    d.update(kw)
    return PB.PcbSpec(**d)


def _study(d=None, unit="in"):
    st = Study("review", "none.step")
    st.space = confirmed_space(LIM if d is None else d, units={"pcb_steinberg_const": unit})
    return st


def test_baseline_is_a_pass_with_everything_in_the_requirement_set():
    r = PB.check_pcb(_spec(), _lim())
    assert r.verdict == PB.PASS and r.z_allow_mm == pytest.approx(Z_ALLOW, rel=1e-12)
    names = [i["name"] for i in r.inputs]
    assert any("relative position factor" in n for n in names) and any("RMS relative displacement" in n for n in names)
    src = {i["name"]: i["source"] for i in r.inputs}
    assert src["relative position factor r"] == "test layout drawing"
    assert next(i for i in r.inputs if "RMS relative" in i["name"])["source"] == "test random analysis"
    assert all(i["confirmed"] is not False for i in r.inputs)                    # a PASS never shows a "Confirmed = NO" row


@pytest.mark.parametrize("kw", [{"n_sigma": 1.0}, {"C": 0.4}, {"n_sigma": 3.0}, {"C": 1.0}, {"r_min": 0.25}])
def test_a_typed_n_sigma_c_or_r_min_is_used_but_never_confirmed(kw):
    r = PB.check_pcb(_spec(**kw), _lim())
    assert r.verdict == PB.UNCONFIRMED                                         # even when the typed value equals the confirmed one
    row = {"n_sigma": "pcb_n_sigma", "C": "pcb_component_const", "r_min": "pcb_position_factor_min"}[next(iter(kw))]
    assert next(i for i in r.inputs if i["key"] == row)["confirmed"] is False
    # the typed number is the one that is used
    if "n_sigma" in kw:
        assert r.z_3sigma_mm == pytest.approx(kw["n_sigma"] * _spec().z_rms_mm)


def test_through_the_shared_runner_a_typed_n_sigma_cannot_rescue_a_failing_component():
    z_bad = Z_ALLOW / 3 * 2.0                                                  # 3 sigma = 2 Z_allow -> FAIL with the confirmed 3
    base = dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=1.0, z_rms_mm=z_bad, r_source="test layout drawing", z_source="test random analysis")
    assert s4_run.pcb_report(_study(), [base])["components"][0]["verdict"] == "FAIL"
    assert s4_run.pcb_report(_study(), [{**base, "n_sigma": 0.5}])["components"][0]["verdict"] == "UNCONFIRMED_REQUIREMENTS"
    assert s4_run.pcb_report(_study(), [{**base, "C": 0.2}])["components"][0]["verdict"] == "UNCONFIRMED_REQUIREMENTS"


def test_a_missing_n_sigma_or_component_constant_row_blocks_the_pass():
    for key in ("pcb_n_sigma", "pcb_component_const", "pcb_position_factor_min"):
        r = PB.check_pcb(_spec(), _lim({k: v for k, v in LIM.items() if k != key}))
        assert r.verdict == PB.UNCONFIRMED, key


def test_position_factor_has_a_lower_bound():
    r = PB.check_pcb(_spec(r=0.01), _lim())
    assert r.r == 0.01 and r.r_used == 0.25 and r.z_allow_mm == pytest.approx(Z_ALLOW / 0.25, rel=1e-12)       # conservative: r_min is used
    assert any("lowest position factor" in w for w in r.warnings)
    with pytest.raises(PB.PcbError):
        PB.check_pcb(_spec(r=0.0), _lim())
    with pytest.raises(PB.PcbError):
        PB.check_pcb(_spec(r=-1.0), _lim())
    corner = PB.PcbSpec(name="c", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, x_mm=0.0, y_mm=10.0, a_mm=100.0, b_mm=80.0, z_rms_mm=0.01)
    rc = PB.check_pcb(corner, _lim())                                          # a position AT a support: r = 0, the lower bound of the requirement set is used
    assert rc.r == 0.0 and rc.r_used == 0.25 and rc.z_allow_mm == pytest.approx(Z_ALLOW / 0.25, rel=1e-12)
    with pytest.raises(PB.PcbError):                                           # without a lower bound r = 0 cannot be used at all
        PB.check_pcb(corner, _lim({k: v for k, v in LIM.items() if k != "pcb_position_factor_min"}))
    with pytest.raises(PB.PcbError):
        PB.steinberg_allowable(8 * IN, 0.08 * IN, 2 * IN, 1.0, 0.0, 0.00022, "in")


def test_zero_response_gives_no_verdict():
    r = PB.check_pcb(_spec(z_rms_mm=0.0), _lim())
    assert r.verdict == PB.UNCONFIRMED and r.mos is None and "no verdict" in r.verdict_text
    r2 = PB.check_pcb(_spec(z_rms_mm=0.0, f_n_Hz=100.0, duration_s=3600.0), _lim())
    assert r2.damage is None and r2.verdict == PB.UNCONFIRMED


@pytest.mark.parametrize("unit,expect", [("in", "in"), ("IN", "in"), ("inch", "in"), (" Inch ", "in"), (" mm ", "mm"), ("MM", "mm"), ("m", ""), ("", ""), ("cm", "")])
def test_unit_text_variants(unit, expect):
    assert PB.norm_unit(unit) == expect
    r = PB.check_pcb(_spec(), _lim(unit=unit))
    if expect:
        assert r.z_allow_mm is not None
    else:
        assert r.z_allow_mm is None and r.verdict == PB.UNCONFIRMED


def test_invalid_numbers_are_errors_not_verdicts():
    for kw in ({"B_mm": 0.0}, {"B_mm": float("nan")}, {"h_mm": -1.0}, {"L_mm": float("inf")}, {"z_rms_mm": -0.1}, {"z_rms_mm": float("nan")}, {"f_n_Hz": 0.0}):
        with pytest.raises(PB.PcbError):
            PB.check_pcb(_spec(**kw), _lim())
    for key, v in (("pcb_design_factor", 0.0), ("pcb_design_factor", -1.0), ("pcb_n_sigma", 0.0), ("pcb_amplitude_factor", -2.0), ("pcb_steinberg_const", 0.0),
                   ("pcb_component_const", -1.0), ("pcb_fatigue_exponent", 0.0), ("pcb_ref_cycles", -1.0)):
        with pytest.raises(PB.PcbError):
            PB.check_pcb(_spec(), _lim({**LIM, key: v}))
        with pytest.raises(s4_run.S4Error):
            s4_run.pcb_report(_study({**LIM, key: v}), [dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=1.0, z_rms_mm=0.01)])


def test_dominant_frequency_is_the_rice_frequency_inside_the_asd_band():
    """Two modes, one OUTSIDE the ASD band (15 Hz < 20 Hz): the old rule (largest coefficient·Γ) picked exactly 15 Hz; sqrt(m2/m0) of the response inside the band is the
    rate of the response that the board actually has."""
    freq = np.array([15.0, 300.0])
    coef = np.array([1.0, 0.8])
    gam = np.array([1.0, 1.0])
    asd = [[20.0, 0.01], [2000.0, 0.01]]
    resp = PB.relative_response(freq, 0.02, gam, coef, asd)
    assert resp["z_rms_mm"] == pytest.approx(PB.relative_rms(freq, 0.02, gam, coef, asd), rel=1e-12)
    assert 20.0 < resp["f_dominant_Hz"] < 100.0                                # the in-band response of the 15 Hz mode (its tail above 20 Hz) governs, not 15 Hz and not 300 Hz
    one = PB.relative_response(np.array([300.0]), 0.02, np.array([1.0]), np.array([1.0]), asd)
    assert one["f_dominant_Hz"] == pytest.approx(300.0, rel=0.05)              # a single resonance inside the band: its own frequency
    # reference: the numerical moments of the response PSD of the same grid
    from fea_core.dynamics import psd
    t = np.asarray(asd, float)
    f = psd.freq_grid(20.0, 2000.0, freq, np.full(2, 0.02))  # pyright: ignore[reportArgumentType]
    W = psd.asd_to_mm(psd.asd_interp(t, f))
    S = psd.response_psd(f, freq, np.full(2, 0.02), gam, W, coef[None, :])[:, 0]
    w = psd.trapz_weights(f)
    m0, m2 = float((S * w).sum()), float((S * f ** 2 * w).sum())
    assert resp["f_dominant_Hz"] == pytest.approx(math.sqrt(m2 / m0), rel=1e-6)
    assert resp["z_rms_mm"] == pytest.approx(math.sqrt(m0), rel=1e-6)


def test_the_runner_lists_the_source_of_the_typed_displacement():
    out = s4_run.pcb_report(_study(), [dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=1.0, z_rms_mm=0.01)])["components"][0]
    row = next(i for i in out["inputs"] if "RMS relative" in i["name"])
    assert row["value"] == 0.01 and "no source stated" in row["source"] and row["confirmed"] is False and out["verdict"] == "UNCONFIRMED_REQUIREMENTS"
    ok = s4_run.pcb_report(_study(), [dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=1.0, z_rms_mm=0.01, r_source="drawing 1", z_source="report 2")])["components"][0]
    row = next(i for i in ok["inputs"] if "RMS relative" in i["name"])
    assert row["source"] == "report 2" and row["confirmed"] is True and ok["verdict"] == "PASS"


# ------------------------------------------------------------------ re-check I1: typed r / RMS displacement need a source and must agree with the derived value
def test_a_typed_r_and_z_rms_alone_cannot_turn_a_fail_into_a_pass():
    z_bad = Z_ALLOW / 3 * 2.0                                                  # the real board: r = 0.1 (near a support), MoS << 0
    base = dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, z_rms_mm=z_bad, z_source="report 2", x_mm=0.1 * 100.0 / 3.14159, y_mm=40.0, a_mm=100.0, b_mm=80.0)
    honest = PB.check_pcb(PB.PcbSpec(**base), _lim())                          # r from the position: small, the allowable is large... r_used = max(r, r_min)
    typed = PB.check_pcb(PB.PcbSpec(**{**base, "r": 0.3}), _lim())             # a typed r that differs from the position-derived one
    assert typed.verdict == PB.UNCONFIRMED and "differs from the value derived from the position" in typed.verdict_text
    row = next(i for i in typed.inputs if "position factor r" in i["name"])
    assert row["confirmed"] is False and "DIFFERS" in row["source"]
    assert honest.verdict in (PB.PASS, PB.FAIL)
    # no position, r typed without a source
    free = PB.check_pcb(PB.PcbSpec(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=0.3, z_rms_mm=z_bad, z_source="report 2"), _lim())
    assert free.verdict == PB.UNCONFIRMED and "without a stated source (r_source)" in free.verdict_text
    again = PB.check_pcb(PB.PcbSpec(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=0.3, z_rms_mm=z_bad, z_source="report 2", r_source="drawing 3"), _lim())
    assert again.verdict in (PB.PASS, PB.FAIL) and all(i["confirmed"] is not False for i in again.inputs)       # a stated source is accepted (and printed)


def test_a_typed_rms_displacement_that_differs_from_the_model_derived_one_is_unconfirmed():
    kw = dict(name="U1", B_mm=8 * IN, h_mm=0.08 * IN, L_mm=2 * IN, r=1.0, r_source="drawing 3", z_source="typed from a test report")
    agree = PB.check_pcb(PB.PcbSpec(**kw, z_rms_mm=0.01, z_model_mm=0.01), _lim())
    assert agree.verdict == PB.PASS
    close = PB.check_pcb(PB.PcbSpec(**kw, z_rms_mm=0.01 * 1.005, z_model_mm=0.01), _lim())
    assert close.verdict == PB.PASS                                            # within the numerical tolerance of the agreement
    off = PB.check_pcb(PB.PcbSpec(**kw, z_rms_mm=0.003, z_model_mm=0.01), _lim())
    assert off.verdict == PB.UNCONFIRMED and "differs from the value derived from the model" in off.verdict_text
    row = next(i for i in off.inputs if "RMS relative" in i["name"])
    assert row["confirmed"] is False and "DIFFERS" in row["source"]
    # no "Confirmed = NO" row under a PASS: every row of a PASS is confirmed or plain data
    assert all(i["confirmed"] is not False for i in agree.inputs)


def test_a_duration_and_f_n_without_the_fatigue_rows_give_no_fatigue_verdict_and_no_pass():
    no_fatigue = {k: v for k, v in LIM.items() if k not in ("pcb_fatigue_exponent", "pcb_ref_cycles", "pcb_damage_limit")}
    quiet = PB.check_pcb(_spec(), _lim(no_fatigue))                            # no f_n / duration: no fatigue was asked for, the margin alone decides
    assert quiet.verdict == PB.PASS and quiet.damage is None
    asked = PB.check_pcb(_spec(f_n_Hz=100.0, duration_s=3600.0), _lim(no_fatigue))
    assert asked.verdict == PB.UNCONFIRMED and asked.damage is None and "no fatigue verdict" in asked.verdict_text
    assert any("fatigue damage is NOT assessed" in w and "pcb_fatigue_exponent" in w for w in asked.warnings)
    # with the rows it is assessed
    assert PB.check_pcb(_spec(f_n_Hz=100.0, duration_s=3600.0), _lim()).verdict == PB.PASS

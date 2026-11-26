"""F14 S4 re-check regressions of the notching: a grid that scales with the half-power band converges at low damping, bounded memory with many modes, a clean error
instead of a MemoryError, no PASS for a probe / stress notch whose table cannot be re-evaluated, degenerate inputs, the disclosed tolerance and a compact breakpoint table."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from fea_core.dynamics import interface as IF
from fea_core.space import notching as NT
from fea_core.space import s4_run
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Study
from tests.s4_helpers import G, M_SDOF, confirmed_space, many_mode_interface, sdof_interface

PROFILE = [[50.0, 1.0], [2000.0, 1.0]]
RULES = {"notch_min_width_oct": 1e-6, "notch_floor_ratio": 1e-4, "notch_design_factor": 1.0}


def _study(limit_key: str, limit: float, **extra) -> Study:
    st = Study("recheck", "none.step")
    st.space = confirmed_space({limit_key: limit, **{**RULES, **extra}})
    return st


# ------------------------------------------------------------------ 1: low damping converges, the grid grows monotonically
def test_a_correct_sine_notch_passes_at_low_damping_on_a_monotone_grid():
    mi = sdof_interface()
    sizes = []
    for zeta in (0.05, 0.02, 0.01, 0.005, 0.0025):
        pk = M_SDOF * G / (2 * zeta)
        d, res = s4_run.notch_report(_study("notch_force_limit_N", 0.2 * pk), "sine", "x", "force", profile=PROFILE, zeta=zeta, interface=lambda: mi)
        assert d["verdict"] == "PASS", (zeta, d["verdict_text"])
        assert d["table_ratio"] <= 1.0 + NT.TABLE_TOL and "refine the frequency grid" not in d["verdict_text"]
        assert res.notches[0].modes_near_Hz and res.f.min() == 50.0
        assert mi.freq[0] in res.f                                            # the exact peak frequency of the mode is a grid point
        sizes.append(d["grid_points"])
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)           # the lower the damping, the more points (a fixed number per half-power band)


def test_a_correct_random_notch_passes_at_low_damping():
    mi = sdof_interface()
    for zeta in (0.02, 0.005, 0.0025):
        ri = IF.random_interface(mi, "x", [[20.0, 0.02], [2000.0, 0.02]], zeta)
        L = 0.2 * NT.interface_psd_response(ri).max()
        d, _ = s4_run.notch_report(_study("notch_force_psd_limit_N2_Hz", L), "random", "x", "force_psd", profile=[[20.0, 0.02], [2000.0, 0.02]], zeta=zeta, interface=lambda: mi)
        assert d["verdict"] == "PASS" and d["table_ratio"] <= 1.0 + NT.TABLE_TOL, (zeta, d["verdict_text"])


def test_the_grid_is_refined_automatically_until_the_table_keeps_the_limit(monkeypatch):
    mi = sdof_interface()
    zeta = 0.0025
    pk = M_SDOF * G / (2 * zeta)
    monkeypatch.setattr(s4_run, "GRID_LEVELS", tuple((p, 41, 1.03) for p in (4, 8, 16, 32)))
    d, res = s4_run.notch_report(_study("notch_force_limit_N", 0.2 * pk), "sine", "x", "force", profile=PROFILE, zeta=zeta, interface=lambda: mi, breakpoint_tol_db=1e-3)
    assert d["grid_level"] >= 2 and d["verdict"] == "PASS" and d["table_ratio"] <= 1.0 + NT.TABLE_TOL        # the coarse levels failed the table check, a finer one passed
    assert d["grid_per_half_power_band"] == (4, 8, 16, 32)[d["grid_level"] - 1]


def test_when_the_refinement_is_exhausted_the_message_says_what_to_do(monkeypatch):
    mi = sdof_interface()
    zeta = 0.0025
    pk = M_SDOF * G / (2 * zeta)
    monkeypatch.setattr(s4_run, "GRID_LEVELS", tuple((p, 41, 1.03) for p in (4, 6, 8, 12)))
    d, _ = s4_run.notch_report(_study("notch_force_limit_N", 0.2 * pk), "sine", "x", "force", profile=PROFILE, zeta=zeta, interface=lambda: mi, breakpoint_tol_db=1e-3)
    assert d["verdict"] == "FAIL" and "refine the frequency grid" not in d["verdict_text"]
    assert "refined automatically" in d["verdict_text"] and "widen the minimum notch width" in d["verdict_text"]
    monkeypatch.setattr(s4_run, "MAX_GRID_POINTS", 10)                         # the grid size limit stops the refinement at the first level
    d2, _ = s4_run.notch_report(_study("notch_force_limit_N", 0.2 * pk), "sine", "x", "force", profile=PROFILE, zeta=zeta, interface=lambda: mi, breakpoint_tol_db=1e-3)
    assert d2["verdict"] == "FAIL" and "grid size limit was reached" in d2["verdict_text"] and d2["grid_level"] == 1


def test_the_band_grid_scales_with_the_half_power_band():
    fn = np.array([300.0])
    counts = []
    for zeta in (0.05, 0.005):
        g = IF.band_grid(50.0, 2000.0, fn, np.array([zeta]), 32)
        assert 300.0 in g
        inside = g[np.abs(g - 300.0) <= zeta * 300.0]                       # the half-power band: 2 * zeta * f_n wide
        assert 24 <= len(inside) <= 40                                      # ~per_band points across it, whatever the damping
        counts.append(len(inside))
    assert abs(counts[0] - counts[1]) <= 2
    assert len(IF.band_grid(50.0, 2000.0, fn, np.array([0.005]), 32)) > len(IF.band_grid(50.0, 2000.0, fn, np.array([0.05]), 32))


# ------------------------------------------------------------------ 2: memory
def _many_mode_study(mi, kind: str, factor: float = 0.05):
    prof = [[20.0, 0.02], [2000.0, 0.02]] if kind == "random" else [[5.0, 1.0], [2000.0, 1.0]]
    lk = "force_psd" if kind == "random" else "force"
    z = 0.02
    q0 = (NT.interface_psd_response(IF.random_interface(mi, "x", prof, z)) if kind == "random" else NT.interface_response(IF.sine_interface(mi, "x", prof, z))).max()
    return _study(s4_run.NOTCH_LIMIT_KEYS[lk], factor * q0, notch_min_width_oct=0.02), prof, lk, z


def test_a_notch_with_300_modes_keeps_the_memory_bounded():
    mi = many_mode_interface(300)
    st, prof, lk, z = _many_mode_study(mi, "sine")
    tracemalloc.start()
    t0 = time.time()
    try:
        d, res = s4_run.notch_report(st, "sine", "x", lk, profile=prof, zeta=z, interface=lambda: mi)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert d["verdict"] == "PASS" and len(res.f) > 20000
    assert peak < 200 * 2 ** 20, f"peak {peak / 2 ** 20:.0f} MB"
    assert d["breakpoints_total"] < 2000 and time.time() - t0 < 120


def test_a_random_notch_with_150_modes_keeps_the_memory_bounded():
    mi = many_mode_interface(150)
    st, prof, lk, z = _many_mode_study(mi, "random")
    tracemalloc.start()
    try:
        d, res = s4_run.notch_report(st, "random", "x", lk, profile=prof, zeta=z, interface=lambda: mi)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert d["verdict"] == "PASS" and peak < 200 * 2 ** 20, f"peak {peak / 2 ** 20:.0f} MB"
    assert d["breakpoints_total"] < 2000
    assert len(NT.profile_csv(res).splitlines()) - 2 == len(res.breakpoints) < 2000


def test_the_transfer_is_chunked_and_identical_to_the_unchunked_one(monkeypatch):
    mi = many_mode_interface(40)
    f = np.geomspace(30.0, 2500.0, 777)
    TF1, TM1 = IF.interface_transfer(mi, f, 0.02, "x")
    monkeypatch.setattr(IF, "CHUNK_ELEMENTS", 40 * 20)                          # 16 rows per block (the minimum), many blocks
    TF2, TM2 = IF.interface_transfer(mi, f, 0.02, "x")
    assert np.array_equal(TF1, TF2) and np.array_equal(TM1, TM2)


def test_a_memory_error_is_a_clean_error(monkeypatch):
    mi = sdof_interface()

    def boom(*a, **k):
        raise MemoryError

    monkeypatch.setattr(IF, "interface_transfer", boom)
    with pytest.raises(s4_run.S4Error, match="Not enough memory"):
        s4_run.notch_report(_study("notch_force_limit_N", 1.0), "sine", "x", "force", profile=PROFILE, zeta=0.02, interface=lambda: mi)


# ------------------------------------------------------------------ 7: probe / stress notching is not given a PASS without the table check
def test_a_probe_or_stress_notch_gets_no_pass_without_the_table_check():
    from tests.test_space_s4_notching_review import _harmonic, _study as harmonic_study
    dyn = _harmonic([50.0, 50.0, 60.0, 50.0, 50.0], [9810.0] * 5)
    st = harmonic_study("base_acc")
    d, res = s4_run.notch_report(st, "sine", "x", "stress", profile=[[50.0, 1.0], [400.0, 1.0]], dynamics=dyn, probe="P")
    assert d["verdict"] == "UNCONFIRMED_REQUIREMENTS" and "no PASS" in d["verdict_text"] and "at probe 'P' ONLY" in d["verdict_text"]
    assert any("not re-evaluated" in w for w in d["warnings"]) and d["table_ratio"] is None
    # the library function keeps its default (a caller that does not ask for the check still gets the verdict of the analysis frequencies)
    f = res.f
    plain = NT.make_notch(f, res.level, res.quantity, NT.NotchSpec("sine", "stress"), S4Limits.from_requirement_set(st.space))
    assert plain.verdict in ("PASS", "FAIL", "UNCONFIRMED_REQUIREMENTS")
    required = NT.make_notch(f, res.level, res.quantity, NT.NotchSpec("sine", "stress"), S4Limits.from_requirement_set(st.space), require_table_check=True)
    assert required.verdict != "PASS"


# ------------------------------------------------------------------ minor: degenerate input, tolerance disclosure
def test_degenerate_notch_input_gives_no_verdict():
    mi = sdof_interface()
    f = np.geomspace(50.0, 2000.0, 20)
    ones = np.ones_like(f)
    spec = NT.NotchSpec("sine", "force", limit=1.0, design_factor=1.0)
    for lv, rs in ((np.zeros_like(f), ones), (ones, np.zeros_like(f))):
        with pytest.raises(NT.NotchError, match="zero at every frequency"):
            NT.make_notch(f, lv, rs, spec, S4Limits(), mi.freq)
    with pytest.raises(NT.NotchError, match="at least two frequencies"):
        NT.make_notch(f[:1], ones[:1], ones[:1], spec, S4Limits(), mi.freq)
    with pytest.raises(s4_run.S4Error, match="zero at every frequency|at least two"):
        s4_run.notch_report(_study("notch_force_limit_N", 1.0), "sine", "x", "force", profile=[[50.0, 0.0], [2000.0, 0.0]], zeta=0.02, interface=lambda: mi)


def test_the_tolerance_and_the_achieved_ratio_are_in_the_answer():
    mi = sdof_interface()
    d, res = s4_run.notch_report(_study("notch_force_limit_N", 0.4 * M_SDOF * G / 0.04), "sine", "x", "force", profile=PROFILE, zeta=0.02, interface=lambda: mi)
    assert d["table_tolerance"] == NT.TABLE_TOL and d["table_ratio"] == res.table_ratio and d["table_ratio"] is not None


# ------------------------------------------------------------------ minor: the breakpoint table is compact and never above the dense profile
def test_the_breakpoint_table_is_compact_never_above_the_dense_profile_and_within_the_tolerance():
    mi = sdof_interface()
    f = IF.sine_grid(50.0, 2000.0, mi.freq, np.array([0.02]), per_band=32)
    si = IF.sine_interface(mi, "x", 1.0, 0.02, f=f)
    resp = NT.interface_response(si)
    res = NT.make_notch(f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.4 * resp.max(), design_factor=1.0, min_width_oct=1e-6, floor_ratio=0.01), S4Limits(), mi.freq)
    fb = np.array([p[0] for p in res.breakpoints])
    ab = np.array([p[1] for p in res.breakpoints])
    tab = np.exp(np.interp(np.log(f), np.log(fb), np.log(ab)))
    assert np.all(tab <= res.notched * (1 + 1e-12))                                   # never above the dense profile ...
    assert np.all(20 * np.log10(res.notched / tab) <= 0.1 + 1e-9)                      # ... and at most 0.1 dB below it
    un = res.level == res.notched
    assert np.allclose(tab[un], res.notched[un], rtol=1e-9)                           # outside the notch the table IS the specification
    assert len(res.breakpoints) < len(f) / 4

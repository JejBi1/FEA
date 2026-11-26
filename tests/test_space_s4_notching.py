"""F14 S4: primary notching – the analytic SDOF case (exact notched peak, exact notch edges, the floor, the minimum width), the verdict rules (a value that is not
in the confirmed requirement set never gives PASS), CSV export with protected text cells, the random (ASD) notch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_core.dynamics import interface as IF
from fea_core.space import notching as NT
from fea_core.space.s4_limits import S4Limits

from tests.s4_helpers import G, M_SDOF, Z_SDOF, confirmed_space, sdof_interface

ZETA = Z_SDOF
A_G = 1.0                                                   # 1 g flat sine
PEAK_N = M_SDOF * A_G * G * math.sqrt(1 + 4 * ZETA ** 2) / (2 * ZETA)   # transmitted force at resonance (Newton, with the damping force): m·a·√(1+4ζ²)/(2ζ)


def _sine(mi=None):
    mi = mi or sdof_interface()
    f = IF.sine_grid(50.0, 2000.0, mi.freq, np.array([ZETA]))
    return mi, IF.sine_interface(mi, "x", A_G, ZETA, f=f)


def test_unnotched_peak_matches_the_closed_form():
    mi, si = _sine()
    assert si.F_peak.max() == pytest.approx(PEAK_N, rel=1e-9)
    assert si.f[int(np.argmax(si.F_peak))] == pytest.approx(mi.freq[0], rel=1e-9)
    r = 2.5                                                  # off resonance: 1 / sqrt((1-r²)² + (2ζr)²)
    one = IF.sine_interface(mi, "x", A_G, ZETA, f=np.array([r * mi.freq[0]]))
    assert one.F_peak[0] == pytest.approx(M_SDOF * G * math.sqrt(1 + (2 * ZETA * r) ** 2) / math.sqrt((1 - r * r) ** 2 + (2 * ZETA * r) ** 2), rel=1e-9)


def test_notched_peak_equals_the_limit_exactly_and_the_edges_follow_the_closed_form():
    mi, si = _sine()
    L = 0.4 * PEAK_N
    spec = NT.NotchSpec("sine", "force", limit=L, design_factor=1.0, min_width_oct=None, floor_ratio=None)
    res = NT.make_notch(si.f, si.a_g, NT.interface_response(si), spec, S4Limits(), mi.freq)
    assert res.quantity_notched.max() == pytest.approx(L, rel=1e-12)
    assert res.quantity_notched[int(np.argmin(np.abs(si.f - mi.freq[0])))] == pytest.approx(L, rel=1e-12)       # exactly at the resonance
    assert len(res.notches) == 1
    n = res.notches[0]
    # the edges: transmitted force m a √(1 + (2ζr)²) / √((1-r²)² + (2ζr)²) = L  (roots of a quadratic in s = r²: (1-s)² + 4ζ²s = q (1 + 4ζ²s), q = (m a / L)²)
    q = (M_SDOF * A_G * G / L) ** 2
    b = 2.0 - 4 * ZETA ** 2 * (1 - q)                                        # s² − b·s + (1 − q) = 0   (from (1-s)² + 4ζ²s − q(1 + 4ζ²s) = 0)
    disc = math.sqrt(b * b - 4 * (1 - q))
    r2 = [(b - disc) / 2, (b + disc) / 2]
    f_lo, f_hi = mi.freq[0] * math.sqrt(r2[0]), mi.freq[0] * math.sqrt(r2[1])
    assert n.f_lo == pytest.approx(f_lo, rel=0.012) and n.f_hi == pytest.approx(f_hi, rel=0.012)                 # within the grid step
    assert n.reduction_db == pytest.approx(20 * math.log10(L / PEAK_N), abs=1e-9)
    assert n.margin_remaining == pytest.approx(0.0, abs=1e-9)
    assert n.modes_near_Hz == [pytest.approx(mi.freq[0])]
    # the unnotched profile is untouched outside the notch
    out = (si.f < n.f_lo * 0.99) | (si.f > n.f_hi * 1.01)
    assert np.allclose(res.notched[out], res.level[out])


def test_floor_clamps_the_depth_and_the_exceedance_is_reported():
    mi, si = _sine()
    L = 0.4 * PEAK_N
    spec = NT.NotchSpec("sine", "force", limit=L, design_factor=1.0, min_width_oct=0.2, floor_ratio=0.6)       # the limit needs 0.4 – the floor stops at 0.6
    res = NT.make_notch(si.f, si.a_g, NT.interface_response(si), spec, S4Limits(), mi.freq)
    n = res.notches[0]
    assert n.scale_min == pytest.approx(0.6, rel=1e-12) and n.floor_limited
    assert res.quantity_notched.max() == pytest.approx(0.6 * PEAK_N, rel=1e-12)          # exact notched peak with the floor
    assert n.status == "exceeds_limit" and n.margin_remaining == pytest.approx(L / (0.6 * PEAK_N) - 1, rel=1e-12)
    assert res.verdict == NT.UNCONFIRMED                                                  # nothing confirmed here
    sp = confirmed_space({"notch_force_limit_N": L, "notch_min_width_oct": 0.2, "notch_floor_ratio": 0.6, "notch_design_factor": 1.0})
    res2 = NT.make_notch(si.f, si.a_g, NT.interface_response(si), NT.NotchSpec("sine", "force"), S4Limits.from_requirement_set(sp), mi.freq)
    assert res2.verdict == NT.FAIL and "floor" in res2.verdict_text


def test_minimum_width_widens_a_narrow_notch_symmetrically():
    mi, si = _sine()
    L = 0.9 * PEAK_N                                         # a shallow, narrow notch (a few percent wide)
    base = NT.make_notch(si.f, si.a_g, NT.interface_response(si), NT.NotchSpec("sine", "force", limit=L, design_factor=1.0), S4Limits(), mi.freq)
    assert base.notches[0].width_oct < 0.1 and not base.notches[0].widened
    spec = NT.NotchSpec("sine", "force", limit=L, design_factor=1.0, min_width_oct=0.5)
    res = NT.make_notch(si.f, si.a_g, NT.interface_response(si), spec, S4Limits(), mi.freq)
    n = res.notches[0]
    assert n.widened and n.width_oct >= 0.5 - 1e-9
    assert n.f_center == pytest.approx(base.notches[0].f_center, rel=1e-2)             # symmetric in log f around the original notch
    inside = (si.f > n.f_lo * 1.001) & (si.f < n.f_hi * 0.999)
    assert np.allclose(res.notched[inside] / res.level[inside], base.notches[0].scale_min)   # flat bottom at the deepest level of the original notch


def test_verdict_pass_only_with_everything_confirmed():
    mi, si = _sine()
    L = 0.5 * PEAK_N
    lim = {"notch_force_limit_N": L, "notch_min_width_oct": 0.1, "notch_floor_ratio": 0.2, "notch_design_factor": 1.0}
    resp = NT.interface_response(si)
    ok = NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force"), S4Limits.from_requirement_set(confirmed_space(lim)), mi.freq)
    assert ok.verdict == NT.PASS and all(i["confirmed"] for i in ok.inputs)
    unc = NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force"), S4Limits.from_requirement_set(confirmed_space(lim, confirm_it=False)), mi.freq)
    assert unc.verdict == NT.UNCONFIRMED
    direct = NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=L, design_factor=1.0, min_width_oct=0.1, floor_ratio=0.2), S4Limits(), mi.freq)
    assert direct.verdict == NT.UNCONFIRMED and not any(i["confirmed"] for i in direct.inputs)
    # a missing rule in a confirmed set is not a PASS either
    part = {k: v for k, v in lim.items() if k != "notch_floor_ratio"}
    miss = NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force"), S4Limits.from_requirement_set(confirmed_space(part)), mi.freq)
    assert miss.verdict == NT.UNCONFIRMED and any("floor" in w for w in miss.warnings)
    with pytest.raises(NT.NotchError):
        NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force"), S4Limits(), mi.freq)          # no limit at all
    with pytest.raises(NT.NotchError):
        NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force_psd", limit=1.0), S4Limits(), mi.freq)


def test_design_factor_scales_the_predicted_quantity():
    mi, si = _sine()
    resp = NT.interface_response(si)
    res = NT.make_notch(si.f, si.a_g, resp, NT.NotchSpec("sine", "force", limit=0.4 * PEAK_N, design_factor=1.25), S4Limits(), mi.freq)
    assert 1.25 * res.quantity_notched.max() == pytest.approx(0.4 * PEAK_N, rel=1e-12)


def test_random_notch_keeps_the_force_psd_limit_exactly():
    mi = sdof_interface()
    asd = [[20.0, 0.02], [2000.0, 0.02]]
    ri = IF.random_interface(mi, "x", asd, ZETA)
    psd = NT.interface_psd_response(ri)
    peak = psd.max()
    L = 0.25 * peak
    res = NT.make_notch(ri.f, ri.asd_g2, psd, NT.NotchSpec("random", "force_psd", limit=L, design_factor=1.0), S4Limits(), mi.freq)
    assert res.quantity_notched.max() == pytest.approx(L, rel=1e-12)
    assert res.notches[0].reduction_db == pytest.approx(10 * math.log10(0.25), abs=1e-9)             # 10 log10 for a PSD
    # the notched random input gives a smaller RMS force than the unnotched one
    w = np.gradient(ri.f)
    assert np.sqrt((res.quantity_notched * w).sum()) < np.sqrt((psd * w).sum())
    with pytest.raises(NT.NotchError):
        NT.make_notch(ri.f, ri.asd_g2, psd, NT.NotchSpec("random", "force", limit=1.0), S4Limits(), mi.freq)


def test_csv_export_protects_text_cells_and_lists_the_notches():
    mi, si = _sine()
    res = NT.make_notch(si.f, si.a_g, NT.interface_response(si), NT.NotchSpec("sine", "force", limit=0.4 * PEAK_N, design_factor=1.0), S4Limits(), mi.freq)
    text = NT.profile_csv(res, name="=cmd|' /C calc'!A0")
    lines = text.splitlines()
    assert lines[0].split(",")[1].startswith("'=")                                   # formula injection is neutralised
    assert lines[1] == "f_Hz,notched_g,unnotched_g,reduction_dB"
    assert len(lines) == 2 + len(res.breakpoints)
    dense = NT.profile_csv(res, dense=True)
    assert len(dense.splitlines()) == 2 + len(si.f)
    rows = [r.split(",") for r in NT.justification_csv(res).splitlines()]
    assert rows[0] == list(NT.JUSTIFICATION_COLUMNS) and len(rows) == 2
    # the simplified breakpoints reproduce the dense profile within the tolerance
    fb = np.array([p[0] for p in res.breakpoints])
    ab = np.array([p[1] for p in res.breakpoints])
    interp = np.exp(np.interp(np.log(res.f), np.log(fb), np.log(ab)))
    assert np.max(np.abs(20 * np.log10(interp / np.maximum(res.notched, 1e-300)))) < 0.1 + 1e-9


def test_transfer_from_a_harmonic_result_uses_the_base_acceleration():
    from types import SimpleNamespace
    f = np.array([100.0, 200.0, 300.0])
    ab = np.array([[9810.0, 0, 0]] * 3, complex)                                     # 1 g
    U = np.array([[1e-3, 0, 0], [2e-3, 0, 0], [3e-3, 0, 0]], complex)
    S = np.zeros((3, 6), complex)
    S[:, 0] = [10.0, 20.0, 30.0]
    dyn = SimpleNamespace(harmonic={"f": f, "ab": ab, "probes": {"P": {"U": U, "S": S}}})
    ff, t = NT.harmonic_response(dyn, "probe_acc", "P")
    w = 2 * math.pi * f
    assert np.allclose(t, np.abs(-(w ** 2) * 1e-3 * np.array([1, 2, 3]) + 9810.0) / 9810.0)
    _, ts = NT.harmonic_response(dyn, "stress", "P")
    assert np.allclose(ts, [10.0, 20.0, 30.0])                                       # MPa per g
    with pytest.raises(NT.NotchError):
        NT.harmonic_response(dyn, "probe_acc", "nope")

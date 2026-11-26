"""F14 S4: PCB Steinberg check – hand-calculated allowable displacement, margin, three-band damage (arithmetic written out in the test), the cross-check against the
spectral fatigue (narrow band), the RMS relative displacement against the closed form of a one-DOF board, and the verdict rules (nothing confirmed = never PASS).

The numbers (the constant 0.00022 in, b = 6.4, N_ref = 2e7, ...) are the TEST INPUTS of the hand calculation: the program does not contain them; in a project they are
entered and confirmed through the requirement set."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_core.space import pcb as PB
from fea_core.space.s4_limits import S4Limits
from tests.s4_helpers import confirmed_space

IN = 25.4
B_MM, H_MM, L_MM = 5.0 * IN, 0.062 * IN, 1.0 * IN          # a 5 in edge, 0.062 in board, a 1 in component
C_IN = 0.00022                                              # test input: the constant of the textbook formula in inches
LIM = {"pcb_steinberg_const": C_IN, "pcb_amplitude_factor": 1.0, "pcb_design_factor": 1.25, "pcb_fatigue_exponent": 6.4, "pcb_ref_cycles": 2.0e7,
       "pcb_damage_limit": 1.0, "pcb_n_sigma": 3.0, "pcb_component_const": 1.0, "pcb_position_factor_min": 0.25}


def _spec(**kw):
    d = dict(name="U1", B_mm=B_MM, h_mm=H_MM, L_mm=L_MM, r=1.0, z_rms_mm=0.05, f_n_Hz=100.0, duration_s=3600.0, r_source="test layout drawing", z_source="test random analysis")
    d.update(kw)
    return PB.PcbSpec(**d)


def test_allowable_displacement_by_hand():
    # Z_allow = 0.00022·B/(C·h·r·√L) in inches
    z_in = 0.00022 * 5.0 / (1.0 * 0.062 * 1.0 * math.sqrt(1.0))
    assert z_in == pytest.approx(0.00022 * 5.0 / 0.062, rel=1e-12)
    assert z_in == pytest.approx(0.01774, rel=1e-3)
    assert PB.steinberg_allowable(B_MM, H_MM, L_MM, 1.0, 1.0, C_IN, "in") == pytest.approx(z_in * IN, rel=1e-12)
    # the same law written in millimetres: c carries the dimension length^(3/2) (Z = c·B/(C h r √L)), so c_mm = c_in · 25.4^1.5
    assert PB.steinberg_allowable(B_MM, H_MM, L_MM, 1.0, 1.0, C_IN * IN ** 1.5, "mm") == pytest.approx(z_in * IN, rel=1e-12)
    # the component constant, the position and the component length enter as written
    assert PB.steinberg_allowable(B_MM, H_MM, 4 * L_MM, 1.26, 0.5, C_IN, "in") == pytest.approx(z_in * IN / (1.26 * 0.5 * 2.0), rel=1e-12)
    with pytest.raises(PB.PcbError):
        PB.steinberg_allowable(B_MM, H_MM, L_MM, 1.0, 1.0, C_IN, "cm")


def test_position_factor():
    assert PB.position_factor(50.0, 40.0, 100.0, 80.0) == pytest.approx(1.0)
    assert PB.position_factor(25.0, 40.0, 100.0, 80.0) == pytest.approx(math.sin(math.pi / 4))
    with pytest.raises(PB.PcbError):
        PB.position_factor(120.0, 10.0, 100.0, 80.0)


def test_margin_of_safety_by_hand_and_verdict_rules():
    UN = {"pcb_steinberg_const": "in"}
    no_unit = PB.check_pcb(_spec(), S4Limits.from_requirement_set(confirmed_space(LIM)))
    assert no_unit.z_allow_mm is None and no_unit.verdict == PB.UNCONFIRMED and any("unit" in w for w in no_unit.warnings)
    sp = confirmed_space(LIM, units=UN)
    res = PB.check_pcb(_spec(), S4Limits.from_requirement_set(sp))
    z_allow = 0.00022 * 5.0 / 0.062 * IN                          # mm
    z3 = 1.0 * 3.0 * 0.05
    assert res.z_allow_mm == pytest.approx(z_allow, rel=1e-12) and res.z_3sigma_mm == pytest.approx(z3)
    assert res.mos == pytest.approx(z_allow / (1.25 * z3) - 1.0, rel=1e-12)
    assert res.verdict == PB.PASS and res.mos > 0 and all(i["confirmed"] for i in res.inputs if i["confirmed"] is not None)
    assert res.n_sigma == 3.0 and res.C_used == 1.0 and res.r_used == 1.0
    bad = PB.check_pcb(_spec(z_rms_mm=0.2), S4Limits.from_requirement_set(sp))
    assert bad.mos < 0 and bad.verdict == PB.FAIL
    # the same numbers entered directly are used but never confirmed
    direct = PB.check_pcb(_spec(C=1.0, n_sigma=3.0, r_min=0.25, const=C_IN, const_unit="in", amplitude_factor=1.0, design_factor=1.25, fatigue_exponent=6.4, ref_cycles=2e7,
                                damage_limit=1.0))
    assert direct.mos == pytest.approx(res.mos, rel=1e-12) and direct.verdict == PB.UNCONFIRMED and not any(i["confirmed"] for i in direct.inputs if i.get("key"))
    # nothing at all: no allowable, no verdict
    none = PB.check_pcb(_spec())
    assert none.z_allow_mm is None and none.mos is None and none.verdict == PB.UNCONFIRMED
    unc = confirmed_space(LIM, confirm_it=False, units=UN)
    assert PB.check_pcb(_spec(), S4Limits.from_requirement_set(unc)).verdict == PB.UNCONFIRMED


def test_three_band_damage_by_hand_and_against_the_spectral_fatigue():
    b, n_ref, f_n, T = 6.4, 2.0e7, 100.0, 3600.0
    sigma = 0.05
    z_allow = 0.00022 * 5.0 / 0.062 * IN
    p = [math.erf(1 / math.sqrt(2)), math.erf(2 / math.sqrt(2)) - math.erf(1 / math.sqrt(2)), math.erf(3 / math.sqrt(2)) - math.erf(2 / math.sqrt(2))]
    assert p == pytest.approx(PB.band_fractions())
    by_hand = sum(pi * f_n * T / (n_ref * (z_allow / (i * sigma)) ** b) for i, pi in zip((1, 2, 3), p))
    got = PB.three_band_damage(sigma, 1.0, z_allow, b, n_ref, f_n, T)
    assert got["damage"] == pytest.approx(by_hand, rel=1e-12)
    assert [r["cycles"] for r in got["bands"]] == pytest.approx([pi * f_n * T for pi in p])
    nb = PB.narrow_band_damage(sigma, 1.0, z_allow, b, n_ref, f_n, T)
    # closed form of the Rayleigh damage with the same law: ν T / N_ref · (√2 σ / Z_allow)^b · Γ(1 + b/2)
    exact = f_n * T / n_ref * (math.sqrt(2) * sigma / z_allow) ** b * math.gamma(1 + b / 2)
    assert nb == pytest.approx(exact, rel=1e-9)
    assert 0.95 < got["damage"] / nb < 1.10                          # the discrete three-band rule against the continuous Rayleigh distribution
    # the double-amplitude convention doubles the displacement measure
    d2 = PB.three_band_damage(sigma, 2.0, z_allow, b, n_ref, f_n, T)["damage"]
    assert d2 == pytest.approx(2 ** b * got["damage"], rel=1e-12)


def test_relative_rms_matches_the_closed_form_of_a_one_dof_board():
    m, k, zeta = 1.0e-3, 4.0e3, 0.02
    wn = math.sqrt(k / m)
    fn = wn / (2 * math.pi)
    asd = [[2.0, 0.02], [20000.0, 0.02]]
    z = PB.relative_rms(np.array([fn]), zeta, np.array([math.sqrt(m)]), np.array([1.0 / math.sqrt(m)]), asd)
    w_mm = 0.02 * 9810.0 ** 2
    assert z == pytest.approx(math.sqrt(w_mm / (8 * zeta * wn ** 3)), rel=0.02)          # infinite flat band: z² = W / (8 ζ ω_n³)


def test_check_with_damage_fails_when_the_damage_limit_is_exceeded():
    sp = confirmed_space({**LIM, "pcb_damage_limit": 1e-9}, units={"pcb_steinberg_const": "in"})
    res = PB.check_pcb(_spec(), S4Limits.from_requirement_set(sp))
    assert res.mos > 0 and res.damage_ok is False and res.verdict == PB.FAIL
    assert res.to_dict()["damage_narrow_band"] == pytest.approx(res.damage_nb)

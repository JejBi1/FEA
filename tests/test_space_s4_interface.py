"""F14 S4: interface forces of the modal dynamics – mesh-free 2-DOF reference (exact modal decoupling), the quasi-static limit M_total·a_base, the residual of a
truncated basis, the random RMS against a direct integration, moments about a reference point."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.dynamics import interface as IF

M1, M2, K1, K2 = 2.0e-3, 1.0e-3, 4.0e6, 2.5e6          # t, t, N/mm, N/mm (base – m1 – m2 chain)
ZETA = 0.03
G = 9810.0


def two_dof(n_modes: int = 2, residual_basis: bool = True) -> tuple[IF.ModalInterface, np.ndarray, np.ndarray]:
    """Chain base –k1– m1 –k2– m2 along x; the interface reaction is the force of the base spring k1 (K·u at the held node)."""
    M = np.diag([M1, M2])
    K = np.array([[K1 + K2, -K2], [-K2, K2]])
    w2, V = np.linalg.eig(np.linalg.solve(M, K))
    order = np.argsort(w2.real)
    w2, V = w2.real[order], V.real[:, order]
    for j in range(2):
        V[:, j] /= np.sqrt(V[:, j] @ M @ V[:, j])
    gamma = np.array([V[:, j] @ M @ np.ones(2) for j in range(2)])
    freq = np.sqrt(w2) / (2 * np.pi)
    Fj = -K1 * V[0, :]                                            # K·φ at the held node (force of the support on the structure per unit q)
    k = n_modes
    g3 = np.zeros((k, 3))
    g3[:, 0] = gamma[:k]
    F = np.zeros((k, 3))
    F[:, 0] = Fj[:k]
    x_if = np.array([0.0, 0.0, 0.0])
    # the held node sits at x = 0, the structure's masses are at x = 10, 25 mm: a base acceleration in y gives a moment through the lever arm
    grp = IF.InterfaceGroup(IF.ALL_GROUP, F, np.zeros((k, 3)), np.diag([M1 + M2, 0.0, 0.0]) * 1.0, np.zeros((3, 3)), 1)
    grp.F_static = np.zeros((3, 3))
    grp.F_static[0, 0] = M1 + M2
    mi = IF.ModalInterface(freq[:k], g3, g3 ** 2, M1 + M2, x_if, {IF.ALL_GROUP: grp})
    return mi, M, K


def exact_force(M: np.ndarray, K: np.ndarray, f: float, zeta: float) -> complex:
    """Direct frequency-domain solution with modal damping (C = Σ 2ζω_j Mφφᵀ M), unit base acceleration; the force that the support exerts on the structure = Newton's
    law Σ m_i (ü_i + a_b) = Σ m_i (1 − ω²u_i): the elastic reaction of the base spring PLUS the damping force of the modal damper."""
    w2, V = np.linalg.eig(np.linalg.solve(M, K))
    o = np.argsort(w2.real)
    w2, V = w2.real[o], V.real[:, o]
    C = np.zeros((2, 2))
    for j in range(2):
        v = V[:, j] / np.sqrt(V[:, j] @ M @ V[:, j])
        C += 2 * zeta * np.sqrt(w2[j]) * np.outer(M @ v, M @ v)
    w = 2 * np.pi * f
    u = np.linalg.solve(K - w ** 2 * M + 1j * w * C, -M @ np.ones(2))
    return complex(np.diag(M) @ (1.0 - w ** 2 * u))


def test_complete_basis_matches_the_direct_frequency_response():
    mi, M, K = two_dof(2)
    f = np.array([20.0, 120.0, 300.0, 900.0, 1500.0])
    TF, _ = IF.interface_transfer(mi, f, ZETA, "x", residual=False)
    ref = np.array([exact_force(M, K, x, ZETA) for x in f])
    assert np.allclose(TF[:, 0], ref, rtol=1e-9, atol=1e-12)


def test_quasi_static_limit_is_total_mass_times_base_acceleration():
    for n in (2, 1):                                              # complete basis and a truncated one (needs the residual)
        mi, _, _ = two_dof(n)
        sp = IF.sine_interface(mi, "x", 1.0, ZETA, f=np.array([1e-3, 0.05, 1.0]))
        # f → 0: M_total · a_base (a_base = 1 g), well inside the 0.5 % of the gate
        assert abs(sp.F_peak[0] - (M1 + M2) * G) / ((M1 + M2) * G) < 1e-4


def test_truncated_basis_with_residual_is_closer_at_high_frequency_than_without():
    mi, M, K = two_dof(1)
    f = np.array([2000.0, 5000.0])
    with_res, _ = IF.interface_transfer(mi, f, ZETA, "x", residual=True)
    no_res, _ = IF.interface_transfer(mi, f, ZETA, "x", residual=False)
    ref = np.array([exact_force(M, K, x, ZETA) for x in f])
    assert np.all(np.abs(with_res[:, 0] - ref) < np.abs(no_res[:, 0] - ref))


def test_static_residual_formula():
    mi, _, _ = two_dof(1)
    F_res, _ = mi.residual()
    g = mi.group()
    expect = g.F_static[0] + g.F[0] * mi.gamma[0, 0] / mi.omega[0] ** 2
    assert np.allclose(F_res[0], expect)


def test_random_rms_matches_direct_integration():
    mi, M, K = two_dof(2)
    asd = [[20.0, 0.01], [2000.0, 0.01]]                          # flat 0.01 g²/Hz
    r = IF.random_interface(mi, "x", asd, ZETA)
    # reference: dense direct frequency-domain integration with the exact solution
    from fea_core.dynamics import psd
    fine = psd.freq_grid(20.0, 2000.0, mi.freq, ZETA, per_peak=401, per_decade=2000)
    T = np.array([exact_force(M, K, x, ZETA) for x in fine])
    ref = float(np.sqrt(np.sum(np.abs(T) ** 2 * 0.01 * G * G * psd.trapz_weights(fine))))
    assert r.F_rms[0] == pytest.approx(ref, rel=2e-3)
    assert r.F_rms[1] == 0.0 and r.F_rms_total == pytest.approx(r.F_rms[0])


def test_moment_about_the_reference_point_and_groups():
    mi, _, _ = two_dof(2)
    g = mi.group()
    g.M[:, 2] = 3.0 * g.F[:, 0]                                   # an interface force at a lever arm of 3 mm
    TF, TM = IF.interface_transfer(mi, np.array([100.0, 400.0]), ZETA, "x", residual=False)
    assert np.allclose(TM[:, 2], 3.0 * TF[:, 0])
    with pytest.raises(IF.InterfaceError):
        mi.group("nope")


def test_direction_parsing():
    assert np.allclose(IF.unit_direction("y"), [0, 1, 0])
    assert np.allclose(IF.unit_direction(3), [0, 0, 1])
    assert np.allclose(IF.unit_direction([1, 1, 0]), np.array([1, 1, 0]) / np.sqrt(2))
    with pytest.raises(IF.InterfaceError):
        IF.unit_direction("w")
    with pytest.raises(IF.InterfaceError):
        IF.unit_direction([0, 0, 0])


# ------------------------------------------------------------------ review finding I8: the damping force of the support
@pytest.mark.parametrize("zeta", [0.02, 0.05, 0.1, 0.3])
def test_sine_force_is_newtons_transmitted_force_with_the_damping_term(zeta):
    """SDOF base excitation: the support force is m·(absolute acceleration) = m·a·√(1 + (2ζr)²)/√((1−r²)² + (2ζr)²), at and far above the resonance too."""
    from tests.s4_helpers import M_SDOF, sdof_interface
    mi = sdof_interface()
    fn = float(mi.freq[0])
    for r in (0.3, 0.8, 1.0, 1.5, 3.0, 10.0):
        sp = IF.sine_interface(mi, "x", 1.0, zeta, f=np.array([r * fn]))
        true = M_SDOF * G * np.sqrt(1 + (2 * zeta * r) ** 2) / np.sqrt((1 - r * r) ** 2 + (2 * zeta * r) ** 2)
        assert sp.F_peak[0] == pytest.approx(true, rel=1e-9), (zeta, r)
    one = IF.sine_interface(mi, "x", 1.0, zeta, f=np.array([fn]))
    assert one.F_peak[0] == pytest.approx(M_SDOF * G * np.sqrt(1 + 4 * zeta ** 2) / (2 * zeta), rel=1e-9)       # the S4A closed form


@pytest.mark.parametrize("zeta", [0.02, 0.1, 0.3])
def test_random_force_is_miles_with_the_damping_term(zeta):
    """White-noise base: F_rms = m·G·√(W π f_n/(4ζ))·√(1 + 4ζ²) (Miles for the absolute acceleration, the elastic reaction alone misses the √(1 + 4ζ²))."""
    from tests.s4_helpers import M_SDOF, sdof_interface
    mi = sdof_interface()
    fn = float(mi.freq[0])
    W = 0.02
    r = IF.random_interface(mi, "x", [[1.0, W], [4.0e5, W]], zeta)
    miles = M_SDOF * G * np.sqrt(W * np.pi * fn / (4 * zeta)) * np.sqrt(1 + 4 * zeta ** 2)
    assert r.F_rms[0] == pytest.approx(miles, rel=5e-3)


def test_structural_damping_factor_is_one_plus_i_eta():
    from tests.s4_helpers import M_SDOF, sdof_interface
    mi = sdof_interface()
    fn = float(mi.freq[0])
    eta = 0.06
    TF, _ = IF.interface_transfer(mi, np.array([fn]), 0.0, "x", eta=eta, residual=False)
    # m·a·(1 + iη)/(1 + iη − r²) at r = 1: |T| = m·a·√(1 + η²)/η
    assert abs(TF[0, 0]) * G == pytest.approx(M_SDOF * G * np.sqrt(1 + eta ** 2) / eta, rel=1e-9)

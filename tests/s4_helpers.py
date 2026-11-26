"""Shared helpers of the F14 S4 tests: a one-mode (SDOF) modal interface with a closed form, and a confirmed requirement set with S4 limits.
All numbers are ARBITRARY TEST VALUES (not values of any standard)."""
from __future__ import annotations

import numpy as np

import verification.cases  # noqa: F401  (the cases first: space_cases imports from it)
from fea_core.dynamics import interface as IF
from fea_core.space import requirements as RQ
from fea_core.space.model import LimitEntry, SpaceSettings
from verification import space_cases as SC

G = 9810.0
M_SDOF, K_SDOF, Z_SDOF = 1.0e-3, 4.0e3, 0.02                 # t (1 kg), N/mm, damping ratio  ->  f_n = 318.31 Hz


def sdof_interface(m: float = M_SDOF, k: float = K_SDOF) -> IF.ModalInterface:
    """One mode, complete basis: φ = 1/√m, Γ = √m, reaction K·φ = −k/√m, static reaction m (per unit base acceleration, along x)."""
    phi = 1.0 / np.sqrt(m)
    freq = np.array([np.sqrt(k / m) / (2 * np.pi)])
    gam = np.array([[np.sqrt(m), 0.0, 0.0]])
    F = np.array([[-k * phi, 0.0, 0.0]])
    Fs = np.zeros((3, 3))
    Fs[0, 0] = m
    grp = IF.InterfaceGroup(IF.ALL_GROUP, F, np.zeros((1, 3)), Fs, np.zeros((3, 3)), 1)
    return IF.ModalInterface(freq, gam, gam ** 2, m, np.zeros(3), {IF.ALL_GROUP: grp})


def many_mode_interface(K: int, seed: int = 1, f_lo: float = 30.0, f_hi: float = 2500.0, m: float = M_SDOF) -> IF.ModalInterface:
    """K random modes (frequencies in [f_lo, f_hi], participation vectors summing to ~m): a synthetic interface whose reactions follow Newton (F_j = -w_j^2 Gamma_j)."""
    rng = np.random.default_rng(seed)
    freq = np.sort(rng.uniform(f_lo, f_hi, K))
    gam = rng.normal(0.0, 1.0, (K, 3)) * np.sqrt(m / K)
    F = -((2 * np.pi * freq) ** 2)[:, None] * gam
    grp = IF.InterfaceGroup(IF.ALL_GROUP, F, np.zeros((K, 3)), np.eye(3) * m, np.zeros((3, 3)), 10)
    return IF.ModalInterface(freq, gam, gam ** 2, m, np.zeros(3), {IF.ALL_GROUP: grp})


def confirmed_space(limits: dict[str, float], confirm_it: bool = True, units: dict[str, str] | None = None) -> SpaceSettings:
    """A space setting whose requirement set carries the given limits (document / revision / clause filled) and is confirmed."""
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    for key, v in limits.items():
        rs.limits.append(LimitEntry(key=key, value=v, unit=(units or {}).get(key, ""), document="test document", revision="test revision", clause="test clause"))
    sp = SpaceSettings(requirement_set=rs)
    if confirm_it:
        RQ.confirm(rs, "test engineer", "2026-01-01", scope=sp)
    return sp

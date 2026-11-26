"""Stresses and nodal forces from the displacements as in ccx (fea_core.recovery) (without Gmsh and ccx)."""
import numpy as np
import pytest

from fea_core.study import Study, Material
from tests.test_thermal import bar_mesh


def _study(alpha=12e-6):
    st = Study("t", "x.step")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9, alpha=alpha), default=True)
    return st


def test_uniform_strain_exact():
    """A linear displacement field (constant ε) → exact stress at all nodes, the internal forces sum to zero."""
    from fea_core.recovery import recover
    m = bar_mesh()
    eps = np.array([[1e-3, 2e-4, -1e-4], [2e-4, -5e-4, 3e-4], [-1e-4, 3e-4, 2e-4]])
    U = m.nodes @ eps.T
    S, RF = recover(_study(), m, U)
    E, nu = 210000.0, 0.3
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    sig = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
    ref = np.array([sig[0, 0], sig[1, 1], sig[2, 2], sig[0, 1], sig[1, 2], sig[2, 0]])   # order of ccx
    assert np.abs(S - ref).max() < 1e-9 * np.abs(ref).max()
    assert np.abs(RF.sum(axis=0)).max() < 1e-9 * np.abs(RF).max()
    # force on the face x = 100 = σ·n·A
    end = np.unique(m.surfaces["x100"])
    assert RF[end].sum(axis=0) == pytest.approx(sig[:, 0] * 100.0, rel=1e-9)


def test_free_thermal_expansion_no_stress():
    from fea_core.recovery import recover
    m = bar_mesh()
    a, dT = 12e-6, 50.0
    U = a * dT * m.nodes
    S, RF = recover(_study(a), m, U, np.full(len(m.nodes), dT))
    assert np.abs(S).max() < 1e-9
    assert np.abs(RF).max() < 1e-9

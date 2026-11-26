"""Verification cases (Gmsh + ccx). Not in the default run (pyproject addopts): `pytest -m slow` = the cases of seconds to ~1.5 min,
`pytest -m very_slow` = the long ones (verification/tiers.py: KK4 ~55 min, K6, K2, KK2, Chrono), `pytest -m ""` = everything."""
import pytest

from fea_core import config
from fea_core.solve import run_study
from verification import cases as C
from verification import tiers


def _params():
    return [pytest.param(k, marks=getattr(pytest.mark, tiers.tier(k)), id=k) for k in C.CASES]


@pytest.mark.parametrize("key", _params())
def test_case(key, tmp_path):
    case = C.CASES[key]
    study, parts = case.build(tmp_path)
    if case.run is not None:                    # own calculation (mechanisms – run_motion), as run_all
        res = case.run(study, tmp_path / "run", parts)
    else:
        res = run_study(study, tmp_path / "run")
    if key == "V6":
        C._V1_REF.pop("tip", None)  # comparison with V1 only in run_all
    checks = case.evaluate(res, study)
    failed = [f"{c.name}: {c.value:.6g} vs {c.ref:.6g} ({100 * c.error:.3f} %)" for c in checks if not c.ok]
    assert not failed, "\n".join(failed)
    # Linear cases have equilibrium at the level of rounding; nonlinear ones (contact, friction, NLGEOM) only with the precision
    # of the convergence of ccx → the project limit config.EQUILIBRIUM_TOL (above it the calculation ends with an error anyway).
    nonlinear = getattr(study, "is_nonlinear", lambda: False)()
    tol = config.EQUILIBRIUM_TOL if nonlinear else 1e-4
    for st in getattr(res, "steps", None) or []:
        eq = getattr(st, "equilibrium", None) or {}
        if "imbalance" in eq and not eq.get("dynamic"):     # implicit dynamics: inertia forces, not checked (contacts/run)
            assert eq["imbalance"] < tol, f"{st.name}: nerovnováha {eq['imbalance']:.3g} (mez {tol:g})"

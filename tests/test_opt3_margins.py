"""Margins / hotspots speed-ups (release 1.5.1): the memoised classification, the walk over the nodes of one part and the scaled equivalent stress give the same rows as the
direct evaluation (the stress values to round-off, everything else exactly)."""
from __future__ import annotations

import numpy as np
import pytest

import verification.cases  # noqa: F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core import hotspots as HS
from fea_core.space import margins as MG
from fea_core.space.model import SpaceSettings
from verification import space_cases as SC

LOAD_STEP = "Load"


def _setup():
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("protoflight"))
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, SC.TEST_SIGMA)], buckling=True)
    return st, res


def _rows(summ):
    return [r.to_dict() for r in summ.rows], [s.to_dict() for s in summ.singularities], summ.verdict, summ.reasons


def _close(a, b, rtol=1e-13):
    """Equal in structure and in every non-float value, floats to `rtol`."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k], rtol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, rtol)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=rtol, abs=1e-300) or (np.isnan(a) and np.isnan(b))
    else:
        assert a == b


def test_margins_equal_the_direct_evaluation(monkeypatch):
    st, res = _setup()
    new = _rows(MG.compute_margins(st, res))
    monkeypatch.setattr(MG._Engine, "_equivalent", staticmethod(lambda S, tens, jm, measure, cache: MG.equivalent(S, measure)))
    ref = _rows(MG.compute_margins(st, res))
    _close(new, ref)


@pytest.mark.parametrize("measure", ["von_mises", "max_principal", "max_shear"])
@pytest.mark.parametrize("j", [1.0, 1.25, 0.8])
def test_scaled_equivalent_stress_is_the_direct_one_to_round_off(measure, j):
    S0 = np.random.default_rng(5).standard_normal((400, 6)) * 50.0
    cache: dict = {}
    got = MG._Engine._equivalent(j * S0, {"mechanical": S0}, j, measure, cache)
    ref = MG.equivalent(j * S0, measure)
    assert np.abs(got - ref).max() <= 1e-14 * max(np.abs(ref).max(), 1.0)
    if j == 1.0:
        assert np.array_equal(got, ref)
    again = MG._Engine._equivalent(j * S0, {"mechanical": S0}, j, measure, cache)         # the second call is served from the cache
    assert np.array_equal(again, got) and measure in cache


def test_scaled_equivalent_stress_is_not_used_with_a_thermal_part_or_a_non_positive_factor():
    S0 = np.random.default_rng(6).standard_normal((50, 6))
    for tens, j in (({"mechanical": S0, "thermal": S0}, 1.2), ({"mechanical": S0}, -1.0), ({"mechanical": S0}, 0.0), ({"thermal": S0}, 1.2)):
        cache: dict = {}
        S = j * S0
        got = MG._Engine._equivalent(S, tens, j, "von_mises", cache)
        assert np.array_equal(got, MG.equivalent(S, "von_mises")) and not cache


def _classifier(st, res, vals, memo=None):
    from fea_core.results import StepResult
    S = np.zeros((len(res.mesh.nodes), 6))
    S[:, 0] = vals
    return HS.Classifier(st, res, StepResult(name="x", U=np.zeros((0, 3)), S=S), vals, memo)


def test_memoised_classification_is_identical_and_independent_of_the_field():
    st, res = _setup()
    n = len(res.mesh.nodes)
    rng = np.random.default_rng(2)
    memo: dict = {}
    for _ in range(3):                                           # several fields on one memo (cases / rules of one evaluation)
        vals = rng.random(n) * 100.0
        shared = _classifier(st, res, vals, memo)
        fresh = _classifier(st, res, vals)
        for node in rng.integers(0, n, 25):
            a, b = shared.classify(int(node), 3, True), fresh.classify(int(node), 3, True)
            assert a.keys() == b.keys()
            for k in a:
                assert np.array_equal(a[k], b[k]), k
    assert len(memo) > 25


def test_walk_over_the_nodes_of_a_part_equals_the_masked_walk_including_ties():
    st, res = _setup()
    n = len(res.mesh.nodes)
    rng = np.random.default_rng(4)
    nodes = np.unique(rng.choice(n, n // 2, replace=False))
    mask = np.zeros(n, bool)
    mask[nodes] = True
    for kind in ("random", "ties"):
        vals = rng.random(n) * 10.0 if kind == "random" else np.round(rng.random(n) * 4.0)      # many equal maxima
        vals[nodes[3]] = np.nan                                                                  # a non-finite value never governs
        a = HS.scan_maxima(_classifier(st, res, vals), allowed=mask, max_places=6, tags=True)
        b = HS.scan_maxima(_classifier(st, res, vals), max_places=6, tags=True, nodes=nodes)
        assert [p["node"] for p in a[0]] == [p["node"] for p in b[0]]
        assert (a[1] or {}).get("node") == (b[1] or {}).get("node")
        for pa, pb in zip(a[0], b[0]):
            assert pa["classification"] == pb["classification"] and pa["reasons"] == pb["reasons"] and pa["tags"] == pb["tags"]


def test_principal_stresses_are_computed_once_per_case_for_max_principal_and_max_shear(monkeypatch):
    S0 = np.random.default_rng(6).standard_normal((300, 6)) * 40.0
    calls = []
    real = MG.principal
    monkeypatch.setattr(MG, "principal", lambda s: (calls.append(1), real(s))[1])
    cache: dict = {}
    got = {m: MG._Engine._equivalent(1.1 * S0, {"mechanical": S0}, 1.1, m, cache) for m in ("von_mises", "max_principal", "max_shear")}
    assert len(calls) == 1
    monkeypatch.undo()
    for m, v in got.items():
        assert v == pytest.approx(MG.equivalent(1.1 * S0, m), rel=1e-13, abs=1e-300)
    again = MG._Engine._equivalent(1.1 * S0, {"mechanical": S0}, 1.1, "max_shear", {})
    assert np.array_equal(again, got["max_shear"])                    # the shared eigenvalues give the very numbers of an own evaluation


def test_scan_maxima_refuses_a_mask_together_with_nodes_and_nodes_that_are_not_ascending_unique():
    st, res = _setup()
    n = len(res.mesh.nodes)
    vals = np.random.default_rng(8).random(n)
    mask = np.ones(n, bool)
    good = np.array([2, 5, 9])
    with pytest.raises(ValueError, match="not both"):
        HS.scan_maxima(_classifier(st, res, vals), allowed=mask, nodes=good)
    for bad in ([5, 5, 9], [9, 5, 2], [-1, 3], [0, n], [[1, 2], [3, 4]]):
        with pytest.raises(ValueError, match="ascending"):
            HS.scan_maxima(_classifier(st, res, vals), nodes=np.array(bad))
    assert HS.scan_maxima(_classifier(st, res, vals), nodes=np.array([], np.int64)) == ([], None)
    places, _gov = HS.scan_maxima(_classifier(st, res, vals), nodes=good, max_places=1)
    assert places[0]["node"] == int(good[np.argmax(vals[good])]) + 1          # the places carry 1-based node numbers

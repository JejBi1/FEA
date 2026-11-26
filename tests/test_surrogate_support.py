"""Where the surrogate may answer (F15-S4 review fixes C1, I1, I2): the box and the convex hull of the USABLE points, gaps, failed regions,
collinear designs, one-at-a-time designs, the stated band level with few hold-out points, the edge warning, the transform from the training
values only, the adaptively chosen points. The scenarios are those of the independent review (failed region, 30-point designs, collinear
and star designs); the model must refuse or flag where it has no analyses – never answer with a narrow band and `reliable: True`."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.surrogate import SurrogateError, fit_surrogate, lhs_design
from fea_core.surrogate import support as S
from fea_core.surrogate.fit import SURROGATE_KPIS, min_points
from fea_core.surrogate.model import backends_available

BACKENDS = ["rbf", pytest.param("gp", marks=pytest.mark.skipif(not backends_available()["gp"], reason="scikit-learn is not installed"))]


def stress(Z):
    """Plate with a hole: a = d / W (0.1 … 0.9), second parameter the width; the stress rises steeply for a thin ligament."""
    a = Z[:, 0]
    return 100.0 / (Z[:, 1] * (1 - a)) * (3.0 - 3.14 * a + 3.667 * a ** 2 - 1.527 * a ** 3) * 50


def pts(X, y, ok=None, kpi="max_stress"):
    keys = [f"p{i}" for i in range(X.shape[1])]
    return [{"params": dict(zip(keys, map(float, x))), kpi: float(v), "converged": bool(True if ok is None else ok[i])} for i, (x, v) in enumerate(zip(X, y))]


# ------------------------------------------------------------------ C1: the failed region
@pytest.mark.parametrize("backend", BACKENDS)
def test_failed_region_is_refused_not_answered_with_a_narrow_band(backend):
    """LHS in [0.1, 0.9] × [40, 100]; the points with a > 0.65 fail (a thin ligament does not mesh). The old model answered at a = 0.85 with a
    narrow band and `reliable: True`; now the box is that of the converged points and the failed region is refused."""
    for seed in range(3):
        X = lhs_design([(0.1, 0.9), (40, 100)], 60, seed=seed)
        ok = X[:, 0] <= 0.65
        sm = fit_surrogate(pts(X, stress(X), ok), backend=backend, seed=seed, bounds={"p0": (0.1, 0.9), "p1": (40, 100)})
        assert sm.continuous[0].hi == pytest.approx(X[ok, 0].max()) and sm.continuous[0].hi < 0.66      # the box = the usable points
        assert any("failed" in w and "p0 in [0.6" in w for w in sm.warnings), sm.warnings            # which region failed
        assert any("analysed points cover p0" in w and "[0.1; 0.9]" in w for w in sm.warnings), sm.warnings
        with pytest.raises(SurrogateError, match="Extrapolation refused"):
            sm.predict({"p0": 0.85, "p1": 70.0})
        with pytest.raises(SurrogateError, match="outside the range of the analysed points"):
            sm.predict({"p0": 0.7, "p1": 70.0})
        r = sm.predict({"p0": 0.85, "p1": 70.0}, extrapolate=True)                  # asked for explicitly: flagged everywhere
        assert r["extrapolated"] and not r["inside_support"] and not any(k["reliable"] for k in r["kpis"].values())
        assert any("EXTRAPOLATION" in w for w in r["warnings"])
        inside = sm.predict({"p0": 0.3, "p1": 70.0})
        assert inside["inside_support"] and inside["kpis"]["max_stress"]["lower"] < stress(np.array([[0.3, 70.0]]))[0] < inside["kpis"]["max_stress"]["upper"]


def test_failed_region_in_one_parameter():
    X = lhs_design([(0.1, 0.9)], 40, seed=1)
    ok = X[:, 0] <= 0.65
    sm = fit_surrogate(pts(X, stress(np.c_[X[:, 0], np.full(len(X), 70.0)]), ok), backend="rbf", bounds={"p0": (0.1, 0.9)})
    with pytest.raises(SurrogateError, match="outside the range"):
        sm.predict({"p0": 0.85})
    assert sm.predict({"p0": 0.5})["inside_support"]
    assert sm.predict({"p0": 0.5})["inside_hull"] is None                       # one parameter: the range decides, no hull


def test_failed_points_inside_the_design_make_a_gap_that_is_refused():
    """A failure region in the MIDDLE of the box: inside the convex hull, so only the gap tests (and the failed points) can see it."""
    X = lhs_design([(0.0, 1.0), (0.0, 1.0)], 120, seed=3)
    hole = np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.5) < 0.3
    sm = fit_surrogate(pts(X, 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2, ~hole), backend="rbf")
    assert any("failed" in w for w in sm.warnings)
    with pytest.raises(SurrogateError, match="failed analysis|gap in the design"):
        sm.predict({"p0": 0.5, "p1": 0.5})
    assert sm.predict({"p0": 0.15, "p1": 0.5})["inside_support"] and sm.predict({"p0": 0.85, "p1": 0.15})["inside_support"]
    # the same hole without any failed point (a design with a gap): the distance test alone refuses the centre
    keep = ~hole
    sm2 = fit_surrogate(pts(X[keep], (100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2)[keep]), backend="rbf")
    with pytest.raises(SurrogateError, match="gap in the design"):
        sm2.predict({"p0": 0.5, "p1": 0.5})


# ------------------------------------------------------------------ C1: degenerate designs
@pytest.mark.parametrize("kpis", [None, ["max_stress"]])
def test_collinear_design_is_refused_at_the_fit(kpis):
    """mode "list" with two numeric parameters = points on a line of the parameter space: no surrogate of both parameters exists."""
    t = np.sort(np.random.default_rng(0).random(30))
    with pytest.raises(SurrogateError, match="span only 1 of the 2"):
        fit_surrogate(pts(np.c_[t, t], 100 + 60 * t), kpis=kpis, backend="rbf")


def test_one_at_a_time_design_refuses_the_interaction_corner():
    """Points only on the two axes: the response has an interaction that no point can see. Anywhere off the axes there is no analysis."""
    t = np.linspace(0.0, 1.0, 20)
    Xs = np.vstack([np.c_[t, np.zeros(20)], np.c_[np.zeros(19), t[1:]]])                 # the usual star: both axes in equal steps
    f = lambda Z: 100 + 60 * Z[:, 0] + 40 * Z[:, 1] + 150 * Z[:, 0] * Z[:, 1]      # noqa: E731
    sm = fit_surrogate(pts(Xs, f(Xs)), backend="rbf", seed=1)
    for corner in ({"p0": 0.9, "p1": 0.9}, {"p0": 0.5, "p1": 0.5}):
        with pytest.raises(SurrogateError, match="Extrapolation refused"):
            sm.predict(corner)
    assert sm.predict({"p0": 0.5, "p1": 0.0})["inside_support"]                  # on the axis


def test_one_parameter_with_the_data_in_a_narrow_range_and_wide_bounds():
    x = 0.4 + 0.2 * np.sort(np.random.default_rng(0).random(30))
    sm = fit_surrogate(pts(x[:, None], 100 * np.exp(3 * x)), backend="rbf", bounds={"p0": (0.0, 1.0)})
    with pytest.raises(SurrogateError, match="outside the range"):
        sm.predict({"p0": 0.95})
    assert any("analysed points cover p0" in w for w in sm.warnings)


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_support_does_not_refuse_ordinary_latin_hypercube_queries(d):
    """SUPPORT_K: inside the hull of a Latin hypercube the nearest analysed point is never farther than SUPPORT_K median spacings."""
    for seed in range(4):
        X = lhs_design([(0.0, 1.0)] * d, min_points(d), seed=seed)
        U = X.copy()
        sup = S.Support(U, X)
        Q = np.random.default_rng(seed).random((1500, d))
        inside = sup.in_hull(Q)
        assert inside.sum() > 20
        assert np.all(sup.distance_ratio(Q[inside]) <= S.SUPPORT_K), sup.distance_ratio(Q[inside]).max()


@pytest.mark.parametrize("d", [2, 3])
def test_hull_by_linear_programming_agrees_with_qhull(d, monkeypatch):
    """Above HULL_MAX_POINTS (or when Qhull fails) every query is one linear program; both ways must give the same answer."""
    X = lhs_design([(0.0, 1.0)] * d, 40, seed=2)
    Q = np.random.default_rng(1).random((300, d)) * 1.2 - 0.1
    a = S.Support(X, X).in_hull(Q)
    monkeypatch.setattr(S, "HULL_MAX_POINTS", 5)
    b = S.Support(X, X).in_hull(Q)
    assert np.array_equal(a, b) and 0 < a.sum() < len(Q)


def test_qhull_failure_falls_back_to_linear_programs(monkeypatch):
    import scipy.spatial
    X = lhs_design([(0.0, 1.0)] * 2, 30, seed=0)

    def boom(*a, **k):
        raise RuntimeError("Qhull failed")
    monkeypatch.setattr(scipy.spatial, "ConvexHull", boom)
    sup = S.Support(X, X)
    assert sup.in_hull(np.array([[0.5, 0.5], [3.0, 3.0]])).tolist() == [True, False]          # decided, not silently answered


def test_rank_check():
    t = np.linspace(0, 1, 10)
    assert S.spanned_rank(np.c_[t, t]) == 1 and S.spanned_rank(np.c_[t, t ** 2]) == 2 and S.spanned_rank(np.c_[t, np.zeros(10)]) == 1
    assert S.spanned_rank(np.zeros((1, 2))) == 0


# ------------------------------------------------------------------ I2: coverage near the edges
def test_edge_of_the_analysed_range_carries_a_warning_and_the_limitation_is_stated():
    from fea_core.surrogate import LIMITATIONS, LIMITATIONS_SHORT
    X = lhs_design([(0.0, 1.0), (0.0, 1.0)], 40, seed=1)
    sm = fit_surrogate(pts(X, 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2), backend="rbf")
    lo, hi = sm.continuous[0].lo, sm.continuous[0].hi
    edge = sm.predict({"p0": lo + 0.01 * (hi - lo), "p1": 0.5})
    assert edge["near_edge"] and any("edge of the analysed range" in w for w in edge["warnings"]) and edge["kpis"]["max_stress"]["value"] > 0
    mid = sm.predict({"p0": 0.5, "p1": 0.5})
    assert not mid["near_edge"] and not any("edge" in w for w in mid["warnings"])
    for text in (LIMITATIONS, LIMITATIONS_SHORT):
        assert "average" in text.lower() and "edge" in text.lower()
    assert sm.describe()["limitations"] == LIMITATIONS


@pytest.mark.parametrize("backend", BACKENDS)
def test_coverage_inside_the_analysed_region_on_the_reviewer_functions(backend):
    """The reviewer measured 0.83-0.95 over 30-60 point designs; where the model answers (the region of the converged points) the band must
    hold on average. Seeds are few (the full study is in the review scripts), the mean is checked, never a single seed."""
    cases = [(lambda X: 100.0 / (X[:, 1] * (1 - X[:, 0])) * (3.0 - 3.14 * X[:, 0] + 3.667 * X[:, 0] ** 2 - 1.527 * X[:, 0] ** 3), [(0.1, 0.5), (40, 100)], 45),
             (lambda X: np.maximum(50 + 120 * X[:, 0] ** 2, 140 - 90 * X[:, 0] + 60 * X[:, 1]), [(0, 1), (0, 1)], 45),
             (lambda X: 80 + 60 * np.sin(4 * X[:, 0]) ** 2 + 30 * X[:, 0], [(0, 1)], 30)]
    for fn, bounds, n in cases:
        cov = []
        for seed in range(4 if backend == "rbf" else 2):
            X = lhs_design(bounds, n, seed=seed)
            sm = fit_surrogate(pts(X, fn(X)), backend=backend, seed=seed, bounds={f"p{i}": b for i, b in enumerate(bounds)})
            lo, hi = np.array(bounds).T
            Q = lo + np.random.default_rng(100 + seed).random((1500, len(bounds))) * (hi - lo)
            Q = Q[sm.supported("max_stress", (), Q)]
            assert len(Q) > 200
            v, a, b = sm.predict_arrays("max_stress", (), Q)
            cov.append(float(np.mean((a <= fn(Q)) & (fn(Q) <= b))))
        assert np.mean(cov) >= 0.85, (n, cov)


# ------------------------------------------------------------------ I1: the stated level
def test_the_default_holdout_makes_the_band_exact_and_the_default_minimum_is_nine():
    from fea_core.surrogate.fit import holdout_minimum
    assert holdout_minimum() == 9 and holdout_minimum(0.95) == 19 and holdout_minimum(0.9, 5) == 5
    for n in (min_points(2), 25, 40):
        X = lhs_design([(0.1, 0.5), (40, 100)], n, seed=3)
        sm = fit_surrogate(pts(X, stress(X)), backend="rbf", seed=3)
        c = sm.members[()]["max_stress"].cal
        assert c.n_cal >= 9 and c.exact and c.guaranteed == 0.9
        p = sm.predict({"p0": 0.3, "p1": 70.0})
        assert p["level"] == 0.9 and p["kpis"]["max_stress"]["level"] == 0.9 and sm.describe()["guaranteed_level"] == 0.9


@pytest.mark.parametrize("n_hold", [5, 6, 8])
def test_fewer_than_nine_holdout_points_state_the_guaranteed_level_everywhere(n_hold):
    """The reviewer's case: 6-8 hold-out points were reported as a 90 % band and `reliable`. Now every report says n / (n + 1)."""
    X = lhs_design([(0.1, 0.5), (40, 100)], 30, seed=3)
    sm = fit_surrogate(pts(X, stress(X)), backend="rbf", seed=3, validation_fraction=0.05, min_holdout=n_hold)
    m = sm.members[()]["max_stress"]
    assert m.cal.n_cal == n_hold and not m.cal.exact and m.cal.guaranteed == pytest.approx(n_hold / (n_hold + 1)) and not m.cal.reliable
    p = sm.predict({"p0": 0.3, "p1": 70.0})
    g = n_hold / (n_hold + 1)
    assert p["level"] == pytest.approx(g) and p["requested_level"] == 0.9 and p["kpis"]["max_stress"]["level"] == pytest.approx(g)
    assert not p["kpis"]["max_stress"]["reliable"] and any(f"{g:.0%}" in w for w in p["warnings"])
    d = sm.describe()
    assert d["guaranteed_level"] == pytest.approx(g) and not sm.reliable and d["members"][0]["guaranteed"] == pytest.approx(g)
    from fea_core.surrogate.fit import describe_text
    assert f"band {g:.0%}" in describe_text(sm)[0]


# ------------------------------------------------------------------ transform from the training values only; R² of a constant result
def test_a_holdout_value_never_changes_the_fit():
    """The log transform is decided by the training values: a hold-out value of any size or sign leaves the fitted mean untouched."""
    X = lhs_design([(0, 1), (0, 1)], 45, seed=2)
    y = 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2
    sm = fit_surrogate(pts(X, y), backend="rbf", seed=4)
    mem = sm.members[()]["max_stress"]
    Q = np.random.default_rng(1).random((200, 2))
    v1 = sm.predict_arrays("max_stress", (), Q)[0]
    y2 = y.copy()
    y2[mem.cal_ids] *= np.random.default_rng(0).uniform(0.5, 2.0, len(mem.cal_ids))
    y3 = y.copy()
    y3[mem.cal_ids[0]] = -5.0
    for yy in (y2, y3):
        sm2 = fit_surrogate(pts(X, yy), backend="rbf", seed=4)
        m2 = sm2.members[()]["max_stress"]
        assert m2.cal_ids == mem.cal_ids and m2.transform == mem.transform == "log"
        assert np.array_equal(sm2.predict_arrays("max_stress", (), Q)[0], v1)
    bad = fit_surrogate(pts(X, y3), backend="rbf", seed=4).members[()]["max_stress"]
    assert not bad.cal.reliable and any("cannot be scored" in w for w in bad.cal.reasons)          # a value the log cannot take: the band says so


def test_a_nearly_constant_result_is_not_flipped_to_unreliable_by_r2():
    X = lhs_design([(0, 1), (0, 1)], 40, seed=1)
    rng = np.random.default_rng(0)
    sm = fit_surrogate(pts(X, 7.5 * (1 + 1e-7 * rng.standard_normal(40))), backend="rbf")
    c = sm.members[()]["max_stress"].cal
    assert not c.r2_tested and c.reliable, c.reasons
    assert any("constant" in w for w in c.reasons)
    # a result that varies a lot but is badly predicted is still unreliable
    y = 10.0 + 40.0 * (X[:, 0] > 0.5) + 5.0 * rng.standard_normal(40)
    sm2 = fit_surrogate(pts(X, y), backend="rbf")
    assert sm2.members[()]["max_stress"].cal.r2_tested and not sm2.reliable


# ------------------------------------------------------------------ refit_with
def _row(params, **kpi):
    return {"params": params, "kpi": {k: kpi.get(k) for k in SURROGATE_KPIS}}


def test_refit_with_keeps_the_calibration_and_refuses_a_point_beyond_the_data():
    X = lhs_design([(0, 1), (0, 1)], 40, seed=0)
    y = 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2
    sm = fit_surrogate(pts(X, y), backend="rbf", seed=2)
    old = sm.members[()]["max_stress"]
    new = _row({"p0": 0.5, "p1": 0.5}, max_stress=137.5)
    sm2 = sm.refit_with([new])
    m2 = sm2.members[()]["max_stress"]
    assert m2.cal_ids == old.cal_ids and len(sm.rows) in m2.train_ids and len(sm2.rows) == 41 and sm2.rows[-1]["adaptive"]
    with pytest.raises(SurrogateError, match="outside the analysed region"):               # a verified point must not widen the box
        sm.refit_with([_row({"p0": 5.0, "p1": 0.5}, max_stress=900.0)])
    with pytest.raises(SurrogateError, match="outside the analysed region"):
        sm.refit_with([_row({"p0": 0.99, "p1": 0.99}, max_stress=900.0)])                  # inside the box, outside the hull
    with pytest.raises(SurrogateError, match="lacks a parameter"):
        sm.refit_with([_row({"p0": 0.5}, max_stress=1.0)])


def test_adaptively_chosen_points_never_calibrate_the_model():
    X = lhs_design([(0, 1), (0, 1)], 50, seed=5)
    y = 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2
    points = pts(X, y)
    for i in range(0, 50, 5):
        points[i]["adaptive"] = True                                                      # DesignPoint.adaptive of the points "verified" with FEM
    for seed in range(6):
        sm = fit_surrogate(points, backend="rbf", seed=seed)
        cal = set(sm.members[()]["max_stress"].cal_ids)
        assert cal and not cal & {i for i in range(0, 50, 5)}
        assert all(i in sm.members[()]["max_stress"].train_ids for i in range(0, 50, 5))
    assert sm.members[()]["max_stress"].cal.n_cal >= 9


def test_suggested_points_lie_inside_the_analysed_region():
    X = lhs_design([(0.1, 0.9), (40, 100)], 60, seed=0)
    ok = X[:, 0] <= 0.65
    sm = fit_surrogate(pts(X, stress(X), ok), backend="rbf")
    nxt = sm.suggest(3, seed=1)
    assert len(nxt) == 3
    for p in nxt:
        assert p["p0"] <= sm.continuous[0].hi + 1e-9 and sm.predict(p)["inside_support"]

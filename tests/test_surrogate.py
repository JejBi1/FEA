"""Surrogate model with conformal error bars (F15-S4): sampling, calibration, coverage on analytic functions, refusals, determinism,
save / load without pickle. Both backends: scipy RBF (always) and the Gaussian process (only where scikit-learn is installed)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_core.surrogate import (LIMITATIONS, SurrogateError, conformal_quantile, fit_surrogate, lhs_design, load_surrogate, split_indices)
from fea_core.surrogate.calibrate import Calibration, judge, required_calibration
from fea_core.surrogate.design import assign_categories, validation_size
from fea_core.surrogate.fit import holdout_minimum, min_points, min_train
from fea_core.surrogate.model import backends_available, resolve_backend

BACKENDS = ["rbf", pytest.param("gp", marks=pytest.mark.skipif(not backends_available()["gp"], reason="scikit-learn is not installed"))]


def stress(X):
    """A smooth positive 'KPI' of two parameters: a Peterson-like decay times a ripple."""
    return 80.0 * np.exp(-X[:, 0] / 60.0) * (1.0 + 0.25 * np.sin(X[:, 1] / 12.0)) + 10.0


def beam(X):
    """Cantilever deflection of 4 parameters: F L^3 / (3 E I), I = b h^3 / 12 (the KPI spans orders of magnitude)."""
    L, b, h, E = X.T
    return 100.0 * L ** 3 / (3.0 * E * b * h ** 3 / 12.0)


BOUNDS2 = [(10.0, 60.0), (0.0, 40.0)]
BOUNDS4 = [(100.0, 200.0), (10.0, 20.0), (5.0, 15.0), (60000.0, 210000.0)]


def points(fn, bounds, n, seed, noise=0.0, key="max_stress"):
    X = lhs_design(bounds, n, seed=seed)
    y = fn(X)
    if noise:
        y = y * (1.0 + noise * np.random.default_rng(seed + 1000).standard_normal(n))
    keys = [f"p{i}" for i in range(len(bounds))]
    return [{"params": dict(zip(keys, map(float, X[i]))), "kpi": {key: float(y[i])}} for i in range(n)], keys


# ------------------------------------------------------------------ design and split
def test_lhs_design_deterministic_in_bounds_and_stratified():
    a = lhs_design(BOUNDS2, 20, seed=3)
    assert np.array_equal(a, lhs_design(BOUNDS2, 20, seed=3))
    assert not np.array_equal(a, lhs_design(BOUNDS2, 20, seed=4))
    for j, (lo, hi) in enumerate(BOUNDS2):
        assert lo <= a[:, j].min() and a[:, j].max() <= hi
        strata = np.floor((a[:, j] - lo) / (hi - lo) * 20).astype(int)
        assert sorted(strata) == list(range(20)), "Latin hypercube: exactly one point per stratum"
    s = lhs_design(BOUNDS2, 16, seed=1, method="sobol")
    assert s.shape == (16, 2)


def test_lhs_design_refuses_bad_input():
    with pytest.raises(ValueError):
        lhs_design([], 5)
    with pytest.raises(ValueError):
        lhs_design([(0, 1)] * 6, 5)                       # more than 5 continuous parameters
    with pytest.raises(ValueError):
        lhs_design([(1.0, 1.0)], 5)
    with pytest.raises(ValueError):
        lhs_design([(0, 1)], 0)


def test_split_is_seeded_disjoint_and_never_below_five():
    tr, va = split_indices(30, seed=2)
    assert len(va) == validation_size(30) == 6 and len(tr) == 24
    assert not set(tr) & set(va) and sorted(set(tr) | set(va)) == list(range(30))
    assert (tr == split_indices(30, seed=2)[0]).all()
    assert len(split_indices(12, seed=0)[1]) == 5          # 20 % would be 3 – the minimum is 5


def test_assign_categories_is_balanced():
    c = assign_categories(["A", "B", "C"], 10, seed=1)
    assert sorted(c) == sorted(["A"] * 4 + ["B"] * 3 + ["C"] * 3) or sorted(c)[:3] == ["A", "A", "A"]
    assert {x: c.count(x) for x in "ABC"}.keys() == {"A", "B", "C"} and max(map(c.count, "ABC")) - min(map(c.count, "ABC")) <= 1


# ------------------------------------------------------------------ conformal quantile
def test_conformal_quantile_finite_sample_rule():
    s = np.arange(1, 20) / 19.0                           # 19 scores
    q, exact = conformal_quantile(s, 0.9)
    assert exact and q == pytest.approx(s[math.ceil(20 * 0.9) - 1])          # the 18th smallest
    q, exact = conformal_quantile(s[:8], 0.9)
    assert not exact and q == s[7]                                           # too few: the largest, flagged
    q, exact = conformal_quantile(s[:9], 0.9)
    assert exact and q == s[8]                                               # 9 points are the smallest exact set at 90 %
    with pytest.raises(ValueError):
        conformal_quantile([], 0.9)


def test_judge_rules():
    ok = judge(Calibration(n_train=40, n_cal=20, exact=True, coverage_cv=0.9, r2=0.99))
    assert ok.reliable and ok.guaranteed == 0.9
    assert not judge(Calibration(n_train=40, n_cal=20, exact=True, coverage_cv=0.9, r2=0.8)).reliable          # R² < 0.9
    assert not judge(Calibration(n_train=40, n_cal=20, exact=True, coverage_cv=0.4, r2=0.99)).reliable         # real under-coverage
    five = judge(Calibration(n_train=40, n_cal=5, exact=False, coverage_cv=0.9, r2=0.99))
    assert five.guaranteed == pytest.approx(5 / 6) and not five.reliable
    for n_cal in (6, 7, 8):                                   # I1: fewer than 9 hold-out points never make a "reliable" 90 % band
        c = judge(Calibration(n_train=40, n_cal=n_cal, exact=False, coverage_cv=0.9, r2=0.99))
        assert not c.reliable and c.guaranteed == pytest.approx(n_cal / (n_cal + 1)) and any("guaranteed" in w for w in c.reasons)
    nine = judge(Calibration(n_train=40, n_cal=9, exact=True, coverage_cv=0.9, r2=0.99))
    assert nine.reliable and nine.guaranteed == 0.9
    assert (required_calibration(0.9), required_calibration(0.95), required_calibration(0.8), required_calibration(0.5)) == (9, 19, 4, 1)
    # a practically constant result: R² only measures noise, it must not flip the model
    flat = judge(Calibration(n_train=40, n_cal=20, exact=True, coverage_cv=0.9, r2=-0.2, spread=1e-7, half_band=2e-7, scale=7.5))
    assert flat.reliable and not flat.r2_tested and any("constant" in w for w in flat.reasons)
    wide = judge(Calibration(n_train=40, n_cal=20, exact=True, coverage_cv=0.9, r2=-0.2, spread=10.0, half_band=20.0, scale=7.5))
    assert not wide.reliable and wide.r2_tested                 # a band as wide as the spread of a result that is NOT constant: still unreliable


# ------------------------------------------------------------------ coverage over seeds
def _coverage(backend, fn, bounds, n, seeds, noise=0.0, n_test=400):
    """Empirical coverage of the 90 % band on fresh uniform points of the region where the model answers (the claim is: inside the
    analysed region, on average), per seed. The share of the box that is inside that region is checked as well (the support must not
    refuse ordinary queries of an ordinary design)."""
    out = []
    for seed in seeds:
        pts, keys = points(fn, bounds, n, seed, noise)
        sm = fit_surrogate(pts, backend=backend, seed=seed)
        rng = np.random.default_rng(10_000 + seed)
        lo, hi = np.array(bounds).T
        T = lo + rng.random((n_test, len(bounds))) * (hi - lo)
        ok = sm.supported("max_stress", (), T)
        # the convex hull of a Latin hypercube covers most of a 2D box but only about half of a 4D box (the corners are far from every point)
        assert ok.mean() >= (0.80 if len(bounds) <= 2 else 0.30), "the support refuses too much of an ordinary Latin-hypercube design"
        T = T[ok]
        truth = fn(T)
        if noise:
            truth = truth * (1.0 + noise * rng.standard_normal(len(T)))
        v, a, b = sm.predict_arrays("max_stress", (), T)
        out.append(float(np.mean((truth >= a) & (truth <= b))))
    return np.array(out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_coverage_of_the_band_over_seeds_smooth_function(backend):
    cov = _coverage(backend, stress, BOUNDS2, 60, range(8 if backend == "rbf" else 4))
    assert cov.mean() >= 0.85, cov                    # marginal guarantee ≥ 90 % (n_cal = 12 → 12/13 on average); a noise-free smooth function is covered
    #                                                   conservatively (held-out points of a space-filling design lie in holes) – the upper side is checked with noise
    assert cov.min() >= 0.6, cov                      # the conformal coverage of one calibration set is Beta-distributed


@pytest.mark.parametrize("backend", BACKENDS)
def test_coverage_with_mesh_noise(backend):
    cov = _coverage(backend, stress, BOUNDS2, 70, range(8 if backend == "rbf" else 4), noise=0.01)
    assert 0.85 <= cov.mean() <= 0.99, cov


def test_coverage_four_parameters_cantilever_rbf():
    cov = _coverage("rbf", beam, BOUNDS4, 110, range(5), n_test=300)
    assert cov.mean() >= 0.85, cov


def test_conformal_scaling_is_model_agnostic_and_widens_a_bad_model():
    """The same data with a deliberately wrong sigma: q compensates (a tiny sigma gives a big q, a huge one a small q)."""
    pts, keys = points(stress, BOUNDS2, 50, 0, noise=0.02)
    sm = fit_surrogate(pts, backend="rbf", seed=0)
    q0 = next(iter(next(iter(sm.members.values())).values())).cal.q
    m = next(iter(next(iter(sm.members.values())).values()))
    be = m.backend
    base_rmse = be.rmse
    for factor in (0.1, 10.0):
        be.rmse = base_rmse * factor
        from fea_core.surrogate.calibrate import scores_of
        Xc, yc = sm._rows_xy(m.cal_ids, "max_stress")
        mu, sd = be.predict(sm._unit(Xc))
        q, _ = conformal_quantile(scores_of(m.to_std(yc), mu, sd), 0.9)
        assert q * factor == pytest.approx(q0, rel=1e-6)       # the band μ ± qσ does not depend on the scale of σ
    be.rmse = base_rmse


# ------------------------------------------------------------------ refusals and warnings
def test_refuses_too_small_training_set():
    pts, _ = points(stress, BOUNDS2, 10, 0)
    with pytest.raises(SurrogateError, match="too small"):
        fit_surrogate(pts, backend="rbf")
    assert min_points(2) > 10 and min_train(2) == 10 and min_train(1) == 6
    pts, _ = points(stress, BOUNDS2, min_points(2), 0)
    fit_surrogate(pts, backend="rbf")                                 # exactly the minimum works


def test_warns_with_few_points_per_dimension():
    pts, _ = points(stress, BOUNDS2, 22, 0)
    sm = fit_surrogate(pts, backend="rbf")
    assert any("training points" in w for w in sm.warnings)


def test_refuses_more_than_five_continuous_and_constant_only_inputs():
    X = lhs_design([(0, 1)] * 5, 80, seed=0)
    pts = [{"params": {f"p{j}": float(X[i, j]) for j in range(5)} | {"extra": float(i)}, "kpi": {"mass": 1.0 + X[i, 0]}} for i in range(80)]
    with pytest.raises(SurrogateError, match="At most 5"):
        fit_surrogate(pts, backend="rbf")
    const = [{"params": {"a": 1.0}, "kpi": {"mass": 1.0 + i}} for i in range(40)]
    with pytest.raises(SurrogateError, match="continuous"):
        fit_surrogate(const, backend="rbf")


@pytest.mark.parametrize("backend", BACKENDS)
def test_extrapolation_is_refused_unless_requested(backend):
    pts, keys = points(stress, BOUNDS2, 40, 1)
    sm = fit_surrogate(pts, backend=backend)
    inside = {keys[0]: 35.0, keys[1]: 20.0}
    r = sm.predict(inside)
    assert not r["extrapolated"] and r["kpis"]["max_stress"]["lower"] < r["kpis"]["max_stress"]["value"] < r["kpis"]["max_stress"]["upper"]
    with pytest.raises(SurrogateError, match="Extrapolation refused"):
        sm.predict({keys[0]: 80.0, keys[1]: 20.0})
    with pytest.raises(SurrogateError, match="Extrapolation refused"):
        sm.predict({keys[0]: 35.0, keys[1]: -3.0})
    r = sm.predict({keys[0]: 80.0, keys[1]: 20.0}, extrapolate=True)
    assert r["extrapolated"] and any("EXTRAPOLATION" in w for w in r["warnings"])
    with pytest.raises(SurrogateError, match="Unknown parameter"):
        sm.predict({**inside, "zzz": 1.0})
    with pytest.raises(SurrogateError, match="Missing parameter"):
        sm.predict({keys[0]: 35.0})
    with pytest.raises(SurrogateError, match="finite"):
        sm.predict({keys[0]: float("nan"), keys[1]: 20.0})


def test_inside_box_but_outside_the_hull_is_refused():
    """Two clusters of points leave a corner of the box empty: inside the box, outside the convex hull -> a refusal (never a narrow band)."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.random((25, 2)) * [1.0, 0.5], rng.random((25, 2)) * [0.5, 1.0] + [0.0, 0.0]])
    X[:, 0], X[:, 1] = X[:, 0] * 10, X[:, 1] * 10
    pts = [{"params": {"a": float(x), "b": float(y)}, "kpi": {"mass": float(1 + x + 0.5 * y)}} for x, y in X]
    sm = fit_surrogate(pts, backend="rbf")
    with pytest.raises(SurrogateError, match="convex hull"):
        sm.predict({"a": 9.8, "b": 9.8})
    r = sm.predict({"a": 9.8, "b": 9.8}, extrapolate=True)
    assert r["inside_hull"] is False and r["extrapolated"] and any("convex hull" in w for w in r["warnings"])
    assert not r["kpis"]["mass"]["reliable"]
    ok = sm.predict({"a": 2.0, "b": 2.0})
    assert ok["inside_hull"] is True and ok["inside_support"] and not ok["extrapolated"]


def test_unreliable_model_says_so():
    rng = np.random.default_rng(3)
    X = lhs_design(BOUNDS2, 50, seed=2)
    y = 10.0 + 40.0 * (X[:, 0] > 35) + 5.0 * rng.standard_normal(50)            # a jump plus noise: no smooth surrogate fits it
    pts = [{"params": {"a": float(a), "b": float(b)}, "kpi": {"max_disp": float(v)}} for (a, b), v in zip(X, y)]
    sm = fit_surrogate(pts, backend="rbf", kpis=["max_disp"])
    r = sm.predict({"a": 30.0, "b": 10.0})
    assert not sm.reliable and not r["kpis"]["max_disp"]["reliable"]
    assert any("not reliable" in w for w in r["warnings"]), r["warnings"]


def test_failed_points_and_missing_kpi_are_left_out():
    pts, keys = points(stress, BOUNDS2, 40, 0)
    pts[0]["converged"] = False
    pts[1]["kpi"]["max_stress"] = None
    pts[2]["kpi"]["max_stress"] = float("inf")
    sm = fit_surrogate(pts, backend="rbf")
    assert any("1 of 40 analyses failed" in w for w in sm.warnings)        # the region of the failed point is reported
    assert len(sm.rows) == 39                            # the failed point is not a row
    m = next(iter(sm.members.values()))["max_stress"]
    assert m.cal.n_train + m.cal.n_cal == 37             # two rows without a finite value


def test_unknown_kpi_is_an_error():
    pts, _ = points(stress, BOUNDS2, 40, 0)
    with pytest.raises(SurrogateError, match="Unknown KPI"):
        fit_surrogate(pts, kpis=["nope"], backend="rbf")


def test_gp_backend_without_scikit_learn_is_a_clear_error():
    if backends_available()["gp"]:
        pytest.skip("scikit-learn is installed")
    with pytest.raises(Exception, match="scikit-learn"):
        resolve_backend("gp")
    assert resolve_backend("auto") == "rbf"


# ------------------------------------------------------------------ categorical parameters
@pytest.mark.parametrize("backend", BACKENDS)
def test_categorical_parameter_gets_one_model_per_category(backend):
    X = lhs_design(BOUNDS2, 90, seed=0)
    cats = assign_categories(["steel", "alu", "ti"], 90, seed=0)
    off = {"steel": 0.0, "alu": 60.0, "ti": 30.0}
    pts = [{"params": {"a": float(X[i, 0]), "b": float(X[i, 1]), "mat": cats[i]}, "kpi": {"max_stress": float(stress(X[i:i + 1])[0] + off[cats[i]])}}
           for i in range(90)]
    sm = fit_surrogate(pts, backend=backend)
    assert len(sm.groups()) == 3
    for mat in off:
        r = sm.predict({"a": 35.0, "b": 20.0, "mat": mat})
        truth = float(stress(np.array([[35.0, 20.0]]))[0] + off[mat])
        assert abs(r["kpis"]["max_stress"]["value"] - truth) / truth < 0.05
    with pytest.raises(SurrogateError, match="not in the study|was not"):
        sm.predict({"a": 35.0, "b": 20.0, "mat": "brass"})
    d = sm.describe()
    assert {m["category"][0] for m in d["members"]} == set(off)


def test_a_category_with_too_few_points_is_left_out_with_a_warning():
    X = lhs_design(BOUNDS2, 60, seed=0)
    pts = [{"params": {"a": float(X[i, 0]), "b": float(X[i, 1]), "mat": "steel" if i < 52 else "rare"}, "kpi": {"mass": float(1 + X[i, 0])}}
           for i in range(60)]
    sm = fit_surrogate(pts, backend="rbf")
    assert any("rare" in w for w in sm.warnings)
    with pytest.raises(SurrogateError, match="rare"):
        sm.predict({"a": 30.0, "b": 10.0, "mat": "rare"})


# ------------------------------------------------------------------ determinism and persistence
@pytest.mark.parametrize("backend", BACKENDS)
def test_same_seed_same_model(backend):
    pts, keys = points(stress, BOUNDS2, 40, 5, noise=0.01)
    a = fit_surrogate(pts, backend=backend, seed=7)
    b = fit_surrogate(pts, backend=backend, seed=7)
    q = {keys[0]: 33.0, keys[1]: 17.0}
    assert a.predict(q)["kpis"] == b.predict(q)["kpis"]
    assert a.describe()["members"] == b.describe()["members"]
    c = fit_surrogate(pts, backend=backend, seed=8)
    assert c.members[()]["max_stress"].cal_ids != a.members[()]["max_stress"].cal_ids          # another split


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_roundtrip_without_pickle(backend, tmp_path):
    pts, keys = points(stress, BOUNDS2, 45, 2, noise=0.01)
    sm = fit_surrogate(pts, backend=backend, seed=1)
    f = tmp_path / "surrogate.npz"
    sm.save(f)
    with np.load(f, allow_pickle=False) as z:                      # loads without pickle = no object arrays inside
        assert set(z.files) == {"meta", "P", "K"} and all(z[k].dtype != object for k in z.files)
    sm2 = load_surrogate(f)
    q = {keys[0]: 41.0, keys[1]: 9.0}
    a, b = sm.predict(q)["kpis"]["max_stress"], sm2.predict(q)["kpis"]["max_stress"]
    for k in ("value", "lower", "upper"):
        assert b[k] == pytest.approx(a[k], rel=1e-9)
    assert sm2.describe()["members"] == sm.describe()["members"]


def test_load_refuses_damaged_files(tmp_path):
    f = tmp_path / "x.npz"
    np.savez(f, meta=np.array("{not json"), P=np.zeros((3, 2)), K=np.zeros((3, 7)))
    with pytest.raises(SurrogateError):
        load_surrogate(f)
    g = tmp_path / "y.npz"
    np.savez(g, a=np.zeros(3))
    with pytest.raises(SurrogateError):
        load_surrogate(g)
    pts, _ = points(stress, BOUNDS2, 40, 0)
    sm = fit_surrogate(pts, backend="rbf")
    h = tmp_path / "z.npz"
    sm.save(h)
    with np.load(h) as z:
        meta, P, K = str(z["meta"]), z["P"], z["K"]
    import json
    m = json.loads(meta)
    m["members"][0]["train"] = [0, 9999]
    np.savez(h, meta=np.array(json.dumps(m)), P=P, K=K)
    with pytest.raises(SurrogateError, match="damaged"):
        load_surrogate(h)


# ------------------------------------------------------------------ the rest of the API
def test_validation_table_curve_grid_suggest_and_refit():
    pts, keys = points(stress, BOUNDS2, 40, 4)
    sm = fit_surrogate(pts, backend="rbf", seed=0)
    tab = sm.validation_table("max_stress")
    assert len(tab) == max(validation_size(40), holdout_minimum()) == 9 and all({"analysed", "predicted", "lower", "upper", "inside"} <= set(r) for r in tab)
    c = sm.curve("max_stress", keys[0], {keys[1]: 20.0}, n=25)
    assert len(c["x"]) == 25 and np.all(c["lower"] <= c["value"]) and np.all(c["value"] <= c["upper"])
    g = sm.grid("max_stress", keys[0], keys[1], {}, n=12)
    assert g["value"].shape == (12, 12)
    with pytest.raises(SurrogateError):
        sm.grid("max_stress", keys[0], keys[0], {})
    nxt = sm.suggest(3, seed=1)
    assert len(nxt) == 3 and all(set(p) == set(keys) for p in nxt)
    new = [{"params": nxt[0], "kpi": {"max_stress": float(stress(np.array([[nxt[0][keys[0]], nxt[0][keys[1]]]]))[0]), **{k: None for k in ()}}}]
    new[0]["kpi"] = {k: new[0]["kpi"].get(k) for k in ("max_stress_clean", "max_stress", "max_disp", "min_safety", "mass", "freq1", "print_safety")}
    sm2 = sm.refit_with(new)
    assert len(sm2.rows) == 41
    m_old, m_new = sm.members[()]["max_stress"], sm2.members[()]["max_stress"]
    assert m_new.cal_ids == m_old.cal_ids and 40 in m_new.train_ids        # the hold-out set is kept, the new point is training data


def test_limitations_text_and_describe():
    pts, _ = points(stress, BOUNDS2, 40, 0)
    d = fit_surrogate(pts, backend="rbf").describe()
    assert "scatter" in LIMITATIONS and d["limitations"] == LIMITATIONS
    assert d["members"][0]["kpi"] == "max_stress" and d["level"] == 0.9 and d["n_points"] == 40


def test_load_refuses_crafted_files(tmp_path):
    """A foreign .fea can carry anything in surrogate.npz: ranges, indices and sizes are checked, nothing but SurrogateError comes out."""
    import json
    pts, _ = points(stress, BOUNDS2, 40, 0)
    sm = fit_surrogate(pts, backend="rbf")
    f = tmp_path / "ok.npz"
    sm.save(f)
    with np.load(f) as z:
        meta, P, K = json.loads(str(z["meta"])), z["P"], z["K"]

    def crafted(name, edit):
        m = json.loads(json.dumps(meta))
        edit(m)
        g = tmp_path / f"{name}.npz"
        np.savez(g, meta=np.array(json.dumps(m)), P=P, K=K)
        with pytest.raises(SurrogateError):
            load_surrogate(g)
    crafted("nan_range", lambda m: m["params"][0].update(lo=float("nan")))
    crafted("level", lambda m: m.update(level=2.0))
    crafted("members", lambda m: m.update(members=m["members"] * 300))
    crafted("hyper", lambda m: m["members"][0].update(hyper={}))
    crafted("missing_key", lambda m: m.pop("kpis"))


@pytest.mark.skipif(not backends_available()["gp"], reason="scikit-learn is not installed")
def test_numpy_gaussian_process_predicts_like_scikit_learn():
    """The prediction is numpy (no scikit-learn object is kept): the formulas must be those of scikit-learn's GaussianProcessRegressor."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from fea_core.surrogate.model import GPBackend
    X = lhs_design([(0.0, 1.0)] * 3, 40, seed=1)
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2]
    p = {"amplitude": 0.8, "length_scales": [0.4, 0.9, 1.7], "noise": 1e-3}
    mine = GPBackend().fit(X, y, params=p)
    k = ConstantKernel(0.8, "fixed") * Matern(length_scale=[0.4, 0.9, 1.7], length_scale_bounds="fixed", nu=2.5) + WhiteKernel(1e-3, "fixed")
    ref = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=1e-10).fit(X, y)
    Q = np.random.default_rng(0).random((50, 3)) * 1.4 - 0.2
    mu, sd = mine.predict(Q)
    mu_r, sd_r = ref.predict(Q, return_std=True)
    assert np.allclose(mu, mu_r, rtol=1e-8, atol=1e-10) and np.allclose(sd, np.maximum(sd_r, 1e-9), rtol=1e-6, atol=1e-8)
    with pytest.raises(Exception, match="out of range"):
        GPBackend().fit(X, y, params={**p, "noise": 5.0})


def test_calibration_from_dict_takes_only_declared_fields_with_their_types():
    """The old reader did setattr(c, k, v) for any key that the object had (to_dict, __class__, ...) and any type."""
    c = Calibration.from_dict({"to_dict": "x", "__class__": "y", "__dict__": {"q": 1}, "evil": 1, "q": "abc", "n_cal": 7.5, "n_train": True, "reliable": "yes",
                               "exact": False, "r2": None, "level": 0.8, "reasons": "<img src=x>", "median_rel_err": 0.25})
    assert callable(c.to_dict) and type(c) is Calibration and not hasattr(c, "evil")
    assert c.q == 1.0 and c.n_cal == 0 and c.n_train == 0 and c.reliable is False               # wrong types are ignored, defaults stay
    assert c.exact is False and math.isnan(c.r2) and c.level == 0.8 and c.median_rel_err == 0.25 and c.reasons == []
    assert Calibration.from_dict({"reasons": ["a" * 1000] * 50}).reasons == ["a" * 300] * 20      # bounded

"""Designs that are not space-filling (F15-S4 re-check N1, N5–N7): grids with few levels, one-at-a-time stars, holes in a Latin hypercube,
analyses without a value of the KPI, the hull of many points. The scenarios are those of the re-check (`probe_support.py`): the model must
refuse or flag where its calibration says nothing – never answer with a narrow band and `reliable: True`."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.surrogate import SurrogateError, fit_surrogate, lhs_design
from fea_core.surrogate import support as S
from fea_core.surrogate.model import backends_available

BACKENDS = ["rbf", pytest.param("gp", marks=pytest.mark.skipif(not backends_available()["gp"], reason="scikit-learn is not installed"))]


def pts(X, y, ok=None, kpi="max_stress", extra=None):
    keys = [f"p{i}" for i in range(X.shape[1])]
    out = []
    for i, (x, v) in enumerate(zip(X, y)):
        d = {"params": dict(zip(keys, map(float, x))), kpi: float(v), "converged": bool(True if ok is None else ok[i])}
        if extra is not None:
            d.update({k: (None if f[i] is None else float(f[i])) for k, f in extra.items()})
        out.append(d)
    return out


def grid(nx, ny):
    a, b = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    return np.c_[a.ravel(), b.ravel()]


def refused(sm, **q):
    try:
        sm.predict(q)
    except SurrogateError as e:
        return str(e)
    return ""


F_Y4 = lambda Z: 100 + 50 * Z[:, 0] + 300 * Z[:, 1] ** 4                       # noqa: E731
F_SIN = lambda Z: 200 + 50 * Z[:, 0] + 80 * np.sin(2 * np.pi * Z[:, 1])        # noqa: E731


# ------------------------------------------------------------------ grids: a parameter with few levels is answered AT its levels
@pytest.mark.parametrize("backend", BACKENDS)
def test_grid_20x3_is_refused_between_the_three_levels(backend):
    """The re-check case: sin(2 pi b) analysed at b = 0, 0.5, 1 only. The query b = 0.15 was answered with `reliable: True` and a band of 0 %."""
    X = grid(20, 3)
    sm = fit_surrogate(pts(X, F_SIN(X)), backend=backend, seed=1)
    assert any("p1 has only 3 analysed levels" in w for w in sm.warnings), sm.warnings
    for b in (0.15, 0.25, 0.4, 0.9):
        assert "between the analysed levels" in refused(sm, p0=0.5, p1=b), b
    # the levels themselves are analysed: answered, inside the band, with a mid-level value of the dense parameter
    for b in (0.0, 0.5, 1.0):
        r = sm.predict({"p0": 0.5, "p1": b})
        k = r["kpis"]["max_stress"]
        assert r["inside_support"] and k["lower"] <= F_SIN(np.array([[0.5, b]]))[0] <= k["upper"]


@pytest.mark.parametrize("nx,ny", [(30, 2), (15, 4), (20, 5)])
def test_coarse_grids_are_refused_off_the_levels(nx, ny):
    X = grid(nx, ny)
    sm = fit_surrogate(pts(X, F_Y4(X)), backend="rbf", seed=2)
    levels = np.linspace(0, 1, ny)
    for b in (levels[:-1] + np.diff(levels) / 2):
        assert "between the analysed levels" in refused(sm, p0=0.5, p1=float(b))
    assert refused(sm, p0=0.5, p1=float(levels[1])) == ""
    # a query at a level is covered like the calibration points: the coverage of the band over the answered queries
    rng = np.random.default_rng(0)
    Q = np.c_[rng.random(600), rng.choice(levels, 600)]
    Q = Q[sm.supported("max_stress", (), Q)]
    assert len(Q) > 400
    _, a, b = sm.predict_arrays("max_stress", (), Q)
    assert np.mean((a <= F_Y4(Q)) & (F_Y4(Q) <= b)) >= 0.75


def test_a_grid_with_enough_levels_is_still_answered_between_them():
    """"Unless the levels are dense": 7 x 7 levels are interpolated."""
    X = grid(7, 7)
    sm = fit_surrogate(pts(X, F_Y4(X)), backend="rbf", seed=1)
    assert sm.members[()]["max_stress"].support.kind == "space-filling"
    r = sm.predict({"p0": 0.42, "p1": 0.58})
    assert r["inside_support"] and r["kpis"]["max_stress"]["lower"] <= F_Y4(np.array([[0.42, 0.58]]))[0] <= r["kpis"]["max_stress"]["upper"]


def test_anisotropic_grid_is_refused_between_the_coarse_rows():
    """Per-direction spacing: 30 values of one parameter, 7 of the other – the median spacing of the points is that of the dense direction,
    the cell centre is far from every point in units of it."""
    X = grid(30, 7)
    sm = fit_surrogate(pts(X, F_Y4(X)), backend="rbf", seed=1)
    assert refused(sm, p0=0.5, p1=1 / 12)                                  # the centre of a cell between two rows
    assert refused(sm, p0=0.5, p1=1 / 6) == ""                            # on a row


# ------------------------------------------------------------------ one-at-a-time ("star") designs
@pytest.mark.parametrize("backend", BACKENDS)
def test_star_design_answers_on_its_lines_only(backend):
    t = np.linspace(0, 1, 20)
    X = np.vstack([np.c_[t, np.zeros(20)], np.c_[np.zeros(19), t[1:]]])
    f = lambda Z: 100 + 60 * Z[:, 0] + 40 * Z[:, 1] + 150 * Z[:, 0] * Z[:, 1]       # noqa: E731
    sm = fit_surrogate(pts(X, f(X)), backend=backend, seed=1)
    assert sm.members[()]["max_stress"].support.kind == "lines"
    assert any("one parameter at a time" in w for w in sm.warnings), sm.warnings
    for q in ({"p0": 0.5, "p1": 0.1}, {"p0": 0.3, "p1": 0.3}, {"p0": 0.9, "p1": 0.02}):
        assert "off the lines" in refused(sm, **q), q
    for q in ({"p0": 0.5, "p1": 0.0}, {"p0": 0.0, "p1": 0.7}):
        assert refused(sm, **q) == "", q
    # no random query in the box is answered unless it lies on a line
    Q = np.random.default_rng(0).random((800, 2))
    assert not sm.supported("max_stress", (), Q).any()


# ------------------------------------------------------------------ holes in a Latin hypercube
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("r", [0.2, 0.25])
def test_an_empty_disc_in_a_latin_hypercube_is_refused(backend, r):
    """The re-check case: the points of a 75-point Latin hypercube within r of the centre are missing, and the response has a peak there.
    The interior of the hole is not answered (it was, with coverage 0 and `reliable: True` for half of the fits)."""
    share = []
    for s in range(3 if backend == "gp" else 6):
        X = lhs_design([(0, 1), (0, 1)], 75, seed=s)
        X = X[np.linalg.norm(X - 0.5, axis=1) > r]
        f = lambda Z: 100 + 200 * np.exp(-((Z[:, 0] - 0.5) ** 2 + (Z[:, 1] - 0.5) ** 2) / 0.01)      # noqa: E731
        sm = fit_surrogate(pts(X, f(X)), backend=backend, seed=s)
        Q = 0.5 + (np.random.default_rng(5).random((1500, 2)) - 0.5) * 2 * 0.7 * r
        share.append(sm.supported("max_stress", (), Q).mean())
        assert "gap in the design" in refused(sm, p0=0.5, p1=0.5)
    assert np.mean(share) < 0.1 and max(share) < 0.2, share


def test_ordinary_latin_hypercubes_are_not_refused_by_the_hole_rule():
    for d, n in ((1, 40), (2, 19), (2, 60), (3, 60), (4, 60), (5, 112)):
        for seed in range(6):
            X = lhs_design([(0.0, 1.0)] * d, n, seed=seed)
            sup = S.Support(X, X)
            Q = np.random.default_rng(seed).random((800, d))
            inside = sup.in_hull(Q)
            assert sup.supported(Q[inside]).all(), (d, n, seed)


def test_two_clusters_with_a_gap_between_them():
    A = lhs_design([(0, 0.3), (0, 1)], 30, seed=1)
    B = lhs_design([(0.7, 1.0), (0, 1)], 30, seed=2)
    X = np.vstack([A, B])
    sm = fit_surrogate(pts(X, 100 + 80 * np.sin(6 * X[:, 0]) + 20 * X[:, 1]), backend="rbf", seed=1)
    assert "gap in the design" in refused(sm, p0=0.5, p1=0.5)
    assert any("empty ball" in w for w in sm.warnings), sm.warnings


# ------------------------------------------------------------------ the edge of the range
def test_the_outer_edge_shell_is_not_reliable_and_says_so():
    X = lhs_design([(0.0, 1.0), (0.0, 1.0)], 40, seed=1)
    sm = fit_surrogate(pts(X, 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2), backend="rbf")
    lo, hi = sm.continuous[0].lo, sm.continuous[0].hi
    edge = sm.predict({"p0": lo + 0.01 * (hi - lo), "p1": 0.5})
    assert edge["near_edge"] and not edge["kpis"]["max_stress"]["reliable"] and any("not marked reliable" in w for w in edge["warnings"])
    mid = sm.predict({"p0": 0.5, "p1": 0.5})
    assert not mid["near_edge"] and mid["kpis"]["max_stress"]["reliable"]


# ------------------------------------------------------------------ N5: analyses without a value of the KPI are unknown regions too
def test_rows_without_a_value_of_the_kpi_count_like_failed_points():
    X = lhs_design([(0.0, 1.0), (0.0, 1.0)], 150, seed=3)
    near = np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.5) < 0.12
    stress = 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2
    disp = np.where(near, np.nan, 1.0 + 0.5 * X[:, 0])
    sm = fit_surrogate(pts(X, stress, extra={"max_disp": [None if np.isnan(v) else v for v in disp]}), backend="rbf", seed=1)
    assert set(sm.members[()]) >= {"max_stress", "max_disp"}
    sup = sm.members[()]["max_disp"].support
    assert len(sup.failed) == int(near.sum()) and sup.near_failed(np.array([[0.5, 0.5]]))[0] and sup.near_failed(np.array([[0.51, 0.49]]))[0]
    with pytest.raises(SurrogateError, match="failed analysis|gap in the design"):
        sm.predict({"p0": 0.5, "p1": 0.5}, kpis=["max_disp"])
    assert sm.predict({"p0": 0.5, "p1": 0.5}, kpis=["max_stress"])["inside_support"]


# ------------------------------------------------------------------ N6 / N7: the hull of many points, memory in blocks
@pytest.mark.parametrize("d", [2, 3])
def test_hull_of_many_points_is_exact_and_needs_no_linear_programs(d, monkeypatch):
    n = 900
    X = np.random.default_rng(3).random((n, d)) ** 0.7
    Q = np.random.default_rng(4).random((700, d)) * 1.4 - 0.2
    full = S.Support(X, X)
    monkeypatch.setattr(S, "HULL_MAX_POINTS", 10 ** 6)
    exact = full.in_hull(Q)                                                   # Qhull of all the points
    monkeypatch.setattr(S, "HULL_MAX_POINTS", 50)
    grown = S.Support(X, X)
    assert np.array_equal(grown.in_hull(Q), exact) and 0 < exact.sum() < len(Q)
    assert grown._eq is not False and grown._eq is not None                  # Qhull facets, not one linear program per query


def test_temporary_matrices_are_processed_in_blocks(monkeypatch):
    X = lhs_design([(0.0, 1.0)] * 3, 60, seed=0)
    Q = np.random.default_rng(0).random((1000, 3))
    a = S.Support(X, X).supported(Q)
    monkeypatch.setattr(S, "CHUNK", 1000)                                     # blocks of a few rows
    b = S.Support(X, X)
    assert np.array_equal(b.supported(Q), a)
    assert all(s.stop - s.start <= 1000 for s in S._blocks(10_000, 3)) and len(S._blocks(10_000, 3)) > 1


def test_the_peak_memory_of_many_points_and_many_queries_is_bounded():
    """1 500 points in 4 parameters (1 000 Qhull facets) and 30 000 queries: the temporary distance / facet matrices are processed in blocks
    of CHUNK elements (16 MB), the numpy peak stays far below what one (queries x facets x parameters) array would take (~1 GB)."""
    import tracemalloc
    X = lhs_design([(0.0, 1.0)] * 4, 1500, seed=1)
    Q = np.random.default_rng(0).random((30_000, 4))
    tracemalloc.start()
    try:
        sup = S.Support(X, X)
        ok = sup.supported(Q)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert ok.mean() > 0.5 and len(sup._eq) > 500                           # the hull is the Qhull of a subset grown to the whole set
    assert peak < 120e6, peak / 1e6


def test_a_model_of_2000_points_answers_a_surface_quickly():
    import time
    X = lhs_design([(0.0, 1.0)] * 3, 2000, seed=1)
    sup = S.Support(X, X)
    g = np.stack(np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40)), -1).reshape(-1, 2)
    Q = np.c_[g, np.full(len(g), 0.5)]
    t = time.time()
    ok = sup.supported(Q)
    assert time.time() - t < 20 and ok.any()


# ------------------------------------------------------------------ N10: the report states the rule of THIS fit
class _Doc:
    def __init__(self):
        self.paras: list[str] = []

    def p(self, text="", **kw):
        self.paras.append(text)

    def table(self, *a, **kw):
        pass


def _report(sm):
    from types import SimpleNamespace

    from fea_core.report.surrogate import surrogate_section
    D = _Doc()
    surrogate_section(D, SimpleNamespace(surrogate=sm, study=None, step_path=None))
    return D.paras


def test_report_prints_the_actual_holdout_rule_and_the_design_limits():
    X = lhs_design([(0.1, 0.5), (40, 100)], 30, seed=3)
    y = 100 + 50 * X[:, 0] + 20 * X[:, 1]
    default = _report(fit_surrogate(pts(X, y), backend="rbf", seed=3))
    assert any("(at least 9, chosen at random" in p for p in default)
    few = _report(fit_surrogate(pts(X, y), backend="rbf", seed=3, validation_fraction=0.05, min_holdout=5))
    assert any("(at least 5, chosen at random" in p for p in few) and not any("at least 9" in p for p in few)
    g = grid(20, 3)
    paras = _report(fit_surrogate(pts(g, F_SIN(g)), backend="rbf", seed=1))
    assert any("fewer than 6 levels" in p for p in paras)
    t = np.linspace(0, 1, 20)
    star = np.vstack([np.c_[t, np.zeros(20)], np.c_[np.zeros(19), t[1:]]])
    assert any("one parameter at a time" in p for p in _report(fit_surrogate(pts(star, 100 + 60 * star[:, 0] + 40 * star[:, 1]), backend="rbf", seed=1)))


def test_irregular_designs_are_not_refused_wholesale():
    """The points of ONE category of a split design (a random quarter of a Latin hypercube) or random points have larger natural gaps than a
    Latin hypercube: only the core of the largest gap goes (the gap scale follows the design), not a third of the region as with the median alone."""
    for d, n in ((1, 24), (2, 24), (2, 60), (3, 40)):
        share = []
        for seed in range(12):
            X = lhs_design([(0.0, 1.0)] * d, 4 * n, seed=seed)
            X = X[np.random.default_rng(seed).permutation(4 * n)[:n]]
            sup = S.Support(X, X)
            Q = np.random.default_rng(seed + 50).random((1000, d))
            Q = Q[sup.in_hull(Q)]
            share.append(1 - sup.supported(Q).mean())
        assert np.mean(share) < 0.08, (d, n, share)


# ------------------------------------------------------------------ N1: no case of the re-check is answered `reliable` with a low coverage
def _reliable_answers(sm, Q, f, limit=40):
    """(number of answered queries, number answered with `reliable: True`, number of those inside the band) for the rows of Q."""
    sup = sm.supported("max_stress", (), Q)
    n_rel = n_hit = 0
    for i in np.flatnonzero(sup)[:limit]:
        k = sm.predict({"p0": float(Q[i, 0]), "p1": float(Q[i, 1])})["kpis"]["max_stress"]
        if k["reliable"]:
            n_rel += 1
            n_hit += int(k["lower"] <= f(Q[i:i + 1])[0] <= k["upper"])
    return int(sup.sum()), n_rel, n_hit


def test_the_rim_of_a_hole_is_answered_but_never_reliable():
    """The queries at the rim of an empty disc are as close to data as ordinary ones, so they are answered – but the data lie on one side
    only and the peak inside the hole is not seen: they carry a warning and are not `reliable` (the re-check: coverage 0 with `reliable: true`)."""
    f = lambda Z: 100 + 200 * np.exp(-((Z[:, 0] - 0.5) ** 2 + (Z[:, 1] - 0.5) ** 2) / 0.01)      # noqa: E731
    saw_rim = False
    for r in (0.2, 0.25):
        for s in range(8):
            X = lhs_design([(0, 1), (0, 1)], 75, seed=s)
            X = X[np.linalg.norm(X - 0.5, axis=1) > r]
            sm = fit_surrogate(pts(X, f(X)), backend="rbf", seed=s)
            Q = 0.5 + (np.random.default_rng(900 + s).random((600, 2)) - 0.5) * 2 * 0.7 * r
            n_ans, n_rel, n_hit = _reliable_answers(sm, Q, f)
            assert n_rel == 0 or n_hit / n_rel >= 0.7, (r, s, n_rel, n_hit)
            sup = sm.supported("max_stress", (), Q)
            for i in np.flatnonzero(sup)[:5]:
                res = sm.predict({"p0": float(Q[i, 0]), "p1": float(Q[i, 1])})
                saw_rim |= any("rim of an empty region" in w for w in res["warnings"])
    assert saw_rim


@pytest.mark.parametrize("name,X,f", [
    ("grid 20x3 sin", grid(20, 3), F_SIN), ("grid 30x2 y^4", grid(30, 2), F_Y4), ("grid 15x4 y^4", grid(15, 4), F_Y4),
    ("star with an interaction", np.vstack([np.c_[np.linspace(0, 1, 20), np.zeros(20)], np.c_[np.zeros(19), np.linspace(0, 1, 20)[1:]]]),
     lambda Z: 100 + 60 * Z[:, 0] + 40 * Z[:, 1] + 150 * Z[:, 0] * Z[:, 1]),
])
def test_the_reviewer_designs_never_answer_reliably_with_a_low_coverage(name, X, f):
    """Random queries over the whole box: whatever the model answers and calls reliable lies inside the band at least 70 % of the time
    (the stated level is 90 %); the coarse designs answer only AT their levels / lines."""
    sm = fit_surrogate(pts(X, f(X)), backend="rbf", seed=2)
    Q = np.random.default_rng(3).random((1500, 2))
    levels = np.unique(X[:, 1]) if "grid" in name else None
    if levels is not None:
        Q[:, 1] = np.random.default_rng(4).choice(levels, len(Q)) + (np.arange(len(Q)) % 2) * 0.013       # half of the queries are off the levels
    n_ans, n_rel, n_hit = _reliable_answers(sm, Q, f)
    off = Q[sm.supported("max_stress", (), Q)]
    if levels is not None:
        assert not np.any(np.min(np.abs(off[:, 1:2] - levels[None, :]), axis=1) > 1e-6), "an off-level query was answered"
    assert (n_ans == 0) == ("star" in name)
    assert n_rel == 0 or n_hit / n_rel >= 0.7, (name, n_rel, n_hit)

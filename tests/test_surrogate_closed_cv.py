"""1.5.1 optimisation of the surrogate fit: the cross-validation of the backends in closed form (one matrix inverse) instead of a refit per left-out point.
The loops of refits are the reference; the closed form must give the same figures (~1e-9 relative) and the same choice of the smoothing."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.surrogate import fit as F
from fea_core.surrogate import model as M
from fea_core.surrogate.calibrate import scores_of

pytest.importorskip("scipy")


def _design(n: int, d: int, seed: int, kind: str = "lhs") -> np.ndarray:
    from scipy.stats import qmc
    if kind == "lhs":
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    return np.random.default_rng(seed).uniform(size=(n, d))


def _response(X: np.ndarray, seed: int, noise: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    f = np.sin(3.0 * X[:, 0]) + (X[:, 1] ** 2 if X.shape[1] > 1 else 0.0) + noise * rng.normal(size=len(X))
    return (f - f.mean()) / f.std()


def _loop_errors(X, y, smoothing, folds):
    """prediction - data when the fold is left out: the refit loop of the old `_cv_rmse`."""
    err = np.empty(len(y))
    for f in np.unique(folds):
        te = folds == f
        err[te] = M.RBFBackend._interpolator(X[~te], y[~te], smoothing)(X[te]) - y[te]
    return err


CASES = [(n, d, seed, kind) for n in (5, 8, 15, 40, 96) for d in (1, 2, 4) for seed in (0, 1) for kind in ("lhs", "random") if n >= d + 3]


@pytest.mark.parametrize("n,d,seed,kind", CASES)
def test_rbf_leave_one_out_errors_equal_the_refits(n, d, seed, kind):
    X = _design(n, d, seed, kind)
    y = _response(X, seed)
    folds = np.arange(n)
    for s in M.RBFBackend.SMOOTHING:
        fast = M.rbf_cv_errors(X, y, s, folds)
        if fast is None:                                    # badly conditioned: the callers use the loop
            continue
        ref = _loop_errors(X, y, s, folds)
        assert np.max(np.abs(fast - ref)) <= 1e-8 * max(np.max(np.abs(ref)), 1e-3), (s, np.max(np.abs(fast - ref)))


@pytest.mark.parametrize("n,d", [(130, 2), (240, 4)])
def test_rbf_ten_fold_errors_equal_the_refits(n, d):
    X = _design(n, d, 3)
    y = _response(X, 3)
    folds = np.arange(n) % 10
    for s in (0.0, 1e-4, 1e-1):
        fast = M.rbf_cv_errors(X, y, s, folds)
        assert fast is not None
        ref = _loop_errors(X, y, s, folds)
        assert np.max(np.abs(fast - ref)) <= 1e-8 * np.max(np.abs(ref))


@pytest.mark.parametrize("n,d,seed,kind", [c for c in CASES if c[0] in (8, 40, 96)])
def test_rbf_selects_the_same_smoothing_and_rmse_as_the_loop(n, d, seed, kind):
    X = _design(n, d, seed, kind)
    y = _response(X, seed)
    be = M.RBFBackend()
    loop = [be._cv_rmse(X, y, s) for s in be.SMOOTHING]                       # the earlier selection
    k = int(np.argmin(loop))
    new = M.RBFBackend().fit(X, y)
    assert new.smoothing == be.SMOOTHING[k]
    assert new.rmse == pytest.approx(max(loop[k], 1e-6), rel=1e-9)


def test_rbf_exact_data_and_ties_are_decided_by_the_loop():
    X = _design(30, 2, 5)
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1]                                     # a plane: every smoothing is exact up to rounding, the winner is a tie
    y = (y - y.mean()) / y.std()
    be = M.RBFBackend()
    loop = [be._cv_rmse(X, y, s) for s in be.SMOOTHING]
    new = M.RBFBackend().fit(X, y)
    assert new.smoothing == be.SMOOTHING[int(np.argmin(loop))]


def test_singular_points_give_the_same_error_as_before():
    X = _design(12, 2, 0)
    X[5] = X[2]                                                                 # a duplicate point: the interpolation is singular
    y = _response(X, 0)
    assert M.rbf_cv_errors(X, y, 0.0, np.arange(12)) is None
    ref = [M.RBFBackend()._cv_rmse(X, y, s) for s in M.RBFBackend.SMOOTHING]
    if not np.any(np.isfinite(ref)):
        with pytest.raises(M.BackendError, match="singular"):
            M.RBFBackend().fit(X, y)
    else:                                                                       # a smoothing regularises it: both paths choose alike
        new = M.RBFBackend().fit(X, y)
        assert new.smoothing == M.RBFBackend.SMOOTHING[int(np.argmin(ref))]


# ---- singular / rank-deficient folds: the closed form must not invent a finite error where the refit is singular -------------------------------------------------
def _degenerate_designs():
    """(label, X, y, folds): leaving a fold out makes the refit singular (or nearly so) although the full set is regular."""
    out = []
    for d in (1, 2, 3):                                                         # a simplex: n = d + 1 points, every refit has d points left
        X = np.vstack([np.zeros(d), np.eye(d)])
        out.append((f"simplex d={d}", X, np.arange(1.0, d + 2.0), np.arange(d + 1)))
    X = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10)])               # all points but one on a line
    X[9, 1] = 1.0
    out.append(("collinear-minus-1", X, np.sin(3.0 * X[:, 0]) + 2.0 * X[:, 1], np.arange(10)))
    X = np.column_stack([np.linspace(0.0, 1.0, 130), np.zeros(130)])             # 10-fold: one fold holds every point off the line
    X[0::10, 1] = 1.0
    out.append(("degenerate 10-fold", X, np.sin(3.0 * X[:, 0]) + 2.0 * X[:, 1], np.arange(130) % 10))
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8):                                  # nearly on a line: the loop gives round-off, the closed form must not claim better
        X = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10)])
        X[8, 1], X[9, 1] = eps, 1.0
        out.append((f"near-collinear {eps:g}", X, np.sin(3.0 * X[:, 0]) + 2.0 * X[:, 1], np.arange(10)))
    X = np.random.default_rng(0).uniform(size=(9, 3))                            # three coordinates, one fold of 3 leaves 6 points on a plane
    X[:6, 2] = 0.5
    out.append(("plane + fold", X, X[:, 0] + X[:, 1], np.arange(9) % 3))
    return out


DEGENERATE = _degenerate_designs()
DEGENERATE_IDS = [c[0] for c in DEGENERATE]


def _old_fit_outcome(X, y):
    """What the fit did before the closed form: the loop of refits for every smoothing, the smallest error wins, all infinite = BackendError."""
    be = M.RBFBackend()
    scores = [be._cv_rmse(X, y, s) for s in be.SMOOTHING]
    if not np.any(np.isfinite(scores)):
        return "error", None, None
    k = int(np.argmin(scores))
    return "ok", be.SMOOTHING[k], max(float(scores[k]), 1e-6)


@pytest.mark.parametrize("label,X,y,folds", DEGENERATE, ids=DEGENERATE_IDS)
def test_degenerate_folds_are_not_answered_by_the_closed_form(label, X, y, folds, recwarn):
    be = M.RBFBackend()
    for s in be.SMOOTHING:
        fast = M.rbf_cv_errors(X, y, s, folds)
        if fast is None:
            continue
        ref = np.empty(len(y))                                                  # an answer is only allowed where it is the answer of the refits
        for f in np.unique(folds):
            te = folds == f
            ref[te] = be._interpolator(X[~te], y[~te], s)(X[te]) - y[te]
        assert np.max(np.abs(fast - ref)) <= 1e-9 * max(np.max(np.abs(ref)), 1e-3), (label, s)
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)], [str(w.message) for w in recwarn.list]


@pytest.mark.parametrize("label,X,y,folds", DEGENERATE, ids=DEGENERATE_IDS)
def test_fit_outcome_of_degenerate_designs_equals_the_loop_of_refits(label, X, y, folds):
    kind, smoothing, rmse = _old_fit_outcome(X, y)
    if kind == "error":
        with pytest.raises(M.BackendError, match="singular"):
            M.RBFBackend().fit(X, y)
        return
    new = M.RBFBackend().fit(X, y)
    assert new.smoothing == smoothing
    assert new.rmse == pytest.approx(rmse, rel=1e-9)


def test_degenerate_designs_that_are_singular_stay_an_error():
    """The classes of the review: simplex, collinear-minus-one and the degenerate fold must NOT fit (the base raised BackendError)."""
    for label, X, y, _folds in DEGENERATE:
        if label.startswith(("simplex", "collinear", "degenerate")):
            with pytest.raises(M.BackendError, match="singular"):
                M.RBFBackend().fit(X, y)


def test_fit_of_degenerate_designs_matches_the_base_tree():
    """Optional: set FEA_BASE_TREE to a checkout from before the closed form; the outcome (error / smoothing / rmse) of every degenerate design must match it."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    base = os.environ.get("FEA_BASE_TREE")
    if not base:
        pytest.skip("FEA_BASE_TREE not set")
    code = "\n".join([
        "import sys, json, warnings",
        "warnings.simplefilter('ignore')",
        "sys.path.insert(0, sys.argv[1])",
        "from fea_core.surrogate import model as M",
        "designs = json.loads(sys.stdin.read())",
        "import numpy as np",
        "out = {}",
        "for label, X, y in designs:",
        "    try:",
        "        b = M.RBFBackend().fit(np.array(X), np.array(y)); out[label] = ['ok', b.smoothing, b.rmse]",
        "    except M.BackendError:",
        "        out[label] = ['error', None, None]",
        "print(json.dumps(out))",
    ])
    payload = json.dumps([[c[0], c[1].tolist(), c[2].tolist()] for c in DEGENERATE])
    here = str(Path(__file__).resolve().parent.parent)
    res = {}
    for name, root in (("base", base), ("new", here)):
        r = subprocess.run([sys.executable, "-c", code, root], input=payload, capture_output=True, text=True, check=True, encoding="utf-8")
        res[name] = json.loads(r.stdout.strip().splitlines()[-1])
    for label, b in res["base"].items():
        n = res["new"][label]
        assert n[0] == b[0] and n[1] == b[1], (label, b, n)
        if b[0] == "ok":
            assert n[2] == pytest.approx(b[2], rel=1e-9)


def test_closed_form_does_not_lose_regular_designs_to_the_new_guards():
    """The guards must not refuse a regular design (the speed-up stays)."""
    for n, d in ((15, 2), (40, 3), (96, 4), (130, 2)):
        X = _design(n, d, 7)
        y = _response(X, 7)
        folds = np.arange(n) if n <= 120 else np.arange(n) % 10
        for s in (1e-4, 1e-2, 1.0):
            assert M.rbf_cv_errors(X, y, s, folds) is not None, (n, d, s)


def test_a_single_fold_is_left_to_the_loop():
    X = _design(20, 2, 0)
    y = _response(X, 0)
    assert M.rbf_cv_errors(X, y, 0.0, np.zeros(20, dtype=int)) is None            # nothing left to refit on
    be = M.restore_backend("gp", X, y, {"amplitude": 1.0, "noise": 1e-2, "length_scales": [0.5, 0.5]})
    assert be.cv_predict(X, y, np.zeros(20, dtype=int)) is None


def test_cv_scores_settle_a_tie_between_a_loop_score_and_a_lone_closed_form_winner(monkeypatch):
    """A smoothing the closed form refused (loop value) ties the lone closed-form winner within round-off: the winner must be the loop's, not the closed form's."""
    be = M.RBFBackend()
    sm = be.SMOOTHING
    fast = {sm[0]: None, sm[1]: 1.0 + 3e-12, sm[2]: 2.0}                         # the lone closed-form winner is within 1e-8 (relative) of the loop's value of sm[0]
    for s in sm[3:]:
        fast[s] = 5.0
    loop_calls = []

    def loop(self, X, y, s):
        loop_calls.append(s)
        return {sm[0]: 1.0, sm[1]: 1.0 + 2e-12}.get(s, 9.0)

    monkeypatch.setattr(M.RBFBackend, "_cv_rmse_closed", lambda self, X, y, s: fast[s])
    monkeypatch.setattr(M.RBFBackend, "_cv_rmse", loop)
    scores = be._cv_scores(np.zeros((3, 1)), np.zeros(3))
    assert sm[1] in loop_calls and sm[0] in loop_calls and sm[2] not in loop_calls
    assert scores[1] == 1.0 + 2e-12 and int(np.argmin(scores)) == 0


GP_PARAMS = [{"amplitude": 1.0, "noise": 1e-3, "ls": 0.5}, {"amplitude": 3.0, "noise": 1e-2, "ls": 0.8}, {"amplitude": 0.5, "noise": 5e-2, "ls": 0.3}]


@pytest.mark.parametrize("n,d,seed", [(8, 1, 0), (15, 2, 1), (40, 3, 2), (96, 4, 3), (130, 2, 4)])
@pytest.mark.parametrize("hp", GP_PARAMS)
def test_gp_cv_prediction_equals_the_refits(n, d, seed, hp):
    X = _design(n, d, seed)
    y = _response(X, seed)
    p = {"amplitude": hp["amplitude"], "noise": hp["noise"], "length_scales": [hp["ls"] * (1 + 0.2 * j) for j in range(d)]}
    be = M.restore_backend("gp", X, y, p)
    k = n if n <= 100 else 10
    folds = np.arange(n) % k
    fast = be.cv_predict(X, y, folds)
    assert fast is not None
    mu, sd = np.empty(n), np.empty(n)
    for f in range(k):
        te = folds == f
        mu[te], sd[te] = M.restore_backend("gp", X[~te], y[~te], be.params()).predict(X[te])
    assert np.max(np.abs(fast[0] - mu)) <= 1e-8 * max(1.0, np.max(np.abs(mu)))
    assert np.max(np.abs(fast[1] - sd) / sd) <= 1e-7
    assert np.array_equal(scores_of(y, mu, sd) <= 2.0, scores_of(y, *fast) <= 2.0)


def test_gp_that_is_badly_conditioned_is_left_to_the_loop():
    X = _design(60, 2, 0)
    y = _response(X, 0)
    be = M.restore_backend("gp", X, y, {"amplitude": 10.0, "noise": 1e-9, "length_scales": [8.0, 8.0]})
    assert be.cv_predict(X, y, np.arange(60)) is None


def _rows(n, seed=0, kpis=("max_stress", "max_disp", "mass")):
    rng = np.random.default_rng(seed)
    X = _design(n, 3, seed) * 10.0
    out = []
    for i in range(n):
        out.append({"params": {"a": float(X[i, 0]), "b": float(X[i, 1]), "c": float(X[i, 2])},
                    "kpi": {k: float(10.0 + np.sin(0.3 * X[i, 0] + j) + 0.1 * (X[i, 1] - 5.0) ** 2 + 0.05 * X[i, 2] + 0.01 * rng.normal()) for j, k in enumerate(kpis)}})
    return out


def _summary(sm):
    out = {}
    for g, ms in sm.members.items():
        for k, m in ms.items():
            c = m.cal
            out[(g, k)] = (m.backend.params(), c.q, c.coverage_cv, c.coverage_cal, c.r2, c.median_rel_err, c.max_rel_err, c.half_band, c.reliable, list(m.train_ids), list(m.cal_ids))
    return out


def _close(a, b):
    if isinstance(a, dict):
        return all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == pytest.approx(b, rel=1e-8, abs=1e-12)
    return a == b


@pytest.mark.parametrize("n", [30, 60, 130])
def test_fit_is_the_same_with_and_without_the_closed_form(n, monkeypatch):
    rows = _rows(n)
    kp = ["max_stress", "max_disp", "mass"]
    new = F.fit_surrogate(rows, ["a", "b", "c"], kp, backend="rbf")
    monkeypatch.setattr(M, "rbf_cv_errors", lambda *a, **k: None)             # every closed form refused: the loops of refits, as before
    old = F.fit_surrogate(rows, ["a", "b", "c"], kp, backend="rbf")
    assert _close(_summary(new), _summary(old))
    for g in new.members:
        for k in new.members[g]:
            q = np.array([[3.0, 4.0, 5.0], [6.5, 2.0, 8.0]])
            assert np.allclose(new.members[g][k].backend.predict(q / 10.0)[0], old.members[g][k].backend.predict(q / 10.0)[0], rtol=1e-12)


def test_gp_fit_cross_validation_is_the_same_with_and_without_the_closed_form(monkeypatch):
    """The Gaussian-process hyperparameters need scikit-learn; with fixed ones (`restore_backend`) the CV of `_fit_member` is compared directly."""
    rows = _rows(40)
    sm = F.fit_surrogate(rows, ["a", "b", "c"], ["max_stress"], backend="rbf")
    sm.backend = "gp"
    hyper = {"amplitude": 1.5, "noise": 5e-3, "length_scales": [0.8, 1.2, 1.5]}
    m = sm.members[()]["max_stress"]
    new = F._fit_member(sm, (), "max_stress", m.train_ids, m.cal_ids, hyper=hyper)
    monkeypatch.setattr(M.GPBackend, "cv_predict", lambda self, *a, **k: None)
    old = F._fit_member(sm, (), "max_stress", m.train_ids, m.cal_ids, hyper=hyper)
    assert new.cal.coverage_cv == old.cal.coverage_cv and not np.isnan(new.cal.coverage_cv)
    assert new.cal.q == old.cal.q


def test_kpis_on_the_same_points_share_one_support_and_a_kpi_with_gaps_has_its_own():
    rows = _rows(40)
    for i in (3, 11):
        rows[i]["kpi"]["mass"] = None                                            # two analyses without a value of this KPI
    sm = F.fit_surrogate(rows, ["a", "b", "c"], ["max_stress", "max_disp", "mass"], backend="rbf")
    ms = sm.members[()]
    assert ms["max_stress"].support is ms["max_disp"].support
    assert ms["mass"].support is not ms["max_stress"].support
    assert sm.members[()]["max_stress"].support.U.shape[0] == len(ms["max_stress"].train_ids) + len(ms["max_stress"].cal_ids)

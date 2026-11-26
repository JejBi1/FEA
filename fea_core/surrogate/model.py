"""Regression backends of the surrogate (F15-S4): a Gaussian process (hyperparameters by scikit-learn, optional module `surrogate`;
prediction in plain numpy) and a scipy radial-basis interpolation as the fallback. Both work on inputs scaled to the unit box and
standardised outputs and return (mean, σ); the conformal calibration (calibrate.py) turns σ into a band that holds, whatever the backend.

A backend can be rebuilt from `params()` WITHOUT optimisation: that is how a saved model is restored deterministically (no pickle)
and how the cross-validation refits cheaply. The Gaussian process keeps only numpy arrays (the training inputs, the Cholesky factor
and the weights) after the fit: scikit-learn is needed to OPTIMISE the hyperparameters, never to predict or to restore a file, and no
scikit-learn object stays in the memory of the GUI or the MCP process (the module itself stays imported once it was used).
"""
from __future__ import annotations

import importlib.util
import math
import warnings
from typing import Any, cast

import numpy as np

# hyperparameter bounds of the Gaussian process (the optimiser and the check of a loaded file use the same numbers)
AMP_BOUNDS = (1e-3, 1e3)
LS_BOUNDS = (0.03, 30.0)
NOISE_BOUNDS = (1e-9, 0.1)
JITTER = 1e-10                                      # added to the diagonal of the covariance matrix
RBF_SMOOTHING_MAX = 1.0
RBF_RMSE_MAX = 1e6


class BackendError(RuntimeError):
    """The backend cannot be fitted (singular data, library missing, hyperparameters out of range)."""


def backends_available() -> dict[str, bool]:
    """{"gp": scikit-learn installed, "rbf": True}. Does not import scikit-learn."""
    return {"gp": importlib.util.find_spec("sklearn") is not None, "rbf": True}


def resolve_backend(name: str = "auto") -> str:
    avail = backends_available()
    if name == "auto":
        return "gp" if avail["gp"] else "rbf"
    if name not in avail:
        raise BackendError(f"Unknown surrogate backend '{name}' (auto | gp | rbf)")
    if not avail[name]:
        raise BackendError("The Gaussian-process backend needs scikit-learn (optional module 'surrogate'); use backend 'rbf' or install the module")
    return name


def nn_distances(Xq: np.ndarray, X: np.ndarray, skip_self: bool = False) -> np.ndarray:
    """Distance of every row of Xq to the nearest row of X (k-d tree). skip_self: Xq is X, ignore the zero distance of a point to itself."""
    from scipy.spatial import KDTree
    Xq, X = np.atleast_2d(Xq), np.atleast_2d(X)
    tree = KDTree(X)
    if skip_self:
        if len(X) < 2:
            return np.ones(len(Xq))
        return tree.query(Xq, k=2)[0][:, 1]
    return tree.query(Xq, k=1)[0]


def _inside(v: Any, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo * (1 - 1e-6) <= v <= hi * (1 + 1e-6)


def check_params(name: str, raw: Any, d: int) -> dict:
    """The hyperparameters of a loaded file: right keys, right types, finite and inside the bounds of the optimiser. BackendError otherwise."""
    if not isinstance(raw, dict):
        raise BackendError("hyperparameters are not a dictionary")
    params: dict[str, Any] = raw
    if name == "rbf":
        s, r = params.get("smoothing"), params.get("rmse")
        if not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s) or not 0.0 <= s <= RBF_SMOOTHING_MAX * (1 + 1e-6):
            raise BackendError("the smoothing is out of range")
        if not _inside(r, 1e-12, RBF_RMSE_MAX):
            raise BackendError("the interpolation error is out of range")
        return {"smoothing": float(s), "rmse": float(cast(float, r))}
    ls, amp, noise = params.get("length_scales"), params.get("amplitude"), params.get("noise")
    if not _inside(amp, *AMP_BOUNDS) or not _inside(noise, *NOISE_BOUNDS):
        raise BackendError("the amplitude or the noise is out of range")
    if not isinstance(ls, list) or len(ls) != d or not all(_inside(v, *LS_BOUNDS) for v in ls):
        raise BackendError("the length scales are out of range")
    return {"amplitude": float(cast(float, amp)), "length_scales": [float(v) for v in ls], "noise": float(cast(float, noise))}


# ---------------------------------------------------------------------------------------------------- closed-form cross-validation
# The leave-out errors of a linear smoother come out of ONE matrix inverse (Rippa 1999 for radial basis functions; the same identity for a Gaussian process):
# no refit per point. The formulas are valid in exact arithmetic; a system that is badly conditioned (or singular) is NOT answered here - the callers
# fall back to the loop of refits, which is the reference (`tests/test_surrogate_closed_cv.py` compares the two).
RBF_CV_COND_MAX = 1e7                               # largest 1-norm condition number at which the closed form agrees with the loop of refits: ~5e-12 (relative) for the
GP_CV_COND_MAX = 1e6                                # radial basis system, ~1e-9 for the Gaussian process (measured with 0-1e9 on random and Latin hypercube designs)
GP_CV_VAR_SLACK = 1e-8                              # round-off allowed in the variance of a left-out point (relative to the amplitude)
RBF_CV_PIVOT_MIN = 1e-6                             # smallest singular value of A_BB (the left-out block of the inverse) relative to max|diag A|: a refit that leaves a
                                                    # (nearly) singular system has A_BB ~ 0 and the closed form would invent a finite error where the loop of refits fails or
                                                    # returns the (equally ill-conditioned) answer. The error of the closed form is ~2e-16 / ratio (measured on collinear-minus-one
                                                    # designs); regular systems (5 473 measured) stay above 6e-6, singular ones (480) below 4e-14


def _tps_matrix(X: np.ndarray, smoothing: float) -> np.ndarray | None:
    """The saddle-point matrix of scipy.interpolate.RBFInterpolator (thin plate spline, linear tail, smoothing on the diagonal):
    [[K + s I, P], [P^T, 0]] with K_ij = r^2 log r and P = [1, x] (the polynomial part in the shifted, scaled coordinates as scipy does). None for a degenerate set."""
    from scipy.spatial.distance import cdist
    n, d = X.shape
    lo, hi = X.min(axis=0), X.max(axis=0)
    scale = 0.5 * (hi - lo)
    if n < d + 1 or not np.all(scale > 0.0):
        return None
    d2 = cdist(X, X, "sqeuclidean")
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(d2 > 0.0, 0.5 * d2 * np.log(d2), 0.0)                   # r^2 log r, 0 at r = 0
    K[np.diag_indices(n)] += float(smoothing)
    m = n + d + 1
    M = np.zeros((m, m))
    M[:n, :n] = K
    M[:n, n] = 1.0
    M[:n, n + 1:] = (X - 0.5 * (lo + hi)) / scale
    M[n:, :n] = M[:n, n:].T
    return M


def rbf_cv_errors(X: np.ndarray, y: np.ndarray, smoothing: float, folds: np.ndarray) -> np.ndarray | None:
    """prediction - data of every point when its fold is left out and the thin-plate-spline interpolation (linear tail, `smoothing`) is refitted to the rest - from one
    inverse of the system matrix M. With A = (M^-1)[:n, :n] and c = A y (the coefficients of the full fit), the points B of one fold have the error  -(A_BB)^-1 c_B
    (a single point: -c_i / A_ii). Proof: put the left-out predictions in place of the data of B; the full system is then solved by the refit with the coefficients of B zero,
    so c' = c + A[:, B] (yhat_B - y_B) must vanish on B. The smoothing is a diagonal term of every point, so the argument holds with it. None when the matrix is singular or
    badly conditioned (the loop of refits decides then)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    folds = np.asarray(folds)
    if X.ndim != 2 or len(folds) != len(y) or len(X) != len(y):
        return None
    n, d = X.shape
    groups = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    # a refit needs a fold left in and at least d + 1 points for the linear tail (the same ValueError the loop of refits raises); the loop decides these cases
    if len(groups) < 2 or any(n - len(B) < d + 1 for B in groups):
        return None
    M = _tps_matrix(X, smoothing)
    if M is None or not np.all(np.isfinite(M)):
        return None
    from scipy.linalg import LinAlgError, inv       # scipy's LU inverse: numpy's `inv` (a solve with the identity) is far slower on a busy machine for matrices this small
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                                            # an ill-conditioned matrix is answered by the condition number below
            Minv = inv(M, check_finite=False)
    except (LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(Minv)) or not np.linalg.norm(M, 1) * np.linalg.norm(Minv, 1) <= RBF_CV_COND_MAX:
        return None
    A = Minv[:n, :n]
    c = A @ y
    dmax = float(np.max(np.abs(np.diag(A))))
    if not dmax > 0.0:
        return None
    err = np.empty(n)
    with np.errstate(all="ignore"):
        for B in groups:
            AB = A[np.ix_(B, B)]
            # leaving B out makes the refit singular (all points but one on a line, a simplex, a degenerate fold) exactly when A_BB is singular: not answered here
            if not np.min(np.linalg.svd(AB, compute_uv=False)) > RBF_CV_PIVOT_MIN * dmax:
                return None
            err[B] = -c[B] / AB[0, 0] if len(B) == 1 else -np.linalg.solve(AB, c[B])
    return err if np.all(np.isfinite(err)) else None


class RBFBackend:
    """Thin-plate-spline interpolation (scipy.interpolate.RBFInterpolator, linear tail) with the smoothing chosen by leave-one-out
    cross-validation. No native error bar: σ is the constant leave-one-out RMSE (homoscedastic) and the conformal factor makes it a
    band. A distance-dependent σ would be wrong for a space-filling design: a held-out point of a Latin hypercube sits farther from
    the remaining points than a random query does, so its larger σ would under-cover the queries (measured: 88 % instead of 90 %)."""
    name = "rbf"
    SMOOTHING = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    LOO_MAX = 120                                       # above this the smoothing is chosen by 10-fold cross-validation

    def __init__(self):
        self.smoothing = 0.0
        self.rmse = 1.0
        self.h = 1.0
        self._interp = None
        self._X = np.zeros((0, 0))

    @staticmethod
    def _interpolator(X, y, smoothing):
        from scipy.interpolate import RBFInterpolator
        return RBFInterpolator(X, y, kernel="thin_plate_spline", degree=1, smoothing=float(smoothing))

    def _folds(self, n: int) -> np.ndarray:
        return np.arange(n) if n <= self.LOO_MAX else np.arange(n) % 10

    def _cv_rmse_closed(self, X: np.ndarray, y: np.ndarray, smoothing: float) -> float | None:
        """`_cv_rmse` from one matrix inverse (rbf_cv_errors); None where that cannot be trusted."""
        err = rbf_cv_errors(X, y, smoothing, self._folds(len(y)))
        return None if err is None else float(np.sqrt(np.mean(err ** 2)))

    def _cv_rmse(self, X: np.ndarray, y: np.ndarray, smoothing: float) -> float:
        """The leave-one-out (up to LOO_MAX points, else 10-fold) RMSE of the interpolation: the loop of refits - the reference of the closed form."""
        n = len(y)
        folds = self._folds(n)
        err = np.empty(n)
        for f in np.unique(folds):
            te = folds == f
            try:
                err[te] = self._interpolator(X[~te], y[~te], smoothing)(X[te]) - y[te]
            except (np.linalg.LinAlgError, ValueError):
                return float("inf")
        return float(np.sqrt(np.mean(err ** 2)))

    def _cv_scores(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """The cross-validation RMSE of every smoothing. They come from the closed form; a smoothing for which that cannot be trusted, and every smoothing that could win
        against another within the accuracy of the closed form (a tie - the choice must be the one of the loop), is computed by the loop of refits."""
        fast = [self._cv_rmse_closed(X, y, s) for s in self.SMOOTHING]
        scores = [float("nan") if v is None else v for v in fast]
        for i, v in enumerate(fast):
            if v is None:
                scores[i] = self._cv_rmse(X, y, self.SMOOTHING[i])
        finite = [v for v in scores if math.isfinite(v)]
        if finite:                                                                   # the winner counts the loop-computed scores too
            thr = min(finite) * (1.0 + 1e-8) + 1e-10                                   # the closed form is accurate to ~1e-11 (relative)
            near = [i for i, v in enumerate(scores) if math.isfinite(v) and v <= thr]
            if len(near) > 1:                                                        # a clear winner is not settled again
                for i in near:
                    if fast[i] is not None:
                        scores[i] = self._cv_rmse(X, y, self.SMOOTHING[i])
        return scores

    def cv_predict(self, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(mean, sigma) of every point when its fold is left out and the model is refitted to the rest with THESE hyperparameters (what `restore_backend` + `predict` give,
        without the refits); None when the closed form cannot be trusted."""
        err = rbf_cv_errors(X, y, self.smoothing, folds)
        return None if err is None else (np.asarray(y, dtype=float) + err, np.full(len(err), self.rmse))

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0, params: dict | None = None) -> "RBFBackend":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if params is not None:                           # fixed hyperparameters: no tuning
            p = check_params("rbf", params, X.shape[1])
            self.smoothing, self.rmse = p["smoothing"], p["rmse"]
        else:
            scores = self._cv_scores(X, y)
            if not np.any(np.isfinite(scores)):
                raise BackendError("The radial-basis interpolation is singular for these points (duplicate parameter values?)")
            k = int(np.argmin(scores))
            self.smoothing, self.rmse = float(self.SMOOTHING[k]), max(float(scores[k]), 1e-6)
        try:
            self._interp = self._interpolator(X, y, self.smoothing)
        except (np.linalg.LinAlgError, ValueError) as e:
            raise BackendError(f"The radial-basis interpolation failed: {e}") from None
        self._X = X
        nn = nn_distances(X, X, skip_self=True) if len(X) > 1 else np.ones(1)
        self.h = max(float(np.median(nn)), 1e-9)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        assert self._interp is not None
        mu = np.asarray(self._interp(X), dtype=float)
        return mu, np.full(len(X), self.rmse)

    def spread(self, X: np.ndarray) -> np.ndarray:
        """Where is the model least informed (for choosing the next analysis): the distance to the nearest training point in units of
        the median spacing."""
        return nn_distances(np.atleast_2d(np.asarray(X, dtype=float)), self._X) / self.h

    def params(self) -> dict:
        return {"smoothing": self.smoothing, "rmse": self.rmse}


def _matern52(A: np.ndarray, B: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """Matérn ν = 5/2 covariance (unit amplitude) between the rows of A and B with one length scale per dimension."""
    from scipy.spatial.distance import cdist
    r = cdist(A / ls, B / ls)
    s = math.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


class GPBackend:
    """Gaussian process regression: amplitude · anisotropic Matérn 5/2 + white noise (mesh noise) on standardised inputs / outputs;
    hyperparameters by maximum likelihood (scikit-learn, fixed seed, deterministic), prediction in numpy (the same formulas as
    scikit-learn). σ = predictive standard deviation including the noise."""
    name = "gp"
    RESTARTS = 4

    def __init__(self):
        self._p: dict = {}
        self._X = np.zeros((0, 0))
        self._L = np.zeros((0, 0))
        self._w = np.zeros(0)

    @staticmethod
    def _optimise(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
        from sklearn.gaussian_process import GaussianProcessRegressor  # pyright: ignore[reportMissingImports]  # optional module surrogate
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # pyright: ignore[reportMissingImports]  # optional module surrogate
        d = X.shape[1]
        k = (ConstantKernel(1.0, AMP_BOUNDS) * Matern(length_scale=np.ones(d), length_scale_bounds=LS_BOUNDS, nu=2.5)
             + WhiteKernel(1e-4, NOISE_BOUNDS))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # ConvergenceWarning at a bound: the hyperparameters are still usable
            try:
                gp = GaussianProcessRegressor(kernel=k, alpha=JITTER, normalize_y=False, n_restarts_optimizer=GPBackend.RESTARTS,
                                              random_state=int(seed)).fit(X, y)
            except np.linalg.LinAlgError as e:
                raise BackendError(f"The Gaussian process failed: {e}") from None
        kf = gp.kernel_
        p = {"amplitude": float(kf.k1.k1.constant_value), "length_scales": [float(v) for v in np.atleast_1d(kf.k1.k2.length_scale)],
             "noise": float(kf.k2.noise_level)}
        del gp
        return p

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0, params: dict | None = None) -> "GPBackend":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        p = check_params("gp", params, X.shape[1]) if params is not None else check_params("gp", self._optimise(X, y, seed), X.shape[1])
        amp, ls, noise = p["amplitude"], np.asarray(p["length_scales"]), p["noise"]
        K = amp * _matern52(X, X, ls)
        K[np.diag_indices_from(K)] += noise + JITTER
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as e:
            raise BackendError(f"The Gaussian process failed: {e}") from None
        from scipy.linalg import cho_solve
        self._p, self._X, self._L = p, X, L
        self._w = cho_solve((L, True), y)
        return self

    def cv_predict(self, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(mean, sigma) of every point when its fold is left out and the process is refitted to the rest with THESE hyperparameters, from one inverse of the covariance
        matrix K: with Q = K^-1 and alpha = Q y, a fold B has the mean y_B - (Q_BB)^-1 alpha_B and the variance diag((Q_BB)^-1) - JITTER (the jitter is in K, not in the
        prediction of a new point). None when K is badly conditioned (the loop of refits decides then)."""
        from scipy.linalg import cho_solve
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        amp, ls, noise = self._p["amplitude"], np.asarray(self._p["length_scales"]), self._p["noise"]
        folds = np.asarray(folds)
        if len(X) != len(y) or len(folds) != len(y) or len(np.unique(folds)) < 2:           # a refit needs something left in: the loop decides
            return None
        K = amp * _matern52(X, X, ls)
        K[np.diag_indices_from(K)] += noise + JITTER
        n = len(y)
        try:
            L = np.linalg.cholesky(K)
            Q = cho_solve((L, True), np.eye(n))
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(Q)) or not np.linalg.norm(K, 1) * np.linalg.norm(Q, 1) <= GP_CV_COND_MAX:
            return None
        alpha = Q @ y
        mu, var = np.empty(n), np.empty(n)
        for f in np.unique(folds):
            B = np.flatnonzero(folds == f)
            if len(B) == 1:
                mu[B] = y[B] - alpha[B] / Q[B, B]
                var[B] = 1.0 / Q[B, B] - JITTER
            else:
                QB = Q[np.ix_(B, B)]
                try:
                    mu[B] = y[B] - np.linalg.solve(QB, alpha[B])
                    var[B] = np.diag(np.linalg.inv(QB)) - JITTER
                except np.linalg.LinAlgError:
                    return None
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(var))) or np.any(var < -GP_CV_VAR_SLACK * max(amp, 1.0)):
            return None                                                                  # the variance of a left-out point is >= 0 in exact arithmetic: round-off beyond the slack = not trusted
        return mu, np.maximum(np.sqrt(np.maximum(var, 0.0)), 1e-9)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.linalg import solve_triangular
        X = np.atleast_2d(np.asarray(X, dtype=float))
        amp, ls, noise = self._p["amplitude"], np.asarray(self._p["length_scales"]), self._p["noise"]
        Ks = amp * _matern52(X, self._X, ls)
        mu = Ks @ self._w
        V = solve_triangular(self._L, Ks.T, lower=True)
        var = np.maximum(amp + noise - np.sum(V * V, axis=0), 0.0)
        return np.asarray(mu, dtype=float), np.maximum(np.sqrt(var), 1e-9)

    def spread(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)[1]

    def params(self) -> dict:
        return dict(self._p)


def make_backend(name: str):
    """A new backend object: "gp" (the fit needs scikit-learn) or "rbf"."""
    name = resolve_backend(name)
    return GPBackend() if name == "gp" else RBFBackend()


def restore_backend(name: str, X: np.ndarray, y: np.ndarray, params: dict):
    """A backend rebuilt from saved hyperparameters (no optimisation; a Gaussian process needs no scikit-learn for that)."""
    if name not in ("gp", "rbf"):
        raise BackendError(f"Unknown surrogate backend '{name}'")
    return (GPBackend() if name == "gp" else RBFBackend()).fit(X, y, params=params)

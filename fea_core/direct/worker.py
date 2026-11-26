"""Direct solver process: python -m fea_core.direct.worker JOB → JOB/x.npy (n × states), JOB/direct_info.json.

Input: JOB/indptr.npy (int64), indices.npy (int32), data.npy (float64) = the symmetric positive
definite matrix (constraints eliminated symmetrically, the diagonal of a constrained DOF = d), rhs.npy (n × states); JOB/matrix.json
{"format": "upper" | "full"} (missing = "full"); optional JOB/direct_options.json {"engine", "threads", "precision", "reserve_gb"}.

One analysis and ONE factorization for all right-hand sides, then one solve. Precision of the factor (engine = rslab):
  float64   when the memory plan fits and the factor is not very large per DOF,
  float32   when the factor has more than FILL_FLOAT32 entries per DOF (compact bodies – 0.42–0.50× of the float64 time) or float64 does
            not fit; refined in float64 by a block conjugate-gradient loop preconditioned with the factor (3–18 iterations),
  nothing   when even float32 does not fit → exit code 3, the caller continues with the iterative solver.
Acceptance by the backward error ||r|| / (||K||_1 ||x|| + ||b||) (the relative residual ||r|| / ||b|| of thin parts is 5e-8 … 3e-7 with
every solver and of an ill-conditioned system up to 1e-3 and cannot be used as the criterion); the relative residual is checked only
against a very loose limit (RESIDUAL_GUARD), as a sanity check of a nearly singular system.

A float32 factor that fails (a negative pivot of an ill-conditioned but positive definite matrix, a zero pivot, the refinement does not
converge) is repeated in float64 when its memory plan fits; when it does not fit the worker declines like for a missing memory (exit code 3,
the caller continues with the iterative solver). Only a float64 factor may report a free body: the float32 pivots are not a proof of it.

Exit codes: 0 ok, 3 does not fit into memory (nothing was factored, or the float32 factor failed and float64 does not fit), 5 numerical
failure of the float64 factor (zero pivot, not positive definite = a free body, a missing constraint), 6 accuracy not reached.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

GB = 2**30
FILL_FLOAT32 = 800            # factor entries per DOF above which the float32 factor is faster (measured 17. 11. 2026)
MEM_MARGIN = 0.88             # the plan (+ the arrays of this process) may use this part of the free RAM
MEM_EXTRA_GB = 0.25           # copies of the matrix made while factoring (not part of the engine's plan)
BACKWARD_TOL = 1e-12          # acceptance: backward error of every right-hand side
RESIDUAL_GUARD = 1e-2         # ||r|| / ||b|| above this = nearly singular system (sanity only: a float64 solution of a system with the
#                               condition number 1e12 – backward error 6e-17 – has 1e-4 … 1e-3, the thin sheet 3e-7)
CG_TOL = 1e-10                # refinement stops at ||r|| / ||b|| (the same as in the lab measurements)
CG_MAXITER = 100
EXIT_NOFIT, EXIT_NUMERIC, EXIT_ACCURACY = 3, 5, 6


class WorkerFail(Exception):
    def __init__(self, code: int, kind: str, message: str):
        super().__init__(message)
        self.code, self.kind, self.message = code, kind, message


class NotFactored(Exception):
    """The factorization failed or the factor is not positive definite: `str(e)` = what happened, `e.final` = the message of a failure of
    the float64 factor. In float32 it is not yet a proof of a singular matrix – `solve_system` repeats it in float64; a float64 failure
    becomes WorkerFail(EXIT_NUMERIC, e.final)."""

    def __init__(self, detail: str, final: str):
        super().__init__(detail)
        self.final = final


def peak_gb() -> float:
    """Peak working set of this process [GB]."""
    import psutil
    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset / GB
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GB


THREADED_MATMAT_COLS = 8                 # fewer columns: the thread hand-over costs more than it saves (4–5 columns were slower, +50 % transient memory)


class SymmetricOperator:
    """y = K·x for the symmetric K stored as its upper triangle (CSR, with the diagonal). Products with ≥ 8 columns run in column chunks on `threads`
    threads (the sparse kernels of SciPy release the GIL; the columns are independent, so the result is the same to the last bit)."""

    def __init__(self, U: sp.csr_matrix, threads: int = 1):
        self.U = U
        self.n = U.shape[0]
        self.diag = U.diagonal()
        self.UT = U.T                                   # a CSC view of the same arrays, no copy
        self.threads = max(1, int(threads))

    def _block(self, X: np.ndarray) -> np.ndarray:
        return self.U @ X + self.UT @ X - self.diag[:, None] * X

    def matmat(self, X: np.ndarray) -> np.ndarray:
        k = X.shape[1] if X.ndim == 2 else 0
        if self.threads < 2 or k < THREADED_MATMAT_COLS:
            return self._block(X)
        from concurrent.futures import ThreadPoolExecutor
        out = np.empty(X.shape, dtype=np.float64)
        parts = np.array_split(np.arange(k), min(self.threads, k // 2))

        def one(cols):
            out[:, cols] = self._block(np.ascontiguousarray(X[:, cols], dtype=np.float64))
        with ThreadPoolExecutor(len(parts)) as ex:
            list(ex.map(one, parts))
        return out

    def norm1(self, block: int = 200_000) -> float:
        """||K||_1 = ||K||_inf (symmetric): the largest absolute row sum of the whole matrix, in blocks of rows (no full |K| copy)."""
        n = self.n
        rows = np.zeros(n)
        cols = np.zeros(n)
        U = self.U
        for lo in range(0, n, block):
            sub = U[lo:lo + block]
            a = abs(sub)
            rows[lo:lo + block] += np.asarray(a.sum(axis=1)).ravel()
            cols += np.bincount(sub.indices, weights=np.abs(sub.data), minlength=n)
        return float(np.max(rows + cols - np.abs(self.diag)))


def load_upper(job: Path, fmt: str) -> sp.csr_matrix:
    indptr = np.load(job / "indptr.npy")
    indices = np.load(job / "indices.npy", mmap_mode="r")
    data = np.load(job / "data.npy", mmap_mode="r")
    n = len(indptr) - 1
    if fmt == "upper":
        return sp.csr_matrix((np.asarray(data), np.asarray(indices), indptr), shape=(n, n), copy=False)
    K = sp.csr_matrix((np.asarray(data), np.asarray(indices), indptr), shape=(n, n))
    U = sp.triu(K, format="csr")
    U.sort_indices()
    return U


def block_pcg(op: SymmetricOperator, B: np.ndarray, prec, tol: float, maxiter: int):
    """Conjugate gradients for all columns of B at once, preconditioned by prec(R) (one call of the factor for all active columns).
    Stops per column at ||r|| / ||b|| < tol (the recurrence residual). Returns (X, iterations per column, converged per column).
    """
    X = np.zeros_like(B)
    R = B.copy()
    nb = np.linalg.norm(B, axis=0)
    active = nb > 0                                    # a zero right-hand side (a load state without load) has the solution 0
    nb[nb == 0] = 1.0
    its = np.zeros(B.shape[1], int)
    if not active.any():
        return X, its.tolist(), [True] * B.shape[1]
    Z = prec(R)
    P = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    for it in range(1, maxiter + 1):
        a_ = np.flatnonzero(active)
        Q = op.matmat(P[:, a_])
        alpha = rz[a_] / np.einsum("ij,ij->j", P[:, a_], Q)
        X[:, a_] += alpha * P[:, a_]
        R[:, a_] -= alpha * Q
        done = np.linalg.norm(R[:, a_], axis=0) / nb[a_] < tol
        its[a_] = it
        active[a_[done]] = False
        if not active.any():
            break
        a_ = np.flatnonzero(active)
        Za = prec(R[:, a_])
        rz_new = np.einsum("ij,ij->j", R[:, a_], Za)
        P[:, a_] = Za + (rz_new / rz[a_]) * P[:, a_]
        rz[a_] = rz_new
    return X, its.tolist(), (~active).tolist()


def accuracy(op: SymmetricOperator, X: np.ndarray, B: np.ndarray, knorm: float, AX: np.ndarray | None = None):
    """(backward error, relative residual) per right-hand side from the TRUE residual B − K·X (AX = K·X when it is already known)."""
    nr = np.linalg.norm((op.matmat(X) if AX is None else AX) - B, axis=0)
    nb = np.linalg.norm(B, axis=0)
    nx = np.linalg.norm(X, axis=0)
    den = knorm * nx + nb
    bwd = np.where(den > 0, nr / np.where(den > 0, den, 1.0), nr)     # b = 0 and x = 0 (a load state without load): the residual itself (0)
    rel = nr / np.where(nb > 0, nb, 1.0)
    return bwd, rel


def choose_precision(requested: str, fill: float, plans: dict[str, float], avail_gb: float, reserve_gb: float):
    """(precision, reason) or raises WorkerFail(EXIT_NOFIT). plans = {"float64": GB, "float32": GB} of the engine."""
    budget = MEM_MARGIN * avail_gb - reserve_gb
    fits = {p: v + MEM_EXTRA_GB <= budget for p, v in plans.items()}
    if requested in plans:
        if not fits[requested]:
            raise WorkerFail(EXIT_NOFIT, "memory", f"{requested} factorization needs about {plans[requested]:.1f} GB, "
                                                   f"{max(budget, 0):.1f} GB are usable")
        return requested, f"requested {requested}"
    if fits["float64"] and fill <= FILL_FLOAT32:
        return "float64", f"float64 fits ({plans['float64']:.1f} GB of {budget:.1f} GB), factor {fill:.0f} entries per DOF"
    if fits["float32"]:
        why = (f"factor {fill:.0f} entries per DOF > {FILL_FLOAT32} (float32 is faster)" if fits["float64"] else
               f"float64 needs {plans['float64']:.1f} GB, {max(budget, 0):.1f} GB are usable")
        return "float32", why
    raise WorkerFail(EXIT_NOFIT, "memory", f"the direct solver needs at least {plans['float32']:.1f} GB (float32 factor), "
                                           f"{max(budget, 0):.1f} GB are usable")


def solve_system(U: sp.csr_matrix, B: np.ndarray, opts: dict, info: dict, log=print) -> np.ndarray:
    """Solves K·X = B for the symmetric K given by its upper triangle; returns X (n, k) float64. `info` is filled in as it goes
    (it stays useful when WorkerFail is raised).
    """
    import psutil
    from .engines import EngineError, make
    t0 = time.time()
    n = U.shape[0]
    threads = int(opts.get("threads") or 0) or 1
    op = SymmetricOperator(U, threads)
    knorm = op.norm1()
    eng = make(str(opts.get("engine") or "rslab"), n, U.indptr, U.indices, U.data, threads)
    info.update({"engine": eng.name, "version": eng.version, "n": n, "nnz": int(U.nnz), "threads": threads,
                 "knorm": knorm, "cases": B.shape[1]})
    rss_before = psutil.Process().memory_info().rss / GB             # what this process holds before the engine allocates (the matrix, the right-hand sides)
    info["rss_before_gb"] = rss_before
    t = time.time()
    eng.analyze()
    info["t_analyze"] = time.time() - t
    info["factor_nnz"] = eng.factor_nnz
    fill = eng.factor_nnz / n
    info["fill_per_dof"] = fill
    plans = {p: eng.plan_gb(p, B.shape[1]) for p in ("float64", "float32")}
    avail = psutil.virtual_memory().available / GB
    info["plan_gb"] = {k: round(v, 3) for k, v in plans.items()}
    info["free_gb"] = round(avail, 3)
    reserve = float(opts.get("reserve_gb", 1.0))
    precision, why = choose_precision(str(opts.get("precision") or "auto"), fill, plans, avail, reserve)
    info["precision"], info["precision_reason"] = precision, why
    log(f"analysis {info['t_analyze']:.1f} s: factor {eng.factor_nnz / 1e6:.0f} M entries ({fill:.0f} per DOF); "
        f"{precision} – {why}")

    def factor(prec: str) -> None:
        t = time.time()
        try:
            eng.factor(prec)
        except EngineError as e:
            info["t_factor"] = info.get("t_factor", 0.0) + time.time() - t
            raise NotFactored(str(e), f"{e} – the matrix is singular (a free body, missing constraint?)") from None
        info["t_factor"] = info.get("t_factor", 0.0) + time.time() - t
        npos, nneg, nzero = eng.inertia
        info["inertia"] = [npos, nneg, nzero]
        info["n_perturbed"] = eng.n_perturbed
        if nneg or nzero or eng.n_perturbed:
            detail = f"negative / zero pivots {nneg} / {nzero}, perturbed {eng.n_perturbed}"
            raise NotFactored(detail, f"the matrix is not positive definite ({detail}) – a free body or an unconstrained motion?")

    def run(prec: str):
        factor(prec)
        t = time.time()
        if prec == "float64":
            X = eng.solve(B)
            info["t_solve"] = time.time() - t
            its = [0] * B.shape[1]
            checked = None                           # (bwd, rel) of the returned X when its last check was not followed by a refinement
            for _ in range(2):                       # a float64 factor is backward stable; at most two refinement steps for safety
                AX = op.matmat(X)                    # K·X once: the check and the refinement residual use the same product
                bwd, rel = accuracy(op, X, B, knorm, AX)
                if np.all(bwd <= BACKWARD_TOL):
                    checked = (bwd, rel)
                    break
                X = X + eng.solve(B - AX)
                its = [i + 1 for i in its]
            return X, its, checked
        X, its, ok = block_pcg(op, B, eng.solve, CG_TOL, CG_MAXITER)
        info["t_solve"] = time.time() - t
        if not all(ok):
            info["not_converged"] = [i for i, o in enumerate(ok) if not o]
        return X, its, None

    def is_good(bwd, rel) -> bool:
        return bool(np.all(np.isfinite(bwd)) and np.all(bwd <= BACKWARD_TOL) and np.all(rel <= RESIDUAL_GUARD)
                    and not info.get("not_converged"))

    why_failed = ""                                 # why the float32 factor / refinement was not accepted ("" = it was)
    try:
        X, its, checked = run(precision)
        bwd, rel = checked if checked is not None else accuracy(op, X, B, knorm)
        good = is_good(bwd, rel)
        if not good:
            why_failed = (f"the refinement did not reach the accuracy (backward error {float(np.nanmax(bwd)):.1e})"
                          if precision == "float32" else "")
    except NotFactored as e:
        if precision == "float64":                  # only an exact (float64) factor is a proof of a singular system
            raise WorkerFail(EXIT_NUMERIC, "numeric", e.final) from None
        X = its = bwd = rel = None
        good = False
        why_failed = f"the factorization failed: {e}"
        info["float32_failure"] = str(e)
    if not good and precision == "float32":
        # the float32 factor failed (a negative pivot of an ill-conditioned matrix, a zero pivot) or the refinement did not reach the accuracy:
        # repeat with a float64 factor when it fits (the float32 factor is freed first, so the free memory is measured after that)
        eng.release()
        budget = MEM_MARGIN * psutil.virtual_memory().available / GB - reserve
        if plans["float64"] + MEM_EXTRA_GB > budget:
            raise WorkerFail(EXIT_NOFIT, "memory",
                             f"the float32 factor was not accepted ({why_failed}) and the float64 factor needs {plans['float64']:.1f} GB, "
                             f"{max(budget, 0):.1f} GB are usable")
        log(f"float32 factor not accepted ({why_failed}) – float64 factor")
        eng = make(str(opts.get("engine") or "rslab"), n, U.indptr, U.indices, U.data, threads)
        eng.analyze()
        info["precision"], info["precision_reason"] = "float64", f"float32 factor not accepted ({why_failed})"
        info.pop("not_converged", None)
        precision = "float64"
        try:
            X, its, checked = run(precision)
        except NotFactored as e:
            raise WorkerFail(EXIT_NUMERIC, "numeric", e.final) from None
        bwd, rel = checked if checked is not None else accuracy(op, X, B, knorm)
        good = is_good(bwd, rel)
    info["iterations"] = its
    info["backward_error"] = [float(v) for v in bwd]
    info["rel_residual"] = [float(v) for v in rel]
    if not good:
        raise WorkerFail(EXIT_ACCURACY, "accuracy",
                         f"the solution did not reach the accuracy: backward error {float(np.nanmax(bwd)):.1e}, residual "
                         f"{float(np.nanmax(rel)):.1e}" + (f", the refinement did not converge in {CG_MAXITER} iterations"
                                                            if info.get("not_converged") else "")
                         + " – the system is ill-conditioned (a free body?)")
    info["t_total_solver"] = time.time() - t0
    info["growth_gb"] = max(peak_gb() - rss_before, 0.0)           # measured growth of the process = what the memory plan of the engine predicts
    eng.release()
    return X


def main(job: Path) -> int:
    t0 = time.time()
    (job / "x.npy").unlink(missing_ok=True)
    (job / "direct_info.json").unlink(missing_ok=True)
    opts = {}
    if (job / "direct_options.json").exists():
        opts = json.loads((job / "direct_options.json").read_text(encoding="utf-8"))
    fmt = "full"
    if (job / "matrix.json").exists():
        fmt = json.loads((job / "matrix.json").read_text(encoding="utf-8")).get("format", "full")
    info: dict = {}
    try:
        U = load_upper(job, fmt)
        B = np.load(job / "rhs.npy").reshape(U.shape[0], -1)
        t_load = time.time() - t0
        X = solve_system(U, B, opts, info, log=lambda s: print(s, flush=True))
        info["t_load"] = t_load
        info["format"] = fmt
        info["peak_gb"] = peak_gb()
        np.save(job / "x.npy", np.asfortranarray(X))
        code = 0
    except WorkerFail as e:
        info["error"] = {"kind": e.kind, "message": e.message}
        info["peak_gb"] = peak_gb()
        print(f"direct solver: {e.message}", flush=True)
        code = e.code
    info["t_total"] = time.time() - t0
    (job / "direct_info.json").write_text(json.dumps(info), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

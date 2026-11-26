"""Natural frequencies and the modal basis on the direct path: K·φ = λ·M·φ by the shift-invert Lanczos method (ARPACK of SciPy, BSD) over ONE factor of
K + s·M (rslab, shifted below the spectrum, so the matrix is symmetric positive definite also for a free body). Glue only – the eigenvalue iteration is the library's,
the factorization is the direct solver's (`engines`); this module chooses the shift, builds the operators, and PROVES that no mode is missing:

  * the number of eigenvalues below a bound is the number of negative pivots of the LDLᵀ factor of K − σM (Sylvester's law of inertia: the Sturm count). After the
    iteration the count at a bound just above the largest mode found is compared with the number of modes found; a difference = a missed mode (a repeated or
    clustered eigenvalue – a Lanczos method started from one vector finds one vector per eigenspace in exact arithmetic) → the iteration is repeated with the found
    vectors deflated (the operator is projected M-orthogonally against them). A count that does not come out is an error, never a silently missing mode.
  * a bound of a window (or of the count) that cannot be used as it is moves OUTWARD by a tiny fraction of itself and the count PROVES that no eigenvalue lies between the requested
    and the moved bound (`Shifted.count_below`); the lower bound of a window of a free body starts above the rounding noise of its rigid-body modes;
  * the residual ‖Kφ − λMφ‖ of every pair and the M-orthonormality of the whole set are checked – the report (`info["residual"]`) says which criterion each pair met: the relative
    residual, or the backward error (the rounding of K·φ itself);
  * a failed check repeats the analysis with something that matters (`attempt_plan`: the shift, the driving shift of a window, the Lanczos basis);
  * the start vector is fixed (a seeded generator) – the same input gives the same modes.

Convention: the eigenvalues are λ = ω² [1/s²] in the units of the mesh (N, mm, t); a free body has its rigid-body modes at λ ≈ 0 (rounding noise ± 1e-11·λ_max): they
are returned as modes (the caller clamps λ to 0) with a real shape (ccx leaves the ones with a negative eigenvalue out; see `fea_core.direct.modal_run`).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

GB = 2**30
SHIFT_REL = 1e-9             # s = SHIFT_REL · λ_ref, λ_ref = min over the DOFs that carry mass of K_ii / M_ii (the Rayleigh quotient of a unit vector, so λ₁ ≤ λ_ref). The
#                              rigid-body modes of a free body sit at ± 1e-15·λ_typ of rounding noise, far below s; the first elastic mode of a slender model is at
#                              1e-7 … 1e-4·λ_ref – the shift lies between (a shift that is too large is not wrong, only slower: the Lanczos iteration and the static
#                              refinement contract by s/(λ₁ + s)). The factor must be positive definite – checked, the shift grows by 1e3 until it is.
SHIFT_REL_FREE = 1e-6        # a free body: the factor of K + sM has a condition number λ_max/s and the rounding of the solves pollutes the elastic modes with the rigid
#                              ones (residual of the modes ∝ 1/s: 5e-8 at 1e-9, 7e-9 at 1e-8, 8e-10 at 1e-7, 5e-11 at 1e-6 on a free beam) – a larger shift there
PROBE_TOL = float(os.environ.get("FEA_EIGEN_PROBE_TOL", "1e-6"))             # the floor of the probe limit, see `Shifted.probe_limit`
PROBE_FIXED = "FEA_EIGEN_PROBE_TOL" in os.environ                             # the limit is fixed by the user (tests, diagnostics): no size dependence
PROBE_SAT = 0.05             # a K − σM that is singular to working precision leaves a relative residual of |g|/√n in a solve (g ~ N(0,1): the component of the random right-hand side
#                              along the null vector – the solve returns it amplified by 1/ε and does not reproduce it); a regular one ε·λ_max/(distance of σ to the nearest eigenvalue)
#                              (measured on a 400×6×2 mm strip, n = 40 521: 3e-8 / relative distance, 4e-3 on the eigenvalue, 2.6e-6 at 1 % below it). The limit is PROBE_SAT/√n (three vectors:
#                              a singular σ is missed with a probability below 1e-4) – a fixed 1e-6 flagged every shift within 3 % of a low mode of a slender model as singular
MOVE_REL = 1e-9              # a bound that cannot be used as it is (a zero or perturbed pivot, a singular probe) moves outward by this fraction of |σ| first …
MOVE_GROW = 10.0             # … then 10 times more at every step …
MOVE_MAX = 1e-3              # … up to this fraction (the window is widened by at most that – an eigenvalue in the widened zone counts as inside, see `count_below`)
SHIFT_RETRY = 1e3            # an analysis whose residual / orthogonality check (or Sturm count) fails is repeated with the shift ×30, ÷30, ×1e3 (`SHIFT_STEP`), a window with its driving shift
#                              lifted by this factor per attempt, the Lanczos basis larger (see `attempt_plan`)
SHIFT_STEP = 30.0
SHIFT_TRIES = 4
RIGID_BAND = 2.0             # the lower bound of a window of a free body is at least RIGID_BAND × the strain-energy limit of a rigid-body mode (see `rigid_limit`)
RIGID_ENERGY = 10.0          # a mode is a rigid-body motion of a free body when its strain energy φᵀKφ ≤ RIGID_ENERGY · ε · ‖K‖₁ · ‖φ‖² (rounding noise of K·φ, measured
#                              ±0.03 against the limit 10 on a free beam) AND it lies 1e4 times below the first mode above that limit
STURM_REL = 1e-7             # the bound of the count lies this far (relative) above the largest mode found
STURM_RES = 64.0             # … and at least this many ε·λ_ref (the old floor 1e-9·λ_ref was 1.7e5 above the first mode of a slender strip: the bound spanned the next modes and cost a pass)
MAX_PASSES = 4               # the iteration with deflation is repeated at most this many times
RES_TOL = 1e-8               # ‖Kφ − λMφ‖ ≤ RES_TOL · max(‖Kφ‖, λ‖Mφ‖) (and a floor from the rounding of K)
ORTHO_TOL = 1e-8             # |φᵢᵀ M φⱼ − δᵢⱼ|
ARPACK_TOL = 1e-12
BACKWARD_FLOOR = 1e-12      # … or the backward error ‖Kφ − λMφ‖/(‖K‖₁‖φ‖) ≤ this (the rounding of K·φ itself, 1e4 × ε)
SEED = 20261118
RESIDUAL_BLOCK = 16                      # columns per block of the residual check (`_solve_modes`)
EXIT_NOFIT, EXIT_NUMERIC, EXIT_ACCURACY, EXIT_INCOMPLETE = 3, 5, 6, 7


class EigenFail(Exception):
    """The eigenvalue analysis cannot be completed on this path: `code` = exit code of the worker, `kind` = memory | numeric | accuracy | incomplete."""

    def __init__(self, code: int, kind: str, message: str, retry: bool | None = None):
        super().__init__(message)
        self.code, self.kind, self.message = code, kind, message
        self.retry = (kind == "accuracy") if retry is None else retry         # another shift / driving shift / basis may repair it (`solve_modes`)


@dataclass
class EigenOptions:
    modes: int = 10
    lam_min: float = 0.0                 # ω² [1/s²]: only the modes above it (0 = from the bottom)
    lam_max: float | None = None         # ω² [1/s²]: only the modes below it
    threads: int = 4
    engine: str = "rslab"
    reserve_gb: float = 1.0
    ncv: int = 0                         # Lanczos vectors (0 = by the number of modes)
    seed: int = SEED
    free: bool = False                   # the model has no constraint at all (a free body: rigid-body modes at 0) – the shift of SHIFT_REL_FREE
    statics_optional: bool = False       # the static solutions beside the modes are an extra (the unit accelerations the state cache keeps): where K cannot be solved (a body held in
                                         # one direction only is singular) they are skipped – `info["statics_skipped"]` says why – and the modes are returned as they are

    @classmethod
    def from_dict(cls, d: dict) -> "EigenOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EigenResult:
    lam: np.ndarray                      # (m,) ascending
    X: np.ndarray                        # (n, m) M-orthonormal, the sign: the largest component positive
    info: dict = field(default_factory=dict)
    statics: np.ndarray | None = None    # (n, states) K⁻¹·rhs from the same factor
    rigid: np.ndarray | None = None      # (m,) bool: the rigid-body motions of a free body (their strain energy is rounding noise)


THREADED_MATMAT_COLS = 8                 # fewer columns: the thread hand-over costs more than it saves (4–5 columns were slower, +50 % transient memory)


class Sym:
    """y = A·x for the symmetric A stored as its upper triangle (CSR with the diagonal); x: (n,) or (n, k). Products with several columns run in
    column chunks on `threads` threads (the sparse kernels of SciPy release the GIL)."""

    def __init__(self, U: sp.csr_matrix, threads: int = 1):
        self.U = U
        self.UT = U.T                                  # a CSC view of the same arrays, no copy
        self.n = U.shape[0]
        self.diag = U.diagonal()
        self.threads = max(1, int(threads))
        self._norm1: float | None = None

    def _block(self, X: np.ndarray) -> np.ndarray:
        return self.U @ X + self.UT @ X - self.diag[:, None] * X

    def mv(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return self.U @ X + self.UT @ X - self.diag * X
        k = X.shape[1]
        if self.threads < 2 or k < THREADED_MATMAT_COLS:
            return self._block(X)
        out = np.empty_like(X, dtype=np.float64)
        parts = np.array_split(np.arange(k), min(self.threads, k // 2))

        def one(cols):
            out[:, cols] = self._block(np.ascontiguousarray(X[:, cols], dtype=np.float64))
        with ThreadPoolExecutor(len(parts)) as ex:
            list(ex.map(one, parts))
        return out

    def norm1(self, block: int = 200_000) -> float:
        """‖A‖₁ (the largest absolute column sum), in blocks of rows (computed once)."""
        if self._norm1 is not None:
            return self._norm1
        rows = np.zeros(self.n)
        cols = np.zeros(self.n)
        U = self.U
        for lo in range(0, self.n, block):
            sub = U[lo:lo + block]
            rows[lo:lo + block] += np.asarray(abs(sub).sum(axis=1)).ravel()
            cols += np.bincount(sub.indices, weights=np.abs(sub.data), minlength=self.n)
        self._norm1 = float(np.max(rows + cols - np.abs(self.diag)))
        return self._norm1


def rigid_limit(Ksym: Sym, MU: sp.csr_matrix) -> float:
    """The eigenvalue λ [1/s²] below which a mode of a FREE body is numerically a rigid-body motion: the strain-energy limit of the classification (`RIGID_ENERGY`·ε·‖K‖₁·‖φ‖²,
    see the end of `_solve_modes`) with ‖φ‖² of a mass-normalized rigid translation, n/tr(M) (tr M ≈ 1.5·the mass of one translation direction for the consistent mass of tet10;
    the estimate is conservative, the rounding noise of the rigid modes measured on a 400×6×2 mm free strip is 1e-3 against a limit of 6). A window of a free body starts above it."""
    eps = float(np.finfo(np.float64).eps)
    tr = float(MU.diagonal().sum())
    if tr <= 0.0:
        return 0.0
    return RIGID_BAND * RIGID_ENERGY * eps * Ksym.norm1() * Ksym.n / tr


def reference_scale(KU: sp.csr_matrix, MU: sp.csr_matrix) -> float:
    """λ_ref = the smallest K_ii / M_ii over the DOFs that carry mass: the Rayleigh quotient of the unit vector e_i is K_ii/M_ii ≥ λ₁, so the first eigenvalue is
    not above it – the scale from which the shift is derived."""
    dk, dm = KU.diagonal(), MU.diagonal()
    ok = (dm > 0) & (dk > 0)
    if not ok.any():
        raise EigenFail(EXIT_NUMERIC, "numeric", "the model has no mass (check the densities)")
    return float(np.min(dk[ok] / dm[ok]))


class Shifted:
    """The factor of K − σM by the engine (one analysis for every σ: the pattern is the same). `count_below` = the Sturm count."""

    def __init__(self, KU, MU, opts: EigenOptions, info: dict, lam_ref: float):
        from .engines import make
        self.KU, self.MU, self.opts, self.info, self.lam_ref = KU, MU, opts, info, lam_ref
        self.n = KU.shape[0]
        self.eng = make(opts.engine, self.n, KU.indptr, KU.indices, KU.data, opts.threads)
        self.sigma: float | None = None
        t = time.time()
        self.eng.analyze()
        info["t_analyze"] = time.time() - t
        info["factor_nnz"] = self.eng.factor_nnz
        info["engine"], info["version"] = self.eng.name, self.eng.version
        self.plan_gb = self.eng.plan_gb("float64", 1)
        info["plan_gb"] = round(self.plan_gb, 3)

    def check_memory(self, extra_gb: float, avail_gb: float) -> None:
        need = self.plan_gb + extra_gb
        budget = 0.88 * avail_gb - self.opts.reserve_gb
        self.info.update({"need_gb": round(need, 3), "free_gb": round(avail_gb, 3)})
        if need > budget:
            raise EigenFail(EXIT_NOFIT, "memory", f"the eigenvalue analysis needs about {need:.1f} GB (factor {self.plan_gb:.1f} GB + vectors {extra_gb:.1f} GB), "
                                                  f"{max(budget, 0):.1f} GB are usable")

    def factor(self, sigma: float) -> tuple[int, int, int]:
        """Factors K − σM; → the inertia (positive, negative, zero)."""
        from .engines import EngineError
        vals = self.KU.data - sigma * self.MU.data if sigma != 0.0 else self.KU.data
        t = time.time()
        try:
            self.eng.factor("float64", vals)
        except EngineError as e:
            raise EigenFail(EXIT_NUMERIC, "numeric", f"the factorization of K − σM failed ({e})") from None
        finally:
            del vals
        self.info["t_factor"] = self.info.get("t_factor", 0.0) + time.time() - t
        self.info["n_factor"] = self.info.get("n_factor", 0) + 1
        self.sigma = float(sigma)
        return self.eng.inertia

    def solve(self, B: np.ndarray) -> np.ndarray:
        B2 = np.asarray(B, np.float64)
        return self.eng.solve(B2.reshape(self.n, -1)).reshape(B2.shape)

    def probe_limit(self) -> float:
        """The relative residual of a probe solve above which K − σM counts as singular: max(`PROBE_TOL`, `PROBE_SAT`/√n) – see `PROBE_SAT`."""
        if PROBE_FIXED:
            return PROBE_TOL
        return max(PROBE_TOL, PROBE_SAT / float(np.sqrt(self.n)))

    def probe(self) -> float:
        """The largest relative residual ‖(K − σM)y − x‖/‖x‖ of solves with the CURRENT factor for three fixed pseudo-random vectors x: a shift that sits on an eigenvalue (a singular
        K − σM, which the inertia does not show: no zero pivot is reported for it) leaves |g|/√n (0.03 on the 6-fold eigenvalue of a 10³ grid), any other shift ε·λ_max/distance –
        1e-13 … 1e-5 (`PROBE_SAT`)."""
        if not hasattr(self, "_ks"):
            self._ks, self._ms = Sym(self.KU), Sym(self.MU)
        x = np.random.default_rng(7).standard_normal((self.n, 3))
        y = self.solve(x)
        r = self._ks.mv(y) - self.sigma * self._ms.mv(y) - x
        return float(np.max(np.linalg.norm(r, axis=0) / np.linalg.norm(x, axis=0)))

    def trial(self, s: float, solves: bool) -> tuple[bool, bool, int]:
        """Factors K − sM → (clean, usable, the number of negative pivots): clean = no zero / perturbed pivot (the count is the inertia of a nearby matrix), usable = clean and – for a
        factor that will SOLVE (solves) at a shift above zero – the probe residual is below `probe_limit`. The factor stays in the engine."""
        try:
            npos, nneg, nzero = self.factor(s)
        except EigenFail as e:                              # a shift exactly on an eigenvalue: the engine refuses the factorization ("numerically rank-deficient") – no usable shift
            if e.kind != "numeric":
                raise
            self.refused = e.message
            return False, False, -1
        clean = nzero == 0 and self.eng.n_perturbed == 0
        ok = clean and (not solves or s <= 0.0 or self.probe() <= self.probe_limit())
        return clean, ok, int(nneg)

    def count_below(self, sigma: float, solves: bool = False, side: int = 1) -> int:
        """The number of eigenvalues of the pencil below the bound σ (Sylvester's law). The factor of the LAST shift tried stays in the engine (`self.sigma`: the shift the count
        was taken at). solves = the factor will also be used to SOLVE (the driving factor of a window above zero): a singular K − σM, which the inertia does not show (no zero pivot is
        reported for a σ exactly on an eigenvalue), is then detected by the solve probe (`probe_limit`). `side`: −1 = σ is the LOWER bound of a window, +1 = the upper bound (a Sturm
        check, the end of a window).

        A bound that cannot be used as it is (zero or perturbed pivot: σ is on an eigenvalue to rounding; a failed probe) moves OUTWARD – down for a lower bound, up for an upper
        one – by `MOVE_REL`·|σ|, then 10 times more per step, at most `MOVE_MAX`·|σ| (the old rule moved by 1e-7·min K_ii/M_ii, far above the low modes of a slender model: the modes between
        the requested and the moved bound were counted as "below the window" and the proof of completeness passed silently – measured: a clamped strip with f_min 10.4 Hz returned the
        modes from 593 Hz, 7 missing). The rule that makes this a proof:
          * the count at the requested σ itself is taken when its inertia is clean (a failed probe only); a moved shift is accepted when its count is EQUAL – no eigenvalue lies between
            σ and the shift, the gap is empty; a shift whose count differs crossed an eigenvalue, that direction is closed, the other one is tried ("shift the other way");
          * when both directions are closed (or the pivot at σ itself is not clean: σ sits ON an eigenvalue) the first usable shift on the OUTWARD side is taken and the eigenvalues in
            the zone between σ and the shift count as INSIDE the window – the window is widened by at most `MOVE_MAX`·|σ| (5e-4 in frequency), the number of eigenvalues in the zone
            is in `info["bounds"]`;
          * no usable shift within `MOVE_MAX`·|σ| = `EigenFail` (never a guess).
        A count alone (solves = False) needs only a clean inertia."""
        s0 = float(sigma)
        out = -1 if side < 0 else 1
        clean0, ok0, n0 = self.trial(s0, solves)
        if ok0:
            return n0
        ref = n0 if clean0 else None                    # the count at σ itself, when its inertia can be trusted
        scale = max(abs(s0), 1e-14 * self.lam_ref)
        closed = {1: False, -1: False}
        fallback: tuple[float, int] | None = None
        d = MOVE_REL * scale
        while d <= MOVE_MAX * scale * 1.0001 and not all(closed.values()):
            for sgn in (out, -out):
                if closed[sgn]:
                    continue
                s = s0 + sgn * d
                clean, ok, n = self.trial(s, solves)
                if clean and ref is not None and n != ref:
                    closed[sgn] = True                  # an eigenvalue lies between σ and s: a larger move crosses it as well
                    if ok and sgn == out and fallback is None:
                        fallback = (s, n)
                    continue
                if not ok or (ref is None and sgn != out):
                    continue                            # σ on an eigenvalue: only the outward side counts it inside the window
                return self._moved(s0, s, n, ref, out)
            d *= MOVE_GROW
        if fallback is not None:
            s, n = fallback
            self.trial(s, solves)                       # the factor of this shift again (only the last one stays in the engine)
            return self._moved(s0, s, n, ref, out)
        why = f"; the factorization said: {self.refused}" if getattr(self, "refused", None) else ""
        raise EigenFail(EXIT_NUMERIC, "numeric", f"the inertia of K − σM at the bound σ = {s0:.6g} could not be determined (zero or perturbed pivots, a singular shifted matrix "
                                                 f"within {MOVE_MAX:g}·σ{why})")

    def _moved(self, s0: float, s: float, n: int, ref: int | None, out: int) -> int:
        self.info.setdefault("bounds", []).append({"requested": s0, "used": s, "side": "lower" if out < 0 else "upper", "count": int(n),
                                                   "eigenvalues_between": None if ref is None else abs(int(n) - int(ref))})
        return int(n)

    def release_factor(self) -> None:
        self.eng.release_factor()


def _normalize_sign(X: np.ndarray) -> None:
    """The sign of every column: its largest component positive (in place)."""
    k = np.abs(X).argmax(axis=0)
    s = np.sign(X[k, np.arange(X.shape[1])])
    s[s == 0] = 1.0
    X *= s


def m_orthonormalize(X: np.ndarray, Msym: Sym, passes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky QR in the M inner product, repeated (CholQR2): → (Q, M·Q) with QᵀMQ = I. The leading columns of an already M-orthonormal set stay as they are. A
    set that is linearly dependent in the mass norm raises `EigenFail`."""
    Q = np.array(X, dtype=np.float64, copy=True)
    MQ = Msym.mv(Q)
    for _ in range(passes):
        G = Q.T @ MQ
        G = 0.5 * (G + G.T)
        try:
            Rf = sla.cholesky(G, lower=False)
        except sla.LinAlgError:
            raise EigenFail(EXIT_ACCURACY, "accuracy", "two modes are linearly dependent in the mass norm") from None
        Q = sla.solve_triangular(Rf, Q.T, trans="T", lower=False).T
        MQ = sla.solve_triangular(Rf, MQ.T, trans="T", lower=False).T
    return Q, MQ


def solve_modes(KU: sp.csr_matrix, MU: sp.csr_matrix, opts: EigenOptions, info: dict, log=print, rhs: np.ndarray | None = None,
                avail_gb: float | None = None) -> EigenResult:
    """The lowest `opts.modes` modes of the window [lam_min, lam_max], all of them proved by the Sturm count, mass-normalized; `rhs` (n, s) = static right-hand sides
    solved from the same factor (refinement of the shifted factor against K). An analysis whose residual / orthogonality check fails (a badly conditioned factor of a
    nearly free model, a shift too far from the wanted modes) is repeated with something that matters (`attempt_plan`: the shift of the positive definite factor – larger,
    then smaller than the first choice; the driving shift of a window lifted inside the proven gap; a larger Lanczos basis) – the check is the arbiter, never the shift."""
    base = SHIFT_REL_FREE if opts.free else SHIFT_REL
    for attempt in range(SHIFT_TRIES):
        plan = attempt_plan(attempt, base)
        info_try: dict = {}
        try:
            res = _solve_modes(KU, MU, opts, info_try, log, rhs, avail_gb, plan["shift_rel"], plan["lift"], plan["ncv_scale"])
        except EigenFail as e:
            retries = list(info.get("retries", []))
            info.update(info_try)
            if not e.retry or attempt == SHIFT_TRIES - 1:
                info["retries"] = retries
                raise
            nxt = attempt_plan(attempt + 1, base)
            retries.append({"attempt": attempt + 1, "why": e.message, "next": nxt})
            info["retries"] = retries
            info["shift_retries"] = attempt + 1
            log(f"The check of the modes failed ({e.message}) – repeating with {describe_plan(nxt, opts)}")
            continue
        retries = list(info.get("retries", []))
        info.update(info_try)
        info["retries"] = retries
        info["shift_retries"] = len(retries)
        res.info = info
        return res
    raise AssertionError("unreachable")


def attempt_plan(attempt: int, base_rel: float) -> dict:
    """What the attempt `attempt` (0 = the first) of an analysis does differently: the shift of the positive definite factor (`shift_rel`·λ_ref: ×30, ÷30, ×1e3 of the first choice –
    a free body needs it large against the rounding of the rigid-body modes, a very slender one small: ×1e3 of the free strip's default 1e-6 loses the modes in 1/(λ + s) and the Sturm
    count refuses the set, 1e-9 is both faster and more accurate there; measured `scratch/f14s0r/retry_spd2.py`), the factor by which
    the driving shift of a window is lifted above the requested bound (inside the proven gap, `lift_shift`; a window of a free body with its shift close to the rigid-body modes
    loses accuracy ∝ 1/σ – measured: residual 1.3e-7 at f_min 10 Hz, 2e-10 at 100 Hz), the size of the Lanczos basis."""
    rel = base_rel * (1.0, SHIFT_STEP, 1.0 / SHIFT_STEP, SHIFT_RETRY)[min(attempt, 3)]
    return {"shift_rel": rel, "lift": SHIFT_RETRY ** attempt, "ncv_scale": 1 + attempt}


def describe_plan(plan: dict, opts: EigenOptions) -> str:
    if opts.lam_min and opts.lam_min > 0.0:
        return f"the driving shift lifted ×{plan['lift']:g} inside the gap below the first mode of the window and a {plan['ncv_scale']}× larger Lanczos basis"
    return f"the shift {plan['shift_rel']:g}·λ_ref and a {plan['ncv_scale']}× larger Lanczos basis"


def lift_shift(sh: Shifted, sigma_lo: float, n_lo: int, lift: float) -> float:
    """The driving shift of a window lifted from the lower bound σ_lo up to `lift`·σ_lo while the count stays n_lo – no eigenvalue lies between σ_lo and the new shift, so the modes of
    the window are still the ones just above it (the count is the proof, a candidate that crossed an eigenvalue or is not usable to solve is divided by 30). The factor of the chosen
    shift stays in the engine; no candidate = σ_lo itself."""
    cand = sigma_lo * float(lift)
    while cand > 1.5 * sigma_lo:
        clean, ok, n = sh.trial(cand, True)
        if ok and n == n_lo:
            return float(cand)
        cand /= 30.0
    sh.factor(sigma_lo)
    return float(sigma_lo)


def _solve_modes(KU: sp.csr_matrix, MU: sp.csr_matrix, opts: EigenOptions, info: dict, log, rhs: np.ndarray | None, avail_gb: float | None,
                 shift_rel: float, lift: float = 1.0, ncv_scale: int = 1) -> EigenResult:
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigsh
    from . import eigen_vectors_gb
    t_all = time.time()
    n = KU.shape[0]
    Ksym, Msym = Sym(KU, opts.threads), Sym(MU, opts.threads)
    if avail_gb is None:
        import psutil
        avail_gb = psutil.virtual_memory().available / GB
    lam_ref = reference_scale(KU, MU)
    s = shift_rel * lam_ref
    k_req = int(opts.modes)
    if k_req < 1:
        raise EigenFail(EXIT_NUMERIC, "numeric", "the number of modes must be at least 1")
    if k_req >= n - 2:
        raise EigenFail(EXIT_NUMERIC, "numeric", "more modes requested than the model has degrees of freedom")
    info.update({"n": n, "nnz": int(KU.nnz), "lam_ref": lam_ref, "threads": opts.threads, "modes_requested": k_req})
    ncv = min(n, int(opts.ncv) or max(2 * k_req + 1, 20)) * max(1, int(ncv_scale))
    ncv = min(n, ncv)
    sh = Shifted(KU, MU, opts, info, lam_ref)
    sh.check_memory(eigen_vectors_gb(n, k_req, ncv), avail_gb)
    lam_lo = float(opts.lam_min or 0.0)
    # ---- the factor that drives the iteration: K − σM below the spectrum (positive definite, also for a free body), or at the lower bound of the window
    n_lo = 0
    if lam_lo > 0.0:
        if opts.free:                                   # a free body has its rigid-body modes at λ = 0 ± rounding noise: a window above zero starts above them
            band = rigid_limit(Ksym, MU)
            if lam_lo < band:
                info["window_raised"] = {"requested": lam_lo, "used": band, "why": "below the rounding noise of the rigid-body modes of a free body"}
                log(f"The lower bound of the window (λ = {lam_lo:.3g}) lies within the rounding noise of the rigid-body modes of a free body – raised to {band:.3g} "
                    f"({np.sqrt(band) / (2.0 * np.pi):.3g} Hz)")
                lam_lo = band
        n_lo = sh.count_below(lam_lo, solves=True, side=-1)      # the modes below the window; this factor (indefinite) is the driving one
        sigma0 = float(sh.sigma)
        want = float(lift)
        if opts.free:                                   # the rigid-body cluster at λ = 0 pollutes the Lanczos operator ∝ 1/σ (residual 1.3e-7 at σ = λ(10 Hz), 2e-10 at λ(100 Hz) on a free
            want = max(want, SHIFT_REL_FREE * lam_ref / max(sigma0, 1e-300))        # beam): the driving shift goes up to the shift of the positive definite path while the count stays
        if want > 1.5:
            sigma0 = lift_shift(sh, sigma0, n_lo, want)
            if sigma0 > float(lam_lo):
                info["sigma0_lifted"] = {"from": float(lam_lo), "to": sigma0, "count": int(n_lo)}
        spd = False
    else:
        sigma0 = -s
        for _ in range(6):
            npos, nneg, nzero = sh.factor(sigma0)
            if nneg == 0 and nzero == 0 and sh.eng.n_perturbed == 0:
                break
            sigma0 *= 1e3                               # rounding noise of a free body larger than the shift: the shift further below the spectrum
        else:
            raise EigenFail(EXIT_NUMERIC, "numeric", "K − σM is not positive definite for a shift below the spectrum (a negative stiffness – check the model)")
        spd = True
    info.update({"sigma0": sigma0, "n_below_window": n_lo, "spd": spd})
    n_cap = None
    if opts.lam_max is not None:                        # the number of modes in the window (the count at the upper bound replaces the factor – restored below)
        n_cap = sh.count_below(float(opts.lam_max), side=1) - n_lo
        if n_cap < 1:
            raise EigenFail(EXIT_INCOMPLETE, "incomplete", "no natural frequency in the requested range")
        k_req = min(k_req, n_cap)
        sh.factor(sigma0)
    v0 = np.random.default_rng(opts.seed).standard_normal(n)
    Kop = LinearOperator((n, n), matvec=Ksym.mv, matmat=Ksym.mv, dtype=np.float64)
    Mop = LinearOperator((n, n), matvec=Msym.mv, matmat=Msym.mv, dtype=np.float64)
    found_lam = np.zeros(0)
    found_X = np.zeros((n, 0))
    found_MX = np.zeros((n, 0))
    stats = {"solves": 0, "t_arpack": 0.0}

    def arpack(m: int) -> tuple[np.ndarray, np.ndarray]:
        V, MV = found_X, found_MX

        def solve(x):
            stats["solves"] += 1
            y = sh.solve(x)
            if V.shape[1]:
                y = y - V @ (MV.T @ y)                  # Q = I − V·VᵀM: the iteration stays M-orthogonal to the modes already found
            return y
        OPinv = LinearOperator((n, n), matvec=solve, dtype=np.float64)
        start = v0 - V @ (MV.T @ v0) if V.shape[1] else v0
        kk = min(m, n - 2)
        nc = min(n, max(ncv, 2 * kk + 1, 20))
        t = time.time()
        try:
            vals, vecs = eigsh(Kop, k=kk, M=Mop, sigma=sigma0, which="LA", OPinv=OPinv, v0=start, ncv=nc, tol=ARPACK_TOL, maxiter=max(500, 50 * kk))
        except ArpackNoConvergence as e:
            raise EigenFail(EXIT_ACCURACY, "accuracy", f"the Lanczos iteration did not converge ({len(e.eigenvalues)} of {kk} modes after {stats['solves']} solves)") from None
        except ArpackError as e:
            raise EigenFail(EXIT_ACCURACY, "accuracy", f"the Lanczos iteration failed: {e}") from None
        stats["t_arpack"] += time.time() - t
        return vals, vecs

    statics = None
    statics_done = rhs is None or not rhs.shape[1]

    def statics_of() -> np.ndarray | None:
        """The static solutions from the current factor; where they are an extra and cannot be had, None and the reason in the info (the analysis goes on without them)."""
        try:
            return static_solve(Ksym, sh, np.asarray(rhs, np.float64), info)
        except EigenFail as e:
            if not opts.statics_optional:
                raise
            info["statics_skipped"] = e.message
            log(f"The static solutions beside the modes are not available ({e.message[:120]}) - the modes without them")
            return None
    sturm_log: list[dict] = []
    m_need = k_req
    for attempt in range(MAX_PASSES):
        add = m_need - found_X.shape[1]
        if add > 0:
            vals, vecs = arpack(add)
            p0 = found_X.shape[1]
            found_lam = np.concatenate([found_lam, vals])
            Q, MQ = m_orthonormalize(np.column_stack([found_X, vecs]), Msym)
            found_X, found_MX = Q, MQ
            if p0 == 0 and not statics_done and spd:                # the static solutions need the factor of a shift below the spectrum (positive definite) – before the Sturm counts replace it
                statics = statics_of()
                statics_done = True
        order = np.argsort(found_lam, kind="stable")
        if not (order == np.arange(len(order))).all():          # already sorted (the usual case): no permuted copy of the two sets (0.8 GB each at 1M DOF, 100 modes)
            found_lam, found_X, found_MX = found_lam[order], found_X[:, order], found_MX[:, order]
        got = min(len(found_lam), m_need)
        top = float(found_lam[got - 1])
        sigma_ub = top + max(STURM_REL * abs(top), STURM_RES * float(np.finfo(np.float64).eps) * lam_ref)       # a few ε of the model scale: the resolution of the count, never a fraction of λ_ref that spans modes
        if n_cap is not None and got >= n_cap:
            sigma_ub = float(opts.lam_max)
        total = sh.count_below(sigma_ub, side=1) - n_lo
        sturm_log.append({"sigma_ub": sigma_ub, "count": int(total), "found": int(got)})
        if total == got:
            break
        if total < got:
            raise EigenFail(EXIT_INCOMPLETE, "incomplete", f"the Sturm count ({total} modes below {sigma_ub:.6g}) is smaller than the number of modes found ({got}) – "
                                                         "a mode was found twice; the analysis cannot be trusted", retry=True)
        m_need = total                                  # a mode was missed (or a partner of the last one lies within STURM_REL): find the rest with the found ones deflated
        log(f"Sturm count: {total} modes below the bound, {got} found – repeating with the found modes deflated")
        sh.factor(sigma0)                               # the driving factor again (the count replaced it)
    else:
        raise EigenFail(EXIT_INCOMPLETE, "incomplete", f"the Sturm count does not agree with the modes found after {MAX_PASSES} passes: "
                        f"{sturm_log[-1]['count']} modes expected, {sturm_log[-1]['found']} found", retry=True)
    if not statics_done:
        if not spd:                                      # a window above zero: the static solution needs a shifted factor of its own
            sh.factor(-s)
        statics = statics_of()
    sh.release_factor()
    info.update({"sturm": sturm_log, "arpack_solves": stats["solves"], "t_arpack": stats["t_arpack"], "arpack_passes": len(sturm_log)})
    k_out = min(k_req, len(found_lam))
    # the result sets: no copy when the found set is the result already (all modes kept, C-contiguous), else one C-ordered copy. X is taken and its source released BEFORE MX is
    # copied, so the transient is one set (0.8 GB at 1M DOF, 100 modes), not two. (rebound, not deleted: the closure `arpack` refers to the names)
    X = np.ascontiguousarray(found_X[:, :k_out])
    found_X = np.zeros((n, 0))
    MX = np.ascontiguousarray(found_MX[:, :k_out])
    found_MX = np.zeros((n, 0))
    # ---- Rayleigh quotients and the check of every pair, in blocks of columns (the columns are independent: the same numbers as for the whole set, without the
    # full-size K·X, M·X·λ and K·X − λ·M·X, ~3 GB at 1M DOF and 100 modes)
    lam = np.empty(k_out)
    rn, kn = np.empty(k_out), np.empty(k_out)
    lo = 0
    while lo < k_out:
        hi = min(lo + RESIDUAL_BLOCK, k_out)
        if k_out - hi == 1:                              # a single trailing column would go through the one-vector kernel: its last λ would differ by round-off from the
            hi = k_out                                   # product of the whole set – it joins the previous block, so the frequencies are bit-identical to the whole-set check
        Xb = np.ascontiguousarray(X[:, lo:hi])
        KXb = Ksym.mv(Xb)
        lam[lo:hi] = np.einsum("ij,ij->j", Xb, KXb)
        Rb = MX[:, lo:hi] * lam[None, lo:hi]
        np.subtract(KXb, Rb, out=Rb)
        rn[lo:hi] = np.linalg.norm(Rb, axis=0)
        kn[lo:hi] = np.linalg.norm(KXb, axis=0)
        del Xb, KXb, Rb
        lo = hi
    mn = np.linalg.norm(MX, axis=0) * np.abs(lam)
    eps = float(np.finfo(np.float64).eps)
    knorm = Ksym.norm1()
    phi2 = np.einsum("ij,ij->j", X, X)
    phin = np.sqrt(phi2)
    # the rigid-body modes of a free body: λ ≤ the strain energy of rounding noise AND 1e4 times below the first mode above
    floor_r = RIGID_ENERGY * eps * knorm * phi2
    above = lam[lam > floor_r]
    rigid = (lam <= floor_r) & (lam <= 1e-4 * (above.min() if len(above) else np.inf))
    # what is checked: a mode passes when ‖Kφ − λMφ‖ ≤ max(RES_TOL·max(‖Kφ‖, λ‖Mφ‖), BACKWARD_FLOOR·‖K‖₁‖φ‖) – the relative residual OR the backward error (the floor is the rounding
    # of K·φ itself: the stiffness matrix carries 1e-16·‖K‖ of noise, a mode of a slender model with λ₁ ≪ λ_max cannot be better than that). The report says which of the two held.
    rel = rn / np.maximum(np.maximum(kn, mn), 1e-300)
    bwd = rn / np.maximum(knorm * phin, 1e-300)
    limit = np.maximum(RES_TOL * np.maximum(kn, mn), BACKWARD_FLOOR * knorm * phin)
    elastic = ~rigid
    by_floor = elastic & (rel > RES_TOL)
    info["residual"] = {"criterion": f"‖Kφ − λMφ‖ ≤ max({RES_TOL:g}·max(‖Kφ‖, λ‖Mφ‖), {BACKWARD_FLOOR:g}·‖K‖₁·‖φ‖)", "modes": int(k_out),
                        "max_relative": float(rel[elastic].max()) if elastic.any() else 0.0,           # the elastic modes: the rigid-body ones are rounding noise in a relative measure
                        "max_backward_error": float(bwd.max()) if k_out else 0.0,
                        "accepted_by_backward_error": int(by_floor.sum()),
                        "max_of_limit": float((rn / np.maximum(limit, 1e-300)).max()) if k_out else 0.0}
    info["max_residual"] = info["residual"]["max_relative"]
    if (rn > limit).any():
        j = int(np.argmax(rn / limit))
        raise EigenFail(EXIT_ACCURACY, "accuracy", f"mode {j + 1} (λ = {lam[j]:.6g}) has a residual {rel[j]:.2e} > {RES_TOL:.0e} (backward error {bwd[j]:.1e} > {BACKWARD_FLOOR:.0e})")
    off = float(np.abs(X.T @ MX - np.eye(k_out)).max()) if k_out else 0.0
    info["max_orthogonality_error"] = off
    if off > ORTHO_TOL:
        raise EigenFail(EXIT_ACCURACY, "accuracy", f"the modes are not orthonormal in the mass norm (error {off:.2e})")
    _normalize_sign(X)
    info["rigid"] = [int(j) for j in np.nonzero(rigid)[0]]
    info["t_total"] = time.time() - t_all
    info["modes"] = int(k_out)
    return EigenResult(lam, X, info, statics, rigid)


def static_solve(Ksym: Sym, sh: Shifted, B: np.ndarray, info: dict, tol: float = 1e-12, maxit: int = 40) -> np.ndarray:
    """K·X = B from the CURRENT factor of K + s·M (σ = −s < 0, far below the first mode): the refinement X ← X + (K + sM)⁻¹(B − K·X) contracts by s/(λ₁ + s) per step and
    runs until the backward error ‖r‖/(‖K‖₁‖x‖ + ‖b‖) stops improving (a model with a rigid block on a soft support has a huge ‖K‖₁‖x‖: the first solution would pass
    the tolerance with a residual of 1 % of the load – the criterion alone does not say that the iteration has converged). When it does not converge (the shift is not
    far enough below the first mode) K itself is factored and the system solved with it; a free body (K singular) fails there → `EigenFail`."""
    if sh.sigma is None or sh.sigma >= 0:
        raise EigenFail(EXIT_NUMERIC, "numeric", "the static solution needs the factor of the shifted matrix")
    t = time.time()
    knorm = Ksym.norm1()
    bn = np.linalg.norm(B, axis=0)

    def backward(X):
        R = B - Ksym.mv(X)
        nr = np.linalg.norm(R, axis=0)
        den = knorm * np.linalg.norm(X, axis=0) + bn
        return R, np.where(den > 0, nr / np.where(den > 0, den, 1.0), nr)

    def refine(X, limit: int):
        hist: list[float] = []
        for _ in range(limit):
            R, bwd = backward(X)
            hist.append(float(bwd.max()))
            if bwd.max() <= 1e-15 or (len(hist) >= 3 and hist[-1] <= tol and hist[-1] > 0.5 * hist[-3]):      # the round-off floor
                return X, hist, True
            X = X + sh.solve(R)
        return X, hist, False

    X, hist, floor = refine(sh.solve(B), maxit)
    if not floor:
        inertia = sh.factor(0.0)                          # the shift is too close to the first mode: the plain K (positive definite for a constrained body)
        info["static_fallback_sigma0"] = True
        X, hist2, floor = refine(sh.solve(B), 6)
        hist = hist + hist2
        if not (floor or hist[-1] <= tol):
            raise EigenFail(EXIT_NUMERIC, "numeric", f"the static solution does not converge (backward error {hist[-1]:.1e}; history "
                                                     f"{', '.join(f'{h:.1e}' for h in hist[:4])}) – a free body or a missing constraint?")
        if inertia[1] or inertia[2] or sh.eng.n_perturbed:
            raise EigenFail(EXIT_NUMERIC, "numeric", "the stiffness matrix is not positive definite – a free body or a missing constraint?")
    info["static_backward_error"] = hist[-1]
    info["static_iterations"] = len(hist) - 1
    info["t_static"] = time.time() - t
    return X

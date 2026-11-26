"""Ready-made sparse direct solvers behind one small interface (`Engine`) – the worker `fea_core.direct.worker` only uses this.

Our code is glue only: the factorization, the ordering and the triangular solves come from the library. A second engine (MUMPS, a
separate program) can be added later as another class here without touching the worker – the plan F13 keeps it as the fallback if
rslab ever fails.

Matrix: the symmetric positive definite stiffness matrix as the UPPER triangle in CSR (indptr int64, indices, data). The same arrays are
the LOWER triangle in CSC, which is what rslab wants – no transposition, no second copy of the values in float64.
"""
from __future__ import annotations

import numpy as np

GB = 2**30


class EngineError(RuntimeError):
    """The engine could not factor the matrix (zero pivot, not positive definite)."""


class Engine:
    """Interface: analyze() once, plan_gb(), factor(precision) once, solve(B) many times, release()."""

    name = "?"
    version = ""

    def __init__(self, n: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, threads: int):
        raise NotImplementedError

    factor_nnz = 0                 # entries of the factor (after analyze)
    inertia = (0, 0, 0)            # (positive, negative, zero) eigenvalues of D (after factor)
    n_perturbed = 0                # pivots changed by the engine (after factor)

    def analyze(self) -> None:
        raise NotImplementedError

    def plan_gb(self, precision: str, nrhs: int) -> float:
        """Estimated peak memory of the engine alone (analysis + factorization + solution) [GB]."""
        raise NotImplementedError

    def factor(self, precision: str, values: np.ndarray | None = None) -> None:
        """Factors the matrix (`values` = other values on the SAME pattern, e.g. K − σM of the eigenvalue analysis; None = the values of the constructor)."""
        raise NotImplementedError

    def solve(self, B: np.ndarray) -> np.ndarray:
        """B (n, k) float64 → X (n, k) float64; the arithmetic is the precision of the factor."""
        raise NotImplementedError

    def release_factor(self) -> None:
        """Frees the factor only (the analysis stays – the next `factor` of the same pattern needs no new analysis)."""

    def release(self) -> None:
        pass


class RslabEngine(Engine):
    """rslab (pure Rust, MIT, pip wheel): supernodal LDL^T (Bunch-Kaufman), parallel nested dissection, float32 / float64."""

    name = "rslab"

    def __init__(self, n: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, threads: int):
        import rslab
        import scipy.sparse as sp                  # here, not at the top: this module is imported only by the worker process
        self._rslab = rslab
        self.version = rslab.__version__
        self.n = int(n)
        self.threads = int(threads)
        self._data = data                      # float64 values of the upper triangle (not copied)
        # analysis: the pattern only; values in float32 are enough and take half the memory
        A = sp.csc_matrix((np.asarray(data, dtype=np.float32), indices, indptr), shape=(n, n))
        A.has_canonical_format = True
        self._A = A
        self._sym = None
        self._f = None
        self._dtype = None

    def analyze(self) -> None:
        self._sym = self._rslab.analyze(self._A, path="ldlt", threads=self.threads)
        self.factor_nnz = int(self._sym.factor_nnz)
        self._A = None                         # the float32 copy of the pattern is only the input of the analysis (~370 MB at 1M DOF): freed before the factor is made

    def plan_gb(self, precision: str, nrhs: int) -> float:
        plan = self._sym.memory_plan(np.dtype(precision).name, nrhs=max(int(nrhs), 1))
        return plan["peak_bytes"] / GB

    def factor(self, precision: str, values: np.ndarray | None = None) -> None:
        dt = np.dtype(precision)
        self._f = None                         # a new factor replaces the old one (memory: never two at once)
        vals = np.ascontiguousarray(self._data if values is None else values, dtype=dt)
        try:
            self._f = self._sym.factor(vals)
        except RuntimeError as e:              # numerically zero pivot in exact mode = singular or not positive definite
            raise EngineError(f"rslab: {e}") from None
        finally:
            del vals
        self._dtype = dt
        self.inertia = tuple(int(x) for x in self._f.inertia)
        self.n_perturbed = int(self._f.n_perturbed)

    def solve(self, B: np.ndarray) -> np.ndarray:
        X = self._f.solve_many(np.ascontiguousarray(B, dtype=self._dtype))
        return np.asarray(X, dtype=np.float64).reshape(B.shape)

    def release_factor(self) -> None:
        self._f = None

    def release(self) -> None:
        self._f = None
        self._sym = None


ENGINES = {"rslab": RslabEngine}              # the Python module that carries an engine: fea_core.direct.LIBRARY


def make(name: str, n: int, indptr, indices, data, threads: int) -> Engine:
    try:
        cls = ENGINES[name]
    except KeyError:
        raise EngineError(f"Unknown direct solver engine “{name}” (available: {', '.join(ENGINES)})") from None
    return cls(n, indptr, indices, data, threads)

"""Direct-solver glue (release 1.5.1): the threaded K·X, the product reused by the accuracy check and the engine's pattern copy give the same numbers as before."""
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from fea_core.direct import eigen as E
from fea_core.direct import worker as W

needs_rslab = pytest.mark.skipif(importlib.util.find_spec("rslab") is None, reason="rslab (module direct) is needed")


def _upper(n=600, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.02, random_state=seed, format="csr")
    K = (A + A.T + sp.identity(n) * n * 0.05).tocsr()
    U = sp.triu(K, format="csr")
    U.sort_indices()
    return K, U, rng


@pytest.mark.parametrize("k", [1, 3, 4, 5, 7, 8, 9, 16])
def test_threaded_matmat_is_bit_identical_to_the_serial_product(k):
    K, U, rng = _upper()
    X = rng.standard_normal((K.shape[0], k))
    serial = W.SymmetricOperator(U, 1).matmat(X)
    assert np.array_equal(W.SymmetricOperator(U, 4).matmat(X), serial)
    assert np.abs(serial - K @ X).max() < 1e-11


def test_accuracy_with_a_known_product_equals_the_one_that_computes_it():
    K, U, rng = _upper()
    op = W.SymmetricOperator(U, 2)
    B = rng.standard_normal((K.shape[0], 5))
    X = B / 7.0
    a = W.accuracy(op, X, B, op.norm1())
    b = W.accuracy(op, X, B, op.norm1(), op.matmat(X))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@needs_rslab
def test_engine_frees_the_pattern_copy_after_the_analysis_and_still_factors_and_solves():
    from fea_core.direct.engines import make
    K, U, rng = _upper(400)
    K = K + sp.identity(400) * 50.0
    U = sp.triu(K, format="csr")
    U.sort_indices()
    eng = make("rslab", 400, U.indptr, U.indices, U.data, 2)
    eng.analyze()
    assert eng._A is None
    eng.factor("float64")
    B = rng.standard_normal((400, 3))
    X = eng.solve(B)
    assert np.abs(K @ X - B).max() < 1e-9 * np.abs(B).max()
    eng.factor("float64", U.data * 2.0)                       # a second factor of the same pattern needs no pattern copy either
    assert np.abs(2.0 * (K @ eng.solve(B)) - B).max() < 1e-9 * np.abs(B).max()


def test_threads_are_used_from_eight_columns_on(monkeypatch):
    """4–5 columns were slower threaded: the threshold is 8 columns (both the solver operator and the eigen operator)."""
    import fea_core.direct.eigen as EE
    K, U, rng = _upper()
    for cls, name in ((W.SymmetricOperator, "matmat"), (EE.Sym, "mv")):
        calls = []
        op = cls(U, 4)
        orig = op._block
        monkeypatch.setattr(op, "_block", lambda X, orig=orig, calls=calls: (calls.append(X.shape[1]), orig(X))[1])
        for k in (4, 5, 7):
            calls.clear()
            getattr(op, name)(rng.standard_normal((K.shape[0], k)))
            assert calls == [k], (name, k, calls)                     # one serial block
        calls.clear()
        getattr(op, name)(rng.standard_normal((K.shape[0], 8)))
        assert len(calls) == 4 and sum(calls) == 8, (name, calls)     # four chunks on four threads


# ---- the refinement path of solve_system (the K·X of the check is reused by the refinement step)
class _LuEngine:
    """A scipy LU behind the Engine interface: the refinement path runs without rslab."""
    name, version, factor_nnz, inertia, n_perturbed = "lu-test", "0", 1, (0, 0, 0), 0

    def __init__(self, n, indptr, indices, data, threads):
        U = sp.csr_matrix((data, indices, indptr), shape=(n, n))
        self.lu = None
        self.K = (U + U.T - sp.diags(U.diagonal())).tocsc()

    def analyze(self):
        pass

    def plan_gb(self, precision, nrhs):
        return 0.001

    def factor(self, precision, values=None):
        from scipy.sparse.linalg import splu
        self.lu = splu(self.K)
        self.inertia = (self.K.shape[0], 0, 0)

    def solve(self, B):
        return self.lu.solve(np.asarray(B, np.float64))

    def release_factor(self):
        self.lu = None

    def release(self):
        self.lu = None


def test_refinement_path_equals_the_separate_products(monkeypatch):
    """The forced refinement (a backward tolerance nobody reaches): two steps X ← X + K⁻¹(B − K·X) with ONE product per step. The reference computes every product separately
    (as before the change): the iterations, the backward errors and the residuals must be identical to the last bit."""
    from fea_core.direct import engines
    monkeypatch.setitem(engines.ENGINES, "lu-test", _LuEngine)
    K, U, rng = _upper(300)
    n = K.shape[0]
    B = rng.standard_normal((n, 6))
    opts = {"engine": "lu-test", "precision": "float64", "threads": 4}
    # reference: the old way, a product for the check and another one for the residual
    op = W.SymmetricOperator(U, 4)
    eng = _LuEngine(n, U.indptr, U.indices, U.data, 4)
    eng.factor("float64")
    X = eng.solve(B)
    for _ in range(2):
        X = X + eng.solve(B - op.matmat(X))
    bwd_ref, rel_ref = W.accuracy(op, X, B, op.norm1())
    monkeypatch.setattr(W, "BACKWARD_TOL", 0.0)
    info = {}
    with pytest.raises(W.WorkerFail):
        W.solve_system(U, B, opts, info, log=lambda s: None)
    assert info["iterations"] == [2] * 6
    assert info["backward_error"] == [float(v) for v in bwd_ref]
    assert info["rel_residual"] == [float(v) for v in rel_ref]


def test_accepted_float64_solution_is_the_unrefined_one_and_reports_the_checked_errors(monkeypatch):
    from fea_core.direct import engines
    monkeypatch.setitem(engines.ENGINES, "lu-test", _LuEngine)
    K, U, rng = _upper(300)
    B = rng.standard_normal((K.shape[0], 4))
    info = {}
    X = W.solve_system(U, B, {"engine": "lu-test", "precision": "float64", "threads": 4}, info, log=lambda s: None)
    eng = _LuEngine(K.shape[0], U.indptr, U.indices, U.data, 4)
    eng.factor("float64")
    assert np.array_equal(X, eng.solve(B))                     # accepted at once: no refinement step changed it
    assert info["iterations"] == [0] * 4
    bwd, rel = W.accuracy(W.SymmetricOperator(U, 4), X, B, W.SymmetricOperator(U, 4).norm1())
    assert info["backward_error"] == [float(v) for v in bwd] and info["rel_residual"] == [float(v) for v in rel]


# ---- the frequencies of the residual check do not depend on how the columns are cut into blocks
def _laplace(m=9):
    A = sp.diags([2 * np.ones(m), -np.ones(m - 1), -np.ones(m - 1)], [0, 1, -1]).tocsr()
    I = sp.identity(m, format="csr")
    K3 = (sp.kron(sp.kron(A, I), I) + sp.kron(sp.kron(I, A), I) + sp.kron(sp.kron(I, I), A)) * 1e6
    d = 1.0 + 0.3 * np.random.default_rng(3).random(K3.shape[0])
    K3 = (sp.diags(d) @ K3 @ sp.diags(d)).tocsr()
    KU = sp.triu(K3).tocsr()
    KU.sort_indices()
    rows = np.repeat(np.arange(KU.shape[0]), np.diff(KU.indptr))
    MU = sp.csr_matrix(((rows == KU.indices).astype(float), KU.indices, KU.indptr), shape=KU.shape)
    return KU, MU


@needs_rslab
@pytest.mark.parametrize("modes", [1, 16, 17, 18, 32, 33])
def test_frequencies_are_bit_identical_to_the_check_of_the_whole_set(modes):
    """k % 16 == 1 left one column in the last block (the one-vector kernel: λ differed by round-off). The whole-set product is the reference."""
    KU, MU = _laplace()
    res = E.solve_modes(KU, MU, E.EigenOptions(modes=modes, threads=4), {}, log=lambda s: None)
    assert len(res.lam) == modes
    whole = np.einsum("ij,ij->j", res.X, E.Sym(KU, 4).mv(res.X))
    assert np.array_equal(res.lam, whole)
    assert res.X.flags.c_contiguous

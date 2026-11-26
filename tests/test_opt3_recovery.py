"""Stress recovery speed-ups (release 1.5.1): the tuned element pass equals the reference formulas, `recover_batch` with an orthotropic field equals the per-field `recover`."""
import numpy as np

from fea_core.loads import GAUSS4, W_GAUSS4, _tet10_dN, inv3
from fea_core.recovery import _det3, _element_pass, recover, recover_batch, element_stiffness
from tests.test_recovery import _study
from tests.test_thermal import bar_mesh


def _reference_pass(nodes, conn, Ue, Ce):
    """The element pass as it was before the tuning (np.linalg.det, 3-index einsums)."""
    X = nodes[conn]
    n = len(conn)
    ip_sig = np.empty((n, 4, 6))
    fe = np.zeros((n, 10, 3))
    for g in range(4):
        dN = _tet10_dN(*GAUSS4[g])
        J = np.tensordot(dN, X, axes=(1, 1)).swapaxes(0, 1)
        dNx = inv3(J) @ dN
        G = dNx @ Ue
        eps = np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2], G[:, 0, 1] + G[:, 1, 0], G[:, 0, 2] + G[:, 2, 0], G[:, 1, 2] + G[:, 2, 1]], axis=1)
        sig = np.einsum("eij,ej->ei", Ce if Ce.ndim == 3 else Ce[:, g], eps)
        ip_sig[:, g] = sig
        st = np.empty((n, 3, 3))
        st[:, 0, 0], st[:, 1, 1], st[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
        st[:, 0, 1] = st[:, 1, 0] = sig[:, 3]
        st[:, 0, 2] = st[:, 2, 0] = sig[:, 4]
        st[:, 1, 2] = st[:, 2, 1] = sig[:, 5]
        fe += (W_GAUSS4 * np.abs(np.linalg.det(J)))[:, None, None] * np.einsum("eja,ejk->eak", dNx, st)
    return ip_sig, fe


def _field(m, seed=3):
    """A stand-in of `ortho.ElementField`: a different (symmetric, positive) C at every Gauss point."""
    rng = np.random.default_rng(seed)
    C0 = element_stiffness(_study(), m)
    Cg = np.repeat(C0[:, None], 4, axis=1) * (1.0 + 0.3 * rng.random((len(C0), 4, 1, 1)))

    class F:
        def chunk(self, lo, hi, X):
            return Cg[lo:hi], np.zeros((hi - lo, 4, 6))
    return F(), Cg


def _displacements(m, B=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, len(m.nodes), 3)) * 1e-3 + np.einsum("ij,bj->bi", np.eye(3), np.zeros((B, 3)))[:, None, :]


def test_det3_matches_linalg_det():
    J = np.random.default_rng(0).standard_normal((500, 3, 3))
    assert np.allclose(_det3(J), np.linalg.det(J), rtol=1e-13, atol=1e-14)


def test_element_pass_equals_reference():
    m = bar_mesh()
    U = _displacements(m, 1)[0]
    C = element_stiffness(_study(), m)
    ip, fe = _element_pass(m.nodes, m.tets, U[m.tets], C)
    ip0, fe0 = _reference_pass(m.nodes, m.tets, U[m.tets], C)
    assert np.abs(ip - ip0).max() <= 1e-12 * np.abs(ip0).max()
    assert np.abs(fe - fe0).max() <= 1e-12 * np.abs(fe0).max()
    _, Cg = _field(m)
    ip, fe = _element_pass(m.nodes, m.tets, U[m.tets], Cg)                # C at the Gauss points
    ip0, fe0 = _reference_pass(m.nodes, m.tets, U[m.tets], Cg)
    assert np.abs(ip - ip0).max() <= 1e-12 * np.abs(ip0).max()
    assert np.abs(fe - fe0).max() <= 1e-12 * np.abs(fe0).max()


def test_recover_batch_with_field_equals_per_field_recover():
    m = bar_mesh()
    U = _displacements(m, 3)
    field, _ = _field(m)
    S, RF = recover_batch(_study(), m, U, None, field)
    for b in range(len(U)):
        s1, r1 = recover(_study(), m, U[b], None, None, field)
        assert np.abs(S[b] - s1).max() <= 1e-11 * np.abs(s1).max()
        assert np.abs(RF[b] - r1).max() <= 1e-11 * np.abs(r1).max()
    S2, RF2 = recover_batch(_study(), m, U, None, field, forces=False)
    assert RF2 is None and np.array_equal(S2, S)

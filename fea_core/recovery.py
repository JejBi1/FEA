"""Stresses and nodal forces from the displacements in the same way as ccx (C3D10) – for solvers other than ccx (the direct solver rslab, hypre).

ccx: stresses at the 4 Gauss points (gauss3d4) → linear extrapolation to the corners (IP j lies at corner j), midpoints of edges =
average of the corners, node = plain average over all elements of the node. Verified 13. 10. 2026 (prototype of the stress variants): from the displacements at full precision agreement with the .frd of ccx 8·10⁻⁶ (max., plate + foot 197 k DOF);
the projection of NGSolve (`Set`) differed up to 28 % at the midpoints of edges. Nodal forces = Σ_g w·|J|·Bᵀσ (the same 4 points).
This is neither the stiffness matrix nor a solver – only a recomputation from the result (like printing.failure, joints.forces).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .loads import GAUSS4, GAUSS4_A as _A, GAUSS4_B as _B, W_GAUSS4, _tet10_dN, inv3
from .mesh import Mesh

EXTRAP = np.full((4, 4), -_B / (_A - _B)) + np.eye(4) / (_A - _B)   # roh i ← hodnoty v IP j
MID_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])  # nodes 4–9 of C3D10 (ccx order)
VOIGT_TO_CCX = [0, 1, 2, 3, 5, 4]        # xx yy zz xy xz yz → ccx xx yy zz xy yz zx


def element_stiffness(study, mesh: Mesh, elems: np.ndarray | None = None) -> np.ndarray:
    """C (E, 6, 6) per element (Voigt xx yy zz xy xz yz, engineering shear; 3D printing = C_el as for ccx).

    elems = only the selected elements (isotropic materials do not assemble C for the whole mesh).
    """
    from .solve_large import element_stiffness_all, iso_stiffness
    C = element_stiffness_all(study, mesh)
    if C is not None:
        return C if elems is None else C[elems]
    Cp = np.array([iso_stiffness(study.material_of(p).E, study.material_of(p).nu) for p in mesh.part_names])
    return Cp[mesh.tet_part if elems is None else mesh.tet_part[elems]]


def _det3(J: np.ndarray) -> np.ndarray:
    """Determinants of 3×3 matrices (E, 3, 3) by the explicit expansion (≈ 3× faster than `np.linalg.det` for small matrices; the same value up to round-off)."""
    return (J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]) - J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0])
            + J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]))


def _chunks(n: int, size: int = 200_000):
    for lo in range(0, n, size):
        yield lo, min(n, lo + size)


def _element_pass(nodes: np.ndarray, conn: np.ndarray, Ue: np.ndarray, Ce: np.ndarray,
                  alpha_e: np.ndarray | None = None, dTe: np.ndarray | None = None, a6: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Core of the recomputation for a batch of C3D10 elements: (σ at 4 IP (E, 4, 6) in the order xx yy zz xy xz yz, fe (E, 10, 3) = ∫Bᵀσ).

    Ue = displacements of the nodes of the elements (E, 10, 3), Ce = C of the elements (E, 6, 6) – or at the Gauss points (E, 4, 6, 6) for a cylindrically
    orthotropic material; thermal load: alpha_e (E,) and dTe = ΔT of the nodes of the elements (E, 10) → σ = C·(ε − α·ΔT·(1,1,1,0,0,0)), ΔT at the IP by
    the shape functions. a6 (E, 4, 6) = the thermal strain per unit ΔT at the Gauss points (anisotropic expansion, replaces alpha_e).
    """
    from .results import tet10_shape
    X = nodes[conn]
    n = len(conn)
    dN = [_tet10_dN(*p) for p in GAUSS4]
    Ng = [tet10_shape(np.asarray(p)) for p in GAUSS4] if dTe is not None else None
    ip_sig = np.empty((n, 4, 6))
    fe = np.zeros((n, 10, 3))
    st = np.empty((n, 3, 3))
    for g in range(4):
        J = np.tensordot(dN[g], X, axes=(1, 1)).swapaxes(0, 1)            # (E, 3, 3)
        dNx = inv3(J) @ dN[g]                                             # (E, 3, 10)
        G = dNx @ Ue                                                      # ∂u_k/∂x_a
        eps = np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2], G[:, 0, 1] + G[:, 1, 0],
                        G[:, 0, 2] + G[:, 2, 0], G[:, 1, 2] + G[:, 2, 1]], axis=1)
        if dTe is not None:
            if a6 is None:
                eps[:, :3] -= (alpha_e * (dTe @ Ng[g]))[:, None]
            else:
                eps -= a6[:, g] * (dTe @ Ng[g])[:, None]
        sig = np.matmul(Ce if Ce.ndim == 3 else Ce[:, g], eps[:, :, None])[:, :, 0]   # xx yy zz xy xz yz
        ip_sig[:, g] = sig
        st[:, 0, 0], st[:, 1, 1], st[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
        st[:, 0, 1] = st[:, 1, 0] = sig[:, 3]
        st[:, 0, 2] = st[:, 2, 0] = sig[:, 4]
        st[:, 1, 2] = st[:, 2, 1] = sig[:, 5]
        detJ = np.abs(_det3(J))
        fe += (W_GAUSS4 * detJ)[:, None, None] * np.matmul(dNx.swapaxes(1, 2), st)
    return ip_sig, fe


def element_nodal_forces(study, mesh: Mesh, U: np.ndarray, elems: np.ndarray,
                         dT: np.ndarray | None = None) -> np.ndarray:
    """Internal nodal forces fe (k, 10, 3) of the selected C3D10 elements: ∫Bᵀσ dV over the 4 Gauss points (as RF in `recover`).

    dT = T − T_ref per node (thermal load) → σ = C(ε − αΔT). The matrix C is assembled only for the selected elements.
    """
    if mesh.tets.shape[1] != 10:
        raise ValueError("Element nodal forces are only available for tet10 (C3D10)")
    elems = np.asarray(elems, int)
    if len(elems) == 0:
        return np.zeros((0, 10, 3))
    conn = mesh.tets[elems]
    alpha_e = dTe = None
    if dT is not None:
        alpha_e = np.array([study.material_of(p).alpha or 0.0 for p in mesh.part_names])[mesh.tet_part[elems]]
        dTe = np.asarray(dT, float)[conn]
    return _element_pass(mesh.nodes, conn, U[conn], element_stiffness(study, mesh, elems), alpha_e, dTe)[1]


def element_ip_stress(study, mesh: Mesh, U: np.ndarray, elems: np.ndarray, dT: np.ndarray | None = None,
                      C: np.ndarray | None = None, field=None) -> np.ndarray:
    """Stresses at the 4 integration points of the selected C3D10 elements (k, 4, 6) in the order of the `*EL PRINT S` of ccx (xx yy zz xy yz zx) – what the
    forces of the welds and of the clamped parts (`joints.evaluate.step_forces`) are made of; the same σ as in `recover` (thermal: σ = C·(ε − α·ΔT)).

    C = the matrices of all elements when the model has printed parts (otherwise the isotropic C per part), field = `ortho.ElementField` (orthotropic materials).
    """
    if mesh.tets.shape[1] != 10:
        raise ValueError("Stresses at the integration points are only available for tet10 (C3D10)")
    elems = np.asarray(elems, int)
    if len(elems) == 0:
        return np.zeros((0, 4, 6))
    conn = mesh.tets[elems]
    dTe = None if dT is None else np.asarray(dT, float)[conn]
    if field is not None:
        import copy
        sub = copy.copy(field)                                            # the field of the selected elements only (its chunks are contiguous ranges)
        sub.part = field.part[elems]
        Cg, Ag = sub.chunk(0, len(elems), mesh.nodes[conn])
        sig = _element_pass(mesh.nodes, conn, U[conn], Cg, None, dTe, Ag)[0]
    else:
        Ce = element_stiffness(study, mesh, elems) if C is None else C[elems]
        alpha = None if dT is None else np.array([study.material_of(p).alpha or 0.0 for p in mesh.part_names])[mesh.tet_part[elems]]
        sig = _element_pass(mesh.nodes, conn, U[conn], Ce, alpha, dTe)[0]
    return sig[:, :, VOIGT_TO_CCX]


def recover(study, mesh: Mesh, U: np.ndarray, dT: np.ndarray | None = None,
            C: np.ndarray | None = None, field=None) -> tuple[np.ndarray, np.ndarray]:
    """(S (N, 6) in ccx order, RF (N, 3) = internal nodal forces ∫Bᵀσ) from the displacements U (N, 3).

    dT = T − T_ref per node (thermal load): σ = C·(ε − α·ΔT·(1,1,1,0,0,0)), ΔT at the IP by the shape functions.
    field = `ortho.ElementField` (orthotropic materials: C and the thermal strain at the Gauss points; replaces C and the isotropic α).
    """
    if mesh.tets.shape[1] != 10:
        raise ValueError("Stress recovery is available only for tet10 (C3D10)")
    if C is None and field is None:
        C = element_stiffness(study, mesh)
    alpha = None
    if dT is not None and field is None:
        alpha = np.array([study.material_of(p).alpha or 0.0 for p in mesh.part_names])[mesh.tet_part]
    n = len(mesh.nodes)
    S_acc = np.zeros((n, 6))
    RF = np.zeros((n, 3))
    for lo, hi in _chunks(len(mesh.tets), 200_000 if field is None else 20_000):
        conn = mesh.tets[lo:hi]
        if field is not None:
            Cg, Ag = field.chunk(lo, hi, mesh.nodes[conn])
            corner_sig, fe = _element_pass(mesh.nodes, conn, U[conn], Cg, None, None if dT is None else dT[conn], Ag)
        else:
            corner_sig, fe = _element_pass(mesh.nodes, conn, U[conn], C[lo:hi],
                                           None if dT is None else alpha[lo:hi],
                                           None if dT is None else dT[conn])
        corner = np.einsum("ij,ejk->eik", EXTRAP, corner_sig)             # (E, 4, 6)
        mids = 0.5 * (corner[:, MID_EDGES[:, 0]] + corner[:, MID_EDGES[:, 1]])
        allv = np.concatenate([corner, mids], axis=1)                     # (E, 10, 6)
        flat = conn.ravel()
        for k in range(6):
            S_acc[:, k] += np.bincount(flat, weights=allv[:, :, k].ravel(), minlength=n)
        for k in range(3):
            RF[:, k] += np.bincount(flat, weights=fe[:, :, k].ravel(), minlength=n)
    cnt = np.bincount(mesh.tets.ravel(), minlength=n).astype(float)
    S = np.divide(S_acc, cnt[:, None], out=np.zeros_like(S_acc), where=cnt[:, None] > 0)
    return S[:, VOIGT_TO_CCX], RF


def recover_batch(study, mesh: Mesh, U: np.ndarray, C: np.ndarray | None = None, field=None, forces: bool = True,
                  stress: bool = True) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`recover` for a batch of displacement fields U (B, N, 3) at once (the modes of a modal analysis): the geometry of an element chunk (Jacobians, ∂N/∂x) is made
    once for all B fields. → (S (B, N, 6) in ccx order – None for stress=False, RF (B, N, 3) = the internal nodal forces – None for forces=False). The arithmetic is that of
    `recover` (the same points and rules); orthotropic parts (`field`: C at the Gauss points) share the geometry and C of a chunk for all B fields as well. No thermal load."""
    if mesh.tets.shape[1] != 10:
        raise ValueError("Stress recovery is available only for tet10 (C3D10)")
    U = np.asarray(U, np.float64)
    B, n = U.shape[0], len(mesh.nodes)
    if field is None and C is None:
        C = element_stiffness(study, mesh)
    from .results import tet10_shape  # noqa: F401  (kept next to `recover`: the same shape functions are used by `_element_pass`)
    S_acc: Any = np.zeros((B, n, 6)) if stress else None
    RF = np.zeros((B, n, 3)) if forces else None
    dN = [_tet10_dN(*p) for p in GAUSS4]
    step = max(5_000, 200_000 // max(B, 1)) if field is None else max(2_000, min(20_000, 200_000 // max(B, 1)))
    for lo, hi in _chunks(len(mesh.tets), step):
        conn = mesh.tets[lo:hi]
        X = mesh.nodes[conn]
        if field is None:
            assert C is not None
            Ce = C[lo:hi]                                                     # (E, 6, 6)
        else:
            Ce = field.chunk(lo, hi, X)[0]                                    # at the Gauss points (E, 4, 6, 6)
        Ue = U[:, conn]                                                   # (B, E, 10, 3)
        ip: Any = np.empty((B, hi - lo, 4, 6)) if stress else None
        fe = np.zeros((B, hi - lo, 10, 3)) if forces else None
        for g in range(4):
            J = np.tensordot(dN[g], X, axes=(1, 1)).swapaxes(0, 1)        # (E, 3, 3)
            dNx = inv3(J) @ dN[g]                                         # (E, 3, 10)
            G = np.einsum("eaj,bejk->beak", dNx, Ue)                      # ∂u_k/∂x_a
            eps = np.stack([G[..., 0, 0], G[..., 1, 1], G[..., 2, 2], G[..., 0, 1] + G[..., 1, 0],
                            G[..., 0, 2] + G[..., 2, 0], G[..., 1, 2] + G[..., 2, 1]], axis=-1)       # (B, E, 6)
            sig = np.einsum("eij,bej->bei", Ce if Ce.ndim == 3 else Ce[:, g], eps)
            if stress:
                ip[:, :, g] = sig
            if forces:
                st = np.empty((B, hi - lo, 3, 3))
                st[..., 0, 0], st[..., 1, 1], st[..., 2, 2] = sig[..., 0], sig[..., 1], sig[..., 2]
                st[..., 0, 1] = st[..., 1, 0] = sig[..., 3]
                st[..., 0, 2] = st[..., 2, 0] = sig[..., 4]
                st[..., 1, 2] = st[..., 2, 1] = sig[..., 5]
                detJ = np.abs(_det3(J))
                fe += (W_GAUSS4 * detJ)[None, :, None, None] * np.einsum("eja,bejk->beak", dNx, st)
        flat = conn.ravel()
        if stress:
            corner = np.einsum("ij,bejk->beik", EXTRAP, ip)               # (B, E, 4, 6)
            mids = 0.5 * (corner[:, :, MID_EDGES[:, 0]] + corner[:, :, MID_EDGES[:, 1]])
            allv = np.concatenate([corner, mids], axis=2)                 # (B, E, 10, 6)
        for b in range(B):
            if stress:
                for k in range(6):
                    S_acc[b, :, k] += np.bincount(flat, weights=allv[b, :, :, k].ravel(), minlength=n)
            if forces:
                for k in range(3):
                    RF[b, :, k] += np.bincount(flat, weights=fe[b, :, :, k].ravel(), minlength=n)
    if not stress:
        return None, RF
    cnt = np.bincount(mesh.tets.ravel(), minlength=n).astype(float)
    S = np.divide(S_acc, cnt[None, :, None], out=np.zeros_like(S_acc), where=cnt[None, :, None] > 0)
    return S[:, :, VOIGT_TO_CCX], RF

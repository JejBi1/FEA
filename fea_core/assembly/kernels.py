"""Numba kernels of the own matrix assembly of the C3D10 (tet10) element: stiffness matrix, thermal load, CSR of the equation system.

Imported only in the assembly worker (`fea_core.assembly.worker`) and in tests – numba is a module of the program (accel), the core does not
need it.

The matrix is kept in 3×3 NODE blocks (BSR layout: `indptr` / `indices` per node, `data` (nnzb, 3, 3)); the upper triangle of the blocks
(node column ≥ node row) or the whole matrix. Integration = the 4 Gauss points of ccx C3D10 (the same points as `fea_core.recovery` and
`fea_core.loads.GAUSS4`), so K·u equals the internal nodal forces that `recovery.recover` computes from u. Rows are assembled node by node:
every thread writes only its own block rows and adds the element contributions in ascending element order → the matrix is bitwise identical for any
number of threads. Material of an element = an index into a table of 6×6 matrices (Voigt xx yy zz xy xz yz, engineering shear): isotropic per part,
3D-printing groups (shell / core with the orientation of the lines) and general anisotropy are the same case.
"""
from __future__ import annotations

import numba as nb
import numpy as np

GA, GB = 0.5854101966249685, 0.1381966011250105          # the Gauss points of C3D10 (loads.GAUSS4_A / _B)
GAUSS4 = np.array([[GB, GB, GB], [GA, GB, GB], [GB, GA, GB], [GB, GB, GA]])
W4 = 1.0 / 24.0


def tet10_dN(r: float, s: float, t: float) -> np.ndarray:
    """Derivatives of the shape functions of C3D10 (ccx order) by (r, s, t), shape (3, 10) – the same as `loads._tet10_dN`."""
    u = 1 - r - s - t
    a = -(4 * u - 1)
    return np.array([
        [a, 4 * r - 1, 0, 0, 4 * (u - r), 4 * s, -4 * s, -4 * t, 4 * t, 0],
        [a, 0, 4 * s - 1, 0, -4 * r, 4 * r, 4 * (u - s), -4 * t, 0, 4 * t],
        [a, 0, 0, 4 * t - 1, -4 * r, 0, -4 * s, 4 * (u - t), 4 * r, 4 * s],
    ])


def tet10_N(r: float, s: float, t: float) -> np.ndarray:
    """Shape functions of C3D10 (ccx order), shape (10,) – the same as `results.tet10_shape`."""
    u = 1 - r - s - t
    return np.array([u * (2 * u - 1), r * (2 * r - 1), s * (2 * s - 1), t * (2 * t - 1),
                     4 * u * r, 4 * r * s, 4 * s * u, 4 * u * t, 4 * r * t, 4 * s * t])


DN4 = np.ascontiguousarray(np.stack([tet10_dN(*p) for p in GAUSS4]))        # (4, 3, 10)
N4 = np.ascontiguousarray(np.stack([tet10_N(*p) for p in GAUSS4]))          # (4, 10)


def iso_stiffness(E: float, nu: float) -> np.ndarray:
    """Isotropic C (6, 6), Voigt xx yy zz xy xz yz, engineering shear."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


# ---------------------------------------------------------------------------------------------- adjacency and pattern
@nb.njit(cache=True)
def node_elements(nn, tets):
    """CSR node → (element, local index of the node in the element); the elements of a node are in ascending order."""
    ne, k = tets.shape
    ptr = np.zeros(nn + 1, np.int64)
    for e in range(ne):
        for j in range(k):
            ptr[tets[e, j] + 1] += 1
    for a in range(nn):
        ptr[a + 1] += ptr[a]
    pos = ptr[:-1].copy()
    el = np.empty(ne * k, np.int32)
    loc = np.empty(ne * k, np.int8)
    for e in range(ne):
        for j in range(k):
            n = tets[e, j]
            el[pos[n]] = e
            loc[pos[n]] = j
            pos[n] += 1
    return ptr, el, loc


@nb.njit(parallel=True, cache=True)
def _row_counts(nn, tets, ptr, el, upper, xptr, xcol):
    cnt = np.zeros(nn, np.int64)
    for a in nb.prange(nn):
        m = (ptr[a + 1] - ptr[a]) * 10 + (xptr[a + 1] - xptr[a])
        buf = np.empty(m, np.int32)
        q = 0
        for p in range(ptr[a], ptr[a + 1]):
            e = el[p]
            for j in range(10):
                b = tets[e, j]
                if (not upper) or b >= a:
                    buf[q] = b
                    q += 1
        for p in range(xptr[a], xptr[a + 1]):          # the couplings of the constraints (not from any element)
            b = xcol[p]
            if (not upper) or b >= a:
                buf[q] = b
                q += 1
        s = np.sort(buf[:q])
        c = 0
        for i in range(q):
            if i == 0 or s[i] != s[i - 1]:
                c += 1
        cnt[a] = c
    return cnt


@nb.njit(parallel=True, cache=True)
def _row_fill(nn, tets, ptr, el, upper, indptr, indices, xptr, xcol):
    for a in nb.prange(nn):
        m = (ptr[a + 1] - ptr[a]) * 10 + (xptr[a + 1] - xptr[a])
        buf = np.empty(m, np.int32)
        q = 0
        for p in range(ptr[a], ptr[a + 1]):
            e = el[p]
            for j in range(10):
                b = tets[e, j]
                if (not upper) or b >= a:
                    buf[q] = b
                    q += 1
        for p in range(xptr[a], xptr[a + 1]):
            b = xcol[p]
            if (not upper) or b >= a:
                buf[q] = b
                q += 1
        s = np.sort(buf[:q])
        w = indptr[a]
        for i in range(q):
            if i == 0 or s[i] != s[i - 1]:
                indices[w] = s[i]
                w += 1


_NO_EXTRA = (np.zeros(1, np.int64), np.zeros(0, np.int32))


def extra_csr(nn: int, pa: np.ndarray, pb: np.ndarray, upper: bool) -> tuple[np.ndarray, np.ndarray]:
    """CSR (xptr (nn + 1,), xcol) of the extra block couplings (pa[i], pb[i]) that no element makes (constraints: a node and the masters of a dependent
    node): the pair is symmetric – the upper triangle keeps (min, max), the whole matrix both orientations; duplicates are merged by the pattern."""
    pa = np.asarray(pa, np.int64)
    pb = np.asarray(pb, np.int64)
    if upper:
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        rows, cols = lo, hi
    else:
        rows, cols = np.concatenate([pa, pb]), np.concatenate([pb, pa])
    order = np.argsort(rows, kind="stable")
    xptr = np.zeros(nn + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=nn), out=xptr[1:])
    return xptr, np.ascontiguousarray(cols[order], dtype=np.int32)


def pattern(nn, tets, ptr, el, upper, extra=None):
    """Block pattern (node × node) with sorted columns; upper = only the columns ≥ the row. → (indptr int64, indices int32).
    extra = (xptr, xcol) of `extra_csr`: couplings of the constraints that no element makes (None = the pattern of the elements only)."""
    xptr, xcol = _NO_EXTRA if extra is None else extra
    if len(xptr) == 1:
        xptr = np.zeros(nn + 1, np.int64)
    cnt = _row_counts(nn, tets, ptr, el, upper, xptr, xcol)
    indptr = np.zeros(nn + 1, np.int64)
    np.cumsum(cnt, out=indptr[1:])
    indices = np.empty(indptr[-1], np.int32)
    _row_fill(nn, tets, ptr, el, upper, indptr, indices, xptr, xcol)
    return indptr, indices


# ---------------------------------------------------------------------------------------------- the check of the elements
@nb.njit(parallel=True, cache=True)
def min_jacobian(nodes, tets, DN):
    """The smallest determinant of the Jacobian over the 4 Gauss points of every element (E,); NaN when one of them is NaN (damaged coordinates).
    The same arithmetic as the stiffness kernel. ccx accepts an element only when the determinant is positive at every integration point
    ("nonpositive jacobian": an inverted element = a reversed node order, a flat element = zero volume), the own assembly asks the same."""
    ne = tets.shape[0]
    out = np.empty(ne)
    for e in nb.prange(ne):
        w = np.inf
        for g in range(4):
            j00 = j01 = j02 = j10 = j11 = j12 = j20 = j21 = j22 = 0.0
            for n in range(10):
                b = tets[e, n]
                d0 = DN[g, 0, n]
                d1 = DN[g, 1, n]
                d2 = DN[g, 2, n]
                x = nodes[b, 0]
                y = nodes[b, 1]
                z = nodes[b, 2]
                j00 += d0 * x
                j01 += d0 * y
                j02 += d0 * z
                j10 += d1 * x
                j11 += d1 * y
                j12 += d1 * z
                j20 += d2 * x
                j21 += d2 * y
                j22 += d2 * z
            det = j00 * (j11 * j22 - j12 * j21) + j01 * (j12 * j20 - j10 * j22) + j02 * (j10 * j21 - j11 * j20)
            if det != det:
                w = det                      # NaN stays (every later comparison with it is False)
                break
            if det < w:
                w = det
        out[e] = w
    return out


# ---------------------------------------------------------------------------------------------- stiffness matrix
@nb.njit(parallel=True, cache=True)
def stiffness(nodes, tets, mat_of, Cm, ptr, el, loc, indptr, indices, DN, upper, data):
    """data (nnzb, 3, 3) += the blocks of the stiffness matrix. mat_of (E,) index into Cm (M, 6, 6)."""
    nn = nodes.shape[0]
    for a in nb.prange(nn):
        r0 = indptr[a]
        r1 = indptr[a + 1]
        X = np.empty((10, 3))
        G = np.empty((3, 10))
        CB = np.empty((6, 3))
        slot = np.empty(10, np.int64)
        for p in range(ptr[a], ptr[a + 1]):
            e = el[p]
            la = loc[p]
            C = Cm[mat_of[e]]
            for j in range(10):
                b = tets[e, j]
                X[j, 0] = nodes[b, 0]
                X[j, 1] = nodes[b, 1]
                X[j, 2] = nodes[b, 2]
                if upper and b < a:
                    slot[j] = -1
                    continue
                lo = r0
                hi = r1
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if indices[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid
                slot[j] = lo
            for g in range(4):
                j00 = j01 = j02 = j10 = j11 = j12 = j20 = j21 = j22 = 0.0
                for n in range(10):
                    d0 = DN[g, 0, n]
                    d1 = DN[g, 1, n]
                    d2 = DN[g, 2, n]
                    x = X[n, 0]
                    y = X[n, 1]
                    z = X[n, 2]
                    j00 += d0 * x
                    j01 += d0 * y
                    j02 += d0 * z
                    j10 += d1 * x
                    j11 += d1 * y
                    j12 += d1 * z
                    j20 += d2 * x
                    j21 += d2 * y
                    j22 += d2 * z
                A0 = j11 * j22 - j12 * j21
                B0 = j12 * j20 - j10 * j22
                C0 = j10 * j21 - j11 * j20
                det = j00 * A0 + j01 * B0 + j02 * C0
                i00 = A0 / det
                i01 = (j02 * j21 - j01 * j22) / det
                i02 = (j01 * j12 - j02 * j11) / det
                i10 = B0 / det
                i11 = (j00 * j22 - j02 * j20) / det
                i12 = (j02 * j10 - j00 * j12) / det
                i20 = C0 / det
                i21 = (j01 * j20 - j00 * j21) / det
                i22 = (j00 * j11 - j01 * j10) / det
                for n in range(10):
                    d0 = DN[g, 0, n]
                    d1 = DN[g, 1, n]
                    d2 = DN[g, 2, n]
                    G[0, n] = i00 * d0 + i01 * d1 + i02 * d2
                    G[1, n] = i10 * d0 + i11 * d1 + i12 * d2
                    G[2, n] = i20 * d0 + i21 * d1 + i22 * d2
                w = W4 * abs(det)
                ax = G[0, la]
                ay = G[1, la]
                az = G[2, la]
                for j in range(10):
                    s = slot[j]
                    if s < 0:
                        continue
                    bx = G[0, j]
                    by = G[1, j]
                    bz = G[2, j]
                    for i in range(6):
                        CB[i, 0] = C[i, 0] * bx + C[i, 3] * by + C[i, 4] * bz
                        CB[i, 1] = C[i, 1] * by + C[i, 3] * bx + C[i, 5] * bz
                        CB[i, 2] = C[i, 2] * bz + C[i, 4] * bx + C[i, 5] * by
                    for c in range(3):
                        data[s, 0, c] += w * (ax * CB[0, c] + ay * CB[3, c] + az * CB[4, c])
                        data[s, 1, c] += w * (ay * CB[1, c] + ax * CB[3, c] + az * CB[5, c])
                        data[s, 2, c] += w * (az * CB[2, c] + ax * CB[4, c] + ay * CB[5, c])


# ---------------------------------------------------------------------------------------------- cylindrically orthotropic material
# The matrices of such a material are rotated at every Gauss point (the axes of the material are radial / tangential / axial about an axis through the
# point `ax[m, :3]` with the unit direction `ax[m, 3:]`) – the same as ccx evaluates *ORIENTATION, SYSTEM=CYLINDRICAL: C = Tᵀ·C_local·T with the transformation
# of the strain (engineering shear, the order xx yy zz xy xz yz).
@nb.njit(cache=True)
def cyl_axes(axrow, x, y, z, R):
    """R (3, 3) = the rows e_r, e_θ = e_z × e_r, e_z of the cylindrical system at the point (x, y, z)."""
    dx = x - axrow[0]
    dy = y - axrow[1]
    dz = z - axrow[2]
    az0 = axrow[3]
    az1 = axrow[4]
    az2 = axrow[5]
    t = dx * az0 + dy * az1 + dz * az2
    rx = dx - t * az0
    ry = dy - t * az1
    rz = dz - t * az2
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    if r > 0.0:
        rx /= r
        ry /= r
        rz /= r
    else:                                        # on the axis: any perpendicular direction (the caller rejects nodes on the axis)
        rx, ry, rz = az1, -az0, 0.0
        if abs(az0) + abs(az1) < 1e-12:
            rx, ry, rz = 1.0, 0.0, 0.0
    R[0, 0] = rx
    R[0, 1] = ry
    R[0, 2] = rz
    R[1, 0] = az1 * rz - az2 * ry
    R[1, 1] = az2 * rx - az0 * rz
    R[1, 2] = az0 * ry - az1 * rx
    R[2, 0] = az0
    R[2, 1] = az1
    R[2, 2] = az2


@nb.njit(cache=True)
def strain_transform(R, T):
    """T (6, 6): the strain (xx yy zz xy xz yz, engineering shear) in the axes R (rows) from the strain in the global axes."""
    for k in range(6):
        if k < 3:
            p = k
            q = k
        elif k == 3:
            p = 0
            q = 1
        elif k == 4:
            p = 0
            q = 2
        else:
            p = 1
            q = 2
        a0 = R[p, 0]
        a1 = R[p, 1]
        a2 = R[p, 2]
        b0 = R[q, 0]
        b1 = R[q, 1]
        b2 = R[q, 2]
        if p == q:
            T[k, 0] = a0 * a0
            T[k, 1] = a1 * a1
            T[k, 2] = a2 * a2
            T[k, 3] = a0 * a1
            T[k, 4] = a0 * a2
            T[k, 5] = a1 * a2
        else:
            T[k, 0] = 2.0 * a0 * b0
            T[k, 1] = 2.0 * a1 * b1
            T[k, 2] = 2.0 * a2 * b2
            T[k, 3] = a0 * b1 + a1 * b0
            T[k, 4] = a0 * b2 + a2 * b0
            T[k, 5] = a1 * b2 + a2 * b1


@nb.njit(cache=True)
def rotate_C(Cl, T, work, Cg):
    """Cg = Tᵀ·Cl·T (work, Cg: (6, 6) buffers)."""
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += Cl[i, k] * T[k, j]
            work[i, j] = s
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += T[k, i] * work[k, j]
            Cg[i, j] = s


@nb.njit(parallel=True, cache=True)
def stiffness_cyl(nodes, tets, mat_of, Cm, kind, ax, ptr, el, loc, indptr, indices, DN, N, upper, data):
    """`stiffness` for a mesh with a cylindrically orthotropic material: kind[m] == 1 → Cm[m] is the matrix in the local axes of the cylindrical system
    ax[m] = (point on the axis, unit direction) and is rotated at every Gauss point; kind[m] == 0 → Cm[m] is in the global axes (as in `stiffness`)."""
    nn = nodes.shape[0]
    for a in nb.prange(nn):
        r0 = indptr[a]
        r1 = indptr[a + 1]
        X = np.empty((10, 3))
        G = np.empty((3, 10))
        CB = np.empty((6, 3))
        slot = np.empty(10, np.int64)
        Cg = np.empty((6, 6))
        T = np.empty((6, 6))
        work = np.empty((6, 6))
        R = np.empty((3, 3))
        for p in range(ptr[a], ptr[a + 1]):
            e = el[p]
            la = loc[p]
            mi = mat_of[e]
            C = Cm[mi]
            for j in range(10):
                b = tets[e, j]
                X[j, 0] = nodes[b, 0]
                X[j, 1] = nodes[b, 1]
                X[j, 2] = nodes[b, 2]
                if upper and b < a:
                    slot[j] = -1
                    continue
                lo = r0
                hi = r1
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if indices[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid
                slot[j] = lo
            for g in range(4):
                j00 = j01 = j02 = j10 = j11 = j12 = j20 = j21 = j22 = 0.0
                xg = yg = zg = 0.0
                for n in range(10):
                    d0 = DN[g, 0, n]
                    d1 = DN[g, 1, n]
                    d2 = DN[g, 2, n]
                    x = X[n, 0]
                    y = X[n, 1]
                    z = X[n, 2]
                    j00 += d0 * x
                    j01 += d0 * y
                    j02 += d0 * z
                    j10 += d1 * x
                    j11 += d1 * y
                    j12 += d1 * z
                    j20 += d2 * x
                    j21 += d2 * y
                    j22 += d2 * z
                    xg += N[g, n] * x
                    yg += N[g, n] * y
                    zg += N[g, n] * z
                if kind[mi] == 1:
                    cyl_axes(ax[mi], xg, yg, zg, R)
                    strain_transform(R, T)
                    rotate_C(C, T, work, Cg)
                else:
                    for i in range(6):
                        for j in range(6):
                            Cg[i, j] = C[i, j]
                A0 = j11 * j22 - j12 * j21
                B0 = j12 * j20 - j10 * j22
                C0 = j10 * j21 - j11 * j20
                det = j00 * A0 + j01 * B0 + j02 * C0
                i00 = A0 / det
                i01 = (j02 * j21 - j01 * j22) / det
                i02 = (j01 * j12 - j02 * j11) / det
                i10 = B0 / det
                i11 = (j00 * j22 - j02 * j20) / det
                i12 = (j02 * j10 - j00 * j12) / det
                i20 = C0 / det
                i21 = (j01 * j20 - j00 * j21) / det
                i22 = (j00 * j11 - j01 * j10) / det
                for n in range(10):
                    d0 = DN[g, 0, n]
                    d1 = DN[g, 1, n]
                    d2 = DN[g, 2, n]
                    G[0, n] = i00 * d0 + i01 * d1 + i02 * d2
                    G[1, n] = i10 * d0 + i11 * d1 + i12 * d2
                    G[2, n] = i20 * d0 + i21 * d1 + i22 * d2
                w = W4 * abs(det)
                ax_ = G[0, la]
                ay = G[1, la]
                az = G[2, la]
                for j in range(10):
                    s = slot[j]
                    if s < 0:
                        continue
                    bx = G[0, j]
                    by = G[1, j]
                    bz = G[2, j]
                    for i in range(6):
                        CB[i, 0] = Cg[i, 0] * bx + Cg[i, 3] * by + Cg[i, 4] * bz
                        CB[i, 1] = Cg[i, 1] * by + Cg[i, 3] * bx + Cg[i, 5] * bz
                        CB[i, 2] = Cg[i, 2] * bz + Cg[i, 4] * bx + Cg[i, 5] * by
                    for c in range(3):
                        data[s, 0, c] += w * (ax_ * CB[0, c] + ay * CB[3, c] + az * CB[4, c])
                        data[s, 1, c] += w * (ay * CB[1, c] + ax_ * CB[3, c] + az * CB[5, c])
                        data[s, 2, c] += w * (az * CB[2, c] + ax_ * CB[4, c] + ay * CB[5, c])


# ---------------------------------------------------------------------------------------------- consistent mass matrix
# The same 14-point rule (degree 5, positive weights) as `fea_core.dynamics.mass.mass_times` – the consistent mass of the C3D10 element that ccx writes
# for *FREQUENCY (verified: φᵀMφ = 1 of the modes of ccx to all digits). Its weights are defined once there; the kernels get the arrays.
def _tet14():
    from ..dynamics.mass import TET14_PTS, TET14_W
    return np.ascontiguousarray(TET14_PTS), np.ascontiguousarray(TET14_W)


_P14, W14 = _tet14()
DN14 = np.ascontiguousarray(np.stack([tet10_dN(*p) for p in _P14]))          # (14, 3, 10)
N14 = np.ascontiguousarray(np.stack([tet10_N(*p) for p in _P14]))            # (14, 10)


@nb.njit(parallel=True, cache=True)
def mass_weights(nodes, tets, DN, wq, out):
    """out (E, 14) = w_q·|det J| at the 14 points of every element (the same arithmetic as `dynamics.mass._qp_data`, which takes |det| too)."""
    ne = tets.shape[0]
    nq = DN.shape[0]
    for e in nb.prange(ne):
        for g in range(nq):
            j00 = j01 = j02 = j10 = j11 = j12 = j20 = j21 = j22 = 0.0
            for n in range(10):
                b = tets[e, n]
                d0 = DN[g, 0, n]
                d1 = DN[g, 1, n]
                d2 = DN[g, 2, n]
                x = nodes[b, 0]
                y = nodes[b, 1]
                z = nodes[b, 2]
                j00 += d0 * x
                j01 += d0 * y
                j02 += d0 * z
                j10 += d1 * x
                j11 += d1 * y
                j12 += d1 * z
                j20 += d2 * x
                j21 += d2 * y
                j22 += d2 * z
            det = j00 * (j11 * j22 - j12 * j21) + j01 * (j12 * j20 - j10 * j22) + j02 * (j10 * j21 - j11 * j20)
            out[e, g] = wq[g] * abs(det)


@nb.njit(parallel=True, cache=True)
def mass_matrix(tets, rho, wd, N, ptr, el, loc, indptr, indices, upper, data):
    """data (nnzb, 3, 3) += the blocks of the consistent mass matrix (m_ab·I₃ per node pair): m_ab = Σ_e ρ_e Σ_q wd[e, q]·N_a(q)·N_b(q). Rows are filled node by
    node, every thread writes only its own block rows, the elements of a node in ascending order → bitwise identical for any number of threads."""
    nn = ptr.shape[0] - 1
    nq = N.shape[0]
    for a in nb.prange(nn):
        r0 = indptr[a]
        r1 = indptr[a + 1]
        for p in range(ptr[a], ptr[a + 1]):
            e = el[p]
            la = loc[p]
            for j in range(10):
                b = tets[e, j]
                if upper and b < a:
                    continue
                lo = r0
                hi = r1
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if indices[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid
                s = 0.0
                for q in range(nq):
                    s += wd[e, q] * N[q, la] * N[q, j]
                s *= rho[e]
                data[lo, 0, 0] += s
                data[lo, 1, 1] += s
                data[lo, 2, 2] += s


@nb.njit(parallel=True, cache=True)
def block_matmat(nn, indptr, indices, data, M, upper, nparts):
    """Y = K·M for the matrix in node blocks (M (3·nn, k)); upper = only the upper triangle is stored: K = U + Uᵀ − (the diagonal blocks).
    The product with U is parallel over the rows, the product with the transposed off-diagonal part is parallel over RANGES OF COLUMNS (every part
    writes only its own entries of Y; the blocks of a row are sorted, so a part finds its first block by a binary search) – no conflicts."""
    k = M.shape[1]
    Y = np.zeros((3 * nn, k))
    for a in nb.prange(nn):
        for p in range(indptr[a], indptr[a + 1]):
            b = indices[p]
            for i in range(3):
                for c in range(3):
                    v = data[p, i, c]
                    for q in range(k):
                        Y[3 * a + i, q] += v * M[3 * b + c, q]
    if upper:
        bounds = np.empty(nparts + 1, np.int64)
        for t in range(nparts + 1):
            bounds[t] = (nn * t) // nparts
        for t in nb.prange(nparts):
            lo = bounds[t]
            hi = bounds[t + 1]
            for a in range(nn):
                start = max(lo, a + 1)                # the diagonal block is already complete in the product with U
                s0 = indptr[a]
                s1 = indptr[a + 1]
                while s0 < s1:
                    mid = (s0 + s1) >> 1
                    if indices[mid] < start:
                        s0 = mid + 1
                    else:
                        s1 = mid
                p = s0
                while p < indptr[a + 1] and indices[p] < hi:
                    b = indices[p]
                    for i in range(3):
                        for c in range(3):
                            v = data[p, i, c]
                            for q in range(k):
                                Y[3 * b + c, q] += v * M[3 * a + i, q]
                    p += 1
    return Y


@nb.njit(parallel=True, cache=True)
def diag_positions(nn, indptr, indices):
    """Position of the diagonal block of every block row (binary search; −1 = the row has no diagonal block)."""
    pos = np.empty(nn, np.int64)
    for a in nb.prange(nn):
        lo = indptr[a]
        hi = indptr[a + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            if indices[mid] < a:
                lo = mid + 1
            else:
                hi = mid
        pos[a] = lo if (lo < indptr[a + 1] and indices[lo] == a) else -1
    return pos


# ---------------------------------------------------------------------------------------------- thermal load
@nb.njit(parallel=True, cache=True)
def thermal_forces(nodes, tets, mat_of, Cm, alpha_e, dT, DN, N, fe):
    """fe (E, 10, 3) = ∫ Bᵀ·C·ε_th dV of every element (the nodal forces of a thermal load), ε_th = α·ΔT·(1, 1, 1, 0, 0, 0), ΔT at the Gauss point
    by the shape functions of the nodal ΔT (dT (N,)). Written per element; the scatter to the nodes is done by the caller (`np.bincount`, deterministic).
    """
    ne = tets.shape[0]
    for e in nb.prange(ne):
        al = alpha_e[e]
        X = np.empty((10, 3))
        for k in range(10):
            for c in range(3):
                fe[e, k, c] = 0.0
        if al == 0.0:
            continue
        C = Cm[mat_of[e]]
        for j in range(10):
            b = tets[e, j]
            X[j, 0] = nodes[b, 0]
            X[j, 1] = nodes[b, 1]
            X[j, 2] = nodes[b, 2]
        for g in range(4):
            j00 = j01 = j02 = j10 = j11 = j12 = j20 = j21 = j22 = 0.0
            dTg = 0.0
            for n in range(10):
                d0 = DN[g, 0, n]
                d1 = DN[g, 1, n]
                d2 = DN[g, 2, n]
                x = X[n, 0]
                y = X[n, 1]
                z = X[n, 2]
                j00 += d0 * x
                j01 += d0 * y
                j02 += d0 * z
                j10 += d1 * x
                j11 += d1 * y
                j12 += d1 * z
                j20 += d2 * x
                j21 += d2 * y
                j22 += d2 * z
                dTg += N[g, n] * dT[tets[e, n]]
            A0 = j11 * j22 - j12 * j21
            B0 = j12 * j20 - j10 * j22
            C0 = j10 * j21 - j11 * j20
            det = j00 * A0 + j01 * B0 + j02 * C0
            i00 = A0 / det
            i01 = (j02 * j21 - j01 * j22) / det
            i02 = (j01 * j12 - j02 * j11) / det
            i10 = B0 / det
            i11 = (j00 * j22 - j02 * j20) / det
            i12 = (j02 * j10 - j00 * j12) / det
            i20 = C0 / det
            i21 = (j01 * j20 - j00 * j21) / det
            i22 = (j00 * j11 - j01 * j10) / det
            et = al * dTg
            sxx = (C[0, 0] + C[0, 1] + C[0, 2]) * et
            syy = (C[1, 0] + C[1, 1] + C[1, 2]) * et
            szz = (C[2, 0] + C[2, 1] + C[2, 2]) * et
            sxy = (C[3, 0] + C[3, 1] + C[3, 2]) * et
            sxz = (C[4, 0] + C[4, 1] + C[4, 2]) * et
            syz = (C[5, 0] + C[5, 1] + C[5, 2]) * et
            w = W4 * abs(det)
            for n in range(10):
                d0 = DN[g, 0, n]
                d1 = DN[g, 1, n]
                d2 = DN[g, 2, n]
                gx = i00 * d0 + i01 * d1 + i02 * d2
                gy = i10 * d0 + i11 * d1 + i12 * d2
                gz = i20 * d0 + i21 * d1 + i22 * d2
                fe[e, n, 0] += w * (gx * sxx + gy * sxy + gz * sxz)
                fe[e, n, 1] += w * (gx * sxy + gy * syy + gz * syz)
                fe[e, n, 2] += w * (gx * sxz + gy * syz + gz * szz)


@nb.njit(parallel=True, cache=True)
def thermal_forces_aniso(nodes, tets, mat_of, Cm, kind, ax, Am, dT, DN, N, fe):
    """`thermal_forces` for a mesh with orthotropic materials (anisotropic expansion): Am[m] = the thermal strain per unit ΔT (xx yy zz xy xz yz,
    engineering shear) in the global axes (kind 0) or (α1, α2, α3, 0, 0, 0) in the axes of the cylindrical system (kind 1; Cm[m] is then local too)."""
    ne = tets.shape[0]
    for e in nb.prange(ne):
        X = np.empty((10, 3))
        Cg = np.empty((6, 6))
        T = np.empty((6, 6))
        work = np.empty((6, 6))
        R = np.empty((3, 3))
        for k in range(10):
            for c in range(3):
                fe[e, k, c] = 0.0
        mi = mat_of[e]
        if Am[mi, 0] == 0.0 and Am[mi, 1] == 0.0 and Am[mi, 2] == 0.0 and Am[mi, 3] == 0.0 and Am[mi, 4] == 0.0 and Am[mi, 5] == 0.0:
            continue
        C = Cm[mi]
        for j in range(10):
            b = tets[e, j]
            X[j, 0] = nodes[b, 0]
            X[j, 1] = nodes[b, 1]
            X[j, 2] = nodes[b, 2]
        for g in range(4):
            j00 = j01 = j02 = j10 = j11 = j12 = j20 = j21 = j22 = 0.0
            dTg = 0.0
            xg = yg = zg = 0.0
            for n in range(10):
                d0 = DN[g, 0, n]
                d1 = DN[g, 1, n]
                d2 = DN[g, 2, n]
                x = X[n, 0]
                y = X[n, 1]
                z = X[n, 2]
                j00 += d0 * x
                j01 += d0 * y
                j02 += d0 * z
                j10 += d1 * x
                j11 += d1 * y
                j12 += d1 * z
                j20 += d2 * x
                j21 += d2 * y
                j22 += d2 * z
                dTg += N[g, n] * dT[tets[e, n]]
                xg += N[g, n] * x
                yg += N[g, n] * y
                zg += N[g, n] * z
            e0 = Am[mi, 0]
            e1 = Am[mi, 1]
            e2 = Am[mi, 2]
            e3 = Am[mi, 3]
            e4 = Am[mi, 4]
            e5 = Am[mi, 5]
            if kind[mi] == 1:
                cyl_axes(ax[mi], xg, yg, zg, R)
                strain_transform(R, T)
                rotate_C(C, T, work, Cg)
                # the expansion tensor in the global axes: sum of alpha_i e_i e_i^T (engineering shear: twice the off-diagonal)
                a1 = Am[mi, 0]
                a2 = Am[mi, 1]
                a3 = Am[mi, 2]
                e0 = a1 * R[0, 0] * R[0, 0] + a2 * R[1, 0] * R[1, 0] + a3 * R[2, 0] * R[2, 0]
                e1 = a1 * R[0, 1] * R[0, 1] + a2 * R[1, 1] * R[1, 1] + a3 * R[2, 1] * R[2, 1]
                e2 = a1 * R[0, 2] * R[0, 2] + a2 * R[1, 2] * R[1, 2] + a3 * R[2, 2] * R[2, 2]
                e3 = 2.0 * (a1 * R[0, 0] * R[0, 1] + a2 * R[1, 0] * R[1, 1] + a3 * R[2, 0] * R[2, 1])
                e4 = 2.0 * (a1 * R[0, 0] * R[0, 2] + a2 * R[1, 0] * R[1, 2] + a3 * R[2, 0] * R[2, 2])
                e5 = 2.0 * (a1 * R[0, 1] * R[0, 2] + a2 * R[1, 1] * R[1, 2] + a3 * R[2, 1] * R[2, 2])
            else:
                for i in range(6):
                    for j in range(6):
                        Cg[i, j] = C[i, j]
            e0 *= dTg
            e1 *= dTg
            e2 *= dTg
            e3 *= dTg
            e4 *= dTg
            e5 *= dTg
            A0 = j11 * j22 - j12 * j21
            B0 = j12 * j20 - j10 * j22
            C0 = j10 * j21 - j11 * j20
            det = j00 * A0 + j01 * B0 + j02 * C0
            i00 = A0 / det
            i01 = (j02 * j21 - j01 * j22) / det
            i02 = (j01 * j12 - j02 * j11) / det
            i10 = B0 / det
            i11 = (j00 * j22 - j02 * j20) / det
            i12 = (j02 * j10 - j00 * j12) / det
            i20 = C0 / det
            i21 = (j01 * j20 - j00 * j21) / det
            i22 = (j00 * j11 - j01 * j10) / det
            sxx = Cg[0, 0] * e0 + Cg[0, 1] * e1 + Cg[0, 2] * e2 + Cg[0, 3] * e3 + Cg[0, 4] * e4 + Cg[0, 5] * e5
            syy = Cg[1, 0] * e0 + Cg[1, 1] * e1 + Cg[1, 2] * e2 + Cg[1, 3] * e3 + Cg[1, 4] * e4 + Cg[1, 5] * e5
            szz = Cg[2, 0] * e0 + Cg[2, 1] * e1 + Cg[2, 2] * e2 + Cg[2, 3] * e3 + Cg[2, 4] * e4 + Cg[2, 5] * e5
            sxy = Cg[3, 0] * e0 + Cg[3, 1] * e1 + Cg[3, 2] * e2 + Cg[3, 3] * e3 + Cg[3, 4] * e4 + Cg[3, 5] * e5
            sxz = Cg[4, 0] * e0 + Cg[4, 1] * e1 + Cg[4, 2] * e2 + Cg[4, 3] * e3 + Cg[4, 4] * e4 + Cg[4, 5] * e5
            syz = Cg[5, 0] * e0 + Cg[5, 1] * e1 + Cg[5, 2] * e2 + Cg[5, 3] * e3 + Cg[5, 4] * e4 + Cg[5, 5] * e5
            w = W4 * abs(det)
            for n in range(10):
                d0 = DN[g, 0, n]
                d1 = DN[g, 1, n]
                d2 = DN[g, 2, n]
                gx = i00 * d0 + i01 * d1 + i02 * d2
                gy = i10 * d0 + i11 * d1 + i12 * d2
                gz = i20 * d0 + i21 * d1 + i22 * d2
                fe[e, n, 0] += w * (gx * sxx + gy * sxy + gz * sxz)
                fe[e, n, 1] += w * (gx * sxy + gy * syy + gz * syz)
                fe[e, n, 2] += w * (gx * sxz + gy * syz + gz * szz)


# ---------------------------------------------------------------------------------------------- corrections of the constraints
@nb.njit(cache=True)
def add_scalar_entries(indptr, indices, data, rows, cols, vals, upper):
    """data[block (r // 3, c // 3)][r % 3, c % 3] += v for the scalar entries (rows, cols, vals); serial (a fixed order of the sums). The entries are
    those of a SYMMETRIC matrix (both triangles): for the upper triangle the entries below the block diagonal are skipped. → −1, or the index of the
    first entry whose block is not in the pattern (the pattern was built without it)."""
    for k in range(len(rows)):
        r = rows[k]
        c = cols[k]
        a = r // 3
        b = c // 3
        if upper and a > b:
            continue
        lo = indptr[a]
        hi = indptr[a + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            if indices[mid] < b:
                lo = mid + 1
            else:
                hi = mid
        if lo >= indptr[a + 1] or indices[lo] != b:
            return k
        data[lo, r - 3 * a, c - 3 * b] += vals[k]
    return -1


# ---------------------------------------------------------------------------------------------- couplings made by the constraints
@nb.njit(cache=True)
def node_neighbours(nodes_list, tets, ptr, el):
    """CSR (nptr, ncol) of the nodes that share an element with every node of `nodes_list` (the node itself included), sorted."""
    k = len(nodes_list)
    nptr = np.zeros(k + 1, np.int64)
    for pass_ in range(2):
        out = np.empty(0, np.int32)
        if pass_ == 1:
            out = np.empty(nptr[k], np.int32)
        for i in range(k):
            a = nodes_list[i]
            m = (ptr[a + 1] - ptr[a]) * 10
            buf = np.empty(m, np.int32)
            q = 0
            for p in range(ptr[a], ptr[a + 1]):
                e = el[p]
                for j in range(10):
                    buf[q] = tets[e, j]
                    q += 1
            s = np.sort(buf[:q])
            c = 0
            for t in range(q):
                if t == 0 or s[t] != s[t - 1]:
                    if pass_ == 1:
                        out[nptr[i] + c] = s[t]
                    c += 1
            if pass_ == 0:
                nptr[i + 1] = nptr[i] + c
        if pass_ == 1:
            return nptr, out
    return nptr, np.empty(0, np.int32)


@nb.njit(cache=True)
def coupling_pairs(dep_nodes, nbr_ptr, nbr_col, m_ptr, m_col, dep_pos):
    """The block couplings that eliminating the dependent nodes creates: for a dependent node s with the neighbours N(s) and the masters M(s)
    (dep_pos[node] = the index of a dependent node in `dep_nodes`, −1 for the others):
        (a, m)   a in N(s), m in M(s)            – the neighbours of s now couple to its masters
        (m, m')  m in M(s), m' in M(t)           – for every dependent t that is a neighbour of s or s itself
    → (rows, cols) with duplicates (the pattern merges them)."""
    k = len(dep_nodes)
    total = 0
    for i in range(k):
        mi = m_ptr[i + 1] - m_ptr[i]
        total += (nbr_ptr[i + 1] - nbr_ptr[i]) * mi
        for q in range(nbr_ptr[i], nbr_ptr[i + 1]):
            j = dep_pos[nbr_col[q]]
            if j >= 0:
                total += mi * (m_ptr[j + 1] - m_ptr[j])
    pa = np.empty(total, np.int64)
    pb = np.empty(total, np.int64)
    w = 0
    for i in range(k):
        for q in range(nbr_ptr[i], nbr_ptr[i + 1]):
            a = nbr_col[q]
            for u in range(m_ptr[i], m_ptr[i + 1]):
                pa[w] = a
                pb[w] = m_col[u]
                w += 1
            j = dep_pos[a]
            if j >= 0:
                for u in range(m_ptr[i], m_ptr[i + 1]):
                    for v in range(m_ptr[j], m_ptr[j + 1]):
                        pa[w] = m_col[u]
                        pb[w] = m_col[v]
                        w += 1
    return pa, pb


# ---------------------------------------------------------------------------------------------- scalar CSR of the equation system
@nb.njit(parallel=True, cache=True)
def scalar_counts(nn, indptr, indices, fixed, upper, prune):
    """Number of entries of every scalar row (3 per node) of the system with the constraints eliminated; upper = only the columns ≥ the row.
    prune: a constrained row keeps only its diagonal and a free row loses the columns of constrained DOFs; otherwise the pattern of the matrix stays
    (the couplings of the constrained DOFs become explicit zeros – the convention of the earlier NGSolve worker)."""
    counts = np.zeros(3 * nn, np.int64)
    for a in nb.prange(nn):
        r0 = indptr[a]
        r1 = indptr[a + 1]
        for i in range(3):
            r = 3 * a + i
            if fixed[r] and (prune or r1 == r0):         # a node without elements has no pattern: only its diagonal
                counts[r] = 1
                continue
            c = 0
            for k in range(r0, r1):
                b = indices[k]
                for cc in range(3):
                    col = 3 * b + cc
                    if upper and col < r:
                        continue
                    if prune and fixed[col]:
                        continue
                    c += 1
            counts[r] = c
    return counts


@nb.njit(parallel=True, cache=True)
def scalar_fill(a0, a1, indptr, indices, data, fixed, upper, prune, d, rowptr, base, sidx, sdat):
    """Fills the scalar CSR (int32 columns, float64 values) of the nodes a0 ≤ a < a1: the symmetric elimination of the constraints, the diagonal of a
    constrained DOF = d (the same convention as `hypre.apply_dirichlet`); the entries in the rows and columns of constrained DOFs are 0 (or removed, prune).
    sidx / sdat hold only this range: the row r starts at rowptr[r] − base (base = rowptr[3·a0]) – the matrix is converted in chunks of rows."""
    for a in nb.prange(a0, a1):
        r0 = indptr[a]
        r1 = indptr[a + 1]
        for i in range(3):
            r = 3 * a + i
            w = rowptr[r] - base
            if fixed[r] and (prune or r1 == r0):
                sidx[w] = r
                sdat[w] = d
                continue
            for k in range(r0, r1):
                b = indices[k]
                for cc in range(3):
                    col = 3 * b + cc
                    if upper and col < r:
                        continue
                    if fixed[col]:
                        if prune:
                            continue
                        sidx[w] = col
                        sdat[w] = d if col == r else 0.0
                        w += 1
                        continue
                    sidx[w] = col
                    sdat[w] = 0.0 if fixed[r] else data[k, i, cc]
                    w += 1

"""The equation system K·u = f of the linear path from the own assembly (no files, no process – the worker `fea_core.assembly.worker` reads the
job files and calls this; tests call it directly).

  assemble(nodes, tets, mat_of, Cm, fixed, g, F, upper=…, dT=…, alpha_e=…) → Assembled       (K in node blocks + everything around it)
  Assembled.to_system() → System       the whole scalar CSR in memory (tests, small models)
  Assembled.write(directory)           the scalar CSR streamed to the .npy files of the solver in chunks of rows (the worker)
  build_system(...)                    = assemble(...).to_system()

K is assembled in 3×3 node blocks (`kernels.stiffness`), the thermal load of every load state is added to the right-hand side (`kernels.thermal_forces`),
the constraints are eliminated symmetrically exactly as `hypre.apply_dirichlet` does it (the same convention as the earlier NGSolve worker): the rows
and columns of the constrained DOFs are zero, their diagonal is d = the mean of the diagonal of the free DOFs, the right-hand side of a constrained DOF
is d·g, the right-hand side of a free one is reduced by K·g (the prescribed displacements). The pattern of the matrix stays (the couplings of the
constrained DOFs are explicit zeros): removing them (`prune=True`) makes the factor 2–6 % smaller, but rslab factors the small models 1.7× slower
(60 k DOF: 0.75 against 0.45 s – the isolated unknowns split the elimination tree) and analyses the large ones 1.5× slower (658 k DOF 4.8 → 7.0 s; the
factorization is 15 % faster, in total 0.7 s of 31 s), so the default is the pattern of the earlier worker. The result is the scalar CSR of
the system: the upper triangle (the direct solver) or the whole matrix (hypre), columns sorted.

The input is checked like CalculiX checks it: every Gauss-point Jacobian determinant of every element has to be finite and positive (an inverted, flat or
damaged element is an `ElementError` that names the elements – the earlier code took |det| and accepted them, or produced NaN that the rigid-body check
did not see), the indices of the nodes, the coordinates and the material matrices have to be valid, and the matrix has to be finite.
The reduction can also carry beam elements on its extra nodes (the bolts: `Reduction.elem_dofs` / `elem_k`) – their stiffness is added to the extended matrix
like the correction of the elimination. The last `hom_cols` right-hand sides are HOMOGENEOUS (no prescribed displacements, no thermal load): the unit
states of the bolt preload (an initial elongation of a bolt) are solved on the same factorization as the load states.
Memory: the block matrix (72 B per block) is the only large array of the assembly – the scalar CSR (12 B per entry, 9 entries per block) is made and
written in chunks of rows (`Assembled.write`), so the worker needs about 2.5 × less memory than when it held both (the hypre path: the whole matrix).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import AssemblyError, ElementError          # noqa: F401  (AssemblyError is imported from here by the callers and the tests)
from . import kernels as K
from .reduction import Reduction, ReductionError          # noqa: F401
from .mass import MassInputs                              # noqa: F401  (the type of `assemble(mass=)`)

CHUNK_ELEMENTS = 300_000          # the thermal element forces are computed per chunk of elements (240 B per element)
CHUNK_ENTRIES = 1 << 22           # scalar entries per chunk of rows when the system is converted and written (12 B each = 50 MB)
LISTED_ELEMENTS = 20              # elements named in an ElementError (the number of all of them is `count`)


@dataclass
class System:
    indptr: np.ndarray            # (n + 1,) int64
    indices: np.ndarray           # (nnz,) int32
    data: np.ndarray              # (nnz,) float64
    rhs: np.ndarray               # (n, states) float64
    d: float                      # the diagonal of the constrained DOFs
    upper: bool
    info: dict = field(default_factory=dict)
    mdata: np.ndarray | None = None      # (nnz,) the mass matrix on the same pattern (the constrained rows and columns zero), None = no mass requested
    MR: np.ndarray | None = None         # (3·nodes, 6) M·(the rigid-body motions) of the unreduced system, see `Assembled`


def _canon(nodes, tets, mat_of, Cm):
    nodes = np.ascontiguousarray(nodes, np.float64)
    tets = np.ascontiguousarray(tets, np.int32)
    mat_of = np.ascontiguousarray(mat_of, np.int32)
    Cm = np.ascontiguousarray(Cm, np.float64)
    if tets.ndim != 2 or tets.shape[1] != 10:
        raise AssemblyError("The own assembly is only available for tet10 (C3D10)")
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise AssemblyError("Invalid node coordinates of the assembly")
    if Cm.ndim != 3 or Cm.shape[1:] != (6, 6) or len(mat_of) != len(tets) or (len(mat_of) and (mat_of.min() < 0 or mat_of.max() >= len(Cm))):
        raise AssemblyError("Invalid material table of the assembly")
    if len(tets) and (tets.min() < 0 or tets.max() >= len(nodes)):         # the kernels do not check the indices (an index outside = a crash)
        raise AssemblyError("An element refers to a node that does not exist")
    if not np.isfinite(Cm).all():
        raise AssemblyError("The material matrices contain NaN or infinity")
    return nodes, tets, mat_of, Cm


def check_elements(nodes: np.ndarray, tets: np.ndarray) -> None:
    """Raises `ElementError` for the elements whose Jacobian determinant is not finite and positive at one of the 4 Gauss points – the rule of ccx
    ("*ERROR in e_c3d: nonpositive jacobian": an inverted element = a reversed node order, a flat one = zero volume; a sliver with a tiny positive
    determinant passes in both)."""
    jmin = K.min_jacobian(nodes, tets, K.DN4)
    bad = np.flatnonzero(~(jmin > 0.0))                    # NaN is "not > 0" too
    if len(bad):
        first = ", ".join(str(int(e) + 1) for e in bad[:8])
        raise ElementError(
            f"{len(bad):,} element(s) with a Jacobian determinant that is not positive and finite at an integration point (inverted, flat or damaged "
            f"elements – CalculiX rejects them as well); CalculiX element numbers of the first ones: {first}", bad[:LISTED_ELEMENTS], len(bad))


class BlockMatrix:
    """The stiffness matrix in node blocks (the whole matrix or its upper triangle): symmetric products and the diagonal."""

    def __init__(self, nn: int, indptr, indices, data, upper: bool):
        self.nn, self.indptr, self.indices, self.data, self.upper = nn, indptr, indices, data, upper
        self.dpos = K.diag_positions(nn, indptr, indices)

    def matmat(self, M: np.ndarray) -> np.ndarray:
        """K·M (M: (n,) or (n, k)); for the upper triangle U: K = U + Uᵀ − (the diagonal blocks) – parallel, without a transposed copy."""
        import numba
        M2 = np.ascontiguousarray(np.asarray(M, np.float64).reshape(3 * self.nn, -1))
        Y = K.block_matmat(self.nn, self.indptr, self.indices, self.data, M2, self.upper, 4 * numba.get_num_threads())
        return Y.reshape(np.shape(M)) if np.ndim(M) == 1 else Y

    def diagonal(self) -> np.ndarray:
        """The scalar diagonal (n,); 0 for a node without elements."""
        out = np.zeros((self.nn, 3))
        ok = self.dpos >= 0
        blk = self.data[self.dpos[ok]]
        out[ok] = np.stack([blk[:, 0, 0], blk[:, 1, 1], blk[:, 2, 2]], axis=1)
        return out.ravel()

    def max_row_blocks(self) -> int:
        """The largest number of blocks in a row of the WHOLE matrix (for the upper triangle: the row + the column of the node)."""
        cnt = np.diff(self.indptr)
        if self.upper:
            cnt = cnt + np.bincount(self.indices, minlength=self.nn) - (self.dpos >= 0)
        return int(cnt.max()) if len(cnt) else 0


def rigid_body_residual(K_: BlockMatrix, nodes: np.ndarray) -> float:
    """Relative residual of K·u for the three translations and rotations (must be zero for a free body):
    max |K·u| / (max |K| · max |x − x̄| · the largest number of entries in a row). inf for a matrix that cannot be judged: with NaN or infinity in it
    (the earlier version returned 0.0 for NaN, the comparison `scale > 0` being False), zero (no stiffness) or on nodes without any extent."""
    nn = len(nodes)
    X = nodes - nodes.mean(axis=0)
    modes = []
    for dd in range(3):
        u0 = np.zeros((K_.nn, 3))               # K_.nn > nn when the matrix has the extra nodes of the constraints (their rows are still empty here)
        u0[:nn, dd] = 1.0
        modes.append(u0.ravel())
    for i, j in ((0, 1), (1, 2), (2, 0)):
        u0 = np.zeros((K_.nn, 3))
        u0[:nn, i] = -X[:, j]
        u0[:nn, j] = X[:, i]
        modes.append(u0.ravel())
    res = float(np.abs(K_.matmat(np.column_stack(modes))).max())
    kmax = float(max(K_.data.max(), -K_.data.min()))                    # not np.abs(data).max(): that is a copy of the whole matrix (72 B per block)
    scale = kmax * float(np.abs(X).max()) * (3 * K_.max_row_blocks())
    if not (np.isfinite(res) and np.isfinite(scale) and scale > 0):
        return float("inf")
    return res / scale


def thermal_load(nodes, tets, mat_of, Cm, alpha_e, dT: np.ndarray) -> np.ndarray:
    """Nodal forces (N, 3) of the thermal load ΔT (N,): ∫ Bᵀ·C·α·ΔT·(1, 1, 1, 0, 0, 0) dV (4 Gauss points, summed in a fixed order)."""
    nn = len(nodes)
    dT = np.ascontiguousarray(dT, np.float64)
    alpha_e = np.ascontiguousarray(alpha_e, np.float64)
    f = np.zeros((nn, 3))
    for lo in range(0, len(tets), CHUNK_ELEMENTS):
        hi = min(len(tets), lo + CHUNK_ELEMENTS)
        fe = np.empty((hi - lo, 10, 3))
        K.thermal_forces(nodes, tets[lo:hi], mat_of[lo:hi], Cm, alpha_e[lo:hi], dT, K.DN4, K.N4, fe)
        flat = tets[lo:hi].ravel()
        for c in range(3):
            f[:, c] += np.bincount(flat, weights=fe[:, :, c].ravel(), minlength=nn)
    return f


def thermal_load_aniso(nodes, tets, mat_of, Cm, kind, ax, Am, dT: np.ndarray) -> np.ndarray:
    """`thermal_load` for orthotropic materials (the anisotropic expansion: `kernels.thermal_forces_aniso`)."""
    nn = len(nodes)
    dT = np.ascontiguousarray(dT, np.float64)
    f = np.zeros((nn, 3))
    for lo in range(0, len(tets), CHUNK_ELEMENTS):
        hi = min(len(tets), lo + CHUNK_ELEMENTS)
        fe = np.empty((hi - lo, 10, 3))
        K.thermal_forces_aniso(nodes, tets[lo:hi], mat_of[lo:hi], Cm, kind, ax, Am, dT, K.DN4, K.N4, fe)
        flat = tets[lo:hi].ravel()
        for c in range(3):
            f[:, c] += np.bincount(flat, weights=fe[:, :, c].ravel(), minlength=nn)
    return f


class _Npy:
    """A 1-D .npy file written in pieces (the same file as `np.save` writes, but the array does not have to exist in memory)."""

    def __init__(self, path: Path, dtype, n: int):
        self.path, self.n, self.written = path, int(n), 0
        self.fp = open(path, "wb")
        np.lib.format.write_array_header_1_0(self.fp, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": (self.n,)})

    def write(self, a: np.ndarray) -> None:
        self.fp.write(memoryview(np.ascontiguousarray(a)).cast("B"))
        self.written += len(a)

    def close(self, check: bool = True) -> None:
        self.fp.close()
        if check and self.written != self.n:
            raise AssemblyError(f"{self.path.name}: {self.written} entries written, {self.n} expected")


@dataclass
class Assembled:
    """The assembled system before the scalar CSR is made: K in node blocks (`indptr`, `indices`, `data`), the constraints (`fixed`, the diagonal `d`),
    the right-hand sides and the row pointer of the scalar matrix. The scalar rows are made by `fill` / `to_system` / `write`."""
    nn: int
    indptr: np.ndarray            # (nn + 1,) block rows
    indices: np.ndarray           # (nnzb,) block columns
    data: np.ndarray              # (nnzb, 3, 3)
    fixed: np.ndarray             # (3 nn,) bool
    d: float
    rhs: np.ndarray               # (3 nn, states) after the elimination of the constraints
    upper: bool
    prune: bool
    rowptr: np.ndarray            # (3 nn + 1,) int64 – rows of the scalar CSR
    info: dict
    mdata: np.ndarray | None = None      # (nnzb, 3, 3) the mass matrix on the pattern of K (after the elimination of the dependent DOFs), None = not requested
    MR: np.ndarray | None = None         # (3 nn, 6) M·R of the UNREDUCED system: R = the rigid-body motions (three translations, three rotations about the origin)
    #                                      with the constrained DOFs set to zero – Γ = φᵀ·M·R (participation factors) and r·M·r (the effective total mass) of the
    #                                      modal analysis, φ = the mode with the dependent DOFs expanded

    def node_ranges(self, max_entries: int = CHUNK_ENTRIES) -> list[tuple[int, int]]:
        """Ranges of nodes [a0, a1) whose scalar rows hold about `max_entries` entries (at least one node)."""
        cum = self.rowptr[::3]                                     # the entries before every node (nn + 1)
        out, a = [], 0
        while a < self.nn:
            b = int(np.searchsorted(cum, cum[a] + max_entries, side="right")) - 1
            b = min(max(b, a + 1), self.nn)
            out.append((a, b))
            a = b
        return out

    def fill(self, a0: int, a1: int) -> tuple[np.ndarray, np.ndarray]:
        """(columns int32, values float64) of the scalar rows of the nodes [a0, a1) (the elimination of the constraints included)."""
        base = int(self.rowptr[3 * a0])
        n = int(self.rowptr[3 * a1]) - base
        sidx = np.empty(n, np.int32)
        sdat = np.empty(n, np.float64)
        K.scalar_fill(a0, a1, self.indptr, self.indices, self.data, self.fixed, self.upper, self.prune, self.d, self.rowptr, base, sidx, sdat)
        return sidx, sdat

    def fill_mass(self, a0: int, a1: int) -> np.ndarray:
        """The values of the mass matrix on the scalar pattern of `fill(a0, a1)` (the constrained DOFs: zero rows and columns, zero diagonal)."""
        base = int(self.rowptr[3 * a0])
        n = int(self.rowptr[3 * a1]) - base
        sidx = np.empty(n, np.int32)
        sdat = np.empty(n, np.float64)
        K.scalar_fill(a0, a1, self.indptr, self.indices, self.mdata, self.fixed, self.upper, self.prune, 0.0, self.rowptr, base, sidx, sdat)
        return sdat

    def to_system(self) -> System:
        """The whole scalar CSR in memory."""
        tic = time.perf_counter()
        sidx, sdat = self.fill(0, self.nn)
        mdat = self.fill_mass(0, self.nn) if self.mdata is not None else None
        self.info["times"]["convert"] = self.info["times"].get("convert", 0.0) + time.perf_counter() - tic
        return System(self.rowptr, sidx, sdat, np.ascontiguousarray(self.rhs), self.d, self.upper, self.info, mdat, self.MR)

    def write(self, directory: Path, chunk_entries: int = CHUNK_ENTRIES) -> dict:
        """indptr.npy, indices.npy, data.npy, rhs.npy of the solver into `directory`; the scalar rows are made and written chunk by chunk.
        → {"convert": s, "write": s}."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        t_convert = t_write = 0.0
        tic = time.perf_counter()
        np.save(directory / "indptr.npy", self.rowptr)
        nnz = int(self.rowptr[-1])
        fi, fd = _Npy(directory / "indices.npy", np.int32, nnz), _Npy(directory / "data.npy", np.float64, nnz)
        fm = _Npy(directory / "mdata.npy", np.float64, nnz) if self.mdata is not None else None
        t_write += time.perf_counter() - tic
        try:
            for a0, a1 in self.node_ranges(chunk_entries):
                tic = time.perf_counter()
                sidx, sdat = self.fill(a0, a1)
                mdat = self.fill_mass(a0, a1) if fm is not None else None
                t_convert += time.perf_counter() - tic
                tic = time.perf_counter()
                fi.write(sidx)
                fd.write(sdat)
                if fm is not None:
                    fm.write(mdat)
                t_write += time.perf_counter() - tic
                del sidx, sdat, mdat
        except BaseException:                       # the first error is the one to report (not "entries written, expected")
            for f in (fi, fd, fm):
                if f is not None:
                    f.close(check=False)
            raise
        for f in (fi, fd, fm):
            if f is not None:
                f.close()
        tic = time.perf_counter()
        np.save(directory / "rhs.npy", np.ascontiguousarray(self.rhs))
        if self.MR is not None:
            np.save(directory / "MR.npy", np.ascontiguousarray(self.MR))
        t_write += time.perf_counter() - tic
        self.info["times"]["convert"] = self.info["times"].get("convert", 0.0) + t_convert
        return {"convert": t_convert, "write": t_write}


@dataclass
class Aniso:
    """Orthotropic materials of the assembly: the table of the matrices C (`Cm` of `assemble`) is in the global axes (kind 0) or in the local axes of a
    cylindrical system (kind 1, rotated at every Gauss point); A = the thermal strain per unit ΔT (xx yy zz xy xz yz, engineering shear) in the global
    axes (kind 0) or (α1, α2, α3, 0, 0, 0) in the local axes (kind 1)."""
    kind: np.ndarray              # (M,) int8
    ax: np.ndarray                # (M, 6) a point on the axis and its unit direction (kind 1)
    A: np.ndarray                 # (M, 6)

    @property
    def cylindrical(self) -> bool:
        return bool(np.any(self.kind == 1))


def _extra_couplings(red: Reduction | None, nne: int, tets, ptr, el, upper: bool, cliques=()):
    """The extra block pattern (`kernels.extra_csr`) that the elimination of the dependent DOFs needs, or None: the neighbours of a dependent node couple to
    its masters, the masters of neighbouring dependent nodes couple to each other, the extra nodes have their diagonal block. `cliques` = lists of nodes that
    all couple to each other (the dense block of a point mass with a distributing coupling, `mass.MassInputs.dist_nodes`)."""
    pa, pb = [], []
    for idx in cliques:
        idx = np.asarray(idx, np.int64)
        pa.append(np.repeat(idx, len(idx)))
        pb.append(np.tile(idx, len(idx)))
    if red is None:
        red = Reduction(nne)
    if red.n_extra:
        ex = np.arange(red.n_mesh, nne, dtype=np.int64)
        pa.append(ex)                               # the diagonal blocks of the extra nodes (the couplings between them come from the dependent nodes)
        pb.append(ex)
    if red.n_elem:                                  # the beam elements on the extra nodes: every node of an element couples to all of its nodes
        en = red.elem_dofs[:, ::3] // 3             # (m, 4) the nodes of the elements
        for i in range(en.shape[1]):
            for j in range(i, en.shape[1]):
                pa.append(en[:, i].astype(np.int64))
                pb.append(en[:, j].astype(np.int64))
    if red.n_dep:
        dep_nodes = np.unique(red.dep // 3)
        dep_pos = np.full(nne, -1, np.int64)
        dep_pos[dep_nodes] = np.arange(len(dep_nodes))
        nbr_ptr, nbr_col = K.node_neighbours(dep_nodes.astype(np.int64), tets, ptr, el)
        # the masters (node numbers) of every dependent node: the columns of the rows of its DOFs, unique
        rows = np.repeat(np.arange(red.n_dep), np.diff(red.ptr))
        key = np.unique(dep_pos[red.dep[rows] // 3] * np.int64(nne) + red.col // 3)
        m_node = key // nne
        m_col = (key % nne).astype(np.int64)
        m_ptr = np.zeros(len(dep_nodes) + 1, np.int64)
        np.cumsum(np.bincount(m_node, minlength=len(dep_nodes)), out=m_ptr[1:])
        a, b = K.coupling_pairs(dep_nodes.astype(np.int64), nbr_ptr, nbr_col, m_ptr, m_col, dep_pos)
        pa.append(a)
        pb.append(b)
    if not pa:
        return None
    return K.extra_csr(nne, np.concatenate(pa), np.concatenate(pb), upper)


def _dependent_rows(nn: int, indptr, indices, data, upper: bool, dep: np.ndarray):
    """The scalar rows of the dependent DOFs of the (symmetric) matrix in node blocks: a scipy CSR matrix (m, 3 nn). For the upper triangle the entries of
    a dependent column in the rows above are the transposed blocks."""
    import scipy.sparse as sp
    dnode = np.zeros(nn, bool)
    dnode[dep // 3] = True
    rows_b = np.repeat(np.arange(nn, dtype=np.int64), np.diff(indptr))
    s1 = dnode[rows_b]
    r_list, c_list, b_list = [rows_b[s1]], [indices[s1].astype(np.int64)], [data[s1]]
    if upper:
        s2 = dnode[indices] & (rows_b != indices)
        r_list.append(indices[s2].astype(np.int64))
        c_list.append(rows_b[s2])
        b_list.append(data[s2].transpose(0, 2, 1))
    del rows_b
    rn, cn, B = np.concatenate(r_list), np.concatenate(c_list), np.concatenate(b_list)
    dmap = np.full(3 * nn, -1, np.int64)
    dmap[dep] = np.arange(len(dep))
    rr, cc, vv = [], [], []
    for i in range(3):
        pos = dmap[3 * rn + i]
        keep = pos >= 0
        for j in range(3):
            rr.append(pos[keep])
            cc.append(3 * cn[keep] + j)
            vv.append(B[keep, i, j])
    return sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(len(dep), 3 * nn)).tocsr()


def _eliminate_dependent(Kb: BlockMatrix, red: Reduction, A, fixed: np.ndarray, upper: bool):
    """Corr = K_ID·A + (K_ID·A)ᵀ + Aᵀ·K_DD·A as scalar COO entries (rows, cols, vals): what the elimination u_D = A·u_I adds to the matrix of the
    independent DOFs (I = not dependent). K_ID, K_DD = the entries of the dependent rows / columns of K."""
    import scipy.sparse as sp
    n = 3 * Kb.nn
    Kd = _dependent_rows(Kb.nn, Kb.indptr, Kb.indices, Kb.data, upper, red.dep)
    isdep = np.zeros(n, bool)
    isdep[red.dep] = True
    keep_cols = sp.diags((~isdep).astype(float))
    Kdi = (Kd @ keep_cols).tocsr()                   # (m, n): the columns of the independent DOFs
    Kdd = Kd[:, red.dep].tocsr()                     # (m, m)
    C1 = (Kdi.T @ A).tocsr()                         # (n, n): K_ID·A
    C2 = (A.T @ (Kdd @ A)).tocsr()
    corr = (C1 + C1.T + C2).tocoo()
    return corr.row.astype(np.int64), corr.col.astype(np.int64), corr.data


def _element_entries(red: Reduction):
    """The matrices of the beam elements on the extra nodes as scalar COO entries (rows, cols, vals) of the extended matrix (both triangles)."""
    d = red.elem_dofs
    rows = np.broadcast_to(d[:, :, None], red.elem_k.shape).ravel()
    cols = np.broadcast_to(d[:, None, :], red.elem_k.shape).ravel()
    return np.ascontiguousarray(rows), np.ascontiguousarray(cols), np.ascontiguousarray(red.elem_k.ravel())


def release_held_directions(R: np.ndarray, red: Reduction | None) -> np.ndarray:
    """R with the directions held by the single-node constraint equations (the cylindrical supports: u_dep = Σ c·u_master of the DOFs of ONE node) projected out at every
    such node: R_node ← (I − P)·R_node, P = the orthogonal projector on the span of the equations a = e_dep − Σ c·e_master. These directions are not free motions of the
    structure, so the effective total mass `rᵀ·M·r` (the denominator of the mass fractions) leaves the mass in them out, as ccx does with its local boundary DOFs.
    Equations that couple nodes (ties, extra nodes) are not touched."""
    if red is None or not red.n_dep:
        return R
    nd = red.n_dep
    owner = red.dep // 3
    rows = np.repeat(np.arange(nd), np.diff(red.ptr))
    foreign = np.zeros(nd, bool)
    np.logical_or.at(foreign, rows, (red.col // 3) != owner[rows])
    nodes = np.unique(owner[~foreign])
    if not len(nodes):
        return R
    slot = {int(n): i for i, n in enumerate(nodes)}
    E = np.zeros((len(nodes), 3, 3))
    count = np.zeros(len(nodes), int)
    for i in np.flatnonzero(~foreign):
        n = int(owner[i])
        k = slot[n]
        r = count[k]
        count[k] += 1
        E[k, r, int(red.dep[i]) % 3] += 1.0
        for j in range(int(red.ptr[i]), int(red.ptr[i + 1])):
            E[k, r, int(red.col[j]) % 3] -= float(red.val[j])
    _u, sv, vt = np.linalg.svd(E)                                   # rows beyond the rank are zero: their singular values are
    keep = sv > 1e-9 * np.maximum(sv[:, :1], 1e-300)
    P = np.einsum("nki,nk,nkj->nij", vt, keep.astype(float), vt)    # (n, 3, 3)
    out = R.copy()
    dofs = 3 * nodes[:, None] + np.arange(3)                         # (n, 3)
    out[dofs] = R[dofs] - np.einsum("nij,njc->nic", P, R[dofs])
    return out


def rigid_motions(nodes: np.ndarray, extra_xyz: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """R (3·(N + extra), 6): the three translations and the three rotations about the origin of the whole system (the mesh nodes u = e_d, u = e_d × x; of a
    pair of extra nodes the displacement node moves like the point and the rotation node turns by e_d); the constrained DOFs (`fixed`: *BOUNDARY-type fixations) are
    zero. The directions held by constraint EQUATIONS (cylindrical supports) are NOT removed here - the participation factors use the motion of the whole body; the effective
    total mass (`total_free`) is measured with `release_held_directions(R, ...)`, which removes them (as the local boundary DOFs of ccx)."""
    nn, nx = len(nodes), len(extra_xyz)
    R = np.zeros((3 * (nn + nx), 6))
    ids = np.arange(nn)
    ex_disp = nn + np.arange(0, nx, 2)
    ex_rot = nn + np.arange(1, nx, 2)
    for d in range(3):
        R[3 * ids + d, d] = 1.0
        R[3 * ex_disp + d, d] = 1.0
        e = np.zeros(3)
        e[d] = 1.0
        cm = np.cross(e, nodes)
        R[:3 * nn, 3 + d] = cm.ravel()
        if nx:
            cp = np.cross(e, extra_xyz[0::2])
            for c in range(3):
                R[3 * ex_disp + c, 3 + d] = cp[:, c]
            R[3 * ex_rot + d, 3 + d] = 1.0
    R[fixed] = 0.0
    return R


def _add_dense_mass(Mb: BlockMatrix, idx: np.ndarray, A3: np.ndarray, m: float, upper: bool, chunk_entries: int = CHUNK_ENTRIES) -> None:
    """M += m·BᵀB for a point mass whose reference displacement is u_ref = B·u of the nodes `idx` (B = A3 (3, n, 3) as (3, 3n)); the dense block, in chunks of
    rows, through the scalar entries of the block matrix (the pattern holds the clique of the nodes, see `_extra_couplings`)."""
    n = len(idx)
    B = np.asarray(A3, float).reshape(3, 3 * n)
    dofs = (3 * np.asarray(idx, np.int64)[:, None] + np.arange(3)).ravel()
    step = max(1, chunk_entries // (3 * n))
    for lo in range(0, 3 * n, step):
        hi = min(3 * n, lo + step)
        G = m * (B[:, lo:hi].T @ B)
        bad = K.add_scalar_entries(Mb.indptr, Mb.indices, Mb.data, np.repeat(dofs[lo:hi], 3 * n), np.tile(dofs, hi - lo), G.ravel(), upper)
        if bad >= 0:
            raise AssemblyError("Internal error: the pattern of the matrix lacks a coupling of a point mass with a distributing constraint")


def assemble(nodes, tets, mat_of, Cm, fixed, g, F, *, upper: bool, dT: np.ndarray | None = None,
             alpha_e: np.ndarray | None = None, prune: bool = False, red: Reduction | None = None, aniso: Aniso | None = None,
             hom_cols: int = 0, mass: "MassInputs | None" = None) -> Assembled:
    """The equation system of the linear statics in node blocks: see the module docstring.

    nodes (N, 3), tets (E, 10) in the order of ccx, mat_of (E,) index of the matrix C of the element in Cm (M, 6, 6); fixed (3N,) bool = the
    constrained DOFs, g (3N,) or (3N, states) their prescribed values (one column = the same for every state, one column per state = per-state values: the unit displacements of
    check 3 share one matrix; the values of the constrained extra DOFs of `red` are `red.g_extra`, unless the rows of g hold values for them); F (3N, states) the nodal loads; dT (3N/3 = N, states) the temperature change of the nodes of
    every state (None = no thermal load) with alpha_e (E,) the expansion coefficient of every element (an isotropic one; `aniso` replaces it).
    red = the constraints of `Reduction` (dependent DOFs, extra nodes, springs): fixed / g / F then have 3·(N + extra nodes) rows – the DOFs of the extra
    nodes after those of the mesh; the dependent DOFs are eliminated (u_D = A·u + c, TᵀKT: the reduced matrix stays symmetric positive definite).
    aniso = orthotropic materials (`Aniso`): the cylindrical ones are rotated at every Gauss point, the thermal strain is a tensor.
    hom_cols = the number of the LAST columns of F that are homogeneous: their right-hand side is F alone (the prescribed displacements `g` and the offsets
    of the dependent DOFs are not applied, the constrained DOFs get 0); `dT` belongs to the first columns.
    mass = the mass of the model (`mass.MassInputs`): the consistent mass matrix M is assembled on the SAME pattern as K, transformed by the same elimination of the
    dependent DOFs (TᵀMT; the rows and columns of the constrained DOFs are zero, their diagonal too – no spurious modes), `Assembled.mdata` / `Assembled.MR`.
    """
    t = {}
    tic = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal tic
        now = time.perf_counter()
        t[name] = t.get(name, 0.0) + now - tic
        tic = now

    nodes, tets, mat_of, Cm = _canon(nodes, tets, mat_of, Cm)
    nn = len(nodes)
    nne = nn if red is None else nn + red.n_extra
    if red is not None and red.n_mesh != nn:
        raise AssemblyError("The constraints belong to a mesh with another number of nodes")
    n = 3 * nne
    fixed = np.array(fixed, dtype=bool).reshape(n)
    g = np.array(g, dtype=np.float64).reshape(n, -1)         # (n, 1) = one set of prescribed values for every state, (n, ns) = one per state
    F = np.array(F, dtype=np.float64).reshape(n, -1)
    ns = F.shape[1] - int(hom_cols)                  # the columns before the homogeneous ones
    if hom_cols < 0 or ns < 0:
        raise AssemblyError("Invalid number of homogeneous right-hand sides")
    if g.shape[1] not in (1, ns):
        raise AssemblyError("The prescribed values g need one column or one per state")
    lap("prepare")
    check_elements(nodes, tets)
    if not np.isfinite(nodes).all():                       # a node outside the elements (an orphan) with damaged coordinates
        raise AssemblyError("The node coordinates contain NaN or infinity")
    lap("check_elements")
    ptr, el, loc = K.node_elements(nne, tets)
    orphan = np.repeat(ptr[1:] == ptr[:-1], 3)      # a node without elements has no stiffness: constrained at 0 (otherwise a singular row)
    orphan[3 * nn:] = False                         # … but not an extra node (its stiffness comes from the elimination of the dependent DOFs)
    if orphan.any():
        fixed |= orphan
        g[orphan] = 0.0
    dep = np.zeros(0, np.int64)
    A = None
    c = None
    extra = None
    if red is not None:
        if len(red.fixed_extra):
            fixed[3 * nn:] |= red.fixed_extra
            # The prescribed value of a constrained extra DOF is `red.g_extra` (it replaces the row of g, as it always did). A caller that prescribes it PER STATE (check 3: the
            # displacement of a rigid support in every unit state) puts the values into that row of g instead: a row of g with a value in any column is kept, red.g_extra is
            # then not used for the DOF.
            ge = g[3 * nn:]
            use = red.fixed_extra & ~(ge != 0.0).any(axis=1)
            ge[use] = red.g_extra[use, None]
        try:
            red.validate(fixed)
        except ReductionError as e:
            raise AssemblyError(str(e)) from None
        dep = red.dep
        if red.n_dep:
            import scipy.sparse as sp
            A = red.matrix()
            c = red.c[:, None].copy()
            if fixed.any():                           # a master that is constrained: its value is known → into the offset (g = 0 → just dropped)
                c = c + A @ np.where(fixed[:, None], g, 0.0)
                A = (A @ sp.diags((~fixed).astype(float))).tocsr()
            A.eliminate_zeros()
    cliques = [idx for idx, _ in mass.dist_nodes] if mass is not None else []
    if red is not None or cliques:
        extra = _extra_couplings(red, nne, tets, ptr, el, upper, cliques)
    indptr, indices = K.pattern(nne, tets, ptr, el, upper, extra)
    lap("pattern")
    data = np.zeros((len(indices), 3, 3))
    if aniso is not None and aniso.cylindrical:
        K.stiffness_cyl(nodes, tets, mat_of, Cm, np.ascontiguousarray(aniso.kind, np.int8), np.ascontiguousarray(aniso.ax, np.float64),
                        ptr, el, loc, indptr, indices, K.DN4, K.N4, upper, data)
    else:
        K.stiffness(nodes, tets, mat_of, Cm, ptr, el, loc, indptr, indices, K.DN4, upper, data)
    lap("stiffness")
    mdata = None
    body_mass = 0.0
    if mass is not None:                              # the consistent mass of the bodies (the same pattern, the same node-by-node order of the sums)
        if len(mass.rho) != len(tets):
            raise AssemblyError("The densities do not match the elements")
        wd = np.empty((len(tets), len(K.W14)))
        K.mass_weights(nodes, tets, K.DN14, K.W14, wd)
        mdata = np.zeros((len(indices), 3, 3))
        K.mass_matrix(tets, np.ascontiguousarray(mass.rho, np.float64), wd, K.N14, ptr, el, loc, indptr, indices, upper, mdata)
        body_mass = float((wd.sum(axis=1) * mass.rho).sum())
        del wd
    del ptr, el, loc
    lap("mass")
    Kb = BlockMatrix(nne, indptr, indices, data, upper)
    Mb = None
    MR = None
    if mass is not None:
        Mb = BlockMatrix(nne, indptr, indices, mdata, upper)
        if len(mass.surf_nodes):                      # surface mass: m·I₃ on the diagonal block of the node (as the *MASS elements of ccx)
            agg = np.bincount(mass.surf_nodes, weights=mass.surf_mass, minlength=nne)
            nz = np.flatnonzero(agg)
            pos = Mb.dpos[nz]
            if (pos < 0).any():
                raise AssemblyError("A surface mass is on a node without elements")
            for c_ in range(3):
                mdata[pos, c_, c_] += agg[nz]
        for k_ in range(len(mass.rigid_node)):         # rigid point mass: m·I₃ on the displacement node, the inertia tensor on the rotation node of its pair
            e0 = int(mass.rigid_node[k_])
            if red is None or e0 < nn or e0 + 1 >= nne or Mb.dpos[e0] < 0 or Mb.dpos[e0 + 1] < 0:
                raise AssemblyError("A point mass with a rigid coupling refers to extra nodes that do not exist")
            for c_ in range(3):
                mdata[Mb.dpos[e0], c_, c_] += mass.rigid_mass[k_]
            mdata[Mb.dpos[e0 + 1]] += mass.rigid_inertia[k_]
        # M·R of the unreduced system (the rigid-body motions with the constrained DOFs zero): the participation factors Γ = φᵀ·M·R, the effective total mass
        R = rigid_motions(nodes, red.extra_xyz if red is not None else np.zeros((0, 3)), fixed)
        isdep0 = np.zeros(n, bool)
        isdep0[dep] = True

        def mass_times(Rm: np.ndarray) -> np.ndarray:
            out = Mb.matmat(Rm)
            for (idx, A3), m_ in zip(mass.dist_nodes, mass.dist_mass):
                dofs = (3 * np.asarray(idx, np.int64)[:, None] + np.arange(3)).ravel()
                if fixed[dofs].any() or isdep0[dofs].any():
                    raise AssemblyError("A point mass with a distributing coupling shares nodes with a support or with a constraint equation")
                B = np.asarray(A3, float).reshape(3, 3 * len(idx))
                out[dofs] += m_ * (B.T @ (B @ Rm[dofs]))
            return out

        MR = mass_times(R)
        Rf = release_held_directions(R, red)                 # the directions held by the cylindrical supports are not free motions
        MRf = MR if Rf is R else mass_times(Rf)
        total_free = np.einsum("ij,ij->j", Rf, MRf)          # rᵀ·M·r of the free DOFs per rigid-body motion: the effective total mass (translations) / inertia (rotations)
        lap("mass_motion")
    rbm = rigid_body_residual(Kb, nodes)
    if not np.isfinite(rbm):
        raise AssemblyError("The stiffness matrix is zero or contains NaN / infinity (check the materials and the node coordinates)")
    diag = Kb.diagonal()
    isdep = np.zeros(n, bool)
    isdep[dep] = True
    free = ~fixed & ~isdep
    free[3 * nn:] = False                            # the mean of the stiffness of the mesh (the extra DOFs have a different scale)
    if not free.any():
        raise AssemblyError("All degrees of freedom are constrained")
    d = float(diag[free].mean())
    if not (np.isfinite(d) and d > 0):
        raise AssemblyError("The stiffness of the free degrees of freedom is not positive (check the materials)")
    if red is not None and len(red.spring_nodes):    # elastic support: the springs are added to the diagonal blocks
        if (Kb.dpos[red.spring_nodes] < 0).any():
            raise AssemblyError("A spring is on a node without elements")
        Kb.data[Kb.dpos[red.spring_nodes]] += red.spring_blocks
    rhs = F
    t0 = g.copy()                                    # the known part of the solution: the prescribed values and the offsets of the dependent DOFs
    if c is not None:
        t0[dep] = c
    Kt0 = Kb.matmat(t0) if np.any(t0) else None      # the prescribed displacements: the other rows lose K·t0 ((n, 1) or (n, ns))
    lap("checks")
    if dT is not None:
        dT = np.asarray(dT, np.float64).reshape(nn, -1)
        if aniso is not None:
            Ca = np.ascontiguousarray(Cm)
            for k in range(dT.shape[1]):
                if np.any(dT[:, k]):
                    rhs[:3 * nn, k] += thermal_load_aniso(nodes, tets, mat_of, Ca, np.ascontiguousarray(aniso.kind, np.int8),
                                                          np.ascontiguousarray(aniso.ax, np.float64), np.ascontiguousarray(aniso.A, np.float64),
                                                          dT[:, k]).ravel()
        else:
            if alpha_e is None:
                raise AssemblyError("A thermal load needs the expansion coefficient of the elements")
            for k in range(dT.shape[1]):
                if np.any(dT[:, k]):
                    rhs[:3 * nn, k] += thermal_load(nodes, tets, mat_of, Cm, alpha_e, dT[:, k]).ravel()
    lap("thermal")
    if Kt0 is not None:
        rhs[:, :ns] -= Kt0
    n_corr = 0
    if A is not None:                                # elimination of the dependent DOFs: the loads of the dependent DOFs go to their masters, the matrix gets Corr
        rhs += A.T @ rhs[dep]
        cr, cc_, cv = _eliminate_dependent(Kb, red, A, fixed, upper)
        bad = K.add_scalar_entries(Kb.indptr, Kb.indices, Kb.data, cr, cc_, cv, upper)
        if bad >= 0:
            raise AssemblyError("Internal error: the pattern of the matrix lacks a coupling of the constraints")
        n_corr = len(cv)
        if Mb is not None:                           # the same elimination of the mass matrix: TᵀMT
            mr, mc, mv = _eliminate_dependent(Mb, red, A, fixed, upper)
            bad = K.add_scalar_entries(Mb.indptr, Mb.indices, Mb.data, mr, mc, mv, upper)
            if bad >= 0:
                raise AssemblyError("Internal error: the pattern of the matrix lacks a coupling of the constraints (mass)")
        fixed = fixed | isdep
    if red is not None and red.n_elem:               # the beam elements on the extra nodes (bolts)
        er, ec, ev = _element_entries(red)
        bad = K.add_scalar_entries(Kb.indptr, Kb.indices, Kb.data, er, ec, ev, upper)
        if bad >= 0:
            raise AssemblyError("Internal error: the pattern of the matrix lacks a coupling of a beam element")
    if mass is not None:                              # the dense blocks of the point masses with a distributing coupling (their nodes are free masters)
        for (idx, A3), m_ in zip(mass.dist_nodes, mass.dist_mass):
            _add_dense_mass(Mb, idx, A3, float(m_), upper)
    lap("constraints")
    rhs[fixed, :ns] = d * g[fixed]
    rhs[fixed, ns:] = 0.0
    counts = K.scalar_counts(nne, indptr, indices, fixed, upper, prune)
    rowptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=rowptr[1:])
    lap("pattern_scalar")
    info = {"ndof": n, "nnz": int(rowptr[-1]), "nnz_blocks": int(len(indices)), "format": "upper" if upper else "full",
            "rigid_body_residual": rbm, "d": d, "pruned": bool(prune), "orphan_nodes": int(orphan.sum() // 3), "constrained": int(fixed.sum()), "times": t}
    if red is not None:
        info.update({"dependent_dofs": int(red.n_dep), "extra_nodes": int(red.n_extra), "springs": int(len(red.spring_nodes)),
                     "beam_elements": int(red.n_elem), "correction_entries": int(n_corr), "constraints": dict(red.labels)})
    if mass is not None:
        info["mass"] = {"bodies_t": body_mass, "added_t": float(mass.surf_mass.sum() + mass.rigid_mass.sum() + mass.dist_mass.sum()),
                        "total_free": [float(v) for v in total_free], **mass.labels}
    return Assembled(nne, indptr, indices, data, fixed, d, np.ascontiguousarray(rhs), upper, bool(prune), rowptr, info, mdata, MR)


def build_system(nodes, tets, mat_of, Cm, fixed, g, F, *, upper: bool, dT: np.ndarray | None = None,
                 alpha_e: np.ndarray | None = None, prune: bool = False, log=lambda s: None, red: Reduction | None = None,
                 aniso: Aniso | None = None, hom_cols: int = 0, mass: "MassInputs | None" = None) -> System:
    """`assemble(...).to_system()`: the whole scalar CSR in memory (tests, the warm-up; the worker streams it – `Assembled.write`)."""
    return assemble(nodes, tets, mat_of, Cm, fixed, g, F, upper=upper, dT=dT, alpha_e=alpha_e, prune=prune, red=red, aniso=aniso,
                    hom_cols=hom_cols, mass=mass).to_system()

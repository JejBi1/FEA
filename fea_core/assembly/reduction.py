"""The linear constraints of the own assembly: what `fea_core.linear_constraints` hands to the assembly worker (no numba, no heavy imports).

A `Reduction` describes the constraints that are NOT a plain "this DOF is fixed" (that is `fixed` / `g` of `system.assemble`):

* **dependent DOFs**: u[dep[i]] = Σ_k A[i, k]·u[k] + c[i] – a master-slave elimination (the transformation u = T·w, the reduced matrix TᵀKT is symmetric
  positive definite – no Lagrange multipliers). The masters are free DOFs of the mesh or DOFs of the **extra nodes**; a master must not be dependent itself.
  Cylindrical support (one dependent DOF per direction at a node), rigid and distributing remote constraints, *TIE (three DOFs of a slave node from the
  nodes of the master face) are all this;
* **extra nodes**: points outside the mesh (the reference point of a rigid remote constraint: node 2i = the displacement, node 2i + 1 = the rotation)
  with their own DOFs (numbered after the DOFs of the mesh: node `n_mesh + i`); `fixed_extra` / `g_extra` = the constrained components;
* **springs**: 3×3 blocks added to the diagonal blocks of nodes (elastic support, the same stiffness as the springs written for ccx);
* **elements on the extra nodes**: `elem_dofs` (m, 12) / `elem_k` (m, 12, 12) = the stiffness matrices of beam elements whose nodes are extra nodes (the bolts:
  one exact Timoshenko element between the rigid bearing faces) – added to the matrix of the extended system.

Everything is in the global numbering of DOFs: 3·node + component; the extra nodes follow the nodes of the mesh.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ReductionError(ValueError):
    """The constraints cannot be handled by the own assembly (a master that is dependent itself, a DOF dependent twice, a dependent DOF that is also
    constrained…) – the caller falls back to CalculiX, which gives its own message for such a model. `dof` = the DOF of the conflict (global numbering) when
    there is one – `linear_constraints` names the constraints that hold its node."""

    def __init__(self, message: str, dof: int | None = None):
        super().__init__(message)
        self.dof = dof


FILE = "reduction.npz"


@dataclass
class Reduction:
    n_mesh: int                                 # nodes of the mesh
    extra_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))      # (nx, 3) the extra nodes (points of the rigid couplings)
    fixed_extra: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))   # (3 nx,) constrained components of the extra DOFs
    g_extra: np.ndarray = field(default_factory=lambda: np.zeros(0))             # (3 nx,) their prescribed values
    dep: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))       # (m,) dependent DOFs
    ptr: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))       # (m + 1,) the rows of A (CSR)
    col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))       # (nnz,) the master DOFs
    val: np.ndarray = field(default_factory=lambda: np.zeros(0))                 # (nnz,) the coefficients
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))                   # (m,) the offsets
    spring_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))    # (k,) nodes with a spring (unique)
    spring_blocks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))     # (k, 3, 3) added to their diagonal blocks
    elem_dofs: np.ndarray = field(default_factory=lambda: np.zeros((0, 12), np.int64))   # (m, 12) the DOFs of beam elements on the extra nodes (4 nodes of 3 DOFs)
    elem_k: np.ndarray = field(default_factory=lambda: np.zeros((0, 12, 12)))            # (m, 12, 12) their stiffness matrices (symmetric)
    labels: dict = field(default_factory=dict)  # for the protocol: {"cylindrical": n, "rigid": n, "distributing": n, "tie": n, "springs": n}

    def __post_init__(self) -> None:
        """The components of the extra nodes default to free with a zero prescribed value."""
        self.extra_xyz = np.asarray(self.extra_xyz, float).reshape(-1, 3)
        nx3 = 3 * len(self.extra_xyz)
        self.fixed_extra = np.asarray(self.fixed_extra, bool) if len(self.fixed_extra) else np.zeros(nx3, bool)
        self.g_extra = np.asarray(self.g_extra, float) if len(self.g_extra) else np.zeros(nx3)
        if len(self.fixed_extra) != nx3 or len(self.g_extra) != nx3:
            raise ReductionError("The constrained components of the extra nodes do not match their number")
        self.elem_dofs = np.asarray(self.elem_dofs, np.int64).reshape(-1, 12)
        self.elem_k = np.asarray(self.elem_k, float).reshape(-1, 12, 12)
        if len(self.elem_dofs) != len(self.elem_k):
            raise ReductionError("The element DOFs and matrices on the extra nodes do not match")

    # ------------------------------------------------------------------ sizes
    @property
    def n_extra(self) -> int:
        return int(len(self.extra_xyz))

    @property
    def n_nodes(self) -> int:
        """Nodes of the extended system (mesh + extra)."""
        return self.n_mesh + self.n_extra

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_dep(self) -> int:
        return int(len(self.dep))

    @property
    def empty(self) -> bool:
        """Nothing to eliminate or to add (the plan may exist only for the loads of a distributing point mass)."""
        return not (self.n_dep or self.n_extra or len(self.spring_nodes) or len(self.elem_dofs))

    @property
    def n_elem(self) -> int:
        """Beam elements on the extra nodes (bolts)."""
        return int(len(self.elem_dofs))

    @property
    def multi_node(self) -> bool:
        """The reduction couples nodes (extra nodes, ties, distributing equations) – not only the DOFs of one node (cylindrical support, springs). Decided by the
        equations themselves (a dependent DOF with a master of ANOTHER node), not by the labels: they are only descriptions, and a file from an older writer has none."""
        if self.n_extra or self.labels.get("distributing") or self.labels.get("tie"):
            return True
        if self.n_dep:
            owner = np.repeat(self.dep // 3, np.diff(self.ptr))
            return bool((self.col // 3 != owner).any())
        return False

    def matrix(self):
        """A as a scipy CSR matrix (m, n_dof)."""
        import scipy.sparse as sp
        return sp.csr_matrix((self.val, self.col, self.ptr), shape=(self.n_dep, self.n_dof))

    # ------------------------------------------------------------------ construction
    @staticmethod
    def from_rows(n_mesh: int, rows: list[tuple[int, list[tuple[int, float]], float]], **kw) -> "Reduction":
        """From a list of equations (dependent DOF, [(master DOF, coefficient), …], offset): u_dep = Σ coefficient·u_master + offset."""
        dep = np.array([r[0] for r in rows], np.int64)
        ptr = np.zeros(len(rows) + 1, np.int64)
        np.cumsum([len(r[1]) for r in rows], out=ptr[1:])
        col = np.array([k for r in rows for k, _ in r[1]], np.int64)
        val = np.array([v for r in rows for _, v in r[1]], float)
        c = np.array([r[2] for r in rows], float)
        return Reduction(n_mesh, dep=dep, ptr=ptr, col=col, val=val, c=c, **kw)

    def merge(self, other: "Reduction") -> "Reduction":
        """The union of two reductions of the same mesh (their extra nodes follow each other: the masters of `other` that refer to extra nodes are shifted)."""
        if other.n_mesh != self.n_mesh:
            raise ReductionError("Reductions of different meshes cannot be merged")
        shift = 3 * self.n_extra
        base = 3 * self.n_mesh
        col = np.where(other.col >= base, other.col + shift, other.col)
        dep = np.where(other.dep >= base, other.dep + shift, other.dep)
        edofs = np.where(other.elem_dofs >= base, other.elem_dofs + shift, other.elem_dofs)
        lab = dict(self.labels)
        for k, v in other.labels.items():
            lab[k] = lab.get(k, 0) + v
        return Reduction(
            self.n_mesh, np.vstack([self.extra_xyz, other.extra_xyz]), np.concatenate([self.fixed_extra, other.fixed_extra]),
            np.concatenate([self.g_extra, other.g_extra]), np.concatenate([self.dep, dep]),
            np.concatenate([self.ptr[:-1], other.ptr + self.ptr[-1]]), np.concatenate([self.col, col]), np.concatenate([self.val, other.val]),
            np.concatenate([self.c, other.c]), np.concatenate([self.spring_nodes, other.spring_nodes]),
            np.concatenate([self.spring_blocks, other.spring_blocks]), np.concatenate([self.elem_dofs, edofs]),
            np.concatenate([self.elem_k, other.elem_k]), lab)

    def validate(self, fixed: np.ndarray | None = None) -> None:
        """Raises `ReductionError` for a DOF that is dependent twice, a master that is dependent, a dependent DOF that is constrained, an index outside."""
        n = self.n_dof
        if self.n_dep:
            if len(self.ptr) != self.n_dep + 1 or self.ptr[-1] != len(self.col) or len(self.col) != len(self.val):
                raise ReductionError("Inconsistent arrays of the constraint equations")
            if self.dep.min() < 0 or self.dep.max() >= n or (len(self.col) and (self.col.min() < 0 or self.col.max() >= n)):
                raise ReductionError("A constraint equation refers to a DOF that does not exist")
            if self.dep.max() >= 3 * self.n_mesh:
                raise ReductionError("A dependent DOF must belong to a node of the mesh")
            u, cnt = np.unique(self.dep, return_counts=True)
            if (cnt > 1).any():
                d = int(u[cnt > 1][0])
                raise ReductionError(f"DOF {d % 3 + 1} of node {d // 3 + 1} is dependent in two constraint equations (two constraints on one node)", d)
            isdep = np.zeros(n, bool)
            isdep[self.dep] = True
            if len(self.col) and isdep[self.col].any():
                d = int(self.col[isdep[self.col]][0])
                raise ReductionError(f"DOF {d % 3 + 1} of node {d // 3 + 1} is dependent and also the master of another constraint equation", d)
            if fixed is not None and fixed[self.dep].any():
                d = int(self.dep[fixed[self.dep]][0])
                raise ReductionError(f"DOF {d % 3 + 1} of node {d // 3 + 1} is dependent in a constraint equation and also constrained", d)
        if len(self.spring_nodes) and (self.spring_nodes.min() < 0 or self.spring_nodes.max() >= self.n_mesh):
            raise ReductionError("A spring refers to a node that does not exist")
        if len(self.elem_dofs):
            if self.elem_dofs.min() < 3 * self.n_mesh or self.elem_dofs.max() >= n:
                raise ReductionError("A beam element must lie on the extra nodes")
            if not np.isfinite(self.elem_k).all():
                raise ReductionError("The stiffness of a beam element contains NaN or infinity")

    # ------------------------------------------------------------------ files
    def save(self, directory: Path | str) -> Path:
        path = Path(directory) / FILE
        np.savez(path, n_mesh=np.int64(self.n_mesh), extra_xyz=self.extra_xyz, fixed_extra=self.fixed_extra, g_extra=self.g_extra, dep=self.dep,
                 ptr=self.ptr, col=self.col, val=self.val, c=self.c, spring_nodes=self.spring_nodes, spring_blocks=self.spring_blocks,
                 elem_dofs=self.elem_dofs, elem_k=self.elem_k, labels=np.array(json.dumps(self.labels)))          # a text array (no pickle): the worker reports the kinds of constraints in assemble.json
        return path

    @staticmethod
    def load(directory: Path | str) -> "Reduction | None":
        path = Path(directory) / FILE
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as z:
            labels = json.loads(str(z["labels"])) if "labels" in z.files else {}
            ed = z["elem_dofs"] if "elem_dofs" in z.files else np.zeros((0, 12), np.int64)       # a file of an older writer has no elements
            ek = z["elem_k"] if "elem_k" in z.files else np.zeros((0, 12, 12))
            return Reduction(int(z["n_mesh"]), z["extra_xyz"], z["fixed_extra"], z["g_extra"], z["dep"], z["ptr"], z["col"], z["val"], z["c"],
                             z["spring_nodes"], z["spring_blocks"], ed, ek, labels)

    # ------------------------------------------------------------------ the solution of the extended system → the DOFs of the mesh
    def expand(self, X: np.ndarray) -> np.ndarray:
        """X (n_dof, states) = the solution of the reduced system (the dependent DOFs hold 0) → the same with the dependent DOFs u = A·u + c
        (returns a new array; the rows of the extra nodes stay)."""
        out = np.array(X, dtype=float, copy=True)
        if self.n_dep:
            out[self.dep] = self.matrix() @ out + self.c[:, None]
        return out

"""Parallel solution of K·x = b by hypre (PCG + BoomerAMG for elasticity) in MPI processes.

The caller supplies the system in the order node×3 components (the degrees of freedom of a node are adjacent); the Dirichlet conditions are applied
here (symmetric elimination, diagonal = mean of the diagonal of K, so that the conditioning of the AMG does not get worse).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .. import config

HYPRE_BIN = config.HYPRE_BIN


class HypreError(RuntimeError):
    pass


def mpiexec() -> str | None:
    """mpiexec: on Linux preferably from the hypre environment (the same MPI as HYPRE), otherwise from PATH; on Windows also from the
    MS-MPI folder (MSMPI_BIN, Program Files) – `solvers.find_mpiexec`.
    """
    from ..solvers import find_mpiexec
    return find_mpiexec()


def available() -> bool:
    """The hypre solver can be used: numba (the own matrix assembly), mpi4py, the HYPRE library and mpiexec (`solvers.hypre_missing`)."""
    from ..solvers import hypre_missing
    return not hypre_missing()


def apply_dirichlet(K: sp.csr_matrix, F: np.ndarray, fixed: np.ndarray, values: np.ndarray):
    """Symmetric elimination: K_ff stays, the constrained rows / columns → d·I, the right-hand side adjusted."""
    free = ~fixed
    g = np.zeros(K.shape[0])
    g[fixed] = values[fixed]
    F = np.array(F, dtype=float, copy=True)
    if F.ndim == 1:
        F = F[:, None]
    if np.any(g):
        F -= (K @ g)[:, None]
    d = float(K.diagonal()[free].mean())
    Df = sp.diags(free.astype(float))
    Kb = (Df @ K @ Df + sp.diags(fixed.astype(float) * d)).tocsr()
    Kb.sort_indices()
    F[fixed, :] = d * g[fixed, None]
    return Kb, F


def solve(K: sp.csr_matrix, F: np.ndarray, rbm: np.ndarray, workdir: str | Path, nproc: int | None = None,
          tol: float = 1e-10, maxiter: int = 2000, amg: dict | None = None,
          cpoints: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Solves K·X = F (F: n × states). Returns (X, info)."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    K = K.tocsr()
    np.save(wd / "indptr.npy", K.indptr.astype(np.int64))
    np.save(wd / "indices.npy", K.indices.astype(np.int32))
    np.save(wd / "data.npy", K.data.astype(np.float64))
    np.save(wd / "rhs.npy", np.ascontiguousarray(F if F.ndim == 2 else F[:, None], dtype=np.float64))
    np.save(wd / "rbm.npy", np.ascontiguousarray(rbm, dtype=np.float64))
    (wd / "cpts.npy").unlink(missing_ok=True)
    if cpoints is not None:
        np.save(wd / "cpts.npy", np.asarray(cpoints, dtype=np.int32))
    return run_prepared(wd, nproc, tol, maxiter, amg)


def run_prepared(workdir: str | Path, nproc: int | None = None, tol: float = 1e-10, maxiter: int = 2000,
                 amg: dict | None = None, cancel=None, on_line=None) -> tuple[np.ndarray, dict]:
    """Runs hypre over the system stored in the workdir (indptr/indices/data/rhs/rbm.npy).
    cancel (threading.Event) set → mpiexec and all MPI ranks are killed, HypreError("Analysis canceled by user");
    the output of mpiexec / the ranks goes to workdir/hypre.log (and to on_line).
    """
    if not available():
        from ..solvers import backend_error
        raise HypreError(backend_error("ngs_hypre") or "hypre is not available")
    wd = Path(workdir)
    for f in wd.glob("x_*.npy"):
        f.unlink()
    (wd / "info.json").unlink(missing_ok=True)
    n = int(np.load(wd / "rhs.npy", mmap_mode="r").shape[0])
    if n % 3:
        raise HypreError("Number of degrees of freedom is not a multiple of 3")
    from ..ccx.parallel import cap_threads                 # FEA_THREADS also caps the MPI processes (an explicit nproc cannot raise it)
    nproc = cap_threads(nproc) or config.NUM_THREADS
    nproc = max(1, min(nproc, n // 3000 or 1))          # small problems on fewer processes
    nodes = n // 3
    bounds = [3 * (nodes * r // nproc) for r in range(nproc + 1)]
    (wd / "job.json").write_text(json.dumps({"bounds": bounds, "tol": tol, "maxiter": maxiter, "amg": amg or {}}),
                                 encoding="utf-8")
    env = config.child_env()
    var = "PATH" if config.IS_WIN else "LD_LIBRARY_PATH"
    env[var] = os.pathsep.join([str(HYPRE_BIN)] + [d for d in env.get(var, "").split(os.pathsep) if d])
    env["OMP_NUM_THREADS"] = "1"
    from ..procs import run_tree, tail, Canceled
    log_path = wd / "hypre.log"
    try:                                     # run_tree releases the unused memory before the launch (MPI processes get it)
        rc = run_tree([mpiexec(), "-n", str(nproc), sys.executable, "-m", "fea_core.hypre.worker", str(wd)],
                      log_path, cancel, env=env, on_line=on_line)
    except Canceled:
        raise HypreError("Analysis canceled by user") from None
    if rc != 0 or not (wd / "info.json").exists():
        raise HypreError(f"hypre failed (code {rc}):\n{tail(log_path, 4000)}")
    X = np.vstack([np.load(wd / f"x_{r}.npy") for r in range(nproc)])
    info = json.loads((wd / "info.json").read_text(encoding="utf-8"))
    bad = [c for c in info["cases"] if not c["rel_residual"] <= 10 * tol]
    if bad:                                   # solve_large.run_ngs keeps the system files only with FEA_KEEP_SOLVER_FILES=1
        raise HypreError(f"hypre did not converge: {bad}")
    return X, info


def rotation_modes(X: np.ndarray) -> np.ndarray:
    """Three rigid rotations of the body in the order node×3 (interpolation vectors of BoomerAMG)."""
    c = X - X.mean(axis=0)
    n = len(X)
    R = np.zeros((3 * n, 3))
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    i = 3 * np.arange(n)
    R[i, 0], R[i + 1, 0] = -y, x          # kolem z
    R[i + 1, 1], R[i + 2, 1] = -z, y      # kolem x
    R[i + 2, 2], R[i, 2] = x, -z          # kolem y
    return R

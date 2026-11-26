"""Eigenvalue process of the direct path: python -m fea_core.direct.eigen_worker JOB

Input (written by the assembly worker, `fea_core.assembly.worker`, with a `mass.npz` in the job folder):
  JOB/indptr.npy, indices.npy, data.npy   the stiffness matrix K (upper triangle, CSR; the constrained DOFs eliminated: rows / columns zero, diagonal d)
  JOB/mdata.npy                           the mass matrix M on the SAME pattern (the constrained rows / columns zero, zero diagonal)
  JOB/rhs.npy                             static right-hand sides, solved from the same factor when eigen_options.json says "statics": true
  JOB/eigen_options.json                  {"modes", "lam_min", "lam_max", "threads", "engine", "reserve_gb", "statics"}
Output: JOB/eig_lam.npy (m,), JOB/eig_X.npy (n, m) float64 (M-orthonormal; the dependent DOFs are expanded by the caller), JOB/statics.npy, JOB/eigen_info.json.
Exit codes: 0 ok, 3 does not fit into memory, 5 numerical failure (a negative stiffness, no mass), 6 accuracy not reached (the iteration, a residual, the
orthogonality), 7 a mode could not be proved complete by the Sturm count (see `fea_core.direct.eigen`).

A separate process like the direct solver: the main program holds neither the matrices nor the eigenvectors (RAM for the solver), a crash of the library does not
bring the GUI down and the analysis can be cancelled (the process tree is killed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .eigen import EigenFail, EigenOptions, solve_modes
from .worker import peak_gb


def main(job: Path) -> int:
    t0 = time.time()
    for f in ("eig_lam.npy", "eig_X.npy", "statics.npy", "eigen_info.json"):
        (job / f).unlink(missing_ok=True)
    opts = json.loads((job / "eigen_options.json").read_text(encoding="utf-8"))
    info: dict = {}
    code = 0
    try:
        indptr = np.load(job / "indptr.npy")
        indices = np.load(job / "indices.npy", mmap_mode="r")
        n = len(indptr) - 1
        KU = sp.csr_matrix((np.asarray(np.load(job / "data.npy", mmap_mode="r")), np.asarray(indices), indptr), shape=(n, n), copy=False)
        MU = sp.csr_matrix((np.asarray(np.load(job / "mdata.npy", mmap_mode="r")), np.asarray(indices), indptr), shape=(n, n), copy=False)
        rhs = np.load(job / "rhs.npy").reshape(n, -1) if opts.get("statics") else None
        info["t_load"] = time.time() - t0
        res = solve_modes(KU, MU, EigenOptions.from_dict(opts), info, log=lambda s: print(s, flush=True), rhs=rhs)
        np.save(job / "eig_lam.npy", res.lam)
        np.save(job / "eig_X.npy", np.ascontiguousarray(res.X))
        if res.statics is not None:
            np.save(job / "statics.npy", np.ascontiguousarray(res.statics))
        f = np.sqrt(np.maximum(res.lam, 0.0)) / (2 * np.pi)
        print(f"eigen: {len(res.lam)} modes, {f[0]:.6g} … {f[-1]:.6g} Hz; analysis {info['t_analyze']:.1f} s, {info['n_factor']} factorizations "
              f"{info['t_factor']:.1f} s, Lanczos {info['arpack_solves']} solves {info['t_arpack']:.1f} s, residual {info['max_residual']:.1e}", flush=True)
    except EigenFail as e:
        info["error"] = {"kind": e.kind, "message": e.message}
        print(f"eigen: {e.message}", flush=True)
        code = e.code
    info["peak_gb"] = peak_gb()
    info["t_worker"] = time.time() - t0
    (job / "eigen_info.json").write_text(json.dumps(info), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

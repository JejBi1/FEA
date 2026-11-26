"""Assembly process: python -m fea_core.assembly.worker JOB [direct|hypre]

Reads the job files of the linear path (written by `fea_core.solve_large`):
  nodes.npy (N, 3), tets.npy (E, 10) int32 in the order of ccx, tet_part.npy (E,), materials.json [{E, nu, alpha}] per part,
  optionally C_tab.npy (M, 6, 6) + C_idx.npy (E,) = the matrix C of every element for 3D printing (global axes; without them the isotropic C per part),
  F.npy (3N, states), fixed.npy (3N,) bool, g.npy (3N,) or (3N, states) = the prescribed values (one column per state), optionally dT.npy (N, states) = the temperature change of the nodes of every state,
  optionally reduction.npz (`reduction.Reduction`: dependent DOFs, extra nodes, springs – F, fixed and g then have 3 (N + extra nodes) rows) and
  aniso.npz (kind, ax, A of `system.Aniso`: orthotropic materials; C_tab.npy is then the table of all materials),
  optionally mass.npz (`mass.MassInputs`: the densities and the added masses – the mass matrix is then assembled too, direct only),
  assembly.json {"threads": n, "hom_cols": k}  (the last k right-hand sides are homogeneous: the unit states of the bolt preload)
and writes the equation system with the constraints eliminated (`system.build_system`):
  direct → JOB/direct/{indptr,indices,data,rhs}.npy + matrix.json (the upper triangle, for `fea_core.direct`); with a mass also mdata.npy (the mass matrix on
           the same pattern) and MR.npy (M·R of the rigid-body motions of the unreduced system, for the participation factors)
  hypre  → JOB/hypre/{indptr,indices,data,rhs,rbm}.npy (the whole matrix)
  JOB/assemble.json (sizes, times of the phases, the rigid-body check, the peak memory of the process).
An element with a Jacobian determinant that is not positive (inverted, flat, NaN) is not a crash: the process writes JOB/assembly_error.json (kind
"elements", the first elements and their count) and exits with the code 3 – `assembly.run_worker` raises `ElementError` from it, the caller adds the parts.
Any other `AssemblyError` (all DOFs constrained, NaN in the materials…) exits with the code 4 and kind "assembly" – the message without a traceback.

A separate process so that the main program does not hold the matrix (RAM for the solver) and does not load numba, and so that the assembly can be
cancelled (the process tree is killed). The first run compiles the numba kernels (≈ 20 s, then they are cached) – a line starting with ">> " tells the
caller (it goes to the log of the calculation); the lines starting with ">> " are the only ones that are shown to the user.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import ERROR_FILE, EXIT_ASSEMBLY, EXIT_ELEMENTS, NOTICE_PREFIX, AssemblyError, ElementError

COMPILE_NOTICE_AFTER = 2.0          # s: a warm-up call of the kernels that takes longer is a compilation (loading from the cache takes ≈ 0.5 s)
COMPILE_NOTICE = "Compiling the assembly kernel (first use after the installation or an update, about 20 s – later runs load it from the cache)"


def say(text: str) -> None:
    print(text, flush=True)


def warm_up(constraints: bool = False, orthotropic: str = "", mass: bool = False, uppers: tuple = (True, False)) -> float:
    """Loads (or compiles) every kernel on a tiny model with the dtypes of the real run; announces a compilation. → the time [s].
    constraints = the job has dependent DOFs / extra nodes / springs (the elimination kernels); orthotropic = "cartesian" | "cylindrical" (the anisotropic
    thermal load; the cylindrical stiffness rotated at the Gauss points), mass = the mass matrix (the density kernels), uppers = the matrix formats (True = the
    upper triangle of the direct solver, False = the whole matrix of hypre; a direct job loads only the first – every variant costs ≈ 0.3 s of cache loading) –
    each kernel is compiled at the first use after an installation only."""
    from .mass import MassInputs
    from .system import Aniso, build_system
    from .reduction import Reduction

    t0 = time.perf_counter()
    timer = threading.Timer(COMPILE_NOTICE_AFTER, say, args=(NOTICE_PREFIX + COMPILE_NOTICE,))
    timer.daemon = True
    timer.start()
    try:
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5], [.5, 0, .5], [0, .5, .5]], float)
        tets = np.arange(10, dtype=np.int32)[None, :]
        from .kernels import iso_stiffness
        for upper in uppers:
            build_system(nodes, tets, np.zeros(1, np.int32), iso_stiffness(1.0, 0.3)[None], np.zeros(30, bool), np.zeros(30), np.zeros((30, 1)),
                         upper=upper, dT=np.ones((10, 1)), alpha_e=np.ones(1))
        if constraints or orthotropic:
            red = None
            n = 30
            fixed = np.zeros(30, bool)
            fixed[:3] = True
            if constraints:
                n = 36
                fixed = np.zeros(36, bool)
                fixed[:3] = True
                red = Reduction.from_rows(10, [(4, [(5, 0.5), (7, 0.25)], 0.0), (3 * 7 + 1, [(3 * 8, 1.0)], 0.1)],
                                          extra_xyz=np.zeros((2, 3)), spring_nodes=np.array([2], np.int64), spring_blocks=np.eye(3)[None])
            an = None
            if orthotropic:
                an = Aniso(np.array([1 if orthotropic == "cylindrical" else 0], np.int8), np.array([[0, 0, 0, 0, 0, 1.0]]), np.ones((1, 6)))
            for upper in uppers:
                build_system(nodes, tets, np.zeros(1, np.int32), iso_stiffness(1.0, 0.3)[None], fixed, np.zeros(n), np.zeros((n, 1)),
                             upper=upper, dT=np.ones((10, 1)), alpha_e=None if an is not None else np.ones(1), red=red, aniso=an)
        if mass:
            mi = MassInputs(np.ones(1), np.array([0], np.int64), np.ones(1), np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3, 3)),
                            [(np.array([4, 5, 6], np.int64), np.ones((3, 3, 3)) / 9.0)], np.ones(1))
            red2 = Reduction.from_rows(10, [(4, [(5, 0.5), (7, 0.25)], 0.0)], extra_xyz=np.zeros((2, 3)))
            mi.rigid_node, mi.rigid_mass, mi.rigid_inertia = np.array([10], np.int64), np.ones(1), np.eye(3)[None]
            fx = np.zeros(36, bool)
            fx[:3] = True
            build_system(nodes, tets, np.zeros(1, np.int32), iso_stiffness(1.0, 0.3)[None], fx, np.zeros(36), np.zeros((36, 1)), upper=True, red=red2, mass=mi)
    finally:
        timer.cancel()
    return time.perf_counter() - t0


def load_materials(job: Path, tet_part: np.ndarray):
    """(Cm (M, 6, 6), mat_of (E,), alpha_e (E,)) from the job files."""
    from .kernels import iso_stiffness
    mats = json.loads((job / "materials.json").read_text(encoding="utf-8"))
    alpha = np.array([float(m.get("alpha") or 0.0) for m in mats])
    tet_part = np.asarray(tet_part)
    if (job / "C_tab.npy").exists():
        return np.load(job / "C_tab.npy"), np.load(job / "C_idx.npy"), alpha[tet_part]
    Cm = np.stack([iso_stiffness(float(m["E"]), float(m["nu"])) for m in mats])
    return Cm, tet_part, alpha[tet_part]


def peak_gb() -> float | None:
    """The peak memory of this process [GB] (the working set on Windows, ru_maxrss on Linux)."""
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**30
    except (ImportError, AttributeError):
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20              # kB → GB
        except (ImportError, OSError):
            return None


TIMES_OF_ASSEMBLY = ("prepare", "check_elements", "pattern", "stiffness", "mass", "mass_motion", "checks", "thermal", "constraints", "pattern_scalar", "convert")


def main(job: Path, solver: str) -> int:
    t0 = time.perf_counter()
    opts = {}
    if (job / "assembly.json").exists():
        opts = json.loads((job / "assembly.json").read_text(encoding="utf-8"))
    import numba as nb
    from .system import assemble
    t_import = time.perf_counter() - t0
    threads = int(opts.get("threads") or 0) or nb.config.NUMBA_NUM_THREADS
    nb.set_num_threads(max(1, min(threads, nb.config.NUMBA_NUM_THREADS)))
    from .reduction import Reduction
    from .mass import MassInputs
    red = Reduction.load(job)
    has_aniso = (job / "aniso.npz").exists()
    ortho_kind = ""
    if has_aniso:
        with np.load(job / "aniso.npz", allow_pickle=False) as z:
            ortho_kind = "cylindrical" if (z["kind"] == 1).any() else "cartesian"
    mass = MassInputs.load(job)
    if mass is not None and solver != "direct":
        raise AssemblyError("The mass matrix is assembled for the direct solver only")
    t_compile = warm_up(constraints=red is not None, orthotropic=ortho_kind, mass=mass is not None, uppers=(True,) if solver == "direct" else (True, False))
    t1 = time.perf_counter()
    nodes = np.load(job / "nodes.npy")
    tets = np.load(job / "tets.npy")
    tet_part = np.load(job / "tet_part.npy")
    Cm, mat_of, alpha_e = load_materials(job, tet_part)
    F = np.load(job / "F.npy")
    fixed = np.load(job / "fixed.npy")
    g = np.load(job / "g.npy")
    dT = np.load(job / "dT.npy") if (job / "dT.npy").exists() else None
    upper = solver == "direct"
    aniso = None
    if has_aniso:
        from .system import Aniso
        with np.load(job / "aniso.npz", allow_pickle=False) as z:
            aniso = Aniso(z["kind"], z["ax"], z["A"])
    if red is not None and not upper and red.multi_node:
        raise AssemblyError("The iterative solver cannot do constraints that couple several nodes (rigid / distributing remote constraints, *TIE)")
    t_read = time.perf_counter() - t1
    asm = assemble(nodes, tets, mat_of, Cm, fixed, g, F, upper=upper, dT=dT, alpha_e=alpha_e, red=red, aniso=aniso, hom_cols=int(opts.get("hom_cols") or 0),
                   mass=mass)
    del tets, tet_part, F, g
    t2 = time.perf_counter()
    hj = job / ("direct" if upper else "hypre")
    asm.write(hj)                                   # the scalar rows are made and written chunk by chunk (the block matrix is the only large array)
    if upper:
        (hj / "matrix.json").write_text(json.dumps({"format": "upper", "mass": mass is not None}), encoding="utf-8")
    else:
        from ..hypre import rotation_modes
        np.save(hj / "rbm.npy", rotation_modes(nodes))
    t_write_total = time.perf_counter() - t2
    times = dict(asm.info["times"])
    t_assemble = sum(times.get(k, 0.0) for k in TIMES_OF_ASSEMBLY)
    t_write = max(t_write_total - times.get("convert", 0.0), 0.0)
    out = {**{k: v for k, v in asm.info.items() if k != "times"}, "assembler": "own", "threads": int(nb.get_num_threads()),
           "numba": nb.__version__, "t_import": t_import, "t_compile": t_compile, "t_read": t_read, "t_assemble": t_assemble,
           "t_write": t_write, "t_total": time.perf_counter() - t0, "times": times, "peak_gb": peak_gb()}
    (job / "assemble.json").write_text(json.dumps(out), encoding="utf-8")
    say(f"assembled {out['ndof']:,} DOF, {out['nnz'] / 1e6:.1f} M nonzeros ({out['format']}) in {t_assemble:.2f} s "
        f"(kernels {t_compile:.1f} s, reading {t_read:.2f} s, writing {t_write:.2f} s, {out['threads']} threads)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "direct"))
    except ElementError as e:                       # a mesh problem, not a crash: the caller names the parts (solve_large) and does not change the solver
        print(f"ERROR: {e}", flush=True)
        (Path(sys.argv[1]) / ERROR_FILE).write_text(json.dumps({"kind": "elements", "message": str(e), "elements": e.elements, "count": e.count}),
                                                    encoding="utf-8")
        sys.exit(EXIT_ELEMENTS)
    except AssemblyError as e:                      # an input the system cannot be built from (all DOFs constrained, NaN material…): the message only
        print(f"ERROR: {e}", flush=True)
        (Path(sys.argv[1]) / ERROR_FILE).write_text(json.dumps({"kind": "assembly", "message": str(e)}), encoding="utf-8")
        sys.exit(EXIT_ASSEMBLY)
    except Exception:  # noqa: BLE001 – the caller shows the tail of the log
        traceback.print_exc()
        sys.exit(1)

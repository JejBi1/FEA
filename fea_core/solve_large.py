"""Linear path outside ccx: the own matrix assembly (`fea_core.assembly`) + the direct solver (`fea_core.direct`, one factorization for all
load states) or hypre (parallel PCG with AMG for elasticity, for models that do not fit into the memory of the direct solver).

The same tet10 mesh and the same loads as for ccx (loads.nodal_loads), the same evaluation of reactions and the same equilibrium check
(solve.make_step). Bolts (beam elements between rigid bearing faces, the preload as unit states on the same factorization – `joints.own`, direct solver only) and
welds (the forces of the segments from the stresses at the integration points – `recovery.element_ip_stress`) are part of it since F13 step 1b, part 2. The assembly, the direct solver and hypre run in separate processes (the backend keys `ngs_direct` / `ngs_hypre` are historical:
the matrix was assembled by NGSolve until F13 step 1a; FEA_ASSEMBLY=ngsolve brings that assembly back for one release).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, hypre
from .load_kinds import load_handler
from .loads import nodal_loads
from .mesh import Mesh
from .study import Study, Displacement, Fixed, LoadCase


class LargeSolverError(RuntimeError):
    pass


class MeshQualityError(LargeSolverError):
    """The mesh itself cannot be solved: elements with a non-positive Jacobian determinant (inverted, flat, damaged). CalculiX rejects the same mesh
    ("nonpositive jacobian"), so `auto` does NOT continue with another solver – it would only hide the cause behind a CalculiX error."""


def mesh_quality_message(err, mesh: Mesh) -> str:
    """The message for `assembly.ElementError` with the part and the place of the first elements (the assembly worker knows only the numbers)."""
    rows = []
    for e in err.elements[:5]:
        part = mesh.part_names[int(mesh.tet_part[e])] if 0 <= e < len(mesh.tet_part) else "?"
        c = mesh.nodes[mesh.tets[e, :4]].mean(axis=0)
        rows.append(f"{e + 1}: {part} ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})")
    more = f" (and {err.count - len(rows):,} more)" if err.count > len(rows) else ""
    return (f"The mesh cannot be solved: {err.count:,} element(s) have a Jacobian determinant that is not positive and finite at an integration point "
            f"(inverted, flat or damaged elements). CalculiX rejects them as well (\"nonpositive jacobian\"), so no other solver is tried. "
            f"First elements (CalculiX element number: part, centre in mm): {'; '.join(rows)}{more}. Generate the mesh again (another element size) "
            f"or correct the geometry.")


class DirectUnavailable(LargeSolverError):
    """The direct solver does not fit into the memory (or is not installed): "auto" continues with the iterative solver."""


def unsupported(study: Study, solver: str = "direct") -> list[str]:
    """What the own assembly with the given solver ("direct" | "hypre") cannot do yet (then ccx is used)."""
    from .load_kinds import CONSTRAINT_HANDLERS
    out = []
    for lc in study.load_cases:                    # per load kind: fea_core.load_kinds (ngs_unsupported)
        for l in lc.loads:
            h = load_handler(l)
            why = h.ngs_unsupported(l, lc) if h is not None and h.ngs_unsupported is not None else None
            if why:
                out.append(why)
    kinds = {getattr(c, "kind", "") for c in study.constraints}
    for k, ch in CONSTRAINT_HANDLERS.items():      # constraint kinds in registry order (cylindrical, remote, elastic)
        if ch.ngs_unsupported and k in kinds:
            out.append(ch.ngs_unsupported)
    from .linear_constraints import unsupported_reasons
    out += unsupported_reasons(study, solver)      # what only the direct solver can do (constraints that couple several nodes)
    if study.frame.active:
        out.append("beam structure (ccx U1 elements)")
    from .ortho import has_ortho
    if has_ortho(study) and _has_print(study):
        out.append("orthotropic material together with 3D printing (ccx)")
    if study.is_nonlinear():
        out.append("contact / nonlinear analysis (ccx only)")
    if getattr(study, "cyclic", None) is not None and study.cyclic.enabled:
        out.append("cyclic symmetry (ccx constraint equations)")
    return list(dict.fromkeys(out))


def _has_print(study: Study) -> bool:
    return any(getattr(m, "is_print", False) for m in study.materials.values())


# hypre memory (8 processes, tet10 elasticity), measured 12. 10. 2026: ~0.6 GB + 4.8 GB / mil. DOF
# default settings, ~0.6 GB + 3.4 GB / mil. DOF with aggressive coarsening (2.2× more iterations)
MEM_FAST = (0.6, 4.8)
MEM_LEAN = (0.6, 3.4)


def amg_for_memory(ndof: int) -> tuple[dict, str]:
    """Fast AMG setting if it fits into the free memory, otherwise the economical one; otherwise a clear error."""
    import psutil
    free = psutil.virtual_memory().available / 2**30
    mdof = ndof / 1e6
    fast = MEM_FAST[0] + MEM_FAST[1] * mdof
    lean = MEM_LEAN[0] + MEM_LEAN[1] * mdof
    if fast < 0.85 * free:
        return {}, f"fast mode, memory estimate {fast:.1f} GB"
    if lean < 0.85 * free:
        return {"agg_levels": 1}, f"memory-saving mode, memory estimate {lean:.1f} GB (available {free:.1f} GB)"
    raise LargeSolverError(
        f"Not enough free memory for the analysis: estimate {lean:.1f} GB, available {free:.1f} GB. Close other applications or coarsen the mesh.")


def iso_stiffness(E: float, nu: float) -> np.ndarray:
    """Isotropic C (6×6), Voigt xx yy zz xy xz yz, engineering shear."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


def element_stiffness_table(study: Study, mesh: Mesh) -> tuple[np.ndarray, np.ndarray] | None:
    """(table (P + G, 6, 6), index (E,) int32) of the matrix C of the elements if the mesh contains printed parts (otherwise None – the isotropic
    path per part): the first P matrices are the isotropic ones of the parts, then the G groups of printed elements (global axes); C of the element e
    = table[index[e]]. The own assembly takes the table, the stress recovery the full array (`element_stiffness_all`)."""
    from .printing import element_stiffness_groups
    pr = element_stiffness_groups(study, mesh)
    if pr is None:
        return None
    el, gid, groups = pr
    n_parts = len(mesh.part_names)
    iso = np.array([iso_stiffness(study.material_of(p).E, study.material_of(p).nu) for p in mesh.part_names]).reshape(n_parts, 6, 6)
    idx = np.ascontiguousarray(mesh.tet_part, dtype=np.int32).copy()
    idx[el] = n_parts + gid
    return np.concatenate([iso, groups]), idx


def element_stiffness_all(study: Study, mesh: Mesh) -> np.ndarray | None:
    """C (E, 6, 6) of all elements if the mesh contains printed parts (otherwise None – the isotropic path per part)."""
    ti = element_stiffness_table(study, mesh)
    return None if ti is None else ti[0][ti[1]]


def _worker(stage: str, job: Path, log, backend: str = "ngs_hypre", cancel: threading.Event | None = None,
            extra: tuple = ()) -> None:
    """The NGSolve worker (the earlier assembly, used only with FEA_ASSEMBLY=ngsolve) in a separate process (cancellable – the process tree is
    killed, output in ngs_<stage>.log)."""
    from .procs import run_tree, tail, Canceled
    log_path = Path(job) / f"ngs_{stage}.log"
    try:                                     # run_tree releases the unused memory of this process before the launch
        rc = run_tree([sys.executable, "-m", "fea_core.ngs.worker", stage, str(job), *extra], log_path, cancel)
    except Canceled:
        raise LargeSolverError("Analysis canceled by user") from None
    if rc != 0:
        out = tail(log_path)
        from .solvers import import_failure
        msg = import_failure(backend, out)        # a missing or broken module (an installation without the module) – not a traceback printout
        raise LargeSolverError(msg or f"NGSolve ({stage}) failed:\n{out}")


def run_ngs_hypre(study: Study, mesh: Mesh, workdir: Path, warnings: list[str], log,
                  cancel: threading.Event | None = None, thermal: list | None = None):
    return run_ngs(study, mesh, workdir, warnings, log, cancel, thermal, "hypre")


def run_ngs(study: Study, mesh: Mesh, workdir: Path, warnings: list[str], log,
            cancel: threading.Event | None = None, thermal: list | None = None, solver: str = "hypre"):
    """The own assembly builds the system, it is solved by hypre (AMG, iterative, per state) or by the direct solver (`fea_core.direct`,
    ONE factorisation for all states). Stresses and nodal forces from the displacements as in ccx (`recovery.recover`).
    thermal = results of the thermal states (ThermalResult) for the thermal load.
    The large system files are deleted at the end also after an error or cancellation (FEA_KEEP_SOLVER_FILES=1 keeps them).
    """
    bad = unsupported(study, solver)
    if bad:
        raise LargeSolverError("The direct / iterative solver cannot do this study yet: " + ", ".join(bad) + " – use CalculiX.")
    from .solvers import backend_error
    err = backend_error(f"ngs_{solver}")          # missing numba / rslab / hypre / MS-MPI → what to install
    if err:
        if solver == "direct" and study.solver.backend == "auto":
            raise DirectUnavailable(err)
        raise LargeSolverError(err)
    job = Path(workdir) / "ngs"
    job.mkdir(parents=True, exist_ok=True)
    _remove_solver_files(job)
    keep = keep_solver_files()
    from .loads import nodal_cache
    try:
        with nodal_cache():                       # the nodal loads of a state are integrated once (F.npy and make_step)
            return _run_ngs_job(study, mesh, job, warnings, log, cancel, thermal, solver)
    except BaseException:
        if keep:                                  # e.g. hypre did not converge – the system stays for diagnostics
            log(f"Solver files kept for diagnostics (FEA_KEEP_SOLVER_FILES): {job}")
        raise
    finally:
        if not keep:
            _remove_solver_files(job)


KEEP_FILES_ENV = "FEA_KEEP_SOLVER_FILES"


def keep_solver_files() -> bool:
    """FEA_KEEP_SOLVER_FILES=1 keeps the system files of the assembly / hypre / the direct solver (large .npy) after the calculation."""
    import os
    return os.environ.get(KEEP_FILES_ENV, "").strip().lower() in ("1", "true", "yes")


def _remove_solver_files(job: Path) -> None:
    """Deletes the large temporary files of the job (the input arrays, the matrix, right-hand sides, solutions, mesh copy)."""
    for f in [*job.glob("*.npy"), *(job / "hypre").glob("*.npy"), *(job / "direct").glob("*.npy"), job / "mesh.npz", job / "reduction.npz", job / "aniso.npz", job / "mass.npz"]:
        try:
            f.unlink(missing_ok=True)
        except OSError:                           # still open by a killed process (Windows) – a later run deletes it
            pass


def _times_of_assembly(ainfo: dict) -> dict:
    if ainfo.get("assembler") == "own":
        tc = ainfo.get("t_compile") or 0.0          # the warm-up is ≈ 0.5 s when the kernels load from the cache; longer = the first-use compilation
        return {"compile": tc if tc > 2.0 else None, "assembly": ainfo.get("t_assemble"), "assembly_process": ainfo.get("t_total")}
    return {"mesh_import": ainfo.get("t_mesh"), "assembly": ainfo.get("t_assemble"), "assembly_process": ainfo.get("t_total")}


def _assembler_text(ainfo: dict) -> str:
    """What assembled the matrix (the solver protocol): the own assembly with its threads, or the NGSolve fallback."""
    if ainfo.get("assembler") == "own":
        return f"own (numba, {ainfo.get('threads', '?')} threads)"
    return "NGSolve"


def direct_protocol(study: Study, info: dict, ainfo: dict, ndof: int) -> dict:
    """Solver protocol (Results.info["solver"]) of the direct solver: what was chosen and why, sizes, memory plan against the measured
    peak, time per phase, accuracy per load state. The reason of choosing the backend is added by the caller (`solve._run_mechanics`).
    """
    prec = info["precision"]
    names = [lc.name for lc in study.load_cases]
    return {
        "backend": "ngs_direct", "engine": f"{info['engine']} {info['version']}", "ndof": ndof, "nnz": ainfo["nnz"],
        "matrix_format": ainfo.get("format", "upper"), "assembler": _assembler_text(ainfo), "factor_nnz": info["factor_nnz"],
        "fill_per_dof": info["fill_per_dof"],
        "precision": prec, "precision_reason": info["precision_reason"], "threads": info["threads"],
        "memory": {"estimate_gb": info["plan_gb"][prec], "growth_gb": info.get("growth_gb"), "peak_gb": info["peak_gb"],
                   "free_gb": info["free_gb"]},
        "times": {**_times_of_assembly(ainfo), "load": info.get("t_load"), "analysis": info["t_analyze"], "factorization": info["t_factor"],
                  "solution": info["t_solve"]},
        "cases": [{"name": nm, "backward_error": info["backward_error"][i], "rel_residual": info["rel_residual"][i],
                   "iterations": info["iterations"][i]} for i, nm in enumerate(names)],
        "inertia": info.get("inertia"),
    }


def hypre_protocol(study: Study, info: dict, ainfo: dict, ndof: int, nproc: int, mode: str) -> dict:
    """Solver protocol of the iterative solver (hypre)."""
    names = [lc.name for lc in study.load_cases]
    return {
        "backend": "ngs_hypre", "engine": "hypre BoomerAMG + PCG", "ndof": ndof, "nnz": ainfo["nnz"], "matrix_format": "full",
        "assembler": _assembler_text(ainfo), "processes": nproc, "amg_mode": mode,
        "memory": {"peak_gb": info.get("peak_gb_total")},
        "times": {**_times_of_assembly(ainfo), "setup": info.get("t_setup")},
        "cases": [{"name": nm, "rel_residual": ci["rel_residual"], "iterations": ci["iterations"]}
                  for nm, ci in zip(names, info["cases"])],
    }


def _assemble(job: Path, solver: str, log, cancel, threads: int | None = None, hom_cols: int = 0) -> dict:
    """The assembly builds the system into the files of the solver (the whole matrix for hypre, the upper triangle for the direct solver) in a
    separate process; the main process does not load it. Returns assemble.json (sizes, times)."""
    from . import assembly
    if assembly.assembler() == "ngsolve":
        log("Assembling stiffness matrix (NGSolve)")
        _worker("assemble", job, log, f"ngs_{solver}", cancel, (solver,))
        ainfo = json.loads((job / "assemble.json").read_text(encoding="utf-8"))
        ainfo.setdefault("assembler", "ngsolve")
    else:
        log("Assembling stiffness matrix")
        try:
            ainfo = assembly.run_worker(job, solver, threads, cancel, on_notice=log, hom_cols=hom_cols)
        except assembly.ElementError:             # a mesh problem: `_run_ngs_job` names the parts (MeshQualityError)
            raise
        except RuntimeError as e:
            raise LargeSolverError(str(e)) from None
    if ainfo["rigid_body_residual"] > 1e-8:
        raise LargeSolverError(f"The stiffness matrix failed the rigid-body motion check ({ainfo['rigid_body_residual']:.2e})")
    if cancel is not None and cancel.is_set():
        raise LargeSolverError("Analysis canceled by user")
    return ainfo


def _assemble_and_solve_direct(study: Study, job: Path, neq: str, log, cancel, hom_cols: int = 0):
    """Assembly + the direct solver (exclusive in the program, see `direct.exclusive`) → (X, info of the worker, assemble.json)."""
    from . import direct
    from .ccx import parallel
    threads = parallel.cap_threads(study.solver.nproc) or parallel.budget_threads()      # beside ccx side jobs (side_jobs) the direct solver has its share of the cores
    ainfo = _assemble(job, "direct", log, cancel, threads, hom_cols)
    est = direct.estimate_gb(int(ainfo["ndof"]))
    log(f"Solving the system of {neq} equations directly ({direct.engine_name()}, one factorization for {len(study.load_cases)} states, "
        f"memory estimate {est:.1f} GB before the analysis)")
    X, info = direct.run_prepared(job / "direct", threads=threads, cancel=cancel)
    log(f"  analysis {info['t_analyze']:.1f} s, factorization {info['t_factor']:.1f} s ({info['factor_nnz'] / 1e6:.0f} M nonzeros, "
        f"{info['precision']}), solution {info['t_solve']:.1f} s"
        + (f", {max(info['iterations'])} refinement iterations" if info["precision"] == "float32" else "")
        + f", backward error up to {max(info['backward_error']):.1e}, memory {info.get('growth_gb', info['peak_gb']):.1f} GB "
          f"(plan {info['plan_gb'][info['precision']]:.1f} GB)")
    return X, info, ainfo


@dataclass
class ModelFiles:
    """What `write_model_files` found: the constraint plan of the own assembly, whether the study has orthotropic parts, the table of the matrices C of 3D printing
    (`element_stiffness_table`) and the per-element C of the stress recovery (None = the isotropic path per part)."""
    plan: object
    ortho_on: bool
    ti: object
    pr: object

    @property
    def nx(self) -> int:
        return 0 if self.plan is None else self.plan.red.n_extra


def write_model_files(study: Study, mesh: Mesh, job: Path, solver: str) -> ModelFiles:
    """The model files of the job folder that every run of the own assembly reads (materials, mesh, the table of matrices C, orthotropy, the constraints of the
    reduction) – the static solutions and the eigenvalue analysis of the direct solver write the same ones."""
    from . import ortho as ortho_mod, linear_constraints
    from .assembly.reduction import ReductionError
    from . import assembly
    mats = [{"E": study.material_of(p).E, "nu": study.material_of(p).nu, "alpha": study.material_of(p).alpha or 0.0}
            for p in mesh.part_names]
    (job / "materials.json").write_text(json.dumps(mats), encoding="utf-8")
    ti = element_stiffness_table(study, mesh)
    pr = None if ti is None else ti[0][ti[1]]  # 3D printing: C per element in global axes (the same groups as in ccx) – for the stress recovery
    ortho_on = ortho_mod.has_ortho(study, mesh)
    try:                                      # the constraints of the own assembly: springs, dependent DOFs, extra nodes (rigid couplings)
        plan = linear_constraints.prepare(study, mesh, solver)
    except ReductionError as e:
        raise LargeSolverError(f"The constraints cannot be eliminated: {e}") from None
    if (plan is not None or ortho_on) and assembly.assembler() != "own":
        raise LargeSolverError("The NGSolve assembly (FEA_ASSEMBLY=ngsolve) cannot do constraint equations, springs or orthotropic materials")
    if assembly.assembler() == "own":
        np.save(job / "nodes.npy", mesh.nodes)
        np.save(job / "tets.npy", np.ascontiguousarray(mesh.tets, dtype=np.int32))
        np.save(job / "tet_part.npy", np.ascontiguousarray(mesh.tet_part, dtype=np.int32))
        if ti is not None:                    # the table of the matrices C and the index of every element (a few groups instead of an array per element)
            np.save(job / "C_tab.npy", ti[0])
            np.save(job / "C_idx.npy", ti[1])
        elif ortho_on:                        # orthotropic parts: the table of the parts (C in the global axes or the local axes of a cylindrical system)
            ents = ortho_mod.part_entries(study, mesh)
            np.save(job / "C_tab.npy", np.stack([e[0] for e in ents]))
            np.save(job / "C_idx.npy", np.ascontiguousarray(mesh.tet_part, dtype=np.int32))
            np.savez(job / "aniso.npz", kind=np.array([e[1] for e in ents], np.int8), ax=np.stack([e[2] for e in ents]), A=np.stack([e[3] for e in ents]))
        if plan is not None and not plan.red.empty:
            plan.red.save(job)
    else:                                     # the NGSolve worker (FEA_ASSEMBLY=ngsolve)
        mesh.save(job / "mesh.npz")
        if pr is not None:
            np.save(job / "C_el.npy", pr)
    return ModelFiles(plan, ortho_on, ti, pr)


def constraint_arrays(study: Study, mesh: Mesh, nx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(fixed (3N + 3·nx,) bool, g (3N + 3·nx,)) of the Dirichlet constraints (Fixed, Displacement) + the extra DOFs (their constrained components are in
    reduction.npz, so False / 0 here)."""
    n = len(mesh.nodes)
    fixed = np.zeros((n, 3), bool)
    gval = np.zeros((n, 3))
    for c in study.constraints:
        idx = mesh.surface_nodes(c.selection)
        if isinstance(c, Fixed):
            fixed[idx] = True
        elif isinstance(c, Displacement):
            for d, v in enumerate((c.ux, c.uy, c.uz)):
                if v is not None:
                    fixed[idx, d] = True
                    gval[idx, d] = v
    return np.concatenate([fixed.ravel(), np.zeros(3 * nx, bool)]), np.concatenate([gval.ravel(), np.zeros(3 * nx)])


def load_vector(study: Study, mesh: Mesh, lc, plan, nx: int = 0, temp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(the load vector of a state for the own assembly (3N + 3·nx,), the nodal forces of the mesh (N, 3)): consistent nodal loads, the distribution of a distributing
    remote load, the weight of a point mass with a distributing coupling on the nodes of its face, and the load of a rigid remote load on its extra DOFs
    (with `temp` = the node temperatures of the state also the thermal elongation of the bolts)."""
    f = nodal_loads(study, mesh, lc)[0]
    for l in lc.loads:                         # e.g. distributing remote load = distribution of F and M to the nodes of the face
        h = load_handler(l)
        if h is not None and h.ngs_nodal is not None:
            f = f + h.ngs_nodal(study, mesh, l)
    if plan is not None:
        fm = plan.mass_nodal(study, lc)           # the weight of a point mass with a distributing coupling: on the nodes of its face
        if fm is not None:
            f = f + fm
    fe = f.ravel()
    if plan is not None and nx:                # a rigid remote load acts on the extra DOFs of its rigid body, the thermal elongation of a bolt too
        fe = np.concatenate([fe, plan.extra_loads(study, lc) + plan.thermal_elongation(temp, study.thermal.reference_temperature)])
    return fe, f


def _run_ngs_job(study: Study, mesh: Mesh, job: Path, warnings: list[str], log, cancel, thermal, solver: str):
    """The body of run_ngs in the prepared job folder (the files are removed by the caller)."""
    from .results import Results
    from .solve import constraint_masks, make_step
    from .recovery import recover
    from . import assembly
    from . import ortho as ortho_mod
    mf = write_model_files(study, mesh, job, solver)
    plan, ortho_on, pr, nx = mf.plan, mf.ortho_on, mf.pr, mf.nx
    t0 = time.time()
    n = len(mesh.nodes)
    # constraints (the same masks as in ccx) and prescribed values
    masks = constraint_masks(study, mesh, own=True)
    fixed_all, g_all = constraint_arrays(study, mesh, nx)
    t_ref = study.thermal.reference_temperature
    temps = [None] * len(study.load_cases)
    if study.has_thermal_loads():                 # thermal load: ΔT per node, the forces are computed by the assembly
        from .thermal.field import node_temperatures
        tmap = {t.name: t for t in (thermal or [])}
        temps = [node_temperatures(study, mesh, lc, tmap) for lc in study.load_cases]
    geom = joint_elems = None
    if study.active_bolts() or study.active_welds():       # the geometry of the joints from the mesh (the same as for ccx): the forces of the bolts and welds
        from .joints.evaluate import joint_geometry, required_elements
        geom = joint_geometry(study, mesh)
        warnings.extend(w for w in geom["warnings"] if w not in warnings)
        joint_elems = required_elements(geom)
    Fs, fnodes = [], []
    for k, lc in enumerate(study.load_cases):
        fe, f = load_vector(study, mesh, lc, plan, nx, temps[k])
        fnodes.append(f)
        Fs.append(fe)
    n_units = 0
    if plan is not None and study.joints.preload_case and plan.cut_bolts():    # the preload of the bolts: one unit state per bolt on the same factorization
        U0 = plan.unit_loads()
        n_units = U0.shape[1]
        Fs.extend(np.concatenate([np.zeros(3 * n), U0[:, j]]) for j in range(n_units))
    np.save(job / "F.npy", np.column_stack(Fs))
    if study.has_thermal_loads():
        np.save(job / "dT.npy", np.column_stack([np.zeros(n) if t is None else t - t_ref for t in temps]))
    np.save(job / "fixed.npy", fixed_all)           # (the constrained components of the extra DOFs are in reduction.npz)
    np.save(job / "g.npy", g_all)
    del Fs
    neq = f"{3 * n:,}".replace(",", " ")          # thousands separator only in the number (the commas of the text stay)
    try:
        if solver == "direct":
            from . import direct
            try:
                with direct.exclusive(cancel, log):   # one direct run at a time in this program: they would share the same free memory
                    X, info, ainfo = _assemble_and_solve_direct(study, job, neq, log, cancel, n_units)
            except direct.DirectMemoryError as e:
                if study.solver.backend == "auto":
                    raise DirectUnavailable(str(e)) from None
                raise LargeSolverError(str(e)) from None
            except direct.DirectError as e:
                raise LargeSolverError(str(e)) from None
            protocol = direct_protocol(study, info, ainfo, 3 * n)
        else:
            from .ccx.parallel import cap_threads            # FEA_THREADS also caps the MPI processes of hypre
            nproc = cap_threads(study.solver.nproc) or config.NUM_THREADS
            ainfo = _assemble(job, solver, log, cancel, nproc)
            amg, mode = amg_for_memory(3 * n)
            log(f"Solving a system of {neq} equations (hypre BoomerAMG, {nproc} processes, {mode})")
            try:
                X, info = hypre.run_prepared(job / "hypre", nproc=nproc, amg=amg, cancel=cancel)
            except hypre.HypreError as e:
                raise LargeSolverError(str(e)) from None
            info["amg_mode"] = mode
            for c, ci in zip(study.load_cases, info["cases"]):
                log(f"  {c.name}: {ci['iterations']} iterations, rel. residual {ci['rel_residual']:.1e}")
            protocol = hypre_protocol(study, info, ainfo, 3 * n, nproc, mode)
    except assembly.ElementError as e:
        raise MeshQualityError(mesh_quality_message(e, mesh)) from None
    if cancel is not None and cancel.is_set():
        raise LargeSolverError("Analysis canceled by user")
    log("Computing stresses (from displacements like ccx)")
    C_el = pr if pr is not None else None
    field = ortho_mod.ElementField(study, mesh) if ortho_on else None
    Xf = plan.red.expand(X) if plan is not None else X       # the dependent DOFs (cylindrical support, rigid / distributing coupling, *TIE, bolt faces) from their masters
    ns = len(study.load_cases)
    cases = [(lc, Xf[:, k], temps[k], fnodes[k], 0.0) for k, lc in enumerate(study.load_cases)]
    preload_gaps_mm = None
    if n_units:                                              # the preload state: the elongations g of the bolts that give N = F_V (joints.own)
        from .joints.own import preload_gaps
        from .joints.model import PRELOAD_CASE
        F_V = np.array([bp.chain.bolt.preload for bp in plan.cut_bolts()])
        try:
            preload_gaps_mm = preload_gaps(plan.bolt_matrix(Xf[:, ns:]), F_V)
        except ValueError as e:
            raise LargeSolverError(str(e)) from None
        cases.append((LoadCase(PRELOAD_CASE, []), Xf[:, ns:] @ preload_gaps_mm, None, np.zeros((n, 3)),
                      float(sum(bp.chain.bolt.preload for bp in plan.bolts))))
    steps = []
    for lc, x, T, f_app, ref in cases:
        U = x[:3 * n].reshape(n, 3)
        S, RF = recover(study, mesh, U, None if T is None else T - t_ref, C_el, field)
        if T is not None:
            from .thermal.field import force_scale
            ref = max(ref, force_scale(study, mesh, T))
        remote = None
        if plan is not None and plan.support_pairs:   # rigid remote support: the point moves with the extra nodes, the reaction = Σ(RF − F) of its face
            remote = {sel: {"u": mo[:3], "rot": mo[3:], "own": True} for sel, mo in plan.motion(x).items()}
        st = make_step(study, mesh, lc, U, S, RF, masks, ref_force=ref, remote=remote, f_applied=f_app)
        if T is not None:
            st.fields["T"] = T
        if geom is not None:                          # the forces of the bolts and welds: RF − F at the bearing faces, the stresses at the integration points
            from .joints.evaluate import step_forces, to_json
            from .recovery import element_ip_stress
            sig_e = element_ip_stress(study, mesh, U, joint_elems, None if T is None else T - t_ref, C_el, field)
            sig = {int(e) + 1: sig_e[i] for i, e in enumerate(joint_elems)}
            st.joints = to_json(step_forces(study, mesh, geom, RF, f_app, sig))
        steps.append(st)
    del X, Xf, fnodes
    if plan is not None:
        protocol["constraints"] = {**plan.red.labels, "dependent_dofs": int(plan.red.n_dep), "extra_nodes": int(plan.red.n_extra)}
        for w in plan.warnings:
            if w not in warnings:
                warnings.append(w)
    if plan is not None and plan.bolts:
        protocol["bolts"] = {"count": len(plan.bolts), "beam_elements": int(plan.red.n_elem), "with_cut": len(plan.cut_bolts()),
                             "preload_states": int(n_units)}
        if preload_gaps_mm is not None:
            protocol["bolts"]["preload_elongation_mm"] = [float(v) for v in preload_gaps_mm]
    if ortho_on:
        protocol["orthotropic"] = True
    info.update({"assemble": ainfo, "t_total": time.time() - t0})
    protocol["times"]["recovery_and_total"] = time.time() - t0
    out = {"backend": f"ngs_{solver}", solver: info, "solver": protocol}
    if geom is not None:                              # the same structure as the ccx path (`solve._run_mechanics`): the assessment, the report and the GUI read it
        from .joints.evaluate import joints_info
        out["joints"] = joints_info(geom, n_units > 0)
    from .solve import print_results
    print_results(study, mesh, steps, out, warnings, log)
    return Results(mesh, steps, warnings, out)

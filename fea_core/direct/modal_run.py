"""Natural frequencies and the modal basis of the dynamics on the direct path – the side of the main program.

The chain (every heavy step is a separate process, nothing large stays in this one except the modes that the results hold anyway):
  mass + stiffness   `assembly.worker` assembles K and the consistent mass matrix M on one pattern (numba), with the constraints of `linear_constraints` eliminated
                     (TᵀKT, TᵀMT) and the added masses of the study (`assembly.mass`)
  eigenvalues        `direct.eigen_worker` factors K + s·M with rslab, iterates with ARPACK (SciPy), proves the completeness by the Sturm count (`direct.eigen`)
  here               the dependent DOFs are expanded, the participation factors Γ = φᵀ·M·R and the effective masses come from the rigid-body motions R of the
                     assembly (M·R is written by the worker), the static solutions of the dynamics from the SAME factor
Same results as the CalculiX path (`eigen.run_eigen`, `dynamics.modal.run_basis`) – the engines are compared by `cross_check`.

Routing (`decide`): the direct path computes every study whose stiffness the own assembly builds, where it is at least as fast (`min_work`: a small problem keeps CalculiX); CalculiX stays for what needs a geometric stiffness (buckling,
prestressed modal analysis, Campbell), cyclic symmetry, bolts (their beam elements have no mass matrix) and frames, contact / nonlinear states, shells and 2D – and whenever the user chooses it
(`Study.solver.modal_engine`, FEA_MODAL_ENGINE) or the direct solver is not installed / does not fit into the memory. A failed direct analysis under `auto` continues
in CalculiX with the reason (never a silently missing mode).
"""
from __future__ import annotations

import gc
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODE_BLOCK = 16                          # modes expanded / copied at a time (the main process holds no second full copy of the modes)
ENGINE_ENV = "FEA_MODAL_ENGINE"          # auto | direct | ccx (tests, diagnostics)
ENGINES = ("auto", "direct", "ccx")
MIN_WORK_ENV = "FEA_MODAL_MIN_WORK"      # overrides MIN_WORK (0 = the direct path whenever it can run: tests, gates)
MIN_WORK = 7e5                           # DOF·(modes + 10) below which `auto` keeps CalculiX – see `min_work`


class ModalUnsupported(RuntimeError):
    """The direct eigenvalue analysis cannot be used for this study / mesh (the reason is the message)."""


def engine_setting(study) -> str:
    """auto | direct | ccx: the environment variable overrides the setting of the study (`Study.solver.modal_engine`)."""
    v = os.environ.get(ENGINE_ENV, "").strip().lower() or str(getattr(study.solver, "modal_engine", "auto") or "auto").lower()
    return v if v in ENGINES else "auto"


def min_work() -> float:
    """The size DOF·(modes + 10) from which `auto` takes the direct path. Measured on this machine (4 threads, whole `run_eigen` on the same mesh, a cantilever, the
    first 10 / 20 / 100 modes): the direct path has a fixed start of about 6 s (the assembly process loads the compiled kernels, the eigen process rslab and SciPy) and then
    grows slowly, CalculiX starts at once and grows faster with the size and much faster with the number of modes – 11.6 k DOF: 2.1 / 3.0 / 15.4 s (ccx) against
    6.6 / 7.4 / 9.8 s; 19.9 k DOF: 3.8 / 6.0 / 26.8 against 8.0 / 8.2 / 13.6; 36.7 k DOF: 8.1 / 12.6 / 91 against 8.4 / 9.2 / 19.5; 66 k DOF 20 modes 34.6 against 18.8 s,
    100 modes 355 against 37 s. Break-even DOF·(modes + 10) ≈ 6–7·10⁵ in all three columns. Environment `FEA_MODAL_MIN_WORK` overrides it."""
    try:
        return float(os.environ.get(MIN_WORK_ENV, "").strip() or MIN_WORK)
    except ValueError:
        return MIN_WORK


def unsupported(study, dynamics: bool = False) -> list[str]:
    """What the direct eigenvalue analysis cannot do for this study (CalculiX then): what the statics of the own assembly cannot do (beam elements of frames,
    contact, cyclic symmetry, orthotropy with printing, …), the bolts (their beam elements have no mass matrix; the statics take them since F13 step 1b) and the
    analyses that need the geometric stiffness. Welds are only meshed bodies (the root contact is opened in the mesh) – their mass and stiffness are the mesh's. dynamics = also what the static solutions of the
    dynamics need (the loads of the excitation cases)."""
    import copy
    from .. import assembly
    from ..solve_large import unsupported as ngs_unsupported
    st = study
    if not dynamics:
        st = copy.copy(study)
        st.load_cases = []                        # the load cases do not enter a free-vibration analysis
    why = list(ngs_unsupported(st, "direct"))
    if study.active_bolts():                      # the statics take bolts since F13 step 1b: the beam elements of the own assembly have a stiffness but no mass matrix
        why.append("bolts (the beam elements of the own assembly have no mass matrix – CalculiX gives them the density of steel)")
    if assembly.assembler() != "own":
        why.append("the NGSolve assembly (FEA_ASSEMBLY=ngsolve) has no mass matrix")
    if dynamics:
        pr = study.dynamics.prestress
        if pr.enabled and pr.case:
            why.append("prestressed modal analysis (stress stiffness – CalculiX)")
        if pr.enabled and pr.rotating and pr.rpm_list:
            why.append("Campbell diagram (centrifugal stiffening – CalculiX)")
    return list(dict.fromkeys(why))


def decide(study, mesh, dynamics: bool = False, modes: int | None = None) -> tuple[str, str]:
    """("direct" | "ccx", reason) for the natural frequencies / the modal basis of the study on this mesh. `Study.solver.modal_engine` = "direct" turns a refusal into
    `ModalUnsupported`; "auto" gives CalculiX with the reason."""
    return decide_full(study, mesh, dynamics, modes)[:2]


def decide_full(study, mesh, dynamics: bool = False, modes: int | None = None) -> tuple[str, str, bool]:
    """(`decide`, direct_possible): True when "ccx" was chosen only by the size rule of `auto` – the direct analysis could do the study, so it takes over when
    CalculiX fails (an error of ccx on *TIE with a normal elastic support, an access violation on a loaded machine)."""
    setting = engine_setting(study)
    if setting == "ccx":
        return "ccx", "chosen by the user", False
    if setting == "auto" and study.solver.backend == "ccx":
        return "ccx", "the solver of the study is CalculiX", False

    def refuse(text: str):
        if setting == "direct":
            raise ModalUnsupported(f"The direct eigenvalue analysis cannot be used: {text}")
        return "ccx", text, False

    why = unsupported(study, dynamics)
    if why:
        return refuse("CalculiX only: " + ", ".join(why))
    from ..solvers import direct_missing
    miss = direct_missing()
    if miss:
        return refuse("the direct solver is not installed – missing " + ", ".join(miss))
    from .. import linear_constraints
    refused = linear_constraints.check(study, mesh)
    if refused:
        return refuse("CalculiX only: " + "; ".join(refused))
    from . import eigen_vectors_gb, estimate_gb, reserve_gb
    import psutil
    ndof = 3 * len(mesh.nodes)
    k = int(modes or (study.dynamics.modes if dynamics else study.modal.modes))
    need = estimate_gb(ndof, "float32") * 1.7 + eigen_vectors_gb(ndof, k)               # a float64 factor + the vectors of the analysis (the estimate the worker checks too)
    free = psutil.virtual_memory().available / 2**30
    if need > 0.85 * free - reserve_gb():
        return refuse(f"the direct eigenvalue analysis would need about {need:.1f} GB, free {free:.1f} GB")
    work = ndof * (k + 10)
    if setting == "auto" and work < min_work():
        return "ccx", (f"a small problem ({ndof} DOF, {k} modes): CalculiX is at least as fast – the direct analysis starts two processes (about 6 s), "
                       f"it pays from DOF·(modes + 10) = {min_work():.0e}, here {work:.1e}"), True
    return "direct", f"linear free-vibration analysis – shift-invert Lanczos over one rslab factor (memory estimate {need:.1f} GB, free {free:.1f} GB)", True


@dataclass
class DirectModes:
    lam: np.ndarray                      # (m,) ω² [1/s²] (the Rayleigh quotients; rigid-body modes ≈ 0 ± noise)
    shapes: np.ndarray                   # (m, N, 3) float64, φᵀMφ = 1, the dependent DOFs expanded
    gamma: np.ndarray                    # (m, 6) participation factors (x y z rx ry rz about the origin), Γ = φᵀ·M·R
    total_free: np.ndarray               # (6,) the effective total mass of the free DOFs (rᵀ M r, constrained DOFs left out – like ccx)
    mass_bodies: float                   # [t] the mass of the body elements
    mass_added: float                    # [t] the added masses
    rigid: np.ndarray = None                           # (m,) bool: the rigid-body motions of a free body
    statics: list = field(default_factory=list)        # the static solutions (N, 3) of the requested cases (from the same factor)
    statics_extra: list = field(default_factory=list)  # … and their full vectors (3·(N + extra nodes)) – for the motion of the extra nodes
    info: dict = field(default_factory=dict)
    plan: object = None
    cache: object = None                               # the folder of the state cache entry that holds this analysis (`modal_cache`)
    columns_all: list = field(default_factory=list)    # the right-hand-side columns of the entry: the 3 unit accelerations + one per non-gravity case
    coeffs_all: list = field(default_factory=list)     # per requested static case: its coefficients over `columns_all`
    cols: Any = None                                # (columns, 3·(N + extra nodes)) the solutions of the columns
    content: Any = None                             # the content hash of the stored modes (`state_cache.content_of`): what a recovery stored beside them is tied to

    @property
    def freq(self) -> np.ndarray:
        """Frequencies [Hz]; a rigid-body mode of a free body (its strain energy is rounding noise, `rigid`) and any negative λ give 0."""
        lam = np.where(self.rigid, 0.0, self.lam)
        return np.sqrt(np.maximum(lam, 0.0)) / (2.0 * np.pi)

    @property
    def eff_mass(self) -> np.ndarray:
        return self.gamma ** 2


def _clean(job: Path) -> None:
    for f in list(job.glob("*.npy")) + list(job.glob("*.npz")) + list(job.glob("*.json")):
        try:
            f.unlink()
        except OSError:
            pass
    shutil.rmtree(job / "direct", ignore_errors=True)


def run_modes(study, mesh, workdir, log, cancel=None, on_line=None, *, modes: int, f_min: float = 0.0, f_max: float | None = None,
              static_cases: list | None = None, name: str = "modal", threads: int | None = None, share_modes: int | None = None) -> DirectModes:
    """The lowest `modes` natural modes of the study in [f_min, f_max] [Hz] (the proof of completeness is part of the analysis) and the static solutions of
    `static_cases` (LoadCase list; the excitation cases of the dynamics) from the same factor. Raises `ModalUnsupported` for a mesh / mass the own assembly cannot
    take, `direct.DirectMemoryError` / `direct.EigenError` / `solve_large.MeshQualityError` for the failures of the processes.

    The state cache (`fea_core.state_cache`): the analysis of the same study and mesh in the same window is computed ONCE – a later call (the dynamics after the natural
    frequencies, the 3-axis analysis, the interface forces) loads it memory-mapped. The right-hand sides are the three unit accelerations (every case that is one global gravity
    load is a combination of them) plus one column per other case. `share_modes` = the mode count another analysis of the same window will ask for: the analysis computes
    max(modes, share_modes) modes once and each caller takes its first ones."""
    from .. import assembly, direct, linear_constraints, solve_large as SL, state_cache
    from .. import gravity_states as GS
    from ..assembly import mass as asm_mass
    from ..ccx import parallel
    from . import modal_cache as MCH
    import contextlib
    import time
    workdir = Path(workdir)
    job = workdir / name
    job.mkdir(parents=True, exist_ok=True)
    _clean(job)
    keep = SL.keep_solver_files()
    t_start = time.time()
    tm: dict = {}
    cases = list(static_cases or [])
    caching = state_cache.enabled()
    from ..eigen import is_free_body
    free = is_free_body(study)                                                                    # a free body: the rigid-body modes at 0 Hz, no static solution
    f_min = float(f_min or 0.0)
    f_max = float(f_max) if f_max else None
    # the unit accelerations are solved beside the modes (a few back-substitutions) so that a later analysis finds them – only for a body whose stiffness can be solved from the
    # factor of the analysis: a supported one and a window from 0 Hz (a window above zero is driven by an indefinite shift, a free body is singular)
    columns, extra_cases, coeffs = MCH.plan_columns(cases) if (cases or (caching and not free and f_min <= 0.0)) else ([], [], [])
    digest = state_cache.study_digest(study, mesh) if caching else ""
    n = len(mesh.nodes)

    def stored() -> Any:
        """The stored analysis that answers this request (None = compute): with the columns asked for, or – for the unit accelerations that were only an extra – one that is known
        to lack them (its static solution was not possible)."""
        nonlocal columns, extra_cases, coeffs
        hit = MCH.lookup(digest, n, None, modes, f_min, f_max, columns)
        if hit is None and not cases and columns:
            hit = MCH.lookup(digest, n, None, modes, f_min, f_max, [], skipped=True)
            if hit is not None:
                columns, extra_cases, coeffs = [], [], []
        if hit is None:
            return None
        log(f"Natural frequencies: the analysis of this model is stored - not computed again ({hit.name})")
        return MCH.build(hit, modes, columns, coeffs, t_start, "stored analysis")
    if caching:
        got = stored()
        if got is not None:
            return got
    k_run = max(int(modes), int(share_modes or 0)) if caching else int(modes)
    with MCH.compute_lock(digest, cancel, log) if caching else contextlib.nullcontext():       # one analysis of a model at a time: the one that waited finds it stored
        if caching:
            got = stored()
            if got is not None:
                return got
        try:
            mf = SL.write_model_files(study, mesh, job, "direct")
            plan: Any = mf.plan
            nx = mf.nx
            try:
                mi = asm_mass.build(study, mesh, plan)
            except ValueError as e:
                raise ModalUnsupported(str(e)) from None
            if mi.dense_terms > linear_constraints.fill_limit():
                raise ModalUnsupported(f"a point mass with a distributing coupling is a dense block of Σ terms² = {mi.dense_terms:.2g} (the limit with the free memory is "
                                       f"{linear_constraints.fill_limit():.2g})")
            mi.save(job)
            fixed, _ = SL.constraint_arrays(study, mesh, nx)
            np.save(job / "fixed.npy", fixed)
            np.save(job / "g.npy", np.zeros(len(fixed)))                      # homogeneous constraints: a prescribed displacement does not enter a free vibration
            col_cases = [GS.unit_case(a) for a in range(3)] + list(extra_cases) if columns else []
            cols_F = [SL.load_vector(study, mesh, lc, plan, nx)[0] for lc in col_cases] or [np.zeros(3 * n + 3 * nx)]
            np.save(job / "F.npy", np.column_stack(cols_F))
            th = int(parallel.cap_threads(threads or study.solver.nproc) or parallel.budget_threads() or 0) or None
            tm["prepare"] = time.time() - t_start
            with direct.exclusive(cancel, log):
                t1 = time.time()
                try:
                    ainfo = SL._assemble(job, "direct", log, cancel, th)
                except assembly.ElementError as e:
                    raise SL.MeshQualityError(SL.mesh_quality_message(e, mesh)) from None
                except SL.LargeSolverError as e:
                    raise direct.DirectError(str(e)) from None
                tm["assembly_process"] = time.time() - t1
                t1 = time.time()
                ndof = 3 * (n + nx)
                lam_min = (2.0 * np.pi * float(f_min)) ** 2 if f_min and f_min > 0 else 0.0
                lam_max = (2.0 * np.pi * float(f_max)) ** 2 if f_max else None
                log(f"Natural frequencies (direct solver: rslab + ARPACK, {int(k_run)} modes, {ndof:,} equations)".replace(",", " "))
                # the static solutions of the cases the caller asked for are part of the answer (a failure is the analysis's); the unit accelerations that only go beside the modes are
                # an extra: where the worker cannot solve them (a body held in one direction is singular) it skips them and says so (`statics_skipped`) – one run, never a second one
                opts = {"modes": int(k_run), "lam_min": lam_min, "lam_max": lam_max, "statics": bool(col_cases), "free": bool(free), "statics_optional": not cases}
                lam, X, S, info = direct.run_eigen_prepared(job / "direct", opts, threads=th, cancel=cancel, on_line=on_line, mmap_modes=True)
                if col_cases and S is None:
                    columns, extra_cases, coeffs, col_cases = [], [], [], []              # the unit accelerations were an extra: the modes are complete without them
                tm["eigen_process"] = time.time() - t1
            MR = np.load(job / "direct" / "MR.npy")
            t1 = time.time()
            # the dependent DOFs from their masters (homogeneous: no offsets), Γ and the shapes by blocks of modes. The modes of the worker (n_eq × m) stay in their file (a read-only map:
            # clean pages the system can drop) – the main process holds the shapes (m × N × 3) and one block, never a resident copy of the modes and a full expanded one besides (three
            # float64 copies of 100 modes × 400 k DOF are 1 GB; the program keeps its memory for the solver). The map is dropped before the files are deleted (Windows locks a mapped file).
            m_modes = X.shape[1]
            gamma = np.empty((m_modes, MR.shape[1]))
            shapes = np.empty((m_modes, n, 3))
            for lo in range(0, m_modes, MODE_BLOCK):
                hi = min(m_modes, lo + MODE_BLOCK)
                Xc = plan.red.expand(X[:, lo:hi]) if plan is not None else np.array(X[:, lo:hi])
                gamma[lo:hi] = Xc.T @ MR                                         # (block, 6)
                shapes[lo:hi] = Xc[:3 * n].T.reshape(-1, n, 3)
                del Xc
            del X
            gc.collect()
        finally:
            if not keep:
                SL._remove_solver_files(job)
        cols = np.stack([np.asarray(plan.red.expand(S[:, [j]])[:, 0] if plan is not None else S[:, j]) for j in range(S.shape[1])]) if S is not None else None
        st_full = []
        for cf in coeffs:
            v = np.zeros(0 if cols is None else cols.shape[1])
            for j in np.nonzero(cf)[0]:
                v += float(cf[j]) * (cols[j] if cols is not None else 0.0)
            st_full.append(v)
        statics = [v[:3 * n].reshape(n, 3) for v in st_full]
        mass = ainfo.get("mass", {})
        rigid = np.zeros(len(lam), bool)
        rigid[list(info.get("rigid", []))] = True
        tm["expand"] = time.time() - t1
        tm["total"] = time.time() - t_start
        out = DirectModes(lam, shapes, gamma, np.asarray(ainfo["mass"]["total_free"], float), float(mass.get("bodies_t", 0.0)), float(mass.get("added_t", 0.0)),
                          rigid, statics, st_full, {"assemble": ainfo, "eigen": info, "job": name, "times": tm}, plan)
        out.columns_all, out.coeffs_all, out.cols = list(columns), [np.asarray(c) for c in coeffs], cols
        if caching:
            out.cache = MCH.store(workdir, digest, out, n, nx, k_run, f_min, f_max, columns, cols, skipped=info.get("statics_skipped"))      # (sets out.content: the hash of THESE shapes)
        return MCH.take(out, int(modes)) if caching else out


def protocol(dm: DirectModes) -> dict:
    """What the results record about the direct eigenvalue analysis (`Results.info["modal_solver"]`): engine, sizes, times, the proof of completeness."""
    e, a = dm.info.get("eigen", {}), dm.info.get("assemble", {})
    shared = {k: dm.info[k] for k in ("shared_analysis", "cache") if dm.info.get(k)}      # one analysis serves several callers (`state_cache`)
    return {**({"state_cache": shared} if shared else {}), "engine": "direct", "library": f"{e.get('engine', 'rslab')} {e.get('version', '')}".strip(), "eigensolver": "ARPACK (SciPy), shift-invert",
            "ndof": a.get("ndof"), "nnz": a.get("nnz"), "factor_nnz": e.get("factor_nnz"), "threads": e.get("threads"),
            "modes": e.get("modes"), "shift": e.get("sigma0"), "factorizations": e.get("n_factor"), "lanczos_solves": e.get("arpack_solves"),
            "sturm_checks": e.get("sturm"), "max_residual": e.get("max_residual"), "residual": e.get("residual"), "max_orthogonality_error": e.get("max_orthogonality_error"),
            "bounds": e.get("bounds"), "window_raised": e.get("window_raised"), "retries": e.get("retries") or None,
            "memory": {"peak_gb": e.get("peak_gb"), "estimate_gb": e.get("need_gb"), "free_gb": e.get("free_gb")},
            "times": {"assembly": a.get("t_assemble"), "assembly_process": a.get("t_total"), "analysis": e.get("t_analyze"), "factorization": e.get("t_factor"),
                      "lanczos": e.get("t_arpack"), "statics": e.get("t_static"), "eigen_process": e.get("t_worker"), **{f"main_{k}": v for k, v in dm.info.get("times", {}).items()}}}

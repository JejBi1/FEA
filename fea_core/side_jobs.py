"""Natural frequencies / buckling and dynamics side by side with the mechanical states (run_study).

The three are independent ccx jobs on the same mesh (eigen.run_eigen and dynamics.run_dynamics compute their own prestress
steps, they do not use the static result), so they need not wait for the statics. The machine is split by
`ccx.parallel.plan` over all their jobs (load cases + modal + buckling + dynamics): the mechanics gets the rest of the
share (`parallel.budget` – its own parallel steps plan inside it), the side jobs run in `solve.Background` threads with
their own share. Results, warnings and errors are taken in the order of the serial calculation (mechanics, eigen,
dynamics): an error of the mechanics stops the side jobs; the side job results are joined after the mechanics, their
progress is deferred to their phases (progress.Progress.later), their log lines do not move the bar.
Beside the DIRECT solver (ngs_direct, the automatic choice of a linear study) the side jobs run too when the memory allows: the direct run
is one share (its worker gets the threads of the share, `parallel.budget_threads`), the ccx side jobs the others, `beside_direct` plans them
from the free memory (the estimate of the direct solver with a margin + the ccx jobs must fit into 70 % of it). Measured 18. 11. 2026
(206 k DOF, 1 load case + 10 modes, 8 cores): ccx statics and modal side by side 33 s, direct statics and then modal 41.5 s.
Serial (the old way) when there is nothing beside the mechanics, the plan gives one job (memory, cores, FEA_CCX_PARALLEL=0) or the
mechanics runs in the iterative hypre solver (its MPI processes take all cores).
With FEA_THREADS set by the user (`config.threads_explicit`) the shares add up to at most FEA_THREADS threads (`parallel.plan`, `beside_direct`).
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from . import config
from .ccx import parallel

DIRECT_MARGIN = 1.4        # the estimate of the direct solver before its analysis is optimistic by up to 35 % for the most compact bodies
FREE_SHARE = 0.7           # the direct solver + the ccx side jobs may plan with this part of the free memory


def beside_direct(ndof: int, n_side: int, free_gb: float | None = None, cores: int | None = None) -> tuple[int, int]:
    """(jobs, threads per job) of the direct solver (one job) and its ccx side jobs; (1, cores) = the side jobs do not run beside it
    (too few cores or not enough free memory for both – then the machine is left to the direct solver and the side jobs follow)."""
    from . import direct
    capped = cores is None and config.threads_explicit()        # FEA_THREADS set by the user: jobs × threads <= NUM_THREADS
    cores = cores or config.NUM_THREADS
    if cores < 2 * parallel.MIN_THREADS or n_side < 1:
        return 1, cores
    if free_gb is None:
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 2**30
        except Exception:  # noqa: BLE001 – without the measurement nothing runs beside the direct solver
            return 1, cores
    rest = FREE_SHARE * free_gb - DIRECT_MARGIN * direct.estimate_gb(ndof) - direct.reserve_gb()
    n_ccx = min(n_side, int(rest // parallel.ccx_memory_gb(ndof))) if rest > 0 else 0
    n = min(n_ccx + 1, cores // parallel.MIN_THREADS if capped else (2 * cores) // parallel.MIN_THREADS)
    if n < 2:
        return 1, cores
    return n, (cores // n if capped else max(parallel.MIN_THREADS, -(-cores // n)))


class SideJobs:
    def __init__(self, study, mesh, workdir, log, on_line, cancel, thermal, prog=None, backend: str | None = None):
        """backend = the solver of the load cases decided by `solve.decide_backend` (None = decided here)."""
        self.study, self.mesh, self.prog = study, mesh, prog
        self.eigen_bg = self.dyn_bg = None
        self.mech_share = None
        # the natural frequencies / the modal basis on the direct path (`direct.modal_run`) are not side jobs: they take the whole machine and one direct run at a time
        # (`direct.exclusive`), so they follow the mechanics (`finish`); buckling and what stays in ccx run beside it
        from .direct import modal_run
        modal_ccx = bool(study.modal.enabled) and self._engine(modal_run, False) == "ccx"
        dyn = bool(study.dynamics.enabled) and self._engine(modal_run, True) == "ccx"
        self.rest_modal = bool(study.modal.enabled) and not modal_ccx
        self.rest_dyn = bool(study.dynamics.enabled) and not dyn
        kinds = (("modal",) if modal_ccx else ()) + (("buckling",) if study.buckling.enabled and study.load_cases else ())
        n_eig = len(kinds)
        if not (n_eig or dyn) or not study.load_cases or not parallel.enabled():
            self.rest_modal, self.rest_dyn = bool(study.modal.enabled), bool(study.dynamics.enabled)
            return
        if backend is None:
            from .solve import decide_backend
            backend = decide_backend(study, mesh)[0]
        if backend not in ("ccx", "ngs_direct"):        # see the module docstring: the iterative solver runs alone
            return
        ndof = 3 * len(mesh.nodes)
        n_side = n_eig + int(dyn)
        if backend == "ccx":
            workers, threads = parallel.plan(len(study.load_cases) + n_side, ndof)
        else:
            workers, threads = beside_direct(ndof, n_side)
        if workers < 2:
            self.rest_modal, self.rest_dyn = bool(study.modal.enabled), bool(study.dynamics.enabled)
            return
        side = min(n_side, workers - 1)                 # at least one share stays for the mechanics
        self.mech_share = (workers - side, threads)
        from .progress import CcxLines
        from .solve import Background
        workdir = Path(workdir)

        def lines(key, parser):
            return prog.lines(on_line, key, parser, True) if prog is not None else on_line

        def eigen(stop, warnings):
            from .eigen import run_eigen
            return run_eigen(study, mesh, workdir / "eigen", log, eig_lines, stop, thermal, kinds=kinds)

        def dynamics(stop, warnings):
            from .dynamics.run import run_dynamics
            return run_dynamics(study, mesh, workdir / "dynamics", log, dyn_lines, stop)

        eig_lines = lines("eigen", CcxLines(n_eig)) if n_eig else None
        dyn_lines = lines("dynamics", CcxLines(2, "Dynamics")) if dyn else None
        if n_eig and dyn and side >= 2:
            self.eigen_bg = Background(_shared(eigen, side - 1, threads), cancel, "eigen")
            self.dyn_bg = Background(_shared(dynamics, 1, threads), cancel, "dynamics")
        elif n_eig and dyn:                             # one share for both: one after another beside the mechanics
            both = Background(_shared(lambda s, w: (eigen(s, w), dynamics(s, w)), side, threads), cancel,
                              "eigen + dynamics")
            self.eigen_bg = self.dyn_bg = both
        elif n_eig:
            self.eigen_bg = Background(_shared(eigen, side, threads), cancel, "eigen")
        else:
            self.dyn_bg = Background(_shared(dynamics, side, threads), cancel, "dynamics")
        log(f"Natural frequencies / dynamics computed concurrently with the load cases "
            f"({workers} CalculiX jobs × {threads} threads)")

    def _engine(self, modal_run, dynamics: bool) -> str:
        """The engine ("direct" | "ccx") of the natural frequencies / the modal basis; a refusal of a forced direct engine is raised later, where the analysis runs."""
        try:
            return modal_run.decide(self.study, self.mesh, dynamics)[0]
        except modal_run.ModalUnsupported:
            return "direct"

    @property
    def active(self) -> bool:
        return self.mech_share is not None

    def mech_budget(self):
        """Share of the machine for the mechanical states (context manager)."""
        return parallel.budget(*self.mech_share) if self.active else nullcontext()

    def abort(self, log=None):
        for bg in {id(b): b for b in (self.eigen_bg, self.dyn_bg) if b is not None}.values():
            bg.abort(log)

    def join(self, res) -> None:
        """Results of the side jobs into res (as solve._add_eigen / _add_dynamics); errors in the serial order."""
        from .solve import StudyError
        from .eigen import EigenError
        from .dynamics.modal import DynamicsError
        prog = self.prog
        both = self.eigen_bg is not None and self.eigen_bg is self.dyn_bg
        out = None
        if self.eigen_bg is not None:
            if prog is not None:
                prog.phase("eigen", "Natural frequencies / buckling")
            try:
                out, _ = self.eigen_bg.result()
            except (EigenError, DynamicsError) as e:
                self.abort()
                raise StudyError(str(e)) from None
            except BaseException:
                self.abort()
                raise
            res.modes = out[0] if both else out
            from .eigen import takeover_warnings
            for note in takeover_warnings(res.modes):
                if note not in res.warnings:
                    res.warnings.append(note)
        if self.dyn_bg is not None:
            if prog is not None:
                prog.phase("dynamics", "Dynamics (modal superposition)")
            try:
                dyn = out[1] if both else self.dyn_bg.result()[0]
            except DynamicsError as e:
                raise StudyError(str(e)) from None
            res.dynamics = dyn
            for w in res.dynamics.warnings:
                if w not in res.warnings:
                    res.warnings.append(w)


def _shared(fn, workers: int, threads: int):
    """fn(stop, warnings) run inside a share of the machine (parallel.budget is per thread → set in the job's thread)."""
    def run(stop, warnings):
        with parallel.budget(workers, threads):
            return fn(stop, warnings)
    return run

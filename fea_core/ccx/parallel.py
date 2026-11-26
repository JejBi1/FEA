"""Parallel calculation of load states in ccx: each *STEP as a separate job over a common model.
Linear statics = steps independent of each other (constraints and loads are written again in every step, OP=NEW), but ccx
in every *STEP assembles and factorises the matrix again, and the factorisation scales badly with 8 threads. Several
jobs with fewer threads at once are therefore faster (verified 13. 10. 2026 with the direct solver of that time, scratch/opt_mech/split.py: example 2,
210 k DOF, 5 steps: 1 job × 8 threads 93.7 s → 5 jobs × 3 threads 56.8 s, U and S equal to 1e-8).

Job `job.inp` is split into `job_model.inc` (everything before the first *STEP) and `job_s{k}.inp` (*INCLUDE + k-th step);
the results of each job are read separately (.frd, .dat) and joined in step order (`read_frd_steps`).
The number of parallel jobs is limited by free memory (estimate of the ccx memory from the DOF) and by the number of cores
(the logical processors; with FEA_THREADS set by the user (`config.threads_explicit`) jobs × threads <= FEA_THREADS, see `plan`).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .runner import CcxError, CcxRun, run_ccx

MIN_THREADS = 3          # PaStiX (a developer's ccx) and the direct solver of `side_jobs.beside_direct`: with fewer threads per job the factorisation slows down more than
#                          the concurrency gains (5 × 2 = 70 s with the multi-threaded solver of 13. 10. 2026). The triangular solve of the SPOOLES of the shipped ccx is ONE thread, its factorization
#                          uses the threads of the job only in the own build with the serial-solve patch (`runner.spooles_threads_safe`) – `plan` plans for the one-thread case


def enabled() -> bool:
    """Parallel steps can be switched off by the environment variable FEA_CCX_PARALLEL=0 (debugging, comparison)."""
    import os
    return os.environ.get("FEA_CCX_PARALLEL", "1").strip() not in ("0", "false", "ne")


def ccx_memory_gb(ndof: int) -> float:
    """Estimate of peak ccx memory (direct solver, tet10) [GB]: 214 k DOF 1.4 GB, 706 k DOF 6.8 GB (CLAUDE.md). Checked against the own ccx_spooles.exe (SPOOLES, a
    static case V1 (a bar), peak working set of the ccx process, 19. 11. 2026): 66 k DOF 0.30 GB, 168 k DOF 1.00 GB, 331 k DOF 2.42 GB – the formula gives 0.45 / 1.07 / 2.34 GB, so it is
    unchanged (conservative for small models, +3 % at 331 k DOF – inside the 0.8 factor of `plan`; the measured growth is DOF^1.31, the formula's 1.32). The PaStiX of the
    retired ccx_static.exe needed more: 0.47 GB at 66 k and 3.39 GB at 331 k DOF. Other procedures of the own ccx (3 runs each, 19. 11. 2026): modal M1 168 k DOF 1.02 GB (formula 1.07),
    contact K1 60 k DOF 0.41 GB (0.42), bolted contact K6 73 k DOF 0.40 GB (0.49), buckling B1 66 k DOF 0.32 GB (0.46) – at or below the formula; a shell (Sh3, 166 k DOF: 2.6 GB, 2.5× the
    formula) is above it, but the shell path does not use it."""
    return 0.2 + 1.2 * (max(ndof, 1) / 214e3) ** 1.32


_tls = threading.local()


@contextmanager
def budget(workers: int, threads: int):
    """In this thread at most `workers` concurrent ccx jobs with `threads` threads each (a share of the machine – other
    independent jobs of the same calculation run beside it, see fea_core.ccx.concurrent). plan() then plans inside the
    share without measuring the memory again (the share was planned with it) and run_ccx without an explicit thread count
    uses workers × threads.
    """
    old = getattr(_tls, "budget", None)
    _tls.budget = (max(1, int(workers)), max(1, int(threads)))
    try:
        yield
    finally:
        _tls.budget = old


def current_budget() -> tuple[int, int] | None:
    return getattr(_tls, "budget", None)


def budget_threads() -> int | None:
    """Thread count of one ccx job in this thread's share (None = no share → config.NUM_THREADS)."""
    b = current_budget()
    return None if b is None else min(b[0] * b[1], config.NUM_THREADS)


def cap_threads(n: int | None) -> int | None:
    """Threads (MPI processes) a worker of its own asks for (`study.solver.nproc`, an option of the caller) under the user's FEA_THREADS: with it set, at most the thread count of this
    thread's share of the machine (`budget_threads`) or FEA_THREADS itself, so the explicit request cannot raise the cap; without FEA_THREADS the request is unchanged. 0 / None = no request (None)."""
    if not n:
        return None
    n = int(n)
    return min(n, budget_threads() or config.NUM_THREADS) if config.threads_explicit() else n


def _single_thread_solver() -> bool:
    """The equation solver of the ccx in use is SPOOLES (the shipped Windows and Linux builds), whose triangular solve always runs on one thread (the multi-threaded one is racy,
    see `runner.one_equation_thread`); a build without the markers of the checked, ordered SPOOLES (`runner.THREAD_MARKERS`) runs the factorization on one thread too. The own Windows build since 1.4.2 factorizes on the
    threads of its job (markers `FEA_SPOOLES_FACTOR_CHECK=2` and `FEA_SPOOLES_ORDERED_ASSEMBLY=1`, a factorization whose result is checked); that does not change the plan – the factorization is the smaller part of a state, every state in its own job
    stays the best choice and the peak memory of a job is the same (V1 52 k nodes, ccx peak working set 0.88 GB with one equation thread and 0.89 GB with 8, 20. 11. 2026).
    Not so in a developer's PaStiX build (`ccx_pastix`, FEA_CCX naming `ccx_static`)."""
    return not config.ccx_solvers()["PASTIX"]


def _logical_cpus(cores: int, explicit: bool) -> int:
    """Logical processors (SMT) for the cap of the concurrent single-thread jobs; `cores` given by the caller = 2 × (a test, a plan for another machine)."""
    if explicit:
        return 2 * cores
    try:
        import psutil
        return max(cores, psutil.cpu_count(logical=True) or 0)
    except Exception:  # noqa: BLE001
        return 2 * cores


def plan(n_steps: int, ndof: int, free_gb: float | None = None, cores: int | None = None) -> tuple[int, int]:
    """(number of parallel jobs, threads per job). (1, NUM_THREADS) = count the old way with one job.

    FEA_THREADS set by the user (`config.threads_explicit`) caps the TOTAL CPU use: jobs × threads <= FEA_THREADS (the logical processors of SMT are not used for more jobs –
    FEA_THREADS=4 on 8 logical processors gave 7 jobs × 1 thread before; now 4 jobs × 1 thread, 2 × 2 for two states); the number of jobs is still limited by the free memory
    and the number of states. Without FEA_THREADS the plan below is unchanged.

    The triangular solve of every job is ONE thread (SPOOLES; the factorization of a marked build uses the job's threads, see `_single_thread_solver`), so mainly the assembly and the stress recovery use the threads of a job and the time of a state is
    spent on one core: the more states run at once, the better, up to the logical processors and the memory (`ccx_memory_gb`, 80 % of the free RAM). Measured
    20. 11. 2026 with the own ccx_spooles.exe on 8 physical / 16 logical cores, V1 (66 k DOF) with N load states, wall seconds (3 runs for N = 5, 2 for the rest,
    jobs × threads): N = 5: 5×1 10.1, 5×2 9.8, 5×3 9.5, 5×4 9.6, 5×8 9.7, 4×2 16.1, 3×3 16.5, 2×4 22.4, 1×8 28.4; N = 10: 5×3 18.6, 6×2 19.7, 8×1 19.5, 8×2 18.3,
    10×1 11.1, 10×2 10.9; N = 16: 5×3 32.8, 8×2 20.1, 12×1 21.2, 16×1 14.2, 16×2 14.0. The time is that of the waves of jobs – every state in its own job is the
    best, the threads per job do not matter (1 … 8). Example 2 (the captured deck of its 5 mechanical states, 210 k DOF, bolts, welds, preload; one run each): 5×1 81.7, 5×2 80.4,
    5×3 80.1, 5×4 78.4, 3×3 145.7, 2×4 197.1 s, results equal to 1e-9. A machine of 4 cores / 8 logical processors (affinity of 8 logical processors, FEA_THREADS=4, V1, 5 states,
    2 runs): 1×4 28.5 s (the former plan for few cores), 2×2 26.1, 3×2 17.7, 5×1 10.3, 5×2 10.1. Up to 5 states on 8 cores the plan is the former one (5 × 3); a PaStiX build keeps
    the former rule (its factorization uses the threads).
    """
    b = current_budget()
    if b is not None and cores is None and free_gb is None:      # inside a share of the machine (ccx.concurrent)
        n = max(1, min(n_steps, b[0]))
        return (n, b[1]) if n > 1 else (1, budget_threads())
    explicit = cores is not None
    capped = not explicit and config.threads_explicit()          # FEA_THREADS set by the user: jobs × threads <= NUM_THREADS
    cores = cores or config.NUM_THREADS
    single = _single_thread_solver()
    if n_steps < 2 or cores < (2 if single else 2 * MIN_THREADS):
        return 1, cores
    if free_gb is None:
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 2**30
        except Exception:  # noqa: BLE001
            return 1, cores
    by_mem = int((0.8 * free_gb) // ccx_memory_gb(ndof))
    old_cap = (2 * cores) // MIN_THREADS                              # even with SMT: 5 × 3 threads on 8 cores
    if capped:                                                        # the user's FEA_THREADS is the budget of ALL concurrent jobs: no SMT oversubscription
        n = max(1, min(n_steps, by_mem, cores if single else cores // MIN_THREADS))
        return (n, cores // n) if n >= 2 else (1, cores)              # cores // n >= 1 (>= MIN_THREADS for PaStiX): n × threads <= cores
    n = max(1, min(n_steps, by_mem, _logical_cpus(cores, explicit) if single else old_cap))
    if n < 2:
        return 1, cores
    t = -(-cores // n)
    return n, (max(MIN_THREADS, t) if n <= old_cap else max(1, t))


def split_steps(text: str) -> tuple[str, list[str]]:
    """(model – everything before the first *STEP, [text of step 1, …]) from the text of .inp. Searches only lines starting with *STEP."""
    lines = text.splitlines(True)
    starts = [i for i, ln in enumerate(lines) if ln[:5].upper() == "*STEP"]
    if not starts:
        raise ValueError("The .inp job has no *STEP")
    ends = starts[1:] + [len(lines)]
    return "".join(lines[:starts[0]]), ["".join(lines[a:b]) for a, b in zip(starts, ends)]


@dataclass
class ParallelRun:
    jobs: list[str]               # job names in step order (job_s0, job_s1, …)
    runs: list[CcxRun]
    workers: int
    threads: int


def run_steps(workdir: str | Path, job: str, ndof: int, exe: Path | None = None, on_line=None,
              cancel: threading.Event | None = None, workers: int | None = None,
              threads: int | None = None) -> ParallelRun:
    """Computes the steps of the job `job.inp` in parallel. Error of any job → CcxError (the others are cancelled)."""
    workdir = Path(workdir)
    text = (workdir / f"{job}.inp").read_text(encoding="latin-1")
    model, steps = split_steps(text)
    if workers is None or threads is None:
        w, t = plan(len(steps), ndof)
        workers, threads = workers or w, threads or t
    (workdir / f"{job}_model.inc").write_text(model, encoding="latin-1")
    jobs = []
    for k, body in enumerate(steps):
        name = f"{job}_s{k}"
        (workdir / f"{name}.inp").write_text(f"*INCLUDE, INPUT={job}_model.inc\n" + body, encoding="latin-1")
        jobs.append(name)
    stop = threading.Event()
    runs: list[CcxRun | None] = [None] * len(jobs)
    errs: list[str] = []
    sem = threading.Semaphore(workers)
    lock = threading.Lock()

    def watch():                                    # cancellation by the user → stop all jobs
        while not stop.is_set():
            if cancel is not None and cancel.wait(0.2):
                stop.set()
                return
            if cancel is None:
                stop.wait(0.5)

    failed = threading.Event()                      # a job failed → the others were stopped by us, not by the user

    def fail(msg: str):
        with lock:
            errs.append(msg)
        failed.set()
        stop.set()

    def go(k: int, name: str):
        with sem:
            if stop.is_set():
                return
            try:
                prefix = f"[step {k + 1}] "
                r = run_ccx(workdir, name, exe=exe, cancel=stop, threads=threads,
                            on_line=(lambda s: on_line(prefix + s)) if on_line else None)
            except CcxError as e:
                if failed.is_set() and not (cancel is not None and cancel.is_set()):
                    return                          # stopped because a sibling job failed – its error is reported
                fail(f"step {k + 1}: {e}")
                return
            except BaseException as e:  # noqa: BLE001 – e.g. ccx not startable: the other jobs must stop too
                fail(f"step {k + 1}: {e}")
                return
            runs[k] = r
            if not r.finished or r.errors:
                fail(f"step {k + 1}: " + "; ".join(r.errors or [f"return code {r.returncode}"]))

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    th = [threading.Thread(target=go, args=(k, n), daemon=True) for k, n in enumerate(jobs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if cancel is not None and cancel.is_set():
        stop.set()
        raise CcxError("Analysis canceled by user")
    stop.set()
    if errs:
        raise CcxError("CalculiX ended with an error:\n  " + "\n  ".join(errs))
    return ParallelRun(jobs, [r for r in runs if r is not None], workers, threads)


def read_frd_steps(workdir: str | Path, jobs: list[str]):
    """Joins the .frd of the jobs step by step into one FrdFile (step k+1 = job k), as if it were one job."""
    from .frd import read_frd
    out = None
    for k, name in enumerate(jobs):
        path = Path(workdir) / f"{name}.frd"
        f = read_frd(path)
        steps = f.steps()
        if not steps:
            raise CcxError(f"CalculiX job {name} wrote no results (empty .frd): {path}")
        last = steps[-1]
        blocks = [b for b in f.blocks if b.step == last]
        for b in blocks:
            b.step = k + 1
        if out is None:
            f.blocks = blocks
            out = f
        else:
            out.blocks.extend(blocks)
    return out

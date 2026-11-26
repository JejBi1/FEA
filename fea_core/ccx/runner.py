"""Running CalculiX as a separate process (progress, cancellation, errors)."""
from __future__ import annotations

import collections
import functools
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import config, dllguard, memory


class CcxError(RuntimeError):
    pass


class CcxCrash(CcxError):
    """CalculiX died without diagnosing the model: no `*ERROR` line and no "Job finished" (an access violation under load, a fatal exit code such as 0xFFFFFFFF, a kill by the system).
    Only this – never an `*ERROR` of ccx about the input, which is a verdict on the model – lets another engine take the analysis over (`fea_core.eigen`, `fea_core.dynamics.modal`)."""


@dataclass
class CcxRun:
    returncode: int
    finished: bool            # ccx vypsal "Job finished"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def crashed(self) -> bool:
        """The process ended without a verdict of its own on the input (see `CcxCrash`)."""
        return not self.errors and not self.finished


FAER_FAILURE_WORDS = ("failed", "no memory")        # the lines of faer.c "FEA faer: ... - the system of equations is solved with SPOOLES" that report a failure; any other such line is a structural "not applicable"


@dataclass
class SpoolesCheck:
    """What the check of the own SPOOLES build (patch ccx_2.23_spooles_serial_solve.patch) reported in the output of one ccx run: the multi-threaded factorization
    failed its check `failed` times and was repeated; `confirmed` = the LAST repeat passed its own test solve ("threaded loss confirmed"; a repeat whose test solve
    could not be done is never confirmed); `matrix` = the factorization on one thread fails the check as well ("the matrix, not the threads"); `bookkeeping` = SPOOLES
    reported that its own hand-over between the threads was violated; `unchecked` = a test solve could not be done (no memory) at ANY time of the run (it stays set: a run
    with an unchecked factorization is never called correct). `faer` = the faer solver (patch ccx_2.23_faer.patch) handed a system of equations to SPOOLES that many times because
    its factorization failed or its factors failed the check of faer.c; `faer_structural` = systems whose form faer.c does not take (SPOOLES solves them as it always did: not a
    failure, no note)."""
    failed: int = 0
    confirmed: bool = False
    matrix: bool = False
    bookkeeping: bool = False
    unchecked: bool = False
    faer: int = 0
    faer_tried: int = 0                 # factorizations that were offered to faer (the line of faer.c before it starts); `faer_tried - faer - faer_singular - faer_structural` of them ended with faer's factors
    faer_singular: int = 0              # systems that faer declined because they are numerically singular (SPOOLES solves them as before; not a failure, no note)
    faer_structural: int = 0            # systems whose structure faer.c does not take (column counts that do not add up, a form it does not know, too large for its indices): not a failure, no note

    def any(self) -> bool:
        return bool(self.failed or self.matrix or self.bookkeeping or self.confirmed or self.unchecked or self.faer)

    def feed(self, line: str) -> bool:
        """Takes a line of the ccx output; True when it belongs to the check (its `*WARNING` lines are then not repeated as raw warnings)."""
        if "*WARNING in spooles" in line:
            if "failed the check" in line:
                self.failed += 1
                self.bookkeeping = self.bookkeeping or "bookkeeping" in line
                return True
            if "the matrix, not the threads" in line or "fails the check as well" in line:
                self.matrix = True
                return True
            if "bookkeeping of the aggregates" in line:
                self.bookkeeping = True
                return True
            if "could not be checked" in line:
                self.unchecked = True
                self.confirmed = False                   # the factors used after this line have no verdict (a later passing repeat confirms again)
                return True
        elif "threaded loss confirmed" in line:
            self.confirmed = True
        elif line.startswith("Factoring the system of equations using the") and "faer solver" in line:
            self.faer_tried += 1
        elif line.startswith("faer: the system of equations is numerically singular"):
            self.faer_singular += 1
        elif "in faer:" in line and "*WARNING" in line:                              # the factors of faer failed their check (or it could not be done): SPOOLES solved the system
            self.faer += 1
            return True
        elif line.startswith("FEA faer:") and "solved with SPOOLES" in line:
            if any(w in line for w in FAER_FAILURE_WORDS):                           # faer could not factorize the system (zero pivot, no memory, ...): SPOOLES did
                self.faer += 1
            else:                                                                    # "not applicable": the structure of the system is not one that faer takes - no failure
                self.faer_structural += 1
            return True
        return False


def spooles_check_note(job: str, chk: SpoolesCheck) -> str:
    """The text for the user about a run in which the check of the SPOOLES factors fired ("" = nothing to say)."""
    if chk.matrix:
        return (f"CalculiX equation solver ({job}): the factors of the stiffness matrix fail the internal check also on one thread, so the cause is the matrix "
                "(nearly singular or very badly scaled: check the supports, the contacts and very different stiffnesses); the results of this job may be inaccurate.")
    if chk.failed or chk.bookkeeping:
        if chk.confirmed and not chk.unchecked:          # only a repeat whose own test solve passed, in a run in which EVERY check could be done, may be called correct
            return (f"CalculiX equation solver ({job}): a rare error of the multi-threaded factorization ({chk.failed} failed checks) was detected and the factorization was "
                    "repeated; the final factors passed the check, the results are correct.")
        if chk.confirmed:                                # the repeat passed, but another factorization of the run had no verdict
            return (f"CalculiX equation solver ({job}): a rare error of the multi-threaded factorization ({chk.failed} failed checks) was detected and the factorization was "
                    "repeated and the final factors passed the check, but the check of another factorization of this job could not be done (not enough memory for the test solve), "
                    "so the results of this job were not fully verified.")
        if chk.unchecked:
            return (f"CalculiX equation solver ({job}): a rare error of the multi-threaded factorization ({chk.failed} failed checks) was detected and the factorization was "
                    "repeated, but the check of the final factors could not be done (not enough memory for the test solve), so the results of this job were not verified.")
        return (f"CalculiX equation solver ({job}): a rare error of the multi-threaded factorization ({chk.failed} failed checks) was detected and the factorization was "
                "repeated, but the check did not confirm the final factors, so the results of this job were not verified.")
    if chk.unchecked:
        return (f"CalculiX equation solver ({job}): the internal check of the factors could not be done (not enough memory for the test solve), "
                "so the results of this job were not verified by it.")
    if chk.faer:
        return (f"CalculiX equation solver ({job}): the faer solver could not factorize {chk.faer} system(s) of equations of this job or its factors failed the internal check; "
                "SPOOLES solved them instead.")
    return ""


_SOLVER_EVENTS: collections.deque = collections.deque(maxlen=200)       # (monotonic time, work folder, job, SpoolesCheck, note) of the runs in which the check fired
_SOLVER_LOCK = threading.Lock()
_FAER_USAGE: collections.deque = collections.deque(maxlen=400)       # (monotonic time, work folder, job, systems solved by faer) of every ccx run of a ccx that has the faer solver


def take_solver_events(root: str | Path, since: float = 0.0) -> list[dict]:
    """The check events (`SpoolesCheck`) of the ccx runs in the folder `root` (and below) that ended after the monotonic time `since`, removed from the register: the
    study run that owns the folder puts them into its results (`fea_core.solve`). Every event has the keys job, failed, confirmed, matrix, bookkeeping, unchecked, faer, note."""
    try:
        base = Path(root).resolve()
    except OSError:
        return []
    out, keep = [], []
    with _SOLVER_LOCK:
        for ev in _SOLVER_EVENTS:
            t, wd, job, chk, note = ev
            if t >= since and (wd == base or base in wd.parents):
                out.append({"job": job, "failed": chk.failed, "confirmed": chk.confirmed, "matrix": chk.matrix, "bookkeeping": chk.bookkeeping,
                            "unchecked": chk.unchecked, "faer": chk.faer, "note": note})
            else:
                keep.append(ev)
        if out:
            _SOLVER_EVENTS.clear()
            _SOLVER_EVENTS.extend(keep)
    return out


def take_faer_usage(root: str | Path, since: float = 0.0) -> list[int]:
    """For every ccx run in the folder `root` (and below) that ended after the monotonic time `since` in a ccx WITH the faer solver: the number of systems of equations that faer
    factorized (not SPOOLES: unsymmetric systems, small ones, fallbacks), removed from the register. [] = no such run (a ccx without faer, or another folder)."""
    try:
        base = Path(root).resolve()
    except OSError:
        return []
    out, keep = [], []
    with _SOLVER_LOCK:
        for ev in _FAER_USAGE:
            t, wd, _job, n = ev
            if t >= since and (wd == base or base in wd.parents):
                out.append(n)
            else:
                keep.append(ev)
        if out:
            _FAER_USAGE.clear()
            _FAER_USAGE.extend(keep)
    return out


def _note_faer_usage(workdir: Path, job: str, chk: SpoolesCheck) -> None:
    try:
        wd = workdir.resolve()
    except OSError:
        wd = workdir
    with _SOLVER_LOCK:
        _FAER_USAGE.append((time.monotonic(), wd, job, max(0, chk.faer_tried - chk.faer - chk.faer_singular - chk.faer_structural)))


def _note_solver_event(workdir: Path, job: str, chk: SpoolesCheck) -> str:
    note = spooles_check_note(job, chk)
    if note:
        try:
            wd = workdir.resolve()
        except OSError:
            wd = workdir
        with _SOLVER_LOCK:
            _SOLVER_EVENTS.append((time.monotonic(), wd, job, chk, note))
    return note


# ccx warnings known to be harmless in our inputs (substring of the *WARNING line) – not passed to the result warnings.
# Everything else is kept (earlier every warning containing "ignored" was dropped – that could hide a real problem).
HARMLESS_WARNINGS = (
    "no elastic constants",                       # *MASS elements of added mass have no material (fea_core.dynamics.mass)
    "a nonlinear geometric analysis is requested",  # heat conduction job: ccx asks for NLGEOM by itself, no effect
)


def relevant_warnings(warnings: list[str]) -> list[str]:
    """The ccx warnings without the known harmless ones (HARMLESS_WARNINGS)."""
    return [w for w in warnings if not any(h in w for h in HARMLESS_WARNINGS)]


def _env(threads: int | None, deterministic: bool | None = None) -> dict:
    env = config.ccx_env(deterministic=deterministic)
    if threads:
        n = str(max(1, int(threads)))
        for k in ("OMP_NUM_THREADS", "CCX_NPROC_STIFFNESS", "CCX_NPROC_RESULTS"):
            env[k] = n
    return env


def _names(inp: Path, *words: bytes) -> list[bool]:
    """For every word: does the input contain it (any step)? Searched in the bytes of the file (a few ms for tens of MB)."""
    import mmap
    try:
        with open(inp, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return [mm.find(w) >= 0 for w in words]
    except (OSError, ValueError):                  # missing or empty file
        return [False] * len(words)


def uses_spooles(inp: Path) -> bool:
    """The input asks for `SOLVER=SPOOLES` (any step)."""
    return _names(inp, b"SOLVER=SPOOLES")[0]


FAER_OFF_COMMENT = b"** FEA_SOLVER=SPOOLES"        # written by ccx.inp.write_inp when the study asks for SPOOLES explicitly: the faer solver of ccx is then not used for this job
SERIAL_SOLVE_MARKER = b"FEA_SPOOLES_SERIAL_SOLVE=1"        # the first marker of the patch: only the serial solve (the first candidate of F17 stage A) – NOT enough for threads
ORDERED_MARKER = b"FEA_SPOOLES_ORDERED_ASSEMBLY=1"          # printed by `ccx -v` only when the program is linked with the ordered SPOOLES (the link fails without it)
FACTOR_CHECK_MARKER = b"FEA_SPOOLES_FACTOR_CHECK=2"        # printed by `ccx -v` of a build with the patch ccx_2.23_spooles_serial_solve.patch of 24. 11. 2026 (and so inside the file)
FACTOR_CHECK_MARKER_V1 = b"FEA_SPOOLES_FACTOR_CHECK=1"     # the first fea_judge (until 23. 11. 2026): after a verdict "the matrix" the later systems still ran UNCHECKED on the threads - NOT accepted
THREAD_MARKERS = (FACTOR_CHECK_MARKER, ORDERED_MARKER)      # a ccx may use several equation-solver threads only with BOTH
# (the first marker of that patch, FEA_SPOOLES_SERIAL_SOLVE=1, is NOT enough: a build that has only the serial solve – the first candidate of F17 stage A – can lose
# an update in the multi-threaded factorization without any message, 3 in 1 260 runs on a 16 core machine, so it keeps ONE thread)


@functools.lru_cache(maxsize=32)
def _has_marker(path: str, mtime_ns: int, size: int) -> bool:
    """Does the file carry BOTH markers of the checked, ordered SPOOLES (`THREAD_MARKERS`)? (the arguments only key the cache: a replaced binary is looked at again)"""
    import mmap
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return all(mm.find(m) >= 0 for m in THREAD_MARKERS)
    except (OSError, ValueError):
        return False


def spooles_threads_safe(exe: str | Path | None) -> bool:
    """The ccx `exe` carries the patches that make the SPOOLES solve serial, check every multi-threaded factorization (repeating it when the check fails) and
    sum in an order that does not depend on the timing (the ordered SPOOLES; its marker is printed only by a program that is linked with it), so several equation-solver
    threads are safe. Read from the executable itself (the marker strings of `ccx -v`, both must be there), not from a name, a manifest or a version: a ccx from FEA_CCX,
    an older installation, a third-party build or an earlier candidate with only the serial solve or only the check has no marker pair and keeps ONE thread.
    `FEA_SPOOLES_THREADS=1` switches the threads off for the marked build too."""
    if exe is None:
        return False
    try:
        st = Path(exe).stat()
    except OSError:
        return False
    return _has_marker(str(exe), st.st_mtime_ns, st.st_size)


def spooles_thread_cap() -> int | None:
    """`FEA_SPOOLES_THREADS` = the most equation-solver threads of a SPOOLES job (1 = the single-thread rule of the earlier versions; 0 counts as 1 –
    the safe reading of "no threads"); None = not set, negative or not a number."""
    try:
        n = int(os.environ.get("FEA_SPOOLES_THREADS", "").strip())
    except ValueError:
        return None
    return max(1, n) if n >= 0 else None


def _team_size(env: dict) -> int:
    """The thread count that `OMP_NUM_THREADS` of `env` asks for: the first number of an OpenMP list ("4,2" = 4 threads, 2 in the nested team), `config.NUM_THREADS`
    when the value is empty, junk or below 1 (the user's own value is kept in `env` – `config.ccx_env` only sets a default)."""
    raw = str(env.get("OMP_NUM_THREADS") or "").split(",")[0].strip()
    try:
        n = int(raw)
    except ValueError:
        return max(1, int(config.NUM_THREADS))
    return n if n >= 1 else max(1, int(config.NUM_THREADS))


def one_equation_thread(inp: Path, exe: str | Path | None = None) -> bool:
    """The equation solver of this input must run on ONE thread: when the input asks for SPOOLES (a step without `SOLVER=` is solved by SPOOLES too) and `exe`
    does not carry the checked-factorization patch (`spooles_threads_safe`; no `exe` = an unknown binary = one thread), or `FEA_SPOOLES_THREADS=1`. An input that asks for PaStiX
    and for no SPOOLES step (a developer's `ccx_pastix`; PaStiX threads are limited by the same variable) keeps its threads.
    The multi-threaded SPOOLES of CalculiX (`FrontMtx_MT_solve`) returns WRONG results without any message: B1 buckling factor 1.67 instead of 21.65 with
    ccx_static.exe, and in the unpatched own build (verified 18. 11. 2026 on the natural frequencies M1 and the buckling B1): 2 threads right, 3, 4 and 8 threads
    wrong by 1e-3 .. 0.99 and different from run to run – a data race of the parallel solve; it shows only in decks with many solves of one factorization.
    The multi-threaded FACTORIZATION is not safe either: in rare runs (3 in 1 260 modal / buckling analyses with 4 threads, 16 cores) one thread's update of one front is
    silently not assembled and the eigenvalues are wrong (reproduced to the digit by dropping that update). The own build (patch
    `installer/*/patches/ccx_2.23_spooles_serial_solve.patch`) therefore solves serially, CHECKS the factors of every multi-threaded factorization with a test solve
    (row-wise backward error, limit 1e-12; the factorization on one thread is checked too) and factorizes again (same threads, then one thread) when the check fails; it says so in `ccx -v`."""
    pastix, spooles = _names(inp, b"SOLVER=PASTIX", b"SOLVER=SPOOLES")
    if pastix and not spooles:
        return False
    return spooles_thread_cap() == 1 or not spooles_threads_safe(exe)


def run_ccx(workdir: str | Path, jobname: str, exe: Path | None = None,
            on_line: Callable[[str], None] | None = None,
            cancel: threading.Event | None = None, timeout: float | None = None,
            threads: int | None = None, deterministic: bool | None = None) -> CcxRun:
    """Runs `ccx -i jobname` in workdir. Output goes to jobname.log and to on_line.
    threads = number of ccx threads (None = config.NUM_THREADS; parallel jobs of ccx.parallel get fewer).
    deterministic = fixed OpenMP scheduling (OMP_DYNAMIC=FALSE, OMP_SCHEDULE=static; the nonlinear jobs ask for it, see config.ccx_env). Since the Intel
    library of earlier versions is gone (its reproducibility switch made linear runs ~25 % slower) this is all it does: no measurable cost (V1, 4 threads:
    8.05 s against 8.46 s) and NO bit-identical repeats – the threads of the equation solver (PaStiX) add in another order, so two runs differ by
    round-off (~1e-9 relative) in both modes.
    """
    workdir = Path(workdir)
    exe = Path(exe or config.CCX_EXE)
    if not exe.exists():
        note = config.ccx_env_note()                                         # a leftover FEA_CCX naming the removed build: say why it is not used
        msg = f"CalculiX not found: {exe}"
        raise CcxError(msg + "\n" + note if note else msg)
    if not threads:                          # share of the machine of this thread (independent jobs side by side)
        from .parallel import budget_threads
        threads = budget_threads()
    dllguard.check_workdir(workdir, "CalculiX")  # ccx runs with cwd = workdir: no planted DLLs (fea_core.dllguard)
    log_path = workdir / f"{jobname}.log"
    memory.release_unused()                  # ccx gets the memory that this process does not use (fea_core.memory)
    env = _env(threads, deterministic)
    inp = workdir / f"{jobname}.inp"
    if _names(inp, FAER_OFF_COMMENT)[0]:
        env["FEA_FAER"] = "0"                                  # SPOOLES was asked for by name (a ccx without faer ignores the variable)
    if one_equation_thread(inp, exe):
        env["CCX_NPROC_EQUATION_SOLVER"] = "1"                 # the multi-threaded solve of an unmarked SPOOLES is racy (see one_equation_thread)
    elif spooles_threads_safe(exe):
        pastix, spooles = _names(inp, b"SOLVER=PASTIX", b"SOLVER=SPOOLES")
        if not (pastix and not spooles):                       # any SPOOLES step (also a mixed deck): the factorization on the threads of this job, the solve serial
            n = _team_size(env)
            env["CCX_NPROC_EQUATION_SOLVER"] = str(max(1, min(n, spooles_thread_cap() or n)))
    proc = subprocess.Popen(
        [str(exe), "-i", jobname], cwd=workdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="latin-1", creationflags=config.NO_WINDOW,
    )
    errors, warnings, finished = [], [], False
    chk = SpoolesCheck()
    timed_out = threading.Event()
    done = threading.Event()

    def watch():                             # cancellation and time limit (the read loop below blocks on the pipe)
        t0 = time.monotonic()
        while proc.poll() is None and not done.is_set():
            if cancel is not None and cancel.is_set():
                proc.kill()
                return
            if timeout is not None and time.monotonic() - t0 > timeout:
                timed_out.set()
                proc.kill()
                return
            if cancel is not None:
                cancel.wait(0.2)
            else:
                done.wait(0.2)
    killer = None
    if cancel is not None or timeout is not None:
        killer = threading.Thread(target=watch, daemon=True, name="fea-ccx-watch")
        killer.start()
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                s = line.strip()
                if chk.feed(s):
                    pass                                    # the check of the SPOOLES factors: one note for the user after the run (spooles_check_note), not the raw lines
                elif "*ERROR" in s:
                    errors.append(s)
                elif "*WARNING" in s:
                    warnings.append(s)
                elif s.startswith("Job finished"):
                    finished = True
                if on_line:
                    on_line(line.rstrip("\n"))
        rc = proc.wait()
    finally:
        done.set()
        if proc.poll() is None:              # an exception while reading (on_line, disk full, KeyboardInterrupt) – no orphaned ccx
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
    if cancel is not None and cancel.is_set():
        raise CcxError("Analysis canceled by user")
    if timed_out.is_set():
        raise subprocess.TimeoutExpired(proc.args, timeout)
    if config.faer_in_ccx(exe):
        _note_faer_usage(workdir, jobname, chk)
    if chk.any():
        note = _note_solver_event(workdir, jobname, chk)
        if note and note not in warnings:
            warnings.append(note)
    return CcxRun(rc, finished, errors, warnings, log_path)

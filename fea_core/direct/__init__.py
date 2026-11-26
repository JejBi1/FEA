"""Direct solver of the linear path (backend "ngs_direct"): one factorization for all load states, in a separate process.

The matrix is assembled by the own assembly (`fea_core.assembly`, numba) and written as the upper triangle; the process
`python -m fea_core.direct.worker JOB` factors it with a ready-made sparse direct solver (engine `rslab`, pure Rust, MIT, a pip wheel – see
`engines.py`; further engines can be added there) and writes the solutions. Our code is glue: reading the files, choosing the precision of the
factor from the memory plan, refinement of a float32 factor in float64, the acceptance check. A crash of the library or running out of memory
does not bring down the GUI, and the calculation can be cancelled (the process tree is killed).

The earlier solver key `ngs_pardiso` (the direct solver of the versions before 1.5) is still loaded as an alias of this backend.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .. import config

ENGINE_ENV = "FEA_DIRECT_ENGINE"
LIBRARY = {"rslab": "rslab"}                  # Python module that carries the engine (a constant here: the status checks of the GUI / MCP
#                                               must not import the engines, which import scipy.sparse)
PRECISION_ENV = "FEA_DIRECT_PRECISION"        # auto | float64 | float32 (tests, diagnostics)
GB = 2**30

# Estimate of the memory BEFORE the analysis (the exact plan is made by the worker after the analysis, from the elimination tree).
# Entries of the factor per DOF of tet10 models measured 17. 11. 2026: thin sheet / tube / beam 365–450, plates 450–540, steel on PLA 990,
# compact block 1093 at 150–300 k DOF; the factor grows with n^(1/3). Bytes per entry of the factor with the fronts and the analysis:
# 6.5 (float32), 11.2 (float64) – measured on the compact block.
FILL_PER_DOF = 800.0
FILL_REF_DOF = 250_000
BYTES_PER_ENTRY = {"float32": 6.8, "float64": 11.4}
MATRIX_BYTES_PER_DOF = 1200.0          # the upper triangle, its copies in the engine and the solution vectors
MEM_MARGIN = 0.85


class DirectError(RuntimeError):
    """The direct solver failed (singular matrix, accuracy not reached, the process crashed)."""


class DirectMemoryError(DirectError):
    """The direct solver does not fit into the free memory (the caller can continue with the iterative solver)."""


def engine_name() -> str:
    return os.environ.get(ENGINE_ENV, "").strip() or "rslab"


def library_name() -> str:
    """Python module of the engine's library (rslab)."""
    return LIBRARY.get(engine_name(), engine_name())


def available() -> bool:
    """The engine's library is installed (module `direct` of the program carries it; checked without importing it)."""
    mod = LIBRARY.get(engine_name())
    try:
        return mod is not None and importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def estimate_gb(ndof: int, precision: str = "float32") -> float:
    """Estimated memory of the direct solver [GB] before its analysis (conservative for thin parts, optimistic for the most compact bodies)."""
    fill = FILL_PER_DOF * max(ndof / FILL_REF_DOF, 0.05) ** (1 / 3)
    return ndof * (fill * BYTES_PER_ENTRY[precision] + MATRIX_BYTES_PER_DOF) / GB


def eigen_vector_count(modes: int, ncv: int = 0) -> int:
    """Number of vectors of the length of the system that the eigenvalue analysis holds at once: the Lanczos basis (`ncv`, 0 = the default of the analysis: 2·modes + 1, at least 20),
    the found modes, their M-products and the work copies (3·modes) and the few vectors of the operators (8). ONE estimate for the routing of the main program (`modal_run.decide`)
    and the memory check of the worker (`eigen.Shifted.check_memory`) – they used to differ (2k + 12 against ncv + 3k + 8)."""
    k = max(1, int(modes))
    return int(ncv or max(2 * k + 1, 20)) + 3 * k + 8


def eigen_vectors_gb(ndof: int, modes: int, ncv: int = 0) -> float:
    """Memory of the vectors of the eigenvalue analysis [GB] (float64), see `eigen_vector_count`."""
    return eigen_vector_count(modes, ncv) * ndof * 8 / GB


def reserve_gb() -> float:
    """Memory that must stay free: the process is stopped when the free RAM drops below it (swapping would freeze the machine)."""
    import psutil
    return max(0.8, 0.04 * psutil.virtual_memory().total / GB)


def fits(ndof: int, free_gb: float | None = None) -> bool:
    """The direct solver (float32 factor) is expected to fit into the free memory."""
    if free_gb is None:
        import psutil
        free_gb = psutil.virtual_memory().available / GB
    return estimate_gb(ndof) <= MEM_MARGIN * free_gb - reserve_gb()


# exit codes of the worker (worker.py)
EXIT_NOFIT, EXIT_NUMERIC, EXIT_ACCURACY = 3, 5, 6

# "Out of memory" is recognised from what the process says, not from the exit code: Rust aborts a failed allocation with "memory allocation of N
# bytes failed" (Windows exit code 0xC0000409 and Linux SIGABRT -6 / 134 are the same for ANY abort, a stack overflow 0xC00000FD or a breakpoint
# 0x80000003 are other crashes and must not be taken for memory). Exit codes alone: 0xC0000017 = STATUS_NO_MEMORY (Windows), and on Linux
# SIGKILL (-9 / 137) that we did not send (cancel and the memory guard are handled before) - the kernel's OOM killer, which leaves no output.
_OOM_TEXT = re.compile(r"memory allocation of \d+ bytes failed|MemoryError|Unable to allocate|std::bad_alloc|Cannot allocate memory"
                       r"|out of memory", re.I)


def is_out_of_memory(rc: int | None, text: str) -> bool:
    """The process ended because of a missing memory (see above); any other crash is a library error."""
    if _OOM_TEXT.search(text or ""):
        return True
    if rc is None:
        return False
    if sys.platform == "win32":
        return (rc & 0xFFFFFFFF) == 0xC0000017
    return rc in (-9, 137)


def _code_text(rc: int | None) -> str:
    """Exit code for a message: a Windows NTSTATUS in hexadecimal (0xC0000409), otherwise the number."""
    if rc is not None and sys.platform == "win32" and (rc < 0 or rc > 0x7FFFFFFF):
        return f"0x{rc & 0xFFFFFFFF:08X}"
    return str(rc)


# One direct solver run at a time in this program (process): the memory plan of a run takes MEM_MARGIN of the FREE memory, so two runs at
# once (the main calculation and the 3D printing creep calculation beside it, a parametric study) would both plan with the same free memory
# and the guard below would then stop the one that is still running. The second run waits (cancellable) until the first one has freed its memory.
_EXCLUSIVE = threading.RLock()


@contextmanager
def exclusive(cancel: threading.Event | None = None, log=None):
    """Context of a direct solver run (matrix assembly + factorization + solution): runs of other threads wait for it."""
    if not _EXCLUSIVE.acquire(blocking=False):
        if log is not None:
            log("Waiting for another direct solver run of this program to finish (they share the memory, one at a time)")
        while not _EXCLUSIVE.acquire(timeout=0.2):
            if cancel is not None and cancel.is_set():
                raise DirectError("Analysis canceled by user")
    try:
        yield
    finally:
        _EXCLUSIVE.release()


def run_prepared(workdir: str | Path, threads: int | None = None, cancel=None, precision: str | None = None,
                 on_line=None) -> tuple[np.ndarray, dict]:
    """Solves the system in the workdir (indptr / indices / data / rhs.npy, matrix.json) → (X (n, states), info of the worker).
    cancel (threading.Event) set → the process is killed, DirectError("Analysis canceled by user"). The free memory is watched while the worker
    runs: below `reserve_gb()` the worker is killed and DirectMemoryError raised (the factorization would start swapping).
    One direct run at a time in the program (`exclusive`): a run that has to wait starts its memory guard only after it got its turn.
    """
    with exclusive(cancel):
        return _run_prepared(workdir, threads, cancel, precision, on_line)


def _run_prepared(workdir, threads, cancel, precision, on_line):
    from ..procs import Canceled, run_tree, tail
    wd = Path(workdir)
    if not available():
        from ..solvers import backend_error
        raise DirectError(backend_error("ngs_direct") or "the direct solver is not available")
    for f in ("x.npy", "direct_info.json"):
        (wd / f).unlink(missing_ok=True)
    opts = {"engine": engine_name(), "threads": int(threads or config.NUM_THREADS),
            "precision": precision or os.environ.get(PRECISION_ENV, "auto"), "reserve_gb": reserve_gb()}
    (wd / "direct_options.json").write_text(json.dumps(opts), encoding="utf-8")
    env = config.child_env()
    stop = threading.Event()                  # user cancel or memory guard
    done = threading.Event()
    low: list[float] = []

    def guard():
        import psutil
        while not done.is_set():
            if cancel is not None and cancel.is_set():
                stop.set()
                return
            free = psutil.virtual_memory().available / GB
            if free < opts["reserve_gb"]:
                low.append(free)
                stop.set()
                return
            done.wait(0.2)

    threading.Thread(target=guard, daemon=True, name="fea-direct-guard").start()
    log_path = wd / "direct.log"
    try:
        rc = run_tree([sys.executable, "-m", "fea_core.direct.worker", str(wd)], log_path, stop, env=env, on_line=on_line)
    except Canceled:
        if low:
            raise DirectMemoryError(f"The free memory of the computer fell to {low[0]:.1f} GB (limit {opts['reserve_gb']:.1f} GB) while the "
                                    "direct solver was running – it was stopped before the machine started swapping. Other programs (or "
                                    "other calculations) may be using the memory; free memory or use the iterative solver.") from None
        raise DirectError("Analysis canceled by user") from None
    finally:
        done.set()
    info: dict = {}
    if (wd / "direct_info.json").exists():
        info = json.loads((wd / "direct_info.json").read_text(encoding="utf-8"))
    err = info.get("error")
    if rc == EXIT_NOFIT:
        raise DirectMemoryError(f"The direct solver does not fit into the memory: {err['message'] if err else tail(log_path, 500)}")
    if rc != 0 or err or not (wd / "x.npy").exists():
        if err:
            raise DirectError(f"Direct solver: {err['message']}")
        text = tail(log_path, 3000)
        if is_out_of_memory(rc, text):
            raise DirectMemoryError(f"The direct solver ran out of memory (the process ended with code {_code_text(rc)}).\n{text}")
        last = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
        raise DirectError(f"The direct solver failed (code {_code_text(rc)}){': ' + last[:160] if last else ''}\n{text}")
    return np.load(wd / "x.npy"), info


# ---------------------------------------------------------------------------------------------------- the eigenvalue process (fea_core.direct.eigen_worker)
EXIT_EIGEN_INCOMPLETE = 7          # a mode could not be proved complete by the Sturm count (eigen.py)


class EigenError(DirectError):
    """The eigenvalue analysis of the direct path failed in a way that is the analysis' (a Lanczos / residual / completeness check that did not pass, a negative
    stiffness): `kind` = accuracy | incomplete | numeric. The caller continues in CalculiX (or reports it, when the direct path was chosen by the user)."""

    def __init__(self, message: str, kind: str = "numeric"):
        super().__init__(message)
        self.kind = kind


def run_eigen_prepared(workdir: str | Path, options: dict, threads: int | None = None, cancel=None, on_line=None, mmap_modes: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """The eigenvalue analysis of the system in the workdir (indptr / indices / data / mdata / rhs.npy of the assembly worker run with a mass) →
    (λ (m,), X (n, m) M-orthonormal, the static solutions (n, s) | None, the info of the worker). Like `run_prepared`: one direct run at a time in the program,
    the free memory is watched, cancel kills the process; a memory problem is `DirectMemoryError`, the other failures `EigenError`.
    mmap_modes: the modes X (n, m) come back as a read-only memory map of the file (the caller reads them by blocks and must drop the map BEFORE deleting the file – Windows keeps a
    mapped file locked) instead of a second resident copy of m·n float64."""
    with exclusive(cancel):
        return _run_eigen(workdir, options, threads, cancel, on_line, mmap_modes)


def _run_eigen(workdir, options, threads, cancel, on_line, mmap_modes=False):
    from ..procs import Canceled, run_tree, tail
    wd = Path(workdir)
    if not available():
        from ..solvers import backend_error
        raise DirectError(backend_error("ngs_direct") or "the direct solver is not available")
    for f in ("eig_lam.npy", "eig_X.npy", "statics.npy", "eigen_info.json"):
        (wd / f).unlink(missing_ok=True)
    opts = {"engine": engine_name(), "threads": int(threads or config.NUM_THREADS), "reserve_gb": reserve_gb(), **options}
    (wd / "eigen_options.json").write_text(json.dumps(opts), encoding="utf-8")
    stop = threading.Event()
    done = threading.Event()
    low: list[float] = []

    def guard():
        import psutil
        while not done.is_set():
            if cancel is not None and cancel.is_set():
                stop.set()
                return
            free = psutil.virtual_memory().available / GB
            if free < opts["reserve_gb"]:
                low.append(free)
                stop.set()
                return
            done.wait(0.2)

    threading.Thread(target=guard, daemon=True, name="fea-eigen-guard").start()
    log_path = wd / "eigen.log"
    try:
        rc = run_tree([sys.executable, "-m", "fea_core.direct.eigen_worker", str(wd)], log_path, stop, env=config.child_env(), on_line=on_line)
    except Canceled:
        if low:
            raise DirectMemoryError(f"The free memory of the computer fell to {low[0]:.1f} GB (limit {opts['reserve_gb']:.1f} GB) while the eigenvalue analysis was "
                                    "running – it was stopped before the machine started swapping.") from None
        raise DirectError("Analysis canceled by user") from None
    finally:
        done.set()
    info: dict = {}
    if (wd / "eigen_info.json").exists():
        info = json.loads((wd / "eigen_info.json").read_text(encoding="utf-8"))
    err = info.get("error")
    if rc == EXIT_NOFIT:
        raise DirectMemoryError(f"The eigenvalue analysis does not fit into the memory: {err['message'] if err else tail(log_path, 500)}")
    if err:
        raise EigenError(f"Eigenvalue analysis: {err['message']}", err.get("kind", "numeric"))
    if rc != 0 or not (wd / "eig_lam.npy").exists():
        text = tail(log_path, 3000)
        if is_out_of_memory(rc, text):
            raise DirectMemoryError(f"The eigenvalue analysis ran out of memory (the process ended with code {_code_text(rc)}).\n{text}")
        last = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
        raise DirectError(f"The eigenvalue analysis failed (code {_code_text(rc)}){': ' + last[:160] if last else ''}\n{text}")
    statics = np.load(wd / "statics.npy") if (wd / "statics.npy").exists() else None
    return np.load(wd / "eig_lam.npy"), np.load(wd / "eig_X.npy", mmap_mode="r" if mmap_modes else None), statics, info

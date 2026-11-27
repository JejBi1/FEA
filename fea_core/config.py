"""Paths to external programs and default settings (Windows and Linux).

Units throughout the application: mm, N, MPa, t (tonne), s.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import memory as _memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""           # extension of executable files

TOOLS_DIR = PROJECT_ROOT / "tools"
CCX_TOOLS = TOOLS_DIR / "calculix"
# hypre (conda-forge, the installer puts it into .venv/hypre_env): Windows Library\bin\HYPRE.dll, Linux lib/libHYPRE.so
HYPRE_ENV = PROJECT_ROOT / ".venv" / "hypre_env"
HYPRE_BIN = HYPRE_ENV / "Library" / "bin" if IS_WIN else HYPRE_ENV / "lib"
HYPRE_LIB_NAME = "HYPRE.dll" if IS_WIN else "libHYPRE.so"
MPIEXEC_DIR = HYPRE_ENV / "Library" / "bin" if IS_WIN else HYPRE_ENV / "bin"   # own mpiexec of the hypre environment
# Linux: system libraries that were missing from the system and that the installer downloaded (libGLU for gmsh, xcb for Qt…)
SYS_LIBS = PROJECT_ROOT / ".venv" / "sys_libs" / "lib"


def user_config_dir() -> Path:
    """User settings (material library, print presets): APPDATA/FEAStudio | $XDG_CONFIG_HOME/FEAStudio."""
    if IS_WIN:
        base = os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "FEAStudio"


def user_data_dir() -> Path:
    """User data (examples, documents, MCP working folders): LOCALAPPDATA/FEAStudio | $XDG_DATA_HOME/FEAStudio."""
    if IS_WIN:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "FEAStudio"


def _find_tool(names: tuple[str, ...], root: Path | None = None) -> Path | None:
    """First existing file from `names` directly in `root` or in its subfolder (unpacked archives have a subfolder)."""
    root = root or CCX_TOOLS
    if not root.is_dir():
        return None
    for name in names:
        for p in [root / name, *sorted(root.glob(f"*/{name}"))]:
            if p.is_file():
                return p
    return None


def which(name: str) -> str | None:
    """Program from PATH WITHOUT the working folder (shutil.which on Windows searches it first – a planted .exe next to
    the project would run). Empty and relative PATH entries are skipped.
    """
    dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip() and os.path.isabs(d.strip())]
    return shutil.which(name, path=os.pathsep.join(dirs)) if not IS_WIN else _which_win(name, dirs)


def _which_win(name: str, dirs: list[str]) -> str | None:
    exts = [""] if Path(name).suffix else [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
    for d in dirs:
        for e in exts:
            p = Path(d.strip().strip('"')) / (name + e)
            if p.is_file():
                return str(p)
    return None


RETIRED_CCX_PREFIX = "ccx_i4"          # the CalculiX build with Intel MKL PARDISO of earlier versions: removed, never started
GONE_CCX_PREFIX = "ccx_static"         # the third-party CalculiX of dhondt.de of earlier versions: the update deletes it; a developer's copy that still exists is run


def retired_ccx(path) -> bool:
    """Does the path name a CalculiX build that is gone: the removed MKL build (`ccx_i4`, `ccx_i4.exe`: never started, even if a file of that name exists), or
    the third-party `ccx_static.exe` of earlier versions when the file no longer exists (the update of the installation deleted it)? A `ccx_static.exe` that
    still exists (a developer keeps it for comparisons) is the developer's choice and runs."""
    p = Path(str(path))
    name = p.name.lower()
    return name.startswith(RETIRED_CCX_PREFIX) or (name.startswith(GONE_CCX_PREFIX) and not p.is_file())


def ccx_env_note() -> str:
    """Text for a variable FEA_CCX left over from an installation that used a build that is gone ("" = nothing wrong): `find_ccx` skips it, and this tells the
    user why (otherwise the program would start a file that was deleted by the update or fails to load its Intel libraries)."""
    env = os.environ.get("FEA_CCX", "")
    if env and retired_ccx(env):
        if Path(env).name.lower().startswith(RETIRED_CCX_PREFIX):
            return (f"The variable FEA_CCX points at {env} – the CalculiX build with Intel MKL of earlier versions no longer exists, so the variable is "
                    "ignored. Remove it (or point it at another ccx program).")
        return (f"The variable FEA_CCX points at {env} – the third-party CalculiX (ccx_static.exe) of earlier versions was removed by the update and replaced by the "
                "program's own build (ccx_spooles.exe), so the variable is ignored. Remove it (or point it at another ccx program).")
    return ""


def find_ccx() -> Path:
    """Path to ccx: FEA_CCX (a developer's override: any ccx program; not the removed `ccx_i4` and not a `ccx_static.exe` that the update deleted – `retired_ccx`,
    the reason is given by `ccx_env_note`) > the program's own build in
    `tools/calculix`: `ccx_pastix` (a local Linux build with PaStiX, installer/linux/build_ccx.sh), `ccx_spooles` (the shipped build, Windows
    `ccx_spooles.exe` made by installer/windows/build_ccx.ps1, Linux made by build_ccx.sh) > ccx in PATH. The third-party `ccx_static.exe` of
    dhondt.de of earlier versions (PaStiX, TAUCS, Scotch ... linked statically – licences) is no longer looked for."""
    env = os.environ.get("FEA_CCX")
    if env and not retired_ccx(env):
        return Path(env)
    p = _find_tool((f"ccx_pastix{EXE}", f"ccx_spooles{EXE}", f"ccx_2.23{EXE}"))
    if p:
        return p
    for name in ("ccx_2.23", "ccx"):
        w = which(name)
        if w:
            return Path(w)
    return CCX_TOOLS / f"ccx_spooles{EXE}"            # does not exist – the runner reports a clear error


CCX_EXE = find_ccx()
CCX_DIR = CCX_EXE.parent


def ccx_solvers() -> dict[str, bool]:
    """Linear solvers that the found ccx can use: SPOOLES all (the shipped build has SPOOLES and, in front of it, the faer solver - `faer_in_ccx`: the name in the input stays SPOOLES); PaStiX only in a locally built Linux `ccx_pastix`
    (installer/linux/build_ccx.sh – never distributed, licence question L4) and in the retired third-party `ccx_static.exe` that a developer may
    still name in FEA_CCX for comparisons; a ccx from PATH is unknown → no.
    """
    name = CCX_EXE.name.lower()
    return {"PASTIX": name.startswith(("ccx_pastix", "ccx_static")), "SPOOLES": True}


FAER_MARKER = b"FEA_FAER=1"        # `ccx -v` prints it and the file carries it when the program's ccx has the faer solver (installer/*/patches/ccx_2.23_faer.patch)


@functools.lru_cache(maxsize=16)
def _carries(path: str, mtime_ns: int, size: int, marker: bytes) -> bool:
    """Does the file contain `marker`? (the arguments besides the path only key the cache: a replaced binary is looked at again)"""
    import mmap
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.find(marker) >= 0
    except (OSError, ValueError):                     # missing, unreadable or empty file
        return False


def ccx_carries(exe: str | Path | None, marker: bytes) -> bool:
    """The ccx program `exe` carries `marker` (read from the executable itself, not from a name, a manifest or a version)."""
    if exe is None:
        return False
    try:
        st = Path(exe).stat()
    except OSError:
        return False
    return _carries(str(exe), st.st_mtime_ns, st.st_size, marker)


def faer_enabled() -> bool:
    """`FEA_FAER=0` in the environment switches the faer solver of the program's ccx off (SPOOLES only); anything else leaves it on."""
    return os.environ.get("FEA_FAER", "").strip() != "0"


def faer_in_ccx(exe: str | Path | None = None) -> bool:
    """The ccx (`exe`, default the one that was found) has the faer solver built in (marker) and it is not switched off: its direct solves of symmetric matrices
    (Cholesky / LBL^T with the ordering of SPOOLES) and unsymmetric ones (LU) are made by faer, SPOOLES takes over for a matrix that faer cannot factorize or that
    fails the check of the factors (the reason is in the log of ccx). The input of ccx still says SOLVER=SPOOLES – the only word ccx knows for it."""
    return faer_enabled() and ccx_carries(CCX_EXE if exe is None else exe, FAER_MARKER)


def solver_label(solver: str, exe: str | Path | None = None) -> str:
    """The name of the equation solver for results and reports: `solver_name`'s SPOOLES is computed by faer in a ccx that has it (FAER, with SPOOLES as the fallback)."""
    return "FAER" if solver == "SPOOLES" and faer_in_ccx(exe) else solver


def mortar_supported() -> bool:
    """Is `*CONTACT PAIR, TYPE=MORTAR` (Hertz case K2) written as asked? NO by default, on every platform: the mortar matrix assembled by ccx 2.23 is
    rounding-dependent. Windows (third-party `ccx_static.exe` and the own `ccx_spooles.exe`, 18. 11. 2026, K2 and the meshes of size × 1.03 / × 0.96): in the
    2nd iteration it contains NaN entries (2 of 6.4 million; the values differ from a NaN-free run by 1e-16 only) and SPOOLES stops with "matrix found to be
    singular". Linux (27. 11. 2026, the `ccx_spooles` of 1.5.1, the K2 deck of the GitHub runner): the same stop with one AND with two equation threads, the
    same deck with eight threads converged – it passed the earlier Linux gates by chance of the rounding. No solver setting helps. The study keeps
    `method = "mortar"` and ccx gets surface-to-surface with K = 50·E (`contacts.model.effective_method`, a visible warning; K2 p0 −1.2 %).
    FEA_MORTAR=1 writes mortar anyway (a ccx of your own, a study that is known to converge), FEA_MORTAR=0 is the default.
    """
    env = os.environ.get("FEA_MORTAR", "").strip()
    return env == "1"


# `Study.solver.solver`: the stored word of "automatic" is still "PARDISO" – the name of the solver that earlier versions chose by default (the Intel one,
# which is gone); studies saved by them carry it, and it means "the best equation solver of this ccx for the procedure" (solver_name).
AUTO_SOLVER = "PARDISO"
KNOWN_SOLVERS = (AUTO_SOLVER, "PASTIX", "SPOOLES", "ITERATIVE SCALING", "ITERATIVE CHOLESKY")


# Procedures for which the PaStiX of ccx is slower than (or fails against) a ONE-thread SPOOLES – measured 18. 11. 2026 with ccx_static.exe
# (4 threads, the same case back to back; docs/projekt/F13_plan_speed_memory_licences_native.md, step 2), ccx time PaStiX : SPOOLES:
#   modal   M1 66 k DOF 40 s : 9 s, 168 k DOF 166 s : 52 s (many solves of one shifted matrix, PaStiX refines every one)
#   shell   Sh1 8.3 s : 0.9 s; Sh1 164 k DOF the PaStiX job dies on its 2nd step (the pattern of the matrix doubles), SPOOLES 29 s;
#           the one-step Sh3 164 k DOF 61 s : 78 s is no reason to keep PaStiX
#   twod, beam  small matrices, equal (A1 0.3 s : 0.3 s)
#   mortar  Linux only: its PaStiX segfaults in the 2nd iteration, the SPOOLES solves the matrix (the Windows ccx_static does not – substituted)
# Everything else is faster with PaStiX: static V1 66 k DOF 5.0 s : 7.3 s, 168 k DOF 16 s : 33 s; buckling B1 17.8 s : 22.5 s; contact K1 59 k DOF
# 28 s : 158 s; transient heat T6 60 s : 189 s; steady heat T2 62 k nodes 10 s : 24 s.
# Repeated 19. 11. 2026 (3 runs, 4 threads, the same ccx_static.exe on the PaStiX deck and on the SPOOLES deck, a machine that other programs kept at 5-10 busy cores):
# B1 buckling PaStiX 21.1 s (21.0-22.5) : SPOOLES 24.3 s (23.2-32.1) – the order of the first measurement stands (the 64.7 s of PaStiX in the first table of F13 step 5d
# was taken on an overloaded machine – spinning PaStiX threads suffer from that, a one-thread SPOOLES much less – and is withdrawn). The shipped Windows ccx has no PaStiX
# at all (step 5d): there SPOOLES is the answer for every procedure; this table is the reason for the choice in a PaStiX build (Linux ccx_pastix, a developer's FEA_CCX).
SPOOLES_PROCEDURES = frozenset({"modal", "shell", "twod", "beam", "mortar"})


def solver_name(requested: str = AUTO_SOLVER, procedure: str = "static") -> str:
    """Linear solver for `*STATIC, SOLVER=`: the requested one if ccx can do it (ccx_solvers). AUTO_SOLVER ("PARDISO", the stored word of "automatic")
    becomes PASTIX where PaStiX is built in (ccx_static, Linux ccx_pastix) and the procedure is not one of SPOOLES_PROCEDURES, otherwise SPOOLES;
    an unavailable PASTIX → SPOOLES. The solver actually used is reported by the calculation (Results.info["ccx_solver"]).
    `procedure`: static | buckle | heat | modal | shell | twod | beam | mortar.
    """
    req = str(requested).strip().upper()
    if req not in KNOWN_SOLVERS:                # value from a foreign study file → only known constants, never input
        req = AUTO_SOLVER
    if req == AUTO_SOLVER:
        return "PASTIX" if ccx_solvers()["PASTIX"] and procedure not in SPOOLES_PROCEDURES else "SPOOLES"
    return req if ccx_solvers().get(req, True) else "SPOOLES"


def _threads_env() -> int:
    """FEA_THREADS as a positive integer (whitespace around it is ignored; only ASCII digits – '²' and other digit-like characters are junk), 0 = not set / not valid."""
    env = os.environ.get("FEA_THREADS", "").strip()
    return int(env) if env.isascii() and env.isdigit() and int(env) > 0 else 0


def _physical_cores() -> int:
    """Physical cores (SMT / Hyper-Threading does not help the linear solver): FEA_THREADS > system > half of the logical ones.
    Windows: GetLogicalProcessorInformationEx (ctypes), Linux: /proc/cpuinfo – without importing psutil (~90 ms at start).
    """
    n = _threads_env()
    if n:
        return n
    try:
        if IS_WIN:
            import ctypes
            k32 = ctypes.windll.kernel32
            size = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))     # 0 = RelationProcessorCore
            buf = ctypes.create_string_buffer(size.value)
            if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
                n, off = 0, 0
                while off < size.value:
                    n += int.from_bytes(buf.raw[off:off + 4], "little") == 0       # Relationship = RelationProcessorCore
                    off += int.from_bytes(buf.raw[off + 4:off + 8], "little")      # Size
                if n:
                    return n
        else:
            per_pkg: dict[str, int] = {}                  # physical id → number of cores of the package ("cpu cores")
            phys = "0"
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("cpu cores"):
                    per_pkg.setdefault(phys, int(line.split(":")[1]))
            if per_pkg:
                return sum(per_pkg.values())
    except Exception:  # noqa: BLE001 – only a refinement of the estimate
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def threads_explicit() -> bool:
    """The user set FEA_THREADS (a positive number). Then NUM_THREADS is also the cap of the TOTAL CPU use of concurrent CalculiX work: (number of concurrent ccx jobs)
    × (threads per job) <= NUM_THREADS (`ccx.parallel.plan`, `side_jobs.beside_direct`, the background thermal / creep jobs of `solve`). Without FEA_THREADS the plans
    count with the logical processors (SMT) as before."""
    return _threads_env() > 0


NUM_THREADS = _physical_cores()                   # notebook 8 cores / 16 threads: 8
LARGE_DOF = 600_000     # from this number of DOF the "auto" backend chooses hypre instead of ccx when the direct solver is not installed
EQUILIBRIUM_TOL = 1e-3# relative imbalance of reactions and loads → calculation error

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # helper processes without a console window (Windows)


def _prepend_path(env: dict, var: str, *dirs: Path) -> None:
    cur = env.get(var, "")
    env[var] = os.pathsep.join([str(d) for d in dirs] + ([cur] if cur else []))


def child_env(env: dict | None = None) -> dict:
    """Environment of helper processes: on Linux adds the downloaded system libraries (.venv/sys_libs) to LD_LIBRARY_PATH."""
    e = dict(os.environ if env is None else env)
    if not IS_WIN and SYS_LIBS.is_dir():
        _prepend_path(e, "LD_LIBRARY_PATH", SYS_LIBS)
    return e


def run_logged(cmd: list, log: Path, cancel=None, env: dict | None = None, timeout: float | None = None) -> int | None:
    """Starts a helper process (cwd = project root) with output to the file `log` (not a pipe – it does not block)
    and waits for the end. Returns the return code; None = cancelled (cancel); after `timeout` [s] subprocess.TimeoutExpired.
    """
    _memory.release_unused()                 # the helper gets the memory that this process does not use (fea_core.memory)
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=child_env(env), stdout=out, stderr=subprocess.STDOUT,
                             creationflags=NO_WINDOW)
        t0 = time.monotonic()
        while True:
            try:                             # returns as soon as the process ends (no fixed sleep after it)
                p.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                p.kill()
                p.wait()
                return None
            if timeout is not None and time.monotonic() - t0 > timeout:
                p.kill()
                p.wait()
                raise subprocess.TimeoutExpired(p.args, timeout)
    return p.returncode


def dev_mode() -> bool:
    """A developer's switch: `FEA_DEV=1` lets test variables of the solvers through to the process of ccx (see `ccx_env`)."""
    return os.environ.get("FEA_DEV", "").strip() == "1"


DEV_ONLY_CCX_VARIABLES = ("FEA_SPOOLES_CHECK_TOL", "FEA_FAER_CHECK_TOL", "FEA_FAER_METHOD", "FEA_FAER_ORDER", "FEA_FAER_VERBOSE", "FEA_FAER_MIN_EQ", "FEA_FAER_UNSYM", "FEA_FAER_INJECT", "FEA_FAER_COND_MAX")       # test switches of the own ccx build (the limit of its check of the SPOOLES factors; 0 = off): never from a user's environment


def ccx_env(deterministic: bool | None = None) -> dict:
    """Environment for the ccx process (the libraries next to ccx in LD_LIBRARY_PATH on Linux, number of threads).

    WARNING:
      - CCX_NPROC_EQUATION_SOLVER > 1 breaks an unpatched SPOOLES (a data race of its parallel solve) → do not set here: `run_ccx` sets it per job – to 1 for every
        SPOOLES input of a ccx without the marker of the serial solve, to the threads of the job for the own build since 1.4.2 (multi-threaded factorization, serial
        solve; `ccx.runner.spooles_threads_safe`), not at all for a pure PaStiX input (`ccx.runner.one_equation_thread`).
      - FEA_SPOOLES_CHECK_TOL (the limit of the own build's check of the SPOOLES factors, 0 = off) is removed from the environment of ccx unless FEA_DEV=1 (`dev_mode`).
      - deterministic: fixed OpenMP scheduling (OMP_DYNAMIC=FALSE, OMP_SCHEDULE=static). If not given, it follows the environment variable
        FEA_CCX_DETERMINISTIC. This is all that is left of the mode since the Intel library of earlier versions went (its reproducibility switches made
        linear runs ~25 % slower and repeatable): measured 18. 11. 2026 with ccx_static.exe, 4 threads, 6 runs of V1 per mode: 8.05 s against 8.46 s, and the results of repeated runs differ
        bit for bit in BOTH modes (PaStiX threads sum in another order; PL1 the same) – so it costs nothing and promises only a tolerance, never identity.
    """
    env = child_env()
    if not dev_mode():                      # the limit of the internal check of the SPOOLES factors is a test switch: a stray value must not weaken the check of a user's run
        for k in DEV_ONLY_CCX_VARIABLES:
            env.pop(k, None)
    if not IS_WIN:
        _prepend_path(env, "LD_LIBRARY_PATH", CCX_DIR)
    n = str(NUM_THREADS)
    if threads_explicit():                  # FEA_THREADS is the user's cap: a stray OMP_NUM_THREADS of the shell must not raise it (`ccx.runner._env` sets the job's share)
        env["OMP_NUM_THREADS"] = n
    else:
        env.setdefault("OMP_NUM_THREADS", n)
    env.pop("CCX_NPROC_EQUATION_SOLVER", None)
    env.setdefault("PASTIX_MIXED_PRECISION", "0")      # ccx_pastix built from the published pastix.c: the float factorization ends with "Solution contains NaN"
                                                        # on stiff models and repeats in double (V1: 15 s instead of 9.5 s); the Windows ccx_static always computes in double
    if threads_explicit():
        env["CCX_NPROC_STIFFNESS"] = env["CCX_NPROC_RESULTS"] = n
    else:
        env.setdefault("CCX_NPROC_STIFFNESS", n)
        env.setdefault("CCX_NPROC_RESULTS", n)

    if deterministic is None:
        deterministic = os.environ.get("FEA_CCX_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes", "on")

    if deterministic:
        env.setdefault("OMP_DYNAMIC", "FALSE")
        env.setdefault("OMP_SCHEDULE", "static")

    return env

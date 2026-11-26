"""Availability of solvers: what is installed, what is missing and how to add it (without Qt, only light imports – nothing is loaded).

Keys such as `Study.solver.backend`: ccx | ngs_direct | ngs_hypre ("auto" chooses only from the available ones, see
`solve.choose_backend`; the stored name ngs_pardiso of the earlier versions = ngs_direct, `canonical`). The GUI greys out the unavailable items according to `backend_status()`, the readiness check
(`solve.check_study` → `check_backend`) reports a manually chosen unavailable solver BEFORE meshing, MCP and REST list the state
in `fea_study_status` and `/api/status`.

Components (verified on an installation by modules, see fea_core/modules.json):
  ccx          CalculiX program (module core; the own build ccx_spooles[.exe] with SPOOLES; PaStiX only in a developer's local ccx_pastix)
  ngs_direct   the own matrix assembly (numba, module accel) + the direct solver rslab (module direct, one factorization for all load states)
  ngs_hypre    the own matrix assembly (numba, module accel) + module hypre (mpi4py + hypre library in .venv/hypre_env) + MPI: Windows
               Microsoft MPI (mpiexec, installed separately with administrator rights), Linux MPICH from the hypre environment
The keys ngs_direct / ngs_hypre are historical (the matrix was assembled by NGSolve until the release 1.4.2); the user sees `TITLES`. The module
ngsolve is needed only by the flexible bodies of mechanisms (and by the debugging fallback FEA_ASSEMBLY=ngsolve).
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

from . import config

KEYS = ("ccx", "ngs_direct", "ngs_hypre")
TITLES = {"auto": "Automatic", "ccx": "CalculiX", "ngs_direct": "Direct solver (rslab)", "ngs_hypre": "Iterative solver (hypre)"}
ALIASES = {"ngs_pardiso": "ngs_direct",       # the name of the direct solver before the release 1.4.2 (the solver key stored by earlier versions)
           "direct": "ngs_direct", "hypre": "ngs_hypre"}     # short names for users of MCP / REST / CLI (the stored keys keep the historical prefix)
# Results saved before the release 1.4.2 say "ngs_pardiso": the calculation really ran with the Intel MKL PARDISO solver that no longer exists, so a
# RESULT is labelled by what ran (`result_title`) – the alias above is only for the solver SETTING of a study (the user's choice loads as the
# direct solver).
LEGACY_RESULT_TITLES = {"ngs_pardiso": "NGSolve + MKL PARDISO (earlier version)"}      # the key names the earlier components


def canonical(key: str | None) -> str | None:
    """The current name of a solver key of a study setting (`ngs_pardiso` → `ngs_direct`)."""
    return ALIASES.get(key, key) if key else key


def result_title(backend: str | None) -> str:
    """Title of the solver that computed a stored result (`Results.info["backend"]`): the real one, also for the earlier versions."""
    b = str(backend or "")
    return LEGACY_RESULT_TITLES.get(b) or TITLES.get(b, b)



# missing components: (description for the user, program module that installs it; "" = MS-MPI separately)
NGSOLVE = ("module ngsolve", "ngsolve")
NUMBA = ("numba (module accel)", "accel")
DIRECT = ("the direct solver library rslab (module direct)", "direct")
HYPRE = ("module hypre", "hypre")
MSMPI = ("Microsoft MPI (mpiexec)", "")
MPI_LINUX = ("MPI (mpiexec in the hypre environment)", "hypre")
CCX = ("the CalculiX program (ccx)", "core")


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):          # sub-package of the missing package
        return False


def find_mpiexec() -> str | None:
    """mpiexec: on Linux preferably from the hypre environment (the same MPI as the HYPRE library), otherwise from PATH; on Windows also
    from the MS-MPI folder (MSMPI_BIN, ProgramFiles/Microsoft MPI/Bin) – a program started before the installation of MS-MPI does not have a new PATH.
    The mpiexec from the conda hypre environment is NOT used on Windows (not verified).
    """
    own = config.MPIEXEC_DIR / ("mpiexec" + config.EXE)
    if not config.IS_WIN and own.exists():
        return str(own)
    w = config.which("mpiexec")
    if w:
        return w
    if config.IS_WIN:
        pf = os.environ.get("ProgramFiles") or os.environ.get("ProgramW6432") or "C:\\Program Files"
        for d in (os.environ.get("MSMPI_BIN", ""), str(Path(pf) / "Microsoft MPI" / "Bin")):
            if d and (Path(d) / "mpiexec.exe").is_file():
                return str(Path(d) / "mpiexec.exe")
    return None


def _missing(key: str) -> list[tuple[str, str]]:
    if key == "ccx":
        return [] if config.CCX_EXE.exists() else [CCX]
    from . import assembly
    if assembly.assembler() == "ngsolve":                # the debugging fallback FEA_ASSEMBLY=ngsolve: NGSolve assembles, numba is not used
        out = [] if _has("ngsolve") else [NGSOLVE]
    else:                                                # the own matrix assembly
        out = [] if _has("numba") else [NUMBA]
    if key == "ngs_direct":
        from . import direct
        if not _has(direct.library_name()):
            out.append(DIRECT)
    elif key == "ngs_hypre":
        if not _has("mpi4py") or not (config.HYPRE_BIN / config.HYPRE_LIB_NAME).exists():
            out.append(HYPRE)
        if find_mpiexec() is None:
            out.append(MSMPI if config.IS_WIN else MPI_LINUX)
    return out


def hypre_missing() -> list[str]:
    """Missing components of the iterative solver (assembly + hypre; empty = usable; `hypre.available()`)."""
    return [m[0] for m in _missing("ngs_hypre")]


def direct_missing() -> list[str]:
    """Missing components of the direct solver (assembly + rslab; empty = usable): what `solve.choose_backend` names when it does not use it."""
    return [m[0] for m in _missing("ngs_direct")]


def _fix(miss: list[tuple[str, str]]) -> str:
    """How to install the missing components (one sentence)."""
    if CCX in miss:
        return ("Run the installation again (python installer/fea_setup.py install) or set the variable FEA_CCX "
                "to the path of the ccx program.")
    mods = sorted({m for _, m in miss if m})
    parts = []
    if mods:
        parts.append(f"Tools → Program modules… (or python -m fea_core.modules install {' '.join(mods)})")
    if MSMPI in miss:
        parts.append("Microsoft MPI: Tools → Program modules… → hypre, or python installer/fea_setup.py mpi "
                     "(administrator rights)")
    if not parts:
        return "Run the repair installation (python installer/fea_setup.py install)."
    return "Install the missing parts: " + "; ".join(parts) + ". Then restart FEA Studio."


def install_hint() -> str:
    """One hint how to install all the missing solvers (union of the missing components), "" = nothing is missing."""
    miss: list[tuple[str, str]] = []
    for k in KEYS:
        for m in _missing(k):
            if m not in miss:
                miss.append(m)
    return _fix(miss) if miss else ""


def backend_status() -> dict[str, dict]:
    """{key: {"ok", "title", "missing": [component descriptions], "fix": how to install}} for ccx, ngs_direct, ngs_hypre."""
    out = {}
    for k in KEYS:
        miss = _missing(k)
        out[k] = {"ok": not miss, "title": TITLES[k], "missing": [m[0] for m in miss], "fix": _fix(miss) if miss else ""}
    return out


def linear_solvers() -> dict[str, bool]:
    """Linear solvers inside CalculiX (`Study.solver.solver`) that the found ccx can do (config.ccx_solvers)."""
    return config.ccx_solvers()


def linear_solver_note(name: str) -> str:
    """Why the CalculiX linear solver is not possible (empty = possible; the automatic choice always is)."""
    n = str(name).upper()
    if n == config.AUTO_SOLVER or linear_solvers().get(n, True):
        return ""
    return "this CalculiX version does not contain it"


def backend_error(key: str) -> str | None:
    """Clear error when the solver `key` is not installed (None = available or "auto")."""
    key = canonical(key)
    if key == "auto" or key not in KEYS:
        return None
    miss = _missing(key)
    if not miss:
        return None
    what = ", ".join(m[0] for m in miss)
    if key == "ccx":
        note = config.ccx_env_note()
        text = f"The “CalculiX” solver is not installed – missing {what}. {_fix(miss)}"
        return text + "\n" + note if note else text
    return (f"The “{TITLES[key]}” solver is not installed – missing {what}. {_fix(miss)} Or choose the Automatic solver.")


def module_error(key: str, module: str, broken: str | None = None) -> str:
    """Error of the helper process of the solver `key` (assembly, direct solver, hypre) which lacks the Python module `module`
    (ModuleNotFoundError in the output of the process), or – `broken` = the last line of its error – has it installed but cannot load it
    (a damaged installation: a DLL that does not load, numba and llvmlite or numpy of different versions).
    """
    top = str(module).split(".")[0]
    comp = {"ngsolve": NGSOLVE, "netgen": NGSOLVE, "pyngcore": NGSOLVE, "numba": NUMBA, "llvmlite": NUMBA, "rslab": DIRECT, "mpi4py": HYPRE}.get(
        top, (f"Python module {top}", ""))
    if broken:
        return (f"The “{TITLES.get(key, key)}” solver cannot run – {comp[0]} is installed but cannot be loaded ({broken[:300]}). "
                "Run the repair installation (python installer/fea_setup.py install --force), then restart FEA Studio. Or choose the Automatic solver.")
    return (f"The “{TITLES.get(key, key)}” solver cannot run – missing {comp[0]}. {_fix([comp])} Or choose the Automatic solver.")


_SOLVER_PACKAGES = ("numba", "llvmlite", "rslab", "mpi4py", "ngsolve", "netgen", "scipy", "numpy")


def import_failure(key: str, text: str) -> str | None:
    """The clear error for the output of a helper process of the solver `key` (the end of its log) when it shows that a Python module could not be
    imported – not installed (`No module named 'numba'`) or installed and broken (`ImportError: DLL load failed …` in the files of numba / llvmlite /
    rslab…, the last line of a traceback). None = the output shows no such failure (the caller shows the output as it is)."""
    text = text or ""
    miss = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", text)
    if miss:
        return module_error(key, miss.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last = next((ln for ln in reversed(lines) if re.match(r"^(\w+\.)*\w*(Error|Exception)\b", ln) or ln.startswith(("ImportError", "OSError"))), "")
    if re.match(r"^(ImportError|OSError)\b", last):
        pkg = next((m.group(1) for m in (re.search(r"site-packages[\\/]([\w]+)", ln) for ln in lines) if m and m.group(1) in _SOLVER_PACKAGES), None)
        if pkg:
            return module_error(key, pkg, broken=last)
    return None


def backend_ignored(study) -> bool:
    """The choice of the solver is not used: nonlinear calculation (always ccx), 2D section, shells, frames alone (own solver),
    a study without load states (heat, natural frequencies, dynamics – always ccx).
    """
    frame_only = study.frame.active and not str(study.geometry or "").strip()      # = beams.run.frame_only
    return (not study.load_cases or study.twod.enabled or study.shell.enabled or frame_only
            or study.is_nonlinear())


def modal_engine_errors(study) -> list[str]:
    """A forced direct engine of the natural frequencies / the modal basis (`Study.solver.modal_engine`, FEA_MODAL_ENGINE) that cannot run here: what is missing or what it
    cannot do – before meshing. "auto" and "ccx" have no errors (auto continues in CalculiX with the reason)."""
    if not (study.modal.enabled or study.dynamics.enabled):
        return []
    from .direct import modal_run
    if modal_run.engine_setting(study) != "direct":
        return []
    out = []
    for dynamics in {False, bool(study.dynamics.enabled)} if study.modal.enabled else {True}:
        try:
            why = modal_run.unsupported(study, dynamics)
        except Exception:  # noqa: BLE001 – the check must not break the study check
            why = []
        if why:
            out.append(f"The direct natural-frequency analysis cannot do this study: {', '.join(why)} – choose Automatic or CalculiX for the natural frequencies.")
    miss = direct_missing()
    if miss:
        out.append(f"The direct natural-frequency analysis needs {', '.join(miss)} – install it (Tools → Program modules…) or choose Automatic / CalculiX.")
    return list(dict.fromkeys(out))


def check_backend(study, backend: str | None = None) -> list[str]:
    """Errors of a manually chosen solver (`backend` = override of the study setting, MCP): not installed / cannot do the study.
    Before meshing – so that one does not wait for the mesh needlessly. "auto" and a study where the choice is not used have no errors.
    """
    errs0 = modal_engine_errors(study)
    b = canonical(backend or study.solver.backend or "auto")
    if b == "auto" or backend_ignored(study):
        return errs0
    if b not in KEYS:
        return [f"Unknown solver “{b}” – choose Automatic, CalculiX, the direct solver or the iterative solver (hypre)."]
    err = backend_error(b)
    errs = errs0 + ([err] if err else [])
    if b != "ccx":
        from .solve_large import unsupported
        why = unsupported(study, "hypre" if b == "ngs_hypre" else "direct")
        if why:
            errs.append(f"The “{TITLES[b]}” solver cannot handle this study: {', '.join(why)} – choose Automatic or CalculiX.")
    return errs


def solver_warnings(study) -> list[str]:
    """Warnings about the CalculiX setup: a variable FEA_CCX left over from the removed MKL build (it is skipped), and an EXPLICIT choice of the equation
    solver PaStiX where the automatic choice would use SPOOLES (config.SPOOLES_PROCEDURES):
    a shell model with several load cases (the PaStiX analysis of the 2nd step stopped a 164 k DOF job, and it is slower even on small models:
    Sh1 8.3 s against 0.9 s) and a mortar contact where CalculiX solves it (the mortar matrix of the PaStiX build ends with NaN – Hertz case K2).
    Nothing when the solver is automatic or PaStiX is not in this CalculiX (the run then replaces it and says so)."""
    note = config.ccx_env_note()                                                # FEA_CCX of an earlier MKL installation: skipped, the user is told why
    out = [note] if note else []
    if str(study.solver.solver or "").strip().upper() != "PASTIX" or not linear_solvers().get("PASTIX"):
        return out
    if study.shell.enabled and len(study.load_cases) > 1:
        out.append("Equation solver PaStiX with a shell model of several load cases: the PaStiX analysis of the 2nd step can stop the job and "
                   "shells are slower with it than with SPOOLES – choose the automatic solver or SPOOLES")
    if study.active_contacts() and study.contact_settings.method == "mortar" and config.mortar_supported():
        out.append("Equation solver PaStiX with the mortar contact: the mortar matrix of CalculiX ends with NaN in PaStiX (Hertz case K2) – "
                   "choose the automatic solver or SPOOLES")
    return out


def summary() -> dict:
    """Brief state for MCP / REST: {key: "ok" | "missing …"} + the linear solvers of CalculiX."""
    st = backend_status()
    d = {k: ("ok" if v["ok"] else "missing " + ", ".join(v["missing"])) for k, v in st.items()}
    d["ccx_linear_solvers"] = [k for k, ok in linear_solvers().items() if ok]
    fixes = [v["fix"] for v in st.values() if v["fix"]]
    if fixes:
        d["fix"] = fixes[0] if len(set(fixes)) == 1 else " | ".join(dict.fromkeys(fixes))
    note = config.ccx_env_note()
    if note:
        d["note"] = note
    return d


def table() -> list[str]:
    """Overview of the solvers for `python -m fea_core.solvers` (installer: fea_setup.py doctor)."""
    lines = ["Solvers (√ available, – missing):"]
    for k, v in backend_status().items():
        lines.append((f"  {'√' if v['ok'] else '–'} {k:12s} {v['title']:18s}"
                      + ("" if v["ok"] else f" missing: {', '.join(v['missing'])}")).rstrip())
        if v["fix"]:
            lines.append(f"      {v['fix']}")
    have = linear_solvers()
    lines.append("  CalculiX system solver: " + ", ".join(f"{n} {'√' if ok else '–'}" for n, ok in have.items())
                 + f"  ({config.CCX_EXE.name})")
    mpi = find_mpiexec()
    lines.append(f"  mpiexec: {mpi or '–'}")
    note = config.ccx_env_note()
    if note:
        lines.append(f"  ! {note}")
    return lines


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(table()))

"""Own assembly of the stiffness matrix of the linear path (C3D10, numba) – replaces the NGSolve worker of the earlier versions.

`kernels.py` (numba kernels), `system.py` (`assemble` / `build_system`: K, the thermal load, the elimination of the constraints → the CSR of the system,
the check of the elements), `worker.py` (the process `python -m fea_core.assembly.worker JOB direct|hypre` reads the job files and writes the files of
the solver).
This package imports nothing heavy (no numba, no scipy): the main program only starts the worker (`run_worker`) and asks whether it can run
(`available`). numba is the module `accel` of the program.

Debug fallback for one release: FEA_ASSEMBLY=ngsolve makes `solve_large` use the NGSolve worker (`fea_core.ngs.worker`) again (needs the module ngsolve).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


class AssemblyError(RuntimeError):
    """The system cannot be assembled from the given data (input arrays, material table, thermal load without the expansion coefficient…)."""


class ElementError(AssemblyError):
    """Elements the system cannot be built from: a Jacobian determinant that is not positive and finite at an integration point – an inverted
    (reversed node order), flat or damaged (NaN coordinates) element. CalculiX rejects them as well ("nonpositive jacobian").
    `elements` = the first of them (0-based rows of Mesh.tets), `count` = how many there are."""

    def __init__(self, message: str, elements=(), count: int | None = None):
        super().__init__(message)
        self.elements = [int(e) for e in elements]
        self.count = int(len(self.elements) if count is None else count)


ERROR_FILE = "assembly_error.json"          # written by the worker for an input problem (not for a crash): {"kind": "elements" | "assembly", "message", …}
EXIT_ELEMENTS = 3                           # exit code of the worker for a mesh problem (ElementError)
EXIT_ASSEMBLY = 4                           # … and for any other AssemblyError (the message is the whole story, no traceback)
ASSEMBLER_ENV = "FEA_ASSEMBLY"            # own (default) | ngsolve (the earlier assembly by NGSolve, kept for one release as a fallback)
NOTICE_PREFIX = ">> "                       # lines of the worker that are shown to the user


def assembler() -> str:
    """Which assembly the linear path uses: "own" (default) or "ngsolve" (FEA_ASSEMBLY=ngsolve)."""
    v = os.environ.get(ASSEMBLER_ENV, "").strip().lower()
    return "ngsolve" if v == "ngsolve" else "own"


def available() -> bool:
    """The chosen assembly can run: numba (module accel of the program) for the own one, NGSolve (module ngsolve) for the debugging fallback
    FEA_ASSEMBLY=ngsolve – checked without importing them."""
    try:
        return importlib.util.find_spec("ngsolve" if assembler() == "ngsolve" else "numba") is not None
    except (ImportError, ValueError):
        return False


def run_worker(job: str | Path, solver: str, threads: int | None = None, cancel=None, on_notice=None, hom_cols: int = 0) -> dict:
    """Assembles the system of the job folder into `job/direct` (the upper triangle) or `job/hypre` (the whole matrix), in a separate process.
    → assemble.json (sizes, times). cancel (threading.Event) set → the process tree is killed, RuntimeError("Analysis canceled by user");
    on_notice(text) gets the lines of the worker that are meant for the user (the first-use compilation).
    hom_cols = the number of the last right-hand sides that are homogeneous (the unit states of the bolt preload, see `system.assemble`).
    """
    import json
    import sys

    from .. import config
    from ..procs import Canceled, run_tree, tail
    job = Path(job)
    (job / "assemble.json").unlink(missing_ok=True)
    (job / ERROR_FILE).unlink(missing_ok=True)
    (job / "assembly.json").write_text(json.dumps({"threads": int(threads or config.NUM_THREADS), "hom_cols": int(hom_cols)}), encoding="utf-8")

    def line(ln: str) -> None:
        if on_notice is not None and ln.startswith(NOTICE_PREFIX):
            on_notice(ln[len(NOTICE_PREFIX):])

    log_path = job / "assembly.log"
    try:
        rc = run_tree([sys.executable, "-m", "fea_core.assembly.worker", str(job), solver], log_path, cancel,
                      env=config.child_env(), on_line=line)
    except Canceled:
        raise RuntimeError("Analysis canceled by user") from None
    if rc in (EXIT_ELEMENTS, EXIT_ASSEMBLY) and (job / ERROR_FILE).exists():
        try:
            err = json.loads((job / ERROR_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            err = {}
        if err.get("kind") == "elements":
            raise ElementError(str(err.get("message") or "Invalid elements"), err.get("elements") or [], err.get("count"))
        if err.get("kind") == "assembly" and err.get("message"):
            raise AssemblyError(str(err["message"]))
    if rc != 0 or not (job / "assemble.json").exists():
        text = tail(log_path)
        from ..solvers import import_failure
        raise RuntimeError(import_failure(f"ngs_{solver}", text) or f"The assembly of the stiffness matrix failed (code {rc}):\n{text}")
    return json.loads((job / "assemble.json").read_text(encoding="utf-8"))

"""Long jobs outside the GUI thread (mesh and calculation run in separate processes as before)."""
from __future__ import annotations

import logging
import os
import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal

log = logging.getLogger("fea_gui.workers")


class _TraceBus(QObject):
    """Tracebacks of failed background jobs → the solver log of the main window (signal = queued into the GUI thread)."""
    trace = Signal(str)


TRACE = _TraceBus()


def report_trace(what: str) -> None:
    """Called in the except block of a worker thread: the full traceback goes to the Python log and to the solver log
    (MainWindow connects TRACE.trace); the failed signal carries only the short message for the user.
    """
    tb = traceback.format_exc()
    log.error("%s failed:\n%s", what, tb)
    try:
        TRACE.trace.emit(f"{what} failed – traceback:\n{tb.rstrip()}")
    except RuntimeError:                     # application is closing
        pass


def kill_mesh_workers(under: object = None) -> int:
    """Backup for cancellation: terminates the Gmsh meshing processes (python -m fea_core.mesh.worker) started by this program.
    Normally it is enough to set `cancel` of the thread – generate_mesh ends the process itself. Returns the number of terminated processes.
    under = only the workers whose job file is below this folder (the work folder of one run): the meshing of another job of the program is left alone.
    """
    try:
        import psutil
    except ImportError:
        return 0
    root = None
    if under is not None:
        root = _norm_path(under)
        if not root:                                       # an empty folder is no folder: it must not match every worker
            return 0
    n = 0
    for p in psutil.Process().children(recursive=True):
        try:
            cmd = p.cmdline()
            if any("fea_core.mesh.worker" in a for a in cmd) and (root is None or any(_is_inside(a, root) for a in cmd)):
                p.kill()
                n += 1
        except psutil.Error:
            pass
    return n


def _norm_path(path: object) -> str:
    """The path as the system compares it (case, separators, `..`); empty for an empty text."""
    text = str(path).strip()
    return os.path.normcase(os.path.normpath(text)) if text else ""


def _is_inside(arg: str, root: str) -> bool:
    """`arg` (an argument of a command line) is `root` or lies below it: a path prefix that ends at a separator, not a substring (`checks_17` is not inside `checks_1`)."""
    a = _norm_path(arg)
    return bool(a) and (a == root or a.startswith(root.rstrip(os.sep) + os.sep))


def kill_ccx_processes() -> int:
    """"Stop and keep the results" (F8, spec 4.9): terminates the running CalculiX (ccx*) processes of this program WITHOUT
    a cancellation setting – the nonlinear calculation then ends as a non-convergence and the core returns the last converted increment
    (StepResult.partial). Returns the number of terminated processes.
    """
    try:
        import psutil
    except ImportError:
        return 0
    n = 0
    for p in psutil.Process().children(recursive=True):
        try:
            if p.name().lower().startswith("ccx"):
                p.kill()
                n += 1
        except psutil.Error:
            pass
    return n


class TaskThread(QThread):
    """Calculation in the background: fn(thread) → result; signals for the log, solver lines, progress and finished variants."""
    log = Signal(str)
    solver_line = Signal(str)
    progress = Signal(int, int, str)
    outcome = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except Exception as e:  # noqa: BLE001
            report_trace(f"Background job {getattr(self.fn, '__qualname__', '')}".rstrip())
            self.failed.emit(str(e) or e.__class__.__name__)


class MeshThread(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, study, parts, workdir, parent=None):
        super().__init__(parent)
        self.study, self.parts, self.workdir = study, parts, workdir
        self.cancel = threading.Event()

    def run(self):
        try:
            from fea_core.solve import mesh_study           # also with the bolts (imprints, without the bolt parts) and welds
            parts = self.parts() if callable(self.parts) else self.parts    # callable = own copy read in this thread
            mesh, warnings = mesh_study(self.study, parts, self.workdir, self.cancel)
            mesh.report["study_warnings"] = list(warnings)     # taking the mesh over into the calculation (prepare_mesh)
            mesh.report.setdefault("warnings", [])
            for w in warnings:
                if w not in mesh.report["warnings"]:
                    mesh.report["warnings"].append(w)
            self.done.emit(mesh)
        except Exception as e:  # noqa: BLE001
            report_trace("Meshing")
            self.failed.emit(str(e) or e.__class__.__name__)


class SolveThread(QThread):
    log = Signal(str)
    solver_line = Signal(str)
    progress = Signal(float, float, str)          # value 0–1, end of the phase, text (fea_core.progress)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, study, workdir, parent=None, mesh=None, parts=None):
        super().__init__(parent)
        self.study, self.workdir = study, workdir
        self.mesh, self.parts = mesh, parts         # finished mesh (meshing inputs unchanged) → without Gmsh
        self.cancel = threading.Event()

    def run(self):
        try:
            from fea_core.solve import run_study
            res = run_study(self.study, self.workdir, on_log=self.log.emit,
                            on_solver_line=self.solver_line.emit, cancel=self.cancel, mesh=self.mesh, parts=self.parts,
                            on_progress=self.progress.emit)
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001
            report_trace("Analysis")
            self.failed.emit(str(e) or e.__class__.__name__)

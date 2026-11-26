"""Document life cycle of the main window: opening, saving and change tracking, drag and drop, the busy state, undo / redo, geometry
replacement and the update from a CAD add-in.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from fea_core import materials as matlib
from fea_core.geometry import legacy_step_names, read_step, bbox
from fea_core.native_cad import is_native, to_step as native_to_step
from fea_core.joints.levels import LEVELS as JOINT_LEVELS, settings_level as joint_settings_level
from fea_core.project import save_project, load_project, load_study_json
from fea_core.selectors import resolve, resolve_edge
from fea_core.study import Study, LoadCase
from fea_core.i18n import keep_names, tr

from .motion_panel import mech_signature
from .joints_panel import change_level
from . import session_files
from .workers import MeshThread, SolveThread

from .main_defs import (APP_NAME, STAGES, STAGE, F8_SOURCE, CAD_SOURCE, DROP_SUFFIXES, NATIVE_FILTER, nice_floor, _InputBlocker,
                        _busy_progress, _batched, msg_via_main as msg)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class ProjectMixin(_Base):
    """Document life cycle of the main window: opening, saving and change tracking, drag and drop, the busy state, undo / redo,
    geometry replacement and the update from a CAD add-in."""

    # ------------------------------------------------------------------ new geometry keeping the settings
    def replace_geometry_dialog(self):
        if self.study is None or self.step_path is None:
            msg(self, APP_NAME, "Open a model or a project with STEP geometry first.")
            return
        last = self.settings.value("last_dir", "")
        p, _ = QFileDialog.getOpenFileName(self, "Replace geometry – modified model", last,
                                           f"STEP (*.step *.stp *.STEP *.STP);;{NATIVE_FILTER}")
        if p:
            self.update_geometry(Path(p))

    def update_geometry(self, path: Path, trigger: str = "step", solve: bool = False) -> dict:
        """New STEP (modified model from the CAD) keeping the settings: proposed remapping (fea_core.cad_link) →
        window to check (if there is something to check) → new study, iteration; solve = calculate after a certain remapping.
        """
        from fea_core import cad_import
        from fea_core.cad_link.update import apply_report, plan_update
        if self.busy():
            msg(self, APP_NAME, "Meshing or analysis in progress – wait for it to finish.")
            return {"applied": False, "reason": "busy"}
        if self.study is None or self.step_path is None:
            return {"applied": False, "reason": "no study"}
        path = Path(path)
        warn: list[str] = []
        old_step = self.step_path

        def read(_th):
            """Worker: new STEP (or native part) → NEW shapes + CAD metadata; the remapping against the displayed parts
            runs in the GUI thread (it reads the shapes shown in the 3D view)."""
            p = native_to_step(path, warn) if is_native(path) else path   # modified Inventor / Fusion part
            parts = read_step(p, report=warn)
            if not parts:
                raise RuntimeError("The STEP contains no solid")
            return p, parts, cad_import.load_sidecar(old_step), cad_import.load_sidecar(p, warn)
        try:
            path, new_parts, old_meta, new_meta = self.run_busy(read, f"Reading {path.name}…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            rep = plan_update(self.study, self.solid_parts(), new_parts, old_meta, new_meta)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            msg(self, APP_NAME, f"The new geometry cannot be loaded:\n{e}", QMessageBox.Critical)
            return {"applied": False, "reason": str(e)}
        QApplication.restoreOverrideCursor()
        reject, reject_anchors = set(), set()
        if rep.needs_review():
            from PySide6.QtWidgets import QDialog
            from .remap_dialog import RemapDialog
            dlg = RemapDialog(self, rep, f"{path.name}: {len(new_parts)} parts")
            if dlg.exec() != QDialog.Accepted:
                self.log_msg("Geometry update cancelled – the current model is kept.")
                return {"applied": False, "reason": "cancelled"}
            reject, reject_anchors = dlg.rejected_selections(), dlg.rejected_anchors()
        new = apply_report(self.study, rep, new_step=str(path), new_parts=new_parts, new_meta=new_meta,
                           trigger=trigger, results=self.results, old_parts=self.solid_parts(),
                           reject=reject, reject_anchors=reject_anchors)
        self.motion_result = self._motion_sig = None
        self._load_model(path, new, None, parts=new_parts, geo_warn=warn)
        self.dirty = True
        self.reset_undo()                         # undo/redo boundary: snapshots do not contain the geometry
        self.update_title()
        it = new.cad_link.iterations[-1]
        self.log_msg(f"Geometry updated ({path.name}) – iteration {it.n}; selections: "
                     + ", ".join(f"{k} {v}" for k, v in it.remap.items()))
        items = [("info", f"Iterace {it.n}: " + ("; ".join(it.changes) or "geometry unchanged"))]
        items += [("warning", n) for n in rep.notes]
        blocking = rep.blocking() + [s for s in reject if s not in rep.blocking()]
        if blocking:
            items.append(("warning", "Selections without remapping (fix the faces): " + ", ".join(blocking)))
        self.messages.set_source("geometry update", items, "Model")
        if solve and not blocking and not rep.notes:
            QTimer.singleShot(0, self.run_solve)
        return {"applied": True, "iteration": it.n, "remap": it.remap, "not_remapped": blocking}

    def open_iterations(self):
        if self.study is None or self.study.cad_link is None or not self.study.cad_link.iterations:
            msg(self, APP_NAME, "The study has no geometry iterations yet – they are created by the first geometry update (Tools → Replace geometry, CAD add-in \"Update in FEA Studio\").")
            return
        from .remap_dialog import IterationsDialog
        IterationsDialog(self, self.study).exec()

    def open_modules_dialog(self):
        from .modules_dialog import ModulesDialog
        if getattr(self, "_modules_dlg", None) is None:
            self._modules_dlg = ModulesDialog(self)
        self._modules_dlg.show()
        self._modules_dlg.raise_()

    def open_calc_dialog(self):
        from .calc_dialog import CalcDialog
        if getattr(self, "_calcs_dlg", None) is None:
            self._calcs_dlg = CalcDialog(self)
        self._calcs_dlg.show()
        self._calcs_dlg.raise_()

    def open_beam_calc(self):
        from .beam_calc import BeamCalcDialog
        if getattr(self, "_calc_dlg", None) is None:
            self._calc_dlg = BeamCalcDialog(self)
        self._calc_dlg.show()
        self._calc_dlg.raise_()

    # ================================================================== file
    def geometry_str(self) -> str:
        """Path to the STEP for Study.geometry ("" = frame system without CAD)."""
        return str(self.step_path) if self.step_path else ""

    def frame_only(self) -> bool:
        """Study with frames only (no STEP geometry)."""
        return self.study is not None and not self.parts and self.step_path is None

    @_batched
    def new_frame_study(self):
        """New frame-system study without STEP geometry (Study.geometry == "")."""
        if self.busy():
            msg(self, APP_NAME, "Analysis in progress – wait for it to finish.")
            return
        if not self.confirm_discard():
            return
        st = Study(name="Frame", geometry="")
        st.add_material(matlib.default(), default=True)
        st.load_cases.append(LoadCase("Load case 1"))
        self._load_model(None, st, None)
        self.project_path = None
        self.dirty = False
        self.reset_undo()
        self.update_title()
        self.set_stage(STAGE["Members"])

    def open_dialog(self):
        last = self.settings.value("last_dir", "")
        p, _ = QFileDialog.getOpenFileName(self, "Open", last,
                                           "Model nebo projekt (*.step *.stp *.fea *.ipt *.f3d *.f3z);;Projekt FEA "
                                           f"(*.fea);;STEP (*.step *.stp);;{NATIVE_FILTER}")
        if p:
            self.settings.setValue("last_dir", str(Path(p).parent))
            self.open_file(p)

    # ------------------------------------------------------------------ dropping a file onto the window
    @staticmethod
    def dropped_files(mime) -> list[Path]:
        """Files .step / .stp / .fea dropped (Explorer, desktop, CAD)."""
        if mime is None or not mime.hasUrls():
            return []
        return [Path(u.toLocalFile()) for u in mime.urls()
                if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in DROP_SUFFIXES]

    @staticmethod
    def _accept_copy(e):
        """Accept a drop always as a copy: the program only reads the file. The suggested action with Shift would be a move
        (MoveAction) and Explorer would then delete the original file.
        """
        e.setDropAction(Qt.CopyAction)
        e.accept()

    def dragEnterEvent(self, e):
        if self.dropped_files(e.mimeData()):
            self._accept_copy(e)
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if self.dropped_files(e.mimeData()):
            self._accept_copy(e)
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        files = self.dropped_files(e.mimeData())
        if not files:
            super().dropEvent(e)
            return
        self._accept_copy(e)
        if len(files) > 1:
            self.log_msg(f"{len(files)} files dropped – only the first one is opened: {files[0].name}")
        self.raise_()
        self.activateWindow()
        # open only after returning from the drop: the "Save changes?" question inside dropEvent would block Explorer
        QTimer.singleShot(0, lambda p=str(files[0]): self.open_file(p))

    def busy(self) -> bool:
        """Meshing, calculation or report is running (the thread works with the study or the mesh)."""
        return (self._thread is not None or getattr(self, "_report_thread", None) is not None
                or getattr(self.fat_panel, "_thread", None) is not None or bool(self._aux_threads)
                or self._busy_task is not None or self._mcp_job_active())

    def run_busy(self, fn, title: str, on_done=None, cancellable: bool = False, on_cancel=None):
        """Runs fn(thread) in a worker thread (TaskThread) while the window stays painted: a modal progress window
        (shown after 0.3 s, Cancel sets thread.cancel when cancellable), user input to other windows is blocked. Returns
        fn's result (or on_done(result)); the exception of fn is raised again here (in the GUI thread).
        Rules for fn: no Qt widgets, no triangulation of shapes displayed in the 3D view (read new shapes / use
        shadow_parts()).
        """
        from PySide6.QtCore import QEventLoop
        from .workers import TaskThread
        if self._busy_task is not None:
            raise RuntimeError("Another operation is in progress – wait for it to finish.")
        box: dict = {}

        def job(th):
            try:
                box["result"] = fn(th)
            except BaseException as e:  # noqa: BLE001 – raised again in the GUI thread
                box["error"] = e
                box["trace"] = traceback.format_exc()
        t = TaskThread(job, self)
        loop = QEventLoop()
        t.finished.connect(loop.quit)
        dlg = _busy_progress(title, cancellable, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setToolTip("A long operation is running – the window stays responsive")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(10 ** 9)            # shown by the timer below (QProgressDialog shows only on setValue)
        dlg.hide()

        def _cancel():
            if t.cancel.is_set():
                return
            t.cancel.set()
            dlg.setLabelText(f"{title}\n{tr('Cancelling…')}")
            # QProgressDialog.cancel() hides the window – keep it shown until the worker really stops
            QTimer.singleShot(0, dlg, lambda: dlg.show() if t.isRunning() else None)
            if on_cancel is not None:
                on_cancel()
        dlg.canceled.connect(_cancel)
        blocker = _InputBlocker(dlg)
        app = QApplication.instance()
        app.installEventFilter(blocker)
        self._busy_task = t
        cursor = QApplication.overrideCursor() is None
        if cursor:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        show = QTimer(self)
        show.setSingleShot(True)
        show.timeout.connect(lambda: dlg.show() if t.isRunning() else None)
        try:
            t.start()
            show.start(300)
            if not t.isFinished():
                loop.exec()
            t.wait()
        finally:
            show.stop()
            app.removeEventFilter(blocker)
            self._busy_task = None
            if cursor:
                QApplication.restoreOverrideCursor()
            dlg.hide()
            dlg.deleteLater()
        if "error" in box:
            self.log.appendPlainText(f"{title} failed – traceback:\n{box['trace'].rstrip()}")
            raise box["error"]
        res = box.get("result")
        return on_done(res) if on_done is not None else res

    def shadow_parts(self, fresh: bool = False) -> list:
        """Parts of the current STEP read again (new OCP shapes, never displayed) for OCP work in worker threads –
        intersections / triangulation of the shapes shown in the 3D view would collide with the GUI thread.
        Call it in the worker. The cached copy is for run_busy jobs (one at a time); a thread that may run in parallel
        with them (mesh, print variants) takes fresh=True = its own copy."""
        if self.step_path is None:
            return []
        p = Path(self.step_path)
        try:
            key = (str(p), p.stat().st_mtime_ns, legacy_step_names(self.study))
        except OSError:
            key = (str(p), 0, False)
        sh = self._shadow
        if not fresh and sh is not None and sh[0] == key:
            return sh[1]
        parts = read_step(p, legacy_names=key[2])
        if self.surface_names:                     # plate parts of shells (read_step skips them)
            from fea_core.shell.geometry import read_step_surfaces
            parts += [q for q in read_step_surfaces(p) if q.name in self.surface_names]
        order = {q.name: i for i, q in enumerate(self.parts)}       # same parts in the same order as displayed
        parts = sorted((q for q in parts if q.name in order), key=lambda q: order[q.name])
        if not fresh:
            self._shadow = (key, parts)
        return parts

    def shadow_solid_parts(self) -> list:
        """shadow_parts(fresh=True) without the plate parts of shells – for the 3D mesh in a worker thread."""
        return [p for p in self.shadow_parts(fresh=True) if p.name not in self.surface_names]

    @_batched
    def open_file(self, path):
        if self.busy():
            msg(self, APP_NAME, "Meshing, analysis or report generation is in progress – wait for it to finish or cancel the analysis (Solve / Mesh step).")
            return
        if not self.confirm_discard():
            return
        path = opened = Path(path)
        old_study = self.study
        wd = self._workdir / f"proj_{int(time.time() * 1000)}"

        def read(_th):
            """File reading in the worker thread (zip / JSON / native CAD / STEP → NEW OCP shapes, not displayed yet)."""
            out = {"motion": None, "surrogate": None, "surrogate_warn": [], "native_warn": [], "geo_warn": []}
            suf = path.suffix.lower()
            if suf == ".fea":
                study, step, results = load_project(path, wd)
                from fea_core.project import load_motion, load_surrogate_file
                out["motion"] = load_motion(wd)             # mechanism result (Motion step), may be missing
                out["surrogate"] = load_surrogate_file(wd, out["surrogate_warn"])  # surrogate of a parametric study; a damaged one is dropped, the project opens
            elif suf == ".json":                           # study JSON (fea_core run, MCP fea_study_save, REST)
                study, step = load_study_json(path)        # Save then asks for the .fea project (JSON is not overwritten)
                results = None
            else:
                study, results, step = None, None, path
                if is_native(path):                        # .ipt / .f3d → STEP in the cache (module native)
                    step = native_to_step(path, out["native_warn"])
            out["parts"] = (read_step(step, report=out["geo_warn"], legacy_names=legacy_step_names(study))
                            if step is not None else None)
            out.update(study=study, step=step, results=results)
            return out
        try:
            r = self.run_busy(read, f"Opening {path.name}…")
            if path.suffix.lower() == ".fea":
                self.motion_result = r["motion"]
                self._motion_sig = mech_signature(r["study"]) if self.motion_result is not None else None
                new_path = path
            else:
                self.motion_result = self._motion_sig = None
                new_path = None
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._load_model(r["step"], r["study"], r["results"], parts=r["parts"], geo_warn=r["geo_warn"])
            self.surrogate_model = (r or {}).get("surrogate")
            for w in (r or {}).get("surrogate_warn", []):
                self.log_msg(w)
                self.messages.add("warning", "results", w, "Solve")
            if self.surrogate_model is not None and self.study is not None:
                why = self.surrogate_model.check_study(self.study, self.geometry_str())      # a project saved with a study that changed since
                if why:
                    self.messages.add("warning", "results", "The surrogate model of this project belongs to an earlier version of the study: " + why,
                                      "Solve")
            for w in r["native_warn"]:
                self.log_msg(f"Native file: {w}")
                self.messages.add("warning", "geometry check", w, "Model")
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            if self.study is not old_study:
                # model already replaced, error only when the panels are restored: the study belongs to the new file → the path
                # and title of the previous project must not stay (Ctrl+S would overwrite it with a foreign study); save is possible only as a new file
                self.project_path = None
                self.dirty = True
                self.reset_undo()
                self.update_title()
            msg(self, APP_NAME, f"The file cannot be opened:\n{e}", QMessageBox.Critical)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.project_path = new_path               # only after a successful load (Ctrl+S does not overwrite another file)
        self.dirty = False
        self.reset_undo()
        self.update_title()
        session_files.note_opened(self, opened)    # File → Recent files
        self.set_stage(STAGE["Materials"] if self.results is None else STAGE["Results"])

    def _load_model(self, step: Path | None, study: Study | None, results, parts=None, geo_warn=None):
        self.surrogate_model = None              # a surrogate belongs to the study it was fitted for (the open flow sets it after loading)
        self._flow_revealed = set()              # optional steps open in the previous study are not kept on the bar
        self.model_checks = None                 # the model quality checks (F14 S3) belong to the previous model, also the run kept by the MCP session
        self._space_checks_ui = None
        mcp_sess = getattr(self, "_mcp_session", None)
        if mcp_sess is not None:
            mcp_sess.model_checks = None
        geo_warn = list(geo_warn or [])          # geometry check (only when a model is loaded, not when calculating)
        legacy = legacy_step_names(study)        # saved study with parts "NAUO1" (SolidWorks STEP, version 1.0.0)
        if step is None:                         # frame system without CAD (Study.geometry == "")
            parts = []
        elif parts is None:                      # parts = parts already loaded (geometry update)
            parts = read_step(step, report=geo_warn, legacy_names=legacy)
        surfaces = []                            # plate parts from STEP (read_step skips them) – shells
        want = set((study.shell.parts if study is not None else {}) or {})
        if step is not None and (not parts or want - {p.name for p in parts}):
            from fea_core.shell.geometry import read_step_surfaces
            try:
                surfaces = [p for p in read_step_surfaces(step) if p.name not in {q.name for q in parts}]
            except Exception as e:  # noqa: BLE001
                geo_warn.append(f"cannot load surface parts: {e}")
        if step is not None and not parts and not surfaces:
            raise RuntimeError("The STEP file contains no body or face")
        self.surface_names = {p.name for p in surfaces}
        if surfaces:                             # faces were loaded as plate parts of shells – the read_step warning does not apply
            geo_warn = [g for g in geo_warn if "no volume" not in g and "no solid" not in g]
        parts = list(parts) + surfaces
        self.step_path = Path(step) if step is not None else None
        self.parts = parts
        # part names from the CAD (and the file name = default study name) are not translated in the English GUI
        keep_names([p.name for p in parts] + ([self.step_path.stem] if self.step_path else []))
        self.face_cache = {}
        self.mesh = results.mesh if results is not None else None
        self.results = results
        self.fat_panel.result = None             # fatigue and joint assessment belong to the previous model
        self.joints_panel.assessment = None
        self._rigid = None
        new_study = study is None and step is not None
        cad_msgs: list[tuple[str, str]] = []
        cad_log: list[str] = []
        if study is None:
            study = Study(name=self.step_path.stem, geometry=str(self.step_path))
            study.add_material(matlib.default(), default=True)
            bb = np.array([bbox(p.shape) for p in parts])
            diag = float(np.linalg.norm(bb[:, 3:].max(axis=0) - bb[:, :3].min(axis=0)))
            study.mesh.size = nice_floor(diag / 30)
            study.load_cases.append(LoadCase("Load case 1"))
            if surfaces and len(surfaces) == len(parts):    # only faces → shell model (the user enters the thickness)
                from fea_core.shell.model import ShellSettings
                study.shell = ShellSettings(True, {p.name: None for p in surfaces})
        if new_study:                            # new study from a STEP: materials from the CAD metadata (<name>.json next to it)
            cad_msgs, cad_log = self._cad_metadata(Path(step), study, [p.name for p in parts])
            try:                                 # fastener components of the CAD assembly → virtual bolts
                from fea_core import cad_import
                added, bw = cad_import.apply_fasteners(study, parts, cad_import.load_sidecar(Path(step)))
            except Exception as e:  # noqa: BLE001 – bolts are an extra, the model opens without them too
                added, bw = [], [f"Virtual bolts from the assembly cannot be created: {e}"]
            if added:
                text = (f"Virtual bolts from the CAD assembly ({len(added)}): {', '.join(added[:8])}"
                        + (" …" if len(added) > 8 else "") + " – assessment in the Joints step")
                cad_msgs.append(("info", text))
                cad_log.append(text)
            for w in bw:
                cad_msgs.append(("warning", w))
                cad_log.append(w)
        self.study = study
        self._mesh_sig = self.study_signature(self.MESH_KEYS) if self.mesh is not None else None
        self.view.set_parts(parts)
        self.view.clear_persist()
        self.log_msg(f"Loaded: {self.step_path.name} – {len(parts)} bodies" if self.step_path else
                     f"Frame without CAD: {len(study.frame.nodes)} nodes, {len(study.frame.members)} members")
        self.geometry_warnings = list(geo_warn)   # read by the model info, the properties panel and the table dock
        for w in geo_warn:
            self.log_msg(f"Geometry check: {w}")
        if legacy:
            self.log_msg("STEP parts are named as in the older program version (SolidWorks occurrences “NAUO1”…) so that the materials, selections and bolts of the saved study stay valid")
        name_warn: list[str] = []                # only the message and the log, not a window (selected faces are reported by resolve_all)
        if not new_study and step is not None:   # material of a part that is not in the STEP (renamed in the CAD?)
            names = {p.name for p in parts}
            lost = [n for n in study.part_materials if n not in names]
            if lost:
                listing = ", ".join(lost[:8]) + (" …" if len(lost) > 8 else "")
                name_warn.append(f"A material is assigned to parts that are not in the STEP file (renamed in CAD?): {listing}")
                self.log_msg(name_warn[-1])
        for line in cad_log:
            self.log_msg(line)
        for src in ("mesh", "solver", "results"):           # messages belong to the previous model
            self.messages.set_source(src, [])
        if geo_warn or name_warn:
            self.messages.set_source("geometry check", [("warning", w) for w in geo_warn + name_warn], "Model")
        else:
            self.messages.set_source("geometry check", [])
        self.messages.set_source(CAD_SOURCE, cad_msgs, "Materials")
        if self._path_dlg is not None:
            self._path_dlg.close()
        self.messages.set_source(F8_SOURCE, [])
        if results is not None:
            self.mesh_messages(results.mesh)
            self.messages.set_source("results", [("warning", w) for w in results.warnings], unique=True)
            self.f8_messages(results)
        missing = self.resolve_all()
        self.refresh_all()
        self._notify_space_checks()
        if geo_warn:
            msg(self, APP_NAME, "The geometry check found problems (the calculation may fail or be inaccurate):\n\n"
                + "\n".join(geo_warn[:20]) + ("\n…" if len(geo_warn) > 20 else ""))
        if missing:
            msg(self, APP_NAME, "Some selected faces were not found in the model (did the model change?):\n\n"
                + "\n".join(missing))

    def _cad_metadata(self, step: Path, study: Study, part_names: list[str]) -> tuple[list, list[str]]:
        """CAD metadata next to the STEP (<name>.json – Fusion add-in, SolidWorks / Inventor macros, REST import):
        materials of the parts by matching the library (fea_core.cad_import). Returns (messages [(level, text)], log lines);
        nothing blocks – a material outside the library is only a warning, the part gets the default material of the study.
        """
        from fea_core import cad_import
        warn: list[str] = []
        try:
            meta = cad_import.load_sidecar(step, warn)
            if meta is None:
                return [("warning", w) for w in warn], list(warn)
            res = cad_import.apply_metadata(study, part_names, meta)
        except Exception as e:  # noqa: BLE001 – metadata is an extra, the model opens without it too
            text = f"CAD metadata cannot be used: {e}"
            return [("warning", text)], [text]
        items = res.messages(study.default_material)
        lines = [f"{res.summary()} – file {cad_import.sidecar_path(step).name}"]
        for r in res.rows:
            if r.get("material"):
                lines.append(f"Material from CAD: {r['name']} – “{r['material_match']['cad_material']}” → {r['material']}")
        lines += [t for _, t in items[1:]]
        return items, lines

    def resolve_all(self) -> list[str]:
        self.sel_faces, self.sel_edges, self.sel_points = {}, {}, {}
        missing = []
        names = [p.name for p in self.parts]
        for name, sel in self.study.selections.items():
            faces = []
            for fsel in sel.faces:
                r = resolve(fsel, self.parts, self.face_cache)
                if r is None:
                    missing.append(f"{name}: {fsel.describe()}")
                else:
                    faces.append(r)
            self.sel_faces[name] = faces
            if sel.edges:
                edges = []
                for esel in sel.edges:
                    r = resolve_edge(esel, self.parts, self.edge_cache)
                    if r is None:
                        missing.append(f"{name}: {esel.describe()}")
                    else:
                        edges.append(r)
                self.sel_edges[name] = edges
            if sel.points:
                pts = []
                for psel in sel.points:
                    if psel.part in names:
                        pts.append((names.index(psel.part), tuple(psel.point)))
                    else:
                        missing.append(f"{name}: {psel.describe()}")
                self.sel_points[name] = pts
        return missing

    def save(self):
        if self.project_path is None:
            return self.save_as()
        return self._save(self.project_path)

    def export_report(self):
        from .report_dialog import export_report
        if self.study is None:
            msg(self, APP_NAME, "Open a model or project first.")
            return
        export_report(self)

    def save_as(self):
        if self.study is None:
            return False
        last = self.settings.value("last_dir", "")
        default = str(Path(last) / (self.study.name + ".fea")) if last else self.study.name + ".fea"
        p, _ = QFileDialog.getSaveFileName(self, "Save project", default, "FEA project (*.fea)")
        if not p:
            return False
        if not p.lower().endswith(".fea"):
            p += ".fea"
        return self._save(Path(p))

    def _save(self, path: Path):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            sur_warn: list[str] = []
            save_project(path, self.study, self.geometry_str(), self.results, motion=self.motion_result,
                         surrogate=getattr(self, "surrogate_model", None), warnings=sur_warn)
        except Exception as e:  # noqa: BLE001
            msg(self, APP_NAME, f"Saving failed:\n{e}", QMessageBox.Critical)
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.project_path = path
        self.dirty = False
        self.update_title()
        self.log_msg(f"Saved: {path}")
        for w in sur_warn:
            self.log_msg(w)
            self.messages.add("warning", "results", w, "Solve")
        session_files.note_saved(self, path)       # recent files; the autosave of this session is no longer needed
        link = self.study.cad_link
        if link is not None and link.document_id:    # the CAD add-in finds the project even without a running GUI
            from fea_core.cad_link.registry import remember
            remember(link.document_id, path, link.document)
        return True

    def confirm_discard(self):
        if not self.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "The study has unsaved changes. Save?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return bool(self.save())
        return r == QMessageBox.Discard

    def changed(self, invalidate_results=True, invalidate_mesh=False):
        """The study was edited: discard what the change invalidates, undo snapshot, title, refresh of the panels."""
        level = "mesh" if invalidate_mesh else ("results" if invalidate_results else "none")
        return self.mark_changed(level, refresh=True)

    # effect of a study change (shared with undo / redo – _restore): names ↔ fea_core.joints.levels.LEVELS
    CHANGE_LEVELS = ("none", "assess", "results", "mesh")

    def mark_changed(self, level: str = "results", refresh: bool | None = None) -> bool:
        """One place for "the study changed": level = what the change invalidates –
        "none" (only the file: display, fatigue settings, recognition rules), "assess" (only post-processing / joint
        assessment – the mesh and the FEM results stay; the caller recomputes the assessment), "results" (FEM results),
        "mesh" (mesh and results). Marks the study dirty, writes an undo snapshot and the title; refresh (default for
        results / mesh) refreshes the panels. Returns False when the user kept the running analysis and the change was reverted.
        """
        if level not in self.CHANGE_LEVELS:
            raise ValueError(f"unknown change level {level!r}")
        if not self._confirm_change_while_busy(level):
            return False
        self.dirty = True
        self._discard_for(level)
        self.push_undo()
        self.update_title()
        self._notify_space_checks()
        if refresh is None:
            refresh = level in ("results", "mesh")
        if refresh:
            if self._change_batch:                   # inside batch_changes(): one refresh at the end
                self._refresh_pending = True
            else:
                self.refresh_all()
                self._sync_mesh_view()
        return True

    def _notify_space_checks(self) -> None:
        """The model quality checks dialog (if open) follows the study: a run that no longer belongs to it is shown as out of date (F14 S3)."""
        dlg = getattr(self, "_space_checks_dlg", None)
        if dlg is not None:
            dlg.on_study_changed()

    def _discard_for(self, level: str):
        """Discards what a change of the given level invalidates (mark_changed, undo / redo)."""
        if level == "mesh":
            self.mesh = None
        if level in ("results", "mesh"):
            if self.results is not None:
                self.results = None
                self.log_msg("Study changed – results discarded, recomputation needed.")
                self.messages.set_source("results", [])
                self.messages.set_source(F8_SOURCE, [])
                if self._path_dlg is not None:
                    self._path_dlg.close()
            self.fat_panel.result = None
            self._space_3axis = None                      # the kept 3-axis launch result belongs to the model it was computed for (it is digest-checked as well)
        if level != "none":
            self.joints_panel.assessment = None

    def batch_changes(self):
        """Context: several changed() calls inside refresh the panels only once at the end."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self._change_batch += 1
            try:
                yield
            finally:
                self._change_batch -= 1
                if not self._change_batch:
                    self.flush_refresh()
        return _cm()

    def flush_refresh(self):
        """Refresh postponed by batch_changes() (immediately, e.g. in tests)."""
        if self._refresh_pending and not self._change_batch:
            self._refresh_pending = False
            self.refresh_all()
            self._sync_mesh_view()

    def _thread_result_stale(self, level: str) -> bool:
        """Would the result of the running mesh / analysis thread be discarded after this change (signature check at its end)?"""
        t = self._thread
        if t is None:
            return self._mcp_job_active() and self._mcp_job_stale()
        if self.study is None or self._thread_sig is None:
            return level != "none"
        if isinstance(t, MeshThread):
            return self.study_signature(self.MESH_KEYS) != self._thread_sig
        if isinstance(t, SolveThread):
            return self.study_signature() != self._thread_sig
        return level != "none"                         # print variants / sweep (own study key)

    def _mcp_job_stale(self) -> bool:
        """A GUI-visible MCP job (fea_job_start) runs: do the calculation inputs differ from the study at its start
        (McpBridgeMixin._watch_mcp_job records it as _mcp_job_before)?"""
        before = getattr(self, "_mcp_job_before", None)
        if self.study is None or not before:
            return True
        try:
            old = self.calc_inputs(json.loads(before), joints=True)
        except ValueError:
            return True
        return json.dumps(old, sort_keys=True, default=str) != self.study_signature()

    def _confirm_change_while_busy(self, level: str) -> bool:
        """First change of the study while a mesh / analysis runs: the result of that run will be discarded at its end –
        ask once (Continue / Cancel = the change is reverted). Tests replace MainWindow.ask."""
        mcp = self._thread is None and self._mcp_job_active()
        job_before = getattr(self, "_mcp_job_before", None)
        warned = (job_before is not None and getattr(self, "_mcp_warned_for", None) == job_before) if mcp \
            else self._busy_warned
        if warned or not self._thread_result_stale(level):
            return True
        thread = self._thread
        self._asking_change = True               # a run finishing during the question waits for the answer (_defer_done)
        try:
            keep = self.ask("A mesh or analysis is running. This change makes its result invalid – it will be discarded when "
                            "the run finishes.\n\nKeep the change? (No = the change is reverted)")
        finally:
            self._asking_change = False
        running = self._mcp_job_active() if mcp else (self._thread is not None and self._thread is thread)
        if keep:
            if mcp:
                self._mcp_warned_for = job_before
            elif running:
                self._busy_warned = True         # accepted: further changes of this run do not ask again
        else:
            self._revert_to_snapshot(running)
        self._run_deferred_done()                # the end of a run that came during the question, judged after the answer
        return keep

    def _defer_done(self, fn) -> bool:
        """The end of a mesh / analysis arrived while the "change during a run" question is open: it is handled after
        the answer (No = the change is reverted and the result stays valid). True = deferred."""
        if not getattr(self, "_asking_change", False):
            return False
        self._deferred_done = getattr(self, "_deferred_done", []) + [fn]
        return True

    def _run_deferred_done(self):
        pending, self._deferred_done = getattr(self, "_deferred_done", []), []
        for fn in pending:
            fn()

    def _revert_to_snapshot(self, running: bool = True):
        """Back to the last undo snapshot (the state before the change that was just made); results and mesh stay."""
        if self.study is None or not self._undo:
            return
        link = self.study.cad_link
        self.study = Study.from_dict(json.loads(self._undo[self._undo_pos]))
        self.study.cad_link = link
        self.resolve_all()
        self.refresh_all()
        self.log_msg("Change reverted – the running analysis stays valid." if running
                     else "Change reverted – the result of the finished run stays valid.")

    def _sync_mesh_view(self):
        """Mesh discarded (change of mesh settings, refinement, selections…), but the 3D window still shows it → back to the parts
        with the face selection.
        """
        if getattr(self, "_mesh_view", False) and self.mesh is None:
            self.set_stage(self.stage)

    def on_mesh_show(self, _=None):
        """The "Show mesh" switch in the Mesh step: the mesh surface, or the parts with face selection (refinement)."""
        if STAGES[getattr(self, "stage", 0)] == "Mesh":
            self.set_stage(self.stage)

    # ------------------------------------------------------------------ undo / redo
    def reset_undo(self):
        self._undo = [json.dumps(self.study.to_dict())] if self.study else []
        self._undo_pos = len(self._undo) - 1

    def push_undo(self):
        if self.study is None:
            return
        snap = json.dumps(self.study.to_dict())
        if self._undo and self._undo[self._undo_pos] == snap:
            return
        del self._undo[self._undo_pos + 1:]
        self._undo.append(snap)
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._undo_pos = len(self._undo) - 1

    def undo(self):
        if self.study is not None and self._undo_pos > 0:
            self._restore(self._undo_pos - 1)

    def redo(self):
        if self.study is not None and self._undo_pos < len(self._undo) - 1:
            self._restore(self._undo_pos + 1)

    # effect of a change: nothing < assessment only < FEA calculation < mesh; the joint settings by effect are in the core
    # (fea_core.joints.levels – shared with MCP fea_joints_*)
    LEVELS = JOINT_LEVELS

    def joints_change_level(self, old: dict, new: dict) -> str:
        """Effect of the difference in bolts / welds / joint settings between two dictionaries of the study (undo / redo) – the same as
        changes in the Joints step (joints_panel.change_level). Unclear (other bolts / welds, order) = new mesh.
        """
        oj, nj = old.get("joints") or {}, new.get("joints") or {}
        lv = self.LEVELS.index(joint_settings_level(oj, nj))
        preload = bool(nj.get("preload_case", True))
        for key, kind in (("bolts", "bolt"), ("welds", "weld")):
            a, b = old.get(key) or [], new.get(key) or []
            if [x.get("name") for x in a] != [x.get("name") for x in b]:
                return "mesh"
            for x, y in zip(a, b):
                lv = max(lv, self.LEVELS.index(change_level(kind, x, y, preload) or ""))
        return self.LEVELS[lv]

    def _restore(self, pos):
        old = self.study.to_dict()
        new = json.loads(self._undo[pos])
        mesh_ok = self.mesh is not None and self.valid_mesh() is not None     # before switching the study
        self._undo_pos = pos
        link = self.study.cad_link                 # the iteration history is not restored (it is not a settings change)
        self.study = Study.from_dict(new)
        self.study.cad_link = link
        old.pop("cad_link", None)
        new.pop("cad_link", None)
        self.resolve_all()

        # post-processing fields (EN 1990 categories, combinations, printing sweep / variants…) are not calculation inputs;
        # the joints stay complete – joints_change_level decides about their assessment-only fields
        old_c, new_c = self.calc_inputs(old), self.calc_inputs(new)

        def differ(keys):
            return [k for k in keys
                    if json.dumps(old_c.get(k), sort_keys=True) != json.dumps(new_c.get(k), sort_keys=True)]
        joints = ("bolts", "welds", "joints")
        try:
            jl = self.joints_change_level(old, new) if differ(joints) else ""
        except Exception as e:  # noqa: BLE001 – unclear = discard as before
            self.log_msg(f"Undo: the effect of the joint changes is unclear ({e}) – the mesh is discarded")
            jl = "mesh"
        level = self.LEVELS.index(jl)
        if differ(("geometry", "mesh", "selections", "ignored_parts")):
            level = 3
        elif differ((set(old_c) | set(new_c)) - set(joints)):
            level = max(level, 2)                  # other calculation inputs (constraints, loads, materials…)
        # assessment (required safety, Stress Tool) is post-processing only – like the fatigue settings it keeps the results
        if level < 3 and mesh_ok and differ(joints):
            self._mesh_sig = self.study_signature(self.MESH_KEYS)   # joints outside the meshing inputs – the mesh stays valid
        # results are discarded only when the calculation inputs change (same rules as mark_changed)
        self._discard_for(self.CHANGE_LEVELS[level])
        if level == 1:                             # only the joint assessment – recalculate from the valid results
            self.joints_panel.compute(force=True)
        if level < 2 and json.dumps(old.get("fatigue"), sort_keys=True) != json.dumps(new.get("fatigue"), sort_keys=True):
            self.fat_panel.result = None
        self.dirty = True
        self.update_title()
        self.refresh_all()
        self.set_stage(STAGE["Solve"] if STAGES[self.stage] == "Results" and self.results is None else self.stage)
        self.log_msg("Undo" if pos < len(self._undo) - 1 else "Redo")

    def update_title(self):
        name = self.project_path.name if self.project_path else (self.step_path.name if self.step_path else
                                                                 (self.study.name if self.study else ""))
        self.setWindowTitle(f"{name}{' *' if self.dirty else ''} – {APP_NAME}" if name else APP_NAME)
        if hasattr(self, "doc_label"):
            title = self.study.name if self.study is not None and self.study.name else ""
            parts = [t for t in (title, name if name != title else "") if t]
            self.doc_label.setText(" · ".join(parts) + (" *" if self.dirty else ""))

    def cad_update_start(self, step: Path):
        """Start from the CAD add-in (`--cad-update STEP`): opens the project linked to the document (local registry)
        and uses the new geometry; an unknown project = the STEP as a new study.
        """
        from fea_core import cad_import
        from fea_core.cad_link.registry import lookup
        meta = cad_import.load_sidecar(step) or {}
        link = meta.get("link") if isinstance(meta.get("link"), dict) else {}
        fea = lookup(str(link.get("document_id", "")))
        if fea is None:
            self.open_file(str(step))
            return
        self.open_file(str(fea))
        if self.study is not None and self.project_path == fea:
            self.update_geometry(step, "cad")

    def _same_cad_document(self, step: Path) -> bool:
        """Does the new STEP from the add-in belong to the CAD document of the open project? An unknown id on one side = yes
        (STEP without metadata, older project); known and different ids = no – otherwise the settings of project A would be remapped
        to the geometry of document B (and calculated at once).
        """
        from fea_core import cad_import
        meta = cad_import.load_sidecar(step) or {}
        link = meta.get("link") if isinstance(meta.get("link"), dict) else {}
        new_id = str(link.get("document_id", "") or "")
        old_id = self.cad_link_info().get("document_id", "") if self.study is not None else ""
        return not new_id or not old_id or new_id == old_id

    def cad_link_info(self) -> dict:
        """For the CAD add-in: the open study, the CAD document id and the tokens of the faces / bodies the study uses (the add-in
        translates them to the new version of the model – findEntityByToken).
        """
        d = {"gui_running": True, "study": self.study.name if self.study is not None else None,
             "busy": self.busy(), "tokens": [], "body_tokens": []}
        if self.study is None or self.step_path is None:
            return d
        link = self.study.cad_link
        d["document_id"] = link.document_id if link is not None else ""
        try:
            from fea_core import cad_import
            from fea_core.cad_link.facetable import used_tokens
            meta = cad_import.load_sidecar(self.step_path)
            d["tokens"], d["body_tokens"] = used_tokens(self.study, self.solid_parts(), meta)
            ml = (meta or {}).get("link") if isinstance((meta or {}).get("link"), dict) else {}
            d["document_id"] = d["document_id"] or str(ml.get("document_id", ""))
        except Exception as e:  # noqa: BLE001 – the add-in then translates all tokens of the last export
            d["error"] = str(e)
        return d

"""The sketch editor in the main window: File → New → "2D model from sketch…" / "Beam system from sketch…", Model step → "Edit sketch…", and
the building of the edited sketch into the study (first build, or a rebuild with the setup kept through the CAD-link remap, with the
remap window when something needs a look).

`win` is the MainWindow (typed as Any: the functions use only its public document API – study, project_path, dirty, _load_model,
reset_undo / push_undo, run_busy …)."""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout)

from fea_core import materials as matlib
from fea_core.sketch.build import apply_rebuild, attach_first, build_geometry, plan_rebuild
from fea_core.sketch.expr import SketchError
from fea_core.sketch.model import MODES, Sketch, new_sketch
from fea_core.study import LoadCase, Study

from ..main_defs import APP_NAME, STAGE, msg_via_main as msg
from ..widgets import help_tip
from .editor import MODE_LABEL, SketchEditor


class NewSketchDialog(QDialog):
    """Kind of the new sketch: the problem type and the thickness of a 2D model, or the plane of a beam system."""

    PLANES = {"XZ – vertical plane (x right, y up = global Z)": "XZ", "XY – horizontal plane": "XY", "YZ – vertical plane": "YZ"}

    def __init__(self, parent, kind: str):
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle("New 2D model from sketch" if kind == "twod" else "New beam system from sketch")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self.mode = QComboBox()
        self.thickness = QLineEdit("10")
        self.plane = QComboBox()
        if kind == "twod":
            for m in MODES:
                self.mode.addItem(MODE_LABEL[m], m)
            help_tip(self.mode, "Plane stress (plate), plane strain (long prism), axisymmetric", "Plane stress (thin plate), plane strain (long prism), axisymmetric (x = radius, y = axis)")
            form.addRow("Problem", self.mode)
            self.thickness.setToolTip("Thickness of a planar problem in mm (a number or an expression)")
            form.addRow("Thickness [mm]", self.thickness)
            self.mode.currentIndexChanged.connect(lambda _i: self.thickness.setEnabled(self.mode.currentData() != "axisym"))
            self.plane.addItem("XY – the sketch plane is the global XY plane", "XY")
            self.plane.addItem("XZ", "XZ")
            self.plane.addItem("YZ", "YZ")
            help_tip(self.plane, "Where the sketch lies in global coordinates", "Where the sketch lies in global coordinates (gravity and other global vectors act in it accordingly)")
            form.addRow("Plane", self.plane)
        else:
            for text, key in self.PLANES.items():
                self.plane.addItem(text, key)
            help_tip(self.plane, "Plane of the frame in global coordinates", "The plane of the frame in global coordinates; the nodes are held against motion out of it")
            form.addRow("Plane", self.plane)
        note = QLabel("Typed coordinates in mm and named parameters; no constraint solver.")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def make(self) -> Sketch:
        if self.kind == "twod":
            t = self.thickness.text().strip() or "10"
            return new_sketch("twod", self.mode.currentData(), self.plane.currentData(), float(t) if t.replace(".", "", 1).isdigit() else t)
        return new_sketch("frame", None, self.plane.currentData())


# ------------------------------------------------------------------ entry points
def dispatch(win: Any, what: str) -> None:
    """twod | frame = a new study from a sketch; edit = the sketch of the open study."""
    if what == "edit":
        edit_sketch(win)
    else:
        new_sketch_study(win, what)


def new_sketch_study(win: Any, kind: str) -> bool:
    if win.busy():
        msg(win, APP_NAME, "Analysis in progress – wait for it to finish.")
        return False
    if not win.confirm_discard():
        return False
    dlg = NewSketchDialog(win, kind)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False
    try:
        sk = dlg.make()
    except SketchError as e:
        msg(win, APP_NAME, str(e), QMessageBox.Icon.Critical)
        return False
    ed = SketchEditor(win, sk, "New 2D model" if kind == "twod" else "New beam system")
    if ed.exec() != QDialog.DialogCode.Accepted:
        return False
    return build_new(win, ed.sketch, "Sketch" if kind == "twod" else "Frame sketch")


def edit_sketch(win: Any) -> bool:
    st = win.study
    if st is None or st.sketch is None:
        msg(win, APP_NAME, "This study has no sketch (File → New → 2D model from sketch… / Beam system from sketch…).")
        return False
    if win.busy():
        msg(win, APP_NAME, "Analysis in progress – wait for it to finish.")
        return False
    ed = SketchEditor(win, st.sketch.copy(), "Edit sketch")
    if ed.exec() != QDialog.DialogCode.Accepted:
        return False
    return apply_edit(win, ed.sketch)


# ------------------------------------------------------------------ building into the window
def _workdir(win: Any):
    wd = win._workdir / f"sketch_{int(time.time() * 1000)}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def build_new(win: Any, sk: Sketch, name: str) -> bool:
    """A new study from the finished sketch (the window shows it like an opened model)."""
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        wd = _workdir(win)
        st = Study(name=name, geometry="")
        st.add_material(matlib.default(), default=True)
        st.load_cases.append(LoadCase("Load case 1"))
        b = build_geometry(sk, wd, document=name)
        b.warnings += attach_first(st, b, sk)
        win._load_model(b.step_path, st, None, parts=b.parts if b.step_path else None, geo_warn=b.warnings)
    except SketchError as e:
        msg(win, APP_NAME, f"The sketch cannot be built:\n{e}", QMessageBox.Icon.Critical)
        return False
    finally:
        QApplication.restoreOverrideCursor()
    win.project_path = None
    win.dirty = True
    win.reset_undo()
    win.update_title()
    win.log_msg(f"Sketch built: {len(b.regions)} region(s)" if b.kind == "twod" else f"Beam sketch built: {len(b.frame.members)} members")
    win.set_stage(STAGE["Materials"] if b.kind == "twod" else STAGE["Members"])
    return True


def apply_edit(win: Any, sk: Sketch) -> bool:
    """The edited sketch of the open study goes into the model; the setup is kept (the remap window opens when something needs a look)."""
    st = win.study
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        first = not (st.sketch.built or {}).get("snapshot")
        if first:                                         # the study had a sketch that was never built
            b = build_geometry(sk, _workdir(win), document=st.name)
            b.warnings += attach_first(st, b, sk)
            win._load_model(b.step_path, st, None, parts=b.parts if b.step_path else None, geo_warn=b.warnings)
            win.dirty = True
            win.reset_undo()
            win.update_title()
            return True
        plan = plan_rebuild(st, _workdir(win), sketch=sk, old_parts=win.solid_parts() if win.step_path is not None else None)
    except SketchError as e:
        msg(win, APP_NAME, f"The sketch cannot be built:\n{e}", QMessageBox.Icon.Critical)
        return False
    finally:
        QApplication.restoreOverrideCursor()
    reject: set = set()
    reject_anchors: set = set()
    rep = plan.report
    if rep is not None and rep.needs_review():
        from .. import remap_dialog
        dlg = remap_dialog.RemapDialog(win, rep, "Sketch changed: " + (", ".join(p.old or p.new or "" for p in rep.parts[:4])))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            win.log_msg("Sketch update cancelled – the current model is kept.")
            return False
        reject, reject_anchors = dlg.rejected_selections(), dlg.rejected_anchors()
    try:
        new = apply_rebuild(plan, reject=reject, reject_anchors=reject_anchors)
    except SketchError as e:
        msg(win, APP_NAME, f"The sketch cannot be built:\n{e}", QMessageBox.Icon.Critical)
        return False
    b = plan.build
    if b.kind == "frame":
        win._load_model(None, new, None)
        win.push_undo()
    else:
        win._load_model(b.step_path, new, None, parts=b.parts, geo_warn=b.warnings)
        win.reset_undo()
    win.dirty = True
    win.update_title()
    items = [("info", "Sketch rebuilt: " + (", ".join(new.cad_link.iterations[-1].changes[:6]) if b.kind == "twod" and new.cad_link else "members updated"))]
    items += [("warning", w) for w in b.warnings]
    if rep is not None:
        items += [("warning", n) for n in rep.notes]
        blocking = rep.blocking()
        if blocking:
            items.append(("warning", "Selections without remapping (fix the faces): " + ", ".join(blocking)))
    win.messages.set_source("sketch", items, "Model")
    return True


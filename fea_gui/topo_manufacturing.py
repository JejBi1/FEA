"""Section "Manufacturing" of the topology optimisation dialog: symmetry planes, extrusion direction, overhang limit with the build direction.

The settings are turned into the `symmetry` / `extrusion` / `overhang` / `compare_unconstrained` fields of `fea_core.topo.TopologyOptSpec`
(`spec_kwargs`); the build direction is shown as the same orange arrow with the outline of the build plate as in the 3D printing step.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout)

from .widgets import help_tip

AXES = (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0)))
BUILD_DIRS = (("+Z (usual)", (0.0, 0.0, 1.0)), ("−Z", (0.0, 0.0, -1.0)), ("+X", (1.0, 0.0, 0.0)), ("−X", (-1.0, 0.0, 0.0)),
              ("+Y", (0.0, 1.0, 0.0)), ("−Y", (0.0, -1.0, 0.0)))
FROM_PRINT = "From the 3D printing settings"
LOWEST = "Lowest layer of the model"
BUILD_KEY = "print:build"                 # the same set / colour as the arrow of the 3D printing step
C_BUILD = (220, 120, 20)


class ManufacturingBox(QGroupBox):
    """Group box with the manufacturing constraints; `win` = the main window (3D view, selected faces), `study`, `mesh` = the model of the optimisation."""

    def __init__(self, win, study, mesh) -> None:
        super().__init__("Manufacturing constraints")
        self.win = win
        self.study = study
        nodes = np.asarray(mesh.nodes, dtype=float)
        self.lo, self.hi = nodes.min(axis=0), nodes.max(axis=0)
        self.centre = 0.5 * (self.lo + self.hi)
        self.setToolTip("Shapes that can be made: symmetric, prismatic, printable without supports")
        v = QVBoxLayout(self)

        # --- symmetry
        v.addWidget(QLabel("<b>Symmetry</b> – identical material on both sides of a mirror plane"))
        grid = QGridLayout()
        self.sym_on: list[QCheckBox] = []
        self.sym_pos: list[QDoubleSpinBox] = []
        for i, (name, _) in enumerate(AXES):
            cb = QCheckBox(f"Plane perpendicular to {name}")
            help_tip(cb, f"Mirror plane perpendicular to the {name} axis", "The density of the material is identical on both sides of the plane "
                     "(up to 3 mutually perpendicular planes). The mesh should be symmetric about the plane; elements without a mirror "
                     "image are reported.")
            sp = QDoubleSpinBox()
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(3)
            sp.setSuffix(" mm")
            sp.setValue(float(self.centre[i]))
            sp.setToolTip(f"{name} coordinate of the plane (default: the centre of the model)")
            cb.toggled.connect(sp.setEnabled)
            sp.setEnabled(False)
            grid.addWidget(cb, i, 0)
            grid.addWidget(sp, i, 1)
            self.sym_on.append(cb)
            self.sym_pos.append(sp)
        v.addLayout(grid)
        self.btn_sym_face = QPushButton("Plane from the selected face")
        help_tip(self.btn_sym_face, "Takes the position of a planar face selected in 3D",
                 "Select a planar face perpendicular to X, Y or Z in the 3D view before opening the dialog; its position sets the plane.")
        self.btn_sym_face.clicked.connect(self._sym_from_face)
        v.addWidget(self.btn_sym_face)

        # --- extrusion
        row = QHBoxLayout()
        self.ext_on = QCheckBox("Extrusion along")
        help_tip(self.ext_on, "The shape is a prism along the direction (milling, profile)",
                 "Elements are grouped into columns along the direction and decided together; a protected element makes its whole column solid.")
        self.ext_dir = QComboBox()
        for name, _ in AXES:
            self.ext_dir.addItem(name)
        self.ext_dir.setEnabled(False)
        self.ext_on.toggled.connect(self.ext_dir.setEnabled)
        row.addWidget(self.ext_on)
        row.addWidget(self.ext_dir)
        row.addStretch(1)
        v.addLayout(row)

        # --- overhang
        self.ov_on = QCheckBox("Overhang – printable without supports")
        help_tip(self.ov_on, "Surfaces not flatter than the angle from the build plate",
                 "Additive manufacturing without supports: a solid element needs support below it (a cone around the build direction); the "
                 "unsupported elements are removed and the volume is refilled. Approximate – element level on an unstructured mesh; the surface "
                 "of the result is measured afterwards. Use a filter radius of at least 3 element sizes.")
        v.addWidget(self.ov_on)
        form = QGridLayout()
        self.ov_angle = QDoubleSpinBox()
        self.ov_angle.setRange(10.0, 80.0)
        self.ov_angle.setValue(45.0)
        self.ov_angle.setSuffix(" °")
        help_tip(self.ov_angle, "Smallest angle between a surface and the build plate")
        self.ov_dir = QComboBox()
        for name, _ in BUILD_DIRS:
            self.ov_dir.addItem(name)
        self.has_print = self._printing_dir() is not None
        if self.has_print:
            self.ov_dir.addItem(FROM_PRINT)
        self.ov_dir.setToolTip("Build direction = the Z axis of the printer in the model coordinates")
        self.btn_ov_face = QPushButton("From the face on the plate")
        help_tip(self.btn_ov_face, "Build direction from a planar face lying on the plate",
                 "Select the planar face that lies on the printer bed in the 3D view before opening the dialog; the build direction is its inward normal.")
        self.btn_ov_face.clicked.connect(self._ov_from_face)
        self.ov_base = QComboBox()
        self.ov_base.addItem(LOWEST)
        for sname in study.selections:
            self.ov_base.addItem(sname)
        self.ov_base.setToolTip("Where the part stands on the plate: the lowest layer of the model or a selection")
        form.addWidget(QLabel("Angle from the plate:"), 0, 0)
        form.addWidget(self.ov_angle, 0, 1)
        form.addWidget(QLabel("Build direction:"), 1, 0)
        form.addWidget(self.ov_dir, 1, 1)
        form.addWidget(self.btn_ov_face, 1, 2)
        form.addWidget(QLabel("Base:"), 2, 0)
        form.addWidget(self.ov_base, 2, 1)
        v.addLayout(form)
        self._ov_widgets = (self.ov_angle, self.ov_dir, self.btn_ov_face, self.ov_base)
        self.ov_on.toggled.connect(self._ov_toggled)
        self.ov_dir.currentIndexChanged.connect(lambda *_: self.show_build_arrow())
        self._ov_toggled(False)

        self.compare = QCheckBox("Also run without the constraints (shows the compliance penalty, doubles the time)")
        help_tip(self.compare, "Reference run for the compliance penalty of the constraints")
        v.addWidget(self.compare)
        note = QLabel("The constraints act on the elements of the mesh – approximate. For the overhang use a filter radius of at least 3 element sizes.")
        note.setWordWrap(True)
        v.addWidget(note)

    # ------------------------------------------------------------------ state
    def set_running(self, running: bool) -> None:
        self.setEnabled(not running)

    def any_active(self) -> bool:
        return any(c.isChecked() for c in self.sym_on) or self.ext_on.isChecked() or self.ov_on.isChecked()

    def _printing_dir(self) -> list[float] | None:
        """Build direction from the print settings when the parts of the mesh share one (for the combo item)."""
        printing = getattr(self.study, "printing", None)
        dirs = [np.asarray(sp.build_dir, float) for sp in (printing.parts.values() if printing is not None else [])]
        if not dirs or any(np.linalg.norm(d) < 1e-12 for d in dirs):
            return None
        dirs = [d / np.linalg.norm(d) for d in dirs]
        return dirs[0].tolist() if all(np.linalg.norm(d - dirs[0]) < 1e-6 for d in dirs) else None

    def _ov_toggled(self, on: bool) -> None:
        for w in self._ov_widgets:
            w.setEnabled(on)
        self.show_build_arrow()

    def build_vector(self) -> list[float] | str:
        i = self.ov_dir.currentIndex()
        if i < len(BUILD_DIRS):
            return list(BUILD_DIRS[i][1])
        return "from_print"

    # ------------------------------------------------------------------ spec
    def spec_kwargs(self) -> dict:
        """Fields of TopologyOptSpec from the form."""
        sym = []
        for i, (_, axis) in enumerate(AXES):
            if self.sym_on[i].isChecked():
                p = self.centre.copy()
                p[i] = self.sym_pos[i].value()
                sym.append({"point": [float(x) for x in p], "normal": list(axis)})
        ext = {"direction": list(AXES[self.ext_dir.currentIndex()][1])} if self.ext_on.isChecked() else None
        base = "min" if self.ov_base.currentIndex() == 0 else self.ov_base.currentText()
        ov = ({"build_dir": self.build_vector(), "angle_deg": float(self.ov_angle.value()), "base": base}
              if self.ov_on.isChecked() else None)
        return {"symmetry": sym, "extrusion": ext, "overhang": ov, "compare_unconstrained": bool(self.compare.isChecked())}

    def describe(self, kw: dict) -> list[str]:
        """Lines for the log: what is switched on."""
        out = []
        if kw["symmetry"]:
            out.append("Symmetry: " + ", ".join(AXES[int(np.argmax(np.abs(p["normal"])))][0] + f" = {p['point'][int(np.argmax(np.abs(p['normal'])))]:.3f} mm"
                                                for p in kw["symmetry"]))
        if kw["extrusion"]:
            out.append("Extrusion along " + AXES[int(np.argmax(np.abs(kw["extrusion"]["direction"])))][0])
        if kw["overhang"]:
            ov = kw["overhang"]
            out.append(f"Overhang: {ov['angle_deg']:.0f} ° from the plate, build direction {self.ov_dir.currentText()}, base: {self.ov_base.currentText()}")
        return out

    # ------------------------------------------------------------------ faces from the 3D view
    def _selected_plane(self):
        """(point, normal) of the planar face selected in 3D (the last one), None + a message otherwise."""
        view = getattr(self.win, "view", None)
        faces = view.selected_faces() if view is not None and getattr(view, "mode", "") == "faces" else []
        if not faces or not hasattr(self.win, "face_prop"):
            QMessageBox.information(self, "Topology optimization", "First select a planar face in the 3D view (before opening this dialog).")
            return None
        props = self.win.face_prop(*faces[-1])
        n = np.asarray(props["normal"], float)
        if props["type"] != "plane" or np.linalg.norm(n) < 1e-12:
            QMessageBox.information(self, "Topology optimization", "The selected face is not planar.")
            return None
        return np.asarray(props["point"], float), n / np.linalg.norm(n)

    def _sym_from_face(self) -> None:
        got = self._selected_plane()
        if got is None:
            return
        p, n = got
        k = int(np.argmax(np.abs(n)))
        if abs(n[k]) < 0.9999:
            QMessageBox.information(self, "Topology optimization", "Only planes perpendicular to the X, Y or Z axis can be set here.")
            return
        self.sym_on[k].setChecked(True)
        self.sym_pos[k].setValue(float(p[k]))

    def _ov_from_face(self) -> None:
        got = self._selected_plane()
        if got is None:
            return
        d = -got[1]
        d = np.where(np.abs(d) < 1e-9, 0.0, np.round(d, 9))
        for i, (_, vec) in enumerate(BUILD_DIRS):
            if np.allclose(vec, d, atol=1e-6):
                self.ov_dir.setCurrentIndex(i)
                return
        QMessageBox.information(self, "Topology optimization", "Only the directions of the axes X, Y, Z can be set here; use the MCP tool for an oblique direction.")

    # ------------------------------------------------------------------ 3D arrow of the build direction
    def show_build_arrow(self) -> None:
        view = getattr(self.win, "view", None)
        if view is None or not hasattr(view, "add_arrows"):
            return
        view.remove_face_set(BUILD_KEY)
        if self.ov_on.isChecked():
            vec = self.build_vector()
            raw = self._printing_dir() if isinstance(vec, str) else vec
            d = np.asarray(raw, float) if raw is not None else None
            if d is not None and np.linalg.norm(d) > 1e-12:
                d = d / np.linalg.norm(d)
                corners = np.array([[x, y, z] for x in (self.lo[0], self.hi[0]) for y in (self.lo[1], self.hi[1]) for z in (self.lo[2], self.hi[2])])
                s = corners @ d
                diag = float(np.linalg.norm(self.hi - self.lo)) or 1.0
                foot = self.centre + (s.min() - self.centre @ d) * d
                length = max(1.3 * float(s.max() - s.min()), 0.35 * diag)
                tip = foot + length * d
                e1 = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(d, e1)
                r = 0.45 * diag
                q = [foot + r * (a * e1 + b * e2) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
                view.add_lines(BUILD_KEY, [(tip, foot)], C_BUILD, 3.0, label="build direction")
                view.add_arrows(BUILD_KEY, [(tip, d, 0.25 * length)], C_BUILD)
                view.add_lines(BUILD_KEY, [(q[i], q[(i + 1) % 4]) for i in range(4)], C_BUILD, 1.5)
        view._update()

    def clear_arrow(self) -> None:
        view = getattr(self.win, "view", None)
        if view is not None and hasattr(view, "remove_face_set"):
            view.remove_face_set(BUILD_KEY)
            view._update()

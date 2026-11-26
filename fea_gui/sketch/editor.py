"""The window of the sketch editor: tools, the drawing area, the parameter table, the entity list, the settings and the coordinate entry.

The editor changes a COPY of the sketch (`SketchEditor.sketch`); `accept()` is possible only when the sketch can be built (the check runs
the same code as the real build, so the user sees the regions / members and the first error with the entity that causes it).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from fea_core.sketch.expr import SketchError
from fea_core.sketch.model import MODES, SUPPORT_KINDS, Sketch, SketchRegion

from .. import theme
from ..widgets import help_tip
from .canvas import History, SketchCanvas
from .tools import TOOLS, CopyTool, PointTool

MODE_LABEL = {"plane_stress": "Plane stress", "plane_strain": "Plane strain", "axisym": "Axisymmetric"}
TOOL_LABEL = {"select": "Select", "line": "Line", "polyline": "Polyline", "rect": "Rectangle", "circle": "Circle", "arc3": "Arc (3 points)",
              "arc_center": "Arc (centre)", "point": "Node / support", "trim": "Trim", "move": "Move", "mirror": "Mirror", "copy": "Copy"}
TOOL_TIP = {"select": "Pick entities (Ctrl adds); Delete removes them", "line": "Line by two points, chained",
            "polyline": "Polyline through several points (click the first point to close)", "rect": "Rectangle by two opposite corners",
            "circle": "Circle: centre, then a point or the radius", "arc3": "Arc through three points",
            "arc_center": "Arc: centre, start, end (counter-clockwise)", "point": "Beam node or support at a point",
            "trim": "Cut a line, arc or circle back to the nearest crossings", "move": "Move the selected entities",
            "mirror": "Mirror copies of the selection about a line", "copy": "Linear copies of the selection"}
HOLE_CHOICES = {"Automatic": None, "Hole": True, "Own part": False}


def _cell(table: QTableWidget, row: int, col: int) -> str:
    it = table.item(row, col)
    return it.text() if it is not None else ""


class SketchEditor(QDialog):
    def __init__(self, parent, sketch: Sketch, title: str = "Sketch", materials: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("sketchEditor")
        self.resize(1180, 760)
        self.sketch = sketch
        self.materials = materials or []
        self.history = History(sketch)
        self.canvas = SketchCanvas(sketch, self.history)
        self.problem = ""
        self._busy = False
        self.result_info: dict = {}
        self._build_ui()
        self.canvas.sketchChanged.connect(self.refresh_panels)
        self.canvas.hintChanged.connect(self._hint)
        self.canvas.selectionChanged.connect(self._select_in_table)
        self.canvas.cursorMoved.connect(lambda x, y: self.cursor_label.setText(f"x = {x:.4g}   y = {y:.4g} mm"))
        self.select_tool("select")
        self.refresh_panels()

    # ------------------------------------------------------------------ layout
    def _build_ui(self) -> None:
        sk = self.sketch
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.toolbar = QToolBar()
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toolbar2 = QToolBar()                     # the second row: copies, undo, view, snapping
        self.toolbar2.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        grp = QActionGroup(self)
        grp.setExclusive(True)
        self.tool_actions: dict[str, QAction] = {}
        for name in TOOL_LABEL:
            if name == "point" and sk.kind != "frame":
                continue
            a = QAction(TOOL_LABEL[name], self)
            a.setCheckable(True)
            a.setToolTip(TOOL_TIP[name])
            a.triggered.connect(lambda _=False, n=name: self.select_tool(n))
            grp.addAction(a)
            self.toolbar.addAction(a)
            self.tool_actions[name] = a
            if name in ("select", "arc_center", "point"):
                self.toolbar.addSeparator()
        self.kind_box = QComboBox()
        self.kind_box.addItems(list(SUPPORT_KINDS))
        self.kind_box.setToolTip("Kind of the support put by the Node / support tool")
        self.kind_box.setVisible(sk.kind == "frame")
        self.toolbar2.addWidget(self.kind_box)
        self.copies = QSpinBox()
        self.copies.setRange(1, 200)
        self.copies.setPrefix("copies: ")
        self.copies.setToolTip("Number of copies of the Copy tool")
        self.toolbar2.addWidget(self.copies)
        self.toolbar2.addSeparator()
        self.act_undo = self.toolbar2.addAction("Undo", self.undo)
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_redo = self.toolbar2.addAction("Redo", self.redo)
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.toolbar2.addAction("Delete", self.canvas.delete_selected).setToolTip("Delete the selected entities")
        self.toolbar2.addAction("Fit", self.canvas.fit_all).setToolTip("Show the whole sketch  [F]")
        self.toolbar2.addSeparator()
        for text, tip, attr in (("Grid", "Snap to the grid", "snap_grid"), ("Object snap", "Snap to end points and mid points", "snap_objects"),
                                ("Ids", "Show the ids of the entities", "show_ids")):
            a = self.toolbar2.addAction(text)
            a.setCheckable(True)
            a.setChecked(True)
            a.setToolTip(tip)
            a.toggled.connect(lambda on, at=attr: (setattr(self.canvas, at, on), self.canvas.viewport().update()))
        lay.addWidget(self.toolbar)
        lay.addWidget(self.toolbar2)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.canvas)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(4, 0, 0, 0)
        sl.addWidget(self._settings_box())
        sl.addWidget(self._params_box(), 2)
        sl.addWidget(self._entities_box(), 3)
        self.regions_box = self._regions_box()
        sl.addWidget(self.regions_box, 2)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 360])
        lay.addWidget(split, 1)

        row = QHBoxLayout()
        self.hint_label = QLabel("")
        self.hint_label.setTextFormat(Qt.TextFormat.PlainText)         # messages may contain text from a foreign file
        self.hint_label.setWordWrap(True)
        row.addWidget(self.hint_label, 3)
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("x, y     @dx, dy     @r<angle     P3     (expressions of the parameters allowed)")
        help_tip(self.entry, "Typed point: x, y | @dx, dy | @r<angle | point id (expressions allowed)", "Typed coordinates in mm: x, y (absolute)  |  @dx, dy (relative to the last point)  |  @r<angle (polar, degrees)  |  a point id")
        self.entry.returnPressed.connect(self._entered)
        row.addWidget(self.entry, 3)
        self.cursor_label = QLabel("")
        self.cursor_label.setMinimumWidth(190)
        row.addWidget(self.cursor_label)
        lay.addLayout(row)

        btns = QHBoxLayout()
        self.btn_dxf = QPushButton("Import DXF…")
        self.btn_dxf.setToolTip("Adds the lines, arcs, circles and polylines of a DXF drawing (module sketch)")
        self.btn_dxf.clicked.connect(self.import_dxf_dialog)
        btns.addWidget(self.btn_dxf)
        self.btn_check = QPushButton("Check")
        self.btn_check.setToolTip("Finds the regions / members and the first problem of the drawing")
        self.btn_check.clicked.connect(self.check)
        btns.addWidget(self.btn_check)
        btns.addStretch(1)
        self.btn_apply = QPushButton("Apply sketch")
        theme.set_primary(self.btn_apply)
        self.btn_apply.setToolTip("Builds the sketch into the model (constraints, loads and materials stay)")
        self.btn_apply.clicked.connect(self.accept)
        btns.addWidget(self.btn_apply)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

    def _settings_box(self) -> QGroupBox:
        sk = self.sketch
        box = QGroupBox("Sketch")
        f = QVBoxLayout(box)
        if sk.kind == "twod":
            row = QHBoxLayout()
            row.addWidget(QLabel("Problem"))
            self.mode_box = QComboBox()
            for m in MODES:
                self.mode_box.addItem(MODE_LABEL[m], m)
            self.mode_box.setCurrentIndex(MODES.index(sk.mode))
            self.mode_box.setToolTip("Plane stress, plane strain, or axisymmetric (x = radius, y = axis)")
            self.mode_box.currentIndexChanged.connect(self._mode_changed)
            row.addWidget(self.mode_box, 1)
            f.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel("Thickness [mm]"))
            self.thickness = QLineEdit(str(sk.thickness))
            self.thickness.setToolTip("Thickness of a planar problem: a number or an expression of the parameters")
            self.thickness.editingFinished.connect(self._thickness_changed)
            row.addWidget(self.thickness, 1)
            f.addLayout(row)
            self.thickness.setEnabled(sk.mode != "axisym")
        else:
            row = QHBoxLayout()
            row.addWidget(QLabel("Default section"))
            self.section = QLineEdit(str(sk.frame.get("section", "IPE 200")))
            help_tip(self.section, "Section of members without their own (library name, e.g. IPE 200)", "Cross-section of the members that have none of their own (library name, e.g. IPE 200)")
            self.section.editingFinished.connect(self._section_changed)
            row.addWidget(self.section, 1)
            f.addLayout(row)
            self.chk_plane = QCheckBox("Hold the nodes against motion out of the plane")
            self.chk_plane.setChecked(bool(sk.frame.get("plane_restraint", True)))
            help_tip(self.chk_plane, "Fix out-of-plane motion at every node (plane frame)", "A plane frame loaded in its plane: out-of-plane displacements and rotations are fixed at every node")
            self.chk_plane.toggled.connect(lambda on: self._frame_flag("plane_restraint", on))
            f.addWidget(self.chk_plane)
            self.chk_split = QCheckBox("Split members where they cross")
            self.chk_split.setChecked(bool(sk.frame.get("split", False)))
            self.chk_split.setToolTip("Crossing lines become connected at a node (otherwise they pass each other)")
            self.chk_split.toggled.connect(lambda on: self._frame_flag("split", on))
            f.addWidget(self.chk_split)
        return box

    def _params_box(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        v = QVBoxLayout(box)
        self.params = QTableWidget(0, 3)
        self.params.setHorizontalHeaderLabels(["Name", "Expression", "Value [mm]"])
        self.params.horizontalHeader().setStretchLastSection(True)
        self.params.verticalHeader().setVisible(False)
        self.params.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.params.itemChanged.connect(self._param_edited)
        self.params.itemSelectionChanged.connect(self._param_selected)
        help_tip(self.params, "Named dimensions for coordinates, radii and thickness (L = 200, h = L/10)", "Named dimensions: coordinates, radii and the thickness may use them (L = 200, h = L/10)")
        v.addWidget(self.params)
        row = QHBoxLayout()
        self.p_name = QLineEdit()
        self.p_name.setPlaceholderText("name")
        self.p_expr = QLineEdit()
        self.p_expr.setPlaceholderText("value or expression")
        self.p_expr.returnPressed.connect(self.set_param_from_fields)
        row.addWidget(self.p_name, 1)
        row.addWidget(self.p_expr, 2)
        b = QPushButton("Set")
        b.setToolTip("Adds the parameter or changes its value")
        b.clicked.connect(self.set_param_from_fields)
        row.addWidget(b)
        b = QPushButton("Delete")
        b.setToolTip("Deletes the selected parameter (not while something uses it)")
        b.clicked.connect(self.delete_param)
        row.addWidget(b)
        v.addLayout(row)
        return box

    def _entities_box(self) -> QGroupBox:
        box = QGroupBox("Entities")
        v = QVBoxLayout(box)
        self.entities = QTableWidget(0, 4)
        self.entities.setHorizontalHeaderLabels(["Id", "Type", "Name", "Details"])
        self.entities.horizontalHeader().setStretchLastSection(True)
        self.entities.verticalHeader().setVisible(False)
        self.entities.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.entities.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.entities.itemChanged.connect(self._entity_edited)
        self.entities.itemSelectionChanged.connect(self._entity_selected)
        help_tip(self.entities, "The name of an entity names the selection of its edges", "The name of an entity is the name of the selection of its edges (for supports and loads)")
        v.addWidget(self.entities)
        row = QHBoxLayout()
        self.hole_box = QComboBox()
        self.hole_box.addItems(list(HOLE_CHOICES))
        help_tip(self.hole_box, "Closed entity inside a region: automatic, hole, or a part of its own", "A closed entity inside a region: automatic (a hole), always a hole, or a part of its own (an insert)")
        self.hole_box.activated.connect(self._hole_changed)
        self.hole_box.setVisible(self.sketch.kind == "twod")
        row.addWidget(self.hole_box)
        self.chk_constr = QCheckBox("Construction")
        self.chk_constr.setToolTip("Drawn only as a help: not part of the geometry")
        self.chk_constr.clicked.connect(self._construction_changed)
        row.addWidget(self.chk_constr)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _regions_box(self) -> QGroupBox:
        box = QGroupBox("Regions" if self.sketch.kind == "twod" else "Members")
        v = QVBoxLayout(box)
        self.regions = QTableWidget(0, 3)
        self.regions.setHorizontalHeaderLabels(["Name", "Area [mm²]", "Boundary"] if self.sketch.kind == "twod" else ["Member", "Nodes", "Length [mm]"])
        self.regions.horizontalHeader().setStretchLastSection(True)
        self.regions.verticalHeader().setVisible(False)
        self.regions.itemChanged.connect(self._region_edited)
        self.regions.setToolTip("Result of the check; the name of a region is the name of the part (editable)")
        v.addWidget(self.regions)
        return box

    # ------------------------------------------------------------------ tools and entry
    def select_tool(self, name: str) -> None:
        cls = TOOLS[name]
        tool = cls(self.canvas)
        if isinstance(tool, PointTool):
            tool.kind = self.kind_box.currentText()
        if isinstance(tool, CopyTool):
            tool.count = self.copies.value()
        self.canvas.set_tool(tool)
        if name in self.tool_actions:
            self.tool_actions[name].setChecked(True)
        self.canvas.setFocus()

    def type_text(self, text: str) -> None:
        """What the user types into the status bar (also the entry point of the GUI tests)."""
        tool = self.canvas.tool
        if tool is None:
            return
        if isinstance(tool, PointTool):
            tool.kind = self.kind_box.currentText()
        if isinstance(tool, CopyTool):
            tool.count = self.copies.value()
        tool.entry(text)
        self.canvas.viewport().update()

    def _entered(self) -> None:
        text = self.entry.text()
        self.entry.clear()
        if text.strip():
            self.type_text(text)

    def _hint(self, text: str) -> None:
        self.hint_label.setText(text)

    # ------------------------------------------------------------------ undo / redo
    def undo(self) -> None:
        if self.history.undo():
            self.canvas.regions = []
            self.canvas.refresh()
            self.refresh_panels()

    def redo(self) -> None:
        if self.history.redo():
            self.canvas.regions = []
            self.canvas.refresh()
            self.refresh_panels()

    # ------------------------------------------------------------------ panels
    def refresh_panels(self) -> None:
        sk = self.sketch
        self._busy = True
        try:
            vals = {}
            try:
                vals = sk.values()
            except SketchError as e:
                self._hint(str(e))
            self.params.setRowCount(len(sk.params))
            for i, (k, v) in enumerate(sk.params.items()):
                a = QTableWidgetItem(k)
                a.setFlags(a.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.params.setItem(i, 0, a)
                self.params.setItem(i, 1, QTableWidgetItem(str(v)))
                b = QTableWidgetItem(f"{vals[k]:.6g}" if k in vals else "?")
                b.setFlags(b.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.params.setItem(i, 2, b)
            self.entities.setRowCount(len(sk.entities))
            for i, e in enumerate(sk.entities):
                cells = [e.id, e.type + (" (construction)" if e.construction else ""), e.name, self._details(e)]
                for j, t in enumerate(cells):
                    it = QTableWidgetItem(t)
                    if j != 2:
                        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.entities.setItem(i, j, it)
            if sk.kind == "twod":
                self.thickness.setEnabled(sk.mode != "axisym")
        finally:
            self._busy = False
        self.act_undo.setEnabled(bool(self.history.undo_stack))
        self.act_redo.setEnabled(bool(self.history.redo_stack))
        self.canvas.viewport().update()

    def _details(self, e) -> str:
        pts = ", ".join(e.points)
        if e.center:
            pts = f"centre {e.center}" + (f", {pts}" if pts else "")
        if e.radius is not None:
            pts += f", R = {e.radius}"
        return pts

    def _param_selected(self) -> None:
        r = self.params.currentRow()
        if r >= 0 and self.params.item(r, 0) is not None:
            self.p_name.setText(_cell(self.params, r, 0))
            self.p_expr.setText(_cell(self.params, r, 1))

    def _param_edited(self, item: QTableWidgetItem) -> None:
        if self._busy or item.column() != 1:
            return
        name_item = self.params.item(item.row(), 0)
        if name_item is None:
            return
        name, text = name_item.text(), item.text().strip()
        self.canvas.commit(lambda sk: sk.set_param(name, text))
        self.refresh_panels()

    def set_param_from_fields(self) -> None:
        name, expr = self.p_name.text().strip(), self.p_expr.text().strip()
        if not name or not expr:
            self._hint("Type the name and the value (a number or an expression) of the parameter")
            return
        if self.canvas.commit(lambda sk: sk.set_param(name, expr), f"{name} = {expr}"):
            self.p_name.clear()
            self.p_expr.clear()
        self.refresh_panels()

    def delete_param(self) -> None:
        name = self.p_name.text().strip()
        if name:
            self.canvas.commit(lambda sk: sk.remove_param(name), f"Parameter {name} deleted")
            self.refresh_panels()

    def _entity_edited(self, item: QTableWidgetItem) -> None:
        if self._busy or item.column() != 2:
            return
        eid = _cell(self.entities, item.row(), 0)
        text = item.text()
        self.canvas.commit(lambda sk: sk.set_name(eid, text))
        self.refresh_panels()

    def _entity_selected(self) -> None:
        if self._busy:
            return
        ids = sorted({_cell(self.entities, i.row(), 0) for i in self.entities.selectedIndexes()})
        self.canvas.selected = set(ids)
        self.canvas.viewport().update()
        if len(ids) == 1:
            e = self.sketch.entity(ids[0])
            self.chk_constr.setChecked(e.construction)
            self.hole_box.setCurrentIndex(list(HOLE_CHOICES.values()).index(e.hole))

    def _select_in_table(self, ids: list) -> None:
        self._busy = True
        try:
            self.entities.clearSelection()
            for r in range(self.entities.rowCount()):
                if _cell(self.entities, r, 0) in ids:
                    self.entities.selectRow(r)
        finally:
            self._busy = False
        if len(ids) == 1:
            e = self.sketch.entity(ids[0])
            self.chk_constr.setChecked(e.construction)
            self.hole_box.setCurrentIndex(list(HOLE_CHOICES.values()).index(e.hole))

    def _hole_changed(self, _i: int) -> None:
        val = list(HOLE_CHOICES.values())[self.hole_box.currentIndex()]
        ids = sorted(self.canvas.selected)

        def run(sk):
            for eid in ids:
                sk.entity(eid).hole = val
        if ids:
            self.canvas.commit(run)

    def _construction_changed(self, on: bool) -> None:
        ids = sorted(self.canvas.selected)

        def run(sk):
            for eid in ids:
                sk.entity(eid).construction = bool(on)
        if ids:
            self.canvas.commit(run)
            self.refresh_panels()

    def _mode_changed(self, _i: int) -> None:
        mode = self.mode_box.currentData()

        def run(sk):
            sk.mode = mode
        self.canvas.commit(run)
        self.refresh_panels()

    def _thickness_changed(self) -> None:
        text = self.thickness.text().strip()
        if text == str(self.sketch.thickness):
            return

        def run(sk):
            old = sk.thickness
            sk.thickness = float(text) if text.replace(".", "", 1).isdigit() else text
            try:
                sk.thickness_value()
            except SketchError:
                sk.thickness = old
                raise
        self.canvas.commit(run)

    def _section_changed(self) -> None:
        text = self.section.text().strip()
        if text:
            self.canvas.commit(lambda sk: sk.frame.__setitem__("section", text))

    def _frame_flag(self, key: str, on: bool) -> None:
        self.canvas.commit(lambda sk: sk.frame.__setitem__(key, bool(on)))

    def _region_edited(self, item: QTableWidgetItem) -> None:
        if self._busy or item.column() != 0 or self.sketch.kind != "twod":
            return
        row = item.row()
        reg = self.canvas.regions[row] if row < len(self.canvas.regions) else None
        if reg is None:
            return
        name, pt = item.text().strip(), [round(reg["point"][0], 6), round(reg["point"][1], 6)]

        def run(sk):
            sk.regions = [r for r in sk.regions if r.name != name]
            sk.regions.append(SketchRegion(pt, name, None))
        regions_keep = list(self.canvas.regions)
        if name and self.canvas.commit(run, f"Region named {name}"):
            self.canvas.regions = regions_keep
            self.check()

    # ------------------------------------------------------------------ check / DXF / apply
    def check(self) -> bool:
        """Builds the sketch in memory: regions (2D) or members (1D) shown in the lists, or the first problem with its entity."""
        sk = self.sketch
        self.canvas.problems = set()
        self.problem = ""
        try:
            if sk.kind == "twod":
                from fea_core.sketch.build import construct
                cons = construct(sk)
                self.canvas.regions = [{"rings": r.rings, "name": r.name, "point": r.point} for r in cons.regions]
                self._busy = True
                self.regions.setRowCount(len(cons.regions))
                for i, r in enumerate(cons.regions):
                    self.regions.setItem(i, 0, QTableWidgetItem(r.name))
                    a = QTableWidgetItem(f"{r.area:.6g}")
                    a.setFlags(a.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.regions.setItem(i, 1, a)
                    b = QTableWidgetItem(", ".join(r.entities[:8]))
                    b.setFlags(b.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.regions.setItem(i, 2, b)
                self._busy = False
                self.result_info = {"regions": [{"name": r.name, "area": r.area} for r in cons.regions], "warnings": list(cons.warnings)}
                self._hint(f"OK: {len(cons.regions)} region(s), total area {sum(r.area for r in cons.regions):.6g} mm²")
            else:
                from fea_core.sketch.frame import build_frame
                fb = build_frame(sk)
                self._busy = True
                self.regions.setRowCount(len(fb.members))
                for i, (n, m) in enumerate(fb.members.items()):
                    self.regions.setItem(i, 0, QTableWidgetItem(n))
                    self.regions.setItem(i, 1, QTableWidgetItem(f"{m['start']} – {m['end']}"))
                    self.regions.setItem(i, 2, QTableWidgetItem(""))
                    for c in range(3):
                        it = self.regions.item(i, c)
                        if it is not None:
                            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._busy = False
                self.result_info = {"nodes": len(fb.nodes), "members": len(fb.members), "warnings": list(fb.warnings)}
                self._hint(f"OK: {len(fb.nodes)} node(s), {len(fb.members)} member(s)" + (" – " + "; ".join(fb.warnings) if fb.warnings else ""))
        except SketchError as e:
            self._busy = False
            self.canvas.regions = []
            self.canvas.problems = set(e.entities)
            self.problem = str(e)
            self._hint(str(e))
            self.canvas.viewport().update()
            return False
        self.canvas.viewport().update()
        return True

    def import_dxf_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Import DXF", "", "DXF drawing (*.dxf)")
        if p:
            self.import_dxf_file(p)

    def import_dxf_file(self, path: str) -> dict | None:
        from fea_core.sketch.dxf import import_dxf
        info = self.canvas.commit(lambda sk: import_dxf(sk, path))
        if isinstance(info, dict):
            self._hint(f"DXF: {info['imported']} curves imported ({info['units']}, scale {info['scale_to_mm']:g})"
                       + (" – " + "; ".join(info["warnings"]) if info["warnings"] else ""))
            self.canvas.fit_all()
            self.refresh_panels()
            return info
        return None

    def accept(self) -> None:
        if not self.check():
            return                                     # the problem is shown; the window stays
        super().accept()

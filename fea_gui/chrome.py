"""Window chrome of the main window: header (menus, phase tabs, document name, phase actions, Run), step tabs,
bar above the 3D window (selection filter, colour, section, view), the tool shelf on the left (SHELF dispatch), compaction of the
bars in a narrow / low window and the appearance switch. Mixed into `main.MainWindow` (attribute names unchanged – the guide,
tours and tests use them).

Imported by `fea_gui.main` for `MainWindow`; the constants below come from `fea_gui.main_defs`.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QSlider, QToolBar, QWidget)
from fea_core.i18n import tr

from . import i18n, theme
from .widgets import help_tip
from .main_defs import (ANALYSIS_LABELS, ElidedLabel, FRAME_ONLY_OFF, GREEN, GREY, HEADER_ACTIONS, MATERIAL_COLORS,
                        NO_MATERIAL_COLOR, OPTIONAL_STAGES, ORANGE, PHASES, PHASE_OF, PHASE_TIPS, PICK_FILTER_ACTIONS,
                        PICK_FILTER_SHORT, RED, SHELF, SHELF_LABELS, STAGE, STAGES, STAGE_COLORS, STAGE_LONG, STAGE_TIPS,
                        VIEW_DIRECTIONS, _remember_ui, fmt, phase_label)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class ChromeMixin(_Base):
    """Header, step tabs, view bar, shelf and their compaction (part of MainWindow)."""

    def _build_toolbars(self):
        """Header in one line (Studio × Tables design): ▲ FEA Studio, menus File / Edit / View /
        Tools / Help, phase tabs (tb_flow), document name on the right, Replace geometry and Run calculation.
        """
        from PySide6.QtWidgets import QMenu, QToolButton
        tb = QToolBar("Menus")
        tb.setObjectName("tb_file")
        tb.setMovable(False)
        logo = QLabel("▲")
        logo.setObjectName("brandLogo")
        tb.addWidget(logo)
        brand = QLabel("FEA Studio")
        brand.setObjectName("brand")
        brand.setToolTip("FEA Studio – finite element strength analysis (CalculiX, Gmsh)")
        self._brand_act = tb.addWidget(brand)
        a_open = self._act("Open…", self.open_dialog, "Ctrl+O", "Open STEP or .fea project", "open")
        a_save = self._act("Save", self.save, "Ctrl+S", "Save .fea project", "save")
        a_save_as = self._act("Save as…", self.save_as, "Ctrl+Shift+S", icon="save_as")
        a_report = self._act("Report…", self.export_report, "Ctrl+P", "Technical analysis report to Word (.docx)",
                             "report")
        a_undo = self._act("Undo", self.undo, "Ctrl+Z", "Undo the last study change", "undo")
        a_redo = self._act("Redo", self.redo, "Ctrl+Y", "Redo the undone change", "redo")
        self.menu_buttons: dict[str, QToolButton] = {}

        def menu(title, tip):
            b = QToolButton()
            b.setObjectName("menuButton")
            b.setText(title)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setToolTip(tip)
            m = QMenu(b)
            b.setMenu(m)
            tb.addWidget(b)
            self.menu_buttons[title] = b
            return m
        m = menu("File", "Open, save, report, replace geometry, new model from a sketch")
        self.menu_file = self.menu_buttons["File"]        # target of the guide (Save, Report)
        for a in (a_open, a_save, a_save_as):
            m.addAction(a)
        m.addSeparator()
        m.addAction(a_report)
        m.addSeparator()
        m.addAction("Replace geometry (keep setup)…", self.replace_geometry_dialog)
        m.addSeparator()
        m.addAction("New 2D model from sketch…", lambda: self._sketch_action("twod"))
        m.addAction("New beam system from sketch…", lambda: self._sketch_action("frame"))
        m = menu("Edit", "Undo, redo")
        m.addAction(a_undo)
        m.addAction(a_redo)
        m = self.view_menu = menu("Show", "View, section, workflow steps, appearance, language")
        m.addAction(self._act("Fit (whole model)", lambda: self.view.fit_all(), "F", "Fit all", "fit"))
        sub = m.addMenu("View direction")
        for key, txt in VIEW_DIRECTIONS:
            sub.addAction(theme.icon(key), txt, lambda k=key: self.view.set_orientation(k))
        m.addSeparator()
        steps = m.addMenu("Optional workflow steps")
        steps.aboutToShow.connect(lambda: self._fill_more_menu(steps))
        m.addSeparator()
        self.act_panel_window = a = m.addAction("Step panel in a separate window")
        a.setCheckable(True)
        a.setToolTip("Moves the step panel into its own window; closing the window returns it")
        a.toggled.connect(lambda on: self.set_panel_detached(on))
        self.act_tables_window = a = m.addAction("Tables in a separate window")
        a.setCheckable(True)
        a.setToolTip("Bottom Tables dock in its own window; closing or dragging back returns it")
        a.toggled.connect(lambda on: self.set_tables_floating(on))
        m.addSeparator()
        # the Appearance / Language / Tools menus belong to buttons without a parent – _menu_sources keeps them (otherwise Python would delete them)
        self._menu_sources = [self._theme_button(), i18n.language_button(self), self._tools_button()]
        th, lang, tools = (b.menu() for b in self._menu_sources)
        th.setTitle("Appearance")
        th.setIcon(theme.icon("theme"))
        m.addMenu(th)
        lang.setTitle("Language / Jazyk")
        lang.setIcon(self._menu_sources[1].icon())
        m.addMenu(lang)
        mt = menu("Tools", "Calculators, new beam structure, mesh convergence, modules")
        for a in tools.actions():
            mt.addAction(a)
        self.help_menu = menu("Help", "Help, What's this?, tours, manual")
        self.addToolBar(tb)
        self.tb_file = tb
        self._build_section_controls()
        self._build_flow_bar()
        rt = QToolBar("Document")
        rt.setObjectName("tb_right")
        rt.setMovable(False)
        rt.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        spacer = QWidget()                               # the last bar in the row gets the remaining space → everything to the right
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rt.addWidget(spacer)
        self.doc_label = QLabel("")
        self.doc_label.setObjectName("docLabel")
        self.doc_label.setTextFormat(Qt.PlainText)         # name of a study / file from a foreign project
        self._doc_act = rt.addWidget(self.doc_label)
        b = QPushButton("Replace geometry…")             # phase actions (HEADER_ACTIONS, _update_header_actions)
        b.clicked.connect(lambda: self._header_action(0))
        self.btn_replace_geo = self.btn_phase = b
        self._replace_act = rt.addWidget(b)
        b = QPushButton("")
        b.clicked.connect(lambda: self._header_action(1))
        self.btn_phase2 = b
        self._phase2_act = rt.addWidget(b)
        b = QPushButton("Run analysis")
        theme.set_icon(b, "run")
        theme.set_primary(b)
        help_tip(b, "Go to the Solve step and run the analysis  [F5]",
                 "Switches to the Solve step and runs the analysis (the mesh is created if missing). Same as the Run button in the Solve step.")
        b.clicked.connect(self.run_from_header)
        self.btn_run_header = b
        self._run_act = rt.addWidget(b)
        self._act("Run analysis", self.run_from_header, "F5", "Run analysis")
        self.addToolBar(rt)
        self.tb_right = rt
        for t in (self.tb_file, self.tb_flow, rt):     # header 32 px as in the design (the style adds a 6 px border)
            t.layout().setContentsMargins(4, 0, 4, 0)
        self._build_shelf()
        self._build_view_bar()

    def _update_header_actions(self, phase: str):
        """Header buttons by phase (design): Model Replace geometry, Assembly Recognise joints, Calculation mesh
        convergence, Results Export and Create report (instead of Run calculation).
        """
        acts = HEADER_ACTIONS.get(phase, [])
        self._header_acts = acts
        for i, (b, wa) in enumerate(((self.btn_phase, self._replace_act), (self.btn_phase2, self._phase2_act))):
            if i < len(acts):
                text, _slot, short, long = acts[i]
                b.setText(text)
                help_tip(b, short, long)
            wa.setVisible(i < len(acts) and self._header_lvl < 2)
        self._run_act.setVisible(phase != "Evaluation")
        theme.set_primary(self.btn_phase2, phase == "Evaluation")

    def _header_action(self, i: int):
        acts = getattr(self, "_header_acts", [])
        if i < len(acts):
            self._shelf_run(acts[i][1])

    def run_from_header(self):
        """Run calculation from the header: Solve step + its button (same readiness check and availability)."""
        if self.busy() or self.study is None:
            return
        self.set_stage(STAGE["Solve"])
        if self.btn_run.isEnabled():
            self.btn_run.click()

    def _build_section_controls(self):
        """Section through the model: axis menu, position slider and "flip" in the panel that the Section button above the 3D window expands."""
        from PySide6.QtWidgets import QMenu, QToolButton, QWidgetAction
        b = QToolButton()
        b.setObjectName("sec_button")
        b.setText("Section")
        theme.set_icon(b, "section")
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setPopupMode(QToolButton.InstantPopup)
        b.setCheckable(True)
        help_tip(b, "Section the model by a plane normal to an axis or by the plane of a selected face",
                 "Section by a plane normal to an axis or by the plane of a selected face (select a face, then “face”).<br>The slider moves the plane within the model extent, “flip” shows the other part. In the results the section through the mesh interior is shown in colour too.")
        menu = QMenu(b)
        box = QWidget()
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.addWidget(QLabel("Plane"), 0, 0)
        self.sec_axis = QComboBox()
        self.sec_axis.addItems(["off", "X", "Y", "Z", "face"])
        help_tip(self.sec_axis, "Section by a plane normal to an axis, or by the plane of the selected face",
                 "Section by a plane normal to an axis or by the plane of the selected face (select a face, then “face”).<br>In the results the section through the inside of the mesh is also shown in color.")
        self.sec_axis.currentIndexChanged.connect(self.on_sec_axis)
        self.sec_axis.currentIndexChanged.connect(self._sync_sec_button)
        lay.addWidget(self.sec_axis, 0, 1)
        lay.addWidget(QLabel("Position"), 1, 0)
        self.sec_pos = QSlider(Qt.Horizontal)
        self.sec_pos.setRange(0, 1000)
        self.sec_pos.setValue(500)
        self.sec_pos.setFixedWidth(180)
        self.sec_pos.setToolTip("Section plane position within the model extent (for a face-based section, offset along its normal)")
        self.sec_pos.valueChanged.connect(self.update_section)
        lay.addWidget(self.sec_pos, 1, 1)
        self._sec_face = None              # (point, normal) of the section plane from a face
        self.sec_flip = QCheckBox("flip")
        self.sec_flip.setToolTip("Show the other part of the model (flip the section side)")
        self.sec_flip.toggled.connect(self.update_section)
        lay.addWidget(self.sec_flip, 2, 1)
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(box)
        menu.addAction(wa)
        b.setMenu(menu)
        self.sec_button = b

    def _sync_sec_button(self, k):
        """The Section button shows that the section is on (checked, "Section Y")."""
        on = k > 0
        self.sec_button.setChecked(on)
        self.sec_button.setText(f"Section {self.sec_axis.currentText()}" if on and k != self.SEC_FACE else
                                ("Face section" if on else "Section"))

    # ------------------------------------------------------------------ tool shelf on the left (Studio layout)
    def _build_shelf(self):
        """Vertical tool shelf: commands of the current step in groups (SHELF, changes in set_stage).
        A shelf command = click on the button of the step panel (same behaviour and availability) or a window method.
        """
        sh = QToolBar("Tools")
        sh.setObjectName("tb_shelf")
        sh.setMovable(False)
        sh.setFloatable(False)
        sh.setOrientation(Qt.Vertical)
        sh.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        sh.setIconSize(QSize(18, 18))

        def label(text):
            lb = QLabel(text)
            lb.setAlignment(Qt.AlignCenter)
            return sh.addWidget(lb)
        # step commands: one set of actions per step, visible only in its own step (_update_shelf)
        self._shelf_label = QLabel("")
        self._shelf_label.setAlignment(Qt.AlignCenter)
        self._shelf_label_act = sh.addWidget(self._shelf_label)
        self._shelf_acts: dict[str, list[QAction]] = {}
        self._shelf_heads: dict[str, list[QAction]] = {}       # group headings (labels) per step
        for stage, cmds in SHELF.items():
            acts, heads = [], []
            for item in cmds:
                if len(item) == 2:                             # ("#", "NÁZEV") = group heading
                    if acts or heads:
                        sep = sh.addSeparator()
                        sep.setVisible(False)
                        heads.append(sep)
                    h = label(item[1])
                    h.setVisible(False)
                    heads.append(h)
                    continue
                txt, ic, target, tip = item
                a = QAction(txt, self)
                theme.set_icon(a, ic)
                a.setToolTip(tip)
                a.triggered.connect(lambda _=False, t=target: self._shelf_run(t))
                a.setVisible(False)
                sh.addAction(a)
                acts.append(a)
            self._shelf_acts[stage] = acts
            self._shelf_heads[stage] = heads
        self._shelf_lvl = 0
        self.addToolBar(Qt.LeftToolBarArea, sh)
        self.tb_shelf = sh

    # ------------------------------------------------------------------ bar above the 3D window
    def _build_view_bar(self):
        """Bar above the 3D window (design): on the left the help for the selection, on the right Select ▾ (selection filter P/B/H/E/V),
        View ▾, Section, Fit. (An overlay directly in the 3D window is not possible – the OCCT window is native.)
        """
        from PySide6.QtWidgets import QMenu, QToolButton
        vb = QFrame()                                    # not a QToolBar: the help on the left is shortened in a narrow window
        vb.setObjectName("tb_view")
        row = QHBoxLayout(vb)
        row.setContentsMargins(6, 3, 6, 3)
        row.setSpacing(3)
        self.view_title = QLabel("")                     # view description per step (update_view_info)
        self.view_title.setObjectName("viewTitle")
        self.view_title.setTextFormat(Qt.PlainText)
        row.addWidget(self.view_title)
        self.view_legend = QLabel("")                    # legend of material colours (colour by material)
        self.view_legend.setObjectName("viewLegend")
        self.view_legend.setTextFormat(Qt.RichText)
        row.addWidget(self.view_legend)
        self.view_hint = ElidedLabel("")
        self.view_hint.setObjectName("viewHint")
        row.addWidget(self.view_hint, 1)

        def vb_add(w):
            w.setIconSize(QSize(16, 16))
            row.addWidget(w)
        pb = QToolButton()
        pb.setObjectName("pick_button")
        pb.setPopupMode(QToolButton.InstantPopup)
        pb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        help_tip(pb, "What a click in 3D selects: face, body, hole, edge, point (P / B / H / E / V)",
                 "3D selection filter: <b>Face</b> (P), <b>Body</b> (B), <b>Hole</b> (H – all coaxial cylindrical faces), <b>Edge</b> (E – length, radius and center), <b>Point</b> (V – vertex or point on a face).")
        pm = QMenu(pb)
        fg = QActionGroup(self)
        self.filter_actions: dict[str, QAction] = {}
        for key, txt, sc, tip in PICK_FILTER_ACTIONS:
            a = self._act(txt, lambda _=False, k=key: self.set_pick_filter(k), sc, PICK_FILTER_SHORT[key], key)
            a.setWhatsThis(tr(tip).replace(".\n", ".<br>").replace("\n", " "))     # tr before editing the rows (catalogue)
            a.setCheckable(True)
            fg.addAction(a)
            pm.addAction(a)
            self.filter_actions[key] = a
        self.filter_actions["face"].setChecked(True)
        pb.setMenu(pm)
        self.pick_button = pb
        vb_add(pb)
        cb = QToolButton()                               # part colour: by part / by material
        cb.setObjectName("color_button")
        cb.setPopupMode(QToolButton.InstantPopup)
        cb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        cb.setToolTip("Part color in 3D: each part a different color, or by material (with legend)")
        cm = QMenu(cb)
        cg = QActionGroup(self)
        self.color_actions = {}
        for key, txt in (("part", "part"), ("material", "material")):
            a = cm.addAction(f"Color: {txt}", lambda k=key: self.set_color_mode(k))
            a.setCheckable(True)
            cg.addAction(a)
            self.color_actions[key] = a
        cb.setMenu(cm)
        self.color_button = cb
        vb_add(cb)
        qb = QToolButton()                               # results: quantity (menu from r_qty)
        qb.setObjectName("qty_button")
        qb.setPopupMode(QToolButton.InstantPopup)
        qb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        qb.setToolTip("Displayed result quantity")
        qm = QMenu(qb)
        qm.aboutToShow.connect(lambda: self._fill_qty_menu(qm))
        qb.setMenu(qm)
        self.qty_button = qb
        vb_add(qb)
        def menu_button(name, tip, items, slot):
            b = QToolButton()
            b.setObjectName(name)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setToolTip(tip)
            mm = QMenu(b)
            grp = QActionGroup(self)
            acts = {}
            for key, txt in items:
                a = mm.addAction(txt, lambda k=key, bb=b: slot(k, bb))
                a.setCheckable(True)
                grp.addAction(a)
                acts[key] = a
            b.setMenu(mm)
            b._acts = acts
            first = items[0][0]
            acts[first].setChecked(True)
            b.setText(acts[first].text())
            vb_add(b)
            return b
        self.transp_button = menu_button(
            "transp_button", "Parts solid or transparent (bolts and holes inside the assembly are visible)",
            (("solid", "Solid"), ("transparent", "Transparent")), self._on_transparency)
        self.style_button = menu_button(
            "style_button", "Part style: shaded with face edges, shaded, wireframe",
            (("shaded_edges", "Shaded with edges"), ("shaded", "Shaded"), ("wire", "Wireframe")),
            self._on_display_style)
        self.mesh_show_button = menu_button(
            "mesh_show_button", "What to show in 3D in the Mesh and Solve steps: the finite element mesh or the parts",
            (("mesh", "Show: Mesh"), ("parts", "Show: Parts")), self._on_mesh_show_mode)
        self.mesh_color_button = menu_button(
            "mesh_color_button", "Mesh color: uniform, or by element quality (1 = regular tetrahedron)",
            (("plain", "Color: uniform"), ("quality", "Color: element quality")), self._on_mesh_color)
        self.view_edges = QToolButton()                  # results: mesh edges (r_edges)
        self.view_edges.setObjectName("edges_button")
        self.view_edges.setText("Mesh edges")
        self.view_edges.setCheckable(True)
        self.view_edges.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.view_edges.setToolTip("Show mesh element edges over the color map")
        self.view_edges.toggled.connect(lambda on: self.r_edges.setChecked(on))
        vb_add(self.view_edges)
        db = QToolButton()
        db.setObjectName("dir_button")
        db.setText("View")
        theme.set_icon(db, "iso")
        db.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        db.setPopupMode(QToolButton.InstantPopup)
        db.setToolTip("View direction: isometric, front, top, right (more in the 3D view context menu)")
        dm = QMenu(db)
        for key, txt in VIEW_DIRECTIONS:
            dm.addAction(theme.icon(key), txt, lambda k=key: self.view.set_orientation(k))
        db.setMenu(dm)
        vb_add(db)
        vb_add(self.sec_button)
        fit = QToolButton()
        fit.setObjectName("fit_button")
        fit.setText("Fit")
        theme.set_icon(fit, "fit")
        fit.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        fit.setToolTip("Show the whole model  [F]")
        fit.clicked.connect(lambda: self.view.fit_all())
        vb_add(fit)
        self._center_layout.insertWidget(0, vb)
        self.tb_view = vb
        mode = str(self.settings.value("view/color", "part")) if _remember_ui() else "part"
        self._color_mode = mode if mode in self.color_actions else "part"
        self.color_actions[self._color_mode].setChecked(True)
        self.color_button.setText(self.color_actions[self._color_mode].text())
        self._sync_pick_button("face")
        self.r_qty.currentIndexChanged.connect(lambda _: self.update_view_info())
        self.r_step.currentIndexChanged.connect(lambda _: self.update_view_info())
        self.r_edges.toggled.connect(lambda on: self.view_edges.setChecked(on))
        from .properties import PropertiesCard           # Properties of the selected item above the step panel
        self.props_card = PropertiesCard(self)
        self._side_layout.insertWidget(1, self.props_card)
        pc = self.props_card.schedule
        for sig in (self.view.partsPicked, self.view.facesPicked, self.con_list.currentRowChanged,
                    self.load_list.currentRowChanged, self.lc_combo.currentIndexChanged,
                    self.mat_parts.itemSelectionChanged, self.joints_panel.bolt_list.currentRowChanged):
            sig.connect(pc)
        if hasattr(self, "model_table"):
            self.model_table.itemSelectionChanged.connect(pc)

    @staticmethod
    def _pick(button, key):
        button._acts[key].setChecked(True)
        button.setText(button._acts[key].text())

    def _on_transparency(self, key, button):
        self._pick(button, key)
        self.view.set_user_transparency(0.65 if key == "transparent" else 0.0)

    def _on_display_style(self, key, button):
        self._pick(button, key)
        self.view.set_display_style(key)

    def _on_mesh_show_mode(self, key, button):
        self._pick(button, key)
        if self.ms_show.isChecked() != (key == "mesh"):
            self.ms_show.setChecked(key == "mesh")

    def _on_mesh_color(self, key, button):
        self._pick(button, key)
        self._mesh_color = key
        if getattr(self, "_mesh_view", False):
            self.show_mesh()

    def _fill_qty_menu(self, menu):
        menu.clear()
        for i in range(self.r_qty.count()):
            a = menu.addAction(self.r_qty.itemText(i), lambda k=i: self.r_qty.setCurrentIndex(k))
            a.setCheckable(True)
            a.setChecked(i == self.r_qty.currentIndex())

    def set_color_mode(self, mode: str):
        """Part colour in 3D: "part" (each part a different colour) or "material" (by material, legend above the 3D view)."""
        self._color_mode = mode
        self.color_actions[mode].setChecked(True)
        self.color_button.setText(self.color_actions[mode].text())
        if _remember_ui():
            self.settings.setValue("view/color", mode)
        self.apply_part_colors()
        self.update_view_info()

    def material_colors(self) -> dict[str, tuple]:
        """Colour by the materials of the study (order by the first part with a material), a part without a material is grey."""
        out: dict[str, tuple] = {}
        for p in self.parts:
            try:
                m = self.study.material_of(p.name).name
            except (KeyError, AttributeError):
                continue
            if m not in out:
                out[m] = MATERIAL_COLORS[len(out) % len(MATERIAL_COLORS)]
        return out

    def apply_part_colors(self):
        if self.study is None or not self.parts:
            return
        if self._color_mode != "material":
            self.view.set_part_colors(None)
            return
        cols = self.material_colors()
        per = []
        for p in self.parts:
            try:
                per.append(cols[self.study.material_of(p.name).name])
            except (KeyError, AttributeError):
                per.append(NO_MATERIAL_COLOR)
        self.view.set_part_colors(per)

    def update_view_info(self):
        """Bar above the 3D window: description of the view per step, material legend, buttons per step (part colour outside
        results, quantity and mesh edges in results).
        """
        if not hasattr(self, "view_title"):
            return
        import html
        name = STAGES[getattr(self, "stage", 0)]
        st = self.study
        res_view = name in ("Results", "Fatigue") and self.results is not None
        self.color_button.setVisible(name in ("Model", "Materials", "3D printing"))
        self.transp_button.setVisible(name in ("Joints", "Contacts", "Members"))
        self.style_button.setVisible(name in ("Constraints", "Loads", "Heat", "Motion", "Dynamics"))
        mesh_stage = name == "Mesh" and self.mesh is not None
        self.mesh_show_button.setVisible(mesh_stage)
        self.mesh_color_button.setVisible(mesh_stage and self.ms_show.isChecked())
        if mesh_stage:
            self._pick(self.mesh_show_button, "mesh" if self.ms_show.isChecked() else "parts")
        self.qty_button.setVisible(res_view)
        self.view_edges.setVisible(res_view)
        legend = ""
        if st is None:
            title = ""
        elif res_view:
            title = f"{self.r_qty.currentText()} · {self.r_step.currentText()}"
            self.qty_button.setText(f"Quantity: {self.r_qty.currentText()}")
        elif name == "Mesh" or name == "Solve":
            m = self.mesh
            title = (f"{m.element_type} · {fmt(st.mesh.size, 3)} mm · {fmt(len(m.tets))} elements · mesh valid √"
                     if m is not None else f"The mesh is created during the analysis · {fmt(st.mesh.size, 3)} mm")
        elif name in ("Constraints", "Loads", "Heat", "Dynamics", "Motion"):
            lc = st.load_cases[self.lc_combo.currentIndex()] if st.load_cases and self.lc_combo.currentIndex() >= 0 \
                else None
            title = (f"{lc.name} · {len(lc.loads)} loads · {len(st.constraints)} supports" if lc is not None
                     else f"{len(st.constraints)} supports")
        elif name in ("Joints", "Contacts", "Members"):
            title = f"Joints: {len(st.bolts)} bolts · {len(st.welds)} welds · {len(st.contacts)} contacts"
        else:
            title = "Color by material" if self._color_mode == "material" else "Color by part"
        if st is not None and not res_view and self._color_mode == "material" and name not in ("Mesh",):
            cols = self.material_colors()
            counts: dict[str, int] = {}
            missing = 0
            for p in self.parts:
                try:
                    k = st.material_of(p.name).name
                    counts[k] = counts.get(k, 0) + 1
                except (KeyError, AttributeError):
                    missing += 1
            chips = [f"<span style='color:#{r:02x}{g:02x}{b:02x}'>■</span> {html.escape(k)} · {counts[k]}"
                     for k, (r, g, b) in cols.items()]
            if missing:
                r, g, b = NO_MATERIAL_COLOR
                chips.append(f"<span style='color:#{r:02x}{g:02x}{b:02x}'>■</span> no material · {missing}")
            legend = "&nbsp;&nbsp;".join(chips)
        self.view_title.setText(title)
        self.view_title.setVisible(bool(title))
        self.view_legend.setText(legend)
        self.view_legend.setVisible(bool(legend))

    def _sync_pick_button(self, key: str):
        """Text of the Select button and the help on the left above the 3D window by the selection filter."""
        txt = next(t for k, t, _, _ in PICK_FILTER_ACTIONS if k == key)
        self.pick_button.setText(f"Select: {txt}")
        theme.set_icon(self.pick_button, key)
        self.view_hint.setText(PICK_FILTER_SHORT[key])

    def _shelf_run(self, target):
        """Shelf command: "attribute" of the window – button (click, only if enabled) or method; "stage:Step" switches the step;
        a tuple (attribute, value) sets the text of a menu (load type).
        """
        def resolve(path):
            o = self
            for part in path.split("."):
                o = getattr(o, part, None)
                if o is None:
                    return None
            return o
        if isinstance(target, str) and target.startswith("@"):     # panel button of the step by its text
            from PySide6.QtWidgets import QAbstractButton
            panel = self.panel_stack.currentWidget()
            for b in panel.findChildren(QAbstractButton) if panel is not None else []:
                if b.text() == target[1:]:
                    if b.isEnabled():
                        b.click()
                    return
            return
        if isinstance(target, str) and ":" in target:
            kind, path = target.split(":", 1)
            if kind == "stage":
                self.set_stage(STAGE[path])
                return
            obj = resolve(path)
            if obj is None:
                return
            if kind == "focus":
                obj.setFocus()
                if hasattr(obj, "selectAll"):
                    obj.selectAll()
            elif kind == "popup":
                obj.showPopup()
            elif kind == "raise":
                obj.show()
                obj.raise_()
            elif kind == "menu":
                obj.showMenu()
            return
        if isinstance(target, tuple):
            combo = resolve(target[0])
            if combo is not None:
                i = combo.findText(target[1])
                if i >= 0:
                    combo.setCurrentIndex(i)
            return
        obj = resolve(target)
        if obj is None:
            return
        if hasattr(obj, "click"):
            if obj.isEnabled():
                obj.click()
        elif callable(obj):
            obj()

    def _update_shelf(self, name: str):
        """The shelf shows the commands of the current step; a step without commands = only selection and views."""
        if not hasattr(self, "_shelf_acts"):
            return
        for stage, acts in self._shelf_acts.items():
            for a in acts:
                a.setVisible(stage == name)
        self._shelf_stage = name
        has = bool(self._shelf_acts.get(name))
        self._shelf_label.setText(SHELF_LABELS.get(name, name).upper() if has else "")
        self._shelf_has_cmds = has
        self._fit_shelf()

    def _fit_shelf(self):
        """Shelf in a low window: first without the step name, then buttons with the icon only (name in the label), so that the commands
        do not hide in the dropdown button ⋮.
        """
        sh = getattr(self, "tb_shelf", None)
        if sh is None or getattr(self, "_fitting_shelf", False):
            return
        self._fitting_shelf = True
        try:
            avail = sh.height() if sh.isVisible() else 10 ** 6
            for lvl in range(3):
                self._shelf_level(lvl)
                if sh.sizeHint().height() <= avail:
                    break
        finally:
            self._fitting_shelf = False

    def _shelf_level(self, lvl: int):
        self._shelf_lvl = lvl
        self._shelf_label_act.setVisible(False)          # the step name was replaced by the group headings (SHELF ("#", …))
        cur = getattr(self, "_shelf_stage", None)
        for stage, heads in self._shelf_heads.items():
            for h in heads:
                h.setVisible(stage == cur and (lvl < 1 or h.isSeparator()))
        style = Qt.ToolButtonIconOnly if lvl >= 2 else Qt.ToolButtonTextUnderIcon
        if self.tb_shelf.toolButtonStyle() != style:
            self.tb_shelf.setToolButtonStyle(style)

    # ------------------------------------------------------------------ progress bar by phases (F11)
    def _build_flow_bar(self):
        """Phase tabs in the header (tb_flow: Model, Assembly, Conditions, Calculation, Results) and step tabs of the current phase
        above the step panel (tb_steps, coloured dot = state of the step). Visibility is controlled by `_apply_flow_visibility`
        (required steps always, optional ones only when used / pinned / opened).
        """
        from PySide6.QtWidgets import QButtonGroup, QMenu, QToolBar, QToolButton
        wf = QToolBar("Phase")
        wf.setObjectName("tb_flow")
        wf.setMovable(False)
        wf.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._flow_cols = {name: GREY for name in STAGES}   # colours of the step states (refresh_flow)
        self._flow_sig = None
        self._icon_cache: dict = {}
        self._flow_pins: set[str] = set()            # pinned optional steps (+ menu)
        self._flow_show_all = False
        self._flow_revealed: set[str] = set()        # optional steps opened in this session
        self._phase_last: dict[str, str] = {}        # last opened step of the phase (click on the phase tab)
        if _remember_ui():
            try:
                self._flow_pins = {n for n in json.loads(str(self.settings.value("flow/pinned", "[]")))
                                   if n in OPTIONAL_STAGES}
            except (ValueError, TypeError):
                self._flow_pins = set()
            self._flow_show_all = str(self.settings.value("flow/show_all", "0")) in ("1", "true", "True")
        self._phase_buttons: dict[str, QToolButton] = {}
        self._phase_acts: dict[str, QAction] = {}
        pg = QButtonGroup(self)
        pg.setExclusive(True)
        for phase, names in PHASES:
            b = QToolButton()
            b.setObjectName("phaseTab")
            b.setText(phase_label(phase))
            b.setCheckable(True)
            b.setToolTip(f"{phase_label(phase)}: {PHASE_TIPS[phase]}")
            b.setWhatsThis(f"<b>{phase_label(phase)}</b><p>Workflow phase: {PHASE_TIPS[phase]}. The steps of the phase are tabs above the panel on the right.</p>")
            b.clicked.connect(lambda _=False, ph=phase: self.goto_phase(ph))
            pg.addButton(b)
            self._phase_buttons[phase] = b
            self._phase_acts[phase] = wf.addWidget(b)
        self._phase_group = pg
        self.addToolBar(wf)
        self.tb_flow = wf
        st = QToolBar("Steps")
        st.setObjectName("tb_steps")
        st.setMovable(False)
        st.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        st.setIconSize(QSize(14, 14))
        grp = QActionGroup(self)
        self.stage_actions = []                      # in STAGES order (also hidden ones – widgetForAction returns them)
        for name in STAGES:
            i = STAGE[name]
            a = QAction(theme.step_icon(0, GREY), name, self)
            a.setCheckable(True)
            a.setToolTip(f"{name}: {STAGE_TIPS[name]}")
            a.setWhatsThis(f"{STAGE_LONG[name]}<br>{STAGE_COLORS}")
            a.triggered.connect(lambda _=False, k=i: self.set_stage(k))
            grp.addAction(a)
            st.addAction(a)
            self.stage_actions.append(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        st.addWidget(spacer)
        mb = QToolButton()
        mb.setObjectName("flowMore")
        theme.set_icon(mb, "plus")
        mb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        mb.setPopupMode(QToolButton.InstantPopup)
        mb.setToolTip("Optional steps: 3D printing, contacts, members, motion, heat, dynamics, fatigue")
        mb.setWhatsThis("<b>Optional steps</b><p>A step the study uses is among the tabs by itself; pin the others here (check box) or show all steps.</p>")
        mm = QMenu(mb)
        mm.aboutToShow.connect(lambda: self._fill_more_menu(mm))
        mb.setMenu(mm)
        self.flow_more = mb
        st.addWidget(mb)
        from .popout import popout_button
        pb = popout_button("Step panel in a separate window (larger, also on a second monitor)")
        pb.setWhatsThis("<b>Step panel in a separate window</b><p>The panel of the current step (materials, supports, loads, member nodes…) moves into its own window that can be enlarged or moved to a second monitor; the step tabs and the 3D view stay in the main window and the 3D window gets the freed space. Closing the window returns the panel.</p><p>Single sections of a panel have the same button in their header; the bottom Tables dock has it next to its tabs.</p>")
        pb.clicked.connect(lambda: self.set_panel_detached(not self.panel_detached()))
        self.btn_panel_popout = pb
        st.addWidget(pb)
        self._side_layout.insertWidget(0, st)
        self.tb_steps = st
        self._fit_levels = (-1, -1)                 # compaction of the header – see _fit_toolbars
        self._header_lvl = 0
        self._header_phase = None
        from PySide6.QtGui import QShortcut
        for key, d in (("Ctrl+PgDown", 1), ("Ctrl+PgUp", -1)):         # next / previous visible step
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda d=d: self.step_stage(d))

    def goto_phase(self, phase: str):
        """Phase tab: the last opened step of the phase, otherwise the first visible and enabled one."""
        names = dict(PHASES)[phase]
        last = self._phase_last.get(phase)
        if last in names and self.stage_visible(last) and self.stage_enabled(last):
            self.set_stage(STAGE[last])
            return
        for s in names:
            if self.stage_visible(s) and self.stage_enabled(s):
                self.set_stage(STAGE[s])
                return

    def stage_visible(self, name: str) -> bool:
        """Is the step on the progress bar? Required always; optional only when the study uses it (colour ≠ grey), it is pinned,
        all steps are shown, it was opened in this session, or it is current. A frame system without CAD hides
        the steps FRAME_ONLY_OFF and shows Frames.
        """
        fo = self.study is not None and self.frame_only()
        if fo and name in FRAME_ONLY_OFF:
            return False
        if name not in OPTIONAL_STAGES:
            return True
        if fo and name == "Members":
            return True
        return (self._flow_cols.get(name, GREY) != GREY or name in self._flow_pins or self._flow_show_all
                or name in self._flow_revealed or STAGES[getattr(self, "stage", 0)] == name)

    def stage_enabled(self, name: str) -> bool:
        """Is the step usable? (A hidden QAction always reports isEnabled() False – hence our own source of truth.)"""
        return not (self.study is not None and self.frame_only() and name in FRAME_ONLY_OFF)

    def _stage_used(self, name: str) -> bool:
        return self._flow_cols.get(name, GREY) != GREY

    def _flow_icon(self, n: int, color: str):
        key = (n, color)
        ic = self._icon_cache.get(key)
        if ic is None:
            ic = self._icon_cache[key] = theme.step_icon(n, color)
        return ic

    @staticmethod
    def _summary_color(colors: list[str]) -> str:
        for c in (RED, ORANGE, GREEN):
            if c in colors:
                return c
        return GREY

    def _apply_flow_visibility(self, fit: bool = True, force: bool = False):
        """Phase tabs (visible if the phase has at least one visible step; checked = phase of the current step) and the step tabs
        of the current phase with a state dot; then measures the header again (_fit_toolbars).
        """
        if not hasattr(self, "_phase_acts"):
            return
        cur = STAGES[getattr(self, "stage", 0)]
        vis = {n: self.stage_visible(n) for n in STAGES}
        sig = (tuple(vis.items()), tuple(self._flow_cols.items()), cur)
        if not force and sig == self._flow_sig:
            return
        self._flow_sig = sig
        for phase, names in PHASES:
            shown = [s for s in names if vis[s]]
            for s in names:
                a = self.stage_actions[STAGE[s]]
                a.setIcon(self._flow_icon(0, self._flow_cols[s]))
                a.setVisible(vis[s] and phase == PHASE_OF[cur])
            self._phase_acts[phase].setVisible(bool(shown))
            b = self._phase_buttons[phase]
            if b.isChecked() != (phase == PHASE_OF[cur]):
                b.setChecked(phase == PHASE_OF[cur])
        if self._header_phase != PHASE_OF[cur] and hasattr(self, "btn_phase"):
            self._header_phase = PHASE_OF[cur]
            self._update_header_actions(PHASE_OF[cur])
        if fit:
            self._fit_toolbars()

    def _fill_phase_menu(self, phase: str, menu):
        """Menu of a phase button: all steps of the phase with a state icon (hidden ones with a suffix), choice = switch the step."""
        menu.clear()
        cur = STAGES[getattr(self, "stage", 0)]
        for s in dict(PHASES)[phase]:
            col = self._flow_cols.get(s, GREY)
            hidden = not self.stage_visible(s)
            act = menu.addAction(self._flow_icon(0, col), s + (" (hidden)" if hidden else ""))
            act.setCheckable(True)
            act.setChecked(s == cur)
            act.setEnabled(self.stage_enabled(s))
            act.setToolTip(f"{s}: {STAGE_TIPS[s]}")
            act.triggered.connect(lambda _=False, k=STAGE[s]: self.set_stage(k))

    def _fill_more_menu(self, menu):
        """Analyses menu: optional steps (tick = pin to the bar), "Show all steps". The items are real check boxes
        (QWidgetAction) – in this menu style a checkable item without a tick would not be recognisable; the choice closes the menu.
        """
        from PySide6.QtWidgets import QCheckBox, QWidgetAction
        menu.clear()
        theme.add_menu_section(menu, "Show steps")
        self._more_checks: dict[str, QCheckBox] = {}

        def add_check(key, text, on, enabled, tip, slot):
            cb = QCheckBox(text)
            cb.setStyleSheet("QCheckBox { padding: 5px 14px 5px 12px; }")
            cb.setChecked(on)
            cb.setEnabled(enabled)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda v, f=slot: (menu.close(), f(v)))
            act = QWidgetAction(menu)
            act.setDefaultWidget(cb)
            menu.addAction(act)
            self._more_checks[key] = cb

        for s in OPTIONAL_STAGES:
            used = self._stage_used(s)
            add_check(s, ANALYSIS_LABELS[s] + (" – used in the study" if used else ""),
                      used or s in self._flow_pins or self._flow_show_all or s in self._flow_revealed,
                      not used and self.stage_enabled(s), f"{s}: {STAGE_TIPS[s]}",
                      lambda on, nm=s: self._pin_stage(nm, on))
        menu.addSeparator()
        add_check("all", "Show all steps", self._flow_show_all, True,
                  "Also show unused optional steps on the bar", self._set_show_all)

    def _save_flow_prefs(self):
        if _remember_ui():
            self.settings.setValue("flow/pinned", json.dumps(sorted(self._flow_pins)))
            self.settings.setValue("flow/show_all", "1" if self._flow_show_all else "0")

    def _pin_stage(self, name: str, on: bool):
        if on:
            self._flow_pins.add(name)
            self._save_flow_prefs()
            self.set_stage(STAGE[name])
        else:
            self._flow_pins.discard(name)
            self._flow_revealed.discard(name)
            self._save_flow_prefs()
            self._apply_flow_visibility()

    def _set_show_all(self, on: bool):
        self._flow_show_all = bool(on)
        self._save_flow_prefs()
        self._apply_flow_visibility()

    def step_stage(self, d: int):
        """Ctrl+PgDown / Ctrl+PgUp: next / previous visible (and enabled) step."""
        k = getattr(self, "stage", 0)
        while True:
            k += d
            if not 0 <= k < len(STAGES):
                return
            if self.stage_visible(STAGES[k]) and self.stage_enabled(STAGES[k]):
                self.set_stage(k)
                return

    # ------------------------------------------------------------------ bars by window width
    def _fit_toolbars(self):
        """Header in a narrower window: first without the document name, then without the Replace geometry button (it is in the File menu),
        without the program name and finally Run calculation as an icon only – the phase tabs and menus always stay.
        """
        if getattr(self, "_fitting", False) or not hasattr(self, "tb_right"):
            return
        from PySide6.QtWidgets import QToolBar, QToolButton
        self._fitting = True
        try:
            width = self.width() - 12
            row = [self.tb_file, self.tb_flow, self.tb_right]
            lvl = 0
            for lvl in range(5):
                self._header_level(lvl)
                if sum(t.sizeHint().width() for t in row) <= width:
                    break
            self._fit_levels = (lvl, 0)
            for tb in self.findChildren(QToolBar):          # the » dropdown button stays visible in the dark theme
                ext = tb.findChild(QToolButton, "qt_toolbar_ext_button")
                if ext is not None and ext.property("feaIcon") != "more":
                    theme.set_icon(ext, "more")                 # vertical dots – the button is narrow (~12 px)
                    ext.setIconSize(QSize(12, 12))
                    ext.setToolTip("More toolbar items that did not fit into the narrow window")
        finally:
            self._fitting = False

    def _header_level(self, lvl: int):
        self._header_lvl = lvl
        self._doc_act.setVisible(lvl < 1)
        n = len(getattr(self, "_header_acts", []))
        self._replace_act.setVisible(lvl < 2 and n > 0)
        self._phase2_act.setVisible(lvl < 2 and n > 1)
        self._brand_act.setVisible(lvl < 3)
        self.btn_run_header.setText("" if lvl >= 4 else tr("Run analysis"))

    @staticmethod
    def _set_qss(w, css: str):
        """Style of one widget; not recomputed when unchanged (polishing costs time at every compaction of the bar)."""
        if w.styleSheet() != css:
            w.setStyleSheet(css)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_toolbars()
        QTimer.singleShot(0, self._fit_shelf)       # the height of the shelf is known only after the window layout

    def _sketch_action(self, what: str) -> None:
        """File → New … from sketch, Model step → Edit sketch (the editor and the building are in fea_gui.sketch)."""
        from .sketch.hooks import dispatch
        dispatch(self, what)

    def _tools_button(self):
        """Tools menu: beam calculator, new frame system, mesh convergence."""
        from PySide6.QtWidgets import QMenu, QToolButton
        b = QToolButton()
        b.setText("Tools")
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        theme.set_icon(b, "tools")                 # in a narrow window only the icon (_fit_toolbars)
        b.setPopupMode(QToolButton.InstantPopup)
        help_tip(b, "Calculators, new beam structure, mesh convergence, modules",
                 "Tools: analytic beam calculator (beam, curved bar, buckling, EN 1993 check), friction, Hertz, press-fit, snap-fit, mechanism and damping calculators, new frame structure without CAD, mesh convergence")
        m = QMenu(b)
        m.addAction("Beam calculator…", self.open_beam_calc)
        m.addAction("Calculators (friction, Hertz, press fit, snap fit, heat)…", self.open_calc_dialog)
        m.addAction("Selection by criteria (Worksheet)…", self.open_criteria_selection_dialog)
        m.addAction("Section force (Free Body Cut)…", self.open_section_force_dialog)
        m.addAction("New frame (no CAD)", self.new_frame_study)
        m.addAction("Mesh convergence…", self.open_convergence_dialog)
        m.addSeparator()
        m.addAction("Replace geometry (keep setup)…", self.replace_geometry_dialog)
        m.addAction("Geometry iterations…", self.open_iterations)
        m.addSeparator()
        m.addAction("Program modules…", self.open_modules_dialog)
        b.setMenu(m)
        self.tools_button = b
        return b

    def _theme_button(self):
        """Appearance switch (by system / light / dark), stored in QSettings."""
        from PySide6.QtWidgets import QMenu, QToolButton
        b = QToolButton()
        b.setText("Appearance")
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setPopupMode(QToolButton.InstantPopup)
        b.setToolTip("Application appearance: follow Windows, light or dark")
        m = QMenu(b)
        grp = QActionGroup(m)
        cur = str(self.settings.value("theme", "system"))
        for key, txt in theme.MODES.items():
            a = m.addAction(txt, lambda k=key: self.set_theme(k))
            a.setCheckable(True)
            a.setChecked(key == cur)
            grp.addAction(a)
        b.setMenu(m)
        theme.set_icon(b, "theme")
        return b

    def set_theme(self, mode: str):
        """Switches the appearance at run time: palette + style, icons, 3D window background, colours of the step states."""
        self.settings.setValue("theme", mode)
        self.dark = theme.apply(QApplication.instance(), mode)
        theme.refresh_icons(self)
        self.view.set_dark_background(self.dark)
        self.refresh_flow()
        self._flow_sig = None
        self._fit_toolbars()                      # the style changes the widths of the bar buttons (polish once more afterwards)
        QTimer.singleShot(0, self._fit_toolbars)

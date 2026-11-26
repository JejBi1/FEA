"""Model set-up of the main window: the panels Model, Materials and Constraints, picking faces, edges, holes and points in 3D,
material assignment, sections and the supports.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import html
from html import escape
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton, QVBoxLayout, QAbstractItemView)

from fea_core import materials as matlib
from fea_core.geometry import face_props, bbox
from fea_core.selectors import FaceSelector, EdgeSelector, PointSelector
from fea_core.study import Material, Fixed, Displacement, CylindricalSupport, RemoteSupport, ElasticSupport
from fea_core.i18n import tr

from .safe_browser import SafeTextBrowser
from . import theme
from .print_panel import PRINT_FIELDS
from .widgets import Section, dspin, help_tip, panel_header, fit_rows, set_placeholder, table

from .main_defs import APP_NAME, STAGES, STAGE, PICK_FILTER_ACTIONS, fmt, msg_via_main as msg


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class SetupMixin(_Base):
    """Model set-up of the main window: the panels Model, Materials and Constraints, picking faces, edges, holes and points in
    3D, material assignment, sections and the supports."""

    # ------------------------------------------------------------------ panel Model
    def _panel_model(self, lay):
        lay.addWidget(panel_header("Model", "STEP geometry from CAD (parts = bodies, units mm) or a .fea project",
                                   STAGE["Model"] + 1))
        b = QPushButton("Open STEP / project…")
        theme.set_primary(b)
        b.setToolTip("Load STEP geometry (assembly parts = bodies) or a saved .fea project  [Ctrl+O]")
        b.setWhatsThis("<b>Open model</b><p>STEP (.step, .stp) creates a new study: each body is a part, part names are taken from the CAD (bolts and welds are recognized from them). A .fea project contains geometry, settings and results.</p><p>Model units are mm.</p>")
        self.btn_open_model = b
        b.clicked.connect(self.open_dialog)
        lay.addWidget(b)
        b = QPushButton("New frame (no CAD)")
        b.setToolTip("New study from members only (frame, truss) without STEP geometry")
        b.setWhatsThis("<b>Frame structure without CAD</b><p>A study without a solid: frames, trusses, beams. Nodes, members, cross-sections (EN library, parametric, custom) and supports are entered in the <i>Beams</i> step, member loads (nodal force, distributed load, gravity) in the <i>Loads</i> step. Solution: the program’s own exact beam solver (Timoshenko) – statics, natural frequencies and linear buckling without CalculiX; EN 1993-1-1 check.</p><p>Member material = the default material of the study (Materials step).</p>")
        b.clicked.connect(self.new_frame_study)
        self.btn_new_frame = b
        lay.addWidget(b)
        row = QHBoxLayout()
        for text, tip, what in (("New 2D from sketch…", "Draw a 2D section (plane stress / strain, axisymmetric) – no CAD", "twod"),
                                ("New beams from sketch…", "Draw a beam system – no CAD", "frame")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setWhatsThis("<b>Sketch</b><p>Lines, arcs, circles, rectangles and polylines from typed coordinates and named parameters (no constraint solver). A 2D sketch becomes parts of a plane or axisymmetric model, a beam sketch becomes nodes and members. Changing a parameter and applying the sketch again keeps constraints, loads and materials.</p>")
            b.clicked.connect(lambda _=False, w=what: self._sketch_action(w))
            row.addWidget(b)
        lay.addLayout(row)
        b = QPushButton("Edit sketch…")
        b.setToolTip("Change the drawing of this model (parameters, points, entities); the setup is kept")
        b.clicked.connect(lambda: self._sketch_action("edit"))
        b.setVisible(False)
        self.btn_edit_sketch = b
        lay.addWidget(b)

        self.model_table = table(["Show", "Analyze", "Part", "State", "Volume [cm³]", "Material"], stretch_last=True,
                                 placeholder="Open a STEP file or a project…", rows=(3, 12))
        self.model_table.itemChanged.connect(self._on_model_table_check)
        self.model_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.model_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_table.customContextMenuRequested.connect(self._on_model_table_menu)
        self.model_table.itemSelectionChanged.connect(self._on_model_table_selection)
        self.model_table.itemDoubleClicked.connect(lambda _: self.view.zoom_selection())
        self.model_table.setToolTip("Model bodies: Show = in 3D, Analyze = in the analysis; right-click = in bulk")
        self.model_table.setWhatsThis(
            "<b>Model bodies</b><p><i>Show</i> – the body is visible in the 3D view (display only, no effect on the analysis; a hidden body cannot be picked by clicking).</p><p><i>Analyze</i> – the body is in the analysis. A switched-off body is neither meshed nor analyzed (magnets, pins, purchased parts); in 3D it is transparent. The change discards the mesh; selections, supports and loads of the other bodies remain.</p><p>Right-click: switch the selected bodies on / off in bulk, isolate, switch off by name (e.g. <tt>*bolt*</tt>).</p>")
        lay.addWidget(self.model_table)

        row = QHBoxLayout()
        self.btn_toggle_ignore_part = QPushButton("Exclude / include in analysis")
        self.btn_toggle_ignore_part.setToolTip("Excludes the selected bodies from the analysis and meshing, or brings them back")
        self.btn_toggle_ignore_part.clicked.connect(self._toggle_selected_model_parts_ignore)
        row.addWidget(self.btn_toggle_ignore_part)
        self.btn_toggle_hide_part = QPushButton("Hide / show")
        self.btn_toggle_hide_part.setToolTip("Hides the selected bodies in the 3D view or shows them again (no effect on the analysis)")
        self.btn_toggle_hide_part.clicked.connect(self._toggle_selected_model_parts_hidden)
        row.addWidget(self.btn_toggle_hide_part)
        self.btn_show_all_parts = QPushButton("Show all")
        self.btn_show_all_parts.setToolTip("Shows all hidden bodies in 3D")
        self.btn_show_all_parts.clicked.connect(lambda: self.set_hidden([p.name for p in self.parts], False))
        row.addWidget(self.btn_show_all_parts)
        lay.addLayout(row)

        self.model_info = SafeTextBrowser()
        self.model_info.setMaximumHeight(130)
        self.model_info.setToolTip("Model parts with volume and assigned material, STEP geometry check warnings")
        lay.addWidget(self.model_info)

    def _panel_materials(self, lay):
        lay = self._scrolled(lay)                   # the content (part list + properties + orthotropy) fits only in the scroll area
        lay.addWidget(panel_header("Materials", "Select parts (here or by clicking in 3D, Ctrl = add) and assign a material",
                                   STAGE["Materials"] + 1))
        sec = Section("Parts", key="mat.parts", tip="Select the parts that get the material (Ctrl / Shift = several, or click in 3D)")
        slay = QVBoxLayout(sec.body)
        slay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(sec)
        self.mat_parts = QListWidget()
        fit_rows(self.mat_parts, 3, 10)             # the part list grows with the number of parts, scroll bar above 10 rows
        self.mat_parts.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mat_parts.itemSelectionChanged.connect(self.on_mat_parts_sel)
        help_tip(self.mat_parts, "Parts and their material – select parts (Ctrl / Shift = several, or click in 3D)",
                 "Parts and their material. Select parts (Ctrl / Shift = several, or by clicking in 3D), then choose a material and “Assign to selected”.")
        slay.addWidget(self.mat_parts)
        sec = Section("Material", key="mat.material", tip="Material from the library, converter of designations, summary of properties")
        slay = QVBoxLayout(sec.body)
        slay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(sec)
        mat_lay, lay = lay, slay
        self.mat_combo = QComboBox()
        self.mat_combo.currentIndexChanged.connect(self.on_mat_combo)
        help_tip(self.mat_combo, "Material from the library or the study – values are loaded into the form",
                 "Material from the built-in library (with the source of the values), from your library or from the study.<br>Selecting it loads the properties into the form below – you can edit them before assigning.")
        lay.addWidget(self.mat_combo)
        self.mat_find = QLineEdit()
        self.mat_find.setPlaceholderText("Converter: ČSN / EN / W.Nr. / AISI / JIS… designation (e.g. 12050, St 52-3)")
        self.mat_find.setClearButtonEnabled(True)
        self.mat_find.textChanged.connect(self.on_mat_find)
        help_tip(self.mat_find, "Converter: type a designation in any standard (ČSN, EN, AISI…)",
                 "Designation converter: type a designation in any standard (ČSN 11 373, 1.0038, S235JR, A36, 6061…) – it selects the matching library material, otherwise it offers similar ones (links under the field).")
        lay.addWidget(self.mat_find)
        self.mat_conv = QLabel()
        self.mat_conv.setWordWrap(True)
        self.mat_conv.setVisible(False)             # an empty label would leave a gap in the layout
        self.mat_conv.linkActivated.connect(lambda url: self.mat_combo.setCurrentText(self._mat_links[int(url[4:])]))
        self._mat_links: list[str] = []
        lay.addWidget(self.mat_conv)
        # material summary from the form; full properties and orthotropy are in a pop-up window (clearer
        # than a long scrolling panel) – the fields m_* and ortho_box live in the window, the panel logic reads them unchanged
        self.mat_summary = QLabel()
        self.mat_summary.setWordWrap(True)
        self.mat_summary.setTextFormat(Qt.RichText)
        help_tip(self.mat_summary, "Material in the form – edit with “Material properties…”",
                 "Summary of the material loaded into the form (from the list or edited). Edit values, thermal properties, source and orthotropy in the “Material properties…” window.")
        lay.addWidget(self.mat_summary)
        b = QPushButton("Material properties…")
        help_tip(b, "Edit material properties (E, ν, ρ, Re, Rm, heat, orthotropy) in a window",
                 "Opens a window with the material properties: name, E, ν, ρ, Re, Rm, thermal properties, source and orthotropic constants. The window can stay open – parts can still be selected in 3D and assigned.")
        b.clicked.connect(self.open_material_dialog)
        self.btn_mat_props = b
        lay.addWidget(b)
        self._build_material_dialog()
        sec = Section("Assignment", key="mat.assign", tip="Assign the material from the form to parts, save it to your library")
        lay = QVBoxLayout(sec.body)
        lay.setContentsMargins(0, 0, 0, 0)
        mat_lay.addWidget(sec)
        h = QHBoxLayout()
        b1 = QPushButton("Assign to selected")
        b1.setToolTip("Assign the material from the form to the parts selected in the list (or by clicking in 3D)")
        b1.clicked.connect(lambda: self.assign_material(False))
        self.btn_assign_sel = b1
        theme.set_primary(b1)
        b2 = QPushButton("Assign to all")
        b2.setToolTip("Assign the material from the form to all parts of the model")
        b2.clicked.connect(lambda: self.assign_material(True))
        h.addWidget(b1)
        h.addWidget(b2)
        lay.addLayout(h)
        b3 = QPushButton("Save to my material library")
        help_tip(b3, "Save the material from the form to your own library (offered in other projects too)",
                 "Save the material from the form to your own library (it is offered in other projects too):<br>"
                 + str(matlib.user_library_path()))
        b3.clicked.connect(self.save_user_material)
        lay.addWidget(b3)
        mat_lay.addStretch(1)

    def _build_material_dialog(self):
        """Non-modal "Material properties" window: form (m_*) + orthotropy (ortho_box) + assignment."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Material properties")
        dlg.setModal(False)
        outer = QVBoxLayout(dlg)
        lay = self._scrolled(outer)
        g = Section("Mechanical properties", key="matdlg.mech", detachable=False,
                    tip="Name, elastic constants, density and strengths – editable, then assign")
        f = QFormLayout(g.body)
        g_th = Section("Thermal properties", key="matdlg.thermal", detachable=False,
                       tip="Conductivity, expansion, heat capacity – for the Heat step and thermal stress")
        f_th = QFormLayout(g_th.body)
        g_src = Section("Source", key="matdlg.source", detachable=False,
                        tip="Source of the values (standard, catalog, measurement) – in the report")
        f_src = QFormLayout(g_src.body)
        self.m_name = QLineEdit()
        self.m_E = dspin(210000, 1, 1e7, 0, suffix=" MPa")
        self.m_nu = dspin(0.3, 0.0, 0.499, 3)
        self.m_rho = dspin(7850, 0, 1e5, 0, suffix=" kg/m³")
        self.m_Re = dspin(235, 0, 1e5, 0, suffix=" MPa")
        self.m_Rm = dspin(360, 0, 1e5, 0, suffix=" MPa")
        self.m_src = QLineEdit()
        self.m_name.setToolTip("Material name – assigned to parts under this name (changing the name = new material)")
        self.m_E.setToolTip("Tensile Young's modulus (steel 210 000, aluminum 70 000, PLA ~3 000 MPa)")
        self.m_nu.setToolTip("Poisson's ratio (steel 0.3, aluminum 0.33, plastics 0.35–0.4)")
        self.m_rho.setToolTip("Density – for self-weight, rotation and transient heat conduction (steel 7850 kg/m³)")
        self.m_Re.setToolTip("Yield strength Re / Rp0.2 – safety factor to Re in results, weld assessment")
        self.m_Rm.setToolTip("Tensile strength – S-N curve estimate for fatigue, weld and thread assessment")
        self.m_src.setToolTip("Source of values (standard, catalog, measurement) – included in the report")
        # heat: 0 = not entered (None); α is shown in 10⁻⁶/K, stored in 1/K
        self.m_k = dspin(0.0, 0, 1e4, 3, suffix=" W/(m·K)")
        self.m_alpha = dspin(0.0, 0, 1e4, 3, suffix=" ·10⁻⁶/K")
        self.m_cp = dspin(0.0, 0, 1e6, 1, suffix=" J/(kg·K)")
        for w in (self.m_k, self.m_alpha, self.m_cp):
            w.setSpecialValueText("not specified")
        self.m_k.setToolTip("Thermal conductivity – required for heat conduction (Heat step)")
        self.m_alpha.setToolTip("Coefficient of thermal expansion – required for thermal stress")
        self.m_cp.setToolTip("Specific heat capacity – transient heat conduction only (with density ρ)")
        for lbl, w in (("Name", self.m_name), ("E", self.m_E), ("ν", self.m_nu), ("ρ", self.m_rho),
                       ("Re (Rp0.2)", self.m_Re), ("Rm", self.m_Rm)):
            f.addRow(lbl, w)
        for lbl, w in (("λ (conductivity)", self.m_k), ("α (expansion)", self.m_alpha), ("c_p (heat capacity)", self.m_cp)):
            f_th.addRow(lbl, w)
        f_src.addRow("Source", self.m_src)
        self._mat_print: dict = {}          # print properties (E_z, Rm_z…) of the selected material – keep
        self._mat_form_src = None           # (name, repr) of the material loaded into the form
        self._mat_select = None             # material that should stay selected after the list is refreshed (assigned)
        self.m_print = QLabel()
        self.m_print.setWordWrap(True)
        f.addRow(self.m_print)
        f.setRowVisible(self.m_print, False)        # an empty row would leave a gap (it is shown with text)
        self._mat_form = f
        lay.addWidget(g)
        lay.addWidget(g_th)
        lay.addWidget(g_src)
        from .ortho_box import OrthoBox
        self.ortho_box = OrthoBox(self)          # general orthotropic material (Material.ortho, ortho_csys)
        lay.addWidget(self.ortho_box)
        from .plasticity_box import PlasticityBox
        self.plasticity_box = PlasticityBox(self)    # elastoplastic metal (Material.plastic, fea_core.plasticity)
        lay.addWidget(self.plasticity_box)
        lay.addStretch(1)
        h = QHBoxLayout()
        b1 = QPushButton("Assign to selected")
        b1.setToolTip("Assign the material from the form to the parts selected in the list (or by clicking in 3D)")
        b1.clicked.connect(lambda: self.assign_material(False))
        theme.set_primary(b1)
        b2 = QPushButton("Assign to all")
        b2.setToolTip("Assign the material from the form to all parts of the model")
        b2.clicked.connect(lambda: self.assign_material(True))
        b3 = QPushButton("Save to library")
        b3.setToolTip("Save the material from the form to your own library (offered in other projects too)")
        b3.clicked.connect(self.save_user_material)
        h.addWidget(b1)
        h.addWidget(b2)
        h.addWidget(b3)
        h.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.close)
        h.addWidget(bb)
        outer.addLayout(h)
        dlg.resize(580, 780)
        self.mat_dialog = dlg
        for w in (self.m_E, self.m_nu, self.m_rho, self.m_Re, self.m_Rm, self.m_k, self.m_alpha, self.m_cp):
            w.valueChanged.connect(self.update_mat_summary)
        for w in (self.m_name, self.m_src):
            w.textChanged.connect(self.update_mat_summary)
        self.ortho_box.check.toggled.connect(self.update_mat_summary)
        self.plasticity_box.check.toggled.connect(self.update_mat_summary)
        self.update_mat_summary()

    def open_material_dialog(self):
        """Shows the material properties window (non-modal – selecting parts in 3D keeps working)."""
        self.mat_dialog.show()
        self.mat_dialog.raise_()
        self.mat_dialog.activateWindow()

    def update_mat_summary(self, *_):
        """Short summary of the material form in the panel (full values are in the Material properties window)."""
        if not hasattr(self, "mat_summary"):
            return
        def opt(w, unit):
            return f"{fmt(w.value())} {unit}" if w.value() > 0 else tr("not specified")
        name = html.escape(self.m_name.text().strip() or "—")
        lines = [f"<b>{name}</b>",
                 f"E {fmt(self.m_E.value())} MPa · ν {fmt(self.m_nu.value())} · ρ {fmt(self.m_rho.value())} kg/m³",
                 f"Re {opt(self.m_Re, 'MPa')} · Rm {opt(self.m_Rm, 'MPa')}"]
        if self.m_k.value() > 0 or self.m_alpha.value() > 0 or self.m_cp.value() > 0:
            lines.append(f"λ {opt(self.m_k, 'W/(m·K)')} · α {opt(self.m_alpha, '·10⁻⁶/K')} · "
                         f"c_p {opt(self.m_cp, 'J/(kg·K)')}")
        if self.ortho_box.isChecked():
            lines.append("<i>" + tr("orthotropic material") + "</i>")
        elif self._mat_print:
            lines.append("<i>" + tr("printed material – print properties in the 3D printing step") + "</i>")
        if self.plasticity_box.isChecked():
            lines.append("<i>" + tr("elastoplastic material – nonlinear analysis") + "</i>")
        self.mat_summary.setText("<br>".join(lines))

    # ------------------------------------------------------------------ Constraints panel
    def _panel_constraints(self, lay):
        lay.addWidget(panel_header("Constraints", "Click to select faces in 3D (click empty space = clear), then choose a support",
                                   STAGE["Constraints"] + 1))
        self.con_list = QListWidget()
        self.con_list.currentRowChanged.connect(self.on_con_row)
        help_tip(self.con_list, "Study supports – a click on a row highlights the faces and loads the values",
                 "Study supports. A click on a row highlights their faces and loads the values into the form (edit with the “Save changes…”, “Faces from selection” and “Delete” buttons).")
        set_placeholder(self.con_list, "No supports yet – select faces and choose a support")
        fit_rows(self.con_list, 3, 8)
        lay.addWidget(self.con_list)
        b = QPushButton("Fixed support from selection")
        b.setToolTip("Fix the selected faces (ux = uy = uz = 0)")
        b.setWhatsThis("<b>Fixed support</b><p>The simplest support: the selected faces cannot move in any direction. First select the faces by clicking in 3D (Face filter P), then press the button.</p><p>The model must be supported so that it cannot move or rotate freely – the Solve step checks the rigid body motions.</p><p>Edges of a fixed support tend to be singularities – do not evaluate the stress directly on them.</p>")
        self.btn_fixed = b
        b.clicked.connect(self.add_fixed)
        lay.addWidget(b)
        g = Section("Prescribed displacement", key="con.disp", tip="Support with selected displacement components",
                    help="Support with selected displacement components: symmetry (only the normal component = 0), sliding support, or an imposed displacement (non-zero value).")
        grid = QGridLayout(g.body)
        self.d_chk, self.d_val = [], []
        for i, ax in enumerate("xyz"):
            c = QCheckBox(f"u{ax}")
            c.setChecked(True)
            help_tip(c, f"Prescribe the displacement in the {ax} direction (unticked = free direction)",
                     f"Prescribe displacement in direction {ax} (unchecked = free direction; symmetry: normal component only = 0)")
            v = dspin(0.0, -1e4, 1e4, 4, suffix=" mm")
            v.setToolTip(f"Displacement value u{ax} [mm] (0 = support / symmetry, nonzero = imposed displacement)")
            grid.addWidget(c, i, 0)
            grid.addWidget(v, i, 1)
            self.d_chk.append(c)
            self.d_val.append(v)
        b = QPushButton("Displacement from selection (checked components)")
        b.setToolTip("New support: prescribe the checked displacement components on the selected faces (other directions free)")
        b.clicked.connect(self.add_displacement)
        grid.addWidget(b, 3, 0, 1, 2)
        b = QPushButton("Save changes to the selected displacement")
        b.setToolTip("Overwrite the components and values of the prescribed displacement selected in the supports list")
        b.clicked.connect(self.update_displacement)
        grid.addWidget(b, 4, 0, 1, 2)
        lay.addWidget(g)
        g = Section("Cylindrical support (hole / pin)", key="con.cyl", tip="Support in the cylinder axis components (radial, tangential, axial)",
                    help="Support in the components of the cylinder axes (like Cylindrical Support): a ticked component has zero displacement.<br>The axis is determined from the selected cylindrical faces (coaxial, same radius – see the “Hole” H selection filter).")
        grid = QGridLayout(g.body)
        self.cyl_chk = []
        for i, (txt, tip) in enumerate((("radially", "Prevent radial displacement of the face (normal to the cylinder axis)"),
                                        ("tangentially (rotation)", "Prevent rotation about the cylinder axis"),
                                        ("axially", "Prevent displacement along the cylinder axis"))):
            c = QCheckBox(txt)
            c.setToolTip(tip)
            c.setChecked(True)                   # default as in the core (CylindricalSupport): all three components
            grid.addWidget(c, 0, i)
            self.cyl_chk.append(c)
        b = QPushButton("Cylindrical support from selection")
        b.setToolTip("New cylindrical support on the selected cylindrical faces (hole for a pin / bolt, pin in a bearing)")
        b.clicked.connect(self.add_cylindrical)
        grid.addWidget(b, 1, 0, 1, 3)
        b = QPushButton("Save changes to the selected cylindrical support")
        b.setToolTip("Overwrite the constrained components of the cylindrical support selected in the list")
        b.clicked.connect(self.update_cylindrical)
        grid.addWidget(b, 2, 0, 1, 3)
        lay.addWidget(g)
        self._panel_remote_support(lay)
        self._panel_elastic(lay)
        from .cyclic_gui import CyclicBox             # cyclic symmetry of a sector (fea_core.cyclic)
        self.cyclic_box = CyclicBox(self)
        lay.addWidget(self.cyclic_box)
        h = QHBoxLayout()
        b = QPushButton("Faces from selection")
        b.setToolTip("Replace the faces of the selected support with the current selection")
        b.clicked.connect(lambda: self.update_faces_of(self.con_list, "constraint"))
        h.addWidget(b)
        b = QPushButton("Delete")
        b.setToolTip("Delete the support selected in the list (Ctrl+Z to undo)")
        b.clicked.connect(lambda: self.delete_item(self.con_list, "constraint"))
        h.addWidget(b)
        lay.addLayout(h)

    def _panel_remote_support(self, lay):
        """Remote constraint (like Remote Displacement): point, constrained components ux…rz, rigid / distributing coupling of a face."""
        g = Section("Remote support (point off the face)", key="con.remote", collapsed=True,
                    tip="Faces attached to a remote point (displacements and rotations at the point)",
                    help="Faces attached to a remote point; the ticked displacements (ux, uy, uz) and rotations about the point (rx, ry, rz) are zero at the point. Displacements only = hinge.<br>Rigid: the face does not deform (it must not share nodes with a fixed support / displacement). Distributing: holds only the weighted average of the face, the face deforms (dense equations – suitable for smaller faces). Reaction = force and moment about the point.")
        self.rs_group = g                    # collapsible section (the Constraints panel has no scroll bar – the list of constraints would shrink)
        grid = QGridLayout(g.body)
        grid.setContentsMargins(0, 0, 0, 0)
        self.rs_pt = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        hp = QHBoxLayout()
        for s, ax in zip(self.rs_pt, "xyz"):
            s.setToolTip(f"{ax} coordinate of the remote point [mm] (circle center: Edge filter → “Use center”)")
            hp.addWidget(s)
        grid.addWidget(QLabel("Point [mm]"), 0, 0)
        grid.addLayout(hp, 0, 1, 1, 3)
        self.register_point_input(self.rs_pt)
        b = QPushButton("Selection centroid")
        b.setToolTip("Point = centroid of the selected faces (weighted by area)")
        b.clicked.connect(self.remote_point_from_selection)
        grid.addWidget(b, 1, 0, 1, 2)
        self.rs_coupling = QComboBox()
        self.rs_coupling.addItems(["rigid", "distributing (RBE3)"])
        help_tip(self.rs_coupling, "Rigid = face acts as a rigid body, distributing (RBE3) = face deforms",
                 "Rigid: the face moves as a rigid body (adds stiffness).<br>Distributing (RBE3): holds only the weighted average of the displacements, the face may deform.")
        grid.addWidget(self.rs_coupling, 1, 2, 1, 2)
        self.rs_chk = []
        hc = QHBoxLayout()
        for i, txt in enumerate(("ux", "uy", "uz", "rx", "ry", "rz")):
            c = QCheckBox(txt)
            c.setChecked(True)
            c.setToolTip(("point displacement in direction " if i < 3 else "rotation about point axis ") + "xyz"[i % 3])
            hc.addWidget(c)
            self.rs_chk.append(c)
        grid.addLayout(hc, 2, 0, 1, 4)
        b = QPushButton("From selection")
        b.setToolTip("New remote support on the selected faces")
        b.clicked.connect(self.add_remote_support)
        grid.addWidget(b, 3, 0, 1, 2)
        b = QPushButton("Save changes")
        b.setToolTip("Save the point, components and type of the selected remote support")
        b.clicked.connect(self.update_remote_support)
        grid.addWidget(b, 3, 2, 1, 2)
        lay.addWidget(g)

    def _panel_elastic(self, lay):
        """Elastic support of a face (Winkler foundation, like Elastic Support): stiffness per area or total."""
        g = Section("Elastic support (foundation, springs)", key="con.elastic", collapsed=True,
                    tip="Faces on an elastic foundation (Winkler foundation)",
                    help="Selected faces resting on an elastic foundation (Winkler foundation, like Elastic Support in Ansys): stiffness k [N/mm³] per unit area, or total stiffness K [N/mm] distributed by area.<br>Normal to the face (foundation, rubber pad) or in all directions. Linear – it also transmits tension.")
        self.el_group = g                    # collapsible section as the remote constraint
        f = QFormLayout(g.body)
        f.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        self.el_k = dspin(1.0, 1e-9, 1e12, 6, 0.1)
        help_tip(self.el_k, "Support stiffness: per area k [N/mm³], or total K [N/mm]",
                 "Support stiffness: per area k [N/mm³] (pressure [MPa] per 1 mm of compression; concrete ~0.05–0.2, rubber 5 mm ~1–10), or total stiffness K [N/mm] (distributed by face area)")
        h.addWidget(self.el_k, 1)
        self.el_unit = QComboBox()
        self.el_unit.addItem("N/mm³ (per area)", True)
        self.el_unit.addItem("N/mm (total)", False)
        help_tip(self.el_unit, "Stiffness per area [N/mm³], or total [N/mm]",
                 "Stiffness per unit area k [N/mm³] (foundation), or total stiffness K [N/mm] of all selected faces together (e.g. the catalog stiffness of a rubber bushing)")
        h.addWidget(self.el_unit)
        f.addRow("Stiffness", h)
        self.el_dir = QComboBox()
        self.el_dir.addItem("normal to face", "normal")
        self.el_dir.addItem("in all directions (x, y, z)", "all")
        help_tip(self.el_dir, "Springs normal to the face only (foundation), or in all directions",
                 "Normal to the face: springs only in the normal direction (foundation – the face may slide;<br>for a support without sliding add a support against displacement and rotation in the plane).<br>In all directions: the same stiffness in x, y, z (rubber insert, bonded joint).")
        f.addRow("Direction", self.el_dir)
        h = QHBoxLayout()
        b = QPushButton("From selection")
        b.setToolTip("New elastic support on the faces selected in 3D")
        b.setWhatsThis("<b>Elastic support</b><p>Select faces (e.g. the bottom face of a plate on a foundation), enter the stiffness and direction and press <i>From selection</i>. The support reaction is in the results under the selection name.</p><p>A support only normal to the face does not prevent motion in the plane of the face – the model must also be fixed otherwise (rigid-body motion check in the Solve step).</p>")
        b.clicked.connect(self.add_elastic)
        h.addWidget(b)
        b = QPushButton("Save changes")
        b.setToolTip("Save the stiffness and direction of the elastic support selected in the supports list")
        b.clicked.connect(self.update_elastic)
        h.addWidget(b)
        f.addRow(h)
        lay.addWidget(g)

    def elastic_form(self) -> tuple[float, str, bool]:
        return float(self.el_k.value()), self.el_dir.currentData(), bool(self.el_unit.currentData())

    def add_elastic(self):
        sels = self.need_faces()
        if not sels:
            return
        k, d, per_area = self.elastic_form()
        name = self.new_selection("Elastic support", sels)
        self.study.constraints.append(ElasticSupport(name, k, d, per_area))
        self.view.set_selected_faces([])
        self.changed()

    def update_elastic(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or not isinstance(self.study.constraints[row], ElasticSupport):
            msg(self, APP_NAME, "Select an elastic support in the list.")
            return
        c = self.study.constraints[row]
        c.stiffness, c.direction, c.per_area = self.elastic_form()
        self.changed()
        self.con_list.setCurrentRow(row)

    # ================================================================== face selection
    def on_faces_picked(self, faces):
        self.update_status()

    @staticmethod
    def arrow_point(p):
        """Arrow point: centroid of a planar face (lies on it unless it has holes), otherwise a point on the face."""
        return p["centroid"] if p["type"] == "plane" else p["point"]

    def face_prop(self, pi, fi) -> dict:
        props = self.face_cache.get(pi)
        if props is None:
            props = self.face_cache[pi] = [face_props(f) for f in self.view.part_faces[pi]]
        return props[fi]

    def selected_selectors(self) -> list[FaceSelector]:
        out = []
        for pi, fi in self.view.selected_faces():
            props = self.face_cache.get(pi)
            if props is None:
                props = self.face_cache[pi] = [face_props(f) for f in self.view.part_faces[pi]]
            out.append(FaceSelector.from_props(self.parts[pi].name, props[fi]))
        return out

    def need_faces(self):
        sels = self.selected_selectors()
        if not sels:
            msg(self, APP_NAME, "First select one or more faces in the 3D view.")
        return sels

    def edge_prop(self, pi, ei) -> dict:
        from fea_core.geometry import edge_props
        props = self.edge_cache.get(pi)
        if props is None:
            props = self.edge_cache[pi] = [edge_props(e) for e in self.view.part_edges(pi)]
        return props[ei]

    def selected_edge_selectors(self) -> list[EdgeSelector]:
        return [EdgeSelector.from_props(self.parts[pi].name, self.edge_prop(pi, ei))
                for pi, ei in self.view.selected_edges()]

    def selected_point_selectors(self) -> list[PointSelector]:
        return [PointSelector(self.parts[pi].name, [float(c) for c in q]) for pi, q in self.view.selected_points()]

    def load_target(self) -> str | None:
        """What the force from the form acts on: edges (filter Edge), points (filter Point), otherwise faces; no selection → None."""
        f = self.view.pick_filter
        e, pts, fc = self.view.selected_edges(), self.view.selected_points(), self.view.selected_faces()
        if f == "edge" and e:
            return "edges"
        if f == "point" and pts:
            return "points"
        if fc:
            return "faces"
        return "edges" if e else "points" if pts else None

    def new_entity_selection(self, prefix, entity) -> str:
        """New edge / point selection from the current 3D selection."""
        prefix = tr(prefix)                         # new names in the program language
        k = 1
        while f"{prefix} {k}" in self.study.selections:
            k += 1
        name = f"{prefix} {k}"
        if entity == "edges":
            self.study.add_selection(name, edges=self.selected_edge_selectors())
            self.sel_edges[name] = self.view.selected_edges()
        else:
            self.study.add_selection(name, points=self.selected_point_selectors())
            self.sel_points[name] = self.view.selected_points()
        return name

    def new_selection(self, prefix, sels) -> str:
        prefix = tr(prefix)                         # new names in the program language
        k = 1
        while f"{prefix} {k}" in self.study.selections:
            k += 1
        name = f"{prefix} {k}"
        self.study.add_selection(name, sels)
        self.sel_faces[name] = self.view.selected_faces()
        return name

    # ================================================================== materials
    def refresh_materials(self, select: str | None = None):
        """Part list and material menu after a change of the study. The selected material (or `select`, e.g. the one just
        assigned) stays selected; the form is reloaded only when the selection changes (unfinished values stay).
        """
        rows = {i.row() for i in self.mat_parts.selectedIndexes()} if self.mat_parts.count() == len(self.parts) else set()
        self.mat_parts.blockSignals(True)
        self.mat_parts.clear()
        for i, p in enumerate(self.parts):
            m = self.study.material_of(p.name).name if self.study else "–"
            self.mat_parts.addItem(f"{p.name}   [{m}]")
            if i in rows:
                self.mat_parts.item(i).setSelected(True)
        self.mat_parts.blockSignals(False)
        prev = self.mat_combo.currentText()
        select = select or self._mat_select
        self._mat_select = None
        want = select or prev
        self.mat_combo.blockSignals(True)
        self.mat_combo.clear()
        names = [m.name for m in matlib.all_materials()]
        if self.study:
            names += [n for n in self.study.materials if n not in names]
        self.mat_combo.addItems(names)
        i = self.mat_combo.findText(want) if want else -1
        if i >= 0:
            self.mat_combo.setCurrentIndex(i)
        self.mat_combo.blockSignals(False)
        if i < 0 or select is not None or want != prev or self._mat_form_src != self._mat_source(want):
            self.on_mat_combo(self.mat_combo.currentIndex())

    def _mat_source(self, name):
        """Material by name (the study has priority over the library) as text – the form is reloaded only when
        the material changed (undo / redo, another study).
        """
        m = self.study.materials.get(name) if self.study else None
        if m is None:
            try:
                m = matlib.by_name(name)
            except KeyError:
                return None
        return name, repr(m)

    def on_mat_combo(self, _):
        name = self.mat_combo.currentText()
        if not name:
            return
        m = self.study.materials.get(name) if self.study else None
        if m is None:
            try:
                m = matlib.by_name(name)
            except KeyError:
                return
        self._mat_form_src = (name, repr(m))
        self.m_name.setText(m.name)
        self.m_E.setValue(m.E)
        self.m_nu.setValue(m.nu)
        self.m_rho.setValue(m.rho * 1e12)
        self.m_Re.setValue(m.Re or 0)
        self.m_Rm.setValue(m.Rm or 0)
        self.m_k.setValue(m.k or 0)
        self.m_alpha.setValue((m.alpha or 0) * 1e6)
        self.m_cp.setValue(m.cp or 0)
        self.m_src.setText(m.source)
        self._mat_print = {k: getattr(m, k) for k in PRINT_FIELDS} if m.is_print else {}
        self.m_print.setText(
            f"<i>Printed material (E, Rm = layer plane XY): E_z {fmt(m.E_z)} MPa, Rm_z {fmt(m.Rm_z)} MPa – shell, infill and other properties in the 3D printing step.</i>" if m.is_print else
            f"<i>{self.ortho_box.describe(m)}</i>" if m.is_ortho else "")
        self._mat_form.setRowVisible(self.m_print, bool(self.m_print.text()))
        self.ortho_box.load(m)
        self.plasticity_box.load(m)
        self.update_mat_summary()

    def on_mat_find(self, text):
        """Designation converter: equivalents in other standards, matching / similar material from the library."""
        from fea_core import grades
        self._mat_links = []
        if not text.strip():
            self.mat_conv.setText("")
            self.mat_conv.setVisible(False)
            return
        hits = grades.lookup(text)
        if not hits:
            self.mat_conv.setText("<i>The designation is not in the converter.</i>")
            self.mat_conv.setVisible(True)
            return
        g, approx = hits[0]

        def link(name, extra=""):
            self._mat_links.append(name)
            return f"<a href='mat:{len(self._mat_links) - 1}'>{escape(str(name))}</a>{extra}"
        rows = [f"<b>{g.key}</b> – {g.family_label}{' (approximate match only)' if approx else ''}",
                g.summary(html=True)]
        if len(hits) > 1:
            rows.append("Also: " + ", ".join(f"{h.key}{' (≈)' if a else ''}" for h, a in hits[1:4]))
        mats = {m.name: m for m in matlib.all_materials()}
        if self.study:
            mats.update(self.study.materials)
        same, similar = [], []
        q = grades.profile(g)
        for m in mats.values():
            mg = grades.grade_of(m.name)
            if mg is g:
                same.append(m.name)
            elif mg is not None:
                d = grades.distance(q, grades.profile(mg, rm=m.Rm))
                if d <= 6:
                    similar.append((d, m.name, grades.difference(q, grades.profile(mg, rm=m.Rm))))
        if same:
            rows.append("In the library: " + ", ".join(link(n) for n in same))
            if self.mat_combo.findText(same[0]) >= 0:
                self.mat_combo.setCurrentText(same[0])
        else:
            rows.append("Not in the library" + (" – similar: " + ", ".join(
                link(n, f" <small>({escape(str(why))})</small>" if why else "") for _, n, why in sorted(similar)[:4])
                if similar else " or anything similar – enter the properties and save them to your library."))
        try:
            from fea_core.fatigue import pragtic
            _, exact, _ = pragtic.find(text, sn_only=True)
            rows.append(f"Fatigue data (PragTic): {len(exact)} records with S-N – Fatigue step.")
        except FileNotFoundError:
            pass
        self.mat_conv.setText("<br>".join(rows))
        self.mat_conv.setVisible(True)

    def on_mat_parts_sel(self):
        # valid rows only: when the model is replaced the list briefly keeps the rows of the previous model
        rows = [i.row() for i in self.mat_parts.selectedIndexes() if i.row() < len(self.parts)]
        if STAGES[getattr(self, "stage", 0)] == "Materials":
            self.view.set_selected_parts(rows)
        if rows:
            m = self.study.material_of(self.parts[rows[0]].name)
            i = self.mat_combo.findText(m.name)
            if i >= 0:
                self.mat_combo.setCurrentIndex(i)

    def on_parts_picked(self, idx):
        """Click on parts in 3D in part-selection mode: in the Materials step (and 3D printing) selects the parts in the list."""
        names = [self.parts[i].name for i in idx if 0 <= i < len(self.parts)]
        self.st_info.setText(("Selected parts: " + ", ".join(names)) if names else "")
        stage = STAGES[getattr(self, "stage", 0)]
        if stage == "Model":
            if hasattr(self, "model_table"):
                self.model_table.blockSignals(True)
                self.model_table.clearSelection()
                for i in idx:
                    if i < self.model_table.rowCount():
                        self.model_table.selectRow(i)
                self.model_table.blockSignals(False)
                if idx and idx[0] < self.model_table.rowCount():
                    it = self.model_table.item(idx[0], 0)
                    if it is not None:
                        self.model_table.scrollToItem(it)
            return
        if stage == "Materials":
            lw, after = self.mat_parts, self.on_mat_parts_sel
        elif stage == "3D printing":
            lw, after = self.print_panel.parts, self.print_panel.on_part_sel
        else:
            return
        lw.blockSignals(True)
        lw.clearSelection()
        for i in idx:
            it = lw.item(i)
            if it is not None:
                it.setSelected(True)
        lw.blockSignals(False)
        if idx and lw.item(idx[0]) is not None:
            lw.scrollToItem(lw.item(idx[0]))
        after()

    def material_from_form(self, name: str) -> Material:
        """Material from the form of the Materials step (print properties are taken from the selected material; an orthotropic
        material has E = E1, ν = ν12).
        """
        ortho, ocs = self.ortho_box.values()
        E, nu = (ortho["E1"], ortho["nu12"]) if ortho else (self.m_E.value(), self.m_nu.value())
        return Material(name, E, nu, self.m_rho.value() * 1e-12,
                        Re=self.m_Re.value() or None, Rm=self.m_Rm.value() or None, source=self.m_src.text(),
                        k=self.m_k.value() or None, alpha=(self.m_alpha.value() * 1e-6) or None,
                        cp=self.m_cp.value() or None, ortho=ortho, ortho_csys=ocs, plastic=self.plasticity_box.values(),
                        **self._mat_print)

    def save_user_material(self):
        name = self.m_name.text().strip()
        if not name:
            msg(self, APP_NAME, "Enter a material name.")
            return
        if name in {m.name for m in matlib.LIBRARY}:
            msg(self, APP_NAME, "A built-in material cannot be overwritten – choose a different name.")
            return
        try:
            m = self.material_from_form(name)
            matlib.save_user_material(m)
        except ValueError as e:                  # unreadable plasticity table
            msg(self, APP_NAME, str(e))
            return
        except matlib.UserLibraryError as e:          # an unreadable library file is not overwritten
            msg(self, APP_NAME, str(e), QMessageBox.Critical)
            return
        self.log_msg(f"Material '{name}' saved to {matlib.user_library_path()}")
        self.refresh_materials(select=name)

    # ================================================================== hole selection, section
    def select_holes(self):
        from fea_core.geometry import coaxial_cylinders
        sel = self.view.selected_faces()
        if not sel:
            msg(self, APP_NAME, "Select the cylindrical face of a hole or pin.")
            return
        out = list(sel)
        for pi, fi in sel:
            for j in coaxial_cylinders(self.view.part_faces[pi], fi):
                if (pi, j) not in out:
                    out.append((pi, j))
        self.view.set_selected_faces(out, emit=True)
        self.log_msg(f"Selection extended to {len(out)} faces (coaxial cylindrical faces).")

    # ================================================================== selection filter, edges
    def set_pick_filter(self, key: str):
        """Selection filter in 3D: Face (P) / Body (B) / Hole (H) / Edge (E). Hole also extends the already selected
        cylindrical faces to whole holes (previously the "Whole hole" button).
        """
        a = self.filter_actions.get(key)
        if a is not None and not a.isChecked():
            a.setChecked(True)
        self.view.set_filter(key)
        if hasattr(self, "pick_button"):
            self._sync_pick_button(key)
        if key != "edge":
            self.on_edge_picked(None)
        if key == "hole" and self.view.mode == "faces":
            from fea_core.geometry import cylinder_axis
            sel = self.view.selected_faces()
            if any(cylinder_axis(self.view.part_faces[pi][fi]) is not None for pi, fi in sel):
                self.select_holes()
        txt = next(t for k, t, _, _ in PICK_FILTER_ACTIONS if k == key)
        self.statusBar().showMessage(f"Selection filter: {txt}", 3000)

    @staticmethod
    def _pt(p, dec=6) -> str:
        return "(" + "; ".join(fmt(float(c) + 0.0, dec) for c in p) + ")"

    def describe_edge(self, d) -> str:
        part = self.parts[d["part"]].name if 0 <= d.get("part", -1) < len(self.parts) else ""
        s = f"Edge {d['edge'] + 1} of part {part}: {d['label']}, length {fmt(d['length'], 6)} mm"
        if d["type"] == "circle":
            s += (f", R {fmt(d['radius'], 6)} mm (Ø {fmt(2 * d['radius'], 6)} mm), center {self._pt(d['center'])} mm, axis {self._pt(np.round(d['axis'], 6), 4)}")
        elif d["type"] == "ellipse":
            s += f", semi-axes {fmt(d['major'], 6)} / {fmt(d['minor'], 6)} mm, center {self._pt(d['center'])} mm"
        else:
            s += f", edge center {self._pt(d['mid'])} mm"
            if d["type"] == "line":
                s += f" (from {self._pt(d['start'])} to {self._pt(d['end'])})"
        return s

    def on_edge_picked(self, info):
        self._edge_info = info
        self.btn_edge_point.setVisible(info is not None)
        if info is None and self.st_info.text().startswith("Edge "):
            self.st_info.setText("")
        if info is not None:
            self.btn_edge_point.setText("Use circle center" if "center" in info else "Use edge center")
            self.st_info.setText(self.describe_edge(info))
            self.log_msg(self.describe_edge(info))

    def register_point_input(self, spins, auto=None):
        """Point field [mm] (three QDoubleSpinBox) for "Use centre" of an edge; auto = check box of the automatic point
        ("face centre"), which is cleared when a point is inserted. The first visible field is used (open form).
        """
        self._point_inputs.append((list(spins), auto))

    def use_edge_point(self):
        """Centre of a circle / edge to the clipboard and into the point field of the open form (remote load / constraint)."""
        d = self._edge_info
        if d is None:
            return
        p = [float(c) + 0.0 for c in d["point"]]
        txt = "; ".join(fmt(c, 6) for c in p)
        QApplication.clipboard().setText(txt)
        what = "Circle center" if "center" in d else "Edge center"
        target = next(((s, a) for s, a in self._point_inputs if all(w.isVisible() for w in s)), None)
        if target is None:
            self.statusBar().showMessage(f"{what} ({txt}) mm copied to the clipboard (no form with a point – remote load / support – is open).", 8000)
            return
        spins, auto = target
        if auto is not None:
            auto.setChecked(False)
        for s, c in zip(spins, p):
            s.setValue(c)
        self.statusBar().showMessage(f"{what} ({txt}) mm inserted into the point field and copied to the clipboard.", 8000)

    SEC_FACE = 4                           # the item "face" in sec_axis

    def _sec_range(self, n):
        bb = np.array([bbox(p.shape) for p in self.parts])
        lo3, hi3 = bb[:, :3].min(axis=0), bb[:, 3:].max(axis=0)
        corners = np.array([[x, y, z] for x in (lo3[0], hi3[0]) for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])])
        proj = corners @ n
        return float(proj.min()), float(proj.max())

    def section_plane(self):
        """(point, normal) of the section plane – the side (x − o)·n ≤ 0 is shown; None = section off."""
        k = self.sec_axis.currentIndex()
        if k <= 0 or not self.parts:
            return None
        if k == self.SEC_FACE:
            if self._sec_face is None:
                return None
            n = np.asarray(self._sec_face[1], float)
        else:
            n = np.zeros(3)
            n[k - 1] = 1.0
        lo, hi = self._sec_range(n)
        s = lo + (hi - lo) * self.sec_pos.value() / 1000.0
        return n * s, (-n if self.sec_flip.isChecked() else n)

    def on_sec_axis(self, k):
        """Section by the plane of a face: normal and position from the first selected face (the slider then moves it)."""
        if k == self.SEC_FACE:
            sel = self.view.selected_faces()
            if not sel:
                msg(self, APP_NAME, "For a section along a face plane, first select a face (e.g. in the Supports step), then choose “face”.")
                self.sec_axis.setCurrentIndex(0)
                return
            p = self.face_prop(*sel[0])
            n = np.asarray(p["normal"], float)
            if np.linalg.norm(n) < 1e-12:
                return
            n /= np.linalg.norm(n)
            self._sec_face = (np.asarray(p["point"], float), n)
            lo, hi = self._sec_range(n)
            self.sec_pos.blockSignals(True)
            self.sec_pos.setValue(int(round(1000 * (float(self._sec_face[0] @ n) - lo) / max(hi - lo, 1e-12))))
            self.sec_pos.blockSignals(False)
        self.update_section()

    def update_section(self, *_):
        if STAGES[getattr(self, "stage", 0)] == "Results" and self.results is not None:
            self.view.clear_section()       # results: section through the mesh in numpy (fea_core.section), not OCCT
            if getattr(self, "_sec_timer", None) is None:   # the slider: redraw only after a short pause
                self._sec_timer = QTimer(self)
                self._sec_timer.setSingleShot(True)
                self._sec_timer.timeout.connect(self.show_result)
            self._sec_timer.start(120)
            return
        self.apply_model_section()

    def apply_model_section(self):
        plane = self.section_plane()
        if plane is None:
            self.view.clear_section()
        else:                                    # OCCT keeps the side the plane normal points to
            self.view.set_section_plane(plane[0], -plane[1])

    def assign_material(self, all_parts):
        if self.study is None:
            return
        rows = list(range(len(self.parts))) if all_parts else [i.row() for i in self.mat_parts.selectedIndexes()]
        if not rows and not (all_parts and self.study.frame.active or all_parts and not self.parts):
            msg(self, APP_NAME, "Select parts in the list.")
            return
        name = self.m_name.text().strip() or "Material"
        try:
            m = self.material_from_form(name)
        except ValueError as e:                  # unreadable plasticity table
            msg(self, APP_NAME, str(e))
            return
        self.study.materials[name] = m
        for r in rows:
            self.study.part_materials[self.parts[r].name] = name
        if all_parts:
            self.study.default_material = name
        # study materials are not deleted (as in the 3D printing step): an unassigned one can be a calibrated printed material,
        # a material of print variants (printing.variants) or have overrides of print properties (printing.materials);
        # the report states only the materials used
        self._mat_select = name                  # the assigned material stays selected in the list and in the form
        self.changed()

    # ================================================================== constraints
    def add_fixed(self):
        if self.study is not None and self.study.shell.enabled and self.view.selected_edges() and \
                not self.view.selected_faces():           # shells: clamping of the edge of a plate part / sheet
            name = self.new_entity_selection("Fixed support", "edges")
            self.study.constraints.append(Fixed(name))
            self.view.clear_selection()
            self.changed()
            return
        sels = self.need_faces()
        if sels:
            name = self.new_selection("Fixed support", sels)
            self.study.constraints.append(Fixed(name))
            self.view.set_selected_faces([])
            self.changed()

    def add_displacement(self):
        sels = self.need_faces()
        if not sels:
            return
        vals = [v.value() if c.isChecked() else None for c, v in zip(self.d_chk, self.d_val)]
        if all(v is None for v in vals):
            msg(self, APP_NAME, "Check at least one displacement component.")
            return
        name = self.new_selection("Displacement", sels)
        self.study.constraints.append(Displacement(name, *vals))
        self.view.set_selected_faces([])
        self.changed()

    def on_con_row(self, row):
        if 0 <= row < len(self.study.constraints):
            c = self.study.constraints[row]
            self.view.set_selected_faces(self.sel_faces.get(c.selection, []))
            if isinstance(c, Displacement):
                for chk, val, v in zip(self.d_chk, self.d_val, (c.ux, c.uy, c.uz)):
                    chk.setChecked(v is not None)
                    val.setValue(v or 0.0)
            elif isinstance(c, CylindricalSupport):
                for chk, v in zip(self.cyl_chk, (c.radial, c.tangential, c.axial)):
                    chk.setChecked(v)
            elif isinstance(c, RemoteSupport):
                self.rs_group.setChecked(True)
                for s, v in zip(self.rs_pt, c.point):
                    s.setValue(float(v))
                for chk, v in zip(self.rs_chk, c.dofs):
                    chk.setChecked(bool(v))
                self.rs_coupling.setCurrentIndex(0 if c.coupling == "rigid" else 1)
            elif isinstance(c, ElasticSupport):
                self.el_group.setChecked(True)
                self.el_k.setValue(float(c.stiffness))
                self.el_unit.setCurrentIndex(0 if c.per_area else 1)
                self.el_dir.setCurrentIndex(max(0, self.el_dir.findData(c.direction)))
            self.update_status()

    def cylinder_faces(self, sels) -> bool:
        """Cylindrical constraint / bearing force only on cylindrical faces (hole, pin)."""
        if any(s.type != "cylinder" for s in sels):
            msg(self, APP_NAME, "Select only the cylindrical faces of the hole or pin (the Hole selection filter, H, selects the whole hole).")
            return False
        return True

    def normal_force_dir(self, sels):
        """Unit outward normal for "Force normal to the face" (area-weighted average); only planar faces, otherwise a
        message and None (the same check when adding and when editing a load).
        """
        n = np.sum([np.asarray(s.normal, float) * s.area for s in sels], axis=0) if sels else np.zeros(3)
        if np.linalg.norm(n) < 1e-12 or any(s.type != "plane" for s in sels):
            msg(self, APP_NAME, "A force normal to the face can only be applied to planar faces with the same normal.")
            return None
        return n / np.linalg.norm(n)

    def add_cylindrical(self):
        sels = self.need_faces()
        if not sels or not self.cylinder_faces(sels):
            return
        flags = [c.isChecked() for c in self.cyl_chk]
        if not any(flags):
            msg(self, APP_NAME, "Check at least one constrained component.")
            return
        name = self.new_selection("Cylindrical support", sels)
        self.study.constraints.append(CylindricalSupport(name, *flags))
        self.view.set_selected_faces([])
        self.changed()

    def update_cylindrical(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or \
                not isinstance(self.study.constraints[row], CylindricalSupport):
            msg(self, APP_NAME, "Select a cylindrical support in the list.")
            return
        flags = [c.isChecked() for c in self.cyl_chk]
        if not any(flags):
            msg(self, APP_NAME, "Check at least one constrained component.")
            return
        c = self.study.constraints[row]
        c.radial, c.tangential, c.axial = flags
        self.changed()
        self.con_list.setCurrentRow(row)

    def update_displacement(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or not isinstance(self.study.constraints[row], Displacement):
            msg(self, APP_NAME, "Select a prescribed displacement in the list.")
            return
        vals = [v.value() if c.isChecked() else None for c, v in zip(self.d_chk, self.d_val)]
        if all(v is None for v in vals):
            msg(self, APP_NAME, "Check at least one displacement component.")
            return
        c = self.study.constraints[row]
        c.ux, c.uy, c.uz = vals
        self.changed()
        self.con_list.setCurrentRow(row)

    def remote_point_from_selection(self):
        """Point of the remote constraint = centroid of the selected faces (weighted by area)."""
        sels = self.need_faces()
        if not sels:
            return
        w = np.array([s.area for s in sels])
        p = (np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        for s, v in zip(self.rs_pt, p):
            s.setValue(float(v))

    def remote_form(self) -> tuple[list[float], list[bool], str] | None:
        dofs = [c.isChecked() for c in self.rs_chk]
        if not any(dofs):
            msg(self, APP_NAME, "Check at least one constrained component (ux … rz).")
            return None
        return ([s.value() for s in self.rs_pt], dofs,
                "distributing" if self.rs_coupling.currentIndex() == 1 else "rigid")

    def add_remote_support(self):
        sels = self.need_faces()
        if not sels:
            return
        form = self.remote_form()
        if form is None:
            return
        name = self.new_selection("Remote support", sels)
        self.study.constraints.append(RemoteSupport(name, *form))
        self.view.set_selected_faces([])
        self.changed()

    def update_remote_support(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or not isinstance(self.study.constraints[row], RemoteSupport):
            msg(self, APP_NAME, "Select a remote support in the list.")
            return
        form = self.remote_form()
        if form is None:
            return
        c = self.study.constraints[row]
        c.point, c.dofs, c.coupling = form
        self.changed()
        self.con_list.setCurrentRow(row)

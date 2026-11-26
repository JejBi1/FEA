"""Loads of the main window: the panel Loads, the load cases, the load forms and the add / update / delete of loads and list items.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QListWidget, QMessageBox,
                               QPushButton, QWidget)

from fea_core.study import Force, Pressure, Gravity, RemoteLoad, LoadCase, BearingLoad, ThermalLoad, Rotation
from fea_core.i18n import tr

from . import theme
from .widgets import Section, check_item, checked, dspin, help_tip, panel_header, fit_rows, set_checks, set_placeholder

from .main_defs import (APP_NAME, STAGE, BEARING, THERMAL_LOAD, ROTATION, FORCE, FRAME_NODE, FRAME_MEMBER, P_MODES, LOAD_TIPS,
                        VEC_TIPS, LOAD_SHORT, VEC_SHORT, G_STD, fmt, form_hide_empty_rows, msg_via_main as msg)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class LoadsMixin(_Base):
    """Loads of the main window: the panel Loads, the load cases, the load forms and the add / update / delete of loads and
    list items."""

    # ------------------------------------------------------------------ Loads panel
    def _panel_loads(self, lay):
        lay.addWidget(panel_header("Loads", "Load case = one analysis step; loads act on the selected faces",
                                   STAGE["Loads"] + 1))
        h = QHBoxLayout()
        self.lc_combo = QComboBox()
        self.lc_combo.currentIndexChanged.connect(lambda _: self.refresh_loads())
        help_tip(self.lc_combo, "Load case the loads are added to (each = one analysis)",
                 "The load case the loads are added to. Each case = a separate analysis (separate results); fatigue combines the cases according to the load history.")
        h.addWidget(self.lc_combo, 1)
        for txt, slot, tip in (("+", self.add_lc, "New load case"),
                               ("Rename", self.rename_lc, "Rename the selected load case"),
                               ("−", self.delete_lc, "Delete the selected load case with its loads")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        lay.addLayout(h)
        h_tools = QHBoxLayout()
        b_comb = QPushButton("Combinations (EN 1990)...")
        b_comb.setToolTip("Load case combinations and envelopes per ČSN EN 1990 (A2)")
        b_comb.clicked.connect(self.open_combinations_dialog)
        h_tools.addWidget(b_comb)
        b_imp = QPushButton("Imprint on face...")
        b_imp.setToolTip("Split a face for a load / support / contact by a geometric imprint (A10)")
        b_imp.clicked.connect(self.open_imprint_dialog)
        h_tools.addWidget(b_imp)
        lay.addLayout(h_tools)
        self.load_list = QListWidget()
        self.load_list.currentRowChanged.connect(self.on_load_row)
        help_tip(self.load_list, "Loads of the selected case – a click highlights the faces and loads the values",
                 "Loads of the selected case. A click on a row highlights the faces and loads the values into the form (“Save changes of the selected” overwrites them).")
        set_placeholder(self.load_list, "No loads yet – select faces and fill in the form")
        fit_rows(self.load_list, 3, 8)
        lay.addWidget(self.load_list)
        g = Section("New load", key="ld.new", tip="Type, values and target of the load (faces / edges / points)",
                    help="Load form: type, values and target (selected faces / edges / points).<br>“Add from selection” adds it to the load case chosen above.")
        f = QFormLayout(g.body)
        self.ld_type = QComboBox()
        self.ld_type.addItems([FORCE, "Force normal to face", "Pressure", "Moment (remote point)",
                               "Force at remote point", "Bearing load (hole / pin)", "Gravity", ROTATION,
                               THERMAL_LOAD, FRAME_NODE, FRAME_MEMBER])
        self.ld_type.currentIndexChanged.connect(self.on_ld_type)
        f.addRow("Type", self.ld_type)
        self.ld_target = QLabel("")
        self.ld_target.setWordWrap(True)
        self.ld_target.setToolTip("What the force acts on, based on the current 3D selection and selection filter (Face / Edge / Point)")
        self.ld_target_lbl = QLabel("Acts on")
        f.addRow(self.ld_target_lbl, self.ld_target)
        # frame loads (fea_core.beams.model.FrameNodeLoad / FrameMemberLoad)
        self.ld_fr_node = QComboBox()
        self.ld_fr_node.setToolTip("Frame node where the force and moment act (or click a node in 3D)")
        self.ld_fr_node_lbl = QLabel("Node")
        f.addRow(self.ld_fr_node_lbl, self.ld_fr_node)
        self.ld_fr_member = QComboBox()
        self.ld_fr_member.setToolTip("Member with distributed load (or click a member in 3D)")
        self.ld_fr_member_lbl = QLabel("Member")
        f.addRow(self.ld_fr_member_lbl, self.ld_fr_member)
        self.ld_fr_axes = QComboBox()
        self.ld_fr_axes.addItem("global axes (x, y, z)", "global")
        self.ld_fr_axes.addItem("member local axes (x along, y, z of cross-section)", "local")
        help_tip(self.ld_fr_axes, "Distributed load components in global axes, or in the member's local axes",
                 "Distributed load components in the global axes (gravity, snow, wind) or in the member's local axes (x = along the axis, z = the “up” axis of the section)")
        self.ld_fr_axes_lbl = QLabel("Axes")
        f.addRow(self.ld_fr_axes_lbl, self.ld_fr_axes)
        # local coordinate system of the components (force, remote load, bearing force, gravity)
        hc = QHBoxLayout()
        self.ld_csys = QComboBox()
        help_tip(self.ld_csys, "Component coordinate system: global, or local (Cartesian / cylindrical)",
                 "Component coordinate system: the global model axes, or a local system (Cartesian / cylindrical).<br>Cylindrical (r, θ, z) for forces only – the direction changes over the face / edge, the entered force is the sum of the magnitudes (the tangential component around the shaft end face = torque ⅔·F_θ·R).")
        self.ld_csys.currentIndexChanged.connect(lambda _: self.on_ld_type(0))
        hc.addWidget(self.ld_csys, 1)
        b = QPushButton("Systems…")
        b.setToolTip("Create / edit local coordinate systems")
        b.clicked.connect(self.edit_csys)
        hc.addWidget(b)
        self.ld_csys_btn = b
        self.ld_csys_lbl = QLabel("Coord. system")
        f.addRow(self.ld_csys_lbl, hc)
        self.ld_vec = [dspin(0.0, -1e9, 1e9, 3) for _ in range(3)]
        hv = QHBoxLayout()
        for s in self.ld_vec:
            hv.addWidget(s)
        self.ld_vec_lbl = QLabel("F [N]")
        f.addRow(self.ld_vec_lbl, hv)
        self.ld_val = dspin(0.0, -1e9, 1e9, 3)
        self.ld_val_lbl = QLabel("Value")
        f.addRow(self.ld_val_lbl, self.ld_val)
        hm = QHBoxLayout()
        self.ld_fr_mom = [dspin(0.0, -1e12, 1e12, 3) for _ in range(3)]
        for sp_, ax in zip(self.ld_fr_mom, "xyz"):
            sp_.setToolTip(f"{ax} component of the nodal moment [N·mm] (right-handed about the global axis)")
            hm.addWidget(sp_)
        self.ld_fr_mom_lbl = QLabel("M [N·mm]")
        f.addRow(self.ld_fr_mom_lbl, hm)
        hq = QHBoxLayout()
        self.ld_fr_var = QCheckBox("linearly to")
        self.ld_fr_var.setToolTip("Linearly varying load: q at the member start (fields above) → q at the end (these fields)")
        hq.addWidget(self.ld_fr_var)
        self.ld_fr_qend = [dspin(0.0, -1e9, 1e9, 4) for _ in range(3)]
        for sp_, ax in zip(self.ld_fr_qend, "xyz"):
            sp_.setToolTip(f"{ax} component of the distributed load at the member end [N/mm]")
            hq.addWidget(sp_)
        self.ld_fr_qend_lbl = QLabel("q at the end")
        f.addRow(self.ld_fr_qend_lbl, hq)
        # variable pressure: linear p = p0 + g·(x − origin), hydrostatic p = p0 + ρ·g·depth below the level
        self.ld_p_mode = QComboBox()
        self.ld_p_mode.addItems(P_MODES)
        help_tip(self.ld_p_mode, "Pressure distribution: constant, linear or hydrostatic",
                 "Constant pressure, linearly varying p = p₀ + gradient·(x − origin), or hydrostatic liquid pressure p = p₀ + ρ·g·depth below the surface (p₀ above the surface).")
        self.ld_p_mode.currentIndexChanged.connect(lambda _: self.on_ld_type(0))
        self.ld_p_mode_lbl = QLabel("Pressure profile")
        f.addRow(self.ld_p_mode_lbl, self.ld_p_mode)

        def vec_row(label, vals, dec=6, rng=1e7):
            sp = [dspin(v, -rng, rng, dec) for v in vals]
            h = QHBoxLayout()
            for x in sp:
                h.addWidget(x)
            lbl = QLabel(label)
            f.addRow(lbl, h)
            return sp, lbl
        self.ld_p_grad, self.ld_p_grad_lbl = vec_row("Gradient [MPa/mm]", (0, 0, 0), 8, 1e6)
        self.ld_p_org, self.ld_p_org_lbl = vec_row("Origin [mm]", (0, 0, 0), 3)
        for s, ax in zip(self.ld_p_grad, "xyz"):
            s.setToolTip(f"Pressure change per 1 mm in direction {ax} [MPa/mm]: p = p₀ + gradient·(x − origin)")
        for s, ax in zip(self.ld_p_org, "xyz"):
            s.setToolTip(f"{ax} coordinate [mm] of the point where pressure p₀ applies (for hydrostatic pressure, a point on the surface)")
        self.register_point_input(self.ld_p_org)
        self.ld_p_rho = dspin(1000.0, 0.0, 1e6, 2, suffix=" kg/m³")
        self.ld_p_rho.setToolTip("Liquid density (water 1000, oil 850–900 kg/m³); g = 9.81 m/s²")
        self.ld_p_rho_lbl = QLabel("Liquid ρ")
        f.addRow(self.ld_p_rho_lbl, self.ld_p_rho)
        self.ld_p_gdir, self.ld_p_gdir_lbl = vec_row("Gravity direction", (0, 0, -1), 4, 1e3)
        # rotation: axis, speed, angular acceleration, rotating parts
        self.ld_rot_pt, self.ld_rot_pt_lbl = vec_row("Axis point", (0, 0, 0), 3)
        self.register_point_input(self.ld_rot_pt)
        self.ld_rot_ax, self.ld_rot_ax_lbl = vec_row("Axis direction", (0, 0, 1), 4, 1e3)
        for s, ax in zip(self.ld_p_gdir, "xyz"):
            s.setToolTip(f"{ax} component of the gravity direction (downward; length is irrelevant) – pressure increases with depth in this direction")
        for s, ax in zip(self.ld_rot_pt, "xyz"):
            s.setToolTip(f"{ax} coordinate [mm] of any point on the rotation axis")
        for s, ax in zip(self.ld_rot_ax, "xyz"):
            s.setToolTip(f"{ax} component of the rotation axis direction (length is irrelevant; positive rotation is right-handed about it)")
        hr = QHBoxLayout()
        self.ld_rot_rpm = dspin(0.0, -1e7, 1e7, 2, suffix=" ot/min")
        self.ld_rot_rpm.setToolTip("Speed [rpm] – centrifugal force ρ·ω²·r (ω = 2π·n/60)")
        self.ld_rot_alpha = dspin(0.0, -1e9, 1e9, 3, suffix=" rad/s²")
        self.ld_rot_alpha.setToolTip("Angular acceleration (positive = accelerating about the axis direction by the right-hand rule)")
        hr.addWidget(self.ld_rot_rpm)
        self.ld_rot_alpha_lbl = QLabel("α")
        hr.addWidget(self.ld_rot_alpha_lbl)
        hr.addWidget(self.ld_rot_alpha)
        b = QPushButton("Axis from selection")
        b.setToolTip("Axis from the selected cylindrical face or circle (Edge filter)")
        b.clicked.connect(self.rotation_axis_from_selection)
        hr.addWidget(b)
        self.ld_rot_axis_btn = b
        self.ld_rot_w_lbl = QLabel("Speed")
        f.addRow(self.ld_rot_w_lbl, hr)
        self.ld_rot_parts = QListWidget()
        self.ld_rot_parts.setMaximumHeight(90)
        self.ld_rot_parts.setToolTip("The checked parts rotate (nothing checked = all parts)")
        self.ld_rot_parts_lbl = QLabel("Rotates")
        f.addRow(self.ld_rot_parts_lbl, self.ld_rot_parts)
        self.ld_pt = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        hp = QHBoxLayout()
        for s, ax in zip(self.ld_pt, "xyz"):
            s.setToolTip(f"{ax} coordinate of the remote point [mm] where the force / moment acts")
            hp.addWidget(s)
        self.ld_pt_auto = QCheckBox("face center")
        self.ld_pt_auto.setChecked(True)
        self.ld_pt_auto.setToolTip("Point = centroid of the selected faces (uncheck for a custom point, e.g. the end of a lever)")
        hp.addWidget(self.ld_pt_auto)
        self.ld_pt_lbl = QLabel("Point [mm]")
        f.addRow(self.ld_pt_lbl, hp)
        self.register_point_input(self.ld_pt, self.ld_pt_auto)
        self.ld_coupling = QComboBox()
        self.ld_coupling.addItems(["distributing (does not add stiffness)", "rigid"])
        help_tip(self.ld_coupling, "Distributing = the face deforms, rigid = the face acts as a rigid body",
                 "How the point acts on the faces: distributing = forces distributed by area, the face deforms freely; rigid = the face moves as a rigid body (adds stiffness).")
        self.ld_coupling_lbl = QLabel("Constraint")
        f.addRow(self.ld_coupling_lbl, self.ld_coupling)
        # thermal load: field from a thermal state (for a transient one at a time) or uniform temperature of the parts
        self.ld_th_src = QComboBox()
        help_tip(self.ld_th_src, "Temperature from a thermal case (Heat step), or a uniform part temperature",
                 "Temperature field from a thermal case (Heat step), or a uniform temperature of the selected parts.<br>Stress from ΔT = T − reference temperature (Heat step).")
        self.ld_th_src.currentIndexChanged.connect(lambda _: self.on_ld_type(0))
        self.ld_th_src_lbl = QLabel("Temperature from")
        f.addRow(self.ld_th_src_lbl, self.ld_th_src)
        self.ld_th_time_w = QWidget()
        ht = QHBoxLayout(self.ld_th_time_w)
        ht.setContentsMargins(0, 0, 0, 0)
        self.ld_th_time = dspin(0.0, 0.0, 1e9, 3, suffix=" s")
        self.ld_th_time.setToolTip("Time of the transient thermal case whose temperature field is used (linear between snapshots)")
        self.ld_th_end = QCheckBox("end of the state")
        self.ld_th_end.setToolTip("Use the temperature field at the end of the transient case")
        self.ld_th_end.setChecked(True)
        self.ld_th_end.toggled.connect(lambda on: self.ld_th_time.setEnabled(not on))
        self.ld_th_time.setEnabled(False)
        ht.addWidget(self.ld_th_time, 1)
        ht.addWidget(self.ld_th_end)
        self.ld_th_time_lbl = QLabel("Time")
        f.addRow(self.ld_th_time_lbl, self.ld_th_time_w)
        self.ld_th_parts = QListWidget()
        self.ld_th_parts.setMaximumHeight(90)
        self.ld_th_parts.setToolTip("The checked parts have the specified temperature (nothing checked = all parts)")
        self.ld_th_parts_lbl = QLabel("Parts")
        f.addRow(self.ld_th_parts_lbl, self.ld_th_parts)
        self._ld_form = f
        hb = QHBoxLayout()
        b = QPushButton("Add from selection")
        b.setToolTip("Add the load from the form to the selected case on the selected faces")
        b.setWhatsThis("<b>Add load</b><p>1. Select the load case above. 2. Select faces by clicking in 3D (edges with the E filter, points with the V filter). 3. Choose the load type and fill in the values. 4. Press “Add from selection”. Arrows in 3D show the direction.</p><p>A force also works on edges and points; gravity, rotation and temperature need no selection.</p>")
        self.btn_add_load = b
        theme.set_primary(b)
        b.clicked.connect(self.add_load)
        hb.addWidget(b)
        b = QPushButton("Save changes to selected")
        b.setToolTip("Overwrite the values of the load selected in the list (faces stay)")
        b.clicked.connect(self.update_load)
        hb.addWidget(b)
        f.addRow(hb)
        lay.addWidget(g)
        h = QHBoxLayout()
        b = QPushButton("Faces from selection")
        help_tip(b, "Replace the faces of the selected load with the current selection",
                 "Replace the faces (edges / points for an edge / point force) of the selected load with the current selection")
        b.clicked.connect(lambda: self.update_faces_of(self.load_list, "load"))
        h.addWidget(b)
        b = QPushButton("Delete")
        b.setToolTip("Delete the load selected in the list (Ctrl+Z to undo)")
        b.clicked.connect(lambda: self.delete_item(self.load_list, "load"))
        h.addWidget(b)
        lay.addLayout(h)
        self.on_ld_type(0)

    # ================================================================== loads
    def current_lc(self) -> LoadCase | None:
        i = self.lc_combo.currentIndex()
        if self.study and 0 <= i < len(self.study.load_cases):
            return self.study.load_cases[i]
        return None

    def lc_name_taken(self, name: str, own=None) -> bool:
        """The name of the load state is already used by another state (also "Bolt preload" from the results)."""
        from fea_core.joints.model import PRELOAD_CASE
        if name == PRELOAD_CASE or any(lc.name == name and lc is not own for lc in self.study.load_cases):
            msg(self, APP_NAME, f"Load case '{name}' already exists – choose a different name.")
            return True
        return False

    def add_lc(self):
        if self.study is None:
            return
        k = len(self.study.load_cases) + 1
        while any(lc.name == f"Load case {k}" for lc in self.study.load_cases):
            k += 1
        name, ok = QInputDialog.getText(self, APP_NAME, "Load case name:", text=f"Load case {k}")
        if ok and name.strip() and not self.lc_name_taken(name.strip()):
            self.study.load_cases.append(LoadCase(name.strip()))
            self.changed()
            self.lc_combo.setCurrentIndex(len(self.study.load_cases) - 1)

    def rename_lc(self):
        lc = self.current_lc()
        if lc is None:
            return
        name, ok = QInputDialog.getText(self, APP_NAME, "New name:", text=lc.name)
        if ok and name.strip():
            self.set_lc_name(lc, name.strip())

    def set_lc_name(self, lc: LoadCase, name: str):
        """Renames the load state including the references (fatigue channels, long-term creep states of 3D printing)."""
        if name == lc.name or self.lc_name_taken(name, lc):
            return
        old = lc.name
        st = self.study
        if st.fatigue is not None:
            for ch in st.fatigue.history.channels:
                if ch.load_case == old:
                    ch.load_case = name
        cr = st.printing.creep
        cr.cases = [name if c == old else c for c in cr.cases]
        lc.name = name
        self.changed()

    def ask(self, text: str) -> bool:
        """Yes / No question (default No). GUI tests replace it (w.ask = lambda *_: True)."""
        return QMessageBox.question(self, APP_NAME, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def delete_lc(self):
        lc = self.current_lc()
        if lc is None:
            return
        st = self.study
        n = len(lc.loads)
        if not self.ask(f"Delete load case '{lc.name}'"
                        + (f" including its loads ({n})" if n else "") + "?\nReferences to it (fatigue channels, long-term creep states) are removed as well.\nYou can undo it with Undo (Ctrl+Z)."):
            return
        for l in lc.loads:
            self._drop_selection(getattr(l, "selection", None))
        st.load_cases.remove(lc)
        dropped = []
        if st.fatigue is not None:                  # fatigue and creep channels referring to a deleted state
            n = len(st.fatigue.history.channels)
            st.fatigue.history.channels = [ch for ch in st.fatigue.history.channels if ch.load_case != lc.name]
            if n != len(st.fatigue.history.channels):
                dropped.append(f"fatigue channels: {n - len(st.fatigue.history.channels)}")
        cr = st.printing.creep
        if lc.name in cr.cases:
            cr.cases = [c for c in cr.cases if c != lc.name]
            dropped.append("long-term creep case")
        if dropped:
            self.log_msg(f"Load case '{lc.name}' and its references deleted ({', '.join(dropped)})")
        self.changed()

    def on_ld_type(self, i):
        t = self.ld_type.currentText()
        th = t == THERMAL_LOAD
        th_case = self.ld_th_src.currentData() if th else None
        uniform = th and th_case is None
        transient = False
        if th_case and self.study is not None:
            c = next((c for c in self.study.thermal.cases if c.name == th_case), None)
            transient = c is not None and c.transient
        frn, frm = t == FRAME_NODE, t == FRAME_MEMBER
        for w in (self.ld_fr_node, self.ld_fr_node_lbl, self.ld_fr_mom_lbl, *self.ld_fr_mom):
            w.setVisible(frn)
        for w in (self.ld_fr_member, self.ld_fr_member_lbl, self.ld_fr_axes, self.ld_fr_axes_lbl, self.ld_fr_var,
                  self.ld_fr_qend_lbl, *self.ld_fr_qend):
            w.setVisible(frm)
        vec = t in (FORCE, "Moment (remote point)", "Force at remote point", "Gravity", BEARING, FRAME_NODE,
                    FRAME_MEMBER)
        val = t in ("Force normal to face", "Pressure") or uniform
        pt = t in ("Moment (remote point)", "Force at remote point")
        pmode = self.ld_p_mode.currentIndex() if t == "Pressure" else -1
        rot = t == ROTATION
        csys_ok = t in (FORCE, "Moment (remote point)", "Force at remote point", "Gravity", BEARING)
        for w in (self.ld_csys, self.ld_csys_btn, self.ld_csys_lbl):
            w.setVisible(csys_ok)
        target = self.load_target() if t == FORCE else None
        for w in (self.ld_target, self.ld_target_lbl):
            w.setVisible(t == FORCE)
        if t == FORCE:
            n = {"faces": len(self.view.selected_faces()), "edges": len(self.view.selected_edges()),
                 "points": len(self.view.selected_points())}
            self.ld_target.setText({"faces": f"faces ({n['faces']}) – uniformly by area",
                                    "edges": f"edges ({n['edges']}) – uniformly by length (Edge filter)",
                                    "points": f"points ({n['points']}) – each F/n (Point filter)"}.get(
                target, "nothing – select faces, edges (Edge filter) or points (Point filter)"))
        for w in (self.ld_p_mode, self.ld_p_mode_lbl):
            w.setVisible(t == "Pressure")
        for w in self.ld_p_grad + [self.ld_p_grad_lbl]:
            w.setVisible(pmode == 1)
        for w in self.ld_p_org + [self.ld_p_org_lbl]:
            w.setVisible(pmode in (1, 2))
        for w in self.ld_p_gdir + [self.ld_p_gdir_lbl, self.ld_p_rho, self.ld_p_rho_lbl]:
            w.setVisible(pmode == 2)
        self.ld_p_org_lbl.setText("Liquid level – point [mm]" if pmode == 2 else "Origin [mm]")
        for w in (self.ld_rot_pt + self.ld_rot_ax + [self.ld_rot_pt_lbl, self.ld_rot_ax_lbl, self.ld_rot_w_lbl,
                                                    self.ld_rot_rpm, self.ld_rot_alpha, self.ld_rot_alpha_lbl,
                                                    self.ld_rot_axis_btn, self.ld_rot_parts,
                                                    self.ld_rot_parts_lbl]):
            w.setVisible(rot)
        for w in self.ld_vec + [self.ld_vec_lbl]:
            w.setVisible(vec)
        for w in (self.ld_val, self.ld_val_lbl):
            w.setVisible(val)
        for w in self.ld_pt + [self.ld_pt_lbl, self.ld_pt_auto, self.ld_coupling, self.ld_coupling_lbl]:
            w.setVisible(pt)
        for w in (self.ld_th_src, self.ld_th_src_lbl):
            w.setVisible(th)
        for w in (self.ld_th_time_w, self.ld_th_time_lbl):
            w.setVisible(th and transient)
        for w in (self.ld_th_parts, self.ld_th_parts_lbl):
            w.setVisible(uniform)
        form_hide_empty_rows(self._ld_form)
        cyl = False
        cs_name = self.ld_csys.currentData()
        if cs_name and self.study is not None and cs_name in self.study.coordinate_systems:
            cyl = self.study.coordinate_systems[cs_name].kind == "cylindrical"
        comp = "F_r, F_θ, F_z" if cyl else "F"
        self.ld_vec_lbl.setText({"Moment (remote point)": "M [N·mm]", "Gravity": "g [mm/s²]",
                                 FRAME_MEMBER: "q [N/mm]"}.get(t, f"{comp} [N]"))
        self.ld_val_lbl.setText(("p [MPa]", "p₀ at origin [MPa]", "p nad hladinou [MPa]")[max(pmode, 0)]
                                if t == "Pressure" else "T [°C]" if th else "F [N]")
        self.ld_val.setToolTip("Positive pressure acts into the material" if t == "Pressure" else
                               "Uniform part temperature (stress from ΔT = T − reference temperature)" if th else
                               "Positive force pushes into the face (against the outward normal)")
        prev = getattr(self, "_ld_prev_type", None)
        self._ld_prev_type = t
        if t != prev:
            # change of type: a residual force from the form must not stay as g [mm/s²] (−200 → g = −200) and vice versa
            # g as force [N]; when an existing load is loaded into the form (fill_load_form) g is not filled in –
            # its vector is written right afterwards
            if t == "Gravity" and not getattr(self, "_ld_filling", False):
                for s, v in zip(self.ld_vec, (0.0, 0.0, -G_STD)):
                    s.setValue(v)
            elif prev == "Gravity":
                for s in self.ld_vec:
                    s.setValue(0.0)
        help_tip(self.ld_type, LOAD_SHORT.get(t, "Load type"),
                 tr(LOAD_TIPS.get(t, "")).replace(".\n", ".<br>").replace("\n", " "))
        tip = VEC_TIPS.get(t, "Force components [N] in the axes of the chosen coordinate system (x, y, z; cylindrical r, θ, z)")
        short = VEC_SHORT.get(t, "{} component of the force [N] in the axes of the selected coordinate system")
        for s, ax in zip(self.ld_vec, "xyz"):
            help_tip(s, short.format(ax), f"{tip}<br>Field: component {ax}")

    def form_csys(self, cylindrical_ok: bool) -> tuple[bool, str | None]:
        """(OK?, system name from the form); cylindrical only for a force."""
        name = self.ld_csys.currentData() or None
        if name and not cylindrical_ok and self.study.coordinate_systems.get(name) is not None \
                and self.study.coordinate_systems[name].kind == "cylindrical":
            msg(self, APP_NAME, "A cylindrical coordinate system is only available for a force (the direction would otherwise depend on the point) – choose a Cartesian system or global axes.")
            return False, None
        return True, name

    def pressure_from_form(self, sel: str) -> Pressure:
        """Pressure from the form: constant / linear / hydrostatic."""
        p0 = self.ld_val.value()
        mode = self.ld_p_mode.currentIndex()
        org = [s.value() for s in self.ld_p_org]
        if mode == 1:
            g = [s.value() for s in self.ld_p_grad]
            return Pressure(sel, p0, g, org) if any(g) else Pressure(sel, p0)
        if mode == 2:
            d = np.array([s.value() for s in self.ld_p_gdir])
            if np.linalg.norm(d) == 0 or self.ld_p_rho.value() <= 0:
                raise ValueError("Hydrostatic pressure: enter a nonzero gravity direction and a positive liquid density.")
            g = d / np.linalg.norm(d) * self.ld_p_rho.value() * 1e-12 * G_STD
            return Pressure(sel, p0, g.tolist(), org, clip=True)
        return Pressure(sel, p0)

    def rotation_from_form(self) -> Rotation:
        ax = np.array([s.value() for s in self.ld_rot_ax])
        if np.linalg.norm(ax) == 0:
            raise ValueError("Rotation: the axis direction must not be zero.")
        w = self.ld_rot_rpm.value() * 2 * math.pi / 60
        if w == 0 and self.ld_rot_alpha.value() == 0:
            raise ValueError("Rotation: enter the speed or angular acceleration.")
        return Rotation([s.value() for s in self.ld_rot_pt], (ax / np.linalg.norm(ax)).tolist(), w,
                        self.ld_rot_alpha.value(), checked(self.ld_rot_parts))

    def rotation_axis_from_selection(self):
        """Rotation axis from the selected circle (filter Edge) or a cylindrical face."""
        from fea_core.geometry import cylinder_axis
        o = d = None
        e = self.view.selected_edge()
        if e is not None and "center" in e and "axis" in e:
            o, d = e["center"], e["axis"]
        else:
            for pi, fi in self.view.selected_faces():
                ax = cylinder_axis(self.view.part_faces[pi][fi])
                if ax is not None:
                    o, d = ax[0], ax[1]
                    break
        if o is None:
            msg(self, APP_NAME, "Select a cylindrical face (shaft, hole) or circle (Edge filter).")
            return
        for sp, v in zip(self.ld_rot_pt, o):
            sp.setValue(float(v) + 0.0)
        for sp, v in zip(self.ld_rot_ax, d):
            sp.setValue(round(float(v), 9) + 0.0)

    def edit_csys(self):
        if self.study is None:
            return
        from .csys_dialog import CsysDialog
        CsysDialog(self).exec()
        self.refresh_csys_combo()

    def refresh_csys_combo(self):
        cur = self.ld_csys.currentData()
        self.ld_csys.blockSignals(True)
        self.ld_csys.clear()
        self.ld_csys.addItem("global axes (x, y, z)", None)
        if self.study is not None:
            for k, c in self.study.coordinate_systems.items():
                self.ld_csys.addItem(f"{k} ({'cylindrical r, θ, z' if c.kind == 'cylindrical' else 'Cartesian'})", k)
        i = self.ld_csys.findData(cur) if cur else 0
        self.ld_csys.setCurrentIndex(max(i, 0))
        self.ld_csys.blockSignals(False)
        self.ortho_box.refresh_csys()               # axes of the orthotropic material (Materials step)

    def add_load(self):
        lc = self.current_lc()
        if lc is None:
            msg(self, APP_NAME, "Create a load case first.")
            return
        t = self.ld_type.currentText()
        vec = [s.value() for s in self.ld_vec]
        ok, csys = self.form_csys(t == FORCE)
        if not ok:
            return
        if t == "Gravity":
            lc.loads.append(Gravity(vec, csys))
            self.changed()
            return
        if t in (FRAME_NODE, FRAME_MEMBER):
            l = self.frame_load_from_form()
            if l is not None:
                lc.loads.append(l)
                self.changed()
            return
        if t == THERMAL_LOAD:
            l = self.thermal_load_from_form()
            if l is not None:
                lc.loads.append(l)
                self.changed()
            return
        if t == ROTATION:
            try:
                l = self.rotation_from_form()
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
            lc.loads = [x for x in lc.loads if not isinstance(x, Rotation)] + [l]   # one rotation per state
            self.changed()
            return
        if t == FORCE and self.load_target() in ("edges", "points"):
            target = self.load_target()
            name = self.new_entity_selection("Force on edge" if target == "edges" else "Force at point", target)
            lc.loads.append(Force(name, vec, csys))
            self.view.clear_selection()
            self.changed()
            if any(isinstance(l, Force) and self.study.selections.get(l.selection) is not None
                   and self.study.selections[l.selection].entity in ("edges", "points") for l in lc.loads):
                self.log_msg("A force on an edge / at a point is singular in a solid model – stress near it is not valid (grows with mesh refinement); evaluate away from the point of application.")
            return
        sels = self.need_faces()
        if not sels:
            return
        if t == FORCE:
            name = self.new_selection("Force", sels)
            lc.loads.append(Force(name, vec, csys))
        elif t == "Force normal to face":
            n = self.normal_force_dir(sels)
            if n is None:
                return
            name = self.new_selection("Force", sels)
            lc.loads.append(Force(name, (-self.ld_val.value() * n).tolist()))
        elif t == "Pressure":
            try:
                self.pressure_from_form("")
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
            name = self.new_selection("Pressure", sels)
            lc.loads.append(self.pressure_from_form(name))
        elif t == BEARING:
            if not self.cylinder_faces(sels):
                return
            name = self.new_selection("Bearing load", sels)
            lc.loads.append(BearingLoad(name, vec, csys))
        else:
            name = self.new_selection("Moment" if t.startswith("Moment") else "Remote force", sels)
            lc.loads.append(self.remote_load_from_form(name, sels, vec, csys))
        self.view.clear_selection()
        self.changed()

    def remote_load_from_form(self, name, sels, vec, csys) -> RemoteLoad:
        """Remote force / moment from the form; automatic point = centroid of the faces (weighted by area)."""
        if self.ld_pt_auto.isChecked():
            w = np.array([s.area for s in sels])
            pt = ((np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / w.sum()).tolist()
        else:
            pt = [s.value() for s in self.ld_pt]
        coupling = "rigid" if self.ld_coupling.currentIndex() == 1 else "distributing"
        what = "moment" if self.ld_type.currentText().startswith("Moment") else "force"
        return RemoteLoad(name, pt, coupling=coupling, csys=csys, **{what: vec})

    def frame_load_from_form(self):
        """Frame load from the form (force / moment at a node, distributed load of a member)."""
        from fea_core.beams.model import FrameMemberLoad, FrameNodeLoad
        vec = [s.value() for s in self.ld_vec]
        if self.ld_type.currentText() == FRAME_NODE:
            node = self.ld_fr_node.currentText()
            if not node:
                msg(self, APP_NAME, "The study has no frame nodes (Members step).")
                return None
            mom = [s.value() for s in self.ld_fr_mom]
            if not any(vec) and not any(mom):
                msg(self, APP_NAME, "Enter a force or moment.")
                return None
            return FrameNodeLoad(node, vec, mom)
        member = self.ld_fr_member.currentText()
        if not member:
            msg(self, APP_NAME, "The study has no members (Members step).")
            return None
        q_end = [s.value() for s in self.ld_fr_qend] if self.ld_fr_var.isChecked() else None
        if not any(vec) and not (q_end and any(q_end)):
            msg(self, APP_NAME, "Enter the distributed load q [N/mm].")
            return None
        return FrameMemberLoad(member, vec, q_end, self.ld_fr_axes.currentData())

    def refresh_frame_load_form(self):
        fr = self.study.frame
        for cb, names in ((self.ld_fr_node, list(fr.nodes)), (self.ld_fr_member, list(fr.members))):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            if cur in names:
                cb.setCurrentText(cur)
            cb.blockSignals(False)

    def thermal_load_from_form(self) -> ThermalLoad | None:
        """Thermal load from the form: field of a thermal state (time for a transient one), or uniform temperature of the parts."""
        case = self.ld_th_src.currentData()
        if case:
            c = next((c for c in self.study.thermal.cases if c.name == case), None)
            t = None
            if c is not None and c.transient and not self.ld_th_end.isChecked():
                t = self.ld_th_time.value()
                if t > c.time:
                    msg(self, APP_NAME, f"Time {fmt(t)} s is past the end of the transient case ({fmt(c.time)} s).")
                    return None
            return ThermalLoad(case=case, time=t)
        return ThermalLoad(value=self.ld_val.value(), parts=checked(self.ld_th_parts))

    def refresh_thermal_load_form(self):
        """Menu of thermal states and the part list for the thermal load (after a change of the study)."""
        cur = self.ld_th_src.currentData()
        cur_text = self.ld_th_src.currentText()
        self.ld_th_src.blockSignals(True)
        self.ld_th_src.clear()
        for c in self.study.thermal.cases:
            self.ld_th_src.addItem(f"thermal case: {c.name}", c.name)
        self.ld_th_src.addItem("uniform part temperature", None)
        i = self.ld_th_src.findData(cur) if cur else self.ld_th_src.findText(cur_text)
        self.ld_th_src.setCurrentIndex(max(i, 0))
        self.ld_th_src.blockSignals(False)
        for lst in (self.ld_th_parts, self.ld_rot_parts):          # model parts, the tick stays
            on = set(checked(lst))
            lst.clear()
            for p in self.parts:
                check_item(lst, p.name, p.name in on)
        self.refresh_csys_combo()
        self.refresh_frame_load_form()
        self.on_ld_type(0)

    def on_load_row(self, row):
        lc = self.current_lc()
        if lc and 0 <= row < len(lc.loads):
            l = lc.loads[row]
            sel = getattr(l, "selection", None)
            self.view.set_selected_faces(self.sel_faces.get(sel, []) if sel else [])
            self.view.set_selected_edges(self.sel_edges.get(sel, []) if sel else [])
            self.view.set_selected_points(self.sel_points.get(sel, []) if sel else [])
            self.fill_load_form(l)
            self.update_status()

    def fill_load_form(self, l):
        self._ld_filling = True                  # change of type from an existing load – do not overwrite its vector
        try:
            self._fill_load_form(l)
        finally:
            self._ld_filling = False

    def _fill_load_form(self, l):
        def setvec(v, spins=None):
            for s, x in zip(spins or self.ld_vec, v):
                s.setValue(float(x))
        if hasattr(l, "csys"):
            self.ld_csys.setCurrentIndex(max(self.ld_csys.findData(l.csys), 0) if l.csys else 0)
        k = getattr(l, "kind", "")
        if k == "frame_node":
            self.ld_type.setCurrentText(FRAME_NODE)
            self.ld_fr_node.setCurrentText(l.node)
            setvec(l.force)
            setvec(l.moment, self.ld_fr_mom)
            return
        if k == "frame_member":
            self.ld_type.setCurrentText(FRAME_MEMBER)
            self.ld_fr_member.setCurrentText(l.member)
            setvec(l.q)
            self.ld_fr_var.setChecked(l.q_end is not None)
            setvec(l.q_end or [0, 0, 0], self.ld_fr_qend)
            self.ld_fr_axes.setCurrentIndex(max(0, self.ld_fr_axes.findData(l.axes)))
            return
        if isinstance(l, Force):
            self.ld_type.setCurrentText(FORCE)
            setvec(l.vector)
        elif isinstance(l, Pressure):
            self.ld_type.setCurrentText("Pressure")
            self.ld_val.setValue(l.value)
            if l.variable and l.clip:
                g = np.asarray(l.gradient, float)
                self.ld_p_mode.setCurrentIndex(2)
                self.ld_p_rho.setValue(float(np.linalg.norm(g)) / (1e-12 * G_STD))
                setvec(g / np.linalg.norm(g), self.ld_p_gdir)
            elif l.variable:
                self.ld_p_mode.setCurrentIndex(1)
                setvec(l.gradient, self.ld_p_grad)
            else:
                self.ld_p_mode.setCurrentIndex(0)
            if l.variable:
                setvec(l.origin or [0, 0, 0], self.ld_p_org)
        elif isinstance(l, Rotation):
            self.ld_type.setCurrentText(ROTATION)
            setvec(l.point, self.ld_rot_pt)
            setvec(l.axis, self.ld_rot_ax)
            self.ld_rot_rpm.setValue(l.omega * 60 / (2 * math.pi))
            self.ld_rot_alpha.setValue(l.alpha)
            set_checks(self.ld_rot_parts, l.parts)
        elif isinstance(l, BearingLoad):
            self.ld_type.setCurrentText(BEARING)
            setvec(l.vector)
        elif isinstance(l, Gravity):
            self.ld_type.setCurrentText("Gravity")
            setvec(l.vector)
        elif isinstance(l, ThermalLoad):
            self.ld_type.setCurrentText(THERMAL_LOAD)
            i = self.ld_th_src.findData(l.case) if l.case else self.ld_th_src.count() - 1
            self.ld_th_src.setCurrentIndex(max(i, 0))
            self.ld_th_end.setChecked(l.time is None)
            self.ld_th_time.setValue(l.time or 0.0)
            if l.value is not None:
                self.ld_val.setValue(l.value)
            set_checks(self.ld_th_parts, l.parts)
            self.on_ld_type(0)
        elif isinstance(l, RemoteLoad):
            moment = any(l.moment)
            self.ld_type.setCurrentText("Moment (remote point)" if moment else "Force at remote point")
            setvec(l.moment if moment else l.force)
            self.ld_pt_auto.setChecked(False)
            for s, x in zip(self.ld_pt, l.point):
                s.setValue(float(x))
            self.ld_coupling.setCurrentIndex(1 if l.coupling == "rigid" else 0)

    def update_load(self):
        lc = self.current_lc()
        row = self.load_list.currentRow()
        if lc is None or not (0 <= row < len(lc.loads)):
            msg(self, APP_NAME, "Select a load in the list.")
            return
        l = lc.loads[row]
        t = self.ld_type.currentText()
        vec = [s.value() for s in self.ld_vec]
        sel = getattr(l, "selection", None)
        ok, csys = self.form_csys(t == FORCE)
        if not ok:
            return
        entity = self.study.selections[sel].entity if sel and sel in self.study.selections else "faces"
        if t == "Gravity":
            new = Gravity(vec, csys)
        elif t in (FRAME_NODE, FRAME_MEMBER):
            new = self.frame_load_from_form()
            if new is None:
                return
        elif t == THERMAL_LOAD:
            new = self.thermal_load_from_form()
            if new is None:
                return
        elif t == ROTATION:
            try:
                new = self.rotation_from_form()
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
        elif sel is None:
            msg(self, APP_NAME, "Gravity, rotation and temperature can only be changed to a load without faces.")
            return
        elif entity in ("edges", "points") and t != FORCE:
            msg(self, APP_NAME, "Only a force (components) can be applied to an edge / point.")
            return
        elif t == FORCE:
            new = Force(sel, vec, csys)
        elif t == "Force normal to face":
            sels = self.study.selections[sel].faces
            n = self.normal_force_dir(sels)
            if n is None:
                return
            new = Force(sel, (-self.ld_val.value() * n).tolist())
        elif t == "Pressure":
            try:
                new = self.pressure_from_form(sel)
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
        elif t == BEARING:
            if not self.cylinder_faces(self.study.selections[sel].faces):
                return
            new = BearingLoad(sel, vec, csys)
        else:
            new = self.remote_load_from_form(sel, self.study.selections[sel].faces, vec, csys)
        if sel and getattr(new, "selection", None) != sel:
            # a load on faces converted to gravity / rotation / temperature: its selection is removed, unless something else uses it
            # (constraint, other load, thermal condition, mesh refinement)
            self._drop_selection(sel)
        lc.loads[row] = new
        self.changed()
        self.load_list.setCurrentRow(row)

    # ================================================================== shared list edits
    def _drop_selection(self, name):
        if not name:
            return
        used = [c.selection for c in self.study.constraints]
        used += [getattr(l, "selection", None) for lc in self.study.load_cases for l in lc.loads]
        used += [getattr(bc, "selection", None) for c in self.study.thermal.cases for bc in c.loads]
        used += [r.get("selection") for r in self.study.mesh.local]
        used += [x for c in self.study.contacts for x in (c.a, c.b)]                 # F8: kontakty, hmota, sondy
        used += [m.selection for m in self.study.masses]
        used += [p.selection for p in self.study.dynamics.probes]
        used += [mo.target for s in self.study.analysis.sequence for mo in s.prescribed]
        mech = self.study.mechanism                                                    # F8: mechanismus
        used += [x for j in mech.joints for x in (j.selection_a, j.selection_b)]
        used += [x for f in mech.forces for x in (f.selection_a, f.selection_b)]
        used += [ld.selection for ld in mech.loads]
        if used.count(name) <= 1:
            self.study.selections.pop(name, None)
            self.sel_faces.pop(name, None)
            self.sel_edges.pop(name, None)
            self.sel_points.pop(name, None)

    def _items(self, kind):
        if kind == "constraint":
            return self.study.constraints
        if kind == "load":
            lc = self.current_lc()
            return lc.loads if lc else []
        if kind == "thermal":
            c = self.thermal_panel.current_case()
            return c.loads if c else []
        return self.study.mesh.local

    def delete_item(self, lw, kind):
        if self.study is None:
            return
        row = lw.currentRow()
        items = self._items(kind)
        if not (0 <= row < len(items)):
            return
        it = items[row]
        sel = it.get("selection") if isinstance(it, dict) else getattr(it, "selection", None)
        self._drop_selection(sel)
        items.pop(row)
        self.changed(invalidate_mesh=(kind == "refine"))

    def update_faces_of(self, lw, kind):
        if self.study is None:
            return
        row = lw.currentRow()
        items = self._items(kind)
        if not (0 <= row < len(items)):
            msg(self, APP_NAME, "Select an item in the list.")
            return
        it = items[row]
        name = it.get("selection") if isinstance(it, dict) else getattr(it, "selection", None)
        if not name:
            msg(self, APP_NAME, "The selected item has no faces (gravity, rotation, temperature, heat generation in a part, sphere refinement).")
            return
        entity = self.study.selections[name].entity if name in self.study.selections else "faces"
        if entity in ("edges", "points"):                 # force on an edge / to a point: edges / points from the selection
            new = self.selected_edge_selectors() if entity == "edges" else self.selected_point_selectors()
            if not new:
                msg(self, APP_NAME, "Select edges in 3D (Edge filter)." if entity == "edges" else
                    "Select points in 3D (Point filter).")
                return
            setattr(self.study.selections[name], entity, new)
            if entity == "edges":
                self.sel_edges[name] = self.view.selected_edges()
            else:
                self.sel_points[name] = self.view.selected_points()
            self.changed(invalidate_mesh=True)
            return
        sels = self.need_faces()
        if not sels:
            return
        self.study.selections[name].faces = sels
        self.sel_faces[name] = self.view.selected_faces()
        self.changed(invalidate_mesh=True)

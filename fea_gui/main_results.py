"""Results of the main window: the panel Results, the choice of the displayed field, state, combination and frame, and the display of
static, dynamic, frame, 2D and shell results.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QSlider, QSpinBox, QVBoxLayout, QWidget, QFrame, QScrollArea, QTabWidget)

from fea_core.gravity_states import str_missing_by_superposition
from fea_core.i18n import tr

from .safe_browser import SafeTextBrowser
from . import dyn_results as dynr, frame_results as frs, twod_gui, shell_gui, submodel_dialog as smd
from .eigen_gui import MODE_QUANTITIES, ModesTable, mode_info_html, mode_quantity, mode_quantities
from .extras import open_space_dialog
from .legend import Legend
from .widgets import help_tip, panel_header, table, report_soft

from .main_defs import (STAGES, STAGE, STRESS_ERROR_TIP, QUANTITIES, PEEQ_Q, TOPO_Q, TOPO_SHAPE_Q, TOPO_MISSING, CONTACT_Q,
                        STATUS_NOTE, THERMAL_QUANTITIES, SAFETY_QTY, NO_MIN_QTY, fmt, NearestPoints)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class ResultsMixin(_Base):
    """Results of the main window: the panel Results, the choice of the displayed field, state, combination and frame, and the
    display of static, dynamic, frame, 2D and shell results."""

    # ------------------------------------------------------------------ Results panel
    def _panel_results(self, lay):
        lay.addWidget(panel_header("Results", "Click in the model to show the value (probe)", STAGE["Results"] + 1))
        from .result_overview import ResultOverview   # verdict of safety against the requirement + numbers of the state
        self.res_overview = ResultOverview(self)
        self.res_overview.setVisible(False)
        lay.addWidget(self.res_overview)
        outer = lay                                 # content in the scroll area (legend + summary + maximum places)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout()
        self.r_step = QComboBox()
        self.r_step.setToolTip("Load case (or thermal case “Heat: …”) whose results are displayed")
        self.r_qty = QComboBox()
        help_tip(self.r_qty, "Displayed quantity: stress, displacement, safety factor, temperature…",
                 "Displayed quantity: von Mises stress, displacements, principal stress, stress components, safety factor to Re, for printing the Tsai-Wu/Hill safety factor, temperature, ccx stress error estimate;<br>for a mode / buckling shape the normalized displacement (max |u| = 1)")
        for name, unit in QUANTITIES:
            self.r_qty.addItem(f"{name} [{unit}]")
        self.r_scale = QComboBox()
        self.r_scale.addItems(["undeformed", "automatic", "1×", "10×", "100×", "1000×"])
        self.r_scale.setCurrentIndex(1)
        help_tip(self.r_scale, "Deformation scale in 3D (does not change the result values)",
                 "Deformation scale in 3D (automatic = the largest displacement scaled to 5 % of the model diagonal); the result values are not affected by the scale")
        self.r_bands = QSpinBox()
        self.r_bands.setRange(4, 16)
        self.r_bands.setValue(12)
        self.r_bands.setToolTip("Number of legend color bands")
        self.r_csys = QComboBox()
        self.r_csys.setToolTip("Local coordinate system for stresses and displacements (e.g. cylindrical system)")
        self.r_csys.addItem("Global Cartesian", None)
        self.r_csys_lbl = QLabel("System")
        self.r_edges = QCheckBox("element edges")
        self.r_edges.setToolTip("Show mesh element edges over the color map")
        self.r_outline = QCheckBox("undeformed model outline")
        self.r_outline.setToolTip("Show the undeformed model outline (for comparison with the deformed shape)")
        self.r_part = QComboBox()
        self.r_part.setToolTip("Show only the selected assembly part with recalculated color scale (Display Group, A5)")
        self.r_part.addItem("All parts", None)
        self.r_part.currentIndexChanged.connect(lambda _: self.show_result())
        self.r_time = QComboBox()                   # snapshot of a transient thermal state
        self.r_time.setToolTip("Snapshot time of the transient thermal case")
        self.r_time_lbl = QLabel("Time")
        for lbl, w in (("Load case", self.r_step), ("Assembly part", self.r_part), (self.r_time_lbl, self.r_time),
                       ("Quantity", self.r_qty), (self.r_csys_lbl, self.r_csys),
                       ("Deformation", self.r_scale), ("Number of bands", self.r_bands)):
            f.addRow(lbl, w)
        self.r_time.setVisible(False)
        self.r_time_lbl.setVisible(False)
        # dynamics (F8): frequency / time slider, phase of the harmonic response, absolute displacement (+ base motion)
        self.r_dyn = QWidget()
        hd = QHBoxLayout(self.r_dyn)
        hd.setContentsMargins(0, 0, 0, 0)
        self.r_dyn_slider = QSlider(Qt.Horizontal)
        help_tip(self.r_dyn_slider, "Frequency / time whose field is shown (arrows ← → step by one point)",
                 "Frequency of the harmonic response / time of the transient response whose field is shown (arrows ← → step by one point)")
        self.r_dyn_val = QLabel("")
        self.r_dyn_val.setMinimumWidth(90)
        self.r_phase = QSpinBox()
        self.r_phase.setRange(0, 360)
        self.r_phase.setSingleStep(15)
        self.r_phase.setSuffix(" °")
        self.r_phase.setWrapping(True)
        help_tip(self.r_phase, "Phase φ of the harmonic response for the “at phase φ” quantities",
                 "Phase φ of the harmonic response (the moment of the period ωt = φ) for the “at phase φ” quantities;<br>the amplitude over phase does not depend on the phase")
        self.r_abs = QCheckBox("absolute")
        help_tip(self.r_abs, "Absolute displacement = relative + base motion (base excitation only)",
                 "Absolute displacement = relative + base motion (base excitation only); otherwise displacement relative to the base")
        hd.addWidget(self.r_dyn_slider, 1)
        hd.addWidget(self.r_dyn_val)
        hd.addWidget(self.r_phase)
        hd.addWidget(self.r_abs)
        self.r_dyn_lbl = QLabel("Frequency")
        f.insertRow(2, self.r_dyn_lbl, self.r_dyn)
        self.r_dyn.setVisible(False)
        self.r_dyn_lbl.setVisible(False)
        self.r_dyn_slider.valueChanged.connect(lambda _: self.show_result())
        self.r_phase.valueChanged.connect(lambda _: self.show_result())
        self.r_abs.toggled.connect(lambda _: self.show_result())
        f.addRow(self.r_edges)
        f.addRow(self.r_outline)
        lay.addLayout(f)
        grid = QGridLayout()                      # result tool buttons in 2 columns (full labels)
        grid.setSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        def add_btn(btn):
            n = grid.count()
            grid.addWidget(btn, n // 2, n % 2)

        b = QPushButton("Line plot…")
        help_tip(b, "Values along a line: chart, line in 3D, stress linearization, CSV",
                 "Values of the quantity along a line between two points (click in the model or enter coordinates): chart, line in 3D, stress linearization through the wall (membrane + bending, ASME VIII-2) and CSV export")
        b.clicked.connect(self.open_path_dialog)
        self.btn_path = b
        add_btn(b)
        b = QPushButton("Mesh convergence…")
        help_tip(b, "Run on progressively finer meshes and check that the quantity settles",
                 "Run the study on progressively finer meshes (globally or in a sphere around a point) and check that the monitored quantity settles (Richardson extrapolation, GCI). The results of the main window do not change.")
        b.clicked.connect(self.open_convergence_dialog)
        self.btn_conv = b
        add_btn(b)
        b = QPushButton("Print detail (voxels)…")
        help_tip(b, "Local voxel model of the print beads at the critical spot (indicative)",
                 "Local voxel model of beads (perimeters, layers, infill) around the critical spot of a printed part – boundary displacements from this result, safety factor in the bead axes (INDICATIVE)")
        b.clicked.connect(self.open_voxel_dialog)
        self.btn_voxel = b
        add_btn(b)
        b = QPushButton("Submodel…")
        help_tip(b, "Finer local model of a region driven by this result (box or sphere)",
                 "Submodel: the parts are cut by a box or a sphere around the critical spot and meshed finer; the displacement of this result is prescribed on the cut faces, supports and loads inside the region are transferred. Cut check = submodel vs global stress at the cut.")
        b.clicked.connect(self.open_submodel_dialog)
        self.btn_submodel = b
        add_btn(b)
        b = QPushButton("Section force…")
        help_tip(b, "Resultant internal forces and moments in a cut (Free Body Cut)",
                 "Calculation of resultant internal forces and moments in a section (Free Body Cut, A3): N, Vy, Vz, Mx, My, Mz")
        b.clicked.connect(self.open_section_force_dialog)
        self.btn_sec_force = b
        add_btn(b)
        b = QPushButton("Export (VTU, CSV)...")
        b.setToolTip("Export results to ParaView (VTU/PVD) or CSV tables (nodal values and reactions, A6)")
        b.clicked.connect(self.open_export_dialog)
        add_btn(b)

        b_anim = QPushButton("Animate...")
        b_anim.setToolTip("Deformation animation (A5) - smooth cycle of deflection scaling")
        b_anim.clicked.connect(self.toggle_animation)
        self.btn_anim = b_anim
        add_btn(b_anim)

        b_iso = QPushButton("Isosurface...")
        b_iso.setToolTip("Extraction and export of an isosurface at the chosen value (marching tetrahedra, A5)")
        b_iso.clicked.connect(self.open_isosurface_dialog)
        self.btn_iso = b_iso
        add_btn(b_iso)

        b_vec = QPushButton("Stress vectors...")
        b_vec.setToolTip("Principal stress vectors σ1, σ2, σ3 and export (A5)")
        b_vec.clicked.connect(self.open_vectors_dialog)
        self.btn_vec = b_vec
        add_btn(b_vec)

        b = QPushButton("Margins of safety (ECSS)…")
        help_tip(b, "Space structures: requirement set, allowables, margins of safety outside singularities")
        b.clicked.connect(lambda: open_space_dialog(self))
        self.btn_space = b
        add_btn(b)
        b = QPushButton("Plasticity assessment…")
        help_tip(b, "PEEQ check per EN 1993-1-5 (5 %) and ASME VIII-2 local failure, residual stress")
        b.clicked.connect(self.open_plasticity_dialog)
        self.btn_plast = b
        add_btn(b)

        b_topo = QPushButton("Topology (BESO)...")
        b_topo.setToolTip("Topology optimization to lighten the structure for 3D printing (BESO, C4a)")
        b_topo.clicked.connect(self.open_topo_dialog)
        self.btn_topo = b_topo
        add_btn(b_topo)
        lay.addLayout(grid)
        self.r_step.currentIndexChanged.connect(lambda _: self.on_r_step())
        for w in (self.r_qty, self.r_scale, self.r_time, self.r_csys):
            w.currentIndexChanged.connect(lambda _: self.show_result())
        self.r_bands.valueChanged.connect(lambda _: self.show_result())
        self.r_edges.toggled.connect(lambda _: self.show_result())
        self.r_outline.toggled.connect(lambda _: self.show_result())
        self.legend = Legend()
        self.legend.setMinimumHeight(250)           # 16 bands fit, the results panel is tall
        self.legend.setWhatsThis("<b>Legend</b><p>Color bands of the displayed quantity from the minimum to the maximum of the displayed case.</p>")
        lay.addWidget(self.legend)
        self.res_info = SafeTextBrowser()
        self.res_info.setMinimumHeight(200)
        help_tip(self.res_info, "Summary: maxima, reactions, equilibrium, safety factor; a click in the model = probe",
                 "State summary: maxima, support reactions, equilibrium check, safety factor per part;<br>a click in the model gives the value at the point (probe)")
        # maximum places with singularity classification (same as MCP fea_results_hotspots and the report)
        self.hs_box = QWidget()
        v = QVBoxLayout(self.hs_box)
        v.setContentsMargins(2, 4, 2, 2)
        self.hs_title = QLabel("<b>Highest stresses</b>")
        v.addWidget(self.hs_title)
        self.hs_table = table(["#", "value", "part", "location [mm]", "classification"], 160)
        self.hs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.hs_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.hs_table.setWordWrap(True)             # long classification ("singularity – clamped edge") on 2 lines
        help_tip(self.hs_table, "Spots of the highest values with classification (singularity, boundary condition…)",
                 "Separate spots of the highest values (clusters of ~3 elements) with automatic classification:<br>singularity (support edge, sharp inner edge, point load, bolt) – do not evaluate by the nodal value; boundary condition – check the support; real spot – check by mesh convergence.<br>A click on a row marks the spot in 3D and shows the value (probe).")
        self.hs_table.currentCellChanged.connect(lambda r, *_: self.on_hotspot_row(r))
        v.addWidget(self.hs_table)
        self.hs_note = QLabel("")
        self.hs_note.setWordWrap(True)
        self.hs_note.setTextFormat(Qt.RichText)
        v.addWidget(self.hs_note)
        self.res_tabs = QTabWidget()                # state summary | maximum places (saves panel height)
        self.res_tabs.addTab(self.res_info, "Summary")
        self.res_tabs.addTab(self.hs_box, "Highest stresses")
        self.modes_table = ModesTable(self)          # natural frequency / buckling: frequencies and effective mass fractions
        self.res_tabs.addTab(self.modes_table, "Mode shapes")
        self.res_tabs.setTabToolTip(2, "Natural frequencies and buckling factors, effective mass fractions x/y/z and sums")
        self.res_tabs.setTabVisible(2, False)
        self.frame_box = frs.FrameResultsBox(self)    # frames: extremes per member and EN 1993 assessment
        self.res_tabs.addTab(self.frame_box, "Members")
        self.res_tabs.setTabToolTip(3, "Internal forces per member and EN 1993-1-1 check")
        self.res_tabs.setTabVisible(3, False)
        self.dyn_box = dynr.DynResultsBox(self)      # F8: dynamics plots (FRF, time history, PSD, SRS) and mode table
        self.TAB_DYN = self.res_tabs.addTab(self.dyn_box, "Dynamics")
        self.res_tabs.setTabToolTip(self.TAB_DYN, "Dynamics charts in probes (FRF, time history, response PSD, pulse SRS) and table of modes with effective mass")
        self.res_tabs.setTabVisible(self.TAB_DYN, False)
        self.contact_box = dynr.ContactResultsBox(self)   # F8: contact pair forces, force–displacement curve, partial result
        self.TAB_CONTACT = self.res_tabs.addTab(self.contact_box, "Contacts")
        self.res_tabs.setTabToolTip(self.TAB_CONTACT, "Contact pairs of the displayed case (forces, area, pressure, opening, slip) and force–travel curve")
        self.res_tabs.setTabVisible(self.TAB_CONTACT, False)
        lay.addWidget(self.res_tabs)
        self.r_scroll = QScrollArea()
        self.r_scroll.setWidgetResizable(True)
        self.r_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.r_scroll.setFrameShape(QFrame.NoFrame)
        self.r_scroll.setWidget(inner)
        outer.addWidget(self.r_scroll, 100)

    # ================================================================== results
    def _post_key(self) -> str:
        """Fingerprint of the study inputs of the post-processing (combinations, load-case names, Stress Tool settings, materials and
        their assignment) – key of the GUI caches on the results (the cached value is the same as a new computation)."""
        st = self.study
        return repr(([c.to_dict() for c in getattr(st, "combinations", [])], [lc.name for lc in st.load_cases],
                     st.assessment, sorted(st.materials.items()), sorted(st.part_materials.items()), st.default_material))

    def _safety_field(self, st, criterion):
        """Safety field of the Stress Tool (assessment.compute_safety_field, capped at 10), cached on the step per (criterion, inputs)
        – redraws (section, animation, colour bands) do not recompute it."""
        key = (criterion, self._post_key(), id(self.results.mesh))
        cache = st.__dict__.setdefault("_gui_safety", {})
        if key not in cache:
            from fea_core.assessment import compute_safety_field
            args = () if criterion is None else (criterion,)
            cache.clear()                              # one entry per step (inputs changed → the old one is useless)
            cache[key] = np.minimum(compute_safety_field(self.study, self.results.mesh, st, *args)[0], 10.0)
        return cache[key]

    def _combination(self, res, ci):
        """compute_combination for combination `ci` of the study, cached on the results (key = definitions of all combinations and the
        inputs of the safety envelopes; at most 4 entries – every combination holds U and S of the whole mesh)."""
        from fea_core.combinations import compute_combination
        comb = self.study.combinations[ci]
        key = (ci, self._post_key())
        cache = res.__dict__.setdefault("_gui_comb_cache", {})
        if key not in cache:
            val = compute_combination(self.study, res, comb)
            while len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = val
        return cache[key]

    def result_field(self, st, qi):
        name = QUANTITIES[qi][0]
        if name == "von Mises":
            return st.mises
        if name == "Total displacement":
            return st.umag
        # local coordinate system of the results (A4): displacement and stress components are transformed at each node
        cs_name = None
        if getattr(self, "r_csys", None) is not None and self.r_csys.currentIndex() > 0 and self.study:
            cs_name = self.r_csys.currentData()
            if cs_name not in self.study.coordinate_systems:
                cs_name = None
        comps = {"σx": 0, "σy": 1, "σz": 2, "τxy": 3, "τyz": 4, "τzx": 5}
        if cs_name is not None and (name in ("Displacement X", "Displacement Y", "Displacement Z") or name.removeprefix("Stress ") in comps):
            from fea_core import csys as _csys
            try:
                if name.startswith("Displacement "):
                    U_loc = _csys.transform_displacements(self.study, cs_name, st.U, self.results.mesh.nodes, "results")
                    return U_loc[:, "XYZ".index(name[-1])]
                S_loc = _csys.transform_stress(self.study, cs_name, st.S, self.results.mesh.nodes, "results")
                return S_loc[:, comps[name.split()[-1]]]
            except _csys.CsysError as e:           # e.g. a point on the axis of a cylindrical system → global components
                self.log_msg(f"Coordinate system “{cs_name}”: {e} – global components shown.")
        if name.startswith("Displacement "):
            return st.U[:, "XYZ".index(name[-1])]
        if name.startswith("1st principal"):
            return st.principal[:, 0]
        if name.startswith("3rd principal"):
            return st.principal[:, 2]
        if name.removeprefix("Stress ") in comps:
            return st.S[:, comps[name.split()[-1]]]

        if name == "Safety factor to Re":
            re_node = self.node_re()
            vm = np.maximum(st.mises, 1e-9)
            return np.minimum(re_node / vm, 10.0)
        if name == "Safety factor – Tresca":
            return self._safety_field(st, "tresca")
        if name == "Safety factor – Mohr-Coulomb":
            return self._safety_field(st, "mohr_coulomb")
        if name == "Safety factor (Stress Tool)":
            return self._safety_field(st, None)                   # criterion per material
        if name == "Stress error estimate":
            v = st.fields.get("stress_error")
            return np.full(len(st.U), np.nan) if v is None else np.asarray(v, float)
        if name.startswith("3D printing"):
            # non-printed parts: safety 10 (scale limit), index 0 – only printed parts are assessed
            key = "print_safety" if "safety factor" in name else "print_index"
            v = st.fields.get(key)
            if v is None:
                return np.full(len(st.U), np.nan)
            return np.minimum(np.nan_to_num(v, nan=10.0, posinf=10.0), 10.0) if key == "print_safety" \
                else np.nan_to_num(v, nan=0.0)
        if name == "Temperature":
            T = st.fields.get("T")
            return np.full(len(st.U), np.nan) if T is None else np.asarray(T, float)
        if name in CONTACT_Q:                         # contact (F8): pressure, status, opening, slip, friction stress
            v = st.fields.get(CONTACT_Q[name])
            return np.full(len(st.U), np.nan) if v is None else np.asarray(v, float)
        if name == PEEQ_Q:                            # plasticity: equivalent plastic strain (nodal, from ccx)
            v = st.fields.get("peeq")
            return np.full(len(st.U), np.nan) if v is None else np.maximum(np.asarray(v, float), 0.0)
        if name in TOPO_Q:                            # topology optimisation: nodal density 0 to 1
            v = st.fields.get("topo_density")
            return np.full(len(st.U), np.nan) if v is None else np.asarray(v, float)
        raise KeyError(name)

    @staticmethod
    def thermal_field(th, qi, frame):
        """Field of a thermal state; frame = index of the snapshot of a transient state (None = end). Flux only at the end."""
        name = THERMAL_QUANTITIES[qi][0]
        if name == "Temperature":
            return th.T if frame is None else th.frames[frame]
        if frame is not None:
            return None                       # the flux is stored only at the end of the state
        if name == "Heat flux |q|":
            return th.flux_mag
        return th.flux[:, "xyz".index(name[-1])]

    def current_result(self):
        """("step", i) | ("thermal", j) according to the choice in the Results step, None = nothing."""
        if self.results is None:
            return None
        d = self.r_step.currentData()
        if not d:
            return None
        kind, *rest = d.split(":")
        if kind == "mode":                          # ("mode", (set, shape)) – natural shape / buckling shape
            j, k = int(rest[0]), int(rest[1])
            ok = 0 <= j < len(self.results.modes) and 0 <= k < len(self.results.modes[j].values)
            return ("mode", (j, k)) if ok else None
        if kind == "dyn":                           # ("dyn", (druh, k)) – dynamika (dyn_results.QUANTS)
            ok = getattr(self.results, "dynamics", None) is not None and rest and rest[0] in dynr.QUANTS
            return ("dyn", (rest[0], int(rest[1]) if len(rest) > 1 else 0)) if ok else None
        if kind == "combination":
            i = int(rest[0])
            ok = self.study is not None and hasattr(self.study, "combinations") and 0 <= i < len(self.study.combinations)
            return ("combination", i) if ok else None
        if kind == "sub":                           # submodel (Results.submodels, submodel_dialog)
            return smd.current(self.results, rest)
        i = int(rest[0])
        r2 = self.results.twod
        n = (len(self.results.steps) if kind == "step" else len(self.results.thermal) if kind == "thermal" else
             len(r2.steps) if kind == "twod" and r2 is not None else
             len(self.results.shell.steps) if kind == "shell" and self.results.shell is not None else 0)
        return (kind, i) if 0 <= i < n else None

    def _mark_unavailable_quantities(self, cur, qlist) -> None:
        """The stress error estimate is computed only by CalculiX: for a result without it (the direct solver that the automatic choice
        uses for a linear study) the item is greyed out with the reason instead of offering an empty map."""
        names = [q[0] for q in qlist]
        res = self.results
        if "Stress error estimate" not in names or res is None:
            return
        if cur and cur[0] == "step" and 0 <= cur[1] < len(res.steps):
            has = "stress_error" in res.steps[cur[1]].fields
        else:
            has = any("stress_error" in s.fields for s in res.steps)
        item = self.r_qty.model().item(names.index("Stress error estimate"))
        if item is not None:
            item.setEnabled(has)
            if has:
                item.setToolTip("")
            elif str_missing_by_superposition(res.info, res.steps[cur[1]].name if cur and cur[0] == "step" and 0 <= cur[1] < len(res.steps) else None):
                item.setToolTip(tr("Not available for gravity cases that are combinations of unit states: the stress error estimate is not linear in the load."))
            else:
                item.setToolTip(tr("Not in this result: the stress error estimate is computed only by CalculiX. Choose the solver "
                                   "“Direct sparse solver – CalculiX” in the Solve step and solve again."))

    def result_qlist(self, cur=None) -> list[tuple[str, str]]:
        """Quantities of the menu by kind of result: bodies, frames (alone / mixed model), heat, modes, 2D."""
        cur = cur if cur is not None else self.current_result()
        kind = cur[0] if cur else "step"
        res = self.results
        solids = res is not None and len(res.mesh.nodes) > 0
        if kind == "thermal":
            return THERMAL_QUANTITIES
        if kind == "twod":
            return [(n, u) for n, u, _ in twod_gui.quantities(res.twod)]
        if kind == "shell":
            return [(n, u) for n, u, _, _ in shell_gui.SHELL_Q]
        if kind == "sub":
            return smd.QLIST
        if kind == "mode":
            return mode_quantities(res.modes[cur[1][0]]) if solids and cur else (MODE_QUANTITIES if solids
                                                                                  else [frs.MODE_FRAME_Q])
        if kind == "dyn":
            return dynr.QUANTS[cur[1][0]]
        has_frame = res is not None and any(st.frame for st in res.steps)
        if not solids and has_frame:
            return frs.FRAME_QUANTITIES
        return QUANTITIES + (frs.FRAME_QUANTITIES if has_frame else [])

    def select_result(self, data: str):
        """Selects a result in the menu of the Results step (data of the r_step item, e.g. "mode:0:2")."""
        if STAGES[self.stage] != "Results" and self.results is not None:
            self.set_stage(STAGE["Results"])
        k = self.r_step.findData(data)
        if k >= 0:
            self.r_step.setCurrentIndex(k)

    def deformation_scale(self, U) -> float:
        """Deformation scale by the choice "Deformation" (automatic = max. displacement at 5 % of the model diagonal)."""
        m = self.results.mesh
        umax = float(np.linalg.norm(U, axis=1).max()) if len(U) else 0.0
        scale = {0: 0.0, 2: 1.0, 3: 10.0, 4: 100.0, 5: 1000.0}.get(self.r_scale.currentIndex())
        if scale is None:
            diag = (float(np.linalg.norm(m.nodes.max(axis=0) - m.nodes.min(axis=0))) if len(m.nodes) else
                    self.view.model_size())                       # frame / 2D without a 3D mesh
            scale = 0.05 * diag / umax if umax > 0 else 0.0
        return scale

    def displayed(self, p) -> np.ndarray:
        """Point of the undeformed mesh in the position shown in 3D (displacement × scale of the displayed result)."""
        p = np.asarray(p, float)
        scale, U = self._r_view
        if not scale or U is None or self.results is None:
            return p
        cur = self.current_result()
        if cur and cur[0] == "twod":
            r2 = self.results.twod
            fr = r2.frame2d()
            try:
                return p + scale * fr.vec3d(r2.probe(fr.to2d(p), "U", cur[1]))
            except Exception as e:  # noqa: BLE001
                report_soft("results", self, "Displacement of a 2D point", e)
                return p
        try:
            from fea_core.results import tet10_shape
            e, xi = self.results.locate(p)
            conn = self.results.mesh.tets[e]
            return p + scale * (tet10_shape(xi) @ U[conn])
        except Exception as e:  # noqa: BLE001 – outside the mesh → no displacement
            report_soft("results", self, "Displacement of a point", e)
            return p

    def undeformed(self, x) -> np.ndarray:
        """Point clicked on the displayed (deformed) model → undeformed position (displacement of the nearest surface node)."""
        x = np.asarray(x, float)
        scale, U = self._r_view
        pr = getattr(self, "_probe", None)
        if not scale or U is None or pr is None:
            return x
        tree, ids = pr[0], pr[1]
        _, j = tree.query(x)
        n = int(ids[j])
        return x - scale * U[n] if n >= 0 else x

    def current_frame(self, th):
        """Index of the snapshot of the transient state selected in r_time; None = end of the state (also a steady one)."""
        if not th.transient or th.frames is None or not len(th.times):
            return None
        k = self.r_time.currentIndex()
        return None if k < 0 or k >= len(th.times) - 1 else k

    def on_r_step(self):
        """Change of the result: quantities by kind (mechanics / heat), times of the transient state, then display."""
        self._sync_result_controls()
        self.show_result()
        self.res_overview.refresh()                 # numbers of the displayed state on the Assessment card

    def _sync_result_controls(self):
        cur = self.current_result()
        kind = cur[0] if cur else "step"
        qlist = self.result_qlist(cur)
        key = (kind, tuple(q[0] for q in qlist))
        self.refresh_r_csys()
        is_step = kind == "step"
        self.r_csys.setVisible(is_step)
        self.r_csys_lbl.setVisible(is_step)
        if getattr(self, "_r_kind", None) != key:
            old = self.r_qty.currentText().split(" [")[0]
            self.r_qty.blockSignals(True)
            self.r_qty.clear()
            for name, unit in qlist:
                self.r_qty.addItem(f"{name} [{unit}]")
            names = [q[0] for q in qlist]
            self.r_qty.setCurrentIndex(names.index(old) if old in names else 0)
            self.r_qty.blockSignals(False)
            self._r_kind = key
        self._mark_unavailable_quantities(cur, qlist)
        th = self.results.thermal[cur[1]] if kind == "thermal" and cur else None
        transient = th is not None and th.transient and th.frames is not None and len(th.times) > 0
        self.r_time.blockSignals(True)
        if transient:
            items = [f"t = {fmt(float(t))} s" for t in th.times]
            if [self.r_time.itemText(i) for i in range(self.r_time.count())] != items:
                self.r_time.clear()
                self.r_time.addItems(items)
                self.r_time.setCurrentIndex(len(items) - 1)
        else:
            self.r_time.clear()
        self.r_time.blockSignals(False)
        self.r_time.setVisible(transient)
        self.r_time_lbl.setVisible(transient)
        self.r_scale.setEnabled(kind != "thermal")
        # dynamics: frequency slider (harmonic) / time (transient), phase, absolute displacement
        ax = dynr.axis(self.results.dynamics, cur[1][0]) if kind == "dyn" else None
        self.r_dyn.setVisible(ax is not None)
        self.r_dyn_lbl.setVisible(ax is not None)
        if ax is not None:
            vals, label, _unit = ax
            self.r_dyn_lbl.setText(label)
            s = self.r_dyn_slider
            if s.maximum() != len(vals) - 1 or getattr(self, "_dyn_axis", None) != cur[1][0]:
                s.blockSignals(True)
                s.setRange(0, len(vals) - 1)
                d = self.results.dynamics
                if cur[1][0] == "harm":                 # default = frequency of the stress maximum (envelope)
                    h = d.harmonic
                    s.setValue(int(np.asarray(h["env_S_at"])[int(np.argmax(h["env_S"]))]))
                else:
                    tr = d.transient
                    te = float(np.asarray(tr["t_env"])[np.asarray(tr["env_S_at"])[int(np.argmax(tr["env_S"]))]])
                    s.setValue(int(np.argmin(np.abs(vals - te))))
                s.blockSignals(False)
                self._dyn_axis = cur[1][0]
            self.r_phase.setVisible(cur[1][0] == "harm")
            self.r_abs.setVisible(dynr.has_base(self.results.dynamics, cur[1][0]))

    def node_re(self):
        m = self.results.mesh
        re_part = np.array([self.study.material_of(p).Re or np.inf for p in m.part_names])
        re_node = np.full(len(m.nodes), np.inf)
        for pi, re in enumerate(re_part):
            np.minimum.at(re_node, m.tets[m.tet_part == pi].ravel(), re)
        return re_node

    def show_result(self):
        if self.results is None or STAGES[self.stage] != "Results":
            return
        from .cyclic_gui import full_results          # cyclic symmetry: the sector shown as the full 360° model
        full = full_results(self)
        if full is None:
            return self._show_result()
        sector, self.results = self.results, full
        try:
            return self._show_result()
        finally:
            self.results = sector

    def _show_result(self):
        res = self.results
        self._probe = self._probe_exact = None
        if not getattr(self, "_anim_frame", False):   # the displayed result changed → cached animation frames are stale
            self.view.anim_reset()
            self._anim_state = {}
        cur = self.current_result()
        if cur is None:
            self.res_info.setHtml("<i>The results contain no load case or thermal case.</i>")
            self.view.clear_results()
            self.legend.clear()
            self.clear_hotspots()
            return
        kind, si = cur
        qi = max(self.r_qty.currentIndex(), 0)
        m = res.mesh
        anim_frame = kind == "step" and getattr(self, "_anim_frame", False)   # frame of the deformation animation
        qlist = self.result_qlist(cur)
        qname = qlist[min(qi, len(qlist) - 1)][0]
        self._probe_labels = None
        if kind == "twod":
            return self.show_twod_result(si, qi)
        if kind == "shell":
            return self.show_shell_result(si, qi)
        if kind == "sub":
            return smd.show(self, si, qi)
        if kind == "step" and (len(m.nodes) == 0 or frs.is_frame_quantity(qname)):
            return self.show_frame_result(si, qi)
        if kind == "mode" and len(m.nodes) == 0:
            return self.show_frame_mode(si)
        if kind == "dyn":
            return self.show_dyn_result(si, qi)
        tri, elem = m.boundary_faces()
        if hasattr(self, "r_part") and self.r_part.currentIndex() > 0:
            pname = self.r_part.currentText()
            if pname in m.part_names:
                pi = m.part_names.index(pname)
                kept_faces = (m.tet_part[elem] == pi)
                tri = tri[kept_faces]
        surf_nodes = np.unique(tri)
        st = th = frame = None
        if kind == "thermal":
            th = res.thermal[si]
            frame = self.current_frame(th)
            name, unit = THERMAL_QUANTITIES[min(qi, len(THERMAL_QUANTITIES) - 1)]
            vals = self.thermal_field(th, min(qi, len(THERMAL_QUANTITIES) - 1), frame)
            scale = 0.0
            X = m.nodes
            missing = "Heat flux is stored only for the end of a transient case – choose the last time."
            self._r_view = (0.0, None)
        elif kind == "mode":
            ms = res.modes[si[0]]
            shape = np.asarray(ms.shapes[si[1]], float)
            mq = mode_quantities(ms)                  # normalised shape | nonlinear buckling displacement [mm]
            qm = min(qi, len(mq) - 1)
            name, unit = mq[qm]
            vals = mode_quantity(shape, qm)
            scale = self.deformation_scale(shape)
            X = m.nodes + scale * shape
            missing = "The shape is not available."
            self._r_view = (scale, shape)
        elif kind == "combination":
            comb = self.study.combinations[si]
            from fea_core.combinations import EnvelopeResult
            try:
                st = self._combination(res, si)
            except (ValueError, KeyError) as e:
                self.res_info.setHtml(f"<i>Combination “{escape(str(comb.name))}” cannot be computed: {escape(str(e))}</i>")
                self.view.clear_results()
                self.legend.clear()
                self.clear_hotspots()
                return
            if isinstance(st, EnvelopeResult):          # envelope: one field over the nodes, without deformation
                unit = {"umag": "mm", "safety": "–"}.get(st.field_name, "MPa")
                name = f"Envelope {st.envelope_type}: {st.field_name}"
                vals = np.asarray(st.values, float)
                scale = 0.0
                X = m.nodes
                missing = "The envelope has no values."
                self._r_view = (0.0, None)
            else:
                name, unit = QUANTITIES[min(qi, len(QUANTITIES) - 1)]
                vals = self.result_field(st, min(qi, len(QUANTITIES) - 1))
                scale = self.deformation_scale(st.U)
                X = m.nodes + scale * st.U
                missing = "The quantity is not available in the combination."
                self._r_view = (scale, st.U)
        else:
            st = res.step(si)
            name, unit = QUANTITIES[min(qi, len(QUANTITIES) - 1)]
            vals = self.result_field(st, min(qi, len(QUANTITIES) - 1))
            anim_factor = getattr(self, "_anim_factor", 1.0)
            scale = self.deformation_scale(st.U) * anim_factor            # deformation scale
            if name in TOPO_Q:
                scale = 0.0                                                # density belongs to the undeformed shape
            X = m.nodes + scale * st.U
            missing = ("The case has no thermal load." if name == "Temperature" else
                       "The stress error estimate is not available for gravity cases that are combinations of unit states: it is not linear in the load."
                       if name == "Stress error estimate" and str_missing_by_superposition(getattr(res, "info", None), st.name) else
                       "The stress error estimate is computed only by CalculiX – choose the solver “Direct sparse solver – CalculiX” in the Solve step (the automatic choice uses the rslab direct solver for linear studies, its result does not have the estimate)." if name == "Stress error estimate" else
                       "The case has no contact pairs (Contacts step) – the contact quantity is not available."
                       if name in CONTACT_Q else
                       TOPO_MISSING if name in TOPO_Q else
                       "The result has no plastic strain – tick Plasticity in the Material properties window and solve again."
                       if name == PEEQ_Q else
                       "The quantity is not available in the result (no printed part).")
            self._r_view = (scale, st.U)
            if not anim_frame:                         # animation frame: only the deformed bands change
                self.contact_box.fill(res, si)
        if vals is None or np.isnan(vals).all():
            if name.startswith("3D printing"):
                missing = ("The 3D print quantity is not in the result – no part has a printed material. A material such as “PLA print XY (approximate)” is a solid isotropic material without layers and infill; to assess the print, choose a printed material in the 3D print step (e.g. “Print PLA Bambu Basic”) and run the calculation again.")
            self.res_info.setHtml(f"<i>{missing}</i>")
            self.view.clear_results()
            self.legend.clear()
            self.clear_hotspots()
            self.res_tabs.setCurrentIndex(0)          # the message is in the Summary – otherwise the view would stay empty without an explanation
            self.view.show_parts(True)                 # the model stays visible (without results)
            self.view.set_mode("results")
            return
        sv = vals[surf_nodes]
        vmin, vmax = float(sv.min()), float(sv.max())
        # section: trimmed surface + coloured section through the mesh interior (plane in the undeformed position)
        plane = self.section_plane()
        tris = outline = None
        probe_pts, probe_vals, probe_ids = X[surf_nodes], sv, surf_nodes
        shape_view = name == TOPO_SHAPE_Q and plane is None and kind == "step"
        if shape_view:
            # resulting shape ρ ≥ 0.5: closed surface of the isosurface instead of the mesh surface, without deformation, legend 0.5–1
            tris = self.topo_shape_tris(res)
            if tris is not None:
                vmin, vmax = 0.5, 1.0
                if len(tris[0]):
                    probe_pts, probe_vals = tris[0].reshape(-1, 3), tris[1].ravel()
                    probe_ids = np.full(len(probe_vals), -1)
        elif name in TOPO_Q:
            vmin, vmax = 0.0, 1.0
        if plane is not None:
            from fea_core.section import plane_section, clip_surface, section_outline
            o, nrm = plane
            Ps, Vs = clip_surface(m.nodes, tri, vals, o, nrm, X)
            Pc, Vc = plane_section(m.nodes, m.tets, vals, o, nrm, X)
            tris = (np.concatenate([Ps, Pc]), np.concatenate([Vs, Vc]))
            outline = section_outline(Pc)
            if len(Vc):
                vmin, vmax = min(vmin, float(Vc.min())), max(vmax, float(Vc.max()))
            kept = (m.nodes[surf_nodes] - o) @ nrm <= 0
            probe_pts = np.vstack([X[surf_nodes][kept], Pc.reshape(-1, 3)])
            probe_vals = np.concatenate([sv[kept], Vc.ravel()])
            probe_ids = np.concatenate([surf_nodes[kept], np.full(Vc.size, -1)])
        from .contour import band_edges
        if name == "Contact – contact state":          # discrete states (−1 not contact, 0 no contact, 1 sliding, 2 sticking)
            edges = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
        else:
            edges = band_edges(vmin, vmax, self.r_bands.value())
        colors = self.view.show_contour(X, tri, vals, edges, f"{name} [{unit}]", edges_on=self.r_edges.isChecked(),
                                        reverse=(name in SAFETY_QTY), show_scale=False, tris=tris, outline=outline,
                                        deform=(m.nodes, st.U, scale) if anim_frame and tris is None else None)
        self.legend.set_data(f"{name} [{unit}]", edges, colors, lambda v: fmt(float(v)))
        if self.r_outline.isChecked():
            self.view.show_parts(True, 0.85)
            self.view.set_mode("results")
        else:
            self.view.show_parts(False)
        imax = surf_nodes[int(np.argmax(sv))]
        # maximum places with singularity classification (table); note at the legend if the maximum is a singularity
        if kind in ("mode", "combination") or name in TOPO_Q:   # mode / buckling shape, combination, density: no maximum-places table
            self.clear_hotspots()
            self.res_tabs.setTabText(1, "Highest stresses")
            note, sing_xyz, r_cl = "", None, 0.0
        else:
            hot_key = (id(res), kind, si, name, frame, self.section_plane() is None)
            if anim_frame and getattr(self, "_anim_hot", (None,))[0] == hot_key:
                note, sing_xyz, r_cl = self._anim_hot[1]          # the table and the classification do not depend on the scale
                self._hs_X = X                                      # row click marks the current deformed position
            else:
                note, sing_xyz, r_cl = self.fill_hotspots(kind, si, name, unit, vals, X, frame)
                self._anim_hot = (hot_key, (note, sing_xyz, r_cl))
        self.legend.set_note(note)
        # max/min markers only on the displayed part (trimmed surface + section)
        jmax, jmin = int(np.argmax(probe_vals)), int(np.argmin(probe_vals))
        if name not in SAFETY_QTY and name != "Contact – contact state" and name not in TOPO_Q:
            lbl = f"max {fmt(float(probe_vals[jmax]))}"
            if sing_xyz is not None and probe_ids[jmax] >= 0 and \
                    np.linalg.norm(m.nodes[probe_ids[jmax]] - sing_xyz) <= r_cl:
                lbl += " (singularity)"
            self.view.mark(probe_pts[jmax], lbl, (170, 0, 0))
        if name not in NO_MIN_QTY:
            self.view.mark(probe_pts[jmin], f"min {fmt(float(probe_vals[jmin]))}",
                           (170, 0, 0) if name in SAFETY_QTY else (0, 60, 170))
        self._probe = (NearestPoints(probe_pts), probe_ids, probe_vals, unit, name)
        if th is not None:
            # thermal state: geometry without deformation → probe by interpolation with the element shape functions
            if name == "Temperature" and frame is None:
                self._probe_exact = lambda p, _n=th.name: float(res.probe(p, "T", _n)[0])
            else:
                self._probe_exact = lambda p, _v=vals: self.interpolate(_v, p)
            self.fill_thermal_info(th, frame)
        elif kind == "mode":
            self.res_info.setHtml(mode_info_html(res.modes[si[0]], si[1], self.study)
                                  + f"<br>Deformation scale: {fmt(scale, 3)}×")
            self.modes_table.select(*si)
            if self.res_tabs.currentIndex() == 1:
                self.res_tabs.setCurrentIndex(0)
        elif kind == "combination" and hasattr(st, "governing_summary"):
            names = escape(", ".join(str(n) for n in st.combination_names))
            gov = escape(", ".join(f"{k}: {v} nodes" for k, v in st.governing_summary.items()))
            self.res_info.setHtml(f"<b>{escape(str(st.name))}</b><br>Envelope {escape(str(st.envelope_type))} of field {escape(str(st.field_name))} from combinations: {names}<br>Governing combination: {gov}<br>Range: {fmt(st.min_val)} to {fmt(st.max_val)} {escape(str(unit))}")
        else:
            if not anim_frame:
                self.fill_result_info(st, scale, imax, vals)
            if st.frame:                               # mixed model: members via the result of the bodies
                frs.overlay(self, res, st, scale)
        if name == "Stress error estimate":
            self.legend.set_note(tr(STRESS_ERROR_TIP).replace("\n", " "))    # translate before joining the lines
        if name == "Contact – contact state":
            self.legend.set_note(STATUS_NOTE)
        dlg = self._path_dlg                       # progress line (results are redrawn → draw again)
        if dlg is not None and dlg.isVisible():
            dlg.draw_line()

    # ------------------------------------------------------------------ dynamics results (F8)
    def show_dyn_result(self, si, qi):
        """Dynamics field (dyn_results.field) as a colour map on the mesh surface (+ section), summary and plot tab."""
        res = self.results
        d = res.dynamics
        m = res.mesh
        sub, k = si
        self.clear_hotspots()
        self.contact_box.fill(None, None)
        if len(m.nodes) == 0 or d.shapes.shape[1] != len(m.nodes):
            self.res_info.setHtml("<i>Dynamics fields are shown only on the solid mesh – for members see the charts (Dynamics tab).</i>")
            self.view.clear_results()
            self.legend.clear()
            return
        idx = self.r_dyn_slider.value()
        phase = float(self.r_phase.value())
        absolute = self.r_abs.isChecked() and self.r_abs.isVisible()
        try:
            vals, U, name, unit = dynr.field(d, sub, k, idx, qi, phase, absolute)
        except Exception as e:  # noqa: BLE001 – missing data in the result (older file)
            self.res_info.setHtml(f"<i>The dynamics result cannot be shown: {escape(str(e))}</i>")
            self.view.clear_results()
            self.legend.clear()
            return
        ax = dynr.axis(d, sub)
        if ax is not None:
            xs, _lbl, u = ax
            self.r_dyn_val.setText(f"{fmt(float(xs[min(idx, len(xs) - 1)]), 5)} {u}")
        scale = 0.0 if U is None else self.deformation_scale(U)
        X = m.nodes + scale * U if U is not None else m.nodes
        self._r_view = (scale, U) if U is not None else (0.0, None)
        tri, _ = m.boundary_faces()
        surf_nodes = np.unique(tri)
        if not np.isfinite(vals).any():
            self.res_info.setHtml("<i>The quantity is not available in the result.</i>")
            self.view.clear_results()
            self.legend.clear()
            return
        vals = np.nan_to_num(vals, nan=0.0)
        sv = vals[surf_nodes]
        vmin, vmax = float(sv.min()), float(sv.max())
        plane = self.section_plane()
        tris = outline = None
        probe_pts, probe_vals, probe_ids = X[surf_nodes], sv, surf_nodes
        if plane is not None:
            from fea_core.section import plane_section, clip_surface, section_outline
            o, nrm = plane
            Ps, Vs = clip_surface(m.nodes, tri, vals, o, nrm, X)
            Pc, Vc = plane_section(m.nodes, m.tets, vals, o, nrm, X)
            tris = (np.concatenate([Ps, Pc]), np.concatenate([Vs, Vc]))
            outline = section_outline(Pc)
            if len(Vc):
                vmin, vmax = min(vmin, float(Vc.min())), max(vmax, float(Vc.max()))
            kept = (m.nodes[surf_nodes] - o) @ nrm <= 0
            probe_pts = np.vstack([X[surf_nodes][kept], Pc.reshape(-1, 3)])
            probe_vals = np.concatenate([sv[kept], Vc.ravel()])
            probe_ids = np.concatenate([surf_nodes[kept], np.full(Vc.size, -1)])
        from .contour import band_edges
        edges = band_edges(vmin, vmax, self.r_bands.value())
        colors = self.view.show_contour(X, tri, vals, edges, f"{name} [{unit}]", edges_on=self.r_edges.isChecked(),
                                        show_scale=False, tris=tris, outline=outline)
        self.legend.set_data(f"{name} [{unit}]", edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note(dynr.UNCERTAIN if sub != "mode" else "")
        self.view.show_parts(self.r_outline.isChecked(), 0.85)
        if len(probe_vals):
            jmax = int(np.argmax(probe_vals))
            self.view.mark(probe_pts[jmax], f"max {fmt(float(probe_vals[jmax]))}", (170, 0, 0))
        self._probe = (NearestPoints(probe_pts), probe_ids, probe_vals, unit, name)
        self.res_info.setHtml(dynr.info_html(d, sub, k, idx, phase, absolute, vals, name, unit, scale))

    # ------------------------------------------------------------------ results of frames and 2D (F7)
    def _finish_lines(self, title, edges, colors, pts, vals, labels, unit, name, reverse=False):
        """Legend, max / min markers and probe for results drawn as lines (frames)."""
        self.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note("")
        vals = np.asarray(vals, float)
        if len(vals):
            jmax, jmin = int(np.nanargmax(vals)), int(np.nanargmin(vals))
            self.view.mark(pts[jmax], f"max {fmt(float(vals[jmax]))}", (170, 0, 0))
            if vals[jmin] < 0 or name.split(" – ")[-1] in frs.DIAGRAMS:
                self.view.mark(pts[jmin], f"min {fmt(float(vals[jmin]))}", (0, 60, 170))
            self._probe = (NearestPoints(pts), np.full(len(pts), -1), vals, unit, name)
            self._probe_labels = labels

    def show_frame_result(self, si, qi):
        res = self.results
        st = res.steps[si]
        qlist = self.result_qlist()
        name, unit = qlist[min(qi, len(qlist) - 1)]
        if not frs.is_frame_quantity(name):
            name, unit = frs.FRAME_QUANTITIES[0]
        self.view.clear_results()
        self.clear_hotspots()
        self.view.show_parts(bool(self.parts), 0.85)
        self.view.set_mode("results")
        out = frs.step_draw(self, res, st, name, unit, self.r_bands.value())
        if out is None:
            self.res_info.setHtml("<i>The case has no member results.</i>")
            self.legend.clear()
            return
        title, edges, colors, pts, vals, labels, scale = out
        self._r_view = (0.0, None)
        self._finish_lines(title, edges, colors, pts, vals, labels, unit, name)
        self.res_info.setHtml(frs.step_info_html(self, res, st)
                              + (f"<br>Deformation scale: {fmt(scale, 3)}×" if scale else ""))
        self.frame_box.fill(res, st)
        self.view._update()

    def show_frame_mode(self, si):
        res = self.results
        ms = res.modes[si[0]]
        self.view.clear_results()
        self.clear_hotspots()
        self.view.set_mode("results")
        out = frs.mode_draw(self, res, ms, si[1], self.r_bands.value())
        info = mode_info_html(ms, si[1], self.study)
        if out is None:
            self.legend.clear()
            self.res_info.setHtml(info + "<br><i>The member shape is not available (buckling of members alone is stored since version F7; run the analysis again).</i>")
            return
        title, edges, colors, pts, vals, labels, scale = out
        self._r_view = (0.0, None)
        self._finish_lines(title, edges, colors, pts, vals, None, "–", title)
        self.res_info.setHtml(info + f"<br>Deformation scale: {fmt(scale, 3)}×")
        self.modes_table.select(*si)
        self.view._update()

    def show_twod_result(self, i, qi):
        res = self.results
        r2 = res.twod
        self.clear_hotspots()
        self.view.show_parts(bool(self.parts), 0.85)
        self.view.set_mode("results")
        title, edges, colors, X3, vals, scale, U3, key = twod_gui.draw(self, r2, i, qi, self.r_bands.value())
        if self.r_outline.isChecked():
            self.view.show_parts(True, 0.85)
        self.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note("2D section: 3D parts shown for orientation only (transparent); values apply in the section plane")
        jmax, jmin = int(np.argmax(vals)), int(np.argmin(vals))
        self.view.mark(X3[jmax], f"max {fmt(float(vals[jmax]))}", (170, 0, 0))
        self.view.mark(X3[jmin], f"min {fmt(float(vals[jmin]))}", (0, 60, 170))
        self._r_view = (scale, U3)
        fr = r2.frame2d()
        self._probe = (NearestPoints(X3), np.arange(len(X3)), vals, title.split("[")[-1].rstrip("]"), title.split(" [")[0])
        self._probe_exact = lambda p, _k=key, _i=i: float(r2.probe(fr.to2d(self.undeformed(p)), _k, _i))
        self.res_info.setHtml(twod_gui.info_html(self, r2, i, key))
        dlg = self._path_dlg
        if dlg is not None and dlg.isVisible():
            dlg.draw_line()

    def show_shell_result(self, i, qi):
        """Result of the shell model (Results.shell) on the mid-surface: contours, markers, probe, summary."""
        rs = self.results.shell
        self.clear_hotspots()
        self.view.show_parts(False)
        self.view.set_mode("results")
        title, edges, colors, X, vals, scale, U, key, layer = shell_gui.draw(self, rs, i, qi, self.r_bands.value())
        if self.r_outline.isChecked():
            self.view.show_parts(True, 0.85)
        self.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note("Shells: contours on the mid-surface; layer depends on the quantity (top / middle / bottom surface, max of surfaces)")
        jmax, jmin = int(np.argmax(vals)), int(np.argmin(vals))
        self.view.mark(X[jmax], f"max {fmt(float(vals[jmax]))}", (170, 0, 0))
        if key not in ("mises", "membrane", "bending", "u", "thickness"):
            self.view.mark(X[jmin], f"min {fmt(float(vals[jmin]))}", (0, 60, 170))
        self._r_view = (scale, U)
        self._probe = (NearestPoints(X), np.arange(len(X)), vals, title.split("[")[-1].rstrip("]"), title.split(" [")[0])
        self._probe_exact = lambda p, _v=vals: self.interpolate_shell(rs, _v, p)   # displayed field (layer)
        self.res_info.setHtml(shell_gui.info_html(self, rs, i))

    def interpolate_shell(self, rs, vals, p) -> float:
        """Nodal field of the shells at a point (nearest point of the mid-surface, tri6 shape functions)."""
        from fea_core.shell.loads import tri6_shape
        q = rs.probe(self.undeformed(p), "u", 0, "mid")
        e = q["element"]
        X = rs.nodes[rs.tri[e]]
        a, b, d = X[0], X[1], X[2]
        u, v = b - a, d - a
        G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
        xi, eta = np.clip(np.linalg.solve(G, [u @ (np.asarray(q["point"]) - a), v @ (np.asarray(q["point"]) - a)]), 0, 1)
        N, _ = tri6_shape(xi, eta)
        return float(N @ np.asarray(vals)[rs.tri[e]])

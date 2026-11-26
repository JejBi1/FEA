"""Mesh and Solve of the main window: both panels, mesh settings and refinements, the mesh and solve threads with their callbacks,
solver choice, readiness check and progress bars.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget, QFrame, QScrollArea)

from fea_core import config
from fea_core.solve import check_study
from fea_core.study import Study
from fea_core.i18n import decimal, tr

from . import twod_gui, shell_gui, theme
from .eigen_gui import EigenBox
from .widgets import Section, TaskProgress, dspin, help_tip, panel_header, fit_rows, set_placeholder
from .workers import MeshThread, SolveThread, kill_mesh_workers

from .main_defs import (APP_NAME, STAGES, STAGE, element_quality_nodes, GREEN, ORANGE, RED, STORED_SOLVER, BACKENDS, F8_SOURCE,
                        fmt, dot_icon, msg_via_main as msg)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class MeshSolveMixin(_Base):
    """Mesh and Solve of the main window: both panels, mesh settings and refinements, the mesh and solve threads with their
    callbacks, solver choice, readiness check and progress bars."""

    # ------------------------------------------------------------------ Mesh panel
    def _panel_mesh(self, lay):
        lay.addWidget(panel_header("Mesh", "Quadratic tetrahedra tet10 (C3D10), Gmsh", STAGE["Mesh"] + 1))
        g = Section("Settings", key="mesh.settings", tip="Global mesh settings – the element size applies to the whole model")
        f = QFormLayout(g.body)
        self.ms_size = dspin(5.0, 0.01, 1e5, 3, suffix=" mm")
        self.ms_min = dspin(0.0, 0.0, 1e5, 3, suffix=" mm")
        self.ms_min.setSpecialValueText("auto")
        self.ms_curv = QSpinBox()
        self.ms_curv.setRange(0, 72)
        self.ms_curv.setValue(12)
        self.ms_curv.setSpecialValueText("off")
        help_tip(self.ms_size, "Target element size [mm] – smaller = more accurate and slower",
                 "Target tet10 element size [mm]. Smaller = more accurate and slower. Start with ~1/10–1/20 of the part size, at least 2 elements across thin walls; check the convergence (reduce by ~30 % and compare the results).")
        help_tip(self.ms_min, "Smallest allowed element size [mm] (auto = by Gmsh)",
                 "Smallest allowed element size [mm] (auto = by Gmsh) – limits the refinement in small details (fillets, holes)")
        help_tip(self.ms_curv, "Elements per circle circumference – refines the mesh in holes and fillets",
                 "Number of elements around the whole circle – refines the mesh in holes and fillets (12 is usually enough, more for stress concentration at a hole)")
        f.addRow("Element size", self.ms_size)
        f.addRow("Min. size", self.ms_min)
        f.addRow("Elements on circle", self.ms_curv)
        for w in (self.ms_size, self.ms_min, self.ms_curv):
            w.valueChanged.connect(self.on_mesh_settings)
        lay.addWidget(g)
        g = Section("Local refinement", key="mesh.refine", tip="Smaller elements on the selected faces and in a sphere around a point")
        v = QVBoxLayout(g.body)
        v.setContentsMargins(f.contentsMargins())     # same margins as the Settings section
        self.ref_list = QListWidget()
        self.ref_list.currentRowChanged.connect(self.on_ref_row)
        self.ref_list.setToolTip("Local mesh refinement on selected faces (element size at the face)")
        set_placeholder(self.ref_list, "No local refinement yet")
        fit_rows(self.ref_list, 2, 6)
        v.addWidget(self.ref_list)
        h = QHBoxLayout()
        self.ref_size = dspin(1.0, 0.01, 1e5, 3, suffix=" mm")
        self.ref_size.setToolTip("Element size on the selected faces [mm] (smaller than the global size)")
        b = QPushButton("Refine selection")
        help_tip(b, "Refine the mesh on faces selected in 3D (notches, holes, transitions)",
                 "Refine the mesh on faces selected in 3D – stress concentration spots (notches, holes, transitions).<br>Turn off “Show mesh” to select faces.")
        b.clicked.connect(self.add_refinement)
        h.addWidget(self.ref_size)
        h.addWidget(b)
        v.addLayout(h)
        b = QPushButton("Delete refinement")
        b.setToolTip("Delete the refinement selected in the list (faces and spheres)")
        b.clicked.connect(lambda: self.delete_item(self.ref_list, "refine"))
        v.addWidget(b)
        # refinement in a sphere around a point (MeshSettings.local {"center", "radius", "size"}, Gmsh Ball field)
        gb = Section("Refine around point (sphere)", key="mesh.ball", collapsed=True,
                     tip="Smaller elements inside a sphere around a point – also inside the volume",
                     help="Smaller elements inside a sphere around a point – also inside the volume (notch, weld root, stress maximum spot).<br>Outside the sphere the global element size is kept.")
        fb = QFormLayout(gb.body)
        hp = QHBoxLayout()
        self.ball_pt = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        for s, ax in zip(self.ball_pt, "xyz"):
            help_tip(s, f"{ax} coordinate of the sphere center [mm]",
                     f"{ax} coordinate of the sphere center [mm] (enter it, “Point from selection”, or the edge / circle center with the Edge filter → “Use center” in the status bar)")
            hp.addWidget(s)
        self.register_point_input(self.ball_pt)
        b = QPushButton("Point from selection")
        help_tip(b, "Sphere center from the selected point, edge / circle, or faces",
                 "Sphere center = the last point selected in 3D with the Point filter (V), otherwise the center of the selected edge / circle (Edge filter), otherwise the centroid of the selected faces")
        b.clicked.connect(self.ball_point_from_selection)
        hp.addWidget(b)
        fb.addRow("Center [mm]", hp)
        hr = QHBoxLayout()
        self.ball_r = dspin(5.0, 0.001, 1e6, 3, 1.0, " mm")
        self.ball_r.setToolTip("Refinement sphere radius [mm] – usually 2–5 × the detail size")
        self.ball_h = dspin(0.5, 0.001, 1e5, 3, 0.1, " mm")
        self.ball_h.setToolTip("Element size inside the sphere [mm] (smaller than the global size)")
        hr.addWidget(QLabel("R"))
        hr.addWidget(self.ball_r, 1)
        hr.addWidget(QLabel("element"))
        hr.addWidget(self.ball_h, 1)
        fb.addRow("Sphere", hr)
        b = QPushButton("Refine around point")
        b.setToolTip("Add a refinement in a sphere (center, radius, element size)")
        b.setWhatsThis("<b>Refine around point</b><p>The mesh inside the sphere will be finer (element size “element”), also inside the volume – suitable for a stress maximum from the results or a weld root. Enter the point as numbers, select it with the Point filter (V) and press “Point from selection”, or take the edge center (Edge filter → “Use center”).</p><p>The refinement is shown in the list above.</p>")
        b.clicked.connect(self.add_ball_refinement)
        self.btn_ball = b
        fb.addRow(b)
        v.addWidget(gb)
        lay.addWidget(g)
        h = QHBoxLayout()
        self.btn_mesh = QPushButton("Generate mesh")
        help_tip(self.btn_mesh, "Generate the mesh (Gmsh, tet10) – the analysis also creates it by itself",
                 "Generate the mesh (Gmsh, quadratic tetrahedra tet10) and show it.<br>Not required: Run analysis creates the mesh by itself.")
        self.btn_mesh.clicked.connect(self.make_mesh)
        theme.set_primary(self.btn_mesh)
        self.btn_mesh_cancel = QPushButton("Cancel")
        self.btn_mesh_cancel.setEnabled(False)
        self.btn_mesh_cancel.setToolTip("Terminate the Gmsh meshing process")
        self.btn_mesh_cancel.clicked.connect(self.cancel_mesh)
        h.addWidget(self.btn_mesh, 1)
        h.addWidget(self.btn_mesh_cancel)
        lay.addLayout(h)
        self.ms_show = QCheckBox("Show mesh")
        self.ms_show.setChecked(True)
        help_tip(self.ms_show, "On = the mesh is shown in 3D; off = parts and face selection",
                 "On: the surface of the finished mesh is shown in 3D (parts hidden).<br>Off: parts and face selection by clicking – e.g. for Refine selection.")
        self.ms_show.toggled.connect(self.on_mesh_show)
        lay.addWidget(self.ms_show)
        self.mesh_info = QLabel("")
        self.mesh_info.setWordWrap(True)
        lay.addWidget(self.mesh_info)

    # ------------------------------------------------------------------ Solve panel
    def _panel_solve(self, lay):
        outer = lay                                 # content in the scroll area (natural frequencies, buckling, 2D)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(inner)
        outer.addWidget(sc, 100)
        lay.addWidget(panel_header("Solve", "Statics – direct solver (rslab or CalculiX) or iterative PCG/AMG (hypre)", STAGE["Solve"] + 1))
        self.ready_list = QListWidget()
        fit_rows(self.ready_list, 4, 12)
        help_tip(self.ready_list, "Readiness check: errors (cannot run) and warnings",
                 "Study readiness check: errors (the analysis cannot be run) and warnings (materials, supports, rigid body motions, loads…). Fix the errors in the steps listed.")
        lay.addWidget(self.ready_list)
        self.solver_section = Section("Solver", key="solve.solver", collapsed=True,
                                      tip="Solver choice: automatic, direct (CalculiX, rslab) or iterative (hypre)")
        f = QFormLayout(self.solver_section.body)
        from fea_core.solvers import backend_status, install_hint, linear_solver_note
        avail = backend_status()                  # solvers not installed are greyed out (cannot be selected; the check reports the error)
        self.backend_combo = QComboBox()
        for key, txt in BACKENDS:
            st_ = avail.get(key)
            missing = st_ is not None and not st_["ok"]
            self.backend_combo.addItem(f"{txt} – not installed" if missing else txt, key)   # template for i18n
            if missing:
                it = self.backend_combo.model().item(self.backend_combo.count() - 1)
                it.setEnabled(False)
                it.setToolTip(f"Missing {', '.join(st_['missing'])}")
        help_tip(self.backend_combo, "Solver: automatic, direct (CalculiX, rslab) or iterative PCG + AMG",
                 "<b>Solver</b><p><b>Automatic</b> (recommended): contacts and other nonlinear analyses and beam structures → CalculiX; everything else linear (bolts, welds, cylindrical, elastic and remote supports, *TIE, orthotropy included) → the direct solver rslab with the program's own matrix assembly (one factorization for all load cases, also for a single case) while it fits into the free memory, then the iterative hypre solver (if installed); if the chosen solver fails, the calculation continues with the next one (CalculiX at the end). The automatic choice skips solvers that are not installed; the reason of the choice is in the properties of the calculation.</p><p><b>Direct sparse solver – CalculiX</b>: verified by all verification problems, memory grows fast (limit ~1–2 million DOF).<br><b>rslab</b>: direct, one factorization for all cases (module direct); the factor is stored in float32 and refined in float64 when that is faster or when float64 does not fit into memory.<br><b>hypre</b>: iterative PCG with AMG, memory grows linearly – large models (module hypre, Microsoft MPI on Windows).</p><p>In other programs: direct = ANSYS Direct (Sparse), SOLIDWORKS Direct Sparse, the Nastran direct solver; iterative = ANSYS Iterative (PCG), SOLIDWORKS FFEPlus.</p><p>Grayed-out solvers are not installed – button <i>Install solvers…</i> (Tools → Program modules…).</p>")
        self.backend_combo.currentIndexChanged.connect(self.on_backend)
        f.addRow("Solver", self.backend_combo)
        self.solver_combo = QComboBox()
        for txt, key in (("Automatic", config.AUTO_SOLVER), ("PaStiX", "PASTIX"), ("SPOOLES", "SPOOLES")):   # stored word of "automatic": config.AUTO_SOLVER
            self.solver_combo.addItem(txt, key)
        for i in range(self.solver_combo.count()):
            note = linear_solver_note(self.solver_combo.itemData(i))
            if note:                              # PaStiX in the Linux CalculiX
                it = self.solver_combo.model().item(i)
                it.setEnabled(False)
                it.setToolTip(f"Not available – {note}")
        self.solver_combo.currentIndexChanged.connect(lambda _=0: self.on_solver(self.solver_combo.currentData()))
        help_tip(self.solver_combo, "Equation solver in CalculiX: Automatic (default), PaStiX, SPOOLES",
                 "<b>Equation solver (CalculiX)</b><p>Direct solver of the equation system inside CalculiX (the rslab and hypre solvers do not use it): <b>Automatic</b> (default): in the CalculiX of this program the <b>faer</b> solver for symmetric systems (statics, contact without friction, heat, buckling, natural frequencies; typically 1.5 to 3 times faster than SPOOLES; every factorization is checked by a test solve and SPOOLES takes over when it fails) and SPOOLES for the rest (friction, radiation, small systems). <b>SPOOLES</b> – always SPOOLES, the solver that was used before faer (the factorization uses the threads of the job, the solution of the triangular systems one; always available). <b>PaStiX</b> is faster for statics, buckling, heat and contact, but it is not part of the program (its licence does not fit into the distributed GPL-2.0 CalculiX); it exists only in a CalculiX that you build yourself with PaStiX (Linux, installer/linux/build_ccx.sh). The results are the same.</p><p>A grayed-out solver is not in this installation – CalculiX then uses SPOOLES, which is stated in the log or the result warnings.</p>")
        f.addRow("Equation solver (CalculiX)", self.solver_combo)
        self.modal_engine_combo = QComboBox()
        for txt, key in (("Automatic", "auto"), ("Direct (rslab + ARPACK)", "direct"), ("CalculiX", "ccx")):
            self.modal_engine_combo.addItem(txt, key)
        help_tip(self.modal_engine_combo, "Natural frequencies and modal basis: automatic, direct or CalculiX",
                 "<b>Natural frequencies and the modal basis of the dynamics</b><p><b>Automatic</b> (default): the direct path where it can do the study – the program's own mass and stiffness matrices, one rslab factorization and the ARPACK Lanczos iteration; the number of modes is <i>proved</i> by the Sturm count (inertia of K − σM), every pair has its residual checked, and the static corrections of the dynamics come from the same factor. CalculiX *FREQUENCY stays for what needs the stress stiffness (buckling, prestressed modal analysis, Campbell), cyclic symmetry, bolts (their beam elements have no mass matrix), frames, contact and nonlinear states, shells and 2D, or when the direct solver is not installed or does not fit into the memory; a failed direct analysis continues in CalculiX with the reason. A small problem (DOF × (modes + 10) below 7·10⁵) stays in CalculiX – the direct analysis needs about 6 s to start its two processes.</p><p><b>Direct</b> / <b>CalculiX</b>: force the engine (Direct reports a study it cannot do instead of switching).</p><p>The results say which engine computed the modes; with <i>verify with CalculiX</i> (natural frequencies / dynamics) the same modes are also computed in CalculiX and compared – frequencies, effective masses, MAC.</p>")
        self.modal_engine_combo.currentIndexChanged.connect(self.on_modal_engine)
        f.addRow("Natural frequencies", self.modal_engine_combo)
        miss = {k: v for k, v in avail.items() if not v["ok"]}
        self.btn_install_solvers = QPushButton("Install solvers…")
        help_tip(self.btn_install_solvers, "Opens Program modules – install the missing solvers",
                 "<b>Install more solvers</b><p>" + "<br>".join(
                     f"{v['title']}: missing {', '.join(v['missing'])}" for v in miss.values())
                 + "</p><p>" + install_hint() + "</p>")
        self.btn_install_solvers.clicked.connect(self.open_modules_dialog)
        f.addRow(self.btn_install_solvers)
        f.setRowVisible(self.btn_install_solvers, bool(miss))     # only when something is missing
        lay.addWidget(self.solver_section)
        self.backend_combo.currentIndexChanged.connect(self._backend_badge)
        self._backend_badge()
        self.eigen_box = EigenBox(self)          # natural frequencies and linear buckling (Study.modal, Study.buckling)
        lay.addWidget(self.eigen_box)
        self.twod_box = twod_gui.TwoDBox(self)    # 2D calculation by a section (Study.twod)
        lay.addWidget(self.twod_box)
        self.shell_box = shell_gui.ShellBox(self)  # shell model of plates and sheets (Study.shell)
        lay.addWidget(self.shell_box)
        h = QHBoxLayout()
        self.btn_run = QPushButton("Run analysis")
        self.btn_run.setToolTip("Run analysis: check → mesh → statics → equilibrium → results")
        self.btn_run.setWhatsThis("<b>Run analysis</b><p>First the study is checked (list above). Then the mesh is generated, the thermal cases are solved and the linear statics of each load case. At the end the program verifies the equilibrium of reactions and loads – a result with an imbalance is not returned.</p><p>When finished, the Results step opens. Progress is in the log below.</p>")
        theme.set_primary(self.btn_run)
        self.btn_run.clicked.connect(self.run_solve)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_solve)
        self.btn_cancel.setToolTip("Abort the running analysis (terminates solver processes)")
        self.btn_stop_keep = QPushButton("Stop and keep results")
        self.btn_stop_keep.setEnabled(False)
        self.btn_stop_keep.clicked.connect(self.stop_keep_solve)
        help_tip(self.btn_stop_keep, "Stop the analysis and keep the last converged increment (partial)",
                 "Nonlinear analysis (contact, step sequence): stops CalculiX and keeps the last converged increment as a PARTIAL result (red label). If there is no converged increment the analysis ends with an error.")
        h.addWidget(self.btn_run)
        h.addWidget(self.btn_cancel)
        h.addWidget(self.btn_stop_keep)
        lay.addLayout(h)
        self.solve_progress = TaskProgress()      # green bar 0–100 % (progress from the core, fea_core.progress)
        lay.addWidget(self.solve_progress)
        self.solve_info = QLabel("")
        self.solve_info.setWordWrap(True)
        lay.addWidget(self.solve_info)
        from .nl_monitor import NlMonitor           # convergence of a nonlinear analysis (A7), live and from the results
        self.nl_monitor = NlMonitor(self)
        lay.addWidget(self.nl_monitor)
        lay.addStretch(1)

    # ================================================================== mesh
    def on_mesh_settings(self):
        if self.study is None or getattr(self, "_filling", False):
            return
        self.study.mesh.size = self.ms_size.value()
        self.study.mesh.min_size = self.ms_min.value() or None
        self.study.mesh.curvature_points = self.ms_curv.value()
        self.changed(invalidate_mesh=True)

    def add_refinement(self):
        if getattr(self, "_mesh_view", False) and not self.view.selected_faces():
            msg(self, APP_NAME, "The mesh is displayed in 3D – turn off “Show mesh”, select part faces and then click Refine selection.")
            return
        sels = self.need_faces()
        if sels:
            name = self.new_selection("Refinement", sels)
            self.study.mesh.local.append({"selection": name, "size": self.ref_size.value()})
            self.view.set_selected_faces([])
            self.changed(invalidate_mesh=True)

    def on_ref_row(self, row):
        if 0 <= row < len(self.study.mesh.local):
            r = self.study.mesh.local[row]
            if r.get("selection"):
                self.view.set_selected_faces(self.sel_faces.get(r["selection"], []))
            elif r.get("center") is not None:
                for s, v in zip(self.ball_pt, r["center"]):
                    s.setValue(float(v))
                self.ball_r.setValue(float(r.get("radius", 1.0)))
                self.ball_h.setValue(float(r.get("size", 1.0)))

    @staticmethod
    def describe_refinement(r) -> str:
        if r.get("selection"):
            return f"{r['selection']}: {fmt(r['size'])} mm"
        c = r.get("center") or [0, 0, 0]
        return (f"Sphere ({'; '.join((fmt(float(x)) for x in c))}) R {fmt(float(r.get('radius', 0)))} mm: {fmt(float(r['size']))} mm")

    def ball_point_from_selection(self):
        """Sphere centre from the 3D selection: point (filter Point) > centre of an edge / circle > centroid of the faces."""
        p = None
        pts = self.view.selected_points()
        e = self.view.selected_edge()
        if pts:
            p = pts[-1][1]
        elif e is not None:
            p = e["point"]
        elif self.view.selected_edges():
            p = self.edge_prop(*self.view.selected_edges()[-1])["point"]
        elif self.view.selected_faces():
            sels = self.selected_selectors()
            w = np.array([s.area for s in sels])
            p = (np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        if p is None:
            msg(self, APP_NAME, "Select a point (Point filter V), an edge (Edge filter E) or faces in 3D.")
            return
        for s, v in zip(self.ball_pt, p):
            s.setValue(float(v) + 0.0)

    def add_ball_refinement(self):
        if self.study is None:
            return
        if self.ball_h.value() >= self.study.mesh.size:
            msg(self, APP_NAME, "The element size in the sphere must be smaller than the global element size.")
            return
        self.study.mesh.local.append({"center": [s.value() for s in self.ball_pt], "radius": self.ball_r.value(),
                                      "size": self.ball_h.value()})
        self.changed(invalidate_mesh=True)

    # meshing inputs (Gmsh); the geometry is checked by _load_model, materials and the 3D printing settings do not change the mesh – the assignment
    # of shells and infill is recomputed on the finished mesh (fea_core.printing.region.ensure)
    MESH_KEYS = ("mesh", "selections", "bolts", "welds", "joints", "ignored_parts")

    # display / post-processing only – never inputs of the mesh or the FE calculation (results stay valid when they change)
    POST_KEYS = ("fatigue", "hidden_parts", "assessment", "combinations", "features")

    PRINT_POST_KEYS = ("sweep", "variants", "required_safety")     # printing: definitions of variants / sweep, requirement

    @classmethod
    def calc_inputs(cls, d: dict, joints: bool = False) -> dict:
        """Study.to_dict() without the post-processing fields (POST_KEYS, EN 1990 category of the load cases, printing
        sweep / variants); joints=True also drops the joint fields that only the assessment uses (fea_core.joints.levels)."""
        d = {k: v for k, v in d.items() if k not in cls.POST_KEYS}
        if isinstance(d.get("load_cases"), list):
            d["load_cases"] = [{k: v for k, v in lc.items() if k != "category"} if isinstance(lc, dict) else lc
                               for lc in d["load_cases"]]
        if isinstance(d.get("printing"), dict):
            d["printing"] = {k: v for k, v in d["printing"].items() if k not in cls.PRINT_POST_KEYS}
        if joints:
            from fea_core.joints import levels
            js = d.get("joints") if isinstance(d.get("joints"), dict) else {}
            preload = bool(js.get("preload_case"))
            if js:
                d["joints"] = {k: v for k, v in js.items() if k not in levels.SETTINGS_ASSESS}
            bolt_post = levels.BOLT_ASSESS | (set() if preload else levels.BOLT_PRELOAD)
            for key, post in (("bolts", bolt_post), ("welds", levels.WELD_ASSESS)):
                if isinstance(d.get(key), list):
                    d[key] = [{k: v for k, v in x.items() if k not in post} if isinstance(x, dict) else x
                              for x in d[key]]
        return d

    def study_signature(self, keys=None) -> str:
        """Signature of the study (without the fatigue settings) – the result of a thread started on another study is discarded."""
        if keys is self.MESH_KEYS:                    # meshing inputs – shared with MCP (fea_core.signature)
            from fea_core.signature import mesh_inputs
            return json.dumps(mesh_inputs(self.study, geometry=False), sort_keys=True, default=str)
        d = self.calc_inputs(self.study.to_dict(), joints=True)
        if keys:
            d = {k: d.get(k) for k in keys}
        return json.dumps(d, sort_keys=True, default=str)

    def valid_mesh(self):
        """Mesh that can be taken over into the calculation (the meshing inputs have not changed since it was created), otherwise None."""
        if self.mesh is not None and self._mesh_sig is not None and self._mesh_sig == self.study_signature(self.MESH_KEYS):
            return self.mesh
        return None

    def make_mesh(self):
        if self.study is None or self._thread is not None:
            return
        self.btn_mesh.setEnabled(False)
        self.btn_mesh_cancel.setEnabled(True)
        self.mesh_info.setText("Meshing…")
        self.status_progress.start("Meshing (Gmsh)", 0.95)
        self.log_msg("Meshing (Gmsh)…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self._thread_sig = self.study_signature(self.MESH_KEYS)
        t = MeshThread(Study.from_dict(self.study.to_dict()), self.shadow_solid_parts,     # parts read in the thread
                       self._workdir / f"mesh_{int(time.time() * 1000)}", self)
        t.done.connect(self.on_mesh_done)
        t.failed.connect(self.on_mesh_failed)
        t.finished.connect(self._thread_finished)
        self._thread = t
        t.start()

    def cancel_mesh(self):
        if isinstance(self._thread, MeshThread):
            self._thread.cancel.set()
            n = kill_mesh_workers()
            self.log_msg("Cancelling meshing…" + (f" (Gmsh processes terminated: {n})" if n else ""))

    def on_mesh_done(self, mesh, cancelled=None):
        t = self._thread                         # the done signal arrives before finished
        if cancelled is None:
            cancelled = isinstance(t, MeshThread) and t.cancel.is_set()
        if self._defer_done(lambda: self.on_mesh_done(mesh, cancelled)):
            return
        if cancelled:
            self.log_msg("Meshing canceled.")
            self.mesh_progress_finish("cancelled", "Meshing canceled")
            self.refresh_all()
            return
        if self.study is None or self.study_signature(self.MESH_KEYS) != self._thread_sig:
            self.log_msg("The study changed during meshing – mesh discarded, generate it again.")
            self.mesh_progress_finish("cancelled", "Mesh discarded – study changed")
            self.refresh_all()
            return
        self.mesh_progress_finish("done", "Mesh done")
        self.mesh = mesh
        self._mesh_sig = self._thread_sig
        r = mesh.report
        self.log_msg(f"Mesh: {fmt(r['nodes'])} nodes, {fmt(r['elements'])} elements, min. SJ {r['min_scaled_jacobian']:.3f}")
        for w in r.get("warnings", []):
            self.log_msg(f"Warning: {w}")
        for t in r.get("ties", []):
            self.log_msg(f"*TIE connection: part {t['part']}, contact {fmt(t['area_mm2'])} mm²")
        self.mesh_messages(mesh)
        self.refresh_all()
        if STAGES[self.stage] == "Mesh":
            self.ms_show.blockSignals(True)          # a new mesh is always shown
            self.ms_show.setChecked(True)
            self.ms_show.blockSignals(False)
            self.set_stage(self.stage)

    def on_mesh_failed(self, text):
        t = self._thread                         # the done signal arrives before finished
        if isinstance(t, MeshThread) and t.cancel.is_set():
            self.log_msg("Meshing canceled.")
            self.mesh_info.setText("Meshing canceled.")
            self.mesh_progress_finish("cancelled", "Meshing canceled")
            return
        self.mesh_progress_finish("error", "Meshing failed – see the log for the reason")
        self.log_msg(f"Mesh ERROR: {text}")
        self.messages.set_source("mesh", [("error", f"Meshing failed: {text}")])
        self.dock_messages.raise_()
        msg(self, APP_NAME, f"Meshing failed:\n{text}", QMessageBox.Critical)
        self.refresh_all()

    def f8_messages(self, res):
        """Messages of the nonlinear calculation and dynamics (F8): partial result with advice, list of tasks, checks of the modal
        basis. Texts that another source already reports (result warnings) are left out (unique).
        """
        items = []
        nl = (getattr(res, "info", None) or {}).get("nonlinear") or {}
        if nl:
            items.append(("info", f"Nonlinear analysis: {('states separately (branches)' if nl.get('mode') == 'branches' else 'step sequence')}, CalculiX jobs {len(nl.get('jobs') or [])}"))
            for job, p in (nl.get("partial") or {}).items():
                t = p.get("last_step_time")
                items.append(("warning", f"Nonlinear calculation ({job}) did not converge – partial result: the last converted step {p.get('last_step')} of {p.get('of_steps')}"
                                         + (f", step time {fmt(float(t), 4)}" if t is not None else "")
                                         + f", increment cut-back {p.get('cutbacks', 0)}×"))
                for e in p.get("errors") or []:
                    items.append(("warning", f"CalculiX ({job}): {e}"))
            if nl.get("partial"):
                from fea_core.contacts.results import ADVICE
                items.append(("info", f"Nonlinear analysis: {ADVICE}"))
        d = getattr(res, "dynamics", None)
        if d is not None:
            items.append(("info", f"Dynamics: {d.K} modes {fmt(float(d.freq[0]), 5)}–{fmt(float(d.freq[-1]), 5)} Hz, model mass {fmt(1000 * d.model_mass)} kg" if d.K else "Dynamics: no modes"))
            items += [("warning", w) for w in d.warnings]
            items.append(("warning", "Dynamics: damping is an uncertain input (sensitivity ±50 %) – verify resonance results by measuring the damping"))
        self.messages.set_source(F8_SOURCE, items, "Results", unique=True)

    def mesh_messages(self, mesh):
        """Messages of the "mesh" source: mesh size, meshing warnings, *TIE connections."""
        r = mesh.report
        items = [("info", f"Mesh: {fmt(r.get('nodes', len(mesh.nodes)))} nodes, {fmt(r.get('elements', len(mesh.tets)))} elements, min. scaled Jacobian {fmt(r.get('min_scaled_jacobian'), 3)}")]
        items += [("warning", w) for w in r.get("warnings", []) if w not in r.get("study_warnings", [])]
        items += [("info", f"*TIE connection: part {t['part']} – contact {fmt(t['area_mm2'])} mm², tolerance {fmt(t['tolerance_mm'], 3)} mm") for t in r.get("ties", [])]
        self.messages.set_source("mesh", items)

    def _thread_finished(self):
        QApplication.restoreOverrideCursor()
        self._thread = None
        self._busy_warned = False                 # the next run asks again
        self.flush_solver_lines()
        self.btn_mesh.setEnabled(True)
        self.btn_mesh_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_stop_keep.setEnabled(False)
        if getattr(self, "_stop_timer", None) is not None:
            self._stop_timer.stop()
        self.refresh_ready()

    def show_mesh(self):
        if self.mesh is None:
            return
        tri, _ = self.mesh.boundary_faces()
        if getattr(self, "_mesh_color", "plain") == "quality":
            q = element_quality_nodes(self.mesh)
            edges = np.linspace(0.0, 1.0, 11)
            self.view.show_contour(self.mesh.nodes, tri, q, edges, tr("Element quality (1 = regular)"), "%.2f",
                                   edges_on=True, reverse=True)
            return
        self.view.show_mesh(self.mesh.nodes, tri)

    # ================================================================== calculation
    def on_solver(self, name):
        if self.study and self.study.solver.solver != name:
            self.study.solver.solver = name
            self.changed()
            QTimer.singleShot(0, self._show_stored_solver)      # the temporary item of the replaced value goes (not inside the handler of the menu)

    def _show_stored_solver(self) -> None:
        """Selects in the equation-solver menu what the study stores – the menu must never show the choice of the previous study. A stored value that
        the menu does not offer but CalculiX uses as written (`ITERATIVE SCALING` / `ITERATIVE CHOLESKY` of an edited or foreign study file) gets a temporary item
        with that value (selecting it changes nothing); a word that is no solver at all is what `config.solver_name` makes of it: Automatic.
        Signals of the menu are blocked here, the study is not changed."""
        c = self.solver_combo
        word = str(getattr(self.study.solver, "solver", "") or config.AUTO_SOLVER).strip().upper() if self.study else config.AUTO_SOLVER
        if word not in config.KNOWN_SOLVERS:
            word = config.AUTO_SOLVER
        c.blockSignals(True)
        try:
            for i in range(c.count() - 1, -1, -1):
                if c.itemText(i).endswith(STORED_SOLVER) and c.itemData(i) != word:
                    c.removeItem(i)
            i = c.findData(word)
            if i < 0:
                c.addItem(word + STORED_SOLVER, word)
                i = c.count() - 1
                c.model().item(i).setToolTip("Stored in the study; this menu does not offer it, CalculiX uses it as written. Choose another item to replace it.")
            c.setCurrentIndex(i)
        finally:
            c.blockSignals(False)

    def _backend_badge(self, _=0):
        """Badge of the collapsed Solver section: the chosen solver."""
        from fea_core.solvers import TITLES
        self.solver_section.set_badge(TITLES.get(self.backend_combo.currentData(), ""))

    def on_backend(self, _):
        key = self.backend_combo.currentData()
        if self.study and key and self.study.solver.backend != key:
            self.study.solver.backend = key
            self.changed()

    def on_modal_engine(self, _=0):
        key = self.modal_engine_combo.currentData()
        if self.study and key and getattr(self.study.solver, "modal_engine", "auto") != key:
            self.study.solver.modal_engine = key
            self.changed()

    def study_for_solve(self) -> Study:
        """Copy of the study for the calculation. With thermal states the empty load states are left out (a new study has an
        empty "Load state 1" – a study with heat only then does not need constraints).
        """
        st = Study.from_dict(self.study.to_dict())
        if st.thermal.cases or st.modal.enabled or st.dynamics.enabled:   # also a study with only frequencies / dynamics
            st.load_cases = [lc for lc in st.load_cases if lc.loads]
        return st

    def no_static(self) -> bool:
        """Study without statics (no non-empty load state), but with heat or natural frequencies – loads
        are not needed; constraints only for statics (natural frequencies without constraints = free body, only a warning).
        """
        st = self.study
        return st is not None and (bool(st.thermal.cases) or st.modal.enabled or st.dynamics.enabled) and \
            not any(lc.loads for lc in st.load_cases)

    def check(self) -> tuple[list[str], list[str]]:
        """Readiness check of the study as it will be calculated (see study_for_solve)."""
        st = self.study_for_solve()
        errors, warnings = check_study(st, [p.name for p in self.parts])
        if st.shell.enabled and not st.twod.enabled and not self.frame_only():   # shell model (fea_core.shell)
            from fea_core.shell.run import check_study as check_shell
            errors = [e for e in errors if "supports are currently faces only" not in e]   # shells: constraints also on edges
            errors = errors + [e for e in check_shell(st) if e not in errors]
            warnings = warnings + ["Shell model: parts are analyzed as mid-surfaces with thickness (S6) – results “Shells: …”"]
        return errors, warnings

    def rigid_errors(self) -> list[str]:
        """Rigid-motion check errors that block the calculation (see rigid_check)."""
        return self.rigid_check()[0]

    def rigid_check(self) -> tuple[list[str], list[str]]:
        """Rigid-motion check (fea_core.mcp.tools_mesh.rigid_body_check): constraints must prevent 3 displacements and
        3 rotations. A study with heat only and no constraints is not checked (missing constraints are reported by check_study).
        With a mesh exactly from the constraint nodes, without a mesh from the points of the CAD faces; result cached by constraints and selections.
        Returns (errors, warnings): an approximate result (a constraint that the geometric check does not know exactly – "note")
        is only a warning and does not block the calculation – as MCP fea_check_readiness.
        """
        st = self.study
        if st is None or not st.constraints or self.no_static() or not self.parts or st.shell.enabled or \
                st.twod.enabled:                  # 2D / shells: constraints are converted in the core (own check)
            return [], []
        sig = self.study_signature(("constraints", "selections")) + f"|{id(self.mesh)}|{id(self.parts)}"
        if self._rigid is not None and self._rigid[0] == sig:
            return list(self._rigid[1][0]), list(self._rigid[1][1])
        errs: list[str] = []
        warns: list[str] = []
        try:
            from fea_core.mcp.session import Session
            from fea_core.mcp.tools_mesh import rigid_body_check
            if self._rb_sess is None or self._rb_sess._parts is not self.parts:
                self._rb_sess = Session(self._workdir / "rigid")
                self._rb_sess._parts = self.parts          # GUI parts (the cache of face properties stays)
            sess = self._rb_sess
            sess.study = Study.from_dict(st.to_dict())
            sess.mesh = sess.mesh_sig = None
            m = self.mesh
            if m is not None and all(c.selection in m.surfaces for c in st.constraints):
                sess.mesh = m
                sess.mesh_sig = sess.mesh_signature()
            r = rigid_body_check(sess)
            if not r.get("ok", True):
                text = "The supports do not prevent a rigid-body motion of the model – free: " + "; ".join(r["free_rigid_modes"])
                if r.get("note"):
                    warns.append(text + f" ({r['note']} The calculation can be started; if the problem is really free, the solver ends with an error.)")
                else:
                    errs.append(text)
        except Exception as e:  # noqa: BLE001 – the auxiliary check must not block the calculation
            self.warn("study check", f"The rigid-body motion check failed (the analysis can still run): {e}", "Solve")
        self._rigid = (sig, (errs, warns))
        return list(errs), list(warns)

    def refresh_ready(self, check_result=None):
        """check_result = (errors, warnings) of self.check() already computed for this study state (refresh_all) – not computed again."""
        self.ready_list.clear()
        if self.study is None:
            if getattr(self, "messages", None) is not None:
                self.messages.set_source("study check", [])
            return
        errors, warnings = check_result if check_result is not None else self.check()
        if STAGES[getattr(self, "stage", 0)] == "Solve":   # rigid-motion check only in the Solve step (cache)
            rb_err, rb_warn = self.rigid_check()
            errors, warnings = errors + rb_err, warnings + rb_warn
        if self.study.thermal.cases or self.study.modal.enabled or self.study.dynamics.enabled:
            empty = [lc.name for lc in self.study.load_cases if not lc.loads]
            if empty:
                warnings = warnings + [f"An empty load case is skipped during the analysis: {', '.join(empty)}"]
        for e in errors:
            QListWidgetItem(dot_icon(RED), e, self.ready_list)
        for w in warnings:
            QListWidgetItem(dot_icon(ORANGE), w, self.ready_list)
        if not errors:
            QListWidgetItem(dot_icon(GREEN), "The study is ready to run", self.ready_list)
        self.btn_run.setEnabled(not errors and self._thread is None)
        if getattr(self, "messages", None) is not None:     # message table: study check
            self.messages.set_source("study check", [("error", e) for e in errors]
                                     + [("warning", w) for w in warnings]
                                     + ([] if errors else [("info", "The study is ready to run")]))

    def run_solve(self):
        if self.study is None or self._thread is not None:
            return
        errors, _ = self.check()
        rb_err, rb_warn = self.rigid_check()
        errors += rb_err
        if errors:
            msg(self, APP_NAME, "The study is not ready:\n\n" + "\n".join(errors))
            return
        for w in rb_warn:
            self.log_msg(f"Warning: {w}")
        self.study.geometry = self.geometry_str()
        self._thread_sig = self.study_signature()
        self._thread_mesh_sig = self.study_signature(self.MESH_KEYS)
        wd = self._workdir / f"run_{int(time.time() * 1000)}"
        self.log_msg(f"Analysis in folder {wd}")
        mesh = self.valid_mesh()
        t = SolveThread(self.study_for_solve(), wd, self, mesh=mesh,
                        parts=self.solid_parts() if mesh is not None else None)
        t.log.connect(self.log_msg)
        t.solver_line.connect(self.append_solver_line)
        t.progress.connect(self.progress_report)
        t.done.connect(self.on_solve_done)
        t.failed.connect(self.on_solve_failed)
        t.finished.connect(self._thread_finished)
        self._thread = t
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_stop_keep.setEnabled(self.study.is_nonlinear())
        self._stop_keep = False
        self.solve_info.setText("Computing…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self._t_solve = time.time()
        self.progress_start("Study check")
        t.start()

    # ------------------------------------------------------------------ green progress bar (Solve panel + status bar)
    def progress_bars(self):
        return [b for b in (getattr(self, "solve_progress", None), getattr(self, "status_progress", None)) if b]

    def progress_start(self, text: str, end: float = 0.05):
        for b in self.progress_bars():
            b.start(text, end)

    def progress_report(self, value: float, end: float, text: str):
        for b in self.progress_bars():
            b.report(value, end, text)

    def progress_finish(self, state: str, text: str):
        """End of a job: the bar in the Solve panel keeps the result, the status bar clears it after a while."""
        for b in self.progress_bars():
            b.finish(state, text)
        QTimer.singleShot(6000, lambda: self.status_progress.hide() if self._thread is None else None)

    def mesh_progress_finish(self, state: str, text: str):
        self.status_progress.finish(state, text)
        QTimer.singleShot(6000, lambda: self.status_progress.hide() if self._thread is None else None)

    def cancel_solve(self):
        if isinstance(self._thread, SolveThread):
            self._thread.cancel.set()
            kill_mesh_workers()                  # the calculation may be meshing right now (Gmsh cancel is not checked)
            self.log_msg("Canceling analysis…")

    def stop_keep_solve(self):
        """Stops the nonlinear calculation and keeps the last converted increment (the core returns it as a partial
        result – StepResult.partial). The ccx processes are terminated repeatedly until the calculation thread ends (jobs
        waiting in the parallel queue would otherwise start).
        """
        if not isinstance(self._thread, SolveThread):
            return
        from .workers import kill_ccx_processes
        self._stop_keep = True
        self.btn_stop_keep.setEnabled(False)
        self.log_msg("Stopping the nonlinear analysis – the last converged increment will be kept…")
        if getattr(self, "_stop_timer", None) is None:
            self._stop_timer = QTimer(self)
            self._stop_timer.setInterval(300)
            self._stop_timer.timeout.connect(lambda: kill_ccx_processes() if self._thread is not None else None)
        kill_ccx_processes()
        self._stop_timer.start()

    def on_solve_done(self, res):
        if self._defer_done(lambda: self.on_solve_done(res)):
            return
        if self.study is None or self.study_signature() != self._thread_sig:
            self.progress_finish("cancelled", "Result discarded – study changed")
            self.solve_info.setText("The study changed during the analysis – result discarded.")
            self.log_msg("The study changed during the analysis – results discarded, run the analysis again.")
            msg(self, APP_NAME, "The study changed during the analysis – the results would not match it and were discarded.\nRun the analysis again.")
            return
        self.results = res
        self.mesh = res.mesh
        self._mesh_sig = self._thread_mesh_sig
        self.dirty = True
        if self.study.cad_link is not None:           # link with the CAD: results summary into the current iteration
            try:
                from fea_core.cad_link.history import record
                record(self.study, res, self.solid_parts())
            except Exception as e:  # noqa: BLE001 – the iteration summary does not discard the result
                self.log_msg(f"The iteration summary cannot be written: {e}")
        dt = time.time() - self._t_solve
        res._gui_solve_s = dt                          # calculation time for the status bar (not saved)
        self.progress_finish("done", decimal(f"Done in {dt:.1f} s"))
        self.solve_info.setText(f"Done in {dt:.1f} s.")
        self.log_msg(f"Analysis completed in {dt:.1f} s.")
        for th in res.thermal:
            bal = th.balance
            self.log_msg(f"Thermal state {th.name}: T {fmt(float(th.T.min()))} … {fmt(float(th.T.max()))} °C"
                         + (f", balance {fmt(100 * bal['imbalance'], 2)} %" if "imbalance" in bal else ""))
        for w in res.warnings:
            self.log_msg(f"Warning: {w}")
        for ms in res.modes:
            self.log_msg(f"{ms.name}: " + ", ".join(fmt(float(v), 5) for v in ms.values)
                         + (" Hz" if ms.kind == "modal" else " (λ factors)"))
        self.mesh_messages(res.mesh)
        self.messages.set_source("results", [("warning", w) for w in res.warnings], unique=True)
        self.f8_messages(res)
        self.messages.set_source("solver", [("info", decimal(f"Analysis completed in {dt:.1f} s"))])
        self.fat_panel.result = None              # fatigue and joint assessment belong to the previous results
        self.joints_panel.assessment = None
        self.update_title()
        self.refresh_all()
        if res.steps and (self.study.active_bolts() or self.study.active_welds()):
            self.joints_panel.on_results()
        self.set_stage(STAGE["Results"])

    def on_solve_failed(self, text):
        if getattr(self, "_stop_keep", False):      # stopped before the first increment was converted
            text = ("Calculation stopped – no converted increment to keep (stop only after the first increments, or reduce dt₀).\n\n" + text)
        cancelled = "cancelled" in text.lower()
        self.progress_finish("cancelled" if cancelled else "error",
                             "Cancelled" if cancelled else "Analysis failed – see the log for the reason")
        self.solve_info.setText("Analysis failed.")
        self.log_msg(f"ERROR: {text}")
        self.messages.set_source("solver", [("warning" if cancelled else "error",
                                            "Analysis canceled" if cancelled else f"Analysis failed: {text}")])
        if not cancelled:
            self.dock_messages.raise_()
        if self.joints_panel.offer_ignore(text):      # free bodies → offer to exclude parts
            return
        msg(self, APP_NAME, f"Analysis failed:\n\n{text}", QMessageBox.Critical)

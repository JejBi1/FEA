"""Main window of FEA Studio.

Layout ("Studio × Tables" design): at the top the header (menus, phase tabs, phase actions, Run calculation), on the left the shelf of step commands in groups (SHELF), in the centre the view bar (selection, colour, section) above the 3D window, on the right the study tree, the step tabs of the phase, the Properties card and the step panel, at the bottom the table tabs (table_dock), log, messages and the status bar. All calculation logic is in fea_core – the GUI only edits Study and calls the core.

The window is put together from mixins: `chrome.py` (header, shelf, bars), `ipc_bridge.py` (MCP bridge) and the members of the former
single file in `main_project.py` (file life cycle, undo), `main_setup.py` / `main_loads.py` (Model, Materials, Constraints, Loads),
`main_mesh_solve.py`, `main_results.py` / `main_results_tools.py`, `main_refresh.py` (redraw, study tree) and `main_layout.py` (docks,
pop-out windows). The constants and helpers are in `main_defs.py`; every name is re-exported here (panels and tests import from
`fea_gui.main`). This file keeps the construction of the window, the step switch and the start of the program.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget

from fea_core.study import Study

from . import i18n, theme
from .user_settings import app_settings
from .view import FeaView
from .widgets import TaskProgress, compact, help_tip
from .workers import MeshThread, SolveThread, kill_mesh_workers

from .chrome import ChromeMixin
from .ipc_bridge import McpBridgeMixin
from .main_layout import LayoutMixin
from .main_project import ProjectMixin
from .main_setup import SetupMixin
from .main_loads import LoadsMixin
from .main_mesh_solve import MeshSolveMixin
from .main_results import ResultsMixin
from .main_results_tools import ResultsToolsMixin
from .main_refresh import RefreshMixin

# re-exported: panels, tests and scripts import these names from fea_gui.main
from fea_core.study import LoadCase  # noqa: F401
from .main_defs import (  # noqa: F401
    APP_NAME, STAGES, STAGE, PHASES, PHASE_OF, OPTIONAL_STAGES, PHASE_TIPS, ANALYSIS_LABELS, PHASE_LABEL, ElidedLabel,
    HEADER_ACTIONS, element_quality_nodes, phase_label, MATERIAL_COLORS, NO_MATERIAL_COLOR, VIEW_DIRECTIONS, STAGE_TIPS,
    STAGE_LONG, STAGE_COLORS, GREEN, ORANGE, RED, GREY, C_CONSTRAINT, C_FORCE, C_PRESSURE, C_REFINE, BEARING, THERMAL_LOAD,
    ROTATION, FORCE, FRAME_NODE, FRAME_MEMBER, FRAME_ONLY_OFF, SHELF, SHELF_LABELS, P_MODES, LOAD_TIPS, VEC_TIPS,
    LOAD_SHORT, VEC_SHORT, G_STD, C_EDGE, C_ELASTIC, STRESS_ERROR_TIP, STORED_SOLVER, BACKENDS, QUANTITIES, PEEQ_Q, TOPO_Q,
    TOPO_SHAPE_Q, TOPO_MISSING, CONTACT_Q, STATUS_NOTE, THERMAL_QUANTITIES, SAFETY_QTY, NO_MIN_QTY, HS_STRESS, HS_THERMAL,
    HS_TOP, F8_SOURCE, CAD_SOURCE, DROP_SUFFIXES, NATIVE_FILTER, PICK_FILTER_ACTIONS, PICK_FILTER_SHORT, fmt, nice_floor,
    dot_icon, msg, msg_via_main, _prewarm_imports, _remember_ui, _InputBlocker, _busy_progress, _batched,
    form_hide_empty_rows, NearestPoints, _LazyPanel, LAZY_PANELS, _SHELL_ATTRS)


# the declarations of MainWindowHost (type checking only) must match what the mixins and this class implement
# pyright: reportIncompatibleMethodOverride=error
class MainWindow(ChromeMixin, McpBridgeMixin, LayoutMixin, ProjectMixin, SetupMixin, LoadsMixin,
                 MeshSolveMixin, ResultsMixin, ResultsToolsMixin, RefreshMixin, QMainWindow):
    STAGE = STAGE
    STAGES = STAGES
    flowChanged = Signal()           # after restoring the step state (refresh_flow) – setup guide, see flow_state
    jitNotice = Signal(str)          # first use of the numba kernels (fea_core.jitnotice) – emitted from the thread that compiles

    def __init__(self):
        i18n.install(QApplication.instance())        # GUI language (cs | en) – before the windows are created (also QLocale for numbers)
        super().__init__()
        self.settings = app_settings()
        self.dark = theme.apply(QApplication.instance(), str(self.settings.value("theme", "system")), stylesheet=False)
        self.setWindowTitle(APP_NAME)
        self.resize(1600, 950)
        # stav dokumentu
        self.project_path: Path | None = None
        self.step_path: Path | None = None
        self.parts = []
        self.study: Study | None = None
        self.mesh = None
        self.results = None
        self.motion_result = None         # mechanism result (fea_core.motion.MotionResult, Motion step)
        self.model_checks = None          # space.model_checks.CheckRun of the last model quality checks (space_checks dialog, MCP), for the Word report
        self._motion_sig = None           # signature of the mechanism that motion_result belongs to (motion_panel)
        self.face_cache: dict = {}        # part -> [face_props]
        self.sel_faces: dict[str, list] = {}   # selection -> [(part, face)]
        self.dirty = False
        self._thread = None
        self._workdir = Path(tempfile.mkdtemp(prefix="feastudio_"))
        self._undo: list[str] = []
        self._undo_pos = -1
        self._thread_sig = None                   # study signature when the mesh / calculation started (study_signature)
        self._mesh_sig = None                     # signature of the meshing inputs (MESH_KEYS) that self.mesh belongs to
        self._thread_mesh_sig = None
        self._rigid = None                        # cache of the rigid-motion check (signature, result)
        self._rb_sess = None
        self._change_batch = 0                    # batch_changes() depth – one refresh at the end
        self._refresh_pending = False
        self._busy_warned = False                 # "change during a running analysis" asked once per run
        self._busy_task = None                    # run_busy() worker (open, geometry update, joints, MCP tools)
        self._shadow = None                       # (key, parts) – parts re-read for worker threads (never displayed)

        self.view = FeaView()
        self.view.facesPicked.connect(self.on_faces_picked)
        self.view.probed.connect(self.on_probe)
        self.view.partsPicked.connect(self.on_parts_picked)
        self.view.edgePicked.connect(self.on_edge_picked)
        self.view.edgesPicked.connect(lambda _: self.update_status())
        self.view.pointsPicked.connect(lambda _: self.update_status())
        self.sel_edges: dict[str, list] = {}   # edge selection -> [(part, edge)]
        self.sel_points: dict[str, list] = {}  # point selection -> [(part, (x, y, z))]
        self.edge_cache: dict = {}             # part -> [edge_props] (order of geometry.list_edges = FeaView.part_edges)
        self.view.pickNote.connect(lambda t: self.statusBar().showMessage(t, 4000))
        self.view.contextMenuAt.connect(self.view_context_menu)     # right click without a drag (drag = rotate)
        self.view.screen_pick = self.frame_screen_pick              # click on a node / member of the frame
        self.spacemouse = None                    # 3D mouse 3Dconnexion (Windows raw input), see showEvent
        if sys.platform == "win32":
            try:
                from .spacemouse import SpaceMouse
                self.spacemouse = SpaceMouse(self.settings, self)
                self.spacemouse.motion.connect(self._space_motion)
                self.spacemouse.button.connect(self._space_button)
            except Exception:  # noqa: BLE001 – the program works without it
                self.spacemouse = None
        self._point_inputs: list[tuple] = []      # point field [mm] in forms (Use centre of edge), see register_point_input
        self._edge_info = None                    # selected edge (filter Edge)
        self._hs_rows: list = []                  # maximum places in the results table
        self._diag_sess = None
        self._pick_hook = None                    # a click in the results is taken by the dialog (progress along a line, convergence)
        self._aux_threads: list = []              # dialog threads (convergence, print refinement) – busy()
        self._path_dlg = self._conv_dlg = None
        self._r_view = (0.0, None)                # (deformation scale, displacements) of the displayed result
        self.surface_names: set[str] = set()      # plate parts from STEP (shells; the 3D mesh does not mesh them)
        self._voxel_dlg = None

        self._build_tree()
        # containers first into the window, then the panels: moving a finished widget tree under a new parent reapplies
        # the style (QSS) to all children – previously 3× (stack → splitter → window), window start ~2 s slower
        self.panel_stack = QStackedWidget()
        self.panel_stack.setMinimumWidth(460)
        # "Studio" layout: left tool shelf (tb_shelf), centre 3D window, right study tree above the step panel
        side = QSplitter(Qt.Vertical)
        side.setObjectName("side_split")
        side.addWidget(self.tree)
        side_panel = QWidget()                           # step tabs of the phase (tb_steps) above the step panel
        self._side_layout = QVBoxLayout(side_panel)
        self._side_layout.setContentsMargins(0, 0, 0, 0)
        self._side_layout.setSpacing(0)
        self._side_layout.addWidget(self.panel_stack, 1)
        side.addWidget(side_panel)
        side.setStretchFactor(0, 1)                      # tree about a third of the height, step panel the rest
        side.setStretchFactor(1, 2)
        side.setChildrenCollapsible(False)
        center = QWidget()                               # view bar (tb_view) above the 3D window
        self._center_layout = QVBoxLayout(center)
        self._center_layout.setContentsMargins(0, 0, 0, 0)
        self._center_layout.setSpacing(0)
        self._center_layout.addWidget(self.view, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(center)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        self.side_split = side
        self._build_panels()
        self._build_toolbars()
        self._build_docks()
        theme.apply_stylesheet(QApplication.instance())   # QSS once over the finished window (theme.apply)
        split.setSizes([1180, 520])
        side.setSizes([300, 600])
        self.st_sel = QLabel("")
        self.st_info = QLabel("")
        self.statusBar().addWidget(self.st_info, 1)
        self.btn_edge_point = QPushButton("Use center")
        help_tip(self.btn_edge_point, "Circle / arc center (else edge midpoint) to the clipboard and the point field",
                 "Copy the circle / arc center (edge midpoint for other edges) to the clipboard and to the point field of the open form (remote load / support)")
        self.btn_edge_point.clicked.connect(self.use_edge_point)
        self.btn_edge_point.setVisible(False)
        self.statusBar().addPermanentWidget(self.btn_edge_point)
        self.status_progress = TaskProgress()
        self.status_progress.setFixedWidth(340)
        self.statusBar().addPermanentWidget(self.status_progress)
        self.statusBar().addPermanentWidget(self.st_sel)
        self.st_model = QLabel("")
        self.st_model.setObjectName("modelStatus")
        self.st_model.setTextFormat(Qt.PlainText)
        help_tip(self.st_model, "Result equilibrium, mesh element count, solve time and units",
                 "Model summary: largest relative imbalance of reactions and loads over the load cases (√ = below the check limit, otherwise the result is invalid), number of mesh elements, time of the last solve in this window and the program units (mm, N, MPa, t, s).")
        self.statusBar().addPermanentWidget(self.st_model)
        self.set_stage(0)
        self.refresh_all()
        from .tour import install_help                   # Help bar (F1), example guide, manual
        install_help(self)
        from .extras import install_extras              # Tools: parametric study, job queue; View: Python console; recent files
        install_extras(self)
        compact(self.panel_stack)                        # narrower panels (menus, long form rows)
        self._started = True                             # optional panels built later: compact + form tips in _build_lazy_panel
        # Dynamic IPC Bridge (control from an LLM / MCP)
        from fea_core.mcp.session import Session
        self._mcp_session = Session(base_dir=self._workdir)
        self._mcp_tools_cache = None
        from .setup_confirm import install_setup_confirm     # AI calculations ask for a confirmation of the setup (Tools menu)
        install_setup_confirm(self)
        self.ipc_server = None
        import os as _os
        if _os.environ.get("FEA_NO_IPC") != "1":     # tests: the pipe is per user – other FEA Studio clients would reach the test window
            try:
                from .ipc_server import IpcServer
                self.ipc_server = IpcServer(self)
            except Exception as e:  # noqa: BLE001
                self.log_msg(f"Warning: the IPC server could not be started: {e}")
        # the numba kernels are compiled at their first call (nothing is precompiled at installation): say so in the log, from any thread
        from fea_core import jitnotice
        self.jitNotice.connect(self._on_jit_notice)
        self._jit_sink = self.jitNotice.emit
        jitnotice.add_sink(self._jit_sink)
        # dropping .step / .stp / .fea from Explorer anywhere on the window (also over the 3D OCCT window, which is a native window
        # without its own OLE target; Windows passes the drop to the main window and Qt sends it by position: dragEnterEvent below)
        self.setAcceptDrops(True)
        if os.environ.get("FEA_PREWARM") == "1":
            QTimer.singleShot(800, _prewarm_imports)     # result libraries in the background (first display without waiting)

    # ================================================================== 3D mouse, context menu of the 3D window
    def showEvent(self, e):
        super().showEvent(e)
        self._fit_toolbars()
        if not getattr(self, "_docks_sized", False):      # bottom dock low – the 3D window has priority (Studio layout)
            self._docks_sized = True
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self.dock_log], [max(110, int(self.height() * 0.15))], Qt.Vertical))
            from .widgets import _ui_settings, remember_ui
            if remember_ui() and str(_ui_settings().value("ui/popout/panel_detached", False)).lower() in ("1", "true"):
                QTimer.singleShot(0, lambda: self.set_panel_detached(True))      # step panel in its own window as last time
        sm = self.spacemouse
        if sm is not None and not sm.registered:
            try:
                sm.attach(int(self.winId()))      # raw input registration – with no device / driver it does nothing
            except Exception:  # noqa: BLE001
                pass

    def nativeEvent(self, eventType, message):
        sm = self.spacemouse
        if sm is not None and sm.registered:
            try:
                sm.handle(message)
            except Exception:  # noqa: BLE001 – a faulty message must not crash the window
                pass
        return super().nativeEvent(eventType, message)

    def _space_motion(self, v, dt):
        if QApplication.activeModalWidget() is None:
            self.view.space_motion(v, dt)

    def _space_button(self, action):
        if QApplication.activeModalWidget() is not None:
            return
        if action == "fit":
            self.view.fit_all()
        elif action in ("roll_cw", "roll_ccw"):
            self.view.roll(math.pi / 2 if action == "roll_ccw" else -math.pi / 2)
        else:
            self.view.set_orientation(action)

    def spacemouse_settings(self):
        from .spacemouse import SpaceMouseDialog
        if self.spacemouse is not None:
            SpaceMouseDialog(self.spacemouse, self).exec()

    # ================================================================== kroky postupu
    @_batched
    def set_stage(self, k):
        name = STAGES[k]
        if not self.stage_visible(name):             # a hidden optional step (tree, guide, IPC) appears on the bar
            self._flow_revealed.add(name)
        self.stage = k
        self._phase_last[PHASE_OF[name]] = name
        self.stage_actions[k].setChecked(True)
        self._apply_flow_visibility()                # current phase and step (narrow bar) + possibly a new step
        for attr, (stage, _) in LAZY_PANELS.items():
            if stage == name:
                getattr(self, attr)                   # builds the panel at its first display
        self.panel_stack.setCurrentIndex(k)
        self._update_panel_window_title()
        self._update_shelf(name)                     # step commands in the tool shelf
        self.update_view_info()                      # popis pohledu nad 3D oknem
        if hasattr(self, "props_card"):
            self.props_card.schedule()
        if hasattr(self, "table_dock") and self.table_dock.set_stage(name) and name != "Solve" and not self.busy():
            self.dock_tables.raise_()                # bottom dock: the step table (for Solve and during a calculation the log stays)
        if name == "Results" and hasattr(self, "res_overview"):
            self.res_overview.refresh()
        v = self.view
        # the 3D window shows the mesh (Mesh step, "Show mesh") – after the mesh is discarded it returns to the parts (changed)
        self._mesh_view = name == "Mesh" and self.mesh is not None and self.ms_show.isChecked()
        v.show_persist(not (name == "Results" and self.results is not None))   # frame system (persistent objects)
        if name == "Results" and self.results is not None:
            v.show_parts(False)
            v.show_sets(False)
            v.set_mode("results")
            v.clear_section()
            self.show_result()
        elif name == "Mesh" and self.mesh is not None and self.ms_show.isChecked():
            v.show_parts(False)
            v.show_sets(False)
            v.set_mode("faces")
            self.show_mesh()
        elif name == "Fatigue":
            self.fat_panel.refresh()
            if self.fat_panel.result is not None:
                self.fat_panel.show_map()
            else:
                v.clear_results()
                v.show_parts(True)
                v.show_sets(False)
                v.set_mode("model")
        else:
            v.clear_results()
            v.show_parts(True)
            v.show_sets(name in ("Joints", "Contacts", "Constraints", "Motion", "Heat", "Loads", "Dynamics", "Mesh",
                                 "Solve"))
            v.set_mode("faces" if name in ("Joints", "Contacts", "Members", "Constraints", "Motion", "Heat", "Loads",
                                           "Dynamics", "Mesh") else "model")
        if name == "3D printing":
            self.print_panel.legend.hide()
            self.print_panel.refresh()
        if name == "Joints":
            self.joints_panel.compute()
            if hasattr(self, "table_dock"):
                self.table_dock.refresh()                # joint assessment into the Bolts / Welds tables
        if name == "Solve":
            self.refresh_ready()
        if name == "Members":
            self.frame_panel.refresh()
        if name == "Heat":
            self.thermal_panel.update_info()
        if name == "Contacts":
            self.contact_panel.update_info()
        if name == "Dynamics":
            self.dynamics_panel.update_info()
        if name == "Motion":
            self.motion_panel.update_info()
        elif self._built("motion_panel") is not None:
            self.motion_panel.reset_pose()          # mechanism animation only in the Motion step (otherwise the CAD position)
        if name == "Materials":            # parts selected in the list are highlighted in 3D too
            v.set_selected_parts([i.row() for i in self.mat_parts.selectedIndexes()])
        if not (name == "Results" and self.results is not None):
            self.apply_model_section()
        if v.mode != "faces":          # face selection stays between the Constraints / Loads / Mesh steps
            v.set_selected_faces([])
        self.refresh_sets()
        self.update_status()

    def closeEvent(self, e):
        if not self.confirm_discard():
            e.ignore()
            return
        self._closing = True
        try:
            from fea_core import jitnotice
            jitnotice.remove_sink(self._jit_sink)
        except (ImportError, AttributeError):
            pass
        if getattr(self, "ipc_server", None) is not None:
            try:
                self.ipc_server.close()
            except Exception:  # noqa: BLE001
                pass
            self.ipc_server = None
        if self._thread is not None:
            if isinstance(self._thread, (SolveThread, MeshThread)):
                self._thread.cancel.set()
                kill_mesh_workers()
            self._thread.wait(5000)
        for t in list(self._aux_threads):         # convergence, print refinement, dialogs (topology: cancel())
            c = getattr(t, "cancel", None)
            if callable(getattr(c, "set", None)):
                c.set()
            elif callable(c):
                c()
            kill_mesh_workers()
            t.wait(5000)
        super().closeEvent(e)
        # temporary work folder (meshes, calculations, unpacked projects) – removed only on a normal close, remains after a crash
        import shutil
        shutil.rmtree(self._workdir, ignore_errors=True)

def main(app=None, splash=None):
    app = app or QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    def excepthook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        sys.stderr.write(text)
        QApplication.restoreOverrideCursor()
        msg(None, APP_NAME, "Unexpected error:\n\n" + text[-1800:], QMessageBox.Critical)

    sys.excepthook = excepthook
    w = MainWindow()
    from fea_core import i18n as _i18n
    snapshot = _i18n.startup_snapshot()           # translations of the window texts (no study opened yet – no names from files)
    QTimer.singleShot(1500, lambda: _i18n.persist_startup(snapshot))   # the next start does not load the catalogue
    w.show()
    if splash is not None:
        splash.finish(w)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--cad-update" in sys.argv and args:       # CAD add-in without a running GUI: project linked to the document
        QTimer.singleShot(100, lambda: w.cad_update_start(Path(args[0])))
    elif args:
        QTimer.singleShot(100, lambda: w.open_file(args[0]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

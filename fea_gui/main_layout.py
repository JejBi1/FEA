"""Layout of the main window: the view context menu, the study tree, docks and step panels (also the lazily built ones), the log and
message helpers, the pop-out windows.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDockWidget, QHeaderView, QPlainTextEdit, QTreeWidget, QVBoxLayout, QWidget, QFrame, QScrollArea

from .fatigue_panel import FatiguePanel
from .joints_panel import JointsPanel
from .messages import MessagesPanel
from . import theme
from .thermal_panel import ThermalPanel
from .widgets import compact, help_tip, scroll_body

from .main_defs import APP_NAME, STAGES, STAGE, FRAME_NODE, FRAME_MEMBER, _LazyPanel, LAZY_PANELS, _SHELL_ATTRS


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class LayoutMixin(_Base):
    """Layout of the main window: the view context menu, the study tree, docks and step panels (also the lazily built ones),
    the log and message helpers, the pop-out windows."""

    def build_view_menu(self):
        """Context menu of the 3D window (right click without a drag): show all, zoom to selection, views, 3D mouse."""
        from PySide6.QtWidgets import QMenu
        v = self.view
        m = QMenu(self)
        m.addAction("Fit all\tF", v.fit_all)
        a = m.addAction("Zoom to selection (double-click)", v.zoom_selection)
        a.setEnabled(v.has_selection())
        if v.mode == "faces" and (v.selected_faces() or v.selected_edges() or v.selected_points()):
            def clear():
                v.clear_selection()
                self.update_status()
            m.addAction("Clear selection", clear)

        # context actions for a body / part (exclude from the calculation in the Model section)
        sel_parts = v.selected_parts() if hasattr(v, "selected_parts") else []
        cur_pi = None
        if hasattr(v, "_part_at_cursor"):
            cur_pi = v._part_at_cursor()
        if cur_pi is None and hasattr(v, "_face_at_cursor"):
            cur_face = v._face_at_cursor()
            if cur_face is not None and 0 <= cur_face[0] < len(self.parts):
                cur_pi = cur_face[0]

        if not sel_parts and cur_pi is not None and 0 <= cur_pi < len(self.parts):
            v.set_selected_parts([cur_pi])
            sel_parts = [cur_pi]

        target_parts = [self.parts[i].name for i in sel_parts if 0 <= i < len(self.parts)]

        if self.study and target_parts:
            m.addSeparator()
            ign = set(self.study.ignored_parts)
            all_ignored = all(p in ign for p in target_parts)
            if len(target_parts) == 1:
                pname = target_parts[0]
                if all_ignored:
                    m.addAction(f"Include part '{pname}' in the analysis", lambda parts=target_parts: self.set_ignored(parts, False))
                else:
                    m.addAction(f"Exclude part '{pname}' from the analysis", lambda parts=target_parts: self.set_ignored(parts, True))
            else:
                if all_ignored:
                    m.addAction(f"Include selected parts ({len(target_parts)}) in the analysis", lambda parts=target_parts: self.set_ignored(parts, False))
                else:
                    m.addAction(f"Exclude selected parts ({len(target_parts)}) from the analysis", lambda parts=target_parts: self.set_ignored(parts, True))

        m.addSeparator()
        views = m.addMenu("View")
        for key, txt in (("iso", "Isometric"), ("front", "Front"), ("back", "Back"), ("top", "Top"),
                         ("bottom", "Bottom"), ("left", "Left"), ("right", "Right")):
            views.addAction(txt, lambda k=key: v.set_orientation(k))
        if self.spacemouse is not None:
            m.addSeparator()
            m.addAction("3D mouse (3Dconnexion)…", self.spacemouse_settings)
        return m

    def frame_screen_pick(self, x, y) -> bool:
        """Click in 3D on a node / member of the frame: Frames step = table row, Loads step with a frame load = node / member into the form.
        True = the click is handled (face selection does not change).
        """
        if self.study is None or not self.study.frame.nodes:
            return False
        stage = STAGES[getattr(self, "stage", 0)]
        t = self.ld_type.currentText() if hasattr(self, "ld_type") else ""
        if stage not in ("Members", "Loads") or (stage == "Loads" and t not in (FRAME_NODE, FRAME_MEMBER)):
            return False
        hit = self.frame_panel.pick(x, y)
        if hit is None:
            return False
        kind, name = hit
        if stage == "Members":
            self.frame_panel.select(kind, name)
        elif kind == "node" and t == FRAME_NODE:
            self.ld_fr_node.setCurrentText(name)
        elif kind == "member" and t == FRAME_MEMBER:
            self.ld_fr_member.setCurrentText(name)
        else:
            return False
        self.statusBar().showMessage(("Node " if kind == "node" else "Member ") + name, 4000)
        return True

    def frame_assessment_changed(self):
        """Inputs of the EN 1993 assessment changed – FEM results stay valid, the assessment is recomputed."""
        if self.results is not None and STAGES[self.stage] == "Results":
            self.show_result()

    def view_context_menu(self, pos):
        self.build_view_menu().exec(pos)

    # ================================================================== stavba UI
    def _act(self, text, slot, shortcut=None, tip=None, icon=None):
        a = QAction(text, self)
        if icon:
            theme.set_icon(a, icon)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip((tip or text) + (f"  [{shortcut}]" if shortcut else ""))
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_tree(self):
        self.tree = QTreeWidget()
        self.tree.setMinimumHeight(110)               # above the step panel (Studio layout)
        self.tree.setHeaderLabels(["Study", "Info"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.itemClicked.connect(self.on_tree_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_menu)
        help_tip(self.tree, "Study tree – a click switches to the step, the dot = item state",
                 "Study tree: parts, supports, load cases, mesh and results; the dot = item state (green done, orange warnings, red error). A click switches to the corresponding step.<br>Right-click a part to exclude the body from the analysis.")

    def _build_docks(self):
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(theme.mono_font())
        self.log.setWhatsThis("<b>Solver log</b><p>Progress of meshing and analysis (Gmsh and CalculiX output), warnings and errors. If the analysis fails, the reason is here – most often a missing support (free body) or material. An overview of errors and warnings is in the “Messages” dock.</p>")
        self._solver_buf: list[str] = []                 # ccx lines are appended in batches (solver_line → timer)
        self._solver_timer = QTimer(self)
        self._solver_timer.setSingleShot(True)
        self._solver_timer.setInterval(100)
        self._solver_timer.timeout.connect(self.flush_solver_lines)
        from .workers import TRACE
        TRACE.trace.connect(self.flush_and_log_trace)
        from .table_dock import PhaseTables             # table by step (parts, loads, mesh, results)
        self.table_dock = PhaseTables(self)
        dt = QDockWidget("Tables", self)
        dt.setObjectName("dock_tables")
        dt.setWidget(self.table_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, dt)
        self.dock_tables = dt
        d = QDockWidget("Solver log", self)
        d.setObjectName("dock_log")
        d.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)
        self.tabifyDockWidget(dt, d)
        self.resizeDocks([d], [130], Qt.Vertical)
        dj = QDockWidget("Joints – utilization", self)          # tables of bolts and welds (colour by utilisation)
        dj.setObjectName("dock_joints")
        dj.setWidget(self.joints_panel.tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, dj)
        self.tabifyDockWidget(d, dj)
        dj.hide()                                         # tables of bolts and welds are in the tabs of the Tables dock
        self.messages = MessagesPanel(self)               # message table (check, mesh, solver, results)
        dm = QDockWidget("Messages", self)
        dm.setObjectName("dock_messages")
        dm.setWidget(self.messages)
        self.messages.dock = dm
        self.addDockWidget(Qt.BottomDockWidgetArea, dm)
        self.tabifyDockWidget(d, dm)
        self.resizeDocks([d, dj, dm], [170, 170, 170], Qt.Vertical)
        d.raise_()
        self.dock_log, self.dock_joints, self.dock_messages = d, dj, dm
        dt.topLevelChanged.connect(self._on_tables_floating)
        from .popout import CloseToDock
        self._tables_close = CloseToDock(dt)              # closing the floating Tables window docks it back
        for dk in (dt, d, dj, dm):                      # bottom docks: the title is the dock tab (proposal – no title bar)
            if self.dockWidgetArea(dk) == Qt.BottomDockWidgetArea:
                dk.setTitleBarWidget(QWidget())

    def _build_panels(self):
        self.joints_panel = JointsPanel(self)
        self.thermal_panel = ThermalPanel(self)
        panels = {stage: QWidget() for stage, _ in LAZY_PANELS.values()}     # placeholders of the lazily built panels
        for name, b in (("Model", self._panel_model), ("Materials", self._panel_materials),
                        ("Joints", self.joints_panel),
                        ("Constraints", self._panel_constraints), ("Heat", self.thermal_panel),
                        ("Loads", self._panel_loads), ("Mesh", self._panel_mesh), ("Solve", self._panel_solve),
                        ("Results", self._panel_results)):
            if isinstance(b, QWidget):
                panels[name] = b
                continue
            w = QWidget()
            _, lay = scroll_body(w)           # scroll bar – the panel is below the tree
            b(lay)
            lay.addStretch(1)
            panels[name] = w
        self.fat_panel = FatiguePanel(self)
        panels["Fatigue"] = self.fat_panel
        for name in STAGES:                 # panel order = step order (index only via STAGE)
            self.panel_stack.addWidget(panels[name])
        assert self.panel_stack.count() == len(STAGES)

    # attributes of the lazily built panels (3D printing, Members, Contacts – F8, Dynamics – F8, Motion – F8)
    print_panel = _LazyPanel("print_panel")      # pyright: ignore[reportAssignmentType]  # descriptor: the attribute yields the built panel

    frame_panel = _LazyPanel("frame_panel")      # pyright: ignore[reportAssignmentType]  # descriptor: the attribute yields the built panel

    contact_panel = _LazyPanel("contact_panel")      # pyright: ignore[reportAssignmentType]  # descriptor: the attribute yields the built panel

    dynamics_panel = _LazyPanel("dynamics_panel")      # pyright: ignore[reportAssignmentType]  # descriptor: the attribute yields the built panel

    motion_panel = _LazyPanel("motion_panel")      # pyright: ignore[reportAssignmentType]  # descriptor: the attribute yields the built panel

    def _build_lazy_panel(self, attr: str):
        """Builds the panel of an optional step in place of its placeholder (same index in the stack, same post-processing as at start)."""
        if attr in self.__dict__:
            return self.__dict__[attr]
        stage, cls = LAZY_PANELS[attr]
        panel = cls(self)
        self.__dict__[attr] = panel
        k = STAGE[stage]
        stack = self.panel_stack
        current = stack.currentIndex()
        old = stack.widget(k)
        stack.insertWidget(k, panel)
        stack.removeWidget(old)
        old.deleteLater()
        if current == k:
            stack.setCurrentIndex(k)
        if getattr(self, "_started", False):          # built after the start: what __init__ did for the other panels
            compact(panel)
            if getattr(self, "_form_tips", None) is not None:
                self._form_tips.add(panel)
            if self.study is not None:
                panel.refresh()
        return panel

    def _built(self, attr: str):
        """The panel if it is already built, otherwise None (refreshing a panel nobody has seen is not needed – it refreshes at its build)."""
        return self.__dict__.get(attr)

    def _panel_color(self, attr: str, *args) -> str:
        """Step colour without building the panel: color() of a panel reads only the study (through win)."""
        panel = self._built(attr)
        if panel is not None:
            return panel.color(*args)
        cls = LAZY_PANELS[attr][1]
        shell_type = type("_Shell", (), {n: cls.__dict__[n] for n in _SHELL_ATTRS if n in cls.__dict__})
        shell = shell_type()
        shell.win = self
        return shell.color(*args)

    # ------------------------------------------------------------------ Materials panel
    @staticmethod
    def _scrolled(outer):
        """Content of the panel in a scroll area (without a horizontal scroll bar); returns the content layout."""
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(inner)
        outer.addWidget(sc, 100)
        return lay

    # ================================================================== other
    def _on_jit_notice(self, text: str) -> None:
        """Message of fea_core.jitnotice (GUI thread): into the log and the status line, so that the first fatigue / print / picture
        does not look stuck."""
        self.log_msg(text)
        self.statusBar().showMessage(text, 20000)

    def log_msg(self, text):
        if self._solver_buf:                     # keep the order with buffered solver lines
            self.flush_solver_lines()
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def append_solver_line(self, line: str):
        """A line of the solver output (ccx, Gmsh) from a worker thread: buffered, the log is updated at most every 100 ms –
        appending every ccx line separately kept the GUI thread busy during long calculations.
        """
        if not line.strip():
            return
        self._solver_buf.append(line)
        if not self._solver_timer.isActive():
            self._solver_timer.start()

    def flush_solver_lines(self):
        if self._solver_timer.isActive():
            self._solver_timer.stop()
        if self._solver_buf:
            text, self._solver_buf = "\n".join(self._solver_buf), []
            self.log.appendPlainText(text)

    def flush_and_log_trace(self, text: str):
        """Traceback of a failed background job (workers.report_trace) – into the solver log only, the user sees the short message."""
        self.flush_solver_lines()
        self.log.appendPlainText(text)

    def warn(self, source: str, text: str, stage: str | None = None):
        """A problem the user should see: the solver log and a warning in the Messages dock."""
        self.log_msg(f"Warning: {text}")
        if getattr(self, "messages", None) is not None:
            self.messages.add("warning", source, text, stage)

    # ================================================================== separate windows (step panel, Tables dock)
    def panel_detached(self) -> bool:
        p = getattr(self, "_panel_pop", None)
        return p is not None and p.is_detached()

    def set_panel_detached(self, on: bool) -> None:
        """Step panel (panel_stack) into its own window and back; the step tabs and the 3D view stay in the main window."""
        if bool(on) == self.panel_detached():
            if on:
                self._panel_pop.raise_window()
            return
        if getattr(self, "_panel_pop", None) is None:
            from .popout import Popout
            self._panel_pop = Popout(self.panel_stack, "Step panel", "panel", owner=self,
                                     note="The step panel is open in a separate window.", default_size=(620, 860))
            self._panel_pop.changed.connect(self._on_panel_detached)
        if on:
            self._panel_pop.detach()
        else:
            self._panel_pop.attach()

    def _on_panel_detached(self, detached: bool) -> None:
        from .widgets import _ui_settings, remember_ui
        from . import theme
        a = getattr(self, "act_panel_window", None)
        if a is not None and a.isChecked() != detached:
            a.blockSignals(True)
            a.setChecked(detached)
            a.blockSignals(False)
        b = getattr(self, "btn_panel_popout", None)
        if b is not None:
            theme.set_icon(b, "dock_in" if detached else "popout")
            b.setToolTip("Return the step panel into the main window" if detached
                         else "Step panel in a separate window (larger, also on a second monitor)")
        self.panel_stack.setMinimumWidth(0 if detached else 460)
        if detached:
            self._update_panel_window_title()
            sizes = self.side_split.sizes()          # the tree gets the side space, the placeholder stays small
            self.side_split.setSizes([sum(sizes) - 90, 90])
        else:
            self.side_split.setSizes([300, 600])
        if remember_ui() and not getattr(self, "_closing", False):
            s = _ui_settings()
            s.setValue("ui/popout/panel_detached", bool(detached))
            s.sync()

    def _update_panel_window_title(self) -> None:
        p = getattr(self, "_panel_pop", None)
        if p is not None and p.window is not None:
            p.window.setWindowTitle(f"{STAGES[self.stage]} – {APP_NAME}")

    def set_tables_floating(self, on: bool) -> None:
        """Bottom Tables dock as a floating window (with a title bar – it can be dragged back to the edge) and back."""
        d = self.dock_tables
        if bool(on) == d.isFloating():
            if on:
                d.raise_()
            return
        if on:
            w, h = max(d.width(), 900), max(d.height() + 120, 380)
            d.setTitleBarWidget(None)               # own title bar: moving, docking back by dragging, closing
            d.setFloating(True)
            d.resize(w, h)
            d.show()
            d.raise_()
        else:
            d.setFloating(False)

    def _on_tables_floating(self, floating: bool) -> None:
        from . import theme
        d = self.dock_tables
        if not floating:
            d.setTitleBarWidget(QWidget())          # docked: the tab is the title (no title bar)
        a = getattr(self, "act_tables_window", None)
        if a is not None and a.isChecked() != floating:
            a.blockSignals(True)
            a.setChecked(floating)
            a.blockSignals(False)
        b = getattr(self.table_dock, "btn_popout", None)
        if b is not None:
            theme.set_icon(b, "dock_in" if floating else "popout")
            b.setToolTip("Return the tables to the bottom of the main window" if floating
                         else "Tables in a separate window")

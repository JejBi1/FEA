"""Confirmation of the setup before an AI calculation: the dialog of the main window (F15-S1).

When an AI assistant calls `fea_setup_review` and the window asks before AI calculations (Tools → "Ask before an AI calculation",
on by default), the session of the window (`Session.confirm_ui`) opens this non-modal dialog: the numbered picture, the table of
the supports and loads (the same numbers as in the picture), and the buttons Confirm / Reject / Show in 3D. Clicking a row marks
its faces in the 3D window and zooms to them; while the dialog is open the numbered face sets are drawn in the 3D window too.

Only a click on Confirm here makes the setup "confirmed by the user": `setup_proof.record_decision` is called from this module only
(the tools never take a decision from an argument), and `tools_setup.gate` accepts nothing else while a window is open. The digest is
computed again at the click – a setup that was changed since the review (by the user in the window, or by the assistant) is not
confirmed. The window's own calculations (Run) never ask.

The dialog is built lazily: `import fea_gui.main` only wires `install_setup_confirm` (a menu action and the session hook).
"""
from __future__ import annotations

import html
from typing import Any

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Slot
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
                               QSplitter, QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core.mcp import setup_proof

from . import theme
from .widgets import help_tip, table

SETTING = "mcp/confirm_ai"            # QSettings: ask before an AI calculation (absent = on)
KEY_PREFIX = "setupconf:"             # keys of the numbered face sets in the 3D window
COLUMNS = ["No.", "Kind", "Selection", "Faces", "Value", "Cases"]


def cell_tip(text: str) -> str:
    """Tooltip of a table cell: Qt shows a text that looks like HTML as HTML, and the cells hold names from the study / CAD files –
    so the tooltip is always rich text with the cell text escaped."""
    return "<p style='white-space:pre-wrap'>" + html.escape(str(text)) + "</p>"


def ask_setting(settings) -> bool:
    """The stored choice "Ask before an AI calculation" (default: yes)."""
    v = settings.value(SETTING, True)
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def current_digest(main) -> str:
    """The digest of the setup that is in the window NOW (the session is handed the window's study, shapes and file, as the IPC
    bridge does before a tool). Raises when the session is busy (a calculation is running) or nothing is open."""
    if main.study is None:
        raise RuntimeError("No study is open in the window")
    sess = main._mcp_session
    with sess.use(wait=2.0):
        sess.study = main.study
        sess._parts = main.parts
        sess.step_path = main.step_path
        return setup_proof.session_digest(sess)


class SetupConfirmDialog(QDialog):
    """The review of one setup (payload of `tools_setup.build_review`): picture, table, decision."""

    def __init__(self, host: GuiConfirmHost):
        super().__init__(host.main)
        self.host = host
        self.payload: dict = {}
        self.stale = False
        self.setWindowTitle("Confirm the setup before the calculation")
        self.setModal(False)
        self.resize(1100, 900)
        lay = QVBoxLayout(self)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        self.head.setTextFormat(Qt.TextFormat.PlainText)                   # names of the study come from files
        lay.addWidget(self.head)
        split = QSplitter(Qt.Orientation.Vertical)
        self.scroller = QScrollArea()
        self.scroller.setWidgetResizable(True)
        self.scroller.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroller.setMinimumHeight(280)
        self.picture = QLabel("")
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroller.setWidget(self.picture)
        split.addWidget(self.scroller)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        self.table = table(COLUMNS, 110)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setProperty("_i18n_done", True)
        help_tip(self.table, "Supports and loads of the setup – click a row to see its faces in the 3D window")
        self.table.itemSelectionChanged.connect(self.show_selected)
        bl.addWidget(self.table, 1)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        bl.addWidget(self.summary)
        split.addWidget(box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([570, 230])
        split.setChildrenCollapsible(False)
        lay.addWidget(split, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self.status)
        h = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        theme.set_primary(self.btn_confirm)
        help_tip(self.btn_confirm, "Yes, calculate exactly this setup (numbers as in the picture)")
        self.btn_confirm.clicked.connect(lambda: self.decide("confirmed"))
        h.addWidget(self.btn_confirm)
        self.btn_reject = QPushButton("Reject")
        help_tip(self.btn_reject, "No – the assistant is told to ask what to change")
        self.btn_reject.clicked.connect(lambda: self.decide("rejected"))
        h.addWidget(self.btn_reject)
        self.btn_show = QPushButton("Show in 3D")
        help_tip(self.btn_show, "Mark the faces of the selected row in the 3D window and zoom to them")
        self.btn_show.clicked.connect(self.show_selected)
        h.addWidget(self.btn_show)
        h.addStretch(1)
        b = QPushButton("Close")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)

    # ------------------------------------------------------------------ content
    def load(self, payload: dict) -> None:
        self.payload = payload
        self.stale = False
        rows = payload.get("rows", [])
        self.head.setText(f"The AI assistant wants to calculate the study “{payload.get('study', '')}”. Check the numbered picture and the "
                          f"table – confirm only if this is exactly what should be computed.  (setup {str(payload.get('digest', ''))[:12]})")
        pix = QPixmap(str(payload.get("image", "")))
        if not pix.isNull():
            self.picture.setPixmap(pix.scaledToWidth(min(pix.width(), 980), Qt.TransformationMode.SmoothTransformation))
        else:
            self.picture.setText("(the picture is not available)")
        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            vals = [str(r["no"]), r["kind"], r.get("selection") or "", setup_proof.faces_cell(r), r["value"], ", ".join(r["cases"])]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0 and r.get("color"):
                    col = QColor(*(int(255 * x) for x in r["color"]))
                    it.setBackground(col)
                    it.setForeground(QColor("white") if col.lightness() < 150 else QColor("black"))
                it.setToolTip(cell_tip(v))
                self.table.setItem(i, c, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.summary.setText("\n".join(payload.get("summary", [])))
        self.host.draw_numbers(rows)
        rec = setup_proof.current_record(self.host.main._mcp_session, str(payload.get("digest")))
        self.set_state(rec["decision"] if rec and rec.get("by") == "user" else None, rec)

    def set_state(self, decision: str | None, rec: dict | None = None) -> None:
        if self.stale:
            self.status.setText("The setup has changed since this review – it is NOT confirmed. Ask the assistant for a new review.")
            self.btn_confirm.setEnabled(False)
            self.btn_reject.setEnabled(False)
            return
        when = (rec or {}).get("at", "")
        if decision == "confirmed":
            self.status.setText(f"Confirmed by you ({when}). The assistant may start the calculation.")
        elif decision == "rejected":
            self.status.setText(f"Rejected by you ({when}). The assistant will ask what to change.")
        else:
            self.status.setText("Waiting for your decision. The calculation does not start before you confirm.")
        self.btn_confirm.setEnabled(decision != "confirmed")
        self.btn_reject.setEnabled(decision != "rejected")

    # ------------------------------------------------------------------ actions
    def decide(self, decision: str) -> None:
        main = self.host.main
        try:
            now = current_digest(main)
        except Exception as e:  # noqa: BLE001 – a busy session / nothing open: tell the user, decide nothing
            self.status.setText(f"The decision cannot be recorded now: {e}")
            return
        if now != self.payload.get("digest"):
            self.stale = True
            self.set_state(None)
            return
        rec = setup_proof.record_decision(main._mcp_session, now, decision, "user", "gui")
        self.set_state(decision, rec)
        main.statusBar().showMessage("Setup " + ("confirmed" if decision == "confirmed" else "rejected") + " – the assistant has been told", 6000)

    def show_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows:
            self.host.show_item(self.payload.get("rows", [])[rows[0].row()])

    def closeEvent(self, e) -> None:
        self.host.clear_numbers()
        super().closeEvent(e)


class GuiConfirmHost(QObject):
    """`Session.confirm_ui` of the window: opens the dialog (in the GUI thread) and draws the numbers in the 3D window."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.dlg: SetupConfirmDialog | None = None
        self._pending: dict | None = None
        self._keys: list[str] = []

    def open_review(self, payload: dict) -> None:
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():      # any other thread: the GUI thread opens it
            self._pending = payload
            QMetaObject.invokeMethod(self, "_open_pending", Qt.ConnectionType.QueuedConnection)
            return
        self._show(payload)

    @Slot()
    def _open_pending(self) -> None:
        if self._pending is not None:
            payload, self._pending = self._pending, None
            self._show(payload)

    def _show(self, payload: dict) -> None:
        if self.dlg is None:
            self.dlg = SetupConfirmDialog(self)
            self.dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.dlg.destroyed.connect(lambda *_: setattr(self, "dlg", None))
        self.dlg.load(payload)
        self.dlg.show()
        self.dlg.raise_()
        self.main.statusBar().showMessage("The AI assistant asks you to confirm the setup of the calculation", 8000)

    # ------------------------------------------------------------------ the 3D window
    def draw_numbers(self, rows: list[dict]) -> None:
        """The faces of every row as a coloured set labelled with its number (the same colours as in the picture)."""
        self.clear_numbers()
        m = self.main
        v = m.view
        try:
            for r in rows:
                sels = [s.strip() for s in str(r.get("selection") or "").split(",") if s.strip()]
                faces = [f for s in sels for f in m.sel_faces.get(s, [])]
                if not faces or not r.get("color"):
                    continue
                key = f"{KEY_PREFIX}{r['no']}"
                v.set_face_set(key, faces, tuple(int(255 * x) for x in r["color"]), label=str(r["no"]), transparency=0.35)
                self._keys.append(key)
            if self._keys:
                v.show_sets(True)
        except Exception as e:  # noqa: BLE001 – the labels in the 3D window are a help; the picture of the dialog is the proof
            m.log_msg(f"Setup review: the numbers could not be drawn in the 3D window: {e}")

    def clear_numbers(self) -> None:
        v = self.main.view
        for k in self._keys:
            try:
                v.remove_face_set(k)
            except Exception:  # noqa: BLE001 – the window may be closing
                pass
        self._keys = []

    def show_item(self, row: dict) -> None:
        """Marks what the row stands for in the 3D window (faces, edges or points of its selection) and zooms to it."""
        m = self.main
        v = m.view
        sels = [s.strip() for s in str(row.get("selection") or "").split(",") if s.strip()]
        faces = [f for s in sels for f in m.sel_faces.get(s, [])]
        edges = [e for s in sels for e in m.sel_edges.get(s, [])]
        points = [p for s in sels for p in m.sel_points.get(s, [])]
        v.set_selected_faces(faces)
        v.set_selected_edges(edges)
        v.set_selected_points(points)
        v.zoom_selection()                          # without a selection: the whole model
        v.update()


def set_confirm_enabled(main, on: bool) -> None:
    """Applies and stores the choice "Ask before an AI calculation"."""
    main._mcp_session.require_confirmation = bool(on)
    main.settings.setValue(SETTING, bool(on))
    a = getattr(main, "act_confirm_ai", None)
    if a is not None and a.isChecked() != bool(on):
        a.blockSignals(True)
        a.setChecked(bool(on))
        a.blockSignals(False)


def install_setup_confirm(main: Any) -> None:
    """Wired at the end of MainWindow.__init__ (after the MCP session exists): the session hook and Tools → "Ask before an AI
    calculation"."""
    host = main._setup_confirm = GuiConfirmHost(main)
    sess = main._mcp_session
    sess.confirm_ui = host
    sess.require_confirmation = ask_setting(main.settings)
    menus = [b.menu() for b in (getattr(main, "menu_buttons", {}).get("Tools"), getattr(main, "tools_button", None))
             if b is not None and b.menu() is not None]
    a = main.act_confirm_ai = main._act("Ask before an AI calculation", lambda: None,
                                        tip="Confirm the setup before an AI assistant starts a calculation")
    a.setCheckable(True)
    a.setChecked(bool(sess.require_confirmation))
    a.toggled.connect(lambda on: set_confirm_enabled(main, on))
    a.setWhatsThis("When an AI assistant (MCP) wants to calculate, it first shows you a numbered picture and a table of the supports and "
                   "loads; the calculation starts only after you click Confirm in the dialog. Switch off to let the assistant calculate "
                   "without asking. Calculations you start yourself (Run) never ask.")
    seen: set[int] = set()
    for m in menus:
        if id(m) in seen:
            continue
        seen.add(id(m))
        m.addSeparator()
        m.addAction(a)

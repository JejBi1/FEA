"""Test correlation dashboard (F14 stage S5, `fea_core.space.correlation`): import of a vibration-test file (CSV / UFF 58 / 55) on a worker thread, the sensor map to the FE model, the
setup of the measured functions (frame, normalisation, unit) and the result as a verdict, the PASS / FAIL table, the MAC heat map with the mode pairs and the overlay of the measured
and the predicted response curves.

Every limit comes from the requirement set that the user entered and confirmed (Tools - Space structures); this dialog has no field for a limit and no way to confirm anything -
without a confirmed set every row is UNCONFIRMED. The verdict shown is never the stored one on its own word: `corr_run.verify_stored` compares the limits of the study, the test file
and the FE modes with what the result was computed from (a stale result is OUTDATED / UNCONFIRMED_REQUIREMENTS, a result whose inputs are not at hand is UNVERIFIABLE). The limits of
the requirement set (value, unit, not applicable, source) are shown read-only on the Limits tab. The heavy modules (matplotlib, scipy) are imported at the first
drawing / correlation only. All text that comes from a test file or a study file (node labels, function names, probe names) is shown as plain text.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSplitter, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from fea_core.space import corr_run as CR
from fea_core.space import correlation as C

from .space_corr_functions import FunctionSetupTable
from .widgets import LazyPlot, dspin, fmt, help_tip, table

GREEN, RED, AMBER, GREY = QColor("#1b7f3b"), QColor("#c0392b"), QColor("#b36b00"), QColor("#6b6b6b")
STATUS_COLOUR = {C.PASS: GREEN, C.FAIL: RED, C.UNCONFIRMED: AMBER, C.VERDICT_UNCONFIRMED: AMBER, C.VERDICT_PARTIAL: AMBER, C.VERDICT_OUTDATED: AMBER, C.VERDICT_UNVERIFIABLE: AMBER,
                 C.NOT_APPLICABLE: GREY}
LIMIT_COLUMNS = ["Limit", "Value", "Unit", "Not applicable", "Document", "Revision", "Clause", "Status", "Meaning", "Hint of the standard (not normative)"]
SENSOR_COLUMNS = ["Test node", "x [mm]", "y [mm]", "z [mm]", "Probe / selection", "DOFs"]
MAX_TABLE_ROWS = 500
_VERDICT_NOTE = {C.VERDICT_UNCONFIRMED: " - a limit, a comparison or the confirmation of the requirement set is missing; not an ECSS result",
                 C.VERDICT_PARTIAL: " - part of the test data was left out of the correlation (see below); not a PASS of the test",
                 C.VERDICT_OUTDATED: " - the stored result is not current; correlate again",
                 C.VERDICT_UNVERIFIABLE: " - the solved results or the test data the result was computed from are not at hand; import, solve and correlate again"}


def _item(text: Any, editable: bool = False, right: bool = False, color: QColor | None = None) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    flags = it.flags()
    it.setFlags(flags | Qt.ItemFlag.ItemIsEditable if editable else flags & ~Qt.ItemFlag.ItemIsEditable)
    if right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color is not None:
        it.setForeground(color)
    return it


def _pct(v: float | None) -> str:
    return "–" if v is None else fmt(100.0 * v, 3)


def _plain_label(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setTextFormat(Qt.TextFormat.PlainText)
    lab.setWordWrap(True)
    return lab


class CorrelationDialog(QDialog):
    def __init__(self, win: Any):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Test correlation (ECSS)")
        self.setModal(False)
        self.resize(1080, 780)
        self.result: C.CorrelationResult | None = None
        self.test: C.TestData | None = getattr(win, "_corr_test", None)
        self.tam: C.MassMatrix | None = getattr(win, "_corr_tam", None)
        self._thread: Any = None
        self._close_pending = False              # the dialog was closed while a worker ran: it closes when the worker has finished (`done`)
        self._run_ctx: tuple[Any, Any, str, str, bool] | None = None
        lay = QVBoxLayout(self)
        intro = QLabel("Correlation of the FE modes and responses with a vibration test. Every metric is judged only against the limits of the requirement set "
                       "that you entered and confirmed (Tools - Space structures); without them the numbers are shown and every row is UNCONFIRMED.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addLayout(self._build_controls())
        lay.addLayout(self._build_import_options())
        self.l_data = _plain_label()
        lay.addWidget(self.l_data)
        self.l_req = _plain_label()
        lay.addWidget(self.l_req)
        self.l_verdict = _plain_label()
        lay.addWidget(self.l_verdict)
        self.l_cov = _plain_label()
        lay.addWidget(self.l_cov)
        self.l_na = _plain_label()
        lay.addWidget(self.l_na)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_summary(), "PASS / FAIL")
        self.tabs.addTab(self._build_mac(), "MAC")
        self.tabs.addTab(self._build_response(), "Response curves")
        self.tabs.addTab(self._build_sensors(), "Sensors")
        self.tabs.addTab(self._build_functions(), "Functions")
        self.tabs.addTab(self._build_limits(), "Limits")
        lay.addWidget(self.tabs, 1)
        h = QHBoxLayout()
        h.addStretch(1)
        b = QPushButton("Close")
        help_tip(b, "Close the dialog")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)
        self.refresh()

    # ------------------------------------------------------------------ access
    @property
    def study(self) -> Any:
        return self.win.study

    @property
    def results(self) -> Any:
        return getattr(self.win, "results", None)

    def _warn(self, text: str) -> None:
        from .main_defs import msg_via_main
        msg_via_main(self, "Test correlation (ECSS)", text)

    def _changed(self, damping: bool = False) -> None:
        """The study changed. Only the project file changes for the sensors / setups / stored result (the results stay valid); a damping table written into the dynamics analysis
        goes the way of every edit of the dynamics (`MainWindow.changed()`), so the results that depend on it are discarded."""
        if damping:
            changed = getattr(self.win, "changed", None)
            if callable(changed):
                changed()
                return
        mark = getattr(self.win, "mark_changed", None)
        if callable(mark):
            mark("none")

    # ------------------------------------------------------------------ layout
    def _build_controls(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self.b_import = QPushButton("Import test data…")
        help_tip(self.b_import, "Read a CSV or UFF (dataset 58 functions, 55 mode shapes) file")
        self.b_import.clicked.connect(self.import_file)
        h.addWidget(self.b_import)
        h.addWidget(QLabel("Pairing:"))
        self.c_method = QComboBox()
        for m in C.METHODS:
            self.c_method.addItem(m, m)
        help_tip(self.c_method, "hungarian = best one-to-one pairing by MAC; max_mac = best partner of each test mode; order = by number")
        h.addWidget(self.c_method)
        h.addWidget(QLabel("Min. MAC of a pair:"))
        self.s_minmac = dspin(0.0, 0.0, 1.0, 2, 0.05)
        help_tip(self.s_minmac, "A pair below this MAC is dropped (the test mode then has no partner = FAIL); 0 = keep every pair")
        h.addWidget(self.s_minmac)
        h.addWidget(QLabel("Test modes from [Hz]:"))
        self.s_fmin = dspin(0.0, 0.0, 1e6, 2, 10.0)
        help_tip(self.s_fmin, "Test modes below this frequency are left out (0 = no limit); the result then cannot be PASS")
        h.addWidget(self.s_fmin)
        h.addWidget(QLabel("to [Hz]:"))
        self.s_fmax = dspin(0.0, 0.0, 1e6, 2, 10.0)
        help_tip(self.s_fmax, "Test modes above this frequency are left out (0 = no limit); the result then cannot be PASS")
        h.addWidget(self.s_fmax)
        self.k_damping = QCheckBox("Write measured damping into the dynamics")
        help_tip(self.k_damping, "Measured damping ratios of the paired modes become the modal damping table of the dynamics analysis; its results are then discarded")
        h.addWidget(self.k_damping)
        h.addWidget(QLabel("Mass of COC / AOC:"))
        self.c_orth = QComboBox()
        for text, key in (("SEREP of the FE modes (indicative)", "serep"), ("imported matrix (TAM)", "tam"), ("not computed", "none")):
            self.c_orth.addItem(text, key)
        help_tip(self.c_orth, "The mass reduced to the sensor DOFs for the cross-orthogonality (ECSS-E-ST-32-11C 3.2.39, mandatory): the imported mass matrix of the test-analysis model (TAM; the only "
                              "source that can PASS or FAIL the COC / AOC rows), a SEREP computed from the FE modes and the mass of the study (INDICATIVE only: the COC of the projection of a test "
                              "shape onto the compared FE modes; the rows stay UNCONFIRMED whatever the value, and a value outside the limit is a warning hint, not a FAIL), or not computed "
                              "(its rows are then UNCONFIRMED)")
        self.c_orth.currentIndexChanged.connect(self._orth_changed)
        h.addWidget(self.c_orth)
        self.b_tam = QPushButton("Import mass matrix…")
        help_tip(self.b_tam, "Read the CSV mass matrix of the test-analysis model (header ,N1:x,N1:y,...; rows N1:x,m11,...); its DOFs must be the sensor DOFs, its axes the GLOBAL axes of the FE model")
        self.b_tam.clicked.connect(self.import_tam)
        h.addWidget(self.b_tam)
        self.k_tamaxes = QCheckBox("TAM in global axes")
        help_tip(self.k_tamaxes, "Tick only if the imported mass matrix is written in the GLOBAL axes of the FE model (the test shapes are rotated into them). With rotated sensors (UFF 2420) the "
                                 "COC / AOC rows stay UNCONFIRMED until you state this")
        h.addWidget(self.k_tamaxes)
        self.b_run = QPushButton("Correlate")
        help_tip(self.b_run, "Compare the solved natural modes and the response with the test data")
        self.b_run.clicked.connect(self.run)
        h.addWidget(self.b_run)
        self.b_rows = QPushButton("Add limit rows to requirement set")
        help_tip(self.b_rows, "Adds the empty corr_* limit rows (no value) to the requirement set; this withdraws a confirmation")
        self.b_rows.clicked.connect(self.add_limit_rows)
        h.addWidget(self.b_rows)
        h.addStretch(1)
        return h

    def _orth_changed(self, *_a: Any) -> None:
        self.b_tam.setEnabled(self.c_orth.currentData() == "tam")
        self.k_tamaxes.setEnabled(self.c_orth.currentData() == "tam")

    def import_tam(self) -> None:
        """Reads a CSV mass matrix (the TAM); a refused file is a normal outcome and is shown in a message."""
        path, _ = QFileDialog.getOpenFileName(self, "Import mass matrix", str(Path.home()), "Mass matrix (*.csv);;All files (*)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tam = C.load_mass_file(path)
        except (C.TestDataError, OSError) as e:
            self._warn(str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.tam = self.win._corr_tam = tam
        self.k_tamaxes.setChecked(False)                                   # a statement about the axes of the PREVIOUS matrix does not hold for this one: state it again
        if self.study is not None:
            CR.remember_tam(self.study, tam, getattr(self.results, "mesh", None))
        self.c_orth.setCurrentIndex(max(self.c_orth.findData("tam"), 0))
        self.l_orth.setText(f"Mass matrix: {C._plain(tam.source, 60)} - {len(tam.dofs)} DOFs" + (f", unit {C._plain(tam.unit, 12)}" if tam.unit else ""))

    def _build_import_options(self) -> QHBoxLayout:
        """How the test file is read: the convention of the local sensor axes (UFF 2420) and the unit of the damping of UFF 55 modes."""
        h = QHBoxLayout()
        h.addWidget(QLabel("Local sensor axes (UFF 2420) are the:"))
        self.c_rotation = QComboBox()
        self.c_rotation.addItem("columns of the 3 x 3 matrix (OpenModal)", "columns")
        self.c_rotation.addItem("rows of the 3 x 3 matrix", "rows")
        help_tip(self.c_rotation, "The file does not say whether the columns or the rows of the rotation matrix are the local axes. Data at rotated sensors stay UNCONFIRMED until you confirm the convention")
        self.c_rotation.currentIndexChanged.connect(self._rotation_changed)
        h.addWidget(self.c_rotation)
        self.k_rotation = QCheckBox("Convention confirmed for this file")
        help_tip(self.k_rotation, "Tick only after checking the sensor directions of the file (a rotated sensor measures along a different axis under the other convention)")
        self.k_rotation.toggled.connect(self._rotation_confirmed)
        h.addWidget(self.k_rotation)
        h.addWidget(QLabel("Damping of UFF 55 modes:"))
        self.c_dunit = QComboBox()
        for text, key in (("(not stated)", ""), ("ratio", "ratio"), ("percent", "percent")):
            self.c_dunit.addItem(text, key)
        help_tip(self.c_dunit, "A UFF 55 does not state the unit of its damping; a value above 0.2 without a unit is refused. Takes effect at the next import")
        h.addWidget(self.c_dunit)
        self.l_rot = _plain_label()
        h.addWidget(self.l_rot, 1)
        return h

    def _sync_rotation(self) -> None:
        t = self.test
        self.c_rotation.blockSignals(True)
        self.k_rotation.blockSignals(True)
        try:
            busy = self._thread is not None                                 # the test data are read by a worker: the convention must not change under it
            self.c_rotation.setEnabled(not busy)
            if t is not None:
                self.c_rotation.setCurrentIndex(max(self.c_rotation.findData(t.rotation), 0))
            amb = t is not None and bool(t.rotation_ambiguous)
            self.k_rotation.setEnabled(amb and not busy)
            self.k_rotation.setChecked(bool(t is not None and t.rotation_confirmed))
            if t is None or not amb:
                self.l_rot.setText("" if t is None else "No rotated sensor depends on the convention.")
            else:
                nodes = ", ".join(C._plain(n, 20) for n in t.rotation_ambiguous[:8]) + (" ..." if len(t.rotation_ambiguous) > 8 else "")
                if t.rotation_confirmed:
                    via = " through the MCP interface (an agent): check it yourself" if t.rotation_confirmed_via == "mcp" else ""
                    self.l_rot.setText(f"Rotated sensors: {nodes}. Convention confirmed{via}.")
                else:
                    self.l_rot.setText(f"Rotated sensors: {nodes}. The convention is NOT confirmed: the rows that depend on them are UNCONFIRMED.")
        finally:
            self.c_rotation.blockSignals(False)
            self.k_rotation.blockSignals(False)

    def _rotation_changed(self, *_a: Any) -> None:
        if self.test is None or self.study is None or self._thread is not None:
            return
        CR.set_rotation(self.study, self.test, str(self.c_rotation.currentData()))
        self.result = None
        self._changed()
        self.refresh()

    def _rotation_confirmed(self, checked: bool) -> None:
        if self.test is None or self.study is None or self._thread is not None:
            return
        CR.confirm_rotation(self.study, self.test, "gui", bool(checked))
        self._changed()
        self._sync_rotation()
        self.show_status()

    def _build_limits(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        note = _plain_label("The correlation limits (corr_*) of the requirement set of the study, read-only: enter and confirm them in Tools - Space structures. A criterion declared "
                            "not applicable is not judged: check who declared it and on what basis.")
        v.addWidget(note)
        self.l_limits_by = _plain_label()
        v.addWidget(self.l_limits_by)
        self.t_limits_all = table(LIMIT_COLUMNS, placeholder="No requirement set")
        v.addWidget(self.t_limits_all, 1)
        return w

    def fill_limits(self) -> None:
        """The corr_* limits of the study as they are NOW (value, unit, not applicable, source) - read-only."""
        t = self.t_limits_all
        t.setRowCount(0)
        if self.study is None:
            return
        lim = C.limits_of_study(self.study)
        for k, e in lim.status().items():
            i = t.rowCount()
            t.insertRow(i)
            status = "not applicable" if e["not_applicable"] else ("usable" if e["usable"] else "NOT usable: " + C._plain(e["reason"], 100))
            vals = [k, "-" if e["not_applicable"] else (fmt(e["value"], 6) if e["value"] is not None else "not entered"), e["unit"], "YES" if e["not_applicable"] else "no",
                    C._plain(e["document"]), C._plain(e["revision"]), C._plain(e["clause"]), status, e["meaning"], C.LIMIT_HINTS.get(k, "")]
            for j, x in enumerate(vals):
                col = AMBER if (j == 3 and e["not_applicable"]) or (j == 7 and not e["usable"] and not e["not_applicable"]) else None
                t.setItem(i, j, _item(x, right=j == 1, color=col))
        if not lim.set_confirmed:
            self.l_limits_by.setText(f"Requirement set {C._plain(lim.set_name, 60) or '(none)'}: NOT confirmed - no limit is usable.")
        else:
            via = {"mcp": " through the MCP interface (an agent): check it yourself", "gui": " through the GUI"}.get(lim.confirmed_via, "")
            self.l_limits_by.setText(f"Requirement set {C._plain(lim.set_name, 60)}: confirmed by {C._plain(lim.confirmed_by, 40)} on {C._plain(lim.date, 12)}{via}.")
        na = C.not_applicable_limits(lim)
        if na:
            self.l_limits_by.setText(self.l_limits_by.text() + f"  {len(na)} criterion/criteria declared NOT APPLICABLE.")
            self.l_limits_by.setStyleSheet(f"color: {AMBER.name()};")
        else:
            self.l_limits_by.setStyleSheet("")

    def _build_summary(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.t_rows = table(["Item", "Metric", "Value", "Limit", "Status"], placeholder="Import test data and press Correlate")
        self.t_limits = table(["Limit", "Value", "Document", "Revision", "Clause", "Usable"], placeholder="")
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.addWidget(self.t_rows)
        sp.addWidget(self.t_limits)
        sp.setStretchFactor(0, 3)
        v.addWidget(sp)
        return w

    def _build_mac(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.plot_mac = LazyPlot()
        self.plot_mac.setMinimumHeight(300)
        self.t_pairs = table(self.PAIR_COLUMNS)
        self._pair_header_tips()
        top = QHBoxLayout()
        top.addWidget(QLabel("Matrix:"))
        self.c_matrix = QComboBox()
        self.c_matrix.addItem("MAC", "mac")
        self.c_matrix.addItem("COC (cross-orthogonality)", "coc")
        help_tip(self.c_matrix, "MAC of the test modes against the FE modes, or their mass-weighted cross-orthogonality COC (ECSS-E-ST-32-11C 3.2.39)")
        self.c_matrix.currentIndexChanged.connect(lambda *_: self.draw_mac())
        top.addWidget(self.c_matrix)
        self.l_orth = _plain_label()
        top.addWidget(self.l_orth, 1)
        v.addLayout(top)
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.addWidget(self.plot_mac)
        sp.addWidget(self.t_pairs)
        sp.setStretchFactor(0, 3)
        v.addWidget(sp)
        return w

    def _build_response(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        h = QHBoxLayout()
        h.addWidget(QLabel("Function:"))
        self.c_func = QComboBox()
        self.c_func.currentIndexChanged.connect(self._draw_response)
        h.addWidget(self.c_func, 1)
        v.addLayout(h)
        self.plot_frf = LazyPlot()
        self.plot_frf.setMinimumHeight(300)
        v.addWidget(self.plot_frf, 1)
        self.l_frf = _plain_label()
        v.addWidget(self.l_frf)
        return w

    def _build_sensors(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        note = QLabel("Where each test node is in the FE model: a point in mm (interpolated in the mesh), or the name of a probe / surface selection of the study. "
                      "DOFs: the directions measured, e.g. xyz or z.")
        note.setWordWrap(True)
        v.addWidget(note)
        self.t_sensors = table(SENSOR_COLUMNS)
        triggers = QAbstractItemView.EditTrigger
        self.t_sensors.setEditTriggers(triggers.DoubleClicked | triggers.EditKeyPressed | triggers.AnyKeyPressed)
        v.addWidget(self.t_sensors, 1)
        h = QHBoxLayout()
        for text, tip, slot in (("Map nodes named like a probe", "Test nodes whose label equals a probe name of the study get that probe", self.auto_map),
                                ("Save sensors", "Validate the table and store the sensors in the study", self.save_sensors)):
            b = QPushButton(text)
            help_tip(b, tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        return w

    def _build_functions(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        note = QLabel("How each measured function is compared: a model curve of the same quantity (from the unit of the function), the same frame (absolute, or relative to the base) "
                      "and the same normalisation. A function with an unknown unit, or without a chosen frame when the model has a base excitation, stays UNCONFIRMED. "
                      "A controller PSD without node and direction needs a probe and a DOF here.")
        note.setWordWrap(True)
        v.addWidget(note)
        self.fn_table = FunctionSetupTable()
        v.addWidget(self.fn_table, 1)
        h = QHBoxLayout()
        b = QPushButton("Save function setup")
        help_tip(b, "Store the frame, normalisation, unit, probe and DOF of the functions in the study")
        b.clicked.connect(self.save_functions)
        h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        return w

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        self._orth_changed()
        self.fill_sensors()
        self.fn_table.fill(self.test, CR.load_setups(self.study) if self.study is not None else {})
        self._sync_rotation()
        self.fill_limits()
        self.show_data()
        has = self.study is not None
        self.b_run.setEnabled(has and self.test is not None and self._thread is None)
        self.b_rows.setEnabled(has)
        self.show_status()

    def show_data(self) -> None:
        t = self.test
        if t is None:
            self.l_data.setText("No test data are imported.")
            return
        extra = ""
        local = sum(1 for c in t.node_cs.values() if c)
        if local or t.unresolved:
            extra = f"; local sensor systems: {local} node(s), {len(t.unresolved)} left out (system not fully known)"
        self.l_data.setText(f"Test data: {t.source or 'file'} - {len(t.modes)} modes, {len(t.functions)} functions, {len(t.nodes())} nodes"
                            + (f"; {len(t.warnings)} notes" if t.warnings else "") + extra)
        self.l_data.setToolTip("\n".join(t.warnings[:30]))

    def show_status(self) -> None:
        """The requirement-set line, the VERIFIED verdict (never the stored one on its own word) and the coverage line."""
        st = CR.verify_stored(self.study, self.results, self.test) if self.study is not None else None
        lim = st["limits"] if st is not None else (C.limits_of_study(self.study) if self.study is not None else None)
        if lim is None:
            self.l_req.setText("")
        elif not lim.set_confirmed:
            self.l_req.setText(f"Requirement set: {C._plain(lim.set_name, 60) or 'none'} - not confirmed")
        else:
            via = {"gui": "", "mcp": " - confirmed through the MCP interface (an agent): check it yourself"}.get(lim.confirmed_via, f" via {C._plain(lim.confirmed_via, 12)}" if lim.confirmed_via else "")
            self.l_req.setText(f"Requirement set: {C._plain(lim.set_name, 60)} - confirmed by {C._plain(lim.confirmed_by, 40)} on {C._plain(lim.date, 12)}{via}")
        if st is None:
            self.l_verdict.setText("")
            self.l_verdict.setToolTip("")
            self.l_cov.setText("")
            self.l_cov.setToolTip("")
            na0 = C.not_applicable_limits(lim) if lim is not None else []
            self.l_na.setText("NOT APPLICABLE (not judged): " + "; ".join(C.not_applicable_text(x) for x in na0[:3]) if na0 else "")
            self.l_na.setToolTip("\n".join(C.not_applicable_text(x) for x in na0))
            return
        v = str(st["verdict"])
        why = ("; ".join(C._plain(r, 160) for r in st["why"]) if st["why"] else "")
        if self.result is None:
            self.l_verdict.setText(f"Stored result of the last correlation: {v}" + (f" (was {st['stored_verdict']}; {why})" if not st["current"] else "") + " - press Correlate to recompute")
        else:
            self.l_verdict.setText(f"Verdict: {v}" + _VERDICT_NOTE.get(v, "") + (f" ({why})" if not st["current"] else "")
                                   + (f"   Notes: {len(self.result.warnings)} (hover)" if self.result.warnings else ""))
        self.l_verdict.setStyleSheet(f"font-weight: bold; color: {STATUS_COLOUR.get(v, AMBER).name()};")
        self.l_verdict.setToolTip(("Verified against the current study: " + ", ".join(st["verified"]) if st["verified"] else "") + ("\nNot verifiable: " + "; ".join(st["unverified"]) if st["unverified"] else "")
                                  + ("\n" + "\n".join(self.result.warnings) if self.result is not None else ""))
        na = C.not_applicable_limits(st["limits"])
        self.l_na.setText("NOT APPLICABLE (not judged): " + "; ".join(C.not_applicable_text(x) for x in na[:3]) + (f"; ... ({len(na)} criteria, hover)" if len(na) > 3 else "") if na else "")
        self.l_na.setToolTip("\n".join(C.not_applicable_text(x) for x in na))
        self.l_na.setStyleSheet(f"color: {AMBER.name()};")
        res = st["result"]
        cov = res.get("coverage") or {}
        opts = res.get("options") or {}
        if cov:
            try:
                opt_text = C.CorrelationOptions.from_dict(opts).describe() if opts else "-"
            except (ValueError, TypeError):
                opt_text = "-"                                                      # a damaged record in the study file
            line = (f"Correlated: {cov.get('test_modes_used', 0)} of {cov.get('test_modes_total', 0)} test modes, {cov.get('dofs_used', 0)} DOFs, "
                    f"{cov.get('functions_compared', 0)} of {cov.get('functions_total', 0)} functions; options: {opt_text}")
            reasons = [str(r) for r in cov.get("reasons") or []]
            if reasons:
                line += "\nLEFT OUT: " + "; ".join(C._plain(r, 200) for r in reasons[:3]) + (f"; ... ({len(reasons)} items, hover)" if len(reasons) > 3 else "")
            self.l_cov.setText(line)
            self.l_cov.setToolTip("\n".join(reasons))
        else:
            self.l_cov.setText("")
            self.l_cov.setToolTip("")

    def import_file(self) -> None:
        if self._thread is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import test data", str(Path.home()), "Test data (*.csv *.uff *.unv *.uf);;All files (*)")
        if not path:
            return
        self._start_import(path)

    def _start_import(self, path: str) -> None:
        """Reads the file on a worker thread (up to 32 MB: the window stays responsive); the result is taken over in the main thread (`_import_done`)."""
        from .workers import TaskThread

        rotation, dunit = str(self.c_rotation.currentData()), str(self.c_dunit.currentData() or "")

        def job(_th: Any) -> Any:
            try:
                return C.load_test_file(path, rotation=rotation, damping_unit=dunit)
            except (C.TestDataError, OSError) as e:                        # a refused file is a normal outcome, not a crash
                return e
        t = TaskThread(job, self)
        t.done.connect(self._import_done)
        t.failed.connect(lambda m: self._import_done(C.TestDataError(m)))
        t.finished.connect(self._import_finished)
        self._thread = t
        self.b_import.setEnabled(False)
        self.b_run.setEnabled(False)
        self.l_data.setText("Reading the test file…")
        reg = getattr(self.win, "register_aux", None)
        if callable(reg):
            reg(t)
        t.start()

    def _import_finished(self) -> None:
        t, self._thread = self._thread, None
        unreg = getattr(self.win, "unregister_aux", None)
        if callable(unreg) and t is not None:
            unreg(t)
        if t is not None:
            t.deleteLater()
        self.b_import.setEnabled(True)
        self.refresh()
        if self._close_pending:
            self._close_pending = False
            self.done(0)

    def wait_for_import(self, msec: int = 20000) -> None:
        """Blocks (processing events) until the import thread has finished - for tests and scripts."""
        end = time.monotonic() + msec / 1000.0
        app = QApplication.instance()
        while self._thread is not None and time.monotonic() < end:
            if app is not None:
                app.processEvents()
            if self._thread is not None:
                self._thread.wait(20)

    def _import_done(self, obj: Any) -> None:
        if isinstance(obj, Exception):
            self._warn(str(obj))
            return
        td: C.TestData = obj
        self.test = td
        self.win._corr_test = td
        if self.study is not None:
            CR.adopt_rotation(self.study, td)
            CR.remember_test(self.study, td)
            have = {s.node for s in CR.load_sensors(self.study)}
            new = [s for s in C.auto_sensors(td, self.study) if s.node not in have]
            if new:
                CR.upsert_sensors(self.study, new)
                self._changed()
        self.result = None
        self.refresh()
        self.tabs.setCurrentIndex(3 if td.modes and not self._all_mapped() else 4 if td.functions and not td.modes else 0)

    def done(self, r: int) -> None:
        """Closing (X, Esc) while a worker runs: the window is only hidden and closes itself when the worker has finished (the dialog is deleted on close, a running thread must not be
        destroyed, and waiting for it would freeze the program for as long as the SEREP mass of a large model takes). Showing it again takes the pending close back."""
        if self._thread is not None:
            self._close_pending = True
            self.hide()
            return
        super().done(r)

    def showEvent(self, ev: Any) -> None:
        self._close_pending = False
        super().showEvent(ev)

    def _all_mapped(self) -> bool:
        if self.test is None or self.study is None:
            return True
        mapped = {s.node for s in CR.load_sensors(self.study)}
        return all(n in mapped for n in self.test.nodes())

    # ------------------------------------------------------------------ sensors
    def fill_sensors(self) -> None:
        t = self.t_sensors
        t.setRowCount(0)
        if self.study is None:
            return
        sens = {s.node: s for s in CR.load_sensors(self.study)}
        nodes = list(sens) + [n for n in (self.test.nodes() if self.test else []) if n not in sens]
        for n in nodes[:MAX_TABLE_ROWS]:
            s = sens.get(n)
            r = t.rowCount()
            t.insertRow(r)
            pt = s.point if s is not None and s.point is not None else [None, None, None]
            t.setItem(r, 0, _item(n))
            for k in range(3):
                t.setItem(r, 1 + k, _item(fmt(pt[k], 6) if pt[k] is not None else "", editable=True, right=True))
            t.setItem(r, 4, _item((s.probe or s.selection) if s is not None else "", editable=True))
            t.setItem(r, 5, _item("".join(s.dofs) if s is not None else "xyz", editable=True))

    def _sensors_from_table(self) -> list[C.Sensor]:
        items: list[dict[str, Any]] = []
        t = self.t_sensors
        for r in range(t.rowCount()):
            def cell(c: int, r: int = r) -> str:
                it = t.item(r, c)
                return it.text().strip() if it is not None else ""
            coords = [cell(1), cell(2), cell(3)]
            ref = cell(4)
            if not any(coords) and not ref:
                continue                                                               # an unmapped node: left out
            d: dict[str, Any] = {"node": cell(0), "dofs": cell(5) or "xyz"}
            if any(coords):
                try:
                    d["point"] = [float(c.replace(",", ".")) for c in coords]
                except ValueError:
                    raise ValueError(f"sensor {cell(0)}: the coordinates must be numbers") from None
            if ref:
                d["probe" if ref in {p.name for p in self.study.dynamics.probes} else "selection"] = ref
            items.append(d)
        return C.sensors_from_dicts(items, self.study)

    def save_sensors(self) -> bool:
        """Validates the table and stores the sensors; False (with a message) when an entry is bad."""
        if self.study is None:
            return False
        try:
            sens = self._sensors_from_table()
        except ValueError as e:
            self._warn(str(e))
            return False
        CR.save_sensors(self.study, sens)
        self._changed()
        return True

    def save_functions(self) -> bool:
        """Stores the function setup of the table; False (with a message) when a field is bad."""
        if self.study is None:
            return False
        try:
            CR.update_setups(self.study, self.fn_table.values())
        except ValueError as e:
            self._warn(str(e))
            return False
        self._changed()
        self.fn_table.fill(self.test, CR.load_setups(self.study))
        return True

    def auto_map(self) -> None:
        if self.study is None or self.test is None:
            return
        have = {s.node for s in CR.load_sensors(self.study)}
        new = [s for s in C.auto_sensors(self.test, self.study) if s.node not in have]
        if new:
            CR.upsert_sensors(self.study, new)
            self._changed()
        self.fill_sensors()

    # ------------------------------------------------------------------ the correlation
    def add_limit_rows(self) -> None:
        sp = CR.ensure_space(self.study)
        if sp.requirement_set is None:
            self._warn("There is no requirement set: create it in Tools - Space structures first.")
            return
        added = C.install_limit_rows(sp.requirement_set)
        self._changed()
        self.show_status()
        self._warn(f"{len(added)} empty limit rows were added to the requirement set. Enter the values and confirm the set in Tools - Space structures."
                   if added else "The correlation limit rows are already in the requirement set.")

    def run(self) -> None:
        """Starts the correlation on a worker thread (the SEREP mass of a large model takes seconds: the window stays responsive); the study is written in the main thread (`_run_done`)."""
        if self.study is None or self.test is None or self._thread is not None:
            return
        if not self.save_sensors() or not self.save_functions():
            return
        from .workers import TaskThread
        try:
            opt = C.CorrelationOptions(method=str(self.c_method.currentData()), pair_min_mac=float(self.s_minmac.value()), f_min_hz=float(self.s_fmin.value()) or None,
                                       f_max_hz=float(self.s_fmax.value()) or None, orth_mass=str(self.c_orth.currentData()), tam_global_axes=self.k_tamaxes.isChecked(),
                                       tam_global_axes_via="gui" if self.k_tamaxes.isChecked() else "")
            opt.validate()
        except ValueError as e:
            self._warn(str(e))
            return
        study, results, test, tam = self.study, self.results, self.test, self.tam
        self._run_apply = self.k_damping.isChecked()
        self._run_ctx = (study, test, test.fingerprint(), test.rotation, bool(test.rotation_confirmed))      # the result is taken over only for these, in this state (`_run_done`)

        def job(_th: Any) -> Any:
            try:
                return CR.run_correlation(study, results, test, options=opt, tam=tam, defer=True)[0]
            except (ValueError, IndexError, TypeError, KeyError) as e:        # an actionable refusal is a normal outcome, not a crash
                return e
        t = TaskThread(job, self)
        t.done.connect(self._run_done)
        t.failed.connect(self._run_failed)
        t.finished.connect(self._import_finished)
        self._thread = t
        self.b_import.setEnabled(False)
        self.b_run.setEnabled(False)
        self._sync_rotation()                                              # the rotation controls stay locked while the worker reads the test data
        self.l_verdict.setText("Correlating…")
        reg = getattr(self.win, "register_aux", None)
        if callable(reg):
            reg(t)
        t.start()

    def _run_failed(self, msg: str) -> None:
        self._warn(msg)

    def _run_done(self, obj: Any) -> None:
        """The result of the worker, in the main thread: the damping table and the stored result are written to the study here."""
        ctx, self._run_ctx = self._run_ctx, None
        if isinstance(obj, Exception):
            self._warn(str(obj))
            return
        if self.study is None or self.test is None:
            return
        if ctx is not None and (ctx[0] is not self.study or ctx[1] is not self.test
                                or (ctx[2], ctx[3], ctx[4]) != (self.test.fingerprint(), self.test.rotation, bool(self.test.rotation_confirmed))):
            self._warn("The correlation result was discarded: the study or the test data (or their rotation convention) changed while it was running. Correlate again.")
            return
        try:
            table_ = CR.finish_correlation(self.study, self.test, obj, apply_damping=bool(getattr(self, "_run_apply", False)))
        except ValueError as e:
            self._warn(str(e))
            return
        self.result = obj
        if table_ is not None:
            self._changed(damping=True)
            self._warn("The measured damping ratios were written to the damping table of the dynamics analysis. The results that depend on it were discarded: "
                       "solve the dynamics again to use them.")
        else:
            self._changed()
        self.show_result()

    # ------------------------------------------------------------------ display
    def show_result(self) -> None:
        r = self.result
        if r is None:
            return
        self.show_status()
        t = self.t_rows
        t.setRowCount(0)
        for x in r.rows[:MAX_TABLE_ROWS]:
            i = t.rowCount()
            t.insertRow(i)
            lim = f"{x.relation} {fmt(x.limit, 4)} {x.unit}".strip() if x.limit is not None else "–"
            vals = [x.item, x.metric, fmt(x.value, 4), lim, x.status]
            for j, v in enumerate(vals):
                t.setItem(i, j, _item(v, right=j in (2, 3), color=STATUS_COLOUR.get(x.status) if j == 4 else None))
            status_item = t.item(i, 4)
            if x.note and status_item is not None:
                status_item.setToolTip(x.note)
        lt = self.t_limits
        lt.setRowCount(0)
        for k, e in r.limits.status().items():
            i = lt.rowCount()
            lt.insertRow(i)
            usable = "declared not applicable" if e["not_applicable"] else ("yes" if e["usable"] else "no: " + e["reason"])
            for j, v in enumerate([k, "N/A" if e["not_applicable"] else fmt(e["value"], 4), e["document"], e["revision"], e["clause"], usable]):
                lt.setItem(i, j, _item(v, right=j == 1, color=None if e["usable"] or e["not_applicable"] or j != 5 else AMBER))
        self.fill_pairs()
        self.draw_mac()
        self.c_func.blockSignals(True)
        self.c_func.clear()
        for k, m in enumerate(r.responses):
            self.c_func.addItem(m.name or f"{m.node}:{m.dof}", k)
        self.c_func.blockSignals(False)
        self._draw_response()

    PAIR_COLUMNS = ["Test mode", "f test [Hz]", "FE mode", "f FE [Hz]", "df [%]", "MAC", "Max. other MAC", "Damping (test)", "COC", "Max. other COC", "Modulus diff.", "Out of FE span"]
    PAIR_COLUMNS_INDICATIVE = ["Test mode", "f test [Hz]", "FE mode", "f FE [Hz]", "df [%]", "MAC", "Max. other MAC", "Damping (test)", "COC (indicative)", "Max. other COC (indicative)",
                               "Modulus diff.", "Out of FE span"]
    COC_TIP = ("Mass-weighted cross-orthogonality (ECSS-E-ST-32-11C 3.2.39). With the imported mass matrix (TAM) the value is judged against its limit; with the SEREP mass the program "
               "computed from the FE modes it is INDICATIVE only (the COC of the projection of the test shape onto the compared FE modes): the row stays UNCONFIRMED, never PASS and never FAIL")

    def _pair_header_tips(self) -> None:
        """The tooltips of the header of the pairs table (set again whenever the header labels are replaced)."""
        for col, tip in ((8, self.COC_TIP), (9, self.COC_TIP),
                         (10, "The modulus difference of ECSS-E-ST-32-11C Table 5-2: the standard gives no formula; this is the program's own definition (unit-length shapes, largest | |test| - |FE| | as a "
                              "fraction of the largest test amplitude)"),
                         (11, "Share of the test shape the reduced mass cannot judge, Euclidean norm over the sensor DOFs: outside the span of the compared FE modes (SEREP, a warning above 0.05; "
                              "the SEREP rows are UNCONFIRMED anyway) or in the null space of the imported matrix (TAM; above 0.05 the COC / AOC rows of the mode are UNCONFIRMED)")):
            hdr = self.t_pairs.horizontalHeaderItem(col)
            if hdr is not None:
                hdr.setToolTip(tip)

    def fill_pairs(self) -> None:
        t = self.t_pairs
        t.setRowCount(0)
        o = self.result.modal.orth if self.result is not None and self.result.modal is not None else None
        t.setHorizontalHeaderLabels(self.PAIR_COLUMNS_INDICATIVE if o is not None and o.indicative else self.PAIR_COLUMNS)
        self._pair_header_tips()
        if self.result is None or self.result.modal is None:
            return
        for p in self.result.modal.pairs[:MAX_TABLE_ROWS]:
            i = t.rowCount()
            t.insertRow(i)
            for j, v in enumerate([p.test_number, fmt(p.test_freq, 5), p.fe_number, fmt(p.fe_freq, 5), _pct(p.freq_error), fmt(p.mac, 4), fmt(p.offdiag, 3), fmt(p.damping, 3),
                                   "–" if p.coc is None else fmt(p.coc, 4), "–" if p.coc_offdiag is None else fmt(p.coc_offdiag, 3), "–" if p.modulus_diff is None else fmt(p.modulus_diff, 3),
                                   self._span_text(p.test_index)]):
                t.setItem(i, j, _item(v, right=j > 0))

    def _span_text(self, test_index: int) -> str:
        """The out-of-span residual of a test mode (SEREP; the share of the shape the FE modes of the SEREP basis cannot represent at the sensors; TAM: the share in the null space of the
        matrix); "–" when there is none."""
        o = self.result.modal.orth if self.result is not None and self.result.modal is not None else None
        r = None if o is None else o.residual
        return "–" if r is None or test_index >= len(r) or not np.isfinite(r[test_index]) else fmt(float(r[test_index]), 3)

    def draw_mac(self) -> None:
        """Heat map of the MAC matrix (rows: test modes, columns: FE modes); the chosen pairs are framed."""
        m = self.result.modal if self.result is not None else None
        if m is None or m.mac.size == 0:
            return
        coc = self.c_matrix.currentData() == "coc"
        mat = m.orth.coc if coc else m.mac
        held = m.orth.hold_of(m.orth.mode_hold)
        if m.orth.computed:                                     # one line per whole translatable template (a composed sentence is translated word by word)
            self.l_orth.setText("Reduced mass: {0}".format(C._plain(m.orth.label, 80)) + ("\nRows UNCONFIRMED: {0}".format(C._plain(held, 220)) if held else ""))
        else:
            self.l_orth.setText("Cross-orthogonality not computed: {0}".format(C._plain("; ".join(m.orth.problems), 300)))
        self.l_orth.setToolTip(C._plain("; ".join(m.orth.warnings), 1200))
        if mat is None:
            self.plot_mac.axes().clear()
            canvas0 = self.plot_mac.canvas
            if canvas0 is not None:
                canvas0.draw_idle()
            return
        ax = self.plot_mac.axes()
        im = ax.imshow(np.nan_to_num(mat, nan=0.0), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto", origin="upper")
        fig, canvas = self.plot_mac.fig, self.plot_mac.canvas
        assert fig is not None and canvas is not None                              # created by axes()
        fig.colorbar(im, ax=ax, label="COC" if coc else "MAC")
        nt, nf = mat.shape
        ax.set_xticks(range(nf))
        ax.set_xticklabels([f"{n}\n{f:.4g}" for n, f in zip(m.fe_numbers, m.fe_freq)], fontsize=7)
        ax.set_yticks(range(nt))
        ax.set_yticklabels([f"{n}  {f:.4g}" for n, f in zip(m.test_numbers, m.test_freq)], fontsize=7)
        ax.set_xlabel("FE mode, f [Hz]")
        ax.set_ylabel("test mode, f [Hz]")
        if nt * nf <= 400:
            for i in range(nt):
                for j in range(nf):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] < 0.6 else "black")
        from matplotlib.patches import Rectangle
        for p in m.pairs:
            ax.add_patch(Rectangle((p.fe_index - 0.5, p.test_index - 0.5), 1, 1, fill=False, edgecolor="#ff4d4d", linewidth=1.6))
        canvas.draw_idle()

    def _draw_response(self, *_a: Any) -> None:
        r = self.result
        if r is None or not r.responses:
            why = "; ".join(f"{C._plain(i.name, 40)}: {i.problems[0]}" for i in (r.issues if r else [])[:3])
            self.l_frf.setText(("No measured function was compared (the functions need a predicted curve from the dynamics results at a probe named like the sensor, a known unit and "
                                "the choices of the Functions tab)." + (f" {why}" if why else "")) if r else "")
            return
        k = self.c_func.currentData()
        m = r.responses[int(k) if k is not None else 0]
        ax = self.plot_frf.axes()
        ax.semilogy(m.freq, m.test_amp, label="test", color="#1f77b4")
        ax.semilogy(m.freq, m.fe_amp, label="FE model", color="#d62728", linestyle="--")
        ax.axvline(m.f_peak_test, color="#1f77b4", alpha=0.35)
        ax.axvline(m.f_peak_fe, color="#d62728", alpha=0.35)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(("PSD" if m.kind == "psd" else "amplitude") + (f" [{m.unit}]" if m.unit else ""))
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        frf_canvas = self.plot_frf.canvas
        assert frf_canvas is not None
        frf_canvas.draw_idle()
        self.l_frf.setText(f"Peak: test {fmt(m.f_peak_test, 5)} Hz, FE {fmt(m.f_peak_fe, 5)} Hz (shift {_pct(m.peak_shift)} %); peak amplitude error "
                           f"{fmt(m.peak_error_db, 3)} dB; RMS error {_pct(m.rms_error)} %" + (f" - UNCONFIRMED: {'; '.join(m.unconfirmed)}" if m.unconfirmed else ""))


def open_correlation_dialog(win: Any) -> Any:
    """Tools - Test correlation (ECSS)...: the dialog is single per window and reused."""
    from .main import APP_NAME, msg
    if win.study is None:
        msg(win, APP_NAME, "Open a model or a project first.")
        return None
    dlg = getattr(win, "_corr_dlg", None)
    if dlg is None:
        dlg = win._corr_dlg = CorrelationDialog(win)
        dlg.finished.connect(lambda *_: setattr(win, "_corr_dlg", None))
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    else:
        dlg.refresh()
    dlg.show()
    dlg.raise_()
    return dlg


__all__ = ["CorrelationDialog", "open_correlation_dialog"]

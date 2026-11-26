"""Space structures (ECSS), stage S2: the launch-environment setup tab, the 3-axis run / envelope viewer and the mass / centre-of-mass / frequency budget card.

Two tabs added to the space dialog (`space_panel.SpaceDialog`, one hook line): "Launch environment" (levels with their sine / random / shock tables and sources, launch axis,
QSL combination, the S2 limits of the requirement set, confirmation by the user, generation of the QSL load cases) and "Launch loads" (3-axis run over one modal basis,
envelopes per kind / part, margins with the envelopes, mass properties and budgets). The program contains no launcher or standard value: everything is entered by the user,
with its source, and an edit after the confirmation withdraws it. Computation is in `fea_core.space` (the same as MCP and the Word report).
"""
from __future__ import annotations

import time
from html import escape
from typing import Any

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget

from fea_core.space import budgets as BG
from fea_core.space import environments as EV
from fea_core.space import loads as LD
from fea_core.space.model import LimitEntry, RequirementError

from .space_panel import _editable_table, _item, _number
from .widgets import fmt, help_tip, table

GREEN, RED, AMBER = QColor("#1b7f3b"), QColor("#c0392b"), QColor("#b36b00")
STATUS_COLOR = {"PASS": GREEN, "FAIL": RED}
KIND_TITLES = {"sine": "Sine", "random": "Random (ASD)", "shock": "Shock (SRS)"}


def exact(v: Any) -> str:
    """A value as text that reads back to the very same number (repr), without a trailing '.0' of a whole number: the tables must not round what the user entered
    (an edit of one cell re-reads all cells; a rounded cell would silently change the confirmed value)."""
    if v is None:
        return ""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    s = repr(float(v)) if isinstance(v, float) else str(v)
    return s[:-2] if s.endswith(".0") else s


KIND_COLUMNS = {"sine": ["Frequency [Hz]", "Amplitude [g]"], "random": ["Frequency [Hz]", "ASD [g2/Hz]"], "shock": ["Frequency [Hz]", "SRS [g]"]}


class _KindPage(QWidget):
    """The table of one kind (points, extra fields, source) at the level that is selected."""

    def __init__(self, ctrl: "LaunchLoads", kind: str):
        super().__init__()
        self.ctrl, self.kind = ctrl, kind
        lay = QVBoxLayout(self)
        self.t = _editable_table(KIND_COLUMNS[kind], "No points: add rows")
        self.t.setMaximumHeight(170)
        self.t.itemChanged.connect(lambda *_a: ctrl.commit())
        lay.addWidget(self.t)
        h = QHBoxLayout()
        b_add, b_del = QPushButton("Add point"), QPushButton("Remove point")
        help_tip(b_add, "Append a [frequency, value] row - the frequencies must increase")
        b_add.clicked.connect(self.add_row)
        b_del.clicked.connect(self.del_row)
        h.addWidget(b_add)
        h.addWidget(b_del)
        h.addStretch(1)
        lay.addLayout(h)
        f = QFormLayout()
        self.fields: dict[str, QLineEdit] = {}
        extra = {"sine": [("sweep_rate_oct_min", "Sweep rate [oct/min]"), ("sweeps", "Sweeps")],
                 "random": [("duration_s", "Duration per axis [s]"), ("sigma", "Sigma multiple of the RMS stress (optional)")], "shock": []}[kind]
        for key, label in extra + [("document", "Source document"), ("revision", "Revision"), ("clause", "Clause"), ("note", "Note")]:
            e = QLineEdit()
            e.editingFinished.connect(ctrl.commit)
            self.fields[key] = e
            f.addRow(label, e)
        lay.addLayout(f)

    def add_row(self) -> None:
        self.t.blockSignals(True)
        r = self.t.rowCount()
        self.t.insertRow(r)
        for c in range(2):
            self.t.setItem(r, c, _item("", True, right=True))
        self.t.blockSignals(False)

    def del_row(self) -> None:
        r = self.t.currentRow()
        if r >= 0:
            self.t.removeRow(r)
            self.ctrl.commit()

    def load(self, t: EV.EnvTable | None) -> None:
        self.t.blockSignals(True)
        self.t.setRowCount(0)
        for r, p in enumerate(t.points if t else []):
            self.t.insertRow(r)
            for c in range(2):
                self.t.setItem(r, c, _item(exact(p[c]), True, right=True))
        self.t.blockSignals(False)
        for key, e in self.fields.items():
            v = getattr(t, key, None) if t else None
            e.blockSignals(True)
            e.setText(exact(v))
            e.blockSignals(False)

    def read(self) -> EV.EnvTable | None:
        """The table as entered (None when nothing is entered). A cell that is not a number is an error (ValueError with the cell)."""
        pts: list[list[float]] = []
        for r in range(self.t.rowCount()):
            row = []
            for c in range(2):
                it = self.t.item(r, c)
                v = _number(it.text() if it else "", f"row {r + 1}")
                row.append(v)
            if row[0] is None and row[1] is None:
                continue
            if row[0] is None or row[1] is None:
                raise ValueError(f"row {r + 1}: both numbers are needed")
            pts.append([float(row[0]), float(row[1])])
        vals: dict[str, Any] = {"kind": self.kind, "points": pts}
        for key, e in self.fields.items():
            txt = e.text().strip()
            vals[key] = txt if key in ("document", "revision", "clause", "note") else (_number(txt, key) if txt else None)
        if not pts and not any(vals.get(k) for k in ("document", "revision", "clause", "note", "duration_s", "sweep_rate_oct_min", "sweeps", "sigma")):
            return None
        return EV.EnvTable.from_dict(vals)


class LaunchLoads:
    """Controller of the two tabs; `dlg` = the space dialog (window, study, confirmation name, `_edited`, `_warn`)."""

    def __init__(self, dlg: Any):
        self.dlg = dlg
        self.win = dlg.win
        self._loading = False
        self._thread: Any = None
        self.res3: LD.ThreeAxisResult | None = None
        self._part_rows: list[tuple[str, str, list[float]]] = []
        self.tab_env, self.tab_loads = self._build_env(), self._build_loads()          # kept: a test (and a caller) finds the tabs by the widget, not by their position or title
        dlg.tabs.addTab(self.tab_env, "Launch environment")
        dlg.tabs.addTab(self.tab_loads, "Launch loads")
        dlg.loads_refresh = self.refresh
        dlg.loads = self

    # ------------------------------------------------------------------ access
    @property
    def study(self) -> Any:
        return self.dlg.study

    def space(self, create: bool = False) -> Any:
        return self.dlg.space(create)

    def es(self, create: bool = False) -> EV.EnvironmentSet | None:
        sp = self.space(create)
        es = EV.environments_of(sp)
        if es is None and create and sp is not None:
            es = EV.EnvironmentSet()
        return es

    def _store(self, es: EV.EnvironmentSet) -> None:
        sp = self.space(True)
        if sp is not None:
            EV.store_environments(sp, es)
        self.dlg._edited()
        self.update_status()

    # ================================================================== tab 1: environment
    def _build_env(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel("Launch environments of the project: sine, random and shock tables per level, from the launcher guide or the project requirements, with "
                       "the source. The program contains NO environment: the tables stay empty until you enter and confirm them.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        top = QHBoxLayout()
        top.addWidget(QLabel("Launch axis"))
        self.c_axis = QComboBox()
        self.c_axis.addItems(["", *EV.LAUNCH_AXES])
        help_tip(self.c_axis, "Axis of the launch (axial); the other two are lateral")
        self.c_axis.currentTextChanged.connect(lambda *_a: self.commit())
        top.addWidget(self.c_axis)
        top.addWidget(QLabel("QSL combination"))
        self.c_comb = QComboBox()
        self.c_comb.addItems(list(EV.QSL_COMBINATIONS))
        self.c_comb.setCurrentText("axial_lateral")
        help_tip(self.c_comb, "How the signed quasi-static cases are combined")
        self.c_comb.currentTextChanged.connect(lambda *_a: self.commit())
        top.addWidget(self.c_comb)
        top.addWidget(QLabel("Level"))
        self.c_level = QComboBox()
        self.c_level.addItems(list(EV.ENV_LEVELS))
        self.c_level.currentTextChanged.connect(lambda *_a: self.show_level())
        top.addWidget(self.c_level)
        top.addWidget(QLabel("Shock combination"))
        self.c_combine = QComboBox()
        self.c_combine.addItems(["cqc", "srss"])
        self.c_combine.currentTextChanged.connect(lambda *_a: self.commit())
        top.addWidget(self.c_combine)
        self.k_mm = QCheckBox("Missing mass")
        self.k_mm.setChecked(True)
        help_tip(self.k_mm, "Static response of the residual mass to the shock ZPA")
        self.k_mm.toggled.connect(lambda *_a: self.commit())
        top.addWidget(self.k_mm)
        top.addStretch(1)
        lay.addLayout(top)
        dmp = QHBoxLayout()
        dmp.addWidget(QLabel("Modal damping ratio"))
        self.e_damp = QLineEdit()
        self.e_damp.setMaximumWidth(80)
        help_tip(self.e_damp, "Damping ratio of all modes (0 - 1); part of the confirmation")
        self.e_damp.editingFinished.connect(self.commit)
        dmp.addWidget(self.e_damp)
        dmp.addWidget(QLabel("Source"))
        self.e_damp_src = QLineEdit()
        help_tip(self.e_damp_src, "Where the damping ratio comes from (test, launcher guide, document)")
        self.e_damp_src.editingFinished.connect(self.commit)
        dmp.addWidget(self.e_damp_src, 1)
        lay.addLayout(dmp)
        self.kinds = QTabWidget()
        self.pages = {k: _KindPage(self, k) for k in EV.ENV_KINDS}
        for k, pg in self.pages.items():
            self.kinds.addTab(pg, KIND_TITLES[k])
        lay.addWidget(self.kinds, 2)
        h = QHBoxLayout()
        self.l_env = QLabel("")
        self.l_env.setWordWrap(True)
        h.addWidget(self.l_env, 1)
        self.b_env_confirm = QPushButton("Confirm environments")
        help_tip(self.b_env_confirm, "Confirm the tables after checking them against the source")
        self.b_env_confirm.clicked.connect(self.confirm_env)
        h.addWidget(self.b_env_confirm)
        self.b_env_withdraw = QPushButton("Withdraw")
        self.b_env_withdraw.clicked.connect(self.withdraw_env)
        h.addWidget(self.b_env_withdraw)
        self.b_qsl = QPushButton("Generate QSL load cases")
        help_tip(self.b_qsl, "Signed quasi-static cases from the QSL factors of the requirement set")
        self.b_qsl.clicked.connect(self.generate_qsl)
        h.addWidget(self.b_qsl)
        lay.addLayout(h)
        self.l_issues = QLabel("")
        self.l_issues.setWordWrap(True)
        lay.addWidget(self.l_issues)
        lay.addWidget(QLabel("Limits of the requirement set used by this stage (QSL factors, mass / centre-of-mass budgets, minimum frequencies) - editing one withdraws "
                             "the confirmation of the requirement set"))
        self.t_lim = _editable_table(["Limit", "Value", "Unit", "Document", "Revision", "Clause"], "No limit rows: add the S2 rows")
        self.t_lim.setMaximumHeight(200)
        self.t_lim.itemChanged.connect(self._limit_edited)
        lay.addWidget(self.t_lim)
        b_rows = QPushButton("Add the empty S2 limit rows")
        help_tip(b_rows, "Add the empty rows of the QSL factors, budgets and minimum frequencies")
        b_rows.clicked.connect(self.add_limit_rows)
        lay.addWidget(b_rows)
        return w

    def show_level(self) -> None:
        es = self.es()
        lv = es.level(self.c_level.currentText()) if es is not None else None
        self._loading = True
        try:
            for k, pg in self.pages.items():
                pg.load(lv.table(k) if lv else None)
            self.c_combine.setCurrentText(lv.combine if lv else "cqc")
            self.k_mm.setChecked(lv.missing_mass if lv else True)
        finally:
            self._loading = False

    def commit(self) -> None:
        """Takes the widgets into the environment set (a new level / table is created on demand) and stores it."""
        if self._loading or self.study is None:
            return
        es = self.es(True)
        if es is None:
            return
        try:
            tables = {k: pg.read() for k, pg in self.pages.items()}
        except ValueError as e:
            self.dlg._warn(str(e))
            return
        try:
            es.damping_zeta = _number(self.e_damp.text(), "damping ratio")
            es.damping_source = self.e_damp_src.text().strip()
        except ValueError as e:
            self.dlg._warn(str(e))
            return
        es.launch_axis = self.c_axis.currentText()
        es.qsl_combination = self.c_comb.currentText()
        name = self.c_level.currentText()
        lv = es.level(name)
        if lv is None and not any(tables.values()):
            self._store(es)
            return
        lv = es.ensure_level(name)
        lv.combine = self.c_combine.currentText()
        lv.missing_mass = self.k_mm.isChecked()
        for k, t in tables.items():
            setattr(lv, k, t)
        self._store(es)

    def confirm_env(self) -> None:
        es = self.es()
        if es is None:
            return
        try:
            EV.confirm_environment(es, self.dlg.by.text(), via="gui")
        except RequirementError as e:
            self.dlg._warn("The environments cannot be confirmed yet:\n\n" + "\n".join(f"• {m}" for m in e.issues[:12]))
            return
        self._store(es)

    def withdraw_env(self) -> None:
        es = self.es()
        if es is not None:
            es.withdraw_confirmation()
            self._store(es)

    def update_status(self) -> None:
        es = self.es()
        if es is None:
            self.l_env.setText("No environments entered.")
            self.l_issues.setText("")
            self.b_env_confirm.setEnabled(self.study is not None)
            self.b_env_withdraw.setEnabled(False)
            return
        if es.is_confirmed():
            via = " (through MCP - check the values and confirm here yourself)" if es.confirmed_via == "mcp" else ""
            col = "#b36b00" if es.confirmed_via == "mcp" else "#1b7f3b"
            self.l_env.setText(f"<b style='color:{col}'>Confirmed</b> by {escape(es.confirmed_by)} on {escape(es.date)}{via}.")
        elif es.confirmed:
            self.l_env.setText("<b style='color:#c0392b'>NOT confirmed</b> - the environments were edited after the confirmation.")
        else:
            self.l_env.setText("<b style='color:#b36b00'>NOT confirmed</b> - results of the launch analysis stay UNCONFIRMED_INPUTS.")
        self.b_env_confirm.setEnabled(True)
        self.b_env_withdraw.setEnabled(es.confirmed)
        probs = EV.validate_environment(es)
        self.l_issues.setText("<span style='color:#c0392b'>" + "<br>".join(escape(p) for p in probs[:6]) + "</span>" if probs else "")

    def generate_qsl(self) -> None:
        st = self.study
        if st is None:
            return
        try:
            cases = LD.generate_qsl(st)
        except LD.LoadsError as e:
            self.dlg._warn(str(e))
            return
        self.win.mark_changed("results")
        self.win.log_msg(f"Space structures: {len(cases)} quasi-static launch load cases generated - solve the study to get their states")
        self.dlg._edited()

    # ---- limits table
    def fill_limits(self) -> None:
        rs = self.dlg.rs()
        self._loading = True
        try:
            self.t_lim.setRowCount(0)
            rows = [x for x in (rs.limits if rs else []) if x.key in BG.S2_LIMITS]
            for r, x in enumerate(rows):
                self.t_lim.insertRow(r)
                vals = [(x.key, False), (exact(x.value), True), (x.unit, False), (x.document, True), (x.revision, True), (x.clause, True)]
                for c, (txt, ed) in enumerate(vals):
                    it = _item(txt, ed, right=c == 1)
                    if c == 0:
                        it.setToolTip(BG.S2_LIMITS[x.key][1])
                    self.t_lim.setItem(r, c, it)
        finally:
            self._loading = False

    def _limit_edited(self, item: QTableWidgetItem) -> None:
        rs = self.dlg.rs()
        if self._loading or rs is None:
            return
        key_item = self.t_lim.item(item.row(), 0)
        x: LimitEntry | None = next((e for e in rs.limits if key_item is not None and e.key == key_item.text()), None)
        if x is None:
            return
        txt = item.text().strip()
        try:
            if item.column() == 1:
                x.value = _number(txt, x.key)
            elif item.column() == 3:
                x.document = txt
            elif item.column() == 4:
                x.revision = txt
            elif item.column() == 5:
                x.clause = txt
        except ValueError as e:
            self.dlg._warn(str(e))
            self.fill_limits()
            return
        self.dlg._edited()

    def add_limit_rows(self) -> None:
        rs = self.dlg.rs()
        if rs is None:
            self.dlg._warn("There is no requirement set: create it on the Requirement set tab first.")
            return
        BG.install_limit_rows(rs)
        self.fill_limits()
        self.dlg._edited()

    # ================================================================== tab 2: loads
    def _build_loads(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        h = QHBoxLayout()
        h.addWidget(QLabel("Level"))
        self.c_run_level = QComboBox()
        self.c_run_level.addItems(list(EV.ENV_LEVELS))
        h.addWidget(self.c_run_level)
        self.k_kind = {k: QCheckBox(KIND_TITLES[k].split(" ")[0]) for k in EV.ENV_KINDS}
        for cb in self.k_kind.values():
            cb.setChecked(True)
            h.addWidget(cb)
        self.k_axis = {a: QCheckBox(LD.AXIS_NAMES[a]) for a in LD.AXES}
        for cb in self.k_axis.values():
            cb.setChecked(True)
            h.addWidget(cb)
        self.b_run = QPushButton("Run 3-axis analysis")
        help_tip(self.b_run, "Apply the environment along X, Y, Z over one modal basis")
        self.b_run.clicked.connect(self.run_3axis)
        h.addWidget(self.b_run)
        self.b_margins = QPushButton("Margins with envelopes")
        help_tip(self.b_margins, "Margins of safety of the solved states plus the envelopes as extra cases")
        self.b_margins.clicked.connect(self.margins)
        h.addWidget(self.b_margins)
        h.addStretch(1)
        lay.addLayout(h)
        self.l_run = QLabel("")
        self.l_run.setWordWrap(True)
        lay.addWidget(self.l_run)
        self.t_env = table(["Kind", "Max S along X [MPa]", "Y", "Z", "Envelope max S [MPa]", "Governing axis", "Max |U| [mm]", "Damage (sum of axes)"],
                           placeholder="Run the 3-axis analysis first", rows=(2, 5))
        lay.addWidget(self.t_env)
        lay.addWidget(QLabel("Envelope by part (a click marks the place in the 3D view)"))
        self.t_part = table(["Kind", "Part", "Max S [MPa]", "Axis", "Max |U| [mm]", "Damage (sum of axes)"], placeholder="No envelope yet", rows=(2, 8))
        self.t_part.itemSelectionChanged.connect(self._part_clicked)
        lay.addWidget(self.t_part)
        self.l_margins = QLabel("")
        self.l_margins.setWordWrap(True)
        lay.addWidget(self.l_margins)
        h2 = QHBoxLayout()
        self.b_mass = QPushButton("Mass, centre of mass and budgets")
        help_tip(self.b_mass, "Mass, centre of mass, inertia and minimum frequency against the limits")
        self.b_mass.clicked.connect(self.mass_card)
        h2.addWidget(self.b_mass)
        self.l_mass = QLabel("")
        self.l_mass.setWordWrap(True)
        h2.addWidget(self.l_mass, 1)
        lay.addLayout(h2)
        self.t_bud = table(["Quantity", "Value", "Limit", "Unit", "Source of the limit", "Status"], placeholder="Compute the budgets first", rows=(2, 8))
        lay.addWidget(self.t_bud)
        return w

    def _mesh(self) -> Any:
        win = self.win
        mesh = win.valid_mesh() if hasattr(win, "valid_mesh") else None
        if mesh is None and getattr(win, "results", None) is not None:
            mesh = win.results.mesh
        return mesh

    def run_3axis(self) -> None:
        st = self.study
        if st is None or self._thread is not None:
            return
        mesh = self._mesh()
        if mesh is None:
            self.dlg._warn("Mesh the study first (Mesh step): the launch analysis runs on the mesh.")
            return
        kinds = [k for k, cb in self.k_kind.items() if cb.isChecked()]
        axes = tuple(a for a, cb in self.k_axis.items() if cb.isChecked())
        if not kinds or not axes:
            self.dlg._warn("Choose at least one kind and one axis.")
            return
        from fea_core.study import Study

        from .workers import TaskThread
        study = Study.from_dict(st.to_dict())                       # the run reads its own copy: the user may edit meanwhile
        level = self.c_run_level.currentText()
        wd = self.win._workdir / f"space3_{int(time.time() * 1000)}"

        def job(th: Any) -> LD.ThreeAxisResult:
            return LD.run_3axis(study, mesh, wd, th.log.emit, level=level, kinds=kinds, axes=axes, cancel=th.cancel)
        t = TaskThread(job, self.win)                              # the window outlives the dialog (it is deleted on close)
        t.log.connect(self.win.log_msg)
        t.done.connect(self._on_done)
        t.failed.connect(self._on_failed)
        t.finished.connect(self._on_finished)
        self._thread = t
        self.win.register_aux(t)
        self.b_run.setEnabled(False)
        self.b_run.setText("Running…")
        self.l_run.setText("The modal basis and the three axes are being computed…")
        t.start()

    def _on_finished(self) -> None:
        self.win.unregister_aux(self._thread)
        self._thread = None
        try:
            self.b_run.setEnabled(True)
            self.b_run.setText("Run 3-axis analysis")
        except RuntimeError:                                       # the dialog was closed while the run was going on
            pass

    def _on_failed(self, text: str) -> None:
        try:
            self.l_run.setText("<span style='color:#c0392b'>" + escape(text) + "</span>")
        except RuntimeError:
            self.win.log_msg(f"Space structures: the 3-axis analysis failed: {text}")

    def _on_done(self, res: LD.ThreeAxisResult) -> None:
        LD.remember(self.win, res)                                 # kept on the window for the next opening of the dialog and the report (digest-checked against the study)
        self.res3 = res
        try:
            self.l_margins.setText("")
            self.show_result()
        except RuntimeError:                                       # the dialog was closed while the run was going on
            self.win.log_msg("Space structures: the 3-axis analysis finished - open the dialog to see the envelopes")

    def show_result(self) -> None:
        r = self.res3
        self.t_env.setRowCount(0)
        self.t_part.setRowCount(0)
        self._part_rows = []
        if r is None:
            self.l_run.setText("")
            return
        mesh = self._mesh()
        conf = r.confirmed_now(self.study, mesh)
        stale = r.stale_reason(self.study, mesh)
        col = "#1b7f3b" if conf and not r.blockers else "#b36b00"
        dm = r.damping or {}
        self.l_run.setText(f"<b style='color:{col}'>{'Inputs confirmed' if conf else 'UNCONFIRMED_INPUTS'}</b> - {escape(r.label)} analysis, level {escape(r.level)}, "
                           f"{len(r.freq)} modes ({escape(r.engine or 'n/a')}), {r.seconds:.1f} s, damping {fmt(float(dm.get('zeta', 0.0)), 3)}"
                           + ("" if dm.get("from_environment") else " (NOT entered in the environments)")
                           + ("" if conf else " - the environment or the requirement set is not confirmed" + (f"; {escape(stale)}" if stale else "") + ": indicative only")
                           + ("<br><b>No PASS possible:</b> " + escape("; ".join(r.blockers[:3])) if r.blockers else "")
                           + ("<br>" + escape("; ".join(r.warnings[:3])) if r.warnings else ""))
        for i, k in enumerate(r.kinds):
            e = r.envelopes[k]
            j = int(e.S.argmax())
            pa = r.per_axis[k]
            cells = [k, *[fmt(float(pa[a].S.max()), 4) if a in pa else "–" for a in LD.AXES], fmt(float(e.S[j]), 4), LD.AXIS_NAMES[int(e.axis_S[j])],
                     fmt(float(e.U.max()), 4), fmt(float(e.D.max()), 3) if e.D is not None else "–"]
            self.t_env.insertRow(i)
            for c, txt in enumerate(cells):
                self.t_env.setItem(i, c, _item(txt, right=c >= 1 and c != 5))
        row = 0
        for k in r.kinds:
            for name, p in r.parts.get(k, {}).items():
                self.t_part.insertRow(row)
                cells = [k, name, fmt(p["max_S_MPa"], 4), p["axis_S"], fmt(p["max_U_mm"], 4), fmt(p["max_D"], 3) if "max_D" in p else "–"]
                for c, txt in enumerate(cells):
                    self.t_part.setItem(row, c, _item(txt, right=c in (2, 4, 5)))
                self._part_rows.append((k, name, list(p["at_mm"])))
                row += 1

    def _part_clicked(self) -> None:
        r = self.t_part.currentRow()
        view = getattr(self.win, "view", None)
        if view is not None and 0 <= r < len(self._part_rows):
            k, name, xyz = self._part_rows[r]
            view.mark(xyz, f"{k}: {name}", (200, 100, 0), key="space_env", below=True)

    def margins(self) -> None:
        st, res = self.study, self.win.results
        if st is None or res is None or not res.steps:
            self.dlg._warn("Solve the study first: the margins of the load states are computed from its results.")
            return
        res3 = self.res3

        def job(_th=None):
            return LD.margins_with_envelopes(st, res, res3)
        run = getattr(self.win, "run_busy", None)                           # off the GUI thread: the window stays painted
        try:
            summ = run(job, "Computing the margins…") if run is not None else job()
        except Exception as e:  # noqa: BLE001 - shown to the user, the dialog stays usable
            self.dlg._warn(f"The margins could not be computed: {e}")
            return
        # run_busy spins a nested event loop: a queued open / a finishing solve may have replaced the study or the results meanwhile
        if self.study is not st or self.win.results is not res or self.res3 is not res3:
            self.dlg._warn("The study or its results changed while the margins were computed - the outdated margins were discarded. Compute them again.")
            return
        w = summ.worst
        col = "#1b7f3b" if summ.signoff_ready else {"FAIL": "#c0392b"}.get(summ.verdict, "#b36b00")
        self.l_margins.setText(f"<b style='color:{col}'>{escape(summ.verdict_text)}</b>"
                               + (f" - lowest MoS {fmt(w.mos, 3)} ({escape(w.part)}, {escape(w.case)}, {escape(w.check)})" if w and w.mos is not None else "")
                               + ("<br>" + escape("; ".join(summ.reasons[:3])) if summ.reasons else ""))

    # ---- mass card
    def mass_card(self) -> None:
        st = self.study
        mesh = self._mesh()
        if st is None or mesh is None:
            self.dlg._warn("Mesh the study first: the mass is integrated on the mesh.")
            return
        from fea_core.space.mass_props import mass_properties
        run = getattr(self.win, "run_busy", None)

        def job(_th=None):
            try:
                return mass_properties(st, mesh)
            except ValueError as e:                                         # expected (no density, ...): a normal warning, no traceback in the log
                return e
        try:
            props = run(job, "Integrating the mass…") if run is not None else job()
        except Exception as e:  # noqa: BLE001 - e.g. "Another operation is in progress"; shown to the user, the dialog stays usable
            self.dlg._warn(str(e))
            return
        if isinstance(props, ValueError):
            self.dlg._warn(str(props))
            return
        # run_busy spins a nested event loop: a queued open / a finishing solve may have replaced the study or the mesh meanwhile
        if self.study is not st or self._mesh() is not mesh:
            self.dlg._warn("The study or its mesh changed while the mass was integrated - the outdated result was discarded. Run it again.")
            return
        items = list(BG.mass_budget(props, st.space).items)
        notes = list(props.notes)
        es = EV.environments_of(st.space)
        data = BG.modal_data_of(self.win.results) if getattr(self.win, "results", None) is not None else None
        if data is not None:
            freq, frac, _src = data
        elif self.res3 is not None and self.res3.current(st, mesh):
            freq, frac = self.res3.freq, self.res3.eff_frac
        else:
            freq = frac = None
        if freq is not None:
            rep = BG.frequency_budget(freq, frac, es.launch_axis if es is not None else "", st.space)
            items += rep.items
            notes += rep.notes
        else:
            notes.append("no natural frequencies yet: solve with the natural-frequency analysis or run the 3-axis analysis")
        w, _v = props.principal()
        self.l_mass.setText(f"<b>{fmt(props.mass_kg, 5)} kg</b>, centre of mass ({', '.join(fmt(float(x), 5) for x in props.cog)}) mm, principal moments "
                            f"{', '.join(fmt(float(x) * 1e-3, 4) for x in w)} kg m2" + ("<br>" + escape(" ".join(notes[:3])) if notes else ""))
        self.t_bud.setRowCount(0)
        for r, i in enumerate(items):
            self.t_bud.insertRow(r)
            for c, txt in enumerate([i.name, fmt(i.value, 5), fmt(i.limit, 5), i.unit, i.source or "–", i.status]):
                it = _item(txt, right=c in (1, 2))
                if c == 5:
                    it.setForeground(STATUS_COLOR.get(i.status, AMBER))
                    it.setToolTip(escape(i.note))
                self.t_bud.setItem(r, c, it)

    # ------------------------------------------------------------------ lifecycle
    def refresh(self) -> None:
        """Reload from the study of the window (the dialog is reused between openings)."""
        es = self.es()
        self._loading = True
        try:
            self.c_axis.setCurrentText(es.launch_axis if es else "")
            self.c_comb.setCurrentText(es.qsl_combination if es else "axial_lateral")
            self.e_damp.setText(exact(es.damping_zeta) if es else "")
            self.e_damp_src.setText(es.damping_source if es else "")
        finally:
            self._loading = False
        self.show_level()
        self.fill_limits()
        self.update_status()
        self.res3 = LD.recall(self.win, self.study, self._mesh()) if self.study is not None else None
        self.show_result()
        has = self.study is not None
        for b in (self.b_run, self.b_margins, self.b_mass, self.b_qsl):
            b.setEnabled(has)


def add_loads_tabs(dlg: Any) -> LaunchLoads:
    """Hook of `SpaceDialog.__init__`: adds the two tabs of stage S2."""
    return LaunchLoads(dlg)


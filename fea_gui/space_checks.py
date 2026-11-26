"""FEM model quality checks of a spacecraft structure (ECSS-E-ST-32-11C 6.4.1 and common FEM practice): the seven checks of `fea_core.space.model_checks` in one non-modal dialog.

Tab "Checks": a traffic-light table (one row per check, filled while the run goes on) with the verdict and the detail of the selected check, a one-click run of all
seven checks in a worker thread (`TaskThread`, cancellable); tab "Budgets": the mass and centre-of-mass budgets and the launch interface that the checks compare the model
with (the user's data, not limits); tab "Tolerances": the limits of the requirement set that decide the checks. The program ships no tolerance of a standard: a limit
counts only while the requirement set is confirmed (Space structures dialog), and a check without a confirmed limit is UNCONFIRMED, never PASS. The run is kept in
`win.model_checks` for the Word report (chapter "Model quality checks"). Computation and verdicts are in `fea_core.space.model_checks` (shared with MCP).
The dialog follows the study of the window: every change of the study (and a newly opened project) reaches `on_study_changed`, so a run that no longer belongs to the study is
shown as OUT OF DATE (table, verdict, stale note) and never as a result of the present model. A close during a run cancels it and the dialog goes away when the run has ended.
"""
from __future__ import annotations

import time
from html import escape
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from fea_core.i18n import tr
from fea_core.space import model_checks as MC
from fea_core.space._checks_run import CheckCancelled
from fea_core.space.model import LimitEntry

from .widgets import TaskProgress, fmt, help_tip, table
from .workers import TaskThread

LIGHT = {MC.PASS: "#1b7f3b", MC.WARN: "#d99a1e", MC.FAIL: "#c0392b", MC.UNCONFIRMED: "#8a8f98", MC.SKIPPED: "#8a8f98"}
TEXT_COLOUR = {MC.PASS: QColor("#1b7f3b"), MC.WARN: QColor("#b36b00"), MC.FAIL: QColor("#c0392b"), MC.UNCONFIRMED: QColor("#5f6670"), MC.SKIPPED: QColor("#5f6670")}
NOT_RUN = "not run"
LIMIT_STATE_TEXT = {"usable": "confirmed", "not confirmed": "not confirmed", "incomplete": "incomplete", "missing": "not entered"}
LIMIT_STATE_COLOUR = {"usable": QColor("#1b7f3b"), "not confirmed": QColor("#b36b00"), "incomplete": QColor("#b36b00"), "missing": QColor("#5f6670")}
VERDICT_COLOUR = {"PASS": "#1b7f3b", "WARN": "#b36b00", "FAIL": "#c0392b"}
VERDICT_NOTE = {"PASS": "all seven checks passed with confirmed limits",
                "WARN": "no check failed, but at least one has a warning",
                "FAIL": "at least one check failed - see the findings; the model must not be used for margins of safety",
                "UNCONFIRMED_REQUIREMENTS": "at least one check has no confirmed limit - the numbers are indicative, this is not a verification result",
                "INCOMPLETE": "not all seven checks were run or one of them could not be completed - this is not a verification of the model"}
LIMIT_COLS = ["Limit", "What it limits", "Value", "Document", "Revision", "Clause", "State"]
AMBER = "#b36b00"
STALE_TEXT = "outdated"


_TIP_NEUTRAL = {ord("="): "&#61;", ord("("): "&#40;", ord("]"): "&#93;", ord("@"): "&#64;"}


def rich_tip(text: str) -> str:
    """A tooltip that shows `text` exactly as written. Qt reads a tooltip that looks like HTML as rich text, so a part name such as `<img src=...>` would be interpreted. The text is
    escaped and put in an explicit rich-text wrapper: escaped text alone is read as rich text by Qt only when it contains `&lt;` (`O&#x27;Brien` or `a &amp; b` would show the entities).
    The characters that the regex layer of the rich-text guard (`fea_core.richtext.safe_rich`: `src=`, `url(`, `](`, `@import`) reads as markup even inside escaped text are written as
    numeric entities, so the guard leaves the tip alone and does not blank a part name that looks like an attribute."""
    return "<qt>" + escape(str(text)).translate(_TIP_NEUTRAL) + "</qt>"


def _optional_number(text: str, what: str) -> float | None:
    s = text.strip().replace(",", ".")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{what}: a number is expected") from None
    if v != v or abs(v) == float("inf"):
        raise ValueError(f"{what}: a finite number is expected")
    return v


def _item(text, editable: bool = False, color: QColor | None = None, right: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    flags = it.flags()
    it.setFlags(flags | Qt.ItemFlag.ItemIsEditable if editable else flags & ~Qt.ItemFlag.ItemIsEditable)
    if color is not None:
        it.setForeground(color)
    if right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return it


def _plain(res: MC.CheckResult) -> str:
    """The detail of one check as plain text (summary, findings, notes, the limits that were consulted, the scalar numbers) - names from files must never be rich text.
    The pieces are translated one by one (a line with numbers in it would not match the catalogue as a whole)."""
    lines = [f"{res.number}. {tr(res.title)} - {tr(res.status)}", tr(res.summary), ""]
    if res.findings:
        lines += [tr("Findings:")] + [f"  ! {tr(f)}" for f in res.findings] + [""]
    if res.limits:
        lines.append(tr("Limits consulted:"))
        for name, p in res.limits.items():
            state = tr("confirmed - decides the check") if p.get("decided") else tr("not confirmed / missing")
            lines.append(f"  {name} = {fmt(p.get('value'), 5)}  ({state}) - {tr(str(p.get('description', '')))}")
        lines.append("")
    if res.notes:
        lines += [tr("Notes:")] + [f"  . {tr(n)}" for n in res.notes] + [""]
    if res.options:
        lines += [tr("Settings that differ from the defaults:"), f"  {MC.options_text(res.options)}", ""]
    nums = [(k, v) for k, v in res.values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums:
        lines.append(tr("Numbers:"))
        lines += [f"  {k} = {fmt(float(v), 6)}" for k, v in nums[:30]]
    return chr(10).join(lines).strip()


class ModelChecksDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Model quality checks (ECSS)")
        self.setModal(False)
        self.resize(1000, 760)
        self.job: TaskThread | None = None
        self._loading = False
        self._results: dict[str, MC.CheckResult] = {}
        self._limit_names: list[str] = []
        self._snapshot = None                    # the study as it was when the run started (what the results belong to)
        self._opt: dict = {}
        self._done = 0
        self._work_dir: Path | None = None       # the work folder of the running job (Cancel kills only the meshing under it)
        self._stale = False                      # the shown results belong to an older state of the study
        self._close_when_done = False            # closed during a run: the dialog goes away when the run has ended
        self._iface_ticked: set[str] = set()     # the ticked launch-interface selections (kept while the dialog is rebuilt)
        lay = QVBoxLayout(self)
        intro = QLabel("Seven checks of the finite element model (ECSS-E-ST-32-11C 6.4.1 and common FEM practice) that the model must pass before its results are used for margins of safety. "
                       "The program contains NO tolerance of a standard: the tolerances are limits of the requirement set that you enter and confirm. Without a confirmed "
                       "limit a check shows its numbers but is UNCONFIRMED and never PASS; only physical invariants (six rigid-body modes of a free body, equilibrium) can "
                       "fail without one.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        h = QHBoxLayout()
        self.l_state = QLabel("")
        self.l_state.setWordWrap(True)
        h.addWidget(self.l_state, 1)
        self.b_reqs = QPushButton("Requirement set...")
        help_tip(self.b_reqs, "Open the Space structures dialog (requirement set, confirmation)")
        self.b_reqs.clicked.connect(self.open_requirements)
        h.addWidget(self.b_reqs)
        lay.addLayout(h)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_checks(), "Checks")
        self.tabs.addTab(self._build_budgets(), "Budgets")
        self.tabs.addTab(self._build_limits(), "Tolerances")
        lay.addWidget(self.tabs, 1)
        b = QHBoxLayout()
        b.addStretch(1)
        c = QPushButton("Close")
        help_tip(c, "Close the dialog (a running check is cancelled)")
        c.clicked.connect(self.close)
        b.addWidget(c)
        lay.addLayout(b)
        self._restore_ui()
        self.refresh()

    # ------------------------------------------------------------------ access
    @property
    def study(self):
        return self.win.study

    def rs(self):
        sp = getattr(self.study, "space", None)
        return sp.requirement_set if sp is not None else None

    # ------------------------------------------------------------------ tab 1: checks
    def _build_checks(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        h = QHBoxLayout()
        self.b_run = QPushButton("Run all checks")
        help_tip(self.b_run, "Run the seven checks on copies of the study (the study and its results do not change)",
                 "<b>Run all checks</b><p>The checks run in the background on COPIES of the study: statics and natural frequencies of the model with its supports removed, "
                 "unit motions of the launch interface, a uniform temperature change, the effective modal mass, the mesh convergence and the equilibrium of the solved "
                 "states. A run can take minutes on a large model; it can be cancelled.</p>")
        self.b_run.clicked.connect(self.run)
        h.addWidget(self.b_run)
        self.b_cancel = QPushButton("Cancel")
        help_tip(self.b_cancel, "Cancel the running checks")
        self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(self.cancel)
        h.addWidget(self.b_cancel)
        self.cb_conv = QCheckBox("Mesh convergence (slow)")
        self.cb_conv.setChecked(True)
        help_tip(self.cb_conv, "Check 6 solves the model on finer meshes - without it the verdict is INCOMPLETE")
        h.addWidget(self.cb_conv)
        h.addStretch(1)
        lay.addLayout(h)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        self.l_verdict = QLabel("")
        self.l_verdict.setWordWrap(True)
        lay.addWidget(self.l_verdict)
        self.t_checks = table(["", "#", "Check", "Status", "Result"], placeholder="", rows=(7, 7))
        self.t_checks.itemSelectionChanged.connect(self._row_selected)
        lay.addWidget(self.t_checks, 2)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("Select a check to see its findings, the limits it used and its numbers")
        lay.addWidget(self.detail, 2)
        self.l_stale = QLabel("")
        self.l_stale.setWordWrap(True)
        lay.addWidget(self.l_stale)
        return w

    def _fill_rows(self) -> None:
        t = self.t_checks
        t.setRowCount(0)
        for i, cid in enumerate(MC.CHECK_IDS):
            t.insertRow(i)
            res = self._results.get(cid)
            status = res.status if res is not None else NOT_RUN
            old = res is not None and self._stale
            light = _item("")
            light.setBackground(QColor("#d0d3d8" if old else LIGHT.get(status, "#d0d3d8")))      # an outdated result has no traffic light
            light.setToolTip(f"{status} ({STALE_TEXT})" if old else status)
            t.setItem(i, 0, light)
            t.setItem(i, 1, _item(str(i + 1), right=True))
            t.setItem(i, 2, _item(MC.CHECK_TITLES[cid]))
            st = _item(f"{status} ({STALE_TEXT})" if old else status, color=QColor(TEXT_COLOUR[MC.SKIPPED]) if old else TEXT_COLOUR.get(status))
            t.setItem(i, 3, st)
            summary = _item(res.summary if res is not None else "")
            if res is not None:
                summary.setToolTip(rich_tip("; ".join(res.findings[:3])) if res.findings else "")          # findings carry part and file names: shown as written, never read as HTML
            t.setItem(i, 4, summary)
        t.resizeColumnToContents(0)
        t.setColumnWidth(0, 22)

    def _show_results(self) -> None:
        self._fill_rows()
        res = [self._results[c] for c in MC.CHECK_IDS if c in self._results]
        if not res:
            self.l_verdict.setText("")
            self.detail.setPlainText("")
            return
        v = MC.overall_verdict(res)
        col = VERDICT_COLOUR.get(v, AMBER)
        if self._stale:
            self.l_verdict.setText(f"<b style='color:#b36b00'>Verdict: OUT OF DATE</b> - the study or its tolerances changed after this run (it said {escape(v)}); run the checks again")
        else:
            self.l_verdict.setText(f"<b style='color:{col}'>Verdict: {escape(v)}</b> - {escape(VERDICT_NOTE.get(v, ''))}")
        sett = {r.id: r.options for r in res if r.options}
        if sett:                                           # every setting that differs from the default is printed next to the verdict
            self.l_verdict.setText(self.l_verdict.text() + "<br>" + escape(tr("Settings that differ from the defaults:")) + " "
                                   + escape("; ".join(f"{i + 1}: {MC.options_text(sett[c])}" for i, c in enumerate(MC.CHECK_IDS) if c in sett)))
        first = next((i for i, c in enumerate(MC.CHECK_IDS) if c in self._results and self._results[c].status == MC.FAIL), 0)
        self.t_checks.selectRow(first)

    def _row_selected(self) -> None:
        r = self.t_checks.currentRow()
        res = self._results.get(MC.CHECK_IDS[r]) if 0 <= r < len(MC.CHECK_IDS) else None
        self.detail.setPlainText(_plain(res) if res is not None else "")

    # ------------------------------------------------------------------ tab 2: budgets
    def _build_budgets(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("What the model is compared with (your data - not limits; leave empty to skip the comparison). The tolerance of the comparison is a limit on the "
                             "Tolerances tab."))
        f = QFormLayout()
        self.e_mass = QLineEdit()
        self.e_mass.setPlaceholderText("not given")
        help_tip(self.e_mass, "Mass budget of the whole model in kg (compared with the FE mass)")
        f.addRow("Mass budget [kg]", self.e_mass)
        h = QHBoxLayout()
        self.e_cog = []
        for ax in "xyz":
            e = QLineEdit()
            e.setPlaceholderText(ax)
            help_tip(e, f"Centre-of-mass budget, {ax} coordinate [mm]")
            self.e_cog.append(e)
            h.addWidget(e)
        f.addRow("Centre of mass budget [mm]", h)
        lay.addLayout(f)
        lay.addWidget(QLabel("Launch interface: selections that carry the supports of the launch interface (none ticked = every support of the study)"))
        self.l_iface = QListWidget()
        self.l_iface.setMaximumHeight(160)
        help_tip(self.l_iface, "Faces on which the unit motions of check 3 are enforced")
        lay.addWidget(self.l_iface)
        lay.addStretch(1)
        return w

    def _ticked(self) -> list[str]:
        return [self.l_iface.item(i).text() for i in range(self.l_iface.count()) if self.l_iface.item(i).checkState() == Qt.CheckState.Checked]

    def fill_interface(self) -> None:
        """The selections of the study; the ticks of the user stay (also across a re-opened dialog and a cancelled run) for the selections that still exist."""
        if self.l_iface.count():
            self._iface_ticked = set(self._ticked())
        self.l_iface.blockSignals(True)
        self.l_iface.clear()
        st = self.study
        for name in (st.selections if st is not None else {}):
            it = QListWidgetItem(str(name))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if str(name) in self._iface_ticked else Qt.CheckState.Unchecked)
            self.l_iface.addItem(it)
        self.l_iface.blockSignals(False)

    def options(self) -> dict:
        """The budgets and settings of the dialog as the option dict of `run_model_checks` (raises ValueError for text that is not a number)."""
        opt: dict = {"include_convergence": self.cb_conv.isChecked()}
        mass = _optional_number(self.e_mass.text(), "Mass budget")
        if mass is not None:
            if mass <= 0:
                raise ValueError("Mass budget: a positive number is expected")
            opt["mass_budget_t"] = mass / 1000.0                 # kg -> t (the units of the program)
        cog = [_optional_number(e.text(), f"Centre of mass {ax}") for e, ax in zip(self.e_cog, "xyz")]
        if any(c is not None for c in cog):
            if any(c is None for c in cog):
                raise ValueError("Centre of mass budget: give all three coordinates or none")
            opt["cog_budget_mm"] = [float(c) for c in cog if c is not None]
        names = self._ticked()
        if names:
            opt["interface_selections"] = names
        return opt

    # ------------------------------------------------------------------ tab 3: tolerances
    def _build_limits(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Tolerances of the checks: limits of the requirement set with their source (document, revision, clause). Enter them from your project documents; an "
                             "edit withdraws the confirmation of the set - confirm it again in the Space structures dialog. Only confirmed limits decide a check."))
        self.t_lim = table(LIMIT_COLS, placeholder="")
        triggers = QAbstractItemView.EditTrigger
        self.t_lim.setEditTriggers(triggers.DoubleClicked | triggers.EditKeyPressed | triggers.AnyKeyPressed)
        self.t_lim.itemChanged.connect(self._limit_edited)
        self.t_lim.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.t_lim.setColumnWidth(1, 330)
        lay.addWidget(self.t_lim, 1)
        return w

    def fill_limits(self) -> None:
        rows = MC.limits_overview(self.study) if self.study is not None else []
        t: QTableWidget = self.t_lim
        self._loading = True
        try:
            t.setRowCount(0)
            self._limit_names = [r["name"] for r in rows]
            for i, r in enumerate(rows):
                t.insertRow(i)
                state = r["state"]
                t.setItem(i, 0, _item(r["name"]))
                desc = _item(r["description"])
                desc.setToolTip(r["description"])
                t.setItem(i, 1, desc)
                t.setItem(i, 2, _item("" if r["value"] is None else fmt(r["value"], 8), True, right=True))
                t.setItem(i, 3, _item(r["document"], True))
                t.setItem(i, 4, _item(r["revision"], True))
                t.setItem(i, 5, _item(r["clause"], True))
                t.setItem(i, 6, _item(LIMIT_STATE_TEXT.get(state, state), color=LIMIT_STATE_COLOUR.get(state)))
        finally:
            self._loading = False

    def _lim_text(self, row: int, col: int) -> str:
        it = self.t_lim.item(row, col)
        return it.text() if it is not None else ""

    def _limit_edited(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() not in (2, 3, 4, 5) or not 0 <= item.row() < len(self._limit_names):
            return
        rs = self.rs()
        if rs is None:
            self._warn("There is no requirement set yet. Create it first: Space structures (ECSS) > Requirement set > New template.")
            self.fill_limits()
            return
        r = item.row()
        name = self._limit_names[r]
        try:
            value = _optional_number(self._lim_text(r, 2), name)
        except ValueError as e:
            self._warn(str(e))
            self.fill_limits()
            return
        doc, rev, clause = (self._lim_text(r, c).strip() for c in (3, 4, 5))
        entry = next((x for x in rs.limits if x.key == name), None)
        if entry is None:
            entry = LimitEntry(name)
            rs.limits.append(entry)
        entry.value, entry.document, entry.revision, entry.clause = value, doc, rev, clause
        if value is None and not (doc or rev or clause or entry.note):
            rs.limits.remove(entry)
        other = getattr(self.win, "_space_dlg", None)          # the requirement-set dialog owns the margins and the confirmation: its own edit hook clears them
        if other is not None:
            other._edited()                                    # marks the project changed, clears the margins that were computed before the edit, shows the confirmation
        else:
            mark = getattr(self.win, "mark_changed", None)
            if callable(mark):
                mark("none")
        self.fill_limits()
        self.update_state()

    # ------------------------------------------------------------------ state
    def _warn(self, text: str) -> None:
        """Message to the user as plain text (the text may contain names from an imported file)."""
        from .main_defs import msg_via_main
        msg_via_main(self, "Model quality checks (ECSS)", text)

    def update_state(self) -> None:
        """The state line (requirement set, scope-aware like the margins of the S1 dialog) and the stale note of the shown run."""
        rs = self.rs()
        if self.study is None:
            self.l_state.setText("")
            self.l_stale.setText("")
            return
        sp = getattr(self.study, "space", None)
        usable = sum(1 for r in MC.limits_overview(self.study) if r["state"] == "usable")
        total = len(MC.LIMIT_DESCRIPTIONS)
        if rs is None:
            self.l_state.setText("<b style='color:#b36b00'>No requirement set</b> - every check will be UNCONFIRMED (numbers only, never PASS).")
        elif sp is not None and sp.requirement_confirmed():
            tail = "" if usable else " - enter the tolerances (Tolerances tab) and confirm the set again."
            if rs.confirmed_via == "mcp":                                     # as in the Space structures dialog: an agent's confirmation is amber, the user checks it
                self.l_state.setText(f"<b style='color:#b36b00'>Requirement set confirmed through MCP</b> (an agent) by {escape(rs.confirmed_by)} on {escape(rs.date)}: {usable} of {total} "
                                     f"tolerances usable - check the values and confirm the set in the Space structures dialog yourself." + tail)
            else:
                via = f" (via {escape(rs.confirmed_via.upper())})" if rs.confirmed_via else ""
                self.l_state.setText(f"<b style='color:#1b7f3b'>Requirement set confirmed</b> by {escape(rs.confirmed_by)} on {escape(rs.date)}{via}: {usable} of {total} tolerances usable" + tail)
        else:
            self.l_state.setText("<b style='color:#b36b00'>Requirement set NOT confirmed</b> - its tolerances do not count, every check will be UNCONFIRMED (numbers only, never PASS).")
        run = getattr(self.win, "model_checks", None)
        if self.job is not None and self._snapshot is not None:          # a run is going on: its results belong to the snapshot it started from
            stale = bool(self._results) and (str(self._snapshot.name) != str(self.study.name) or MC.study_digest(self._snapshot) != MC.study_digest(self.study))
        else:
            stale = bool(run is not None and self._results and not MC.run_is_current(run, self.study))
        if stale != self._stale:
            self._stale = stale
            self._show_results()
        self.l_stale.setText("<b style='color:#b36b00'>The study or its tolerances changed after these results - run the checks again.</b>" if stale else "")

    def on_study_changed(self) -> None:
        """The study of the window was edited or another one was loaded: follow it (a running job keeps its own snapshot; only the notes are updated)."""
        if self.job is not None:
            self.update_state()
            return
        self.refresh()

    def open_requirements(self) -> None:
        from .extras import open_space_dialog
        open_space_dialog(self.win)

    def refresh(self) -> None:
        """Reload from the study of the window (the dialog is reused between openings); the results of a run of this study are shown again."""
        run = getattr(self.win, "model_checks", None)
        self._results = {r.id: r for r in run.results} if run is not None and self.study is not None else {}          # an old run is shown as OUT OF DATE, not dropped
        self._stale = bool(self._results and not MC.run_is_current(run, self.study))
        self.fill_interface()
        self.fill_limits()
        self._show_results()
        self.update_state()
        self.b_run.setEnabled(self.study is not None and self.job is None)

    # ------------------------------------------------------------------ the run
    def run(self) -> None:
        win = self.win
        st = win.study
        if st is None or self.job is not None:
            return
        if win.frame_only():
            self._warn("The checks need a solid model with a mesh (a frame system without geometry has none).")
            return
        if win.busy():
            self._warn("Meshing, analysis or report generation is in progress - wait for it to finish.")
            return
        try:
            opt = self.options()
            MC.options_from_dict(opt)                             # the bounds of the numerical settings (the same as MCP): a typo must not start a run that cannot finish
        except ValueError as e:
            self._warn(str(e))
            return
        calc = win.study_for_solve()
        calc.geometry = win.geometry_str()
        limits = MC.limits_of_study(st)                           # the confirmed requirement set of the study - the only source of tolerances
        mesh, results = win.valid_mesh(), win.results
        study_dir = Path(win.step_path).parent if win.step_path else None
        wd = win._workdir / f"checks_{int(time.time() * 1000)}"
        snapshot = type(st).from_dict(st.to_dict())               # what the run belongs to (the user may edit the study meanwhile)

        def job(t):
            parts = win.shadow_solid_parts()                      # an own copy of the geometry, read in this thread
            try:
                return MC.run_model_checks(calc, mesh, results, limits, opt, workdir=wd, parts=parts, study_dir=study_dir, log=t.log.emit, cancel=t.cancel,
                                           on_check=t.outcome.emit)
            except CheckCancelled:
                return None                                       # a cancellation is not an error: no traceback in the log
        self._work_dir = wd
        self._results = {}
        self._stale = False
        self._snapshot, self._opt = snapshot, opt
        self._done = 0
        self._show_results()
        t = TaskThread(job, win)                                  # a child of the window, not of the dialog: the thread outlives a dialog that is closed meanwhile
        t.log.connect(win.log_msg)
        t.outcome.connect(self.on_check)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self.on_finished)
        self.job = t
        win.register_aux(t)
        self.b_run.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.progress.start(f"Check 1/7: {MC.CHECK_TITLES[MC.CHECK_IDS[0]]}", 1.0 / 7)
        win.log_msg("Model quality checks: 7 checks")
        t.start()

    def cancel(self) -> None:
        if self.job is not None:
            self.job.cancel.set()
            from .workers import kill_mesh_workers
            if self._work_dir is not None:                # only the meshing of THIS run: the main window may be meshing too (no folder known = nothing to kill, never everything)
                kill_mesh_workers(under=self._work_dir)
            self.win.log_msg("Cancelling the model quality checks...")

    def on_check(self, res) -> None:
        self._results[res.id] = res
        self._done += 1
        self._fill_rows()
        n = len(MC.CHECK_IDS)
        if self._done < n:
            self.progress.report(self._done / n, (self._done + 1) / n, f"Check {self._done + 1}/{n}: {MC.CHECK_TITLES[MC.CHECK_IDS[self._done]]}")

    def on_done(self, results) -> None:
        if results is None:                                       # cancelled (see job): back to the last finished run
            self.progress.finish("cancelled", "Cancelled")
            self.refresh()
            return
        self._results = {r.id: r for r in results}
        run = MC.make_run(self._snapshot, results, self._opt, getattr(self.win, "model_checks", None))
        self.win.model_checks = run                               # for the Word report
        sess = getattr(self.win, "_mcp_session", None)            # the report that an MCP client asks for is built from the session of the window
        if sess is not None:
            sess.model_checks = run
        self._show_results()
        verdict = MC.overall_verdict(results)
        self.progress.finish("done", f"Model checks: {verdict}")
        self.win.log_msg(MC.format_text(results, self._snapshot))
        self.win.messages.add("error" if verdict == "FAIL" else "info" if verdict == "PASS" else "warning", "results", f"Model quality checks: {verdict}", "Space")
        self.update_state()

    def on_failed(self, text: str) -> None:
        cancelled = self.job is not None and self.job.cancel.is_set()
        self.progress.finish("cancelled" if cancelled else "error", "Cancelled" if cancelled else "Checks failed")
        if not cancelled:
            self.l_verdict.setText(f"<span style='color:#c0392b'>The checks could not be run: {escape(text)}</span>")
            self.win.log_msg(f"Model quality checks ERROR: {text}")
            self.win.messages.add("error", "solver", f"Model quality checks failed: {text}")
        else:
            self.refresh()                                        # back to the last finished run (the cancelled one is not kept)

    def on_finished(self) -> None:
        self.win.unregister_aux(self.job)
        self.job = None
        self.b_run.setEnabled(self.study is not None)
        self.b_cancel.setEnabled(False)
        if self._close_when_done:                                 # closed during the run: the dialog goes away now that its thread has ended
            self._close_when_done = False
            self.done(0)                                          # (a hidden dialog gets no `finished` from close(): the window must forget it)

    def _remember_ui(self) -> None:
        """The ticks and budgets stay with the window while the dialog is closed and opened again (cleared when another project is loaded)."""
        self._iface_ticked = set(self._ticked()) if self.l_iface.count() else self._iface_ticked
        self.win._space_checks_ui = {"iface_ticked": sorted(self._iface_ticked), "mass_text": self.e_mass.text(), "cog_text": [e.text() for e in self.e_cog], "with_convergence": self.cb_conv.isChecked()}

    def _restore_ui(self) -> None:
        ui = getattr(self.win, "_space_checks_ui", None) or {}
        self._iface_ticked = set(ui.get("iface_ticked", []))
        self.e_mass.setText(ui.get("mass_text", ""))
        for e, t in zip(self.e_cog, ui.get("cog_text", [])):
            e.setText(t)
        self.cb_conv.setChecked(bool(ui.get("with_convergence", True)))

    def closeEvent(self, e) -> None:
        self._remember_ui()
        if self.job is not None:                                  # never delete a dialog whose thread is still running: cancel, hide, close when the thread has ended
            self.cancel()
            self._close_when_done = True
            self.hide()
            e.ignore()
            return
        super().closeEvent(e)


def open_checks_dialog(win):
    """Space structures dialog > Model quality checks...: one dialog per window, reused."""
    from .main import APP_NAME, msg
    if win.study is None:
        msg(win, APP_NAME, "Open a model or a project first.")
        return None
    dlg = getattr(win, "_space_checks_dlg", None)
    if dlg is None:
        dlg = win._space_checks_dlg = ModelChecksDialog(win)
        dlg.finished.connect(lambda *_: setattr(win, "_space_checks_dlg", None))
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    else:
        dlg._close_when_done = False                # opened again while a cancelled run is still ending
        if dlg.job is None:
            dlg.refresh()
    dlg.show()
    dlg.raise_()
    return dlg


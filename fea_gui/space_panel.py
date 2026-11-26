"""Space structures (ECSS): requirement set, material allowables and margins of safety (`fea_core.space`).

One non-modal dialog with three tabs: the requirement set (factor table, margin rules, confirmation by the user), the allowables of the
materials of the study, and the margins of safety of the solved study with the singularity justification table (a click on a row marks
the place in the 3D view). The program ships no standard number: everything here is entered and confirmed by the user, and an edit
after the confirmation withdraws it. Computation and checks are in `fea_core.space` (the same as MCP and the Word report).
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from fea_core.space import margins as MG
from fea_core.space import requirements as RQ
from fea_core.space.model import (ALLOWABLE_KEYS, BUCKLING_ALLOWABLE, PART_CLASSES, STRESS_MEASURES, THERMAL_HANDLING,
                                  VERIFICATION_APPROACHES, MaterialAllowable, RequirementError, SpaceSettings)

from .space_limits_view import limits_html
from .widgets import fmt, help_tip, table

GREEN, RED, AMBER = QColor("#1b7f3b"), QColor("#c0392b"), QColor("#b36b00")
STATUS_TEXT = {"pass": "PASS", "fail": "FAIL", "unconfirmed": "unconfirmed"}


def _item(text, editable: bool = False, right: bool = False, color: QColor | None = None) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    flags = it.flags()
    it.setFlags(flags | Qt.ItemFlag.ItemIsEditable if editable else flags & ~Qt.ItemFlag.ItemIsEditable)
    if right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color is not None:
        it.setForeground(color)
    return it


def _editable_table(cols: list[str], placeholder: str = "") -> QTableWidget:
    t = table(cols, placeholder=placeholder)
    triggers = QAbstractItemView.EditTrigger
    t.setEditTriggers(triggers.DoubleClicked | triggers.EditKeyPressed | triggers.AnyKeyPressed)
    return t


def _number(text: str, what: str) -> float | None:
    s = text.strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{what}: a number is expected") from None


class SpaceDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Space structures (ECSS)")
        self.setModal(False)
        self.resize(1020, 760)
        self._loading = False
        self._factor_rows: list = []
        self.summary: MG.MarginSummary | None = None
        self._rows: list[MG.MarginRow] = []
        self._sings: list[MG.SingularityRow] = []
        lay = QVBoxLayout(self)
        intro = QLabel("Requirement set, material allowables and margins of safety (MoS = allowable / (J · σ) − 1). The program contains "
                       "NO standard number: enter the factors from your project's documents and confirm them. Until the requirement set "
                       "and the allowables are confirmed, the result is UNCONFIRMED_REQUIREMENTS and never PASS.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_requirements(), "Requirement set")
        self.tabs.addTab(self._build_allowables(), "Material allowables")
        self.tabs.addTab(self._build_margins(), "Margins of safety")
        from .space_loads_panel import add_loads_tabs                 # launch environment / launch loads tabs (F14 S2)
        add_loads_tabs(self)
        from .space_s4_panel import add_s4_tab                # stage S4: interfaces, notching, bolts, PCB, rails, limits (own module)
        add_s4_tab(self)
        lay.addWidget(self.tabs, 1)
        h = QHBoxLayout()
        h.addWidget(QLabel("Confirmed by:"))
        self.by = QLineEdit()
        help_tip(self.by, "Name of the person who confirms (stored with the confirmation and printed in the report)")
        h.addWidget(self.by, 1)
        b = QPushButton("Model quality checks...")
        help_tip(b, "FEM model checks before the margins are used: mass, free-free modes, unit motions, isothermal, modal mass, mesh, equilibrium")
        b.clicked.connect(self.open_checks)
        h.addWidget(b)
        b = QPushButton("Close")
        help_tip(b, "Close the dialog")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)
        self.refresh()

    # ------------------------------------------------------------------ access to the study
    @property
    def study(self):
        return self.win.study

    def space(self, create: bool = False) -> SpaceSettings | None:
        st = self.study
        if st is None:
            return None
        if st.space is None and create:
            st.space = SpaceSettings()
        return st.space

    def rs(self):
        sp = self.space()
        return sp.requirement_set if sp is not None else None

    def _install_set(self, rs) -> None:
        sp = self.space(True)
        if sp is not None:
            sp.requirement_set = rs

    def _edited(self, keep_summary: bool = False) -> None:
        """A user edit of the study data: the project is changed (file only – the results stay valid). The margins shown were computed
        from the data before the edit, so they are cleared (`keep_summary` = only a note of a singularity changed, which moves no margin)."""
        mark = getattr(self.win, "mark_changed", None)
        if callable(mark):
            mark("none")
        if not keep_summary and self.summary is not None:
            self.summary = None
            self.show_summary()
            self.l_verdict.setText("The data were edited after the margins were computed – compute the margins again.")
        self.update_status()

    def _warn(self, text: str) -> None:
        """Message to the user as plain text (the text may contain names from an imported file)."""
        from .main_defs import msg_via_main
        msg_via_main(self, "Space structures (ECSS)", text)

    def open_checks(self) -> None:
        from .space_checks import open_checks_dialog
        open_checks_dialog(self.win)

    # ------------------------------------------------------------------ tab 1: requirement set
    def _build_requirements(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        f = QFormLayout()
        self.e_name = QLineEdit()
        self.e_standard = QLineEdit()
        self.e_revision = QLineEdit()
        self.c_approach = QComboBox()
        self.c_approach.addItem("(not selected)", "")
        for a in VERIFICATION_APPROACHES:
            self.c_approach.addItem(a.replace("_", " "), a)
        help_tip(self.e_name, "Name of the requirement set (project / launcher)")
        help_tip(self.e_standard, "Standard the set comes from (document identifier)")
        help_tip(self.e_revision, "Edition of the standard (e.g. revision and corrigendum)")
        help_tip(self.c_approach, "Verification approach: selects the column of the factor table")
        f.addRow("Name", self.e_name)
        f.addRow("Standard", self.e_standard)
        f.addRow("Revision", self.e_revision)
        f.addRow("Verification approach", self.c_approach)
        lay.addLayout(f)
        for e in (self.e_name, self.e_standard, self.e_revision):
            e.editingFinished.connect(self._meta_edited)
        self.c_approach.currentIndexChanged.connect(self._meta_edited)
        h = QHBoxLayout()
        for text, tip, slot in (("New from ECSS template", "Empty template: factor rows and rules without values", self.new_template),
                                ("Import…", "Read a requirement set (JSON) or the factor table (CSV)", self.import_set),
                                ("Export…", "Write the requirement set (JSON) or the factor table (CSV)", self.export_set)):
            b = QPushButton(text)
            help_tip(b, tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        self.show_all = QCheckBox("Show all approaches")
        help_tip(self.show_all, "Show the factor rows of every verification approach, not only of the selected one")
        self.show_all.toggled.connect(lambda _: self.fill_factors())
        h.addWidget(self.show_all)
        h.addStretch(1)
        lay.addLayout(h)
        lay.addWidget(QLabel("Factors (value, document, revision, clause: where you took it from)"))
        self.t_factors = _editable_table(["Factor", "Structure class", "Approach", "Value", "Document", "Revision", "Clause", "Note"],
                                         "No requirement set – create one from the template")
        help_tip(self.t_factors, "Double-click a Value / Document / Revision / Clause / Note cell to edit")
        self.t_factors.itemChanged.connect(self._factor_edited)
        lay.addWidget(self.t_factors, 3)
        lay.addWidget(QLabel("Margin rules (the design factor J is the product of the listed factors)"))
        self.t_rules = _editable_table(["Check", "Structure class", "Allowable", "Stress measure", "Factors of J", "Thermal handling",
                                        "Thermal factors", "Clause", "Revision"], "No rules")
        help_tip(self.t_rules, "Which factors make up J for each check; edit the text cells, choose the others from the lists")
        self.t_rules.itemChanged.connect(self._rule_edited)
        lay.addWidget(self.t_rules, 2)
        self.l_issues = QLabel("")
        self.l_issues.setWordWrap(True)
        lay.addWidget(self.l_issues)
        self.l_limits = QLabel("")                                      # the limits and the criteria declared not applicable, read-only, visible at the confirmation
        self.l_limits.setWordWrap(True)
        self.l_limits.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.l_limits)
        h = QHBoxLayout()
        self.l_conf = QLabel("")
        self.l_conf.setWordWrap(True)
        h.addWidget(self.l_conf, 1)
        self.b_confirm = QPushButton("Confirm requirement set")
        help_tip(self.b_confirm, "Confirm that every value was checked against its source document")
        self.b_confirm.clicked.connect(self.confirm_set)
        h.addWidget(self.b_confirm)
        self.b_withdraw = QPushButton("Withdraw confirmation")
        help_tip(self.b_withdraw, "Withdraw the confirmation (the result becomes UNCONFIRMED_REQUIREMENTS)")
        self.b_withdraw.clicked.connect(self.withdraw_set)
        h.addWidget(self.b_withdraw)
        lay.addLayout(h)
        return w

    def fill_requirements(self) -> None:
        rs = self.rs()
        self._loading = True
        try:
            for e, v in ((self.e_name, rs.name if rs else ""), (self.e_standard, rs.standard if rs else ""),
                         (self.e_revision, rs.revision if rs else "")):
                e.setText(v)
                e.setEnabled(rs is not None)
            k = self.c_approach.findData(rs.verification_approach if rs else "")
            self.c_approach.setCurrentIndex(max(k, 0))
            self.c_approach.setEnabled(rs is not None)
        finally:
            self._loading = False
        self.fill_factors()
        self.fill_rules()
        self.update_status()

    def fill_factors(self) -> None:
        rs = self.rs()
        t = self.t_factors
        self._loading = True
        try:
            t.setRowCount(0)
            self._factor_rows = []
            if rs is None:
                return
            ap = rs.verification_approach
            for f in rs.factors:
                if not self.show_all.isChecked() and f.approach not in ("", ap):
                    continue
                r = t.rowCount()
                t.insertRow(r)
                self._factor_rows.append(f)
                for c, (txt, edit) in enumerate(((f.key, False), (f.structure_class, False), (f.approach or "any", False),
                                                 ("" if f.value is None else fmt(f.value, 8), True), (f.document, True), (f.revision, True),
                                                 (f.clause, True), (f.note, True))):
                    t.setItem(r, c, _item(txt, edit, right=(c == 3)))
        finally:
            self._loading = False

    def _factor_edited(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        r, c = item.row(), item.column()
        if not 0 <= r < len(self._factor_rows):
            return
        f = self._factor_rows[r]
        text = item.text()
        try:
            if c == 3:
                v = _number(text, "Value")
                if v is not None and not v > 0:
                    raise ValueError("Value: a positive number is expected")
                f.value = v
            elif c == 4:
                f.document = text.strip()
            elif c == 5:
                f.revision = text.strip()
            elif c == 6:
                f.clause = text.strip()
            elif c == 7:
                f.note = text.strip()
        except ValueError as e:
            self._warn(str(e))
            self._loading = True
            item.setText("" if f.value is None else fmt(f.value, 8))
            self._loading = False
            return
        self._edited()

    def fill_rules(self) -> None:
        rs = self.rs()
        t = self.t_rules
        self._loading = True
        try:
            t.setRowCount(0)
            if rs is None:
                return
            for i, rule in enumerate(rs.rules):
                t.insertRow(i)
                t.setItem(i, 0, _item(rule.check))
                t.setItem(i, 1, _item(rule.structure_class))
                buck = rule.check == "buckling"
                t.setCellWidget(i, 2, self._combo([BUCKLING_ALLOWABLE] if buck else ["", *ALLOWABLE_KEYS], rule.allowable,
                                                  lambda v, rule=rule: self._rule_set(rule, "allowable", v), not buck))
                t.setCellWidget(i, 3, self._combo(["", *STRESS_MEASURES], rule.stress_measure,
                                                  lambda v, rule=rule: self._rule_set(rule, "stress_measure", v), not buck))
                t.setItem(i, 4, _item(", ".join(rule.factors), True))
                t.setCellWidget(i, 5, self._combo(list(THERMAL_HANDLING), rule.thermal_handling,
                                                  lambda v, rule=rule: self._rule_set(rule, "thermal_handling", v), not buck))
                t.setItem(i, 6, _item(", ".join(rule.thermal_factors), True))
                t.setItem(i, 7, _item(rule.clause, True))
                t.setItem(i, 8, _item(rule.revision, True))
        finally:
            self._loading = False

    def _combo(self, values: list[str], current: str, setter, enabled: bool = True) -> QComboBox:
        cb = QComboBox()
        for v in values:
            cb.addItem(v.replace("_", " ") if v else "(not chosen)", v)
        cb.setCurrentIndex(max(cb.findData(current), 0))
        cb.setEnabled(enabled)
        cb.currentIndexChanged.connect(lambda _i, cb=cb: None if self._loading else setter(cb.currentData() or ""))
        return cb

    def _rule_set(self, rule, attr: str, value: str) -> None:
        setattr(rule, attr, value)
        self._edited()

    def _rule_edited(self, item: QTableWidgetItem) -> None:
        rs = self.rs()
        if self._loading or rs is None or not 0 <= item.row() < len(rs.rules):
            return
        rule = rs.rules[item.row()]
        keys = [x.strip() for x in item.text().replace(";", ",").split(",") if x.strip()]
        if item.column() == 4:
            rule.factors = keys
        elif item.column() == 6:
            rule.thermal_factors = keys
        elif item.column() == 7:
            rule.clause = item.text().strip()
        elif item.column() == 8:
            rule.revision = item.text().strip()
        else:
            return
        self._edited()

    def _meta_edited(self, *_a) -> None:
        rs = self.rs()
        if self._loading or rs is None:
            return
        rs.name, rs.standard, rs.revision = self.e_name.text().strip(), self.e_standard.text().strip(), self.e_revision.text().strip()
        ap = self.c_approach.currentData() or ""
        changed = ap != rs.verification_approach
        rs.verification_approach = ap
        if changed:
            self.fill_factors()
        self._edited()

    def new_template(self) -> None:
        if self.study is None:
            return
        rs = self.rs()
        if rs is not None and (rs.factors or rs.rules) and QMessageBox.question(
                self, "Space structures (ECSS)", "Replace the current requirement set by the empty template?") != QMessageBox.StandardButton.Yes:
            return
        self._install_set(RQ.ecss_32_10c_template())
        self.fill_requirements()
        self._edited()

    def import_set(self) -> None:
        if self.study is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import requirement set", "", "Requirement set (*.json *.csv)")
        if not path:
            return
        from fea_core.pathcheck import is_unc
        if is_unc(path):
            self._warn("Network paths (UNC) are not allowed.")
            return
        try:
            p = Path(path)
            if p.stat().st_size > RQ.MAX_JSON_BYTES:
                raise RequirementError("The file is too large")
            text = p.read_text(encoding="utf-8-sig")
            if p.suffix.lower() == ".csv":
                rs = self.rs()
                if rs is None:
                    raise RequirementError("Create a requirement set first (the CSV file holds only the factor table)")
                rs.factors = RQ.factors_from_csv(text)
                rs.withdraw_confirmation()
            else:
                self._install_set(RQ.from_json(text))
        except (OSError, UnicodeDecodeError, RequirementError) as e:
            self._warn(f"The file could not be imported: {e}")
            return
        self.fill_requirements()
        self._edited()
        QMessageBox.information(self, "Space structures (ECSS)",
                                "Imported. The confirmation is not taken over: check the values and confirm the set yourself.")

    def export_set(self) -> None:
        rs = self.rs()
        if rs is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export requirement set", "requirement_set.json", "Requirement set (*.json);;Factor table (*.csv)")
        if not path:
            return
        from fea_core.pathcheck import is_unc
        if is_unc(path):
            self._warn("Network paths (UNC) are not allowed.")
            return
        p = Path(path)
        try:
            p.write_text(RQ.factors_to_csv(rs) if p.suffix.lower() == ".csv" else RQ.to_json(rs), encoding="utf-8")
        except OSError as e:
            self._warn(f"The file could not be written: {e}")

    def confirm_set(self) -> None:
        rs = self.rs()
        if rs is None:
            return
        try:
            RQ.confirm(rs, self.by.text(), via="gui", scope=self.space(True))
        except RequirementError as e:
            self._warn("The requirement set cannot be confirmed yet:\n\n" + "\n".join(f"• {m}" for m in e.issues[:12]))
            return
        self._edited()

    def withdraw_set(self) -> None:
        rs = self.rs()
        if rs is not None:
            rs.withdraw_confirmation()
            self._edited()

    def update_status(self) -> None:
        self.fill_allow_status()
        rs = self.rs()
        self.l_limits.setText(limits_html(rs, self.space()))
        if rs is None:
            self.l_conf.setText("No requirement set: margins are indicative only.")
            self.l_issues.setText("")
            self.b_confirm.setEnabled(False)
            self.b_withdraw.setEnabled(False)
            return
        sp = self.space()
        if sp is not None and sp.requirement_confirmed():
            if rs.confirmed_via == "mcp":
                self.l_conf.setText(f"<b style='color:#b36b00'>Confirmed through MCP</b> (an agent) by {escape(rs.confirmed_by)} on "
                                    f"{escape(rs.date)} - check the values and confirm the set here yourself.")
            else:
                self.l_conf.setText(f"<b style='color:#1b7f3b'>Confirmed</b> by {escape(rs.confirmed_by)} on {escape(rs.date)}"
                                    + (f" (via {escape(rs.confirmed_via.upper())})" if rs.confirmed_via else "") + ".")
        elif rs.confirmed:
            self.l_conf.setText("<b style='color:#c0392b'>NOT confirmed</b> – the set was edited after the confirmation.")
        else:
            self.l_conf.setText("<b style='color:#b36b00'>NOT confirmed</b> – margins stay UNCONFIRMED_REQUIREMENTS.")
        self.b_confirm.setEnabled(True)
        self.b_withdraw.setEnabled(rs.confirmed)
        issues = [i for i in RQ.validate(rs) if i.level == "error"]
        self.l_issues.setText("<span style='color:#c0392b'>" + "<br>".join(escape(str(i)) for i in issues[:6]) + "</span>" if issues else "")

    # ------------------------------------------------------------------ tab 2: material allowables
    def _build_allowables(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("A / B-basis allowables of the materials of the study, in MPa, from the material handbook or the data sheet "
                             "(without them the material strength Re / Rm is used for an indicative margin only)."))
        self.t_allow = _editable_table(["Material", "Basis", "Fy [MPa]", "Fu [MPa]", "Fsy [MPa]", "Fsu [MPa]", "T [°C]", "Source", "Status"],
                                       "No materials")
        help_tip(self.t_allow, "Double-click a cell to edit; basis A or B counts as verified")
        self.t_allow.itemChanged.connect(self._allow_edited)
        lay.addWidget(self.t_allow, 1)
        h = QHBoxLayout()
        b = QPushButton("Confirm selected material")
        help_tip(b, "Confirm the allowables of the selected material (uses the name below)")
        b.clicked.connect(self.confirm_allowable)
        h.addWidget(b)
        b = QPushButton("Withdraw confirmation")
        help_tip(b, "Withdraw the confirmation of the selected material")
        b.clicked.connect(self.withdraw_allowable)
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        return w

    def fill_allowables(self) -> None:
        t = self.t_allow
        self._loading = True
        try:
            t.setRowCount(0)
            st = self.study
            if st is None:
                return
            for r, (name, mat) in enumerate(st.materials.items()):
                t.insertRow(r)
                rec = RQ.allowable_of(mat) or MaterialAllowable()
                name_item = _item(name)
                name_item.setData(Qt.ItemDataRole.UserRole, name)
                t.setItem(r, 0, name_item)
                t.setItem(r, 1, _item(rec.basis, True))
                for c, k in enumerate(("Fy", "Fu", "Fsy", "Fsu"), 2):
                    v = rec.value(k)
                    t.setItem(r, c, _item("" if v is None else fmt(v, 8), True, right=True))
                t.setItem(r, 6, _item("" if rec.temp_C is None else fmt(rec.temp_C, 6), True, right=True))
                t.setItem(r, 7, _item(rec.source, True))
                t.setItem(r, 8, _item(""))
            self.fill_allow_status()
        finally:
            self._loading = False

    def fill_allow_status(self) -> None:
        st = self.study
        if st is None or not hasattr(self, "t_allow"):
            return
        was = self._loading
        self._loading = True
        try:
            for r in range(self.t_allow.rowCount()):
                it = self.t_allow.item(r, 0)
                mat = st.materials.get(it.data(Qt.ItemDataRole.UserRole)) if it else None
                rec = RQ.allowable_of(mat) if mat is not None else None
                if rec is None:
                    text, col = "not entered", AMBER
                elif rec.is_verified():
                    text, col = (f"confirmed by {rec.confirmed_by} through MCP (check it)", AMBER) if rec.confirmed_via == "mcp"                         else (f"confirmed by {rec.confirmed_by}", GREEN)
                elif rec.is_confirmed():
                    text, col = "confirmed, but basis is not A or B", AMBER
                else:
                    probs = rec.problems()
                    text, col = ("NOT confirmed" + (f" ({probs[0]})" if probs else "")), AMBER
                self.t_allow.setItem(r, 8, _item(text, color=col))
        finally:
            self._loading = was

    def _allow_edited(self, item: QTableWidgetItem) -> None:
        st = self.study
        if self._loading or st is None or item.column() in (0, 8):
            return
        name_item = self.t_allow.item(item.row(), 0)
        mat = st.materials.get(name_item.data(Qt.ItemDataRole.UserRole)) if name_item else None
        if mat is None:
            return
        r = item.row()
        try:
            vals = {k: _number(self._cell(r, c), k) for c, k in enumerate(("Fy", "Fu", "Fsy", "Fsu"), 2)}
            temp = _number(self._cell(r, 6), "T")
        except ValueError as e:
            self._warn(str(e))
            self.fill_allowables()
            return
        old = RQ.allowable_of(mat) or MaterialAllowable()
        rec = MaterialAllowable.from_dict({**old.to_dict(), **vals, "temp_C": temp, "basis": self._cell(r, 1).strip(),
                                           "source": self._cell(r, 7).strip()})
        if all(v is None for v in vals.values()) and not rec.basis and not rec.source and temp is None:
            mat.allowables = None
        else:
            RQ.set_allowable(mat, rec)
        self._edited()

    def _cell(self, row: int, col: int) -> str:
        it = self.t_allow.item(row, col)
        return it.text() if it is not None else ""

    def _selected_material(self):
        st = self.study
        r = self.t_allow.currentRow()
        it = self.t_allow.item(r, 0) if r >= 0 else None
        return st.materials.get(it.data(Qt.ItemDataRole.UserRole)) if st is not None and it is not None else None

    def confirm_allowable(self) -> None:
        mat = self._selected_material()
        if mat is None:
            self._warn("Select a material in the table.")
            return
        try:
            RQ.confirm_allowable(mat, self.by.text(), via="gui")
        except RequirementError as e:
            self._warn("\n".join(f"• {m}" for m in e.issues[:8]))
            return
        self._edited()

    def withdraw_allowable(self) -> None:
        mat = self._selected_material()
        rec = RQ.allowable_of(mat) if mat is not None else None
        if mat is not None and rec is not None:
            rec.withdraw_confirmation()
            mat.allowables = rec.to_dict()
            self._edited()

    # ------------------------------------------------------------------ tab 3: margins
    def _build_margins(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        h = QHBoxLayout()
        self.b_compute = QPushButton("Compute margins of safety")
        help_tip(self.b_compute, "Margins of the solved study outside singularities (linear studies)")
        self.b_compute.clicked.connect(self.compute)
        h.addWidget(self.b_compute)
        self.l_verdict = QLabel("")
        self.l_verdict.setWordWrap(True)
        h.addWidget(self.l_verdict, 1)
        lay.addLayout(h)
        lay.addWidget(QLabel("Structure class of the parts (default: metallic)"))
        self.t_parts = _editable_table(["Part", "Structure class"], "No parts")
        self.t_parts.setMaximumHeight(110)
        lay.addWidget(self.t_parts)
        lay.addWidget(QLabel("Lowest margins of safety (a click marks the place in the 3D view)"))
        self.t_mos = table(["Part", "Case", "Check", "MoS", "MoS incl. singularities", "J", "Design stress [MPa]", "Allowable", "Status"],
                           placeholder="Compute the margins first", rows=(3, 9))
        self.t_mos.itemSelectionChanged.connect(self._mos_clicked)
        lay.addWidget(self.t_mos, 2)
        lay.addWidget(QLabel("Singularity justification table (a note is required before sign-off)"))
        self.t_sing = _editable_table(["#", "Part", "Location [mm]", "Cause", "MoS at the place", "Cases", "Justification"], "No singular place")
        self.t_sing.itemSelectionChanged.connect(self._sing_clicked)
        self.t_sing.itemChanged.connect(self._sing_edited)
        help_tip(self.t_sing, "Double-click the Justification cell and write why the singular place is acceptable")
        lay.addWidget(self.t_sing, 2)
        self.l_open = QLabel("")
        self.l_open.setWordWrap(True)
        lay.addWidget(self.l_open)
        return w

    def fill_parts(self) -> None:
        t = self.t_parts
        self._loading = True
        try:
            t.setRowCount(0)
            names = []
            res = self.win.results
            if res is not None and res.mesh is not None and res.mesh.part_names:
                names = list(res.mesh.part_names)
            elif getattr(self.win, "parts", None):
                names = [p.name for p in self.win.parts]
            elif self.study is not None:
                names = list(self.study.part_materials)
            sp = self.space()
            for r, name in enumerate(names):
                t.insertRow(r)
                t.setItem(r, 0, _item(name))
                cur = sp.class_of(name) if sp is not None else "metallic"
                cb = QComboBox()
                for c in PART_CLASSES:
                    cb.addItem(c.replace("_", " "), c)
                cb.setCurrentIndex(max(cb.findData(cur), 0))
                cb.currentIndexChanged.connect(lambda _i, cb=cb, name=name: self._part_class(name, cb.currentData()))
                t.setCellWidget(r, 1, cb)
        finally:
            self._loading = False

    def _part_class(self, part: str, cls: str) -> None:
        if self._loading:
            return
        sp = self.space(True)
        if sp is None:
            return
        if cls == sp.default_class:
            sp.part_classes.pop(part, None)
        else:
            sp.part_classes[part] = cls
        self._edited()

    def compute(self) -> None:
        st, res = self.study, self.win.results
        if st is None or res is None or not (res.steps or res.modes):
            self._warn("Solve the study first: the margins are computed from its results.")
            return
        win = self.win
        try:
            from fea_core.space import loads as LD

            owner = LD.snapshot_owner(win)                                  # the kept 3-axis result read on the GUI thread: the job never touches the window
            kept = owner.kept

            def job(_th=None):
                return LD.sign_off_margins(st, res, owner)                  # with the kept 3-axis envelopes when a current result exists
            run = getattr(win, "run_busy", None)                            # off the GUI thread: the window stays painted (a few seconds on a large model)
            summary = run(job, "Computing the margins…") if run is not None else job()
            # run_busy spins a nested event loop: a queued open / a finishing solve may have replaced the study or the results meanwhile
            if self.study is not st or self.win.results is not res or LD.kept_result(self.win) is not kept:
                self._warn("The study or its results changed while the margins were computed - the outdated margins were discarded. Compute them again.")
                return
            self.summary = summary
        except Exception as e:  # noqa: BLE001 - shown to the user, the dialog stays usable
            self.summary = None
            self._warn(f"The margins could not be computed: {e}")
            return
        self.show_summary()

    def show_summary(self) -> None:
        s = self.summary
        self._loading = True
        try:
            self.t_mos.setRowCount(0)
            self.t_sing.setRowCount(0)
            if s is None:
                self.l_verdict.setText("")
                self.l_open.setText("")
                return
            col = "#1b7f3b" if s.signoff_ready else {MG.FAIL: "#c0392b"}.get(s.verdict, "#b36b00")
            w = s.worst
            self.l_verdict.setText(f"<b style='color:{col}'>{escape(s.verdict_text)}</b>"
                                   + (f" – lowest MoS {fmt(w.mos, 3)} ({escape(w.part)}, {escape(w.case)}, {escape(w.check)})" if w and w.mos is not None else ""))
            self._rows = s.lowest(40) + [r for r in s.rows if r.mos is None][:10]          # rows without a margin: the reason is in the tooltip
            for r, row in enumerate(self._rows):
                self.t_mos.insertRow(r)
                cells = [row.part, row.case, row.check, fmt(row.mos, 3), fmt(row.mos_raw, 3), fmt(row.design_factor, 5),
                         fmt(row.design_stress, 4), fmt(row.allowable, 5), STATUS_TEXT.get(row.status, row.status)]
                for c, txt in enumerate(cells):
                    it = _item(txt, right=3 <= c <= 7)
                    if c == 8:
                        it.setForeground({"pass": GREEN, "fail": RED}.get(row.status, AMBER))
                        it.setToolTip(escape("; ".join(row.reasons[:4])))               # names from files: never rich text
                    if row.mos is None:
                        it.setToolTip(escape("; ".join(row.reasons[:4])))               # names from files: never rich text
                    self.t_mos.setItem(r, c, it)
            self._sings = s.singularities
            for r, sg in enumerate(self._sings):
                self.t_sing.insertRow(r)
                loc = ", ".join(fmt(v, 5) for v in sg.location)
                note = sg.justification.note if sg.justified and sg.justification else ""
                for c, (txt, edit) in enumerate(((f"S{r + 1}", False), (sg.part, False), (loc, False), (sg.cause, False),
                                                 (fmt(sg.mos_raw, 3), False), (", ".join(sg.cases[:3]), False), (note, True))):
                    it = _item(txt, edit, right=c == 4)
                    if c == 6 and not note:
                        it.setToolTip("NOT JUSTIFIED – write a note before sign-off")
                        it.setBackground(QColor("#fdf1e1"))
                    self.t_sing.setItem(r, c, it)
            self.l_open.setText(self._open_html(s))
        finally:
            self._loading = False

    @staticmethod
    def _open_html(s: MG.MarginSummary) -> str:
        """"No open items" only when the summary is ready for sign-off (PASS, every singularity justified) - never under FAIL / UNCONFIRMED."""
        if s.signoff_ready:
            return "<b style='color:#1b7f3b'>No open items – ready for sign-off.</b>"
        items = s.open_items or ["not ready for sign-off"]
        return "<b>Open items:</b><br>" + "<br>".join("• " + escape(i) for i in items[:8])

    def _mark(self, xyz, label: str, singular: bool) -> None:
        view = getattr(self.win, "view", None)
        if view is not None and xyz:
            view.mark(xyz, label, (200, 100, 0) if singular else (20, 20, 20), key="space_mos", below=True)

    def _mos_clicked(self) -> None:
        r = self.t_mos.currentRow()
        rows = self._rows
        if 0 <= r < len(rows) and rows[r].location:
            row = rows[r]
            self._mark(row.location, f"MoS {fmt(row.mos, 3)}: {row.part} / {row.case}", False)

    def _sing_clicked(self) -> None:
        r = self.t_sing.currentRow()
        sings = self._sings
        if 0 <= r < len(sings):
            self._mark(sings[r].location, f"S{r + 1}: {sings[r].cause}", True)

    def _sing_edited(self, item: QTableWidgetItem) -> None:
        sings = self._sings
        if self._loading or item.column() != 6 or not 0 <= item.row() < len(sings):
            return
        sg = sings[item.row()]
        sp = self.space(True)
        text = item.text().strip()
        if sp is None:
            return
        if not text:
            if sg.justification is not None and sg.justification in sp.justifications:
                sp.justifications.remove(sg.justification)
            sg.justification = None
        else:
            sg.justification = MG.add_justification(sp, sg.part, sg.location, text, cause=sg.cause, author=self.by.text().strip())
        self._edited(keep_summary=True)
        if self.summary is not None:
            self.l_open.setText(self._open_html(self.summary))

    # ------------------------------------------------------------------ lifecycle
    def refresh(self) -> None:
        """Reload everything from the study of the window (the dialog is reused between openings)."""
        self.summary = None
        self.fill_requirements()
        self.fill_allowables()
        self.fill_parts()
        self.show_summary()
        self.b_compute.setEnabled(self.study is not None)
        refresh_loads = getattr(self, "loads_refresh", None)         # the S2 tabs (space_loads_panel)
        if callable(refresh_loads):
            refresh_loads()
        s4 = getattr(self, "s4", None)
        if s4 is not None:
            s4.refresh()


def open_space_dialog(win):
    """Tools → Space structures (ECSS)…: the dialog is single per window and reused."""
    from .main import APP_NAME, msg
    if win.study is None:
        msg(win, APP_NAME, "Open a model or a project first.")
        return None
    dlg = getattr(win, "_space_dlg", None)
    if dlg is None:
        dlg = win._space_dlg = SpaceDialog(win)
        dlg.finished.connect(lambda *_: setattr(win, "_space_dlg", None))
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    else:
        dlg.refresh()
    dlg.show()
    dlg.raise_()
    return dlg

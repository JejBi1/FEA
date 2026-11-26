"""Space structures, stage S4: interface forces, notching, ECSS bolts, PCB (Steinberg), CubeSat rails and the limits they need (`fea_core.space.s4_*`).

One tab of the space dialog (`space_panel.SpaceDialog`) with six pages. The calculations are those of the MCP tools (`fea_core.space.s4_run`); this module only reads the
fields, runs the modal analysis in a background thread where one is needed and shows the answer with its verdict. The program ships no standard number: every limit,
factor and convention is a row of the requirement set (page "Limits": value + document / revision / clause); numbers typed into a calculation page are used but never
confirmed, and the verdict stays UNCONFIRMED_REQUIREMENTS until the requirement set is confirmed by the user (tab "Requirement set").
The ECSS bolt chain follows ECSS-E-HB-32-23A Rev.1 (6 February 2023), the clause of each formula is shown with its margin; the deviations of the program are listed on the page.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from fea_core.space import s4_run, s4_store
from fea_core.space.model import LimitEntry
from fea_core.space.s4_limits import LIMIT_DESCRIPTIONS, OBSOLETE_LIMITS, install_template

from fea_core.i18n import tr

from .widgets import fmt, help_tip, table

GREEN, RED, AMBER = "#1b7f3b", "#c0392b", "#b36b00"
VERDICT_COLOUR = {"PASS": GREEN, "FAIL": RED}


def _number(text: str, what: str) -> float | None:
    s = text.strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{what}: a number is expected") from None


def _numbers(text: str, n: int, what: str) -> list[float]:
    vals = [x for x in re.split(r"[\s;]+", text.strip()) if x]
    if len(vals) != n:
        raise ValueError(f"{what}: {n} numbers are expected")
    return [float(v.replace(",", ".")) for v in vals]


def _table_rows(text: str, what: str) -> list[list[float]] | None:
    """A profile typed as lines 'f value' (decimal comma allowed, separators: space, tab, semicolon); empty = None."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            vals = _numbers(line, 2, what)
        except ValueError:
            raise ValueError(f"{what}: every line must be 'frequency value'") from None
        rows.append(vals)
    return rows or None


def _item(text: Any, editable: bool = False, right: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    fl = it.flags()
    it.setFlags(fl | Qt.ItemFlag.ItemIsEditable if editable else fl & ~Qt.ItemFlag.ItemIsEditable)
    if right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return it


def _fill(t: QTableWidget, rows: list[list[Any]], right_from: int = 1) -> None:
    t.setRowCount(0)
    for r, row in enumerate(rows):
        t.insertRow(r)
        for c, v in enumerate(row):
            t.setItem(r, c, _item(v, right=c >= right_from))


def _status(v: Any) -> str:
    return {"PASS": "PASS", "FAIL": "FAIL"}.get(v or "", "UNCONFIRMED")


class _Page(QWidget):
    """A calculation page: fields on top (`self.form`), the button, the verdict line, the table(s) and the notes."""

    def __init__(self, panel: "S4Panel", intro: str):
        super().__init__()
        self.panel = panel
        self.lay = QVBoxLayout(self)
        self.intro = QLabel(intro)
        self.intro.setWordWrap(True)
        self.lay.addWidget(self.intro)
        self.form = QFormLayout()
        self.lay.addLayout(self.form)
        self.row = QHBoxLayout()
        self.lay.addLayout(self.row)
        self.l_verdict = QLabel("")
        self.l_verdict.setTextFormat(Qt.TextFormat.PlainText)
        self.l_verdict.setWordWrap(True)
        self.lay.addWidget(self.l_verdict)
        self.t_main = table([], placeholder="")
        self.lay.addWidget(self.t_main, 3)
        self.l_notes = QLabel("")
        self.l_notes.setTextFormat(Qt.TextFormat.PlainText)
        self.l_notes.setWordWrap(True)
        self.lay.addWidget(self.l_notes)

    def button(self, text: str, tip: str, slot) -> QPushButton:
        b = QPushButton(text)
        help_tip(b, tip)
        b.clicked.connect(slot)
        self.row.addWidget(b)
        return b

    def table_cols(self, t: QTableWidget, cols: list[str]) -> None:
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)

    def show_verdict(self, verdict: str | None, text: str = "") -> None:
        v = verdict or ""
        self.l_verdict.setStyleSheet(f"color:{VERDICT_COLOUR.get(v, AMBER)}; font-weight:bold")
        self.l_verdict.setText((_status(v) + (" - " + tr(text) if text else "")) if v else tr(text))

    def show_notes(self, lines: list[str]) -> None:
        self.l_notes.setText("\n".join(tr(str(x)) for x in lines if x))


class S4Panel(QWidget):
    """The tab "Interfaces, bolts, PCB" of the space dialog."""

    def __init__(self, dlg):
        super().__init__()
        self.dlg = dlg
        self.win = dlg.win
        self._thread = None
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self.page_limits = self._build_limits()
        self.page_if = self._build_interface()
        self.page_notch = self._build_notch()
        self.page_bolts = self._build_bolts()
        self.page_pcb = self._build_pcb()
        self.page_rails = self._build_rails()
        for p, name in ((self.page_if, "Interface forces"), (self.page_notch, "Notching"), (self.page_bolts, "Bolts (ECSS)"), (self.page_pcb, "PCB (Steinberg)"),
                        (self.page_rails, "CubeSat rails"), (self.page_limits, "Limits")):
            self.tabs.addTab(p, name)

    # ------------------------------------------------------------------ access
    @property
    def study(self):
        return self.win.study

    def cache(self) -> dict:
        return self.win.__dict__.setdefault("space_s4_cache", {})

    def _warn(self, text: str) -> None:
        from .main_defs import msg_via_main
        msg_via_main(self, "Space structures (ECSS)", text)

    def _stored(self, section: str, payload: Any, key: str | None = None) -> None:
        s4_store.record(self.study, section, payload, key)
        mark = getattr(self.win, "mark_changed", None)
        if callable(mark):
            mark("none")

    def clear_verdicts(self) -> None:
        """A verdict on a page belongs to the state of the study and of the limits at the time of the calculation: when the dialog is reloaded (another study, another set) or a
        limit is edited the pages show nothing that could be read as the present result (the stored records are judged by `s4_verify` when they are printed)."""
        for page in (self.page_if, self.page_notch, self.page_bolts, self.page_pcb, self.page_rails):
            page.show_verdict(None, "")
            page.show_notes([])
        for page in (self.page_notch, self.page_bolts, self.page_pcb, self.page_rails):
            page.t_main.setRowCount(0)
        self.b_n_export.setEnabled(False)
        self._notch_res = None

    def refresh(self) -> None:
        self.clear_verdicts()
        self.fill_limits()
        st = self.study
        names = [p.name for p in getattr(self.win, "parts", [])] if st is not None else []
        self.c_bolt_joint.clear()
        self.c_bolt_joint.addItem("(enter the stiffness below)", "")
        if st is not None:
            for b in st.bolts:
                self.c_bolt_joint.addItem(b.name, b.name)
        for c in (self.c_bolt_mat, self.c_part_mat):
            c.clear()
            c.addItem("(none)", "")
            if st is not None:
                for m in st.materials:
                    c.addItem(m, m)
        self.e_rail_parts.setPlaceholderText("Rail part names, comma separated" + (f" (parts: {', '.join(names[:6])}...)" if names else ""))

    def _run(self, job, done, page: _Page, busy: str) -> None:
        """Runs job(thread) in the background; done(result) in the GUI thread."""
        from .workers import TaskThread
        if self._thread is not None:
            self._warn("A calculation is already running.")
            return
        t = TaskThread(job, self)
        t.log.connect(self.win.log_msg)
        t.done.connect(done)
        t.failed.connect(lambda m: self._failed(page, m))
        t.finished.connect(lambda: self._finished(t))
        self._thread = t
        self.win.register_aux(t)
        page.show_verdict(None, busy)
        t.start()

    def _finished(self, t) -> None:
        self.win.unregister_aux(t)
        self._thread = None

    def _failed(self, page: _Page, message: str) -> None:
        page.show_verdict(None, "")
        page.show_notes([message])
        self._warn(message)

    def _mesh_and_dir(self):
        mesh = self.win.valid_mesh() if hasattr(self.win, "valid_mesh") else None
        wd = self.win._workdir
        return mesh, wd

    # ------------------------------------------------------------------ interface forces
    def _build_interface(self) -> _Page:
        p = _Page(self, "Interface force and moment of a base excitation: modal base-reaction vectors, sine spectrum and RMS in random. The quasi-static limit of the "
                        "reaction equals the total mass times the base acceleration (checked). Runs the modal basis of the study (supports needed).")
        self.c_if_dir = QComboBox()
        for d in ("x,y,z", "x", "y", "z"):
            self.c_if_dir.addItem(d, d)
        self.e_if_sine = QPlainTextEdit()
        self.e_if_sine.setPlaceholderText("Sine amplitude, lines 'f[Hz] a[g]' (empty = base acceleration of the harmonic settings)")
        self.e_if_asd = QPlainTextEdit()
        self.e_if_asd.setPlaceholderText("ASD, lines 'f[Hz] W[g2/Hz]' (empty = the random settings)")
        for e in (self.e_if_sine, self.e_if_asd):
            e.setMaximumHeight(70)
        self.e_if_zeta = QLineEdit()
        help_tip(self.e_if_zeta, "Damping ratio; empty = the damping of the study (uncertain input, +-50 %)")
        self.s_if_modes = QSpinBox()
        self.s_if_modes.setRange(0, 400)
        self.s_if_modes.setSpecialValueText("study")
        help_tip(self.s_if_modes, "Number of modes of the basis (study = the setting of the dynamics)")
        self.e_if_ref = QLineEdit()
        self.e_if_ref.setPlaceholderText("x y z [mm] (empty = centroid of the held nodes)")
        self.c_if_group = QComboBox()
        self.c_if_group.addItem("all", "")
        p.form.addRow("Base axis", self.c_if_dir)
        p.form.addRow("Sine profile", self.e_if_sine)
        p.form.addRow("Random ASD", self.e_if_asd)
        p.form.addRow("Damping ratio", self.e_if_zeta)
        p.form.addRow("Modes", self.s_if_modes)
        p.form.addRow("Moment reference point", self.e_if_ref)
        p.form.addRow("Held nodes (group)", self.c_if_group)
        p.button("Compute interface forces", "Run the modal basis and compute the interface forces and moments", self.run_interface)
        p.table_cols(p.t_main, ["Base axis", "Sine peak |F| [N]", "at f [Hz]", "Sine peak |M| [N·mm]", "Random F rms [N]", "3 sigma F [N]", "Random M rms [N·mm]"])
        self.t_modal = table(["Mode", "f [Hz]", "Gamma x", "Gamma y", "Gamma z", "F_j x", "F_j y", "F_j z"], placeholder="Modal base-reaction vectors appear here")
        p.lay.insertWidget(p.lay.count() - 1, self.t_modal, 2)
        return p

    def run_interface(self) -> None:
        st = self.study
        if st is None:
            return
        p = self.page_if
        try:
            sine, asd = _table_rows(self.e_if_sine.toPlainText(), "sine profile"), _table_rows(self.e_if_asd.toPlainText(), "ASD")
            zeta = _number(self.e_if_zeta.text(), "damping ratio")
            ref = _numbers(self.e_if_ref.text(), 3, "reference point") if self.e_if_ref.text().strip() else None
        except ValueError as e:
            self._warn(str(e))
            return
        mesh, wd = self._mesh_and_dir()
        dirs, modes, group = self.c_if_dir.currentData(), int(self.s_if_modes.value()) or None, self.c_if_group.currentData() or None
        study = copy.deepcopy(st)
        cache = self.cache()

        def job(th):
            import time
            mi, cached = s4_run.get_interface(study, mesh, lambda: wd / f"space_interface_{int(time.time() * 1000)}", cache, modes, ref, log=th.log.emit, cancel=th.cancel)
            return s4_run.interface_report(study, mi, dirs, sine, asd, zeta, group, cached)

        def done(d):
            self._stored("interface", d)
            rows = []
            for ax, item in (d.get("directions") or {}).items():
                s, r = item.get("sine"), item.get("random")
                rows.append([ax, fmt(s["F_peak_max_N"]) if s else "-", fmt(s["f_at_F_peak_Hz"]) if s else "-", fmt(s["M_peak_max_Nmm"]) if s else "-",
                             fmt(r["F_rms_total_N"]) if r else "-", fmt(r["F_nsigma_total_N"]) if r else "-", fmt(r["M_rms_total_Nmm"]) if r else "-"])
            _fill(p.t_main, rows)
            _fill(self.t_modal, [[m["mode"], fmt(m["f_Hz"]), *[fmt(v, 3) for v in m["gamma"]], *[fmt(v, 3) for v in m["F_N"]]] for m in d["modal_table"]])
            cur = self.c_if_group.currentData()
            self.c_if_group.blockSignals(True)
            self.c_if_group.clear()
            self.c_if_group.addItem("all", "")
            for g in d["groups"]:
                if g != "all":
                    self.c_if_group.addItem(g, g)
            i = self.c_if_group.findData(cur)
            self.c_if_group.setCurrentIndex(max(i, 0))
            self.c_if_group.blockSignals(False)
            chk = d.get("static_mass_check") or {}
            p.show_verdict(None, f"total mass {fmt(d['total_mass_kg'])} kg, {d['modes']} modes ({d.get('engine')}); static reaction vs mass·a (relative error x y z): "
                                 + ", ".join(fmt(v, 2) for v in chk.get("relative_error", [])))
            p.show_notes([d.get("damping", ""), *d.get("warnings", []), d.get("hint", "")])

        self._run(job, done, p, "Computing the modal basis and the interface forces ...")

    # ------------------------------------------------------------------ notching
    def _build_notch(self) -> _Page:
        p = _Page(self, "Primary notching: the input profile is reduced where the predicted quantity would exceed its limit. The limit, the design factor, the minimum notch "
                        "width and the floor come from the requirement set (page Limits); values typed here are used but never confirmed.")
        self.c_n_kind = QComboBox()
        self.c_n_kind.addItem("sine", "sine")
        self.c_n_kind.addItem("random", "random")
        self.c_n_limit = QComboBox()
        for k, label in (("force", "interface force [N]"), ("moment", "interface moment [N·mm]"), ("probe_acc", "probe acceleration [g]"), ("stress", "stress [MPa]"),
                         ("force_psd", "force PSD [N2/Hz] (random)")):
            self.c_n_limit.addItem(label, k)
        self.c_n_dir = QComboBox()
        for d in ("x", "y", "z"):
            self.c_n_dir.addItem(d, d)
        self.e_n_profile = QPlainTextEdit()
        self.e_n_profile.setPlaceholderText("Profile, lines 'f[Hz] value' (g for sine, g2/Hz for random; empty = the dynamics settings)")
        self.e_n_profile.setMaximumHeight(70)
        self.e_n_limit, self.e_n_j, self.e_n_w, self.e_n_floor, self.e_n_probe = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        for e, tip in ((self.e_n_limit, "Limit value; empty = from the requirement set"), (self.e_n_j, "Design factor J; empty = from the requirement set"),
                       (self.e_n_w, "Minimum notch width [octaves]; empty = from the requirement set"), (self.e_n_floor, "Floor: lowest notched level / specification level"),
                       (self.e_n_probe, "Probe name of the harmonic result (probe acceleration / stress limits)")):
            help_tip(e, tip)
        p.form.addRow("Profile type", self.c_n_kind)
        p.form.addRow("Limited quantity", self.c_n_limit)
        p.form.addRow("Base axis", self.c_n_dir)
        p.form.addRow("Input profile", self.e_n_profile)
        p.form.addRow("Limit", self.e_n_limit)
        p.form.addRow("Design factor J", self.e_n_j)
        p.form.addRow("Minimum notch width [oct]", self.e_n_w)
        p.form.addRow("Floor ratio", self.e_n_floor)
        p.form.addRow("Probe", self.e_n_probe)
        p.button("Compute notched profile", "Compute the notches and the justification table", self.run_notch)
        self.b_n_export = p.button("Export CSV...", "Write the notched profile and the justification table as CSV", self.export_notch)
        self.b_n_export.setEnabled(False)
        p.table_cols(p.t_main, ["#", "f lo [Hz]", "f hi [Hz]", "Width [oct]", "Input unnotched", "Input notched", "Reduction [dB]", "Quantity unnotched", "Quantity notched",
                                "Limit", "Margin", "Remark"])
        self._notch_res = None
        return p

    def run_notch(self) -> None:
        st = self.study
        if st is None:
            return
        p = self.page_notch
        try:
            kw: dict[str, Any] = dict(limit=_number(self.e_n_limit.text(), "limit"), design_factor=_number(self.e_n_j.text(), "design factor"),
                      min_width_oct=_number(self.e_n_w.text(), "minimum notch width"), floor_ratio=_number(self.e_n_floor.text(), "floor ratio"),
                      profile=_table_rows(self.e_n_profile.toPlainText(), "profile"), probe=self.e_n_probe.text().strip() or None)
        except ValueError as e:
            self._warn(str(e))
            return
        kind, lk, direction = self.c_n_kind.currentData(), self.c_n_limit.currentData(), self.c_n_dir.currentData()
        mesh, wd = self._mesh_and_dir()
        study = copy.deepcopy(st)
        cache = self.cache()
        dyn = getattr(self.win.results, "dynamics", None) if self.win.results is not None else None

        def job(th):
            import time
            return s4_run.notch_report(study, kind, direction, lk, dynamics=dyn, interface=lambda: s4_run.get_interface(
                study, mesh, lambda: wd / f"space_interface_{int(time.time() * 1000)}", cache, None, None, log=th.log.emit, cancel=th.cancel)[0], **kw)

        def done(res):
            d, r = res
            self._notch_res = r
            self._stored("notch", d, f"{kind} {direction} {lk}")
            rows = [[j["notch"], fmt(j["f_lo_Hz"]), fmt(j["f_hi_Hz"]), fmt(j["width_oct"], 3), fmt(j["input_unnotched"]), fmt(j["input_notched"]), fmt(j["reduction_dB"], 3),
                     fmt(j["quantity_unnotched_peak"]), fmt(j["quantity_notched_peak"]), fmt(j["limit"]), fmt(j["margin_remaining"], 3),
                     ("widened; " if j["widened_to_min_width"] == "yes" else "") + ("floor-limited; " if j["floor_limited"] == "yes" else "") + j["status"]]
                    for j in d["justification"]]
            _fill(p.t_main, rows)
            p.show_verdict(d["verdict"], d["verdict_text"])
            table_note = []
            if d.get("table_ratio") is not None:                      # the tolerance and the achieved ratio of the check of the exported table are disclosed
                table_note = [tr("Check of the exported breakpoint table between the analysis frequencies: J x response / limit = {0} (accepted up to 1 + {1}, the numerical "
                                 "tolerance of the check; frequency grid refined {2} time(s) to {3} points)").format(fmt(d["table_ratio"], 5), fmt(d.get("table_tolerance"), 3),
                                                                                                                     max(int(d.get("grid_level", 1)) - 1, 0), d.get("grid_points", "-"))]
            p.show_notes([*d["warnings"], *table_note, *(["Not usable in the requirement set: " + ", ".join(d["limits_needed"])] if d["limits_needed"] else [])])
            self.b_n_export.setEnabled(True)

        self._run(job, done, p, "Computing ...")

    def export_notch(self) -> None:
        from fea_core.space import notching as NT
        r = self._notch_res
        if r is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export the notched profile", "notched_profile.csv", "CSV (*.csv)")
        if not path:
            return
        from pathlib import Path
        p = Path(path).with_suffix(".csv")
        p.write_text(NT.profile_csv(r), encoding="utf-8", newline="")
        p.with_name(p.stem + "_justification.csv").write_text(NT.justification_csv(r), encoding="utf-8", newline="")
        self.win.log_msg(f"Notched profile written: {p}")

    # ------------------------------------------------------------------ bolts
    def _build_bolts(self) -> _Page:
        p = _Page(self, "Formula chain after ECSS-E-HB-32-23A Rev.1 (6 February 2023); the clause of each formula is shown next to its margin. The safety factor J multiplies the "
                        "external load and the thermally induced load, not the preload. Every factor and coefficient comes from the requirement set (page Limits). Forces in N, lengths in mm, stiffness in N/mm.")
        self.c_bolt_joint = QComboBox()
        help_tip(self.c_bolt_joint, "Bolt of the study: stiffness, area and preload from its VDI 2230 evaluation")
        self.c_bolt_mat, self.c_part_mat = QComboBox(), QComboBox()
        help_tip(self.c_bolt_mat, "Material of the bolt: Fy, Fu, Fsy, Fsu from its confirmed allowables")
        help_tip(self.c_part_mat, "Material of the clamped / threaded part: thread Fsu and bearing allowable")
        self.f_bolt: dict[str, QLineEdit] = {}
        p.form.addRow("Bolt of the study", self.c_bolt_joint)
        p.form.addRow("Bolt material", self.c_bolt_mat)
        p.form.addRow("Part material", self.c_part_mat)
        for key, label, tip in (("d_mm", "Diameter d", "Nominal diameter [mm]"), ("pitch_mm", "Pitch", "Thread pitch [mm]"), ("A_s_mm2", "Stress area A_s", "Tensile stress area [mm2]"),
                                ("k_b", "Bolt stiffness", "[N/mm]"), ("k_c", "Clamped parts stiffness", "[N/mm]"), ("F_nom", "Nominal preload", "[N]"),
                                ("F_nom_min", "Lowest nominal preload", "[N]; empty = nominal preload (lower for a self-locking nut)"),
                                ("embedding_um", "Embedding", "Sum of the settlements [um]"), ("alpha_bolt", "Bolt expansion", "[1/K]"),
                                ("alpha_c", "Clamped parts expansion", "[1/K], one layer of the grip length"), ("L_grip_mm", "Grip length", "[mm]"),
                                ("T_ref", "Assembly temperature", "[degC]"), ("T_min", "Minimum temperature", "[degC]"), ("T_max", "Maximum temperature", "[degC]"),
                                ("F_ax", "Axial force (separating)", "Limit load per bolt [N]"), ("F_sh", "Shear force", "Limit load per bolt [N]"),
                                ("F_K_req", "Required clamp force", "[N]; empty = 0 (the joint only must not open)"),
                                ("engaged_mm", "Engaged thread", "[mm]"), ("wrench_mm", "Nut width across flats", "[mm]; empty = threaded hole (coefficient c1 = 1)"), ("d_w_mm", "Bearing face diameter", "[mm]"), ("d_h_mm", "Hole diameter", "[mm]"),
                                ("t_bearing_mm", "Plate thickness (hole bearing)", "[mm]")):
            e = QLineEdit()
            help_tip(e, tip)
            self.f_bolt[key] = e
            p.form.addRow(label, e)
        p.button("Assess the bolt", "Compute the margins of safety of the bolt", self.run_bolt)
        p.table_cols(p.t_main, ["Check", "Demand", "Capacity", "J", "J_T", "MoS", "Status", "Formula"])
        return p

    def run_bolt(self) -> None:
        st = self.study
        if st is None:
            return
        p = self.page_bolts
        raw: dict[str, Any] = {"name": self.c_bolt_joint.currentData() or "bolt"}
        try:
            vals = {k: _number(e.text(), k) for k, e in self.f_bolt.items()}
            if self.c_bolt_joint.currentData():
                raw["joint"] = self.c_bolt_joint.currentData()
            for k, v in vals.items():
                if v is None or k == "alpha_c":
                    continue
                raw[k] = v
            if vals.get("alpha_c") is not None and vals.get("L_grip_mm"):
                raw["layers"] = [{"t_mm": vals["L_grip_mm"], "alpha": vals["alpha_c"]}]
        except ValueError as e:
            self._warn(str(e))
            return
        try:
            d = s4_run.bolts_report(st, [raw], self.c_bolt_mat.currentData() or None, self.c_part_mat.currentData() or None)
        except s4_run.S4Error as e:
            self._warn(str(e))
            return
        r = d["bolts"][0]
        self._stored("bolts", r, str(r["bolt"]))
        _fill(p.t_main, [[m["check"], fmt(m["demand"]), fmt(m["capacity"]), fmt(m["design_factor"], 4), fmt(m.get("thermal_factor"), 4), fmt(m["mos"], 3), _status(m["status"]), m["formula"]] for m in r["margins"]],
              right_from=1)
        p.show_verdict(r["verdict"], r["verdict_text"])
        th = r["dF_th_N"]
        unconf = [f"{m['check']}: {u}" for m in r["margins"] for u in m.get("unconfirmed", [])]
        p.show_notes([f"Phi = {fmt(r['phi'], 4)}; preload F_p,min / F_p,max = {fmt(r['F_pmin_N'])} / {fmt(r['F_pmax_N'])} N; thermal preload change at T_min / T_max = "
                      f"{fmt(th.get('T_min'))} / {fmt(th.get('T_max'))} N (the preload range above carries J_T = 1; each check applies its factor J_T, column J_T, to the thermally induced increment)", d["formula_note"], *[w for w in r["warnings"] if w != d["formula_note"]], *unconf,
                      *(["Not usable in the requirement set: " + ", ".join(d["limits_needed"])] if d["limits_needed"] else []),
                      "Deviations and limits of the program:", *[f"  - {x}" for x in d["deviations"]]])

    # ------------------------------------------------------------------ PCB
    def _build_pcb(self) -> _Page:
        p = _Page(self, "PCB vibration check (Steinberg): Z_allow = c·B / (C·h·r·sqrt(L)), margin and three-band damage. The constant c with its unit, the amplitude convention, "
                        "the design factor and the fatigue law come from the requirement set (page Limits).")
        self.f_pcb: dict[str, QLineEdit] = {}
        for key, label, tip in (("name", "Component", "Name"), ("B_mm", "Board edge B", "Board edge parallel to the component [mm]"), ("h_mm", "Board thickness h", "[mm]"),
                                ("L_mm", "Component length L", "[mm]"), ("C", "Component constant C", "Typed C is never confirmed; empty = limits row"),
                                ("r", "Position factor r", "Relative position factor; empty = from the position"),
                                ("x_mm", "Position x", "[mm] from the board corner"), ("y_mm", "Position y", "[mm]"), ("a_mm", "Board size a", "[mm]"), ("b_mm", "Board size b", "[mm]"),
                                ("r_source", "Source of r", "Source of a typed r (drawing, analysis); none = not confirmed"),
                                ("z_rms_mm", "RMS relative displacement", "1 sigma of the board against its supports [mm]; empty = model"),
                                ("z_source", "Source of the RMS displacement", "Source of a typed RMS displacement (report, test); none = not confirmed"),
                                ("f_n_Hz", "Natural frequency [Hz]", "Cycle rate of the damage"), ("duration_h", "Exposure [h]", "Duration of the exposure")):
            e = QLineEdit()
            help_tip(e, tip)
            self.f_pcb[key] = e
            p.form.addRow(label, e)
        self.e_pcb_centre, self.e_pcb_sup, self.e_pcb_normal = QLineEdit(), QPlainTextEdit(), QLineEdit()
        self.e_pcb_sup.setMaximumHeight(60)
        self.e_pcb_centre.setPlaceholderText("x y z of the board centre [mm]")
        self.e_pcb_sup.setPlaceholderText("Support points, one per line 'x y z' [mm]")
        self.e_pcb_normal.setPlaceholderText("Board normal 'nx ny nz'")
        p.form.addRow("Model: centre", self.e_pcb_centre)
        p.form.addRow("Model: supports", self.e_pcb_sup)
        p.form.addRow("Model: normal", self.e_pcb_normal)
        self.e_pcb_asd = QPlainTextEdit()
        self.e_pcb_asd.setMaximumHeight(60)
        self.e_pcb_asd.setPlaceholderText("ASD lines 'f[Hz] W[g2/Hz]' (empty = the random settings)")
        p.form.addRow("Model: ASD", self.e_pcb_asd)
        p.button("Check the component", "Compute Z_allow, the margin and the damage", self.run_pcb)
        p.table_cols(p.t_main, ["Quantity", "Value"])
        return p

    def run_pcb(self) -> None:
        st = self.study
        if st is None:
            return
        p = self.page_pcb
        try:
            c: dict[str, Any] = {"name": self.f_pcb["name"].text().strip() or "component"}
            for k, e in self.f_pcb.items():
                if k in ("r_source", "z_source"):
                    if e.text().strip():
                        c[k] = e.text().strip()
                elif k != "name" and e.text().strip():
                    c[k] = _number(e.text(), k)
            for k in ("B_mm", "h_mm", "L_mm"):
                c.setdefault(k, 0.0)
            if self.e_pcb_centre.text().strip():                      # also with a typed RMS displacement: it is compared with the model-derived one
                sup = [_numbers(line, 3, "support point") for line in self.e_pcb_sup.toPlainText().splitlines() if line.strip()]
                c["geometry"] = {"centre": _numbers(self.e_pcb_centre.text(), 3, "centre"), "supports": sup, "normal": _numbers(self.e_pcb_normal.text(), 3, "normal")}
            asd = _table_rows(self.e_pcb_asd.toPlainText(), "ASD")
        except ValueError as e:
            self._warn(str(e))
            return
        mesh, wd = self._mesh_and_dir()
        study = copy.deepcopy(st)

        def job(th):
            import time
            return s4_run.pcb_report(study, [c], mesh=mesh, workdir=lambda: wd / f"space_pcb_{int(time.time() * 1000)}", asd=asd, log=th.log.emit)

        def done(d):
            r = d["components"][0]
            self._stored("pcb", r, str(r["component"]))
            db = r.get("damage_three_band") or {}
            _fill(p.t_main, [["r", fmt(r["r"], 4)], ["Z_allow [mm]", fmt(r["Z_allow_mm"])], ["sigma_z [mm]", fmt(r["Z_sigma_mm"])], ["Z_n-sigma [mm]", fmt(r["Z_nsigma_mm"])],
                             ["Design factor J", fmt(r["design_factor"])], ["MoS", fmt(r["MoS"], 3)], ["Damage (three-band)", fmt(db.get("damage"))],
                             ["Damage (narrow band, fatigue.spectral)", fmt(r["damage_narrow_band"])], ["Damage limit", fmt(r["damage_limit"])]], right_from=1)
            p.show_verdict(r["verdict"], r["verdict_text"])
            p.show_notes([*r["warnings"], *(["Not usable in the requirement set: " + ", ".join(d["limits_needed"])] if d["limits_needed"] else [])])

        self._run(job, done, p, "Computing ...")

    # ------------------------------------------------------------------ rails
    def _build_rails(self) -> _Page:
        p = _Page(self, "CubeSat deployer rails: 'Apply supports' writes the wall-normal supports on the outer faces of the rails and the axial support at the rail end(s) "
                        "into the study; 'Check' compares the reactions of the supports and the lateral deflection of the solved study with the limits of the requirement set.")
        self.e_rail_parts = QLineEdit()
        self.c_rail_axis = QComboBox()
        for a in ("x", "y", "z"):
            self.c_rail_axis.addItem(a, a)
        self.c_rail_axial = QComboBox()
        for a, label in (("none", "none"), ("min", "min end"), ("max", "max end"), ("both", "both ends")):
            self.c_rail_axial.addItem(label, a)
        self.f_rail = {k: QLineEdit() for k in ("reaction_limit_N", "axial_reaction_limit_N", "lateral_deflection_limit_mm")}
        p.form.addRow("Rail parts", self.e_rail_parts)
        p.form.addRow("Deployment axis", self.c_rail_axis)
        p.form.addRow("Axial support at", self.c_rail_axial)
        for k, label in (("reaction_limit_N", "Support reaction limit [N]"), ("axial_reaction_limit_N", "Axial reaction limit [N]"),
                         ("lateral_deflection_limit_mm", "Lateral deflection limit [mm]")):
            self.f_rail[k].setPlaceholderText("empty = from the requirement set")
            p.form.addRow(label, self.f_rail[k])
        p.button("Apply supports", "Write the rail supports into the study (replaces earlier rail supports)", self.apply_rails)
        p.button("Check the rails", "Reactions and lateral deflection of the solved study against the limits", self.check_rails)
        p.table_cols(p.t_main, ["Support / rail", "Kind", "State", "Force of the margin / deflection", "Limit", "MoS", "Remark"])
        return p

    def _rail_spec(self):
        from fea_core.space import rails as RL
        parts = [x.strip() for x in self.e_rail_parts.text().replace(";", ",").split(",") if x.strip()]
        v = {k: _number(e.text(), k) for k, e in self.f_rail.items()}
        return RL.RailSpec(parts, self.c_rail_axis.currentData(), self.c_rail_axial.currentData(), reaction_limit_N=v["reaction_limit_N"],
                           axial_reaction_limit_N=v["axial_reaction_limit_N"], lateral_deflection_limit_mm=v["lateral_deflection_limit_mm"])

    def apply_rails(self) -> None:
        from fea_core.space import rails as RL
        st = self.study
        if st is None:
            return
        try:
            res = RL.apply_rail_supports(st, self.win.parts, self._rail_spec())
        except (RL.RailError, ValueError) as e:
            self._warn(str(e))
            return
        self.win.changed(invalidate_mesh=True)
        p = self.page_rails
        _fill(p.t_main, [[s["selection"], s["kind"], "", s["fixed_component"], "", "", f"{s['faces']} face(s)"] for s in res["supports"]], right_from=9)
        p.show_verdict(None, f"{len(res['supports'])} rail supports written into the study")
        p.show_notes(res["warnings"])

    def check_rails(self) -> None:
        from fea_core.space import rails as RL
        st, res = self.study, self.win.results
        if st is None or res is None or not res.steps:
            self._warn("Solve the study first: the rail check reads its reactions and displacements.")
            return
        p = self.page_rails
        try:
            r = RL.check_rails(st, res, self._rail_spec(), s4_run.limits_of(st))
        except (RL.RailError, ValueError) as e:
            self._warn(str(e))
            return
        self._stored("rails", r.to_dict())
        rows = [[x.support, x.kind, x.state, fmt(x.effective_N), fmt(x.limit_N), fmt(x.mos, 3),
                 f"the wall / end plate would have to PULL (state {', '.join(x.pull_states[:2])}); force of the margin = |push| + |pull| of the opposite support" if x.pulls else ""] for x in r.rows]
        rows += [[x["part"], "deflection", x["state"], fmt(x["max_lateral_deflection_mm"]), fmt(x["limit_mm"]), fmt(x["mos"], 3), ""] for x in r.deflection]
        _fill(p.t_main, rows)
        p.show_verdict(r.verdict, r.verdict_text)
        p.show_notes(r.warnings)

    # ------------------------------------------------------------------ limits (rows of the requirement set)
    def _build_limits(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel("The limits, factors and conventions of the checks of this tab. Enter the value with its unit and the document / revision / clause it comes from, then "
                      "confirm the requirement set (tab Requirement set). Nothing here is a standard value; a row without a value is simply not available.")
        info.setWordWrap(True)
        lay.addWidget(info)
        h = QHBoxLayout()
        b = QPushButton("Add the rows of these checks")
        help_tip(b, "Add an empty row for every limit of this tab to the requirement set")
        b.clicked.connect(self.add_limit_rows)
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.t_limits = table(["Key", "Meaning", "Unit", "Value", "Document", "Revision", "Clause", "Note"], placeholder="No requirement set - create one in the tab Requirement set")
        triggers = QAbstractItemView.EditTrigger
        self.t_limits.setEditTriggers(triggers.DoubleClicked | triggers.EditKeyPressed | triggers.AnyKeyPressed)
        self.t_limits.itemChanged.connect(self._limit_edited)
        lay.addWidget(self.t_limits, 1)
        self._limit_loading = False
        return w

    def fill_limits(self) -> None:
        rs = self.dlg.rs()
        self._limit_loading = True
        try:
            self.t_limits.setRowCount(0)
            if rs is None:
                return
            have = {x.key: x for x in rs.limits}
            keys = [k for k in LIMIT_DESCRIPTIONS if k in have] + [k for k in OBSOLETE_LIMITS if k in have]       # obsolete rows stay visible, marked as ignored
            for r, k in enumerate(keys):
                x = have[k]
                self.t_limits.insertRow(r)
                for c, (txt, ed) in enumerate(((k, False), (LIMIT_DESCRIPTIONS[k][1] if k in LIMIT_DESCRIPTIONS else OBSOLETE_LIMITS[k], False), (x.unit, True), ("" if x.value is None else repr(x.value), True),
                                               (x.document, True), (x.revision, True), (x.clause, True), (x.note, True))):
                    it = _item(txt, ed, right=c == 3)
                    if c == 1:
                        it.setToolTip(txt)
                    self.t_limits.setItem(r, c, it)
        finally:
            self._limit_loading = False

    def add_limit_rows(self) -> None:
        rs = self.dlg.rs()
        if rs is None:
            self._warn("Create the requirement set first (tab Requirement set, 'New from ECSS template').")
            return
        added = install_template(rs)
        if added:
            self.dlg._edited()
        self.fill_limits()

    def _limit_edited(self, item: QTableWidgetItem) -> None:
        if self._limit_loading:
            return
        rs = self.dlg.rs()
        if rs is None:
            return
        key_item = self.t_limits.item(item.row(), 0)
        if key_item is None:
            return
        key = key_item.text()
        x = next((y for y in rs.limits if y.key == key), None)
        if x is None:
            x = LimitEntry(key=key)
            rs.limits.append(x)
        txt = item.text().strip()
        try:
            if item.column() == 3:
                x.value = _number(txt, "value")
            else:
                setattr(x, {2: "unit", 4: "document", 5: "revision", 6: "clause", 7: "note"}[item.column()], txt)
        except ValueError as e:
            self._warn(str(e))
            self.fill_limits()
            return
        self.dlg._edited(keep_summary=True)
        self.clear_verdicts()


def add_s4_tab(dlg) -> S4Panel:
    """Adds the tab to the space dialog (called once from `SpaceDialog.__init__`); the dialog calls `panel.refresh()` when it is reloaded."""
    panel = S4Panel(dlg)
    dlg.s4 = panel
    dlg.tabs.addTab(panel, "Interfaces, bolts, PCB")
    panel.refresh()
    return panel


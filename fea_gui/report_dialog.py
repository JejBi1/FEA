""""Word report" dialog: report data → fea_core.report in a thread (the images are drawn by the headless renderer)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout)


class ReportThread(QThread):
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, sess, path, options, parent=None):
        super().__init__(parent)
        self.sess, self.path, self.options = sess, path, options

    def run(self):
        try:
            from fea_core.report import build_report
            self.done.emit(build_report(self.sess, self.path, self.options, on_log=self.log.emit))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


def default_title(study, results) -> str:
    """Report title as in the core (fea_core.report.word._build): Structural / Thermal / Structural and thermal
    FEM calculation. Without results the empty load states of a study with heat are not counted (they are not sent to the calculation).
    """
    therm = bool(study.thermal.cases)
    if results is not None:
        mech = bool(results.steps)
    else:
        mech = any(lc.loads for lc in study.load_cases) if therm else bool(study.load_cases)
    kind = ("FE thermal analysis" if therm and not mech else
            "FE structural and thermal analysis" if therm else "FE structural analysis")
    return f"{kind} – {study.name}"


class ReportDialog(QDialog):
    """Report data; the name and the company are remembered in QSettings."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        st = win.study
        s = win.settings
        self.setWindowTitle("Word report")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        info = []
        if win.results is None:
            info.append("Results are not available – the report will contain only the setup.")
        else:
            info.append(f"Load cases: {len(win.results.steps)}")
            if win.results.thermal:
                info.append(f"thermal cases: {len(win.results.thermal)}")
            if win.joints_panel.assessment is not None:
                info.append("joint checks: yes")
            elif st.active_bolts() or st.active_welds():
                info.append("joint checks: will be run")
            info.append("fatigue: " + ("yes" if win.fat_panel.result is not None else "no (Fatigue step)"))
        lay.addWidget(QLabel(" · ".join(info)))
        f = QFormLayout()
        self.title = QLineEdit(default_title(st, win.results))
        self.title.setToolTip("Report title (title page and footer); empty = default by analysis type")
        self.author = QLineEdit(s.value("report/author", ""))
        self.author.setToolTip("Author name on the title page and in document properties (remembered)")
        self.company = QLineEdit(s.value("report/company", ""))
        self.company.setToolTip("Company on the title page (remembered)")
        self.subject = QPlainTextEdit()
        self.subject.setToolTip("Introductory paragraph below the title page details – subject of the analysis, job, structure description (optional)")
        self.subject.setPlaceholderText("Subject of the analysis, job, structure description…")
        self.subject.setFixedHeight(70)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Author's conclusion (separate paragraphs with an empty line)")
        self.notes.setFixedHeight(90)
        self.notes.setToolTip("Author's own conclusion – a separate “Conclusion” chapter at the end of the report.\nSeparate paragraphs with an empty line; empty = the chapter is omitted.")
        self.safety = QDoubleSpinBox()
        self.safety.setRange(0.1, 10.0)
        self.safety.setSingleStep(0.1)
        self.safety.setDecimals(2)
        req = (getattr(st, "assessment", None) or {}).get("required_safety") if st is not None else None
        self.safety.setValue(float(req) if req is not None else float(s.value("report/safety", 1.0)))  # from the Assessment card
        self.safety.setToolTip("Static assessment PASS / FAIL: min. safety factor to yield strength Re/σ_vM ≥ this value\n(summary and tables per case and part). 3D printing has its own requirement in the 3D printing step.")
        self.images = QCheckBox("images (model, stress, displacement, safety factor)")
        self.images.setChecked(True)
        self.images.setToolTip("Insert images: model with loads for each case, stress, displacement and safety factor maps, fatigue, printing…\nOff = faster, tables and text only.")
        self.keep_images = QCheckBox()
        self.keep_images.setToolTip("Report images (PNG) are saved next to the report in the <name>_obrazky folder, otherwise they are deleted after insertion into Word.\nAn existing foreign folder is not overwritten (the suffix _2, … is used).")
        self.keep_images.setChecked(s.value("report/keep_images", "false") == "true")
        self.images.toggled.connect(self.keep_images.setEnabled)
        self.details = QCheckBox("detailed checks of each joint")
        self.details.setChecked(True)
        self.details.setToolTip("Tables of individual checks of each bolt and weld (VDI 2230, EN 1993-1-8);\noff = only the joint utilization overview")
        self.language = QComboBox()
        from fea_core import i18n
        for key, name in i18n.LANGS.items():
            self.language.addItem(name, key)
        self.language.setProperty("_i18n_done", True)      # language names are not translated
        lang = str(s.value("report/language", "") or i18n.language())
        self.language.setCurrentIndex(max(0, self.language.findData(lang)))
        self.language.setToolTip("Report language (texts, figure captions, number format) – independent of the program language")
        self.ecss = QCheckBox("Structural Analysis Report (ECSS) instead")
        self.ecss.setToolTip("ECSS structural analysis report instead of the calculation report")
        self.ecss.setEnabled(getattr(st, "space", None) is not None)
        self.open_after = QCheckBox("open in Word when finished")
        self.open_after.setChecked(s.value("report/open", "true") == "true")
        self.open_after.setToolTip("Open the report in the associated program (Word) when finished")
        for lbl, w in (("Name", self.title), ("Prepared by", self.author), ("Company", self.company),
                       ("Subject", self.subject), ("Conclusion", self.notes), ("Required safety factor to Re", self.safety),
                       ("Report language", self.language)):
            f.addRow(lbl, w)
        f.addRow(self.images)
        f.addRow(self.keep_images)
        f.addRow(self.details)
        f.addRow(self.ecss)
        f.addRow(self.open_after)
        lay.addLayout(f)
        h = QHBoxLayout()
        self.path = QLineEdit(self._default_path())
        self.path.setToolTip("Path to the output .docx file (an existing file will be overwritten)")
        b = QPushButton("…")
        b.setToolTip("Choose report file")
        b.setFixedWidth(30)
        b.clicked.connect(self._browse)
        h.addWidget(QLabel("File"))
        h.addWidget(self.path)
        h.addWidget(b)
        lay.addLayout(h)
        self.path.textChanged.connect(self._update_keep_label)
        self._update_keep_label()
        self._ecss_tips = {w: w.toolTip() for w in self._ecss_unused()}
        self.ecss.toggled.connect(self._ecss_toggled)
        self._ecss_toggled(self.ecss.isChecked())
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Generate report")
        bb.button(QDialogButtonBox.Ok).setToolTip("Generate the report in the background (progress in the log); if joint checks are missing, they are computed first")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Close without generating the report")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ecss_unused(self):
        """The controls that the Structural Analysis Report (ECSS) does not use: it has no conclusion chapter, no images, no joint checks and no single safety factor."""
        return (self.notes, self.safety, self.images, self.keep_images, self.details)

    def _ecss_toggled(self, on: bool) -> None:
        """With the ECSS report chosen the controls it ignores are disabled (and say why), so nothing typed there is silently dropped."""
        for w in self._ecss_unused():
            w.setEnabled(not on and (w is not self.keep_images or self.images.isChecked()))
            w.setToolTip("Not used by the Structural Analysis Report (ECSS): it has its own chapters, no images and no single safety factor" if on else self._ecss_tips[w])

    def _default_path(self) -> str:
        w = self.win
        base = w.project_path.parent if w.project_path else (Path(w.settings.value("last_dir", "")) or Path.cwd())
        from fea_core.mcp.session import safe_name
        return str(Path(base) / f"{safe_name(w.study.name)}_protokol.docx")

    def _update_keep_label(self, *_):
        stem = Path(self.path.text().strip() or "protokol").stem
        self.keep_images.setText(f"Keep images (folder {stem}_obrazky)")

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save report", self.path.text(), "Word (*.docx)")
        if p:
            self.path.setText(p if p.lower().endswith(".docx") else p + ".docx")

    def options(self):
        from fea_core.report import ReportOptions
        s = self.win.settings
        s.setValue("report/author", self.author.text())
        s.setValue("report/company", self.company.text())
        s.setValue("report/safety", self.safety.value())
        s.setValue("report/open", "true" if self.open_after.isChecked() else "false")
        s.setValue("report/keep_images", "true" if self.keep_images.isChecked() else "false")
        s.setValue("report/language", self.language.currentData())
        return ReportOptions(language=self.language.currentData(), title=(None if self.ecss.isChecked() and self.title.text().strip() == default_title(self.win.study, self.win.results) else self.title.text().strip() or None), author=self.author.text().strip(),
                             company=self.company.text().strip(), subject=self.subject.toPlainText().strip(),
                             notes=self.notes.toPlainText().strip(), required_safety=self.safety.value(),
                             images=self.images.isChecked(), joint_details=self.details.isChecked(),
                             keep_images=self.images.isChecked() and self.keep_images.isChecked(), ecss_sar=self.ecss.isChecked())


def export_report(win) -> None:
    """Step from MainWindow: dialog → (joint assessment) → report in a thread → open in Word."""
    from fea_core.report import session_from
    if win.study is None:
        return
    dlg = ReportDialog(win)
    if dlg.exec() != QDialog.Accepted:
        return
    opts = dlg.options()
    path = Path(dlg.path.text())
    st = win.study
    if win.results is not None and (st.active_bolts() or st.active_welds()) and win.joints_panel.assessment is None:
        win.joints_panel.compute()
    # parts are reloaded from the STEP in a thread: the triangulation of shapes shown in the 3D view would collide with the GUI;
    # the thread gets a copy of the study (the user may change it in the GUI meanwhile)
    from fea_core.space.corr_run import transfer_test_data
    from fea_core.study import Study
    copy = Study.from_dict(st.to_dict())
    transfer_test_data(st, copy)             # the measured test data are recalled by the identity of the study: the copy must inherit them (else the correlation is UNVERIFIABLE)
    sess = session_from(copy, win.step_path, win.results, None, win.fat_panel.result,
                        win.joints_panel.assessment, workdir=tempfile.mkdtemp(prefix="fea_report_"),
                        motion=getattr(win, "motion_result", None))           # Motion section (mechanism)
    sess.param_study = getattr(win, "param_study", None)                       # section Parametric study (param_dialog)
    setattr(sess, "_space_3axis", getattr(win, "_space_3axis", None))                      # 3-axis launch envelopes (space_loads_panel)
    sess.surrogate = getattr(win, "surrogate_model", None)                     # section Surrogate model (param_dialog, tab Surrogate)
    sess.model_checks = getattr(win, "model_checks", None)                       # chapter Model quality checks (space_checks)
    th = ReportThread(sess, path, opts, win)
    th.log.connect(win.log_msg)
    open_after = dlg.open_after.isChecked()

    def done(r):
        win.statusBar().showMessage(f"Report saved: {r['path']}", 10000)
        for w in r.get("warnings", []):
            win.log_msg(f"Report – warning: {w}")
        if open_after:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(r["path"])))

    def failed(text):
        from .main import msg
        msg(win, "Report", f"The report could not be generated:\n{text}")

    th.done.connect(done)
    th.failed.connect(failed)
    th.finished.connect(lambda: setattr(win, "_report_thread", None))
    th.finished.connect(th.deleteLater)
    win._report_thread = th
    win.statusBar().showMessage("Generating report…")
    th.start()

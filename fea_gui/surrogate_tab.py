"""Tab "Surrogate model" of the parametric study dialog (fea_core.surrogate, F15-S4): fit a model with calibrated 90 % error bars to the
design points, validation on the held-out points, sliders with the live prediction (value ± band), 1D / 2D response plots, "Verify
this point with FEM" (one full analysis that joins the training set) and the next points to analyse.

The model is kept in `win.surrogate_model` (saved in the .fea, listed in the report). Fitting and the verification run in a background
thread; the sliders only call `SurrogateModel.predict` (milliseconds). The error bars cover interpolation and mesh noise only.
"""
from __future__ import annotations

import math
import time
from html import escape
from typing import Any, cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSlider, QSpinBox, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from fea_core.surrogate import LIMITATIONS
from . import theme
from .main_defs import msg_via_main
from .widgets import LazyPlot, TaskProgress, fmt, help_tip, table

ORANGE = "#b36b00"
RED = "#d23b3b"
SLIDER_STEPS = 1000
GRID_FINE, GRID_COARSE = 40, 16      # cells per side of the response surface: at rest / while a slider is dragged (every cell is a support test)
CURVE_FINE, CURVE_COARSE = 80, 30
SETTLE_MS = 180                      # the fine picture is drawn this long after the last slider movement
DECIMALS = 4                         # of the value boxes
OVERSHOOT = 0.5                      # a value box may be typed this far (× the range) beyond the sampled range – to see the refusal
BACKENDS = [("Automatic", "auto"), ("Gaussian process (scikit-learn)", "gp"), ("Radial basis (scipy)", "rbf")]


def _item(text: str, right: bool = True) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setTextAlignment((Qt.AlignmentFlag.AlignRight if right else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
    return it


def _plain(text: str) -> QLabel:
    """A label that never interprets its text as rich text (parameter keys carry names of materials, parts and sketch parameters from a study file)."""
    lab = QLabel(text)
    lab.setTextFormat(Qt.TextFormat.PlainText)
    return lab


def _tip(text: str) -> str:
    """A tool tip that shows `text` literally (it may come from a file: never interpreted as markup)."""
    return "<p>" + escape(text) + "</p>"


def _mpl(text: str) -> str:
    """Text for matplotlib without mathtext: a name with $ would be parsed as a formula."""
    return text.replace("$", r"\$")


def _pct(v, dec: int = 0) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100.0 * v:.{dec}f} %"


class SurrogatePanel(QWidget):
    def __init__(self, dlg):
        super().__init__(dlg)
        self.dlg, self.win = dlg, dlg.win
        self.model: Any = getattr(self.win, "surrogate_model", None)          # SurrogateModel | None
        self._job: Any = None                                              # the running TaskThread
        self._coarse = False                                               # a slider is being dragged: the picture is drawn coarsely
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self._settled)
        self._ctl: dict = {}
        self._clamp: dict[str, tuple[float, float]] = {}                   # per parameter: the range of the analysed points as the value box can show it
        self._suggested: list[dict] = []
        self._busy_note = ""
        lay = QVBoxLayout(self)
        intro = QLabel("A surrogate model answers “what is σ_max / displacement / mass / f₁ for these parameter values” in milliseconds, "
                       "with an error bar (90 % when at least 9 points are held out), and refuses where no analysis exists. It is fitted to the "
                       "design points of the study (best: mode Latin hypercube with 30–100 converged points); 20 % of the points (at least 9) "
                       "are held out and calibrate the band.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        h = QHBoxLayout()
        self.val_frac = QSpinBox()
        self.val_frac.setRange(5, 50)
        self.val_frac.setValue(20)
        self.val_frac.setSuffix(" %")
        help_tip(self.val_frac, "Share of the points held out of the fit; they calibrate the error band (at least 9 points: fewer cannot guarantee 90 %)")
        h.addWidget(self.val_frac)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        help_tip(self.seed, "Fixes the hold-out split and the optimiser – the same data give the same model")
        h.addWidget(QLabel("Seed"))
        h.addWidget(self.seed)
        self.backend = QComboBox()
        from fea_core.surrogate.model import backends_available
        avail = backends_available()
        for txt, key in BACKENDS:
            self.backend.addItem(txt, key)
            if not avail.get(key, True) and key != "auto":
                i = self.backend.count() - 1
                cast(QStandardItemModel, self.backend.model()).item(i).setEnabled(False)
        help_tip(self.backend, "Gaussian process needs the module surrogate (scikit-learn); radial basis always works")
        h.addWidget(QLabel("Method"))
        h.addWidget(self.backend, 1)
        f.addRow("Held-out points", h)
        lay.addLayout(f)
        h = QHBoxLayout()
        self.btn_fit = QPushButton("Fit surrogate model")
        theme.set_primary(self.btn_fit)
        help_tip(self.btn_fit, "Fit the model to the design points of the study and validate it on the held-out points")
        self.btn_fit.clicked.connect(self.fit)
        h.addWidget(self.btn_fit)
        h.addStretch(1)
        lay.addLayout(h)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.status)
        self.inner = QTabWidget()
        lay.addWidget(self.inner, 1)
        self._build_quality()
        self._build_explore()
        lim = QLabel(LIMITATIONS)
        lim.setWordWrap(True)
        lim.setProperty("muted", True)
        lay.addWidget(lim)
        self.rebuild()

    # ------------------------------------------------------------------ pages
    def _build_quality(self):
        page = QWidget()
        v = QVBoxLayout(page)
        self.mtab = table(["Result", "Category", "Fit / held-out", "R² (held-out)", "Median error", "Band factor", "Band level", "Coverage (CV)",
                           "Reliable"], 120)
        self.mtab.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.mtab.horizontalHeader().setStretchLastSection(True)
        help_tip(self.mtab, "Accuracy per result on the held-out points; band level = the coverage the calibration guarantees (an average over the "
                            "analysed region); coverage = share of points inside the band (cross-validated)")
        v.addWidget(self.mtab)
        self.val_kpi = QComboBox()
        help_tip(self.val_kpi, "Result shown in the table of the held-out points")
        self.val_kpi.currentIndexChanged.connect(lambda _: self._fill_validation())
        h = QHBoxLayout()
        h.addWidget(QLabel("Held-out points of"))
        h.addWidget(self.val_kpi, 1)
        v.addLayout(h)
        self.vtab = table(["Analysed"], 160)
        help_tip(self.vtab, "Held-out design points: analysed value against the prediction with its band")
        v.addWidget(self.vtab, 1)
        self.inner.addTab(page, "Quality")

    def _build_explore(self):
        page = QWidget()
        v = QVBoxLayout(page)
        self.form = QFormLayout()
        v.addLayout(self.form)
        self.rtab = table(["Result", "Value", "Lower", "Upper", "± band", "Band level", "Reliable"], 110)
        help_tip(self.rtab, "Prediction at the chosen parameter values with its band (band level = the coverage the calibration guarantees)")
        v.addWidget(self.rtab)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.msg)
        h = QHBoxLayout()
        self.p_kpi = QComboBox()
        help_tip(self.p_kpi, "Result drawn in the plot")
        self.p_x = QComboBox()
        help_tip(self.p_x, "Parameter on the horizontal axis")
        self.p_y = QComboBox()
        help_tip(self.p_y, "Second parameter: a 2D map of the result; none = a curve with its band")
        for c in (self.p_kpi, self.p_x, self.p_y):
            c.currentIndexChanged.connect(lambda _: self._plot())
        h.addWidget(self.p_kpi, 1)
        h.addWidget(QLabel("vs."))
        h.addWidget(self.p_x, 1)
        h.addWidget(QLabel("and"))
        h.addWidget(self.p_y, 1)
        v.addLayout(h)
        self.plot_w = LazyPlot()
        help_tip(self.plot_w, "Response of the surrogate model with its band, dots = analysed points; the plot is blank where no analysis exists")
        v.addWidget(self.plot_w, 1)
        h = QHBoxLayout()
        self.btn_verify = QPushButton("Verify this point with FEM")
        help_tip(self.btn_verify, "Run one full analysis at these values, compare it with the band and add it to the training set")
        self.btn_verify.clicked.connect(self.verify)
        h.addWidget(self.btn_verify)
        self.btn_suggest = QPushButton("Suggest next points")
        help_tip(self.btn_suggest, "Where the model is least informed: the points worth analysing next")
        self.btn_suggest.clicked.connect(self.suggest)
        h.addWidget(self.btn_suggest)
        self.btn_run_suggested = QPushButton("Analyse suggested points")
        help_tip(self.btn_run_suggested, "Run the suggested points (full analyses), add them to the training set and refit")
        self.btn_run_suggested.clicked.connect(self.run_suggested)
        self.btn_run_suggested.setEnabled(False)
        h.addWidget(self.btn_run_suggested)
        h.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        h.addWidget(self.btn_cancel)
        v.addLayout(h)
        self.inner.addTab(page, "Explore")

    # ------------------------------------------------------------------ state
    def study_changed(self):
        """The design points changed (a study finished / was loaded): the fit button needs points."""
        res = self.dlg.result
        n = len(res.points) if res is not None else 0
        self.btn_fit.setEnabled(self._job is None and n >= 1)
        if self.model is None and self._job is None:
            self.status.setText(f"{n} design points – run a study (mode Latin hypercube) and fit the model." if n else
                                "Run a parametric study first (mode Latin hypercube gives the best training points).")

    def refresh_stale(self) -> str:
        """Compares the study of the window with the one the model was fitted for ("" = the same; the sentence says what changed)."""
        sm = self.model
        if sm is None or self.win.study is None:
            return ""
        return sm.check_study(self.win.study, getattr(self.win, "step_path", None))

    def showEvent(self, e):
        super().showEvent(e)
        if self.model is not None and self.refresh_stale() != getattr(self, "_shown_stale", ""):
            self.rebuild()

    def rebuild(self):
        """Controls, tables and plot for the current model."""
        self.refresh_stale()
        self._shown_stale = self.model.stale if self.model is not None else ""
        sm = self.model
        has = sm is not None
        self.inner.setEnabled(has)
        for b in (self.btn_verify, self.btn_suggest):
            b.setEnabled(has and self._job is None)
        self.btn_run_suggested.setEnabled(has and bool(self._suggested) and self._job is None)
        self._ctl.clear()
        self._clamp.clear()
        while self.form.rowCount():
            self.form.removeRow(0)
        for c in (self.p_kpi, self.p_x, self.p_y, self.val_kpi):
            c.blockSignals(True)
            c.clear()
        if not has:
            for c in (self.p_kpi, self.p_x, self.p_y, self.val_kpi):
                c.blockSignals(False)
            self.mtab.setRowCount(0)
            self.vtab.setRowCount(0)
            self.rtab.setRowCount(0)
            self.msg.setText("")
            self.study_changed()
            return
        from fea_core.report.param import param_label
        from fea_core.sweep import kpi_columns
        self._label = dict(kpi_columns())
        for k in sm.kpis:
            self.p_kpi.addItem(str(self._label.get(k, k)), k)
            self.val_kpi.addItem(str(self._label.get(k, k)), k)
        self.p_y.addItem("(none)", None)
        for p in sm.continuous:
            self.p_x.addItem(param_label(p.key), p.key)
            self.p_y.addItem(param_label(p.key), p.key)
        if self.p_x.count() > 1:
            self.p_y.setCurrentIndex(0)
        for c in (self.p_kpi, self.p_x, self.p_y, self.val_kpi):
            c.blockSignals(False)
        for p in sm.params:
            if p.kind == "continuous":
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setRange(0, SLIDER_STEPS)
                sp = QDoubleSpinBox()
                span = p.hi - p.lo
                sp.setRange(p.lo - OVERSHOOT * span, p.hi + OVERSHOOT * span)
                sp.setDecimals(DECIMALS)
                # the value box rounds to its decimals: the slider must never end up a rounding step beyond the analysed range (a refusal)
                self._clamp[p.key] = (math.ceil(p.lo * 10 ** DECIMALS) / 10 ** DECIMALS, math.floor(p.hi * 10 ** DECIMALS) / 10 ** DECIMALS)
                sp.setSingleStep(span / 100.0)
                sp.setGroupSeparatorShown(False)
                mid = 0.5 * (p.lo + p.hi)
                sp.setValue(mid)
                sl.setValue(SLIDER_STEPS // 2)
                help_tip(sl, f"Sampled range {fmt(p.lo, 5)} … {fmt(p.hi, 5)}")
                help_tip(sp, "Value; a value outside the sampled range is refused (no prediction)")
                sl.valueChanged.connect(lambda v, p=p, sp=sp: self._slider_moved(p, sp, v))
                sp.valueChanged.connect(lambda v, p=p, sl=sl: self._spin_changed(p, sl, v))
                row = QHBoxLayout()
                row.addWidget(sl, 1)
                row.addWidget(sp)
                self.form.addRow(_plain(param_label(p.key)), row)
                self._ctl[p.key] = (sl, sp)
            else:
                cb = QComboBox()
                cb.setProperty("_i18n_done", True)
                cb.addItems(p.categories)
                cb.currentIndexChanged.connect(lambda _: self._update_prediction())
                self.form.addRow(_plain(param_label(p.key)), cb)
                self._ctl[p.key] = (cb,)
        self._fill_models()
        self._fill_validation()
        self._update_prediction()
        self._show_status()
        self.study_changed()

    def _show_status(self):
        sm = self.model
        if sm is None:
            return
        lines = []
        d = sm.describe()
        if sm.stale:
            lines.append(f"<span style='color:{RED}'>! {escape(sm.stale)}</span>")
        pct = f"{100.0 * sm.guaranteed_level:.0f} %"
        if sm.reliable:
            lines.append(f"Surrogate model ({'Gaussian process' if sm.backend == 'gp' else 'radial basis'}) from {d['n_points']} design points: "
                         f"validated, the {pct} band is guaranteed on average over the analysed region (it can be lower near the edges).")
        elif not sm.stale:
            lines.append(f"<span style='color:{ORANGE}'>! The model is NOT reliable for every result (see Quality) – use it only as a rough guide.</span>")
        if sm.guaranteed_level < sm.level:
            lines.append(f"<span style='color:{ORANGE}'>! The band holds for {pct} of the new points, not {100.0 * sm.level:.0f} %: too few held-out points.</span>")
        lines += [f"<span style='color:{ORANGE}'>! {escape(w)}</span>" for w in d["warnings"][:5]]
        self.status.setText("<br>".join(lines))

    # ------------------------------------------------------------------ fit
    def _points(self):
        res = self.dlg.result
        return res

    def fit(self):
        from fea_core.surrogate import SurrogateError, fit_surrogate
        from .workers import TaskThread
        res = self._points()
        if res is None or not res.points:
            self.status.setText("Run a parametric study first.")
            return
        if self._job is not None or self.dlg.thread is not None:
            return
        keys, pts = list(res.keys), list(res.points)
        bounds = dict(getattr(res, "bounds", {}) or {})
        frac, seed, backend = self.val_frac.value() / 100.0, self.seed.value(), self.backend.currentData()
        fp, sweep = getattr(res, "fingerprint", ""), dict(getattr(res, "sweep", {}) or {})
        if fp and self.win.study is not None:
            from fea_core.surrogate.stale import study_fingerprint
            if study_fingerprint(self.win.study, keys, getattr(self.win, "step_path", None)) != fp:
                self.status.setText(f"<span style='color:{RED}'>The study was changed after the parametric study ran – its points describe "
                                    "another model. Run the study again, then fit.</span>")
                return

        def job(t):
            try:
                return fit_surrogate(pts, keys=keys, validation_fraction=frac, seed=seed, backend=backend, bounds=bounds or None,
                                     fingerprint=fp, sweep=sweep or None)
            except (SurrogateError, ValueError) as e:
                return str(e)
        t = TaskThread(job, self)
        t.done.connect(self._fit_done)
        t.failed.connect(self._failed)
        t.finished.connect(self._finished)
        self._start(t, "Fitting the surrogate model…")

    def _start(self, t, text: str):
        self._job = t
        self.win.register_aux(t)
        for b in (self.btn_fit, self.btn_verify, self.btn_suggest, self.btn_run_suggested):
            b.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.start(text, 0.0)
        t.start()

    def _finished(self):
        self.win.unregister_aux(self._job)
        self._job = None
        self.btn_cancel.setEnabled(False)
        has = self.model is not None
        self.btn_verify.setEnabled(has)
        self.btn_suggest.setEnabled(has)
        self.btn_run_suggested.setEnabled(has and bool(self._suggested))
        res = self.dlg.result
        self.btn_fit.setEnabled(res is not None and bool(res.points))

    def _failed(self, text: str):
        self.progress.finish("error", "Failed")
        self.status.setText(f"<span style='color:{RED}'>{escape(text)}</span>")
        self.win.log_msg(f"Surrogate model ERROR: {text}")

    def _fit_done(self, out):
        if isinstance(out, str):                                    # a refusal: too few points, too many parameters …
            self.progress.finish("error", "Refused")
            self.status.setText(f"<span style='color:{RED}'>{escape(out)}</span>")
            self.win.log_msg(f"Surrogate model: {out}")
            return
        self._set_model(out)
        self.rebuild()
        self.progress.finish("done", "Surrogate model fitted")
        self.win.log_msg(f"Surrogate model fitted: {len(out.rows)} design points, " + ("reliable" if out.reliable else "NOT reliable"))
        self.win.messages.add("info" if out.reliable else "warning", "results", "Surrogate model fitted" + ("" if out.reliable else " – not reliable"), "Solve")

    def _set_model(self, model):
        self.model = model
        self.win.surrogate_model = model                           # saved in the .fea, listed in the report
        self.win.dirty = True
        self.win.update_title()

    def cancel(self):
        if self._job is not None:
            self._job.cancel.set()

    def shutdown(self):
        if self._job is not None:
            self.cancel()
            self._job.wait(15000)

    # ------------------------------------------------------------------ tables
    def _fill_models(self):
        d = self.model.describe()
        t = self.mtab
        t.setRowCount(len(d["members"]))
        for r, m in enumerate(d["members"]):
            vals = [self._label.get(m["kpi"], m["kpi"]), ", ".join(m["category"]) or "–", f"{m['n_train']} / {m['n_validation']}",
                    fmt(m["r2"], 4), _pct(m["median_rel_err"], 1), fmt(m["q"], 3), _pct(m["guaranteed"]), _pct(m["coverage_cv"]),
                    "yes" if m["reliable"] else "NO"]
            for c, txt in enumerate(vals):
                it = _item(txt, c >= 2)
                if c in (6, 8) and m["reasons"]:
                    it.setToolTip(_tip("; ".join(m["reasons"])))
                if (c == 8 and not m["reliable"]) or (c == 6 and m["guaranteed"] is not None and m["guaranteed"] < self.model.level):
                    it.setForeground(Qt.GlobalColor.red)
                t.setItem(r, c, it)
        t.setProperty("_i18n_done", True)

    def _fill_validation(self):
        sm = self.model
        t = self.vtab
        if sm is None or self.val_kpi.currentData() is None:
            t.setRowCount(0)
            return
        from fea_core.report.param import param_label
        keys = [p.key for p in sm.params]
        rows = sm.validation_table(self.val_kpi.currentData())
        t.setColumnCount(len(keys) + 5)
        t.setHorizontalHeaderLabels([param_label(k) for k in keys] + ["Analysed", "Predicted", "Lower", "Upper", "Inside"])
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [fmt(row["params"].get(k), 5) if isinstance(row["params"].get(k), (int, float)) else str(row["params"].get(k)) for k in keys]
            vals += [fmt(row["analysed"], 5), fmt(row["predicted"], 5), fmt(row["lower"], 5), fmt(row["upper"], 5), "yes" if row["inside"] else "NO"]
            for c, txt in enumerate(vals):
                it = _item(txt, c < len(vals) - 1)
                if c == len(vals) - 1 and not row["inside"]:
                    it.setForeground(Qt.GlobalColor.red)
                t.setItem(r, c, it)
        t.setProperty("_i18n_done", True)

    # ------------------------------------------------------------------ prediction
    def _slider_moved(self, p, sp, v):
        lo, hi = self._clamp.get(p.key, (p.lo, p.hi))
        sp.blockSignals(True)
        sp.setValue(min(max(p.lo + (p.hi - p.lo) * v / SLIDER_STEPS, lo), hi))
        sp.blockSignals(False)
        self._coarse = True                          # dragging: a coarse picture now, the fine one when the slider rests
        self._settle.start()
        self._update_prediction()

    def _settled(self):
        self._coarse = False
        self._plot()

    def _spin_changed(self, p, sl, v):
        sl.blockSignals(True)
        sl.setValue(int(round(min(max((v - p.lo) / (p.hi - p.lo), 0.0), 1.0) * SLIDER_STEPS)))
        sl.blockSignals(False)
        self._update_prediction()

    def values(self) -> dict:
        out = {}
        for p in self.model.params:
            w = self._ctl[p.key]
            out[p.key] = w[1].value() if p.kind == "continuous" else w[0].currentText()
        return out

    def _update_prediction(self):
        sm = self.model
        if sm is None or not self._ctl:
            return
        from fea_core.surrogate import SurrogateError
        try:
            r = sm.predict(self.values())
        except SurrogateError as e:
            self.rtab.setRowCount(0)
            self.msg.setText(f"<span style='color:{RED}'>{escape(str(e))}</span>")
            self.btn_verify.setEnabled(False)
            self._plot()
            return
        self.btn_verify.setEnabled(self._job is None and not sm.stale)
        t = self.rtab
        t.setRowCount(len(r["kpis"]))
        for i, (k, v) in enumerate(r["kpis"].items()):
            rel = v["half_width_rel"]
            vals = [self._label.get(k, k), fmt(v["value"], 5), fmt(v["lower"], 5), fmt(v["upper"], 5), "–" if rel is None else f"± {100 * rel:.1f} %",
                    _pct(v["level"]), "yes" if v["reliable"] else "NO"]
            for c, txt in enumerate(vals):
                t.setItem(i, c, _item(txt, c > 0))
        t.setProperty("_i18n_done", True)
        self.msg.setText("<br>".join(f"<span style='color:{ORANGE}'>! {escape(w)}</span>" for w in r["warnings"]))
        self._plot()

    # ------------------------------------------------------------------ plot
    def _fixed(self) -> dict:
        return self.values() if self._ctl else {}

    def _plot(self):
        sm = self.model
        if sm is None or not self._ctl or self.p_kpi.currentData() is None or self.p_x.currentData() is None:
            return
        import numpy as np
        from fea_core.report.param import param_label
        from fea_core.surrogate import SurrogateError
        kpi, kx, ky = self.p_kpi.currentData(), self.p_x.currentData(), self.p_y.currentData()
        fixed = self._fixed()
        grp = sm._group_of(fixed) if sm.categorical else ()
        ax = self.plot_w.axes()
        try:
            if ky is None or ky == kx:
                c = sm.curve(kpi, kx, fixed, CURVE_COARSE if self._coarse else CURVE_FINE, group=grp)
                ok = c["supported"]                                     # only where analyses exist
                ax.plot(c["x"], np.where(ok, c["value"], np.nan), color="#2563eb", lw=1.6, label="surrogate")
                ax.fill_between(c["x"], np.where(ok, c["lower"], np.nan), np.where(ok, c["upper"], np.nan), color="#2563eb", alpha=0.18,
                                label=f"{100.0 * sm.guaranteed_level:.0f} % band")
                m = sm.members.get(grp, {}).get(kpi)
                if m is not None:
                    ids = m.train_ids + m.cal_ids
                    X, y = sm._rows_xy(ids, kpi)
                    j = [p.key for p in sm.continuous].index(kx)
                    ax.plot(X[:, j], y, "o", color="#888888", ms=3.5, alpha=0.8, label="analysed")
                ax.plot([fixed[kx]], [sm.predict_arrays(kpi, grp, np.array([[fixed[p.key] for p in sm.continuous]]))[0][0]], "o", color="#e69a1a", ms=7)
                ax.set_ylabel(_mpl(str(self._label.get(kpi, kpi))))
                ax.legend(fontsize=7, loc="best")
            else:
                g = sm.grid(kpi, kx, ky, fixed, GRID_COARSE if self._coarse else GRID_FINE, group=grp)
                if g["supported"].any():
                    cs = ax.contourf(g["x"], g["y"], np.ma.masked_where(~g["supported"], g["value"]), 14, cmap="viridis")
                    cast(Any, self.plot_w.fig).colorbar(cs, ax=ax, label=_mpl(str(self._label.get(kpi, kpi))))
                else:                                                   # no analysis in this slice of the parameter space
                    ax.text(0.5, 0.5, "No analysis here", transform=ax.transAxes, ha="center", va="center", color="#888888")
                m = sm.members.get(grp, {}).get(kpi)
                if m is not None:
                    keys = [p.key for p in sm.continuous]
                    X, _ = sm._rows_xy(m.train_ids + m.cal_ids, kpi)
                    ax.plot(X[:, keys.index(kx)], X[:, keys.index(ky)], ".", color="white", ms=4, mec="black", mew=0.4)
                ax.plot([fixed[kx]], [fixed[ky]], "o", color="#e69a1a", ms=7, mec="black")
                ax.set_ylabel(_mpl(param_label(str(ky))))
        except SurrogateError:
            pass
        ax.set_xlabel(_mpl(param_label(kx)))
        ax.grid(True, alpha=0.3)
        cast(Any, self.plot_w.canvas).draw_idle()

    # ------------------------------------------------------------------ verification with FEM, next points
    def suggest(self):
        if self.model is None:
            return
        self._suggested = self.model.suggest(3, seed=self.seed.value())
        if not self._suggested:
            self.msg.setText("No suggestion.")
            return
        from fea_core.report.param import param_label
        lines = ["Next points to analyse (the model is least informed there):"]
        for i, p in enumerate(self._suggested, 1):
            lines.append(f"{i}. " + ", ".join(f"{param_label(k)} = {fmt(v, 5) if isinstance(v, (int, float)) else v}" for k, v in p.items()))
        self.msg.setText("<br>".join(escape(x) for x in lines))
        self.btn_run_suggested.setEnabled(self._job is None)

    def run_suggested(self):
        if self._suggested:
            self._analyse(list(self._suggested))

    def verify(self):
        if self.model is not None:
            self._analyse([self.values()])

    def _analyse(self, points: list[dict]):
        """Full analyses at the given parameter points (a study copy per point, as the study itself) → the training set → refit."""
        from fea_core.sweep import ParametricStudySpec, SweepError, check_spec, param_from_key, run_parametric_study
        from fea_core.surrogate import SurrogateError, training_set
        from .workers import TaskThread
        win, sm = self.win, self.model
        if win.study is None or self._job is not None or self.dlg.thread is not None:
            return
        if win.busy():
            QMessageBox.information(self, "Surrogate model", "Meshing or analysis in progress – wait for it to finish.")
            return
        if self.refresh_stale():
            msg_via_main(self, "Surrogate model", sm.stale)
            self.rebuild()
            return
        keys = [p.key for p in sm.params]
        try:
            spec = ParametricStudySpec.from_params([param_from_key(k, [pt[k] for pt in points]) for k in keys], mode="list", max_points=len(points))
            errs = check_spec(win.study, spec)
        except SweepError as e:
            errs = [str(e)]
        if errs:
            msg_via_main(self, "Surrogate model", "\n".join(errs))
            return
        errors, _ = win.check()
        if errors:
            msg_via_main(self, "Surrogate model", "The study is not ready:\n\n" + "\n".join(errors))
            return
        st = win.study_for_solve()
        st.geometry = win.geometry_str()
        mesh = win.valid_mesh()
        new_geo = any(k.startswith("sketch:") for k in keys)
        parts_needed = win.parts and not new_geo and (mesh is None or "mesh_size" in keys)
        wd = win._workdir / f"surrogate_{int(time.time() * 1000)}"
        try:
            predicted = [sm.predict(pt) for pt in points]
        except SurrogateError as e:
            msg_via_main(self, "Surrogate model", str(e))
            return
        from fea_core.surrogate.stale import study_fingerprint
        fp = study_fingerprint(win.study, keys, getattr(win, "step_path", None))

        def job(t):
            parts = win.shadow_solid_parts() if parts_needed else None
            res = run_parametric_study(st, spec, wd, parts=parts, mesh=None if new_geo else mesh, cancel=t.cancel, on_log=t.log.emit)
            good = [p for p in res.points if p.converged]
            for p in good:
                p.adaptive = True                                       # chosen by the model: joins the training set, never the calibration
            new_model = sm.refit_with(training_set(good, keys)[0], study_fingerprint=fp) if good else None
            return res, new_model
        t = TaskThread(job, self)
        t.log.connect(win.log_msg)
        t.done.connect(lambda out, pts=points, pred=predicted: self._analysed(out, pts, pred))
        t.failed.connect(self._failed)
        t.finished.connect(self._finished)
        self._start(t, f"Analysing {len(points)} point(s)…")

    def _analysed(self, out, points, predicted):
        res, new_model = out
        lines = []
        kpi = self.p_kpi.currentData() or self.model.kpis[0]
        for p, pred, dp in zip(points, predicted, res.points):
            if not dp.converged:
                lines.append(f"Point failed: {escape(dp.error[:160])}")
                continue
            v = getattr(dp, kpi, None)
            k = pred["kpis"].get(kpi)
            if k is not None and v is not None and math.isfinite(v):
                inside = k["lower"] <= v <= k["upper"]
                lines.append(f"{escape(str(self._label.get(kpi, kpi)))}: analysis {fmt(v, 5)} against the surrogate {fmt(k['value'], 5)} "
                             f"({fmt(k['lower'], 5)} … {fmt(k['upper'], 5)}) – " + ("inside the band." if inside else
                                                                              f"<span style='color:{RED}'>OUTSIDE the band</span>."))
        if new_model is not None:
            self._set_model(new_model)
            base = self.dlg.result
            if base is not None:                                    # the analysed points join the study table and the report
                nxt = max([q.index for q in base.points] + [-1]) + 1
                for i, dp in enumerate(q for q in res.points if q.converged):
                    dp.index = nxt + i
                    base.points.append(dp)
                self.dlg.points = list(base.points)
                self.dlg._fill_results()
            self._suggested = []
        self.progress.finish("done", "Done")
        self.rebuild()
        self.msg.setText("<br>".join(lines + ["The analysed points were added to the training set and the model was refitted."]) if new_model else "<br>".join(lines))

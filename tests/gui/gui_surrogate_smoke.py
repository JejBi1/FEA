"""Surrogate model in the parametric study dialog (F15-S4): a Latin-hypercube study over two sketch parameters of a plate with a hole, fit,
validation tables, sliders with the live prediction, refusal outside the range, plots, "Verify this point with FEM", save / reopen.

python tests/gui/hidden_run.py tests/gui/gui_surrogate_smoke.py OUT_DIR        (FEA_LANG=cs | en)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_surrogate").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")
_warn: list[str] = []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: _warn.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)

app = QApplication(sys.argv)
import verification.cases  # noqa: E402,F401 – the import order of the verification package
from fea_core.project import save_project  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
from verification.sketch_cases import _plate_study  # noqa: E402

fails: list[str] = []


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


def wait(pred, limit=900.0):
    t0 = time.time()
    while not pred() and time.time() - t0 < limit:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


st = _plate_study(OUT / "base")
proj = OUT / "plate.fea"
save_project(proj, st, st.geometry_path(None))

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()
check(w.study is not None and w.study.sketch is not None, "plate project with a sketch opened")

from fea_gui.extras import open_param_dialog  # noqa: E402

dlg = open_param_dialog(w)
tab = dlg.surrogate_tab
check(dlg.tabs.count() == 2 and not tab.btn_fit.isEnabled(), "tab Surrogate model exists, fit disabled without points")
dlg.mode.setCurrentIndex(dlg.mode.findData("lhs"))
dlg.max_points.setValue(25)
row = dlg.rows[0]
row.kind.setCurrentIndex(row.kind.findData("sketch_param"))
row.target.setCurrentText("R")
row.values.setText("3, 10")
dlg.add_row()
row2 = dlg.rows[1]
row2.kind.setCurrentIndex(row2.kind.findData("sketch_param"))
row2.target.setCurrentText("W")
row2.values.setText("35, 60")
app.processEvents()
check("25" in dlg.est.text() and "seed" in dlg.est.text(), f"estimate ({dlg.est.text()[:90]})")
dlg.btn_run.click()
check(wait(lambda: dlg.thread is None), "Latin-hypercube sketch study finished")
ok = [p for p in dlg.points if p.converged]
check(len(ok) == 25, f"25 points solved ({len(ok)}; errors: {[p.error[:80] for p in dlg.points if not p.converged][:2]})")
check(dlg.result is not None and set(dlg.result.bounds) == {"sketch:R", "sketch:W"}, "sampled box kept in the result")
check(tab.btn_fit.isEnabled(), "fit enabled after the study")
tab.backend.setCurrentIndex(tab.backend.findData("rbf"))
tab.fit()
check(wait(lambda: tab._job is None and tab.model is not None, 120), "surrogate fitted")
sm = tab.model
check(sm is not None and w.surrogate_model is sm and w.dirty, "model kept on the window, project marked changed")
check(tab.mtab.rowCount() >= 1 and tab.vtab.rowCount() >= 9, f"quality tables filled ({tab.mtab.rowCount()} models, {tab.vtab.rowCount()} held-out points)")
check(tab.rtab.rowCount() >= 1 and tab.plot_w.canvas is not None, "prediction table and plot drawn")
tab.inner.setCurrentIndex(1)
dlg.tabs.setCurrentIndex(1)
dlg.grab().save(str(OUT / "surrogate_tab.png"))
# the slider moves the prediction
sl, sp = tab._ctl["sketch:R"]
v0 = tab.rtab.item(0, 1).text()
r_lo, r_hi = next((p.lo, p.hi) for p in tab.model.params if p.key == "sketch:R")   # the slider spans the range of the analysed points, not the requested one
sl.setValue(800)                                                                    # (the very end of one parameter is a point of the hull: refused with the others at mid-range)
app.processEvents()
check(abs(sp.value() - (r_lo + 0.8 * (r_hi - r_lo))) < 1e-3 and 3.0 <= r_lo < r_hi <= 10.0 and tab.rtab.rowCount() >= 1 and tab.rtab.item(0, 1).text() != v0,
      "slider changes the value and the prediction")
sl.setValue(1000)
app.processEvents()
check(abs(sp.value() - r_hi) < 1e-3 and tab.rtab.rowCount() == 0 and any(s in tab.msg.text() for s in ("convex hull", "konvex")),
      "the very end of the range of one parameter (the others at mid-range) is outside the hull: refused with the reason")
sl.setValue(800)
app.processEvents()
sp.setValue(13.0)                                              # beyond the sampled range → refusal, no prediction
app.processEvents()
check(tab.rtab.rowCount() == 0 and any(s in tab.msg.text() for s in ("Extrapolation refused", "Extrapolace odmítnuta")) and not tab.btn_verify.isEnabled(),
      "outside the range: refused")
sp.setValue(6.0)
app.processEvents()
check(tab.rtab.rowCount() >= 1 and tab.btn_verify.isEnabled(), "back inside: predicted again")
tab.p_y.setCurrentIndex(tab.p_y.findData("sketch:W"))
app.processEvents()
check(tab.plot_w.canvas is not None, "2D map drawn")
tab.suggest()
check(len(tab._suggested) == 3 and tab.btn_run_suggested.isEnabled(), "three next points suggested")
# the study changes after the fit: the model says stale, "Verify" is off, the answers are not reliable; back to the same study: fresh again
check(tab.mtab.item(0, 6).text() == "90 %" and tab.rtab.item(0, 5).text() == "90 %", "the band level (90 %, 10 held-out points) is shown in both tables")
old_size = w.study.mesh.size
w.study.mesh.size = old_size * 2.0
tab.rebuild()
check(bool(tab.model.stale) and not tab.btn_verify.isEnabled() and tab.rtab.item(0, tab.rtab.columnCount() - 1).text() == "NO", "changed study: stale, no verification, not reliable")
check("changed after the surrogate" in tab.status.text(), "the status line says what to do")
w.study.mesh.size = old_size
tab.rebuild()
check(not tab.model.stale and tab.btn_verify.isEnabled(), "the study as it was: fresh again")
# verification with a full analysis
n_rows = len(sm.rows)
n_pts = len(dlg.points)
tab.verify()
check(wait(lambda: tab._job is None, 300), "verification analysis finished")
check(tab.model is not None and len(tab.model.rows) == n_rows + 1 and len(dlg.points) == n_pts + 1,
      f"the analysed point joined the training set and the table ({len(tab.model.rows) if tab.model else '-'} rows)")
check(any(s in tab.msg.text() for s in ("inside the band", "OUTSIDE", "uvnitř pásu", "MIMO pás")), "the comparison with the band is shown")
# save and reopen
proj2 = OUT / "plate_surrogate.fea"
check(w._save(proj2), "project saved with the surrogate")
import zipfile  # noqa: E402

with zipfile.ZipFile(proj2) as z:
    check("surrogate.npz" in z.namelist(), "surrogate.npz is in the .fea")
q = tab.values()
before = tab.model.predict(q)["kpis"]["max_stress"]["value"]
dlg.close()
w.open_file(proj2)
app.processEvents()
check(getattr(w, "surrogate_model", None) is not None, "surrogate restored after reopening")
if getattr(w, "surrogate_model", None) is not None:
    after = w.surrogate_model.predict(q)["kpis"]["max_stress"]["value"]
    check(abs(after - before) < 1e-9 * max(1.0, abs(before)), f"the same prediction after reopening ({before:.4f} / {after:.4f})")
dlg2 = open_param_dialog(w)
check(dlg2.surrogate_tab.model is not None and dlg2.surrogate_tab.mtab.rowCount() >= 1, "the dialog shows the restored model")
dlg2.close()
print("\nFAILS:", fails if fails else "none")
w.dirty = False
sys.exit(1 if fails else 0)

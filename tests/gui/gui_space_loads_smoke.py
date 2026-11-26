"""Space structures stage S2 in the real main window: the launch-environment tab (tables typed into the widgets, sources, confirmation, withdrawal by an edit), the S2 limits of
the requirement set, generation of the QSL load cases, the 3-axis run in a thread, the envelope tables and the mass / budget card. Every number is an ARBITRARY TEST VALUE.

python tests/gui/hidden_run.py tests/gui/gui_space_loads_smoke.py OUT_DIR
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_space_loads").resolve()
OUT.mkdir(parents=True, exist_ok=True)
_warn = []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication(sys.argv)
from fea_core.geometry import Part, read_step, write_step  # noqa: E402
from fea_core.project import save_project  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.study import Fixed, LoadCase, MeshSettings, Study  # noqa: E402
import fea_gui.main as gui_main  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
from verification.cases import STEEL, box  # noqa: E402

fails = []
gui_main.msg = lambda parent, title, text, icon=None: _warn.append(str(text))          # a modal box would block the test


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


def wait(pred, limit=600.0):
    t0 = time.time()
    while not pred() and time.time() - t0 < limit:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


step = OUT / "kons.step"
write_step([Part("Konzola", box(0, 0, 0, 80, 10, 10))], step)
parts = read_step(step)
st = Study("Konzola", str(step), mesh=MeshSettings(size=4.0))
st.add_material(STEEL, default=True)
st.add_selection("vetknuti", find_faces(parts, x=0))
st.constraints.append(Fixed("vetknuti"))
st.load_cases.append(LoadCase("Prazdny", []))
st.dynamics.enabled, st.dynamics.modes = True, 6
st.solver.modal_engine = "ccx"
proj = OUT / "kons.fea"
save_project(proj, st, step)

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()

from fea_core.space import environments as EV  # noqa: E402
from fea_core.space import loads as LD  # noqa: E402
from fea_gui.extras import open_space_dialog  # noqa: E402

dlg = open_space_dialog(w)
ctl = dlg.loads
check(dlg is not None and ctl is not None and dlg.tabs.indexOf(ctl.tab_env) >= 0 and dlg.tabs.indexOf(ctl.tab_loads) >= 0
      and dlg.tabs.indexOf(ctl.tab_env) != dlg.tabs.indexOf(ctl.tab_loads), "the dialog has the two stage S2 tabs")
dlg.new_template()                                    # requirement set template (the limit rows are added by the S2 tab)
ctl.add_limit_rows()
check(ctl.t_lim.rowCount() == 14, "the 14 empty S2 limit rows are shown")
rs = w.study.space.requirement_set
check(all(x.value is None for x in rs.limits), "no limit has a value out of the box")

# --- environment tab: type a sine table into the widgets
ctl.c_axis.setCurrentText("z")
check(EV.environments_of(w.study.space).launch_axis == "z", "the launch axis reaches the study")
page = ctl.pages["sine"]
for f, a in ((20.0, 5.0), (100.0, 5.0), (400.0, 8.0)):
    page.add_row()
    r = page.t.rowCount() - 1
    page.t.item(r, 0).setText(str(f))
    page.t.item(r, 1).setText(str(a))
page.fields["sweep_rate_oct_min"].setText("2")
page.fields["document"].setText("test document")
page.fields["revision"].setText("A")
page.fields["clause"].setText("test clause")
page.fields["clause"].editingFinished.emit()
es = EV.environments_of(w.study.space)
check(es is not None and es.level("qualification") is not None and es.level("qualification").sine.points == [[20.0, 5.0], [100.0, 5.0], [400.0, 8.0]],
      "the sine table typed into the widgets reaches the study")
check(es.level("qualification").sine.clause == "test clause" and es.level("qualification").sine.sweep_rate_oct_min == 2.0, "source and sweep rate stored")
rpage = ctl.pages["random"]
for f, v in ((20.0, 0.01), (200.0, 0.03), (1000.0, 0.03)):
    rpage.add_row()
    r = rpage.t.rowCount() - 1
    rpage.t.item(r, 0).setText(str(f))
    rpage.t.item(r, 1).setText(str(v))
for k, v in (("duration_s", "120"), ("document", "test document"), ("revision", "A"), ("clause", "test clause")):
    rpage.fields[k].setText(v)
rpage.fields["clause"].editingFinished.emit()
check(len(EV.environments_of(w.study.space).level("qualification").random.points) == 3, "the random table is stored")
page.t.item(0, 1).setText("abc")
check(bool(_warn), "a cell that is not a number is refused with a message")
_warn.clear()
page.t.item(0, 1).setText("5.123456789012")                                    # more digits than ':g' shows: the tables must not round them
ctl.e_damp.setText("0.03")
ctl.e_damp_src.setText("test report 1")
ctl.e_damp_src.editingFinished.emit()
es = EV.environments_of(w.study.space)
check(es.damping_zeta == 0.03 and es.damping_source == "test report 1", "the damping ratio and its source typed into the widgets reach the environments")
ctl.e_damp.setText("1.5")
ctl.e_damp.editingFinished.emit()
check("damping_zeta" in ctl.l_issues.text(), "a damping ratio outside (0, 1) is reported as a problem")
ctl.e_damp.setText("0.03")
ctl.e_damp.editingFinished.emit()

# --- confirmation of the environments
ctl.confirm_env()
check(bool(_warn) and not EV.environments_of(w.study.space).is_confirmed(), "confirmation without a name is refused")
_warn.clear()
dlg.by.setText("Test Engineer")
ctl.confirm_env()
check(EV.environments_of(w.study.space).is_confirmed() and "Confirmed" in ctl.l_env.text(), "environments confirmed through the widgets (via the GUI)")
check(EV.environments_of(w.study.space).confirmed_via == "gui", "confirmed_via = gui")
ctl.show_level()                                                               # reload the widgets from the study ...
ctl.commit()                                                                   # ... and take them back: no value changed, so the confirmation must stay
check(EV.environments_of(w.study.space).is_confirmed(), "re-reading the unchanged tables does not withdraw the confirmation (no rounding of the cells)")
check(EV.environments_of(w.study.space).level("qualification").sine.points[0][1] == 5.123456789012, "a value with 13 digits survives a reload of the table exactly")
ctl.e_damp.setText("0.031")
ctl.e_damp.editingFinished.emit()
check(not EV.environments_of(w.study.space).is_confirmed(), "an edit of the damping ratio withdraws the confirmation")
ctl.e_damp.setText("0.03")
ctl.e_damp.editingFinished.emit()
dlg.by.setText("Test Engineer")
ctl.confirm_env()
check(EV.environments_of(w.study.space).is_confirmed(), "confirmed again")
page.t.item(1, 1).setText("6")
check(not EV.environments_of(w.study.space).is_confirmed() and "NOT confirmed" in ctl.l_env.text(), "an edit after the confirmation withdraws it")

# --- QSL limits typed into the table, then the generator
for r in range(ctl.t_lim.rowCount()):
    key = ctl.t_lim.item(r, 0).text()
    if key in ("qsl_axial_g", "qsl_lateral_g"):
        ctl.t_lim.item(r, 1).setText("6.5" if key == "qsl_axial_g" else "3.25")
        ctl.t_lim.item(r, 3).setText("test document")
        ctl.t_lim.item(r, 4).setText("A")
        ctl.t_lim.item(r, 5).setText("test clause")
n0 = len(w.study.load_cases)
ctl.generate_qsl()
check(len(w.study.load_cases) == n0 + 14, f"14 QSL load cases generated ({len(w.study.load_cases) - n0})")
check(not _warn, "no message on the generation")

# --- mesh, 3-axis run in a thread, envelopes
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed")
ctl.c_run_level.setCurrentText("qualification")
ctl.k_kind["shock"].setChecked(False)
ctl.run_3axis()
check(not _warn, f"the run started without a message {_warn[:1]}")
check(wait(lambda: (ctl._thread is None and ctl.res3 is not None) or bool(_warn) or "c0392b" in ctl.l_run.text()), "the 3-axis run finished")
r3 = ctl.res3
check(r3 is not None and r3.kinds == ["sine", "random"] and r3.status == "UNCONFIRMED_INPUTS",
      "result: sine + random, UNCONFIRMED_INPUTS (the environment edit withdrew the confirmation)")
check(r3 is not None and r3.current(w.study, ctl._mesh()), "the result of the thread (computed on a copy of the study) is current for the study of the window (digest)")
check(r3 is not None and r3.damping.get("zeta") == 0.03 and r3.damping.get("from_environment") is True and ("0,03" in ctl.l_run.text() or "0.03" in ctl.l_run.text()),"the damping of the environments was used and is shown")
check(r3 is not None and r3.complete and r3.label == "3-axis" and any("no sigma" in b for b in r3.blockers)
      and ("No PASS possible" in ctl.l_run.text() or "PASS není možné" in ctl.l_run.text()),
      "the level has no shock table, so the run is a complete 3-axis run; the random table without sigma forbids a PASS and the tab says so")
check(ctl.t_env.rowCount() == 2 and ctl.t_part.rowCount() >= 2 and "UNCONFIRMED_INPUTS" in ctl.l_run.text(), "the envelope tables are filled and say UNCONFIRMED_INPUTS")
if r3 is not None:
    import numpy as np
    check(np.array_equal(r3.envelopes["random"].S, np.max(np.stack([r3.per_axis["random"][a].S for a in (0, 1, 2)]), axis=0)),
          "the envelope is the maximum of the three axes")
ctl.t_part.selectRow(0)
app.processEvents()
check(not _warn, "clicking an envelope row marks the place without an error")
check(getattr(w, "_space_3axis", None) is not None, "the result is kept on the window for the report")
check(LD.recall(w, w.study, ctl._mesh()) is r3, "the kept result is recalled for the study and the mesh it was computed for")

# --- mass / budget card
ctl.mass_card()
check("kg" in ctl.l_mass.text() and ctl.t_bud.rowCount() >= 1, f"mass card: {ctl.l_mass.text()[:70]}")
check(all(ctl.t_bud.item(r, 5).text() in ("UNCONFIRMED", "SKIPPED") for r in range(ctl.t_bud.rowCount())),
      "an unconfirmed requirement set gives no PASS / FAIL in the budget card")
check(not _warn, f"no message {_warn[:1]}")

# --- the study keeps everything over save / load
saved = OUT / "kons_s2.fea"
check(w._save(saved), "project saved")
from fea_core.project import load_project  # noqa: E402
st2, _s, _r = load_project(saved, OUT / "reload")
es2 = EV.environments_of(st2.space)
check(es2 is not None and es2.level("qualification").random.points[1] == [200.0, 0.03] and LD.qsl_status(st2)["names"] and es2.damping_zeta == 0.03,
      "environments, the damping and the QSL registry survive save / load")
check(LD.qsl_status(w.study)["current_with_requirement_set"], "the generated QSL cases are current")

# --- a change of the study discards the kept result (window handler + digest)
w.study.dynamics.damping.zeta = 0.05
w.mark_changed("results")
check(getattr(w, "_space_3axis", None) is None and LD.recall(w, w.study, ctl._mesh()) is None, "a change of the study discards the kept 3-axis result")
ctl.refresh()
check(ctl.res3 is None and ctl.t_env.rowCount() == 0, "the reopened tab shows no result of the earlier model state")
w.study.dynamics.damping.zeta = 0.02

# --- reopening the dialog shows the stored values
dlg.close()
dlg2 = open_space_dialog(w)
check(dlg2.loads.pages["sine"].t.rowCount() == 3 and dlg2.loads.c_axis.currentText() == "z", "a reopened dialog shows the stored environment")
dlg2.close()
w.close()
print("FAILS:", fails)
sys.exit(1 if fails else 0)

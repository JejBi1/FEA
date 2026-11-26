"""FEM model quality checks (ECSS) dialog in the real main window: button in the Space structures dialog, traffic-light table, budgets, tolerances edited in the table (an edit
withdraws the confirmation), a run in the worker thread without confirmed limits (nothing may be PASS) and with confirmed limits, the kept run and the Word report chapter.

python tests/gui/hidden_run.py tests/gui/gui_space_checks_smoke.py OUT_DIR        (FEA_LANG=cs | en)
The numbers of the limits are EXAMPLE values of the test - the program ships none.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_space_checks").resolve()
OUT.mkdir(parents=True, exist_ok=True)
_warn = []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication(sys.argv)
from fea_core import i18n  # noqa: E402
from fea_core.geometry import Part, read_step, write_step  # noqa: E402
from fea_core.project import save_project  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.space import model_checks as MC  # noqa: E402
from fea_core.space.model import SpaceSettings  # noqa: E402
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study  # noqa: E402
import fea_gui.main as gui_main  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
from verification.cases import STEEL, box  # noqa: E402
from verification import space_cases as SC  # noqa: E402  (after the cases: cyclic import)

fails = []
gui_main.msg = lambda parent, title, text, icon=None: _warn.append(str(text))          # a modal box would block the test
import fea_gui.main_defs as main_defs  # noqa: E402
main_defs.msg_via_main = lambda parent, title, text, *a, **k: _warn.append(str(text))


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


step = OUT / "tyc.step"
write_step([Part("Tyc", box(0, 0, 0, 60, 10, 10))], step)
parts = read_step(step)
st = Study("Tyc", str(step), mesh=MeshSettings(size=4.0))
st.add_material(STEEL, default=True)
st.add_selection("vetknuti", find_faces(parts, x=0))
st.add_selection("konec", find_faces(parts, x=60))
st.constraints.append(Fixed("vetknuti"))
st.load_cases.append(LoadCase("Tah", [Force("konec", [5000.0, 0, 0])]))
st.modal.enabled, st.modal.modes = True, 6
proj = OUT / "tyc.fea"
save_project(proj, st, step)

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()
check(w.study is not None and w.study.name == "Tyc", "project opened")
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed")
w.set_stage(w.STAGE["Solve"])
w.run_solve()
check(wait(lambda: w.results is not None and w._thread is None), "solved")

# --- the requirement set with arbitrary test factors, NOT confirmed, no limit yet
w.study.space = SpaceSettings(requirement_set=SC.make_requirement_set("protoflight", confirmed=False))
rs = w.study.space.requirement_set
from fea_gui.extras import open_space_dialog  # noqa: E402
sp = open_space_dialog(w)
check(sp is not None and hasattr(sp, "open_checks"), "the Space structures dialog has the button of the model checks")
sp.open_checks()
cd = getattr(w, "_space_checks_dlg", None)
check(cd is not None, "the checks dialog opens from the space panel")
from fea_gui.space_checks import open_checks_dialog  # noqa: E402
check(open_checks_dialog(w) is cd, "one dialog per window")
check(cd.t_checks.rowCount() == 7 and all(cd.t_checks.item(i, 3).text() == "not run" for i in range(7)), "traffic-light table: seven rows, nothing run yet")
check(cd.t_lim.rowCount() == len(MC.LIMIT_DESCRIPTIONS), "tolerances table lists every limit of the checks")
check("NOT confirmed" in cd.l_state.text() or "nepotvrzen" in cd.l_state.text().lower(), f"state line: set not confirmed ({cd.l_state.text()[:70]})")

# --- run without a confirmed limit: numbers, but never PASS
cd.cb_conv.setChecked(False)                                  # the mesh convergence runs several meshes - not needed here
cd.e_mass.setText("0.0471")                                   # the bar 60 x 10 x 10 mm of steel [kg]
cd.b_run.click()
check(cd.job is not None and not cd.b_run.isEnabled() and cd.b_cancel.isEnabled(), "the run goes on in a worker thread (Run disabled, Cancel enabled)")
check(wait(lambda: cd.job is None), "the checks finished")
by = {r: cd._results.get(r) for r in MC.CHECK_IDS}
check(all(v is not None for v in by.values()), f"seven results in the table: {[v.status if v else None for v in by.values()]}")
check(all(v.status != MC.PASS for v in by.values() if v), "no limit confirmed: no check says PASS")
check(by["mass"].status == MC.UNCONFIRMED and by["mass"].values.get("fe_mass_kg"), f"mass: numbers shown, UNCONFIRMED ({by['mass'].summary[:70]})")
check(all(cd.t_checks.item(i, 0).background().color().name() == {"UNCONFIRMED": "#8a8f98", "SKIPPED": "#8a8f98", "WARN": "#d99a1e", "FAIL": "#c0392b",
                                                                    "PASS": "#1b7f3b"}[by[c].status].lower() for i, c in enumerate(MC.CHECK_IDS) if by[c]),
      "traffic-light colours follow the status")
check("UNCONFIRMED_REQUIREMENTS" in cd.l_verdict.text() or "INCOMPLETE" in cd.l_verdict.text() or "FAIL" in cd.l_verdict.text(), f"verdict shown: {cd.l_verdict.text()[:90]}")
check(w.model_checks is not None and MC.run_is_current(w.model_checks, w.study), "the run is kept for the report and belongs to the present study")
cd.t_checks.selectRow(0)
check(i18n.tr("Mass and centre of mass") in cd.detail.toPlainText(), "the detail of the selected check is shown (in the language of the program)")

# --- the tolerances are typed into the table: a bare edit does not confirm anything, a requirement-set confirmation is the user's step
check(not rs.is_confirmed() and "Test Engineer" not in str(rs.confirmed_by), "nothing is confirmed yet")
for name, val in (("mass_tol_rel", "0.01"), ("cad_mass_tol_rel", "0.01"), ("solver_mass_tol_rel", "0.001")):
    r = cd._limit_names.index(name)
    cd.t_lim.item(r, 2).setText(val)
    cd.t_lim.item(r, 3).setText("test document")
    cd.t_lim.item(r, 4).setText("test revision")
    cd.t_lim.item(r, 5).setText("test clause")
check({x.key: x.value for x in rs.limits} == {"mass_tol_rel": 0.01, "cad_mass_tol_rel": 0.01, "solver_mass_tol_rel": 0.001}, "the typed limits reach the requirement set")
check(MC.limits_of_study(w.study).value("mass_tol_rel") is None, "typed but not confirmed: the limits do not decide")
r = cd._limit_names.index("mass_tol_rel")
cd.t_lim.item(r, 2).setText("abc")
check(bool(_warn) and rs.limits[0].value == 0.01, "a value that is not a number is refused with a message")
_warn.clear()
sp.by.setText("Test Engineer")
sp.confirm_set()
check(rs.is_confirmed(), f"the user confirms the requirement set in the Space structures dialog ({sp.l_conf.text()[:50]})")
cd.update_state()
check("3" in cd.l_state.text() and ("confirmed" in cd.l_state.text() or "potvrzen" in cd.l_state.text()), f"state line: 3 limits usable ({cd.l_state.text()[:90]})")
check({x["name"]: x["state"] for x in MC.limits_overview(w.study)}["mass_tol_rel"] == "usable", "limits overview: usable")
cd.t_lim.item(r, 2).setText("0.02")
check(not rs.is_confirmed() and "NOT confirmed" in sp.l_conf.text(), "an edit of a limit withdraws the confirmation (also shown in the Space structures dialog)")
cd.t_lim.item(r, 2).setText("0.01")
sp.confirm_set()
check(rs.is_confirmed(), "confirmed again")

# --- run again: the confirmed limits decide the mass check
cd.b_run.click()
check(wait(lambda: cd.job is None), "the second run finished")
by = {r_: cd._results.get(r_) for r_ in MC.CHECK_IDS}
check(by["mass"].status in (MC.PASS, MC.WARN), f"mass with confirmed limits: {by['mass'].status} ({by['mass'].summary[:100]})")
check(by["mass"].limits.get("mass_tol_rel", {}).get("decided") is True, "the limit of the mass check decided")
check(by["mesh_convergence"].status == MC.SKIPPED and "INCOMPLETE" in cd.l_verdict.text(), "convergence not run: the verdict is INCOMPLETE, not a pass")
run = w.model_checks
check(run is not None and MC.run_is_current(run, w.study) and run.confirmed_by == "Test Engineer", "the kept run carries the confirmation of the set")
cd.t_lim.item(cd._limit_names.index("equilibrium_tol"), 2).setText("1e-6")
check("changed" in cd.l_stale.text() or "změnil" in cd.l_stale.text(), "an edit after the run marks the results as old")
rs.limits.pop()                                               # the equilibrium limit typed above is not needed any more

cd.tabs.setCurrentIndex(0)
app.processEvents()
cd.grab().save(str(OUT / "checks_dialog.png"))
cd.tabs.setCurrentIndex(2)
app.processEvents()
cd.grab().save(str(OUT / "checks_limits.png"))
cd.tabs.setCurrentIndex(0)

# --- cancel: the run stops at the next check, the kept run and the table of the last finished run stay
prev_run = w.model_checks
cd.b_run.click()
check(cd.job is not None, "a third run started")
cd.b_cancel.click()
check(wait(lambda: cd.job is None), "the run was cancelled and the thread ended")
check(w.model_checks is prev_run and cd.b_run.isEnabled() and not cd.b_cancel.isEnabled(), "cancelled: nothing is kept, Run is enabled again")
check("could not be run" not in cd.l_verdict.text() and "nešlo spustit" not in cd.l_verdict.text(), "a cancelled run is not shown as an error")

# --- the Word report carries the chapter
from fea_core.report import ReportOptions, build_report, session_from  # noqa: E402
sess = session_from(Study.from_dict(w.study.to_dict()), w.step_path, w.results, None, None, None, workdir=str(OUT / "rep"))
sess.model_checks = run
rep = build_report(sess, OUT / "report.docx", ReportOptions(images=False))
from docx import Document  # noqa: E402
doc = Document(rep["path"])
heads = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
want = "Model quality checks" if i18n.language() != "cs" else "Kontroly kvality modelu"
check(any(want in h for h in heads), f"the report has the chapter '{want}': {heads[-6:]}")
text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(c.text for t in doc.tables for row in t.rows for c in row.cells)
check("PASS" not in "\n".join(p.text for p in doc.paragraphs if "Verdict" in p.text or "Verdikt" in p.text), "the chapter verdict is not PASS for a partial run")
check("test clause" in text and "Test Engineer" in text, "the chapter prints the provenance of the limits and who confirmed them")

# --- texts
keys = ["Run all checks", "Model quality checks (ECSS)", "Tolerances", "Budgets", "Model quality checks..."]
tr = {k: i18n.tr(k) for k in keys}
if i18n.language() == "cs":
    check(all(tr[k] != k for k in keys), f"Czech texts of the dialog: {tr}")
else:
    check(all(tr[k] == k for k in keys), "English texts are the originals")

cd.close()
sp.close()
w.dirty = False
w.close()
print("FAILS:", fails)
sys.exit(1 if fails else 0)

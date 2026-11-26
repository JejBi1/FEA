"""Model quality checks dialog, regression of the review (counterexamples of the independent review):
a run that no longer belongs to the study is OUT OF DATE (table, verdict, report) while the dialog stays open; a new project clears the kept run; the confirmation of an agent is
amber and scope-aware like in the Space structures dialog; an edit of a limit goes through the Space dialog's own edit hook; the ticks of the launch interface survive a re-opened
dialog; Cancel neither logs a traceback nor kills the meshing of the main window; closing during a run cancels it, hides the dialog and ends it when the thread has ended.

python tests/gui/hidden_run.py tests/gui/gui_space_checks_review_smoke.py OUT_DIR        (FEA_LANG=cs | en)
The numbers of the limits are EXAMPLE values of the test - the program ships none.
"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_space_checks_review").resolve()
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
from fea_core.space.model import LimitEntry, SpaceSettings  # noqa: E402
from fea_core.space.requirements import confirm  # noqa: E402
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study  # noqa: E402
import fea_gui.main as gui_main  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
import fea_gui.workers as workers  # noqa: E402
from verification.cases import STEEL, box  # noqa: E402
from verification import space_cases as SC  # noqa: E402

fails = []
traces = []
workers.report_trace = lambda what: traces.append(what)                      # a traceback of a background job would be logged here
gui_main.msg = lambda parent, title, text, icon=None: _warn.append(str(text))
import fea_gui.main_defs as main_defs  # noqa: E402
main_defs.msg_via_main = lambda parent, title, text, *a, **k: _warn.append(str(text))


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text, flush=True)
    if not cond:
        fails.append(text)


def wait(pred, limit=900.0):
    t0 = time.time()
    while not pred() and time.time() - t0 < limit:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


def project(name, length):
    step = OUT / f"{name}.step"
    write_step([Part("Tyc", box(0, 0, 0, length, 10, 10))], step)
    parts = read_step(step)
    st = Study("Tyc", str(step), mesh=MeshSettings(size=4.0))                 # the SAME study name in both projects
    st.add_material(STEEL, default=True)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=length))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tah", [Force("konec", [5000.0, 0, 0])]))
    st.modal.enabled, st.modal.modes = True, 6
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    rs.limits = [LimitEntry(k, v, "", "test document", "test revision", "test clause") for k, v in (("mass_tol_rel", 0.01), ("cad_mass_tol_rel", 0.01), ("equilibrium_tol", 1e-3))]
    confirm(rs, "Test Engineer", "2026-01-01", via="mcp")                       # an agent's confirmation
    st.space = SpaceSettings(requirement_set=rs)
    proj = OUT / f"{name}.fea"
    save_project(proj, st, step)
    return proj


p1, p2 = project("one", 60), project("two", 80)
w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(p1)
app.processEvents()
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed")
w.set_stage(w.STAGE["Solve"])
w.run_solve()
check(wait(lambda: w.results is not None and w._thread is None), "solved")

from fea_gui.space_checks import open_checks_dialog  # noqa: E402
cd = open_checks_dialog(w)
cd.cb_conv.setChecked(False)
cd.e_mass.setText("0.0471")
rs = w.study.space.requirement_set

# --- the confirmation of an agent: amber, scope-aware
check("#b36b00" in cd.l_state.text() and ("MCP" in cd.l_state.text()), f"a confirmation made through MCP is amber and says so ({cd.l_state.text()[:80]})")
w.study.space.default_class = "composite"                                      # the classes are not those that were confirmed
cd.update_state()
check("#b36b00" in cd.l_state.text() and "MCP" not in cd.l_state.text() and "#1b7f3b" not in cd.l_state.text(), "scope changed: not confirmed (as in the Space dialog)")
w.study.space.default_class = "metallic"
cd.update_state()
check(w.study.space.requirement_confirmed() and "#b36b00" in cd.l_state.text(), "scope restored: confirmed again")
rs.confirmed_via = "gui"
cd.update_state()
check("#1b7f3b" in cd.l_state.text(), "confirmed through the program itself: green")
rs.confirmed_via = "mcp"

# --- a run, then the study is edited while the dialog stays open
cd.b_run.click()
check(wait(lambda: cd.job is None), "the checks finished")
check(all(cd._results.get(c) is not None for c in MC.CHECK_IDS) and cd.l_stale.text() == "", "seven results, current: no stale note")
check(cd.job is None and w.model_checks is not None and w.model_checks.confirmed_via == "mcp", "the kept run carries how the set was confirmed")
w.study.load_cases[0].loads[0].vector = [9000.0, 0, 0]
w.changed()
app.processEvents()
check(cd.l_stale.text() != "", "A: the study was edited: the dialog shows the stale note without a manual refresh")
check(all("(" in cd.t_checks.item(i, 3).text() and cd.t_checks.item(i, 3).text().split("(")[0].strip() in MC.STATUSES for i in range(7)),
      f"A: every row of the table says outdated: {[cd.t_checks.item(i, 3).text() for i in range(7)]}")
check("OUT OF DATE" in cd.l_verdict.text() or "ZASTARAL" in cd.l_verdict.text().upper() or "NEAKTU" in cd.l_verdict.text().upper(), f"A: the verdict says so: {cd.l_verdict.text()[:90]}")
check(not MC.run_is_current(w.model_checks, w.study), "A: the kept run is not current")
from fea_core.report import space as RS  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
D = _Doc()
RS.checks_section(D, SimpleNamespace(study=w.study, model_checks=w.model_checks))
text = "\n".join(p.text for p in D.doc.paragraphs)
cs = i18n.language() == "cs"
check(("ZASTARALÝ" if cs else "OUT OF DATE") in text and ("Verdikt modelu: PASS" if cs else "Verdict of the model: PASS") not in text
      and ("Verdikt modelu: NEPOTVRZENO" if cs else "Verdict of the model: UNCONFIRMED") not in text, "B: the report chapter says OUT OF DATE, not a verdict of the present model")

# --- another project with the same study name
w.dirty = False
w.open_file(p2)
app.processEvents()
check(w.model_checks is None, "C: a new project clears the kept run of the window")
cd = getattr(w, "_space_checks_dlg", None)
check(cd is not None and all(cd.t_checks.item(i, 3).text() == i18n.tr("not run") for i in range(7)) and cd.l_stale.text() == "", "C: the open dialog shows nothing of the old project")
check(not RS.has_checks(SimpleNamespace(model_checks=w.model_checks), w.study), "C: the report has no chapter for the other project")

# --- the ticks of the launch interface survive a re-opened dialog and a cancel
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed the second project")
cd = open_checks_dialog(w)
cd.cb_conv.setChecked(False)
names = [cd.l_iface.item(i).text() for i in range(cd.l_iface.count())]
from PySide6.QtCore import Qt  # noqa: E402
cd.l_iface.item(names.index("vetknuti")).setCheckState(Qt.CheckState.Checked)
cd.close()
app.processEvents()
check(getattr(w, "_space_checks_dlg", None) is None, "the dialog closed (no run)")
cd = open_checks_dialog(w)
check(cd.options().get("interface_selections") == ["vetknuti"], f"the tick of the interface survived the re-opened dialog: {cd.options().get('interface_selections')}")
cd.cb_conv.setChecked(False)

# --- Cancel while the main window meshes: no traceback, the mesh of the main window is not killed
cd.b_run.click()
check(cd.job is not None and cd.job.parent() is w, "the job is a child of the window (it outlives the dialog)")
check(wait(lambda: cd._done >= 1, 600), "the first check finished")
w.mesh = None
w.make_mesh()
check(w._thread is not None, "the main window started meshing while the checks run")
cd.b_cancel.click()
check(wait(lambda: cd.job is None, 300), "cancelled: the job ended")
check(wait(lambda: w._thread is None, 300) and w.mesh is not None, "E: the meshing of the main window was not killed by the Cancel of the checks")
check(not traces, f"a cancellation logs no traceback: {traces[:1]}")
check("could not be run" not in cd.l_verdict.text(), "a cancelled run is not an error")
check(cd.options().get("interface_selections") == ["vetknuti"], "the tick of the interface survived the cancel")

# --- a limit edited with the Space dialog open goes through its own edit hook
from fea_gui.extras import open_space_dialog  # noqa: E402
sp = open_space_dialog(w)
hooked = []
orig = sp._edited
sp._edited = lambda *a, **k: (hooked.append(1), orig(*a, **k))[1]
r = cd._limit_names.index("mass_tol_rel")
cd.t_lim.item(r, 2).setText("0.02")
rs = w.study.space.requirement_set
check(bool(hooked) and not rs.is_confirmed() and "#c0392b" in sp.l_conf.text(), "an edit of a tolerance reaches the Space dialog's edit hook (margins cleared, confirmation withdrawn and shown red)")
sp.close()

# --- closing during a run
if w.valid_mesh() is None:
    w.make_mesh()
    check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed again")
cd = open_checks_dialog(w)
cd.cb_conv.setChecked(False)
cd.b_run.click()
check(wait(lambda: cd._done >= 1, 600), "a run is going on")
job = cd.job
t0 = time.time()
cd.close()
check(time.time() - t0 < 5.0 and not cd.isVisible(), "F: close returns at once and hides the dialog")
check(job.parent() is w, "F: the thread does not belong to the dialog that was closed")
check(wait(lambda: not job.isRunning(), 600), "F: the cancelled thread ended")
check(wait(lambda: getattr(w, "_space_checks_dlg", None) is None, 30), "F: the dialog went away after its thread had ended")
check(len(w._aux_threads) == 0 and not w.busy(), "F: no thread left registered, the window is not busy")
check(not traces, f"F: no traceback was logged: {traces[:1]}")

# --- N6: a finding with part / file names is never read as HTML in the summary tooltip
from PySide6.QtGui import QTextDocument  # noqa: E402
cd = open_checks_dialog(w)
evil = "qzx '<img src=\"//host/x.png\">': url(//h/a) & O'Brien < B"
cd._results["mass"] = MC.CheckResult("mass", 1, MC.CHECK_TITLES["mass"], MC.WARN, "<b>bold</b> summary", {}, {}, [evil, "second"], [], 0.1)
cd._fill_rows()
tip = cd.t_checks.item(0, 4).toolTip()
doc = QTextDocument()
doc.setHtml(tip)
check("<img" not in tip and doc.toPlainText() == evil + "; second", f"N6: the tooltip shows the finding as written: {doc.toPlainText()[:80]!r}")
check(cd.t_checks.item(0, 4).text() == "<b>bold</b> summary", "N6: the summary cell itself is plain text")
cd._results.clear()
cd.close()
app.processEvents()

# --- texts
keys = ["Settings that differ from the defaults:"]
tr = {k: i18n.tr(k) for k in keys}
if i18n.language() == "cs":
    check(all(tr[k] != k for k in keys), f"Czech texts: {tr}")

w.dirty = False
w.close()
print("FAILS:", fails)
sys.exit(1 if fails else 0)

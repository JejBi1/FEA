"""Space structures (ECSS) dialog in the real main window: Tools menu / results button, requirement set with arbitrary TEST factors,
confirmation, allowables, margins of a solved bar, singularity justification, marker in 3D, unsaved-changes flag, project round trip.

python tests/gui/hidden_run.py tests/gui/gui_space_smoke.py OUT_DIR
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_space").resolve()
OUT.mkdir(parents=True, exist_ok=True)
_warn = []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication(sys.argv)
from fea_core.geometry import Part, read_step, write_step  # noqa: E402
from fea_core.project import save_project  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study  # noqa: E402
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


step = OUT / "tyc.step"
write_step([Part("Tyc", box(0, 0, 0, 60, 10, 10))], step)
parts = read_step(step)
st = Study("Tyc", str(step), mesh=MeshSettings(size=4.0))
st.add_material(STEEL, default=True)
st.add_selection("vetknuti", find_faces(parts, x=0))
st.add_selection("konec", find_faces(parts, x=60))
st.constraints.append(Fixed("vetknuti"))
st.load_cases.append(LoadCase("Tah", [Force("konec", [5000.0, 0, 0])]))       # 5000 N / 100 mm2 = 50 MPa
proj = OUT / "tyc.fea"
save_project(proj, st, step)

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()
check(w.study is not None and w.study.name == "Tyc", "project opened")

from fea_gui.extras import open_space_dialog  # noqa: E402
check(hasattr(w, "btn_space"), "results panel has the margins button")
check(w.study.space is None, "a study without space settings stays untouched")
dlg = open_space_dialog(w)
check(dlg is not None and dlg is open_space_dialog(w), "one dialog per window")
dlg.new_template()
check(dlg.t_rules.rowCount() == 5 and w.study.space is not None and w.dirty, "template installed, project marked as changed")

# --- arbitrary TEST numbers, entered through the dialog widgets
dlg.c_approach.setCurrentIndex(dlg.c_approach.findData("analysis_only"))
rs = w.study.space.requirement_set
test_values = {"FOSY": 1.0625, "FOSU": 1.34375, "KQ": 1.125}
for r in range(dlg.t_factors.rowCount()):
    key, cls = dlg.t_factors.item(r, 0).text(), dlg.t_factors.item(r, 1).text()
    if cls == "metallic" and key in test_values:
        dlg.t_factors.item(r, 3).setText(str(test_values[key]))
        dlg.t_factors.item(r, 5).setText("test revision")
        dlg.t_factors.item(r, 6).setText("test clause")
check(rs.entry("FOSY", "metallic").value == 1.0625, "factor value typed into the table reaches the study")
dlg.t_factors.item(0, 3).setText("-3")
check(bool(_warn) and rs.factors[0].value is None, "a negative factor is refused with a message")
_warn.clear()
for r, rule in enumerate(rs.rules):
    if rule.structure_class == "metallic":
        dlg.t_rules.item(r, 4).setText("FOSY, KQ" if rule.check == "yield" else "FOSU")
        dlg.t_rules.cellWidget(r, 3).setCurrentIndex(dlg.t_rules.cellWidget(r, 3).findData("von_mises"))
        dlg.t_rules.cellWidget(r, 5).setCurrentIndex(dlg.t_rules.cellWidget(r, 5).findData("same_as_mechanical"))
        dlg.t_rules.item(r, 7).setText("test clause")
        dlg.t_rules.item(r, 8).setText("test revision")
dlg.confirm_set()
check(bool(_warn) and not rs.is_confirmed(), "confirmation without a name is refused")
_warn.clear()
dlg.by.setText("Test Engineer")
dlg.confirm_set()
check(rs.is_confirmed() and "Confirmed" in dlg.l_conf.text(), f"requirement set confirmed ({dlg.l_conf.text()[:60]})")
dlg.t_factors.item(1, 7).setText("edited note")
check(not rs.is_confirmed() and "NOT confirmed" in dlg.l_conf.text(), "an edit after the confirmation withdraws it")
dlg.confirm_set()
check(rs.is_confirmed(), "confirmed again")

# --- allowables (test numbers)
t = dlg.t_allow
t.item(0, 1).setText("B")
t.item(0, 2).setText("276")
t.item(0, 3).setText("310")
t.item(0, 7).setText("test data sheet")
t.setCurrentCell(0, 0)
dlg.confirm_allowable()
mat = w.study.materials[w.study.default_material]
from fea_core.space.requirements import allowable_of  # noqa: E402
check(allowable_of(mat).is_verified(), "allowables entered and confirmed through the table")

# --- solve and margins
before = w.study.to_dict()["space"]
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed")
w.set_stage(w.STAGE["Solve"])
w.run_solve()
check(wait(lambda: w.results is not None and w._thread is None), "solved")
dlg.refresh()
dlg.compute()
s = dlg.summary
check(s is not None and s.rows, "margins computed")
if s is not None and s.rows:
    y = next(r for r in s.rows if r.check == "yield" and r.mos is not None)
    ref = 276.0 / (1.0625 * 1.125 * 50.0) - 1.0
    check(abs(y.mos - ref) / ref < 0.08, f"yield MoS {y.mos:.3f} close to the hand value {ref:.3f} of the uniform tension (outside the clamp)")
    check(dlg.t_mos.rowCount() > 0 and dlg.l_verdict.text().strip() != "", f"verdict shown: {dlg.l_verdict.text()[:60]}")
    dlg.t_mos.selectRow(0)
    app.processEvents()
    check(not _warn, "clicking a margin row marks the place without an error")
    if s.singularities:
        check(dlg.t_sing.rowCount() == len(s.singularities), "singularity table filled")
        dlg.t_sing.item(0, 6).setText("accepted: rigid clamp of the test model")
        check(s.singularities[0].justified and w.study.space.justifications, "justification note stored")
check(w.study.to_dict()["space"] != before, "the study carries the justification / part class edits")

# --- the verdict of the margins tab goes through the 3-axis envelopes of a CURRENT result (review fix C3)
import numpy as np  # noqa: E402
from fea_core.space import environments as EV  # noqa: E402
from fea_core.space import loads as LD  # noqa: E402
lv = EV.EnvLevel("qualification")
lv.sine = EV.EnvTable("sine", [[20.0, 5.0], [400.0, 5.0]], document="test", revision="A", clause="c")
es = EV.EnvironmentSet(levels=[lv], launch_axis="z", damping_zeta=0.03, damping_source="test")
EV.confirm_environment(es, "Test Engineer", "2026-01-01")
EV.store_environments(w.study.space, es)
n = len(w.results.mesh.nodes)
fld = LD.AxisField(np.full(n, 1.0e5, np.float32), np.zeros(n, np.float32), None)
per = {"sine": {0: fld, 1: fld, 2: fld}}
r3 = LD.ThreeAxisResult("qualification", ["sine"], [0, 1, 2], per, {"sine": LD.envelope_of(per["sine"], "sine")}, {}, np.array([100.0]), np.zeros((1, 3)), None, True, True,
                        env_digest=es.content_digest(), run_digest=LD.compute_run_digest(w.study, w.results.mesh), level_kinds=["sine"],
                        damping={"kind": "modal", "zeta": 0.03, "source": "test", "from_environment": True})
LD.remember(w, r3)
dlg.compute()
check(dlg.summary is not None and dlg.summary.verdict != "PASS" and any(r.case.startswith("3-axis") for r in dlg.summary.rows),
      f"the margins tab includes the envelope of the kept 3-axis result: {dlg.summary.verdict if dlg.summary else None}")
LD.remember(w, None)                                                          # drops the kept result (also the copy that is attached to the results)
dlg.compute()
check(dlg.summary is not None and not any(r.case.startswith("3-axis") for r in dlg.summary.rows), "without a kept result the margins are those of the load states only")

# --- project round trip keeps the settings and the confirmation
saved = OUT / "tyc_space.fea"
check(w._save(saved), "project saved")
from fea_core.project import load_project  # noqa: E402
st2, _step, _res = load_project(saved, OUT / "reload")
check(st2.space is not None and st2.space.requirement_set.is_confirmed(), "requirement set and confirmation survive save / load")
check(allowable_of(st2.materials[st2.default_material]).is_verified(), "allowables survive save / load")

dlg.close()
w.close()
print("FAILS:", fails)
sys.exit(1 if fails else 0)

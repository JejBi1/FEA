"""Space structures: the margins / mass card computed through MainWindow.run_busy (a worker thread, a nested event loop). The job runs off the GUI thread without touching the window,
the result of a job is thrown away when a queued event (an open, a finishing solve) replaced the study / the results / the mesh meanwhile, a nested call is refused with a message.

python tests/gui/hidden_run.py tests/gui/gui_space_busy_smoke.py OUT_DIR
"""
import copy
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_space_busy").resolve()
OUT.mkdir(parents=True, exist_ok=True)
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
_warn = []
gui_main.msg = lambda parent, title, text, icon=None: _warn.append(str(text))


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
st.load_cases.append(LoadCase("Tah", [Force("konec", [5000.0, 0, 0])]))
proj = OUT / "tyc.fea"
save_project(proj, st, step)
w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()
from fea_gui.extras import open_space_dialog  # noqa: E402

dlg = open_space_dialog(w)
dlg.new_template()
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed")
w.set_stage(w.STAGE["Solve"])
w.run_solve()
check(wait(lambda: w.results is not None and w._thread is None), "solved")
dlg.refresh()
dwarn = []
dlg._warn = lambda text: dwarn.append(str(text))
ctl = dlg.loads
import fea_core.space.mass_props as MP  # noqa: E402
from fea_core.space import loads as LD  # noqa: E402
from fea_core.space import margins as MG  # noqa: E402

# 1) the margins tab: the job runs in a worker thread, reads the kept result from a snapshot (not from the window), the summary equals a direct computation
seen = {}
real_sign = LD.sign_off_margins


def spy(study, results, owner, *a, **k):
    seen["worker"] = threading.current_thread() is not threading.main_thread()
    seen["busy"] = w.busy()
    seen["snapshot"] = owner is not w and isinstance(owner, LD.KeptSnapshot)
    return real_sign(study, results, owner, *a, **k)


LD.sign_off_margins = spy
dlg.compute()
check(dlg.summary is not None and dlg.summary.rows and not dwarn, f"the margins through run_busy give a summary {dwarn[:1]}")
check(seen == {"worker": True, "busy": True, "snapshot": True}, f"the job ran in a worker thread on a snapshot, not on the window: {seen}")
ref = MG.compute_margins(w.study, w.results)
check([r.to_dict() for r in ref.rows] == [r.to_dict() for r in dlg.summary.rows] and ref.verdict == dlg.summary.verdict, "the summary equals a direct computation")
LD.sign_off_margins = real_sign

# 2) a queued event replaces the results / the study during the job -> the outdated summary is discarded
dlg.summary = None
dlg.show_summary()
res0, st0 = w.results, w.study


def slow_sign(study, results, owner, *a, **k):
    time.sleep(0.6)
    return real_sign(study, results, owner, *a, **k)


LD.sign_off_margins = slow_sign
QTimer.singleShot(150, lambda: setattr(w, "results", copy.copy(res0)))       # a finishing solve replaces the results while the job runs
dwarn.clear()
dlg.compute()
check(dlg.summary is None and dwarn and "changed while" in dwarn[-1], f"results replaced during the job: the margins are discarded ({dwarn[-1:]})")
w.results = res0
QTimer.singleShot(150, lambda: setattr(w, "study", copy.copy(st0)))         # an open replaces the study
dwarn.clear()
dlg.compute()
check(dlg.summary is None and dwarn and "changed while" in dwarn[-1], f"study replaced during the job: the margins are discarded ({dwarn[-1:]})")
w.study = st0

# 3) a nested call (a queued click during the job) is refused with a message, the outer one still delivers
inner = {}


def nested():
    inner["busy"] = w._busy_task is not None
    dlg.compute()
    inner["warn"] = list(dwarn)


dwarn.clear()
QTimer.singleShot(150, nested)
dlg.compute()
check(inner.get("busy") is True and inner.get("warn") and "Another operation" in inner["warn"][-1], f"a nested compute is refused with a message {inner}")
check(dlg.summary is not None and dlg.summary.rows, "the outer compute still delivers its summary")
LD.sign_off_margins = real_sign

# 4) the launch-loads tab: margins through run_busy (the GUI smoke of LaunchLoads.margins), and its discard on a changed result
dwarn.clear()
ctl.margins()
check(not dwarn and "MoS" in ctl.l_margins.text(), f"LaunchLoads.margins through run_busy: {ctl.l_margins.text()[:60]} {dwarn[:1]}")
text_before = ctl.l_margins.text()
real_mwe = LD.margins_with_envelopes


def slow_mwe(*a, **k):
    time.sleep(0.6)
    return real_mwe(*a, **k)


LD.margins_with_envelopes = slow_mwe
ctl.l_margins.setText("")
QTimer.singleShot(150, lambda: setattr(w, "results", copy.copy(res0)))
dwarn.clear()
ctl.margins()
check(ctl.l_margins.text() == "" and dwarn and "changed while" in dwarn[-1], f"LaunchLoads.margins: results replaced during the job -> discarded ({dwarn[-1:]})")
w.results = res0
LD.margins_with_envelopes = real_mwe
ctl.margins()
check(ctl.l_margins.text() == text_before, "the same margins again after the discard")

# 5) the mass card: through run_busy, a study replaced during the job, a nested call, a ValueError (no traceback in the log)
dwarn.clear()
ctl.mass_card()
check("kg" in ctl.l_mass.text() and not dwarn, f"the mass card through run_busy: {ctl.l_mass.text()[:40]}")
real_mp = MP.mass_properties


def slow_mp(*a, **k):
    time.sleep(0.6)
    return real_mp(*a, **k)


MP.mass_properties = slow_mp
ctl.l_mass.setText("")
QTimer.singleShot(150, lambda: setattr(w, "study", copy.copy(st0)))
dwarn.clear()
ctl.mass_card()
check(ctl.l_mass.text() == "" and dwarn and "changed while" in dwarn[-1], f"the mass card: study replaced during the job -> discarded ({dwarn[-1:]})")
w.study = st0
dwarn.clear()
QTimer.singleShot(150, lambda: (inner.update(busy2=w._busy_task is not None), ctl.mass_card(), inner.update(warn2=list(dwarn))))
ctl.mass_card()
check(inner.get("busy2") is True and inner.get("warn2") and "Another operation" in inner["warn2"][-1], f"a nested mass card is refused with a message {inner.get('warn2')}")
check("kg" in ctl.l_mass.text(), "the outer mass card still delivers")


def no_density(*a, **k):
    raise ValueError("review: no density")


MP.mass_properties = no_density
n_log = len(w.log.toPlainText())
dwarn.clear()
ctl.mass_card()
check(dwarn and "no density" in dwarn[-1], f"an expected ValueError of the mass card is shown: {dwarn[-1:]}")
check("Traceback" not in w.log.toPlainText()[n_log:], "the expected ValueError leaves no traceback in the log")
MP.mass_properties = real_mp
check(w._busy_task is None, "no busy task is left")
print("FAILS:", fails)
w.dirty = False
w.close()
sys.exit(1 if fails else 0)

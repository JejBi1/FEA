"""Space structures, stage S4 (interface forces, notching, bolts, PCB, rails, limits) in the REAL main window, in English or Czech (FEA_LANG=cs): the tab is built, the pages run
with a real mesh and the modal basis of CalculiX, a limit edit and a reload clear the verdicts, every tooltip of the tab is at most 80 characters, a typed number never gives PASS.

python tests/gui/hidden_run.py tests/gui/gui_space_s4_smoke.py OUT_DIR           (English)
FEA_LANG=cs python tests/gui/hidden_run.py tests/gui/gui_space_s4_smoke.py OUT_DIR   (Czech)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_space_s4").resolve()
OUT.mkdir(parents=True, exist_ok=True)
_warn = []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication(sys.argv)
from fea_core.geometry import Part, read_step, write_step  # noqa: E402
from fea_core.i18n import language  # noqa: E402
from fea_core.project import save_project  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.study import Fixed, LoadCase, MeshSettings, Study  # noqa: E402
import fea_gui.main as gui_main  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
from verification.cases import STEEL, box  # noqa: E402

fails = []
gui_main.msg = lambda parent, title, text, icon=None: _warn.append(str(text))
LANG = language()
print("language:", LANG)


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


step = OUT / "bar.step"
write_step([Part("Bar", box(0, 0, 0, 80, 10, 10))], step)
parts = read_step(step)
st = Study("Bar", str(step), mesh=MeshSettings(size=4.0))
st.add_material(STEEL, default=True)
st.add_selection("root", find_faces(parts, x=0))
st.constraints.append(Fixed("root"))
st.load_cases.append(LoadCase("F", []))
st.dynamics.enabled = True
st.dynamics.modes = 8
st.dynamics.damping.zeta = 0.02
st.solver.modal_engine = "ccx"
proj = OUT / "bar.fea"
save_project(proj, st, step)

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()
check(w.study is not None and w.study.name == "Bar", "project opened")
w.set_stage(w.STAGE["Mesh"])
w.make_mesh()
check(wait(lambda: w._thread is None and w.valid_mesh() is not None), "meshed")

from fea_gui.extras import open_space_dialog  # noqa: E402
dlg = open_space_dialog(w)
s4 = dlg.s4
check(s4 is not None and dlg.tabs.indexOf(s4) >= 0, "the S4 tab exists in the space dialog")
names = [s4.tabs.tabText(i) for i in range(s4.tabs.count())]
print("pages:", names)
check(len(names) == 6, "six pages")
if LANG == "cs":
    check(names[0] != "Interface forces" and names[4] != "CubeSat rails", "the page names are translated")

# --- every tooltip of the tab: at most 80 characters in the active language
long_tips = [(type(x).__name__, x.toolTip()) for x in s4.findChildren(QWidget) if len(x.toolTip()) > 80]
check(not long_tips, f"no tooltip longer than 80 characters ({long_tips[:2]})")

# --- the pages with a real modal basis
s4.c_if_dir.setCurrentIndex(s4.c_if_dir.findData("y"))
s4.e_if_sine.setPlainText("10 1\n3000 1")
s4.e_if_asd.setPlainText("20 0.01\n2000 0.01")
s4.run_interface()
check(wait(lambda: s4._thread is None), "interface forces computed")
check(s4.page_if.t_main.rowCount() == 1 and s4.t_modal.rowCount() >= 1, "interface table and modal table filled")
check(any("interface" in k for k in __import__("fea_core.space.s4_store", fromlist=["x"]).get(w.study)), "the result is stored in the study")
peak = float(__import__("fea_core.space.s4_store", fromlist=["x"]).get(w.study)["interface"]["directions"]["y"]["sine"]["F_peak_max_N"])
check(peak > 0 and not _warn, f"sine peak force {peak:.3f} N, no message box ({_warn[:1]})")

s4.e_n_profile.setPlainText("10 1\n3000 1")
s4.e_n_limit.setText(str(0.5 * peak))
s4.e_n_j.setText("1")
s4.e_n_w.setText("0.1")
s4.e_n_floor.setText("0.1")
s4.c_n_dir.setCurrentIndex(s4.c_n_dir.findData("y"))
s4.run_notch()
check(wait(lambda: s4._thread is None), "notching computed")
check(s4.page_notch.t_main.rowCount() >= 1 and s4.b_n_export.isEnabled(), "notch table filled, export enabled")
check(s4.page_notch.l_verdict.text().strip() != "" and "PASS" not in s4.page_notch.l_verdict.text().split(" - ")[0], f"a typed limit gives no PASS ({s4.page_notch.l_verdict.text()[:50]})")

for k, v in (("d_mm", "6"), ("pitch_mm", "1"), ("A_s_mm2", "20.1"), ("k_b", "350000"), ("k_c", "400000"), ("F_nom", "6000"), ("F_ax", "500"), ("F_sh", "100")):
    s4.f_bolt[k].setText(v)
s4.run_bolt()
check(s4.page_bolts.t_main.rowCount() >= 3 and "PASS" not in s4.page_bolts.l_verdict.text().split(" - ")[0], "bolt page: margins listed, no PASS for typed numbers")

for k, v in (("name", "U1"), ("B_mm", "127"), ("h_mm", "1.575"), ("L_mm", "25.4"), ("r", "1"), ("z_rms_mm", "0.05"), ("f_n_Hz", "100"), ("duration_h", "1")):
    s4.f_pcb[k].setText(v)
s4.run_pcb()
check(wait(lambda: s4._thread is None) and s4.page_pcb.t_main.rowCount() == 9, "PCB page computed")

# --- a limit edit and a reload clear the verdicts of the pages
dlg.new_template()
s4.add_limit_rows()
check(s4.t_limits.rowCount() > 20, "the limits page lists the rows of the requirement set")
s4.run_pcb()
wait(lambda: s4._thread is None)
check(s4.page_pcb.l_verdict.text().strip() != "", "the PCB page shows a verdict again")
s4.t_limits.item(0, 3).setText("1")
check(s4.page_pcb.l_verdict.text() == "" and s4.page_pcb.t_main.rowCount() == 0, "a limit edit clears the verdict and the table of the pages")
s4.run_bolt()
check(s4.page_bolts.l_verdict.text().strip() != "", "the bolt page shows a verdict")
dlg.refresh()
check(s4.page_bolts.l_verdict.text() == "" and s4.page_bolts.t_main.rowCount() == 0, "a reload of the dialog clears the pages")

dlg.close()
w.close()
print("FAILS:", fails)
sys.exit(1 if fails else 0)

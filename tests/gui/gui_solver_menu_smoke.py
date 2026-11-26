"""The equation-solver menu of the Solve step after the removal of Intel MKL: Automatic / PaStiX / SPOOLES, the stored word of "automatic" is the data
of the first item, an old study that stores "PARDISO" selects Automatic, a choice is stored and restored, nothing in the menu names PARDISO.

python tests/gui/hidden_run.py tests/gui/gui_solver_menu_smoke.py OUT_DIR [project.fea]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

PROJECT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("examples/example1_bracket.fea")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

app = QApplication(sys.argv)
from fea_core import config  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402

fails = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        fails.append(text)


w = MainWindow()
w.resize(1600, 950)
w.show()
w.open_file(PROJECT)
app.processEvents()
c = w.solver_combo
data = [c.itemData(i) for i in range(c.count())]
check(data == [config.AUTO_SOLVER, "PASTIX", "SPOOLES"], f"menu: {data}")
check(not any("PARDISO" in c.itemText(i).upper() or "MKL" in c.itemText(i).upper() for i in range(c.count())), "no item names PARDISO / MKL")
check(c.itemText(0) == "Automatic", f"first item = Automatic ({c.itemText(0)})")
check(c.currentData() == config.AUTO_SOLVER, "an old study with the stored word PARDISO shows Automatic")
w.study.solver.solver = "SPOOLES"
w.refresh_all()
app.processEvents()
check(c.currentData() == "SPOOLES", "a study with SPOOLES selects it")
c.setCurrentIndex(c.findData("PASTIX") if config.ccx_solvers()["PASTIX"] else c.findData(config.AUTO_SOLVER))
app.processEvents()
want = "PASTIX" if config.ccx_solvers()["PASTIX"] else config.AUTO_SOLVER
check(w.study.solver.solver == want, f"the choice is stored in the study ({w.study.solver.solver})")
c.setCurrentIndex(0)
app.processEvents()
check(w.study.solver.solver == config.AUTO_SOLVER, "Automatic is stored as the word of the earlier versions")
if not config.ccx_solvers()["PASTIX"]:
    check(not c.model().item(1).isEnabled(), "PaStiX greyed out where this ccx does not have it")

# --- a stored value that the menu does not offer is shown truthfully (not the choice of the previous study) and is not changed by looking at it
w.study.solver.solver = "SPOOLES"
w.refresh_all()
app.processEvents()
w.study.solver.solver = "ITERATIVE SCALING"
w.refresh_all()
app.processEvents()
check(c.currentData() == "ITERATIVE SCALING", f"stored ITERATIVE SCALING: the menu shows it ({c.currentText()!r}), not SPOOLES of the previous state")
check(w.study.solver.solver == "ITERATIVE SCALING", "… and the study keeps the stored value")
check([c.itemData(i) for i in range(c.count())][:3] == [config.AUTO_SOLVER, "PASTIX", "SPOOLES"], "… the three offered items stay in front")
c.setFocus()
c.showPopup()
app.processEvents()
c.hidePopup()
app.processEvents()
check(w.study.solver.solver == "ITERATIVE SCALING", "opening and closing the menu does not change the study")
c.setCurrentIndex(c.findData("SPOOLES"))                      # the user chooses an offered solver: the temporary item goes
app.processEvents()
check(w.study.solver.solver == "SPOOLES" and not any(c.itemText(i).endswith(" (stored in the study)") for i in range(c.count())),
      "choosing an offered solver replaces the stored value and removes the temporary item")
w.study.solver.solver = "NOSUCHSOLVER"                         # a word that is no solver: CalculiX takes Automatic (config.solver_name)
w.refresh_all()
app.processEvents()
check(c.currentData() == config.AUTO_SOLVER, "an unknown stored word shows Automatic (what the calculation does)")

# --- the Properties card of the unsolved study: the stored word of "automatic" is not shown as PARDISO
from PySide6.QtWidgets import QFormLayout, QLabel  # noqa: E402
from fea_core.i18n import tr  # noqa: E402


def rows(card) -> dict:
    out = {}
    for r in range(card.form.rowCount()):
        lab = card.form.itemAt(r, QFormLayout.LabelRole)
        val = card.form.itemAt(r, QFormLayout.FieldRole)
        if lab is not None and val is not None and isinstance(lab.widget(), QLabel) and isinstance(val.widget(), QLabel):
            out[lab.widget().text()] = val.widget().text()
    return out


w.study.solver.solver = config.AUTO_SOLVER
w.results = None
w.set_stage(w.STAGE["Solve"])
w.props_card.refresh()
app.processEvents()
system = rows(w.props_card).get(tr("System"), "")
check(system == tr("Automatic") and "PARDISO" not in system.upper(), f"Properties, unsolved default study: System = {system!r}")
w.study.solver.solver = "SPOOLES"
w.props_card.refresh()
app.processEvents()
check(rows(w.props_card).get(tr("System"), "") == "SPOOLES", "… an explicit choice is shown as it is")
w.study.solver.solver = "ITERATIVE CHOLESKY"
w.props_card.refresh()
app.processEvents()
check(rows(w.props_card).get(tr("System"), "") == "ITERATIVE CHOLESKY", "… also a stored value that the menu does not offer")
w.dirty = False
w.close()
print("\nSELHALO:" if fails else "\nVŠE V POŘÁDKU", len(fails))
sys.exit(1 if fails else 0)

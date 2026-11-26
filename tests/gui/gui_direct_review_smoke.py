"""Review fixes of the direct solver in the GUI: the stress error estimate (CalculiX only) is greyed out with the reason in a result
without it, the Properties card of the Solve step labels a stored MKL PARDISO result by what ran (not "direct solver rslab").

python tests/gui/hidden_run.py tests/gui/gui_direct_review_smoke.py OUT_DIR [project.fea]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_direct_review")
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("examples/example1_bracket.fea")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

app = QApplication(sys.argv)
from fea_core.i18n import tr  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402

fails = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        fails.append(text)


def rows(card) -> dict:
    out = {}
    form = card.form
    for r in range(form.rowCount()):
        lab = form.itemAt(r, QFormLayout.LabelRole)
        val = form.itemAt(r, QFormLayout.FieldRole)
        if lab is not None and val is not None and isinstance(lab.widget(), QLabel) and isinstance(val.widget(), QLabel):
            out[lab.widget().text()] = val.widget().text()
    return out


w = MainWindow()
w.resize(1600, 950)
w.show()
w.open_file(PROJECT)
app.processEvents()
res = w.results
check(res is not None and len(res.steps) > 0, "projekt s výsledky")

# --- the stress error estimate: with the field the item is enabled, without it greyed out with the reason
w.set_stage(w.STAGE["Results"])
app.processEvents()
names = [w.r_qty.itemText(i).split(" [")[0] for i in range(w.r_qty.count())]
k = names.index("Stress error estimate")
item = w.r_qty.model().item(k)
has_field = all("stress_error" in s.fields for s in res.steps)
print("example has the ccx estimate:", has_field)
for s in res.steps:                                  # the same result as from the direct solver
    s.fields.pop("stress_error", None)
w._sync_result_controls()
app.processEvents()
check(not item.isEnabled(), "výsledek bez odhadu chyby: veličina je zašedlá")
check("CalculiX" in item.toolTip(), f"… s důvodem v tipu ({item.toolTip()[:60]}…)")
w.r_qty.setCurrentIndex(0)
other = w.r_qty.model().item(0)
check(other.isEnabled(), "ostatní veličiny zůstávají dostupné")
# a gravity case that CalculiX results carry as a combination of unit states says so; another state of the result gives the usual reason
res.info["backend"] = "ccx"
qlist = w.result_qlist(("step", 0))
for cases, expect_unit in (([res.steps[0].name], True), (["not a case of this result"], False)):
    res.info["gravity_superposition"] = {"cases": cases, "family": "ccx"}
    w._mark_unavailable_quantities(("step", 0), qlist)
    tip = w.r_qty.model().item(names.index("Stress error estimate")).toolTip()
    unit_tip = tr("Not available for gravity cases that are combinations of unit states: the stress error estimate is not linear in the load.")
    check((tip == unit_tip) == expect_unit, f"odhad chyby: tip pro {'složený' if expect_unit else 'jiný'} stav ({tip[:50]}…)")
res.info.pop("gravity_superposition", None)
import numpy as np  # noqa: E402
for s in res.steps:
    s.fields["stress_error"] = np.full(len(s.U), 3.0)
w._sync_result_controls()
check(item.isEnabled() and item.toolTip() == "", "výsledek s odhadem: veličina je dostupná")

# --- the Properties card (Solve step): a result computed by MKL PARDISO is not called the direct solver of rslab
res.info["backend"] = "ngs_pardiso"
res.info.pop("solver", None)
w.set_stage(w.STAGE["Solve"])
w.props_card.refresh()
app.processEvents()
sw = rows(w.props_card).get(tr("Software"), "")
check("PARDISO" in sw and "dřívější" in sw or "earlier" in sw, f"karta Vlastnosti: starý výsledek = „{sw}“")
check("přímý řešič" not in sw and "direct solver" not in sw, "… a není označen jako přímý řešič rslab")
res.info["backend"] = "ngs_direct"
w.props_card.refresh()
app.processEvents()
sw = rows(w.props_card).get(tr("Software"), "")
check("rslab" in sw and "NGSolve" not in sw and "PARDISO" not in sw, f"nový výsledek ngs_direct: „{sw}“")

w.dirty = False
w.close()
print("\nSELHALO:" if fails else "\nVŠE V POŘÁDKU", len(fails))
sys.exit(1 if fails else 0)

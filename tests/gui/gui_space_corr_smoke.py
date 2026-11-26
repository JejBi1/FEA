"""Test correlation dashboard (F14-S5) in the real main window: Tools menu entry, import of a CSV test file on a worker thread, correlation against synthetic natural modes with a
confirmed requirement set with arbitrary TEST limits (PASS), the stale-result check after a limit edit, the damping transfer through MainWindow.changed() (the results are discarded)
and the project round trip of the stored result with its provenance.

python tests/gui/hidden_run.py tests/gui/gui_space_corr_smoke.py OUT_DIR
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".scratch/gui_space_corr").resolve()
OUT.mkdir(parents=True, exist_ok=True)
_warn: list[str] = []
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)

app = QApplication(sys.argv)
from fea_core.geometry import Part, read_step, write_step  # noqa: E402
from fea_core.project import load_project, save_project  # noqa: E402
from fea_core.results import Results  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.space import corr_run as CR  # noqa: E402
from fea_core.space import correlation as C  # noqa: E402
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study  # noqa: E402
import fea_gui.main as gui_main  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
from test_space_correlation import EXAMPLE, FIXTURE_MODEL_KEY, SENSORS, _box_mesh, _limits_set, _modes, _tam_of_gram, _test_data, _unit_mass_gram  # noqa: E402
_real_build = CR.build_orth_mass          # the synthetic bar has a tet4 mesh and no densities: its SEREP mass is the Gram matrix of the modes under a unit nodal mass (as in test_space_correlation)
CR.build_orth_mass = lambda study, mesh, modeset, opt, tam=None: (C.OrthMass("serep", study=study, gram=_unit_mass_gram(modeset), gram_modes=list(range(len(modeset.values))))
                                                                    if opt.orth_mass == "serep" else _real_build(study, mesh, modeset, opt, tam))
_real_orth = C.orthogonality                # the smoke test exercises the pipeline, not the rule that SEREP rows are indicative (test_space_corr_round4.py): the SEREP mass becomes a TAM-like
_kept: list = []                            # mass (a regularised SEREP matrix, as in test_space_correlation), the only route that can PASS


def _lifted_orth(test_phi, fe_phi, dofs, mass, modeset=None, fe_idx=None, test_numbers=None, axes_hold=""):
    if mass is not None and mass.kind == "serep" and mass.gram is not None:
        probe = _real_orth(test_phi, fe_phi, dofs, mass, modeset, fe_idx, test_numbers, axes_hold)
        if not probe.computed:
            return probe
        mass = _tam_of_gram(mass, test_phi, fe_phi, dofs, modeset, fe_idx)
        if mass.study is not None:
            _kept.append(mass.tam)
            CR.remember_tam(mass.study, mass.tam)
    return _real_orth(test_phi, fe_phi, dofs, mass, modeset, fe_idx, test_numbers, axes_hold)


C.orthogonality = _lifted_orth
_real_fp = C.model_fingerprint
C.model_fingerprint = lambda study, mesh: _real_fp(study, mesh) or FIXTURE_MODEL_KEY       # the synthetic tet4 bar cannot be fingerprinted: a fixed key stands for its state
from verification.cases import STEEL, box  # noqa: E402

fails: list[str] = []
gui_main.msg = lambda parent, title, text, icon=None: _warn.append(str(text))          # a modal box would block the test


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


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
check(w.study is not None and w.study.name == "Tyc", "project opened")

# synthetic natural modes of a bar (the analytic fields of the test module) and a confirmed set with arbitrary TEST limits
mesh = _box_mesh()
fe_modes = _modes(mesh)
w.study.space = _limits_set(EXAMPLE)
w.results = Results(mesh=mesh, steps=[], modes=[fe_modes])
csv = OUT / "modes.csv"
csv.write_text(C.write_modes_csv(_test_data((mesh, fe_modes)).modes), encoding="utf-8")
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(csv), ""))

from fea_gui.extras import open_correlation_dialog  # noqa: E402
dlg = open_correlation_dialog(w)
check(dlg is not None and dlg is open_correlation_dialog(w), "Tools - Test correlation: one dialog per window")
dlg.import_file()
check(dlg._thread is not None and dlg.test is None, "the import runs on a worker thread (nothing taken over yet)")
dlg.wait_for_import()
check(dlg._thread is None and dlg.test is not None and len(dlg.test.modes) == 6 and not _warn, f"test file read ({dlg.l_data.text()[:70]})")
t = dlg.t_sensors
by_node = {s.node: s for s in SENSORS}
for r in range(t.rowCount()):
    s = by_node[t.item(r, 0).text()]
    for k in range(3):
        t.item(r, 1 + k).setText(str(s.point[k]))
dlg.run()
dlg.wait_for_import()
check(dlg.result is not None and dlg.result.verdict == "PASS" and dlg.l_verdict.text().startswith("Verdict: PASS") and not _warn, f"PASS with confirmed TEST limits ({dlg.l_verdict.text()[:40]})")
check(w.dirty and CR.state_of(w.study)["result"]["provenance"]["limits_fingerprint"], "the result is stored with its provenance and the project is marked as changed")
check(w.results is not None, "without a damping transfer the results stay valid")

# a saved and reopened project: the stored result is still current
save_project(proj, w.study, step)
bst = load_project(proj, OUT / "reload")[0]
CR.remember_tam(bst, CR.recall_tam(w.study))                                           # the lab's matrix of this session (the PASS comes through a TAM, which is session data)
check(CR.verify_stored(bst, w.results, dlg.test)["verdict"] == "PASS", "the stored result survives a save / load as current (the test data of the session at hand)")
check(CR.verify_stored(bst, w.results)["verdict"] == "UNVERIFIABLE", "a reopened project without the test data: the stored result is UNVERIFIABLE, not PASS")
check(dlg.t_limits_all.rowCount() == len(C.LIMIT_INFO) and dlg.c_rotation.currentData() == "columns", "the Limits tab and the rotation setting are there")
w.results, kept = None, w.results
dlg.refresh()
check("UNVERIFIABLE" in dlg.l_verdict.text(), f"the discarded results: the dashboard says UNVERIFIABLE ({dlg.l_verdict.text()[:50]})")
w.results = kept
dlg.refresh()

# a limit edited after the confirmation: the stored PASS is no longer shown
limit = next(x for x in w.study.space.requirement_set.limits if x.key == "corr_mac_min")
limit.value = 0.5
dlg.show_status()
check("Verdict: PASS" not in dlg.l_verdict.text() and "UNCONFIRMED_REQUIREMENTS" in dlg.l_verdict.text(), f"stale result is not shown as PASS ({dlg.l_verdict.text()[:90]})")
limit.value = 0.9                                                                      # the value is restored: the content is the confirmed one again
from fea_core.space.requirements import confirm  # noqa: E402
confirm(w.study.space.requirement_set, "test engineer", "2026-01-01", scope=w.study.space)
dlg.show_status()
check(dlg.l_verdict.text().startswith("Verdict: PASS"), "the same limits confirmed again are the same judgement basis: the result is current again")
limit.value = 0.8
confirm(w.study.space.requirement_set, "test engineer", "2026-01-02", scope=w.study.space)           # other limits, confirmed: the old PASS is OUTDATED
dlg.show_status()
check("Verdict: OUTDATED" in dlg.l_verdict.text(), f"re-confirmed with other limits: OUTDATED ({dlg.l_verdict.text()[:60]})")
limit.value = 0.9
confirm(w.study.space.requirement_set, "test engineer", "2026-01-01", scope=w.study.space)

# the damping transfer goes through MainWindow.changed(): the results are discarded
dlg.k_damping.setChecked(True)
dlg.run()
dlg.wait_for_import()
check(w.results is None and w.study.dynamics.damping.kind == "modal" and w.study.dynamics.damping.table, "damping written into the dynamics, the dependent results were discarded")
check(any("discarded" in m for m in _warn), "the user is told that the results were discarded")
dlg.close()
w.dirty = False
w.close()
print("FAILED: %d" % len(fails) if fails else "ALL OK")
sys.exit(1 if fails else 0)

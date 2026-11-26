"""A pass through the section "Manufacturing constraints" of the topology dialog (fea_gui/topo_manufacturing.py), without ccx.

  python tests/gui/hidden_run.py tests/gui/gui_topo_manufacturing_smoke.py OUT_DIR

A small mirror-symmetric block and a substitute solver: the form → TopologyOptSpec, the planes / the build direction from a selected face, the arrow
of the build direction in the 3D view, a run in the worker thread with the three constraints and the result lines in the log (checks, penalty).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_topo_manufacturing_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

FAIL: list[str] = []
MESSAGES: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "ERROR ") + text)
    if not cond:
        FAIL.append(text)


class FakeView:
    """Records what the dialog draws in 3D and offers a selected face."""

    def __init__(self):
        self.mode = "faces"
        self.sets: dict[str, int] = {}
        self.faces: list[tuple[int, int]] = []

    def selected_faces(self):
        return list(self.faces)

    def add_lines(self, key, *a, **k):
        self.sets[key] = self.sets.get(key, 0) + 1

    def add_arrows(self, key, *a, **k):
        self.sets[key] = self.sets.get(key, 0) + 10

    def remove_face_set(self, key):
        self.sets.pop(key, None)

    def _update(self):
        pass


class FakeWin(QWidget):
    def __init__(self):
        super().__init__()
        self.view = FakeView()
        self.props = {"type": "plane", "normal": [0.0, 0.0, 1.0], "point": [10.0, 5.0, 7.5]}

    def face_prop(self, pi, fi):
        return self.props

    def log_msg(self, *a):
        pass


def main():
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: MESSAGES.append(str(text))))
    import fea_core.ccx.inp as inp_mod
    import fea_core.solve as solve_mod
    from fea_core.topo import TopologyOptSpec
    from fea_gui.topo_dialog import TopologyOptDialog
    from tests.test_topo_constraints import mirrored_mesh
    from tests.test_topo_feedback import beam_study

    app = QApplication(sys.argv)
    mesh = mirrored_mesh(nx=6, ny=2, nz=2)
    mesh.surfaces = {}
    study = beam_study()

    def fake_run_study(st, workdir, **kw):
        inp_mod._ELEMENT_SCALE.used += 1
        x = mesh.nodes
        U = np.column_stack([np.zeros(len(x)), -1e-3 * (x[:, 0] + 0.3 * x[:, 1] + 0.2 * x[:, 2]) ** 2, np.zeros(len(x))])
        return type("R", (), {"steps": [type("S", (), {"U": U})()]})()
    solve_mod.run_study = fake_run_study

    win = FakeWin()
    dlg = TopologyOptDialog(win, study, mesh, None)
    box = dlg.manufacturing
    dlg.show()
    app.processEvents()

    # the default = no constraints, the old call of the spec
    kw = box.spec_kwargs()
    check(kw == {"symmetry": [], "extrusion": None, "overhang": None, "compare_unconstrained": False}, "default form = no manufacturing constraint")
    check(not box.any_active() and not box.ov_angle.isEnabled() and not box.ext_dir.isEnabled(), "the controls of unused constraints are disabled")

    # symmetry plane from a selected face (z = 7.5)
    win.view.faces = [(0, 0)]
    box.btn_sym_face.click()
    check(box.sym_on[2].isChecked() and abs(box.sym_pos[2].value() - 7.5) < 1e-9, "symmetry plane Z taken from the selected face (z = 7.5)")
    win.props = {"type": "plane", "normal": [0.6, 0.0, 0.8], "point": [0.0, 0.0, 0.0]}
    box.btn_sym_face.click()
    check(any("X, Y or Z" in m for m in MESSAGES), "an oblique face is refused with a message")
    win.props = {"type": "cylinder", "normal": [0.0, 0.0, 1.0], "point": [0.0, 0.0, 0.0]}
    box.btn_sym_face.click()
    check(any("not planar" in m for m in MESSAGES), "a non-planar face is refused with a message")

    # extrusion + overhang
    box.sym_on[2].setChecked(False)
    box.sym_on[0].setChecked(True)
    box.sym_pos[0].setValue(30.0)
    box.ext_on.setChecked(True)
    box.ext_dir.setCurrentIndex(2)
    box.ov_on.setChecked(True)
    box.ov_angle.setValue(50.0)
    box.ov_dir.setCurrentIndex(4)                                   # +Y
    check(win.view.sets.get("print:build", 0) >= 11, "the build direction arrow and the outline of the plate are drawn")
    box.ov_dir.setCurrentIndex(0)
    win.props = {"type": "plane", "normal": [0.0, -1.0, 0.0], "point": [0.0, 0.0, 0.0]}
    box.btn_ov_face.click()
    check(box.build_vector() == [0.0, 1.0, 0.0], "build direction from the face on the plate = its inward normal (+Y)")
    kw = box.spec_kwargs()
    check(kw["symmetry"] == [{"point": [30.0, 5.0, 5.0], "normal": [1.0, 0.0, 0.0]}], "symmetry from the form (plane x = 30)")
    check(kw["extrusion"] == {"direction": [0.0, 0.0, 1.0]} and kw["overhang"]["angle_deg"] == 50.0 and kw["overhang"]["base"] == "min",
          "extrusion and overhang from the form")
    spec = TopologyOptSpec(**kw)
    check(spec.has_constraints and len(box.describe(kw)) == 3, "the form gives a valid TopologyOptSpec and 3 log lines")
    box.ov_on.setChecked(False)
    check("print:build" not in win.view.sets, "the arrow disappears with the overhang switched off")

    # a run in the worker thread: symmetry + comparison run
    box.ext_on.setChecked(False)
    box.sym_pos[0].setValue(30.0)
    box.compare.setChecked(True)
    dlg.sp_target_vol.setValue(60.0)
    dlg.sp_er.setValue(10.0)
    dlg.sp_filter_r.setValue(7.0)
    dlg.sp_max_iter.setValue(6)
    for i in range(dlg.list_nd.count()):
        dlg.list_nd.item(i).setSelected(False)
    dlg.btn_run.click()
    check(not box.isEnabled(), "the manufacturing section is disabled during the run")
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(lambda: loop.quit() if dlg.worker is not None and dlg.worker.isFinished() else None)
    timer.start()
    QTimer.singleShot(120000, loop.quit)
    loop.exec()
    for _ in range(5):
        app.processEvents()
    log = dlg.txt_log.toPlainText()
    check(dlg.worker is not None and dlg.worker.isFinished() and dlg.opt_result is not None, "the optimisation finished")
    check(any(w in log for w in ("Symmetry: X = 30.000 mm", "Symetrie: X = 30.000 mm", "Symmetry: x = 30.000 mm")), "the log names the symmetry plane")
    check(any(w in log for w in ("identical", "shodná")) and any(w in log for w in ("Compliance penalty of the constraints", "Cena omezení v poddajnosti")),
          "the log shows the check of the mirror images and the compliance penalty")
    check(box.isEnabled() and dlg.btn_run.text() != "Stop", "the controls are restored after the run")
    dlg.grab().save(str(OUT / "topo_dialog.png"))
    info = dlg.opt_result.constraint_info if dlg.opt_result is not None else None
    check(info is not None and info["symmetry"]["max_group_spread"] == 0.0, "the result has the constraint block, mirror groups identical")
    dlg.close()
    check("print:build" not in win.view.sets, "closing the dialog clears the 3D arrow")
    print("\n".join(FAIL) if FAIL else "ALL OK")
    sys.exit(1 if FAIL else 0)


main()

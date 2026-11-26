"""run_busy helpers without a main window (Qt offscreen): the input blocker leaves windows opened during the operation
operable, the progress window does not hide on Escape, and post-processing fields are not calculation inputs."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_blocker_keeps_new_message_box_operable(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from fea_gui.main import _InputBlocker, _busy_progress
    old = QtWidgets.QPushButton("old window")       # existed before the operation – stays blocked
    old.show()
    dlg = _busy_progress("busy", False, None)
    dlg.show()
    blocker = _InputBlocker(dlg)
    app.installEventFilter(blocker)
    try:
        hits = []
        old.clicked.connect(lambda: hits.append("old"))
        QTest.mouseClick(old, Qt.LeftButton)
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "FEA Studio", "Analysis failed",
                                    QtWidgets.QMessageBox.Ok)
        box.show()
        app.processEvents()
        ok = box.button(QtWidgets.QMessageBox.Ok)
        ok.clicked.connect(lambda: hits.append("ok"))
        QTest.mouseClick(ok, Qt.LeftButton)
        app.processEvents()
        assert hits == ["ok"] and not box.isVisible()
    finally:
        app.removeEventFilter(blocker)
        dlg.hide()
        old.close()


@pytest.mark.parametrize("cancellable", [False, True])
def test_progress_escape(app, cancellable):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from fea_gui.main import _busy_progress
    dlg = _busy_progress("busy", cancellable, None)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    hits = []
    dlg.canceled.connect(lambda: hits.append("canceled"))
    dlg.show()
    app.processEvents()
    QTest.keyClick(dlg, Qt.Key_Escape)
    app.processEvents()
    dlg.close()
    app.processEvents()
    if cancellable:
        assert hits and hits[0] == "canceled"
    else:
        assert hits == [] and dlg.isVisible()
    dlg.hide()


def test_calc_inputs_strip_post_processing():
    from fea_core.study import Study
    from fea_gui.main import MainWindow
    st = Study("s", "")
    d = st.to_dict()
    d["load_cases"] = [{"name": "LC1", "loads": [], "category": "cat_B"}]
    d["combinations"] = [{"name": "K1"}]
    d["features"] = ["combinations", "load_categories"]
    d["printing"] = {"parts": {}, "variants": [{"name": "v"}], "sweep": {"x": 1}, "required_safety": 2.0}
    d["joints"] = {"preload_case": False, "slip_mu": 0.3, "weld_colors": ["#ff0000"]}
    d["bolts"] = [{"name": "B1", "strength": "8.8", "mu_T": 0.2, "preload": 1000.0}]
    c = MainWindow.calc_inputs(json.loads(json.dumps(d)), joints=True)
    assert "combinations" not in c and "features" not in c
    assert c["load_cases"] == [{"name": "LC1", "loads": []}]
    assert c["printing"] == {"parts": {}}
    assert c["joints"] == {"preload_case": False}
    assert c["bolts"] == [{"name": "B1", "strength": "8.8"}]        # preload is assessment-only without the preload state
    d["joints"]["preload_case"] = True
    c = MainWindow.calc_inputs(json.loads(json.dumps(d)), joints=True)
    assert c["bolts"] == [{"name": "B1", "strength": "8.8", "preload": 1000.0}]
    assert MainWindow.calc_inputs(d)["joints"] == d["joints"]     # undo keeps the joints for joints_change_level

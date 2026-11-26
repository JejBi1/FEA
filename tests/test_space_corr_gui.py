"""F14 S5: the test correlation dashboard (`fea_gui.space_corr_panel`) with a minimal window (Qt offscreen, no 3D view), as the dialog of stage S1."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from test_space_correlation import _orthogonality_mass  # noqa: F401, E402  (autouse fixture)
from test_space_correlation import EXAMPLE, FREQS, SENSORS, _box_mesh, _limits_set, _modes, _test_data  # noqa: E402  (the synthetic FE model of the gate)

from fea_core.results import Results  # noqa: E402
from fea_core.space import corr_run as CR  # noqa: E402
from fea_core.space import correlation as C  # noqa: E402
from verification.space_cases import block_study  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeWin(QtWidgets.QWidget):
    def __init__(self, confirmed: bool):
        super().__init__()
        mesh = _box_mesh()
        self.fe = (mesh, _modes(mesh))
        self.study = block_study()
        self.study.space = _limits_set(EXAMPLE, confirmed=confirmed)
        self.results = Results(mesh=mesh, steps=[], modes=[self.fe[1]])
        self.n_changed = 0
        self.n_full_changes = 0

    def mark_changed(self, level="results", refresh=None):
        assert level == "none"                      # the results stay valid: only the file changes
        self.n_changed += 1
        return True

    def changed(self, invalidate_results=True, invalidate_mesh=False):
        """The way of every edit of the dynamics (`MainWindow.changed`): the results that depend on the study are discarded."""
        self.n_full_changes += 1
        self.results = None
        return True


def _make(app, monkeypatch, tmp_path, confirmed: bool):
    from fea_gui.space_corr_panel import CorrelationDialog
    warnings: list[str] = []
    monkeypatch.setattr(CorrelationDialog, "_warn", lambda self, text: warnings.append(text))
    w = FakeWin(confirmed)
    d = CorrelationDialog(w)
    d.warnings, d.fake, d.tmp = warnings, w, tmp_path
    return d


def _import(d, monkeypatch, text: str, name: str = "modes.csv"):
    p = d.tmp / name
    p.write_text(text)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(p), "")))
    d.import_file()
    d.wait_for_import()                                                          # the file is read on a worker thread


def _fill_sensors(d):
    t = d.t_sensors
    by_node = {s.node: s for s in SENSORS}
    for r in range(t.rowCount()):
        s = by_node[t.item(r, 0).text()]
        for k in range(3):
            t.item(r, 1 + k).setText(str(s.point[k]))


def test_dashboard_confirmed_limits_give_pass_and_draw_the_mac(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    assert not d.b_run.isEnabled() and "No test data" in d.l_data.text()
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    assert d.warnings == [] and "6 modes" in d.l_data.text() and d.b_run.isEnabled() and d.t_sensors.rowCount() == 8
    assert d.tabs.currentIndex() == 3                              # the sensors are not mapped: the sensor tab is shown
    _fill_sensors(d)
    d.run()
    d.wait_for_import()
    assert d.warnings == [] and d.result is not None and d.result.verdict == C.PASS
    assert d.l_verdict.text().startswith("Verdict: PASS")
    assert d.t_rows.rowCount() == len(d.result.rows) and d.t_pairs.rowCount() == 6 and d.t_limits.rowCount() == len(C.LIMIT_INFO)
    statuses = {d.t_rows.item(i, 4).text() for i in range(d.t_rows.rowCount())}
    assert statuses == {"PASS"}
    assert d.plot_mac.canvas is not None and len(d.plot_mac.fig.axes[0].images) == 1                    # the heat map
    assert d.fake.n_changed >= 2 and CR.state_of(d.fake.study)["result"]["verdict"] == "PASS" and len(CR.load_sensors(d.fake.study)) == 8
    assert not [w for w in d.findChildren(QtWidgets.QLineEdit) if not isinstance(w.parent(), QtWidgets.QAbstractSpinBox)]    # no field for a limit or a confirmation
    d.close()


def test_dashboard_unconfirmed_limits_never_show_pass(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, False)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.run()
    d.wait_for_import()
    assert d.result.verdict == C.VERDICT_UNCONFIRMED and "not an ECSS result" in d.l_verdict.text()
    assert {d.t_rows.item(i, 4).text() for i in range(d.t_rows.rowCount())} == {"UNCONFIRMED"}
    assert "PASS" not in d.l_verdict.text().replace("PASS / FAIL", "")
    d.close()


def test_dashboard_refuses_a_bad_file_and_a_bad_sensor_cleanly(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, "mode,frequency_hz\n1,abc\n", "bad.csv")
    assert d.test is None and d.warnings and "not a number" in d.warnings[0]
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    d.t_sensors.item(0, 1).setText("12")
    d.t_sensors.item(0, 2).setText("x")
    d.t_sensors.item(0, 3).setText("1")
    d.warnings.clear()
    d.save_sensors()
    assert d.warnings and "numbers" in d.warnings[0] and CR.load_sensors(d.fake.study) == []
    d.run()                                                                                           # the bad sensor table stops the run
    d.wait_for_import()
    assert d.result is None and d.warnings
    d.close()


def test_dashboard_response_tab_and_limit_rows(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    f = np.linspace(5.0, 200.0, 300)
    u = np.zeros((300, 3), complex)
    u[:, 1] = C.modal_frf(f, np.array(FREQS), 0.02, np.ones(6), np.ones(6), "receptance")
    d.fake.results.dynamics = SimpleNamespace(harmonic={"f": f, "probes": {"100": {"U": u}}}, random=None)
    d.fake.study.dynamics.probes = [SimpleNamespace(name="100", point=[10.0, 5.0, 5.0], selection=None)]    # type: ignore[list-item]  # test double of Probe
    td = _test_data(d.fake.fe)
    td.functions = [C.TestFunction("100:y", "100", "y", f, -((2 * np.pi * f) ** 2) * u[:, 1] / 9810.0 * 1.2, "frf", "g")]
    CR.update_setups(d.fake.study, {"100:y": {"frame": "relative"}})
    _import(d, monkeypatch, C.write_modes_csv(td.modes) + "\n")
    d.test = td
    _fill_sensors(d)
    d.t_sensors.item(0, 4).setText("100")
    d.run()
    d.wait_for_import()
    assert d.result is not None and len(d.result.responses) == 1 and d.c_func.count() == 1
    assert "amplitude error" in d.l_frf.text() and d.plot_frf.canvas is not None
    n0 = len(d.fake.study.space.requirement_set.limits)
    d.add_limit_rows()
    assert len(d.fake.study.space.requirement_set.limits) == n0                                            # the rows exist already
    d.fake.study.space = None
    d.add_limit_rows()
    assert d.warnings[-1].startswith("There is no requirement set")
    d.close()


def test_the_menu_entry_and_the_opening_function_exist(app, tmp_path):
    import inspect

    from fea_gui import extras
    assert "open_correlation_dialog" in inspect.getsource(extras.install_extras)
    from fea_gui.space_corr_panel import open_correlation_dialog
    w = FakeWin(True)
    dlg = open_correlation_dialog(w)
    assert dlg is not None and open_correlation_dialog(w) is dlg                                           # single per window
    dlg.close()
    app.processEvents()                                                                                      # the dialog deletes itself on close: let it happen while the window is alive
    app.sendPostedEvents(None, 0)


# ------------------------------------------------------------------ the review fixes
def _run_pass(d, monkeypatch):
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.run()
    d.wait_for_import()
    assert d.result is not None and d.result.verdict == C.PASS and d.l_verdict.text().startswith("Verdict: PASS")


def test_the_import_runs_on_a_worker_thread(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    p = tmp_path / "m.csv"
    p.write_text(C.write_modes_csv(_test_data(d.fake.fe).modes))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(p), "")))
    d.import_file()
    assert d._thread is not None and not d.b_import.isEnabled() and d.test is None and "Reading" in d.l_data.text()     # the window is not blocked: nothing is taken over yet
    d.import_file()                                                                                               # a second import while one is running is ignored
    d.wait_for_import()
    assert d._thread is None and d.b_import.isEnabled() and d.test is not None and "6 modes" in d.l_data.text() and d.warnings == []
    d.close()


def test_a_stored_pass_is_never_shown_after_the_limits_changed_or_for_a_forged_record(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _run_pass(d, monkeypatch)
    limit = next(x for x in d.fake.study.space.requirement_set.limits if x.key == "corr_mac_min")
    limit.value = 0.5                                                                      # edited after the confirmation
    d.show_status()
    assert "Verdict: PASS" not in d.l_verdict.text() and "UNCONFIRMED_REQUIREMENTS" in d.l_verdict.text() and "no longer the one" in d.l_verdict.text()
    d.result = None                                                                        # the dialog is opened again later
    d.refresh()
    assert d.l_verdict.text().startswith("Stored result of the last correlation: UNCONFIRMED_REQUIREMENTS (was PASS;")
    d.close()
    d2 = _make(app, monkeypatch, tmp_path, True)
    d2.fake.study.space.extra["correlation"] = {"result": {"verdict": "PASS", "rows": []}}
    d2.refresh()
    assert "Stored result of the last correlation: OUTDATED" in d2.l_verdict.text() and "no provenance" in d2.l_verdict.text()
    d2.close()


def test_a_damping_transfer_goes_the_way_of_every_dynamics_edit_and_discards_the_results(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.k_damping.setChecked(True)
    d.run()
    d.wait_for_import()
    assert d.result is not None and d.fake.n_full_changes == 1 and d.fake.results is None
    assert d.fake.study.dynamics.damping.kind == "modal" and d.fake.study.dynamics.damping.table[0][1] == 0.01
    assert "discarded" in d.warnings[-1]
    d.close()
    d = _make(app, monkeypatch, tmp_path, True)
    _run_pass(d, monkeypatch)
    assert d.fake.n_full_changes == 0 and d.fake.results is not None                          # without a transfer the results stay valid
    d.close()


def test_the_coverage_the_partial_scope_and_the_way_the_set_was_confirmed_are_shown(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    d.fake.study.space.requirement_set.confirmed_via = "mcp"
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.s_fmax.setValue(100.0)
    d.run()
    d.wait_for_import()
    assert d.result is not None and d.result.verdict == C.VERDICT_PARTIAL and d.l_verdict.text().startswith("Verdict: PARTIAL_SCOPE") and "not a PASS of the test" in d.l_verdict.text()
    assert "Correlated: 4 of 6 test modes" in d.l_cov.text() and "LEFT OUT" in d.l_cov.text() and "frequency window" in d.l_cov.toolTip() and "0 - 100" in d.l_cov.text()
    assert "confirmed through the MCP interface" in d.l_req.text() and "test engineer" in d.l_req.text()
    d.close()


def test_the_function_tab_stores_the_frame_the_normalisation_and_the_unit(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    td = _test_data(d.fake.fe)
    f = np.linspace(5.0, 200.0, 50)
    td.functions = [C.TestFunction("100:y", "100", "y", f, np.ones(50), "frf", "NONE"), C.TestFunction("coh", "100", "y", f, np.ones(50), "info", "")]
    d.test = td
    d.refresh()
    tbl = d.fn_table.table
    assert tbl.rowCount() == 2 and "not known" in tbl.item(0, 10).text() and tbl.item(1, 3).text() == "information only" and not tbl.cellWidget(1, 4).isEnabled()
    tbl.cellWidget(0, 4).setCurrentIndex(tbl.cellWidget(0, 4).findData("absolute"))
    tbl.item(0, 6).setText("g/g")
    tbl.item(0, 7).setText("p1")
    assert d.save_functions()
    su = CR.load_setups(d.fake.study)["100:y"]
    assert (su.frame, su.unit, su.probe) == ("absolute", "g/g", "p1") and tbl.item(0, 3).text() == "acceleration per base acceleration" and tbl.item(0, 10).text() == ""
    tbl.item(0, 9).setText("abc")
    d.warnings.clear()
    assert not d.save_functions() and "reference force must be a number" in d.warnings[0]
    tbl.item(0, 9).setText("")
    tbl.item(0, 6).setText("furlong")
    assert not d.save_functions() and "not understood" in d.warnings[-1]
    d.close()


def test_names_from_a_study_file_are_shown_as_plain_text(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    CR.state_of(d.fake.study).clear()
    d.fake.study.space.extra["correlation"] = {"sensors": [{"node": "100", "point": [1, 2, 3], "probe": "<b>bold</b>\nnext line", "dofs": "xyz"}]}
    d.fill_sensors()
    assert "\n" not in d.t_sensors.item(0, 4).text() and d.t_sensors.item(0, 4).text().startswith("<b>bold</b>")        # shown as the text it is (the cells are plain text)
    assert d.l_verdict.textFormat() == QtWidgets.QLabel().textFormat().PlainText and d.l_data.textFormat() == d.l_verdict.textFormat() == d.l_cov.textFormat() == d.l_req.textFormat()
    d.close()

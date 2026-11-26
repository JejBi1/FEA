"""The cross-orthogonality controls of the test-correlation dashboard (F14 S5): the mass source, the import of a TAM matrix, the COC columns, the MAC / COC switch and the hints.
GUI test: run it through tests/gui/hidden_run.py (python tests/gui/hidden_run.py <pytest_main.py> tests/test_space_corr_gui_orth.py)."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from test_space_corr_gui import FakeWin, _fill_sensors, _import, _make, app  # noqa: E402, F401
from test_space_correlation import SENSORS, _orthogonality_mass, _test_data  # noqa: E402, F401  (the autouse fixture)

from fea_core.space import corr_orth as O  # noqa: E402
from fea_core.space import correlation as C  # noqa: E402


def test_the_dashboard_shows_the_coc_and_switches_the_heat_map(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    assert d.c_orth.currentData() == "serep" and not d.b_tam.isEnabled()
    d.run()
    d.wait_for_import()
    assert d.warnings == [] and d.result.verdict == C.PASS and d.result.modal.orth.computed
    cols = [d.t_pairs.horizontalHeaderItem(j).text() for j in range(d.t_pairs.columnCount())]
    assert cols[-4:] == ["COC", "Max. other COC", "Modulus diff.", "Out of FE span"]
    assert float(d.t_pairs.item(0, cols.index("COC")).text()) > 0.99
    assert "Reduced mass" in d.l_orth.text()
    d.c_matrix.setCurrentIndex(d.c_matrix.findData("coc"))
    assert len(d.plot_mac.fig.axes[0].images) == 1 and d.plot_mac.fig.axes[-1].get_ylabel() == "COC"
    metrics = {d.t_rows.item(i, 1).text() for i in range(d.t_rows.rowCount())}
    assert {"COC diagonal", "COC off-diagonal", "AOC off-diagonal", "modulus difference"} <= metrics
    lim = d.t_limits_all
    header = [lim.horizontalHeaderItem(j).text() for j in range(lim.columnCount())]
    rows = {lim.item(i, 0).text(): lim.item(i, header.index("Hint of the standard (not normative)")).text() for i in range(lim.rowCount())}
    assert "0,90" in rows["corr_coc_diag_min"] and rows["corr_mac_min"] == ""                  # a hint, never a value
    d.close()


def test_without_a_mass_the_rows_are_unconfirmed_and_a_tam_file_can_be_imported(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.c_orth.setCurrentIndex(d.c_orth.findData("none"))
    d.run()
    d.wait_for_import()
    assert d.result.verdict == C.VERDICT_UNCONFIRMED and d.result.modal.orth.problems
    d.c_orth.setCurrentIndex(d.c_orth.findData("tam"))
    assert d.b_tam.isEnabled()
    d.run()
    d.wait_for_import()
    assert not d.result.modal.orth.computed and "no mass matrix" in d.result.modal.orth.problems[0]            # nothing imported yet
    dofs = [(s.node, x) for s in SENSORS for x in s.dofs]
    lines = [",".join([""] + [f"{n}:{x}" for n, x in dofs])] + [",".join([f"{n}:{x}"] + [("1" if i == j else "0") for j in range(len(dofs))]) for i, (n, x) in enumerate(dofs)]
    p = tmp_path / "tam.csv"
    p.write_text("# unit: kg\n" + "\n".join(lines) + "\n")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(p), "")))
    d.import_tam()
    assert d.warnings == [] and d.tam is not None and "24 DOFs" in d.l_orth.text() and d.c_orth.currentData() == "tam"
    d.run()
    d.wait_for_import()
    assert d.result.modal.orth.computed and d.result.modal.orth.source == "tam"
    bad = tmp_path / "bad.csv"
    bad.write_text(",A:x\nA:x,-1\n")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
    d.import_tam()
    assert d.warnings and "positive semi-definite" in d.warnings[-1] and d.tam.source == "tam.csv"                # refused: the old matrix stays
    assert isinstance(d.tam, O.MassMatrix) and np.allclose(d.tam.matrix, np.eye(24))
    d.close()


def test_the_correlation_runs_on_a_worker_thread_and_the_study_is_written_in_the_main_thread(app, monkeypatch, tmp_path):
    from fea_core.space import corr_run as CR
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.run()
    assert d._thread is not None and d.result is None and not d.b_run.isEnabled() and "Correlating" in d.l_verdict.text()      # nothing taken over yet: the window stays responsive
    assert CR.state_of(d.fake.study).get("result") is None                                                                      # and the study was not written by the worker
    d.wait_for_import()
    assert d._thread is None and d.result is not None and d.result.verdict == C.PASS and d.b_run.isEnabled()
    assert CR.state_of(d.fake.study)["result"]["verdict"] == "PASS"
    d.close()


def test_the_tam_axes_box_is_only_for_a_tam_and_reaches_the_options(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    assert not d.k_tamaxes.isChecked() and not d.k_tamaxes.isEnabled() and "GLOBAL" in d.k_tamaxes.toolTip()
    d.c_orth.setCurrentIndex(d.c_orth.findData("tam"))
    assert d.k_tamaxes.isEnabled()
    d.k_tamaxes.setChecked(True)
    d.run()
    d.wait_for_import()
    assert d.result.options.tam_global_axes is True and "global axes" in d.result.options.describe()
    d.close()


def test_the_serep_mass_is_marked_indicative_in_the_list_the_headers_and_the_rows(app, monkeypatch, tmp_path):
    """The other tests of this file judge the SEREP rows as a TAM's (autouse fixture); here the real rule is back: indicative value, UNCONFIRMED rows, marked columns."""
    monkeypatch.setattr(C, "orthogonality", O.orthogonality)
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    assert "indicative" in d.c_orth.itemText(d.c_orth.findData("serep")) and "UNCONFIRMED" in d.c_orth.toolTip() and "INDICATIVE" in d.c_orth.toolTip()
    d.run()
    d.wait_for_import()
    assert d.result.verdict == C.VERDICT_UNCONFIRMED and d.result.modal.orth.indicative == C.INDICATIVE_REASON
    cols = [d.t_pairs.horizontalHeaderItem(j).text() for j in range(d.t_pairs.columnCount())]
    assert cols[-4:] == ["COC (indicative)", "Max. other COC (indicative)", "Modulus diff.", "Out of FE span"]
    assert "INDICATIVE" in d.t_pairs.horizontalHeaderItem(cols.index("COC (indicative)")).toolTip() and "program's own definition" in d.t_pairs.horizontalHeaderItem(cols.index("Modulus diff.")).toolTip()
    assert float(d.t_pairs.item(0, cols.index("COC (indicative)")).text()) > 0.99                    # the number is shown
    assert "indicative" in d.l_orth.text()                                                           # and the reason beside the reduced mass
    coc_status = {d.t_rows.item(i, 4).text() for i in range(d.t_rows.rowCount()) if d.t_rows.item(i, 1).text() in ("COC diagonal", "COC off-diagonal", "AOC off-diagonal")}
    assert coc_status == {"UNCONFIRMED"}
    d.c_orth.setCurrentIndex(d.c_orth.findData("none"))
    d.close()

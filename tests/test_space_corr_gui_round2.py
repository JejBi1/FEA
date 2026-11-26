"""Round-2 review fixes of the correlation dashboard (F14-S5, 25. 11. 2026): a worker result is dropped when the study or the test data changed while it ran; the statement about the axes
of the TAM is withdrawn at an import and records the interface; closing the window during a run neither blocks nor destroys the running thread."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from test_space_corr_gui import EXAMPLE, FakeWin, _fill_sensors, _import, _make, app  # noqa: E402, F401
from test_space_correlation import _limits_set, _orthogonality_mass, _test_data  # noqa: E402, F401

from fea_core.space import corr_run as CR  # noqa: E402
from fea_core.space import correlation as C  # noqa: E402
from verification.space_cases import block_study  # noqa: E402


def _started(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    return d


def test_the_result_is_dropped_when_another_study_is_loaded_while_the_worker_runs(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    study_a = d.fake.study
    d.run()
    assert d._thread is not None
    study_b = block_study()
    study_b.name = "another project"
    study_b.space = _limits_set(EXAMPLE, confirmed=True)
    damping_b = C.damping_fingerprint(study_b)
    d.fake.study = study_b                                                       # the user opens another project before the result is taken over
    d.wait_for_import()
    assert "result" not in CR.state_of(study_a) and "result" not in CR.state_of(study_b)       # neither study got the result
    assert C.damping_fingerprint(study_b) == damping_b and d.result is None
    assert d.warnings and "discarded" in d.warnings[-1] and "study or the test data" in d.warnings[-1]
    d.close()


def test_the_result_is_dropped_when_the_test_data_are_replaced_while_the_worker_runs(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    d.run()
    assert d._thread is not None
    d.test = _test_data(d.fake.fe)                                               # another test dataset object
    d.wait_for_import()
    assert "result" not in CR.state_of(d.fake.study) and d.result is None and "discarded" in d.warnings[-1]
    d.close()


def test_an_undisturbed_run_is_still_taken_over(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    d.run()
    d.wait_for_import()
    assert d.result is not None and CR.state_of(d.fake.study)["result"]["verdict"] == "PASS" and not any("discarded" in w for w in d.warnings)
    d.close()


def test_closing_during_a_run_hides_the_window_waits_for_nothing_and_closes_it_when_the_worker_is_done(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    d.show()
    d.run()
    assert d._thread is not None
    closed: list[int] = []
    d.finished.connect(closed.append)
    d.close()
    assert d._thread is not None and d._close_pending and not d.isVisible() and not closed        # hidden at once, the thread is alive, nothing was waited for
    d.wait_for_import()
    assert d._thread is None and not d._close_pending and closed == [0]                           # the worker finished: the dialog closed itself
    assert CR.state_of(d.fake.study)["result"]["verdict"] == "PASS"                              # (the result of the run is kept)


def test_showing_the_dialog_again_takes_the_pending_close_back(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    d.show()
    d.run()
    closed: list[int] = []
    d.finished.connect(closed.append)
    d.close()
    assert d._close_pending
    d.show()
    assert not d._close_pending
    d.wait_for_import()
    assert d.isVisible() and closed == []
    d.close()
    assert closed == [0]                                                                          # no worker: it closes now


def test_importing_a_mass_matrix_withdraws_the_axes_statement_and_the_run_records_the_interface(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    tam_file = tmp_path / "m.csv"
    tam_file.write_text("# unit: kg\n,A:x,A:y\nA:x,2,0\nA:y,0,2\n")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(tam_file), "")))
    d.k_tamaxes.setChecked(True)
    d.import_tam()
    assert not d.k_tamaxes.isChecked() and d.c_orth.currentData() == "tam"                       # the statement was about the PREVIOUS matrix
    assert d.tam is not None and d.tam.rank == 2 and d.tam.study_key == CR.study_key(d.fake.study)
    d.c_orth.setCurrentIndex(d.c_orth.findData("serep"))
    d.k_tamaxes.setEnabled(True)
    d.k_tamaxes.setChecked(True)
    d.run()
    d.wait_for_import()
    opt = d.result.options
    assert opt.tam_global_axes and opt.tam_global_axes_via == "gui" and CR.state_of(d.fake.study)["options"]["tam_global_axes_via"] == "gui"
    d.close()

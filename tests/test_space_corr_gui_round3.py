"""Round-3 review of the correlation dashboard (F14-S5, 25. 11. 2026): the rotation convention of the test data cannot change under a running correlation, and a result computed in another
state is discarded (a stored current PASS of a file that was read with the other convention)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from test_space_corr_gui import _fill_sensors, _import, _make, app  # noqa: E402, F401
from test_space_correlation import _orthogonality_mass, _test_data  # noqa: E402, F401

from fea_core.space import corr_run as CR  # noqa: E402
from fea_core.space import correlation as C  # noqa: E402


def _started(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    return d


def test_the_rotation_controls_are_locked_while_the_worker_runs_and_unlocked_after(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    assert d.c_rotation.isEnabled()
    d.run()
    assert d._thread is not None and not d.c_rotation.isEnabled() and not d.k_rotation.isEnabled()
    before = d.test.rotation
    d.c_rotation.setCurrentIndex(1 - d.c_rotation.currentIndex())                         # a programmatic change reaches the slot: it must refuse while the worker runs
    assert d.test.rotation == before
    d.wait_for_import()
    assert d.c_rotation.isEnabled() and d.result is not None and CR.state_of(d.fake.study)["result"]["verdict"] == "PASS"
    d.close()


def test_a_convention_changed_behind_the_guard_during_the_run_voids_the_result(app, monkeypatch, tmp_path):
    d = _started(app, monkeypatch, tmp_path)
    d.run()
    assert d._thread is not None
    d.test.set_rotation("rows" if d.test.rotation == "columns" else "columns")             # a script / an agent flips the convention while the worker reads the shapes
    d.wait_for_import()
    assert d.result is None and "result" not in CR.state_of(d.fake.study)
    assert d.warnings and ("rotation convention" in d.warnings[-1])
    assert d.c_rotation.isEnabled() and d._thread is None
    d.close()

"""F14 S5 re-check fixes in the GUI: the dashboard shows the verified verdict (UNVERIFIABLE / OUTDATED, never a stale PASS - N1), the read-only table of the correlation limits with their
not-applicable flags and sources (N5), and the convention of the local sensor axes as an explicit setting that the user confirms (N3); the Space dialog shows the limits before the
user confirms the requirement set (N5)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from test_space_corr_gui import FakeWin, _fill_sensors, _import, _make  # noqa: E402, F401
from test_space_correlation import _orthogonality_mass, _test_data  # noqa: E402, F401  (the autouse fixture)

from fea_core.space import corr_run as CR  # noqa: E402
from fea_core.space import correlation as C  # noqa: E402
from fea_core.space.model import LimitEntry  # noqa: E402
from fea_core.space.requirements import confirm  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _run_pass(d, monkeypatch):
    _import(d, monkeypatch, C.write_modes_csv(_test_data(d.fake.fe).modes))
    _fill_sensors(d)
    d.run()
    d.wait_for_import()
    assert d.result is not None and d.result.verdict == C.PASS


def test_dashboard_never_shows_a_stale_pass_when_the_results_are_gone(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _run_pass(d, monkeypatch)
    assert d.l_verdict.text().startswith("Verdict: PASS")
    d.fake.results = None                                                    # the model was edited: its results were discarded
    d.result = None
    d.refresh()
    text = d.l_verdict.text()
    assert "UNVERIFIABLE" in text and "was PASS" in text and "PASS" not in text.replace("was PASS", "").replace("UNVERIFIABLE", "") and "solved results are not at hand" in text
    assert "Verified against the current study" in d.l_verdict.toolTip() or "Not verifiable" in d.l_verdict.toolTip()
    d.close()


def test_dashboard_shows_outdated_after_the_sensors_changed(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    _run_pass(d, monkeypatch)
    CR.state_of(d.fake.study)["sensors"][0]["point"] = [90.0, 1.0, 1.0]
    d.refresh()
    assert "OUTDATED" in d.l_verdict.text() and "sensors of the study changed" in d.l_verdict.text()
    d.close()


def test_limits_tab_lists_every_correlation_limit_read_only_with_value_unit_na_and_source(app, monkeypatch, tmp_path):
    d = _make(app, monkeypatch, tmp_path, True)
    t = d.t_limits_all
    assert [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())] == ["Limit", "Value", "Unit", "Not applicable", "Document", "Revision", "Clause", "Status", "Meaning", "Hint of the standard (not normative)"]
    assert t.rowCount() == len(C.LIMIT_INFO)
    row = {t.item(i, 0).text(): [t.item(i, j).text() for j in range(t.columnCount())] for i in range(t.rowCount())}
    assert row["corr_mac_min"][1:8] == ["0.9", "-", "no", "doc", "rev", "c1", "usable"] and row["corr_peak_amp_error_max_db"][2] == "dB"
    from PySide6.QtCore import Qt
    assert all(not (t.item(i, j).flags() & Qt.ItemFlag.ItemIsEditable) for i in range(t.rowCount()) for j in range(t.columnCount()))
    assert "confirmed by test engineer" in d.l_limits_by.text()
    d.close()
    # a not-applicable flag set through MCP, and an unconfirmed set
    w = _make(app, monkeypatch, tmp_path, True)
    rs = w.fake.study.space.requirement_set
    rs.limits[:] = [x for x in rs.limits if x.key != "corr_comac_min"]
    rs.limits.append(LimitEntry(key="corr_comac_min", value=None, unit="-", document="Doc-1", revision="B", clause="4.2", not_applicable=True, set_via="mcp"))
    confirm(rs, "test engineer", "2026-01-01", via="gui", scope=w.fake.study.space)
    w.refresh()
    row = {w.t_limits_all.item(i, 0).text(): [w.t_limits_all.item(i, j).text() for j in range(9)] for i in range(w.t_limits_all.rowCount())}
    assert row["corr_comac_min"][1:8] == ["-", "-", "YES", "Doc-1", "B", "4.2", "not applicable"]
    assert "1 criterion/criteria declared NOT APPLICABLE" in w.l_limits_by.text()
    assert "NOT APPLICABLE (not judged): corr_comac_min" in w.l_na.text() and "MCP interface (an agent)" in w.l_na.text()
    w.close()
    u = _make(app, monkeypatch, tmp_path, False)
    assert "NOT confirmed" in u.l_limits_by.text() and u.t_limits_all.item(0, 7).text().startswith("NOT usable")
    u.close()


def test_the_rotation_convention_is_an_explicit_setting_that_the_user_confirms(app, monkeypatch, tmp_path):
    from test_space_corr_recheck import _rotated_modes_uff
    d = _make(app, monkeypatch, tmp_path, True)
    assert not d.k_rotation.isEnabled() and d.c_rotation.currentData() == "columns"
    _import(d, monkeypatch, _rotated_modes_uff(d.fake.fe, "columns"), "rot.uff")
    assert d.test is not None and d.test.rotation == "columns" and d.k_rotation.isEnabled() and not d.k_rotation.isChecked()
    assert "NOT confirmed" in d.l_rot.text() and "UNCONFIRMED" in d.l_rot.text()
    _fill_sensors(d)
    d.run()
    d.wait_for_import()
    assert d.result.verdict == C.VERDICT_UNCONFIRMED and {x.status for x in d.result.rows} == {C.UNCONFIRMED}
    d.k_rotation.setChecked(True)                                            # the user confirms the convention for this file
    assert d.test.rotation_confirmed and d.test.rotation_confirmed_via == "gui" and CR.state_of(d.fake.study)["rotation"]["confirmed"] is True
    d.run()
    d.wait_for_import()
    assert d.result.verdict == C.PASS and d.l_verdict.text().startswith("Verdict: PASS")
    d.c_rotation.setCurrentIndex(d.c_rotation.findData("rows"))              # the other convention: the confirmation is gone, the data are read again
    assert d.test.rotation == "rows" and not d.test.rotation_confirmed and not d.k_rotation.isChecked() and d.result is None
    assert "OUTDATED" in d.l_verdict.text()
    d.close()


def test_the_space_dialog_shows_the_limits_and_the_flags_before_the_requirement_set_is_confirmed(app, monkeypatch, tmp_path):
    from fea_gui.space_panel import SpaceDialog
    w = FakeWin(True)
    rs = w.study.space.requirement_set
    rs.limits.append(LimitEntry(key="corr_rms_error_max_x", value=None, unit="-", document="Doc-2", revision="C", clause="7.1", not_applicable=True, set_via="mcp"))
    rs.withdraw_confirmation()
    d = SpaceDialog(w)
    assert "NOT APPLICABLE" in d.l_limits.text() and "corr_rms_error_max_x" in d.l_limits.text() and "entered through the MCP interface (an agent) - check it" in d.l_limits.text()
    assert "corr_mac_min" in d.l_limits.text() and "0.9" in d.l_limits.text() and d.b_confirm.isEnabled()
    d.close()
    w2 = FakeWin(True)
    w2.study.space.requirement_set = None
    d2 = SpaceDialog(w2)
    assert d2.l_limits.text() == ""
    d2.close()

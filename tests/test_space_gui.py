"""F14 S1: the space-structures dialog (fea_gui.space_panel) with a minimal window (Qt offscreen, no 3D view)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

import verification.cases  # noqa: E402, F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.space import requirements as RQ  # noqa: E402
from verification import space_cases as SC  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeWin(QtWidgets.QWidget):
    def __init__(self, solved: bool = True):
        super().__init__()
        mesh = SC.box_tet_mesh()
        self.study = SC.block_study()
        self.results = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA, spike=3.0)]) if solved else None
        self.parts = []
        self.n_changed = 0
        self.marks: list = []

        class View:
            def mark(inner, xyz, text, color=None, key=None, below=False):
                self.marks.append((list(xyz), text, key))
        self.view = View()

    def mark_changed(self, level="results", refresh=None):
        assert level == "none"                      # only the file changes: the results stay valid
        self.n_changed += 1
        return True


@pytest.fixture
def dlg(app, monkeypatch):
    from fea_gui.space_panel import SpaceDialog
    warnings: list[str] = []
    monkeypatch.setattr(SpaceDialog, "_warn", lambda self, text: warnings.append(text))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))
    w = FakeWin()
    d = SpaceDialog(w)
    d.warnings = warnings
    d.fake = w
    return d


def _set_values(d, approach="protoflight"):
    """Fill the metallic factors and rules through the widgets with the arbitrary test numbers."""
    d.c_approach.setCurrentIndex(d.c_approach.findData(approach))
    t = d.t_factors
    for r in range(t.rowCount()):
        key, cls = t.item(r, 0).text(), t.item(r, 1).text()
        if cls == "metallic":
            t.item(r, 3).setText(str(SC.test_factor(key, cls, approach)))
            t.item(r, 5).setText("test revision")
            t.item(r, 6).setText("test clause")
    for r in range(d.t_rules.rowCount()):
        if d.t_rules.item(r, 1).text() != "metallic":
            continue
        chk = d.t_rules.item(r, 0).text()
        d.t_rules.item(r, 4).setText(", ".join(SC.TEST_RULES[("metallic", chk)]))
        cb = d.t_rules.cellWidget(r, 3)
        cb.setCurrentIndex(cb.findData("von_mises"))
        cb = d.t_rules.cellWidget(r, 5)
        cb.setCurrentIndex(cb.findData("same_as_mechanical"))
        d.t_rules.item(r, 7).setText("test clause")
        d.t_rules.item(r, 8).setText("test revision")


def test_template_edit_confirm_and_withdraw(dlg):
    d = dlg
    assert d.rs() is None and not d.b_confirm.isEnabled()
    d.new_template()
    rs = d.rs()
    assert rs is not None and len(rs.rules) == 5 and d.t_factors.rowCount() == 0         # no approach chosen: nothing to edit
    _set_values(d)
    assert d.t_factors.rowCount() == 28 and rs.entry("FOSY", "metallic").value == SC.test_factor("FOSY", "metallic", "protoflight")
    assert rs.rule("yield", "metallic").factors == SC.TEST_RULES[("metallic", "yield")]
    assert rs.rule("yield", "metallic").stress_measure == "von_mises" and d.fake.n_changed > 0
    d.show_all.setChecked(True)
    assert d.t_factors.rowCount() == 84
    d.show_all.setChecked(False)
    # a wrong value is refused and the cell shows the old one
    old = rs.factors[0].value
    d.t_factors.item(0, 3).setText("-1")
    assert d.warnings and rs.factors[0].value == old
    d.warnings.clear()
    d.t_factors.item(0, 3).setText("abc")
    assert d.warnings and rs.factors[0].value == old
    d.warnings.clear()
    # confirmation: a name is required, problems are listed
    d.confirm_set()
    assert d.warnings and "name of the person" in d.warnings[0] and not rs.is_confirmed()
    d.warnings.clear()
    d.by.setText("Test Engineer")
    d.confirm_set()
    assert rs.is_confirmed() and "Confirmed" in d.l_conf.text() and d.b_withdraw.isEnabled()
    # an edit withdraws it
    d.t_factors.item(0, 7).setText("changed note")
    assert not rs.is_confirmed() and "NOT confirmed" in d.l_conf.text()
    d.confirm_set()
    assert rs.is_confirmed()
    d.withdraw_set()
    assert not rs.confirmed and not rs.is_confirmed()


def test_incomplete_set_cannot_be_confirmed(dlg):
    d = dlg
    d.new_template()
    d.by.setText("Test Engineer")
    d.confirm_set()
    assert d.warnings and not d.rs().is_confirmed()


def test_allowables_table(dlg):
    d = dlg
    t = d.t_allow
    assert t.rowCount() == 1 and "not entered" in t.item(0, 8).text()
    t.item(0, 1).setText("B")
    t.item(0, 2).setText("276")
    t.item(0, 3).setText("310,5")                                        # decimal comma accepted
    t.item(0, 7).setText("test sheet")
    mat = d.study.materials["TestAlloy"]
    rec = RQ.allowable_of(mat)
    assert rec.Fy == 276.0 and rec.Fu == 310.5 and rec.basis == "B" and not rec.is_confirmed()
    d.confirm_allowable()                                                # nothing selected
    assert d.warnings
    d.warnings.clear()
    t.setCurrentCell(0, 0)
    d.confirm_allowable()                                                # no name
    assert d.warnings and not RQ.allowable_of(mat).is_confirmed()
    d.warnings.clear()
    d.by.setText("Test Engineer")
    d.confirm_allowable()
    assert RQ.allowable_of(mat).is_verified() and "confirmed by Test Engineer" in t.item(0, 8).text()
    t.item(0, 2).setText("280")                                          # an edit withdraws the confirmation
    assert not RQ.allowable_of(mat).is_confirmed() and "NOT confirmed" in t.item(0, 8).text()
    t.item(0, 2).setText("x")
    assert d.warnings
    d.warnings.clear()
    for c in (1, 2, 3, 7):                                               # all cells cleared: the record disappears
        t.item(0, c).setText("")
    assert mat.allowables is None


def test_margins_tab(dlg):
    d = dlg
    d.new_template()
    _set_values(d)
    d.by.setText("Test Engineer")
    d.confirm_set()
    d.t_allow.item(0, 1).setText("B")
    d.t_allow.item(0, 2).setText(str(SC.TEST_FY))
    d.t_allow.item(0, 3).setText(str(SC.TEST_FU))
    d.t_allow.item(0, 7).setText("test sheet")
    d.t_allow.setCurrentCell(0, 0)
    d.confirm_allowable()
    d.compute()
    s = d.summary
    assert s is not None and s.verdict == "PASS" and "PASS" in d.l_verdict.text()
    assert d.t_mos.rowCount() >= 2 and d.t_sing.rowCount() == 1
    assert "NOT JUSTIFIED" in d.t_sing.item(0, 6).toolTip() and "Open items" in d.l_open.text() and "justified" in d.l_open.text()
    d.t_mos.selectRow(0)
    assert d.fake.marks and d.fake.marks[-1][2] == "space_mos"
    d.t_sing.selectRow(0)
    assert d.fake.marks[-1][0] == s.singularities[0].location
    d.by.setText("Test Engineer")
    d.t_sing.item(0, 6).setText("accepted: rigid clamp of the test")
    sp = d.study.space
    assert len(sp.justifications) == 1 and sp.justifications[0].author == "Test Engineer"
    assert s.signoff_ready and "ready for sign-off" in d.l_open.text()
    d.t_sing.item(0, 6).setText("")                                      # an empty note removes the justification
    assert not sp.justifications and not s.signoff_ready
    # structure class of a part
    assert d.t_parts.rowCount() == 1
    cb = d.t_parts.cellWidget(0, 1)
    cb.setCurrentIndex(cb.findData("composite"))
    assert sp.part_classes == {"Block": "composite"}
    cb.setCurrentIndex(cb.findData("metallic"))
    assert sp.part_classes == {}


def test_compute_needs_results(app, monkeypatch):
    from fea_gui.space_panel import SpaceDialog
    warnings: list[str] = []
    monkeypatch.setattr(SpaceDialog, "_warn", lambda self, text: warnings.append(text))
    d = SpaceDialog(FakeWin(solved=False))
    d.compute()
    assert warnings and d.summary is None


def test_import_export(dlg, tmp_path, monkeypatch):
    d = dlg
    d.new_template()
    _set_values(d)
    d.by.setText("Test Engineer")
    d.confirm_set()
    path = tmp_path / "rs.json"
    csv_path = tmp_path / "factors.csv"
    FD = QtWidgets.QFileDialog
    monkeypatch.setattr(FD, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    d.export_set()
    monkeypatch.setattr(FD, "getSaveFileName", staticmethod(lambda *a, **k: (str(csv_path), "")))
    d.export_set()
    assert path.is_file() and csv_path.read_text(encoding="utf-8").startswith("key,structure_class")
    d.study.space.requirement_set = None
    d.refresh()
    monkeypatch.setattr(FD, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    d.import_set()
    rs = d.rs()
    assert rs is not None and rs.entry("FOSY", "metallic").value is not None and not rs.is_confirmed()      # imported: not confirmed
    bad = Path(tmp_path / "bad.json")
    bad.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(FD, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
    d.import_set()
    assert d.warnings and "could not be imported" in d.warnings[-1]
    monkeypatch.setattr(FD, "getOpenFileName", staticmethod(lambda *a, **k: (r"\\server\share\x.json", "")))
    d.warnings.clear()
    d.import_set()
    assert d.warnings and "UNC" in d.warnings[0]


def test_untrusted_text_stays_plain(dlg):
    """Names that come from a requirement set (a foreign file) must not reach a rich-text label unescaped."""
    d = dlg
    d.new_template()
    _set_values(d)
    evil = '<img src="//host/x.png">'
    d.by.setText(evil)
    d.confirm_set()
    assert d.rs().is_confirmed() and "<img" not in d.l_conf.text() and "&lt;img" in d.l_conf.text()
    d.rs().factors[0].key = evil
    d.update_status()
    assert "<img" not in d.l_issues.text()


def test_no_open_items_message_only_when_ready_for_sign_off(dlg):
    d = dlg
    d.new_template()
    _set_values(d)
    d.by.setText("Test Engineer")
    d.confirm_set()
    mat = d.study.materials[d.study.default_material or ""]
    SC.confirm_test_allowables(d.study, fy=100.0, fu=110.0)             # allowables below the stress: a FAIL without any other reason
    assert mat.allowables is not None
    d.refresh()
    d.compute()
    s = d.summary
    assert s is not None and s.verdict == "FAIL"
    assert "ready for sign-off" not in d.l_open.text() and "negative" in d.l_open.text()
    SC.confirm_test_allowables(d.study)
    d.refresh()
    d.compute()
    assert d.summary is not None and d.summary.verdict == "PASS" and "PASS – not ready for sign-off" in d.l_verdict.text()
    assert "ready for sign-off" not in d.l_open.text().replace("not ready for sign-off", "")


def test_an_edit_clears_the_stale_summary(dlg):
    d = dlg
    d.new_template()
    _set_values(d)
    d.by.setText("Test Engineer")
    d.confirm_set()
    SC.confirm_test_allowables(d.study)
    d.refresh()
    d.compute()
    assert d.summary is not None and d.t_mos.rowCount() > 0
    row = d.t_factors.item(0, 3)
    row.setText(str(float(row.text()) + 0.25))                          # an edit of a factor
    assert d.summary is None and d.t_mos.rowCount() == 0 and "compute the margins again" in d.l_verdict.text()
    d.compute()
    assert d.summary is not None and d.summary.verdict == "UNCONFIRMED_REQUIREMENTS"
    n = d.t_mos.rowCount()
    d.by.setText("Test Engineer")
    sg = d.summary.singularities
    if sg:                                                              # a note of a singularity moves no margin: the table stays
        d.t_sing.item(0, 6).setText("accepted")
        assert d.summary is not None and d.t_mos.rowCount() == n


def test_tooltips_built_from_file_strings_are_escaped(dlg):
    """A material name from a foreign file reaches the reasons of a margin row and so its tooltip: Qt takes a tooltip that looks like HTML as rich text."""
    d = dlg
    d.new_template()
    _set_values(d)
    d.by.setText("Test Engineer")
    d.confirm_set()
    evil = '<img src="//host/x.png">'
    d.study.materials[d.study.default_material or ""].name = evil               # allowables not entered: the reason names the material
    d.compute()
    tips = [d.t_mos.item(r, c).toolTip() for r in range(d.t_mos.rowCount()) for c in (3, 8)]
    assert any("&lt;img" in t for t in tips) and not any("<img" in t for t in tips)

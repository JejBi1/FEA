"""F14 S4: the tab of the space dialog (fea_gui.space_s4_panel) with a minimal window (Qt offscreen, no 3D view, no OCCT): the limits table writes rows of the
requirement set, the calculation pages show verdicts and never PASS for typed numbers, the thread pages run with a stubbed modal basis."""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

import verification.cases  # noqa: E402, F401  (first the cases)
from fea_core.space import s4_run, s4_store  # noqa: E402
from tests.s4_helpers import sdof_interface  # noqa: E402
from verification import space_cases as SC  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeWin(QtWidgets.QWidget):
    def __init__(self, tmp: Path):
        super().__init__()
        mesh = SC.box_tet_mesh()
        self.study = SC.block_study()
        self.results = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA)])
        self.parts = []
        self._workdir = tmp
        self.logs: list[str] = []
        self.marks: list = []
        self.aux: list = []

    def mark_changed(self, level="results", refresh=None):
        self.marks.append(level)
        return True

    def changed(self, invalidate_results=True, invalidate_mesh=False):
        self.marks.append("changed")

    def valid_mesh(self):
        return None

    def log_msg(self, text):
        self.logs.append(text)

    def register_aux(self, t):
        self.aux.append(t)

    def unregister_aux(self, t):
        self.aux.remove(t)


@pytest.fixture
def dlg(app, monkeypatch, tmp_path):
    from fea_gui.space_panel import SpaceDialog
    warnings: list[str] = []
    monkeypatch.setattr(SpaceDialog, "_warn", lambda self, text: warnings.append(text))
    from fea_gui import space_s4_panel as P
    monkeypatch.setattr(P.S4Panel, "_warn", lambda self, text: warnings.append(text))
    w = FakeWin(tmp_path)
    d = SpaceDialog(w)
    d.warnings = warnings
    d.fake = w
    return d


PCB_LIMITS = {"pcb_steinberg_const": 0.00022, "pcb_amplitude_factor": 1.0, "pcb_design_factor": 1.25, "pcb_fatigue_exponent": 6.4, "pcb_ref_cycles": 2e7, "pcb_damage_limit": 1.0,
              "pcb_n_sigma": 3.0, "pcb_component_const": 1.0, "pcb_position_factor_min": 0.25}


def _wait(app, dlg, seconds: float = 60.0) -> None:
    t0 = time.time()
    while dlg.s4._thread is not None and time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert dlg.s4._thread is None


def test_tab_exists_and_the_dialog_refreshes_it(dlg):
    assert dlg.s4 is not None and dlg.tabs.indexOf(dlg.s4) >= 0
    assert [dlg.s4.tabs.tabText(i) for i in range(dlg.s4.tabs.count())] == ["Interface forces", "Notching", "Bolts (ECSS)", "PCB (Steinberg)", "CubeSat rails", "Limits"]
    dlg.refresh()                                             # reloading the dialog reloads the tab
    assert dlg.s4.c_bolt_mat.count() >= 2                    # (none) + the materials of the study


def test_limits_page_adds_rows_and_edits_the_requirement_set(dlg):
    s4 = dlg.s4
    dlg.new_template()
    assert dlg.rs() is not None and not dlg.rs().limits
    s4.add_limit_rows()
    keys = [s4.t_limits.item(r, 0).text() for r in range(s4.t_limits.rowCount())]
    assert "notch_min_width_oct" in keys and "pcb_steinberg_const" in keys and "bolt_slip_mu_min" in keys
    r = keys.index("pcb_steinberg_const")
    s4.t_limits.item(r, 3).setText("0,00022")                 # a decimal comma
    s4.t_limits.item(r, 2).setText("in")
    s4.t_limits.item(r, 4).setText("test document")
    x = next(y for y in dlg.rs().limits if y.key == "pcb_steinberg_const")
    assert x.value == pytest.approx(0.00022) and x.unit == "in" and x.document == "test document"
    s4.t_limits.item(r, 3).setText("abc")                     # not a number: refused, the value stays
    assert dlg.warnings and x.value == pytest.approx(0.00022)
    # the confirmation is withdrawn by an edit: confirm FIRST (a complete set: approach, a rule, every row with its document / revision / clause), then edit, then assert
    from fea_core.space import requirements as RQ
    dlg.study.space.requirement_set = SC.make_requirement_set("protoflight", confirmed=False)
    s4.add_limit_rows()
    for rr in range(s4.t_limits.rowCount()):
        for c, txt in ((3, "0,00022" if s4.t_limits.item(rr, 0).text() == "pcb_steinberg_const" else "1"), (4, "test document"), (5, "test revision"), (6, "test clause")):
            s4.t_limits.item(rr, c).setText(txt)
    RQ.confirm(dlg.rs(), "test engineer", "2026-01-01", scope=dlg.study.space)
    assert dlg.rs().is_confirmed() and dlg.study.space.requirement_confirmed()
    x = next(y for y in dlg.rs().limits if y.key == "pcb_steinberg_const")
    s4.t_limits.item(next(i for i in range(s4.t_limits.rowCount()) if s4.t_limits.item(i, 0).text() == "pcb_steinberg_const"), 3).setText("0,00033")      # an edit of a confirmed set
    assert not dlg.rs().is_confirmed() and not dlg.study.space.requirement_confirmed()
    assert x.value == pytest.approx(0.00033)


def test_limits_page_shows_an_obsolete_row_and_the_thermal_factor_row(dlg):
    from fea_core.space.model import LimitEntry
    s4 = dlg.s4
    dlg.new_template()
    s4.add_limit_rows()
    dlg.rs().limits.append(LimitEntry(key="bolt_combined_exponent", value=2.0, unit="1", document="d", revision="r", clause="c"))
    s4.fill_limits()
    rows = {s4.t_limits.item(r, 0).text(): s4.t_limits.item(r, 1).text() for r in range(s4.t_limits.rowCount())}
    assert "bolt_thermal_factor" in rows and "obsolete" in rows["bolt_combined_exponent"]


def test_bolt_page_never_passes_typed_numbers(dlg):
    s4 = dlg.s4
    for k, v in (("d_mm", "6"), ("pitch_mm", "1"), ("A_s_mm2", "20,1"), ("k_b", "350000"), ("k_c", "400000"), ("F_nom", "6000"), ("embedding_um", "6"), ("alpha_bolt", "11.7e-6"),
                 ("alpha_c", "23.4e-6"), ("L_grip_mm", "12"), ("T_ref", "20"), ("T_min", "-40"), ("T_max", "80"), ("F_ax", "3000"), ("F_sh", "800")):
        s4.f_bolt[k].setText(v)
    s4.run_bolt()
    p = s4.page_bolts
    rows = {p.t_main.item(r, 0).text(): p.t_main.item(r, 6).text() for r in range(p.t_main.rowCount())}
    assert {"gapping", "slip", "tensile_yield"} <= set(rows) and "PASS" not in rows.values()
    assert "UNCONFIRMED" in p.l_verdict.text() and "ECSS-E-HB-32-23A" in p.l_notes.text()
    assert "bolts" in s4_store.get(dlg.study) and dlg.fake.marks[-1] == "none"
    notes = p.l_notes.text()                                   # the deviations of the program are listed on the page (the panel docstring says so), no code name in the text
    assert "Deviations and limits of the program" in notes and "concentric joints only" in notes and "DEVIATIONS" not in notes
    assert "F_nom_min" in s4.f_bolt
    s4.f_bolt["k_b"].setText("zero")
    s4.run_bolt()
    assert any("number is expected" in w for w in dlg.warnings)


def test_pcb_page(app, dlg):
    s4 = dlg.s4
    for k, v in (("name", "U1"), ("B_mm", "127"), ("h_mm", "1.575"), ("L_mm", "25.4"), ("r", "1"), ("r_source", "drawing D-1"), ("z_rms_mm", "0.05"), ("z_source", "report R-1"), ("f_n_Hz", "100"), ("duration_h", "1")):
        s4.f_pcb[k].setText(v)
    s4.run_pcb()
    _wait(app, dlg)
    p = s4.page_pcb
    assert p.t_main.rowCount() == 9 and "UNCONFIRMED" in p.l_verdict.text()                    # no constant in the requirement set: no allowable, no verdict
    assert "pcb" in s4_store.get(dlg.study)
    # with the requirement set carrying the constants and confirmed, the page may give a verdict
    from tests.s4_helpers import confirmed_space
    dlg.study.space = confirmed_space(PCB_LIMITS, units={"pcb_steinberg_const": "in"})
    s4.run_pcb()
    _wait(app, dlg)
    assert p.l_verdict.text().startswith(("PASS", "FAIL"))
    # a typed component constant is used but never confirmed: no PASS even with the same numbers
    s4.f_pcb["C"].setText("1")
    s4.run_pcb()
    _wait(app, dlg)
    assert p.l_verdict.text().startswith("UNCONFIRMED")
    # a typed r or RMS displacement without its source is never confirmed (re-check I1), also through the page
    s4.f_pcb["C"].setText("")
    s4.f_pcb["r_source"].setText("")
    s4.run_pcb()
    _wait(app, dlg)
    assert p.l_verdict.text().startswith("UNCONFIRMED") and "r_source" in p.l_verdict.text()


def test_rails_page(dlg):
    s4 = dlg.s4
    mesh_nodes = np.array([[0, 42, 42], [100, 50, 50]], float)
    mesh = SimpleNamespace(part_names=["R1"], tets=np.array([[0, 1, 0, 1]]), tet_part=np.array([0]), nodes=mesh_nodes)
    U = np.zeros((2, 3))
    U[1, 1] = 0.25
    from fea_core.space.model import SpaceSettings
    dlg.study.space = dlg.study.space or SpaceSettings()
    dlg.study.space.extra["rails"] = {"axis": "x", "axial": "none", "rail_parts": ["R1"],
                                      "supports": [{"selection": "Rail R1 +y", "part": "R1", "kind": "wall", "axis": "y", "sign": 1, "component": 1}]}
    dlg.fake.results = SimpleNamespace(steps=[SimpleNamespace(name="Launch", reactions={"Rail R1 +y": np.array([0, -400.0, 0])}, U=U)], mesh=mesh)
    s4.e_rail_parts.setText("R1")
    s4.f_rail["reaction_limit_N"].setText("800")
    s4.f_rail["lateral_deflection_limit_mm"].setText("0,5")
    s4.check_rails()
    p = s4.page_rails
    assert p.t_main.rowCount() == 2 and "UNCONFIRMED" in p.l_verdict.text()
    assert p.t_main.item(0, 5).text() in ("1", "1,000")
    s4.apply_rails()                                           # no parts in the window: reported, nothing written
    assert dlg.warnings


def test_interface_and_notch_pages_with_a_stubbed_modal_basis(app, dlg, monkeypatch):
    s4 = dlg.s4
    mi = sdof_interface()
    monkeypatch.setattr(s4_run, "get_interface", lambda *a, **k: (mi, False))
    s4.e_if_sine.setPlainText("10 1\n3000 1")
    s4.e_if_asd.setPlainText("20 0,01\n2000 0,01")
    s4.e_if_zeta.setText("0,02")
    s4.c_if_dir.setCurrentIndex(s4.c_if_dir.findData("x"))
    s4.run_interface()
    _wait(app, dlg)
    p = s4.page_if
    assert p.t_main.rowCount() == 1 and s4.t_modal.rowCount() == 1 and not dlg.warnings
    assert "interface" in s4_store.get(dlg.study)
    peak = s4_store.get(dlg.study)["interface"]["directions"]["x"]["sine"]["F_peak_max_N"]
    n = s4.page_notch
    s4.e_n_profile.setPlainText("10 1\n3000 1")
    s4.e_n_limit.setText(str(0.5 * peak))
    s4.e_n_j.setText("1")
    s4.e_n_w.setText("0,1")
    s4.e_n_floor.setText("0,1")
    s4.run_notch()
    _wait(app, dlg)
    assert n.t_main.rowCount() >= 1 and "UNCONFIRMED" in n.l_verdict.text() and s4.b_n_export.isEnabled()
    assert "notch" in s4_store.get(dlg.study)
    # a bad profile line is refused before anything runs
    s4.e_n_profile.setPlainText("10 1\nabc")
    s4.run_notch()
    assert any("every line" in w for w in dlg.warnings)


# ------------------------------------------------------------------ review finding C2: the pages never show a verdict of an earlier state
def _pcb_verdict(app, dlg):
    from tests.s4_helpers import confirmed_space
    s4 = dlg.s4
    dlg.study.space = confirmed_space(PCB_LIMITS, units={"pcb_steinberg_const": "in"})
    for k, v in (("name", "U1"), ("B_mm", "127"), ("h_mm", "1.575"), ("L_mm", "25.4"), ("r", "1"), ("r_source", "drawing D-1"), ("z_rms_mm", "0.01"), ("z_source", "report R-1"), ("f_n_Hz", "100"), ("duration_h", "1")):
        s4.f_pcb[k].setText(v)
    s4.run_pcb()
    _wait(app, dlg)
    assert s4.page_pcb.l_verdict.text().startswith(("PASS", "FAIL")) and s4.page_pcb.t_main.rowCount() == 9
    return s4


def test_a_limit_edit_clears_the_verdicts_of_the_pages(app, dlg):
    s4 = _pcb_verdict(app, dlg)
    s4.fill_limits()
    keys = [s4.t_limits.item(r, 0).text() for r in range(s4.t_limits.rowCount())]
    s4.t_limits.item(keys.index("pcb_design_factor"), 3).setText("99")              # the user changes the design factor
    p = s4.page_pcb
    assert p.l_verdict.text() == "" and p.t_main.rowCount() == 0 and p.l_notes.text() == ""
    assert not dlg.study.space.requirement_confirmed()


def test_reloading_the_dialog_clears_the_verdicts_of_the_pages(app, dlg):
    s4 = _pcb_verdict(app, dlg)
    dlg.fake.study = SC.block_study()                                              # another project
    dlg.refresh()
    for page in (s4.page_if, s4.page_notch, s4.page_bolts, s4.page_pcb, s4.page_rails):
        assert page.l_verdict.text() == "" and page.l_notes.text() == ""
    assert s4.page_pcb.t_main.rowCount() == 0 and not s4.b_n_export.isEnabled()

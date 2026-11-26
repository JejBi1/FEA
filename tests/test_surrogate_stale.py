"""A surrogate belongs to the study it was fitted for (F15-S4 review fix I3): the study fingerprint, the stale mark in predict / report / save,
the clearing when a new study opens or a new parametric study runs (MCP and the GUI bridge), `refit_with` refusing rows of a changed study."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from fea_core.project import load_project, load_surrogate_file, save_project
from fea_core.study import Force, LoadCase, Material, Pressure, Study
from fea_core.surrogate import SurrogateError, fit_surrogate, lhs_design, study_fingerprint
from fea_core.surrogate.fit import SURROGATE_KPIS
from fea_core.sweep import scale_study_loads


def make_study() -> Study:
    st = Study(name="P", geometry="")
    st.materials["S"] = Material(name="S", E=210000.0, nu=0.3, rho=7.85e-9, Re=235.0)
    st.materials["A"] = Material(name="A", E=70000.0, nu=0.33, rho=2.7e-9, Re=160.0)
    st.part_materials["P1"] = "S"
    st.load_cases += [LoadCase("L1", [Force("f", [0, 0, -100.0])]), LoadCase("L2", [Pressure("p", 2.0)])]
    return st


def model(study: Study, keys=("a", "b")):
    """A smooth response over 60 points (a seed for which the model is judged reliable)."""
    X = lhs_design([(10.0, 60.0), (0.0, 40.0)], 60, seed=1)
    y = 80.0 * np.exp(-X[:, 0] / 60.0) * (1.0 + 0.25 * np.sin(X[:, 1] / 12.0)) + 10.0
    pts = [{"params": {"a": float(x[0]), "b": float(x[1])}, "max_stress": float(v), "converged": True} for x, v in zip(X, y)]
    sm = fit_surrogate(pts, keys=list(keys), backend="rbf", seed=1, fingerprint=study_fingerprint(study, list(keys)),
                       sweep={"mode": "lhs", "seed": 1, "params": [{"kind": "mesh_size", "bounds": [1, 2]}]})
    return sm


def test_fingerprint_changes_with_the_study_and_ignores_names_and_post_processing():
    st = make_study()
    fp = study_fingerprint(st, [])
    assert fp == study_fingerprint(copy.deepcopy(st), []) and len(fp) == 64
    other = copy.deepcopy(st)
    other.name = "renamed"
    other.hidden_parts = ["P1"]
    assert study_fingerprint(other, []) == fp                                      # not inputs of the calculation
    for edit in (lambda s: s.load_cases[0].loads[0].__setattr__("vector", [0, 0, -101.0]), lambda s: s.mesh.__setattr__("size", 3.3),
                 lambda s: s.materials["S"].__setattr__("E", 200000.0), lambda s: s.part_materials.__setitem__("P1", "A"),
                 lambda s: s.load_cases.pop()):
        changed = copy.deepcopy(st)
        edit(changed)
        assert study_fingerprint(changed, []) != fp


def test_swept_parameters_are_neutralised_so_taking_over_a_point_is_not_a_change():
    st = make_study()
    cases = [("mesh_size", lambda s: s.mesh.__setattr__("size", 7.7)),
             ("prop:S:E", lambda s: s.materials["S"].__setattr__("E", 190000.0)),
             ("material:P1", lambda s: s.part_materials.__setitem__("P1", "A")),
             ("load_factor", lambda s: scale_study_loads(s, 2.5)),
             ("case_factor:L1", lambda s: scale_study_loads(s, 1.7, ["L1"]))]
    for key, edit in cases:
        moved = copy.deepcopy(st)
        edit(moved)
        assert study_fingerprint(moved, [key]) == study_fingerprint(st, [key]), key
        assert study_fingerprint(moved, []) != study_fingerprint(st, []), key       # without the key the same edit IS a change
    moved = copy.deepcopy(st)
    scale_study_loads(moved, 2.0, ["L1"])
    assert study_fingerprint(moved, ["case_factor:L2"]) != study_fingerprint(st, ["case_factor:L2"])     # another case: still seen
    moved = copy.deepcopy(st)
    moved.load_cases.pop()                                                          # the structure of the loads is compared even when swept
    assert study_fingerprint(moved, ["load_factor"]) != study_fingerprint(st, ["load_factor"])


def test_the_geometry_file_is_part_of_the_fingerprint(tmp_path):
    st = make_study()
    a, b = tmp_path / "a.step", tmp_path / "b.step"
    a.write_bytes(b"ISO-10303-21; model A")
    b.write_bytes(b"ISO-10303-21; model B")
    assert study_fingerprint(st, [], a) == study_fingerprint(st, [], a) != study_fingerprint(st, [], b)
    assert study_fingerprint(st, [], a) != study_fingerprint(st, [], None)


def test_predict_report_and_save_mark_a_stale_model(tmp_path):
    st = make_study()
    sm = model(st)
    assert sm.check_study(st) == "" and not sm.predict({"a": 35.0, "b": 20.0})["stale"] and sm.reliable
    st.load_cases[0].loads[0].vector = [0, 0, -500.0]                               # the model changed after the sweep
    why = sm.check_study(st)
    assert "changed after the surrogate was fitted" in why and sm.stale
    r = sm.predict({"a": 35.0, "b": 20.0})
    assert r["stale"] and not r["kpis"]["max_stress"]["reliable"] and any(w.startswith("STALE") for w in r["warnings"])
    assert not sm.reliable and sm.describe()["stale"]
    # report
    from fea_core.report.surrogate import surrogate_section

    class Doc:
        def __init__(self):
            self.paras = []

        def p(self, text="", **kw):
            self.paras.append(text)

        def table(self, *a, **kw):
            pass
    D = Doc()
    surrogate_section(D, SimpleNamespace(surrogate=sm, study=st, step_path=None))
    assert any(p.startswith("WARNING") and "changed after the surrogate" in p for p in D.paras)
    fresh = model(make_study())
    D2 = Doc()
    surrogate_section(D2, SimpleNamespace(surrogate=fresh, study=make_study(), step_path=None))
    assert not any(p.startswith("WARNING") for p in D2.paras)
    # save: the project still carries the surrogate with its ORIGINAL fingerprint, says so, and the reopened project shows it stale again
    warns: list[str] = []
    fea = tmp_path / "p.fea"
    save_project(fea, st, "", None, surrogate=sm, warnings=warns)
    assert warns and "earlier version of the study" in warns[0]
    wd = tmp_path / "wd"
    study2, _, _ = load_project(fea, wd)
    back = load_surrogate_file(wd)
    assert back is not None and back.fingerprint == sm.fingerprint
    assert back.check_study(study2) and not back.check_study(make_study())          # stale against the edited study, fresh against the original


def test_saved_with_the_same_study_it_stays_fresh(tmp_path):
    st = make_study()
    sm = model(st)
    warns: list[str] = []
    fea = tmp_path / "p.fea"
    save_project(fea, st, "", None, surrogate=sm, warnings=warns)
    assert warns == []
    wd = tmp_path / "wd"
    study2, _, _ = load_project(fea, wd)
    back = load_surrogate_file(wd)
    assert back is not None and back.check_study(study2) == ""


def test_refit_with_refuses_rows_of_a_changed_study():
    st = make_study()
    sm = model(st)
    row = {"params": {"a": 35.0, "b": 20.0}, "kpi": {k: (95.0 if k == "max_stress" else None) for k in SURROGATE_KPIS}}
    assert len(sm.refit_with([row], study_fingerprint=sm.fingerprint).rows) == 61           # the same study: accepted
    with pytest.raises(SurrogateError, match="changed after the surrogate"):
        sm.refit_with([row], study_fingerprint="0" * 64)
    st.mesh.size = 9.9
    sm.check_study(st)                                                                      # the mesh size is not swept in this model's keys
    with pytest.raises(SurrogateError, match="changed after the surrogate"):
        sm.refit_with([row])
    kept = model(make_study()).refit_with([row])
    assert kept.fingerprint and kept.sweep["mode"] == "lhs"                                 # the fingerprint and the sweep travel with the refit


def test_a_model_without_a_fingerprint_says_so_and_is_not_marked_stale():
    from fea_core.surrogate.fit import NO_FINGERPRINT
    X = lhs_design([(0, 1), (0, 1)], 30, seed=0)
    sm = fit_surrogate([{"params": {"a": float(x[0]), "b": float(x[1])}, "max_stress": float(50 + 40 * x[0]), "converged": True} for x in X], backend="rbf")
    assert NO_FINGERPRINT in sm.warnings and sm.check_study(make_study()) == ""


# ------------------------------------------------------------------ the sweep stamps the result; a new sweep clears the surrogate
def test_sweep_specification_and_the_adaptive_flag_are_part_of_the_result():
    from fea_core.sweep import DesignPoint, ParametricStudyResult, ParametricStudySpec, SweepParam
    spec = ParametricStudySpec.from_params([SweepParam("mesh_size", bounds=(1.0, 3.0))], mode="lhs", max_points=4, seed=2)
    assert spec.to_dict()["mode"] == "lhs" and spec.to_dict()["seed"] == 2
    dp = DesignPoint(index=0, params={"mesh_size": 1.0}, adaptive=True)
    assert dp.to_dict()["adaptive"] is True and "adaptive" not in DesignPoint(index=1, params={}).to_dict()
    res = ParametricStudyResult(points=[dp], sweep=spec.to_dict(), fingerprint="ab" * 32)
    assert res.sweep["mode"] == "lhs" and res.fingerprint == "ab" * 32           # stamped by run_parametric_study (the sweep) and by the caller (the fingerprint)
    # the fit takes both from the result: a design point flagged adaptive never calibrates the model
    X = lhs_design([(0, 1), (0, 1)], 30, seed=0)
    pts = [DesignPoint(index=i, params={"a": float(x[0]), "b": float(x[1])}, max_stress=float(50 + 40 * x[0] + 20 * x[1]), adaptive=bool(i % 6 == 0))
           for i, x in enumerate(X)]
    sm = fit_surrogate(pts, keys=["a", "b"], kpis=["max_stress"], backend="rbf", fingerprint=res.fingerprint, sweep=res.sweep)
    assert sm.fingerprint == res.fingerprint and sm.sweep == res.sweep
    assert not set(sm.members[()]["max_stress"].cal_ids) & {i for i in range(30) if i % 6 == 0}


def test_gui_bridge_takes_the_surrogate_and_the_sweep_of_the_session_back_into_the_window(tmp_path):
    """fea_surrogate_fit counts as read-only for the study, but it changes the surrogate: the window must get it (and lose it when a new
    sweep or a new study clears it in the session). Needs Qt for the import only – the method is called on a stand-in window."""
    pytest.importorskip("PySide6")
    from fea_core.mcp.session import Session
    from fea_gui.ipc_bridge import McpBridgeMixin
    sm = model(make_study())
    sess = Session(tmp_path / "mcp")
    win = SimpleNamespace(surrogate_model=None, param_study=None, dirty=False, update_title=lambda: None)
    sess.surrogate, sess.param_study = sm, SimpleNamespace(points=[1])
    McpBridgeMixin._mcp_sync_surrogate(win, sess)                         # type: ignore[arg-type]  # a stand-in for the window
    assert win.surrogate_model is sm and win.param_study is sess.param_study and win.dirty
    win.dirty = False
    McpBridgeMixin._mcp_sync_surrogate(win, sess)
    assert not win.dirty                                                  # nothing changed: the project is not marked
    sess.surrogate = None                                                 # fea_param_study_run / a new study cleared it
    McpBridgeMixin._mcp_sync_surrogate(win, sess)
    assert win.surrogate_model is None

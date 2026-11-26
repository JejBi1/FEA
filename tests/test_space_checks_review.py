"""Regression tests of the review of the F14 S3 wiring (the counterexample scripts of the independent review became these tests).

C1  options that weaken a check can no longer give a PASS (bounds of every number, check 1 / 3 / 6, the settings are printed next to the verdict)
I1  check 7 on supports that share nodes (two symmetry planes)        I2  check 7 on states without a net force (pure moment, uniform temperature)
I3  lost reactions / records are never judged by a stored scalar       I4  a partial re-run keeps the settings per check
I6  the confirmation (scope-aware, through which interface) reaches the run, the tool answer and the chapter
+ the minor findings of the report (stale verdict, local paths, per-check time, settings table) and of the tool (the confirm error names the missing field).
The numbers of the limits are EXAMPLE values of the tests - the program ships none.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.geometry import Part  # noqa: E402
from fea_core.mcp.session import FeaError  # noqa: E402
from fea_core.report import space as RS  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.solve import prepare_mesh, run_study  # noqa: E402
from fea_core.space import _checks_common as cc  # noqa: E402
from fea_core.space import _checks_equil as eq  # noqa: E402
from fea_core.space import model_checks as mc  # noqa: E402
from fea_core.space.model import LimitEntry, SpaceSettings  # noqa: E402
from fea_core.space.requirements import confirm  # noqa: E402
from fea_core.study import Fixed, Force, Gravity, LoadCase, Material, MeshSettings, RemoteLoad, ThermalLoad  # noqa: E402
from tests.test_space_checks import (LIMITS, LIMITS_DICT, _rslab_on_the_path_of_the_worker, analytic_mass, build_sat, needs_ccx, needs_direct, slow)  # noqa: E402, F401
from verification import space_cases as SC  # noqa: E402
from verification.cases import _study, box  # noqa: E402

EQ_LIMIT = cc.ModelCheckLimits.from_entries({"equilibrium_tol": 1e-6}, trusted=True)          # an example value of the tests, not of a standard


@pytest.fixture(scope="module")
def sat(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sat_review")
    st, parts, mesh = build_sat(wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd)


def _result(cid: str, status: str, summary: str = "ok", **kw) -> cc.CheckResult:
    n = cc.CHECK_IDS.index(cid) + 1
    return cc.CheckResult(cid, n, cc.CHECK_TITLES[cid], status, summary, kw.get("values", {}), kw.get("limits", {}), kw.get("findings", []), kw.get("notes", []), 0.5)


def _all(status: str = cc.PASS, **kw) -> list[cc.CheckResult]:
    return [_result(c, status, **kw) for c in cc.CHECK_IDS]


def _rs(names, confirmed: bool = True, via: str = ""):
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    rs.limits = [LimitEntry(k, LIMITS_DICT[k], "", "test document", "test revision", "test clause") for k in names]
    if confirmed:
        confirm(rs, "test engineer", "2026-01-01", via=via)
    return rs


def _text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------- C1: bounds of the options
@pytest.mark.parametrize("bad", [
    {"convergence_ratio": 1.0}, {"convergence_ratio": 0.8}, {"convergence_ratio": 1.01}, {"convergence_ratio": 1.29}, {"convergence_ratio": float("nan")},
    {"convergence_ratio": 50.0}, {"convergence_ratio": "1.5"}, {"convergence_max_dof": 1e12}, {"convergence_max_dof": -1}, {"convergence_max_dof": float("nan")},
    {"thermal_dT_K": 0.0}, {"thermal_dT_K": 1e9}, {"thermal_dT_K": float("inf")}, {"thermal_alpha_per_K": 0.0}, {"thermal_alpha_per_K": 1.0},
    {"thermal_dT_K": 1000.0, "thermal_alpha_per_K": 1e-3}, {"mass_budget_t": float("nan")}, {"mass_budget_t": -1.0}, {"mass_budget_t": 0.0}, {"cog_budget_mm": [1, 2]},
    {"cog_budget_mm": [1, 2, float("inf")]}, {"part_mass_budget_t": {"A": -1}}, {"density_range_t_mm3": [0.0, 1.0]}, {"density_range_t_mm3": [1.0, 0.5]},
    {"cte_range_per_K": [1e-6, float("nan")]}, {"unit_states": ["UX", "UX"]}, {"unit_states": ["XX"]}, {"solver_mass_check": "yes"}, {"include_convergence": 1},
    {"threads": True}, {"threads": 0}, {"convergence_levels": 1}, {"free_free_modes": 3}, {"interface_selections": [3]}, {"convergence_case": True}])
def test_every_option_is_bounded_and_a_weakening_value_is_an_error(bad):
    with pytest.raises(ValueError):
        mc.options_from_dict(bad)
    with pytest.raises(ValueError):
        mc.run_model_checks(SC.block_study(), None, None, None, cc.ModelCheckOptions(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in bad.items()}), workdir=Path("nowhere"))


def test_the_mcp_tool_refuses_them_with_the_name_of_the_option():
    from fea_core.mcp.tools_space import _check_options
    for bad, name in (({"convergence_ratio": 1.0}, "convergence_ratio"), ({"convergence_max_dof": 1e12}, "convergence_max_dof"), ({"thermal_dT_K": 1e9}, "thermal_dT_K"),
                      ({"solver_mass_check": "no"}, "solver_mass_check")):
        with pytest.raises(FeaError, match=name):
            _check_options(bad)
    with pytest.raises(FeaError, match="requirement set"):
        _check_options({"mass_tol_rel": 0.5})
    ok = _check_options({"convergence_ratio": mc.MIN_CONVERGENCE_RATIO, "thermal_dT_K": -50, "unit_states": ["UX"], "threads": 2})
    assert ok.convergence_ratio == mc.MIN_CONVERGENCE_RATIO and ok.unit_states == ("UX",)


def test_the_smallest_accepted_ratio_is_a_documented_numerical_setting():
    assert mc.MIN_CONVERGENCE_RATIO == 1.3 and mc.MAX_CONVERGENCE_RATIO > mc.MIN_CONVERGENCE_RATIO
    import inspect
    from fea_core.mcp import tools_space
    src = inspect.getsource(tools_space)
    assert "1.3..4" in src and "convergence_max_dof" in src and "numerical" in src
    assert "Numerical settings, NOT limits of a standard" in inspect.getsource(cc)


def test_check_6_with_a_ratio_of_one_is_skipped_never_passed(sat):
    from fea_core.space import _checks_conv as cconv
    for ratio in (1.0, 0.5, 1.05, float("nan")):
        ctx = cc.CheckContext(sat.study, sat.mesh, LIMITS, cc.ModelCheckOptions(convergence_ratio=ratio), sat.wd / "ratio", lambda s: None, parts=sat.parts)
        r = cconv.run_convergence_check(ctx)
        assert r.status == cc.SKIPPED and "ratio" in r.summary, (ratio, r.status, r.summary)
    assert mc.overall_verdict([r]) == "INCOMPLETE"


def test_check_3_with_no_unit_state_is_skipped(sat):
    from fea_core.space import _checks_static as cs
    ctx = cc.CheckContext(sat.study, sat.mesh, LIMITS, cc.ModelCheckOptions(unit_states=(), interface_selections=["launch"]), sat.wd / "u0", lambda s: None, parts=sat.parts)
    r = cs.run_unit_displacement_check(ctx)
    assert r.status == cc.SKIPPED and "no unit state" in r.summary


@needs_direct
@slow
def test_check_3_with_fewer_than_six_unit_states_is_never_a_pass(sat):
    from fea_core.space import _checks_static as cs
    ctx = cc.CheckContext(sat.study, sat.mesh, LIMITS, cc.ModelCheckOptions(unit_states=("UX", "RY", "RZ"), interface_selections=["launch"], threads=4), sat.wd / "u3", lambda s: None,
                          parts=sat.parts)
    r = cs.run_unit_displacement_check(ctx)
    assert r.status == cc.UNCONFIRMED and "3 of the 6 unit states" in r.summary and "UY" in r.summary and r.values["worst_reaction_ratio"] < 1e-8    # clean, but only half of the motions


def test_check_1_without_its_solver_cross_check_or_with_a_widened_density_range_is_never_a_full_pass(sat):
    from fea_core.space import _checks_mass as cmass
    m, c = analytic_mass()

    def run(**kw):
        opt = cc.ModelCheckOptions(mass_budget_t=m, cog_budget_mm=tuple(c), solver_mass_check=False, threads=2, **kw)
        return cmass.run_mass_check(cc.CheckContext(sat.study, sat.mesh, LIMITS, opt, sat.wd / "m1", lambda s: None, parts=sat.parts))
    off = run()
    assert off.status == cc.UNCONFIRMED and "solver_mass_check" in off.summary and "not complete" in off.findings[-1]
    wide = run(density_range_t_mm3=(1e-15, 1.0))
    assert wide.status == cc.UNCONFIRMED and any("wider than the default" in f for f in wide.findings)
    narrow = run(density_range_t_mm3=(5e-9, 1e-8))
    assert not any("wider than the default" in f for f in narrow.findings)
    assert cc.range_widened((1e-15, 1.0), cc.DEFAULT_DENSITY_RANGE) and not cc.range_widened(cc.DEFAULT_DENSITY_RANGE, cc.DEFAULT_DENSITY_RANGE)
    assert cc.range_widened((1e-9, 1e-2), cc.DEFAULT_CTE_RANGE) and not cc.range_widened((1e-7, 1e-4), cc.DEFAULT_CTE_RANGE)


def test_settings_that_differ_from_the_defaults_are_printed_next_to_the_verdict_and_in_the_report():
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"]))
    opts = {"mass_budget_t": 0.0125, "solver_mass_check": False, "unit_states": ["UX"], "convergence_ratio": 1.4, "threads": 2}
    results = mc.make_run(st, _all(), opts).results
    by = {r.id: r.options for r in results}
    assert by["mass"] == {"mass_budget_t": 0.0125, "solver_mass_check": False, "threads": 2} and by["unit_displacement"] == {"unit_states": ["UX"], "threads": 2}
    assert by["mesh_convergence"] == {"convergence_ratio": 1.4, "threads": 2} and by["equilibrium"] == {"solver_mass_check": False, "threads": 2}
    assert cc.nondefault_options(cc.ModelCheckOptions()) == {}
    d = mc.to_dict(results, st)
    assert d["settings_that_differ_from_the_defaults"]["mass"]["solver_mass_check"] is False and d["checks"][0]["options"]["mass_budget_t"] == 0.0125
    text = mc.format_text(results, st)
    assert "settings that differ from the defaults: mass_budget_t = 12.5 kg; solver_mass_check = no; threads = 2" in text
    sess = SimpleNamespace(study=st, model_checks=mc.make_run(st, results, opts))
    D = _Doc()
    RS.checks_section(D, sess)
    t = _text(D)
    for needle in ("differ from the defaults", "solver_mass_check", "12.5 kg", "convergence_ratio", "1.4", "unit_states", "UX", "Finished"):
        assert needle in t, needle
    quiet = SimpleNamespace(study=st, model_checks=mc.make_run(st, _all(), {}))
    D = _Doc()
    RS.checks_section(D, quiet)
    assert "default settings" in _text(D)


# ---------------------------------------------------------------------------------------------------------------- I4: settings per check, time per check
def test_a_partial_rerun_does_not_relabel_the_other_checks_and_every_check_has_its_time():
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"]))
    first = mc.make_run(st, _all(), {"mass_budget_t": 0.01, "interface_selections": ["launch"]})
    second = mc.make_run(st, [_result("mass", cc.FAIL, "mass off")], {"mass_budget_t": 0.02}, previous=first)
    by = {r.id: r for r in second.results}
    assert by["mass"].options == {"mass_budget_t": 0.02} and by["unit_displacement"].options == {"interface_selections": ["launch"]}
    assert by["mass"].finished and by["equilibrium"].finished
    by["equilibrium"].finished = "2020-01-01 10:00"                              # an earlier run: its time stays with it
    third = mc.make_run(st, [_result("mass", cc.PASS)], {}, previous=second)
    assert {r.id: r for r in third.results}["equilibrium"].finished == "2020-01-01 10:00" and {r.id: r for r in third.results}["mass"].options == {}
    D = _Doc()
    RS.checks_section(D, SimpleNamespace(study=st, model_checks=third))
    t = _text(D)
    assert "2020-01-01 10:00" in t and "interface_selections" in t and "launch" in t
    rows = RS._settings_rows(third.results)
    assert ["interface_selections", "launch", "3, 4"] in rows and not any(r[0] == "mass_budget_t" for r in rows)            # no budget of the mass check any more: it ran without one


# ---------------------------------------------------------------------------------------------------------------- I6: the confirmation, scope-aware, and through which interface
def test_the_limits_follow_the_scope_aware_confirmation_and_carry_the_interface():
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"], via="mcp"))
    assert mc.limits_of_study(st).value("mass_tol_rel") == LIMITS_DICT["mass_tol_rel"]
    run = mc.make_run(st, _all())
    assert run.confirmed_via == "mcp" and run.confirmed_by == "test engineer" and mc.run_is_current(run, st)
    assert mc.confirmation_info(st)["to_be_checked_by_the_user"] is True and "confirmed via mcp" in mc.format_text(run.results, st)
    st.space.default_class = "composite"                                          # the structure classes are not those that were confirmed: S1 says "not confirmed"
    assert st.space.requirement_confirmed() is False and st.space.requirement_set.is_confirmed() is True
    assert mc.limits_of_study(st).value("mass_tol_rel") is None and {r["name"]: r["state"] for r in mc.limits_overview(st)}["mass_tol_rel"] == "not confirmed"
    assert not mc.run_is_current(run, st)                                         # ... and the run made under the confirmation is out of date
    assert mc.confirmation_info(st)["confirmed"] is False
    st.space.default_class = "metallic"
    assert mc.run_is_current(run, st)
    st.space.requirement_set.confirmed_via = "gui"                                # confirmed again through the program: another basis of the verdicts
    assert not mc.run_is_current(run, st)


def test_the_chapter_prints_who_confirmed_and_how_and_warns_about_an_agents_confirmation():
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"], via="mcp"))
    D = _Doc()
    RS.checks_section(D, SimpleNamespace(study=st, model_checks=mc.make_run(st, _all())))
    t = _text(D)
    assert "Confirmed via" in t and "MCP interface (an agent)" in t and "confirmed through the MCP interface by an agent" in t
    st2 = SC.block_study()
    st2.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"], via="gui"))
    D2 = _Doc()
    RS.checks_section(D2, SimpleNamespace(study=st2, model_checks=mc.make_run(st2, _all())))
    assert "GUI (the user)" in _text(D2) and "by an agent" not in _text(D2)


# ---------------------------------------------------------------------------------------------------------------- minor findings of the report
def test_an_outdated_run_is_not_printed_as_a_green_pass():
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"]))
    run = mc.make_run(st, _all())
    st.materials[st.default_material or "Ocel S235"].E *= 1.1
    D = _Doc()
    warns = RS.checks_section(D, SimpleNamespace(study=st, model_checks=run))
    t = _text(D)
    assert "Verdict of the model: OUT OF DATE" in t and "Verdict of the model: PASS" not in t and "PASS - out of date" in t and "OUT OF DATE: the study changed" in t
    assert warns and "run them again" in warns[0]
    pars = [p for p in D.doc.paragraphs if p.text.startswith("Verdict of the model")]
    assert pars and all(r.font.color.rgb is not None and str(r.font.color.rgb) != "1B7F3B" for p in pars for r in p.runs)          # not the green of a pass


def test_local_paths_in_the_texts_of_the_checks_are_removed_at_the_source_and_in_the_chapter():
    path = "C:\\Users\\someone\\Documents\\sat.step"
    r = cc.skipped(1, f"the check could not be completed: OSError: cannot read {path}")
    assert "someone" not in r.summary and "someone" not in r.findings[0] and "<path>/sat.step" in r.summary
    j = cc.Judge(cc.ModelCheckLimits.template())
    j.violate(f"the file {path} could not be read")
    res = j.finish(1, f"re-meshed from {path}", {}, ["work folder C:\\Users\\someone\\work\\fea\\checks_1 and /home/user/x/model.step"], 0.0)
    assert "someone" not in res.summary + "".join(res.findings) + "".join(res.notes) and "/home/user" not in "".join(res.notes)
    assert cc.scrub_paths("1/K and 3 / 4, E·α·ΔT = 12 MPa, and/or") == "1/K and 3 / 4, E·α·ΔT = 12 MPa, and/or"
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"]))
    raw = _result("mass", cc.FAIL, f"summary of {path}", findings=[f"finding {path}"], notes=[f"note {path}"])         # a record that was made by other code
    D = _Doc()
    RS.checks_section(D, SimpleNamespace(study=st, model_checks=mc.make_run(st, [raw])))
    t = _text(D)
    assert "someone" not in t and "C:\\" not in t and "sat.step" in t


def test_the_numerical_settings_of_the_equilibrium_check_are_values_and_are_printed():
    assert eq.CONSISTENCY_TOL > 0 and eq.NET_LOAD_SHARE > 0
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"]))
    res = _result("equilibrium", cc.PASS, values={"worst_imbalance": 1e-13, "consistency_tolerance": eq.CONSISTENCY_TOL, "net_load_share": eq.NET_LOAD_SHARE})
    D = _Doc()
    RS.checks_section(D, SimpleNamespace(study=st, model_checks=mc.make_run(st, [res])))
    t = _text(D)
    assert "consistency_tolerance" in t and "net_load_share" in t


# ---------------------------------------------------------------------------------------------------------------- the confirm error of the tool
def test_the_confirm_error_names_the_missing_field_and_the_rules_description_lists_it(tmp_path):
    import anyio
    from mcp import Client
    from fea_core.mcp.server import build_server
    from fea_core.mcp.session import Session
    sess = Session(tmp_path / "m")
    sess.study = SC.block_study()
    out: list = []

    async def main():
        srv = build_server(sess, groups_spec="space")
        async with Client(srv, mode="legacy") as c:
            tool = next(t for t in (await c.list_tools()).tools if t.name == "fea_space_requirement_set")
            out.append(tool.description)
            await c.call_tool("fea_space_requirement_set", {"action": "template"})
            data = {"verification_approach": "analysis_only",
                    "factors": [{"key": "FOSY", "structure_class": "metallic", "approach": "analysis_only", "value": 1.1, "document": "d", "revision": "r", "clause": "c"}],
                    "rules": [{"check": "yield", "structure_class": "metallic", "allowable": "Fy", "stress_measure": "von_mises", "factors": ["FOSY"],
                               "thermal_handling": "same_as_mechanical"}]}
            await c.call_tool("fea_space_requirement_set", {"action": "set", "data": data})
            r = await c.call_tool("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"})
            out.append((bool(r.is_error), r.content[0].text))
    anyio.run(main)
    import inspect
    from fea_core.mcp import tools_space
    assert "document, revision, clause (EACH rule needs all three" in inspect.getsource(tools_space)          # the full description (the list shows its first line)
    assert out[1][0] is True and "document / revision / clause" in out[1][1] and "ALL THREE of document, revision and clause" in out[1][1]


# ---------------------------------------------------------------------------------------------------------------- check 7: the reviewer's counterexamples
L, W, H = 60.0, 10.0, 10.0


def _bar(wd: Path, name: str, dims=(L, W, H), size: float = 4.0):
    wd.mkdir(parents=True, exist_ok=True)
    length, width, height = dims
    st, parts = _study(name, wd / f"{name}.step", [Part("Bar", box(0, 0, 0, length, width, height))], MeshSettings(size=size))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.add_selection("end", find_faces(parts, x=length))
    st.add_selection("bottom", find_faces(parts, normal=[0, 0, -1]))
    st.add_selection("top", find_faces(parts, normal=[0, 0, 1]))
    m = st.materials[st.default_material]
    st.materials[st.default_material] = Material(m.name, m.E, m.nu, m.rho, Re=m.Re, Rm=m.Rm, alpha=1.2e-5)
    return st, parts


def _mesh(st, parts, wd):
    s2 = copy.deepcopy(st)
    s2.modal.enabled = True
    mesh, _ = prepare_mesh(s2, wd, wd, lambda s: None, parts=parts)
    return mesh


def _check7(st, parts, mesh, wd, results=None, limits=EQ_LIMIT):
    return mc.run_model_checks(st, mesh, results, limits, {"threads": 2}, workdir=wd / "chk", checks=["equilibrium"], parts=parts)[0]


@pytest.fixture(scope="module")
def solved_bar(tmp_path_factory):
    """One clamped bar with a tip force, four pure moments (5e3 ... 5e6 N·mm) and a uniform temperature change - solved once."""
    wd = tmp_path_factory.mktemp("bar7")
    st, parts = _bar(wd, "bar7")
    st.constraints.append(Fixed("clamp"))
    st.load_cases.append(LoadCase("Tip", [Force("end", [0, 0, -100.0])]))
    for m in (5e3, 5e4, 5e5, 5e6):
        st.load_cases.append(LoadCase(f"M {m:g}", [RemoteLoad("end", [L, W / 2, H / 2], moment=[0.0, m, 0.0])]))
    st.load_cases.append(LoadCase("Heat", [ThermalLoad(value=120.0)]))
    mesh = _mesh(st, parts, wd)
    res = run_study(copy.deepcopy(st), wd / "solved", mesh=mesh)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, results=res, wd=wd)


@needs_ccx
@slow
def test_check_7_states_without_a_net_force_are_judged_on_their_own_scale(solved_bar):
    """Pure moments (a force that is only rounding noise) and a uniform temperature change gave a hard FAIL at a limit of 1e-6 and even at the solver's own 1e-3 on the scale of the
    stored totals; the scales the solver recorded make them PASS."""
    s = solved_bar
    assert all("force_scale" in st.equilibrium and "moment_scale" in st.equilibrium for st in s.results.steps)
    r = _check7(s.study, s.parts, s.mesh, s.wd, s.results)
    assert r.status == cc.PASS, (r.summary, r.findings)
    rows = {row["state"]: row for row in r.values["states"]}
    assert set(rows) >= {"Tip", "M 5000", "M 5e+06", "Heat"} and r.values["worst_imbalance"] < 1e-9
    assert all(row["consistent_with_record"] and ("nodal" in row["scale_of"] or "resultant" in row["scale_of"]) for row in rows.values())
    assert rows["M 5e+06"]["force_scale_N"] > 1.0                                           # not the 7e-5 N noise floor of the old scale
    assert rows["M 5e+06"]["force_divisor"] == "nodal norms" and rows["Heat"]["force_divisor"] == "nodal norms" and rows["Tip"]["force_divisor"] == "resultant"
    assert r.values["consistency_tolerance"] == eq.CONSISTENCY_TOL and r.values["net_load_share"] == eq.NET_LOAD_SHARE


@needs_ccx
@slow
def test_check_7_still_detects_a_changed_reaction_with_the_solver_scales(solved_bar):
    s = solved_bar
    for fac in (1 + 1e-4, 1 + 5e-4):                                      # relative to the SOLVER's scale (the sum of the nodal reaction norms), a 1e-4 change is 1e-5 of it
        bad = copy.deepcopy(s.results)
        bad.steps[0].reactions["clamp"] = bad.steps[0].reactions["clamp"] * fac
        r = _check7(s.study, s.parts, s.mesh, s.wd, bad)
        assert r.status == cc.FAIL and any("recomputed" in f for f in r.findings), (fac, r.status, r.findings)
    bad = copy.deepcopy(s.results)
    bad.steps[0].reaction_moments["clamp"] = bad.steps[0].reaction_moments["clamp"] * (1 + 1e-4)
    assert _check7(s.study, s.parts, s.mesh, s.wd, bad).status == cc.FAIL


@needs_ccx
@slow
def test_check_7_old_records_without_scales_keep_the_resultant_scale_of_a_net_force(solved_bar):
    s = solved_bar
    old = copy.deepcopy(s.results)
    for st in old.steps:
        st.equilibrium.pop("force_scale", None)
        st.equilibrium.pop("moment_scale", None)
    r = _check7(s.study, s.parts, s.mesh, s.wd, old)
    rows = {row["state"]: row for row in r.values["states"]}
    assert rows["Tip"]["force_divisor"] == "resultant"
    assert rows["Tip"]["imbalance"] < 1e-9                                                  # a state with a force needs no solver scale (the self-equilibrated ones: see N4 of the re-check)


@needs_ccx
@slow
def test_check_7_lost_reactions_or_records_are_never_a_pass(solved_bar):
    s = solved_bar
    lost = copy.deepcopy(s.results)
    lost.steps[0].reactions = {}
    r = _check7(s.study, s.parts, s.mesh, s.wd, lost)
    assert r.status == cc.FAIL and any("missing" in f and "Tip" in f for f in r.findings), r.findings
    lost.steps[0].equilibrium = {"imbalance": 0.0}                                              # the record replaced by a perfect scalar: the same
    r = _check7(s.study, s.parts, s.mesh, s.wd, lost)
    assert r.status == cc.FAIL and any("missing" in f for f in r.findings)
    novec = copy.deepcopy(s.results)
    novec.steps[0].equilibrium = {"imbalance": 0.0}                                             # reactions there, record without residual vectors and scales
    r = _check7(s.study, s.parts, s.mesh, s.wd, novec)
    assert r.status == cc.FAIL and any("no residual vectors" in f for f in r.findings)
    nomoment = copy.deepcopy(s.results)
    nomoment.steps[0].moment_point = None
    assert _check7(s.study, s.parts, s.mesh, s.wd, nomoment).status == cc.FAIL


def test_check_7_a_state_that_cannot_be_recomputed_is_judged_from_the_vectors_of_the_record_never_from_the_scalar():
    good = {"imbalance": 1e-14, "force_residual": [0, 0, 1e-9], "moment_residual": [0, 0, 0], "force_scale": 100.0, "moment_scale": 1000.0, "deformed": True}
    assert eq.record_balance(good)["imbalance"] == pytest.approx(1e-11)
    assert eq.record_balance({"imbalance": 0.0, "deformed": True}) is None and eq.record_balance({**good, "force_scale": 0.0}) is None
    assert eq.record_balance({**good, "force_residual": [0, 0, math.nan]}) is None


@needs_ccx
@slow
def test_check_7_supports_that_share_nodes_do_not_give_a_false_fail(tmp_path):
    """A block clamped on two adjacent faces (two symmetry planes: Fixed on x = 0 and on z = 0 share an edge): the per-support reactions count the shared nodes twice, the
    total reaction of the solver's record is the state's - this was a hard FAIL of check 7 on a clean model."""
    st, parts = _bar(tmp_path, "adj", dims=(20.0, 10.0, 10.0), size=3.0)
    st.constraints += [Fixed("clamp"), Fixed("bottom")]
    st.load_cases += [LoadCase("Shear", [Force("top", [50.0, 0.0, 0.0])]), LoadCase("Pull", [Force("end", [100.0, 0.0, 30.0])]), LoadCase("G", [Gravity()])]
    mesh = _mesh(st, parts, tmp_path)
    assert eq.shared_support_selections(st, mesh, {"clamp", "bottom"}) == ["clamp", "bottom"]
    r = _check7(st, parts, mesh, tmp_path)
    assert r.status == cc.PASS, (r.summary, r.findings)
    rows = r.values["states"]
    assert len(rows) >= 3 and all(row["reaction_basis"] == "per support minus the overlap recorded by the solver" and row["consistent_with_record"] and row["imbalance"] < 1e-9
                                  for row in rows)
    assert any("hold the same node and direction" in n for n in r.notes)
    nolim = _check7(st, parts, mesh, tmp_path, limits=None)                                      # without a limit: numbers, no FAIL (the invariant 1e-3 holds)
    assert nolim.status == cc.UNCONFIRMED
    st2, parts2 = _bar(tmp_path / "two", "two", dims=(20.0, 10.0, 10.0), size=3.0)               # the same two supports on the same face: fully shared
    st2.add_selection("clamp2", find_faces(parts2, x=0))
    st2.constraints += [Fixed("clamp"), Fixed("clamp2")]
    st2.load_cases.append(LoadCase("Tip", [Force("end", [0, 0, -100.0])]))
    r2 = _check7(st2, parts2, _mesh(st2, parts2, tmp_path / "two"), tmp_path / "two")
    assert r2.status == cc.PASS, (r2.summary, r2.findings)


def test_shared_support_selections_are_found_per_dof_of_the_constraint_masks():
    """Two supports share (node, DOF) only when both hold the same direction of the same node: symmetry planes with different directions share nodes but no DOF."""
    from fea_core.study import Displacement
    nodes = {"a": np.array([1, 2]), "b": np.array([2, 3]), "c": np.array([3, 4])}
    mesh = SimpleNamespace(nodes=np.zeros((6, 3)), surface_nodes=lambda n: nodes[n])
    fixed2 = SimpleNamespace(constraints=[Fixed("a"), Fixed("b")])
    assert eq.shared_support_selections(fixed2, mesh, {"a", "b"}) == ["a", "b"] and eq.shared_support_selections(fixed2, mesh, {"a"}) == []
    planes = SimpleNamespace(constraints=[Displacement("a", ux=0.0), Displacement("b", uz=0.0), Displacement("c", uy=0.0)])        # share node 2 and 3, never the same DOF
    assert eq.shared_support_selections(planes, mesh, {"a", "b", "c"}) == []
    same = SimpleNamespace(constraints=[Displacement("a", ux=0.0), Displacement("b", ux=0.0, uz=0.0)])
    assert eq.shared_support_selections(same, mesh, {"a", "b"}) == ["a", "b"]
    apart = SimpleNamespace(constraints=[Fixed("a"), Fixed("c")])
    assert eq.shared_support_selections(apart, mesh, {"a", "c"}) == []


def test_the_solver_records_the_scales_of_its_balance():
    from fea_core.solve import equilibrium
    x = np.array([[0.0, 0, 0], [10.0, 0, 0], [0, 10, 0]])
    F = np.array([[0.0, 0, 5], [0, 0, 0], [0, 0, 0]])
    R = np.array([[0.0, 0, -5], [0, 0, 0], [0, 0, 0]])
    e = equilibrium(x, F, R, np.zeros(3), np.zeros(3))
    assert e["force_scale"] == pytest.approx(5.0) and e["moment_scale"] > 0 and e["imbalance"] == pytest.approx(0.0, abs=1e-15)
    assert json.dumps(e)

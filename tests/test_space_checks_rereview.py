"""Regression tests of the RE-CHECK of the F14 S3 wiring (the probes `rc_*.py` of the second independent review became these tests).

N1  check 7 judges a net force on the RESULTANT scale, a self-equilibrated state on the nodal-norm scale of the solver     N2  a weaker convergence setting is never a full PASS
N3  shared supports: per-support reactions are verified (overlap per DOF), lost / zeroed / scaled reactions are caught     N4  results of an older version: UNCONFIRMED, never a false FAIL
N5  an interface that names every support finds no stray support                                                           N6  the summary tooltip is never read as HTML
N7  `scrub_paths` removes the path only                                                                                    N8  a renamed state with empty reactions is not a PASS
N9  `None` options are a clean validation error                                                                            N10 `kill_mesh_workers(under=)` matches a path prefix
The numbers of the limits are EXAMPLE values of the tests - the program ships none.
"""
from __future__ import annotations

import copy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (registers the cases; space_cases imports from it)
from fea_core.mcp.session import FeaError  # noqa: E402
from fea_core.report import space as RS  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
from fea_core.solve import run_study  # noqa: E402
from fea_core.space import _checks_common as cc  # noqa: E402
from fea_core.space import _checks_equil as eq  # noqa: E402
from fea_core.space import model_checks as mc  # noqa: E402
from fea_core.space.model import SpaceSettings  # noqa: E402
from fea_core.study import Displacement, Fixed, Force, LoadCase  # noqa: E402
from tests.test_space_checks import (LIMITS, LIMITS_DICT, _rslab_on_the_path_of_the_worker, build_sat, needs_ccx, needs_direct, slow)  # noqa: E402, F401
from tests.test_space_checks_review import EQ_LIMIT, _bar, _check7, _mesh, _rs, _text, sat, solved_bar  # noqa: E402, F401
from verification import space_cases as SC  # noqa: E402

LIM4 = cc.ModelCheckLimits.from_entries({"equilibrium_tol": 1e-4}, trusted=True)          # an example value of the tests, not of a standard


def _with_error(res, rel: float, key: str = "clamp"):
    """A result as a solver with a real force imbalance of `rel` (share of the applied force) would have stored it: the reactions AND its own record carry the error."""
    bad = copy.deepcopy(res)
    s = bad.steps[0]
    r0 = np.asarray(s.reactions[key], float)
    d = r0 * rel
    s.reactions[key] = r0 + d
    e = s.equilibrium
    e["force_residual"] = (np.asarray(e["force_residual"]) + d).tolist()
    e["reaction_total"] = (np.asarray(e["reaction_total"]) + d).tolist()
    e["imbalance"] = float(max(np.linalg.norm(e["force_residual"]) / e["force_scale"], np.linalg.norm(e["moment_residual"]) / e["moment_scale"]))
    return bad


# ---------------------------------------------------------------------------------------------------------------- N1: the divisor of check 7
def test_the_divisor_is_the_resultant_for_a_net_force_and_the_nodal_norms_for_a_self_equilibrated_state():
    f, kind = eq._scale(np.array([0.0, 0.0, -100.0]), np.array([0.0, 0.0, 100.0]), 19503.0, 19503.0)
    assert (f, kind) == (pytest.approx(100.0), "resultant")                                    # not the 19 503 N of the nodal norms
    f, kind = eq._scale(np.zeros(3), np.array([0.0, 0.0, 3.0]), 0.0, 1234.0)                    # a pure moment: no applied force at all
    assert (f, kind) == (1234.0, "nodal")
    f, kind = eq._scale(np.array([1e-12, 0.0, 0.0]), np.zeros(3), 500.0, 77.0)                  # the sum of rounding noise of a temperature change
    assert (f, kind) == (77.0, "nodal")
    assert eq._scale(np.zeros(3), np.zeros(3), 0.0, None) == (None, "nodal")                    # nothing to divide by: no scale
    assert eq._imbalance(np.zeros(3), None) == 0.0 and eq._imbalance(np.array([1.0, 0, 0]), None) is None and eq._imbalance(np.array([1.0, 0, 0]), 4.0) == 0.25


def test_the_limit_text_the_values_and_the_report_name_the_divisor():
    assert "RESULTANT" in cc.LIMIT_DESCRIPTIONS["equilibrium_tol"] and "nodal" in cc.LIMIT_DESCRIPTIONS["equilibrium_tol"]
    assert "resultant" in eq.DIVISOR_RULE and "nodal norms" in eq.DIVISOR_RULE
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs(["mass_tol_rel"]))
    state = {"state": "Tip", "imbalance": 1.2e-9, "force_scale_N": 100.0, "force_divisor": "resultant", "moment_scale_Nmm": 3000.0, "moment_divisor": "resultant"}
    res = cc.CheckResult("equilibrium", 7, cc.CHECK_TITLES["equilibrium"], cc.PASS, "ok", {"states": [state], "divisor": eq.DIVISOR_RULE}, {}, [], [], 0.5)
    D = _Doc()
    RS.checks_section(D, SimpleNamespace(study=st, model_checks=mc.make_run(st, [res])))
    t = _text(D)
    assert "divisor of the relative imbalance" in t and "resultant max(|applied|, |reactions|)" in t and "Force divisor" in t and "100" in t


@pytest.fixture(scope="module", params=[6, 20, 60])
def cantilever(request, tmp_path_factory):
    """A cantilever of L/h = 6 / 20 / 60 under 100 N at the tip - the divisor of the solver (sum of the nodal norms) is 19x / 64x / 195x the load."""
    ratio = request.param
    wd = tmp_path_factory.mktemp(f"cant{ratio}")
    st, parts = _bar(wd, f"cant{ratio}", dims=(10.0 * ratio, 10.0, 10.0), size=4.0 if ratio < 60 else 5.0)
    st.constraints.append(Fixed("clamp"))
    st.load_cases.append(LoadCase("Tip", [Force("end", [0, 0, -100.0])]))
    mesh = _mesh(st, parts, wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd, results=run_study(copy.deepcopy(st), wd / "solved", mesh=mesh), ratio=ratio)


@needs_ccx
@slow
def test_n1_a_force_imbalance_of_one_percent_fails_a_limit_of_1e_4_on_every_slenderness(cantilever):
    c = cantilever
    clean = _check7(c.study, c.parts, c.mesh, c.wd, c.results, limits=LIM4)
    assert clean.status == cc.PASS, (clean.summary, clean.findings)
    row = clean.values["states"][0]
    assert row["force_divisor"] == "resultant" and row["force_scale_N"] == pytest.approx(100.0, rel=1e-3)
    assert c.results.steps[0].equilibrium["force_scale"] > 15 * row["force_scale_N"]              # the nodal-norm scale is far larger: that is the dilution
    for rel in (0.01, 0.05):
        r = _check7(c.study, c.parts, c.mesh, c.wd, _with_error(c.results, rel), limits=LIM4)
        assert r.status == cc.FAIL, (c.ratio, rel, r.status, r.summary)
        assert r.values["states"][0]["force_imbalance"] == pytest.approx(rel, rel=0.06)
        assert any("exceeds the limit" in f for f in r.findings)
    assert any("resultant" in n for n in clean.notes) and clean.values["divisor"] == eq.DIVISOR_RULE


@needs_ccx
@slow
def test_n1_pure_moments_and_a_temperature_change_still_pass_on_their_own_scale(solved_bar):
    s = solved_bar
    r = _check7(s.study, s.parts, s.mesh, s.wd, s.results, limits=EQ_LIMIT)
    assert r.status == cc.PASS, (r.summary, r.findings)
    rows = {row["state"]: row for row in r.values["states"]}
    assert rows["Tip"]["force_divisor"] == "resultant" and rows["Tip"]["moment_divisor"] == "resultant"
    assert rows["Heat"]["force_divisor"] == "nodal norms" and rows["Heat"]["moment_divisor"] == "nodal norms"
    assert rows["M 5e+06"]["force_divisor"] == "nodal norms" and rows["M 5e+06"]["moment_divisor"] == "resultant"       # no net force; the moment IS the load


# ---------------------------------------------------------------------------------------------------------------- N4: results of an older version
@needs_ccx
@slow
def test_n4_old_records_without_scales_are_unconfirmed_for_self_equilibrated_states_never_a_false_fail(solved_bar):
    s = solved_bar
    old = copy.deepcopy(s.results)
    for st in old.steps:
        st.equilibrium.pop("force_scale", None)
        st.equilibrium.pop("moment_scale", None)
    r = _check7(s.study, s.parts, s.mesh, s.wd, old, limits=EQ_LIMIT)
    assert r.status == cc.UNCONFIRMED and not any("exceeds" in f for f in r.findings), (r.status, r.summary)
    assert "recompute" in r.summary
    rows = {row["state"]: row for row in r.values["states"]}
    assert rows["Tip"]["force_divisor"] == "resultant" and rows["Tip"]["imbalance"] < 1e-9 and rows["Tip"]["scale_missing"] == []       # a net force needs no scale of the solver
    assert rows["Heat"]["scale_missing"] == ["force", "moment"] and rows["Heat"]["imbalance"] is None
    assert rows["M 5000"]["scale_missing"] == ["force"] and rows["M 5000"]["moment_divisor"] == "resultant"
    assert r.values["worst_imbalance"] < 1e-9


# ---------------------------------------------------------------------------------------------------------------- N3: shared supports
@pytest.fixture(scope="module")
def shared2(tmp_path_factory):
    """Two Fixed supports that share the edge x = 0, z = 0 (two symmetry planes)."""
    wd = tmp_path_factory.mktemp("shared2")
    st, parts = _bar(wd, "shared2", dims=(20.0, 10.0, 10.0), size=2.5)
    st.constraints += [Fixed("clamp"), Fixed("bottom")]
    st.load_cases.append(LoadCase("Pull", [Force("end", [100.0, 0.0, 30.0])]))
    mesh = _mesh(st, parts, wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd, results=run_study(copy.deepcopy(st), wd / "solved", mesh=mesh))


@pytest.fixture(scope="module")
def shared3(tmp_path_factory):
    """Three supports: two Fixed that share an edge and an x-held third one that shares the edge x = L, z = 0 with the second."""
    wd = tmp_path_factory.mktemp("shared3")
    st, parts = _bar(wd, "shared3", dims=(20.0, 10.0, 10.0), size=2.5)
    st.constraints += [Fixed("clamp"), Fixed("bottom"), Displacement("end", ux=0.0)]
    st.load_cases.append(LoadCase("Push", [Force("top", [40.0, 0.0, -30.0])]))
    mesh = _mesh(st, parts, wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd, results=run_study(copy.deepcopy(st), wd / "solved", mesh=mesh))


@pytest.fixture(scope="module")
def planes3(tmp_path_factory):
    """Three symmetry planes written as one held direction each (x = 0: ux, z = 0: uz, a side face: uy): they share nodes but no DOF."""
    wd = tmp_path_factory.mktemp("planes3")
    st, parts = _bar(wd, "planes3", dims=(20.0, 10.0, 10.0), size=2.5)
    from fea_core.selectors import find_faces
    st.add_selection("side", find_faces(parts, normal=[0, -1, 0]))
    st.constraints += [Displacement("clamp", ux=0.0), Displacement("bottom", uz=0.0), Displacement("side", uy=0.0)]
    st.load_cases.append(LoadCase("Pull", [Force("end", [100.0, 0.0, 0.0]), Force("top", [0.0, 0.0, 30.0])]))
    mesh = _mesh(st, parts, wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd, results=run_study(copy.deepcopy(st), wd / "solved", mesh=mesh))


def _chk(s, res, limits=EQ_LIMIT):
    return _check7(s.study, s.parts, s.mesh, s.wd, res, limits=limits)


@needs_ccx
@slow
def test_n3_the_solver_records_the_overlap_and_a_clean_result_passes(shared2, shared3, planes3):
    for s, shared in ((shared2, ["clamp", "bottom"]), (shared3, ["clamp", "bottom", "end"]), (planes3, [])):
        e = s.results.steps[0].equilibrium
        assert "reaction_overlap" in e and "reaction_moment_overlap" in e
        assert eq.shared_support_selections(s.study, s.mesh, set(s.results.steps[0].reactions)) == shared
        r = _chk(s, s.results)
        assert r.status == cc.PASS, (r.summary, r.findings)
        row = r.values["states"][0]
        assert row["consistent_with_record"] and row["imbalance"] < 1e-9
        assert row["reaction_basis"] == ("per support" if not shared else "per support minus the overlap recorded by the solver") and not row["reaction_fallback"]
    assert np.linalg.norm(shared2.results.steps[0].equilibrium["reaction_overlap"]) > 1e-3              # the shared edge IS counted twice (a few 1e-4 of the 100 N)
    assert np.linalg.norm(planes3.results.steps[0].equilibrium["reaction_overlap"]) < 1e-9              # different directions: nothing counted twice


@needs_ccx
@slow
@pytest.mark.parametrize("fac", [1.5, 0.0, 2.0])
def test_n3_a_scaled_or_zeroed_per_support_reaction_of_shared_supports_is_caught(shared2, shared3, fac):
    for s, key in ((shared2, "clamp"), (shared3, "bottom")):
        bad = copy.deepcopy(s.results)
        bad.steps[0].reactions[key] = bad.steps[0].reactions[key] * fac
        r = _chk(s, bad)
        assert r.status == cc.FAIL, (key, fac, r.status, r.summary)
        assert any("recomputed" in f or "exceeds" in f for f in r.findings)


@needs_ccx
@slow
def test_n3_all_reactions_zeroed_and_a_missing_support_key_are_caught(shared2, shared3):
    bad = copy.deepcopy(shared2.results)
    st = bad.steps[0]
    st.reactions = {k: np.zeros(3) for k in st.reactions}
    st.reaction_moments = {k: np.zeros(3) for k in st.reaction_moments}
    r = _chk(shared2, bad)
    assert r.status == cc.FAIL and any("exceeds" in f or "recomputed" in f for f in r.findings), (r.status, r.findings)
    for s, key in ((shared2, "bottom"), (shared3, "end")):                    # a support whose reaction is not in the result at all
        lost = copy.deepcopy(s.results)
        del lost.steps[0].reactions[key]
        r = _chk(s, lost)
        assert r.status == cc.FAIL and any("missing" in f and key in f for f in r.findings), (key, r.status, r.findings)
        lost2 = copy.deepcopy(s.results)
        lost2.steps[0].reaction_moments.pop(key, None)
        r = _chk(s, lost2)
        assert r.status == cc.FAIL and any("missing" in f and key in f for f in r.findings), (key, r.status, r.findings)


@needs_ccx
@slow
def test_n3_three_symmetry_planes_with_a_doubled_reaction_are_caught(planes3):
    bad = copy.deepcopy(planes3.results)
    bad.steps[0].reactions["clamp"] = bad.steps[0].reactions["clamp"] * 2.0
    r = _chk(planes3, bad)
    assert r.status == cc.FAIL, (r.status, r.summary)
    zero = copy.deepcopy(planes3.results)
    zero.steps[0].reactions["bottom"] = np.zeros(3)
    assert _chk(planes3, zero).status == cc.FAIL
    gone = copy.deepcopy(planes3.results)
    del gone.steps[0].reactions["bottom"]
    assert _chk(planes3, gone).status == cc.FAIL


@needs_ccx
@slow
def test_n3_a_result_without_the_recorded_overlap_falls_back_and_is_never_a_full_pass(shared2, planes3):
    old = copy.deepcopy(shared2.results)
    for st in old.steps:
        st.equilibrium.pop("reaction_overlap", None)
        st.equilibrium.pop("reaction_moment_overlap", None)
    r = _chk(shared2, old)
    assert r.status == cc.UNCONFIRMED and "cannot be verified" in r.summary, (r.status, r.summary)
    row = r.values["states"][0]
    assert row["reaction_fallback"] and "total of the solver record" in row["reaction_basis"]
    assert any("NOT verified" in n for n in r.notes)
    old.steps[0].reactions["clamp"] = old.steps[0].reactions["clamp"] * 1.5                     # the fallback cannot see this - but it must not PASS
    assert _chk(shared2, old).status != cc.PASS
    nodof = copy.deepcopy(planes3.results)                                                       # no DOF overlap: the per-support reactions are verified as before, no fallback needed
    for st in nodof.steps:
        st.equilibrium.pop("reaction_overlap", None)
        st.equilibrium.pop("reaction_moment_overlap", None)
    assert _chk(planes3, nodof).status == cc.PASS
    doubled = copy.deepcopy(nodof)
    doubled.steps[0].reactions["clamp"] = doubled.steps[0].reactions["clamp"] * 2.0
    assert _chk(planes3, doubled).status == cc.FAIL


# ---------------------------------------------------------------------------------------------------------------- N8: a renamed state
@needs_ccx
@slow
def test_n8_a_state_that_is_no_load_case_with_empty_reactions_is_not_a_pass_on_the_record_alone(solved_bar):
    s = solved_bar
    bad = copy.deepcopy(s.results)
    bad.steps[0].name = "Other name"                                                           # not a load case of the study any more
    bad.steps[0].reactions = {}
    r = _check7(s.study, s.parts, s.mesh, s.wd, bad, limits=EQ_LIMIT)
    assert r.status == cc.FAIL and any("Other name" in f and "missing" in f for f in r.findings), (r.status, r.findings)
    keep = copy.deepcopy(s.results)
    keep.steps[0].name = "Other name"                                                          # reactions present: judged from the record, as before (and not a FAIL)
    r2 = _check7(s.study, s.parts, s.mesh, s.wd, keep, limits=EQ_LIMIT)
    assert not any("missing" in f for f in r2.findings)


# ---------------------------------------------------------------------------------------------------------------- N2: weakened convergence
@needs_direct
@needs_ccx
@slow
def test_n2_a_weaker_convergence_setting_is_never_a_full_pass(tmp_path):
    """The reviewer's case on a bar: the defaults decide, fewer levels / a smaller ratio / fewer frequencies give no verdict (UNCONFIRMED with the reason) - even with a loose limit
    that the weak run would meet."""
    from verification.space_check_cases import sc6_build
    from fea_core.solve import prepare_mesh
    st, parts = sc6_build(tmp_path)
    s2 = copy.deepcopy(st)
    mesh, _ = prepare_mesh(s2, tmp_path, tmp_path, lambda s: None, parts=parts)
    lim = cc.ModelCheckLimits.from_entries({**LIMITS_DICT, "freq_convergence_tol": 0.5, "stress_convergence_tol": 10.0}, trusted=True)
    run = {}
    for tag, kw in (("default", {}), ("two_levels", {"convergence_levels": 2}), ("ratio", {"convergence_ratio": 1.3}), ("one_freq", {"convergence_frequencies": 1}),
                    ("reviewer", {"convergence_ratio": 1.3, "convergence_levels": 2, "convergence_frequencies": 1})):
        run[tag] = mc.run_model_checks(st, mesh, None, lim, {"threads": 2, **kw}, workdir=tmp_path / tag, checks=["mesh_convergence"], parts=parts)[0]
    assert run["default"].status in (cc.PASS, cc.WARN), (run["default"].status, run["default"].summary)
    needles = {"two_levels": "mesh levels", "ratio": "smaller than the default", "one_freq": "natural frequencies", "reviewer": "mesh levels"}
    for tag, needle in needles.items():
        r = run[tag]
        assert r.status == cc.UNCONFIRMED and needle in r.summary, (tag, r.status, r.summary)
        assert r.options                                                                         # the setting is printed next to the verdict
    assert all(x in run["reviewer"].summary for x in ("mesh levels", "smaller than the default", "natural frequencies"))


@needs_direct
@needs_ccx
@slow
def test_n2_the_reviewers_case_on_the_test_satellite_is_no_longer_a_pass(sat):
    """The case of the re-check: the defaults give WARN on the test satellite (a singularity suspect), and the weakest accepted setting gave a full PASS."""
    weak = {"threads": 4, "convergence_ratio": 1.3, "convergence_levels": 2, "convergence_frequencies": 1}
    r_def = mc.run_model_checks(sat.study, sat.mesh, None, LIMITS, {"threads": 4}, workdir=sat.wd / "n2_def", checks=["mesh_convergence"], parts=sat.parts)[0]
    r_weak = mc.run_model_checks(sat.study, sat.mesh, None, LIMITS, weak, workdir=sat.wd / "n2_weak", checks=["mesh_convergence"], parts=sat.parts)[0]
    assert r_def.status in (cc.PASS, cc.WARN), (r_def.status, r_def.summary)
    assert r_weak.status == cc.UNCONFIRMED, (r_weak.status, r_weak.summary)
    assert all(x in r_weak.summary for x in ("mesh levels", "smaller than the default", "natural frequencies"))
    assert mc.overall_verdict([r_weak]) == "INCOMPLETE"


def test_n2_the_bounds_still_let_a_weak_setting_run_so_that_it_is_the_status_that_stops_it():
    weak = mc.options_from_dict({"convergence_ratio": 1.3, "convergence_levels": 2, "convergence_frequencies": 1})
    d = cc.ModelCheckOptions()
    assert weak.convergence_levels < d.convergence_levels and weak.convergence_ratio < d.convergence_ratio and weak.convergence_frequencies < d.convergence_frequencies


# ---------------------------------------------------------------------------------------------------------------- N5: the interface names every support
@needs_direct
@slow
def test_n5_an_interface_that_names_every_support_cannot_find_a_stray_support(tmp_path):
    st, parts, mesh = build_sat(tmp_path)
    bad = copy.deepcopy(st)
    bad.constraints.append(Displacement("end", uy=0.0))                                         # a stray support that grounds the model
    from fea_core.space import _checks_static as cs
    both = cc.CheckContext(bad, mesh, LIMITS, cc.ModelCheckOptions(interface_selections=["launch", "end"], threads=4), tmp_path / "both", lambda s: None, parts=parts)
    r = cs.run_unit_displacement_check(both)
    assert r.status == cc.UNCONFIRMED and "every support of the study" in r.summary, (r.status, r.summary)
    assert any("no support was tested as stray grounding" in n for n in r.notes)
    one = cc.CheckContext(bad, mesh, LIMITS, cc.ModelCheckOptions(interface_selections=["launch"], threads=4), tmp_path / "one", lambda s: None, parts=parts)
    r1 = cs.run_unit_displacement_check(one)
    assert r1.status == cc.FAIL and any("'end'" in f and "not on the interface" in f for f in r1.findings)
    assert any("declared by the caller" in n for n in r1.notes)


# ---------------------------------------------------------------------------------------------------------------- N6: the tooltip
def test_n6_the_summary_tooltip_is_escaped_text_in_an_explicit_rich_text_wrapper():
    import html
    from PySide6.QtGui import Qt
    from fea_core import richtext
    from fea_gui.space_checks import rich_tip
    for evil in ("part '<img src=\"//host/x.png\">': density 7.85e-06 outside the range", "state 'A < B': imbalance 1 > 0.5", "plain text without any markup", "a &lt;b&gt; & more",
                 "O'Brien part & co", "<img src=\"\\\\host\\share\\a.png\">", "x url(//host/a.png) y ![i](//h/p.png) @import 'a'", "<style>x</style>"):
        tip = rich_tip(evil)
        inner = tip[len("<qt>"):-len("</qt>")]
        assert tip.startswith("<qt>") and tip.endswith("</qt>") and "<" not in inner and ">" not in inner
        assert Qt.mightBeRichText(tip)                                                         # always read as rich text, so the entities are decoded to the original characters
        assert html.unescape(inner) == evil                                                    # what Qt shows is the text as written
        assert richtext.safe_rich(tip) == tip                                                  # the guard of the program does not blank or change it
    assert rich_tip("a < b") == "<qt>a &lt; b</qt>"


# ---------------------------------------------------------------------------------------------------------------- N7: scrub_paths
@pytest.mark.parametrize("text,expected", [
    ("see C:\\Users\\me\\job.log - force / moment imbalance 0.2 exceeds 1e-3 N/mm", "see <path>/job.log - force / moment imbalance 0.2 exceeds 1e-3 N/mm"),
    ("Meshing failed: C:\\Users\\me\\work dir\\part_000.brep could not be read", "Meshing failed: <path>/part_000.brep could not be read"),
    ("file D:/proj/a b/model.step (STEP) and E:/x/y.inp differ; reaction 12 N/mm", "file <path>/model.step (STEP) and <path>/y.inp differ; reaction 12 N/mm"),
    ("posix /home/user/proj/model.step missing, ratio a/b and c / d", "posix <path>/model.step missing, ratio a/b and c / d"),
    ("UNC \\\\srv\\share\\a.step missing", "UNC <path>/a.step missing"),
    ("density 7.85e-09 t/mm3 is outside 1e-11-2.3e-08 t/mm3; 1/K; kg/m2; force / moment imbalance", "density 7.85e-09 t/mm3 is outside 1e-11-2.3e-08 t/mm3; 1/K; kg/m2; force / moment imbalance"),
    ("see https://www.pragtic.com/vmat.php for data", "see https://www.pragtic.com/vmat.php for data"),
])
def test_n7_scrub_paths_replaces_the_path_token_not_the_text_after_it(text, expected):
    assert cc.scrub_paths(text) == expected


# ---------------------------------------------------------------------------------------------------------------- N9: None options
@pytest.mark.parametrize("name", ["unit_states", "density_range_t_mm3", "cte_range_per_K", "cog_budget_mm"])
@pytest.mark.parametrize("bad", [None, 5, "x"])
def test_n9_a_none_or_wrong_type_sequence_option_is_a_validation_error_not_a_crash(name, bad):
    if name == "cog_budget_mm" and bad is None:
        pytest.skip("None is the default of the centre-of-mass budget")
    with pytest.raises(ValueError, match=name):
        mc.options_from_dict({name: bad})
    from fea_core.mcp.tools_space import _check_options
    with pytest.raises(FeaError, match=name):
        _check_options({name: bad})
    with pytest.raises(ValueError, match=name):
        mc.run_model_checks(SC.block_study(), None, None, None, cc.ModelCheckOptions(**{name: bad}), workdir=Path("nowhere"))


# ---------------------------------------------------------------------------------------------------------------- N10: kill_mesh_workers
def test_n10_kill_mesh_workers_under_matches_a_path_prefix_with_a_separator_boundary(tmp_path):
    pytest.importorskip("psutil")
    from fea_gui.workers import kill_mesh_workers
    root = tmp_path / "run ž 1"
    dirs = {"under": root / "checks_1764000000000" / "check_convergence" / "level 1", "sibling": root / "checks_17640000000001" / "lvl", "main": root / "mesh"}
    procs = {}
    try:
        for tag, d in dirs.items():
            d.mkdir(parents=True)
            procs[tag] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "-m", "fea_core.mesh.worker", str(d / "mesh_job.json")])
        time.sleep(1.5)
        assert kill_mesh_workers(under="") == 0 and kill_mesh_workers(under="  ") == 0               # an empty folder is no folder: it matches nothing
        assert all(p.poll() is None for p in procs.values())
        under = root / "checks_1764000000000"
        spelled = Path(str(under).upper().replace("\\", "/")) if sys.platform == "win32" else under                      # another spelling of the same folder
        assert kill_mesh_workers(under=spelled) >= 1                                                  # (the venv launcher of the dummy process counts as well)
        time.sleep(0.8)
        alive = {t: p.poll() is None for t, p in procs.items()}
        assert alive == {"under": False, "sibling": True, "main": True}, alive                       # the sibling 'checks_17640000000001' shares the TEXT prefix, not the folder
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()

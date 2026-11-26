"""F14 S3 wiring: the MCP tool `fea_space_model_checks`, the kept run, the limits overview and the Word report chapter "Model quality checks".

The core checks themselves are tested in `test_space_checks.py`. Here: the tool is in the lazy group "space", it has no way to take a tolerance from the caller (no argument,
no option, and a limit that the user did not confirm does not count), the run is kept for the report, the report prints the chapter. The numbers in the limits are EXAMPLE
values of the tests - the program ships none.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

pytest.importorskip("mcp")
pytest.importorskip("docx")
from mcp import Client  # noqa: E402

import verification.cases  # noqa: E402, F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.mcp.server import build_server  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402
from fea_core.report import space as RS  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
from fea_core.space import _checks_common as cc  # noqa: E402
from fea_core.space import model_checks as mc  # noqa: E402
from fea_core.space.model import LimitEntry, SpaceSettings  # noqa: E402
from tests.test_space_checks import (LIMITS_DICT, _rslab_on_the_path_of_the_worker, analytic_mass, build_sat)  # noqa: E402, F401  (the fixture is autouse here too)
from verification import space_cases as SC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sat(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sat_wire")
    st, parts, mesh = build_sat(wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd)


def _session(sat, tmp_path, space: SpaceSettings | None = None) -> Session:
    sess = Session(tmp_path / "mcp")
    sess.study = copy.deepcopy(sat.study)
    sess.study.space = space
    sess.mesh = sat.mesh
    sess.mesh_sig = sess.mesh_signature()
    sess._parts = sat.parts
    sess.step_path = sat.wd / "sat.step"
    return sess


def _call(sess: Session, steps, groups: str = "space"):
    """steps = [(tool, args, expect_error)] -> list of parsed answers (dict) / error texts."""
    out: list = []

    async def main():
        srv = build_server(sess, groups_spec=groups)
        async with Client(srv, mode="legacy") as c:
            for name, args, expect_err in steps:
                r = await c.call_tool(name, args)
                text = r.content[0].text
                assert bool(r.is_error) == expect_err, (name, args, text)
                out.append(text if expect_err else json.loads(text))
    anyio.run(main)
    return out


def _rs_with_limits(names, confirmed: bool):
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    rs.limits = [LimitEntry(k, LIMITS_DICT[k], "", "test document", "test revision", "test clause") for k in names]
    if confirmed:
        from fea_core.space.requirements import confirm
        confirm(rs, "test engineer", "2026-01-01")
    return rs


def _result(cid: str, status: str, summary: str = "ok", **kw) -> cc.CheckResult:
    n = cc.CHECK_IDS.index(cid) + 1
    return cc.CheckResult(cid, n, cc.CHECK_TITLES[cid], status, summary, kw.get("values", {}), kw.get("limits", {}), kw.get("findings", []), kw.get("notes", []), 0.5)


def _all_results(statuses: dict[str, str] | None = None, **kw) -> list[cc.CheckResult]:
    st = statuses or {}
    return [_result(c, st.get(c, cc.PASS), **kw) for c in cc.CHECK_IDS]


# ---------------------------------------------------------------------------------------------------------------- the tool in its group
def test_the_tool_is_in_the_lazy_group_and_has_no_way_to_pass_a_tolerance(tmp_path):
    async def main():
        srv = build_server(Session(tmp_path / "b"), groups_spec="core")
        async with Client(srv, mode="legacy") as c:
            assert not any(t.name == "fea_space_model_checks" for t in (await c.list_tools()).tools)
            r = json.loads((await c.call_tool("fea_tools_enable", {"groups": "space"})).content[0].text)
            assert "fea_space_model_checks" in r["added"]
            tool = next(t for t in (await c.list_tools()).tools if t.name == "fea_space_model_checks")
            assert set(tool.input_schema["properties"]) == {"checks", "options", "timeout_s"}      # no `limits`, no `confirmed`, no tolerance argument
    anyio.run(main)
    code = ("import sys; from fea_core.mcp.server import build_server; build_server(groups_spec='core'); "
            "print('LOADED', 'fea_core.mcp.tools_space' in sys.modules, 'fea_core.space.model_checks' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip().endswith("LOADED False False"), r.stdout


def test_a_tolerance_cannot_be_passed_as_an_option_or_an_argument(tmp_path):
    sess = Session(tmp_path / "e")
    steps = [
        ("fea_space_model_checks", {"options": {"mass_tol_rel": 0.5}}, True),                         # a limit is not an option
        ("fea_space_model_checks", {"options": {"limits": {"mass_tol_rel": {"value": 0.5, "confirmed": True}}}}, True),
        ("fea_space_model_checks", {"limits": {"mass_tol_rel": 0.5}}, True),                          # not an argument either
        ("fea_space_model_checks", {"options": {"threads": 0}}, True),
        ("fea_space_model_checks", {"options": {"threads": 1000}}, True),
        ("fea_space_model_checks", {"options": {"convergence_levels": 50}}, True),
        ("fea_space_model_checks", {"options": ["mass"]}, True),
        ("fea_space_model_checks", {"checks": ["nope"]}, True),
        ("fea_space_model_checks", {}, True),                                                          # no study open
    ]
    got = _call(sess, steps)
    assert "requirement set" in got[0] and "mass_tol_rel" in got[0]
    assert "Unknown model-check option 'limits'" in got[1]
    assert "Unknown check" in got[7] and "No study" in got[8]


def test_unconfirmed_limits_never_decide_and_a_tool_call_cannot_confirm_them(sat, tmp_path):
    names = ["mass_tol_rel", "cad_mass_tol_rel"]
    sp = SpaceSettings(requirement_set=_rs_with_limits(names, confirmed=False))
    sess = _session(sat, tmp_path, sp)
    args = {"checks": ["mass"], "options": {"mass_budget_t": analytic_mass()[0], "solver_mass_check": False, "threads": 2}}
    got = _call(sess, [
        ("fea_space_model_checks", args, False),                                                      # 0: limits entered, set NOT confirmed
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "x", "data": {"name": "y"}}, True),       # 1: no value + confirmation in one call
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),            # 2: the confirmation of the user's request
        ("fea_space_model_checks", args, False),                                                      # 3: now the limits decide
        ("fea_space_model_checks", {**args, "options": {**args["options"], "mass_tol_rel": 0.9}}, True),         # 4: a number from the caller is refused
        ("fea_space_requirement_set", {"action": "set", "data": {"limits": [{"key": "mass_tol_rel", "value": 1e-9, "document": "d", "revision": "r", "clause": "c"}]}}, False),
        ("fea_space_model_checks", args, False),                                                      # 6: edited after the confirmation: withdrawn again
    ])
    first = got[0]
    assert first["checks"][0]["status"] == "UNCONFIRMED" and first["verdict"] == "INCOMPLETE"
    assert {r["name"]: r["state"] for r in first["limits"]}["mass_tol_rel"] == "not confirmed"
    assert got[2]["requirement_set"]["is_confirmed"] is True
    second = got[3]
    assert second["checks"][0]["status"] == "UNCONFIRMED" and "solver_mass_check" in second["checks"][0]["summary"], second["checks"][0]     # the limit decides, but the solver cross-check of the
    assert {r["name"]: r["state"] for r in second["limits"]}["mass_tol_rel"] == "usable"                                                        # mass was switched off: never a full PASS
    assert second["checks"][0]["limits"]["mass_tol_rel"]["decided"] is True and "mass_tol_rel" not in second["checks"][0]["summary"].split("no verdict:")[-1]
    assert second["requirement_set"]["confirmed_via"] == "mcp" and second["requirement_set"]["to_be_checked_by_the_user"] is True and "confirmed via mcp" in second["text"]
    assert first["requirement_set"]["confirmed"] is False
    assert "Unknown model-check option" in got[4]
    assert got[6]["checks"][0]["status"] == "UNCONFIRMED"
    assert {r["name"]: r["state"] for r in got[6]["limits"]}["mass_tol_rel"] == "not confirmed"
    assert sess.model_checks is not None and sess.model_checks.results[0].id == "mass" and "Model quality checks" in second["text"]


def test_the_study_of_the_session_is_not_changed_by_the_tool(sat, tmp_path):
    sess = _session(sat, tmp_path, None)
    before = json.dumps(sess.study.to_dict(), sort_keys=True)
    _call(sess, [("fea_space_model_checks", {"checks": ["mass"], "options": {"solver_mass_check": False}}, False)])
    assert json.dumps(sess.study.to_dict(), sort_keys=True) == before


# ---------------------------------------------------------------------------------------------------------------- limits overview, digest, kept run (no solver)
def test_limits_overview_states():
    st = SC.block_study()
    assert {r["state"] for r in mc.limits_overview(st)} == {"missing"} and len(mc.limits_overview(st)) == len(cc.LIMIT_DESCRIPTIONS)
    rs = _rs_with_limits(["mass_tol_rel", "cog_tol_mm", "equilibrium_tol"], confirmed=False)
    rs.limits[1].clause = ""
    st.space = SpaceSettings(requirement_set=rs)
    by = {r["name"]: r for r in mc.limits_overview(st)}
    assert by["mass_tol_rel"]["state"] == "not confirmed" and by["mass_tol_rel"]["value"] == LIMITS_DICT["mass_tol_rel"]
    assert by["cog_tol_mm"]["state"] == "incomplete" and by["eff_mass_cutoff_hz"]["state"] == "missing"
    from fea_core.space.requirements import confirm
    confirm(rs, "test engineer", "2026-01-01")
    by = {r["name"]: r["state"] for r in mc.limits_overview(st)}
    assert by["mass_tol_rel"] == "usable" and by["equilibrium_tol"] == "usable" and by["cog_tol_mm"] == "incomplete"
    assert mc.limits_of_study(st).value("mass_tol_rel") == LIMITS_DICT["mass_tol_rel"] and mc.limits_of_study(st).value("cog_tol_mm") is None
    assert mc.limits_of_study(SC.block_study()).value("mass_tol_rel") is None


def test_the_digest_follows_the_study_and_the_usable_limits_but_not_the_justifications():
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs_with_limits(["mass_tol_rel"], confirmed=True))
    d0 = mc.study_digest(st)
    run = mc.make_run(st, _all_results())
    assert mc.run_is_current(run, st) and not mc.run_is_current(None, st)
    from fea_core.space.margins import add_justification
    add_justification(st.space, "Block", [0.0, 0.0, 0.0], "a note", cause="test", author="t", date="2026-01-02")
    assert mc.study_digest(st) == d0                                           # a justification of a singular place does not make the checks old
    st.space.requirement_set.limits[0].value = 0.5                             # the limit changed -> the confirmation is gone -> not usable any more
    assert mc.study_digest(st) != d0 and not mc.run_is_current(run, st)
    st2 = SC.block_study()
    st2.space = copy.deepcopy(st.space)
    run2 = mc.make_run(st2, _all_results())
    st2.name = "another"
    assert not mc.run_is_current(run2, st2)


def test_a_rerun_of_some_checks_keeps_the_others_of_the_same_state_and_their_own_settings():
    st = SC.block_study()
    first = mc.make_run(st, _all_results({"mass": cc.UNCONFIRMED}), {"mass_budget_t": 0.5, "threads": 2})
    again = mc.make_run(st, [_result("mass", cc.PASS)], {"solver_mass_check": False}, previous=first)
    assert [r.id for r in again.results] == list(cc.CHECK_IDS) and {r.id: r.status for r in again.results}["mass"] == cc.PASS
    by = {r.id: r for r in again.results}
    assert by["mass"].options == {"solver_mass_check": False} and by["mass"].finished                  # the re-run carries ITS settings ...
    assert by["equilibrium"].options == {"threads": 2} and by["free_free"].options == {"threads": 2}   # ... and does not relabel the others (the budget of the first run is not theirs)
    assert "mass_budget_t" not in by["mass"].options and first.results[0].options == {"mass_budget_t": 0.5, "threads": 2}
    st.materials[st.default_material or "Ocel S235"].E *= 2                    # another state of the study: nothing is taken over
    other = mc.make_run(st, [_result("mass", cc.FAIL)], None, previous=first)
    assert [r.id for r in other.results] == ["mass"]


# ---------------------------------------------------------------------------------------------------------------- the report chapter
def _text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def _sess_with_run(statuses=None, confirmed: bool = True, results=None, options=None):
    st = SC.block_study()
    st.space = SpaceSettings(requirement_set=_rs_with_limits(["mass_tol_rel", "equilibrium_tol"], confirmed=confirmed))
    res = results or _all_results(statuses, findings=["the density of part 'Block' looks 1000 times too large"], notes=["the mass of 3 parts was compared"],
                                  values={"fe_mass_kg": 12.5, "cog_error_mm": 0.25, "note_path": "C:\\Users\\x\\a.txt", "label": "ok"})
    run = mc.make_run(st, res, options or {"mass_budget_t": 0.0125, "cog_budget_mm": [1.0, 2.0, 3.0], "interface_selections": ["launch"]})
    return SimpleNamespace(study=st, model_checks=run), st


def test_the_chapter_shows_the_verdict_the_checks_the_limits_and_their_provenance():
    sess, st = _sess_with_run({"mass": cc.FAIL, "isothermal": cc.WARN, "equilibrium": cc.UNCONFIRMED})
    assert RS.has_checks(sess, st)
    D = _Doc()
    warns = RS.checks_section(D, sess)
    text = _text(D)
    assert warns == []
    for needle in ("Verdict of the model: FAIL", "FEM model quality checks", "Mass and centre of mass", "Global force and moment equilibrium", "UNCONFIRMED", "WARN",
                   "the density of part 'Block' looks 1000 times too large", "Numbers computed by the checks", "fe_mass_kg", "12.5", "Tolerances of the checks",
                   "mass_tol_rel", "confirmed - decides the check", "not entered", "test document", "test clause", "test engineer", "12.5 kg", "launch",
                   "no tolerance of a standard"):
        assert needle in text, needle
    assert "C:\\" not in text and "note_path" not in text and "a.txt" not in text          # nothing that looks like a path


def test_the_chapter_never_calls_unconfirmed_or_partial_runs_a_pass():
    sess, st = _sess_with_run({c: cc.UNCONFIRMED for c in cc.CHECK_IDS}, confirmed=False)
    D = _Doc()
    RS.checks_section(D, sess)
    text = _text(D)
    assert "Verdict of the model: UNCONFIRMED_REQUIREMENTS" in text and "not an ECSS verification result" in text and "Verdict of the model: PASS" not in text
    assert "NOT confirmed" in text and "(not confirmed)" in text
    sess2, st2 = _sess_with_run(results=[_result("mass", cc.PASS)])
    D2 = _Doc()
    RS.checks_section(D2, sess2)
    assert "Verdict of the model: INCOMPLETE" in _text(D2) and "not a verification of the model" in _text(D2)
    sess3, st3 = _sess_with_run()
    D3 = _Doc()
    RS.checks_section(D3, sess3)
    assert "Verdict of the model: PASS" in _text(D3) and "changed after these checks" not in _text(D3)


def test_a_chapter_of_an_old_state_says_so_and_another_study_has_none():
    sess, st = _sess_with_run()
    st.materials[st.default_material or "Ocel S235"].E *= 1.1
    D = _Doc()
    warns = RS.checks_section(D, sess)
    assert "run them again" in _text(D) and warns and "Model quality checks" in warns[0]
    st.name = "a different study"
    assert not RS.has_checks(sess, st) and not RS.has_checks(SimpleNamespace(model_checks=None), st)


def test_the_chapter_is_written_by_the_report_engine(sat, tmp_path):
    from fea_core.report import ReportOptions, build_report, session_from
    st = copy.deepcopy(sat.study)
    st.space = SpaceSettings(requirement_set=_rs_with_limits(["mass_tol_rel"], confirmed=True))
    sess = session_from(st, sat.wd / "sat.step", None, sat.parts)
    sess.model_checks = mc.make_run(st, _all_results({"mass": cc.UNCONFIRMED}), {"mass_budget_t": 0.01})
    r = build_report(sess, tmp_path / "r.docx", ReportOptions(images=False))
    from docx import Document
    doc = Document(r["path"])
    heads = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert any("Model quality checks" in h for h in heads), heads
    assert not any("model quality checks" in w.lower() for w in r["warnings"]), r["warnings"]
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Verdict of the model: UNCONFIRMED_REQUIREMENTS" in text


# ---------------------------------------------------------------------------------------------------------------- equilibrium: no false FAIL on a support that carries a moment
def test_the_equilibrium_record_is_compared_by_its_residuals_not_by_the_scaled_imbalance():
    from fea_core.space import _checks_equil as eq
    rec = {"force_residual_N": [1e-9, 2e-9, 0.0], "moment_residual_Nmm": [0.0, 0.0, 3e-6], "force_scale_N": 1.0, "moment_scale_Nmm": 100.0, "imbalance": 4.3e-6}
    same = {"force_residual": [1e-9, 2e-9, 0.0], "moment_residual": [0.0, 0.0, 3e-6], "imbalance": 6.4e-7}          # the same residuals, a larger scale of the solver
    assert eq._consistent(rec, same)
    assert not eq._consistent(rec, {**same, "force_residual": [0.5, 0.0, 0.0]})                                    # a reaction that was changed after the solution
    assert not eq._consistent(rec, {**same, "moment_residual": [0.0, 0.0, 1.0]})
    assert not eq._consistent({"imbalance": 1e-9, **rec}, {"imbalance": 2e-9})                                       # a record without residual vectors is compared with nothing


def test_a_cantilever_under_gravity_is_not_reported_as_out_of_equilibrium(tmp_path):
    """The unit-gravity states of a bar clamped at one end: the clamp carries a moment, the solver's scale (sum of the nodal reaction norms) is larger than the totals per
    support - the relative imbalances differ by a factor of ~7 although the residuals are the same numbers (this was a false FAIL of check 7)."""
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.solve import prepare_mesh
    from fea_core.study import Fixed, MeshSettings
    from verification.cases import _study, box
    st, parts = _study("cantilever", tmp_path / "bar.step", [Part("Bar", box(0, 0, 0, 60, 10, 10))], MeshSettings(size=4.0))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.constraints.append(Fixed("clamp"))
    st.modal.enabled = True
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    res = mc.run_model_checks(st, mesh, None, None, {"threads": 2, "include_convergence": False}, workdir=tmp_path / "w", checks=["mass", "equilibrium"], parts=parts)
    by = {r.id: r for r in res}
    assert by["equilibrium"].status == cc.UNCONFIRMED, (by["equilibrium"].status, by["equilibrium"].findings)       # no limit: the numbers, no verdict - and no FAIL
    states = by["equilibrium"].values["states"]
    assert len(states) >= 3 and all(s["consistent_with_record"] for s in states if "consistent_with_record" in s), states

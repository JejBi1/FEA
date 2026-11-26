"""F14 S2: launch environments, QSL generator, 3-axis envelopes, mass properties and budgets (`fea_core.space.loads` / `environments` / `mass_props` / `budgets`,
the MCP tools of `tools_space_loads`).

The verification gate of the stage (Miles on a 1-DOF system, the bit-identical 3-axis envelope, analytic mass / centre of mass / Steiner inertia, the f_min check) runs
through `verification/space_load_cases.py`. Every level, factor and limit here is an ARBITRARY TEST VALUE, not a value of any standard.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import anyio
import numpy as np
import pytest

import verification.cases  # noqa: F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.space import budgets as BG
from fea_core.space import environments as EV
from fea_core.space import loads as LD
from fea_core.space import mass_props as MP
from fea_core.space.model import RequirementError, SpaceSettings
from fea_core.study import Study
from verification import space_cases as SC
from verification import space_load_cases as SL

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "fea_core" / "space"


# ------------------------------------------------------------------ the verification gate
@pytest.mark.parametrize("key", ["S2A", "S2B", "S2C", "S2D", "S2E"])
def test_verification_gate(key, tmp_path):
    case = SL.SPACE_LOAD_CASES[key]
    st, parts = case.build(tmp_path)
    assert case.run is not None
    out = case.run(st, tmp_path, parts)
    bad = [f"{c.name}: {c.value} vs {c.ref} (err {c.error:.2e}, tol {c.tol:g})" for c in case.evaluate(out, st) if not c.ok]
    assert not bad, bad


@pytest.mark.slow
def test_verification_gate_s2f_independent_three_axis(tmp_path):
    """S2F: the real solver; the 3-axis run against three complete single-axis analyses."""
    from fea_core import config
    if not config.CCX_EXE.exists():
        pytest.skip("CalculiX is needed")
    case = SL.SPACE_LOAD_CASES["S2F"]
    st, parts = case.build(tmp_path)
    out = case.run(st, tmp_path, parts)
    bad = [f"{c.name}: {c.value} vs {c.ref} (err {c.error:.2e}, tol {c.tol:g})" for c in case.evaluate(out, st) if not c.ok]
    assert not bad, bad


def test_cases_are_registered_for_the_run_all():
    assert {"S2A", "S2B", "S2C", "S2D", "S2E", "S2F"} <= set(verification.cases.CASES)


# ------------------------------------------------------------------ no standard number anywhere in the new modules
def test_new_modules_ship_no_number_of_a_standard():
    """The environments, limits and QSL factors are empty in every template; the modules bind no ECSS factor name to a number."""
    es = EV.environment_template()
    assert not re.search(r'"points":\s*\[\s*\[', json.dumps(es.to_dict()))
    assert all(not lv.kinds() for lv in es.levels) and es.launch_axis == ""
    from fea_core.space.requirements import ecss_32_10c_template
    rs = ecss_32_10c_template()
    BG.install_limit_rows(rs)
    assert {x.key for x in rs.limits} == set(BG.S2_LIMITS) and all(x.value is None for x in rs.limits)
    names = {"fosy", "fosu", "kq", "kp", "km", "kld", "kt"}
    for f in ("environments.py", "loads.py", "mass_props.py", "budgets.py"):
        tree = ast.parse((SPACE_DIR / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                assert not any(isinstance(t, ast.Name) and t.id.lower() in names for t in node.targets), f


# ------------------------------------------------------------------ environments: confirmation mechanism
def _filled() -> EV.EnvironmentSet:
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z")
    return es


def test_environment_confirmation_withdrawn_by_edit_and_import():
    es = _filled()
    assert not es.is_confirmed()
    EV.confirm_environment(es, "Test Engineer", "2026-01-01", via="gui")
    assert es.is_confirmed() and es.confirmed_via == "gui"
    es.levels[0].random.points[1][1] = 0.05                                    # an edit of a table
    assert es.confirmed and not es.is_confirmed()
    EV.confirm_environment(es, "Test Engineer")
    assert es.is_confirmed()
    es.launch_axis = "x"
    assert not es.is_confirmed()
    EV.confirm_environment(es, "Test Engineer")
    imported = EV.from_json(EV.to_json(es))
    assert not imported.confirmed and not imported.is_confirmed() and imported.levels[0].sine.points == es.levels[0].sine.points


def test_environment_cannot_be_confirmed_without_source_name_or_valid_table():
    es = _filled()
    with pytest.raises(RequirementError):
        EV.confirm_environment(es, "")
    es.levels[0].shock.clause = "<clause>"
    with pytest.raises(RequirementError) as e:
        EV.confirm_environment(es, "Test Engineer")
    assert any("clause" in m or "source" in m for m in e.value.issues)
    es.levels[0].shock.clause = "c"
    es.levels[0].sine.points = [[100.0, 5.0], [50.0, 5.0]]                     # decreasing frequencies
    assert any("increase" in m for m in EV.validate_environment(es))
    with pytest.raises(RequirementError):
        EV.confirm_environment(es, "Test Engineer")


def test_environments_round_trip_through_the_study_file():
    st = SL.steel_study()
    st.space = SpaceSettings()
    es = _filled()
    EV.confirm_environment(es, "Test Engineer", "2026-01-01")
    EV.store_environments(st.space, es)
    st2 = Study.from_dict(json.loads(json.dumps(st.to_dict())))
    es2 = EV.environments_of(st2.space)
    assert es2 is not None and es2.is_confirmed() and es2.levels[0].random.points == es.levels[0].random.points
    es2.levels[0].random.sigma = 4.0
    EV.store_environments(st2.space, es2)                                     # a stale confirmation is withdrawn on storing
    es3 = EV.environments_of(st2.space)
    assert es3 is not None and not es3.is_confirmed()


# ------------------------------------------------------------------ budgets: never PASS on unconfirmed inputs, FAIL only with confirmed ones
def test_budget_verdicts():
    props = MP.MassProperties(0.0012, np.array([10.0, 0.0, 5.0]), np.eye(3))                # 1.2 kg
    full = SL.space_with_limits({"mass_max_kg": 1.5, "cog_min_x_mm": 8.0, "cog_max_x_mm": 12.0})
    assert BG.mass_budget(props, full).verdict == BG.PASS
    tight = SL.space_with_limits({"mass_max_kg": 1.0})
    rep = BG.mass_budget(props, tight)
    assert rep.verdict == BG.FAIL and rep.items[0].status == BG.FAIL
    unconf = SL.space_with_limits({"mass_max_kg": 1.0}, confirmed=False)
    assert BG.mass_budget(props, unconf).verdict == BG.UNCONFIRMED                           # the same numbers, no FAIL and no PASS
    ok_unconf = SL.space_with_limits({"mass_max_kg": 2.0}, confirmed=False)
    assert BG.mass_budget(props, ok_unconf).verdict == BG.UNCONFIRMED
    assert BG.mass_budget(props, SpaceSettings()).verdict == BG.UNCONFIRMED                  # no limit at all
    lim = SL.space_with_limits({"mass_max_kg": 2.0})
    lim.requirement_set.limits[0].clause = ""                                                # a limit without its source cannot decide (and the edit withdraws the set)
    assert BG.mass_budget(props, lim).verdict == BG.UNCONFIRMED


# ------------------------------------------------------------------ QSL generator
def test_qsl_needs_factors_and_launch_axis():
    st = SL.steel_study()
    with pytest.raises(LD.LoadsError, match="requirement set"):
        LD.generate_qsl(st)
    st.space = SL.space_with_limits({})
    es = EV.environment_template()
    EV.store_environments(st.space, es)
    with pytest.raises(LD.LoadsError, match="launch axis"):
        LD.generate_qsl(st)
    es.launch_axis = "x"
    EV.store_environments(st.space, es)
    with pytest.raises(LD.LoadsError, match="QSL factors missing"):
        LD.generate_qsl(st)


def test_qsl_regeneration_and_status():
    st = SL.steel_study()
    st.space = SL.space_with_limits({"qsl_axial_g": 5.0, "qsl_lateral_g": 2.5})
    es = EV.environment_template()
    es.launch_axis = "z"
    EV.store_environments(st.space, es)
    cases = LD.generate_qsl(st)
    assert len(cases) == 14 and len({c.name for c in cases}) == 14                           # the default combination is axial_lateral
    assert LD.qsl_status(st)["current_with_requirement_set"] and LD.qsl_status(st)["requirement_set_confirmed"]
    z = next(c for c in st.load_cases if c.name == "QSL +Z")
    assert z.loads[0].vector == [0.0, 0.0, 5.0 * 9810.0]
    assert next(c for c in cases if c.name == "QSL +X").factors[0] == 2.5 and next(c for c in cases if c.name == "QSL -Y").factors[1] == -2.5   # lateral2 = lateral
    for x in st.space.requirement_set.limits:                                              # the factors change -> the registered cases are no longer current
        if x.key == "qsl_axial_g":
            x.value = 6.0
    assert not LD.qsl_status(st)["current_with_requirement_set"]
    from fea_core.study import LoadCase
    st.load_cases = [c for c in st.load_cases if c.name != "QSL +Z +X"]
    st.load_cases.append(LoadCase("QSL +Z +X", []))                                          # a foreign case with a generated name is never overwritten
    st.space.extra[LD.QSL_KEY]["names"].remove("QSL +Z +X")
    with pytest.raises(LD.LoadsError, match="exists"):
        LD.generate_qsl(st)
    assert LD.remove_qsl(st) == 13 and any(c.name == "QSL +Z +X" for c in st.load_cases) and LD.QSL_KEY not in st.space.extra


# ------------------------------------------------------------------ mass properties: invariants beyond the analytic case (S2A)
def test_inertia_is_symmetric_positive_and_parallel_axis_consistent():
    mesh = SL.box_mesh_tet10(30.0, 20.0, 10.0, (3, 2, 1))
    st = SL.steel_study()
    p = MP.mass_properties(st, mesh)
    assert np.allclose(p.inertia_cog, p.inertia_cog.T)
    w, _ = p.principal()
    assert (w > 0).all()
    pt = np.array([100.0, -20.0, 7.0])
    d = p.cog - pt
    assert np.allclose(p.inertia_about(pt), p.inertia_cog + p.mass_t * (d @ d * np.eye(3) - np.outer(d, d)))
    assert np.allclose(p.inertia_about(p.cog), p.inertia_cog)
    tet4 = SC.box_tet_mesh(6, 4, 2, 5.0)                                                    # tet4 meshes are integrated too (second moment exact for degree 2)
    q = MP.mass_properties(st, tet4)
    m = 7.85e-9 * 30.0 * 20.0 * 10.0
    assert q.mass_t == pytest.approx(m, rel=1e-12)
    assert q.inertia_cog[0, 0] == pytest.approx(m / 12 * (20.0 ** 2 + 10.0 ** 2), rel=1e-10)


def test_result_notes_missing_duration_and_sweep_rate():
    mesh = SL.single_tet10()
    basis = SL.synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
    lv = SL.three_axis_level()
    lv.random.duration_s = None
    lv.sine.sweep_rate_oct_min = None
    per, _w = LD.analyze_axes(SL.steel_study(), mesh, basis, SL.synthetic_statics(basis), Path("."), lambda s: None, lv, ["sine", "random"])
    r = LD.build_result(SL.steel_study(), mesh, lv, ["sine", "random"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    assert any("duration" in n for n in r.notes) and any("sweep rate" in n for n in r.notes)
    assert r.envelopes["sine"].D is None or r.envelopes["sine"].D.max() == 0.0                  # no sweep rate: no sweep damage


# ------------------------------------------------------------------ 3-axis: margins never PASS on an unconfirmed environment
def _envelope_result(mesh, level: EV.EnvLevel, s_value: float, env_confirmed: bool) -> LD.ThreeAxisResult:
    n = len(mesh.nodes)
    f = LD.AxisField(np.full(n, s_value, np.float32), np.zeros(n, np.float32), None)
    per = {"sine": {0: f, 1: f, 2: f}}
    env = LD.envelope_of(per["sine"], "sine")
    return LD.ThreeAxisResult(level.name, ["sine"], [0, 1, 2], per, {"sine": env}, {}, np.array([100.0]), np.zeros((1, 3)), None, env_confirmed, True,
                              env_digest="", level_kinds=["sine"], damping={"kind": "modal", "zeta": 0.02, "source": "test", "from_environment": True})


def _seal(r3: LD.ThreeAxisResult, study, mesh) -> LD.ThreeAxisResult:
    """Binds a hand-made result to the present state of the study, the mesh and the environments (what run_3axis does)."""
    es = EV.environments_of(study.space)
    r3.env_digest = es.content_digest() if es is not None else ""
    r3.run_digest = LD.compute_run_digest(study, mesh)
    return r3


def test_margins_from_envelopes_follow_the_confirmation_of_the_environment():
    from fea_core.space import margins as MG
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("protoflight"))
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA)])
    lv = SL.three_axis_level()
    es = EV.EnvironmentSet(levels=[lv], launch_axis="z")
    EV.store_environments(st.space, es)
    r3 = _seal(_envelope_result(mesh, lv, 100.0, True), st, mesh)
    base = MG.compute_margins(st, res)
    assert base.verdict == MG.PASS
    s_unc = LD.margins_with_envelopes(st, res, r3)                                          # the environment is not confirmed
    dyn = [r for r in s_unc.rows if r.case.startswith("3-axis")]
    assert dyn and all(r.status == "unconfirmed" for r in dyn) and s_unc.verdict == MG.UNCONFIRMED
    assert any("environment" in x for x in s_unc.reasons)
    EV.confirm_environment(es, "Test Engineer")
    EV.store_environments(st.space, es)
    _seal(r3, st, mesh)
    s_ok = LD.margins_with_envelopes(st, res, r3)
    j = SC.design_factor("metallic", "yield", "protoflight")
    y = next(r for r in s_ok.rows if r.case.startswith("3-axis") and r.check == "yield")
    assert y.mos == pytest.approx(SC.TEST_FY / (j * 100.0) - 1.0, rel=1e-12) and y.status == "pass" and s_ok.verdict == MG.PASS
    es.levels[0].sine.points[0][1] = 6.0                                                    # an edit after the run: the result is stale, never PASS
    EV.store_environments(st.space, es)
    assert not r3.current(st, mesh) and LD.margins_with_envelopes(st, res, r3).verdict == MG.UNCONFIRMED
    r3.sigma = None                                                                         # the random RMS enters margins only with a user-entered sigma
    r3.kinds = ["random"]
    r3.envelopes["random"] = r3.envelopes["sine"]
    assert r3.scalar_sources() == []


# ------------------------------------------------------------------ MCP tools
def _session(tmp_path):
    from fea_core.mcp.session import Session
    sess = Session(tmp_path / "mcp")
    sess.study = SL.steel_study("S2 mcp")
    mesh = SL.box_mesh_tet10(40.0, 20.0, 10.0)
    sess.mesh = mesh
    sess.mesh_sig = sess.mesh_signature()
    return sess


def _run(sess, steps):
    from mcp import Client
    from fea_core.mcp.server import build_server
    out: list = []

    async def main():
        srv = build_server(sess, groups_spec="space")
        async with Client(srv, mode="legacy") as c:
            for name, args, expect_err in steps:
                r = await c.call_tool(name, args)
                text = r.content[0].text
                assert bool(r.is_error) == expect_err, (name, text)
                out.append(text if expect_err else json.loads(text))
    anyio.run(main)
    return out


def test_mcp_group_lists_the_four_tools(tmp_path):
    from mcp import Client
    from fea_core.mcp.server import build_server

    async def main():
        srv = build_server(_session(tmp_path), groups_spec="space")
        async with Client(srv, mode="legacy") as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"fea_space_qsl_generate", "fea_space_env_set", "fea_space_run_3axis", "fea_space_mass_budget"} <= names
    anyio.run(main)


def test_mcp_environment_workflow_and_no_self_confirmation(tmp_path):
    sess = _session(tmp_path)
    lvl = {"name": "qualification", "combine": "srss",
           "sine": {"points": [[20, 5], [100, 5], [400, 8]], "sweep_rate_oct_min": 2, "document": "test doc", "revision": "A", "clause": "c1"},
           "random": {"points": [[20, 0.01], [200, 0.03], [1000, 0.03]], "duration_s": 120, "sigma": 3, "document": "test doc", "revision": "A", "clause": "c2"},
           "shock": {"points": [[100, 20], [1000, 200]], "document": "test doc", "revision": "A", "clause": "c3"}}
    steps = [
        ("fea_space_env_set", {"action": "get"}, False),                                                            # 0
        ("fea_space_env_set", {"action": "template"}, False),                                                       # 1
        ("fea_space_env_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, True),                        # 2: empty template
        ("fea_space_env_set", {"action": "set", "data": {"launch_axis": "z", "levels": [lvl]}, "confirmed_by": "Test Engineer"}, True),   # 3: values + confirm in one call
        ("fea_space_env_set", {"action": "set", "data": {"launch_axis": "z", "levels": [lvl]}}, False),             # 4
        ("fea_space_env_set", {"action": "confirm"}, True),                                                         # 5: a name is required
        ("fea_space_env_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),                       # 6
        ("fea_space_env_set", {"action": "set", "data": {"levels": [{"name": "qualification", "random": {"sigma": 4}}]}}, False),   # 7: an edit withdraws it
        ("fea_space_env_set", {"action": "set", "data": {"bogus": 1}}, True),                                       # 8
        ("fea_space_env_set", {"action": "set", "data": {"levels": [{"name": "flight"}]}}, False),                  # 9: unknown level name -> reported by validation
    ]
    o = _run(sess, steps)
    assert o[0]["environments"] is None
    assert o[1]["environments"]["levels_with_tables"] == {"qualification": [], "protoflight": [], "acceptance": []}
    assert "confirmed_by" in o[3] or "own" in o[3]
    assert o[4]["environments"]["levels_with_tables"]["qualification"] == ["sine", "random", "shock"] and not o[4]["environments"]["is_confirmed"]
    assert o[6]["environments"]["is_confirmed"] and o[6]["environments"]["confirmed_via"] == "mcp"
    assert not o[7]["environments"]["is_confirmed"] and any("withdrawn" in n for n in o[7]["notes"])
    assert any("flight" in p for p in o[9]["environments"]["problems"])
    es = EV.environments_of(sess.study.space)
    assert es is not None and es.levels[0].random.sigma == 4.0 and es.levels[0].random.points[1] == [200.0, 0.03]      # a partial update keeps the other fields


def test_mcp_qsl_and_mass_budget(tmp_path):
    sess = _session(tmp_path)
    lims = [{"key": k, "value": v, "document": "test doc", "revision": "A", "clause": "c"}
            for k, v in (("qsl_axial_g", 6.0), ("qsl_lateral_g", 3.0), ("mass_max_kg", 1.0), ("cog_min_x_mm", 15.0), ("cog_max_x_mm", 25.0))]
    steps = [
        ("fea_space_qsl_generate", {}, True),                                                                       # 0: no requirement set
        ("fea_space_requirement_set", {"action": "template"}, False),                                               # 1
        ("fea_space_env_set", {"action": "limits"}, False),                                                         # 2
        ("fea_space_requirement_set", {"action": "set", "data": {"verification_approach": "analysis_only", "limits": lims}}, False),   # 3
        ("fea_space_env_set", {"action": "template"}, False),                                                       # 4
        ("fea_space_qsl_generate", {}, True),                                                                       # 5: launch axis missing
        ("fea_space_env_set", {"action": "set", "data": {"launch_axis": "x", "qsl_combination": "none"}}, False),   # 6
        ("fea_space_qsl_generate", {}, False),                                                                      # 7
        ("fea_space_mass_budget", {}, False),                                                                       # 8
        ("fea_space_qsl_generate", {"remove": True}, False),                                                        # 9
    ]
    o = _run(sess, steps)
    assert "requirement set" in o[0]
    assert len(o[2]["added"]) == len(BG.S2_LIMITS) and o[3]["requirement_set"]["limits"]
    assert "launch axis" in o[5]
    gen = o[7]
    assert len(gen["generated"]) == 6 and gen["load_cases_total"] == 6 and not gen["status"]["requirement_set_confirmed"]
    assert {c["name"] for c in gen["generated"]} == {"QSL +X", "QSL -X", "QSL +Y", "QSL -Y", "QSL +Z", "QSL -Z"}
    mb = o[8]
    m = 7.85e-9 * 40 * 20 * 10 * 1000.0
    assert mb["mass_properties"]["mass_kg"] == pytest.approx(m, rel=1e-12) and mb["mass_properties"]["cog_mm"][0] == pytest.approx(20.0)
    assert mb["mass_budget"]["verdict"] == "UNCONFIRMED" and mb["frequency_check"]["verdict"] == "SKIPPED"       # the set is not confirmed: no PASS / FAIL
    assert "parts" not in mb["mass_properties"] and o[9]["removed"] == 6


def test_mcp_run_3axis_wiring(tmp_path, monkeypatch):
    """The tool passes level / kinds / axes to the orchestrator, keeps the result for the budget tool and the report, and reports the errors of the orchestrator."""
    from fea_core.mcp import tools_mesh
    sess = _session(tmp_path)
    sess.study.space = SpaceSettings()
    EV.store_environments(sess.study.space, _filled())
    monkeypatch.setattr(tools_mesh, "readiness", lambda s: {"ready": True, "errors": []})
    mesh = SL.single_tet10()
    basis = SL.synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
    st = SL.steel_study()
    lv = SL.three_axis_level()
    per, _w = LD.analyze_axes(st, mesh, basis, SL.synthetic_statics(basis), tmp_path, lambda s: None, lv, ["random"])
    canned = LD.build_result(st, mesh, lv, ["random"], [0, 1, 2], per, basis, env_confirmed=False, req_confirmed=False)
    seen: list[dict] = []

    def fake(study, mesh_, wd, log, *, level, kinds, axes, on_line=None, cancel=None):
        seen.append({"level": level, "kinds": kinds, "axes": axes})
        if level == "acceptance":
            raise LD.LoadsError("the level 'acceptance' has no table")
        return _seal(canned, study, mesh_)                                      # what the real run does: the digest of the study, the mesh and the environments
    monkeypatch.setattr(LD, "run_3axis", fake)
    o = _run(sess, [("fea_space_run_3axis", {"level": "Qualification", "kinds": ["random"], "axes": "zx"}, False),
                    ("fea_space_run_3axis", {"level": "acceptance"}, True),
                    ("fea_space_run_3axis", {"level": "qualification", "axes": "q"}, True),
                    ("fea_space_mass_budget", {}, False)])
    assert seen == [{"level": "qualification", "kinds": ["random"], "axes": (0, 2)}]                # a level without a table is refused before any run
    assert o[0]["level"] == "qualification" and o[0]["status"] == "UNCONFIRMED_INPUTS" and set(o[0]["kinds"]) == {"random"}
    assert set(o[0]["kinds"]["random"]["per_axis_max_S_MPa"]) == {"X", "Y", "Z"}
    assert "no table" in o[1] and "axes" in o[2]
    fc = o[3]["frequency_check"]                                                            # the frequencies of the kept 3-axis run, launch axis z, no limit entered
    assert fc["lowest_Hz"][:3] == [85.0, 130.0, 210.0] and fc["launch_axis"] == "z" and fc["verdict"] == "UNCONFIRMED"
    assert [i["status"] for i in fc["items"]] == ["UNCONFIRMED", "UNCONFIRMED"] and fc["items"][0]["value"] == 85.0


# ------------------------------------------------------------------ the real chain (needs the solver)
@pytest.mark.slow
@pytest.mark.parametrize("engine", ["ccx", "direct"])
def test_run_3axis_on_a_cantilever_matches_independent_runs(tmp_path, engine):
    """run_3axis over one basis against three completely independent single-axis `run_study` runs (each with its own modal basis and its own statics), for both modal
    engines. The two chains compute the same numbers; the tolerance is engine-aware: 1e-9 of the maximum for ccx (measured 8e-12, the last bits of single-precision
    fields), 1e-6 for the direct engine (rslab computes in float32; measured 2.9e-8 - 1e-9 is below its resolution)."""
    from fea_core import config
    if not config.CCX_EXE.exists():
        pytest.skip("CalculiX is needed")
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.solve import StudyError, prepare_mesh, run_study
    from fea_core.study import Fixed, LoadCase, MeshSettings
    from verification.cases import _study, box
    st, parts = _study("S2 cantilever", tmp_path / "model.step", [Part("Konzola", box(0, 0, 0, 80.0, 12.0, 7.0))], MeshSettings(size=4.0))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.constraints.append(Fixed("clamp"))
    st.load_cases.append(LoadCase("F", []))
    st.solver.backend, st.solver.modal_engine = "auto", engine
    st.dynamics.enabled, st.dynamics.modes = True, 8
    st.dynamics.damping.zeta = 0.02
    st.space = SpaceSettings()
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z", damping_zeta=0.03, damping_source="test report")
    EV.store_environments(st.space, es)
    try:
        mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    except StudyError as e:
        if "rslab" in str(e):
            pytest.skip("the direct solver library (module direct) is not installed")
        raise
    r3 = LD.run_3axis(st, mesh, tmp_path / "run", lambda s: None, level="qualification")
    assert r3.status == "UNCONFIRMED_INPUTS" and r3.kinds == ["sine", "random", "shock"] and len(r3.freq) >= 3 and r3.complete
    assert r3.damping["zeta"] == 0.03 and r3.run_digest == LD.compute_run_digest(st, mesh) and r3.current(st, mesh)
    f_ext = float(r3.freq.max()) * 1.01                                  # the shock table is continued at its ZPA above its last point, as the space path does
    tol = 1e-9 if engine == "ccx" else 1e-6
    for a in (0, 1, 2):
        s1 = LD._axis_study(st, es.levels[0], ["sine", "random", "shock"], a, srs_extend_to=f_ext)
        assert s1.dynamics.damping.zeta == 0.03
        d = run_study(s1, tmp_path / f"ind{a}", mesh=mesh, parts=parts).dynamics
        for kind, ref in (("random", np.asarray(d.random["S_rms"], np.float32)), ("sine", np.asarray(d.harmonic["env_S"], np.float32)),
                          ("shock", np.asarray(d.shock["mises"], np.float32))):
            got = r3.per_axis[kind][a].S
            assert np.abs(got.astype(np.float64) - ref).max() <= tol * max(float(ref.max()), 1e-12), (engine, a, kind)
        got_u = r3.per_axis["random"][a].U.astype(np.float64)
        ref_u = np.linalg.norm(np.asarray(d.random["U_rms"], float), axis=1).astype(np.float32).astype(np.float64)       # the field is stored in single precision
        assert np.abs(got_u - ref_u).max() <= tol * max(float(ref_u.max()), 1e-12), (engine, a, "U")
    env = r3.envelopes["random"]
    assert np.array_equal(env.S, np.max(np.stack([r3.per_axis["random"][a].S for a in (0, 1, 2)]), axis=0))
    assert env.D is not None and env.D.max() >= env.D_axis_max.max()


# ------------------------------------------------------------------ report subsection
def test_report_subsection_prints_inputs_sources_and_status():
    pytest.importorskip("docx")
    from types import SimpleNamespace

    from fea_core.dynamics.model import PointMass
    from fea_core.report import space as RS
    from fea_core.report import space_loads as RL
    from fea_core.report.word import _Doc
    mesh = SL.box_mesh_tet10(40.0, 20.0, 10.0)
    st = SL.steel_study("S2 report")
    st.masses.append(PointMass("Support", 1.0e-3, point=[60.0, 5.0, 5.0], name="unit"))
    st.space = SL.space_with_limits({"qsl_axial_g": 6.0, "qsl_lateral_g": 3.0, "mass_max_kg": 1.0, "f_min_axial_hz": 100.0})
    es = _filled()
    EV.store_environments(st.space, es)
    LD.generate_qsl(st)
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", 10.0)])
    sess = SimpleNamespace(study=st, results=res)
    assert RL.has_loads(st, sess)
    D = _Doc()
    warns = RS.space_section(D, sess)                                          # the chapter calls the subsection
    assert not [w for w in warns if "Mass properties" in w], warns
    text = "\n".join([p.text for p in D.doc.paragraphs] + [c.text for t in D.doc.tables for row in t.rows for c in row.cells])
    for needle in ("Launch loads, mass properties and budgets", "Launch environments entered by the user", "NOT confirmed", "test",
                   "Quasi-static launch cases", "QSL +Z", "Mass budget", "UNCONFIRMED", "Inertia tensor about the centre of mass", "Mass by part"):
        assert needle in text, needle
    assert "PASS" not in text.split("Launch loads, mass properties and budgets")[1]          # nothing here can be a PASS: the set is confirmed but the environment is not and mass fails
    # with a 3-axis result kept for the study the envelopes are printed, labelled by the state of the inputs
    basis = SL.synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
    lv = es.levels[0]
    per, _w = LD.analyze_axes(SL.steel_study(), SL.single_tet10(), basis, SL.synthetic_statics(basis), Path("."), lambda s: None, lv, ["random", "shock"])
    r3 = LD.build_result(st, SL.single_tet10(), lv, ["random", "shock"], [0, 1, 2], per, basis, env_confirmed=False, req_confirmed=True, env_digest=es.content_digest(),
                         run_digest=LD.compute_run_digest(st, res.mesh))
    LD.remember(sess, r3)
    D3 = _Doc()
    RL.loads_section(D3, sess)
    t3 = chr(10).join([p.text for p in D3.doc.paragraphs] + [c.text for t in D3.doc.tables for row in t.rows for c in row.cells])
    for needle in ("3-axis dynamic envelopes", "UNCONFIRMED_INPUTS", "Envelope by part - random", "Lowest natural frequencies"):
        assert needle in t3, needle
    # a study without any of these inputs prints nothing
    st2 = SL.steel_study()
    st2.space = SpaceSettings()
    res2 = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", 10.0)])
    assert not RL.has_loads(st2, SimpleNamespace(study=st2, results=res2))
    D2 = _Doc()
    assert RL.loads_section(D2, SimpleNamespace(study=st2, results=res2)) == [] and not D2.doc.tables


def test_run_3axis_tool_is_gated_like_the_solver_tools(tmp_path):
    """The tool that starts the modal analysis stands behind the same confirmation gate as fea_solve_run, and the window gets its arguments without the MCP context."""
    from pathlib import Path as _P

    from fea_core.mcp import tools_space_loads
    sess = _session(tmp_path)
    sess.require_confirmation = True
    sess.study.space = SpaceSettings()
    EV.store_environments(sess.study.space, _filled())
    o = _run(sess, [("fea_space_run_3axis", {"level": "qualification"}, True),
                    ("fea_space_run_3axis", {"level": "qualification", "confirmed": "0" * 12}, True)])
    assert "requires the user's confirmation" in o[0] and "digest does not match" in o[1]
    seen: dict = {}

    class Reg:
        def tool(self, **kw):
            def deco(fn):
                seen[fn.__name__] = fn
                return fn
            return deco
    tools_space_loads.register(Reg(), type(sess)(_P(".")))
    assert seen["fea_space_run_3axis"].ipc_args({"a": 1, "ctx": object(), "confirmed": "x"}) == {"a": 1, "confirmed": "x"}

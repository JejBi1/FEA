"""Setup confirmation before an AI-started calculation (F15-S1): digest, numbered table and picture, the gate of
`fea_solve_run` / `fea_job_start`, the three ways of asking the user (window, MCP elicitation, the honest agent fallback).

Fast tests (no meshing; the calculation is replaced by a stub where only the gate is of interest); one end-to-end test with a real
calculation is marked slow. The window path of the dialog is in tests/gui/gui_setup_confirm_smoke.py (run through hidden_run).
"""
import copy
import json
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client, types  # noqa: E402

from fea_core import signature  # noqa: E402
from fea_core.mcp import setup_proof as sp  # noqa: E402
from fea_core.mcp.server import build_server  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402
from fea_core.selectors import FaceSelector  # noqa: E402
from fea_core.study import (Displacement, Fixed, Force, Gravity, LoadCase, Material, Pressure, Study)  # noqa: E402


# ===================================================================== digest (no geometry)

def face(x=0.0, part="P", area=200.0):
    return FaceSelector(part, "plane", area, [x, 10, 5], [x, 10, 5], [1, 0, 0], [x, 0, 0, x, 20, 10])


def make_study(force=-100.0, sel="Support", load_sel="Tip", case="Pull", young=210000.0, name="beam"):
    st = Study(name, "beam.step")
    st.add_material(Material("Steel", young, 0.3, 7.85e-9, Re=235.0))
    st.part_materials["P"] = "Steel"
    st.add_selection(sel, faces=[face(0.0)])
    st.add_selection(load_sel, faces=[face(200.0)])
    st.constraints.append(Fixed(sel))
    st.load_cases.append(LoadCase(case, [Force(load_sel, [0, 0, force])]))
    return st


def test_digest_survives_save_load_and_renames(tmp_path):
    st = make_study()
    d0 = sp.setup_digest(st)
    assert len(d0) == 64 and int(d0, 16) >= 0
    f = tmp_path / "s.json"
    st.save(f)
    assert sp.setup_digest(Study.load(f)) == d0
    assert sp.setup_digest(Study.from_dict(copy.deepcopy(st.to_dict()))) == d0
    # names that are only labels: the study, the selections (references follow), the material, the load case
    assert sp.setup_digest(make_study(name="other name")) == d0
    assert sp.setup_digest(make_study(sel="A", load_sel="B", case="Tension")) == d0
    renamed = make_study()
    renamed.materials["Steel"].name = "S235 (renamed)"
    assert sp.setup_digest(renamed) == d0
    # 100 and 100.0 are the same number, the solver choice and the Gmsh threads do not decide the answer
    assert sp.setup_digest(make_study(force=-100)) == d0
    other = make_study()
    other.solver.backend = "ngs_direct"
    other.mesh.threads = 4
    assert sp.setup_digest(other) == d0


def test_digest_changes_with_every_input():
    base = sp.setup_digest(make_study())
    assert sp.setup_digest(make_study(force=-101.0)) != base                    # a value
    assert sp.setup_digest(make_study(young=200000.0)) != base                  # a material property
    moved = make_study()
    moved.selections["Support"].faces = [face(5.0)]                              # another face
    assert sp.setup_digest(moved) != base
    swapped = make_study()                                                        # the same two selections, swapped roles
    swapped.constraints[0].selection, swapped.load_cases[0].loads[0].selection = "Tip", "Support"
    assert sp.setup_digest(swapped) != base
    other_mat = make_study()
    other_mat.add_material(Material("Alu", 70000.0, 0.33, 2.7e-9))
    other_mat.part_materials["P"] = "Alu"
    assert sp.setup_digest(other_mat) != base
    more = make_study()
    more.load_cases.append(LoadCase("G", [Gravity()]))
    assert sp.setup_digest(more) != base                                        # a new load
    pr = make_study()
    pr.load_cases[0].loads.append(Pressure("Tip", 2.0))
    assert sp.setup_digest(pr) != base
    dc = make_study()
    dc.constraints.append(Displacement("Tip", ux=0.1))
    assert sp.setup_digest(dc) != base
    # geometry (content hash) and the faces the selections resolved to
    assert sp.setup_digest(make_study(), geometry_hash="a" * 64) != base
    assert sp.setup_digest(make_study(), geometry_hash="a" * 64) != sp.setup_digest(make_study(), geometry_hash="b" * 64)
    r1 = sp.setup_digest(make_study(), resolved={"Support": ["P#1"], "Tip": ["P#2"]})
    r2 = sp.setup_digest(make_study(), resolved={"Support": ["P#1"], "Tip": ["P#3"]})
    assert r1 != r2 and r1 != base


def test_digest_covers_every_mesh_input():
    """Everything that decides the mesh (signature.mesh_inputs) decides the digest too."""
    base = sp.setup_digest(make_study())

    def changed(mut):
        st = make_study()
        mut(st)
        return sp.setup_digest(st) != base
    assert changed(lambda s: setattr(s.mesh, "size", 3.0))
    assert changed(lambda s: setattr(s.mesh, "order", 1))
    assert changed(lambda s: s.mesh.local.append({"selection": "Tip", "size": 1.0}))
    assert changed(lambda s: s.ignored_parts.append("P"))
    assert changed(lambda s: setattr(s.joints, "contact_joint", not s.joints.contact_joint))
    assert changed(lambda s: s.imprints.append({"selection": "Tip", "kind": "x"}))
    assert set(signature.BASE_KEYS) <= {"selections", "mesh", "bolts", "welds", "joints", "ignored_parts"}      # the list this test knows


def test_digest_ignores_post_processing():
    base = sp.setup_digest(make_study())
    st = make_study()
    st.hidden_parts.append("P")
    st.load_cases[0].category = "permanent"
    st.assessment = {"required_safety": 2.0}
    assert sp.setup_digest(st) == base


def test_prefix_matching():
    d = sp.setup_digest(make_study())
    assert sp.digest_matches(d, d) and sp.digest_matches(d, d[:12]) and sp.digest_matches(d, d[:12].upper())
    assert not sp.digest_matches(d, d[:11]) and not sp.digest_matches(d, "0" * 12) and not sp.digest_matches(d, None)


# ===================================================================== MCP: review, gate, ways of asking

@pytest.fixture(scope="module")
def beam_step(tmp_path_factory):
    from fea_core.geometry import Part, write_step
    from verification.cases import box
    d = tmp_path_factory.mktemp("setup_geo")
    p = d / "beam.step"
    write_step([Part("Beam", box(0, 0, 0, 200, 20, 10))], p)
    return p


class Client2:
    """MCP client in memory + the session of the server."""

    def __init__(self, c, sess):
        self.c, self.sess = c, sess

    async def raw(self, tool, **args):
        return await self.c.call_tool(tool, args)

    async def call(self, tool, **args):
        r = await self.c.call_tool(tool, args)
        t = r.content[0].text
        if r.is_error:
            return {"__error__": t}
        return json.loads(t) if t[:1] in "{[" else {"__text__": t}


def run(tmp_path, body, elicit=None, confirm=True):
    async def main():
        sess = Session(tmp_path / "mcp")
        if confirm is not None:
            sess.require_confirmation = confirm
        srv = build_server(sess, groups_spec="all")
        async with Client(srv, mode="legacy", elicitation_callback=elicit) as c:
            await body(Client2(c, sess))
    anyio.run(main)


async def make_beam(s, step, tmp_path):
    await s.call("fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"))
    await s.call("fea_materials_assign", part="*", material_name="Ocel S235")
    await s.call("fea_constraint_add_fixed", name="Fixed end", selection={"x": 0})
    await s.call("fea_load_add_force", case_name="Pull", name="Tip", selection={"x": 200}, force=[0, 0, -100])
    await s.call("fea_load_add_gravity", case_name="Pull")


@pytest.fixture
def stub_solve(monkeypatch):
    """The calculation itself is not the subject: solve_core returns at once (the gate and the stamp stay real)."""
    import fea_core.mcp.tools_results as tr
    calls = []

    def fake(sess, backend="auto", timeout_s=600, cancel=None, on_progress=None, on_log=None, on_line=None, confirmation=None):
        from fea_core.mcp import tools_setup
        tools_setup.verify(sess, confirmation)
        calls.append(confirmation)
        return {"ok": True, "setup_confirmed_by": confirmation["confirmed_by"] if confirmation else None}
    monkeypatch.setattr(tr, "solve_core", fake)
    return calls


def test_review_tool_gives_picture_table_digest(tmp_path, beam_step):
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        r = await s.raw("fea_setup_review", output_path=str(tmp_path / "review.png"))
        assert [type(x).__name__ for x in r.content] == ["TextContent", "ImageContent"]
        d = json.loads(r.content[0].text)
        assert d["digest"] == sp.session_digest(s.sess) and d["digest_short"] == d["digest"][:12]
        rows = sp.setup_table(s.sess.study, s.sess)
        assert [x["no"] for x in rows] == [1, 2, 3]
        assert [x["kind"] for x in rows] == ["Fixed support", "Force", "Self-weight"]
        assert rows[0]["selection"] == "Fixed end" and rows[0]["n"] == 1 and rows[0]["size"] == 200.0 and rows[0]["entity"] == "faces"
        assert rows[0]["ids"] == ["Beam#%d" % int(rows[0]["ids"][0].split("#")[1])]
        assert rows[1]["cases"] == ["Pull"] and "F = (0; 0; -100) N" in rows[1]["value"]
        assert rows[2]["entity"] == "model" and "9810" in rows[2]["value"]
        for no, line in enumerate(d["table"].splitlines()[2:], 1):                 # the markdown table: one row per item
            assert line.startswith(f"| {no} |")
        assert d["confirmation"]["required"] is True and "waiting" in d["confirmation"]["state"]
        assert "confirmed='%s'" % d["digest_short"] in d["next"]
        assert (tmp_path / "review.png").stat().st_size > 5000
        # the picture carries the numbers of the table: one badge per row, the colours of the legend are the colours of the table
        from fea_core.mcp.scenes import model_scene
        sc = model_scene(s.sess, None, None, False, None, numbered=True)
        badges = sorted(int(m["text"]) for m in sc.markers if m.get("badge"))
        assert badges == [1, 2, 3] and sorted(sc.numbers) == [1, 2, 3]
        assert [t.split(".")[0] for _, t in sc.swatches if t[:1].isdigit()] == ["1", "2", "3"]
    run(tmp_path, body)


def test_picture_numbers_every_kind_of_item(tmp_path, beam_step):
    """Edge / point forces, pressure, a remote load, rotation, thermal boundary conditions: every item has its badge and row."""
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        await s.call("fea_load_add_pressure", case_name="Press", name="Top", selection={"z": 10}, pressure=1.5)
        await s.call("fea_load_add_remote", case_name="Press", name="Rem", selection={"x": 200}, point=[200, 10, 5], force=[0, 10, 0])
        await s.call("fea_load_add_rotation", case_name="Spin", axis_point=[0, 0, 0], axis_direction=[0, 0, 1], omega=10.0)
        await s.call("fea_thermal_case_create", name="Heat")
        await s.call("fea_thermal_add_temperature", case="Heat", name="Hot", selection={"x": 0}, value_C=80.0)
        await s.call("fea_thermal_add_convection", case="Heat", name="Air", selection={"x": 200}, h_W_m2K=10.0, ambient_C=20.0)
        from fea_core.mcp.scenes import model_scene
        st = s.sess.study
        n_items = len(sp.numbered_items(st))
        assert n_items == 1 + 5 + 2, n_items          # support, force + gravity + pressure + remote + rotation, 2 heat conditions
        sc = model_scene(s.sess, None, None, False, None, numbered=True)
        assert sorted(int(m["text"]) for m in sc.markers if m.get("badge")) == list(range(1, n_items + 1))
        rows = sp.setup_table(st, s.sess, sc.numbers)
        assert len(rows) == n_items and all("color" in r for r in rows)
        assert {r["group"] for r in rows} == {"constraint", "load", "thermal"}
        assert any(r["cases"] == ["Heat"] for r in rows)
        d = await s.call("fea_setup_review", output_path=str(tmp_path / "all.png"))
        assert d["digest"] == sp.session_digest(s.sess)
    run(tmp_path, body)


def test_gate_default_off_headless(tmp_path, beam_step, stub_solve):
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        assert sp.confirmation_required(s.sess) is False
        d = await s.call("fea_solve_run")
        assert d["ok"] and "setup_confirmed_by" in d and stub_solve == [None]
        d = await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png"))
        assert d["confirmation"]["required"] is False
    run(tmp_path, body, confirm=None)


def test_env_variable_turns_the_confirmation_on(tmp_path, beam_step, stub_solve, monkeypatch):
    monkeypatch.setenv("FEA_MCP_CONFIRM", "1")

    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        assert sp.confirmation_required(s.sess) is True
        assert "requires the user's confirmation" in (await s.call("fea_solve_run"))["__error__"]
        s.sess.require_confirmation = False                                          # the session setting wins (the window sets it)
        assert (await s.call("fea_solve_run"))["ok"]
    run(tmp_path, body, confirm=None)


def test_refused_without_digest_or_with_a_stale_one(tmp_path, beam_step, stub_solve):
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        msg = (await s.call("fea_solve_run"))["__error__"]
        assert "fea_setup_review" in msg and "confirmed=<digest_short>" in msg
        assert "fea_setup_review" in (await s.call("fea_job_start", kind="solve"))["__error__"]
        rev = await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png"))
        short = rev["digest_short"]
        assert "does not match" in (await s.call("fea_solve_run", confirmed="0" * 12))["__error__"]
        assert "does not match" in (await s.call("fea_solve_run", confirmed=short[:6]))["__error__"]      # too short = not a digest
        assert stub_solve == []
        # a changed value changes the digest: the digest of the old review is stale
        await s.call("fea_load_add_force", case_name="Pull", name="Tip", selection="Tip", force=[0, 0, -200])
        err = (await s.call("fea_solve_run", confirmed=short))["__error__"]
        assert "changed since the review" in err and stub_solve == []
        new = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r2.png")))["digest_short"]
        assert new != short
    run(tmp_path, body)


def test_agent_fallback_is_marked_honestly(tmp_path, beam_step, stub_solve):
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        short = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png")))["digest_short"]
        d = await s.call("fea_solve_run", confirmed=short)
        assert d["ok"] and d["setup_confirmed_by"] == "agent"
        assert stub_solve[-1]["confirmed_by"] == "agent" and stub_solve[-1]["channel"] == "agent" and stub_solve[-1]["digest"].startswith(short)
        d = await s.call("fea_job_start", kind="solve", confirmed=short)           # a job may start too (the body is the stub)
        assert "job_id" in d, d
        rev = await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png"))
        assert rev["confirmation"]["by"] == "agent" and rev["confirmation"]["state"] == "confirmed"
    run(tmp_path, body)


def elicit_answer(action, confirm=True, seen=None):
    async def cb(ctx, params):
        if seen is not None:
            seen.append(params.message)
        return types.ElicitResult(action=action, content={"confirm": confirm} if action == "accept" else None)
    return cb


def test_elicitation_accept_counts_as_the_user(tmp_path, beam_step, stub_solve):
    seen: list[str] = []

    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        short = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png")))["digest_short"]
        d = await s.call("fea_solve_run", confirmed=short)
        assert d["ok"] and d["setup_confirmed_by"] == "user"
        rec = stub_solve[-1]
        assert rec["confirmed_by"] == "user" and rec["channel"] == "elicitation"
        assert len(seen) == 1 and "| 1 | Fixed support |" in seen[0] and "Fixed end" in seen[0] and short in seen[0]
        again = await s.call("fea_solve_run", confirmed=short)                      # the same setup: the user is not asked twice
        assert again["ok"] and len(seen) == 1
        # a changed setup is asked again
        await s.call("fea_load_add_force", case_name="Pull", name="Tip", selection="Tip", force=[0, 0, -300])
        short2 = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r2.png")))["digest_short"]
        assert (await s.call("fea_solve_run", confirmed=short2))["ok"] and len(seen) == 2
    run(tmp_path, body, elicit=elicit_answer("accept", seen=seen))


@pytest.mark.parametrize("answer", [("decline", True), ("cancel", True), ("accept", False)])
def test_elicitation_refusal_stops_the_calculation(tmp_path, beam_step, stub_solve, answer):
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        short = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png")))["digest_short"]
        err = (await s.call("fea_solve_run", confirmed=short))["__error__"]
        assert "declined" in err and stub_solve == []
        assert s.sess.confirmation["decision"] == "rejected" and s.sess.confirmation["by"] == "user"
        assert (await s.call("fea_job_start", kind="solve", confirmed=short))["__error__"]
    run(tmp_path, body, elicit=elicit_answer(answer[0], answer[1]))


def test_no_digest_no_question(tmp_path, beam_step, stub_solve):
    seen: list[str] = []

    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        # without the digest the user is not even asked: the agent has to show the review first
        assert "requires the user's confirmation" in (await s.call("fea_solve_run"))["__error__"]
        assert seen == []
    run(tmp_path, body, elicit=elicit_answer("accept", seen=seen))


class FakeWindow:
    """The part of the FEA Studio window that matters here: the session says that a window decides."""

    def __init__(self):
        self.opened = []

    def open_review(self, payload):
        self.opened.append(payload)


def test_window_route_only_the_click_counts(tmp_path, beam_step, stub_solve):
    win = FakeWindow()

    async def body(s):
        s.sess.confirm_ui = win
        await make_beam(s, beam_step, tmp_path)
        r = await s.raw("fea_setup_review", output_path=str(tmp_path / "r.png"))
        assert [type(x).__name__ for x in r.content] == ["TextContent"]            # with a window: text only, the user sees the picture there
        d = json.loads(r.content[0].text)
        assert "FEA Studio window" in d["picture_shown_to_user"] and "click Confirm" in d["next"]
        assert len(win.opened) == 1
        pay = win.opened[0]
        assert pay["digest"] == d["digest"] and [x["no"] for x in pay["rows"]] == [1, 2, 3] and Path(pay["image"]).is_file()
        short = d["digest_short"]
        # the agent passes the right digest, the user has not clicked: refused
        err = (await s.call("fea_solve_run", confirmed=short))["__error__"]
        assert "has not confirmed it in the FEA Studio window" in err and stub_solve == []
        assert "has not confirmed" in (await s.call("fea_job_start", kind="solve", confirmed=short))["__error__"]
        # a decision of the agent kind is not accepted in this route either
        sp.record_decision(s.sess, d["digest"], "confirmed", "agent", "agent")
        assert "has not confirmed" in (await s.call("fea_solve_run", confirmed=short))["__error__"]
        # the click in the window
        sp.record_decision(s.sess, d["digest"], "confirmed", "user", "gui")
        ok = await s.call("fea_solve_run", confirmed=short)
        assert ok["ok"] and ok["setup_confirmed_by"] == "user" and stub_solve[-1]["channel"] == "gui"
        # the setup changed after the click: the click does not hold
        await s.call("fea_load_add_force", case_name="Pull", name="Tip", selection="Tip", force=[0, 0, -250])
        assert "has not confirmed" in (await s.call("fea_solve_run"))["__error__"]
        assert "changed since the review" in (await s.call("fea_solve_run", confirmed=short))["__error__"]
        # a rejection is reported as such
        cur = sp.session_digest(s.sess)
        sp.record_decision(s.sess, cur, "rejected", "user", "gui")
        assert "REJECTED" in (await s.call("fea_solve_run", confirmed=cur[:12]))["__error__"]
    run(tmp_path, body)


def test_gate_is_checked_again_under_the_lock(tmp_path, beam_step):
    """A setup that changes between the gate and the calculation (another client) is caught by solve_core's own check."""
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        from fea_core.mcp import tools_setup
        short = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png")))["digest_short"]
        conf = tools_setup.gate(s.sess, short)
        tools_setup.verify(s.sess, conf)                                             # unchanged: passes
        await s.call("fea_load_add_force", case_name="Pull", name="Tip", selection="Tip", force=[0, 0, -999])
        from fea_core.mcp.session import FeaError
        with pytest.raises(FeaError, match="changed after it was confirmed"):
            tools_setup.verify(s.sess, conf)
    run(tmp_path, body)


def test_member_only_study_has_a_review_and_a_digest(tmp_path):
    """A frame without geometry: table with the member loads, a picture of the members (numbers are not drawn there), the digest
    follows the supports and the loads of the members."""
    async def body(s):
        await s.call("fea_study_new", name="Frame", workdir=str(tmp_path / "wd"), material="Ocel S235")
        await s.call("fea_frame_section_set", name="R", shape="RECT", dims={"b": 20, "h": 40})
        await s.call("fea_frame_node_set", nodes=[{"name": "A", "xyz": [0, 0, 0]}, {"name": "B", "xyz": [2000, 0, 0]}])
        await s.call("fea_frame_member_set", name="beam", start="A", end="B", section="R")
        await s.call("fea_frame_support_set", node="A", dofs="fixed")
        await s.call("fea_frame_load_member", case_name="Snow", member="beam", q=[0, 0, -5])
        d = await s.call("fea_setup_review", output_path=str(tmp_path / "frame.png"))
        assert "__error__" not in d, d
        assert "Bar load" in d["table"] and "Snow" in d["table"] and any("Member-only" in n for n in d.get("notes", [])), d
        assert (tmp_path / "frame.png").stat().st_size > 2000
        await s.call("fea_frame_support_set", node="A", dofs="pinned")
        d2 = await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "frame2.png"))
        assert d2["digest"] != d["digest"]                                         # a support of a member system is part of the setup
    run(tmp_path, body)


def test_rest_registry_has_the_tool_and_no_ctx_in_the_schema():
    from fea_core.rest.registry import collect_tools
    tools = collect_tools(Session())
    assert "fea_setup_review" in tools
    for name in ("fea_solve_run", "fea_job_start"):
        props = tools[name].parameters["properties"]
        assert "confirmed" in props and "ctx" not in props, name


def test_results_info_roundtrip(tmp_path):
    """The record in Results.info is saved with the results."""
    import numpy as np
    from fea_core.mesh import Mesh
    from fea_core.results import Results
    info = {"digest": "ab" * 32, "confirmed_by": "user", "channel": "gui", "at": "2026-11-23T10:00:00Z"}
    m = Mesh(nodes=np.zeros((4, 3)), tets=np.array([[0, 1, 2, 3]]), tet_part=np.array([0]), part_names=["P"])
    res = Results(m, [], [], {"setup_confirmation": info, "inp_map": object()})
    res.save(tmp_path / "r.npz")
    assert Results.load(tmp_path / "r.npz").info["setup_confirmation"] == {**info, "read_from_file": True}      # a stamp read from a file says so


@pytest.mark.slow
def test_e2e_confirmed_calculation_is_stamped_and_in_the_report(tmp_path, beam_step):
    """A real calculation after the confirmation: the record is in the results, in the summary and in the Word report."""
    pytest.importorskip("docx")

    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        await s.call("fea_load_add_gravity", case_name="Pull")        # (replaces the same load)
        short = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png")))["digest_short"]
        d = await s.call("fea_solve_run", confirmed=short)
        assert d["ok"] and d["setup_confirmed_by"] == "agent", d
        info = s.sess.results.info["setup_confirmation"]
        assert info["confirmed_by"] == "agent" and info["digest"].startswith(short)
        r = await s.call("fea_report_word", output_path=str(tmp_path / "rep.docx"), images=True, keep_images=True)
        assert Path(r["path"]).is_file()
        assert not [w for w in r["warnings"] if "etup" in w], r["warnings"]            # neither the digest nor the picture failed
        assert any(Path(x).name.startswith("setup_") for x in r["images"]), r["images"]    # the numbered picture is in the report
        from docx import Document
        doc = Document(r["path"])
        text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(c.text for t in doc.tables for row in t.rows for c in row.cells)
        assert "Setup confirmation" in text and short in text and "agent" in text
    run(tmp_path, body)


# ===================================================================== references by name (review fix C1)

class _Names:
    """The names of the four things that other inputs refer to by name."""
    def __init__(self, sel="RefSel", mat="RefAlu", case="RefCase", csys="RefCsys"):
        self.sel, self.mat, self.case, self.csys = sel, mat, case, csys


def _ref_study(n: _Names, install=None):
    from fea_core.study import CoordinateSystem
    st = make_study()
    st.add_selection(n.sel, faces=[face(100.0)])
    st.add_material(Material(n.mat, 70000.0, 0.33, 2.7e-9))
    st.coordinate_systems[n.csys] = CoordinateSystem(n.csys, "cylindrical")
    st.load_cases.append(LoadCase(n.case, [Force("Tip", [5000.0, 0, 0])]))
    if install:
        install(st, n)
    return st


def _install_buckling(st, n):
    st.buckling.enabled, st.buckling.case = True, n.case


def _install_nlbuckling(st, n):
    st.nlbuckling.enabled, st.nlbuckling.case = True, n.case


def _install_prestress(st, n):
    st.dynamics.enabled = True
    st.dynamics.prestress.enabled, st.dynamics.prestress.case = True, n.case


def _install_excitation(st, n):
    from fea_core.dynamics.model import Excitation
    st.dynamics.enabled = True
    st.dynamics.harmonic.enabled = True
    st.dynamics.harmonic.excitation.append(Excitation(kind="force", case=n.case))


def _install_step(st, n):
    from fea_core.contacts.model import LoadStep
    st.analysis.sequence.append(LoadStep("s1", case=n.case))


def _install_motion_target(st, n):
    from fea_core.contacts.model import LoadStep, Motion
    st.analysis.sequence.append(LoadStep("s1", prescribed=[Motion(n.sel)]))


def _install_motion_csys(st, n):
    from fea_core.contacts.model import LoadStep, Motion
    st.analysis.sequence.append(LoadStep("s1", prescribed=[Motion("Tip", csys=n.csys)]))


def _install_creep(st, n):
    st.printing.creep.enabled = True
    st.printing.creep.cases = [n.case]


def _install_frame_material(st, n):
    from fea_core.beams.model import FrameMember
    st.frame.add_node("n1", [0, 0, 0])
    st.frame.add_node("n2", [1000, 0, 0])
    st.frame.members["m1"] = FrameMember("m1", "n1", "n2", "sec", material=n.mat)


def _install_conductance_master(st, n):
    from fea_core.thermal.model import ContactConductance
    st.thermal.conductances.append(ContactConductance(selection="Tip", master=n.sel))


def _install_conductance_slave(st, n):
    from fea_core.thermal.model import ContactConductance
    st.thermal.conductances.append(ContactConductance(selection=n.sel, master="Tip"))


def _install_cyclic(left):
    def install(st, n):
        from fea_core.cyclic import CyclicSymmetry
        st.cyclic = CyclicSymmetry(sectors=8, left=n.sel if left else "Tip", right="Tip" if left else n.sel)
    return install


def _install_mechanism(attr):
    def install(st, n):
        from fea_core.motion.model import Body, MotionJoint
        st.mechanism.bodies += [Body("b1", ["P"]), Body("b2", ["P"])]
        st.mechanism.joints.append(MotionJoint("j1", "revolute", "b1", "b2", **{attr: n.sel}))
    return install


def _install_force_element(attr):
    def install(st, n):
        from fea_core.motion.model import Body, ForceElement
        st.mechanism.bodies += [Body("b1", ["P"]), Body("b2", ["P"])]
        st.mechanism.forces.append(ForceElement("f1", "spring", "b1", "b2", **{attr: n.sel}))
    return install


def _install_contact(side):
    def install(st, n):
        from fea_core.contacts.model import ContactPair
        st.contacts.append(ContactPair("c1", n.sel if side == "a" else "Tip", n.sel if side == "b" else "Tip"))
    return install


def _install_load_csys(st, n):
    st.load_cases[0].loads[0].csys = n.csys


def _install_ortho(st, n):
    st.coordinate_systems[n.csys].kind = "cartesian"
    st.materials[n.mat].ortho_csys = n.csys
    st.part_materials["P"] = n.mat


def _install_mesh_local(st, n):
    st.mesh.local.append({"selection": n.sel, "size": 1.0})


# (label, kind of the referenced item, installer)
_REFERENCES = [
    ("buckling.case", "case", _install_buckling),
    ("nlbuckling.case", "case", _install_nlbuckling),
    ("dynamics prestress case", "case", _install_prestress),
    ("dynamics excitation case", "case", _install_excitation),
    ("sequence step case", "case", _install_step),
    ("printing creep cases", "case", _install_creep),
    ("sequence prescribed motion target", "sel", _install_motion_target),
    ("sequence prescribed motion csys", "csys", _install_motion_csys),
    ("frame member material", "mat", _install_frame_material),
    ("thermal conductance master", "sel", _install_conductance_master),
    ("thermal conductance selection", "sel", _install_conductance_slave),
    ("cyclic left", "sel", _install_cyclic(True)),
    ("cyclic right", "sel", _install_cyclic(False)),
    ("mechanism joint selection_a", "sel", _install_mechanism("selection_a")),
    ("mechanism joint selection_b", "sel", _install_mechanism("selection_b")),
    ("mechanism force selection_a", "sel", _install_force_element("selection_a")),
    ("mechanism force selection_b", "sel", _install_force_element("selection_b")),
    ("contact a", "sel", _install_contact("a")),
    ("contact b", "sel", _install_contact("b")),
    ("load csys", "csys", _install_load_csys),
    ("orthotropic material csys", "csys", _install_ortho),
    ("mesh local refinement selection", "sel", _install_mesh_local),
]


def _mutate_content(st, n, kind):
    if kind == "sel":
        st.selections[n.sel].faces = [face(55.0)]
    elif kind == "mat":
        st.materials[n.mat].E = 7000.0
    elif kind == "csys":
        st.coordinate_systems[n.csys].origin = [1.0, 2.0, 3.0]
    else:                                            # swap the names of the two load cases (the content of each stays)
        a, b = st.load_cases[0], st.load_cases[-1]
        a.name, b.name = b.name, a.name


@pytest.mark.parametrize("label,kind,install", _REFERENCES, ids=[r[0] for r in _REFERENCES])
def test_digest_follows_every_reference_by_name(label, kind, install):
    """A name that an input refers to stands for the content of the item: a rename keeps the digest, a changed item or two swapped
    load case names change it (the load case / selection / material / csys is not otherwise part of this study)."""
    n = _Names()
    ref = _ref_study(n, install)
    bare = _ref_study(n)
    # the reference is really in the study file and is not just one of the ignored sections
    d = ref.to_dict()
    for k in ("selections", "coordinate_systems", "name") + (() if kind == "csys" else ("materials", "load_cases")):
        d.pop(k)
    assert json.dumps(d).count(f'"{getattr(n, kind)}"') >= 1, label
    base = sp.setup_digest(ref)
    assert base != sp.setup_digest(bare)
    # renamed item + reference that follows = the same digest
    other = _Names("Face2", "Mat2", "Case2", "Csys2")
    assert sp.setup_digest(_ref_study(other, install)) == base, label
    # the item changed (a swap of the names of the two load cases for a case) = another digest
    changed = _ref_study(n, install)
    _mutate_content(changed, n, kind)
    assert sp.setup_digest(changed) != base, label
    if kind == "case":                               # without a reference the swap of names is only a rename
        plain = _ref_study(n)
        _mutate_content(plain, n, kind)
        assert sp.setup_digest(plain) == sp.setup_digest(bare)


def test_digest_name_equal_to_a_fixed_choice_is_not_a_reference():
    """The bolt category "A" and the steel grade "S235" are choices, not names: a selection / material of that name may be renamed."""
    assert sp.setup_digest(make_study(sel="A")) == sp.setup_digest(make_study(sel="Support"))
    c = make_study()
    c.add_material(Material("S235", 210000.0, 0.3, 7.85e-9))
    d = make_study()
    d.add_material(Material("Grade", 210000.0, 0.3, 7.85e-9))
    assert sp.setup_digest(c) == sp.setup_digest(d)


# ===================================================================== review fixes I1, I2, I3

_GATED_TOOLS = [
    ("fea_mesh_convergence", {}),
    ("fea_param_study_run", {"parameters": [{"kind": "load_factor", "values": [1.0, 2.0]}]}),
    ("fea_sweep_run", {"background": False}),
    ("fea_submodel_run", {}),
    ("fea_topo_opt_run", {}),
]


@pytest.mark.parametrize("tool,args", _GATED_TOOLS, ids=[t[0] for t in _GATED_TOOLS])
def test_every_calculating_tool_is_gated(tmp_path, beam_step, tool, args):
    """The same gate as fea_solve_run stands before the other tools that start solver runs (nothing is computed without it)."""
    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        if tool == "fea_sweep_run":
            await s.call("fea_sweep_define", load_factors=[1.0, 2.0])
        err = (await s.call(tool, **args))["__error__"]
        assert "requires the user's confirmation" in err, err
        err = (await s.call(tool, confirmed="0" * 12, **args))["__error__"]
        assert "digest does not match" in err, err
        assert s.sess.confirmation is None
    run(tmp_path, body)


def test_gated_tools_hide_ctx_from_the_window():
    """The window gets the arguments without the MCP context, for every gated tool."""
    from fea_core.mcp import tools_setup
    seen = {}

    class Reg:
        def tool(self, **kw):
            def deco(fn):
                seen[fn.__name__] = fn
                return fn
            return deco
    reg = Reg()
    from fea_core.mcp import tools_advanced, tools_mesh, tools_param, tools_results, tools_submodel
    sess = Session(Path("."))
    for mod in (tools_mesh, tools_param, tools_advanced, tools_results, tools_submodel):
        mod.register(reg, sess)
    for name, _ in _GATED_TOOLS:
        assert seen[name].ipc_args({"a": 1, "ctx": object(), "confirmed": "x"}) == {"a": 1, "confirmed": "x"}, name
    assert tools_setup.strip_ctx({"ctx": 1}) == {}


def test_failed_elicitation_is_an_error_not_the_agent_fallback(tmp_path, beam_step, stub_solve):
    """The client declared the elicitation capability and the request failed: the calculation does not start and the digest the agent
    passed back is not taken as the user's answer."""
    async def failing(ctx, params):
        raise RuntimeError("the form could not be shown")

    async def body(s):
        await make_beam(s, beam_step, tmp_path)
        short = (await s.call("fea_setup_review", return_image=False, output_path=str(tmp_path / "r.png")))["digest_short"]
        err = (await s.call("fea_solve_run", confirmed=short))["__error__"]
        assert "could not be asked" in err and "NOT started" in err, err
        assert stub_solve == [] and s.sess.confirmation is None
    run(tmp_path, body, elicit=failing)


def test_ask_user_without_the_capability_is_the_fallback():
    from fea_core.mcp import tools_setup
    assert tools_setup.ask_user(None, "x") == "unsupported"


def test_report_uses_only_the_stamp_in_the_results():
    """The decision of the session belongs to the last setup asked about, not to the setup that was calculated."""
    from types import SimpleNamespace
    from fea_core.report import setup_confirm as scf
    sess = SimpleNamespace(confirmation={"digest": "ab" * 32, "decision": "confirmed", "by": "user", "channel": "gui", "at": "t"})
    assert scf.record_of(sess, SimpleNamespace(info={})) is None
    assert not scf.has_confirmation(sess, SimpleNamespace(info={}))
    assert scf.record_of(sess, None) is None
    info = {"digest": "cd" * 32, "confirmed_by": "agent", "channel": "agent", "at": "t"}
    assert scf.record_of(sess, SimpleNamespace(info={"setup_confirmation": info})) == info


def test_stamp_read_from_a_file_is_marked(tmp_path):
    import numpy as np
    from fea_core.mesh import Mesh
    from fea_core.results import Results
    info = {"digest": "ab" * 32, "confirmed_by": "user", "channel": "gui", "at": "2026-11-23T10:00:00Z"}
    m = Mesh(nodes=np.zeros((4, 3)), tets=np.array([[0, 1, 2, 3]]), tet_part=np.array([0]), part_names=["P"])
    res = Results(m, [], [], {"setup_confirmation": info})
    assert "read_from_file" not in res.info["setup_confirmation"]               # made in this session
    res.save(tmp_path / "r.npz")
    assert Results.load(tmp_path / "r.npz").info["setup_confirmation"] == {**info, "read_from_file": True}

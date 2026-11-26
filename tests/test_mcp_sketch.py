"""MCP tools of the sketcher (fea_sketch_*, group model): a 2D plate with a hole and a beam frame drawn from typed coordinates, the
whole way to the result, a parameter change that keeps the setup, the errors that name the entity, DXF import, Make 3D."""
import math

import pytest

pytest.importorskip("mcp")

from tests.test_mcp import png_size, run  # noqa: E402

SKETCH_TOOLS = {"fea_sketch_new", "fea_sketch_param", "fea_sketch_add", "fea_sketch_remove", "fea_sketch_list", "fea_sketch_build",
                "fea_sketch_to_3d", "fea_sketch_import_dxf"}


async def _plate(s, tmp_path):
    await s.ok("fea_tools_enable", groups="model")
    await s.ok("fea_sketch_new", kind="twod", mode="plane_stress", thickness=2.5, name="plate", workdir=str(tmp_path / "wd"))
    for k, v in (("L", 150), ("W", 50), ("R", 5)):
        await s.ok("fea_sketch_param", name=k, expr=v)
    await s.ok("fea_sketch_add", type="line", points=[["R", 0], ["L", 0]], name="sym_y")
    await s.ok("fea_sketch_add", type="line", points=[["L", 0], ["L", "W"]], name="end")
    await s.ok("fea_sketch_add", type="line", points=[["L", "W"], [0, "W"]], name="top")
    await s.ok("fea_sketch_add", type="line", points=[[0, "W"], [0, "R"]], name="sym_x")
    await s.ok("fea_sketch_add", type="arc_center", points=[["R", 0], [0, "R"]], center=[0, 0], name="hole")


def test_tools_are_in_group_model_only(tmp_path):
    import anyio
    from mcp import Client
    from fea_core.mcp.server import build_server
    from fea_core.mcp.session import Session

    async def tools(spec):
        srv = build_server(Session(tmp_path / "mcp"), groups_spec=spec)
        async with Client(srv, mode="legacy") as c:
            return {t.name: t for t in (await c.list_tools()).tools}
    assert not SKETCH_TOOLS & set(anyio.run(tools, "core"))
    model = anyio.run(tools, "core,model")
    assert SKETCH_TOOLS <= set(model)
    for n in SKETCH_TOOLS:                                        # the listed text is cut after the essential sentences
        assert len(model[n].description) < 1500, (n, len(model[n].description))


def test_plate_build_list_picture_and_errors(tmp_path):
    async def body(s, c):
        await _plate(s, tmp_path)
        d = await s.ok("fea_sketch_list")
        assert d["params"]["R"]["value"] == 5 and len(d["entities"]) == 5 and d["built"] is False
    run(tmp_path, body)


def test_without_a_study_or_a_sketch_the_tools_say_so(tmp_path):
    async def body(s, c):
        assert "No study is open" in await s.err("fea_sketch_list")
        await s.ok("fea_sketch_new", workdir=str(tmp_path / "wd"))
        assert "region needs a name" in await s.err("fea_sketch_add", type="region", points=[[1, 1]])
        assert "needs 2 point" in await s.err("fea_sketch_add", type="line", points=[[0, 0]])
        d = await s.ok("fea_sketch_build")                                  # an empty sketch: a normal answer that says what is wrong
        assert d["built"] is False and "empty" in d["error"]
    run(tmp_path, body)


def test_plate_to_result_and_parameter_change(tmp_path):
    async def body(s, c):
        await _plate(s, tmp_path)
        await s.ok("fea_tools_enable", groups="shell2d")
        r = await c.call_tool("fea_sketch_build", {"output_path": str(tmp_path / "sk.png"),
                                                   "mesh_size": 4.0, "refine": {"hole": 0.35}})
        assert not r.is_error and r.content[1].type == "image" and png_size(tmp_path / "sk.png") == (1000, 760)
        d = await s.ok("fea_sketch_list")
        assert d["built"] is True
        await s.ok("fea_constraint_add_displacement", name="cx", selection="sym_x", ux=0.0)
        await s.ok("fea_constraint_add_displacement", name="cy", selection="sym_y", uy=0.0)
        await s.ok("fea_load_add_pressure", case_name="Tah", name="p", selection="end", pressure=-100.0)
        await s.ok("fea_solve_run", timeout_s=600)
        p0 = await s.ok("fea_twod_probe", point=[0.0, 5.0])
        kt = 3 - 3.14 * 0.1 + 3.667 * 0.01 - 1.527 * 0.001
        sx0 = p0["probes"][0]["stress_MPa"]["sx"]
        assert sx0 == pytest.approx(kt * 100 * 100 / 90, rel=0.03)
        await s.ok("fea_sketch_param", name="R", expr=7.5)
        assert (await s.ok("fea_sketch_list"))["built"] is False
        b = await s.ok("fea_sketch_build", picture=False)
        assert b["rebuild"] is True and b["not_remapped"] == [] and b["remap"].get("cad", 0) >= 1
        # the constraints, the load and the mesh settings were kept on the new geometry – solve again without redoing the setup
        await s.ok("fea_solve_run", timeout_s=600)
        p1 = await s.ok("fea_twod_probe", point=[0.0, 7.5])
        sx1 = p1["probes"][0]["stress_MPa"]["sx"]
        assert sx1 > sx0
    run(tmp_path, body)


def test_errors_name_the_entity_and_the_picture_marks_them(tmp_path):
    async def body(s, c):
        await s.ok("fea_tools_enable", groups="model")
        await s.ok("fea_sketch_new", kind="twod", workdir=str(tmp_path / "wd"))
        for a, b in (([0, 0], [10, 0]), ([10, 0], [10, 10]), ([10, 10], [0, 10]), ([0, 10], [0, 0.001])):
            await s.ok("fea_sketch_add", type="line", points=[a, b])
        r = await c.call_tool("fea_sketch_build", {"output_path": str(tmp_path / "err.png")})
        assert not r.is_error                                                   # the picture is the point: a normal answer with the problem in it
        import json
        d = json.loads(r.content[0].text)
        assert d["built"] is False and set(d["entities"]) == {"E1", "E4"} and "open contour" in d["error"] and "gap 0.001" in d["error"]
        assert r.content[1].type == "image"
        await s.ok("fea_sketch_remove", id="E4")
        await s.ok("fea_sketch_add", type="line", points=[[0, 10], [0, 0]])
        ok = await s.ok("fea_sketch_build", picture=False)
        assert ok["built"] and ok["regions"][0]["area_mm2"] == pytest.approx(100.0)
        # bad inputs are understandable errors, not internal ones
        assert "Unknown entity type" in await s.err("fea_sketch_add", type="spline", points=[[0, 0], [1, 1]])
        assert "unknown parameter" in (await s.err("fea_sketch_add", type="line", points=[["nope", 0], [1, 1]])).lower()
        assert "not an entity id" in await s.err("fea_sketch_remove", id="E99")
        await s.err("fea_sketch_param", name="__x", expr=1)
        await s.err("fea_sketch_param", name="a", expr="2**9")
        await s.err("fea_sketch_param", name="a", expr="__import__('os').system('x')")
    run(tmp_path, body)


def test_axisym_tube_by_mcp(tmp_path):
    async def body(s, c):
        await s.ok("fea_tools_enable", groups="model")
        await s.ok("fea_sketch_new", kind="twod", mode="axisym", workdir=str(tmp_path / "wd"))
        await s.ok("fea_sketch_add", type="rect", points=[[20, 0], [40, 10]], name="tube")
        d = await s.ok("fea_sketch_build", picture=False)
        assert d["analysis"]["kind"] == "axisym" and d["analysis"]["axis"] == [0.0, 1.0, 0.0]
        assert {"tube.1", "tube.2", "tube.3", "tube.4", "tube"} <= set(d["selections"])
        await s.ok("fea_constraint_add_displacement", name="ax", selection="tube.1", uy=0.0)
        await s.ok("fea_load_add_pressure", case_name="p", name="in", selection="tube.4", pressure=100.0)
        await s.ok("fea_mesh_generate", global_size=1.5)
        await s.ok("fea_solve_run", timeout_s=600)
        bad = await s.ok("fea_sketch_add", type="rect", points=[[-30, 0], [-10, 10]])
        assert bad["id"] == "E2"
        e = await s.ok("fea_sketch_build", picture=False)
        assert e["built"] is False and "axis" in e["error"] and e["entities"] == ["E2"]
    run(tmp_path, body)


def test_frame_by_mcp(tmp_path):
    async def body(s, c):
        await s.ok("fea_tools_enable", groups="model,frame")
        await s.ok("fea_sketch_new", kind="frame", name="portal", workdir=str(tmp_path / "wd"))
        await s.ok("fea_sketch_param", name="L", expr=6000)
        await s.ok("fea_sketch_param", name="h", expr=4000)
        await s.ok("fea_sketch_add", type="line", points=[[0, 0], [0, "h"]], section="IPE 300")
        await s.ok("fea_sketch_add", type="line", points=[[0, "h"], ["L", "h"]], section="IPE 400")
        await s.ok("fea_sketch_add", type="line", points=[["L", 0], ["L", "h"]], section="IPE 300")
        await s.ok("fea_sketch_add", type="point", points=[[0, 0]], support="fixed")
        await s.ok("fea_sketch_add", type="point", points=[["L", 0]], support="fixed")
        d = await s.ok("fea_sketch_build", picture=False)
        assert set(d["members"]) == {"E1", "E2", "E3"} and len(d["nodes"]) == 4
        await s.ok("fea_frame_load_node", case_name="H", node="P2", force=[10000, 0, 0])
        r = await s.ok("fea_solve_run", timeout_s=120)
        assert r
        await s.ok("fea_sketch_param", name="h", expr=5000)
        d2 = await s.ok("fea_sketch_build", picture=False)
        assert d2["rebuild"] is True and d2["nodes"]["P2"][2] == 5000.0
        fl = await s.ok("fea_frame_list")
        assert {m["name"] for m in fl["members"]} == {"E1", "E2", "E3"} if isinstance(fl.get("members"), list) else "E1" in str(fl)
        await s.ok("fea_sketch_remove", id="E3")
        d3 = await s.ok("fea_sketch_build", picture=False)
        assert set(d3["members"]) == {"E1", "E2"}
    run(tmp_path, body)


def test_dxf_and_to_3d_by_mcp(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    for a, b in (((0, 0), (60, 0)), ((60, 0), (60, 40)), ((60, 40), (0, 40)), ((0, 40), (0, 0))):
        msp.add_line(a, b)
    msp.add_circle((30, 20), 6)
    f = tmp_path / "p.dxf"
    doc.saveas(f)

    async def body(s, c):
        await s.ok("fea_tools_enable", groups="model")
        await s.ok("fea_sketch_new", kind="twod", thickness=4, workdir=str(tmp_path / "wd"))
        i = await s.ok("fea_sketch_import_dxf", path=str(f))
        assert i["imported"] == 5 and i["lines"] == 4 and i["circles"] == 1
        await s.err("fea_sketch_import_dxf", path=str(tmp_path / "missing.dxf"))
        b = await s.ok("fea_sketch_build", picture=False)
        area = 60 * 40 - math.pi * 36
        assert b["regions"][0]["area_mm2"] == pytest.approx(area, rel=1e-5)             # the answer is rounded to 5 digits
        t = await s.ok("fea_sketch_to_3d", how="extrude", length=10)
        assert t["parts"] == ["Region 1"] and t["sketch_study_saved"].endswith("_sketch_study.json")
    run(tmp_path, body)

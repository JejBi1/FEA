"""Sketcher (fea_core.sketch, F15-S5): expressions, the model, regions, exact arcs against OCP, stable ids, the rebuild with the
settings kept, the beam system, DXF import, errors that name the entity. No solver is run here (the verification cases SK1–SK5 do that)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from fea_core.sketch.expr import SketchError, eval_params, evaluate, names_in
from fea_core.sketch.model import Sketch, new_sketch, resolve

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ expressions
def test_expressions_numbers_functions_parameters():
    assert evaluate("2*(3+4)") == 14
    assert evaluate("sqrt(16)+sin(pi/2)") == pytest.approx(5.0)
    assert evaluate("L/10", {"L": 200.0}) == 20.0
    assert evaluate("2**3") == 8
    assert evaluate("rad(180)") == pytest.approx(math.pi)
    assert evaluate("1,5") == 1.5                       # a decimal comma typed by a Czech user
    assert eval_params({"L": 200, "h": "L/10", "w": "h*2"}) == {"L": 200.0, "h": 20.0, "w": 40.0}
    assert names_in("a + sin(b) * pi") == {"a", "b"}


@pytest.mark.parametrize("bad", ["__import__('os')", "a.b", "[1,2]", "x[0]", "2**9", "1/0", "lambda: 1", "open('x')", "'a'+'b'", "",
                                 "1e999", "sqrt(-1)", "foo(1)", "(1).real", "2 if 1 else 3", "x"])
def test_expressions_refuse(bad):
    with pytest.raises(SketchError):
        evaluate(bad, {})


def test_parameter_cycles_and_names():
    with pytest.raises(SketchError, match="cycle"):
        eval_params({"a": "b+1", "b": "a+1"})
    with pytest.raises(SketchError, match="unknown parameter"):
        eval_params({"a": "zz"})
    sk = new_sketch("twod")
    for name in ("1x", "sin", "pi", "a b", "", "x" * 40):
        with pytest.raises(SketchError):
            sk.set_param(name, 1)


# ------------------------------------------------------------------ model
def _plate(d=20.0, w=100.0):
    sk = new_sketch("twod", "plane_stress", thickness=5)
    sk.set_param("W", w)
    sk.set_param("d", d)
    sk.add_entity("rect", [[0, 0], ["W", "W"]], name="plate")
    sk.add_entity("circle", center=["W/2", "W/2"], radius="d/2", name="hole")
    return sk


def test_ids_are_stable_and_never_reused():
    sk = _plate()
    ids = [e.id for e in sk.entities]
    assert ids == ["E1", "E2"]
    sk.remove_entity("E1")
    e = sk.add_entity("line", [[0, 0], [1, 1]])
    assert e.id == "E3"                                           # E1 is not given out again
    d = sk.to_dict()
    sk2 = Sketch.from_dict(json.loads(json.dumps(d)))
    assert sk2 is not None and sk2.to_dict() == d
    assert sk2.add_entity("line", [[2, 2], [3, 3]]).id == "E4"


def test_same_expressions_are_one_point_and_failed_add_leaves_nothing():
    sk = new_sketch("twod")
    a = sk.add_entity("line", [[0, 0], [10, 0]])
    b = sk.add_entity("line", [[10, 0], [10, 10]])
    assert a.points[1] == b.points[0]
    n = len(sk.points)
    with pytest.raises(SketchError):
        sk.add_entity("line", [[5, 5], ["nope", 1]])
    assert len(sk.points) == n


def test_remove_param_in_use_is_refused():
    sk = _plate()
    with pytest.raises(SketchError, match="used by"):
        sk.remove_param("W")
    sk.remove_entity("E1")
    sk.remove_entity("E2")
    sk.remove_param("W")


def test_snapping_merges_points_that_differ_by_rounding():
    sk = new_sketch("twod")
    sk.add_entity("line", [[0, 0], [0.1 + 0.2, 0]])
    sk.add_entity("line", [[0.3, 0], [0.3, 1]])
    res = resolve(sk)
    assert res.segs[0].p1 == res.segs[1].p0


def test_study_file_roundtrip_and_old_programs_warn():
    from fea_core.study import Study
    st = Study("x", "")
    st.sketch = _plate()
    d = st.to_dict()
    assert "sketch" in d["features"] and d["sketch"]["kind"] == "twod"
    st2 = Study.from_dict(json.loads(json.dumps(d)))
    assert st2.sketch is not None and st2.sketch.to_dict() == st.sketch.to_dict()
    assert "sketch" not in Study("y", "").to_dict()                  # written only when used
    bad = dict(d, sketch={"kind": "twod", "entities": [{"id": "E1", "type": "line", "points": ["P9", "P8"]}]})
    from fea_core.study import StudyFormatError
    with pytest.raises(StudyFormatError):
        Study.from_dict(bad)


# ------------------------------------------------------------------ regions and exact geometry (OCP)
def test_plate_with_hole_exact_area_and_selections(tmp_path):
    from fea_core.sketch.build import build_study
    st, b = build_study(_plate(), tmp_path, "plate")
    assert [r["name"] for r in b.regions] == ["Region 1"]
    area = 100.0 * 100.0 - math.pi * 10.0 ** 2
    assert b.regions[0]["area_mm2"] == pytest.approx(area, rel=1e-12)
    assert b.parts[0].volume() == pytest.approx(area * 5.0, rel=1e-9)
    assert set(st.selections) >= {"plate", "plate.1", "plate.2", "plate.3", "plate.4", "hole"}
    assert st.selections["hole"].faces[0].type == "cylinder" and st.selections["hole"].faces[0].area == pytest.approx(2 * math.pi * 10 * 5)
    assert len(st.selections["plate"].faces) == 4
    assert st.twod.enabled and st.twod.kind == "plane_stress" and st.twod.thickness == 5.0 and st.geometry.endswith(".step")


def test_arcs_are_exact_not_polylines(tmp_path):
    """A quarter-circle notch: the area equals the analytic value (a polyline would be 1e-5 off)."""
    from fea_core.sketch.build import build_geometry
    sk = new_sketch("twod", thickness=2)
    sk.add_entity("line", [[0, 0], [50, 0]])
    sk.add_entity("line", [[50, 0], [50, 50]])
    sk.add_entity("line", [[50, 50], [10, 50]])
    sk.add_entity("arc3", [[10, 50], [10 - 10 * math.sin(math.pi / 4), 50 - 10 + 10 * math.cos(math.pi / 4)], [0, 40]])
    sk.add_entity("line", [[0, 40], [0, 0]])
    b = build_geometry(sk, tmp_path)
    # the corner (0, 50) is cut by a quarter circle of radius 10 around (10, 40) – the quarter disc is outside the region?
    # the arc goes from (10, 50) over the middle to (0, 40): the centre is (10, 40), the region keeps the square minus the corner piece
    area = b.regions[0]["area_mm2"]
    assert area == pytest.approx(50 * 50 - (10 * 10 - math.pi * 100 / 4), rel=1e-12)
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(b.parts[0].shape, p)
    assert p.Mass() == pytest.approx(area * 2.0, rel=1e-9)


def test_chord_splits_a_region_and_the_hole_stays_a_hole(tmp_path):
    from fea_core.sketch.build import build_geometry
    sk = _plate()
    sk.add_entity("line", [[0, "W/2"], ["W", "W/2"]], name="cut")
    b = build_geometry(sk, tmp_path)
    assert len(b.regions) == 2                                   # the two half discs of the hole are NOT regions
    assert sum(r["area_mm2"] for r in b.regions) == pytest.approx(100 * 100 - math.pi * 100, rel=1e-9)
    cut = b.selections["cut"]                                     # the chord is a face on both parts, in two pieces (the part inside the hole has none)
    assert len(cut) == 4 and sum(f.area for f in cut) == pytest.approx(2 * 80 * 5, rel=1e-9)


def test_hole_override_gives_a_part_of_its_own(tmp_path):
    from fea_core.sketch.build import build_geometry
    sk = _plate()
    sk.entity("E2").hole = False                                 # a pin in the hole = a second part
    b = build_geometry(sk, tmp_path)
    assert len(b.regions) == 2
    assert sorted(round(r["area_mm2"], 3) for r in b.regions) == sorted([round(math.pi * 100, 3), round(10000 - math.pi * 100, 3)])


def test_region_names_by_seed_point(tmp_path):
    from fea_core.sketch.build import build_geometry
    from fea_core.sketch.model import SketchRegion
    sk = _plate()
    sk.entity("E2").hole = False
    sk.regions.append(SketchRegion(["W/2", "W/2"], "Pin", "Alloy"))
    sk.regions.append(SketchRegion([2, 2], "Plate"))
    b = build_geometry(sk, tmp_path)
    assert {r["name"] for r in b.regions} == {"Pin", "Plate"}
    with pytest.raises(SketchError, match="not inside"):
        sk.regions.append(SketchRegion([500, 500], "Nowhere"))
        build_geometry(sk, tmp_path)


def test_axisym_revolve_volume_and_axis_error(tmp_path):
    from fea_core.sketch.build import build_geometry
    sk = new_sketch("twod", "axisym")
    sk.add_entity("rect", [[20, 0], [40, 10]])
    b = build_geometry(sk, tmp_path)
    assert b.parts[0].volume() == pytest.approx(math.pi * (40 ** 2 - 20 ** 2) * 10, rel=1e-9)
    assert b.twod.kind == "axisym" and b.twod.axis == [0.0, 1.0, 0.0]
    bad = new_sketch("twod", "axisym")
    bad.add_entity("rect", [[-10, 0], [10, 10]])
    with pytest.raises(SketchError, match="axis") as ei:
        build_geometry(bad, tmp_path)
    assert ei.value.entities == ["E1"]


def test_errors_name_the_entity(tmp_path):
    from fea_core.sketch.build import build_geometry
    open_loop = new_sketch("twod")
    for a, b in (([0, 0], [10, 0]), ([10, 0], [10, 10]), ([10, 10], [0, 10]), ([0, 10], [0, 0.001])):
        open_loop.add_entity("line", [a, b])
    with pytest.raises(SketchError, match="open contour") as ei:
        build_geometry(open_loop, tmp_path)
    assert set(ei.value.entities) == {"E1", "E4"}               # the two ends of the gap
    assert "gap 0.001" in str(ei.value)
    bow = new_sketch("twod")
    bow.add_entity("polyline", [[0, 0], [10, 10], [10, 0], [0, 10]], closed=True)
    with pytest.raises(SketchError, match="crosses itself") as ei:
        build_geometry(bow, tmp_path)
    assert ei.value.entities == ["E1"]
    empty = new_sketch("twod")
    with pytest.raises(SketchError, match="empty"):
        build_geometry(empty, tmp_path)
    flat = new_sketch("twod")
    flat.add_entity("line", [[0, 0], [10, 0]])
    flat.add_entity("line", [[10, 0], [0, 0.0]])
    with pytest.raises(SketchError):
        build_geometry(flat, tmp_path)
    col = new_sketch("twod")
    with pytest.raises(SketchError, match="line"):
        col.add_entity("arc3", [[0, 0], [1, 1], [2, 2]])
        resolve(col)


# ------------------------------------------------------------------ rebuild keeps the study (the "regeneration remap")
def test_rebuild_keeps_selections_loads_materials_and_reports_losses(tmp_path):
    from fea_core.sketch.build import build_study, is_built, rebuild_study
    from fea_core.study import Displacement, LoadCase, Pressure
    st, _ = build_study(_plate(), tmp_path, "plate")
    st.constraints.append(Displacement("plate.1", uy=0.0))
    st.load_cases.append(LoadCase("p", [Pressure("hole", 50.0)]))
    st.mesh.local = [{"selection": "hole", "size": 1.0}]
    before = {k: [f.area for f in s.faces] for k, s in st.selections.items()}
    st.sketch.set_param("d", 30)
    assert not is_built(st.sketch)
    new, rep, b = rebuild_study(st, tmp_path)
    assert is_built(new.sketch) and rep.blocking() == []
    assert new.constraints[0].selection == "plate.1" and new.load_cases[0].loads[0].selection == "hole"
    assert new.mesh.local == [{"selection": "hole", "size": 1.0}] and new.twod.thickness == 5.0
    assert [s.status for s in rep.selections if s.name == "hole"] == ["cad"]       # by the entity id, not by guessing
    assert new.selections["hole"].faces[0].area == pytest.approx(2 * math.pi * 15 * 5)
    assert new.selections["plate.1"].faces[0].area == pytest.approx(before["plate.1"][0] * 1.0)
    assert new.cad_link is not None and new.cad_link.iterations[-1].trigger == "sketch"
    assert b.parts[0].volume() == pytest.approx((10000 - math.pi * 225) * 5, rel=1e-9)
    # a parameter given to the rebuild instead of editing the sketch first
    new2, rep2, _ = rebuild_study(new, tmp_path, params={"W": 120})
    assert new2.sketch.params["W"] == 120 and rep2.blocking() == []
    with pytest.raises(SketchError, match="Unknown sketch parameter"):
        rebuild_study(new2, tmp_path, params={"zz": 1})
    # an entity removed: the selection that used it is reported, the study keeps its old selector
    sk3 = new2.sketch.copy()
    sk3.remove_entity("E2")
    new3, rep3, _ = rebuild_study(new2, tmp_path, sketch=sk3)
    assert "hole" in rep3.blocking()


def test_two_regions_keep_their_parts_when_a_parameter_moves(tmp_path):
    from fea_core.sketch.build import build_study, rebuild_study
    sk = new_sketch("twod", thickness=2)
    sk.set_param("a", 40)
    sk.add_entity("rect", [[0, 0], ["a", 20]], name="left")
    sk.add_entity("rect", [["a", 0], [100, 20]], name="right")
    sk.entity("E1").hole = None
    st, b = build_study(sk, tmp_path, "two")
    assert len(b.regions) == 2
    names = [r["name"] for r in b.regions]
    new, rep, b2 = rebuild_study(st, tmp_path, params={"a": 60})
    assert [r["name"] for r in b2.regions] == names
    assert rep.blocking() == [] and all(p.old == p.new for p in rep.parts)


# ------------------------------------------------------------------ beams
def test_frame_nodes_members_supports_and_split(tmp_path):
    from fea_core.sketch.build import build_study
    sk = new_sketch("frame")
    sk.set_param("L", 4000)
    sk.add_entity("line", [[0, 0], ["L", 0]])
    sk.add_support([0, 0], "pinned")
    sk.add_support(["L", 0], "roller_x")
    sk.add_support(["L/2", 0], "node")                             # a node at mid-span for a load
    st, b = build_study(sk, tmp_path, "beam")
    fr = st.frame
    assert set(fr.members) == {"E1_P1", "E1_P3"} and set(fr.nodes) == {"P1", "P2", "P3"}
    assert fr.nodes["P3"].xyz == [2000.0, 0.0, 0.0]               # the plane is XZ by default: sketch y = global Z
    sup = {s.node: s.dofs for s in fr.supports}
    assert sup["P1"][:3] == [True, True, True] and sup["P2"][2] is True and sup["P2"][0] is False
    assert all(d[1] and d[3] and d[5] for d in sup.values())       # out of the plane (Y): uy, rx, rz
    assert fr.members["E1_P1"].up == [0.0, 0.0, 1.0]


def test_frame_arc_circle_and_crossing_split():
    from fea_core.sketch.frame import build_frame
    sk = new_sketch("frame")
    sk.add_entity("arc3", [[0, 0], [1000, 1000], [2000, 0]])
    sk.add_entity("circle", center=[5000, 0], radius=500)
    fb = build_frame(sk)
    arc = fb.members["E1"]
    assert arc["up"] == "radial" and arc["through"] == pytest.approx([1000.0, 0.0, 1000.0])
    assert {"E2_t000", "E2_t500"} == {k for k in fb.members if k.startswith("E2")}
    x = new_sketch("frame")
    x.frame["split"] = True
    x.add_entity("line", [[0, 0], [10, 10]])
    x.add_entity("line", [[0, 10], [10, 0]])
    fbx = build_frame(x)
    assert len(fbx.members) == 4 and "X_E1_E2" in fbx.nodes


def test_frame_rebuild_keeps_user_settings_and_removes_what_is_gone(tmp_path):
    from fea_core.beams.model import FrameMemberLoad, FrameNodeLoad
    from fea_core.study import LoadCase
    from fea_core.sketch.build import build_study, rebuild_study
    sk = new_sketch("frame")
    sk.set_param("h", 3000)
    sk.add_entity("line", [[0, 0], [0, "h"]])
    sk.add_entity("line", [[0, "h"], [4000, "h"]], section="IPE 300")
    sk.add_support([0, 0], "fixed")
    st, _ = build_study(sk, tmp_path, "frame")
    st.frame.members["E1"].releases_end = [False, False, False, False, True, False]
    st.frame.members["E1"].section = "HEA 200"
    st.frame.sections["HEA 200"] = type(st.frame.sections["IPE 200"])("HEA 200", library="HEA 200")
    st.load_cases.append(LoadCase("c", [FrameNodeLoad("P2", [100, 0, 0]), FrameMemberLoad("E2", [0, 0, -1.0])]))
    new, rep, b = rebuild_study(st, tmp_path, params={"h": 3500})
    assert rep is None
    assert new.frame.nodes["P2"].xyz == [0.0, 0.0, 3500.0]
    assert new.frame.members["E1"].releases_end[4] and new.frame.members["E1"].section == "HEA 200"
    assert len(new.load_cases[0].loads) == 2
    sk2 = new.sketch.copy()
    sk2.remove_entity("E2")
    new2, _, b2 = rebuild_study(new, tmp_path, sketch=sk2)
    assert set(new2.frame.members) == {"E1"} and "P3" not in new2.frame.nodes
    assert any("E2" in w for w in b2.warnings)                      # the member load that points at the removed member


# ------------------------------------------------------------------ Make 3D
def test_extrude_and_revolve_give_a_normal_3d_study(tmp_path):
    from fea_core.sketch.build import extrude_to_3d
    st, parts, step = extrude_to_3d(_plate(), tmp_path, "extrude", length=30)
    assert parts[0].volume() == pytest.approx((10000 - math.pi * 100) * 30, rel=1e-9)
    assert st.twod.enabled is False and st.sketch is None and "hole" in st.selections
    sk = new_sketch("twod")
    sk.add_entity("rect", [[10, 0], [20, 30]])
    st2, parts2, _ = extrude_to_3d(sk, tmp_path, "revolve", angle=90)
    assert parts2[0].volume() == pytest.approx(math.pi * (400 - 100) * 30 / 4, rel=1e-9)
    with pytest.raises(SketchError):
        extrude_to_3d(sk, tmp_path, "extrude")


# ------------------------------------------------------------------ DXF (module "sketch" = ezdxf; skipped when it is not installed)
def test_dxf_import_lines_arcs_circles_and_units(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from fea_core.sketch.build import build_geometry
    from fea_core.sketch.dxf import import_dxf
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((100, 0), (100, 60))
    msp.add_line((100, 60), (20, 60))
    msp.add_arc((20, 40), 20, 90, 180)                              # from (20, 60) to (0, 40)
    msp.add_line((0, 40), (0, 0))
    msp.add_circle((60, 30), 8)
    msp.add_spline([(0, 0), (1, 1), (2, 0)])
    msp.add_text("hello")
    f = tmp_path / "part.dxf"
    doc.saveas(f)
    sk = new_sketch("twod", thickness=3)
    info = import_dxf(sk, f)
    assert info["imported"] == 6 and info["lines"] == 4 and info["arcs"] == 1 and info["circles"] == 1
    assert info["skipped"] == {"SPLINE": 1, "TEXT": 1} and "Not imported" in info["warnings"][0]
    b = build_geometry(sk, tmp_path / "w")
    area = 100 * 60 - (20 * 20 - math.pi * 400 / 4) - math.pi * 64
    assert b.regions[0]["area_mm2"] == pytest.approx(area, rel=1e-6)
    inch = ezdxf.new("R2010", setup=False)
    inch.header["$INSUNITS"] = 1
    inch.modelspace().add_line((0, 0), (1, 0))
    g = tmp_path / "inch.dxf"
    inch.saveas(g)
    sk2 = new_sketch("twod")
    import_dxf(sk2, g)
    assert resolve(sk2).segs[0].length() == pytest.approx(25.4)
    with pytest.raises(SketchError, match="no lines"):
        empty = ezdxf.new("R2010", setup=False)
        h = tmp_path / "empty.dxf"
        empty.saveas(h)
        import_dxf(new_sketch("twod"), h)
    bad = tmp_path / "bad.dxf"
    bad.write_text("this is not a DXF file", encoding="utf-8")
    with pytest.raises(SketchError, match="cannot be read"):
        import_dxf(new_sketch("twod"), bad)


def test_dxf_lwpolyline_bulge_and_layers(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from fea_core.sketch.dxf import import_dxf
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0, 0), (10, 0, 1), (10, 10, 0), (0, 10, 0)], format="xyb", close=True, dxfattribs={"layer": "A"})
    msp.add_line((0, 0), (1, 1), dxfattribs={"layer": "B"})
    f = tmp_path / "lw.dxf"
    doc.saveas(f)
    sk = new_sketch("twod")
    info = import_dxf(sk, f, layers=["A"])
    assert info["arcs"] == 1 and info["lines"] == 3 and info["layers"] == ["A"]
    arcs = [s for s in resolve(sk).segs if s.kind == "arc"]
    assert len(arcs) == 1 and arcs[0].sweep == pytest.approx(math.pi) and arcs[0].r == pytest.approx(5.0)


def test_dxf_without_ezdxf_says_how_to_install(monkeypatch, tmp_path):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "ezdxf":
            raise ImportError("no ezdxf")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    from fea_core.sketch.dxf import import_dxf
    with pytest.raises(SketchError, match="module 'sketch'"):
        import_dxf(new_sketch("twod"), tmp_path / "x.dxf")


# ------------------------------------------------------------------ picture and import weight
def test_png_has_ids_and_regions(tmp_path):
    from fea_core.sketch.build import construct
    from fea_core.sketch.render import render_png
    sk = _plate()
    cons = construct(sk)

    class B:
        pass
    b = B()
    b.cons = cons
    p = render_png(sk, tmp_path / "s.png", b)
    assert p.stat().st_size > 5000
    assert render_png(sk, tmp_path / "t.png", None).stat().st_size > 3000


def test_importing_the_sketch_package_loads_neither_ocp_nor_shapely():
    code = ("import sys; import fea_core.sketch; from fea_core.sketch import new_sketch; from fea_core.study import Study; "
            "print('LOADED', [m for m in ('OCP', 'shapely', 'ezdxf', 'matplotlib') if m in sys.modules])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip().endswith("LOADED []"), r.stdout


def test_arrays_are_plain_floats_in_the_saved_sketch():
    sk = _plate()
    json.dumps(sk.to_dict())                                         # no numpy types in the study file
    assert all(isinstance(v, (int, float, str)) for xy in sk.points.values() for v in xy)


# ------------------------------------------------------------------ example 11
def test_example_11_builds_and_follows_a_parameter(tmp_path):
    from examples import sketches
    from fea_core.sketch.build import rebuild_study
    plate = sketches.build_plate(tmp_path / "plate")
    assert {"wall", "eye", "bolt_1", "bolt_2"} <= set(plate.selections) and plate.twod.thickness == 8.0
    assert [c.selection for c in plate.constraints] == ["wall"] and plate.load_cases[0].loads[0].selection == "eye"
    new, rep, _ = rebuild_study(plate, tmp_path / "plate", params={"d_eye": 24})
    assert rep is not None and rep.blocking() == [] and new.load_cases[0].loads[0].selection == "eye"
    assert new.selections["eye"].faces[0].area == pytest.approx(math.pi * 24 * 8, rel=1e-9)
    frame = sketches.build_frame(tmp_path / "frame")
    assert set(frame.frame.members) == {"E1", "E2", "E3", "E4"} and len(frame.load_cases) == 2
    new2, _, _ = rebuild_study(frame, tmp_path / "frame", params={"rise": 2000})
    assert new2.frame.nodes["P3"].xyz[2] == 6000.0 and len(new2.load_cases[0].loads) == 2


def test_project_roundtrip_keeps_the_sketch_and_the_rebuild_works(tmp_path):
    """.fea = model.step + model.json (the sidecar) + study.json (with the sketch): after loading, a parameter change still remaps."""
    from fea_core.project import load_project, save_project
    from fea_core.sketch.build import build_study, rebuild_study, sketch_parameters
    from fea_core.study import Displacement
    st, _ = build_study(_plate(), tmp_path / "w", "plate")
    st.constraints.append(Displacement("plate.1", uy=0.0))
    assert sketch_parameters(st) == {"W": {"expr": 100.0, "value": 100.0}, "d": {"expr": 20.0, "value": 20.0}}
    save_project(tmp_path / "p.fea", st, st.geometry)
    st2, step, _ = load_project(tmp_path / "p.fea", tmp_path / "unpacked")
    assert st2.sketch is not None and st2.sketch.to_dict() == st.sketch.to_dict() and step is not None
    from fea_core.sketch.build import is_built
    assert is_built(st2.sketch)
    new, rep, _ = rebuild_study(st2, tmp_path / "again", params={"d": 40})
    assert rep is not None and rep.blocking() == [] and [c.selection for c in new.constraints] == ["plate.1"]
    assert new.selections["hole"].faces[0].area == pytest.approx(2 * math.pi * 20 * 5)


def test_a_crafted_sketch_in_a_study_file_is_refused_or_sanitised():
    from fea_core.study import Study, StudyFormatError
    base = new_sketch("twod").to_dict()
    bad_line = dict(base, points={"P1": [0, 0]}, entities=[{"id": "E1", "type": "line", "points": ["P1"]}])
    with pytest.raises(SketchError, match="wrong number"):
        Sketch.from_dict(bad_line)
    with pytest.raises(StudyFormatError):
        Study.from_dict({"name": "x", "geometry": "", "sketch": bad_line})
    huge = dict(base, params={"a": "2**" + "9" * 3})
    sk = Sketch.from_dict(huge)
    assert sk is not None
    with pytest.raises(SketchError):
        sk.values()
    odd = dict(base, regions=[{"point": [1, 1], "name": "A\x00B<img src=//h/x>\n" + "x" * 200, "material": 5}], frame={"section": 7, "material": ["x"]},
               points={"P1": [0, 0], "P2": [1, 1]}, supports=[{"point": "P9", "kind": "fixed"}, {"point": "P1", "kind": "rocket"}])
    sk2 = Sketch.from_dict(odd)
    assert sk2 is not None and len(sk2.regions[0].name) <= 80 and "\x00" not in sk2.regions[0].name and sk2.regions[0].material == "5"
    assert sk2.supports == [] and sk2.frame["section"] == "7"
    with pytest.raises(SketchError):
        Sketch.from_dict(dict(base, regions=[{"point": [1], "name": "x"}]))


def test_restore_drops_the_point_cache():
    """Undo / a failed import restore the object in place; the cache that finds an equal point must not outlive that."""
    sk = new_sketch("twod")
    sk.add_entity("line", [[0, 0], [1, 0]])
    saved = sk.copy()
    sk.add_entity("line", [[5, 5], [6, 6]])
    sk.restore(saved)
    assert len(sk.points) == 2 and len(sk.entities) == 1
    e = sk.add_entity("line", [[5, 5], [7, 7]])                 # [5, 5] is new again: it must get a new point, not a stale id
    assert [sk.points[p] for p in e.points] == [[5, 5], [7, 7]]

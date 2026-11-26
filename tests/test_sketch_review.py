"""Regression tests of the review of the sketcher (F15-S5): errors of the expression evaluator, names and ids, stable names of the split
members, limits and corrupt input of the DXF import, a file with a wrong type of value, text of the user in the picture."""
from __future__ import annotations

import json
import struct

import pytest

from fea_core.sketch.expr import SketchError, evaluate, names_in
from fea_core.sketch.model import Sketch, new_sketch


# ------------------------------------------------------------------ the evaluator never raises anything but SketchError
@pytest.mark.parametrize("text", ["5 % 0", "5 % (2 - 2)", "1 / 0", "0 ** -1", "9" * 399, "-" * 390 + "1",
                                  "1e308 * 10", "sqrt(-1)", "atan2(0)", "hypot(1e308, 1e308) * 10"])
def test_bad_expressions_raise_only_sketch_error(text):
    with pytest.raises(SketchError):
        evaluate(text)
    try:
        names_in(text)
    except SketchError:
        pass


def test_huge_numbers_and_wrong_types_are_sketch_errors():
    with pytest.raises(SketchError):
        evaluate(10 ** 400)
    with pytest.raises(SketchError):
        evaluate("1" + "0" * 5000)                                     # over the digit limit of int() and over MAX_LEN
    with pytest.raises(SketchError):
        evaluate(None)
    assert evaluate("7 % 4") == 3.0 and evaluate("2 ** 8") == 256.0
    assert evaluate("(" * 150 + "1" + ")" * 150) == 1.0                       # deep but valid


def test_set_param_and_add_entity_roll_back_on_any_error():
    sk = new_sketch("twod")
    sk.set_param("a", 5)
    for bad in ("5 % 0", "9" * 399, 10 ** 400):
        with pytest.raises(SketchError):
            sk.set_param("b", bad)
        assert "b" not in sk.params
        with pytest.raises(SketchError):
            sk.set_param("a", bad)
        assert sk.params["a"] == 5
    n_pts = len(sk.points)
    for bad in ("5 % 0", "9" * 399):
        with pytest.raises(SketchError):
            sk.add_entity("circle", center=[1, 2], radius=bad)
        with pytest.raises(SketchError):
            sk.add_entity("line", [[0, 0], [bad, 1]])
    assert len(sk.points) == n_pts and not sk.entities and sk.next_point == 1       # no stray point stays behind


# ------------------------------------------------------------------ names and ids
def test_entity_names_are_checked_when_added_and_when_loaded():
    sk = new_sketch("twod")
    sk.add_entity("line", [[0, 0], [10, 0]], name="bottom")
    for bad in ("bottom", "E7", "side.1", " lead", "x" * 81, "tab\there"):
        with pytest.raises(SketchError):
            sk.add_entity("line", [[0, 0], [0, 10]], name=bad)
    assert [e.id for e in sk.entities] == ["E1"] and len(sk.points) == 2          # the failed calls added nothing
    d = sk.to_dict()
    d2 = json.loads(json.dumps(d))
    d2["entities"].append({"id": "E5", "type": "line", "points": ["P1", "P2"], "name": "bottom"})
    with pytest.raises(SketchError, match="already used"):
        Sketch.from_dict(d2)
    d3 = json.loads(json.dumps(d))
    d3["entities"].append({"id": "E1", "type": "line", "points": ["P1", "P2"]})
    with pytest.raises(SketchError, match="two entities with the id E1"):
        Sketch.from_dict(d3)
    d4 = json.loads(json.dumps(d))
    d4["entities"].append({"id": "E6", "type": "line", "points": ["P1", "P2"], "name": "E9"})
    with pytest.raises(SketchError, match="looks like an entity id"):
        Sketch.from_dict(d4)
    d5 = json.loads(json.dumps(d))
    d5["entities"].append({"id": "E6", "type": "line", "points": ["P1", "P2"], "name": "wall.2"})
    with pytest.raises(SketchError, match="sides"):
        Sketch.from_dict(d5)
    loaded = Sketch.from_dict(d)
    assert loaded is not None and loaded.entities[0].name == "bottom"


# ------------------------------------------------------------------ stable names of the split members
def test_split_pieces_keep_their_names_when_another_node_is_added():
    from fea_core.sketch.frame import build_frame
    sk = new_sketch("frame")
    sk.set_param("L", 4000)
    sk.add_entity("line", [[0, 0], ["L", 0]])
    sk.add_support(["L/2", 0], "node")
    first = build_frame(sk)
    assert set(first.members) == {"E1_P1", "E1_P3"} and set(first.nodes) == {"P1", "P2", "P3"}
    sk.add_support(["L/4", 0], "node")                                # a new node in the first piece
    second = build_frame(sk)
    assert set(second.members) == {"E1_P1", "E1_P4", "E1_P3"}
    assert second.members["E1_P3"]["start"] == "P3" and second.members["E1_P3"]["end"] == "P2"      # the piece after the mid node is the same piece
    assert first.members["E1_P3"]["end"] == second.members["E1_P3"]["end"]


def test_crossing_and_circle_nodes_do_not_depend_on_other_curves():
    from fea_core.sketch.frame import build_frame
    sk = new_sketch("frame")
    sk.frame["split"] = True
    sk.add_entity("line", [[0, 0], [10, 10]])
    sk.add_entity("line", [[0, 10], [10, 0]])
    before = build_frame(sk)
    assert "X_E1_E2" in before.nodes and "E1_X_E1_E2" in before.members and "E2_X_E1_E2" in before.members
    sk.add_entity("line", [[20, 0], [30, 10]])                          # an unrelated curve does not rename the crossing
    after = build_frame(sk)
    assert "X_E1_E2" in after.nodes and set(before.nodes) <= set(after.nodes) and set(before.members) <= set(after.members)
    c = new_sketch("frame")
    c.add_entity("circle", center=[100, 0], radius=10)
    fb = build_frame(c)
    assert {"E1_t000", "E1_t500"} <= set(fb.nodes) and set(fb.members) == {"E1_t000", "E1_t500"}


# ------------------------------------------------------------------ the study file
def test_a_sketch_with_a_wrong_type_in_the_file_is_a_format_error():
    from fea_core.study import StudyFormatError, _load_sketch
    good = new_sketch("twod").to_dict()
    assert _load_sketch(good) is not None and _load_sketch(None) is None and _load_sketch("x") is None
    for patch in ({"next_entity": "abc"}, {"next_point": [1]}, {"entities": [{"id": "E1", "type": "line", "points": 5}]},
                  {"entities": "abc"}, {"regions": [{"point": 5}]}):
        with pytest.raises(StudyFormatError):
            _load_sketch({**good, **patch})


# ------------------------------------------------------------------ the picture
def test_picture_does_not_interpret_user_text_as_mathtext_and_needs_matplotlib(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from fea_core.sketch.render import render_png
    sk = new_sketch("twod")
    sk.set_param("a", 5)
    sk.add_entity("line", [[0, 0], [10, 0]], name="$\\mathrm{x}^{")                 # an unbalanced mathtext would raise in matplotlib
    png = render_png(sk, tmp_path / "a.png")
    assert png.exists() and png.stat().st_size > 1000
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", None)
    with pytest.raises(SketchError, match="plots"):
        render_png(sk, tmp_path / "b.png")


# ------------------------------------------------------------------ DXF
def _doc(ezdxf):
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    return doc


def test_dxf_cap_tiny_bulge_non_finite_and_corrupt_input(tmp_path, monkeypatch):
    ezdxf = pytest.importorskip("ezdxf")
    from fea_core.sketch import dxf as D
    assert D.MAX_BYTES == 16 << 20
    doc = _doc(ezdxf)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0, 0, 0, 1e-9), (100, 0, 0, 0, 0), (100, 50, 0, 0, 0)], format="xyseb")      # bulge 1e-9: a straight segment
    msp.add_line((0, 0), (float("nan"), 5))
    msp.add_circle((50, 50), 5)
    f = tmp_path / "a.dxf"
    doc.saveas(f)
    sk = new_sketch("twod")
    info = D.import_dxf(sk, f)
    assert info["arcs"] == 0 and info["circles"] == 1 and info["skipped"].get("LINE (non-finite)") == 1
    assert "Not imported" in " ".join(info["warnings"])
    # a file over the cap is refused before ezdxf reads it
    big = tmp_path / "big.dxf"
    big.write_bytes(b"0" * ((16 << 20) + 1))
    with pytest.raises(SketchError, match="too large"):
        D.import_dxf(new_sketch("twod"), big)
    # a reader that fails in a way ezdxf does not document: a SketchError, nothing else
    for exc in (MemoryError(), RecursionError(), KeyError("x"), IndexError("i"), struct.error("s"), OverflowError("o")):
        def boom(path, _e=exc):
            raise _e
        monkeypatch.setattr(ezdxf, "readfile", boom)
        with pytest.raises(SketchError):
            D.import_dxf(new_sketch("twod"), f)
    monkeypatch.undo()
    # a conversion that fails half-way leaves the sketch as it was
    sk2 = new_sketch("twod")
    sk2.add_entity("line", [[0, 0], [5, 0]])
    before = json.dumps(sk2.to_dict(), sort_keys=True)
    real = Sketch.add_entity
    calls = {"n": 0}

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError()
        return real(self, *a, **kw)

    monkeypatch.setattr(Sketch, "add_entity", flaky)
    with pytest.raises(SketchError):
        D.import_dxf(sk2, f)
    assert json.dumps(sk2.to_dict(), sort_keys=True) == before


def test_dxf_tool_uses_the_module_cap_and_to_3d_a_safe_file_name():
    import inspect
    from fea_core.mcp import tools_sketch
    src = inspect.getsource(tools_sketch)
    assert "MAX_DXF_BYTES" in src and "64 << 20" not in src and "safe_name(st.name)" in src and "st.name.replace" not in src

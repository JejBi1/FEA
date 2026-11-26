"""Editing operations of the sketch (fea_core.sketch.ops): typed entry, move, copy / mirror, intersections, trim."""
from __future__ import annotations

import math

import pytest

from fea_core.sketch.expr import SketchError
from fea_core.sketch.model import new_sketch, resolve
from fea_core.sketch.ops import add_expr, copy_entities, curve_intersections, mirror_function, move_entities, parse_entry, trim


def _sk():
    sk = new_sketch("twod")
    sk.set_param("W", 100)
    return sk


def test_entry_absolute_relative_polar_and_point_ids():
    sk = _sk()
    assert parse_entry("10, 20", sk) == (10.0, 20.0)
    assert parse_entry("W/2; W", sk) == ("W/2", "W")
    assert parse_entry("sqrt(4), 3", sk) == ("sqrt(4)", 3.0)
    assert parse_entry("@5, -2", sk, (10.0, 20.0)) == (15.0, 18.0)
    assert parse_entry("@5, 2", sk, ("W", 20.0)) == ("W+5", 22.0)
    assert parse_entry("@10<90", sk, (0.0, 0.0)) == (0.0, 10.0)
    assert parse_entry("@W<30", sk, (1.0, 1.0)) == ("1+(W*cos(rad(30)))", "1+(W*sin(rad(30)))")
    assert parse_entry("2<180", sk) == (-2.0, 0.0)
    pid = sk.add_point(3, 4)
    assert parse_entry(pid, sk) == pid
    assert add_expr("a", 0) == "a" and add_expr(2, 3) == 5.0


@pytest.mark.parametrize("bad", ["", "10", "1,2,3", "@1,2", "a, 2", "(1, 2", "1, __import__('x')", "@1<", "<30"])
def test_entry_errors_are_sentences(bad):
    with pytest.raises(SketchError):
        parse_entry(bad, _sk(), None)


def test_move_keeps_parameters_and_shared_points_stretch():
    sk = _sk()
    a = sk.add_entity("line", [[0, 0], ["W", 0]])
    b = sk.add_entity("line", [["W", 0], ["W", 50]])
    assert move_entities(sk, [a.id], 0, 10) == 2
    assert sk.points[a.points[0]] == [0.0, 10.0] and sk.points[a.points[1]] == ["W", 10.0]
    assert sk.points[b.points[0]] == ["W", 10.0]                     # shared with the moved line: the second line stretches
    assert resolve(sk).segs[1].p0 == (100.0, 10.0)
    with pytest.raises(SketchError):
        move_entities(sk, [a.id], "nope", 0)
    assert sk.points[a.points[0]] == [0.0, 10.0]


def test_copy_and_mirror():
    sk = _sk()
    a = sk.add_entity("arc_center", [[10, 0], [0, 10]], center=[0, 0])
    c = sk.add_entity("circle", center=[5, 5], radius="W/10")
    r = sk.add_entity("rect", [[0, 0], [4, 2]])
    cp = copy_entities(sk, [a.id, c.id, r.id], lambda x, y: (x + 100, y))
    assert [e.type for e in cp] == ["arc_center", "circle", "rect"] and cp[1].radius == "W/10"
    f = mirror_function((0, 0), (0, 1))                                # about the y axis
    mr = copy_entities(sk, [a.id, r.id], f, mirror=True)
    segs = {s.entity: s for s in resolve(sk).segs}
    arc = segs[mr[0].id]
    assert arc.p0 == pytest.approx((0.0, 10.0), abs=1e-9)
    assert arc.sweep == pytest.approx(math.pi / 2) and arc.c == pytest.approx((0.0, 0.0), abs=1e-9)
    assert mr[1].type == "polyline" and mr[1].closed
    with pytest.raises(SketchError):
        mirror_function((1, 1), (1, 1))
    n = len(sk.entities)
    with pytest.raises(SketchError):
        copy_entities(sk, [a.id, "E99"], f)
    assert len(sk.entities) == n                                       # a failed copy leaves nothing behind


def test_intersections_line_circle_arc():
    sk = _sk()
    sk.add_entity("line", [[-10, 0], [10, 0]])
    sk.add_entity("circle", center=[0, 0], radius=5)
    sk.add_entity("arc_center", [[0, 5], [-5, 0]], center=[0, 0])
    sk.add_entity("line", [[0, -10], [0, 10]])
    s = resolve(sk).segs
    pts = sorted(curve_intersections(s[0], s[1]))
    assert pts == [pytest.approx((-5.0, 0.0)), pytest.approx((5.0, 0.0))]
    on_arc = curve_intersections(s[0], s[2])                          # the arc covers 90°–180°: only (-5, 0)
    assert len(on_arc) == 1 and on_arc[0] == pytest.approx((-5.0, 0.0))
    assert curve_intersections(s[1], s[1]) == []                      # concentric = nothing
    assert len(curve_intersections(s[3], s[1])) == 2


def test_trim_line_arc_and_circle():
    sk = _sk()
    h = sk.add_entity("line", [[0, 0], [100, 0]])
    sk.add_entity("line", [[30, -10], [30, 10]])
    sk.add_entity("line", [[60, -10], [60, 10]])
    r = trim(sk, h.id, (45, 0))                                        # the middle part between the two crossings goes
    assert r["kept"] == ["E1", "E4"]
    segs = {s.entity: s for s in resolve(sk).segs}
    assert segs["E1"].p1 == pytest.approx((30.0, 0.0)) and segs["E4"].p0 == pytest.approx((60.0, 0.0)) and segs["E4"].p1 == (100.0, 0.0)
    with pytest.raises(SketchError, match="nothing to trim"):          # E1 now only touches the vertical line at its end
        trim(sk, "E1", (10, 0))
    two = _sk()
    t1 = two.add_entity("line", [[0, 0], [100, 0]])
    two.add_entity("line", [[30, -10], [30, 10]])
    r2 = trim(two, t1.id, (10, 0))                                     # an end part: the entity keeps its id on the rest
    assert r2["kept"] == ["E1"] and resolve(two).segs_of("E1")[0].p0 == pytest.approx((30.0, 0.0))
    c = new_sketch("twod")
    ring = c.add_entity("circle", center=[0, 0], radius=10)
    c.add_entity("line", [[-20, 0], [20, 0]])
    c.add_entity("line", [[0, -20], [0, 20]])
    trim(c, ring.id, (7.07, 7.07))                                     # the first-quadrant part of the circle goes
    arc = resolve(c).segs_of(ring.id)[0]
    assert arc.kind == "arc" and arc.sweep == pytest.approx(3 * math.pi / 2) and arc.r == pytest.approx(10.0)
    a = new_sketch("twod")
    ar = a.add_entity("arc3", [[10, 0], [0, 10], [-10, 0]])
    a.add_entity("line", [[-5, -5], [-5, 20]])
    a.add_entity("line", [[5, -5], [5, 20]])
    out = trim(a, ar.id, (0, 10))
    assert len(out["kept"]) == 2
    kept = [s for s in resolve(a).segs if s.kind == "arc"]
    assert sorted(round(s.sweep, 6) for s in kept) == [round(math.pi / 3, 6)] * 2
    rect = new_sketch("twod")
    rect.add_entity("rect", [[0, 0], [10, 10]])
    with pytest.raises(SketchError, match="rectangle"):
        trim(rect, "E1", (1, 0))

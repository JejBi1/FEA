"""Editing operations of the sketch that the editor (and anybody else) uses: typed coordinate entry, move, copy / mirror, trim.

Pure functions over `Sketch` (no Qt, no OCP): the editor of fea_gui.sketch only calls them, the tests call them directly.

Typed entry (`parse_entry`), one point per line:
    x, y            absolute coordinates (numbers or expressions of the parameters: `W/2, d + 5`)
    @dx, dy         relative to the previous point of the tool
    @r<angle        polar relative to the previous point (angle in degrees, counter-clockwise from +x)
    r<angle         polar about the origin
    P3              an existing point (exactly connected)
"""
from __future__ import annotations

import math
import re
from typing import Any

from .expr import SketchError, evaluate
from .model import Entity, Resolved, Seg, Sketch, resolve

Coord = str | float
PointRef = str | tuple[Coord, Coord]


def _split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch in ",;" and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _num(a: Coord) -> float | None:
    if isinstance(a, (int, float)):
        return float(a)
    try:
        return float(str(a).strip())
    except ValueError:
        return None


def _wrap(e: Coord) -> str:
    s = f"{e:.10g}" if isinstance(e, (int, float)) else str(e).strip()
    return s if re.fullmatch(r"[A-Za-z0-9_.]+", s) else f"({s})"


def _clean(v: float) -> float:
    return round(float(v), 10) + 0.0


def add_expr(a: Coord, b: Coord) -> Coord:
    """a + b as a number when both are numbers, otherwise as an expression (the parameters stay in it)."""
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return _clean(na + nb)
    if nb is not None and nb == 0:
        return a
    if na is not None and na == 0:
        return b
    return f"{_wrap(a)}+{_wrap(b)}"


def parse_entry(text: str, sk: Sketch, last: tuple[Coord, Coord] | None = None) -> PointRef:
    """A point from the text typed into the status bar (see the module text). → a point id or (x, y) expressions; SketchError when it
    cannot be evaluated (the message is shown to the user)."""
    t = (text or "").strip()
    if not t:
        raise SketchError("Type a point: x, y   @dx, dy   @r<angle   or an existing point id (P1)")
    if t in sk.points:
        return t
    relative = t.startswith("@")
    if relative:
        t = t[1:].strip()
        if last is None:
            raise SketchError("A relative point (@…) needs a previous point – give the first point as x, y")
    env = sk.values()
    if "<" in t and "," not in t:
        r, a = (x.strip() for x in t.split("<", 1))
        evaluate(r, env)
        evaluate(a, env)
        nr, na = _num(r), _num(a)
        if nr is not None and na is not None:
            dx, dy = _clean(nr * math.cos(math.radians(na))), _clean(nr * math.sin(math.radians(na)))
        else:
            dx, dy = f"{_wrap(r)}*cos(rad({a}))", f"{_wrap(r)}*sin(rad({a}))"
    else:
        parts = _split_top(t)
        if len(parts) != 2 or not all(parts):
            raise SketchError(f"'{text}': give two coordinates 'x, y' (numbers or expressions), '@dx, dy' or '@r<angle'")
        evaluate(parts[0], env)
        evaluate(parts[1], env)
        nx, ny = _num(parts[0]), _num(parts[1])
        dx = nx if nx is not None else parts[0]
        dy = ny if ny is not None else parts[1]
    if relative:
        assert last is not None
        return (add_expr(last[0], dx), add_expr(last[1], dy))
    return (dx, dy)


def parse_vector(text: str, sk: Sketch) -> tuple[Coord, Coord]:
    """A displacement typed as `dx, dy` (an optional leading @; numbers or expressions)."""
    parts = _split_top((text or "").strip().lstrip("@"))
    if len(parts) != 2 or not all(parts):
        raise SketchError("Type the vector as dx, dy (numbers or expressions)")
    env = sk.values()
    evaluate(parts[0], env)
    evaluate(parts[1], env)
    nx, ny = _num(parts[0]), _num(parts[1])
    return (nx if nx is not None else parts[0], ny if ny is not None else parts[1])


def point_coords(sk: Sketch, p: PointRef) -> tuple[Coord, Coord]:
    """The coordinate expressions of a point reference."""
    if isinstance(p, str):
        x, y = sk.points[p]
        return x, y
    return p


def add_point_ref(sk: Sketch, p: PointRef) -> str:
    return sk.point_ref(p if isinstance(p, str) else [p[0], p[1]])


# ------------------------------------------------------------------ move

def entity_point_ids(sk: Sketch, ids: list[str]) -> list[str]:
    out: list[str] = []
    for eid in ids:
        e = sk.entity(eid)
        for p in [*e.points, *([e.center] if e.center else [])]:
            if p not in out:
                out.append(p)
    return out


def move_entities(sk: Sketch, ids: list[str], dx: Coord, dy: Coord) -> int:
    """Moves the points of the entities by (dx, dy) (numbers or expressions). Points shared with other entities move too (those
    entities stretch). Returns the number of points moved."""
    pts = entity_point_ids(sk, ids)
    evaluate(dx, sk.values())
    evaluate(dy, sk.values())
    old = {p: list(sk.points[p]) for p in pts}
    try:
        for p in pts:
            x, y = sk.points[p]
            sk.move_point(p, add_expr(x, dx), add_expr(y, dy))
    except SketchError:
        for p, xy in old.items():
            sk.points[p] = xy
        raise
    return len(pts)


# ------------------------------------------------------------------ copy and mirror

def copy_entities(sk: Sketch, ids: list[str], f, mirror: bool = False) -> list[Entity]:
    """Copies of the entities with every point moved by f(x, y) → (x, y) (numeric, mm). `mirror` = f reverses the orientation
    (arcs given by the centre and the ends swap their ends; a rectangle becomes a closed polyline). The copies have no name."""
    env = sk.values()
    saved = (dict(sk.points), sk.next_point, len(sk.entities), sk.next_entity)
    new: list[Entity] = []

    def P(pid: str) -> list:
        x, y = sk.point_values(pid, env)
        nx, ny = f(x, y)
        return [_clean(nx), _clean(ny)]
    try:
        for eid in ids:
            e = sk.entity(eid)
            kw: dict[str, Any] = dict(construction=e.construction, hole=e.hole, section=e.section, material=e.material)
            if e.type == "line":
                n = sk.add_entity("line", [P(p) for p in e.points], **kw)
            elif e.type == "arc3":
                n = sk.add_entity("arc3", [P(p) for p in e.points], **kw)
            elif e.type == "arc_center":
                ends = [P(p) for p in e.points]
                n = sk.add_entity("arc_center", ends[::-1] if mirror else ends, center=P(e.center or ""), **kw)
            elif e.type == "circle":
                n = sk.add_entity("circle", center=P(e.center or ""), radius=e.radius, **kw)
            elif e.type == "rect":
                (x0, y0), (x1, y1) = (sk.point_values(p, env) for p in e.points)
                if mirror:
                    cs = [f(x0, y0), f(x1, y0), f(x1, y1), f(x0, y1)]
                    n = sk.add_entity("polyline", [[_clean(a), _clean(b)] for a, b in cs], closed=True, **kw)
                else:
                    n = sk.add_entity("rect", [P(p) for p in e.points], **kw)
            else:
                n = sk.add_entity("polyline", [P(p) for p in e.points], closed=e.closed, **kw)
            new.append(n)
    except SketchError:
        sk.points, sk.next_point = saved[0], saved[1]
        del sk.entities[saved[2]:]
        sk.next_entity = saved[3]
        raise
    return new


def mirror_function(p0: tuple[float, float], p1: tuple[float, float]):
    """The reflection about the line through p0 and p1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n2 = dx * dx + dy * dy
    if n2 <= 1e-18:
        raise SketchError("The two points of the mirror line coincide")

    def f(x: float, y: float) -> tuple[float, float]:
        ax, ay = x - p0[0], y - p0[1]
        t = (ax * dx + ay * dy) / n2
        fx, fy = p0[0] + t * dx, p0[1] + t * dy
        return 2 * fx - x, 2 * fy - y
    return f


# ------------------------------------------------------------------ intersections and trim

def curve_intersections(a: Seg, b: Seg, tol: float = 1e-9) -> list[tuple[float, float]]:
    """Points where the two curves meet (lines, arcs, circles); only the points that lie on both curves (their ends included)."""
    pts: list[tuple[float, float]] = []
    if a.kind == "line" and b.kind == "line":
        r = (a.p1[0] - a.p0[0], a.p1[1] - a.p0[1])
        s = (b.p1[0] - b.p0[0], b.p1[1] - b.p0[1])
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) > 1e-12 * math.hypot(*r) * math.hypot(*s):
            qp = (b.p0[0] - a.p0[0], b.p0[1] - a.p0[1])
            t = (qp[0] * s[1] - qp[1] * s[0]) / den
            u = (qp[0] * r[1] - qp[1] * r[0]) / den
            e1, e2 = tol * 10 / max(math.hypot(*r), 1e-12), tol * 10 / max(math.hypot(*s), 1e-12)
            if -e1 <= t <= 1 + e1 and -e2 <= u <= 1 + e2:
                pts.append((a.p0[0] + t * r[0], a.p0[1] + t * r[1]))
        return pts
    if a.kind != "line" and b.kind == "line":
        a, b = b, a
    if a.kind == "line":                                         # line × arc / circle
        assert b.c is not None
        d = (a.p1[0] - a.p0[0], a.p1[1] - a.p0[1])
        f = (a.p0[0] - b.c[0], a.p0[1] - b.c[1])
        A = d[0] ** 2 + d[1] ** 2
        B = 2 * (f[0] * d[0] + f[1] * d[1])
        C = f[0] ** 2 + f[1] ** 2 - b.r ** 2
        disc = B * B - 4 * A * C
        if disc >= 0 and A > 0:
            for sg in ((-1, 1) if disc > 0 else (1,)):
                t = (-B + sg * math.sqrt(disc)) / (2 * A)
                e1 = tol * 10 / math.sqrt(A)
                if -e1 <= t <= 1 + e1:
                    p = (a.p0[0] + t * d[0], a.p0[1] + t * d[1])
                    if b.kind == "circle" or b.param(p) <= 1 + 1e-9:
                        pts.append(p)
        return pts
    assert a.c is not None and b.c is not None                    # arc / circle × arc / circle
    dx, dy = b.c[0] - a.c[0], b.c[1] - a.c[1]
    dd = math.hypot(dx, dy)
    if dd < 1e-12 or dd > a.r + b.r + tol or dd < abs(a.r - b.r) - tol:
        return pts
    x = (dd * dd + a.r ** 2 - b.r ** 2) / (2 * dd)
    h2 = a.r ** 2 - x * x
    h = math.sqrt(max(h2, 0.0))
    mx, my = a.c[0] + x * dx / dd, a.c[1] + x * dy / dd
    for sg in ((-1, 1) if h > 1e-12 else (1,)):
        p = (mx - sg * h * dy / dd, my + sg * h * dx / dd)
        if all(s.kind == "circle" or s.param(p) <= 1 + 1e-9 for s in (a, b)):
            pts.append(p)
    return pts


def _piece(sk: Sketch, e: Entity, seg: Seg, ta: float, tb: float, env) -> tuple[str, list[str], str | None]:
    """The part [ta, tb] of the curve as (type, point ids, centre id)."""
    pa, pb = seg.at(ta), seg.at(tb)
    def pid(q) -> str:
        return sk.add_point(_clean(q[0]), _clean(q[1]))
    if seg.kind == "line":
        return "line", [pid(pa), pid(pb)], None
    assert seg.c is not None
    cid = e.center if e.type in ("arc_center", "circle") and e.center else pid(seg.c)
    ends = [pid(pa), pid(pb)] if seg.sign > 0 else [pid(pb), pid(pa)]
    return "arc_center", ends, cid


def trim(sk: Sketch, eid: str, click: tuple[float, float]) -> dict:
    """Removes the part of the entity (line, arc, circle) between its two nearest intersections with other curves around `click`; the
    entity keeps its id on one remaining part, the other part (if any) becomes a new entity. Returns {"kept": [ids], "removed": (ta, tb)}."""
    e = sk.entity(eid)
    if e.type not in ("line", "arc3", "arc_center", "circle"):
        raise SketchError(f"{eid}: trim works on lines, arcs and circles (a rectangle or a polyline would have to be split first)")
    res: Resolved = resolve(sk)
    seg = res.segs_of(eid)[0]
    ts: list[float] = []
    for o in res.segs:
        if o.entity == eid or o.construction:
            continue
        for p in curve_intersections(seg, o, res.tol):
            t = seg.param(p)
            if seg.kind == "circle":
                t %= 1.0
            if 1e-9 < t < 1 - 1e-9 or seg.kind == "circle":
                ts.append(t)
    ts = sorted(set(round(t, 12) for t in ts))
    tc = seg.param(click)
    if seg.kind == "circle":
        tc %= 1.0
        if len(ts) < 2:
            raise SketchError(f"{eid}: a circle needs at least two intersections with other curves to be trimmed")
        lo = max([t for t in ts if t < tc], default=ts[-1] - 1.0)
        hi = min([t for t in ts if t > tc], default=ts[0] + 1.0)
        a0, a1 = hi, lo + 1.0                                         # the part that stays: from hi round to lo
        saved_pts = dict(sk.points)
        c = e.center
        p0 = seg.at(a0 % 1.0)
        p1 = seg.at(a1 % 1.0)
        e.type, e.radius = "arc_center", None
        e.points = [sk.add_point(_clean(p0[0]), _clean(p0[1])), sk.add_point(_clean(p1[0]), _clean(p1[1]))]
        e.center = c
        sk._drop_orphans()
        del saved_pts
        return {"kept": [eid], "removed": (lo, hi)}
    lo = max([t for t in ts if t < tc], default=0.0)
    hi = min([t for t in ts if t > tc], default=1.0)
    if lo <= 1e-9 and hi >= 1 - 1e-9:
        raise SketchError(f"{eid}: nothing to trim against on this side – delete the entity instead")
    keep = []
    if lo > 1e-9:
        keep.append((0.0, lo))
    if hi < 1 - 1e-9:
        keep.append((hi, 1.0))
    kept = []
    first = True
    for ta, tb in keep:
        typ, pts, cid = _piece(sk, e, seg, ta, tb, res.env)
        if first:
            e.type, e.points, e.center, e.radius = typ, pts, cid, None
            kept.append(eid)
            first = False
        else:
            n = sk.add_entity(typ, pts, center=cid, construction=e.construction, hole=e.hole, section=e.section, material=e.material)
            kept.append(n.id)
    sk._drop_orphans()
    return {"kept": kept, "removed": (lo, hi)}

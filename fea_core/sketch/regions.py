"""2D sketch → closed regions (OCP, exact) with the checks that tell the user which entity is wrong.

Procedure:
  1. Checks on a densified copy of the curves (shapely): a polyline that crosses itself, an open contour (a curve that ends in
     the middle of nowhere) – the error names the entity and the place of the free end and the gap to the nearest other entity.
     The end points of other curves are inserted into the densified curves, so a T-junction onto an arc closes.
  2. The EXACT geometry: lines / arcs / circles as OCP edges, a big planar face split by all of them (BRepAlgoAPI_Splitter) = the
     cells of the arrangement; the outer cell is dropped.
  3. Which cells are material: a cell whose boundary is a whole inner wire of another cell is a hole (even-odd nesting; an island
     inside a hole is material again); a closed entity (circle, rectangle, closed polyline) decides for the cells inside it
     (`Entity.hole`). A chord that splits a region leaves two material regions (= two parts that can have different materials).
  4. Every material cell is a Region: the part name, the exact face, its boundary edges with the ids of the entities.
"""
# The OCP stubs of the type-check environment miss most of its classes (dynamic attributes) – the rule is off for this file only:
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .expr import SketchError
from .model import Plane3, Resolved, Seg


# ------------------------------------------------------------------ densified copy (shapely)

def _shapely():
    try:
        import shapely  # noqa: F401
        from shapely import geometry, ops
    except ImportError:
        raise SketchError("The 2D sketch needs the package shapely (module 'beams'); install it in Tools → Program modules") from None
    return geometry, ops


def _curve_param(seg: Seg, p) -> float:
    return seg.param(p)


def _dist_to_seg(seg: Seg, p) -> float:
    return seg.distance(p)


def densified(res: Resolved, segs: list[Seg]) -> dict[str, list[tuple[float, float]]]:
    """Polylines of the curves for shapely: arcs by 0.5°; the end points of other curves that lie on a curve become its vertices
    (a T-junction onto a chord polygon would otherwise miss by the sagitta)."""
    geometry, _ = _shapely()
    lines = {s.id: geometry.LineString(s.polyline()) for s in segs}
    ids = list(lines)
    from shapely.strtree import STRtree
    tree = STRtree([lines[i] for i in ids])
    by_id = {s.id: s for s in segs}
    extra: dict[str, list[tuple[float, tuple[float, float]]]] = {}
    tol_on = max(res.tol * 5, 1e-12)
    for s in segs:
        for p in {s.p0, s.p1}:
            for k in tree.query(geometry.Point(p).buffer(tol_on)):
                o = by_id[ids[int(k)]]
                if o is s or (o.kind != "circle" and (math.hypot(p[0] - o.p0[0], p[1] - o.p0[1]) <= res.tol
                                                       or math.hypot(p[0] - o.p1[0], p[1] - o.p1[1]) <= res.tol)):
                    continue
                if _dist_to_seg(o, p) <= tol_on:
                    t = _curve_param(o, p)
                    if o.kind == "circle":
                        t %= 1.0
                    if 0.0 < t < 1.0:
                        extra.setdefault(o.id, []).append((t, p))
    out: dict[str, list[tuple[float, float]]] = {}
    for s in segs:
        if s.kind == "line":
            pts = [(0.0, s.p0)] + sorted(extra.get(s.id, [])) + [(1.0, s.p1)]
        else:
            n = max(8, int(math.ceil(math.degrees(s.sweep) / 0.5)))
            pts = [(i / n, s.at(i / n)) for i in range(n + 1)]
            pts[0], pts[-1] = (0.0, s.p0), (1.0, s.p1)               # exactly the snapped end points
            pts = sorted(pts + extra.get(s.id, []), key=lambda q: q[0])
        poly = [q[1] for q in pts]
        if s.kind == "circle":
            poly[-1] = poly[0]
        out[s.id] = poly
    return out


def check_linework(res: Resolved, segs: list[Seg]) -> dict:
    """Errors of the drawing (raised as SketchError with the entity ids) and the polygons of the cells for the cross-check.

    Returns {"polygons": [...], "dense": polylines}.
    """
    geometry, ops = _shapely()
    dense = densified(res, segs)
    sk = res.sketch
    ents = {e.id: e for e in sk.entities}
    # a polyline / rectangle that crosses itself
    for eid in {s.entity for s in segs}:
        e = ents[eid]
        if e.type != "polyline":
            continue
        pts = []
        for s in [x for x in segs if x.entity == eid]:
            pts.append(s.p0)
        last = [x for x in segs if x.entity == eid][-1]
        pts.append(last.p1)
        ls = geometry.LineString(pts)
        simple = ls.is_simple if not (e.closed and len(pts) > 2) else geometry.LinearRing(pts).is_simple
        if not simple:
            raise SketchError(f"{eid}: the polyline crosses itself (self-intersection)", [eid])
    by_id = {s.id: s for s in segs}
    owner = {s.id: s.entity for s in segs}
    lines = [geometry.LineString(p) for p in dense.values()]
    noded = ops.unary_union(lines)
    polys, cuts, dangles, invalid = ops.polygonize_full(noded)
    problems: list[str] = []
    bad: list[str] = []
    tree_ids = list(dense)
    from shapely.strtree import STRtree
    segtree = STRtree([geometry.LineString(dense[i]) for i in tree_ids])

    def nearest_other(pt, exclude_entity: str):
        best = (math.inf, "")
        for k in segtree.query(geometry.Point(pt).buffer(0.2 * res.size)):
            sid = tree_ids[int(k)]
            if owner[sid] == exclude_entity:
                continue
            d = geometry.Point(pt).distance(geometry.LineString(dense[sid]))
            if d < best[0]:
                best = (d, owner[sid])
        return best

    # free ends: an end point of a curve that no other curve touches (counted on the snapped end points, so a gap shows as two ends)
    count: dict[tuple[float, float], int] = {}
    for sg in segs:
        if sg.kind != "circle":
            for p in (sg.p0, sg.p1):
                count[p] = count.get(p, 0) + 1
    tol_on = max(res.tol * 5, 1e-12)
    for sg in segs:
        if sg.kind == "circle":
            continue
        for p in (sg.p0, sg.p1):
            if count[p] != 1:
                continue
            if any(o is not sg and _dist_to_seg(o, p) <= tol_on for o in segs):
                continue                                          # the end lies on another curve (T-junction)
            gap, other = nearest_other(p, sg.entity)
            text = f"{sg.entity}: open contour – the curve ends at ({p[0]:.6g}, {p[1]:.6g}) without joining another entity"
            if math.isfinite(gap) and gap < 0.05 * res.size:
                text += f" (gap {gap:.3g} mm to {other}; connect the end points)"
            problems.append(text)
            bad.append(sg.entity)
    if not problems:
        for geom in list(getattr(dangles, "geoms", [])) + list(getattr(cuts, "geoms", [])):
            mid = geom.interpolate(0.5, normalized=True)
            hit = min(by_id, key=lambda i: geometry.LineString(dense[i]).distance(mid))
            problems.append(f"{owner[hit]}: the curve does not belong to any closed contour (a free line inside or outside the regions)")
            bad.append(owner[hit])
    for geom in getattr(invalid, "geoms", []):
        problems.append("a contour that is not a valid ring (touches itself)")
    if problems:
        raise SketchError("; ".join(dict.fromkeys(problems)), list(dict.fromkeys(bad)))
    return {"polygons": list(polys.geoms) if hasattr(polys, "geoms") else list(polys), "dense": dense}


# ------------------------------------------------------------------ exact geometry (OCP)

class EdgeIds:
    """Numbers the edges of OCP by identity (TShape + location, orientation ignored)."""

    def __init__(self):
        self.bucket: dict[int, list[tuple[object, int]]] = {}
        self.count = 0

    def id(self, edge) -> int:
        b = self.bucket.setdefault(hash(edge), [])
        for e, i in b:
            if e.IsSame(edge):                    # type: ignore[attr-defined]
                return i
        b.append((edge, self.count))
        self.count += 1
        return self.count - 1


def seg_edge(seg: Seg, pl: Plane3, w: float):
    """Exact OCP edge of a curve of the sketch in the plane shifted by w along the normal."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

    def P(xy):
        q = pl.pt(xy[0], xy[1], w)
        return gp_Pnt(float(q[0]), float(q[1]), float(q[2]))
    if seg.kind == "line":
        mk = GC_MakeSegment(P(seg.p0), P(seg.p1))
        if not mk.IsDone():
            raise SketchError(f"{seg.entity}: the line cannot be created", [seg.entity])
        return BRepBuilderAPI_MakeEdge(mk.Value()).Edge()
    if seg.kind == "arc":
        mk = GC_MakeArcOfCircle(P(seg.p0), P(seg.mid()), P(seg.p1))
        if not mk.IsDone():
            raise SketchError(f"{seg.entity}: the arc cannot be created (points too close or collinear)", [seg.entity])
        return BRepBuilderAPI_MakeEdge(mk.Value()).Edge()
    assert seg.c is not None
    c = pl.pt(seg.c[0], seg.c[1], w)
    ax = gp_Ax2(gp_Pnt(float(c[0]), float(c[1]), float(c[2])), gp_Dir(*map(float, pl.n)), gp_Dir(*map(float, pl.ex)))
    return BRepBuilderAPI_MakeEdge(gp_Circ(ax, float(seg.r))).Edge()


def _faces(shape) -> list:
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face(ex.Current()))
        ex.Next()
    return out


def _wires(face) -> list:
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(face, TopAbs_WIRE)
    while ex.More():
        out.append(TopoDS.Wire(ex.Current()))
        ex.Next()
    return out


def _edges(shape) -> list:
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        out.append(TopoDS.Edge(ex.Current()))
        ex.Next()
    return out


def interior_point(face, pl: Plane3, size: float) -> tuple[float, float]:
    """A point well inside the face (centroid of the largest triangle of its triangulation) in the sketch coordinates."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(face, size / 40.0, False, 0.3)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbTriangles() == 0:
        raise SketchError("A region cannot be meshed for the check – the sketch is degenerate here")
    best, area = None, -1.0
    for i in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(i).Get()
        pa, pb, pc = (np.array([tri.Node(k).X(), tri.Node(k).Y(), tri.Node(k).Z()]) for k in (a, b, c))
        ar = 0.5 * float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
        if ar > area:
            area, best = ar, (pa + pb + pc) / 3.0
    x, y, _ = pl.to_local(best)
    return x, y


def wire_rings(face, pl: Plane3, n: int = 24) -> list[list[tuple[float, float]]]:
    """Boundary of a face as polygons in the sketch coordinates: the outer ring first, then the holes (for the picture)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    outer = BRepTools.OuterWire_s(face)
    rings = []
    for w in sorted(_wires(face), key=lambda w: 0 if w.IsSame(outer) else 1):
        pts: list[tuple[float, float]] = []
        ex = BRepTools_WireExplorer(w, face)
        while ex.More():
            e = TopoDS.Edge(ex.Current())
            ad = BRepAdaptor_Curve(e)
            t0, t1 = ad.FirstParameter(), ad.LastParameter()
            ts = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
            if e.Orientation() == TopAbs_REVERSED:
                ts.reverse()
            for t in ts[:-1]:
                q = ad.Value(t)
                x, y, _ = pl.to_local([q.X(), q.Y(), q.Z()])
                pts.append((x, y))
            ex.Next()
        if pts:
            rings.append(pts)
    return rings


def face_area(face) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, p)
    return float(p.Mass())


@dataclass
class Region:
    index: int
    face: object                                    # OCP face in the sketch plane (shifted by `w`)
    area: float
    point: tuple[float, float]                      # a point inside, sketch coordinates
    anchor: str                                     # the lowest entity id of the boundary (stable name of the region)
    entities: list[str]                             # all entities on the boundary
    edges: list[tuple[object, list[str]]] = field(default_factory=list)   # (OCP edge, owner curve ids)
    name: str = ""
    material: str | None = None
    rings: list = field(default_factory=list)       # [outer ring, hole rings…] as lists of (x, y) – for the picture


@dataclass
class RegionSet:
    regions: list[Region]
    warnings: list[str]
    cells: int = 0


def _eid_num(eid: str) -> int:
    digits = "".join(ch for ch in eid.split(".")[0] if ch.isdigit())
    return int(digits) if digits else 0


def find_regions(res: Resolved, pl: Plane3, w: float) -> RegionSet:
    """Closed regions of the sketch (material only) in the plane shifted by `w`. Raises SketchError (with entity ids)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import List_TopoDS_Shape
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS
    sk = res.sketch
    segs = [s for s in res.segs if not s.construction]
    if not segs:
        raise SketchError("The sketch is empty: draw closed contours (lines, arcs, circles, rectangles)")
    warnings: list[str] = []
    chk = check_linework(res, segs)
    # bounding box of all curves
    xs, ys = [], []
    for s in segs:
        for p in s.polyline(5.0):
            xs.append(p[0])
            ys.append(p[1])
    margin = 0.5 * res.size + 1.0
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    o = pl.pt(0.0, 0.0, w)
    ax3 = gp_Ax3(gp_Pnt(*map(float, o)), gp_Dir(*map(float, pl.n)), gp_Dir(*map(float, pl.ex)))
    base = BRepBuilderAPI_MakeFace(gp_Pln(ax3), x0, x1, y0, y1).Face()
    sp = BRepAlgoAPI_Splitter()
    a, t = List_TopoDS_Shape(), List_TopoDS_Shape()
    a.Append(base)
    tool_edges = {}
    for s in segs:
        e = seg_edge(s, pl, w)
        tool_edges[s.id] = e
        t.Append(e)
    sp.SetArguments(a)
    sp.SetTools(t)
    sp.SetFuzzyValue(max(0.5 * res.tol, 1e-9))
    sp.Build()
    if not sp.IsDone():
        raise SketchError("The regions of the sketch cannot be found (OCCT failed) – check the entities for overlaps")
    cells = _faces(sp.Shape())
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    def local_box(face):
        bb = Bnd_Box()
        BRepBndLib.Add_s(face, bb)
        lo, hi = bb.CornerMin(), bb.CornerMax()
        pts = [pl.to_local([x, y, z]) for x in (lo.X(), hi.X()) for y in (lo.Y(), hi.Y()) for z in (lo.Z(), hi.Z())]
        return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
    eps = 1e-6 * res.size + 1e-7
    inner = []
    for f in cells:
        bx = local_box(f)
        if bx[0] <= x0 + eps and bx[1] <= y0 + eps and bx[2] >= x1 - eps and bx[3] >= y1 - eps:
            continue                                  # the outer cell (touches the whole border of the base face)
        inner.append(f)
    if not inner:
        raise SketchError("The sketch has no closed region: draw a closed contour")
    ids = EdgeIds()
    # owners of the pieces
    owners: dict[int, list[str]] = {}
    for sid, te in tool_edges.items():
        pcs = [p for p in sp.Modified(te)]
        pieces = [TopoDS.Edge(p) for p in pcs] if pcs else [te]
        for pe in pieces:
            lst = owners.setdefault(ids.id(pe), [])
            if sid not in lst:
                lst.append(sid)
    info = []
    for f in inner:
        ws = _wires(f)
        outer_w = BRepTools.OuterWire_s(f)
        outer_edges = {ids.id(e) for e in _edges(outer_w)}
        inner_wires = []
        for wr in ws:
            if wr.IsSame(outer_w):
                continue
            inner_wires.append({ids.id(e) for e in _edges(wr)})
        info.append({"face": f, "outer": outer_edges, "inner": inner_wires, "all": {ids.id(e) for e in _edges(f)}})
    # (a) nesting by wires: the outer wire of a cell is a whole inner wire of another cell
    parent: dict[int, int | None] = {}
    for i, a_ in enumerate(info):
        parent[i] = None
        for j, b_ in enumerate(info):
            if i != j and any(a_["outer"] == w_ or a_["outer"] <= w_ for w_ in b_["inner"]):
                parent[i] = j
                break

    def depth(i: int, seen=()) -> int:
        p = parent[i]
        return 0 if p is None or p in seen else 1 + depth(p, seen + (i,))
    # (b) closed entities (circle, rectangle, closed polyline) decide for the cells inside them
    geometry, _ = _shapely()
    closed: dict[str, Any] = {}
    for e in sk.entities:
        if e.construction:
            continue
        ss = [s for s in segs if s.entity == e.id]
        if not ss:
            continue
        if e.type == "circle":
            closed[e.id] = geometry.Polygon(ss[0].polyline())
        elif e.type == "rect" or (e.type == "polyline" and e.closed and len(e.points) > 2):
            closed[e.id] = geometry.Polygon([s.p0 for s in ss])
    cdepth = {k: sum(1 for k2, p2 in closed.items() if k2 != k and p2.area > poly.area and p2.contains(poly.representative_point()))
              for k, poly in closed.items()}
    ent_by_id = {e.id: e for e in sk.entities}
    regions: list[Region] = []
    tiny = 1e-9 * res.size ** 2
    for i, d in enumerate(info):
        f = d["face"]
        ar = face_area(f)
        bound_ids = sorted({s.split(".")[0] for eid in d["all"] for s in owners.get(eid, [])}, key=_eid_num)
        if ar <= tiny:
            raise SketchError(f"A region of zero area at {', '.join(bound_ids[:4]) or 'the sketch'} – entities overlap or are nearly "
                              f"identical", bound_ids[:6])
        pt = interior_point(f, pl, res.size)
        void = depth(i) % 2 == 1
        inside = [(closed[k].area, k) for k in closed if closed[k].contains(geometry.Point(pt))]
        if inside:
            innermost = ent_by_id[min(inside)[1]]
            if innermost.hole is not None:
                void = bool(innermost.hole)
            elif cdepth[innermost.id] % 2 == 1:
                void = True
        if void:
            continue
        reg = Region(len(regions), f, ar, pt, bound_ids[0] if bound_ids else "", bound_ids, rings=wire_rings(f, pl))
        for e in _edges(f):
            reg.edges.append((e, [s for s in owners.get(ids.id(e), [])]))
        regions.append(reg)
    if not regions:
        raise SketchError("The sketch has no material region (everything is a hole – check Entity.hole)")
    # cross-check against the polygons of the densified copy (the number of the cells of the arrangement)
    poly_n = len(chk["polygons"])
    if poly_n != len(inner):
        warnings.append(f"The region count differs between the exact ({len(inner)}) and the approximate ({poly_n}) check – "
                        f"look at the regions in the sketch picture")
    return RegionSet(regions, warnings, len(inner))

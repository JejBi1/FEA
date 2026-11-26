"""1D sketch → the beam system of the study (`Study.frame`, fea_core.beams): points → nodes, lines / arcs → members.

  * a node per end point (points closer than 1e-6 of the size are one node); the name of the node = the id of its point ("P3");
  * a line → a member named like the entity ("E2"; "E5.1" for the sides of a rectangle / polyline); an arc → a member with `through`
    (and the section height in the plane of the arc, `up = "radial"`); a circle → two half-circle members;
  * a point that lies inside another member splits it there (a node for a load or a support at mid-span); the parts are named after the
    node where they START ("E2_P1", "E2_P5"), so a new split elsewhere on the curve does not rename the others (loads stay on their
    piece); the crossings of `frame.split` are nodes "X_<curve>_<curve>" (+ "_2" for a second crossing of the same pair), the nodes of a
    circle "E3_t000", "E3_t500" (position on the circle in thousandths) – all of them independent of the other curves;
  * supports (`Sketch.supports`: fixed | pinned | roller_x | roller_y | node = a node without a support) become `FrameSupport`s in
    the global components; `frame.plane_restraint` (default on) holds every node against the motion out of the plane of the
    sketch (translation along the normal, rotations about the two in-plane axes) – a plane frame loaded in its plane;
  * section and material of a member: `Entity.section / material`, otherwise `Sketch.frame["section"/"material"]` (an existing
    member keeps the section the user changed in the Beams step unless the entity names one).

The update of a study (`apply_frame`) matches nodes and members by their names: positions, ends and arcs are taken from the
sketch, everything else the user set (section, releases, orientation roll, segments, loads) stays; what the sketch no longer has is
removed and the loads that point at it are reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .expr import SketchError
from .model import Plane3, Seg, Sketch, resolve
from .regions import _curve_param, _dist_to_seg


@dataclass
class FrameBuild:
    nodes: dict[str, list[float]] = field(default_factory=dict)
    members: dict[str, dict] = field(default_factory=dict)
    supports: dict[str, list[bool]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    size: float = 1.0
    plane: Plane3 | None = None


def _num(pid: str) -> int:
    d = "".join(ch for ch in pid if ch.isdigit())
    return int(d) if d else 0


def _intersections(a: Seg, b: Seg, tol: float) -> list[tuple[float, float]]:
    """Crossing points (interior of both) of two lines or a line and an arc / circle."""
    def line_line(p, q):
        r = (p.p1[0] - p.p0[0], p.p1[1] - p.p0[1])
        s = (q.p1[0] - q.p0[0], q.p1[1] - q.p0[1])
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) < 1e-12 * (math.hypot(*r) * math.hypot(*s)):
            return []
        qp = (q.p0[0] - p.p0[0], q.p0[1] - p.p0[1])
        t = (qp[0] * s[1] - qp[1] * s[0]) / den
        u = (qp[0] * r[1] - qp[1] * r[0]) / den
        if 0 < t < 1 and 0 < u < 1:
            return [(p.p0[0] + t * r[0], p.p0[1] + t * r[1])]
        return []

    def line_arc(ln, ar):
        assert ar.c is not None
        d = (ln.p1[0] - ln.p0[0], ln.p1[1] - ln.p0[1])
        f = (ln.p0[0] - ar.c[0], ln.p0[1] - ar.c[1])
        A = d[0] ** 2 + d[1] ** 2
        B = 2 * (f[0] * d[0] + f[1] * d[1])
        C = f[0] ** 2 + f[1] ** 2 - ar.r ** 2
        disc = B * B - 4 * A * C
        if disc <= 0:
            return []
        out = []
        for sgn in (-1, 1):
            t = (-B + sgn * math.sqrt(disc)) / (2 * A)
            if 0 < t < 1:
                p = (ln.p0[0] + t * d[0], ln.p0[1] + t * d[1])
                u = _curve_param(ar, p)
                if ar.kind == "circle" or 0 < u < 1:
                    out.append(p)
        return out
    if a.kind == "line" and b.kind == "line":
        return line_line(a, b)
    if a.kind == "line":
        return line_arc(a, b)
    if b.kind == "line":
        return line_arc(b, a)
    return []


def build_frame(sk: Sketch) -> FrameBuild:
    """Nodes, members and supports of a beam sketch (numeric, in the global coordinates)."""
    if sk.kind != "frame":
        raise SketchError("Only a beam sketch makes members")
    res = resolve(sk)
    pl = sk.plane3()
    if not pl.axis_aligned():
        raise SketchError("A beam sketch must lie in a plane parallel to XY, XZ or YZ")
    fb = FrameBuild(size=res.size, plane=pl)
    segs = [s for s in res.segs if not s.construction]
    if not segs:
        raise SketchError("The beam sketch has no lines or arcs")

    def g(xy) -> list[float]:
        return [float(v) for v in pl.pt(xy[0], xy[1])]
    # names of the nodes: the lowest point id at a place
    place: dict[tuple[float, float], str] = {}
    for pid in sorted(res.pts, key=_num):
        place.setdefault(res.pts[pid], pid)
    node_pts = [res.pts[p] for p in sk.node_points()]

    def node_at(xy, auto: str) -> str:
        key = next((k for k in place if math.hypot(k[0] - xy[0], k[1] - xy[1]) <= res.tol), None)
        if key is None:
            place[(float(xy[0]), float(xy[1]))] = auto
            return auto
        return place[key]
    # elementary pieces: each curve is cut at the nodes that lie in its interior (and at the crossings with `split`)
    cuts: dict[str, list[tuple[float, tuple[float, float], str | None]]] = {}
    for s in segs:
        cl: list[tuple[float, tuple[float, float], str | None]] = []
        for p in node_pts:
            if s.kind != "circle" and (math.hypot(p[0] - s.p0[0], p[1] - s.p0[1]) <= res.tol or math.hypot(p[0] - s.p1[0], p[1] - s.p1[1]) <= res.tol):
                continue
            if _dist_to_seg(s, p) <= max(res.tol * 5, 1e-12):
                t = _curve_param(s, p)
                if s.kind == "circle":
                    t %= 1.0
                if 0 < t < 1:
                    cl.append((t, p, None))
        cuts[s.id] = cl
    if sk.frame.get("split"):
        for i, a in enumerate(segs):
            for b in segs[i + 1:]:
                if a.entity == b.entity and a.kind == "line" and b.kind == "line":
                    continue
                for k, p in enumerate(sorted(_intersections(a, b, res.tol), key=lambda q: _curve_param(a, q)), start=1):
                    crossing_name = f"X_{a.id}_{b.id}" + ("" if k == 1 else f"_{k}")     # a name of the pair, not a running number
                    for s in (a, b):
                        t = _curve_param(s, p)
                        if s.kind == "circle":
                            t %= 1.0
                        cuts[s.id].append((t, p, crossing_name))
    for s in segs:
        pts = sorted(cuts[s.id], key=lambda c: c[0])
        # merge the cuts that coincide (the same crossing from several curves)
        merged: list[tuple[float, tuple[float, float], str | None]] = []
        for c in pts:
            if merged and abs(c[0] - merged[-1][0]) * max(s.length(), 1e-9) <= res.tol * 5:
                continue
            merged.append(c)
        cuts[s.id] = merged
    ents = {e.id: e for e in sk.entities}
    for s in segs:
        e = ents[s.entity]
        cl = cuts[s.id]
        if s.kind == "circle":
            # a ring: two half circles from the point at angle 0 and the opposite point (a cut that falls in the way splits further)
            ts = [0.0, 0.5] if not cl else sorted({0.0, 0.5, *[c[0] for c in cl]})
            ts.append(1.0)
            nodes_t = []
            for t in ts:
                xy = s.at(t % 1.0)
                nodes_t.append((t, xy, node_at(xy, f"{s.entity}_t{int(round((t % 1.0) * 1000)):03d}")))      # by the position, not the index
        else:
            nodes_t = [(0.0, s.p0, node_at(s.p0, f"{s.entity}_a")), *[(c[0], c[1], node_at(c[1], c[2] or f"{s.entity}_c{int(round(c[0] * 1000)):03d}"))
                                                                      for c in cl],
                       (1.0, s.p1, node_at(s.p1, f"{s.entity}_b"))]
        n_parts = len(nodes_t) - 1
        for k in range(n_parts):
            t0, xy0, n0 = nodes_t[k]
            t1, xy1, n1 = nodes_t[k + 1]
            if n0 == n1:
                raise SketchError(f"{s.entity}: the member would start and end in the same node", [s.entity])
            # a split piece is named after the node where it starts: a new node elsewhere on the curve keeps the other names
            name = s.id if n_parts == 1 else (n0 if n0.startswith(f"{s.entity}_") else f"{s.id}_{n0}")
            fb.nodes.setdefault(n0, g(xy0))
            fb.nodes.setdefault(n1, g(xy1))
            m: dict = {"start": n0, "end": n1, "entity": s.entity, "section": e.section, "material": e.material}
            if s.kind == "line":
                ax = np.asarray(g(xy1)) - np.asarray(g(xy0))
                ax = ax / max(np.linalg.norm(ax), 1e-300)
                m["up"] = [float(v) for v in (pl.ey if abs(float(ax @ pl.ey)) < 0.99 else pl.ex)]
            else:
                tm = 0.5 * (t0 + t1)
                m["through"] = g(s.at(tm % 1.0 if s.kind == "circle" else tm))
                m["up"] = "radial"
            fb.members[name] = m
    # supports
    n_out = int(np.argmax(np.abs(pl.n)))
    ex_i, ey_i = int(np.argmax(np.abs(pl.ex))), int(np.argmax(np.abs(pl.ey)))
    restrain_plane = bool(sk.frame.get("plane_restraint", True))
    plane_dofs = [False] * 6
    if restrain_plane:
        plane_dofs[n_out] = True
        for k in range(3):
            if k != n_out:
                plane_dofs[3 + k] = True
    for name in fb.nodes:
        fb.supports[name] = list(plane_dofs)
    for sup in sk.supports:
        name = place.get(res.pts[sup.point])
        if name is None or name not in fb.nodes:
            fb.warnings.append(f"Support at {sup.point}: the point is not on any member – ignored")
            continue
        d = list(fb.supports[name])
        if sup.kind == "fixed":
            d = [True] * 6
        elif sup.kind == "pinned":
            d[0] = d[1] = d[2] = True
        elif sup.kind == "roller_x":                    # free to move along the sketch x: holds the sketch y
            d[ey_i] = True
        elif sup.kind == "roller_y":
            d[ex_i] = True
        fb.supports[name] = d
    for name in [n for n, d in fb.supports.items() if not any(d)]:
        del fb.supports[name]
    # the same names for the same things between the builds: nodes by the point id, members by the curve id
    return fb


def apply_frame(study, fb: FrameBuild, sk: Sketch) -> list[str]:
    """Writes the members of the sketch into `study.frame` (see the module text); returns the warnings. Updates `sk.built`."""
    from ..beams.model import FrameMember, FrameSupport, SectionDef
    from ..beams.sections import get as get_section
    fr = study.frame
    prev = sk.built or {}
    warn: list[str] = list(fb.warnings)
    old_nodes, old_members = set(prev.get("nodes", [])), set(prev.get("members", []))
    for name in old_members - set(fb.members):
        fr.members.pop(name, None)
    for name in old_nodes - set(fb.nodes):
        fr.nodes.pop(name, None)
    sk_nodes = set(fb.nodes) | old_nodes
    fr.supports = [s for s in fr.supports if s.node not in sk_nodes]
    for name, xyz in fb.nodes.items():
        if name in fr.nodes:
            fr.nodes[name].xyz = list(xyz)
        else:
            fr.add_node(name, xyz)
    default_sec = str(sk.frame.get("section") or "IPE 200")
    default_mat = str(sk.frame["material"]) if sk.frame.get("material") else None
    for name, m in fb.members.items():
        sec = m["section"] or None
        cur = fr.members.get(name)
        sec_name = sec or (cur.section if cur is not None else default_sec)
        if sec_name not in fr.sections:
            sd = SectionDef(sec_name, library=sec_name)
            try:
                get_section(sec_name)
            except Exception as e:  # noqa: BLE001 – an unknown profile name
                raise SketchError(f"{m['entity']}: section '{sec_name}' is not in the library ({e}) – define it in the Beams step or "
                                  f"use a name such as 'IPE 200'", [m["entity"]]) from None
            fr.sections[sec_name] = sd
        if cur is None:
            fr.members[name] = FrameMember(name, m["start"], m["end"], sec_name, m["material"] or default_mat, m["up"], 0.0,
                                           m.get("through"))
        else:
            cur.start, cur.end, cur.up, cur.through = m["start"], m["end"], m["up"], m.get("through")
            if sec:
                cur.section = sec
            if m["material"]:
                cur.material = m["material"]
    for node, dofs in fb.supports.items():
        fr.supports.append(FrameSupport(node, list(dofs)))
    sk.built = {**{k: v for k, v in prev.items() if k not in ("nodes", "members")}, "nodes": sorted(fb.nodes), "members": sorted(fb.members)}
    gone_n, gone_m = (old_nodes - set(fb.nodes)), (old_members - set(fb.members))
    for lc in study.load_cases:
        for l in lc.loads:
            node = getattr(l, "node", None)
            member = getattr(l, "member", None)
            members = list(getattr(l, "members", None) or [])
            if node in gone_n or member in gone_m or (members and set(members) & gone_m):
                what = node or member or ", ".join(members)
                warn.append(f"Load case '{lc.name}': a load refers to {what}, which the sketch no longer has – remove or move it")
    return warn

"""Data model of the sketch (`Study.sketch`, written to the study file only when a sketch is used).

A sketch is the drawing of a 2D section (`kind` = "twod": plane stress, plane strain or axisymmetric) or of a beam system
(`kind` = "frame"). Everything is typed: named parameters (`params`, expressions), points (`points`, every coordinate an
expression) and entities (`entities`) that refer to the points. There is NO geometric constraint solver: a point is where its
two expressions say, a parameter change moves the points that use it. Ids of points ("P1") and entities ("E1") are stable and
never reused – they are the anchors of the selections, the loads and the members when the sketch is built again.

Entity types (`points` = ids of points, `center` = id of a point):
  line        points [start, end]
  arc3        points [start, through, end]                       the arc passes through the three points
  arc_center  center, points [start, end]                        counter-clockwise from start to end, radius = |start − center|
  circle      center, radius (expression)
  rect        points [corner, opposite corner]                   axis-parallel, sides E<n>.1 … .4 (bottom, right, top, left)
  polyline    points [p1, p2, …], closed                          segments E<n>.1 …
Entity attributes: name (the selection of its edges is called like this), construction (drawn but not part of the geometry),
hole (closed entity inside a region: True = hole, False = a part of its own, None = automatic), section / material (frame).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .expr import SketchError, check_name, eval_params, evaluate, names_in

KINDS = ("twod", "frame")
MODES = ("plane_stress", "plane_strain", "axisym")
ENTITY_TYPES = ("line", "arc3", "arc_center", "circle", "rect", "polyline")
SUPPORT_KINDS = ("fixed", "pinned", "roller_x", "roller_y", "node")
MAX_ENTITIES = 20000
MAX_POINTS = 60000
MAX_PARAMS = 300
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,23}$")
NAME_MAX = 80


def _known(klass, d: dict | None):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


def _clean_text(v) -> str:
    """A name from a file: text without control characters, at most NAME_MAX characters."""
    return "".join(c for c in str(v or "") if ord(c) >= 32)[:NAME_MAX].strip()


def _num_or_text(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v).strip()


# ------------------------------------------------------------------ plane of the sketch in 3D

@dataclass(frozen=True)
class Plane3:
    """Plane of the sketch: point = origin + x·ex + y·ey, normal n = ex × ey. Axisym: x = radius, y = the axis of rotation."""
    origin: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    n: np.ndarray

    def pt(self, x: float, y: float, w: float = 0.0) -> np.ndarray:
        return self.origin + x * self.ex + y * self.ey + w * self.n

    def to_local(self, p) -> tuple[float, float, float]:
        d = np.asarray(p, float) - self.origin
        return float(d @ self.ex), float(d @ self.ey), float(d @ self.n)

    def axis_aligned(self) -> bool:
        return all(abs(float(np.max(np.abs(v))) - 1.0) < 1e-9 for v in (self.ex, self.ey, self.n))


NAMED_PLANES = {"XY": ([1, 0, 0], [0, 1, 0]), "XZ": ([1, 0, 0], [0, 0, 1]), "YZ": ([0, 1, 0], [0, 0, 1])}


def make_plane(spec, origin=None) -> Plane3:
    """Plane from "XY" | "XZ" | "YZ" or {"origin": [..], "normal": [..], "xdir": [..]}."""
    if spec is None:
        spec = "XY"
    if isinstance(spec, str):
        key = spec.strip().upper()
        if key not in NAMED_PLANES:
            raise SketchError(f"Unknown plane '{spec}': use XY, XZ, YZ or give origin, normal and xdir")
        ex, ey = (np.array(v, float) for v in NAMED_PLANES[key])
        return Plane3(np.asarray(origin if origin is not None else [0.0, 0.0, 0.0], float), ex, ey, np.cross(ex, ey))
    if not isinstance(spec, dict):
        raise SketchError("plane must be a name (XY, XZ, YZ) or {origin, normal, xdir}")
    try:
        o = np.asarray(spec.get("origin", origin if origin is not None else [0, 0, 0]), float).reshape(3)
        nrm = np.asarray(spec.get("normal", [0, 0, 1]), float).reshape(3)
        xd = np.asarray(spec.get("xdir", [1, 0, 0]), float).reshape(3)
    except (TypeError, ValueError):
        raise SketchError("plane: origin, normal and xdir must be vectors [x, y, z]") from None
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(nrm)) and np.all(np.isfinite(xd))):
        raise SketchError("plane: the vectors must be finite numbers")
    ln = float(np.linalg.norm(nrm))
    if ln < 1e-12:
        raise SketchError("plane: the normal has zero length")
    nrm = nrm / ln
    xd = xd - (xd @ nrm) * nrm
    lx = float(np.linalg.norm(xd))
    if lx < 1e-9:
        raise SketchError("plane: xdir is parallel to the normal")
    ex = xd / lx
    return Plane3(o, ex, np.cross(nrm, ex), nrm)


# ------------------------------------------------------------------ items of the sketch

@dataclass
class Entity:
    id: str
    type: str
    points: list[str] = field(default_factory=list)
    center: str | None = None
    radius: str | float | None = None
    name: str = ""
    closed: bool = False
    construction: bool = False
    hole: bool | None = None
    section: str | None = None
    material: str | None = None

    @property
    def label(self) -> str:
        """Name under which the selections of its edges appear (the name, otherwise the id)."""
        return self.name or self.id

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        for k in ("points", "center", "radius", "name", "closed", "construction", "hole", "section", "material"):
            v = getattr(self, k)
            if v not in (None, "", False, []):
                d[k] = list(v) if isinstance(v, list) else v
        return d


@dataclass
class SketchSupport:
    """Support of a point of a beam sketch (the point becomes a node): fixed | pinned | roller_x (moves along x) | roller_y | node (no support – a node for a load / a split)."""
    point: str
    kind: str = "fixed"


@dataclass
class SketchRegion:
    """Names a region of a 2D sketch: the region that contains the point `point` (expressions) is the part `name`; `material` =
    name of a material of the study (None = the default material)."""
    point: list = field(default_factory=lambda: ["0", "0"])
    name: str = ""
    material: str | None = None


@dataclass
class Sketch:
    kind: str = "twod"
    mode: str = "plane_stress"                                  # twod: plane_stress | plane_strain | axisym
    thickness: str | float = "1"                                # planar problems [mm] (expression)
    plane: str | dict = "XY"                                    # "XY" | "XZ" | "YZ" | {origin, normal, xdir}; axisym: x = radius, y = axis
    params: dict = field(default_factory=dict)                  # {name: expression} in mm
    points: dict = field(default_factory=dict)                  # {id: [x_expr, y_expr]}
    entities: list[Entity] = field(default_factory=list)
    supports: list[SketchSupport] = field(default_factory=list)  # frame
    regions: list[SketchRegion] = field(default_factory=list)    # twod
    frame: dict = field(default_factory=dict)                   # frame defaults {section, material, split, plane_restraint}
    next_entity: int = 1
    next_point: int = 1
    built: dict = field(default_factory=dict)                   # what the last build created in the study (frame: nodes / members)

    # ---------------------------------------------------------------- plane / parameters
    def plane3(self) -> Plane3:
        return make_plane(self.plane)

    def values(self) -> dict[str, float]:
        return eval_params(self.params)

    def thickness_value(self, env: dict[str, float] | None = None) -> float:
        t = evaluate(self.thickness, env if env is not None else self.values())
        if not t > 0:
            raise SketchError("The thickness must be positive")
        return t

    def set_param(self, name: str, expr) -> None:
        check_name(name)
        if isinstance(expr, bool) or expr is None:
            raise SketchError(f"Parameter '{name}': give a number or an expression")
        if name not in self.params and len(self.params) >= MAX_PARAMS:
            raise SketchError(f"At most {MAX_PARAMS} parameters")
        old = self.params.get(name)
        self.params[name] = _num_or_text(expr)
        try:
            self.values()
        except Exception:             # any failure leaves the parameters as they were
            if old is None:
                del self.params[name]
            else:
                self.params[name] = old
            raise

    def remove_param(self, name: str) -> None:
        if name not in self.params:
            raise SketchError(f"Parameter '{name}' does not exist")
        users = [k for k, v in self.params.items() if k != name and name in names_in(v)]
        users += [f"point {p}" for p, xy in self.points.items() if any(name in names_in(c) for c in xy)]
        users += [f"{e.id} radius" for e in self.entities if e.radius is not None and name in names_in(e.radius)]
        if name in names_in(self.thickness):
            users.append("thickness")
        if users:
            raise SketchError(f"Parameter '{name}' is used by: " + ", ".join(users[:6]))
        del self.params[name]

    # ---------------------------------------------------------------- points / entities
    def add_point(self, x, y) -> str:
        xy = [_num_or_text(x), _num_or_text(y)]
        idx = self.__dict__.get("_pidx")
        if idx is None or idx[0] != len(self.points):            # the same expressions = the same point (exactly connected)
            idx = (len(self.points), {(repr(a), repr(b)): k for k, (a, b) in self.points.items()})
        key = (repr(xy[0]), repr(xy[1]))
        if key in idx[1]:
            self.__dict__["_pidx"] = idx
            return idx[1][key]
        if len(self.points) >= MAX_POINTS:
            raise SketchError(f"At most {MAX_POINTS} points")
        pid = f"P{self.next_point}"
        self.next_point += 1
        self.points[pid] = xy
        idx[1][key] = pid
        self.__dict__["_pidx"] = (len(self.points), idx[1])
        return pid

    def point_ref(self, p) -> str:
        """A point id ("P3" – must exist) or a pair [x, y] (expressions → a new or an existing point)."""
        if isinstance(p, str):
            key = p.strip()
            if key in self.points:
                return key
            raise SketchError(f"Point '{p}' does not exist (points: {', '.join(list(self.points)[:12]) or 'none'})")
        if isinstance(p, (list, tuple)) and len(p) == 2:
            return self.add_point(p[0], p[1])
        raise SketchError(f"A point must be an id or [x, y], got {p!r}")

    def entity(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise SketchError(f"Entity '{eid}' does not exist (entities: {', '.join(x.id for x in self.entities[:12]) or 'none'})")

    def add_entity(self, type: str, points=None, center=None, radius=None, name: str = "", closed: bool = False,
                   construction: bool = False, hole: bool | None = None, section: str | None = None,
                   material: str | None = None) -> Entity:
        if type not in ENTITY_TYPES:
            raise SketchError(f"Unknown entity type '{type}' (allowed: {', '.join(ENTITY_TYPES)})")
        if len(self.entities) >= MAX_ENTITIES:
            raise SketchError(f"At most {MAX_ENTITIES} entities")
        pts = list(points or [])
        need = {"line": 2, "arc3": 3, "arc_center": 2, "circle": 0, "rect": 2}.get(type)
        if need is not None and len(pts) != need:
            raise SketchError(f"{type} needs {need} point(s)" + (" and a center" if type in ("arc_center", "circle") else "")
                              + f", got {len(pts)}")
        if type == "polyline" and len(pts) < 2:
            raise SketchError("A polyline needs at least 2 points")
        if type in ("arc_center", "circle") and center is None:
            raise SketchError(f"{type} needs a center point")
        if type == "circle" and radius is None:
            raise SketchError("A circle needs a radius (a number or an expression)")
        if name:
            self.check_name(name, f"E{self.next_entity}")          # length, control characters, looks like an id / a side, unique
        # the points are created only after everything else is checked; a failed call leaves no stray point behind
        saved = (dict(self.points), self.next_point)
        try:
            ids = [self.point_ref(p) for p in pts]
            cid = self.point_ref(center) if center is not None else None
            env = self.values()
            if type == "circle":
                if evaluate(radius, env) <= 0:
                    raise SketchError("The radius of the circle must be positive")
            for p in ids + ([cid] if cid else []):
                self.point_values(p, env)
        except Exception:             # whatever failed, no stray point stays behind
            self.points, self.next_point = saved
            self.__dict__.pop("_pidx", None)
            raise
        e = Entity(f"E{self.next_entity}", type, ids, cid, None if radius is None else _num_or_text(radius), name or "",
                   bool(closed) if type == "polyline" else False, bool(construction), hole, section, material)
        self.next_entity += 1
        self.entities.append(e)
        return e

    def check_name(self, name: str, eid: str | None) -> None:
        """The name of an entity (= the name of the selection of its edges) must be unique, must not look like an id (E3, E3.2) or
        like a side (name.1) and must be printable text."""
        if not name:
            return
        if len(name) > NAME_MAX or any(ord(c) < 32 for c in name) or name != name.strip():
            raise SketchError("The name is too long (at most 80 characters), has leading / trailing spaces or contains control characters")
        if re.fullmatch(r"E\d+(\.\d+)?", name) and name != eid:
            raise SketchError(f"'{name}' looks like an entity id – choose another name")
        if re.fullmatch(r".+\.\d+", name):
            raise SketchError(f"'{name}' ends with .<number> (that is the notation of the sides of a rectangle / polyline) – choose another name")
        if any(e.name == name and e.id != eid for e in self.entities):
            raise SketchError(f"The name '{name}' is already used by another entity")

    def set_name(self, eid: str, name: str) -> None:
        e = self.entity(eid)
        self.check_name(name.strip(), eid)
        e.name = name.strip()

    def point_values(self, pid: str, env: dict[str, float] | None = None) -> tuple[float, float]:
        env = env if env is not None else self.values()
        try:
            x, y = self.points[pid]
        except KeyError:
            raise SketchError(f"Point '{pid}' does not exist") from None
        try:
            return evaluate(x, env), evaluate(y, env)
        except SketchError as e:
            raise SketchError(f"Point {pid}: {e}", [pid]) from None

    def used_points(self) -> set[str]:
        used = {p for e in self.entities for p in e.points} | {e.center for e in self.entities if e.center}
        return used | {s.point for s in self.supports}

    def node_points(self) -> set[str]:
        """Points that are places of the drawing (end points of curves, supports) – not the centres of arcs / circles and not the
        point an arc passes through."""
        out = {s.point for s in self.supports}
        for e in self.entities:
            if e.type == "arc3":
                out |= {e.points[0], e.points[-1]}
            elif e.type != "circle":
                out |= set(e.points)
        return out

    def _drop_orphans(self) -> None:
        self.__dict__.pop("_pidx", None)
        used = self.used_points()
        for pid in [p for p in self.points if p not in used]:
            del self.points[pid]

    def remove_entity(self, eid: str) -> None:
        self.entities.remove(self.entity(eid))
        self._drop_orphans()

    def move_point(self, pid: str, x, y) -> None:
        self.point_values(pid)
        self.__dict__.pop("_pidx", None)
        old = self.points[pid]
        self.points[pid] = [_num_or_text(x), _num_or_text(y)]
        try:
            self.point_values(pid)
        except SketchError:
            self.points[pid] = old
            raise

    def add_support(self, point, kind: str = "fixed") -> SketchSupport:
        if self.kind != "frame":
            raise SketchError("Supports belong to beam sketches (kind frame)")
        if kind not in SUPPORT_KINDS:
            raise SketchError(f"Unknown support '{kind}' (allowed: {', '.join(SUPPORT_KINDS)})")
        pid = self.point_ref(point)
        self.supports = [s for s in self.supports if s.point != pid]
        s = SketchSupport(pid, kind)
        self.supports.append(s)
        return s

    def remove_support(self, point: str) -> None:
        self.supports = [s for s in self.supports if s.point != point]
        self._drop_orphans()

    def restore(self, other: "Sketch") -> None:
        """Takes over everything of `other` (undo, a failed import): the object stays the same, the caches are dropped."""
        self.__dict__.clear()
        self.__dict__.update(other.__dict__)

    # ---------------------------------------------------------------- JSON
    def copy(self) -> "Sketch":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "plane": copy.deepcopy(self.plane), "params": dict(self.params),
                   "points": {k: list(v) for k, v in self.points.items()},
                   "entities": [e.to_dict() for e in self.entities],
                   "next_entity": self.next_entity, "next_point": self.next_point}
        if self.kind == "twod":
            d["mode"] = self.mode
            d["thickness"] = self.thickness
            if self.regions:
                d["regions"] = [{"point": list(r.point), "name": r.name, "material": r.material} for r in self.regions]
        else:
            if self.supports:
                d["supports"] = [{"point": s.point, "kind": s.kind} for s in self.supports]
            if self.frame:
                d["frame"] = dict(self.frame)
        if self.built:
            d["built"] = copy.deepcopy(self.built)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "Sketch | None":
        if not d or not isinstance(d, dict):
            return None
        kind = d.get("kind", "twod")
        if kind not in KINDS:
            raise SketchError(f"Unknown sketch kind '{kind}'")
        sk = cls(kind=kind)
        mode = d.get("mode", "plane_stress")
        sk.mode = mode if mode in MODES else "plane_stress"
        sk.thickness = _num_or_text(d.get("thickness", "1"))
        plane = d.get("plane", "XY")
        sk.plane = plane if isinstance(plane, (str, dict)) else "XY"
        params = d.get("params") or {}
        if not isinstance(params, dict) or len(params) > MAX_PARAMS:
            raise SketchError("The sketch parameters are not valid")
        sk.params = {check_name(str(k)): _num_or_text(v) for k, v in params.items()}
        pts = d.get("points") or {}
        if not isinstance(pts, dict) or len(pts) > MAX_POINTS:
            raise SketchError("The sketch points are not valid")
        for k, v in pts.items():
            if not ID_RE.match(str(k)) or not isinstance(v, (list, tuple)) or len(v) != 2:
                raise SketchError(f"Sketch point {k!r} is not valid")
            sk.points[str(k)] = [_num_or_text(c) for c in v]
        ents = d.get("entities") or []
        if not isinstance(ents, list) or len(ents) > MAX_ENTITIES:
            raise SketchError("The sketch entities are not valid")
        for e in ents:
            if not isinstance(e, dict) or e.get("type") not in ENTITY_TYPES or not ID_RE.match(str(e.get("id", ""))):
                raise SketchError(f"Sketch entity {e.get('id') if isinstance(e, dict) else e!r} is not valid")
            ent = _known(Entity, e)
            if any(x.id == ent.id for x in sk.entities):
                raise SketchError(f"The sketch has two entities with the id {ent.id}")
            ent.points = [str(p) for p in ent.points or []]
            ent.name = _clean_text(ent.name)
            sk.check_name(ent.name, ent.id)                    # unique, not an id (E3) and not a side (name.1): the selections would clash
            if ent.center is not None:
                ent.center = str(ent.center)
            need = {"line": 2, "arc3": 3, "arc_center": 2, "circle": 0, "rect": 2}.get(ent.type)
            if (need is not None and len(ent.points) != need) or (ent.type == "polyline" and len(ent.points) < 2)                     or (ent.type in ("arc_center", "circle") and not ent.center) or (ent.type == "circle" and ent.radius is None):
                raise SketchError(f"Sketch entity {ent.id} ({ent.type}) has a wrong number of points")
            ent.closed, ent.construction = bool(ent.closed), bool(ent.construction)
            ent.hole = ent.hole if isinstance(ent.hole, bool) else None
            ent.section = str(ent.section)[:NAME_MAX] if ent.section else None
            ent.material = str(ent.material)[:NAME_MAX] if ent.material else None
            ent.radius = None if ent.radius is None else _num_or_text(ent.radius)
            for p in ent.points + ([ent.center] if ent.center else []):
                if p not in sk.points:
                    raise SketchError(f"Entity {ent.id} refers to the missing point {p}")
            sk.entities.append(ent)
        sk.supports = [_known(SketchSupport, s) for s in d.get("supports") or [] if isinstance(s, dict)]
        sk.supports = [s for s in sk.supports if s.point in sk.points and s.kind in SUPPORT_KINDS]
        sk.regions = [_known(SketchRegion, r) for r in d.get("regions") or [] if isinstance(r, dict)]
        for r in sk.regions:
            r.name = _clean_text(r.name)
            r.material = str(r.material)[:NAME_MAX] if r.material else None
            if not (isinstance(r.point, (list, tuple)) and len(r.point) == 2):
                raise SketchError("A region name has no valid point")
            r.point = [_num_or_text(c) for c in r.point]
        fr_raw = d.get("frame")
        fr: dict = fr_raw if isinstance(fr_raw, dict) else {}
        sk.frame = {"section": str(fr["section"])[:NAME_MAX]} if fr.get("section") else {}
        if fr.get("material"):
            sk.frame["material"] = str(fr["material"])[:NAME_MAX]
        sk.frame["plane_restraint"] = bool(fr.get("plane_restraint", True))
        sk.frame["split"] = bool(fr.get("split", False))
        nums = [int(re.sub(r"\D", "", e.id) or 0) for e in sk.entities]
        sk.next_entity = max(int(d.get("next_entity", 1) or 1), max(nums, default=0) + 1)
        pn = [int(re.sub(r"\D", "", k) or 0) for k in sk.points]
        sk.next_point = max(int(d.get("next_point", 1) or 1), max(pn, default=0) + 1)
        sk.built = dict(d.get("built") or {})
        return sk


def new_sketch(kind: str = "twod", mode: str | None = None, plane=None, thickness: float | str = 1.0) -> Sketch:
    """Empty sketch. twod: mode plane_stress | plane_strain | axisym; frame: a beam system (default plane XZ = vertical)."""
    if kind not in KINDS:
        raise SketchError(f"Unknown kind '{kind}' (twod | frame)")
    sk = Sketch(kind=kind)
    if kind == "twod":
        sk.mode = mode or "plane_stress"
        if sk.mode not in MODES:
            raise SketchError(f"Unknown mode '{mode}' (allowed: {', '.join(MODES)})")
        sk.thickness = _num_or_text(thickness)
        sk.plane = plane if plane is not None else "XY"
    else:
        sk.plane = plane if plane is not None else "XZ"
        sk.frame = {"section": "IPE 200", "plane_restraint": True, "split": False}
    pl = make_plane(sk.plane)
    if kind == "frame" and not pl.axis_aligned():
        raise SketchError("A beam sketch must lie in a plane parallel to XY, XZ or YZ (the supports and the plane restraint use the global axes)")
    return sk


# ------------------------------------------------------------------ numeric geometry

@dataclass
class Seg:
    """One elementary curve of the sketch. line: p0 → p1; arc: centre c, radius r, starts at angle a0 and turns by `sweep` (> 0)
    in the direction `sign` (+1 counter-clockwise); circle: a full circle (p0 = p1 = the point at angle 0)."""
    entity: str
    id: str
    kind: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    c: tuple[float, float] | None = None
    r: float = 0.0
    a0: float = 0.0
    sweep: float = 0.0
    sign: int = 1
    construction: bool = False

    def at(self, t: float) -> tuple[float, float]:
        """Point at the parameter t ∈ [0, 1] along the curve."""
        if self.kind == "line":
            return (self.p0[0] + t * (self.p1[0] - self.p0[0]), self.p0[1] + t * (self.p1[1] - self.p0[1]))
        assert self.c is not None
        a = self.a0 + self.sign * t * self.sweep
        return (self.c[0] + self.r * math.cos(a), self.c[1] + self.r * math.sin(a))

    def mid(self) -> tuple[float, float]:
        return self.at(0.5)

    def param(self, p) -> float:
        """Parameter t of the projection of the point p on the curve (a line: t may lie outside [0, 1]; an arc: angle-based, in [0, 1)
        along the arc, > 1 beyond its end; a circle: [0, 1))."""
        if self.kind == "line":
            dx, dy = self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]
            return ((p[0] - self.p0[0]) * dx + (p[1] - self.p0[1]) * dy) / max(dx * dx + dy * dy, 1e-300)
        assert self.c is not None
        a = math.atan2(p[1] - self.c[1], p[0] - self.c[0])
        d = ((a - self.a0) * self.sign) % (2 * math.pi)
        if self.kind != "circle" and d > 2 * math.pi - 1e-9:        # a hair before the start of the arc
            d -= 2 * math.pi
        return d / self.sweep

    def distance(self, p) -> float:
        """Distance of the point p from the curve."""
        if self.kind == "line":
            q = self.at(min(max(self.param(p), 0.0), 1.0))
            return math.hypot(p[0] - q[0], p[1] - q[1])
        assert self.c is not None
        t = self.param(p)
        if self.kind == "circle" or 0.0 <= t <= 1.0:
            return abs(math.hypot(p[0] - self.c[0], p[1] - self.c[1]) - self.r)
        return min(math.hypot(p[0] - self.p0[0], p[1] - self.p0[1]), math.hypot(p[0] - self.p1[0], p[1] - self.p1[1]))

    def length(self) -> float:
        if self.kind == "line":
            return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        return self.r * self.sweep

    def polyline(self, step_deg: float = 0.5) -> list[tuple[float, float]]:
        if self.kind == "line":
            return [self.p0, self.p1]
        n = max(8, int(math.ceil(math.degrees(self.sweep) / step_deg)))
        return [self.at(i / n) for i in range(n + 1)]


def _circle3(a, b, c) -> tuple[tuple[float, float], float] | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12 * max(1.0, abs(ax) + abs(bx) + abs(cx)) ** 2:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def _ang(c, p) -> float:
    return math.atan2(p[1] - c[1], p[0] - c[0])


@dataclass
class Resolved:
    """Numeric sketch: parameter values, snapped point coordinates, elementary curves. `size` = the diagonal of the extent."""
    sketch: Sketch
    env: dict
    pts: dict
    segs: list[Seg]
    size: float
    tol: float

    def segs_of(self, entity: str) -> list[Seg]:
        return [s for s in self.segs if s.entity == entity]


def resolve(sk: Sketch) -> Resolved:
    """Evaluates the parameters and the expressions and turns the entities into elementary curves. Points closer than 1e-6 of the
    size of the sketch are one point (typed coordinates that differ by rounding). Errors name the entity."""
    env = sk.values()
    raw = {pid: sk.point_values(pid, env) for pid in sk.points}
    size = 0.0
    if raw:
        xs = [v[0] for v in raw.values()]
        ys = [v[1] for v in raw.values()]
        size = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    for e in sk.entities:                                        # a circle may be bigger than its centre point shows
        if e.type == "circle" and e.radius is not None:
            try:
                size = max(size, 2.0 * abs(evaluate(e.radius, env)))
            except SketchError:
                pass
    size = max(size, 1e-9)
    tol = 1e-6 * size
    canon: dict[str, tuple[float, float]] = {}
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    cell = max(tol * 4, 1e-12)
    for pid, (x, y) in raw.items():
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in cells.get((key[0] + dx, key[1] + dy), ()):
                    if math.hypot(q[0] - x, q[1] - y) <= tol:
                        hit = q
                        break
                if hit:
                    break
            if hit:
                break
        if hit is None:
            hit = (x, y)
            cells.setdefault(key, []).append(hit)
        canon[pid] = hit
    segs: list[Seg] = []

    def line(e: Entity, sid: str, a, b):
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= tol:
            raise SketchError(f"{e.id}: zero-length line", [e.id])
        segs.append(Seg(e.id, sid, "line", a, b, construction=e.construction))

    for e in sk.entities:
        P = [canon[p] for p in e.points]
        if e.type == "line":
            line(e, e.id, P[0], P[1])
        elif e.type == "rect":
            (x0, y0), (x1, y1) = P
            xa, xb, ya, yb = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            if xb - xa <= tol or yb - ya <= tol:
                raise SketchError(f"{e.id}: the rectangle has a zero side", [e.id])
            cs = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
            for i in range(4):
                line(e, f"{e.id}.{i + 1}", cs[i], cs[(i + 1) % 4])
        elif e.type == "polyline":
            m = len(P) if e.closed and len(P) > 2 else len(P) - 1
            for i in range(m):
                line(e, f"{e.id}.{i + 1}", P[i], P[(i + 1) % len(P)])
        elif e.type == "circle":
            assert e.center is not None
            c = canon[e.center]
            try:
                r = evaluate(e.radius, env)
            except SketchError as ex:
                raise SketchError(f"{e.id}: radius – {ex}", [e.id]) from None
            if r <= tol:
                raise SketchError(f"{e.id}: the radius of the circle must be positive", [e.id])
            p0 = (c[0] + r, c[1])
            segs.append(Seg(e.id, e.id, "circle", p0, p0, c, r, 0.0, 2 * math.pi, 1, e.construction))
        elif e.type == "arc_center":
            assert e.center is not None
            c = canon[e.center]
            s, t = P
            r = math.hypot(s[0] - c[0], s[1] - c[1])
            if r <= tol or math.hypot(t[0] - c[0], t[1] - c[1]) <= tol:
                raise SketchError(f"{e.id}: the arc end coincides with its centre", [e.id])
            a0, a1 = _ang(c, s), _ang(c, t)
            sweep = (a1 - a0) % (2 * math.pi)
            if sweep <= 1e-9:
                raise SketchError(f"{e.id}: the arc has the same start and end direction (use a circle)", [e.id])
            rt = math.hypot(t[0] - c[0], t[1] - c[1])
            end = t if abs(rt - r) <= 10 * tol else (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
            segs.append(Seg(e.id, e.id, "arc", s, end, c, r, a0, sweep, 1, e.construction))
        elif e.type == "arc3":
            s, m, t = P
            cc = _circle3(s, m, t)
            if cc is None:
                raise SketchError(f"{e.id}: the three points of the arc lie on a line (use a line)", [e.id])
            c, r = cc
            a0, am, a1 = _ang(c, s), _ang(c, m), _ang(c, t)
            ccw = (am - a0) % (2 * math.pi) < (a1 - a0) % (2 * math.pi)
            sweep = ((a1 - a0) if ccw else (a0 - a1)) % (2 * math.pi)
            segs.append(Seg(e.id, e.id, "arc", s, t, c, r, a0, sweep, 1 if ccw else -1, e.construction))
    return Resolved(sk, env, canon, segs, size, tol)

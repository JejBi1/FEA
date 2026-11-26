"""Tools: sketcher for 1D / 2D models without CAD (fea_core.sketch, F15-S5) – typed coordinates and named parameters, no constraint solver.

fea_sketch_new → fea_sketch_param / fea_sketch_add (lines, arcs, circles, rectangles, polylines; points with supports; region names)
→ fea_sketch_build (2D: regions → parts + a selection per entity; 1D: nodes and members; a PNG of the sketch with the ids) → the
ordinary tools (materials, constraints, loads, mesh, solve). Changing a parameter and building again keeps the constraints, loads,
materials and mesh settings (the selections follow the entity ids). Sketch parameters are the inputs of a later parametric sweep.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image

from ..sketch.model import Sketch
from ..study import Study
from .common import out
from .session import FeaError, Session, safe_name, user_file

ACCEPT = {"certain": frozenset({"same", "cad"}), "auto": frozenset({"same", "cad", "probable", "split"})}
LIST_LIMIT = 250


def _any(sess: Session) -> Any:
    """The session without the attribute types the checker infers from Session.reset (every attribute starts as None there)."""
    return sess


def _sk(sess: Session) -> tuple[Study, Sketch]:
    st: Study = _any(sess).require_study()
    if st.sketch is None:
        raise FeaError("The study has no sketch – start one with fea_sketch_new")
    return st, st.sketch


def _guard(fn):
    """Errors of the sketch (SketchError) and wrong argument types → FeaError with the text."""
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        from ..sketch.expr import SketchError
        try:
            return fn(*a, **k)
        except SketchError as e:
            raise FeaError(str(e)) from None
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            raise FeaError(f"Invalid sketch input: {e}") from None
    return wrapped


def _points(items) -> list:
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        raise FeaError("points must be a list of point ids ('P1') or [x, y] pairs (numbers or expressions)")
    return list(items)


def _one_point(p):
    if p is None:
        return None
    return p


def _params_table(sk) -> dict:
    vals = sk.values()
    return {k: {"expr": v, "value": vals[k]} for k, v in sk.params.items()}


def _entity_dict(sk, e, vals) -> dict:
    d: dict[str, Any] = {"id": e.id, "type": e.type}
    if e.name:
        d["name"] = e.name
    if e.points:
        d["points"] = list(e.points)
    if e.center:
        d["center"] = e.center
    if e.radius is not None:
        d["radius"] = e.radius
    for k in ("closed", "construction", "hole", "section", "material"):
        v = getattr(e, k)
        if v not in (None, False, ""):
            d[k] = v
    return d


def sketch_state(sess: Session) -> dict:
    """The sketch as a table (parameters with values, points with their numbers, entities, supports, region names)."""
    from ..sketch.build import is_built
    st, sk = _sk(sess)
    vals = sk.values()
    d: dict[str, Any] = {"kind": sk.kind, "plane": sk.plane, "params": _params_table(sk)}
    if sk.kind == "twod":
        d["mode"] = sk.mode
        d["thickness"] = sk.thickness
    pts = {}
    for i, pid in enumerate(sk.points):
        if i >= LIST_LIMIT:
            d["points_truncated"] = len(sk.points)
            break
        x, y = sk.point_values(pid, vals)
        pts[pid] = {"expr": list(sk.points[pid]), "mm": [x, y]}
    d["points"] = pts
    d["entities"] = [_entity_dict(sk, e, vals) for e in sk.entities[:LIST_LIMIT]]
    if len(sk.entities) > LIST_LIMIT:
        d["entities_truncated"] = len(sk.entities)
    if sk.supports:
        d["supports"] = [{"point": s.point, "kind": s.kind} for s in sk.supports]
    if sk.regions:
        d["regions"] = [{"point": list(r.point), "name": r.name, "material": r.material} for r in sk.regions]
    d["built"] = bool((sk.built or {}).get("snapshot")) and is_built(sk)
    d["geometry_built_before"] = bool((sk.built or {}).get("snapshot"))
    return d


def _install(sess: Session, st, step_path: Path | None, parts) -> None:
    """The study with the new geometry goes into the session (as fea_geometry_replace does)."""
    sess = _any(sess)
    sess.study = st
    sess.step_path = step_path
    sess._parts = parts if step_path is not None else None
    sess.geometry_warnings = []
    sess._face_props, sess._edge_props, sess._tess = {}, {}, None
    sess.changed(mesh=True)


def build_sketch(sess: Session, accept: str = "auto") -> tuple[dict, Any]:
    """Body of fea_sketch_build: first build or rebuild with the remap (the caller holds the lock). → (summary, SketchBuild | None)."""
    from ..sketch.build import attach_first, build_geometry, rebuild_study
    from ..sketch.expr import SketchError
    if accept not in ACCEPT:
        raise FeaError("accept must be 'auto' (certain + probable matches and split faces) or 'certain'")
    st, sk = _sk(sess)
    S = _any(sess)
    if S.workdir is None:
        S.workdir = S.new_workdir(st.name)
    wd = Path(S.workdir)
    first = not (sk.built or {}).get("snapshot")
    d: dict[str, Any] = {"built": False}
    try:
        if first:
            b = build_geometry(sk, wd, document=st.name)
            notes = attach_first(st, b, sk)
            if b.kind == "twod":
                _install(sess, st, Path(str(b.step_path)).resolve(), b.parts)
            else:
                sess.changed(mesh=True)
            d["rebuild"] = False
        else:
            new, rep, b = rebuild_study(st, wd, old_parts=None if sk.kind == "frame" else S.parts, accept=ACCEPT[accept])
            notes = []
            _install(sess, new, Path(str(b.step_path)).resolve() if b.step_path else None, b.parts)
            d["rebuild"] = True
            if rep is not None:
                d["remap"] = rep.counts()
                d["not_remapped"] = rep.blocking(ACCEPT[accept])
                if rep.notes:
                    d["notes"] = list(rep.notes)
            st = new
            sk = new.sketch or sk
    except SketchError as e:
        d["error"] = str(e)
        d["entities"] = list(e.entities)
        d["next"] = "fix the entities named above (fea_sketch_list shows the numbers, fea_sketch_remove / fea_sketch_add change them), then fea_sketch_build again"
        return d, e
    d["built"] = True
    if b.kind == "twod":
        d["regions"] = b.regions
        d["selections"] = {k: len(v) for k, v in b.selections.items()}
        d["thickness_mm"] = b.twod.thickness if b.twod.kind != "axisym" else None
        d["analysis"] = {"kind": b.twod.kind, "axis": b.twod.axis if b.twod.kind == "axisym" else None}
        d["next"] = ("fea_materials_assign → fea_constraint_add_fixed / fea_load_add_* with selection = an entity name or id "
                     "(E1, E3.2, the name you gave) → fea_mesh_generate → fea_solve_run")
    else:
        fb = b.frame
        d["nodes"] = {n: [round(v, 6) for v in xyz] for n, xyz in list(fb.nodes.items())[:LIST_LIMIT]}
        d["members"] = {n: {"start": m["start"], "end": m["end"], "entity": m["entity"],
                            "section": (st.frame.members[n].section if n in st.frame.members else None)}
                        for n, m in list(fb.members.items())[:LIST_LIMIT]}
        d["supports"] = [{"point": s.point, "kind": s.kind} for s in sk.supports]
        d["plane_restraint"] = bool(sk.frame.get("plane_restraint", True))
        d["next"] = ("fea_frame_section_set / fea_frame_member_set to change sections, fea_frame_load_node / fea_frame_load_member "
                     "(node = point id, member = entity id) → fea_solve_run")
    warns = list(getattr(b, "warnings", []) or []) + (notes if isinstance(notes, list) else [])
    if warns:
        d["warnings"] = warns
    return d, b


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_new(kind: str = "twod", mode: str = "plane_stress", thickness: float | str = 1.0, plane: str | dict | None = None,
                       name: str | None = None, workdir: str | None = None, material: str | None = None) -> str:
        """Starts a drawn model (2D section or beam system) without CAD and replaces the open study. kind 'twod' (mode plane_stress | plane_strain | axisym; thickness [mm], a number or an expression; axisym: x = radius, y = axis) or 'frame' (default plane XZ: x right, y up). plane 'XY' | 'XZ' | 'YZ' or {origin, normal, xdir}. Typed coordinates in mm and named parameters, no constraint solver. Next: fea_sketch_param, fea_sketch_add, fea_sketch_build."""
        from .. import materials as matlib
        from ..sketch.model import new_sketch
        from ..study import Study
        sk = new_sketch(kind, mode if kind == "twod" else None, plane, thickness)
        mat = None
        if material:
            try:
                mat = matlib.by_name(material)
            except KeyError:
                raise FeaError(f"Material '{material}' is not in the catalogue (fea_materials_list)") from None
        with sess.use():
            S = _any(sess)
            S.reset()
            st = Study(name=name or ("Frame sketch" if kind == "frame" else "Sketch"), geometry="")
            st.add_material(mat or matlib.default(), default=True)
            st.sketch = sk
            S.study = st
            S.workdir = S.new_workdir(st.name, workdir)
            return out({"study": st.name, "kind": kind, "mode": sk.mode if kind == "twod" else None, "workdir": str(S.workdir),
                        "next": "fea_sketch_param(name, expr) for named dimensions, fea_sketch_add(type, points=[[x, y], …]) "
                                "(coordinates and radii may be expressions of the parameters), fea_sketch_build"})

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_param(name: str, expr: float | str | None = None) -> str:
        """Sets a named sketch parameter to a number or an expression (mm); expr=None deletes it. Expressions: + - * / ** (exponent <= 8), parentheses, other parameters, sin cos tan asin acos atan atan2 sqrt abs min max hypot rad deg, pi (angles in radians), e.g. L=200, h='L/10'. Coordinates, radii and the thickness may use parameters; a changed parameter moves the geometry at the next fea_sketch_build and keeps constraints, loads and materials. A parameter that is in use cannot be deleted."""
        with sess.use():
            st, sk = _sk(sess)
            if expr is None:
                sk.remove_param(name)
            else:
                sk.set_param(name, expr)
            sess.changed()
            return out({"params": _params_table(sk), "built": False, "next": "fea_sketch_build applies the change"})

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_add(type: str, points: list | None = None, center: str | list | None = None, radius: float | str | None = None,
                       name: str | None = None, closed: bool = False, construction: bool = False, hole: bool | None = None,
                       section: str | None = None, material: str | None = None, support: str | None = None) -> str:
        """Adds a sketch entity, a beam node / support or a 2D region name. Coordinates are numbers or expressions in mm; a point is [x, y] or an existing id 'P3'. type: line [p1, p2] | arc3 [start, through, end] | arc_center (center, points [start, end], counter-clockwise) | circle (center, radius) | rect [corner, opposite corner] | polyline (points, closed) | point (points [[x, y]], support = fixed | pinned | roller_x | roller_y | node: a beam node for supports and loads, splits a member there) | region (points [[x, y]] inside a 2D region, name, material: names that part). name = a second name for the selection of its edges (always also the id E3; sides of a rect / polyline E3.1 …). hole: a closed entity inside a region (None automatic, False = a part of its own). construction: drawn, not part of the geometry. section / material: beam members."""
        from ..sketch.model import SketchRegion
        with sess.use():
            st, sk = _sk(sess)
            pts = _points(points)
            if type == "point":
                if len(pts) != 1:
                    raise FeaError("a point needs points=[[x, y]] (one point)")
                s = sk.add_support(pts[0], support or "node")
                sess.changed()
                return out({"point": s.point, "support": s.kind, "xy_mm": sk.point_values(s.point), "built": False})
            if type == "region":
                if len(pts) != 1 or not isinstance(pts[0], (list, tuple)) or len(pts[0]) != 2:
                    raise FeaError("a region needs points=[[x, y]] (a point inside it as numbers or expressions), name and optionally material")
                if not name:
                    raise FeaError("a region needs a name")
                if sk.kind != "twod":
                    raise FeaError("Regions belong to 2D sketches")
                sk.regions = [r for r in sk.regions if r.name != name]
                sk.regions.append(SketchRegion([c if isinstance(c, (int, float)) else str(c) for c in pts[0]], name, material))
                sess.changed()
                return out({"region": name, "built": False, "next": "fea_sketch_build"})
            if support:
                raise FeaError("support is for type 'point' (or add the entity, then a point with a support at its end)")
            e = sk.add_entity(type, pts, _one_point(center), radius, name or "", closed, construction, hole, section, material)
            sess.changed()
            return out({"id": e.id, "entity": _entity_dict(sk, e, sk.values()), "built": False, "next": "fea_sketch_build when the drawing is finished"})

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_remove(id: str) -> str:
        """Removes a sketch entity (E3), a support / node point (P2) or a region name. Ids are never reused; a selection, load or constraint that used a removed entity is reported by the next fea_sketch_build."""
        with sess.use():
            st, sk = _sk(sess)
            key = (id or "").strip()
            if any(e.id == key for e in sk.entities):
                sk.remove_entity(key)
            elif any(s.point == key for s in sk.supports):
                sk.remove_support(key)
            elif any(r.name == key for r in sk.regions):
                sk.regions = [r for r in sk.regions if r.name != key]
            else:
                raise FeaError(f"'{id}' is not an entity id, a support point or a region name (fea_sketch_list)")
            sess.changed()
            return out({"removed": key, "entities": [e.id for e in sk.entities], "built": False})

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_list(picture: bool = False):
        """Lists the sketch: parameters with values, points (expressions and mm), entities, supports, region names, and whether the model corresponds to it (built). picture=True also returns a PNG with the entity ids."""
        with sess.use():
            d = sketch_state(sess)
            if not picture:
                return out(d)
            from ..sketch.render import render_png
            st, sk = _sk(sess)
            from .tools_results import _png_path
            p = _png_path(sess, None, "sketch")
            render_png(sk, p, None, 900, 700)
            return [out(d), Image(path=p)]

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_build(accept: str = "auto", mesh_size: float | None = None, refine: dict | None = None, picture: bool = True,
                         output_path: str | None = None):
        """Builds the sketch into the model: 2D parts + a selection per entity edge, or beam nodes and members. 2D: closed regions become parts (planar: extruded by the thickness, axisym: revolved by 360 deg); the selection of an entity edge is called by its id and its name (E5.2 = second side of a rect / polyline) – use it in constraints, loads, refinement. Beam sketch: nodes = point ids, members = entity ids. Build again after changes: constraints, loads, materials and mesh settings are kept (the selections follow the entity ids; accept 'auto' | 'certain'); what could not be kept is listed. mesh_size [mm] = 2D element size, refine {entity name: size [mm]} = local refinement at those edges. Errors name the entity (open contour with the gap size, self-intersection, across the axis) and the PNG marks it. Returns regions / members and the PNG."""
        from ..sketch.render import render_png
        with sess.use(op="building the sketch"):
            d, b = build_sketch(sess, accept)
            st, sk = _sk(sess)
            if d.get("built") and (mesh_size is not None or refine):
                if mesh_size is not None:
                    if not (isinstance(mesh_size, (int, float)) and mesh_size > 0 and math.isfinite(mesh_size)):
                        raise FeaError("mesh_size must be a positive number [mm]")
                    st.mesh.size = float(mesh_size)
                if refine:
                    loc = []
                    for k, v in refine.items():
                        if k not in st.selections:
                            raise FeaError(f"refine: '{k}' is not a selection (entity name / id of the built sketch: {', '.join(list(st.selections)[:10])})")
                        if not (isinstance(v, (int, float)) and v > 0):
                            raise FeaError(f"refine: the size of '{k}' must be a positive number [mm]")
                        loc.append({"selection": k, "size": float(v)})
                    st.mesh.local = loc
                sess.changed(mesh=True)
                d["mesh"] = {"size": st.mesh.size, "local": st.mesh.local}
            if not picture:
                return out(d)
            from .tools_results import _png_path
            p = _png_path(sess, output_path, "sketch")
            problem = b if isinstance(b, Exception) else None
            render_png(sk, p, None if problem else b, 1000, 760, problem=problem)       # type: ignore[arg-type]
            d["image"] = str(p)
            return [out(d), Image(path=p)]

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_to_3d(how: str = "extrude", length: float | str | None = None, angle: float = 360.0):
        """Turns the regions of a 2D sketch into a normal 3D study (extrude or revolve) and replaces the open study; the sketch study is saved first. how 'extrude' (length [mm] along the sketch normal) or 'revolve' (angle [deg] about the sketch y axis, regions at x >= 0). The selections of the entity edges are kept (caps are not named); the 3D model no longer follows the sketch."""
        from ..sketch.build import extrude_to_3d
        with sess.use(op="sketch to 3D"):
            st, sk = _sk(sess)
            if sk.kind != "twod":
                raise FeaError("Only a 2D sketch can be turned into 3D")
            S = _any(sess)
            if S.workdir is None:
                S.workdir = S.new_workdir(st.name)
            wd = Path(S.workdir)
            saved = wd / f"{safe_name(st.name)}_sketch_study.json"
            try:
                st.save(saved)
            except OSError as e:
                raise FeaError(f"The sketch study cannot be saved before the conversion: {e}") from None
            st3, parts, step = extrude_to_3d(sk, wd, how, length, angle, f"{st.name} 3D")
            old_mat = dict(st.materials)
            for k, m in old_mat.items():
                st3.materials.setdefault(k, m)
            st3.default_material = st.default_material if st.default_material in st3.materials else st3.default_material
            S.reset()
            S.study, S.step_path, S._parts = st3, Path(step).resolve(), parts
            S.workdir = wd
            return out({"study": st3.name, "parts": [p.name for p in parts], "selections": {k: len(v.faces) for k, v in st3.selections.items()},
                        "sketch_study_saved": str(saved), "next": "the ordinary 3D workflow: fea_materials_assign, constraints, loads, fea_mesh_generate, fea_solve_run"})

    @mcp.tool(structured_output=False)
    @_guard
    def fea_sketch_import_dxf(path: str, layers: list[str] | None = None, scale: float | None = None, max_curves: int = 5000) -> str:
        """Adds the lines, arcs, circles and polylines of a DXF drawing to the sketch (optional module 'sketch' = ezdxf). Units from the file ($INSUNITS: inch, cm, m … converted to mm; none = mm), scale overrides; layers = only these layers. Splines, ellipses, text and blocks are skipped and counted; a closed polyline without arcs stays one closed polyline. Then fea_sketch_build."""
        from ..sketch.dxf import MAX_BYTES as MAX_DXF_BYTES, import_dxf
        with sess.use(op="DXF import"):
            st, sk = _sk(sess)
            p = user_file(path, MAX_DXF_BYTES, "DXF file")
            if scale is not None and not (math.isfinite(scale) and scale > 0):
                raise FeaError("scale must be a positive number")
            info = import_dxf(sk, p, layers, scale, int(max_curves))
            sess.changed()
            info["built"] = False
            info["next"] = "fea_sketch_list(picture=True) to look at it, fea_sketch_build"
            return out(info)

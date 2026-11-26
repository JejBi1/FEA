"""Sketch → geometry → study; and the rebuild of a study after the sketch changed (parameters, points, entities).

2D (`kind` = "twod"): every material region becomes a part – the face of the region extruded by the thickness (planar problems,
symmetric about the sketch plane) or revolved by 360° about the axis (axisymmetric) – written to `sketch_<n>.step` with a sidecar
`sketch_<n>.json` (the same file the CAD add-ins write, see fea_core.cad_import): its "faces" table gives every side face the
token "<region>/<curve id>" (the ids of the sketch entities are the identity), so the update goes through the ordinary
`cad_link.plan_update` – selections, loads, constraints, materials and mesh settings are kept and what disappeared is reported. The
existing 2D path (`fea_core.twod`) then cuts the mid-plane. A side face of the curve `E3` is the selection `E3` (or the name of
the entity); `E5.2` is the second side of a rectangle / polyline.

1D (`kind` = "frame"): see `frame.py` (nodes and members are matched by their ids, no geometry matching).
"""
# The OCP stubs of the type-check environment miss most of its classes (dynamic attributes) – the rule is off for this file only:
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .expr import SketchError
from .model import Plane3, Resolved, Sketch, resolve

STEP_PREFIX = "sketch_"
SOURCE = "FEA Studio sketch"


def snapshot(sk: Sketch) -> dict:
    """The sketch as it is now, without the `built` record (stored in `Sketch.built["snapshot"]` after a build)."""
    d = sk.to_dict()
    d.pop("built", None)
    return d


def as_built(sk: Sketch) -> Sketch:
    """The sketch as it was at the last build (the geometry of the study belongs to it), or a copy of the current one."""
    snap = (sk.built or {}).get("snapshot")
    if isinstance(snap, dict):
        old = Sketch.from_dict(snap)
        if old is not None:
            return old
    return sk.copy()


def is_built(sk: Sketch) -> bool:
    """True when the sketch has not changed since its last build (the study geometry corresponds to it)."""
    return (sk.built or {}).get("snapshot") == snapshot(sk)


def _segkey(sid: str) -> tuple[int, int]:
    m = re.match(r"^[A-Za-z]*(\d+)(?:\.(\d+))?$", sid)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (10 ** 9, 0)


@dataclass
class RegionInfo:
    index: int
    name: str
    token: str                                   # stable identity of the region (the lowest curve id on its boundary)
    area: float
    entities: list[str]
    material: str | None
    solid: object
    sides: list[tuple[list[str], object]] = field(default_factory=list)      # ([curve ids], OCP side face)
    rings: list = field(default_factory=list)                               # boundary polygons (outer, holes) for the picture
    point: tuple[float, float] = (0.0, 0.0)                                 # a point inside the region (sketch coordinates)


@dataclass
class Constructed:
    """The geometry of a 2D sketch in memory (before it is written)."""
    sketch: Sketch
    res: Resolved
    plane: Plane3
    thickness: float
    regions: list[RegionInfo]
    rows: list[dict]
    warnings: list[str]

    @property
    def tokens(self) -> set[str]:
        return {r["token"] for r in self.rows}


@dataclass
class SketchBuild:
    """Result of building a sketch. 2D: STEP + sidecar + parts read back + selections; 1D: a FrameBuild."""
    sketch: Sketch
    kind: str
    step_path: Path | None = None
    parts: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    twod: Any = None                                      # TwoDSettings (2D)
    selections: dict = field(default_factory=dict)        # label → [FaceSelector] (2D)
    regions: list[dict] = field(default_factory=list)     # name, token, area_mm2, entities, material
    frame: Any = None                                     # FrameBuild (1D)
    warnings: list[str] = field(default_factory=list)
    size: float = 1.0
    cons: Constructed | None = None


# ------------------------------------------------------------------ 2D: construction

def _solid_of(shape):
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    if not ex.More():
        raise SketchError("The region could not be turned into a solid")
    return TopoDS.Solid(ex.Current())


def _volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return float(p.Mass())


def _check_axis(res: Resolved) -> None:
    """Axisymmetric: nothing may be on the negative side of the axis (x < 0)."""
    bad = []
    for s in res.segs:
        if s.construction:
            continue
        lo = min(p[0] for p in s.polyline(5.0))
        if lo < -res.tol:
            bad.append(s.entity)
    if bad:
        ids = list(dict.fromkeys(bad))
        raise SketchError(f"{', '.join(ids[:6])}: the contour crosses the axis (x < 0). In an axisymmetric sketch x is the radius – "
                          f"draw the section on the side x ≥ 0 only", ids)


def _region_names(sk: Sketch, regs, pl: Plane3, w: float) -> tuple[list[str], list[str | None]]:
    """Names and materials of the regions: by the seed points of `Sketch.regions`, otherwise "Region <n>" (by the order of the ids)."""
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    env = sk.values()
    names: list[str | None] = [None] * len(regs)
    mats: list[str | None] = [None] * len(regs)
    from .expr import evaluate
    for seed in sk.regions:
        try:
            x, y = evaluate(seed.point[0], env), evaluate(seed.point[1], env)
        except SketchError as e:
            raise SketchError(f"Region name '{seed.name}': the seed point – {e}") from None
        q = pl.pt(x, y, w)
        hit = None
        for i, r in enumerate(regs):
            st = BRepClass_FaceClassifier(r.face, gp_Pnt(*map(float, q)), 1e-7).State()
            if st in (TopAbs_IN, TopAbs_ON):
                hit = i
                break
        if hit is None:
            raise SketchError(f"The seed point of the region '{seed.name}' ({x:g}, {y:g}) is not inside any region")
        if seed.name:
            names[hit] = seed.name
        if seed.material:
            mats[hit] = seed.material
    used = {n for n in names if n}
    out = []
    for i, n in enumerate(names):
        if n:
            out.append(n)
            continue
        k = i + 1
        while f"Region {k}" in used:
            k += 1
        out.append(f"Region {k}")
        used.add(out[-1])
    return out, mats


def side_faces(op, edge, solid, pl: Plane3, how: str, amount: float, tol: float) -> list:
    """The face of the solid that the edge of the base face has swept (`Generated` of OCCT; for the edges that lie perpendicular to
    the axis of a full revolution it answers nothing, so the face is then the one that contains the swept middle point of the edge).
    `how` = "prism" (amount = the vector length along the normal) | "revolve" (amount = angle [rad]). Edges on the axis → []."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS
    from .regions import _faces
    gen = [TopoDS.Face(f) for f in op.Generated(edge)]
    if gen:
        return gen
    ad = BRepAdaptor_Curve(edge)
    pts = [ad.Value(ad.FirstParameter() + a * (ad.LastParameter() - ad.FirstParameter())) for a in (0.25, 0.5, 0.75)]
    loc = [pl.to_local([p.X(), p.Y(), p.Z()]) for p in pts]
    if how == "revolve" and all(abs(q[0]) <= tol for q in loc):
        return []
    m = pts[1]
    q = np.array([m.X(), m.Y(), m.Z()])
    if how == "prism":
        q = q + pl.n * (0.5 * amount)
    else:
        ang = 0.5 * min(amount, math.pi)
        d = q - pl.origin
        y = float(d @ pl.ey)
        rad = d - y * pl.ey
        q = pl.origin + y * pl.ey + math.cos(ang) * rad + math.sin(ang) * np.cross(pl.ey, rad)
    vtx = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, q))).Vertex()
    best, bd = None, math.inf
    for f in _faces(solid):
        dss = BRepExtrema_DistShapeShape(f, vtx)
        if dss.IsDone() and dss.Value() < bd:
            best, bd = f, dss.Value()
    return [best] if best is not None and bd <= max(tol * 10, 1e-7) else []


def construct(sk: Sketch) -> Constructed:
    """Regions, solids and the face table of a 2D sketch in memory."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Vec
    from ..geometry import face_props
    from .regions import find_regions
    if sk.kind != "twod":
        raise SketchError("Only a 2D sketch has a geometry (a beam sketch makes members)")
    res = resolve(sk)
    pl = sk.plane3()
    axisym = sk.mode == "axisym"
    t = 1.0 if axisym else sk.thickness_value(res.env)
    w = 0.0 if axisym else -t / 2.0
    if axisym:
        _check_axis(res)
    rs = find_regions(res, pl, w)
    regs = sorted(rs.regions, key=lambda r: min((_segkey(s) for _, ids in r.edges for s in ids), default=(10 ** 9, 0)))
    names, mats = _region_names(sk, regs, pl, w)
    infos: list[RegionInfo] = []
    rows: list[dict] = []
    seen: set[str] = set()
    warnings = list(rs.warnings)
    for i, r in enumerate(regs):
        if axisym:
            ax = gp_Ax1(gp_Pnt(*map(float, pl.origin)), gp_Dir(*map(float, pl.ey)))
            op = BRepPrimAPI_MakeRevol(r.face, ax, 2.0 * math.pi)
        else:
            op = BRepPrimAPI_MakePrism(r.face, gp_Vec(*map(float, pl.n * t)))
        op.Build()
        if not op.IsDone():
            raise SketchError(f"Region {i + 1}: the solid cannot be created (check the contour near {', '.join(r.entities[:4])})",
                              r.entities[:6])
        solid = _solid_of(op.Shape())
        if _volume(solid) < 0:
            solid = solid.Reversed()
        ids_all = [s for _, ids in r.edges for s in ids]
        token = min(ids_all, key=_segkey) if ids_all else f"R{i + 1}"
        while token in seen:
            token += "+"
        seen.add(token)
        info = RegionInfo(i, names[i], token, r.area, r.entities, mats[i], solid, rings=r.rings, point=r.point)
        for edge, owners in r.edges:
            faces = side_faces(op, edge, solid, pl, "revolve" if axisym else "prism", 2.0 * math.pi if axisym else t, res.tol)
            if not faces or not owners:
                continue
            for face in faces:
                pr = face_props(face)
                info.sides.append((list(owners), face))
                for sid in owners:
                    rows.append({"token": f"{token}/{sid}", "part": i, "type": pr["type"], "area": float(pr["area"]),
                                 "centroid": [float(v) for v in pr["centroid"]], "point": [float(v) for v in pr["point"]],
                                 "normal": [float(v) for v in pr["normal"]]})
        infos.append(info)
    return Constructed(sk, res, pl, t, infos, rows, warnings)


# ------------------------------------------------------------------ 2D: write, read back, selections

def selection_label(sk: Sketch, sid: str) -> list[str]:
    """Labels of the selection(s) a curve id belongs to. The id always works: "E3" → ["E3"]; "E5.2" → ["E5.2", "E5"]. An entity with a name
    gets the same selections under its name as well: "wall" → ["wall", "E6"]; "plate.2" → ["plate.2", "plate", "E5.2", "E5"]."""
    eid = sid.split(".")[0]
    ent = next((e for e in sk.entities if e.id == eid), None)
    bases = ([ent.name] if ent is not None and ent.name else []) + [eid]
    sub = sid.split(".", 1)[1] if "." in sid else None
    out: list[str] = []
    for base in bases:
        out += [f"{base}.{sub}", base] if sub else [base]
    return out


def _twod_settings(sk: Sketch, pl: Plane3, t: float):
    from ..twod.model import TwoDSettings
    if sk.mode == "axisym":
        return TwoDSettings(True, "axisym", [float(v) for v in pl.origin], [float(v) for v in pl.ey], [float(v) for v in pl.ex], 1.0)
    return TwoDSettings(True, sk.mode, [float(v) for v in pl.origin], [float(v) for v in pl.n], [float(v) for v in pl.ex], float(t))


def _next_stem(workdir: Path) -> str:
    n = 1
    for p in workdir.glob(f"{STEP_PREFIX}*.step"):
        m = re.match(rf"^{STEP_PREFIX}(\d+)$", p.stem)
        if m:
            n = max(n, int(m.group(1)) + 1)
    return f"{STEP_PREFIX}{n}"


def _meta(sk: Sketch, cons: Constructed, document: str) -> dict:
    env = cons.res.env
    return {"source": SOURCE, "document": document, "units": "mm",
            "params": {k: {"expression": str(v), "value": float(env[k]), "unit": "mm"} for k, v in sk.params.items()},
            "parts": [{"name": r.name, "token": f"R/{r.token}"} for r in cons.regions],
            "faces": [dict(row) for row in cons.rows]}


def build_geometry(sk: Sketch, workdir: str | Path, stem: str | None = None, document: str = "Sketch") -> SketchBuild:
    """Builds the sketch: 2D → STEP + sidecar in `workdir`, the parts read back from the file, the selections of the entities;
    1D → the nodes and members (`FrameBuild`). Raises SketchError (with `entities`) when the drawing is not valid."""
    if sk.kind == "frame":
        from .frame import build_frame
        fb = build_frame(sk)
        return SketchBuild(sk, "frame", frame=fb, warnings=list(fb.warnings), size=fb.size)
    from ..cad_link.facetable import bind_rows, face_rows
    from ..cad_link.faces import Geometry
    from ..geometry import Part, read_step, write_step
    from ..selectors import FaceSelector
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    cons = construct(sk)
    stem = stem or _next_stem(wd)
    step = wd / f"{stem}.step"
    write_step([Part(r.name, r.solid) for r in cons.regions], step)
    meta = _meta(sk, cons, document)
    (wd / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    warn: list[str] = []
    parts = read_step(step, report=warn)
    if len(parts) != len(cons.regions):
        raise SketchError(f"The written STEP has {len(parts)} bodies instead of {len(cons.regions)} regions")
    by_name = {p.name: i for i, p in enumerate(parts)}
    geo = Geometry(parts)
    rows = face_rows(meta)
    bound = bind_rows(rows, geo, lambda row: by_name.get(cons.regions[row["part"]].name) if "part" in row else None)
    selections: dict[str, list] = {}
    unbound = []
    for k, row in enumerate(rows):
        hit = bound.get(k)
        sid = row["token"].split("/", 1)[1]
        if hit is None:
            unbound.append(sid)
            continue
        pf = geo.part(hit[0])
        assert pf is not None
        sel = FaceSelector.from_props(parts[hit[0]].name, pf.faces[hit[1]])
        for label in selection_label(sk, sid):
            lst = selections.setdefault(label, [])
            if not any(abs(x.area - sel.area) < 1e-9 and x.part == sel.part and np.allclose(x.centroid, sel.centroid, atol=1e-9) for x in lst):
                lst.append(sel)
    warnings = list(cons.warnings) + [f"geometry check: {w}" for w in warn]
    if unbound:
        warnings.append("Side faces that could not be matched to the STEP (selections incomplete): " + ", ".join(sorted(set(unbound))[:8]))
    info = [{"name": r.name, "token": r.token, "area_mm2": r.area, "entities": r.entities, "material": r.material}
            for r in cons.regions]
    return SketchBuild(sk, "twod", step, parts, meta, _twod_settings(sk, cons.plane, cons.thickness), selections, info,
                       None, warnings, cons.res.size, cons)


# ------------------------------------------------------------------ the study

def _nice_size(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(v))
    return max(m * e for m in (1.0, 2.0, 2.5, 5.0) if m * e <= v * (1 + 1e-9))


def attach_first(st, b: SketchBuild, sk: Sketch) -> list[str]:
    """Writes the first build of a sketch into a study (new or existing): geometry, 2D settings, a selection per entity, the
    materials of the named regions – or the members of a beam sketch. Returns warnings."""
    from ..study import Selection
    st.sketch = sk
    if b.kind == "frame":
        from .frame import apply_frame
        warn = apply_frame(st, b.frame, sk)
        sk.built["snapshot"] = snapshot(sk)
        st.mesh.size = _nice_size(b.size / 30.0)
        return warn
    sk.built["snapshot"] = snapshot(sk)
    st.geometry = str(b.step_path)
    st.twod = b.twod
    st.mesh.size = _nice_size(b.size / 25.0)
    for label, faces in b.selections.items():
        st.selections[label] = Selection(label, list(faces))
    for r in (b.cons.regions if b.cons else []):
        if r.material and r.material in st.materials:
            st.part_materials[r.name] = r.material
    return []


def build_study(sk: Sketch, workdir: str | Path, name: str = "Sketch", material: str | None = None):
    """New study from a sketch → (Study, SketchBuild). 2D: geometry = the written STEP, 2D settings, a selection per entity;
    1D: a study without geometry with the members of the sketch."""
    from .. import materials as matlib
    from ..study import Study
    sk = sk.copy()
    b = build_geometry(sk, workdir, document=name)
    st = Study(name=name, geometry="")
    if material:
        try:
            st.add_material(matlib.by_name(material), default=True)
        except KeyError:
            raise SketchError(f"Material '{material}' is not in the catalogue (fea_materials_list)") from None
    else:
        st.add_material(matlib.default(), default=True)
    b.warnings += attach_first(st, b, sk)
    return st, b


def _resolved(new_rows: list[dict], old_tokens: set[str], new_cons: Constructed, old_cons: Constructed) -> tuple[list, list]:
    """The "resolved" part of the sidecar: every token of the previous build → the faces of the new build (empty = disappeared)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    by_tok: dict[str, list[dict]] = {}
    for r in new_rows:
        by_tok.setdefault(r["token"], []).append({k: r[k] for k in ("type", "area", "centroid", "point", "normal")})
    resolved = [{"token": tok, "faces": by_tok.get(tok, [])} for tok in sorted(old_tokens)]
    bodies = []
    new_by = {r.token: r for r in new_cons.regions}
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    for r in old_cons.regions:
        n = new_by.get(r.token)
        rows = []
        if n is not None:
            p = GProp_GProps()
            BRepGProp.VolumeProperties_s(n.solid, p)
            c = p.CentreOfMass()
            bb = Bnd_Box()
            BRepBndLib.Add_s(n.solid, bb)
            lo, hi = bb.CornerMin(), bb.CornerMax()
            rows.append({"volume": float(p.Mass()), "center": [c.X(), c.Y(), c.Z()],
                         "bbox": [lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()]})
        bodies.append({"token": f"R/{r.token}", "bodies": rows})
    return resolved, bodies


def _without_section_plane(notes: list[str]) -> list[str]:
    """The geometry remap warns that the 2D section plane stays unchanged; a sketch derives the plane from itself every time."""
    out = []
    for n in notes:
        head, sep, tail = n.partition("check: ")
        if sep:
            items = [x for x in tail.split(", ") if x != "2D section plane"]
            if not items:
                continue
            n = head + sep + ", ".join(items)
        out.append(n)
    return out


@dataclass
class RebuildPlan:
    """A proposal of the rebuild (nothing is changed yet): the new build and the remap report; `apply_rebuild` uses it."""
    study: Any
    sketch: Sketch
    build: SketchBuild
    report: Any = None                                # cad_link RemapReport (2D), None for a beam sketch
    new_meta: dict | None = None
    old_parts: list | None = None


def plan_rebuild(study, workdir: str | Path, sketch: Sketch | None = None, params: dict | None = None,
                 old_parts: list | None = None) -> RebuildPlan:
    """Builds the sketch of `study` again (after a change of `params`, or with a changed `sketch`) and proposes how the settings
    are carried over (cad_link.plan_update for 2D). The report lists what could not be matched; `rep.needs_review()` says whether
    the user should look at it. 1D: nothing to match – the members are found by their ids."""
    from ..cad_link.update import plan_update
    from ..geometry import read_step
    if study.sketch is None:
        raise SketchError("The study has no sketch")
    old_sk = as_built(study.sketch)
    new_sk = (sketch if sketch is not None else study.sketch).copy()
    for k, v in (params or {}).items():
        if k not in new_sk.params:
            raise SketchError(f"Unknown sketch parameter '{k}' (parameters: {', '.join(new_sk.params) or 'none'})")
        new_sk.set_param(k, v)
    if new_sk.kind == "frame":
        return RebuildPlan(study, new_sk, build_geometry(new_sk, workdir))
    b = build_geometry(new_sk, workdir, document=study.name)
    assert b.step_path is not None and b.cons is not None
    old_cons = construct(old_sk)
    old_meta: dict | None = {"parts": [{"name": r.name, "token": f"R/{r.token}"} for r in old_cons.regions], "faces": old_cons.rows}
    if old_parts is None:
        old_parts = read_step(study.geometry_path(None))
    # the old geometry must be the one of the old sketch (a manual "Replace geometry" in between makes the tokens useless)
    vol_old = sum(p.volume() for p in old_parts)
    vol_ctx = sum(_volume(r.solid) for r in old_cons.regions)
    if abs(vol_old - vol_ctx) > 1e-4 * max(vol_ctx, 1e-12):
        old_meta = None
        b.warnings.append("The geometry of the study does not match its sketch (replaced by hand?) – the selections are remapped "
                          "by their geometry only")
    new_meta = dict(b.meta)
    if old_meta is not None:
        new_meta["resolved"], new_meta["resolved_bodies"] = _resolved(b.cons.rows, old_cons.tokens, b.cons, old_cons)
    rep = plan_update(study, old_parts, b.parts, old_meta, new_meta)
    rep.notes = _without_section_plane(rep.notes)
    return RebuildPlan(study, new_sk, b, rep, new_meta, old_parts)


def apply_rebuild(plan: RebuildPlan, accept=None, reject=(), reject_anchors=()):
    """The new study of a plan (the old one is not changed). `accept` = the remap states that are used (default: certain, probable,
    split); `reject` / `reject_anchors` as in cad_link.update.apply_report (the choices of the remap dialog)."""
    from ..cad_link.update import DEFAULT_ACCEPT, apply_report
    from ..study import Selection
    from ..twod.model import TwoDSettings
    study, new_sk, b = plan.study, plan.sketch, plan.build
    if b.kind == "frame":
        return apply_frame_to_copy(study, b.frame, new_sk, b)
    new = apply_report(study, plan.report, accept if accept is not None else DEFAULT_ACCEPT, new_step=str(b.step_path),
                       new_parts=b.parts, new_meta=plan.new_meta, trigger="sketch", source=SOURCE, old_parts=plan.old_parts,
                       reject=reject, reject_anchors=reject_anchors)
    new_sk.built = {**new_sk.built, "snapshot": snapshot(new_sk)}
    new.sketch = new_sk
    old_t = study.twod
    t = b.twod
    assert isinstance(t, TwoDSettings)
    new.twod = TwoDSettings(True, t.kind, t.origin, t.axis, t.xdir, t.thickness, list(old_t.parts), old_t.size)
    for label, faces in b.selections.items():
        if label not in new.selections:
            new.selections[label] = Selection(label, list(faces))
    for r in (b.cons.regions if b.cons else []):
        if r.material and r.material in new.materials and r.name not in new.part_materials:
            new.part_materials[r.name] = r.material
    return new


def sketch_parameters(study) -> dict[str, dict]:
    """The named parameters of the sketch of a study – the inputs a parametric study can sweep (`rebuild_study(study, dir, params={...})`
    builds the geometry for one value of each): {name: {"expr": text or number, "value": mm}}; empty when the study has no sketch."""
    sk = study.sketch
    if sk is None:
        return {}
    vals = sk.values()
    return {k: {"expr": v, "value": vals[k]} for k, v in sk.params.items()}


def rebuild_study(study, workdir: str | Path, sketch: Sketch | None = None, params: dict | None = None, old_parts: list | None = None,
                  accept=None):
    """plan_rebuild + apply_rebuild: → (new study, RemapReport | None, SketchBuild). 2D goes through cad_link (the selections of the
    entities are found again by the token = the curve id, the rest by the geometry); 1D updates the members by their ids. The report
    says what could not be carried over; `rep.blocking()` lists the selections that lost their faces."""
    plan = plan_rebuild(study, workdir, sketch, params, old_parts)
    return apply_rebuild(plan, accept), plan.report, plan.build


def apply_frame_to_copy(study, fb, sk: Sketch, b: SketchBuild | None = None):
    """A copy of the study with the members of the sketch updated (1D rebuild)."""
    from ..study import Study
    from .frame import apply_frame
    new = Study.from_dict(study.to_dict())
    new.load_warnings = list(getattr(study, "load_warnings", []))
    warn = apply_frame(new, fb, sk)
    if b is not None:
        b.warnings += [w for w in warn if w not in b.warnings]
    sk.built["snapshot"] = snapshot(sk)
    new.sketch = sk
    return new


# ------------------------------------------------------------------ "Make 3D"

def extrude_to_3d(sk: Sketch, workdir: str | Path, how: str = "extrude", length=None, angle: float = 360.0,
                  name: str = "Sketch 3D", material: str | None = None):
    """The regions of a 2D sketch as a normal 3D study: extruded by `length` mm along the sketch normal, or revolved by `angle`° about
    the sketch y axis (the regions must lie at x ≥ 0). Selections: the side faces as in the sketch + "<entity>" caps are not named."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Vec
    from .. import materials as matlib
    from ..cad_link.facetable import bind_rows, face_rows
    from ..cad_link.faces import Geometry
    from ..geometry import Part, face_props, read_step, write_step
    from ..selectors import FaceSelector
    from ..study import Selection, Study
    from .expr import evaluate
    from .regions import find_regions
    if sk.kind != "twod":
        raise SketchError("Only a 2D sketch can be turned into 3D")
    if how not in ("extrude", "revolve"):
        raise SketchError("how must be 'extrude' or 'revolve'")
    res = resolve(sk)
    pl = sk.plane3()
    rs = find_regions(res, pl, 0.0)
    if how == "revolve":
        a = float(angle)
        if not 0.0 < a <= 360.0:
            raise SketchError("The angle must be in (0, 360] degrees")
        low = [s.entity for s in res.segs if not s.construction and min(p[0] for p in s.polyline(5.0)) < -res.tol]
        if low:
            raise SketchError(f"{', '.join(dict.fromkeys(low))}: crosses the axis of revolution (x < 0)", list(dict.fromkeys(low)))
    else:
        if length is None:
            raise SketchError("extrude needs a length [mm] (a number or an expression)")
        L = evaluate(length, res.env)
        if abs(L) <= res.tol:
            raise SketchError("The extrusion length must not be zero")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    regs = sorted(rs.regions, key=lambda r: min((_segkey(s) for _, ids in r.edges for s in ids), default=(10 ** 9, 0)))
    names, _ = _region_names(sk, regs, pl, 0.0)
    parts, rows = [], []
    for i, r in enumerate(regs):
        if how == "revolve":
            op = BRepPrimAPI_MakeRevol(r.face, gp_Ax1(gp_Pnt(*map(float, pl.origin)), gp_Dir(*map(float, pl.ey))), math.radians(a))
        else:
            op = BRepPrimAPI_MakePrism(r.face, gp_Vec(*map(float, pl.n * L)))
        op.Build()
        solid = _solid_of(op.Shape())
        if _volume(solid) < 0:
            solid = solid.Reversed()
        parts.append(Part(names[i], solid))
        for edge, owners in r.edges:
            for face in side_faces(op, edge, solid, pl, "revolve" if how == "revolve" else "prism", math.radians(a) if how == "revolve" else L, res.tol):
                pr = face_props(face)
                for sid in owners:
                    rows.append({"token": sid, "part": i, "type": pr["type"], "area": float(pr["area"]),
                                 "centroid": [float(v) for v in pr["centroid"]], "point": [float(v) for v in pr["point"]],
                                 "normal": [float(v) for v in pr["normal"]]})
    step = wd / f"{_next_stem(wd).replace(STEP_PREFIX, 'sketch3d_')}.step"
    write_step(parts, step)
    back = read_step(step)
    geo = Geometry(back)
    byname = {p.name: i for i, p in enumerate(back)}
    meta = {"faces": rows, "parts": [{"name": p.name} for p in parts]}
    frows = face_rows(meta)
    bound = bind_rows(frows, geo, lambda row: byname.get(parts[row["part"]].name) if "part" in row else None)
    st = Study(name=name, geometry=str(step))
    st.add_material(matlib.by_name(material) if material else matlib.default(), default=True)
    sels: dict[str, list] = {}
    for k, row in enumerate(frows):
        hit = bound.get(k)
        if hit is None:
            continue
        pf = geo.part(hit[0])
        assert pf is not None
        sel = FaceSelector.from_props(back[hit[0]].name, pf.faces[hit[1]])
        for label in selection_label(sk, rows[k]["token"]):
            lst = sels.setdefault(label, [])
            if not any(x.part == sel.part and abs(x.area - sel.area) < 1e-9 and np.allclose(x.centroid, sel.centroid, atol=1e-9) for x in lst):
                lst.append(sel)
    for label, faces in sels.items():
        st.selections[label] = Selection(label, faces)
    from ..geometry import bbox
    bb = np.array([bbox(p.shape) for p in back])
    diag = float(np.linalg.norm(bb[:, 3:].max(axis=0) - bb[:, :3].min(axis=0)))
    st.mesh.size = _nice_size(diag / 30.0)
    return st, back, step

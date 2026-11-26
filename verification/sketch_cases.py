"""Verification of the sketcher (fea_core.sketch, F15-S5): models drawn with typed coordinates and parameters, solved by the existing
2D (ccx CPS6 / CAX6) and beam (Timoshenko) paths, against analytic references and against the same problems built the old way.

  SK1  plate with a hole drawn in the sketcher (plane stress) = case A3 (V3 plate, Heywood Kt) and identical to the STEP-section route
       on the same mesh size
  SK2  thick axisymmetric tube from the sketcher = Lamé (case A1 reference), also the volume of the revolved body against the 2D model
  SK3  portal frame from the sketcher = case F3 (same members, Timoshenko solver) exactly + the slope–deflection method
  SK4  parameter change keeps all constraints / loads / materials / mesh settings and the results move as expected:
       hole diameter 10 → 15 mm (plate), frame height 4 → 5 m (portal frame)
  SK5  error cases: open contour, self-intersecting polyline, contour across the axis → clear error with the entity id
  SK6  DXF import (module sketch = ezdxf, only when it is installed): the plate with a hole and a rounded corner from a DXF drawing

Run: python -m verification.run_all SK1 SK2 SK3 SK4 SK5 [SK6]
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from fea_core.beams import sections as S
from fea_core.beams.model import FrameNodeLoad, SectionDef
from fea_core.sketch.build import build_study, rebuild_study
from fea_core.sketch.expr import SketchError
from fea_core.sketch.model import new_sketch
from fea_core.solve import run_study
from fea_core.study import Displacement, LoadCase, Pressure
from verification.cases import AtLeast, Case, Check, E, NU, V3_L, V3_R, V3_T, V3_W, v3_build

SKETCH_CASES: dict[str, Case] = {}


class _Out(dict):
    """The results of the runs of one case; `.warnings` for run_all."""

    def __init__(self, *a, warnings=None, **k):
        super().__init__(*a, **k)
        self._w = list(warnings or [])

    @property
    def warnings(self):
        return list(self._w)


def _plate_sketch(radius=V3_R):
    """The quarter plate of V3 / A3 (half length 150, half width 50, hole R = 5 at the origin, half thickness as the 2D thickness)."""
    sk = new_sketch("twod", "plane_stress", thickness=V3_T)
    for k, v in (("L", V3_L), ("W", V3_W), ("R", radius)):
        sk.set_param(k, v)
    sk.add_entity("line", [["R", 0], ["L", 0]], name="sym_y")
    sk.add_entity("line", [["L", 0], ["L", "W"]], name="end")
    sk.add_entity("line", [["L", "W"], [0, "W"]], name="top")
    sk.add_entity("line", [[0, "W"], [0, "R"]], name="sym_x")
    sk.add_entity("arc_center", [["R", 0], [0, "R"]], center=[0, 0], name="hole")
    return sk


def _plate_study(wd: Path, radius=V3_R):
    st, _ = build_study(_plate_sketch(radius), wd, "SK plate with a hole")
    st.mesh.size = 4.0
    st.mesh.local = [{"selection": "hole", "size": 0.35}]
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0)]
    st.load_cases = [LoadCase("Tah 100 MPa", [Pressure("end", -100.0)])]
    return st


def _kt_ref(d_over_w: float, w_full=100.0) -> float:
    """Heywood / Peterson Kt of a plate with a central hole (net section stress 100·W/(W − d))."""
    kt = 3 - 3.14 * d_over_w + 3.667 * d_over_w ** 2 - 1.527 * d_over_w ** 3
    return kt * 100.0 * w_full / (w_full * (1 - d_over_w))


# ------------------------------------------------------------------ SK1
def sk1_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return _plate_study(wd), []


def sk1_run(study, workdir: Path, parts):
    (workdir / "step_route").mkdir(parents=True, exist_ok=True)
    res = run_study(study, workdir / "sketch")
    st3, parts3 = v3_build(workdir / "step_route")                  # the same plate made the old way (STEP + section)
    from fea_core.twod.model import TwoDSettings
    st3.twod = TwoDSettings(True, "plane_stress", [0, 0, V3_T / 2], [0, 0, 1], [1, 0, 0], thickness=V3_T)
    st3.constraints = [c for c in st3.constraints if c.selection != "sym_z"]
    st3.mesh.size = 4.0
    st3.mesh.local = [{"selection": "otvor", "size": 0.35}]
    ref = run_study(st3, workdir / "step_route" / "run", parts=parts3)
    return _Out(sketch=res, step=ref, warnings=res.warnings)


def sk1_eval(out, st):
    r, o = out["sketch"].twod, out["step"].twod
    ref = _kt_ref(0.1)
    sx, sx_o = float(r.probe([0.0, V3_R], "sx")), float(o.probe([0.0, V3_R], "sx"))
    return [
        Check("SK1 σx max at the hole (Kt Heywood, d/W = 0.1)", sx, ref, 0.03, "MPa"),
        Check("SK1 σx max in the nodes = at the hole", float(r.step(0)["S"][:, 0].max()), ref, 0.03, "MPa"),
        Check("SK1 reaction Fx sym_x = 100 MPa·50·2.5", -float(r.step(0)["reactions"]["sym_x"][0]), 100 * V3_W * V3_T, 0.001, "N"),
        Check("SK1 σx at the hole = the STEP-section route (same mesh size)", sx, sx_o, 0.01, "MPa"),
        Check("SK1 max |u| = the STEP-section route", float(np.linalg.norm(r.step(0)["U"], axis=1).max()),
              float(np.linalg.norm(o.step(0)["U"], axis=1).max()), 0.005, "mm"),
        Check("SK1 volume of the 3D body / the 2D model", float(r.info["sections"][0]["ratio_3d_to_2d"]), 1.0, 1e-9),
    ]


SKETCH_CASES["SK1"] = Case("SK1", "Sketcher: plate with a hole drawn from typed coordinates (plane stress) = V3 / A3 and the STEP-section route",
                           sk1_build, sk1_eval, run=sk1_run)

# ------------------------------------------------------------------ SK2
RI, RO, H2, P2 = 20.0, 40.0, 10.0, 100.0


def sk2_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    sk = new_sketch("twod", "axisym")
    for k, v in (("ri", RI), ("ro", RO), ("H", H2)):
        sk.set_param(k, v)
    sk.add_entity("line", [["ri", 0], ["ro", 0]], name="bottom")
    sk.add_entity("line", [["ro", 0], ["ro", "H"]], name="outer")
    sk.add_entity("line", [["ro", "H"], ["ri", "H"]], name="top")
    sk.add_entity("line", [["ri", "H"], ["ri", 0]], name="inner")
    st, _ = build_study(sk, wd, "SK2 thick tube (axisymmetric)")
    st.mesh.size = 1.0
    st.constraints.append(Displacement("bottom", uy=0.0))
    st.load_cases = [LoadCase("p = 100 MPa", [Pressure("inner", P2)])]
    return st, []


def sk2_eval(res, st):
    r = res.twod
    a = P2 * RI ** 2 / (RO ** 2 - RI ** 2)
    sth_i, sth_o = a * (1 + RO ** 2 / RI ** 2), 2 * a
    u_i = RI / E * (sth_i + NU * P2)
    zc = H2 / 2
    return [
        Check("SK2 σθ at ri (Lamé)", float(r.probe([RI, zc], "st")), sth_i, 0.002, "MPa"),
        Check("SK2 σθ at ro (Lamé)", float(r.probe([RO, zc], "st")), sth_o, 0.002, "MPa"),
        Check("SK2 σr at ri = −p", -float(r.probe([RI, zc], "sr")), P2, 0.003, "MPa"),
        Check("SK2 u_r(ri)", float(r.probe([RI, zc], "U")[0]), u_i, 0.002, "mm"),
        Check("SK2 volume of the revolved body / 2π·r_c·A of the 2D model", float(r.info["sections"][0]["ratio_3d_to_2d"]), 1.0, 1e-9),
        Check("SK2 volume of the revolved body = π(ro² − ri²)H", float(r.info["sections"][0]["volume_3d"]),
              math.pi * (RO ** 2 - RI ** 2) * H2, 1e-9, "mm³"),
    ]


SKETCH_CASES["SK2"] = Case("SK2", "Sketcher: thick-walled tube drawn in the sketcher (axisymmetric, revolved) = Lamé (A1)", sk2_build, sk2_eval)

# ------------------------------------------------------------------ SK3
H3, L3, h3 = 10000.0, 6000.0, 4000.0


def _rigid_axial(name: str, lib: str) -> SectionDef:
    d = S.get(lib).to_dict()
    d["A"] *= 1e6                      # the slope–deflection method assumes axially rigid members (as case F3)
    return SectionDef(name, props=d)


def _portal(h=h3, wd: Path | None = None):
    sk = new_sketch("frame")
    sk.set_param("L", L3)
    sk.set_param("h", h)
    sk.add_entity("line", [[0, 0], [0, "h"]], section="IPE 300")                  # E1 = column A–B (as F3 'sloup1')
    sk.add_entity("line", [[0, "h"], ["L", "h"]], section="IPE 400")              # E2 = crossbar (F3 'pricel')
    sk.add_entity("line", [["L", 0], ["L", "h"]], section="IPE 300")              # E3 = column D–C (F3 'sloup2')
    sk.add_support([0, 0], "fixed")
    sk.add_support(["L", 0], "fixed")
    st, _ = build_study(sk, wd or Path("."), "SK3 portal frame")
    st.frame.settings.shear_deformation = False
    st.frame.sections["IPE 300"] = _rigid_axial("IPE 300", "IPE 300")
    st.frame.sections["IPE 400"] = _rigid_axial("IPE 400", "IPE 400")
    st.load_cases = [LoadCase("H", [FrameNodeLoad("P2", [H3, 0, 0])])]
    return st


def sk3_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return _portal(wd=wd), []


def sk3_run(study, workdir: Path, parts):
    from verification.frame_cases import f3_build
    from fea_core.beams.run import run_frame_tim
    res = run_study(study, workdir / "sketch")
    st3, _ = f3_build(workdir / "f3")
    (workdir / "f3").mkdir(parents=True, exist_ok=True)
    ref = run_frame_tim(st3, workdir / "f3" / "run")
    return _Out(sketch=res, f3=ref, warnings=res.warnings)


def _slope_deflection(h: float):
    Ic, Ib = S.get("IPE 300").Iy, S.get("IPE 400").Iy
    k = Ib * h / (Ic * L3)
    return H3 * h / 2 * (3 * k + 1) / (6 * k + 1), H3 * h / 2 * 3 * k / (6 * k + 1)


def sk3_eval(out, st):
    a, b = out["sketch"].step(0).frame, out["f3"].step(0).frame
    mb, mt = _slope_deflection(h3)
    pairs = (("E1", "sloup1"), ("E2", "pricel"), ("E3", "sloup2"))
    worst = max(float(np.max(np.abs(np.asarray(a["members"][x]["My"]) - np.asarray(b["members"][y]["My"]))))
                / max(float(np.max(np.abs(np.asarray(b["members"][y]["My"])))), 1.0) for x, y in pairs)
    ra = np.asarray(a["reactions"]["P1"])
    rb = np.asarray(b["reactions"]["A"])
    return [
        Check("SK3 My along all members = case F3 (max. relative difference)", worst, 0.0, 1e-9),
        Check("SK3 reaction of the foot A = case F3", float(np.linalg.norm(ra - rb)) / float(np.linalg.norm(rb)), 0.0, 1e-9),
        Check("SK3 displacement of the corner B = case F3", float(np.linalg.norm(np.asarray(a["nodes"]["P2"]) - np.asarray(b["nodes"]["B"]))
                                                                  / np.linalg.norm(np.asarray(b["nodes"]["B"]))), 0.0, 1e-9),
        Check("SK3 moment at the foot (slope–deflection)", abs(a["members"]["E1"]["My"][0]), mb, 1e-5, "N·mm"),
        Check("SK3 moment at the corner (slope–deflection)", abs(a["members"]["E1"]["My"][-1]), mt, 1e-5, "N·mm"),
        Check("SK3 sum of the horizontal reactions = H", -(a["reactions"]["P1"][0] + a["reactions"]["P4"][0]), H3, 1e-7, "N"),
    ]


SKETCH_CASES["SK3"] = Case("SK3", "Sketcher: portal frame drawn from typed coordinates = case F3 (Timoshenko solver) exactly + slope–deflection",
                           sk3_build, sk3_eval, run=sk3_run)


# ------------------------------------------------------------------ SK4
def sk4_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return _plate_study(wd), []


def sk4_run(study, workdir: Path, parts):
    base = run_study(study, workdir / "base")
    new, rep, b = rebuild_study(study, workdir, params={"R": 7.5})        # hole Ø 10 → 15 mm
    big = run_study(new, workdir / "hole15")
    # the frame: height 4 m → 5 m
    pst = _portal(wd=workdir / "frame")
    f4 = run_study(pst, workdir / "frame" / "h4")
    pnew, _, _ = rebuild_study(pst, workdir / "frame", params={"h": 5000})
    f5 = run_study(pnew, workdir / "frame" / "h5")
    return _Out(base=base, big=big, new=new, old=study, rep=rep, f4=f4, f5=f5, pst=pst, pnew=pnew, warnings=big.warnings + f5.warnings)


def sk4_eval(out, st):
    old, new, rep = out["old"], out["new"], out["rep"]
    r0, r1 = out["base"].twod, out["big"].twod
    sx0, sx1 = float(r0.probe([0.0, V3_R], "sx")), float(r1.probe([0.0, 7.5], "sx"))
    ref1 = _kt_ref(0.15)
    kept = (len(new.constraints) == len(old.constraints) and [c.selection for c in new.constraints] == [c.selection for c in old.constraints]
            and [l.selection for lc in new.load_cases for l in lc.loads] == [l.selection for lc in old.load_cases for l in lc.loads]
            and new.mesh.local == old.mesh.local and new.default_material == old.default_material and new.twod.thickness == old.twod.thickness)
    mb4, mt4 = _slope_deflection(4000.0)
    mb5, mt5 = _slope_deflection(5000.0)
    m4 = out["f4"].step(0).frame["members"]["E1"]["My"]
    m5 = out["f5"].step(0).frame["members"]["E1"]["My"]
    pn, po = out["pnew"], out["pst"]
    return [
        Check("SK4 constraints, loads, mesh settings and material kept after the rebuild (1 = yes)", float(kept), 1.0, 0.0),
        Check("SK4 selections not remapped (the report)", float(len(rep.blocking())), 0.0, 0.0),
        Check("SK4 the hole selection followed its entity (by id, status cad = 1)", float(next(s.status for s in rep.selections if s.name == "hole") == "cad"), 1.0, 0.0),
        Check("SK4 hole Ø 15: σx max (Kt Heywood, d/W = 0.15)", sx1, ref1, 0.04, "MPa"),
        AtLeast("SK4 σx max grows with the hole (Ø 15 vs Ø 10)", sx1 / sx0, 1.0, 0.0),
        Check("SK4 reaction Fx unchanged (the load is a pressure on the end)", -float(r1.step(0)["reactions"]["sym_x"][0]),
              -float(r0.step(0)["reactions"]["sym_x"][0]), 0.001, "N"),
        Check("SK4 frame height 4 m: moment at the foot (slope–deflection)", abs(m4[0]), mb4, 1e-5, "N·mm"),
        Check("SK4 frame height 5 m: moment at the foot (slope–deflection)", abs(m5[0]), mb5, 1e-5, "N·mm"),
        Check("SK4 frame height 5 m: moment at the corner (slope–deflection)", abs(m5[-1]), mt5, 1e-5, "N·mm"),
        Check("SK4 frame: loads, supports and sections kept (1 = yes)",
              float(len(pn.load_cases[0].loads) == 1 and len(pn.frame.supports) == len(po.frame.supports)
                    and pn.frame.members["E1"].section == "IPE 300" and pn.frame.nodes["P2"].xyz[2] == 5000.0), 1.0, 0.0),
        Check("SK4 frame: the moment at the corner changed with the height (not the old value)", abs(m5[-1]) / abs(m4[-1]), mt5 / mt4, 1e-4),
        Check("SK4 frame: the old corner moment reference of 4 m was different (sanity)", float(abs(mt5 - mt4) / mt4 > 0.01), 1.0, 0.0),
    ]


SKETCH_CASES["SK4"] = Case("SK4", "Sketcher: a parameter change (hole Ø 10 → 15 mm, frame height 4 → 5 m) keeps constraints / loads / materials",
                           sk4_build, sk4_eval, run=sk4_run)


# ------------------------------------------------------------------ SK5
def sk5_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return _plate_study(wd), []


def _error(sk) -> SketchError | None:
    from fea_core.sketch.build import build_geometry
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            build_geometry(sk, td)
    except SketchError as e:
        return e
    return None


def sk5_run(study, workdir: Path, parts):
    out: dict = {}
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    gap = new_sketch("twod")
    for a, b in zip(sq, sq[1:] + [[0, 0.001]]):
        gap.add_entity("line", [a, b])
    out["gap"] = _error(gap)
    bow = new_sketch("twod")
    bow.add_entity("polyline", [[0, 0], [10, 10], [10, 0], [0, 10]], closed=True)
    out["bow"] = _error(bow)
    axis = new_sketch("twod", "axisym")
    axis.add_entity("rect", [[-10, 0], [10, 10]])
    out["axis"] = _error(axis)
    free = new_sketch("twod")
    free.add_entity("rect", [[0, 0], [10, 10]])
    free.add_entity("line", [[5, 5], [8, 20]])
    out["free"] = _error(free)
    zero = new_sketch("twod")
    zero.add_entity("rect", [[0, 0], [10, 10]])
    zero.add_entity("rect", [[0, 0], [10, 10]])
    out["dup"] = _error(zero)
    ok = new_sketch("twod")
    ok.add_entity("rect", [[0, 0], [10, 10]])
    out["ok"] = _error(ok)
    return _Out(out)


def sk5_eval(out, st):
    def names(e, *ids):
        return float(e is not None and set(e.entities) >= set(ids))
    return [
        Check("SK5 open contour: error names both ends of the gap (E1, E4)", names(out["gap"], "E1", "E4"), 1.0, 0.0),
        Check("SK5 open contour: the gap size is in the text", float(out["gap"] is not None and "gap 0.001" in str(out["gap"])), 1.0, 0.0),
        Check("SK5 self-intersecting polyline: error names E1", names(out["bow"], "E1"), 1.0, 0.0),
        Check("SK5 contour across the axis (axisymmetric): error names E1", names(out["axis"], "E1"), 1.0, 0.0),
        Check("SK5 a free line crossing the plate: error names E2", names(out["free"], "E2"), 1.0, 0.0),
        Check("SK5 two identical rectangles: no crash (overlapping curves merge into one region)", float(out["dup"] is None), 1.0, 0.0),
        Check("SK5 a valid sketch gives no error", float(out["ok"] is None), 1.0, 0.0),
    ]


SKETCH_CASES["SK5"] = Case("SK5", "Sketcher: error cases (open contour, self-intersection, across the axis) name the entity",
                           sk5_build, sk5_eval, run=sk5_run)


# ------------------------------------------------------------------ SK6 (needs ezdxf)
def sk6_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return _plate_study(wd), []


def sk6_run(study, workdir: Path, parts):
    import ezdxf
    workdir.mkdir(parents=True, exist_ok=True)
    from fea_core.sketch.build import build_geometry
    from fea_core.sketch.dxf import import_dxf
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((100, 0), (100, 60))
    msp.add_line((100, 60), (20, 60))
    msp.add_arc((20, 40), 20, 90, 180)
    msp.add_line((0, 40), (0, 0))
    msp.add_circle((60, 30), 8)
    f = workdir / "plate.dxf"
    doc.saveas(f)
    sk = new_sketch("twod", thickness=3)
    info = import_dxf(sk, f)
    b = build_geometry(sk, workdir / "dxf")
    return _Out(info=info, build=b, warnings=info["warnings"])


def sk6_eval(out, st):
    area = 100 * 60 - (20 * 20 - math.pi * 400 / 4) - math.pi * 64
    b = out["build"]
    return [
        Check("SK6 DXF: area of the region (rectangle − rounded corner − hole)", b.regions[0]["area_mm2"], area, 1e-9, "mm²"),
        Check("SK6 DXF: volume of the extruded part", b.parts[0].volume(), area * 3.0, 1e-9, "mm³"),
        Check("SK6 DXF: curves imported (4 lines + 1 arc + 1 circle)", float(out["info"]["imported"]), 6.0, 0.0),
    ]


if importlib.util.find_spec("ezdxf") is not None:
    SKETCH_CASES["SK6"] = Case("SK6", "Sketcher: plate with a rounded corner and a hole imported from a DXF drawing (ezdxf)", sk6_build, sk6_eval,
                               run=sk6_run)

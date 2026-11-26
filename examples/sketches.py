"""Example 11 – models drawn in the sketcher (no CAD):

  example11_sketch_plate.fea  – a bracket plate (plane stress): outline of lines and arcs, an eye and two bolt holes drawn from
                                typed coordinates and the parameters L, H, r, d_eye; fixed along the wall edge, 3 kN on the eye
  example11_sketch_frame.fea  – a gable frame of members (HEA 240 columns, IPE 300 rafters) drawn from the parameters L, h, rise;
                                snow and wind, pinned feet

Change a parameter (Model step → Edit sketch…, table Parameters) and apply: constraints, loads and materials stay.
Run:  python -m examples.sketches [--solve] [--out FOLDER]
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from fea_core.beams.model import FrameMemberLoad
from fea_core.sketch.build import build_study
from fea_core.sketch.model import new_sketch
from fea_core.study import Fixed, Force, LoadCase

HERE = Path(__file__).resolve().parent


def plate_sketch():
    sk = new_sketch("twod", "plane_stress", "XY", 8)
    for k, v in (("L", 160), ("H", 70), ("r", 10), ("d_eye", 16), ("d_bolt", 9)):
        sk.set_param(k, v)
    sk.add_entity("line", [[0, 0], ["L-r", 0]])
    sk.add_entity("arc_center", [["L-r", 0], ["L", "r"]], center=["L-r", "r"])
    sk.add_entity("line", [["L", "r"], ["L", "H-r"]])
    sk.add_entity("arc_center", [["L", "H-r"], ["L-r", "H"]], center=["L-r", "H-r"])
    sk.add_entity("line", [["L-r", "H"], [0, "H"]])
    sk.add_entity("line", [[0, "H"], [0, 0]], name="wall")
    sk.add_entity("circle", center=["L-30", "H/2"], radius="d_eye/2", name="eye")
    sk.add_entity("circle", center=[20, "H/4"], radius="d_bolt/2", name="bolt_1")
    sk.add_entity("circle", center=[20, "3*H/4"], radius="d_bolt/2", name="bolt_2")
    return sk


def build_plate(workdir: Path):
    st, _ = build_study(plate_sketch(), workdir, "Example 11 – bracket plate drawn in the sketcher", material="Ocel S235")
    st.constraints.append(Fixed("wall"))
    st.load_cases.append(LoadCase("Load on the eye 3 kN", [Force("eye", [0.0, -3000.0, 0.0])]))
    st.mesh.size = 3.0
    st.mesh.local = [{"selection": "eye", "size": 1.0}]
    return st


def frame_sketch():
    sk = new_sketch("frame")
    for k, v in (("L", 12000), ("h", 4000), ("rise", 1500)):
        sk.set_param(k, v)
    sk.add_entity("line", [[0, 0], [0, "h"]], section="HEA 240")
    sk.add_entity("line", [[0, "h"], ["L/2", "h+rise"]], section="IPE 300")
    sk.add_entity("line", [["L/2", "h+rise"], ["L", "h"]], section="IPE 300")
    sk.add_entity("line", [["L", "h"], ["L", 0]], section="HEA 240")
    sk.add_support([0, 0], "pinned")
    sk.add_support(["L", 0], "pinned")
    return sk


def build_frame(workdir: Path):
    st, _ = build_study(frame_sketch(), workdir, "Example 11 – gable frame drawn in the sketcher", material="Ocel S235")
    snow, wind = -0.8e-3 * 6000.0, 0.4e-3 * 6000.0                    # N/mm: 0.8 kN/m² snow and 0.4 kN/m² wind on the frame spacing 6 m
    st.load_cases.append(LoadCase("Snow", [FrameMemberLoad("E2", [0, 0, snow]), FrameMemberLoad("E3", [0, 0, snow])]))
    st.load_cases.append(LoadCase("Wind from the left", [FrameMemberLoad("E1", [wind, 0, 0])]))
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solve", action="store_true", help="solve and save with results")
    ap.add_argument("--out", default=str(HERE), help="output folder (default examples/)")
    a = ap.parse_args(argv)
    from fea_core.project import save_project
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fea_example11_") as tmp:
        for name, fn in (("example11_sketch_plate", build_plate), ("example11_sketch_frame", build_frame)):
            wd = Path(tmp) / name
            st = fn(wd)
            res = None
            if a.solve:
                from fea_core.solve import run_study
                res = run_study(st, wd / "run")
            step = st.geometry if st.geometry else ""
            save_project(out / f"{name}.fea", st, step, res)
            print(f"{name}.fea saved" + (" with results" if res is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

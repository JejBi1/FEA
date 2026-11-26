"""Reference small-satellite benchmark (F14 stage S6): a clean, self-contained CubeSat-class metallic bolted assembly built from code (no CAD file),
that exercises the space chain S1 (margins) -> S2 (launch loads) -> S3 (model checks) -> S5 (test correlation) and writes the Structural Analysis Report (ECSS) of
`fea_core.report.space_sar`. Stage S4 (interfaces, notching, bolts, PCB) is NOT part of this benchmark: the report therefore says "interfaces/bolts not assessed in this build"
(a build without the S4 modules) or "not run" (S4 present, nothing computed) in chapter 7 and is not a sign-off - which is also the true state of the benchmark.

    python -m verification.cubesat_benchmark --out <folder> [--build-only] [--no-report] [--no-crosscheck] [--convergence] [--threads N]

The model (mm, z = launch axis, z = 0 is the interface to the dispenser):
  * "Frame"   - an aluminium base plate (z 0..6), a top ring (z 104..110) and four corner rails 8 x 8 x 110 (one body);
  * "Panel -X/+X/-Y/+Y" - four 2.5 mm side panels, 0.2 mm off the rails, so that they are held ONLY by the bolts (the joint carries no friction: conservative);
  * "Bolt 1..16" - sixteen M3 bolts (shank 3 mm in a 3.4 mm clearance hole of the panel, tapped hole of the rail, head on the panel); the joints are bonded in the
    mesh (the shank to the rail, the head to the panel) - a linear model of the joint WITHOUT preload;
  * "PCB" - a 80 x 80 x 1.6 mm orthotropic board on four brass standoffs, with a surface mass of the population and a rigidly coupled point mass (the payload).
Every number of the requirement set, of the environments and of the allowables below is an ILLUSTRATIVE TEST VALUE chosen so that the chain can be exercised: none is
a value of ECSS or of a launcher, and none may be copied into a project. Nothing here claims that the model complies with a CubeSat standard.
The mesh is deliberately coarse (about 35 000 elements, 180 000 equations; the whole chain takes 6 to 7 minutes on a computer that is busy with other work, of which the
meshing, the model checks and the comparison of the two modal engines are the larger parts): the mesh-convergence model check is therefore left out and the Structural Analysis
Report correctly says that the checks are incomplete (no sign-off) - `--convergence` runs it (several more minutes). Findings that are real properties of a coarse model are
left in the report, not tuned away: the modal basis of 14 modes covers a small part of the effective mass (many local panel modes), so the frequency budget stays UNCONFIRMED, and
the ~70 sharp inner edges are singular places that nobody has justified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from fea_core.dynamics.model import PointMass, SurfaceMass
from fea_core.geometry import Part, read_step, write_step, _solids
from fea_core.selectors import find_faces
from fea_core.study import Fixed, LoadCase, Material, MeshSettings, Study

NAME = "CubeSat-class benchmark (illustrative test values)"
ILLUSTRATIVE = ("ILLUSTRATIVE TEST VALUES: every factor, limit, load level and allowable of this benchmark was chosen to exercise the program; "
                "none is a value of ECSS, of a launcher or of a material data sheet.")

# ------------------------------------------------------------------ geometry (mm)
SIDE, HEIGHT, RAIL = 100.0, 110.0, 8.0
PLATE_T, RING_T = 6.0, 6.0
PANEL_T, GAP = 2.5, 0.2
HOLE_D, SHANK_D, HEAD_D, HEAD_H, THREAD = 3.4, 3.0, 5.5, 2.0, 6.0
BOLT_POS = [(4.0, 20.0), (4.0, 90.0), (96.0, 20.0), (96.0, 90.0)]          # (along the panel, z): over the rails at both panel edges
PCB_Z, PCB_T, PCB_XY = 26.0, 1.6, (10.0, 90.0)
POST_XY = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)]
PAD = (40.0, 60.0, 4.0)                                                    # payload mount on the board: x/y range and height
PAYLOAD_KG, PCB_POPULATION_KG_M2 = 0.150, 1.0


def box(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float) -> Any:
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()


def cyl(p: tuple[float, float, float], d: tuple[float, float, float], r: float, h: float) -> Any:
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*d)), r, h).Shape()


def fuse(*shapes: Any) -> Any:
    out = shapes[0]
    for s in shapes[1:]:
        out = BRepAlgoAPI_Fuse(out, s).Shape()
    return out


def cut(shape: Any, *tools: Any) -> Any:
    for t in tools:
        shape = BRepAlgoAPI_Cut(shape, t).Shape()
    return shape


def solid(shape: Any) -> Any:
    s = _solids(shape)
    if len(s) != 1:
        raise ValueError(f"one solid expected, got {len(s)}")
    return s[0]


# (panel name, outward unit direction, the rail face point as a function of (u, z), the panel box)
def _faces() -> list[tuple[str, tuple[float, float, float], Any, tuple[float, ...]]]:
    t = GAP + PANEL_T
    return [("Panel -X", (-1.0, 0.0, 0.0), lambda u, z: (0.0, u, z), (-t, 0.0, PLATE_T, PANEL_T, SIDE, HEIGHT - PLATE_T - RING_T)),
            ("Panel +X", (1.0, 0.0, 0.0), lambda u, z: (SIDE, u, z), (SIDE + GAP, 0.0, PLATE_T, PANEL_T, SIDE, HEIGHT - PLATE_T - RING_T)),
            ("Panel -Y", (0.0, -1.0, 0.0), lambda u, z: (u, 0.0, z), (0.0, -t, PLATE_T, SIDE, PANEL_T, HEIGHT - PLATE_T - RING_T)),
            ("Panel +Y", (0.0, 1.0, 0.0), lambda u, z: (u, SIDE, z), (0.0, SIDE + GAP, PLATE_T, SIDE, PANEL_T, HEIGHT - PLATE_T - RING_T))]


def _add(p: tuple[float, float, float], d: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (p[0] + d[0] * s, p[1] + d[1] * s, p[2] + d[2] * s)


def geometry() -> list[Part]:
    """The parts in the global position (the order is the order in the STEP file)."""
    frame = fuse(box(0, 0, 0, SIDE, SIDE, PLATE_T),
                 cut(box(0, 0, HEIGHT - RING_T, SIDE, SIDE, RING_T), box(RAIL, RAIL, HEIGHT - RING_T - 1, SIDE - 2 * RAIL, SIDE - 2 * RAIL, RING_T + 2)),
                 *[box(x, y, 0, RAIL, RAIL, HEIGHT) for x in (0.0, SIDE - RAIL) for y in (0.0, SIDE - RAIL)])
    panels, bolts, taps = [], [], []
    for name, d, face, pbox in _faces():
        inward = (-d[0], -d[1], -d[2])
        holes = []
        for u, z in BOLT_POS:
            f = face(u, z)
            taps.append(cyl(f, inward, SHANK_D / 2, THREAD))                                        # the tapped hole of the rail (filled by the shank)
            holes.append(cyl(_add(f, d, GAP + PANEL_T + 1.0), inward, HOLE_D / 2, PANEL_T + 2.0))   # the clearance hole of the panel
            bolts.append(fuse(cyl(_add(f, inward, THREAD), d, SHANK_D / 2, THREAD + GAP + PANEL_T),
                              cyl(_add(f, d, GAP + PANEL_T), d, HEAD_D / 2, HEAD_H)))
        panels.append((name, cut(box(*pbox), *holes)))
    frame = cut(frame, *taps)
    x0, x1 = PCB_XY
    parts = [Part("Frame", solid(frame), color=(0.82, 0.86, 0.90))]
    parts += [Part(n, solid(s), color=(0.7, 0.75, 0.8)) for n, s in panels]
    parts += [Part(f"Bolt {k}", solid(b), color=(0.55, 0.57, 0.6)) for k, b in enumerate(bolts, 1)]
    parts += [Part(f"Standoff {k}", solid(cyl((x, y, PLATE_T), (0, 0, 1), 2.5, PCB_Z - PLATE_T)), color=(0.85, 0.72, 0.35)) for k, (x, y) in enumerate(POST_XY, 1)]
    parts.append(Part("PCB", solid(box(x0, x0, PCB_Z, x1 - x0, x1 - x0, PCB_T)), color=(0.2, 0.5, 0.3)))
    parts.append(Part("Payload mount", solid(box(PAD[0], PAD[0], PCB_Z + PCB_T, PAD[1] - PAD[0], PAD[1] - PAD[0], PAD[2])), color=(0.25, 0.25, 0.28)))
    return parts


# ------------------------------------------------------------------ the study
def materials() -> dict[str, Material]:
    al = Material("Aluminium (illustrative)", 70000.0, 0.33, 2.7e-9, Re=240.0, Rm=290.0, alpha=2.3e-5, source="illustrative test value")
    steel = Material("Stainless steel (illustrative)", 193000.0, 0.30, 7.9e-9, Re=210.0, Rm=500.0, alpha=1.6e-5, source="illustrative test value")
    brass = Material("Brass (illustrative)", 100000.0, 0.34, 8.4e-9, Re=200.0, Rm=380.0, alpha=1.9e-5, source="illustrative test value")
    ortho = {"E1": 22000.0, "E2": 22000.0, "E3": 9000.0, "nu12": 0.15, "nu13": 0.25, "nu23": 0.25, "G12": 8000.0, "G13": 3500.0, "G23": 3500.0}
    pcb = Material("PCB FR-4 (illustrative orthotropic)", 22000.0, 0.15, 1.9e-9, Rm=300.0, ortho=ortho, source="illustrative test value")
    return {m.name: m for m in (al, steel, brass, pcb)}


def build_study(step: Path, *, mesh_size: float = 10.0, engine: str = "auto") -> tuple[Study, list[Part]]:
    """Writes the STEP file and returns the study (without the space settings: `add_space` adds them) and the parts read back from it."""
    write_step(geometry(), step)
    parts = read_step(step)
    mats = materials()
    st = Study(NAME, str(step))
    st.add_material(mats["Aluminium (illustrative)"], default=True)
    for m in list(mats.values())[1:]:
        st.add_material(m)
    for p in parts:
        if p.name.startswith("Bolt"):
            st.part_materials[p.name] = "Stainless steel (illustrative)"
        elif p.name.startswith("Standoff"):
            st.part_materials[p.name] = "Brass (illustrative)"
        elif p.name == "PCB":
            st.part_materials[p.name] = "PCB FR-4 (illustrative orthotropic)"
    st.add_selection("Interface", find_faces(parts, part="Frame", z=0.0))
    st.add_selection("PCB population", find_faces(parts, part="PCB", z=PCB_Z + PCB_T, where=lambda f: f.get("area", 0.0) > 100.0))
    st.add_selection("Payload attachment", find_faces(parts, part="Payload mount", z=PCB_Z + PCB_T + PAD[2]))
    st.constraints.append(Fixed("Interface"))
    st.masses += [SurfaceMass("PCB population", PCB_POPULATION_KG_M2, name="PCB population (illustrative)"),
                  PointMass("Payload attachment", PAYLOAD_KG * 1e-3, point=[0.5 * (PAD[0] + PAD[1]), 0.5 * (PAD[0] + PAD[1]), PCB_Z + PCB_T + PAD[2] + 8.0],
                            inertia=[PAYLOAD_KG * 1e-3 * 20.0 ** 2 / 6.0] * 3, coupling="distributing", name="Payload 150 g (illustrative)")]
    sizes = {p.name: 2.6 for p in parts if p.name.startswith("Bolt")} | {"Frame": 10.0}
    sizes.update({p.name: 6.0 for p in parts if p.name in ("PCB", "Payload mount")} | {p.name: 12.0 for p in parts if p.name.startswith(("Panel", "Standoff"))})
    st.mesh = MeshSettings(size=mesh_size, part_sizes=sizes, curvature_points=10)
    st.solver.backend, st.solver.modal_engine = "auto", engine
    st.modal.enabled, st.modal.modes = True, 12
    st.dynamics.enabled, st.dynamics.modes = True, 14
    st.dynamics.damping.kind, st.dynamics.damping.zeta = "modal", 0.03
    st.load_cases.append(LoadCase("Dead weight", []))
    return st, parts


# ------------------------------------------------------------------ the space settings: ILLUSTRATIVE test values
BY, DATE = "illustrative test engineer (no person)", "2026-01-01"
DOC = "illustrative test document - not ECSS, not a launcher guide"
ILLUSTRATIVE_LIMITS = {                 # name -> (value, unit); the numbers are chosen to exercise the chain, none comes from a standard
    "qsl_axial_g": (12.0, "g"), "qsl_lateral_g": (8.0, "g"), "mass_max_kg": (3.0, "kg"), "f_min_axial_hz": (150.0, "Hz"), "f_min_lateral_hz": (100.0, "Hz"),
    "mass_tol_rel": (0.02, "-"), "cog_tol_mm": (2.0, "mm"), "cad_mass_tol_rel": (0.2, "-"), "part_mass_tol_rel": (0.02, "-"), "solver_mass_tol_rel": (0.01, "-"),
    "rigid_energy_ratio_max": (1e-6, "-"), "rigid_fit_tol": (1e-3, "-"), "unit_disp_energy_ratio_max": (1e-6, "-"), "unit_disp_reaction_ratio_max": (1e-6, "-"),
    "isothermal_stress_ratio_max": (1e-3, "-"), "thermal_support_reaction_ratio_max": (1e-3, "-"), "thermal_support_stress_ratio_max": (5.0, "-"),
    "eff_mass_min_fraction": (0.2, "-"), "eff_mass_cutoff_hz": (1300.0, "Hz"), "freq_convergence_tol": (0.05, "-"), "stress_convergence_tol": (0.5, "-"), "equilibrium_tol": (1e-3, "-"),
    "corr_mac_min": (0.9, "-"), "corr_mac_offdiag_max": (0.3, "-"), "corr_comac_min": (0.9, "-"), "corr_freq_error_max": (0.05, "-"), "corr_peak_freq_shift_max": (0.03, "-"),
    "corr_peak_amp_error_max_db": (3.0, "dB"), "corr_rms_error_max": (0.2, "-")}


def requirement_set(study: Study) -> Any:
    """A requirement set whose every number is an arbitrary test value (binary fractions that cannot be mistaken for a standard value), marked as such."""
    from fea_core.space import budgets as BG
    from fea_core.space import correlation as C
    from fea_core.space.model import LimitEntry, SpaceSettings
    from fea_core.space.requirements import ecss_32_10c_template
    from verification import cases as _cases  # noqa: F401 - the registry first (space_cases imports from it)
    from verification import space_cases as SC
    rs = ecss_32_10c_template()
    rs.name, rs.verification_approach = "ILLUSTRATIVE test set - not ECSS values", "protoflight"
    rs.standard, rs.revision = "ILLUSTRATIVE (no standard)", "illustrative"            # the template is named after a standard: the report must not claim one
    for f in rs.factors:
        if f.approach == "protoflight":
            f.value, f.document, f.revision, f.clause = SC.test_factor(f.key, f.structure_class, "protoflight"), DOC, "illustrative", "illustrative value"
            f.note = "arbitrary test number"
    for r in rs.rules:
        keys = SC.TEST_RULES.get((r.structure_class, r.check))
        if keys is None:
            continue
        r.factors, r.document, r.revision, r.clause = list(keys), DOC, "illustrative", "illustrative rule"
        if r.check != "buckling":
            r.stress_measure, r.thermal_handling, r.thermal_factors = "von_mises", "same_as_mechanical", []
    BG.install_limit_rows(rs)
    C.install_limit_rows(rs)
    have = {x.key for x in rs.limits}
    for k, (_v, u) in ILLUSTRATIVE_LIMITS.items():
        if k not in have:
            rs.limits.append(LimitEntry(key=k, unit=u))
    for x in rs.limits:
        if x.key in ILLUSTRATIVE_LIMITS:
            x.value, x.document, x.revision, x.clause = ILLUSTRATIVE_LIMITS[x.key][0], DOC, "illustrative", "illustrative limit"
            x.unit = x.unit or ILLUSTRATIVE_LIMITS[x.key][1]
    study.space = SpaceSettings(requirement_set=rs)
    return rs


def add_space(study: Study, *, confirm_all: bool = True) -> None:
    """Allowables of every material, the requirement set, the launch environments and the quasi-static cases (all illustrative); confirmed when `confirm_all`."""
    from fea_core.space import environments as EV
    from fea_core.space import loads as LD
    from fea_core.space.model import MaterialAllowable
    from fea_core.space.requirements import confirm, confirm_allowable, set_allowable
    from verification import cases as _cases  # noqa: F401
    from verification.space_load_cases import three_axis_level
    for m in study.materials.values():
        set_allowable(m, MaterialAllowable(basis="B", Fy=float(m.Re or 200.0), Fu=float(m.Rm or 300.0), source="illustrative test value (not a data sheet)"))
        if confirm_all:
            confirm_allowable(m, BY, DATE)
    rs = requirement_set(study)
    lv = three_axis_level()
    for k in lv.kinds():
        t = lv.table(k)
        t.document, t.revision, t.clause = DOC, "illustrative", "illustrative table"
    es = EV.EnvironmentSet(levels=[lv], launch_axis="z", damping_zeta=0.03, damping_source="illustrative test value")
    EV.store_environments(study.space, es)
    if confirm_all:
        confirm(rs, BY, DATE, scope=study.space)
        EV.confirm_environment(es, BY, DATE)
        EV.store_environments(study.space, es)
    LD.generate_qsl(study)


def sensors() -> list[Any]:
    from fea_core.space import correlation as C
    pts = [[50.0, 8.0, 108.0], [92.0, 50.0, 108.0], [8.0, 92.0, 108.0], [-3.0, 25.0, 35.0], [-3.0, 75.0, 80.0], [103.0, 50.0, 55.0], [50.0, -3.0, 55.0], [50.0, 103.0, 70.0],
           [30.0, 30.0, 27.6], [70.0, 30.0, 27.6], [30.0, 70.0, 27.6], [50.0, 50.0, 31.6]]
    return C.sensors_from_dicts([{"node": f"S{k}", "point": p, "dofs": "xyz"} for k, p in enumerate(pts, 1)])


def correlate(study: Study, results: Any, shift: float = 0.015) -> tuple[Any, Any]:
    """The correlation S5 against a SYNTHETIC test (no hardware): the FE modes at the sensors with the frequencies moved by `shift` (1.5 % by default).
    Returns (the result, the test data): the program keeps the test data only by a weak reference, so the caller must hold them while the stored result is verified."""
    from fea_core.space import corr_run as CR
    from fea_core.space import correlation as C
    ms = next(m for m in results.modes if m.kind == "modal")
    sens = sensors()
    td = C.fe_test_data(ms, results.mesh, sens, damping=0.03)
    for m in td.modes:
        m.frequency_hz *= 1.0 + shift
    td.source = "synthetic (the model's own modes, frequencies shifted)"
    res, _ = CR.run_correlation(study, results, td, sensors=sens)
    return res, td


def crosscheck_modes(study: Study, mesh: Any, wd: Path, log: Any) -> dict[str, Any]:
    """The natural frequencies of the direct path (own assembly + rslab + ARPACK) against CalculiX on the same mesh."""
    import copy

    from fea_core.eigen import run_eigen
    out: dict[str, Any] = {}
    for engine in ("direct", "ccx"):
        s = copy.deepcopy(study)
        s.solver.modal_engine = engine
        sets = run_eigen(s, mesh, wd / f"modes_{engine}", log, kinds=("modal",))
        ms = next(m for m in sets if m.kind == "modal")
        out[engine] = {"engine": ms.engine, "frequencies": [float(v) for v in ms.values]}
    n = min(len(out["direct"]["frequencies"]), len(out["ccx"]["frequencies"]))
    d, c = out["direct"]["frequencies"][:n], out["ccx"]["frequencies"][:n]
    out["max_relative_difference"] = max(abs(a - b) / b for a, b in zip(d, c)) if n else None
    out["modes_compared"] = n
    return out


def run_benchmark(out: Path, *, crosscheck: bool = True, convergence: bool = False, report: bool = True, threads: int = 4, log: Any = print) -> dict[str, Any]:
    """The whole chain; returns the facts (timings, the modal cross-check, the verdicts, the path of the report)."""
    import os

    from fea_core.project import save_project
    from fea_core.report import ReportOptions, session_from
    from fea_core.solve import run_study
    from fea_core.space import loads as LD
    from fea_core.space import model_checks as MC
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    facts: dict[str, Any] = {"illustrative": ILLUSTRATIVE}
    t0 = time.time()
    st, parts = build_study(out / "model.step")
    add_space(st)
    st.solver.nproc = threads
    log(f"benchmark: {len(parts)} parts, {len(st.load_cases)} load cases")
    res = run_study(st, out / "run", study_dir=out, on_log=log, parts=parts)
    facts["statics_s"] = round(time.time() - t0, 1)
    facts["mesh"] = {"nodes": int(len(res.mesh.nodes)), "elements": int(len(res.mesh.tets))}
    sess = session_from(st, out / "model.step", res, parts, workdir=out / "work")
    sess.mesh_sig = sess.mesh_signature()
    t1 = time.time()
    r3 = LD.run_3axis(st, res.mesh, out / "r3", log, level="qualification")
    LD.remember(sess, r3)
    facts["three_axis_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    chk = MC.run_for_session(sess, options={"threads": threads, "include_convergence": convergence}, workdir=out / "checks", log=log)
    facts["checks"] = chk.get("verdict")
    facts["checks_s"] = round(time.time() - t1, 1)
    corr, test_data = correlate(st, res)                                  # test_data stays referenced until the report is written
    facts["correlation"] = corr.verdict
    if crosscheck:
        t1 = time.time()
        facts["modal_crosscheck"] = crosscheck_modes(st, res.mesh, out / "xcheck", log)
        facts["crosscheck_s"] = round(time.time() - t1, 1)
    save_project(out / "benchmark.fea", st, out / "model.step", res)
    if report:
        from fea_core.report.space_sar import build_sar
        r = build_sar(sess, out / "SAR_cubesat_benchmark.docx", ReportOptions(ecss_sar=True, language="en", subject=ILLUSTRATIVE), on_log=log)
        facts["report"] = {"file": Path(r["path"]).name, "verdict": r["verdict"], "chapters": {c["title"]: c["status"] for c in r["chapters"]}}
    del test_data
    facts["total_s"] = round(time.time() - t0, 1)
    (out / "benchmark.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verification.cubesat_benchmark", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True, help="the output folder (model.step, benchmark.fea, benchmark.json, the report)")
    ap.add_argument("--build-only", action="store_true", help="only build the study and the mesh")
    ap.add_argument("--no-report", action="store_true", help="skip the Word report")
    ap.add_argument("--no-crosscheck", action="store_true", help="skip the rslab-against-CalculiX comparison of the natural frequencies")
    ap.add_argument("--convergence", action="store_true", help="also run the mesh-convergence model check (several minutes)")
    ap.add_argument("--threads", type=int, default=4, help="solver threads (default 4)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.build_only:
        from fea_core.solve import prepare_mesh
        out.mkdir(parents=True, exist_ok=True)
        st, parts = build_study(out / "model.step")
        mesh, _ = prepare_mesh(st, out, out, print, parts=parts)
        print(f"mesh: {len(mesh.nodes)} nodes, {len(mesh.tets)} elements")
        return 0
    facts = run_benchmark(out, crosscheck=not args.no_crosscheck, convergence=args.convergence, report=not args.no_report, threads=args.threads)
    print(json.dumps(facts, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

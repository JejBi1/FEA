"""Verification of the surrogate model with error bars (fea_core.surrogate, F15-S4) on cheap finite-element sweeps (2D, plane stress).

  SU1  plate with a hole drawn in the sketcher, hole radius and plate width swept (Latin hypercube, 30 fitted + 10 held-out points):
       the surrogate of the maximum stress against direct analyses at 100 NEW random points (median error, coverage of the 90 % band) and
       the analyses against the Heywood / Peterson stress concentration factor Kt
  SU2  cantilever strip, four parameters (length L, thickness b, depth h swept in the sketch, Young's modulus E): the surrogate of the tip
       deflection against analyses at 100 new random points and the analyses against Timoshenko beam theory
  SU3  load factor swept with a categorical material (steel / aluminium / titanium: one model per category): the displacement is
       linear in the load and inversely proportional to E; refusal outside the sampled range, refusal of an unknown category; the band
       against 18 fresh analyses (6 per material)
  SU4  a GRID design (plate with a hole: 12 hole radii × 3 widths): the width has three levels, so the model must refuse a width between
       them (real FE analyses are the proof that the design is not space-filling), answer at the three widths and cover fresh analyses there

The error bars are the split-conformal bands of fea_core.surrogate (the radial-basis backend unless scikit-learn is installed: the
calibration is the same). The surrogate answers only where converged analyses exist (the box of the usable points, their convex hull, no
gaps): the NEW points of SU1 / SU2 are random points of that region (candidates of the whole box are drawn and the ones that the model
refuses are skipped; the share is checked). Run: python -m verification.run_all SU1 SU2 SU3 (several minutes each: every point is a Gmsh + CalculiX run).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from fea_core.surrogate import SurrogateError, fit_surrogate
from fea_core.sketch.build import build_study
from fea_core.sketch.model import new_sketch
from fea_core.study import Displacement, Force, LoadCase, Pressure, Study
from fea_core.sweep import ParametricStudySpec, SweepParam, param_from_key, run_parametric_study
from verification.cases import STEEL, AtLeast, Case, Check, NU
from verification.sketch_cases import _kt_ref, _plate_sketch

SURROGATE_CASES: dict[str, Case] = {}

N_NEW = 100                      # new random points that the surrogate is checked against (SU1, SU2): 20 cannot tell a 90 % band from a 70 % one
COVERAGE_MIN = 0.80              # coverage of the 90 % band at the new points. A band calibrated on 10 held-out points has a realised coverage that is
                                 # itself random (Beta(10, 1): below 0.8 in 11 % of the fits) and 100 new points add the binomial scatter (sd 3 %): a band that
                                 # really holds 90 % has 0.08 % to fall below 80 of 100 at a fixed realised coverage, one that holds 70 % passes with 1.6 %
N_NEW_SU3 = 18                   # fresh analyses of SU3 (6 per material) and N_NEW_SU4 = 12 of SU4: a weak test (see `min_inside`): the 100-point cases
N_NEW_SU4 = 12                   # SU1 / SU2 carry the coverage claim, these two check that the band is not far off and that the points are real analyses


def min_inside(n: int, level: float = 0.9, alpha: float = 0.03) -> int:
    """The least number of n fresh analyses inside the band that a band holding `level` passes with probability above 1 - alpha (exact
    binomial). n = 18: 14 (an honest band fails with 2.8 %, a band that holds 70 % passes with 33 %); n = 12: 9 (2.6 %, 49 %)."""
    from math import comb
    cdf = 0.0
    for k in range(n + 1):
        cdf += comb(n, k) * level ** k * (1 - level) ** (n - k)         # P(X <= k)
        if cdf > alpha:
            return k
    return n


class _Out(dict):
    """The runs of one case; `.warnings` for run_all."""

    def __init__(self, *a, warnings=None, **k):
        super().__init__(*a, **k)
        self._w = list(warnings or [])

    @property
    def warnings(self):
        return list(self._w)


def _new_points(bounds: dict[str, tuple[float, float]], n: int, seed: int) -> list[dict]:
    """n random points, uniform in the box (NOT a Latin hypercube: independent of the design the surrogate was fitted to)."""
    rng = np.random.default_rng(seed)
    keys = list(bounds)
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])
    X = lo + rng.random((n, len(keys))) * (hi - lo)
    return [{k: float(f"{X[i, j]:.8g}") for j, k in enumerate(keys)} for i in range(n)]


def _new_supported(sm, kpi: str, bounds: dict[str, tuple[float, float]], n: int, seed: int) -> tuple[list[dict], float]:
    """n random points of the region where the surrogate answers, and the share of the box that this region covers (the rest of the box –
    corners outside the convex hull of the design – is refused by `predict` and has no analysis to be compared with)."""
    cand = _new_points(bounds, 20 * n, seed)
    keys = [p.key for p in sm.continuous]
    ok = sm.supported(kpi, (), np.array([[c[k] for k in keys] for c in cand]))
    return [c for c, o in zip(cand, ok) if o][:n], float(np.mean(ok))


def _solve_points(study: Study, points: list[dict], workdir: Path, parts=None):
    """The listed parameter points as a parametric study (the same engine as the training points)."""
    keys = list(points[0])
    spec = ParametricStudySpec.from_params([param_from_key(k, [p[k] for p in points]) for k in keys], mode="list", max_points=len(points))
    return run_parametric_study(study, spec, workdir, parts=parts)


def _compare(sm, res, kpi: str) -> dict:
    """The surrogate against the analyses at the new points: relative errors and the coverage of the band."""
    err, inside, widths = [], [], []
    for dp in res.points:
        if not dp.converged:
            continue
        truth = float(getattr(dp, kpi))
        r = sm.predict(dp.params)["kpis"][kpi]
        err.append(abs(r["value"] - truth) / abs(truth))
        inside.append(r["lower"] <= truth <= r["upper"])
        widths.append((r["upper"] - r["lower"]) / 2.0 / abs(truth))
    return {"n": len(err), "median": float(np.median(err)), "max": float(np.max(err)), "coverage": float(np.mean(inside)),
            "half_width": float(np.median(widths))}


# ------------------------------------------------------------------ SU1
SU1_BOUNDS = {"sketch:R": (3.0, 10.0), "sketch:W": (40.0, 60.0)}


def su1_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    st, _ = build_study(_plate_sketch(), wd, "SU1 plate with a hole")
    st.mesh.size = 4.0
    st.mesh.local = [{"selection": "hole", "size": 0.4}]
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0)]
    st.load_cases = [LoadCase("Tah 100 MPa", [Pressure("end", -100.0)])]
    return st, []


def su1_run(study: Study, workdir: Path, parts):
    spec = ParametricStudySpec.from_params([SweepParam("sketch_param", target=k.split(":")[1], bounds=b) for k, b in SU1_BOUNDS.items()],
                                           mode="lhs", max_points=40, seed=11)
    train = run_parametric_study(study, spec, workdir / "train")
    sm = fit_surrogate(train.points, keys=train.keys, kpis=["max_stress"], validation_fraction=0.25, seed=0, bounds=train.bounds, sweep=train.sweep)
    pts, share = _new_supported(sm, "max_stress", SU1_BOUNDS, N_NEW, 2024)
    new = _solve_points(study, pts, workdir / "new")
    return _Out(model=sm, train=train, new=new, share=share, warnings=train.warnings + new.warnings)


def su1_eval(out, st):
    sm, new = out["model"], out["new"]
    c = _compare(sm, new, "max_stress")
    kt = [abs(float(dp.max_stress) - _kt_ref(dp.params["sketch:R"] / dp.params["sketch:W"])) / _kt_ref(dp.params["sketch:R"] / dp.params["sketch:W"])
          for dp in new.points if dp.converged]
    m = sm.members[()]["max_stress"]
    return [
        Check(f"SU1 points solved: 40 training + {N_NEW} new (1 = all)", float(sum(p.converged for p in out["train"].points) == 40 and c["n"] == N_NEW), 1.0, 0.0),
        Check("SU1 held-out points used for the calibration (30 fitted + 10)", float(m.cal.n_train == 30 and m.cal.n_cal == 10), 1.0, 0.0),
        AtLeast("SU1 share of the sampled box where the surrogate answers (hull of 40 points)", out["share"], 0.7, 0.0),
        Check("SU1 the band is exact: guaranteed level 0.90 (10 calibration points; 1 = yes)", float(m.cal.exact and m.cal.guaranteed == 0.9), 1.0, 0.0),
        Check("SU1 analyses against Kt Heywood / Peterson: median relative difference", float(np.median(kt)), 0.0, 0.03),
        Check(f"SU1 surrogate against the analyses at {N_NEW} new points: median relative error", c["median"], 0.0, 0.02),
        AtLeast(f"SU1 coverage of the 90 % band at {N_NEW} new points", c["coverage"], COVERAGE_MIN, 0.0),
        Check("SU1 band is not trivial: median half-width of the band below 10 %", c["half_width"], 0.0, 0.10),
        AtLeast("SU1 validation R² of the model", float(m.cal.r2), 0.9, 0.0),
    ]


SURROGATE_CASES["SU1"] = Case("SU1", "Surrogate model: stress of a plate with a hole over hole radius and width (40 + 100 analyses), error bar coverage",
                              su1_build, su1_eval, run=su1_run)

# ------------------------------------------------------------------ SU2
SU2_BOUNDS = {"sketch:L": (120.0, 200.0), "sketch:b": (8.0, 15.0), "sketch:h": (10.0, 18.0), "prop:Ocel S235:E": (100000.0, 210000.0)}
SU2_FORCE = 200.0              # [N] at the tip


def su2_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    sk = new_sketch("twod", "plane_stress", thickness="b")
    for k, v in (("L", 160.0), ("h", 14.0), ("b", 10.0)):
        sk.set_param(k, v)
    sk.add_entity("line", [[0, 0], ["L", 0]], name="bottom")
    sk.add_entity("line", [["L", 0], ["L", "h"]], name="tip")
    sk.add_entity("line", [["L", "h"], [0, "h"]], name="top")
    sk.add_entity("line", [[0, "h"], [0, 0]], name="wall")
    st, _ = build_study(sk, wd, "SU2 cantilever strip")
    st.mesh.size = 2.5
    st.constraints += [Displacement("wall", ux=0.0, uy=0.0)]
    st.load_cases = [LoadCase("Tip force", [Force("tip", [0.0, -SU2_FORCE, 0.0])])]
    return st, []


def _timoshenko(L, b, h, E, nu=NU, P=SU2_FORCE) -> float:
    I = b * h ** 3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    return P * L ** 3 / (3.0 * E * I) + P * L / (5.0 / 6.0 * G * b * h)


def su2_run(study: Study, workdir: Path, parts):
    mat = study.default_material
    bounds = {k.replace("Ocel S235", mat): b for k, b in SU2_BOUNDS.items()}
    ps = []
    for k, b in bounds.items():
        ps.append(param_from_key(k, bounds=b))
    spec = ParametricStudySpec.from_params(ps, mode="lhs", max_points=80, seed=5)
    train = run_parametric_study(study, spec, workdir / "train")
    sm = fit_surrogate(train.points, keys=train.keys, kpis=["max_disp"], validation_fraction=0.2, seed=0, bounds=train.bounds, sweep=train.sweep)
    pts, share = _new_supported(sm, "max_disp", bounds, N_NEW, 77)
    new = _solve_points(study, pts, workdir / "new")
    return _Out(model=sm, train=train, new=new, mat=mat, share=share, warnings=train.warnings + new.warnings)


def su2_eval(out, st):
    sm, new, mat = out["model"], out["new"], out["mat"]
    c = _compare(sm, new, "max_disp")
    beam = []
    for dp in new.points:
        if dp.converged:
            p = dp.params
            beam.append(abs(dp.max_disp - _timoshenko(p["sketch:L"], p["sketch:b"], p["sketch:h"], p[f"prop:{mat}:E"])) / dp.max_disp)
    m = sm.members[()]["max_disp"]
    return [
        Check(f"SU2 points solved: 80 training + {N_NEW} new (1 = all)", float(sum(p.converged for p in out["train"].points) == 80 and c["n"] == N_NEW), 1.0, 0.0),
        Check("SU2 analyses against Timoshenko beam theory (L, b, h, E): median relative difference", float(np.median(beam)), 0.0, 0.04),
        AtLeast("SU2 share of the sampled box where the surrogate answers (the hull of 80 points in 4 parameters is a part of the box)", out["share"], 0.25, 0.0),
        Check(f"SU2 surrogate against the analyses at {N_NEW} new points: median relative error", c["median"], 0.0, 0.03),
        AtLeast(f"SU2 coverage of the 90 % band at {N_NEW} new points", c["coverage"], COVERAGE_MIN, 0.0),
        AtLeast("SU2 validation R² of the model", float(m.cal.r2), 0.9, 0.0),
    ]


SURROGATE_CASES["SU2"] = Case("SU2", "Surrogate model: tip deflection of a cantilever strip over L, b, h and E (80 + 100 analyses), error bar coverage",
                              su2_build, su2_eval, run=su2_run)

# ------------------------------------------------------------------ SU3
SU3_MATERIALS = {"Hlinik": (70000.0, 0.33, 2.7e-9), "Titan": (110000.0, 0.34, 4.5e-9)}
SU3_BOUNDS = {"load_factor": (0.5, 3.0)}
SU3_POINTS = 96                      # 24 per material: 15 fitted + 9 held out (7 + 9 is the least that gives a 90 % band, but fits a log curve to a few %)


def su3_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    st, _ = build_study(_plate_sketch(), wd, "SU3 plate")
    st.mesh.size = 4.0
    st.mesh.local = [{"selection": "hole", "size": 0.5}]
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0)]
    st.load_cases = [LoadCase("Tah 100 MPa", [Pressure("end", -100.0)])]
    for name, (e, nu, rho) in SU3_MATERIALS.items():
        st.add_material(dataclasses.replace(STEEL, name=name, E=e, nu=nu, rho=rho, Re=300.0))
    return st, []


def su3_run(study: Study, workdir: Path, parts):
    from fea_core.geometry import read_step
    part = read_step(study.geometry_path(None))[0].name        # the one region of the plate
    mats = [study.default_material, *SU3_MATERIALS]
    spec = ParametricStudySpec.from_params([SweepParam("load_factor", bounds=SU3_BOUNDS["load_factor"]),
                                            SweepParam("material", values=mats, target=part)], mode="lhs", max_points=SU3_POINTS, seed=3)
    # the load factor sweeps with a material: every point is solved (no linear shortcut)
    train = run_parametric_study(study, spec, workdir / "train")
    sm = fit_surrogate(train.points, keys=train.keys, kpis=["max_disp"], validation_fraction=0.25, seed=0, bounds=train.bounds, sweep=train.sweep)
    # fresh analyses: 6 random load factors per material, inside the region where the model answers
    key = f"material:{part}"
    rng = np.random.default_rng(31)
    fresh = []
    for m in mats:
        for lf in rng.uniform(0.9, 2.6, N_NEW_SU3 // len(mats)):
            q = {"load_factor": float(f"{lf:.6g}"), key: m}
            if sm.supported("max_disp", (m,), np.array([[q["load_factor"]]]))[0]:
                fresh.append(q)
    new = _solve_points(study, fresh, workdir / "new") if fresh else None
    return _Out(model=sm, train=train, new=new, mats=mats, part=part, warnings=train.warnings + (new.warnings if new else []))


def su3_eval(out, st):
    sm, mats, part = out["model"], out["mats"], out["part"]
    key = f"material:{part}"
    disp = {m: sm.predict({"load_factor": 2.0, key: m})["kpis"]["max_disp"] for m in mats}
    steel = mats[0]
    e = {steel: STEEL.E, **{k: v[0] for k, v in SU3_MATERIALS.items()}}
    checks = [Check(f"SU3 points solved (1 = all {SU3_POINTS})", float(sum(p.converged for p in out["train"].points) == SU3_POINTS), 1.0, 0.0)]
    for m in mats[1:]:
        checks.append(Check(f"SU3 displacement ratio {m} / steel at load factor 2 = E ratio", disp[m]["value"] / disp[steel]["value"], e[steel] / e[m], 0.03))
    checks.append(Check("SU3 displacement of the steel plate is linear in the load factor (u(2.6) / u(0.8) = 3.25)",
                        sm.predict({"load_factor": 2.6, key: steel})["kpis"]["max_disp"]["value"]
                        / sm.predict({"load_factor": 0.8, key: steel})["kpis"]["max_disp"]["value"], 3.25, 0.03))
    new = out["new"]
    inside, err = [], []
    for dp in (new.points if new else []):
        if dp.converged:
            r = sm.predict(dp.params)["kpis"]["max_disp"]
            inside.append(r["lower"] <= dp.max_disp <= r["upper"])
            err.append(abs(r["value"] - dp.max_disp) / dp.max_disp)
    need = min_inside(len(inside))
    checks.append(AtLeast(f"SU3 fresh analyses inside the band (at least {need} of {len(inside)}; an honest 90 % band fails this with at most 3 %)", float(sum(inside)), float(need), 0.0))
    checks.append(Check("SU3 median relative error of the surrogate at the fresh analyses", float(np.median(err)) if err else 1.0, 0.0, 0.03))
    refused = []
    for q in ({"load_factor": 5.0, key: steel}, {"load_factor": 0.1, key: steel}, {"load_factor": 2.0, key: "Mramor"}):
        try:
            sm.predict(q)
            refused.append(0.0)
        except SurrogateError:
            refused.append(1.0)
    checks.append(Check("SU3 refusal above / below the sampled range and of an unknown material (3 of 3)", float(sum(refused)), 3.0, 0.0))
    ex = sm.predict({"load_factor": 5.0, key: steel}, extrapolate=True)
    checks.append(Check("SU3 extrapolate=True predicts with an explicit warning (1 = yes)",
                        float(ex["extrapolated"] and any("EXTRAPOLATION" in w for w in ex["warnings"])), 1.0, 0.0))
    return checks


SURROGATE_CASES["SU3"] = Case("SU3", "Surrogate model: load factor with a categorical material (one model per material), refusal outside the range",
                              su3_build, su3_eval, run=su3_run)


# ------------------------------------------------------------------ SU4: a grid design with a coarse parameter
SU4_R = [float(f"{v:.6g}") for v in np.linspace(3.0, 10.0, 12)]
SU4_W = [40.0, 50.0, 60.0]


def su4_run(study: Study, workdir: Path, parts):
    spec = ParametricStudySpec.from_params([SweepParam("sketch_param", values=SU4_R, target="R"), SweepParam("sketch_param", values=SU4_W, target="W")],
                                           mode="grid", max_points=36)
    train = run_parametric_study(study, spec, workdir / "train")
    sm = fit_surrogate(train.points, keys=train.keys, kpis=["max_stress"], validation_fraction=0.25, seed=0, sweep=train.sweep)
    rng = np.random.default_rng(8)
    fresh = []
    for i in range(N_NEW_SU4):
        q = {"sketch:R": float(f"{rng.uniform(3.6, 9.4):.6g}"), "sketch:W": SU4_W[i % 3]}
        if sm.supported("max_stress", (), np.array([[q[k.key]] for k in sm.continuous]).T)[0]:
            fresh.append(q)
    new = _solve_points(study, fresh, workdir / "new")
    return _Out(model=sm, train=train, new=new, warnings=train.warnings + new.warnings)


def su4_eval(out, st):
    sm, new = out["model"], out["new"]
    c = _compare(sm, new, "max_stress")
    kinds = sm.members[()]["max_stress"].support.kind

    def refuses(q):
        try:
            sm.predict(q)
        except SurrogateError:
            return 1.0
        return 0.0
    n_in = int(round(c["coverage"] * c["n"]))
    return [
        Check("SU4 points solved: 36 grid points + fresh analyses (1 = all)", float(sum(p.converged for p in out["train"].points) == 36 and c["n"] == len(new.points)), 1.0, 0.0),
        Check("SU4 the design is classified as a design with a coarse parameter (1 = yes)", float(kinds == "levels"), 1.0, 0.0),
        Check("SU4 the fit says that the width has only 3 levels (1 = yes)", float(any("sketch:W has only 3 analysed levels" in w for w in sm.warnings)), 1.0, 0.0),
        Check("SU4 refusal of a width BETWEEN the analysed widths (45 mm and 55 mm; 2 of 2)", refuses({"sketch:R": 6.0, "sketch:W": 45.0}) + refuses({"sketch:R": 6.0, "sketch:W": 55.0}), 2.0, 0.0),
        Check("SU4 answer at an analysed width (1 = answered)", 1.0 - refuses({"sketch:R": 6.1, "sketch:W": 50.0}), 1.0, 0.0),
        Check("SU4 surrogate against the fresh analyses at the three widths: median relative error", c["median"], 0.0, 0.03),
        AtLeast(f"SU4 fresh analyses inside the band (at least {min_inside(c['n'])} of {c['n']}; an honest 90 % band fails this with at most 3 %)", float(n_in), float(min_inside(c["n"])), 0.0),
    ]


SURROGATE_CASES["SU4"] = Case("SU4", "Surrogate model: a grid design (12 hole radii x 3 widths) – refusal between the three widths, answers at them",
                              su1_build, su4_eval, run=su4_run)

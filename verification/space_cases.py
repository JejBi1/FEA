"""Verification of the space-structures margins of safety (F14 S1, `fea_core.space.margins`): hand-calculated reference cases.

No solver is involved: a structured tetrahedral block with a prescribed (synthetic) stress field stands for a solved study, so every
margin follows from one line of arithmetic:

    MoS = allowable / (J · σ) − 1,   J = product of the design factors of the rule,   buckling: MoS = λ / J − 1.

The factor values below are ARBITRARY TEST NUMBERS (binary fractions such as 1.0625 that cannot be mistaken for a standard value, and
different for every structure class and verification approach so that a wrong class or approach shows) - they are not values of any
standard; the program ships no factor. Cases: S1A = every structure class × verification approach; S1B = singularity separation (a
spike at the clamped edge must not govern, but must appear in the justification table); S1C = combined mechanical + thermal handling.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from fea_core.eigen import ModeSet
from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.space import margins as MG
from fea_core.space.model import (PART_CLASSES, STRUCTURE_CLASSES, VERIFICATION_APPROACHES, MaterialAllowable, SpaceSettings)
from fea_core.space.requirements import confirm, confirm_allowable, ecss_32_10c_template, set_allowable
from fea_core.study import Fixed, LoadCase, Material, MeshSettings, Selection, Study

from .cases import Case, Check

SPACE_CASES: dict[str, Case] = {}

# ------------------------------------------------------------------ arbitrary test numbers (NOT standard values)
TEST_FY, TEST_FU = 276.0, 310.0          # MPa, test allowables
TEST_SIGMA = 120.0                       # MPa, uniform test stress
BASE_FACTORS = {"FOSY": 1.0625, "FOSU": 1.34375, "KQ": 1.125, "KP": 1.0390625, "KM": 1.15625, "KLD": 1.0234375, "KMP": 1.0078125}
TEST_RULES = {                           # factor keys that form the design factor of each rule (test choice, not the standard's)
    ("metallic", "yield"): ["FOSY", "KQ"], ("metallic", "ultimate"): ["FOSU", "KQ", "KM"],
    ("joints_inserts", "yield"): ["FOSY", "KP"], ("joints_inserts", "ultimate"): ["FOSU", "KP", "KLD"],
    ("buckling", "buckling"): ["FOSU", "KLD", "KMP"],
}
TEST_LAMBDA = 4.5                        # lowest buckling load factor of the synthetic modal result


def test_factor(key: str, cls: str, approach: str) -> float:
    """Arbitrary test value of a factor: the base number plus exact binary fractions per class and approach."""
    return BASE_FACTORS[key] + 0.015625 * STRUCTURE_CLASSES.index(cls) + 0.03125 * VERIFICATION_APPROACHES.index(approach)


def design_factor(cls: str, check: str, approach: str) -> float:
    j = 1.0
    for k in TEST_RULES[(cls, check)]:
        j *= test_factor(k, cls, approach)
    return j


# ------------------------------------------------------------------ synthetic models
def box_tet_mesh(nx: int = 8, ny: int = 2, nz: int = 2, h: float = 5.0) -> Mesh:
    """Conforming 4-node tetrahedral mesh of a box (nx·h × ny·h × nz·h) from the six tetrahedra of each cube; the face x = 0 is the
    selection "Support"."""
    def idx(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    nodes = np.array([[i * h, j * h, k * h] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)], float)
    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        for perm in itertools.permutations(range(3)):
            c = [i, j, k]
            path = [idx(*c)]
            for ax in perm:
                c[ax] += 1
                path.append(idx(*c))
            t = np.array(path)
            vol = np.linalg.det(nodes[t[1:]] - nodes[t[0]])
            tets.append(t if vol > 0 else t[[0, 2, 1, 3]])
    m = Mesh(nodes, np.array(tets, np.int64), np.zeros(len(tets), np.int64), ["Block"])
    tri, _ = m.boundary_faces()
    m.surfaces["Support"] = tri[(nodes[tri[:, :3], 0] <= nodes[:, 0].min() + 1e-9).all(axis=1)]
    return m


def block_study(material: Material | None = None) -> Study:
    st = Study(name="space margins reference", geometry="block.step", mesh=MeshSettings(size=5.0))
    st.add_material(material or Material("TestAlloy", E=70000.0, nu=0.33, rho=2.7e-9, Re=250.0, Rm=300.0), default=True)
    st.selections["Support"] = Selection("Support")
    st.constraints.append(Fixed("Support"))
    st.load_cases.append(LoadCase("Load", []))
    return st


def uniform_state(mesh: Mesh, name: str, sigma: float, spike: float = 0.0) -> StepResult:
    """Uniaxial σ_x = sigma everywhere; `spike` × sigma at the nodes of the support face (a singularity of the clamped edge)."""
    S = np.zeros((len(mesh.nodes), 6))
    S[:, 0] = sigma
    if spike:
        S[mesh.nodes[:, 0] <= mesh.nodes[:, 0].min() + 1e-9, 0] = sigma * spike
    return StepResult(name=name, U=np.zeros((len(mesh.nodes), 3)), S=S)


def make_requirement_set(approach: str, rules: dict[tuple[str, str], list[str]] | None = None, thermal: str = "same_as_mechanical",
                         thermal_factors: list[str] | None = None, confirmed: bool = True):
    """A requirement set filled with the arbitrary test factors for `approach` (and confirmed by "test engineer")."""
    rs = ecss_32_10c_template()
    rs.name = "test requirement set (arbitrary numbers)"
    rs.verification_approach = approach
    for f in rs.factors:
        if f.approach == approach:
            f.value = test_factor(f.key, f.structure_class, approach)
            f.revision, f.clause = "test revision", "test clause"
    for r in rs.rules:
        keys = (rules or TEST_RULES).get((r.structure_class, r.check))
        if keys is None:
            continue
        r.factors = list(keys)
        r.revision, r.clause = "test revision", "test clause"
        if r.check != "buckling":
            r.stress_measure = "von_mises"
            r.thermal_handling = thermal
            r.thermal_factors = list(thermal_factors or [])
    if confirmed:
        confirm(rs, "test engineer", "2026-01-01")
    return rs


def confirm_test_allowables(study: Study, fy: float = TEST_FY, fu: float = TEST_FU, basis: str = "B") -> None:
    mat = study.materials[study.default_material or ""]
    set_allowable(mat, MaterialAllowable(basis=basis, Fy=fy, Fu=fu, source="test data sheet"))
    confirm_allowable(mat, "test engineer", "2026-01-01")


def synthetic_results(mesh: Mesh, steps: list[StepResult], buckling: bool = False) -> Results:
    res = Results(mesh=mesh, steps=steps)
    if buckling:
        res.modes = [ModeSet("buckling", "Buckling", [TEST_LAMBDA, 2 * TEST_LAMBDA], np.zeros((2, 1, 3)), case="Load")]
    return res


class _Out(dict):
    """The runs for the evaluation; `.warnings` for run_all."""

    @property
    def warnings(self) -> list[str]:
        return []


def _build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return block_study(), []


# ------------------------------------------------------------------ S1A: every structure class × verification approach
def s1a_run(study: Study, workdir: Path, parts) -> _Out:
    mesh = box_tet_mesh()
    out = _Out()
    for approach in VERIFICATION_APPROACHES:
        for cls in STRUCTURE_CLASSES:
            st = block_study()
            confirm_test_allowables(st)
            st.space = SpaceSettings(requirement_set=make_requirement_set(approach), default_class=cls if cls in PART_CLASSES else "metallic")
            confirm(st.space.requirement_set, "test engineer", "2026-01-01", scope=st.space)       # confirmed for these classes
            res = synthetic_results(mesh, [uniform_state(mesh, "Load", TEST_SIGMA)], buckling=True)
            out[(approach, cls)] = MG.compute_margins(st, res)
    return out


def s1a_eval(out: _Out, study: Study) -> list[Check]:
    checks: list[Check] = []
    for (approach, cls), summ in out.items():
        tag = f"S1A {cls} / {approach}"
        if cls == "composite":                     # no stress-based rule exists: never assessed, never PASS
            checks.append(Check(f"{tag}: not assessed -> UNCONFIRMED_REQUIREMENTS", float(summ.verdict == MG.UNCONFIRMED), 1.0, 0.0))
            continue
        if cls == "metallic":
            checks.append(Check(f"{tag}: everything confirmed, margins >= 0 -> PASS", float(summ.verdict == MG.PASS), 1.0, 0.0))
        if cls == "buckling":
            ref = TEST_LAMBDA / design_factor(cls, "buckling", approach) - 1.0
            got = [r.mos for r in summ.rows if r.check == "buckling" and r.mos is not None]
            checks.append(Check(f"{tag}: MoS = lambda / J - 1", got[0] if got else float("nan"), ref, 1e-12))
            continue
        for check, allow in (("yield", TEST_FY), ("ultimate", TEST_FU)):
            ref = allow / (design_factor(cls, check, approach) * TEST_SIGMA) - 1.0
            got = [r.mos for r in summ.rows if r.check == check and r.mos is not None]
            checks.append(Check(f"{tag}: {check} MoS = F / (J sigma) - 1", got[0] if got else float("nan"), ref, 1e-12))
    return checks


SPACE_CASES["S1A"] = Case("S1A", "Space margins: MoS = F/(J sigma) - 1, every structure class x verification approach (arbitrary test factors)",
                          _build, s1a_eval, run=s1a_run)


# ------------------------------------------------------------------ S1B: singularity separation
def s1b_run(study: Study, workdir: Path, parts) -> _Out:
    mesh = box_tet_mesh()
    st = block_study()
    confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=make_requirement_set("protoflight"))
    res = synthetic_results(mesh, [uniform_state(mesh, "Load", TEST_SIGMA, spike=3.0)])
    return _Out(summary=MG.compute_margins(st, res))


def s1b_eval(out: _Out, study: Study) -> list[Check]:
    s = out["summary"]
    y = next(r for r in s.rows if r.check == "yield")
    j = design_factor("metallic", "yield", "protoflight")
    return [Check("S1B yield MoS outside the singularity (nominal stress)", y.mos or float("nan"), TEST_FY / (j * TEST_SIGMA) - 1.0, 1e-12),
            Check("S1B yield MoS at the clamped edge (3x spike), reported separately", y.mos_raw or float("nan"),
                  TEST_FY / (j * 3.0 * TEST_SIGMA) - 1.0, 1e-12),
            Check("S1B the clamped edge is in the singularity table", float(len(s.singularities) >= 1), 1.0, 0.0),
            Check("S1B no sign-off before the singularity is justified", float(s.signoff_ready), 0.0, 0.0)]


SPACE_CASES["S1B"] = Case("S1B", "Space margins: a singular clamped edge does not govern the margin and goes to the justification table",
                          _build, s1b_eval, run=s1b_run)


# ------------------------------------------------------------------ S1C: combined mechanical + thermal loads
SIGMA_MECH, SIGMA_THERM = 100.0, 20.0


def s1c_run(study: Study, workdir: Path, parts) -> _Out:
    from fea_core.space.model import CaseComponent, MarginCase
    mesh = box_tet_mesh()
    out = _Out()
    for handling, extra in (("same_as_mechanical", None), ("none", None), ("separate", ["KMP"])):
        st = block_study()
        st.load_cases = [LoadCase("Mech", []), LoadCase("Therm", [])]
        confirm_test_allowables(st)
        st.space = SpaceSettings(requirement_set=make_requirement_set("analysis_only", thermal=handling, thermal_factors=extra),
                                 cases=[MarginCase("Mech+Therm", [CaseComponent("Mech", 1.0, "mechanical"),
                                                                    CaseComponent("Therm", 1.0, "thermal")])])
        res = synthetic_results(mesh, [uniform_state(mesh, "Mech", SIGMA_MECH), uniform_state(mesh, "Therm", SIGMA_THERM)])
        if handling == "same_as_mechanical":                  # a relieving thermal load (scale -1): ECSS-E-ST-32-10C 4.3.2.1 c, factor 1,0 instead of J
            st.space.cases.append(MarginCase("Mech-Therm", [CaseComponent("Mech", 1.0, "mechanical"), CaseComponent("Therm", -1.0, "thermal")]))
            st.space.cases.append(MarginCase("RelMech+Therm", [CaseComponent("Mech", -0.1, "mechanical"), CaseComponent("Therm", 1.0, "thermal")]))        # sigma_m = -10, sigma_th = +20
        out[handling] = MG.compute_margins(st, res)
    return out


def s1c_eval(out: _Out, study: Study) -> list[Check]:
    jm = design_factor("metallic", "yield", "analysis_only")
    j_t = {"same_as_mechanical": jm, "none": 1.0, "separate": test_factor("KMP", "metallic", "analysis_only")}
    checks = []
    for handling, summ in out.items():
        row = next(r for r in summ.rows if r.case == "Mech+Therm" and r.check == "yield")
        design = jm * SIGMA_MECH + j_t[handling] * SIGMA_THERM
        checks.append(Check(f"S1C thermal handling '{handling}': MoS = F / (J_m s_m + J_t s_t) - 1", row.mos or float("nan"),
                            TEST_FY / design - 1.0, 1e-12))
    rel = next(r for r in out["same_as_mechanical"].rows if r.case == "Mech-Therm" and r.check == "yield")
    checks.append(Check("S1C relieving thermal load: MoS = F / (J_m s_m - 1,0 s_t) - 1 (factor 1,0 on the relieving part, ECSS-E-ST-32-10C 4.3.2.1 c)", rel.mos or float("nan"),
                        TEST_FY / (jm * SIGMA_MECH - SIGMA_THERM) - 1.0, 1e-12))
    relm = next(r for r in out["same_as_mechanical"].rows if r.case == "RelMech+Therm" and r.check == "yield")
    checks.append(Check("S1C relieving mechanical load against a dominant thermal one: MoS = F / (1,0 s_m + J_m s_th) - 1 (ECSS-E-ST-32-10C 4.3.2.1 c, third combination)",
                        relm.mos or float("nan"), TEST_FY / (-0.1 * SIGMA_MECH + jm * SIGMA_THERM) - 1.0, 1e-12))
    return checks


SPACE_CASES["S1C"] = Case("S1C", "Space margins: combined mechanical + thermal load, distinct factor handling",
                          _build, s1c_eval, run=s1c_run)


# ------------------------------------------------------------------ S1D: a solved bar in tension (the real solver, margins outside the clamp)
def s1d_build(wd: Path):
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces

    from .cases import _study, box
    wd.mkdir(parents=True, exist_ok=True)
    st, parts = _study("S1D bar in tension", wd / "model.step", [Part("Bar", box(0, 0, 0, 60, 10, 10))], MeshSettings(size=4.0))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.add_selection("end", find_faces(parts, x=60))
    st.constraints.append(Fixed("clamp"))
    from fea_core.study import Force
    st.load_cases.append(LoadCase("Tension", [Force("end", [5000.0, 0.0, 0.0])]))          # 5000 N / 100 mm2 = 50 MPa
    confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=make_requirement_set("protoflight"))
    return st, parts


def s1d_eval(res: Results, study: Study) -> list[Check]:
    summ = MG.compute_margins(study, res)
    y = next(r for r in summ.rows if r.check == "yield" and r.mos is not None)
    j = design_factor("metallic", "yield", "protoflight")
    ref = TEST_FY / (j * 50.0) - 1.0
    return [Check("S1D yield MoS of the solved bar, outside the clamp = F / (J F/A) - 1", y.mos or float("nan"), ref, 0.05),
            Check("S1D the clamp edge is not the governing place", float(y.classification != "singularity"), 1.0, 0.0),
            Check("S1D everything confirmed -> PASS", float(summ.verdict == MG.PASS), 1.0, 0.0)]


SPACE_CASES["S1D"] = Case("S1D", "Space margins: a solved bar in tension, MoS outside the clamp = F/(J F/A) - 1 (real solver, arbitrary test factors)",
                          s1d_build, s1d_eval)


# ---- launch loads, mass properties and budgets (F14 S2): the cases live in their own module and join here
from .space_load_cases import SPACE_LOAD_CASES  # noqa: E402

SPACE_CASES.update(SPACE_LOAD_CASES)

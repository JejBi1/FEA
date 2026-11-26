"""A normal elastic support through the REAL CalculiX with `*STATIC, SOLVER=SPOOLES` – the deck that used to end with "matrix found to be singular" (fea_core.springs).

tests/test_springs_deck.py checks the text of the deck only. Here a real Gmsh mesh goes through ccx (statics, natural frequencies and a nonlinear NLGEOM step) and the
answers are compared with closed-form solutions, not with the code that writes the deck:

  slanted  a prism with a bottom face inclined by 30° (normal spring on it), vertical pressure p on the top, the vertical wall holds ux = 0:
           the foundation carries the vertical force F = p·A_top along the normal n = (sin θ, 0, −cos θ): R = F/cos θ·(−n) (R_x = −F·tan θ, R_z = F), and
           the mean normal displacement is p/k (the inclined area is A_top/cos θ); the body moves as a rigid one on the springs: f1 = ½π⁻¹·√(k·A·cos²θ/m).
  curved   a quarter of a cylinder (r = 20) lying on its curved face (normal = radial spring), pressure p on the flat top: a rigid body sinking by δ presses the
           radial springs with k·δ·cos φ → δ = 4p/(π·k), R_x/R_z = 2/π, f1 = ½π⁻¹·√(k/(ρ·r)).
SOLVER=SPOOLES is asked for explicitly: PaStiX of the Windows ccx_static lets the zero rows of the auxiliary node pass, SPOOLES (the Linux package, every natural-
frequency run) stopped. Reactions: R = −k·u (the stiffnesses written to the deck), the equilibrium of the program, the deck of every job carries the fixed DOFs."""
from __future__ import annotations

import math

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec

from fea_core import config, springs
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import run_study
from fea_core.study import Displacement, ElasticSupport, LoadCase, MeshSettings, Pressure
from verification.cases import _study, box, cyl, solid

pytestmark = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")

K, P = 10.0, 1.0                    # N/mm³, MPa
THETA = math.radians(30.0)
WIDTH, TOP, DEPTH = 40.0, 60.0, 20.0
RADIUS = 20.0


def _wedge():
    t = math.tan(THETA)
    poly = BRepBuilderAPI_MakePolygon()
    for x, z in [(0, 0), (WIDTH, WIDTH * t), (WIDTH, TOP), (0, TOP)]:
        poly.Add(gp_Pnt(x, 0, z))
    poly.Close()
    return solid(BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(poly.Wire()).Face(), gp_Vec(0, DEPTH, 0)).Shape())


def _study_of(kind: str, wd, nlgeom: bool):
    if kind == "slanted":
        st, parts = _study("slanted support", wd / "m.step", [Part("Wedge", _wedge())], MeshSettings(size=6.0))
        st.add_selection("base", find_faces(parts, normal=[math.sin(THETA), 0.0, -math.cos(THETA)]))
        st.add_selection("top", find_faces(parts, z=TOP))
    else:
        quarter = solid(BRepAlgoAPI_Common(cyl((0, 0, 0), (0, 1, 0), RADIUS, DEPTH), box(0, 0, -RADIUS, RADIUS, DEPTH, RADIUS)).Shape())
        st, parts = _study("curved support", wd / "m.step", [Part("Quarter", quarter)], MeshSettings(size=4.0))
        st.add_selection("base", find_faces(parts, radius=RADIUS))
        st.add_selection("top", find_faces(parts, z=0))
    st.add_selection("wall", find_faces(parts, x=0))
    st.add_selection("front", find_faces(parts, y=0))
    st.constraints += [ElasticSupport("base", K, "normal"), Displacement("wall", ux=0.0), Displacement("front", uy=0.0)]
    st.load_cases.append(LoadCase("pressure", [Pressure("top", P)]))
    st.solver.solver = "SPOOLES"
    st.solver.backend = "ccx"
    if nlgeom:
        st.analysis.kind = "nonlinear"
        st.contact_settings.nlgeom = True
    else:
        st.modal.enabled, st.modal.modes = True, 3
    return st, parts


@pytest.fixture(scope="module", params=["slanted", "curved"])
def run(request, tmp_path_factory):
    kind = request.param
    out = {"kind": kind}
    for tag, nlgeom in (("lin", False), ("nl", True)):
        wd = tmp_path_factory.mktemp(f"{kind}_{tag}")
        st, parts = _study_of(kind, wd, nlgeom)
        out[tag] = (st, run_study(st, wd / "run", parts=parts), wd / "run")
    return out


def _force_top() -> dict:
    return {"slanted": P * WIDTH * DEPTH, "curved": P * RADIUS * DEPTH}


def test_spooles_solves_the_statics_the_natural_frequencies_and_the_nonlinear_step(run):
    for tag, expected in (("lin", {"static": "SPOOLES", "modal": "SPOOLES"}), ("nl", {"static": "SPOOLES"})):
        res = run[tag][1]
        assert res.info["ccx_solvers"] == expected, (tag, res.info["ccx_solvers"])
        assert res.step(0).equilibrium["imbalance"] < 1e-5
    assert run["nl"][1].info["nonlinear"]["mode"]                          # the nonlinear route really ran (a branch with NLGEOM)
    decks = {f"{tag}/{p.relative_to(run[tag][2]).as_posix()}": p.read_text(encoding="utf-8", errors="replace") for tag in ("lin", "nl") for p in run[tag][2].rglob("*.inp")}
    assert sorted(decks) == ["lin/eigen/modal.inp", "lin/job.inp", "nl/nl_b1.inp"], sorted(decks)          # statics, natural frequencies, the nonlinear branch
    assert any("NLGEOM" in t and "SOLVER=SPOOLES" in t for t in decks.values()), list(decks)       # the nonlinear deck asks SPOOLES too
    assert any("*FREQUENCY" in t for t in decks.values())
    for name, text in decks.items():
        if "*STEP" in text:
            assert "NSPR, 2, 3, 0.\n" in text, name                       # every job that has a step fixes the DOFs 2 and 3 of the auxiliary nodes


def test_the_reaction_of_the_foundation_is_the_closed_form_one_and_balances_the_load(run):
    st, res, _ = run["lin"]
    s = res.step(0)
    F = _force_top()[run["kind"]]
    R = s.reactions["base"]
    assert s.equilibrium["imbalance"] < 1e-5
    assert R[2] == pytest.approx(F, rel=1e-5)                              # the whole load goes through the springs (+ the wall in x)
    assert abs(R[1]) < 1e-5 * F
    if run["kind"] == "slanted":
        assert R[0] == pytest.approx(-F * math.tan(THETA), rel=1e-5)       # along the normal of the inclined face
        n = np.array([math.sin(THETA), 0.0, -math.cos(THETA)])
        assert np.linalg.norm(np.cross(R / np.linalg.norm(R), n)) < 1e-6
    else:
        assert -R[0] / R[2] == pytest.approx(2 / math.pi, rel=0.03)        # rigid body on radial springs
    assert s.reactions["wall"][0] == pytest.approx(-R[0], rel=1e-5)


def test_the_reaction_is_minus_k_times_the_displacement_with_the_written_stiffnesses(run):
    st, res, _ = run["lin"]
    s = res.step(0)
    tab = springs.table(st, res.mesh)[0]
    u = s.U[tab["idx"]]
    forces = -(tab["k"] * np.einsum("ij,ij->i", u, tab["normal"]))[:, None] * tab["normal"]
    assert forces.sum(axis=0) == pytest.approx(s.reactions["base"], rel=1e-9, abs=1e-9)
    # the stiffnesses written to the deck are k per area: the total of the nodal ones is k·A of the face (rounding of 0.5 % per node at most)
    assert tab["k"].sum() == pytest.approx(K * tab["area"], rel=0.003)
    # and the displacement of the real run answers the closed form: the deck's springs have the stiffness that was asked for
    if run["kind"] == "slanted":
        assert -float(np.einsum("ij,ij->i", u, tab["normal"]).mean()) == pytest.approx(-P / K, rel=0.01)    # n points out of the body: u·n = −p/k
    else:
        top = np.abs(res.mesh.nodes[:, 2]) < 1e-6
        assert -float(s.U[top, 2].mean()) == pytest.approx(4 * P / (math.pi * K), rel=0.01)


def test_the_first_natural_frequency_is_the_body_on_the_springs(run):
    st, res, _ = run["lin"]
    (modes,) = [m for m in res.modes if m.kind == "modal"]
    f = np.asarray(modes.values, float)
    assert len(f) == 3 and (f > 0).all() and f[1] > 10 * f[0]               # no spurious mechanism: the next one is the elastic body (≫ the rigid one)
    rho = st.materials[st.default_material].rho
    if run["kind"] == "slanted":
        k_eff = K * (WIDTH * DEPTH / math.cos(THETA)) * math.cos(THETA) ** 2
        volume = DEPTH * WIDTH * (TOP + (TOP - WIDTH * math.tan(THETA))) / 2
        ref = math.sqrt(k_eff / (rho * volume)) / (2 * math.pi)
    else:
        ref = math.sqrt(K / (rho * RADIUS)) / (2 * math.pi)
    assert f[0] == pytest.approx(ref, rel=0.01)


def test_the_nonlinear_step_gives_the_same_small_displacements(run):
    (_, lin, _), (_, nl, _) = run["lin"], run["nl"]
    a, b = lin.step(0), nl.step(0)
    assert np.abs(a.U - b.U).max() < 1e-3 * np.abs(a.U).max()              # 0.1 mm on 20–60 mm: geometric nonlinearity is a rounding
    assert b.reactions["base"] == pytest.approx(a.reactions["base"], rel=1e-3, abs=1e-6)
    assert b.equilibrium["imbalance"] < 1e-5


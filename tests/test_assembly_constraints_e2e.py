"""The constraints of the own assembly end to end (F13 step 1b), nothing mocked: a real Gmsh mesh, the job files of `solve_large._run_ngs_job` (reduction.npz,
aniso.npz, the padded F / fixed / g), the assembly worker, the direct solver, the expansion of the dependent DOFs, the stress recovery, the reactions and
`make_step` – against CalculiX on the SAME mesh: displacements, stresses, reactions, the motion and the reaction of a remote constraint, the equilibrium.

Studies: (1) cylindrical support + elastic support (sharing nodes) + a prescribed displacement + five load states (force, pressure, weight, temperature,
a combination); (2) two bodies joined by *TIE (the second finer) + a rigid remote support + rigid remote loads (two points of one face) + a distributing
remote load + weight and temperature; (3) a distributing remote support + a rigid remote load; (4) orthotropic materials: a rotated Cartesian one and a
cylindrical one with thermal expansion.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import choose_backend_reason, prepare_mesh, run_study
from fea_core.study import (CoordinateSystem, CylindricalSupport, Displacement, ElasticSupport, Fixed, Force, Gravity, LoadCase, Material, MeshSettings,
                            Pressure, RemoteLoad, RemoteSupport, ThermalLoad)
from verification.cases import _study, box, cyl, solid
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_chain = pytest.mark.skipif(not (HAS_DIRECT and config.CCX_EXE.exists()), reason="numba + rslab (modules accel, direct) and CalculiX are needed")


@pytest.fixture(scope="module", autouse=True)
def _rslab_on_the_path_of_the_worker():
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it too.
    Module scope: the fixtures of the studies run the whole chain themselves (other tests import the installer, which clears PYTHONPATH of the process)."""
    mp = pytest.MonkeyPatch()
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        mp.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    yield
    mp.undo()


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _compare(ref, res, tol_u=2e-5, tol_s=5e-5, tol_r=1e-4, skip_nodes=(), tol_eq=1e-6):
    """The own path against ccx: displacements, stresses, reactions (forces and moments), the remote constraints, the equilibrium.
    skip_nodes = nodes left out of the comparison of the stresses, tol_eq = the imbalance of the own path."""
    keep = np.ones(len(ref.mesh.nodes), bool)
    keep[list(skip_nodes)] = False
    assert len(ref.steps) == len(res.steps)
    f_scale = max((np.abs(v).max() for st in ref.steps for v in st.reactions.values()), default=1.0)
    m_scale = max((np.abs(v).max() for st in ref.steps for v in (st.reaction_moments or {}).values()), default=1.0)
    size = float(np.ptp(ref.mesh.nodes, axis=0).max())
    f_scale = max(f_scale, m_scale / size)
    for a, b in zip(ref.steps, res.steps):
        assert _rel(b.U, a.U) < tol_u, (a.name, "U", _rel(b.U, a.U))
        assert _rel(b.S[keep], a.S[keep]) < tol_s, (a.name, "S", _rel(b.S[keep], a.S[keep]))
        assert b.equilibrium["imbalance"] < tol_eq, a.name
        for k in a.reactions:
            assert np.abs(a.reactions[k] - b.reactions[k]).max() / f_scale < tol_r, (a.name, k, a.reactions[k], b.reactions[k])
            assert np.abs(a.reaction_moments[k] - b.reaction_moments[k]).max() / max(m_scale, 1e-300) < tol_r, (a.name, k, "moment")
        for sel, info in (a.remote or {}).items():
            o = b.remote[sel]
            assert np.abs(np.subtract(o["force"], info["force"])).max() / f_scale < tol_r, (a.name, sel)
            assert np.abs(np.subtract(o["moment"], info["moment"])).max() / max(m_scale, 1e-300) < tol_r, (a.name, sel)
            noise = 1e-5 * np.abs(a.U).max()          # a held component of the point of a distributing support comes from the 6-digit displacements of the .frd of ccx
            assert np.abs(np.subtract(o["u"], info["u"])).max() <= 2e-5 * np.abs(info["u"]).max() + noise, (a.name, sel)
            assert np.abs(np.subtract(o["rot"], info["rot"])).max() <= 2e-5 * np.abs(info["rot"]).max() + noise, (a.name, sel)


def _both(st, parts, wd, solver="direct"):
    """(ccx result, own result, the log of the own run) on one mesh."""
    from fea_core.solve_large import run_ngs
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.solver.backend = "ccx"
    ref = run_study(st, wd / "ccx", mesh=mesh, parts=parts)
    logs: list[str] = []
    res = run_ngs(st, mesh, wd / "own", [], logs.append, solver=solver)
    return ref, res, logs, mesh


# ------------------------------------------------------------------ (1) cylindrical support + elastic support + prescribed displacement
def _study_cyl_elastic(wd: Path):
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 12), cyl((30, 20, -1), (0, 0, 1), 6, 14)).Shape())
    st, parts = _study("cyl + elastic", wd / "model.step", [Part("Deska", plate)], MeshSettings(size=4.0))
    st.materials[st.default_material].alpha = 1.2e-5
    for name, kw in (("dira", dict(radius=6)), ("dno", dict(z=0)), ("x0", dict(x=0)), ("x1", dict(x=60)), ("vrch", dict(z=12))):
        st.add_selection(name, find_faces(parts, **kw))
    st.constraints += [CylindricalSupport("dira", radial=True, tangential=False, axial=False), ElasticSupport("dno", 30.0, "normal"),
                       Displacement("x0", ux=0.004)]
    st.load_cases += [LoadCase("sila", [Force("x1", [-800.0, 300.0, 0.0])]), LoadCase("tlak", [Pressure("vrch", 2.0)]),
                      LoadCase("tiha", [Gravity([0.0, 0.0, -9810.0])]), LoadCase("teplo", [ThermalLoad(value=70.0)]),
                      LoadCase("kombinace", [Force("x1", [0.0, 0.0, -500.0]), Pressure("vrch", 1.0), ThermalLoad(value=40.0)])]
    return st, parts


@pytest.fixture(scope="module")
def cyl_elastic(tmp_path_factory):
    wd = tmp_path_factory.mktemp("cyl_elastic")
    st, parts = _study_cyl_elastic(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_cylindrical_and_elastic_support_with_a_prescribed_displacement_equal_ccx(cyl_elastic):
    st, parts, wd, ref, res, logs, mesh = cyl_elastic
    _compare(ref, res)
    info = res.info["solver"]
    assert info["backend"] == "ngs_direct" and info["constraints"]["cylindrical"] > 100 and info["constraints"]["springs"] > 100
    # the cylindrical support holds the radial component at the nodes of the hole (u·e_r = 0), the prescribed face moves by the prescription
    from fea_core.cylinder import support_directions
    idx, D = support_directions(st, mesh)["dira"]
    for s in res.steps:
        assert np.abs(np.einsum("nj,nkj->nk", s.U[idx], D)).max() < 1e-12 * np.abs(s.U).max()
    x0 = np.abs(mesh.nodes[:, 0]) < 1e-9
    assert np.allclose(res.steps[0].U[x0, 0], 0.004, atol=1e-12)
    # the foundation and the support share the nodes of the circle: the reaction of the foundation is counted once (imbalance of the state with the pressure)
    assert res.step(1).equilibrium["imbalance"] < 1e-9 and abs(res.step(1).reactions["dno"][2]) > 0.5 * abs(res.step(1).equilibrium["applied_exact"][2])


@needs_chain
def test_the_iterative_solver_takes_cylindrical_and_elastic_supports(cyl_elastic):
    from fea_core import hypre
    if not hypre.available():
        pytest.skip("the iterative solver (module hypre, MPI) is not installed")
    from fea_core.solve_large import run_ngs
    st, parts, wd, ref, res, logs, mesh = cyl_elastic
    st.solver.nproc = 2
    hy = run_ngs(st, mesh, wd / "hypre", [], lambda s: None, solver="hypre")
    _compare(ref, hy, tol_u=5e-5, tol_s=2e-4)


@needs_chain
def test_auto_chooses_the_direct_solver_for_these_studies_and_the_iterative_one_only_where_it_can(cyl_elastic):
    st, parts, wd, ref, res, logs, mesh = cyl_elastic
    st.solver.backend = "auto"
    backend, reason = choose_backend_reason(st, mesh)
    assert backend == "ngs_direct", reason
    from fea_core.solve_large import unsupported
    assert unsupported(st) == [] and unsupported(st, "hypre") == []
    st.constraints.append(RemoteSupport("x1", [80.0, 20.0, 6.0], [True] * 6, "rigid"))
    try:
        assert unsupported(st) == [] and any("direct solver only" in w for w in unsupported(st, "hypre"))
        from fea_core import solvers
        assert any("cannot handle this study" in e for e in solvers.check_backend(st, "ngs_hypre"))
        assert not solvers.check_backend(st, "ngs_direct")
    finally:
        st.constraints.pop()


# ------------------------------------------------------------------ (2) *TIE + rigid remote support and loads + distributing load
def _study_tie_remote(wd: Path):
    st, parts = _study("tie + remote", wd / "model.step", [Part("A", box(0, 0, 0, 60, 24, 16)), Part("B", box(60, 0, 0, 60, 24, 16))], MeshSettings(size=6.0))
    st.materials[st.default_material].alpha = 1.2e-5
    st.mesh.separate = ["B"]
    st.add_selection("b_cela", find_faces(parts, part="B", normal=[-1, 0, 0]))
    st.mesh.local = [{"selection": "b_cela", "size": 3.6}]
    st.add_selection("a0", find_faces(parts, part="A", x=0))
    st.add_selection("b1", find_faces(parts, part="B", x=120))
    st.add_selection("b_vrch", find_faces(parts, part="B", z=16))
    st.constraints.append(RemoteSupport("a0", [-20.0, 12.0, 8.0], [True] * 6, "rigid"))
    st.load_cases += [
        LoadCase("rigid", [RemoteLoad("b1", [140.0, 12.0, 8.0], [0.0, 0.0, -400.0], [0.0, 3000.0, 0.0], "rigid")]),
        LoadCase("distribuovane", [RemoteLoad("b_vrch", [90.0, 12.0, 30.0], [100.0, 0.0, -250.0], [0.0, 0.0, 1500.0], "distributing"), Pressure("b_vrch", 0.5)]),
        LoadCase("tiha a teplo", [Gravity([0.0, 0.0, -9810.0]), ThermalLoad(value=60.0)]),
        LoadCase("dva body", [RemoteLoad("b1", [130.0, 12.0, 8.0], [50.0, 100.0, -100.0], [500.0, 0.0, 0.0], "rigid"), ThermalLoad(value=30.0)])]
    return st, parts


@pytest.fixture(scope="module")
def tie_remote(tmp_path_factory):
    wd = tmp_path_factory.mktemp("tie_remote")
    st, parts = _study_tie_remote(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_tie_rigid_remote_support_and_loads_equal_ccx(tie_remote):
    st, parts, wd, ref, res, logs, mesh = tie_remote
    _compare(ref, res)
    c = res.info["solver"]["constraints"]
    assert c["tie"] > 100 and c["rigid"] == 2 and res.info["solver"]["backend"] == "ngs_direct"
    assert set(res.step(0).remote) == {"a0"} and res.step(0).remote["a0"]["coupling"] == "rigid"
    # the rigid support holds the point: its displacement and rotation are zero, the reaction balances the loads (force and moment about the point)
    for s in res.steps:
        r = s.remote["a0"]
        assert np.abs(r["u"]).max() == 0.0 and np.abs(r["rot"]).max() == 0.0
    s = res.step(0)
    assert np.allclose(s.remote["a0"]["force"], [0.0, 0.0, 400.0], atol=2e-3)


@needs_chain
def test_a_rigid_load_moves_its_face_as_a_rigid_body_and_a_tie_follows_the_master_face(tie_remote):
    st, parts, wd, ref, res, logs, mesh = tie_remote
    face = mesh.surface_nodes("b1")
    p = np.array([140.0, 12.0, 8.0])
    U = res.step(0).U
    # the face of the rigid load: u = u_ref + θ × (x − p) – fit the six parameters, the residual vanishes
    X = mesh.nodes[face] - p
    rows = np.zeros((3 * len(face), 6))
    for k, x in enumerate(X):
        rows[3 * k:3 * k + 3, :3] = np.eye(3)
        rows[3 * k:3 * k + 3, 3:] = -np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])      # θ × x = −[x]× θ
    q, *_ = np.linalg.lstsq(rows, U[face].ravel(), rcond=None)
    assert np.abs(rows @ q - U[face].ravel()).max() < 1e-12 * np.abs(U).max()
    # the nodes of the slave face of the tie lie on the master face (their displacement is the interpolation of the master nodes)
    from fea_core.ties import tie_weights
    t = mesh.report["ties"][0]
    w = tie_weights(mesh, t["slave"], t["master"], float(t["tolerance_mm"]))
    interp = np.einsum("nk,nkc->nc", w["weights"], U[w["masters"]])
    assert np.abs(interp - U[w["nodes"]]).max() < 1e-12 * np.abs(U).max()


@needs_chain
def test_the_iterative_solver_refuses_what_couples_several_nodes_and_auto_does_not_fall_back_to_it(tie_remote):
    from fea_core.solve_large import LargeSolverError, run_ngs
    st, parts, wd, ref, res, logs, mesh = tie_remote
    with pytest.raises(LargeSolverError, match="direct solver only"):
        run_ngs(st, mesh, wd / "hy", [], lambda s: None, solver="hypre")
    st.solver.backend = "auto"
    assert choose_backend_reason(st, mesh)[0] == "ngs_direct"


# ------------------------------------------------------------------ (3) distributing remote support + rigid remote load
def _study_distributing(wd: Path):
    st, parts = _study("distributing support", wd / "model.step", [Part("Konzola", box(0, 0, 0, 80, 16, 16))], MeshSettings(size=4.0))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("x1", find_faces(parts, x=80))
    st.add_selection("vrch", find_faces(parts, z=16))
    st.constraints.append(RemoteSupport("x0", [-15.0, 8.0, 8.0], [True] * 6, "distributing"))
    st.load_cases += [LoadCase("rigid", [RemoteLoad("x1", [100.0, 8.0, 8.0], [0.0, 50.0, -300.0], [0.0, 1000.0, 400.0], "rigid")]),
                      LoadCase("tlak", [Pressure("vrch", 1.0)])]
    return st, parts


@pytest.fixture(scope="module")
def distributing(tmp_path_factory):
    wd = tmp_path_factory.mktemp("distributing")
    st, parts = _study_distributing(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_a_distributing_remote_support_equals_ccx(distributing):
    st, parts, wd, ref, res, logs, mesh = distributing
    _compare(ref, res, tol_s=1e-4)
    assert res.info["solver"]["constraints"]["distributing"] == 6
    assert res.step(0).remote["x0"]["coupling"] == "distributing"


# ------------------------------------------------------------------ (4) orthotropic materials
ORTHO_A = {"E1": 100000.0, "E2": 20000.0, "E3": 10000.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.4, "G12": 5000.0, "G13": 4000.0, "G23": 3000.0,
           "alpha1": 2e-6, "alpha2": 15e-6, "alpha3": 9e-6}
ORTHO_B = {"E1": 10000.0, "E2": 100000.0, "E3": 10000.0, "nu12": 0.03, "nu13": 0.0, "nu23": 0.0, "G12": 5000.0, "G13": 4000.0, "G23": 4000.0,
           "alpha1": 12e-6, "alpha2": 3e-6, "alpha3": 5e-6}


def _study_ortho(wd: Path):
    ring = BRepAlgoAPI_Cut(cyl((300, 0, 0), (0, 0, 1), 110, 10), cyl((300, 0, 0), (0, 0, 1), 90, 10)).Shape()
    st, parts = _study("ortho", wd / "model.step", [Part("Tyc", box(0, 0, 0, 100, 10, 10)), Part("Prstenec", ring)], MeshSettings(size=8.0))
    th = np.radians(30.0)
    st.coordinate_systems["mat30"] = CoordinateSystem("mat30", "cartesian", [0, 0, 0], [0, 0, 1], [float(np.cos(th)), float(np.sin(th)), 0.0])
    st.coordinate_systems["valec"] = CoordinateSystem("valec", "cylindrical", [300, 0, 0], [0, 0, 1], [1, 0, 0])
    st.add_material(Material("Ortotropni A", ORTHO_A["E1"], ORTHO_A["nu12"], alpha=1e-5, ortho=ORTHO_A, ortho_csys="mat30"), default=True)
    st.add_material(Material("Ortotropni B", ORTHO_B["E1"], ORTHO_B["nu12"], alpha=1e-5, ortho=ORTHO_B, ortho_csys="valec"))
    st.part_materials["Prstenec"] = "Ortotropni B"
    for name, kw in (("x0", dict(part="Tyc", x=0)), ("x1", dict(part="Tyc", x=100)), ("vnitrni", dict(part="Prstenec", radius=90)),
                     ("dno", dict(part="Prstenec", z=0)), ("vnejsi", dict(part="Prstenec", radius=110))):
        st.add_selection(name, find_faces(parts, **kw))
    st.constraints += [Fixed("x0"), Displacement("dno", uz=0.0), CylindricalSupport("vnejsi", radial=False, tangential=True, axial=False)]
    st.load_cases += [LoadCase("tah a tlak", [Pressure("x1", -10.0), Pressure("vnitrni", 10.0)]), LoadCase("teplo", [ThermalLoad(value=80.0)]),
                      LoadCase("tiha", [Gravity([0.0, 0.0, -9810.0])])]
    return st, parts


@pytest.fixture(scope="module")
def orthotropic(tmp_path_factory):
    wd = tmp_path_factory.mktemp("ortho")
    st, parts = _study_ortho(wd)
    for m in st.materials.values():
        m.rho = 7.85e-9
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_orthotropic_materials_rotated_cartesian_and_cylindrical_with_expansion_equal_ccx(orthotropic):
    st, parts, wd, ref, res, logs, mesh = orthotropic
    _compare(ref, res, tol_u=3e-5, tol_s=1e-4)
    sp = res.info["solver"]
    assert sp["backend"] == "ngs_direct" and sp["orthotropic"] is True
    assert np.abs(res.step(1).fields["T"] - 80.0).max() < 1e-9
    # the thermal state of an anisotropic material: the expansion differs by direction (the cylindrical ring: a different radial and tangential strain)
    x = mesh.nodes
    ring = np.hypot(x[:, 0] - 300, x[:, 1]) > 80
    assert np.abs(res.step(1).U[ring]).max() > 0


@needs_chain
def test_auto_runs_orthotropic_and_remote_studies_in_the_direct_solver_through_run_study(tmp_path):
    """The whole `run_study` with the backend auto: the choice, the protocol and the equal results – nothing passed by hand."""
    st, parts = _study_distributing(tmp_path)
    st.solver.backend = "ccx"
    ref = run_study(st, tmp_path / "ccx", parts=parts)
    st.solver.backend = "auto"
    res = run_study(st, tmp_path / "auto", mesh=ref.mesh, parts=parts)
    assert res.info["backend"] == "ngs_direct" and "one factorization" in res.info["solver"]["reason"]
    _compare(ref, res, tol_s=1e-4)


@needs_chain
def test_the_iterative_solver_takes_orthotropic_materials(orthotropic):
    """Orthotropy is node-local (the matrix C of the element): the iterative solver can do it and is not refused (the comment in `choose_backend_reason` once said the
    opposite) – V16 is verified with hypre; here the rotated Cartesian and the cylindrical material of the study against ccx."""
    from fea_core import hypre
    if not hypre.available():
        pytest.skip("the iterative solver (module hypre, MPI) is not installed")
    from fea_core.solve_large import run_ngs, unsupported
    st, parts, wd, ref, res, logs, mesh = orthotropic
    assert unsupported(st, "hypre") == [] and unsupported(st) == []
    st.solver.nproc = 2
    hy = run_ngs(st, mesh, wd / "hypre", [], lambda s: None, solver="hypre")
    assert hy.info["solver"]["backend"] == "ngs_hypre" and hy.info["solver"]["orthotropic"] is True
    _compare(ref, hy, tol_u=5e-5, tol_s=2e-4)


# ------------------------------------------------------------------ (5) point masses: the weight and the inertia of an added mass in the statics
def _study_point_mass(wd: Path, coupling: str):
    from fea_core.dynamics.model import PointMass
    from fea_core.study import Rotation
    st, parts = _study(f"point mass {coupling}", wd / "model.step", [Part("Blok", box(0, 0, 0, 12, 12, 40))], MeshSettings(size=4.0))
    st.add_selection("dno", find_faces(parts, z=0))
    st.add_selection("vrch", find_faces(parts, z=40))
    st.constraints.append(Fixed("dno"))
    J = 5e-3 * 300 / 12
    st.masses.append(PointMass("vrch", 5e-3, point=[6, 6, 55], inertia=[J, 2 * J, 3 * J] if coupling == "rigid" else None, coupling=coupling))
    st.load_cases += [LoadCase("tiha z", [Gravity([0.0, 0.0, -9810.0])]), LoadCase("tiha x", [Gravity([-9810.0, 0.0, 0.0])]),
                      LoadCase("rotace", [Rotation([0.0, 0.0, -100.0], [0.0, 1.0, 0.0], 30.0, 50.0)])]
    return st, parts


@needs_chain
@pytest.mark.parametrize("coupling", ["rigid", "distributing"])
def test_the_weight_and_the_inertia_of_a_point_mass_equal_ccx(tmp_path, coupling):
    st, parts = _study_point_mass(tmp_path, coupling)
    ref, res, logs, mesh = _both(st, parts, tmp_path)
    # ccx averages the nodal stress over the elements of the node – including the auxiliary SPRINGA that `dynamics.mass.write` attaches to the FIRST node of the
    # face of a distributing point mass (its stress is zero → the nodal stress of that one node is 4/5 of the recovered one): that node is left out
    _compare(ref, res, skip_nodes=[int(mesh.surface_nodes("vrch")[0])] if coupling == "distributing" else ())
    assert res.info["solver"]["constraints"]["point_mass"] == 1
    assert (res.info["solver"]["constraints"].get("rigid", 0) == 1) == (coupling == "rigid")
    # the mass acts: a free end load of the weight of 5 kg (about 49 N) bends the post
    assert np.abs(res.step(1).U).max() > 0 and abs(res.step(0).reactions["dno"][2]) > 5e-3 * 9810 * 0.99


# ------------------------------------------------------------------ (6) several holes in ONE cylindrical support, pin supports with a free rotation of the remote point
def _study_two_holes(wd: Path):
    plate = box(0, 0, 0, 60, 30, 10)
    for x in (15.0, 45.0):
        plate = BRepAlgoAPI_Cut(plate, cyl((x, 15, -1), (0, 0, 1), 4, 12)).Shape()
    st, parts = _study("two holes", wd / "model.step", [Part("Deska", solid(plate))], MeshSettings(size=4.0))
    st.materials[st.default_material].alpha = 1.2e-5
    st.add_selection("dirky", find_faces(parts, radius=4))
    st.add_selection("x1", find_faces(parts, x=60))
    st.add_selection("vrch", find_faces(parts, z=10))
    st.constraints.append(CylindricalSupport("dirky", radial=True, tangential=False, axial=True))
    st.load_cases += [LoadCase("sila", [Force("x1", [500.0, 300.0, 0.0])]), LoadCase("tlak a teplo", [Pressure("vrch", 1.0), ThermalLoad(value=50.0)])]
    return st, parts


@needs_chain
def test_several_holes_of_one_cylindrical_support_equal_ccx(tmp_path):
    st, parts = _study_two_holes(tmp_path)
    ref, res, logs, mesh = _both(st, parts, tmp_path)
    _compare(ref, res)
    from fea_core.cylinder import selection_axes
    assert len(selection_axes(mesh, "dirky")) == 2                       # two axes in one selection (the nodes of every hole have their own radial / axial directions)
    assert res.info["solver"]["constraints"]["cylindrical"] > 40


@needs_chain
@pytest.mark.parametrize("coupling", ["rigid", "distributing"])
def test_pin_supports_with_a_free_rotation_of_the_point_equal_ccx(tmp_path, coupling):
    """The remote supports hold only some components: the point moves and turns (u, rot ≠ 0 in `StepResult.remote`), the reaction and the moment about the point
    are measured – the same numbers as ccx (RF of the reference node for the rigid one, the nodes of the equations for the distributing one)."""
    st, parts = _study("pin supports", tmp_path / "model.step", [Part("Nosnik", box(0, 0, 0, 80, 12, 12))], MeshSettings(size=4.0))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("x1", find_faces(parts, x=80))
    st.add_selection("vrch", find_faces(parts, z=12))
    st.constraints += [RemoteSupport("x0", [-10.0, 6.0, 6.0], [True, True, True, True, False, False], coupling),
                       RemoteSupport("x1", [90.0, 6.0, 6.0], [False, True, True, False, False, False], coupling)]
    st.load_cases += [LoadCase("tlak", [Pressure("vrch", 1.0)]), LoadCase("sila", [Force("x1", [300.0, 0.0, -200.0])])]
    ref, res, logs, mesh = _both(st, parts, tmp_path)
    _compare(ref, res, tol_s=1e-4)
    r = res.step(0).remote["x0"]
    assert np.abs(r["rot"]).max() > 1e-5 and np.abs(res.step(1).remote["x1"]["u"]).max() > 1e-4          # the point moves and turns
    assert abs(r["moment"][1]) < 1e-6 * abs(res.step(0).equilibrium["applied_exact"][2]) * 80 + 1e-3         # a free rotation carries no moment (RBE3 / rigid body)


# ------------------------------------------------------------------ (7) a remote support and an elastic support on the SAME face: both backends report the same, complete result
@needs_chain
@pytest.mark.parametrize("coupling", ["rigid", "distributing"])
def test_a_remote_support_with_an_elastic_support_on_its_face_reports_the_same_in_both_backends(tmp_path, coupling):
    """`StepResult.remote` = the reaction of the support at its point (the foundation is NOT part of it – the own path used to add the spring moment to it, 2141 → 2518
    N·mm), the reaction of the selection = support + foundation (the ccx path used to overwrite it with the support alone: the foundation vanished from the table)."""
    plate = box(0, 0, 0, 60, 40, 12)
    st, parts = _study("support + foundation", tmp_path / "model.step", [Part("Deska", plate)], MeshSettings(size=5.0))
    for name, kw in (("x0", dict(x=0)), ("x1", dict(x=60)), ("vrch", dict(z=12))):
        st.add_selection(name, find_faces(parts, **kw))
    st.constraints += [RemoteSupport("x0", [-10.0, 20.0, 6.0], [True, True, True, True, False, False], coupling), ElasticSupport("x0", 20.0, "normal")]
    st.load_cases += [LoadCase("sila", [Force("x1", [-800.0, 300.0, -200.0])]), LoadCase("tlak", [Pressure("x0", 2.0), Pressure("vrch", 1.0)])]
    ref, res, logs, mesh = _both(st, parts, tmp_path)
    _compare(ref, res, tol_s=1e-4)
    from fea_core.springs import reactions as spring_reactions
    x = mesh.nodes
    p = np.array([-10.0, 20.0, 6.0])
    c0 = 0.5 * (x.min(axis=0) + x.max(axis=0))
    for a, b in zip(ref.steps, res.steps):
        fnd = spring_reactions(st, mesh, b.U)["x0"]                                                        # the foundation: −k·u at the nodes of the face
        assert np.abs(fnd).max() > 1e-3                                                                    # (the spring carries something)
        for r in (a, b):                                                                                   # the selection = the support at the point + the foundation
            sup_f = np.asarray(r.remote["x0"]["force"])
            sup_m = np.cross(p - c0, sup_f) + np.asarray(r.remote["x0"]["moment"])
            assert np.allclose(r.reactions["x0"], sup_f + fnd.sum(axis=0), atol=2e-3 * np.abs(sup_f).max() + 1e-6), (r.name, "force")
            fm = np.cross(x - c0, fnd).sum(axis=0)
            assert np.abs(fm).max() > 1e-3 * np.abs(sup_m).max()                                           # a moment of the foundation really is in it
            assert np.allclose(r.reaction_moments["x0"], sup_m + fm, atol=2e-3 * np.abs(sup_m).max() + 1e-6), (r.name, "moment")


# ------------------------------------------------------------------ (8) *TIE whose master nodes are constrained; curved *TIE faces against ccx
def ccx_untied(wd: Path) -> set[int]:
    """The nodes that ccx left free (`gentiedmpc` writes them to `<job>_WarnNodeMissTiedContact.nam`; no file = every node tied), 1-based."""
    out: set[int] = set()
    for f in (wd / "ccx").glob("*WarnNodeMissTiedContact.nam"):
        out |= {int(w) for w in f.read_text().split() if w.isdigit()}
    return out


def _study_pin_in_hole(wd: Path):
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 12), cyl((30, 20, -1), (0, 0, 1), 6, 14)).Shape())
    st, parts = _study("pin in a hole", wd / "model.step", [Part("Deska", plate), Part("Cep", cyl((30, 20, 0), (0, 0, 1), 6, 30))], MeshSettings(size=5.0))
    st.materials[st.default_material].alpha = 1.2e-5
    st.mesh.separate = ["Cep"]
    st.add_selection("x0", find_faces(parts, part="Deska", x=0))
    st.add_selection("top", find_faces(parts, part="Cep", z=30))
    st.add_selection("dno", find_faces(parts, part="Deska", z=0))
    # the rim of the hole at the bottom is a node of the master face of the tie AND of the face with the prescribed displacement
    st.constraints += [Fixed("x0"), Displacement("dno", uz=0.002)]
    st.load_cases += [LoadCase("sila", [Force("top", [300.0, 100.0, -200.0])]), LoadCase("teplo", [ThermalLoad(value=50.0)]),
                      LoadCase("vzdalene", [RemoteLoad("top", [30.0, 20.0, 5000.0], [10.0, 0.0, -50.0], [0.0, 0.0, 2000.0], "rigid")])]
    return st, parts


@pytest.fixture(scope="module")
def pin_in_hole(tmp_path_factory):
    wd = tmp_path_factory.mktemp("pin_in_hole")
    st, parts = _study_pin_in_hole(wd)
    return (st, parts, wd) + _both(st, parts, wd)


@needs_chain
def test_the_force_through_a_tie_into_a_constrained_master_node_is_a_reaction_in_both_backends(pin_in_hole):
    """The master face of the tie (the hole) shares its bottom rim with a face of a prescribed displacement: the force that the pin passes through the tie goes into
    that support. Both backends used to stop with "not in equilibrium" (the force was in no reaction: 0.105 %, and below the tolerance the reaction would have been
    silently too small)."""
    st, parts, wd, ref, res, logs, mesh = pin_in_hole
    _compare(ref, res, tol_u=1e-5, tol_s=1e-4, tol_eq=1e-5)           # (the rigid load 5000 mm away: the moment scale of the check is large)
    for r in (ref, res):
        for s in r.steps:
            applied = np.asarray(s.equilibrium["applied_exact"])
            assert s.equilibrium["imbalance"] < 1e-5, (s.name, s.equilibrium["imbalance"])
            # the total of the reactions (a node shared by two supports once – the table by selections counts the rim of the hole in both) balances the loads
            scale = max(np.abs(v).max() for v in s.reactions.values())
            assert np.allclose(s.equilibrium["reaction_total"], -applied, atol=1e-5 * scale), (s.name, s.equilibrium["reaction_total"], applied)
    assert res.info["solver"]["constraints"]["tie"] > 100


@needs_chain
def test_a_curved_tie_equals_ccx_on_the_same_nodes_and_the_same_untied_set(pin_in_hole):
    """Gate of the curved *TIE: displacements ≤ 1e-5 and stresses ≤ 1e-4 of the maximum against ccx, the same slave nodes without a master face (the pin of a
    perfect fit: the nodes at the ends of the hole)."""
    st, parts, wd, ref, res, logs, mesh = pin_in_hole
    for a, b in zip(ref.steps, res.steps):
        assert _rel(b.U, a.U) < 1e-5, (a.name, "U", _rel(b.U, a.U))
        assert _rel(b.S, a.S) < 1e-4, (a.name, "S", _rel(b.S, a.S))
    from fea_core.ties import mesh_ties
    ours = {int(n) + 1 for w in mesh_ties(mesh) for n in w["untied"]}
    assert ours == ccx_untied(wd) and len(ours) > 0
    assert any("not tied" in w for w in res.warnings)


@needs_chain
def test_a_tied_shell_of_a_cylinder_equals_ccx(tmp_path):
    """Two coaxial tubes, the outer one meshed separately and tied over the whole cylinder (both faces curved)."""
    def tube(r0, r1):
        return solid(BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), r1, 20), cyl((0, 0, -1), (0, 0, 1), r0, 22)).Shape())
    st, parts = _study("tubes", tmp_path / "model.step", [Part("Vnitrni", tube(8, 12)), Part("Vnejsi", tube(12, 16))], MeshSettings(size=4.0))
    st.materials[st.default_material].alpha = 1.2e-5
    st.mesh.separate = ["Vnejsi"]
    st.add_selection("dno", find_faces(parts, part="Vnitrni", z=0))
    st.add_selection("vnejsi", find_faces(parts, part="Vnejsi", radius=16))
    st.add_selection("vrch", find_faces(parts, part="Vnejsi", z=20))
    st.constraints.append(Fixed("dno"))
    st.load_cases += [LoadCase("tlak", [Pressure("vnejsi", 5.0)]), LoadCase("sila", [Force("vrch", [0.0, 2000.0, -500.0])]), LoadCase("teplo", [ThermalLoad(value=40.0)])]
    ref, res, logs, mesh = _both(st, parts, tmp_path)
    for a, b in zip(ref.steps, res.steps):
        assert _rel(b.U, a.U) < 1e-5, (a.name, "U", _rel(b.U, a.U))
        assert _rel(b.S, a.S) < 1e-4, (a.name, "S", _rel(b.S, a.S))
        assert b.equilibrium["imbalance"] < 1e-6
    from fea_core.ties import mesh_ties
    assert {int(n) + 1 for w in mesh_ties(mesh) for n in w["untied"]} == ccx_untied(tmp_path)
    assert res.info["solver"]["constraints"]["tie"] > 200

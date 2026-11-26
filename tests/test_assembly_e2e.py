"""The own assembly end to end through `solve_large.run_ngs` (nothing mocked): a real Gmsh mesh, the job files written by `_run_ngs_job`, the assembly
worker, the direct / iterative solver, the stress recovery and `make_step`, against CalculiX on the SAME mesh.

The study combines what the glue has to carry into the worker: 3D-printing groups (the table of matrices C per element, `element_stiffness_table`),
a prescribed displacement (the vector g of the constraints), a thermal load (ΔT per node and state) and – in a variant of the mesh – a node without
any element (constrained at zero, finite results, equilibrium). A mesh with an inverted element is a mesh error that names the element and the part,
and `auto` does not hide it behind a CalculiX run (ccx rejects it as well: "nonpositive jacobian").
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config, materials as matlib
from fea_core.geometry import Part
from fea_core.mesh import Mesh
from fea_core.printing import PrintSpec
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh, run_study
from fea_core.study import Displacement, Fixed, Force, LoadCase, MeshSettings, ThermalLoad
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba (module accel) is not installed")
needs_chain = pytest.mark.skipif(not (HAS_DIRECT and config.CCX_EXE.exists()),
                                 reason="numba + rslab (modules accel, direct) and CalculiX are needed")

TOP_DISPLACEMENT = 0.02
DT = 60.0 - 20.0                       # the thermal state: 60 °C against the reference temperature 20 °C


@pytest.fixture(autouse=True)
def _rslab_on_the_path_of_the_worker(monkeypatch):
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it
    too (other tests import the installer, which clears PYTHONPATH of the process)."""
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))


def _hypre_ok() -> bool:
    from fea_core import hypre
    return hypre.available()


@pytest.fixture(scope="module")
def block(tmp_path_factory):
    """A printed block 12×12×20 (grid infill, shell + core groups), rollers on three faces, the top face pulled by a prescribed displacement; state 1 = that
    displacement only, state 2 = additionally a uniform temperature of 60 °C. The mesh and the CalculiX result are made once."""
    wd = tmp_path_factory.mktemp("asm_e2e")
    pla = matlib.by_name("Tisk PLA Bambu Basic")
    spec = PrintSpec("Blok", build_dir=[0, 0, 1], layer_height=0.2, line_width=0.5, walls=2, top_layers=3, bottom_layers=3, infill=0.2, pattern="grid")
    st, parts = _study("asm e2e", wd / "model.step", [Part("Blok", box(0, 0, 0, 12, 12, 20))], MeshSettings(size=3.0))
    st.add_material(pla, default=True)
    st.materials.pop("Ocel S235")
    st.printing.parts["Blok"] = spec
    for name, kw in (("x0", dict(x=0)), ("y0", dict(y=0)), ("z0", dict(z=0)), ("vrch", dict(z=20))):
        st.add_selection(name, find_faces(parts, **kw))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0), Displacement("vrch", uz=TOP_DISPLACEMENT)]
    st.load_cases.append(LoadCase("Tah", []))
    st.load_cases.append(LoadCase("Ohrev", [ThermalLoad(value=60.0)]))
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.solver.backend = "ccx"
    ref = run_study(st, wd / "ccx", mesh=mesh, parts=parts)
    return st, parts, mesh, ref, wd


def _run(st, mesh, wd, name, solver, nproc=None):
    from fea_core.solve_large import run_ngs
    old = st.solver.nproc
    if nproc:
        st.solver.nproc = nproc
    try:
        logs: list[str] = []
        return run_ngs(st, mesh, wd / name, [], logs.append, solver=solver), logs
    finally:
        st.solver.nproc = old


def _compare(ref, res, tol_u=2e-5, tol_s=5e-5):
    for a, b in zip(ref.steps, res.steps):
        assert np.abs(a.U - b.U).max() / np.abs(a.U).max() < tol_u, a.name
        assert np.abs(a.S - b.S).max() / np.abs(a.S).max() < tol_s, a.name
        assert b.equilibrium["imbalance"] < 1e-6, a.name
        for k in a.reactions:                                       # the constraint reactions (prescribed displacement and rollers)
            scale = max(np.abs(v).max() for v in a.reactions.values())
            assert np.abs(a.reactions[k] - b.reactions[k]).max() / scale < 1e-4, (a.name, k)


@needs_chain
def test_direct_chain_with_print_groups_prescribed_displacement_and_thermal_load(block, monkeypatch):
    from fea_core.solve_large import element_stiffness_table
    st, parts, mesh, ref, wd = block
    monkeypatch.setenv("FEA_KEEP_SOLVER_FILES", "1")                # to look at what the glue wrote for the worker
    res, logs = _run(st, mesh, wd, "direct", "direct")
    _compare(ref, res)
    # the prescribed displacement is really a prescription (not a load): the top face moves by it in the state 1, in the state 2 with the thermal strain
    top = np.abs(mesh.nodes[:, 2] - 20.0) < 1e-9
    assert np.allclose(res.step(0).U[top, 2], TOP_DISPLACEMENT, atol=1e-12) and np.allclose(res.step(1).U[top, 2], TOP_DISPLACEMENT, atol=1e-12)
    assert res.step(1).fields["T"].min() == pytest.approx(60.0) and res.step(1).umag.max() > 1.5 * res.step(0).umag.max()
    # the job files of the worker: the table of the printing groups (shell / core), the constraints, the temperature change of every state
    job = wd / "direct" / "ngs"
    ti = element_stiffness_table(st, mesh)
    assert ti is not None and len(ti[0]) > 1, "the study has to produce print groups"
    n_groups = len(ti[0]) - len(mesh.part_names)
    assert n_groups >= 2                                             # shell and core (and more by the fraction f of the elements)
    assert np.array_equal(np.load(job / "C_tab.npy"), ti[0]) and np.array_equal(np.load(job / "C_idx.npy"), ti[1])
    assert not np.allclose(np.load(job / "C_tab.npy")[len(mesh.part_names):, 2, 2], np.load(job / "C_tab.npy")[0, 2, 2])      # anisotropic, not the isotropic part
    g, fixed = np.load(job / "g.npy").reshape(-1, 3), np.load(job / "fixed.npy").reshape(-1, 3)
    assert np.allclose(g[top, 2], TOP_DISPLACEMENT) and fixed[top, 2].all()
    assert g[~top].max() == 0.0 and np.abs(g[:, :2]).max() == 0.0
    dT = np.load(job / "dT.npy")
    assert dT.shape == (len(mesh.nodes), 2) and np.allclose(dT[:, 0], 0.0) and np.allclose(dT[:, 1], DT)
    F = np.load(job / "F.npy")
    assert F.shape == (3 * len(mesh.nodes), 2) and np.abs(F).max() == 0.0                    # no mechanical load: all in the prescription and the temperature
    info = res.info["direct"]["assemble"]
    assert info["assembler"] == "own" and info["rigid_body_residual"] < 1e-12 and info["orphan_nodes"] == 0
    assert json.loads(json.dumps(res.info["solver"]))["assembler"].startswith("own")
    assert any("Assembling stiffness matrix" in ln for ln in logs)


@needs_chain
@pytest.mark.skipif(not _hypre_ok(), reason="the iterative solver (module hypre, MPI) is not installed")
def test_iterative_chain_gives_the_same_solution(block):
    st, parts, mesh, ref, wd = block
    res, _ = _run(st, mesh, wd, "hypre", "hypre", nproc=2)
    _compare(ref, res, tol_u=5e-5, tol_s=2e-4)


@needs_chain
def test_a_node_without_an_element_is_constrained_and_leaves_no_trace_in_the_results(block):
    """The mesh of a calculation can contain a node that no element uses: it gets zero displacement, zero stress and nothing breaks (no NaN in the
    stresses, the reactions or the equilibrium that `make_step` evaluates)."""
    st, parts, mesh, ref, wd = block
    n = len(mesh.nodes)
    mesh2 = Mesh(np.vstack([mesh.nodes, [[500.0, 500.0, 500.0]]]), mesh.tets, mesh.tet_part, list(mesh.part_names), dict(mesh.surfaces), dict(mesh.report),
                 None, {k: v.copy() for k, v in mesh.cell_data.items()}, dict(mesh.edges), dict(mesh.points), {})
    res, _ = _run(st, mesh2, wd, "orphan", "direct")
    assert res.info["direct"]["assemble"]["orphan_nodes"] == 1
    for a, b in zip(ref.steps, res.steps):
        assert len(b.U) == n + 1 and np.isfinite(b.U).all() and np.isfinite(b.S).all() and np.isfinite(b.mises).all()
        assert np.all(b.U[-1] == 0.0) and np.all(b.S[-1] == 0.0) and b.equilibrium["imbalance"] < 1e-6
        assert np.abs(a.U - b.U[:n]).max() / np.abs(a.U).max() < 2e-5                              # the same solution as without it
        for k in a.reactions:
            assert np.isfinite(b.reactions[k]).all()


def _inverted(mesh: Mesh, element: int) -> Mesh:
    tets = mesh.tets.copy()
    tets[element] = tets[element][[0, 1, 3, 2, 4, 8, 7, 6, 5, 9]]              # two corners swapped (and their midpoints): the same shape, negative volume
    return replace(mesh, tets=tets, cell_data={k: v.copy() for k, v in mesh.cell_data.items()})


@needs_chain
def test_an_inverted_element_is_a_mesh_error_with_the_element_and_the_part(block):
    from fea_core.solve_large import MeshQualityError
    st, parts, mesh, ref, wd = block
    bad = _inverted(mesh, 17)
    with pytest.raises(MeshQualityError) as e:
        _run(st, bad, wd, "inverted", "direct")
    text = str(e.value)
    assert "18: Blok (" in text and "1 element(s)" in text and "nonpositive jacobian" in text      # CalculiX element number = index + 1
    assert "Traceback" not in text and "assembly of the stiffness matrix failed" not in text


@needs_chain
def test_auto_does_not_hide_a_mesh_error_behind_calculix(block, monkeypatch):
    """CalculiX rejects the same element, so the fallback of `auto` (direct solver fails → CalculiX) must not run for a mesh problem."""
    import fea_core.solve as solve
    from fea_core.solve_large import MeshQualityError
    st, parts, mesh, ref, wd = block
    st2 = type(st).from_dict(st.to_dict())
    st2.solver.backend = "auto"

    def no_ccx(*a, **k):
        raise AssertionError("auto continued with CalculiX after a mesh error")
    monkeypatch.setattr(solve, "_ccx_job", no_ccx)
    with pytest.raises(MeshQualityError, match="nonpositive jacobian"):
        run_study(st2, wd / "auto_bad", mesh=_inverted(mesh, 5), parts=parts)


@needs_numba
def test_valid_curved_gmsh_mesh_and_its_straight_edge_version_pass_the_check(tmp_path):
    """The rule of the check (every Gauss-point determinant positive) accepts what Gmsh makes: a mesh of a cylinder with curved edge midpoints, and the
    straight-edge fallback of the mesher (midpoints in the middle of the chords)."""
    from fea_core.assembly import kernels as K
    from fea_core.assembly.system import check_elements
    from verification.cases import cyl
    st, parts = _study("cyl", tmp_path / "m.step", [Part("Valec", cyl((0, 0, 0), (0, 0, 1), 10.0, 30.0))], MeshSettings(size=3.0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.add_selection("z30", find_faces(parts, z=30))
    st.constraints.append(Fixed("z0"))
    st.load_cases.append(LoadCase("F", [Force("z30", [0.0, 0.0, -100.0])]))
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    nodes, tets = np.ascontiguousarray(mesh.nodes), np.ascontiguousarray(mesh.tets, dtype=np.int32)
    j = K.min_jacobian(nodes, tets, K.DN4)
    assert (j > 0).all()
    curved = np.abs(nodes[tets[:, 4]] - 0.5 * (nodes[tets[:, 0]] + nodes[tets[:, 1]])).max()
    assert curved > 1e-3, "the cylinder mesh should have curved edges"
    check_elements(nodes, tets)
    straight = nodes.copy()
    for m, (a, b) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
        straight[tets[:, 4 + m]] = 0.5 * (nodes[tets[:, a]] + nodes[tets[:, b]])
    assert (K.min_jacobian(straight, tets, K.DN4) > 0).all()
    check_elements(straight, tets)

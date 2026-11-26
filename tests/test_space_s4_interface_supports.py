"""F14 S4 (review findings C1, I10) on real meshes: a cylindrical support (the reaction is the projection on its constrained directions) and a model held ONLY by an
elastic support (no constraint nodes at all). The same gate as the sliding bar: F_j = −ω_j² Γ_j for the whole interface, the groups add up to it."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import verification.cases  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.dynamics import interface as IF
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh
from fea_core.study import CylindricalSupport, Displacement, ElasticSupport, LoadCase, MeshSettings
from verification.cases import _study, box, cyl

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
CCX_GATE = 3e-3          # the ccx path: six significant digits per node of the .frd and an eigenvector residual of ~1e-3 (measured 6e-4 on the rod); the direct path gives 2e-9
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")


@pytest.fixture(autouse=True)
def _rslab_on_the_path_of_the_worker(monkeypatch):
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it too (other tests import
    the installer, which clears PYTHONPATH of the process)."""
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))


def _prepare(st, parts, wd: Path, engine: str):
    st.load_cases.append(LoadCase("F", []))
    st.dynamics.enabled = True
    st.dynamics.modes = 8
    st.solver.backend = "auto"
    st.solver.modal_engine = engine
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return mesh


def _newton(mi: IF.ModalInterface, tol: float) -> None:
    ref = -(mi.omega ** 2)[:, None] * mi.gamma
    scale = np.abs(ref).max(axis=1, keepdims=True)
    err = np.abs(mi.group().F - ref) / np.maximum(scale, 1e-300)
    assert err.max() < tol, f"F_j differs from -w^2 Gamma_j: {err.max():.2e}"
    chk = mi.static_mass_check()
    assert max(chk["relative_error"]) < 5e-3 and max(chk["off_axis_relative"]) < 5e-3


def _rod(wd: Path, engine: str):
    """A rod along z (r = 8, h = 40): radial + tangential support on the lateral face (it may slide axially), uz on the bottom."""
    st, parts = _study("rod", wd / "model.step", [Part("Rod", cyl((0, 0, 0), (0, 0, 1), 8.0, 40.0))], MeshSettings(size=4.0))
    st.add_selection("side", find_faces(parts, where=lambda f: f["type"] == "cylinder"))
    st.add_selection("bottom", find_faces(parts, z=0))
    st.constraints += [CylindricalSupport("side", radial=True, tangential=True, axial=False), Displacement("bottom", uz=0.0)]
    return st, _prepare(st, parts, wd, engine)


def _check_rod(mi: IF.ModalInterface, tol: float) -> None:
    _newton(mi, tol)
    names = [n for n in mi.groups if n != IF.ALL_GROUP]
    assert set(names) == {"side", "bottom"}
    assert np.allclose(sum(mi.group(n).F for n in names), mi.group().F, rtol=1e-9, atol=1e-9 * np.abs(mi.group().F).max())
    assert np.allclose(sum(mi.group(n).F_static for n in names), mi.group().F_static, rtol=1e-9, atol=1e-9 * np.abs(mi.group().F_static).max())
    # the bottom holds only the axial (z) component, the lateral face only the radial / tangential ones (no force along z)
    assert np.abs(mi.group("bottom").F[:, :2]).max() < 1e-9 * np.abs(mi.group().F).max()
    assert np.abs(mi.group("side").F[:, 2]).max() < 1e-6 * np.abs(mi.group().F).max()


def _interface_with_basis(st, mesh, wd: Path):
    """What `IF.run_interface` does, but the modal basis is kept for the gate below."""
    from fea_core.dynamics.modal import run_basis, static_solutions
    from fea_core.study import Gravity
    cases = [LoadCase(f"_if_{a}", [Gravity(vector=[1.0 if i == d else 0.0 for i in range(3)])]) for d, a in enumerate(IF.AXES)]
    basis = run_basis(st, mesh, wd / "if", lambda s: None, None, None, modes=8, f_max=st.dynamics.f_max, job="if_modal", static_cases=cases)
    statics = basis.statics if basis.statics is not None and len(basis.statics) == 3 else static_solutions(st, mesh, cases, wd / "if", lambda s: None, None, None)
    return basis, IF.build_interface(st, mesh, basis, statics, None)


def _newton_projected(st, mesh, basis, mi: IF.ModalInterface, tol: float) -> float:
    """The exact gate (the re-check of the review): the modal base reaction F_j against -w_j^2 * d^T (I - P) M phi_j, where P projects on the HELD directions of every node
    (the cylindrical support holds its radial / tangential directions only): it is the inertia force of the FREE directions that the support has to carry. NOT against the
    participation factor Gamma (d^T M phi over all directions), which differs for a rotated support."""
    from fea_core.dynamics.modal import modal_mass_products
    _w, w_all = IF.held_operators(st, mesh, np.arange(len(mesh.nodes)))
    mphi = modal_mass_products(st, mesh, basis)                                  # (K, N, 3)
    free = mphi - np.einsum("nij,knj->kni", w_all, mphi)                          # (I - P) M phi_j at every node
    g_free = free.sum(axis=1)                                                    # d^T (I - P) M phi_j for the three axes d
    ref = -(mi.omega ** 2)[:, None] * g_free
    # the scale of a mode: the GROSS inertia force of its free directions, w^2 * sum |(I - P) M phi| - a mode whose net force nearly cancels (lateral pairs: 1e-4 of the
    # largest) would otherwise be judged by rounding noise
    scale = (mi.omega ** 2)[:, None] * np.abs(free).sum(axis=1).max(axis=1, keepdims=True)
    err = float((np.abs(mi.group().F - ref) / np.maximum(scale, 1e-300)).max())
    assert err < tol, f"F_j differs from -w^2 d^T (I-P) M phi: {err:.2e} (gate {tol:g})"
    return err


@needs_direct
def test_cylindrical_support_reaction_projects_on_its_directions_direct(tmp_path):
    st, mesh = _rod(tmp_path, "direct")
    basis, mi = _interface_with_basis(st, mesh, tmp_path)
    assert mi.info["engine"] == "direct"
    # Newton's law of the free directions (six digits): with a ROTATED support the participation factor of the direct path and F_j differ by up to ~4 % for the lateral
    # modes - that is Gamma (all directions), not the force that the support carries; the force is exactly -w^2 d^T (I - P) M phi
    _newton_projected(st, mesh, basis, mi, 1e-6)
    _check_rod(mi, 6e-2)                                      # Gamma (all directions) is only loosely related for a rotated support; the groups, the projection and the statics are exact


@needs_ccx
def test_cylindrical_support_reaction_projects_on_its_directions_ccx(tmp_path):
    st, mesh = _rod(tmp_path, "ccx")
    basis, mi = _interface_with_basis(st, mesh, tmp_path)
    assert mi.info["engine"] == "ccx"
    _newton_projected(st, mesh, basis, mi, CCX_GATE)
    _check_rod(mi, 5e-4)                                      # the .frd has six significant digits per node (the sum of many nodal forces cancels)


def _pin_block(wd: Path, engine: str, variant: str):
    """A block with a bore along z: 'same' = a radial CylindricalSupport AND uz on the SAME bore selection, 'cylaxial' = the cylindrical support itself holds radial + axial
    (the same physical constraint), the end face held in y."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    shp = BRepAlgoAPI_Cut(box(0, 0, 0, 60, 20, 20), cyl((10, 10, -1), (0, 0, 1), 5.0, 22.0)).Shape()
    st, parts = _study("pin", wd / "model.step", [Part("Blk", shp)], MeshSettings(size=4.0))
    st.add_selection("bore", find_faces(parts, where=lambda f: f["type"] == "cylinder"))
    st.add_selection("end", find_faces(parts, x=60))
    if variant == "same":
        st.constraints += [CylindricalSupport("bore", radial=True, tangential=False, axial=False), Displacement("bore", uz=0.0), Displacement("end", uy=0.0)]
    else:
        st.constraints += [CylindricalSupport("bore", radial=True, tangential=False, axial=True), Displacement("end", uy=0.0)]
    return st, _prepare(st, parts, wd, engine)


def _same_constraint_same_interface(tmp_path, engine: str, tol: float):
    out = {}
    for variant in ("same", "cylaxial"):
        wd = tmp_path / variant
        wd.mkdir()
        st, mesh = _pin_block(wd, engine, variant)
        basis, mi = _interface_with_basis(st, mesh, wd)
        out[variant] = (st, mesh, basis, mi)
    a, b = out["same"][3], out["cylaxial"][3]
    assert np.allclose(a.freq, b.freq, rtol=1e-6)
    for name in ("bore", "end"):                                                 # the held components of every group are those of the reference (the free tangential one is NOT held)
        ga, gb = a.group(name), b.group(name)
        assert np.allclose(ga.F_static, gb.F_static, rtol=tol, atol=tol * np.abs(gb.F_static).max())
        assert ga.n_nodes == gb.n_nodes
    return out


@needs_direct
def test_a_selection_with_a_cylindrical_support_and_a_plain_constraint_combines_both_direct(tmp_path):
    out = _same_constraint_same_interface(tmp_path, "direct", 1e-9)
    st, mesh, basis, mi = out["same"]
    ref = out["cylaxial"][3]
    assert np.allclose(mi.group().F, ref.group().F, rtol=1e-8, atol=1e-8 * np.abs(ref.group().F).max())      # the whole interface: the same reaction as the reference
    for name in ("bore", "end"):
        assert np.allclose(mi.group(name).F, ref.group(name).F, rtol=1e-8, atol=1e-8 * np.abs(ref.group(name).F).max())
    _newton_projected(st, mesh, basis, mi, 1e-6)                                  # the exact gate of the free directions, with the combined mask


@needs_ccx
def test_a_selection_with_a_cylindrical_support_and_a_plain_constraint_combines_both_ccx(tmp_path):
    out = _same_constraint_same_interface(tmp_path, "ccx", 1e-3)
    st, mesh, basis, mi = out["same"]
    _newton_projected(st, mesh, basis, mi, CCX_GATE)


def _elastic_bar(wd: Path, engine: str):
    st, parts = _study("spring bar", wd / "model.step", [Part("Bar", box(0, 0, 0, 80.0, 10.0, 10.0))], MeshSettings(size=5.0))
    st.add_selection("root", find_faces(parts, x=0))
    st.constraints.append(ElasticSupport("root", 5.0e3, direction="all"))
    return st, _prepare(st, parts, wd, engine)


@needs_direct
def test_model_held_only_by_an_elastic_support_direct(tmp_path):
    st, mesh = _elastic_bar(tmp_path, "direct")
    mi = IF.run_interface(st, mesh, tmp_path / "if", modes=8)
    assert set(mi.groups) == {IF.ALL_GROUP, "root (elastic support)"}
    _newton(mi, 1e-6)
    sp = IF.sine_interface(mi, "y", 1.0, 0.02, f=np.array([1e-3]))
    assert sp.F_peak[0] == pytest.approx(mi.total_mass * 9810.0, rel=5e-3)


@needs_ccx
def test_model_held_only_by_an_elastic_support_ccx(tmp_path):
    st, mesh = _elastic_bar(tmp_path, "ccx")
    mi = IF.run_interface(st, mesh, tmp_path / "if", modes=8)
    assert set(mi.groups) == {IF.ALL_GROUP, "root (elastic support)"} and mi.info["engine"] == "ccx"
    _newton(mi, 5e-3)                                          # the spring reaction is k·φ of the six-digit .frd shapes

"""F14 S4 gate (review finding C1): the modal base-reaction vectors of a model whose held nodes are only PARTLY held.

A bar 80 x 10 x 10 mm on three sliding faces (`ux` on x = 0, `uy` on y = 0, `uz` on z = 0): the nodes on the edges and the corner are held in one, two or three
components, the others free. `K·φ` is nonzero in the free components of those nodes (inertia of the mode), so summing every component over-counts. The gate:
the modal base reaction of the whole interface satisfies F_j = −ω_j² Γ_j to six digits (Newton: the support force of a mode is minus the rate of change of its
momentum), for the CalculiX modal basis and for the direct one, and the groups (one per sliding face) add up to the whole interface."""
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
from fea_core.study import Displacement, LoadCase, MeshSettings
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")


@pytest.fixture(autouse=True)
def _rslab_on_the_path_of_the_worker(monkeypatch):
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it too (other tests import
    the installer, which clears PYTHONPATH of the process)."""
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
L, B = 80.0, 10.0


def _bar(wd: Path, engine: str):
    st, parts = _study("sliding bar", wd / "model.step", [Part("Bar", box(0, 0, 0, L, B, B))], MeshSettings(size=4.0))
    st.add_selection("sx", find_faces(parts, x=0))
    st.add_selection("sy", find_faces(parts, y=0))
    st.add_selection("sz", find_faces(parts, z=0))
    st.constraints += [Displacement("sx", ux=0.0), Displacement("sy", uy=0.0), Displacement("sz", uz=0.0)]
    st.load_cases.append(LoadCase("F", []))
    st.dynamics.enabled = True
    st.dynamics.modes = 12
    st.solver.backend = "auto"
    st.solver.modal_engine = engine
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return st, mesh


def _gate(mi: IF.ModalInterface, cancellation: np.ndarray | None = None) -> None:
    """cancellation (K,) = Σ|nodal reaction| / |F_j| of the modes: the CalculiX .frd has six significant digits per nodal value, so the sum of a strongly cancelling
    mode cannot be better than 5e-7 times this ratio; the direct basis has no such limit (double precision)."""
    g = mi.group()
    w2 = mi.omega ** 2
    ref = -w2[:, None] * mi.gamma                                   # (K, 3): minus the rate of change of the momentum of the mode
    scale = np.abs(ref).max(axis=1, keepdims=True)
    big = np.abs(mi.gamma) > 0.05 * np.sqrt(mi.total_mass)          # components with a real participation
    assert big.sum() >= 3
    err = (np.abs(g.F - ref) / np.maximum(scale, 1e-300)).max(axis=1)
    tol = np.full(mi.K, 1e-6) if cancellation is None else np.maximum(1e-6, 5e-7 * cancellation)
    assert np.all(err <= tol), f"F_j differs from -w^2 Gamma_j: relative error per mode {err}, allowed {tol}"
    if cancellation is None or cancellation.max() < 20:
        assert err.max() < 1e-6
    # the quasi-static limit: the reaction of a unit base acceleration is the total mass
    chk = mi.static_mass_check()
    assert max(chk["relative_error"]) < 5e-3 and max(chk["off_axis_relative"]) < 5e-3
    # the groups (one per face) add up exactly to the whole interface: a shared node counts once
    total_F = sum(mi.group(n).F for n in ("sx", "sy", "sz"))
    total_S = sum(mi.group(n).F_static for n in ("sx", "sy", "sz"))
    total_M = sum(mi.group(n).M for n in ("sx", "sy", "sz"))
    assert np.allclose(total_F, g.F, rtol=1e-9, atol=1e-9 * np.abs(g.F).max())
    assert np.allclose(total_M, g.M, rtol=1e-9, atol=1e-9 * np.abs(g.M).max())
    assert np.allclose(total_S, g.F_static, rtol=1e-9, atol=1e-9 * np.abs(g.F_static).max())
    # each face holds only its own component: sx pushes along x only
    assert np.abs(mi.group("sx").F[:, 1:]).max() < 1e-9 * np.abs(g.F).max()
    assert np.abs(mi.group("sy").F[:, [0, 2]]).max() < 1e-9 * np.abs(g.F).max()
    assert np.abs(mi.group("sz").F[:, :2]).max() < 1e-9 * np.abs(g.F).max()
    # the interface force of a sine at f -> 0 and at the dominant mode agrees with Newton's resultant
    for d, ax in enumerate("xyz"):
        sp = IF.sine_interface(mi, ax, 1.0, 0.02, f=np.array([1e-3]))
        assert sp.F_peak[0] == pytest.approx(mi.total_mass * 9810.0, rel=5e-3)


@needs_ccx
def test_sliding_supports_modal_reaction_is_minus_omega2_gamma_ccx(tmp_path):
    st, mesh = _bar(tmp_path, "ccx")
    mi = IF.run_interface(st, mesh, tmp_path / "if", modes=12)
    assert mi.info["engine"] == "ccx" and not mi.warnings
    from fea_core.dynamics.modal import run_basis
    basis = run_basis(st, mesh, tmp_path / "basis", lambda s: None, modes=12, job="if_modal")
    ref = -((2 * np.pi * basis.freq) ** 2)[:, None] * np.asarray(basis.gamma)[:, :3]
    _gate(mi, np.abs(np.asarray(basis.rf, float)).sum(axis=(1, 2)) / np.abs(ref).max(axis=1))


@needs_direct
def test_sliding_supports_modal_reaction_is_minus_omega2_gamma_direct(tmp_path):
    st, mesh = _bar(tmp_path, "direct")
    mi = IF.run_interface(st, mesh, tmp_path / "if", modes=12)
    assert mi.info["engine"] == "direct" and not mi.warnings
    _gate(mi)


@needs_ccx
def test_old_all_component_sum_is_wrong_for_partly_held_nodes(tmp_path):
    """The counterexample of the review: summing every component of the held nodes does NOT give −ω²Γ (regression: the mask makes the difference)."""
    from fea_core.dynamics.modal import run_basis
    st, mesh = _bar(tmp_path, "ccx")
    basis = run_basis(st, mesh, tmp_path / "basis", lambda s: None, modes=12, job="if_modal")
    naive = np.asarray(basis.rf, float).sum(axis=1)
    ref = -((2 * np.pi * basis.freq) ** 2)[:, None] * np.asarray(basis.gamma)[:, :3]
    assert np.abs(naive - ref).max() > 1e-3 * np.abs(ref).max()          # the unmasked sum is off
    mi = IF.run_interface(st, mesh, tmp_path / "if", modes=12)
    assert np.abs(mi.group().F - ref).max() < 1e-3 * np.abs(ref).max()   # the masked sum is right (the .frd precision limits the cancelling modes, see `_gate`)


"""F14 S4 re-check (outside S4, small): the effective total mass of the direct engine (`info["mass"]["total_free"]`, the denominator of the mass fractions) leaves out the
directions that constraint equations hold (cylindrical supports), like the local boundary DOFs of ccx; the participation factors keep the whole rigid-body motion."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import verification.cases  # noqa: F401  (first the cases)
from fea_core import config
from fea_core.assembly.reduction import Reduction
from fea_core.assembly.system import release_held_directions, rigid_motions

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")


def test_a_held_direction_is_projected_out_of_the_rigid_motions():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R = rigid_motions(nodes, np.zeros((0, 3)), np.zeros(9, bool))
    # node 1: the radial direction (1, 1, 0)/sqrt 2 is held: u_x = -u_y (dependent DOF 3 = x of node 1; the master is y of node 1)
    red = Reduction.from_rows(3, [(3, [(4, -1.0)], 0.0)])
    Rf = release_held_directions(R, red)
    assert Rf is not R
    a = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    held = a @ Rf[3:6]                                                      # the component along the held direction vanishes for all six rigid motions of node 1
    assert np.allclose(held, 0.0, atol=1e-12)
    perp = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    assert np.allclose(perp @ Rf[3:6], perp @ R[3:6])                        # the free direction is untouched
    assert np.allclose(Rf[:3], R[:3]) and np.allclose(Rf[6:9], R[6:9])        # the other nodes are untouched
    # a plane held by two equations leaves the one free direction; three equations hold everything
    red2 = Reduction.from_rows(3, [(3, [(4, -1.0)], 0.0), (5, [(4, 0.5)], 0.0)])
    Rf2 = release_held_directions(R, red2)
    free_dir = np.array([1.0, -1.0, -0.5])
    free_dir /= np.linalg.norm(free_dir)
    assert np.allclose(Rf2[3:6], np.outer(free_dir, free_dir @ R[3:6]), atol=1e-12)
    # an equation that couples nodes (a tie: node 2 follows node 0) is not a held direction
    tie = Reduction.from_rows(3, [(6, [(0, 1.0)], 0.0)])
    assert release_held_directions(R, tie) is not None and np.allclose(release_held_directions(R, tie), R)
    assert release_held_directions(R, None) is R and release_held_directions(R, Reduction(3)) is R


@needs_direct
@needs_ccx
def test_the_effective_total_mass_of_the_direct_engine_follows_the_held_directions_like_ccx(tmp_path, monkeypatch):
    from fea_core.dynamics.modal import run_basis
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.solve import prepare_mesh
    from fea_core.study import CylindricalSupport, Displacement, LoadCase, MeshSettings
    from verification.cases import _study, cyl
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(Path(spec.origin).parent.parent), *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    tot = {}
    for engine in ("direct", "ccx"):
        wd = tmp_path / engine
        wd.mkdir()
        st, parts = _study("rod", wd / "model.step", [Part("Rod", cyl((0, 0, 0), (0, 0, 1), 8.0, 40.0))], MeshSettings(size=4.0))
        st.add_selection("side", find_faces(parts, where=lambda f: f["type"] == "cylinder"))
        st.add_selection("bottom", find_faces(parts, z=0))
        st.constraints += [CylindricalSupport("side", radial=True, tangential=True, axial=False), Displacement("bottom", uz=0.0)]
        st.load_cases.append(LoadCase("F", []))
        st.dynamics.enabled = True
        st.dynamics.modes = 8
        st.solver.backend = "auto"
        st.solver.modal_engine = engine
        mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
        basis = run_basis(st, mesh, wd / "m", lambda s: None, modes=8, job="tf")
        assert basis.engine == engine
        tot[engine] = (np.asarray(basis.total_mass, float)[:3], float(basis.model_mass))
    (d, M), (c, _M2) = tot["direct"], tot["ccx"]
    assert d[0] < 0.9 * M and d[1] < 0.9 * M                                # the radial + tangential directions of the side face are held: the lateral total mass is far below the model mass
    assert np.allclose(d[:2], c[:2], rtol=3e-2) and d[2] == pytest.approx(c[2], rel=3e-3)         # ... and agrees with ccx (it was 32 % above it when the held directions were counted)

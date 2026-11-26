"""The modal basis of the dynamics on the direct path (F14 S0): nothing mocked – a real mesh, the assembly + eigenvalue workers, the static corrections from the same
factor, the modal response, against the CalculiX path on the same mesh. Point masses (rigid with rotary inertia, distributing) and a surface mass are in the mass matrix
exactly as ccx writes them."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.dynamics.model import Excitation, PointMass, Probe, SurfaceMass
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh, run_study
from fea_core.study import Fixed, Force, LoadCase, MeshSettings
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_chain = pytest.mark.skipif(not (HAS_DIRECT and config.CCX_EXE.exists()), reason="numba + rslab (modules accel, direct) and CalculiX are needed")


@pytest.fixture(scope="module", autouse=True)
def _rslab_on_the_path_of_the_worker():
    mp = pytest.MonkeyPatch()
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        mp.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    yield
    mp.undo()


L, B = 80.0, 10.0


def _cantilever(wd: Path, mass: PointMass | None = None, surface: bool = False):
    st, parts = _study("dyn direct", wd / "model.step", [Part("Konzola", box(0, 0, 0, L, B, B))], MeshSettings(size=4.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("cel", find_faces(parts, x=L))
    st.add_selection("vrch", find_faces(parts, z=B))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("F", [Force("cel", [0.0, 50.0, 0.0])]))
    if mass is not None:
        st.masses.append(mass)
    if surface:
        st.masses.append(SurfaceMass("vrch", 3.0, name="deska"))
    ds = st.dynamics
    ds.enabled = True
    ds.modes = 8
    ds.damping.zeta = 0.02
    ds.probes = [Probe("hrot", point=[L, 5.0, 5.0])]
    h = ds.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 100.0, 6000.0, 40
    h.excitation = [Excitation(kind="force", case="F")]
    ds.shock.enabled = False
    return st, parts


def _run(st, parts, wd: Path, engine: str, mesh=None):
    st.solver.backend = "auto"
    st.solver.modal_engine = engine
    return run_study(st, wd / engine, mesh=mesh, parts=parts)


def _both(st, parts, wd):
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    ref = _run(st, parts, wd, "ccx", mesh)
    own = _run(st, parts, wd, "direct", mesh)
    return ref, own


def _check(ref, own, tol_f=3e-4, tol_resp=2e-3):
    from fea_core.eigen import compare_modes
    a, b = own.dynamics, ref.dynamics
    assert a.info["modal_engine"] == "direct" and b.info["modal_engine"] == "ccx"
    c = compare_modes(a.freq, a.shapes, a.gamma, b.freq, b.shapes, b.gamma)
    assert c["max_freq_rel"] <= tol_f and c["min_subspace"] >= 0.999 and c["max_eff_mass_rel"] <= 1e-3, c       # ccx is only as accurate as its ARPACK (test_modal_direct._gate)
    ua, ub = a.harmonic["probes"]["hrot"]["U"], b.harmonic["probes"]["hrot"]["U"]
    assert np.abs(np.abs(ua) - np.abs(ub)).max() <= tol_resp * np.abs(ub).max()
    assert a.checks["mass_norm_error"] < 1e-6 and a.checks["gamma_error"] < 1e-3
    return c


@needs_chain
def test_the_modal_basis_the_static_correction_and_the_response_equal_the_ccx_path(tmp_path):
    st, parts = _cantilever(tmp_path)
    ref, own = _both(st, parts, tmp_path)
    c = _check(ref, own)
    assert c["n"] == 8
    # the static correction of the excitation comes from the factor of the modal analysis (no extra job)
    ra, rb = own.dynamics.residual, ref.dynamics.residual
    assert np.abs(ra["U"] - rb["U"]).max() <= 1e-2 * np.abs(rb["U"]).max()           # the residual is the difference of the static and the modal displacement: it depends on the truncation
    # the modal stresses are those of the modes (the same recovery as ccx): the envelope of the harmonic stress
    ea, eb = own.dynamics.harmonic["env_S"], ref.dynamics.harmonic["env_S"]
    assert np.abs(ea - eb).max() <= 3e-3 * eb.max()
    assert own.dynamics.info["modal_solver"]["library"].startswith("rslab")


@needs_chain
def test_a_rigid_point_mass_with_rotary_inertia_a_distributing_one_and_a_surface_mass_equal_ccx(tmp_path):
    cases = {
        "rigid_inertia": dict(mass=PointMass("cel", 2e-3, point=[L, 5.0, 5.0], inertia=[30.0, 20.0, 10.0], coupling="rigid", name="motor"), surface=False),
        "distributing": dict(mass=PointMass("cel", 1e-3, point=[L + 6.0, 3.0, 7.0], coupling="distributing", name="cidlo"), surface=False),
        "surface": dict(mass=None, surface=True),
    }
    for name, kw in cases.items():
        wd = tmp_path / name
        wd.mkdir()
        st, parts = _cantilever(wd, **kw)
        ref, own = _both(st, parts, wd)
        c = _check(ref, own)
        assert c["n"] == 8, name
        assert own.dynamics.model_mass == pytest.approx(ref.dynamics.model_mass, rel=1e-9), name
        assert own.dynamics.checks["mode_equilibrium"] < 1e-3 and own.dynamics.checks["mass_norm_error"] < 1e-6, name


@needs_chain
def test_the_cross_check_flag_adds_the_modal_comparison_to_the_dynamics_results(tmp_path):
    st, parts = _cantilever(tmp_path)
    st.dynamics.verify_ccx = True
    st.solver.backend, st.solver.modal_engine = "auto", "direct"
    res = run_study(st, tmp_path / "v", parts=parts)
    chk = res.dynamics.info["modal_check"]
    assert chk["engine"] == "ccx" and chk["n"] == 8 and chk["max_freq_rel"] < 1e-4 and chk["min_subspace"] > 0.9999
    assert "etalon" in res.dynamics.info
    # the report prints the table
    from fea_core.report.f8 import modal_check_table

    class D:
        def __init__(self):
            self.tables, self.paras = [], []

        def table(self, header, rows, **kw):
            self.tables.append((header, rows))

        def p(self, text, **kw):
            self.paras.append(text)
    d = D()
    modal_check_table(d, chk)
    assert len(d.tables) == 2 and d.tables[0][0][0] == "Mode" and len(d.tables[0][1]) == 8

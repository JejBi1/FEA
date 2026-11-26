"""F14 S4 verification gate on a real mesh: nothing mocked – a tetrahedral cantilever (Gmsh), the modal basis of CalculiX (the direct engine gives the same data
where the rslab module is installed), the static reactions of the unit base accelerations, then the interface forces. Gate: the quasi-static limit of the sine
interface reaction equals M_total · a_base to 0.5 %, the moment about the reference point is M · a · lever arm, the reaction of a body with an added point mass too."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import verification.cases  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.dynamics import interface as IF
from fea_core.dynamics.model import PointMass
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh
from fea_core.study import Fixed, LoadCase, MeshSettings
from verification.cases import _study, box

pytestmark = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
L, B, G = 80.0, 10.0, 9810.0


def _model(wd: Path, point_mass: float = 0.0):
    st, parts = _study("interface forces", wd / "model.step", [Part("Konzola", box(0, 0, 0, L, B, B))], MeshSettings(size=4.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("cel", find_faces(parts, x=L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("F", []))
    if point_mass:
        st.masses.append(PointMass("cel", point_mass, name="pm"))
    st.dynamics.enabled = True
    st.dynamics.modes = 12
    st.solver.backend = "auto"
    st.solver.modal_engine = "ccx"
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return st, mesh


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    wd = tmp_path_factory.mktemp("if_plain")
    st, mesh = _model(wd)
    return st, mesh, IF.run_interface(st, mesh, wd / "if", modes=12)


@pytest.fixture(scope="module")
def with_mass(tmp_path_factory):
    wd = tmp_path_factory.mktemp("if_mass")
    st, mesh = _model(wd, point_mass=2.0e-3)
    return st, mesh, IF.run_interface(st, mesh, wd / "if", modes=12)


def test_quasi_static_limit_equals_total_mass_times_base_acceleration(plain):
    _, mesh, mi = plain
    assert mi.total_mass > 0 and not mi.warnings
    chk = mi.static_mass_check()
    assert max(chk["relative_error"]) < 5e-3 and max(chk["off_axis_relative"]) < 5e-3
    for d in "xyz":
        sp = IF.sine_interface(mi, d, 1.0, 0.02, f=np.array([1e-3, 1e-2]))                 # 1 g, f → 0
        assert abs(sp.F_peak[0] - mi.total_mass * G) / (mi.total_mass * G) < 5e-3, d      # the gate: < 0.5 %
    assert mi.total_mass == pytest.approx(L * B * B * 7.85e-9, rel=1e-3)               # the model mass is the mass of the body


def test_quasi_static_limit_also_with_a_truncated_basis(plain):
    st, mesh, _ = plain
    mi = plain[2]
    # keep only the first mode: the static residual makes the f → 0 limit independent of the number of modes
    k = 1
    from fea_core.dynamics.interface import InterfaceGroup, ModalInterface
    g = mi.group()
    tr = ModalInterface(mi.freq[:k], mi.gamma[:k], mi.eff_mass[:k], mi.total_mass, mi.ref_point,
                        {IF.ALL_GROUP: InterfaceGroup(IF.ALL_GROUP, g.F[:k], g.M[:k], g.F_static, g.M_static, g.n_nodes)})
    for d in "xyz":
        sp = IF.sine_interface(tr, d, 1.0, 0.02, f=np.array([1e-3]))
        assert abs(sp.F_peak[0] - tr.total_mass * G) / (tr.total_mass * G) < 5e-3


def test_moment_about_the_reference_point_is_mass_acceleration_lever_arm(plain):
    _, mesh, mi = plain
    # the reference point = the centroid of the held nodes (x = 0, y = z = B/2): the centre of mass of the bar is L/2 along x
    assert mi.ref_point == pytest.approx([0.0, B / 2, B / 2], abs=0.05)
    sp = IF.sine_interface(mi, "y", 1.0, 0.02, f=np.array([1e-3]))
    lever = L / 2
    assert sp.M_peak[0] == pytest.approx(mi.total_mass * G * lever, rel=5e-3)
    spx = IF.sine_interface(mi, "x", 1.0, 0.02, f=np.array([1e-3]))
    assert spx.M_peak[0] < 1e-3 * mi.total_mass * G * L                                # an axial acceleration along the lever arm: no moment


def test_point_mass_adds_to_the_reaction_and_the_moment(plain, with_mass):
    _, _, a = plain
    _, _, b = with_mass
    assert b.total_mass == pytest.approx(a.total_mass + 2.0e-3, rel=1e-9)
    sp = IF.sine_interface(b, "y", 1.0, 0.02, f=np.array([1e-3]))
    assert sp.F_peak[0] == pytest.approx(b.total_mass * G, rel=5e-3)
    # moment about the root: the bar (L/2) and the point mass at the free end (a point mass on the face x = L)
    expect = (a.total_mass * L / 2 + 2.0e-3 * L) * G
    assert sp.M_peak[0] == pytest.approx(expect, rel=2e-2)
    assert b.freq[0] < a.freq[0]                                                         # the added mass lowers the first frequency


def test_spectra_and_random_rms_are_consistent(plain):
    _, _, mi = plain
    sp = IF.sine_interface(mi, "y", [[10.0, 1.0], [5000.0, 1.0]], 0.02)
    assert np.all(np.isfinite(sp.F_peak)) and sp.F_peak.max() > mi.total_mass * G           # the cantilever resonance amplifies the interface force
    f_peak = float(sp.f[int(np.argmax(sp.F_peak))])
    assert f_peak == pytest.approx(mi.freq[0], rel=0.02)
    r = IF.random_interface(mi, "y", [[20.0, 0.01], [2000.0, 0.01]], 0.02)
    assert r.F_rms_total > 0 and r.F_rms[1] > 0.9 * r.F_rms_total                          # a lateral base motion: the force is lateral
    assert r.summary()["F_nsigma_total_N"] == pytest.approx(3.0 * r.F_rms_total)
    # the groups: the only constraint is the whole interface
    assert set(mi.groups) == {IF.ALL_GROUP, "vetknuti"}
    assert np.allclose(mi.group("vetknuti").F_static, mi.group().F_static)
    rows = mi.modal_table()
    assert len(rows) == mi.K and rows[0]["f_Hz"] == pytest.approx(mi.freq[0])

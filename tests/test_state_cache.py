"""The state cache (`fea_core.state_cache`, `fea_core.gravity_states`, `fea_core.direct.modal_cache`): what several analyses of ONE study on ONE mesh share.

* the modal basis: ONE eigenvalue analysis serves the natural frequencies, the dynamics, the 3-axis analysis and the interface forces (the first modes of a larger analysis equal a
  smaller analysis to the tolerance of the eigenvalue solver – the gate S2F 1e-9);
* the unit-gravity states: every gravity case of a linear study (the QSL cases, the unit base accelerations, the unit-gravity statics of the model checks) is a combination of
  three unit states – the results equal the solved ones to round-off (the direct path) / to the 6 digits of the CalculiX output (ccx), the equilibrium check runs on the
  combined state.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config, gravity_states as GS, state_cache
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh, run_study
from fea_core.study import Displacement, Fixed, Gravity, LoadCase, MeshSettings
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
G = 9810.0


@pytest.fixture(scope="module", autouse=True)
def _rslab_on_the_path_of_the_worker():
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it too."""
    mp = pytest.MonkeyPatch()
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        mp.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    yield
    mp.undo()


def _beam(wd: Path, name: str = "beam"):
    st, parts = _study(name, wd / "model.step", [Part("Nosnik", box(0, 0, 0, 120, 20, 10))], MeshSettings(size=5.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.constraints.append(Fixed("vetknuti"))
    return st, parts


def _gravity_cases(st, scale: float = 3.0) -> None:
    """Six signed axis cases and two combinations (the shape of the QSL set)."""
    for ax in range(3):
        for s in (1, -1):
            v = [0.0, 0.0, 0.0]
            v[ax] = s * G * scale
            st.load_cases.append(LoadCase(f"G{'xyz'[ax]}{'+' if s > 0 else '-'}", [Gravity(vector=v)]))
    st.load_cases.append(LoadCase("Gxz", [Gravity(vector=[scale * G, 0.0, -2 * G])]))
    st.load_cases.append(LoadCase("Gxyz", [Gravity(vector=[-G, 2 * G, 0.5 * G])]))


@pytest.fixture(scope="module")
def beam(tmp_path_factory):
    wd = tmp_path_factory.mktemp("state_cache_beam")
    st, parts = _beam(wd)
    _gravity_cases(st)
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return st, mesh, wd


# ----------------------------------------------------------------------------------------------------------------------------------------- keys
def test_the_digest_covers_what_changes_a_linear_state_and_nothing_else(beam):
    st, mesh, _ = beam
    base = state_cache.study_digest(st, mesh)
    import copy
    other = copy.deepcopy(st)
    other.name = "another name"
    other.solver.nproc = 7
    other.load_cases = other.load_cases[:2]
    other.modal.enabled, other.modal.modes = True, 17
    other.dynamics.enabled, other.dynamics.modes = True, 23
    assert state_cache.study_digest(other, mesh) == base                    # the name, the threads, the load cases and the eigenvalue settings are not part of a state
    changed = copy.deepcopy(st)
    mat = changed.materials[changed.default_material]
    mat.E = mat.E * 1.001
    assert state_cache.study_digest(changed, mesh) != base                  # a material is
    more = copy.deepcopy(st)
    more.constraints.append(Fixed("vetknuti"))
    assert state_cache.study_digest(more, mesh) != base                     # so is a constraint
    from fea_core.study import RemoteLoad
    soft = copy.deepcopy(st)
    soft.load_cases.append(LoadCase("remote soft", [RemoteLoad("vetknuti", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], coupling="distributing")]))
    hard = copy.deepcopy(st)
    hard.load_cases.append(LoadCase("remote rigid", [RemoteLoad("vetknuti", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], coupling="rigid")]))
    assert state_cache.study_digest(soft, mesh) == base                     # a distributing remote load does not stiffen the model
    assert state_cache.study_digest(hard, mesh) != base                     # a rigid one adds a rigid body to every state
    moved = copy.copy(mesh)
    moved.nodes = np.array(mesh.nodes) + 1e-6
    assert state_cache.study_digest(st, moved) != base                      # and the mesh


def test_a_load_case_is_pure_gravity_only_when_it_is_one_global_gravity_load():
    assert np.allclose(GS.pure_gravity(LoadCase("a", [Gravity(vector=[1.0, 0.0, -2.0])])), [1.0, 0.0, -2.0])
    assert GS.pure_gravity(LoadCase("b", [Gravity(vector=[0, 0, -G], csys="local")])) is None
    assert GS.pure_gravity(LoadCase("c", [Gravity(vector=[0, 0, -G]), Gravity(vector=[G, 0, 0])])) is None
    assert GS.pure_gravity(LoadCase("d", [])) is None


def test_a_prescribed_displacement_or_a_nonlinear_study_is_not_superposed(beam):
    import copy
    st, mesh, _ = beam
    assert GS.superposable(st, mesh) is None
    pre = copy.deepcopy(st)
    pre.add_selection("end", find_faces([Part("Nosnik", box(0, 0, 0, 120, 20, 10))], x=120))
    pre.constraints.append(Displacement("end", uz=0.1))
    assert "prescribed" in (GS.superposable(pre, mesh) or "")
    nl = copy.deepcopy(st)
    nl.contact_settings.nlgeom = True
    assert GS.superposable(nl, mesh) is not None
    assert GS.plan(nl, mesh, "ccx") is None


# ----------------------------------------------------------------------------------------------------------------------------------------- unit-gravity states
def _run(st, mesh, wd, tag, env):
    os.environ["FEA_STATE_CACHE"] = env
    try:
        state_cache.clear()
        return run_study(st, wd / tag, study_dir=wd, on_log=lambda s: None, mesh=mesh)
    finally:
        os.environ.pop("FEA_STATE_CACHE", None)


def _same(a, b, rel_u, rel_s):
    assert [s.name for s in a.steps] == [s.name for s in b.steps]
    for sa, sb in zip(a.steps, b.steps):
        for x, y, rel in ((sa.U, sb.U, rel_u), (sa.S, sb.S, rel_s)):
            assert np.abs(x - y).max() <= rel * max(np.abs(y).max(), 1e-30), sa.name
        for sel, r in sb.reactions.items():
            assert np.abs(sa.reactions[sel] - r).max() <= rel_s * max(np.abs(r).max(), 1e-30), (sa.name, sel)
        assert sa.equilibrium["imbalance"] <= config.EQUILIBRIUM_TOL and sb.equilibrium["imbalance"] <= config.EQUILIBRIUM_TOL


@needs_direct
def test_gravity_cases_from_three_unit_states_equal_the_solved_ones_on_the_direct_path(beam):
    st, mesh, wd = beam
    st.solver.backend = "auto"
    sup = _run(st, mesh, wd, "sup_direct", "1")
    ref = _run(st, mesh, wd, "ref_direct", "0")
    assert sup.info["backend"] == "ngs_direct"
    info = sup.info["gravity_superposition"]
    assert info["unit_states"] == ["x", "y", "z"] and info["solved"] == ["x", "y", "z"] and len(info["cases"]) == len(st.load_cases)
    assert "gravity_superposition" not in ref.info
    _same(sup, ref, 1e-10, 1e-10)


@needs_ccx
def test_gravity_cases_from_three_unit_states_equal_the_solved_ones_in_ccx(beam):
    import copy
    st, mesh, wd = beam
    c = copy.deepcopy(st)
    c.solver.backend = "ccx"
    sup = _run(c, mesh, wd, "sup_ccx", "1")
    ref = _run(c, mesh, wd, "ref_ccx", "0")
    assert sup.info["backend"] == "ccx" and sup.info["gravity_superposition"]["solved"] == ["x", "y", "z"]
    _same(sup, ref, 1e-5, 1e-4)                                              # the .frd has 6 significant digits: a combination of rounded unit states


@needs_ccx
def test_the_stored_unit_states_serve_the_unit_gravity_statics_of_the_checks_without_a_solve(beam, monkeypatch):
    import copy
    st, mesh, wd = beam
    c = copy.deepcopy(st)
    c.solver.backend = "ccx"
    state_cache.clear()
    first = run_study(c, wd / "serve_1", study_dir=wd, on_log=lambda s: None, mesh=mesh)          # stores the three unit states
    assert first.info["gravity_superposition"]["solved"] == ["x", "y", "z"]
    unit = copy.deepcopy(c)
    unit.load_cases = [LoadCase(f"Unit gravity {ax}", [Gravity(vector=[G * (i == k) for i in range(3)])]) for k, ax in enumerate("XYZ")]
    import fea_core.solve as S

    def no_solver(*a, **k):
        raise AssertionError("the solver was called although the unit states are stored")
    monkeypatch.setattr(S, "_ccx_job", no_solver)
    second = run_study(unit, wd / "serve_2", study_dir=wd, on_log=lambda s: None, mesh=mesh)
    assert second.info["gravity_superposition"]["stored"] == ["x", "y", "z"] and not second.info["gravity_superposition"]["solved"]
    monkeypatch.undo()
    direct = _run(unit, mesh, wd, "serve_ref", "0")
    _same(second, direct, 1e-5, 1e-4)
    for k in range(3):                                                       # the mass and the centre of mass of the check come from the reactions of these steps
        assert second.steps[k].equilibrium["reaction_total"] == pytest.approx(direct.steps[k].equilibrium["reaction_total"], rel=1e-5)


def test_the_cache_off_solves_every_case(beam, monkeypatch):
    st, mesh, wd = beam
    monkeypatch.setenv("FEA_STATE_CACHE", "0")
    assert GS.plan(st, mesh, "ccx") is None
    assert not state_cache.enabled()


def test_a_few_gravity_cases_are_solved_as_they_are(beam):
    import copy
    st, mesh, _ = beam
    few = copy.deepcopy(st)
    few.load_cases = few.load_cases[:2]
    assert GS.plan(few, mesh, "ccx") is None                                 # two cases would not be fewer than the two unit states


def test_a_single_gravity_case_is_made_from_stored_unit_states_only(beam, tmp_path):
    import copy
    st, mesh, _ = beam
    one = copy.deepcopy(st)
    one.load_cases = one.load_cases[:1]                                      # "Gx+"
    state_cache.clear()
    assert GS.plan(one, mesh, "ccx") is None                                 # nothing stored: one case is solved as it is
    n = len(mesh.nodes)
    GS.save_unit(tmp_path, state_cache.study_digest(st, mesh), 0, "ccx", np.zeros((n, 3)), np.zeros((n, 6)), np.zeros((n, 3)), None, {})
    pl = GS.plan(one, mesh, "ccx")
    assert pl is not None and list(pl.cached) == [0] and not pl.missing
    assert GS.plan(one, mesh, "own") is None                                 # the own assembly needs the applied forces of its own unit states


def test_the_section_stresses_of_a_combined_state_are_the_same_combination_of_those_of_its_unit_states():
    import types

    from fea_core import section_forces as SF
    stub_a = types.SimpleNamespace(_sig_ip={1: np.full((4, 6), 2.0), 2: np.full((4, 6), 4.0)})
    stub_b = types.SimpleNamespace(_sig_ip={1: np.full((4, 6), 10.0), 2: np.full((4, 6), 20.0)})
    step = types.SimpleNamespace(_combo=[(stub_a, 0.5), (stub_b, -0.1)])
    out = SF._fetch_section_stresses(step, np.array([0, 1]), None)
    assert np.allclose(out[1], 0.5 * 2.0 - 0.1 * 10.0) and np.allclose(out[2], 0.5 * 4.0 - 0.1 * 20.0)
    assert step._sig_ip is out


# ----------------------------------------------------------------------------------------------------------------------------------------- the modal basis
def test_recover_batch_without_stresses_gives_the_same_nodal_forces(beam):
    from fea_core.recovery import recover_batch
    st, mesh, _ = beam
    rng = np.random.default_rng(1)
    U = rng.standard_normal((3, len(mesh.nodes), 3)) * 1e-3
    S, RF = recover_batch(st, mesh, U)
    S0, RF0 = recover_batch(st, mesh, U, stress=False)
    assert S0 is None and np.array_equal(RF, RF0) and S.shape == (3, len(mesh.nodes), 6)


def _modal_study(wd: Path, modal: int, dyn: int):
    st, parts = _beam(wd)
    st.load_cases.append(LoadCase("G", [Gravity(vector=[0.0, 0.0, -G])]))
    st.modal.enabled, st.modal.modes = True, modal
    st.dynamics.enabled, st.dynamics.modes = True, dyn
    st.dynamics.damping.kind, st.dynamics.damping.zeta = "modal", 0.03
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    return st, mesh


@needs_direct
def test_one_eigenvalue_analysis_serves_the_natural_frequencies_and_the_dynamics(tmp_path, monkeypatch):
    from fea_core import direct
    st, mesh = _modal_study(tmp_path, 6, 9)
    st.solver.modal_engine = "direct"
    calls = []
    real = direct.run_eigen_prepared

    def counting(job, opts, *a, **k):
        calls.append(int(opts["modes"]))
        return real(job, opts, *a, **k)
    monkeypatch.setattr(direct, "run_eigen_prepared", counting)
    res = run_study(st, tmp_path / "run", study_dir=tmp_path, on_log=lambda s: None, mesh=mesh)
    assert calls == [9]                                                      # ONE analysis with the larger mode count
    ms = res.modes[0]
    assert len(ms.values) == 6 and res.dynamics.K == 9 and ms.engine == "direct"
    assert ms.engine_info["state_cache"]["shared_analysis"] == {"modes_computed": 9, "modes_used": 6} or ms.engine_info["state_cache"]["cache"]["modes_used"] == 6
    monkeypatch.setenv("FEA_STATE_CACHE", "0")
    state_cache.clear()
    calls.clear()
    ref = run_study(st, tmp_path / "ref", study_dir=tmp_path, on_log=lambda s: None, mesh=mesh)
    assert sorted(calls) == [6, 9]
    assert np.allclose(ms.values, ref.modes[0].values, rtol=1e-9, atol=0.0)             # the gate of S2F
    assert np.allclose(res.dynamics.freq, ref.dynamics.freq, rtol=1e-9, atol=0.0)
    sh, sr = np.asarray(ms.shapes, float), np.asarray(ref.modes[0].shapes, float)
    assert np.minimum(np.abs(sh - sr), np.abs(sh + sr)).max() < 1e-6                     # float32 shapes, a mode of a repeated pair may differ in sign
    assert np.abs(res.dynamics.stress.astype(float) - ref.dynamics.stress.astype(float)).max() <= 1e-6 * np.abs(ref.dynamics.stress).max()


@needs_direct
def test_the_stored_basis_serves_the_3_axis_analysis_style_basis_and_the_interface_forces(tmp_path, monkeypatch):
    from fea_core import direct
    from fea_core.dynamics import interface as IF
    from fea_core.dynamics.modal import run_basis
    st, mesh = _modal_study(tmp_path, 6, 9)
    st.solver.modal_engine = "direct"
    calls = []
    real = direct.run_eigen_prepared

    def counting(job, opts, *a, **k):
        calls.append(int(opts["modes"]))
        return real(job, opts, *a, **k)
    monkeypatch.setattr(direct, "run_eigen_prepared", counting)
    unit = [GS.unit_case(a, 1.0) for a in range(3)]
    b1 = run_basis(st, mesh, tmp_path / "w1", lambda s: None, static_cases=unit)
    b2 = run_basis(st, mesh, tmp_path / "w2", lambda s: None, static_cases=unit)             # the 3-axis analysis: the same basis and the same unit statics
    assert calls == [9] and np.array_equal(b1.freq, b2.freq)
    assert np.abs(b1.stress.astype(float) - b2.stress.astype(float)).max() == 0.0 and b2.engine_info.get("state_cache")
    for s1, s2 in zip(b1.statics, b2.statics):
        assert np.abs(s1["U"] - s2["U"]).max() <= 1e-12 * np.abs(s1["U"]).max()
        assert np.abs(s1["S"].astype(float) - s2["S"].astype(float)).max() <= 1e-6 * np.abs(s1["S"]).max()
        assert np.abs(s1["RF"] - s2["RF"]).max() <= 1e-9 * np.abs(s1["RF"]).max()
    mi = IF.run_interface(st, mesh, tmp_path / "w3", modes=9, f_max=st.dynamics.f_max, log=lambda s: None)         # reactions only, from the stored basis
    assert calls == [9]
    monkeypatch.setenv("FEA_STATE_CACHE", "0")
    state_cache.clear()
    ref = IF.run_interface(st, mesh, tmp_path / "w4", modes=9, f_max=st.dynamics.f_max, log=lambda s: None)
    assert np.allclose(mi.freq, ref.freq, rtol=1e-9, atol=0.0)
    for name, g in ref.groups.items():
        assert np.allclose(mi.groups[name].F, g.F, rtol=1e-8, atol=1e-8 * np.abs(g.F).max()), name
        assert np.allclose(mi.groups[name].F_static, g.F_static, rtol=1e-8, atol=1e-8 * np.abs(g.F_static).max()), name


@needs_ccx
def test_a_ccx_modal_basis_of_the_same_study_is_computed_once(tmp_path, monkeypatch):
    from fea_core.dynamics import modal as DM
    st, mesh = _modal_study(tmp_path, 6, 9)
    st.solver.modal_engine = "ccx"
    calls = []
    real = DM._arrays_ccx

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(DM, "_arrays_ccx", counting)
    b1 = DM.run_basis(st, mesh, tmp_path / "w1", lambda s: None)
    b2 = DM.run_basis(st, mesh, tmp_path / "w2", lambda s: None)
    assert len(calls) == 1 and np.array_equal(b1.freq, b2.freq) and np.array_equal(b1.stress, b2.stress)
    assert b2.engine == "ccx" and b2.engine_info.get("stored_analysis")
    b3 = DM.run_basis(st, mesh, tmp_path / "w3", lambda s: None, modes=5)                      # another mode count is another analysis
    assert len(calls) == 2 and b3.K == 5


@needs_ccx
def test_unit_base_accelerations_in_ccx_come_from_one_job_and_equal_the_separate_ones(tmp_path, monkeypatch):
    from fea_core.dynamics import modal as DM
    st, mesh = _modal_study(tmp_path, 6, 9)
    st.solver.backend = "ccx"
    cases = [GS.unit_case(a, 1.0) for a in range(3)]
    (tmp_path / "ref").mkdir()
    ref = DM.static_solutions(st, mesh, cases, tmp_path / "ref", lambda s: None)
    got = GS.static_states(st, mesh, cases, tmp_path / "got", lambda s: None)
    assert got is not None and len(got) == 3
    for r, g in zip(ref, got):
        assert np.abs(r["U"] - g["U"]).max() <= 1e-5 * np.abs(r["U"]).max()
        assert np.abs(r["S"].astype(float) - g["S"].astype(float)).max() <= 1e-4 * np.abs(r["S"]).max()
        assert np.abs(r["RF"] - g["RF"]).max() <= 1e-5 * np.abs(r["RF"]).max()
    again = GS.static_states(st, mesh, cases[:1], tmp_path / "again", lambda s: None)          # stored: no second job
    assert not (tmp_path / "again" / "grav_units.inp").exists() and again is not None

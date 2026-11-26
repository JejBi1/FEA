"""Release 1.5.1 optimisations of the space model checks – no repeated solver work, and the numbers do not change:

  1 check 3: the six unit states share ONE assembly and ONE factorization (one right-hand side and one column of the prescribed values each) = the single-state solutions
  2 the PCB check computes the modal basis once for all components (the basis is passed, bit-identical)
  3 check 6: level 0 is the results of the caller when (and only when) they prove the same study state and mesh
  4 the effective-mass basis without the modal stresses / reactions = the same frequencies, effective masses and participation factors
  5 `static_solutions` in concurrent jobs = the one job (equal to round-off)
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from fea_core import config
from fea_core.space import _checks_common as cc
from fea_core.space import _checks_conv as cconv
from fea_core.space import _checks_static as cs
from fea_core.study import LoadCase
from tests.test_space_checks import L_BAR, W_BAR, H_BAR, build_sat, needs_ccx, needs_direct, slow, _rslab_on_the_path_of_the_worker  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")


@pytest.fixture(scope="module")
def sat(tmp_path_factory):
    wd = tmp_path_factory.mktemp("sat151")
    st, parts, mesh = build_sat(wd)
    return SimpleNamespace(study=st, parts=parts, mesh=mesh, wd=wd)


def _ctx(sat, sub, study=None, results=None, **opt):
    return cc.CheckContext(study or sat.study, sat.mesh, cc.ModelCheckLimits.template(), cc.ModelCheckOptions(threads=4, **opt), sat.wd / sub, lambda s: None, results=results)


# ------------------------------------------------------------------------------------------------------------------ 1: one factorization for the unit states
@needs_direct
@slow
def test_the_unit_states_share_one_assembly_and_equal_the_single_state_solutions(sat, monkeypatch):
    from fea_core import solve_large as SL
    mesh = sat.mesh
    nodes = np.unique(mesh.surface_nodes("end"))
    rng = np.random.default_rng(5)
    states = [cs.PrescribedState(f"s{k}", nodes, rng.standard_normal((len(nodes), 3)) * 0.01) for k in range(3)]
    st = copy.deepcopy(sat.study)
    st.masses = []
    calls = []
    real = SL._assemble
    monkeypatch.setattr(SL, "_assemble", lambda *a, **k: calls.append(1) or real(*a, **k))
    from fea_core import direct
    infos: list = []
    real_rp = direct.run_prepared

    def spy(*a, **k):
        X, info = real_rp(*a, **k)
        infos.append(info)
        return X, info
    monkeypatch.setattr(direct, "run_prepared", spy)
    many = cs.solve_prescribed(_ctx(sat, "many", st), st, states, "many")
    assert len(calls) == 1                                        # one assembly (and one factorization) for the three states
    assert [o.info["backward_error"] for o in many] == infos[0]["backward_error"]          # every state has the backward error of its own right-hand side
    for k, s in enumerate(states):
        one = cs.solve_prescribed(_ctx(sat, f"one{k}", st), st, [s], f"one{k}")[0]
        assert np.abs(many[k].U - one.U).max() <= 1e-12 * np.abs(one.U).max()
        assert np.abs(many[k].S - one.S).max() <= 1e-9 * np.abs(one.S).max()
        assert np.abs(many[k].RF - one.RF).max() <= 1e-9 * np.abs(one.RF).max()
    assert len(calls) == 4


@needs_direct
@slow
def test_states_with_other_constrained_dofs_are_separate_factorizations(sat, monkeypatch):
    from fea_core import solve_large as SL
    st = copy.deepcopy(sat.study)
    st.masses = []
    a, b = np.unique(sat.mesh.surface_nodes("end")), np.unique(sat.mesh.surface_nodes("top_B"))
    states = [cs.PrescribedState("a", a, np.full((len(a), 3), 0.01)), cs.PrescribedState("b", b, np.full((len(b), 3), 0.02)),
              cs.PrescribedState("a2", a, np.full((len(a), 3), -0.01))]
    calls = []
    real = SL._assemble
    monkeypatch.setattr(SL, "_assemble", lambda *x, **k: calls.append(1) or real(*x, **k))
    out = cs.solve_prescribed(_ctx(sat, "mixed", st), st, states, "mixed")
    assert [o.name for o in out] == ["a", "b", "a2"] and len(calls) == 2
    assert np.abs(out[0].U + out[2].U).max() <= 1e-12 * np.abs(out[0].U).max()      # linear: the opposite values give the opposite displacement


# ------------------------------------------------------------------------------------------------------------------ 2: the PCB basis once
@needs_direct
@slow
def test_the_pcb_report_computes_the_modal_basis_once_for_all_components(sat, monkeypatch):
    from fea_core.dynamics import modal as DM
    from fea_core.space import pcb as PB, s4_run
    asd = [[20.0, 0.02], [2000.0, 0.02]]
    geo = {"centre": [L_BAR, W_BAR / 2, H_BAR], "support_selection": "launch", "normal": [0, 0, 1]}
    comps = [dict(name=f"U{k}", B_mm=100.0, h_mm=1.6, L_mm=20.0 + k, r=1.0, geometry=dict(geo)) for k in range(3)]
    calls = []
    real = DM.run_basis
    monkeypatch.setattr(DM, "run_basis", lambda *a, **k: calls.append(k.get("stresses", True)) or real(*a, **k))
    st = copy.deepcopy(sat.study)
    st.dynamics.modes = 8
    rep = s4_run.pcb_report(st, comps, mesh=sat.mesh, workdir=lambda: sat.wd / "pcb_once", asd=asd, zeta=0.03)
    assert calls == [False]                                       # ONE basis, without the modal stresses (the board displacement reads the shapes and Γ only)
    ref = PB.z_rms_from_study(st, sat.mesh, sat.wd / "pcb_ref", centre=geo["centre"], supports="launch", normal=geo["normal"], direction="z", asd=s4_run.profile_from_study(st, "random", asd),
                              zeta=0.03, modes=8)
    for row in rep["components"]:
        assert row["z_rms_from_model"]["z_rms_mm"] == ref["z_rms_mm"]       # bit-identical: the same basis arrays


# ------------------------------------------------------------------------------------------------------------------ 3: level 0 of the convergence check
def _results_with_digest(sat, study, modes=0, steps=True):
    from fea_core.results import Results, StepResult  # noqa: F401
    from fea_core.solve import study_state_digest
    from types import SimpleNamespace as NS
    res = NS(mesh=sat.mesh, steps=[NS(name=lc.name) for lc in study.load_cases] if steps else [], modes=[NS(kind="modal", values=np.arange(1.0, modes + 1))] if modes else [],
             info={"study_digest": study_state_digest(study)}, twod=None, shell=None)
    return res


def test_level_0_is_reused_only_when_the_results_prove_the_same_study_and_mesh(sat):
    st = copy.deepcopy(sat.study)
    st.modal.enabled, st.modal.modes, st.modal.f_min, st.modal.f_max = True, 8, 0.0, None
    good = _results_with_digest(sat, st, modes=8)
    assert cconv.reusable_level0(_ctx(sat, "c0", st, good), 5) is good
    assert cconv.reusable_level0(_ctx(sat, "c0", st, None), 5) is None                          # no results
    # the study changed after the run
    st2 = copy.deepcopy(st)
    st2.load_cases[0].loads[0].vector = [0.0, 0.0, -101.0]
    assert cconv.reusable_level0(_ctx(sat, "c1", st2, good), 5) is None
    st3 = copy.deepcopy(st)
    st3.mesh.size = 4.0
    assert cconv.reusable_level0(_ctx(sat, "c1b", st3, good), 5) is None
    no_digest = _results_with_digest(sat, st, modes=8)
    no_digest.info = {}
    assert cconv.reusable_level0(_ctx(sat, "c2", st, no_digest), 5) is None                     # results of an older run / file: nothing proves them
    # another mesh
    other = _results_with_digest(sat, st, modes=8)
    other.mesh = copy.deepcopy(sat.mesh)
    other.mesh.nodes = other.mesh.nodes + 1e-3
    assert cconv.reusable_level0(_ctx(sat, "c3", st, other), 5) is None
    same_content = _results_with_digest(sat, st, modes=8)
    same_content.mesh = copy.deepcopy(sat.mesh)
    assert cconv.reusable_level0(_ctx(sat, "c3b", st, same_content), 5) is same_content        # another object, the same content
    # other load cases / too few modes / a window
    assert cconv.reusable_level0(_ctx(sat, "c4", st, _results_with_digest(sat, st, modes=8, steps=False)), 5) is None
    assert cconv.reusable_level0(_ctx(sat, "c5", st, _results_with_digest(sat, st, modes=3)), 5) is None
    assert cconv.reusable_level0(_ctx(sat, "c5b", st, _results_with_digest(sat, st, modes=0)), 5) is None
    stw = copy.deepcopy(st)
    stw.modal.f_min = 10.0
    assert cconv.reusable_level0(_ctx(sat, "c6", stw, _results_with_digest(sat, stw, modes=8)), 5) is None
    stn = copy.deepcopy(st)
    stn.modal.enabled = False
    assert cconv.reusable_level0(_ctx(sat, "c7", stn, _results_with_digest(sat, stn, modes=8)), 5) is None
    assert cconv.reusable_level0(_ctx(sat, "c8", stn, _results_with_digest(sat, stn, modes=0)), 0) is not None      # no frequency followed: the load cases are enough


def test_the_digest_ignores_the_space_settings_and_their_feature_entry_only():
    from fea_core.solve import study_state_digest
    from fea_core.space.model import SpaceSettings
    from fea_core.study import Study
    st = Study("d", "none.step")
    d0 = study_state_digest(st)
    st.space = copy.deepcopy(st.space)
    assert study_state_digest(st) == d0
    st.space = SpaceSettings()                                    # created after the run: the "features" list of the study dict changes with it
    assert "space" in st.features() and study_state_digest(st) == d0
    st.mesh.size = st.mesh.size + 1.0
    assert study_state_digest(st) != d0


def test_the_digest_of_a_result_is_that_of_the_study_when_run_study_was_entered(sat, monkeypatch):
    """A study edited while the run goes on (the mesh is made in the run) is not the one the result belongs to: the digest is taken on entry."""
    from fea_core import solve
    from fea_core.results import Results
    st = copy.deepcopy(sat.study)
    st.load_cases, st.modal.enabled = [], False
    entry = solve.study_state_digest(st)

    def prepare(study, workdir, study_dir, log, mesh, parts, cancel):
        study.mesh.size += 1.0                                    # the run changes the study object
        return sat.mesh, []
    monkeypatch.setattr(solve, "prepare_mesh", prepare)
    monkeypatch.setattr(Results, "save", lambda self, path: None)
    res = solve.run_study(st, sat.wd / "entry_digest", mesh=sat.mesh, parts=sat.parts)
    assert res.info["study_digest"] == entry != solve.study_state_digest(st)


def test_a_partial_nonlinear_result_is_never_level_0(sat):
    st = copy.deepcopy(sat.study)
    assert cconv.reusable_level0(_ctx(sat, "p0", st, _results_with_digest(sat, st)), 0) is not None
    stopped = _results_with_digest(sat, st)
    stopped.steps[0].partial = True                               # "Stop and keep results" of a nonlinear run
    assert cconv.reusable_level0(_ctx(sat, "p1", st, stopped), 0) is None
    flagged = _results_with_digest(sat, st)
    flagged.info["nonlinear"] = {"mode": "sequence", "partial": {"job": {"last_step": 2, "of_steps": 5}}}
    assert cconv.reusable_level0(_ctx(sat, "p2", st, flagged), 0) is None
    converged = _results_with_digest(sat, st)
    converged.info["nonlinear"] = {"mode": "sequence", "partial": {}}
    assert cconv.reusable_level0(_ctx(sat, "p3", st, converged), 0) is converged


def test_the_mesh_of_level_0_must_agree_in_the_edge_point_and_tie_data_too(sat):
    """The one digest of the mesh (`state_cache.mesh_digest`, also `loads.mesh_fingerprint` and `mesh_key`) covers the edge / point selections, the tie table and the node aliases; a result on
    a mesh that differs there is not level 0."""
    from fea_core.space.loads import mesh_fingerprint
    st = copy.deepcopy(sat.study)
    ties = [{"part": "A", "slave": "t_s", "master": "t_m", "area_mm2": 10.0}]
    variants = [("edges", {"e1": np.array([[0, 1, 2]], np.int32)}), ("points", {"p1": np.array([3], np.int32)}), ("report", {**sat.mesh.report, "ties": ties}),
                ("node_alias", np.arange(len(sat.mesh.nodes))[::-1].copy())]
    assert cconv.mesh_key(sat.mesh) == cconv.mesh_key(copy.deepcopy(sat.mesh))
    for name, val in variants:
        other = copy.deepcopy(sat.mesh)
        setattr(other, name, val)
        assert mesh_fingerprint(other) != mesh_fingerprint(sat.mesh), name       # (what the old fingerprint of the content could not see)
        res = _results_with_digest(sat, st)
        res.mesh = other
        assert cconv.reusable_level0(_ctx(sat, f"mk_{name}", st, res), 0) is None, name
        back = _results_with_digest(sat, st)                                       # and the other way round: the checks mesh has it, the results mesh has not
        ctx = _ctx(sat, f"mk2_{name}", st, back)
        ctx.mesh = other
        assert cconv.reusable_level0(ctx, 0) is None, name


def _fake_levels(monkeypatch, level_modes):
    """`run_study` replaced by results with the given natural frequencies per call (no solver): the calls of the convergence check, in order."""
    from fea_core import solve
    calls: list = []

    def fake(study, workdir, **kw):
        calls.append(kw)
        modes = level_modes[len(calls) - 1]
        return SimpleNamespace(mesh=kw.get("mesh") or SimpleNamespace(nodes=np.zeros((3, 3)), tets=np.zeros((1, 10), int)), steps=[], warnings=[], info={}, twod=None, shell=None,
                               modes=[SimpleNamespace(kind="modal", values=np.asarray(modes, float))] if modes is not None else [])
    monkeypatch.setattr(solve, "run_study", fake)
    return calls


def _conv_ctx(sat, sub, st, results, n_freq):
    from tests.test_space_checks import LIMITS
    return cc.CheckContext(st, sat.mesh, LIMITS, cc.ModelCheckOptions(include_convergence=True, convergence_levels=2, convergence_frequencies=n_freq, threads=4), sat.wd / sub,
                           lambda s: None, parts=[], study_dir=sat.wd, results=results)


def test_level_0_from_the_results_gives_the_row_of_the_full_run_also_with_more_stored_modes(sat, monkeypatch):
    """The rigid-body cut of the frequencies is made on the leading `n` modes, as the full path does it: a very soft elastic mode (0.05 Hz) survives although the study stored many
    more modes up to tens of kHz (a threshold from ALL stored modes dropped it and shifted the indices)."""
    st = copy.deepcopy(sat.study)
    st.load_cases, st.modal.enabled, st.modal.modes, st.modal.f_min, st.modal.f_max = [], True, 30, 0.0, None
    lead = [0.05, 10.0, 20.0, 30.0, 40.0]
    stored = np.concatenate([lead, np.linspace(50.0, 6.0e4, 25)])
    level1 = [0.05, 9.9, 19.8, 29.7, 39.6]
    _fake_levels(monkeypatch, [lead, level1])                                   # the full path: level 0 is run with n modes, level 1 too
    full = cconv.run_convergence_check(_conv_ctx(sat, "f0", st, None, 5))
    stored_res = _results_with_digest(sat, st, modes=0)
    stored_res.modes = [SimpleNamespace(kind="modal", values=stored)]
    calls = _fake_levels(monkeypatch, [level1])                                 # reuse: only level 1 is run
    reused = cconv.run_convergence_check(_conv_ctx(sat, "f1", st, stored_res, 5))
    assert len(calls) == 1
    assert full.values["levels"][0]["frequencies_Hz"] == reused.values["levels"][0]["frequencies_Hz"] == lead
    assert full.values["frequency_convergence"] == reused.values["frequency_convergence"] and len(reused.values["frequency_convergence"]) == 5


def test_no_followed_frequency_leaves_no_frequency_key_in_the_level_row(sat, monkeypatch):
    st = copy.deepcopy(sat.study)
    st.load_cases, st.modal.enabled, st.modal.modes, st.modal.f_min, st.modal.f_max = [], True, 5, 0.0, None
    res = _results_with_digest(sat, st, modes=5)
    calls = _fake_levels(monkeypatch, [None])
    r = cconv.run_convergence_check(_conv_ctx(sat, "nf", st, res, 0))
    assert len(calls) == 1 and all("frequencies_Hz" not in row for row in r.values["levels"])


def test_the_states_of_one_matrix_are_grouped_only_with_the_own_assembly():
    a = np.zeros(12, bool)
    a[:3] = True
    b = a.copy()
    b[5] = True
    fixed = [a, a.copy(), b, a.copy()]
    assert cs.group_states(fixed, True) == [[0, 1, 3], [2]]
    assert cs.group_states(fixed, False) == [[0], [1], [2], [3]]               # FEA_ASSEMBLY=ngsolve reads g.npy as one vector: one state per group


def test_every_state_gets_the_backward_error_of_its_own_column():
    info = {"backward_error": [1e-15, 5e-14, 2e-16]}
    assert [cs.column_backward_error(info, k) for k in range(3)] == [1e-15, 5e-14, 2e-16]
    assert cs.column_backward_error(info, 7) == 5e-14 and cs.column_backward_error({}, 0) == 0.0


@needs_direct
@needs_ccx
@slow
def test_the_convergence_check_with_the_results_of_the_study_gives_the_numbers_of_the_full_run(sat):
    from fea_core.solve import run_study
    from tests.test_space_checks import LIMITS
    st = copy.deepcopy(sat.study)
    st.masses = []
    st.modal.enabled, st.modal.modes = True, 5
    res = run_study(st, sat.wd / "l0_study", mesh=sat.mesh)
    assert res.info.get("study_digest")
    opts = dict(threads=4, convergence_levels=2, convergence_frequencies=5)
    plain = cconv.run_convergence_check(cc.CheckContext(st, sat.mesh, LIMITS, cc.ModelCheckOptions(include_convergence=True, **opts), sat.wd / "conv_plain", lambda s: None,
                                                        parts=sat.parts, study_dir=sat.wd))
    msgs: list[str] = []
    fast = cconv.run_convergence_check(cc.CheckContext(st, sat.mesh, LIMITS, cc.ModelCheckOptions(include_convergence=True, **opts), sat.wd / "conv_fast", msgs.append,
                                                       parts=sat.parts, study_dir=sat.wd, results=res))
    assert any("results of the study are used" in m for m in msgs)
    a, b = plain.values["levels"][0], fast.values["levels"][0]
    assert a["nodes"] == b["nodes"]
    assert np.allclose(a["frequencies_Hz"], b["frequencies_Hz"], rtol=1e-9, atol=0)
    assert b["max_mises_MPa"] == pytest.approx(a["max_mises_MPa"], rel=1e-9)
    assert fast.values["levels"][1]["frequencies_Hz"] == plain.values["levels"][1]["frequencies_Hz"]
    assert plain.status == fast.status


def test_the_pcb_report_computes_the_modal_basis_once_and_hands_it_to_every_component(sat, monkeypatch):
    """Fast pin without a solver: one `pcb_basis` call, the same basis object in every component's `z_rms_from_study`, the results of the components untouched."""
    from fea_core.space import pcb as PB, s4_run
    made, got = [], []
    basis = object()
    monkeypatch.setattr(PB, "pcb_basis", lambda *a, **k: made.append(1) or basis)
    monkeypatch.setattr(PB, "z_rms_from_study", lambda *a, **k: got.append(k["basis"]) or {"z_rms_mm": 0.123, "modes": 8, "damping": "zeta 0.03", "dominant_f_Hz": 210.0})
    geo = {"centre": [L_BAR, W_BAR / 2, H_BAR], "support_selection": "launch", "normal": [0, 0, 1]}
    comps = [dict(name=f"U{k}", B_mm=100.0, h_mm=1.6, L_mm=20.0 + k, r=1.0, geometry=dict(geo)) for k in range(3)]
    rep = s4_run.pcb_report(sat.study, comps, mesh=sat.mesh, workdir=lambda: sat.wd / "pcb_fast", asd=[[20.0, 0.02], [2000.0, 0.02]], zeta=0.03)
    assert len(made) == 1 and got == [basis] * 3
    assert [row["z_rms_from_model"]["z_rms_mm"] for row in rep["components"]] == [0.123] * 3


# ------------------------------------------------------------------------------------------------------------------ 4: the basis without stresses
def _fake_modal_run(monkeypatch, sat, statics=None):
    from fea_core.direct import modal_run
    rng = np.random.default_rng(2)
    k, n = 4, len(sat.mesh.nodes)
    dm = SimpleNamespace(freq=np.array([0.0, 120.0, 300.0, 810.0]), shapes=rng.standard_normal((k, n, 3)), statics=statics, gamma=np.zeros((k, 6)), eff_mass=np.zeros((k, 6)),
                         total_free=np.zeros(6), cols=None, columns_all=[], coeffs_all=[], cache=None)
    monkeypatch.setattr(modal_run, "run_modes", lambda *a, **kw: dm)
    monkeypatch.setattr(modal_run, "protocol", lambda d: {})
    return dm


def test_the_basis_arrays_without_stresses_are_the_same_but_the_stresses_and_the_element_matrices_are_not_built(sat, monkeypatch):
    """`_arrays_direct(with_stress=False, with_reactions=False)`: the frequencies, shapes, Γ and effective masses are those of the full call (same objects of the modal run); no stress recovery, no element matrices."""
    from fea_core import recovery
    from fea_core.dynamics import modal as DM
    dm = _fake_modal_run(monkeypatch, sat)
    built = []
    real_rb = recovery.recover_batch
    monkeypatch.setattr(recovery, "recover_batch", lambda *a, **k: built.append("recover") or real_rb(*a, **k))
    real_es = recovery.element_stiffness
    monkeypatch.setattr(recovery, "element_stiffness", lambda *a, **k: built.append("C") or real_es(*a, **k))
    args = (sat.study, sat.mesh, sat.wd / "arr", lambda s: None, None, None, 4, None, "x", None)
    lite = DM._arrays_direct(*args, with_stress=False, with_reactions=False)
    assert built == [] and lite["stress"].shape == (4, 0, 6) and not np.any(lite["rfs"])
    for key in ("freq", "shapes", "participation", "eff_mass", "total"):
        assert np.array_equal(np.asarray(lite[key]), np.asarray(DM._arrays_direct(*args)[key])), key
    assert "C" in built and "recover" in built and lite["freq"] is dm.freq


def test_a_basis_without_the_modal_reactions_reports_none_and_the_warnings_cope(sat):
    from fea_core.dynamics.modal import basis_warnings
    basis = SimpleNamespace(checks={"mass_norm_error": 1e-9, "gamma_error": 1e-9, "mode_equilibrium": None, "mode_equilibrium_checked": False, "mass_fraction_xyz": [0.95] * 3},
                            K=3, freq=np.array([1.0, 2.0, 3.0]), rf_from_inertia=False, prestress_nlgeom=False)
    assert basis_warnings(basis, {0, 1, 2}, None) == []                      # (None is "not computed", not a number to compare)
@needs_direct
@slow
def test_the_basis_without_modal_stresses_has_the_same_modes_and_effective_masses(sat):
    from fea_core.dynamics.modal import run_basis
    st = copy.deepcopy(sat.study)
    st.dynamics.modes = 8
    full = run_basis(st, sat.mesh, sat.wd / "basis_full", lambda s: None, modes=8)
    lite = run_basis(st, sat.mesh, sat.wd / "basis_lite", lambda s: None, modes=8, with_stress=False, with_reactions=False)
    assert full.engine == lite.engine == "direct"
    for name in ("freq", "shapes", "gamma", "eff_mass", "total_mass"):
        assert np.array_equal(getattr(full, name), getattr(lite, name)), name
    for k in ("mass_norm_error", "gamma_error", "mass_fraction_xyz", "model_mass_t"):
        assert full.checks[k] == lite.checks[k], k
    assert lite.stress.shape == (lite.K, 0, 6) and full.stress.shape[1] == len(sat.mesh.nodes)
    assert lite.checks["mode_equilibrium_checked"] is False and lite.checks["mode_equilibrium"] is None
    from fea_core.space import _checks_modal as cm
    r = cm.run_effective_mass_check(cc.CheckContext(st, sat.mesh, cc.ModelCheckLimits.template(), cc.ModelCheckOptions(threads=4, modes=8), sat.wd / "em", lambda s: None))
    assert r.values["frequencies_Hz"] == full.freq[:10].tolist()           # the check takes the basis without stresses: the same frequencies


# ------------------------------------------------------------------------------------------------------------------ 5: static solutions in concurrent ccx jobs
@needs_ccx
def test_static_solutions_in_concurrent_jobs_equal_the_one_job(sat, monkeypatch):            # (4 s: in the fast suite)
    from fea_core.ccx import parallel
    from fea_core.dynamics.modal import static_solutions
    from fea_core.study import Force
    cases = [LoadCase(f"c{k}", [Force("end", v)]) for k, v in enumerate(([100.0, 0, 0], [0, 100.0, 0], [0, 0, -100.0]))]
    (sat.wd / "ss_serial").mkdir(exist_ok=True)
    (sat.wd / "ss_par").mkdir(exist_ok=True)
    monkeypatch.setenv("FEA_CCX_PARALLEL", "0")
    serial = static_solutions(sat.study, sat.mesh, cases, sat.wd / "ss_serial", lambda s: None)
    monkeypatch.setenv("FEA_CCX_PARALLEL", "1")
    planned = []
    real = parallel.plan
    monkeypatch.setattr(parallel, "plan", lambda *a, **k: planned.append(real(*a, **k)) or planned[-1])
    par = static_solutions(sat.study, sat.mesh, cases, sat.wd / "ss_par", lambda s: None)
    assert len(serial) == len(par) == 3
    if planned and planned[-1][0] < 2:
        pytest.skip("the machine plans one job: nothing concurrent to compare")
    for a, b in zip(serial, par):
        for k in ("U", "S", "RF"):                                # measured: U 2.4e-10, S 5e-12, RF identical (relative to the maximum); the margin is for the thread-dependent sum order
            assert np.abs(a[k] - b[k]).max() <= 1e-8 * np.abs(a[k]).max(), k
    assert config.CCX_EXE.exists()

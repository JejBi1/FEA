"""F14 S2, review round: regression tests of the counterexamples of the independent review (ce1 - ce4) and of the fixes that followed.

C1  a frequency item with a usable limit but no value cannot hide behind a PASS          (budgets.frequency_budget)
C2  a kept 3-axis result is only used for the study / mesh / damping / environments it was computed for (digest), and is cleared by the changes
C3  every verdict that is shown (MCP margins, report chapter, margins tab) goes through the envelopes (loads.sign_off_margins)
I1  the frequency PASS / FAIL needs the confirmed environment; the QSL status regenerates the cases and compares them
I2  modes below / above the shock table; I3 the damage of the axes is summed; I4 damping is an environment input; I5 silent omissions block a PASS
I6  mass properties as the solver sees the added masses; minor: validations of limits, environments, `run_dynamics(basis=...)`, S3 mass = S2 mass.
Every number here is an ARBITRARY TEST VALUE, not a value of any standard.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import verification.cases  # noqa: F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.dynamics.model import PointMass
from fea_core.space import budgets as BG
from fea_core.space import environments as EV
from fea_core.space import loads as LD
from fea_core.space import mass_props as MP
from fea_core.space import margins as MG
from fea_core.space.model import SpaceSettings
from verification import space_cases as SC
from verification import space_load_cases as SL


# ====================================================================== C1: the frequency check (threshold = significance of ONE mode)
# a COMPLETE basis: every direction has >= 90 % of its mass in the computed modes, the lowest significant modes are above the limits (x 210 Hz, y 140 Hz, z 150 Hz)
FREQ = np.array([95.0, 140.0, 150.0, 210.0])
FRAC = np.array([[0.0, 0.02, 0.0], [0.0, 0.92, 0.0], [0.0, 0.0, 0.93], [0.95, 0.0, 0.0]])
LIM = {"f_min_axial_hz": 200.0, "f_min_lateral_hz": 120.0, "f_min_eff_mass_fraction": 0.5}


def _fb(freq, frac, limits=None, **kw):
    return BG.frequency_budget(np.array(freq, float), np.array(frac, float), "x", SL.space_with_limits(limits or LIM, env_axis="x", **kw))


def test_c1_a_complete_basis_with_no_significant_mode_below_the_limits_passes():
    rep = _fb(FREQ, FRAC)
    assert rep.verdict == BG.PASS and [i.status for i in rep.items] == [BG.PASS, BG.PASS]
    assert rep.items[0].value == 210.0 and rep.items[1].value == 140.0                 # the lowest SIGNIFICANT mode per direction is reported (lateral: y, not z at 150)
    assert "mode 4" in rep.items[0].name and "mode 2 (y)" in rep.items[1].name
    assert "lowest significant mode" in rep.items[1].note


@pytest.mark.parametrize("label, freq, frac, items", [
    # the three split-mass cases of the review: the earlier rule said "f_1 lies above the highest computed frequency" and passed all of them
    ("30 % at 30 Hz, f_min 200 Hz", [30.0, 140.0, 300.0], [[0.30, 0, 0], [0, 0.92, 0.0], [0.30, 0, 0]], [0]),
    ("49 % at 30 Hz + 49 % at 300 Hz", [30.0, 140.0, 300.0], [[0.49, 0, 0], [0, 0.92, 0], [0.49, 0, 0]], [0]),
    ("lateral z 45 % at 20 Hz + 45 % at 25 Hz", [20.0, 25.0, 140.0, 250.0], [[0, 0, 0.45], [0, 0, 0.45], [0, 0.92, 0], [0.95, 0, 0]], [1]),
])
def test_c1_split_mass_is_never_a_pass(label, freq, frac, items):
    rep = _fb(freq, frac)
    assert rep.verdict != BG.PASS, label
    for i in items:
        it = rep.items[i]
        assert it.status == BG.UNCONFIRMED, (label, it.name, it.status)
        assert it.note and ("effective mass" in it.note), it.note                      # says WHY (the split mass), not only "unconfirmed"
    assert rep.verdict == BG.UNCONFIRMED                                                # and not FAIL either: no single mode reached the confirmed fraction


def test_c1_a_significant_mode_below_f_min_fails_and_is_reported():
    rep = _fb([30.0, 140.0, 300.0, 400.0], [[0.55, 0, 0], [0, 0.92, 0], [0.40, 0, 0.0], [0, 0, 0.95]])
    ax = rep.items[0]
    assert ax.status == BG.FAIL and ax.value == 30.0 and "BELOW f_min" in ax.note and "mode 1" in ax.note
    assert rep.verdict == BG.FAIL
    # the same data with the environment / the set unconfirmed: the number is shown, the decision is not made
    assert _fb([30.0, 140.0, 300.0, 400.0], [[0.55, 0, 0], [0, 0.92, 0], [0.40, 0, 0.0], [0, 0, 0.95]], env_confirmed=False).items[0].status == BG.UNCONFIRMED
    assert _fb([30.0, 140.0, 300.0, 400.0], [[0.55, 0, 0], [0, 0.92, 0], [0.40, 0, 0.0], [0, 0, 0.95]], confirmed=False).items[0].status == BG.UNCONFIRMED
    # a lateral mode under its limit fails the lateral item, whichever lateral axis carries it
    lat = _fb([100.0, 140.0, 210.0, 260.0], [[0, 0, 0.6], [0, 0.92, 0], [0.95, 0, 0], [0, 0, 0.35]])
    assert lat.items[1].status == BG.FAIL and lat.items[1].value == 100.0 and "mode 1 (z)" in lat.items[1].name


def test_c1_the_basis_must_reach_f_min_and_account_for_the_mass():
    high = _fb(FREQ, FRAC, {**LIM, "f_min_axial_hz": 400.0})                          # the computed 210 Hz mode IS under 400 Hz: a known violation, not a missing mode
    assert high.items[0].status == BG.FAIL and high.items[0].value == 210.0 and high.verdict == BG.FAIL
    short = _fb([95.0, 140.0, 150.0], [[0, 0.02, 0], [0, 0.92, 0], [0, 0, 0.93]])    # the basis ends at 150 Hz < 200 Hz and holds no axial mass: a mode under 200 Hz may be missing
    assert short.items[0].status == BG.UNCONFIRMED and short.verdict == BG.UNCONFIRMED and "BELOW f_min" in short.items[0].note
    only_axial = _fb([250.0, 260.0], [[0.6, 0, 0], [0.35, 0, 0]])                    # axial 95 % accounted, but no lateral mass at all in the basis
    assert only_axial.items[0].status == BG.PASS
    assert only_axial.items[1].status == BG.UNCONFIRMED and "account for 0.0 %" in only_axial.items[1].note and only_axial.verdict == BG.UNCONFIRMED
    low_total = _fb([210.0, 140.0, 150.0], [[0.6, 0, 0], [0, 0.92, 0], [0, 0, 0.93]])
    assert low_total.items[0].status == BG.UNCONFIRMED and "account for 60.0 %" in low_total.items[0].note            # 60 % of the axial mass computed: 40 % unaccounted
    nothing = BG.frequency_budget(np.array([]), np.zeros((0, 3)), "x", SL.space_with_limits(LIM, env_axis="x"))
    assert [i.status for i in nothing.items] == [BG.UNCONFIRMED, BG.UNCONFIRMED] and nothing.verdict == BG.UNCONFIRMED
    unconf = _fb(FREQ, FRAC, confirmed=False)
    assert unconf.verdict == BG.UNCONFIRMED                                         # an unusable limit never decides


def test_c1_unknown_effective_masses_are_never_a_pass():
    """N6: an axis without a total mass gave fractions 'divided by 1'; it is UNKNOWN (NaN) and the check says so."""
    frac = np.array(FRAC, float)
    frac[:, 0] = np.nan                                                             # the x total mass is unknown
    rep = BG.frequency_budget(FREQ, frac, "x", SL.space_with_limits(LIM, env_axis="x"))
    assert rep.items[0].status == BG.UNCONFIRMED and "unknown" in rep.items[0].note and rep.verdict == BG.UNCONFIRMED
    res = SimpleNamespace(modes=[SimpleNamespace(kind="modal", eff_mass=[[1.0, 1.0, 1.0]], total_mass=[0.0, 2.0, 2.0], values=[100.0], name="m", engine="e")], dynamics=None)
    f, fr, _src = BG.modal_data_of(res)
    assert np.isnan(fr[0, 0]) and fr[0, 1] == 0.5                                  # NOT 1.0 / 1.0
    dyn = SimpleNamespace(K=1, model_mass=0.0, eff_mass=[[1.0, 1.0, 1.0, 0, 0, 0]], freq=[100.0])
    assert np.isnan(BG.modal_data_of(SimpleNamespace(modes=[], dynamics=dyn))[1]).all()
    b = SL.synthetic_basis([85.0], [0])
    b.total_mass, b.model_mass = np.zeros(6), 0.0
    assert np.isnan(LD.eff_fractions(b)).all()                                      # zero / unknown total mass: unknown, not 1.0
    b.model_mass = b.eff_mass[0, 0] * 2
    assert LD.eff_fractions(b)[0, 0] == 0.5                                         # a known model mass is used per axis


def test_c1_without_a_threshold_every_mode_counts():
    """No confirmed fraction: the lowest mode of the model decides (as before), a mode under f_min fails even if its mass sits in another direction."""
    lim = {"f_min_axial_hz": 90.0, "f_min_lateral_hz": 90.0}
    ok = _fb(FREQ[1:], FRAC[1:], lim)
    assert ok.items[0].value == 140.0 and ok.items[0].status == BG.PASS and any("every mode counts" in n for n in ok.notes)
    bad = _fb(FREQ, FRAC, {**lim, "f_min_axial_hz": 100.0})                          # the 95 Hz mode (y, 2 %) is below 100 Hz and counts
    assert bad.items[0].status == BG.FAIL and bad.items[0].value == 95.0


def test_c1_a_limit_with_a_value_and_no_number_is_unconfirmed_not_skipped():
    it = BG.judge("x", None, "f_min_axial_hz", SL.space_with_limits(LIM), "min")
    assert it.status == BG.UNCONFIRMED
    it = BG.judge("x", None, "mass_max_kg", SL.space_with_limits(LIM), "max")            # no such limit entered: nothing to check
    assert it.status == BG.SKIPPED


# ====================================================================== I1: frequency PASS needs the confirmed environment
def test_i1_frequency_needs_the_confirmed_environment():
    f, fr = FREQ, FRAC
    ok = BG.frequency_budget(f, fr, "x", SL.space_with_limits(LIM, env_axis="x"))
    assert ok.verdict == BG.PASS and not any("not confirmed" in n for n in ok.notes)
    unconf = BG.frequency_budget(f, fr, "x", SL.space_with_limits(LIM, env_axis="x", env_confirmed=False))
    assert unconf.verdict == BG.UNCONFIRMED and all(i.status == BG.UNCONFIRMED for i in unconf.items)
    assert any("not confirmed" in n for n in unconf.notes) and any("launch environment" in i.note for i in unconf.items)
    fail = BG.frequency_budget(f, fr, "x", SL.space_with_limits({**LIM, "f_min_axial_hz": 250.0}, env_axis="x", env_confirmed=False))
    assert fail.items[0].status == BG.UNCONFIRMED                                  # a FAIL is not given on the strength of an unconfirmed launch axis either
    no_env = BG.frequency_budget(f, fr, "x", SL.space_with_limits(LIM))           # the space settings carry no environment at all
    assert no_env.verdict == BG.UNCONFIRMED
    assert BG.frequency_budget(f, fr, "x", SL.space_with_limits(LIM), env_confirmed=True).verdict == BG.PASS      # explicit override of the caller


# ====================================================================== limit values
def test_limit_values_are_validated_everywhere():
    for key, bad in (("f_min_axial_hz", 0.0), ("f_min_lateral_hz", -5.0), ("f_min_eff_mass_fraction", 0.0), ("f_min_eff_mass_fraction", 1.5),
                     ("qsl_axial_g", 0.0), ("qsl_lateral_g", -3.0), ("mass_max_kg", 0.0), ("mass_min_kg", -1.0)):
        assert BG.limit_value_problem(key, bad), (key, bad)
        sp = SL.space_with_limits({key: bad}, confirmed=False)
        lv = BG.limit_of(sp, key)
        assert lv.value is None and not lv.usable and "invalid" in lv.why_not                 # never usable, whatever the confirmation
        from fea_core.space.requirements import confirmation_problems, validate
        assert any(key in str(i) for i in validate(sp.requirement_set) if i.level == "error")
        assert any(key in m for m in confirmation_problems(sp.requirement_set, "x"))        # the set cannot be confirmed with it
    assert not BG.limit_value_problem("f_min_eff_mass_fraction", 1.0) and not BG.limit_value_problem("mass_min_kg", 0.0)
    assert not BG.limit_value_problem("something_else", -1.0)                               # limits of other stages are not judged here
    f, fr = FREQ, FRAC
    rep = BG.frequency_budget(f, fr, "x", SL.space_with_limits({"f_min_axial_hz": -5.0, "f_min_lateral_hz": 0.0}, confirmed=False, env_axis="x"))
    assert rep.verdict == BG.UNCONFIRMED                                                    # ce3: a limit of 0 or below used to let every model PASS


# ====================================================================== I1: QSL status regenerates and compares
def _qsl_study(axis: str = "y", comb: str = "axial_lateral"):
    st = SL.steel_study("qsl review")
    st.space = SL.space_with_limits({"qsl_axial_g": 7.5, "qsl_lateral_g": 3.25})
    es = EV.environment_template()
    es.launch_axis, es.qsl_combination = axis, comb
    EV.store_environments(st.space, es)
    LD.generate_qsl(st)
    return st


def test_i1_qsl_status_follows_edits_deletions_axis_and_combination():
    st = _qsl_study()
    s = LD.qsl_status(st)
    assert s["current_with_requirement_set"] and not s["problems"] and s["requirement_set_confirmed"] and not s["environment_confirmed"]
    assert not s["usable_for_verification"]                                                 # the environment is not confirmed
    case = next(c for c in st.load_cases if c.name == "QSL +Y")
    case.loads[0].vector = [0.0, 0.1 * 9810.0, 0.0]                                          # ce4: the agent / the user weakens the case
    s = LD.qsl_status(st)
    assert not s["current_with_requirement_set"] and any("QSL +Y" in p and "edited" in p for p in s["problems"])
    LD.generate_qsl(st)                                                                     # regenerated: current again
    assert LD.qsl_status(st)["current_with_requirement_set"]
    es = EV.environments_of(st.space)
    es.launch_axis = "x"
    EV.store_environments(st.space, es)
    s = LD.qsl_status(st)
    assert not s["current_with_requirement_set"] and any("launch axis" in p for p in s["problems"])
    es.launch_axis = "y"
    es.qsl_combination = "axial_both_laterals"
    EV.store_environments(st.space, es)
    s = LD.qsl_status(st)
    assert not s["current_with_requirement_set"] and any("combination" in p for p in s["problems"])
    es.qsl_combination = "axial_lateral"
    EV.store_environments(st.space, es)
    assert LD.qsl_status(st)["current_with_requirement_set"]
    del st.load_cases[0]
    s = LD.qsl_status(st)
    assert not s["current_with_requirement_set"] and any("missing" in p for p in s["problems"])
    LD.remove_qsl(st)
    assert LD.qsl_status(st)["names"] == [] and not LD.qsl_status(st)["current_with_requirement_set"]


def test_i1_qsl_status_needs_confirmed_environment_and_requirement_set_for_use():
    st = _qsl_study()
    es = EV.environments_of(st.space)
    es.levels[0].random = EV.EnvTable("random", [[20.0, 0.01], [200.0, 0.03]], document="d", revision="r", clause="c")
    EV.confirm_environment(es, "Test Engineer")
    EV.store_environments(st.space, es)
    s = LD.qsl_status(st)
    assert s["environment_confirmed"] and s["requirement_set_confirmed"] and s["usable_for_verification"]
    for x in st.space.requirement_set.limits:
        if x.key == "qsl_axial_g":
            x.value = 8.0
    s = LD.qsl_status(st)
    assert not s["current_with_requirement_set"] and not s["usable_for_verification"]


def test_i1_qsl_factor_of_zero_or_below_is_refused():
    st = SL.steel_study("qsl zero")
    st.space = SL.space_with_limits({"qsl_axial_g": 0.0, "qsl_lateral_g": -3.0}, confirmed=False)
    es = EV.environment_template()
    es.launch_axis = "x"
    EV.store_environments(st.space, es)
    with pytest.raises(LD.LoadsError, match="invalid value"):
        LD.generate_qsl(st)
    assert not st.load_cases


def test_qsl_generator_limits_are_documented():
    doc = (LD.qsl_cases.__doc__ or "")
    for needle in ("ONE axial factor", "GLOBAL axes", "sqrt(2)", "0 or below is refused"):
        assert needle in doc, needle
    help_text = (Path(__file__).resolve().parents[1] / "fea_gui" / "help" / "space.html").read_text(encoding="utf-8")
    assert "&radic;2" in help_text and "axial_both_laterals" in help_text


# ====================================================================== C2: the kept result follows the model
def _kept(tmp_path, damping: bool = True):
    from fea_core.mcp.session import Session
    sess = Session(tmp_path / "mcp")
    sess.study = SL.steel_study("ce8")
    sess.mesh = SL.single_tet10()
    sess.mesh_sig = sess.mesh_signature()
    sess.study.space = SL.space_with_limits({"f_min_axial_hz": 80.0, "f_min_lateral_hz": 80.0}, env_axis="z", env_confirmed=False)
    es = EV.environments_of(sess.study.space)
    if damping:
        es.damping_zeta, es.damping_source = 0.03, "test report 1"
    EV.confirm_environment(es, "Test Engineer")
    EV.store_environments(sess.study.space, es)
    lv = es.levels[0]
    basis = SL.synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
    per, _ = LD.analyze_axes(sess.study, sess.mesh, basis, SL.synthetic_statics(basis), tmp_path, lambda s: None, lv, ["random"])
    r3 = LD.build_result(sess.study, sess.mesh, lv, ["random"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True, env_digest=es.content_digest(),
                         run_digest=LD.compute_run_digest(sess.study, sess.mesh))
    LD.remember(sess, r3)
    return sess, r3


def test_c2_a_changed_model_never_recalls_the_old_result(tmp_path):
    """ce8: E / 100 and zeta 0.02 -> 0.001 followed by sess.changed(): the 3-axis result (and its frequencies) was still used and 'confirmed'."""
    sess, r3 = _kept(tmp_path)
    assert LD.recall(sess, sess.study, sess.mesh) is r3 and r3.current(sess.study, sess.mesh)
    mat = next(iter(sess.study.materials.values()))
    e0 = mat.E
    mat.E = e0 / 100.0                                                                       # a change that does NOT call Session.changed(): the digest alone must catch it
    assert LD.recall(sess, sess.study, sess.mesh) is None and not r3.current(sess.study, sess.mesh)
    assert "study, the mesh or the damping" in r3.stale_reason(sess.study, sess.mesh)
    mat.E = e0
    assert LD.recall(sess, sess.study, sess.mesh) is r3                                      # back to the same inputs: the same digest
    sess.study.dynamics.damping.zeta = 0.001                                                 # the study damping is not used when the environment has its own: digest of the
    assert r3.current(sess.study, sess.mesh) is False                                        # whole study dict still changes - conservative
    sess.study.dynamics.damping.zeta = 0.02
    sess.changed()                                                                           # the explicit clearing
    assert sess._space_3axis is None and LD.recall(sess, sess.study, sess.mesh) is None


def test_c2_other_studies_meshes_and_environments_are_not_matched(tmp_path):
    sess, r3 = _kept(tmp_path)
    other = SL.steel_study("ce8")                                                            # the same NAME, another study (no match by name any more)
    assert LD.recall(sess, other, sess.mesh) is None
    assert LD.recall(sess, sess.study, SL.box_mesh_tet10(40.0, 20.0, 10.0)) is None         # another mesh
    assert LD.recall(sess, sess.study, None) is None                                         # no mesh to compare with: never current
    es = EV.environments_of(sess.study.space)
    es.levels[0].random.sigma = 4.0                                                          # an environment edit
    EV.store_environments(sess.study.space, es)
    assert LD.recall(sess, sess.study, sess.mesh) is None and "environments were edited" in r3.stale_reason(sess.study, sess.mesh)
    assert not r3.confirmed_now(sess.study, sess.mesh)


def test_c2_the_digest_covers_the_environment_damping_and_mesh(tmp_path):
    sess, _ = _kept(tmp_path)
    d0 = LD.compute_run_digest(sess.study, sess.mesh)
    assert LD.compute_run_digest(sess.study, sess.mesh) == d0
    from fea_core.study import Study
    assert LD.compute_run_digest(Study.from_dict(json.loads(json.dumps(sess.study.to_dict()))), sess.mesh) == d0                 # a copy / a reloaded project is the same study
    es = EV.environments_of(sess.study.space)
    es.damping_zeta = 0.05
    EV.store_environments(sess.study.space, es)
    assert LD.compute_run_digest(sess.study, sess.mesh) != d0
    m2 = SL.single_tet10()
    m2.nodes = m2.nodes + 1e-9
    assert LD.mesh_fingerprint(m2) != LD.mesh_fingerprint(SL.single_tet10())
    es.damping_zeta = 0.03
    es.levels[0].combine = "srss" if es.levels[0].combine == "cqc" else "cqc"
    EV.store_environments(sess.study.space, es)
    assert LD.compute_run_digest(sess.study, sess.mesh) != d0                                # the combination rule is part of the environment content


def test_c2_the_budget_tool_does_not_use_a_stale_basis(tmp_path):
    """ce8: fea_space_mass_budget took the frequencies of the stale basis and judged them 'against the limits'."""
    from tests.test_space_loads import _run
    sess, r3 = _kept(tmp_path)
    sess.mesh = SL.box_mesh_tet10(40.0, 20.0, 10.0)                                          # the budget tool needs a mesh; the kept result belongs to another one
    sess.mesh_sig = sess.mesh_signature()
    o = _run(sess, [("fea_space_mass_budget", {}, False)])[0]
    assert o["frequency_check"]["verdict"] == "SKIPPED"                                      # the result of the other mesh is not used
    sess.mesh = SL.single_tet10()
    sess.mesh_sig = sess.mesh_signature()
    o = _run(sess, [("fea_space_mass_budget", {}, False)])[0]
    assert o["frequency_check"]["lowest_Hz"][:3] == [85.0, 130.0, 210.0]
    assert o["environment_confirmed"] is True and o["frequency_check"]["verdict"] == "PASS"             # confirmed set, confirmed environment, f_1 = 85 Hz >= 80 Hz


def test_c2_the_result_is_attached_to_the_run_and_results_are_checked_too(tmp_path):
    sess, r3 = _kept(tmp_path)
    sess.results = SimpleNamespace(mesh=sess.mesh)
    LD.remember(sess, r3)
    assert LD.attached(sess.results) is r3
    LD.forget(sess)
    assert LD.recall(sess, sess.study, sess.mesh) is r3                                      # the copy on the results (they are discarded with the study) is still valid
    mat = next(iter(sess.study.materials.values()))
    mat.E *= 2
    assert LD.recall(sess, sess.study, sess.mesh) is None


# ====================================================================== C3: every verdict goes through the envelopes
def _margin_fixture(tmp_path, s_value: float, blockers: list[str] | None = None):
    from fea_core.mcp.session import Session
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("protoflight"))
    lv = SL.three_axis_level()
    es = EV.EnvironmentSet(levels=[lv], launch_axis="z")
    EV.confirm_environment(es, "Test Engineer", "2026-01-01")
    EV.store_environments(st.space, es)
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA)])
    n = len(mesh.nodes)
    f = LD.AxisField(np.full(n, s_value, np.float32), np.zeros(n, np.float32), None)
    per = {"sine": {0: f, 1: f, 2: f}}
    r3 = LD.ThreeAxisResult("qualification", ["sine"], [0, 1, 2], per, {"sine": LD.envelope_of(per["sine"], "sine")}, {}, np.array([100.0]), np.zeros((1, 3)), None,
                            True, True, level_kinds=["sine"], damping={"kind": "modal", "zeta": 0.02, "source": "t", "from_environment": True}, blockers=list(blockers or []))
    sess = Session(tmp_path / "m")
    sess.study, sess.mesh, sess.results = st, mesh, res
    sess.mesh_sig = sess.mesh_signature()
    LD.remember(sess, r3)
    r3.env_digest = es.content_digest()
    r3.run_digest = LD.compute_run_digest(st, mesh)
    return sess, st, res, mesh, r3


def test_c3_mcp_margins_report_and_panel_verdict_see_a_failing_envelope(tmp_path):
    """ce9: compute_margins (what fea_space_margins, the margins tab and the report chapter used) said PASS while the 3-axis envelope was 10 x Fy."""
    from fea_core.report import space as RS
    from fea_core.report.word import _Doc
    from tests.test_space_loads import _run
    pytest.importorskip("docx")
    sess, st, res, mesh, r3 = _margin_fixture(tmp_path, 10.0 * SC.TEST_FY)
    assert MG.compute_margins(st, res).verdict == MG.PASS                                     # without the envelope: PASS (the reason for the finding)
    assert LD.sign_off_margins(st, res, sess, mesh).verdict == MG.FAIL
    o = _run(sess, [("fea_space_margins", {}, False)])[0]
    assert o["verdict"] == "FAIL" and any(r["case"].startswith("3-axis") for r in o["lowest"])
    D = _Doc()
    RS.space_section(D, sess)
    text = "\n".join(p.text for p in D.doc.paragraphs)
    assert "Verdict: FAIL" in text and "3-axis qualification sine envelope" in "\n".join(c.text for t in D.doc.tables for row in t.rows for c in row.cells)
    # the kept result of ANOTHER model state is not used, but it is not forgotten either (N1): the plain margins would be a static PASS, so the verdict says that
    # the 3-axis result is out of date and cannot be PASS
    mat = next(iter(st.materials.values()))
    mat.E *= 2
    s = LD.sign_off_margins(st, res, sess, mesh)
    assert s.verdict == MG.UNCONFIRMED and any("3-axis result is out of date" in r and "re-run" in r for r in s.reasons)
    assert not any(r.case.startswith("3-axis") for r in s.rows)                              # the stale envelope itself is not shown as a margin
    mat.E /= 2
    assert LD.sign_off_margins(st, res, sess, mesh).verdict == MG.FAIL                      # the study is the one of the run again: the failing envelope is back


def test_c3_a_blocked_result_is_never_a_pass_in_any_verdict(tmp_path):
    from fea_core.report import space as RS
    from fea_core.report.word import _Doc
    from tests.test_space_loads import _run
    pytest.importorskip("docx")
    sess, st, res, mesh, r3 = _margin_fixture(tmp_path, 10.0, ["the damping is not entered in the environments (it is the damping setting of the study)"])
    s = LD.sign_off_margins(st, res, sess, mesh)
    assert s.verdict == MG.UNCONFIRMED and any("damping" in r for r in s.reasons)
    assert all(r.status == "unconfirmed" for r in s.rows if r.case.startswith("3-axis"))
    o = _run(sess, [("fea_space_margins", {}, False)])[0]
    assert o["verdict"] == MG.UNCONFIRMED and any("damping" in x for x in o["reasons"])
    D = _Doc()
    RS.space_section(D, sess)
    assert "Verdict: UNCONFIRMED_REQUIREMENTS" in "\n".join(p.text for p in D.doc.paragraphs)
    r3.blockers = []                                                                         # the same result without blockers is the PASS it should be
    assert LD.sign_off_margins(st, res, sess, mesh).verdict == MG.PASS


def test_c3_a_negative_margin_stays_fail_even_with_blockers(tmp_path):
    sess, st, res, mesh, r3 = _margin_fixture(tmp_path, 10.0 * SC.TEST_FY, ["partial run"])
    s = LD.sign_off_margins(st, res, sess, mesh)
    assert s.verdict == MG.FAIL                                                              # a blocker forbids a PASS, it does not hide a negative margin


# ====================================================================== I2: the shock table against the modes
def _shock_level(points):
    lv = EV.EnvLevel("qualification", combine="srss", missing_mass=True)
    lv.shock = EV.EnvTable("shock", points, document="t", revision="-", clause="c")
    return lv


def test_i2_modes_below_the_srs_table_warn_and_block_a_pass(tmp_path):
    """ce5: a 40 / 60 Hz mode below a table that starts at 100 Hz responds with ZERO and nothing said so."""
    mesh = SL.single_tet10()
    st = SL.steel_study("ce5")
    basis = SL.synthetic_basis([40.0, 60.0, 5000.0], [0, 1, 2])
    lv = _shock_level([[100.0, 20.0], [1000.0, 200.0]])
    per, _ = LD.analyze_axes(st, mesh, basis, SL.consistent_statics(basis), tmp_path, lambda s: None, lv, ["shock"])
    res = LD.build_result(st, mesh, lv, ["shock"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    assert any("below it respond with ZERO" in b and "mode 1 (40 Hz" in b and "mode 2 (60 Hz" in b for b in res.blockers)
    assert any("ZERO" in w for w in res.warnings)                                            # a warning, not only a blocker
    assert any("above the top of the shock table" in n and "mode 3 (5000 Hz" in n for n in res.notes)       # the 5000 Hz mode takes the ZPA: said in a note
    # N2: NO mass-share threshold: a mode with a negligible share of the effective mass below the table blocks too (a local mode can carry the stress of a part)
    b2 = SL.synthetic_basis([40.0, 60.0, 5000.0], [0, 1, 2])
    b2.eff_mass[0, :] = b2.eff_mass[0, :] * 1e-4
    blk, notes = LD.srs_range_findings(b2.freq, LD.eff_fractions(b2), lv.shock, [0, 1, 2])
    assert len(blk) == 1 and "mode 1 (40 Hz" in blk[0] and "mode 2 (60 Hz" in blk[0] and "above the top" in notes[0]
    assert not hasattr(LD, "SRS_SIGNIFICANT_FRACTION")                                        # the threshold is gone
    # the blocker is removed by extending the table down to the lowest mode - not by anything else
    lv_ext = _shock_level([[30.0, 20.0], [100.0, 20.0], [1000.0, 200.0]])
    assert LD.srs_range_findings(b2.freq, LD.eff_fractions(b2), lv_ext.shock, [0, 1, 2])[0] == []
    # modes inside the range: nothing to say
    b3 = SL.synthetic_basis([150.0, 230.0, 310.0], [0, 1, 2])
    assert LD.srs_range_findings(b3.freq, LD.eff_fractions(b3), lv.shock, [0, 1, 2]) == ([], [])


def test_i2_many_small_modes_below_the_table_all_block():
    """rc probe D: 12 modes of 0.9 % each under the table (a global threshold of 1 % would have passed every one of them)."""
    lv = _shock_level([[100.0, 20.0], [1000.0, 200.0]])
    K = 12
    bm = SL.synthetic_basis([30.0 + k for k in range(K)] + [300.0], [0] * (K + 1))
    tot = float(bm.total_mass[0])
    bm.eff_mass[:K, 0] = 0.009 * tot
    blk, _ = LD.srs_range_findings(bm.freq, LD.eff_fractions(bm), lv.shock, [0])
    assert len(blk) == 1 and "12 mode(s) below it" in blk[0] and "..." in blk[0]
    # a rigid-body / zero frequency is not a mode
    assert LD.srs_range_findings(np.array([0.0, 300.0]), np.array([[0.5, 0, 0], [0.5, 0, 0]]), lv.shock, [0])[0] == []
    # unknown fractions (no total mass) do not hide the modes
    bz = SL.synthetic_basis([40.0, 150.0], [0, 0])
    bz.total_mass, bz.model_mass = np.zeros(6), 0.0
    assert len(LD.srs_range_findings(bz.freq, LD.eff_fractions(bz), lv.shock, [0])[0]) == 1


def test_i2_modes_above_the_table_take_the_zpa_in_the_space_path_only(tmp_path):
    from fea_core.dynamics.run import run_dynamics
    mesh = SL.single_tet10()
    st = SL.steel_study("ce5b")
    basis = SL.synthetic_basis([40.0, 60.0, 5000.0], [0, 1, 2])
    stat = SL.consistent_statics(basis)
    lv = _shock_level([[100.0, 20.0], [1000.0, 200.0]])
    zpa = 200.0 * 9810.0
    ext = run_dynamics(LD._axis_study(st, lv, ["shock"], 2, srs_extend_to=5050.0), mesh, tmp_path / "e", lambda s: None, basis=basis, statics_by_key=stat)
    plain = run_dynamics(LD._axis_study(st, lv, ["shock"], 2), mesh, tmp_path / "p", lambda s: None, basis=basis, statics_by_key=stat)       # the single-axis behaviour
    assert ext.shock["Sa"][2] == pytest.approx(zpa, rel=1e-12) and plain.shock["Sa"][2] == 0.0
    assert ext.shock["Sa"][0] == 0.0 and ext.shock["Sa"][1] == 0.0                           # below the table: zero in both (and blocked by the result)
    per, _ = LD.analyze_axes(st, mesh, basis, stat, tmp_path, lambda s: None, lv, ["shock"], (2,))
    assert np.array_equal(per["shock"][2].S, np.asarray(ext.shock["mises"], np.float32))      # analyze_axes extends the table by itself


# ====================================================================== I3: damage is summed over the axes
def test_i3_the_damage_of_the_axes_adds_up(tmp_path):
    mesh = SL.single_tet10()
    basis = SL.synthetic_basis([85.0, 86.0, 87.0], [0, 1, 2])
    for k in range(3):                                                                       # every axis loads the same nodes equally (ce7)
        basis.stress[k] = 0.0
        basis.stress[k, :, 0] = 4000.0
    st = SL.steel_study("ce7")
    st.dynamics.damping.zeta = 0.03
    lv = SL._level_random([[20.0, 0.5], [2000.0, 0.5]], duration_s=120.0)
    per, _ = LD.analyze_axes(st, mesh, basis, SL.consistent_statics(basis), tmp_path, lambda s: None, lv, ["random"])
    env = LD.envelope_of(per["random"], "random")
    d = [per["random"][a].D.astype(np.float64) for a in (0, 1, 2)]
    assert d[0].max() > 0
    assert np.allclose(env.D, (d[0] + d[1] + d[2]), rtol=1e-6) and env.D.max() > 2.5 * max(x.max() for x in d)       # ~3 x one axis, not the maximum
    assert np.array_equal(env.D_axis_max, np.max(np.stack(d), axis=0).astype(np.float32))                             # the plan's max(D), kept as information
    res = LD.build_result(st, mesh, lv, ["random"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    kd = res.summary()["kinds"]["random"]
    assert kd["max_D"] == pytest.approx(float(env.D.max())) and kd["max_D_single_axis"] == pytest.approx(float(env.D_axis_max.max()))
    assert "sum over the axes" in kd["damage_rule"] and set(kd["per_axis_max_D"]) == {"X", "Y", "Z"}
    p = next(iter(res.parts["random"].values()))
    assert p["max_D"] > p["max_D_single_axis"]
    one = LD.envelope_of({0: per["random"][0]}, "random")                                    # a single axis: sum = max
    assert np.array_equal(one.D, per["random"][0].D)


# ====================================================================== I4: damping is an environment input
def test_i4_damping_is_part_of_the_environment_and_its_confirmation():
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z")
    assert es.damping_zeta is None
    EV.confirm_environment(es, "Test Engineer", "2026-01-01")
    es.damping_zeta, es.damping_source = 0.03, "qualification test report"
    assert es.confirmed and not es.is_confirmed()                                            # an edit of the damping withdraws the confirmation
    assert EV.validate_environment(es) == []
    es.damping_source = ""
    assert any("damping_source" in p for p in EV.validate_environment(es))                   # a value without its source cannot be confirmed
    with pytest.raises(Exception, match="damping_source"):
        EV.confirm_environment(es, "Test Engineer")
    es.damping_source, es.damping_zeta = "x", 1.5
    assert any("damping_zeta" in p for p in EV.validate_environment(es))
    es.damping_zeta = 0.0
    assert any("damping_zeta" in p for p in EV.validate_environment(es))
    es.damping_zeta = 0.03
    back = EV.from_json(EV.to_json(es))
    assert back.damping_zeta == 0.03 and back.damping_source == "x"


def test_i4_the_run_uses_the_environment_damping_and_says_so(tmp_path):
    st = SL.steel_study("i4")
    st.space = SpaceSettings()
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z")
    EV.store_environments(st.space, es)
    spec, info, warn = LD.effective_damping(st)                                              # not set: the study damping with a warning
    assert info["from_environment"] is False and spec.zeta == st.dynamics.damping.zeta and warn and "not entered in the environments" in warn[0]
    es.damping_zeta, es.damping_source = 0.05, "test report"
    EV.store_environments(st.space, es)
    spec, info, warn = LD.effective_damping(st)
    assert info == {"kind": "modal", "zeta": 0.05, "source": "test report", "from_environment": True} and spec.zeta == 0.05 and spec.kind == "modal" and not warn
    assert LD._axis_study(st, es.levels[0], ["random"], 0).dynamics.damping.zeta == 0.05    # the run uses it, not the study damping (0.02)
    mesh = SL.single_tet10()
    basis = SL.synthetic_basis([150.0, 230.0, 310.0], [0, 1, 2])
    lv = es.levels[0]
    per, w = LD.analyze_axes(st, mesh, basis, SL.consistent_statics(basis), tmp_path, lambda s: None, lv, ["random"])
    res = LD.build_result(st, mesh, lv, ["random"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    assert res.damping["zeta"] == 0.05 and not any("damping" in b for b in res.blockers)
    assert res.summary()["damping"]["source"] == "test report" and any("damping: modal, zeta 0.05" in n for n in res.notes)
    es.damping_zeta = None
    EV.store_environments(st.space, es)                                                      # without it: warning and a blocker
    per, _ = LD.analyze_axes(st, mesh, basis, SL.consistent_statics(basis), tmp_path, lambda s: None, lv, ["random"])
    res2 = LD.build_result(st, mesh, lv, ["random"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    assert any("damping" in b for b in res2.blockers) and any("not entered in the environments" in w for w in res2.warnings)


def test_i4_the_report_and_the_summary_show_the_damping():
    pytest.importorskip("docx")
    from fea_core.report import space_loads as RL
    from fea_core.report.word import _Doc
    st = SL.steel_study("i4 report")
    st.space = SpaceSettings()
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z", damping_zeta=0.04, damping_source="test report 7")
    EV.store_environments(st.space, es)
    D = _Doc()
    RL.environment_tables(D, es)
    text = "\n".join([p.text for p in D.doc.paragraphs] + [c.text for t in D.doc.tables for row in t.rows for c in row.cells])
    assert "Modal damping ratio of the launch analysis" in text and "test report 7" in text and "0.04" in text
    es.damping_zeta = None
    D = _Doc()
    RL.environment_tables(D, es)
    assert "NOT entered in the environments" in "\n".join(c.text for t in D.doc.tables for row in t.rows for c in row.cells)


# ====================================================================== I5: silent omissions
def test_i5_random_without_sigma_blocks_a_pass(tmp_path):
    mesh = SL.single_tet10()
    st = SL.steel_study("i5")
    basis = SL.synthetic_basis([150.0, 230.0, 310.0], [0, 1, 2])
    lv = SL._level_random([[20.0, 0.01], [2000.0, 0.01]], sigma=None)
    per, _ = LD.analyze_axes(st, mesh, basis, SL.consistent_statics(basis), tmp_path, lambda s: None, lv, ["random"])
    res = LD.build_result(st, mesh, lv, ["random"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    assert any("no sigma" in b and "NOT compared" in b for b in res.blockers) and res.scalar_sources() == []
    sess_dir = Path(tmp_path)
    sess, st2, res2, mesh2, r3 = _margin_fixture(sess_dir, 1.0, list(res.blockers))
    s = LD.sign_off_margins(st2, res2, sess, mesh2)
    assert s.verdict == MG.UNCONFIRMED and any("sigma" in r for r in s.reasons)             # ce9: the margins said PASS without the random environment


def test_i5_a_partial_run_is_labelled_and_never_called_3_axis(tmp_path):
    mesh = SL.single_tet10()
    st = SL.steel_study("i5b")
    basis = SL.synthetic_basis([150.0, 230.0, 310.0], [0, 1, 2])
    lv = SL.three_axis_level()
    full, _ = LD.analyze_axes(st, mesh, basis, SL.consistent_statics(basis), tmp_path, lambda s: None, lv, ["sine", "random", "shock"])
    r_full = LD.build_result(st, mesh, lv, ["sine", "random", "shock"], [0, 1, 2], full, basis, env_confirmed=True, req_confirmed=True)
    assert r_full.complete and r_full.label == "3-axis" and not any("partial" in b for b in r_full.blockers)
    assert all(s.name.startswith("3-axis qualification") for s in r_full.scalar_sources())
    r_axes = LD.build_result(st, mesh, lv, ["sine", "random", "shock"], [0, 2], {k: {a: v[a] for a in (0, 2)} for k, v in full.items()}, basis,
                             env_confirmed=True, req_confirmed=True)
    assert not r_axes.complete and r_axes.label == "X+Z-axis only" and any("partial run - axes X, Z only" in b for b in r_axes.blockers)
    assert all(s.name.startswith("X+Z-axis only qualification") and "3-axis" not in s.name for s in r_axes.scalar_sources())
    assert r_axes.summary()["label"] == "X+Z-axis only" and r_axes.summary()["complete"] is False
    r_kinds = LD.build_result(st, mesh, lv, ["random"], [0, 1, 2], {"random": full["random"]}, basis, env_confirmed=True, req_confirmed=True)
    assert not r_kinds.complete and "kinds random only" in r_kinds.label and any("kinds random only (the level also has sine, shock)" in b for b in r_kinds.blockers)
    assert all("3-axis" not in s.name and "random only" in s.name for s in r_kinds.scalar_sources())


# ====================================================================== I6: mass properties as the solver sees them
def _mass_mesh():
    return SL.box_mesh_tet10(40.0, 20.0, 10.0)


def test_i6_rotary_inertia_only_for_a_rigid_coupling():
    """ce3: the solver keeps only the translation of a distributing point mass; the inertia tensor of the report must not contain its rotary inertia."""
    mesh = _mass_mesh()
    I0 = [[3.0e1, 0, 0], [0, 4.0e1, 0], [0, 0, 5.0e1]]
    res = {}
    for coupling in ("rigid", "distributing"):
        s = SL.steel_study("i6")
        s.masses.append(PointMass("Support", 2.0e-3, point=[55.0, 3.0, 12.0], inertia=I0, coupling=coupling, name="unit"))
        res[coupling] = MP.mass_properties(s, mesh)
    diff = np.diag(res["rigid"].inertia_cog - res["distributing"].inertia_cog)
    assert diff == pytest.approx([30.0, 40.0, 50.0], rel=1e-9)                                # exactly the own rotary inertia
    assert any("keeps only the translation" in n and "NOT part of the inertia tensor" in n for n in res["distributing"].notes)
    assert not res["rigid"].notes and res["distributing"].added[0]["rotary_inertia"] is False and res["rigid"].added[0]["rotary_inertia"] is True
    assert res["rigid"].mass_t == res["distributing"].mass_t                                  # the mass and the centre of mass do not depend on it
    # a distributing mass without inertia has nothing to neglect
    s = SL.steel_study("i6b")
    s.masses.append(PointMass("Support", 2.0e-3, point=[55.0, 3.0, 12.0], coupling="distributing", name="plain"))
    assert not MP.mass_properties(s, mesh).notes


def test_i6_nonpositive_and_nonphysical_added_masses_are_refused():
    mesh = _mass_mesh()
    for m in (0.0, -5.0e-3, float("nan")):
        s = SL.steel_study("i6c")
        s.masses.append(PointMass("Support", m, point=[55.0, 3.0, 12.0], name="bad"))
        with pytest.raises(ValueError, match="mass must be a positive number"):
            MP.mass_properties(s, mesh)
    s = SL.steel_study("i6d")
    s.masses.append(PointMass("Support", 2.0e-3, point=[55.0, 3.0, 12.0], inertia=[1.0, 1.0, 100.0], name="bad"))      # Izz > Ixx + Iyy
    with pytest.raises(ValueError, match="triangle inequality"):
        MP.mass_properties(s, mesh)
    s = SL.steel_study("i6e")
    s.masses.append(PointMass("Support", 2.0e-3, point=[55.0, 3.0, 12.0], inertia=[1.0, float("inf"), 1.0], name="bad"))
    with pytest.raises(ValueError, match="finite"):
        MP.mass_properties(s, mesh)
    s = SL.steel_study("i6f")
    s.masses.append(PointMass("Support", 2.0e-3, point=[55.0, 3.0, 12.0], inertia=[3.0, 4.0, 5.0], coupling="distributing", name="ok"))   # a physical tensor passes
    assert MP.mass_properties(s, mesh).mass_t > 0


def test_i6_the_mcp_tool_reports_a_rejected_mass_as_an_error(tmp_path):
    from tests.test_space_loads import _run, _session
    sess = _session(tmp_path)
    sess.study.masses.append(PointMass("Support", -1.0e-3, point=[55.0, 3.0, 12.0], name="neg"))
    o = _run(sess, [("fea_space_mass_budget", {}, True)])[0]
    assert "positive number" in o


def test_s3_uses_the_same_mass_as_s2():
    """One report cannot show two masses: the model check (S3 check 1) is the S2 integration in another form."""
    from fea_core.space import _checks_mass as CM
    mesh = _mass_mesh()
    s = SL.steel_study("s3 mass")
    s.masses.append(PointMass("Support", 2.0e-3, point=[55.0, 3.0, 12.0], inertia=[3.0, 4.0, 5.0], name="unit"))
    p2 = MP.mass_properties(s, mesh)
    p3 = CM.mass_properties(s, mesh)
    assert p3.mass_t == p2.mass_t and p3.bodies_t == p2.bodies_t and p3.added_t == p2.added_t and np.array_equal(p3.cog, p2.cog) and p3.volume_mm3 == p2.volume_mm3
    assert p3.parts["Block"]["rho_t_mm3"] == pytest.approx(7.85e-9) and p3.parts["Block"]["elements"] == len(mesh.tets)
    assert p3.added[0]["mass_t"] == pytest.approx(2.0e-3) and p3.added[0]["point_mm"] == p2.added[0]["point_mm"]
    with pytest.raises(ValueError, match="tet10"):
        CM.mass_properties(s, SC.box_tet_mesh())


# ====================================================================== minor: resolve_level, environments
def _env_study(combine: str = "srss"):
    st = SL.steel_study("resolve")
    st.space = SpaceSettings()
    lv = SL.three_axis_level()
    lv.combine = combine
    EV.store_environments(st.space, EV.EnvironmentSet(levels=[lv], launch_axis="z"))
    return st


def test_resolve_level_rejects_bad_combine_levels_and_kinds_and_accepts_missing_sources():
    st = _env_study("abs")
    with pytest.raises(LD.LoadsError, match="srss or cqc"):
        LD.resolve_level(st, "qualification")                                                # ce6: 'abs' was accepted
    LD.resolve_level(st, "qualification", ["sine", "random"])                                # the shock is not run: its combination does not matter
    st = _env_study()
    with pytest.raises(LD.LoadsError, match="unknown level"):
        LD.resolve_level(st, "flight")
    with pytest.raises(LD.LoadsError, match="unknown kind"):
        LD.resolve_level(st, "qualification", ["sine", "bogus"])                             # ce2: silently dropped
    with pytest.raises(LD.LoadsError, match="has no table"):
        LD.resolve_level(_env_study(), "protoflight")
    # a table without document / revision / clause computes (UNCONFIRMED) instead of being refused
    es = EV.environments_of(st.space)
    es.levels[0].sine.document = es.levels[0].sine.clause = ""
    EV.store_environments(st.space, es)
    _es, lv, use = LD.resolve_level(st, "qualification")
    assert use == ["sine", "random", "shock"] and not lv.sine.provenance_ok
    es.levels[0].sine.points = [[100.0, 5.0], [50.0, 5.0]]                                   # a structurally wrong table is still refused
    EV.store_environments(st.space, es)
    with pytest.raises(LD.LoadsError, match="increase"):
        LD.resolve_level(st, "qualification")


def test_environments_reject_infinite_and_oversized_input_cleanly():
    from fea_core.space.model import RequirementError
    for bad in (float("inf"), float("nan"), 2.5, "1e999"):
        with pytest.raises(ValueError):
            EV.EnvTable.from_dict({"kind": "sine", "points": [[1, 1], [2, 1]], "sweeps": bad})
    with pytest.raises(ValueError, match="finite"):
        EV.EnvTable.from_dict({"kind": "random", "points": [[1, 1], [float("inf"), 1]]})
    with pytest.raises(ValueError, match="at most"):
        EV.EnvTable.from_dict({"kind": "sine", "points": [[1, 1], [2, 1]], "document": "x" * 100_000})
    with pytest.raises(ValueError, match="at most"):
        EV.EnvironmentSet.from_dict({"notes": "x" * 2_000_000})
    with pytest.raises(RequirementError):
        EV.from_json('{"levels": [{"name": "qualification", "sine": {"points": [[1, 1], [2, 1]], "sweeps": 1e999}}]}')
    sp = SpaceSettings()
    sp.extra["environments"] = {"levels": [{"name": "qualification", "sine": {"points": [[1, 1], [2, 1]], "sweeps": float("inf")}}]}        # a stored record
    assert EV.environments_of(sp) is None                                                    # unreadable, not a crash (ce2 B: OverflowError)


def test_mcp_environment_set_rejects_inf_long_text_and_unknown_axes(tmp_path):
    from tests.test_space_loads import _run, _session
    sess = _session(tmp_path)
    tbl = {"points": [[5, 1], [100, 1]], "document": "d", "revision": "r", "clause": "c"}
    o = _run(sess, [("fea_space_env_set", {"action": "template"}, False),
                    ("fea_space_env_set", {"action": "set", "data": {"levels": [{"name": "qualification", "sine": {**tbl, "sweeps": "1e999"}}]}}, True),
                    ("fea_space_env_set", {"action": "set", "data": {"notes": "x" * 2_000_000}}, True),
                    ("fea_space_env_set", {"action": "set", "data": {"levels": [{"name": "qualification", "sine": {**tbl, "note": "y" * 100_000}}]}}, True),
                    ("fea_space_env_set", {"action": "set", "data": {"damping_zeta": 0.03, "damping_source": "test report"}}, False),
                    ("fea_space_env_set", {"action": "set", "data": {"damping_zeta": "x"}}, True),
                    ("fea_space_run_3axis", {"level": "qualification", "kinds": ["sine", "bogus"]}, True),
                    ("fea_space_run_3axis", {"level": "qualification", "axes": "xq"}, True)])
    assert "Invalid data" in o[1] and "Invalid data" in o[2] and "at most" in o[3]
    assert o[4]["environments"]["damping_zeta"] == 0.03 and o[4]["environments"]["damping_source"] == "test report"
    assert "Invalid data" in o[5] and ("axes" in o[7])


# ====================================================================== minor: run_dynamics with a given basis
def test_run_dynamics_with_a_given_basis_is_checked(tmp_path):
    from fea_core.dynamics.modal import DynamicsError
    from fea_core.dynamics.run import run_dynamics
    mesh = SL.single_tet10()
    st = SL.steel_study("rd")
    basis = SL.synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
    lv = _shock_level([[50.0, 20.0], [1000.0, 200.0]])
    st1 = LD._axis_study(st, lv, ["shock"], 0)
    stat = SL.consistent_statics(basis)
    ok = run_dynamics(st1, mesh, tmp_path / "ok", lambda s: None, basis=basis, statics_by_key=stat)
    assert ok.shock is not None
    with pytest.raises(DynamicsError, match="statics_by_key"):
        run_dynamics(st1, mesh, tmp_path / "a", lambda s: None, basis=basis)
    with pytest.raises(DynamicsError, match="static solutions.*miss"):
        run_dynamics(st1, mesh, tmp_path / "b", lambda s: None, basis=basis, statics_by_key={})
    with pytest.raises(DynamicsError, match="another mesh"):
        run_dynamics(st1, SL.box_mesh_tet10(40.0, 20.0, 10.0), tmp_path / "c", lambda s: None, basis=basis, statics_by_key=stat)
    st2 = LD._axis_study(st, lv, ["shock"], 0)
    st2.dynamics.prestress.enabled = True
    with pytest.raises(DynamicsError, match="prestress"):
        run_dynamics(st2, mesh, tmp_path / "d", lambda s: None, basis=basis, statics_by_key=stat)


# ====================================================================== minor: the panel does not round, the report prints the tables
def test_panel_text_is_repr_exact():
    pytest.importorskip("PySide6")
    from fea_gui.space_loads_panel import exact
    for v in (0.1, 1234567.891234, 1e-05, 123456789.0123, 0.30000000000000004, 7.85e-9, 5.0, 100.0):
        assert float(exact(v)) == v, v
    assert exact(5.0) == "5" and exact(2) == "2" and exact(None) == "" and exact("text") == "text" and exact(0.1) == "0.1"


def test_report_prints_the_table_values_not_only_the_range():
    pytest.importorskip("docx")
    from fea_core.report import space_loads as RL
    from fea_core.report.word import _Doc
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z")
    D = _Doc()
    RL.environment_tables(D, es)
    text = "\n".join(p.text for p in D.doc.paragraphs)
    assert "qualification / random table [Hz; g2/Hz]: 20: 0.01; 200: 0.03; 1000: 0.03" in text
    assert "qualification / sine table [Hz; g]: 20: 5; 100: 5; 400: 8" in text
    assert "qualification / shock table [Hz; g]: 100: 20; 1000: 200" in text


def test_report_qsl_section_names_the_generator_limits_and_the_problems():
    pytest.importorskip("docx")
    from fea_core.report import space_loads as RL
    from fea_core.report.word import _Doc
    st = _qsl_study()
    next(c for c in st.load_cases if c.name == "QSL +Y").loads[0].vector = [0.0, 1.0, 0.0]
    D = _Doc()
    RL.qsl_table(D, st)
    text = "\n".join(p.text for p in D.doc.paragraphs)
    assert "NOT current" in text and "QSL +Y" in text and "Generator limits" in text and "one axial factor for both signs" in text


# ====================================================================== F14-S2 re-check: digest coverage, stale results, damping range, QSL csys, mesh surfaces
def _digest_study():
    st = SL.steel_study("dig")
    st.space = SL.space_with_limits({"f_min_axial_hz": 80.0}, env_axis="z", env_confirmed=False)
    es = EV.environments_of(st.space)
    es.damping_zeta, es.damping_source = 0.03, "t"
    EV.store_environments(st.space, es)
    return st


def test_n1_the_digest_ignores_what_the_dynamics_does_not_use_and_follows_the_rest():
    """rc probe B: renaming the study, the thread count or a static load case (the QSL cases the generator adds) made a valid 3-axis result stale; a physical edit must."""
    from fea_core.study import Fixed, Gravity, LoadCase
    mesh = SL.single_tet10()
    d0 = LD.compute_run_digest(_digest_study(), mesh)
    assert LD.compute_run_digest(_digest_study(), mesh) == d0

    def digest_after(fn) -> str:
        s = _digest_study()
        fn(s)
        return LD.compute_run_digest(s, mesh)

    def env_edit(fn):
        def f(s):
            es = EV.environments_of(s.space)
            fn(es)
            EV.store_environments(s.space, es)
        return f

    def set_mat(attr: str, mul: float):
        return lambda s: setattr(next(iter(s.materials.values())), attr, getattr(next(iter(s.materials.values())), attr) * mul)

    # NOT part of the dynamic result
    same = {"study name": lambda s: setattr(s, "name", "renamed"),
            "solver.nproc": lambda s: setattr(s.solver, "nproc", 4),
            "a static load case": lambda s: s.load_cases.append(LoadCase("QSL +X", [Gravity(vector=[9810.0, 0, 0])])),
            "hidden parts": lambda s: s.hidden_parts.append("X"),
            "the requirement set": lambda s: setattr(s.space.requirement_set.limits[0], "value", 123.0)}
    for label, fn in same.items():
        assert digest_after(fn) == d0, label
    # part of it
    differs = {"E": set_mat("E", 0.5), "rho": set_mat("rho", 2.0), "nu": set_mat("nu", 0.9),
               "a support": lambda s: s.constraints.append(Fixed("Support")),
               "dynamics.modes": lambda s: setattr(s.dynamics, "modes", s.dynamics.modes + 5),
               "a point mass": lambda s: s.masses.append(PointMass("Support", 1e-3, point=[1.0, 1.0, 1.0], name="u")),
               "ignored parts": lambda s: s.ignored_parts.append("X"),
               "environment damping": env_edit(lambda es: setattr(es, "damping_zeta", 0.05)),
               "environment damping source": env_edit(lambda es: setattr(es, "damping_source", "other")),
               "environment notes": env_edit(lambda es: setattr(es, "notes", "x"))}
    for label, fn in differs.items():
        assert digest_after(fn) != d0, label
    # a load case that the dynamics refers to (a force excitation, a prestress case) IS part of it; so is one with a rigidly coupled load
    from fea_core.dynamics.model import Excitation
    from fea_core.study import RemoteLoad

    def with_case(s):
        s.load_cases.append(LoadCase("Pull", [Gravity(vector=[0, 0, -9810.0])]))
    d_case = digest_after(with_case)
    assert d_case == d0                                                                      # not referenced: out

    def referenced(s):
        with_case(s)
        s.dynamics.harmonic.excitation.append(Excitation(kind="force", case="Pull", dof=None))
    d_ref = digest_after(referenced)

    def referenced_edited(s):
        referenced(s)
        s.load_cases[-1].loads[0].vector = [0, 0, -1.0]
    assert d_ref != d0 and digest_after(referenced_edited) != d_ref
    assert digest_after(lambda s: s.load_cases.append(LoadCase("R", [RemoteLoad("Support", [0, 0, 0], coupling="rigid")]))) != d0
    assert digest_after(lambda s: s.load_cases.append(LoadCase("R", [RemoteLoad("Support", [0, 0, 0], coupling="distributing")]))) == d0
    # the payload is documented: what is left out is named in the docstring
    doc = LD.run_digest_payload.__doc__ or ""
    assert "INCLUDED" in doc and "LEFT OUT" in doc and "nproc" in doc and "QSL" in doc


def test_n5_the_mesh_fingerprint_follows_the_named_surfaces():
    a, b = SL.box_mesh_tet10(40.0, 20.0, 10.0), SL.box_mesh_tet10(40.0, 20.0, 10.0)
    assert LD.mesh_fingerprint(a) == LD.mesh_fingerprint(b)
    assert set(a.surfaces) == {"Top", "Support"}
    b.surfaces["Support"] = a.surfaces["Top"].copy()                                         # the supports sit on other faces: another model
    assert LD.mesh_fingerprint(a) != LD.mesh_fingerprint(b)
    c = SL.box_mesh_tet10(40.0, 20.0, 10.0)
    c.surfaces["Extra"] = c.surfaces["Top"].copy()
    assert LD.mesh_fingerprint(a) != LD.mesh_fingerprint(c)
    d = SL.box_mesh_tet10(40.0, 20.0, 10.0)
    d.surfaces = dict(reversed(list(a.surfaces.items())))                                    # the order of the dict does not matter
    assert LD.mesh_fingerprint(a) == LD.mesh_fingerprint(d)


def test_n1_a_stale_kept_result_never_turns_back_into_a_static_pass(tmp_path):
    """rc probe C: a FAILING kept envelope, then an edit of the study: the plain static margins (PASS) came back silently."""
    sess, st, res, mesh, r3 = _margin_fixture(tmp_path, 10.0 * SC.TEST_FY)
    assert LD.sign_off_margins(st, res, sess, mesh).verdict == MG.FAIL
    st.name = st.name + " rev B"                                                              # nothing physical changed: the result is still current
    assert LD.sign_off_margins(st, res, sess, mesh).verdict == MG.FAIL and r3.current(st, mesh)
    st.solver.nproc = 2
    assert r3.current(st, mesh)
    mat = next(iter(st.materials.values()))
    mat.E *= 2                                                                                # a physical change: the result is out of date
    s = LD.sign_off_margins(st, res, sess, mesh)
    assert s.verdict == MG.UNCONFIRMED and any("3-axis result is out of date - re-run" in r for r in s.reasons)
    assert LD.sign_off_margins(st, res, sess, None).verdict == MG.UNCONFIRMED                 # no explicit mesh: the one of the results
    # a static FAIL is not hidden by it
    res_bad = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", 10.0 * SC.TEST_SIGMA)])
    assert LD.sign_off_margins(st, res_bad, sess, mesh).verdict == MG.FAIL
    # no kept result at all: the plain margins, no stale blocker
    LD.remember(sess, None)
    sess.results = res
    assert not any("out of date" in r for r in LD.sign_off_margins(st, res, sess, mesh).reasons)


def test_n3_a_damping_ratio_outside_0_1_is_refused_before_the_run():
    st = SL.steel_study("dz")
    st.space = SpaceSettings()
    for z in (0.0, -0.1, 1.0, 1.5):
        es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z", damping_zeta=z, damping_source="t")
        EV.store_environments(st.space, es)
        with pytest.raises(LD.LoadsError, match="damping"):
            LD.resolve_level(st, "qualification")
        assert any("damping_zeta" in p for p in EV.validate_environment(es)), z
    for z in (float("nan"), float("inf")):                                                    # a non-finite record is not even readable: no environments at all
        es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z", damping_zeta=z, damping_source="t")
        EV.store_environments(st.space, es)
        with pytest.raises(LD.LoadsError):
            LD.resolve_level(st, "qualification")
        assert any("damping_zeta" in p for p in EV.validate_environment(es)), z
    es = EV.EnvironmentSet(levels=[SL.three_axis_level()], launch_axis="z", damping_zeta=0.03, damping_source="t")
    EV.store_environments(st.space, es)
    assert LD.resolve_level(st, "qualification")[1].name == "qualification"
    es.damping_zeta = None                                                                    # not entered: allowed (the study damping is used and blocks the PASS)
    EV.store_environments(st.space, es)
    assert LD.resolve_level(st, "qualification")[1].name == "qualification"


def test_n4_a_qsl_case_in_a_local_coordinate_system_is_not_the_generated_one():
    """rc probe F: the same vector in a local system is another direction."""
    from fea_core.study import Gravity, LoadCase
    st = SL.steel_study("q")
    st.space = SL.space_with_limits({"qsl_axial_g": 7.5, "qsl_lateral_g": 3.25}, env_axis="y")
    LD.generate_qsl(st)
    assert LD.qsl_status(st)["current_with_requirement_set"]
    c = next(c for c in st.load_cases if c.name == "QSL +Y")
    c.loads[0].csys = "rotated"
    status = LD.qsl_status(st)
    assert not status["current_with_requirement_set"] and not status["usable_for_verification"] and any("QSL +Y" in p and "differs" in p for p in status["problems"])
    c.loads[0].csys = None
    assert LD.qsl_status(st)["current_with_requirement_set"]
    st.load_cases.insert(0, LoadCase("QSL -Y", [Gravity(vector=[0.0, -1.0, 0.0])]))          # a second case with the same name in front of the generated one
    assert not LD.qsl_status(st)["current_with_requirement_set"]


def test_n3_n6_end_to_end_unset_damping_and_zero_mass_never_pass(tmp_path):
    """rc probe E: no damping in the confirmed environments, a tiny envelope: the result is blocked, so is the verdict."""
    sess, st, res, mesh, r3 = _margin_fixture(tmp_path, 1.0)
    es = EV.environments_of(st.space)
    es.damping_zeta = None
    EV.store_environments(st.space, es)
    lv = es.levels[0]
    n = len(mesh.nodes)
    fl = LD.AxisField(np.full(n, 1.0, np.float32), np.zeros(n, np.float32), None)
    per = {"sine": {0: fl, 1: fl, 2: fl}}
    bb = SimpleNamespace(freq=np.array([100.0]), eff_mass=np.array([[1e-3, 1e-3, 1e-3, 0, 0, 0]]), total_mass=np.array([1e-3] * 3 + [0, 0, 0]), model_mass=1e-3, engine="x")
    rn = LD.build_result(st, mesh, lv, ["sine"], [0, 1, 2], per, bb, env_confirmed=True, req_confirmed=True, env_digest=es.content_digest(),
                         run_digest=LD.compute_run_digest(st, mesh))
    assert any("damping is not entered" in b for b in rn.blockers)
    assert LD.margins_with_envelopes(st, res, rn).verdict != MG.PASS

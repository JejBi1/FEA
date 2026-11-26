"""F14 S4: CubeSat rail supports – the support faces and constraints found on four box rails, the stored setup (no name parsing), the removal of every earlier support, and the
check of reactions and lateral deflection against the limits of the requirement set (never PASS without confirmed limits; a wall that would have to pull is reported in EVERY
state and the margin uses the conservative force on the pushing side). Review findings I4, I5."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import verification.cases  # noqa: F401  (first the cases)
from fea_core import config
from fea_core.geometry import Part
from fea_core.space import rails as RL
from fea_core.space.s4_limits import S4Limits
from fea_core.study import Displacement, Study
from tests.s4_helpers import confirmed_space
from verification.cases import box

RAILS = {"R1": (42, 42), "R2": (42, -50), "R3": (-50, 42), "R4": (-50, -50)}        # (y0, z0) of the 8 × 8 rails along x = 0…100
LIM = {"rail_reaction_limit_N": 800.0, "rail_axial_reaction_limit_N": 1500.0, "rail_lateral_deflection_limit_mm": 0.5}


def _parts(rails=None):
    return [Part(n, box(0, y0, z0, 100, 8, 8)) for n, (y0, z0) in (rails or RAILS).items()]            # box() takes SIZES: 8 x 8 rails


def _spec(**kw):
    d = dict(rail_parts=list(RAILS), axis="x", axial="max")
    d.update(kw)
    return RL.RailSpec(**d)


def _lim(**kw):
    return S4Limits.from_requirement_set(confirmed_space({**LIM, **kw}))


def test_rail_faces_and_constraints():
    parts = _parts()
    found = RL.rail_faces(parts, _spec())
    walls = [f for f in found if f["kind"] == "wall"]
    assert len(walls) == 8 and len([f for f in found if f["kind"] == "axial"]) == 4
    assert {f["name"] for f in walls if f["part"] == "R1"} == {"Rail R1 +y", "Rail R1 +z"}          # only the faces on the outer envelope
    assert {f["name"] for f in walls if f["part"] == "R4"} == {"Rail R4 -y", "Rail R4 -z"}
    st = Study(name="rails", geometry="x.step")
    out = RL.apply_rail_supports(st, parts, _spec())
    assert len(out["supports"]) == 12 and len(st.constraints) == 12
    by = {c.selection: c for c in st.constraints}
    assert isinstance(by["Rail R1 +y"], Displacement) and by["Rail R1 +y"].uy == 0.0 and by["Rail R1 +y"].ux is None and by["Rail R1 +y"].uz is None
    assert by["Rail R2 -z"].uz == 0.0 and by["Rail R2 axial max"].ux == 0.0
    # the setup is stored with the study: part, kind, axis, sign of every support
    setup = RL.setup_of(st)
    r1 = next(x for x in setup["supports"] if x["selection"] == "Rail R1 +y")
    assert (r1["part"], r1["kind"], r1["axis"], r1["sign"], r1["component"]) == ("R1", "wall", "y", 1, 1)
    ax = next(x for x in setup["supports"] if x["selection"] == "Rail R2 axial max")
    assert (ax["kind"], ax["axis"], ax["sign"], ax["component"]) == ("axial", "x", 1, 0)
    # the second call replaces, it does not duplicate
    RL.apply_rail_supports(st, parts, _spec())
    assert len(st.constraints) == 12 and len([s for s in st.selections if s.startswith("Rail ")]) == 12
    # both ends / the opposite end
    assert len([f for f in RL.rail_faces(parts, _spec(axial="both")) if f["kind"] == "axial"]) == 8
    assert all(f["name"].endswith("axial min") for f in RL.rail_faces(parts, _spec(axial="min")) if f["kind"] == "axial")
    assert not [f for f in RL.rail_faces(parts, _spec(axial="none")) if f["kind"] == "axial"]
    assert any("axial" in w for w in RL.apply_rail_supports(Study(name="r", geometry="x.step"), parts, _spec(axial="none"))["warnings"])
    with pytest.raises(RL.RailError):
        RL.rail_faces(parts, _spec(rail_parts=["nope"]))
    with pytest.raises(RL.RailError):
        RL.rail_faces(parts, _spec(axial="top"))


def test_a_new_apply_removes_every_earlier_rail_support():                                          # I5
    parts = _parts()
    st = Study(name="rails", geometry="x.step")
    RL.apply_rail_supports(st, parts, _spec(axial="min"))
    assert len([c for c in st.constraints if "axial" in c.selection]) == 4
    other = Displacement("Rail notes", ux=0.0)                                                  # a constraint of the user on a selection that merely starts with "Rail "
    st.add_selection("Rail notes", faces=RL.rail_faces(parts, _spec())[0]["faces"])
    st.constraints.append(other)
    out = RL.apply_rail_supports(st, parts, _spec(axial="none"))                                 # no axial support: the old axial ones must go
    assert out["removed"] == 12 and not [c for c in st.constraints if "axial" in c.selection] and not [s for s in st.selections if "axial" in s]
    assert other in st.constraints and "Rail notes" in st.selections                              # the user's own constraint is not ours: untouched
    RL.apply_rail_supports(st, parts, _spec(rail_parts=["R1", "R2"], axial="none"))              # fewer rails: R3, R4 supports must go
    names = sorted(c.selection for c in st.constraints if c.selection != "Rail notes")
    assert names == sorted(["Rail R1 +y", "Rail R1 -y", "Rail R1 +z", "Rail R2 +y", "Rail R2 -y", "Rail R2 -z"]) and {x["part"] for x in RL.setup_of(st)["supports"]} == {"R1", "R2"}
    # a project saved before the setup was stored: the helper's own names are recognised by their pattern, the user's are not
    st.space.extra.pop(RL.SETUP_KEY)
    RL.apply_rail_supports(st, parts, _spec(axial="min"))
    assert len([c for c in st.constraints if c.selection.startswith("Rail R")]) == 12
    assert other in st.constraints


def test_part_names_with_spaces_work_and_other_rail_selections_are_ignored():                       # I4
    rails = {"Rail 1": (42, 42), "Rail 2": (-50, -50)}
    parts = _parts(rails)
    st = Study(name="rails", geometry="x.step")
    RL.apply_rail_supports(st, parts, RL.RailSpec(list(rails), "x", "max"))
    st.add_selection("Rail notes", faces=RL.rail_faces(parts, RL.RailSpec(list(rails), "x", "max"))[0]["faces"])
    res = _results(st, {"A": {"Rail Rail 1 +y": [0, -5000.0, 0], "Rail Rail 1 +z": [0, 0, -10.0], "Rail Rail 2 -y": [0, 5.0, 0], "Rail Rail 2 -z": [0, 0, 10.0],
                              "Rail Rail 1 axial max": [-100.0, 0, 0], "Rail Rail 2 axial max": [-100.0, 0, 0]}}, part_names=["Rail 1", "Rail 2"])
    r = RL.check_rails(st, res, RL.RailSpec(list(rails), "x", "max"), _lim())
    assert r.verdict == RL.FAIL and {x.part for x in r.rows} == {"Rail 1", "Rail 2"}                 # 5000 N > the limit 800 N: found although the part name has a space
    assert len(r.deflection) == 2 and all(x.support != "Rail notes" for x in r.rows)


def _results(study, states: dict, lat_defl: float = 0.0, part_names=("R1", "R4")):
    nodes = np.array([[0, 42, 42], [100, 50, 50], [0, -50, -50], [100, -42, -42]], float)
    mesh = SimpleNamespace(part_names=list(part_names), tets=np.array([[0, 1, 0, 1], [2, 3, 2, 3]]), tet_part=np.array([0, 1]), nodes=nodes)
    steps = []
    for name, reac in states.items():
        U = np.zeros((4, 3))
        U[1, 1] = lat_defl
        steps.append(SimpleNamespace(name=name, reactions={k: np.array(v, float) for k, v in reac.items()}, U=U))
    return SimpleNamespace(steps=steps, mesh=mesh)


def _study_for(rails=("R1", "R4"), axial="max"):
    st = Study(name="rails", geometry="x.step")
    parts = [p for p in _parts() if p.name in rails]
    RL.apply_rail_supports(st, parts, RL.RailSpec(list(rails), "x", axial))
    return st


def test_check_margins_by_hand_and_verdict_rules():
    st = _study_for()
    # wall on +y pushes toward −y (R_y < 0 = compression); wall on −y pushes toward +y; the axial support carries 900 N
    res = _results(st, {"Launch": {"Rail R1 +y": [0, -400.0, 0], "Rail R4 -y": [0, 300.0, 0], "Rail R1 axial max": [-900.0, 0, 0]}}, lat_defl=0.25)
    spec = RL.RailSpec(["R1", "R4"], "x", "max")
    ok = RL.check_rails(st, res, spec, _lim())
    rows = {r.support: r for r in ok.rows}
    assert rows["Rail R1 +y"].reaction_N == pytest.approx(400.0) and rows["Rail R1 +y"].mos == pytest.approx(800 / 400 - 1)
    assert rows["Rail R4 -y"].mos == pytest.approx(800 / 300 - 1) and rows["Rail R1 axial max"].mos == pytest.approx(1500 / 900 - 1)
    assert ok.deflection[0]["part"] == "R1" and ok.deflection[0]["max_lateral_deflection_mm"] == pytest.approx(0.25)
    assert ok.deflection[0]["mos"] == pytest.approx(0.5 / 0.25 - 1)
    assert ok.verdict == RL.PASS and not ok.warnings and "bilateral linear supports" in ok.verdict_text                 # the limitation is part of the verdict text
    # direct numbers are never confirmed
    direct = RL.check_rails(st, res, RL.RailSpec(["R1", "R4"], "x", "max", reaction_limit_N=800.0, axial_reaction_limit_N=1500.0, lateral_deflection_limit_mm=0.5), S4Limits())
    assert direct.verdict == RL.UNCONFIRMED
    unc = RL.check_rails(st, res, spec, S4Limits.from_requirement_set(confirmed_space(LIM, confirm_it=False)))
    assert unc.verdict == RL.UNCONFIRMED
    # an exceeded limit with confirmed inputs
    bad = RL.check_rails(st, _results(st, {"Launch": {"Rail R1 +y": [0, -900.0, 0], "Rail R4 -y": [0, 300.0, 0], "Rail R1 axial max": [-900.0, 0, 0]}}), spec, _lim())
    assert bad.verdict == RL.FAIL
    with pytest.raises(RL.RailError):
        RL.check_rails(Study(name="x", geometry="x.step"), res, spec)
    with pytest.raises(RL.RailError):
        RL.check_rails(st, _results(st, {}), spec)


def test_a_pulling_wall_is_found_in_every_state_not_only_in_the_worst_one():                        # I4
    st = _study_for()
    spec = RL.RailSpec(["R1", "R4"], "x", "max")
    res = _results(st, {"A": {"Rail R1 +y": [0, -100.0, 0], "Rail R4 -y": [0, 100.0, 0], "Rail R1 axial max": [-500.0, 0, 0]},
                        "B": {"Rail R1 +y": [0, +60.0, 0], "Rail R4 -y": [0, 300.0, 0], "Rail R1 axial max": [-500.0, 0, 0]}})        # B: R1 +y would pull 60 N
    r = RL.check_rails(st, res, spec, _lim())
    row = next(x for x in r.rows if x.support == "Rail R1 +y")
    assert row.pulls and row.pull_states == ["B"] and row.state == "A"                               # the worst state (by force) is A, the pull is in B
    assert any("PULL" in w and "Rail R1 +y" in w for w in r.warnings) and f"{1} support(s) would pull" in r.verdict_text
    # the pull of a state with a smaller force than the worst one is still found when the worst state does not pull
    res2 = _results(st, {"A": {"Rail R1 +y": [0, -50.0, 0], "Rail R4 -y": [0, 100.0, 0]}, "B": {"Rail R1 +y": [0, +60.0, 0], "Rail R4 -y": [0, 100.0, 0]}})
    assert next(x for x in RL.check_rails(st, res2, spec, _lim()).rows if x.support == "Rail R1 +y").pulls


def test_the_margin_uses_the_conservative_force_of_the_pushing_side():                             # I5
    st = _study_for()
    spec = RL.RailSpec(["R1", "R4"], "x", "max")
    # a lateral load of 100 N shared by the bilateral walls: +y wall pushes 50 N, the -y wall pulls 50 N -> the real wall carries 100 N
    res = _results(st, {"S": {"Rail R1 +y": [0, -50.0, 0], "Rail R4 -y": [0, -50.0, 0], "Rail R1 axial max": [-10.0, 0, 0]}})
    r = RL.check_rails(st, res, spec, _lim(rail_reaction_limit_N=120.0))
    rows = {x.support: x for x in r.rows}
    assert rows["Rail R1 +y"].reaction_N == pytest.approx(50.0) and rows["Rail R1 +y"].effective_N == pytest.approx(100.0)
    assert rows["Rail R1 +y"].mos == pytest.approx(120.0 / 100.0 - 1) and rows["Rail R4 -y"].pulls and rows["Rail R4 -y"].effective_N == pytest.approx(50.0)
    tight = RL.check_rails(st, res, spec, _lim(rail_reaction_limit_N=80.0))                      # 80 N > 50 N (the halved force) but < 100 N (the real one)
    assert tight.verdict == RL.FAIL
    # two pushing walls on one side share the pull of the opposite side in proportion to their push
    st4 = _study_for(("R1", "R2", "R3", "R4"))
    spec4 = RL.RailSpec(list(RAILS), "x", "max")
    res4 = _results(st4, {"G": {"Rail R1 +y": [0, -30.0, 0], "Rail R2 +y": [0, -10.0, 0], "Rail R3 -y": [0, -20.0, 0], "Rail R4 -y": [0, -20.0, 0],
                                "Rail R1 axial max": [-1.0, 0, 0]}}, part_names=list(RAILS))
    res4.mesh.tets = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]])
    res4.mesh.tet_part = np.array([0, 1, 2, 3])
    r4 = {x.support: x for x in RL.check_rails(st4, res4, spec4, _lim()).rows}
    assert r4["Rail R1 +y"].effective_N == pytest.approx(30.0 * (1 + 40.0 / 40.0)) and r4["Rail R2 +y"].effective_N == pytest.approx(10.0 * (1 + 40.0 / 40.0))


def test_axial_both_uses_the_whole_load_on_the_pushing_end_and_flags_the_pulling_one():           # re-check: it halved the force and never flagged a pull
    st = _study_for(("R1", "R4"), "both")
    spec = RL.RailSpec(["R1", "R4"], "x", "both")
    # 1 g along +x: the max ends push (R_x < 0 on the structure), the min ends would have to PULL; each takes half of the load, 200 N in all
    res = _results(st, {"gx": {"Rail R1 axial max": [-50.0, 0, 0], "Rail R4 axial max": [-50.0, 0, 0], "Rail R1 axial min": [-50.0, 0, 0], "Rail R4 axial min": [-50.0, 0, 0]}})
    r = RL.check_rails(st, res, spec, _lim(rail_axial_reaction_limit_N=120.0))
    rows = {x.support: x for x in r.rows}
    assert rows["Rail R1 axial max"].reaction_N == pytest.approx(50.0) and rows["Rail R1 axial max"].effective_N == pytest.approx(100.0)     # 50 + the share of the 100 N pull
    assert rows["Rail R1 axial min"].pulls and rows["Rail R1 axial min"].reaction_N == pytest.approx(-50.0) and rows["Rail R1 axial min"].effective_N == pytest.approx(50.0)
    assert not rows["Rail R1 axial max"].pulls and r.verdict == RL.PASS and "2 support(s) would pull" in r.verdict_text
    assert any("end plate" in w and "Rail R1 axial min" in w for w in r.warnings)
    for lim, verdict in ((90.0, RL.FAIL), (120.0, RL.PASS), (99.0, RL.FAIL)):                     # the real force on the pushing end is 100 N, not the halved 50 N
        assert RL.check_rails(st, res, spec, _lim(rail_axial_reaction_limit_N=lim)).verdict == verdict
    # the load in the other direction: the roles swap
    rev = _results(st, {"gx-": {k: [+50.0, 0, 0] for k in ("Rail R1 axial max", "Rail R4 axial max", "Rail R1 axial min", "Rail R4 axial min")}})
    rr = {x.support: x for x in RL.check_rails(st, rev, spec, _lim(rail_axial_reaction_limit_N=120.0)).rows}
    assert rr["Rail R1 axial min"].effective_N == pytest.approx(100.0) and rr["Rail R1 axial max"].pulls


def test_axial_min_flags_an_end_that_would_have_to_pull():
    st = _study_for(("R1", "R4"), "min")
    spec = RL.RailSpec(["R1", "R4"], "x", "min")
    pushed = _results(st, {"gx-": {"Rail R1 axial min": [+80.0, 0, 0], "Rail R4 axial min": [+80.0, 0, 0]}})            # the load presses the structure onto the held end
    ok = RL.check_rails(st, pushed, spec, _lim(rail_axial_reaction_limit_N=100.0))
    assert not any(x.pulls for x in ok.rows) and ok.verdict == RL.PASS and {x.effective_N for x in ok.rows if x.kind == "axial"} == {80.0}
    pulled = _results(st, {"gx": {"Rail R1 axial min": [-80.0, 0, 0], "Rail R4 axial min": [-80.0, 0, 0]}})               # the load pulls the structure away from it
    bad = RL.check_rails(st, pulled, spec, _lim(rail_axial_reaction_limit_N=100.0))
    axial = [x for x in bad.rows if x.kind == "axial"]
    assert axial and all(x.pulls and x.pull_states == ["gx"] and x.reaction_N == pytest.approx(-80.0) for x in axial)
    assert any("PULL" in w and "end plate" in w and "choose the end that the load pushes against" in w for w in bad.warnings) and "support(s) would pull" in bad.verdict_text


def test_the_pull_threshold_is_relative_to_the_total_reaction():                                    # I5
    st = _study_for()
    spec = RL.RailSpec(["R1", "R4"], "x", "max")
    big = {"Rail R1 +y": [0, -1000.0, 0], "Rail R4 -y": [0, 300.0, 0], "Rail R1 axial max": [-900.0, 0, 0]}
    noise = RL.check_rails(st, _results(st, {"S": {**big, "Rail R4 -y": [0, -0.5, 0]}}), spec, _lim(rail_reaction_limit_N=5000.0))      # 0.5 N pull of ~1900 N total
    assert not any(x.pulls for x in noise.rows)
    real = RL.check_rails(st, _results(st, {"S": {**big, "Rail R4 -y": [0, -50.0, 0]}}), spec, _lim(rail_reaction_limit_N=5000.0))      # 50 N: 2.6 % of the total
    assert next(x for x in real.rows if x.support == "Rail R4 -y").pulls
    tight = RL.check_rails(st, _results(st, {"S": {**big, "Rail R4 -y": [0, -0.5, 0]}}), RL.RailSpec(["R1", "R4"], "x", "max", pull_rel_tol=1e-6), _lim(rail_reaction_limit_N=5000.0))
    assert next(x for x in tight.rows if x.support == "Rail R4 -y").pulls                          # the same 0.5 N counts with a stricter numerical setting
    with pytest.raises(RL.RailError):
        RL.check_rails(st, _results(st, {"S": big}), RL.RailSpec(["R1", "R4"], "x", "max", pull_rel_tol=2.0), _lim())


def test_a_named_rail_without_a_support_row_gives_no_verdict():                                    # I4
    st = _study_for(("R1",))                                                                      # only R1 has supports
    res = _results(st, {"S": {"Rail R1 +y": [0, -100.0, 0], "Rail R1 axial max": [-10.0, 0, 0]}})
    r = RL.check_rails(st, res, RL.RailSpec(["R1", "R4"], "x", "max"), _lim())
    assert r.verdict == RL.UNCONFIRMED and "R4" in r.verdict_text and any("No support row" in w for w in r.warnings)
    ok = RL.check_rails(st, res, RL.RailSpec(["R1"], "x", "max"), _lim())
    assert ok.verdict == RL.PASS
    assert RL.check_rails(st, res, RL.RailSpec([], "x", "max"), _lim()).verdict == RL.PASS            # no rail named = every rail of the stored setup


needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")


@needs_ccx
def test_the_stage_own_four_rail_model_gives_a_sensible_verdict(tmp_path):
    """The model of the stage on a real mesh: four 8 x 8 rails with panels, the wall supports + the axial support, four gravity states (1 g in ±x, +y, +z). The walls share a
    lateral load by halves (one pulls); the conservative margin carries the whole weight on the pushing pair, so the rail force is W/2 per pushing wall (a rigid body between
    the walls): the verdict with generous limits is PASS, with limits below the real force it is FAIL, and the pulls are reported."""
    from fea_core.solve import run_study
    from fea_core.study import Gravity, LoadCase, MeshSettings
    from verification.cases import _study
    geo = _parts() + [Part("PanelTop", box(10, -42, 44, 80, 84, 4)), Part("PanelBottom", box(10, -42, -48, 80, 84, 4)), Part("PanelLeft", box(10, 44, -42, 80, 4, 84)),
                      Part("PanelRight", box(10, -48, -42, 80, 4, 84))]
    st, parts = _study("cubesat", tmp_path / "model.step", geo, MeshSettings(size=6.0))
    spec = RL.RailSpec(list(RAILS), "x", "min")
    RL.apply_rail_supports(st, parts, spec)
    for name, v in (("gx", [9810.0, 0, 0]), ("gy", [0, 9810.0, 0]), ("gz", [0, 0, 9810.0]), ("gx-", [-9810.0, 0, 0])):
        st.load_cases.append(LoadCase(name, [Gravity(vector=v)]))
    res = run_study(st, tmp_path / "run", tmp_path, parts=parts)
    weight = abs(sum(np.asarray(v)[:3] for v in res.steps[1].reactions.values())[1])        # state gy: the sum of the reactions = -weight
    assert weight == pytest.approx(10.25, rel=0.02)
    ok = RL.check_rails(st, res, spec, _lim(rail_reaction_limit_N=100.0, rail_axial_reaction_limit_N=100.0, rail_lateral_deflection_limit_mm=1.0))
    assert ok.verdict == RL.PASS, (ok.verdict_text, ok.warnings)
    walls = {r.support: r for r in ok.rows}
    assert any(r.pulls for r in ok.rows)                                                          # the bilateral walls do pull in the lateral states ...
    assert walls["Rail R1 +y"].effective_N == pytest.approx(weight / 2, rel=0.05) and walls["Rail R1 +z"].effective_N == pytest.approx(weight / 2, rel=0.05)   # ... and the
    # conservative force of the pushing pair is the whole weight shared by the two walls
    low = RL.check_rails(st, res, spec, _lim(rail_reaction_limit_N=4.0, rail_axial_reaction_limit_N=100.0, rail_lateral_deflection_limit_mm=1.0))
    assert low.verdict == RL.FAIL                                                                 # 4 N < 5.1 N (the real force) although > 2.56 N (the halved one)


@needs_ccx
@pytest.mark.parametrize("axial", ["both", "min"])
def test_the_four_rail_model_at_1g_axial_carries_a_quarter_of_the_weight_per_pushing_end(tmp_path, axial):
    """The re-check model (four rails, 1 g along ±x): the real force on the 4 pushing ends of one side is W, 0.25 W each; the bilateral model with axial='both' halves it (0.125 W)
    and one end pulls. The conservative margin gives 0.25 W per pushing end: the limits 0.15 W and 0.20 W FAIL, 0.3 W passes; with axial='min' the load in +x makes the held end PULL."""
    from fea_core.solve import run_study
    from fea_core.study import Gravity, LoadCase, MeshSettings
    from verification.cases import _study
    geo = _parts() + [Part("PanelTop", box(10, -42, 44, 80, 84, 4)), Part("PanelBottom", box(10, -42, -48, 80, 84, 4)), Part("PanelLeft", box(10, 44, -42, 80, 4, 84)),
                      Part("PanelRight", box(10, -48, -42, 80, 4, 84))]
    st, parts = _study("cubesat", tmp_path / "model.step", geo, MeshSettings(size=6.0))
    spec = RL.RailSpec(list(RAILS), "x", axial)
    RL.apply_rail_supports(st, parts, spec)
    for name, v in (("gx", [9810.0, 0, 0]), ("gx-", [-9810.0, 0, 0])):
        st.load_cases.append(LoadCase(name, [Gravity(vector=v)]))
    res = run_study(st, tmp_path / "run", tmp_path, parts=parts)
    weight = abs(sum(np.asarray(v)[0] for v in res.steps[1].reactions.values()))                # state gx-: the reactions along x sum to the weight
    assert weight == pytest.approx(10.25, rel=0.02)
    kw = dict(rail_reaction_limit_N=1e4, rail_lateral_deflection_limit_mm=1.0)
    ok = RL.check_rails(st, res, spec, _lim(rail_axial_reaction_limit_N=0.3 * weight, **kw))
    axial_rows = [r for r in ok.rows if r.kind == "axial"]
    pushing = [r for r in axial_rows if r.effective_N > 0.2 * weight]
    assert len(axial_rows) == (8 if axial == "both" else 4) and len(pushing) == len(axial_rows)         # every end is the pushing one in one of the two states
    assert all(r.effective_N == pytest.approx(0.25 * weight, rel=0.03) for r in pushing)                # 0.25 W each, not the halved 0.125 W
    assert ok.verdict == RL.PASS
    for k in (0.15, 0.20):
        assert RL.check_rails(st, res, spec, _lim(rail_axial_reaction_limit_N=k * weight, **kw)).verdict == RL.FAIL
    pulling = [r for r in axial_rows if r.pulls]
    assert pulling and all(r.pull_states for r in pulling)
    if axial == "min":
        assert {r.support.split()[-1] for r in pulling} == {"min"} and len(pulling) == 4
    else:
        assert len(pulling) == 8 and len(axial_rows) == 8                                         # every end pulls in one of the two states: the bilateral plates cannot be exact

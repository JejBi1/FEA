import math
from pathlib import Path

import numpy as np
import pytest

from fea_core.sweep import (
    DesignPoint, ParametricStudySpec, identify_pareto, scale_study_loads, run_parametric_study
)
from fea_core.study import (
    Study, Material, LoadCase, Force, Pressure, Gravity,
    RemoteLoad, BearingLoad, Rotation, ThermalLoad
)
from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult


def test_spec_combinations():
    spec = ParametricStudySpec(
        load_factors=[0.5, 1.0, 1.5],
        mesh_sizes=[5.0, 10.0],
        max_points=20,
    )
    combos = spec.combinations()
    assert len(combos) == 6
    assert combos[0] == {"load_factor": 0.5, "mesh_size": 5.0}
    assert combos[-1] == {"load_factor": 1.5, "mesh_size": 10.0}
    assert len(spec.warnings) == 0


def test_spec_one_element_mesh_sizes_and_max_points():
    # a 1-element list of mesh_sizes must not be ignored
    spec1 = ParametricStudySpec(
        mesh_sizes=[3.5],
    )
    combos1 = spec1.combinations()
    assert len(combos1) == 1
    assert combos1[0] == {"mesh_size": 3.5}

    # max_points limits the number and generates a warning
    spec2 = ParametricStudySpec(
        load_factors=[0.5, 1.0, 1.5, 2.0],
        mesh_sizes=[2.0, 4.0],
        max_points=3,
    )
    combos2 = spec2.combinations()
    assert len(combos2) == 3
    assert len(spec2.warnings) == 1
    assert "max_points=3" in spec2.warnings[0]


def test_pareto_front():
    # DP0: light and low stress -> pareto
    dp0 = DesignPoint(index=0, params={}, mass=1.0, max_stress=100.0, converged=True)
    # DP1: heavier and higher stress -> dominated by DP0
    dp1 = DesignPoint(index=1, params={}, mass=2.0, max_stress=150.0, converged=True)
    # DP2: heavier but lower stress -> trade-off -> pareto
    dp2 = DesignPoint(index=2, params={}, mass=1.5, max_stress=80.0, converged=True)
    # DP3: error
    dp3 = DesignPoint(index=3, params={}, mass=0.5, max_stress=50.0, converged=False)

    pareto = identify_pareto([dp0, dp1, dp2, dp3])
    assert pareto == [0, 2]


def test_scale_study_loads():
    st = Study(name="ScaleTest", geometry="model.step")
    lc = LoadCase("LC1", loads=[
        Force("F1", [100.0, 0.0, -50.0]),
        Pressure("P1", value=2.0, gradient=[0.1, 0.0, 0.2]),
        Gravity(vector=[0.0, 0.0, -9810.0]),
        RemoteLoad("R1", point=[0.0, 0.0, 0.0], force=[10.0, 20.0, 30.0], moment=[1.0, 2.0, 3.0]),
        BearingLoad("B1", vector=[500.0, 0.0, 0.0]),
        Rotation(point=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], omega=100.0, alpha=50.0),
        ThermalLoad(value=120.0),
    ])
    st.load_cases.append(lc)

    scale_study_loads(st, factor=4.0)

    # Force: linearly 4x
    assert lc.loads[0].vector == [400.0, 0.0, -200.0]
    # Pressure: value and gradient 4x
    assert lc.loads[1].value == pytest.approx(8.0)
    assert lc.loads[1].gradient == pytest.approx([0.4, 0.0, 0.8])
    # Gravity: linearly 4x
    assert lc.loads[2].vector == [0.0, 0.0, -39240.0]
    # Remote: force i moment 4x
    assert lc.loads[3].force == [40.0, 80.0, 120.0]
    assert lc.loads[3].moment == [4.0, 8.0, 12.0]
    # Bearing: linearly 4x
    assert lc.loads[4].vector == [2000.0, 0.0, 0.0]
    # Rotation: omega scales with sqrt(factor) = 2x, alpha linearly = 4x
    assert lc.loads[5].omega == pytest.approx(200.0)
    assert lc.loads[5].alpha == pytest.approx(200.0)
    # ThermalLoad: the temperature in °C is NOT scaled
    assert lc.loads[6].value == 120.0


def test_pure_load_sweep_shortcut(tmp_path, monkeypatch):
    st = Study(name="LinearBeam", geometry="beam.step")
    mat = Material(name="STEEL", E=210000.0, nu=0.3, rho=7.85e-9, Re=250.0)
    st.materials["STEEL"] = mat
    st.part_materials["Part1"] = "STEEL"
    st.load_cases.append(LoadCase("Load1", [Force("end", [0.0, 0.0, -100.0])]))

    # Dummy mesh and results
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    tets = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes, tets, np.zeros(1, dtype=np.int32), ["Part1"])

    U_base = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    S_base = np.array([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
    step_base = StepResult("Load1", U_base, S_base, reactions={"end": np.array([0.0, 0.0, 100.0])})
    mock_res = Results(mesh, [step_base], [], {})

    solve_count = 0

    def mock_run_study(study, workdir, cancel=None, mesh=None, **kw):
        nonlocal solve_count
        solve_count += 1
        return mock_res

    monkeypatch.setattr("fea_core.sweep.run_study", mock_run_study)

    spec = ParametricStudySpec(load_factors=[1.0, 2.0, 0.5])
    res = run_parametric_study(st, spec, tmp_path / "sweep_out")

    # A linear purely load sweep must solve the study exactly ONCE
    assert solve_count == 1
    assert len(res.points) == 3

    # DP0 (factor 1.0): max stress = 50 MPa
    assert res.points[0].max_stress == pytest.approx(50.0)
    # DP1 (factor 2.0): max stress = 100 MPa, max disp = 2x
    assert res.points[1].max_stress == pytest.approx(100.0)
    assert res.points[1].max_disp == pytest.approx(res.points[0].max_disp * 2.0)
    # DP2 (factor 0.5): max stress = 25 MPa
    assert res.points[2].max_stress == pytest.approx(25.0)


def test_part_nodes_element_indexing_fix(tmp_path, monkeypatch):
    """Check that min_safety does not index the nodes by the indices of the elements tet_part."""
    st = Study(name="MultiElemTest", geometry="model.step")
    mat1 = Material(name="M1", E=210000.0, nu=0.3, rho=7.85e-9, Re=300.0)
    mat2 = Material(name="M2", E=210000.0, nu=0.3, rho=7.85e-9, Re=150.0)
    st.materials["M1"] = mat1
    st.materials["M2"] = mat2
    st.part_materials["Part1"] = "M1"
    st.part_materials["Part2"] = "M2"
    st.load_cases.append(LoadCase("LC", [Force("F", [10.0, 0.0, 0.0])]))

    # Mesh with 6 nodes and 3 elements
    # Elements 0 and 1 belong to Part1 (tet_part=0)
    # Element 2 belongs to Part2 (tet_part=1)
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ])
    tet_part = np.array([0, 0, 1], dtype=np.int32)
    mesh = Mesh(nodes, tets, tet_part, ["Part1", "Part2"])

    # Stresses at the nodes 0..5
    # Node 5 has a high stress of 100 MPa in Part2 (Re=150 -> safety = 150/100 = 1.5)
    # Nodes 0..4 have a stress of 50 MPa in Part1 (Re=300 -> safety = 300/50 = 6.0)
    S = np.zeros((6, 6))
    S[:5, 0] = 50.0
    S[5, 0] = 100.0
    step = StepResult("LC", np.zeros((6, 3)), S)
    mock_res = Results(mesh, [step], [], {})

    monkeypatch.setattr("fea_core.sweep.run_study", lambda st, wd, **kw: mock_res)

    spec = ParametricStudySpec(load_factors=[1.0])
    res = run_parametric_study(st, spec, tmp_path / "safety_test")

    # The minimum safety must be 1.5 (from Part2), neither infinity nor wrongly indexed
    assert res.points[0].min_safety == pytest.approx(1.5)
    assert res.points[0].min_safety_raw == pytest.approx(1.5)


# ---------------------------------------------------------------- review fixes (shortcut, safety, Pareto) on the shared engine

def _beam_study(**extra):
    st = Study(name="S", geometry="beam.step")
    st.materials["STEEL"] = Material(name="STEEL", E=210000.0, nu=0.3, rho=7.85e-9, Re=250.0)
    st.part_materials["Part1"] = "STEEL"
    st.load_cases.append(LoadCase("Load1", [Force("end", [0.0, 0.0, -100.0])]))
    return st


def _mock_results(S0=50.0, fields=None):
    nodes = np.array([[0.0, 0, 0], [10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
    mesh = Mesh(nodes, np.array([[0, 1, 2, 3]]), np.zeros(1, dtype=np.int32), ["Part1"])
    U = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    S = np.array([[S0, 0, 0, 0, 0, 0]] * 4, float)
    step = StepResult("Load1", U, S, reactions={"end": np.array([0.0, 0.0, 100.0])},
                      equilibrium={"reaction_total": [0.0, 0.0, 100.0], "imbalance": 1e-9},
                      fields=dict(fields or {}))
    return Results(mesh, [step], [], {})


@pytest.mark.parametrize("variant", ["thermal", "displacement", "bolt"])
def test_shortcut_blocked_by_unscaled_inputs(tmp_path, monkeypatch, variant):
    """Thermal load, nonzero prescribed displacement and bolt preload are not scaled → every point is solved."""
    from fea_core.study import Displacement
    from fea_core.joints.model import BoltSpec
    st = _beam_study()
    if variant == "thermal":
        st.load_cases[0].loads.append(ThermalLoad(value=80.0))
    elif variant == "displacement":
        st.constraints.append(Displacement("tah", ux=0.1))
    else:
        st.bolts.append(BoltSpec(name="B1"))
        assert st.active_bolts()
    assert not _linear_scalable(st) and shortcut_blockers(st)
    calls = []
    monkeypatch.setattr("fea_core.sweep.run_study", lambda s, wd, **kw: calls.append(s) or _mock_results())
    res = run_parametric_study(st, ParametricStudySpec(load_factors=[1.0, 2.0]), tmp_path / "o")
    assert len(calls) == 2 and not res.linear_shortcut
    assert res.shortcut_reason and "full_solve_reason" in res.to_dict()
    assert any("does not scale" in w for w in res.warnings)


def test_pareto_uses_safety_and_requirement():
    a = DesignPoint(0, {}, mass=1.0, max_stress=100.0, min_safety=1.2)
    b = DesignPoint(1, {}, mass=1.0, max_stress=100.0, min_safety=2.0)      # same mass/stress, safer → dominates a
    c = DesignPoint(2, {}, mass=0.8, max_stress=120.0, min_safety=1.4)      # lighter → trade-off
    assert identify_pareto([a, b, c]) == [1, 2]
    assert identify_pareto([a, b, c], required_safety=1.5) == [1]
    # stress outside singularities is the stress objective when known
    d = DesignPoint(3, {}, mass=1.0, max_stress=500.0, max_stress_clean=90.0, min_safety=2.0)
    assert identify_pareto([b, d]) == [3]


def test_plastic_default_material_blocks_shortcut(tmp_path, monkeypatch):
    """Regression: a plastic default material + one explicitly assigned elastic part was reported linear without parts →
    the sweep rescaled an elastoplastic result."""
    from fea_core.plasticity import PlasticitySpec
    st = _beam_study()
    st.materials["PL"] = Material(name="PL", E=210000.0, nu=0.3, rho=7.85e-9, Re=250.0, plastic=PlasticitySpec())
    st.default_material = "PL"
    st.part_materials = {"Other": "STEEL"}          # Part1 falls back to the plastic default
    assert st.is_nonlinear()
    assert not _linear_scalable(st)
    calls = []
    monkeypatch.setattr("fea_core.sweep.run_study", lambda s, wd, **kw: calls.append(s) or _mock_results())
    res = run_parametric_study(st, ParametricStudySpec(load_factors=[1.0, 2.0]), tmp_path / "o")
    assert len(calls) == 2 and not res.linear_shortcut


def test_linear_study_stays_linear():
    st = _beam_study()
    st.default_material = "STEEL"
    assert not st.is_nonlinear()
    assert _linear_scalable(st, [{"load_factor": -1.0}, {"load_factor": 2.0}])


def test_negative_factor_with_clipped_pressure_blocks_shortcut(tmp_path, monkeypatch):
    """Regression: max(0, λ·g·d) ≠ λ·max(0, g·d) for λ < 0 – the wetted side of a hydrostatic pressure moves."""
    st = _beam_study()
    st.load_cases[0].loads.append(Pressure("wall", 0.0, gradient=[0.0, 0.0, -1e-5], origin=[0.0, 0.0, 5.0], clip=True))
    assert _linear_scalable(st, [{"load_factor": 0.0}, {"load_factor": 2.0}])          # λ ≥ 0 stays exact
    assert not _linear_scalable(st, [{"load_factor": 1.0}, {"load_factor": -1.0}])
    calls = []
    monkeypatch.setattr("fea_core.sweep.run_study", lambda s, wd, **kw: calls.append(s) or _mock_results())
    res = run_parametric_study(st, ParametricStudySpec(load_factors=[1.0, -1.0]), tmp_path / "o")
    assert len(calls) == 2 and not res.linear_shortcut
    st.load_cases[0].loads[-1].clip = False                              # unclipped = linear in λ
    assert _linear_scalable(st, [{"load_factor": 1.0}, {"load_factor": -1.0}])



# --------------------------------------------------------------------- generic parameters (C4b wiring)
from fea_core.sweep import (SweepParam, SweepError, apply_params, check_spec, parse_values, _linear_scalable, scale_step,
                            shortcut_blockers)


def _study2():
    st = Study(name="P", geometry="m.step")
    st.materials["S"] = Material(name="S", E=210000.0, nu=0.3, rho=7.85e-9, Re=235.0)
    st.materials["A"] = Material(name="A", E=70000.0, nu=0.33, rho=2.7e-9, Re=160.0)
    st.part_materials["P1"] = "S"
    st.load_cases += [LoadCase("L1", [Force("f", [0, 0, -100.0])]), LoadCase("L2", [Pressure("p", 2.0)])]
    return st


def test_parse_values():
    assert parse_values("1, 1.5 2") == [1.0, 1.5, 2.0]
    assert parse_values("0:1:5") == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert parse_values("1,5; 2,5") == [1.5, 2.5]
    assert parse_values("grid, gyroid", numeric=False) == ["grid", "gyroid"]
    with pytest.raises(SweepError):
        parse_values("a, b")


def test_grid_and_list_modes():
    grid = ParametricStudySpec.from_params([SweepParam("case_factor", [1, 2], target="L1"),
                                            SweepParam("material_prop", [1e5, 2e5, 3e5], target="S", field="E")])
    assert len(grid.combinations()) == 6 and grid.total() == 6
    lst = ParametricStudySpec.from_params([SweepParam("mesh_size", [5, 4, 3]), SweepParam("load_factor", [1, 2, 3])],
                                          mode="list")
    assert lst.combinations() == [{"mesh_size": 5, "load_factor": 1}, {"mesh_size": 4, "load_factor": 2},
                                  {"mesh_size": 3, "load_factor": 3}]
    bad = ParametricStudySpec.from_params([SweepParam("mesh_size", [5, 4]), SweepParam("load_factor", [1])], mode="list")
    assert any("same number" in e for e in check_spec(_study2(), bad))


def test_apply_params_copies_and_changes():
    st = _study2()
    v = apply_params(st, {"case_factor:L1": 3.0, "prop:S:E": 100000.0, "material:P1": "A", "mesh_size": 2.0})
    assert v.load_cases[0].loads[0].vector == [0.0, 0.0, -300.0]
    assert v.load_cases[1].loads[0].value == 2.0                   # other case unchanged
    assert v.materials["S"].E == 100000.0 and v.part_materials["P1"] == "A" and v.mesh.size == 2.0
    assert st.materials["S"].E == 210000.0 and st.load_cases[0].loads[0].vector == [0, 0, -100.0]   # original unchanged
    with pytest.raises(SweepError):
        apply_params(st, {"case_factor:nope": 2.0})
    errs = check_spec(st, ParametricStudySpec.from_params([SweepParam("material", ["Unknown"], target="P1"),
                                                           SweepParam("print", [2, 3], target="P1", field="walls")]))
    assert len(errs) == 2


def test_linear_shortcut_per_case_factors(tmp_path, monkeypatch):
    """Case factors in a linear study: one analysis, KPI of each case scaled by its own factor."""
    st = _study2()
    nodes = np.array([[0.0, 0, 0], [10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
    mesh = Mesh(nodes, np.array([[0, 1, 2, 3]]), np.zeros(1, dtype=np.int32), ["P1"])
    s1 = StepResult("L1", np.full((4, 3), 0.1), np.array([[50.0, 0, 0, 0, 0, 0]] * 4))
    s2 = StepResult("L2", np.full((4, 3), 0.01), np.array([[10.0, 0, 0, 0, 0, 0]] * 4))
    calls = []
    monkeypatch.setattr("fea_core.sweep.run_study", lambda *a, **k: calls.append(1) or Results(mesh, [s1, s2], [], {}))
    assert _linear_scalable(st)
    spec = ParametricStudySpec.from_params([SweepParam("case_factor", [1.0, 0.1], target="L1"),
                                            SweepParam("case_factor", [1.0, 10.0], target="L2")])
    res = run_parametric_study(st, spec, tmp_path)
    assert len(calls) == 1 and res.linear_shortcut and len(res.points) == 4
    p = {(d.params["case_factor:L1"], d.params["case_factor:L2"]): d for d in res.points}
    assert p[(1.0, 1.0)].max_stress == pytest.approx(50.0)
    assert p[(0.1, 10.0)].max_stress == pytest.approx(100.0)           # L2 ×10 governs
    assert p[(0.1, 1.0)].min_safety == pytest.approx(235.0 / 10.0)     # L1 5 MPa, L2 10 MPa


# ---------------------------------------------------------------- review fixes: KPI meaning, keep_results, mesh, rotation
def test_scaled_step_is_a_deep_copy_and_drops_nonlinear_fields():
    base = _mock_results(fields={"print_safety": np.full(4, 3.0), "stress_error": np.full(4, 2.0)}).steps[0]
    new, dropped = scale_step(base, 2.0)
    assert "print_safety" in dropped and "print_safety" not in new.fields
    assert new.fields["stress_error"] is not base.fields["stress_error"]
    new.fields["stress_error"][:] = 99.0
    assert base.fields["stress_error"][0] == 2.0                    # base untouched
    assert new.reactions["end"][2] == pytest.approx(200.0) and base.reactions["end"][2] == 100.0
    assert new.equilibrium["reaction_total"][2] == pytest.approx(200.0)
    assert new.equilibrium["imbalance"] == pytest.approx(1e-9)       # ratio unchanged


def test_min_safety_none_without_re(tmp_path, monkeypatch):
    """A material without Re has no safety (inf = unknown, not 0 – that would read as a failure)."""
    st = _beam_study()
    st.materials["STEEL"].Re = None
    monkeypatch.setattr("fea_core.sweep.run_study", lambda s, wd, **kw: _mock_results())
    res = run_parametric_study(st, ParametricStudySpec(load_factors=[1.0, 2.0]), tmp_path / "o")
    assert res.linear_shortcut
    assert all(math.isinf(p.min_safety) and math.isinf(p.min_safety_raw) for p in res.points)
    assert all(p.to_dict()["min_safety"] is None and p.to_dict()["min_safety_raw"] is None for p in res.points)
    assert "–" in res.summary_table()


def _fake_hotspots(real_node):
    """hotspots() answer: the highest place (node 1) is a singularity, the next one is `real_node` (1-based)."""
    def hs(sess, step, top_n=5, field="mises", tags=False):
        return {"hotspots": [{"node": 1, "classification": "singularity"},
                             {"node": 2, "classification": "affected by boundary condition"},
                             {"node": real_node, "classification": "probably real concentration"}]}
    return hs


@pytest.mark.parametrize("shortcut", [True, False])
def test_min_safety_is_taken_outside_singularities(tmp_path, monkeypatch, shortcut):
    """min_safety = Re / von Mises at the first place that is neither a singularity nor affected by a boundary condition (the GUI
    Assessment card); min_safety_raw = Re / the raw maximum. The Pareto feasibility uses min_safety."""
    st = _beam_study()
    res0 = _mock_results()
    res0.steps[0].S[:] = 0.0
    res0.steps[0].S[0, 0] = 400.0                    # node 1: the raw maximum (a singularity), over the yield strength
    res0.steps[0].S[3, 0] = 100.0                    # node 4: the real concentration
    res0.steps[0].S[1:3, 0] = 30.0
    monkeypatch.setattr("fea_core.mcp.diagnostics.hotspots", _fake_hotspots(4))
    monkeypatch.setattr("fea_core.sweep.run_study", lambda s, wd, **kw: res0)
    spec = (ParametricStudySpec(load_factors=[1.0, 2.0]) if shortcut else
            ParametricStudySpec.from_params([SweepParam("load_factor", [1.0, 2.0]), SweepParam("mesh_size", [5.0])]))
    out = run_parametric_study(st, spec, tmp_path / "o")
    assert out.linear_shortcut is shortcut
    p1, p2 = out.points
    assert p1.max_stress == pytest.approx(400.0)
    assert p1.max_stress_clean == pytest.approx(100.0)
    assert p1.min_safety == pytest.approx(2.5)                       # 250 / 100
    assert p1.min_safety_raw == pytest.approx(250.0 / 400.0)         # 0.625: the singularity would read as a failure
    if shortcut:                                                     # exact scaling of all of them
        assert p2.max_stress_clean == pytest.approx(200.0) and p2.min_safety == pytest.approx(1.25)
        assert p2.min_safety_raw == pytest.approx(250.0 / 800.0)
    # feasibility: with 1.5 required the point with the singular raw safety 0.6 stays feasible (its real safety is 2.5)
    a = DesignPoint(0, {}, mass=1.0, max_stress=400.0, max_stress_clean=100.0, min_safety=2.5, min_safety_raw=0.6)
    assert identify_pareto([a], required_safety=1.5) == [0]


def test_negative_factor_with_rotation_fails_like_the_full_path(tmp_path, monkeypatch):
    """ω scales with √factor – a negative factor on a rotation with ω ≠ 0 is an error of the point in the full path; the linear
    shortcut reports the same error instead of rescaling the baseline."""
    st = _beam_study()
    st.load_cases[0].loads.append(Rotation(point=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], omega=10.0))
    monkeypatch.setattr("fea_core.sweep.run_study", lambda s, wd, **kw: _mock_results())
    short = run_parametric_study(st, ParametricStudySpec(load_factors=[1.0, -1.0, 2.0]), tmp_path / "a")
    assert short.linear_shortcut
    assert [p.converged for p in short.points] == [True, False, True]
    assert "Negative load_factor" in short.points[1].error
    full = run_parametric_study(st, ParametricStudySpec.from_params(
        [SweepParam("load_factor", [-1.0]), SweepParam("mesh_size", [5.0])]), tmp_path / "b")
    assert not full.linear_shortcut and not full.points[0].converged
    assert full.points[0].error == short.points[1].error             # the same message as the full path


def _mock_run(workdirs):
    def run(study, wd, **kw):
        wd = Path(wd)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "job.inp").write_text("*NODE\n", encoding="utf-8")
        workdirs.append(wd)
        return _mock_results()
    return run


@pytest.mark.parametrize("shortcut", [True, False])
def test_keep_results(tmp_path, monkeypatch, shortcut):
    """keep_results=False deletes the work folder of every evaluated point; True keeps it and saves DP_xxx/results.npz
    (the shortcut: the baseline scaled by the factor of the point)."""
    from fea_core.results import Results
    runs = []
    monkeypatch.setattr("fea_core.sweep.run_study", _mock_run(runs))
    spec = (ParametricStudySpec(load_factors=[1.0, 2.0]) if shortcut else
            ParametricStudySpec.from_params([SweepParam("load_factor", [1.0, 2.0]), SweepParam("mesh_size", [5.0])]))
    out = run_parametric_study(_beam_study(), spec, tmp_path / "drop")
    assert out.linear_shortcut is shortcut and len(out.points) == 2
    assert runs and not any(d.exists() for d in runs), "work folders of the evaluated points are deleted"
    assert not list((tmp_path / "drop").glob("DP_*")) and not (tmp_path / "drop" / "baseline").exists()
    runs.clear()
    out = run_parametric_study(_beam_study(), spec, tmp_path / "keep", keep_results=True)
    saved = sorted((tmp_path / "keep").glob("DP_*/results.npz"))
    assert len(saved) == 2 and all(d.exists() for d in runs)
    if shortcut:
        # the stored result of DP1 = 2 x the baseline (stress 50 MPa, displacement 0.1 mm)
        r = Results.load(saved[1])
        assert float(r.steps[0].S[0, 0]) == pytest.approx(100.0) and float(r.steps[0].U[1, 0]) == pytest.approx(0.2)
        assert float(Results.load(saved[0]).steps[0].S[0, 0]) == pytest.approx(50.0)


def test_failed_point_keeps_its_work_folder(tmp_path, monkeypatch):
    runs = []

    def run(study, wd, **kw):
        wd = Path(wd)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "job.log").write_text("fail\n", encoding="utf-8")
        runs.append(wd)
        if len(runs) == 1:
            raise RuntimeError("ccx failed")
        return _mock_results()
    monkeypatch.setattr("fea_core.sweep.run_study", run)
    spec = ParametricStudySpec.from_params([SweepParam("mesh_size", [5.0, 4.0])])
    out = run_parametric_study(_beam_study(), spec, tmp_path / "o")
    assert [p.converged for p in out.points] == [False, True]
    assert runs[0].exists() and not runs[1].exists()                 # the failed one stays for the diagnosis


def test_taken_over_mesh_is_used_and_not_changed(tmp_path, monkeypatch):
    """mesh=: the baseline and the points whose mesh signature matches use it (no meshing, no parts); the mesh of the caller is
    copied (the print assignment written into a taken-over mesh would otherwise change the mesh shown in the GUI)."""
    mesh = _mock_results().mesh
    mesh.cell_data["marker"] = np.zeros(1)
    seen = []

    def run(study, wd, mesh=None, parts=None, **kw):
        seen.append((mesh, parts))
        if mesh is not None:
            mesh.cell_data["print_shell"] = np.ones(1)               # what _ensure_print does
        return _mock_results()
    monkeypatch.setattr("fea_core.sweep.run_study", run)
    st = _beam_study()
    # pure load sweep (shortcut): the baseline gets the mesh
    run_parametric_study(st, ParametricStudySpec(load_factors=[1.0, 2.0]), tmp_path / "a", mesh=mesh, parts=["P"])
    assert seen[0][0] is not None and seen[0][1] is None and seen[0][0] is not mesh
    assert "print_shell" not in mesh.cell_data and "marker" in seen[0][0].cell_data
    # materials do not change the mesh signature -> every point uses the mesh; mesh_size does -> re-meshed (with the parts)
    seen.clear()
    st.materials["B"] = Material(name="B", E=70000.0, nu=0.33, rho=2.7e-9, Re=100.0)
    run_parametric_study(st, ParametricStudySpec.from_params([SweepParam("material", ["STEEL", "B"], target="Part1")]),
                         tmp_path / "b", mesh=mesh, parts=["P"])
    assert [m is not None for m, _ in seen] == [True, True] and all(p is None for _, p in seen)
    seen.clear()
    run_parametric_study(st, ParametricStudySpec.from_params([SweepParam("mesh_size", [3.0, 2.0])]), tmp_path / "c",
                         mesh=mesh, parts=["P"])
    assert [m is not None for m, _ in seen] == [False, False] and all(p == ["P"] for _, p in seen)

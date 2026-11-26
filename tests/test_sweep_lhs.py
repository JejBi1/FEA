"""Parametric study, mode "lhs" (Latin hypercube) and the kind sketch_param (F15-S4 / F15-S5): the points, the checks, the 2D KPI and the
rebuild of the geometry per design point."""
from __future__ import annotations

import pytest

from fea_core.sweep import (MAX_LHS_PARAMS, ParametricStudySpec, SweepError, SweepParam, apply_params, check_spec, has_sketch_params,
                            run_parametric_study)
from fea_core.study import Study


def spec(mode="lhs", n=12, seed=0, **kw):
    ps = kw.pop("params", None) or [SweepParam("load_factor", bounds=(0.5, 2.0)), SweepParam("mesh_size", bounds=(2.0, 6.0))]
    return ParametricStudySpec.from_params(ps, mode=mode, max_points=n, seed=seed)


def test_lhs_points_are_seeded_inside_the_bounds_and_distinct():
    a = spec().combinations()
    assert len(a) == 12 and a == spec().combinations() and a != spec(seed=1).combinations()
    assert all(0.5 <= r["load_factor"] <= 2.0 and 2.0 <= r["mesh_size"] <= 6.0 for r in a)
    assert len({r["load_factor"] for r in a}) == 12                         # a Latin hypercube never repeats a value
    assert spec(n=7).total() == 7 and spec(n=500).total() == 200           # the hard ceiling of one study


def test_lhs_categorical_values_are_spread_evenly():
    ps = [SweepParam("load_factor", bounds=(1.0, 2.0)), SweepParam("material", values=["steel", "alu", "ti"], target="Part")]
    rows = spec(params=ps, n=12).combinations()
    cats = [r["material:Part"] for r in rows]
    assert {c: cats.count(c) for c in ("steel", "alu", "ti")} == {"steel": 4, "alu": 4, "ti": 4}


def test_lhs_spec_errors():
    with pytest.raises(SweepError, match="at least one continuous"):
        spec(params=[SweepParam("material", values=["a", "b"], target="P")]).combinations()
    with pytest.raises(SweepError, match="lo < hi"):
        spec(params=[SweepParam("load_factor", bounds=(2.0, 1.0))]).combinations()
    with pytest.raises(SweepError, match="at most"):
        spec(params=[SweepParam("case_factor", target=f"c{i}", bounds=(1.0, 2.0)) for i in range(MAX_LHS_PARAMS + 1)]).combinations()
    with pytest.raises(SweepError, match="together"):
        spec(params=[SweepParam("load_factor", values=[1, 2], bounds=(1.0, 2.0))]).combinations()
    with pytest.raises(SweepError, match="only in mode lhs"):
        spec(mode="grid").combinations()
    with pytest.raises(SweepError, match="bounds must be"):
        SweepParam.from_dict({"kind": "load_factor", "bounds": ["a", 2]})


def test_sweep_param_roundtrip_and_keys():
    p = SweepParam("sketch_param", target="W", bounds=(30, 80))
    assert p.key == "sketch:W" and SweepParam.from_dict(p.to_dict()) == p
    assert ParametricStudySpec.from_params([p], mode="lhs", max_points=9, seed=4).to_dict()["seed"] == 4
    assert has_sketch_params({"sketch:W": 1.0, "mesh_size": 2.0}) and not has_sketch_params({"mesh_size": 2.0})


def test_check_spec_lhs_and_sketch_param_need_a_sketch():
    st = Study(name="x", geometry="")
    errs = check_spec(st, spec(params=[SweepParam("sketch_param", target="W", bounds=(30, 80))]))
    assert any("no sketch" in e for e in errs)
    errs = check_spec(st, spec(params=[SweepParam("load_factor")]))
    assert any("at least one continuous" in e for e in errs)
    with pytest.raises(SweepError, match="no sketch"):
        apply_params(st, {"sketch:W": 50.0}, workdir=".")


@pytest.mark.slow
def test_lhs_sketch_parameter_sweep_rebuilds_the_geometry_per_point(tmp_path):
    """A 2D plate with a hole drawn in the sketcher: hole radius and plate width swept; every point rebuilds, meshes and solves; the
    2D KPI (max stress, displacement, mass) are filled and move the right way (a bigger hole = a higher stress and a lighter plate)."""
    import verification.cases  # noqa: F401 – the import order of the verification package
    from verification.sketch_cases import _plate_study
    st = _plate_study(tmp_path / "base")
    sp = ParametricStudySpec.from_params([SweepParam("sketch_param", target="R", bounds=(3.0, 12.0)),
                                          SweepParam("sketch_param", target="W", bounds=(40.0, 60.0))], mode="lhs", max_points=4, seed=2)
    assert check_spec(st, sp) == []
    res = run_parametric_study(st, sp, tmp_path / "run")
    assert [p.converged for p in res.points] == [True] * 4, [p.error for p in res.points]
    assert all(p.max_stress > 100 and p.max_disp > 0 and p.mass > 0 and p.max_stress_clean is None for p in res.points)
    by_r = sorted(res.points, key=lambda p: p.params["sketch:R"])
    assert by_r[-1].max_stress > by_r[0].max_stress
    assert len({p.params["sketch:W"] for p in res.points}) == 4
    assert res.sweep["mode"] == "lhs" and res.sweep["seed"] == 2 and res.fingerprint == ""     # the caller stamps the fingerprint

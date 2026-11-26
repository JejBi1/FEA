"""The stale mark of a surrogate (F15-S4 re-check N2, N4): a swept load factor hides ONLY the common magnitude of the loads it scales –
a changed direction, one force of two, an application point, a temperature or another case's load makes the model stale."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from fea_core.study import Force, LoadCase, Material, Pressure, RemoteLoad, Rotation, Study, ThermalLoad
from fea_core.surrogate import fit_surrogate, lhs_design, study_fingerprint
from fea_core.sweep import scale_study_loads


def make_study() -> Study:
    st = Study(name="P", geometry="")
    st.materials["S"] = Material(name="S", E=210000.0, nu=0.3, rho=7.85e-9, Re=235.0)
    st.part_materials["P1"] = "S"
    st.load_cases += [LoadCase("L1", [Force("f", [0, 0, -100.0]), RemoteLoad("r", point=[10.0, 0.0, 0.0], force=[0, 50.0, 0])]),
                      LoadCase("L2", [Pressure("p", 2.0), ThermalLoad(value=120.0), Rotation(omega=30.0)])]
    return st


def same(keys, edit) -> bool:
    st = make_study()
    ch = copy.deepcopy(st)
    edit(ch)
    return study_fingerprint(ch, keys) == study_fingerprint(st, keys)


K = ["load_factor", "mesh_size"]


@pytest.mark.parametrize("name,edit", [
    ("force direction", lambda s: setattr(s.load_cases[0].loads[0], "vector", [100.0, 0, 0])),
    ("one of two forces x5", lambda s: setattr(s.load_cases[0].loads[0], "vector", [0, 0, -500.0])),
    ("remote point 10 -> 400 mm", lambda s: setattr(s.load_cases[0].loads[1], "point", [400.0, 0, 0])),
    ("thermal 120 -> 400 degC", lambda s: setattr(s.load_cases[1].loads[1], "value", 400.0)),
    ("another case's pressure x10", lambda s: setattr(s.load_cases[1].loads[0], "value", 20.0)),
    ("a load removed", lambda s: s.load_cases[0].loads.pop()),
    ("the rotation speed alone x4", lambda s: setattr(s.load_cases[1].loads[2], "omega", 120.0)),
])
def test_a_swept_load_factor_still_sees_these_changes(name, edit):
    assert not same(K, edit), name


@pytest.mark.parametrize("factor", [3.0, 0.37, 11.0, -1.0])
def test_the_common_factor_of_a_taken_over_point_is_not_a_change(factor):
    st = make_study()
    if factor < 0:                                       # the rotation (omega ~ sqrt(factor)) cannot be reversed: the other loads can
        st.load_cases[1].loads = [ld for ld in st.load_cases[1].loads if not isinstance(ld, Rotation)]
    ch = copy.deepcopy(st)
    scale_study_loads(ch, factor)
    assert study_fingerprint(ch, K) == study_fingerprint(st, K)
    assert study_fingerprint(ch, []) != study_fingerprint(st, [])


def test_a_case_factor_hides_only_its_own_case():
    k = ["case_factor:L1"]
    assert same(k, lambda s: scale_study_loads(s, 1.7, ["L1"]))
    assert not same(k, lambda s: setattr(s.load_cases[0].loads[0], "vector", [100.0, 0, 0]))          # the direction of the case itself
    assert not same(k, lambda s: setattr(s.load_cases[1].loads[0], "value", 20.0))                  # another case: the pressure x10
    assert not same(k, lambda s: scale_study_loads(s, 1.7, ["L2"]))
    both = ["load_factor", "case_factor:L1"]
    assert same(both, lambda s: (scale_study_loads(s, 2.0), scale_study_loads(s, 1.5, ["L1"])))
    assert not same(both, lambda s: setattr(s.load_cases[1].loads[0], "value", 20.0))


def test_a_swept_property_hides_nothing_of_the_loads():
    assert not same(["mesh_size"], lambda s: scale_study_loads(s, 3.0))


def test_a_swept_key_that_is_constant_over_the_points_does_not_make_the_model_stale():
    """N4: the fit drops a swept parameter that has one value at every point, but the stored fingerprint neutralised it – so must the check."""
    st = make_study()
    keys = ["prop:S:E", "mesh_size"]
    X = lhs_design([(1e5, 2e5)], 40, seed=1)
    pts = [{"params": {"prop:S:E": float(x[0]), "mesh_size": 5.0}, "max_stress": float(100 + 1e-4 * x[0]), "converged": True} for x in X]
    sm = fit_surrogate(pts, keys=keys, backend="rbf", fingerprint=study_fingerprint(st, keys))
    assert [p.key for p in sm.params] == ["prop:S:E"] and sm.fingerprint_keys == keys
    assert sm.check_study(st) == ""
    st.mesh.size = 2.0                                                     # the swept mesh size at another value: not a change
    st.materials["S"].E = 150000.0
    assert sm.check_study(st) == ""
    assert sm.refit_with([{"params": {"prop:S:E": 1.5e5, "mesh_size": 5.0}, "kpi": {"max_stress": 115.0}}]).fingerprint_keys == keys
    st.load_cases[0].loads[0].vector = [0, 0, -500.0]
    assert sm.check_study(st)                                              # a real change
    # the keys travel with the file
    import tempfile
    from pathlib import Path

    from fea_core.surrogate import load_surrogate
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "s.npz"
        sm.save(f)
        assert load_surrogate(f).fingerprint_keys == keys


def test_the_fingerprint_does_not_flip_on_rounding_of_a_rescaled_copy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        st = make_study()
        st.load_cases[0].loads[0].vector = [float(v) for v in rng.normal(size=3) * 1000]
        ch = copy.deepcopy(st)
        scale_study_loads(ch, float(rng.uniform(0.1, 10.0)))
        assert study_fingerprint(ch, K) == study_fingerprint(st, K)

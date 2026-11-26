"""COC / AOC on a REAL solved tet10 model (F14-S5, review rounds 3 and 4): a cantilever of 6837 nodes solved by ccx with a heavy point mass on the tip face. The reviewer's case: FE modes 3 - 10
compared, test mode 4 = 0.8 phi_4 + 0.6 phi_1 (M-normalised parts, true COC 0.80) - the projection onto the compared window gives COC 0.9957 (the part in mode 1 is invisible to it and the added
mass sits at the tip, which the corner-node sensors do not carry). Owner decision 25. 11. 2026: the SEREP COC is INDICATIVE - its rows are UNCONFIRMED, never PASS and never FAIL, so this
case must not be a PASS (it was, before round 3) and the good tests are not judged either. Needs Gmsh and CalculiX (slow)."""
from __future__ import annotations

import numpy as np
import pytest

from verification import cases as _cases  # noqa: F401  (import order of the verification package)
from verification.dynamics_cases import _beam
from verification.space_cases import make_requirement_set

from fea_core.space import corr_orth as O
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from fea_core.space.model import LimitEntry, SpaceSettings
from fea_core.space.requirements import confirm

pytestmark = pytest.mark.slow

MASS = 8e-4                                                   # [t]; the beam itself is 7.85e-5 t


def _limits_space(values):
    rs = make_requirement_set("protoflight", confirmed=False)
    for k, v in values.items():
        rs.limits.append(LimitEntry(key=k, value=v, unit=C.LIMIT_INFO[k][0], document="doc", revision="rev", clause="c1"))
    sp = SpaceSettings(requirement_set=rs)
    confirm(rs, "test engineer", "2026-01-01", scope=sp)
    return sp


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from fea_core.solve import run_study
    wd = tmp_path_factory.mktemp("real_cantilever")
    st, _parts = _beam(wd, "round3 real cantilever", size=2.5, mass=MASS)
    st.dynamics.enabled = False
    st.modal.enabled, st.modal.modes = True, 10
    st.solver.modal_engine = "ccx"
    st.save(wd / "study.json")
    res = run_study(st, wd / "run")
    ms = CR.modal_sets(res)[0]
    st.space = _limits_space({"corr_mac_min": 0.9, "corr_mac_offdiag_max": 0.5, "corr_comac_min": 0.0, "corr_freq_error_max": 0.05, "corr_coc_diag_min": 0.9, "corr_coc_offdiag_max": 0.1,
                              "corr_aoc_offdiag_max": 0.1, "corr_modulus_diff_max": 1.0})
    return st, res, res.mesh, ms


def _sensors(st, mesh, tip):
    nodes = mesh.nodes

    def nearest(p, mask=None):
        d = np.linalg.norm(nodes - np.asarray(p, float), axis=1)
        return int(np.argmin(d if mask is None else np.where(mask, d, np.inf)))
    along = [(15, 0, 0), (30, 10, 0), (45, 0, 10), (60, 10, 10), (75, 0, 5), (90, 10, 5), (50, 5, 10), (20, 5, 0)]
    base = [{"node": str(100 + k), "point": [float(v) for v in nodes[nearest(p)]]} for k, p in enumerate(along)]
    return C.sensors_from_dicts(base + tip, st)


CORNER_TIP = [{"node": "200", "point": [100.0, 0.0, 0.0]}, {"node": "201", "point": [100.0, 10.0, 10.0]}]
FACE_SELECTION = [{"node": "200", "selection": "cel"}]


def _run(model, tip, target=None, a1=0.0, noise=0.0, seed=1):
    st, res, mesh, ms = model
    S = _sensors(st, mesh, tip)
    dofs = [(s.node, d) for s in S for d in s.dofs]
    fphi, _ = C.fe_shapes_at_sensors(ms, mesh, S, dofs)
    dn = np.sqrt(np.diag(O.serep_gram(st, mesh, ms, list(range(len(ms.values))))))
    td = C.fe_test_data(ms, mesh, S, damping=0.01)
    td.modes = [m for m in td.modes if m.number >= 3]
    rng = np.random.default_rng(seed)
    if target is not None:
        a3 = float(np.sqrt(1 - a1 * a1))
        shp = a3 * fphi[:, target - 1] / dn[target - 1] + a1 * fphi[:, 0] / dn[0]
        for i, key in enumerate(dofs):
            td.modes[target - 3].shape[key] = float(shp[i])
    if noise:
        for m in td.modes:
            rms = float(np.sqrt(np.mean([v * v for v in m.shape.values()])))
            for key in list(m.shape):
                m.shape[key] += noise * rms * float(rng.standard_normal())
    opt = C.CorrelationOptions(fe_modes=list(range(3, 11)), f_min_hz=float(ms.values[2]) * 0.9)
    r, _ = CR.run_correlation(st, res, td, sensors=S, options=opt, store=False)
    return r


def _coc_row(r, target):
    return next(x for x in r.rows if x.metric == "COC diagonal" and x.item.startswith(f"test mode {target} <->"))


@pytest.mark.parametrize("tip", [CORNER_TIP, FACE_SELECTION], ids=["tip sensors at the corner nodes", "one sensor on the tip face"])
def test_the_reviewers_mixed_test_mode_is_never_a_pass_and_the_indicative_value_is_shown(model, tip):
    ctrl = _run(model, tip)
    o = ctrl.modal.orth
    assert o.computed and o.info["basis_modes"] == 8 and o.info["fe_modes"] == 8 and o.indicative == C.INDICATIVE_REASON
    assert not o.mode_hold and o.hold == "" and np.nanmin(np.diag(o.coc[:, :])) > 0.999                    # a good test: the FE modes themselves, the indicative COC is 1
    assert ctrl.verdict == C.VERDICT_UNCONFIRMED and not any(x.status == C.PASS for x in ctrl.rows if x.metric.startswith(("COC", "AOC")))
    r = _run(model, tip, target=4, a1=0.6)
    p = next(x for x in r.modal.pairs if x.test_number == 4)
    row = _coc_row(r, 4)
    assert p.coc is not None and p.coc > 0.9 and row.value is not None and row.value > 0.9            # the projection onto the window: the number that was a PASS in round 2 (true COC 0.80)
    assert row.status == C.UNCONFIRMED and C.INDICATIVE_REASON in row.note and r.verdict != C.PASS       # ... and is now UNCONFIRMED, never PASS, never FAIL
    assert not any(x.status == C.FAIL for x in r.rows if x.metric.startswith(("COC", "AOC")))


@pytest.mark.parametrize("sigma", [0.01, 0.02])
def test_a_good_test_with_noise_on_the_real_model_is_not_held(model, sigma):
    """The reviewer's noise cases on the real layout (a sensor on the face carrying the added mass): the indicative COC is computed, no hold rule fires (there are none), every row is UNCONFIRMED
    with the indicative reason and none is PASS or FAIL."""
    for tip in (CORNER_TIP, FACE_SELECTION):
        for seed in range(3):
            r = _run(model, tip, noise=sigma, seed=seed)
            o = r.modal.orth
            assert o.computed and not o.mode_hold and o.hold == "", (sigma, seed, o.mode_hold, o.hold)
            orth_rows = [x for x in r.rows if x.metric.startswith(("COC", "AOC"))]
            assert orth_rows and {x.status for x in orth_rows} == {C.UNCONFIRMED} and all(x.value is not None and C.INDICATIVE_REASON in x.note for x in orth_rows)
            assert np.nanmax(o.residual) < 0.05

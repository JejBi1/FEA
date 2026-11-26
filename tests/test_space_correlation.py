"""F14 stage S5 - vibration test correlation (`fea_core.space.correlation`).

Gate of the stage:
  * a synthetic dataset exported from the FE model itself gives MAC = 1, a frequency error of 0 and zero FRF error (CSV exactly, UFF to its six printed digits);
  * perturbed datasets (shifted frequencies, swapped mode order, noisy sensors, missing DOFs) test the pairing and the pass / fail outcome;
  * the importers reject truncated, malformed and oversized files cleanly (only `TestDataError`, never a crash).
The limits used here are EXAMPLE values chosen by the tests - the program ships none; without a confirmed requirement set no metric may be PASS.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core.eigen import ModeSet
from fea_core.mesh import Mesh
from fea_core.space import _corr_io as io
from fea_core.space import correlation as C
from fea_core.space.model import LimitEntry, SpaceSettings
from fea_core.space.requirements import confirm
from fea_core.study import Study
from verification.space_cases import make_requirement_set

L, W, H = 100.0, 20.0, 10.0
FREQS = [10.0, 12.5, 62.0, 75.0, 120.0, 150.0]


# ------------------------------------------------------------------ a small FE model with analytic modes
def _box_mesh(nx: int = 5, ny: int = 2, nz: int = 2) -> Mesh:
    xs, ys, zs = np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1), np.linspace(0, H, nz + 1)
    nodes = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k          # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]    # corners 000 001 010 011 100 101 110 111
                for t in ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)):
                    tets.append([c[v] for v in t])
    tets_a = np.array(tets, dtype=np.int64)
    return Mesh(nodes=nodes, tets=tets_a, tet_part=np.zeros(len(tets_a), dtype=np.int64), part_names=["bar"])


def _modes(mesh: Mesh) -> ModeSet:
    x, y, z = mesh.nodes[:, 0] / L, mesh.nodes[:, 1] / W - 0.5, mesh.nodes[:, 2] / H - 0.5
    psi1, psi2 = np.sin(np.pi * x / 2), np.sin(3 * np.pi * x / 2)         # orthogonal on the bar
    fields = [(np.zeros_like(x), psi1, 0 * x), (0 * x, 0 * x, psi1), (0 * x, psi2, 0 * x), (0 * x, 0 * x, psi2), (x, 0 * x, 0 * x),
              (0 * x, -z * x, y * x)]
    sh = np.zeros((len(fields), len(x), 3), dtype=np.float32)
    for k, (a, b, c) in enumerate(fields):
        u = np.stack([a, b, c], axis=1)
        sh[k] = u / np.abs(u).max()
    return ModeSet("modal", "modes", list(FREQS), sh)


SENSORS = C.sensors_from_dicts([{"node": str(100 + k), "point": [10.0 + 12.0 * k, 5.0 + (k % 3) * 4.0, 2.0 + (k % 2) * 6.0]} for k in range(8)])


@pytest.fixture(scope="module")
def fe():
    mesh = _box_mesh()
    return mesh, _modes(mesh)


# example limits of the orthogonality checks of THESE tests (not values of a standard)
ORTH_EXAMPLE = {"corr_coc_diag_min": 0.8, "corr_coc_offdiag_max": 0.6, "corr_aoc_offdiag_max": 0.6, "corr_modulus_diff_max": 0.5}


def _limits_set(values: dict[str, float], confirmed: bool = True, provenance: bool = True) -> SpaceSettings:
    rs = make_requirement_set("protoflight", confirmed=False)
    for k, v in values.items():
        rs.limits.append(LimitEntry(key=k, value=v, unit=C.LIMIT_INFO[k][0], document="doc" if provenance else "", revision="rev" if provenance else "",
                                    clause="c1" if provenance else ""))
    sp = SpaceSettings(requirement_set=rs)
    if confirmed:
        confirm(rs, "test engineer", "2026-01-01", scope=sp)
    return sp


EXAMPLE = {"corr_mac_min": 0.9, "corr_mac_offdiag_max": 0.3, "corr_comac_min": 0.9, "corr_freq_error_max": 0.05, "corr_peak_freq_shift_max": 0.03,
           "corr_peak_amp_error_max_db": 3.0, "corr_rms_error_max": 0.2, **ORTH_EXAMPLE}


def _unit_mass_gram(ms: ModeSet) -> np.ndarray:
    """Gram matrix of ALL the modes under a unit mass at every node (the synthetic model has no densities and a tet4 mesh), reduced to its diagonal: the analytic modes of this model are only
    orthogonal in the continuum, not on the coarse nodes, and the guard against a mass that is not the mass of the modes (off-diagonal terms above 1e-3) is tested in test_space_corr_orthogonality.py."""
    flat = np.asarray(ms.shapes, dtype=np.float64).reshape(len(ms.values), -1)
    return np.diag(np.sum(flat * flat, axis=1))


FIXTURE_MODEL_KEY = "fixture-model"


def _tam_of_gram(gram_mass: C.OrthMass, tp, fp, dofs, modeset, fe_idx) -> C.OrthMass:
    """The pipeline tests need a COC / AOC that can PASS, which only an imported TAM can (owner decision 25. 11. 2026; the SEREP route is indicative). This is a TAM made from the Gram matrix G of
    the compared FE modes: M = (Phi_a^+)^T G Phi_a^+ + eps (I - Phi_a Phi_a^+) - it reproduces G exactly (so the FE orthogonality and the generalised masses are those of the model) and gives the
    directions outside the FE modes a small, full-rank mass (a regularised SEREP: the matrix is blind to shapes outside the FE modes, which is fine for a test of the PIPELINE, not of the physics)."""
    from fea_core.space import corr_orth as O
    m = fp.shape[1]
    G = O._serep_gram_of(gram_mass, modeset, fe_idx, m)
    pin = np.linalg.pinv(np.asarray(fp, dtype=np.float64))
    ma = pin.T @ G @ pin
    n = len(dofs)
    ma = ma + 0.02 * float(np.trace(ma)) / n * (np.eye(n) - np.asarray(fp, dtype=np.float64) @ pin)
    tam = C.MassMatrix(list(dofs), 0.5 * (ma + ma.T), "fixture-tam", "", rank=n)
    tam.model_key = FIXTURE_MODEL_KEY
    return C.OrthMass("tam", study=gram_mass.study, tam=tam, gen_mass=np.diag(G).copy())


@pytest.fixture(autouse=True)
def _orthogonality_mass(monkeypatch):
    """The synthetic model has no densities (and a tet4 mesh), so the SEREP mass of the study cannot be computed from it: every correlation of these tests takes the Gram matrix of the
    modes under a unit nodal mass instead (`OrthMass(gram=...)`) and turns it into an imported-TAM-like mass (`_tam_of_gram`): the SEREP route is INDICATIVE (never PASS / FAIL) and a stored
    SEREP PASS is not accepted as current, so a pipeline test (verdict, provenance, storing, report, MCP, GUI) reaches PASS the only way the program allows: through a TAM. The rule itself
    is tested in test_space_corr_orthogonality.py and test_space_corr_round4.py (with the real `orthogonality`)."""
    from fea_core.space import corr_run as CR
    real = C.correlate
    kept: list[Any] = []                                                       # strong references: the session keeps only weak references to the TAM

    def wrapped(test, modeset=None, mesh=None, *a, orth_mass=None, **k):
        if orth_mass is None and modeset is not None:
            orth_mass = C.OrthMass("serep", gram=_unit_mass_gram(modeset), gram_modes=list(range(len(modeset.values))))
        return real(test, modeset, mesh, *a, orth_mass=orth_mass, **k)

    def build(study, mesh, modeset, opt, tam=None):
        return (C.OrthMass("serep", study=study, gram=_unit_mass_gram(modeset), gram_modes=list(range(len(modeset.values)))) if opt.orth_mass == "serep"
                else real_build(study, mesh, modeset, opt, tam))

    real_orth = C.orthogonality

    def lifted(test_phi, fe_phi, dofs, mass, modeset=None, fe_idx=None, test_numbers=None, axes_hold=""):
        if mass is not None and mass.kind == "serep" and mass.gram is not None:
            probe = real_orth(test_phi, fe_phi, dofs, mass, modeset, fe_idx, test_numbers, axes_hold)              # the SEREP availability rules (sensors, condition, DOFs) hold for the lifted mass too
            if not probe.computed:
                return probe
            mass = _tam_of_gram(mass, test_phi, fe_phi, dofs, modeset, fe_idx)
            if mass.study is not None:
                kept.append(mass.tam)
                CR.remember_tam(mass.study, mass.tam)
        return real_orth(test_phi, fe_phi, dofs, mass, modeset, fe_idx, test_numbers, axes_hold)

    real_build = CR.build_orth_mass
    monkeypatch.setattr(C, "correlate", wrapped)
    monkeypatch.setattr(C, "orthogonality", lifted)
    monkeypatch.setattr(CR, "build_orth_mass", build)
    real_fp = C.model_fingerprint
    monkeypatch.setattr(C, "model_fingerprint", lambda study, mesh: real_fp(study, mesh) or FIXTURE_MODEL_KEY)     # the synthetic tet4 model cannot be fingerprinted: a fixed key stands for its state


def _lim(values: dict[str, float] | None = None) -> C.CorrelationLimits:
    sp = _limits_set(EXAMPLE if values is None else {**ORTH_EXAMPLE, **values})                 # a partial set of the older keys gets the orthogonality limits too
    return C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)


def _test_data(fe) -> C.TestData:
    mesh, ms = fe
    return C.fe_test_data(ms, mesh, SENSORS, damping=0.01)


# ------------------------------------------------------------------ gate 1: the dataset exported from the FE model itself
def test_csv_roundtrip_of_the_fe_modes_gives_mac_one_and_zero_frequency_error(fe):
    mesh, ms = fe
    td = C.parse_csv(C.write_modes_csv(_test_data(fe).modes))
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim())
    assert r.verdict == C.PASS, r.text()
    assert len(r.modal.pairs) == 6 and not r.modal.unpaired_test
    for p in r.modal.pairs:
        assert abs(p.mac - 1.0) < 1e-12 and p.freq_error == 0.0 and p.test_number == p.fe_number and p.damping == 0.01
    assert np.allclose(np.diag(r.modal.mac), 1.0, atol=1e-12)
    assert all(abs(v - 1.0) < 1e-12 for v in r.modal.comac.values() if v is not None) and len(r.modal.comac) == 24


def test_uff55_roundtrip_of_the_fe_modes(fe):
    mesh, ms = fe
    td = C.parse_uff(C.write_uff55(_test_data(fe).modes))
    assert [m.number for m in td.modes] == [1, 2, 3, 4, 5, 6] and td.modes[0].damping == pytest.approx(0.01)
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim())
    assert r.verdict == C.PASS
    assert all(abs(p.mac - 1.0) < 1e-6 and abs(p.freq_error) < 1e-5 for p in r.modal.pairs)


def _frf_pair(fe, with_noise: float = 0.0):
    mesh, ms = fe
    dofs = [("100", "y"), ("103", "z")]
    phi, _ = C.fe_shapes_at_sensors(ms, mesh, SENSORS, dofs)
    f = np.linspace(2.0, 200.0, 400)
    h = C.modal_frf(f, np.array(FREQS), 0.02, phi[0], phi[1], "accelerance")
    return f, h


def test_frf_from_the_model_against_itself_has_zero_error_csv_and_uff(fe):
    f, h = _frf_pair(fe)
    fn = C.TestFunction("100:y", "100", "y", f, h, "frf", "g/N")
    for text, parse, tol in ((C.write_functions_csv([fn]), C.parse_csv, 1e-12), (C.write_uff58([fn]), C.parse_uff, 1e-5)):
        back = parse(text).functions[0]
        assert back.node == "100" and back.dof == "y" and back.kind == "frf" and back.freq.size == f.size
        m = C.compare_response(back.freq, back.values, f, h, "frf", None, "100:y")
        assert abs(m.peak_shift) <= (0.0 if tol < 1e-9 else 1e-5) and abs(m.peak_error_db) < 20 * tol and m.rms_error < 10 * tol and m.points >= 399, (parse, m)           # a regular UFF grid has a six-digit increment: the last point may fall just outside


def test_psd_response_uses_ten_log_and_reports_the_level():
    f = np.linspace(20, 2000, 500)
    asd = 0.04 / (1 + ((f - 300) / 40) ** 2)
    m = C.compare_response(f, asd, f, 2 * asd, "psd")
    assert m.peak_error_db == pytest.approx(10 * math.log10(2), rel=1e-12) and m.level_error == pytest.approx(math.sqrt(2) - 1, rel=1e-9)
    m = C.compare_response(f, asd, f, 2 * asd, "frf")
    assert m.peak_error_db == pytest.approx(20 * math.log10(2), rel=1e-12)


def test_response_peak_shift_amplitude_and_rms():
    f = np.linspace(10, 200, 1900)
    base = lambda f0, a: a / np.sqrt((1 - (f / f0) ** 2) ** 2 + (2 * 0.03 * f / f0) ** 2)      # noqa: E731
    m = C.compare_response(f, base(100, 1.0), f, base(102, 1.0), "frf")
    assert m.peak_shift == pytest.approx(0.02, abs=2e-3) and abs(m.peak_error_db) < 0.1 and 0 < m.rms_error < 0.5
    with pytest.raises(ValueError):
        C.compare_response(f, base(100, 1.0), f + 1000.0, base(100, 1.0), "frf")             # no common band
    with pytest.raises(ValueError):
        C.compare_response(f[:1], np.ones(1), f[:1], np.ones(1))


# ------------------------------------------------------------------ gate 2: perturbed datasets
def test_shifted_frequencies_are_measured_and_judged_by_the_confirmed_limit(fe):
    mesh, ms = fe
    td = _test_data(fe)
    for m in td.modes:
        m.frequency_hz *= 1.03                                    # the FE model is 2.9 % too soft
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim())
    assert all(p.freq_error == pytest.approx(1 / 1.03 - 1) for p in r.modal.pairs) and r.verdict == C.PASS     # 2.9 % < 5 %
    tight = _lim({**EXAMPLE, "corr_freq_error_max": 0.02})
    r = C.correlate(td, ms, mesh, SENSORS, limits=tight)
    assert r.verdict == C.FAIL and {x.metric for x in r.rows if x.status == C.FAIL} == {"frequency error"}


@pytest.mark.parametrize("method", ["hungarian", "max_mac"])
def test_swapped_mode_order_is_paired_by_the_shape(fe, method):
    mesh, ms = fe
    td = _test_data(fe)
    td.modes = [td.modes[i] for i in (3, 0, 5, 1, 4, 2)]                         # the test numbers the modes in another order
    for k, m in enumerate(td.modes, 1):
        m.number = k
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim(), options=C.CorrelationOptions(method=method))
    assert r.verdict == C.PASS and sorted((p.test_freq, p.fe_freq) for p in r.modal.pairs) == [(f, f) for f in FREQS]
    bad = C.correlate(td, ms, mesh, SENSORS, limits=_lim(), options=C.CorrelationOptions(method="order"))      # pairing by the order is wrong -> the MAC says so
    assert bad.verdict == C.FAIL and min(p.mac for p in bad.modal.pairs) < 0.5


def test_noisy_sensors_lower_the_mac_but_keep_the_pairing(fe):
    mesh, ms = fe
    rng = np.random.default_rng(7)
    td = _test_data(fe)
    for m in td.modes:
        m.shape = {k: v * (1.0 + 0.05 * rng.standard_normal()) for k, v in m.shape.items()}          # 5 % noise of every sensor
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim())
    macs = [p.mac for p in r.modal.pairs]
    assert all(p.test_number == p.fe_number for p in r.modal.pairs) and 0.95 < min(macs) < 1.0
    assert r.verdict == C.PASS
    assert C.correlate(td, ms, mesh, SENSORS, limits=_lim({**EXAMPLE, "corr_mac_min": 0.9999})).verdict == C.FAIL


def test_missing_dofs_drop_out_of_the_mac(fe):
    mesh, ms = fe
    td = _test_data(fe)
    rng = np.random.default_rng(3)
    for m in td.modes:                                           # every mode lacks 30 % of its DOFs, an other third of the sensors is dead in mode 1
        for k in list(m.shape):
            if rng.random() < 0.3:
                del m.shape[k]
    for k in [k for k in td.modes[0].shape if k[0] in ("100", "101")]:
        del td.modes[0].shape[k]
    no_cross = _lim({**EXAMPLE, "corr_mac_offdiag_max": 1.0, "corr_comac_min": 0.0})              # these two judged separately below
    r = C.correlate(td, ms, mesh, SENSORS, limits=no_cross)
    assert all(abs(p.mac - 1.0) < 1e-12 and p.test_number == p.fe_number for p in r.modal.pairs)       # the same shape on fewer DOFs
    # a test mode with a DOF that was not measured has no COC / AOC (the reduced mass is defined for all the sensor DOFs together): those rows are UNCONFIRMED, the rest passes
    assert r.verdict == C.VERDICT_UNCONFIRMED and {x.metric for x in r.rows if x.status == C.UNCONFIRMED} <= {"COC diagonal", "COC off-diagonal", "AOC off-diagonal"}
    assert not any(x.status == C.FAIL for x in r.rows) and any("measured DOF" in x.note for x in r.rows if x.status == C.UNCONFIRMED)
    assert min(p.dofs for p in r.modal.pairs) < 24 and r.modal.pairs[0].dofs == 12
    amb = C.correlate(td, ms, mesh, SENSORS, limits=_lim())                                   # mode 1 lost two sensors: the rest cannot tell it from mode 3
    assert amb.verdict == C.FAIL and any(x.metric == "MAC off-diagonal" and x.status == C.FAIL and x.value > 0.5 for x in amb.rows)
    few = C.sensors_from_dicts([{"node": "100", "point": SENSORS[0].point, "dofs": "y"}, {"node": "101", "point": SENSORS[1].point, "dofs": "z"}])
    r = C.correlate(td, ms, mesh, few, limits=_lim())
    assert any("cannot tell the modes apart" in w for w in r.warnings)                       # two DOFs for six modes: warned


def test_a_test_mode_without_an_fe_partner_fails_even_without_limits(fe):
    mesh, ms = fe
    td = _test_data(fe)
    extra = copy.deepcopy(td.modes[0])
    extra.number, extra.frequency_hz = 7, 400.0
    td.modes.append(extra)
    r = C.correlate(td, ms, mesh, SENSORS, limits=None, options=C.CorrelationOptions(fe_modes=[1, 2, 3, 4, 5, 6]))
    assert r.modal.unpaired_test and r.verdict == C.FAIL
    only = C.correlate(_test_data(fe), ms, mesh, SENSORS, options=C.CorrelationOptions(pair_min_mac=0.999, f_max_hz=100.0))
    assert only.verdict == C.VERDICT_UNCONFIRMED                                              # a clean, filtered dataset: nothing to fail, nothing confirmed


def test_unpaired_fe_modes_are_listed_and_cross_terms_are_small(fe):
    mesh, ms = fe
    td = _test_data(fe)
    td.modes = td.modes[:4]
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim())
    assert r.modal.unpaired_fe == [5, 6] and len(r.modal.pairs) == 4
    # FE mode 5 moves along x, an axis that no sensor measures here: the sensors cannot see it, so the SEREP mass of the six FE modes is refused and the orthogonality rows are UNCONFIRMED
    assert r.verdict == C.VERDICT_UNCONFIRMED and r.modal.orth.problems and "condition" in r.modal.orth.problems[0]
    assert {x.metric for x in r.rows if x.status == C.UNCONFIRMED} <= {"COC diagonal", "COC off-diagonal", "AOC off-diagonal"}
    ok = C.correlate(td, ms, mesh, SENSORS, limits=_lim(), options=C.CorrelationOptions(fe_modes=[1, 2, 3, 4]))
    assert ok.modal.unpaired_fe == [] and ok.verdict == C.PASS
    assert all(p.offdiag is not None and p.offdiag < 0.3 for p in r.modal.pairs)
    assert r.modal.auto_mac.shape == (4, 4) and np.allclose(np.diag(r.modal.auto_mac), 1.0)


# ------------------------------------------------------------------ limits: only from the confirmed requirement set
def test_without_a_confirmed_requirement_set_nothing_is_ever_pass(fe):
    mesh, ms = fe
    td = _test_data(fe)
    assert set(C.LIMIT_INFO) == {"corr_mac_min", "corr_mac_offdiag_max", "corr_comac_min", "corr_freq_error_max", "corr_peak_freq_shift_max",
                                 "corr_peak_amp_error_max_db", "corr_rms_error_max", "corr_coc_diag_min", "corr_coc_offdiag_max", "corr_aoc_offdiag_max", "corr_modulus_diff_max"}
    assert all(not e.usable and e.value is None for e in C.CorrelationLimits().entries.values())          # the empty template: no value anywhere
    for lim in (None, C.CorrelationLimits.from_requirement_set(None),
                *(C.CorrelationLimits.from_requirement_set(s.requirement_set, s) for s in (_limits_set(EXAMPLE, confirmed=False), _limits_set(EXAMPLE, provenance=False)))):
        r = C.correlate(td, ms, mesh, SENSORS, limits=lim)
        assert r.verdict == C.VERDICT_UNCONFIRMED and all(x.status == C.UNCONFIRMED for x in r.rows) and r.modal.pairs[0].mac == pytest.approx(1.0)


def test_an_edit_after_the_confirmation_or_a_changed_scope_withdraws_the_limits():
    sp = _limits_set(EXAMPLE)
    assert C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp).value("corr_mac_min") == 0.9
    sp.requirement_set.limits[0].value = 0.1                                                  # edited after the confirmation
    assert C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp).value("corr_mac_min") is None
    sp = _limits_set(EXAMPLE)
    sp.default_class = "composite"                                                           # the structure classes were part of the confirmation
    assert C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp).value("corr_mac_min") is None
    st = Study(name="x", geometry="a.step")
    st.space = _limits_set({"corr_mac_min": 0.8, "corr_freq_error_max": -1.0})
    lim = C.limits_of_study(st)
    assert lim.value("corr_mac_min") == 0.8 and lim.value("corr_freq_error_max") is None and "negative" in lim.entries["corr_freq_error_max"].reason
    assert C.limits_of_study(Study(name="y", geometry="a.step")).set_confirmed is False


def test_limit_rows_are_added_empty_and_withdraw_a_confirmation():
    sp = _limits_set({})
    assert sp.requirement_set.is_confirmed()
    added = C.install_limit_rows(sp.requirement_set)
    assert set(added) == set(C.LIMIT_INFO) and not sp.requirement_set.is_confirmed()
    assert all(x.value is None and x.clause == "<clause>" for x in sp.requirement_set.limits if x.key in C.LIMIT_INFO)
    assert C.install_limit_rows(sp.requirement_set) == []


def test_only_the_metrics_that_were_computed_need_a_limit(fe):
    mesh, ms = fe
    r = C.correlate(_test_data(fe), ms, mesh, SENSORS, limits=_lim({"corr_mac_min": 0.9, "corr_freq_error_max": 0.05, "corr_comac_min": 0.9, "corr_mac_offdiag_max": 0.3}))
    assert r.verdict == C.PASS and {x.metric for x in r.rows} == {"MAC", "frequency error", "MAC off-diagonal", "CoMAC", "COC diagonal", "COC off-diagonal", "AOC off-diagonal", "modulus difference"}
    r = C.correlate(_test_data(fe), ms, mesh, SENSORS, limits=_lim({"corr_mac_min": 0.9}))
    assert r.verdict == C.VERDICT_UNCONFIRMED and {x.status for x in r.rows} == {C.PASS, C.UNCONFIRMED}


# ------------------------------------------------------------------ damping transfer, sensors, adapters
def test_measured_damping_goes_into_the_modal_damping_table(fe):
    mesh, ms = fe
    td = _test_data(fe)
    for m, z in zip(td.modes, (0.01, 0.02, None, 0.03, 1.5, 0.05)):             # a missing and an impossible ratio are skipped
        m.damping = z
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim())
    st = Study(name="d", geometry="a.step")
    table = C.transfer_damping(st, r.modal.pairs)
    assert table == [[10.0, 0.01], [12.5, 0.02], [75.0, 0.03], [150.0, 0.05]]
    assert st.dynamics.damping.kind == "modal" and st.dynamics.damping.table == table
    from fea_core.dynamics.modal import zetas
    assert zetas(st.dynamics.damping, np.array([10.0, 75.0])).tolist() == [0.01, 0.03]
    assert C.damping_table(r.modal.pairs, "test") == table
    for m in td.modes:
        m.damping = None
    with pytest.raises(ValueError):
        C.transfer_damping(st, C.correlate(td, ms, mesh, SENSORS).modal.pairs)


def test_sensor_definitions_are_validated_and_a_far_sensor_is_warned(fe):
    mesh, ms = fe
    st = Study(name="s", geometry="a.step")
    st.dynamics.probes = [SimpleNamespace(name="500", point=[10.0, 5.0, 5.0], selection=None)]    # type: ignore[list-item]  # test double of Probe
    assert C.sensors_from_dicts([{"node": "500", "probe": "500", "dofs": "x,z"}], st)[0].dofs == ("x", "z")
    for bad in ([{"node": "1"}], [{"node": "1", "point": [1, 2]}], [{"node": "1", "point": [1, 2, float("nan")]}], [{"node": "1", "point": [1, 2, 3]}] * 2,
                [{"node": "2", "probe": "nope"}], [{"node": "1", "point": [1, 2, 3], "dofs": "rx"}], ["x"]):
        with pytest.raises(ValueError):
            C.sensors_from_dicts(bad, st)
    far = C.sensors_from_dicts([{"node": "9", "point": [50.0, 5.0, 60.0]}])
    _, warn = C.fe_shapes_at_sensors(ms, mesh, far, [("9", "x")])
    assert warn and "outside the mesh" in warn[0]
    td = _test_data(fe)
    assert C.auto_sensors(td, st) == [] and [s.node for s in C.auto_sensors(C.TestData(modes=[C.TestMode(1, 1.0, None, {("500", "x"): 1.0})]), st)] == ["500"]
    r = C.correlate(td, ms, mesh, SENSORS[:6])                                        # two test nodes have no position
    assert any("without a sensor position" in w for w in r.warnings)


def test_predicted_curves_from_the_dynamics_results():
    f = np.linspace(5, 100, 50)
    u = (1 + 0j) * np.ones((50, 3))
    dyn = SimpleNamespace(harmonic={"f": f, "probes": {"p": {"U": u}}}, random={"f": f, "probes": {"p": {"A_psd_g2": np.ones((50, 3))}}})
    assert C.predicted_from_dynamics(dyn, "p", "y", "harmonic_displacement").values[0] == 1
    a = C.predicted_from_dynamics(dyn, "p", "z", "harmonic_acceleration")
    assert a.values[0] == pytest.approx(-((2 * np.pi * 5) ** 2) / 9810.0) and a.kind == "frf"
    assert C.predicted_from_dynamics(dyn, "p", "x", "random_acceleration_psd").kind == "psd"
    for q, p in (("harmonic_displacement", "other"), ("what", "p")):
        with pytest.raises(ValueError):
            C.predicted_from_dynamics(dyn, p, "x", q)


def test_the_whole_correlation_with_responses(fe):
    mesh, ms = fe
    f, h = _frf_pair(fe)
    td = _test_data(fe)
    td.functions = [C.TestFunction("100:y", "100", "y", f, h, "frf", "g/N")]
    pred = {"100:y": C.Predicted(f, h * 1.1, "frf", "", "acceleration", "", "force")}                  # the model curve says what it is: acceleration per force, no base motion
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim(), predicted=pred)
    assert r.verdict == C.PASS and r.responses[0].peak_error_db == pytest.approx(20 * math.log10(1.1)) and r.responses[0].peak_shift == 0
    assert {x.metric for x in r.rows if x.item.startswith("response")} == {"peak frequency shift", "peak amplitude error", "RMS error"}
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim({**EXAMPLE, "corr_peak_amp_error_max_db": 0.5}), predicted=pred)
    assert r.verdict == C.FAIL
    d = r.to_dict()
    assert d["verdict"] == C.FAIL and d["modal"]["pairs"][0]["mac"] == pytest.approx(1.0) and d["limits"]["corr_mac_min"]["usable"] and "No limit of a standard" in d["note"]
    import json
    json.dumps(d)                                                                           # serialisable (stored in the study)
    assert "verdict" in r.text()


def test_options_are_validated():
    for bad in ({"method": "x"}, {"pair_min_mac": 2}, {"damping_at": "x"}, {"response_band_hz": [5, 1]}, {"f_min_hz": 5, "f_max_hz": 1}):
        with pytest.raises(ValueError):
            C.CorrelationOptions.from_dict(bad)
    o = C.CorrelationOptions.from_dict({"method": "max_mac", "fe_modes": [1, 2], "response_band_hz": [1, 5]})
    assert C.CorrelationOptions.from_dict(o.to_dict()) == o


# ------------------------------------------------------------------ gate 3: parser security
def _valid_uff(fe) -> str:
    f, h = _frf_pair(fe)
    fn = C.TestFunction("100:y", "100", "y", f[:60], h[:60], "frf", "g")
    return C.write_uff55(_test_data(fe).modes[:2]) + C.write_uff58([fn])


def test_uff_truncated_at_every_line_and_byte_is_rejected_cleanly(fe):
    text = _valid_uff(fe)
    assert len(C.parse_uff(text).modes) == 2
    lines = text.splitlines(keepends=True)
    for n in range(1, len(lines)):                               # cut after n lines
        part = "".join(lines[:n])
        try:
            C.parse_uff(part)
        except io.TestDataError:
            pass                                                  # a cut at a dataset boundary may leave a valid shorter file
    for n in range(0, len(text), 97):                            # cut anywhere inside a line
        try:
            C.parse_uff(text[:n])
        except io.TestDataError:
            pass
    with pytest.raises(io.TestDataError):
        C.parse_uff(text[:len(text) // 2])                        # the middle of a dataset: truncated


def test_fuzzed_uff_and_csv_never_raise_anything_but_testdataerror(fe):
    rng = np.random.default_rng(11)
    sources = [_valid_uff(fe), C.write_modes_csv(_test_data(fe).modes[:2]), C.write_functions_csv([C.TestFunction("a", "1", "x", np.arange(1, 30.0), np.arange(1, 30.0))])]
    for text in sources:
        parse = C.parse_uff if text.startswith("    -1") else C.parse_csv
        raw = bytearray(text.encode())
        for _ in range(250):
            b = bytearray(raw)
            for _ in range(int(rng.integers(1, 6))):
                b[int(rng.integers(0, len(b)))] = int(rng.integers(32, 127)) if rng.random() < 0.8 else int(rng.choice([0, 10, 13, 69, 45, 43, 46]))
            try:
                data = C.decode_bytes(bytes(b))
                parse(data)
            except io.TestDataError:
                pass


def test_uff_malformed_inputs(fe):
    good = _valid_uff(fe).splitlines()
    i58 = next(i for i, ln in enumerate(good) if ln.strip() == "58")
    cases = {
        "no dataset": "hello\nworld\n",
        "not closed": "    -1\n    58\n" + "\n".join(good[i58 + 1:i58 + 8]),
        "binary marker": "    -1\n    58b\nxx\n    -1\n",
        "bad count": "\n".join(good[:i58 + 12] + good[i58 + 12:i58 + 14] + ["    -1"]),
        "text in data": "\n".join(good[:i58 + 12] + ["abc def"] * 30 + ["    -1"]),
        "nan in data": "\n".join(good[:i58 + 12] + ["nan"] * 200 + ["    -1"]),
    }
    for name, text in cases.items():
        with pytest.raises(io.TestDataError):
            C.parse_uff(text)
    r7 = i58 + 7                                                                              # the record with the ordinate type, the number of points and the spacing
    declared = f"{2:10d}{10 ** 9:10d}{0:10d}" + "".join(f"{0.0:13.5E}" for _ in range(3))
    with pytest.raises(io.TestDataError, match="outside 1 to"):                              # a declared size of a billion points is refused before anything is allocated
        C.parse_uff("\n".join(good[:r7] + [declared] + good[r7 + 1:]))
    three = f"{2:10d}{3:10d}{0:10d}" + "".join(f"{0.0:13.5E}" for _ in range(3))
    decreasing = "".join(f"{v:13.5E}" for v in (5.0, 1.0, 4.0, 1.0, 6.0, 2.0))
    with pytest.raises(io.TestDataError, match="strictly increasing"):
        C.parse_uff("\n".join(good[:r7] + [three] + good[r7 + 1:r7 + 5] + [decreasing, "    -1"]))
    ok = "".join(f"{v:13.5E}" for v in (4.0, 1.0, 5.0, 1.0, 6.0, 2.0))
    assert C.parse_uff("\n".join(good[:r7] + [three] + good[r7 + 1:r7 + 5] + [ok, "    -1"])).functions[0].freq.tolist() == [4.0, 5.0, 6.0]


def test_files_are_bounded(tmp_path, monkeypatch, fe):
    p = tmp_path / "a.csv"
    p.write_text(C.write_modes_csv(_test_data(fe).modes[:1]))
    assert C.load_test_file(p).modes
    with pytest.raises(io.TestDataError, match="too large"):
        C.load_test_file(p, max_bytes=100)
    for attr, value, text in (("MAX_LINES", 5, "a,b\n" * 20), ("MAX_LINE_CHARS", 50, "mode,frequency_hz\n" + "1," * 40), ("MAX_MODES", 2, C.write_modes_csv(_test_data(fe).modes[:3])),
                              ("MAX_NODES_PER_MODE", 3, C.write_modes_csv(_test_data(fe).modes[:1])), ("MAX_TOTAL_VALUES", 10, C.write_modes_csv(_test_data(fe).modes[:2])),
                              ("MAX_FUNCTION_POINTS", 5, C.write_functions_csv([C.TestFunction("a", "1", "x", np.arange(1, 30.0), np.arange(1, 30.0))])),
                              ("MAX_FUNCTIONS", 1, "function,node,dof,frequency_hz,value\na,1,x,1,1\nb,1,x,1,1\n")):
        with monkeypatch.context() as mp:
            mp.setattr(io, attr, value)
            with pytest.raises(io.TestDataError):
                C.parse_csv(text)
    with monkeypatch.context() as mp:
        mp.setattr(io, "MAX_FUNCTION_POINTS", 10)
        with pytest.raises(io.TestDataError):
            C.parse_uff(_valid_uff(fe))
    with monkeypatch.context() as mp:
        mp.setattr(io, "MAX_DATASETS", 1)
        with pytest.raises(io.TestDataError, match="too many datasets"):
            C.parse_uff(_valid_uff(fe))
    with pytest.raises(io.TestDataError, match="binary"):
        C.decode_bytes(b"mode,frequency_hz\n1,2\x00\n")
    big = tmp_path / "big.csv"
    big.write_bytes(b"x" * ((32 << 20) + 1))
    with pytest.raises(io.TestDataError, match="too large"):
        C.load_test_file(big)


def test_paths_are_checked(tmp_path):
    for bad in ("\\\\server\\share\\a.csv", "//server/share/a.csv", "\\\\?\\UNC\\server\\share\\a.csv"):
        with pytest.raises(io.TestDataError, match="UNC"):
            C.load_test_file(bad)
    with pytest.raises(io.TestDataError, match="does not exist"):
        C.load_test_file(tmp_path / "nope.csv")
    with pytest.raises(io.TestDataError, match="does not exist"):
        C.load_test_file(tmp_path)                                                           # a folder is not a file
    with pytest.raises(io.TestDataError):
        C.load_test_file(tmp_path / "a.csv", "zip")


def test_csv_malformed_inputs():
    bad = {
        "empty": "",
        "no frequency": "mode,node,dof,value\n1,1,x,1\n",
        "unknown layout": "a,b,c\n1,2,3\n",
        "duplicate column": "mode,frequency_hz,frequency\n1,2,3\n",
        "nan": "mode,frequency_hz\n1,nan\n",
        "text freq": "mode,frequency_hz\n1,abc\n",
        "inf": "mode,frequency_hz\n1,inf\n",
        "negative f": "mode,frequency_hz\n1,-4\n",
        "two frequencies": "mode,frequency_hz,node,dof,value\n1,5,1,x,1\n1,6,1,y,1\n",
        "bad dof": "mode,frequency_hz,node,dof,value\n1,5,1,q,1\n",
        "duplicate dof": "mode,frequency_hz,node,dof,value\n1,5,1,x,1\n1,5,1,x,2\n",
        "extra cells": "mode,frequency_hz\n1,5,7,8\n",
        "bad damping": "mode,frequency_hz,damping\n1,5,3\n",
        "decreasing f": "function,node,dof,frequency_hz,value\na,1,x,5,1\na,1,x,4,1\n",
        "bad kind": "function,node,dof,frequency_hz,value,kind\na,1,x,5,1,zzz\n",
        "huge field": "mode,frequency_hz,node,dof,value\n1,5," + "x" * 3000 + ",x,1\n" + "\"" + "y" * 3000 + "\n" + "y" * 3000 + "\n",
    }
    for name, text in bad.items():
        with pytest.raises(io.TestDataError):
            C.parse_csv(text)
    with pytest.raises(io.TestDataError):
        C.parse_csv("mode;frequency_hz;node;dof;value\n1;5,0;1;-z;2\n")                       # a decimal comma is not a number
    assert C.parse_csv("mode;frequency_hz;node;dof;value\n1;5.0;1;-z;2\n").modes[0].shape == {("1", "z"): -2.0}


def test_csv_variants_and_label_cleaning():
    td = C.parse_csv("# comment\nMode,Frequency,Zeta,Node,Direction,Value\n1,5.5,0.02,A=1,-z,2\n1,,,A=1,x,\n2,9,,B,+Y,3\n")
    m1, m2 = td.modes
    assert m1.frequency_hz == 5.5 and m1.damping == 0.02 and list(m1.shape) == [("A_1", "z")] and m1.shape[("A_1", "z")] == -2.0 and m2.shape == {("B", "y"): 3.0}
    td = C.parse_csv("function,node,dof,frequency_hz,amplitude,phase_deg,kind\nf,1,x,1,2,90,psd\nf,1,x,2,3,0,psd\n")
    fn = td.functions[0]
    assert fn.kind == "psd" and fn.values[0] == pytest.approx(2j) and fn.values[1] == 3
    assert C.clean_label("=cmd|' /C calc'!A0\x00\n" + "z" * 100).startswith("_cmd") and len(C.clean_label("z" * 100)) == io.MAX_LABEL
    assert "=" not in C.clean_label("=1+1") and C.parse_dof("-3") == ("z", -1.0) and C.parse_dof("uy") == ("y", 1.0) and C.parse_dof("z-") == ("z", -1.0)
    with pytest.raises(io.TestDataError):
        C.parse_dof("")


def test_uff58_directions_even_spacing_and_double_precision():
    f = np.arange(1.0, 6.0)
    rec = [
        "    -1", "    58", "title", "NONE", "func", "NONE", "NONE",
        f"{4:5d}{1:10d}{0:5d}{0:10d} {'NONE':<10s}{12:10d}{-3:4d} {'NONE':<10s}{1:10d}{3:4d}",
        f"{4:10d}{5:10d}{1:10d}{1.0:13.5E}{1.0:13.5E}{0.0:13.5E}",
        f"{18:10d}{0:5d}{0:5d}{0:5d} {'Frequency':<20s} {'Hz':<20s}", f"{12:10d}{0:5d}{0:5d}{0:5d} {'Acc':<20s} {'g':<20s}",
        f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20s} {'NONE':<20s}", f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20s} {'NONE':<20s}",
        "".join(f"{v:20.12E}" for v in f[:4]), f"{f[4]:20.12E}", "    -1"]
    fn = C.parse_uff("\n".join(rec)).functions[0]
    assert fn.node == "12" and fn.dof == "z" and fn.reference_node == "1" and fn.domain == "frequency" and fn.unit == "g"
    assert fn.freq.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0] and fn.values.tolist() == [-1.0, -2.0, -3.0, -4.0, -5.0]            # direction -Z: the sign is applied
    assert C.TestData(functions=[fn]).summary()["functions"][0]["points"] == 5


def test_uff55_unsupported_analysis_is_skipped_with_a_warning(fe):
    text = _valid_uff(fe).replace(f"{1:10d}{2:10d}{2:10d}{8:10d}", f"{1:10d}{1:10d}{2:10d}{8:10d}", 1)
    td = C.parse_uff(text)
    assert len(td.modes) == 1 and any("skipped" in w for w in td.warnings)
    other = "    -1\n   164\n         1SI units\n    -1\n" + _valid_uff(fe)
    td = C.parse_uff(other)
    assert td.skipped_datasets == 1 and len(td.modes) == 2


def test_nothing_is_pickled_or_executed():
    import ast
    import inspect
    for mod in (io, C):
        tree = ast.parse(inspect.getsource(mod))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
        assert not imported & {"pickle", "marshal", "subprocess", "os", "shelve", "yaml", "ctypes"}, imported
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not calls & {"eval", "exec", "compile", "__import__", "open"}, calls              # the only file access is Path.open in `_read_text`
        assert "allow_pickle" not in inspect.getsource(mod).replace("allow_pickle=False", "")


# ------------------------------------------------------------------ the shared runner and the MCP tools
def _results(fe):
    from fea_core.results import Results
    mesh, ms = fe
    return Results(mesh=mesh, steps=[], modes=[ms])


def _study_with_limits(values: dict[str, float], confirmed: bool = True) -> Study:
    from verification.space_cases import block_study
    st = block_study()
    st.space = _limits_set(values, confirmed=confirmed)
    return st


def test_runner_stores_the_result_in_the_study_and_it_survives_a_save(fe, tmp_path):
    from fea_core.space import corr_run as CR
    st = _study_with_limits(EXAMPLE)
    td = _test_data(fe)
    td.modes[0].damping = 0.012
    res, table = CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), apply_damping=True)
    assert res.verdict == C.PASS and table is not None and table[0] == [10.0, 0.012] and st.dynamics.damping.table == table
    rec = CR.state_of(st)
    assert rec["source"] == "FE model" and rec["result"]["verdict"] == C.PASS and len(rec["sensors"]) == 8 and rec["damping_transferred"] == table
    from fea_core.study import Study as S
    p = tmp_path / "s.fea"
    st.save(p)
    back = S.load(p)
    assert CR.state_of(back)["result"]["modal"]["pairs"][0]["mac"] == pytest.approx(1.0) and len(CR.load_sensors(back)) == 8
    assert back.space.requirement_confirmed()                                                 # storing a result does not touch the confirmation
    with pytest.raises(ValueError, match="no natural modes"):
        CR.run_correlation(st, SimpleNamespace(modes=[], mesh=None), td, sensors=list(SENSORS))
    with pytest.raises(ValueError, match="mode_set"):
        CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), mode_set=3)
    CR.clear_state(st)
    assert CR.state_of(st) == {}


def test_runner_predicts_the_responses_from_the_dynamics_results(fe):
    from fea_core.space import corr_run as CR
    st = _study_with_limits(EXAMPLE)
    f = np.linspace(5.0, 200.0, 300)
    u = np.zeros((300, 3), complex)
    u[:, 1] = C.modal_frf(f, np.array(FREQS), 0.02, np.ones(6), np.ones(6), "receptance")
    res = _results(fe)
    res.dynamics = SimpleNamespace(harmonic={"f": f, "probes": {"p100": {"U": u}}}, random=None)
    sens = C.sensors_from_dicts([{"node": "100", "point": [10.0, 5.0, 5.0], "dofs": "y"}])
    sens[0].probe = "p100"
    td = C.TestData(functions=[C.TestFunction("100:y", "100", "y", f, -((2 * np.pi * f) ** 2) * u[:, 1] / 9810.0, "frf", "g")])
    r, _ = CR.run_correlation(st, res, td, sensors=sens, setups={"100:y": C.FunctionSetup(frame="relative")})
    assert r.responses[0].rms_error < 1e-12 and r.responses[0].peak_shift == 0
    r, _ = CR.run_correlation(st, SimpleNamespace(modes=[], mesh=None, dynamics=None), td, sensors=sens)
    assert not r.responses and any("no dynamics response" in w for w in r.warnings)


def _mcp(sess, steps):
    import json

    import anyio
    out = []

    async def main():
        from mcp import Client

        from fea_core.mcp.server import build_server
        srv = build_server(sess, groups_spec="space")
        async with Client(srv, mode="legacy") as c:
            for name, args, expect_err in steps:
                r = await c.call_tool(name, args)
                text = r.content[0].text
                assert bool(r.is_error) == expect_err, (name, text)
                out.append(text if expect_err else json.loads(text))
    anyio.run(main)
    return out


def test_mcp_import_and_correlate_with_limits_only_from_the_confirmed_set(fe, tmp_path):
    pytest.importorskip("mcp")
    from fea_core.mcp.session import Session
    from verification.space_cases import block_study
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    sess.results = _results(fe)
    f = tmp_path / "modes.csv"
    f.write_text(C.write_modes_csv(_test_data(fe).modes))
    sensors = [s.to_dict() for s in SENSORS]
    ap = "protoflight"
    base = make_requirement_set(ap, confirmed=False)
    data = {"name": "t", "verification_approach": ap, "factors": [x.to_dict() for x in base.factors if x.value is not None],
            "rules": [r.to_dict() for r in base.rules if r.factors],
            "limits": [{"key": k, "value": v, "unit": "", "document": "doc", "revision": "rev", "clause": "c1"} for k, v in EXAMPLE.items()]}
    o = _mcp(sess, [
        ("fea_space_correlate", {}, True),                                                              # 0: no test data yet
        ("fea_space_test_import", {"path": str(f)}, False),                                              # 1: nodes unmapped
        ("fea_space_test_import", {"sensors": sensors}, False),                                          # 2
        ("fea_space_correlate", {}, False),                                                              # 3: no requirement set: UNCONFIRMED
        ("fea_space_requirement_set", {"action": "template"}, False),                                    # 4
        ("fea_space_requirement_set", {"action": "set", "data": data}, False),                          # 5
        ("fea_space_correlate", {}, False),                                                              # 6: the set is not confirmed
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),   # 7
        ("fea_space_correlate", {"matrices": True}, False),                                              # 8: PASS
        ("fea_space_correlate", {"method": "nonsense"}, True),                                           # 9
        ("fea_space_test_import", {"path": str(tmp_path / "nope.csv")}, True),                           # 10
        ("fea_space_correlate", {"limit_rows": True}, False),                                            # 11: the rows exist already: nothing is withdrawn
        ("fea_space_test_import", {"clear": True}, False),                                               # 12
        ("fea_space_correlate", {}, True),                                                               # 13
    ])
    assert "No test data" in o[0]
    assert o[1]["dataset"]["modes"][0]["frequency_Hz"] == 10.0 and len(o[1]["unmapped_nodes"]) == 8 and o[1]["sensors"] == []
    assert o[2]["unmapped_nodes"] == [] and len(o[2]["sensors"]) == 8
    assert o[3]["verdict"] == "UNCONFIRMED_REQUIREMENTS" and o[3]["modal"]["pairs"][0]["mac"] == pytest.approx(1.0) and not o[3]["requirement_set_confirmed"]
    assert o[6]["verdict"] == "UNCONFIRMED_REQUIREMENTS" and all(not v["usable"] for v in o[6]["limits"].values())
    assert o[8]["verdict"] == "PASS" and o[8]["requirement_set_confirmed"] and o[8]["confirmed_by"] == "Test Engineer" and len(o[8]["modal"]["mac"]) == 6
    assert o[8]["limits"]["corr_mac_min"]["usable"] and "text" in o[8]
    assert "method" in o[9]
    assert "does not exist" in o[10]
    assert o[11]["verdict"] == "PASS"
    assert sess.study.space.extra["correlation"]["result"]["verdict"] == "PASS"
    assert "No test data" in o[13]


def test_limit_rows_are_added_by_the_tool_and_withdraw_a_confirmation(fe, tmp_path):
    pytest.importorskip("mcp")
    from fea_core.mcp.session import Session
    from verification.space_cases import block_study
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    sess.study.space = _limits_set({})
    sess.results = _results(fe)
    f = tmp_path / "modes.csv"
    f.write_text(C.write_modes_csv(_test_data(fe).modes))
    o = _mcp(sess, [("fea_space_test_import", {"path": str(f), "sensors": [s.to_dict() for s in SENSORS]}, False),
                    ("fea_space_correlate", {"limit_rows": True}, False)])
    assert o[1]["verdict"] == "UNCONFIRMED_REQUIREMENTS" and not o[1]["requirement_set_confirmed"] and "withdrawn" in o[1]["limit_rows"]
    assert {x.key for x in sess.study.space.requirement_set.limits} == set(C.LIMIT_INFO) and not sess.study.space.requirement_set.is_confirmed()


def test_mcp_tools_have_no_argument_that_enters_or_confirms_a_limit_and_the_group_is_lazy(tmp_path):
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from fea_core.mcp.server import build_server
    from fea_core.mcp.session import Session
    names: dict[str, set[str]] = {}

    async def main():
        srv = build_server(Session(tmp_path / "m"), groups_spec="core")
        async with Client(srv, mode="legacy") as c:
            assert not any(t.name.startswith("fea_space_") for t in (await c.list_tools()).tools)
            await c.call_tool("fea_tools_enable", {"groups": "space"})
            for t in (await c.list_tools()).tools:
                if t.name in ("fea_space_test_import", "fea_space_correlate"):
                    names[t.name] = set(t.input_schema.get("properties", {}))
    anyio.run(main)
    assert set(names) == {"fea_space_test_import", "fea_space_correlate"}
    for props in names.values():
        assert not props & {"limits", "limit", "confirmed_by", "confirm", "confirmed", "thresholds", "mac_min"}, props


def test_mcp_import_refuses_bad_files_cleanly(fe, tmp_path):
    pytest.importorskip("mcp")
    from fea_core.mcp.session import Session
    from verification.space_cases import block_study
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    good = tmp_path / "ok.uff"
    good.write_text(_valid_uff(fe))
    cut = tmp_path / "cut.uff"
    cut.write_text(_valid_uff(fe)[:700])
    bad = tmp_path / "bad.csv"
    bad.write_text("mode,frequency_hz\n1,abc\n")
    binary = tmp_path / "b.uff"
    binary.write_bytes(b"    -1\n    58\n\x00\x01\x02")
    o = _mcp(sess, [
        ("fea_space_test_import", {"path": str(cut)}, True), ("fea_space_test_import", {"path": str(bad)}, True), ("fea_space_test_import", {"path": str(binary)}, True),
        ("fea_space_test_import", {"path": "\\\\server\\share\\a.uff"}, True), ("fea_space_test_import", {"path": str(good), "format": "zip"}, True),
        ("fea_space_test_import", {"path": str(good)}, False),
        ("fea_space_test_import", {"path": str(good), "append": True}, False),
        ("fea_space_test_import", {"sensors": [{"node": "1"}]}, True),
    ])
    assert "truncated" in o[0] or "malformed" in o[0] or "closed" in o[0]
    assert "not a number" in o[1] and "binary" in o[2] and "UNC" in o[3]
    assert len(o[5]["dataset"]["modes"]) == 2 and len(o[6]["dataset"]["modes"]) == 4 and o[5]["dataset"]["functions"][0]["node"] == "100"
    assert "point" in o[7]


# ------------------------------------------------------------------ report subsection
def _doc_text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def test_report_subsection_prints_the_stored_result_with_the_provenance_of_the_limits(fe):
    pytest.importorskip("docx")
    from fea_core.report import space_corr as RC
    from fea_core.report.word import _Doc
    from fea_core.space import corr_run as CR
    st = _study_with_limits(EXAMPLE)
    assert not RC.has_correlation(st)
    td = _test_data(fe)
    td.modes[1].frequency_hz *= 1.2                                                       # a 17 % frequency error: a FAIL row
    f, h = _frf_pair(fe)
    td.functions = [C.TestFunction("100:y", "100", "y", f, h, "frf", "g/N")]
    CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), predicted={"100:y": C.Predicted(f, h * 1.1, "frf", "", "acceleration", "", "force")})
    assert RC.has_correlation(st)
    D = _Doc()
    warns = RC.correlation_section(D, st, "9", _results(fe))
    text = _doc_text(D)
    assert [w for w in warns if "mass-orthogonal" not in w and "SEREP basis" not in w] == [] and "Verdict of the test correlation: UNVERIFIABLE" in text and "The result stored in the study says FAIL" in text      # `predicted` was given: not recomputable
    for needle in ("MAC matrix", "Paired modes: test against FE model", "Response curves: predicted against measured", "Correlation limits (entered and confirmed by the user)",
                   "corr_mac_min", "doc", "rev", "c1", "frequency error", "FAIL", "test engineer", "protoflight", "no correlation limit of a standard"):
        assert needle in text, needle
    # the chapter of the space report calls it at its end
    from fea_core.report import space as RS
    from verification import space_cases as SC
    mesh2 = SC.box_tet_mesh()
    st.materials.update(SC.block_study().materials)
    SC.confirm_test_allowables(st)
    sess = SimpleNamespace(study=st, results=SC.synthetic_results(mesh2, [SC.uniform_state(mesh2, "Load", SC.TEST_SIGMA)]))
    D3 = _Doc()
    RS.space_section(D3, sess, "7")
    t3 = _doc_text(D3)
    assert "Verdict: " in t3 and "Vibration test correlation" in t3 and "Paired modes: test against FE model" in t3
    # an unconfirmed set: the verdict says so and no row claims a pass
    st2 = _study_with_limits(EXAMPLE, confirmed=False)
    CR.run_correlation(st2, _results(fe), _test_data(fe), sensors=list(SENSORS))
    D2 = _Doc()
    RC.correlation_section(D2, st2)
    t2 = _doc_text(D2)
    assert "UNCONFIRMED_REQUIREMENTS" in t2 and "not an ECSS verification result" in t2 and "PASS" not in t2.replace("Correlation criteria", "")
    assert RC.append_correlation(_Doc(), SimpleNamespace(space=None)) == []


# ------------------------------------------------------------------ a real modal solution (solver, not analytic fields)
def _solver_available() -> bool:
    import importlib.util
    from fea_core import config
    direct = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
    return direct or config.CCX_EXE.exists()


@pytest.mark.skipif(not _solver_available(), reason="a modal solver (direct path or CalculiX) is needed")
def test_real_modes_of_a_bar_exported_as_uff_and_a_stiffer_twin_as_the_test(tmp_path):
    """The gate on a real modal solution: the modes of a clamped steel bar are exported to UFF 55 at eight sensors (the dataset 'from the FE model itself') -> MAC 1, 0 % error;
    a twin with E x 1.21 plays the test: every frequency is 10 % higher (f ~ sqrt(E)), the shapes are the same -> MAC 1 and a frequency error of exactly 1 / 1.1 - 1."""
    from fea_core.eigen import run_eigen
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.solve import prepare_mesh
    from fea_core.study import Fixed, Material, MeshSettings
    from verification.cases import RHO, _study, box
    wd = tmp_path / "bar"
    wd.mkdir()
    st, parts = _study("bar", wd / "bar.step", [Part("Bar", box(0, 0, 0, 100.0, 20.0, 10.0))], MeshSettings(size=5.0))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.constraints.append(Fixed("clamp"))
    st.modal.enabled, st.modal.modes = True, 6
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    ms = run_eigen(st, mesh, wd / "m1", lambda s: None)[0]
    assert ms.kind == "modal" and len(ms.values) == 6 and ms.values[0] > 0
    sensors = C.sensors_from_dicts([{"node": str(200 + k), "point": [15.0 + 10.0 * k, 10.0 + (k % 3) * 2.0, 5.0 + (k % 2) * 3.0]} for k in range(8)])
    self_td = C.parse_uff(C.write_uff55(C.fe_test_data(ms, mesh, sensors, damping=0.01).modes))
    r = C.correlate(self_td, ms, mesh, sensors, limits=_lim())
    assert r.verdict == C.PASS and len(r.modal.pairs) == 6, r.text()
    assert all(abs(p.mac - 1.0) < 1e-6 and abs(p.freq_error) < 1e-5 for p in r.modal.pairs)
    twin = copy.deepcopy(st)
    twin.materials["Ocel S235"] = Material("Ocel S235", 210000.0 * 1.21, 0.3, RHO, Re=235.0, Rm=360.0, alpha=1.2e-5)
    ms2 = run_eigen(twin, mesh, wd / "m2", lambda s: None)[0]
    test_td = C.parse_csv(C.write_modes_csv(C.fe_test_data(ms2, mesh, sensors).modes))
    r = C.correlate(test_td, ms, mesh, sensors, limits=_lim({**EXAMPLE, "corr_freq_error_max": 0.1}))
    assert r.verdict == C.PASS and len(r.modal.pairs) == 6, r.text()
    assert all(abs(p.mac - 1.0) < 1e-6 and p.test_number == p.fe_number for p in r.modal.pairs)
    assert all(p.freq_error == pytest.approx(1 / 1.1 - 1, abs=2e-4) for p in r.modal.pairs), [p.freq_error for p in r.modal.pairs]
    tight = C.correlate(test_td, ms, mesh, sensors, limits=_lim({**EXAMPLE, "corr_freq_error_max": 0.05}))
    assert tight.verdict == C.FAIL and {x.metric for x in tight.rows if x.status == C.FAIL} == {"frequency error"}


def test_complex_test_shapes_with_arbitrary_phase_give_the_same_mac(fe):
    mesh, ms = fe
    rng = np.random.default_rng(5)
    td = _test_data(fe)
    for m in td.modes:
        factor = np.exp(1j * rng.uniform(0, 2 * np.pi)) * (0.5 + rng.random())      # a complex mode is only defined up to a complex factor
        m.shape = {k: complex(v) * factor for k, v in m.shape.items()}
    back = C.parse_uff(C.write_uff55(td.modes))                              # complex shapes survive the UFF round trip (data type 5)
    assert all(isinstance(v, complex) for v in back.modes[0].shape.values())
    r = C.correlate(back, ms, mesh, SENSORS, limits=_lim())
    assert r.verdict == C.PASS and all(abs(p.mac - 1.0) < 1e-5 for p in r.modal.pairs), r.text()
    assert r.modal.auto_mac.size == 0 and all(v is None or v > 0.999 for v in r.modal.comac.values())


def test_mode_selection_options(fe):
    mesh, ms = fe
    td = _test_data(fe)
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim(), options=C.CorrelationOptions(f_min_hz=60.0, f_max_hz=130.0))
    assert [p.test_number for p in r.modal.pairs] == [3, 4, 5] and r.modal.fe_numbers == [3, 4, 5]
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim(), options=C.CorrelationOptions(fe_modes=[1, 3, 5]))
    assert [p.fe_number for p in r.modal.pairs if p.mac > 0.99] == [1, 3, 5] and len(r.modal.unpaired_test) == 3 and r.verdict == C.FAIL
    with pytest.raises(ValueError, match="no test mode"):
        C.correlate(td, ms, mesh, SENSORS, options=C.CorrelationOptions(f_min_hz=1000.0))
    with pytest.raises(ValueError, match="sensors"):
        C.correlate(td, ms, mesh, [])

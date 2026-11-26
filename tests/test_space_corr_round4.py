"""Round-4 review of the COC / AOC and the owner decision of 25. 11. 2026 (F14-S5): the SEREP COC / AOC are INDICATIVE only - every row is UNCONFIRMED, never PASS and never FAIL - and only an
imported TAM can judge them; the SEREP basis is the compared modes and its Gram matrix is cheap; the null space of a TAM is told from its mass by the noise floor of the file; the notes name the
modes; a Gram matrix given to `orthogonality` is never guessed from its shape."""
from __future__ import annotations

import numpy as np
import pytest

import test_space_correlation as T
from fea_core.eigen import ModeSet
from fea_core.space import corr_orth as O
from fea_core.space import correlation as C
from test_space_corr_coc_round2 import _one_tet
from test_space_corr_orthogonality import _limits
from verification.space_cases import block_study
from verification.space_corr_cases import _generalised_chain

NAMES = ("COC diagonal", "COC off-diagonal", "AOC off-diagonal")


def _modal_rows(orth, test, fe, dofs, lim=None):
    modal = C.modal_correlation(np.arange(1, test.shape[1] + 1) * 10.0, test, np.arange(1, fe.shape[1] + 1) * 10.0, fe, dofs, orth=orth)
    return C.evaluate(modal, [], lim or _limits({"corr_mac_min": 0.0, "corr_mac_offdiag_max": 1.0, "corr_comac_min": 0.0, "corr_modulus_diff_max": 1.0}))


def _csv(dofs, m, fmt="%.9e"):
    return "# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in dofs) + "\n" + "\n".join(f"{dofs[i][0]}:{dofs[i][1]}," + ",".join(fmt % float(v) for v in m[i]) for i in range(len(dofs))) + "\n"


# ------------------------------------------------------------------ 1: the rule - a SEREP row is never PASS and never FAIL
def test_the_heavy_tip_false_pass_of_round_4_is_unconfirmed():
    """8 masses, two of 1000 at the free end, the 4 upper modes (the heavy tip hardly moves), the tip 10 % off: the SEREP COC is 1.00 and the Euclidean residual 0.003 - the COC of the
    full mass matrix is 0.52. The round-4 reviewer got PASS rows for it; now the SEREP rows are UNCONFIRMED (the TAM of the full mass matrix gives the 0.52 and FAILs it)."""
    ph, mh = _generalised_chain([1.0] * 6 + [1000.0] * 2)
    dofs = [(f"N{i}", "x") for i in range(8)]
    band = sorted(int(i) for i in np.argsort(np.abs(ph[6:, :]).max(axis=0))[:4])
    fe = ph[:, band]
    tt = fe.copy()
    tt[7, 0] += 0.10 * np.abs(fe[:, 0]).max()
    g = ph.T @ mh @ ph
    serep = O.orthogonality(tt, fe, dofs, O.OrthMass("serep", gram=g[np.ix_(band, band)]))
    assert serep.residual[0] < 0.05 and serep.coc[0, 0] > 0.999
    rows = _modal_rows(serep, tt, fe, dofs)
    coc = [x for x in rows if x.metric == "COC diagonal" and x.item.startswith("test mode 1 ")][0]
    assert coc.status == C.UNCONFIRMED and coc.value > 0.999 and C.INDICATIVE_REASON in coc.note
    assert not any(x.status == C.PASS for x in rows if x.metric in NAMES) and C.verdict_of(rows) == C.VERDICT_UNCONFIRMED
    tam = O.orthogonality(tt, fe, dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, mh)))
    assert tam.coc[0, 0] == pytest.approx(0.5188, abs=1e-3)
    assert [x.status for x in _modal_rows(tam, tt, fe, dofs) if x.metric == "COC diagonal" and x.item.startswith("test mode 1 ")] == [C.FAIL]


def test_a_uniaxial_sensor_set_gets_an_informative_value_and_only_the_indicative_reason():
    """Sensors that see only part of the DOFs of a 12-mass chain (6 DOFs, 4 compared modes): the value is computed and shown; the reason of the rows is the indicative one and nothing that
    pretends to be a finding about solved modes outside the window."""
    phi, m = _generalised_chain([1.0] * 12)
    keep = [1, 3, 5, 7, 9, 11]
    fe = phi[keep][:, :4]
    dofs = [(f"S{i}", "y") for i in keep]
    g = phi.T @ m @ phi
    o = O.orthogonality(fe.copy(), fe, dofs, O.OrthMass("serep", gram=g[:4, :4]))
    assert o.computed and o.hold == "" and not o.mode_hold and o.hold_of([0, 1, 2, 3]) == C.INDICATIVE_REASON
    rows = _modal_rows(o, fe.copy(), fe, dofs)
    orth_rows = [x for x in rows if x.metric in NAMES]
    assert orth_rows and {x.status for x in orth_rows} == {C.UNCONFIRMED} and all(x.value is not None and x.note == C.INDICATIVE_REASON for x in orth_rows)
    assert "inside the frequency span" not in " ".join(x.note for x in rows)


def test_a_correlation_with_the_serep_mass_is_never_pass_whatever_the_limits_and_the_text_says_why():
    mesh, ms = T._box_mesh(), None
    ms = T._modes(mesh)
    td = C.fe_test_data(ms, mesh, T.SENSORS, damping=0.01)
    om = C.OrthMass("serep", gram=T._unit_mass_gram(ms), gram_modes=list(range(len(ms.values))))
    r = C.correlate(td, ms, mesh, T.SENSORS, limits=T._lim(), orth_mass=om)
    assert r.verdict == C.VERDICT_UNCONFIRMED and r.modal.orth.indicative == C.INDICATIVE_REASON
    assert not any(x.status == C.PASS for x in r.rows if x.metric in NAMES) and any(x.status == C.PASS for x in r.rows if x.metric == "MAC")           # MAC is judged as before
    assert "COC / AOC UNCONFIRMED: indicative (SEREP reduction from the FE modes)" in r.text()
    d = r.to_dict()
    assert d["modal"]["orthogonality"]["indicative"] == C.INDICATIVE_REASON and d["verdict"] == C.VERDICT_UNCONFIRMED
    # a value outside the limit: still not a FAIL, a warning hint in the result
    noisy = C.fe_test_data(ms, mesh, T.SENSORS, damping=0.01)
    rng = np.random.default_rng(3)
    for mode in noisy.modes:
        for k in list(mode.shape):
            mode.shape[k] = mode.shape[k] * (1.0 + 0.02 * float(rng.standard_normal()))
    tight = T._lim({**T.EXAMPLE, "corr_coc_diag_min": 0.99999999})
    r2 = C.correlate(noisy, ms, mesh, T.SENSORS, limits=tight, orth_mass=om)
    hinted = [x for x in r2.rows if x.indicative_outside]
    assert hinted and all(x.status == C.UNCONFIRMED for x in hinted) and not any(x.status == C.FAIL for x in r2.rows if x.metric in NAMES)
    assert any("indicative SEREP COC / AOC value(s) lie outside their limit" in w for w in r2.warnings)


# ------------------------------------------------------------------ 2: the basis is the compared modes, the Gram matrix is cheap
def test_the_gram_matrix_is_the_same_cached_and_block_by_block_and_exactly_symmetric(monkeypatch):
    from fea_core.dynamics.mass import mass_times
    mesh, st = _one_tet(), block_study()
    rng = np.random.default_rng(7)
    shapes = rng.standard_normal((7, 10, 3)).astype(np.float32)
    ms = ModeSet("modal", "m", [float(k + 1) for k in range(7)], shapes)
    idx = [0, 2, 3, 6]
    one = O.serep_gram(st, mesh, ms, idx)
    ref = np.array([[float(np.sum(shapes[i].astype(float) * mass_times(st, mesh, shapes[j].astype(float)))) for j in idx] for i in idx])
    assert np.allclose(one, ref, rtol=1e-12, atol=0) and np.array_equal(one, one.T)
    monkeypatch.setattr(O, "SEREP_GRAM_CACHE_BYTES", 1)                                       # nothing is cached: the modes are converted block by block, the blocks above the diagonal formed
    monkeypatch.setattr(O, "SEREP_GRAM_STEP", 3)
    many = O.serep_gram(st, mesh, ms, idx)
    assert np.allclose(many, one, rtol=1e-12, atol=0) and np.array_equal(many, many.T)
    with pytest.raises(ValueError, match="not valid"):
        O.serep_gram(st, mesh, ms, [0, 0])


def test_the_modes_are_converted_once_for_a_model_whose_compared_modes_fit_the_cache():
    assert O.gram_cached(6837, 40) and O.gram_cached(67_040, 40) and O.gram_cached(200_000, 20)       # 160 KB / 1.6 MB / 4.8 MB per mode: one conversion
    assert not O.gram_cached(1_000_000, 40) and not O.gram_cached(200_000, 30) and O.gram_cached(1_000_000, 5)
    assert O.SEREP_GRAM_STEP == 8                                                              # M Phi is formed 8 modes at a time (the RAM of `mass_times`)


def test_only_the_compared_modes_enter_the_gram_matrix(monkeypatch):
    mesh, st = _one_tet(), block_study()
    seen = []
    real = O.serep_gram
    monkeypatch.setattr(O, "serep_gram", lambda s, m, ms, idx: (seen.append(list(idx)), real(s, m, ms, idx))[1])
    rng = np.random.default_rng(1)
    ms = ModeSet("modal", "m", [float(k + 1) for k in range(30)], rng.standard_normal((30, 10, 3)).astype(np.float32))
    fe = rng.standard_normal((9, 3))
    o = O.orthogonality(fe.copy(), fe, [(f"N{i}", "x") for i in range(9)], O.OrthMass("serep", study=st, mesh=mesh), ms, [4, 11, 20], [5, 6, 7], "")
    assert seen == [[4, 11, 20]] and o.computed and o.info["basis_modes"] == 3 == o.info["fe_modes"] and "basis_extra" not in o.info          # of the 30 solved modes only the 3 compared


def test_the_obsolete_basis_machinery_is_gone():
    for name in ("serep_extra_indices", "SEREP_POOL_MAX", "SEREP_HOLD_COND"):
        assert not hasattr(O, name) and not hasattr(C, name)


# ------------------------------------------------------------------ 3: the TAM null-space band is the noise floor of the file
@pytest.mark.parametrize("ev, rank", [([-1e-6, 1e-7, 3.0, 100.0], 2), ([1.0, 2.0, 3.0], 3), ([-5e-5, 1e-5, 1e-4, 10.0], 1), ([-3e-5, 1e-3, 1.0, 1.0], 3)])
def test_the_rank_band_follows_the_most_negative_eigenvalue_between_the_clamps(ev, rank):
    band, got = O.mass_rank_band(np.array(sorted(ev)))
    top = max(ev)
    assert O.TAM_BAND_MIN * top <= band <= O.TAM_RANK_TOL * top and got == rank


def test_a_negative_eigenvalue_beyond_the_noise_band_is_no_mass():
    for ev in ([-0.01, 1.0], [-2e-4, 1.0, 1.0], [0.0, -1.0], [0.0, 0.0]):
        with pytest.raises(ValueError):
            O.mass_rank_band(np.array(sorted(ev)))
    assert O.mass_rank_band(np.array([-9e-5, 1.0, 2.0]))[1] == 2                                # 9e-5 of the largest: still the rounding of a file


def test_a_full_rank_diagonal_tam_with_a_mass_spread_of_2e4_is_read_as_full_rank():
    n = 30
    dofs = [(f"N{i}", "x") for i in range(n)]
    spread = np.diag(np.geomspace(1.0, 2e4, n))
    for fmt in ("%.9e", "%.6e", "%.5e"):
        tam = O.parse_mass_csv(_csv(dofs, spread, fmt), "spread.csv")
        assert tam.rank == n
    masses = [1.0, 2.0, 5.0, 10.0, 100.0, 1e3, 5e3, 2e4]                                        # the generalised chain with the same spread: its modes are orthogonal in diag(masses)
    phi, m = _generalised_chain(masses)
    d8 = [(f"N{i}", "x") for i in range(8)]
    tam8 = O.parse_mass_csv(_csv(d8, m, "%.6e"), "chain.csv")
    assert tam8.rank == 8
    o = O.orthogonality(phi[:, :4], phi[:, :4], d8, O.OrthMass("tam", tam=tam8))
    assert o.computed and o.info["rank"] == 8 and not o.hold and not o.mode_hold and o.residual is None and np.allclose(np.diag(o.coc), 1.0, atol=1e-5)


@pytest.mark.parametrize("fmt", ["%.7e", "%.6e", "%.5e"])
def test_a_rank_deficient_serep_matrix_is_singular_at_every_printing_precision(fmt):
    phi, m = _generalised_chain([1.0] * 12)
    fe = phi[:, :4]
    pin = np.linalg.pinv(fe)
    dofs = [(f"N{i}", "x") for i in range(12)]
    tam = O.parse_mass_csv(_csv(dofs, pin.T @ (fe.T @ m @ fe) @ pin, fmt), "serep.csv")
    assert tam.rank == 4
    o = O.orthogonality(fe, fe, dofs, O.OrthMass("tam", tam=tam))
    assert o.info["rank"] == 4 and any("singular (rank 4 of 12" in w for w in o.warnings)


def test_a_genuinely_indefinite_matrix_is_refused_by_the_reader_and_by_a_matrix_built_without_it():
    dofs = [("A", "x"), ("B", "x")]
    with pytest.raises(Exception, match="positive semi-definite"):
        O.parse_mass_csv("# unit: kg\n,A:x,B:x\nA:x,1,0\nB:x,0,-0.01\n")
    built = O.MassMatrix(dofs, np.diag([1.0, -0.5]))                                            # no reader: the rank is not known, the spectrum is read when the matrix is used
    o = O.orthogonality(np.eye(2), np.eye(2), dofs, O.OrthMass("tam", tam=built))
    assert not o.computed and "positive semi-definite" in o.problems[0]
    rounded = O.MassMatrix(dofs, np.diag([1.0, -1e-9]))                                         # rounding level: a null space, not a refusal
    ok = O.orthogonality(np.eye(2)[:, :1], np.eye(2)[:, :1], dofs, O.OrthMass("tam", tam=rounded))
    assert ok.computed and ok.info["rank"] == 1
    spread = O.MassMatrix([(f"N{i}", "x") for i in range(6)], np.diag([1.0, 3.0, 20.0, 300.0, 4000.0, 2e4]))
    full = O.orthogonality(np.eye(6)[:, :2], np.eye(6)[:, :2], spread.dofs, O.OrthMass("tam", tam=spread))
    assert full.computed and full.info["rank"] == 6 and not any("singular" in w for w in full.warnings)


# ------------------------------------------------------------------ 4: the notes name the modes, the Gram matrix is never guessed
def test_a_note_that_says_modes_are_excluded_names_them():
    phi, m = _generalised_chain([1.0] * 10)
    dofs = [(f"N{i}", "x") for i in range(10)]
    fe = phi[:, :4]
    test = fe.copy()
    test[3, 1] = np.nan
    test[0, 3] = np.nan
    g = phi.T @ m @ phi
    o = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g[:4, :4]), None, None, [11, 12, 13, 14])
    assert o.computed and any("test mode(s) 12, 14 have a DOF that was not measured" in n for n in o.notes)
    z = fe.copy()
    z[:, 2] = 0.0                                                                              # a zero shape: no COC for it
    o2 = O.orthogonality(z, fe, dofs, O.OrthMass("serep", gram=g[:4, :4]), None, None, [21, 22, 23, 24])
    assert any("test mode(s) 23 could not be computed" in n for n in o2.notes)


def test_the_fe_modes_of_a_tam_hold_are_named_by_their_mode_numbers():
    dofs = [(f"N{i}", "x") for i in range(8)]
    phi, _m = _generalised_chain([1.0] * 8)
    fe = phi[:, 3:6]
    tam = O.MassMatrix(dofs, np.outer(phi[:, 0], phi[:, 0]) + np.outer(phi[:, 7], phi[:, 7]), "blind.csv")
    orth = O.orthogonality(fe + 0.0, fe, dofs, O.OrthMass("tam", tam=tam), None, [3, 4, 5])
    assert "FE mode(s) 4, 5, 6 " in orth.hold
    assert any("test modes 1, 2, 3" in w for w in orth.warnings)


def test_a_gram_matrix_needs_explicit_labels_when_it_is_not_the_compared_modes_in_order():
    phi, m = _generalised_chain([1.0] * 10)
    dofs = [(f"N{i}", "x") for i in range(10)]
    g = phi.T @ m @ phi                                                                        # all 10 modes
    fe = phi[:, [2, 4, 5]]
    test = fe.copy()
    bare = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g), None, [2, 4, 5])
    assert not bare.computed and "gram_modes" in bare.problems[0] and "10 x 10" in bare.problems[0]       # no guess from the shape
    labelled = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g, gram_modes=list(range(10))), None, [2, 4, 5])
    assert labelled.computed and np.allclose(np.diag(labelled.coc), 1.0)
    swapped = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g[np.ix_([5, 4, 2, 9], [5, 4, 2, 9])], gram_modes=[5, 4, 2, 9]), None, [2, 4, 5])
    assert swapped.computed and np.allclose(swapped.coc, labelled.coc)                            # the labels, not the order, say which row is which mode
    lost = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g[:3, :3], gram_modes=[0, 1, 2]), None, [2, 4, 5])
    assert not lost.computed and "does not cover the compared FE mode(s) 5, 6" in lost.problems[0]
    dup = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g[:3, :3], gram_modes=[2, 2, 4]), None, [2, 4, 5])
    assert not dup.computed and "once" in dup.problems[0]
    assert O.OrthMass("serep", gram=g).fingerprint() != O.OrthMass("serep", gram=g, gram_modes=list(range(10))).fingerprint()

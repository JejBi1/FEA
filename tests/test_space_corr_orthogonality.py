"""Mass-weighted cross-orthogonality COC, auto-orthogonality AOC and modulus difference of the test correlation (F14 stage S5, ECSS-E-ST-32-11C 3.2.39, Table 5-2, 5.8.1.3 b):
the formulas against closed forms, the two sources of the reduced mass (SEREP of the FE modes, an imported TAM matrix), the bounded reader of the matrix, and - above all - that a
correlation whose orthogonality cannot be computed is never a PASS."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_core.eigen import ModeSet
from fea_core.mesh import Mesh
from fea_core.space import _corr_io as io
from fea_core.space import corr_orth as O
from fea_core.space import correlation as C

# ------------------------------------------------------------------ a closed-form two-mass chain (fixed - k - m1 - k - m2)


def _chain_modes(m1: float, m2: float, k: float = 1.0) -> tuple[list[float], np.ndarray]:
    """Eigenvalues and eigenvectors [1, r] of K = k [[2, -1], [-1, 1]], M = diag(m1, m2) from the quadratic m1 m2 l^2 - k (m1 + 2 m2) l + k^2 = 0 (no eigen-solver)."""
    b = k * (m1 + 2 * m2)
    disc = math.sqrt(b * b - 4 * m1 * m2 * k * k)
    lam = [(b - disc) / (2 * m1 * m2), (b + disc) / (2 * m1 * m2)]
    return lam, np.array([[1.0, (2 * k - lm * m1) / k] for lm in lam]).T          # (dof, mode)


def _coc_by_hand(phi_t: np.ndarray, phi_f: np.ndarray, m: list[float]) -> list[list[float]]:
    out = []
    for i in range(phi_t.shape[1]):
        row = []
        for j in range(phi_f.shape[1]):
            num = sum(phi_t[d, i] * m[d] * phi_f[d, j] for d in range(2))
            ei = sum(phi_t[d, i] * m[d] * phi_t[d, i] for d in range(2))
            ej = sum(phi_f[d, j] * m[d] * phi_f[d, j] for d in range(2))
            row.append(abs(num) / math.sqrt(ei * ej))
        out.append(row)
    return out


DOFS2 = [("A", "x"), ("B", "x")]


def test_the_closed_form_chain_is_orthogonal_in_its_own_mass_and_gives_the_hand_cross_orthogonality():
    _, phi_f = _chain_modes(1.0, 1.0)                                   # the golden-ratio modes [1, 1.618] and [1, -0.618]
    assert phi_f[1, 0] == pytest.approx((1 + math.sqrt(5)) / 2) and phi_f[1, 1] == pytest.approx((1 - math.sqrt(5)) / 2)
    tam = O.OrthMass("tam", tam=O.MassMatrix(DOFS2, np.eye(2)))
    r = O.orthogonality(phi_f, phi_f, DOFS2, tam)
    assert r.computed and np.allclose(r.coc, np.eye(2), atol=1e-14) and np.allclose(r.aoc, np.eye(2), atol=1e-14)
    _, phi_t = _chain_modes(1.0, 1.2)                                   # the test article has a 20 % heavier tip mass
    r = O.orthogonality(phi_t, phi_f, DOFS2, tam)
    ref = np.array(_coc_by_hand(phi_t, phi_f, [1.0, 1.0]))
    assert np.allclose(r.coc, ref, atol=1e-13) and ref[0, 1] > 0.01 and ref[1, 0] > 0.01 and ref[0, 0] < 1.0     # a real off-diagonal term, a diagonal below one
    aoc_ref = abs(sum(phi_t[d, 0] * phi_t[d, 1] for d in range(2))) / math.sqrt(sum(phi_t[d, 0] ** 2 for d in range(2)) * sum(phi_t[d, 1] ** 2 for d in range(2)))
    assert r.aoc[0, 1] == pytest.approx(aoc_ref, abs=1e-13) and aoc_ref > 0.01          # the test modes are not orthogonal in the FE mass


def test_the_coc_is_not_squared_and_ignores_the_sign_and_the_scale_of_the_shapes():
    _, phi_f = _chain_modes(1.0, 1.0)
    _, phi_t = _chain_modes(1.0, 1.2)
    tam = O.OrthMass("tam", tam=O.MassMatrix(DOFS2, np.diag([3.0, 5.0])))
    base = O.orthogonality(phi_t, phi_f, DOFS2, tam).coc
    scaled = O.orthogonality(phi_t * np.array([-7.0, 0.01]), phi_f * np.array([2.0, -3.0]), DOFS2, tam).coc
    assert np.allclose(base, scaled, atol=1e-13)
    num = abs(phi_t[:, 0] @ np.diag([3.0, 5.0]) @ phi_f[:, 1])
    den = math.sqrt((phi_t[:, 0] @ np.diag([3.0, 5.0]) @ phi_t[:, 0]) * (phi_f[:, 1] @ np.diag([3.0, 5.0]) @ phi_f[:, 1]))
    assert base[0, 1] == pytest.approx(num / den, abs=1e-14) and base[0, 1] != pytest.approx((num / den) ** 2, abs=1e-6)


def test_complex_test_shapes_use_the_hermitian_form():
    _, phi_f = _chain_modes(1.0, 1.0)
    z = np.exp(1j * 0.7)
    tam = O.OrthMass("tam", tam=O.MassMatrix(DOFS2, np.eye(2)))
    a = O.orthogonality(phi_f.astype(complex) * z, phi_f, DOFS2, tam)
    assert np.allclose(a.coc, np.eye(2), atol=1e-13)                     # an arbitrary phase of a whole mode changes nothing


# ------------------------------------------------------------------ the SEREP mass of the FE modes
def _chain(n: int, m: float = 2.0, k: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """n equal masses on springs, one end fixed: the closed-form modes sin((2r - 1) j pi / (2n + 1)) and the lumped mass matrix."""
    j = np.arange(1, n + 1)
    phi = np.stack([np.sin((2 * r - 1) * j * math.pi / (2 * n + 1)) for r in range(1, n + 1)], axis=1)
    return phi, m * np.eye(n)


def test_serep_with_every_dof_measured_reproduces_the_full_mass_orthogonality():
    n, nd = 5, 6                                                         # five FE modes, every one of the six DOFs measured (SEREP needs more DOFs than modes)
    phi_all, mass = _chain(nd)
    phi = phi_all[:, :n]
    dofs = [(f"N{i}", "x") for i in range(nd)]
    g = phi.T @ mass @ phi
    rng = np.random.default_rng(1)
    test = phi @ (np.eye(n) + 0.05 * rng.standard_normal((n, n)))        # test modes = mixtures of the FE modes
    full = O.orthogonality(test, phi, dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, mass)))
    serep = O.orthogonality(test, phi, dofs, O.OrthMass("serep", gram=g))
    assert np.allclose(serep.coc, full.coc, atol=1e-12) and np.allclose(serep.aoc, full.aoc, atol=1e-12)
    assert serep.info["condition"] < 10 and serep.info["fe_mass_orthogonality_error"] < 1e-12
    same = O.orthogonality(phi, phi, dofs, O.OrthMass("serep", gram=g))
    assert np.allclose(same.coc, np.eye(n), atol=1e-12) and np.allclose(same.aoc, np.eye(n), atol=1e-12)


def test_serep_with_fewer_dofs_equals_the_full_orthogonality_of_the_expanded_shape():
    n, keep = 8, [0, 2, 3, 5, 7]
    phi_all, mass = _chain(n)
    phi = phi_all[:, :3]                                                 # three FE modes
    g = phi.T @ mass @ phi
    dofs = [(f"N{i}", "x") for i in keep]
    coef = np.array([[1.0, 0.1, -0.05], [0.2, 1.0, 0.1], [-0.1, 0.15, 1.0]])
    test_full = phi @ coef                                               # test shapes in the span of the FE modes, all DOFs
    got = O.orthogonality(test_full[keep], phi[keep], dofs, O.OrthMass("serep", gram=g))
    ref_dofs = [(f"N{i}", "x") for i in range(n)]
    ref = O.orthogonality(test_full, phi, ref_dofs, O.OrthMass("tam", tam=O.MassMatrix(ref_dofs, mass)))
    assert np.allclose(got.coc, ref.coc, atol=1e-12) and np.allclose(got.aoc, ref.aoc, atol=1e-12)        # reduction and expansion agree for a shape in the span


def test_serep_refuses_what_it_cannot_do_and_says_why():
    phi, mass = _chain(6)
    g = phi.T @ mass @ phi
    dofs = [(f"N{i}", "x") for i in range(4)]
    r = O.orthogonality(phi[:4], phi[:4], dofs, O.OrthMass("serep", gram=g))                          # 4 DOFs for 6 modes
    assert not r.computed and "more sensor DOFs than FE modes" in r.problems[0] and "no remedy" in r.problems[0]
    r = O.orthogonality(phi, phi, [(f"N{i}", "x") for i in range(6)], O.OrthMass("serep", gram=g))                 # as many DOFs as modes: every shape is exactly representable
    assert not r.computed and "more sensor DOFs than FE modes" in r.problems[0]
    r = O.orthogonality(phi[:, :1], phi[:, :1], [(f"N{i}", "x") for i in range(6)], O.OrthMass("serep", gram=g[:1, :1]))        # one FE mode: COC is always 1
    assert not r.computed and "at least two FE modes" in r.problems[0]
    big, bmass = _chain(8)
    blind = big[:, :6].copy()
    blind[:, 5] = 0.0                                                    # a mode no sensor sees
    r = O.orthogonality(blind, blind, [(f"N{i}", "x") for i in range(8)], O.OrthMass("serep", gram=blind.T @ bmass @ blind + np.eye(6) * 1e-9))
    assert not r.computed and "condition number" in r.problems[0]
    dofs8 = [(f"N{i}", "x") for i in range(8)]
    assert not O.orthogonality(big[:, :6], big[:, :6], dofs8, O.OrthMass("serep", gram=np.eye(3))).computed       # a Gram matrix of the wrong size
    assert not O.orthogonality(big[:, :6], big[:, :6], dofs8, O.OrthMass("serep")).computed                           # no study, no mesh, no gram
    r = O.orthogonality(phi, phi, [(f"N{i}", "x") for i in range(6)], None)
    assert not r.computed and r.source == "none" and r.problems


def test_a_test_mode_with_an_unmeasured_dof_gets_no_orthogonality_but_the_others_do():
    phi_all, mass = _chain(6)
    phi = phi_all[:, :5]
    dofs = [(f"N{i}", "x") for i in range(6)]
    test = phi.astype(complex)
    test[2, 1] = np.nan
    r = O.orthogonality(test, phi, dofs, O.OrthMass("serep", gram=phi.T @ mass @ phi))
    assert r.computed and np.isnan(r.coc[1]).all() and np.isnan(r.aoc[1]).all() and np.isfinite(r.coc[0]).all() and any("not measured" in n for n in r.notes)
    all_bad = np.full_like(test, np.nan)
    assert not O.orthogonality(all_bad, phi, dofs, O.OrthMass("serep", gram=phi.T @ mass @ phi)).computed


def test_the_serep_gram_of_a_real_tet10_mesh_is_the_mass_of_the_body():
    from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
    from fea_core.dynamics.mass import total_mass
    from verification.space_cases import block_study
    x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([x, [(x[a] + x[b]) / 2 for a, b in mids]])
    mesh = Mesh(nodes, np.arange(10)[None, :], np.zeros(1, int), ["Block"])
    st = block_study()
    shapes = np.zeros((3, 10, 3), np.float32)
    shapes[0, :, 0] = 1.0                                                # rigid translations along x and y
    shapes[1, :, 1] = 1.0
    shapes[2, :, 0], shapes[2, :, 1] = nodes[:, 1] / 3.0, -nodes[:, 0] / 2.0          # a rotation about z
    ms = ModeSet("modal", "m", [1.0, 2.0, 3.0], shapes)
    g = O.serep_gram(st, mesh, ms, [0, 1, 2])
    m = total_mass(st, mesh)
    assert g[0, 0] == pytest.approx(m, rel=1e-12) and g[1, 1] == pytest.approx(m, rel=1e-12) and abs(g[0, 1]) < 1e-12 * m
    assert np.allclose(g, g.T) and g[2, 2] > 0
    assert O.serep_gram(st, mesh, ms, [1, 0])[0, 0] == pytest.approx(m, rel=1e-12)        # the order of `idx` is the order of the rows
    with pytest.raises(ValueError, match="mesh"):
        O.serep_gram(st, mesh, ModeSet("modal", "m", [1.0], np.zeros((1, 7, 3), np.float32)), [0])
    fp = O.OrthMass("serep", study=st, mesh=mesh).fingerprint()
    st.materials[next(iter(st.materials))].rho *= 2.0
    assert O.OrthMass("serep", study=st, mesh=mesh).fingerprint() != fp                  # another density: another mass model


# ------------------------------------------------------------------ the imported matrix
GOOD = "# unit: kg\n,A:x,A:y,B:x\nA:x,2,0,0.5\nA:y,0,2,0\nB:x,0.5,0,1\n"


def test_the_mass_matrix_csv_is_read_and_fingerprinted():
    mm = O.parse_mass_csv(GOOD, "tam.csv")
    assert mm.dofs == [("A", "x"), ("A", "y"), ("B", "x")] and mm.unit == "kg" and mm.matrix[0, 2] == 0.5 and mm.summary()["dofs"] == 3
    assert O.parse_mass_csv(GOOD.replace("2,0,0.5", "2,0,0.6").replace("0.5,0,1", "0.6,0,1")).fingerprint() != mm.fingerprint()
    assert O.parse_mass_csv(GOOD.replace(",", ";")).dofs == mm.dofs                       # the delimiter is detected
    assert O.parse_mass_csv(GOOD.replace("kg", "t")).unit == "t"


@pytest.mark.parametrize("text, why", [
    ("", "header row"),
    (",A:x,A:y\nA:x,1,0\n", "square"),
    (",A:x,A:x\nA:x,1,0\nA:x,0,1\n", "twice"),
    (",A:x,B:x\nA:x,1,0.5\nB:x,0.1,1\n", "symmetric"),
    (",A:x,B:x\nA:x,1,2\nB:x,2,1\n", "positive semi-definite"),
    (",A:x,B:x\nA:x,1,0\nB:x,0,nan\n", "finite|number"),
    (",A:x,B:x\nB:x,1,0\nA:x,0,1\n", "order of the columns"),
    (",A:rx,B:x\nA:rx,1,0\nB:x,0,1\n", "translations"),
    (",A:-x,B:x\nA:-x,1,0\nB:x,0,1\n", "signed"),
    (",A,B:x\nA,1,0\nB:x,0,1\n", "node:dof"),
    ("# unit: m/s\n,A:x\nA:x,1\n", "not a mass unit"),
    (",A:x,B:x\nA:x,0,0\nB:x,0,0\n", "zero"),
    (",A:x,B:x\nA:x,1,0,5\nB:x,0,1\n", "cells"),
])
def test_a_bad_mass_matrix_is_refused_with_a_reason(text, why):
    with pytest.raises(io.TestDataError, match=why):
        O.parse_mass_csv(text)


def test_the_mass_matrix_reader_is_bounded(tmp_path, monkeypatch):
    p = tmp_path / "m.csv"
    p.write_text(GOOD)
    assert O.load_mass_file(p).sha256 and len(O.load_mass_file(p).sha256[0]) == 64
    with pytest.raises(io.TestDataError, match="does not exist"):
        O.load_mass_file(tmp_path / "nope.csv")
    with pytest.raises(io.TestDataError, match="UNC"):
        O.load_mass_file("\\\\server\\share\\m.csv")
    monkeypatch.setattr(O, "MAX_MASS_DOFS", 2)
    with pytest.raises(io.TestDataError, match="1 to 2 DOFs"):
        O.parse_mass_csv(GOOD)
    (tmp_path / "bin.csv").write_bytes(b"\x00\x01")
    with pytest.raises(io.TestDataError, match="binary"):
        O.load_mass_file(tmp_path / "bin.csv")


def test_the_tam_must_have_exactly_the_sensor_dofs_and_its_fe_orthogonality_is_reported():
    phi, mass = _chain(4)
    dofs = [(f"N{i}", "x") for i in range(4)]
    tam = O.MassMatrix(dofs[::-1], mass[::-1, ::-1])                        # the rows may come in any order
    r = O.orthogonality(phi, phi, dofs, O.OrthMass("tam", tam=tam))
    assert r.computed and np.allclose(r.coc, np.eye(4), atol=1e-12) and r.info["fe_offdiagonal_max"] < 1e-12 and r.info["fe_diagonal_min"] == pytest.approx(1.0)
    short = O.orthogonality(phi[:3], phi, dofs[:3], O.OrthMass("tam", tam=tam))
    assert not short.computed and "not sensor DOFs: N3:x" in short.problems[0]
    other = O.MassMatrix([("Q", "x")] * 1, np.eye(1))
    assert "missing in the matrix" in O.orthogonality(phi, phi, dofs, O.OrthMass("tam", tam=other)).problems[0]
    skew = O.MassMatrix(dofs, np.diag([1.0, 1.0, 1.0, 9.0]))               # a TAM that does not belong to the model: the FE modes are not orthogonal through it
    assert O.orthogonality(phi, phi, dofs, O.OrthMass("tam", tam=skew)).info["fe_offdiagonal_max"] > 0.05
    assert not O.orthogonality(phi, phi, dofs, O.OrthMass("tam")).computed


# ------------------------------------------------------------------ the modulus difference
def test_the_modulus_difference_by_hand():
    dofs = [("a", "x"), ("b", "x"), ("c", "x")]
    t = np.array([[1.0], [2.0], [2.0]])
    f = np.array([[1.0], [2.0], [1.0]])
    (v, where), = O.modulus_difference(t, f, [(0, 0)], dofs)
    tn, fn = np.array([1, 2, 2]) / 3.0, np.array([1, 2, 1]) / math.sqrt(6.0)
    d = np.abs(tn - fn)
    assert v == pytest.approx(d.max() / tn.max(), abs=1e-14) and where == "c:x" and v == pytest.approx(0.3876275643, abs=1e-9)
    assert O.modulus_difference(t * -4.0, f * 0.5, [(0, 0)], dofs)[0][0] == pytest.approx(v, abs=1e-14)                         # sign and scale do not matter
    assert O.modulus_difference(t * np.exp(0.4j), f, [(0, 0)], dofs)[0][0] == pytest.approx(v, abs=1e-14)                       # nor the phase of a complex shape
    assert O.modulus_difference(t, t, [(0, 0)], dofs)[0][0] == 0.0
    nan = t.astype(complex)
    nan[0, 0] = np.nan
    assert O.modulus_difference(nan, f, [(0, 0)], dofs)[0][0] is not None
    one = np.array([[np.nan], [np.nan], [1.0]], complex)
    assert O.modulus_difference(one, f, [(0, 0)], dofs)[0] == (None, "")


# ------------------------------------------------------------------ the rows and the verdict: never a false PASS
def _limits(extra: dict[str, float] | None = None, na: tuple[str, ...] = ()) -> C.CorrelationLimits:
    from test_space_correlation import EXAMPLE, _limits_set
    sp = _limits_set({**EXAMPLE, **(extra or {})})
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    for k in na:
        lim.entries[k] = C.UsedLimit(None, "doc", "rev", "c1", False, "declared not applicable", True, "gui")
    return lim


def _modal(orth: O.OrthMass | None, test_shift: float = 0.0, n: int = 5):
    phi_all, mass = _chain(n + 3)                                        # n FE modes at n + 3 DOFs
    phi = phi_all[:, :n]
    dofs = [(f"N{i}", "x") for i in range(n + 3)]
    g = phi.T @ mass @ phi
    f = np.array([10.0 * (r + 1) for r in range(n)])
    test = phi.copy()
    if test_shift:
        test = phi @ (np.eye(n) + test_shift * np.ones((n, n)))
    om = None if orth is None else (O.OrthMass("serep", gram=g) if orth.kind == "serep" else (O.OrthMass("tam", tam=O.MassMatrix(dofs, mass)) if orth.kind == "tam" and orth.tam is None else orth))
    r = O.orthogonality(test, phi, dofs, om) if om is not None else None
    return C.modal_correlation(f, test, f, phi, dofs, options=C.CorrelationOptions(), orth=r)


def test_without_a_reduced_mass_the_orthogonality_rows_are_unconfirmed_and_the_verdict_is_never_pass():
    lim = _limits()
    for modal in (_modal(None), _modal(O.OrthMass("none", why="switched off"))):
        rows = C.evaluate(modal, [], lim)
        orth_rows = [r for r in rows if r.metric in ("COC diagonal", "COC off-diagonal", "AOC off-diagonal")]
        assert len(orth_rows) == 5 + 5 + 5 and all(r.status == C.UNCONFIRMED and r.value is None and r.note for r in orth_rows)
        assert C.verdict_of(rows) == C.VERDICT_UNCONFIRMED                 # every MAC / frequency / modulus row passes, and still no PASS
        assert all(r.status == C.PASS for r in rows if r not in orth_rows)
    assert all("switched off" in r.note for r in C.evaluate(_modal(O.OrthMass("none", why="switched off")), [], lim) if r.metric == "COC diagonal")


def test_with_an_imported_tam_the_rows_are_judged_against_the_confirmed_limits():
    lim = _limits()
    ok = C.evaluate(_modal(O.OrthMass("tam")), [], lim)
    assert C.verdict_of(ok) == C.PASS
    coc = {r.metric: r for r in ok if r.item.startswith("test mode 1 <->") or r.item == "test mode 1"}
    assert coc["COC diagonal"].value == pytest.approx(1.0) and coc["COC diagonal"].limit == 0.8 and coc["modulus difference"].value == pytest.approx(0.0, abs=1e-12)
    bad = C.evaluate(_modal(O.OrthMass("tam"), test_shift=0.4), [], _limits({"corr_coc_diag_min": 0.999, "corr_coc_offdiag_max": 0.01}))
    assert C.verdict_of(bad) == C.FAIL and any(r.status == C.FAIL and r.metric == "COC diagonal" for r in bad) and any(r.status == C.FAIL and r.metric == "COC off-diagonal" for r in bad)
    aoc = C.evaluate(_modal(O.OrthMass("tam"), test_shift=0.4), [], _limits({"corr_aoc_offdiag_max": 0.01}))
    assert any(r.status == C.FAIL and r.metric == "AOC off-diagonal" for r in aoc)


def test_the_serep_rows_are_indicative_never_pass_and_never_fail_whatever_the_value():
    """Owner decision 25. 11. 2026: only a TAM can PASS / FAIL a COC / AOC row; the SEREP number is shown and the row is UNCONFIRMED with the stated reason."""
    names = ("COC diagonal", "COC off-diagonal", "AOC off-diagonal")
    ok = C.evaluate(_modal(O.OrthMass("serep")), [], _limits())
    orth_rows = [r for r in ok if r.metric in names]
    assert len(orth_rows) == 15 and all(r.status == C.UNCONFIRMED and r.value is not None and C.INDICATIVE_REASON in r.note and not r.indicative_outside for r in orth_rows)
    assert C.INDICATIVE_REASON.startswith("indicative (SEREP reduction from the FE modes) - a PASS needs a test-analysis-model (TAM) mass matrix from the test lab")
    assert C.verdict_of(ok) == C.VERDICT_UNCONFIRMED and all(r.status == C.PASS for r in ok if r not in orth_rows)             # MAC, frequency and modulus difference are judged as before
    lenient = {"corr_mac_min": 0.0, "corr_mac_offdiag_max": 1.0, "corr_comac_min": 0.0, "corr_modulus_diff_max": 1.0}          # the shifted shape must not fail the MAC rows here
    bad = C.evaluate(_modal(O.OrthMass("serep"), test_shift=0.4), [], _limits({"corr_coc_diag_min": 0.999, "corr_coc_offdiag_max": 0.01, "corr_aoc_offdiag_max": 0.01, **lenient}))
    hinted = [r for r in bad if r.metric in names and r.indicative_outside]
    assert hinted and all(r.status == C.UNCONFIRMED and "warning hint only" in r.note and r.to_dict()["indicative_outside"] for r in hinted)
    assert not any(r.status == C.FAIL for r in bad if r.metric in names) and C.verdict_of(bad) == C.VERDICT_UNCONFIRMED       # an informative number does not make a FAIL verdict
    na = _limits(na=("corr_coc_diag_min", "corr_coc_offdiag_max", "corr_aoc_offdiag_max"))
    rows = C.evaluate(_modal(O.OrthMass("serep")), [], na)                  # declared not applicable: not judged at all
    assert all(r.status == C.NOT_APPLICABLE for r in rows if r.metric in names) and C.verdict_of(rows) == C.PASS


def test_a_missing_limit_or_a_declared_not_applicable_criterion():
    m = _modal(O.OrthMass("serep"))
    sp_less = _limits()
    del sp_less.entries["corr_coc_diag_min"]
    sp_less.entries["corr_coc_diag_min"] = C.UsedLimit()
    rows = C.evaluate(m, [], sp_less)
    assert all(r.status == C.UNCONFIRMED for r in rows if r.metric == "COC diagonal") and C.verdict_of(rows) == C.VERDICT_UNCONFIRMED
    na = _limits(na=("corr_coc_diag_min", "corr_coc_offdiag_max", "corr_aoc_offdiag_max", "corr_modulus_diff_max"))
    rows = C.evaluate(_modal(None), [], na)                               # not computable AND declared not applicable: not judged, not a hold-back
    assert all(r.status == C.NOT_APPLICABLE for r in rows if r.metric in ("COC diagonal", "COC off-diagonal", "AOC off-diagonal", "modulus difference"))
    assert C.verdict_of(rows) == C.PASS
    assert C.not_applicable_limits(na) and {x["limit"] for x in C.not_applicable_limits(na)} >= {"corr_coc_diag_min", "corr_modulus_diff_max"}


def test_a_stored_pass_is_recomputed_against_the_current_limits_for_the_new_keys():
    from fea_core.space import corr_run as CR
    lim = _limits()
    row = {"item": "x", "metric": "COC diagonal", "value": 0.7, "status": C.PASS, "limit_key": "corr_coc_diag_min"}      # 0.7 < 0.8: a PASS that no longer follows
    assert CR._recompute_rows([row], lim)
    assert not CR._recompute_rows([{**row, "value": 0.95}], lim)


def test_limit_hints_are_hints_and_no_default_value_exists():
    assert set(C.LIMIT_HINTS) <= set(C.LIMIT_INFO) and "0,90" in C.LIMIT_HINTS["corr_coc_diag_min"] and "not normative" in C.LIMIT_HINTS["corr_coc_diag_min"]
    empty = C.CorrelationLimits()
    assert all(empty.value(k) is None for k in C.LIMIT_HINTS)
    from fea_core.space.model import RequirementSet
    rs = RequirementSet(name="x")
    C.install_limit_rows(rs)
    rows = {x.key: x for x in rs.limits}
    assert rows["corr_coc_diag_min"].value is None and rows["corr_aoc_offdiag_max"].value is None and rows["corr_modulus_diff_max"].value is None


def test_options_validate_the_mass_source():
    assert C.CorrelationOptions().orth_mass == "serep"
    assert C.CorrelationOptions.from_dict({"orth_mass": "tam"}).orth_mass == "tam" and "tam" in C.CorrelationOptions(orth_mass="tam").describe()
    with pytest.raises(ValueError, match="orth_mass"):
        C.CorrelationOptions.from_dict({"orth_mass": "guyan"})


# ------------------------------------------------------------------ the runner: the TAM, the provenance and the verification of a stored result
def _fe_and_td():
    import test_space_correlation as T
    mesh = T._box_mesh()
    ms = T._modes(mesh)
    return (mesh, ms), T._test_data((mesh, ms))


def test_the_runner_uses_an_imported_tam_and_verifies_the_stored_result_against_it(monkeypatch):
    import test_space_correlation as T
    from fea_core.space import corr_run as CR
    fe, td = _fe_and_td()
    mesh, ms = fe
    st = T._study_with_limits(T.EXAMPLE)
    sens = list(T.SENSORS)
    dofs = [(s.node, d) for s in sens for d in s.dofs]
    # a TAM = a unit lumped mass at the sensor DOFs; the FE modes at those DOFs are NOT orthogonal in it, which the report of the TAM says
    tam = O.MassMatrix(dofs, np.eye(len(dofs)), "unit.csv", "kg", ["0" * 64])
    results = T._results(fe)
    opt = C.CorrelationOptions(orth_mass="tam")
    lim_ok = T._limits_set({**T.EXAMPLE, "corr_coc_offdiag_max": 1.0, "corr_aoc_offdiag_max": 1.0, "corr_coc_diag_min": 0.0})
    st.space = lim_ok
    res, _ = CR.run_correlation(st, results, td, sensors=sens, options=opt, tam=tam)
    assert res.modal.orth.source == "tam" and res.modal.orth.computed and res.inputs["orth_mass"]["source"] == "tam"
    assert all(p.coc is not None and p.coc > 0.99 for p in res.modal.pairs)             # the test is the FE model itself
    v = CR.verify_stored(st, results, td)
    assert v is not None and v["current"] and "mass matrix (TAM)" in v["verified"], v["why"]
    CR.remember_tam(st, None)
    v = CR.verify_stored(st, results, td)
    assert not v["current"] and any("mass matrix (TAM)" in u for u in v["unverified"]) and v["verdict"] == C.VERDICT_UNVERIFIABLE
    other = O.MassMatrix(dofs, 2.0 * np.eye(len(dofs)), "other.csv")                    # (the session keeps only a weak reference: hold it)
    CR.remember_tam(st, other)
    v = CR.verify_stored(st, results, td)
    assert any("not the one the cross-orthogonality" in r for r in v["reasons"]) and v["verdict"] == C.VERDICT_OUTDATED
    res2, _ = CR.run_correlation(st, results, td, sensors=sens, options=C.CorrelationOptions(orth_mass="tam"), tam=None)                 # the TAM of the session is taken
    assert res2.modal.orth.source == "tam"
    CR.remember_tam(st, None)
    res3, _ = CR.run_correlation(st, results, td, sensors=sens, options=C.CorrelationOptions(orth_mass="tam"))
    assert not res3.modal.orth.computed and "no mass matrix" in res3.modal.orth.problems[0] and res3.verdict == C.VERDICT_UNCONFIRMED
    res4, _ = CR.run_correlation(st, results, td, sensors=sens, options=C.CorrelationOptions(orth_mass="none"))
    assert "switched off" in res4.modal.orth.problems[0] and res4.verdict == C.VERDICT_UNCONFIRMED


def test_the_serep_mass_of_the_study_is_part_of_the_provenance(monkeypatch):
    """With the unit-mass Gram of the synthetic tet4 model the fingerprint is the Gram's; the real route (study + tet10 mesh) fingerprints the densities and added masses."""
    import test_space_correlation as T
    from fea_core.space import corr_run as CR
    fe, td = _fe_and_td()
    results = T._results(fe)
    st = T._study_with_limits(T.EXAMPLE)
    # the real route on a tet4 mesh: the mass cannot be computed (tet10 only): the rows are UNCONFIRMED with the reason, never a crash and never a PASS
    res, _ = CR.run_correlation(st, results, td, sensors=list(T.SENSORS))
    assert not res.modal.orth.computed and "tet10" in res.modal.orth.problems[0] and res.verdict == C.VERDICT_UNCONFIRMED
    assert res.inputs["orth_mass"]["computed"] is False

"""Review fixes of the COC / AOC (F14-S5, 25. 11. 2026): the SEREP COC is the COC of the PROJECTION of a test shape onto the compared FE modes - INDICATIVE only (owner decision 25. 11. 2026): the
numbers, the out-of-span residual and the warnings are shown, every SEREP row is UNCONFIRMED; ill-conditioned sensors, a study mass that is not the mass of the modes, a TAM that does not fit or
whose axes are not stated, an old stored result, the TAM bound to the study, the bounded mass-matrix reader (300 and 2000 DOFs), the AOC notes and the NaN above 1."""
from __future__ import annotations

import math
import time
import tracemalloc

import numpy as np
import pytest

from test_space_corr_orthogonality import _chain, _limits
from fea_core.space import _corr_io as io
from fea_core.space import corr_orth as O
from fea_core.space import correlation as C

ND, NM = 10, 5                                                           # sensor DOFs, FE modes of the comparison


def _setup(nm: int = NM):
    phi_all, mass = _chain(ND)
    phi = phi_all[:, :nm]
    dofs = [(f"N{i}", "x") for i in range(ND)]
    g = phi.T @ mass @ phi
    d = np.sqrt(np.diag(g))
    return phi, mass, g, d, dofs


def _rows(test, fe, g, dofs, lim=None, f_t=None):
    orth = O.orthogonality(test, fe, dofs, O.OrthMass("serep", gram=g))
    nt, nf = test.shape[1], fe.shape[1]
    modal = C.modal_correlation(np.arange(1, nt + 1) * 10.0, test, np.arange(1, nf + 1) * 10.0, fe, dofs, orth=orth)
    return orth, modal, C.evaluate(modal, [], lim or _limits())


def _status(rows, metric, mode_item="test mode 1"):
    return [r.status for r in rows if r.metric == metric and r.item.startswith(mode_item)]


# ------------------------------------------------------------------ HIGH 1: the COC of a projection must not confirm a shape that matches nothing
def test_a_random_test_shape_against_two_fe_modes_is_unconfirmed_not_a_confirmed_coc_of_one():
    phi, _mass, g, _d, dofs = _setup()
    rng = np.random.default_rng(3)
    t = rng.standard_normal((ND, 1))
    orth, modal, rows = _rows(t, phi[:, :2], g[:2, :2], dofs)
    assert orth.coc[0].max() > 0.5 and modal.pairs[0].mac < 0.5                      # (a COC of the projection can be large while the MAC says the shape is foreign)
    assert orth.residual is not None and orth.residual[0] > 0.3 and not orth.mode_hold and not orth.hold      # no hold rules: the row is UNCONFIRMED because the value is indicative
    assert any("test mode(s) 1 have more than 5 % of their shape outside the span" in w for w in orth.warnings)         # the warning names the mode
    assert set(_status(rows, "COC diagonal")) == {C.UNCONFIRMED} and set(_status(rows, "AOC off-diagonal", "test mode 1")) <= {C.UNCONFIRMED}
    assert [r for r in rows if r.metric == "COC diagonal"][0].value is not None       # the number is shown, with the reason beside it
    assert C.INDICATIVE_REASON in [r for r in rows if r.metric == "COC diagonal"][0].note
    assert C.verdict_of(rows) != C.PASS


def test_the_sums_of_squares_of_a_serep_coc_row_are_one_whatever_the_shape_is():
    phi, _mass, g, _d, dofs = _setup()
    t = np.random.default_rng(5).standard_normal((ND, 3))
    orth = O.orthogonality(t, phi, dofs, O.OrthMass("serep", gram=g))
    assert np.allclose((orth.coc ** 2).sum(axis=1), 1.0, atol=1e-12)               # the documented property of the projection: it is why the residual is needed


def test_an_fe_mode_plus_a_residual_outside_the_span_is_warned_above_the_tolerance_and_never_judged():
    phi, _mass, g, _d, dofs = _setup()
    q, _ = np.linalg.qr(phi)
    rng = np.random.default_rng(1)
    r = rng.standard_normal(ND)
    r -= q @ (q.T @ r)
    r /= np.linalg.norm(r)
    for ratio, held in ((0.75, True), (0.1, False)):
        t = phi.copy()
        t[:, 0] = phi[:, 0] + ratio * np.linalg.norm(phi[:, 0]) * r
        orth, _m, rows = _rows(t, phi, g, dofs)
        want = ratio ** 2 / (1 + ratio ** 2)
        assert orth.residual[0] == pytest.approx(want, rel=1e-9)                     # r = sin^2 of the angle to the span
        assert not orth.mode_hold and any("outside the span" in w for w in orth.warnings) is held
        assert set(_status(rows, "COC diagonal")) == {C.UNCONFIRMED}                  # a SEREP row is never PASS, whatever the residual


def test_the_coc_depends_on_the_selected_fe_modes_and_the_selection_cannot_hide_the_difference():
    phi, _mass, g, d, dofs = _setup(8)
    t = np.zeros((ND, 2))
    t[:, 0] = 0.8 * phi[:, 0] / d[0] + 0.6 * phi[:, 4] / d[4]                         # 0.8 phi_1 + 0.6 phi_5 (mass-normalised), the second test mode is FE mode 2
    t[:, 1] = phi[:, 1]
    o8, _m8, r8 = _rows(t, phi, g, dofs)
    assert o8.coc[0, 0] == pytest.approx(0.8, abs=1e-9) and not o8.warnings           # all eight modes: the shape is inside the span, the COC is the true 0.8 ...
    assert set(_status(r8, "COC diagonal")) == {C.UNCONFIRMED}                       # ... and still only indicative
    o2, _m2, r2 = _rows(t, phi[:, :2], g[:2, :2], dofs)
    assert o2.coc[0, 0] > 0.99 and o2.residual[0] > 0.2                                # two modes: COC 0.9989 for a shape that is 36 % phi_5
    assert o2.coc[0, 0] > o8.coc[0, 0] and any("outside the span" in w for w in o2.warnings)
    assert _status(r2, "COC diagonal")[0] == C.UNCONFIRMED and C.verdict_of(r2) != C.PASS       # the selection cannot turn the number into a PASS


def test_serep_refuses_one_fe_mode_and_as_many_dofs_as_modes_and_does_not_recommend_dropping_modes():
    phi, _mass, g, _d, dofs = _setup()
    r1 = O.orthogonality(np.random.default_rng(2).standard_normal((ND, 1)), phi[:, :1], dofs, O.OrthMass("serep", gram=g[:1, :1]))
    assert not r1.computed and "at least two FE modes" in r1.problems[0]
    big, mass = _chain(6)
    r2 = O.orthogonality(big, big, [(f"N{i}", "x") for i in range(6)], O.OrthMass("serep", gram=big.T @ mass @ big))
    assert not r2.computed and "more sensor DOFs than FE modes" in r2.problems[0] and "no remedy" in r2.problems[0]
    assert "restrict" not in r2.problems[0] and "fe_modes" not in r2.problems[0]


# ------------------------------------------------------------------ MEDIUM 2: a mass that is not the mass of the modes
def test_a_study_mass_that_is_not_the_mass_of_the_modes_warns_and_prints_the_error():
    phi, _mass, g, _d, dofs = _setup()
    good = O.orthogonality(phi, phi, dofs, O.OrthMass("serep", gram=g))
    assert good.hold == "" and good.info["fe_mass_orthogonality_error"] < 1e-12 and not good.warnings
    for eps, held in ((5e-4, False), (5e-3, True), (0.2, True)):
        gg = g.copy()
        gg[0, 1] = gg[1, 0] = eps * math.sqrt(gg[0, 0] * gg[1, 1])
        o, _m, rows = _rows(phi, phi, gg, dofs)
        assert o.info["fe_mass_orthogonality_error"] == pytest.approx(eps, rel=1e-9)
        assert o.hold == "" and set(_status(rows, "COC diagonal")) == {C.UNCONFIRMED}       # no hold rules: the SEREP rows are indicative anyway
        assert any("not mass-orthogonal" in w and f"{eps:.3g}" in w for w in o.warnings) is held


def test_the_report_prints_the_orthogonality_error_the_residual_and_the_indicative_notice():
    pytest.importorskip("docx")
    from fea_core.report import space_corr as RC
    phi, _mass, g, _d, dofs = _setup()
    gg = g.copy()
    gg[0, 1] = gg[1, 0] = 0.01 * math.sqrt(gg[0, 0] * gg[1, 1])
    orth, modal, rows = _rows(phi + 0.0, phi, gg, dofs)
    from fea_core.report.word import _Doc
    D = _Doc()
    d = modal.to_dict(True)
    RC._orthogonality(D, d, d["pairs"], [])
    text = "\n".join([p.text for p in D.doc.paragraphs] + [c.text for t in D.doc.tables for row in t.rows for c in row.cells])
    assert "largest off-diagonal of the FE modes under the mass of the study 0.01" in text and "out-of-span residual" in text
    assert "INDICATIVE ONLY" in text and "PROJECTION" in text and "needs a test-analysis-model (TAM) mass matrix" in text and "never makes a FAIL" in text
    assert "PROGRAM'S OWN definition" in text and "Out-of-span residual" in text and "COC (indicative)" in text


# ------------------------------------------------------------------ MEDIUM 3: the condition of the mass-normalised sensor partition
@pytest.mark.parametrize("weak, warns, computed", [(1 / 50, False, True), (1 / 200, True, True), (1 / 2000, True, True), (1e-10, False, False)])
def test_the_condition_of_the_mass_normalised_sensor_partition_warns_and_refuses(weak, warns, computed):
    q, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((ND, NM)))
    fe = q * np.array([1.0, 1.0, 1.0, 1.0, weak])                                      # with G = I the mass-normalised partition is this matrix: cond = 1 / weak
    o = O.orthogonality(q[:, :4], fe, [(f"N{i}", "x") for i in range(ND)], O.OrthMass("serep", gram=np.eye(NM)))
    assert o.computed is computed
    if computed:
        assert o.info["condition"] == pytest.approx(1 / weak, rel=1e-6)
        assert o.hold == "" and any("condition number" in w for w in o.warnings) is warns


def test_one_percent_noise_with_a_weakly_seen_mode_is_warned_and_not_a_confirmed_row():
    q, _ = np.linalg.qr(np.random.default_rng(6).standard_normal((ND, NM)))
    weak = 1e-3                                                                      # the reviewer's case: cond 1.3e4 gave COC diag 0.016 and a passing-looking AOC
    fe = q * np.array([1.0, 1.0, 1.0, 1.0, weak])
    t = fe[:, :4] + 0.01 * np.abs(fe[:, :4]).max() * np.random.default_rng(7).standard_normal((ND, 4))
    o, _m, rows = _rows(t, fe, np.eye(NM), [(f"N{i}", "x") for i in range(ND)])
    assert o.computed and any("condition number" in w for w in o.warnings)
    assert {r.status for r in rows if r.metric in ("COC diagonal", "COC off-diagonal", "AOC off-diagonal")} == {C.UNCONFIRMED}


# ------------------------------------------------------------------ MEDIUM 4: the bounded mass-matrix reader accepts the advertised sizes
def _tam_text(n: int, fmt: str) -> str:
    rng = np.random.default_rng(n)
    m = np.eye(n) * 5.0 + 0.001 * rng.random((n, n))
    m = 0.5 * (m + m.T)
    lab = [f"N{i // 3 + 1}:{'xyz'[i % 3]}" for i in range(n)]
    return "# unit: kg\n," + ",".join(lab) + "\n" + "\n".join(lab[i] + "," + ",".join(fmt % v for v in m[i]) for i in range(n)) + "\n"


def test_a_300_dof_mass_matrix_with_full_precision_numbers_is_read(tmp_path):
    text = _tam_text(300, "%r")
    assert max(len(x) for x in text.splitlines()) > io.MAX_LINE_CHARS                  # a row longer than the 4096 characters of a test-file line
    p = tmp_path / "tam300.csv"
    p.write_text(text)
    mm = O.load_mass_file(p)
    assert mm.matrix.shape == (300, 300) and len(mm.dofs) == 300 and mm.unit == "kg"


@pytest.mark.slow
def test_a_2000_dof_mass_matrix_is_read(tmp_path):
    text = _tam_text(2000, "%.4g")
    p = tmp_path / "tam2000.csv"
    p.write_text(text)
    assert p.stat().st_size < O.MAX_MASS_BYTES
    t0 = time.time()
    mm = O.load_mass_file(p)
    assert mm.matrix.shape == (2000, 2000) and time.time() - t0 < 120


def test_the_mass_matrix_reader_checks_the_header_and_the_row_count_before_it_parses_the_body():
    txt = ("," * 4000 + "\n") * 2000                                                 # 8 MB of separators: the old reader built 8 million cells before the bound
    tracemalloc.start()
    with pytest.raises(io.TestDataError, match="1 to 2000 DOFs"):
        O.parse_mass_csv(txt)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 80 * 2 ** 20, peak
    head = "," + ",".join(f"N{i}:x" for i in range(3))
    with pytest.raises(io.TestDataError, match="square: 3 DOF columns but 5 rows"):
        O.parse_mass_csv(head + "\n" + "\n".join(f"N{min(i, 2)}:x,1,0,0" for i in range(5)))
    with pytest.raises(io.TestDataError, match="too long"):
        O.parse_mass_csv("," + "N1:x," * (O.MAX_MASS_LINE_CHARS // 5 + 10) + "\n")


# ------------------------------------------------------------------ MEDIUM 5 / 6: the TAM axes and a TAM that does not fit
def _box_case():
    import test_space_correlation as T
    mesh = T._box_mesh()
    ms = T._modes(mesh)
    td = T._test_data((mesh, ms))
    dofs = [(s.node, d) for s in T.SENSORS for d in s.dofs]
    return T, mesh, ms, td, dofs


def test_a_tam_with_rotated_sensors_stays_unconfirmed_until_the_user_states_global_axes():
    T, mesh, ms, td, dofs = _box_case()
    fphi, _w = C.fe_shapes_at_sensors(ms, mesh, T.SENSORS, dofs)                      # a TAM that fits: the SEREP mass of the modes at the sensors (unit-mass Gram, diagonal)
    pinv = np.linalg.pinv(fphi)
    tam = O.MassMatrix(dofs, pinv.T @ np.diag(np.sum(np.asarray(ms.shapes, float).reshape(len(ms.values), -1) ** 2, axis=1)) @ pinv, "fit.csv", "kg")
    td.cs_matrices = {"100": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])}            # a node that the file rotated (UFF 2420)
    lim = C.CorrelationLimits.from_requirement_set(T._limits_set({**T.EXAMPLE, "corr_coc_offdiag_max": 1.0, "corr_aoc_offdiag_max": 1.0, "corr_coc_diag_min": 0.0}).requirement_set,
                                                   T._limits_set({**T.EXAMPLE, "corr_coc_offdiag_max": 1.0, "corr_aoc_offdiag_max": 1.0, "corr_coc_diag_min": 0.0}))
    res = C.correlate(td, ms, mesh, T.SENSORS, limits=lim, orth_mass=O.OrthMass("tam", tam=tam))
    assert res.modal.orth.computed and "global axes" in res.modal.orth.hold and res.verdict == C.VERDICT_UNCONFIRMED
    coc = [r for r in res.rows if r.metric == "COC diagonal"]
    assert coc and all(r.status == C.UNCONFIRMED and "global axes" in r.note and r.value is not None for r in coc)
    assert any("global axes" in w for w in res.warnings)
    ok = C.correlate(td, ms, mesh, T.SENSORS, limits=lim, orth_mass=O.OrthMass("tam", tam=tam), options=C.CorrelationOptions(orth_mass="tam", tam_global_axes=True))
    assert ok.modal.orth.hold == "" and ok.verdict == C.PASS
    plain = T._test_data((mesh, ms))                                                   # no rotated sensor: nothing to confirm
    assert C.correlate(plain, ms, mesh, T.SENSORS, limits=lim, orth_mass=O.OrthMass("tam", tam=tam)).modal.orth.hold == ""
    assert C.CorrelationOptions.from_dict(C.CorrelationOptions(orth_mass="tam", tam_global_axes=True).to_dict()).tam_global_axes is True
    assert C.CorrelationOptions.from_dict({"tam_global_axes": "yes"}).tam_global_axes is False            # only a real boolean true counts


def test_a_tam_that_does_not_fit_the_model_is_a_warning_and_holds_the_rows_without_a_wrong_clause():
    phi, _mass, g, _d, dofs = _setup()
    b = np.random.default_rng(8).standard_normal((ND, ND))
    bad = O.MassMatrix(dofs, b @ b.T + 0.1 * np.eye(ND), "bad.csv")
    orth, modal, rows = _rows(phi[:, :4] + 0.0, phi[:, :4], g[:4, :4], dofs)           # (the serep rows, for the structure of the call below)
    o = O.orthogonality(phi[:, :4], phi[:, :4], dofs, O.OrthMass("tam", tam=bad))
    assert o.computed and o.info["fe_offdiagonal_max"] > O.TAM_MAX_FE_OFFDIAG
    assert "does not reproduce the orthogonality" in o.hold and "6.3.2 b" not in o.hold and any("does not reproduce" in w for w in o.warnings)
    m2 = C.modal_correlation(np.arange(1, 5) * 10.0, phi[:, :4], np.arange(1, 5) * 10.0, phi[:, :4], dofs, orth=o)
    r2 = C.evaluate(m2, [], _limits())
    assert {r.status for r in r2 if r.metric == "COC diagonal"} == {C.UNCONFIRMED} and all("does not reproduce" in r.note for r in r2 if r.metric == "COC diagonal")
    tam_ok = O.MassMatrix(dofs, 2.0 * np.eye(ND), "ok.csv")                            # the lumped mass of the chain: the FE modes are orthogonal through it
    assert O.orthogonality(phi[:, :4], phi[:, :4], dofs, O.OrthMass("tam", tam=tam_ok)).hold == ""


# ------------------------------------------------------------------ MEDIUM 7: an old stored result
def test_a_result_stored_before_the_orthogonality_existed_says_so():
    import test_space_correlation as T
    from fea_core.space import corr_run as CR
    mesh = T._box_mesh()
    ms = T._modes(mesh)
    fe = (mesh, ms)
    st = T._study_with_limits(T.EXAMPLE)
    td = T._test_data(fe)
    res = T._results(fe)
    CR.run_correlation(st, res, td, sensors=list(T.SENSORS))
    assert "orth_mass" in CR.state_of(st)["result"]["provenance"]
    prov = CR.state_of(st)["result"]["provenance"]
    del prov["orth_mass"]                                                              # what a build before COC / AOC stored
    v = CR.verify_stored(st, res, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "stored before the cross-orthogonality" in v["reasons"][0] and "corr_coc_diag_min" in v["reasons"][0]


# ------------------------------------------------------------------ LOW: the TAM is bound to the study; `clear` drops it
def test_a_tam_is_bound_to_the_study_it_was_imported_for():
    from fea_core.space import corr_run as CR
    from verification.space_cases import block_study
    a, b = block_study(), block_study()
    b.name = "another model"
    tam = O.MassMatrix([("A", "x")], np.eye(1), "m.csv")
    CR.remember_tam(a, tam)
    assert tam.study_key == CR.study_key(a) != CR.study_key(b)
    opt = C.CorrelationOptions(orth_mass="tam")
    assert CR.build_orth_mass(a, None, None, opt, tam).kind == "tam"
    other = CR.build_orth_mass(b, None, None, opt, tam)
    assert other.kind == "none" and "another study" in other.why
    CR.remember_tam(b, tam)                                                            # a second study must not take it over silently: the key stays the first one
    assert tam.study_key == CR.study_key(a) and CR.build_orth_mass(b, None, None, opt).kind == "none"


def test_clear_drops_the_imported_mass_matrix_too(tmp_path):
    pytest.importorskip("mcp")
    import test_space_correlation as T
    from fea_core.mcp.session import Session
    from verification.space_cases import block_study
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    uff = tmp_path / "ok.uff"
    mesh = T._box_mesh()
    ms = T._modes(mesh)
    uff.write_text(C.write_uff55(T._test_data((mesh, ms)).modes))
    tam = tmp_path / "m.csv"
    tam.write_text("# unit: kg\n,A:x,A:y\nA:x,2,0\nA:y,0,2\n")
    out = T._mcp(sess, [("fea_space_test_import", {"path": str(uff), "mass_matrix": str(tam)}, False), ("fea_space_test_import", {}, False),
                        ("fea_space_test_import", {"clear": True}, False), ("fea_space_test_import", {}, False)])
    assert out[0]["mass_matrix"]["dofs"] == 2 and out[1]["mass_matrix"]["dofs"] == 2
    assert out[2]["mass_matrix"] is None and out[3]["mass_matrix"] is None
    from fea_core.space import corr_run as CR
    assert CR.recall_tam(sess.study) is None


# ------------------------------------------------------------------ LOW: the AOC notes, the NaN above 1
def test_the_aoc_note_of_a_complete_test_mode_names_the_incomplete_one():
    phi, _mass, g, _d, dofs = _setup()
    t = phi[:, :4].astype(complex).copy()
    t[3, 2] = np.nan                                                                   # test mode 3 lacks a DOF
    orth = O.orthogonality(t, phi, dofs, O.OrthMass("serep", gram=g))
    modal = C.modal_correlation(np.arange(1, 5) * 10.0, t, np.arange(1, NM + 1) * 10.0, phi, dofs, orth=orth)
    notes = {r.item: r.note for r in C.evaluate(modal, [], _limits()) if r.metric == "AOC off-diagonal"}
    assert "because test mode(s) 3 have a DOF that was not measured" in notes["test mode 1"] and notes["test mode 1"].startswith("the AOC of test mode 1 is not given")
    assert "no AOC for test mode 3: it has a DOF that was not measured" == notes["test mode 3"]


def test_a_coc_above_one_beyond_rounding_is_not_a_number_not_a_clipped_one():
    num = np.array([[1.0 + 1e-9, 1.0 + 1e-3, 0.5]])
    out = O._normalised(num, np.array([1.0]), np.ones(3))
    assert out[0, 0] == 1.0 and math.isnan(out[0, 1]) and out[0, 2] == 0.5

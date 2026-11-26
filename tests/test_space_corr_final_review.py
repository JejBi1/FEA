"""Final review of the COC / AOC (F14-S5, 25. 11. 2026, two independent reviews): the noise floor of the printed digits of a TAM file (a singular matrix printed to 6 - 8 digits is never read as
full rank), no exemption for a stored SEREP record that declares itself not indicative, the report of an earlier-build SEREP record and of N/A criteria, the TAM bound to the state of the model at
the correlation, an FE mode that the TAM does not see, the generalised masses of the FE modes through the TAM, the axes statement of a new matrix, the Czech of the composed notes."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from fea_core import i18n
from fea_core.eigen import ModeSet
from fea_core.mesh import Mesh
from fea_core.results import Results
from fea_core.space import corr_orth as O
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from fea_core.space.model import LimitEntry, SpaceSettings
from fea_core.space.requirements import confirm
from test_space_corr_orthogonality import _limits
from verification import cases as _cases  # noqa: F401  (import order of the verification package)
from verification.space_cases import block_study, make_requirement_set
from verification.space_corr_cases import _generalised_chain

ALL_LIMITS = {"corr_mac_min": 0.9, "corr_mac_offdiag_max": 0.3, "corr_comac_min": 0.9, "corr_freq_error_max": 0.05, "corr_coc_diag_min": 0.9, "corr_coc_offdiag_max": 0.1,
              "corr_aoc_offdiag_max": 0.1, "corr_modulus_diff_max": 0.2}


def _csv(dofs, m, fmt):
    return ("# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in dofs) + "\n"
            + "\n".join(f"{dofs[i][0]}:{dofs[i][1]}," + ",".join(fmt % float(v) for v in m[i]) for i in range(len(dofs))) + "\n")


def _serep_like(rng, n, m, spread=2.0):
    fe = rng.standard_normal((n, m))
    pin = np.linalg.pinv(fe)
    return fe, pin.T @ np.diag(10.0 ** rng.uniform(0, spread, m)) @ pin


# ------------------------------------------------------------------ 1: the noise floor of the printed digits
@pytest.mark.parametrize("n, m", [(5, 4), (6, 4), (8, 4), (31, 30)], ids=["5 DOFs / 4 modes (1 null)", "6 / 4 (2 null)", "8 / 4 (4 null)", "31 / 30"])
@pytest.mark.parametrize("fmt", ["%.6g", "%.7g", "%.8g"])
def test_a_singular_tam_printed_to_few_digits_is_never_read_as_full_rank(n, m, fmt):
    """The reviewer's case: a rank-m SEREP matrix printed to 6 - 8 digits has null-space eigenvalues of both signs or none; the band must stand above the rounding either way. Before: 139 of 300
    matrices of 5 DOFs / 4 modes at %.6g read FULL rank (the null-space hold could not fire, a false PASS)."""
    rng = np.random.default_rng(1)
    dofs = [(f"N{i}", "x") for i in range(n)]
    ranks = []
    for _ in range(25 if n < 20 else 8):
        _fe, mm = _serep_like(rng, n, m)
        ranks.append(O.parse_mass_csv(_csv(dofs, mm, fmt)).rank)
    assert max(ranks) <= m, ranks                                                    # a null direction is never mass
    assert sum(r == m for r in ranks) >= 0.8 * len(ranks), ranks                     # and the true rank is read (a genuine eigenvalue below the rounding may read lower, never higher)


def test_the_reviewers_chain_with_a_payload_is_not_a_false_pass_with_a_six_digit_tam():
    """6 masses (one heavy), rank 5 of 6 SEREP TAM printed with %.6g: the null-space hold fires, the verdict is not PASS (before: 41 rows PASS, verdict PASS, true COC off-diagonal 0.244)."""
    masses = [1.0, 1.0, 30.0, 1.0, 1.0, 1.0]
    phi, m = _generalised_chain(masses)
    dofs = [(f"N{i}", "x") for i in range(6)]
    fe = phi[:, :5]
    pin = np.linalg.pinv(fe)
    serep = pin.T @ (fe.T @ m @ fe) @ pin
    qq, _ = np.linalg.qr(fe)
    nvec = np.linalg.svd(np.eye(6) - qq @ qq.T)[0][:, 0]
    test = fe.copy()
    test[:, 3] = fe[:, 3] + 0.29 * np.linalg.norm(fe[:, 3]) * nvec
    full = abs(test[:, 3] @ m @ fe[:, 0]) / math.sqrt((test[:, 3] @ m @ test[:, 3]) * (fe[:, 0] @ m @ fe[:, 0]))
    assert full > 0.2                                                                # with the true mass the shape is far from orthogonal to FE mode 1
    lim = _limits({"corr_coc_diag_min": 0.9, "corr_coc_offdiag_max": 0.1, "corr_aoc_offdiag_max": 0.1, "corr_mac_min": 0.9, "corr_comac_min": 0.5, "corr_mac_offdiag_max": 0.6})
    for fmt in ("%.6g", "%.7g", "%.17g"):
        tam = O.parse_mass_csv(_csv(dofs, serep, fmt), "lab_serep.csv")
        assert tam.rank == 5, fmt
        orth = O.orthogonality(test, fe, dofs, O.OrthMass("tam", tam=tam, gen_mass=np.diag(fe.T @ m @ fe).copy()))
        assert orth.mode_hold and 3 in orth.mode_hold, fmt                           # test mode 4 has its share in the null space
        modal = C.modal_correlation(np.arange(1, 6) * 10.0, test, np.arange(1, 6) * 10.0, fe, dofs, orth=orth)
        rows = C.evaluate(modal, [], lim)
        assert C.verdict_of(rows) != C.PASS and any(r.status == C.UNCONFIRMED and r.metric.startswith(("COC", "AOC")) for r in rows), fmt


def test_full_rank_matrices_with_a_wide_mass_spread_keep_reading_full_rank_at_six_digits():
    n = 10
    dofs = [(f"N{i}", "x") for i in range(n)]
    rng = np.random.default_rng(4)
    for k in (3, 4.3, 5):                                                            # eigenvalue spread up to 1e5: still well above the rounding of 6 digits
        for _ in range(20):
            q, _r = np.linalg.qr(rng.standard_normal((n, n)))
            mm = (q * np.logspace(0, -k, n)) @ q.T
            assert O.parse_mass_csv(_csv(dofs, mm, "%.6g")).rank == n, k
    mm = np.diag(np.logspace(0, -4, n))                                              # a diagonal matrix with a large spread keeps full rank too
    assert O.parse_mass_csv(_csv(dofs, mm, "%.6g")).rank == n


def test_hand_written_round_numbers_are_exact_and_the_rounding_bound_follows_the_printed_digits():
    dofs = [("A", "x"), ("B", "x"), ("C", "x")]
    hand = "# unit: kg\n,A:x,B:x,C:x\nA:x,1,0,0\nB:x,0,1000,0\nC:x,0,0,2\n"
    t = O.parse_mass_csv(hand)
    assert t.noise == 0.0 and t.rank == 3                                            # at most three significant digits: taken as exact (a spread 1e3 must not read as null space)
    m = np.diag([1.0, 2.0, 3.0]) + 0.123456
    fixed = O.parse_mass_csv(_csv(dofs, m, "%.6f"))
    assert fixed.noise == pytest.approx(0.5e-6 * 3, rel=1e-9)                        # fixed point: half a unit of the 6th decimal in every cell (F norm of a 3 x 3 matrix)
    g6 = O.parse_mass_csv(_csv(dofs, m, "%.6g"))
    assert 0 < g6.noise < 1e-5 * float(np.abs(m).max()) * 3
    g9 = O.parse_mass_csv(_csv(dofs, m, "%.9e"))
    assert g9.noise < g6.noise / 100 and g9.rank == 3
    b, r = O.mass_rank_band(np.array([-2e-3, 1e-3, 1.0]), noise=5e-3)               # a negative eigenvalue within the noise is the rounding of the digits
    assert r == 1 and b >= 5e-3
    with pytest.raises(ValueError, match="negative eigenvalue"):
        O.mass_rank_band(np.array([-2e-2, 1e-3, 1.0]), noise=5e-3)                    # beyond the noise: no mass matrix
    with pytest.raises(ValueError, match="digits"):
        O.mass_rank_band(np.array([1e-3, 2e-3, 3e-3]), noise=1.0)                    # the digits do not resolve any mass


# ------------------------------------------------------------------ 5: an FE mode the TAM does not see
def test_an_fe_mode_invisible_at_the_sensors_holds_the_rows_without_a_warning():
    dofs = [(f"N{i}", "x") for i in range(6)]
    phi, _ = _generalised_chain([1.0] * 6)
    tam = O.MassMatrix(dofs, np.eye(6), "m.csv", "kg")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                               # no RuntimeWarning (All-NaN slice, invalid value)
        zero = O.orthogonality(phi[:, :3], np.zeros((6, 3)), dofs, O.OrthMass("tam", tam=tam))
        one = phi[:, :3].copy()
        one[:, 2] = 0.0
        part = O.orthogonality(phi[:, :2], one, dofs, O.OrthMass("tam", tam=tam))
    assert zero.info["fe_offdiagonal_max"] is None and zero.info["fe_diagonal_min"] is None                 # the None branch is reachable now (the diagonal is masked, not zeroed)
    assert "invisible" in zero.hold and "1, 2, 3" in zero.hold
    assert part.info["fe_offdiagonal_max"] is not None and part.info["fe_offdiagonal_max"] < 1e-9 and "FE mode(s) 3 are invisible" in part.hold
    fe = phi[:, :3]
    nrm = np.diag(1.0 / np.sqrt(np.diag(fe.T @ fe)))
    assert O.orthogonality(fe @ nrm, fe @ nrm, dofs, O.OrthMass("tam", tam=tam)).hold == ""        # a healthy matrix is not held (the unit mass: the chain modes are orthogonal)


# ------------------------------------------------------------------ 6: the generalised masses of the FE modes through the TAM
def _heavy_chain(band):
    ph, mh = _generalised_chain([1.0] * 6 + [1000.0] * 2)
    return ph[:, band], mh, [(f"N{i}", "x") for i in range(8)]


def test_a_tam_that_leaves_out_a_heavy_payload_is_held_by_the_generalised_masses_when_the_modes_move_it():
    """The reviewer's T1a, with the compared modes that move the payload (the lowest four): the bare structure (the payload of 1000 left out) as a SEREP matrix reproduces the orthogonality of
    the FE modes exactly and would give COC 1 to a shape with a 10 % error at the payload; its generalised masses are those of another model."""
    fe, mh, dofs = _heavy_chain([0, 1, 2, 3])
    gen = np.diag(fe.T @ mh @ fe).copy()
    pin = np.linalg.pinv(fe)
    bare = O.MassMatrix(dofs, pin.T @ np.diag(np.diag(fe.T @ fe)) @ pin, "bare.csv", "kg")
    test = fe.copy()
    test[7, 0] += 0.10 * np.abs(fe[:, 0]).max()
    o = O.orthogonality(test, fe, dofs, O.OrthMass("tam", tam=bare, gen_mass=gen))
    assert o.info["fe_offdiagonal_max"] < 1e-9                                       # the orthogonality alone is fooled
    assert o.info["genmass_checked"] and o.info["genmass_dev_max"] > 0.5 and "generalised masses" in o.hold
    modal = C.modal_correlation(np.array([10.0, 20.0, 30.0, 40.0]), test, np.array([10.0, 20.0, 30.0, 40.0]), fe, dofs, orth=o)
    rows = C.evaluate(modal, [], _limits({"corr_coc_diag_min": 0.9, "corr_modulus_diff_max": 1.0}))
    assert all(r.status == C.UNCONFIRMED for r in rows if r.metric.startswith(("COC", "AOC"))) and C.verdict_of(rows) != C.PASS
    good = O.orthogonality(fe, fe, dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, mh, "true.csv", "kg"), gen_mass=gen))
    assert good.hold == "" and good.info["genmass_dev_max"] < 1e-9


def test_the_generalised_masses_cannot_see_a_payload_that_the_compared_modes_hardly_move():
    """DOCUMENTED LIMIT (help and report say so): the upper modes of the heavy-payload chain leave the payload at rest, so the bare structure reproduces their orthogonality AND their generalised
    masses (to 1e-3) - it is the TAM's mass fidelity for other shapes that no check of the program can verify, and the laboratory is responsible for it."""
    fe, mh, dofs = _heavy_chain([2, 3, 4, 5])
    gen = np.diag(fe.T @ mh @ fe).copy()
    test = fe.copy()
    test[7, 0] += 0.10 * np.abs(fe[:, 0]).max()
    true = abs(test[:, 0] @ mh @ fe[:, 0]) / math.sqrt((test[:, 0] @ mh @ test[:, 0]) * (fe[:, 0] @ mh @ fe[:, 0]))
    o = O.orthogonality(test, fe, dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, np.eye(8), "bare.csv", "kg"), gen_mass=gen))
    assert o.hold == "" and o.info["genmass_dev_max"] < 0.01 and o.coc[0, 0] > true + 0.3


def test_rescaled_modal_masses_are_held_but_one_global_factor_and_a_few_percent_are_not():
    p12, m12 = _generalised_chain([1.0] * 12)
    dofs = [(f"N{i}", "x") for i in range(12)]
    fe = p12[:, :4]
    g = fe.T @ m12 @ fe
    pin = np.linalg.pinv(fe)
    q, _ = np.linalg.qr(fe)
    null = np.eye(12) - q @ q.T

    def run(scale, unit=1.0):
        mt = unit * (pin.T @ (np.diag(scale) @ g) @ pin + 0.05 * null)
        return O.orthogonality(fe, fe, dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, mt, "t.csv", "t", rank=12), gen_mass=np.diag(g).copy()))
    bad = run(np.array([100.0, 1.0, 1.0, 1.0]))                                      # the reviewer's T1c
    assert "generalised masses" in bad.hold and bad.info["genmass_dev_max"] > 1.0
    assert run(np.array([1.0, 1.0, 1.0, 1.0]), unit=1000.0).hold == ""              # t instead of kg: one global factor
    ok = run(np.array([1.0, 1.04, 0.97, 1.03]))
    assert ok.hold == "" and 0.0 < ok.info["genmass_dev_max"] < O.TAM_MAX_GENMASS_DEV
    assert "generalised masses" in run(np.array([1.0, 1.25, 1.0, 1.0])).hold         # 25 % in one mode: above the 10 % tolerance


def test_without_a_study_the_generalised_masses_are_not_compared_and_it_says_so_and_with_a_study_that_cannot_be_formed_the_rows_are_held():
    fe, mh, dofs = _heavy_chain([2, 3, 4, 5])
    t = O.MassMatrix(dofs, mh, "true.csv", "kg")
    o = O.orthogonality(fe, fe, dofs, O.OrthMass("tam", tam=t))
    assert o.hold == "" and o.info["genmass_checked"] is False and any("NOT compared with the study" in n for n in o.notes)
    st = block_study()
    tet4 = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]), np.array([[0, 1, 2, 3]]), np.zeros(1, int), ["Block"])
    ms = ModeSet("modal", "m", [1.0, 2.0, 3.0, 4.0], np.zeros((4, 4, 3), dtype=np.float32))
    o2 = O.orthogonality(fe, fe, dofs, O.OrthMass("tam", tam=t, study=st, mesh=tet4), ms, [0, 1, 2, 3])
    assert "could not be computed" in o2.hold and "generalised masses" in o2.hold


# ------------------------------------------------------------------ a small tet10 model: the pipeline with a TAM of the study's own mass
def _limits_space(vals, na=()):
    rs = make_requirement_set("protoflight", confirmed=False)
    for k, v in vals.items():
        if k in na:
            rs.limits.append(LimitEntry(key=k, value=None, unit=C.LIMIT_INFO[k][0], document="doc", revision="rev", clause="c1", not_applicable=True, set_via="gui"))
        else:
            rs.limits.append(LimitEntry(key=k, value=v, unit=C.LIMIT_INFO[k][0], document="doc", revision="rev", clause="c1"))
    sp = SpaceSettings(requirement_set=rs)
    confirm(rs, "reviewer", "2026-11-25", via="gui", scope=sp)
    return sp


def _tet10_bar(nx: int = 4):
    xs, ys, zs = np.linspace(0, 100.0, nx + 1), np.linspace(0, 20.0, 2), np.linspace(0, 10.0, 2)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], float)
    idx = lambda i, j, k: (i * 2 + j) * 2 + k          # noqa: E731
    t4 = []
    for i in range(nx):
        c = [idx(i + a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
        for t in ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)):
            t4.append([c[v] for v in t])
    nodes, mid, t10 = list(corners), {}, []
    for tet in t4:
        ms_ = []
        for a, b in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]:
            u, v = tet[a], tet[b]
            key = (min(u, v), max(u, v))
            if key not in mid:
                mid[key] = len(nodes)
                nodes.append(0.5 * (corners[u] + corners[v]))
            ms_.append(mid[key])
        t10.append(tet + ms_)
    mesh = Mesh(np.array(nodes, float), np.array(t10, np.int64), np.zeros(len(t10), np.int64), ["Block"])
    st = block_study()
    from fea_core.dynamics.mass import mass_times
    x = mesh.nodes[:, 0] / 100.0
    raw = np.zeros((4, len(mesh.nodes), 3))
    raw[0, :, 1], raw[1, :, 2] = np.sin(np.pi * x / 2), np.sin(np.pi * x / 2)
    raw[2, :, 1], raw[3, :, 2] = np.sin(3 * np.pi * x / 2), np.sin(3 * np.pi * x / 2)
    done: list[np.ndarray] = []
    for k in range(4):
        v = raw[k].copy()
        for u in done:
            mu = mass_times(st, mesh, u)
            v -= (np.sum(v * mu) / np.sum(u * mu)) * u
        done.append(v / np.abs(v).max())
    ms = ModeSet("modal", "modes", [10.0, 20.0, 60.0, 80.0], np.stack(done).astype(np.float32))
    sensors = C.sensors_from_dicts([{"node": f"S{k}", "point": [15.0 + 14.0 * k, 10.0, 5.0], "dofs": "yz"} for k in range(6)])
    return st, mesh, ms, sensors


def _tam_of_study(st, mesh, ms, sensors, rescale=None):
    dofs = [(s.node, d) for s in sensors for d in s.dofs]
    fphi, _ = C.fe_shapes_at_sensors(ms, mesh, sensors, dofs)
    g = O.serep_gram(st, mesh, ms, [0, 1, 2, 3])
    if rescale is not None:
        g = np.diag(rescale) @ g
    pin = np.linalg.pinv(fphi)
    q, _ = np.linalg.qr(fphi)
    return O.MassMatrix(dofs, pin.T @ g @ pin + 1e-3 * float(np.diag(g).max()) * (np.eye(len(dofs)) - q @ q.T), "fit_tam.csv", "t", rank=len(dofs))


def _run_tam(tam, opt=None, vals=None, na=()):
    st, mesh, ms, sensors = _tet10_bar()
    st.space = _limits_space(vals or {**ALL_LIMITS, "corr_coc_offdiag_max": 0.99, "corr_aoc_offdiag_max": 0.99}, na)
    results = Results(mesh=mesh, steps=[], modes=[ms])
    td = C.fe_test_data(ms, mesh, sensors, damping=0.01)
    matrix = tam(st, mesh, ms, sensors) if callable(tam) else tam
    r, _ = CR.run_correlation(st, results, td, sensors=sensors, options=opt or C.CorrelationOptions(orth_mass="tam"), tam=matrix)
    st.keepalive = (td, matrix)                                                       # the session (GUI window / MCP session) holds the test data and the matrix: the study keeps weak references only
    return st, mesh, ms, results, td, r


def test_the_tam_of_the_study_is_a_pass_and_after_a_density_edit_the_stored_result_is_outdated():
    """The reviewer's T2: the density multiplied by 5 after the correlation kept the stored PASS current (the fingerprint of the model was compared only in the live run)."""
    st, mesh, ms, results, td, r = _run_tam(_tam_of_study)
    assert r.verdict == C.PASS and r.modal.orth.hold == "" and r.modal.orth.info["genmass_checked"] and r.modal.orth.info["genmass_dev_max"] < 1e-6
    prov = CR.state_of(st)["result"]["provenance"]["orth_mass"]
    assert prov["source"] == "tam" and prov["model_fingerprint"] == C.model_fingerprint(st, mesh) != ""
    v = CR.verify_stored(st, results, td)
    assert v["verdict"] == C.PASS and v["current"] and "model state of the mass matrix (TAM)" in v["verified"]
    next(iter(st.materials.values())).rho *= 5.0
    v = CR.verify_stored(st, results, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and not v["current"] and any("no longer the one the imported mass matrix (TAM) was bound to" in x for x in v["reasons"])
    st.materials[next(iter(st.materials))].rho /= 5.0
    assert CR.verify_stored(st, results, td)["verdict"] == C.PASS                     # back to the state it was bound to
    CR.state_of(st)["result"]["provenance"]["orth_mass"].pop("model_fingerprint")      # a record that does not say which state of the model it was bound to (an earlier build)
    v = CR.verify_stored(st, results, td)
    assert v["verdict"] != C.PASS and any("does not say which state of the model" in x for x in v["reasons"])


def test_a_tam_with_rescaled_modal_masses_is_held_through_the_pipeline():
    _st, _mesh, _ms, _results, _td, r = _run_tam(lambda st, mesh, ms, sensors: _tam_of_study(st, mesh, ms, sensors, rescale=np.array([1.0, 1.5, 1.0, 1.0])))
    o = r.modal.orth
    assert o.computed and "generalised masses" in o.hold and r.verdict != C.PASS
    assert all(x.status == C.UNCONFIRMED for x in r.rows if x.metric.startswith(("COC", "AOC")))


def test_a_new_matrix_does_not_inherit_the_axes_statement_of_an_earlier_one():
    """A run without options takes the stored options, but `tam_global_axes = True` was stated for the OLD matrix: a newly given one must be stated again."""
    st, mesh, ms, results, td, _r = _run_tam(_tam_of_study, C.CorrelationOptions(orth_mass="tam", tam_global_axes=True, tam_global_axes_via="gui"))
    assert CR.load_options(st).tam_global_axes is True
    sensors = _tet10_bar()[3]
    r2, _ = CR.run_correlation(st, results, td, sensors=sensors, tam=_tam_of_study(st, mesh, ms, sensors))
    assert r2.options.tam_global_axes is False and r2.options.tam_global_axes_via == "" and CR.load_options(st).tam_global_axes is False
    r3, _ = CR.run_correlation(st, results, td, sensors=sensors, options=C.CorrelationOptions(orth_mass="tam", tam_global_axes=True, tam_global_axes_via="gui"), tam=_tam_of_study(st, mesh, ms, sensors))
    assert r3.options.tam_global_axes is True                                         # explicit options are the caller's statement
    r4, _ = CR.run_correlation(st, results, td, sensors=sensors)                      # no new matrix: the stored statement stays (it is about the stored matrix)
    assert r4.options.tam_global_axes is True


def test_a_copy_of_the_study_inherits_the_session_tam_with_the_test_data():
    from fea_core.study import Study
    st, _mesh, _ms, results, td, r = _run_tam(_tam_of_study)
    assert r.verdict == C.PASS
    cp = Study.from_dict(st.to_dict())
    assert CR.verify_stored(cp, results, td)["verdict"] == C.VERDICT_UNVERIFIABLE      # the copy knows neither the test data nor the matrix
    assert CR.transfer_test_data(st, cp) is True
    assert CR.recall_tam(cp) is CR.recall_tam(st) and CR.verify_stored(cp, results)["verdict"] == C.PASS


# ------------------------------------------------------------------ 3, 7: the report
def _doc_text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def _serep_record(vals, na=()):
    st, mesh, ms, sensors = _tet10_bar()
    st.space = _limits_space(vals, na)
    results = Results(mesh=mesh, steps=[], modes=[ms])
    td = C.fe_test_data(ms, mesh, sensors, damping=0.01)
    CR.run_correlation(st, results, td, sensors=sensors, options=C.CorrelationOptions(orth_mass="serep"))
    return st, results, td


def test_the_report_of_an_earlier_build_serep_record_prints_the_indicative_notice_and_shows_its_coc_rows_unconfirmed():
    pytest.importorskip("docx")
    from fea_core.report import space_corr as RC
    from fea_core.report.word import _Doc
    st, results, td = _serep_record(ALL_LIMITS)
    res = CR.state_of(st)["result"]
    om = res["provenance"]["orth_mass"]
    om.pop("indicative")                                                              # an earlier build wrote no flag ...
    res["modal"]["orthogonality"]["indicative"] = ""
    for row in res["rows"]:
        if row["metric"].startswith(("COC", "AOC")):
            row["status"] = C.PASS                                                    # ... and judged the SEREP rows
    D = _Doc()
    RC.correlation_section(D, st, "1", results)
    text = _doc_text(D)
    assert "INDICATIVE ONLY" in text and "Verdict of the test correlation: OUTDATED" in text
    crit = next(t for t in D.doc.tables if [c.text for c in t.rows[0].cells][:2] == ["Item", "Metric"])
    status = {(r.cells[0].text, r.cells[1].text): r.cells[4].text for r in crit.rows[1:] if r.cells[1].text.startswith(("COC", "AOC"))}
    assert status and set(status.values()) == {"UNCONFIRMED"}                         # never PASS in the table
    assert "stored as PASS / FAIL by an earlier version" in text


def test_when_every_coc_aoc_criterion_is_declared_not_applicable_the_notice_does_not_say_unconfirmed():
    pytest.importorskip("docx")
    from fea_core.report import space_corr as RC
    from fea_core.report.word import _Doc
    na = ("corr_coc_diag_min", "corr_coc_offdiag_max", "corr_aoc_offdiag_max")
    st, results, td = _serep_record(ALL_LIMITS, na)
    D = _Doc()
    RC.correlation_section(D, st, "1", results)
    text = _doc_text(D)
    assert "INDICATIVE ONLY" in text and "every COC / AOC row is UNCONFIRMED" not in text and "no COC / AOC row is judged" in text
    st2, results2, _td2 = _serep_record(ALL_LIMITS, na[:1])                           # only one declared N/A: the other COC / AOC rows are UNCONFIRMED
    D2 = _Doc()
    RC.correlation_section(D2, st2, "1", results2)
    assert "every COC / AOC row is UNCONFIRMED" in _doc_text(D2)


def test_the_report_prints_who_stated_the_axes_of_the_tam_and_what_is_not_checked():
    pytest.importorskip("docx")
    from fea_core.report import space_corr as RC
    from fea_core.report.word import _Doc
    st, mesh, ms, results, td, r = _run_tam(_tam_of_study, C.CorrelationOptions(orth_mass="tam", tam_global_axes=True, tam_global_axes_via="mcp"))
    assert r.verdict == C.PASS
    D = _Doc()
    RC.correlation_section(D, st, "1", results)
    text = _doc_text(D)
    assert "Imported TAM in the global axes of the FE model" in text and "stated by MCP interface (an agent)" in text
    assert "6.3.2 b" in text and "needs the stiffness matrix of the TAM and is NOT done" in text and "responsibility of the test laboratory" in text
    assert "generalised masses of the FE modes through this matrix" in text
    st2, _m2, _s2, results2, _t2, r2 = _run_tam(_tam_of_study, C.CorrelationOptions(orth_mass="tam"))
    D2 = _Doc()
    RC.correlation_section(D2, st2, "1", results2)
    assert "NOT stated" in _doc_text(D2) and r2.verdict == C.PASS


def test_the_docstring_attributes_the_reduced_fem_quote_to_6_2_2_b():
    assert "6.2.2 b" in O.__doc__ and "The reduced FEM shall consist of a mass and stiffness matrix" in O.__doc__ and "6.2 / 6.3" not in O.__doc__


# ------------------------------------------------------------------ 8: Czech
@pytest.fixture
def czech():
    cat = i18n.CATALOG_CS
    i18n.use_catalog(__import__("json").loads(cat.read_text(encoding="utf-8")))
    i18n.set_language("cs")
    yield
    i18n.set_language("en")
    i18n.reload()


def test_the_composed_notes_are_translated_whole_into_czech(czech):
    note = C.INDICATIVE_REASON + "; " + C.INDICATIVE_OUTSIDE_NOTE
    cz = i18n.tr(note)
    assert "orientační" in cz and "warning hint" not in cz and "does not fail" not in cz and "SEREP reduction" not in cz
    for text in ("Reduced mass: {0}".format("SEREP reduction of the FE modes (mass of the study), indicative") + "\nRows UNCONFIRMED: {0}".format(C.INDICATIVE_REASON),
                 "Reduced mass: {0}".format("imported mass matrix lab.csv"),
                 "Cross-orthogonality not computed: {0}".format("no mass for the orthogonality checks was given")):
        cz = i18n.tr(text)
        assert "Reduced mass" not in cz and "Rows UNCONFIRMED" not in cz and "orthogonality not computed" not in cz and "indicative" not in cz, cz
    assert i18n.tr(C.INDICATIVE_OUTSIDE_NOTE) != C.INDICATIVE_OUTSIDE_NOTE

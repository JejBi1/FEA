"""Verification of the cross-orthogonality of the test correlation (F14 stage S5, `fea_core.space.corr_orth`, ECSS-E-ST-32-11C 3.2.39, Table 5-2, 5.8.1.3 b) against hand calculations.

S5A  mass-weighted cross-orthogonality COC, auto-orthogonality AOC and modulus difference on lumped-mass chains with closed-form modes:
     * a fixed-free chain of 8 equal masses (modes sin((2r-1) j pi / (2n+1))), the first 6 modes compared: a model against itself gives COC = AOC = identity; every DOF measured, the
       reduced mass of the SEREP route (Gram matrix of the modes) and the full mass matrix of the TAM route give the same matrices;
     * a two-mass chain by hand (roots of m1 m2 l^2 - k (m1 + 2 m2) l + k^2 = 0, no eigen-solver): a test article with a 20 % heavier tip mass against the model - the off-diagonal COC
       terms and the AOC of the test modes by the hand formula |phi_t^T M phi_f| / sqrt((phi_t^T M phi_t)(phi_f^T M phi_f));
     * fewer DOFs than the chain has: SEREP reduction equals the full-DOF orthogonality of the shape expanded through the FE modes (reduction and expansion agree);
     * the modulus difference of two three-component shapes by hand;
     * the consistent mass of one quadratic tetrahedron (rho V) from the generalised mass of the SEREP reduction (`serep_gram`);
     * the SEREP COC is that of the PROJECTION onto the compared FE modes and is INDICATIVE only (owner decision 25. 11. 2026): a test shape = FE mode 1 + 0.75 |phi_1| r with r orthogonal to the
       span has the out-of-span residual 0.75^2 / (1 + 0.75^2) = 0.36 by hand and a high COC of its projection - its row is UNCONFIRMED (never PASS, never FAIL); 0.8 phi_1 + 0.6 phi_5 against the
       modes 1 and 2 only has COC 0.9989, against all the modes the true COC 0.8, both UNCONFIRMED; one FE mode is refused;
     * the LIMITATION of the Euclidean residual and of the projection: 8 masses, two of 1000 at the free end, the 4 upper modes (the heavy tip hardly moves), the test tip 10 % off: the Euclidean
       residual is 0.0027 and the SEREP COC 1.00 (it would pass any limit), while the COC of the full mass matrix is 0.52 - the SEREP row is UNCONFIRMED, not PASS (the false PASS of the review);
       with masses of 10 and a tip 2 % off the SEREP COC is close to the full-mass one and the row is UNCONFIRMED all the same;
     * the reviewer's real-model case in closed form (12 masses, two of 1000 at the tip, FE modes 3 - 8 compared): a test mode 0.8 phi_4 + 0.6 phi_1 has the true COC 0.80, the projection onto the
       window gives > 0.99 (the false PASS of the review) - the SEREP row is UNCONFIRMED, the TAM with the full mass matrix gives the true 0.80;
     * a SEREP matrix printed with %.5e / %.6e / %.7e loads as a singular PSD matrix of rank 4, a full-rank diagonal matrix with a mass spread of 2e4 loads as full rank, a matrix with a negative
       eigenvalue beyond the noise band is refused;
     * a singular TAM: the rank-4 matrix of the SEREP reduction of a 12-mass chain, a test mode = FE mode 1 + 0.75 |phi_1| r with r in its null space: the share in the null space
       0.75^2 / (1 + 0.75^2) by hand, rows UNCONFIRMED; one FE mode through a rank-1 matrix: UNCONFIRMED (the fit cannot be checked);
     * the verdict: no reduced mass -> UNCONFIRMED_REQUIREMENTS (never PASS), SEREP -> UNCONFIRMED_REQUIREMENTS whatever the value, a TAM and passing limits -> PASS, a TAM and a limit just
       above / below the hand value of the diagonal COC -> FAIL / PASS.
The limits of the verdict checks are values of THIS case, not values of a standard.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.eigen import ModeSet
from fea_core.mesh import Mesh
from fea_core.space import corr_orth as O
from fea_core.space import correlation as C

from .cases import Case, Check
from .space_cases import _build, block_study

SPACE_CORR_CASES: dict[str, Case] = {}


class _Out(dict):
    """The run for the evaluation; `.warnings` for run_all."""

    @property
    def warnings(self) -> list[str]:
        return []


def _chain_modes(n: int) -> np.ndarray:
    j = np.arange(1, n + 1)
    return np.stack([np.sin((2 * r - 1) * j * math.pi / (2 * n + 1)) for r in range(1, n + 1)], axis=1)


def _generalised_chain(masses: list[float], k: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Modes (max-normalised columns, ascending frequency) and the mass matrix of a fixed-free spring-mass chain: the dense generalised eigenproblem K phi = lambda M phi (numpy eigh)."""
    n = len(masses)
    kk = np.zeros((n, n))
    for i in range(n):
        kk[i, i] += k + (k if i + 1 < n else 0.0)
        if i + 1 < n:
            kk[i, i + 1] = kk[i + 1, i] = -k
    mm = np.diag(np.asarray(masses, float))
    lm = np.diag(1.0 / np.sqrt(np.diag(mm)))
    _w, v = np.linalg.eigh(lm @ kk @ lm)
    phi = lm @ v
    return phi / np.abs(phi).max(axis=0), mm


def _two_mass(m1: float, m2: float, k: float = 1.0) -> np.ndarray:
    """Modes [1, r] of K = k [[2, -1], [-1, 1]], M = diag(m1, m2), from the closed-form roots of the characteristic quadratic (columns = modes)."""
    b = k * (m1 + 2.0 * m2)
    disc = math.sqrt(b * b - 4.0 * m1 * m2 * k * k)
    lam = [(b - disc) / (2.0 * m1 * m2), (b + disc) / (2.0 * m1 * m2)]
    return np.array([[1.0, (2.0 * k - lm * m1) / k] for lm in lam]).T


def _cross(a: np.ndarray, b: np.ndarray, m: list[float], i: int, j: int) -> float:
    """|a_i^T M b_j| / sqrt((a_i^T M a_i)(b_j^T M b_j)) of two-DOF shapes, plain arithmetic."""
    num = sum(a[d, i] * m[d] * b[d, j] for d in range(2))
    ea = sum(a[d, i] * m[d] * a[d, i] for d in range(2))
    eb = sum(b[d, j] * m[d] * b[d, j] for d in range(2))
    return abs(num) / math.sqrt(ea * eb)


def s5a_run(study, workdir: Path, parts) -> _Out:
    out = _Out()
    # (1) a chain of 8 masses, the first 6 modes, against itself, all DOFs measured
    nd, n, m = 8, 6, 2.0
    phi_all = _chain_modes(nd)
    phi = phi_all[:, :n]
    mass = m * np.eye(nd)
    dofs = [(f"N{i}", "x") for i in range(nd)]
    tam = O.OrthMass("tam", tam=O.MassMatrix(dofs, mass))
    gram = phi.T @ mass @ phi
    self_tam = O.orthogonality(phi, phi, dofs, tam)
    self_serep = O.orthogonality(phi, phi, dofs, O.OrthMass("serep", gram=gram))
    out["self_coc"] = float(np.abs(self_tam.coc - np.eye(n)).max())
    out["self_aoc"] = float(np.abs(self_tam.aoc - np.eye(n)).max())
    out["self_serep"] = float(np.abs(self_serep.coc - np.eye(n)).max())
    # (2) a perturbed test: every FE mode mixed with the others; SEREP (all DOFs) = TAM
    rng = np.random.default_rng(5)
    test = phi @ (np.eye(n) + 0.05 * rng.standard_normal((n, n)))
    a, b = O.orthogonality(test, phi, dofs, tam), O.orthogonality(test, phi, dofs, O.OrthMass("serep", gram=gram))
    out["serep_vs_tam"] = float(max(np.abs(a.coc - b.coc).max(), np.abs(a.aoc - b.aoc).max()))
    # (3) two masses, hand formula
    phi_f, phi_t = _two_mass(1.0, 1.0), _two_mass(1.0, 1.2)
    d2 = [("A", "x"), ("B", "x")]
    r2 = O.orthogonality(phi_t, phi_f, d2, O.OrthMass("tam", tam=O.MassMatrix(d2, np.eye(2))))
    out["coc01"], out["coc10"], out["coc00"] = float(r2.coc[0, 1]), float(r2.coc[1, 0]), float(r2.coc[0, 0])
    out["aoc01"] = float(r2.aoc[0, 1])
    out["hand"] = {"coc01": _cross(phi_t, phi_f, [1.0, 1.0], 0, 1), "coc10": _cross(phi_t, phi_f, [1.0, 1.0], 1, 0), "coc00": _cross(phi_t, phi_f, [1.0, 1.0], 0, 0),
                   "aoc01": _cross(phi_t, phi_t, [1.0, 1.0], 0, 1)}
    # (4) fewer DOFs: reduction = expansion for a shape in the span of the FE modes
    n8, keep = 8, [0, 2, 3, 5, 7]
    p8, m8 = _chain_modes(n8)[:, :3], 2.0 * np.eye(8)
    g3 = p8.T @ m8 @ p8
    t8 = p8 @ np.array([[1.0, 0.1, -0.05], [0.2, 1.0, 0.1], [-0.1, 0.15, 1.0]])
    red = O.orthogonality(t8[keep], p8[keep], [(f"N{i}", "x") for i in keep], O.OrthMass("serep", gram=g3))
    full_dofs = [(f"N{i}", "x") for i in range(n8)]
    full = O.orthogonality(t8, p8, full_dofs, O.OrthMass("tam", tam=O.MassMatrix(full_dofs, m8)))
    out["reduced_vs_full"] = float(max(np.abs(red.coc - full.coc).max(), np.abs(red.aoc - full.aoc).max()))
    # (5) the modulus difference by hand
    md, where = O.modulus_difference(np.array([[1.0], [2.0], [2.0]]), np.array([[1.0], [2.0], [1.0]]), [(0, 0)], [("a", "x"), ("b", "x"), ("c", "x")])[0]
    out["md"], out["md_dof"] = float(md or float("nan")), where
    # (6) the generalised mass of one quadratic tetrahedron
    x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([x, [(x[p] + x[q]) / 2 for p, q in mids]])
    mesh = Mesh(nodes, np.arange(10)[None, :], np.zeros(1, int), ["Block"])
    st = block_study()
    rho = st.materials[next(iter(st.materials))].rho
    shapes = np.zeros((2, 10, 3), np.float32)
    shapes[0, :, 0] = 1.0
    shapes[1, :, 1] = 1.0
    g = O.serep_gram(st, mesh, ModeSet("modal", "m", [1.0, 2.0], shapes), [0, 1])
    out["gram_x"], out["gram_y"], out["gram_xy"], out["mass_ref"] = float(g[0, 0]), float(g[1, 1]), float(abs(g[0, 1])), float(rho * 2 * 3 * 4 / 6)
    # (7) the verdict
    lim = C.CorrelationLimits()
    for k, v in {"corr_mac_min": 0.9, "corr_mac_offdiag_max": 0.3, "corr_comac_min": 0.9, "corr_freq_error_max": 0.05, "corr_coc_diag_min": 0.9, "corr_coc_offdiag_max": 0.1,
                 "corr_aoc_offdiag_max": 0.1, "corr_modulus_diff_max": 0.2}.items():
        lim.entries[k] = C.UsedLimit(v, "case", "-", "-", True, "")
    f = np.array([10.0 * (r + 1) for r in range(n)])

    def rows_of(shapes_t: np.ndarray, orth: O.OrthResult | None, limits: C.CorrelationLimits = lim, fe: np.ndarray = phi, fd=dofs) -> list[C.MetricRow]:
        modal = C.modal_correlation(f[:shapes_t.shape[1]], shapes_t, f[:fe.shape[1]], fe, fd, orth=orth)
        return C.evaluate(modal, [], limits)

    def verdict(shapes_t: np.ndarray, orth: O.OrthResult | None) -> str:
        return C.verdict_of(rows_of(shapes_t, orth))
    out["v_none"] = verdict(phi, None)
    out["v_pass"] = verdict(phi, self_tam)
    out["v_serep"] = verdict(phi, self_serep)
    mixed = phi @ (np.eye(n) + 0.3 * np.ones((n, n)))
    out["v_fail"] = verdict(mixed, O.orthogonality(mixed, phi, dofs, tam))
    serep_bad_rows = [r for r in rows_of(mixed, O.orthogonality(mixed, phi, dofs, O.OrthMass("serep", gram=gram))) if r.metric in ("COC diagonal", "COC off-diagonal", "AOC off-diagonal")]
    out["serep_bad_never_fail"] = float(all(r.status == C.UNCONFIRMED for r in serep_bad_rows) and any(r.indicative_outside for r in serep_bad_rows))
    # (7b) a TAM judges by the hand value: the diagonal COC of the two-mass chain against a limit just below / above it
    hand00 = _cross(phi_t, phi_f, [1.0, 1.0], 0, 0)
    tam2 = O.orthogonality(phi_t, phi_f, d2, O.OrthMass("tam", tam=O.MassMatrix(d2, np.eye(2))))
    for tag, lv in (("below", hand00 - 1e-3), ("above", hand00 + 1e-3)):
        lm = C.CorrelationLimits()
        for k, v in {"corr_mac_min": 0.0, "corr_mac_offdiag_max": 1.0, "corr_comac_min": 0.0, "corr_freq_error_max": 1.0, "corr_coc_diag_min": lv, "corr_coc_offdiag_max": 1.0,
                     "corr_aoc_offdiag_max": 1.0, "corr_modulus_diff_max": 1.0}.items():
            lm.entries[k] = C.UsedLimit(v, "case", "-", "-", True, "")
        modal2 = C.modal_correlation(np.array([10.0, 20.0]), phi_t, np.array([10.0, 20.0]), phi_f, d2, orth=tam2)
        r00 = [r for r in C.evaluate(modal2, [], lm) if r.metric == "COC diagonal" and r.item.startswith("test mode 1 ")][0]
        out[f"tam_{tag}"] = r00.status
    # (8) the COC of the projection: a shape outside the span of the compared modes
    q, _ = np.linalg.qr(phi)
    r = np.random.default_rng(2).standard_normal(nd)
    r -= q @ (q.T @ r)
    r /= np.linalg.norm(r)
    ratio = 0.75
    off = phi.copy()
    off[:, 0] = phi[:, 0] + ratio * np.linalg.norm(phi[:, 0]) * r
    so = O.orthogonality(off, phi, dofs, O.OrthMass("serep", gram=gram))
    out["resid"], out["resid_hand"] = float(so.residual[0]), ratio ** 2 / (1 + ratio ** 2)
    out["coc_high"] = float(so.coc[0, 0])
    off_rows = rows_of(off, so)
    out["off_status"] = [r.status for r in off_rows if r.metric == "COC diagonal" and r.item.startswith("test mode 1 ")][0]
    out["off_value_pass"] = float(so.coc[0, 0] >= 0.9)
    out["off_warned"] = float(any("test mode(s) 1 " in w for w in so.warnings) and not so.mode_hold)
    gall = phi_all.T @ mass @ phi_all
    dn = np.sqrt(np.diag(gall))
    mix = np.stack([0.8 * phi_all[:, 0] / dn[0] + 0.6 * phi_all[:, 4] / dn[4], phi_all[:, 1]], axis=1)
    o_all = O.orthogonality(mix, phi_all[:, :6], dofs, O.OrthMass("serep", gram=gall[:6, :6]))
    o_two = O.orthogonality(mix, phi_all[:, :2], dofs, O.OrthMass("serep", gram=gall[:2, :2]))
    out["coc_all"], out["coc_two"] = float(o_all.coc[0, 0]), float(o_two.coc[0, 0])
    st_two = [r.status for r in rows_of(mix, o_two, lim, phi_all[:, :2]) if r.metric == "COC diagonal"]
    st_all = [r.status for r in rows_of(mix, o_all, lim, phi_all[:, :6]) if r.metric == "COC diagonal"]
    out["mix_rows_unconfirmed"] = float(set(st_two) == {C.UNCONFIRMED} and set(st_all) == {C.UNCONFIRMED})
    one = O.orthogonality(phi_all[:, :1], phi_all[:, :1], dofs, O.OrthMass("serep", gram=gall[:1, :1]))
    out["one_refused"] = float(not one.computed)
    # (9) the Euclidean residual does not bound the mass-weighted error: heavy tip, upper modes. The projection onto the window gives 1.00 for a tip that is 10 % off: the row must not be PASS
    for tag, heavy, dh in (("heavy", 1000.0, 0.10), ("light", 10.0, 0.02)):
        ph, mh = _generalised_chain([1.0] * 6 + [heavy] * 2)
        amp = np.abs(ph[6:, :]).max(axis=0)
        band = sorted(int(i) for i in np.argsort(amp)[:4])
        fe = ph[:, band]
        gall = ph.T @ mh @ ph
        tt = fe.copy()
        tt[7, 0] += dh * np.abs(fe[:, 0]).max()
        alone = O.orthogonality(tt, fe, dofs, O.OrthMass("serep", gram=gall[np.ix_(band, band)]))
        out[f"{tag}_r"] = float(alone.residual[0])
        out[f"{tag}_coc_alone"] = float(alone.coc[0, 0])
        out[f"{tag}_coc_full"] = abs(tt[:, 0] @ mh @ fe[:, 0]) / math.sqrt((tt[:, 0] @ mh @ tt[:, 0]) * (fe[:, 0] @ mh @ fe[:, 0]))
        hrows = [x for x in rows_of(tt, alone, lim, fe) if x.metric == "COC diagonal" and x.item.startswith("test mode 1 ")]
        out[f"{tag}_status"] = hrows[0].status
        tam_full = O.orthogonality(tt, fe, dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, mh)))                  # the TAM of the full mass matrix judges it (the lab's matrix)
        out[f"{tag}_tam_coc"] = float(tam_full.coc[0, 0])
    # (9b) the reviewer's real-model case in a closed form: the window leaves out the lowest modes, the test mode holds 0.6 of mode 1 (true COC 0.8)
    p12, m12c = _generalised_chain([1.0] * 10 + [1000.0] * 2)
    d12c = [(f"M{i}", "x") for i in range(12)]
    g12 = p12.T @ m12c @ p12
    dn12 = np.sqrt(np.diag(g12))
    win = [2, 3, 4, 5, 6, 7]
    mix = p12[:, win].copy()
    mix[:, 1] = 0.8 * p12[:, 3] / dn12[3] + 0.6 * p12[:, 0] / dn12[0]
    only = O.orthogonality(mix, p12[:, win], d12c, O.OrthMass("serep", gram=g12[np.ix_(win, win)]))
    tam_mix = O.orthogonality(mix, p12[:, win], d12c, O.OrthMass("tam", tam=O.MassMatrix(d12c, m12c)))
    out["mix_true"] = abs(mix[:, 1] @ m12c @ p12[:, 3]) / math.sqrt((mix[:, 1] @ m12c @ mix[:, 1]) * (p12[:, 3] @ m12c @ p12[:, 3]))
    out["mix_alone"], out["mix_tam"] = float(only.coc[1, 1]), float(tam_mix.coc[1, 1])
    mrows = [x for x in rows_of(mix, only, lim, p12[:, win], d12c) if x.metric == "COC diagonal" and x.item.startswith("test mode 2 ")]
    out["mix_status"] = mrows[0].status
    # (9c) a SEREP matrix printed with 5, 6 and 7 digits: a singular PSD matrix of rank 4; a full-rank matrix with a wide mass spread: full rank; a real negative eigenvalue: refused
    pr, mr = _generalised_chain([1.0] * 12)
    fr = pr[:, :4]
    pinr = np.linalg.pinv(fr)
    msr = pinr.T @ (fr.T @ mr @ fr) @ pinr
    d12 = [(f"N{i}", "x") for i in range(12)]
    for fmt, tag in (("%.5e", "tam5_rank"), ("%.6e", "tam6_rank"), ("%.7e", "tam7_rank")):
        txt = "# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in d12) + "\n" + "\n".join(f"{d12[i][0]}:{d12[i][1]}," + ",".join(fmt % float(v) for v in msr[i]) for i in range(12)) + "\n"
        out[tag] = float(O.parse_mass_csv(txt, "serep.csv").rank)
    d30 = [(f"N{i}", "x") for i in range(30)]
    spread = np.diag(np.geomspace(1.0, 2e4, 30))
    txt = "# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in d30) + "\n" + "\n".join(f"{d30[i][0]}:{d30[i][1]}," + ",".join("%.6e" % float(v) for v in spread[i]) for i in range(30)) + "\n"
    out["spread_rank"] = float(O.parse_mass_csv(txt, "spread.csv").rank)
    rng5, worst = np.random.default_rng(1), 0                       # final review: a singular 5-DOF / 4-mode matrix printed to 6 digits never reads full rank (the rounding of the digits is the noise floor)
    d5 = [(f"N{i}", "x") for i in range(5)]
    for _ in range(25):
        p5 = np.linalg.pinv(rng5.standard_normal((5, 4)))
        m5 = p5.T @ np.diag(10.0 ** rng5.uniform(0, 2, 4)) @ p5
        t5 = ("# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in d5) + "\n"
              + "\n".join(f"{d5[i][0]}:{d5[i][1]}," + ",".join("%.6g" % float(v) for v in m5[i]) for i in range(5)) + "\n")
        worst = max(worst, O.parse_mass_csv(t5, "s5.csv").rank)
    out["tam5dof_rank_max"] = float(worst)
    try:
        O.parse_mass_csv("# unit: kg\n,A:x,B:x\nA:x,1,0\nB:x,0,-0.01\n")
        out["indefinite_refused"] = 0.0
    except Exception:
        out["indefinite_refused"] = 1.0
    # (10) a singular TAM (the matrix of the SEREP reduction of a 12-mass chain): the null space has no mass
    n12 = 12
    p12, m12 = _generalised_chain([1.0] * n12)
    d12 = [(f"N{i}", "x") for i in range(n12)]
    f12 = p12[:, :4]
    pin = np.linalg.pinv(f12)
    tam12 = O.MassMatrix(d12, pin.T @ (f12.T @ m12 @ f12) @ pin, "serep.csv", "kg", rank=4)
    q12, _ = np.linalg.qr(f12)
    r12 = np.random.default_rng(3).standard_normal(n12)
    r12 -= q12 @ (q12.T @ r12)
    r12 /= np.linalg.norm(r12)
    t12 = f12.copy()
    t12[:, 0] = f12[:, 0] + 0.75 * np.linalg.norm(f12[:, 0]) * r12
    st12 = O.orthogonality(t12, f12, d12, O.OrthMass("tam", tam=tam12))
    out["tam_share"], out["tam_share_hand"] = float(st12.residual[0]), 0.75 ** 2 / (1 + 0.75 ** 2)
    out["tam_coc"], out["tam_held"] = float(st12.coc[0, 0]), float(0 in st12.mode_hold and 1 not in st12.mode_hold)
    out["tam_full_coc"] = abs(t12[:, 0] @ m12 @ f12[:, 0]) / math.sqrt((t12[:, 0] @ m12 @ t12[:, 0]) * (f12[:, 0] @ m12 @ f12[:, 0]))
    rank1 = O.orthogonality(np.random.default_rng(4).random((n12, 1)) + 0.2, f12[:, :1], d12, O.OrthMass("tam", tam=O.MassMatrix(d12, np.ones((n12, n12)))))
    out["tam_one_mode_held"] = float(bool(rank1.hold) and "only one FE mode" in rank1.hold)
    return out


def s5a_eval(out: _Out, study) -> list[Check]:
    h = out["hand"]
    return [
        Check("model against itself: max |COC - I| (8-mass chain, 6 modes, closed form)", out["self_coc"], 0.0, 1e-12),
        Check("model against itself: max |AOC - I|", out["self_aoc"], 0.0, 1e-12),
        Check("SEREP of the full chain: max |COC - I|", out["self_serep"], 0.0, 1e-12),
        Check("all DOFs measured: SEREP and full mass matrix give the same COC and AOC", out["serep_vs_tam"], 0.0, 1e-12),
        Check("two masses, tip mass +20 %: COC(test 1, FE 2) by hand", out["coc01"], out["hand"]["coc01"], 1e-10),
        Check("two masses, tip mass +20 %: COC(test 2, FE 1) by hand", out["coc10"], h["coc10"], 1e-10),
        Check("two masses, tip mass +20 %: COC diagonal (test 1, FE 1) by hand", out["coc00"], h["coc00"], 1e-10),
        Check("two masses, tip mass +20 %: AOC of the two test modes by hand", out["aoc01"], h["aoc01"], 1e-10),
        Check("off-diagonal COC is a real non-zero term (not squared: > 0.01)", float(out["coc01"] > 0.01 and out["coc10"] > 0.01), 1.0, 0.0),
        Check("fewer DOFs: SEREP reduction = full-DOF orthogonality of the expanded shape", out["reduced_vs_full"], 0.0, 1e-12),
        Check("modulus difference of [1 2 2] and [1 2 1] by hand", out["md"], 0.3876275643, 1e-9),
        Check("modulus difference: the DOF of the largest difference is the third", float(out["md_dof"] == "c:x"), 1.0, 0.0),
        Check("generalised mass of a unit x translation of a tetrahedron = rho V", out["gram_x"], out["mass_ref"], 1e-12, "t"),
        Check("generalised mass of a unit y translation of a tetrahedron = rho V", out["gram_y"], out["mass_ref"], 1e-12, "t"),
        Check("x and y translations are mass-orthogonal", out["gram_xy"], 0.0, 1e-12, "t"),
        Check("no reduced mass: verdict UNCONFIRMED_REQUIREMENTS, never PASS", float(out["v_none"] == C.VERDICT_UNCONFIRMED), 1.0, 0.0),
        Check("a TAM and passing limits: verdict PASS", float(out["v_pass"] == C.PASS), 1.0, 0.0),
        Check("the SEREP reduction of the very same model, passing values: verdict UNCONFIRMED_REQUIREMENTS (indicative, never PASS)", float(out["v_serep"] == C.VERDICT_UNCONFIRMED), 1.0, 0.0),
        Check("mixed test modes against tight orthogonality limits through a TAM: verdict FAIL", float(out["v_fail"] == C.FAIL), 1.0, 0.0),
        Check("... through SEREP the rows stay UNCONFIRMED with a warning hint, never FAIL", out["serep_bad_never_fail"], 1.0, 0.0),
        Check("TAM, diagonal COC limit 0.001 below the hand value: PASS", float(out["tam_below"] == C.PASS), 1.0, 0.0),
        Check("TAM, diagonal COC limit 0.001 above the hand value: FAIL", float(out["tam_above"] == C.FAIL), 1.0, 0.0),
        Check("FE mode 1 + 0.75 |phi_1| r (r outside the span): out-of-span residual 0.75^2 / (1 + 0.75^2) by hand", out["resid"], out["resid_hand"], 1e-9),
        Check("that shape has a high COC of its projection (> 0.9) and a warning names its test mode (no hold)", float(out["coc_high"] > 0.9 and out["off_warned"] == 1.0), 1.0, 0.0),
        Check("... its COC diagonal row is UNCONFIRMED although the value is above the limit (never PASS)", float(out["off_status"] == C.UNCONFIRMED and out["off_value_pass"] == 1.0), 1.0, 0.0),
        Check("0.8 phi_1 + 0.6 phi_5 against all the modes: the true COC 0.8", out["coc_all"], 0.8, 1e-9),
        Check("... against the modes 1 and 2 only: COC of the projection > 0.99; both selections give UNCONFIRMED rows", float(out["coc_two"] > 0.99 and out["mix_rows_unconfirmed"] == 1.0), 1.0, 0.0),
        Check("one FE mode (COC is always 1): not computed", out["one_refused"], 1.0, 0.0),
        Check("heavy tip 10 % off, upper modes: the Euclidean residual stays below 0.05 (it does not bound the mass-weighted error)", float(out["heavy_r"] < 0.05), 1.0, 0.0),
        Check("... the SEREP COC is 1.00 (it would pass any limit), the COC of the full mass matrix 0.52: the SEREP row is UNCONFIRMED (the false PASS of the review is gone)",
              float(out["heavy_coc_alone"] > 0.999 and out["heavy_coc_full"] < 0.9 and out["heavy_status"] == C.UNCONFIRMED), 1.0, 0.0),
        Check("... the TAM of the full mass matrix gives that 0.52", out["heavy_tam_coc"], out["heavy_coc_full"], 1e-9),
        Check("masses of 10, tip 2 %: the SEREP row is UNCONFIRMED all the same, the full-mass COC is above 0.99", float(out["light_status"] == C.UNCONFIRMED and out["light_coc_full"] > 0.99), 1.0, 0.0),
        Check("closed form of the reviewer's real-model case: 0.8 phi_4 + 0.6 phi_1 has the true COC 0.80", out["mix_true"], 0.8, 1e-9),
        Check("... the projection onto the window gives a COC above 0.99 (the false PASS of the review) and its row is UNCONFIRMED", float(out["mix_alone"] > 0.99 and out["mix_status"] == C.UNCONFIRMED), 1.0, 0.0),
        Check("... the TAM of the full mass matrix gives the true 0.80", out["mix_tam"], 0.8, 1e-9),
        Check("a rank-4 SEREP matrix printed with 5 digits (%.5e) loads as a singular PSD matrix of rank 4", out["tam5_rank"], 4.0, 0.0),
        Check("... with 6 digits (%.6e)", out["tam6_rank"], 4.0, 0.0),
        Check("... with 7 digits (%.7e)", out["tam7_rank"], 4.0, 0.0),
        Check("a full-rank diagonal matrix with a mass spread of 2e4 (30 DOFs) is read as full rank", out["spread_rank"], 30.0, 0.0),
        Check("a singular 5-DOF / 4-mode matrix printed with 6 digits (%.6g) is never read as full rank (25 random matrices: the largest rank read is 4)", out["tam5dof_rank_max"], 4.0, 0.0),
        Check("a matrix with a negative eigenvalue of 1 % of the largest is refused", out["indefinite_refused"], 1.0, 0.0),
        Check("rank-4 TAM of a 12-mass chain, FE mode 1 + 0.75 |phi_1| r (r in the null space): share in the null space 0.75^2 / (1 + 0.75^2)", out["tam_share"], out["tam_share_hand"], 1e-9),
        Check("... the TAM gives COC 1.00 (blind to r), the full mass matrix 0.80; the rows of that mode only are UNCONFIRMED",
              float(out["tam_coc"] > 0.999 and 0.79 < out["tam_full_coc"] < 0.81 and out["tam_held"] == 1.0), 1.0, 0.0),
        Check("one FE mode through a rank-1 matrix: held (the fit of the matrix cannot be checked)", out["tam_one_mode_held"], 1.0, 0.0),
    ]


SPACE_CORR_CASES["S5A"] = Case("S5A", "Test correlation: mass-weighted cross-orthogonality COC / AOC and modulus difference on chains with closed-form modes; the verdict is never PASS without a reduced mass",
                               _build, s5a_eval, run=s5a_run)

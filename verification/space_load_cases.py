"""Verification of the launch loads and mass properties of the space-structures verification (F14 S2, `fea_core.space.loads` / `mass_props` / `budgets`).

S2A  mass, centre of mass and the Steiner-shifted inertia tensor of a solid block + an offset point mass with rotary inertia + a surface mass: closed-form
     (a second, independent evaluation: the parallel-axis theorem applied body by body).
S2B  Miles' equation on a 1-DOF system through the whole random-vibration chain: relative displacement RMS and stress RMS against sqrt(pi/2 fn Q W) / w^2;
     the sine amplitude at resonance A / (2 zeta w^2) through the harmonic chain.
S2C  BOOKKEEPING check of the orchestrator: the 3-axis envelope is bit-identical to the elementwise maximum of three single-axis `run_dynamics` calls (sine, random
     and shock; a SYNTHETIC modal basis built twice with the same numbers), and the damage envelope is the SUM over the axes. It proves the envelope / sum
     arithmetic and that nothing is shared between the axes; it does NOT test the dynamics themselves (both sides run the same code on the same synthetic numbers).
S2D  the minimum-frequency check and the budget verdicts (arbitrary test limits, confirmed / unconfirmed; split-mass cases never PASS).
S2E  the QSL generator: the signed cases and their factors (arbitrary test factors).
S2F  INDEPENDENT check: `run_3axis` over one real modal basis (CalculiX) against three complete single-axis `run_study` analyses of a cantilever, each with its
     own modal basis and its own static solutions.

Every number that looks like a requirement here (a factor, an ASD level, a limit) is an ARBITRARY TEST VALUE entered by this module - not a value of any
standard; the program ships none.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from fea_core.dynamics.model import PointMass, SurfaceMass
from fea_core.mesh import Mesh
from fea_core.space import budgets as BG
from fea_core.space import environments as EV
from fea_core.space import loads as LD
from fea_core.space import mass_props as MP
from fea_core.space.model import SpaceSettings
from fea_core.space.requirements import confirm, ecss_32_10c_template
from fea_core.study import Material, Study

from .cases import Case, Check

SPACE_LOAD_CASES: dict[str, Case] = {}


class _Out(dict):
    """The runs for the evaluation; `.warnings` for run_all."""

    @property
    def warnings(self) -> list[str]:
        return []

RHO = 7.85e-9                                   # t/mm³ (steel, as in CLAUDE.md)
TEST_G = 9810.0


# ------------------------------------------------------------------ synthetic meshes and studies
def tet10_from_tet4(m4: Mesh, surfaces: dict[str, np.ndarray] | None = None) -> Mesh:
    """tet10 mesh (ccx node order) with straight edges from a tet4 mesh: the mid-edge nodes are added; boundary triangles of `surfaces` become tri6."""
    nodes = [list(x) for x in m4.nodes]
    mid: dict[tuple[int, int], int] = {}

    def mid_of(a: int, b: int) -> int:
        k = (a, b) if a < b else (b, a)
        if k not in mid:
            mid[k] = len(nodes)
            nodes.append([(m4.nodes[a][i] + m4.nodes[b][i]) / 2.0 for i in range(3)])
        return mid[k]

    tets = []
    for t in m4.tets:
        a, b, c, d = (int(x) for x in t)
        tets.append([a, b, c, d, mid_of(a, b), mid_of(b, c), mid_of(c, a), mid_of(a, d), mid_of(b, d), mid_of(c, d)])
    m = Mesh(np.array(nodes, float), np.array(tets, np.int64), np.asarray(m4.tet_part).copy(), list(m4.part_names))
    for name, tri in (surfaces or {}).items():
        m.surfaces[name] = np.array([[a, b, c, mid_of(a, b), mid_of(b, c), mid_of(c, a)] for a, b, c in (tuple(int(v) for v in r[:3]) for r in tri)], np.int64)
    return m


def box_mesh_tet10(a: float, b: float, c: float, n: tuple[int, int, int] = (4, 2, 2)) -> Mesh:
    """Conforming tet10 mesh of the box [0,a]×[0,b]×[0,c] with the face selections "Top" (z = c) and "Support" (x = 0)."""
    import itertools
    nx, ny, nz = n
    hx, hy, hz = a / nx, b / ny, c / nz

    def idx(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    nodes = np.array([[i * hx, j * hy, k * hz] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)], float)
    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        for perm in itertools.permutations(range(3)):
            cc = [i, j, k]
            path = [idx(*cc)]
            for ax in perm:
                cc[ax] += 1
                path.append(idx(*cc))
            t = np.array(path)
            vol = np.linalg.det(nodes[t[1:]] - nodes[t[0]])
            tets.append(t if vol > 0 else t[[0, 2, 1, 3]])
    m4 = Mesh(nodes, np.array(tets, np.int64), np.zeros(len(tets), np.int64), ["Block"])
    tri, _ = m4.boundary_faces()
    tol = 1e-9 * max(a, b, c)
    top = tri[(nodes[tri[:, :3], 2] >= c - tol).all(axis=1)]
    sup = tri[(nodes[tri[:, :3], 0] <= tol).all(axis=1)]
    return tet10_from_tet4(m4, {"Top": top, "Support": sup})


def steel_study(name: str = "S2 test") -> Study:
    st = Study(name, "x.step")
    st.add_material(Material("Steel", 210000.0, 0.3, RHO, Rm=360.0), default=True)
    return st


# ------------------------------------------------------------------ S2A: mass, centre of mass, inertia
A_, B_, C_ = 40.0, 20.0, 10.0
PT_MASS, PT_AT = 2.0e-3, (55.0, 3.0, 12.0)                   # t, mm  (a point mass outside the block)
PT_I = [[3.0e1, 2.0, 0.0], [2.0, 4.0e1, 1.0], [0.0, 1.0, 5.0e1]]          # t·mm², about its own point
SURF_KG_M2 = 3.0                                              # kg/m² on the top face


def _s2a_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return steel_study("S2A mass properties"), []


def _s2a_run(study: Study, workdir: Path, parts: Any) -> dict[str, Any]:
    mesh = box_mesh_tet10(A_, B_, C_)
    st = steel_study("S2A")
    st.masses.append(PointMass("Support", PT_MASS, point=list(PT_AT), inertia=PT_I, name="unit"))
    st.masses.append(SurfaceMass("Top", SURF_KG_M2, name="board"))
    return _Out(props=MP.mass_properties(st, mesh), block_only=MP.mass_properties(steel_study("S2A"), mesh))


def _reference() -> tuple[float, np.ndarray, np.ndarray]:
    """Independent reference: each body about its own centre, shifted body by body with the parallel-axis theorem."""
    bodies: list[tuple[float, np.ndarray, np.ndarray]] = []
    m_blk = RHO * A_ * B_ * C_
    I_blk = m_blk / 12.0 * np.diag([B_ ** 2 + C_ ** 2, A_ ** 2 + C_ ** 2, A_ ** 2 + B_ ** 2])
    bodies.append((m_blk, np.array([A_ / 2, B_ / 2, C_ / 2]), I_blk))
    bodies.append((PT_MASS, np.array(PT_AT), np.array(PT_I)))
    m_sf = SURF_KG_M2 * 1e-9 * A_ * B_                       # a thin plate in the plane z = C: I about its own centre
    I_sf = m_sf / 12.0 * np.diag([B_ ** 2, A_ ** 2, A_ ** 2 + B_ ** 2])
    bodies.append((m_sf, np.array([A_ / 2, B_ / 2, C_]), I_sf))
    M = sum(b[0] for b in bodies)
    c = sum(b[0] * b[1] for b in bodies) / M
    I = np.zeros((3, 3))
    for m, p, Ii in bodies:
        d = p - c
        I += Ii + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return M, c, I


def _s2a_eval(out: dict[str, Any], study: Study) -> list[Check]:
    p = out["props"]
    M, c, I = _reference()
    scale = float(np.abs(I).max())
    checks = [Check("S2A total mass = block + point mass + surface mass", p.mass_t, M, 1e-12, "t"),
              Check("S2A block alone: mass = rho a b c", out["block_only"].mass_t, RHO * A_ * B_ * C_, 1e-12, "t")]
    for i, ax in enumerate("xyz"):
        checks.append(Check(f"S2A centre of mass {ax}", float(p.cog[i]), float(c[i]), 1e-10, "mm"))
    for (i, j) in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        checks.append(Check(f"S2A inertia I{'xyz'[i]}{'xyz'[j]} about the centre of mass (Steiner)", float(p.inertia_cog[i, j]) / scale, float(I[i, j]) / scale, 1e-9))
    blk = out["block_only"]
    checks.append(Check("S2A block alone: Ixx = m (b^2 + c^2) / 12", float(blk.inertia_cog[0, 0]), RHO * A_ * B_ * C_ / 12 * (B_ ** 2 + C_ ** 2), 1e-10, "t mm2"))
    return checks


SPACE_LOAD_CASES["S2A"] = Case("S2A", "Space loads: mass, centre of mass and Steiner-shifted inertia of a block + point mass + surface mass", _s2a_build, _s2a_eval,
                               run=_s2a_run)


# ------------------------------------------------------------------ synthetic modal basis (no solver)
def single_tet10() -> Mesh:
    X = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([X, [(X[a] + X[b]) / 2 for a, b in mids]])
    return Mesh(nodes, np.arange(10)[None, :], np.zeros(1, np.int64), ["A"])


def synthetic_basis(freqs: list[float], gamma_axis: list[int], mass_t: float = 1.0e-3, seed: int = 7) -> Any:
    """A modal basis on `single_tet10` without a solver: mode k moves the whole body along the axis gamma_axis[k] (normalised by the mass), its stress is a
    fixed uniaxial pattern that differs from node to node (seeded), so that the three axes govern different nodes."""
    from fea_core.dynamics.modal import ModalBasis
    rng = np.random.default_rng(seed)
    K, N = len(freqs), 10
    shapes = np.zeros((K, N, 3))
    stress = np.zeros((K, N, 6), np.float32)
    gamma = np.zeros((K, 6))
    eff = np.zeros((K, 6))
    for k, ax in enumerate(gamma_axis):
        shapes[k, :, ax] = 1.0 / math.sqrt(mass_t)
        gamma[k, ax] = math.sqrt(mass_t)
        eff[k, ax] = mass_t
        stress[k, :, ax] = (50.0 * (0.5 + rng.random(N))).astype(np.float32)          # [MPa per unit modal coordinate] uniaxial in the shaking direction
    b = ModalBasis(np.array(freqs, float), shapes, stress, gamma, eff, np.array([mass_t, mass_t, mass_t, 0.0, 0.0, 0.0]), np.zeros((K, 3)))
    b.model_mass = mass_t
    b.engine = "synthetic"
    return b


def synthetic_statics(basis: Any, seed: int = 11) -> dict[Any, Any]:
    rng = np.random.default_rng(seed)
    out: dict[Any, Any] = {}
    for a in range(3):
        d = np.zeros(3)
        d[a] = 1.0
        out[("b", tuple(np.round(d, 12)))] = {"U": rng.normal(size=(10, 3)) * 1e-6, "S": (rng.normal(size=(10, 6)) * 1e-3).astype(np.float32),
                                              "RF": np.zeros((1, 3))}
    return out


def consistent_statics(basis: Any) -> dict[Any, Any]:
    """The static states of the unit accelerations that a COMPLETE modal basis reproduces exactly (u = sum gamma phi / w^2): the residual static correction is then zero,
    as for a one-mass system whose only mode carries all the mass."""
    out: dict[Any, Any] = {}
    w2 = basis.omega ** 2
    for a in range(3):
        d = np.zeros(3)
        d[a] = 1.0
        coef = basis.gamma[:, a] / w2
        out[("b", tuple(np.round(d, 12)))] = {"U": np.einsum("k,kni->ni", coef, basis.shapes), "S": np.einsum("k,kni->ni", coef, basis.stress.astype(float)).astype(np.float32),
                                              "RF": np.zeros((1, 3))}
    return out


# ------------------------------------------------------------------ S2B: Miles
FN_MILES, ZETA_MILES, W_MILES = 100.0, 0.05, 0.04              # Hz, -, g²/Hz (arbitrary test values)
S0_MILES = 80.0                                                # MPa per unit modal coordinate
SINE_A_MILES = 4.0                                             # g, flat sine level of the same 1-DOF system


def _s2b_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return steel_study("S2B Miles"), []


def _level_random(asd: list[list[float]], duration_s: float | None = 600.0, sigma: float | None = 3.0) -> EV.EnvLevel:
    lv = EV.EnvLevel("qualification")
    lv.random = EV.EnvTable("random", asd, duration_s=duration_s, sigma=sigma, document="test", revision="-", clause="test clause")
    return lv


def _s2b_run(study: Study, workdir: Path, parts: Any) -> dict[str, Any]:
    mesh = single_tet10()
    st = steel_study("S2B")
    st.dynamics.damping.zeta = ZETA_MILES
    basis = synthetic_basis([FN_MILES], [2])
    basis.stress[:] = 0.0
    basis.stress[0, :, 2] = S0_MILES                       # uniaxial σ_zz = s0 · q at every node
    lv = _level_random([[0.1, W_MILES], [20000.0, W_MILES]])
    per, _w = LD.analyze_axes(st, mesh, basis, consistent_statics(basis), workdir, lambda s: None, lv, ["random"], (2,))
    lv_sine = EV.EnvLevel("qualification")
    lv_sine.sine = EV.EnvTable("sine", [[20.0, SINE_A_MILES], [500.0, SINE_A_MILES]], document="test", revision="-", clause="c")       # a flat sine level
    per_s, _w = LD.analyze_axes(st, mesh, basis, consistent_statics(basis), workdir, lambda s: None, lv_sine, ["sine"], (2,))
    return _Out(field=per["random"][2], sine=per_s["sine"][2], gamma=float(basis.gamma[0, 2]), phi=float(basis.shapes[0, 0, 2]))


def _s2b_eval(out: dict[str, Any], study: Study) -> list[Check]:
    f = out["field"]
    wn = 2 * math.pi * FN_MILES
    q_factor = 1.0 / (2 * ZETA_MILES)
    a_rms = math.sqrt(math.pi / 2.0 * FN_MILES * q_factor * W_MILES * TEST_G ** 2)             # Miles, mm/s²
    u_miles = a_rms / wn ** 2                                                                # relative displacement RMS, mm
    return [Check("S2B Miles: relative acceleration RMS (w^2 x displacement RMS) = sqrt(pi/2 fn Q W)", float(f.U[0]) * wn ** 2 / TEST_G,
                  a_rms / TEST_G, 5e-4, "g"),
            Check("S2B Miles: relative displacement RMS = a_rms / w^2", float(f.U[0]), u_miles, 5e-4, "mm"),
            Check("S2B Miles: von Mises RMS = s0 * Gamma * u_rms (uniaxial modal stress)", float(f.S[0]), S0_MILES * out["gamma"] * u_miles, 5e-4, "MPa"),
            Check("S2B sine at resonance: relative displacement amplitude = A / (2 zeta w^2)", float(out["sine"].U[0]), SINE_A_MILES * TEST_G / (2 * ZETA_MILES * wn ** 2),
                  2e-3, "mm"),
            Check("S2B sine at resonance: von Mises amplitude = s0 * Gamma * x", float(out["sine"].S[0]),
                  S0_MILES * out["gamma"] * SINE_A_MILES * TEST_G / (2 * ZETA_MILES * wn ** 2), 2e-3, "MPa")]


SPACE_LOAD_CASES["S2B"] = Case("S2B", "Space loads: Miles' equation and the resonant sine amplitude of a 1-DOF system (RMS acceleration, displacement and stress)", _s2b_build, _s2b_eval, run=_s2b_run)


# ------------------------------------------------------------------ S2C: bookkeeping - the 3-axis envelope = max of three single-axis calls (synthetic basis)
def three_axis_level() -> EV.EnvLevel:
    lv = EV.EnvLevel("qualification", combine="srss")
    lv.sine = EV.EnvTable("sine", [[20.0, 5.0], [100.0, 5.0], [400.0, 8.0]], sweep_rate_oct_min=2.0, sweeps=1, document="test", revision="-", clause="c")
    lv.random = EV.EnvTable("random", [[20.0, 0.01], [200.0, 0.03], [1000.0, 0.03]], duration_s=120.0, sigma=3.0, document="test", revision="-", clause="c")
    lv.shock = EV.EnvTable("shock", [[100.0, 20.0], [1000.0, 200.0]], document="test", revision="-", clause="c")
    return lv


def independent_single_axis(study: Study, mesh: Mesh, basis: Any, statics: dict[Any, Any], lv: EV.EnvLevel, axis: int, wd: Path) -> Any:
    """A single-axis dynamics run written WITHOUT the orchestrator: the specs are set directly and `run_dynamics` is called for this axis only."""
    import copy

    from fea_core.dynamics.model import Excitation, HarmonicSpec, RandomSpec, ShockSpec
    from fea_core.dynamics.run import run_dynamics
    st = copy.copy(study)
    ds = copy.deepcopy(study.dynamics)
    ds.enabled = True
    sine, rnd, sh = lv.sine, lv.random, lv.shock
    assert sine is not None and rnd is not None and sh is not None
    ds.harmonic = HarmonicSpec(enabled=True, f_min=sine.points[0][0], f_max=sine.points[-1][0], sweep_rate=float(sine.sweep_rate_oct_min or 0.0),
                               sweeps=int(sine.sweeps or 1), excitation=[Excitation(kind="base_acc", dof=axis + 1, table=[list(p) for p in sine.points], unit="g")])
    ds.random = RandomSpec(enabled=True, asd=[list(p) for p in rnd.points], dof=axis + 1, duration_h=float(rnd.duration_s or 3600.0) / 3600.0)
    ds.shock = ShockSpec(enabled=True, pulse="srs_table", srs_table=[list(p) for p in sh.points], dof=axis + 1, combine=lv.combine, missing_mass=lv.missing_mass,
                         transient=False)
    st.dynamics = ds
    return run_dynamics(st, mesh, wd / f"indep_{axis}", lambda s: None, basis=basis, statics_by_key=statics)


def _s2c_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return steel_study("S2C 3-axis"), []


def _s2c_run(study: Study, workdir: Path, parts: Any) -> dict[str, Any]:
    mesh = single_tet10()
    st = steel_study("S2C")
    st.dynamics.damping.zeta = 0.03
    st.dynamics.modes = 3
    basis = synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
    sk = synthetic_statics(basis)
    lv = three_axis_level()
    per, _w = LD.analyze_axes(st, mesh, basis, sk, workdir, lambda s: None, lv, ["sine", "random", "shock"])
    res = LD.build_result(st, mesh, lv, ["sine", "random", "shock"], [0, 1, 2], per, basis, env_confirmed=True, req_confirmed=True)
    indep = {}
    for a in (0, 1, 2):                                                         # each independent run builds ITS OWN basis and statics: nothing is shared
        own_basis = synthetic_basis([85.0, 130.0, 210.0], [0, 1, 2])
        indep[a] = independent_single_axis(st, mesh, own_basis, synthetic_statics(own_basis), lv, a, workdir)
        assert own_basis is not basis
    return _Out(res=res, indep=indep)


def independent_arrays(out: Any, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "sine":
        return np.asarray(out.harmonic["env_S"], np.float32), np.asarray(out.harmonic["env_U"], np.float32)
    if kind == "random":
        return np.asarray(out.random["S_rms"], np.float32), np.linalg.norm(np.asarray(out.random["U_rms"], float), axis=1).astype(np.float32)
    return np.asarray(out.shock["mises"], np.float32), np.linalg.norm(np.asarray(out.shock["U"], float), axis=1).astype(np.float32)


def _s2c_eval(out: dict[str, Any], study: Study) -> list[Check]:
    res, indep = out["res"], out["indep"]
    checks: list[Check] = []
    for k in ("sine", "random", "shock"):
        arrs = {a: independent_arrays(indep[a], k) for a in (0, 1, 2)}
        s_ref = np.max(np.stack([arrs[a][0] for a in (0, 1, 2)]), axis=0)
        u_ref = np.max(np.stack([arrs[a][1] for a in (0, 1, 2)]), axis=0)
        same_axes = all(np.array_equal(res.per_axis[k][a].S, arrs[a][0]) and np.array_equal(res.per_axis[k][a].U, arrs[a][1]) for a in (0, 1, 2))
        checks += [Check(f"S2C {k}: every per-axis field = the separate single-axis call (bitwise)", float(same_axes), 1.0, 0.0),
                   Check(f"S2C {k}: envelope of sigma_vM = max over the 3 single-axis calls (bitwise)", float(np.array_equal(res.envelopes[k].S, s_ref)), 1.0, 0.0),
                   Check(f"S2C {k}: envelope of |U| = max over the 3 single-axis calls (bitwise)", float(np.array_equal(res.envelopes[k].U, u_ref)), 1.0, 0.0),
                   Check(f"S2C {k}: the three axes govern different nodes (the envelope is not one axis)", float(len(np.unique(res.envelopes[k].axis_S)) > 1), 1.0, 0.0)]
    d_axes = [np.asarray(indep[a].random["D"], np.float32) for a in (0, 1, 2)]
    d_sum = (d_axes[0].astype(np.float64) + d_axes[1].astype(np.float64) + d_axes[2].astype(np.float64)).astype(np.float32)
    d_max = np.max(np.stack(d_axes), axis=0)
    e = res.envelopes["random"]
    checks.append(Check("S2C random damage envelope = SUM over the 3 single-axis calls (the axes are tested one after the other; bitwise)",
                        float(e.D is not None and np.array_equal(e.D, d_sum)), 1.0, 0.0))
    checks.append(Check("S2C the plan's max(D) over the axes is kept as information (bitwise)", float(e.D_axis_max is not None and np.array_equal(e.D_axis_max, d_max)), 1.0, 0.0))
    checks.append(Check("S2C the summed damage is never below the damage of one axis", float(e.D is not None and e.D_axis_max is not None and bool((e.D >= e.D_axis_max).all())), 1.0, 0.0))
    return checks


SPACE_LOAD_CASES["S2C"] = Case("S2C", "Space loads: bookkeeping of the 3-axis envelope (bit-identical to the maximum of three single-axis calls on a synthetic basis; damage summed)", _s2c_build, _s2c_eval,
                               run=_s2c_run)


# ------------------------------------------------------------------ S2D: frequency and budget checks (arbitrary test limits)
def space_with_limits(limits: dict[str, float], confirmed: bool = True, env_axis: str | None = None, env_confirmed: bool = True) -> SpaceSettings:
    """A space setting with the requirement set of the test limits (confirmed or not) and, with `env_axis`, environments with that launch axis (confirmed or not)."""
    rs = ecss_32_10c_template()
    rs.verification_approach = "analysis_only"
    BG.install_limit_rows(rs)
    for x in rs.limits:
        if x.key in limits:
            x.value = limits[x.key]
            x.document, x.revision, x.clause = "test document", "test revision", "test clause"
    for r in rs.rules:
        if r.check == "yield" and r.structure_class == "metallic":
            r.factors = ["FOSY"]
            r.stress_measure, r.thermal_handling, r.allowable = "von_mises", "none", "Fy"
            r.document, r.revision, r.clause = "test", "test", "test"
    for f in rs.factors:
        if f.key == "FOSY" and f.structure_class == "metallic" and f.approach == "analysis_only":
            f.value, f.document, f.revision, f.clause = 1.25, "test", "test", "test"
    sp = SpaceSettings(requirement_set=rs)
    if confirmed:
        confirm(rs, "test engineer", "2026-01-01", scope=sp)
    if env_axis is not None:
        es = EV.EnvironmentSet(levels=[three_axis_level()], launch_axis=env_axis)
        if env_confirmed:
            EV.confirm_environment(es, "test engineer", "2026-01-01")
        EV.store_environments(sp, es)
    return sp


def _s2d_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return steel_study("S2D budgets"), []


def _s2d_run(study: Study, workdir: Path, parts: Any) -> dict[str, Any]:
    # a COMPLETE basis (every direction has >= 90 % of its effective mass in the computed modes): x 95 % at 210 Hz, y 92 % at 140 Hz (+ 2 % at 95 Hz), z 93 % at 150 Hz
    freq = np.array([95.0, 140.0, 150.0, 210.0])
    frac = np.array([[0.0, 0.02, 0.0], [0.0, 0.92, 0.0], [0.0, 0.0, 0.93], [0.95, 0.0, 0.0]])
    lim = {"f_min_axial_hz": 200.0, "f_min_lateral_hz": 120.0, "f_min_eff_mass_fraction": 0.5}
    sp = space_with_limits(lim, env_axis="x")
    y_ok = np.array([0.0, 0.92, 0.0])
    # the three SPLIT-MASS cases of the review (none of them may be PASS: the earlier rule assumed that f_1 lies above the highest computed frequency)
    split_30 = ([30.0, 140.0, 300.0], np.array([[0.30, 0, 0], y_ok, [0.30, 0, 0]]))                     # 30 % at 30 Hz + 30 % at 300 Hz, f_min 200 Hz
    split_49 = ([30.0, 140.0, 300.0], np.array([[0.49, 0, 0], y_ok, [0.49, 0, 0]]))                     # 49 % at 30 Hz + 49 % at 300 Hz
    split_lat = ([20.0, 25.0, 140.0, 250.0], np.array([[0, 0, 0.45], [0, 0, 0.45], y_ok, [0.95, 0, 0]]))  # lateral z: 45 % at 20 Hz + 45 % at 25 Hz
    return _Out({"ok": BG.frequency_budget(freq, frac, "x", sp),
                 "low": BG.frequency_budget(freq, frac, "x", space_with_limits({**lim, "f_min_axial_hz": 250.0}, env_axis="x")),
                 "unconf": BG.frequency_budget(freq, frac, "x", space_with_limits(lim, confirmed=False, env_axis="x")),
                 "env_unconf": BG.frequency_budget(freq, frac, "x", space_with_limits(lim, env_axis="x", env_confirmed=False)),
                 "nothr": BG.frequency_budget(freq, frac, "x", space_with_limits({"f_min_axial_hz": 90.0, "f_min_lateral_hz": 90.0}, env_axis="x")),
                 "noaxis": BG.frequency_budget(freq, frac, "", sp),
                 "split_30": BG.frequency_budget(np.array(split_30[0]), split_30[1], "x", sp),
                 "split_49": BG.frequency_budget(np.array(split_49[0]), split_49[1], "x", sp),
                 "split_lat": BG.frequency_budget(np.array(split_lat[0]), split_lat[1], "x", sp),
                 "split_fail": BG.frequency_budget(np.array([30.0, 140.0, 300.0, 400.0]), np.array([[0.55, 0, 0], y_ok, [0.40, 0, 0], [0, 0, 0.95]]), "x", sp),
                 "short": BG.frequency_budget(np.array([95.0, 140.0, 150.0]), np.array([[0, 0.02, 0], y_ok, [0, 0, 0.93]]), "x", sp)})


def _status(rep: Any) -> list[str]:
    return [i.status for i in rep.items]


def _s2d_eval(out: dict[str, Any], study: Study) -> list[Check]:
    ok = out["ok"]
    return [Check("S2D first axial frequency (lowest mode with >= 50 % effective mass in x) = 210 Hz", float(ok.items[0].value or 0.0), 210.0, 1e-12, "Hz"),
            Check("S2D first lateral frequency = 140 Hz (y, 92 %; the 150 Hz z mode is higher)", float(ok.items[1].value or 0.0), 140.0, 1e-12, "Hz"),
            Check("S2D both limits met, complete basis, confirmed -> PASS", float(ok.verdict == BG.PASS), 1.0, 0.0),
            Check("S2D axial limit above the first significant mode -> FAIL", float(out["low"].verdict == BG.FAIL), 1.0, 0.0),
            Check("S2D the same numbers with an unconfirmed requirement set -> UNCONFIRMED, never PASS", float(out["unconf"].verdict == BG.UNCONFIRMED), 1.0, 0.0),
            Check("S2D without a threshold f_1 = the lowest mode (95 Hz) and a limit of 90 Hz holds",
                  float(out["nothr"].items[0].value == 95.0 and out["nothr"].verdict == BG.PASS), 1.0, 0.0),
            Check("S2D without a launch axis no frequency item is judged", float(not out["noaxis"].items), 1.0, 0.0),
            Check("S2D unconfirmed launch environment (the axis) -> no PASS, no FAIL", float(out["env_unconf"].verdict == BG.UNCONFIRMED), 1.0, 0.0),
            Check("S2D split mass 30 % at 30 Hz + 30 % at 300 Hz (f_min 200 Hz): the axial item is UNCONFIRMED (60 % accounted), the verdict is not PASS",
                  float(out["split_30"].items[0].status == BG.UNCONFIRMED and out["split_30"].verdict != BG.PASS), 1.0, 0.0),
            Check("S2D split mass 49 % at 30 Hz + 49 % at 300 Hz: UNCONFIRMED (49 % under f_min without a significant mode), never PASS",
                  float(out["split_49"].items[0].status == BG.UNCONFIRMED and "effective mass" in out["split_49"].items[0].note and out["split_49"].verdict != BG.PASS), 1.0, 0.0),
            Check("S2D lateral axis with 45 % + 45 % at 20 / 25 Hz (f_min 120 Hz): the lateral item is UNCONFIRMED, never PASS",
                  float(out["split_lat"].items[1].status == BG.UNCONFIRMED and out["split_lat"].verdict != BG.PASS), 1.0, 0.0),
            Check("S2D a significant mode (55 %) under f_min = FAIL with that mode as the item value (30 Hz)",
                  float(out["split_fail"].items[0].status == BG.FAIL and out["split_fail"].items[0].value == 30.0 and out["split_fail"].verdict == BG.FAIL), 1.0, 0.0),
            Check("S2D a basis that ends under f_min and holds no axial mass -> UNCONFIRMED (never PASS, never FAIL, never skipped)",
                  float(out["short"].items[0].status == BG.UNCONFIRMED and out["short"].verdict == BG.UNCONFIRMED), 1.0, 0.0)]


SPACE_LOAD_CASES["S2D"] = Case("S2D", "Space loads: minimum natural frequency check and budget verdicts (arbitrary test limits)", _s2d_build, _s2d_eval, run=_s2d_run)


# ------------------------------------------------------------------ S2E: QSL generator
def _s2e_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    return steel_study("S2E QSL"), []


QSL_AX, QSL_LAT, QSL_LAT2 = 7.5, 3.25, 2.0                       # arbitrary test factors [g]


def _s2e_run(study: Study, workdir: Path, parts: Any) -> dict[str, Any]:
    out = _Out()
    for comb in ("none", "axial_lateral", "axial_both_laterals"):
        st = steel_study("S2E")
        st.space = space_with_limits({"qsl_axial_g": QSL_AX, "qsl_lateral_g": QSL_LAT, "qsl_lateral_2_g": QSL_LAT2})
        es = EV.environment_template()
        es.launch_axis, es.qsl_combination = "y", comb
        EV.store_environments(st.space, es)
        out[comb] = (st, LD.generate_qsl(st))
    return out


def _s2e_eval(out: dict[str, Any], study: Study) -> list[Check]:
    checks: list[Check] = []
    for comb, n in (("none", 6), ("axial_lateral", 14), ("axial_both_laterals", 14)):
        st, cases = out[comb]
        checks.append(Check(f"S2E {comb}: number of generated cases", float(len(cases)), float(n), 0.0))
        checks.append(Check(f"S2E {comb}: registered in the study", float(len(st.load_cases)), float(n), 0.0))
    st, cases = out["axial_lateral"]
    by = {c.name: c for c in cases}
    ax = by["QSL -Y"].factors
    checks += [Check("S2E axial -Y: factor = -axial level on the launch axis only", float(ax[1] == -QSL_AX and ax[0] == 0 and ax[2] == 0), 1.0, 0.0),
               Check("S2E lateral +X uses qsl_lateral_g", float(by["QSL +X"].factors[0]), QSL_LAT, 0.0, "g"),
               Check("S2E second lateral +Z uses qsl_lateral_2_g", float(by["QSL +Z"].factors[2]), QSL_LAT2, 0.0, "g"),
               Check("S2E combination +Y -X carries both factors", float(by["QSL +Y -X"].factors[1] == QSL_AX and by["QSL +Y -X"].factors[0] == -QSL_LAT), 1.0, 0.0)]
    g = next(c for c in st.load_cases if c.name == "QSL -Y").loads[0]
    checks.append(Check("S2E gravity vector = factor x 9810 mm/s2", float(g.vector[1]), -QSL_AX * TEST_G, 1e-12, "mm/s2"))
    LD.generate_qsl(st)                                                       # regenerate: replaced, not duplicated
    checks.append(Check("S2E regeneration replaces the generated cases", float(len(st.load_cases)), 14.0, 0.0))
    checks.append(Check("S2E the status of the generated cases is current (a regeneration gives the same names and gravity vectors)",
                        float(LD.qsl_status(st)["current_with_requirement_set"]), 1.0, 0.0))
    next(c for c in st.load_cases if c.name == "QSL -Y").loads[0].vector = [0.0, -0.5 * QSL_AX * TEST_G, 0.0]
    checks.append(Check("S2E an edited generated case makes the status non-current", float(not LD.qsl_status(st)["current_with_requirement_set"]), 1.0, 0.0))
    return checks


SPACE_LOAD_CASES["S2E"] = Case("S2E", "Space loads: signed QSL cases from the requirement-set factors (arbitrary test factors)", _s2e_build, _s2e_eval, run=_s2e_run)


# ------------------------------------------------------------------ S2F: the 3-axis run over ONE basis = three INDEPENDENT full single-axis analyses (real solver)
CANTILEVER_MM = (80.0, 12.0, 7.0)                                 # a steel cantilever clamped at x = 0 (arbitrary test geometry)
S2F_TOL = 1.0e-9                                                  # of the maximum; measured 8e-12 with CalculiX (single-precision fields). The direct engine (rslab, float32) needs ~1e-6


def cantilever_study(wd: Path, engine: str = "ccx") -> tuple[Study, list[Any], EV.EnvironmentSet]:
    """The cantilever study of S2F with the environments of `three_axis_level` (damping from the environments) and the modal `engine` (ccx | direct | auto)."""
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.study import Fixed, LoadCase, MeshSettings

    from .cases import _study, box
    wd.mkdir(parents=True, exist_ok=True)
    st, parts = _study("S2F cantilever", wd / "model.step", [Part("Konzola", box(0, 0, 0, *CANTILEVER_MM))], MeshSettings(size=4.0))
    st.add_selection("clamp", find_faces(parts, x=0))
    st.constraints.append(Fixed("clamp"))
    st.load_cases.append(LoadCase("F", []))
    st.solver.backend, st.solver.modal_engine = "auto", engine
    st.dynamics.enabled, st.dynamics.modes = True, 8
    st.dynamics.damping.zeta = 0.02
    st.space = SpaceSettings()
    es = EV.EnvironmentSet(levels=[three_axis_level()], launch_axis="z", damping_zeta=0.03, damping_source="test report")
    EV.store_environments(st.space, es)
    return st, parts, es


def _s2f_build(wd: Path):
    st, parts, _es = cantilever_study(wd)
    return st, parts


def _s2f_run(study: Study, workdir: Path, parts: Any) -> dict[str, Any]:
    """`run_3axis` (one modal basis and one set of unit-acceleration statics for the three axes) against three COMPLETE single-axis `run_study` runs, each of which
    computes its own modal basis and its own statics with the real solver (nothing is shared)."""
    from fea_core.solve import prepare_mesh, run_study
    es = EV.environments_of(study.space)
    assert es is not None
    mesh, _ = prepare_mesh(study, workdir, workdir, lambda s: None, parts=parts)
    r3 = LD.run_3axis(study, mesh, workdir / "run", lambda s: None, level="qualification")
    f_ext = float(r3.freq.max()) * 1.01                           # the shock table is continued at its ZPA above its last point, as the space path does
    kinds = ("sine", "random", "shock")
    dev: dict[str, float] = {}
    for a in (0, 1, 2):
        s1 = LD._axis_study(study, es.levels[0], list(kinds), a, srs_extend_to=f_ext)
        d = run_study(s1, workdir / f"independent_{a}", mesh=mesh, parts=parts).dynamics
        ref = {"sine": np.asarray(d.harmonic["env_S"], np.float32), "random": np.asarray(d.random["S_rms"], np.float32), "shock": np.asarray(d.shock["mises"], np.float32)}
        for k in kinds:
            got = r3.per_axis[k][a].S.astype(np.float64)
            dev[f"{k} S axis {a}"] = float(np.abs(got - ref[k]).max() / max(float(ref[k].max()), 1e-12))
        ref_u = np.linalg.norm(np.asarray(d.random["U_rms"], float), axis=1).astype(np.float32).astype(np.float64)
        dev[f"random U axis {a}"] = float(np.abs(r3.per_axis["random"][a].U.astype(np.float64) - ref_u).max() / max(float(ref_u.max()), 1e-12))
    return _Out(r3=r3, dev=dev, engine=r3.engine, modes=len(r3.freq))


def _s2f_eval(out: dict[str, Any], study: Study) -> list[Check]:
    r3 = out["r3"]
    checks = [Check("S2F the basis has several modes and the run is a complete 3-axis run", float(out["modes"] >= 3 and r3.complete), 1.0, 0.0)]
    checks += [Check(f"S2F {name}: relative deviation from the independent single-axis analysis (own basis, own statics)", dev, 0.0, S2F_TOL) for name, dev in out["dev"].items()]
    env = r3.envelopes["random"]
    checks.append(Check("S2F the random envelope = the elementwise maximum of the three per-axis fields",
                        float(np.array_equal(env.S, np.max(np.stack([r3.per_axis["random"][a].S for a in (0, 1, 2)]), axis=0))), 1.0, 0.0))
    return checks


SPACE_LOAD_CASES["S2F"] = Case("S2F", "Space loads: the 3-axis run over one basis agrees with three independent single-axis analyses (real solver, a cantilever)", _s2f_build, _s2f_eval,
                               run=_s2f_run)

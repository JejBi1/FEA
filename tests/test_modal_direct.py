"""Natural frequencies on the direct path (F14 S0): the own mass matrix, the shift-invert Lanczos iteration with the Sturm proof of completeness, the modes of real
meshes against CalculiX on the SAME mesh. Nothing of the chain is mocked except where a failure has to be forced (a missed mode, a failing analysis): a real Gmsh
mesh, the assembly worker (numba), the eigenvalue worker (rslab + ARPACK), the stress recovery, ccx.

Gate (the task of S0): frequencies ≤ 1e-5 of CalculiX (the 7 digits of its output), effective masses ≤ 1e-4, MAC ≥ 0.9999 (a repeated or clustered eigenvalue as a subspace).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh
from fea_core.study import ElasticSupport, Fixed, Material, MeshSettings
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")
needs_chain = pytest.mark.skipif(not (HAS_DIRECT and config.CCX_EXE.exists()), reason="numba + rslab (modules accel, direct) and CalculiX are needed")


@pytest.fixture(scope="module", autouse=True)
def _rslab_on_the_path_of_the_worker():
    """The workers are separate processes: where rslab is found by this process (a development copy outside the site-packages), they must find it too."""
    mp = pytest.MonkeyPatch()
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        mp.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    yield
    mp.undo()


# ------------------------------------------------------------------ a clamped and a free beam
L_B, W_B, H_B = 120.0, 20.0, 10.0


def _beam(wd: Path, constrained: bool, size: float = 4.0, name: str = "beam"):
    st, parts = _study(name, wd / "model.step", [Part("Nosnik", box(0, 0, 0, L_B, W_B, H_B))], MeshSettings(size=size))
    if constrained:
        st.add_selection("vetknuti", find_faces(parts, x=0))
        st.constraints.append(Fixed("vetknuti"))
    return st, parts


def ccx_modes(st, mesh, wd: Path, tries: int = 5):
    """The ModeSet of CalculiX. The ccx_static.exe of this machine sometimes ends at its start with an access violation (3221225477) while other programs hold the
    memory – the same input runs without a fault when repeated outside the test (checked by hand), so the reference is repeated with a pause; when it keeps failing the
    reference is not available and the test is skipped (never the program under test)."""
    import time
    from fea_core.ccx.runner import CcxError
    from fea_core.eigen import run_eigen
    for attempt in range(tries):
        try:
            return run_eigen(st, mesh, wd, lambda s: None)[0]
        except CcxError as e:
            if "3221225477" not in str(e):
                raise
            time.sleep(3.0)
    pytest.skip("CalculiX ends with an access violation on this machine at the moment (the reference of the comparison is not available)")


def _ccx_and_direct(st, parts, wd: Path, modes: int, mesh=None):
    """(the ModeSet of CalculiX, the ModeSet of the direct path, the mesh) on one mesh."""
    from fea_core.eigen import run_eigen
    st.modal.enabled = True
    st.modal.modes = modes
    if mesh is None:
        mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.solver.modal_engine = "ccx"
    ref = ccx_modes(st, mesh, wd / "ccx")
    st.solver.modal_engine = "direct"
    own = run_eigen(st, mesh, wd / "direct", lambda s: None)[0]
    st.solver.modal_engine = "auto"
    return ref, own, mesh


def _gate(ref, own, strict=(1e-5, 1e-4, 0.9999), loose=(3e-4, 1e-3, 0.999), first=5):
    """The direct modes against ccx on the same mesh. The gate of the task (frequencies 1e-5, effective masses 1e-4, MAC 0.9999) holds for the lowest modes; of the higher
    ones ccx itself is only as accurate as its ARPACK run – measured against the DENSE solution of the same K and M (`scratch` / `test_the_direct_modes_are_the_exact_
    eigenpairs_of_the_discrete_model`): ccx 1.6e-4 in the frequency and 4.6e-4 in the effective mass of 12 modes, the direct path 7e-10 and 3e-12. So the first `first`
    compared modes are held to the strict tolerances, all of them to the loose ones (the ccx accuracy)."""
    from fea_core.eigen import compare_modes
    c = compare_modes(own.values, own.shapes, own.participation, ref.values, ref.shapes, ref.participation, total=ref.total_mass)
    assert c["n"] > 0
    fr, em, sub = c["freq_rel"], c["eff_mass_rel"], c["subspace"]
    assert max(fr[:first]) <= strict[0], ("frequencies of the lowest modes", fr[:first])
    assert max(em[:first]) <= strict[1], ("effective masses of the lowest modes", em[:first])
    assert min(sub[:first]) >= strict[2], ("MAC of the lowest modes", sub[:first])
    assert max(fr) <= loose[0] and max(em) <= loose[1] and min(sub) >= loose[2], (fr, em, sub)
    return c


@pytest.fixture(scope="module")
def clamped(tmp_path_factory):
    wd = tmp_path_factory.mktemp("modal_clamped")
    st, parts = _beam(wd, True)
    return (st, parts, wd) + _ccx_and_direct(st, parts, wd, 12)


@pytest.fixture(scope="module")
def free(tmp_path_factory):
    wd = tmp_path_factory.mktemp("modal_free")
    st, parts = _beam(wd, False, name="beam free")
    return (st, parts, wd) + _ccx_and_direct(st, parts, wd, 14)


@needs_chain
def test_a_clamped_beam_equals_ccx_in_frequencies_effective_masses_and_shapes(clamped):
    st, parts, wd, ref, own, mesh = clamped
    c = _gate(ref, own)
    assert own.engine == "direct" and ref.engine == "ccx" and c["n"] == 12
    assert own.engine_info["sturm_checks"][-1]["count"] == 12 and own.engine_info["max_residual"] < 1e-8
    # the participation factors Γ (x y z rx ry rz) are those of ccx up to the sign of the mode
    g, g_ref = np.abs(np.array(own.participation)), np.abs(np.array(ref.participation))
    assert np.abs(g[:, :3] - g_ref[:, :3]).max() <= 1e-3 * g_ref[:, :3].max()


@needs_chain
def test_a_free_beam_has_six_rigid_modes_at_zero_and_the_elastic_ones_equal_ccx(free):
    st, parts, wd, ref, own, mesh = free
    assert np.all(np.array(own.values[:6]) == 0.0) and min(own.values[6:]) > 100.0
    c = _gate(ref, own)
    assert c["n"] == 8                                         # 14 modes, the 6 rigid ones are not compared (ccx leaves out the negative ones)
    assert own.total_mass[0] == pytest.approx(7.85e-9 * L_B * W_B * H_B, rel=1e-9)
    # the six rigid modes carry the whole mass: the sum of their effective masses is the total mass in x, y and z (the participation of a free body)
    em = np.array(own.eff_mass)[:6, :3].sum(axis=0)
    assert np.allclose(em, own.total_mass[:3], rtol=1e-6)


@needs_chain
def test_ccx_modes_of_a_free_body_are_paired_with_their_shapes_by_order(free):
    """The participation factors and the shapes of ccx are numbered consecutively while its eigenvalue table skips the modes with a negative eigenvalue: the shape
    of the mode with 1318 Hz must be that mode (an earlier reading shifted the shapes of a free body by the number of the skipped modes)."""
    st, parts, wd, ref, own, mesh = free
    from fea_core.eigen import mac_matrix
    f_ref = np.array(ref.values)
    for j in range(6, len(f_ref)):
        m = mac_matrix(own.shapes[j:j + 1], ref.shapes[j:j + 1])[0, 0]
        assert m > 0.9999, (j, f_ref[j], m)


@needs_direct
def test_the_direct_modes_are_the_exact_eigenpairs_of_the_discrete_model(tmp_path):
    """The reference without ccx: dense generalized eigenvalue problem of the same K and M (a coarse mesh, 1500 DOF) – the frequencies to 1e-10, the shapes to the
    rounding of the iteration, free and clamped."""
    import scipy.linalg as sla
    import scipy.sparse as sp
    from fea_core.assembly import kernels as K, mass as asm_mass
    from fea_core.assembly.system import assemble
    from fea_core.direct import eigen as E
    for constrained in (True, False):
        wd = tmp_path / ("c" if constrained else "f")
        wd.mkdir()
        st, parts = _beam(wd, constrained, size=12.0)
        st.modal.enabled = True
        mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
        n = len(mesh.nodes)
        fixed = np.zeros((n, 3), bool)
        if constrained:
            fixed[mesh.surface_nodes("vetknuti")] = True
        Cm = np.stack([K.iso_stiffness(st.material_of(p).E, st.material_of(p).nu) for p in mesh.part_names])
        s = assemble(mesh.nodes, mesh.tets, mesh.tet_part, Cm, fixed.ravel(), np.zeros(3 * n), np.zeros((3 * n, 1)), upper=True,
                     mass=asm_mass.build(st, mesh, None)).to_system()
        n3 = 3 * n
        Ku = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n3, n3))
        Mu = sp.csr_matrix((s.mdata, s.indices, s.indptr), shape=(n3, n3))
        Kf, Mf = (Ku + sp.triu(Ku, 1).T).toarray(), (Mu + sp.triu(Mu, 1).T).toarray()
        fr = np.diag(Mf) > 0
        w, V = sla.eigh(Kf[np.ix_(fr, fr)], Mf[np.ix_(fr, fr)])
        info: dict = {}
        r = E.solve_modes(Ku, Mu, E.EigenOptions(modes=12, threads=2), info, log=lambda s: None)
        el = slice(0, 12) if constrained else slice(6, 12)
        lam_r = r.lam[el] if constrained else r.lam[6:12]
        ref = w[:12] if constrained else w[6:12]
        assert np.abs(lam_r - ref).max() <= 1e-10 * ref.max()
        if not constrained:
            assert info["rigid"] == [0, 1, 2, 3, 4, 5] and np.abs(r.lam[:6]).max() < 1e-9 * r.lam[6]
        # the same modes as the dense ones (subspace; the modes are simple here): the shapes to the rounding of the iteration
        Xr = r.X[fr][:, (slice(0, 12) if constrained else slice(6, 12))]
        Vd = V[:, :12] if constrained else V[:, 6:12]
        assert np.abs(np.abs(np.einsum("ij,ij->j", Xr, Mf[np.ix_(fr, fr)] @ Vd)) - 1.0).max() < 1e-8
        # the window f_min … f_max [Hz] of a model (Study.modal.f_min / f_max): exactly the modes in it, counted by the inertia at both ends
        if constrained:
            f_all = np.sqrt(w) / (2 * np.pi)
            lo, hi = 0.5 * (f_all[2] + f_all[3]), 0.5 * (f_all[8] + f_all[9])
            info2: dict = {}
            r2 = E.solve_modes(Ku, Mu, E.EigenOptions(modes=50, lam_min=(2 * np.pi * lo) ** 2, lam_max=(2 * np.pi * hi) ** 2, threads=2), info2, log=lambda s: None)
            assert len(r2.lam) == 6 and info2["n_below_window"] == 3
            assert np.abs(r2.lam - w[3:9]).max() <= 1e-9 * w[8]


def test_read_dat_pairs_the_rows_of_a_table_that_skips_modes_by_order(tmp_path):
    from fea_core.eigen import read_dat
    eig = "\n".join(f"{j:7d}   0.1E+05   0.1E+03   {10.0 * j:.6E}   0.0" for j in (3, 4, 5))
    part = "\n".join(f"{r:7d}   {r}.0E-03  0.0  0.0  0.0  0.0  0.0" for r in (1, 2, 3))
    mass = "\n".join(f"{r:7d}   {r}.0E-06  0.0  0.0  0.0  0.0  0.0" for r in (1, 2, 3))
    text = (" E I G E N V A L U E   O U T P U T\n\n MODE NO    EIGENVALUE\n\n\n" + eig + "\n\n P A R T I C I P A T I O N   F A C T O R S\n\nMODE NO.\n\n" + part
            + "\n\n E F F E C T I V E   M O D A L   M A S S\n\nMODE NO.\n\n" + mass + "\n")
    p = tmp_path / "x.dat"
    p.write_text(text, encoding="latin-1")
    d = read_dat(p, "modal")
    assert d["modes"] == [3, 4, 5]
    assert d["freq"] == [0.0, 0.0, 30.0, 40.0, 50.0]
    assert [row[0] for row in d["participation"]] == [0.0, 0.0, 1e-3, 2e-3, 3e-3]          # the first row of the table belongs to mode 3
    assert [row[0] for row in d["eff_mass"]] == [0.0, 0.0, 1e-6, 2e-6, 3e-6]


# ------------------------------------------------------------------ the mass matrix
@needs_direct
def test_the_mass_matrix_equals_the_consistent_mass_of_the_model_and_does_not_depend_on_the_threads(tmp_path):
    import numba
    import scipy.sparse as sp
    from fea_core.assembly import kernels as K, mass as asm_mass
    from fea_core.assembly.system import assemble, rigid_motions
    from fea_core.dynamics.mass import mass_times, total_mass
    st, parts = _beam(tmp_path, True, size=8.0)
    st.modal.enabled = True
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    n = len(mesh.nodes)
    fixed = np.zeros((n, 3), bool)
    fixed[mesh.surface_nodes("vetknuti")] = True
    Cm = np.stack([K.iso_stiffness(st.material_of(p).E, st.material_of(p).nu) for p in mesh.part_names])
    mi = asm_mass.build(st, mesh, None)
    out = {}
    old = numba.get_num_threads()
    try:
        for th in (1, min(4, old)):
            numba.set_num_threads(th)
            a = assemble(mesh.nodes, mesh.tets, mesh.tet_part, Cm, fixed.ravel(), np.zeros(3 * n), np.zeros((3 * n, 1)), upper=True, mass=mi)
            out[th] = (a.mdata.copy(), a.data.copy(), a)
    finally:
        numba.set_num_threads(old)
    ths = sorted(out)
    assert all(np.array_equal(out[ths[0]][i], out[ths[-1]][i]) for i in (0, 1)), "the mass and stiffness matrices depend on the number of threads"
    a = out[ths[-1]][2]
    s = a.to_system()
    n3 = 3 * n
    Mu = sp.csr_matrix((s.mdata, s.indices, s.indptr), shape=(n3, n3))
    Mf = Mu + sp.triu(Mu, 1).T
    U = np.random.default_rng(3).standard_normal((n, 3))
    U[mesh.surface_nodes("vetknuti")] = 0.0
    free = ~fixed.ravel()
    ref = mass_times(st, mesh, U).ravel()
    assert np.abs((Mf @ U.ravel())[free] - ref[free]).max() <= 1e-12 * np.abs(ref).max()
    assert a.info["mass"]["bodies_t"] == pytest.approx(total_mass(st, mesh), rel=1e-12)
    # the constrained rows and columns are zero (no spurious mode), the diagonal too; the unreduced M·R: the total mass of the free DOFs
    assert Mf.diagonal()[~free].max() == 0.0 and abs(Mf[~free][:, :]).sum() == 0.0
    R = rigid_motions(mesh.nodes, np.zeros((0, 3)), fixed.ravel())
    assert np.allclose(a.MR[free], (Mf @ R)[free], rtol=1e-10, atol=1e-18)
    assert a.info["mass"]["total_free"][0] == pytest.approx(R[:, 0] @ a.MR[:, 0], rel=1e-12)


# ------------------------------------------------------------------ the eigenvalue algorithm
def _laplace3(m: int):
    """3D Laplacian on an m³ grid (the upper triangle in CSR) and the identity as mass on the same pattern: the eigenvalues i²+j²+k² are repeated exactly."""
    import scipy.sparse as sp
    A = sp.diags([2 * np.ones(m), -np.ones(m - 1), -np.ones(m - 1)], [0, 1, -1]).tocsr()
    I = sp.identity(m, format="csr")
    K3 = (sp.kron(sp.kron(A, I), I) + sp.kron(sp.kron(I, A), I) + sp.kron(sp.kron(I, I), A)) * 1e6
    n = K3.shape[0]
    KU = sp.triu(K3).tocsr()
    KU.sort_indices()
    rows = np.repeat(np.arange(n), np.diff(KU.indptr))
    MU = sp.csr_matrix(((rows == KU.indices).astype(float), KU.indices, KU.indptr), shape=(n, n))
    return KU, MU, np.sort(np.linalg.eigvalsh(K3.toarray()))


@needs_direct
def test_repeated_eigenvalues_are_all_found_and_a_cut_inside_a_cluster_is_noticed():
    """10³ grid, 40 modes: the 40th lies inside the triple 1.877e6 – the Sturm count says 41 below the bound, the analysis completes the cluster and returns the lowest 40."""
    from fea_core.direct import eigen as E
    KU, MU, lam = _laplace3(10)
    info: dict = {}
    r = E.solve_modes(KU, MU, E.EigenOptions(modes=40, threads=2), info, log=lambda s: None)
    assert len(r.lam) == 40 and np.abs(r.lam - lam[:40]).max() <= 1e-11 * lam[39]
    assert [x["count"] for x in info["sturm"]] == [41, 41] and info["arpack_passes"] == 2
    G = r.X.T @ (MU @ r.X + MU.T @ r.X - MU.diagonal()[:, None] * r.X)
    assert np.abs(G - np.eye(40)).max() < 1e-10 and info["max_residual"] < 1e-8


@needs_direct
def test_a_missed_mode_is_found_by_the_deflated_iteration(monkeypatch):
    """The library is made to MISS one eigenpair of the first pass (what a Lanczos method started from one vector can do with a repeated eigenvalue): the Sturm count
    sees the difference and the second pass finds the mode with the found ones deflated. Without the check the set would silently lack a mode."""
    import scipy.sparse.linalg as sla
    from fea_core.direct import eigen as E
    KU, MU, lam = _laplace3(10)
    real = sla.eigsh
    calls = {"n": 0}

    def eigsh_missing_one(*a, **kw):
        vals, vecs = real(*a, **kw)
        calls["n"] += 1
        if calls["n"] == 1:
            order = np.argsort(vals)
            drop = order[5]                                    # a mode inside the triple 4–6 (a clean cut after the 11th mode, which is simple)
            keep = np.ones(len(vals), bool)
            keep[drop] = False
            return vals[keep], vecs[:, keep]
        return vals, vecs
    monkeypatch.setattr(sla, "eigsh", eigsh_missing_one)
    info: dict = {}
    r = E.solve_modes(KU, MU, E.EigenOptions(modes=11, threads=2), info, log=lambda s: None)
    assert calls["n"] == 2 and info["sturm"][0]["count"] == 11 and info["sturm"][0]["found"] == 10 and info["sturm"][1]["count"] == 11
    assert np.abs(r.lam - lam[:11]).max() <= 1e-11 * lam[10]
    assert info["max_residual"] < 1e-9


@needs_direct
def test_the_same_input_gives_the_same_modes_and_another_seed_the_same_eigenpairs():
    """The start vector is fixed (a seeded generator): a repeated run repeats (to the rounding of the threaded factorization), another seed finds the same eigenvalues and
    the same eigenspaces (cluster sums of the projectors – the vectors of a repeated eigenvalue may differ by a rotation)."""
    from fea_core.direct import eigen as E
    KU, MU, lam = _laplace3(10)
    a = E.solve_modes(KU, MU, E.EigenOptions(modes=11, threads=2), {}, log=lambda s: None)
    b = E.solve_modes(KU, MU, E.EigenOptions(modes=11, threads=2), {}, log=lambda s: None)
    assert np.abs(a.lam - b.lam).max() <= 1e-12 * lam[10]
    assert np.abs(np.abs(np.einsum("ij,ij->j", a.X, b.X)) - 1.0).max() < 1e-6           # simple modes (index 0, 10): the same vectors … all of them:
    c = E.solve_modes(KU, MU, E.EigenOptions(modes=11, threads=2, seed=99), {}, log=lambda s: None)
    assert np.abs(c.lam - a.lam).max() <= 1e-11 * lam[10]
    P1, P2 = a.X @ a.X.T, c.X @ c.X.T                    # the projector onto the 11 modes is a property of the problem, not of the start vector
    assert np.abs(P1 - P2).max() < 1e-8


@needs_direct
def test_a_set_with_a_duplicate_is_an_error_not_a_result(monkeypatch):
    """A mode returned twice (more modes found than the count allows) must not pass."""
    import scipy.sparse.linalg as sla
    from fea_core.direct import eigen as E
    KU, MU, lam = _laplace3(8)
    real = sla.eigsh

    def eigsh_duplicate(*a, **kw):
        vals, vecs = real(*a, **kw)
        # replace the largest eigenvalue by a copy of the smallest (a "ghost"): the Sturm count sees one mode less than it has found? no – the count equals the
        # number of distinct modes: the set has k entries of which one repeats, so the count at the top is the same – the orthonormality check refuses the copy
        vecs = vecs.copy()
        vecs[:, -1] = vecs[:, 0]
        vals = vals.copy()
        vals[-1] = vals[0]
        return vals, vecs
    monkeypatch.setattr(sla, "eigsh", eigsh_duplicate)
    with pytest.raises(E.EigenFail):
        E.solve_modes(KU, MU, E.EigenOptions(modes=10, threads=2), {}, log=lambda s: None)


@needs_direct
def test_the_window_f_min_f_max_returns_exactly_the_modes_in_it():
    """Lower and upper bound of the window: the Sturm count at both ends gives the number of modes, the iteration finds them (the bounds lie between distinct
    eigenvalues; a bound that sits on an eigenvalue is moved – next test)."""
    from fea_core.direct import eigen as E
    KU, MU, lam = _laplace3(10)
    uniq = np.unique(np.round(lam, 2))
    lo, hi = 0.5 * (uniq[3] + uniq[4]), 0.5 * (uniq[10] + uniq[11])
    want = lam[(lam > lo) & (lam < hi)]
    info: dict = {}
    r = E.solve_modes(KU, MU, E.EigenOptions(modes=100, lam_min=lo, lam_max=hi, threads=2), info, log=lambda s: None)
    assert info["n_below_window"] == int((lam < lo).sum()) and len(r.lam) == len(want) > 10
    assert np.abs(r.lam - want).max() <= 1e-9 * want.max() and info["max_residual"] < 1e-8
    with pytest.raises(E.EigenFail, match="no natural frequency"):
        E.solve_modes(KU, MU, E.EigenOptions(modes=5, lam_min=lam[12] * 1.0000001, lam_max=lam[12] * 1.0000002, threads=2), {}, log=lambda s: None)


@needs_direct
def test_a_bound_on_an_eigenvalue_is_moved_not_trusted_and_the_gap_between_is_empty():
    """A shift exactly on a (repeated) eigenvalue gives a singular K − σM that the LDLᵀ factorization does not report – the solve does not reproduce its right-hand side
    (relative residual 3e-2 on this 10³ grid); the count moves the bound by a fraction of σ until it does, and proves by the count that NO eigenvalue lies between the requested and
    the moved bound (the old rule moved by 1e-7 of the LARGEST scale of the model – far above the low modes of a slender one – and the modes it jumped over were counted as
    "below the window": the proof of completeness passed silently). The six modes of the eigenvalue at σ itself are on the side the inertia puts them, the count at the moved
    shift is the number of eigenvalues below it."""
    from fea_core.direct import eigen as E
    KU, MU, lam = _laplace3(10)
    sig = float(lam[12])                                        # the 6-fold eigenvalue 1.0888e6 (modes 12–17)
    group = np.abs(lam - sig) <= 1e-6 * sig                     # the dense solution scatters the six copies by 1e-12
    assert group.sum() == 6
    for side in (-1, 1):
        info: dict = {}
        sh = E.Shifted(KU, MU, E.EigenOptions(modes=5, threads=2), info, E.reference_scale(KU, MU))
        sh.factor(sig)
        assert sh.probe() > sh.probe_limit()                    # the singular shifted matrix
        n = sh.count_below(sig, solves=True, side=side)         # a factor that will solve
        used = float(sh.sigma)
        assert used != sig and sh.probe() <= sh.probe_limit()
        assert abs(used - sig) <= E.MOVE_MAX * sig              # a small relative move
        lo, hi = sorted((used, sig))
        assert not ((lam > lo) & (lam < hi) & ~group).any()     # the gap between the requested and the moved bound is EMPTY (the group sits at σ, not between)
        assert n == int((lam < used).sum()) and n in (11, 17)   # all six of the group on one side of the moved shift
        assert info["bounds"][-1]["requested"] == sig and info["bounds"][-1]["used"] == used


@needs_direct
def test_a_bound_between_two_eigenvalues_closer_than_the_resolution_widens_the_window_by_a_documented_rule():
    """σ halfway between 5(1 − 1e-15) and 5(1 + 1e-15): no usable shift on either side of σ is free of an eigenvalue (the inertia and the probe cannot tell the two apart) –
    the first usable OUTWARD shift is taken and the eigenvalues between it and σ count as inside (the window is widened by at most MOVE_MAX·σ); never a skipped mode."""
    import scipy.sparse as sp
    from fea_core.direct import eigen as E
    lams = np.array([1, 2, 3, 4, 5 * (1 - 1e-15), 5 * (1 + 1e-15), 7, 8, 9, 10, 11, 12.0])
    KU, MU = sp.diags(lams).tocsr(), sp.identity(len(lams), format="csr")
    for side, expected in ((-1, 4), (1, 6)):
        info: dict = {}
        sh = E.Shifted(KU, MU, E.EigenOptions(modes=3, threads=1), info, E.reference_scale(KU, MU))
        n = sh.count_below(5.0, solves=True, side=side)
        assert n == expected                                    # lower bound: both partners of the pair are in the window; upper bound: both are below it
        b = info["bounds"][-1]
        assert abs(b["used"] - 5.0) <= E.MOVE_MAX * 5.0 and (b["used"] < 5.0) == (side < 0) and (b["eigenvalues_between"] or 1) >= 1


@needs_direct
def test_no_usable_bound_within_the_limit_is_an_error_not_a_guess():
    """A singular probe at every shift within MOVE_MAX·σ (forced: the limit is below any residual) ends in `EigenFail`, never in a count."""
    import scipy.sparse as sp
    from fea_core.direct import eigen as E
    lams = np.arange(1.0, 13.0)
    KU, MU = sp.diags(lams).tocsr(), sp.identity(len(lams), format="csr")
    sh = E.Shifted(KU, MU, E.EigenOptions(modes=3, threads=1), {}, E.reference_scale(KU, MU))
    sh.probe_limit = lambda: -1.0                               # nothing passes
    with pytest.raises(E.EigenFail, match="could not be determined"):
        sh.count_below(5.5, solves=True, side=-1)


@needs_direct
def test_a_bound_exactly_on_an_eigenvalue_that_the_engine_refuses_is_moved():
    """σ = 5 on the eigenvalue 5 of diag(1…12): the engine refuses the factorization (numerically rank-deficient) – the bound moves outward instead of failing."""
    import scipy.sparse as sp
    from fea_core.direct import eigen as E
    lams = np.arange(1.0, 13.0)
    KU, MU = sp.diags(lams).tocsr(), sp.identity(len(lams), format="csr")
    info: dict = {}
    sh = E.Shifted(KU, MU, E.EigenOptions(modes=3, threads=1), info, E.reference_scale(KU, MU))
    assert sh.count_below(5.0, solves=True, side=-1) == 4 and sh.sigma < 5.0           # the eigenvalue 5 is inside the window
    assert sh.count_below(5.0, solves=False, side=1) == 5 and sh.sigma > 5.0           # … and below the upper bound


def test_the_probe_limit_follows_the_size_of_the_system(monkeypatch):
    """A singular K − σM leaves |g|/√n in a probe solve, so the limit is PROBE_SAT/√n (floor PROBE_TOL); an environment value fixes it."""
    from fea_core.direct import eigen as E

    class Fake(E.Shifted):
        def __init__(self, n):
            self.n = n
    assert Fake(1000).probe_limit() == pytest.approx(E.PROBE_SAT / np.sqrt(1000))
    assert Fake(40521).probe_limit() == pytest.approx(E.PROBE_SAT / np.sqrt(40521))
    assert Fake(10 ** 12).probe_limit() == E.PROBE_TOL                                    # never below the floor
    monkeypatch.setattr(E, "PROBE_FIXED", True)
    assert Fake(1000).probe_limit() == E.PROBE_TOL


# ------------------------------------------------------------------ a slender model: the windows of the review, against the dense solution
SL_L, SL_B, SL_H, SL_SIZE = 240.0, 2.0, 2.0, 12.0


def _slender_system(wd: Path, constrained: bool):
    """K, M (upper triangles, CSR) of a slender bar (1134 DOF, L/h = 120) and the dense eigenvalues of the same pencil. λ_ref/λ₁ = 1.7e9 clamped, 4e7 free: the old rule of the
    moved bound (1e-7·min K_ii/M_ii = 170 λ₁ / 4 λ₁) jumps over the low modes."""
    import scipy.linalg as sla
    import scipy.sparse as sp
    from fea_core.assembly import kernels as K, mass as asm_mass
    from fea_core.assembly.system import assemble
    st, parts = _study("slender", wd / "model.step", [Part("Pret", box(0, 0, 0, SL_L, SL_B, SL_H))], MeshSettings(size=SL_SIZE))
    if constrained:
        st.add_selection("vetknuti", find_faces(parts, x=0))
        st.constraints.append(Fixed("vetknuti"))
    st.modal.enabled = True
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    n = len(mesh.nodes)
    fixed = np.zeros((n, 3), bool)
    if constrained:
        fixed[mesh.surface_nodes("vetknuti")] = True
    Cm = np.stack([K.iso_stiffness(st.material_of(p).E, st.material_of(p).nu) for p in mesh.part_names])
    s = assemble(mesh.nodes, mesh.tets, mesh.tet_part, Cm, fixed.ravel(), np.zeros(3 * n), np.zeros((3 * n, 1)), upper=True, mass=asm_mass.build(st, mesh, None)).to_system()
    n3 = 3 * n
    Ku = sp.csr_matrix((s.data, s.indices, s.indptr), shape=(n3, n3))
    Mu = sp.csr_matrix((s.mdata, s.indices, s.indptr), shape=(n3, n3))
    Kf, Mf = (Ku + sp.triu(Ku, 1).T).toarray(), (Mu + sp.triu(Mu, 1).T).toarray()
    fr = np.diag(Mf) > 0
    w = sla.eigh(Kf[np.ix_(fr, fr)], Mf[np.ix_(fr, fr)], eigvals_only=True)
    return Ku, Mu, w


@pytest.fixture(scope="module")
def slender(tmp_path_factory):
    return {c: _slender_system(tmp_path_factory.mktemp("slender_c" if c else "slender_f"), c) for c in (True, False)}


@needs_direct
@pytest.mark.parametrize("rel_f", [0.995, 0.9987, 1.0044], ids=["f1_minus_0.5pct", "f1_minus_0.13pct", "f1_plus_0.44pct"])
def test_a_window_just_below_or_above_the_first_mode_of_a_slender_clamped_model_equals_the_dense_modes(slender, rel_f):
    """The review: a clamped strip (f₁ = 10.454 Hz) with f_min 10.4 / 10.44 / 10.5 Hz returned the modes from 593 Hz, seven missing, and the proof passed. Here the same
    relative positions of the bound (−0.5 %, −0.13 %, +0.44 % of f₁) on a slender bar small enough for the dense reference: exactly the modes of the window, the count below it
    equal to the dense one, the bound not moved (or by less than MOVE_MAX)."""
    from fea_core.direct import eigen as E
    Ku, Mu, w = slender[True]
    assert 1e-7 * E.reference_scale(Ku, Mu) > 2.0 * w[0]                    # the model is slender enough for the old rule to fail
    lam_lo = float(w[0] * rel_f ** 2)
    info: dict = {}
    r = E.solve_modes(Ku, Mu, E.EigenOptions(modes=6, lam_min=lam_lo, threads=2), info, log=lambda s: None)
    ref = w[w >= lam_lo][:6]
    assert info["n_below_window"] == int((w < lam_lo).sum()) and len(r.lam) == 6
    assert np.abs(r.lam - ref).max() <= 1e-8 * ref.max()
    assert info["sturm"][-1]["count"] == 6 and info["sturm"][-1]["found"] == 6
    assert all(abs(b["used"] - b["requested"]) <= E.MOVE_MAX * b["requested"] for b in info.get("bounds", []))


@needs_direct
@pytest.mark.parametrize("f_min", [0.001, 0.1, 1.0])
def test_a_window_of_a_free_slender_model_starts_above_its_rigid_body_modes(slender, f_min):
    """The review: a free strip with f_min 0.001 / 0.1 / 1 Hz, 6 modes, returned 593 Hz and above, five modes (66.5, 183, 199, 359, 548 Hz) missing. A free body has six rigid-body
    modes at λ = 0 ± rounding noise (±1e-3 here, 2 of the 6 above σ = λ(0.001 Hz)): the window starts above them (raised to the rigid-body limit when f_min lies inside the noise), so it
    returns the six lowest ELASTIC modes in all three cases."""
    from fea_core.direct import eigen as E
    Ku, Mu, w = slender[False]
    assert 1e-7 * E.reference_scale(Ku, Mu) > 2.0 * w[6]
    info: dict = {}
    r = E.solve_modes(Ku, Mu, E.EigenOptions(modes=6, lam_min=(2 * np.pi * f_min) ** 2, threads=2, free=True), info, log=lambda s: None)
    assert info["n_below_window"] == 6 and len(r.lam) == 6 and not r.rigid.any()
    assert np.abs(r.lam - w[6:12]).max() <= 1e-8 * w[11]
    assert ("window_raised" in info) == ((2 * np.pi * f_min) ** 2 < E.rigid_limit(E.Sym(Ku), Mu))
    assert info["sturm"][-1]["count"] == 6


@needs_direct
def test_the_residual_report_says_what_was_checked(slender):
    """`max_residual` of the report was 9.3e-6 and passed because the floor 1e-12·‖K‖₁‖φ‖ dominates 1e-8·max(‖Kφ‖, λ‖Mφ‖). The info now carries the criterion, the largest relative
    residual of the elastic modes, the largest backward error, the number of modes accepted only by the backward error, and the largest ratio to the limit."""
    from fea_core.direct import eigen as E
    Ku, Mu, w = slender[True]
    info: dict = {}
    r = E.solve_modes(Ku, Mu, E.EigenOptions(modes=6, threads=2), info, log=lambda s: None)
    rep = info["residual"]
    assert rep["modes"] == 6 and rep["max_of_limit"] <= 1.0 and rep["max_backward_error"] <= E.BACKWARD_FLOOR
    assert info["max_residual"] == rep["max_relative"] > E.RES_TOL                            # a slender model cannot reach 1e-8 relative: the backward error decides
    assert rep["accepted_by_backward_error"] >= 1 and "1e-12" in rep["criterion"] and "1e-08" in rep["criterion"]
    # the same pair of numbers recomputed from the returned modes (the report is the check, not a copy of a constant)
    Ks, Ms = E.Sym(Ku), E.Sym(Mu)
    KX, MX = Ks.mv(r.X), Ms.mv(r.X)
    rn = np.linalg.norm(KX - MX * r.lam, axis=0)
    assert rep["max_backward_error"] == pytest.approx((rn / (Ks.norm1() * np.linalg.norm(r.X, axis=0))).max(), rel=1e-6)


@needs_direct
def test_a_failed_check_repeats_with_what_matters_not_with_the_same_shift(slender, monkeypatch):
    """The retry used to enlarge `shift_rel`, which the shift σ₀ = λ(f_min) of a window does not use: the same failure three times. Now every attempt changes the shift of
    the positive definite factor (larger, then smaller), the driving shift of a window (lifted inside the proven gap), the Lanczos basis – and a REAL failure (a free bar, a window at
    10 Hz, the backward-error limit tightened to 1e-15 so that σ₀ close to the rigid-body modes fails: 1.7e-14) is repaired by the lift."""
    from fea_core.direct import eigen as E
    plans = [E.attempt_plan(a, 1e-9) for a in range(E.SHIFT_TRIES)]
    assert [p["shift_rel"] for p in plans] == [pytest.approx(v) for v in (1e-9, 3e-8, 1e-9 / 30, 1e-6)]
    assert [p["lift"] for p in plans] == [1.0, 1e3, 1e6, 1e9] and [p["ncv_scale"] for p in plans] == [1, 2, 3, 4]
    Ku, Mu, w = slender[False]
    monkeypatch.setattr(E, "SHIFT_REL_FREE", 1e-30)                       # no lift of the free window on the first attempt (it would not fail at all)
    monkeypatch.setattr(E, "BACKWARD_FLOOR", 1e-15)
    logs: list[str] = []
    info: dict = {}
    r = E.solve_modes(Ku, Mu, E.EigenOptions(modes=4, lam_min=(2 * np.pi * 10.0) ** 2, threads=2, free=True), info, log=logs.append)
    assert info["shift_retries"] == 1 and len(info["retries"]) == 1 and "backward error" in info["retries"][0]["why"]
    assert info["retries"][0]["next"]["lift"] == 1e3 and info["sigma0_lifted"]["to"] > 10 * info["sigma0_lifted"]["from"]
    assert info["sigma0_lifted"]["count"] == info["n_below_window"] == 6
    assert np.abs(r.lam - w[6:10]).max() <= 1e-8 * w[9] and info["residual"]["max_backward_error"] <= 1e-15
    assert any("driving shift lifted" in s for s in logs)


@needs_direct
def test_every_attempt_of_an_analysis_that_keeps_failing_differs_and_the_last_error_is_raised(monkeypatch):
    from fea_core.direct import eigen as E
    seen: list[tuple] = []

    def always_inaccurate(KU, MU, opts, info, log, rhs, avail, shift_rel, lift=1.0, ncv_scale=1):
        seen.append((shift_rel, lift, ncv_scale))
        raise E.EigenFail(E.EXIT_ACCURACY, "accuracy", "mode 1 has a residual 1e-3")
    monkeypatch.setattr(E, "_solve_modes", always_inaccurate)
    info: dict = {}
    with pytest.raises(E.EigenFail, match="residual"):
        E.solve_modes(None, None, E.EigenOptions(modes=3), info, log=lambda s: None)
    assert len(seen) == E.SHIFT_TRIES and len(set(seen)) == E.SHIFT_TRIES
    assert len(info["retries"]) == E.SHIFT_TRIES - 1
    # a Sturm count that refuses the set (a shift far above the wanted modes loses them in 1/(λ + s)) is a failure of that attempt too, the next one repairs it; "no natural frequency in
    # the requested range" is the answer and is not retried
    seen.clear()

    def incomplete_twice(KU, MU, opts, info, log, rhs, avail, shift_rel, lift=1.0, ncv_scale=1):
        seen.append(shift_rel)
        if len(seen) < 3:
            raise E.EigenFail(E.EXIT_INCOMPLETE, "incomplete", "the Sturm count does not agree with the modes found after 4 passes", retry=True)
        return E.EigenResult(np.zeros(1), np.zeros((1, 1)), info)
    monkeypatch.setattr(E, "_solve_modes", incomplete_twice)
    info2: dict = {}
    r = E.solve_modes(None, None, E.EigenOptions(modes=3), info2, log=lambda s: None)
    assert len(seen) == 3 and info2["shift_retries"] == 2 and r.info is info2
    seen.clear()

    def no_frequency(KU, MU, opts, info, log, rhs, avail, shift_rel, lift=1.0, ncv_scale=1):
        seen.append(shift_rel)
        raise E.EigenFail(E.EXIT_INCOMPLETE, "incomplete", "no natural frequency in the requested range")
    monkeypatch.setattr(E, "_solve_modes", no_frequency)
    with pytest.raises(E.EigenFail, match="no natural frequency"):
        E.solve_modes(None, None, E.EigenOptions(modes=3), {}, log=lambda s: None)
    assert len(seen) == 1


@needs_direct
def test_the_analysis_and_the_routing_estimate_the_same_vectors(monkeypatch):
    """`direct.eigen_vector_count` is the ONE estimate: the routing (`decide_full`: free memory) and the worker (`check_memory`) used 2k + 12 and ncv + 3k + 8 vectors."""
    from fea_core import direct
    from fea_core.direct import eigen as E
    assert direct.eigen_vector_count(4) == 20 + 12 + 8 and direct.eigen_vector_count(100) == 201 + 300 + 8 and direct.eigen_vector_count(4, ncv=60) == 60 + 12 + 8
    KU, MU, lam = _laplace3(8)
    seen = []
    real = E.Shifted.check_memory
    monkeypatch.setattr(E.Shifted, "check_memory", lambda self, extra, avail: (seen.append(extra), real(self, extra, avail))[1])
    E.solve_modes(KU, MU, E.EigenOptions(modes=5, threads=2), {}, log=lambda s: None)
    n = KU.shape[0]
    assert seen == [pytest.approx(direct.eigen_vectors_gb(n, 5, 20))]


def test_compare_modes_treats_a_cluster_as_a_subspace_and_skips_rigid_modes():
    from fea_core.eigen import compare_modes, mac_matrix
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 50, 3))
    A[1] = A[0] * 0.3 + A[1]
    f = np.array([0.0, 10.0, 10.00001, 20.0, 30.0, 40.0])
    th = 0.7
    B = A.copy()
    B[1] = np.cos(th) * A[1] + np.sin(th) * A[2]                    # another mixture of the pair (modes 2 and 3 are the same eigenspace)
    B[2] = -np.sin(th) * A[1] + np.cos(th) * A[2]
    g = rng.standard_normal((6, 6))
    c = compare_modes(f, A, g, f, B, g)
    assert c["n"] == 5 and min(c["mac"]) < 0.99                       # the diagonal MAC of the rotated pair is low …
    assert min(c["subspace"]) > 0.9999                               # … the subspace comparison says they are the same modes
    assert mac_matrix(A, A).shape == (6, 6) and np.allclose(np.diag(mac_matrix(A, A)), 1.0)


# ------------------------------------------------------------------ recovery of many modes at once
@needs_chain
def test_recover_batch_equals_recover_mode_by_mode(clamped):
    from fea_core.recovery import recover, recover_batch
    st, parts, wd, ref, own, mesh = clamped
    U = own.shapes[:3].astype(np.float64)
    S, RF = recover_batch(st, mesh, U)
    for j in range(3):
        s1, r1 = recover(st, mesh, U[j])
        assert np.abs(S[j] - s1).max() <= 1e-12 * np.abs(s1).max() and np.abs(RF[j] - r1).max() <= 1e-12 * np.abs(r1).max()


# ------------------------------------------------------------------ routing and the setting
def test_the_engine_setting_is_stored_only_when_it_is_not_automatic(tmp_path):
    from fea_core.study import Study
    st, parts = _beam(tmp_path, True)
    assert "modal_engine" not in st.to_dict()["solver"]
    st.solver.modal_engine = "direct"
    d = st.to_dict()
    assert d["solver"]["modal_engine"] == "direct"
    assert Study.from_dict(d).solver.modal_engine == "direct"
    assert Study.from_dict({**d, "solver": {"backend": "auto"}}).solver.modal_engine == "auto"          # a file of an older version


@needs_direct
def test_routing_keeps_ccx_for_what_needs_the_geometric_stiffness_or_beam_elements(tmp_path):
    from fea_core.direct import modal_run
    from fea_core.study import BoltSpec, LoadCase
    st, parts = _beam(tmp_path, True)
    st.modal.enabled = True
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    assert modal_run.decide(st, mesh)[0] == "direct"
    st.solver.backend = "ccx"
    assert modal_run.decide(st, mesh) == ("ccx", "the solver of the study is CalculiX")
    st.solver.backend = "auto"
    st.solver.modal_engine = "ccx"
    assert modal_run.decide(st, mesh) == ("ccx", "chosen by the user")
    st.solver.modal_engine = "auto"
    st.dynamics.enabled = True
    st.dynamics.prestress.enabled, st.dynamics.prestress.case = True, "A"
    eng, why = modal_run.decide(st, mesh, dynamics=True)
    assert eng == "ccx" and "prestressed" in why
    st.solver.modal_engine = "direct"
    with pytest.raises(modal_run.ModalUnsupported, match="prestressed"):
        modal_run.decide(st, mesh, dynamics=True)
    st.solver.modal_engine = "auto"
    st.dynamics.prestress.enabled = False
    st.bolts.append(BoltSpec(name="b", size=4.0))
    eng, why = modal_run.decide(st, mesh)
    assert eng == "ccx" and "bolts" in why
    st.bolts.clear()
    st.load_cases.append(LoadCase("a", []))
    assert modal_run.decide(st, mesh)[0] == "direct"          # a load case does not change the free vibration


@needs_direct
def test_auto_keeps_ccx_for_a_small_problem_and_takes_the_direct_path_by_size_and_modes(tmp_path, monkeypatch):
    from fea_core.direct import modal_run
    st, parts = _beam(tmp_path, True)
    st.modal.enabled = True
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    ndof = 3 * len(mesh.nodes)
    monkeypatch.delenv(modal_run.MIN_WORK_ENV, raising=False)
    assert modal_run.min_work() == modal_run.MIN_WORK
    # a tiny cantilever with 6 modes: the fixed start of the two processes costs more than CalculiX needs
    assert ndof * (st.modal.modes + 10) < modal_run.MIN_WORK
    eng, why = modal_run.decide(st, mesh)
    assert eng == "ccx" and "small problem" in why and f"{ndof} DOF" in why
    # many modes make the same mesh worth it (CalculiX grows with the modes much faster)
    eng, why = modal_run.decide(st, mesh, modes=int(modal_run.MIN_WORK / ndof) + 1)
    assert eng == "direct", why
    # a forced engine is not asked for its size; the environment overrides the constant
    st.solver.modal_engine = "direct"
    assert modal_run.decide(st, mesh)[0] == "direct"
    st.solver.modal_engine = "auto"
    monkeypatch.setenv(modal_run.MIN_WORK_ENV, "0")
    assert modal_run.min_work() == 0.0 and modal_run.decide(st, mesh)[0] == "direct"
    monkeypatch.setenv(modal_run.MIN_WORK_ENV, "nonsense")
    assert modal_run.min_work() == modal_run.MIN_WORK
    # the dynamics are decided by the same size rule (the reason of a feature refusal comes first)
    monkeypatch.delenv(modal_run.MIN_WORK_ENV)
    st.dynamics.enabled = True
    st.dynamics.prestress.enabled, st.dynamics.prestress.case = True, "A"
    assert "prestressed" in modal_run.decide(st, mesh, dynamics=True)[1]


@needs_direct
def test_a_failing_ccx_of_a_small_problem_continues_with_the_direct_analysis(tmp_path, monkeypatch):
    """Under `auto` a small problem goes to CalculiX by the size rule only – when CalculiX fails (it ends with an error on *TIE + a normal elastic support) the direct
    analysis, which can do the study, takes over; a user's choice of CalculiX reports the failure."""
    from fea_core.ccx import runner
    from fea_core.dynamics import modal as DM
    from fea_core.eigen import EigenError, run_eigen
    st, parts = _beam(tmp_path, True)
    st.modal.enabled, st.modal.modes = True, 4
    st.dynamics.enabled, st.dynamics.modes = True, 4
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    monkeypatch.delenv("FEA_MODAL_MIN_WORK", raising=False)
    calls = []

    def failing_ccx(workdir, job, **kw):
        calls.append(job)
        return runner.CcxRun(returncode=0xFFFFFFFF, finished=False, errors=[], warnings=[], log_path=Path(workdir) / f"{job}.log")
    monkeypatch.setattr(runner, "run_ccx", failing_ccx)
    logs = []
    ms = run_eigen(st, mesh, tmp_path / "e", logs.append)[0]
    assert calls == ["modal"] and ms.engine == "direct" and ms.engine_info["reason"].startswith("CalculiX failed")
    assert any("CalculiX failed" in s for s in logs)
    assert ms.values[0] > 0 and len(ms.values) == 4
    calls.clear()
    basis = DM.run_basis(st, mesh, tmp_path / "b", logs.append)
    assert calls == ["dyn_modal"] and basis.engine == "direct" and basis.engine_info["reason"].startswith("CalculiX failed") and basis.K == 4
    # the user chose CalculiX: its failure is reported, nothing is computed behind the back
    st.solver.modal_engine = "ccx"
    with pytest.raises(Exception, match="CalculiX"):
        run_eigen(st, mesh, tmp_path / "e2", lambda s: None)
    assert EigenError is not None


@needs_chain
def test_a_failing_direct_analysis_continues_in_ccx_with_the_reason_and_a_forced_one_reports_it(clamped, monkeypatch):
    from fea_core.direct import DirectMemoryError, modal_run
    from fea_core.eigen import EigenError, run_eigen
    st, parts, wd, ref, own, mesh = clamped
    st.modal.enabled, st.modal.modes = True, 6

    def boom(*a, **k):
        raise DirectMemoryError("the free memory fell to 0.5 GB")
    monkeypatch.setattr(modal_run, "run_modes", boom)
    logs: list[str] = []
    ms = run_eigen(st, mesh, wd / "fallback", logs.append)[0]
    assert ms.engine == "ccx" and "failed" in ms.engine_info["reason"] and "0.5 GB" in ms.engine_info["reason"]
    assert any("continuing" in s or "computing in CalculiX" in s for s in logs)
    assert np.allclose(ms.values[:6], ref.values[:6])
    st.solver.modal_engine = "direct"
    try:
        with pytest.raises(EigenError, match="0.5 GB"):
            run_eigen(st, mesh, wd / "forced", lambda s: None)
    finally:
        st.solver.modal_engine = "auto"


@needs_chain
def test_a_window_means_the_same_in_ccx_and_in_the_direct_path_and_never_lists_phantom_modes(clamped, free):
    """f_min / f_max of `Study.modal`: the lowest `modes` modes INSIDE the window – on both engines. (ccx itself computes the lowest `modes` modes of the model and lists those of
    them that lie in the window; a clamped beam, 4 modes, f_min 4401 Hz gave [0, 0, 0, 5243] from ccx – three phantom 0 Hz modes with zero shapes carrying the participation rows
    of real modes 1–3 – and [5243, 6441, 9579, 10827] from the direct path.)"""
    from fea_core.eigen import run_eigen
    st, parts, wd, ref, own, mesh = clamped
    f = np.array(own.values)
    cases = [(0.5 * (f[2] + f[3]), None, 4, 4),                    # the window starts after the 3rd mode: modes 4–7
             (0.5 * (f[0] + f[1]), 0.5 * (f[3] + f[4]), 6, 3),     # two bounds: only modes 2–4 are inside, 6 were asked for (the window is exhausted)
             (0.0, 0.5 * (f[1] + f[2]), 6, 2)]                     # an upper bound only
    keep = (st.modal.modes, st.modal.f_min, st.modal.f_max, st.modal.enabled)
    try:
        st.modal.enabled = True
        for q, (f_min, f_max, modes, n_inside) in enumerate(cases):
            st.modal.modes, st.modal.f_min, st.modal.f_max = modes, f_min, f_max
            sets = {}
            for engine in ("ccx", "direct"):
                st.solver.modal_engine = engine
                sets[engine] = run_eigen(st, mesh, wd / f"win_{q}_{engine}", lambda s: None)[0]
            a, b = sets["direct"], sets["ccx"]
            assert a.engine == "direct" and b.engine == "ccx"
            assert len(a.values) == len(b.values) == n_inside, (f_min, f_max, a.values, b.values)
            assert min(b.values) > 0.0 and min(a.values) >= f_min and all(float(np.abs(s).max()) > 0.0 for s in b.shapes)      # no phantom zero mode, every shape real
            assert f_max is None or max(max(a.values), max(b.values)) <= f_max
            c = _gate(b, a, first=min(2, n_inside))                      # ccx is only as accurate as its ARPACK run: the strict tolerances for the lowest two modes of the window
            assert c["n"] == n_inside
            assert b.engine_info.get("modes_computed", 0) >= n_inside
        # the second case needed a second CalculiX run: the first (4 modes) held one mode of the window only
        st.modal.modes, st.modal.f_min, st.modal.f_max = 4, cases[0][0], None
        st.solver.modal_engine = "ccx"
        assert run_eigen(st, mesh, wd / "win_runs", lambda s: None)[0].engine_info["ccx_runs"] == 2
        # a free body: the rigid-body modes are 0 Hz and never inside a window above zero – on both engines
        st_f, parts_f, wd_f, ref_f, own_f, mesh_f = free
        keep_f = (st_f.modal.modes, st_f.modal.f_min, st_f.modal.f_max)
        try:
            st_f.modal.modes, st_f.modal.f_min, st_f.modal.f_max = 4, 0.001, None
            res = {}
            for engine in ("ccx", "direct"):
                st_f.solver.modal_engine = engine
                res[engine] = run_eigen(st_f, mesh_f, wd_f / f"win_{engine}", lambda s: None)[0]
            assert len(res["ccx"].values) == len(res["direct"].values) == 4 and min(res["ccx"].values) > 100.0 and min(res["direct"].values) > 100.0
            assert np.allclose(res["ccx"].values, res["direct"].values, rtol=3e-4)
        finally:
            st_f.modal.modes, st_f.modal.f_min, st_f.modal.f_max = keep_f
            st_f.solver.modal_engine = "auto"
    finally:
        st.modal.modes, st.modal.f_min, st.modal.f_max, st.modal.enabled = keep
        st.solver.modal_engine = "auto"


@needs_chain
def test_the_upper_bound_of_the_dynamics_means_the_same_in_ccx_and_in_the_direct_path(clamped):
    """The dynamics have only an upper bound (`f_max`): the lowest `modes` modes below it. ccx lists the lowest k modes that lie below it, the direct path counts the modes below
    it and asks for the lowest of them – the same set, no phantom modes."""
    from fea_core.dynamics import modal as DM
    st, parts, wd, ref, own, mesh = clamped
    f = np.array(own.values)
    f_max = 0.5 * (f[2] + f[3])
    keep = st.solver.modal_engine
    try:
        bases = {}
        for engine in ("ccx", "direct"):
            st.solver.modal_engine = engine
            bases[engine] = DM.run_basis(st, mesh, wd / f"dynwin_{engine}", lambda s: None, modes=6, f_max=f_max)
    finally:
        st.solver.modal_engine = keep
    a, b = bases["direct"], bases["ccx"]
    assert a.engine == "direct" and b.engine == "ccx" and a.K == b.K == 3
    assert np.allclose(a.freq, b.freq, rtol=3e-4) and a.freq.max() <= f_max and b.freq.min() > 0.0


@needs_chain
def test_the_size_rule_of_auto_does_not_change_the_modes_of_a_study(clamped, monkeypatch):
    """`auto` takes CalculiX below DOF·(modes + 10) = 7e5 and the direct path above: the choice must not change what the study means. The same studies – with and without a
    window – on either side of the rule give the same modes (frequencies, count, shapes by MAC), only the engine differs."""
    from fea_core.direct import modal_run
    from fea_core.eigen import run_eigen
    st, parts, wd, ref, own, mesh = clamped
    f = np.array(own.values)
    keep = (st.modal.modes, st.modal.f_min, st.modal.f_max, st.modal.enabled, st.solver.modal_engine)
    try:
        st.modal.enabled, st.solver.modal_engine = True, "auto"
        for q, (f_min, f_max, modes) in enumerate([(0.0, None, 6), (0.5 * (f[1] + f[2]), None, 5), (0.5 * (f[1] + f[2]), 0.5 * (f[5] + f[6]), 6)]):
            st.modal.modes, st.modal.f_min, st.modal.f_max = modes, f_min, f_max
            sets = {}
            for label, work in (("below the rule", "1e12"), ("above the rule", "0")):
                monkeypatch.setenv(modal_run.MIN_WORK_ENV, work)
                sets[label] = run_eigen(st, mesh, wd / f"rule_{q}_{work}", lambda s: None)[0]
            small, large = sets["below the rule"], sets["above the rule"]
            assert small.engine == "ccx" and large.engine == "direct"
            assert len(small.values) == len(large.values) >= 3, (q, small.values, large.values)
            _gate(small, large, first=3)
    finally:
        st.modal.modes, st.modal.f_min, st.modal.f_max, st.modal.enabled, st.solver.modal_engine = keep


@needs_direct
def test_only_a_crash_of_ccx_is_taken_over_an_error_about_the_model_is_not(tmp_path, monkeypatch):
    """The direct analysis takes over from ccx only when ccx DIES (no `*ERROR`, no "Job finished": an access violation, 0xFFFFFFFF) – an `*ERROR` of ccx is a verdict on the model
    and is passed on. A takeover is recorded for the warnings of the results (not only in the log)."""
    from types import SimpleNamespace
    from fea_core import solve
    from fea_core.ccx import runner
    from fea_core.dynamics import modal as DM
    from fea_core.eigen import run_eigen
    st, parts = _beam(tmp_path, True)
    st.modal.enabled, st.modal.modes = True, 4
    st.dynamics.enabled, st.dynamics.modes = True, 4
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    monkeypatch.delenv("FEA_MODAL_MIN_WORK", raising=False)
    calls = []

    def ccx_that(errors, rc):
        def run(workdir, job, **kw):
            calls.append(job)
            return runner.CcxRun(returncode=rc, finished=False, errors=list(errors), warnings=[], log_path=Path(workdir) / f"{job}.log")
        return run
    # a crash: the direct analysis takes over, the results carry a warning
    monkeypatch.setattr(runner, "run_ccx", ccx_that([], 3221225477))
    res = SimpleNamespace(warnings=[], thermal=None, modes=None)
    solve._add_eigen(st, mesh, res, tmp_path / "w", lambda s: None, None, None)
    assert calls == ["modal"] and res.modes[0].engine == "direct" and res.modes[0].engine_info["takeover"].startswith("Natural frequencies: CalculiX stopped")
    assert res.warnings == [res.modes[0].engine_info["takeover"]]
    calls.clear()
    basis = DM.run_basis(st, mesh, tmp_path / "b", lambda s: None)
    assert calls == ["dyn_modal"] and basis.engine == "direct" and basis.engine_info["takeover"].startswith("Dynamics: CalculiX stopped")
    # an error about the model: nothing is computed behind its back
    calls.clear()
    monkeypatch.setattr(runner, "run_ccx", ccx_that(["*ERROR in e_c3d: nonpositive jacobian determinant in element 12"], 201))
    with pytest.raises(runner.CcxError, match="nonpositive jacobian") as ei:
        run_eigen(st, mesh, tmp_path / "e", lambda s: None)
    assert not isinstance(ei.value, runner.CcxCrash) and calls == ["modal"]
    calls.clear()
    with pytest.raises(runner.CcxError, match="nonpositive jacobian"):
        DM.run_basis(st, mesh, tmp_path / "b2", lambda s: None)
    assert calls == ["dyn_modal"]


def test_the_rigid_modes_of_a_free_body_are_found_by_the_same_rule_on_both_sides():
    from fea_core.eigen import compare_modes, rigid_count
    noise = [0.0, 0.0, 0.0013, 0.0009, 0.002, 0.0011]                      # ccx: two modes omitted (negative λ) and four tiny positive ones
    f_ccx = noise + [66.4, 183.2, 199.2, 359.0, 548.3]
    assert rigid_count(f_ccx, True) == 6 and rigid_count(f_ccx, False) == 0
    assert rigid_count([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 66.0], True) == 6
    assert rigid_count([581.9, 1142.2, 3535.1, 5056.8], True) == 0           # a supported body, or a free one without a jump: no rigid mode
    assert rigid_count([], True) == 0
    # the pairing of the cross-check: the direct list has the rigid modes at 0 (or dropped), ccx at 1e-3 Hz: the elastic modes are paired in order, not by the index in the lists
    rng = np.random.default_rng(5)
    A = rng.standard_normal((5, 40, 3))
    g = rng.standard_normal((5, 6))
    f_dir = [66.4, 183.2, 199.2, 359.0, 548.3]                              # the basis of the dynamics dropped the six rigid modes
    f_cc = np.array(f_ccx)
    f_cc[:rigid_count(f_cc, True)] = 0.0                                    # what `_arrays_ccx` / `ccx_modal` do: the rigid-body modes of a free body are 0 Hz
    B = np.concatenate([rng.standard_normal((6, 40, 3)), A])
    gb = np.concatenate([rng.standard_normal((6, 6)), g])
    c = compare_modes(f_dir, A, g, f_cc, B, gb)
    assert c["n"] == 5 and c["n_a"] == 5 and c["n_b"] == 5 and c["max_freq_rel"] < 1e-12 and c["min_mac"] > 0.999999 and c["modes"] == [1, 2, 3, 4, 5]
    # a mode that one side lacks is a mismatch of the pairs behind it – never a silent re-pairing
    c2 = compare_modes([66.4, 199.2, 359.0, 548.3], np.delete(A, 1, axis=0), np.delete(g, 1, axis=0), f_cc, B, gb)
    assert c2["n"] == 4 and c2["max_freq_rel"] > 0.05 and c2["n_a"] == 4
    # tiny positive rigid modes left in the list are still removed when the caller gives the limit of the noise
    c3 = compare_modes(f_dir, A, g, noise + f_ccx[6:], B, gb, rigid_hz=0.05)
    assert c3["n"] == 5 and c3["max_freq_rel"] < 1e-12


def test_ccx_tables_with_an_irregular_numbering_never_raise_an_index_error(tmp_path):
    """`present[r − 1]` of a table whose rows are not exactly 1…len(present) (a gap, a row beyond the eigenvalue table) used to be an IndexError: the rows stay as they are."""
    from fea_core.eigen import _renumber, read_dat
    assert _renumber({1: "a", 2: "b", 4: "c"}, [3, 4, 5]) == {1: "a", 2: "b", 4: "c"}
    assert _renumber({1: "a", 2: "b", 3: "c"}, [3, 4, 5]) == {3: "a", 4: "b", 5: "c"}
    assert _renumber({3: "a", 4: "b", 5: "c"}, [3, 4, 5]) == {3: "a", 4: "b", 5: "c"}
    assert _renumber({1: "a"}, None) == {1: "a"}
    eig = "\n".join(f"{j:7d}   0.1E+05   0.1E+03   {10.0 * j:.6E}   0.0" for j in (3, 4, 5))
    part = "\n".join(f"{r:7d}   {r}.0E-03  0.0  0.0  0.0  0.0  0.0" for r in (1, 2, 7))
    text = " E I G E N V A L U E   O U T P U T\n\n MODE NO    EIGENVALUE\n\n\n" + eig + "\n\n P A R T I C I P A T I O N   F A C T O R S\n\nMODE NO.\n\n" + part + "\n"
    (tmp_path / "x.dat").write_text(text, encoding="latin-1")
    d = read_dat(tmp_path / "x.dat", "modal")
    assert d["modes"] == [3, 4, 5] and len(d["participation"]) == len(d["freq"])


def test_the_main_program_holds_no_second_float64_copy_of_the_modes_for_the_mode_set():
    """`modeset_from_direct` normalizes one mode at a time into float32: the peak allocation is the float32 result plus one mode, not the float64 copy of all the shapes (about 1 GB
    transient for 100 modes × 400 k DOF, against 0.16 GB of float32)."""
    import tracemalloc
    from fea_core.direct.modal_run import DirectModes
    from fea_core.eigen import modeset_from_direct
    rng = np.random.default_rng(2)
    m, n = 60, 800
    shapes = rng.standard_normal((m, n, 3))
    dm = DirectModes(np.linspace(1e6, 9e6, m), shapes, rng.standard_normal((m, 6)), np.ones(6), 1.0, 0.0, np.zeros(m, bool))
    tracemalloc.start()
    ms = modeset_from_direct(dm, n)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    result = m * n * 3 * 4
    assert ms.shapes.dtype == np.float32 and peak < 1.25 * result, (peak, result)          # the old code peaked at 3 × the result (a float64 copy and the float32 cast)
    # the same numbers as the float64 normalization (max |u| = 1, the largest component positive)
    ref = shapes / np.abs(shapes).reshape(m, -1).max(axis=1)[:, None, None]
    sgn = np.sign(np.take_along_axis(ref.reshape(m, -1), np.abs(ref).reshape(m, -1).argmax(axis=1)[:, None], axis=1))[:, 0]
    assert np.allclose(ms.shapes, (ref * sgn[:, None, None]).astype(np.float32), atol=1e-7)


@needs_direct
def test_the_routing_estimate_of_the_memory_is_the_one_of_the_analysis(tmp_path, monkeypatch):
    from fea_core import direct
    from fea_core.direct import modal_run
    st, parts = _beam(tmp_path, True)
    st.modal.enabled = True
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    assert modal_run.decide(st, mesh)[0] == "direct"
    monkeypatch.setattr(direct, "eigen_vectors_gb", lambda ndof, modes, ncv=0: 1e6)         # the vectors of the analysis would need a petabyte
    eng, why = modal_run.decide(st, mesh)
    assert eng == "ccx" and "would need" in why


@needs_chain
def test_the_cross_check_of_the_modes_is_in_the_results_and_survives_saving(clamped, tmp_path):
    from fea_core.eigen import load_arrays, run_eigen, save_arrays
    st, parts, wd, ref, own, mesh = clamped
    st.modal.enabled, st.modal.modes, st.modal.verify_ccx = True, 6, True
    try:
        ms = run_eigen(st, mesh, wd / "verify", lambda s: None)[0]
    finally:
        st.modal.verify_ccx = False
    assert ms.engine == "direct" and ms.check["engine"] == "ccx" and ms.check["n"] == 6
    assert ms.check["max_freq_rel"] < 1e-5 and ms.check["min_subspace"] > 0.9999
    arrays: dict = {}
    meta = save_arrays([ms], arrays)
    back = load_arrays(meta, arrays)[0]
    assert back.engine == "direct" and back.check["n"] == 6 and back.engine_info["sturm_checks"] == ms.engine_info["sturm_checks"]
    assert ms.summary()["engine"] == "direct"


# ------------------------------------------------------------------ combinations: constraints, added masses, orthotropy
ORTHO_A = {"E1": 100000.0, "E2": 20000.0, "E3": 10000.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.4, "G12": 5000.0, "G13": 4000.0, "G23": 3000.0,
           "alpha1": 2e-6, "alpha2": 15e-6, "alpha3": 9e-6}


def _study_tie_ortho_masses(wd: Path):
    """Two bodies joined by *TIE (the second finer, orthotropic with rotated axes), clamped at one end, a rigid point mass with rotary inertia on the free end, a distributing
    point mass on the master face of the tie, a surface mass on the top of the orthotropic part."""
    from fea_core.dynamics.model import PointMass, SurfaceMass
    from fea_core.study import CoordinateSystem
    st, parts = _study("tie ortho masses", wd / "model.step", [Part("A", box(0, 0, 0, 60, 24, 16)), Part("B", box(60, 0, 0, 60, 24, 16))], MeshSettings(size=6.0))
    st.mesh.separate = ["B"]
    st.add_selection("b_cela", find_faces(parts, part="B", normal=[-1, 0, 0]))
    st.mesh.local = [{"selection": "b_cela", "size": 3.6}]
    th = np.radians(30.0)
    st.coordinate_systems["mat30"] = CoordinateSystem("mat30", "cartesian", [0, 0, 0], [0, 0, 1], [float(np.cos(th)), float(np.sin(th)), 0.0])
    st.add_material(Material("Ortotropni A", ORTHO_A["E1"], ORTHO_A["nu12"], rho=1.6e-9, alpha=1e-5, ortho=ORTHO_A, ortho_csys="mat30"))
    st.part_materials["B"] = "Ortotropni A"
    for name, kw in (("a0", dict(part="A", x=0)), ("a1", dict(part="A", x=60)), ("b1", dict(part="B", x=120)), ("b_vrch", dict(part="B", z=16))):
        st.add_selection(name, find_faces(parts, **kw))
    st.constraints.append(Fixed("a0"))
    st.masses += [PointMass("b1", 3e-3, point=[120.0, 12.0, 8.0], inertia=[40.0, 30.0, 25.0], coupling="rigid", name="motor"),
                  PointMass("a1", 1.5e-3, point=[60.0, 12.0, 8.0], coupling="distributing", name="cidlo"),
                  SurfaceMass("b_vrch", 4.0, name="plosch")]
    return st, parts


def _study_cyl_elastic_remote(wd: Path):
    """A plate with a hole: cylindrical support on the hole, a spring foundation on the bottom (shared nodes with the circle), a rigid remote support on one end and a rigid
    point mass with rotary inertia on the other."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.dynamics.model import PointMass
    from fea_core.study import CylindricalSupport, RemoteSupport
    from verification.cases import cyl, solid
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 12), cyl((30, 20, -1), (0, 0, 1), 6, 14)).Shape())
    st, parts = _study("cyl elastic remote", wd / "model.step", [Part("Deska", plate)], MeshSettings(size=4.0))
    for name, kw in (("dira", dict(radius=6)), ("dno", dict(z=0)), ("x0", dict(x=0)), ("x1", dict(x=60))):
        st.add_selection(name, find_faces(parts, **kw))
    st.constraints += [CylindricalSupport("dira", radial=True, tangential=False, axial=False), ElasticSupport("dno", 30.0, "all"),
                       RemoteSupport("x0", [-15.0, 20.0, 6.0], [True] * 6, "rigid")]
    st.masses.append(PointMass("x1", 2e-3, point=[60.0, 20.0, 6.0], inertia=[20.0, 20.0, 20.0], coupling="rigid", name="hmota"))
    return st, parts


@needs_chain
@pytest.mark.parametrize("build", [_study_tie_ortho_masses, _study_cyl_elastic_remote], ids=["tie_orthotropic_masses", "cylindrical_elastic_remote_rigid_mass"])
def test_combinations_of_constraints_added_masses_and_orthotropy_equal_ccx(tmp_path, build):
    """Everything the elimination of the dependent DOFs touches at once. Against ccx the frequencies and the shapes (MAC); the effective masses of ccx are not usable here
    (a *RIGID BODY with a free rotation node makes CalculiX report participation factors and a total mass that include the inertia – dynamics.modal.own_gamma), so they are
    held against the independent integration `check_basis` of the program (own mass_times + the points of the masses): Γ, the normalization φᵀMφ = 1 and the equilibrium
    of the modal reactions of the direct basis."""
    from fea_core.dynamics import modal as DM
    from fea_core.eigen import compare_modes
    st, parts = build(tmp_path)
    ref, own, mesh = _ccx_and_direct(st, parts, tmp_path, 10)
    c = compare_modes(own.values, own.shapes, own.participation, ref.values, ref.shapes, ref.participation, components=0)
    assert c["n"] == 10 and c["max_freq_rel"] <= 3e-4 and max(c["freq_rel"][:5]) <= 1e-4 and c["min_subspace"] >= 0.9999
    st.dynamics.enabled, st.dynamics.modes = True, 10
    basis = DM.run_basis(st, mesh, tmp_path / "basis", lambda s: None)
    assert basis.engine == "direct" and basis.K == 10
    ch = basis.checks
    # the equilibrium of the modal reactions is exact (1e-13) unless an elastic support shares nodes with another support (the cylindrical support on the circle of the
    # foundation: the spring force of those nodes is counted in the support reaction AND in the reaction of the spring – 3e-4 of the reaction, the same in the ccx path;
    # the check of the program warns only above 1e-3)
    assert ch["mass_norm_error"] < 1e-8 and ch["gamma_error"] < 1e-4 and ch["mode_equilibrium"] < 1e-3, ch
    assert np.allclose(basis.freq, own.values, rtol=1e-9)


# ------------------------------------------------------------------ the whole run_study
@needs_chain
def test_run_study_computes_the_statics_and_the_modes_directly_and_the_buckling_in_ccx(tmp_path):
    """Statics in the direct solver, natural frequencies on the direct path (after the statics: one direct run at a time), linear buckling in a ccx side job: all three in
    one study, the results say which engine computed what, and the .fea round trip keeps it."""
    from fea_core.solve import run_study
    from fea_core.study import Force, LoadCase
    st, parts = _beam(tmp_path, True)
    st.add_selection("konec", find_faces(parts, x=L_B))
    st.load_cases.append(LoadCase("tlak", [Force("konec", [-500.0, 0.0, 0.0])]))
    st.modal.enabled, st.modal.modes = True, 6
    st.buckling.enabled, st.buckling.modes = True, 2
    st.solver.backend = "auto"
    res = run_study(st, tmp_path / "run", parts=parts)
    assert res.info["backend"] == "ngs_direct"
    kinds = [(m.kind, m.engine) for m in res.modes]
    assert kinds == [("modal", "direct"), ("buckling", "ccx")], kinds
    assert len(res.modes[0].values) == 6 and res.modes[1].values[0] > 0
    assert res.modes[0].engine_info["reason"].startswith("linear free-vibration analysis")
    from fea_core.results import Results
    res.save(tmp_path / "r.npz")
    back = Results.load(tmp_path / "r.npz")
    assert back.modes[0].engine == "direct" and back.modes[0].engine_info["library"].startswith("rslab")


@needs_chain
def test_a_forced_direct_engine_is_checked_before_meshing(tmp_path):
    from fea_core import solvers
    from fea_core.study import BoltSpec
    st, parts = _beam(tmp_path, True)
    st.modal.enabled = True
    st.solver.modal_engine = "direct"
    assert solvers.check_backend(st) == []
    st.bolts.append(BoltSpec(name="b", size=4.0))
    errs = solvers.check_backend(st)
    assert errs and "natural-frequency" in errs[0] and "bolts" in errs[0]
    st.solver.modal_engine = "auto"
    assert solvers.check_backend(st) == []


@needs_chain
def test_a_welded_t_joint_is_a_direct_modal_study_with_the_modes_of_ccx_and_bolts_are_not(tmp_path, monkeypatch):
    """F13 step 1b put bolts and welds on the static direct path; for the natural frequencies only the welds follow: a weld is a meshed body (the contact of its root is
    opened in the mesh), so its mass and stiffness are the mesh's – the modes of the welded T-joint (W1) equal CalculiX on the same mesh. A bolt is a beam element on the
    extra nodes of the own assembly that has a stiffness but no mass matrix (ccx gives it the density of steel): the modal routing keeps CalculiX for it, with the reason."""
    from fea_core.direct import modal_run
    from fea_core.study import BoltSpec
    from verification.joint_cases import w1_build
    monkeypatch.setenv(modal_run.MIN_WORK_ENV, "0")
    st, parts = w1_build(tmp_path)
    assert len(st.active_welds()) == 2
    ref, own, mesh = _ccx_and_direct(st, parts, tmp_path, 6)
    c = _gate(ref, own, strict=(1e-5, 1e-4, 0.9999), loose=(3e-5, 1e-3, 0.999))
    assert c["n"] == 6 and own.engine == "direct"
    assert modal_run.decide(st, mesh)[0] == "direct"
    st.bolts.append(BoltSpec(name="b", size=4.0))
    eng, why = modal_run.decide(st, mesh)
    assert eng == "ccx" and "bolts" in why and "mass matrix" in why


@needs_chain
def test_a_3d_printed_part_has_the_same_modes_as_in_ccx(tmp_path):
    """3D printing: the matrix C of every element (shell / core groups with the orientation of the lines) in the stiffness and the density per element ρ·(f + (1 − f)·ρ_infill)
    in the mass – frequencies and shapes against ccx on the same mesh."""
    from fea_core import materials as matlib
    from fea_core.printing import PrintSpec
    pla = matlib.by_name("Tisk PLA Bambu Basic")
    spec = PrintSpec("Blok", build_dir=[0, 0, 1], layer_height=0.2, line_width=0.5, walls=2, top_layers=3, bottom_layers=3, infill=0.2, pattern="grid")
    st, parts = _study("print modal", tmp_path / "model.step", [Part("Blok", box(0, 0, 0, 12, 12, 40))], MeshSettings(size=3.0))
    st.add_material(pla, default=True)
    st.materials.pop("Ocel S235")
    st.printing.parts["Blok"] = spec
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints.append(Fixed("z0"))
    ref, own, mesh = _ccx_and_direct(st, parts, tmp_path, 8)
    c = _gate(ref, own, strict=(3e-5, 1e-3, 0.9999), loose=(3e-4, 1e-2, 0.999))      # the pairs of this square bar: ccx 2e-5 on its 4th and 5th mode
    assert c["n"] == 8
    # the mass is the printed one (shell + infill), not the solid one
    from fea_core.dynamics.mass import total_mass
    assert 0.3 < total_mass(st, mesh) / (pla.rho * 12 * 12 * 40) < 0.9
    assert own.total_mass[0] < 0.9 * pla.rho * 12 * 12 * 40

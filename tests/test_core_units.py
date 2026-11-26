"""Unit tests of core helpers without own tests so far: convergence (Richardson / GCI, run_convergence with a mocked solve),
signature (mesh signature), ties (*TIE detection on a tiny synthetic mesh) and contour (band invariants). No Gmsh, no ccx."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from fea_core import contour, convergence, signature, ties
from fea_core.study import ContactPair, Material, Study


# ------------------------------------------------------------------ convergence: Richardson / GCI
def _levels(c, a, p, h0=4.0, r=1.5, n=3):
    return [c + a * (h0 / r ** k) ** p for k in range(n)]


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_richardson_recovers_order_and_limit(p):
    r = 1.5
    vals = _levels(100.0, -5.0, p, r=r)
    out = convergence.richardson(vals, r)
    assert out["monotonic"] is True
    assert out["order"] == pytest.approx(p, rel=1e-9)
    assert out["extrapolated"] == pytest.approx(100.0, rel=1e-9)
    f2, f3 = vals[1], vals[2]
    assert out["gci"] == pytest.approx(1.25 * abs((f3 - f2) / f3) / (r ** p - 1), rel=1e-12)


def test_richardson_uses_last_three_levels():
    r = 2.0
    vals = [7.0] + _levels(10.0, 1.0, 2.0, r=r)            # a wild first level is ignored
    assert convergence.richardson(vals, r)["order"] == pytest.approx(2.0)


def test_richardson_oscillating_and_short_and_constant():
    out = convergence.richardson([1.0, 2.0, 1.5], 1.5)
    assert out["monotonic"] is False and out["order"] is None and out["extrapolated"] is None
    assert convergence.richardson([1.0, 2.0], 1.5) == {"order": None, "extrapolated": None, "gci": None, "monotonic": None}
    const = convergence.richardson([3.0, 3.0, 3.0], 1.5)
    assert const["order"] == math.inf and const["extrapolated"] == 3.0 and const["gci"] == 0.0


def test_richardson_diverging_differences_give_negative_order():
    # |d2| > |d1| (differences grow) → p < 0, no extrapolation
    out = convergence.richardson([1.0, 1.1, 1.4], 1.5)
    assert out["order"] < 0 and out["extrapolated"] is None and out["gci"] is None


# ------------------------------------------------------------------ convergence: run_convergence with a mocked solve
class _Step:
    def __init__(self, n, value, peak):
        self.mises = np.zeros(n)
        self.mises[peak] = value
        self.umag = self.mises / 1000.0
        self.fields = {"stress_error": np.full(n, 3.0)}


def _fake_solver(law, calls, local=False):
    """run_study stand-in: the value of the 'mesh' = law(element size) at node 1 (`local`: size of the last local sphere)."""
    nodes = np.array([[0.0, 0, 0], [10.0, 0, 0], [50.0, 0, 0]])

    def run_study(st, workdir, **kw):
        h = st.mesh.local[-1]["size"] if local and st.mesh.local else st.mesh.size
        calls.append((h, [dict(x) for x in st.mesh.local], st.mesh.size))
        res = types.SimpleNamespace(mesh=types.SimpleNamespace(nodes=nodes, tets=np.zeros((5, 10), int)), modes=[])
        stp = _Step(len(nodes), law(h), 1)
        res.steps = [stp]
        res.step = lambda i: stp
        return res
    return run_study


def test_run_convergence_global_converging(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("fea_core.solve.run_study", _fake_solver(lambda h: 200.0 - 2.0 * h ** 2, calls))
    st = Study("t", "a.step")
    st.mesh.size = 4.0
    st.mesh.local = [{"center": [0, 0, 0], "radius": 5.0, "size": 2.0}]
    spec = convergence.ConvergenceSpec(levels=4, ratio=2.0, tol=0.02)
    out = convergence.run_convergence(st, tmp_path, spec)
    assert [c[2] for c in calls] == [4.0, 2.0, 1.0, 0.5]                 # h0 / r^k
    assert [c[1][0]["size"] for c in calls] == [2.0, 1.0, 0.5, 0.25]      # local sizes refined too
    assert st.mesh.size == 4.0 and st.mesh.local[0]["size"] == 2.0        # the input study is not changed
    assert out["order"] == pytest.approx(2.0) and out["extrapolated"] == pytest.approx(200.0)
    assert out["converged"] and out["verdict"].startswith("Converged")
    assert out["levels"][1]["change"] == pytest.approx((192.0 - 168.0) / 168.0)
    assert out["levels"][0]["stress_error_pct"] == 3.0 and out["levels"][0]["at"] == [10.0, 0.0, 0.0]
    assert out["warnings"] == []


def test_run_convergence_sphere_refinement_and_singularity_warning(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("fea_core.solve.run_study", _fake_solver(lambda h: 100.0 * h ** -0.3, calls, local=True))   # grows
    st = Study("t", "a.step")
    st.mesh.size = 3.0
    spec = convergence.ConvergenceSpec(levels=3, ratio=1.5, point=[10.0, 0, 0], radius=2.0)
    out = convergence.run_convergence(st, tmp_path, spec)
    assert calls[0][1] == [] and all(c[2] == 3.0 for c in calls)           # global size stays, a sphere is added from level 2
    assert calls[1][1] == [{"center": [10.0, 0.0, 0.0], "radius": 2.0, "size": 2.0}]
    assert calls[2][1][-1]["size"] == pytest.approx(3.0 / 1.5 ** 2)
    assert out["order"] == pytest.approx(-0.3)                              # differences grow: p < 0.5
    assert not out["converged"] and any("singularity" in w for w in out["warnings"])


def test_run_convergence_single_level(monkeypatch, tmp_path):
    monkeypatch.setattr("fea_core.solve.run_study", _fake_solver(lambda h: 5.0, []))
    out = convergence.run_convergence(Study("t", "a.step"), tmp_path, convergence.ConvergenceSpec(levels=1))
    assert out["verdict"].startswith("Only one level") and not out["converged"]


def test_region_selects_sphere_or_everything():
    X = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
    spec = convergence.ConvergenceSpec(point=[0, 0, 0], radius=1.5)
    assert convergence._region(X, spec).tolist() == [0, 1]
    assert convergence._region(X, convergence.ConvergenceSpec(point=[100, 0, 0], radius=1)).tolist() == [0, 1, 2]
    assert convergence._region(X, convergence.ConvergenceSpec()).tolist() == [0, 1, 2]


# ------------------------------------------------------------------ signature
def test_mesh_signature_depends_on_mesh_inputs_only():
    st = Study("t", "a.step")
    base = signature.mesh_signature(st)
    assert base == signature.mesh_signature(Study("t", "a.step"))            # deterministic
    st.materials["Steel"] = Material("Steel", 210000.0, 0.3)
    assert signature.mesh_signature(st) == base                               # materials do not change the mesh
    st.mesh.size = 2.5
    assert signature.mesh_signature(st) != base
    st2 = Study("t", "b.step")
    assert signature.mesh_signature(st2) != base
    assert signature.mesh_signature(st2, geometry=False) == signature.mesh_signature(Study("t", "a.step"), geometry=False)
    st3 = Study("t", "a.step")
    st3.ignored_parts = ["Magnet"]
    assert signature.mesh_signature(st3) != base


def test_mesh_signature_contacts_only_mesh_fields():
    st = Study("t", "a.step")
    assert "contacts" not in signature.mesh_inputs(st)                        # no contacts → the old signature
    st.contacts = [ContactPair("c1", "A", "B", kind="frictional", mu=0.2)]
    s1 = signature.mesh_signature(st)
    st.contacts[0].mu = 0.4                                                   # μ does not change the mesh
    st.contacts[0].stiffness = 1e5
    assert signature.mesh_signature(st) == s1
    st.contacts[0].slave = "B"
    assert signature.mesh_signature(st) != s1
    assert set(signature.mesh_inputs(st)["contacts"][0]) == set(signature.CONTACT_MESH_FIELDS)


# ------------------------------------------------------------------ ties
def test_point_tri_dist_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 400
    a, b, c = (rng.normal(size=(n, 3)) for _ in range(3))
    p = rng.normal(size=(n, 3)) * 2
    got = ties._point_tri_dist(p, a, b, c)
    s = np.linspace(0, 1, 121)
    u, v = np.meshgrid(s, s)
    m = (u + v) <= 1
    u, v = u[m], v[m]
    for i in range(0, n, 37):
        q = a[i] + np.outer(u, b[i] - a[i]) + np.outer(v, c[i] - a[i])
        ref = np.linalg.norm(q - p[i], axis=1).min()
        assert got[i] <= ref + 1e-12 and got[i] == pytest.approx(ref, abs=0.03 * max(ref, 1.0))


def _two_block_mesh(gap=0.0):
    """Part A: tet with the face z = 0 on top of nothing; part B: separate tet below, its face z = -gap facing A (own nodes)."""
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],                 # A (face 0-1-2 at z = 0)
                  [0, 0, -gap], [1, 0, -gap], [0, 1, -gap], [0, 0, -1 - gap]], float)   # B
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    faces_a = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
    faces_b = [[4, 5, 6], [4, 7, 5], [5, 7, 6], [4, 6, 7]]
    tri = np.array([f + f for f in faces_a + faces_b])                        # tri6 stand-in (corners repeated as midpoints)
    elem = np.array([0] * 4 + [1] * 4)
    mesh = types.SimpleNamespace(nodes=X, tets=tets, tet_part=np.array([0, 1]), part_names=["A", "B"], surfaces={},
                                 report={}, boundary_faces=lambda: (tri, elem))
    return mesh


def _tie_study(separate=("B",), tol=None):
    return types.SimpleNamespace(mesh=types.SimpleNamespace(separate=list(separate), tie_tolerance=tol, size=1.0),
                                 active_contacts=lambda: [])


def test_ties_detect_touching_faces(monkeypatch):
    monkeypatch.setattr("fea_core.solve.constraint_masks", lambda study, mesh: {})
    mesh = _two_block_mesh()
    warnings = ties.detect(_tie_study(), mesh)
    assert warnings == []
    assert ties.pairs(mesh) == [("__tie1_s", "__tie1_m")]
    s, m = mesh.surfaces["__tie1_s"], mesh.surfaces["__tie1_m"]
    assert len(s) == 1 and set(s[0, :3]) == {4, 5, 6}                         # the face of B on z = 0
    assert len(m) == 1 and set(m[0, :3]) == {0, 1, 2}
    t = mesh.report["ties"][0]
    assert t["area_mm2"] == pytest.approx(0.5) and t["tolerance_mm"] == pytest.approx(0.05)
    assert sorted(ties.tied_part_pairs(mesh)) == [(1, 0)]


def test_ties_detect_gap_beyond_tolerance_warns(monkeypatch):
    monkeypatch.setattr("fea_core.solve.constraint_masks", lambda study, mesh: {})
    mesh = _two_block_mesh(gap=0.2)
    w = ties.detect(_tie_study(), mesh)
    assert len(w) == 1 and "does not touch" in w[0] and not mesh.surfaces
    mesh = _two_block_mesh(gap=0.2)
    assert ties.detect(_tie_study(tol=0.3), mesh) == [] and ties.pairs(mesh)  # explicit tolerance bridges the gap


def test_ties_detect_drops_faces_with_supported_nodes(monkeypatch):
    mesh = _two_block_mesh()
    mask = np.zeros((len(mesh.nodes), 3), bool)
    mask[5, 2] = True                                                         # a node of the slave face is in *BOUNDARY
    monkeypatch.setattr("fea_core.solve.constraint_masks", lambda study, m: {"Fixed": mask})
    w = ties.detect(_tie_study(), mesh)
    assert any("support" in x for x in w)
    assert len(mesh.surfaces["__tie1_s"]) == 0


def test_ties_tolerance_default_and_nothing_separate():
    assert ties.tolerance(_tie_study(tol=0.7), None) == 0.7
    st = _tie_study()
    st.mesh.size = 0.001
    assert ties.tolerance(st, None) == 1e-3
    assert ties.detect(_tie_study(separate=()), _two_block_mesh()) == []


# ------------------------------------------------------------------ contour
def _area(t):
    return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2) if len(t) else 0.0


def _random_surface(n=60, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(3 * n, 3))
    X[:, 2] = 0.0
    tri = np.arange(3 * n).reshape(n, 3)
    return X, tri


def test_contour_bands_preserve_area_and_stay_in_band():
    X, tri = _random_surface()
    vals = 3.0 * X[:, 0] + X[:, 1]                                            # linear field → exact band boundaries
    edges = contour.band_edges(-2.0, 2.0, 8)                                  # values beyond ±2 fall into the end bands
    bands = contour.contour_bands(X, tri, vals, edges)
    assert len(bands) == 8
    assert sum(_area(b) for b in bands) == pytest.approx(_area(X[tri]), rel=1e-9)
    for k, b in enumerate(bands):
        if not len(b):
            continue
        v = 3.0 * b[..., 0] + b[..., 1]
        lo = edges[k] if k > 0 else -np.inf
        hi = edges[k + 1] if k < len(bands) - 1 else np.inf
        assert v.min() >= lo - 1e-9 and v.max() <= hi + 1e-9, k


def test_contour_tri6_and_degenerate_inputs():
    X = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    tri6 = np.array([[0, 1, 2, 3, 4, 5]])
    assert contour.linear_triangles(tri6).shape == (4, 3)
    bands = contour.contour_bands(X, tri6, X[:, 0], contour.band_edges(0, 2, 4))
    assert sum(_area(b) for b in bands) == pytest.approx(2.0)
    empty = contour.triangle_bands(np.zeros((0, 3, 3)), np.zeros((0, 3)), np.linspace(0, 1, 5))
    assert len(empty) == 4 and all(e.shape == (0, 3, 3) for e in empty)
    e = contour.band_edges(5.0, 5.0)                                          # constant field: a non-zero range
    assert len(e) == 13 and e[-1] > e[0]
    assert np.array_equal(contour.band_edges(np.nan, 1.0, 2), [0.0, 0.5, 1.0])
    assert contour.band_colors(12) == contour.BAND_COLORS
    c5 = contour.band_colors(5)
    assert len(c5) == 5 and c5[0] == contour.BAND_COLORS[0] and c5[-1] == contour.BAND_COLORS[-1]


def test_contour_constant_triangle_in_one_band():
    P = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    bands = contour.triangle_bands(P, np.full((1, 3), 0.55), np.linspace(0, 1, 11))
    assert [len(b) for b in bands] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

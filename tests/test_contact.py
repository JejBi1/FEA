"""Fast tests of the contacts (fea_core.contacts) without ccx: the model and the JSON, check of the settings, steps of the branches and
sequences, the scale of the state, reading of the CF from the .dat, an empty CONTACT block in the .frd, disconnection of a conforming joint.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fea_core.study import Study, Material, Force, Pressure, LoadCase
from fea_core.contacts.model import ContactPair, AnalysisSettings, LoadStep, Motion, ContactSettings
from fea_core.contacts.check import check
from fea_core.contacts.run import step_plans, scaled_case


@pytest.fixture(autouse=True)
def _mortar_available(monkeypatch):
    """These tests check what is written for the formulation the study asks for – independent of the ccx build (FEA_MORTAR, see
    config.mortar_supported; the substitution has its own tests below)."""
    monkeypatch.setenv("FEA_MORTAR", "1")


def _study():
    st = Study("t", "x.step")
    st.add_material(Material("ocel", 210000.0, 0.3, 7.85e-9), default=True)
    st.load_cases.append(LoadCase("F", [Force("a", [0, 0, 10.0]), Pressure("b", 2.0)]))
    return st


def test_contact_json_roundtrip():
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictional", mu=0.2, clearance=-0.01)]
    st.contact_settings = ContactSettings(method="mortar", nlgeom=True)
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False, sequence=[
        LoadStep("s1", case="F", scale=0.5, prescribed=[Motion("r", "rot", 3, [[0, 0], [1, 0.1]])],
                 mu_change={"p": 0.3}, cycles=2)])
    d = json.loads(json.dumps(st.to_dict()))
    assert "contacts" in d["features"] and "analysis" in d["features"]
    s2 = Study.from_dict(d)
    assert s2.contacts[0].clearance == -0.01 and s2.contact_settings.method == "mortar"
    assert s2.analysis.sequence[0].prescribed[0].kind == "rot" and s2.analysis.sequence[0].cycles == 2
    assert s2.is_nonlinear()


def test_check_errors():
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictional", mu=0.0),
                   ContactPair("q", "A", "B", kind="interference", clearance=0.01),
                   ContactPair("r", "A", "A", kind="frictionless")]
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False,
                                   sequence=[LoadStep("s", case="neni", mu_change={"x": 0.1},
                                                      prescribed=[Motion("nic")])])
    e, w = check(st, ["A", "B"])
    txt = "\n".join(e)
    for key in ("μ must be > 0", "negative radial", "self-contact", "'neni' does not exist",
                "non-existent pair 'x'", "rigid remote support"):
        assert key in txt, key


def test_step_plans_branches_and_sequence():
    st = _study()
    st.load_cases.append(LoadCase("G", []))
    plans = step_plans(st, preload=True)
    assert [p[0] for p in plans] == ["nl_b1", "nl_b2"]
    assert [s["kind"] for s in plans[0][1]] == ["preload", "case"] and plans[1][1][1]["case_index"] == 1
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False,
                                   sequence=[LoadStep("a", case="F"), LoadStep("b", case="G", cycles=2)])
    (job, steps), = step_plans(st, preload=False)
    assert [s["name"] for s in steps] == ["a (cycle 1)", "b (cycle 1)", "b (cycle 2)"]


def test_scaled_case():
    st = _study()
    lc = scaled_case(st.load_cases[0], 0.5)
    assert lc.loads[0].vector == [0, 0, 5.0] and lc.loads[1].value == 1.0
    assert st.load_cases[0].loads[0].vector == [0, 0, 10.0]       # the original unchanged


def test_contact_print_parse(tmp_path):
    from fea_core.contacts.results import contact_print
    p = tmp_path / "x.dat"
    p.write_text("""
 statistics for slave set C1S, master set C1M and time  0.1000000E+01

   total surface force (fx,fy,fz) and moment about the origin (mx,my,mz)

    1.0E+00  2.0E+00 -3.0E+02  4.0E+00  5.0E+00  6.0E+00

   center of gravity and mean normal

    1.0E+01  2.0E+01  3.0E+01  0.0E+00  0.0E+00 -1.0E+00
""", encoding="latin-1")
    c = contact_print(p)
    assert len(c) == 1 and c[0]["slave"] == "C1S" and c[0]["time"] == 1.0
    assert np.allclose(c[0]["force"], [1, 2, -300]) and np.allclose(c[0]["normal"], [0, 0, -1])


def test_frd_empty_contact_block(tmp_path):
    from fea_core.ccx.frd import read_frd
    head = (" 100CL  101 1.000000000       2                     0    1           1\n"
            " -4  CONTACT     6    1\n"
            " -5  COPEN       1    1    1    0\n -5  CSLIP1      1    1    1    0\n -5  CSLIP2      1    1    1    0\n"
            " -5  CPRESS      1    1    1    0\n -5  CSHEAR1     1    1    1    0\n -5  CSHEAR2     1    1    1    0\n"
            " -3\n")
    disp = (" 100CL  102 1.000000000       2                     0    1           1\n"
            " -4  DISP        4    1\n -5  D1          1    2    1    0\n -5  D2          1    2    2    0\n"
            " -5  D3          1    2    3    0\n -5  ALL         1    2    0    0    1ALL\n"
            " -1         1 1.00000E+00 2.00000E+00 3.00000E+00\n"
            " -1         2 4.00000E+00 5.00000E+00 6.00000E+00\n -3\n")
    p = tmp_path / "x.frd"
    p.write_text("    1PSTEP                         1           1           1\n" + head + disp + "9999\n",
                 encoding="latin-1")
    f = read_frd(p)
    names = [b.name for b in f.blocks]
    assert names == ["CONTACT", "DISP"]
    assert len(f.blocks[0].node_ids) == 0
    assert np.allclose(f.get("DISP").values[1], [4, 5, 6])


def test_split_interface_two_tets():
    from fea_core.mesh import Mesh
    from fea_core.contacts.detect import split_interface
    # two tetrahedra with a common face (0, 1, 2), tet10 with the midpoints of the edges
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    edges = {}

    def mid(a, b):
        k = tuple(sorted((a, b)))
        if k not in edges:
            edges[k] = len(X) + len(edges)
        return edges[k]
    t1 = [0, 1, 2, 3]
    t2 = [0, 2, 1, 4]
    tets = []
    for t in (t1, t2):
        a, b, c, d = t
        tets.append([a, b, c, d, mid(a, b), mid(b, c), mid(c, a), mid(a, d), mid(b, d), mid(c, d)])
    mids = np.array([(X[a] + X[b]) / 2 for (a, b), _ in sorted(edges.items(), key=lambda kv: kv[1])])
    nodes = np.vstack([X, mids])
    mesh = Mesh(nodes, np.array(tets), np.array([0, 1]), ["A", "B"])
    n = split_interface(mesh, "A", "B")
    assert n == 6                                            # 3 corners + 3 midpoints of the common face
    assert len(mesh.nodes) == len(nodes) + 6
    assert not set(mesh.tets[0].tolist()) & set(mesh.tets[1].tolist())   # the parts do not share nodes any more
    a, b = mesh.surfaces["__split_A|B_a"], mesh.surfaces["__split_A|B_b"]
    # the same face geometry, opposite orientation (each side from its own element)
    assert np.allclose(np.sort(mesh.nodes[a].reshape(-1, 3), axis=0), np.sort(mesh.nodes[b].reshape(-1, 3), axis=0))
    assert set(mesh.canonical()[b].ravel().tolist()) == set(a.ravel().tolist())


# ------------------------------------------------------------------ A6: reliable measures of the contact pressure
def _tri6_square(n: int = 4, size: float = 4.0):
    """A square face of n × n cells (2 triangles tri6 per cell), a fine grid (2n+1)² nodes with the midpoints of the edges."""
    m = 2 * n + 1
    h = size / (2 * n)
    X = np.array([[i * h, j * h, 0.0] for j in range(m) for i in range(m)])
    idx = lambda i, j: j * m + i                                             # noqa: E731
    tris = []
    for cj in range(n):
        for ci in range(n):
            i, j = 2 * ci, 2 * cj
            for a, b, c in (((i, j), (i + 2, j), (i + 2, j + 2)), ((i, j), (i + 2, j + 2), (i, j + 2))):
                mid = lambda p, q: ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)  # noqa: E731
                tris.append([idx(*a), idx(*b), idx(*c), idx(*mid(a, b)), idx(*mid(b, c)), idx(*mid(c, a))])
    return X, np.array(tris, np.int64)


def _fake_mesh(X, held=()):
    from types import SimpleNamespace
    return SimpleNamespace(nodes=X, surface_nodes=lambda name: np.asarray(held, np.int64))


def test_free_edge_nodes_square():
    from fea_core.contacts.results import free_edge_nodes
    X, tri = _tri6_square(4)
    e = free_edge_nodes(tri)
    on_edge = (np.isclose(X[:, 0], 0) | np.isclose(X[:, 0], 4) | np.isclose(X[:, 1], 0) | np.isclose(X[:, 1], 4))
    assert set(e.tolist()) == set(np.nonzero(on_edge)[0].tolist())         # corners and midpoints of the edges of the perimeter, not the inside


def _metrics(p_all, fn=None, study=None, mesh=None):
    from fea_core.contacts.results import node_weights, pressure_metrics
    X, tri = _tri6_square(4)
    mesh = mesh or _fake_mesh(X)
    nodes, w = node_weights(mesh, tri)
    p = p_all[nodes]
    return pressure_metrics(study or _study(), mesh, tri, nodes, w, p, float((p * w).sum()) if fn is None else fn,
                            p_all), nodes


def test_pressure_metrics_uniform_and_peaks():
    X, tri = _tri6_square(4)
    p0 = np.full(len(X), 10.0)
    met, nodes = _metrics(p0)
    assert met["p_force_MPa"] == pytest.approx(10.0) and met["area_active_mm2"] == pytest.approx(16.0)
    assert not met["peak_at_edge"] and met["p_max_interior_MPa"] == pytest.approx(10.0)
    # a peak at the perimeter (midpoint of an edge) 3× above the inside → an edge peak
    edge_mid = int(next(n for n in nodes if np.isclose(X[n, 1], 0) and np.isclose(X[n, 0], 0.5)))
    p1 = p0.copy()
    p1[edge_mid] = 30.0
    met, _ = _metrics(p1)
    assert met["peak_at_edge"] and met["p_max_interior_MPa"] == pytest.approx(10.0) and met["peak_edge_ratio"] == pytest.approx(3.0)
    # the same peak in the middle of the face = not an edge peak
    center = int(next(n for n in nodes if np.allclose(X[n], [1.5, 1.5, 0.0])))
    p2 = p0.copy()
    p2[center] = 30.0
    met, _ = _metrics(p2)
    assert not met["peak_at_edge"] and met["p_max_interior_MPa"] == pytest.approx(30.0)


def test_pressure_metrics_symmetry_edge_is_not_free():
    """Nodes held by a constraint (a plane of symmetry) are not a free edge – a peak there is not conspicuous."""
    from fea_core.study import Displacement
    X, tri = _tri6_square(4)
    p = np.full(len(X), 10.0)
    on_x0 = np.nonzero(np.isclose(X[:, 0], 0))[0]
    spike = int(next(n for n in on_x0 if np.isclose(X[n, 1], 0.5)))
    p[spike] = 30.0
    st = _study()
    st.constraints.append(Displacement("sym", ux=0.0))
    met, _ = _metrics(p, study=st, mesh=_fake_mesh(X, held=on_x0))
    assert not met["peak_at_edge"]
    met_free, _ = _metrics(p)                                # without a constraint it is a free edge
    assert met_free["peak_at_edge"]


def test_pressure_force_uses_exact_force_and_threshold():
    """p_force = Fn / area of the nodes with p ≥ 5 % of the average: the noise at the contact boundary does not enlarge the area."""
    X, tri = _tri6_square(4)
    p = np.zeros(len(X))
    p[X[:, 0] <= 2.0 + 1e-9] = 10.0
    met0, _ = _metrics(p, fn=80.0)
    noisy = p.copy()
    noisy[np.isclose(X[:, 0], 3.0)] = 0.1                   # a barely visible noise (1 % of the pressure) beyond the contact boundary
    met, _ = _metrics(noisy, fn=80.0)                       # the exact force from CF
    assert met["area_active_mm2"] == pytest.approx(met0["area_active_mm2"])
    assert met["p_force_MPa"] == pytest.approx(80.0 / met0["area_active_mm2"])
    assert met["area_active_mm2"] == pytest.approx(8.0, rel=0.1)


def test_edge_peak_warnings_text():
    from types import SimpleNamespace
    from fea_core.contacts.results import edge_peak_warnings
    ok = SimpleNamespace(name="a", contacts={"P": {"peak_at_edge": False}})
    bad = SimpleNamespace(name="b", contacts={"P": {"peak_at_edge": True, "peak_edge_ratio": 5.4, "p_max_MPa": 579.0,
                                                    "p_max_interior_MPa": 107.0, "p_force_MPa": 26.8}})
    assert edge_peak_warnings([ok]) == []
    w = edge_peak_warnings([ok, bad, bad])
    assert len(w) == 1 and "Contact 'P'" in w[0] and "free edge" in w[0] and "579" in w[0] and "26.8" in w[0]


# ------------------------------------------------------------------ A7: rotation of a rigid body without NLGEOM
def test_rigid_rotation_warning():
    from fea_core.remote import rigid_rotations, rigid_rotation_note, RIGID_ROT_WARN
    from fea_core.study import RemoteLoad
    st = _study()
    st.load_cases[0].loads.append(RemoteLoad("vrch", [0, 0, 50.0], force=[0, 0, -1.0], coupling="rigid"))
    ref_nodes = {"0:0": {"ref": 101, "rot": 102}}
    refs = {101: np.array([1.0, 0, 0]), 102: np.array([0.0, 0.05, 0.0])}
    items = rigid_rotations(st, ref_nodes, refs, None)
    assert items == [("vrch", pytest.approx(0.05))]
    note = rigid_rotation_note(items, "Krok 1", nlgeom=False)
    assert note and "vrch" in note and "NLGEOM" in note and "distributing" in note
    assert rigid_rotation_note(items, "Krok 1", nlgeom=True) is None          # with large deformations it is fine
    assert rigid_rotation_note([("vrch", 0.5 * RIGID_ROT_WARN)], "Krok 1", nlgeom=False) is None
    # rigid remote constraint (RemoteSupport): the rotation from the reactions of the step
    sup = rigid_rotations(st, {}, None, {"pata": {"rot": np.array([0.0, 0.0, 0.04])}})
    assert sup == [("pata", pytest.approx(0.04))] and rigid_rotation_note(sup, "K", False)


# ---------------------------------------------------------------- default K = 10·E, increments of branches, duplicate auto pair
def _written_pairs(st, pairs, method=None):
    """Text of the contact cards written by write_pairs for `pairs` (no meshing: the mesh report has only the surface names)."""
    import io
    from types import SimpleNamespace
    from fea_core.ccx.inp import _fmt
    from fea_core.contacts.inp import write_pairs
    st.contacts = list(pairs)
    if method:
        st.contact_settings.method = method
    mesh = SimpleNamespace(report={"contacts": [{"pair": c.name, "slave": f"S{i}", "master": f"M{i}", "gap_min_mm": 0.0}
                                               for i, c in enumerate(pairs)]})
    f = io.StringIO()
    write_pairs(f, st, mesh, SimpleNamespace(), lambda x: x, _fmt, 1, 1)
    return f.getvalue()


def test_default_penalty_stiffness_is_10_E_and_is_written():
    from fea_core.contacts.inp import pair_params
    from fea_core.ccx.inp import _fmt
    st = _study()
    st.add_material(Material("hlinik", 70000.0, 0.33, 2.7e-9))
    st.part_materials["B"] = "hlinik"
    c = ContactPair("p", "A", "B", kind="frictionless")
    pp = pair_params(st, None, c)
    assert pp["K"] == pytest.approx(10 * 70000.0) and pp["lambda"] == pytest.approx(pp["K"] / 100)
    # the card of the .inp: K of the smaller E, in *SURFACE BEHAVIOR
    txt = _written_pairs(st, [c])
    assert f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n{_fmt(10 * 70000.0)}\n" in txt
    # explicit values are written as they are
    c2 = ContactPair("q", "A", "B", kind="frictionless", stiffness=1e6, stick_slope=500.0)
    pp2 = pair_params(st, None, c2)
    assert pp2["K"] == 1e6 and pp2["lambda"] == 500.0
    assert f"PRESSURE-OVERCLOSURE=LINEAR\n{_fmt(1e6)}\n" in _written_pairs(st, [c2])
    # a friction pair: the stick slope λ = K/100 under *FRICTION
    c3 = ContactPair("r", "A", "B", kind="frictional", mu=0.2)
    assert f"*FRICTION\n{_fmt(0.2)}, {_fmt(10 * 70000.0 / 100)}\n" in _written_pairs(st, [c3])


def test_interference_keeps_50_E_and_hard_mortar_keeps_its_stick_slope():
    from fea_core.contacts.inp import pair_params
    from fea_core.ccx.inp import _fmt
    st = _study()
    ci = ContactPair("lis", "A", "B", kind="interference", clearance=-0.01)
    pp = pair_params(st, None, ci)
    assert pp["K"] == pytest.approx(50 * 210000.0) and pp["lambda"] == pytest.approx(50 * 2100.0)
    assert f"PRESSURE-OVERCLOSURE=LINEAR\n{_fmt(50 * 210000.0)}\n" in _written_pairs(st, [ci])
    # hard mortar contact: no K card, but the friction stick slope stays at the value of the former K = 50·E (not 5× softer)
    cf = ContactPair("f", "A", "B", kind="frictional", mu=0.2)
    txt = _written_pairs(st, [cf], method="mortar")
    assert "*SURFACE BEHAVIOR" not in txt and f"*FRICTION\n{_fmt(0.2)}, {_fmt(50 * 210000.0 / 100)}\n" in txt
    assert pair_params(st, None, cf)["lambda"] == pytest.approx(50 * 2100.0)
    # mortar with an explicit K writes it, λ = K/100 as for the penalty methods
    cs = ContactPair("s", "A", "B", kind="frictional", mu=0.2, stiffness=1e6)
    txt = _written_pairs(st, [cs], method="mortar")
    assert f"PRESSURE-OVERCLOSURE=LINEAR\n{_fmt(1e6)}\n" in txt and f"{_fmt(1e6 / 100)}\n" in txt


def test_branch_increments_automatic_by_formulation_json_and_step_header():
    from fea_core.contacts.inp import step_header
    from fea_core.contacts.model import branch_increments
    st = _study()
    st.analysis = AnalysisSettings(kind="nonlinear")
    an = st.analysis
    assert (an.dt0, an.dt_max, an.dt_min, an.max_inc) == (None, None, 1e-5, 200)
    # no contact (plasticity, NLGEOM, always nonlinear): the former 0.1 / 0.25
    eff = branch_increments(st)
    assert (eff["dt0"], eff["dt_max"], eff["auto"]) == (0.1, 0.25, ["dt0", "dt_max"])
    # surface-to-surface and node-to-surface contact 0.25 / 0.5, mortar 0.1 / 0.25
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]
    for method, want in (("surface_to_surface", (0.25, 0.5)), ("node_to_surface", (0.25, 0.5)), ("mortar", (0.1, 0.25))):
        st.contact_settings.method = method
        eff = branch_increments(st)
        assert (eff["dt0"], eff["dt_max"]) == want, method
        head, direct, inc = step_header(st)                     # a branch / automatic preload step: no LoadStep
        assert head == "*STEP, INC=200" and direct == "" and inc == f"{want[0]:.6g}, 1., 1e-05, {want[1]:.6g}"
    # the automatic values are not written, an old file (without the keys) is automatic
    d = json.loads(json.dumps(st.to_dict()))
    assert not {"dt0", "dt_min", "dt_max", "max_inc"} & set(d["analysis"])
    assert Study.from_dict(d).analysis.dt0 is None and Study.from_dict(d).analysis.dt_max is None
    # explicit values override, are written and read back; a LoadStep keeps its own
    st.contact_settings.method = "mortar"
    an.dt0, an.dt_min, an.dt_max, an.max_inc = 0.05, 1e-6, 0.1, 500
    d = json.loads(json.dumps(st.to_dict()))
    assert d["analysis"]["dt0"] == 0.05 and d["analysis"]["dt_max"] == 0.1 and d["analysis"]["max_inc"] == 500
    s2 = Study.from_dict(d)
    assert (s2.analysis.dt0, s2.analysis.dt_min, s2.analysis.dt_max, s2.analysis.max_inc) == (0.05, 1e-6, 0.1, 500)
    head, direct, inc = step_header(s2)
    assert head == "*STEP, INC=500" and inc == "0.05, 1., 1e-06, 0.1" and branch_increments(s2)["auto"] == []
    own = LoadStep("s", dt0=0.2, dt_min=1e-4, dt_max=0.3, max_inc=7)
    head, direct, inc = step_header(s2, own)
    assert head == "*STEP, INC=7" and inc == "0.2, 1., 0.0001, 0.3"
    # only one of the pair given: the automatic neighbour never makes dt0 > dt_max
    an.dt0, an.dt_max = 0.4, None                               # mortar automatic dt_max 0.25 < 0.4
    assert (branch_increments(st)["dt0"], branch_increments(st)["dt_max"], branch_increments(st)["auto"]) == (0.4, 0.4, ["dt_max"])
    an.dt0, an.dt_max = None, 0.05                              # automatic dt0 0.1 > 0.05
    assert (branch_increments(st)["dt0"], branch_increments(st)["dt_max"]) == (0.05, 0.05)
    # the increments do not touch the mesh
    from fea_core.signature import mesh_signature
    sig = mesh_signature(st, geometry=False)
    an.dt0, an.dt_max, an.max_inc = 0.3, 0.6, 10
    assert mesh_signature(st, geometry=False) == sig


def test_branch_increments_are_written_in_the_inp(tmp_path):
    """write_inp (a real .inp of a bar without contact): the *STATIC increment line and *STEP, INC= of a branch, of the automatic
    preload step of a sequence and of a LoadStep with its own values."""
    from fea_core.ccx.inp import write_inp
    from test_loads import bar_mesh, bar_study
    st, m = bar_study(), bar_mesh()
    st.load_cases = [LoadCase("F", [Force("x100", [0.0, 0.0, -100.0])])]

    def inp(name, steps):
        write_inp(st, m, tmp_path / f"{name}.inp", nonlinear={"steps": steps, "part_nodes": {}})
        return (tmp_path / f"{name}.inp").read_text()
    st.analysis = AnalysisSettings(kind="nonlinear")
    txt = inp("auto", step_plans(st, preload=False)[0][1])                      # no contact: the former 0.1 / 0.25
    assert "*STEP, INC=200\n" in txt and "\n0.1, 1., 1e-05, 0.25\n" in txt
    st.analysis = AnalysisSettings(kind="nonlinear", dt0=0.3, dt_min=1e-4, dt_max=0.6, max_inc=40)
    txt = inp("own", step_plans(st, preload=True)[0][1])                        # [preload step, state]: both without a LoadStep
    assert txt.count("*STEP, INC=40\n") == 2 and txt.count("\n0.3, 1., 0.0001, 0.6\n") == 2
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False, dt0=0.3, dt_max=0.6, max_inc=40,
                                   sequence=[LoadStep("s1", case="F", dt0=0.02, dt_min=1e-6, dt_max=0.05, max_inc=7)])
    txt = inp("seq", step_plans(st, preload=True)[0][1])                        # automatic preload step (branch values) + own step
    assert "*STEP, INC=40\n" in txt and "\n0.3, 1., 1e-05, 0.6\n" in txt
    assert "*STEP, INC=7\n" in txt and "\n0.02, 1., 1e-06, 0.05\n" in txt


def test_check_increments():
    st = _study()
    st.analysis = AnalysisSettings(kind="nonlinear", dt0=0.5, dt_min=0.6)       # dt_min > dt0

    def inc_errors():
        return [x for x in check(st, ["A", "B"])[0] if "ncrement" in x or "increments" in x]
    assert any("Increment control of the steps (branches" in x for x in inc_errors())
    st.analysis = AnalysisSettings(kind="nonlinear", dt0=0.5, dt_max=2.0)       # dt_max > 1
    assert inc_errors()
    st.analysis = AnalysisSettings(kind="nonlinear", dt0=0.5, dt_max=0.25)      # dt0 > dt_max
    assert any("dt_min ≤ dt0 ≤ dt_max" in x for x in inc_errors())
    st.analysis = AnalysisSettings(kind="nonlinear", max_inc=0)
    assert any("positive integer" in x for x in inc_errors())
    st.analysis = AnalysisSettings(kind="nonlinear", max_inc=2.5)
    assert any("positive integer" in x for x in inc_errors())
    for bad in (None, "x", float("nan"), True):                                 # a null / text / NaN from a foreign file: an error, never a TypeError
        st.analysis = AnalysisSettings(kind="nonlinear", dt_min=bad)
        assert any("dt_min must be a number" in x for x in inc_errors()), bad
    st.analysis = AnalysisSettings(kind="nonlinear", dt0="0.1")
    assert any("dt0 must be a number" in x for x in inc_errors())
    st.analysis = AnalysisSettings(kind="nonlinear")
    assert not inc_errors()
    st.analysis = AnalysisSettings(kind="linear", dt0=5.0)                      # unused in a linear analysis without contact
    assert not inc_errors()
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]             # a contact makes the run nonlinear
    assert inc_errors()
    # plasticity alone (a study of the kind "linear" that runs nonlinearly through step_header)
    st.contacts = []
    st.add_material(Material("plast", 210000.0, 0.3, 7.85e-9, Re=250.0, plastic={"model": "bilinear", "Et": 1000.0}))
    st.default_material = "plast"
    st.analysis = AnalysisSettings(kind="linear", dt0=5.0)
    assert st.is_nonlinear() and inc_errors()
    # steps of a sequence: the same rules (dt0 > dt_max, numbers, integer count)
    st.default_material = "ocel"
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False, sequence=[
        LoadStep("a", case="F", dt0=0.5, dt_max=0.25), LoadStep("b", case="F", max_inc=2.5), LoadStep("c", case="F", dt0=None)])
    txt = "\n".join(inc_errors())
    assert "Step 'a'" in txt and "Step 'b'" in txt and "Step 'c'" in txt and "dt0 must be a number" in txt


def _bolted_study():
    from fea_core.joints.model import BoltSpec
    st = _study()
    st.bolts = [BoltSpec("M10", plates=[{"part": "Upper", "s0": 0.0, "s1": 10.0, "hole": 11.0},
                                        {"part": "Lower", "s0": 10.0, "s1": 20.0, "hole": 11.0}], head_part="Upper")]
    st.joints.contact_joint = True
    return st


def test_all_contacts_user_pair_replaces_automatic_pair_of_the_interface():
    st = _bolted_study()
    auto = st.all_contacts()
    assert [(c.name, c.source) for c in auto] == [("Interface Lower–Upper", "auto")]
    # a pair of the study with the same name replaces the automatic one (as before)
    st.contacts = [ContactPair("Interface Lower–Upper", "Lower", "Upper", kind="frictional", mu=0.3)]
    assert [(c.name, c.mu) for c in st.all_contacts()] == [("Interface Lower–Upper", 0.3)]
    # ... and a pair of the same two parts under another name (e.g. a name from an older version, in either order) as well
    for name, a, b in (("Spára Lower–Upper", "Lower", "Upper"), ("Spára Patka–Deska", "Upper", "Lower")):
        st.contacts = [ContactPair(name, a, b, kind="frictional", mu=0.3, stiffness=2e6)]
        got = st.all_contacts()
        assert [c.name for c in got] == [name], got
        assert [c.name for c in st.active_contacts()] == [name]
    # a pair of other parts leaves the automatic pair alone
    st.contacts = [ContactPair("Jiná", "Upper", "Third", kind="frictional", mu=0.3)]
    assert [c.name for c in st.all_contacts()] == ["Jiná", "Interface Lower–Upper"]
    # a user pair on a face selection (not the part names) does not hide the automatic pair either
    st.contacts = [ContactPair("Sel", "selA", "selB", kind="frictionless")]
    assert [c.name for c in st.all_contacts()] == ["Sel", "Interface Lower–Upper"]


def test_disabled_bonded_or_none_user_pair_on_the_bolted_parts_warns():
    """Design: such a user pair still suppresses the automatic pair (override), but the readiness check says that the bolted joint is then
    not a contact."""
    st = _bolted_study()

    def warns():
        return [w for w in check(st, ["Upper", "Lower"])[1] if "automatic contact interface" in w]
    assert not warns()                                                        # no user pair: the automatic contact joint
    st.contacts = [ContactPair("Spára", "Upper", "Lower", kind="frictional", mu=0.2)]
    assert not warns()                                                        # an enabled contact pair is a proper override
    for kw in ({"enabled": False}, {"kind": "bonded"}, {"kind": "none"}):
        st.contacts = [ContactPair("Spára", "Lower", "Upper", **{"kind": "frictional", "mu": 0.2, **kw})]
        w = warns()
        assert len(w) == 1 and "'Spára'" in w[0] and "NOT a contact" in w[0], (kw, w)
    st.joints.contact_joint = False                                           # contact joint off: nothing to suppress
    assert not warns()


def test_cf_scale_of_a_bolt_interface_is_found_by_the_parts_not_by_the_name():
    """F_clamp_contact is scaled by Fn(ccx CF) / Fn(nodal CPRESS) of the pair of the interface: the automatic pair, a user pair of any name
    (after the deduplication it replaces the automatic one) and parts with "–" in their names."""
    from fea_core.contacts.run import cf_scale
    st = _bolted_study()
    res = lambda name: {name: {"Fn_N": 90.0, "Fn_nodal_N": 100.0, "source": "ccx CF"}}
    assert cf_scale(st, "Upper", "Lower", res("Interface Lower–Upper")) == pytest.approx(0.9)       # the automatic pair
    st.contacts = [ContactPair("Moje spára", "Upper", "Lower", kind="frictional", mu=0.3)]
    assert cf_scale(st, "Upper", "Lower", res("Moje spára")) == pytest.approx(0.9)               # a renamed user pair
    assert cf_scale(st, "Lower", "Upper", res("Moje spára")) == pytest.approx(0.9)               # either order
    assert cf_scale(st, "Upper", "Third", res("Moje spára")) == 1.0                               # another interface
    assert cf_scale(st, "Upper", "Lower", {"Moje spára": {"Fn_N": 90.0, "Fn_nodal_N": 100.0, "source": "nodal"}}) == 1.0
    assert cf_scale(st, "Upper", "Lower", None) == 1.0 and cf_scale(st, "Upper", "Lower", res("neexistuje")) == 1.0
    # part names with the dash that the old name parsing split
    st2 = _study()
    from fea_core.joints.model import BoltSpec
    st2.bolts = [BoltSpec("M8", plates=[{"part": "Deska A–1", "s0": 0.0, "s1": 5.0, "hole": 9.0},
                                        {"part": "Deska B–2", "s0": 5.0, "s1": 10.0, "hole": 9.0}], head_part="Deska A–1")]
    st2.joints.contact_joint = True
    name = st2.all_contacts()[0].name
    assert cf_scale(st2, "Deska A–1", "Deska B–2", res(name)) == pytest.approx(0.9)
    assert cf_scale(st2, "Deska A–1", "Deska C", res(name)) == 1.0


def test_mortar_is_replaced_by_surface_to_surface_where_ccx_cannot_solve_it(monkeypatch):
    """The mortar matrix of ccx 2.23 is rounding-dependent (Windows NaN, Windows and Linux SPOOLES singular – Hertz case K2): the written formulation
    is surface-to-surface with K = 50·E (K2: peak pressure −1.2 %, 10·E gave −6.8 %) and the check warns."""
    from fea_core.contacts.inp import pair_params
    from fea_core.contacts.model import effective_method, mortar_substituted, auto_increments
    from fea_core.ccx.inp import _fmt
    st = _study()
    st.contact_settings.method = "mortar"
    c = ContactPair("p", "A", "B", kind="frictionless")
    st.contacts = [c]
    monkeypatch.setenv("FEA_MORTAR", "1")
    assert effective_method(st) == "mortar" and not mortar_substituted(st)
    assert "TYPE=MORTAR" in _written_pairs(st, [c]) and auto_increments(st) == (0.1, 0.25)
    monkeypatch.setenv("FEA_MORTAR", "0")
    assert effective_method(st) == "surface_to_surface" and mortar_substituted(st)
    txt = _written_pairs(st, [c])
    assert "TYPE=SURFACE TO SURFACE" in txt and "MORTAR" not in txt
    assert f"PRESSURE-OVERCLOSURE=LINEAR\n{_fmt(50 * 210000.0)}\n" in txt           # hard contact -> the stiff penalty
    assert pair_params(st, None, c)["K"] == pytest.approx(50 * 210000.0)
    assert auto_increments(st) == (0.25, 0.5)
    assert st.contact_settings.method == "mortar"                                 # the study keeps the user's choice
    _, w = check(st, ["A", "B"])
    assert any("mortar is not used" in x for x in w)
    # an explicit K is kept
    c2 = ContactPair("q", "A", "B", kind="frictionless", stiffness=1e6)
    assert f"PRESSURE-OVERCLOSURE=LINEAR\n{_fmt(1e6)}\n" in _written_pairs(st, [c2])

"""Registry of load / constraint kinds (fea_core.load_kinds): every kind of study.LOAD_KINDS has a complete handler, and
the dispatch through the registry gives BIT-IDENTICAL nodal loads, resultants, scaling and readiness messages as the
isinstance chains it replaced (kept below verbatim as references `_old_*`). Fast – synthetic meshes, no Gmsh / ccx.
The byte-identical .inp of the verification cases was checked separately (scratch/baseline_inp, 9. 11. 2026).
"""
import copy
import math
from dataclasses import dataclass

import numpy as np
import pytest

from fea_core import load_kinds as K
from fea_core import loads as L
from fea_core import study as S
from fea_core.study import (Force, Pressure, Gravity, Rotation, RemoteLoad, BearingLoad, Inertia, ThermalLoad,
                            FrameNodeLoad, FrameMemberLoad, FrameTemperature, LoadCase, CoordinateSystem)
from test_loads import bar_mesh, bar_study
from test_cylindrical import ring_mesh, _study as ring_study


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ======================================================================= completeness (meta-test)

def test_every_kind_has_complete_handler():
    """A new class in study.LOAD_KINDS / CONSTRAINT_KINDS without a handler in fea_core/load_kinds.py fails here."""
    from dataclasses import fields
    assert K.missing_handlers() == []
    for k, cls in S.LOAD_KINDS.items():
        h = K.LOAD_HANDLERS[k]
        assert h.kind == k and h.cls is cls
        assert {f.name: f.default for f in fields(cls)}["kind"] == k
        assert h.domain in ("solid", "frame", "thermal")
        assert set(h.entities) <= {"faces", "edges", "points"}
        assert h.csys in ("none", "cartesian", "cylindrical")
        assert h.backends <= K.ALL_BACKENDS
        assert callable(h.scale) and callable(h.describe)
        if h.domain == "solid":
            assert h.nodal is not None or h.remote is not None
        else:
            assert h.nodal is None and h.remote is None and h.ccx_cards is None
    for k, cls in S.CONSTRAINT_KINDS.items():
        h = K.CONSTRAINT_HANDLERS[k]
        assert h.cls is cls and callable(h.describe) and h.entities == ("faces",)


def test_selection_fields_match_entities():
    from dataclasses import fields
    for k, cls in S.LOAD_KINDS.items():
        h = K.LOAD_HANDLERS[k]
        if h.domain != "solid":
            continue
        has_sel = "selection" in {f.name for f in fields(cls)}
        assert has_sel == bool(h.entities), k
        assert ("csys" in {f.name for f in fields(cls)}) == (h.csys != "none"), k


def test_missing_handler_is_detected(monkeypatch):
    @dataclass
    class NewLoad:
        selection: str = "x"
        kind: str = "new_load"

    monkeypatch.setitem(S.LOAD_KINDS, "new_load", NewLoad)
    assert ("load", "new_load", "no_handler") in K.missing_handlers()
    incomplete = K.LoadKind("new_load", NewLoad, "solid", entities=("faces",), describe=lambda l, st, n: "x")
    monkeypatch.setitem(K.LOAD_HANDLERS, "new_load", incomplete)
    miss = K.missing_handlers()
    assert ("load", "new_load", "no_scale") in miss and ("load", "new_load", "no_forces") in miss


def test_unknown_kind_is_ignored_as_before():
    m, st = bar_mesh(), bar_study()
    lc = LoadCase("L", [Force("x100", [0, 0, -10.0]), type("Odd", (), {"kind": "odd"})()])
    F, Fr, Mr = L.nodal_loads(st, m, lc)
    assert np.allclose(F.sum(axis=0), [0, 0, -10.0]) and not Fr.any()
    assert np.allclose(L.load_resultant(st, m, lc), [0, 0, -10.0])


# ======================================================================= the old chains (references, verbatim)

def _old_nodal_loads(study, mesh, lc):
    F = np.zeros((len(mesh.nodes), 3))
    Fr, Mr = np.zeros(3), np.zeros(3)
    for l in lc.loads:
        if isinstance(l, Force):
            F += L.force_nodal(study, mesh, l)
        elif isinstance(l, Pressure):
            F += L.pressure_nodal(mesh, l)
        elif isinstance(l, Rotation):
            F += L.rotation_forces(study, mesh, l)
        elif getattr(l, "kind", None) == "inertia":
            F += L.inertia_forces(study, mesh, l)
        elif isinstance(l, Gravity):
            F += L.gravity_nodal_forces(study, mesh, L.global_vector(study, l))
        elif isinstance(l, BearingLoad):
            F += L.bearing_nodal(study, mesh, l)
        elif isinstance(l, RemoteLoad):
            f, mm = L.remote_vectors(study, l)
            Fr += f
            Mr += mm + np.cross(np.asarray(l.point, float), f)
    return F, Fr, Mr


def _old_load_resultant(study, mesh, lc):
    from fea_core import csys
    from fea_core.printing import element_density
    F = np.zeros(3)
    for l in lc.loads:
        if isinstance(l, Force):
            cs = csys.get(study, l.csys)
            if cs is not None and cs.kind == "cylindrical":
                F += L.force_nodal(study, mesh, l).sum(axis=0)
            else:
                F += L.global_vector(study, l)
        elif isinstance(l, Pressure) and l.variable:
            F += L.pressure_nodal(mesh, l).sum(axis=0)
        elif isinstance(l, Pressure):
            F += -l.value * L.vector_area(mesh, l.selection)
        elif isinstance(l, Gravity):
            rho = element_density(study, mesh)
            F += (L.element_volumes(mesh) * rho).sum() * L.global_vector(study, l)
        elif isinstance(l, Rotation):
            F += L.rotation_forces(study, mesh, l).sum(axis=0)
        elif getattr(l, "kind", None) == "inertia":
            F += L.inertia_forces(study, mesh, l).sum(axis=0)
        elif isinstance(l, BearingLoad):
            F += L.global_vector(study, l)
        elif isinstance(l, RemoteLoad):
            F += L.remote_vectors(study, l)[0]
    return F


def _old_scale(study, factor):
    for lc in study.load_cases:
        for l in lc.loads:
            kind = getattr(l, "kind", "")
            if kind in ("force", "bearing", "gravity"):
                if hasattr(l, "vector") and l.vector:
                    l.vector = [float(v) * factor for v in l.vector]
            elif kind == "pressure":
                if hasattr(l, "value"):
                    l.value = float(l.value) * factor
                if hasattr(l, "gradient") and l.gradient:
                    l.gradient = [float(g) * factor for g in l.gradient]
            elif kind == "remote":
                if hasattr(l, "force") and l.force:
                    l.force = [float(v) * factor for v in l.force]
                if hasattr(l, "moment") and l.moment:
                    l.moment = [float(v) * factor for v in l.moment]
            elif kind == "rotation":
                if factor < 0 and getattr(l, "omega", 0.0) != 0.0:
                    raise ValueError("rotation")
                if hasattr(l, "omega"):
                    l.omega = float(l.omega) * math.sqrt(factor) if factor >= 0 else float(l.omega)
                if hasattr(l, "alpha"):
                    l.alpha = float(l.alpha) * factor
            elif kind == "inertia":
                if hasattr(l, "accel") and l.accel:
                    l.accel = [float(a) * factor for a in l.accel]
                if hasattr(l, "alpha") and l.alpha:
                    l.alpha = [float(a) * factor for a in l.alpha]
                if hasattr(l, "omega") and l.omega:
                    if factor < 0 and any(w != 0.0 for w in l.omega):
                        raise ValueError("inertia")
                    l.omega = [float(w) * math.sqrt(factor) for w in l.omega] if factor >= 0 else l.omega
                if hasattr(l, "wmat") and l.wmat:
                    l.wmat = [[float(v) * factor for v in row] for row in l.wmat]
            elif kind == "frame_node":
                if hasattr(l, "force") and l.force:
                    l.force = [float(v) * factor for v in l.force]
                if hasattr(l, "moment") and l.moment:
                    l.moment = [float(v) * factor for v in l.moment]
            elif kind == "frame_member":
                if hasattr(l, "q") and l.q:
                    l.q = [float(v) * factor for v in l.q]
                if hasattr(l, "q_end") and l.q_end:
                    l.q_end = [float(v) * factor for v in l.q_end]
            elif kind == "frame_temperature":
                l.dT, l.grad_z, l.grad_y = (float(v) * factor for v in (l.dT, l.grad_z, l.grad_y))
            elif kind == "thermal":
                pass


# ======================================================================= studies with every kind

def _bar_case():
    """Bar (tet10, faces / edges / points): V1 (face force), V7 (remote moment), V11 (edge, points), V12 (local and
    cylindrical system), V13 (linear and hydrostatic pressure), V14 (rotation), gravity, inertia, non-solid kinds."""
    m, st = bar_mesh(), bar_study()
    st.coordinate_systems["K"] = CoordinateSystem("K", "cartesian", [5, 5, 5], [1, 0, 0], [0, 1, 0.2])
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical", [0, 5, 5], [1, 0, 0], [0, 1, 0])
    loads = [
        Force("x100", [0.0, 0.0, -100.0]),                                   # V1
        Force("x100", [3.0, -2.0, 7.0], csys="K"),
        Force("x100", [0.0, 50.0, 0.0], csys="C"),                           # V12 tangential (torque)
        Force("hrana_cela", [0.0, 0.0, -40.0]),                              # V11 edge
        Force("hrana_horni", [1.0, 2.0, -3.0], csys="C"),
        Force("rohy", [0.0, 0.0, -10.0]), Force("stred", [5.0, 0.0, 0.0], csys="K"),   # V11 points
        Pressure("z10", 0.3),
        Pressure("y0", 0.1, gradient=[0.01, 0.0, -0.002], origin=[10.0, 0.0, 3.0]),   # V13 linear
        Pressure("z10", 0.0, gradient=[0.02, 0.0, 0.0], origin=[47.3, 0.0, 0.0], clip=True),   # V13 hydrostatic
        Gravity([0.0, -9810.0, 120.0]), Gravity([1.0, 2.0, 3.0], csys="K"),
        Rotation([0.0, 5.0, 5.0], [1.0, 0.0, 0.0], omega=300.0, alpha=40.0),   # V14
        Rotation([50.0, 0.0, 0.0], [0.0, 1.0, 1.0], omega=0.0, alpha=-7.0, parts=["Tyc"]),
        Inertia([10.0, 5.0, 5.0], accel=[100.0, -20.0, 3.0], omega=[1.0, 2.0, -3.0], alpha=[0.5, 0.0, 9.0]),
        Inertia(accel=[0.0, 0.0, 0.0], wmat=[[1.0, 0.2, 0.0], [0.2, -3.0, 0.1], [0.0, 0.1, 2.0]]),
        RemoteLoad("x100", [150.0, 5.0, 5.0], [10.0, 0.0, -30.0], [0.0, 1000.0, 2500.0]),   # V7 / V10
        RemoteLoad("x100", [100.0, 5.0, 20.0], [0.0, 7.0, 0.0], [300.0, 0.0, 0.0], coupling="rigid", csys="K"),
        ThermalLoad(value=80.0), FrameNodeLoad("A", [1.0, 2.0, 3.0]), FrameMemberLoad("m", [0, 0, -1.0]),
        FrameTemperature(dT=10.0),
    ]
    return m, st, loads


def _ring_case():
    """Annulus with a hole: bearing load (V9), force and remote load on the hole."""
    m = ring_mesh(order=2)
    st = ring_study(S.Fixed("spodek"))
    st.coordinate_systems["K"] = CoordinateSystem("K", "cartesian", [0, 0, 0], [0, 0, 1], [0.6, 0.8, 0])
    loads = [BearingLoad("vnitrni", [100.0, -30.0, 12.0]), BearingLoad("vnitrni", [0.0, 50.0, 0.0], csys="K"),
             Force("vnitrni", [0.0, 0.0, 20.0]), RemoteLoad("vnitrni", [0.0, 0.0, 10.0], [5.0, 0.0, 0.0], [0, 0, 70.0])]
    return m, st, loads


@pytest.mark.parametrize("case", [_bar_case, _ring_case])
@pytest.mark.parametrize("cached", [False, True])
def test_nodal_loads_and_resultant_bit_identical(case, cached):
    m, st, loads = case()
    import contextlib
    ctx = L.nodal_cache() if cached else contextlib.nullcontext()
    with ctx:
        # all at once and each load alone (the order of the summation is part of the bit-identity)
        for lc in [LoadCase("all", loads)] + [LoadCase(f"l{i}", [l]) for i, l in enumerate(loads)]:
            new = L.nodal_loads(st, m, lc)
            old = _old_nodal_loads(st, m, lc)
            for a, b in zip(new, old):
                assert same(a, b), lc.name
            assert same(L.load_resultant(st, m, lc), _old_load_resultant(st, m, lc)), lc.name


def test_ngs_extra_forces_same_as_before():
    """Right-hand side of the own assembly: nodal loads + distributing forces of every DISTRIBUTING remote load (a rigid one acts on the extra DOFs of its rigid
    body: `linear_constraints.Plan.extra_loads`, nothing on the nodes)."""
    m, st, loads = _bar_case()
    lc = LoadCase("all", loads)
    f_old = L.nodal_loads(st, m, lc)[0]
    for l in lc.loads:
        if isinstance(l, RemoteLoad) and l.coupling != "rigid":
            f_old = f_old + L.distributing_forces(m, l.selection, l.point, *L.remote_vectors(st, l))
    f_new = L.nodal_loads(st, m, lc)[0]
    for l in lc.loads:
        h = K.load_handler(l)
        if h.ngs_nodal is not None:
            f_new = f_new + h.ngs_nodal(st, m, l)
    assert same(f_new, f_old)


# ======================================================================= scaling, checks, texts, backends

@pytest.mark.parametrize("factor", [2.5, 0.0, 1.0, -1.5])
def test_scale_study_loads_same_as_before(factor):
    from fea_core.sweep import scale_study_loads
    _, st, loads = _bar_case()
    loads = [l for l in loads if factor >= 0 or getattr(l, "kind", "") not in ("rotation", "inertia")]
    st.load_cases = [LoadCase("L", loads)]
    a, b = copy.deepcopy(st), copy.deepcopy(st)
    scale_study_loads(a, factor)
    _old_scale(b, factor)
    assert [vars(x) for x in a.load_cases[0].loads] == [vars(x) for x in b.load_cases[0].loads]


@pytest.mark.parametrize("kind", ["rotation", "inertia"])
def test_scale_negative_rotation_rejected(kind):
    from fea_core.sweep import scale_study_loads
    st = bar_study()
    l = Rotation(omega=10.0) if kind == "rotation" else Inertia(omega=[0.0, 1.0, 0.0])
    st.load_cases = [LoadCase("L", [l])]
    with pytest.raises(ValueError):
        scale_study_loads(st, -1.0)


def test_check_loads_entities_and_csys():
    from fea_core.solve import check_loads
    st = bar_study()
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical", [0, 5, 5], [1, 0, 0], [0, 1, 0])
    st.selections["mix"] = S.Selection("mix", st.selections["x0"].faces, edges=st.selections["hrana_cela"].edges)
    st.load_cases = [LoadCase("L", [
        Force("hrana_cela", [0, 0, 1.0]), Force("mix", [0, 0, 1.0]), Pressure("rohy", 1.0),
        BearingLoad("hrana_cela", [1.0, 0, 0]), Gravity(csys="C"), RemoteLoad("x100", [0, 0, 0], csys="C"),
        Force("x100", [0, 1.0, 0], csys="C"), Pressure("x100", 1.0, gradient=[1.0, 2.0]),
        Rotation(axis=[0, 0, 0]), Rotation(omega=0.0)])]
    errors, warnings = check_loads(st, ["Tyc"])
    assert errors == [
        "Case 'L', load 'mix': selection mixes faces, edges, and points – force needs a single kind",
        "Case 'L', load 'rohy': selection contains points – this load type can only be applied to faces",
        "Case 'L', load 'hrana_cela': selection contains edges – this load type can only be applied to faces",
        "Case 'L', load 'gravity': cylindrical coordinate system is only available for force (other loads use Cartesian)",
        "Case 'L', load 'x100': cylindrical coordinate system is only available for force (other loads use Cartesian)",
        "Case 'L', load 'x100': gradient and origin of the variable pressure must have 3 finite components",
        "Case 'L', rotation: axis (point, nonzero direction) and ω, α must be finite numbers"]
    assert len(warnings) == 3 and warnings[0].startswith("Case 'L', load 'hrana_cela': force on edge is singular")
    assert warnings[1:] == ["Case 'L': rotation has ω = 0 and α = 0 (no load)"] * 2


def test_texts_from_registry():
    from fea_core.loads_text import load_text, constraint_text
    _, st, loads = _bar_case()
    for l in loads:
        t = load_text(l, st)
        assert t and isinstance(t, str)
    assert load_text(ThermalLoad(value=1.0), st) == "thermal"
    assert load_text(FrameNodeLoad("A"), st) == "frame_node"
    assert load_text(Force("rohy", [0, 0, -10.0]), st).endswith(", 2 points of F/2 each")
    assert constraint_text(S.Fixed("x0")) == "fixed support (ux = uy = uz = 0)"
    assert constraint_text(S.Displacement("x0", uz=0.5)) == "prescribed displacement: uz = 0.5 mm"
    assert constraint_text(S.ElasticSupport("x0", 2.0)) == "elastic support k = 2 N/mm³, normal to face"
    assert constraint_text(type("C", (), {"kind": "odd"})()) == "odd"


def test_unsupported_messages():
    """What the own assembly (direct solver) cannot do: the constraints of F13 step 1b (cylindrical, remote, elastic support, rigid remote load) are no longer
    in the list; the iterative solver cannot do what couples several nodes."""
    from fea_core.solve_large import unsupported
    st = bar_study()
    st.load_cases = [LoadCase("A", [RemoteLoad("x100", [0, 0, 0], coupling="rigid"), RemoteLoad("x100", [0, 0, 0])])]
    assert unsupported(st) == []
    assert unsupported(st, "hypre") == ["rigid remote load (A) – direct solver only"]
    st.constraints += [S.ElasticSupport("y0", 1.0), S.CylindricalSupport("z10"),
                       S.RemoteSupport("x100", [0, 0, 0])]
    assert unsupported(st) == []
    assert unsupported(st, "hypre") == ["remote support (rigid) – direct solver only", "rigid remote load (A) – direct solver only"]


def test_write_inp_grav_and_nodal_paths(tmp_path):
    """ccx hooks: gravity *DLOAD GRAV without U1 elements; every other solid kind = *CLOAD; SURFACE for the constant
    pressure and the remote load (as before)."""
    from fea_core.ccx.inp import write_inp
    m, st, _ = _bar_case()
    st.load_cases = [LoadCase("L", [Gravity([0.0, 0.0, -9810.0]), Pressure("z10", 0.2), Force("hrana_cela", [0, 0, 1.0]),
                                    Rotation(omega=10.0, axis=[1.0, 0, 0])])]
    write_inp(st, m, tmp_path / "a.inp")
    txt = (tmp_path / "a.inp").read_text()
    assert "*DLOAD\nEALL, GRAV, 9810, 0, 0, -1\n" in txt
    assert "*CLOAD\n" in txt
    sid = list(st.selections).index("z10") + 1
    assert f"*SURFACE, NAME=SSEL{sid}, TYPE=ELEMENT" in txt           # constant pressure keeps its SURFACE
    st.load_cases[0].loads[1] = Pressure("z10", 0.2, gradient=[0.0, 0.0, 0.01])
    write_inp(st, m, tmp_path / "b.inp")
    assert f"*SURFACE, NAME=SSEL{sid}," not in (tmp_path / "b.inp").read_text()

"""Guard against DOFs that nothing stiffens in the decks written for CalculiX (the bug class of the auxiliary nodes).

Found twice in 1.4.2: the auxiliary node of a normal elastic support and the thermal-gap node of a bolt carry a SPRING1 and an *EQUATION in DOF 1 only; ccx still makes DOF 2 and
3 unknowns, with empty rows. PaStiX let them pass, the SPOOLES of the own CalculiX (the only solver of the shipped package) ended with "matrix found to be singular" or the
return code 0xFFFFFFFF. The third node of the class was found by the release check of 1.4.2: the auxiliary node of a ROTATIONAL spring support of a beam (`FrameFE.aux_dof1`),
which only the legacy ccx U1 route writes – no verification case ran it, so nothing noticed.

`tests/deck_dofs.py` parses a deck and lists the nodes whose DOFs have no stiffness and are not held in the block of some step. This file writes a representative deck of every
feature that creates helper nodes or light elements (SPRING1 / SPRING2 / SPRINGA / MASS / GAPUNI, *EQUATION, *RIGID BODY, *COUPLING) on synthetic meshes (no ccx run) and
requires an empty list. A NEW feature that writes helper nodes belongs here: add its deck to `DECKS`.
"""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.ccx.inp import write_inp
from tests.deck_dofs import analyze
from tests.test_remote import _study
from tests.test_thermal import bar_mesh


# ----------------------------------------------------------------------------------------- the analyzer itself
GOOD = """*NODE, NSET=NALL
1, 0, 0, 0
2, 1, 0, 0
3, 0, 1, 0
4, 0, 0, 1
*ELEMENT, TYPE=C3D4, ELSET=EALL
1, 1, 2, 3, 4
*NODE
9, 0, 0, 0
*ELEMENT, TYPE=SPRING1, ELSET=SP
2, 9
*SPRING, ELSET=SP
1
5.
*EQUATION
2
9, 1, 1., 2, 1, -1.
*STEP
*STATIC
*BOUNDARY
1, 1, 3
9, 2, 3, 0.
*END STEP
"""


def test_the_analyzer_accepts_a_node_whose_other_dofs_are_fixed():
    assert analyze(GOOD) == []


def test_the_analyzer_reports_the_empty_rows_of_an_auxiliary_spring_node():
    bad = GOOD.replace("9, 2, 3, 0.\n", "")
    assert analyze(bad) == ["step 1: node 9: DOF 2, 3 have no stiffness and are not held"]


def test_the_analyzer_follows_op_new_of_a_later_step():
    """`*BOUNDARY, OP=NEW` of the second step forgets the first step's conditions: the node must be fixed again there."""
    two = GOOD.replace("*END STEP\n", "*END STEP\n*STEP\n*STATIC\n*BOUNDARY, OP=NEW\n1, 1, 3\n*END STEP\n")
    assert analyze(two) == ["step 2: node 9: DOF 2, 3 have no stiffness and are not held"]
    kept = GOOD.replace("*END STEP\n", "*END STEP\n*STEP\n*STATIC\n*BOUNDARY\n1, 1, 3\n*END STEP\n")      # no OP=NEW: the earlier ones stay
    assert analyze(kept) == []


def test_the_analyzer_resolves_node_sets_and_a_mass_only_node():
    deck = GOOD.replace("9, 2, 3, 0.\n", "NAUX, 2, 3, 0.\n").replace("*NODE\n9, 0, 0, 0\n", "*NODE, NSET=NAUX\n9, 0, 0, 0\n")
    assert analyze(deck) == []
    mass = GOOD.replace("*EQUATION\n2\n9, 1, 1., 2, 1, -1.\n", "*NODE\n8, 1, 1, 1\n*ELEMENT, TYPE=MASS, ELSET=M\n3, 8\n*MASS, ELSET=M\n1.\n")
    assert any("node 8" in p for p in analyze(mass))                           # a point mass nothing holds


# ----------------------------------------------------------------------------------------- decks of the features
def _bar_deck(tmp_path, constraints, loads=None, **kw) -> str:
    m = bar_mesh()
    st = _study(constraints)
    if loads is not None:
        st.load_cases = loads
    if kw.pop("masses", None) is not None:
        pass
    write_inp(st, m, tmp_path / "d.inp", **kw)
    return (tmp_path / "d.inp").read_text()


def _fixed():
    from fea_core.study import Displacement
    return Displacement("x0", ux=0.0, uy=0.0, uz=0.0)


def _two_states():
    from fea_core.study import Force, LoadCase
    return [LoadCase("A", [Force("y0", [0, 0, -10.0])]), LoadCase("B", [Force("y0", [0, 0, 10.0])])]


def deck_plain(tmp_path):
    return _bar_deck(tmp_path, [_fixed()], _two_states())


def deck_elastic_normal(tmp_path):
    from fea_core.study import ElasticSupport
    return _bar_deck(tmp_path, [_fixed(), ElasticSupport("x100", 5.0, "normal")], _two_states())


def deck_elastic_normal_modal(tmp_path):
    from fea_core.study import ElasticSupport
    return _bar_deck(tmp_path, [_fixed(), ElasticSupport("x100", 5.0, "normal")], procedure="modal", eigen_line="4")


def deck_elastic_all(tmp_path):
    from fea_core.study import ElasticSupport
    return _bar_deck(tmp_path, [_fixed(), ElasticSupport("x100", 5.0, "all")], _two_states())


def deck_remote_support_rigid(tmp_path):
    from fea_core.study import RemoteSupport
    return _bar_deck(tmp_path, [RemoteSupport("x0", [-50.0, 5.0, 5.0], [True, True, True, False, True, False], "rigid")], _two_states())


def deck_remote_support_distributing(tmp_path):
    from fea_core.study import RemoteSupport
    return _bar_deck(tmp_path, [RemoteSupport("x0", [-50.0, 5.0, 5.0], [True] * 6, "distributing")], _two_states())


def deck_remote_load_rigid(tmp_path):
    from fea_core.study import LoadCase, RemoteLoad
    loads = [LoadCase("A", [RemoteLoad("x100", [150.0, 5.0, 5.0], [0, 0, -100.0], [1000.0, 0, 0], "rigid")]),
             LoadCase("B", [RemoteLoad("x100", [150.0, 25.0, 5.0], [0, 0, -100.0], [0, 0, 0], "rigid")])]
    return _bar_deck(tmp_path, [_fixed()], loads)


def deck_remote_load_distributing(tmp_path):
    from fea_core.study import LoadCase, RemoteLoad
    loads = [LoadCase("A", [RemoteLoad("x100", [150.0, 5.0, 5.0], [0, 0, -100.0], [1000.0, 0, 0], "distributing")])]
    return _bar_deck(tmp_path, [_fixed()], loads)


def _with_masses(tmp_path, masses, procedure="static", **kw) -> str:
    m = bar_mesh()
    st = _study([_fixed()])
    st.masses = list(masses)
    write_inp(st, m, tmp_path / "d.inp", procedure=procedure, **kw)
    return (tmp_path / "d.inp").read_text()


def deck_point_mass_rigid_with_inertia(tmp_path):
    from fea_core.dynamics.model import PointMass
    return _with_masses(tmp_path, [PointMass("x100", 1e-4, point=[110.0, 5.0, 5.0], inertia=[1e-3, 2e-3, 3e-3], coupling="rigid")])


def deck_point_mass_rigid_without_inertia(tmp_path):
    from fea_core.dynamics.model import PointMass
    return _with_masses(tmp_path, [PointMass("x100", 1e-4, point=[110.0, 5.0, 5.0], coupling="rigid")], "modal", eigen_line="4")


def deck_point_mass_distributing(tmp_path):
    from fea_core.dynamics.model import PointMass
    return _with_masses(tmp_path, [PointMass("x100", 1e-4, point=[110.0, 5.0, 5.0], coupling="distributing")], "modal", eigen_line="4")


def deck_surface_mass(tmp_path):
    from fea_core.dynamics.model import SurfaceMass
    return _with_masses(tmp_path, [SurfaceMass("y0", 5.0)], "modal", eigen_line="4")


def deck_cylindrical_support(tmp_path):
    from fea_core.study import CylindricalSupport, Displacement
    from tests.test_cylindrical import _study as cyl_study, ring_mesh
    m = ring_mesh(order=2)
    st = cyl_study(CylindricalSupport("vnitrni", True, True, False), Displacement("spodek", uz=0.0))
    write_inp(st, m, tmp_path / "c.inp")
    return (tmp_path / "c.inp").read_text()


def _frame_deck(tmp_path, kind):
    import verification.cases  # noqa: F401  (first the cases: cyclic import of the frame cases)
    from fea_core.beams import fem
    from fea_core.beams.run import write_u1_modal_inp, write_u1_static_inp
    from verification import frame_cases as fc
    st, _ = fc.f11_build(tmp_path, spring=True)                  # hinges, truss members, released torsion, a ROTATIONAL spring support
    st.frame.settings.solver = "ccx_u1"
    fe = fem.build(st)
    st.modal.enabled = True
    (write_u1_static_inp if kind == "static" else write_u1_modal_inp)(st, fe, tmp_path / "f.inp")
    return (tmp_path / "f.inp").read_text()


def deck_frame_u1_static(tmp_path):
    return _frame_deck(tmp_path, "static")


def deck_frame_u1_modal(tmp_path):
    return _frame_deck(tmp_path, "modal")


def deck_frame_with_a_body(tmp_path):
    """A mixed model: a beam rigidly connected to a face of the body (`*COUPLING/*KINEMATIC` + the equations), a rotational spring at its far end."""
    from fea_core.beams.model import FrameConnection, FrameMember, FrameNodeLoad, FrameSupport, SectionDef
    from fea_core.study import LoadCase
    m = bar_mesh()
    st = _study([_fixed()])
    fr = st.frame
    fr.sections["R"] = SectionDef("R", shape="RECT", dims={"b": 10.0, "h": 10.0})
    fr.add_node("J", [100.0, 5.0, 5.0])
    fr.add_node("K", [200.0, 5.0, 5.0])
    fr.members["prut"] = FrameMember("prut", "J", "K", "R")
    fr.connections.append(FrameConnection("J", "x100"))
    fr.supports.append(FrameSupport("K", [True, True, True, False, False, False], [0, 0, 0, 0, 5e9, 0]))
    st.load_cases = [LoadCase("a", [FrameNodeLoad("K", [0, 0, -100.0])]), LoadCase("b", [FrameNodeLoad("K", [0, 100.0, 0])])]
    write_inp(st, m, tmp_path / "mixed.inp")
    return (tmp_path / "mixed.inp").read_text()


def deck_thermal(tmp_path):
    from fea_core.thermal.inp import write_case, write_model
    from fea_core.thermal.model import (Convection, FixedTemperature, HeatFlux, HeatGeneration, Radiation, ThermalCase)
    from tests.test_thermal import study
    m = bar_mesh(parts=2)
    st = study(parts=2)
    st.thermal.cases = [ThermalCase("A", [FixedTemperature("x0", 100.0), Convection("x100", 500.0, 20.0), HeatFlux("y0", 1000.0),
                                          Radiation("x100", 0.5, 20.0), HeatGeneration("Dil2", 1.0)]),
                        ThermalCase("B", [FixedTemperature("x0", 100.0)], transient=True, time=95.0, dt=10.0, outputs=4)]
    mp = write_model(st, m, tmp_path / "model.inc")
    text = (tmp_path / "model.inc").read_text()
    for k, case in enumerate(st.thermal.cases):
        write_case(st, m, mp, case, tmp_path / f"case{k}.inp", f"case{k}")
        text += (tmp_path / f"case{k}.inp").read_text()
    return text


def deck_gapuni_support_in_compression_only(tmp_path):
    """A support that carries compression only: GAPUNI from every node of the face to a fixed node (the fixed nodes are in *BOUNDARY of every step)."""
    from fea_core.contacts.model import AnalysisSettings, ContactPair
    from fea_core.study import Force, LoadCase
    from fea_core.contacts.run import step_plans
    m = bar_mesh()
    m.report = {"contacts": [{"pair": "podpora", "slave": "x0", "master": None}]}       # what the contact detection leaves in the mesh report
    st = _study([])
    st.contacts = [ContactPair("podpora", "x0", "", kind="compression_only")]
    st.analysis = AnalysisSettings(kind="nonlinear")
    st.load_cases = [LoadCase("F", [Force("x100", [-100.0, 0, 0])])]
    write_inp(st, m, tmp_path / "g.inp", nonlinear={"steps": step_plans(st, preload=False)[0][1], "part_nodes": {}})
    return (tmp_path / "g.inp").read_text()


def _cyclic_deck(tmp_path, procedure, with_axis):
    from fea_core.study import Fixed, Force, LoadCase, Material
    from tests.test_cyclic import sector_mesh, study_with
    from tests.test_cylindrical import _to_tet10
    m = sector_mesh(with_axis=with_axis)
    m10 = _to_tet10(m)
    m10.surfaces = m.surfaces
    st = study_with(m10)
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9), default=True)
    for name in m.surfaces:
        st.selections[name] = None
    st.constraints = [] if with_axis else [Fixed("bore")]
    st.load_cases = [LoadCase("F", [Force("L", [0, 0, -10.0])])]
    write_inp(st, m10, tmp_path / "c.inp", procedure=procedure, eigen_line="4" if procedure == "modal" else None)
    return (tmp_path / "c.inp").read_text()


def deck_cyclic_static(tmp_path):
    return _cyclic_deck(tmp_path, "static", True)


def deck_cyclic_modal(tmp_path):
    return _cyclic_deck(tmp_path, "modal", False)


def deck_voxel(tmp_path):
    from fea_core.printing import voxel as V
    from tests.test_voxel import PrintSpec, _gen
    g = _gen(PrintSpec("A", walls=1, top_layers=0, bottom_layers=0, infill=1.0), np.array([1.0, 1.0, 0.4]))
    m = V.build_model(g)
    bins = V._angle_bins(g.angle[tuple(m["ijk"].T)], 36)
    V.write_inp(tmp_path / "v.inp", m["nodes"], m["hexes"], bins, V.bin_axes(np.eye(3), 36), np.eye(6) * 1000, np.arange(4),
                [np.zeros((4, 3)), np.ones((4, 3))], step_names=["a", "b"])
    return (tmp_path / "v.inp").read_text()


def deck_shell(tmp_path):
    from fea_core.shell.inp import write_shell_inp
    from fea_core.study import Material
    from tests.test_shell import rect_mesh
    m = rect_mesh(20, 10, 2, 1)
    T = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float)
    F = np.zeros((len(m.nodes), 3))
    F[0] = [1.0, 2.0, 3.0]
    write_shell_inp(tmp_path / "s.inp", m, {0: Material("Ocel", 210000, 0.3, 7.85e-9)}, {0: {1: 0.0, 5: 0.0}}, [("S", F)], "test", {0: T})
    return (tmp_path / "s.inp").read_text()


DECKS = [deck_plain, deck_elastic_normal, deck_elastic_normal_modal, deck_elastic_all, deck_remote_support_rigid, deck_remote_support_distributing,
         deck_remote_load_rigid, deck_remote_load_distributing, deck_point_mass_rigid_with_inertia, deck_point_mass_rigid_without_inertia,
         deck_point_mass_distributing, deck_surface_mass, deck_cylindrical_support, deck_frame_u1_static, deck_frame_u1_modal, deck_frame_with_a_body,
         deck_thermal, deck_gapuni_support_in_compression_only, deck_cyclic_static, deck_cyclic_modal, deck_voxel, deck_shell]


@pytest.mark.parametrize("builder", DECKS, ids=[b.__name__ for b in DECKS])
def test_no_deck_has_an_unconstrained_dof_without_stiffness(builder, tmp_path):
    text = builder(tmp_path)
    assert text.count("*NODE") >= 1
    assert analyze(text) == []


def test_the_decks_really_contain_the_helper_nodes_they_are_meant_to_check(tmp_path):
    """A guard that never meets its subject proves nothing: the decks above must contain the light elements / equations / rigid bodies."""
    expect = {deck_elastic_normal: ("SPRING1", "*EQUATION"), deck_remote_support_rigid: ("*RIGID BODY",), deck_remote_support_distributing: ("*EQUATION",),
              deck_point_mass_rigid_with_inertia: ("TYPE=MASS", "TYPE=SPRINGA", "*RIGID BODY"), deck_point_mass_distributing: ("TYPE=MASS", "*EQUATION"),
              deck_frame_u1_static: ("TYPE=U1", "SPRING1", "*EQUATION"), deck_frame_with_a_body: ("*COUPLING", "TYPE=U1", "SPRING1"),
              deck_gapuni_support_in_compression_only: ("TYPE=GAPUNI",), deck_cylindrical_support: ("*EQUATION",),
              deck_cyclic_static: ("*EQUATION",), deck_cyclic_modal: ("CYCLIC SYMMETRY",)}
    for builder, words in expect.items():
        text = builder(tmp_path)
        for w in words:
            assert w in text, (builder.__name__, w)


def test_the_rotational_spring_of_a_beam_support_fixes_its_auxiliary_node_in_every_step(tmp_path):
    """The case behind the guard: F11 through ccx U1 with a rotational spring (k θz at the node E) stopped SPOOLES with the code 0xFFFFFFFF."""
    import re
    text = deck_frame_u1_static(tmp_path)
    steps = text.split("*STEP")[1:]
    assert len(steps) == 2
    spring = re.search(r"TYPE=SPRING1, ELSET=FSP\d+\n(\d+)\n", text) or re.search(r"TYPE=SPRING1, ELSET=FSP\d+\n\d+, (\d+)\n", text)
    assert spring
    for s in steps:
        assert f"{spring.group(1)}, 2, 3, 0." in s.split("*END STEP")[0]
    assert "*BOUNDARY\n" in steps[0] and "*BOUNDARY, OP=NEW\n" in steps[1]            # the second step starts again (OP=NEW forgets the first)


def test_the_deck_of_bolts_with_a_temperature_state_has_no_empty_rows(tmp_path):
    """The bolts of the ccx route (U1 beams, the cut with a gap DOF, the node of the thermal gap with a SPRING1 and an equation, the *COUPLING nodes) on a real Gmsh mesh -
    the fixture of tests/test_bolts_own_e2e.py (a lap joint M10 with the preload, shear, weight and a temperature change)."""
    from fea_core.solve import prepare_mesh
    from fea_core.thermal.field import node_temperatures
    from tests.test_bolts_own_e2e import _study_lap_thermal
    st, parts = _study_lap_thermal(tmp_path)
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    st.solver.backend = "ccx"
    temps = [node_temperatures(st, mesh, lc, {}) for lc in st.load_cases]
    write_inp(st, mesh, tmp_path / "b.inp", temperatures=temps)
    text = (tmp_path / "b.inp").read_text()
    assert "TYPE=U1" in text and "TYPE=SPRING1" in text and "*COUPLING" in text
    assert analyze(text) == []

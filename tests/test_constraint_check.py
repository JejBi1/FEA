"""What the own assembly would refuse, found BEFORE the run (F13 step 1b, review): `linear_constraints.check` / `prepare`, the choice of the backend with the reason, the
error of a manual choice that names the constraints. Real Gmsh meshes of small blocks, no solver run (a few seconds).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import linear_constraints as LC
from fea_core.assembly.reduction import ReductionError
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import choose_backend_reason, prepare_mesh
from fea_core.study import ElasticSupport, Fixed, Force, LoadCase, MeshSettings, Pressure, RemoteSupport
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="the direct solver (modules accel, direct) is needed for the choice of the backend")


def _block(tmp_path, constraints, parts_spec=None, separate=(), size=6.0, loads=None):
    """A study of box parts, the selections `x0` `x1` `z0` `z1` of the first part, the mesh → (study, mesh)."""
    spec = parts_spec or [("Blok", (0, 0, 0, 60, 24, 16))]
    parts = [Part(n, box(*d)) for n, d in spec]
    st, parts = _study("check", tmp_path / "m.step", parts, MeshSettings(size=size))
    st.mesh.separate = list(separate)
    p0 = spec[0][0]
    for name, kw in (("x0", dict(part=p0, x=0)), ("x1", dict(part=p0, x=spec[0][1][3])), ("z0", dict(part=p0, z=0)), ("z1", dict(part=p0, z=spec[0][1][5]))):
        st.add_selection(name, find_faces(parts, **kw))
    for c in constraints:
        st.constraints.append(c)
    st.load_cases += loads or [LoadCase("tlak", [Pressure("z1", 1.0)])]
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    return st, mesh, parts


def test_a_rigid_face_that_shares_an_edge_with_a_fixed_face_is_refused_with_both_names(tmp_path):
    st, mesh, _ = _block(tmp_path, [RemoteSupport("z0", [30.0, 12.0, -10.0], [True] * 6, "rigid"), Fixed("x0")])
    why = LC.check(st, mesh)
    assert len(why) == 1 and "rigid remote support: 'z0'" in why[0] and "fixed support: 'x0'" in why[0] and "dependent" in why[0]
    with pytest.raises(ReductionError) as ei:
        LC.prepare(st, mesh)
    assert "rigid remote support: 'z0'" in str(ei.value) and "fixed support: 'x0'" in str(ei.value)


def test_two_remote_supports_that_share_nodes_and_a_dependent_tie_master_are_refused_with_the_selections(tmp_path):
    st, mesh, _ = _block(tmp_path, [RemoteSupport("x0", [-10.0, 12.0, 8.0], [True] * 6, "distributing"),
                                    RemoteSupport("z0", [30.0, 12.0, -10.0], [False, False, True, False, False, False], "distributing")])
    why = LC.check(st, mesh)
    assert len(why) == 1 and "'x0'" in why[0] and "'z0'" in why[0]
    # a body tied to a face that is a rigid support: the master nodes of the tie would be dependent
    spec = [("A", (0, 0, 0, 60, 24, 16)), ("B", (60, 0, 0, 40, 24, 16))]
    parts = [Part(n, box(*d)) for n, d in spec]
    st, parts = _study("tie", tmp_path / "t.step", parts, MeshSettings(size=6.0))
    st.mesh.separate = ["B"]
    st.add_selection("a1", find_faces(parts, part="A", x=60))
    st.add_selection("b1", find_faces(parts, part="B", x=100))
    st.constraints.append(RemoteSupport("a1", [40.0, 12.0, 8.0], [True] * 6, "rigid"))
    st.load_cases.append(LoadCase("sila", [Force("b1", [100.0, 50.0, -200.0])]))
    mesh, _ = prepare_mesh(st, tmp_path / "tie", tmp_path / "tie", lambda s: None, parts=parts)
    why = LC.check(st, mesh)
    assert len(why) == 1 and "dependent" in why[0] and "rigid remote support: 'a1'" in why[0] and "TIE master face: 'B'" in why[0]


def test_what_the_assembly_takes_is_not_refused(tmp_path):
    st, mesh, _ = _block(tmp_path, [RemoteSupport("x0", [-10.0, 12.0, 8.0], [True] * 6, "rigid"), ElasticSupport("z0", 5.0, "normal")])
    assert LC.check(st, mesh) == [] and LC.prepare(st, mesh) is not None
    # springs, fixed supports and prescribed displacements are not even looked at
    (tmp_path / "s").mkdir()
    st2, mesh2, _ = _block(tmp_path / "s", [Fixed("x0"), ElasticSupport("z0", 5.0, "normal")])
    assert LC.check(st2, mesh2) == []


def test_a_distributing_support_above_the_limit_is_refused_with_its_name_and_the_limit_follows_the_free_memory(tmp_path, monkeypatch):
    st, mesh, _ = _block(tmp_path, [RemoteSupport("x0", [-10.0, 12.0, 8.0], [True] * 6, "distributing")], size=3.0)
    plan = LC.build(st, mesh)
    fill = float(sum(float(t) ** 2 for t in np.diff(plan.red.ptr)))
    assert fill > 1e5
    monkeypatch.setattr(LC, "MAX_FILL_OWN", fill / 2)
    why = LC.check(st, mesh)
    assert len(why) == 1 and "'x0'" in why[0] and "dense block" in why[0] and f"{fill:.2g}" in why[0]
    monkeypatch.setattr(LC, "MAX_FILL_OWN", fill * 2)
    assert LC.check(st, mesh) == []
    # the share of the free memory (measured: ≈ 4e-8 GB of the assembly peak per unit): 1 GB free → 1.25e7, plenty → the cap
    monkeypatch.setattr(LC, "MAX_FILL_OWN", 1.0e8)
    assert LC.fill_limit(1.0) == pytest.approx(0.5 / 4.0e-8) and LC.fill_limit(1000.0) == 1.0e8
    monkeypatch.setattr(LC, "fill_limit", lambda free_gb=None: fill / 2)
    assert LC.check(st, mesh)


@needs_direct
def test_auto_goes_to_ccx_with_the_reason_and_a_manual_direct_solver_names_the_constraints(tmp_path):
    from fea_core.solve_large import LargeSolverError, run_ngs
    st, mesh, _ = _block(tmp_path, [RemoteSupport("z0", [30.0, 12.0, -10.0], [True] * 6, "rigid"), Fixed("x0")])
    st.solver.backend = "auto"
    backend, reason = choose_backend_reason(st, mesh)
    assert backend == "ccx" and reason.startswith("CalculiX only: ") and "rigid remote support: 'z0'" in reason and "fixed support: 'x0'" in reason
    st.solver.backend = "ngs_direct"
    with pytest.raises(LargeSolverError) as ei:
        run_ngs(st, mesh, tmp_path / "own", [], lambda s: None, solver="direct")
    msg = str(ei.value)
    assert msg.startswith("The constraints cannot be eliminated") and "rigid remote support: 'z0'" in msg and "fixed support: 'x0'" in msg
    # the same study without the shared edge: the direct solver
    st.constraints.pop()
    st.solver.backend = "auto"
    assert choose_backend_reason(st, mesh)[0] == "ngs_direct"


def test_a_constraint_that_cannot_be_built_does_not_break_the_choice_of_the_solver():
    """The pre-check is an addition to the choice: a mesh that the constraints cannot be built on (here a cylindrical support on a mesh without that face) is a
    reason for ccx – which gives its own message – and not an exception of `choose_backend_reason`."""
    from fea_core.study import CylindricalSupport, Study

    class Bare:
        nodes = np.zeros((4, 3))
        surfaces: dict = {}
        report: dict = {}
    st = Study("bare", "m.step")
    st.constraints.append(CylindricalSupport("nikde"))
    why = LC.check(st, Bare())
    assert len(why) == 1 and why[0]

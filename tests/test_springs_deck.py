"""The ccx deck of an elastic support: the auxiliary node of a normal spring carries a spring and an equation in DOF 1 only; its DOFs 2 and 3 must be fixed in
every step. Otherwise ccx makes them unknowns without any stiffness: PaStiX lets the zero rows pass, SPOOLES (the only solver of the Linux package, and the solver of
every natural-frequency run) ends with "matrix found to be singular" and return code 255 (found on Linux by the case V15 through ccx; without ccx here)."""
import re

from fea_core.ccx.inp import write_inp
from fea_core.study import Displacement, ElasticSupport, Force, LoadCase
from tests.test_remote import _study
from tests.test_thermal import bar_mesh


def _boundary_blocks(txt: str) -> list[str]:
    return re.findall(r"\*BOUNDARY[^\n]*\n((?:[^*\n][^\n]*\n)*)", txt)


def test_aux_nodes_of_normal_springs_have_dof_2_and_3_fixed_in_every_step(tmp_path):
    m = bar_mesh()
    st = _study([Displacement("x0", ux=0.0, uy=0.0, uz=0.0), ElasticSupport("x100", 5.0, "normal")])
    st.load_cases = [LoadCase("A", [Force("y0", [0, 0, -10.0])]), LoadCase("B", [Force("y0", [0, 0, 10.0])])]
    write_inp(st, m, tmp_path / "n.inp")
    txt = (tmp_path / "n.inp").read_text()
    assert "*NSET, NSET=NSPR" in txt and "*EQUATION" in txt
    blocks = _boundary_blocks(txt)
    assert len(blocks) == 2 and all("NSPR, 2, 3, 0.\n" in b for b in blocks)        # one *BOUNDARY per load state, none without the fixed DOFs


def test_the_fixed_dofs_are_also_in_a_natural_frequency_run(tmp_path):
    m = bar_mesh()
    st = _study([Displacement("x0", ux=0.0, uy=0.0, uz=0.0), ElasticSupport("x100", 5.0, "normal")])
    write_inp(st, m, tmp_path / "f.inp", procedure="modal", eigen_line="4")
    blocks = _boundary_blocks((tmp_path / "f.inp").read_text())
    assert blocks and all("NSPR, 2, 3, 0.\n" in b for b in blocks)


def test_springs_in_all_directions_have_no_auxiliary_node(tmp_path):
    m = bar_mesh()
    st = _study([Displacement("x0", ux=0.0, uy=0.0, uz=0.0), ElasticSupport("x100", 5.0, "all")])
    write_inp(st, m, tmp_path / "a.inp")
    txt = (tmp_path / "a.inp").read_text()
    assert "NSPR" not in txt and "SPRING1" in txt


def test_the_thermal_gap_node_of_a_bolt_has_dof_2_and_3_fixed_in_every_step(tmp_path):
    """The same case on a bolt: the node of its thermal gap carries a SPRING1 and an equation in DOF 1 only. With the own SPOOLES build of Windows (and on Linux)
    the model of a bolted joint with a temperature state ended with the return code 0xFFFFFFFF right after "Using up to 1 cpu(s) for spooles" (found in the
    merge of the bolts branch with the one that dropped PaStiX; the real run is the fixture of tests/test_bolts_own_e2e.py)."""
    from fea_core.solve import prepare_mesh
    from tests.test_bolts_own_e2e import _study_lap_thermal
    st, parts = _study_lap_thermal(tmp_path)
    mesh, _ = prepare_mesh(st, tmp_path, tmp_path, lambda s: None, parts=parts)
    st.solver.backend = "ccx"
    from fea_core.thermal.field import node_temperatures
    temps = [node_temperatures(st, mesh, lc, {}) for lc in st.load_cases]
    write_inp(st, mesh, tmp_path / "b.inp", temperatures=temps)
    txt = (tmp_path / "b.inp").read_text()
    tgap = re.search(r"\*ELEMENT, TYPE=SPRING1, ELSET=BT1\n\d+, (\d+)\n", txt)
    assert tgap, "a bolt with a temperature state has a thermal gap spring"
    blocks = _boundary_blocks(txt)
    assert len(blocks) == len(st.load_cases) + 1 and all(f"{tgap.group(1)}, 2, 3, 0.\n" in b for b in blocks)      # every state and the preload state

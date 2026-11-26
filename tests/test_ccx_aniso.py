"""ccx material cards of printed parts: a group rotated about the build axis is monoclinic and needs `*ELASTIC, TYPE=ANISO`.

`TYPE=ORTHO` carries 9 constants; ccx drops C14 C24 C34 C56 of a rotated perimeter / ±45° core (up to 19 MPa against C66 156–180 MPa in
three quarters of the groups of a plate with a hole) – dU 2.5e-4, dS 7e-4 against the own assembly, which uses the full C. The card is chosen per group
(`inp.elastic_card`); groups without a coupling keep the byte-identical ORTHO card. Fast (a tiny ccx run for the order of the 21 constants).
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fea_core import config, materials as matlib
from fea_core.ccx.inp import ANISO_ORDER, ORTHO_ORDER, elastic_card, is_orthotropic, write_inp
from fea_core.printing import PrintSpec, line_axes, material_groups
from fea_core.printing.infill import rotation, strain_matrix

PLA = matlib.by_name("Tisk PLA Bambu Basic")


def ortho_c() -> np.ndarray:
    """An orthotropic C (6×6, Voigt xx yy zz xy xz yz, engineering shear) with all constants different."""
    E1, E2, E3, n12, n13, n23 = 3000.0, 2200.0, 1800.0, 0.3, 0.25, 0.28
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -n12 / E1
    S[0, 2] = S[2, 0] = -n13 / E1
    S[1, 2] = S[2, 1] = -n23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / 900.0, 1 / 700.0, 1 / 600.0
    return np.linalg.inv(S)


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def numbers(card: str) -> list[float]:
    return [float(x) for ln in card.splitlines()[1:] for x in ln.split(",") if x.strip()]


# --------------------------------------------------------------------------------------------- the card
def test_orthotropic_card_is_the_legacy_text():
    C = ortho_c()
    d = [C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2], C[3, 3], C[4, 4], C[5, 5]]
    legacy = "*ELASTIC, TYPE=ORTHO\n" + ", ".join(f"{v:.12g}" for v in d[:8]) + f",\n{d[8]:.12g}, 0.\n"
    assert is_orthotropic(C) and elastic_card(C) == legacy
    # rounding noise of a rotation by 0° / 90° (≈ 1e-16 of max|C|) is not a coupling – the card stays ORTHO
    T = strain_matrix(line_axes(rotation([0, 0, 1]), np.pi / 2))
    Cn = T.T @ C @ T
    assert is_orthotropic(Cn) and elastic_card(Cn).startswith("*ELASTIC, TYPE=ORTHO")


def test_monoclinic_card_is_aniso_with_21_constants_in_the_manual_order():
    C = ortho_c()
    T = strain_matrix(rot([0, 0, 1], np.radians(33.0)))
    Cg = T.T @ C @ T
    assert not is_orthotropic(Cg)
    card = elastic_card(Cg)
    assert card.startswith("*ELASTIC, TYPE=ANISO\n")
    lines = card.splitlines()[1:]
    assert [len([x for x in ln.split(",") if x.strip()]) for ln in lines] == [8, 8, 6]       # 8 + 8 + 5 constants + the temperature
    v = numbers(card)
    assert len(v) == 22 and v[-1] == 0.0
    assert [(i, j) for i, j in ANISO_ORDER][:9] == [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)]
    back = np.zeros((6, 6))
    for (i, j), x in zip(ANISO_ORDER, v[:21]):
        back[i, j] = back[j, i] = x
    assert np.allclose(back, Cg, rtol=1e-11, atol=1e-8)
    # the ORTHO subset of the order is the 9 constants of the old card
    assert len(ORTHO_ORDER) == 9 and set(ORTHO_ORDER) <= set(ANISO_ORDER)


# --------------------------------------------------------------------------------------------- ccx reads it right
def hex_block(material: str, orient: str, section_orient: str) -> str:
    nx, ny, nz = 3, 2, 2
    xs, ys, zs = np.linspace(0, 30, nx + 1), np.linspace(0, 20, ny + 1), np.linspace(0, 20, nz + 1)

    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)
    s = "*NODE, NSET=NALL\n" + "".join(f"{nid(i, j, k)}, {xs[i]:.12g}, {ys[j]:.12g}, {zs[k]:.12g}\n"
                                      for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1))
    els = [[nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k), nid(i, j, k + 1), nid(i + 1, j, k + 1),
            nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)] for k in range(nz) for j in range(ny) for i in range(nx)]
    s += "*ELEMENT, TYPE=C3D8, ELSET=EALL\n" + "".join(f"{n + 1}, " + ", ".join(map(str, e)) + "\n" for n, e in enumerate(els))
    left = [nid(0, j, k) for k in range(nz + 1) for j in range(ny + 1)]
    right = [nid(nx, j, k) for k in range(nz + 1) for j in range(ny + 1)]
    s += "*NSET, NSET=LEFT\n" + ", ".join(map(str, left)) + "\n"
    s += orient + "*MATERIAL, NAME=M1\n" + material + "*SOLID SECTION, ELSET=EALL, MATERIAL=M1" + section_orient + "\n"
    s += "*BOUNDARY\nLEFT, 1, 3\n*STEP\n*STATIC\n*CLOAD\n"
    rng = np.random.default_rng(3)
    for n in right:
        s += "".join(f"{n}, {c}, {rng.uniform(-300, 600):.6g}\n" for c in (1, 2, 3))
    return s + "*NODE PRINT, NSET=NALL\nU\n*END STEP\n"


def run_ccx(text: str, wd: Path) -> np.ndarray:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "j.inp").write_text(text)
    subprocess.run([str(config.find_ccx()), "j"], cwd=wd, capture_output=True, text=True, check=False)
    dat = (wd / "j.dat").read_text()
    rows = [[float(x) for x in ln.split()] for ln in dat.splitlines()
            if re.match(r"^\s*\d+\s+[-+0-9.E]+\s+[-+0-9.E]+\s+[-+0-9.E]+\s*$", ln)]
    assert rows, "ccx gave no displacements"
    return np.array(rows)[:, 1:]


@pytest.mark.skipif(not config.find_ccx().exists(), reason="CalculiX is not installed")
@pytest.mark.parametrize("axis, deg", [([0, 0, 1], 33.0), ([1, 2, 3], 41.0)])
def test_ccx_aniso_equals_the_rotated_ortho_reference(tmp_path, axis, deg):
    """The same material three ways: ORTHO + *ORIENTATION (the reference of ccx), ANISO of the global C, ANISO in the material axes + *ORIENTATION.
    The old writer (9 constants of the global C) is wrong by 5–15 % – the check has teeth."""
    C = ortho_c()
    R = rot(axis, np.radians(deg))                       # rows = the material axes in the global ones (as printing.line_axes)
    orient = "*ORIENTATION, NAME=OR1, SYSTEM=RECTANGULAR\n" + ", ".join(f"{v:.12g}" for v in (*R[0], *R[1])) + "\n"
    ref = run_ccx(hex_block(elastic_card(C), orient, ", ORIENTATION=OR1"), tmp_path / "ref")
    Cg = strain_matrix(R).T @ C @ strain_matrix(R)
    assert not is_orthotropic(Cg)
    sc = np.abs(ref).max()
    glob = run_ccx(hex_block(elastic_card(Cg), "", ""), tmp_path / "glob")
    assert np.abs(glob - ref).max() / sc < 2e-6
    d = [C[i, j] for i, j in ANISO_ORDER]                # ANISO forced for the (orthotropic) C in the material axes: ccx rotates it with *ORIENTATION
    forced = ("*ELASTIC, TYPE=ANISO\n" + ", ".join(f"{v:.12g}" for v in d[:8]) + ",\n" + ", ".join(f"{v:.12g}" for v in d[8:16]) + ",\n"
              + ", ".join(f"{v:.12g}" for v in d[16:]) + ", 0.\n")
    loc = run_ccx(hex_block(forced, orient, ", ORIENTATION=OR1"), tmp_path / "loc")
    assert np.abs(loc - ref).max() / sc < 2e-6
    d9 = [Cg[i, j] for i, j in ORTHO_ORDER]
    old = "*ELASTIC, TYPE=ORTHO\n" + ", ".join(f"{v:.12g}" for v in d9[:8]) + f",\n{d9[8]:.12g}, 0.\n"
    assert np.abs(run_ccx(hex_block(old, "", ""), tmp_path / "old") - ref).max() / sc > 1e-2


# --------------------------------------------------------------------------------------------- the deck of a printed part
def printed_box(theta_deg: float):
    """A box of 6-tetrahedra cubes as one printed part with grid infill and perimeters whose lines run at theta in the layer plane."""
    from tests.test_perf_equivalence import box_mesh, study_for
    m = box_mesh((3, 2, 2))
    m.tet_part = np.zeros(len(m.tets), np.int32)
    m.part_names = ["A"]
    n = len(m.tets)
    m.cell_data["print_shell"] = np.full(n, 0.4)
    th = np.radians(theta_deg)
    m.cell_data["print_line"] = np.tile([np.cos(th), np.sin(th), 0.0], (n, 1))
    st = study_for(m, [])
    st.add_material(PLA)
    st.part_materials["A"] = PLA.name
    st.printing.parts["A"] = PrintSpec("A", infill=0.2, pattern="grid")
    return st, m


@pytest.mark.parametrize("theta, kind", [(0.0, "ORTHO"), (90.0, "ORTHO"), (45.0, "ORTHO"), (33.0, "ANISO"), (12.5, "ANISO")])
def test_deck_card_follows_the_line_angle(tmp_path, theta, kind):
    """Lines along the print axes (class 0°, 90°) and at 45° (the ±45° grid is orthotropic in axes rotated by 45°) stay ORTHO – byte-identical decks;
    any other angle makes the grid core monoclinic in the axes of the line → ANISO."""
    st, m = printed_box(theta)
    groups = material_groups(st, m)
    assert groups and all(g["line_bin"] >= 0 for g in groups)
    write_inp(st, m, tmp_path / "a.inp")
    txt = (tmp_path / "a.inp").read_text()
    cards = re.findall(r"\*ELASTIC, TYPE=(\w+)", txt)
    assert cards and set(cards) == {kind}
    assert (kind == "ANISO") == any(not is_orthotropic(g["C"]) for g in groups)

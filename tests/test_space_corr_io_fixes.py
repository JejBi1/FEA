"""F14 S5 review fixes - the importers (`fea_core.space._corr_io`): local sensor coordinate systems (C1), vendor layouts (I1), function identity (I3), signs, CSV keys / damping units,
the delimiter of a dataset and the UFF 58 writer. The UFF files are built by `tests/uff_builders.py` in the layouts of public sample files (the sample files are not copied)."""
from __future__ import annotations

import numpy as np
import pytest
import uff_builders as U

from fea_core.space import _corr_io as io
from fea_core.space import correlation as C


def _rot_modes_text(local: dict[int, list[float]], cs_of: dict[int, int], systems: dict[int, tuple[int, np.ndarray]], first: str = "cs") -> str:
    """One mode (10 Hz) whose node values are given in the local systems; the CS datasets come before / after the mode."""
    mode = U.ds55(1, 10.0, local)
    cs = U.ds2411(cs_of) + U.ds2420(systems)
    return U.join(cs, mode) if first == "cs" else U.join(mode, cs)


# ------------------------------------------------------------------ C1: local sensor coordinate systems
def test_local_systems_rotate_modes_into_the_global_system_in_any_dataset_order():
    r = U.rot_y()
    local = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0], 4: [1.0, 2.0, 3.0]}
    for order in ("cs", "mode"):
        text = _rot_modes_text(local, {1: 1, 2: 1, 3: 1, 4: 0}, {1: (0, r)}, order)
        td = io.parse_uff(text, rotation="rows")
        sh = td.modes[0].shape
        assert [sh[("1", d)] for d in "xyz"] == pytest.approx(list(r[0]))                 # rows convention: a local x component is the first row of R in the global system
        assert [sh[("2", d)] for d in "xyz"] == pytest.approx(list(r[1]))
        assert [sh[("3", d)] for d in "xyz"] == pytest.approx(list(r[2]))
        assert [sh[("4", d)] for d in "xyz"] == [1.0, 2.0, 3.0]                            # node 4 has the global system: untouched
        assert not td.unresolved and any("coordinate systems" in w for w in td.warnings)        # the presence of the datasets is always reported
        col = io.parse_uff(text).modes[0].shape                                           # the default is the columns convention (OpenModal): a local x component is the first COLUMN of R
        assert [col[("1", d)] for d in "xyz"] == pytest.approx(list(r[:, 0])) and [col[("3", d)] for d in "xyz"] == pytest.approx(list(r[:, 2]))


def test_dataset_15_defines_the_node_systems_too_and_a_six_value_mode_rotates_its_rotations():
    r = U.rot_y()
    text = U.join(U.ds15({1: 1}), U.ds2420({1: (0, r)}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}))
    td = io.parse_uff(text, rotation="rows")
    sh = td.modes[0].shape
    assert [sh[("1", d)] for d in ("x", "y", "z")] == pytest.approx(list(r[0])) and [sh[("1", d)] for d in ("rx", "ry", "rz")] == pytest.approx(list(r[1]))


def test_a_node_whose_system_is_not_fully_known_is_left_out_with_a_warning_never_taken_as_global():
    r = U.rot_y()
    local = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0], 4: [1.0, 1.0, 1.0], 5: [1.0, 1.0, 1.0]}
    cs_of = {1: 1, 2: 2, 3: 3, 4: 9, 5: 0}
    systems = {1: (0, r), 2: (1, r), 3: (0, 2.0 * r)}                                        # 2: cylindrical, 3: not a rotation, 9: not defined at all
    td = io.parse_uff(_rot_modes_text(local, cs_of, systems))
    assert set(td.unresolved) == {"2", "3", "4"} and {n for n, _ in td.modes[0].shape} == {"1", "5"}
    assert "cylindrical" in td.unresolved["2"] and "orthonormal" in td.unresolved["3"] and "not defined" in td.unresolved["4"]
    assert all(any(f"node {n} is left out" in w for w in td.warnings) for n in "234")
    old = U.join(U.ds2411({1: 7}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0]}), [U.D, "    18", f"{7:10d}{0:10d}{0:10d}{8:10d}", U.D])
    td = io.parse_uff(old)                                                                    # dataset 18 is recognised but not interpreted
    assert td.unresolved == {"1": "coordinate system 7 is defined only in dataset 18, which is not interpreted"} and any("dataset 18" in w for w in td.warnings)


def test_without_a_node_dataset_the_directions_are_assumed_global_and_that_is_said():
    td = io.parse_uff(U.join(U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]})))
    assert any("no node dataset" in w for w in td.warnings)


def test_function_at_a_node_with_a_local_system_gets_a_global_axis_and_keeps_its_sign():
    r = U.rot_y()
    text = U.join(U.ds2411({1: 1, 2: 0}), U.ds2420({1: (0, r)}), U.ds58("a", node=1, direction=-1, ref_node=0, ref_dir=0), U.ds58("b", node=1, direction=2), U.ds58("c", node=2, direction=-3))
    td = io.parse_uff(text, rotation="rows")
    a, b, c = td.functions
    assert a.axis == pytest.approx(tuple(-r[0])) and a.local_cs == 1 and a.dof == "x"
    assert np.all(a.values.real < 0) and b.axis == pytest.approx(tuple(r[1]))                 # the sign of -X went into the axis, the values are not negated again
    assert c.axis is None and c.dof == "z" and np.all(c.values.real > 0)                      # a global node: the sign is applied to the values (the data are -1 ... -6)
    unknown = io.parse_uff(U.join(U.ds2411({1: 4}), U.ds58("a", node=1, direction=1)))
    assert unknown.functions[0].excluded and "not defined" in unknown.functions[0].excluded


def test_coordinate_system_datasets_are_bounded_and_strict(monkeypatch):
    bad = U.join(U.ds2411({1: 1}), [U.D, "  2420", f"{1:10d}", "None", f"{1:10d}{0:10d}{8:10d}", "CS", U.D], U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]}))
    with pytest.raises(io.TestDataError, match="2420"):
        io.parse_uff(bad)
    with pytest.raises(io.TestDataError, match="2411"):
        io.parse_uff(U.join([U.D, "  2411", f"{1:10d}{0:10d}{1:10d}{8:10d}", U.D], U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]})))
    with monkeypatch.context() as mp:
        mp.setattr(io, "MAX_CS", 1)
        with pytest.raises(io.TestDataError, match="too many coordinate systems"):
            io.parse_uff(U.join(U.ds2420({1: (0, np.eye(3)), 2: (0, np.eye(3))}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]})))
    td = io.parse_uff(U.join(U.ds2420({1: (0, np.eye(3))}, name_blank=True), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]})))      # an empty name record is a record
    assert td.coord_systems[1].matrix is not None


# ------------------------------------------------------------------ I1: vendor layouts
def test_numbers_written_without_a_space_are_read_by_the_widths_of_the_format():
    x = np.arange(1.0, 13.0)
    y = -np.arange(1.0, 13.0) * 1.0e-3
    text = U.join(U.ds58("stuck", x=x, y=y, stuck=True), U.ds55(1, 10.0, {1: [-1.5, -2.5, -3.5], 2: [-1.0, -2.0, -3.0]}, stuck=True))
    assert "E+000-" in text                                 # really no separator between the numbers: the exponent of one number is followed directly by the sign of the next
    td = io.parse_uff(text)
    assert td.functions[0].freq.tolist() == x.tolist() and td.functions[0].values == pytest.approx(y, rel=1e-5)
    assert td.modes[0].shape[("1", "z")] == -3.5 and td.modes[0].shape[("2", "x")] == -1.0
    with pytest.raises(io.TestDataError):                                                      # a count that no width gives is still refused
        io.parse_uff(text.replace("-1.20000E-002", "", 1))


def test_complex_mode_dataset_of_analysis_type_3_is_read_with_the_complex_mac():
    lam = complex(-0.5, 2 * np.pi * 12.0)
    nodes = {7: [0.1 + 0.2j, 0.0, 1.0 - 0.5j], 8: [0.3j, 0.4, 0.2]}
    td = io.parse_uff(U.join(U.ds55(3, 12.0, nodes, analysis=3, eig=lam, head5="999999         3         8        13")))
    m = td.modes[0]
    assert m.number == 3 and m.frequency_hz == pytest.approx(abs(lam) / (2 * np.pi)) and m.damping == pytest.approx(0.5 / abs(lam)) and m.shape[("7", "x")] == pytest.approx(0.1 + 0.2j)
    assert any("complex mode" in w and "complex MAC" in w and "rad/s" in w for w in td.warnings)
    # the complex shape is correlated with the complex MAC: an arbitrary phase of the whole mode does not change it
    phi = 1j * np.array([[1.0], [0.5]])                                                        # a mode that is purely imaginary in the phase the test system chose
    fe = np.array([[1.0], [0.5]])
    assert C.mac_matrix(phi, fe)[0, 0] == pytest.approx(1.0) and C.mac_matrix(phi.real, fe)[0, 0] == 0.0         # its real part alone would say "no correlation"
    with pytest.raises(io.TestDataError, match="outside"):                                     # a real (overdamped) eigenvalue has zeta = 1: not a mode
        io.parse_uff(U.join(U.ds55(3, 12.0, nodes, analysis=3, eig=complex(-100.0, 0.0))))


def test_a_controller_psd_without_node_direction_and_abscissa_type_is_a_frequency_function():
    f = np.arange(0.0, 41.0)
    asd = 1e-3 * (1 + np.sin(f))
    td = io.parse_uff(U.join(U.ds58("Power Spectral Density (PSD)", ftype=9, node=0, direction=0, x=f, y=asd + 0j, ord_dtype=0, ord_unit="g\xb2/Hz", den_dtype=0, den_unit="NONE",
                                    abs_dtype=0, abs_label="Hz", abs_unit="Hz", rsp_name="Pilot 1")))
    fn = td.functions[0]
    assert fn.kind == "psd" and fn.domain == "frequency" and fn.node == "Pilot 1" and fn.dof == "" and fn.unit == "g\xb2/Hz" and not np.iscomplexobj(fn.values)
    assert any("taken as frequency" in w for w in td.warnings)
    tdt = io.parse_uff(U.join(U.ds58("t", ftype=1, abs_dtype=17, abs_label="Time", abs_unit="s")))
    assert tdt.functions[0].domain == "time" and tdt.functions[0].kind == "info"


def test_a_file_without_usable_data_says_what_it_holds():
    with pytest.raises(io.TestDataError) as e:
        io.parse_uff(U.join(U.ds2411({1: 0}), U.ds2420({1: (0, np.eye(3))})))
    assert "datasets found: 2411 x1, 2420 x1" in str(e.value) and "no dataset 55" in str(e.value)
    static = U.join(U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]})).replace(f"{1:10d}{2:10d}{2:10d}{8:10d}", f"{1:10d}{5:10d}{2:10d}{8:10d}", 1)
    with pytest.raises(io.TestDataError, match="analysis type 5 is not a normal / complex mode shape"):
        io.parse_uff(static)


def test_the_delimiter_is_minus_one_in_the_columns_one_to_six_only():
    lines = [U.D, "   164", "title", "        -1", "        -1", U.D]
    blocks = io._uff_blocks(lines)
    assert blocks == [("164", 2, 5)]                                                           # the two "-1" further right are data, not delimiters
    for ln, expect in (("    -1", True), ("-1", True), ("  -1  ", True), ("        -1", False), ("    -1 x", False), ("    -12", False)):
        assert io._is_delim(ln) is expect, ln
    with pytest.raises(io.TestDataError, match="columns 1 to 6"):
        io._uff_blocks(["        -1", "    58", U.D])


# ------------------------------------------------------------------ I3: the identity of a function
def test_functions_get_names_from_id_line_1_type_and_both_directions_and_only_frf_and_psd_are_compared():
    f = np.arange(1.0, 7.0)
    fns = io.parse_uff(U.join(
        U.ds58("H1", ftype=4, node=100, direction=2, ref_node=1, ref_dir=2, x=f), U.ds58("H1", ftype=4, node=100, direction=2, ref_node=2, ref_dir=2, x=f),
        U.ds58("H1", ftype=6, node=100, direction=2, ref_node=1, ref_dir=2, x=f), U.ds58("H1", ftype=3, node=100, direction=2, ref_node=1, ref_dir=2, x=f),
        U.ds58("H1", ftype=18, node=100, direction=2, x=f), U.ds58("H1", ftype=5, node=100, direction=2, ref_node=1, ref_dir=2, x=f),
        U.ds58("A", ftype=2, node=100, direction=2, x=f), U.ds58("A", ftype=9, node=100, direction=2, x=f), U.ds58("A", ftype=10, node=100, direction=2, x=f),
        U.ds58("H1", ftype=4, node=100, direction=2, ref_node=1, ref_dir=2, x=f))).functions
    names = [x.name for x in fns]
    assert len(set(names)) == len(names) == 10                                                # the same ID line and nodes never give the same name (the last one gets a suffix)
    assert names[0] == "H1 - FRF - 100:y ref 1:y" and names[1] == "H1 - FRF - 100:y ref 2:y" and names[2] == "H1 - coherence - 100:y ref 1:y" and names[-1].endswith("#2")
    assert [x.kind for x in fns] == ["frf", "frf", "info", "info", "info", "frf", "psd", "psd", "info", "frf"]       # coherence, cross spectrum, MIF, ESD: information only
    assert fns[0].reference_node == "1" and fns[0].reference_dof == "y" and fns[0].func_type == 4


def test_the_ordinate_quantity_and_the_denominator_of_a_function_are_carried():
    f = np.arange(1.0, 7.0)
    td = io.parse_uff(U.join(U.ds58("a", x=f, ord_dtype=12, ord_unit="(m/s^2)", den_dtype=13, den_unit="N"), U.ds58("b", x=f, ord_dtype=8, ord_unit="mm", den_dtype=0, den_unit="NONE"),
                             U.ds58("c", ftype=5, x=f, ord_dtype=12, ord_unit="g", den_dtype=12, den_unit="g")))
    a, b, c = td.functions
    assert (a.quantity, a.per, a.unit, a.per_unit) == ("acceleration", "force", "(m/s^2)", "N") and (b.quantity, b.per) == ("displacement", "") and (c.quantity, c.per) == ("acceleration", "acceleration")


# ------------------------------------------------------------------ signs
def test_a_psd_is_never_negated_and_an_frf_follows_the_product_of_the_two_directions():
    f = np.arange(1.0, 7.0)
    y = np.arange(1.0, 7.0)
    td = io.parse_uff(U.join(U.ds58("psd", ftype=9, node=12, direction=-3, x=f, y=y), U.ds58("frf", ftype=4, node=12, direction=-3, ref_node=5, ref_dir=3, x=f, y=y),
                             U.ds58("frf2", ftype=4, node=12, direction=-3, ref_node=5, ref_dir=-3, x=f, y=y), U.ds58("frf3", ftype=4, node=12, direction=3, ref_node=5, ref_dir=-3, x=f, y=y)))
    psd, a, b, c = td.functions
    assert psd.values.tolist() == y.tolist() and psd.dof == "z"
    assert a.values.tolist() == (-y).tolist() and b.values.tolist() == y.tolist() and c.values.tolist() == (-y).tolist()
    csv = io.parse_csv("function,node,dof,kind,frequency_hz,value\nA,1,-z,psd,1,2\nA,1,-z,psd,2,3\nB,1,-z,frf,1,2\nB,1,-z,frf,2,3\n")
    assert csv.functions[0].values.tolist() == [2.0, 3.0] and csv.functions[1].values.tolist() == [-2.0, -3.0]


# ------------------------------------------------------------------ CSV
def test_csv_mode_keys_that_are_not_numbers_never_collide_with_numeric_ones():
    td = io.parse_csv("mode,frequency_hz,node,dof,value\n2,5,1,x,1\nA,6,1,x,1\nB,7,1,x,1\n1,8,1,x,1\n")
    assert {m.frequency_hz: m.number for m in td.modes} == {8.0: 1, 5.0: 2, 6.0: 3, 7.0: 4} and len({m.number for m in td.modes}) == 4
    with pytest.raises(io.TestDataError, match="same number"):
        io.parse_csv("mode,frequency_hz\n2,5\n2.0,6\n")


def test_csv_damping_needs_a_unit_above_0_2_and_every_row_is_checked():
    with pytest.raises(io.TestDataError, match="damping_unit"):
        io.parse_csv("mode,frequency_hz,damping\n1,5,3\n")                                    # 3 ... percent? a ratio? the file must say
    with pytest.raises(io.TestDataError, match="damping_unit"):
        io.parse_csv("mode,frequency_hz,damping\n1,5,0.01\n2,6,0.5\n")                        # the second row
    td = io.parse_csv("mode,frequency_hz,damping,damping_unit\n1,5,3,percent\n2,6,0.04,ratio\n3,7,0.5,ratio\n")
    assert [m.damping for m in td.modes] == [pytest.approx(0.03), pytest.approx(0.04), 0.5]
    for bad, why in (("mode,frequency_hz,damping,damping_unit\n1,5,150,percent\n", "ratio"), ("mode,frequency_hz,damping,damping_unit\n1,5,0.01,furlongs\n", "damping_unit"),
                     ("mode,frequency_hz,damping\n1,5,-0.01\n", "ratio"), ("mode,frequency_hz,damping,node,dof,value\n1,5,0.01,1,x,1\n1,,0.02,1,y,1\n", "two different damping")):
        with pytest.raises(io.TestDataError, match=why):
            io.parse_csv(bad)
    back = io.parse_csv(io.write_modes_csv([io.TestMode(1, 5.0, 0.015, {("1", "x"): 1.0})]))
    assert back.modes[0].damping == 0.015                                                      # the writer states the unit


def test_csv_functions_carry_unit_quantity_and_denominator():
    td = io.parse_csv("function,node,dof,frequency_hz,value,unit,quantity,per\nA,1,x,1,2,(m/s^2)/N,acceleration,force\nA,1,x,2,3,(m/s^2)/N,acceleration,force\n")
    fn = td.functions[0]
    assert (fn.unit, fn.quantity, fn.per) == ("(m/s^2)/N", "acceleration", "force")
    for bad in ("quantity,per\nacceleration,newton", "quantity,per\nweight,none"):
        h, row = bad.split("\n")
        with pytest.raises(io.TestDataError):
            io.parse_csv(f"function,node,dof,frequency_hz,value,{h}\nA,1,x,1,2,{row}\n")


# ------------------------------------------------------------------ the writer
def test_write_uff58_carries_name_quantity_denominator_and_exact_abscissa():
    f = np.linspace(2.0, 200.0, 400)                                                           # an increment of 0.4962406... that no 13.5E field can hold
    y = (1.0 + 2.0j) * np.sin(f / 30.0) + 0.1
    fn = C.TestFunction("100:y", "100", "y", f, y, "frf", "g/N", quantity="acceleration", per="force", per_unit="1/s", reference_node="5", reference_dof="z")
    text = io.write_uff58([fn])
    lines = text.splitlines()
    assert lines[2].strip() == "100:y"                                                         # the name is the ID line 1
    assert int(lines[8][0:10]) == 6 and int(lines[8][20:30]) == 0                              # double precision complex, abscissa written value by value (no inexact increment)
    assert int(lines[10][0:10]) == 12 and lines[10][47:67].strip() == "g/N" and int(lines[11][0:10]) == 13
    back = io.parse_uff(text).functions[0]
    assert np.allclose(back.freq, f, rtol=1e-12, atol=0) and np.allclose(back.values, y, rtol=1e-12, atol=0) and (back.quantity, back.per, back.unit) == ("acceleration", "force", "g/N")
    assert back.name.startswith("100:y - FRF - 100:y ref 5:z") and back.reference_dof == "z"
    even = io.write_uff58([C.TestFunction("e", "1", "x", np.arange(1.0, 9.0) * 0.5, np.arange(8.0), "frf", "g")]).splitlines()      # 0.5, 1.0 ... is exact: the even record is used
    assert int(even[8][0:10]) == 4 and int(even[8][20:30]) == 1 and io.parse_uff("\n".join(even)).functions[0].freq.tolist() == (np.arange(1.0, 9.0) * 0.5).tolist()
    psd = io.parse_uff(io.write_uff58([C.TestFunction("p", "1", "z", f[:5], np.ones(5), "psd", "g^2/Hz", quantity="acceleration")])).functions[0]
    assert psd.kind == "psd" and psd.func_type == 9 and psd.unit == "g^2/Hz"


# ------------------------------------------------------------------ the new parsers are as strict as the old ones
def _everything() -> str:
    f = np.arange(1.0, 9.0)
    return U.join(U.ds164(), U.ds2411({1: 1, 2: 0}), U.ds15({3: 1}), U.ds2420({1: (0, U.rot_y())}),
                  U.ds55(1, 10.0, {1: [1.0, 0.5, 0.2], 2: [0.3, -0.2, 0.1]}, stuck=True), U.ds55(2, 12.0, {1: [1.0 + 1j, 0.5, 0.2j], 2: [0.3, -0.2, 0.1]}, analysis=3),
                  U.ds58("H", node=1, direction=-2, ref_node=2, ref_dir=2, x=f, y=f * (1 - 1j), stuck=True), U.ds58("P", ftype=9, node=0, direction=0, x=f, y=f, abs_dtype=0))


def test_truncated_and_fuzzed_files_with_coordinate_systems_are_refused_cleanly():
    text = _everything()
    td = io.parse_uff(text)
    assert len(td.modes) == 2 and len(td.functions) == 2 and td.unit_system["code"] == 9 and td.node_cs == {"1": 1, "2": 0, "3": 1} and not td.unresolved
    lines = text.splitlines(keepends=True)
    for n in range(1, len(lines)):                                   # cut after every line
        try:
            io.parse_uff("".join(lines[:n]))
        except io.TestDataError:
            pass
    rng = np.random.default_rng(23)
    raw = bytearray(text.encode())
    for _ in range(600):
        b = bytearray(raw)
        for _ in range(int(rng.integers(1, 6))):
            b[int(rng.integers(0, len(b)))] = int(rng.integers(32, 127)) if rng.random() < 0.8 else int(rng.choice([0, 10, 13, 69, 45, 43, 46]))
        try:
            io.parse_uff(io.decode_bytes(bytes(b)))
        except io.TestDataError:
            pass


def test_merging_two_files_keeps_the_function_names_unique():
    f = np.arange(1.0, 7.0)
    a = io.parse_uff(U.join(U.ds58("H", node=1, direction=3, x=f)))
    b = io.parse_uff(U.join(U.ds58("H", node=1, direction=3, x=f)))
    assert a.functions[0].name == b.functions[0].name
    a.merge(b)
    assert [x.name for x in a.functions] == ["H - FRF - 1:z", "H - FRF - 1:z #2"]

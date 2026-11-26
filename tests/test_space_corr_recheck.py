"""F14 S5 re-check fixes (N1 - N13): the provenance of a stored correlation result covers EVERY input of the comparison and an unverifiable record is never PASS (N1, N2, N6, N7), the
rotation convention of UFF 2420 is an explicit, confirmed setting (N3), a reduced scope is PARTIAL_SCOPE (N4), criteria declared not applicable are visible (N5), UFF 55 damping (N8),
UFF 58 units (N9), the unit parser (N10), the eigenvalue unit (N11), node systems (N12) and the report (N13). Every scenario of the re-check probes is one test here."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import uff_builders as U
from test_space_correlation import _orthogonality_mass  # noqa: F401  (autouse fixture)
from test_space_correlation import EXAMPLE, FREQS, SENSORS, _box_mesh, _doc_text, _limits_set, _modes, _results, _study_with_limits, _test_data

from fea_core.report import space_corr as RC
from fea_core.report.word import _Doc
from fea_core.space import _corr_io as io
from fea_core.space import _corr_units as UN
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from fea_core.space.model import LimitEntry
from fea_core.space.requirements import confirm

F = np.linspace(2.0, 200.0, 400)


@pytest.fixture(scope="module")
def fe():
    mesh = _box_mesh()
    return mesh, _modes(mesh)


def _dyn_results(fe, zeta: float):
    """Results with a harmonic response of the sensor 100 along y (the modal FRF of the model with the damping `zeta`)."""
    mesh, ms = fe
    phi, _ = C.fe_shapes_at_sensors(ms, mesh, SENSORS, [("100", "y")])
    u = C.modal_frf(F, np.array(FREQS), zeta, phi[0], phi[0], "receptance")
    r = _results(fe)
    r.dynamics = SimpleNamespace(harmonic={"f": F, "probes": {"100": {"U": np.stack([0 * u, u, 0 * u], axis=1)}}}, random=None)
    return r


def _dyn_case(fe, zeta: float = 0.02):
    """(study, test data, results, result): modes + one measured displacement FRF that equals the model curve, correlated with the dynamics results."""
    r = _dyn_results(fe, zeta)
    pred = C.predicted_from_dynamics(r.dynamics, "100", "y", "harmonic_displacement")
    td = _test_data(fe)
    td.functions = [C.TestFunction("100:y", "100", "y", F, np.abs(pred.values), "frf", "mm")]
    st = _study_with_limits(EXAMPLE)
    res, _ = CR.run_correlation(st, r, td, sensors=list(SENSORS), setups={"100:y": C.FunctionSetup(frame="relative")})
    return st, td, r, res


def _report_verdict(st, results=None, test=None) -> str:
    D = _Doc()
    RC.correlation_section(D, st, "9", results)
    return next((ln for ln in _doc_text(D).splitlines() if ln.startswith("Verdict of the test correlation")), "?")


# ------------------------------------------------------------------ N1 / N2: the provenance covers every input
def test_n1_a_pass_is_not_shown_after_the_dynamics_were_solved_again(fe):
    st, td, r, res = _dyn_case(fe)
    assert res.verdict == C.PASS and [x.status for x in res.rows if x.item.startswith("response")] == ["PASS"] * 3
    v = CR.verify_stored(st, r, td)
    assert v["verdict"] == C.PASS and v["current"] and "dynamics responses" in v["verified"] and "test data" in v["verified"]
    assert res.provenance()["curves"].keys() == {"100:y"} and res.provenance()["curves_source"] == "dynamics"
    r2 = _dyn_results(fe, 0.2)                                              # solved again with another damping: the model curve is 10 x lower, the stored PASS is stale
    v = CR.verify_stored(st, r2, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "model curve of 100:y" in " ".join(v["reasons"]) and v["stored_verdict"] == C.PASS
    assert _report_verdict(st, r2, td) == "Verdict of the test correlation: OUTDATED"
    assert _report_verdict(st, r, td) == "Verdict of the test correlation: PASS"
    r3 = _results(fe)                                                       # the dynamics results are gone while the modes are still there
    v = CR.verify_stored(st, r3, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "model curve" in " ".join(v["reasons"])


def test_n1_results_that_are_not_at_hand_make_the_record_unverifiable_never_pass(fe):
    st, td, r, _res = _dyn_case(fe)
    v = CR.verify_stored(st, None, td)                                      # the model was edited and its results discarded
    assert v["verdict"] == C.VERDICT_UNVERIFIABLE and not v["current"] and v["reasons"] == [] and "solved results are not at hand" in v["unverified"][0]
    assert _report_verdict(st, None, td) == "Verdict of the test correlation: UNVERIFIABLE"
    D = _Doc()
    RC.correlation_section(D, st, "9", None)
    text = _doc_text(D)
    assert "says PASS" in text and "could not be verified" in text and "solve the model and correlate again" in text
    st2, td2, r2, _ = _dyn_case(fe)                                         # the test data are not at hand either (a re-opened project)
    st2.keepalive = None
    CR.remember_test(st2, None)
    v = CR.verify_stored(st2, r2)
    assert v["verdict"] == C.VERDICT_UNVERIFIABLE and "test data are not at hand" in v["unverified"][0]
    assert CR.verify_stored(st2, r2, td2)["verdict"] == C.PASS


def test_n1_model_curves_that_the_caller_supplied_cannot_be_recomputed(fe):
    st = _study_with_limits(EXAMPLE)
    td = _test_data(fe)
    td.functions = [C.TestFunction("100:y", "100", "y", F, np.ones(F.size), "frf", "g/N")]
    r = _results(fe)
    CR.run_correlation(st, r, td, sensors=list(SENSORS), predicted={"100:y": C.Predicted(F, np.ones(F.size), "frf", "", "acceleration", "", "force")})
    v = CR.verify_stored(st, r, td)
    assert v["verdict"] == C.VERDICT_UNVERIFIABLE and "supplied by the caller" in v["unverified"][0]


def test_n2_an_edited_sensor_setup_damping_or_test_changes_the_verdict_to_outdated(fe):
    st, td, r, _ = _dyn_case(fe)
    assert CR.verify_stored(st, r, td)["verdict"] == C.PASS
    CR.state_of(st)["sensors"][0]["point"] = [90.0, 1.0, 1.0]               # a moved sensor
    v = CR.verify_stored(st, r, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "sensors of the study changed" in " ".join(v["reasons"])
    assert _report_verdict(st, r, td) == "Verdict of the test correlation: OUTDATED"
    st, td, r, _ = _dyn_case(fe)
    CR.state_of(st)["functions"]["100:y"] = C.FunctionSetup(frame="absolute").to_dict()      # the frame of a function changed from relative to absolute
    v = CR.verify_stored(st, r, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "setups of the measured functions" in " ".join(v["reasons"])
    assert _report_verdict(st, r, td) == "Verdict of the test correlation: OUTDATED"
    st, td, r, _ = _dyn_case(fe)
    st.dynamics.damping.zeta = 0.05                                         # the damping of the dynamics analysis was edited (its results are not re-solved yet)
    st.dynamics.damping.table = [[10.0, 0.05]]
    v = CR.verify_stored(st, r, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "damping table" in " ".join(v["reasons"])
    st, td, r, _ = _dyn_case(fe)
    td.modes[0].frequency_hz *= 1.001                                       # the measured data changed (not only the file bytes)
    v = CR.verify_stored(st, r, td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "test data" in " ".join(v["reasons"])


def test_n2_the_model_modes_changed_after_the_result(fe):
    from fea_core.eigen import ModeSet
    mesh, ms = fe
    st, td, r, _ = _dyn_case(fe)
    shifted = ModeSet("modal", "modes", [f * 1.01 for f in FREQS], ms.shapes)
    r2 = _dyn_results(fe, 0.02)
    r2.modes = [shifted]
    assert CR.verify_stored(st, r2, td)["verdict"] == C.VERDICT_OUTDATED


def test_n2_the_damping_that_the_correlation_wrote_is_part_of_the_result_it_goes_with(fe):
    st = _study_with_limits(EXAMPLE)
    td = _test_data(fe)
    r = _results(fe)
    CR.run_correlation(st, r, td, sensors=list(SENSORS), apply_damping=True)
    assert st.dynamics.damping.table and CR.verify_stored(st, r, td)["verdict"] == C.PASS
    st.dynamics.damping.table = [[1.0, 0.5]]
    assert CR.verify_stored(st, r, td)["verdict"] == C.VERDICT_OUTDATED


# ------------------------------------------------------------------ N6 / N7: damaged records, hand-edited records
@pytest.mark.parametrize("key,value", [("row_counts", {"PASS": float("nan")}), ("row_counts", {"PASS": float("inf")}), ("row_counts", {"PASS": "6"}), ("row_counts", [1, 2]),
                                       ("row_counts", {"PASS": -1}), ("row_counts", {"PASS": True}), ("test_sha256", 5), ("test_sha256", [[1]]), ("test_sha256", ["a", 3]),
                                       ("fe_fingerprint", [1]), ("curves", 5), ("curves", {"a": 1}), ("complete", "yes"), ("limits_fingerprint", 5), ("limits_fingerprint", None),
                                       ("sensors_fingerprint", [1]), ("setups_fingerprint", 7), ("damping_fingerprint", {}), ("test_fingerprint", 1.5), ("rotation", "columns"),
                                       ("rows_total", "many"), ("curves_source", [1])])
def test_n6_a_damaged_provenance_never_crashes_and_is_never_pass(fe, key, value):
    st, td, r, _ = _dyn_case(fe)
    CR.state_of(st)["result"]["provenance"][key] = value
    v = CR.verify_stored(st, r, td)
    assert v is not None and v["verdict"] != C.PASS and not v["current"]
    _report_verdict(st, r, td)                                              # the report does not crash either


def test_n6_a_damaged_result_record_never_crashes(fe):
    st, td, r, _ = _dyn_case(fe)
    res = CR.state_of(st)["result"]
    for bad in ({"verdict": "PASS", "provenance": "x"}, {"verdict": "PASS", "provenance": None}, {"verdict": "PASS", "provenance": {"limits_fingerprint": "x"}, "rows": "no"},
                {"verdict": 5, "provenance": {}}, {"verdict": "PASS", "rows": [1, None, "x"], "provenance": res["provenance"]}):
        CR.state_of(st)["result"] = bad
        v = CR.verify_stored(st, r, td)
        assert v is not None and v["verdict"] != C.PASS
    CR.state_of(st)["result"] = {"verdict": ""}
    assert CR.verify_stored(st, r, td) is None
    st.space.extra["correlation"] = "damaged"
    assert CR.verify_stored(st, r, td) is None


def test_n6_n7_a_hand_edited_row_status_is_recomputed_from_its_value_and_the_current_limit(fe):
    tight = _study_with_limits({**EXAMPLE, "corr_freq_error_max": 0.001})
    td = _test_data(fe)
    for m in td.modes:
        m.frequency_hz *= 1.03
    res, _ = CR.run_correlation(tight, _results(fe), td, sensors=list(SENSORS))
    assert res.verdict == C.FAIL
    rec = CR.state_of(tight)["result"]
    rec["verdict"] = "PASS"                                                 # the verdict, the counts and the statuses of the rows are edited consistently, the values are not
    n = rec["provenance"]["row_counts"]
    n["PASS"] += n["FAIL"]
    n["FAIL"] = 0
    for row in rec["rows"]:
        if row["status"] == "FAIL":
            row["status"] = "PASS"
    v = CR.verify_stored(tight, _results(fe), td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "does not follow from its value" in " ".join(v["reasons"])


def test_n7_the_report_says_that_the_result_is_a_stored_record(fe):
    st, td, r, _ = _dyn_case(fe)
    D = _Doc()
    RC.correlation_section(D, st, "9", r)
    text = _doc_text(D)
    assert "stored result of the last correlation" in text and "tamper-evident" in text and "not a recomputation" in text and "Verified against the current study" in text
    assert "dynamics responses" in text


# ------------------------------------------------------------------ N13: the report prints the CURRENT confirmation
def test_n13_the_report_prints_the_current_confirmation_state_not_the_stored_one(fe):
    st, td, r, _ = _dyn_case(fe)
    D = _Doc()
    RC.correlation_section(D, st, "9", r)
    assert "Confirmed by the user (now)\nyes" in _doc_text(D) or "yes" in _doc_text(D)
    st.space.requirement_set.withdraw_confirmation()                        # confirmation withdrawn after the result
    D = _Doc()
    RC.correlation_section(D, st, "9", r)
    text = _doc_text(D)
    assert "Verdict of the test correlation: UNCONFIRMED_REQUIREMENTS" in text
    assert "NO - the requirement set is not confirmed" in text and "Confirmation when it was correlated" in text
    kv = [ln for ln in text.splitlines() if ln.startswith("Confirmed by the user")]
    assert kv and all("yes" not in ln.lower() for ln in kv)


# ------------------------------------------------------------------ N3: the rotation convention of the local sensor axes
BEAM_R = U.rot_y()                                              # the matrix of coordinate system 1 of the public OpenModal file beam.uff


def _beam_like(columns_convention: bool = True) -> str:
    """The structure of beam.uff: 10 nodes, each with its own copy of the system 1 matrix, FRFs along the local x of node 1."""
    systems = {k: (0, BEAM_R) for k in range(1, 11)}
    return U.join(U.ds164(), U.ds2420(systems), U.ds2411({k: k for k in range(1, 11)}), U.ds58("a", node=1, direction=1, ord_unit="m/s2", den_unit="N"))


def test_n3_beam_uff_node_1_gives_the_column_of_the_matrix_under_the_default_convention():
    td = io.parse_uff(_beam_like())
    assert td.rotation == "columns" and td.functions[0].axis == pytest.approx((-0.448, 0.0, -0.894), abs=1e-3)
    assert td.functions[0].axis == pytest.approx(tuple(BEAM_R[:, 0]))
    rows = io.parse_uff(_beam_like(), rotation="rows")
    assert rows.rotation == "rows" and rows.functions[0].axis == pytest.approx((-0.448, 0.0, 0.894), abs=1e-3)
    assert td.functions[0].node == "1" and td.rotation_ambiguous == ["1"] and td.rotation_unconfirmed       # only the nodes that carry data count
    assert any("COLUMNS or the ROWS" in w for w in td.warnings)


def test_n3_switching_the_convention_re_reads_the_data_and_drops_the_confirmation():
    local = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0]}
    text = U.join(U.ds2420({7: (0, BEAM_R)}), U.ds2411({1: 7, 2: 0}), U.ds55(1, 10.0, local), U.ds58("f", node=1, direction=3))
    td = io.parse_uff(text)
    assert [td.modes[0].shape[("1", d)] for d in "xyz"] == pytest.approx(list(BEAM_R[:, 0])) and td.rotation_ambiguous == ["1"]
    td.rotation_confirmed, td.rotation_confirmed_via = True, "gui"
    axis_cols = td.functions[0].axis
    td.set_rotation("rows")
    assert [td.modes[0].shape[("1", d)] for d in "xyz"] == pytest.approx(list(BEAM_R[0])) and not td.rotation_confirmed and td.rotation == "rows"
    assert td.functions[0].axis == pytest.approx(tuple(BEAM_R[2]))
    td.set_rotation("columns")                                              # back: exactly the first reading (the values are re-derived from the file values, not rotated twice)
    assert [td.modes[0].shape[("1", d)] for d in "xyz"] == pytest.approx(list(BEAM_R[:, 0])) and td.functions[0].axis == pytest.approx(axis_cols)
    assert [td.modes[0].shape[("2", d)] for d in "xyz"] == [0.0, 1.0, 0.0]            # the global node was never touched
    with pytest.raises(io.TestDataError, match="rotation convention"):
        td.set_rotation("diagonal")
    with pytest.raises(io.TestDataError, match="rotation convention"):
        io.parse_uff(text, rotation="x")
    assert td.summary()["rotation"] == {"convention": "columns", "confirmed": False, "confirmed_via": "", "ambiguous_nodes": ["1"], "unconfirmed": True}


def test_n3_a_symmetric_matrix_does_not_depend_on_the_convention():
    r = np.diag([-1.0, -1.0, 1.0])                                          # 180 degrees about z: the rows are the columns
    td = io.parse_uff(U.join(U.ds2420({7: (0, r)}), U.ds2411({1: 7}), U.ds55(1, 10.0, {1: [1.0, 2.0, 3.0]})))
    assert td.rotation_ambiguous == [] and not td.rotation_unconfirmed
    assert io.parse_uff(U.join(U.ds2420({7: (0, r)}), U.ds2411({1: 7}), U.ds55(1, 10.0, {1: [1.0, 2.0, 3.0]})), rotation="rows").modes[0].shape == td.modes[0].shape


def _rotated_modes_uff(fe, convention: str) -> str:
    """The FE modes written in local sensor systems of every node (the same matrix); `convention` = how the writer defines the matrix."""
    mesh, ms = fe
    src = C.fe_test_data(ms, mesh, SENSORS)
    r = U.rot_y(c=np.cos(0.5), s=np.sin(0.5))
    a = r.T if convention == "columns" else r                              # v_global = v_local @ a
    parts = [U.ds2420({7: (0, r)}), U.ds2411({int(s.node): 7 for s in SENSORS})]
    for m in src.modes:
        nodes = {int(s.node): list(np.array([m.shape[(s.node, d)] for d in "xyz"]) @ np.linalg.inv(a)) for s in SENSORS}
        parts.append(U.ds55(m.number, m.frequency_hz, nodes))
    return U.join(*parts)


def test_n3_rotated_sensors_stay_unconfirmed_until_the_user_confirms_the_convention(fe):
    mesh, ms = fe
    td = io.parse_uff(_rotated_modes_uff(fe, "columns"))
    assert td.rotation == "columns" and td.rotation_unconfirmed and len(td.rotation_ambiguous) == 8
    lim = C.CorrelationLimits.from_requirement_set(_limits_set(EXAMPLE).requirement_set, _limits_set(EXAMPLE))
    sp = _limits_set(EXAMPLE)
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    res = C.correlate(td, ms, mesh, SENSORS, limits=lim)
    assert res.verdict == C.VERDICT_UNCONFIRMED and {x.status for x in res.rows} == {C.UNCONFIRMED}
    assert min(p.mac for p in res.modal.pairs) > 0.999                       # the numbers are right (the writer used the columns), only the confirmation is missing
    assert any("not confirmed for this file" in x.note for x in res.rows) and res.coverage["rotation"]["convention"] == "columns" and res.coverage["rotation"]["rows_held_for_nodes"]
    td.rotation_confirmed, td.rotation_confirmed_via = True, "gui"
    assert C.correlate(td, ms, mesh, SENSORS, limits=lim).verdict == C.PASS
    wrong = io.parse_uff(_rotated_modes_uff(fe, "rows"))                    # a file written with the other convention, read with the default: the MAC says so
    wrong.rotation_confirmed = True
    assert min(p.mac for p in C.correlate(wrong, ms, mesh, SENSORS, limits=lim).modal.pairs) < 0.99
    wrong.set_rotation("rows")
    assert min(p.mac for p in C.correlate(wrong, ms, mesh, SENSORS, limits=lim).modal.pairs) > 0.999


def test_n3_the_confirmation_is_stored_with_the_file_and_the_convention_and_is_part_of_the_provenance(fe, tmp_path):
    mesh, ms = fe
    path = tmp_path / "rot.uff"
    path.write_text(_rotated_modes_uff(fe, "columns"))
    st = _study_with_limits(EXAMPLE)
    td = io.load_test_file(path)
    res, _ = CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS))
    assert res.verdict == C.VERDICT_UNCONFIRMED and res.provenance()["rotation"] == {"convention": "columns", "confirmed": False, "ambiguous_nodes": sorted(td.rotation_ambiguous)}
    CR.confirm_rotation(st, td, "gui")
    res, _ = CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS))
    assert res.verdict == C.PASS and res.provenance()["rotation"]["confirmed"] is True and CR.verify_stored(st, _results(fe), td)["verdict"] == C.PASS
    assert CR.state_of(st)["rotation"]["sha256"] == td.sha256 and CR.state_of(st)["rotation"]["via"] == "gui"
    again = io.load_test_file(path)                                         # the same file again: the user's confirmation is taken over
    CR.adopt_rotation(st, again)
    assert again.rotation_confirmed and again.rotation_confirmed_via == "gui"
    other = io.load_test_file(path, rotation="rows")                        # another convention: not confirmed
    CR.adopt_rotation(st, other)
    assert not other.rotation_confirmed
    path.write_text(_rotated_modes_uff(fe, "columns") + "\n")               # another file: not confirmed
    third = io.load_test_file(path)
    CR.adopt_rotation(st, third)
    assert not third.rotation_confirmed
    CR.set_rotation(st, td, "rows")                                         # the convention is changed after the result: the result is outdated, the confirmation withdrawn
    assert not td.rotation_confirmed and CR.state_of(st)["rotation"]["confirmed"] is False
    v = CR.verify_stored(st, _results(fe), td)
    assert v["verdict"] == C.VERDICT_OUTDATED and "test data" in " ".join(v["reasons"])
    CR.set_rotation(st, td, "columns")
    CR.confirm_rotation(st, td, "gui")
    CR.confirm_rotation(st, td, "gui", False)                               # the confirmation withdrawn after the result
    assert CR.verify_stored(st, _results(fe), td)["verdict"] == C.VERDICT_OUTDATED


def test_n3_a_function_at_a_rotated_node_is_unconfirmed_too(fe):
    f = np.linspace(5.0, 200.0, 100)
    td = io.parse_uff(U.join(U.ds2420({7: (0, BEAM_R)}), U.ds2411({1: 7}), U.ds58("a", node=1, direction=2, x=f, y=np.ones(100), ord_unit="g", den_unit="N")))
    sp = _limits_set(EXAMPLE)
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    pred = {"1:y": C.Predicted(f, np.ones(100), "frf", "", "acceleration", "", "force")}
    res = C.correlate(td, limits=lim, predicted=pred)
    assert res.verdict == C.VERDICT_UNCONFIRMED and all("convention" in x for x in res.responses[0].unconfirmed)
    td.rotation_confirmed = True
    assert C.correlate(td, limits=lim, predicted=pred).verdict == C.PASS


# ------------------------------------------------------------------ N4: a reduced scope is PARTIAL_SCOPE
def test_n4_sensors_that_map_fewer_dofs_than_the_test_measured_give_partial_scope(fe):
    mesh, ms = fe
    td = _test_data(fe)
    td.modes = td.modes[:4]                                              # the y and z modes
    for m in td.modes:
        for s in SENSORS:
            m.shape[(s.node, "x")] = 0.01                                # the test measured a small x component at every sensor
    sp = _limits_set(EXAMPLE)
    limits = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    full = C.correlate(td, ms, mesh, SENSORS, limits=limits)
    assert full.verdict == C.PASS and "dofs_omitted" not in full.coverage
    yz = C.sensors_from_dicts([{**s.to_dict(), "dofs": "yz"} for s in SENSORS])
    res = C.correlate(td, ms, mesh, yz, limits=limits, options=C.CorrelationOptions(fe_modes=[1, 2, 3, 4]))      # (the FE modes 5, 6 move along x: no sensor sees them, so no SEREP mass)
    assert all(x.status == C.PASS for x in res.rows)                     # everything that was compared passes ...
    assert res.verdict == C.VERDICT_PARTIAL and not res.coverage["complete"]          # ... and it is still no PASS: the x DOFs of the test were not compared
    assert res.coverage["dofs_omitted"] == [f"{s.node}:x" for s in SENSORS]
    assert any("fewer DOFs than the test measured" in r and "100:x" in r for r in res.coverage["reasons"]) and res.provenance()["complete"] is False
    assert any("measured DOF(s) of mapped test nodes are not compared" in w for w in res.warnings)
    z_only = C.sensors_from_dicts([{**s.to_dict(), "dofs": "z"} for s in SENSORS])
    assert len(C.correlate(td, ms, mesh, z_only, limits=limits).coverage["dofs_omitted"]) == 16


def test_n4_dofs_that_are_exactly_zero_in_the_test_are_not_a_reduction(fe):
    mesh, ms = fe
    td = _test_data(fe)
    for m in td.modes:
        for k in list(m.shape):
            if k[1] == "x":
                m.shape[k] = 0.0                                         # an unmeasured axis padded with zeros
    sp = _limits_set(EXAMPLE)
    limits = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    sens = C.sensors_from_dicts([{**s.to_dict(), "dofs": "yz"} for s in SENSORS])
    res = C.correlate(td, ms, mesh, sens, limits=limits)
    assert "dofs_omitted" not in res.coverage and res.coverage["complete"]


def test_n4_rotational_dofs_with_data_are_never_compared_and_reduce_the_scope(fe):
    mesh, ms = fe
    td = _test_data(fe)
    for m in td.modes:
        m.shape[("100", "rx")] = 0.5
        m.shape[("101", "rz")] = 0.0                                     # zero: nothing measured
    sp = _limits_set(EXAMPLE)
    res = C.correlate(td, ms, mesh, SENSORS, limits=C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp))
    assert res.coverage["dofs_omitted"] == ["100:rx"] and res.verdict == C.VERDICT_PARTIAL


def _resonant_pair():
    f = np.linspace(5, 400, 2000)
    good = 1.0 / np.sqrt((1 - (f / 100) ** 2) ** 2 + (0.04 * f / 100) ** 2)
    bad = good + 30.0 * np.exp(-((f - 300) / 5) ** 2)                    # the test has a second resonance at 300 Hz that the model lacks
    return f, good, bad


def test_n4_a_response_band_that_hides_a_test_resonance_gives_partial_scope_and_names_it():
    f, good, bad = _resonant_pair()
    fn = C.TestFunction("100:y", "100", "y", f, bad, "frf", "g")
    pred = {"100:y": C.Predicted(f, good, "frf", "x", "acceleration", "", "none")}
    td = C.TestData(functions=[fn])
    sp = _limits_set({**EXAMPLE, "corr_rms_error_max": 0.5, "corr_peak_amp_error_max_db": 6.0, "corr_peak_freq_shift_max": 0.5})
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    whole = C.correlate(td, limits=lim, predicted=pred)
    assert whole.verdict == C.FAIL                                       # the whole band sees the resonance the model lacks
    cut = C.correlate(td, limits=lim, predicted=pred, options=C.CorrelationOptions(response_band_hz=(5.0, 200.0)))
    assert cut.verdict == C.VERDICT_PARTIAL and not cut.coverage["complete"]
    assert cut.coverage["functions_band_limited"][0]["test_peaks_outside_band_Hz"][0] == pytest.approx(300.0, abs=1.0)
    assert any("300" in r and "outside the compared band 5 - 200 Hz" in r for r in cut.coverage["reasons"])
    assert cut.provenance()["complete"] is False and cut.responses[0].hidden_peaks_hz
    short = {"100:y": C.Predicted(f[f <= 200], good[f <= 200], "frf", "x", "acceleration", "", "none")}      # a model curve that ends at 200 Hz
    res = C.correlate(td, limits=lim, predicted=short)
    assert res.verdict == C.VERDICT_PARTIAL and any("outside the compared band" in r for r in res.coverage["reasons"])
    inside = C.correlate(td, limits=lim, predicted=pred, options=C.CorrelationOptions(response_band_hz=(250.0, 350.0)))      # the band that holds the resonance hides a different one (the 100 Hz peak)
    assert any(abs(p - 100.0) < 3.0 for p in inside.coverage["functions_band_limited"][0]["test_peaks_outside_band_Hz"])


def test_n4_a_band_that_cuts_off_only_featureless_data_is_not_a_reduction():
    f, good, _bad = _resonant_pair()
    fn = C.TestFunction("100:y", "100", "y", f, good, "frf", "g")
    pred = {"100:y": C.Predicted(f, good, "frf", "x", "acceleration", "", "none")}
    sp = _limits_set(EXAMPLE)
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    res = C.correlate(C.TestData(functions=[fn]), limits=lim, predicted=pred, options=C.CorrelationOptions(response_band_hz=(5.0, 200.0)))
    assert res.coverage["complete"] and res.verdict == C.PASS
    with pytest.raises(ValueError, match="hidden_peak_db"):
        C.CorrelationOptions(hidden_peak_db=0.0).validate()
    assert C.CorrelationOptions.from_dict(C.CorrelationOptions(hidden_peak_db=12.0).to_dict()).hidden_peak_db == 12.0


# ------------------------------------------------------------------ N5: the not-applicable criteria are visible
def _na_study(via: str = "mcp"):
    sp = _limits_set({k: v for k, v in EXAMPLE.items() if k != "corr_comac_min"}, confirmed=False)
    sp.requirement_set.limits.append(LimitEntry(key="corr_comac_min", value=None, unit="-", document="Doc-1", revision="B", clause="4.2", not_applicable=True, set_via=via))
    confirm(sp.requirement_set, "test engineer", "2026-01-01", via="gui", scope=sp)
    from verification.space_cases import block_study
    st = block_study()
    st.space = sp
    return st


def test_n5_the_result_lists_every_not_applicable_limit_with_who_set_it_and_how(fe):
    st = _na_study("mcp")
    lim = C.limits_of_study(st)
    na = C.not_applicable_limits(lim)
    assert na == [{"limit": "corr_comac_min", "meaning": C.LIMIT_INFO["corr_comac_min"][2], "document": "Doc-1", "revision": "B", "clause": "4.2", "entered_via": "mcp",
                   "confirmed_by": "test engineer", "confirmed_via": "gui", "date": "2026-01-01"}]
    td = _test_data(fe)
    res, _ = CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS))
    d = res.to_dict()
    assert d["not_applicable_limits"][0]["entered_via"] == "mcp" and d["limits"]["corr_comac_min"]["set_via"] == "mcp"
    assert "NOT APPLICABLE (not judged): corr_comac_min: not applicable per Doc-1 B, 4.2; entered through the MCP interface (an agent); set confirmed by test engineer through the GUI" in res.text()
    assert {x.status for x in res.rows if x.limit_key == "corr_comac_min"} == {C.NOT_APPLICABLE}
    D = _Doc()
    warns = RC.correlation_section(D, st, "9", _results(fe))
    text = _doc_text(D)
    assert "Correlation criteria declared not applicable" in text and "corr_comac_min" in text and "MCP interface (an agent)" in text and "Doc-1" in text
    assert any("corr_comac_min was declared not applicable" in w and "entered MCP interface (an agent)" in w for w in warns)


def test_n5_the_interface_is_recorded_by_the_program_never_taken_from_the_data():
    from fea_core.space import requirements as RQ
    e = LimitEntry.from_dict({"key": "corr_mac_min", "not_applicable": True, "set_via": "gui", "document": "d", "revision": "r", "clause": "c"})
    assert e.set_via == "gui"                                             # the stored form is read back as it was written
    rs = _limits_set({}, confirmed=False).requirement_set
    rs.limits.append(e)
    back = RQ.from_json(RQ.to_json(rs))                                   # a file cannot claim the flag was entered in the GUI
    assert back.limits[0].set_via == "import" and not back.is_confirmed()
    # the digest covers it: a flag that changes its origin withdraws the confirmation
    rs.limits[0].set_via = "gui"
    d1 = rs.content_digest()
    rs.limits[0].set_via = "mcp"
    assert rs.content_digest() != d1
    assert "set_via" not in LimitEntry(key="k", value=1.0).to_dict()      # a row without the flag keeps its old digest


def test_n5_the_mcp_tool_marks_a_flag_as_set_by_mcp(fe, tmp_path):
    pytest.importorskip("mcp")
    from fea_core.mcp.tools_space import _merge
    rs = _limits_set({}, confirmed=False).requirement_set
    _merge(rs, {"limits": [{"key": "corr_mac_min", "not_applicable": True, "set_via": "gui", "document": "d", "revision": "r", "clause": "c"}]})
    row = rs.limits[0]
    assert row.not_applicable and row.set_via == "mcp"                   # the claim of the agent about itself is overwritten
    _merge(rs, {"limits": [{"key": "corr_mac_min", "note": "edited"}]})
    assert row.set_via == "mcp"
    _merge(rs, {"limits": [{"key": "corr_mac_min", "not_applicable": False, "value": 0.9}]})
    assert not row.not_applicable and row.set_via == ""


def test_n5_the_space_dialog_summary_shows_the_limits_and_the_flags_before_the_user_confirms():
    from fea_gui.space_limits_view import limits_html
    st = _na_study("mcp")
    html = limits_html(st.space.requirement_set, st.space)
    assert "corr_comac_min" in html and "NOT APPLICABLE" in html and "entered through the MCP interface (an agent) - check it" in html and "Doc-1 B 4.2" in html
    assert "corr_mac_min" in html and "0.9" in html and "1 criterion/criteria declared NOT APPLICABLE" in html
    assert limits_html(None) == "" and limits_html(_limits_set({}, confirmed=False).requirement_set) == ""
    rs = _limits_set({}, confirmed=False).requirement_set
    rs.limits.append(LimitEntry(key="corr_rms_error_max", value=None, document="<b>x</b>", revision="r", clause="<script>", not_applicable=True, set_via="import"))
    h = limits_html(rs)
    assert "<script>" not in h and "&lt;script&gt;" in h and "from an imported file" in h


# ------------------------------------------------------------------ N8: the damping of a UFF 55
def test_n8_uff55_damping_above_the_plausible_ratio_needs_a_unit():
    for z in (0.015, 0.2):
        assert io.parse_uff(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]}, damping=z))).modes[0].damping == pytest.approx(z)
    for z in (0.5, 1.5, 12.0):
        with pytest.raises(io.TestDataError, match="ratio or a percent"):
            io.parse_uff(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]}, damping=z)))
    assert io.parse_uff(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]}, damping=1.5)), damping_unit="percent").modes[0].damping == pytest.approx(0.015)
    assert io.parse_uff(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]}, damping=0.5)), damping_unit="ratio").modes[0].damping == pytest.approx(0.5)
    with pytest.raises(io.TestDataError, match="damping unit"):
        io.parse_uff(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]})), damping_unit="pct")
    with pytest.raises(io.TestDataError, match="is outside"):
        io.parse_uff(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]}, damping=150.0)), damping_unit="percent")


# ------------------------------------------------------------------ N9: the numerator and denominator units of a UFF 58
@pytest.mark.parametrize("num,den,quantity,per", [("g", "N", "acceleration", "force"), ("m/s2", "N", "acceleration", "force"), ("g/N", "NONE", "acceleration", "force"),
                                                  ("g/N", "N", "acceleration", "force"), ("mm", "N", "displacement", "force"), ("g", "g", "acceleration", "base_acceleration")])
def test_n9_the_units_of_the_records_9_and_10_are_combined(num, den, quantity, per):
    dd = 12 if per == "base_acceleration" else 13
    fn = io.parse_uff(U.join(U.ds58("f", ord_unit=num, den_unit=den, den_dtype=dd, direction=2, ord_dtype=8 if quantity == "displacement" else 12))).functions[0]
    r = C.resolve_function(fn)
    assert not r.problems and (r.quantity, r.per) == (quantity, per)
    if (num, den) == ("g", "N"):
        assert C._unit_scale(r.unit, r.quantity, r.per, r.psd) == pytest.approx(1.0)               # g/N is the unit of the model curve
    if (num, den) == ("m/s2", "N"):
        assert C._unit_scale(r.unit, r.quantity, r.per, r.psd) == pytest.approx(1.0 / 9.81)


def test_n9_a_unit_that_does_not_fit_is_still_refused_and_the_setup_overrides():
    fn = io.parse_uff(U.join(U.ds58("f", ord_unit="s2", den_unit="N", direction=2))).functions[0]
    assert C.resolve_function(fn).problems
    fn = io.parse_uff(U.join(U.ds58("f", ord_unit="NONE", den_unit="NONE", direction=2))).functions[0]
    assert "not known" in C.resolve_function(fn).problems[0]
    assert not C.resolve_function(fn, C.FunctionSetup(unit="g/N", per="force")).problems
    psd = io.parse_uff(U.join(U.ds58("p", ftype=9, ord_unit="g2", den_unit="Hz", den_dtype=0, direction=2, node=1))).functions[0]
    r = C.resolve_function(psd)
    assert not r.problems and r.psd and r.quantity == "acceleration"


# ------------------------------------------------------------------ N10: the unit parser
@pytest.mark.parametrize("label,dims,factor", [("mN", (1, -2, 1), 1e-3), ("MN", (1, -2, 1), 1e6), ("kN", (1, -2, 1), 1e3), ("N", (1, -2, 1), 1.0), ("Mg", (0, 0, 1), 1e3), ("mg", (1, -2, 0), 9.81e-3),
                                               ("g", (1, -2, 0), 9.81), ("G", (1, -2, 0), 9.81), ("M/S^2", (1, -2, 0), 1.0), ("HZ", (0, -1, 0), 1.0), ("Hz", (0, -1, 0), 1.0), ("m/s2", (1, -2, 0), 1.0),
                                               ("ms^2", (0, 2, 0), 1e-6), ("mm/s^2", (1, -2, 0), 1e-3), ("(m/s^2)^2/Hz", (2, -3, 0), 1.0), ("G/LBF", (0, 0, -1), 9.81 / 4.4482216)])
def test_n10_unit_labels_are_read_with_their_case(label, dims, factor):
    u = UN.parse_unit(label)
    assert u is not None and u.dims == dims and u.factor == pytest.approx(factor)


@pytest.mark.parametrize("label", ["ms2", "MS2", "m/ms2", "m_s", "m_s^2", "g_N", "MG", "Mn", "mn", "kn_", "m s-2", "g rms", "NONE", "", None, "1/", "(g", "g)", "g^", "g^x", "Pa", "m//s"])
def test_n10_ambiguous_or_unknown_labels_are_refused_never_guessed(label):
    assert UN.parse_unit(label) is None


def test_n10_the_refusal_says_why():
    assert "ambiguous" in UN.unit_hint("ms2") and "underscore" in UN.unit_hint("m_s") and UN.unit_hint("xyz") == ""
    fn = C.TestFunction("f", "1", "x", np.array([1.0, 2.0]), np.array([1.0, 2.0]), "frf", "ms2")
    assert "ambiguous" in C.resolve_function(fn).problems[0]
    assert not C.resolve_function(fn, C.FunctionSetup(unit="m/s^2")).problems


# ------------------------------------------------------------------ N11: the unit of the complex eigenvalue
def test_n11_the_complex_eigenvalue_is_assumed_to_be_in_rad_per_second_and_that_is_said_once():
    lam = complex(-0.5, 2 * np.pi * 12.0)
    text = U.join(U.ds55(3, 12.0, {7: [0.1, 0.0, 1.0]}, analysis=3, eig=lam, head5="999999         3         8        13"),
                  U.ds55(4, 12.0, {7: [0.1, 0.0, 1.0]}, analysis=3, eig=lam, head5="999999         4         8        13"))
    td = io.parse_uff(text)
    notes = [w for w in td.warnings if "ASSUMED to be in rad/s" in w]
    assert len(notes) == 1 and "2 pi too low" in notes[0]


# ------------------------------------------------------------------ N12: node systems
def test_n12_system_1_without_a_dataset_2420_is_global_with_a_note():
    text = U.join(U.ds2411({1: 1, 2: 0, 3: 1}), U.ds55(1, 10.0, {1: [1.0, 2.0, 3.0], 2: [0.0, 1.0, 0.0], 3: [1.0, 1.0, 1.0]}))
    td = io.parse_uff(text)
    assert not td.unresolved and td.modes[0].shape[("1", "z")] == 3.0 and not td.rotation_ambiguous
    assert any("displacement system 1" in w and "no dataset 2420" in w and "global" in w for w in td.warnings)
    defined = io.parse_uff(U.join(U.ds2411({1: 1}), U.ds2420({1: (0, BEAM_R)}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]})))          # a system 1 that the file defines is used
    assert defined.rotation_ambiguous == ["1"] and defined.modes[0].shape[("1", "x")] == pytest.approx(BEAM_R[0, 0])
    old = io.parse_uff(U.join(U.ds2411({1: 1}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]}), [U.D, "    18", f"{1:10d}{0:10d}{0:10d}{8:10d}", U.D]))
    assert "1" in old.unresolved                                           # system 1 that only the old dataset 18 would define: left out, not global


def test_n12_a_node_missing_from_the_node_dataset_gets_a_note():
    text = U.join(U.ds2411({1: 0}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0], 5: [1.0, 0.0, 0.0], 6: [0.0, 1.0, 0.0]}))
    td = io.parse_uff(text)
    assert any("2 node(s) of the data are not in the node dataset" in w and "5, 6" in w and "global" in w for w in td.warnings)
    assert not io.parse_uff(U.join(U.ds2411({1: 0, 5: 0}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0], 5: [1.0, 0.0, 0.0]}))).unresolved
    assert any("no node dataset" in w for w in io.parse_uff(U.join(U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0]}))).warnings)


def test_n12_dataset_18_is_excluded_with_a_clear_note():
    text = U.join(U.ds2411({1: 7, 2: 0}), U.ds55(1, 10.0, {1: [1.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0]}), [U.D, "    18", f"{7:10d}{0:10d}{0:10d}{8:10d}", U.D])
    td = io.parse_uff(text)
    assert set(td.unresolved) == {"1"} and {n for n, _ in td.modes[0].shape} == {"2"}
    assert any("dataset 18" in w and "LEFT OUT" in w and "never read as global" in w for w in td.warnings)


# ------------------------------------------------------------------ N14: the help
def test_n14_the_help_page_documents_the_correlation_in_both_languages():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "fea_gui" / "help"
    en = (root / "space.html").read_text(encoding="utf-8")
    cs = (root / "cs" / "space.html").read_text(encoding="utf-8")
    for needle in ("PARTIAL_SCOPE", "OUTDATED", "UNVERIFIABLE", "not applicable", "rad/s", "columns", "tamper-evident", "recompute"):
        assert needle in en, needle
    for needle in ("PARTIAL_SCOPE", "OUTDATED", "UNVERIFIABLE", "rad/s"):
        assert needle in cs, needle
    assert "tamper" in cs.lower() or "falešn" in cs.lower() or "dodatečn" in cs.lower() or "pozměn" in cs.lower()


# ------------------------------------------------------------------ the MCP tools: the rotation convention, the damping unit, the verified verdict
def _mcp_session(tmp_path, values=None):
    pytest.importorskip("mcp")
    from fea_core.mcp.session import Session
    from verification.space_cases import block_study
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    sess.study.space = _limits_set(EXAMPLE if values is None else values)
    return sess


def test_mcp_rotation_convention_is_a_setting_and_a_confirmation_is_recorded_as_mcp(fe, tmp_path):
    from test_space_correlation import _mcp
    sess = _mcp_session(tmp_path)
    sess.results = _results(fe)
    f = tmp_path / "rot.uff"
    f.write_text(_rotated_modes_uff(fe, "columns"))
    sensors = [s.to_dict() for s in SENSORS]
    o = _mcp(sess, [("fea_space_test_import", {"path": str(f), "sensors": sensors}, False),
                    ("fea_space_correlate", {}, False),
                    ("fea_space_test_import", {"rotation_confirmed": True}, False),
                    ("fea_space_correlate", {}, False),
                    ("fea_space_test_import", {"rotation": "rows"}, False),
                    ("fea_space_correlate", {}, False),
                    ("fea_space_test_import", {"rotation": "diagonal"}, True),
                    ("fea_space_test_import", {"path": str(f), "rotation": "rows", "sensors": sensors}, False)])
    assert o[0]["dataset"]["rotation"] == {"convention": "columns", "confirmed": False, "confirmed_via": "", "ambiguous_nodes": o[0]["dataset"]["rotation"]["ambiguous_nodes"], "unconfirmed": True}
    assert len(o[0]["dataset"]["rotation"]["ambiguous_nodes"]) == 8 and any("COLUMNS or the ROWS" in w for w in o[0]["dataset"]["warnings"])
    assert o[1]["verdict"] == C.VERDICT_UNCONFIRMED and o[1]["coverage"]["rotation"]["rows_held_for_nodes"]
    assert o[2]["dataset"]["rotation"]["confirmed"] and o[2]["dataset"]["rotation"]["confirmed_via"] == "mcp"
    assert o[3]["verdict"] == C.PASS and o[3]["provenance"]["rotation"]["confirmed"] is True
    assert o[4]["dataset"]["rotation"]["convention"] == "rows" and not o[4]["dataset"]["rotation"]["confirmed"]
    assert o[5]["verdict"] == C.VERDICT_UNCONFIRMED and min(p["mac"] for p in o[5]["modal"]["pairs"]) < 0.99     # read with the other convention: the shapes do not match
    assert "rotation must be" in o[6] and o[7]["dataset"]["rotation"]["convention"] == "rows"


def test_mcp_damping_unit_and_the_tool_text_mention_the_new_arguments(fe, tmp_path):
    from test_space_correlation import _mcp
    sess = _mcp_session(tmp_path)
    f = tmp_path / "damp.uff"
    f.write_text(U.join(U.ds55(1, 50.0, {1: [1.0, 0.0, 0.0]}, damping=1.5)))
    o = _mcp(sess, [("fea_space_test_import", {"path": str(f)}, True), ("fea_space_test_import", {"path": str(f), "damping_unit": "percent"}, False)])
    assert "ratio or a percent" in o[0] and o[1]["dataset"]["modes"][0]["damping"] == pytest.approx(0.015)
    import anyio

    from fea_core.mcp.server import build_server

    async def helps():
        from mcp import Client
        async with Client(build_server(sess, groups_spec="space"), mode="legacy") as c:
            return [(await c.call_tool("fea_tool_help", {"tool": t})).content[0].text for t in ("fea_space_test_import", "fea_space_correlate")]
    imp, cor = anyio.run(helps)
    for needle in ("rotation = columns | rows", "rotation_confirmed=true ONLY when the user asked", "damping_unit", "rad/s"):
        assert needle in imp, needle
    for needle in ("UNVERIFIABLE", "OUTDATED", "PARTIAL_SCOPE", "hidden_peak_db", "listed in the result with who confirmed the set"):
        assert needle in cor, needle


def test_mcp_a_stored_result_without_the_results_is_unverifiable(fe, tmp_path):
    from test_space_correlation import _mcp
    sess = _mcp_session(tmp_path)
    sess.results = _results(fe)
    f = tmp_path / "modes.csv"
    f.write_text(C.write_modes_csv(_test_data(fe).modes))
    _mcp(sess, [("fea_space_test_import", {"path": str(f), "sensors": [s.to_dict() for s in SENSORS]}, False), ("fea_space_correlate", {}, False)])
    assert CR.verify_stored(sess.study, sess.results)["verdict"] == C.PASS
    sess.results = None
    assert CR.verify_stored(sess.study, sess.results)["verdict"] == C.VERDICT_UNVERIFIABLE


def test_the_session_test_data_do_not_make_the_study_unpicklable_or_unsaveable(fe, tmp_path):
    import copy
    import gc
    import pickle
    st, td, r, _ = _dyn_case(fe)
    assert CR.recall_test(st) is td
    pickle.dumps(st)                                                       # nothing of the session is hung on the study object
    assert copy.deepcopy(st).space.extra["correlation"]["result"]["verdict"] == C.PASS and CR.recall_test(copy.deepcopy(st)) is None
    st.save(tmp_path / "s.fea")
    st.keepalive = None
    del td
    gc.collect()
    assert CR.recall_test(st) is None                                      # the test data are gone with their owner: UNVERIFIABLE, no stale object
    key = id(st)
    del st
    gc.collect()
    assert key not in CR._SESSION_TESTS


def test_n3_the_public_beam_uff_node_1_under_the_column_convention():
    """The real OpenModal sample (10 MB, not part of the repository): skipped when it is not at hand."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / ".scratch" / "review" / "dl" / "beam.uff"
    if not p.is_file():
        pytest.skip("beam.uff is not at hand")
    td = io.load_test_file(p, "uff")
    first = td.functions[0]
    assert first.node == "1" and first.axis == pytest.approx((-0.448, 0.0, -0.894), abs=1e-3)
    assert io.load_test_file(p, "uff", rotation="rows").functions[0].axis == pytest.approx((-0.448, 0.0, 0.894), abs=1e-3)
    assert td.rotation_unconfirmed and td.unit_system["code"] == 9


def test_a_copy_of_the_study_inherits_the_session_test_data(fe):
    """The report dialog hands the report thread a COPY of the study (the user may go on editing): the measured data are recalled by the identity of the study, so the copy has to
    inherit them (`transfer_test_data`) - else the correlation chapter of the report was always UNVERIFIABLE when it came from the GUI."""
    from fea_core.study import Study
    st, td, r, _ = _dyn_case(fe)
    assert CR.verify_stored(st, r)["verdict"] == C.PASS
    cp = Study.from_dict(st.to_dict())
    assert CR.verify_stored(cp, r)["verdict"] == C.VERDICT_UNVERIFIABLE                      # what the dialog used to hand over
    assert CR.transfer_test_data(st, cp) is True and CR.recall_test(cp) is td
    assert CR.verify_stored(cp, r)["verdict"] == C.PASS
    other = Study.from_dict(st.to_dict())
    assert CR.transfer_test_data(other, Study.from_dict(st.to_dict())) is False               # nothing to hand over from a study without session data
    CR.remember_test(cp, None)
    assert CR.recall_test(cp) is None and CR.recall_test(st) is td                           # the copy is independent of the original

"""F14 S5 review fixes - the correlation core, the runner and the report: a stored verdict is never printed from the file alone (C2), curves of different quantity / frame /
normalisation / unit are never compared as equals (C3), PASS is withheld on a reduced scope (I2), unknown selections, zero-padded DOFs, a confirmed "not applicable" criterion,
and an independent sensor layout that does not come from `fe_shapes_at_sensors`."""
from __future__ import annotations

import json
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
import uff_builders as U
from test_space_correlation import _orthogonality_mass  # noqa: F401  (autouse fixture)
from test_space_correlation import EXAMPLE, FREQS, H, L, SENSORS, W, _box_mesh, _doc_text, _lim, _limits_set, _modes, _results, _study_with_limits, _test_data

from fea_core.report import space_corr as RC
from fea_core.report.word import _Doc
from fea_core.space import _corr_io as io
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from fea_core.space.model import LimitEntry
from fea_core.space.requirements import RequirementError, confirm
from fea_core.study import Study

G = 9810.0


@pytest.fixture(scope="module")
def fe():
    mesh = _box_mesh()
    return mesh, _modes(mesh)


def _limit(st, key):
    return next(x for x in st.space.requirement_set.limits if x.key == key)


def _fresh(fe, **kw):
    st = _study_with_limits(EXAMPLE)
    td = _test_data(fe)
    res, _ = CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), **kw)
    st.keepalive = td                                      # the session holds the test data (the GUI window, the MCP session): the study only keeps a weak reference to them
    return st, res


# ------------------------------------------------------------------ C2: a stored verdict is verified, never trusted
def test_a_fresh_result_is_current_and_survives_a_save(fe, tmp_path):
    st, res = _fresh(fe)
    assert res.verdict == C.PASS and res.provenance()["limits_fingerprint"] and res.test_sha256 == [] and res.fe_fingerprint
    v = CR.verify_stored(st, _results(fe))
    assert v["verdict"] == "PASS" and v["current"] and v["reasons"] == [] and v["unverified"] == []
    assert {"limits", "sensors", "setups of the measured functions", "damping table", "test data", "FE modes"} <= set(v["verified"])
    p = tmp_path / "s.fea"
    st.save(p)
    back = Study.load(p)
    v = CR.verify_stored(back, _results(fe))                                                # re-opened: the test data of the session are gone, the result cannot be verified
    assert v["verdict"] == C.VERDICT_UNVERIFIABLE and v["stored_verdict"] == "PASS" and "test data are not at hand" in v["unverified"][0]
    assert any("mass matrix (TAM) is not at hand" in x for x in v["unverified"])             # the imported matrix is session data too (the PASS of the pipeline tests comes through a TAM)
    CR.remember_tam(back, CR.recall_tam(st))                                                # the lab's matrix imported again in the new session
    assert CR.verify_stored(back, _results(fe), st.keepalive)["verdict"] == "PASS"          # the provenance is part of the saved result


def test_an_edit_of_a_limit_a_reconfirmation_or_another_scope_make_the_stored_pass_stale(fe):
    st, _ = _fresh(fe)
    _limit(st, "corr_mac_min").value = 0.99999999                                           # edited after the confirmation: the digest no longer matches
    v = CR.verify_stored(st)
    assert v["verdict"] == C.VERDICT_UNCONFIRMED and not v["current"] and "requirement set" in v["reasons"][0]
    confirm(st.space.requirement_set, "someone else", scope=st.space)                         # confirmed again, with other limits than the result was judged with
    v = CR.verify_stored(st)
    assert v["verdict"] == C.VERDICT_OUTDATED and v["stored_verdict"] == "PASS"
    st, _ = _fresh(fe)
    st.space.default_class = "composite"                                                    # the structure classes are part of the confirmation
    assert CR.verify_stored(st)["verdict"] == C.VERDICT_UNCONFIRMED
    st, _ = _fresh(fe)
    st.space.requirement_set.withdraw_confirmation()
    assert CR.verify_stored(st)["verdict"] == C.VERDICT_UNCONFIRMED


def test_other_test_data_or_other_fe_modes_make_the_stored_result_outdated(fe):
    mesh, ms = fe
    td = _test_data(fe)
    td.sha256 = ["a" * 64]
    st = _study_with_limits(EXAMPLE)
    CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS))
    other = _test_data(fe)
    other.sha256 = ["b" * 64]
    assert CR.verify_stored(st, _results(fe), td)["verdict"] == "PASS"
    assert CR.verify_stored(st, None, td)["verdict"] == C.VERDICT_UNVERIFIABLE                  # the solved results are not at hand: not a PASS
    v = CR.verify_stored(st, None, other)
    assert v["verdict"] == C.VERDICT_OUTDATED and "test data" in v["reasons"][0]
    from fea_core.eigen import ModeSet
    shifted = ModeSet("modal", "modes", [f * 1.01 for f in FREQS], ms.shapes)                  # the model was changed: other frequencies
    from fea_core.results import Results
    v = CR.verify_stored(st, Results(mesh=mesh, steps=[], modes=[shifted]))
    assert v["verdict"] == C.VERDICT_OUTDATED and "FE modes" in v["reasons"][0]
    assert CR.verify_stored(st, Results(mesh=mesh, steps=[], modes=[ms]), td)["verdict"] == "PASS"
    assert "modes" not in CR.verify_stored(st, SimpleNamespace(mesh=None, modes=[]))["verdict"].lower()                   # no modes in the results at all: not a PASS either
    assert CR.verify_stored(st, SimpleNamespace(mesh=None, modes=[]))["verdict"] == C.VERDICT_OUTDATED


def test_a_forged_or_hand_edited_record_never_prints_pass(fe):
    st = _study_with_limits(EXAMPLE, confirmed=False)
    st.space.extra["correlation"] = {"result": {"verdict": "PASS", "requirement_set_confirmed": True, "confirmed_by": "nobody", "rows": [], "limits": {}}}
    v = CR.verify_stored(st)
    assert v["verdict"] == C.VERDICT_UNCONFIRMED and "no provenance" in v["reasons"][0]
    D = _Doc()
    RC.correlation_section(D, st, "9")
    text = _doc_text(D)
    assert "Verdict of the test correlation: PASS" not in text and "UNCONFIRMED_REQUIREMENTS" in text and "not current" in text
    st2 = _study_with_limits(EXAMPLE)                                                       # a confirmed set, a record without provenance
    st2.space.extra["correlation"] = {"result": {"verdict": "PASS", "rows": []}}
    assert CR.verify_stored(st2)["verdict"] == C.VERDICT_OUTDATED
    # the verdict of a genuine record edited by hand: FAIL -> PASS
    tight = _study_with_limits({**EXAMPLE, "corr_freq_error_max": 0.001})
    td = _test_data(fe)
    for m in td.modes:
        m.frequency_hz *= 1.03
    res, _ = CR.run_correlation(tight, _results(fe), td, sensors=list(SENSORS))
    assert res.verdict == C.FAIL and CR.verify_stored(tight, _results(fe))["verdict"] == "FAIL"
    CR.state_of(tight)["result"]["verdict"] = "PASS"
    v = CR.verify_stored(tight, _results(fe))
    assert v["verdict"] == C.VERDICT_OUTDATED and "does not follow from its rows" in " ".join(v["reasons"])
    CR.state_of(tight)["result"]["verdict"] = "FAIL"
    CR.state_of(tight)["result"]["provenance"]["row_counts"]["FAIL"] = 0                      # counts edited too: still a FAIL row in the table
    CR.state_of(tight)["result"]["provenance"]["row_counts"]["PASS"] += 5
    assert CR.verify_stored(tight, _results(fe))["verdict"] != "PASS"


def test_the_report_prints_the_current_verdict_not_the_stored_one(fe):
    st, _ = _fresh(fe)
    D = _Doc()
    RC.correlation_section(D, st, "9", _results(fe))
    assert "Verdict of the test correlation: PASS" in _doc_text(D)
    _limit(st, "corr_mac_min").value = 0.5                                                  # stale now
    D = _Doc()
    RC.correlation_section(D, st, "9", _results(fe))
    t = _doc_text(D)
    assert "Verdict of the test correlation: PASS" not in t and "The result stored in the study says PASS, but it is not current" in t


# ------------------------------------------------------------------ C3: quantity, unit, frame and normalisation
def _sdof(q: float, fn: float = 100.0):
    """A one-DOF oscillator on a base: the dynamics record of a harmonic base excitation of 1 g along x and the two analytic transmissibilities."""
    f = np.linspace(5.0, 400.0, 800)
    w, wn, z = 2 * np.pi * f, 2 * np.pi * fn, 1.0 / (2.0 * q)
    h = 1.0 / (wn ** 2 - w ** 2 + 2j * z * wn * w)
    ab = np.zeros((f.size, 3), complex)
    ab[:, 0] = G
    u = np.zeros((f.size, 3), complex)
    u[:, 0] = -ab[:, 0] * h                                                                 # relative displacement [mm]
    dyn = SimpleNamespace(harmonic={"f": f, "probes": {"p": {"U": u}}, "ab": ab, "ub": -ab / (w[:, None] ** 2)}, random=None)
    res = SimpleNamespace(modes=[], mesh=None, dynamics=dyn)
    sens = C.sensors_from_dicts([{"node": "100", "point": [0.0, 0.0, 0.0], "dofs": "x"}])
    sens[0].probe = "p"
    return f, 1.0 + w ** 2 * h, w ** 2 * h, res, sens                                       # f, absolute and relative transmissibility, results, sensor


def _run1(st, res, sens, fn, setup=None):
    td = C.TestData(functions=[fn])
    return CR.run_correlation(st, res, td, sensors=sens, setups={fn.name: setup} if setup else None, store=False)[0]


@pytest.mark.parametrize("q", [2.0, 10.0])
def test_absolute_and_relative_transmissibility_agree_with_the_right_frame_and_are_flagged_without_one(q):
    f, t_abs, t_rel, res, sens = _sdof(q)
    st = _study_with_limits(EXAMPLE)
    ok = _run1(st, res, sens, C.TestFunction("T", "100", "x", f, t_abs, "frf", "g/g"), C.FunctionSetup(frame="absolute"))
    assert ok.verdict == C.PASS and ok.responses[0].rms_error < 1e-9 and abs(ok.responses[0].peak_error_db) < 1e-9 and ok.responses[0].quantity == "acceleration"
    rel = _run1(st, res, sens, C.TestFunction("T", "100", "x", f, t_rel, "frf", "g/g"), C.FunctionSetup(frame="relative"))
    assert rel.verdict == C.PASS and rel.responses[0].rms_error < 1e-9
    unset = _run1(st, res, sens, C.TestFunction("T", "100", "x", f, t_abs, "frf", "g/g"))
    assert unset.verdict == C.VERDICT_UNCONFIRMED and {r.status for r in unset.rows} == {C.UNCONFIRMED} and "not chosen" in unset.responses[0].unconfirmed[0]
    assert unset.responses[0].rms_error > 0.1                                                  # the relative curve is NOT the absolute one: that is what the choice is for
    wrong = _run1(st, res, sens, C.TestFunction("T", "100", "x", f, t_abs, "frf", "g/g"), C.FunctionSetup(frame="relative"))
    assert wrong.verdict == C.FAIL                                                              # a measurement declared relative that is absolute: compared as such and fails


def test_units_are_converted_and_a_different_quantity_or_an_unknown_unit_is_never_a_pass():
    f, t_abs, _t_rel, res, sens = _sdof(5.0)
    st = _study_with_limits(EXAMPLE)
    ab = C.FunctionSetup(frame="absolute")
    in_si = _run1(st, res, sens, C.TestFunction("A", "100", "x", f, t_abs * 9.81, "frf", "m/s^2"), ab)                    # acceleration in m/s2, the model is in g
    assert in_si.verdict == C.PASS and in_si.responses[0].rms_error < 1e-9
    in_mm = _run1(st, res, sens, C.TestFunction("A", "100", "x", f, t_abs * 9810.0, "frf", "mm/s2"), ab)
    assert in_mm.verdict == C.PASS
    # displacement: m against mm
    w = 2 * np.pi * f
    disp_abs = (res.dynamics.harmonic["probes"]["p"]["U"][:, 0] + res.dynamics.harmonic["ub"][:, 0])
    assert _run1(st, res, sens, C.TestFunction("D", "100", "x", f, disp_abs / 1000.0, "frf", "m"), ab).verdict == C.PASS
    assert _run1(st, res, sens, C.TestFunction("V", "100", "x", f, 1j * w * disp_abs / 1000.0, "frf", "m/s"), ab).verdict == C.PASS           # velocity
    for unit, why in (("NONE", "not known"), ("", "not known"), ("furlong", "not known"), ("kg", "does not fit"), ("m", "does not fit")):     # unknown, or a unit of another quantity
        r = _run1(st, res, sens, C.TestFunction("A", "100", "x", f, t_abs, "frf", unit), ab) if unit != "m" else \
            _run1(st, res, sens, C.TestFunction("A", "100", "x", f, t_abs, "frf", "m", quantity="acceleration"), ab)
        assert r.verdict == C.VERDICT_UNCONFIRMED and not r.responses and r.issues and why in r.issues[0].problems[0], (unit, r.issues)
        assert [x.status for x in r.rows] == [C.UNCONFIRMED] and r.rows[0].metric == "response comparison"
    fixed = _run1(st, res, sens, C.TestFunction("A", "100", "x", f, t_abs, "frf", "NONE"), C.FunctionSetup(frame="absolute", unit="g/g"))           # the user states the unit
    assert fixed.verdict == C.PASS


def test_a_model_curve_of_another_quantity_or_normalisation_is_refused():
    f = np.linspace(1.0, 50.0, 50)
    fn = C.TestFunction("100:y", "100", "y", f, np.ones(50), "frf", "g")
    for pred, why in ((C.Predicted(f, np.ones(50), "frf", "", "displacement", "", "none"), "is a acceleration but the model curve is a displacement"),
                      (C.Predicted(f, np.ones(50), "frf", "", "acceleration", "", "force"), "normalised 'none' but the model curve 'force'"),
                      (C.Predicted(f, np.ones(50), "psd", "", "acceleration", "", "none"), "a PSD is compared only with a PSD"),
                      (C.Predicted(f, np.ones(50)), "does not say what it is")):
        r = C.correlate(C.TestData(functions=[fn]), predicted={"100:y": pred}, limits=_lim())
        assert r.verdict == C.VERDICT_UNCONFIRMED and why in r.issues[0].problems[0]
    r = C.correlate(C.TestData(functions=[fn]), predicted={"100:y": C.Predicted(f, np.ones(50), "frf", "", "acceleration", "", "none")}, limits=_lim())
    assert r.verdict == C.PASS                                                                # identical curves of the same quantity: the numbers agree
    fn_force = C.TestFunction("100:y", "100", "y", f, np.ones(50), "frf", "g", per="force")    # the file says: per force, the unit says: not
    assert "does not fit" in C.correlate(C.TestData(functions=[fn_force]), predicted={"100:y": C.Predicted(f, np.ones(50), "frf", "", "acceleration", "", "none")},
                                         limits=_lim()).issues[0].problems[0]


def test_per_force_functions_use_the_reference_force_and_a_psd_converts_its_unit():
    f = np.linspace(5.0, 300.0, 400)
    w, wn = 2 * np.pi * f, 2 * np.pi * 80.0
    h = 1.0 / (wn ** 2 - w ** 2 + 2j * 0.02 * wn * w)                                       # receptance per unit force [mm/N] of a unit mass (t, N, mm)
    u = np.zeros((f.size, 3), complex)
    u[:, 1] = 10.0 * h                                                                      # the dynamics analysis applied a force of 10 N
    asd = 0.04 / (1 + ((f - 100) / 20) ** 2)
    psd = np.zeros((f.size, 3))
    psd[:, 1] = asd
    dyn = SimpleNamespace(harmonic={"f": f, "probes": {"p": {"U": u}}, "ab": np.zeros((f.size, 3), complex), "ub": np.zeros((f.size, 3), complex)},
                          random={"f": f, "probes": {"p": {"A_psd_g2": psd}}})
    res = SimpleNamespace(modes=[], mesh=None, dynamics=dyn)
    sens = C.sensors_from_dicts([{"node": "7", "point": [0.0, 0.0, 0.0], "dofs": "y"}])
    sens[0].probe = "p"
    st = _study_with_limits(EXAMPLE)
    r = _run1(st, res, sens, C.TestFunction("R", "7", "y", f, h, "frf", "mm/N"), C.FunctionSetup(reference_force_n=10.0))
    assert r.verdict == C.PASS and r.responses[0].rms_error < 1e-12 and r.responses[0].unit == "mm/N"
    r = _run1(st, res, sens, C.TestFunction("R", "7", "y", f, h, "frf", "mm/N"))                                          # no reference force: it cannot be predicted
    assert r.verdict != C.PASS and "reference_force_n" in r.coverage["functions_left_out"][0]["reason"]
    r = _run1(st, res, sens, C.TestFunction("R", "7", "y", f, h * 1000.0, "frf", "um/N"), C.FunctionSetup(reference_force_n=10.0))
    assert r.verdict == C.PASS
    for unit, scale in (("g^2/Hz", 1.0), ("(m/s^2)^2/Hz", 9.81 ** 2), ("(mm/s2)^2/Hz", 9810.0 ** 2)):
        r = _run1(st, res, sens, C.TestFunction("P", "7", "y", f, asd * scale, "psd", unit), C.FunctionSetup(frame="absolute"))
        assert r.verdict == C.PASS and r.responses[0].kind == "psd" and r.responses[0].peak_error_db == pytest.approx(0.0, abs=1e-9), unit
    assert C.correlate(C.TestData(functions=[C.TestFunction("P", "7", "y", f, asd, "psd", "g^2")]), predicted={}, limits=_lim()).issues[0].problems[0].count("does not fit") == 1   # g^2 is not a PSD


def test_a_controller_psd_without_node_and_direction_is_compared_through_the_explicit_map():
    f = np.arange(0.0, 400.0, 2.0)
    asd = 1e-3 + 4e-2 / (1 + ((f - 120) / 15) ** 2)
    td = io.parse_uff(U.join(U.ds58("PSD", ftype=9, node=0, direction=0, x=f, y=asd + 0j, ord_dtype=0, ord_unit="g^2/Hz", den_dtype=0, den_unit="NONE", abs_dtype=0, abs_label="Hz",
                                    abs_unit="Hz", rsp_name="Pilot 1")))
    name = td.functions[0].name
    psd = np.zeros((f.size, 3))
    psd[:, 1] = asd
    res = SimpleNamespace(modes=[], mesh=None, dynamics=SimpleNamespace(harmonic=None, random={"f": f, "probes": {"p": {"A_psd_g2": psd}}}))
    st = _study_with_limits(EXAMPLE)
    r = CR.run_correlation(st, res, td, store=False)[0]
    assert r.verdict != C.PASS and "no translational direction" in r.coverage["functions_left_out"][0]["reason"]          # nothing maps the function to the model
    CR.update_setups(st, {name: {"probe": "p", "dof": "y", "frame": "absolute"}})
    r = CR.run_correlation(st, res, td, store=False)[0]
    assert r.verdict == C.PASS and r.responses[0].rms_error < 1e-5        # the file holds six digits
    desc = CR.describe_functions(td, CR.load_setups(st))[0]
    assert desc["compared"] and desc["quantity"] == "acceleration" and desc["psd"]


def test_function_setups_are_validated_and_stored():
    st = Study(name="x", geometry="a.step")
    CR.update_setups(st, {"A": {"frame": "absolute", "per": "force", "reference_force_n": 5}, "B": {"unit": "g^2/Hz"}})
    CR.update_setups(st, {"A": {"probe": "p1"}})                                              # only the given fields change
    assert CR.load_setups(st)["A"].to_dict() == {"frame": "absolute", "per": "force", "quantity": "", "unit": "", "probe": "p1", "dof": "", "reference_force_n": 5.0}
    for bad in ({"A": {"frame": "both"}}, {"A": {"per": "x"}}, {"A": {"unit": "furlong"}}, {"A": {"dof": "rx"}}, {"A": {"reference_force_n": -1}}, {"A": {"colour": "red"}}, {"A": 3}, []):
        with pytest.raises(ValueError):
            CR.update_setups(st, bad)                                                         # type: ignore[arg-type]  # deliberately wrong types
    CR.state_of(st)["functions"]["A"]["frame"] = "<script>"                                     # damaged stored record: dropped with a note, never an exception
    problems: list[str] = []
    assert "A" not in CR.load_setups(st, problems) and problems and "A" in problems[0]


# ------------------------------------------------------------------ I2: PASS only on the whole scope
def test_a_reduced_scope_gets_a_partial_scope_verdict_with_the_coverage(fe):
    mesh, ms = fe
    st = _study_with_limits(EXAMPLE)
    full, _ = CR.run_correlation(st, _results(fe), _test_data(fe), sensors=list(SENSORS), store=False)
    assert full.verdict == C.PASS and full.coverage["complete"] and full.coverage["test_modes_used"] == 6
    part, _ = CR.run_correlation(st, _results(fe), _test_data(fe), sensors=list(SENSORS), options=C.CorrelationOptions(f_max_hz=100.0))
    assert part.verdict == C.VERDICT_PARTIAL and not part.coverage["complete"] and part.coverage["test_modes_used"] == 4
    assert [m["mode"] for m in part.coverage["test_modes_left_out"]] == [5, 6] and "outside the frequency window" in part.coverage["reasons"][0]
    assert all(r.status == C.PASS for r in part.rows) and part.provenance()["complete"] is False
    unmapped, _ = CR.run_correlation(st, _results(fe), _test_data(fe), sensors=list(SENSORS[:7]), options=C.CorrelationOptions(), store=False)
    assert unmapped.verdict == C.VERDICT_PARTIAL and "1 test node(s) have no sensor position" in unmapped.coverage["reasons"][0]
    td = _test_data(fe)
    td.modes[4].shape = {k: float(np.sin(7.0 * i)) for i, k in enumerate(td.modes[4].shape)}                  # a mode the model does not reproduce, hidden by the window
    assert CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), options=C.CorrelationOptions(), store=False)[0].verdict == C.FAIL
    assert CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), options=C.CorrelationOptions(f_max_hz=100.0), store=False)[0].verdict == C.VERDICT_PARTIAL
    f = np.linspace(5, 200, 50)
    td2 = _test_data(fe)
    td2.functions = [C.TestFunction("100:y", "100", "y", f, np.ones(50), "frf", "g/N")]                       # a function that nobody predicted
    r = CR.run_correlation(st, _results(fe), td2, sensors=list(SENSORS), options=C.CorrelationOptions(), store=False)[0]
    assert r.verdict == C.VERDICT_PARTIAL and r.coverage["functions_left_out"][0]["name"] == "100:y" and "function 100:y was not compared" in r.coverage["reasons"][0]
    assert "scope reduced" in r.text()
    cov_only = C.correlate(C.TestData(modes=td.modes), limits=_lim())                                        # no FE modes at all: nothing was correlated
    assert cov_only.verdict == C.VERDICT_UNCONFIRMED and not cov_only.coverage["complete"]


def test_the_report_prints_the_options_the_coverage_and_every_warning_that_affects_the_verdict(fe):
    st = _study_with_limits(EXAMPLE)
    td = _test_data(fe)
    f = np.linspace(5, 200, 50)
    td.functions = [C.TestFunction(f"fn{k:02d}", "100", "y", f, np.ones(50), "frf", "g/N") for k in range(30)]    # 30 functions nobody predicted: 30 reasons
    CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), options=C.CorrelationOptions(f_max_hz=100.0, pair_min_mac=0.2, fe_modes=[1, 2, 3, 4, 5, 6]))
    D = _Doc()
    warns = RC.correlation_section(D, st, "9", _results(fe))
    t = _doc_text(D)
    assert "PARTIAL_SCOPE" in t and "Verdict of the test correlation: PASS" not in t and "Left out of the correlation" in t
    for needle in ("Minimum MAC of a pair", "0.2", "Test-mode frequency window [Hz]", "100.0", "FE modes selected", "Test modes correlated", "4 of 6", "Measured functions compared", "0 of 30",
                   "2 of 6 test modes lie outside the frequency window"):
        assert needle in t, needle
    assert all(f"function fn{k:02d} was not compared" in t for k in range(30)) and sum(1 for w in warns if "was not compared" in w) == 30        # none truncated


# ------------------------------------------------------------------ I4: how the set was confirmed
def test_the_result_and_the_report_carry_confirmed_via(fe):
    for via, text in (("gui", "GUI (the user)"), ("mcp", "MCP interface (an agent)")):
        st = _study_with_limits(EXAMPLE)
        st.space.requirement_set.confirmed_via = via
        res, _ = CR.run_correlation(st, _results(fe), _test_data(fe), sensors=list(SENSORS))
        assert res.to_dict()["confirmed_via"] == via and CR.state_of(st)["result"]["confirmed_via"] == via
        D = _Doc()
        RC.correlation_section(D, st)
        assert "Confirmed via" in _doc_text(D) and text in _doc_text(D)


# ------------------------------------------------------------------ N/A limits
def _na_set(key: str = "corr_comac_min", provenance: bool = True, value: float | None = None, confirmed: bool = True):
    values = {k: v for k, v in EXAMPLE.items() if k not in (key, "corr_mac_offdiag_max")}
    sp = _limits_set(values, confirmed=False)
    sp.requirement_set.limits.append(LimitEntry(key=key, value=value, unit="-", document="doc" if provenance else "", revision="rev" if provenance else "",
                                                clause="c1" if provenance else "", not_applicable=True))
    sp.requirement_set.limits.append(LimitEntry(key="corr_mac_offdiag_max", value=None, unit="-", document="doc", revision="rev", clause="c2", not_applicable=True))
    if confirmed:
        confirm(sp.requirement_set, "test engineer", "2026-01-01", scope=sp)
    return sp


def test_a_confirmed_not_applicable_criterion_lets_the_correlation_reach_pass(fe):
    sp = _na_set()
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    assert lim.is_not_applicable("corr_comac_min") and lim.value("corr_comac_min") is None and lim.entries["corr_comac_min"].reason == "declared not applicable"
    mesh, ms = fe
    r = C.correlate(_test_data(fe), ms, mesh, SENSORS, limits=lim)
    assert r.verdict == C.PASS and {x.status for x in r.rows if x.metric in ("CoMAC", "MAC off-diagonal")} == {C.NOT_APPLICABLE}
    assert {x.status for x in r.rows} == {C.PASS, C.NOT_APPLICABLE} and r.limits.status()["corr_comac_min"]["not_applicable"]
    assert C.correlate(_test_data(fe), ms, mesh, SENSORS, limits=_lim({"corr_mac_min": 0.9, "corr_freq_error_max": 0.05})).verdict == C.VERDICT_UNCONFIRMED   # not declared: not PASS
    st = _study_with_limits({})
    st.space = sp
    assert C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp).fingerprint() != _lim().fingerprint()
    allna = C.evaluate(None, [], lim)
    assert allna == [] and C.verdict_of([C.MetricRow("x", "y", None, "", None, "", C.NOT_APPLICABLE, "k")]) == C.VERDICT_UNCONFIRMED               # nothing judged is not a PASS


def test_not_applicable_needs_provenance_confirmation_and_no_value():
    with pytest.raises(RequirementError, match="declared not applicable"):
        _na_set(provenance=False)
    sp = _na_set(confirmed=False)
    lim = C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp)
    assert not lim.is_not_applicable("corr_comac_min") and "not confirmed" in lim.entries["corr_comac_min"].reason
    sp = _na_set(value=0.9, confirmed=False)
    assert "must not carry a value" in str(next(i for i in __import__("fea_core.space.requirements", fromlist=["validate"]).validate(sp.requirement_set) if "not applicable" in str(i)))
    assert "declared not applicable but a value" in C.CorrelationLimits.from_requirement_set(sp.requirement_set, sp).entries["corr_comac_min"].reason
    plain = _limits_set(EXAMPLE)                                                              # the digest of a set without N/A entries is the one it always had
    assert "not_applicable" not in json.dumps(plain.requirement_set.content_dict())
    for x in plain.requirement_set.limits:
        assert "not_applicable" not in x.to_dict()
    back = LimitEntry.from_dict(_limit(SimpleNamespace(space=_na_set()), "corr_comac_min").to_dict())
    assert back.not_applicable and back.value is None
    d = _Doc()
    st = _study_with_limits({})
    st.space = _na_set()
    st.space.requirement_set.confirmed_via = "gui"
    mesh = _box_mesh()
    ms = _modes(mesh)
    CR.run_correlation(st, _results((mesh, ms)), _test_data((mesh, ms)), sensors=list(SENSORS))
    RC.correlation_section(d, st)
    assert "declared not applicable" in _doc_text(d) and "N/A" in _doc_text(d)


# ------------------------------------------------------------------ I6, minor: selections and plain text
def test_an_unknown_selection_is_a_clean_error_not_a_key_error(fe):
    mesh, ms = fe
    st = Study(name="s", geometry="a.step")
    with pytest.raises(ValueError, match="not a selection of the study"):
        C.sensors_from_dicts([{"node": "1", "selection": "nope"}], st)
    s = C.sensors_from_dicts([{"node": "1", "selection": "nope"}])[0]                         # without a study the selection is checked where the mesh is at hand
    with pytest.raises(ValueError, match="no surface in the mesh"):
        C.correlate(C.TestData(modes=[io.TestMode(1, 10.0, None, {("1", "x"): 1.0})]), ms, mesh, [s])
    st.selections["face"] = SimpleNamespace()                                                 # type: ignore[assignment]  # only the name is looked up
    assert C.sensors_from_dicts([{"node": "1", "selection": "face"}], st)[0].selection == "face"
    CR.save_sensors(st, [C.Sensor("1", None, "", "gone")])                                    # a stored sensor whose selection was deleted is dropped with a note
    problems: list[str] = []
    assert CR.load_sensors(st, problems) == [] and "not a selection of the study" in problems[0]


def test_sensor_names_from_a_study_file_are_plain_text():
    nasty = "<b>x</b>\n\x07" + "y" * 300
    s = C.sensors_from_dicts([{"node": "1", "probe": nasty, "point": [0, 0, 0]}])[0]
    assert "\n" not in s.probe and "\x07" not in s.probe and len(s.probe) <= 80
    st = Study(name="s", geometry="a.step")
    with pytest.raises(ValueError) as e:
        C.sensors_from_dicts([{"node": "1", "probe": nasty}], st)
    assert "\n" not in str(e.value) and "\x07" not in str(e.value) and len(str(e.value)) < 200
    r = C.sensors_from_dicts([{"node": "2", "selection": nasty, "point": None}])[0]
    assert len(r.selection) <= 80 and "\n" not in r.selection


# ------------------------------------------------------------------ I7: zero-padded DOFs
def test_a_dof_that_is_zero_in_every_mode_is_left_out_of_the_mac_with_a_note(fe):
    mesh, ms = fe
    td = _test_data(fe)
    for m in td.modes:                                                                         # the test measured only z; the file writer padded x and y with zeros
        m.shape = {k: (v if k[1] == "z" else 0.0) for k, v in m.shape.items()}
    back = io.parse_uff(io.write_uff55(td.modes))
    zero = C.correlate(back, ms, mesh, SENSORS, limits=_lim())
    assert zero.coverage["dofs_zero_excluded"] == [f"{s.node}:{d}" for s in SENSORS for d in "xy"] and zero.coverage["dofs_used"] == 8
    assert any("exactly zero in every mode" in w and "16 test DOF" in w for w in zero.warnings)
    assert zero.coverage["complete"] and len(zero.modal.pairs) == 6 and all(p.dofs == 8 for p in zero.modal.pairs)
    z_only = C.sensors_from_dicts([{**s.to_dict(), "dofs": "z"} for s in SENSORS])
    ref = C.correlate(back, ms, mesh, z_only, limits=_lim())
    assert np.allclose(zero.modal.mac, ref.modal.mac, atol=1e-12)                              # the same MAC as with z-only sensors: the zeros did not enter
    allzero = _test_data(fe)
    for m in allzero.modes:
        m.shape = {k: 0.0 for k in m.shape}
    with pytest.raises(ValueError, match="exactly zero"):
        C.correlate(allzero, ms, mesh, SENSORS, limits=_lim())


# ------------------------------------------------------------------ an independent sensor layout (the test shapes do NOT come from fe_shapes_at_sensors)
def _analytic(k: int, p: np.ndarray) -> np.ndarray:
    """The continuous fields of `_modes` at points p (n, 3), without any mesh and without the normalisation."""
    x, y, z = p[:, 0] / L, p[:, 1] / W - 0.5, p[:, 2] / H - 0.5
    psi1, psi2 = np.sin(np.pi * x / 2), np.sin(3 * np.pi * x / 2)
    zero = 0 * x
    return np.stack([(zero, psi1, zero), (zero, zero, psi1), (zero, psi2, zero), (zero, zero, psi2), (x, zero, zero), (zero, -z * x, y * x)][k], axis=1)


def _independent_test(points: np.ndarray) -> C.TestData:
    td = C.TestData(source="analytic")
    for k, f in enumerate(FREQS):
        u = _analytic(k, points)
        td.modes.append(io.TestMode(k + 1, f, None, {(str(100 + i), d): float(u[i, j]) for i in range(len(points)) for j, d in enumerate("xyz")}))
    return td


def test_independent_analytic_test_shapes_correlate_through_the_sensor_operator_and_a_wrong_operator_is_caught():
    mesh = _box_mesh(10, 4, 4)                                                                 # a finer model: the interpolation of the sin(3 pi x / 2) mode is good
    ms = _modes(mesh)
    pts = np.array([s.point for s in SENSORS])
    td = _independent_test(pts)
    r = C.correlate(td, ms, mesh, SENSORS, limits=_lim({"corr_mac_min": 0.95, "corr_freq_error_max": 0.05}))
    assert [p.test_number for p in r.modal.pairs] == [1, 2, 3, 4, 5, 6] and all(p.fe_number == p.test_number for p in r.modal.pairs)
    assert min(p.mac for p in r.modal.pairs) > 0.97 and r.modal.mac.diagonal().min() > 0.97 and max(p.mac for p in r.modal.pairs) < 1.0     # not 1: it is a different computation
    swapped = C.sensors_from_dicts([{**SENSORS[i].to_dict(), "point": SENSORS[(i + 3) % 8].point} for i in range(8)])                       # the sensors are on the wrong points
    bad = C.correlate(td, ms, mesh, swapped, limits=_lim({"corr_mac_min": 0.95}))
    assert bad.verdict == C.FAIL and min(p.mac for p in bad.modal.pairs) < 0.9
    rot = C.sensors_from_dicts([{**s.to_dict(), "dofs": "xyz"} for s in SENSORS])
    td_rot = _independent_test(pts)
    for m in td_rot.modes:                                                                     # a test whose y and z are interchanged (an axis error of the operator would look like this)
        m.shape = {(n, {"x": "x", "y": "z", "z": "y"}[d]): v for (n, d), v in m.shape.items()}
    assert min(p.mac for p in C.correlate(td_rot, ms, mesh, rot, limits=_lim()).modal.pairs) < 0.9


def test_the_sensor_operator_interpolates_nodal_values_at_nodes_centroids_and_selections(fe):
    mesh, ms = fe
    from fea_core.results import Results
    res = Results(mesh, [])
    node = int(np.argmin(np.linalg.norm(mesh.nodes - np.array([40.0, 10.0, 5.0]), axis=1)))
    p = mesh.nodes[node]
    s = C.Sensor("n", [float(p[0]), float(p[1]), float(p[2])])
    nodes, w, outside = C._sensor_operator(res, mesh, s)
    assert outside == 0.0 and w.sum() == pytest.approx(1.0) and nodes[int(np.argmax(w))] == node and w.max() == pytest.approx(1.0)
    phi, _ = C.fe_shapes_at_sensors(ms, mesh, [s], [("n", "x"), ("n", "y"), ("n", "z")])
    assert phi[:, 2] == pytest.approx(ms.shapes[2, node].astype(float))                       # at a mesh node: that node's value
    tet = np.asarray(mesh.tets[0])
    centre = mesh.nodes[tet].mean(axis=0)
    nodes, w, _ = C._sensor_operator(res, mesh, C.Sensor("c", [float(centre[0]), float(centre[1]), float(centre[2])]))
    assert sorted(nodes.tolist()) == sorted(tet.tolist()) and w == pytest.approx([0.25] * 4)
    mesh.surfaces["face"] = np.array([[0, 1, 2]])
    nodes, w, _ = C._sensor_operator(res, mesh, C.Sensor("f", None, "", "face"))
    assert sorted(nodes.tolist()) == [0, 1, 2] and w == pytest.approx([1 / 3] * 3)


# ------------------------------------------------------------------ I8: cost
def test_the_sensor_shapes_do_not_copy_the_mode_array():
    mesh = _box_mesh()
    n_nodes, n_modes = 250_000, 30
    big = SimpleNamespace(shapes=np.zeros((n_modes, n_nodes, 3), np.float32), values=list(range(1, n_modes + 1)), kind="modal")          # 90 MB, untouched pages
    big.shapes[:, :mesh.nodes.shape[0]] = 1.0
    sens = C.sensors_from_dicts([{"node": str(k), "point": [10.0 + 2.0 * k, 5.0, 5.0]} for k in range(16)])
    dofs = [(s.node, d) for s in sens for d in s.dofs]
    tracemalloc.start()
    try:
        phi, _ = C.fe_shapes_at_sensors(big, mesh, sens, dofs, list(range(0, 30, 2)))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert phi.shape == (48, 15) and np.allclose(phi, 1.0) and peak < 3 << 20, peak           # the old code copied the whole array once per DOF (90 MB x 48)


# ------------------------------------------------------------------ the MCP tools
def _mcp_session(tmp_path, fe, via: str = "mcp"):
    pytest.importorskip("mcp")
    from fea_core.mcp.session import Session
    from verification.space_cases import block_study
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    sess.study.space = _limits_set(EXAMPLE)
    sess.study.space.requirement_set.confirmed_via = via
    sess.results = _results(fe)
    f = tmp_path / "modes.csv"
    f.write_text(C.write_modes_csv(_test_data(fe).modes))
    return sess, f


def test_mcp_reports_coverage_provenance_and_how_the_set_was_confirmed_and_refuses_bad_setups_cleanly(fe, tmp_path):
    from test_space_correlation import _mcp
    sess, f = _mcp_session(tmp_path, fe)
    sensors = [s.to_dict() for s in SENSORS]
    o = _mcp(sess, [("fea_space_test_import", {"path": str(f), "sensors": sensors}, False),
                    ("fea_space_correlate", {"f_max_hz": 100.0}, False),
                    ("fea_space_correlate", {}, False),
                    ("fea_space_test_import", {"sensors": [{"node": "9", "selection": "nope"}]}, True),
                    ("fea_space_test_import", {"functions": {"A": {"frame": "both"}}}, True)])
    assert CR.verify_stored(sess.study, sess.results)["verdict"] == "PASS"
    assert o[0]["functions"] == [] and o[0]["dataset"]["sha256"] and len(o[0]["dataset"]["sha256"][0]) == 64
    part, full = o[1], o[2]
    assert part["verdict"] == "PARTIAL_SCOPE" and not part["coverage"]["complete"] and part["coverage"]["test_modes_used"] == 4 and "frequency window" in part["coverage"]["reasons"][0]
    assert full["verdict"] == "PASS" and full["confirmed_via"] == "mcp" and full["provenance"]["limits_fingerprint"] and full["provenance"]["test_sha256"] == o[0]["dataset"]["sha256"]
    assert full["provenance"]["row_counts"]["PASS"] == len(full["rows"]) and full["provenance"]["complete"] is True
    assert "not a selection of the study" in o[3] and "frame must be" in o[4]
    o = _mcp(sess, [("fea_space_test_import", {"functions": {"A": {"frame": "absolute", "unit": "g"}}}, False)])         # a setup is stated AFTER the result: the result no longer matches its inputs
    assert o[0]["dataset"] is not None and CR.load_setups(sess.study)["A"].unit == "g"
    v = CR.verify_stored(sess.study, sess.results)
    assert v["verdict"] == C.VERDICT_OUTDATED and "setups of the measured functions" in " ".join(v["reasons"])


def test_mcp_function_setups_are_listed_with_what_each_function_needs(fe, tmp_path):
    from test_space_correlation import _mcp
    sess, _f = _mcp_session(tmp_path, fe)
    f = np.linspace(5.0, 200.0, 40)
    uff = tmp_path / "f.uff"
    uff.write_text(U.join(U.ds58("H", node=100, direction=2, x=f, ord_unit="NONE"), U.ds58("PSD", ftype=9, node=0, direction=0, x=f, y=np.ones(40), ord_unit="g^2/Hz", rsp_name="Pilot", den_dtype=0, den_unit="NONE"),
                          U.ds58("coh", ftype=6, node=100, direction=2, x=f)))
    o = _mcp(sess, [("fea_space_test_import", {"path": str(uff), "auto_map": False}, False),
                    ("fea_space_test_import", {"functions": {"PSD - PSD - Pilot": {"probe": "p", "dof": "y", "frame": "absolute"}}}, False)])
    a, b, c = o[0]["functions"]
    assert not a["compared"] and "not known" in a["needs"][0] and not b["compared"] and "give them in the function setup" in b["needs"][0] and len(b["needs"]) == 1 and c["note"].startswith("imported as information only")
    assert o[1]["functions"][1]["compared"] and o[1]["functions"][1]["frame"] == "absolute" and o[1]["functions"][1]["needs"] == []


def test_mcp_damping_transfer_invalidates_the_results_like_the_dynamics_tool(fe, tmp_path):
    from test_space_correlation import _mcp
    sess, f = _mcp_session(tmp_path, fe)
    o = _mcp(sess, [("fea_space_test_import", {"path": str(f), "sensors": [s.to_dict() for s in SENSORS]}, False), ("fea_space_correlate", {"apply_damping": True}, False)])
    assert o[1]["damping_table_written"][0] == [10.0, 0.01] and "discarded" in o[1]["note_damping"] and sess.results is None
    assert sess.study.dynamics.damping.kind == "modal" and sess.study.dynamics.damping.table[0] == [10.0, 0.01]
    v = CR.verify_stored(sess.study)                                                           # the results were discarded: the stored result cannot be verified, it is no PASS
    assert v["verdict"] == C.VERDICT_UNVERIFIABLE and v["stored_verdict"] == "PASS" and "solved results are not at hand" in v["unverified"][0]


def test_mcp_can_declare_a_correlation_criterion_not_applicable_and_the_correlation_reaches_pass(fe, tmp_path):
    pytest.importorskip("mcp")
    from test_space_correlation import _mcp

    from fea_core.mcp.session import Session
    from verification.space_cases import block_study, make_requirement_set
    sess = Session(tmp_path / "mcp")
    sess.study = block_study()
    sess.results = _results(fe)
    f = tmp_path / "modes.csv"
    f.write_text(C.write_modes_csv(_test_data(fe).modes))
    ap = "protoflight"
    base = make_requirement_set(ap, confirmed=False)
    lim = [{"key": k, "value": v, "unit": "", "document": "doc", "revision": "rev", "clause": "c1"} for k, v in EXAMPLE.items() if k not in ("corr_comac_min", "corr_mac_offdiag_max")]
    lim += [{"key": "corr_comac_min", "value": 0.9, "unit": "", "document": "doc", "revision": "rev", "clause": "c1"}, {"key": "corr_mac_offdiag_max", "value": 0.3, "unit": "", "document": "doc",
                                                                                                                          "revision": "rev", "clause": "c1"}]
    data = {"name": "t", "verification_approach": ap, "factors": [x.to_dict() for x in base.factors if x.value is not None], "rules": [r.to_dict() for r in base.rules if r.factors],
            "limits": lim}
    na = {"limits": [{"key": "corr_comac_min", "value": None, "document": "doc", "revision": "rev", "clause": "c9", "not_applicable": True},
                     {"key": "corr_mac_offdiag_max", "value": None, "document": "doc", "revision": "rev", "clause": "c9", "not_applicable": True}]}
    o = _mcp(sess, [("fea_space_requirement_set", {"action": "template"}, False), ("fea_space_requirement_set", {"action": "set", "data": data}, False),
                    ("fea_space_requirement_set", {"action": "set", "data": na}, False), ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),
                    ("fea_space_test_import", {"path": str(f), "sensors": [s.to_dict() for s in SENSORS]}, False), ("fea_space_correlate", {}, False)])
    assert o[5]["verdict"] == "PASS" and o[5]["confirmed_via"] == "mcp" and o[5]["limits"]["corr_comac_min"]["not_applicable"] and o[5]["limits"]["corr_comac_min"]["clause"] == "c9"
    assert {r["status"] for r in o[5]["rows"] if r["metric"] in ("CoMAC", "MAC off-diagonal")} == {"N/A"}

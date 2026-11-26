"""Runs and stores the test correlation of a study (F14 stage S5): what the MCP tools `fea_space_test_import` / `fea_space_correlate`, the correlation dashboard of the GUI
and the report chapter share. Headless (no Qt, no `fea_core.mcp` import).

The correlation setup and the last result are kept in `Study.space.extra["correlation"]` (sensors, options, the setups of the measured functions, source name, the result
dictionary with its provenance), so they are saved with the study and printed by the report. The measured data themselves are NOT stored in the study (they are re-imported from
the test file). Nothing here confirms anything: the limits come only from the confirmed requirement set of the study (`correlation.limits_of_study`).

A stored result is never shown as current on its own word: `verify_stored` compares EVERY input of the comparison with the provenance that was stored with the result - the limits of
the study, the sensors, the per-function setups, the damping table, the test data (file and content), the FE modes and the model curves of the measured functions (the dynamics
results actually used). A difference downgrades the verdict (UNCONFIRMED_REQUIREMENTS / OUTDATED); an input that is not at hand (no solved results, no test data in this session)
cannot be checked, so the verdict is UNVERIFIABLE - never PASS. The provenance is tamper-EVIDENT, not tamper-proof: a stored result is a record of an earlier computation, to be
recomputed (Correlate) when it matters; only the row statuses are recomputed here from the stored values and the current limits.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import weakref
from typing import Any

from . import correlation as C
from .model import SpaceSettings

STATE_KEY = "correlation"
MAX_STORED_ROWS = 2000


def ensure_space(study: Any) -> SpaceSettings:
    if study.space is None:
        study.space = SpaceSettings()
    return study.space


def state_of(study: Any) -> dict[str, Any]:
    """The stored correlation record of the study ({} when there is none)."""
    sp = getattr(study, "space", None)
    raw = sp.extra.get(STATE_KEY) if sp is not None else None
    return raw if isinstance(raw, dict) else {}


def load_sensors(study: Any, problems: list[str] | None = None) -> list[C.Sensor]:
    """The stored sensors; an entry that is not valid any more (a probe / selection that was deleted, damaged data) is dropped and named in `problems`."""
    raw = state_of(study).get("sensors")
    out: list[C.Sensor] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        try:
            s = C.sensors_from_dicts([item], study)[0]
        except (ValueError, TypeError, IndexError) as e:
            if problems is not None:
                problems.append(C._plain(str(e), 160))
            continue
        if s.node in seen:
            continue
        seen.add(s.node)
        out.append(s)
    return out


def load_options(study: Any) -> C.CorrelationOptions:
    try:
        return C.CorrelationOptions.from_dict(state_of(study).get("options"))
    except (ValueError, TypeError):
        return C.CorrelationOptions()


def load_setups(study: Any, problems: list[str] | None = None) -> dict[str, C.FunctionSetup]:
    """The stored setups of the measured functions {function name: FunctionSetup}; an invalid entry is dropped and named in `problems`."""
    raw = state_of(study).get("functions")
    out: dict[str, C.FunctionSetup] = {}
    if not isinstance(raw, dict):
        return out
    for name, d in raw.items():
        try:
            out[C.clean_label(name, 90)] = C.FunctionSetup.from_dict(d)
        except (ValueError, TypeError) as e:
            if problems is not None:
                problems.append(f"setup of function {C._plain(name, 40)}: {C._plain(str(e), 120)}")
    return out


def save_sensors(study: Any, sensors: list[C.Sensor]) -> None:
    ensure_space(study).extra.setdefault(STATE_KEY, {})["sensors"] = [s.to_dict() for s in sensors]


def upsert_sensors(study: Any, new: list[C.Sensor]) -> list[C.Sensor]:
    """Adds / replaces sensors by node label and stores them; returns the full list."""
    cur = {s.node: s for s in load_sensors(study)}
    for s in new:
        cur[s.node] = s
    out = list(cur.values())
    save_sensors(study, out)
    return out


def save_setups(study: Any, setups: dict[str, C.FunctionSetup]) -> None:
    ensure_space(study).extra.setdefault(STATE_KEY, {})["functions"] = {k: v.to_dict() for k, v in setups.items()}


def update_setups(study: Any, updates: dict[str, Any]) -> dict[str, C.FunctionSetup]:
    """Merges {function name: {frame, per, quantity, unit, probe, dof, reference_force_n}} into the stored setups (only the fields that are given change); raises ValueError."""
    if not isinstance(updates, dict) or len(updates) > 5000:
        raise ValueError("functions must be an object {function name: setup}")
    cur = load_setups(study)
    for name, d in updates.items():
        if not isinstance(d, dict):
            raise ValueError(f"the setup of function {C._plain(name, 40)} must be an object")
        base = cur.get(name, C.FunctionSetup()).to_dict()
        base.update({k: v for k, v in d.items() if k in base})
        unknown = set(d) - set(base)
        if unknown:
            raise ValueError(f"function {C._plain(name, 40)}: unknown setup field(s) {', '.join(sorted(C._plain(u, 20) for u in unknown))}")
        cur[C.clean_label(name, 90)] = C.FunctionSetup.from_dict(base)
    save_setups(study, cur)
    return cur


def _json_safe(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def store_result(study: Any, result: C.CorrelationResult, sensors: list[C.Sensor], damping_table: list[list[float]] | None = None) -> None:
    """Keeps the sensors, the options and the result dictionary (without the response curves) with its provenance in the study."""
    d = _json_safe(result.to_dict(matrices=True))
    d["rows"] = d["rows"][:MAX_STORED_ROWS]
    rec = ensure_space(study).extra.setdefault(STATE_KEY, {})
    rec.update({"source": result.source, "sensors": [s.to_dict() for s in sensors], "options": result.options.to_dict(), "result": d})
    if damping_table is not None:
        rec["damping_transferred"] = [list(r) for r in damping_table]


_SESSION_TAMS: dict[int, "weakref.ref[C.MassMatrix]"] = {}                # id(study) -> the imported mass matrix (TAM) of this session (never saved)
_SESSION_TESTS: dict[int, "weakref.ref[C.TestData]"] = {}              # id(study) -> a weak reference to the test data of this session (never saved, never pickled with the study)


def remember_test(study: Any, test: C.TestData | None) -> None:
    """Keeps a weak reference to the test data of this session for the study (not on the study object, so a study can still be saved and pickled): the stored result can then be
    verified against them without the caller passing them (the report, the dashboard after a re-open of a window). The measured data themselves are not stored in the study."""
    key = id(study)
    if test is None:
        _SESSION_TESTS.pop(key, None)
        return
    if key not in _SESSION_TESTS:
        try:
            weakref.finalize(study, _SESSION_TESTS.pop, key, None)       # the entry goes with the study (an id may be reused)
        except TypeError:
            return
    _SESSION_TESTS[key] = weakref.ref(test)


def recall_test(study: Any) -> C.TestData | None:
    ref = _SESSION_TESTS.get(id(study))
    t = ref() if ref is not None else None
    return t if isinstance(t, C.TestData) else None


def transfer_test_data(src: Any, dst: Any) -> bool:
    """Lets the COPY `dst` of the study `src` (a report thread works on a copy) recall the test data - and the imported mass matrix (TAM) - of this session as `src` does; True when there were test data to hand over. The
    stored correlation can be verified only against the measured data themselves, which are never stored in the study (`remember_test`): without this a copy always gives
    UNVERIFIABLE."""
    tam = recall_tam(src)
    if tam is not None:
        remember_tam(dst, tam)                        # the imported mass matrix (TAM) is session data too: a stored TAM result cannot be verified on a copy without it
    test = recall_test(src)
    if test is None:
        return False
    remember_test(dst, test)
    return recall_test(dst) is test


def study_key(study: Any) -> str:
    """Identity of the study a TAM belongs to: SHA-256 of its name and geometry file (not of its content: editing a load must not unbind the matrix, a different model must)."""
    return hashlib.sha256(f"{getattr(study, 'name', '')}\x1f{getattr(study, 'geometry', '')}".encode("utf-8", "replace")).hexdigest()


def remember_tam(study: Any, tam: C.MassMatrix | None, mesh: Any = None) -> None:
    """Keeps a weak reference to the imported mass matrix (TAM) of this session for the study (as `remember_test`: the matrix is not stored in the study; the stored result keeps its
    fingerprint and is verified against it when the matrix is at hand). The matrix is bound to the study it is first given for (`study_key`: name and geometry file) and, when the mesh
    is at hand, to the model it was imported for (`model_key`: densities, added masses, mesh); without a mesh the model is bound at the first correlation. Another study does not use it,
    and another model holds the rows (`build_orth_mass`)."""
    key = id(study)
    if tam is None:
        _SESSION_TAMS.pop(key, None)
        return
    if not tam.study_key:
        tam.study_key = study_key(study)
    if not tam.model_key and mesh is not None:
        tam.model_key = C.model_fingerprint(study, mesh)
    if key not in _SESSION_TAMS:
        try:
            weakref.finalize(study, _SESSION_TAMS.pop, key, None)
        except TypeError:
            return
    _SESSION_TAMS[key] = weakref.ref(tam)


def recall_tam(study: Any) -> C.MassMatrix | None:
    ref = _SESSION_TAMS.get(id(study))
    t = ref() if ref is not None else None
    return t if isinstance(t, C.MassMatrix) else None


def build_orth_mass(study: Any, mesh: Any, modeset: Any, opt: C.CorrelationOptions, tam: C.MassMatrix | None = None) -> C.OrthMass:
    """The source of the reduced mass of the orthogonality checks for the option `orth_mass`: serep (computed from the FE modes and the mass of the study), tam (the imported matrix),
    none (not computed: the rows are UNCONFIRMED)."""
    if opt.orth_mass == "none":
        return C.OrthMass("none", why="the orthogonality checks were switched off (orth_mass = none)")
    if opt.orth_mass == "tam":
        t = tam or recall_tam(study)
        if t is None:
            return C.OrthMass("none", why="orth_mass is tam but no mass matrix (TAM) was imported")
        if t.study_key and t.study_key != study_key(study):
            return C.OrthMass("none", why="the imported mass matrix (TAM) was imported for another study (name / geometry): import the matrix of this model")
        now = C.model_fingerprint(study, mesh)
        if not t.model_key and now:
            t.model_key = now                                              # imported without a mesh at hand: bound at its first use
        hold = ""
        if not now:
            hold = ("the state of the model (densities, added masses, mesh) could not be fingerprinted - no mesh, or a mesh that is not tet10 - so the imported mass matrix (TAM) cannot be tied "
                    "to this model: it is not verified that it is the mass matrix of this model")
        elif t.model_key != now:
            hold = ("the imported mass matrix (TAM) was imported for another state of the model (densities, added point / surface masses or the mesh differ from those it was bound to): "
                    "it is not the mass matrix of this model - import it again for the current model")
        return C.OrthMass("tam", study=study, mesh=mesh, tam=t, model_hold=hold)
    return C.OrthMass("serep", study=study, mesh=mesh)


ROTATION_KEY = "rotation"


def adopt_rotation(study: Any, test: C.TestData) -> None:
    """The confirmation of the rotation convention that the user gave earlier for THIS file (same SHA-256, same convention) is taken over by freshly imported test data."""
    rec = state_of(study).get(ROTATION_KEY)
    if (isinstance(rec, dict) and rec.get("confirmed") is True and test.sha256 and rec.get("convention") == test.rotation and rec.get("sha256") == list(test.sha256)
            and test.rotation_ambiguous and not test.rotation_confirmed):
        test.rotation_confirmed, test.rotation_confirmed_via = True, str(rec.get("via") or "")[:12]


def confirm_rotation(study: Any, test: C.TestData, via: str, confirmed: bool = True) -> None:
    """The USER confirms (or withdraws) that the convention of the local sensor axes (rows / columns of dataset 2420) is right for this file. Until then the rows that depend on the
    rotated nodes are UNCONFIRMED. The confirmation is stored with the file's SHA-256 and the convention; `via` = gui | mcp."""
    test.rotation_confirmed, test.rotation_confirmed_via = bool(confirmed), (via if confirmed else "")
    ensure_space(study).extra.setdefault(STATE_KEY, {})[ROTATION_KEY] = {"convention": test.rotation, "confirmed": bool(confirmed), "via": via if confirmed else "",
                                                                         "sha256": list(test.sha256), "nodes": list(test.rotation_ambiguous)}


def set_rotation(study: Any, test: C.TestData, convention: str) -> None:
    """Changes the convention of the test data (columns | rows); the confirmation of the other convention is withdrawn."""
    if convention != test.rotation:
        test.set_rotation(convention)
        confirm_rotation(study, test, "", False)


def clear_state(study: Any) -> None:
    sp = getattr(study, "space", None)
    if sp is not None:
        sp.extra.pop(STATE_KEY, None)


def modal_sets(results: Any) -> list[Any]:
    return [m for m in getattr(results, "modes", []) if getattr(m, "kind", "") == "modal"]


# ------------------------------------------------------------------ the model curves of the measured functions
def auto_predicted(test: C.TestData, sensors: list[C.Sensor], results: Any, setups: dict[str, C.FunctionSetup] | None = None
                   ) -> tuple[dict[str, C.Predicted], list[str], dict[str, str]]:
    """Predicted curves for the measured functions from the dynamics results: the probe of the sensor of the function's node (or the probe named like the node, or the probe of
    the function's setup), along the measured direction, in the quantity, normalisation and frame the user's setup and the unit of the function say.
    Returns (curves, notes, {function name: why it has none})."""
    dyn = getattr(results, "dynamics", None)
    notes: list[str] = []
    out: dict[str, C.Predicted] = {}
    skipped: dict[str, str] = {}
    by_node = {s.node: s for s in sensors}
    setups = setups or {}
    for fn in test.functions:
        if fn.kind not in ("frf", "psd") or fn.excluded or fn.domain != "frequency":
            continue
        setup = setups.get(fn.name) or C.FunctionSetup()
        res = C.resolve_function(fn, setup)
        if res.problems:
            skipped[fn.name] = "; ".join(res.problems)
            continue
        if dyn is None:
            skipped[fn.name] = "the results contain no dynamics response"
            continue
        s = by_node.get(fn.node)
        probe = setup.probe or (s.probe if s is not None and s.probe else fn.node)
        dof = setup.dof or fn.dof
        axis = None if setup.dof else fn.axis
        if dof not in C.TRANSLATIONS:
            skipped[fn.name] = "the function has no translational direction: give the DOF and the probe in the function setup"
            continue
        if fn.kind == "psd":
            if res.quantity != "acceleration":
                skipped[fn.name] = "only an acceleration PSD can be predicted"
                continue
            q, frame = "random_acceleration_psd", "absolute"
        else:
            q, frame = {"displacement": "harmonic_displacement", "velocity": "harmonic_velocity", "acceleration": "harmonic_acceleration"}[res.quantity], setup.frame or "relative"
        try:
            out[fn.name] = C.predicted_from_dynamics(dyn, probe, dof, q, axis=axis, frame=frame, per=res.per, reference_force_n=setup.reference_force_n)
        except ValueError as e:
            skipped[fn.name] = str(e)
    if test.functions and dyn is None and any(f.kind in ("frf", "psd") for f in test.functions):
        notes.append("the results contain no dynamics response: the measured functions were not compared")
    notes += [f"function {n}: {why}" for n, why in skipped.items() if "no dynamics response" not in why]
    return out, notes, skipped


def run_correlation(study: Any, results: Any, test: C.TestData, *, options: C.CorrelationOptions | None = None, sensors: list[C.Sensor] | None = None, mode_set: int = 0,
                    predicted: dict[str, C.Predicted] | None = None, setups: dict[str, C.FunctionSetup] | None = None, apply_damping: bool = False,
                    store: bool = True, tam: C.MassMatrix | None = None, defer: bool = False) -> tuple[C.CorrelationResult, list[list[float]] | None]:
    """The whole correlation of a study: FE modes of `results`, sensors (default: the stored ones, plus probes named like test nodes), the setups of the measured functions
    (default: the stored ones), limits of the confirmed requirement set. Returns the result and the damping table that was written to the study (None when not asked / nothing to
    transfer). Raises ValueError with an actionable message. The caller that applied a damping table must tell the study that it changed (`MainWindow.changed()`).
    `defer=True` (a worker thread of the GUI): only COMPUTES - the study is not written (no damping transfer, nothing stored, `apply_damping` is ignored) and the result is handed to
    `finish_correlation` in the thread that owns the study."""
    opt = options or load_options(study)
    if options is None and tam is not None and opt.tam_global_axes:
        opt = dataclasses.replace(opt, tam_global_axes=False, tam_global_axes_via="")      # the stored statement was made for an earlier matrix: a new one must be stated again
    adopt_rotation(study, test)
    problems: list[str] = []
    sens = list(sensors) if sensors is not None else load_sensors(study, problems)
    known = {s.node for s in sens}
    for s in C.auto_sensors(test, study):
        if s.node not in known:
            sens.append(s)
    su = dict(setups) if setups is not None else load_setups(study, problems)
    modal = modal_sets(results)
    ms = None
    if test.modes:
        if not modal:
            raise ValueError("the results contain no natural modes - solve the modal analysis first")
        if not 0 <= mode_set < len(modal):
            raise ValueError(f"mode_set must be 0 to {len(modal) - 1}")
        ms = modal[mode_set]
    notes: list[str] = [f"stored data dropped: {p}" for p in problems]
    pred, skipped = predicted, {}
    if pred is None and test.functions:
        pred, more, skipped = auto_predicted(test, sens, results, su)
        notes += more
    mesh = getattr(results, "mesh", None)
    if tam is not None:
        remember_tam(study, tam, mesh)
    res = C.correlate(test, ms, mesh, sens, limits=C.limits_of_study(study), options=opt, predicted=pred, setups=su, skipped=skipped,
                      orth_mass=build_orth_mass(study, mesh, ms, opt, tam) if ms is not None else None)
    res.warnings = list(dict.fromkeys(res.warnings + notes))
    res.inputs["curves_source"] = "dynamics" if predicted is None else "given"
    if defer:
        return res, None
    table = _finish(study, test, res, sens, opt, apply_damping)
    if store:
        if setups is not None:
            save_setups(study, su)                                               # the setups the result was computed with are the stored ones
        store_result(study, res, sens, table)
    return res, table


def _finish(study: Any, test: C.TestData, res: C.CorrelationResult, sens: list[C.Sensor], opt: C.CorrelationOptions, apply_damping: bool) -> list[list[float]] | None:
    """The part of a correlation that writes to the study: the damping transfer, the fingerprint of the damping table the result goes with, the weak reference to the test data."""
    table: list[list[float]] | None = None
    if apply_damping:
        if res.modal is None:
            raise ValueError("there are no paired modes to take the damping from")
        table = C.transfer_damping(study, res.modal.pairs, opt.damping_at)
    res.inputs["damping_fingerprint"] = C.damping_fingerprint(study)             # after the transfer: the table the stored result goes with
    remember_test(study, test)
    return table


def finish_correlation(study: Any, test: C.TestData, res: C.CorrelationResult, *, apply_damping: bool = False) -> list[list[float]] | None:
    """Completes a result of `run_correlation(defer=True)` in the thread that owns the study: damping transfer, provenance of the damping table, storing the result. Returns the damping
    table that was written (None when not asked)."""
    sens = _effective_sensors(study, test)
    table = _finish(study, test, res, sens, res.options, apply_damping)
    store_result(study, res, sens, table)
    return table


# ------------------------------------------------------------------ is the stored result still the result of this study?
def _verdict_from_counts(counts: dict[str, int], complete: bool) -> str:
    judged = sum(v for k, v in counts.items() if k != C.NOT_APPLICABLE)
    if counts.get(C.FAIL, 0):
        return C.FAIL
    if not judged or counts.get(C.UNCONFIRMED, 0):
        return C.VERDICT_UNCONFIRMED
    return C.PASS if complete else C.VERDICT_PARTIAL


def _counts_of(v: Any) -> dict[str, int] | None:
    """The row counts of a stored provenance, or None when they are damaged (not a mapping of non-negative whole numbers)."""
    if not isinstance(v, dict):
        return None
    out: dict[str, int] = {}
    for k, n in v.items():
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 or n != int(n) or n > 10**9:
            return None
        out[str(k)] = int(n)
    return out


def _str_list(v: Any) -> list[str] | None:
    return [x for x in v] if isinstance(v, list) and all(isinstance(x, str) for x in v) else None


_ABSOLUTE_KEYS = ("corr_freq_error_max", "corr_peak_freq_shift_max", "corr_peak_amp_error_max_db")


def _recompute_rows(rows: Any, lim: C.CorrelationLimits) -> list[str]:
    """The row statuses recomputed from the stored values and the CURRENT limits (a status that was edited by hand, or that no longer follows from its value, is reported)."""
    out: list[str] = []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict) or r.get("status") not in (C.PASS, C.FAIL):
            continue
        key = r.get("limit_key")
        v = r.get("value")
        if key not in C.LIMIT_INFO or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        limit = lim.value(str(key))
        if limit is None:
            continue
        x = abs(v) if key in _ABSOLUTE_KEYS else v
        expect = C.PASS if (x >= limit if C.LIMIT_INFO[str(key)][1] == ">=" else x <= limit) else C.FAIL
        if expect != r["status"]:
            out.append(f"row {C._plain(r.get('item'), 50)} / {C._plain(r.get('metric'), 30)}: the stored status {r['status']} does not follow from its value against the current limit ({expect})")
        if len(out) >= 5:
            break
    return out


def _effective_sensors(study: Any, test: Any) -> list[C.Sensor]:
    """The sensors of a correlation as `run_correlation` takes them: the stored ones plus probes named like test nodes."""
    sens = load_sensors(study)
    known = {s.node for s in sens}
    for s in C.auto_sensors(test, study):
        if s.node not in known:
            sens.append(s)
    return sens


def verify_stored(study: Any, results: Any = None, test: Any = None) -> dict[str, Any] | None:
    """The stored correlation result with its CURRENT verdict, or None when there is none. The verdict is the stored one only while the stored provenance still holds for EVERY input:
      * the limits of the study (requirement-set content, confirmation, structure classes), the sensors, the per-function setups and the damping table of the study;
      * the test data (SHA-256 of the file(s) and a fingerprint of everything read, rotation convention included) - from `test`, or the test data of this session;
      * the FE modes (frequencies, mesh) and the model curves of the measured functions (the dynamics results that were used) - from `results`;
      * the stored verdict follows from its row counts, its coverage and its rows (recomputed from the stored values against the current limits).
    A difference gives UNCONFIRMED_REQUIREMENTS (the requirement set is not confirmed now) or OUTDATED. An input that is not at hand (no solved results, no test data) cannot be
    checked: the verdict is then UNVERIFIABLE. Never PASS from the file alone. The dictionary: verdict, stored_verdict, current, reasons (what differs), unverified (what could not be
    checked), why (both), verified (what was checked and holds), result, limits."""
    rec = state_of(study)
    res = rec.get("result")
    if not isinstance(res, dict) or not res.get("verdict"):
        return None
    stored = C._plain(res["verdict"], 40)
    lim = C.limits_of_study(study)
    reasons: list[str] = []
    unverified: list[str] = []
    verified: list[str] = []
    if test is None:
        test = recall_test(study)
    try:
        _check_provenance(study, rec, res, lim, results, test, stored, reasons, unverified, verified)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, IndexError) as e:
        reasons.append(f"the stored result is damaged and cannot be verified ({e.__class__.__name__})")
    if not reasons and not unverified:
        verdict = stored
    elif not lim.set_confirmed:
        verdict = C.VERDICT_UNCONFIRMED
    elif reasons:
        verdict = C.VERDICT_OUTDATED
    else:
        verdict = C.VERDICT_UNVERIFIABLE
    return {"verdict": verdict, "stored_verdict": stored, "current": not reasons and not unverified, "reasons": reasons, "unverified": unverified, "why": reasons + unverified,
            "verified": verified, "result": res, "limits": lim}


def _check_provenance(study: Any, rec: dict[str, Any], res: dict[str, Any], lim: C.CorrelationLimits, results: Any, test: Any, stored: str, reasons: list[str],
                      unverified: list[str], verified: list[str]) -> None:
    prov = res.get("provenance")
    if not isinstance(prov, dict) or not isinstance(prov.get("limits_fingerprint"), str) or not prov["limits_fingerprint"]:
        reasons.append("the stored result carries no provenance (written by hand or by an older version): it cannot be verified")
        return
    if "orth_mass" not in prov:
        reasons.append("the result was stored before the cross-orthogonality (COC / AOC) and the modulus difference existed: enter the four new limit rows (corr_coc_diag_min, corr_coc_offdiag_max, "
                       "corr_aoc_offdiag_max, corr_modulus_diff_max) of the requirement set (or declare them not applicable), confirm it and correlate again")
    if prov["limits_fingerprint"] != lim.fingerprint():
        reasons.append("the requirement set (limits, confirmation or structure classes) is no longer the one the result was judged with")
    else:
        verified.append("limits")
    counts = _counts_of(prov.get("row_counts"))
    if counts is None:
        reasons.append("the stored result has no (or damaged) row counts")
    else:
        expect = _verdict_from_counts(counts, prov.get("complete") is True)
        if expect != stored:
            reasons.append(f"the stored verdict {stored} does not follow from its rows ({expect})")
        rows = res.get("rows")
        total = prov.get("rows_total")
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows) or isinstance(total, bool) or not isinstance(total, int) or total < len(rows):
            reasons.append("the rows of the stored result are damaged")
        else:
            if len(rows) != min(total, MAX_STORED_ROWS):
                reasons.append("rows of the stored result are missing")
            elif len(rows) == total and {k: v for k, v in counts.items() if v} != {k: sum(1 for r in rows if r.get("status") == k) for k in counts if any(r.get("status") == k for r in rows)}:
                reasons.append("the rows of the stored result do not match its counts")
            if stored == C.PASS and any(r.get("status") in (C.FAIL, C.UNCONFIRMED) for r in rows):
                reasons.append("the stored verdict PASS contradicts a FAIL / UNCONFIRMED row")
            reasons += _recompute_rows(rows, lim)
            om = prov.get("orth_mass")
            if isinstance(om, dict) and om.get("source") == "serep":
                if any(r.get("status") in (C.PASS, C.FAIL) and str(r.get("metric", "")).startswith(("COC", "AOC")) for r in rows):
                    reasons.append("a COC / AOC row of the SEREP route is stored as PASS / FAIL: those rows are indicative and UNCONFIRMED (a stored result of an earlier build): correlate again")
                if om.get("indicative") is False:
                    reasons.append("the stored record of the SEREP route declares itself not indicative: the program never writes that (the SEREP COC / AOC is always indicative), the record was altered "
                                   "or comes from an earlier build: correlate again")
    rot = prov.get("rotation", {})
    if not isinstance(rot, dict) or not isinstance(prov.get("curves_source", ""), str):
        reasons.append("the stored provenance is damaged (rotation)")
    # inputs that live in the study
    for key, label, now in (("sensors_fingerprint", "sensors", lambda: C.sensors_fingerprint(load_sensors(study))),
                            ("setups_fingerprint", "setups of the measured functions", lambda: C.setups_fingerprint(load_setups(study))),
                            ("damping_fingerprint", "damping table", lambda: C.damping_fingerprint(study))):
        was = prov.get(key)
        if not isinstance(was, str) or not was:
            reasons.append(f"the stored result carries no fingerprint of its {label} (written by an older version): it cannot be verified")
        elif was != now():
            reasons.append(f"the {label} of the study changed after the result was computed")
        else:
            verified.append(label)
    # the test data
    shas = _str_list(prov.get("test_sha256"))
    tfp = prov.get("test_fingerprint")
    if shas is None or not isinstance(tfp, str) or not tfp:
        reasons.append("the stored result carries no fingerprint of the test data (written by an older version): it cannot be verified")
    elif test is None:
        unverified.append("the test data are not at hand (they are not stored in the study): import the test file again")
    else:
        if shas and set(shas) != set(getattr(test, "sha256", []) or []):
            reasons.append("the imported test data are not the file(s) the result was computed from")
        elif test.fingerprint() != tfp:
            reasons.append("the imported test data (content, units, rotation convention) are not the ones the result was computed from")
        else:
            verified.append("test data")
        rot = prov.get("rotation")
        if isinstance(rot, dict) and rot.get("confirmed") is True and test.rotation_ambiguous and not test.rotation_confirmed:
            reasons.append("the convention of the local sensor axes was confirmed when the result was computed and is not confirmed now")
    # the FE modes and the model curves (they come from the solved results)
    fe_fp = prov.get("fe_fingerprint", "")
    if not isinstance(fe_fp, str):
        reasons.append("the stored provenance is damaged (fe_fingerprint)")
        fe_fp = ""
    curves = prov.get("curves", {})
    if not isinstance(curves, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in curves.items()):
        reasons.append("the stored provenance is damaged (curves)")
        curves = {}
    if fe_fp or curves:
        if results is None:
            unverified.append("the solved results are not at hand (the FE modes and the dynamics responses the result was computed from): solve and correlate again to verify")
        else:
            if fe_fp:
                mesh = getattr(results, "mesh", None)
                matched = next((m for m in modal_sets(results) if C.fe_fingerprint(m, mesh) == fe_fp), None)
                if matched is None:
                    reasons.append("the FE modes (frequencies or mesh) are no longer the ones the result was computed from")
                else:
                    verified.append("FE modes")
                    _check_orth_mass(study, prov, mesh, matched, reasons, unverified, verified)
            if curves:
                if prov.get("curves_source") != "dynamics":
                    unverified.append("the model curves were supplied by the caller of the correlation and cannot be recomputed from the study")
                elif test is None:
                    unverified.append("the test data are not at hand: the model curves of the measured functions cannot be recomputed")
                else:
                    pred, _notes, _skipped = auto_predicted(test, _effective_sensors(study, test), results, load_setups(study))
                    changed = [n for n, h in curves.items() if n not in pred or C.curve_fingerprint(pred[n]) != h]
                    if changed:
                        reasons.append("the model curve of " + ", ".join(C._plain(n, 40) for n in changed[:4]) + (" ..." if len(changed) > 4 else "")
                                       + " is no longer the one the result was computed from (the dynamics results changed: damping, excitation, probes or mesh)")
                    else:
                        verified.append("dynamics responses")


def _check_orth_mass(study: Any, prov: dict[str, Any], mesh: Any, modeset: Any, reasons: list[str], unverified: list[str], verified: list[str]) -> None:
    """The reduced mass of the orthogonality checks of a stored result: the SEREP mass (densities and added masses of the study) or the imported TAM matrix must be the one the numbers
    were computed with. A result that has no computed orthogonality has nothing to check."""
    om = prov.get("orth_mass")
    if om is None:
        return
    if not isinstance(om, dict) or not isinstance(om.get("source", ""), str) or not isinstance(om.get("fingerprint", ""), str):
        reasons.append("the stored provenance is damaged (orth_mass)")
        return
    if om.get("computed") is not True:
        return
    src, was = om["source"], om["fingerprint"]
    if src == "tam":
        mk = om.get("model_fingerprint")
        if mk is None:
            reasons.append("the stored result does not say which state of the model (densities, added masses, mesh) its imported mass matrix (TAM) was bound to (a stored result of an earlier build): "
                           "correlate again")
        elif not isinstance(mk, str):
            reasons.append("the stored provenance is damaged (orth_mass model_fingerprint)")
        elif mk:
            try:
                now = C.model_fingerprint(study, mesh)
            except (ValueError, KeyError, AttributeError, TypeError, IndexError):
                now = ""
            if not now:
                unverified.append("the state of the model (densities, added masses, mesh) could not be fingerprinted now: it is not verified that the imported mass matrix (TAM) is still its mass matrix")
            elif now != mk:
                reasons.append("the model (densities, added point / surface masses or the mesh) is no longer the one the imported mass matrix (TAM) was bound to: it is not the mass matrix of this model "
                               "any more - import it again for the current model and correlate again")
            else:
                verified.append("model state of the mass matrix (TAM)")
        tam = recall_tam(study)
        if tam is None:
            unverified.append("the imported mass matrix (TAM) is not at hand (it is not stored in the study): import it again to verify the cross-orthogonality")
        elif tam.fingerprint() != was:
            reasons.append("the imported mass matrix (TAM) is not the one the cross-orthogonality was computed with")
        else:
            verified.append("mass matrix (TAM)")
        return
    if src == "serep":
        try:
            now = build_orth_mass(study, mesh, modeset, C.CorrelationOptions()).fingerprint()
        except (ValueError, KeyError, AttributeError, TypeError):
            now = ""
        if not now:
            unverified.append("the mass of the SEREP reduction could not be recomputed from the study")
        elif now != was:
            reasons.append("the mass of the study (densities, added masses) is no longer the one the cross-orthogonality was computed with")
        else:
            verified.append("mass of the orthogonality checks")
        return
    reasons.append("the stored provenance is damaged (orth_mass source)")


def describe_functions(test: C.TestData, setups: dict[str, C.FunctionSetup] | None = None) -> list[dict[str, Any]]:
    """What each measured function is and what it still needs before it can be compared (for the import tool and the dashboard)."""
    setups = setups or {}
    out: list[dict[str, Any]] = []
    for fn in test.functions:
        row: dict[str, Any] = {"name": fn.name, "kind": fn.kind, "node": fn.node, "dof": fn.dof, "unit": fn.unit, "domain": fn.domain, "points": int(fn.freq.size), "compared": False}
        if fn.kind not in ("frf", "psd"):
            row["note"] = "imported as information only (only FRF types 4 / 5 and PSD types 2 / 9 are compared)"
        elif fn.excluded:
            row["note"] = fn.excluded
        elif fn.domain != "frequency":
            row["note"] = "not a function of frequency"
        else:
            su = setups.get(fn.name) or C.FunctionSetup()
            res = C.resolve_function(fn, su)
            needs = list(res.problems)
            if not (su.dof or fn.dof) and not fn.axis:
                needs.append("the direction (dof) and the probe are not in the file: give them in the function setup")
            row.update({"quantity": res.quantity, "per": res.per, "psd": res.psd, "frame": su.frame, "needs": needs, "compared": not needs})
            if not su.frame:
                row["frame_note"] = "choose frame absolute | relative when the dynamics analysis has a base excitation (else the rows stay UNCONFIRMED)"
        out.append(row)
    return out

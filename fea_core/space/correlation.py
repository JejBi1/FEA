"""Vibration-test correlation of the finite element model (F14 stage S5, the correlation criteria of ECSS-E-ST-32-11C).

Compares the natural modes and the responses of the FE model with a modal / vibration test:
  importers       bounded CSV and UFF 58 / 55 readers (`_corr_io`, re-exported here): `load_test_file`, `parse_csv`, `parse_uff`, `TestData`
  mode metrics    MAC matrix, mode pairing (Hungarian assignment of max-MAC, or max-MAC per test mode), CoMAC per sensor DOF, relative frequency error
  damping         measured modal damping ratios -> `Study.dynamics.damping.table` (`transfer_damping`)
  responses       peak frequency shift, peak amplitude error [dB] and RMS error of a predicted against a measured FRF / ASD (`compare_response`), only between curves of the SAME
                  quantity (displacement / velocity / acceleration), the same frame (absolute / relative to the base), the same normalisation (per force / per base acceleration)
                  and a known, convertible unit (`resolve_function`); anything else is UNCONFIRMED, never PASS
  verdict         every metric is judged ONLY against a limit of the user-confirmed requirement set (`CorrelationLimits.from_requirement_set`)

No number of a standard is built in (F14 plan, section 1): the thresholds are the entries `corr_*` of `RequirementSet.limits` (names and meaning in `LIMIT_INFO`, empty in the
template). A limit that is missing, has no document / revision / clause, or belongs to a requirement set that is not confirmed (or was edited after the confirmation) is
unusable: its rows are `UNCONFIRMED` and the verdict is never PASS. A criterion that the user declared "not applicable" (confirmed, with document / revision / clause) is shown as
N/A and does not hold the verdict back (every such criterion is listed in the result with who confirmed the set and through which interface the flag was entered). A test mode that
no FE mode can be paired with is a FAIL without any limit (the model lacks the mode). A correlation of a reduced scope (test modes outside the frequency window, test nodes without a
sensor, measured DOFs that the sensors do not map, functions that were not compared, test resonances outside the compared band) never gets PASS: its verdict is `PARTIAL_SCOPE` and the
coverage lists what was left out (`CorrelationResult.coverage`). Data at rotated sensors whose local-axes convention (UFF 2420, rows | columns) the user has not confirmed are UNCONFIRMED.
A stored result keeps the fingerprints of all its inputs (`CorrelationResult.provenance`) and is verified against them by `corr_run.verify_stored` (OUTDATED / UNVERIFIABLE).

Definitions (shapes at the sensor DOFs; test shapes may be complex, FE shapes are real). Clauses = ECSS-E-ST-32-11C (31 July 2008), checked against the document:
  MAC(i, j)       |phi_t,i^H phi_a,j|^2 / ((phi_t,i^H phi_t,i)(phi_a,j^H phi_a,j)) over the DOFs measured in test mode i (a DOF that was not measured in a mode drops out of that mode);
                  the definition of 3.2.26 (the standard writes the transpose of real shapes; complex test shapes use the Hermitian form); a DOF that is exactly zero in every test mode
                  (an unmeasured axis padded with zeros by a file writer) is left out of all modes, with a note. MAC is a mandatory technique (Table 5-2, 5.8.1.1b)
  frequency error (f_FE - f_test) / f_test of a pair: the "frequency difference" of Table 5-2 (mandatory); the standard does not define its denominator, the test is the reference
  CoMAC(d)        (sum_j |psi_t,dj psi_a,dj|)^2 / (sum_j |psi_t,dj|^2 sum_j |psi_a,dj|^2) over the paired modes j (3.2.7 and its note: meaningful only for matched pairs), psi = the shape
                  normalised to unit length on its DOFs (Lieven and Ewins). Deviation: 3.2.7 writes the sum over the shape coefficients as given, so that the value depends on how the two
                  mode sets are scaled; the normalisation removes that dependence. Needs at least two pairs
  off-diagonal    the largest MAC between a paired test mode and the FE mode of another pair (and the other way round): the MAC form of the cross terms of the pairs
  COC, AOC        the mass-weighted cross-orthogonality of 3.2.39 (|phi_t^T M phi_FE| / sqrt((phi_t^T M phi_t)(phi_FE^T M phi_FE)), NOT squared: the standard's formula has no square) with the mass
                  reduced to the sensor DOFs (5.8.1.3 b), and the auto-orthogonality of the test modes (5.7.1.4), by `corr_orth` (the reduced mass: an imported TAM matrix, or a SEREP of the FE modes
                  computed from the mass of the study; see there). Orthogonality / cross-orthogonality is MANDATORY (Table 5-2, 5.8.1.1 b); without a reduced mass its rows are UNCONFIRMED.
                  The pairing stays the MAC pairing (the compared FE modes). ONLY a TAM can PASS or FAIL these rows: the SEREP COC / AOC are INDICATIVE (the COC of the projection of a test shape
                  onto the compared FE modes, owner decision 25. 11. 2026, reasons in `corr_orth`): the number is shown, the row is UNCONFIRMED whatever its value, a value outside the limit is a
                  warning hint and never a FAIL
  modulus diff.   the "modulus difference" of Table 5-2 (mandatory): the standard names it and gives no formula; the program's OWN definition is in `corr_orth`
  criteria        Table 5-3 of the standard gives example criteria (not normative: its note a) that differ by class of mode (fundamental bending modes, modes with an effective mass of more than
                  10 % of the total mass, other modes in the relevant range, up to 100 Hz in general). The program has ONE limit per metric for all paired modes, entered by the user; a
                  classification of the modes by effective mass is not implemented
  peak shift      (f_peak,FE - f_peak,test) / f_peak,test;  peak error  20 log10(A_FE / A_test) at the peaks (10 log10 for a PSD);  RMS error  rms(A_FE - A_test) / rms(A_test)
                  on the test frequencies of the common band (the FE curve interpolated linearly)
Limitations: shapes are compared in the global x / y / z directions (the test importer rotates local sensor axes into the global system when their system is fully defined),
rotational DOFs of the test are never compared (a rotational DOF with data makes the scope partial), a sensor is a point of the mesh (interpolated with the element shape functions) and
measures the displacement of that point.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np

from ._corr_io import (DAMPING_UNITS, DEFAULT_ROTATION, DOF_NAMES, FUNCTION_KINDS, MAX_FILE_BYTES, MAX_LABEL, ROTATIONS, TRANSLATIONS, TestData, TestDataError, TestFunction, TestMode,
                       clean_label, decode_bytes, load_test_file, parse_csv, parse_dof, parse_uff, write_functions_csv, write_modes_csv, write_uff55, write_uff58)
from ._corr_units import QUANTITY_DIMS, Unit, fe_unit, parse_unit, unit_hint
from .corr_orth import (INDICATIVE_REASON, MAX_MASS_BYTES, ORTH_MASS, MassMatrix, OrthMass, OrthResult, load_mass_file, mass_model_fingerprint, model_fingerprint, modulus_difference,
                        orthogonality, parse_mass_csv)

__all__ = ["TestData", "TestDataError", "TestFunction", "TestMode", "load_test_file", "parse_csv", "parse_uff", "write_uff55", "write_uff58", "write_modes_csv",
           "write_functions_csv", "clean_label", "decode_bytes", "parse_dof", "DOF_NAMES", "FUNCTION_KINDS", "MAX_FILE_BYTES", "MAX_LABEL", "LIMIT_INFO", "CorrelationLimits", "CorrelationOptions",
           "Sensor", "Predicted", "ModePair", "ModalCorrelation", "ResponseMetrics", "MetricRow", "CorrelationResult", "ResponseIssue", "FunctionSetup", "Resolved", "mac_matrix", "pair_modes", "comac",
           "modal_correlation", "compare_response", "correlate", "fe_shapes_at_sensors", "fe_test_data", "damping_table", "transfer_damping", "modal_frf", "predicted_from_dynamics",
           "install_limit_rows", "limits_of_study", "sensors_from_dicts", "auto_sensors", "resolve_function", "fe_fingerprint", "PASS", "FAIL", "UNCONFIRMED", "NOT_APPLICABLE",
           "VERDICT_UNCONFIRMED", "VERDICT_PARTIAL", "VERDICT_OUTDATED", "VERDICT_UNVERIFIABLE", "FRAMES", "PERS", "sensors_fingerprint", "setups_fingerprint", "damping_fingerprint",
           "curve_fingerprint", "not_applicable_limits", "not_applicable_text", "ROTATIONS", "DEFAULT_ROTATION", "DAMPING_UNITS", "LIMIT_HINTS", "ORTH_MASS", "MAX_MASS_BYTES", "MassMatrix", "OrthMass", "OrthResult",
           "load_mass_file", "parse_mass_csv", "mass_model_fingerprint", "model_fingerprint", "modulus_difference", "orthogonality", "INDICATIVE_REASON"]

PASS, FAIL, UNCONFIRMED, NOT_APPLICABLE = "PASS", "FAIL", "UNCONFIRMED", "N/A"
VERDICT_UNCONFIRMED = "UNCONFIRMED_REQUIREMENTS"
VERDICT_PARTIAL = "PARTIAL_SCOPE"                       # every judged row passed, but part of the test data was left out
VERDICT_OUTDATED = "OUTDATED"                           # a stored result whose limits / test data / FE modes / sensors / setups / dynamics curves are no longer the ones it was computed with
VERDICT_UNVERIFIABLE = "UNVERIFIABLE"                   # a stored result that cannot be checked now: the solved results or the test data it was computed from are not at hand
METHODS = ("hungarian", "max_mac", "order")
FRAMES = ("absolute", "relative")
PERS = ("none", "base_acceleration", "force")
TEMPLATE_DOCUMENT = "ECSS-E-ST-32-11C"
CLAUSE_PLACEHOLDER = "<clause>"
MAX_MATRIX_IN_DICT = 10_000                  # cells of the MAC matrix stored in a result dictionary
G_MM = 9810.0                                # mm/s2 of 1 g (the program's g)

# name -> (unit, relation of the metric to the limit, meaning). No value anywhere in the program: the user enters them in the requirement set.
LIMIT_INFO: dict[str, tuple[str, str, str]] = {
    "corr_mac_min": ("-", ">=", "smallest MAC (diagonal) of a paired mode"),
    "corr_mac_offdiag_max": ("-", "<=", "largest MAC between a paired mode and the partner of another pair (off-diagonal of the pairs)"),
    "corr_comac_min": ("-", ">=", "smallest CoMAC of a sensor DOF"),
    "corr_coc_diag_min": ("-", ">=", "smallest diagonal term of the mass-weighted cross-orthogonality COC of a paired mode"),
    "corr_coc_offdiag_max": ("-", "<=", "largest off-diagonal term of the cross-orthogonality COC between a paired test mode and the FE mode of another pair"),
    "corr_aoc_offdiag_max": ("-", "<=", "largest off-diagonal term of the auto-orthogonality AOC of a test mode with another test mode (5.7.1.4)"),
    "corr_modulus_diff_max": ("-", "<=", "largest modulus difference of a paired mode (fraction of the largest test amplitude; the program's own definition, not the standard's, see corr_orth)"),
    "corr_freq_error_max": ("-", "<=", "largest |f_FE - f_test| / f_test of a paired mode (a fraction: 0.05 = 5 %)"),
    "corr_peak_freq_shift_max": ("-", "<=", "largest |f_peak,FE - f_peak,test| / f_peak,test of a response curve (a fraction)"),
    "corr_peak_amp_error_max_db": ("dB", "<=", "largest |amplitude error| at the response peak [dB]"),
    "corr_rms_error_max": ("-", "<=", "largest relative RMS error of a response curve in the common band (a fraction)"),
}

# Hints only (shown beside the limit, never used as a value): what ECSS-E-ST-32-11C says about the orthogonality limits. Table 5-3 is not normative (its note a).
LIMIT_HINTS: dict[str, str] = {
    "corr_coc_diag_min": "example of Table 5-3 (not normative): diagonal terms > 0,90",
    "corr_coc_offdiag_max": "example of Table 5-3 (not normative): off-diagonal terms < 0,10",
    "corr_aoc_offdiag_max": "5.7.1.4 b (test evaluation): off-diagonal elements less than 10 %",
    "corr_modulus_diff_max": "Table 5-2 makes the modulus difference mandatory; the standard gives no formula and no limit",
}


def _plain(x: Any, limit: int = 80) -> str:
    """A string of a study file reduced to printable plain text of at most `limit` characters (it ends in message boxes, tables and the report)."""
    return "".join(ch if ch.isprintable() else " " for ch in str(x if x is not None else "")).strip()[:limit]


# ------------------------------------------------------------------ limits (only from the confirmed requirement set)
@dataclass
class UsedLimit:
    value: float | None = None
    document: str = ""
    revision: str = ""
    clause: str = ""
    usable: bool = False
    reason: str = "not entered"
    not_applicable: bool = False         # the user declared the criterion not applicable (confirmed, with document / revision / clause)
    set_via: str = ""                    # through which interface the not-applicable flag was entered (gui | mcp | "" = unknown, e.g. an imported file)


@dataclass
class CorrelationLimits:
    """The thresholds of the correlation: usable only when they come from the confirmed requirement set, with their provenance."""
    entries: dict[str, UsedLimit] = field(default_factory=lambda: {k: UsedLimit() for k in LIMIT_INFO})
    set_confirmed: bool = False
    set_name: str = ""
    confirmed_by: str = ""
    date: str = ""
    approach: str = ""
    confirmed_via: str = ""
    requirement_digest: str = ""         # content digest of the requirement set the limits were read from
    scope_digest: str = ""               # structure classes of the parts at that time

    @classmethod
    def from_requirement_set(cls, rs: Any, space: Any = None) -> "CorrelationLimits":
        out = cls()
        if rs is None:
            for e in out.entries.values():
                e.reason = "no requirement set"
            return out
        confirmed = bool(space.requirement_confirmed()) if space is not None else bool(rs.is_confirmed())
        out.set_confirmed, out.set_name, out.confirmed_by, out.date, out.approach = confirmed, rs.name, rs.confirmed_by, rs.date, rs.verification_approach
        out.confirmed_via = str(getattr(rs, "confirmed_via", "") or "")
        out.requirement_digest = rs.content_digest()
        out.scope_digest = space.scope_digest() if space is not None else str(getattr(rs, "scope_digest", ""))
        seen: set[str] = set()
        for x in rs.limits:
            if x.key not in LIMIT_INFO:
                continue
            e = out.entries[x.key]
            if x.key in seen:
                e.value, e.usable, e.not_applicable, e.reason = None, False, False, "the limit appears more than once in the requirement set"
                continue
            seen.add(x.key)
            e.document, e.revision, e.clause = x.document, x.revision, x.clause
            na = bool(getattr(x, "not_applicable", False))
            if na:
                if x.value is not None:
                    e.reason = "declared not applicable but a value is entered"
                elif not (x.document.strip() and x.revision.strip() and x.clause.strip() and x.clause.strip() != CLAUSE_PLACEHOLDER):
                    e.reason = "declared not applicable: document / revision / clause are not filled"
                elif not confirmed:
                    e.reason = "declared not applicable, but the requirement set is not confirmed"
                else:
                    e.not_applicable, e.reason = True, "declared not applicable"
                    e.set_via = str(getattr(x, "set_via", "") or "")
                continue
            if x.value is None or not math.isfinite(x.value):
                e.reason = "not entered"
            elif not (x.document.strip() and x.revision.strip() and x.clause.strip() and x.clause.strip() != CLAUSE_PLACEHOLDER):
                e.value, e.reason = float(x.value), "document / revision / clause are not filled"
            elif x.value < 0:
                e.value, e.reason = float(x.value), "a limit must not be negative"
            elif not confirmed:
                e.value, e.reason = float(x.value), "the requirement set is not confirmed"
            else:
                e.value, e.usable, e.reason = float(x.value), True, ""
        return out

    def value(self, name: str) -> float | None:
        e = self.entries.get(name)
        return e.value if e is not None and e.usable else None

    def is_not_applicable(self, name: str) -> bool:
        e = self.entries.get(name)
        return bool(e is not None and e.not_applicable)

    def status(self) -> dict[str, dict[str, Any]]:
        return {k: {"unit": LIMIT_INFO[k][0], "meaning": LIMIT_INFO[k][2], "value": e.value, "usable": e.usable, "not_applicable": e.not_applicable, "reason": e.reason,
                    "document": e.document, "revision": e.revision, "clause": e.clause, "set_via": e.set_via} for k, e in self.entries.items()}

    def fingerprint(self) -> str:
        """SHA-256 of everything a judgement depends on: the requirement-set content, the structure classes, the confirmation and the usable limits with their provenance.
        A stored result is current only while the fingerprint of the study's limits is still this one."""
        d = {"set": self.set_name, "confirmed": self.set_confirmed, "by": self.confirmed_by, "requirement_digest": self.requirement_digest, "scope_digest": self.scope_digest,
             "confirmed_via": self.confirmed_via,
             "entries": {k: [e.usable, e.not_applicable, e.value, e.document, e.revision, e.clause, e.set_via] for k, e in sorted(self.entries.items())}}
        return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()


def not_applicable_limits(limits: CorrelationLimits) -> list[dict[str, str]]:
    """Every correlation criterion declared not applicable, with its source and who stands behind it: the requirement set that holds the flag was confirmed by `confirmed_by` through
    `confirmed_via` (gui | mcp); `entered_via` = the interface through which the flag itself was entered when that is known."""
    out: list[dict[str, str]] = []
    for k, e in limits.entries.items():
        if e.not_applicable:
            out.append({"limit": k, "meaning": LIMIT_INFO[k][2], "document": _plain(e.document), "revision": _plain(e.revision), "clause": _plain(e.clause),
                        "entered_via": _plain(e.set_via, 12), "confirmed_by": _plain(limits.confirmed_by, 60), "confirmed_via": _plain(limits.confirmed_via, 12),
                        "date": _plain(limits.date, 12)})
    return out


def not_applicable_text(entry: dict[str, str]) -> str:
    """One line about a criterion declared not applicable (for the result text, the dashboard and the report)."""
    via = {"mcp": "through the MCP interface (an agent)", "gui": "through the GUI"}
    ent = f"entered {via.get(entry['entered_via'], 'through an unknown interface (file import or other code)')}"
    conf = f"set confirmed by {entry['confirmed_by'] or '?'} {via.get(entry['confirmed_via'], 'through an unknown interface')}"
    return f"{entry['limit']}: not applicable per {entry['document']} {entry['revision']}, {entry['clause']}; {ent}; {conf}"


def limits_of_study(study: Any) -> CorrelationLimits:
    """The limits of the study: usable only from a confirmed requirement set whose structure classes are still the confirmed ones."""
    sp = getattr(study, "space", None)
    return CorrelationLimits.from_requirement_set(getattr(sp, "requirement_set", None), sp)


def install_limit_rows(rs: Any) -> list[str]:
    """Adds the EMPTY rows of the correlation limits (no value) to a requirement set so the user can fill them; returns the keys added. Editing the set withdraws its confirmation."""
    from .model import LimitEntry
    have = {x.key for x in rs.limits}
    added = []
    for k, (unit, _rel, meaning) in LIMIT_INFO.items():
        if k not in have:
            rs.limits.append(LimitEntry(key=k, value=None, unit=unit, document=TEMPLATE_DOCUMENT, revision="", clause=CLAUSE_PLACEHOLDER, note=meaning))
            added.append(k)
    if added and getattr(rs, "confirmed", False):
        rs.withdraw_confirmation()
    return added


# ------------------------------------------------------------------ options, sensors, predicted curves
@dataclass
class CorrelationOptions:
    """Numerical settings (not limits). `pair_min_mac` = 0 pairs every test mode that an FE mode is left for; the pair is then judged by the limits."""
    method: str = "hungarian"            # hungarian | max_mac | order
    pair_min_mac: float = 0.0            # a pair below this MAC is dropped (the test mode stays unpaired = FAIL)
    f_min_hz: float | None = None        # test modes below / above are left out
    f_max_hz: float | None = None
    fe_modes: list[int] | None = None    # numbers (1-based) of the FE modes to compare (None = all)
    response_band_hz: tuple[float, float] | None = None
    damping_at: str = "fe"               # fe | test: the frequency a measured damping ratio is attached to in the damping table
    orth_mass: str = "serep"             # serep | tam | none: the reduced mass of the orthogonality checks (computed from the FE modes, an imported TAM matrix, not computed)
    tam_global_axes: bool = False        # the user states that the imported TAM is in the GLOBAL axes of the FE model (needed when the test data have rotated sensors, UFF 2420)
    tam_global_axes_via: str = ""        # through which interface the statement was made: gui | mcp (an agent: the user is told to check it) | "" (the program's own API)
    hidden_peak_db: float = 6.0          # a peak of a measured curve that stands this many dB above its surroundings is a resonance (detection of resonances left outside the compared band)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "pair_min_mac": self.pair_min_mac, "f_min_hz": self.f_min_hz, "f_max_hz": self.f_max_hz, "fe_modes": self.fe_modes,
                "response_band_hz": list(self.response_band_hz) if self.response_band_hz else None, "damping_at": self.damping_at, "hidden_peak_db": self.hidden_peak_db, "orth_mass": self.orth_mass,
                "tam_global_axes": self.tam_global_axes, "tam_global_axes_via": self.tam_global_axes_via if self.tam_global_axes else ""}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CorrelationOptions":
        d = d or {}
        o = cls()
        for k in ("method", "damping_at", "orth_mass"):
            if d.get(k):
                setattr(o, k, str(d[k]))
        for k in ("pair_min_mac", "f_min_hz", "f_max_hz", "hidden_peak_db"):
            if d.get(k) is not None:
                setattr(o, k, float(d[k]))
        o.tam_global_axes = d.get("tam_global_axes") is True
        o.tam_global_axes_via = str(d.get("tam_global_axes_via") or "")[:12] if o.tam_global_axes else ""
        if d.get("fe_modes"):
            o.fe_modes = [int(v) for v in d["fe_modes"]]
        if d.get("response_band_hz"):
            lo, hi = d["response_band_hz"]
            o.response_band_hz = (float(lo), float(hi))
        o.validate()
        return o

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        if not 0.0 <= self.pair_min_mac <= 1.0:
            raise ValueError("pair_min_mac must be between 0 and 1")
        if self.damping_at not in ("fe", "test"):
            raise ValueError("damping_at must be fe or test")
        if self.orth_mass not in ORTH_MASS:
            raise ValueError(f"orth_mass must be one of {', '.join(ORTH_MASS)}")
        if not (math.isfinite(self.hidden_peak_db) and self.hidden_peak_db > 0):
            raise ValueError("hidden_peak_db must be a positive number of dB")
        if self.response_band_hz is not None and not (0 <= self.response_band_hz[0] < self.response_band_hz[1]):
            raise ValueError("response_band_hz must be [f_low, f_high] with 0 <= f_low < f_high")
        if self.f_min_hz is not None and self.f_max_hz is not None and self.f_min_hz >= self.f_max_hz:
            raise ValueError("f_min_hz must be below f_max_hz")

    def _axes_note(self) -> str:
        if self.orth_mass != "tam" or not self.tam_global_axes:
            return ""
        via = {"gui": " through the GUI", "mcp": " through the MCP interface (an agent): check it yourself"}.get(self.tam_global_axes_via, f" via {self.tam_global_axes_via}" if self.tam_global_axes_via else "")
        return f" (TAM stated to be in global axes{via})"

    def describe(self) -> str:
        """The options in one line for the report and the dashboard."""
        win = "all test modes" if self.f_min_hz is None and self.f_max_hz is None else f"test modes {self.f_min_hz if self.f_min_hz is not None else 0:g} - {self.f_max_hz if self.f_max_hz is not None else 'inf'} Hz"
        fe = "all FE modes" if self.fe_modes is None else "FE modes " + ", ".join(str(v) for v in self.fe_modes[:20]) + (" ..." if len(self.fe_modes) > 20 else "")
        band = "" if self.response_band_hz is None else f"; response band {self.response_band_hz[0]:g} - {self.response_band_hz[1]:g} Hz"
        return f"pairing {self.method}, minimum MAC of a pair {self.pair_min_mac:g}; {win}; {fe}{band}; reduced mass of the orthogonality checks: {self.orth_mass}{self._axes_note()}"


@dataclass
class Sensor:
    """Where a test node is in the FE model: a point [mm] (interpolated in the mesh), a named probe of the study, or a surface selection (mean of its nodes)."""
    node: str
    point: list[float] | None = None
    probe: str = ""
    selection: str = ""
    dofs: tuple[str, ...] = TRANSLATIONS

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"node": self.node, "dofs": "".join(self.dofs)}
        if self.point is not None:
            d["point"] = [float(v) for v in self.point]
        if self.probe:
            d["probe"] = self.probe
        if self.selection:
            d["selection"] = self.selection
        return d


def sensors_from_dicts(items: list[dict[str, Any]] | None, study: Any = None) -> list[Sensor]:
    """Sensors from dictionaries {node, point | probe | selection, dofs}; a probe is resolved against the probes of the study, a selection must be a selection of the study.
    Names are reduced to plain text. Raises ValueError on a bad entry."""
    out: list[Sensor] = []
    seen: set[str] = set()
    if items is not None and len(items) > 5000:
        raise ValueError("too many sensors (at most 5000)")
    probes = {p.name: p for p in getattr(getattr(study, "dynamics", None), "probes", [])} if study is not None else {}
    selections = getattr(study, "selections", None) if study is not None else None
    for raw in items or []:
        if not isinstance(raw, dict):
            raise ValueError("a sensor must be an object {node, point | probe | selection, dofs}")
        node = clean_label(raw.get("node", ""))
        if not node or node in seen:
            raise ValueError(f"sensor node '{node}' is empty or listed twice")
        seen.add(node)
        dofs_raw = raw.get("dofs") or "xyz"
        if isinstance(dofs_raw, str):
            dofs_raw = [t for t in dofs_raw.replace(" ", "").split(",") if t] if "," in dofs_raw else list(dofs_raw.replace(" ", ""))
        dofs = tuple(dict.fromkeys(parse_dof(str(d))[0] for d in dofs_raw))
        if not dofs or any(d not in TRANSLATIONS for d in dofs):
            raise ValueError(f"sensor {node}: dofs must be among x, y, z")
        pt = raw.get("point")
        s = Sensor(node=node, dofs=dofs, probe=_plain(raw.get("probe")), selection=_plain(raw.get("selection")))
        if pt is not None:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pt)):
                raise ValueError(f"sensor {node}: point must be [x, y, z] in mm")
            s.point = [float(v) for v in pt]
        if s.probe and s.point is None and not s.selection:
            pr = probes.get(s.probe)
            if pr is None:
                raise ValueError(f"sensor {node}: probe '{s.probe}' is not in the study")
            if pr.point is not None:
                s.point = [float(v) for v in pr.point]
            elif pr.selection:
                s.selection = _plain(pr.selection)
        if s.point is None and not s.selection:
            raise ValueError(f"sensor {node}: give a point, a probe of the study or a selection")
        if s.point is None and selections is not None and s.selection not in selections:
            raise ValueError(f"sensor {node}: the selection '{s.selection}' is not a selection of the study")
        out.append(s)
    return out


def auto_sensors(test: TestData, study: Any) -> list[Sensor]:
    """Sensors for the test nodes whose label equals the name of a probe of the study (a convenience; the rest is mapped by the user)."""
    probes = {p.name: p for p in getattr(getattr(study, "dynamics", None), "probes", [])}
    items = [{"node": n, "probe": n} for n in test.nodes() if n in probes]
    return sensors_from_dicts(items, study)


@dataclass
class Predicted:
    """A predicted response curve: frequency [Hz] and values (real amplitude / PSD, or complex), `kind` frf | psd | response. `quantity` (displacement [mm] | velocity [mm/s] |
    acceleration [g]), `frame` (absolute | relative | "" = the model has no base motion, so there is no difference) and `per` (none | base_acceleration | force) say what the
    curve is; without them the curve cannot be compared (UNCONFIRMED)."""
    freq: np.ndarray
    values: np.ndarray
    kind: str = "frf"
    label: str = ""
    quantity: str = ""
    frame: str = ""
    per: str = "none"


# ------------------------------------------------------------------ the comparison of a measured with a predicted function
@dataclass
class FunctionSetup:
    """What the user decides for a measured function (kept in the study): the frame of the measurement (absolute | relative to the base), the normalisation (none |
    base_acceleration | force), the unit / quantity when the file has none, the probe and DOF of the model curve when the file has no node / direction (a controller PSD) and the
    reference force [N] of a per-force function (the model curve is divided by it). Empty = taken from the file."""
    frame: str = ""
    per: str = ""
    quantity: str = ""
    unit: str = ""
    probe: str = ""
    dof: str = ""
    reference_force_n: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "per": self.per, "quantity": self.quantity, "unit": self.unit, "probe": self.probe, "dof": self.dof, "reference_force_n": self.reference_force_n}

    @classmethod
    def from_dict(cls, d: Any) -> "FunctionSetup":
        if not isinstance(d, dict):
            raise ValueError("a function setup must be an object")
        s = cls(frame=str(d.get("frame") or "").strip().lower(), per=str(d.get("per") or "").strip().lower(), quantity=str(d.get("quantity") or "").strip().lower(),
                unit=_plain(d.get("unit"), 40), probe=_plain(d.get("probe")), dof=str(d.get("dof") or "").strip().lower())
        rf = d.get("reference_force_n")
        if rf not in (None, ""):
            try:
                s.reference_force_n = float(rf)
            except (TypeError, ValueError):
                raise ValueError("reference_force_n must be a number") from None
        s.validate()
        return s

    def validate(self) -> None:
        if self.frame and self.frame not in FRAMES:
            raise ValueError(f"frame must be {' or '.join(FRAMES)}")
        if self.per and self.per not in PERS:
            raise ValueError(f"per must be one of {', '.join(PERS)}")
        if self.quantity and self.quantity not in ("displacement", "velocity", "acceleration"):
            raise ValueError("quantity must be displacement, velocity or acceleration")
        if self.dof:
            if parse_dof(self.dof)[0] not in TRANSLATIONS:
                raise ValueError("the dof of a function setup must be x, y or z")
            self.dof = parse_dof(self.dof)[0]
        if self.reference_force_n is not None and not (math.isfinite(self.reference_force_n) and self.reference_force_n > 0):
            raise ValueError("reference_force_n must be a positive number [N]")
        if self.unit and parse_unit(self.unit) is None:
            raise ValueError(f"the unit '{self.unit}' is not understood (examples: g, m/s^2, mm, (m/s^2)/N, g^2/Hz)")


@dataclass
class Resolved:
    """The quantity of a measured function: `quantity`, `per`, the unit and what is missing (`problems` = the comparison cannot be made)."""
    quantity: str = ""
    per: str = "none"
    psd: bool = False
    unit: Unit | None = None
    unit_label: str = ""
    problems: list[str] = field(default_factory=list)


def _dims_of(q: str, per: str, psd: bool) -> tuple[int, int, int]:
    a, b, c = QUANTITY_DIMS[q]
    if per == "base_acceleration":
        d, e, f = QUANTITY_DIMS["acceleration"]
        a, b, c = a - d, b - e, c - f
    elif per == "force":
        d, e, f = QUANTITY_DIMS["force"]
        a, b, c = a - d, b - e, c - f
    if psd:
        return (2 * a, 2 * b + 1, 2 * c)
    return (a, b, c)


def resolve_function(fn: TestFunction, setup: FunctionSetup | None = None) -> Resolved:
    """What a measured function is, from its unit label (or the unit of the user's setup) with the quantity / denominator of the file as a cross-check. A UFF 58 states the
    numerator unit in record 9 and the denominator unit in record 10: "g" and "N" make a g/N function (a label that already holds the denominator is not divided twice). A function
    whose unit is unknown, or does not fit a displacement / velocity / acceleration response (per force / per base acceleration / PSD), has problems and is not compared."""
    s = setup or FunctionSetup()
    r = Resolved(psd=fn.kind == "psd", unit_label=s.unit or fn.unit)
    units: list[Unit] = []
    if s.unit:
        u0 = parse_unit(s.unit)
        units = [u0] if u0 is not None else []
    else:
        num, den = parse_unit(fn.unit), parse_unit(fn.per_unit)
        if num is not None and den is not None:
            units.append(num * den.inv())
        if num is not None:
            units.append(num)
    if not units:
        hint = unit_hint(r.unit_label)
        r.problems.append(f"the unit '{_plain(r.unit_label, 30) or 'none'}' of the measured function is not known - state it in the function setup" + (f" ({hint})" if hint else ""))
        return r
    declared_q = s.quantity or fn.quantity
    declared_p = s.per or {"force": "force", "acceleration": "base_acceleration", "": ""}.get(fn.per, fn.per)
    if declared_p not in ("", "none") + PERS:
        r.problems.append(f"the denominator '{declared_p}' of the function is not supported (force and base acceleration are)")
        r.unit = units[0]
        return r
    r.unit = units[0]
    cands: list[tuple[str, str]] = []
    for u in units:
        cands = [(q, p) for q in ("displacement", "velocity", "acceleration") for p in (("none",) if r.psd else PERS) if _dims_of(q, p, r.psd) == u.dims]
        if declared_q and declared_q != "force":
            cands = [c for c in cands if c[0] == declared_q]
        if declared_p:
            cands = [c for c in cands if c[1] == declared_p]
        if cands:
            r.unit = u
            break
    if declared_q == "force":
        r.problems.append("a force response cannot be compared with the model (displacement, velocity and acceleration can)")
    elif not cands:
        r.problems.append(f"the unit '{_plain(r.unit_label, 30)}'{' / ' + _plain(fn.per_unit, 20) if fn.per_unit and not s.unit and parse_unit(fn.per_unit) is not None else ''} does not fit the declared "
                          f"{declared_q or 'displacement / velocity / acceleration'}"
                          f"{' per ' + declared_p.replace('_', ' ') if declared_p not in ('', 'none') else ''}"
                          f"{' PSD' if r.psd else ''} - check the unit and the setup")
    else:
        r.quantity, r.per = cands[0]
    return r


def _unit_scale(test_unit: Unit, quantity: str, per: str, psd: bool) -> float:
    return test_unit.factor / fe_unit(quantity, per, psd).factor


# ------------------------------------------------------------------ mode metrics
def mac_matrix(test_phi: np.ndarray, fe_phi: np.ndarray) -> np.ndarray:
    """MAC (n_test, n_fe). `test_phi` (n_dof, n_test) may hold NaN = DOF not measured in that mode (it drops out of that mode only); `fe_phi` (n_dof, n_fe) is real."""
    tp = np.asarray(test_phi)
    fp = np.asarray(fe_phi, dtype=np.float64)
    if tp.ndim != 2 or fp.ndim != 2 or tp.shape[0] != fp.shape[0]:
        raise ValueError("the shapes need the same DOFs (rows)")
    out = np.zeros((tp.shape[1], fp.shape[1]))
    for i in range(tp.shape[1]):
        a = tp[:, i]
        rows = np.isfinite(a.real) & np.isfinite(a.imag) if np.iscomplexobj(a) else np.isfinite(a)
        if rows.sum() < 1:
            continue
        a = a[rows]
        b = fp[rows, :]
        num = np.abs(a.conj() @ b) ** 2
        den = float(np.vdot(a, a).real) * np.sum(b * b, axis=0)
        out[i] = np.divide(num, den, out=np.zeros(fp.shape[1]), where=den > 0)
    return np.clip(out, 0.0, 1.0)


def pair_modes(mac: np.ndarray, test_f: np.ndarray, fe_f: np.ndarray, method: str = "hungarian", pair_min_mac: float = 0.0) -> list[tuple[int, int]]:
    """Pairs (test index, FE index). hungarian = one-to-one assignment that maximises the sum of the MAC (SciPy `linear_sum_assignment`; equal MAC are decided by the frequency
    error); max_mac = every test mode takes its best FE mode (an FE mode may be taken twice); order = the i-th test mode with the i-th FE mode. Pairs below `pair_min_mac` are dropped."""
    nt, nf = mac.shape
    if nt == 0 or nf == 0:
        return []
    rel = np.abs(fe_f[None, :] - test_f[:, None]) / np.maximum(np.abs(test_f[:, None]), 1e-12)
    cost = (1.0 - mac) + 1e-9 * np.minimum(rel, 1e3)
    pairs: list[tuple[int, int]]
    if method == "hungarian":
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        pairs = [(int(i), int(j)) for i, j in zip(r, c)]
    elif method == "max_mac":
        pairs = [(i, int(np.argmin(cost[i]))) for i in range(nt)]
    elif method == "order":
        pairs = [(i, i) for i in range(min(nt, nf))]
    else:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    return sorted((i, j) for i, j in pairs if mac[i, j] >= pair_min_mac)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.sqrt(np.sum(np.abs(v) ** 2)))
    return v / n if n > 0 else v


def comac(test_phi: np.ndarray, fe_phi: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """CoMAC of every DOF over the paired modes (NaN for a DOF with fewer than two measured pairs). See the module docstring."""
    nd = test_phi.shape[0]
    psi_t = np.full((nd, len(pairs)), np.nan, dtype=complex)
    psi_a = np.full((nd, len(pairs)), np.nan, dtype=complex)
    for k, (i, j) in enumerate(pairs):
        a = test_phi[:, i]
        rows = np.isfinite(a.real) & np.isfinite(a.imag)
        if rows.sum() == 0:
            continue
        psi_t[rows, k] = _unit(a[rows])
        psi_a[rows, k] = _unit(fe_phi[rows, j].astype(complex))
    out = np.full(nd, np.nan)
    for d in range(nd):
        ok = np.isfinite(psi_t[d].real)
        if ok.sum() < 2:
            continue
        t, a = psi_t[d, ok], psi_a[d, ok]
        den = float(np.sum(np.abs(t) ** 2) * np.sum(np.abs(a) ** 2))
        out[d] = float(np.sum(np.abs(t * a)) ** 2 / den) if den > 0 else np.nan
    return out


@dataclass
class ModePair:
    test_index: int
    test_number: int
    test_freq: float
    fe_index: int
    fe_number: int
    fe_freq: float
    mac: float
    freq_error: float
    damping: float | None = None
    offdiag: float | None = None
    scale: float = 1.0
    dofs: int = 0
    coc: float | None = None             # cross-orthogonality of the pair (diagonal term); None = not computed
    coc_offdiag: float | None = None     # largest COC of the test mode with the FE mode of another pair (and the other way round)
    aoc_offdiag: float | None = None     # largest AOC of the test mode with another test mode
    modulus_diff: float | None = None
    modulus_dof: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"test_mode": self.test_number, "test_freq_Hz": self.test_freq, "fe_mode": self.fe_number, "fe_freq_Hz": self.fe_freq, "mac": self.mac,
                "freq_error": self.freq_error, "damping": self.damping, "mac_offdiag_max": self.offdiag, "dofs": self.dofs, "coc": self.coc, "coc_offdiag_max": self.coc_offdiag,
                "aoc_offdiag_max": self.aoc_offdiag, "modulus_difference": self.modulus_diff, "modulus_dof": self.modulus_dof}


@dataclass
class ModalCorrelation:
    dofs: list[tuple[str, str]]
    test_numbers: list[int]
    fe_numbers: list[int]
    test_freq: np.ndarray
    fe_freq: np.ndarray
    mac: np.ndarray                      # (n_test, n_fe)
    pairs: list[ModePair]
    unpaired_test: list[int]             # test mode numbers without an FE partner
    unpaired_fe: list[int]
    comac: dict[str, float | None]       # "node:dof" -> CoMAC
    auto_mac: np.ndarray                 # (n_test, n_test) AutoMAC of the test modes (sensor placement quality)
    warnings: list[str] = field(default_factory=list)
    orth: OrthResult = field(default_factory=OrthResult)      # the cross- and auto-orthogonality (3.2.39) with the reduced mass they used

    def to_dict(self, matrices: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {"test_modes": self.test_numbers, "fe_modes": self.fe_numbers, "pairs": [p.to_dict() for p in self.pairs],
                             "unpaired_test_modes": self.unpaired_test, "unpaired_fe_modes": self.unpaired_fe, "comac": self.comac,
                             "dofs": [f"{n}:{x}" for n, x in self.dofs]}
        if matrices and self.mac.size <= MAX_MATRIX_IN_DICT:
            d["mac"] = np.round(self.mac, 5).tolist()
            d["test_freq_Hz"] = [float(v) for v in self.test_freq]
            d["fe_freq_Hz"] = [float(v) for v in self.fe_freq]
            d["auto_mac"] = np.round(self.auto_mac, 5).tolist() if 0 < self.auto_mac.size <= MAX_MATRIX_IN_DICT else None
        d["orthogonality"] = self.orth.to_dict(matrices, MAX_MATRIX_IN_DICT)
        return d


def modal_correlation(test_freq: np.ndarray, test_phi: np.ndarray, fe_freq: np.ndarray, fe_phi: np.ndarray, dofs: list[tuple[str, str]],
                      test_numbers: list[int] | None = None, fe_numbers: list[int] | None = None, test_damping: list[float | None] | None = None,
                      options: CorrelationOptions | None = None, orth: OrthResult | None = None) -> ModalCorrelation:
    """The modal metrics from arrays: test_phi (n_dof, n_test) with NaN for a DOF not measured, fe_phi (n_dof, n_fe) at the same DOFs. `orth` = the cross- / auto-orthogonality
    (`corr_orth.orthogonality`); without it they are not computed and their rows are UNCONFIRMED."""
    opt = options or CorrelationOptions()
    opt.validate()
    tf, ff = np.asarray(test_freq, float), np.asarray(fe_freq, float)
    tn = test_numbers or list(range(1, len(tf) + 1))
    fn = fe_numbers or list(range(1, len(ff) + 1))
    warn: list[str] = []
    mac = mac_matrix(test_phi, fe_phi)
    pairs_ij = pair_modes(mac, tf, ff, opt.method, opt.pair_min_mac)
    pairs: list[ModePair] = []
    for i, j in pairs_ij:
        a = test_phi[:, i]
        rows = np.isfinite(a.real) & np.isfinite(a.imag)
        b = fe_phi[rows, j]
        den = float(np.vdot(b, b).real)
        scale = complex(np.vdot(b, a[rows]) / den) if den > 0 else 1.0
        z = test_damping[i] if test_damping and i < len(test_damping) else None
        pairs.append(ModePair(i, tn[i], float(tf[i]), j, fn[j], float(ff[j]), float(mac[i, j]), float((ff[j] - tf[i]) / tf[i]) if tf[i] > 0 else float("nan"), z,
                              None, float(scale.real), int(rows.sum())))
    if len(pairs) >= 2:
        sub = mac[np.ix_([p.test_index for p in pairs], [p.fe_index for p in pairs])]
        for k, p in enumerate(pairs):
            others = np.concatenate([np.delete(sub[k], k), np.delete(sub[:, k], k)])
            p.offdiag = float(others.max()) if others.size else None
    orth = orth if orth is not None else OrthResult(problems=["no reduced mass for the orthogonality checks was given"])
    md = modulus_difference(test_phi, fe_phi, pairs_ij, list(dofs))
    for p, (v, where) in zip(pairs, md):
        p.modulus_diff, p.modulus_dof = v, where
    if orth.coc is not None and orth.aoc is not None:
        def fin(x: float) -> float | None:
            return float(x) if math.isfinite(float(x)) else None
        for p in pairs:
            p.coc = fin(orth.coc[p.test_index, p.fe_index])
            others_t = [orth.aoc[p.test_index, u] for u in range(orth.aoc.shape[1]) if u != p.test_index]
            p.aoc_offdiag = fin(max(others_t)) if others_t and all(math.isfinite(float(v)) for v in others_t) else None
        if len(pairs) >= 2:
            csub = orth.coc[np.ix_([p.test_index for p in pairs], [p.fe_index for p in pairs])]
            for k, p in enumerate(pairs):
                oth = np.concatenate([np.delete(csub[k], k), np.delete(csub[:, k], k)])
                p.coc_offdiag = fin(oth.max()) if oth.size and np.isfinite(oth).all() else None
    paired_t, paired_f = {p.test_index for p in pairs}, {p.fe_index for p in pairs}
    cm = comac(test_phi, fe_phi, pairs_ij) if len(pairs) >= 2 else np.full(len(dofs), np.nan)
    if len(pairs) < 2:
        warn.append("CoMAC needs at least two paired modes")
    n_dof_min = min((p.dofs for p in pairs), default=0)
    if pairs and n_dof_min < len(pairs):
        warn.append(f"a mode is judged on {n_dof_min} DOFs only, fewer than the {len(pairs)} paired modes: the MAC cannot tell the modes apart reliably (add sensors)")
    am = mac_matrix(test_phi, np.where(np.isfinite(test_phi), test_phi, 0.0)) if not np.iscomplexobj(test_phi) else np.zeros((0, 0))     # AutoMAC of real shapes only
    return ModalCorrelation(dofs=list(dofs), test_numbers=tn, fe_numbers=fn, test_freq=tf, fe_freq=ff, mac=mac, pairs=pairs,
                            unpaired_test=[tn[i] for i in range(len(tf)) if i not in paired_t], unpaired_fe=[fn[j] for j in range(len(ff)) if j not in paired_f],
                            comac={f"{n}:{x}": (None if not math.isfinite(float(v)) else float(v)) for (n, x), v in zip(dofs, cm)}, auto_mac=am, warnings=warn, orth=orth)


# ------------------------------------------------------------------ FE shapes at the sensors
def _sensor_operator(res: Any, mesh: Any, s: Sensor) -> tuple[np.ndarray, np.ndarray, float]:
    """(node indices, weights, distance outside the element in barycentric units) of a sensor."""
    if s.point is None:
        try:
            idx = np.asarray(mesh.surface_nodes(s.selection))
        except KeyError:
            raise ValueError(f"sensor {s.node}: the selection '{_plain(s.selection)}' has no surface in the mesh") from None
        if idx.size == 0:
            raise ValueError(f"sensor {s.node}: the selection '{_plain(s.selection)}' has no nodes")
        return idx, np.full(idx.size, 1.0 / idx.size), 0.0
    from ..results import tet10_shape
    p = np.asarray(s.point, float)
    e, xi = res.locate(p)
    conn = np.asarray(mesh.tets[e])
    w = tet10_shape(xi) if conn.size == 10 else np.array([1.0 - xi.sum(), *xi])
    return conn, np.asarray(w), float(max(-xi.min(), xi.sum() - 1.0, 0.0))


def fe_shapes_at_sensors(modeset: Any, mesh: Any, sensors: list[Sensor], dofs: list[tuple[str, str]], mode_idx: list[int] | None = None) -> tuple[np.ndarray, list[str]]:
    """FE mode shapes (n_dof, n_modes) at the sensor DOFs (interpolated with the shape functions of the element that holds the sensor point) and warnings. Only the nodes of the
    sensors are taken out of the mode array (one small block per sensor), never a copy of the whole array."""
    from ..results import Results
    res = Results(mesh, [])
    by_node = {s.node: s for s in sensors}
    shapes = np.asarray(modeset.shapes)
    idx = np.arange(shapes.shape[0]) if mode_idx is None else np.asarray(mode_idx, dtype=np.int64)
    out = np.zeros((len(dofs), len(idx)))
    done: dict[str, np.ndarray] = {}
    warn: list[str] = []
    for d, (node, dof) in enumerate(dofs):
        if node not in done:
            if node not in by_node:
                raise ValueError(f"test node {_plain(node, 40)} has no sensor")
            nodes, w, outside = _sensor_operator(res, mesh, by_node[node])
            if outside > 0.05:
                warn.append(f"sensor {node} lies outside the mesh (its point is extrapolated from the nearest element)")
            block = shapes[np.ix_(idx, np.asarray(nodes, dtype=np.int64))].astype(np.float64)          # (modes, sensor nodes, 3)
            done[node] = np.einsum("p,kpd->kd", w, block)
        out[d] = done[node][:, TRANSLATIONS.index(dof)]
    return out, warn


def fe_test_data(modeset: Any, mesh: Any, sensors: list[Sensor], damping: float | None = None, numeric_nodes: bool = False) -> TestData:
    """A test dataset made from the FE model itself (its modes at the sensors): the reference of the self-consistency check and a way to hand the FE modes to a test team."""
    dofs = [(s.node, d) for s in sensors for d in s.dofs]
    phi, _ = fe_shapes_at_sensors(modeset, mesh, sensors, dofs)
    td = TestData(source="FE model")
    for k, f in enumerate(modeset.values):
        td.modes.append(TestMode(number=k + 1, frequency_hz=float(f), damping=damping, shape={dofs[d]: float(phi[d, k]) for d in range(len(dofs))}))
    return td


def fe_fingerprint(modeset: Any, mesh: Any) -> str:
    """SHA-256 of the FE mode set: its frequencies, the shape of the mode array and the mesh (nodes and elements). A stored result is current only for the same mode set."""
    h = hashlib.sha256()
    h.update(json.dumps([round(float(f), 9) for f in getattr(modeset, "values", [])]).encode("ascii"))
    shp = getattr(modeset, "shapes", None)
    h.update(repr(tuple(np.shape(shp)) if shp is not None else ()).encode("ascii"))
    for name in ("nodes", "tets"):
        arr = getattr(mesh, name, None)
        if arr is not None:
            a = np.ascontiguousarray(arr)
            h.update(repr((name, a.shape, str(a.dtype))).encode("ascii"))
            h.update(a.data.cast("B"))
    return h.hexdigest()


# ------------------------------------------------------------------ damping
def damping_table(pairs: list[ModePair], at: str = "fe") -> list[list[float]]:
    """[[f Hz, zeta], ...] from the measured damping of the paired modes, ascending in frequency (the frequency of the FE mode or of the test mode)."""
    rows: dict[float, list[float]] = {}
    for p in pairs:
        if p.damping is None or not (math.isfinite(p.damping) and 0 < p.damping < 1):
            continue
        rows.setdefault(round(p.fe_freq if at == "fe" else p.test_freq, 9), []).append(p.damping)
    return [[f, float(np.mean(z))] for f, z in sorted(rows.items())]


def transfer_damping(study: Any, pairs: list[ModePair], at: str = "fe") -> list[list[float]]:
    """Writes the measured modal damping of the paired modes into `Study.dynamics.damping` (kind modal, table [[f, zeta]]); returns the table."""
    table = damping_table(pairs, at)
    if not table:
        raise ValueError("no paired mode has a measured damping ratio between 0 and 1")
    d = study.dynamics.damping
    d.kind = "modal"
    d.table = [list(r) for r in table]
    return table


# ------------------------------------------------------------------ responses
@dataclass
class ResponseMetrics:
    name: str
    node: str
    dof: str
    kind: str
    f_peak_test: float
    f_peak_fe: float
    peak_shift: float
    amp_test: float
    amp_fe: float
    peak_error_db: float
    rms_error: float
    level_error: float | None
    band_hz: tuple[float, float]
    points: int
    freq: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    test_amp: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fe_amp: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    warnings: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)        # why the rows of this curve cannot be PASS / FAIL (frame not chosen / does not match ...)
    quantity: str = ""
    unit: str = ""                                              # the unit the two amplitudes are in (the model unit)
    hidden_peaks_hz: list[float] = field(default_factory=list)  # resonances of the MEASURED curve outside the compared band (the band option / a shorter model curve hides them)
    measured_band_hz: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "node": self.node, "dof": self.dof, "kind": self.kind, "f_peak_test_Hz": self.f_peak_test, "f_peak_fe_Hz": self.f_peak_fe,
                "peak_freq_shift": self.peak_shift, "amp_test": self.amp_test, "amp_fe": self.amp_fe, "peak_amp_error_dB": self.peak_error_db,
                "rms_error": self.rms_error, "level_error": self.level_error, "band_Hz": list(self.band_hz), "points": self.points, "warnings": self.warnings,
                "unconfirmed": self.unconfirmed, "quantity": self.quantity, "unit": self.unit, "test_peaks_outside_band_Hz": list(self.hidden_peaks_hz),
                "measured_band_Hz": list(self.measured_band_hz)}


@dataclass
class ResponseIssue:
    """A measured function that could not be compared (unknown / mismatching quantity or unit): reported as an UNCONFIRMED row, never as PASS."""
    name: str
    node: str
    dof: str
    problems: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "node": self.node, "dof": self.dof, "problems": list(self.problems)}


def _amp(v: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(v))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _peaks_outside(f: np.ndarray, amp: np.ndarray, lo: float, hi: float, kind: str, prominence_db: float) -> list[float]:
    """Frequencies of the resonances of a measured curve that lie outside [lo, hi]: local maxima that stand at least `prominence_db` above their surroundings (20 log10 of the
    amplitude, 10 log10 of a PSD)."""
    out = (f < lo) | (f > hi)
    if f.size < 3 or not out.any():
        return []
    from scipy.signal import find_peaks
    pos = amp[amp > 0]
    if pos.size == 0:
        return []
    y = (10.0 if kind == "psd" else 20.0) * np.log10(np.maximum(amp, 0.5 * float(pos.min())))
    idx, _ = find_peaks(y, prominence=prominence_db)
    return [float(f[i]) for i in idx if out[i]]


def compare_response(test_f: np.ndarray, test_v: np.ndarray, fe_f: np.ndarray, fe_v: np.ndarray, kind: str = "frf", band: tuple[float, float] | None = None,
                     name: str = "", node: str = "", dof: str = "", hidden_peak_db: float = 6.0) -> ResponseMetrics:
    """Peak frequency shift, peak amplitude error [dB] and RMS error of the predicted curve against the measured one in the common band (see the module docstring). The two curves
    must already be in the same unit: this function does not know the quantity (`correlate` checks and converts). Resonances of the measured curve outside the compared band
    (cut by `band` or by a shorter predicted curve) are listed in `hidden_peaks_hz`."""
    tf, ff = np.asarray(test_f, float), np.asarray(fe_f, float)
    if tf.size < 2 or ff.size < 2 or not (np.all(np.diff(tf) > 0) and np.all(np.diff(ff) > 0)):
        raise ValueError("a curve needs at least two points with strictly increasing frequency")
    lo, hi = max(tf[0], ff[0]), min(tf[-1], ff[-1])
    if band is not None:
        lo, hi = max(lo, band[0]), min(hi, band[1])
    if not lo < hi:
        raise ValueError("the measured and the predicted curve have no common frequency band")
    at, af = _amp(test_v), _amp(fe_v)
    warn: list[str] = []
    mt = (tf >= lo) & (tf <= hi)
    mf = (ff >= lo) & (ff <= hi)
    if mt.sum() < 2 or mf.sum() < 2:
        raise ValueError("fewer than two points of a curve lie in the common band")
    if (hi - lo) < 0.5 * (tf[-1] - tf[0]):
        warn.append("the common band covers less than half of the measured band")
    f0 = tf[mt]
    a_t = at[mt]
    a_f = np.interp(f0, ff, af)
    it, jf = int(np.argmax(a_t)), int(np.argmax(af[mf]))
    f_peak_t, f_peak_f = float(f0[it]), float(ff[mf][jf])
    p_t, p_f = float(a_t[it]), float(af[mf][jf])
    if not (p_t > 0 and p_f > 0):
        raise ValueError("a curve has no positive peak")
    factor = 10.0 if kind == "psd" else 20.0
    ref = math.sqrt(float(np.mean(a_t ** 2)))
    rms = math.sqrt(float(np.mean((a_f - a_t) ** 2))) / ref if ref > 0 else float("nan")
    level = None
    if kind == "psd":
        lt, lf = float(_trapz(a_t, f0)), float(_trapz(a_f, f0))
        level = (math.sqrt(lf) - math.sqrt(lt)) / math.sqrt(lt) if lt > 0 else None
    return ResponseMetrics(name=name, node=node, dof=dof, kind=kind, f_peak_test=f_peak_t, f_peak_fe=f_peak_f, peak_shift=(f_peak_f - f_peak_t) / f_peak_t,
                           amp_test=p_t, amp_fe=p_f, peak_error_db=factor * math.log10(p_f / p_t), rms_error=rms, level_error=level, band_hz=(float(lo), float(hi)),
                           points=int(mt.sum()), freq=f0, test_amp=a_t, fe_amp=a_f, warnings=warn, hidden_peaks_hz=_peaks_outside(tf, at, float(lo), float(hi), kind, hidden_peak_db),
                           measured_band_hz=(float(tf[0]), float(tf[-1])))


def modal_frf(freq: np.ndarray, modal_hz: np.ndarray, zeta: np.ndarray | float, phi_response: np.ndarray, phi_force: np.ndarray, quantity: str = "accelerance") -> np.ndarray:
    """Transfer function from modal data: sum_k phi_r,k phi_f,k / (w_k^2 - w^2 + 2 i zeta_k w_k w) (mass-normalised shapes), as receptance, velocity or accelerance (-w^2 times)."""
    w = 2 * np.pi * np.asarray(freq, float)[:, None]
    wk = 2 * np.pi * np.asarray(modal_hz, float)[None, :]
    z = np.broadcast_to(np.asarray(zeta, float), (wk.shape[1],))[None, :]
    h = (np.asarray(phi_response)[None, :] * np.asarray(phi_force)[None, :] / (wk ** 2 - w ** 2 + 2j * z * wk * w)).sum(axis=1)
    return {"receptance": h, "velocity": 1j * w[:, 0] * h, "accelerance": -(w[:, 0] ** 2) * h}[quantity]


def _direction(dof: str, axis: tuple[float, float, float] | None) -> np.ndarray:
    if axis is not None:
        a = np.asarray(axis, float)
        n = float(np.linalg.norm(a))
        if not n > 0:
            raise ValueError("the measuring axis is zero")
        return a / n
    d = np.zeros(3)
    d[TRANSLATIONS.index(dof)] = 1.0
    return d


def predicted_from_dynamics(dyn: Any, probe: str, dof: str, quantity: str = "harmonic_acceleration", *, axis: tuple[float, float, float] | None = None, frame: str = "",
                            per: str = "none", reference_force_n: float | None = None) -> Predicted:
    """A predicted curve from the results of the dynamics module at a probe, along the axis `dof` (or the global direction `axis` of a local sensor).

    quantity  harmonic_displacement [mm] | harmonic_velocity [mm/s] | harmonic_acceleration [g] (complex) | random_acceleration_psd (absolute acceleration PSD [g2/Hz])
    frame     relative = relative to the base (what the modal equations give; the default of a harmonic curve); absolute = relative + the base motion of the excitation (needs
              the base record of the harmonic response; the only one of a PSD); without any base excitation the two are the same and the curve carries frame ""
    per       none | base_acceleration (divided by the base acceleration, in g) | force (divided by `reference_force_n` [N])
    """
    frame = frame or ("absolute" if quantity == "random_acceleration_psd" else "relative")
    if frame not in FRAMES:
        raise ValueError(f"frame must be {' or '.join(FRAMES)}")
    if per not in PERS:
        raise ValueError(f"per must be one of {', '.join(PERS)}")
    if quantity == "random_acceleration_psd":
        if per != "none":
            raise ValueError("a PSD cannot be normalised")
        r = getattr(dyn, "random", None)
        if r is None or probe not in r.get("probes", {}):
            raise ValueError(f"the random response has no probe '{probe}'")
        if frame != "absolute":
            raise ValueError("the random response gives the absolute acceleration PSD only")
        if axis is not None and sorted(np.abs(_direction(dof, axis)).round(9).tolist()) != [0.0, 0.0, 1.0]:
            raise ValueError("a PSD can only be predicted along a global axis (the cross terms of the components are not stored)")
        ax = int(np.argmax(np.abs(_direction(dof, axis)))) if axis is not None else TRANSLATIONS.index(dof)
        return Predicted(np.asarray(r["f"], float), np.asarray(r["probes"][probe]["A_psd_g2"])[:, ax], "psd", f"{probe}:{dof} A [g2/Hz]", "acceleration", "absolute", "none")
    if not quantity.startswith("harmonic") or quantity not in ("harmonic_displacement", "harmonic_velocity", "harmonic_acceleration"):
        raise ValueError("quantity must be harmonic_displacement, harmonic_velocity, harmonic_acceleration or random_acceleration_psd")
    h = getattr(dyn, "harmonic", None)
    if h is None or probe not in h.get("probes", {}):
        raise ValueError(f"the harmonic response has no probe '{probe}'")
    f = np.asarray(h["f"], float)
    w = 2 * np.pi * f
    d = _direction(dof, axis)
    u = np.asarray(h["probes"][probe]["U"]) @ d                                   # relative displacement [mm]
    ab = h.get("ab")
    ub = h.get("ub")
    ab_d = np.asarray(ab) @ d if ab is not None else None                          # base acceleration [mm/s2] along the measuring direction
    has_base = ab_d is not None and bool(np.any(np.abs(ab_d) > 0))
    if frame == "absolute" and (ab is None or ub is None):
        raise ValueError("the harmonic response has no base-motion record: only the relative response can be predicted")
    ub_d = np.asarray(ub) @ d if (ub is not None and frame == "absolute") else 0.0
    ab_use = ab_d if (ab_d is not None and frame == "absolute") else 0.0
    if quantity == "harmonic_displacement":
        v, q = u + ub_d, "displacement"
    elif quantity == "harmonic_velocity":
        v, q = 1j * w * (u + ub_d), "velocity"
    else:
        v, q = (-(w ** 2) * u + ab_use) / G_MM, "acceleration"
    if per == "base_acceleration":
        if ab_d is None or not np.all(np.abs(ab_d) > 0):
            raise ValueError("the harmonic response has no base acceleration along this direction to normalise by")
        v = v / (ab_d / G_MM)
    elif per == "force":
        if reference_force_n is None or not reference_force_n > 0:
            raise ValueError("a per-force curve needs reference_force_n > 0 (the amplitude of the force excitation of the dynamics analysis)")
        v = v / float(reference_force_n)
    shown_frame = "" if (ab is not None and not has_base) else frame
    return Predicted(f, v, "frf", f"{probe}:{dof} {q} ({frame})", q, shown_frame, per)


# ------------------------------------------------------------------ fingerprints of the inputs of a comparison (the provenance of a stored result)
def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode("ascii")).hexdigest()


def sensors_fingerprint(sensors: list[Sensor]) -> str:
    """SHA-256 of the sensor definitions (node, point / probe / selection, DOFs)."""
    return _digest(sorted((s.to_dict() for s in sensors), key=lambda d: str(d["node"])))


def setups_fingerprint(setups: dict[str, FunctionSetup] | None) -> str:
    """SHA-256 of the per-function setups (frame, normalisation, quantity, unit, probe, DOF, reference force)."""
    return _digest({k: v.to_dict() for k, v in sorted((setups or {}).items())})


def damping_fingerprint(study: Any) -> str:
    """SHA-256 of the damping of the dynamics analysis (kind, ratio, table ...)."""
    d = getattr(getattr(study, "dynamics", None), "damping", None)
    return _digest(asdict(d) if d is not None and is_dataclass(d) and not isinstance(d, type) else {})


def curve_fingerprint(p: Predicted) -> str:
    """SHA-256 of a predicted curve: what it is (kind, quantity, frame, normalisation) and its samples (single precision: a curve that changes in the 7th digit is the same)."""
    h = hashlib.sha256()
    h.update(json.dumps([p.kind, p.quantity, p.frame, p.per]).encode("ascii"))
    h.update(np.ascontiguousarray(np.asarray(p.freq), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(np.asarray(p.values), dtype=np.complex64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ evaluation
@dataclass
class MetricRow:
    item: str
    metric: str
    value: float | None
    relation: str
    limit: float | None
    unit: str
    status: str
    limit_key: str
    note: str = ""
    indicative_outside: bool = False        # an INDICATIVE value (SEREP) that lies outside the limit: the row stays UNCONFIRMED, the number is a warning hint

    def to_dict(self) -> dict[str, Any]:
        d = {"item": self.item, "metric": self.metric, "value": self.value, "relation": self.relation, "limit": self.limit, "unit": self.unit, "status": self.status,
             "limit_key": self.limit_key, "note": self.note}
        if self.indicative_outside:
            d["indicative_outside"] = True
        return d


def _row(item: str, metric: str, value: float | None, key: str, limits: CorrelationLimits, absolute: bool = False, hold: str = "") -> MetricRow:
    unit, rel, _ = LIMIT_INFO[key]
    lim = limits.value(key)
    v = None if value is None or not math.isfinite(value) else float(value)
    if limits.is_not_applicable(key):
        e = limits.entries[key]
        return MetricRow(item, metric, v, rel, None, unit, NOT_APPLICABLE, key, f"declared not applicable ({e.document} {e.revision}, {e.clause})")
    if v is None:
        return MetricRow(item, metric, None, rel, lim, unit, FAIL, key, "the metric could not be computed")
    if hold:
        return MetricRow(item, metric, v, rel, lim, unit, UNCONFIRMED, key, hold)
    if lim is None:
        why = limits.entries[key].reason or "not usable"
        return MetricRow(item, metric, v, rel, None, unit, UNCONFIRMED, key, f"limit {key}: {why}")
    x = abs(v) if absolute else v
    ok = x >= lim if rel == ">=" else x <= lim
    return MetricRow(item, metric, v, rel, lim, unit, PASS if ok else FAIL, key)


INDICATIVE_OUTSIDE_NOTE = "the indicative value lies outside the limit: a warning hint only, it does not fail the row"


def _orth_row(item: str, metric: str, value: float | None, key: str, limits: CorrelationLimits, hold: str, why: str, orth_hold: str = "", indicative: bool = False) -> MetricRow:
    """A row of the orthogonality checks / modulus difference: a value that was not computed is UNCONFIRMED with the reason (never PASS, never a missing row); a value that exists but
    must not be judged (`hold`: the UFF rotation convention; `orth_hold`: the reduced mass is not trustworthy for it, or its value is only indicative) is UNCONFIRMED with the number shown;
    a criterion that the user declared not applicable is N/A. An `indicative` value (SEREP) never makes a FAIL: when it lies outside the limit the row says so as a hint (`indicative_outside`)."""
    if value is None and not limits.is_not_applicable(key):
        unit, rel, _ = LIMIT_INFO[key]
        return MetricRow(item, metric, None, rel, limits.value(key), unit, UNCONFIRMED, key, why)
    row = _row(item, metric, value, key, limits, hold="; ".join(x for x in (hold, orth_hold) if x))
    if indicative and row.status == UNCONFIRMED and row.value is not None and row.limit is not None:
        if _row(item, metric, value, key, limits).status == FAIL:                  # what the number would be judged to without the hold: a hint, not a verdict
            row.indicative_outside = True
            row.note = (row.note + "; " if row.note else "") + INDICATIVE_OUTSIDE_NOTE            # "; " (not " - "): the Czech translation splits the note there and finds both parts whole
    return row


def _orthogonality_rows(modal: ModalCorrelation, limits: CorrelationLimits, hold: str) -> list[MetricRow]:
    """The rows of the mass-weighted cross-orthogonality (diagonal and off-diagonal terms of the pairs) and of the auto-orthogonality of the test modes. When the reduced mass is not
    available every row is UNCONFIRMED with the reason; a test mode without a complete shape gets UNCONFIRMED rows; every row of the INDICATIVE SEREP route (`OrthResult.indicative`), and a row
    whose reduced mass does not fit / gives its shape no mass (a TAM: `OrthResult.hold`, `mode_hold`), is UNCONFIRMED with the number shown."""
    o = modal.orth
    why = "; ".join(o.problems) if o.problems else ""
    rows: list[MetricRow] = []
    paired = [p.test_index for p in modal.pairs]
    for p in modal.pairs:
        item = f"test mode {p.test_number} <-> FE mode {p.fe_number}"
        miss = why or "no COC for this pair: its test mode has a DOF that was not measured, or the value could not be computed (see the notes)"
        rows.append(_orth_row(item, "COC diagonal", p.coc, "corr_coc_diag_min", limits, hold, miss, o.hold_of([p.test_index]), bool(o.indicative)))
        if len(modal.pairs) >= 2:
            rows.append(_orth_row(item, "COC off-diagonal", p.coc_offdiag, "corr_coc_offdiag_max", limits, hold, miss, o.hold_of(paired), bool(o.indicative)))
    if len(modal.test_numbers) >= 2:
        complete = [i for i in range(len(modal.test_numbers)) if o.aoc is not None and bool(np.isfinite(o.aoc[i, i]))]
        for i, n in enumerate(modal.test_numbers):
            val: float | None = None
            why_i = why
            if o.aoc is not None and i in complete:
                others = [float(o.aoc[i, u]) for u in range(o.aoc.shape[1]) if u != i and u in complete]
                lost = [modal.test_numbers[u] for u in range(o.aoc.shape[1]) if u != i and u not in complete]
                if lost:
                    why_i = (f"the AOC of test mode {n} is not given because test mode(s) {', '.join(str(x) for x in lost[:6])}{' ...' if len(lost) > 6 else ''} have a DOF that was not measured "
                             "(the largest off-diagonal term cannot be taken without them)")
                elif others and all(math.isfinite(v) for v in others):
                    val = max(others)
            elif o.aoc is not None:
                why_i = f"no AOC for test mode {n}: it has a DOF that was not measured"
            rows.append(_orth_row(f"test mode {n}", "AOC off-diagonal", val, "corr_aoc_offdiag_max", limits, hold, why_i or "no AOC could be computed", o.hold_of(complete), bool(o.indicative)))
    return rows


def evaluate(modal: ModalCorrelation | None, responses: list[ResponseMetrics], limits: CorrelationLimits, issues: list[ResponseIssue] | None = None,
             modal_hold: str = "") -> list[MetricRow]:
    """One row per metric and item, judged against the limits of the confirmed requirement set (see the module docstring). `modal_hold` = why the rows of the modal comparison cannot be
    judged (the shapes at rotated sensors depend on an unconfirmed convention): they are UNCONFIRMED."""
    rows: list[MetricRow] = []
    if modal is not None:
        for n in modal.unpaired_test:
            rows.append(MetricRow(f"test mode {n}", "pairing", None, "", None, "", UNCONFIRMED if modal_hold else FAIL, "",
                                  modal_hold or "no FE mode could be paired with this test mode"))
        for p in modal.pairs:
            item = f"test mode {p.test_number} <-> FE mode {p.fe_number}"
            rows.append(_row(item, "MAC", p.mac, "corr_mac_min", limits, hold=modal_hold))
            rows.append(_row(item, "frequency error", p.freq_error, "corr_freq_error_max", limits, absolute=True, hold=modal_hold))
            if p.offdiag is not None:
                rows.append(_row(item, "MAC off-diagonal", p.offdiag, "corr_mac_offdiag_max", limits, hold=modal_hold))
            md_row = _orth_row(item, "modulus difference", p.modulus_diff, "corr_modulus_diff_max", limits, modal_hold, "the modulus difference could not be computed (fewer than two measured DOFs)")
            md_row.note = md_row.note or "the program's own definition of the modulus difference (the standard gives no formula)"
            rows.append(md_row)
        rows += _orthogonality_rows(modal, limits, modal_hold)
        for k, v in modal.comac.items():
            if v is not None:
                rows.append(_row(f"DOF {k}", "CoMAC", v, "corr_comac_min", limits, hold=modal_hold))
    for r in responses:
        item = f"response {r.name or (r.node + ':' + r.dof)}"
        hold = "; ".join(r.unconfirmed)
        rows.append(_row(item, "peak frequency shift", r.peak_shift, "corr_peak_freq_shift_max", limits, absolute=True, hold=hold))
        rows.append(_row(item, "peak amplitude error", r.peak_error_db, "corr_peak_amp_error_max_db", limits, absolute=True, hold=hold))
        rows.append(_row(item, "RMS error", r.rms_error, "corr_rms_error_max", limits, hold=hold))
    for i in issues or []:
        rows.append(MetricRow(f"response {i.name}", "response comparison", None, "", None, "", UNCONFIRMED, "", "; ".join(i.problems)))
    return rows


def verdict_of(rows: list[MetricRow], complete: bool = True) -> str:
    """FAIL if any row failed; UNCONFIRMED_REQUIREMENTS if any row lacks a usable limit / a confirmed comparison or nothing was judged; PARTIAL_SCOPE if every row passed but part of
    the data was left out (`complete` False); PASS only when every judged row passed and nothing was left out. Rows declared not applicable are not judged."""
    judged = [r for r in rows if r.status != NOT_APPLICABLE]
    if any(r.status == FAIL for r in judged):
        return FAIL
    if not judged or any(r.status == UNCONFIRMED for r in judged):
        return VERDICT_UNCONFIRMED
    if not complete:
        return VERDICT_PARTIAL
    return PASS


# ------------------------------------------------------------------ the whole correlation
@dataclass
class CorrelationResult:
    modal: ModalCorrelation | None
    responses: list[ResponseMetrics]
    rows: list[MetricRow]
    verdict: str
    limits: CorrelationLimits
    options: CorrelationOptions
    source: str = ""
    warnings: list[str] = field(default_factory=list)
    issues: list[ResponseIssue] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    test_sha256: list[str] = field(default_factory=list)
    fe_fingerprint: str = ""
    setups: dict[str, dict[str, Any]] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)            # fingerprints of the other inputs of the comparison (`correlate`, `corr_run.run_correlation`)

    def provenance(self) -> dict[str, Any]:
        """What the result was computed from: the limits, the test data (file and content), the FE modes, the sensors, the per-function setups, the model curves (the dynamics
        results actually used), the damping table and the rotation convention. A stored result is shown as current only while all of it still holds (`corr_run.verify_stored`).
        The provenance is tamper-EVIDENT, not tamper-proof: it detects a result that no longer matches its inputs, it does not authenticate the numbers."""
        counts = {s: sum(1 for r in self.rows if r.status == s) for s in (PASS, FAIL, UNCONFIRMED, NOT_APPLICABLE)}
        return {"limits_fingerprint": self.limits.fingerprint(), "requirement_digest": self.limits.requirement_digest, "scope_digest": self.limits.scope_digest,
                "set_confirmed": self.limits.set_confirmed, "test_sha256": list(self.test_sha256), "fe_fingerprint": self.fe_fingerprint, "row_counts": counts,
                "rows_total": len(self.rows), "complete": bool(self.coverage.get("complete", True)), **self.inputs}

    def to_dict(self, matrices: bool = True) -> dict[str, Any]:
        return {"verdict": self.verdict, "source": self.source, "requirement_set_confirmed": self.limits.set_confirmed, "requirement_set": self.limits.set_name,
                "confirmed_by": self.limits.confirmed_by, "confirmed_via": self.limits.confirmed_via, "verification_approach": self.limits.approach, "options": self.options.to_dict(),
                "modal": self.modal.to_dict(matrices) if self.modal is not None else None, "responses": [r.to_dict() for r in self.responses],
                "response_issues": [i.to_dict() for i in self.issues], "function_setups": dict(self.setups), "coverage": self.coverage, "provenance": self.provenance(),
                "not_applicable_limits": not_applicable_limits(self.limits),
                "rows": [r.to_dict() for r in self.rows], "limits": self.limits.status(), "warnings": self.warnings,
                "note": "No limit of a standard is built into the program: a metric without a usable limit of the confirmed requirement set is UNCONFIRMED, never PASS."}

    def text(self) -> str:
        lines = [f"Test correlation - verdict {self.verdict}"]
        if self.coverage and not self.coverage.get("complete", True):
            lines.append("  scope reduced: " + "; ".join(self.coverage.get("reasons", [])[:6]))
        for na in not_applicable_limits(self.limits):
            lines.append("  NOT APPLICABLE (not judged): " + not_applicable_text(na))
        if self.modal is not None:
            resid = self.modal.orth.residual
            for p in self.modal.pairs:
                lines.append(f"  test {p.test_number} {p.test_freq:.3f} Hz <-> FE {p.fe_number} {p.fe_freq:.3f} Hz: MAC {p.mac:.4f}, df {100 * p.freq_error:+.2f} %"
                             + (f", COC {p.coc:.4f}" if p.coc is not None else "") + (f", MD {p.modulus_diff:.3f}" if p.modulus_diff is not None else "")
                             + (f", out-of-span {resid[p.test_index]:.3f}" if resid is not None and p.test_index < len(resid) and math.isfinite(float(resid[p.test_index])) else ""))
            if self.modal.orth.problems:
                lines.append("  cross-orthogonality not computed: " + "; ".join(self.modal.orth.problems))
            elif self.modal.orth.hold_of(self.modal.orth.mode_hold):
                lines.append("  COC / AOC UNCONFIRMED: " + self.modal.orth.hold_of(self.modal.orth.mode_hold)[:600])
            if self.modal.unpaired_test:
                lines.append(f"  unpaired test modes: {self.modal.unpaired_test}")
        for r in self.responses:
            lines.append(f"  {r.name}: peak shift {100 * r.peak_shift:+.2f} %, amplitude {r.peak_error_db:+.2f} dB, RMS {100 * r.rms_error:.2f} %" + (" (UNCONFIRMED)" if r.unconfirmed else ""))
        for i in self.issues:
            lines.append(f"  {i.name}: not compared - {'; '.join(i.problems)}")
        return "\n".join(lines)


def _setup_of(setups: dict[str, FunctionSetup] | None, fn: TestFunction) -> FunctionSetup:
    return (setups or {}).get(fn.name) or FunctionSetup()


def _compare_functions(test: TestData, predicted: dict[str, Predicted] | None, setups: dict[str, FunctionSetup] | None, skipped: dict[str, str] | None, opt: CorrelationOptions,
                       warn: list[str]) -> tuple[list[ResponseMetrics], list[ResponseIssue], dict[str, Any], dict[str, str]]:
    """(responses, issues, coverage of the functions, {function name: fingerprint of the model curve it was compared with})."""
    responses: list[ResponseMetrics] = []
    issues: list[ResponseIssue] = []
    info: list[str] = []
    left_out: list[dict[str, str]] = []
    band_limited: list[dict[str, Any]] = []
    curves: dict[str, str] = {}
    rot_hold = set(test.rotation_ambiguous) if not test.rotation_confirmed else set()
    for fn in test.functions:
        if fn.kind not in ("frf", "psd"):
            info.append(fn.name)
            continue
        if fn.excluded:
            left_out.append({"name": fn.name, "reason": fn.excluded})
            continue
        if fn.domain != "frequency":
            left_out.append({"name": fn.name, "reason": "not a function of frequency"})
            warn.append(f"function {fn.name} is not a function of frequency: skipped")
            continue
        setup = _setup_of(setups, fn)
        res = resolve_function(fn, setup)
        if res.problems:                                                  # unknown / unfit unit: the function is reported, never compared
            issues.append(ResponseIssue(fn.name, fn.node, fn.dof, list(res.problems)))
            continue
        pred = (predicted or {}).get(fn.name) or (predicted or {}).get(f"{fn.node}:{fn.dof}")
        if pred is None:
            left_out.append({"name": fn.name, "reason": (skipped or {}).get(fn.name) or "no model curve (no probe / dynamics response for it)"})
            continue
        hard: list[str] = []
        if not pred.quantity:
            hard.append("the model curve does not say what it is (quantity / unit): it cannot be compared")
        else:
            if pred.quantity != res.quantity:
                hard.append(f"the measured function is a {res.quantity} but the model curve is a {pred.quantity}")
            if pred.per != res.per:
                hard.append(f"the measured function is normalised '{res.per}' but the model curve '{pred.per}'")
            if (pred.kind == "psd") != res.psd:
                hard.append("a PSD is compared only with a PSD")
        if hard:
            issues.append(ResponseIssue(fn.name, fn.node, fn.dof, hard))
            continue
        assert res.unit is not None
        soft: list[str] = []
        if pred.frame and setup.frame != pred.frame:
            soft.append("the response frame (absolute / relative to the base) of the measurement was not chosen" if not setup.frame
                        else f"the measurement is {setup.frame} but the model curve is {pred.frame}")
        if fn.axis is not None and fn.node in rot_hold:
            soft.append(f"the local sensor system of node {fn.node} was read with the {test.rotation} convention of dataset 2420, which the user has not confirmed for this file")
        scale = _unit_scale(res.unit, res.quantity, res.per, res.psd)
        try:
            rm = compare_response(fn.freq, np.asarray(fn.values) * scale, pred.freq, pred.values, fn.kind, opt.response_band_hz, fn.name, fn.node, fn.dof, opt.hidden_peak_db)
        except ValueError as e:
            warn.append(f"function {fn.name}: {e}")
            left_out.append({"name": fn.name, "reason": str(e)})
            continue
        rm.quantity, rm.unit = res.quantity, ("" if res.psd else fe_unit_label(res.quantity, res.per))
        rm.unconfirmed = soft
        warn += [f"function {fn.name}: {w}" for w in rm.warnings]
        responses.append(rm)
        curves[fn.name] = curve_fingerprint(pred)
        if rm.hidden_peaks_hz:
            band_limited.append({"name": fn.name, "band_Hz": [float(rm.band_hz[0]), float(rm.band_hz[1])], "measured_band_Hz": [float(rm.measured_band_hz[0]), float(rm.measured_band_hz[1])],
                                 "test_peaks_outside_band_Hz": [float(v) for v in rm.hidden_peaks_hz]})
    return responses, issues, {"functions_total": len(test.functions), "functions_compared": len(responses), "functions_unconfirmed": [i.name for i in issues] + [r.name for r in responses if r.unconfirmed],
                               "functions_info_only": info, "functions_left_out": left_out, "functions_band_limited": band_limited}, curves


def fe_unit_label(quantity: str, per: str) -> str:
    base = {"displacement": "mm", "velocity": "mm/s", "acceleration": "g"}[quantity]
    return base + {"none": "", "base_acceleration": "/g", "force": "/N"}[per]


def _tam_model_key(mass: OrthMass | None) -> str:
    """The state of the model (`model_fingerprint`) the imported TAM was bound to AND matches at this correlation ("" when it was held because it does not: nothing is then tied)."""
    if mass is None or mass.kind != "tam" or mass.tam is None or mass.model_hold:
        return ""
    return str(mass.tam.model_key or "")


def correlate(test: TestData, modeset: Any = None, mesh: Any = None, sensors: list[Sensor] | None = None, *, limits: CorrelationLimits | None = None,
              options: CorrelationOptions | None = None, predicted: dict[str, Predicted] | None = None, setups: dict[str, FunctionSetup] | None = None,
              skipped: dict[str, str] | None = None, orth_mass: OrthMass | None = None) -> CorrelationResult:
    """Correlates a test dataset with the FE modes (`modeset`: an `eigen.ModeSet` of kind modal, `mesh` its mesh) and with predicted response curves (`predicted`:
    {function name: Predicted}; `setups`: {function name: FunctionSetup}; `skipped`: {function name: why it has no predicted curve}). Without limits (or with an unconfirmed
    requirement set) the numbers are computed and every row is UNCONFIRMED."""
    opt = options or CorrelationOptions()
    opt.validate()
    lim = limits or CorrelationLimits()
    snap = (test.fingerprint(), test.rotation, bool(test.rotation_confirmed))        # what the shapes are read from: a change while the correlation runs makes the result void (checked at the end)
    warn: list[str] = list(test.warnings)
    modal: ModalCorrelation | None = None
    sens = sensors or []
    cov: dict[str, Any] = {"test_modes_total": len(test.modes), "test_modes_used": 0, "test_modes_left_out": [], "fe_modes_total": 0, "fe_modes_used": 0, "nodes_in_data": len(test.nodes()),
                           "nodes_used": [], "nodes_unmapped": [], "nodes_unresolved": dict(test.unresolved), "dofs_used": 0, "dofs_zero_excluded": [], "options": opt.to_dict()}
    reasons: list[str] = []
    fe_fp = ""
    if test.modes and modeset is not None:
        if getattr(modeset, "kind", "modal") != "modal":
            raise ValueError("the FE mode set is not a set of natural modes")
        by_node = {s.node: s for s in sens}
        tmodes = [m for m in test.modes if (opt.f_min_hz is None or m.frequency_hz >= opt.f_min_hz) and (opt.f_max_hz is None or m.frequency_hz <= opt.f_max_hz)]
        if not tmodes:
            raise ValueError("no test mode lies in the frequency range")
        used_ids = {id(m) for m in tmodes}
        out_modes = [m for m in test.modes if id(m) not in used_ids]
        if out_modes:
            cov["test_modes_left_out"] = [{"mode": m.number, "f_Hz": m.frequency_hz, "reason": "outside the frequency window"} for m in out_modes]
            reasons.append(f"{len(out_modes)} of {len(test.modes)} test modes lie outside the frequency window and were not correlated (modes {', '.join(str(m.number) for m in out_modes[:12])}"
                           f"{' ...' if len(out_modes) > 12 else ''})")
        dofs: list[tuple[str, str]] = []
        unmapped: dict[str, None] = {}
        for node in dict.fromkeys(n for m in tmodes for n, _ in m.shape):
            if node not in by_node:
                unmapped[node] = None
        for s in sens:
            for d in s.dofs:
                if any((s.node, d) in m.shape for m in tmodes):
                    dofs.append((s.node, d))
        omitted: dict[str, bool] = {}                                    # "node:dof" -> measured (not exactly zero) in some mode: a DOF of a mapped node that is not compared
        for m in tmodes:
            for (n, d), v in m.shape.items():
                sn = by_node.get(n)
                if sn is None or (d in TRANSLATIONS and d in sn.dofs):
                    continue
                omitted[f"{n}:{d}"] = omitted.get(f"{n}:{d}", False) or bool(np.isfinite(abs(v)) and abs(v) > 0)
        omitted_dofs = [k for k, measured in omitted.items() if measured]
        if omitted_dofs:
            names = ", ".join(omitted_dofs[:10]) + (" ..." if len(omitted_dofs) > 10 else "")
            cov["dofs_omitted"] = omitted_dofs
            warn.append(f"{len(omitted_dofs)} measured DOF(s) of mapped test nodes are not compared: {names}")
            reasons.append(f"{len(omitted_dofs)} measured test DOF(s) are not compared - the sensor maps fewer DOFs than the test measured, or the DOF is a rotation (rotations are never compared): {names}")
        if unmapped:
            names = ", ".join(list(unmapped)[:8]) + (" ..." if len(unmapped) > 8 else "")
            warn.append(f"test nodes without a sensor position are not used: {names}")
            cov["nodes_unmapped"] = list(unmapped)
            reasons.append(f"{len(unmapped)} test node(s) have no sensor position and were left out: {names}")
        if test.unresolved:
            reasons.append(f"{len(test.unresolved)} test node(s) with a local coordinate system that is not fully known were left out: " + ", ".join(list(test.unresolved)[:8]))
        if not dofs:
            raise ValueError("no test DOF is mapped to the FE model: define the sensors (node -> point / probe)")
        tphi = np.full((len(dofs), len(tmodes)), np.nan, dtype=complex)
        for k, m in enumerate(tmodes):
            for d, key in enumerate(dofs):
                if key in m.shape:
                    tphi[d, k] = m.shape[key]
        keep = []
        for d, key in enumerate(dofs):                                   # a DOF that is exactly zero in every mode is no measurement (a uniaxial sensor padded with zeros)
            row = tphi[d]
            fin = np.isfinite(row.real) & np.isfinite(row.imag)
            if fin.any() and not np.any(row[fin] != 0):
                cov["dofs_zero_excluded"].append(f"{key[0]}:{key[1]}")
            else:
                keep.append(d)
        if cov["dofs_zero_excluded"]:
            names = ", ".join(cov["dofs_zero_excluded"][:8]) + (" ..." if len(cov["dofs_zero_excluded"]) > 8 else "")
            warn.append(f"{len(cov['dofs_zero_excluded'])} test DOF(s) are exactly zero in every mode (unmeasured axes written as zeros by the file) and were left out of the MAC: {names}")
        if not keep:
            raise ValueError("every mapped test DOF is exactly zero in all modes")
        dofs, tphi = [dofs[d] for d in keep], tphi[keep]
        if not np.iscomplexobj(tphi) or not np.any(tphi.imag != 0):
            tphi = tphi.real
        fe_idx = [k for k, f in enumerate(modeset.values) if (opt.fe_modes is None or (k + 1) in opt.fe_modes)
                  and (opt.f_max_hz is None or f <= opt.f_max_hz) and (opt.f_min_hz is None or f >= opt.f_min_hz)]
        if not fe_idx:
            raise ValueError("no FE mode lies in the frequency range")
        fphi, w2 = fe_shapes_at_sensors(modeset, mesh, sens, dofs, fe_idx)
        warn += w2
        axes_hold = ""
        rotated = sorted((n for n in dict.fromkeys(n for n, _ in dofs) if n in test.cs_matrices and not np.allclose(test.cs_matrices[n], np.eye(3))), key=lambda x: (len(x), x))
        if orth_mass is not None and orth_mass.kind == "tam" and rotated and not opt.tam_global_axes:
            axes_hold = (f"the test data have rotated sensor systems (UFF 2420: node(s) {', '.join(rotated[:6])}{' ...' if len(rotated) > 6 else ''}) and their shapes are in the GLOBAL axes of the FE model, "
                         "but nobody stated that the imported mass matrix (TAM) is in global axes too (a TAM written in the local sensor axes is a different matrix)")
        orth = orthogonality(tphi, fphi, dofs, orth_mass, modeset, fe_idx, [m.number for m in tmodes], axes_hold)
        warn += orth.warnings + orth.notes
        modal = modal_correlation(np.array([m.frequency_hz for m in tmodes]), tphi, np.array([float(modeset.values[k]) for k in fe_idx]), fphi, dofs,
                                  [m.number for m in tmodes], [k + 1 for k in fe_idx], [m.damping for m in tmodes], opt, orth)
        warn += modal.warnings
        cov.update({"test_modes_used": len(tmodes), "fe_modes_total": len(modeset.values), "fe_modes_used": len(fe_idx), "nodes_used": list(dict.fromkeys(n for n, _ in dofs)), "dofs_used": len(dofs)})
        fe_fp = fe_fingerprint(modeset, mesh)
    elif test.modes:
        warn.append("no FE modes were given: the modal metrics were not computed")
        reasons.append(f"no FE modes were given: the {len(test.modes)} test mode(s) were not correlated")
    responses, issues, fcov, curves = _compare_functions(test, predicted, setups, skipped, opt, warn)
    cov.update(fcov)
    if fcov["functions_left_out"]:
        reasons += [f"function {x['name']} was not compared: {x['reason']}" for x in fcov["functions_left_out"]]
    for b in fcov["functions_band_limited"]:
        pk = ", ".join(f"{v:g}" for v in b["test_peaks_outside_band_Hz"][:8]) + (" ..." if len(b["test_peaks_outside_band_Hz"]) > 8 else "")
        reasons.append(f"function {b['name']}: test resonance(s) at {pk} Hz lie outside the compared band {b['band_Hz'][0]:g} - {b['band_Hz'][1]:g} Hz (the response band option, or a model curve "
                       f"shorter than the measurement) and were not compared; the measurement covers {b['measured_band_Hz'][0]:g} - {b['measured_band_Hz'][1]:g} Hz")
    amb = set(test.rotation_ambiguous)
    held = sorted(amb & set(cov["nodes_used"]), key=lambda x: (len(x), x)) if not test.rotation_confirmed else []
    cov["rotation"] = {"convention": test.rotation, "confirmed": bool(test.rotation_confirmed), "confirmed_via": test.rotation_confirmed_via, "ambiguous_nodes": sorted(amb, key=lambda x: (len(x), x)),
                       "rows_held_for_nodes": held}
    modal_hold = ""
    if held:
        modal_hold = (f"the shapes at the rotated sensors ({', '.join(held[:6])}{' ...' if len(held) > 6 else ''}) depend on the {test.rotation} convention of dataset 2420, "
                      "which the user has not confirmed for this file")
        warn.append(f"UFF 2420 convention: {modal_hold} - the modal rows are UNCONFIRMED")
    if test.functions and not responses and not issues:
        warn.append("no measured function was compared (give `predicted` or run the dynamics response with probes named like the sensors)")
    cov["reasons"] = reasons
    cov["complete"] = not reasons
    if (test.fingerprint(), test.rotation, bool(test.rotation_confirmed)) != snap:
        raise ValueError("the test data or their rotation convention changed while the correlation was running: the result would mix two states - correlate again")
    rows = evaluate(modal, responses, lim, issues, modal_hold)
    n_out = sum(1 for r in rows if r.indicative_outside)
    if n_out:
        warn.append(f"{n_out} indicative SEREP COC / AOC value(s) lie outside their limit (hint only: these rows stay UNCONFIRMED, a PASS or FAIL needs the TAM mass matrix of the test lab)")
    inputs = {"sensors_fingerprint": sensors_fingerprint(sens), "setups_fingerprint": setups_fingerprint(setups), "curves": curves, "curves_source": "given", "test_fingerprint": snap[0],
              "orth_mass": {"source": modal.orth.source if modal is not None else "none", "fingerprint": modal.orth.fingerprint if modal is not None else "", "computed": bool(modal is not None and modal.orth.computed),
                             "indicative": bool(modal is not None and modal.orth.indicative),
                             "model_fingerprint": _tam_model_key(orth_mass) if modal is not None and modal.orth.source == "tam" else ""},
              "rotation": {"convention": test.rotation, "confirmed": bool(test.rotation_confirmed), "ambiguous_nodes": sorted(amb, key=lambda x: (len(x), x))}}
    return CorrelationResult(modal, responses, rows, verdict_of(rows, bool(cov["complete"])), lim, opt, test.source, list(dict.fromkeys(warn)), issues, cov, list(test.sha256), fe_fp,
                             {k: v.to_dict() for k, v in (setups or {}).items()}, inputs)

"""Report subsection "Vibration test correlation" (F14 stage S5, `fea_core.space.correlation`): the stored correlation result of the study with the provenance of every limit it
was judged against. Written only when the study carries a stored result (`Study.space.extra["correlation"]`); the measured data themselves are not stored, only the numbers.

The verdict printed is NEVER the stored one on its own word: `corr_run.verify_stored` compares every input of the comparison (limits, sensors, function setups, damping table, test
data, FE modes, model curves) with the provenance stored with the result; a result that is stale, was edited by hand or has no provenance is printed as UNCONFIRMED_REQUIREMENTS /
OUTDATED, and a result whose inputs are not at hand (no solved results, no test data in this session) as UNVERIFIABLE - with the reasons. The report says plainly that the numbers are a
stored result of an earlier computation (the provenance is tamper-evident, not a recomputation) and what was verified.
The options, the coverage (what was correlated, what was left out) and every warning that affects the verdict are printed in full."""
from __future__ import annotations

from typing import Any

from .word import num

MAX_MATRIX = 12                 # the MAC matrix is printed as a table up to this many modes per side
MAX_ROWS = 120
MAX_PLAIN_WARNINGS = 40         # warnings that do not affect the verdict
_VIA = {"gui": "GUI (the user)", "mcp": "MCP interface (an agent) - to be checked by the user"}
_VIA_ENTERED = {"gui": "GUI", "mcp": "MCP interface (an agent)", "import": "an imported file"}


def stored_result(study: Any) -> dict[str, Any] | None:
    sp = getattr(study, "space", None)
    rec = sp.extra.get("correlation") if sp is not None else None
    res = rec.get("result") if isinstance(rec, dict) else None
    return res if isinstance(res, dict) and res.get("verdict") else None


def has_correlation(study: Any) -> bool:
    return stored_result(study) is not None


def _list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _pct(v: Any) -> str:
    return "–" if v is None else num(100.0 * float(v), 3)


def _txt(v: Any, n: int = 200) -> str:
    return "".join(ch if ch.isprintable() else " " for ch in str(v))[:n]


def _fmt(v: Any, n: int = 4) -> str:
    return "–" if v is None else num(v, n)


def _residual_of(modal: dict[str, Any], resid: list[Any] | None, pair: dict[str, Any]) -> Any:
    """The out-of-span residual of the test mode of a pair (SEREP; None for another mass source)."""
    try:
        return resid[list(modal.get("test_modes") or []).index(pair["test_mode"])] if resid else None
    except (ValueError, IndexError, KeyError):
        return None


def _is_orth_row(r: Any) -> bool:
    return isinstance(r, dict) and str(r.get("metric", "")).startswith(("COC", "AOC"))


def _serep_shown_status(src: str, r: dict[str, Any]) -> str:
    """The status a criteria row is printed with: a COC / AOC row of the SEREP route is never PASS or FAIL (a record of an earlier build may hold one: it is shown UNCONFIRMED)."""
    st = str(r.get("status", ""))
    return "UNCONFIRMED" if src == "serep" and _is_orth_row(r) and st in ("PASS", "FAIL") else st


def _orthogonality(D: Any, modal: dict[str, Any], pairs: list[dict[str, Any]], affecting: list[str], rows: list[Any] | None = None) -> None:
    """The cross-orthogonality / auto-orthogonality / modulus difference of the paired modes and where the reduced mass came from. `rows` = the stored criteria rows (the INDICATIVE notice
    says what the COC / AOC rows are: UNCONFIRMED, or not judged when the user declared every COC / AOC criterion not applicable)."""
    o = modal.get("orthogonality")
    if not isinstance(o, dict):
        D.p("This stored result has no cross-orthogonality (it was computed by an older version): correlate again.", bold=True)
        return
    src = _txt(o.get("source"), 12)
    if o.get("computed"):
        info = o.get("info") or {}
        extra = (f"; {info.get('fe_modes')} compared FE modes (the SEREP basis) at {info.get('sensor_dofs')} sensor DOFs, condition number of the mass-normalised shape matrix "
                 f"{num(info.get('condition'), 3)}, largest off-diagonal of the FE modes under the mass of the study {num(info.get('fe_mass_orthogonality_error'), 3)} (limit 1e-3), largest "
                 f"out-of-span residual of a test mode {_fmt(info.get('residual_max'), 3)} (limit {_fmt(info.get('residual_limit'), 3)})" if src == "serep" else
                 f"; {info.get('sensor_dofs')} DOFs" + (f", unit {_txt(info.get('unit'), 12)}" if info.get("unit") else "") + (f", largest off-diagonal of the FE modes through this matrix {_fmt(info.get('fe_offdiagonal_max'), 3)}" if src == "tam" else "")
                 + (f", rank {info.get('rank')} of {info.get('sensor_dofs')}, largest share of a test mode in the null space {_fmt(info.get('residual_max'), 3)}"
                    if src == "tam" and info.get("residual_max") is not None else "")
                 + (f", largest deviation of the generalised masses of the FE modes through this matrix from those of the study (after one global factor) {_fmt(info.get('genmass_dev_max'), 3)} "
                    f"(limit {_fmt(info.get('genmass_dev_limit'), 3)})" if src == "tam" and info.get("genmass_checked") else
                    ", the generalised masses of the FE modes were NOT compared with the study" if src == "tam" else ""))
        D.p(f"Cross-orthogonality (ECSS-E-ST-32-11C 3.2.39, 5.8.1.3 b): the mass is reduced to the sensor DOFs - {_txt(o.get('mass'), 120)}{extra}; fingerprint {_txt(o.get('fingerprint'), 16)}… "
            "COC = |phi_test^T M phi_FE| / sqrt((phi_test^T M phi_test)(phi_FE^T M phi_FE)) (not squared, as in the standard; the magnitude, a mode shape has an arbitrary sign); "
            "AOC is the same between test modes. Pairs are those of the MAC.", size=9)
        if src == "tam":
            D.p("What the program checks of an imported mass matrix (TAM), and what it does not: the DOFs are the sensor DOFs, the matrix is a mass matrix (symmetric, positive semi-definite to the "
                "rounding of its digits), it reproduces the orthogonality of the FE modes (off-diagonal terms), its generalised masses of the FE modes equal those of the study up to one global "
                "factor, and the model it was bound to is unchanged. A matrix that passes all that can still be wrong for other shapes (the part of a shape outside the FE modes). The comparison of the "
                "TAM modes with the reference model (ECSS-E-ST-32-11C 6.3.2 b) needs the stiffness matrix of the TAM and is NOT done. The mass fidelity of the TAM is the responsibility of the "
                "test laboratory.", size=9, italic=True)
        if src == "serep":                                       # whatever the record says (a record of an earlier build has no indicative flag): the SEREP COC / AOC is always indicative
            orth_rows = [r for r in (rows or []) if _is_orth_row(r)]
            judged = [r for r in orth_rows if r.get("status") != "N/A"]
            reason = _txt(o.get("indicative") or "indicative (SEREP reduction from the FE modes) - a PASS needs a test-analysis-model (TAM) mass matrix from the test lab (ECSS-E-ST-32-11C 5.8.1.3 b, 6.3)", 400)
            if orth_rows and not judged:                         # the user declared every COC / AOC criterion not applicable: no row is UNCONFIRMED, none is judged
                D.p("INDICATIVE ONLY. The COC / AOC of this result come from a SEREP reduction that the program computed from the FE modes (" + reason + "). The user declared every COC / AOC "
                    "criterion not applicable, so no COC / AOC row is judged and the numbers are shown for information only: they are the COC of the PROJECTION of each test shape onto the "
                    "compared FE modes at the sensors, not of the shape (see the out-of-span residual below). Import the reduced mass matrix of the test laboratory (TAM) to judge them.",
                    size=9, italic=True)
            else:
                D.p("INDICATIVE ONLY. The COC / AOC of this result come from a SEREP reduction that the program computed from the FE modes, and every COC / AOC row is UNCONFIRMED whatever its value: "
                    + reason
                    + ". An indicative value never makes a FAIL either: a value outside the limit is a warning hint. Why: the SEREP COC is the COC of the PROJECTION of each test shape onto the compared "
                    "FE modes at the sensors (q = Phi_a^+ psi, mass M_a = (Phi_a^+)^T G Phi_a^+ with G = Phi^T M Phi under the mass of the study; the FE modes are exactly orthogonal in M_a by "
                    "construction, so only the COC of the test modes carries information). A part of a test shape that the compared modes cannot represent is invisible to it, and the review of the "
                    "method found shapes that pass through the projection and fail with the full mass matrix (a heavy tip mass, a shape error at a point the compared modes hardly move, uniaxial "
                    "sensors that resolve some modes only). The out-of-span residual r = 1 - |Phi_a q|^2 / |psi|^2 of every test mode (Euclidean norm over the sensor DOFs) is printed below as a measure of "
                    "that, but it does not bound the mass-weighted error. The MAC, the frequency difference and the modulus difference are not affected. Import the reduced mass matrix of the test "
                    "laboratory (TAM) for a PASS / FAIL of the COC / AOC.", size=9, italic=True)
    else:
        D.p("The cross-orthogonality and the auto-orthogonality were NOT computed: " + "; ".join(_txt(x, 300) for x in _list(o.get("problems"))) + ". Their rows are UNCONFIRMED (Table 5-2 makes the technique mandatory).", bold=True)
        affecting.append("cross-orthogonality not computed: " + "; ".join(str(x) for x in _list(o.get("problems"))))
    if o.get("hold") or o.get("mode_hold"):
        held = [str(o.get("hold") or "")] + [str(v) for v in (o.get("mode_hold") or {}).values()]
        D.p("The COC / AOC rows are UNCONFIRMED although the numbers exist: " + "; ".join(_txt(x, 400) for x in held if x), bold=True)
        affecting.append("cross-orthogonality held: " + "; ".join(_txt(x, 200) for x in held if x))
    for n in _list(o.get("notes")):
        D.bullet(_txt(n, 300))
    resid = o.get("residual") if isinstance(o.get("residual"), list) else None
    ind = " (indicative)" if src == "serep" else ""
    if pairs:
        D.table(["Test mode", "FE mode", "COC" + ind, "Max. COC with another pair" + ind, "Max. AOC with another test mode" + ind, "Modulus difference (program's own definition)", "DOF of the maximum", "Out-of-span residual"],
                [[p["test_mode"], p["fe_mode"], _fmt(p.get("coc")), _fmt(p.get("coc_offdiag_max"), 3), _fmt(p.get("aoc_offdiag_max"), 3), _fmt(p.get("modulus_difference"), 3), _txt(p.get("modulus_dof"), 40) or "–",
                  _fmt(_residual_of(modal, resid, p), 3)]
                 for p in pairs[:MAX_ROWS]], caption="Cross-orthogonality and modulus difference of the paired modes", font=8)
        D.p("Modulus difference (Table 5-2; the standard names it and gives no formula - this is the PROGRAM'S OWN definition, not the standard's): both shapes are scaled to unit length on the "
            "measured DOFs and the value is the largest | |test| - |FE| | over the DOFs as a fraction of the largest test amplitude.", italic=True, size=9)
    coc, tn, fn = o.get("coc"), modal.get("test_modes") or [], modal.get("fe_modes") or []
    if coc and len(tn) <= MAX_MATRIX and len(fn) <= MAX_MATRIX:
        D.table(["Test \\ FE"] + [f"FE {n}" for n in fn], [[f"test {n}"] + [_fmt(v, 3) for v in row] for n, row in zip(tn, coc)], caption="COC matrix (rows: test modes, columns: FE modes)", font=8)


def correlation_section(D: Any, study: Any, sec: str = "", results: Any = None, status: dict[str, Any] | None = None) -> list[str]:
    """Writes the subsection; returns warnings. `results` (optional) lets the FE modes be compared with the stored fingerprint. `status` (optional) is the result of
    `corr_run.verify_stored(study, results)` of the caller, which has it already (the study and the results must be the same)."""
    from ..space import corr_run as CR
    if status is None:
        status = CR.verify_stored(study, results)
    if status is None:
        return []
    res = status["result"]
    rec = study.space.extra.get("correlation", {})
    verdict = str(status["verdict"])
    D.h(f"{sec}.1 Vibration test correlation" if sec else "Vibration test correlation", 2)
    D.p("The natural modes and the responses of the FE model are compared with a vibration test: MAC of the paired modes, relative frequency error, CoMAC per sensor DOF and "
        "the peak frequency shift, peak amplitude error and RMS error of the response curves. Every metric is judged only against a limit of the requirement set that the user "
        "entered and confirmed (document, revision and clause are printed with it); the program contains no correlation limit of a standard. The cross terms are the "
        "off-diagonal MAC of the paired modes. The mass-weighted cross-orthogonality COC and the auto-orthogonality AOC of the test modes (ECSS-E-ST-32-11C 3.2.39, 5.7.1.4, 5.8.1.3; "
        "Table 5-2 makes them mandatory) are computed with the mass reduced to the sensor DOFs - only an imported test-analysis-model mass matrix (TAM) can judge them against a limit, the SEREP "
        "reduction the program makes from the FE modes is indicative and leaves them UNCONFIRMED - and the modulus difference (Table 5-2) as the program defines it (the standard gives "
        "no formula); the source of the reduced mass is printed below. A response curve is compared only with a model curve of the same quantity, "
        "frame (absolute / relative to the base) and normalisation, in a known unit.")
    ok = verdict == "PASS"
    D.p(f"Verdict of the test correlation: {verdict}", bold=True, size=11, color=(0x1B, 0x7F, 0x3B) if ok else (0xC0, 0x39, 0x2B))
    if status["reasons"]:
        D.p(f"The result stored in the study says {status['stored_verdict']}, but it is not current: " + "; ".join(_txt(r) for r in status["reasons"]) + ". Correlate again.",
            bold=True)
    if status["unverified"]:
        D.p(f"The result stored in the study says {status['stored_verdict']}; it could not be verified now: " + "; ".join(_txt(r) for r in status["unverified"])
            + ". Import the test data, solve the model and correlate again to verify it.", bold=True)
    if status["current"]:
        D.p("This is the stored result of the last correlation of this study. Its provenance is tamper-evident (it shows that the inputs are unchanged), it is not a recomputation: "
            "recompute (Correlate) to verify the numbers themselves.", italic=True, size=9)
    if status["current"] and verdict == "PARTIAL_SCOPE":
        D.p("Every judged row passed, but part of the test data was left out of the correlation (see the coverage below): this is not a PASS of the test.", bold=True)
    elif verdict == "UNCONFIRMED_REQUIREMENTS":
        D.p("This is not an ECSS verification result: a limit, a comparison or the confirmation of the requirement set is missing, so the numbers are indicative only.", italic=True)
    opt = res.get("options") or {}
    cov = res.get("coverage") or {}
    prov = res.get("provenance") or {}
    cur = status["limits"]                                                  # the CURRENT confirmation state of the requirement set (the stored one may be stale)
    cur_via = str(cur.confirmed_via or "")
    kv_conf = [("Requirement set", str(cur.set_name or "–")),
               ("Confirmed by the user (now)", "yes" if cur.set_confirmed else "NO - the requirement set is not confirmed"),
               ("Confirmed by", str(cur.confirmed_by or "–") if cur.set_confirmed else "–"),
               ("Confirmed via", _VIA.get(cur_via, cur_via or "–") if cur.set_confirmed else "–")]
    if bool(res.get("requirement_set_confirmed")) != cur.set_confirmed:
        kv_conf.append(("Confirmation when it was correlated", "yes" if res.get("requirement_set_confirmed") else "NO"))
    rot = (cov.get("rotation") or {}) if isinstance(cov, dict) else {}
    kv_rot = []
    if rot.get("ambiguous_nodes"):
        kv_rot.append(("Local sensor axes (UFF 2420)", f"{rot.get('convention')} of the 3 x 3 part are the local axes - " + ("confirmed by the user"
                       + (f" via {_VIA.get(str(rot.get('confirmed_via') or ''), rot.get('confirmed_via'))}" if rot.get("confirmed_via") else "") if rot.get("confirmed")
                       else "NOT confirmed by the user: the rows at rotated nodes are UNCONFIRMED")))
    kv_tam: list[tuple[str, str]] = []
    if opt.get("orth_mass") == "tam":
        kv_tam.append(("Imported TAM in the global axes of the FE model", ("stated by " + _VIA.get(str(opt.get("tam_global_axes_via") or ""), _txt(opt.get("tam_global_axes_via"), 12) or "the program's own API")
                                                                          if opt.get("tam_global_axes") else "NOT stated (needed only when the test has rotated sensor axes: the rows are then UNCONFIRMED)")))
    kv_ver = [("Verified against the current study", ", ".join(status["verified"]) or "nothing"), ("Not verified", "; ".join(_txt(r, 120) for r in status["unverified"]) or "–")]
    D.kv([("Test data", str(res.get("source") or "–")), ("Test file SHA-256", ", ".join(str(h)[:16] + "…" for h in _list(prov.get("test_sha256"))) or "–")]
         + kv_conf + kv_rot + kv_tam + kv_ver + [
          ("Verification approach", str(res.get("verification_approach") or "–")), ("Mode pairing", str(opt.get("method") or "–")),
          ("Minimum MAC of a pair", num(opt.get("pair_min_mac"))), ("Test-mode frequency window [Hz]", f"{opt.get('f_min_hz') if opt.get('f_min_hz') is not None else '–'} - "
                                                                                                        f"{opt.get('f_max_hz') if opt.get('f_max_hz') is not None else '–'}"),
          ("FE modes selected", ", ".join(str(v) for v in opt["fe_modes"]) if opt.get("fe_modes") else "all"),
          ("Response band [Hz]", " - ".join(num(v) for v in opt["response_band_hz"]) if opt.get("response_band_hz") else "whole common band"),
          ("Test modes correlated", f"{cov.get('test_modes_used', '–')} of {cov.get('test_modes_total', '–')}"),
          ("FE modes in the comparison", f"{cov.get('fe_modes_used', '–')} of {cov.get('fe_modes_total', '–')}"),
          ("Test nodes used / in the data", f"{len(cov.get('nodes_used') or [])} / {cov.get('nodes_in_data', '–')}"), ("Sensor DOFs in the MAC", str(cov.get("dofs_used", "–"))),
          ("Measured functions compared", f"{cov.get('functions_compared', '–')} of {cov.get('functions_total', '–')}"),
          ("Sensors mapped", str(len(rec.get("sensors") or [])))])
    affecting: list[str] = []
    na = res.get("not_applicable_limits")
    if not isinstance(na, list):
        na = [{"limit": k, "document": v.get("document"), "revision": v.get("revision"), "clause": v.get("clause"), "entered_via": v.get("set_via"), "confirmed_by": res.get("confirmed_by"),
               "confirmed_via": res.get("confirmed_via")} for k, v in (res.get("limits") or {}).items() if isinstance(v, dict) and v.get("not_applicable")]
    if na:
        D.p("Criteria declared NOT APPLICABLE (they were not judged; the declaration is part of the requirement set that the user confirmed):", bold=True)
        D.table(["Criterion", "Document", "Revision", "Clause", "Entered via", "Set confirmed by", "Confirmed via"],
                [[_txt(x.get("limit"), 40), _txt(x.get("document"), 60), _txt(x.get("revision"), 30), _txt(x.get("clause"), 60), _VIA_ENTERED.get(str(x.get("entered_via") or ""), "unknown"),
                  _txt(x.get("confirmed_by"), 40) or "–", _VIA.get(str(x.get("confirmed_via") or ""), _txt(x.get("confirmed_via"), 12) or "unknown")] for x in na if isinstance(x, dict)],
                caption="Correlation criteria declared not applicable", font=8)
        affecting += [f"criterion {x.get('limit')} was declared not applicable ({_txt(x.get('document'), 40)} {_txt(x.get('revision'), 20)}, {_txt(x.get('clause'), 40)}; "
                      f"entered {_VIA_ENTERED.get(str(x.get('entered_via') or ''), 'unknown')}) and not judged" for x in na if isinstance(x, dict)]
    if cov and not cov.get("complete", True):
        D.p("Left out of the correlation (the verdict cannot be PASS):", bold=True)
        for r in cov.get("reasons") or []:
            D.bullet(_txt(r, 400))
            affecting.append(str(r))
    if cov.get("dofs_zero_excluded"):
        D.p("DOFs that are exactly zero in every test mode (unmeasured axes written as zeros) were left out of the MAC: " + ", ".join(cov["dofs_zero_excluded"][:30]) + ".", italic=True,
            size=9)
    if cov.get("functions_info_only"):
        D.p("Functions imported as information only (not FRF / PSD): " + ", ".join(_txt(n, 60) for n in cov["functions_info_only"][:20]) + ".", italic=True, size=9)
    issues = res.get("response_issues") or []
    if issues:
        D.p("Measured functions that could not be compared (UNCONFIRMED):", bold=True)
        for i in issues:
            D.bullet(f"{_txt(i.get('name'), 80)}: " + "; ".join(_txt(p, 300) for p in i.get("problems") or []))
            affecting.append(f"function {i.get('name')}: " + "; ".join(str(p) for p in i.get("problems") or []))
    modal = res.get("modal")
    warnings: list[str] = []
    if modal:
        pairs = modal.get("pairs") or []
        if pairs:
            D.table(["Test mode", "f test [Hz]", "FE mode", "f FE [Hz]", "Δf [%]", "MAC", "Max. MAC with another pair", "Damping ζ (test)", "DOFs"],
                    [[p["test_mode"], num(p["test_freq_Hz"], 5), p["fe_mode"], num(p["fe_freq_Hz"], 5), _pct(p["freq_error"]), num(p["mac"], 4),
                      num(p.get("mac_offdiag_max"), 3), num(p.get("damping"), 3), p.get("dofs", "–")] for p in pairs[:MAX_ROWS]],
                    caption="Paired modes: test against FE model", font=8)
        _orthogonality(D, modal, pairs, affecting, res.get("rows") or [])
        if modal.get("unpaired_test_modes"):
            D.p("Test modes without an FE partner: " + ", ".join(str(n) for n in modal["unpaired_test_modes"]) + " - these fail the correlation.", bold=True)
        if modal.get("unpaired_fe_modes"):
            D.p("FE modes not paired with a test mode: " + ", ".join(str(n) for n in modal["unpaired_fe_modes"][:40]) + ".", italic=True, size=9)
        mac = modal.get("mac")
        tn, fn = modal.get("test_modes") or [], modal.get("fe_modes") or []
        if mac and len(tn) <= MAX_MATRIX and len(fn) <= MAX_MATRIX:
            tf, ff = modal.get("test_freq_Hz") or [], modal.get("fe_freq_Hz") or []
            D.table(["Test \\ FE"] + [f"FE {n}\n{num(f, 4)} Hz" for n, f in zip(fn, ff)], [[f"test {n}\n{num(f, 4)} Hz"] + [num(v, 3) for v in row] for n, f, row in zip(tn, tf, mac)],
                    caption="MAC matrix (rows: test modes, columns: FE modes)", font=8)
        elif mac:
            D.p(f"The MAC matrix has {len(tn)} x {len(fn)} entries and is not printed (see the study file).", italic=True, size=9)
    resp = res.get("responses") or []
    if resp:
        D.table(["Function", "Quantity", "f peak test [Hz]", "f peak FE [Hz]", "Peak shift [%]", "Peak amplitude error [dB]", "RMS error [%]", "Band [Hz]"],
                [[_txt(r["name"], 80), (r.get("quantity") or "–") + (f" [{r['unit']}]" if r.get("unit") else ""), num(r["f_peak_test_Hz"], 5), num(r["f_peak_fe_Hz"], 5), _pct(r["peak_freq_shift"]),
                  num(r["peak_amp_error_dB"], 3), _pct(r["rms_error"]), f"{num(r['band_Hz'][0], 4)} - {num(r['band_Hz'][1], 4)}"] for r in resp[:MAX_ROWS]],
                caption="Response curves: predicted against measured", font=8)
        for r in resp:
            for u in r.get("unconfirmed") or []:
                D.bullet(f"{_txt(r['name'], 60)}: {_txt(u, 300)} - the rows of this curve are UNCONFIRMED")
                affecting.append(f"function {r['name']}: {u}")
    setups = res.get("function_setups") or {}
    if setups:
        D.table(["Function", "Frame", "Normalisation", "Quantity", "Unit", "Probe", "DOF", "Reference force [N]"],
                [[_txt(k, 70), v.get("frame") or "not chosen", v.get("per") or "from the file", v.get("quantity") or "from the file", v.get("unit") or "from the file", _txt(v.get("probe")) or "–",
                  v.get("dof") or "–", num(v.get("reference_force_n"))] for k, v in list(setups.items())[:MAX_ROWS]], caption="Setup of the measured functions (chosen by the user)", font=8)
    D.p("The limits below are the ones the result was judged with (stored with it).", italic=True, size=9)
    from ..space.correlation import LIMIT_HINTS
    hints = [f"{k}: {h}" for k, h in LIMIT_HINTS.items() if k in (res.get('limits') or {})]
    if hints:
        D.p("What the standard says about the orthogonality limits (hints only, not normative, never used as a default): " + "; ".join(hints) + ".", italic=True, size=9)
    lim = res.get("limits") or {}
    used = [k for k in lim if any(r.get("limit_key") == k for r in res.get("rows") or [])] or list(lim)
    D.table(["Limit", "Meaning", "Value", "Unit", "Document", "Revision", "Clause", "Usable"],
            [[k, lim[k].get("meaning", ""), "N/A" if lim[k].get("not_applicable") else num(lim[k].get("value")), lim[k].get("unit", ""), _txt(lim[k].get("document", "")),
              _txt(lim[k].get("revision", "")), _txt(lim[k].get("clause", "")),
              "declared not applicable" if lim[k].get("not_applicable") else ("yes" if lim[k].get("usable") else "NO: " + str(lim[k].get("reason", "")))] for k in used],
            caption="Correlation limits (entered and confirmed by the user)", font=8)
    rows = res.get("rows") or []
    if rows:
        osrc = _txt(((modal or {}).get("orthogonality") or {}).get("source"), 12) if isinstance(modal, dict) else ""
        D.table(["Item", "Metric", "Value", "Limit", "Status"],
                [[_txt(r["item"], 100), r["metric"], num(r.get("value"), 4), (f"{r['relation']} {num(r['limit'], 4)} {r.get('unit', '')}".strip() if r.get("limit") is not None else "–"),
                  _serep_shown_status(osrc, r)] for r in rows[:MAX_ROWS]], caption="Correlation criteria", status_col=4, font=8)
        n_early = sum(1 for r in rows if isinstance(r, dict) and _serep_shown_status(osrc, r) != str(r.get("status", "")))
        if n_early:
            D.p(f"{n_early} COC / AOC row(s) of the SEREP route were stored as PASS / FAIL by an earlier version of the program; they are shown UNCONFIRMED here because the SEREP COC / AOC is indicative "
                "and never judged (correlate again).", bold=True)
        rows_total = prov.get("rows_total")
        total = rows_total if isinstance(rows_total, int) else len(rows)
        if total > MAX_ROWS:
            D.p(f"{total - MAX_ROWS} more rows are not printed.", italic=True, size=9)
    if rec.get("damping_transferred"):
        D.p("Measured modal damping ratios of the paired modes were written to the damping table of the dynamics analysis: "
            + "; ".join(f"{num(f, 4)} Hz: {num(z, 3)}" for f, z in rec["damping_transferred"][:20]) + ".")
    shown = {str(a) for a in affecting}
    plain = [str(w) for w in (res.get("warnings") or []) if str(w) not in shown]
    for w in plain[:MAX_PLAIN_WARNINGS]:
        D.bullet(_txt(w, 400))
    if len(plain) > MAX_PLAIN_WARNINGS:
        D.p(f"{len(plain) - MAX_PLAIN_WARNINGS} more notes are not printed (see the study file).", italic=True, size=9)
    warnings += [str(w) for w in affecting] + plain[:MAX_PLAIN_WARNINGS]
    return warnings


def append_correlation(D: Any, study: Any, sec: str = "", results: Any = None) -> list[str]:
    """Called at the end of the space chapter; the subsection must never break the report."""
    try:
        return correlation_section(D, study, sec, results)
    except Exception as e:  # noqa: BLE001 - the report continues without the subsection
        return [f"Test correlation subsection: {e}"]

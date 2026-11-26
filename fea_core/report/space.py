"""Report chapter "Space structures: margins of safety" (fea_core.space): the requirement set with its provenance, the allowables, the
lowest margins of safety and the singularity justification table. The chapter is written only when the study carries space settings."""
from __future__ import annotations

from .word import num, vec

MAX_ROWS = 30


def has_space(study, res) -> bool:
    return getattr(study, "space", None) is not None and res is not None and bool(res.steps or getattr(res, "modes", None))


_VIA = {"gui": "GUI (the user)", "mcp": "MCP interface (an agent) - to be checked by the user"}


def _yes(v: bool) -> str:
    return "yes" if v else "NO"


def requirement_tables(D, space, rs) -> None:
    """Requirement set: metadata + confirmation, the factors that carry a value (with document / revision / clause) and the rules."""
    D.kv([("Requirement set", rs.name or "–"), ("Standard", rs.standard or "–"), ("Revision", rs.revision or "–"),
          ("Verification approach", rs.verification_approach or "–"), ("Confirmed by the user", _yes(space.requirement_confirmed())),
          ("Confirmed by", rs.confirmed_by or "–"), ("Confirmed via", _VIA.get(rs.confirmed_via, rs.confirmed_via or "–") if space.requirement_confirmed() else "–"),
          ("Date of confirmation", rs.date or "–")])
    rows = [[f.key, f.structure_class, f.approach or "any", num(f.value), f.document, f.revision, f.clause, f.note]
            for f in rs.factors if f.value is not None]
    if rows:
        D.table(["Factor", "Structure class", "Approach", "Value", "Document", "Revision", "Clause", "Note"], rows,
                caption="Requirement set - factors entered by the user", font=8, widths=(0.08, 0.14, 0.14, 0.08, 0.16, 0.1, 0.14, 0.16))
    else:
        D.p("No factor value is entered in the requirement set.", italic=True)
    rrows = [[r.check, r.structure_class, r.allowable or "–", r.stress_measure or "–", ", ".join(r.factors) or "–",
              r.thermal_handling or "–" if r.check != "buckling" else "–", ", ".join(r.thermal_factors) or "–", r.document, r.revision,
              r.clause] for r in rs.rules]
    if rrows:
        D.table(["Check", "Structure class", "Allowable", "Stress measure", "Design factor J = product of", "Thermal handling",
                 "Thermal factors", "Document", "Revision", "Clause"], rrows, caption="Requirement set - margin rules", font=8)


def allowable_table(D, study) -> None:
    from ..space.requirements import allowable_of
    rows = []
    for m in study.materials.values():
        rec = allowable_of(m)
        if rec is None:
            rows.append([m.name, "–", "–", "–", "–", "–", "–", "–", "no allowables entered"])
            continue
        rows.append([m.name, rec.basis or "–", num(rec.Fy), num(rec.Fu), num(rec.Fsy), num(rec.Fsu), num(rec.temp_C), rec.source or "–",
                     (f"confirmed by {rec.confirmed_by} ({rec.date}" + (f", via {rec.confirmed_via.upper()}" if rec.confirmed_via else "") + ")"
                      if rec.is_confirmed() else "NOT confirmed")
                     + ("" if rec.is_verified() else "; not verified")])
    if rows:
        D.table(["Material", "Basis", "Fy [MPa]", "Fu [MPa]", "Fsy [MPa]", "Fsu [MPa]", "T [°C]", "Source", "Status"], rows,
                caption="Material allowables entered by the user", font=8)


def margin_tables(D, summ) -> None:
    rows = []
    for r in summ.lowest(MAX_ROWS):
        rows.append([r.part, r.case, r.check, r.structure_class, num(r.design_factor), num(r.design_stress), num(r.allowable), num(r.mos, 3),
                     num(r.mos_raw, 3), vec(r.location, 4) if r.location else "–",
                     {"pass": "PASS", "fail": "FAIL"}.get(r.status, "UNCONFIRMED")])
    if rows:
        D.table(["Part", "Case", "Check", "Class", "J", "Design stress [MPa]", "Allowable", "MoS", "MoS incl. singularities", "Location [mm]",
                 "Status"], rows, caption="Lowest margins of safety outside singularities", status_col=10, font=8)
    skipped = [r for r in summ.rows if r.mos is None]
    if skipped:
        D.p(f"{len(skipped)} margins could not be computed (no allowable, no rule, or only singular places) - see the open items.",
            italic=True, size=9)


def singularity_table(D, summ) -> None:
    if not summ.singularities:
        D.p("No singular place was detected among the highest-stress locations.", italic=True)
        return
    rows = [[f"S{i}", s.part, vec(s.location, 4), s.cause, num(s.mos_raw, 3), ", ".join(s.cases[:4]) + (" ..." if len(s.cases) > 4 else ""),
             ", ".join(s.checks), (s.justification.note + (f" ({s.justification.author}, {s.justification.date})" if s.justification.author else ""))
             if s.justified and s.justification else "NOT JUSTIFIED - a note of the user is required before sign-off"]
            for i, s in enumerate(summ.singularities, 1)]
    D.table(["#", "Part", "Location [mm]", "Cause", "MoS at the place", "Cases", "Checks", "Justification"], rows,
            caption="Singularity justification table (places excluded from the margins)", font=8,
            widths=(0.05, 0.1, 0.15, 0.14, 0.09, 0.14, 0.09, 0.24))


def space_section(D, sess, sec: str = "") -> list[str]:
    """Writes the chapter body; returns warnings."""
    from ..space.loads import sign_off_margins
    from ..space.margins import FAIL, PASS
    study, res = sess.study, sess.results
    space = study.space
    mesh = sess.valid_mesh() if hasattr(sess, "valid_mesh") else None
    summ = sign_off_margins(study, res, sess, mesh)                 # with the kept 3-axis envelopes when a current result exists (the verdict of the sign-off)
    D.p("Margin of safety MoS = allowable / (J · σ) − 1, where J is the product of the design factors of the margin rule and σ the "
        "equivalent stress of the unfactored load at the governing place; the governing place is the highest one that is not classified as a "
        "singularity. Every factor, allowable and rule in this chapter was entered by the user; the program contains no standard value.")
    verdict = {PASS: summ.verdict_text, FAIL: "FAIL"}.get(summ.verdict, "UNCONFIRMED_REQUIREMENTS")
    D.p(f"Verdict: {verdict}", bold=True, size=11, color=(0x1B, 0x7F, 0x3B) if summ.signoff_ready else (0xC0, 0x39, 0x2B))
    if summ.verdict == PASS and not summ.signoff_ready:
        D.p("The margins are non-negative, but this is not a sign-off: every singular place that was left out of the margins needs a "
            "note of the user (see the singularity justification table).", italic=True)
    if summ.verdict != PASS:
        D.p("This is not an ECSS verification result: " + (
            "at least one margin with confirmed inputs is negative." if summ.verdict == FAIL else
            "an input is not confirmed or complete, so the numbers are indicative only."), italic=True)
    if space.requirement_set is not None:
        requirement_tables(D, space, space.requirement_set)
    else:
        D.p("No requirement set is defined for this study: the margins are computed without design factors (indicative).", italic=True)
    allowable_table(D, study)
    margin_tables(D, summ)
    singularity_table(D, summ)
    items = summ.open_items
    if items:
        D.p("Open items before sign-off:", bold=True)
        for it in items[:20]:
            D.bullet(it)
    from .space_loads import loads_section             # launch loads, mass properties and budgets (F14 S2)
    from .space_s4 import s4_sections          # stage S4: interface forces, notching, bolts, PCB, rails (own module)
    from .space_corr import append_correlation              # F14-S5: the test correlation subsection (own module)
    return (list(summ.warnings) + loads_section(D, sess) + s4_sections(D, study, sec)
            + append_correlation(D, study, sec, res))


# ------------------------------------------------------------------------------------------------------------------ model quality checks (F14 S3)
MAX_VALUES_PER_CHECK = 14
_STATUS_COLOUR = {"PASS": (0x1B, 0x7F, 0x3B), "WARN": (0xB3, 0x6B, 0x00), "FAIL": (0xC0, 0x39, 0x2B)}
_LIMIT_STATE = {"usable": "confirmed - decides the check", "not confirmed": "NOT confirmed - the check is UNCONFIRMED", "incomplete": "NOT usable - document / revision / clause missing",
                "missing": "not entered"}


def has_checks(sess, study) -> bool:
    """The session keeps a run of the model quality checks (`fea_space_model_checks`, the checks dialog) that belongs to this study."""
    run = getattr(sess, "model_checks", None)
    return run is not None and bool(run.results) and run.study_name == str(getattr(study, "name", ""))


def _scalar_values(values: dict) -> list[list[str]]:
    """The numbers of a check as table rows (name, value): plain scalars only; text that looks like a path is left out."""
    rows = []
    for k, v in values.items():
        if isinstance(v, bool):
            rows.append([k, "yes" if v else "no"])
        elif isinstance(v, (int, float)):
            rows.append([k, num(v, 4)])
        elif isinstance(v, str) and 0 < len(v) <= 60 and not any(c in v for c in ("\\", "/")) and "path" not in k and "dir" not in k:
            rows.append([k, v])
        if len(rows) >= MAX_VALUES_PER_CHECK:
            break
    return rows


def _equilibrium_divisors(D, r) -> None:
    """What the relative imbalance of every static state of check 7 was divided by: the resultant of the applied load / reactions, or (self-equilibrated state) the sum of the
    nodal norms of the solver's record. A reader must be able to see the scale the confirmed limit was applied on."""
    D.p("Check 7 - divisor of the relative imbalance: " + str(r.values.get("divisor", "")), size=9, italic=True)
    rows = [[str(s.get("state", "")), num(s["imbalance"], 3) if s.get("imbalance") is not None else "–", num(s["force_scale_N"], 4) if s.get("force_scale_N") else "–",
             str(s.get("force_divisor", "–")), num(s["moment_scale_Nmm"], 4) if s.get("moment_scale_Nmm") else "–", str(s.get("moment_divisor", "–"))] for s in r.values["states"]]
    D.table(["State", "Imbalance", "Force divisor [N]", "Force basis", "Moment divisor [N mm]", "Moment basis"], rows,
            caption="Check 7: the divisor of the relative imbalance of every static state", font=8, widths=(0.25, 0.12, 0.17, 0.14, 0.18, 0.14))


def _settings_rows(results) -> list[list[str]]:
    """(Setting, Value, Checks) rows of the settings that differ from the defaults: one row per setting and value, the checks that ran with it listed together (a kept run may
    merge checks of several runs, each with its own budgets)."""
    from ..space import model_checks as MC
    grouped: dict[tuple[str, str], list[int]] = {}
    for r in results:
        for k, v in r.options.items():
            grouped.setdefault((k, MC.describe_option(k, v)), []).append(r.number)
    order = {k: i for i, k in enumerate(MC.OPTION_CHECKS)}
    return [[k, v, ", ".join(map(str, nums))] for (k, v), nums in sorted(grouped.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1]))]


def checks_section(D, sess, sec: str = "") -> list[str]:
    """Writes the chapter body "Model quality checks" from the run kept in `sess.model_checks`; returns warnings."""
    from ..space import model_checks as MC
    from ..space._checks_common import scrub_paths
    run = sess.model_checks
    results = run.results
    warnings: list[str] = []
    verdict = MC.overall_verdict(results)
    stale = run.digest != MC.study_digest(sess.study)
    D.p("The model quality checks of ECSS-E-ST-32-11C 6.4.1 and common FEM practice: a finite element model is accepted for margins of safety only when it behaves correctly under the checks "
        "below. The program contains no tolerance of a standard: every tolerance is a limit of the requirement set that the user entered and confirmed, and a check "
        "without a confirmed limit is UNCONFIRMED (its numbers are shown, but it is never PASS). What a check decides without a limit are physical invariants "
        "(a free body has exactly six rigid-body modes; the results are in equilibrium to the tolerance of the solver itself), and those can only produce FAIL.")
    if stale:                                                   # a verdict of another state of the model is never shown as a result of this one
        text = "The study (or its confirmed limits) was changed after these checks were run - run them again before relying on this chapter."
        D.p(f"Verdict of the model: OUT OF DATE (the run of {run.created or '–'} gave {verdict})", bold=True, size=11, color=(0xB3, 0x6B, 0x00))
        D.p(text, italic=True, bold=True)
        warnings.append("Model quality checks: " + text)
    else:
        D.p(f"Verdict of the model: {verdict}", bold=True, size=11, color=_STATUS_COLOUR.get(verdict, (0xC0, 0x39, 0x2B)) if verdict in ("PASS", "WARN", "FAIL") else (0xB3, 0x6B, 0x00))
        if verdict == "INCOMPLETE":
            D.p("Not all seven checks were run or one of them could not be completed: this is not a verification of the model.", italic=True)
        elif verdict == "UNCONFIRMED_REQUIREMENTS":
            D.p("At least one check has no confirmed limit: this is not an ECSS verification result, the numbers are indicative only.", italic=True)
    via = run.confirmed_via
    kv = [("Run on", run.created or "–"), ("Requirement set", run.requirement_set or "–"), ("Confirmed by", run.confirmed_by or "– (not confirmed)"),
          ("Confirmed via", (_VIA.get(via, via) if via else "–") if run.confirmed_by else "–"), ("Date of confirmation", run.date or "–")]
    D.kv(kv)
    if run.confirmed_by and via == "mcp":
        D.p("The requirement set behind the limits was confirmed through the MCP interface by an agent: the user must check every limit against its source and confirm the set "
            "in the program itself.", italic=True, bold=True, color=(0xB3, 0x6B, 0x00))
    rows_set = _settings_rows(results)
    if rows_set:
        D.table(["Setting", "Value", "Checks"], rows_set, caption="Budgets and settings that differ from the defaults (by the number of the check that ran with them)", font=8,
                widths=(0.3, 0.55, 0.15))
    else:
        D.p("Every check ran with the default settings; no budget was given.", italic=True, size=9)
    suffix = " - out of date" if stale else ""
    D.table(["#", "Check", "Status", "Finished", "Result"], [[str(r.number), r.title, r.status + suffix, r.finished or "–", scrub_paths(r.summary)] for r in results],
            caption="FEM model quality checks" + (" (OUT OF DATE: the study changed after the run)" if stale else ""), status_col=2, widths=(0.05, 0.22, 0.14, 0.13, 0.46))
    shown = [r for r in results if r.findings or r.notes]
    if shown:
        D.p("Findings and notes:", bold=True)
        for r in shown:
            D.p(f"{r.number}. {r.title} - {r.status}", bold=True, size=9)
            for f in r.findings[:6]:
                D.bullet(scrub_paths(f))
            for n in r.notes[:4]:
                D.bullet(scrub_paths(n))
    vrows = [[f"{r.number}. {r.title}", k, v] for r in results for k, v in _scalar_values(r.values)]
    if vrows:
        D.table(["Check", "Quantity", "Value"], vrows, caption="Numbers computed by the checks (numerical settings of the equilibrium check included: consistency_tolerance, "
                "net_load_share)", font=8, widths=(0.4, 0.4, 0.2))
    for r in results:
        if r.id == "equilibrium" and r.values.get("states"):
            _equilibrium_divisors(D, r)
    lrows = [[x["name"], x["description"], num(x["value"]) if x["value"] is not None else "–", x["document"] or "–", x["revision"] or "–", x["clause"] or "–",
              _LIMIT_STATE.get(x["state"], x["state"])] for x in run.limits]
    D.table(["Limit", "What it limits", "Value", "Document", "Revision", "Clause", "State"], lrows,
            caption="Tolerances of the checks (limits of the requirement set; none is built into the program)", font=8, widths=(0.14, 0.31, 0.08, 0.12, 0.08, 0.1, 0.17))
    return warnings

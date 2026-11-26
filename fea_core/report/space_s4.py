"""Report subsections of the space chapter for stage S4: interface forces, notching, ECSS bolted joints, PCB check, CubeSat rails.

The subsections are written from the stored summaries of the study (`fea_core.space.s4_store`) – whatever was computed (GUI or MCP) is reported with its verdict and the
provenance of every number that decided it. A verdict other than PASS is printed as such; the bolt chain follows ECSS-E-HB-32-23A Rev.1 (clause per formula, the deviations
of the program are listed). No standard value appears here: all numbers come from the requirement set that the user confirmed.

A stored verdict is printed only while it is still true (`fea_core.space.s4_verify.shown`): when the requirement set, the allowables or the study changed since the calculation
the report says "outdated - recompute" instead; a record without a basis (a hand-edited project file) is UNCONFIRMED_REQUIREMENTS. Bolts, PCB and rails are recomputed from their
stored inputs against the present requirement set and the report prints the RECOMPUTED rows (never stored numbers under a recomputed verdict); when the recomputation differs from
the stored record it says so. A record whose stored inputs are missing gets no verdict. Notching and interface records cannot be recomputed: they are judged by the digests alone,
and the report says "stored result, not recomputed - tamper-evident only (staleness), not a signature".
"""
from __future__ import annotations

from typing import Any

from .word import num, vec

BOLT_NOTE = "formula chain after ECSS-E-HB-32-23A Rev.1 (6 February 2023); the clause of each formula is in the Formula column"


def has_s4(study: Any) -> bool:
    from ..space import s4_store
    return bool(s4_store.get(study))


def _verdict(v: str | None) -> str:
    return {"PASS": "PASS", "FAIL": "FAIL", "OUTDATED": "outdated - recompute"}.get(v or "", "UNCONFIRMED_REQUIREMENTS")


def _status(shown: Any, st: str | None) -> str:
    """The status of one margin row: only a fresh record shows PASS / FAIL (it is the status of the present state)."""
    if not shown.fresh:
        return "outdated" if shown.verdict == "OUTDATED" else "UNCONFIRMED"
    return {"PASS": "PASS", "FAIL": "FAIL"}.get(st or "", "UNCONFIRMED")


def _stale_note(D: Any, shown: Any) -> None:
    """Why the verdict is not the stored one (outdated / no basis / cannot be recomputed), or that the stored result was not reproduced, and how far it is checked."""
    if not shown.fresh or shown.reproduced is False:
        D.p(shown.text, bold=True, size=9)
    if shown.note:
        D.p(shown.note[0].upper() + shown.note[1:] + ".", italic=True, size=8)


def _of(shown: Any, d: dict) -> dict:
    """The record whose numbers are printed: the recomputed one when the record was recomputed, otherwise the stored one."""
    return shown.data if shown.data is not None else d


def _inputs_table(D: Any, inputs: list[dict], caption: str) -> None:
    rows = [[i.get("name", ""), num(i.get("value")) if isinstance(i.get("value"), (int, float)) else str(i.get("value")), i.get("source", ""),
             "data" if i.get("confirmed") is None else "yes" if i.get("confirmed") else "NO"] for i in inputs]
    if rows:
        D.table(["Input", "Value", "Source (document / revision / clause)", "Confirmed"], rows, caption=caption, font=8, widths=(0.3, 0.14, 0.44, 0.12))


def interface_section(D: Any, d: dict, sec: str, study: Any = None) -> None:
    D.h(f"{sec} Interface forces and moments (base reactions in the dynamics)", 2)
    if study is not None and not d.get("basis"):
        D.p("This record carries no basis (digests of the requirement set, the allowables and the study): it cannot be shown to belong to the present study - recompute.",
            bold=True, size=9)
    if study is not None:
        from ..space import s4_verify
        sh = s4_verify.shown(study, "interface", d)
        if sh.verdict == "OUTDATED":
            D.p("The numbers below are those of an earlier state of the study: " + sh.text, bold=True, size=9)
        else:
            D.p(sh.note[0].upper() + sh.note[1:] + ".", italic=True, size=8)
    D.p("The interface force is the elastic reaction of the held nodes (constraints and elastic supports): the sum of the modal base-reaction vectors times the modal "
        "coordinates plus the static residual of the unit base acceleration. Quasi-static limit: the reaction of a unit base acceleration equals the total mass times the "
        "acceleration. Damping is an uncertain input (sensitivity ±50 %).", size=9)
    chk = d.get("static_mass_check") or {}
    D.kv([("Total mass [kg]", num(d.get("total_mass_kg"))), ("Reference point of the moments [mm]", vec(d.get("ref_point_mm", [0, 0, 0]))),
          ("Modes in the basis", str(d.get("modes", "–"))), ("Modal engine", str(d.get("engine", "–"))), ("Damping", str(d.get("damping", "–"))),
          ("Static reaction vs total mass (relative error x, y, z)", vec(chk.get("relative_error", []), 3) if chk else "–")])
    rows = [[r["mode"], num(r["f_Hz"]), vec(r["gamma"], 3), vec(r["F_N"], 3), vec(r["M_Nmm"], 3)] for r in d.get("modal_table", [])]
    if rows:
        D.table(["Mode", "f [Hz]", "Γ (x y z)", "F_j [N per unit q]", "M_j [N·mm per unit q]"], rows, caption="Modal base-reaction vectors (largest force first)", font=8)
    drows = []
    for ax, item in (d.get("directions") or {}).items():
        s, r = item.get("sine"), item.get("random")
        drows.append([ax, num(s["F_peak_max_N"]) if s else "–", num(s["f_at_F_peak_Hz"]) if s else "–", num(s["M_peak_max_Nmm"]) if s else "–",
                      num(r["F_rms_total_N"]) if r else "–", num(r["F_nsigma_total_N"]) if r else "–", num(r["M_rms_total_Nmm"]) if r else "–"])
    if drows:
        D.table(["Base axis", "Sine peak |F| [N]", "at f [Hz]", "Sine peak |M| [N·mm]", "Random F_rms [N]", "3σ F [N] (information)", "Random M_rms [N·mm]"], drows,
                caption="Interface force and moment in the sine and random environments", font=8)
    for w in d.get("warnings", []):
        D.bullet(str(w))


def notch_section(D: Any, items: dict, sec: str, study: Any = None) -> None:
    from ..space import s4_verify
    D.h(f"{sec} Primary notching / force limiting", 2)
    D.p("The notched input is the specification level multiplied by min(1, limit / (J · predicted quantity)); the minimum notch width and the floor of the notch come from "
        "the requirement set. The table lists every notch: the unnotched and the notched input, the reduction and the margin that remains against the limit.", size=9)
    for key, d in items.items():
        sh = s4_verify.shown(study, "notch", d)
        D.p(f"{key}: verdict {_verdict(sh.verdict)} - {sh.text}", bold=True, size=10)
        D.p(sh.note[0].upper() + sh.note[1:] + ".", italic=True, size=8)
        if d.get("table_ratio") is not None:
            D.p(f"Check of the exported breakpoint table between the analysis frequencies: J x response / limit = {num(d['table_ratio'], 5)} (accepted up to 1 + "
                f"{num(d.get('table_tolerance'), 3)}, the numerical tolerance of the check).", size=8)
        _inputs_table(D, d.get("inputs", []), f"Inputs of the notching ({key})")
        rows = [[i + 1, num(n["f_lo"]), num(n["f_hi"]), num(n["width_oct"], 3), num(n["input_unnotched"]), num(n["input_notched"]), num(n["reduction_db"], 3),
                 num(n["quantity_unnotched_peak"]), num(n["quantity_notched_peak"]), num(n["limit"]), num(n["margin_remaining"], 3),
                 ("widened; " if n["widened"] else "") + ("floor-limited; " if n["floor_limited"] else "") + (n["status"] if sh.fresh else "outdated")]
                for i, n in enumerate(d.get("notches", []))]
        if rows:
            D.table(["#", "f lo [Hz]", "f hi [Hz]", "Width [oct]", "Input unnotched", "Input notched", "Reduction [dB]", "Quantity unnotched", "Quantity notched", "Limit",
                     "Margin", "Remark"], rows, caption=f"Notch justification table ({key})", font=7)
        else:
            D.p("No notch is needed: the predicted quantity stays below the limit.", italic=True)
        for w in d.get("warnings", []):
            D.bullet(str(w))


def bolts_section(D: Any, items: dict, sec: str, study: Any = None) -> None:
    from ..space import s4_verify
    D.h(f"{sec} Bolted joints (ECSS-style margins)", 2)
    from ..space import bolts_ecss as BE
    D.p("Note: " + BOLT_NOTE + ". Every factor and coefficient was entered by the user in the requirement set; the safety factor J multiplies the external load and the thermally induced load, not the "
        "preload (7.9.2, ECSS-E-ST-32-10C 4.3.2.1 b with 4.3.2.2 a, 4.3.2.2 b, Table 4-4 notes a / e). Deviations and limits of the program:", size=9)
    for dev in BE.DEVIATIONS:
        D.bullet(dev)
    for name, d in items.items():
        sh = s4_verify.shown(study, "bolts", d)
        D.p(f"Bolt {name}: verdict {_verdict(sh.verdict)}", bold=True, size=10)
        _stale_note(D, sh)
        d = _of(sh, d)                                                  # the recomputed rows when the record was recomputed
        th = d.get("dF_th_N", {})
        D.kv([("Load factor Φ", num(d.get("phi"), 4)), ("Embedding loss [N]", num(d.get("F_emb_N"))), ("Preload F_p,min / F_p,max incl. scatter [N]",
              f"{num(d.get('F_pmin0_N'))} / {num(d.get('F_pmax0_N'))}"), ("Thermal preload change at T_min / T_max [N]", f"{num(th.get('T_min'))} / {num(th.get('T_max'))}"),
              ("Preload after thermal change F_p,min / F_p,max [N] (J_T = 1; the checks use their J_T)", f"{num(d.get('F_pmin_N'))} / {num(d.get('F_pmax_N'))}"),
              ("Minimum clamp force under load [N]", num(d.get("F_clamp_min_N"))), ("Maximum bolt force [N]", num(d.get("F_bolt_max_N")))])
        rows = [[m["check"], num(m["demand"]), num(m["capacity"]), num(m["design_factor"], 4), num(m.get("thermal_factor"), 4), num(m["mos"], 3), _status(sh, m["status"]), m["formula"]]
                for m in d.get("margins", [])]
        D.table(["Check", "Demand", "Capacity", "J", "J_T", "MoS", "Status", "Formula"], rows, caption=f"Margins of safety of bolt {name}", status_col=6, font=7,
                widths=(0.11, 0.08, 0.08, 0.05, 0.05, 0.08, 0.14, 0.41))
        _inputs_table(D, d.get("inputs", []), f"Coefficients used for bolt {name}")
        for m in d.get("margins", []):
            for u in m.get("unconfirmed", []):
                D.bullet(f"{m['check']}: {u}")
        notes = [w for w in d.get("warnings", []) if w != d.get("formula_note")]
        if notes:
            D.p(f"Notes on bolt {name}:", bold=True, size=9)
            for w in notes:
                D.bullet(str(w))


def pcb_section(D: Any, items: dict, sec: str, study: Any = None) -> None:
    from ..space import s4_verify
    D.h(f"{sec} Printed circuit boards (Steinberg method)", 2)
    D.p("Z_allow = c · B / (C · h · r · √L); MoS = Z_allow / (J · Z_nσ) − 1. The constant, the amplitude convention, J and the fatigue law of the three-band damage are "
        "entered and confirmed by the user in the requirement set.", size=9)
    rows = []
    shown = {}
    data: dict[str, dict] = {}
    for name, d in items.items():
        shown[name] = s4_verify.shown(study, "pcb", d)
        d = data[name] = _of(shown[name], d)                              # the recomputed record when it was recomputed
        db = d.get("damage_three_band") or {}
        rows.append([name, num(d.get("Z_allow_mm")), num(d.get("Z_sigma_mm")), num(d.get("Z_nsigma_mm")), num(d.get("MoS"), 3), num(db.get("damage")),
                     num(d.get("damage_narrow_band")), num(d.get("damage_limit")), _verdict(shown[name].verdict)])
    D.table(["Component", "Z_allow [mm]", "σ_z [mm]", "Z_nσ [mm]", "MoS", "Damage 3-band", "Damage narrow band", "Damage limit", "Verdict"], rows,
            caption="PCB displacement margin and fatigue damage", status_col=8, font=8)
    for name in items:
        _stale_note(D, shown[name])
        _inputs_table(D, data[name].get("inputs", []), f"Coefficients used for component {name}")
        for w in data[name].get("warnings", []):
            D.bullet(f"{name}: {w}")


def rails_section(D: Any, d: dict, sec: str, study: Any = None) -> None:
    from ..space import s4_verify
    sh = s4_verify.shown(study, "rails", d)
    D.h(f"{sec} Deployer rails", 2)
    D.p(f"Verdict: {_verdict(sh.verdict)} - {sh.text}", bold=True, size=10)
    if sh.note:
        D.p(sh.note[0].upper() + sh.note[1:] + ".", italic=True, size=8)
    d = _of(sh, d)
    rows = [[r["support"], r["kind"], r["state"], num(r["reaction_N"]), num(r.get("effective_N")), "PULLS (" + ", ".join(r.get("pull_states", [])[:3]) + ")" if r["pulls"] else "",
             num(r["limit_N"]), num(r["mos"], 3)] for r in d.get("supports", [])]
    if rows:
        D.table(["Support", "Kind", "State", "Reaction [N]", "Force of the margin [N]", "", "Limit [N]", "MoS"], rows,
                caption="Reaction of the rail supports (state with the largest force; the margin uses the conservative force |push| + |pull of the opposite wall / end plate|)", font=8)
    rows = [[x["part"], num(x["max_lateral_deflection_mm"]), x["state"], num(x["limit_mm"]), num(x["mos"], 3)] for x in d.get("deflection", [])]
    if rows:
        D.table(["Rail", "Max lateral deflection [mm]", "State", "Limit [mm]", "MoS"], rows, caption="Lateral deflection of the rails", font=8)
    _inputs_table(D, d.get("inputs", []), "Limits used for the rails")
    for w in d.get("warnings", []):
        D.bullet(str(w))


def s4_sections(D: Any, study: Any, sec: str) -> list[str]:
    """Writes the stored S4 results as subsections `sec`.1 ... of the space chapter; returns warnings."""
    from ..space import s4_store
    data = s4_store.get(study)
    warnings: list[str] = []
    n = 0
    from ..space import s4_verify
    with s4_verify.judging(study):                                  # the present basis of the study once for all the records
        for key, fn in (("interface", interface_section), ("notch", notch_section), ("bolts", bolts_section), ("pcb", pcb_section), ("rails", rails_section)):
            if key not in data:
                continue
            n += 1
            try:
                fn(D, data[key], f"{sec}.{n}", study)
            except Exception as e:  # noqa: BLE001 - the report continues without this subsection
                warnings.append(f"Space chapter, {key}: {e}")
    return warnings

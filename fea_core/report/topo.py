"""Report – topology optimisation (fea_core.topo, `Results.info["topology"]`): the setting and the result of the BESO and, with manufacturing
constraints, the table of the constraints (setting, checks, measured values) and the compliance penalty against the run without them."""
from __future__ import annotations

from .word import num

RED = (192, 0, 0)
APPROX = ("The manufacturing constraints act on the elements of the mesh (symmetry: mirror images decide together; extrusion: columns of elements "
          "decide together; overhang: an element needs support below it) – they are APPROXIMATE, not exact surface constraints, and the mesh "
          "limits their accuracy. The surface of the result is measured afterwards on the exported triangles; the numbers below are that "
          "measurement. Check the result in the slicer before printing.")


def has_topology(res) -> bool:
    return res is not None and bool((getattr(res, "info", None) or {}).get("topology"))


def _pct(v) -> str:
    return "–" if v is None else f"{100 * float(v):.1f} %"


def _iterations_text(t: dict) -> str:
    """Number of iterations with the reason why the loop stopped (a repeated design or the rollback limit is NOT convergence) and the rollbacks."""
    reason = t.get("stop_reason")
    txt = f"{t.get('iterations', 0)}"
    if t.get("converged"):
        txt += " (converged)"
    elif reason == "cycle":
        txt += " (stopped: the design repeated itself)"
    elif reason == "rollback_limit":
        txt += " (stopped after the rollbacks did not help)"
    else:
        txt += " (stopped at the limit)"
    if t.get("rollbacks"):
        txt += f", {t['rollbacks']} steps taken back (rollbacks)"
    return txt


def constraint_rows(info: dict) -> list[list[str]]:
    """Rows [constraint, setting, checks / measured] of the constraints table."""
    rows: list[list[str]] = []
    sym = info.get("symmetry")
    if sym:
        planes = "; ".join("normal (" + ", ".join(num(x, 3) for x in p["normal"]) + "), point (" + ", ".join(num(x, 4) for x in p["point"]) + ") mm"
                           for p in sym.get("planes", []))
        rows.append(["Symmetry", planes,
                     f"largest density difference inside a group of mirror images {num(sym.get('max_group_spread', 0.0))} (must be 0); "
                     f"{sym.get('unmatched', 0)} elements without a mirror image; mismatch of the result with its mirror image "
                     f"{_pct(sym.get('mismatch_volume_fraction'))} of the volume (mesh asymmetry); "
                     + (f"run without the constraints: {_pct(sym['reference_mismatch_volume_fraction'])}" if sym.get("reference_mismatch_volume_fraction") is not None else "")])
    ext = info.get("extrusion")
    if ext:
        pr = ext.get("prism") or {}
        ref = ext.get("reference_prism") or {}
        txt = (f"{ext.get('columns', 0)} columns of {num(ext.get('bin_size_mm'), 3)} mm; largest density difference inside a column "
               f"{num(ext.get('max_column_spread', 0.0))} (must be 0)")
        if pr.get("area_spread") is not None:
            txt += (f"; sections of the exported surface at 3 heights: area {', '.join(num(a, 4) for a in pr['areas_mm2'])} mm², "
                    f"spread {_pct(pr['area_spread'])}, smallest overlap {_pct(pr.get('min_iou'))}")
        if ref.get("min_iou") is not None:
            txt += f"; without the constraint the overlap is {_pct(ref['min_iou'])}"
        rows.append(["Extrusion", "direction (" + ", ".join(num(x, 3) for x in ext.get("direction", [])) + ")", txt])
    ov = info.get("overhang")
    if ov:
        sf = ov.get("surface") or {}
        rf = ov.get("reference_surface") or {}
        txt = (f"step that produced the kept design: {ov.get('removed_last_iteration', 0)} unsupported elements removed, {ov.get('added_last_iteration', 0)} supported "
               "elements added" + (f" (volume short by {num(ov.get('deficit_left_mm3'), 3)} mm³)" if ov.get("volume_deficit") else " (volume kept)"))
        if sf.get("downward_area_mm2"):
            txt += (f"; flat patches (connected facets flatter than the angle): {sf.get('flat_patches', 0)}, the largest "
                    f"{num(sf.get('largest_flat_patch_mm2'), 3)} mm²"
                    + (f" (without the constraint {num(rf['largest_flat_patch_mm2'], 3)} mm²)" if rf.get("largest_flat_patch_mm2") is not None else ""))
        if sf.get("downward_area_mm2"):
            txt += (f"; measured on the exported surface: {_pct(sf['violating_area_fraction'])} of the downward area is flatter than "
                    f"{num(sf['angle_deg'], 3)} °, {_pct(sf['violating_area_fraction_beyond_tolerance'])} beyond the mesh tolerance of "
                    f"{num(sf['tolerance_deg'], 3)} °; worst facet {num(sf['worst_angle_deg'], 3)} °, 5 % of the area flatter than {num(sf['p05_angle_deg'], 3)} °")
        if rf.get("downward_area_mm2"):
            txt += (f"; without the constraint: {_pct(rf['violating_area_fraction'])} / {_pct(rf['violating_area_fraction_beyond_tolerance'])}, "
                    f"worst {num(rf['worst_angle_deg'], 3)} °")
        sc = ov.get("strict_check") or {}
        if sc.get("pruned_elements") and sc.get("accepted"):
            txt += (f"; the final check (support through material, not across a thin slit) removed {sc['pruned_elements']} elements "
                    f"({_pct(sc.get('pruned_volume_fraction'))} of the volume), {sc.get('iterations', 0)} more iterations re-routed the load path "
                    f"(compliance {num(sc.get('compliance_loop_Nmm'), 4)} → {num(sc.get('compliance_Nmm'), 4)} N·mm)")
        if ov.get("unsupported_non_design_elements"):
            txt += f"; {ov['unsupported_non_design_elements']} protected elements are not supported from the plate (they count as supported)"
        rows.append(["Overhang", f"build direction ({', '.join(num(x, 3) for x in ov.get('build_dir', []))}), angle {num(ov.get('angle_deg'), 3)} ° "
                     f"from the plate, base: {'lowest layer' if ov.get('base') == 'min' else ov.get('base')}", txt])
    cmp_ = info.get("compliance") or {}
    if cmp_.get("penalty") is not None:
        rows.append(["Compliance penalty", "against the run without the constraints",
                     f"{cmp_['penalty'] * 100:+.1f} % ({num(cmp_['constrained_Nmm'], 4)} vs {num(cmp_['unconstrained_Nmm'], 4)} N·mm at volume "
                     f"{_pct(cmp_.get('volume_fraction'))} vs {_pct(cmp_.get('unconstrained_volume_fraction'))})"])
    return rows


def section(D, sess, o, img_dir, images, warnings, sec: str) -> None:
    res = sess.results
    t = res.info["topology"]
    D.p("Topology optimisation by the BESO method (bi-directional evolutionary structural optimisation): material of the least strain energy is removed "
        "in steps and the stiffness of the elements is fed back into a new analysis in every iteration; the sensitivity is the strain energy of the "
        "element summed over the load states, filtered in the radius given below.", size=9)
    c0, c1 = t.get("initial_compliance_Nmm"), t.get("final_compliance_Nmm")
    rows = [("Target volume", _pct(t.get("target_volume_fraction"))),
            ("Reached volume", _pct(t.get("final_volume_fraction"))),
            ("Iterations", _iterations_text(t)),
            ("Filter radius / removal rate / penalty", f"{num(t.get('filter_radius_mm'), 3)} mm / {_pct(t.get('evolution_rate'))} / {num(t.get('penalty'), 3)}"),
            ("Protected selections", ", ".join(t.get("non_design_selections") or []) or "none")]
    if c0 and c1:
        rows.append(("Compliance (full design → result)", f"{num(c0, 4)} → {num(c1, 4)} N·mm (×{num(c1 / c0, 3)})"))
    D.kv(rows)
    info = t.get("constraints")
    if info:
        D.p("Manufacturing constraints", bold=True)
        D.table(["Constraint", "Setting", "Checks and measured values"], constraint_rows(info), widths=[0.15, 0.30, 0.55],
                caption="Manufacturing constraints of the topology optimisation", font=8)
        D.p("WARNING – APPROXIMATE: " + APPROX, bold=True, size=9, color=RED)
        if info.get("cycle_detected"):
            D.p(f"The solid set repeated itself every {info.get('cycle_period')} iterations; the optimisation was stopped and the best design of the run was kept.", italic=True, size=9)
        for w in info.get("warnings", []):
            D.p(f"! {w}", italic=True, size=9)

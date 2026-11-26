"""Report subsection of the space-structures chapter (F14 stage S2): launch environments, quasi-static cases, mass properties and budgets, the minimum natural
frequency and the 3-axis envelopes. Written after the margins of the chapter (`report.space.space_section` calls `loads_section`); nothing is printed for a study
that has none of these inputs. Every level, factor and limit printed here was entered by the user; the program contains no standard value."""
from __future__ import annotations

from ..space.environments import KIND_UNITS
from .word import num, vec

MAX_ROWS = 30
_VIA = {"gui": "GUI (the user)", "mcp": "MCP interface (an agent) - to be checked by the user"}


def _conf_text(obj) -> str:
    if not obj.is_confirmed():
        return "NOT confirmed" + (" (edited after the confirmation)" if obj.confirmed else "")
    return f"confirmed by {obj.confirmed_by} ({obj.date}" + (f", via {_VIA.get(obj.confirmed_via, obj.confirmed_via)}" if obj.confirmed_via else "") + ")"


def _mesh_of(sess):
    res = getattr(sess, "results", None)
    if res is not None and getattr(res, "mesh", None) is not None:
        return res.mesh
    return sess.valid_mesh() if hasattr(sess, "valid_mesh") else None


def has_loads(study, sess) -> bool:
    from ..space import environments as EV
    from ..space import loads as LD
    sp = getattr(study, "space", None)
    if sp is None:
        return False
    return bool(EV.environments_of(sp) is not None or sp.extra.get(LD.QSL_KEY) or LD.recall(sess, study, _mesh_of(sess)) is not None)


def environment_tables(D, es) -> None:
    D.kv([("Environments", _conf_text(es)), ("Launch axis", es.launch_axis.upper() or "not set"), ("QSL combination", es.qsl_combination),
          ("Modal damping ratio of the launch analysis", (f"{num(es.damping_zeta, 4)} (source: {es.damping_source or 'not stated'})" if es.damping_zeta is not None
                                                          else "NOT entered in the environments - the damping setting of the study is used, no margin from the 3-axis analysis can be declared passed")),
          ("Notes", es.notes or "–")])
    rows = []
    for lv in es.levels:
        for k in lv.kinds():
            t = lv.table(k)
            if t is None:
                continue
            f = [p[0] for p in t.points]
            extra = []
            if t.sweep_rate_oct_min:
                extra.append(f"sweep {num(t.sweep_rate_oct_min)} oct/min" + (f" x{t.sweeps}" if t.sweeps else ""))
            if t.duration_s:
                extra.append(f"{num(t.duration_s)} s per axis")
            if t.sigma:
                extra.append(f"{num(t.sigma)} sigma")
            rows.append([lv.name, k, len(t.points), f"{num(min(f))} - {num(max(f))}", "; ".join(extra) or "–", t.document or "–", t.revision or "–", t.clause or "–"])
    if rows:
        D.table(["Level", "Kind", "Points", "Frequency range [Hz]", "Duration / sweep", "Document", "Revision", "Clause"], rows,
                caption="Launch environments entered by the user", font=8)
        for lv in es.levels:                                            # the values themselves: a report that prints only the range cannot be checked against the source
            for k in lv.kinds():
                t = lv.table(k)
                if t is not None and t.points:
                    D.p(f"{lv.name} / {k} table [Hz; {KIND_UNITS[k][1]}]: " + "; ".join(f"{num(p[0], 6)}: {num(p[1], 6)}" for p in t.points[:MAX_ROWS * 4])
                        + (" ..." if len(t.points) > MAX_ROWS * 4 else ""), size=8)
    else:
        D.p("No environment table is entered.", italic=True)


def qsl_table(D, study) -> None:
    from ..space import loads as LD
    st = LD.qsl_status(study)
    names = set(st["names"])
    if not names:
        return
    rows = []
    for c in study.load_cases:
        if c.name in names and c.loads:
            v = [float(x) / LD.G_MM_S2 for x in c.loads[0].vector]
            rows.append([c.name, *[num(x, 4) for x in v]])
    D.p("Quasi-static launch cases: " + ("current with the requirement set and the environments" if st["current_with_requirement_set"]
                                         else "NOT current - " + "; ".join(st["problems"][:4])), italic=True, size=9)
    if st["current_with_requirement_set"] and not st["usable_for_verification"]:
        D.p("The cases are the generated ones, but the " + " and the ".join(n for n, ok in (("requirement set", st["requirement_set_confirmed"]),
                                                                                           ("launch environments", st["environment_confirmed"])) if not ok)
            + " are not confirmed: they are not a verification input yet.", italic=True, size=9)
    D.p("Generator limits: one axial factor for both signs; the laterals are the two global axes that are not the launch axis; 'axial_both_laterals' applies both "
        "lateral factors at once (resultant = sqrt of the sum of squares).", italic=True, size=8)
    if rows:
        D.table(["Case", "x [g]", "y [g]", "z [g]"], rows[:MAX_ROWS], caption=f"Quasi-static launch cases (body force = factor x g; launch axis {st['launch_axis'].upper()})",
                font=8)


def mass_tables(D, study, mesh, out: dict | None = None) -> list[str]:
    """Writes the mass tables. `out` (optional) receives `props` (the mass properties) and `budget` (the mass budget report), so that the caller does not compute them again."""
    from ..space import budgets as BG
    from ..space.mass_props import mass_properties
    warn: list[str] = []
    props = mass_properties(study, mesh)
    D.kv([("Total mass [kg]", num(props.mass_kg, 5)), ("Centre of mass [mm]", vec(props.cog, 5)), ("Mass of the bodies / added masses [kg]",
          f"{num(props.bodies_t * 1000.0, 5)} / {num(props.added_t * 1000.0, 5)}")])
    I = props.inertia_cog_kg_m2
    D.table(["", "x", "y", "z"], [[ax, *[num(float(I[i, j]), 5) for j in range(3)]] for i, ax in enumerate("xyz")],
            caption="Inertia tensor about the centre of mass [kg m2] (parallel-axis shifted from the mesh, point masses and surface masses)", font=8)
    w, _v = props.principal()
    D.p("Principal moments [kg m2]: " + ", ".join(num(float(x) * 1e-3, 5) for x in w), size=9)
    if props.parts:
        D.table(["Part", "Mass [kg]", "Centre of mass [mm]"], [[k, num(p["mass_t"] * 1000.0, 5), vec(p["cog_mm"], 5)] for k, p in list(props.parts.items())[:MAX_ROWS]],
                caption="Mass by part", font=8)
    for n in props.notes:
        D.p(n, italic=True, size=9)
    rep = BG.mass_budget(props, study.space)
    if out is not None:
        out["props"], out["budget"] = props, rep
    if rep.items:
        D.table(["Quantity", "Value", "Limit", "Unit", "Limit source", "Status"],
                [[i.name, num(i.value, 5), num(i.limit, 5), i.unit, i.source or "–", i.status] for i in rep.items], caption="Mass budget (limits of the requirement set)",
                status_col=5, font=8)
    for n in rep.notes:
        D.p(n, italic=True, size=9)
    return warn


def frequency_table(D, study, sess, res3):
    from ..space import budgets as BG
    from ..space import environments as EV
    data = BG.modal_data_of(sess.results) if getattr(sess, "results", None) is not None else None
    if data is not None:
        freq, frac, src = data
    elif res3 is not None:
        freq, frac, src = res3.freq, res3.eff_frac, "modal basis of the 3-axis run"
    else:
        return None
    es = EV.environments_of(study.space)
    rep = BG.frequency_budget(freq, frac, es.launch_axis if es is not None else "", study.space)
    D.p(f"Lowest natural frequencies [Hz] ({src}): " + ", ".join(num(float(f), 5) for f in sorted(freq)[:6]), size=9)
    if rep.items:
        D.table(["Quantity", "Value [Hz]", "Limit [Hz]", "Limit source", "Status"], [[i.name, num(i.value, 5), num(i.limit, 5), i.source or "–", i.status] for i in rep.items],
                caption="Minimum natural frequency (limits of the requirement set)", status_col=4, font=8)
    for n in rep.notes:
        D.p(n, italic=True, size=9)
    return rep                                               # the budget report (the sign-off of the ECSS report reads its statuses)


def envelope_tables(D, study, res3, mesh=None) -> None:
    from ..space import loads as LD
    conf = res3.confirmed_now(study, mesh)
    dm = res3.damping or {}
    D.kv([("Analysis", res3.label + " analysis of the level " + res3.level), ("Axes", ", ".join(LD.AXIS_NAMES[a] for a in res3.axes)),
          ("Modal basis", f"{len(res3.freq)} modes ({res3.engine or 'n/a'})"),
          ("Damping", (f"{dm.get('kind', 'modal')}, zeta {num(dm.get('zeta'), 4)}; source: {dm.get('source') or 'not stated'}" if dm else "not recorded")
           + ("" if dm.get("from_environment") else " - NOT entered in the environments")),
          ("Inputs", "environment and requirement set confirmed" if conf else "UNCONFIRMED_INPUTS: the environment or the requirement set is not confirmed, or the "
           "environment, the study, the mesh or the damping changed after the run - indicative only")])
    if res3.blockers:
        D.p("No margin that rests on this analysis can be declared passed: " + "; ".join(res3.blockers), italic=True, size=9, bold=True)
    rows = []
    for k in res3.kinds:
        e = res3.envelopes[k]
        j = int(e.S.argmax())
        pa = res3.per_axis[k]
        rows.append([k, *[num(float(pa[a].S.max()), 4) if a in pa else "–" for a in LD.AXES], num(float(e.S[j]), 4), LD.AXIS_NAMES[int(e.axis_S[j])],
                     num(float(e.U.max()), 4), num(float(e.D.max()), 3) if e.D is not None else "–",
                     num(float(e.D_axis_max.max()), 3) if e.D_axis_max is not None else "–", res3.sources.get(k, "–")])
    D.table(["Kind", "max S along X [MPa]", "Y", "Z", "Envelope max S [MPa]", "Governing axis", "Envelope max |U| [mm]", "Damage, summed over the axes", "Largest damage of one axis",
             "Source of the table"], rows,
            caption="3-axis dynamic envelopes (sine: max over frequency and phase; random: RMS = 1 sigma; shock: SRSS / CQC von Mises). The damage of the three axes is "
                    "ADDED (they are tested one after the other on the same hardware); the plan text says max(D) - the per-axis maximum is listed for information.", font=8)
    for k in res3.kinds:
        prow = [[name, num(p["max_S_MPa"], 4), p["axis_S"], vec(p["at_mm"], 4), num(p["max_U_mm"], 4), num(p["max_D"], 3) if "max_D" in p else "–"]
                for name, p in list(res3.parts.get(k, {}).items())[:MAX_ROWS]]
        if prow:
            D.table(["Part", "Max S [MPa]", "Axis", "Place [mm]", "Max |U| [mm]", "Max damage"], prow, caption=f"Envelope by part - {k} (nodal maxima, singularities not separated)",
                    font=8)
    if "random" in res3.kinds and res3.sigma is None:
        D.p("No sigma multiple is entered with the random table: the RMS stress is reported but not compared with an allowable.", italic=True, size=9)
    for n in res3.notes + res3.warnings[:10]:
        D.p(n, italic=True, size=9)


def loads_section(D, sess) -> list[str]:
    """The subsection; returns warnings."""
    from ..space import environments as EV
    from ..space import loads as LD
    study = sess.study
    warnings: list[str] = []
    if not has_loads(study, sess):
        return warnings
    D.h("Launch loads, mass properties and budgets", 2)
    D.p("The environments, quasi-static factors and budgets below were entered by the user with their sources; the program contains no launcher or standard value. A "
        "result that rests on an unconfirmed environment or requirement set is labelled so and is not a verification result.")
    es = EV.environments_of(study.space)
    if es is not None:
        environment_tables(D, es)
    qsl_table(D, study)
    res = getattr(sess, "results", None)
    mesh = res.mesh if res is not None else (sess.valid_mesh() if hasattr(sess, "valid_mesh") else None)
    if mesh is not None and len(mesh.nodes):
        try:
            warnings += mass_tables(D, study, mesh)
        except Exception as e:  # noqa: BLE001 - the report continues without the table
            warnings.append(f"Mass properties: {e}")
    res3 = LD.recall(sess, study, _mesh_of(sess))
    try:
        frequency_table(D, study, sess, res3)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Frequency check: {e}")
    if res3 is not None:
        envelope_tables(D, study, res3, _mesh_of(sess))
    return warnings

"""MCP group "space": requirement set, material allowables, margins of safety and FEM model quality checks of space structures (ECSS, `fea_core.space`).

The program ships no standard number: the requirement set is entered by the user and confirmed by the user. An agent may enter values the
user gave it, but must confirm (`confirmed_by`) only when the user asked for it - a result is `UNCONFIRMED_REQUIREMENTS` until then.
"""
from __future__ import annotations

from .common import out, run_with_timeout
from .session import FeaError, Session, user_file, user_path

_SET_KEYS = {"name", "standard", "revision", "verification_approach", "notes", "factors", "rules", "limits"}


def _space(st, create: bool = False):
    from ..space.model import SpaceSettings
    if st.space is None and create:
        st.space = SpaceSettings()
    return st.space


def _rs_view(rs, full: bool = False, space=None) -> dict:
    """Compact view of a requirement set: metadata, the filled factors (all rows with full=True), rules, problems."""
    from ..space.requirements import confirmation_problems, resolve_rule, validate
    d = rs.to_dict()
    rows = d["factors"]
    filled = [r for r in rows if r["value"] is not None]
    d["factors"] = rows if full else filled
    d["factor_rows"] = {"total": len(rows), "filled": len(filled)}
    d["is_confirmed"] = space.requirement_confirmed() if space is not None else rs.is_confirmed()
    d["rules_status"] = [{"check": r.check, "structure_class": r.structure_class,
                          "complete": resolve_rule(rs, r).complete if r.factors or r.thermal_factors else None,
                          "missing": resolve_rule(rs, r).missing[:4] if r.factors or r.thermal_factors else ["no factors listed"]}
                         for r in rs.rules]
    d["issues"] = [str(i) for i in validate(rs)][:30]
    d["to_confirm"] = confirmation_problems(rs, "x")[:10]
    return d


def _merge(rs, data: dict) -> list[str]:
    """Applies a partial update (upsert of factor rows by key + class + approach, rules by check + class). Returns the notes."""
    from ..space.model import FactorEntry, LimitEntry, MarginRule
    notes: list[str] = []
    bad = set(data) - _SET_KEYS
    if bad:
        raise FeaError(f"Unknown keys in data: {', '.join(sorted(bad))} (allowed: {', '.join(sorted(_SET_KEYS))})")
    try:
        for k in ("name", "standard", "revision", "verification_approach", "notes"):
            if k in data:
                setattr(rs, k, "" if data[k] is None else str(data[k]))
        for raw in data.get("factors") or []:
            new = FactorEntry.from_dict(raw)
            old = next((f for f in rs.factors if (f.key, f.structure_class, f.approach) == (new.key, new.structure_class, new.approach)), None)
            if old is None:
                rs.factors.append(new)
            else:
                for k, v in raw.items():
                    if k in ("value", "document", "revision", "clause", "note"):
                        setattr(old, k, getattr(new, k))
        for raw in data.get("rules") or []:
            new = MarginRule.from_dict(raw)
            old = next((r for r in rs.rules if (r.check, r.structure_class) == (new.check, new.structure_class)), None)
            if old is None:
                rs.rules.append(new)
            else:
                for k in raw:
                    if k in ("allowable", "stress_measure", "factors", "thermal_handling", "thermal_factors", "document", "revision",
                             "clause", "note", "not_applicable"):
                        setattr(old, k, getattr(new, k))
        for raw in data.get("limits") or []:
            new = LimitEntry.from_dict(raw)
            new.set_via = "mcp" if new.not_applicable else ""        # the interface is recorded by the program, never taken from the data
            old = next((x for x in rs.limits if x.key == new.key), None)
            if old is None:
                rs.limits.append(new)
            else:
                for k in raw:
                    if k in ("value", "unit", "document", "revision", "clause", "note", "not_applicable"):
                        setattr(old, k, getattr(new, k))
                if "not_applicable" in raw:
                    old.set_via = "mcp" if old.not_applicable else ""
    except (ValueError, TypeError, AttributeError) as e:
        raise FeaError(f"Invalid data: {e}") from None
    if rs.confirmed and not rs.is_confirmed():
        notes.append("the confirmation of the requirement set was withdrawn by this change - the user must confirm it again")
        rs.withdraw_confirmation()
    return notes


def _check_options(options: dict | None):
    """`options` -> ModelCheckOptions; unknown keys (a limit is NOT an option), non-numbers and out-of-range numbers are errors (the bounds are those of
    `fea_core.space.model_checks.option_problems`: every number is finite and inside a range that keeps a run meaningful and able to finish)."""
    from ..space import model_checks as MC
    if options is not None and not isinstance(options, dict):
        raise FeaError("options must be an object {name: value}")
    try:
        return MC.options_from_dict(options)
    except (ValueError, TypeError) as e:
        text = str(e)
        hint = (". Tolerances are not options: they come only from the confirmed requirement set (fea_space_requirement_set, data.limits)"
                if text.startswith("Unknown model-check option") else "")
        raise FeaError(text + hint) from None


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_space_requirement_set(action: str = "get", data: dict | None = None, confirmed_by: str | None = None,
                                  path: str | None = None, full: bool = False) -> str:
        """Requirement set of the space-structures verification (ECSS): get, template, set, confirm, import, export.
        NO standard number is built in: every factor comes from the set that the user entered and confirmed. action: get | template (installs the EMPTY template: factor rows by structure class x approach, rules, clause placeholders) | set (data = partial update: name, standard, revision, verification_approach qualification_model|protoflight|analysis_only, notes, factors [{key FOSY|FOSU|KQ|KP|KM|KLD|KMP, structure_class metallic|joints_inserts|buckling|composite, approach, value, document, revision, clause}], rules [{check yield|ultimate|buckling, structure_class, allowable Fy|Fu|Fsy|Fsu|lambda, stress_measure von_mises|max_principal|max_shear, factors [keys multiplied into the design factor J], thermal_handling same_as_mechanical|separate|none, thermal_factors, document, revision, clause (EACH rule needs all three, like each factor with a value and each limit), note, not_applicable true = the user declares the check not applicable to the class (needs document, revision, clause)}]) | validate | confirm (ONLY when the user asked, in a call of its own without data: confirmed_by = the user's name; checks provenance and completeness; stored as confirmed_via=mcp, which the report and the panel show) | unconfirm | import / export (path .json; .csv = factor table). MoS = allowable/(J·σ) - 1. Any edit withdraws the confirmation; without a confirmed set the margins are UNCONFIRMED_REQUIREMENTS."""
        from ..space import requirements as RQ
        from ..space.model import RequirementError
        with sess.use():
            st = sess.require_study()
            sp = _space(st)
            rs = sp.requirement_set if sp is not None else None
            act = action.strip().lower()
            notes: list[str] = []
            if confirmed_by is not None and confirmed_by.strip() and (act != "confirm" or data):
                # an agent must not set values and confirm them in one call: the confirmation is a separate step after the user checked
                raise FeaError("confirmed_by is accepted only by action='confirm' in a call of its own (without data): enter the values first, "
                               "let the user check them, then confirm")
            try:
                if act == "get":
                    if rs is None:
                        return out({"requirement_set": None, "hint": "action='template' installs the empty ECSS template"})
                    return out({"requirement_set": _rs_view(rs, full, sp)})
                if act == "template":
                    if rs is not None and (rs.factors or rs.rules) and not (data or {}).get("replace"):
                        raise FeaError("A requirement set exists - pass data={'replace': true} to replace it by the empty template")
                    _space(st, True).requirement_set = rs = RQ.ecss_32_10c_template()
                    notes.append("empty template: no factor has a value; fill the factors and the rules, then confirm")
                elif rs is None:
                    raise FeaError("There is no requirement set - action='template' (or 'import') first")
                elif act == "set":
                    if not data:
                        raise FeaError("data is missing")
                    notes += _merge(rs, data)
                elif act == "validate":
                    pass
                elif act == "confirm":
                    if not confirmed_by or not confirmed_by.strip():
                        raise FeaError("confirmed_by (the name of the user who confirms) is required - confirm only when the user asked")
                    RQ.confirm(rs, confirmed_by, via="mcp", scope=_space(st, True))
                elif act == "unconfirm":
                    rs.withdraw_confirmation()
                elif act == "import":
                    if not path:
                        raise FeaError("path is missing")
                    p = user_file(path, max_bytes=RQ.MAX_JSON_BYTES, what="Requirement set file")
                    text = p.read_text(encoding="utf-8-sig", errors="strict")
                    if p.suffix.lower() == ".csv":
                        rs.factors = RQ.factors_from_csv(text)
                        rs.withdraw_confirmation()
                    else:
                        _space(st, True).requirement_set = rs = RQ.from_json(text)
                    notes.append("imported: the confirmation is not taken over - the user must confirm the set again")
                elif act == "export":
                    if not path:
                        raise FeaError("path is missing")
                    p = user_path(path)
                    if p.suffix.lower() not in (".json", ".csv"):
                        p = p.with_suffix(".json")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(RQ.factors_to_csv(rs) if p.suffix.lower() == ".csv" else RQ.to_json(rs), encoding="utf-8")
                    return out({"exported": str(p), "bytes": p.stat().st_size})
                else:
                    raise FeaError("action must be get | template | set | validate | confirm | unconfirm | import | export")
            except RequirementError as e:
                text = "; ".join(e.issues[:12])
                if "document / revision / clause" in text:
                    text += (" | every margin rule (data.rules), every factor with a value (data.factors) and every limit (data.limits) needs ALL THREE of document, revision and clause "
                             "(the clause placeholder '<clause>' does not count)")
                raise FeaError(text) from None
            except UnicodeDecodeError:
                raise FeaError("The file is not UTF-8 text") from None
            d = {"requirement_set": _rs_view(rs, full, _space(st))}
            if notes:
                d["notes"] = notes
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_space_allowables_set(material: str, basis: str | None = None, Fy: float | None = None, Fu: float | None = None,
                                 Fsy: float | None = None, Fsu: float | None = None, temp_C: float | None = None,
                                 source: str | None = None, confirmed_by: str | None = None, clear: bool = False) -> str:
        """Allowables (A/B basis) of a material for the margins of safety, entered by the user.
        Values in MPa from the material handbook / data sheet: basis A | B, Fy / Fu tension yield / ultimate, Fsy / Fsu shear, temp_C, source (document). Only the given fields change; any edit withdraws the confirmation; confirmed_by = the user's name confirms the record (ONLY when the user asked, in a call of its own without values; stored as confirmed_via=mcp). Materials without confirmed A/B allowables give no verified margin. No value arguments: shows the record; clear=true deletes it."""
        from ..space import requirements as RQ
        from ..space.model import MaterialAllowable, RequirementError
        with sess.use():
            st = sess.require_study()
            mat = st.materials.get(material)
            if mat is None:
                raise FeaError(f"Material '{material}' is not in the study (materials: {', '.join(st.materials) or 'none'})")
            given = {k: v for k, v in (("basis", basis), ("Fy", Fy), ("Fu", Fu), ("Fsy", Fsy), ("Fsu", Fsu), ("temp_C", temp_C),
                                       ("source", source)) if v is not None}
            if confirmed_by is not None and confirmed_by.strip() and (given or clear):
                raise FeaError("confirmed_by must be given in a call of its own, without values: enter the allowables first, let the user "
                               "check them against the source, then confirm")
            if clear:
                mat.allowables = None
                return out({"material": material, "allowables": None})
            rec = RQ.allowable_of(mat) or MaterialAllowable()
            try:
                if given:
                    d = rec.to_dict()
                    d.update(given)
                    rec = MaterialAllowable.from_dict(d)
                    RQ.set_allowable(mat, rec)
                if confirmed_by is not None:
                    RQ.confirm_allowable(mat, confirmed_by, via="mcp")
            except (RequirementError, ValueError) as e:
                raise FeaError(str(e)) from None
            rec = RQ.allowable_of(mat)
            return out({"material": material, "allowables": rec.to_dict() if rec else None,
                        "is_confirmed": bool(rec and rec.is_confirmed()), "is_verified": bool(rec and rec.is_verified()),
                        "problems": rec.problems() if rec else ["no allowables entered"]})

    @mcp.tool(structured_output=False)
    def fea_space_margins(top: int = 10, justify: list[dict] | None = None, author: str | None = None,
                          part_classes: dict | None = None, default_class: str | None = None) -> str:
        """Margins of safety of the solved study outside singularities, verdict PASS | FAIL | UNCONFIRMED_REQUIREMENTS.
        MoS = allowable/(J·σ) - 1 per part, load state / combination and check (yield, ultimate; buckling factors of modal results); returns the lowest `top` and the singularity justification table (each singular place needs a note before sign-off: justify = [{part, xyz [x,y,z], note, cause?}], author). Never PASS while the requirement set (or the structure classes), a rule or an allowable is unconfirmed, a rule or case is missing, the analysis is nonlinear or a part is not assessed; verdict_text says when a PASS is not a sign-off yet. part_classes = {part: metallic|joints_inserts|composite}, default_class (metallic) for the others. Linear studies only."""
        from ..space import margins as MG
        from ..space.model import PART_CLASSES, SpaceSettings
        with sess.use(op="margins of safety"):
            st = sess.require_study()
            res = sess.require_results()
            if not res.steps and not [m for m in res.modes if getattr(m, "kind", "") == "buckling"]:
                raise FeaError("The results contain no load state - solve the study first (fea_solve_run)")
            sp = _space(st, bool(part_classes or default_class is not None or justify)) or SpaceSettings()
            if part_classes:
                bad = {k: v for k, v in part_classes.items() if v not in PART_CLASSES}
                if bad:
                    raise FeaError(f"Unknown structure class {bad} (allowed: {', '.join(PART_CLASSES)})")
                sp.part_classes.update({str(k): str(v) for k, v in part_classes.items()})
            if default_class is not None:
                if default_class not in PART_CLASSES:
                    raise FeaError(f"default_class must be one of {', '.join(PART_CLASSES)}")
                sp.default_class = default_class
            for j in justify or []:
                try:
                    MG.add_justification(sp, str(j["part"]), [float(v) for v in j["xyz"]], str(j["note"]), cause=str(j.get("cause", "")),
                                         author=str(author or ""))
                except (KeyError, TypeError, ValueError) as e:
                    raise FeaError(f"Invalid justification {j!r}: {e} (need part, xyz [x,y,z], note)") from None
            if not 1 <= int(top) <= 200:
                raise FeaError("top must be 1-200")
            from ..space import loads as LD
            summ = LD.sign_off_margins(st, res, sess, sess.valid_mesh())          # with the kept 3-axis envelopes when a current result exists
            return out(summ.to_dict(int(top)))

    @mcp.tool(structured_output=False)
    def fea_space_model_checks(checks: list[str] | None = None, options: dict | None = None, timeout_s: float = 3600) -> str:
        """FEM model quality checks (ECSS-E-ST-32-11C 6.4.1 and common FEM practice), run before the margins are used. checks (default all 7): mass, free_free (six rigid-body modes), unit_displacement (unit motions of the launch interface), isothermal (no stress under uniform dT), effective_mass, mesh_convergence (slow), equilibrium. Per check: status PASS | WARN | FAIL | UNCONFIRMED | SKIPPED, numbers, limits used, findings; overall verdict FAIL > INCOMPLETE (fewer than 7 checks) > UNCONFIRMED_REQUIREMENTS > WARN > PASS.
        No tolerance is built in and none can be passed in: they are read ONLY from `limits` of the requirement set that the USER confirmed (fea_space_requirement_set data.limits [{key, value, document, revision, clause}]; the answer lists the keys and their state). Without a confirmed limit a check reports its numbers but is UNCONFIRMED, never PASS; only physical invariants can FAIL without one. options = budgets / numerical settings, not limits (unknown name = error, every number is bounded): mass_budget_t [t], cog_budget_mm [x,y,z], part_mass_budget_t {part: t}, interface_selections [names, default every support; naming every support = no stray support can be found, UNCONFIRMED], solver_mass_check (false = check 1 is incomplete, never PASS), unit_states (all six UX UY UZ RX RY RZ for a full check 3; fewer = no PASS, none = SKIPPED), density_range_t_mm3 / cte_range_per_K (wider than the default = no full PASS), thermal_dT_K (0.01..1000), thermal_alpha_per_K, free_free_modes, modes, max_modes, convergence_frequencies / _levels / _ratio (element-size ratio of the levels, 1.3..4: a smaller one cannot be judged; fewer levels / frequencies or a smaller ratio than the default 3 / 5 / 1.5 = UNCONFIRMED, never PASS) / _case, convergence_max_dof (DOF of the finest level, 1e4..2e7), include_convergence, threads. Every setting that differs from the default is printed next to the check it changes (answer: settings_that_differ_from_the_defaults, text) and in the report; the answer also says who confirmed the requirement set and via which interface (confirmed via mcp = an agent: the user must check it). Uses the valid mesh and results of the session, works on copies, keeps the run for the Word report."""
        from ..space import model_checks as MC
        _check_options(options)
        ids = list(checks) if checks else None
        if ids is not None:
            unknown = [c for c in ids if c not in MC.CHECK_IDS]
            if unknown:
                raise FeaError(f"Unknown check(s) {', '.join(map(str, unknown))} (known: {', '.join(MC.CHECK_IDS)})")
        with sess.use(op="model quality checks", wait=5):
            sess.require_study()
            logs: list[str] = []
            try:
                return out(run_with_timeout(lambda cancel: MC.run_for_session(sess, options=options, checks=ids, log=logs.append, cancel=cancel),
                                            timeout_s, "Model quality checks"))
            finally:
                for line in logs:
                    sess.add_log(line)

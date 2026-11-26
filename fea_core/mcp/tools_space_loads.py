"""MCP group "space" (stage S2): launch environments, QSL load cases, the 3-axis dynamic envelopes and the mass / centre-of-mass / frequency budgets.

The program ships no launcher or standard number. QSL factors and budgets are named limits of the user's requirement set (`fea_space_requirement_set`,
`limits`), the environment tables (sine, random ASD, shock SRS) are entered with their source in `fea_space_env_set`. An agent may enter values the user gave it,
but must confirm (`confirmed_by`) only when the user asked for it, in a call of its own without data; results rest on the confirmed values or say they do not.
"""
from __future__ import annotations

from mcp.server.mcpserver.context import Context

from .common import out, run_with_timeout
from .session import FeaError, Session, user_file, user_path
from .tools_setup import gate, strip_ctx

_ENV_KEYS = {"launch_axis", "qsl_combination", "notes", "levels", "remove_levels", "damping_zeta", "damping_source"}
_TABLE_KEYS = {"points", "duration_s", "sweep_rate_oct_min", "sweeps", "sigma", "document", "revision", "clause", "note"}


def _space(st, create: bool = False):
    from ..space.model import SpaceSettings
    if st.space is None and create:
        st.space = SpaceSettings()
    return st.space


def _env_view(es, space) -> dict:
    from ..space import environments as EV
    d = es.to_dict()
    d["is_confirmed"] = es.is_confirmed()
    d["problems"] = EV.validate_environment(es)[:30]
    d["to_confirm"] = EV.confirmation_problems(es, "x")[:10]
    d["levels_with_tables"] = {lv.name: lv.kinds() for lv in es.levels}
    return d


def _merge_env(es, data: dict) -> list[str]:
    """Partial update of the environment set: launch_axis, qsl_combination, notes; levels = [{name, combine, missing_mass, sine|random|shock: {points, ...}}]
    (a level / a table is created when absent, only the fields given change; a table of None removes it); remove_levels = [names]."""
    from ..space.environments import ENV_KINDS, MAX_NOTE, EnvTable, bounded_text, parse_number
    bad = set(data) - _ENV_KEYS
    if bad:
        raise FeaError(f"Unknown keys in data: {', '.join(sorted(bad))} (allowed: {', '.join(sorted(_ENV_KEYS))})")
    notes: list[str] = []
    try:
        for k in ("launch_axis", "qsl_combination", "notes"):
            if k in data:
                setattr(es, k, "" if data[k] is None else bounded_text(data[k], k, MAX_NOTE).strip().lower() if k != "notes" else bounded_text(data[k], k, MAX_NOTE))
        if "damping_zeta" in data:
            es.damping_zeta = parse_number(data["damping_zeta"], "damping_zeta")
        if "damping_source" in data:
            es.damping_source = bounded_text(data["damping_source"], "damping_source")
        for name in data.get("remove_levels") or []:
            es.levels = [lv for lv in es.levels if lv.name != str(name)]
        for raw in data.get("levels") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                raise FeaError("every level needs a name (qualification | protoflight | acceptance)")
            lv = es.ensure_level(str(raw["name"]).strip().lower())
            if "combine" in raw:
                lv.combine = str(raw["combine"]).strip().lower()
            if "missing_mass" in raw:
                lv.missing_mass = bool(raw["missing_mass"])
            for kind in ENV_KINDS:
                if kind not in raw:
                    continue
                t = raw[kind]
                if t is None:
                    setattr(lv, kind, None)
                    continue
                if not isinstance(t, dict):
                    raise FeaError(f"{kind}: an object with points, ... is expected")
                badk = set(t) - _TABLE_KEYS
                if badk:
                    raise FeaError(f"{kind}: unknown keys {', '.join(sorted(badk))} (allowed: {', '.join(sorted(_TABLE_KEYS))})")
                cur = lv.table(kind)
                merged = {**(cur.to_dict() if cur is not None else {"kind": kind}), **t, "kind": kind}
                setattr(lv, kind, EnvTable.from_dict(merged))
    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        raise FeaError(f"Invalid data: {e}") from None
    return notes


def _mesh_of(sess: Session):
    mesh = sess.valid_mesh()
    if mesh is None and sess.results is not None:
        mesh = sess.results.mesh
    if mesh is None or not len(mesh.nodes):
        raise FeaError("There is no valid mesh - mesh the study first (fea_mesh_generate).")
    return mesh


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_space_env_set(action: str = "get", data: dict | None = None, confirmed_by: str | None = None, path: str | None = None) -> str:
        """Launch environments (ECSS): sine table, random ASD, shock SRS per level (qualification | protoflight | acceptance), launch axis, QSL combination. NOTHING is built in: the user enters the tables from the launcher guide / requirements WITH the source (document, revision, clause).
        action: get | template (empty levels) | set (data partial update: launch_axis x|y|z, qsl_combination none|axial_lateral|axial_both_laterals, remove_levels [names], damping_zeta (0-1) + damping_source (part of the confirmation), levels [{name, combine srss|cqc, missing_mass, sine {points [[Hz, g]], sweep_rate_oct_min, sweeps, document, revision, clause}, random {points [[Hz, g2/Hz]], duration_s, sigma (optional multiple of the RMS stress used in margins), source}, shock {points [[Hz, SRS g]], source}}]) | validate | confirm (ONLY when the user asked, alone, without data: confirmed_by = the user's name) | unconfirm | import / export (path .json) | limits (adds EMPTY rows qsl_axial_g, qsl_lateral_g, qsl_lateral_2_g, mass_max_kg, mass_min_kg, cog_min/max_x|y|z_mm, f_min_axial_hz, f_min_lateral_hz, f_min_eff_mass_fraction to the requirement set; fill them with fea_space_requirement_set set data={limits:[{key, value, unit, document, revision, clause}]}). Any edit withdraws the confirmation; unconfirmed inputs never give PASS."""
        from ..space import budgets as BG
        from ..space import environments as EV
        from ..space.model import RequirementError
        with sess.use():
            st = sess.require_study()
            sp = _space(st)
            es = EV.environments_of(sp)
            act = action.strip().lower()
            notes: list[str] = []
            if confirmed_by is not None and confirmed_by.strip() and (act != "confirm" or data):
                raise FeaError("confirmed_by is accepted only by action='confirm' in a call of its own (without data): enter the values first, "
                               "let the user check them against the source, then confirm")
            try:
                if act == "get":
                    if es is None:
                        return out({"environments": None, "hint": "action='template' creates the empty levels"})
                    return out({"environments": _env_view(es, sp)})
                if act == "template":
                    if es is not None and any(lv.kinds() for lv in es.levels) and not (data or {}).get("replace"):
                        raise FeaError("Environments exist - pass data={'replace': true} to replace them by the empty template")
                    es = EV.environment_template()
                    notes.append("empty template: no table has points; enter the tables with their source, then confirm")
                elif act == "limits":
                    rs = sp.requirement_set if sp is not None else None
                    if rs is None:
                        raise FeaError("There is no requirement set - fea_space_requirement_set action='template' first")
                    added = BG.install_limit_rows(rs)
                    if added and rs.confirmed:
                        rs.withdraw_confirmation()
                        notes.append("the confirmation of the requirement set was withdrawn: limit rows were added")
                    return out({"added": added, "limits": [x.to_dict() for x in rs.limits], "notes": notes})
                elif es is None and act != "import":
                    raise FeaError("There are no environments - action='template' (or 'import') first")
                elif act == "set":
                    if not data:
                        raise FeaError("data is missing")
                    assert es is not None
                    notes += _merge_env(es, data)
                elif act == "validate":
                    pass
                elif act == "confirm":
                    if not confirmed_by or not confirmed_by.strip():
                        raise FeaError("confirmed_by (the name of the user who confirms) is required - confirm only when the user asked")
                    assert es is not None
                    EV.confirm_environment(es, confirmed_by, via="mcp")
                elif act == "unconfirm":
                    assert es is not None
                    es.withdraw_confirmation()
                elif act == "import":
                    if not path:
                        raise FeaError("path is missing")
                    p = user_file(path, max_bytes=2 << 20, what="Environment file")
                    es = EV.from_json(p.read_text(encoding="utf-8-sig", errors="strict"))
                    notes.append("imported: the confirmation is not taken over - the user must confirm the environments again")
                elif act == "export":
                    if not path:
                        raise FeaError("path is missing")
                    assert es is not None
                    p = user_path(path)
                    if p.suffix.lower() != ".json":
                        p = p.with_suffix(".json")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(EV.to_json(es), encoding="utf-8")
                    return out({"exported": str(p), "bytes": p.stat().st_size})
                else:
                    raise FeaError("action must be get | template | set | validate | confirm | unconfirm | import | export | limits")
            except RequirementError as e:
                raise FeaError("; ".join(e.issues[:8])) from None
            except UnicodeDecodeError:
                raise FeaError("The file is not UTF-8 text") from None
            assert es is not None
            was = es.confirmed
            EV.store_environments(_space(st, True), es)
            if was and not es.confirmed:
                notes.append("the confirmation of the environments was withdrawn by this change - the user must confirm them again")
            d = {"environments": _env_view(es, _space(st))}
            if notes:
                d["notes"] = notes
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_space_qsl_generate(replace: bool = True, remove: bool = False) -> str:
        """Signed quasi-static launch load cases ('QSL +X', 'QSL -Y +Z', ...; one gravity state each, body force = factor x 9810 mm/s2) from the requirement-set limits qsl_axial_g, qsl_lateral_g, qsl_lateral_2_g along the launch axis, combined as the environments say (qsl_combination). Both signs of every axis. replace=True replaces earlier generated cases, remove=True only deletes them. Discards results (solve again)."""
        from ..space import loads as LD
        with sess.use():
            st = sess.require_study()
            if remove:
                n = LD.remove_qsl(st)
                sess.changed()
                return out({"removed": n, "load_cases": [c.name for c in st.load_cases]})
            try:
                cases = LD.generate_qsl(st, replace=replace)
            except LD.LoadsError as e:
                raise FeaError(str(e)) from None
            sess.changed()
            return out({"generated": [c.to_dict() for c in cases], "status": LD.qsl_status(st), "load_cases_total": len(st.load_cases),
                        "note": "solve the study (fea_solve_run) to get the states; fea_space_margins then assesses them"})

    @mcp.tool(structured_output=False)
    def fea_space_run_3axis(level: str, kinds: list[str] | None = None, axes: str = "xyz", margins: bool = False, top: int = 10,
                            timeout_s: float = 1800, confirmed: str | None = None, ctx: Context | None = None) -> str:
        """Applies the sine / random / shock environment of a level along X, Y and Z over ONE modal basis (study.dynamics: modes, f_max; damping = the environment's damping_zeta, else the study's with a warning) and returns per-axis maxima, per-node and per-part ENVELOPES (max over axes of |U| and von Mises with the governing axis; fatigue damage SUMMED over the axes) and the lowest frequencies. Needs a meshed, constrained study; no prestress. kinds = subset of sine|random|shock; axes = letters of xyz; a subset is labelled partial. random = RMS (1 sigma), in margins only as sigma x RMS (no sigma = no PASS). Blockers (partial run, ANY mode below the shock table, no sigma, damping not in the environments) forbid a PASS. margins=True adds the margins of the solved static states plus the envelopes (needs fea_solve_run first); unconfirmed inputs = UNCONFIRMED_REQUIREMENTS, never PASS. The result is kept for fea_space_mass_budget, fea_space_margins and the report only while study, mesh, damping and environments are unchanged (else 're-run', no static PASS). confirmed: as in fea_solve_run."""
        from ..space import loads as LD
        conf = gate(sess, confirmed, ctx)
        letters = [c for c in axes.lower() if c not in " ,;"]
        if not letters or any(c not in "xyz" for c in letters):
            raise FeaError("axes must contain only the letters x, y, z (e.g. 'xyz')")
        ax = tuple(sorted({"xyz".index(c) for c in letters}))
        if not 1 <= int(top) <= 200:
            raise FeaError("top must be 1-200")
        with sess.use(op="3-axis launch analysis", wait=5):
            st = sess.require_study()
            try:
                LD.resolve_level(st, level.strip().lower(), kinds)          # cheap checks first: no meshing for a level that cannot run
            except LD.LoadsError as e:
                raise FeaError(str(e)) from None
            from .tools_setup import verify
            verify(sess, conf)
            from .tools_mesh import do_mesh, readiness
            rd = readiness(sess)
            if not rd["ready"]:
                raise FeaError("The study is not ready for calculation:\n  " + "\n  ".join(rd["errors"]))
            if sess.valid_mesh() is None and not sess.frame_only:
                do_mesh(sess, timeout_s, None)
            mesh = _mesh_of(sess)
            wd = (sess.workdir or sess.new_workdir(st.name)) / "space_3axis"
            logs: list[str] = []

            def work(cancel):
                return LD.run_3axis(st, mesh, wd, logs.append, level=level.strip().lower(), kinds=kinds, axes=ax, cancel=cancel)
            try:
                res3 = run_with_timeout(work, timeout_s, "3-axis launch analysis")
            except LD.LoadsError as e:
                raise FeaError(str(e)) from None
            LD.remember(sess, res3)
            d = res3.summary()
            if margins:
                res = sess.require_results()
                if not res.steps:
                    raise FeaError("margins=True needs the solved static states - run fea_solve_run first (the 3-axis result was computed and kept)")
                d["margins"] = LD.margins_with_envelopes(st, res, res3).to_dict(int(top))
            return out(d)

    fea_space_run_3axis.ipc_args = strip_ctx          # the window gets no MCP context (its own confirmation applies there)

    @mcp.tool(structured_output=False)
    def fea_space_mass_budget(parts: bool = False) -> str:
        """Mass, centre of mass and inertia tensor about it (mesh + point masses with rotary inertia + surface masses, Steiner-shifted; kg, mm, kg*m2) against the requirement-set limits mass_max_kg, mass_min_kg, cog_min/max_x|y|z_mm, and the first axial / lateral natural frequency against f_min_axial_hz / f_min_lateral_hz (axial = launch axis; f_min_eff_mass_fraction = share from which ONE mode is significant: such a mode below f_min = FAIL; basis short of f_min, < 90 % mass computed or split mass = UNCONFIRMED). Needs a mesh; frequencies need modal results or a 3-axis run. PASS / FAIL only with a confirmed requirement set and limits with source, else UNCONFIRMED. parts=True adds per-part values."""
        from ..space import budgets as BG
        from ..space import environments as EV
        from ..space import loads as LD
        from ..space.mass_props import mass_properties
        with sess.use(op="mass properties"):
            st = sess.require_study()
            mesh = _mesh_of(sess)
            try:
                props = mass_properties(st, mesh)
            except ValueError as e:
                raise FeaError(str(e)) from None
            sp = st.space
            d: dict = {"mass_properties": props.to_dict(), "mass_budget": BG.mass_budget(props, sp).to_dict()}
            if not parts:
                d["mass_properties"].pop("parts", None)
            es = EV.environments_of(sp)
            axis = es.launch_axis if es is not None else ""
            data = BG.modal_data_of(sess.results) if sess.results is not None else None
            src = ""
            if data is not None:
                freq, frac, src = data
            else:
                r3 = LD.recall(sess, st, mesh)                                  # only a result of THIS study and mesh (digest), never a stale one
                freq, frac, src = (r3.freq, r3.eff_frac, "modal basis of the 3-axis run") if r3 is not None else (None, None, "")
            if freq is not None and len(freq):
                d["frequency_check"] = {**BG.frequency_budget(freq, frac, axis, sp).to_dict(), "source": src, "launch_axis": axis,
                                        "lowest_Hz": [float(f) for f in sorted(freq)[:6]]}
            else:
                d["frequency_check"] = {"verdict": "SKIPPED", "notes": ["no natural frequencies yet - enable fea_modal_set and solve, or run fea_space_run_3axis"]}
            d["requirement_set_confirmed"] = bool(sp is not None and sp.requirement_confirmed())
            d["environment_confirmed"] = EV.environment_confirmed(sp)                   # the frequency items need it (the launch axis decides which limit applies)
            return out(d)

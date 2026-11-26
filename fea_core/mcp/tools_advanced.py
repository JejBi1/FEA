"""Tools: ε-N (strain-life) fatigue of a stress history, EN 1990 load case categories and the simple parametric sweep (DOE)
as a background job – MCP wiring of fea_core.fatigue.strain_life, fea_core.combinations and fea_core.sweep (the engine of
fea_param_study_run in tools_param.py; fea_sweep_define / fea_sweep_run cover load factors, materials and mesh sizes with a
background job). The submodel lives in tools_submodel.py.

Groups: fea_loadcase_set_category, fea_combination_generate_from_study → results; fea_fatigue_strain_life → fatigue;
fea_sweep_define, fea_sweep_run → model (groups.OVERRIDES).
"""
from __future__ import annotations

import copy
import math
import threading
import time

import numpy as np

from mcp.server.mcpserver.context import Context

from .common import out, run_with_timeout
from .session import FeaError, Session
from .tools_setup import gate, verify, without_ctx

SWEEP_KEY = "_fea_sweep"                 # sess.__dict__: {"spec": ParametricStudySpec, "extra_materials": {}, "result"}
EN_KEYS = {"E", "sigma_f_prime", "b", "eps_f_prime", "c", "K_prime", "n_prime", "source"}


# ===================================================================== ε-N fatigue
def _en_params(st, en_params) -> dict:
    if not en_params:
        return {}
    if not isinstance(en_params, dict):
        raise FeaError("en_params: {material: {E, sigma_f_prime, b, eps_f_prime, c, K_prime, n_prime}} or one such set for all")
    if set(en_params) & (EN_KEYS - {"source"}):
        en_params = {m: dict(en_params) for m in st.materials}
    outp = {}
    for mname, p in en_params.items():
        if mname not in st.materials:
            raise FeaError(f"en_params: material '{mname}' is not in the study ({', '.join(st.materials)})")
        if not isinstance(p, dict):
            raise FeaError(f"en_params['{mname}'] must be a dictionary")
        bad = set(p) - EN_KEYS
        missing = (EN_KEYS - {"source"}) - set(p)
        if bad or missing:
            raise FeaError(f"en_params['{mname}']: " + (f"unknown keys {sorted(bad)}" if bad else f"missing {sorted(missing)}")
                           + " (E [MPa], sigma_f_prime [MPa], b, eps_f_prime, c, K_prime [MPa], n_prime)")
        try:
            vals = {k: float(v) for k, v in p.items() if k != "source"}
        except (TypeError, ValueError):
            raise FeaError(f"en_params['{mname}']: the values must be numbers") from None
        if not (vals["E"] > 0 and vals["sigma_f_prime"] > 0 and vals["eps_f_prime"] > 0 and vals["K_prime"] > 0
                and vals["n_prime"] > 0 and vals["b"] < 0 and vals["c"] < 0):
            raise FeaError(f"en_params['{mname}']: E, σ'f, ε'f, K', n' must be positive and b, c negative")
        outp[mname] = {**vals, "source": str(p.get("source") or "given via MCP")}
    return outp


def strain_life_core(sess: Session, history_curve, notch_rule: str, mean_correction: str, equivalent: str,
                     en_params, repeats: float, top_n: int) -> dict:
    from ..fatigue.settings import FatigueSettings
    from ..fatigue.analysis import run_fatigue
    from .tools_fatigue import parse_history
    st = sess.require_study()
    res = sess.require_results()
    if not res.steps or not len(res.mesh.nodes):
        raise FeaError("The ε-N analysis needs 3D results of mechanical load states (fea_solve_run)")
    if st.is_nonlinear():
        raise FeaError("ε-N by superposition of the states is not available for a nonlinear (contact) result")
    lcs = [s.name for s in res.steps]
    fs = FatigueSettings.from_dict(st.fatigue.to_dict()) if st.fatigue else FatigueSettings()
    if history_curve is not None:
        fs.history = parse_history(history_curve, lcs)
    elif not getattr(fs.history, "channels", None):
        raise FeaError("No load history – give history_curve (as fea_fatigue_run) or define it in the study fatigue settings")
    fs.method = "en"
    fs.equivalent = equivalent
    fs.en_notch_rule = notch_rule
    fs.en_mean_correction = mean_correction
    fs.mean_correction = mean_correction
    fs.repeats = float(repeats)
    fs.en_params.update(_en_params(st, en_params))
    old = st.fatigue
    st.fatigue = fs
    try:
        try:
            fr = run_fatigue(st, res)
        except ImportError as e:            # numba (module accel) missing
            raise FeaError(f"ε-N needs the module 'accel' (numba): {e}") from None
        except (KeyError, ValueError) as e:
            raise FeaError(f"ε-N fatigue cannot be computed: {e}") from None
    except Exception:
        st.fatigue = old
        raise
    sess.fatigue = fr
    sess.joints = None
    n = max(1, min(int(top_n), 50))
    if fr.damage.max() > 0:
        worst = fr.worst(res.mesh, n)
    else:
        order = np.argsort(fr.safety)[:n]
        worst = [{"node": int(j), "xyz": res.mesh.nodes[j], "damage_per_block": 0.0, "total_damage": 0.0,
                  "life_blocks": float("inf"), "safety": float(fr.safety[j]), "amplitude": float(fr.amplitude[j])}
                 for j in order]
    for w in worst:
        w["node"] += 1
    params = {m: v.get("en") for m, v in (fr.sn_used or {}).items() if isinstance(v, dict) and "en" in v}
    return {
        "method": fr.method_lines(), "notch_rule": notch_rule, "mean_correction": mean_correction,
        "equivalent": equivalent, "time_points": len(fr.t), "load_cases": fr.load_cases,
        "repeats_blocks": fr.repeats, "max_total_damage": float(fr.total_damage.max()),
        "min_life_blocks": float(fr.life_blocks.min()), "min_safety": float(np.min(fr.safety)),
        "verdict": ("PASS" if fr.total_damage.max() < 1 else "FAIL") + " (D < 1 for the given number of blocks)",
        "en_parameters": params, "worst": worst, "warnings": fr.warnings,
        "note": ("Elastic FEM stresses corrected to local elastic-plastic strain by the notch rule; parameters estimated "
                 "by the Uniform Material Law are indicative – give measured ones in en_params"),
        "hint": "Maps: fea_results_render_image(field='fatigue_damage' | 'fatigue_life' | 'fatigue_safety')",
    }


# ===================================================================== sweep
def _sweep_state(sess: Session) -> dict:
    return sess.__dict__.setdefault(SWEEP_KEY, {"spec": None, "extra_materials": {}, "result": None})


def _fresh_parts(sess: Session):
    """Own copy of the shapes of the study for a worker thread: the shapes of the session may be the ones displayed in the GUI
    window (the bridge hands them over), and meshing / Boolean operations on those from another thread collide with the 3D view."""
    from pathlib import Path
    from ..geometry import legacy_step_names, read_step
    if sess.step_path is None or not Path(sess.step_path).exists():
        raise FeaError(f"Geometry file not found: {sess.step_path}")
    parts = read_step(sess.step_path, legacy_names=legacy_step_names(sess.study))
    if not parts:
        raise FeaError(f"The STEP file contains no body: {sess.step_path}")
    return parts


def sweep_core(sess: Session, timeout_s: float, keep_results: bool, cancel: threading.Event | None = None,
               on_progress=None, on_log=None) -> dict:
    """Runs the defined sweep on a copy of the study (fea_core.sweep.run_parametric_study: the engine shared with
    fea_param_study_run). The valid mesh of the session is reused (the baseline and every point with the same mesh signature);
    the shapes for meshing are an own copy (see _fresh_parts). keep_results: see run_parametric_study."""
    from ..sweep import run_parametric_study
    stt = _sweep_state(sess)
    spec = stt["spec"]
    if spec is None:
        raise FeaError("No sweep is defined – fea_sweep_define(load_factors=…, materials=…, mesh_sizes=…)")
    st = sess.require_study()
    if not st.load_cases:
        raise FeaError("The study has no load case")
    work = copy.deepcopy(st)
    for name, m in stt["extra_materials"].items():
        work.materials.setdefault(name, copy.deepcopy(m))
    mesh = None if sess.frame_only else sess.valid_mesh()
    sweeps_mesh = any(q.kind == "mesh_size" for q in spec.all_params())
    parts = None if sess.frame_only or (mesh is not None and not sweeps_mesh) else _fresh_parts(sess)
    wd = (sess.workdir or sess.base_dir) / "sweep"
    total = len(spec.combinations())

    def prog(k, n):
        if on_progress is not None:
            on_progress(k / max(n, 1), (k + 1) / max(n, 1), f"Design point {k + 1}/{n}")
    t0 = time.time()
    r = run_with_timeout(lambda c: run_parametric_study(work, spec, wd, on_progress=prog, cancel=c, parts=parts, mesh=mesh,
                                                        on_log=on_log, keep_results=keep_results),
                         timeout_s, "Parametric sweep", cancel)
    stt["result"] = r
    d = r.to_dict()
    d.update({"time_s": round(time.time() - t0, 1), "points_defined": total, "workdir": str(wd),
              "table": r.summary_table(),
              "note": ("min_safety = Re / von Mises at the max. outside singularities (as the GUI Assessment card; None = material "
                       "without Re), min_safety_raw = Re / the raw maximum incl. singularities; Pareto front: mass, stress outside "
                       "singularities, safety" + ("; results of the points saved as DP_xxx/results.npz" if keep_results else ""))})
    return d


def register(mcp, sess: Session) -> None:

    # ------------------------------------------------------------------ ε-N
    @mcp.tool(structured_output=False)
    def fea_fatigue_strain_life(history_curve: dict | list | str | None = None, notch_rule: str = "neuber",
                                mean_correction: str = "morrow", equivalent: str = "mises_signed",
                                en_params: dict | None = None, repeats: float = 1e6, top_n: int = 10) -> str:
        """Strain-life (ε-N) fatigue on the current results: superposed elastic stress history -> notch rule (neuber | glinka) -> cyclic Ramberg-Osgood -> rainflow -> Coffin-Manson-Basquin life with mean_correction none (plain Coffin-Manson) | morrow | swt -> Miner. history_curve as fea_fatigue_run (None = study fatigue history). equivalent mises_signed | max_principal. en_params {material: {E, sigma_f_prime [MPa], b, eps_f_prime, c, K_prime [MPa], n_prime}} or one set for all (default: Uniform Material Law estimate from Rm). repeats = blocks; top_n worst nodes. Returns damage, life [blocks] and safety at the worst nodes and the parameters used."""
        rule = str(notch_rule).strip().lower()
        if rule not in ("neuber", "glinka"):
            raise FeaError("notch_rule: neuber | glinka")
        mean = str(mean_correction).strip().lower()
        if mean in ("coffin_manson", "coffin-manson", "cm"):
            mean = "none"
        if mean not in ("none", "morrow", "swt"):
            raise FeaError("mean_correction: none (Coffin-Manson) | morrow | swt")
        eq = {"mises_signed": "signed_mises", "signed_mises": "signed_mises", "max_principal": "max_principal",
              "principal": "max_principal"}.get(str(equivalent).strip().lower())
        if eq is None:
            raise FeaError("equivalent: mises_signed | max_principal (the critical plane is not available for ε-N)")
        if not float(repeats) > 0:
            raise FeaError("repeats must be positive")
        with sess.use(op="fatigue (ε-N)"):
            return out(strain_life_core(sess, history_curve, rule, mean, eq, en_params, float(repeats), int(top_n)))

    # ------------------------------------------------------------------ EN 1990 categories
    @mcp.tool(structured_output=False)
    def fea_loadcase_set_category(name: str, category: str | None = None) -> str:
        """Sets the EN 1990 category of load case `name` used by the combination generator: 'permanent' or 'G' (permanent action) or a psi category: A-F, H or cat_A ... cat_H (EN 1990 Table A1.1; traffic category G only as 'cat_G' / 'traffic_G'), snow_cz_low, snow_cz_high, wind, temperature. None/'' clears it (the generator then guesses from the name and warns). Results are kept. Returns all load cases with category, psi factors and source."""
        from ..combinations import case_category_rows, normalize_category
        with sess.use():
            st = sess.require_study()
            lc = next((x for x in st.load_cases if x.name == name), None)
            if lc is None:
                raise FeaError(f"Load case '{name}' does not exist (load cases: {', '.join(x.name for x in st.load_cases) or 'none'})")
            if category is None or not str(category).strip():
                lc.category = None
            else:
                try:
                    lc.category = normalize_category(category)
                except ValueError as e:
                    raise FeaError(str(e)) from None
            return out({"load_case": name, "category": lc.category, "load_cases": case_category_rows(st)})

    @mcp.tool(structured_output=False)
    def fea_combination_generate_from_study(formula: str = "6.10", categories: dict[str, str] | None = None,
                                            favorable: list[str] | None = None) -> str:
        """Generates EN 1990 + national annex combinations of ALL load cases from their categories (fea_loadcase_set_category); categories {case: category} overrides them for this call only. A case without a category is guessed from its name with a warning stored in each combination. formula '6.10' | '6.10a_6.10b' | 'EQU' | 'SLS_CHAR' | 'SLS_FREQ' | 'SLS_QUASI' | 'all'; favorable = cases acting favourably. Adds the combinations (same names are replaced) and returns them with the category source per case."""
        from ..combinations import generate_for_study, validate_combination
        with sess.use(op="combination_generate"):
            st = sess.require_study()
            if not st.load_cases:
                raise FeaError("The study has no load case")
            try:
                combs, warns = generate_for_study(st, formula, categories, favorable)
                for c in combs:
                    validate_combination(st, c, others=combs)
            except (ValueError, KeyError) as e:
                raise FeaError(str(e)) from None
            for c in combs:
                st.combinations = [old for old in st.combinations if old.name != c.name]
                st.combinations.append(c)
            return out({"generated_count": len(combs), "warnings": warns,
                        "categories": combs[0].case_categories if combs else {},
                        "combinations": [{k: v for k, v in c.to_dict().items() if k not in ("case_categories", "warnings")}
                                         for c in combs]})

    # ------------------------------------------------------------------ parametric sweep (DOE)
    @mcp.tool(structured_output=False)
    def fea_sweep_define(load_factors: list[float] | None = None, materials: dict[str, list[str]] | None = None,
                         mesh_sizes: list[float] | None = None, max_points: int = 50,
                         required_safety: float | None = None, clear: bool = False) -> str:
        """Defines the parametric sweep (DOE, full factorial) run by fea_sweep_run: load_factors (multiples of all loads), materials {part: [material names from the study or catalogue]}, mesh_sizes [mm]; max_points caps the points (limit 200); required_safety = Pareto feasibility. No arguments: returns the current definition, point count and last result. clear=True deletes it. The study itself is not changed."""
        from ..sweep import ParametricStudySpec
        stt = _sweep_state(sess)
        with sess.use():
            st = sess.require_study()
            if clear:
                stt.update(spec=None, extra_materials={}, result=None)
                return out({"cleared": True})
            if load_factors is None and materials is None and mesh_sizes is None:
                spec = stt["spec"]
                d = {"defined": spec is not None}
                if spec is not None:
                    d.update(spec=spec.to_dict(), points=min(spec.total(), spec.max_points), points_full=spec.total())
                if stt["result"] is not None:
                    d["last_result"] = {k: v for k, v in stt["result"].to_dict().items() if k != "points"}
                return out(d)
            lf = [float(x) for x in (load_factors or [1.0])]
            if not lf or not all(math.isfinite(x) for x in lf):
                raise FeaError("load_factors: finite numbers")
            ms = [float(x) for x in (mesh_sizes or [])]
            if any(not (math.isfinite(x) and x > 0) for x in ms):
                raise FeaError("mesh_sizes: positive element sizes [mm]")
            if not 1 <= int(max_points) <= 200:
                raise FeaError("max_points: 1–200 (each point is a full calculation unless the linear shortcut applies)")
            if required_safety is not None and not float(required_safety) > 0:
                raise FeaError("required_safety must be positive")
            extra = {}
            mats = {}
            names = [] if sess.frame_only else sess.part_names()
            for p, lst in (materials or {}).items():
                if p not in names:
                    raise FeaError(f"materials: part '{p}' does not exist (parts: {', '.join(names)})")
                if not isinstance(lst, list) or not lst:
                    raise FeaError(f"materials['{p}']: a non-empty list of material names")
                for m in lst:
                    if m in st.materials:
                        continue
                    from ..materials import by_name
                    try:
                        extra[m] = by_name(m)
                    except KeyError:
                        raise FeaError(f"Material '{m}' is neither in the study nor in the catalogue (fea_materials_list)") from None
                mats[p] = [str(m) for m in lst]
            spec = ParametricStudySpec(load_factors=lf, materials=mats, mesh_sizes=ms, max_points=int(max_points),
                                       required_safety=None if required_safety is None else float(required_safety))
            n = spec.total()
            stt.update(spec=spec, extra_materials=extra, result=None)
            from ..sweep import shortcut_blockers
            pure = not mats and not ms
            why = shortcut_blockers(st, spec.combinations()) if pure else []
            d = {"spec": spec.to_dict(), "points": min(n, spec.max_points), "points_full": n,
                 "linear_shortcut": pure and not why}
            if n > spec.max_points:
                d["warning"] = f"{n} combinations – only the first {spec.max_points} are evaluated"
            if why:
                d["full_solve_reason"] = "; ".join(why) + " – every point is solved"
            d["next"] = "fea_sweep_run() (background job: fea_job_status / fea_job_result) or fea_sweep_run(background=False)"
            return out(d)

    @mcp.tool(structured_output=False)
    @without_ctx
    def fea_sweep_run(background: bool = True, timeout_s: float = 7200, keep_results: bool = False, confirmed: str | None = None,
                      ctx: Context | None = None) -> str:
        """Runs the sweep from fea_sweep_define. background=True (default) returns a job_id at once - follow with fea_job_status / fea_job_result, stop with fea_job_cancel; False waits. Per point: max von Mises, max outside singularities, max displacement, mass, min_safety = Re / von Mises at the real maximum outside singularities (None = no Re; min_safety_raw incl. singularities), plus the Pareto front (mass, stress, safety). A pure load-factor sweep of a linear study is solved once and rescaled. The valid mesh of the session is reused. keep_results=False (default) deletes the work folder of every evaluated point (a failed point keeps its folder); True keeps them and saves DP_xxx/results.npz. The session study and results are not changed. confirmed: as in fea_solve_run."""
        stt = _sweep_state(sess)
        if stt["spec"] is None:
            raise FeaError("No sweep is defined – fea_sweep_define(load_factors=…, materials=…, mesh_sizes=…)")
        if not float(timeout_s) > 0:
            raise FeaError("timeout_s must be positive")
        conf = gate(sess, confirmed, ctx)
        if not background:
            with sess.use(op="parametric sweep", wait=5):
                verify(sess, conf)
                return out(sweep_core(sess, float(timeout_s), bool(keep_results)))
        from . import tools_jobs as tj
        if sess.study is None:
            raise FeaError("No study is open – fea_study_new / fea_study_open")
        running = [j for j in tj.jobs(sess).values() if j.active]
        if running:
            raise FeaError(f"Job {running[0].id} is already running – wait (fea_job_status) or cancel it (fea_job_cancel)")
        job = tj.Job(f"job-{next(tj._ids)}", "sweep", float(timeout_s), "parametric sweep")

        def body():
            try:
                with sess.use(wait=5, op=f"job {job.id} (parametric sweep)"):
                    job.status = "running"
                    job.ready.set()
                    verify(sess, conf)
                    job.result = sweep_core(sess, float(timeout_s), bool(keep_results), cancel=job.cancel,
                                            on_progress=job.progress, on_log=job.add_log)
                job.status = "done"
                job.progress(1.0, 1.0, "Done")
            except BaseException as e:  # noqa: BLE001 – the job state carries the error
                if isinstance(e, tj.BusyError):     # the session lock was not obtained = the job did not start
                    job.busy_error = e
                job.error = str(e) if isinstance(e, FeaError) else f"{type(e).__name__}: {e}"
                job.status = ("cancelled" if job.cancel.is_set() else
                              "timeout" if "time limit" in job.error else "error")
            finally:
                job.t_end = time.time()
                job.ready.set()
        js = tj.jobs(sess)
        js[job.id] = job
        for old in [k for k, j in js.items() if not j.active][:-10]:
            del js[old]
        job.thread = threading.Thread(target=body, daemon=True, name=f"fea-job {job.id}")
        job.thread.start()
        job.ready.wait(15)
        if job.status == "error" and job.result is None and job.busy_error is not None:
            del js[job.id]
            raise job.busy_error
        d = job.snapshot()
        d["next"] = f"fea_job_status(job_id='{job.id}'), then fea_job_result(job_id='{job.id}')"
        return out(d)

"""MCP: parametric study / DOE (fea_core.sweep, C4b) – sweep of load scale, materials, material properties, mesh size and
3D print settings over a copy of the study; KPI table, Pareto front, take over a design point."""
from __future__ import annotations

import time

from mcp.server.mcpserver.context import Context

from .common import out, run_with_timeout
from .session import FeaError, Session
from .tools_setup import gate, verify, without_ctx


def _round(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return float(f"{f:.5g}") if f == f and abs(f) != float("inf") else None


def table(res) -> list[dict]:
    rows = []
    for p in res.points:
        d = p.to_dict()
        rows.append({"index": p.index, "params": p.params, "ok": p.converged,
                     **{k: _round(d.get(k)) for k in ("max_stress_clean", "max_stress", "max_disp", "min_safety", "min_safety_raw",
                                                      "mass", "freq1", "print_safety")},
                     **({"error": p.error} if p.error else {}), "pareto": p.index in res.pareto_indices})
    return rows


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    @without_ctx
    def fea_param_study_run(parameters: list[dict], mode: str = "grid", max_points: int = 50, seed: int = 0,
                            timeout_s: float = 3600, confirmed: str | None = None, ctx: Context | None = None) -> str:
        """Parametric study (DOE) over a copy of the study: each design point = one analysis, KPI table + Pareto front (mass vs max stress outside singularities). parameters = [{kind, values, target, field}]: kind load_factor (all cases) | case_factor (target = load case) | material (target = part, values = material names of the study) | material_prop (target = material, field E|nu|rho|Re|Rm|alpha|k|E_z|Rm_z) | mesh_size [mm] | print (target = printed part, field walls|top_layers|bottom_layers|infill (0-1)|layer_height|line_width|pattern) | sketch_param (target = name of a sketch parameter; the geometry is rebuilt per point). values = list or text "start:stop:n". mode grid (all combinations) | list (i-th values together) | lhs (max_points Latin-hypercube points: every continuous parameter gets bounds [lo, hi] instead of values, 1-5 of them, seed fixes the points; a material gets values) - the training set for fea_surrogate_fit. A linear study with load factors only is solved once and scaled. CAD parameters of a STEP are not supported. The main study and its results are unchanged; fea_param_study_apply takes over a point (not for sketch parameters). confirmed: as in fea_solve_run."""
        from ..sweep import ParametricStudySpec, SweepError, SweepParam, parse_values, run_parametric_study
        from .tools_mesh import readiness
        if mode not in ("grid", "list", "lhs"):
            raise FeaError("mode must be grid | list | lhs")
        if not parameters:
            raise FeaError("parameters: give at least one {kind, values, …}")
        ps = []
        try:
            for d in parameters:
                if not isinstance(d, dict):
                    raise FeaError("each parameter is an object {kind, values, target, field}")
                vals = d.get("values")
                numeric = d.get("kind") not in ("material",) and not (d.get("kind") == "print" and d.get("field") == "pattern")
                if vals is None and d.get("bounds") is not None:
                    vals = []
                elif isinstance(vals, str):
                    vals = parse_values(vals, numeric)
                elif not isinstance(vals, list):
                    raise FeaError("values: a list or a text 'start:stop:n' (mode lhs: bounds [lo, hi])")
                ps.append(SweepParam.from_dict({"kind": d.get("kind", ""), "values": list(vals), "target": d.get("target"),
                                                "field": d.get("field"), "bounds": d.get("bounds")}))
        except SweepError as e:
            raise FeaError(str(e)) from None
        spec = ParametricStudySpec.from_params(ps, mode=mode, max_points=int(max_points), seed=int(seed))
        conf = gate(sess, confirmed, ctx)
        with sess.use(op="parametric study", wait=5):
            st = sess.require_study()
            verify(sess, conf)
            from ..sweep import check_spec
            errors = check_spec(st, spec)
            if errors:
                raise FeaError("Invalid parametric study:\n  " + "\n  ".join(errors))
            rd = readiness(sess)
            if not rd["ready"]:
                raise FeaError("The study is not ready for calculation:\n  " + "\n  ".join(rd["errors"]))
            wd = sess.workdir / f"param_{time.strftime('%H%M%S')}"
            from ..surrogate.stale import study_fingerprint
            fingerprint = study_fingerprint(st, [p.key for p in spec.all_params()], sess.step_path)     # of the study as the sweep starts
            logs: list[str] = []
            t0 = time.time()
            calc = type(st).from_dict(st.to_dict())
            if calc.thermal.cases or calc.modal.enabled or calc.dynamics.enabled:
                calc.load_cases = [lc for lc in calc.load_cases if lc.loads]
            try:
                mesh = sess.valid_mesh()                      # the baseline / the points with the same mesh signature do not mesh again
                res = run_with_timeout(lambda cancel: run_parametric_study(calc, spec, wd, parts=sess.parts, mesh=mesh,
                                                                           cancel=cancel, on_log=logs.append),
                                       timeout_s, "Parametric study")
            except SweepError as e:
                raise FeaError(str(e)) from None
            finally:
                for line in logs:
                    sess.add_log(line)
            res.fingerprint = fingerprint
            sess.param_study = res
            sess.surrogate = None                           # a surrogate belongs to the parametric study it was fitted to (fea_surrogate_fit again)
            return out({"points": table(res), "parameters": res.keys, "pareto": res.pareto_indices,
                        "linear_shortcut": res.linear_shortcut, "warnings": res.warnings[:10],
                        "time_s": round(time.time() - t0, 1), "workdir": str(wd),
                        "note": "The main study and its results are unchanged; fea_param_study_apply(index) takes over a point"
                                + ("; fea_surrogate_fit builds a surrogate model with error bars from these points" if mode == "lhs" else "")})

    @mcp.tool(structured_output=False)
    def fea_param_study_results() -> str:
        """Table of the last parametric study (KPI per design point, Pareto front, parameters)."""
        with sess.use():
            res = sess.param_study
            if res is None:
                raise FeaError("No parametric study has been run – fea_param_study_run")
            return out({"points": table(res), "parameters": res.keys, "pareto": res.pareto_indices,
                        "linear_shortcut": res.linear_shortcut, "warnings": res.warnings[:10]})

    @mcp.tool(structured_output=False)
    def fea_param_study_apply(index: int) -> str:
        """Takes over the parameters of design point `index` of the last parametric study into the study (results are discarded – run fea_solve_run)."""
        from ..sweep import SweepError, apply_params
        with sess.use():
            st = sess.require_study()
            res = sess.param_study
            if res is None:
                raise FeaError("No parametric study has been run – fea_param_study_run")
            dp = next((p for p in res.points if p.index == int(index)), None)
            if dp is None:
                raise FeaError(f"Design point {index} does not exist (0–{len(res.points) - 1})")
            if any(k.startswith("sketch:") for k in dp.params):
                raise FeaError("A design point with sketch parameters is not taken over here: set the sketch parameters "
                               "(fea_sketch_param) and run fea_sketch_build")
            try:
                new = apply_params(st, dp.params)
            except SweepError as e:
                raise FeaError(str(e)) from None
            sess.study = new
            sess.changed()
            sess.param_study = res
            return out({"applied": int(index), "params": dp.params, "next": "fea_solve_run"})

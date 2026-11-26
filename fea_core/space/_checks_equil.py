"""Check 7 – global force and moment equilibrium of the static states.

For every static state the balance is RECOMPUTED here from the stored reactions: the applied loads (`loads.nodal_loads`: consistent nodal forces + remote loads) plus the
reactions (`StepResult.reactions` per support, `reaction_moments` about the moment point) must vanish. The residual is compared with what the solver recorded
(`StepResult.equilibrium`) – a result whose reactions were changed or lost after the solution does not agree with its own record – and with the acceptance tolerance of the
solver (`config.EQUILIBRIUM_TOL`, the same one `solve.make_step` rejects a result with). The tolerance of the requirement set (`equilibrium_tol`) decides PASS.

DIVISOR of the relative imbalance (printed with every state, in the values of the check, the report and the help):
  * a state with a NET applied force (the resultant of the applied loads is more than `NET_LOAD_SHARE` of the sum of their nodal norms) is judged against the RESULTANT scale
    max(|ΣF applied|, |ΣR|) – the force a reader means by "1 % of the load". The sum of the nodal norms of the loads and reactions (what `solve.equilibrium` divides by) is far larger
    for a slender or a statically indeterminate model (a cantilever at L/h = 60 under 100 N: 19 503 N) and would dilute a 1 % imbalance to 5e-5;
  * the same for the moment (|Σ M applied| against the sum of the nodal moment norms, resultant scale max(|ΣM applied|, |ΣM reactions|));
  * a SELF-EQUILIBRATED part (no net applied force or moment: a pure moment, a uniform temperature change, a bolt preload) has no resultant to divide by: it is judged on the
    scale of its own internal forces, the sums of the nodal norms the solver recorded with the state (`force_scale` / `moment_scale`). A result saved by an older version has
    no such record: its balance is still computed, but the check is UNCONFIRMED ("recompute with this version"), never a verdict from a guessed scale.

Supports that share (node, DOF) (two symmetry planes: `Fixed` on x = 0 and on z = 0) make the per-support reactions count the shared nodes once for every support, so their
sum is not the total reaction. The solver records the overlap (`reaction_overlap`, `reaction_moment_overlap`: Σ per-support − total); the per-support reactions are then judged as
Σ per support − overlap, every support must have its entry (a lost or zeroed or scaled reaction does not agree with the record), and nothing is replaced by the total. A result
without the recorded overlap falls back to the total of the solver's record and is NEVER a full PASS (the per-support reactions cannot be verified).

A state whose balance cannot be recomputed by nature (deformed-position moments of a nonlinear state, cyclic or beam-frame states, a state of no load case of the study) is
judged from the residual vectors and the scales of the solver's record – a bare stored scalar decides nothing (the status cannot be PASS). A state of a study with supports whose
reactions or residual vectors are missing is a violation, also when its name is not that of a load case: the result lost what the balance is made of.

Numerical settings of the check (not limits of a standard; recorded in the values of the check and printed in the report): `CONSISTENCY_TOL`, `NET_LOAD_SHARE`.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .. import config
from ._checks_common import CheckContext, CheckResult, Judge, skipped
from ._checks_run import run_statics

CONSISTENCY_TOL = 1.0e-6      # the recomputed and the recorded residual force / moment of one result must agree to this share of the force / moment scale: they come from the same numbers
NET_LOAD_SHARE = 1.0e-6       # the applied loads have a NET force (moment) when their resultant is more than this share of the sum of their nodal norms; below it the resultant is rounding
                              # noise of a self-equilibrated state (a rounding error of the sums is ~1e-13 of the nodal norms)
DIVISOR_NAME = {"resultant": "resultant", "nodal": "nodal norms"}
FORCE_BASIS = {"resultant": "resultant max(|applied force|, |reactions|)", "nodal": "sum of the nodal force norms (solver record)"}
MOMENT_BASIS = {"resultant": "resultant max(|applied moment|, |reaction moment|)", "nodal": "sum of the nodal moment norms (solver record)"}
DIVISOR_RULE = ("relative to the resultant max(|applied|, |reactions|) when a net force (moment) is applied, to the sum of the nodal norms of the solver's record "
                "(self-equilibrated state: pure moment, temperature change, preload) otherwise")


def shared_support_selections(study: Any, mesh: Any, names: Any) -> list[str]:
    """The names of the support selections (of `names`, the keys of the stored reactions) that hold a (node, DOF) that another one of them holds as well (`solve.constraint_masks`:
    PER DOF – two symmetry planes that constrain different directions share nodes but no DOF and are not counted)."""
    from ..solve import constraint_masks
    try:
        masks = constraint_masks(study, mesh)
    except (KeyError, AttributeError, IndexError, TypeError):
        return []
    keys = [n for n in masks if n in names]
    return [a for a in keys if any(bool(np.any(masks[a] & masks[b])) for b in keys if b != a)]


def _vec3(v: Any) -> np.ndarray | None:
    try:
        a = np.asarray(v, float) if v is not None else np.zeros(0)
    except (TypeError, ValueError):
        return None
    return a if a.shape == (3,) and np.all(np.isfinite(a)) else None


def _pos(v: Any) -> float | None:
    """A positive finite number or None."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) and x > 0 else None


def _imbalance(residual: np.ndarray, scale: float | None) -> float | None:
    """|residual| / scale; a residual of exactly zero is balanced on any scale (None when there is a residual and no scale to judge it on)."""
    r = float(np.linalg.norm(residual))
    if scale is not None:
        return r / scale
    return 0.0 if r == 0.0 else None


def _scale(applied: np.ndarray, reaction: np.ndarray, applied_norm: float, nodal: float | None) -> tuple[float | None, str]:
    """(the divisor, which one) of one relative imbalance: the resultant when the applied load has a net force / moment, else the solver's nodal-norm scale (None when it has none)."""
    net = float(np.linalg.norm(applied))
    if applied_norm > 0 and net > NET_LOAD_SHARE * applied_norm:
        return max(net, float(np.linalg.norm(reaction)), 1e-30), "resultant"
    return nodal, "nodal"


def recompute_balance(study: Any, mesh: Any, lc: Any, st: Any) -> dict[str, Any] | None:
    """The balance of one state from its stored reactions; None when it cannot be recomputed here (no load case behind it, deformed-position moments, cyclic / frame states,
    reactions or the moment point missing). With `missing_reactions` / `invalid_reactions` the entries of the supports are not all there / not finite numbers: nothing is
    recomputed. The divisors are those of the module docstring (`force_scale_basis`, `moment_scale_basis` say which); a divisor is None when the state is self-equilibrated and
    its record has no scale (an older result): its imbalance is then None as well."""
    from ..loads import nodal_loads
    eq = st.equilibrium or {}
    if lc is None or eq.get("deformed") or eq.get("cyclic") or st.frame or st.moment_point is None or not st.reactions:
        return None
    required = list(dict.fromkeys(c.selection for c in getattr(study, "constraints", [])))
    moments = st.reaction_moments or {}
    missing = [n for n in required if n not in st.reactions or n not in moments]
    invalid = [n for n in st.reactions if _vec3(st.reactions[n]) is None] + [n for n in moments if _vec3(moments[n]) is None]
    if missing or invalid:
        return {"missing_reactions": missing, "invalid_reactions": list(dict.fromkeys(invalid))}
    F, Fr, Mr = nodal_loads(study, mesh, lc)
    c0 = np.asarray(st.moment_point, float)
    x = mesh.nodes - c0
    rs = [np.asarray(v, float) for v in st.reactions.values()]
    ms = [np.asarray(v, float) for v in moments.values()]
    shared = shared_support_selections(study, mesh, set(st.reactions))
    ov_f, ov_m = _vec3(eq.get("reaction_overlap")), _vec3(eq.get("reaction_moment_overlap"))
    R = np.sum(rs, axis=0)
    MR = np.sum(ms, axis=0) if ms else np.zeros(3)
    fallback = False
    if ov_f is not None and ov_m is not None:                # the solver recorded what the shared (node, DOF) count more than once
        R, MR = R - ov_f, MR - ov_m
        basis = "per support" if not shared else "per support minus the overlap recorded by the solver"
    elif shared:                                             # an older result: only the total of the record is left (the per-support reactions cannot be verified)
        tot, totm = _vec3(eq.get("reaction_total")), _vec3(eq.get("reaction_moment_total"))
        if tot is None or totm is None:
            return None
        R, MR, fallback = tot, totm, True
        basis = "total of the solver record (the overlap of the supports is not recorded)"
    else:
        basis = "per support"
    f_app = F.sum(axis=0) + Fr
    m_app = np.cross(x, F).sum(axis=0) + Mr - np.cross(c0, Fr)
    fres, mres = f_app + R, m_app + MR
    f_norm = float(np.linalg.norm(F, axis=1).sum() + np.linalg.norm(Fr))
    m_norm = float(np.linalg.norm(np.cross(x, F), axis=1).sum() + np.linalg.norm(Mr - np.cross(c0, Fr)))
    rec_f, rec_m = _pos(eq.get("force_scale")), _pos(eq.get("moment_scale"))
    f_scale, f_kind = _scale(f_app, R, f_norm, rec_f)
    m_scale, m_kind = _scale(m_app, MR, m_norm, rec_m)
    f_imb = _imbalance(fres, f_scale)
    m_imb = _imbalance(mres, m_scale)
    known = [v for v in (f_imb, m_imb) if v is not None]
    return {"force_residual_N": fres.tolist(), "moment_residual_Nmm": mres.tolist(), "applied_force_N": f_app.tolist(), "reaction_force_N": R.tolist(),
            "force_scale_N": f_scale, "moment_scale_Nmm": m_scale,
            "force_scale_basis": FORCE_BASIS[f_kind] if f_scale is not None else "none: self-equilibrated state and the record has no scale",
            "moment_scale_basis": MOMENT_BASIS[m_kind] if m_scale is not None else "none: self-equilibrated state and the record has no scale",
            "consistency_force_scale_N": max(f_scale or 0.0, rec_f or 0.0, 1e-30), "consistency_moment_scale_Nmm": max(m_scale or 0.0, rec_m or 0.0, 1e-30),
            "force_divisor": DIVISOR_NAME[f_kind] if f_imb is not None else "none", "moment_divisor": DIVISOR_NAME[m_kind] if m_imb is not None else "none",
            "scale_missing": [n for n, v in (("force", f_imb), ("moment", m_imb)) if v is None],
            "reaction_basis": basis, "reaction_fallback": fallback, "shared_support_selections": shared,
            "force_imbalance": f_imb, "moment_imbalance": m_imb, "imbalance": max(known) if known else None}


def record_balance(eq: dict[str, Any]) -> dict[str, Any] | None:
    """The balance of a state that cannot be recomputed, from the residual vectors AND the scales of the solver's record; None when the record does not have them. The divisor is
    the resultant (the recorded applied force `applied_exact` / the reaction total) when it is a net force of the state, else the solver's nodal-norm scale."""
    rf, rm = _vec3(eq.get("force_residual")), _vec3(eq.get("moment_residual"))
    sf, sm = _pos(eq.get("force_scale")), _pos(eq.get("moment_scale"))
    if rf is None or rm is None or sf is None or sm is None:
        return None
    applied, tot, totm = _vec3(eq.get("applied_exact")), _vec3(eq.get("reaction_total")), _vec3(eq.get("reaction_moment_total"))
    f_div, m_div = sf, sm
    f_res = max(float(np.linalg.norm(applied)) if applied is not None else 0.0, float(np.linalg.norm(tot)) if tot is not None else 0.0)
    if f_res > NET_LOAD_SHARE * sf:                              # a net force: its resultant is the divisor
        f_div = f_res
    if totm is not None and float(np.linalg.norm(totm)) > NET_LOAD_SHARE * sm:
        m_div = float(np.linalg.norm(totm))
    f_imb, m_imb = float(np.linalg.norm(rf) / f_div), float(np.linalg.norm(rm) / m_div)
    return {"force_imbalance": f_imb, "moment_imbalance": m_imb, "imbalance": max(f_imb, m_imb), "force_scale_N": f_div, "moment_scale_Nmm": m_div,
            "force_scale_basis": FORCE_BASIS["resultant" if f_div != sf else "nodal"], "moment_scale_basis": MOMENT_BASIS["resultant" if m_div != sm else "nodal"],
            "force_divisor": DIVISOR_NAME["resultant" if f_div != sf else "nodal"], "moment_divisor": DIVISOR_NAME["resultant" if m_div != sm else "nodal"]}


def _consistent(rec: dict[str, Any], eq: dict[str, Any]) -> bool:
    """The recomputed balance agrees with the record of the solver: the residual force and moment themselves are the same numbers (they are compared against the larger of the
    scales – the two RELATIVE imbalances need not be). A record without the residual vectors is not consistent with anything: it cannot be compared."""
    rf, rm = _vec3(eq.get("force_residual")), _vec3(eq.get("moment_residual"))
    if rf is None or rm is None:
        return False
    df = float(np.linalg.norm(np.asarray(rec["force_residual_N"]) - rf))
    dm = float(np.linalg.norm(np.asarray(rec["moment_residual_Nmm"]) - rm))
    return df <= CONSISTENCY_TOL * rec.get("consistency_force_scale_N", rec["force_scale_N"]) and dm <= CONSISTENCY_TOL * rec.get("consistency_moment_scale_Nmm", rec["moment_scale_Nmm"])


def _sources(ctx: CheckContext) -> tuple[list[tuple[str, Any, dict[str, Any]]], set[str]]:
    """([(label, Results, {step name: LoadCase})] of the static states that are examined, the names of the load cases of the study that they cover)."""
    out: list[tuple[str, Any, dict[str, Any]]] = []
    covered: set[str] = set()
    if ctx.results is not None and getattr(ctx.results, "steps", None):
        out.append(("results", ctx.results, {lc.name: lc for lc in ctx.study.load_cases}))
        covered = {st.name for st in ctx.results.steps}
    if ctx.shared.get("gravity_results") is not None:
        from ._checks_mass import unit_gravity_cases
        out.append(("unit gravity", ctx.shared["gravity_results"], {lc.name: lc for lc in unit_gravity_cases()}))
    return out, covered


def _lost_supports(study: Any, st: Any) -> bool:
    """The state has no usable stored reactions although the study has supports."""
    return bool(getattr(study, "constraints", None)) and (not st.reactions or st.moment_point is None)


def run_equilibrium_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    notes: list[str] = []
    sources, covered = _sources(ctx)
    todo = [lc for lc in ctx.study.load_cases if lc.name not in covered]          # the load cases of the study are always examined, not only the unit-gravity states
    if todo and ctx.study.constraints:
        res = run_statics(ctx, todo, "check_equilibrium")
        sources.append(("load cases of the study", res, {lc.name: lc for lc in todo}))
    elif todo and sources:
        judge.cannot_decide(f"the load case(s) {', '.join(lc.name for lc in todo)} of the study were not examined: the study has no support to run them with")
    if not sources:
        return skipped(7, "no static state to examine: give the results of the study, or a study with load cases and a support (the unit-gravity states of check 1 need a support)",
                       time.time() - t0)
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for label, res, lcs in sources:
        for st in res.steps:
            lc = lcs.get(st.name)
            eq = st.equilibrium or {}
            rec = recompute_balance(ctx.study, ctx.mesh, lc, st)
            stored = float(eq.get("imbalance", float("nan")))
            row: dict[str, Any] = {"source": label, "state": st.name, "recorded_imbalance": stored,
                                   "recorded_force_residual_N": list(eq.get("force_residual", [])),
                                   "recorded_moment_residual_Nmm": list(eq.get("moment_residual", []))}
            by_nature = bool(eq.get("deformed") or eq.get("cyclic") or st.frame)         # the balance of these cannot be recomputed from the reactions (not "of no load case")
            if rec is not None and ("missing_reactions" in rec or "invalid_reactions" in rec):
                row["recomputed"] = False
                if rec["missing_reactions"]:
                    judge.violate(f"state '{st.name}': the stored reaction (or reaction moment) of the support(s) {', '.join(rec['missing_reactions'])} is missing – the result lost "
                                  "what the balance is made of")
                if rec["invalid_reactions"]:
                    judge.violate(f"state '{st.name}': the stored reaction of {', '.join(rec['invalid_reactions'])} is not a finite 3-vector")
                row.update(rec)
                rows.append(row)
                continue
            if rec is not None:
                row.update(rec)
                row["consistent_with_record"] = _consistent(rec, eq)
                row["scale_of"] = f"force: {rec['force_scale_basis']}; moment: {rec['moment_scale_basis']}"
                val = rec["imbalance"]
                if rec["shared_support_selections"]:
                    if rec["reaction_fallback"]:
                        notes.append(f"state '{st.name}': the supports {', '.join(rec['shared_support_selections'])} hold the same node and direction, and the result has no record of the "
                                     "overlap – the total reaction of the solver's record is used instead of the per-support reactions, which therefore are NOT verified (recompute the "
                                     "state with this version)")
                        judge.cannot_decide(f"state '{st.name}': the per-support reactions of the supports {', '.join(rec['shared_support_selections'])} (shared nodes and directions) "
                                            "cannot be verified – the result was saved by an older version without the recorded overlap; recompute it")
                    else:
                        notes.append(f"state '{st.name}': the supports {', '.join(rec['shared_support_selections'])} hold the same node and direction – their reactions are judged as "
                                     "the sum per support minus the overlap recorded by the solver")
                if rec["scale_missing"]:
                    notes.append(f"state '{st.name}': no net applied {' / '.join(rec['scale_missing'])} and the result has no scale of its internal forces (saved by an older version)")
                    judge.cannot_decide(f"state '{st.name}': a self-equilibrated state (no net applied {' / '.join(rec['scale_missing'])}) of a result saved by an older version has no "
                                        "scale to judge its imbalance on – recompute the study with this version")
                if not row["consistent_with_record"]:
                    judge.violate(f"state '{st.name}': the balance recomputed from the stored reactions ({'-' if val is None else format(val, '.3g')}) differs from the one recorded by "
                                  f"the solver ({stored:.3g}) – or the record has no residual vectors: the reactions of the result were changed or lost after the solution")
                if val is None:
                    rows.append(row)
                    continue
            elif _lost_supports(ctx.study, st) and not by_nature:
                row["recomputed"] = False
                judge.violate(f"state '{st.name}': the stored reactions of the state are missing although the study has supports – the balance cannot be recomputed, the result "
                              "lost them after the solution")
                rows.append(row)
                continue
            else:
                row["recomputed"] = False
                recd = record_balance(eq)
                if recd is None:
                    judge.cannot_decide(f"state '{st.name}': its balance cannot be recomputed here (no load case / deformed / cyclic / beam state / no stored reactions) and the "
                                        "solver's record has no residual vectors and scales – a stored scalar alone decides nothing")
                    notes.append(f"state '{st.name}': the balance could not be recomputed and the solver's record is incomplete")
                    if not np.isfinite(stored):
                        judge.violate(f"state '{st.name}': the equilibrium record is missing or not a number")
                    rows.append(row)
                    continue
                row.update(recd)
                row["scale_of"] = f"force: {recd['force_scale_basis']}; moment: {recd['moment_scale_basis']}"
                val = recd["imbalance"]
                notes.append(f"state '{st.name}': the balance could not be recomputed (no load case / deformed / cyclic / beam state) – judged from the residual vectors and scales "
                             "of the solver's record")
            if not np.isfinite(val):
                judge.violate(f"state '{st.name}': the equilibrium record is missing or not a number")
                continue
            row["imbalance"] = val
            worst = max(worst, val)
            if val > config.EQUILIBRIUM_TOL:
                judge.violate(f"state '{st.name}': force / moment imbalance {val:.3g} exceeds the solver's acceptance tolerance {config.EQUILIBRIUM_TOL:g} – the result is not in equilibrium")
            rows.append(row)
    values: dict[str, Any] = {"states": rows, "worst_imbalance": worst, "solver_acceptance_tolerance": config.EQUILIBRIUM_TOL, "states_checked": len(rows),
                              "consistency_tolerance": CONSISTENCY_TOL, "net_load_share": NET_LOAD_SHARE, "divisor": DIVISOR_RULE}
    judge.compare_max(worst, "equilibrium_tol", "largest relative force / moment imbalance")
    notes.append("the imbalance is " + DIVISOR_RULE)
    summary = f"{len(rows)} static state(s), worst force / moment imbalance {worst:.2e}"
    return judge.finish(7, summary, values, notes, time.time() - t0)

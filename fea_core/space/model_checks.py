"""FEM model quality checks of the structural verification of a spacecraft (F14 stage S3, the checks of ECSS-E-ST-32-11C 6.4.1 and common FEM practice): seven checks that a finite element model
must pass before its results are used for margins of safety.

  1  mass and centre of mass      FE total mass / centre against the budget and the CAD volumes; the mass as the solver sees it (three unit-gravity statics)
  2  free-free modes              every support removed: exactly six rigid-body modes, zero strain energy of them, which parts move independently if there are more
  3  unit enforced displacement   three unit translations and three small rotations enforced on the launch interface: no strain energy, no net interface reaction
  4  isothermal expansion         a uniform temperature change of a one-expansion-coefficient copy of the model, supported statically determinately: no stress;
                                  the support groups of the study must not push on each other under it
  5  effective modal mass         cumulative effective mass in x, y, z up to the cutoff frequency against the total mass
  6  mesh convergence             the first natural frequencies and the governing stress on successively finer meshes (Richardson, GCI)
  7  equilibrium                  the force and moment balance of the static states recomputed from the stored reactions

Entry point: `run_model_checks(study, mesh, results, limits, options, workdir=…)` → a list of `CheckResult` (status PASS / WARN / FAIL / UNCONFIRMED / SKIPPED, the numbers, the
limits that were consulted, actionable findings). No limit of a standard is built in (F14 plan, section 1): every tolerance is an entry of `ModelCheckLimits`, empty in the
template, filled from the confirmed requirement set; a check without its limit computes and reports its numbers and is UNCONFIRMED – never PASS. What can FAIL without a limit
are physical invariants: a free body has exactly six rigid-body modes, a static state must be in equilibrium to the acceptance tolerance of the solver itself.
Sources: ECSS-E-ST-32-11C 6.4.1b names "zero rigid body energy" and the free-free analysis (ideally six zero natural frequencies) as standard FEM checks (check 2 follows it; check 5 follows 5.7.3 and
3.2.13 for the effective modal mass); the unit enforced displacement, the isothermal expansion and the mesh convergence are common finite element practice that the structural FEM standards of the ECSS
require in general terms and are not defined in ECSS-E-ST-32-11C (those documents were not available for the review).
No solver is new: statics `solve.run_study`, modes `eigen.run_eigen` (direct modes or CalculiX), the prescribed-displacement runs of checks 3 and 4 the own assembly + rslab of
`solve_large`, K·u from `recovery.recover`, the mesh refinement of `convergence`.

Wiring for the GUI panel / the MCP group (the S1 stage owns them): `run_for_session(sess, …)` is the body of an MCP tool `fea_space_model_checks`, `table_rows(results)` the rows of
the traffic-light table (id, title, status, summary, colour key), `to_dict(results)` / `format_text(results)` the JSON / text forms for the report.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from ._checks_common import (CHECK_IDS, CHECK_TITLES, FAIL, LIMIT_DESCRIPTIONS, MAX_CONVERGENCE_RATIO, MIN_CONVERGENCE_RATIO, OPTION_CHECKS, PASS, SKIPPED, STATUSES, UNCONFIRMED, WARN,
                             CheckContext, CheckResult, LimitEntry, ModelCheckLimits, ModelCheckOptions, describe_option, nondefault_options, option_problems, overall_verdict,
                             skipped)
from ._checks_run import CheckCancelled

__all__ = ["CHECK_IDS", "CHECK_TITLES", "CheckResult", "LimitEntry", "ModelCheckLimits", "ModelCheckOptions", "LIMIT_DESCRIPTIONS", "STATUSES", "PASS", "WARN", "FAIL", "UNCONFIRMED", "SKIPPED",
           "run_model_checks", "run_for_session", "options_from_dict", "overall_verdict", "to_dict", "format_text", "table_rows", "template_limits",
           "limits_of_study", "limits_overview", "CheckRun", "make_run", "run_is_current", "study_digest", "option_problems", "nondefault_options", "describe_option",
           "confirmation_info", "MIN_CONVERGENCE_RATIO", "MAX_CONVERGENCE_RATIO", "OPTION_CHECKS"]

STATUS_COLOUR = {PASS: "green", WARN: "amber", FAIL: "red", UNCONFIRMED: "grey", SKIPPED: "grey"}


def template_limits() -> dict[str, Any]:
    """The empty template of the limits (name → entry with value None, confirmed False): what a requirement set has to fill and confirm."""
    return ModelCheckLimits.template().to_dict()


def options_from_dict(d: dict[str, Any] | None) -> ModelCheckOptions:
    """Options from a plain dict (an MCP call, a saved setup); an unknown key is an error, and so is a value outside the bounds of the numerical settings (`option_problems`)."""
    names = {f.name for f in fields(ModelCheckOptions)}
    out = ModelCheckOptions()
    for k, v in (d or {}).items():
        if k not in names:
            raise ValueError(f"Unknown model-check option '{k}' (known: {', '.join(sorted(names))})")
        if k in ("cog_budget_mm", "density_range_t_mm3", "cte_range_per_K", "unit_states") and v is not None:
            try:
                v = tuple(v)
            except TypeError:
                raise ValueError(f"option {k} must be a list") from None
        setattr(out, k, v)
    bad = option_problems(out)
    if bad:
        raise ValueError("; ".join(bad))
    return out


def _runners() -> dict[str, Callable[[CheckContext], CheckResult]]:
    from ._checks_conv import run_convergence_check
    from ._checks_equil import run_equilibrium_check
    from ._checks_mass import run_mass_check
    from ._checks_modal import run_effective_mass_check, run_free_free_check
    from ._checks_static import run_isothermal_check, run_unit_displacement_check
    return {"mass": run_mass_check, "free_free": run_free_free_check, "unit_displacement": run_unit_displacement_check, "isothermal": run_isothermal_check,
            "effective_mass": run_effective_mass_check, "mesh_convergence": run_convergence_check, "equilibrium": run_equilibrium_check}


def _make_mesh(study: Any, workdir: Path, study_dir: Any, parts: Any, log: Callable[[str], None], cancel: Any) -> Any:
    from ..solve import prepare_mesh
    st = copy.deepcopy(study)
    st.modal.enabled = True                           # `prepare_mesh` wants something to compute; the mesh itself does not depend on it
    mesh, _ = prepare_mesh(st, workdir / "mesh", study_dir, log, parts=parts, cancel=cancel)
    return mesh


def run_model_checks(study: Any, mesh: Any = None, results: Any = None, limits: ModelCheckLimits | dict[str, Any] | None = None, options: ModelCheckOptions | dict[str, Any] | None = None,
                     *, workdir: str | Path, checks: list[str] | tuple[str, ...] | None = None, parts: Any = None, study_dir: str | Path | None = None,
                     log: Callable[[str], None] | None = None, cancel: Any = None, singular_mask_fn: Callable[..., Any] | None = None,
                     on_check: Callable[[CheckResult], None] | None = None) -> list[CheckResult]:
    """Runs the model checks (`checks` = ids from CHECK_IDS, None = all seven) → one `CheckResult` per requested check, in the order of the standard.

    study     the study as the analysis uses it (it is never changed: every check works on copies)
    mesh      the finished mesh of the study (None = meshed here)
    results   the results of a run of the study (their static states are examined by the equilibrium check, their modal set by the effective-mass check)
    limits    the confirmed tolerances (`ModelCheckLimits`, normally `ModelCheckLimits.from_requirement_set`); a dict is only a set of VALUES (`from_entries`: never
              confirmed, so the checks stay UNCONFIRMED); None / the empty template = no verdicts
    options   budgets and numerical settings (`ModelCheckOptions` or a dict)
    workdir   the folder of the solver runs (required: the checks write there)
    parts     the geometry parts (CAD volumes, re-meshing for the convergence check); study_dir = the folder of the study file (for its STEP)
    singular_mask_fn  (mesh, step, study) → bool mask of the nodes to leave out of the governing stress of check 6 (the hotspot classifier of the margin stage)
    on_check  called with every finished CheckResult (progress of the GUI)
    """
    log = log or (lambda s: None)
    lim = limits if isinstance(limits, ModelCheckLimits) else ModelCheckLimits.from_entries(limits)
    opt = options if isinstance(options, ModelCheckOptions) else options_from_dict(options)
    bad = option_problems(opt)
    if bad:
        raise ValueError("; ".join(bad))
    ids = list(CHECK_IDS if checks is None else checks)
    unknown = [c for c in ids if c not in CHECK_IDS]
    if unknown:
        raise ValueError(f"Unknown check(s) {', '.join(unknown)} (known: {', '.join(CHECK_IDS)})")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    if mesh is None:
        log("Model checks: meshing the study")
        mesh = _make_mesh(study, wd, study_dir, parts, log, cancel)
    ctx = CheckContext(study, mesh, lim, opt, wd, log, cancel, parts, study_dir, results, {}, on_check)
    if singular_mask_fn is not None:
        ctx.shared["singular_mask_fn"] = singular_mask_fn
    runners = _runners()
    out: list[CheckResult] = []
    for cid in [c for c in CHECK_IDS if c in ids]:
        if cancel is not None and cancel.is_set():
            raise CheckCancelled("Analysis canceled by user")
        n = CHECK_IDS.index(cid) + 1
        log(f"Model check {n}/7: {CHECK_TITLES[cid]}")
        t0 = time.time()
        try:
            res = runners[cid](ctx)
        except CheckCancelled:
            raise
        except Exception as e:  # noqa: BLE001 – one failing check must not take the others down; it is reported as not run (and the verdict then cannot be PASS)
            if cancel is not None and cancel.is_set() or "canceled by user" in str(e):
                raise CheckCancelled("Analysis canceled by user") from None
            text = (str(e).splitlines() or [type(e).__name__])[0][:300]
            res = skipped(n, f"the check could not be completed: {type(e).__name__}: {text}", time.time() - t0)
        res.options, res.finished = nondefault_options(opt, n), time.strftime("%Y-%m-%d %H:%M")           # what this check ran with and when: a kept run may merge checks of several runs
        out.append(res)
        if on_check is not None:
            on_check(res)
    return out


# ---------------------------------------------------------------------------------------------------------------- output forms
def confirmation_info(study: Any) -> dict[str, Any]:
    """Who confirmed the requirement set the limits come from and through which interface (S1 rule: a confirmation made through MCP is an agent's – the user must check it);
    scope-aware like the margins (`SpaceSettings.requirement_confirmed`)."""
    sp = getattr(study, "space", None)
    rs = getattr(sp, "requirement_set", None)
    ok = bool(sp is not None and rs is not None and sp.requirement_confirmed())
    return {"requirement_set": "" if rs is None else rs.name, "confirmed": ok, "confirmed_by": rs.confirmed_by if ok and rs is not None else "",
            "confirmed_via": rs.confirmed_via if ok and rs is not None else "", "date": rs.date if ok and rs is not None else "",
            "to_be_checked_by_the_user": bool(ok and rs is not None and rs.confirmed_via == "mcp")}


def confirmation_line(info: dict[str, Any]) -> str:
    if not info.get("confirmed"):
        return "Requirement set: not confirmed - every check is UNCONFIRMED"
    via = info.get("confirmed_via") or "unknown"
    return (f"Requirement set '{info.get('requirement_set', '')}' confirmed by {info.get('confirmed_by', '')} on {info.get('date', '')}, confirmed via {via}"
            + (" - made through MCP by an agent: the user must check the limits and confirm the set in the program" if via == "mcp" else ""))


def to_dict(results: list[CheckResult], study: Any = None) -> dict[str, Any]:
    """JSON form: the verdict, the counts per status and every check (with the settings of that check that differ from the defaults); with `study` also who confirmed the
    requirement set and through which interface."""
    counts = {s: sum(1 for r in results if r.status == s) for s in STATUSES}
    d: dict[str, Any] = {"verdict": overall_verdict(results), "counts": counts, "checks": [r.to_dict() for r in results],
                         "settings_that_differ_from_the_defaults": {r.id: r.options for r in results if r.options},
                         "note": "No limit of a standard is built into the program: a check without a confirmed limit is UNCONFIRMED, never PASS."}
    if study is not None:
        d["requirement_set"] = confirmation_info(study)
    return d


def table_rows(results: list[CheckResult]) -> list[dict[str, Any]]:
    """Rows of the traffic-light table of the panel: number, id, title, status, colour key, one-line summary, the first finding."""
    return [{"number": r.number, "id": r.id, "title": r.title, "status": r.status, "colour": STATUS_COLOUR.get(r.status, "grey"), "summary": r.summary,
             "finding": r.findings[0] if r.findings else "", "seconds": round(r.elapsed_s, 1)} for r in results]


def options_text(opts: dict[str, Any]) -> str:
    """The settings of one check that differ from the defaults as one line: name = value; ..."""
    return "; ".join(f"{k} = {describe_option(k, v)}" for k, v in opts.items())


def format_text(results: list[CheckResult], study: Any = None) -> str:
    """Plain-text report of the checks (for the log, the MCP answer and the report chapter); a setting that differs from the default is printed next to the check it changed."""
    lines = [f"Model quality checks – verdict {overall_verdict(results)}"]
    if study is not None:
        lines.append("  " + confirmation_line(confirmation_info(study)))
    for r in results:
        lines.append(f"  {r.number}. {r.title}: {r.status} – {r.summary}")
        if r.options:
            lines.append(f"       settings that differ from the defaults: {options_text(r.options)}")
        for f in r.findings[:6]:
            lines.append(f"       ! {f}")
        for n in r.notes[:4]:
            lines.append(f"       . {n}")
    return chr(10).join(lines)


def run_for_session(sess: Any, *, options: dict[str, Any] | None = None, checks: list[str] | None = None,
                    workdir: str | Path | None = None, log: Callable[[str], None] | None = None, cancel: Any = None) -> dict[str, Any]:
    """The body of the MCP tool `fea_space_model_checks` (and of the panel's re-run): `sess` is the MCP `Session` (study, mesh, results, geometry parts) or anything with the
    same members. The mesh and the results of the session are used when they exist; the answer is `to_dict(results)` plus the text form.
    The limits are NOT an argument: they come only from the confirmed requirement set of the study (`RequirementSet.limits`); without one every check is UNCONFIRMED."""
    st = sess.require_study()
    limits = limits_of_study(st)
    parts = None
    try:
        parts = sess.parts
    except Exception:  # noqa: BLE001 – without geometry the CAD volumes and the re-meshing are skipped, the other checks run
        parts = None
    wd = Path(workdir) if workdir else sess.new_workdir("model_checks")
    study_dir = Path(sess.step_path).parent if getattr(sess, "step_path", None) else None
    res = run_model_checks(st, sess.valid_mesh(), getattr(sess, "results", None), limits, options, workdir=wd, checks=checks, parts=parts, study_dir=study_dir,
                           log=log or (getattr(sess, "add_log", None) or (lambda s: None)), cancel=cancel)
    d = to_dict(res, st)
    d["text"] = format_text(res, st)
    d["limits"] = limits_overview(st)
    try:                                                 # kept for the Word report (chapter "Model quality checks"); a session that has no such member just does not keep it
        sess.model_checks = make_run(st, res, options, getattr(sess, "model_checks", None))
    except AttributeError:
        pass
    return d


# ---------------------------------------------------------------------------------------------------------------- limits of a study, stored runs (GUI, MCP, report)
def limits_of_study(study: Any) -> ModelCheckLimits:
    """The ONLY source of confirmed tolerances of the model checks: the limits of the confirmed requirement set of the study (empty template without one)."""
    sp = getattr(study, "space", None)
    return ModelCheckLimits.from_requirement_set(getattr(sp, "requirement_set", None), None if sp is None else sp.requirement_confirmed())


def _limit_state(entry: Any, rs_confirmed: bool) -> str:
    if entry is None or entry.value is None or not math.isfinite(entry.value):
        return "missing"
    if not (entry.document.strip() and entry.revision.strip() and entry.clause.strip() and entry.clause.strip() != "<clause>"):
        return "incomplete"
    return "usable" if rs_confirmed else "not confirmed"


def limits_overview(study: Any) -> list[dict[str, Any]]:
    """One row per tolerance of the checks: name, what it limits, the value ENTERED in the requirement set (None = missing), its provenance and the state –
    usable (confirmed set, value, document / revision / clause) | not confirmed (the set is not confirmed or was edited after the confirmation) | incomplete
    (no document / revision / clause) | missing. Only "usable" limits decide a check."""
    sp = getattr(study, "space", None)
    rs = getattr(sp, "requirement_set", None)
    confirmed = bool(sp is not None and rs is not None and sp.requirement_confirmed())
    by_key = {x.key: x for x in (rs.limits if rs is not None else [])}
    rows = []
    for name, text in LIMIT_DESCRIPTIONS.items():
        x = by_key.get(name)
        rows.append({"name": name, "description": text, "value": None if x is None else x.value, "unit": "" if x is None else x.unit,
                     "document": "" if x is None else x.document, "revision": "" if x is None else x.revision, "clause": "" if x is None else x.clause,
                     "state": _limit_state(x, confirmed)})
    return rows


def study_digest(study: Any) -> str:
    """SHA-256 of what the checks depend on: the study WITHOUT its space settings, plus the limits that are usable (so that editing a justification of a singular
    place does not make the checks look old, but a new or withdrawn limit does)."""
    d = study.to_dict()
    d.pop("space", None)
    d["_limits"] = {k: v for k, v in limits_of_study(study).to_dict().items() if v["confirmed"]}
    info = confirmation_info(study)                                  # a set confirmed again (or through another interface) is another basis of the verdicts
    d["_confirmation"] = [info["confirmed_by"], info["confirmed_via"]] if info["confirmed"] else None
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class CheckRun:
    """A finished run of the checks as it is kept for the report: the results (each carries the settings it ran with and when it finished), the state of the study they
    belong to and the confirmation of the requirement set the limits came from."""
    results: list[CheckResult]
    study_name: str
    digest: str
    limits: list[dict[str, Any]]
    options: dict[str, Any] = field(default_factory=dict)           # the options of the LAST run that fed this record (plain data); the settings of each check are in its result
    created: str = ""
    requirement_set: str = ""
    confirmed_by: str = ""
    date: str = ""
    confirmed_via: str = ""                                         # "gui" | "mcp" | "": through which interface the requirement set was confirmed (as the S1 stage shows it)


def make_run(study: Any, results: list[CheckResult], options: ModelCheckOptions | dict[str, Any] | None = None, previous: CheckRun | None = None) -> CheckRun:
    """Keeps a run. The settings of every check are kept WITH its result (`CheckResult.options` / `finished`): a result that did not come through `run_model_checks` gets those of
    `options` now. With a `previous` run of the SAME state of the study the checks of this run replace their namesakes and the others are kept as they were (a re-run of
    one check does not drop the other six, and does not relabel them with its budgets)."""
    opt = options if isinstance(options, ModelCheckOptions) else options_from_dict(options)
    now = time.strftime("%Y-%m-%d %H:%M")
    for r in results:
        if not r.finished:
            r.options, r.finished = nondefault_options(opt, r.number), now
    info = confirmation_info(study)
    plain = json.loads(json.dumps(asdict(opt), default=str))
    run = CheckRun(list(results), str(getattr(study, "name", "")), study_digest(study), limits_overview(study), plain, now, info["requirement_set"], info["confirmed_by"],
                   info["date"], info["confirmed_via"])
    if previous is not None and previous.digest == run.digest and previous.study_name == run.study_name:
        merged = {r.id: r for r in previous.results}
        merged.update({r.id: r for r in results})
        run.results = [merged[c] for c in CHECK_IDS if c in merged]
    return run


def run_is_current(run: CheckRun | None, study: Any) -> bool:
    """True when `run` was made for the present state of `study` (the study content and the usable limits are unchanged)."""
    return run is not None and run.study_name == str(getattr(study, "name", "")) and run.digest == study_digest(study)

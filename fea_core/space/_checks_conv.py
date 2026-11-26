"""Check 6 – mesh convergence summary: the first natural frequencies and the governing stress on successively finer meshes (the refinement and the Richardson / GCI
analysis of `fea_core.convergence`).

One `run_study` per level computes the natural frequencies AND the governing load case together; the governing case is the one with the highest von Mises stress on the
given mesh. The stress is followed at a FIXED point – the place of the maximum on the given mesh – because the global maximum of a singular spot keeps growing with
refinement: a fixed point converges where the stress is regular, and a global maximum that grows with order < 0.5 is reported as a singularity suspect (a warning, not a
convergence failure) – the proper exclusion of the singular regions is the hotspot classifier of the margin stage, which can be attached as `singular_mask_fn`
(mesh, step, study → bool mask of the nodes to leave out). Without it the governing stress is the global maximum.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..convergence import richardson
from ._checks_common import MAX_CONVERGENCE_RATIO, MIN_CONVERGENCE_RATIO, CheckContext, CheckResult, Judge, ModelCheckOptions, skipped
from ._checks_run import cancelled, statics_copy


def _refine(study: Any, k: int, ratio: float) -> tuple[Any, float]:
    """A copy of the study with the element sizes divided by ratio^k (the rule of `convergence.run_convergence`) → (study, global element size)."""
    st = copy.deepcopy(study)
    f = float(ratio) ** k
    st.mesh.size = float(study.mesh.size) / f
    if st.mesh.min_size:
        st.mesh.min_size = st.mesh.min_size / f
    st.mesh.local = [dict(loc, size=float(loc["size"]) / f) for loc in st.mesh.local]
    st.mesh.part_sizes = {k_: float(v) / f for k_, v in st.mesh.part_sizes.items()}
    return st, st.mesh.size


def _level_study(study: Any, case_names: list[str], n_freq: int) -> Any:
    st = statics_copy(study, [lc for lc in study.load_cases if lc.name in case_names])
    need = {t.case for lc in st.load_cases for t in st.thermal_loads(lc) if t.case}
    st.thermal.cases = [c for c in study.thermal.cases if c.name in need]
    st.modal.enabled = n_freq > 0
    if n_freq > 0:
        st.modal.modes = int(n_freq)
        st.modal.f_min, st.modal.f_max = 0.0, None
        st.modal.verify_ccx = False
    return st


def mesh_key(mesh: Any) -> str:
    """Digest of everything of a mesh a `run_study` result depends on – `state_cache.mesh_digest` (nodes, elements, parts, named surfaces, the edge and point selections, the tie
    table of the separately meshed parts, the per-node / per-element data, the node aliases): the one helper the state cache and the convergence reuse share."""
    from ..state_cache import mesh_digest
    return mesh_digest(mesh)


def reusable_level0(ctx: CheckContext, n_freq: int) -> Any:
    """The results of the caller as level 0 of the convergence check (None = solve it). Only a result that PROVES it is the result of this very study state on this very mesh is used:
    its study digest (`solve.study_state_digest`, kept by `run_study`) equals the digest of the study of the checks, its mesh is the mesh of the checks (the same object or the same
    content, `mesh_key`), it holds exactly the load cases of the study, none of them a partial nonlinear result ("Stop and keep results"), and – when frequencies are followed – a
    natural-frequency set of the plain window [0, ∞) with at least `n_freq` modes. Any doubt = None."""
    res, st0 = ctx.results, ctx.study
    if res is None or getattr(res, "twod", None) is not None or getattr(res, "shell", None) is not None:
        return None
    info = getattr(res, "info", None) or {}
    digest = info.get("study_digest")
    if not digest:
        return None
    from ..solve import study_state_digest
    if digest != study_state_digest(st0):
        return None
    if any(getattr(s, "partial", False) for s in res.steps) or (info.get("nonlinear") or {}).get("partial"):
        return None                                                    # a stopped / not converged nonlinear run is not the result of the study state
    if res.mesh is not ctx.mesh and mesh_key(res.mesh) != mesh_key(ctx.mesh):
        return None
    if [s.name for s in res.steps] != [lc.name for lc in st0.load_cases]:
        return None
    if n_freq > 0:
        modal = [m for m in res.modes if m.kind == "modal"]
        md = st0.modal
        if not modal or not md.enabled or float(md.f_min or 0.0) != 0.0 or md.f_max is not None or len(modal[0].values) < n_freq:
            return None
    return res


def _rel_changes(vals: list[float]) -> float | None:
    if len(vals) < 2 or not vals[-1]:
        return None
    return float(abs(vals[-1] - vals[-2]) / abs(vals[-1]))


def run_convergence_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    opt = ctx.options
    st0 = ctx.study
    notes: list[str] = []
    if not opt.include_convergence:
        return skipped(6, "not requested (it needs several complete mesh levels)", time.time() - t0)
    if st0.twod.enabled or st0.shell.enabled or st0.frame.active:
        return skipped(6, "the convergence of 2D / shell / beam models is not examined by this check", time.time() - t0)
    if ctx.parts is None and not Path(st0.geometry_path(ctx.study_dir)).exists():
        return skipped(6, "the geometry (parts or the STEP file) is needed to re-mesh the model on finer levels", time.time() - t0)
    levels = max(2, int(opt.convergence_levels))
    ratio = float(opt.convergence_ratio)
    if not (MIN_CONVERGENCE_RATIO <= ratio <= MAX_CONVERGENCE_RATIO):          # numerical setting: a ratio of one (or a tiny refinement) has no order of convergence to judge
        return skipped(6, f"the element-size ratio {ratio:g} of the levels is outside {MIN_CONVERGENCE_RATIO:g}–{MAX_CONVERGENCE_RATIO:g}: the Richardson extrapolation of nearly equal "
                          "meshes is meaningless, so no convergence is judged", time.time() - t0)
    n_freq = int(opt.convergence_frequencies)
    default = ModelCheckOptions()                                      # a setting that makes the check weaker than the default is allowed (it runs faster) but never gives a full PASS
    if levels < default.convergence_levels:
        judge.cannot_decide(f"only {levels} mesh levels were run (the default is {default.convergence_levels}): the singularity test and the order of convergence need three levels, "
                            "so the check is not complete")
    if ratio < default.convergence_ratio:
        judge.cannot_decide(f"the element-size ratio {ratio:g} of the levels is smaller than the default {default.convergence_ratio:g}: the levels differ less, so a change between them "
                            "is smaller than it would be with the default refinement and the convergence is judged more leniently")
    if n_freq < default.convergence_frequencies:
        judge.cannot_decide(f"only {n_freq} natural frequencies are followed (the default is {default.convergence_frequencies}): a higher mode that does not converge is not seen")
    if not st0.load_cases and not (st0.modal.enabled or n_freq > 0):
        return skipped(6, "the study has neither a load case nor natural frequencies to follow", time.time() - t0)
    n_nodes = len(ctx.mesh.nodes)
    est = 3 * n_nodes * ratio ** (3 * (levels - 1))
    if est > float(opt.convergence_max_dof):
        return skipped(6, f"the finest level would have about {est / 1e6:.1f} million DOF (limit {opt.convergence_max_dof / 1e6:.1f} million) – use fewer levels or a smaller ratio", time.time() - t0)
    from ..solve import run_study
    hook = ctx.shared.get("singular_mask_fn")

    def governing(res: Any, case: str | None) -> tuple[float, np.ndarray | None, int | None]:
        """(the governing stress, the node of it, the index of the state) of the load case (None = the highest of all)."""
        best: tuple[float, np.ndarray | None, int | None] = (-1.0, None, None)
        for i, stp in enumerate(res.steps):
            if case is not None and stp.name != case:
                continue
            mises = np.asarray(stp.mises, float).copy()
            if hook is not None:
                mises[np.asarray(hook(res.mesh, stp, st0), bool)] = 0.0
            j = int(np.argmax(mises))
            if mises[j] > best[0]:
                best = (float(mises[j]), res.mesh.nodes[j].copy(), i)
        return best

    rows: list[dict[str, Any]] = []
    freqs: list[list[float]] = []
    gmax: list[float] = []
    probe: list[float] = []
    case_name: str | None = None
    point: np.ndarray | None = None
    dofs: list[int] = []
    level0_reused = False
    wd = Path(ctx.workdir) / "check_convergence"
    for k in range(levels):
        if cancelled(ctx):
            raise RuntimeError("Analysis canceled by user")
        st_k, h = _refine(st0, k, ratio)
        if k == 0:
            names = [lc.name for lc in st0.load_cases] if st0.load_cases else []
        else:
            names = [case_name] if case_name else []
        st_k = _level_study(st_k, names, n_freq)
        res = reusable_level0(ctx, n_freq) if k == 0 else None
        if res is not None:
            ctx.log(f"Mesh convergence check: level 1/{levels}, element size {h:.3g} mm – the results of the study are used (same study state and mesh)")
            level0_reused = True
        else:
            ctx.log(f"Mesh convergence check: level {k + 1}/{levels}, element size {h:.3g} mm")
            res = run_study(st_k, wd / f"level_{k + 1}", study_dir=ctx.study_dir, on_log=ctx.log, cancel=ctx.cancel, mesh=ctx.mesh if k == 0 else None, parts=None if k == 0 else ctx.parts)
        dofs.append(3 * len(res.mesh.nodes))
        row: dict[str, Any] = {"level": k + 1, "element_size_mm": h, "nodes": int(len(res.mesh.nodes)), "elements": int(len(res.mesh.tets))}
        modal = [m for m in res.modes if m.kind == "modal"]
        if modal and n_freq > 0:
            f = np.asarray(modal[0].values, float)[:n_freq]            # the leading n_freq modes: the rigid-body cut below must not depend on how many more the run stored
            f = f[f > 1e-6 * max(float(f.max()), 1e-300)] if len(f) else f
            freqs.append(f.tolist())
            row["frequencies_Hz"] = f.tolist()
        if res.steps:
            if k == 0:
                _, _, gi = governing(res, None)
                case_name = res.steps[gi].name if gi is not None else None
                want = opt.convergence_case
                if want is not None:                                       # the load case named by the caller instead of the highest stress
                    names_all = [stp.name for stp in res.steps]
                    case_name = names_all[int(want)] if isinstance(want, int) and 0 <= int(want) < len(names_all) else str(want)
                    if case_name not in names_all:
                        return skipped(6, f"the load case '{case_name}' of the convergence check does not exist (cases: {', '.join(names_all)})", time.time() - t0)
            val, node, gi = governing(res, case_name)
            row["case"], row["max_mises_MPa"] = case_name, val
            gmax.append(val)
            if k == 0:
                point = node
            if point is not None and gi is not None:
                from ..paths import line_values
                pv = float(line_values(res, point, point, "mises", gi, n=2)["values"][0])
                probe.append(pv)
                row["mises_at_fixed_point_MPa"] = pv
        rows.append(row)
    values: dict[str, Any] = {"levels": rows, "refinement_ratio": ratio, "governing_case": case_name,
                              "fixed_point_mm": None if point is None else point.tolist(), "dof": dofs}
    # --- frequencies
    n_use = min((len(f) for f in freqs), default=0)
    if len(freqs) == levels and n_use:
        per = []
        for j in range(n_use):
            series = [freqs[k][j] for k in range(levels)]
            rich = richardson(series, ratio)
            per.append({"mode": j + 1, "frequencies_Hz": series, "last_change": _rel_changes(series), **{k_: v for k_, v in rich.items() if k_ != "monotonic" or v is not None}})
        values["frequency_convergence"] = per
        worst = max((p["last_change"] or 0.0) for p in per)
        values["frequency_worst_last_change"] = worst
        judge.compare_max(float(worst), "freq_convergence_tol", "largest relative change of a natural frequency with the last refinement")
    else:
        notes.append("no natural frequencies were followed")
    # --- stress
    if len(probe) == levels:
        rich_p = richardson(probe, ratio)
        rich_g = richardson(gmax, ratio)
        values["stress_convergence"] = {"global_max_MPa": gmax, "fixed_point_MPa": probe, "fixed_point_last_change": _rel_changes(probe), "fixed_point_richardson": rich_p,
                                        "global_max_last_change": _rel_changes(gmax), "global_max_richardson": rich_g}
        growing = len(gmax) >= 3 and all(b > a for a, b in zip(gmax, gmax[1:]))
        singular = bool(growing and (rich_g["order"] is None or rich_g["order"] < 0.5))
        values["singularity_suspect"] = singular
        if singular:
            judge.warn("the global maximum stress keeps growing with the refinement (order < 0.5): a singularity suspect (sharp corner, edge support, point load) – its value has "
                       "no meaning; assess the stress away from it and justify it in the singularity table")
        change = _rel_changes(probe)
        if change is not None:
            judge.compare_max(float(change), "stress_convergence_tol", "relative change of the stress at the fixed point of the maximum with the last refinement")
        gchange = _rel_changes(gmax)
        if hook is not None and gchange is not None:           # the global maximum outside the singular nodes is a stress that must converge as well
            judge.compare_max(float(gchange), "stress_convergence_tol", "relative change of the global maximum stress outside singularities with the last refinement")
        elif gchange is not None:
            notes.append("the global maximum stress includes singular regions (no singularity filter was given): its change is reported, not judged")
    elif st0.load_cases:
        notes.append("no stress was followed")
    if level0_reused:
        notes.append("level 1 is the result of the study itself (the same study state and mesh were solved before)")
    notes.append(f"{levels} levels, element size ratio {ratio:g}; governing stress = " + ("the global maximum without the nodes of the singularity filter" if hook is not None else "the global maximum"
                 " (singular regions are not excluded) and the stress at a fixed point"))
    freq_txt = f"first {n_use} frequencies change by {100 * values.get('frequency_worst_last_change', 0.0):.2g} % at the last refinement" if "frequency_convergence" in values else ""
    st_txt = ""
    if "stress_convergence" in values and values["stress_convergence"]["fixed_point_last_change"] is not None:
        st_txt = f"stress at the fixed point changes by {100 * values['stress_convergence']['fixed_point_last_change']:.2g} %"
    return judge.finish(6, "; ".join(t for t in (freq_txt, st_txt) if t) or "no quantity followed", values, notes, time.time() - t0)

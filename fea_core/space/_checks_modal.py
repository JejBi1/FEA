"""Checks 2 and 5 – the free-free modes (rigid-body behaviour of the model) and the effective modal mass.

Free-free: every support is removed from a copy of the study and the first modes are computed by the program's own modal path (`eigen.run_eigen`: the direct modes
or ccx). A connected free body has exactly six rigid-body modes at 0 Hz; more means a free part or a mechanism, and the shapes of those modes say WHICH parts move
independently of the rest. The zero-energy property of the rigid modes is verified on its own: `recovery.recover_batch` gives K·φ, so the strain energy of every
rigid-body mode relative to the first elastic mode is a number that does not depend on the eigen-solver that produced the mode.
Effective mass: the cumulative effective modal mass in the three translations up to the cutoff frequency against the total mass of the model.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ._checks_common import CheckContext, CheckResult, Judge, skipped
from ._checks_run import modal_copy, run_modes, statics_copy, sub_workdir

RIGID_ZERO = 1.0e-6          # a mode below this fraction of the highest computed frequency is a rigid-body mode (the rule of eigen.compare_modes)
GROUP_TOL = 1.0e-3           # parts whose rigid-body motions differ by less than this share belong to one rigid group (numerical grouping of the shapes, not a limit)
MAX_ENERGY_MODES = 14
NOISE_SHARE = 1.0e-3         # a highest computed frequency below this share of the physical frequency scale means that no elastic mode has been reached yet
MAX_FREE_MODES = 60          # the free-free run grows to this many modes while every computed mode is still a rigid-body mode


def rigid_basis(X: np.ndarray, x0: np.ndarray, lc: float) -> np.ndarray:
    """(3n, 6): the six rigid-body motions u = t + ω × (x − x0) of the nodes X, columns (t_x t_y t_z, ω_x·lc, ω_y·lc, ω_z·lc), node-major."""
    r = (np.asarray(X, float) - x0) / lc
    n = len(r)
    B = np.zeros((n, 3, 6))
    B[:, 0, 0] = B[:, 1, 1] = B[:, 2, 2] = 1.0
    B[:, 0, 4], B[:, 0, 5] = r[:, 2], -r[:, 1]                      # u = t − r × ω  (columns 3..5 = ω)
    B[:, 1, 3], B[:, 1, 5] = -r[:, 2], r[:, 0]
    B[:, 2, 3], B[:, 2, 4] = r[:, 1], -r[:, 0]
    return B.reshape(3 * n, 6)


def frequency_scale(study: Any, mesh: Any, props: Any) -> float:
    """The physical frequency scale c / (2π·L) [Hz] of the model: c = sqrt(E / ρ) with E of the stiffest material and ρ the mean density of the whole model (added
    masses included), L the diagonal of the bounding box. A computed mode far below it is numerical noise (a rigid-body mode), not an elastic one."""
    from ._checks_run import e_max
    L = float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0))) or 1.0
    rho = props.mass_t / props.volume_mm3 if props.volume_mm3 > 0 else 7.85e-9
    return float(np.sqrt(e_max(study, mesh) / rho) / L / (2.0 * np.pi))


def _used_nodes(mesh: Any) -> np.ndarray:
    return np.unique(mesh.tets)


def rigid_groups(mesh: Any, shapes: np.ndarray) -> dict[str, Any]:
    """From the shapes of the rigid-body modes (m, N, 3): which parts move rigidly together, which move independently, and which part deforms in a zero-energy mode.
    → {"groups": [[part, …], …] (the first = the reference group: the part with most elements), "deforming_parts": {part: share of the shape that is not a rigid motion}}."""
    used = _used_nodes(mesh)
    X = mesh.nodes[used]
    x0 = X.mean(axis=0)
    lc = float(np.sqrt(((X - x0) ** 2).sum(axis=1).mean())) or 1.0
    m = len(shapes)
    n_el = np.bincount(mesh.tet_part, minlength=len(mesh.part_names))
    A: dict[int, np.ndarray] = {}
    resid: dict[int, float] = {}
    for p in range(len(mesh.part_names)):
        nodes_p = np.unique(mesh.tets[mesh.tet_part == p])
        if not len(nodes_p):
            continue
        B = rigid_basis(mesh.nodes[nodes_p], x0, lc)
        U = np.stack([np.asarray(shapes[j][nodes_p], float).reshape(-1) for j in range(m)], axis=1)       # (3n, m)
        a, *_ = np.linalg.lstsq(B, U, rcond=None)
        A[p] = a
        nu = float(np.linalg.norm(U))
        resid[p] = float(np.linalg.norm(U - B @ a) / nu) if nu > 0 else 0.0
    if not A:
        return {"groups": [], "deforming_parts": {}}
    ref = max(A, key=lambda p: int(n_el[p]))
    groups: list[list[int]] = [[ref]]
    for p in sorted(A, key=lambda q: -int(n_el[q])):
        if p == ref:
            continue
        for g in groups:
            d = np.linalg.norm(A[p] - A[g[0]])
            if d <= GROUP_TOL * max(np.linalg.norm(A[p]), np.linalg.norm(A[g[0]]), 1e-300):
                g.append(p)
                break
        else:
            groups.append([p])
    names = mesh.part_names
    return {"groups": [[str(names[p]) for p in g] for g in groups],
            "deforming_parts": {str(names[p]): r for p, r in resid.items() if r > GROUP_TOL}, "part_residuals": {str(names[p]): r for p, r in resid.items()}}


def _recover_energy(study: Any, mesh: Any, phi: np.ndarray) -> np.ndarray:
    """2·strain energy φᵀKφ of every shape φ (B, N, 3) from the internal nodal forces K·φ of `recovery.recover_batch` (the element routine of the stress recovery:
    independent of the eigen-solver)."""
    from .. import ortho
    from ..recovery import recover, recover_batch
    from ..solve_large import element_stiffness_all
    field = ortho.ElementField(study, mesh) if ortho.has_ortho(study, mesh) else None
    C = element_stiffness_all(study, mesh)
    phi = np.asarray(phi, np.float64)
    if field is not None:
        out = [recover(study, mesh, phi[b], None, C, field)[1] for b in range(len(phi))]
        RF = np.stack(out)
    else:
        _, RF = recover_batch(study, mesh, phi, C, None, True)
    assert RF is not None
    return np.einsum("bnk,bnk->b", phi, RF)


def run_free_free_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    st0 = ctx.study
    if getattr(st0, "cyclic", None) is not None and st0.cyclic.enabled:
        return skipped(2, "the study uses cyclic symmetry: its free-free behaviour is not tested by this check", time.time() - t0)
    if st0.frame.active:
        return skipped(2, "the study has a beam system: its supports are not removed by this check", time.time() - t0)
    n_modes = max(7, int(ctx.options.free_free_modes))
    from ._checks_mass import mass_properties
    props = ctx.shared.get("mass") or mass_properties(st0, ctx.mesh)
    f_phys = frequency_scale(st0, ctx.mesh, props)
    while True:
        st = modal_copy(st0, n_modes, None, free=True)
        ms = run_modes(ctx, st, "check_free_free")
        f = np.asarray(ms.values, float)
        shapes = np.asarray(ms.shapes)
        top = float(f.max()) if len(f) else 0.0
        rigid = np.ones(len(f), bool) if top < NOISE_SHARE * f_phys else f <= RIGID_ZERO * max(top, 1e-300)
        n_rigid = int(rigid.sum())
        elastic = np.nonzero(~rigid)[0]
        if len(elastic) >= 2 or n_modes >= MAX_FREE_MODES:       # every computed mode is noise-level: the count of the rigid ones is not known yet (CalculiX names them only by the jump above them)
            break
        n_modes = min(MAX_FREE_MODES, 2 * n_modes)
    values: dict[str, Any] = {"engine": ms.engine, "modes_computed": int(len(f)), "rigid_body_modes": n_rigid, "expected_rigid_body_modes": 6,
                              "frequencies_Hz": f[:12].tolist(), "first_elastic_Hz": float(f[elastic[0]]) if len(elastic) else None}
    notes: list[str] = [f"free-free run: every support removed ({len(st0.constraints)} constraint(s) of the study), {len(f)} modes by the {ms.engine} engine"]
    if ctx.shared.get("rigid_body_residual") is not None:
        values["assembly_rigid_body_residual"] = ctx.shared["rigid_body_residual"]
    # --- the invariant: a connected free body has six rigid-body modes
    if n_rigid != 6:
        judge.violate(f"{n_rigid} rigid-body modes found, a connected free body has exactly 6")
    if not len(elastic):
        judge.violate("no elastic mode among the computed ones – increase the number of modes of the check")
    groups: dict[str, Any] = {}
    if n_rigid:
        rig_idx = np.nonzero(rigid)[0]
        groups = rigid_groups(ctx.mesh, shapes[rig_idx])
        values["rigid_groups"] = groups["groups"]
        values["part_rigid_fit_residual"] = groups.get("part_residuals", {})
        if n_rigid > 6:
            extra = n_rigid - 6
            if len(groups["groups"]) > 1:
                parts_txt = " | ".join(", ".join(g) for g in groups["groups"])
                judge.violate(f"{extra} extra rigid-body mode(s): these groups of parts move independently of each other: {parts_txt} – the connection between them is missing or a mechanism")
            else:
                judge.violate(f"{extra} extra rigid-body mode(s) inside one group of parts: a part deforms with zero energy (an internal mechanism) – check the part(s) "
                              f"{', '.join(groups['deforming_parts']) or 'with constraints between nodes'}")
        if groups["deforming_parts"]:
            judge.violate(f"part(s) {', '.join(groups['deforming_parts'])} deform in a zero-energy mode (they are not rigid in the rigid-body modes): a mechanism inside the part")
        # the zero energy of the rigid modes, from K·φ of the stress recovery
        sel = list(rig_idx[:MAX_ENERGY_MODES])
        ref_j = int(elastic[0]) if len(elastic) else None
        if ref_j is not None:
            e = _recover_energy(st, ctx.mesh, shapes[sel + [ref_j]])
            ratio = np.abs(e[:-1]) / max(abs(float(e[-1])), 1e-300)
            values["rigid_energy_ratio_max"] = float(ratio.max())
            values["rigid_energy_ratio"] = ratio.tolist()
            judge.compare_max(float(ratio.max()), "rigid_energy_ratio_max", "strain energy of a rigid-body mode relative to the first elastic mode")
        fit = max(groups.get("part_residuals", {"": 0.0}).values())
        values["rigid_fit_residual_max"] = float(fit)
        judge.compare_max(float(fit), "rigid_fit_tol", "share of a rigid-body mode that is not a rigid-body motion of its part")
    # --- weak connections the mesh itself shows
    try:
        from ..contacts.bonded import edge_only
        thin = edge_only(ctx.mesh)
    except Exception:  # noqa: BLE001 – a diagnostic hint only
        thin = []
    if thin:
        values["edge_only_connections"] = [list(p) for p in thin]
        judge.warn("parts connected only along an edge or at a point: " + ", ".join(f"{a} – {b}" for a, b in thin[:5]))
    summary = f"{n_rigid} rigid-body modes (expected 6), first elastic mode {values['first_elastic_Hz']:.4g} Hz" if values["first_elastic_Hz"] else f"{n_rigid} rigid-body modes"
    return judge.finish(2, summary, values, notes, time.time() - t0)


def _basis_from_run(ctx: CheckContext, st: Any, modes: int, cutoff: float | None) -> Any:
    """The modal basis of the dynamics (`dynamics.modal.run_basis`: direct modes or CalculiX, with its own checks – the normalisation φᵀMφ and Γ = φᵀMr by own integration;
    for a model with a rigid body CalculiX gives wrong participation factors and the basis replaces them by its own) for the check."""
    from ..dynamics.modal import run_basis
    return run_basis(st, ctx.mesh, sub_workdir(ctx, "check_effective_mass"), ctx.log, cancel=ctx.cancel, modes=int(modes), f_max=cutoff, job="check_modal_basis",
                     with_stress=False, with_reactions=False)                                   # only the frequencies, the effective masses and the checks of the basis are read: no modal stresses / reactions


def run_effective_mass_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    opt = ctx.options
    st0 = ctx.study
    notes: list[str] = []
    cutoff = ctx.limits.value("eff_mass_cutoff_hz")
    complete: bool | None = None
    engine = ""
    checks: dict[str, Any] = {}
    f: np.ndarray | None = None
    em: np.ndarray | None = None
    free: np.ndarray | None = None
    from ._checks_mass import mass_properties
    props = ctx.shared.get("mass") or mass_properties(st0, ctx.mesh)
    M = float(props.mass_t)
    if ctx.results is not None:
        modal = [m for m in getattr(ctx.results, "modes", []) if m.kind == "modal"]
        if modal and modal[0].eff_mass and modal[0].total_mass:
            ms = modal[0]
            f, em, free, engine = np.asarray(ms.values, float), np.asarray(ms.eff_mass, float)[:, :3], np.asarray(ms.total_mass, float)[:3], ms.engine
            notes.append(f"the natural frequencies of the given results are used ({len(f)} modes by the {engine} engine)")
    if f is None:
        if st0.constraints == [] and not st0.frame.active:
            return skipped(5, "the study has no support: effective modal masses need a constrained model (the free-free modes are check 2)", time.time() - t0)
        modes = int(opt.modes or (st0.dynamics.modes if st0.dynamics.enabled else st0.modal.modes if st0.modal.enabled else 30))
        while True:
            basis = _basis_from_run(ctx, statics_copy(st0, []), modes, cutoff)
            complete = cutoff is not None and basis.K < modes             # the window [0, cutoff] holds fewer modes than were asked for: it is exhausted
            if cutoff is None or complete or modes >= int(opt.max_modes):
                break
            modes = min(int(opt.max_modes), 2 * modes)
        f, em, free, engine = np.asarray(basis.freq, float), np.asarray(basis.eff_mass, float)[:, :3], np.asarray(basis.total_mass, float)[:3], basis.engine
        checks = {k: v for k, v in (basis.checks or {}).items() if k in ("mass_norm_error", "gamma_error", "gamma_source", "gamma_error_ccx")}
        M = float(basis.model_mass) or M
        notes.append(f"modal basis of the dynamics: {len(f)} modes" + (f" in [0, {cutoff:g}] Hz" if cutoff else "") + f" by the {engine} engine")
        if checks.get("mass_norm_error", 0.0) > 1e-3 or checks.get("gamma_error", 0.0) > 1e-3:          # the limits of `dynamics.modal.basis_warnings`
            judge.warn(f"the modal basis does not match the own mass integration (normalisation {checks.get('mass_norm_error', 0.0):.2g}, participation factors "
                       f"{checks.get('gamma_error', 0.0):.2g}) – verify the effective masses (a bolt replaced by a beam carries mass in CalculiX that the own integration does not include)")
        if checks.get("gamma_source") == "own":
            notes.append("CalculiX gives wrong participation factors for a model with a rigid coupling: the participation factors of the own mass integration are used")
    assert f is not None and em is not None and free is not None
    judge.need("eff_mass_cutoff_hz")
    sel = np.ones(len(f), bool) if not cutoff else f <= float(cutoff)
    cum = em[sel].sum(axis=0)
    frac = cum / M
    frac_free = cum / np.where(free > 0, free, 1.0)
    f_last = float(f[sel].max()) if sel.any() else 0.0
    values: dict[str, Any] = {
        "engine": engine, "modes_used": int(sel.sum()), "highest_mode_Hz": f_last, "cutoff_Hz": cutoff, "total_mass_kg": M * 1000.0, "basis_checks": checks,
        "cumulative_fraction_xyz": frac.tolist(), "cumulative_fraction_of_free_mass_xyz": frac_free.tolist(), "free_mass_kg_xyz": (free * 1000.0).tolist(),
        "missing_mass_kg_xyz": ((M - cum) * 1000.0).tolist(), "eff_mass_fraction_per_mode": (em / M)[:10].tolist(), "frequencies_Hz": f[:10].tolist()}
    if (frac > 1.0 + 1.0e-3).any():                                    # the effective masses of all modes cannot add up to more than the mass of the model
        judge.violate(f"the effective masses add up to {np.round(100 * frac, 1).tolist()} % of the mass (x, y, z) – more than the mass of the model: the engine's participation "
                      "factors are inconsistent (CalculiX with a rigid coupling) – compute the modes with the direct engine")
    reaches = cutoff is None or bool(complete) or float(f.max()) >= float(cutoff) or not len(f)           # a mode at or above the cutoff exists: the window is covered
    values["basis_reaches_cutoff"] = bool(reaches)
    lim = judge.limits.value("eff_mass_min_fraction")
    ok = judge.compare_min(float(frac.min()), "eff_mass_min_fraction", "the smallest cumulative effective mass fraction of the three translations")
    if ok is False:
        if not reaches:
            judge.violate(f"the modal basis stops at {f_last:.1f} Hz, below the cutoff {cutoff:g} Hz – compute more modes (the basis is truncated)")
        else:
            axes = [a for a, v in zip("XYZ", frac) if lim is not None and v < lim]
            judge.violate(f"the effective mass is not complete in {', '.join(axes)} up to {f_last:.1f} Hz – local modes or mass attached through rigid couplings; the missing mass is "
                          f"{np.round((M - cum) * 1000.0, 4).tolist()} kg (x, y, z)")
    elif ok is None and not reaches:
        judge.warn(f"the modal basis stops at {f_last:.1f} Hz, below the cutoff {cutoff:g} Hz")
    if (free < 0.999 * M).any():
        notes.append("the effective mass of the free DOFs is below the total mass: the mass at the constrained nodes follows the base and takes no part in the modes")
    summary = f"cumulative effective mass x/y/z = {', '.join(f'{100 * v:.1f} %' for v in frac)} of {M * 1000:.4g} kg up to {f_last:.1f} Hz"
    return judge.finish(5, summary, values, notes, time.time() - t0)

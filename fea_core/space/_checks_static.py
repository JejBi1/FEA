"""Checks 3 and 4 – the unit enforced displacement (ground check) and the isothermal thermal expansion – on the direct linear path.

Both ask the same question of the model: does it follow a motion that must cost no energy? A rigid-body motion enforced on the launch interface (three unit translations
and three small rotations) – the structure must move as a rigid body: strain energy, stresses and the net interface reaction vanish. A uniform temperature change of a
single-material model supported statically determinately – the free thermal expansion u = α·ΔT·(x − x0) must be the solution: no stress. What breaks them: a node grounded
by a stray constraint or a spring, a constraint equation that does not admit a rigid motion, a hidden closed loop of constraints, a part held by an over-stiff link.
The prescribed displacements are per NODE (a rotation varies across the interface face), which the selection-based constraints of the study cannot express – so the job is
assembled and solved exactly like `solve_large._run_ngs_job` (own assembly + rslab, `write_model_files`, the constraint plan, `recovery.recover` for K·u and the stresses)
with the prescribed values written by this module. Nothing is solved by new code. A study the direct path cannot take (contact, beam structures, …) is reported as SKIPPED.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._checks_common import DEFAULT_CTE_RANGE, UNIT_STATE_NAMES, CheckContext, CheckResult, Judge, jsonable, range_widened, skipped
from ._checks_run import cancelled, e_max, part_of_node, sub_workdir


class PrescribedUnsupported(RuntimeError):
    """The direct path cannot take this study / mesh (the reason is the message): the check is skipped."""


class PrescribedFailed(RuntimeError):
    """The prescribed-displacement analysis failed because of the MODEL (a singular system: a part that is not connected, a missing constraint): the check fails."""


@dataclass
class PrescribedState:
    name: str
    nodes: np.ndarray                                  # (m,) the nodes whose three displacements are prescribed
    values: np.ndarray                                 # (m, 3)
    dT: float = 0.0                                    # uniform temperature change of every node [K] (0 = no thermal load)
    remote: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)       # rigid remote supports: selection -> (displacement, rotation vector) of their point


@dataclass
class StaticOutcome:
    name: str
    U: np.ndarray
    S: np.ndarray
    RF: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


def _clean(job: Path) -> None:
    import shutil
    for f in list(job.glob("*.npy")) + list(job.glob("*.npz")) + list(job.glob("*.json")):
        try:
            f.unlink()
        except OSError:
            pass
    shutil.rmtree(job / "direct", ignore_errors=True)


def group_states(fixed: list[np.ndarray], own: bool) -> list[list[int]]:
    """The indices of the states that share one matrix: the same constrained DOFs (only the prescribed VALUES and the loads differ: one assembly and factorization, one right-hand
    side each). `own` = the own assembly takes one column of g per state; the debugging fallback FEA_ASSEMBLY=ngsolve reads g.npy as ONE vector, so there every state is alone."""
    groups: dict[Any, list[int]] = {}
    for i, fx in enumerate(fixed):
        groups.setdefault(np.packbits(fx).tobytes() if own else i, []).append(i)
    return list(groups.values())


def column_backward_error(info: dict[str, Any], col: int) -> float:
    """The backward error of the solution of right-hand side `col` (`info["backward_error"]` has one entry per column; a missing one = the largest)."""
    be = info.get("backward_error", [0.0]) or [0.0]
    return float(be[col] if col < len(be) else max(be))


def solve_prescribed(ctx: CheckContext, study: Any, states: list[PrescribedState], name: str, alpha_bolt: float | None = None) -> list[StaticOutcome]:
    """One linear static solution per state: the constraints of `study` (Fixed / Displacement / supports of the plan) plus the prescribed node displacements of the state, a
    uniform temperature change of the state, no other load. The own assembly + rslab; one assembly and factorization per group of states that constrain the same DOFs (the unit states of check 3: one factorization,
    one right-hand side each). alpha_bolt = the expansion coefficient the bolts get in the thermal elongation (None = that of the bolt model)."""
    from .. import assembly, direct, linear_constraints, ortho as ortho_mod, solve_large as SL
    from ..ccx import parallel
    from ..recovery import recover
    from ..solvers import direct_missing
    mesh = ctx.mesh
    why = SL.unsupported(study, "direct")
    if why:
        raise PrescribedUnsupported("the direct linear path cannot take this study: " + ", ".join(why))
    miss = direct_missing()
    if miss:
        raise PrescribedUnsupported("the direct solver is not installed – missing " + ", ".join(miss))
    refused = linear_constraints.check(study, mesh)
    if refused:
        raise PrescribedUnsupported("the constraints are not taken by the direct path: " + "; ".join(refused))
    job = sub_workdir(ctx, name) / "job"
    job.mkdir(parents=True, exist_ok=True)
    _clean(job)
    keep = SL.keep_solver_files()
    n = len(mesh.nodes)
    th = int(parallel.cap_threads(ctx.options.threads or study.solver.nproc) or parallel.budget_threads() or 0) or None
    out: list[StaticOutcome] = []
    try:
        try:
            mf = SL.write_model_files(study, mesh, job, "direct")
        except SL.LargeSolverError as e:
            raise PrescribedUnsupported(str(e)) from None
        plan: Any = mf.plan                               # `ModelFiles.plan` / `.pr` are typed `object` in solve_large: the real types are linear_constraints.Plan and an ndarray
        pr: Any = mf.pr
        nx = mf.nx
        base_fixed, base_g = SL.constraint_arrays(study, mesh, nx)
        dep = np.asarray(plan.red.dep if plan is not None else np.zeros(0, np.int64), np.int64)
        field_ = ortho_mod.ElementField(study, mesh) if mf.ortho_on else None
        t_ref = float(study.thermal.reference_temperature)
        from ..study import LoadCase
        prep: list[tuple[PrescribedState, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
        red_changed = False
        for stt in states:
            if cancelled(ctx):
                raise RuntimeError("Analysis canceled by user")
            fixed, g = base_fixed.copy(), base_g.copy()
            for d in range(3):
                dofs = 3 * np.asarray(stt.nodes, np.int64) + d
                if len(dep) and np.isin(dofs, dep).any():
                    raise PrescribedUnsupported("a prescribed node is a dependent node of a constraint equation (a tie / remote / cylindrical face)")
                fixed[dofs] = True
                g[dofs] = stt.values[:, d]
            if stt.remote:                                                  # a rigid remote support: the displacement and rotation of its reference point (extra nodes)
                if plan is None:
                    raise PrescribedUnsupported("the constraint plan has no rigid remote support")
                fx = plan.red.fixed_extra
                for sel, (u_pt, rot_pt) in stt.remote.items():
                    pair = plan.support_pairs.get(sel)
                    if pair is None:
                        raise PrescribedUnsupported(f"'{sel}' is not a rigid remote support of the constraint plan")
                    d0, d1 = plan.pair_dofs(pair)
                    base = 3 * n
                    for dd, val in ((d0, u_pt), (d1, rot_pt)):             # the values of the constrained components only (the assembly takes them from the rows of g)
                        g[dd:dd + 3] = np.where(fx[dd - base:dd - base + 3], val, 0.0)
                        if plan.red.g_extra[dd - base:dd - base + 3].any():        # a value the study itself prescribes there: the state's value replaces it (as the unit state always did)
                            plan.red.g_extra[dd - base:dd - base + 3] = 0.0
                            red_changed = True
            T = None if stt.dT == 0.0 else np.full(n, t_ref + stt.dT)
            fe, _ = SL.load_vector(study, mesh, LoadCase(stt.name, []), plan, nx, None)           # the loads of the extra nodes (none here)
            if T is not None and plan is not None and nx:                      # the thermal elongation of the bolts acts on the extra DOFs (the rows after those of the mesh)
                fe = fe.copy()
                for bp in plan.cut_bolts():
                    a = bp.chain.bolt.alpha if alpha_bolt is None else alpha_bolt
                    fe[3 * n:] += plan.elongation_load(bp, a * stt.dT * bp.chain.bolt.length)
            prep.append((stt, fixed, g, fe, T))
        if red_changed and plan is not None:
            plan.red.save(job)
        # The states that constrain the SAME DOFs share one matrix: one assembly and one factorization, one right-hand side (and one column of g) per state. Only the prescribed
        # VALUES and the loads differ between the six unit states of check 3 (the eliminated rows are the same).
        results: dict[int, StaticOutcome] = {}
        for idx in group_states([p[1] for p in prep], assembly.assembler() == "own"):
            if cancelled(ctx):
                raise RuntimeError("Analysis canceled by user")
            fixed = prep[idx[0]][1]
            g_all = np.column_stack([prep[i][2] for i in idx])
            np.save(job / "F.npy", np.column_stack([prep[i][3] for i in idx]))
            np.save(job / "fixed.npy", fixed)
            np.save(job / "g.npy", g_all[:, 0] if len(idx) == 1 else g_all)
            if any(prep[i][4] is not None for i in idx):
                np.save(job / "dT.npy", np.column_stack([np.full(n, prep[i][0].dT) for i in idx]))
            else:
                (job / "dT.npy").unlink(missing_ok=True)
            try:
                with direct.exclusive(ctx.cancel, ctx.log):
                    ainfo = SL._assemble(job, "direct", ctx.log, ctx.cancel, th)
                    X, info = direct.run_prepared(job / "direct", threads=th, cancel=ctx.cancel)
            except assembly.ElementError as e:
                raise PrescribedFailed(SL.mesh_quality_message(e, mesh)) from None
            except direct.DirectMemoryError as e:
                raise PrescribedUnsupported(str(e)) from None
            except (direct.DirectError, SL.LargeSolverError) as e:
                msg = str(e)
                if "canceled by user" in msg:
                    raise RuntimeError(msg) from None
                if any(w in msg.lower() for w in ("singular", "positive definite", "pivot", "free body")):
                    raise PrescribedFailed("the system is singular even with the interface prescribed: a part is not connected to the rest or a motion is not constrained "
                                           f"({msg.splitlines()[0][:160]}) – see the free-free check") from None
                raise PrescribedUnsupported(msg.splitlines()[0][:200]) from None
            Xf = plan.red.expand(X) if plan is not None else X
            for col, i in enumerate(idx):
                stt, T = prep[i][0], prep[i][4]
                U = np.asarray(Xf[:3 * n, col], float).reshape(n, 3)
                S, RF = recover(study, mesh, U, None if T is None else np.full(n, stt.dT), pr, field_)
                if mesh.report.get("ties"):                                    # the force of a slave node flows into the master face (as in solve.make_step)
                    from ..ties import flow_to_masters
                    RF = RF + flow_to_masters(mesh, RF)
                results[i] = StaticOutcome(stt.name, U, S, RF, {"rigid_body_residual": ainfo.get("rigid_body_residual"), "ndof": ainfo.get("ndof"),
                                                                "backward_error": column_backward_error(info, col), "extra_nodes": nx})
            ctx.shared["rigid_body_residual"] = ainfo.get("rigid_body_residual")
        out = [results[i] for i in range(len(prep))]
    finally:
        if not keep:
            SL._remove_solver_files(job)
    return out


# ------------------------------------------------------------------------------------------------------------------ scales and helpers
@dataclass
class Scales:
    e: float                  # largest Young's modulus [MPa]
    lc: float                 # V^(1/3) [mm]
    h: float                  # mean element size (V / elements)^(1/3) [mm]
    lmax: float               # the largest extent of the bounding box [mm]
    volume: float
    x0: np.ndarray            # centre of the bounding box of the nodes in the mesh
    r_max: float              # the largest distance of a node from x0


def scales(ctx: CheckContext) -> Scales:
    from ._checks_mass import mass_properties
    props = ctx.shared.get("mass") or mass_properties(ctx.study, ctx.mesh)
    used = np.unique(ctx.mesh.tets)
    X = ctx.mesh.nodes[used]
    lo, hi = X.min(axis=0), X.max(axis=0)
    x0 = 0.5 * (lo + hi)
    V = float(props.volume_mm3)
    return Scales(e_max(ctx.study, ctx.mesh), V ** (1.0 / 3.0), (V / max(len(ctx.mesh.tets), 1)) ** (1.0 / 3.0), float((hi - lo).max()), V, x0,
                  float(np.linalg.norm(X - x0, axis=1).max()) or 1.0)


def interface_constraints(ctx: CheckContext) -> tuple[list[Any], list[Any]]:
    """(interface constraints, the others): the interface = the Fixed / Displacement supports and the rigid remote supports that hold all six components, on the interface
    selections (all of them when none are named)."""
    from ..study import Displacement, Fixed, RemoteSupport
    names = ctx.options.interface_selections
    iface, rest = [], []
    for c in ctx.study.constraints:
        rigid_remote = isinstance(c, RemoteSupport) and c.coupling == "rigid" and (len(c.dofs) != 6 or all(c.dofs))
        if (isinstance(c, (Fixed, Displacement)) or rigid_remote) and (names is None or c.selection in names):
            iface.append(c)
        else:
            rest.append(c)
    if names is not None:
        missing = [nm for nm in names if nm not in ctx.study.selections]
        if missing:
            raise PrescribedUnsupported(f"the interface selection(s) {', '.join(missing)} do not exist in the study")
    return iface, rest


def _farthest_nodes(mesh: Any, banned: np.ndarray) -> np.ndarray:
    """Four nodes that fix every rigid-body motion (farthest-point choice), none of them in `banned`."""
    used = np.setdiff1d(np.unique(mesh.tets), banned)
    if len(used) < 4:
        raise PrescribedUnsupported("not enough free nodes to pin the model")
    X = mesh.nodes[used]
    c = X.mean(axis=0)
    i0 = int(np.argmax(np.linalg.norm(X - c, axis=1)))
    i1 = int(np.argmax(np.linalg.norm(X - X[i0], axis=1)))
    d = X[i1] - X[i0]
    d = d / (np.linalg.norm(d) or 1.0)
    rel = X - X[i0]
    i2 = int(np.argmax(np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)))
    nrm = np.cross(X[i1] - X[i0], X[i2] - X[i0])
    nrm = nrm / (np.linalg.norm(nrm) or 1.0)
    i3 = int(np.argmax(np.abs(rel @ nrm)))
    return used[[i0, i1, i2, i3]]


def banned_nodes(ctx: CheckContext, study: Any) -> np.ndarray:
    """Nodes that must not be prescribed: the dependent nodes of constraint equations (tie / remote / cylindrical / bolt faces), the faces of distributing masses."""
    from .. import linear_constraints
    ban: list[np.ndarray] = []
    try:
        plan = linear_constraints.prepare(study, ctx.mesh, "direct")
    except Exception:  # noqa: BLE001 – the solver run reports the reason itself
        plan = None
    if plan is not None:
        ban.append(np.asarray(plan.red.dep, np.int64) // 3)
        ban.extend(np.asarray(idx, np.int64) for idx, _ in plan.mass_ops.values())
        ban.extend(np.asarray(v, np.int64) for v in plan.support_nodes.values())
    return np.unique(np.concatenate(ban)) if ban else np.zeros(0, np.int64)


def rigid_values(X: np.ndarray, x0: np.ndarray, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    """u = t + ω × (x − x0)."""
    return np.asarray(t, float) + np.cross(np.asarray(w, float), np.asarray(X, float) - x0)


def _net_reaction(mesh: Any, RF: np.ndarray, nodes: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(force, moment about x0) of the nodal forces RF at `nodes`."""
    f = RF[nodes]
    return f.sum(axis=0), np.cross(mesh.nodes[nodes] - x0, f).sum(axis=0)


# ------------------------------------------------------------------------------------------------------------------ check 3
STATE_VECTORS = {"UX": (0, False), "UY": (1, False), "UZ": (2, False), "RX": (0, True), "RY": (1, True), "RZ": (2, True)}


def run_unit_displacement_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    opt = ctx.options
    st0 = ctx.study
    notes: list[str] = []
    try:
        iface, rest = interface_constraints(ctx)
        if not iface:
            return skipped(3, "the study has no Fixed / Displacement support or rigid remote support (all six components) on the interface: nothing to enforce the unit "
                              "displacement on (distributing remote, cylindrical and elastic supports are not handled by this check)", time.time() - t0)
        from ..study import RemoteSupport
        remote = [c for c in iface if isinstance(c, RemoteSupport)]
        st = copy.deepcopy(st0)
        st.constraints = [copy.deepcopy(c) for c in rest] + [copy.deepcopy(c) for c in remote]
        sc = scales(ctx)
        mesh = ctx.mesh
        used = np.unique(mesh.tets)
        node_level = [c for c in iface if not isinstance(c, RemoteSupport)]
        iface_nodes = np.intersect1d(np.unique(np.concatenate([mesh.surface_nodes(c.selection) for c in node_level])) if node_level else np.zeros(0, np.int64), used)
        react_nodes = np.intersect1d(np.unique(np.concatenate([mesh.surface_nodes(c.selection) for c in iface])), used)
        delta = 1.0
        theta = delta / sc.r_max                                  # a rotation that moves the farthest node by `delta`
        states: list[PrescribedState] = []
        wanted = list(dict.fromkeys(opt.unit_states))
        if not wanted:
            return skipped(3, "no unit state was requested (unit_states is empty): nothing to enforce on the interface", time.time() - t0)
        missing_states = [n for n in UNIT_STATE_NAMES if n not in wanted]
        if missing_states:                                         # a rigid body has six independent motions: fewer prove nothing about the others
            judge.cannot_decide(f"only {len(wanted)} of the 6 unit states were enforced (not {', '.join(missing_states)}): a stray support or a coupling that blocks only the missing "
                                "motions is not found, so the check is not complete")
        for nm in wanted:
            if nm not in STATE_VECTORS:
                raise PrescribedUnsupported(f"unknown unit state '{nm}' (UX UY UZ RX RY RZ)")
            axis, rot = STATE_VECTORS[nm]
            e = np.zeros(3)
            e[axis] = 1.0
            t_vec, w_vec = (np.zeros(3), theta * e) if rot else (delta * e, np.zeros(3))
            vals = rigid_values(mesh.nodes[iface_nodes], sc.x0, t_vec, w_vec)
            rem = {c.selection: (rigid_values(np.asarray(c.point, float)[None], sc.x0, t_vec, w_vec)[0], w_vec) for c in remote}
            states.append(PrescribedState(nm, iface_nodes, vals, 0.0, rem))
        outcomes = solve_prescribed(ctx, st, states, "check_unit_displacement")
    except PrescribedUnsupported as e:
        return skipped(3, str(e), time.time() - t0)
    except PrescribedFailed as e:
        judge.violate(str(e))
        return judge.finish(3, "", {}, notes, time.time() - t0)
    values: dict[str, Any] = {"interface_selections": sorted({c.selection for c in iface}), "interface_nodes": int(len(react_nodes)),
                              "other_constraints": sorted({c.selection for c in rest}), "unit_translation_mm": delta, "unit_rotation_rad": theta,
                              "reference": {"E_MPa": sc.e, "Lc_mm": sc.lc, "element_size_mm": sc.h}, "states": {}}
    worst_e = worst_r = (0.0, "")
    for stt, o in zip(states, outcomes):
        u_exact = np.zeros((len(mesh.nodes), 3))
        u_exact[used] = rigid_values(mesh.nodes[used], sc.x0, np.zeros(3) if STATE_VECTORS[stt.name][1] else delta * np.eye(3)[STATE_VECTORS[stt.name][0]],
                                     theta * np.eye(3)[STATE_VECTORS[stt.name][0]] if STATE_VECTORS[stt.name][1] else np.zeros(3))
        dref = float(np.linalg.norm(u_exact[used], axis=1).max())
        dev = float(np.linalg.norm((o.U - u_exact)[used], axis=1).max() / dref)
        W = 0.5 * float(np.einsum("nk,nk->", o.U, o.RF))
        w_ref = 0.5 * sc.e * sc.lc * dref ** 2
        f, m = _net_reaction(mesh, o.RF, react_nodes, sc.x0)
        f_ratio = float(np.linalg.norm(f) / (sc.e * sc.lc * dref))
        m_ratio = float(np.linalg.norm(m) / (sc.e * sc.lc ** 2 * dref))
        rn = np.linalg.norm(o.RF, axis=1)
        res_ratio = rn / (sc.e * sc.h * dref)
        worst = int(np.argmax(res_ratio))
        from ..results import von_mises
        sig = float(von_mises(o.S).max() / (sc.e * dref / sc.lc))
        row: dict[str, Any] = {"displacement_deviation": dev, "strain_energy_ratio": W / w_ref, "net_force_ratio": f_ratio, "net_moment_ratio": m_ratio,
                               "reaction_ratio": max(f_ratio, m_ratio), "stress_ratio": sig, "max_nodal_residual_ratio": float(res_ratio[worst]),
                               "worst_node": {"node": worst + 1, "xyz_mm": mesh.nodes[worst].tolist(), "part": part_of_node(mesh, worst)}}
        sus = []
        for c in rest:                                              # which retained constraint holds the force: the grounding suspects
            nn = np.intersect1d(mesh.surface_nodes(c.selection), used)
            if len(nn):
                ff, _ = _net_reaction(mesh, o.RF, nn, sc.x0)
                sus.append({"selection": c.selection, "kind": getattr(c, "kind", ""), "net_force_ratio": float(np.linalg.norm(ff) / (sc.e * sc.lc * dref)),
                            "max_nodal_ratio": float(res_ratio[nn].max())})
        if sus:
            row["retained_constraints"] = sus
        values["states"][stt.name] = row
        if abs(W / w_ref) > worst_e[0]:
            worst_e = (abs(W / w_ref), stt.name)
        if row["reaction_ratio"] > worst_r[0]:
            worst_r = (row["reaction_ratio"], stt.name)
    values["worst_strain_energy_ratio"], values["worst_reaction_ratio"] = worst_e[0], worst_r[0]
    judge.compare_max(worst_e[0], "unit_disp_energy_ratio_max", f"strain energy ratio of the unit state {worst_e[1]}")
    judge.compare_max(worst_r[0], "unit_disp_reaction_ratio_max", f"net interface reaction ratio of the unit state {worst_r[1]}")
    if judge.violations:
        ws = values["states"][worst_r[1] if worst_r[0] >= worst_e[0] else worst_e[1]]
        sus = sorted(ws.get("retained_constraints", []), key=lambda d: -d["max_nodal_ratio"])
        if sus and sus[0]["max_nodal_ratio"] >= 0.1 * ws["max_nodal_residual_ratio"]:
            judge.violate(f"the model is held by a constraint that is not on the interface: '{sus[0]['selection']}' ({sus[0]['kind']}) takes the reaction")
        else:
            wn = ws["worst_node"]
            judge.violate(f"the largest residual force is at node {wn['node']} of part '{wn['part']}' at {np.round(wn['xyz_mm'], 2).tolist()} mm – a stray constraint or a "
                          "constraint equation that does not admit a rigid-body motion")
    notes.append(f"{len(states)} unit state(s) enforced on {len(react_nodes)} interface nodes; constraints that are not on the interface stay active: {len(rest)}")
    if opt.interface_selections is None and len({c.selection for c in iface}) > 1:
        notes.append("every support was treated as the interface – name the launch interface selection(s) (interface_selections) to test the other supports as stray grounding")
        judge.cannot_decide(f"the launch interface is not named (interface_selections): all {len({c.selection for c in iface})} support selections were enforced as the interface, "
                            "so a stray support that grounds the model cannot be told from the interface")
    elif len({c.selection for c in iface}) > 1 and not rest:
        # the caller named EVERY support as the interface: a stray support among them is declared part of the interface and is enforced with it, so nothing is left to be found as
        # stray grounding (the program does not know the launch interface of the study; it only knows what the caller says it is)
        notes.append(f"the interface declared by the caller is every support of the study ({', '.join(sorted({c.selection for c in iface}))}): no support was tested as stray grounding")
        judge.cannot_decide(f"the interface named in interface_selections is every support of the study ({len({c.selection for c in iface})} selections): a stray support that grounds the "
                            "model is declared part of the interface, so it cannot be told from it – name only the launch interface")
    elif opt.interface_selections is not None:
        notes.append(f"the launch interface is the one declared by the caller ({', '.join(opt.interface_selections)}); the other supports ({len(rest)}) were tested as stray grounding")
    notes.append("a rotation is enforced as u = θ × (x − x0) on the interface nodes, exact for the linear model")
    if ctx.shared.get("rigid_body_residual") is not None:
        values["assembly_rigid_body_residual"] = ctx.shared["rigid_body_residual"]
    summary = f"largest strain energy ratio {worst_e[0]:.2e}, largest net interface reaction ratio {worst_r[0]:.2e} over {len(states)} state(s)"
    return judge.finish(3, summary, values, notes, time.time() - t0)


# ------------------------------------------------------------------------------------------------------------------ check 4
def _with_common_alpha(study: Any, alpha: float) -> Any:
    """A copy of the study in which every material has ONE expansion coefficient (no mismatch between materials: the free expansion is then stress-free)."""
    st = copy.deepcopy(study)
    for m in st.materials.values():
        m.alpha = float(alpha)
        if m.ortho:
            for k in ("alpha1", "alpha2", "alpha3"):
                if k in m.ortho:
                    m.ortho[k] = float(alpha)
    return st


def run_isothermal_check(ctx: CheckContext) -> CheckResult:
    t0 = time.time()
    judge = Judge(ctx.limits)
    opt = ctx.options
    st0 = ctx.study
    notes: list[str] = []
    values: dict[str, Any] = {}
    dT = float(opt.thermal_dT_K)
    alpha = float(opt.thermal_alpha_per_K)
    if not (dT != 0.0 and alpha > 0):
        return skipped(4, "the temperature change and the test expansion coefficient must be non-zero / positive", time.time() - t0)
    lo, hi = opt.cte_range_per_K
    if range_widened(opt.cte_range_per_K, DEFAULT_CTE_RANGE):
        judge.cannot_decide(f"the plausible range of the expansion coefficient {lo:.1g}–{hi:.1g} 1/K is wider than the default {DEFAULT_CTE_RANGE[0]:.1g}–{DEFAULT_CTE_RANGE[1]:.1g}: "
                            "the screening of the materials for a unit error is weakened, so the check is not complete")
    for nm, mat in st0.materials.items():
        if mat.alpha is not None and mat.alpha > 0 and not lo <= mat.alpha <= hi:
            judge.warn(f"material '{nm}': expansion coefficient {mat.alpha:.3g} 1/K is outside the plausible range {lo:.1g}–{hi:.1g} 1/K (unit error: it is per K, not per 1e-6 K?)")
    if st0.has_thermal_loads() or st0.thermal.cases:
        notes.append("the thermal cases of the study are not used here: the check applies its own uniform temperature change to a model with one common expansion coefficient")
    try:
        sc = scales(ctx)
        mesh = ctx.mesh
        used = np.unique(mesh.tets)
        eps = alpha * dT
        # (a) the model alone: every support removed, four nodes pinned at their exact free-expansion displacement
        st_a = _with_common_alpha(st0, alpha)
        st_a.constraints = []
        banned = banned_nodes(ctx, st_a)                          # (the plan of st_a is prepared once: the pins and the coupled nodes both use it)
        pins = _farthest_nodes(mesh, banned)
        state_a = PrescribedState("isothermal", pins, eps * (mesh.nodes[pins] - sc.x0), dT)
        out_a = solve_prescribed(ctx, st_a, [state_a], "check_isothermal", alpha_bolt=alpha)[0]
        u_exact = np.zeros((len(mesh.nodes), 3))
        u_exact[used] = eps * (mesh.nodes[used] - sc.x0)
        from ..results import von_mises
        mises = von_mises(out_a.S)
        sig_ref = sc.e * eps
        worst = int(np.argmax(mises))
        dev = float(np.linalg.norm((out_a.U - u_exact)[used], axis=1).max() / (eps * sc.r_max))
        W = 0.5 * float(np.einsum("nk,nk->", out_a.U, out_a.RF))
        # the nodes of couplings the study makes on purpose (rigid / distributing faces, ties, bolt bearing faces, cylindrical supports): where the stress is, how far from them
        coupled = np.intersect1d(banned, used)
        d_coupling = float("inf")
        away = float(mises.max())
        if len(coupled):
            from scipy.spatial import KDTree
            dist = KDTree(mesh.nodes[coupled]).query(mesh.nodes)[0]
            d_coupling = float(dist[worst])
            far = used[dist[used] > 3.0 * sc.h]
            away = float(mises[far].max()) if len(far) else 0.0
        values["free_expansion"] = {"stress_ratio": float(mises.max() / sig_ref), "max_mises_MPa": float(mises.max()), "reference_stress_MPa": sig_ref,
                                    "stress_ratio_away_from_couplings": float(away / sig_ref), "distance_of_maximum_to_a_coupling_mm": d_coupling,
                                    "displacement_deviation": dev, "strain_energy_ratio": float(W / (0.5 * sc.e * eps ** 2 * sc.volume)),
                                    "pinned_nodes": [int(p) + 1 for p in pins],
                                    "worst_node": {"node": worst + 1, "xyz_mm": mesh.nodes[worst].tolist(), "part": part_of_node(mesh, worst)}}
        judge.compare_max(values["free_expansion"]["stress_ratio"], "isothermal_stress_ratio_max",
                          f"von Mises stress of the free thermal expansion relative to E·α·ΔT = {sig_ref:.3g} MPa")
        if judge.violations:
            where = (f"; it lies {d_coupling:.2g} mm from a rigid coupling / tie / bolt face – the coupling does not follow the expansion (stress away from the couplings: "
                     f"{away / sig_ref:.2g} of E·α·ΔT)") if d_coupling <= 3.0 * sc.h else ""
            judge.violate(f"the stress is largest at node {worst + 1} of part '{part_of_node(mesh, worst)}' at {np.round(mesh.nodes[worst], 2).tolist()} mm: the model resists a uniform "
                          f"expansion there (a rigid link / tie across a joint, a part held by a stiff constraint, a hidden closed loop of constraints){where}")
        # (b) the support of the study: does it restrain the free expansion? A support that holds a body only against rigid motion has no reaction under a uniform
        # temperature change; the reaction of one support group (selection) against the others is the force that the others' restraint of the expansion causes.
        if not st0.constraints:
            notes.append("the study has no support: the reaction of the support to the temperature change was not evaluated")
        else:
            st_b = _with_common_alpha(st0, alpha)
            out_b = solve_prescribed(ctx, st_b, [PrescribedState("thermal support", np.zeros(0, np.int64), np.zeros((0, 3)), dT)], "check_thermal_support", alpha_bolt=alpha)[0]
            a_eff = sc.volume / sc.lmax
            f_ref = sc.e * eps * a_eff
            per: dict[str, Any] = {}
            from ..springs import reactions as spring_reactions, supports
            springs = spring_reactions(st_b, mesh, out_b.U) if supports(st_b) else {}
            for sel in sorted({c.selection for c in st0.constraints}):
                nn = np.intersect1d(mesh.surface_nodes(sel), used)
                f, m = _net_reaction(mesh, out_b.RF, nn, sc.x0)
                if sel in springs:
                    f = f + springs[sel].sum(axis=0)
                    m = m + np.cross(mesh.nodes - sc.x0, springs[sel]).sum(axis=0)
                per[sel] = {"net_force_N": f.tolist(), "net_moment_Nmm": m.tolist(), "force_ratio": float(np.linalg.norm(f) / f_ref),
                            "moment_ratio": float(np.linalg.norm(m) / (f_ref * sc.lmax))}
            worst_sel = max(per, key=lambda k: max(per[k]["force_ratio"], per[k]["moment_ratio"]))
            worst_val = max(per[worst_sel]["force_ratio"], per[worst_sel]["moment_ratio"])
            from ..results import von_mises as _vm
            values["support"] = {"per_selection": per, "worst_selection": worst_sel, "worst_ratio": worst_val, "reference_force_N": f_ref,
                                 "max_mises_MPa": float(_vm(out_b.S).max()), "max_stress_ratio": float(_vm(out_b.S).max() / sig_ref)}
            if len(per) > 1:
                ok = judge.compare_max(worst_val, "thermal_support_reaction_ratio_max",
                                       f"reaction of the support group '{worst_sel}' to the uniform temperature change relative to E·α·ΔT·A = {f_ref:.3g} N")
                if ok is False:
                    judge.violate("the supports of the study restrain the thermal expansion against each other (statically indeterminate: the support groups "
                                  f"{', '.join(per)} push on each other) – a free body needs six restraints only")
            else:
                notes.append("one support group: its reaction to a uniform temperature change is zero by equilibrium and says nothing – the stress of the supported model "
                             f"({values['support']['max_stress_ratio']:.2g} of E·α·ΔT) is judged instead")
                judge.compare_max(values["support"]["max_stress_ratio"], "thermal_support_stress_ratio_max",
                                  f"von Mises stress of the supported model under the uniform temperature change relative to E·α·ΔT = {sig_ref:.3g} MPa")
    except PrescribedUnsupported as e:
        return skipped(4, str(e), time.time() - t0, notes)
    except PrescribedFailed as e:
        judge.violate(str(e))
        return judge.finish(4, "", values, notes, time.time() - t0)
    notes.append(f"uniform ΔT = {dT:g} K, every material given the one expansion coefficient {alpha:g} 1/K (a test value); thermal elongation of bolts with the same coefficient")
    fe = values["free_expansion"]
    summary = f"free expansion: stress ratio {fe['stress_ratio']:.2e}" + (f", support reaction ratio {values['support']['worst_ratio']:.2e}, support stress ratio {values['support']['max_stress_ratio']:.2e}" if "support" in values else "")
    return judge.finish(4, summary, jsonable(values), notes, time.time() - t0)

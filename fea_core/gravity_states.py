"""Gravity load cases are linear combinations of three unit states.

A load case that is ONE gravity load in the GLOBAL system (no `csys`) and nothing else gives, in a linear study, the displacements, stresses and nodal forces
u = Σ aₖ·uₖ, where a = the acceleration vector and uₖ = the state of the unit acceleration along the global axis k. The QSL cases of a launch analysis (14 signed
cases), the unit base accelerations of the dynamics and the unit-gravity statics of the model checks are all of this kind: they are solved once as three unit states
and superposed (`fea_core.combinations` does the same for the combinations of the user).

`pure_gravity` says whether a case is such a case. Whether the STUDY allows the superposition (linear, no contact, no joints, no thermal state, homogeneous constraints, …) is
`superposable`; the consumers fall back to solving every case when it says no.
"""
from __future__ import annotations

from typing import Any

import numpy as np

G_MM_S2 = 9810.0                                     # the program's convention (CLAUDE.md): g = 9810 mm/s²
UNIT_PREFIX = "_unit_gravity_"
AXIS_LETTERS = "xyz"


def pure_gravity(lc: Any) -> np.ndarray | None:
    """The acceleration vector (3,) [mm/s²] of a load case that is exactly one global gravity load, else None."""
    from .study import Gravity
    loads = list(getattr(lc, "loads", []) or [])
    if len(loads) != 1 or not isinstance(loads[0], Gravity) or loads[0].csys:
        return None
    v = np.asarray(loads[0].vector, float)
    return v if v.shape == (3,) and bool(np.all(np.isfinite(v))) else None


def unit_case(axis: int, magnitude: float = 1.0, name: str | None = None) -> Any:
    """The load case of the acceleration `magnitude` [mm/s²] along the global axis."""
    from .study import Gravity, LoadCase
    vec = [0.0, 0.0, 0.0]
    vec[axis] = float(magnitude)
    return LoadCase(name or f"{UNIT_PREFIX}{AXIS_LETTERS[axis]}", [Gravity(vector=vec)])


def prescribed_nonzero(study: Any) -> bool:
    """A constraint with a prescribed displacement other than zero (such a state is not a multiple of the load: the superposition does not hold)."""
    from .study import Displacement
    return any(isinstance(c, Displacement) and any(v not in (None, 0, 0.0) for v in (c.ux, c.uy, c.uz)) for c in study.constraints)


def superposable(study: Any, mesh: Any = None) -> str | None:
    """None when every pure-gravity case of the study is a linear combination of the unit states, otherwise the reason it is not (the caller then solves every case)."""
    if study.is_nonlinear(mesh=mesh):
        return "the study is nonlinear (contact, NLGEOM, plasticity, a stepped analysis)"
    if study.active_bolts() or study.active_welds():
        return "bolts / welds (joint forces are evaluated per state)"
    if getattr(study.frame, "active", False):
        return "a beam system"
    if getattr(study.twod, "enabled", False) or getattr(study.shell, "enabled", False):
        return "a 2D / shell model (no solid mesh states)"
    if getattr(study, "cyclic", None) is not None and study.cyclic.enabled:
        return "cyclic symmetry"
    if study.has_thermal_loads() or study.thermal.cases:
        return "thermal states"
    if getattr(study.printing, "creep", None) is not None and study.printing.creep.enabled:
        return "3D printing creep"
    from .study import RemoteSupport
    if any(isinstance(c, RemoteSupport) for c in study.constraints):
        return "a remote support (its reaction is measured per state)"
    if prescribed_nonzero(study):
        return "a prescribed displacement (the state is not proportional to the load)"
    if mesh is not None:
        from .printing import shell_of
        if shell_of(mesh) is not None:
            return "3D printed parts (the failure criterion is evaluated per state)"
    return None


# ======================================================================================================================================== the unit states of a study
# The three unit states (1 g along x, y, z) of a linear study on a mesh are kept in the work folder of the run that computed them (`fea_core.state_cache`, kind "gravunits"):
# the displacements, the stresses, the raw nodal forces RF (K·u in every node), the displacements of the reference points of the remote loads / supports and – on the own assembly – the
# nodal forces the solver applied. Every gravity case of the study (the 14 signed QSL cases of a launch analysis), the unit base accelerations of the dynamics and the unit-gravity
# statics of the model checks are then linear combinations of them: the combined arrays go through `solve.make_step` again, so the reactions and the EQUILIBRIUM CHECK are those of
# the combined state, not of the unit ones.
#
# The key of a unit state is the digest of the model that was ACTUALLY solved (`state_cache.solved_digest`: the study with exactly the load cases of the deck – a rigid remote load of
# another case is part of the model of every state of its deck), and the family of the solver that solved it (CalculiX or the own assembly; a fallback from one to the other changes it).
import dataclasses  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

KIND = "gravunits"
MIN_CASES = 3                                             # fewer gravity cases than this are solved as they are (the unit states would save nothing) unless all unit states are stored


def family_of(backend: str) -> str:
    """"ccx" (CalculiX) or "own" (the own assembly with the direct / iterative solver): the two compute K·u and the applied forces in their own way, a unit state is used only by
    its own family."""
    return "ccx" if backend == "ccx" else "own"


def str_missing_by_superposition(info: Any, step_name: str | None = None) -> bool:
    """Is the CalculiX stress error estimate (STR) of a state missing BECAUSE the state is a combination of unit states: the state `step_name` (any gravity case of the result when
    None) is one of the superposed cases of a CalculiX run (`info["gravity_superposition"]`). A state that is not (another load case, a result of the direct / iterative solver) lacks
    the estimate for the usual reason (only CalculiX computes it)."""
    gs = (info or {}).get("gravity_superposition") if isinstance(info, dict) else None
    if not isinstance(gs, dict):
        return False
    family = gs.get("family") or family_of(str((info or {}).get("backend") or ""))
    cases = [str(c) for c in (gs.get("cases") or [])]
    return family == "ccx" and (bool(cases) if step_name is None else str(step_name) in cases)


def unit_model(study: Any, axes: list[int]) -> Any:
    """The model of a deck of the unit cases ALONE: the study with only those cases. Nothing else of the study changes: the superposition is only planned for a study without active
    bolts (`superposable`), where the bolt preload switch is not part of the model – `state_cache.study_digest` leaves it out then, so this model and the plan's model share their keys."""
    import copy
    st = copy.copy(study)
    st.load_cases = [unit_case(a, G_MM_S2) for a in axes]
    return st


def has_reference_points(study: Any) -> bool:
    """Does a load case have a remote load (a reference node whose displacement the steps of the deck carry)?"""
    from .study import RemoteLoad
    return any(isinstance(ld, RemoteLoad) for lc in study.load_cases for ld in lc.loads)


def reference_signature(cases: list[Any]) -> str:
    """The reference points the unit states of a deck carry the displacements of: the remote loads of `cases` in the order of the deck (the reference nodes are numbered by it) with
    the surface they couple, the point and the coupling. "" without a remote load. A distributing remote load is not part of the key of a state (it changes none of the unit
    displacements), but what the state stores about its reference point is: a state is only used for a deck that has the same ones."""
    from .study import RemoteLoad
    import hashlib
    import json
    refs = [[str(ld.selection), [float(x) for x in ld.point], str(ld.coupling)] for lc in cases for ld in lc.loads if isinstance(ld, RemoteLoad)]
    return hashlib.sha256(json.dumps(refs, separators=(",", ":")).encode("utf-8")).hexdigest()[:24] if refs else ""


def _entry(digest: str, axis: int, family: str, deck: str | None = None, ref_sig: str | None = None) -> Path | None:
    """The intact stored unit state of the axis in the family; deck = "study" only one that was solved in the deck of the whole study (it carries the reference-point displacements of
    its remote loads), None = any; ref_sig = only one that carries the reference points of that signature (`reference_signature`; "" = none), None = any."""
    from . import state_cache as SC
    for folder in SC.entries(KIND, digest):
        m = SC.meta_of(folder)
        if (m is not None and m.get("axis") == axis and m.get("family") == family and (family == "ccx" or (folder / "f_app.npy").is_file()) and (deck is None or m.get("deck") == deck)
                and (ref_sig is None or m.get("ref_sig", "") == ref_sig)):
            return folder
    return None


def unit_arrays(folder: Path) -> dict[str, Any]:
    from . import state_cache as SC
    out: dict[str, Any] = {nm: SC.array(folder, nm) for nm in ("U", "S", "RF")}
    out["f_app"] = SC.array(folder, "f_app") if (Path(folder) / "f_app.npy").is_file() else None
    out["meta"] = SC.meta_of(folder) or {}
    out["refd"] = None
    if (Path(folder) / "refd_keys.npy").is_file():
        keys, vals = SC.array(folder, "refd_keys"), SC.array(folder, "refd_vals")
        out["refd"] = {int(k): np.asarray(vals[i], float) for i, k in enumerate(keys)}
    out["folder"] = Path(folder)
    return out


def save_unit(root: Path, digest: str, axis: int, family: str, U: Any, S: Any, RF: Any, f_app: Any, info: dict[str, Any], refd: dict[int, Any] | None = None,
              deck: str = "study", ref_sig: str = "") -> Path | None:
    from . import state_cache as SC
    arrays: dict[str, np.ndarray] = {"U": np.asarray(U, np.float64), "S": np.asarray(S, np.float64), "RF": np.asarray(RF, np.float64)}
    if f_app is not None:
        arrays["f_app"] = np.asarray(f_app, np.float64)
    if refd:
        keys = sorted(refd)
        arrays["refd_keys"] = np.asarray(keys, np.int64)
        arrays["refd_vals"] = np.stack([np.asarray(refd[k], np.float64) for k in keys])
    return SC.write_entry(Path(root), KIND, digest, f"{AXIS_LETTERS[axis]}_{family}", arrays, {"axis": int(axis), "family": family, "deck": deck, "ref_sig": ref_sig, "info": info})


@dataclasses.dataclass
class Plan:
    digest: str
    family: str
    gravity: list[tuple[int, np.ndarray]]                 # (index in study.load_cases, acceleration vector)
    axes: list[int]
    cached: dict[int, Path]
    missing: list[int]
    ref_sig: str = ""                                      # the reference points of the deck (`reference_signature`)


def plan(study: Any, mesh: Any, backend: str) -> Plan | None:
    """The superposition plan of the gravity cases of `study`, or None when it saves nothing / is not valid (see `superposable`): at least `MIN_CASES` gravity cases and more of
    them than the unit states still to be solved – or every unit state they need is stored already. The unit states are keyed by the model that will be solved: the other cases of the
    study (with their rigid remote loads) and the unit cases of the axes that are missing."""
    from . import state_cache as SC
    if not SC.enabled():
        return None
    if any(str(lc.name).startswith(UNIT_PREFIX) for lc in study.load_cases):
        return None                                           # a case named like a unit state could not be told from one
    grav = [(i, v) for i, lc in enumerate(study.load_cases) for v in [pure_gravity(lc)] if v is not None and bool(np.any(v != 0.0))]
    if not grav:
        return None
    axes = sorted({k for _, v in grav for k in range(3) if v[k] != 0.0})
    if len(grav) < MIN_CASES and not SC.any_entries(KIND):      # too few cases to solve the unit states for them and nothing stored to take them from
        return None
    if superposable(study, mesh) is not None:
        return None
    family = family_of(backend)
    gi = {i for i, _ in grav}
    digest = SC.solved_digest(study, mesh, [lc for i, lc in enumerate(study.load_cases) if i not in gi])
    deck = "study" if has_reference_points(study) else None
    ref_sig = reference_signature([lc for i, lc in enumerate(study.load_cases) if i not in gi])
    cached = {a: f for a in axes for f in [_entry(digest, a, family, deck, ref_sig)] if f is not None}
    missing = [a for a in axes if a not in cached]
    if missing and not (len(grav) >= MIN_CASES and len(grav) > len(missing)):
        return None
    return Plan(digest, family, grav, axes, cached, missing, ref_sig)


def reduced_study(pl: Plan, study: Any) -> Any:
    """The study without the gravity cases and with the unit states still to be solved (a shallow copy: nothing of the study of the caller changes). Its cases are the other cases of
    `study` in their order, then one unit case per missing axis (`expand` finds the unit steps by that position)."""
    import copy
    st = copy.copy(study)
    gi = {i for i, _ in pl.gravity}
    st.load_cases = [lc for i, lc in enumerate(study.load_cases) if i not in gi] + [unit_case(a, G_MM_S2) for a in pl.missing]
    return st


def _origin(step: Any) -> dict[str, Any] | None:
    """The CalculiX job a solved step came from (work folder, job name, step index) – None for a state of the own assembly."""
    wd = getattr(step, "_workdir", None)
    if wd is None:
        return None
    return {"workdir": str(wd), "job": getattr(step, "_job", "job"), "step_index": int(getattr(step, "_step_index", 0))}


def settle_family(pl: Plan, res: Any) -> None:
    """The family of the solver that really solved the reduced study (`res.info["backend"]`): a direct / iterative attempt that failed and was solved by CalculiX is a CalculiX
    state, whatever the plan said. A plan that also took unit states from the other family would mix two solvers – refused (the caller then solves every case)."""
    backend = res.info.get("backend")
    if not backend or not pl.missing:
        return
    fam = family_of(str(backend))
    if fam == pl.family:
        return
    if pl.cached:
        raise ValueError(f"the unit states were solved by {fam} but stored ones are of {pl.family} – two solvers are not mixed")
    pl.family = fam


def _combine(arrs: dict[int, Any], v: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape)
    for a, x in arrs.items():
        c = float(v[a]) / G_MM_S2
        if c != 0.0:
            out += c * np.asarray(x, float)
    return out


def _combine_refd(units: dict[int, dict[str, Any]], v: np.ndarray) -> dict[int, np.ndarray]:
    """The displacements of the reference points (remote loads / supports) of a combined state: the same combination of those of the unit states."""
    used = [a for a in units if float(v[a]) != 0.0]
    keys: set[int] = set()
    for a in used:
        keys |= set(units[a].get("refd") or {})
    out: dict[int, np.ndarray] = {}
    for k in sorted(keys):
        if any(k not in (units[a].get("refd") or {}) for a in used):
            raise ValueError("the unit states do not have the same reference points")
        out[k] = np.asarray(sum(float(v[a]) / G_MM_S2 * np.asarray(units[a]["refd"][k], float) for a in used), float)
    return out


def expand(pl: Plan, study: Any, mesh: Any, res: Any, workdir: Path, log: Any) -> None:
    """Replaces the results of the reduced study by those of `study`: the solved unit steps are stored and dropped, every gravity case is made from the unit states (U, S, RF, the
    reference-point displacements and the applied forces superposed, then `solve.make_step`: reactions + the equilibrium check of the COMBINED state), the other cases keep their steps.
    The order of the steps is that of `study.load_cases`. The steps of the reduced study are those of its cases by POSITION (the other cases, then the unit states)."""
    from .solve import constraint_masks, make_step
    n = len(mesh.nodes)
    gravity = dict(pl.gravity)
    others = [i for i in range(len(study.load_cases)) if i not in gravity]
    solved = list(res.steps)
    if len(solved) != len(others) + len(pl.missing):
        raise ValueError("the results of the reduced study do not match its cases")
    settle_family(pl, res)
    own = pl.family == "own"
    unit_steps = {a: solved[len(others) + j] for j, a in enumerate(pl.missing)}
    other_steps = dict(zip(others, solved[:len(others)]))
    units: dict[int, dict[str, Any]] = {}
    reused: list[str] = []
    for a in pl.axes:
        if a in pl.cached:
            units[a] = unit_arrays(pl.cached[a])
            reused.append(f"{AXIS_LETTERS[a]}: {pl.cached[a].name}")
            continue
        st = unit_steps[a]
        RF, f_app = st._unit_arrays
        info = {"backend": res.info.get("backend"), "solver": res.info.get("solver"), "ccx_solver": res.info.get("ccx_solver")}
        origin = _origin(st)
        if origin is not None:
            info["origin"] = origin                          # where the integration-point stresses of this state can be had from (the section forces of a combined state)
        save_unit(workdir, pl.digest, a, pl.family, st.U, st.S, RF, f_app, info, st.ref_displacements, ref_sig=pl.ref_sig)
        units[a] = {"U": st.U, "S": st.S, "RF": RF, "f_app": f_app, "meta": {}, "step": st, "refd": dict(st.ref_displacements or {})}
    for st in solved:                                       # (the arrays of the solver are not kept on any step that stays)
        if hasattr(st, "_unit_arrays"):
            del st._unit_arrays
    if reused:
        log("Gravity cases: stored unit states used (" + "; ".join(reused) + ")")
    if not res.info.get("backend"):                        # nothing was solved: the facts of the run that made the stored unit states
        m = next(iter(units.values()))["meta"].get("info") or {}
        res.info.update({k: v for k, v in m.items() if v is not None})
        if isinstance(res.info.get("solver"), dict):
            res.info["solver"] = {**res.info["solver"], "stored_unit_states": True}
    masks = constraint_masks(study, mesh, own=own)
    stubs: dict[int, Any] = {}                              # one stand-in per unit state: the section-force cache of its integration-point stresses is shared by every combined state
    for a, u in units.items():
        s = u.get("step")
        o = _origin(s) if s is not None else (u["meta"].get("info") or {}).get("origin")
        if o is not None and (Path(o["workdir"]) / f"{o['job']}.inp").is_file():        # (a stored state: the job of the run that solved it, while its work folder is there)
            stubs[a] = types.SimpleNamespace(_workdir=Path(o["workdir"]), _job=o["job"], _step_index=int(o["step_index"]))
    steps: list[Any] = []
    for i, lc in enumerate(study.load_cases):
        if i not in gravity:
            steps.append(other_steps[i])
            continue
        v = gravity[i]
        U = _combine({a: u["U"] for a, u in units.items()}, v, (n, 3))
        S = _combine({a: u["S"] for a, u in units.items()}, v, (n, 6))
        RF = _combine({a: u["RF"] for a, u in units.items()}, v, (n, 3))
        f_app = None
        if own:
            if any(u["f_app"] is None for a, u in units.items() if float(v[a]) != 0.0):
                raise ValueError("a stored unit state without the applied forces cannot serve the own assembly")
            f_app = _combine({a: u["f_app"] for a, u in units.items() if u["f_app"] is not None}, v, (n, 3))
        st = make_step(study, mesh, lc, U, S, RF, masks, _combine_refd(units, v), 0.0, f_applied=f_app)
        used = [a for a in units if float(v[a]) != 0.0]
        if all(a in stubs for a in used):
            setattr(st, "_combo", [(stubs[a], float(v[a]) / G_MM_S2) for a in used])       # the section forces of this state are those of its unit states, combined (`section_forces`)
        steps.append(st)
    res.steps = steps
    sol = res.info.get("solver")
    if isinstance(sol, dict) and isinstance(sol.get("cases"), list):       # the accuracy table of the direct solver lists the states it solved: the unit ones are named as what they are
        for c in sol["cases"]:
            if isinstance(c, dict) and str(c.get("name", "")).startswith(UNIT_PREFIX):
                c["name"] = f"unit state {str(c['name'])[len(UNIT_PREFIX):]} (1 g; the gravity cases are combinations of the unit states)"
    res.info["gravity_superposition"] = {"cases": [study.load_cases[i].name for i, _ in pl.gravity], "unit_states": [AXIS_LETTERS[a] for a in pl.axes],
                                         "solved": [AXIS_LETTERS[a] for a in pl.missing], "stored": [AXIS_LETTERS[a] for a in pl.cached], "family": pl.family}
    log(f"Gravity cases: {len(pl.gravity)} cases from {len(pl.axes)} unit states ({len(pl.missing)} solved, {len(pl.cached)} stored)")


def solve_units_ccx(study: Any, mesh: Any, axes: list[int], workdir: Path, log: Any, on_line: Any = None, cancel: Any = None) -> dict[int, dict[str, Any]]:
    """The unit states (1 g along the axes) of the study in CalculiX – the unit cases ALONE form the deck (the other cases of the study, with their rigid remote loads, are not part
    of it), so the states are stored under the key of that model; stored in the work folder (`save_unit`) and returned as {axis: {"U", "S", "RF", "f_app": None}}. The states are
    independent linear statics: with several of them the steps run as concurrent jobs where the machine has the room (`ccx.parallel`, as `dynamics.modal.static_solutions`)."""
    from .ccx import parallel
    from .ccx.frd import read_frd
    from .ccx.inp import write_inp
    from .ccx.runner import CcxError, run_ccx
    from .config import CCX_EXE
    from .solve import _nodal
    from . import state_cache as SC
    st = unit_model(study, axes)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    job = "grav_units"
    n = len(mesh.nodes)
    log(f"Unit gravity states ({', '.join(AXIS_LETTERS[a] for a in axes)}): CalculiX")
    write_inp(st, mesh, workdir / f"{job}.inp")
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    workers, threads = parallel.plan(len(axes), 3 * n) if parallel.enabled() else (1, 0)
    if workers > 1:
        prun = parallel.run_steps(workdir, job, 3 * n, exe=exe, on_line=on_line, cancel=cancel, workers=workers, threads=threads)
        frd = parallel.read_frd_steps(workdir, prun.jobs)
        if frd is None:
            raise CcxError("CalculiX (unit gravity states) returned no result")
    else:
        r = run_ccx(workdir, job, exe=exe, on_line=on_line, cancel=cancel)
        if not r.finished or r.errors:
            raise CcxError("CalculiX (unit gravity states) ended with an error:\n  " + "\n  ".join(r.errors or [f"code {r.returncode}"]))
        frd = read_frd(workdir / f"{job}.frd")
    out: dict[int, dict[str, Any]] = {}
    digest = SC.study_digest(st, mesh)                    # the model that was solved: the unit cases only
    for a, cs in zip(axes, frd.steps()):
        U, S, RF = (_nodal(frd, name, cs, n, nc) for name, nc in (("DISP", 3), ("STRESS", 6), ("FORC", 3)))
        save_unit(workdir, digest, a, "ccx", U, S, RF, None, {"backend": "ccx"}, deck="units")
        out[a] = {"U": U, "S": S, "RF": RF, "f_app": None}
    return out


def static_states(study: Any, mesh: Any, cases: list[Any], workdir: Path, log: Any, on_line: Any = None, cancel: Any = None) -> list[dict[str, Any]] | None:
    """The static solutions {"U" (N, 3), "S" (N, 6) float32, "RF" (constrained nodes, 3)} of gravity `cases` (the unit base accelerations of the dynamics, as
    `dynamics.modal.static_solutions` gives them from CalculiX) as combinations of the unit states of the study – stored from an earlier run, or solved here once in ONE CalculiX job
    for the axes that are missing. The states are those of the model of the unit cases alone (`solve_units_ccx`): a stored state of a run whose deck also held a rigid remote load is
    another model and is not taken. None when a case is not a pure gravity case or the study is not superposable (the caller then solves the cases as before)."""
    from . import state_cache as SC
    if not SC.enabled() or not cases or superposable(study, mesh) is not None:
        return None
    vecs = [pure_gravity(lc) for lc in cases]
    if any(v is None for v in vecs):
        return None
    axes = sorted({k for v in vecs for k in range(3) if v is not None and v[k] != 0.0})
    if not axes:
        return None
    digest = SC.study_digest(unit_model(study, axes), mesh)
    units: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for a in axes:
        f = _entry(digest, a, "ccx")
        if f is None:
            missing.append(a)
        else:
            units[a] = unit_arrays(f)
    if units:
        log("Static states of the gravity cases: stored unit states used (" + "; ".join(f"{AXIS_LETTERS[a]}: {u['folder'].name}" for a, u in units.items()) + ")")
    if missing:
        units.update(solve_units_ccx(study, mesh, missing, workdir, log, on_line, cancel))
    from .dynamics.modal import constrained_nodes
    cn = constrained_nodes(study, mesh)
    n = len(mesh.nodes)
    out = []
    for v in vecs:
        assert v is not None
        out.append({"U": _combine({a: u["U"] for a, u in units.items()}, v, (n, 3)),
                    "S": _combine({a: u["S"] for a, u in units.items()}, v, (n, 6)).astype(np.float32),
                    "RF": _combine({a: u["RF"] for a, u in units.items()}, v, (n, 3))[cn]})
    return out

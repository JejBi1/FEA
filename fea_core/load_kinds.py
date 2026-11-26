"""Registry of load and constraint kinds: one handler per kind (study.LOAD_KINDS / study.CONSTRAINT_KINDS key).

Everything that depends on the kind of a load is declared here once and used by the core:
  - loads.nodal_loads / loads.load_resultant  – `nodal`, `remote`, `resultant`
  - ccx/inp.write_inp (sets) and write_loads  – `ccx_sets`, `ccx_cards` (default: nodal forces → *CLOAD)
  - solve.check_loads                         – `entities`, `csys`, `entity_check`, `check`
  - solve_large.unsupported / run_ngs         – `ngs_unsupported`, `ngs_nodal`
  - sweep.scale_study_loads                   – `scale`
  - loads_text.load_text                      – `describe`

A handler only REFERS to the existing functions (loads.force_nodal …, called through the module at call time, so the
per-calculation cache `loads.nodal_cache()` and monkeypatching keep working); the numbers are the same as before the
registry (bit-identical nodal loads, byte-identical .inp – tests/test_load_kinds.py).

Adding a load kind: see `LOAD_HANDLERS` below and CLAUDE.md ("Adding a load kind"); tests/test_load_kinds.py fails while
a class in study.LOAD_KINDS has no complete handler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import study as S

# backends of the 3D solid model (solve.choose_backend); frame / 2D / shell studies have their own solvers
ALL_BACKENDS = frozenset({"ccx", "ngs_hypre", "ngs_direct", "ngs_pardiso"})      # ngs_pardiso = the stored name of ngs_direct


def _loads():
    from . import loads
    return loads


# ======================================================================= load handlers

@dataclass(frozen=True)
class LoadKind:
    """Behaviour of one load kind. Callables get the study, the mesh and the load object `l`.

    domain   "solid" (nodal forces on the tet10 mesh), "frame" (own beam solver, beams/), "thermal" (temperature field,
             thermal/field – no nodal forces here)
    entities selection entities allowed ("faces", "edges", "points"); () = the load has no selection
    csys     "none" | "cartesian" | "cylindrical" – the most general local coordinate system allowed
    nodal(study, mesh, l) -> (N, 3) consistent nodal forces (None = no nodal forces of the solid mesh)
    remote(study, l) -> (force, moment about the ORIGIN) applied at a point outside the mesh (remote load)
    resultant(study, mesh, l) -> (3,) exact resultant force; None = nodal(...).sum(axis=0) (+ remote force)
    scale(l, factor) – in place, the load × factor (sweep.scale_study_loads); raises ValueError if impossible
    describe(l, st, num) -> one sentence with units (loads_text.load_text)
    check(study, lc, l, part_names, what) -> (errors, warnings) – kind-specific readiness check (solve.check_loads)
    entity_check(what, entity) -> (errors, warnings) | None – own rule for the selection entity (None = generic)
    ccx_cards(ctx, l) – own ccx writing in write_loads (ctx = CcxLoadContext); None = nodal → *CLOAD
    ccx_row_mask – default *CLOAD: only rows with a nonzero component (as before for all kinds except Force)
    ccx_sets(l) -> (needs NSET, needs SURFACE) of l.selection in the .inp
    ngs_nodal(study, mesh, l) -> (N, 3) extra nodal forces for the NGSolve backends (distributing remote load)
    ngs_unsupported(l, lc) -> reason (str) why NGSolve cannot compute it (then auto → ccx), None = fine
    backends – solid backends that support the kind (documentation + tests; the fine check is ngs_unsupported)
    """
    kind: str
    cls: type
    domain: str
    entities: tuple = ()
    csys: str = "none"
    nodal: Callable | None = None
    remote: Callable | None = None
    resultant: Callable | None = None
    scale: Callable | None = None
    describe: Callable | None = None
    check: Callable | None = None
    entity_check: Callable | None = None
    ccx_cards: Callable | None = None
    ccx_row_mask: bool = True
    ccx_sets: Callable | None = None
    ngs_nodal: Callable | None = None
    ngs_unsupported: Callable | None = None
    backends: frozenset = ALL_BACKENDS


@dataclass
class CcxLoadContext:
    """What a `ccx_cards` hook may use inside ccx/inp.write_loads of one step."""
    study: object
    mesh: object
    add_nodal: Callable            # add_nodal(F (N, 3), rows (N,) bool | None) → *CLOAD of the mesh nodes
    dload: list                    # *DLOAD lines of the step
    body_forces: bool              # False = the model has U1 elements (bolts, frames) → no *DLOAD GRAV


# ---------------------------------------------------------------- nodal forces (through the loads module: cache)

def _force_nodal(study, mesh, l):
    return _loads().force_nodal(study, mesh, l)


def _pressure_nodal(study, mesh, l):
    return _loads().pressure_nodal(mesh, l)


def _gravity_nodal(study, mesh, l):
    L = _loads()
    return L.gravity_nodal_forces(study, mesh, L.global_vector(study, l))


def _rotation_nodal(study, mesh, l):
    return _loads().rotation_forces(study, mesh, l)


def _inertia_nodal(study, mesh, l):
    return _loads().inertia_forces(study, mesh, l)


def _bearing_nodal(study, mesh, l):
    return _loads().bearing_nodal(study, mesh, l)


def _remote_point(study, l):
    f, m = _loads().remote_vectors(study, l)
    return f, m + np.cross(np.asarray(l.point, float), f)


def _remote_distributing(study, mesh, l):
    """The nodal forces of a DISTRIBUTING remote load (RBE3) for the own assembly; a rigid one acts on the extra DOFs of its rigid body
    (`linear_constraints.Plan.extra_loads`), not on the nodes."""
    L = _loads()
    if l.coupling == "rigid":
        return np.zeros((len(mesh.nodes), 3))
    return L.distributing_forces(mesh, l.selection, l.point, *L.remote_vectors(study, l))


# ---------------------------------------------------------------- resultants (exact, not always Σ nodal)

def _force_resultant(study, mesh, l):
    from . import csys
    cs = csys.get(study, l.csys)
    if cs is not None and cs.kind == "cylindrical":     # the direction turns over the face: resultant ≠ F
        return _loads().force_nodal(study, mesh, l).sum(axis=0)
    return _loads().global_vector(study, l)


def _pressure_resultant(study, mesh, l):
    L = _loads()
    if l.variable:
        return L.pressure_nodal(mesh, l).sum(axis=0)
    return -l.value * L.vector_area(mesh, l.selection)


def _gravity_resultant(study, mesh, l):
    from .printing import element_density
    L = _loads()
    rho = element_density(study, mesh)
    return (L.element_volumes(mesh) * rho).sum() * L.global_vector(study, l)


def _vector_resultant(study, mesh, l):
    return _loads().global_vector(study, l)


def _remote_resultant(study, mesh, l):
    return _loads().remote_vectors(study, l)[0]


# ---------------------------------------------------------------- scaling (sweep.scale_study_loads)

def _scale_vector(l, factor):
    if hasattr(l, "vector") and l.vector:
        l.vector = [float(v) * factor for v in l.vector]


def _scale_pressure(l, factor):
    # A clipped (hydrostatic) pressure is positively homogeneous only: for factor < 0 the scaled gradient moves the
    # wetted side, so the result is not factor × baseline – sweep.shortcut_blockers(factors=) blocks the shortcut then.
    if hasattr(l, "value"):
        l.value = float(l.value) * factor
    if hasattr(l, "gradient") and l.gradient:
        l.gradient = [float(g) * factor for g in l.gradient]


def _scale_force_moment(l, factor):
    if hasattr(l, "force") and l.force:
        l.force = [float(v) * factor for v in l.force]
    if hasattr(l, "moment") and l.moment:
        l.moment = [float(v) * factor for v in l.moment]


def _scale_rotation(l, factor):
    if factor < 0 and getattr(l, "omega", 0.0) != 0.0:
        raise ValueError(f"Negative load_factor={factor} cannot be used for rotation with a nonzero angular velocity omega")
    if hasattr(l, "omega"):
        l.omega = float(l.omega) * math.sqrt(factor) if factor >= 0 else float(l.omega)
    if hasattr(l, "alpha"):
        l.alpha = float(l.alpha) * factor


def _scale_inertia(l, factor):
    if hasattr(l, "accel") and l.accel:
        l.accel = [float(a) * factor for a in l.accel]
    if hasattr(l, "alpha") and l.alpha:
        l.alpha = [float(a) * factor for a in l.alpha]
    if hasattr(l, "omega") and l.omega:
        if factor < 0 and any(w != 0.0 for w in l.omega):
            raise ValueError(f"Negative load_factor={factor} cannot be used for inertia with a nonzero omega")
        l.omega = [float(w) * math.sqrt(factor) for w in l.omega] if factor >= 0 else l.omega
    if hasattr(l, "wmat") and l.wmat:
        l.wmat = [[float(v) * factor for v in row] for row in l.wmat]


def _scale_frame_member(l, factor):
    if hasattr(l, "q") and l.q:
        l.q = [float(v) * factor for v in l.q]
    if hasattr(l, "q_end") and l.q_end:
        l.q_end = [float(v) * factor for v in l.q_end]


def _scale_frame_temperature(l, factor):
    l.dT, l.grad_z, l.grad_y = (float(v) * factor for v in (l.dT, l.grad_z, l.grad_y))


def _scale_none(l, factor):
    """Not scaled (temperature in °C is not a force load)."""


# ---------------------------------------------------------------- descriptions (loads_text)

def _lt():
    from . import loads_text
    return loads_text


def _describe_force(l, st, num):
    T = _lt()
    n = len(st.selections[l.selection].points) if T.entity_of(l, st) == "points" else 0
    each = f", {n} points of F/{n} each" if n > 1 else ""
    return f"{T.force_kind(l, st)}: F = {T.vec_text(l.vector, num)} N{T.csys_suffix(l, st)}{each}"


def _describe_pressure(l, st, num):
    return _lt().pressure_text(l, num)


def _describe_rotation(l, st, num):
    return "rotation: " + _lt().rotation_text(l, num)


def _describe_inertia(l, st, num):
    return "inertia: " + _lt().inertia_text(l, num)


def _describe_remote(l, st, num):
    T = _lt()
    cpl = "distributing (RBE3)" if l.coupling == "distributing" else "rigid"
    return (f"remote load ({cpl}), point {T.vec_text(l.point, num)} mm: F = {T.vec_text(l.force, num)} N, M = {T.vec_text(l.moment, num)} N·mm{T.csys_suffix(l, st)}")


def _describe_bearing(l, st, num):
    T = _lt()
    return f"bearing load F = {T.vec_text(l.vector, num)} N{T.csys_suffix(l, st)}"


def _describe_gravity(l, st, num):
    T = _lt()
    return f"gravity g = {T.vec_text(l.vector, num)} mm/s²{T.csys_suffix(l, st)}"


def _describe_kind(l, st, num):
    """No sentence in loads_text (thermal: mcp.diagnostics.thermal_load_text, frames: report.frame.load_text)."""
    return getattr(l, "kind", "")


# ---------------------------------------------------------------- readiness checks (solve.check_loads)

def _force_entity_check(what, e):
    if e == "mixed":
        return [f"{what}: selection mixes faces, edges, and points – force needs a single kind"], []
    if e in ("edges", "points"):
        return [], [f"{what}: force on {('edge' if e == 'edges' else 'point')} is singular in a solid model – stress and displacement near it grow with mesh refinement; evaluate away from the point of application (or load a small face)"]
    return [], []


def _pressure_check(study, lc, l, part_names, what):
    errors = []
    if l.gradient is not None:
        try:
            ok = len(l.gradient) == 3 and all(np.isfinite(float(v)) for v in l.gradient) and \
                (l.origin is None or (len(l.origin) == 3 and all(np.isfinite(float(v)) for v in l.origin)))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            errors.append(f"{what}: gradient and origin of the variable pressure must have 3 finite components")
    return errors, []


def _rotation_check(study, lc, l, part_names, what):
    errors, warnings = [], []
    try:
        ax = np.asarray(l.axis, float)
        ok = ax.shape == (3,) and np.linalg.norm(ax) > 0 and len(l.point) == 3 and \
            np.isfinite([float(l.omega), float(l.alpha), *map(float, l.point)]).all()
    except (TypeError, ValueError):
        ok = False
    if not ok:
        errors.append(f"Case '{lc.name}', rotation: axis (point, nonzero direction) and ω, α must be finite numbers")
    excl = study.excluded_parts()
    for pn in l.parts:
        if pn not in part_names:
            errors.append(f"Case '{lc.name}', rotation: part '{pn}' is not in the geometry")
        elif pn in excl:
            errors.append(f"Case '{lc.name}', rotation: part '{pn}' is not meshed (omitted / bolt)")
    for pn in (l.parts or [p for p in part_names if p not in excl]):
        try:
            if not study.material_of(pn).rho > 0:
                errors.append(f"Case '{lc.name}', rotation: material of part '{pn}' has no density ρ")
        except KeyError:
            pass                       # missing material reported by check_study
    if l.omega == 0 and l.alpha == 0:
        warnings.append(f"Case '{lc.name}': rotation has ω = 0 and α = 0 (no load)")
    return errors, warnings


# ---------------------------------------------------------------- ccx writing

def _gravity_ccx(ctx: CcxLoadContext, l):
    from .ccx.inp import _fmt
    L = _loads()
    g = L.global_vector(ctx.study, l)
    gn = np.linalg.norm(g)
    if gn > 0 and not ctx.body_forces:
        # U1 elements (bolts, frames) cannot take body forces even outside their ELSET → nodal forces
        Fg = L.gravity_nodal_forces(ctx.study, ctx.mesh, g)
        ctx.add_nodal(Fg, np.abs(Fg).max(axis=1) > 0)
    elif gn > 0:
        d = g / gn
        ctx.dload.append(f"EALL, GRAV, {_fmt(gn)}, {_fmt(d[0])}, {_fmt(d[1])}, {_fmt(d[2])}")


def _remote_ccx(ctx: CcxLoadContext, l):
    """Written by write_inp itself: reference node (*COUPLING / *RIGID BODY) + *CLOAD on REF/ROT (remote_keys)."""


def _pressure_sets(l):
    # constant pressure is written as *CLOAD too (our integration), the SURFACE stays in the deck (byte-identical decks)
    return False, not l.variable


def _remote_sets(l):
    return True, True


# ---------------------------------------------------------------- the table

LOAD_HANDLERS: dict[str, LoadKind] = {h.kind: h for h in (
    # force: faces by area, edges by length, points F/n; cylindrical csys = density in (r, θ, z); *CLOAD without row mask
    LoadKind("force", S.Force, "solid", entities=("faces", "edges", "points"), csys="cylindrical",
             nodal=_force_nodal, resultant=_force_resultant, scale=_scale_vector, describe=_describe_force,
             entity_check=_force_entity_check, ccx_row_mask=False),
    # pressure: constant and variable as consistent *CLOAD (ccx *DLOAD P is inaccurate on tri6)
    LoadKind("pressure", S.Pressure, "solid", entities=("faces",), nodal=_pressure_nodal,
             resultant=_pressure_resultant, scale=_scale_pressure, describe=_describe_pressure,
             check=_pressure_check, ccx_sets=_pressure_sets),
    # gravity: ccx *DLOAD GRAV; with U1 elements (bolts, frames) consistent nodal forces
    LoadKind("gravity", S.Gravity, "solid", csys="cartesian", nodal=_gravity_nodal, resultant=_gravity_resultant,
             scale=_scale_vector, describe=_describe_gravity, ccx_cards=_gravity_ccx),
    # remote: ccx reference node + coupling (inp.write_inp); own assembly: distributing forces (RBE3) on the nodes, rigid on the extra DOFs of the rigid body
    LoadKind("remote", S.RemoteLoad, "solid", entities=("faces",), csys="cartesian", remote=_remote_point,
             resultant=_remote_resultant, scale=_scale_force_moment, describe=_describe_remote,
             ccx_cards=_remote_ccx, ccx_sets=_remote_sets, ngs_nodal=_remote_distributing),
    # bearing: cosine pressure on the half of the cylinder + axial shear (cylinder.bearing_forces)
    LoadKind("bearing", S.BearingLoad, "solid", entities=("faces",), csys="cartesian", nodal=_bearing_nodal,
             resultant=_vector_resultant, scale=_scale_vector, describe=_describe_bearing),
    # thermal: temperature field (thermal/field; *TEMPERATURE in ccx/inp, dT.npy for NGSolve); not scaled
    LoadKind("thermal", S.ThermalLoad, "thermal", scale=_scale_none, describe=_describe_kind),
    # rotation: centrifugal ρω²r⊥ and −ρα×r, consistent *CLOAD (not ccx CENTRIF)
    LoadKind("rotation", S.Rotation, "solid", nodal=_rotation_nodal, scale=_scale_rotation,
             describe=_describe_rotation, check=_rotation_check),
    # frame loads: beam solver (beams/fem.py, beams/timoshenko.py)
    LoadKind("frame_node", S.FrameNodeLoad, "frame", scale=_scale_force_moment, describe=_describe_kind,
             backends=frozenset()),
    LoadKind("frame_member", S.FrameMemberLoad, "frame", scale=_scale_frame_member, describe=_describe_kind,
             backends=frozenset()),
    # inertia: d'Alembert forces of the rigid motion (motion → FEM)
    LoadKind("inertia", S.Inertia, "solid", nodal=_inertia_nodal, scale=_scale_inertia, describe=_describe_inertia),
    # frame temperature: own Timoshenko solver only
    LoadKind("frame_temperature", S.FrameTemperature, "frame", scale=_scale_frame_temperature,
             describe=_describe_kind, backends=frozenset()),
)}


def load_handler(l) -> LoadKind | None:
    """Handler of the load object (by its `kind`); None = unknown kind (ignored by the core as before)."""
    return LOAD_HANDLERS.get(getattr(l, "kind", None))


# ======================================================================= constraint handlers

@dataclass(frozen=True)
class ConstraintKind:
    """Behaviour of one constraint kind (writing of ccx cards stays in ccx/inp – every kind has its own mechanism).

    ngs_unsupported – reason why the NGSolve backends cannot compute it (None = supported: masks / prescribed values)
    describe(c, st, num) -> one sentence (loads_text.constraint_text)
    """
    kind: str
    cls: type
    entities: tuple = ("faces",)
    ngs_unsupported: str | None = None
    describe: Callable | None = None


def _describe_fixed(c, st, num):
    return "fixed support (ux = uy = uz = 0)"


def _describe_displacement(c, st, num):
    parts = [f"{n} = {num(v)} mm" for n, v in zip(("ux", "uy", "uz"), (c.ux, c.uy, c.uz)) if v is not None]
    return "prescribed displacement: " + (", ".join(parts) or "no component")


def _describe_cylindrical(c, st, num):
    on = [n for n, v in (("radially", c.radial), ("tangentially", c.tangential), ("axially", c.axial)) if v]
    return "cylindrical support: " + (", ".join(on) or "no component")


def _describe_remote_support(c, st, num):
    names = ("ux", "uy", "uz", "rx", "ry", "rz")
    on = [n for n, v in zip(names, c.dofs) if v]
    cpl = "rigid" if c.coupling == "rigid" else "distributing"
    return f"remote support ({cpl}) at point {_lt().vec_text(c.point, num)} mm: " + (", ".join(on) or "nothing")


def _describe_elastic(c, st, num):
    unit = "N/mm³" if c.per_area else "N/mm (total)"
    d = "normal to face" if c.direction == "normal" else "in all directions"
    return f"elastic support k = {num(c.stiffness)} {unit}, {d}"


CONSTRAINT_HANDLERS: dict[str, ConstraintKind] = {h.kind: h for h in (
    ConstraintKind("fixed", S.Fixed, describe=_describe_fixed),
    ConstraintKind("displacement", S.Displacement, describe=_describe_displacement),
    # cylindrical: ccx *EQUATION (cylinder.py); own assembly: the same equations eliminated (linear_constraints.py)
    ConstraintKind("cylindrical", S.CylindricalSupport, describe=_describe_cylindrical),
    # remote support: ccx *RIGID BODY / *EQUATION (remote.py); own assembly: extra nodes / the equations eliminated, direct solver only
    ConstraintKind("remote_support", S.RemoteSupport, describe=_describe_remote_support),
    # elastic: ccx SPRING1 (springs.py); own assembly: the same springs on the diagonal blocks
    ConstraintKind("elastic", S.ElasticSupport, describe=_describe_elastic),
)}


def constraint_handler(c) -> ConstraintKind | None:
    return CONSTRAINT_HANDLERS.get(getattr(c, "kind", None))


def missing_handlers() -> list[tuple[str, str, str]]:
    """Kinds of study.LOAD_KINDS / CONSTRAINT_KINDS without a complete handler: [(group, kind, problem)], empty = consistent.
    group "load" | "constraint"; problem no_handler (none / another class), no_scale / no_describe (missing callable),
    no_forces (a solid load without nodal / remote forces), not_in_study (a handler of a kind that study does not know).
    """
    out = []
    for k, cls in S.LOAD_KINDS.items():
        h = LOAD_HANDLERS.get(k)
        if h is None or h.cls is not cls:
            out.append(("load", k, "no_handler"))
            continue
        out += [("load", k, f) for f, v in (("no_scale", h.scale), ("no_describe", h.describe)) if v is None]
        if h.domain == "solid" and h.nodal is None and h.remote is None:
            out.append(("load", k, "no_forces"))
    for k, cls in S.CONSTRAINT_KINDS.items():
        h = CONSTRAINT_HANDLERS.get(k)
        if h is None or h.cls is not cls:
            out.append(("constraint", k, "no_handler"))
        elif h.describe is None:
            out.append(("constraint", k, "no_describe"))
    out += [("load", k, "not_in_study") for k in LOAD_HANDLERS if k not in S.LOAD_KINDS]
    out += [("constraint", k, "not_in_study") for k in CONSTRAINT_HANDLERS if k not in S.CONSTRAINT_KINDS]
    return out

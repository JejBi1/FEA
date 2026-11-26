"""The constraints of the study for the own assembly of the linear path (`fea_core.assembly`, direct solver): the elastic support, the cylindrical support,
the rigid and distributing remote constraints, the rigid remote load and *TIE as a `Reduction` (dependent DOFs u_D = A·u + c, extra nodes, springs).

The constraints are ELIMINATED, not added as equations: the reduced matrix TᵀKT stays symmetric positive definite (no Lagrange multipliers – rslab does not
factor those with its defaults). The same constraint sets as for ccx, so the results agree with the ccx path to the digits that the .frd of ccx holds:
  elastic support        the springs of `springs.table` (the same quantized stiffnesses as written for ccx) as 3×3 blocks on the diagonal blocks
                         (k·I in all directions, k·n·nᵀ normal – the energy ½·k·(n·u)² of the SPRING1 with the equation u_g = n·u);
  cylindrical support    the equations of `cylinder.support_equations` (u·e = 0, RREF with the dependent DOF outside the *BOUNDARY DOFs);
  rigid remote support / load   `*RIGID BODY`: every node of the face u = u_ref + θ × (x − p) with the reference point p as two extra nodes (the
                         displacement and the rotation); the load of a rigid remote load acts on the extra DOFs, the reaction of a support is
                         Σ(RF − F) over the nodes of the face (like the distributing one, `solve.remote_reactions`);
  distributing remote support   the dense equations of `remote.distributing_equations` (RBE3), one dependent DOF each;
  *TIE                   every node of the slave face from the quadratic master face at the point closest to it (`ties.tie_weights`).
  bolts                  a rigid body per bearing face and an exact Timoshenko element between the faces (`joints.own`): the beam elements are added to the matrix
                         of the extended system (`Reduction.elem_dofs` / `elem_k`), the thermal elongation of a bolt is a load on the extra DOFs, the preload
                         the condition N = F_V on unit states (`Plan.unit_loads`, `joints.own.preload_gaps`).
Point mass: the statics (weight, inertia) = the forces m·a at its points – on the extra DOFs of a rigid body, or distributed to the face (RBE3). Not here: contact
(nonlinear) – ccx. Welds need nothing here (the contact of the welded parts is opened in the mesh).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assembly.reduction import Reduction, ReductionError
from .joints.own import BoltChain
from .mesh import Mesh
from .study import (CylindricalSupport, RemoteLoad, RemoteSupport, Study)

# Σ (terms of an equation)² of the distributing remote constraints: the elimination of a dense RBE3 row set makes a dense block of the face DOFs in the reduced
# matrix (the assembly writes ≈ 2·Σ entries, the factor of the direct solver is dense on the face). Measured (a 100 × 100 × 6 block, face z = 0 with a
# distributing support, 1 state, 4 threads, this machine; ccx on the same mesh):
#   face nodes   Σ terms²   own total   assembly   assembly peak   factor        ccx
#       529      1.8e6       5.9 s       0.3 s      0.18 GB         3 M nnz      1.8 s
#     1 973      2.6e7      10.4 s       3.3 s      1.0 GB         18 M nnz     31.5 s
#     5 833      2.3e8      77.9 s      37.6 s      8.0 GB        166 M nnz     refuses (> 4e7)
# → the assembly peak ≈ 3.5–4·10⁻⁸ GB per unit of Σ terms² (a scipy product of the dense block). Up to MAX_FILL_OWN the own assembly is faster than ccx and
# goes beyond what ccx accepts (`remote.MAX_DIST_FILL` = 4·10⁷); the limit is lowered to what the free RAM allows (`fill_limit`). Above it: ccx / a clear error.
MAX_FILL_OWN = 1.0e8
FILL_GB = 4.0e-8                                  # GB of the assembly peak per unit of Σ terms² (the measurement above)
FILL_SHARE_OF_FREE = 0.5                          # the assembly may take this share of the free RAM


def fill_limit(free_gb: float | None = None) -> float:
    """The largest Σ terms² of the distributing supports that the own assembly takes now: `MAX_FILL_OWN`, lowered so that its peak memory
    (`FILL_GB` per unit) stays within the share of the free RAM."""
    if free_gb is None:
        import psutil
        free_gb = psutil.virtual_memory().available / 2**30
    return min(MAX_FILL_OWN, FILL_SHARE_OF_FREE * free_gb / FILL_GB)


@dataclass
class BoltPlan:
    """One bolt on the extra nodes: the chain of its beam elements, the pair of extra nodes of every bearing face, the nodes of all its faces."""
    chain: BoltChain
    pairs: list[int]                                  # the index of the pair of extra nodes of every position of the chain
    nodes: np.ndarray                                 # the nodes of all bearing faces (the mean temperature of the thermal elongation)


@dataclass
class Plan:
    """What `build` found: the reduction + the bookkeeping of the extra nodes for the loads, the reactions and the results."""
    red: Reduction
    bolts: list[BoltPlan] = field(default_factory=list)
    support_pairs: dict[str, int] = field(default_factory=dict)       # selection of a rigid remote support → the index of its pair of extra nodes
    load_pairs: dict[str, tuple[int, np.ndarray]] = field(default_factory=dict)       # selection of a rigid remote load → (pair, the point of the body)
    support_nodes: dict[str, np.ndarray] = field(default_factory=dict)  # rigid remote support → the nodes of its face
    mass_pairs: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)    # a point mass with a rigid coupling (index in `mass_points`) → (pair, its centre)
    mass_ops: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)    # a distributing one → (nodes of its face, RBE3 operator A (6, n, 3))
    mesh: Mesh | None = None
    warnings: list[str] = field(default_factory=list)

    def pair_dofs(self, pair: int) -> tuple[int, int]:
        """The first DOF of the displacement node and of the rotation node of a pair of extra nodes."""
        base = 3 * (self.red.n_mesh + 2 * pair)
        return base, base + 3

    def extra_loads(self, study: Study, lc) -> np.ndarray:
        """The loads of the rigid remote loads of a state on the extra DOFs (3·extra nodes,): the force at the point of the body and the moment about it
        (the moment of another point of the same face is moved: M + (p − p_body) × F, the same as the one rigid body of ccx)."""
        from .loads import remote_vectors
        out = np.zeros(3 * self.red.n_extra)
        for l in lc.loads:
            if isinstance(l, RemoteLoad) and l.coupling == "rigid" and l.selection in self.load_pairs:
                pair, p_body = self.load_pairs[l.selection]
                f, m = remote_vectors(study, l)
                d0, d1 = self.pair_dofs(pair)
                d0, d1 = d0 - 3 * self.red.n_mesh, d1 - 3 * self.red.n_mesh
                out[d0:d0 + 3] += f
                out[d1:d1 + 3] += m + np.cross(np.asarray(l.point, float) - p_body, f)
        if self.mass_pairs:                       # the weight / inertia of a point mass with a rigid coupling: the forces of its points (the centre and the points of the
            from .dynamics.mass import mass_points, static_forces      # rotary inertia) act on the rigid body: the force and the moment about the centre
            mp = mass_points(study, self.mesh)
            for i, k, f in static_forces(study, self.mesh, lc)[1]:
                if i in self.mass_pairs:
                    pair, centre = self.mass_pairs[i]
                    d0, d1 = (d - 3 * self.red.n_mesh for d in self.pair_dofs(pair))
                    out[d0:d0 + 3] += f
                    out[d1:d1 + 3] += np.cross(mp[i]["nodes"][k][0] - centre, f)
        return out

    # ---- bolts: initial elongation of the cut element (the thermal elongation, the unit states of the preload)
    def cut_bolts(self) -> list[BoltPlan]:
        """The bolts with a cut of the beam (without a gap between the clamped parts): they take the preload and the thermal elongation."""
        return [bp for bp in self.bolts if bp.chain.cut is not None]

    def elongation_load(self, bp: BoltPlan, amount: float) -> np.ndarray:
        """The load on the extra DOFs (3·extra nodes,) of an initial elongation `amount` [mm] of the cut element of a bolt: the force k·amount along the axis
        on the two ends (the right-hand side k·g·B of N = k·(B·u − g))."""
        ch = bp.chain
        out = np.zeros(3 * self.red.n_extra)
        a, b = 6 * bp.pairs[ch.cut], 6 * bp.pairs[ch.cut + 1]
        f = ch.k_axial[ch.cut] * amount * ch.e
        out[a:a + 3] -= f
        out[b:b + 3] += f
        return out

    def thermal_elongation(self, T: np.ndarray | None, t_ref: float) -> np.ndarray:
        """The load on the extra DOFs of the thermal elongation g = α·ΔT·L of the bolts (ΔT = the mean of the nodes of their bearing faces, as in ccx). Exact,
        not a penalty spring."""
        out = np.zeros(3 * self.red.n_extra)
        if T is None:
            return out
        for bp in self.cut_bolts():
            dT = float(np.asarray(T)[bp.nodes].mean()) - t_ref
            if dT:
                out += self.elongation_load(bp, bp.chain.bolt.alpha * dT * bp.chain.bolt.length)
        return out

    def unit_loads(self) -> np.ndarray:
        """(3·extra nodes, bolts with a cut): the unit states of the preload – elongation of bolt j by 1 mm."""
        cols = [self.elongation_load(bp, 1.0) for bp in self.cut_bolts()]
        return np.column_stack(cols) if cols else np.zeros((3 * self.red.n_extra, 0))

    def stretch(self, X: np.ndarray) -> np.ndarray:
        """(bolts with a cut, columns): the relative displacement Δu = (u_b − u_a)·e of the ends of the cut element for the columns of the solution X
        (all DOFs of the extended system)."""
        base = 3 * self.red.n_mesh
        X = np.asarray(X, float).reshape(self.red.n_dof, -1)
        out = []
        for bp in self.cut_bolts():
            a, b = base + 6 * bp.pairs[bp.chain.cut], base + 6 * bp.pairs[bp.chain.cut + 1]
            out.append(bp.chain.e @ (X[b:b + 3] - X[a:a + 3]))
        return np.array(out).reshape(len(out), -1)

    def bolt_matrix(self, X_units: np.ndarray) -> np.ndarray:
        """C of `joints.own.preload_gaps`: the force N_i in bolt i per unit elongation of bolt j, from the solution of the unit states."""
        k = np.array([bp.chain.k_axial[bp.chain.cut] for bp in self.cut_bolts()])
        return k[:, None] * (self.stretch(X_units) - np.eye(len(k)))

    def mass_nodal(self, study: Study, lc) -> np.ndarray | None:
        """The nodal forces (N, 3) of the weight / inertia of the point masses with a DISTRIBUTING coupling (the centre is a dependent point of RBE3: the force
        goes to the nodes of the face with the weights of the equation), None = there is none."""
        if not self.mass_ops:
            return None
        from .dynamics.mass import static_forces
        out = np.zeros((len(self.mesh.nodes), 3))
        for i, k, f in static_forces(study, self.mesh, lc)[1]:
            if i in self.mass_ops:
                idx, A = self.mass_ops[i]
                out[idx] += np.einsum("d,dij->ij", f, A[:3])
        return out

    def motion(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """{selection of a rigid remote support: (6,) displacement and rotation of its point} from one column of the solution of the extended system."""
        out = {}
        for sel, pair in self.support_pairs.items():
            d0, d1 = self.pair_dofs(pair)
            out[sel] = np.concatenate([x[d0:d0 + 3], x[d1:d1 + 3]])
        return out


def _spring_blocks(study: Study, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(nodes, blocks (k, 3, 3)) of the elastic supports (the nodes are unique, the blocks of several supports on one node add up)."""
    from .springs import table
    acc: dict[int, np.ndarray] = {}
    for t in table(study, mesh):
        for j, n in enumerate(t["idx"]):
            k = float(t["k"][j])
            if k <= 0:
                continue
            if t["direction"] == "normal":
                nv = np.asarray(t["normal"][j], float)
                blk = k * np.outer(nv, nv)
            else:
                blk = k * np.eye(3)
            acc[int(n)] = acc.get(int(n), 0.0) + blk
    if not acc:
        return np.zeros(0, np.int64), np.zeros((0, 3, 3))
    nodes = np.array(sorted(acc), np.int64)
    return nodes, np.stack([acc[int(n)] for n in nodes])


class _Rows:
    """The equations u_dep = Σ coef·u_master + offset, collected."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, list[tuple[int, float]], float]] = []

    def add(self, dep: int, terms: list[tuple[int, float]], c: float = 0.0) -> None:
        self.rows.append((int(dep), [(int(k), float(v)) for k, v in terms], float(c)))


def unsupported_reasons(study: Study, solver: str = "direct") -> list[str]:
    """Why the own assembly cannot do the constraints of the study for the given solver ("direct" | "hypre"): [] = it can. The solvers differ: the
    iterative solver (hypre) cannot do constraints that couple several nodes (a dense row / the extra nodes are not what its AMG expects) – only the node-local
    ones (springs, cylindrical support, orthotropy); the direct solver does everything of this module."""
    out = []
    if solver != "direct":
        for c in study.constraints:
            if isinstance(c, RemoteSupport):
                out.append(f"remote support ({c.coupling}) – direct solver only")
        for lc in study.load_cases:
            for l in lc.loads:
                if isinstance(l, RemoteLoad) and l.coupling == "rigid":
                    out.append(f"rigid remote load ({lc.name}) – direct solver only")
        if study.mesh.separate:
            out.append("parts joined by *TIE – direct solver only")
        if any(getattr(m, "kind", "") == "point_mass" and m.coupling == "rigid" for m in getattr(study, "masses", [])):
            out.append("point mass with a rigid coupling – direct solver only")
        if study.active_bolts():
            out.append("bolts (rigid bearing faces, beam elements) – direct solver only")
    return list(dict.fromkeys(out))


def build(study: Study, mesh: Mesh, solver: str = "direct") -> Plan | None:
    """The `Plan` of the constraints of the study on the mesh, None = nothing to eliminate (plain supports only).
    Raises `ReductionError` for a combination of constraints that the own assembly cannot do (the caller continues in ccx)."""
    from .remote import check_overlaps, distributing_equations, MAX_DIST_FILL  # noqa: F401
    nm = len(mesh.nodes)
    rows = _Rows()
    labels: dict[str, int] = {}
    extra_xyz: list[np.ndarray] = []
    fixed_extra: list[bool] = []
    plan = Plan(Reduction(nm), mesh=mesh)
    spring_nodes, spring_blocks = _spring_blocks(study, mesh)
    if len(spring_nodes):
        labels["springs"] = int(len(spring_nodes))

    # --- cylindrical support: u·e = 0 at the nodes
    if any(isinstance(c, CylindricalSupport) for c in study.constraints):
        from .cylinder import support_equations
        eqs = support_equations(study, mesh)
        for n, terms in eqs:
            dep = 3 * n + terms[0][0]
            rows.add(dep, [(3 * n + d, -c) for d, c in terms[1:]])
        labels["cylindrical"] = len(eqs)

    # --- remote constraints
    sups = [c for c in study.constraints if isinstance(c, RemoteSupport)]
    loads = [(lc, l) for lc in study.load_cases for l in lc.loads if isinstance(l, RemoteLoad) and l.coupling == "rigid"]
    if sups:
        check_overlaps(study, mesh)

    def rigid_body(idx: np.ndarray, point, dofs, name: str, label: str = "rigid") -> int:
        """A rigid face: the nodes u = u_ref + θ × (x − p) from a pair of extra nodes. → the index of the pair. label = the counter of the protocol."""
        pair = len(extra_xyz) // 2
        p = np.asarray(point, float)
        extra_xyz.extend([p, p])
        fixed_extra.extend([bool(v) for v in dofs])
        e0, e1 = 3 * (nm + 2 * pair), 3 * (nm + 2 * pair + 1)
        r = mesh.nodes[idx] - p
        for i, n in enumerate(idx):
            rx, ry, rz = r[i]
            rows.add(3 * n + 0, [(e0 + 0, 1.0), (e1 + 1, rz), (e1 + 2, -ry)])
            rows.add(3 * n + 1, [(e0 + 1, 1.0), (e1 + 2, rx), (e1 + 0, -rz)])
            rows.add(3 * n + 2, [(e0 + 2, 1.0), (e1 + 0, ry), (e1 + 1, -rx)])
        labels[label] = labels.get(label, 0) + 1
        return pair

    for c in sups:
        if c.coupling == "rigid":
            idx = mesh.surface_nodes(c.selection)
            dofs = list(c.dofs) if len(c.dofs) == 6 else [True] * 6
            plan.support_pairs[c.selection] = rigid_body(idx, c.point, dofs, c.selection)
            plan.support_nodes[c.selection] = idx
    for lc, l in loads:
        if l.selection in plan.load_pairs:
            continue                                  # one rigid body per selection (the point of the first load)
        idx = mesh.surface_nodes(l.selection)
        pair = rigid_body(idx, l.point, [False] * 6, l.selection)
        plan.load_pairs[l.selection] = (pair, np.asarray(l.point, float))

    if any(getattr(m, "kind", "") == "point_mass" for m in getattr(study, "masses", [])):     # point masses: statics = forces m·a at their points
        from .dynamics.mass import mass_points
        from .remote import rbe3
        for i, info in enumerate(mass_points(study, mesh)):
            it = info["item"]
            if it.coupling == "rigid":
                plan.mass_pairs[i] = (rigid_body(mesh.surface_nodes(it.selection), info["point"], [False] * 6, it.name or it.selection), np.asarray(info["point"], float))
            else:
                op = rbe3(mesh, it.selection, info["point"])
                plan.mass_ops[i] = (op["idx"], op["A"])
        labels["point_mass"] = len(plan.mass_pairs) + len(plan.mass_ops)

    if any(c.coupling != "rigid" and any(c.dofs) for c in sups):
        from .cylinder import boundary_dofs
        eqs = distributing_equations(study, mesh, boundary_dofs(study, mesh))
        fill = float(sum(len(t) ** 2 for t in eqs))
        limit = fill_limit()
        if fill > limit:
            names = ", ".join(f"'{c.selection}'" for c in sups if c.coupling != "rigid" and any(c.dofs))
            raise ReductionError(f"The distributing remote support {names} is a dense block of Σ terms² = {fill:.2g} entries "
                                 f"(the own assembly takes up to {limit:.2g} with the free memory)")
        for terms in eqs:
            (n0, d0, _), rest = terms[0], terms[1:]
            rows.add(3 * n0 + d0, [(3 * n + d, -c) for n, d, c in rest])
        labels["distributing"] = len(eqs)

    # --- *TIE of separately meshed parts
    ties = mesh.report.get("ties", []) if study.mesh.separate else []
    if ties:
        from .ties import mesh_ties
        n_tied = 0
        for t, w in zip(ties, mesh_ties(mesh)):       # computed once per mesh: the reactions use the same weights (`solve.make_step`)
            for i, s in enumerate(w["nodes"]):
                for k in range(3):
                    rows.add(3 * int(s) + k, [(3 * int(m) + k, float(wi)) for m, wi in zip(w["masters"][i], w["weights"][i]) if abs(wi) > 1e-14])
            n_tied += len(w["nodes"])
            if len(w["untied"]):                      # as CalculiX (gentiedmpc): no master face under the node → the node stays free
                plan.warnings.append(f"*TIE of part '{t['part']}': {len(w['untied'])} nodes of the slave face have no master face within the tolerance "
                                     f"{t['tolerance_mm']:.3g} mm and are not tied (as in CalculiX)")
        labels["tie"] = n_tied

    # --- bolts: a rigid body per bearing face and the beam elements between the faces
    elem_dofs: list[list[int]] = []
    elem_k: list[np.ndarray] = []
    if study.active_bolts():
        from .joints.fem import bolt_models
        from .joints.own import BoltChainError, chain_of, element_matrices
        owner = np.full(nm, -1, np.int64)                    # node → the index of the bolt whose bearing face holds it (a node is held by ONE rigid body)
        names: list[str] = []
        for bm in bolt_models(study, mesh):
            try:
                ch = chain_of(bm)
            except BoltChainError as e:
                raise ReductionError(str(e)) from None
            bi = len(names)
            names.append(bm.name)
            pairs, all_nodes = [], []
            for p, keys in zip(ch.positions, ch.keys):
                idx = np.unique(np.concatenate([mesh.surface_nodes(k) for k in keys]))
                for prev in all_nodes:
                    if np.intersect1d(prev, idx).size:       # two rigid bodies on one node: a fitted bolt through parts that touch (the hole walls meet in a circle)
                        raise ReductionError(f"{bm.name}: two bearing faces of the bolt share nodes (the hole walls of parts that touch meet in one circle – "
                                             "a fitted bolt); CalculiX ties the two rigid faces at such a node, the own assembly does not")
                other = np.unique(owner[idx][(owner[idx] >= 0) & (owner[idx] != bi)])
                if other.size:                               # the imprints / hole walls of two bolts overlap (bolts closer than their washers): the same node in two rigid bodies
                    raise ReductionError(f"{bm.name}: a bearing face of the bolt shares nodes with the bearing face of the bolt "
                                         + ", ".join(f"'{names[int(o)]}'" for o in other)
                                         + " (the faces of neighbouring bolts overlap); CalculiX ties the rigid faces of both bolts at such a node, the own assembly does not")
                owner[idx] = bi
                all_nodes.append(idx)
                pairs.append(rigid_body(idx, bm.points[p], [False] * 6, bm.name, "bolt_faces"))
            Ke = element_matrices(ch)
            for i in range(ch.n_elements):
                d = []
                for q in (pairs[i], pairs[i + 1]):          # (displacement node, rotation node) of the two ends
                    e0 = 3 * (nm + 2 * q)
                    d += list(range(e0, e0 + 6))
                elem_dofs.append(d)
                elem_k.append(Ke[i])
            plan.bolts.append(BoltPlan(ch, pairs, np.unique(np.concatenate(all_nodes))))
        labels["bolts"] = len(plan.bolts)

    if not rows.rows and not len(spring_nodes) and not extra_xyz and not plan.mass_ops:
        return None
    plan.red = Reduction.from_rows(nm, rows.rows, extra_xyz=np.array(extra_xyz, float).reshape(-1, 3), fixed_extra=np.array(fixed_extra, bool),
                                   g_extra=np.zeros(3 * len(extra_xyz)), spring_nodes=spring_nodes, spring_blocks=spring_blocks,
                                   elem_dofs=np.array(elem_dofs, np.int64).reshape(-1, 12), elem_k=np.array(elem_k, float).reshape(-1, 12, 12), labels=labels)
    return plan


# ------------------------------------------------------------------ the check before the run (what the own assembly would refuse) and the named errors
def _fixed_dofs(study: Study, mesh: Mesh, plan: Plan) -> np.ndarray:
    """The constrained DOFs of the extended system (the nodes of the mesh by Fixed / Displacement, then the constrained components of the extra nodes)."""
    from .study import Displacement, Fixed
    n = len(mesh.nodes)
    fixed = np.zeros((n, 3), bool)
    for c in study.constraints:
        if isinstance(c, Fixed):
            fixed[mesh.surface_nodes(c.selection)] = True
        elif isinstance(c, Displacement):
            idx = mesh.surface_nodes(c.selection)
            for d, v in enumerate((c.ux, c.uy, c.uz)):
                if v is not None:
                    fixed[idx, d] = True
    return np.concatenate([fixed.ravel(), plan.red.fixed_extra])


def _item(what: str, name: str) -> str:
    """"<what>: '<name>'" – the kind is a text of the catalogue of its own, the name is joined WITHOUT a template (see `_describe`)."""
    return what + ": '" + name + "'"


def _describe(c) -> str:
    """One constraint for a message: "<kind>: '<selection>'". The kind is a text of the catalogue of its own and the name is joined to it WITHOUT a template: a list
    of such items ("rigid remote support: 'a', fixed support: 'b'") is translated item by item (a template ending with a hole would swallow the rest of the list)."""
    sel = c.selection
    kind = getattr(c, "kind", "")
    if isinstance(c, RemoteSupport):
        what = "rigid remote support" if c.coupling == "rigid" else "distributing remote support"
    elif kind == "fixed":
        what = "fixed support"
    elif kind == "displacement":
        what = "prescribed displacement"
    elif kind == "cylindrical":
        what = "cylindrical support"
    elif kind == "elastic":
        what = "elastic support"
    else:
        what = "constraint"
    return _item(what, sel)


def holders(study: Study, mesh: Mesh, node: int) -> list[str]:
    """The constraints whose face holds the node (0-based): the names of the selections for an error message ("rigid remote support: 'x0'", …)."""
    out = []

    def has(sel: str) -> bool:
        if sel not in mesh.surfaces:
            return False
        nodes = mesh.surface_nodes(sel)
        i = np.searchsorted(nodes, node)
        return bool(i < len(nodes) and nodes[i] == node)
    for c in study.constraints:
        if has(c.selection):
            out.append(_describe(c))
    seen = set()
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, RemoteLoad) and l.coupling == "rigid" and l.selection not in seen and has(l.selection):
                seen.add(l.selection)
                out.append(_item("rigid remote load", l.selection))
    for m in getattr(study, "masses", []):
        if getattr(m, "kind", "") == "point_mass" and m.coupling == "rigid" and has(m.selection):
            out.append(_item("point mass with a rigid coupling", m.selection))
    if study.active_bolts():
        from .joints.fem import coupling_keys
        for i, b in enumerate(study.active_bolts(), 1):
            if any(has(key) for key, _ in coupling_keys(b, i, mesh)):
                out.append(_item("bolt bearing face", b.name))
    for t in mesh.report.get("ties", []) if study.mesh.separate else []:
        if has(t["slave"]):
            out.append(_item("TIE slave face", t["part"]))
        if has(t["master"]):
            out.append(_item("TIE master face", t["part"]))
    return list(dict.fromkeys(out))


def explain(study: Study, mesh: Mesh, err: ReductionError) -> str:
    """The message of a `ReductionError` with the constraints that meet at the node of the conflict, one per line ("rigid remote support: 'a'", "fixed support: 'b'")."""
    dof = getattr(err, "dof", None)
    if dof is None or dof >= 3 * len(mesh.nodes):
        return str(err)
    names = holders(study, mesh, dof // 3)
    # one constraint per line: the Czech text of a list is made line by line (the nesting of the message is deeper than what the engine splits at commas)
    return f"{err} – the faces meet at this node:\n" + "\n".join(names) if names else str(err)


def prepare(study: Study, mesh: Mesh, solver: str = "direct") -> Plan | None:
    """`build` and the check of the constraint equations against the constrained DOFs, BEFORE any file is written. Raises `ReductionError` whose message names the
    constraints (selections) that conflict, the limit, … – the same errors that the assembly worker would give later as a bare node number."""
    from .remote import RemoteError
    try:
        plan = build(study, mesh, solver)
    except RemoteError as e:
        raise ReductionError(str(e)) from None
    if plan is None or plan.red.empty:
        return plan
    try:
        plan.red.validate(_fixed_dofs(study, mesh, plan))
    except ReductionError as e:
        raise ReductionError(explain(study, mesh, e), e.dof) from None
    return plan


def check(study: Study, mesh: Mesh, solver: str = "direct") -> list[str]:
    """Why the own assembly would refuse the constraints of this study ([] = it takes them): a face that two constraint equations / a constraint and a fixed
    face share, a *TIE master that is dependent, a dense distributing support above `fill_limit()`. `solve.choose_backend_reason` asks it BEFORE choosing the
    direct / iterative solver, so that the study goes to CalculiX with the reason instead of failing after the assembly started."""
    if not (any(isinstance(c, (RemoteSupport, CylindricalSupport)) for c in study.constraints)
            or any(isinstance(l, RemoteLoad) and l.coupling == "rigid" for lc in study.load_cases for l in lc.loads)
            or any(getattr(m, "kind", "") == "point_mass" for m in getattr(study, "masses", []))
            or study.active_bolts()
            or (study.mesh.separate and mesh.report.get("ties"))):
        return []                                    # springs, fixed supports, prescribed displacements and orthotropy cannot conflict
    try:
        prepare(study, mesh, solver)
    except ReductionError as e:
        return [str(e)]
    except Exception as e:  # noqa: BLE001 – the check must not break the choice of the solver: a study that cannot be built goes to ccx, which gives its own message
        return [str(e) or type(e).__name__]
    return []

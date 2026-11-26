"""The mass of the own assembly: what `fea_core.assembly.system.assemble` needs to build the mass matrix M next to the stiffness matrix K (no numba here – the
main program builds this from the study, the assembly worker reads it).

M is the consistent mass of the bodies (14-point rule, density per element – 3D printing included, `printing.element_density`) plus the added mass of the
study (`dynamics.mass`, the same items that are written as *MASS for ccx):
  surface mass      a lumped mass m·I₃ at the nodes of the face (weights ∫N dA)
  point mass, rigid coupling       the mass m and the rotary inertia J at the reference point of a rigid body: the pair of extra nodes of `linear_constraints`
                    (displacement node: m·I₃, rotation node: J), the face follows them (u = u_ref + θ × (x − p) – eliminated like the other constraints). The
                    centre of mass is the reference point, so there is no coupling between the two nodes – exactly what ccx gets from the auxiliary points ±a.
  point mass, distributing coupling   the reference displacement u_ref = A₃·u of the face nodes (RBE3, weights ∫N dA): ½·m·|A₃ u|², a dense block m·A₃ᵀA₃ over the
                    face nodes (no rotary inertia – the same as in ccx, where the distributing constraint does not transfer it)
Everything is in the mesh units: t, mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FILE = "mass.npz"
MAX_DIST_TERMS = 1.0e8          # Σ over the distributing point masses of (3·face nodes)² ·3 – the same unit as the distributing supports (`linear_constraints.fill_limit`)


@dataclass
class MassInputs:
    rho: np.ndarray                                                      # (E,) density per element [t/mm³]
    surf_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))        # lumped masses: node numbers (may repeat)
    surf_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))                   # … and their masses [t]
    rigid_node: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))        # rigid point masses: the displacement node (the rotation node follows it)
    rigid_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))                  # (k,) mass [t]
    rigid_inertia: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))       # (k, 3, 3) inertia tensor about the reference point [t·mm²]
    dist_nodes: list = field(default_factory=list)                       # distributing point masses: [(face nodes (n,), A₃ (3, n, 3))]
    dist_mass: np.ndarray = field(default_factory=lambda: np.zeros(0))                   # (k,) mass [t]
    labels: dict = field(default_factory=dict)

    @property
    def dense_terms(self) -> float:
        """Σ of the squared terms of the dense blocks of the distributing masses (the cost of their fill)."""
        return float(sum(3.0 * (3 * len(n)) ** 2 for n, _ in self.dist_nodes))

    def save(self, directory: Path | str) -> Path:
        path = Path(directory) / FILE
        z = dict(rho=self.rho, surf_nodes=self.surf_nodes, surf_mass=self.surf_mass, rigid_node=self.rigid_node, rigid_mass=self.rigid_mass,
                 rigid_inertia=self.rigid_inertia, dist_mass=self.dist_mass, n_dist=np.int64(len(self.dist_nodes)))
        for i, (idx, A) in enumerate(self.dist_nodes):
            z[f"dist_idx_{i}"] = np.asarray(idx, np.int64)
            z[f"dist_A_{i}"] = np.asarray(A, float)
        np.savez(path, **z)
        return path

    @staticmethod
    def load(directory: Path | str) -> "MassInputs | None":
        path = Path(directory) / FILE
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as z:
            n = int(z["n_dist"])
            return MassInputs(z["rho"], z["surf_nodes"], z["surf_mass"], z["rigid_node"], z["rigid_mass"], z["rigid_inertia"],
                              [(z[f"dist_idx_{i}"], z[f"dist_A_{i}"]) for i in range(n)], z["dist_mass"])


def build(study, mesh, plan=None) -> MassInputs:
    """The mass of the study on the mesh. `plan` = `linear_constraints.Plan` of the constraints (the extra nodes of the rigid point masses); None = none of them.
    Raises `ValueError` for what the own assembly cannot do (a point mass without its rigid body in the plan, the inertia tensor that is not physical)."""
    from ..dynamics.mass import inertia_matrix, mass_items, mass_points, surface_mass_nodes
    from ..dynamics.model import PointMass, SurfaceMass
    from ..printing import element_density
    rho = np.ascontiguousarray(element_density(study, mesh), np.float64)
    sn, sm = [], []
    rn, rm, ri = [], [], []
    dn, dm = [], []
    items = mass_items(study)
    for it in items:
        if isinstance(it, SurfaceMass):
            idx, mm = surface_mass_nodes(mesh, it)
            sn.append(np.asarray(idx, np.int64))
            sm.append(np.asarray(mm, float))
    if any(isinstance(it, PointMass) for it in items):
        mp = mass_points(study, mesh)
        for i, info in enumerate(mp):
            it = info["item"]
            if it.coupling == "rigid":
                if plan is None or i not in plan.mass_pairs:
                    raise ValueError("A point mass with a rigid coupling has no rigid body in the constraints of the assembly")
                pair = plan.mass_pairs[i][0]
                I = inertia_matrix(it)
                J = np.zeros((3, 3)) if I is None else 0.5 * (I + I.T)
                rn.append(plan.red.n_mesh + 2 * pair)
                rm.append(float(it.mass))
                ri.append(J)
            else:
                if plan is None or i not in plan.mass_ops:
                    raise ValueError("A point mass with a distributing coupling has no RBE3 operator in the constraints of the assembly")
                idx, A = plan.mass_ops[i]
                dn.append((np.asarray(idx, np.int64), np.ascontiguousarray(np.asarray(A, float)[:3])))
                dm.append(float(it.mass))
    out = MassInputs(rho, np.concatenate(sn) if sn else np.zeros(0, np.int64), np.concatenate(sm) if sm else np.zeros(0),
                     np.array(rn, np.int64), np.array(rm, float), np.array(ri, float).reshape(-1, 3, 3), dn, np.array(dm, float))
    out.labels = {"surface_masses": len([1 for it in items if isinstance(it, SurfaceMass)]), "rigid_point_masses": len(rn),
                  "distributing_point_masses": len(dn)}
    return out


def total_added(m: MassInputs) -> float:
    """The mass of everything that is not a body element [t]."""
    return float(m.surf_mass.sum() + m.rigid_mass.sum() + m.dist_mass.sum())

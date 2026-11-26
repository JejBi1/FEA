"""Mass properties of the FE model for the space-structures verification (F14 stage S2): total mass, centre of mass and the inertia tensor about the centre of
mass, of the solid mesh (tet4 / tet10, per-element density as the solver uses it) plus the added masses of the study (`PointMass` at its point with its own rotary
inertia, `SurfaceMass` as the nodal masses the solver writes).

The inertia tensor is built from the second moment about the origin, S = Σ ρ x xᵀ dV + Σ m_i p_i p_iᵀ, and shifted to the centre of mass by the parallel-axis
(Steiner) theorem, S_c = S − M c cᵀ, I_c = tr(S_c)·1 − S_c (+ the own rotary inertia of the point masses). Units mm, t, so the inertia is in t·mm²
(1 t·mm² = 1e-3 kg·m²). The integration uses the same 14-point rule as `fea_core.dynamics.mass` for tet10 (exact for straight edges) and a 4-point degree-2 rule
for tet4 - exact for the quadratic integrand of the second moment. No Qt and no `fea_core.mcp` import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

T_MM2_TO_KG_M2 = 1.0e-3                 # 1 t·mm² = 1000 kg · 1e-6 m²

_A4, _B4 = 0.1381966011250105, 0.5854101966249685            # 4-point rule, degree 2: weights 1/24 each on the reference tetrahedron
_TET4_PTS = np.array([[_A4, _A4, _A4], [_B4, _A4, _A4], [_A4, _B4, _A4], [_A4, _A4, _B4]])


@dataclass
class MassProperties:
    mass_t: float
    cog: np.ndarray                              # (3,) mm
    inertia_cog: np.ndarray                      # (3, 3) t·mm², about the centre of mass, global axes
    bodies_t: float = 0.0
    added_t: float = 0.0
    volume_mm3: float = 0.0
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)
    added: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def mass_kg(self) -> float:
        return self.mass_t * 1000.0

    @property
    def inertia_cog_kg_m2(self) -> np.ndarray:
        return self.inertia_cog * T_MM2_TO_KG_M2

    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """(principal moments [t·mm²] ascending, axes as columns) of the inertia tensor about the centre of mass."""
        w, v = np.linalg.eigh(0.5 * (self.inertia_cog + self.inertia_cog.T))
        return w, v

    def inertia_about(self, point: Any) -> np.ndarray:
        """Inertia tensor about another point (parallel-axis theorem): I_p = I_c + M (|d|² 1 − d dᵀ), d = c − p."""
        d = np.asarray(self.cog, float) - np.asarray(point, float)
        return self.inertia_cog + self.mass_t * (float(d @ d) * np.eye(3) - np.outer(d, d))

    def to_dict(self) -> dict[str, Any]:
        w, v = self.principal()
        return {"mass_kg": self.mass_kg, "mass_t": self.mass_t, "cog_mm": [float(x) for x in self.cog],
                "inertia_cog_kg_m2": [[float(x) for x in r] for r in self.inertia_cog_kg_m2],
                "inertia_cog_t_mm2": [[float(x) for x in r] for r in self.inertia_cog],
                "principal_moments_kg_m2": [float(x) * T_MM2_TO_KG_M2 for x in w], "principal_axes": [[float(x) for x in v[:, k]] for k in range(3)],
                "bodies_kg": self.bodies_t * 1000.0, "added_kg": self.added_t * 1000.0, "volume_mm3": self.volume_mm3,
                "parts": {k: {"mass_kg": p["mass_t"] * 1000.0, "cog_mm": p["cog_mm"], "volume_mm3": p["volume_mm3"]} for k, p in self.parts.items()},
                "added_masses": self.added, "notes": list(self.notes)}


def _quadrature(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """(shape functions at the points (q, n_nodes_per_element), weights·|J| (M, q))."""
    if mesh.tets.shape[1] == 10:
        from ..dynamics.mass import _qp_data
        N, wd = _qp_data(mesh)
        return np.asarray(N, float), np.asarray(wd, float)
    X = mesh.nodes[mesh.tets]                                       # (M, 4, 3)
    d = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=1)
    vol = np.abs(np.linalg.det(d)) / 6.0
    N = np.column_stack([1.0 - _TET4_PTS.sum(axis=1), _TET4_PTS])
    return N, np.repeat(vol[:, None] / 4.0, 4, axis=1)


_SYM = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _sym_to_matrix(s: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3))
    for k, (i, j) in enumerate(_SYM):
        m[i, j] = m[j, i] = s[k]
    return m


def _accumulate_element_moments(mesh: Any, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per element: volume, mass, first moment ∫ρ x dV (3), second moment ∫ρ x xᵀ dV (6 symmetric components)."""
    N, wd = _quadrature(mesh)
    ne = len(mesh.tets)
    vol = wd.sum(axis=1)
    mass = vol * rho
    m1 = np.zeros((ne, 3))
    m2 = np.zeros((ne, 6))
    step = 50_000
    for lo in range(0, ne, step):
        hi = min(ne, lo + step)
        X = mesh.nodes[mesh.tets[lo:hi]]                            # (e, n, 3)
        xq = np.einsum("qa,eak->eqk", N, X)                         # (e, q, 3)
        w = wd[lo:hi] * rho[lo:hi, None]                            # (e, q)
        m1[lo:hi] = np.einsum("eq,eqk->ek", w, xq)
        for c, (i, j) in enumerate(_SYM):
            m2[lo:hi, c] = np.einsum("eq,eq->e", w, xq[:, :, i] * xq[:, :, j])
    return vol, mass, m1, m2


def mass_properties(study: Any, mesh: Any) -> MassProperties:
    """Mass, centre of mass and inertia tensor of the study's mesh plus its added masses. Bolts replaced by beam elements and the members of a beam system
    are not part of the mesh: a note says so.

    Added masses as the SOLVER sees them: the rotary inertia of a point mass is part of the inertia tensor only for a RIGID coupling (the solver distributes the mass
    into auxiliary points there); with a distributing coupling the solver keeps the translation only, so the inertia is left out and a note says so. A point mass
    or surface mass of 0 or less, a non-finite number, or an inertia tensor that violates the triangle inequality (`dynamics.mass.rotary_points`) is refused with
    a `ValueError`."""
    from ..dynamics.mass import inertia_matrix, mass_items, point_of, rotary_points, surface_mass_nodes
    from ..dynamics.model import PointMass
    from ..printing import element_density
    rho = np.asarray(element_density(study, mesh), float)
    vol, m_e, m1_e, m2_e = _accumulate_element_moments(mesh, rho)
    parts: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(mesh.part_names):
        sel = mesh.tet_part == i
        m = float(m_e[sel].sum())
        c = m1_e[sel].sum(axis=0) / m if m > 0 else np.zeros(3)
        parts[str(name)] = {"mass_t": m, "volume_mm3": float(vol[sel].sum()), "cog_mm": [float(x) for x in c]}
    bodies = float(m_e.sum())
    M = bodies
    m1 = m1_e.sum(axis=0)
    S = _sym_to_matrix(m2_e.sum(axis=0))
    own = np.zeros((3, 3))
    notes: list[str] = []
    added: list[dict[str, Any]] = []
    added_t = 0.0
    for it in mass_items(study):
        if isinstance(it, PointMass):
            p = np.asarray(point_of(mesh, it), float)
            m = float(it.mass)
            label = it.name or it.selection
            if not (np.isfinite(m) and m > 0):
                raise ValueError(f"Point mass '{label}': the mass must be a positive number (it is {it.mass!r})")
            I = inertia_matrix(it)
            if I is not None:
                if not np.all(np.isfinite(I)):
                    raise ValueError(f"Point mass '{label}': the inertia tensor must be finite")
                rotary_points(it, p, 1.0)                              # ValueError: the tensor violates the triangle inequality (I_xx <= I_yy + I_zz ...)
            rigid = it.coupling == "rigid"
            if I is not None and rigid:
                own = own + np.asarray(I, float)
            elif I is not None and np.any(I):
                notes.append(f"point mass '{label}': a {it.coupling} coupling keeps only the translation in the solver, so its rotary inertia is NOT part of the inertia tensor "
                             "(use a rigid coupling)")
            added.append({"name": label, "kind": "point_mass", "mass_kg": m * 1000.0, "point_mm": [float(x) for x in p], "coupling": it.coupling,
                          "rotary_inertia": I is not None and rigid})
            M += m
            m1 = m1 + m * p
            S = S + m * np.outer(p, p)
        else:
            idx, mm = surface_mass_nodes(mesh, it)
            m = float(np.sum(mm))
            if not (np.isfinite(m) and m > 0):
                raise ValueError(f"Surface mass '{it.name or it.selection}': the mass must be a positive number (it is {m:g} t)")
            X = mesh.nodes[idx]
            added.append({"name": it.name or it.selection, "kind": "surface_mass", "mass_kg": m * 1000.0,
                          "point_mm": [float(x) for x in ((mm[:, None] * X).sum(axis=0) / m if m > 0 else np.zeros(3))]})
            M += m
            m1 = m1 + (mm[:, None] * X).sum(axis=0)
            S = S + np.einsum("n,ni,nj->ij", mm, X, X)
        added_t += m
    c = m1 / M if M > 0 else np.zeros(3)
    Sc = S - M * np.outer(c, c)                                     # second moment about the centre of mass (Steiner)
    Ic = float(np.trace(Sc)) * np.eye(3) - Sc + own
    if hasattr(study, "active_bolts") and study.active_bolts():
        notes.append(f"{len(study.active_bolts())} bolt(s) are replaced by beam elements and are not part of this mass - add their mass as point masses")
    if getattr(getattr(study, "frame", None), "active", False):
        notes.append("the members of the beam system are not part of this mass")
    return MassProperties(M, c, 0.5 * (Ic + Ic.T), bodies, added_t, float(vol.sum()), parts, added, notes)

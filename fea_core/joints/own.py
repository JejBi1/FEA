"""Bolts on the own assembly of the linear path (F13 step 1b, part 2): the bolt as exact Timoshenko beam elements between rigid bearing faces.

What ccx gets (`ccx/inp._write_bolts`, `joints.fem`): a chain of U1 elements (64 or more – the element locks in bending), a rigid body per bearing face
(`*COUPLING, *KINEMATIC`), a cut of the beam in the middle with a gap DOF g (u_M2 = u_M1 + g·e) for the preload and a penalty spring for the thermal
elongation. What the own assembly gets (the same model, the same stiffness `BoltFE.A, I, E, ν`):

* **one beam element per distance between two bearing faces** – the exact Timoshenko matrix (`beams.timoshenko.local_k`, EA/L, GJ/L with J = I_p,
  12EI/(κGA L²) with κ = 0.9, the same κ as the U1 section) is exact for loads at the ends, so the 64 elements of ccx are not needed;
* **a rigid body per bearing face** (`linear_constraints.rigid_body`: every node u = u_ref + θ × (x − p) from a pair of extra nodes) – the ends of the
  elements are these extra nodes (ccx ties the beam node to the REF node of the coupling, DOF 1–6);
* **an initial elongation g of the cut element instead of the gap DOF**: the element carries N = k·(Δu − g) (Δu = the relative displacement of its ends along
  the axis). The thermal elongation of a bolt is a prescribed g = α·ΔT·L (EXACT – ccx needs a penalty spring, error 1e-4), the preload F_V is the condition
  N = F_V: with the unit states "elongation of bolt j by 1 mm" (`unit_loads` – one right-hand side per preloaded bolt on the SAME factorization) the
  bolt forces are N_i = Σ_j C_ij·g_j, C_ij = k_i·(B_iᵀ·x_j − δ_ij), and g = C⁻¹·F_V (`preload_gaps`). The matrix K stays the one of the continuous
  bolts (symmetric positive definite): a gap DOF that is free in the preload state and constrained in the others would need two factorizations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fem import BoltFE

SHEAR_COEFFICIENT = 0.9               # shear area κ·A of the round section (the fifth item of the *BEAM SECTION of the U1 bolt, `timoshenko.shear_areas`)


class BoltChainError(ValueError):
    """The bolt has no two bearing faces to put a beam between."""


@dataclass
class BoltChain:
    """The beam of one bolt: the bearing faces in the order along the axis and the elements between them."""
    bolt: BoltFE
    positions: list[int]               # the index in `bolt.points` of every position of the beam with a bearing face (ascending along the axis)
    keys: list[list[str]]              # the selections (faces of the mesh) of every position – several faces at one point make ONE rigid body
    s: np.ndarray                      # (m,) the distance of the positions from the first point of the beam [mm]
    cut: int | None                    # the element that carries the preload / thermal elongation (the one that contains the cut of the beam); None = no cut
    k_axial: np.ndarray                # (m − 1,) EA/L of every element [N/mm]

    @property
    def n_elements(self) -> int:
        return len(self.positions) - 1

    @property
    def e(self) -> np.ndarray:
        e = np.asarray(self.bolt.e, float)
        return e / np.linalg.norm(e)


def chain_of(b: BoltFE) -> BoltChain:
    """Positions, faces and the cut of the bolt `b` (`joints.fem.bolt_models`): the faces at the same point of the beam form one position."""
    pos = sorted({k for _, k in b.couplings})
    if len(pos) < 2:
        raise BoltChainError(f"{b.name}: the bolt has fewer than two bearing faces in the mesh – no beam between them")
    keys = [[key for key, k in b.couplings if k == p] for p in pos]
    e = np.asarray(b.e, float) / np.linalg.norm(b.e)
    s = np.array([(b.points[p] - b.points[0]) @ e for p in pos], float)
    cut = None
    if b.split is not None:
        s_cut = float((b.points[b.split] - b.points[0]) @ e)
        # the element that contains the cut; a cut AT a bearing position (the middle point of the beam is the node of an intermediate face, e.g. the middle plate
        # of three on a fitted bolt) belongs to the element AFTER the position – as in ccx, where that face is tied to the node before the gap (m1) and the next
        # element starts at the node after it (m2) (`ccx/inp._write_bolts`)
        cut = int(np.clip(np.searchsorted(s, s_cut + 1e-9, side="right") - 1, 0, len(pos) - 2))
    L = np.diff(s)
    if (L <= 0).any():
        raise BoltChainError(f"{b.name}: the bearing faces of the bolt do not lie in the order along its axis")
    return BoltChain(b, pos, keys, s, cut, b.E * b.A / L)


def element_matrices(ch: BoltChain) -> np.ndarray:
    """(m − 1, 12, 12): the exact Timoshenko matrices of the elements in the global axes, DOF order (u_a, θ_a, u_b, θ_b) of 3 components each."""
    from ..beams.timoshenko import Elems, local_k, to_global
    b = ch.bolt
    L = np.diff(ch.s)
    n = len(L)
    e = ch.e
    t1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    t1 /= np.linalg.norm(t1)
    R = np.tile(np.stack([e, t1, np.cross(e, t1)]), (n, 1, 1))              # rows = the local axes x (along the bolt), y, z
    E = np.full(n, float(b.E))
    G = E / (2.0 * (1.0 + float(b.nu)))
    A = np.full(n, float(b.A))
    I = np.full(n, float(b.I))
    As = SHEAR_COEFFICIENT * A
    phi = 12.0 * E * I / (G * As * L * L)
    el = Elems(n, np.arange(n), np.arange(n) + 1, L, R, E, G, np.zeros(n), A, I, I, 2.0 * I, As, As, phi, phi,
               np.zeros((n, 12), bool), np.full(n, np.nan), True)
    return to_global(R, local_k(el))


def preload_gaps(C: np.ndarray, F_V: np.ndarray) -> np.ndarray:
    """The initial elongations g of the cut elements that give the bolt forces N = F_V: C·g = F_V (C = the force in bolt i per unit elongation of bolt j,
    negative definite). Raises `ValueError` for a matrix that is not (numerically) the one of a stable structure.

    A bolt with F_V = 0 (the preload state of a bolt whose preload the settlement consumed, `vdi2230.basics` clamps F_V at 0) gets the condition N = 0: it is a CUT
    in the preload state – the same as ccx, where its gap is free and carries a zero load; the elongation g that solves C·g = F_V is whatever the other bolts
    need for it."""
    C = np.asarray(C, float)
    if C.shape[0] == 0:
        return np.zeros(0)
    sym = np.abs(C - C.T).max() / max(np.abs(C).max(), 1e-300)
    if sym > 1e-6:
        raise ValueError(f"The bolt-force matrix of the preload is not symmetric ({sym:.1e}) – the unit states do not belong to this matrix")
    w = np.linalg.eigvalsh(0.5 * (C + C.T))
    if not (w < 0).all():
        raise ValueError("The preload of the bolts cannot be applied: the structure does not resist the elongation of a bolt (bolts that are the "
                         "only connection of a part – the preload needs the clamped parts to be in contact)")
    return np.linalg.solve(C, np.asarray(F_V, float))


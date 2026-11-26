"""General orthotropic material (outside 3D printing): `Material.ortho` = {E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
[, alpha1, alpha2, alpha3]} in the axes of the material, `Material.ortho_csys` = name of the coordinate system of the study (Cartesian:
axes 1, 2, 3 = x, y, z of the system; cylindrical: 1 = radial, 2 = tangential, 3 = axial; None = global axes).

ccx: *ELASTIC, TYPE=ENGINEERING CONSTANTS (ν_ij: ε_j = −ν_ij·σ_i/E_i) + *ORIENTATION with *SOLID SECTION;
SYSTEM=CYLINDRICAL = points a, b on the axis, X' radially, Y' tangentially, Z' from a to b (ccx 2.23 manual, 7.103). Thermal
expansion *EXPANSION, TYPE=ORTHO (otherwise isotropic alpha). Stresses in the .frd are in global axes (as for printing).
The NGSolve backend and MYSTRAN cannot do it yet (→ ccx / error).
"""
from __future__ import annotations

import numpy as np

KEYS = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23")


def compliance(o: dict) -> np.ndarray:
    """Compliance matrix 6×6 (Voigt: 11 22 33 12 13 23, shear engineering)."""
    E1, E2, E3 = (float(o[k]) for k in ("E1", "E2", "E3"))
    n12, n13, n23 = (float(o[k]) for k in ("nu12", "nu13", "nu23"))
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -n12 / E1
    S[0, 2] = S[2, 0] = -n13 / E1
    S[1, 2] = S[2, 1] = -n23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / float(o["G12"]), 1 / float(o["G13"]), 1 / float(o["G23"])
    return S


def check(study, part_names) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    used = {study.part_materials.get(p, study.default_material) for p in part_names
            if p not in study.excluded_parts()}
    for name, m in study.materials.items():
        o = getattr(m, "ortho", None)
        if not o:
            continue
        what = f"Orthotropic material '{name}'"
        miss = [k for k in KEYS if o.get(k) is None]
        if miss:
            errors.append(f"{what}: missing {', '.join(miss)}")
            continue
        try:
            vals = [float(o[k]) for k in KEYS]
        except (TypeError, ValueError):
            errors.append(f"{what}: constants must be numbers")
            continue
        if min(vals[0:3] + vals[6:9]) <= 0:
            errors.append(f"{what}: moduli E and G must be positive")
            continue
        if np.linalg.eigvalsh(compliance(o)).min() <= 0:
            errors.append(f"{what}: constants are not physically admissible (compliance matrix is not positive definite – check ν: ν_ij² < E_i/E_j)")
        if m.is_print:
            errors.append(f"{what}: a material cannot be both orthotropic and printed (E_z) – use the 3D printing settings for printing")
        cs = getattr(m, "ortho_csys", None)
        if cs and cs not in study.coordinate_systems:
            errors.append(f"{what}: coordinate system '{cs}' does not exist")
        if name in used:
            warnings.append(f"{what}: the safety factor against Re / von Mises does not apply to an orthotropic material – assess stress in the material axes against its strengths")
    return errors, warnings


def axes(study, m) -> tuple[str, list[float]]:
    """(SYSTEM, 6 numbers of the row *ORIENTATION) for the axes of the material."""
    from .csys import matrix
    cs = study.coordinate_systems.get(getattr(m, "ortho_csys", None) or "")
    if cs is None:
        return "RECTANGULAR", [1.0, 0, 0, 0, 1.0, 0]
    if cs.kind == "cylindrical":
        o = np.asarray(cs.origin, float)
        z = np.asarray(cs.z_axis, float)
        z = z / np.linalg.norm(z)
        return "CYLINDRICAL", [*o.tolist(), *(o + z).tolist()]
    R = matrix(cs)
    return "RECTANGULAR", [*R[0].tolist(), *R[1].tolist()]


def material_lines(m, fmt, thermal: bool, t_ref: float) -> str:
    """*ELASTIC (ENGINEERING CONSTANTS) and *EXPANSION of the orthotropic material."""
    o = m.ortho
    v = [float(o[k]) for k in KEYS]
    s = "*ELASTIC, TYPE=ENGINEERING CONSTANTS\n" + ", ".join(fmt(x) for x in v[:8]) + f",\n{fmt(v[8])}, 0.\n"
    if thermal:
        a = [o.get(f"alpha{i}") for i in (1, 2, 3)]
        if all(x is not None for x in a):
            s += f"*EXPANSION, TYPE=ORTHO, ZERO={fmt(t_ref)}\n" + ", ".join(fmt(float(x)) for x in a) + "\n"
        else:
            s += f"*EXPANSION, ZERO={fmt(t_ref)}\n{fmt(m.alpha or 0.0)}\n"
    return s


# ------------------------------------------------------------------ the own assembly (linear path outside ccx)

def strain_transform(R: np.ndarray) -> np.ndarray:
    """T (..., 6, 6): the strain (xx yy zz xy xz yz, engineering shear) in the axes R (..., 3, 3; rows = the axes in the global ones) from the strain in the
    global axes. The same formulas as `assembly.kernels.strain_transform`."""
    R = np.asarray(R, float)
    T = np.zeros(R.shape[:-2] + (6, 6))
    for k, (p, q) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        a, b = R[..., p, :], R[..., q, :]
        if p == q:
            T[..., k, :] = np.stack([a[..., 0] * a[..., 0], a[..., 1] * a[..., 1], a[..., 2] * a[..., 2],
                                     a[..., 0] * a[..., 1], a[..., 0] * a[..., 2], a[..., 1] * a[..., 2]], axis=-1)
        else:
            T[..., k, :] = np.stack([2 * a[..., 0] * b[..., 0], 2 * a[..., 1] * b[..., 1], 2 * a[..., 2] * b[..., 2],
                                     a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0], a[..., 0] * b[..., 2] + a[..., 2] * b[..., 0],
                                     a[..., 1] * b[..., 2] + a[..., 2] * b[..., 1]], axis=-1)
    return T


def stiffness_local(o: dict) -> np.ndarray:
    """C (6, 6) in the axes of the material (the inverse of the compliance)."""
    return np.linalg.inv(compliance(o))


def alpha_local(m) -> np.ndarray:
    """(α1, α2, α3) of the material axes (the isotropic α of the material when the orthotropic constants have none – as ccx writes *EXPANSION)."""
    a = [m.ortho.get(f"alpha{i}") for i in (1, 2, 3)]
    if all(x is not None for x in a):
        return np.array([float(x) for x in a])
    return np.full(3, float(m.alpha or 0.0))


def thermal_voigt(R: np.ndarray | None, a: np.ndarray) -> np.ndarray:
    """The thermal strain per unit ΔT (xx yy zz xy xz yz, engineering shear) in the global axes of a material with the expansion a = (α1, α2, α3) in
    the axes R (rows; None = the global axes)."""
    A = np.diag(a) if R is None else R.T @ np.diag(a) @ R
    return np.array([A[0, 0], A[1, 1], A[2, 2], 2 * A[0, 1], 2 * A[0, 2], 2 * A[1, 2]])


def table_entry(study, m) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """(C (6, 6), kind, ax (6,), A (6,)) of an orthotropic material for the own assembly: kind 0 = C and A in the global axes (no coordinate system or a Cartesian
    one), kind 1 = C and A in the local axes of a cylindrical system, rotated at every Gauss point (ax = a point on the axis and its unit direction)."""
    from .csys import matrix
    cs = study.coordinate_systems.get(getattr(m, "ortho_csys", None) or "")
    Cl = stiffness_local(m.ortho)
    a = alpha_local(m)
    if cs is not None and cs.kind == "cylindrical":
        z = np.asarray(cs.z_axis, float)
        z = z / np.linalg.norm(z)
        return Cl, 1, np.concatenate([np.asarray(cs.origin, float), z]), np.concatenate([a, np.zeros(3)])
    R = np.eye(3) if cs is None else matrix(cs)
    T = strain_transform(R)
    return T.T @ Cl @ T, 0, np.zeros(6), thermal_voigt(R, a)


def cylindrical_axes_at(ax: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Rows (e_r, e_θ, e_z) of the cylindrical system `ax` (a point on the axis, the unit direction) at the points X (..., 3) → (..., 3, 3); the same as the kernel."""
    X = np.asarray(X, float)
    z = ax[3:]
    d = X - ax[:3]
    rv = d - (d @ z)[..., None] * z
    r = np.linalg.norm(rv, axis=-1)
    er = rv / np.where(r > 0, r, 1.0)[..., None]
    et = np.cross(z, er)
    return np.stack([er, et, np.broadcast_to(z, er.shape)], axis=-2)


def part_entries(study, mesh) -> list[tuple[np.ndarray, int, np.ndarray, np.ndarray]]:
    """`table_entry` of every part of the mesh (an isotropic material: its C, kind 0, α·(1, 1, 1, 0, 0, 0))."""
    from .solve_large import iso_stiffness
    out = []
    for p in mesh.part_names:
        m = study.material_of(p)
        if getattr(m, "ortho", None):
            out.append(table_entry(study, m))
        else:
            a = float(m.alpha or 0.0)
            out.append((iso_stiffness(m.E, m.nu), 0, np.zeros(6), np.array([a, a, a, 0.0, 0.0, 0.0])))
    return out


def has_ortho(study, mesh=None) -> bool:
    """A part of the mesh has an orthotropic material (without a mesh: any material of the study has one)."""
    if mesh is None:
        return any(getattr(m, "ortho", None) for m in study.materials.values())
    return any(getattr(study.material_of(p), "ortho", None) for p in mesh.part_names)


class ElementField:
    """The matrix C and the thermal strain of the elements of a study with orthotropic materials at the Gauss points of C3D10 (stress recovery of the own
    assembly): the same values the assembly kernels use. `chunk(lo, hi, X)` → (C (e, 4, 6, 6) in the global axes, A (e, 4, 6)) of the elements lo..hi
    (X = their nodes (e, 10, 3))."""

    def __init__(self, study, mesh):
        from .loads import GAUSS4
        from .results import tet10_shape
        self.part = np.asarray(mesh.tet_part)
        self.entries = part_entries(study, mesh)
        self.N = np.stack([tet10_shape(np.asarray(p)) for p in GAUSS4])          # (4, 10)

    def chunk(self, lo: int, hi: int, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        part = self.part[lo:hi]
        n = hi - lo
        C = np.empty((n, 4, 6, 6))
        A = np.empty((n, 4, 6))
        for pi, (Cm, kind, ax, Am) in enumerate(self.entries):
            sel = np.nonzero(part == pi)[0]
            if not len(sel):
                continue
            if kind == 0:
                C[sel] = Cm
                A[sel] = Am
                continue
            xg = np.einsum("gk,ekc->egc", self.N, X[sel])                      # the Gauss points (e, 4, 3)
            R = cylindrical_axes_at(ax, xg)                                    # (e, 4, 3, 3)
            T = strain_transform(R)                                            # (e, 4, 6, 6)
            C[sel] = np.einsum("egki,kl,eglj->egij", T, Cm, T)
            Ag = np.einsum("egki,k,egkj->egij", R, Am[:3], R)                  # (e, 4, 3, 3) = sum of alpha_k e_k e_k^T
            A[sel] = np.stack([Ag[..., 0, 0], Ag[..., 1, 1], Ag[..., 2, 2], 2 * Ag[..., 0, 1], 2 * Ag[..., 0, 2], 2 * Ag[..., 1, 2]], axis=-1)
        return C, A

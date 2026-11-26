"""Interface forces and moments of the modal dynamics (F14 stage S4): the base reaction of the structure per mode, the sine interface
force / moment spectrum and the RMS interface force / moment in random vibration.

Model (the conventions of `dynamics.response` / `dynamics.psd`): the structure is held at its interface (constraints and elastic supports), the base
moves with the acceleration a_b in the direction d. The relative displacement is u = Σ φ_j q_j + (static residual), q_j = −Γ_j(d)·H_j·a_b. The force that the
support exerts on the structure at the interface is the elastic reaction K·u at the held nodes (the same quantity as the `reaction` of the harmonic result):

    F(f) = Σ_j F_j·q_j(f) + a_b(f)·F_res(d),      M(f) = Σ_j M_j·q_j(f) + a_b(f)·M_res(d)

with the modal base-reaction vectors F_j = Σ_c RF_jc (force) and M_j = Σ_c (x_c − x_ref) × RF_jc (moment about the reference point) taken from the
modal reactions K·φ_j of the held nodes, and the static residual F_res(d) = F_static(d) + Σ_j F_j·Γ_j(d)/ω_j² (the reaction of the unit base acceleration solved
statically MINUS the part the retained modes already carry – the same construction as `response.residual_vectors`). Consequences that the tests pin:
  * f → 0: F → a_b·F_static(d), and F_static(d) = M_total·d for the whole interface (Newton) – independent of the number of modes;
  * a complete basis needs no residual (Σ_j m_eff,j = M_total), a truncated one needs it.
The support also transmits the damping force of the modes: each modal term carries the factor (1 + 2iζ_j f/f_j) (viscous modal damping; (1 + iη) for structural damping),
which is exact for modal damping - the force is then the one of Newton's law for the base (m·a_absolute for a SDOF; the magnitude at resonance is
m·a·√(1 + 4ζ²)/(2ζ)). Without it the force would miss a term of the order 2ζ at resonance (and 4ζ² in the magnitude). The static residual carries no damping.

`ModalInterface` carries only small arrays (K × 3), so the spectra and the notching (`fea_core.space.notching`) work without the mesh. `run_interface` builds it
for a study (modal basis + three unit-acceleration statics from the same factor). Units: N, mm, t, s; base acceleration in mm/s² (g = 9810 mm/s²).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .model import G_MM

AXES = ("x", "y", "z")
ALL_GROUP = "all"


class InterfaceError(RuntimeError):
    """The interface forces cannot be computed (no held nodes, no modes, unknown group)."""


# ------------------------------------------------------------------ data
@dataclass
class InterfaceGroup:
    """The base reaction of one set of held nodes (all of them, or one constraint / elastic support).

    F, M: (K, 3) modal base-reaction force [N per unit modal coordinate] and moment about `ModalInterface.ref_point` [N·mm per unit q];
    F_static, M_static: (3, 3) – row d = reaction force / moment of the interface for a unit base acceleration in the axis d (N, N·mm per mm/s²), from the static
    solution (the full reaction, including what the retained modes carry)."""
    name: str
    F: np.ndarray
    M: np.ndarray
    F_static: np.ndarray
    M_static: np.ndarray
    n_nodes: int = 0


@dataclass
class ModalInterface:
    """Modal interface data of a study: frequencies, participation vectors Γ_j (K, 3), effective masses, groups of held nodes, reference point, total mass."""
    freq: np.ndarray
    gamma: np.ndarray
    eff_mass: np.ndarray
    total_mass: float
    ref_point: np.ndarray
    groups: dict[str, InterfaceGroup]
    info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def K(self) -> int:
        return len(self.freq)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(self.freq, float)

    def group(self, name: str | None = None) -> InterfaceGroup:
        key = name or ALL_GROUP
        if key not in self.groups:
            raise InterfaceError(f"Unknown interface group '{key}' (groups: {', '.join(self.groups)})")
        return self.groups[key]

    def residual(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(F_res, M_res), each (3, 3): row d = the static residual of the unit base acceleration in the axis d (see the module docstring)."""
        g = self.group(name)
        w2 = np.maximum(self.omega ** 2, 1e-300)
        coef = self.gamma[:, :3] / w2[:, None]                 # (K, 3 dirs)
        F_res = g.F_static + np.einsum("kd,ki->di", coef, g.F)
        M_res = g.M_static + np.einsum("kd,ki->di", coef, g.M)
        return F_res, M_res

    def modal_table(self, name: str | None = None) -> list[dict]:
        """One row per mode for reports: f_j, Γ_j, m_eff,j, the modal base-reaction vectors."""
        g = self.group(name)
        rows = []
        for j in range(self.K):
            rows.append({"mode": j + 1, "f_Hz": float(self.freq[j]), "gamma": self.gamma[j, :3].tolist(), "eff_mass_t": self.eff_mass[j, :3].tolist(),
                         "F_N": g.F[j].tolist(), "M_Nmm": g.M[j].tolist()})
        return rows

    def static_mass_check(self) -> dict:
        """The static interface reaction of the unit base acceleration against M_total·d (Newton): the relative errors in the three axes."""
        g = self.group(ALL_GROUP)
        err = [abs(float(g.F_static[d, d]) - self.total_mass) / max(self.total_mass, 1e-300) for d in range(3)]
        off = [float(np.linalg.norm(np.delete(g.F_static[d], d))) / max(self.total_mass, 1e-300) for d in range(3)]
        return {"relative_error": err, "off_axis_relative": off, "total_mass_t": self.total_mass}


# ------------------------------------------------------------------ transfer functions
def unit_direction(direction: Any) -> np.ndarray:
    """'x' | 'y' | 'z' | 1..3 | a vector → unit vector."""
    if isinstance(direction, str):
        s = direction.strip().lower()
        if s in AXES:
            d = np.zeros(3)
            d[AXES.index(s)] = 1.0
            return d
        raise InterfaceError(f"Unknown direction '{direction}' (x, y, z or a vector)")
    a = np.asarray(direction, float)
    if a.shape == ():
        d = np.zeros(3)
        d[int(a) - 1] = 1.0
        return d
    if a.shape != (3,) or not np.linalg.norm(a) > 0:
        raise InterfaceError("A direction is an axis name or a non-zero vector of three numbers")
    return a / np.linalg.norm(a)


BAND_REL_SPAN = 0.35                 # the resonance windows of `band_grid` reach this fraction of f_n to each side (the log base carries the rest)
CHUNK_ELEMENTS = 400_000             # frequencies x modes of one block of `interface_transfer` (a block of complex arrays of that size is ~6 MB; the memory is bounded by it)


def band_grid(f_min: float, f_max: float, fn: np.ndarray, zeta: np.ndarray, per_band: int) -> np.ndarray:
    """Frequencies around every resonance in [f_min, f_max]: the points scale with the half-power band 2·ζ·f_n, so a notch (and the table that is read from it) resolves a
    resonance of ζ = 0.25 % as well as one of 5 %. With f = f_n·(1 + ζ·sinh t) the step is uniform in t near the peak (`per_band` points across the half-power points ±1, one of
    them exactly at f_n) and grows with the distance (a constant relative step of the offset |f − f_n|) up to ±`BAND_REL_SPAN`·f_n. The number of points per mode is
    ~`per_band`·asinh(`BAND_REL_SPAN`/ζ)."""
    fn = np.atleast_1d(np.asarray(fn, float)).ravel()
    ze = np.broadcast_to(np.asarray(zeta, float), fn.shape)
    pb = max(int(per_band), 4)
    parts: list[np.ndarray] = []
    for f0, z in zip(fn, ze):
        if not (f_min < f0 < f_max) or not z > 0:
            continue
        tmax = math.asinh(BAND_REL_SPAN / max(float(z), 1e-6))
        n = max(int(math.ceil(pb * tmax / 2.0)), 1)
        g = f0 * (1.0 + float(z) * np.sinh(np.linspace(-tmax, tmax, 2 * n + 1)))
        parts.append(g[(g >= f_min) & (g <= f_max)])
    return np.concatenate(parts) if parts else np.empty(0)


def sine_grid(f_min: float, f_max: float, fn: np.ndarray, zeta: np.ndarray, per_decade: int = 120, points: int = 12, bias: float = 3.0,
              per_band: int | None = None) -> np.ndarray:
    """Frequencies of the sine interface spectrum: a log base + the anchors of the half-power points of every mode in the range (`response.freq_grid`); with `per_band`
    also `band_grid` (points that scale with the half-power band of every mode - what the notching refines)."""
    from .response import freq_grid
    if not (f_min > 0 and f_max > f_min):
        raise InterfaceError("The frequency range must satisfy 0 < f_min < f_max")
    nb = max(8, int(math.ceil(math.log10(f_max / f_min) * per_decade)) + 1)
    parts = [np.geomspace(f_min, f_max, nb), freq_grid(f_min, f_max, np.asarray(fn, float), np.asarray(zeta, float), points, bias)]
    if per_band:
        parts.append(band_grid(f_min, f_max, np.asarray(fn, float), np.asarray(zeta, float), int(per_band)))
    f = np.unique(np.concatenate(parts))
    return f[np.concatenate([[True], np.diff(f) > 1e-13 * f[1:]])]


def _zeta_vec(zeta: Any, K: int) -> np.ndarray:
    z = np.asarray(zeta, float)
    return np.full(K, float(z)) if z.ndim == 0 else z


def interface_transfer(mi: ModalInterface, f: np.ndarray, zeta: Any, direction: Any, group: str | None = None, eta: float | None = None,
                       residual: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Complex transfer from the base acceleration (1 mm/s² in `direction`) to the interface force (n_f, 3) [N] and moment (n_f, 3) [N·mm]."""
    from .response import frf
    f = np.atleast_1d(np.asarray(f, float))
    d = unit_direction(direction)
    g = mi.group(group)
    z = _zeta_vec(zeta, mi.K)
    p = -(mi.gamma[:, :3] @ d)                                       # (K,) modal force of the base excitation per unit acceleration
    fj = np.maximum(np.asarray(mi.freq, float), 1e-300)
    TF = np.empty((len(f), 3), complex)
    TM = np.empty((len(f), 3), complex)
    rows = max(16, CHUNK_ELEMENTS // max(mi.K, 1))                  # blocks of frequencies: the (n_f, K) arrays never exist as a whole (300 modes x 100 000 frequencies = GBs)
    for i0 in range(0, len(f), rows):
        fb = f[i0:i0 + rows]
        H = frf(fb, mi.freq, z, eta)                                 # (rows, K)
        # the force that the support transmits is the elastic reaction K·φ_j·q_j PLUS the damping force of the same mode: with modal damping the reaction of mode j
        # is K·φ_j·q_j·(1 + 2iζ_j f/f_j) (viscous) or K·φ_j·q_j·(1 + iη) (structural) - exact for modal damping, the 2ζ-order term at resonance, 4ζ² in the magnitude
        damp = (1.0 + 1j * float(eta)) if eta is not None else 1.0 + 2j * z[None, :] * fb[:, None] / fj[None, :]
        q = H * p[None, :] * damp
        TF[i0:i0 + rows] = q @ g.F
        TM[i0:i0 + rows] = q @ g.M
    if residual:
        F_res, M_res = mi.residual(group)
        TF = TF + (d @ F_res)[None, :]
        TM = TM + (d @ M_res)[None, :]
    return TF, TM


def peak_magnitude(T: np.ndarray) -> np.ndarray:
    """max over the phase of the length of the vector Re(T·e^{iφ}) for a complex (n, 3) array → (n,)."""
    from .response import max_over_phase
    return max_over_phase(np.asarray(T).real, -np.asarray(T).imag)


@dataclass
class SineInterface:
    """The sine interface force / moment spectrum for a base acceleration profile a(f)."""
    f: np.ndarray
    a_g: np.ndarray                  # input amplitude [g]
    F: np.ndarray                    # (n, 3) complex [N]
    M: np.ndarray                    # (n, 3) complex [N·mm]
    F_peak: np.ndarray               # (n,) max over the phase of |F| [N]
    M_peak: np.ndarray               # (n,) [N·mm]
    TF_peak: np.ndarray              # (n,) |transfer| per 1 g of input [N/g]
    TM_peak: np.ndarray              # (n,) [N·mm/g]
    direction: list[float]
    group: str
    residual_used: bool

    def summary(self) -> dict:
        i, j = int(np.argmax(self.F_peak)), int(np.argmax(self.M_peak))
        return {"group": self.group, "direction": self.direction, "F_peak_max_N": float(self.F_peak[i]), "f_at_F_peak_Hz": float(self.f[i]),
                "M_peak_max_Nmm": float(self.M_peak[j]), "f_at_M_peak_Hz": float(self.f[j]), "residual_used": self.residual_used}


def sine_interface(mi: ModalInterface, direction: Any, profile: Any, zeta: Any, f: np.ndarray | None = None, group: str | None = None,
                   eta: float | None = None, residual: bool = True, f_min: float | None = None, f_max: float | None = None, per_decade: int = 120,
                   per_band: int | None = None) -> SineInterface:
    """The sine interface force / moment spectrum.

    profile: the input amplitude in g – a table [[f Hz, a g], …] (log-log interpolation between the points as for a sine test profile, no extrapolation: 0 outside)
    or a number (constant over the grid). f = the frequency grid (default: `sine_grid` over the range of the table / f_min–f_max)."""
    from .psd import loglog_interp
    d = unit_direction(direction)
    z = _zeta_vec(zeta, mi.K)
    if isinstance(profile, (int, float)):
        if f is not None:
            grid = np.asarray(f, float)
        elif f_min and f_max:
            grid = sine_grid(float(f_min), float(f_max), mi.freq, z, per_decade, per_band=per_band)
        else:
            raise InterfaceError("A constant input needs the frequency range (f_min, f_max)")
        a = np.full(len(grid), float(profile))
    else:
        t = np.asarray(profile, float)
        grid = sine_grid(float(t[:, 0].min()), float(t[:, 0].max()), mi.freq, z, per_decade, per_band=per_band) if f is None else np.asarray(f, float)
        a = loglog_interp(t, grid)
    TF, TM = interface_transfer(mi, grid, z, d, group, eta, residual)
    amp = a * G_MM
    F = TF * amp[:, None]
    M = TM * amp[:, None]
    return SineInterface(grid, a, F, M, peak_magnitude(F), peak_magnitude(M), peak_magnitude(TF) * G_MM, peak_magnitude(TM) * G_MM,
                         d.tolist(), group or ALL_GROUP, residual)


@dataclass
class RandomInterface:
    """RMS interface force / moment in random vibration (one-sided PSDs in Hz)."""
    f: np.ndarray
    asd_g2: np.ndarray
    F_psd: np.ndarray                # (n, 3) N²/Hz
    M_psd: np.ndarray                # (n, 3) (N·mm)²/Hz
    F_rms: np.ndarray                # (3,) N
    M_rms: np.ndarray                # (3,) N·mm
    F_rms_total: float
    M_rms_total: float
    direction: list[float]
    group: str
    residual_used: bool

    def summary(self, n_sigma: float = 3.0) -> dict:
        """RMS values and the n-sigma values for information (the force of a Gaussian process; n_sigma is a presentation choice, not a standard value)."""
        return {"group": self.group, "direction": self.direction, "F_rms_N": self.F_rms.tolist(), "M_rms_Nmm": self.M_rms.tolist(),
                "F_rms_total_N": self.F_rms_total, "M_rms_total_Nmm": self.M_rms_total,
                "F_nsigma_total_N": n_sigma * self.F_rms_total, "M_nsigma_total_Nmm": n_sigma * self.M_rms_total, "n_sigma": n_sigma,
                "residual_used": self.residual_used}


def random_interface(mi: ModalInterface, direction: Any, asd: Any, zeta: Any, group: str | None = None, eta: float | None = None,
                     residual: bool = True, per_decade: int | None = None, per_peak: int | None = None, ratio: float | None = None) -> RandomInterface:
    """RMS interface force and moment for a base ASD table [[f Hz, g²/Hz], …] (the direction is the base excitation).

    The PSD of each component is |T_i(f)|²·W(f) with the complex transfer of `interface_transfer` (modes + static residual, fully correlated), integrated by the
    trapezium rule on a grid refined at the resonances (`psd.freq_grid`); the RMS of the resultant is √(Σ_i RMS_i²)."""
    from . import psd
    d = unit_direction(direction)
    z = _zeta_vec(zeta, mi.K)
    t = np.asarray(asd, float)
    kw: dict[str, Any] = {} if per_decade is None else {"per_decade": int(per_decade)}
    if per_peak is not None:                          # the grid of a resonance scales with its half-power band (psd.freq_grid): the notching refines it through these two
        kw["per_peak"] = int(per_peak)
    if ratio is not None:
        kw["ratio"] = float(ratio)
    f = psd.freq_grid(float(t[:, 0].min()), float(t[:, 0].max()), mi.freq, z, **kw)  # pyright: ignore[reportArgumentType] – freq_grid takes a scalar or a per-mode array
    W_g = psd.asd_interp(t, f)
    W = psd.asd_to_mm(W_g)
    TF, TM = interface_transfer(mi, f, z, d, group, eta, residual)
    w = psd.trapz_weights(f)
    Fp = (TF.real ** 2 + TF.imag ** 2) * W[:, None]
    Mp = (TM.real ** 2 + TM.imag ** 2) * W[:, None]
    F_rms = np.sqrt((Fp * w[:, None]).sum(axis=0))
    M_rms = np.sqrt((Mp * w[:, None]).sum(axis=0))
    return RandomInterface(f, W_g, Fp, Mp, F_rms, M_rms, float(np.linalg.norm(F_rms)), float(np.linalg.norm(M_rms)), d.tolist(), group or ALL_GROUP, residual)


# ------------------------------------------------------------------ building from a modal basis
def _mixed_projectors(study: Any, mesh: Any, plain: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A selection that carries BOTH a cylindrical support and a plain constraint (Fixed / Displacement, e.g. radial + uz on the same bore): the held directions of a node are the
    UNION of the span of the cylindrical directions and the plain components - {selection: (nodes, P (n, 3, 3))}, P the orthogonal projector on that span. (A selection that also
    carries a remote support keeps the mask of `constraint_masks`.)"""
    from ..cylinder import support_directions
    from ..study import CylindricalSupport, Displacement, Fixed
    dirs = support_directions(study, mesh)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (idx, D) in dirs.items():
        if name not in plain:
            continue
        comp = np.zeros(3, bool)
        simple = True
        for c in study.constraints:
            if c.selection != name or isinstance(c, CylindricalSupport):
                continue
            if isinstance(c, Fixed):
                comp[:] = True
            elif isinstance(c, Displacement):
                comp |= np.array([v is not None for v in (c.ux, c.uy, c.uz)])
            else:
                simple = False
        if not simple:
            continue
        e = np.broadcast_to(np.eye(3)[comp], (len(idx), int(comp.sum()), 3))
        rows = np.concatenate([D, e], axis=1)                                   # (n, k + m, 3)
        _u, s, vt = np.linalg.svd(rows)                                          # vt (n, 3, 3): the right singular vectors; s (n, min(k + m, 3))
        keep = np.zeros((len(idx), 3), bool)
        keep[:, :s.shape[1]] = s > 1e-9 * np.maximum(s[:, :1], 1e-300)
        out[name] = (idx, np.einsum("nki,nk,nkj->nij", vt, keep.astype(float), vt))
    return out


def held_operators(study: Any, mesh: Any, cn: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Which part of the nodal reaction K·φ belongs to a constraint: ({selection: W (n_cn, 3, 3)}, W_all (n_cn, 3, 3)); the reaction of a group is W @ RF.

    RF = K·φ holds in EVERY component of a held node, but only the HELD components are support forces (a free component of a held node carries the inertia
    force of the mode). A `Fixed` / `Displacement` constraint holds the components it names (diagonal 0 / 1 matrix); a cylindrical support projects the force on its
    constrained directions (radial / tangential / axial – the same rule as the reactions of the statics, `solve.make_step`). A node held by several constraints splits the
    force of a component between them by the pseudo-inverse of the sum of their projectors, so the shares of the groups add up exactly to the force of the group "all"
    (`W_all` = the projector on the union of the held directions)."""
    from ..cylinder import projectors
    from ..solve import constraint_masks
    from ..study import CylindricalSupport
    n_c = len(cn)
    pos = -np.ones(len(mesh.nodes), int)
    pos[cn] = np.arange(n_c)
    plain = {c.selection for c in study.constraints if not isinstance(c, CylindricalSupport)}
    has_cyl = any(isinstance(c, CylindricalSupport) for c in study.constraints)
    proj = {k: v for k, v in projectors(study, mesh).items() if k not in plain} if has_cyl else {}
    mixed = _mixed_projectors(study, mesh, plain) if has_cyl else {}
    P: dict[str, np.ndarray] = {}
    for name, m in constraint_masks(study, mesh, own=True).items():
        Pm = np.zeros((n_c, 3, 3))
        if name in proj or name in mixed:
            idx, Pp = proj[name] if name in proj else mixed[name]
            k = pos[idx]
            keep = k >= 0
            Pm[k[keep]] = Pp[keep]
        else:
            for i in range(3):
                Pm[:, i, i] = m[cn, i]
        if Pm.any():
            P[name] = Pm
    S = np.zeros((n_c, 3, 3))
    for Pm in P.values():
        S += Pm
    Sp = np.linalg.pinv(S) if n_c else S
    return {k: Pm @ Sp for k, Pm in P.items()}, S @ Sp


def build_interface(study: Any, mesh: Any, basis: Any, statics: list[dict] | None, ref_point: Any = None) -> ModalInterface:
    """`ModalInterface` of a modal basis (`dynamics.modal.ModalBasis`) and the three static solutions of the unit base acceleration in x, y, z
    (each {"U" (N, 3), "RF" (n_cn, 3)} as in `ModalBasis.statics` / `modal.static_solutions`)."""
    from ..springs import CompactReactions
    from ..springs import supports
    if basis.K == 0:
        raise InterfaceError("The modal basis is empty")
    elastic_only = bool(supports(study))
    has_held = basis.rf is not None and basis.rf_nodes is not None and len(basis.rf_nodes) > 0
    if not has_held and not elastic_only:
        raise InterfaceError("The study has no held nodes (constraints) or elastic supports whose reactions could be taken - the interface forces need a supported model")
    if statics is None or len(statics) != 3:
        raise InterfaceError("The static solutions of the unit base accelerations in x, y, z are needed")
    from ..loads import nodal_loads
    from ..study import Gravity, LoadCase
    cn = np.asarray(basis.rf_nodes, int) if has_held else np.zeros(0, int)
    X = np.asarray(mesh.nodes, float)
    K = basis.K
    # RF of a static solution = K·u in every node: the reaction of a held node is RF minus the external load on it (the tributary weight of the held nodes)
    Fg = [nodal_loads(study, mesh, LoadCase(f"_if_{a}", [Gravity(vector=[1.0 if i == d else 0.0 for i in range(3)])]))[0][cn] for d, a in enumerate(AXES)]
    RF = np.asarray(basis.rf, float) if has_held else np.zeros((K, 0, 3))      # (K, n_c, 3)
    warnings: list[str] = []
    # elastic supports: nodal reactions of the modes and of the statics per support selection, kept only at the nodes of the support (`spr.idx[sel]`; the spring table is made once)
    spr = CompactReactions(study, mesh)
    elastic = list(spr.idx) if elastic_only else []
    sp_modes: dict[str, np.ndarray] = {}
    for j in range(K):
        for sel, R in spr(basis.shapes[j]).items():
            sp_modes.setdefault(sel, np.zeros((K,) + R.shape))[j] = R
    sp_static = [spr(st["U"]) for st in statics]
    sp_nz = {sel: np.nonzero(np.abs(Rm).sum(axis=(0, 2)) > 0)[0] for sel, Rm in sp_modes.items()}     # the support nodes that carry a reaction (positions in `spr.idx[sel]`)
    W_sel, W_all = held_operators(study, mesh, cn) if has_held else ({}, np.zeros((0, 3, 3)))
    nodes_all = []
    if has_held:
        nodes_all.append(X[cn][np.abs(W_all).sum(axis=(1, 2)) > 0])
    for sel in elastic:
        idx = spr.idx[sel][sp_nz[sel]]
        if len(idx):
            nodes_all.append(X[idx])
    if not nodes_all or not sum(len(a) for a in nodes_all):
        raise InterfaceError("No held component or elastic support carries a reaction - the interface forces need a supported model")
    cloud = np.concatenate(nodes_all)
    ref = np.mean(cloud, axis=0) if ref_point is None else np.asarray(ref_point, float)
    if ref.shape != (3,) or not np.all(np.isfinite(ref)):
        raise InterfaceError("ref_point must be three numbers [x, y, z] (mm)")

    def reduce_modes(Rn: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(K, n, 3) nodal reactions at the points xs → force and moment about ref (K, 3)."""
        F = Rn.sum(axis=1)
        M = np.cross((xs - ref)[None, :, :], Rn).sum(axis=1)
        return F, M

    def reduce_static(Rn: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return Rn.sum(axis=0), np.cross(xs - ref, Rn).sum(axis=0)

    def make(name: str, W: np.ndarray | None, springs: list[str]) -> InterfaceGroup:
        F = np.zeros((K, 3))
        M = np.zeros((K, 3))
        Fs = np.zeros((3, 3))
        Ms = np.zeros((3, 3))
        n_nodes = 0
        if W is not None:
            sel = np.nonzero(np.abs(W).sum(axis=(1, 2)) > 0)[0]
            if len(sel):
                xs = X[cn[sel]]
                Wn = W[sel]
                f_, m_ = reduce_modes(np.einsum("nij,knj->kni", Wn, RF[:, sel]), xs)       # only the held components of the nodal reaction K·φ (rotated onto the held directions)
                F += f_
                M += m_
                for d in range(3):
                    # the base-excitation static state is the NEGATIVE of the gravity state (response.residual_vectors): the support pushes the structure with +M·a
                    rs = np.einsum("nij,nj->ni", Wn, -(np.asarray(statics[d]["RF"], float) - Fg[d])[sel])
                    fs, ms = reduce_static(rs, xs)
                    Fs[d] += fs
                    Ms[d] += ms
                n_nodes += len(sel)
        for sel_name in springs:
            Rm = sp_modes[sel_name]
            nz = sp_nz[sel_name]
            if not len(nz):
                continue
            idx = spr.idx[sel_name][nz]
            xs = X[idx]
            f_, m_ = reduce_modes(Rm[:, nz], xs)
            F += f_
            M += m_
            for d in range(3):
                fs, ms = reduce_static(-sp_static[d][sel_name][nz], xs)
                Fs[d] += fs
                Ms[d] += ms
            n_nodes += len(idx)
        return InterfaceGroup(name, F, M, Fs, Ms, n_nodes)

    groups = {ALL_GROUP: make(ALL_GROUP, W_all if has_held else None, elastic)}
    for name, W in W_sel.items():
        groups[name] = make(name, W, [])
    for sel in elastic:
        groups[f"{sel} (elastic support)"] = make(f"{sel} (elastic support)", None, [sel])
    if getattr(basis, "rf_from_inertia", False):
        warnings.append("Prestressed basis: the resultant interface reaction is exact, its split between supports is approximate")
    mi = ModalInterface(np.asarray(basis.freq, float), np.asarray(basis.gamma, float)[:, :3], np.asarray(basis.eff_mass, float)[:, :3], float(basis.model_mass), ref,
                        groups, {"engine": getattr(basis, "engine", ""), "modes": int(K)}, warnings)
    chk = mi.static_mass_check()
    mi.info["static_mass_check"] = chk
    worst = max(chk["relative_error"] + chk["off_axis_relative"])
    if worst > 5e-3:
        mi.warnings.append(f"The static interface reaction of a unit base acceleration differs from the model mass by {100 * worst:.2f} % (> 0.5 %): "
                           "part of the load path leaves the held nodes (e.g. a rigid remote support or a bolt) or the mesh is poor")
    return mi


def run_interface(study: Any, mesh: Any, workdir: str | Path, *, modes: int | None = None, f_max: float | None = None, ref_point: Any = None,
                  log: Callable[[str], None] | None = None, on_line: Any = None, cancel: Any = None) -> ModalInterface:
    """Modal basis of the study (the direct path where it can, otherwise CalculiX) + the static reactions of the unit base accelerations → `ModalInterface`."""
    from ..study import Gravity, LoadCase
    from .modal import run_basis, static_solutions
    log = log or (lambda _s: None)
    cases = [LoadCase(f"_if_{a}", [Gravity(vector=[1.0 if i == d else 0.0 for i in range(3)])]) for d, a in enumerate(AXES)]
    wd = Path(workdir)
    # the interface forces need the modal reactions, not the modal stresses (about half of the recovery): they are left out; the basis itself is the one of the dynamics
    # when the study ran it before (the state cache)
    basis = run_basis(study, mesh, wd, log, on_line, cancel, modes=modes, f_max=f_max, job="if_modal", static_cases=cases, with_stress=False, mapped=True)
    statics = basis.statics
    if statics is None or len(statics) != 3:
        from ..gravity_states import static_states
        statics = static_states(study, mesh, cases, wd, log, on_line, cancel)
        if statics is None:
            log("Interface forces: static reactions of the unit base accelerations (CalculiX)")
            statics = static_solutions(study, mesh, cases, wd, log, on_line, cancel)
    return build_interface(study, mesh, basis, statics, ref_point)

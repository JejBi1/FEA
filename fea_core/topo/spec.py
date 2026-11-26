"""Specification of the topology optimisation (BESO / SIMP)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

MAX_SYMMETRY_PLANES = 3
DEFAULT_OVERHANG_ANGLE = 45.0


def axis_vector(v) -> list[float] | None:
    """'x' / '+y' / '-z' (also with the Unicode minus) → the unit vector of the axis; anything else → None."""
    if not isinstance(v, str):
        return None
    t = v.strip().lower().replace("−", "-")
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    if t not in ("x", "y", "z"):
        return None
    out = [0.0, 0.0, 0.0]
    out["xyz".index(t)] = sign
    return out


def _vec3(v, what: str) -> list[float]:
    """A finite vector of 3 numbers (as a list of floats); an axis letter ('x', '-z') is accepted too."""
    ax = axis_vector(v)
    if ax is not None:
        return ax
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a list of 3 numbers, entered: {v!r}") from None
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise ValueError(f"{what} must be a list of 3 finite numbers, entered: {v!r}")
    return out


def _nonzero_vec3(v, what: str) -> list[float]:
    out = _vec3(v, what)
    if math.sqrt(sum(x * x for x in out)) < 1e-12:
        raise ValueError(f"{what} must be a nonzero vector")
    return out


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def normalize_symmetry(planes) -> list[dict]:
    """Checks the symmetry planes [{"point": [x, y, z] | None, "normal": [nx, ny, nz]}] and returns them as clean dicts.

    A plane without "point" goes through the centre of the design domain (resolved when the optimiser is created). At most 3 planes, mutually
    perpendicular (the group of mirror images is then simple: each mirror acts on one coordinate of the folded position).
    """
    if planes is None:
        return []
    if isinstance(planes, dict):
        planes = [planes]
    out: list[dict] = []
    for i, p in enumerate(planes):
        if not isinstance(p, dict):
            raise ValueError(f"symmetry[{i}] must be {{'point': [x, y, z], 'normal': [nx, ny, nz]}}, entered: {p!r}")
        if p.get("normal") is None:
            raise ValueError(f"symmetry[{i}] needs a 'normal'")
        pt = p.get("point")
        out.append({"point": None if pt is None else _vec3(pt, f"symmetry[{i}].point"),
                    "normal": _nonzero_vec3(p["normal"], f"symmetry[{i}].normal")})
    if len(out) > MAX_SYMMETRY_PLANES:
        raise ValueError(f"at most {MAX_SYMMETRY_PLANES} symmetry planes are supported, entered: {len(out)}")
    units = [_unit(p["normal"]) for p in out]
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if abs(sum(a * b for a, b in zip(units[i], units[j]))) > 1e-6:
                raise ValueError("symmetry planes must be mutually perpendicular (the normals of planes "
                                 f"{i + 1} and {j + 1} are not)")
    return out


def normalize_extrusion(ext) -> dict | None:
    """{"direction": [x, y, z]} (+ optional "bin_size" in mm) or None."""
    if ext is None:
        return None
    if not isinstance(ext, dict):
        raise ValueError(f"extrusion must be {{'direction': [x, y, z]}}, entered: {ext!r}")
    if ext.get("direction") is None:
        raise ValueError("extrusion needs a 'direction'")
    out: dict = {"direction": _nonzero_vec3(ext["direction"], "extrusion.direction")}
    if ext.get("bin_size") is not None:
        try:
            b = float(ext["bin_size"])
        except (TypeError, ValueError):
            raise ValueError(f"extrusion.bin_size must be a number > 0, entered: {ext['bin_size']!r}") from None
        if not (b > 0.0 and math.isfinite(b)):
            raise ValueError(f"extrusion.bin_size must be > 0, entered: {ext['bin_size']}")
        out["bin_size"] = b
    return out


def normalize_overhang(ov) -> dict | None:
    """{"build_dir": [x, y, z] | "from_print", "angle_deg": 45, "base": "min" | <selection>} or None (defaults: +Z, 45 deg, "min")."""
    if ov is None:
        return None
    if not isinstance(ov, dict):
        raise ValueError(f"overhang must be {{'build_dir': ..., 'angle_deg': 45, 'base': 'min'}}, entered: {ov!r}")
    bd = ov.get("build_dir", [0.0, 0.0, 1.0])
    if not (isinstance(bd, str) and bd == "from_print"):
        bd = _nonzero_vec3(bd, "overhang.build_dir")             # a vector or an axis letter
    raw_ang = ov.get("angle_deg", DEFAULT_OVERHANG_ANGLE)
    try:
        ang = float(raw_ang)
    except (TypeError, ValueError):
        raise ValueError(f"overhang.angle_deg must be a number in the interval [5, 85], entered: {raw_ang!r}") from None
    if not (5.0 <= ang <= 85.0):
        raise ValueError(f"overhang.angle_deg must be in the interval [5, 85] (angle of the surface from the build plate), entered: {ang}")
    base = ov.get("base", "min")
    if not isinstance(base, str) or not base:
        raise ValueError(f"overhang.base must be 'min' or the name of a selection, entered: {base!r}")
    return {"build_dir": bd, "angle_deg": ang, "base": base}


@dataclass
class TopologyOptSpec:
    """Parameters of the topology optimisation.

    target_volume_fraction: target volume fraction (e.g. 0.5 = 50 % of the initial volume)
    evolution_rate: rate of the evolution of the volume per iteration (e.g. 0.02 = 2 % per step)
    filter_radius: radius of the filter of the sensitivities in mm
    penalty: penalty exponent of the stiffness p (E = E0 * rho^p)
    max_iterations: maximum number of iterations
    non_design_selections: list of names of selections (faces, edges, points) whose elements must not be removed
    min_density: minimum density of the void elements (e.g. 1e-3 to keep the stiffness matrix regular)
    target_step: index of the load state for the calculation of the sensitivities (only when all_load_cases = False)
    all_load_cases: True = sensitivities from the sum of the compliances of all load states
    method: optimisation method ("beso" | "simp")
    tol: convergence criterion (e.g. relative change of the compliance or the densities)

    Manufacturing constraints (all optional, element level, only for method "beso"; files without them load unchanged):
    symmetry: up to 3 mutually perpendicular mirror planes [{"point": [x, y, z], "normal": [nx, ny, nz]}] – the density is identical on both sides
    extrusion: {"direction": [x, y, z]} – the shape is a prism along the direction (milling / profile), one decision per column of elements
    overhang: {"build_dir": [x, y, z] | "from_print", "angle_deg": 45, "base": "min" | <selection>} – additive manufacturing without supports:
        a solid element needs support below it within a cone of 90° − angle_deg around −build_dir (APPROXIMATE, element level)
    compare_unconstrained: also run the optimisation without the constraints (same settings) to report the compliance penalty
    """
    target_volume_fraction: float = 0.5
    evolution_rate: float = 0.02
    filter_radius: float = 6.0
    penalty: float = 3.0
    max_iterations: int = 50
    non_design_selections: list[str] = field(default_factory=list)
    min_density: float = 1e-3
    target_step: int = 0
    all_load_cases: bool = True
    method: str = "beso"
    tol: float = 1e-3
    symmetry: list[dict] = field(default_factory=list)
    extrusion: dict | None = None
    overhang: dict | None = None
    compare_unconstrained: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.target_volume_fraction <= 1.0):
            raise ValueError(f"target_volume_fraction must be in the interval (0, 1], entered: {self.target_volume_fraction}")
        if self.evolution_rate <= 0.0 or self.evolution_rate > 1.0:
            raise ValueError(f"evolution_rate must be in the interval (0, 1], entered: {self.evolution_rate}")
        if self.filter_radius <= 0.0:
            raise ValueError(f"filter_radius must be > 0, entered: {self.filter_radius}")
        if self.penalty < 1.0:
            raise ValueError(f"penalty exponent must be >= 1.0, entered: {self.penalty}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, entered: {self.max_iterations}")
        if self.min_density < 0.0:
            raise ValueError(f"min_density must be >= 0.0, entered: {self.min_density}")
        self.symmetry = normalize_symmetry(self.symmetry)
        self.extrusion = normalize_extrusion(self.extrusion)
        self.overhang = normalize_overhang(self.overhang)
        if self.has_constraints and str(self.method).lower() != "beso":
            raise ValueError("manufacturing constraints (symmetry / extrusion / overhang) are only available for the discrete BESO method")

    @property
    def has_constraints(self) -> bool:
        """True when some manufacturing constraint is set."""
        return bool(self.symmetry) or self.extrusion is not None or self.overhang is not None

    def unconstrained(self) -> "TopologyOptSpec":
        """The same optimisation without the manufacturing constraints (for the reference run)."""
        d = self.to_dict()
        d.update(symmetry=[], extrusion=None, overhang=None, compare_unconstrained=False)
        return TopologyOptSpec.from_dict(d)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyOptSpec":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

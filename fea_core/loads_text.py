"""Description of loads in Czech (MCP, GUI, report): force on faces / edges / points, local system, variable pressure,
rotation. The numbers are formatted by `num` (the report has its own – decimal comma).
"""
from __future__ import annotations

import math

import numpy as np


def _num(v: float) -> str:
    return f"{float(v):.4g}"


def vec_text(v, num=_num) -> str:
    return "(" + "; ".join(num(float(x)) for x in v) + ")"


def entity_of(l, st) -> str:
    """Kind of load selection: faces | edges | points (without a study a face)."""
    sel = st.selections.get(getattr(l, "selection", None)) if st is not None else None
    return sel.entity if sel is not None else "faces"


def csys_suffix(l, st) -> str:
    name = getattr(l, "csys", None)
    if not name:
        return ""
    cs = st.coordinate_systems.get(name) if st is not None else None
    if cs is not None and cs.kind == "cylindrical":
        return f" in system “{name}” (r, θ, z)"
    return f" in system “{name}”"


def force_kind(l, st) -> str:
    return {"faces": "force (uniform over face)", "edges": "force on edge (uniform along length)",
            "points": "force at point"}.get(entity_of(l, st), "force")


def pressure_text(l, num=_num) -> str:
    if not getattr(l, "variable", False):
        return f"p = {num(l.value)} MPa"
    g = np.asarray(l.gradient, float)
    o = l.origin if l.origin is not None else [0.0, 0.0, 0.0]
    if l.clip:
        gn = float(np.linalg.norm(g))
        rho_kg = gn / 9810.0 * 1e12                   # ρ·g → ρ [kg/m³] at g = 9810 mm/s²
        base = f"hydrostatic: p = {num(l.value)} MPa + {num(gn)} MPa/mm × depth below the surface"
        return (base + f" (level at point {vec_text(o, num)} mm, weight in direction {vec_text(g / gn, num)}; ≈ liquid {num(rho_kg)} kg/m³ at g = 9.81 m/s²)")
    return f"linear: p = {num(l.value)} MPa + {vec_text(g, num)} MPa/mm · (x − {vec_text(o, num)} mm)"


def rotation_text(l, num=_num) -> str:
    parts = []
    if l.omega:
        parts.append(f"ω = {num(l.omega)} rad/s ({num(l.omega * 60 / (2 * math.pi))} rpm)")
    if l.alpha:
        parts.append(f"α = {num(l.alpha)} rad/s²")
    who = "parts " + ", ".join(l.parts) if l.parts else "all parts"
    return (", ".join(parts) or "no rotation") + f"; axis at point {vec_text(l.point, num)} mm, direction {vec_text(l.axis, num)}; {who}"


def inertia_text(l, num=_num) -> str:
    """Inertia forces of the rigid motion (study.Inertia)."""
    parts = []
    for lab, v, u in (("a₀", l.accel, "mm/s²"), ("ω", l.omega, "rad/s"), ("α", l.alpha, "rad/s²")):
        if any(float(x) for x in v):
            parts.append(f"{lab} = {vec_text(v, num)} {u}")
    who = "parts " + ", ".join(l.parts) if l.parts else "all parts"
    return (", ".join(parts) or "no motion") + f"; point {vec_text(l.point, num)} mm; {who}"


def load_text(l, st=None, num=_num) -> str:
    """Value of a load with units (one sentence) – per kind fea_core.load_kinds (LoadKind.describe)."""
    from .load_kinds import load_handler
    h = load_handler(l)
    if h is None or h.describe is None:
        return getattr(l, "kind", "")
    return h.describe(l, st, num)


def constraint_text(c, st=None, num=_num) -> str:
    """Description of a constraint with units (one sentence) – MCP, GUI, report; per kind fea_core.load_kinds."""
    from .load_kinds import constraint_handler
    h = constraint_handler(c)
    if h is None or h.describe is None:
        return getattr(c, "kind", "")
    return h.describe(c, st, num)

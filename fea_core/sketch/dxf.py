"""Import of lines, arcs, circles and polylines from a DXF drawing into a sketch (optional module `sketch` = ezdxf, MIT).

The DXF is read with `ezdxf` (imported only here, lazily). Taken from the model space: LINE, ARC, CIRCLE, LWPOLYLINE and 2D POLYLINE
(a closed polyline without arcs stays one closed `polyline` entity, a polyline with arc segments (bulge) becomes lines and arcs).
Everything else (SPLINE, ELLIPSE, HATCH, TEXT, DIMENSION, blocks…) is counted and reported, not imported. The drawing is taken as
2D: the Z coordinate is ignored (a drawing that is not flat is reported); an extrusion direction −Z (mirrored OCS) is handled,
other directions are skipped. Units: `$INSUNITS` of the file (inch, foot, cm, m, …) → mm; a file without units is taken as mm,
`scale` overrides. The coordinates are written as numbers (10 significant digits); the points of the sketch are merged by the
usual tolerance of the sketch (1e-6 of its size).
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

from .expr import SketchError
from .model import MAX_ENTITIES, Sketch

MAX_BYTES = 16 << 20                       # a drawing of a part; ezdxf holds the whole file as Python objects (memory)
MIN_BULGE = 1e-6                           # |bulge| below it is a straight segment (the arc radius would exceed 1e6 x the chord)
UNIT_MM = {1: 25.4, 2: 304.8, 3: 1609344.0, 4: 1.0, 5: 10.0, 6: 1000.0, 7: 1e6, 8: 25.4e-6, 9: 0.0254, 10: 914.4,
           13: 1e-3, 14: 100.0, 15: 1e4, 16: 1e5, 17: 1e9}
UNIT_NAME = {0: "unitless", 1: "inch", 2: "foot", 3: "mile", 4: "mm", 5: "cm", 6: "m", 7: "km", 8: "microinch", 9: "mil", 10: "yard",
             13: "micron", 14: "dm", 15: "dam", 16: "hm", 17: "gm"}


def _ezdxf():
    try:
        import ezdxf
        return ezdxf
    except ImportError:
        raise SketchError("The DXF import needs the module 'sketch' (package ezdxf): Tools → Program modules…, tick "
                          "\"Sketch (DXF import)\"") from None


def _num(v: float) -> float:
    return float(f"{v:.10g}")


def _finite(*vals) -> bool:
    """All the numbers are finite (a corrupt or hostile DXF can hold nan / inf)."""
    return all(math.isfinite(float(v)) for v in vals)


def import_dxf(sk: Sketch, path: str | Path, layers: list[str] | None = None, scale: float | None = None,
               max_entities: int = 5000) -> dict:
    """Adds the curves of the DXF file `path` to the sketch; returns the summary {imported, skipped, units, scale, …}."""
    if sk.kind != "twod" and sk.kind != "frame":
        raise SketchError("Unknown sketch kind")
    ezdxf = _ezdxf()
    p = Path(path)
    if not p.is_file():
        raise SketchError(f"DXF file does not exist: {p}")
    if p.stat().st_size > MAX_BYTES:
        raise SketchError(f"The DXF file is too large ({p.stat().st_size >> 20} MB, at most {MAX_BYTES >> 20} MB)")
    try:
        doc = ezdxf.readfile(str(p))
    except (MemoryError, RecursionError):
        raise SketchError("The DXF file is too large or too deeply nested to be read") from None
    except (OSError, ezdxf.DXFError, UnicodeDecodeError, ValueError, LookupError, TypeError, AttributeError, ArithmeticError, struct.error) as e:
        raise SketchError(f"The DXF file cannot be read: {e}") from None
    units = int(doc.header.get("$INSUNITS", 0) or 0)
    if scale is None:
        scale = UNIT_MM.get(units, 1.0)
    if not (math.isfinite(scale) and scale > 0):
        raise SketchError("scale must be a positive number")
    want = {x.lower() for x in layers} if layers else None
    cap = min(int(max_entities), MAX_ENTITIES - len(sk.entities))
    counts = {"lines": 0, "arcs": 0, "circles": 0, "polylines": 0}
    skipped: dict[str, int] = {}
    layer_names: set[str] = set()
    zs: list[float] = []
    made = 0
    saved = sk.copy()

    def skip(what: str) -> None:
        skipped[what] = skipped.get(what, 0) + 1

    def P(x, y):
        return [_num(x * scale), _num(y * scale)]

    def add(*a, **kw):
        nonlocal made
        if made >= cap:
            raise SketchError(f"The drawing has more than {cap} curves – too many for a sketch (select layers or simplify the DXF)")
        sk.add_entity(*a, **kw)
        made += 1
    try:
        for e in doc.modelspace():
            t = e.dxftype()
            layer = str(e.dxf.get("layer", "0"))
            if want is not None and layer.lower() not in want:
                continue
            ext = e.dxf.get("extrusion", (0, 0, 1))
            flip = False
            if t in ("ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE"):
                ez = float(ext[2]) if ext is not None else 1.0
                if abs(abs(ez) - 1.0) > 1e-9 and t != "POLYLINE":
                    skipped[t + " (tilted plane)"] = skipped.get(t + " (tilted plane)", 0) + 1
                    continue
                flip = ez < 0
            if t == "LINE":
                a, b = e.dxf.start, e.dxf.end
                if not _finite(*a, *b):
                    skip("LINE (non-finite)")
                    continue
                zs += [a[2], b[2]]
                if math.hypot(b[0] - a[0], b[1] - a[1]) * scale < 1e-9:
                    continue
                add("line", [P(a[0], a[1]), P(b[0], b[1])])
                counts["lines"] += 1
            elif t == "CIRCLE":
                c = e.dxf.center
                if not _finite(*c, e.dxf.radius) or float(e.dxf.radius) <= 0.0:
                    skip("CIRCLE (non-finite or zero radius)")
                    continue
                zs.append(c[2])
                add("circle", center=P(-c[0] if flip else c[0], c[1]), radius=_num(float(e.dxf.radius) * scale))
                counts["circles"] += 1
            elif t == "ARC":
                c, r = e.dxf.center, float(e.dxf.radius)
                a0, a1 = float(e.dxf.start_angle), float(e.dxf.end_angle)
                if not _finite(*c, r, a0, a1) or r <= 0.0:
                    skip("ARC (non-finite or zero radius)")
                    continue
                zs.append(c[2])
                if flip:
                    a0, a1 = 180.0 - a1, 180.0 - a0
                cx = -c[0] if flip else c[0]
                s0, s1 = math.radians(a0), math.radians(a1)
                if abs((a1 - a0) % 360.0) < 1e-9:
                    skipped["ARC (zero sweep)"] = skipped.get("ARC (zero sweep)", 0) + 1
                    continue
                add("arc_center", [P(cx + r * math.cos(s0), c[1] + r * math.sin(s0)), P(cx + r * math.cos(s1), c[1] + r * math.sin(s1))],
                    center=P(cx, c[1]))
                counts["arcs"] += 1
            elif t in ("LWPOLYLINE", "POLYLINE"):
                if t == "POLYLINE" and (e.is_3d_polyline or e.is_polygon_mesh or e.is_poly_face_mesh):
                    skipped["POLYLINE (3D / mesh)"] = skipped.get("POLYLINE (3D / mesh)", 0) + 1
                    continue
                if t == "LWPOLYLINE":
                    pts = [(float(x), float(y), float(b)) for x, y, b in e.get_points("xyb")]
                else:
                    pts = [(float(v.dxf.location[0]), float(v.dxf.location[1]), float(v.dxf.get("bulge", 0.0))) for v in e.vertices]
                if not all(_finite(x, y, b) for x, y, b in pts):
                    skip(t + " (non-finite)")
                    continue
                if flip:
                    pts = [(-x, y, -b) for x, y, b in pts]
                closed = bool(e.closed) if t == "LWPOLYLINE" else bool(e.is_closed)
                if len(pts) < 2:
                    continue
                if not any(abs(b) >= MIN_BULGE for _, _, b in pts):
                    add("polyline", [P(x, y) for x, y, _ in pts], closed=closed and len(pts) > 2)
                    counts["polylines"] += 1
                    continue
                from ezdxf.math import Vec2, bulge_to_arc
                m = len(pts) if closed else len(pts) - 1
                for i in range(m):
                    x0, y0, b0 = pts[i]
                    x1, y1, _ = pts[(i + 1) % len(pts)]
                    if math.hypot(x1 - x0, y1 - y0) * scale < 1e-9:
                        continue
                    if abs(b0) < MIN_BULGE:
                        add("line", [P(x0, y0), P(x1, y1)])
                        counts["lines"] += 1
                    else:
                        cen, _, _, _ = bulge_to_arc(Vec2(x0, y0), Vec2(x1, y1), b0)
                        ends = [P(x0, y0), P(x1, y1)] if b0 > 0 else [P(x1, y1), P(x0, y0)]       # counter-clockwise from the first
                        add("arc_center", ends, center=P(cen.x, cen.y))
                        counts["arcs"] += 1
            else:
                skipped[t] = skipped.get(t, 0) + 1
                continue
            layer_names.add(layer)
    except SketchError:
        sk.restore(saved)              # nothing of a half-imported drawing stays
        raise
    except (MemoryError, RecursionError):
        sk.restore(saved)
        raise SketchError("The DXF drawing is too large or too deeply nested to be converted") from None
    except (ValueError, TypeError, LookupError, AttributeError, ArithmeticError, struct.error, ezdxf.DXFError) as e:
        sk.restore(saved)
        raise SketchError(f"The DXF drawing cannot be converted: {e}") from None
    warnings: list[str] = []
    if zs and (max(zs) - min(zs)) * scale > 1e-6:
        warnings.append("The drawing is not flat (Z coordinates differ) – only X and Y are used")
    if skipped:
        warnings.append("Not imported: " + ", ".join(f"{n}× {k}" for k, n in sorted(skipped.items())))
    if made == 0:
        raise SketchError("The DXF file has no lines, arcs, circles or polylines" + (" on the chosen layers" if want else "")
                          + (f" (found: {', '.join(f'{n}× {k}' for k, n in sorted(skipped.items()))})" if skipped else ""))
    return {"imported": made, **counts, "skipped": skipped, "units": UNIT_NAME.get(units, str(units)), "scale_to_mm": scale,
            "layers": sorted(layer_names), "file": os.path.basename(str(p)), "warnings": warnings}

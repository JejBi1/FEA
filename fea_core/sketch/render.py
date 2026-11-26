"""Picture of a sketch (PNG, matplotlib, headless): curves with their ids, regions filled, holes hatched, points, supports.

Used by the MCP tools (`fea_sketch_build` shows what the agent drew) and by the tests; the GUI editor draws with Qt itself.
"""
from __future__ import annotations

from pathlib import Path

from .expr import SketchError
from .model import Resolved, Sketch, resolve

PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#eeca3b", "#9d755d"]


def render_png(sk: Sketch, path: str | Path, build=None, width: int = 1000, height: int = 760, show_points: bool = True,
               problem: SketchError | None = None) -> Path:
    """Draws the sketch. `build` (SketchBuild, 2D) adds the regions (filled, named); `problem` marks the entities of an error in red."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
    except ImportError:
        raise SketchError("The picture of the sketch needs the module 'plots' (Tools → Program modules…)") from None
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    bad = set(problem.entities) if problem is not None else set()
    try:
        res: Resolved | None = resolve(sk)
    except SketchError as e:
        res = None
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha="center", va="center", wrap=True, color="#c00000", parse_math=False)
    if res is not None:
        if build is not None and getattr(build, "cons", None) is not None:
            for i, r in enumerate(build.cons.regions):
                if not r.rings:
                    continue
                verts, codes = [], []
                for ring in r.rings:
                    verts += list(ring) + [ring[0]]
                    codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1) + [MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), facecolor=PALETTE[i % len(PALETTE)], alpha=0.18, edgecolor="none"))
                cx = sum(p[0] for p in r.rings[0]) / len(r.rings[0])
                cy = sum(p[1] for p in r.rings[0]) / len(r.rings[0])
                ax.text(cx, cy, r.name, ha="center", va="center", fontsize=9, color="#333333", alpha=0.8, parse_math=False)   # user text: no mathtext
        for s in res.segs:
            xs, ys = zip(*s.polyline(2.0))
            red = s.entity in bad
            ax.plot(xs, ys, color="#d62728" if red else ("#999999" if s.construction else "#1f3a5f"), lw=2.2 if red else 1.6,
                    ls="--" if s.construction else "-", solid_capstyle="round")
            mx, my = s.mid()
            ax.annotate(s.id, (mx, my), fontsize=8, color="#d62728" if red else "#7a3b00", xytext=(3, 3), textcoords="offset points",
                        parse_math=False)
        if show_points:
            for pid, (x, y) in res.pts.items():
                ax.plot([x], [y], "o", ms=3, color="#222222")
                if len(res.pts) <= 40:
                    ax.annotate(pid, (x, y), fontsize=7, color="#555555", xytext=(-9, -9), textcoords="offset points", parse_math=False)
        for sup in sk.supports:
            xy = res.pts.get(sup.point)
            if xy is not None:
                ax.plot([xy[0]], [xy[1]], marker="s" if sup.kind == "fixed" else "^", ms=9, mfc="none", mec="#2ca02c")
        if sk.kind == "twod" and sk.mode == "axisym":
            ax.axvline(0.0, color="#2ca02c", lw=1.0, ls="-.")
            ax.text(0.0, 1.0, "axis", transform=ax.get_xaxis_transform(), color="#2ca02c", fontsize=8, ha="center", va="bottom")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("r [mm]" if sk.kind == "twod" and sk.mode == "axisym" else "x [mm]")
    ax.set_ylabel("axis [mm]" if sk.kind == "twod" and sk.mode == "axisym" else "y [mm]")
    mode = sk.mode if sk.kind == "twod" else "beam system"
    ax.set_title(f"Sketch ({mode}); params: " + ", ".join(f"{k}={v}" for k, v in list(sk.params.items())[:6]), fontsize=9, parse_math=False)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return out


"""Solver protocol of a calculation (`Results.info["solver"]`): what solved the linear system, why, how large it was, how much memory it needed,
how long each phase took and how accurate the solution is. One place for the GUI (Properties card), the Word report and the MCP summary.

The data are written by `solve_large` (direct solver, hypre) and `solve._run_mechanics` (ccx). `rows(p)` gives (label, value) in English; with
`translate=True` every value is translated fragment by fragment (the values are composed of short fragments joined by ", "; the translation engine
does not split such a composed text reliably – a template that starts with a hole may swallow the neighbouring fragments), the labels are
left to the caller. The long reasons (`reason`, `precision_reason`) are rows of their own: a whole sentence matches its template.
"""
from __future__ import annotations

from .i18n import tr


def _gb(v) -> str:
    return f"{v:.1f} GB" if v is not None else "?"


def _sec(v) -> str:
    return f"{v:.1f} s"


def _sci(v) -> str:
    return f"{v:.1e}"


def _count(v) -> str:
    return f"{int(v):,}".replace(",", " ")


# phase of the calculation → text of its time (one f-string per phase: every one is a template of the catalogue)
PHASES = {
    "mesh_import": lambda v: f"mesh import {_sec(v)}",
    "compile": lambda v: f"compiling the assembly kernel (first use) {_sec(v)}",
    "assembly": lambda v: f"matrix assembly {_sec(v)}",
    "load": lambda v: f"reading the system {_sec(v)}",
    "analysis": lambda v: f"analysis {_sec(v)}",
    "factorization": lambda v: f"factorization {_sec(v)}",
    "solution": lambda v: f"solution {_sec(v)}",
    "setup": lambda v: f"AMG setup {_sec(v)}",
}


# constraints of the own assembly (`Results.info["solver"]["constraints"]`, `linear_constraints.build` labels) → text (one f-string per kind: a catalogue template)
CONSTRAINTS = {
    "springs": lambda v: f"{_count(v)} nodes with an elastic support",
    "cylindrical": lambda v: f"{_count(v)} cylindrical support equations eliminated",
    "rigid": lambda v: f"{_count(v)} rigid couplings (extra nodes)",
    "distributing": lambda v: f"{_count(v)} distributing coupling equations eliminated",
    "tie": lambda v: f"{_count(v)} nodes of *TIE tied to the master face",
    "point_mass": lambda v: f"{_count(v)} point masses (statics)",
    "bolts": lambda v: f"{_count(v)} bolts as beam elements between rigid bearing faces",
}


def rows(p: dict | None, translate: bool = False) -> list[tuple[str, str]]:
    """[(label, value)] of the protocol `p` (empty when there is none – results of older versions)."""
    if not p:
        return []
    T = tr if translate else (lambda s: s)

    def join(fragments: list[str]) -> str:
        return ", ".join(T(f) for f in fragments)

    out: list[tuple[str, str]] = [("Solver", str(p.get("engine") or p.get("backend")))]
    if p.get("reason"):
        out.append(("Why this solver", T(str(p["reason"]))))
    size = []
    if p.get("ndof"):
        size.append(f"{_count(p['ndof'])} DOF")
    if p.get("nnz"):
        size.append(f"matrix: {p['nnz'] / 1e6:.1f} M nonzeros ({'upper triangle' if p.get('matrix_format') == 'upper' else 'whole matrix'})")
    if p.get("factor_nnz"):
        size.append(f"factor {p['factor_nnz'] / 1e6:.0f} M entries ({p['fill_per_dof']:.0f} per DOF)")
    if size:
        out.append(("Size", join(size)))
    if p.get("assembler"):
        out.append(("Matrix assembly", T(str(p["assembler"]))))
    c = [CONSTRAINTS[k](v) for k, v in (p.get("constraints") or {}).items() if k in CONSTRAINTS and v]
    if p.get("orthotropic"):
        c.append("orthotropic material (rotated at the Gauss points)")
    if c:
        out.append(("Constraints and materials", join(c)))
    b = p.get("bolts") or {}
    if b.get("preload_states"):                  # the preload of the bolts: unit states (an elongation of a bolt) on the same factorization (joints.own)
        out.append(("Bolt preload", join([f"{_count(b['preload_states'])} unit states of the elongation of a bolt on the same factorization"])))
    if p.get("precision"):
        prec = [f"{p['precision']} factor"]
        if p["precision"] == "float32":
            its = [c.get("iterations") for c in p.get("cases", []) if c.get("iterations") is not None]
            prec.append(f"refined in float64 ({min(its)}–{max(its)} iterations)" if its else "refined in float64")
        out.append(("Precision", join(prec)))
        if p.get("precision_reason"):
            out.append(("Why this precision", T(str(p["precision_reason"]))))
    m = p.get("memory") or {}
    if m.get("peak_gb") is not None or m.get("estimate_gb") is not None:
        bits = []
        if m.get("estimate_gb") is not None:
            bits.append(f"plan {_gb(m['estimate_gb'])}")
        if m.get("growth_gb") is not None:
            bits.append(f"measured {_gb(m['growth_gb'])}")
        if m.get("peak_gb") is not None:
            bits.append(f"peak of the process {_gb(m['peak_gb'])}")
        if m.get("free_gb") is not None:
            bits.append(f"free at the start {_gb(m['free_gb'])}")
        out.append(("Memory of the solver", join(bits)))
    t = [PHASES[k](v) for k, v in (p.get("times") or {}).items() if k in PHASES and v is not None]
    if t:
        out.append(("Time per phase", join(t)))
    cases = p.get("cases") or []
    bw = [c["backward_error"] for c in cases if c.get("backward_error") is not None]
    if bw:
        out.append(("Accuracy", T(f"backward error up to {_sci(max(bw))}")))
    rr = [c["rel_residual"] for c in cases if c.get("rel_residual") is not None and c.get("backward_error") is None]
    if rr:
        acc = [f"relative residual up to {_sci(max(rr))}"]
        its = [c["iterations"] for c in cases if c.get("iterations") is not None]
        if its:
            acc.append(f"iterations {min(its)}–{max(its)}")
        out.append(("Accuracy", join(acc)))
    fb = p.get("fallback")
    if fb:
        out.append(("Fallback", T(f"{fb.get('from')} → {fb.get('to')}: {fb.get('reason')}")))
    return out


def brief(p: dict | None) -> dict:
    """The protocol without the per-state lists (MCP summary): what was used, why, memory and time."""
    if not p:
        return {}
    keys = ("backend", "engine", "reason", "ndof", "precision", "memory", "times", "fallback")
    d = {k: p[k] for k in keys if p.get(k) is not None}
    bw = [c["backward_error"] for c in p.get("cases", []) if c.get("backward_error") is not None]
    if bw:
        d["backward_error_max"] = max(bw)
    return d

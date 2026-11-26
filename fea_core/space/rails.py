"""CubeSat deployer rails (F14 stage S4): the support helper and the interface checks.

A rail satellite sits in the deployer on its rails: the deployer walls touch the outer faces of the rails (they hold the satellite laterally – one wall normal per
face) and the rail ends are held axially by the deployer's pusher plate / door at one end or both. `apply_rail_supports` writes exactly that into the study:
  * for every rail part and every lateral side (±y, ±z for a deployment axis x) the faces of the rail that lie on the outer envelope of all rails (plane at the extreme
    coordinate, outward normal along the side) get a `Displacement` constraint of ONLY the wall-normal component (frictionless wall; the wall cannot hold tangentially);
  * the end faces of the rails at the chosen end(s) of the deployment axis get a `Displacement` constraint of the axial component.
Every earlier rail support of the study (found by the setup stored with it, `Study.space.extra["rails"]`: the part, kind, axis and sign of every support – the selection
NAMES are never parsed) is removed first, whatever its kind or rail, so a new apply with another axial end, `axial='none'` or fewer rails leaves nothing stale.
`check_rails` reads the solved study: the reaction of every rail support per load state against the limit of the requirement set (`rail_reaction_limit_N`,
`rail_axial_reaction_limit_N`), the lateral deflection of the rails against the clearance limit (`rail_lateral_deflection_limit_mm`), and flags a wall that would have to
PULL in ANY state (a linear support is bilateral, a wall or a pusher plate is not). THE LIMITATION OF THE LINEAR MODEL: the two opposite supports of an axis - the two walls
of a lateral axis, the two end plates of the deployment axis with axial='both' - split a load between them (each carries half, one of them pulling); a real wall / plate
cannot pull, so the whole load goes to the pushing side. The margin therefore uses the conservative force on the pushing side, |push| + |pull of the opposite side| (shared
among the pushing supports in proportion to their push) - exact for a rigid body between them, not a contact analysis. With a single held end (axial='min' / 'max') an
end that the load would pull away from is flagged (there is no opposite end to take the load: the model is not a deployer). A pull smaller than `pull_rel_tol` (a NUMERICAL setting, relative to the sum of all rail reactions of the state) is noise and ignored. Limits come from the
confirmed requirement set; a value passed directly is used but never confirmed – the verdict is then UNCONFIRMED_REQUIREMENTS. A rail without a support row gives no verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .notching import FAIL, PASS, UNCONFIRMED
from .s4_limits import S4Limits

AXES = ("x", "y", "z")
AXIAL_MODES = ("none", "min", "max", "both")


class RailError(ValueError):
    """The rail inputs are inconsistent."""


@dataclass
class RailSpec:
    rail_parts: list[str] = field(default_factory=list)       # names of the rail parts (they must exist in the model)
    axis: str = "x"                                           # deployment axis
    axial: str = "none"                                       # which rail ends are held axially: none | min | max | both (a project decision: no default support)
    tol_mm: float = 1e-3
    pull_rel_tol: float = 1e-3                                # NUMERICAL setting: a wall pull below this fraction of the sum of the rail reactions of the state is noise
    reaction_limit_N: float | None = None                     # direct (unconfirmed) overrides of the requirement-set limits
    axial_reaction_limit_N: float | None = None
    lateral_deflection_limit_mm: float | None = None


def _unit(axis: str) -> np.ndarray:
    if axis not in AXES:
        raise RailError(f"The axis must be one of {', '.join(AXES)}")
    e = np.zeros(3)
    e[AXES.index(axis)] = 1.0
    return e


def rail_faces(parts: list, spec: RailSpec) -> list[dict]:
    """The support faces of the rails: [{"name", "part", "kind": "wall" | "axial", "axis", "sign", "component", "faces": [FaceSelector]}]."""
    from ..geometry import bbox
    from ..selectors import find_faces
    if spec.axial not in AXIAL_MODES:
        raise RailError(f"axial must be one of {', '.join(AXIAL_MODES)}")
    ax = AXES.index(spec.axis)
    _unit(spec.axis)
    by = {p.name: p for p in parts}
    names = list(spec.rail_parts)
    if not names:
        raise RailError("Name the rail parts (rail_parts)")
    missing = [n for n in names if n not in by]
    if missing:
        raise RailError(f"Unknown rail part(s): {', '.join(missing)} (parts: {', '.join(by)})")
    boxes = {n: bbox(by[n].shape) for n in names}
    lo = np.min([b[:3] for b in boxes.values()], axis=0)
    hi = np.max([b[3:] for b in boxes.values()], axis=0)
    out: list[dict] = []
    for a, aname in enumerate(AXES):
        if a == ax:
            continue
        for sign, ext in ((+1, hi[a]), (-1, lo[a])):
            n = np.zeros(3)
            n[a] = sign
            for part in names:
                plane: dict[str, Any] = {aname: float(ext)}
                faces = find_faces([by[part]], normal=n, tol=spec.tol_mm, **plane)
                if faces:
                    out.append({"name": f"Rail {part} {'+' if sign > 0 else '-'}{aname}", "part": part, "kind": "wall", "axis": aname, "sign": sign, "component": a,
                                "faces": faces})
    ends = []
    if spec.axial in ("min", "both"):
        ends.append((-1, lo[ax]))
    if spec.axial in ("max", "both"):
        ends.append((+1, hi[ax]))
    for sign, ext in ends:
        n = np.zeros(3)
        n[ax] = sign
        for part in names:
            plane = {spec.axis: float(ext)}
            faces = find_faces([by[part]], normal=n, tol=spec.tol_mm, **plane)
            if faces:
                out.append({"name": f"Rail {part} axial {'max' if sign > 0 else 'min'}", "part": part, "kind": "axial", "axis": spec.axis, "sign": sign, "component": ax,
                            "faces": faces})
    if not out:
        raise RailError("No rail support face was found: check rail_parts, the deployment axis and that the rails lie on the outer envelope")
    return out


SETUP_KEY = "rails"


def setup_of(study: Any) -> dict:
    """The stored rail setup of the study ({"axis", "axial", "rail_parts", "supports": [{selection, part, kind, axis, sign, component}]}) or {}."""
    sp = getattr(study, "space", None)
    d = sp.extra.get(SETUP_KEY) if sp is not None else None
    return d if isinstance(d, dict) and isinstance(d.get("supports"), list) else {}


def remove_rail_supports(study: Any) -> int:
    """Removes every rail support written earlier by `apply_rail_supports` (constraints and selections of the stored setup; for a project saved before the setup was
    stored: the `Displacement` constraints on selections named 'Rail ...' that end in the pattern the helper writes). Returns the number of constraints removed."""
    import re
    names = {x.get("selection") for x in setup_of(study).get("supports", [])}
    if not names:
        pat = re.compile(r"^Rail .+ (?:[+-][xyz]|axial (?:min|max))$")
        names = {c.selection for c in study.constraints if type(c).__name__ == "Displacement" and pat.match(str(getattr(c, "selection", "")))}
    n0 = len(study.constraints)
    study.constraints = [c for c in study.constraints if getattr(c, "selection", None) not in names]
    for n in names:
        study.selections.pop(n, None)
    sp = getattr(study, "space", None)
    if sp is not None:
        sp.extra.pop(SETUP_KEY, None)
    return n0 - len(study.constraints)


def apply_rail_supports(study: Any, parts: list, spec: RailSpec) -> dict:
    """Creates the selections and the `Displacement` constraints of the rail supports in the study; EVERY earlier rail support is removed first (see the module docstring) and
    the setup (part, kind, axis, sign of each support) is stored with the study. Returns {"supports": [...], "warnings": [...], "removed": n}."""
    from ..study import Displacement
    from .model import SpaceSettings
    found = rail_faces(parts, spec)
    removed = remove_rail_supports(study)
    sup = []
    for f in found:
        study.add_selection(f["name"], faces=f["faces"])
        comp = {0: "ux", 1: "uy", 2: "uz"}[f["component"]]
        fixed: dict[str, Any] = {comp: 0.0}
        study.constraints.append(Displacement(f["name"], **fixed))
        sup.append({"selection": f["name"], "part": f["part"], "kind": f["kind"], "axis": f["axis"], "sign": int(f["sign"]), "component": int(f["component"]),
                    "fixed_component": comp, "faces": len(f["faces"])})
    if study.space is None:
        study.space = SpaceSettings()
    study.space.extra[SETUP_KEY] = {"axis": spec.axis, "axial": spec.axial, "rail_parts": list(spec.rail_parts),
                                    "supports": [{k: v for k, v in x.items() if k != "faces"} for x in sup]}
    warns = []
    if spec.axial == "none":
        warns.append("No axial support at the rail ends (axial='none'): the model may be free along the deployment axis - choose min, max or both")
    warns.append("The wall supports are bilateral (a linear constraint): a wall cannot pull - the check reports a reaction that would have to pull and uses the conservative "
                 "force |push| + |pull of the opposite wall| for the margin")
    return {"supports": sup, "warnings": warns, "removed": removed}


# ------------------------------------------------------------------ the check
@dataclass
class RailRow:
    support: str
    part: str
    kind: str
    state: str                        # the state with the largest effective force
    reaction_N: float                 # compression (+) / pull (-) along the support normal (inward) in that state
    pulls: bool                       # the wall / end plate would have to pull in ANY state
    limit_N: float | None
    mos: float | None
    effective_N: float = 0.0          # the force of the margin: |push| + the share of |pull| of the opposite side (walls and end plates alike)
    pull_states: list[str] = field(default_factory=list)


@dataclass
class RailResult:
    rows: list[RailRow]
    deflection: list[dict]            # per rail part: max lateral deflection [mm], state, limit, MoS
    verdict: str
    verdict_text: str
    inputs: list[dict]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "verdict_text": self.verdict_text, "inputs": self.inputs, "warnings": self.warnings,
                "supports": [r.__dict__ for r in self.rows], "deflection": self.deflection}


def _get(lim: S4Limits, key: str, override: float | None, name: str, inputs: list[dict]) -> tuple[float | None, bool]:
    v, ok, src = lim.get(key, override)
    inputs.append({"name": name, "key": key, "value": v, "source": src or "not available", "confirmed": ok})
    return v, ok


LINEAR_MODEL_NOTE = ("walls and end plates are bilateral linear supports (a real wall or pusher plate cannot pull): the margin uses the conservative force |push| + |pull of the "
                     "opposite support| on the pushing side (exact for a rigid body between them, not a contact analysis)")


def check_rails(study: Any, results: Any, spec: RailSpec, limits: S4Limits | None = None, parts: list | None = None) -> RailResult:
    """The rail interface reactions and the lateral deflection of the solved study against the limits (see the module docstring)."""
    lim = limits or S4Limits()
    inputs: list[dict] = []
    warnings: list[str] = []
    if not (math.isfinite(spec.pull_rel_tol) and 0.0 <= spec.pull_rel_tol < 1.0):
        raise RailError("pull_rel_tol must be a number from 0 to 1")
    rl, rl_ok = _get(lim, "rail_reaction_limit_N", spec.reaction_limit_N, "reaction limit of a rail support", inputs)
    al, al_ok = _get(lim, "rail_axial_reaction_limit_N", spec.axial_reaction_limit_N, "axial reaction limit", inputs)
    dl, dl_ok = _get(lim, "rail_lateral_deflection_limit_mm", spec.lateral_deflection_limit_mm, "lateral deflection limit (clearance)", inputs)
    for name, v in (("reaction limit", rl), ("axial reaction limit", al), ("lateral deflection limit", dl)):
        if v is not None and not (math.isfinite(v) and v > 0):
            raise RailError(f"The {name} must be a finite positive number (got {v})")
    setup = setup_of(study)
    sups = list(setup.get("supports", []))
    if not sups:
        raise RailError("The study has no rail supports - run apply_rail_supports first")
    if not results.steps:
        raise RailError("The results contain no load state - solve the study first")
    rails = list(spec.rail_parts) or list(dict.fromkeys(x["part"] for x in sups))
    no_support = [r for r in rails if not any(x["part"] == r for x in sups)]
    if no_support:
        warnings.append("No support row exists for rail(s): " + ", ".join(no_support) + " - there is no verdict for them (apply the rail supports with these rails)")
    sel = [x for x in sups if x["part"] in rails]
    # ---- the reaction (force of the support on the structure) of every support in every state; push > 0 = the wall pushes the structure (compression)
    states = []
    for st in results.steps:
        push: dict[str, float] = {}
        total = 0.0
        for x in sel:
            R = st.reactions.get(x["selection"])
            if R is None:
                continue
            R = np.asarray(R, float)
            total += float(np.linalg.norm(R[:3]))
            comp = int(x["component"])
            push[x["selection"]] = -float(x["sign"]) * float(R[comp])        # the support at the side `sign` of its axis pushes the structure inward (a wall AND an end plate)
        states.append((st.name, push, total))
    rows: list[RailRow] = []
    best: dict[str, RailRow] = {}
    pull_any: dict[str, list[str]] = {}
    for sname, push, total in states:
        thr = spec.pull_rel_tol * total
        eff: dict[str, float] = {}
        for ax in AXES:                                                      # the walls of the lateral axes and the end plates of the deployment axis: the same rule
            for sign, other in ((1, -1), (-1, 1)):
                side = [x["selection"] for x in sel if x["axis"] == ax and x["sign"] == sign and x["selection"] in push]
                opp = [x["selection"] for x in sel if x["axis"] == ax and x["sign"] == other and x["selection"] in push]
                pushing = [n for n in side if push[n] > thr]
                total_push = sum(push[n] for n in pushing)
                pull_opp = sum(-push[n] for n in opp if push[n] < -thr)
                for n in side:
                    if push[n] > thr and total_push > 0:
                        eff[n] = push[n] * (1.0 + pull_opp / total_push)           # |push| + the share of |pull| of the opposite side
                    else:
                        eff[n] = abs(push[n])
        for x in sel:
            n = x["selection"]
            if n not in push:
                continue
            wall = x["kind"] == "wall"
            val = push[n]
            if val < -thr:                                                   # a wall or an end plate that would have to PULL (a pusher plate / door cannot)
                pull_any.setdefault(n, []).append(sname)
            e = eff.get(n, abs(val))
            limit = rl if wall else al
            mos = None if limit is None else (math.inf if e <= 0 else limit / e - 1.0)
            row = RailRow(n, x["part"], "wall" if wall else "axial", sname, val, False, limit, mos, e)
            if n not in best or row.effective_N > best[n].effective_N:
                best[n] = row
    for x in sel:
        row = best.get(x["selection"])
        if row is not None:
            row.pull_states = pull_any.get(row.support, [])
            row.pulls = bool(row.pull_states)
            rows.append(row)
    defl: list[dict] = []
    mesh = results.mesh
    lat_axes = [a for a in range(3) if a != AXES.index(setup.get("axis", spec.axis))]
    missing_mesh = []
    for part in rails:
        if part in no_support:
            continue
        if part not in mesh.part_names:
            missing_mesh.append(part)
            continue
        idx = mesh.part_names.index(part)
        nodes = np.unique(mesh.tets[mesh.tet_part == idx])
        best_d, st_best = 0.0, ""
        for st in results.steps:
            u = np.asarray(st.U)[nodes][:, lat_axes]
            m = float(np.sqrt((u ** 2).sum(axis=1)).max()) if len(nodes) else 0.0
            if m > best_d:
                best_d, st_best = m, st.name
        defl.append({"part": part, "max_lateral_deflection_mm": best_d, "state": st_best, "limit_mm": dl,
                     "mos": None if dl is None else (math.inf if best_d <= 0 else dl / best_d - 1.0)})
    if missing_mesh:
        warnings.append("Rail(s) not found in the mesh of the results (no deflection): " + ", ".join(missing_mesh))
    pulls = [r for r in rows if r.pulls]
    if pulls:
        warnings.append("A wall or an end plate would have to PULL (bilateral linear support) at " + "; ".join(f"{r.support} ({', '.join(r.pull_states[:3])})" for r in pulls[:4])
                        + ": the margins use the conservative force |push| + |pull of the opposite support|"
                        + ("; with one held end only (axial='min' / 'max') the pull is not shared - choose the end that the load pushes against, or 'both'"
                           if any(r.kind == "axial" for r in pulls) and spec.axial in ("min", "max") else ""))
    has_wall = any(r.kind == "wall" for r in rows)
    has_axial = any(r.kind == "axial" for r in rows)
    used_ok = [rl_ok if has_wall else True, al_ok if has_axial else True, dl_ok if defl else True]
    needed = [(rl, has_wall), (al, has_axial), (dl, bool(defl))]
    complete = all(v is not None for v, used in needed if used)
    mos_vals = [r.mos for r in rows if r.mos is not None] + [d["mos"] for d in defl if d["mos"] is not None]
    incomplete = bool(no_support or missing_mesh) or any(not any(r.part == p for r in rows) for p in rails if p not in no_support)
    if incomplete or not rows:
        verdict, text = UNCONFIRMED, "no verdict: a named rail has no support row / results (" + "; ".join(warnings[:2]) + ")" if warnings else "no verdict: no rail support has a reaction in the results"
    elif not (all(used_ok) and complete) or not mos_vals:
        verdict, text = UNCONFIRMED, "a limit is not confirmed in the requirement set (or missing) - indicative only"
    elif min(mos_vals) < 0:
        verdict, text = FAIL, "a limit is exceeded"
    else:
        verdict, text = PASS, "all rail reactions and deflections are within the confirmed limits"
    if has_wall or has_axial:
        text += " (" + LINEAR_MODEL_NOTE + (f"; {len(pulls)} support(s) would pull" if pulls else "") + ")"
    return RailResult(rows, defl, verdict, text, inputs, warnings)

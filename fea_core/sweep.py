"""Parametric study and Design of Experiments (DOE / C4b).

A systematic sweep of the inputs of a study over a grid (all combinations) or a list (the i-th values together):
  - load scale: all load cases together (`load_factor`) or one case (`case_factor:<case>`)
  - material of a part (`material:<part>`) or a property of a material (`prop:<material>:<field>`, E, nu, rho, Re, Rm, …)
  - global mesh size (`mesh_size`)
  - 3D print settings of a part (`print:<part>:<field>`, walls, infill, pattern, …)
  - named parameter of the sketch of the study (`sketch:<name>`, F15-S5): the geometry is rebuilt and meshed per point, the settings
    are carried over by the sketch remap (`sketch.build.rebuild_study`)
CAD parameters of a STEP are not swept: the core cannot regenerate the geometry from parameters (the CAD link only replaces a STEP).
Mode "lhs" (F15-S4): continuous parameters get `bounds` instead of values and `max_points` Latin-hypercube points in that box (seeded,
at most 5 continuous parameters; categorical parameters keep their values and are spread evenly) – the training set of the surrogate
model (`fea_core.surrogate`).

KPI of every design point (DP): max. von Mises (all nodes and OUTSIDE singularities), displacement, mass, natural frequencies,
the lowest 3D print safety factor and the safety to Re. `min_safety` = Re / max. von Mises at the first place that is neither a
singularity nor affected by a boundary condition (`mcp.diagnostics.hotspots`, the same classification and the same number as the
GUI Assessment card, minimum over the load states); `min_safety_raw` = Re / max. von Mises per part including singularities
(informative). A material without Re has no safety (inf). A linear study with only load factors is solved once and the KPI are
scaled exactly (superposition); everything else is solved per point on one mesh while the mesh signature does not change.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .assessment import part_nodes as _part_nodes
from .solve import run_study
from .study import Study

KINDS = ("load_factor", "case_factor", "material", "material_prop", "mesh_size", "print", "sketch_param")
SKETCH_PREFIX = "sketch:"
MODES = ("grid", "list", "lhs")
MAX_LHS_PARAMS = 5                                # continuous parameters of a Latin-hypercube design (the surrogate's limit)
MATERIAL_FIELDS = ("E", "nu", "rho", "Re", "Rm", "alpha", "k", "E_z", "Rm_z")
PRINT_FIELDS = {"walls": int, "top_layers": int, "bottom_layers": int, "infill": float, "layer_height": float,
                "line_width": float, "pattern": str}
MAX_POINTS = 200                                  # hard ceiling (each point = one analysis)
HOTSPOTS_TOP = 8                                  # places classified per state – as the GUI (main.HS_TOP)
NOT_REAL = ("singularity", "affected by boundary condition")     # classifications that are not a real stress maximum
# StepResult.fields that do not depend on the load magnitude (kept when rescaling); the other fields are not linear in the
# load (print safety, topology…) and are dropped from the rescaled result
INVARIANT_FIELDS = {"stress_error"}


class SweepError(ValueError):
    """Invalid specification of the parametric study."""


@dataclass
class SweepParam:
    """One swept input. kind: load_factor | case_factor (target = case) | material (target = part, values = material
    names) | material_prop (target = material, field = property) | mesh_size | print (target = part, field = print setting) |
    sketch_param (target = name of a sketch parameter). `bounds` = (lo, hi) of a continuous parameter in mode "lhs" (instead of values)."""
    kind: str
    values: list = field(default_factory=list)
    target: str = ""
    field: str = ""
    bounds: tuple[float, float] | None = None

    @property
    def key(self) -> str:
        if self.kind == "case_factor":
            return f"case_factor:{self.target}"
        if self.kind == "material":
            return f"material:{self.target}"
        if self.kind == "material_prop":
            return f"prop:{self.target}:{self.field}"
        if self.kind == "print":
            return f"print:{self.target}:{self.field}"
        if self.kind == "sketch_param":
            return f"{SKETCH_PREFIX}{self.target}"
        return self.kind

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "target": self.target, "field": self.field, "values": list(self.values)}
        if self.bounds is not None:
            d["bounds"] = [float(self.bounds[0]), float(self.bounds[1])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SweepParam":
        b = d.get("bounds")
        try:
            bounds = (float(b[0]), float(b[1])) if b is not None else None
        except (TypeError, ValueError, IndexError, KeyError):
            raise SweepError("bounds must be two numbers [lo, hi]") from None
        return cls(kind=str(d.get("kind", "")), values=list(d.get("values") or []), target=str(d.get("target") or ""),
                   field=str(d.get("field") or ""), bounds=bounds)


def param_from_key(key: str, values: list | None = None, bounds: tuple[float, float] | None = None) -> SweepParam:
    """The inverse of `SweepParam.key`: 'load_factor', 'mesh_size', 'case_factor:<case>', 'material:<part>', 'prop:<material>:<field>',
    'print:<part>:<field>', 'sketch:<name>' → a SweepParam (the surrogate checks a point with the keys of its parameters)."""
    vals = list(values or [])
    if key in ("load_factor", "mesh_size"):
        return SweepParam(key, vals, bounds=bounds)
    kind, _, rest = key.partition(":")
    if kind == "case_factor" and rest:
        return SweepParam("case_factor", vals, target=rest, bounds=bounds)
    if kind == "material" and rest:
        return SweepParam("material", vals, target=rest, bounds=bounds)
    if kind == "sketch" and rest:
        return SweepParam("sketch_param", vals, target=rest, bounds=bounds)
    if kind in ("prop", "print"):
        target, _, fld = rest.rpartition(":")
        if target and fld:
            return SweepParam("material_prop" if kind == "prop" else "print", vals, target=target, field=fld, bounds=bounds)
    raise SweepError(f"Unknown parameter key '{key}'")


@dataclass
class DesignPoint:
    index: int
    params: dict[str, Any]
    max_stress: float = 0.0          # max. von Mises over the load cases [MPa] (all nodes)
    max_disp: float = 0.0            # max. |U| [mm]
    mass: float = 0.0                # total mass [kg]
    min_safety: float = float("inf") # safety to Re at the max. von Mises OUTSIDE singularities (GUI Assessment card), min. over the states; inf = no Re
    converged: bool = True
    error: str = ""
    max_stress_clean: float | None = None    # max. von Mises outside singularities (diagnostics.hotspots) [MPa]
    min_safety_raw: float = float("inf")     # Re / max. von Mises per part including singularities (informative); inf = no Re
    freq1: float | None = None               # 1st natural frequency [Hz]
    print_safety: float | None = None        # lowest 3D print safety factor (Tsai-Wu / Hill)
    kpi: dict = field(default_factory=dict)  # full KPI per load case (cad_link.history.kpis)
    adaptive: bool = False                   # chosen by the surrogate model ("verify this point"), not by the design: never calibrates it

    def to_dict(self) -> dict:
        def num(v):
            return None if v is None or not math.isfinite(float(v)) else float(v)
        return {**({"adaptive": True} if self.adaptive else {}),
            "index": self.index,
            "params": self.params,
            "max_stress": self.max_stress,
            "max_stress_clean": num(self.max_stress_clean),
            "max_disp": self.max_disp,
            "mass": self.mass,
            "min_safety": num(self.min_safety),
            "min_safety_raw": num(self.min_safety_raw),
            "freq1": num(self.freq1),
            "print_safety": num(self.print_safety),
            "converged": self.converged,
            "error": self.error,
        }


@dataclass
class ParametricStudySpec:
    """Specification of the sweep. Legacy fields (load_factors, materials, mesh_sizes) and `params` are combined:
    mode "grid" = all combinations, "list" = the i-th values of all parameters together (equal lengths), "lhs" = `max_points`
    Latin-hypercube points in the `bounds` of the continuous parameters (seed `seed`; categorical parameters spread evenly)."""
    load_factors: list[float] = field(default_factory=lambda: [1.0])
    materials: dict[str, list[str]] = field(default_factory=dict)  # part_name -> [mat_name1, mat_name2]
    mesh_sizes: list[float] = field(default_factory=list)          # mesh sizes [mm]
    max_points: int = 50                                          # safety ceiling of the number of points
    warnings: list[str] = field(default_factory=list)
    params: list[SweepParam] = field(default_factory=list)
    mode: str = "grid"                                            # grid | list | lhs
    required_safety: float | None = None                          # Pareto: points with min_safety below are infeasible
    seed: int = 0                                                 # mode lhs: the seed of the sampling (the same spec gives the same points)

    def all_params(self) -> list[SweepParam]:
        out = []
        if len(self.load_factors) > 1 or (len(self.load_factors) == 1 and self.load_factors[0] != 1.0):
            out.append(SweepParam("load_factor", list(self.load_factors)))
        for part, mats in self.materials.items():
            if mats:
                out.append(SweepParam("material", list(mats), target=part))
        if len(self.mesh_sizes) >= 1:
            out.append(SweepParam("mesh_size", list(self.mesh_sizes)))
        return out + [p for p in self.params if p.values or p.bounds is not None]

    def total(self) -> int:
        ps = self.all_params()
        if not ps:
            return 1
        if self.mode == "lhs":
            return max(1, min(int(self.max_points), MAX_POINTS))
        if self.mode == "list":
            return len(ps[0].values)
        return int(np.prod([len(p.values) for p in ps]))

    def combinations(self) -> list[dict[str, Any]]:
        ps = self.all_params()
        if not ps:
            if self.mode == "lhs":
                raise SweepError("Mode lhs needs at least one continuous parameter with bounds [lo, hi]")
            return [{}]                      # no variability, only the starting point
        keys = [p.key for p in ps]
        if len(set(keys)) != len(keys):
            raise SweepError("A parameter is given twice: " + ", ".join(sorted({k for k in keys if keys.count(k) > 1})))
        limit = max(1, min(int(self.max_points), MAX_POINTS))
        if self.mode == "lhs":
            return self._lhs_rows(ps, keys, limit)
        if any(p.bounds is not None for p in ps):
            raise SweepError("bounds are used only in mode lhs (grid / list take values)")
        if self.mode == "list":
            n = {len(p.values) for p in ps}
            if len(n) != 1:
                raise SweepError("List mode needs the same number of values for every parameter")
            rows = [dict(zip(keys, vals)) for vals in zip(*[p.values for p in ps])]
        elif self.mode == "grid":
            rows = []
            for prod in itertools.product(*[p.values for p in ps]):
                rows.append(dict(zip(keys, prod)))
                if len(rows) >= limit:
                    break
        else:
            raise SweepError(f"Unknown mode '{self.mode}' ({' | '.join(MODES)})")
        total = self.total()
        if total > limit:
            self.warnings = [
                f"The number of parameter combinations ({total}) exceeded the limit max_points={limit}. The point list was truncated to the first {limit} points."
            ]
            rows = rows[:limit]
        else:
            self.warnings = []
        return rows

    def _lhs_rows(self, ps: list[SweepParam], keys: list[str], n: int) -> list[dict[str, Any]]:
        """Mode lhs: n Latin-hypercube points over the continuous parameters (`bounds`), the categorical ones (`values`, no bounds)
        spread evenly; integer-valued print settings are rounded by `apply_params`."""
        from .surrogate.design import assign_categories, lhs_design
        cont = [(k, p) for k, p in zip(keys, ps) if p.bounds is not None]
        cat = [(k, p) for k, p in zip(keys, ps) if p.bounds is None]
        if not cont:
            raise SweepError("Mode lhs needs at least one continuous parameter with bounds [lo, hi]")
        if len(cont) > MAX_LHS_PARAMS:
            raise SweepError(f"Mode lhs takes at most {MAX_LHS_PARAMS} continuous parameters (got {len(cont)})")
        for k, p in cat:
            if not p.values:
                raise SweepError(f"Parameter {k}: give bounds [lo, hi] (continuous) or values (categorical)")
        for k, p in cont:
            assert p.bounds is not None
            if p.values:
                raise SweepError(f"Parameter {k}: bounds and values together – use bounds (continuous) or values (categorical)")
            if not (math.isfinite(p.bounds[0]) and math.isfinite(p.bounds[1]) and p.bounds[1] > p.bounds[0]):
                raise SweepError(f"Parameter {k}: bounds must be finite with lo < hi")
        X = lhs_design([p.bounds for _, p in cont if p.bounds is not None], n, seed=self.seed)
        cats = {k: assign_categories([str(v) for v in p.values], n, seed=self.seed + 1 + i) for i, (k, p) in enumerate(cat)}
        self.warnings = ([f"The number of Latin-hypercube points was limited to {n} (max_points = {self.max_points}, ceiling {MAX_POINTS})"]
                         if int(self.max_points) > n else [])
        return [{**{k: float(f"{X[i, j]:.8g}") for j, (k, _) in enumerate(cont)}, **{k: cats[k][i] for k, _ in cat}} for i in range(n)]

    def to_dict(self) -> dict:
        d = {"mode": self.mode, "max_points": self.max_points, "params": [p.to_dict() for p in self.all_params()]}
        if self.required_safety is not None:
            d["required_safety"] = self.required_safety
        if self.mode == "lhs":
            d["seed"] = self.seed
        return d

    @classmethod
    def from_params(cls, params: list[SweepParam | dict], mode: str = "grid", max_points: int = 50, seed: int = 0) -> "ParametricStudySpec":
        ps = [p if isinstance(p, SweepParam) else SweepParam.from_dict(p) for p in params]
        return cls(load_factors=[1.0], params=ps, mode=mode, max_points=max_points, seed=int(seed))


@dataclass
class ParametricStudyResult:
    points: list[DesignPoint]
    pareto_indices: list[int] = field(default_factory=list)
    baseline_index: int = 0
    warnings: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)     # parameter keys (table columns)
    linear_shortcut: bool = False                     # one analysis, KPI scaled
    cancelled: bool = False
    shortcut_reason: str = ""                         # a pure load-factor sweep that was NOT scaled: why every point was solved
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)   # mode lhs: the sampled box per parameter key (for the surrogate)
    sweep: dict = field(default_factory=dict)         # the specification of the sweep (ParametricStudySpec.to_dict) – stored with a surrogate
    fingerprint: str = ""                             # surrogate.stale.study_fingerprint of the study the sweep was run on (set by the caller)

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "pareto_indices": self.pareto_indices,
            "baseline_index": self.baseline_index,
            "total_points": len(self.points),
            "parameters": self.keys,
            "linear_shortcut": self.linear_shortcut,
            "cancelled": self.cancelled,
            "warnings": self.warnings,
            **({"bounds": {k: list(v) for k, v in self.bounds.items()}} if self.bounds else {}),
            **({"full_solve_reason": self.shortcut_reason} if self.shortcut_reason else {}),
        }

    def summary_table(self) -> str:
        def f(v, fmt):
            return "–" if v is None or not math.isfinite(v) else format(v, fmt)
        lines = ["| DP | Parameters | Max σ [MPa] | Max σ outside sing. [MPa] | Max u [mm] | Mass [kg] | Safety | Status |",
                 "|---|---|---|---|---|---|---|---|"]
        for p in self.points:
            p_str = ", ".join(f"{k}={v}" for k, v in p.params.items()) if p.params else "default"
            status = "OK" if p.converged else f"ERROR ({p.error})"
            star = " *" if p.index in self.pareto_indices else ""
            lines.append(f"| DP{p.index}{star} | {p_str} | {p.max_stress:.1f} | {f(p.max_stress_clean, '.1f')} | "
                         f"{p.max_disp:.3f} | {p.mass:.3f} | {f(p.min_safety, '.2f')} | {status} |")
        return "\n".join(lines)


def identify_pareto(points: list[DesignPoint], required_safety: float | None = None) -> list[int]:
    """Points of the Pareto front: minimum mass, minimum stress (outside singularities when classified) and maximum safety.

    The safety objective is used only between points that both have a finite safety (a material without Re has none).
    With `required_safety` the points with a known safety below the requirement are infeasible and never on the front.
    """
    valid = [p for p in points if p.converged]
    if required_safety is not None:
        valid = [p for p in valid if p.min_safety is None or not math.isfinite(p.min_safety) or p.min_safety >= required_safety]
    if not valid:
        return []

    def stress(p):
        return p.max_stress_clean if p.max_stress_clean is not None else p.max_stress

    def known(p) -> bool:
        return p.min_safety is not None and math.isfinite(p.min_safety)

    def objectives(p, other):
        """(value of p, value of other) pairs, all to be minimised."""
        obj = [(p.mass, other.mass), (stress(p), stress(other))]
        if known(p) and known(other):
            obj.append((-p.min_safety, -other.min_safety))
        return obj
    pareto = []
    for p in valid:
        dominated = False
        for other in valid:
            if other.index == p.index:
                continue
            ob = objectives(p, other)
            # other dominates p if it is at least as good in all objectives and sharply better in at least one
            if all(o <= v for v, o in ob) and any(o < v for v, o in ob):
                dominated = True
                break
        if not dominated:
            pareto.append(p.index)
    return sorted(pareto)


# --------------------------------------------------------------------- values from text (GUI, MCP)
_RANGE = re.compile(r"^\s*([-+0-9.eE]+)\s*:\s*([-+0-9.eE]+)\s*:\s*(\d+)\s*$")


def parse_values(text: str, numeric: bool = True) -> list:
    """Values from text: "1, 1.5, 2" (also ";" or spaces; decimal comma accepted when ";" separates) or a range
    "start:stop:n" (n evenly spaced values including both ends). numeric=False = names (material, infill pattern)."""
    text = (text or "").strip()
    if not text:
        return []
    if not numeric:
        return [t.strip() for t in re.split(r"[;\n]|,(?!\d)", text) if t.strip()]
    m = _RANGE.match(text)
    if m:
        a, b, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if not 1 <= n <= MAX_POINTS:
            raise SweepError(f"Range '{text}': the number of values must be 1–{MAX_POINTS}")
        return [float(f"{v:.12g}") for v in (np.linspace(a, b, n) if n > 1 else [a])]
    if ";" in text:
        items = [t.strip().replace(",", ".") for t in text.split(";")]
    else:
        items = [t for t in re.split(r"[,\s]+", text) if t]
    try:
        return [float(t) for t in items if t]
    except ValueError:
        raise SweepError(f"Cannot read the numbers '{text}' (use '1, 2, 3' or 'start:stop:n')") from None


# --------------------------------------------------------------------- application of the parameters
def _scale_loads(loads, factor: float) -> None:
    """Scales loads by `factor` – the rule of each kind is `load_kinds.LOAD_HANDLERS[kind].scale` (one place for all
    consumers; test_load_kinds fails for a kind without it)."""
    from .load_kinds import load_handler
    for l in loads:
        h = load_handler(l)
        if h is not None:
            h.scale(l, factor)
        elif hasattr(l, "vector") and l.vector:     # unknown kind: a vector / a number
            l.vector = [float(v) * factor for v in l.vector]
        elif hasattr(l, "value") and isinstance(l.value, (int, float)):
            l.value = float(l.value) * factor


def scale_study_loads(study: Study, factor: float, cases: list[str] | None = None) -> None:
    """Scales the loads of the study (or only of the named load cases) by `factor` according to the kind of load.

    Scaling rules:
      - Force, BearingLoad, Gravity, Pressure (value and gradient), RemoteLoad, frame loads: linearly.
      - Rotation: omega × sqrt(factor) (centrifugal force ~ omega²), alpha × factor; Inertia likewise (wmat × factor).
      - ThermalLoad: NOT SCALED (temperature in °C is not a force load).
    """
    for lc in study.load_cases:
        if cases is None or lc.name in cases:
            _scale_loads(lc.loads, factor)


def _apply_load_param(st: Study, k: str, val) -> bool:
    """Applies a load-scale parameter (load_factor, case_factor:<case>) to st; False = k is another kind of parameter."""
    if k == "load_factor":
        scale_study_loads(st, float(val))
    elif k.startswith("case_factor:"):
        name = k.split(":", 1)[1]
        if name not in {lc.name for lc in st.load_cases}:
            raise SweepError(f"Load case '{name}' does not exist")
        scale_study_loads(st, float(val), [name])
    else:
        return False
    return True


def check_load_scaling(study: Study, params: dict[str, Any]) -> None:
    """Raises what apply_params would raise for the load-scale parameters of a design point (a case that does not exist,
    a negative factor on a rotation / inertia with a nonzero angular velocity – ω scales with √factor) without copying the
    study: only its load states are scaled. The linear shortcut calls it per point so that it fails like the full path."""
    probe = copy.copy(study)
    probe.load_cases = copy.deepcopy(study.load_cases)
    for k, val in params.items():
        _apply_load_param(probe, k, val)


def has_sketch_params(params: dict[str, Any]) -> bool:
    return any(k.startswith(SKETCH_PREFIX) for k in params)


def _rebuild_sketch(study: Study, params: dict[str, Any], workdir: str | Path | None) -> Study:
    """The study with the geometry rebuilt for the sketch parameters among `params` (the rest of the parameters is not touched). The
    settings (constraints, loads, materials, mesh) are carried over by the sketch remap; a selection that lost all its faces is an error."""
    from .sketch.build import rebuild_study
    from .sketch.expr import SketchError
    if study.sketch is None:
        raise SweepError("The study has no sketch (sketch parameters need a model drawn in the sketcher)")
    if workdir is None:
        raise SweepError("Sketch parameters rebuild the geometry: a work folder is needed")
    sk = {k[len(SKETCH_PREFIX):]: v for k, v in params.items() if k.startswith(SKETCH_PREFIX)}
    try:
        new, rep, _ = rebuild_study(study, workdir, params={k: float(v) for k, v in sk.items()})
    except SketchError as e:
        raise SweepError(f"Sketch rebuild failed: {e}") from None
    if rep is not None and rep.blocking():
        raise SweepError("Sketch rebuild lost the faces of: " + ", ".join(rep.blocking()[:5]))
    return new


def apply_params(study: Study, params: dict[str, Any], workdir: str | Path | None = None) -> Study:
    """Copy of the study with the parameters of one design point applied (used for the analysis and for taking over a variant).
    workdir = a folder for the rebuilt geometry; needed only when `params` has sketch parameters (`sketch:<name>`)."""
    st = _rebuild_sketch(study, params, workdir) if has_sketch_params(params) else copy.deepcopy(study)
    for k, val in params.items():
        if k.startswith(SKETCH_PREFIX):
            continue
        if _apply_load_param(st, k, val):
            continue
        if k.startswith("material:"):
            part = k.split(":", 1)[1]
            if str(val) not in st.materials:
                raise SweepError(f"Material '{val}' is not in the study")
            st.part_materials[part] = str(val)
        elif k.startswith("prop:"):
            _, mat, fld = k.split(":", 2)
            if mat not in st.materials:
                raise SweepError(f"Material '{mat}' is not in the study")
            if fld not in MATERIAL_FIELDS:
                raise SweepError(f"Material property '{fld}' cannot be swept ({', '.join(MATERIAL_FIELDS)})")
            st.materials[mat] = replace(st.materials[mat], **{fld: float(val)})
        elif k == "mesh_size":
            if not float(val) > 0:
                raise SweepError("The mesh size must be positive")
            st.mesh.size = float(val)
        elif k.startswith("print:"):
            _, part, fld = k.split(":", 2)
            spec = st.printing.parts.get(part)
            if spec is None:
                raise SweepError(f"Part '{part}' has no 3D print settings")
            if fld not in PRINT_FIELDS:
                raise SweepError(f"Print setting '{fld}' cannot be swept ({', '.join(PRINT_FIELDS)})")
            v = PRINT_FIELDS[fld](val) if PRINT_FIELDS[fld] is not int else int(round(float(val)))
            if fld == "infill" and not 0.0 <= float(v) <= 1.0:
                raise SweepError("Infill must be 0–1")
            setattr(spec, fld, v)
        else:
            raise SweepError(f"Unknown parameter '{k}'")
    return st


def check_spec(study: Study, spec: ParametricStudySpec) -> list[str]:
    """Errors of the specification against the study (empty = ready)."""
    errors = []
    cases = {lc.name for lc in study.load_cases}
    lhs = spec.mode == "lhs"
    for p in spec.all_params():
        if p.kind not in KINDS:
            errors.append(f"Unknown parameter kind '{p.kind}' ({', '.join(KINDS)})")
            continue
        if not p.values and not (lhs and p.bounds is not None):
            errors.append(f"Parameter {p.key}: " + ("no values and no bounds" if lhs else "no values"))
        if p.bounds is not None and p.kind in ("material",):
            errors.append(f"Parameter {p.key}: a material is categorical (values), not continuous (bounds)")
        if p.kind == "sketch_param":
            sk = study.sketch
            if sk is None:
                errors.append("The study has no sketch – sketch parameters need a model drawn in the sketcher")
            elif p.target not in sk.params:
                errors.append(f"Sketch parameter '{p.target}' does not exist (parameters: {', '.join(sk.params) or 'none'})")
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in p.values):
                errors.append(f"Parameter {p.key}: numbers only")
            continue
        if p.kind == "case_factor" and p.target not in cases:
            errors.append(f"Load case '{p.target}' does not exist")
        elif p.kind == "material":
            for v in p.values:
                if str(v) not in study.materials:
                    errors.append(f"Material '{v}' is not in the study (add it in the Materials step first)")
        elif p.kind == "material_prop":
            if p.target not in study.materials:
                errors.append(f"Material '{p.target}' is not in the study")
            if p.field not in MATERIAL_FIELDS:
                errors.append(f"Material property '{p.field}' cannot be swept ({', '.join(MATERIAL_FIELDS)})")
            vals = [*p.values, *(p.bounds or ())]
            if p.field in ("E", "rho") and any(float(v) <= 0 for v in vals):
                errors.append(f"{p.field} must be positive")
            if p.field == "nu" and any(not -1.0 < float(v) < 0.5 for v in vals):
                errors.append("Poisson's ratio must be in (−1; 0.5)")
        elif p.kind == "mesh_size" and (any(float(v) <= 0 for v in p.values) or (p.bounds is not None and p.bounds[0] <= 0)):
            errors.append("The mesh size must be positive")
        elif p.kind == "print":
            if p.target not in study.printing.parts:
                errors.append(f"Part '{p.target}' has no 3D print settings")
            if p.field not in PRINT_FIELDS:
                errors.append(f"Print setting '{p.field}' cannot be swept ({', '.join(PRINT_FIELDS)})")
            elif p.field == "infill" and any(not 0 <= float(v) <= 1 for v in [*p.values, *(p.bounds or ())]):
                errors.append("Infill must be 0–1 (e.g. 0.2 = 20 %)")
    try:
        spec.combinations()
    except SweepError as e:
        errors.append(str(e))
    return errors


def unscaled_inputs(study: Study) -> list[str]:
    """Inputs that a load factor does not scale (the result of a load-factor point is then not factor × baseline)."""
    why = []
    if study.active_bolts() and getattr(getattr(study, "joints", None), "preload_case", True):
        why.append("bolt preload state")
    if study.has_thermal_loads():
        why.append("thermal load (temperature)")
    for c in study.constraints:
        if getattr(c, "kind", "") == "displacement" and any(v not in (None, 0, 0.0) for v in (c.ux, c.uy, c.uz)):
            why.append(f"nonzero prescribed displacement '{c.selection}'")
            break
    return why


def shortcut_blockers(study: Study, combos: list[dict] | None = None, parts=None, mesh=None) -> list[str]:
    """Reasons why the KPI of load factors cannot be scaled from one linear analysis exactly (superposition): nonlinear analysis,
    temperatures, prescribed displacements, bolt preload, 3D printing (the safety criterion is not linear), buckling,
    dynamics, beams / 2D / shells (own results), and a NEGATIVE factor on a clipped (hydrostatic) pressure
    (max(0, λ·g·d) is not λ·max(0, g·d) for λ < 0 – the wetted side moves). combos = the design points (None = factors unknown,
    not checked); parts / mesh tell which materials are really used (plasticity) – without them a plastic default material blocks."""
    why = []
    if combos is not None and any(float(v) < 0 for pt in combos for k, v in pt.items()
                                  if k == "load_factor" or k.startswith("case_factor:")):
        clipped = [l.selection for lc in study.load_cases for l in lc.loads
                   if getattr(l, "kind", "") == "pressure" and getattr(l, "clip", False) and getattr(l, "gradient", None)]
        if clipped:
            why.append(f"negative load factor with a clipped (hydrostatic) pressure '{clipped[0]}'")
    if study.is_nonlinear(mesh=mesh, parts=parts):
        why.append("nonlinear analysis (contact / NLGEOM / sequence / plasticity)")
    why += unscaled_inputs(study)
    if study.printing.parts:
        why.append("3D printing (the print safety is not linear in the load)")
    if study.buckling.enabled:
        why.append("linear buckling (λ is not proportional to the load)")
    if study.dynamics.enabled:
        why.append("dynamics")
    if getattr(study.frame, "active", False):
        why.append("beam members (frame results)")
    if study.twod.enabled or study.shell.enabled:
        why.append("2D / shell model")
    return why


def _linear_scalable(study: Study, combos: list[dict] | None = None, parts=None, mesh=None) -> bool:
    """Can the KPI of load factors be scaled from one linear analysis exactly (superposition)? See shortcut_blockers."""
    return not shortcut_blockers(study, combos, parts, mesh)


# --------------------------------------------------------------------- KPI
def _mass_kg(study: Study, mesh) -> float:
    from .loads import element_volumes
    vols = element_volumes(mesh)
    tot = 0.0
    for p_idx, p_name in enumerate(mesh.part_names):
        try:
            mat = study.material_of(p_name)
        except KeyError:
            mat = None
        rho = mat.rho if mat and mat.rho else 7.85e-9
        tot += float(np.sum(vols[mesh.tet_part == p_idx])) * rho
    return tot * 1000.0


def _re_of(mat) -> float | None:
    """Yield strength used for the safety (None = a material without Re; an orthotropic one has no single Re – as Session.node_re)."""
    if mat is None or getattr(mat, "is_ortho", False) or not mat.Re or mat.Re <= 0:
        return None
    return float(mat.Re)


def _step_rows(study: Study, res, parts=None) -> tuple[list[dict], list[str]]:
    """KPI of every state of a result (one row per `res.steps`): "vm" max. von Mises, "u" max. displacement, "k_raw" safety
    Re / max. von Mises per part including singularities, "vm_real" max. von Mises at the first place that is neither a
    singularity nor affected by a boundary condition (`mcp.diagnostics.hotspots`, top HOTSPOTS_TOP – the classification of the GUI
    Assessment card and of fea_results_hotspots) and "k" the safety to Re at that place. None = not available (no Re / no such place).
    Returns (rows, warnings)."""
    from .mcp.diagnostics import hotspots
    mesh = res.mesh
    warns: list[str] = []
    part_re = []
    for p_idx, p_name in enumerate(mesh.part_names):
        try:
            mat = study.material_of(p_name)
        except KeyError:
            mat = None
        re = _re_of(mat)
        if re is not None:
            nodes = _part_nodes(mesh, p_idx)
            if len(nodes):
                part_re.append((re, nodes))
    rows: list[dict] = []
    sess = re_node = None
    classify = bool(len(mesh.nodes))
    for i, step in enumerate(res.steps):
        row = {"vm": 0.0, "u": 0.0, "k_raw": None, "vm_real": None, "k": None}
        rows.append(row)
        if not len(step.S):
            continue
        vm = step.mises
        row["vm"] = float(np.max(vm))
        row["u"] = float(np.max(step.umag)) if len(step.U) else 0.0
        for re, nodes in part_re:
            vmax = float(np.max(vm[nodes]))
            if vmax > 1e-9:
                k = re / vmax
                row["k_raw"] = k if row["k_raw"] is None else min(row["k_raw"], k)
        if not classify:
            continue
        try:
            if sess is None:
                from .cad_link.history import _session
                sess = _session(study, res, parts)
                re_node = sess.node_re()
            hs = hotspots(sess, i, HOTSPOTS_TOP, "mises").get("hotspots", [])
            real = next((h for h in hs if h.get("classification", "") not in NOT_REAL), None)
            if real is not None:
                n = int(real["node"]) - 1
                v = float(vm[n])
                row["vm_real"] = v
                if np.isfinite(re_node[n]) and v > 1e-9:
                    row["k"] = float(re_node[n] / v)
        except Exception as e:  # noqa: BLE001 – a failed heuristic must not discard the point
            classify = False
            warns.append(f"Classification of the stress maxima failed ({e}) – the safety is taken from the raw maximum per part")
    if not classify:
        for row in rows:
            row["vm_real"], row["k"] = None, row["k_raw"]
    return rows, warns


def _evaluate_2d(dp: DesignPoint, study: Study, r2) -> tuple[list[dict], list[str]]:
    """KPI of a 2D result (plane stress / strain, axisymmetric: `Results.twod`, the 3D `steps` are empty): max. von Mises and |u| over
    the states, mass (area × thickness, or 2π·∫r dA for an axisymmetric model) and the safety to Re per part. A 2D result has no
    singularity classification, so `max_stress_clean` stays None and the surrogate models `max_stress`."""
    from .results import von_mises
    rows: list[dict] = []
    part_nodes = [np.unique(r2.tri[r2.tri_part == i][:, :6]) for i in range(len(r2.part_names))]
    res_re = []
    for i, name in enumerate(r2.part_names):
        try:
            res_re.append(_re_of(study.material_of(name)))
        except KeyError:
            res_re.append(None)
    for s in r2.steps:
        vm = von_mises(s["S"])
        row = {"vm": float(np.max(vm)) if len(vm) else 0.0, "u": float(np.max(np.linalg.norm(s["U"], axis=1))) if len(s["U"]) else 0.0,
               "k_raw": None, "vm_real": None, "k": None}
        for yield_re, nodes in zip(res_re, part_nodes):
            if yield_re is not None and len(nodes):
                vmax = float(np.max(vm[nodes]))
                if vmax > 1e-9:
                    row["k_raw"] = yield_re / vmax if row["k_raw"] is None else min(row["k_raw"], yield_re / vmax)
        row["k"] = row["k_raw"]
        rows.append(row)
    dp.max_stress = max([r["vm"] for r in rows] + [0.0])
    dp.max_disp = max([r["u"] for r in rows] + [0.0])
    dp.max_stress_clean = None
    ks = [r["k_raw"] for r in rows if r["k_raw"] is not None]
    dp.min_safety = dp.min_safety_raw = min(ks) if ks else float("inf")
    X = r2.nodes[r2.tri[:, :3]]                                   # corner nodes (M, 3, 2)
    area = 0.5 * np.abs((X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1]) - (X[:, 2, 0] - X[:, 0, 0]) * (X[:, 1, 1] - X[:, 0, 1]))
    vol = area * (2.0 * math.pi * np.abs(X[:, :, 0].mean(axis=1)) if r2.axisym else float(r2.thickness))
    mass = 0.0
    for i, name in enumerate(r2.part_names):
        try:
            mat = study.material_of(name)
        except KeyError:
            mat = None
        mass += float(np.sum(vol[r2.tri_part == i])) * (mat.rho if mat and mat.rho else 7.85e-9)
    dp.mass = mass * 1000.0
    dp.kpi = {}
    return rows, []


def _evaluate(dp: DesignPoint, study: Study, res, parts=None) -> tuple[list[dict], list[str]]:
    """Fills the KPI of a design point from a result; returns (rows of `_step_rows`, warnings)."""
    if getattr(res, "twod", None) is not None and not res.steps:
        return _evaluate_2d(dp, study, res.twod)
    rows, warns = _step_rows(study, res, parts)
    dp.max_stress = max([r["vm"] for r in rows] + [0.0])
    dp.max_disp = max([r["u"] for r in rows] + [0.0])
    reals = [r["vm_real"] for r in rows if r["vm_real"] is not None]
    dp.max_stress_clean = max(reals) if reals else None
    ks = [r["k"] for r in rows if r["k"] is not None]
    dp.min_safety = min(ks) if ks else float("inf")
    raws = [r["k_raw"] for r in rows if r["k_raw"] is not None]
    dp.min_safety_raw = min(raws) if raws else float("inf")
    dp.mass = _mass_kg(study, res.mesh)
    print_min = None
    for step in res.steps:
        ps = (getattr(step, "fields", None) or {}).get("print_safety")
        if ps is not None and len(ps):
            v = float(np.nanmin(ps))
            print_min = v if print_min is None else min(print_min, v)
    dp.print_safety = print_min
    try:
        from .cad_link.history import kpis
        k = kpis(study, res, parts)
    except Exception:  # noqa: BLE001 – the extra KPI must not discard the point
        k = {}
    dp.kpi = k
    freqs = k.get("frequencies_Hz") or []
    dp.freq1 = freqs[0] if freqs else None
    return rows, warns


def _scaled_point(dp: DesignPoint, base: DesignPoint, case_names: list[str], params: dict) -> None:
    """KPI of a linear study with load factors from the baseline: per case × |factor| (stresses, displacement), safety / |factor|."""
    g = float(params.get("load_factor", 1.0))
    f = [abs(g * float(params.get(f"case_factor:{n}", 1.0))) for n in case_names]
    cases = (base.kpi or {}).get("cases") or []
    rows = base.kpi.get("_steps") or []           # per-case KPI of the baseline (filled by run_parametric_study)
    dp.mass = base.mass
    dp.freq1 = base.freq1
    if rows and len(rows) == len(f):
        dp.max_stress = max(r["vm"] * fi for r, fi in zip(rows, f))
        dp.max_disp = max(r["u"] * fi for r, fi in zip(rows, f))
        ks = [r["k"] / fi for r, fi in zip(rows, f) if r["k"] is not None and fi > 0]
        dp.min_safety = min(ks) if ks else float("inf")
        kr = [r["k_raw"] / fi for r, fi in zip(rows, f) if r["k_raw"] is not None and fi > 0]
        dp.min_safety_raw = min(kr) if kr else float("inf")
        reals = [r["vm_real"] * fi for r, fi in zip(rows, f) if r["vm_real"] is not None]
        dp.max_stress_clean = max(reals) if reals else None
    dp.kpi = {"cases": [{k: (v * fi if k in ("vm_max", "vm_max_clean", "u_max") and v is not None else v)
                         for k, v in c.items() if k != "reactions"} for c, fi in zip(cases, f)]}


def _scaled_value(key: str, v, factor: float):
    """Value of StepResult.equilibrium / remote scaled by the load factor (ratios, flags and texts unchanged)."""
    if isinstance(v, bool) or isinstance(v, str) or v is None:
        return v
    if any(s in key for s in ("imbalance", "ratio", "tol", "rel")):
        return v
    if isinstance(v, (int, float)):
        return float(v) * factor
    if isinstance(v, np.ndarray):
        return v * factor if v.dtype.kind == "f" else v
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return [float(x) * factor for x in v]
    return copy.deepcopy(v)


def scale_step(step, factor: float):
    """Copy of a linear StepResult with the load × factor: U, S, reactions, reference displacements, equilibrium and remote
    supports scaled; fields that are not linear in the load (print safety, topology…) and joints / printing summaries dropped.
    Nothing is shared with `step` (arrays and dictionaries are copied). Returns (step, names of the dropped fields)."""
    from .results import StepResult
    fields = {}
    dropped = []
    for k, v in step.fields.items():
        if k in INVARIANT_FIELDS:
            fields[k] = np.array(v, copy=True)
        else:
            dropped.append(k)
    remote = {}
    for sel, r in (step.remote or {}).items():
        remote[sel] = {k: (_scaled_value(k, v, factor) if k in ("force", "moment", "u", "rot") else copy.deepcopy(v))
                       for k, v in r.items()}
    new = StepResult(
        name=step.name,
        U=step.U * factor,
        S=step.S * factor,
        reactions={k: np.asarray(v) * factor for k, v in step.reactions.items()},
        ref_displacements={k: np.asarray(v) * factor for k, v in step.ref_displacements.items()},
        equilibrium={k: _scaled_value(k, v, factor) for k, v in (step.equilibrium or {}).items()},
        reaction_moments={k: np.asarray(v) * factor for k, v in step.reaction_moments.items()},
        moment_point=None if step.moment_point is None else np.array(step.moment_point, copy=True),
        fields=fields,
        remote=remote,
    )
    if step.joints:
        dropped.append("joints")
    if step.printing:
        dropped.append("printing")
    return new, dropped


def scaled_results(res_base, params: dict) -> tuple[Any, list[str]]:
    """Results of a design point of the linear shortcut: the baseline scaled by the factor of every state (stresses, displacements
    and reactions are linear; a negative factor reverses the signs). Returns (Results, names of the dropped fields)."""
    from .results import Results
    g = float(params.get("load_factor", 1.0))
    steps, dropped = [], set()
    for step in res_base.steps:
        s, d = scale_step(step, g * float(params.get(f"case_factor:{step.name}", 1.0)))
        steps.append(s)
        dropped.update(d)
    scaled = Results(mesh=res_base.mesh, steps=steps, warnings=list(res_base.warnings),
                     info=copy.deepcopy(res_base.info), thermal=res_base.thermal, modes=res_base.modes)
    return scaled, sorted(dropped)


def _mesh_copy(mesh):
    """Shallow copy of the mesh with its own cell_data and report (nodes and elements shared) – a taken-over mesh gets the print
    assignment of the points written into the copy, never into the mesh of the caller (as printing.variants._mesh_copy)."""
    m = copy.copy(mesh)
    m.cell_data = dict(mesh.cell_data)
    m.report = dict(mesh.report)
    return m


def _spec_bounds(spec: ParametricStudySpec) -> dict[str, tuple[float, float]]:
    """The sampled box of a Latin-hypercube study: {parameter key: (lo, hi)} (empty for grid / list)."""
    return {p.key: (float(p.bounds[0]), float(p.bounds[1])) for p in spec.all_params() if p.bounds is not None} if spec.mode == "lhs" else {}


def run_parametric_study(study: Study, spec: ParametricStudySpec, base_workdir: Path,
                         on_progress: Callable[[int, int], None] | None = None, *, parts=None, mesh=None, cancel=None,
                         on_log: Callable[[str], None] | None = None,
                         on_point: Callable[[DesignPoint], None] | None = None,
                         study_dir: str | Path | None = None, keep_results: bool = False) -> ParametricStudyResult:
    """Runs the parametric sweep of the study (a copy per point – the study is not changed).

    parts = geometry already loaded (used when a point has to be meshed; shapes of a worker thread must not be the ones displayed
    in the GUI); mesh = a finished mesh of the study (the baseline and every point whose mesh signature matches it use it instead of
    meshing; the caller's mesh is not changed); cancel = threading.Event (stops between points and in the solver); on_point(dp) after
    every finished point (GUI table).
    keep_results: False = the work folder of every successfully evaluated point (and of the baseline) is deleted once its KPI are
    known (only the KPI are returned; a failed point keeps its folder for the diagnosis); True = the folders stay and the results of
    every point are saved as `DP_xxx/results.npz` (a point of the linear shortcut = the baseline scaled by its load factors).
    """
    errors = check_spec(study, spec)
    if errors:
        raise SweepError("; ".join(errors))
    from .signature import mesh_signature
    combos = spec.combinations()
    total = len(combos)
    keys = list(combos[0].keys()) if combos else []
    points: list[DesignPoint] = []
    base_workdir = Path(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)
    warnings = list(getattr(spec, "warnings", []))
    log = on_log or (lambda s: None)
    kw: dict[str, Any] = {"study_dir": study_dir} if study_dir is not None else {}
    if mesh is not None:
        mesh = _mesh_copy(mesh)

    def solve(st, wd, mesh_=None, shapes_ok=True):
        return run_study(st, wd, cancel=cancel, mesh=mesh_, parts=parts if (mesh_ is None and shapes_ok) else None, on_log=on_log, **kw)

    def add_warnings(ws):
        warnings.extend(w for w in ws if w not in warnings)

    def discard(folder: Path):
        """Work folder of an evaluated point (inside base_workdir only)."""
        if keep_results:
            return
        try:
            f = Path(folder).resolve()
            if f.parent == base_workdir.resolve():
                shutil.rmtree(f, ignore_errors=True)
        except OSError:
            pass

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    load_only = all(all(k == "load_factor" or k.startswith("case_factor:") for k in p) for p in combos)
    blockers = shortcut_blockers(study, combos, parts, mesh) if load_only else []
    if load_only and not blockers:
        log(f"Parametric study: linear study with load factors only – one analysis, {total} points scaled")
        base_dir = base_workdir / "baseline"
        res_base = solve(copy.deepcopy(study), base_dir, mesh)
        add_warnings(res_base.warnings)
        base = DesignPoint(index=-1, params={})
        rows, w = _evaluate(base, study, res_base, parts)
        add_warnings(w)
        base.kpi["_steps"] = rows
        names = [s.name for s in res_base.steps]
        stopped = False
        for idx, params in enumerate(combos):
            if cancelled():
                stopped = True
                break
            if on_progress:
                on_progress(idx, total)
            dp = DesignPoint(index=idx, params=params)
            try:
                check_load_scaling(study, params)            # fails like the full path (negative factor on a rotation …)
                _scaled_point(dp, base, names, params)
                if keep_results:
                    scaled, dropped = scaled_results(res_base, params)
                    if dropped:
                        add_warnings([f"Linear shortcut: fields not proportional to the load were not rescaled and are left out "
                                      f"of the saved results: {', '.join(dropped)}"])
                    dp_dir = base_workdir / f"DP_{idx:03d}"
                    dp_dir.mkdir(parents=True, exist_ok=True)
                    scaled.save(dp_dir / "results.npz")
            except Exception as e:  # noqa: BLE001 – a failed point is reported, the others continue
                dp.converged, dp.error = False, str(e)
            points.append(dp)
            if on_point:
                on_point(dp)
        discard(base_dir)
        if stopped:
            add_warnings([f"Parametric study cancelled after {len(points)} of {total} points"])
        return ParametricStudyResult(points=points, pareto_indices=identify_pareto(points, getattr(spec, "required_safety", None)),
                                     baseline_index=0, warnings=warnings, keys=keys, linear_shortcut=True, cancelled=stopped,
                                     bounds=_spec_bounds(spec), sweep=spec.to_dict())

    if any(k == "load_factor" or k.startswith("case_factor:") for p in combos for k in p):
        unscaled = unscaled_inputs(study)
        if unscaled and any(float(v) != 1.0 for p in combos for k, v in p.items()
                            if k == "load_factor" or k.startswith("case_factor:")):
            add_warnings([f"The load factor does not scale: {', '.join(unscaled)} – these stay at their study values in every point"])
    cached_mesh = mesh
    cached_sig = mesh_signature(study) if mesh is not None else None
    stopped = False
    for idx, params in enumerate(combos):
        if cancelled():
            stopped = True
            break
        if on_progress:
            on_progress(idx, total)
        dp = DesignPoint(index=idx, params=params)
        dp_dir = base_workdir / f"DP_{idx:03d}"
        log(f"Parametric study: point {idx + 1}/{total} " + ", ".join(f"{k} = {v}" for k, v in params.items()))
        try:
            new_geo = has_sketch_params(params)               # the geometry is rebuilt: the shapes and the mesh of the study do not apply
            st = apply_params(study, params, workdir=dp_dir / "sketch")
            sig = mesh_signature(st)
            mesh_arg = cached_mesh if (cached_mesh is not None and cached_sig == sig and not new_geo) else None
            res = solve(st, dp_dir, mesh_arg, shapes_ok=not new_geo)
            if not new_geo:
                cached_mesh, cached_sig = res.mesh, sig
            add_warnings(res.warnings)
            _, w = _evaluate(dp, st, res, None if new_geo else parts)
            add_warnings(w)
            if keep_results:
                res.save(dp_dir / "results.npz")
            else:
                discard(dp_dir)
        except Exception as e:  # noqa: BLE001 – one failed point does not stop the others
            if cancelled():
                stopped = True
                break
            dp.converged = False
            dp.error = str(e)
        points.append(dp)
        if on_point:
            on_point(dp)
    if stopped:
        warnings.append(f"Parametric study cancelled after {len(points)} of {total} points")
    return ParametricStudyResult(points=points, pareto_indices=identify_pareto(points, getattr(spec, "required_safety", None)),
                                 baseline_index=0, warnings=warnings, keys=keys, cancelled=stopped,
                                 shortcut_reason="; ".join(blockers) if load_only and blockers else "", bounds=_spec_bounds(spec),
                                 sweep=spec.to_dict())


def kpi_columns() -> list[tuple[str, str]]:
    """(attribute, label) of the KPI columns of the table / report / chart."""
    return [("max_stress_clean", "max. σ outside singularities [MPa]"), ("max_stress", "max. von Mises [MPa]"),
            ("max_disp", "max. displacement [mm]"), ("min_safety", "min. safety to Re [–]"), ("mass", "mass [kg]"),
            ("freq1", "f1 [Hz]"), ("print_safety", "3D print safety [–]")]

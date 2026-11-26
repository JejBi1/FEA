"""Launch load cases and 3-axis dynamic envelopes of the space-structures verification (F14 stage S2).

* **QSL generator** (`generate_qsl`): signed quasi-static load cases (`Gravity` states, tagged by the prefix "QSL " and registered in `Study.space.extra["qsl"]`)
  from the QSL factors of the requirement set (named limits `qsl_axial_g`, `qsl_lateral_g`, `qsl_lateral_2_g`) and the launch axis / combination rule of the
  environments. The factors are the user's: the program contains none.
* **3-axis orchestrator** (`run_3axis`): the sine / random / shock environment of a level (`environments.py`) is applied along X, Y and Z over ONE modal basis
  (computed once; the static states of the unit accelerations come from the same factorization). Each axis is the ordinary single-axis dynamics run
  (`dynamics.run.run_dynamics`, handed the shared basis), so a per-axis result is exactly what an independent single-axis run gives. The per-node envelope is the
  elementwise maximum over the axes (`max(|U|)`, `max(σ_vM)`, with the axis of the maximum); per-part envelopes are the maxima over the nodes of a part.
  DEVIATION from the plan text (which says max(D)): the fatigue damage D is SUMMED over the axes - the three axes are tested one after the other on the same
  hardware, so the damage accumulates (Miner); the per-axis maximum is kept as information (`KindEnvelope.D_axis_max`).
* Results carry whether the environment AND the requirement set are confirmed (`ThreeAxisResult.inputs_confirmed`); margins computed from them can only be
  `UNCONFIRMED_REQUIREMENTS` otherwise (`margins_with_envelopes`). They also carry `blockers` (reasons that forbid a PASS: a random table without sigma, a mode
  below the shock table, a partial run, damping not entered in the environments) and the digest of everything the run depended on (`run_digest`): a result
  is only used while the study, the mesh and the environments still hash to it.

No Qt and no `fea_core.mcp` import; no standard number.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .budgets import limit_of
from .environments import ENV_KINDS, EnvironmentSet, EnvLevel, EnvTable, environments_of
from .model import SpaceSettings

AXES = (0, 1, 2)
AXIS_NAMES = "XYZ"
QSL_PREFIX = "QSL "
QSL_KEY = "qsl"
RESULT_ATTR = "space_3axis"


class LoadsError(ValueError):
    """The launch loads cannot be generated / run (missing input); the message says what to enter."""


# ====================================================================== quasi-static load cases
@dataclass
class QslCase:
    name: str
    factors: list[float]                 # signed load factor per axis [g]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "factors_g": list(self.factors)}


def _signs(n: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    for _ in range(n):
        out = [s + (k,) for s in out for k in (1, -1)]
    return out


def qsl_levels(space: SpaceSettings | None) -> tuple[dict[str, float], list[str]]:
    """The QSL factors of the requirement set {axial, lateral, lateral2} [g] and what is missing. The values are the user's (limits `qsl_*`); an unconfirmed set
    still gives its values (the cases are then generated but flagged by `qsl_status`)."""
    missing: list[str] = []
    got: dict[str, float] = {}
    for key, name in (("qsl_axial_g", "axial"), ("qsl_lateral_g", "lateral")):
        lv = limit_of(space, key)
        if lv.value is None:
            missing.append(f"{key}: {lv.why_not}")
        else:
            got[name] = lv.value
    lv2 = limit_of(space, "qsl_lateral_2_g")
    if "lateral" in got:
        got["lateral2"] = lv2.value if lv2.value is not None else got["lateral"]
    return got, missing


def qsl_cases(space: SpaceSettings, launch_axis: str, combination: str) -> list[QslCase]:
    """The signed QSL cases: every axis alone (both signs); with `combination` "axial_lateral" the axial with each single lateral, with "axial_both_laterals"
    the axial with both laterals at once (every sign combination). Raises `LoadsError` when a factor or the launch axis is missing.

    LIMITS OF THE GENERATOR (the user checks them against the launcher guide; none is a standard statement):
    * ONE axial factor `qsl_axial_g` is used for BOTH signs (+ and - along the launch axis get the same magnitude). A launcher that gives different compressive and
      tensile axial levels needs the case edited afterwards - an edited generated case makes `qsl_status` non-current.
    * The lateral directions are the two GLOBAL axes of the model that are not the launch axis; no lateral direction at another angle (45 degrees ...) is made.
    * "axial_both_laterals" applies both lateral factors AT ONCE, so the lateral resultant is sqrt(lateral^2 + lateral_2^2) - sqrt(2) times the factor when both
      are equal. Use "axial_lateral" when the requirement gives the lateral level as a resultant.
    * A factor of 0 or below is refused (`budgets.limit_value_problem`): both signs are always generated, a negative factor would only flip them."""
    if launch_axis not in ("x", "y", "z"):
        raise LoadsError("the launch axis is not set (environments: launch_axis x | y | z)")
    lv, missing = qsl_levels(space)
    if missing:
        raise LoadsError("QSL factors missing in the requirement set - " + "; ".join(missing))
    ia = "xyz".index(launch_axis)
    lat = [i for i in AXES if i != ia]
    amp = {ia: lv["axial"], lat[0]: lv["lateral"], lat[1]: lv["lateral2"]}

    def make(axes: tuple[int, ...], signs: tuple[int, ...]) -> QslCase:
        f = [0.0, 0.0, 0.0]
        for a, s in zip(axes, signs):
            f[a] = s * amp[a]
        label = " ".join(f"{'+' if s > 0 else '-'}{AXIS_NAMES[a]}" for a, s in zip(axes, signs))
        return QslCase(QSL_PREFIX + label, f)

    out: list[QslCase] = []
    for a in (ia, lat[0], lat[1]):
        out += [make((a,), s) for s in _signs(1)]
    if combination == "axial_lateral":
        for b in lat:
            out += [make((ia, b), s) for s in _signs(2)]
    elif combination == "axial_both_laterals":
        out += [make((ia, lat[0], lat[1]), s) for s in _signs(3)]
    elif combination != "none":
        raise LoadsError("qsl_combination must be none | axial_lateral | axial_both_laterals")
    return out


def _same_gravity(lc: Any, case: QslCase) -> bool:
    """The load case `lc` is exactly the generated one: one gravity load of `factors x g` in the GLOBAL system (no `csys`) and nothing else. The same vector in a
    local coordinate system is another direction."""
    from ..study import Gravity
    if len(lc.loads) != 1 or not isinstance(lc.loads[0], Gravity) or lc.loads[0].csys:
        return False
    want = np.asarray([f * G_MM_S2 for f in case.factors], float)
    got = np.asarray(lc.loads[0].vector, float)
    return got.shape == (3,) and bool(np.allclose(got, want, rtol=1e-12, atol=1e-9))


def qsl_status(study: Any) -> dict[str, Any]:
    """The registered QSL cases of the study and whether they still are what the generator makes NOW. The cases are regenerated in memory (factors of the
    requirement set, launch axis, combination rule) and compared with the study: every name, and the gravity vector of every case. An edited case, a deleted one, a
    changed factor, axis or combination makes the status non-current (`problems` says what). `usable_for_verification` also needs the requirement set AND the
    environments (the launch axis) to be confirmed."""
    sp = getattr(study, "space", None)
    rec = (sp.extra.get(QSL_KEY) if sp is not None else None) or {}
    names = list(rec.get("names", []))
    cur, _ = qsl_levels(sp)
    es = environments_of(sp)
    axis = es.launch_axis if es is not None else ""
    comb = es.qsl_combination if es is not None else "axial_lateral"
    problems: list[str] = []
    if not names:
        problems.append("no QSL case was generated")
    else:
        if rec.get("levels") != cur:
            problems.append("the QSL factors of the requirement set changed after the generation")
        if rec.get("launch_axis", "") != axis:
            problems.append(f"the launch axis changed after the generation ({rec.get('launch_axis', '') or 'none'} -> {axis or 'none'})")
        if rec.get("combination", "") != comb:
            problems.append(f"the combination rule changed after the generation ({rec.get('combination', '') or 'none'} -> {comb})")
        try:
            expected = qsl_cases(sp, axis, comb) if sp is not None else []
        except LoadsError as e:
            expected = []
            problems.append(f"the cases cannot be regenerated: {e}")
        by_name: dict[str, Any] = {}
        for lc in getattr(study, "load_cases", []):
            by_name.setdefault(lc.name, lc)
        for c in expected:
            lc = by_name.get(c.name)
            if lc is None:
                problems.append(f"case '{c.name}' is missing (deleted or renamed)")
            elif not _same_gravity(lc, c):
                problems.append(f"case '{c.name}' differs from the generated one (edited: its gravity vector is not {[round(f, 6) for f in c.factors]} g)")
        wanted = {c.name for c in expected}
        problems += [f"registered case '{n}' does not belong to the set of the current axis / combination" for n in names if expected and n not in wanted]
    same = not problems
    req = bool(sp is not None and sp.requirement_confirmed())
    env = bool(es is not None and es.is_confirmed())
    return {"names": names, "levels_g": rec.get("levels"), "launch_axis": rec.get("launch_axis", ""), "combination": rec.get("combination", ""),
            "current_with_requirement_set": same, "problems": problems, "requirement_set_confirmed": req, "environment_confirmed": env,
            "usable_for_verification": bool(same and req and env)}


def remove_qsl(study: Any) -> int:
    """Removes the QSL cases that the generator made (by the registered names); returns how many were removed."""
    sp = study.space
    rec = (sp.extra.get(QSL_KEY) if sp is not None else None) or {}
    names = set(rec.get("names", []))
    before = len(study.load_cases)
    study.load_cases = [c for c in study.load_cases if c.name not in names]
    if sp is not None:
        sp.extra.pop(QSL_KEY, None)
    return before - len(study.load_cases)


def generate_qsl(study: Any, *, replace: bool = True) -> list[QslCase]:
    """Adds the signed QSL cases to `study.load_cases` (replacing the earlier generated ones when `replace`). The body force of a case is `factor · 9810 mm/s²`
    along the axis (the loading of the structure by the launch acceleration, in the sense of the sign; both signs are generated). Requires the requirement set
    with the QSL factors and the environments with the launch axis. The generator limits (one axial factor for both signs, global-axis laterals, the sqrt(2)
    resultant of "axial_both_laterals") are listed at `qsl_cases`; the registered cases are checked against a regeneration by `qsl_status`."""
    from ..study import Gravity, LoadCase
    sp = study.space
    if sp is None or sp.requirement_set is None:
        raise LoadsError("there is no requirement set - add it and enter the QSL factors (limits qsl_axial_g, qsl_lateral_g)")
    es = environments_of(sp)
    axis = es.launch_axis if es is not None else ""
    comb = es.qsl_combination if es is not None else "axial_lateral"
    cases = qsl_cases(sp, axis, comb)
    taken = {c.name for c in study.load_cases}
    rec = sp.extra.get(QSL_KEY) or {}
    old = set(rec.get("names", [])) if replace else set()
    if any(c.name in taken - old for c in cases):
        raise LoadsError("a load case with a QSL name exists that the generator did not make - rename it first")
    if old:
        study.load_cases = [c for c in study.load_cases if c.name not in old]
    for c in cases:
        study.load_cases.append(LoadCase(c.name, [Gravity(vector=[f * G_MM_S2 for f in c.factors])]))
    lv, _ = qsl_levels(sp)
    sp.extra[QSL_KEY] = {"names": [c.name for c in cases], "levels": lv, "launch_axis": axis, "combination": comb}
    return cases


G_MM_S2 = 9810.0                                     # the program's convention (CLAUDE.md): g = 9810 mm/s²


# ====================================================================== 3-axis dynamic envelopes
@dataclass
class AxisField:
    """The per-node outcome of one environment kind along one axis: equivalent stress S [MPa] (sine: max over frequency and phase; random: RMS = 1 sigma;
    shock: von Mises of the combined peaks), displacement magnitude U [mm] (the same measures), fatigue damage D (random: for the duration; sine: from the
    sweep; None when not computed)."""
    S: np.ndarray
    U: np.ndarray
    D: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class KindEnvelope:
    """Elementwise maximum over the axes of one kind (S, U), with the axis (0-2) that governs each node. The damage `D` is the SUM over the axes (the axes are
    tested one after the other on the same hardware, so the damage accumulates); `D_axis_max` is the plan's max(D) over the axes, kept as information."""
    kind: str
    S: np.ndarray
    U: np.ndarray
    D: np.ndarray | None
    axis_S: np.ndarray
    axis_U: np.ndarray
    axis_D: np.ndarray | None
    D_axis_max: np.ndarray | None = None


def _stack_max(arrs: list[np.ndarray], axes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    st = np.stack(arrs)
    k = np.argmax(st, axis=0)
    return np.max(st, axis=0), np.asarray(axes, dtype=np.int8)[k]


def envelope_of(per_axis: dict[int, AxisField], kind: str) -> KindEnvelope:
    """The envelope over the axes of `per_axis`: exactly the elementwise maximum of the per-axis arrays (no re-computation, no interpolation)."""
    axes = sorted(per_axis)
    if not axes:
        raise LoadsError("no axis result")
    S, aS = _stack_max([per_axis[a].S for a in axes], axes)
    U, aU = _stack_max([per_axis[a].U for a in axes], axes)
    damage = [d for d in (per_axis[a].D for a in axes) if d is not None]
    if len(damage) == len(axes):
        D_max, aD = _stack_max(damage, axes)
        D = np.sum(np.stack([np.asarray(d, np.float64) for d in damage]), axis=0).astype(np.float32)           # accumulated over the axes (Miner)
    else:
        D = D_max = aD = None
    return KindEnvelope(kind, S, U, D, aS, aU, aD, D_max)


def part_envelopes(mesh: Any, env: KindEnvelope) -> dict[str, dict[str, Any]]:
    """Per part: the largest envelope values over the nodes of the part, where they are, and the governing axis."""
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(mesh.part_names):
        nodes = np.unique(mesh.tets[mesh.tet_part == i])
        if not len(nodes):
            continue
        j = int(nodes[int(np.argmax(env.S[nodes]))])
        d: dict[str, Any] = {"max_S_MPa": float(env.S[j]), "at_mm": [float(x) for x in mesh.nodes[j]], "axis_S": AXIS_NAMES[int(env.axis_S[j])],
                             "max_U_mm": float(env.U[nodes].max())}
        if env.D is not None:
            d["max_D"] = float(env.D[nodes].max())                                         # governing: summed over the axes
            if env.D_axis_max is not None:
                d["max_D_single_axis"] = float(env.D_axis_max[nodes].max())
        out[str(name)] = d
    return out


@dataclass
class ThreeAxisResult:
    level: str
    kinds: list[str]
    axes: list[int]
    per_axis: dict[str, dict[int, AxisField]]
    envelopes: dict[str, KindEnvelope]
    parts: dict[str, dict[str, dict[str, Any]]]
    freq: np.ndarray
    eff_frac: np.ndarray                                  # (K, 3) effective-mass fractions of the basis modes in x, y, z
    sigma: float | None
    env_confirmed: bool
    req_confirmed: bool
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    engine: str = ""
    seconds: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)   # kind -> "document / revision / clause" of the table that was used
    env_digest: str = ""                                  # digest of the environments the run used (a later edit makes the result stale)
    run_digest: str = ""                                  # digest of everything the run depends on: study (without the space settings), mesh, damping, environments
    level_kinds: list[str] = field(default_factory=list)  # every kind the level has a table for (a run of fewer kinds is a partial run)
    damping: dict[str, Any] = field(default_factory=dict)  # the damping the run used: kind, zeta, source text, whether it comes from the environments
    blockers: list[str] = field(default_factory=list)     # reasons that forbid a PASS of any margin that rests on this result

    @property
    def inputs_confirmed(self) -> bool:
        return self.env_confirmed and self.req_confirmed

    @property
    def status(self) -> str:
        return "CONFIRMED_INPUTS" if self.inputs_confirmed else "UNCONFIRMED_INPUTS"

    @property
    def complete(self) -> bool:
        """All three axes AND every kind the level has a table for: only such a run may be called a 3-axis analysis of the level."""
        return set(self.axes) == set(AXES) and set(self.kinds) >= (set(self.level_kinds) or set(self.kinds))

    @property
    def label(self) -> str:
        """'3-axis' for a complete run; otherwise the axes and kinds that were run (a subset must never be called 3-axis)."""
        if self.complete:
            return "3-axis"
        return "+".join(AXIS_NAMES[a] for a in self.axes) + "-axis only" + (f", kinds {'+'.join(self.kinds)} only" if set(self.kinds) < set(self.level_kinds) else "")

    def current(self, study: Any, mesh: Any) -> bool:
        """The study (without its space settings), the mesh, the damping and the environments are still exactly what this result was computed from. A result
        without a digest, or checked without a mesh, is never current."""
        if study is None or mesh is None or not self.run_digest:
            return False
        es = environments_of(getattr(study, "space", None))
        return bool(es is not None and self.env_digest and es.content_digest() == self.env_digest and compute_run_digest(study, mesh) == self.run_digest)

    def stale_reason(self, study: Any, mesh: Any) -> str:
        """'' when current; otherwise what changed (for the message next to the result)."""
        if self.current(study, mesh):
            return ""
        es = environments_of(getattr(study, "space", None))
        if es is None or not self.env_digest or es.content_digest() != self.env_digest:
            return "the environments were edited after the run"
        return "the study, the mesh or the damping changed after the run"

    def confirmed_now(self, study: Any, mesh: Any) -> bool:
        """Environment AND requirement set are confirmed NOW and the result is the one of the present study, mesh and environment."""
        sp = getattr(study, "space", None)
        es = environments_of(sp)
        return bool(es is not None and es.is_confirmed() and sp is not None and sp.requirement_confirmed() and self.current(study, mesh))

    def scalar_sources(self) -> list[Any]:
        """The envelopes as equivalent-stress cases for `margins.compute_margins(extra_sources=...)`. Random: `sigma` times the RMS stress (only when the user entered
        the multiple with the table; otherwise the RMS stress is not compared with an allowable and `blockers` says so). The case name carries the axes and kinds
        of a partial run."""
        from .margins import ScalarSource
        out: list[Any] = []
        for k in self.kinds:
            e = self.envelopes[k]
            if k == "random":
                if self.sigma is None:
                    continue
                out.append(ScalarSource(self.case_name(k, self.sigma), e.S.astype(float) * self.sigma))
            else:
                out.append(ScalarSource(self.case_name(k, None), e.S.astype(float)))
        return out

    def case_name(self, kind: str, sigma: float | None) -> str:
        return source_name(self.level, kind, sigma, self.label)

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {"level": self.level, "label": self.label, "complete": self.complete, "axes": [AXIS_NAMES[a] for a in self.axes], "status": self.status,
                             "environment_confirmed": self.env_confirmed, "requirement_set_confirmed": self.req_confirmed, "engine": self.engine,
                             "seconds": round(self.seconds, 2), "modes": int(len(self.freq)), "lowest_frequencies_Hz": [float(f) for f in np.sort(self.freq)[:8]],
                             "damping": dict(self.damping), "sources": dict(self.sources), "blockers": list(self.blockers), "warnings": list(self.warnings),
                             "notes": list(self.notes), "kinds": {}}
        for k in self.kinds:
            e = self.envelopes[k]
            j = int(np.argmax(e.S))
            kd: dict[str, Any] = {"max_S_MPa": float(e.S[j]), "governing_axis": AXIS_NAMES[int(e.axis_S[j])], "max_U_mm": float(e.U.max()),
                                  "parts": self.parts.get(k, {}),
                                  "per_axis_max_S_MPa": {AXIS_NAMES[a]: float(self.per_axis[k][a].S.max()) for a in self.axes}}
            if e.D is not None:
                kd["max_D"] = float(e.D.max())
                kd["damage_rule"] = "sum over the axes (the axes are tested one after the other on the same hardware)"
                if e.D_axis_max is not None:
                    kd["max_D_single_axis"] = float(e.D_axis_max.max())
                    kd["per_axis_max_D"] = {AXIS_NAMES[a]: float(np.max(d_a)) for a in self.axes for d_a in [self.per_axis[k][a].D] if d_a is not None}
            if k == "random":
                kd["sigma"] = self.sigma
                kd["note"] = "S = RMS (1 sigma) von Mises; compared with an allowable only as sigma x RMS when the user entered sigma"
            d["kinds"][k] = kd
        return d


def source_name(level: str, kind: str, sigma: float | None, label: str = "3-axis") -> str:
    return f"{label} {level} {kind}" + (f" ({sigma:g} sigma)" if kind == "random" and sigma is not None else " envelope")


# ---------------------------------------------------------------------- what a run depends on (the digest of a result)
def mesh_fingerprint(mesh: Any) -> str:
    """Digest of the mesh content (`state_cache.mesh_digest`): nodes, elements, parts, every named surface / edge / point selection the supports and loads sit on, the node aliases,
    the tie table of separately meshed parts and the per-element / per-node data. Two meshes with the same fingerprint give the same dynamic result."""
    from ..state_cache import mesh_digest                # one digest of the mesh for the 3-axis result and the state cache
    return mesh_digest(mesh)


def _strings_in(x: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, dict):
        for v in x.values():
            out |= _strings_in(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            out |= _strings_in(v)
    return out


def run_digest_payload(study: Any) -> dict[str, Any]:
    """The part of the study dict that a 3-axis result depends on.
    INCLUDED: everything of the study (materials, part materials, selections, constraints, mesh settings, solver settings, masses, joints, bolts, welds, thermal,
    dynamics with its modal settings and damping, ...) EXCEPT what is listed here. The mesh itself, the damping that is used and the content of the environments are
    added by `compute_run_digest`.
    LEFT OUT (they do not change the dynamic result): the study name, the space settings (the environments enter separately; the requirement set does not enter the
    dynamics), the display-only hidden parts, `solver.nproc` (threads / processes, not physics), and the STATIC load cases that the dynamics does not use - among them
    the QSL cases the generator makes. A load case stays in the digest when another part of the study refers to it by name (a prestress case, a force excitation, a
    preload case) or when it holds a rigidly coupled load (the rigid-body check of the modal basis reads it)."""
    d = study.to_dict()
    for k in ("space", "hidden_parts", "name"):
        d.pop(k, None)
    if isinstance(d.get("solver"), dict):
        d["solver"] = {k: v for k, v in d["solver"].items() if k != "nproc"}
    lcs = d.pop("load_cases", None) or []
    named = _strings_in(d)
    d["load_cases"] = [lc for lc in lcs if lc.get("name") in named or any(isinstance(ld, dict) and ld.get("coupling") == "rigid" for ld in lc.get("loads", []))]
    return d


def compute_run_digest(study: Any, mesh: Any) -> str:
    """Digest of everything the 3-axis run depends on (`run_digest_payload` lists what is in and what is out of the study dict), the mesh fingerprint, the damping that
    is used and the content of the environments."""
    d = run_digest_payload(study)
    es = environments_of(getattr(study, "space", None))
    payload = {"study": d, "mesh": mesh_fingerprint(mesh), "damping": effective_damping(study)[1], "environment": es.content_digest() if es is not None else ""}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def effective_damping(study: Any) -> tuple[Any, dict[str, Any], list[str]]:
    """(the DampingSpec the 3-axis run uses, its description, warnings). The damping of the launch analysis is an input of the ENVIRONMENT (`damping_zeta` with its
    source text, part of the confirmation); without it the damping setting of the study is used with a warning, and no margin that rests on it can be PASS."""
    from ..dynamics.model import DampingSpec
    es = environments_of(getattr(study, "space", None))
    if es is not None and es.damping_zeta is not None:
        z = float(es.damping_zeta)
        return DampingSpec(kind="modal", zeta=z), {"kind": "modal", "zeta": z, "source": es.damping_source, "from_environment": True}, []
    ds = study.dynamics.damping
    info = {"kind": ds.kind, "zeta": float(ds.zeta), "source": "damping setting of the study (not entered in the environments)", "from_environment": False}
    return copy.deepcopy(ds), info, [f"damping: not entered in the environments - the damping of the study ({ds.kind}, zeta {float(ds.zeta):g}) is used; it is not "
                                     f"confirmed with the environments, so no margin from this analysis can be PASS"]


# ---------------------------------------------------------------------- per-axis study
def _axis_study(study: Any, level: EnvLevel, kinds: list[str], axis: int, srs_extend_to: float | None = None) -> Any:
    """The single-axis study of one axis: the specs of the kinds, the damping of the environments (`effective_damping`). `srs_extend_to` [Hz]: the shock table is
    continued flat at its last value (the ZPA) up to this frequency, so that modes ABOVE the top of the table respond with the ZPA instead of zero (the single-axis
    dynamics gives zero outside a table; the space path does not want that)."""
    from ..dynamics.model import DynamicsSettings, Excitation, HarmonicSpec, RandomSpec, ShockSpec
    st = copy.copy(study)
    ds = DynamicsSettings.from_dict(study.dynamics.to_dict())
    ds.damping = effective_damping(study)[0]
    ds.enabled = True
    ds.harmonic, ds.random, ds.shock = HarmonicSpec(), RandomSpec(), ShockSpec()
    ds.transient.enabled = False
    dof = axis + 1
    t = level.table("sine")
    if "sine" in kinds and t is not None:
        f = [p[0] for p in t.points]
        ds.harmonic = HarmonicSpec(enabled=True, f_min=min(f), f_max=max(f), excitation=[Excitation(kind="base_acc", dof=dof, table=[list(p) for p in t.points], unit="g")],
                                   sweep_rate=float(t.sweep_rate_oct_min or 0.0), sweeps=int(t.sweeps or 1), static_correction=True)
    t = level.table("random")
    if "random" in kinds and t is not None:
        ds.random = RandomSpec(enabled=True, asd=[list(p) for p in t.points], dof=dof, duration_h=float(t.duration_s) / 3600.0 if t.duration_s else 1.0)
    t = level.table("shock")
    if "shock" in kinds and t is not None:
        srs = [list(p) for p in t.points]
        if srs_extend_to is not None and srs and srs_extend_to > srs[-1][0]:
            srs.append([float(srs_extend_to), srs[-1][1]])
        ds.shock = ShockSpec(enabled=True, pulse="srs_table", srs_table=srs, dof=dof, combine=level.combine, missing_mass=level.missing_mass, transient=False)
    st.dynamics = ds
    return st


def _fields_of(out: Any, kinds: list[str], level: EnvLevel) -> dict[str, AxisField]:
    res: dict[str, AxisField] = {}
    if "sine" in kinds and out.harmonic is not None:
        h = out.harmonic
        sw = h.get("sweep")
        s = out.summary().get("harmonic", {})
        res["sine"] = AxisField(np.asarray(h["env_S"], np.float32), np.asarray(h["env_U"], np.float32),
                                np.asarray(sw["D"], np.float32) if sw is not None and "D" in sw else None,
                                {k: s[k] for k in ("f_range_Hz", "max_displacement_mm", "max_mises_MPa", "f_at_max_mises_Hz") if k in s})
    if "random" in kinds and out.random is not None:
        r = out.random
        res["random"] = AxisField(np.asarray(r["S_rms"], np.float32), np.linalg.norm(np.asarray(r["U_rms"], float), axis=1).astype(np.float32),
                                  np.asarray(r["D"], np.float32), dict(r.get("summary", {})))
    if "shock" in kinds and out.shock is not None:
        sh = out.shock
        res["shock"] = AxisField(np.asarray(sh["mises"], np.float32), np.linalg.norm(np.asarray(sh["U"], float), axis=1).astype(np.float32), None,
                                 dict(sh.get("summary", {})))
    return res


def unit_axis_cases(axes: tuple[int, ...] = AXES) -> list[tuple[tuple[Any, ...], Any]]:
    """[(key, LoadCase)] of the static states the dynamics needs per axis (the unit base acceleration): the key is the one `dynamics.run` looks up."""
    from ..study import Gravity, LoadCase
    out = []
    for a in axes:
        d = np.zeros(3)
        d[a] = 1.0
        out.append((("b", tuple(np.round(d, 12))), LoadCase(f"_dyn_res_{a}", [Gravity(vector=[float(x) for x in d])])))
    return out


def analyze_axes(study: Any, mesh: Any, basis: Any, statics_by_key: dict[Any, Any], workdir: Path, log: Callable[[str], None], level: EnvLevel,
                 kinds: list[str], axes: tuple[int, ...] = AXES, on_line: Any = None,
                 cancel: Any = None) -> tuple[dict[str, dict[int, AxisField]], list[str]]:
    """One single-axis dynamics run per axis over the SHARED basis and statics; returns {kind: {axis: AxisField}}. A run per axis is the very function
    (`run_dynamics`) of an independent single-axis analysis (with the damping of the environments, and with the shock table continued at its ZPA above its last point
    when the basis has modes there). Returns (the fields, the warnings of the runs)."""
    from ..dynamics.run import run_dynamics
    per: dict[str, dict[int, AxisField]] = {k: {} for k in kinds}
    warnings: list[str] = []
    ext: float | None = None
    t_sh = level.table("shock")
    if "shock" in kinds and t_sh is not None and t_sh.points and len(basis.freq) and float(np.max(basis.freq)) > t_sh.points[-1][0]:
        ext = float(np.max(basis.freq)) * 1.01
    for a in axes:
        st = _axis_study(study, level, kinds, a, ext)
        log(f"Space loads: {level.name} environment along {AXIS_NAMES[a]}")
        out = run_dynamics(st, mesh, Path(workdir) / f"axis_{AXIS_NAMES[a].lower()}", log, on_line, cancel, basis=basis, statics_by_key=statics_by_key)
        warnings += [w for w in out.warnings if w not in warnings]
        for k, fld in _fields_of(out, kinds, level).items():
            per[k][a] = fld
    return per, warnings


def eff_fractions(basis: Any) -> np.ndarray:
    """(K, 3) effective-mass fractions of the basis modes along x, y, z. An axis whose total mass is zero or unknown (and the model mass is no help) gives NaN - an
    UNKNOWN share, never a division by 1; the checks that read the fractions treat NaN as unknown and do not let it become a PASS."""
    tot = np.asarray(getattr(basis, "total_mass", np.zeros(6)), float)[:3]
    model = float(getattr(basis, "model_mass", 0.0) or 0.0)
    fallback = model if math.isfinite(model) and model > 0 else np.nan
    tot = np.where(tot > 0, tot, fallback)
    return np.asarray(basis.eff_mass, float)[:, :3] / tot


def srs_range_findings(freq: Any, frac: np.ndarray, table: EnvTable, axes: list[int]) -> tuple[list[str], list[str]]:
    """(blockers, notes) of the shock table against the modes. The single-axis shock gives a mode ZERO response outside the table, so ANY mode (positive frequency)
    BELOW the lowest table frequency is left out of the shock response: it blocks every PASS and is listed, whatever its share of the effective mass (a global share
    cannot say whether the mode loads a part that matters; no threshold decides it). The blocker goes away only when the table is extended down to cover the modes.
    A mode ABOVE the table takes the last value (the ZPA, the table is continued flat by `analyze_axes`): a note. `frac` / `axes` only add the shares to the text."""
    if not table.points:
        return [], []
    lo, hi = float(table.points[0][0]), float(table.points[-1][0])
    f = np.asarray(freq, float)

    def share(k: int) -> str:
        parts = [f"{AXIS_NAMES[a]} {frac[k, a] * 100:.1f} %" for a in axes if k < len(frac) and a < frac.shape[1] and math.isfinite(float(frac[k, a]))]
        return "; effective mass " + ", ".join(parts) if parts else ""

    below = [f"mode {k + 1} ({f[k]:.4g} Hz{share(k)})" for k in range(len(f)) if 0 < f[k] < lo]
    above = [f"mode {k + 1} ({f[k]:.4g} Hz{share(k)})" for k in range(len(f)) if f[k] > hi]
    blockers = [f"shock table covers {lo:.4g}-{hi:.4g} Hz; {len(below)} mode(s) below it respond with ZERO in the shock analysis: " + "; ".join(below[:8])
                + (" ..." if len(below) > 8 else "") + " - extend the table down to cover them (every mode below the table blocks a PASS, whatever its mass share)"] if below else []
    notes = [f"modes above the top of the shock table ({hi:.4g} Hz) respond to the last table value (the ZPA): " + "; ".join(above[:6]) + (" ..." if len(above) > 6 else "")] \
        if above else []
    return blockers, notes


def build_result(study: Any, mesh: Any, level: EnvLevel, kinds: list[str], axes: list[int], per: dict[str, dict[int, AxisField]], basis: Any, *,
                 env_confirmed: bool, req_confirmed: bool, warnings: list[str] | None = None, seconds: float = 0.0,
                 env_digest: str = "", run_digest: str = "") -> ThreeAxisResult:
    envs = {k: envelope_of(per[k], k) for k in kinds}
    parts = {k: part_envelopes(mesh, envs[k]) for k in kinds}
    frac = eff_fractions(basis)
    t_rnd, t_sin = level.table("random"), level.table("sine")
    sigma = t_rnd.sigma if t_rnd is not None else None
    srcs: dict[str, str] = {}
    for k in kinds:
        t: EnvTable | None = level.table(k)
        if t is not None:
            srcs[k] = " / ".join(x for x in (t.document, t.revision, t.clause) if x.strip())
    notes: list[str] = []
    blockers: list[str] = []
    warns = list(warnings or [])
    if "random" in kinds and t_rnd is not None and not t_rnd.duration_s:
        notes.append("no duration entered for the random table - the fatigue damage of the random environment is given for one hour")
    if "sine" in kinds and t_sin is not None and not t_sin.sweep_rate_oct_min:
        notes.append("no sweep rate entered for the sine table - the sweep fatigue was not computed")
    if "random" in kinds and sigma is None:
        blockers.append("the random table has no sigma multiple: the RMS stress is reported but is NOT compared with an allowable, so the random environment is not "
                        "verified (enter sigma with the table)")
    t_sh = level.table("shock")
    if "shock" in kinds and t_sh is not None:
        b, n = srs_range_findings(basis.freq, frac, t_sh, list(axes))
        blockers += b
        warns += [x for x in b if x not in warns]
        notes += n
    lk = level.kinds()
    if set(axes) != set(AXES) or not set(lk) <= set(kinds):
        parts_txt = []
        if set(axes) != set(AXES):
            parts_txt.append("axes " + ", ".join(AXIS_NAMES[a] for a in sorted(axes)) + " only")
        if not set(lk) <= set(kinds):
            parts_txt.append("kinds " + ", ".join(kinds) + f" only (the level also has {', '.join(k for k in lk if k not in kinds)})")
        blockers.append("partial run - " + "; ".join(parts_txt) + ": it is not a 3-axis analysis of the level")
    _spec, dinfo, dwarn = effective_damping(study)
    warns += [w for w in dwarn if w not in warns]
    if not dinfo["from_environment"]:
        blockers.append("the damping is not entered in the environments (it is the damping setting of the study)")
    notes.append(f"damping: {dinfo['kind']}, zeta {dinfo['zeta']:g} ({dinfo['source'] or 'no source text'})")
    return ThreeAxisResult(level.name, list(kinds), list(axes), {k: per[k] for k in kinds}, envs, parts, np.asarray(basis.freq, float), frac, sigma,
                           env_confirmed, req_confirmed, warns, notes, getattr(basis, "engine", ""), seconds, srcs, env_digest, run_digest, list(lk), dinfo, blockers)


def resolve_level(study: Any, level: str, kinds: list[str] | None = None) -> tuple[EnvironmentSet, EnvLevel, list[str]]:
    """The environments, the level and the kinds a 3-axis run would use; raises `LoadsError` (cheap, before any meshing or solving) when something is missing or
    invalid: an unknown level name, an unknown kind, a combination other than srss / cqc when a shock is run, a table that is not usable. A table whose source
    (document / revision / clause) is not filled is NOT refused - it computes, and the environment cannot be confirmed, so the result stays UNCONFIRMED."""
    from .environments import ENV_LEVELS, LAUNCH_AXES
    es = environments_of(study.space)
    if es is None:
        raise LoadsError("there are no environments - enter them first (fea_space_env_set)")
    if level not in ENV_LEVELS:
        raise LoadsError(f"unknown level '{level}' (one of {', '.join(ENV_LEVELS)})")
    lv = es.level(level)
    if lv is None or not lv.kinds():
        raise LoadsError(f"the level '{level}' has no table (levels with tables: {', '.join(x.name for x in es.levels if x.kinds()) or 'none'})")
    unknown = [k for k in (kinds or []) if k not in ENV_KINDS]
    if unknown:
        raise LoadsError(f"unknown kind(s) {', '.join(map(str, unknown))} (one of {', '.join(ENV_KINDS)})")
    use = list(dict.fromkeys(kinds or lv.kinds()))
    bad = [k for k in use if k not in lv.kinds()]
    if bad or not use:
        raise LoadsError(f"the level '{level}' has no table for: {', '.join(bad) or 'the requested kinds'}")
    if "shock" in use and lv.combine not in ("srss", "cqc"):
        raise LoadsError(f"the shock combination of the level '{level}' must be srss or cqc (it is '{lv.combine}')")
    z = es.damping_zeta
    if z is not None and not (math.isfinite(z) and 0.0 < z < 1.0):
        raise LoadsError(f"the damping ratio of the environments (damping_zeta {z!r}) must be between 0 and 1 (exclusive) - a zero or negative damping has no finite response")
    problems: list[str] = []
    if es.launch_axis and es.launch_axis not in LAUNCH_AXES:
        problems.append(f"launch_axis must be one of {', '.join(LAUNCH_AXES)}")
    for k in use:
        t = lv.table(k)
        problems += [f"{level} / {k}: {p}" for p in (t.problems(include_source=False) if t is not None else [])]
    if problems:
        raise LoadsError("the environment is not usable: " + "; ".join(problems[:4]))
    if study.dynamics.prestress.enabled:
        raise LoadsError("prestressed dynamics is not supported by the 3-axis analysis (the modal basis is shared between the axes)")
    return es, lv, use


def run_3axis(study: Any, mesh: Any, workdir: Path, log: Callable[[str], None] = lambda s: None, *, level: str, kinds: list[str] | None = None,
              axes: tuple[int, ...] = AXES, on_line: Any = None, cancel: Any = None) -> ThreeAxisResult:
    """The environment `level` of `study.space` along `axes`, over one modal basis (the modal basis of the dynamics: `study.dynamics.modes` / `f_max`, damping
    from the environments, see `effective_damping`). Needs a meshed, constrained study. Prestressed dynamics is not supported (the basis is shared between the axes).
    The result carries the digest of the study, the mesh, the damping and the environments (`run_digest`), the reasons that forbid a PASS (`blockers`) and the
    warnings; a run of fewer than three axes or of fewer kinds than the level has is labelled as such."""
    from ..dynamics.modal import DynamicsError, run_basis, static_solutions
    t0 = time.time()
    sp = study.space
    es, lv, use = resolve_level(study, level, kinds)
    digest = compute_run_digest(study, mesh)                          # of the inputs as they are NOW (before the run)
    base = copy.copy(study)
    base.dynamics = copy.deepcopy(study.dynamics)
    base.dynamics.enabled = True
    base.dynamics.damping = effective_damping(study)[0]
    cases = unit_axis_cases(tuple(axes))
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    log("Space loads: the modal basis (shared by the axes)")
    basis = run_basis(base, mesh, wd, log, on_line, cancel, static_cases=[lc for _, lc in cases], mapped=True)
    if basis.K == 0:
        raise DynamicsError("Space loads: the modal basis is empty (no mode with a positive frequency)")
    if basis.statics is not None and len(basis.statics) == len(cases):
        statics = list(basis.statics)
    else:
        from ..gravity_states import static_states
        statics = static_states(base, mesh, [lc for _, lc in cases], wd, log, on_line, cancel)       # the unit states of the run (or solved once, in one job)
        if statics is None:
            statics = static_solutions(base, mesh, [lc for _, lc in cases], wd, log, on_line, cancel)
    sk = {key: st for (key, _), st in zip(cases, statics)}
    per, warns = analyze_axes(study, mesh, basis, sk, wd, log, lv, use, tuple(axes), on_line, cancel)
    from ..dynamics.modal import basis_warnings
    f_top = max((p[0] for k in use for t in [lv.table(k)] if t is not None for p in t.points), default=None)
    warns += [w for w in basis_warnings(basis, set(axes), f_top) if w not in warns]
    res = build_result(study, mesh, lv, use, list(axes), per, basis, env_confirmed=es.is_confirmed(),
                       req_confirmed=bool(sp is not None and sp.requirement_confirmed()), warnings=warns, seconds=time.time() - t0,
                       env_digest=es.content_digest(), run_digest=digest)
    miss: list[str] = []
    for k in use:
        tk = lv.table(k)
        if tk is None or not tk.provenance_ok:
            miss.append(k)
    if miss:
        res.notes.append(f"the source (document / revision / clause) of the table(s) {', '.join(miss)} is not filled - the environment cannot be confirmed")
    if not res.env_confirmed:
        res.notes.append("the environments are not confirmed - the results are indicative and no margin can be PASS")
    return res


def attach(results: Any, res3: ThreeAxisResult | None) -> None:
    """Keeps the 3-axis result with the solved results (in memory; it is not saved with the project)."""
    setattr(results, RESULT_ATTR, res3)


def attached(results: Any) -> ThreeAxisResult | None:
    r = getattr(results, RESULT_ATTR, None)
    return r if isinstance(r, ThreeAxisResult) else None


def remember(owner: Any, res3: ThreeAxisResult | None) -> None:
    """Keeps the result on `owner` (an MCP session / a window). It is NOT bound to a study by name: `recall` gives it back only while the study, the mesh, the damping
    and the environments still hash to the digest the result carries. Also attached to `owner.results` when there are solved results."""
    setattr(owner, "_" + RESULT_ATTR, res3)
    if getattr(owner, "results", None) is not None:
        attach(owner.results, res3)


def forget(owner: Any) -> None:
    """Drops the kept result (the study changed)."""
    setattr(owner, "_" + RESULT_ATTR, None)


def kept_result(owner: Any) -> ThreeAxisResult | None:
    """The result kept on `owner`, current or not."""
    cand = getattr(owner, "_" + RESULT_ATTR, None)
    if not isinstance(cand, ThreeAxisResult):
        cand = attached(getattr(owner, "results", None)) if getattr(owner, "results", None) is not None else None
    return cand


class KeptSnapshot:
    """A frozen copy of what `kept_result(owner)` answers, taken on the GUI thread: a worker thread that computes the margins reads this object instead of the
    window (no QMainWindow access off the GUI thread, and the result cannot change under the job). `kept` is the result at the time of the snapshot."""

    def __init__(self, owner: Any) -> None:
        self.kept = kept_result(owner)
        setattr(self, "_" + RESULT_ATTR, self.kept)
        self.results = None                       # kept_result must not look further than the snapshot


def snapshot_owner(owner: Any) -> KeptSnapshot:
    return KeptSnapshot(owner)


def recall(owner: Any, study: Any, mesh: Any) -> ThreeAxisResult | None:
    """The kept result when it is the result of THIS study and mesh (digest), else None."""
    cand = kept_result(owner)
    return cand if cand is not None and cand.current(study, mesh) else None


STALE_3AXIS = "3-axis result is out of date - re-run the 3-axis analysis (the study, mesh, damping or environments changed after it)"


# ====================================================================== margins from the envelopes
def margins_with_envelopes(study: Any, results: Any, res3: ThreeAxisResult | None, max_places: int | None = None) -> Any:
    """`margins.compute_margins` with the 3-axis envelopes as extra equivalent-stress cases. Margins that rest on an unconfirmed environment, on a result that is
    stale (the study, mesh, damping or environments changed after the run) or on a result with `blockers` (random without sigma, a mode below the shock table, a
    partial run, damping not entered in the environments) can never be PASS: the rows of the envelope cases become `unconfirmed` and the verdict falls back to
    UNCONFIRMED_REQUIREMENTS. A negative margin with confirmed inputs stays FAIL."""
    from . import margins as MG
    srcs = tuple(res3.scalar_sources()) if res3 is not None else ()
    kw = {} if max_places is None else {"max_places": max_places}
    summ = MG.compute_margins(study, results, extra_sources=srcs, **kw)
    if res3 is None:
        return summ
    es = environments_of(study.space)
    mesh = getattr(results, "mesh", None)
    env_ok = bool(es is not None and es.is_confirmed() and res3.current(study, mesh))
    reasons: list[str] = []
    if not env_ok:
        reasons.append("the launch environments (sine / random / shock tables) are not confirmed, or the environments / study / mesh / damping changed after the 3-axis run")
    reasons += [b for b in res3.blockers if b not in reasons]
    if not reasons:
        return summ
    names = {s.name for s in srcs}
    for r in summ.rows:
        if r.case in names and r.status in ("pass", "fail") and (not env_ok or r.status == "pass"):
            r.negative = r.negative or r.status == "fail"
            r.status = "unconfirmed"
            r.reasons += [x for x in reasons if x not in r.reasons]
    summ.reasons += [x for x in reasons if x not in summ.reasons]
    if summ.verdict == MG.PASS or (summ.verdict == MG.FAIL and not any(r.status == "fail" for r in summ.rows)):
        summ.verdict = MG.UNCONFIRMED
    return summ


def sign_off_margins(study: Any, results: Any, owner: Any, mesh: Any = None, max_places: int | None = None) -> Any:
    """The margins that a report / a tool / the margins tab shows as THE verdict: with the 3-axis envelopes of `owner` when it keeps a result that is current for
    this study and mesh (`recall`), else the plain margins. Every verdict that is shown to a person goes through here, so none can be PASS while a kept dynamic
    envelope fails or is blocked. A kept result that is NOT current (something that the dynamics depends on changed) is not used, but it is not forgotten either: the
    plain margins then carry the blocker `STALE_3AXIS` and cannot be PASS (a negative static margin stays FAIL) - an out-of-date envelope must never silently turn back
    into a static PASS."""
    from . import margins as MG
    m = mesh if mesh is not None else getattr(results, "mesh", None)
    res3 = recall(owner, study, m)
    if res3 is not None:
        return margins_with_envelopes(study, results, res3, max_places)
    plain = MG.compute_margins(study, results) if max_places is None else MG.compute_margins(study, results, max_places=max_places)
    kept = kept_result(owner)
    if kept is not None:
        why = kept.stale_reason(study, m) or "no mesh to compare with"
        reason = f"{STALE_3AXIS} [{why}]"
        if reason not in plain.reasons:
            plain.reasons.append(reason)
        if plain.verdict == MG.PASS:
            plain.verdict = MG.UNCONFIRMED
    return plain

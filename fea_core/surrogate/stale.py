"""Is the surrogate still the surrogate of THIS study? (F15-S4)

A surrogate is fitted to the design points of a parametric study of ONE model. If the model changes afterwards (a load, a constraint,
the geometry, a material that is not swept, the mesh settings) its answers and bands describe a model that no longer exists.
The study fingerprint is a SHA-256 of the inputs of the calculation – the study without the post-processing fields and with the
SWEPT parameters neutralised – plus the digest of the STEP file; it is stored in the surrogate file together with the sweep
specification and compared when the model predicts, is reported or saved, and before more points are added to it.

Neutralised (so that taking over a design point or moving a swept value does not make the model stale): the material of a swept part,
a swept material property, the mesh size, a swept print setting, a swept sketch parameter (and the built geometry of a sketch).
A swept load factor scales the loads of the load cases it applies to by ONE common number, so those loads are compared up to that
number only: the fields that the load handler scales (the force vector, the pressure and its gradient, the force and moment of a remote
load, ω of a rotation as √factor …, found with the handler itself, `load_kinds`) are divided by one common reference of their group
(the norm of the linear ones; sign kept). What stays in the fingerprint: the DIRECTION of every force, the ratios between the loads of
a group (one of two forces ×5, another case's pressure ×10), the application points, the loads that the factor does not scale (the
temperature of a thermal load) and the loads of the cases that are not swept. A group is the set of all load cases for `load_factor`
(minus the cases of a `case_factor`) or one load case for `case_factor:<case>`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# fields of the study that are no inputs of the calculation (display, post-processing, names, file locations)
_IGNORED = ("format", "version", "features", "name", "geometry", "fatigue", "hidden_parts", "assessment", "combinations", "cad_link")
_CHUNK = 1 << 20


_DIGESTS: dict[tuple, str] = {}                         # (path, mtime, size) -> digest: the window asks again every time the tab is shown


def _digest_file(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        if not p.is_file():
            return ""
        stat = p.stat()
        key = (str(p), stat.st_mtime_ns, stat.st_size)
        if key in _DIGESTS:
            return _DIGESTS[key]
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
        if len(_DIGESTS) > 16:
            _DIGESTS.clear()
        _DIGESTS[key] = h.hexdigest()
        return _DIGESTS[key]
    except OSError:
        return ""


def _ones(v: Any) -> Any:
    """The same structure with every number replaced by 1.0 (to find the fields that a load handler scales whatever their values)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return 1.0
    if isinstance(v, list):
        return [_ones(x) for x in v]
    if isinstance(v, dict):
        return {k: _ones(x) for k, x in v.items()}
    return v


def _flat(v: Any) -> list[float]:
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in _flat(e)]
    return []


def _map(v: Any, fn) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return fn(float(v))
    if isinstance(v, list):
        return [_map(x, fn) for x in v]
    return v


def _scaled_fields(load: dict) -> dict[str, float]:
    """{field: exponent} of the numeric fields of one stored load that a load factor g scales (value × g^exponent): found by running the
    scale rule of the load kind (`load_kinds`, the one the sweep uses) on a copy with every number set to 1."""
    from types import SimpleNamespace

    from ..load_kinds import load_handler
    probe = SimpleNamespace(**{k: _ones(v) for k, v in load.items() if isinstance(k, str)})
    before = {k: _flat(v) for k, v in probe.__dict__.items()}
    h = load_handler(probe)
    try:
        if h is not None and h.scale is not None:
            h.scale(probe, 4.0)
        elif getattr(probe, "vector", None):               # a kind without a handler: the generic rule of sweep._scale_loads
            probe.vector = [4.0 * float(x) for x in probe.vector]
        elif isinstance(getattr(probe, "value", None), (int, float)):
            probe.value = 4.0 * float(probe.value)
    except Exception:  # noqa: BLE001 – a rule that raises for this load: its numbers stay in the fingerprint
        return {}
    out: dict[str, float] = {}
    for k, v in probe.__dict__.items():
        a, b = before.get(k, []), _flat(v)
        if a and len(a) == len(b) and a != b:
            e = math.log(max(b[0], 1e-300) / a[0]) / math.log(4.0)
            if abs(e - 1.0) < 1e-6 or abs(e - 0.5) < 1e-6:
                out[k] = 1.0 if abs(e - 1.0) < 1e-6 else 0.5
    return out


def _round(x: float) -> float:
    return float(f"{x:.7g}") + 0.0                           # 7 significant digits (a rescaled copy differs by 1e-16); + 0.0 turns -0.0 into 0.0


def _normalise_group(loads: list) -> None:
    """Divides the numbers that a common load factor scales by ONE reference of the group (in place on the stored loads): the norm of the
    linearly scaled numbers (sign of the first non-zero one), for ω-type numbers (∝ √factor) the square root of that. The direction of
    each force and the ratios inside the group stay; the common factor goes."""
    items = [(ld, f, e) for ld in loads if isinstance(ld, dict) for f, e in _scaled_fields(ld).items()]
    lin = [x for ld, f, e in items if e == 1.0 for x in _flat(ld[f])]
    half = [x for ld, f, e in items if e == 0.5 for x in _flat(ld[f])]
    nrm = math.sqrt(sum(x * x for x in lin))
    if nrm > 0.0 and math.isfinite(nrm):
        ref = nrm * (1.0 if next(x for x in lin if x != 0.0) > 0 else -1.0)
    else:
        sq = sum(x * x for x in half)                      # only ω-type numbers: x² is proportional to the factor
        ref = sq if sq > 0.0 and math.isfinite(sq) else 0.0
    if ref == 0.0:
        return
    for ld, f, e in items:
        div = ref if e == 1.0 else math.sqrt(abs(ref))
        ld[f] = _map(ld[f], lambda x, div=div: _round(x / div))


def _load_groups(d: dict, keys: list[str]) -> list[list]:
    """The groups of loads that the swept load parameters scale together (see the module text)."""
    cases = [lc for lc in d.get("load_cases", []) if isinstance(lc, dict) and isinstance(lc.get("loads"), list)]
    own = {k.partition(":")[2] for k in keys if k.startswith("case_factor:")}
    groups: list[list] = [lc["loads"] for lc in cases if lc.get("name") in own]
    if "load_factor" in keys:
        groups.append([ld for lc in cases if lc.get("name") not in own for ld in lc["loads"]])
    return groups


def _neutralise(d: dict, keys: list[str]) -> None:
    """Replaces the swept values of the study dictionary `d` by None (in place), the swept load factors by a common normalisation of
    the loads they scale. A key that does not fit the study is ignored."""
    try:
        for g in _load_groups(d, keys):
            _normalise_group(g)
    except (KeyError, TypeError, AttributeError, ValueError, StopIteration):
        pass
    for k in keys:
        try:
            kind, _, rest = k.partition(":")
            if k == "mesh_size":
                d["mesh"]["size"] = None
            elif kind == "material":
                d["part_materials"][rest] = None
            elif kind == "prop":
                mat, _, fld = rest.rpartition(":")
                d["materials"][mat][fld] = None
            elif kind == "print":
                part, _, fld = rest.rpartition(":")
                d["printing"]["parts"][part][fld] = None
            elif kind == "sketch":
                d["sketch"]["params"][rest] = None
        except (KeyError, TypeError, AttributeError):
            continue


def study_fingerprint(study, keys: list[str] | None = None, geometry_path: str | Path | None = None) -> str:
    """SHA-256 (hex) of the inputs of the calculation of `study` with the swept parameters `keys` neutralised. `geometry_path` = the STEP
    file (its digest is part of the fingerprint; a study drawn in the sketcher is defined by its sketch instead)."""
    d = copy.deepcopy(study.to_dict()) if hasattr(study, "to_dict") else copy.deepcopy(dict(study))
    for k in _IGNORED:
        d.pop(k, None)
    for lc in d.get("load_cases", []):
        if isinstance(lc, dict):
            lc.pop("category", None)
    sk = d.get("sketch")
    if isinstance(sk, dict):
        sk.pop("built", None)                              # the result of the rebuild, not an input
    _neutralise(d, list(keys or []))
    blob = json.dumps(d, sort_keys=True, default=str, separators=(",", ":"))
    h = hashlib.sha256(blob.encode("utf-8"))
    if not isinstance(sk, dict):
        h.update(b"|geometry|")
        h.update(_digest_file(geometry_path).encode("ascii"))
    return h.hexdigest()


def stale_reason(model, study, geometry_path: str | Path | None = None) -> str:
    """"" = the surrogate belongs to the study (or cannot be compared: no fingerprint was recorded, which the fit and the loader warn
    about); otherwise the sentence that says why not."""
    fp = getattr(model, "fingerprint", "")
    if not fp or study is None:
        return ""
    keys = list(getattr(model, "fingerprint_keys", None) or [p.key for p in model.params])    # the swept keys the stored fingerprint was made with
    now = study_fingerprint(study, keys, geometry_path)
    if now != fp:
        return ("The study was changed after the surrogate was fitted (geometry, loads, constraints, materials or settings): run the parametric "
                "study again and fit a new surrogate – the error bars describe the old model.")
    return ""

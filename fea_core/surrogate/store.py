"""The file of a surrogate (`surrogate.npz` inside a .fea): written without pickle, READ AS UNTRUSTED INPUT (F15-S4).

A .fea can come from anybody, so the loader
  * checks every zip member (name, declared size) and every .npy header (dtype, shape, byte count) BEFORE a byte of data is read, so a
    "zip bomb" or an object array never reaches memory;
  * accepts only known fields with their types and ranges (no attribute is ever set from a name in the file), bounds the nesting of the
    JSON header and every list and string;
  * takes from the file nothing but the design points, the parameter definitions, the split (train / calibration indices, which must be
    unique, disjoint, non-empty and belong to the category and have a value of the KPI) and the hyperparameters (finite, inside the bounds
    of the optimiser) – the transform, the standardisation, the band factor, the accuracy figures and the verdict are RECOMPUTED, so a file
    cannot claim a reliable model or a narrow band;
  * bounds the work: rows, parameters, models and the cubic cost of the fits.
Whatever is wrong ends as `SurrogateError`; the caller (project.load_surrogate_file) then drops the surrogate and the project opens.
"""
from __future__ import annotations

import json
import math
import os
import re
import zipfile
from typing import Any, cast

import numpy as np

from .fit import (FORMAT, NO_FINGERPRINT, MAX_CATEGORIES, MAX_FAILED, MAX_MEMBERS, MAX_PARAMS, MAX_ROWS, SURROGATE_KPIS, ParamSpec, SurrogateError,
                  SurrogateModel, _fit_member, min_train)
from .design import MAX_CONTINUOUS
from .model import BackendError
from .support import spanned_rank

MAX_FILE_BYTES = 16 << 20          # the largest file the program writes is below 9 MB (header 2 M characters + the two tables); a foreign file above this is not read
MAX_META_CHARS = 2_000_000
MAX_DEPTH = 8                       # nesting of the JSON header (the real one is 5 deep)
MAX_NODES = 5000                    # values inside the stored sweep specification
MAX_STRING = 1000
MAX_NAME = 200
MAX_COST = 5e10                     # sum over the models of (points)³: about a minute of work in the worst case
_HEX = re.compile(r"[0-9a-f]{64}")
_ALLOWED = {"meta": 4 * MAX_META_CHARS + 512, "P": MAX_ROWS * MAX_PARAMS * 8 + 512, "K": MAX_ROWS * len(SURROGATE_KPIS) * 8 + 512}


def _bad(what: str) -> SurrogateError:
    return SurrogateError(f"The surrogate file is damaged ({what})")


# ---------------------------------------------------------------------- writing
def _json_default(o: Any):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    return str(o)


def write_npz(sm: SurrogateModel, path) -> None:
    n = len(sm.rows)
    P = np.zeros((n, len(sm.params)))
    for i, r in enumerate(sm.rows):
        for j, p in enumerate(sm.params):
            v = r["params"][p.key]
            P[i, j] = float(v) if p.kind == "continuous" else p.categories.index(str(v))
    K = np.array([[r["kpi"].get(k) if r["kpi"].get(k) is not None else np.nan for k in SURROGATE_KPIS] for r in sm.rows], dtype=float)
    members = [{"group": list(grp), "kpi": k, "train": [int(i) for i in m.train_ids], "cal": [int(i) for i in m.cal_ids], "hyper": m.backend.params()}
               for grp, mems in sm.members.items() for k, m in mems.items()]
    meta = {"format": FORMAT, "params": [p.to_dict() for p in sm.params], "kpis": list(sm.kpis), "level": sm.level, "seed": sm.seed,
            "backend": sm.backend, "validation_fraction": sm.validation_fraction, "min_holdout": sm.min_holdout, "members": members,
            "warnings": list(sm.warnings), "skipped": [{"group": list(g), "reason": r} for g, r in sm.skipped_groups.items()],
            "fingerprint": sm.fingerprint, "fp_keys": list(sm.fingerprint_keys), "sweep": sm.sweep, "failed": sm.failed[:MAX_FAILED],
            "adaptive": [i for i, r in enumerate(sm.rows) if r.get("adaptive")]}
    text = json.dumps(meta, default=_json_default, allow_nan=False)
    with open(path, "wb") as f:
        np.savez(f, meta=np.array(text), P=P, K=K)


# ---------------------------------------------------------------------- reading: bounded
def _read_npy(f, name: str, declared: int) -> np.ndarray:
    """One .npy member read from the zip stream: the header is parsed and checked, then exactly the announced bytes are read."""
    fmt = np.lib.format
    version = fmt.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = fmt.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = fmt.read_array_header_2_0(f)
    else:
        raise _bad(f"{name}: unknown array version")
    if dtype.hasobject or fortran:
        raise _bad(f"{name}: unsupported array")
    if name == "meta":
        if dtype.kind != "U" or shape != () or dtype.itemsize > 4 * MAX_META_CHARS:
            raise _bad("meta: not a bounded string")
    else:
        if dtype != np.dtype("<f8") or len(shape) != 2:
            raise _bad(f"{name}: not a two-dimensional float64 array")
        rows, cols = shape
        if not (0 < rows <= MAX_ROWS) or cols != (len(SURROGATE_KPIS) if name == "K" else cols) or not (0 < cols <= MAX_PARAMS):
            raise _bad(f"{name}: size {shape}")
    count = int(np.prod(shape, dtype=object)) if shape else 1
    nbytes = count * dtype.itemsize
    if nbytes > _ALLOWED[name] or nbytes > declared:
        raise _bad(f"{name}: size")
    data = f.read(nbytes)
    if len(data) != nbytes:
        raise _bad(f"{name}: truncated")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _members(path) -> dict[str, np.ndarray]:
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise SurrogateError(f"Not a surrogate file: {e}") from None
    if size > MAX_FILE_BYTES:
        raise _bad("file too large")
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as e:
        raise SurrogateError(f"Not a surrogate file: {type(e).__name__}") from None
    with zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        if len(infos) > 8 or len(set(names)) != len(names):
            raise _bad("members")
        out: dict[str, np.ndarray] = {}
        for info in infos:
            base = info.filename[:-4] if info.filename.endswith(".npy") else ""
            if base not in _ALLOWED:
                raise _bad(f"unexpected member {info.filename[:40]!r}")
            if info.file_size > _ALLOWED[base] or info.compress_size > _ALLOWED[base]:
                raise _bad(f"{base}: size")
            with zf.open(info) as f:
                out[base] = _read_npy(f, base, info.file_size)
    if set(out) != set(_ALLOWED):
        raise _bad("missing arrays")
    return out


def _check_depth(text: str) -> None:
    depth, in_str, esc = 0, False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise _bad("nesting too deep")
        elif ch in "]}":
            depth -= 1


def _reject_constant(name: str):
    raise ValueError(name)


# ---------------------------------------------------------------------- reading: schema checks
def _int(v, lo: int, hi: int, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise _bad(what)
    return v


def _num(v, lo: float, hi: float, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise _bad(what)
    return float(v)


def _str(v, maxlen: int, what: str, empty: bool = False) -> str:
    if not isinstance(v, str) or len(v) > maxlen or (not v and not empty):
        raise _bad(what)
    return v


def _list(v, maxlen: int, what: str) -> list:
    if not isinstance(v, list) or len(v) > maxlen:
        raise _bad(what)
    return v


def _dict(v, what: str) -> dict:
    if not isinstance(v, dict):
        raise _bad(what)
    return v


def _plain(v, budget: list[int], depth: int = 0):
    """A JSON value reduced to dict / list / str / number / bool / None with bounded size (the stored sweep specification)."""
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        raise _bad("sweep too large")
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            raise _bad("sweep number")
        return v
    if isinstance(v, str):
        return v[:MAX_STRING]
    if isinstance(v, list):
        return [_plain(x, budget, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k)[:MAX_NAME]: _plain(x, budget, depth + 1) for k, x in v.items()}
    raise _bad("sweep value")


def _ids(v, n: int, what: str) -> list[int]:
    ids = [_int(i, 0, n - 1, what) for i in _list(v, n, what)]
    if len(set(ids)) != len(ids):
        raise _bad(f"{what}: repeated rows")
    return ids


def read_model(path) -> SurrogateModel:
    """The surrogate stored in `path`; SurrogateError for anything that is not a well-formed surrogate file."""
    try:
        return _read_model(path)
    except SurrogateError:
        raise
    except (MemoryError, RecursionError):
        raise SurrogateError("The surrogate file is too large or too deeply nested to be read") from None
    except BackendError as e:
        raise _bad(str(e)) from None
    except Exception as e:  # noqa: BLE001 – a file from a foreign .fea: whatever goes wrong inside is "damaged", never a crash of the program
        raise _bad(type(e).__name__) from None


def _read_model(path) -> SurrogateModel:
    arrays = _members(path)
    text = str(arrays["meta"][()])
    if len(text) > MAX_META_CHARS:
        raise _bad("header too large")
    _check_depth(text)
    try:
        meta = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:
        raise _bad(f"header: {type(e).__name__}") from None
    meta = _dict(meta, "header")
    if meta.get("format") not in (1, FORMAT):
        raise SurrogateError("Unknown surrogate file format")
    P, K = arrays["P"], arrays["K"]
    n, m_cols = P.shape
    if K.shape[0] != n or n > MAX_ROWS:
        raise _bad("table sizes")
    level = _num(meta.get("level"), 0.5, 0.999, "level")
    seed = _int(meta.get("seed"), 0, 2 ** 31 - 1, "seed")
    fraction = _num(meta.get("validation_fraction"), 0.05, 0.5, "validation_fraction")
    backend = _str(meta.get("backend"), 8, "backend")
    if backend not in ("gp", "rbf"):
        raise _bad("backend")
    mh = meta.get("min_holdout")
    min_holdout = None if mh is None else _int(mh, 1, MAX_ROWS, "min_holdout")
    # ---- parameters
    specs: list[ParamSpec] = []
    raw = _list(meta.get("params"), MAX_PARAMS, "params")
    if len(raw) != m_cols:
        raise _bad("parameters do not match the table")
    for j, d in enumerate(raw):
        d = _dict(d, "parameter")
        key = _str(d.get("key"), MAX_NAME, "parameter key")
        kind = d.get("kind")
        if kind not in ("continuous", "categorical") or any(s.key == key for s in specs):
            raise _bad("parameter")
        if kind == "continuous":
            col = P[:, j]
            if not np.all(np.isfinite(col)) or not float(col.max()) > float(col.min()):
                raise _bad(f"values of {key[:40]!r}")
            specs.append(ParamSpec(key, "continuous", float(col.min()), float(col.max())))           # the range is that of the points
        else:
            cats = [_str(c, MAX_NAME, "category", empty=True) for c in _list(d.get("categories"), MAX_CATEGORIES, "categories")]
            if not cats or len(set(cats)) != len(cats):
                raise _bad("categories")
            col = P[:, j]
            if not np.all(np.isfinite(col)) or np.any(col != np.round(col)) or col.min() < 0 or col.max() >= len(cats):
                raise _bad(f"values of {key[:40]!r}")
            specs.append(ParamSpec(key, "categorical", categories=cats))
    cont = [p for p in specs if p.kind == "continuous"]
    cats_ = [p for p in specs if p.kind == "categorical"]
    if not cont or len(cont) > MAX_CONTINUOUS:
        raise _bad("continuous parameters")
    # ---- rows
    rows: list[dict] = []
    for i in range(n):
        pr: dict[str, Any] = {}
        for j, p in enumerate(specs):
            pr[p.key] = float(P[i, j]) if p.kind == "continuous" else p.categories[int(P[i, j])]
        kp: dict[str, float | None] = {}
        for c, k in enumerate(SURROGATE_KPIS):
            v = float(K[i, c])
            if math.isinf(v):
                raise _bad("a result value is not finite")
            kp[k] = None if math.isnan(v) else v
        rows.append({"params": pr, "kpi": kp})
    for i in _ids(meta.get("adaptive", []), n, "adaptive rows") if n else []:
        rows[i]["adaptive"] = True
    # ---- the model
    kpis = [_str(k, MAX_NAME, "kpi") for k in _list(meta.get("kpis"), len(SURROGATE_KPIS), "kpis")]
    if not kpis or len(set(kpis)) != len(kpis) or any(k not in SURROGATE_KPIS for k in kpis):
        raise _bad("kpis")
    sm = SurrogateModel(specs, kpis, rows, level, seed, backend, fraction)
    sm.min_holdout = min_holdout
    sm.warnings = [_str(w, 100 * MAX_STRING, "warning", empty=True)[:MAX_STRING] for w in _list(meta.get("warnings", []), 100, "warnings")]
    fp = meta.get("fingerprint", "")
    if fp != "" and not (isinstance(fp, str) and _HEX.fullmatch(fp)):
        raise _bad("fingerprint")
    sm.fingerprint = fp
    sm.fingerprint_keys = [_str(k, MAX_NAME, "fingerprint key") for k in _list(meta.get("fp_keys", []), 4 * MAX_PARAMS, "fingerprint keys")] if fp else []
    if not fp and NO_FINGERPRINT not in sm.warnings:
        sm.warnings.append(NO_FINGERPRINT)
    sm.sweep = cast(dict, _plain(_dict(meta.get("sweep", {}), "sweep"), [MAX_NODES]))
    names = [p.key for p in specs]
    for f in _list(meta.get("failed", []), MAX_FAILED, "failed"):
        f = _dict(f, "failed point")
        item: dict[str, Any] = {}
        for k, v in f.items():
            if k in names:
                item[k] = _num(v, -1e300, 1e300, "failed point") if isinstance(v, (int, float)) and not isinstance(v, bool) else _str(v, MAX_NAME, "failed point", empty=True)
        sm.failed.append(item)
    for s in _list(meta.get("skipped", []), MAX_MEMBERS, "skipped"):
        s = _dict(s, "skipped")
        grp = tuple(_str(g, MAX_NAME, "skipped group", empty=True) for g in _list(s.get("group"), len(cats_), "skipped group"))
        sm.skipped_groups[grp] = _str(s.get("reason"), MAX_STRING, "skipped reason", empty=True)
    # ---- the members: split + hyperparameters from the file, everything else recomputed
    raw_members = _list(meta.get("members"), MAX_MEMBERS, "members")
    if not raw_members:
        raise _bad("no models")
    cost, seen = 0.0, set()
    d = len(cont)
    adaptive = {i for i, r in enumerate(rows) if r.get("adaptive")}
    plan = []
    for dm in raw_members:
        dm = _dict(dm, "member")
        grp = tuple(_str(g, MAX_NAME, "group", empty=True) for g in _list(dm.get("group"), len(cats_), "group"))
        kpi = _str(dm.get("kpi"), MAX_NAME, "kpi")
        if len(grp) != len(cats_) or any(g not in p.categories for g, p in zip(grp, cats_)) or kpi not in SURROGATE_KPIS or kpi not in kpis \
                or (grp, kpi) in seen:
            raise _bad("model")
        seen.add((grp, kpi))
        tr, ca = _ids(dm.get("train"), n, "train rows"), _ids(dm.get("cal"), n, "calibration rows")
        if not ca or len(tr) < min_train(d) or set(tr) & set(ca) or set(ca) & adaptive:
            raise _bad("the split")
        for i in tr + ca:
            r = rows[i]
            if tuple(str(r["params"][p.key]) for p in cats_) != grp or r["kpi"].get(kpi) is None:
                raise _bad("a row does not belong to the model")
        cost += float(len(tr) + len(ca)) ** 3
        if cost > MAX_COST:
            raise _bad("too much work")
        U = sm._unit(np.array([sm._x(rows[i]["params"]) for i in tr + ca]))
        if spanned_rank(U) < d:
            raise _bad("the points do not span the parameters")
        plan.append((grp, kpi, tr, ca, _dict(dm.get("hyper"), "hyperparameters")))
    for grp, kpi, tr, ca, hyper in plan:
        sm.members.setdefault(grp, {})[kpi] = _fit_member(sm, grp, kpi, tr, ca, hyper=hyper)
    sm.kpis = [k for k in kpis if any(k in mems for mems in sm.members.values())]
    if not sm.kpis:
        raise _bad("no models")
    return sm

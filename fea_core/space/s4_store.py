"""The stored S4 results of a study (`Study.space.extra["s4"]`): compact, JSON-able summaries of the last interface-force, notching, bolt, PCB and rail calculations.

They travel with the project file (the `extra` dict of `SpaceSettings` keeps keys of later stages untouched), are shown in the space panel and are written into the report.
Only summaries are kept (no spectra arrays): peaks, RMS values, the notch justification rows, the margins with their inputs and verdicts. A stored result says when
and with which inputs it was computed (`stamp`); it is not recomputed on loading.

A stored verdict is a statement about the state of the study at the time of the calculation, not a fact: every record carries its `basis` – the digest of the requirement set
(content + whether it was confirmed), the digests of the material allowables and a digest of the study (without the space settings). The report and the panel print a stored
verdict only while the basis matches the present study (`fea_core.space.s4_verify.shown`); otherwise they print "outdated - recompute", and a record without a basis (an older or
hand-edited project file) never prints PASS.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any

KEY = "s4"
SECTIONS = ("interface", "notch", "bolts", "pcb", "rails")
MAX_ITEMS = 200


def _plain(x: Any) -> Any:
    """JSON-able copy (numpy scalars and arrays become python numbers / lists; infinities and NaN become strings)."""
    def conv(o: Any) -> Any:
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)
    d = json.loads(json.dumps(x, default=conv))

    def fix(v: Any) -> Any:
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return str(v)
        if isinstance(v, dict):
            return {k: fix(w) for k, w in v.items()}
        if isinstance(v, list):
            return [fix(w) for w in v]
        return v
    return fix(d)


def study_digest(study: Any) -> str:
    """SHA-256 of the study without its space settings and the list of present blocks: what a stored S4 result depends on (model, loads, constraints, materials, dynamics)."""
    d = study.to_dict()
    for k in ("space", "features"):
        d.pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str, ensure_ascii=True).encode("ascii")).hexdigest()


def basis_of(study: Any) -> dict:
    """The state of the study that a verdict depends on: the digest of the requirement set and whether it is confirmed NOW (with the structure classes it was confirmed for),
    the digests of the material allowables (with their verification state) and the digest of the study."""
    from .requirements import allowable_of
    sp = getattr(study, "space", None)
    rs = getattr(sp, "requirement_set", None)
    allow: dict[str, str] = {}
    for name, m in getattr(study, "materials", {}).items():
        rec = allowable_of(m)
        if rec is not None:
            allow[name] = rec.content_digest() + (":verified" if rec.is_verified() else ":unverified")
    return {"req": rs.content_digest() if rs is not None else "", "confirmed": bool(sp is not None and sp.requirement_confirmed()), "allow": allow, "study": study_digest(study)}


def record(study: Any, section: str, payload: Any, key: str | None = None) -> None:
    """Stores `payload` as the result of `section`; with `key` the section holds several named results (a bolt, a component, a notch case)."""
    if section not in SECTIONS:
        raise ValueError(f"Unknown S4 section '{section}'")
    from .model import SpaceSettings
    if study.space is None:
        study.space = SpaceSettings()
    store = study.space.extra.setdefault(KEY, {})
    body = _plain(payload)
    if isinstance(body, dict):
        body.setdefault("stamp", _dt.datetime.now().isoformat(timespec="seconds"))
        body["basis"] = basis_of(study)                                  # always the basis of NOW: a payload cannot bring its own
    if key is None:
        store[section] = body
    else:
        bucket = store.setdefault(section, {})
        if key not in bucket and len(bucket) >= MAX_ITEMS:
            bucket.pop(next(iter(bucket)))
        bucket[key] = body


def get(study: Any) -> dict:
    sp = getattr(study, "space", None)
    d = sp.extra.get(KEY) if sp is not None else None
    return d if isinstance(d, dict) else {}


def clear(study: Any, section: str | None = None) -> None:
    sp = getattr(study, "space", None)
    if sp is None:
        return
    if section is None:
        sp.extra.pop(KEY, None)
    elif isinstance(sp.extra.get(KEY), dict):
        sp.extra[KEY].pop(section, None)

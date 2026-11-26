"""Proof of the setup that is about to be computed (F15-S1): a digest of what decides the answer, a numbered table of the
supports and loads, and the record of who confirmed it.

Used by the MCP tool `fea_setup_review` and the confirmation before an AI calculation (`tools_setup`), by the GUI dialog
(`fea_gui.setup_confirm`) and by the report ("Setup confirmation"). `setup_digest` is the one hashing scheme of the program:
the ECSS structural analysis report (F14-S6) reuses it for its configuration block.

What the digest covers (see `setup_inputs`): the geometry file, the materials as they are assigned to the parts, the selections
(their selectors and the faces / edges they resolve to), constraints, loads, thermal cases, contacts, bolts, welds, mesh settings
and every other input of the solution that the study file holds. What it does NOT cover: names that are only labels (the
study, the selections, the materials, the mechanical load cases, the coordinate systems), the solver choice and the number of
Gmsh threads, and what is computed after the solution (fatigue settings, combinations, the assessment, the link with the CAD).
A renamed selection or load case therefore keeps the digest, a changed value, face, material or geometry changes it.
A name that another input refers to (a selection, a material, a coordinate system, a load case: `buckling.case`, a bar's material,
the master of a conductance, the target of a prescribed motion, ...) is replaced by the content of that item wherever it stands, so
swapping the names of two load cases changes the digest. A name that stays in the digest as it is: a thermal case, a contact pair,
a part.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from .session import Session

DIGEST_SCHEME = 2                      # bump when the canonical form changes (an old digest is then simply stale)
ENV_CONFIRM = "FEA_MCP_CONFIRM"
MIN_PREFIX = 12                        # the digest may be passed back shortened to this many hex characters

# keys of Study.to_dict() that are not inputs of the solution: labels, post-processing, the program version
_NOT_INPUT = ("format", "version", "features", "name", "geometry", "hidden_parts", "fatigue", "assessment", "combinations",
              "cad_link", "solver", "selections", "materials", "default_material", "part_materials", "coordinate_systems",
              "load_cases")
# keys whose strings are labels, part names or choices from a fixed list (bolt category "A", steel grade "S235", ...), never a
# reference to a selection / material / coordinate system / load case; a name that equals such a value in a key not listed only makes
# the digest change more often than needed (a safe error)
_PLAIN_KEYS = frozenset(("name", "kind", "type", "part", "parts", "ignored_parts", "category", "steel_grade", "shear_split",
                         "criterion", "offset", "element", "solver", "backend", "format", "geometry"))


class ConfirmUi(Protocol):
    """The window that shows the review to the user (the GUI sets it on the session; None = headless)."""

    def open_review(self, payload: dict) -> None: ...


# ===================================================================== canonical form and digest

def _number(x: float) -> float | str:
    """12 significant digits; -0.0 = 0.0; integers and floats are the same number (100 = 100.0)."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(f"{x:.12g}") + 0.0


def canonical(x: Any) -> Any:
    """JSON-ready canonical form: sorted keys, numbers rounded to 12 digits, numpy and tuples as lists."""
    if isinstance(x, dict):
        return {str(k): canonical(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [canonical(v) for v in x]
    if isinstance(x, np.ndarray):
        return canonical(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return _number(float(x))
    if x is None or isinstance(x, str):
        return x
    return str(x)


def canonical_json(x: Any) -> str:
    return json.dumps(canonical(x), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


_GEO_CACHE: dict[str, tuple[int, int, str]] = {}      # path -> (mtime_ns, size, sha256)


def geometry_file_hash(path: str | Path | None) -> str | None:
    """SHA-256 of the geometry file (cached by modification time and size); None = no file."""
    if not path:
        return None
    p = Path(path)
    try:
        stt = p.stat()
    except OSError:
        return None
    key = str(p)
    hit = _GEO_CACHE.get(key)
    if hit and hit[0] == stt.st_mtime_ns and hit[1] == stt.st_size:
        return hit[2]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _GEO_CACHE[key] = (stt.st_mtime_ns, stt.st_size, h.hexdigest())
    return h.hexdigest()


def _selection_token(sel, ids: list[str] | None) -> dict:
    """What a selection stands for: the selectors (without the name) and the faces / edges they resolved to."""
    d = sel.to_dict()
    d.pop("name", None)
    tok = {"selectors": d}
    if ids is not None:
        tok["resolved"] = sorted(ids)
    return tok


def _csys_token(cs) -> dict:
    d = asdict(cs)
    d.pop("name", None)
    return d


def _material_token(m, csys_tokens: dict) -> dict:
    d = asdict(m)
    d.pop("name", None)
    d.pop("source", None)
    cs = d.get("ortho_csys")
    if cs in csys_tokens:
        d["ortho_csys"] = csys_tokens[cs]
    return d


def setup_inputs(study, geometry_hash: str | None = None, resolved: dict[str, list[str]] | None = None) -> dict:
    """The structure that `setup_digest` hashes. `geometry_hash` = content hash of the geometry file (None = the file name is used
    instead); `resolved` = selection name → the faces / edges it resolves to in the geometry (`resolved_ids`)."""
    d = study.to_dict()
    for k in _NOT_INPUT:
        d.pop(k, None)
    if isinstance(d.get("mesh"), dict):
        d["mesh"] = {k: v for k, v in d["mesh"].items() if k != "threads"}
    toks = {n: _selection_token(s, (resolved or {}).get(n) if resolved is not None else None) for n, s in study.selections.items()}
    cs_toks = {n: _csys_token(c) for n, c in study.coordinate_systems.items()}
    mat_toks = {n: _material_token(m, cs_toks) for n, m in study.materials.items()}
    lc_toks: dict[str, Any] = {}                      # filled after the loads of the cases are walked (a case token = its content)

    def refs(x: str) -> dict[str, Any]:
        """What the string names, in every namespace of the study (a name may be a selection AND a load case)."""
        hit: dict[str, Any] = {}
        for ns, table in (("selection", toks), ("csys", cs_toks), ("material", mat_toks), ("load_case", lc_toks)):
            if x in table:
                hit[ns] = table[x]
        return hit

    def walk(x, key=None) -> Any:
        """Every string that names a selection, coordinate system, material or load case (whatever field holds it) stands for the
        content of that item: a rename keeps the digest, a change of the item (or a swap of two names) changes it."""
        if isinstance(x, dict):
            return {k: walk(v, k) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v, key) for v in x]
        if isinstance(x, str) and key not in _PLAIN_KEYS:
            hit = refs(x)
            if hit:
                return {"ref": hit}
        return x
    case_loads = [walk([asdict(l) for l in lc.loads]) for lc in study.load_cases]
    lc_toks.update({lc.name: {"loads": loads} for lc, loads in zip(study.load_cases, case_loads)})
    out: dict = walk(d)
    out["load_cases"] = [{"loads": loads} for loads in case_loads]
    out["materials"] = {
        "default": _material_token(study.materials[study.default_material], cs_toks) if study.default_material in study.materials else None,
        "parts": {p: _material_token(study.materials[m], cs_toks) for p, m in sorted(study.part_materials.items()) if m in study.materials},
        "missing": sorted(p for p, m in study.part_materials.items() if m not in study.materials)}
    out["geometry"] = geometry_hash if geometry_hash else {"file": Path(str(study.geometry or "")).name}
    out["scheme"] = DIGEST_SCHEME
    return out


def setup_digest(study, geometry_hash: str | None = None, resolved: dict[str, list[str]] | None = None) -> str:
    """SHA-256 (hex) over the canonical JSON of the inputs of the solution (see the module docstring)."""
    return hashlib.sha256(canonical_json(setup_inputs(study, geometry_hash, resolved)).encode("utf-8")).hexdigest()


def resolved_ids(sess: Session) -> dict[str, list[str]] | None:
    """Selection name → ids of the faces ('Part#i') and edges ('Part~i') of the geometry that its selectors resolve to ('?Part' =
    not found). None when the geometry is not available (a study without a body, a missing file)."""
    from .session import FeaError
    if sess.study is None or sess.frame_only:
        return None
    st = sess.require_study()
    try:
        parts = sess.parts
    except FeaError:
        return None
    from ..selectors import resolve, resolve_edge
    sess.sync_geometry_caches()
    key = canonical_json({n: sel.to_dict() for n, sel in st.selections.items()})
    hit = sess._resolved_cache
    if hit is not None and hit[0] is parts and hit[1] == key:       # the same shapes and selectors: resolved already (the scoring of
        return hit[2]                                                # every face per selector is slow on large assemblies)
    for pi in range(len(parts)):                       # the caches are filled in full (axes of the cylinders) before resolve uses them
        sess.face_props(pi)
        sess.edge_props(pi)
    out: dict[str, list[str]] = {}
    for name, sel in st.selections.items():
        ids = []
        for f in sel.faces:
            r = resolve(f, parts, sess._face_props)
            ids.append(f"{parts[r[0]].name}#{r[1]}" if r else f"?{f.part}")
        for e in sel.edges:
            r = resolve_edge(e, parts, sess._edge_props)
            ids.append(f"{parts[r[0]].name}~{r[1]}" if r else f"?{e.part}")
        ids += [f"{p.part}@" + ",".join(f"{v:.6g}" for v in p.point) for p in sel.points]
        out[name] = sorted(ids)
    sess._resolved_cache = (parts, key, out)
    return out


def session_digest(sess: Session) -> str:
    """The digest of the study of the session with the hash of its geometry file and the resolved faces."""
    st = sess.require_study()
    return setup_digest(st, geometry_file_hash(sess.step_path), resolved_ids(sess))


def digest_matches(digest: str, given: str | None) -> bool:
    """`given` is the digest or its beginning (at least MIN_PREFIX hex characters)."""
    g = (given or "").strip().lower()
    return len(g) >= MIN_PREFIX and digest.startswith(g)


# ===================================================================== the numbered table

@dataclass(frozen=True)
class Item:
    """One numbered item of the setup: a support, a load of a mechanical case or a boundary condition of a thermal case."""
    no: int
    group: str                      # constraint | load | thermal
    obj: Any
    case: str | None                # load case / thermal case; None for a support (it acts in every case)


def numbered_items(study) -> list[Item]:
    """Supports, then the loads of the load cases, then the boundary conditions of the thermal cases – the numbering of the table
    and of the badges in the picture (`scenes.model_scene(numbered=True)`)."""
    items: list[Item] = []
    for c in study.constraints:
        items.append(Item(len(items) + 1, "constraint", c, None))
    for lc in study.load_cases:
        for l in lc.loads:
            items.append(Item(len(items) + 1, "load", l, lc.name))
    for tc in study.thermal.cases:
        for bc in tc.loads:
            items.append(Item(len(items) + 1, "thermal", bc, tc.name))
    return items


KIND_LABELS = {
    "fixed": "Fixed support", "displacement": "Prescribed displacement", "cylindrical": "Cylindrical support",
    "remote_support": "Remote support", "elastic": "Elastic support",
    "force": "Force", "pressure": "Pressure", "gravity": "Self-weight", "rotation": "Rotation", "remote": "Remote load",
    "bearing": "Bearing load", "thermal": "Temperature load", "inertia": "Inertia forces",
    "frame_node": "Bar node load", "frame_member": "Bar load", "frame_temperature": "Bar temperature",
}


def _kind_label(item: Item, study) -> str:
    o = item.obj
    if item.group == "thermal":
        from .diagnostics import thermal_label
        return str(thermal_label(o))
    kind = getattr(o, "kind", type(o).__name__)
    if kind == "pressure":
        return "Hydrostatic pressure" if getattr(o, "clip", False) else ("Variable pressure" if getattr(o, "variable", False) else "Pressure")
    if kind == "force":
        from ..loads_text import entity_of
        return {"edges": "Force on edge", "points": "Force at point"}.get(entity_of(o, study), "Force")
    return KIND_LABELS.get(kind, str(kind))


def _value_text(item: Item, study) -> str:
    o = item.obj
    if item.group == "constraint":
        from ..loads_text import constraint_text
        return constraint_text(o, study)
    if item.group == "thermal":
        from .diagnostics import bc_value_text
        return str(bc_value_text(o))
    kind = getattr(o, "kind", "")
    if kind == "thermal":
        from .diagnostics import thermal_load_text
        return str(thermal_load_text(o))
    if kind in ("frame_node", "frame_member", "frame_temperature"):
        from ..report.frame import load_text as frame_load_text
        return str(frame_load_text(o)[2])
    from ..loads_text import load_text
    return load_text(o, study)


def _selection_names(o) -> list[str]:
    sel = getattr(o, "selection", None)
    if isinstance(sel, (list, tuple)):
        return [str(s) for s in sel]
    return [str(sel)] if sel else []


LEGEND_CHARS = 78                              # longest legend row of the picture


def legend_text(item: Item, study) -> str:
    """The row of the legend of the numbered picture: number, selection (or kind), value with units."""
    names = _selection_names(item.obj)
    kind = _kind_label(item, study)
    head = f"{item.no}. " + (f"{kind} {', '.join(names)}" if item.group == "thermal" and names else ", ".join(names) or kind)
    t = f"{head}: {_value_text(item, study)}"
    return t if len(t) <= LEGEND_CHARS else t[:LEGEND_CHARS - 1] + "…"


MAX_IDS = 12                                   # ids listed per row (the rest is only counted)


def _entities(study, names: list[str], sess) -> dict:
    """Faces / edges / points of the selections of a row: count, size (area / length) and the ids."""
    faces: list[str] = []
    n = 0
    size = 0.0
    entity = None
    known = False
    for name in names:
        sel = study.selections.get(name)
        if sel is None:
            continue
        entity = sel.entity if entity in (None, sel.entity) else "mixed"
        n += len(sel.faces) + len(sel.edges) + len(sel.points)
        res_ids = _resolve_one(sel, sess)
        if res_ids is not None:
            known = True
            faces += res_ids[0]
            size += res_ids[1]
        else:
            size += sum(f.area for f in sel.faces) + sum(e.length for e in sel.edges)
    d: dict = {"entity": entity, "n": n, "size": round(size, 3) if (known or size) else None,
               "size_unit": "mm²" if entity == "faces" else "mm" if entity == "edges" else None}
    if faces:
        d["ids"] = faces[:MAX_IDS] + ([f"+{len(faces) - MAX_IDS} more"] if len(faces) > MAX_IDS else [])
    return d


def _resolve_one(sel, sess) -> tuple[list[str], float] | None:
    """(ids, area or length) of a selection resolved in the geometry of the session; None without geometry."""
    if sess is None or sess.study is None or sess.frame_only:
        return None
    try:
        parts = sess.parts
    except Exception:  # noqa: BLE001 – a table without geometry is still a table
        return None
    from ..selectors import resolve, resolve_edge
    sess.sync_geometry_caches()
    for pi in range(len(parts)):
        sess.face_props(pi)
    ids: list[str] = []
    size = 0.0
    for f in sel.faces:
        r = resolve(f, parts, sess._face_props)
        if r is None:
            ids.append(f"?{f.part}")
            size += f.area
            continue
        ids.append(f"{parts[r[0]].name}#{r[1]}")
        size += float(sess.face_props(r[0])[r[1]]["area"])
    for e in sel.edges:
        r = resolve_edge(e, parts, sess._edge_props)
        if r is None:
            ids.append(f"?{e.part}")
            size += e.length
            continue
        ids.append(f"{parts[r[0]].name}~{r[1]}")
        size += float(sess.edge_props(r[0])[r[1]]["length"])
    return ids, size


def setup_table(study, sess: Session | None = None, colors: dict[int, tuple] | None = None) -> list[dict]:
    """Rows of the setup table: no. | kind | selection | faces (count, area) | value with units | cases.

    `sess` = the session with the geometry (the faces are resolved: ids, count, area; without it the counts come from the
    selectors); `colors` = item number → RGB 0..1 of the picture (`Scene.numbers`). The texts are those of the report
    (`loads_text`)."""
    rows = []
    for it in numbered_items(study):
        names = _selection_names(it.obj)
        row: dict = {"no": it.no, "group": it.group, "kind": _kind_label(it, study), "selection": ", ".join(names) or None,
                     "value": _value_text(it, study), "cases": [it.case] if it.case else ["all"]}
        if names:
            row.update(_entities(study, names, sess))
        else:
            parts = list(getattr(it.obj, "parts", None) or [])
            row.update({"entity": "parts" if parts else "model", "n": len(parts), "size": None, "size_unit": None})
            if parts:
                row["ids"] = parts
        if colors and it.no in colors:
            row["color"] = [round(float(v), 3) for v in colors[it.no]]
        rows.append(row)
    return rows


def faces_cell(r: dict) -> str:
    """The cell "Faces" of a row: what the selection holds (count, area / length) or where a load without selection acts."""
    if not r.get("selection"):
        return ", ".join(r.get("ids", [])) if r.get("ids") else "entire model"
    n, ent = r.get("n", 0), r.get("entity")
    cell = {"faces": f"Faces: {n}", "edges": f"Edges: {n}", "points": f"Points: {n}"}.get(ent or "", f"Items: {n}")
    if r.get("size"):
        cell += f", {r['size']:.4g} mm²" if r.get("size_unit") == "mm²" else f", {r['size']:.4g} mm"
    return cell


def table_text(rows: list[dict]) -> str:
    """The table as markdown (what the agent shows to the user and what the elicitation form carries)."""
    if not rows:
        return "(no supports or loads)"
    lines = ["| No. | Kind | Selection | Faces | Value | Cases |", "|---|---|---|---|---|---|"]
    for r in rows:
        cells = [str(r["no"]), r["kind"], r.get("selection") or "–", faces_cell(r), r["value"], ", ".join(r["cases"])]
        lines.append("| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in cells) + " |")
    return "\n".join(lines)


def summary_lines(study, sess: Session | None = None) -> list[str]:
    """What else is computed besides the numbered items: parts and materials, kinds of analysis, mesh, joints, contacts."""
    out: list[str] = []
    names: list[str] = []
    if sess is not None:
        try:
            names = [p.name for p in sess.parts] if not sess.frame_only else []
        except Exception:  # noqa: BLE001
            names = []
    names = names or sorted(study.part_materials)
    if names:
        mats = []
        for p in names:
            try:
                mats.append(f"{p} [{study.material_of(p).name}]")
            except KeyError:
                mats.append(f"{p} [NO MATERIAL]")
        out.append("Parts: " + ", ".join(mats[:12]) + (f" … ({len(names)} parts)" if len(names) > 12 else ""))
    kinds = []
    if study.load_cases:
        kinds.append("nonlinear" if study.is_nonlinear() else "linear static")
    if study.thermal.cases:
        kinds.append("heat conduction")
    if study.modal.enabled:
        kinds.append("natural frequencies")
    if study.buckling.enabled:
        kinds.append("linear buckling")
    if study.dynamics.enabled:
        kinds.append("dynamics")
    if study.mechanism.active:
        kinds.append("mechanism")
    if kinds:
        out.append("Analysis: " + ", ".join(kinds) + f"; load cases: {len(study.load_cases)}")
    m = study.mesh
    out.append(f"Mesh: element size {m.size:g} mm, " + ("tet10" if m.order == 2 else "tet4")
               + (f", {len(m.local)} local refinements" if m.local else ""))
    extra = []
    if study.active_bolts():
        extra.append(f"{len(study.active_bolts())} bolts")
    if study.active_welds():
        extra.append(f"{len(study.active_welds())} welds")
    if study.active_contacts():
        extra.append(f"{len(study.active_contacts())} contact pairs")
    if study.masses:
        extra.append(f"{len(study.masses)} added masses")
    if study.frame.active:
        extra.append(f"{len(study.frame.members)} bars ({len(study.frame.supports)} supports)")
    if study.ignored_parts:
        extra.append("ignored parts: " + ", ".join(study.ignored_parts[:6]))
    if extra:
        out.append("Also in the model: " + ", ".join(extra))
    return out


# ===================================================================== who confirmed it

def confirmation_required(sess: Session) -> bool:
    """The session setting (the GUI sets it from "Ask before an AI calculation"), otherwise the variable FEA_MCP_CONFIRM."""
    if sess.require_confirmation is not None:
        return bool(sess.require_confirmation)
    return os.environ.get(ENV_CONFIRM, "").strip().lower() in ("1", "true", "yes", "on")


_LOCK = threading.Lock()


def record_decision(sess: Session, digest: str, decision: str, by: str, channel: str) -> dict:
    """Stores the decision on the setup with this digest. `by` = "user" (clicked in the window / answered the form of the MCP
    client) or "agent" (the agent only passed the digest back – not a proof that the user saw anything); `channel` = gui |
    elicitation | agent. The only callers are the GUI dialog, the elicitation and the gate of the tools – never a tool argument."""
    rec = {"digest": digest, "decision": decision, "by": by, "channel": channel,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with _LOCK:
        sess.confirmation = rec
    return rec


def result_info(rec: dict) -> dict:
    """The entry `Results.info["setup_confirmation"]` (and of the report)."""
    return {"digest": rec["digest"], "confirmed_by": rec["by"], "channel": rec["channel"], "at": rec["at"]}


def current_record(sess: Session, digest: str) -> dict | None:
    """The decision on exactly this digest (a decision on an older setup is not returned)."""
    rec = sess.confirmation
    return rec if rec and rec.get("digest") == digest else None

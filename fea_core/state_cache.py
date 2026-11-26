"""Results that several analyses of ONE study on ONE mesh would compute again and again: the modal basis (assembly + eigenvalue analysis + modal stresses) and the
unit-gravity states (the 3 unit states that every gravity case of the study is a linear combination of).

Where the files live: in the WORK FOLDER of the run that computed them (`<work folder>/state_cache/<kind>_<digest>_<tag>/`, plain .npy + meta.json, written atomically), so
they are deleted with it; the main program keeps only a small registry (digest → folders) and maps the arrays read-only (clean pages the system can drop – the RAM stays for the
solver). A later analysis of the same study and mesh in this process – the 3-axis launch analysis after the run, the interface forces, the model checks – finds the folder by the digest
and loads the arrays memory-mapped instead of computing them again.

The key (`study_digest`) is the study WITHOUT what does not change the stiffness, the mass or the loads of a linear state (the name, the space settings, the hidden parts, the
solver thread count, the eigenvalue / dynamics / buckling settings – a result is stored under its own mode count and window – and the load cases, which are hashed one by one) PLUS
the rigidly coupled remote loads of the load cases (they add a rigid body to the model), the mesh content (`mesh_digest`: nodes, elements, parts, every named surface / edge /
point selection, the tie table, the per-element and per-node data), the solver that makes the number (the CalculiX executable with its size and time, the faer switch, the
assembler, the direct engine, its precision and the work limit of the modal analysis) and the program version with the layout of the cache (`SCHEMA`). Anything else that could
change a result is part of the study dictionary and so of the key; a result that cannot be keyed (a prestressed basis, a thermal state, a nonlinear study) is never stored.
`FEA_STATE_CACHE=0` switches the whole cache off.

What is stored is checked: every array carries a content hash in its entry's `meta.json` (`files`), an entry is used only when its schema and program version are the current ones,
every listed file is there with its shape and type, and the content hash agrees (checked once per entry and process, when it is first looked up) – a damaged or half-written
entry is dropped and the result computed again. An entry that already exists when the same key is written again is validated: it is kept when its content is the one being
written, replaced when it is damaged, partial or of another content. Every reuse is logged by the callers with the name of the entry.

Disk: the entries of a work folder stay in it until it is deleted; each analysis of a new study / mesh adds its own (a modal basis of 100 modes at 400 000 DOF is ~1 GB, the unit
states ~5 arrays of N × 3–6 doubles each). `prune` keeps the entries of the key just written and the `KEEP_OTHER` newest of any other key per work folder.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ENV = "FEA_STATE_CACHE"
SCHEMA = 2                                   # change it when the layout of an entry or what a key covers changes
MIN_FREE_GB = 10.0                           # the volume of the work folder keeps at least this much free (the rule of the project)
MAX_PER_KEY = 6                              # folders remembered per (kind, digest)
MAX_KEYS = 48                                # (kind, digest) pairs remembered (the oldest are forgotten – their files stay in their work folders)
KEEP_OTHER = 6                               # entries of other keys kept per work folder when a new one is written (`prune`)
FULL_HASH_BYTES = 1 << 30                    # arrays up to 1 GiB are hashed completely, larger ones by the head, the tail and a stride of blocks (`content_hash`)
SAMPLE_EDGE = 1 << 27                        # ... the first and the last 128 MiB
SAMPLE_BLOCKS = 256                          # ... and 256 blocks of 1 MiB between them (about 512 MiB hashed: the check of a stored array takes a fraction of a second)
RECENT_KEYS = 4                              # the digests written or found lately: `prune` leaves their entries alone (the chain of analyses that is running)
_LOCK = threading.RLock()
_REG: dict[str, list[str]] = {}              # "<kind>:<digest>" -> [folder, …] (the newest last)
_CHECKED: dict[str, tuple] = {}              # folder -> the stamp (size and time of meta.json and of every array file) when its content was last verified
_RECENT: list[str] = []                      # the digests (newest last) written or found lately in this process

_STUDY_DROP = ("space", "hidden_parts", "name", "load_cases", "modal", "dynamics", "buckling", "nlbuckling", "features")      # "features" = the list of the optional sections that are present
_ENV_KEYS = ("FEA_CCX", "FEA_FAER", "FEA_DIRECT_ENGINE", "FEA_DIRECT_PRECISION", "FEA_MODAL_MIN_WORK", "FEA_ASSEMBLY")           # the variables that change a number of a linear state


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def clear() -> None:
    """Forgets every entry (tests; the files stay where they are)."""
    with _LOCK:
        _REG.clear()
        _CHECKED.clear()
        _RECENT.clear()


def mesh_digest(mesh: Any) -> str:
    """Digest of everything of a mesh that a linear result depends on: nodes, elements, parts, the named surfaces, the edge and point selections (edge / point supports and loads),
    the node aliases of a disconnected contact, the tie table of separately meshed parts and the per-element / per-node data (3D printing, tolerances). Two meshes with the same
    digest give the same linear result."""
    h = hashlib.sha256()

    def feed(label: str, arr: Any) -> None:
        a = np.ascontiguousarray(arr)
        h.update(label.encode("utf-8"))
        h.update(str(a.shape).encode("ascii") + str(a.dtype).encode("ascii"))
        h.update(a.tobytes())

    for name in ("nodes", "tets", "tet_part"):
        feed(name, getattr(mesh, name))
    h.update(json.dumps([str(n) for n in mesh.part_names]).encode("utf-8"))
    for group in ("surfaces", "edges", "points", "cell_data", "node_data"):
        d = getattr(mesh, group, None) or {}
        for key in sorted(d, key=str):
            feed(f"{group}:{key}", d[key])
    alias = getattr(mesh, "node_alias", None)
    feed("node_alias", np.zeros(0) if alias is None else alias)
    h.update(json.dumps((getattr(mesh, "report", None) or {}).get("ties", []), sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def file_identity(path: Any) -> list[Any] | None:
    """[path, size, time] of a file (the identity of an executable); None for no path."""
    if not path:
        return None
    p = Path(path)
    try:
        s = p.stat()
        return [str(p), int(s.st_size), int(s.st_mtime_ns)]
    except OSError:
        return [str(p), None, None]


def solver_identity() -> dict[str, Any]:
    """What makes the numbers of a state apart from the study and the mesh: the CalculiX executable (path, size, time), the faer switch, the assembler, the direct engine and its
    precision, the work limit of the modal analysis (which solver the analysis takes) and the variables behind them – a change of any of them is another key."""
    from . import assembly, config
    from .direct import engine_name, modal_run
    from . import __version__
    return {"program": __version__, "ccx": file_identity(config.CCX_EXE), "faer": bool(config.faer_enabled()), "assembler": assembly.assembler(), "direct_engine": engine_name(),
            "min_work": modal_run.MIN_WORK, "env": {k: os.environ.get(k) for k in _ENV_KEYS}}


def study_digest(study: Any, mesh: Any) -> str:
    """The key of the linear states of `study` on `mesh` (see the module text)."""
    from .direct import modal_run
    d = study.to_dict()
    for k in _STUDY_DROP:
        d.pop(k, None)
    if isinstance(d.get("solver"), dict):
        d["solver"] = {k: v for k, v in d["solver"].items() if k != "nproc"}
    if not study.active_bolts() and isinstance(d.get("joints"), dict):
        d["joints"] = {k: v for k, v in d["joints"].items() if k != "preload_case"}          # (without an active bolt the preload switch changes nothing of the model or the deck)
    # a rigidly coupled remote load of a load case adds a rigid body to the MODEL (the stiffness of every state): those cases are part of the key, like in `run_digest_payload`
    rigid = sorted(case_signature(lc) for lc in study.load_cases if any(getattr(ld, "coupling", None) == "rigid" for ld in lc.loads))
    payload = {"schema": SCHEMA, "study": d, "mesh": mesh_digest(mesh), "solver_identity": solver_identity(), "modal_engine": modal_run.engine_setting(study), "rigid_loads": rigid,
               "study_exe": file_identity(getattr(study.solver, "exe", None))}          # (an executable named by the study itself, where the program takes it)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def solved_digest(study: Any, mesh: Any, cases: list[Any]) -> str:
    """The key of the states of the model that is ACTUALLY solved: `study` with exactly the load cases `cases`. The rigid remote loads of the cases that are in the solved deck
    are part of the model, those of the cases that are not are not – a state solved from the unit cases only is not the state of a deck that also holds a rigid load."""
    import copy
    st = copy.copy(study)
    st.load_cases = list(cases)
    return study_digest(st, mesh)


def case_signature(lc: Any) -> str:
    """Identity of a load case by its CONTENT (the name does not matter): the digest of its loads."""
    import dataclasses
    loads = [dataclasses.asdict(x) if (dataclasses.is_dataclass(x) and not isinstance(x, type)) else repr(x) for x in lc.loads]
    return hashlib.sha256(json.dumps(loads, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]


# ----------------------------------------------------------------------------------------------------------------------------------------- content hashes
def content_hash(a: Any) -> str:
    """Hash of an array (shape, type, content) – fast (blake2b). An array above `FULL_HASH_BYTES` (1 GiB) is only SAMPLED: its head and tail (`SAMPLE_EDGE`) and `SAMPLE_BLOCKS` blocks
    of 1 MiB at an even stride – about 512 MiB, so a change of a few elements in the middle of such an array may go unseen (a damaged file is mostly truncated or zeroed in a
    larger part, which this finds; the shape and the type are always part of the hash)."""
    a = np.ascontiguousarray(a)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(a.shape).encode("ascii") + a.dtype.str.encode("ascii"))
    if a.size == 0:
        return h.hexdigest()
    mv = memoryview(a.reshape(-1)).cast("B")
    n, blk = len(mv), 1 << 24
    if n <= FULL_HASH_BYTES:
        for lo in range(0, n, blk):
            h.update(mv[lo:lo + blk])
    else:
        edge, step = SAMPLE_EDGE, 1 << 20
        h.update(mv[:edge])
        h.update(mv[-edge:])
        for lo in range(edge, n - edge, max((n - 2 * edge) // SAMPLE_BLOCKS, step)):
            h.update(mv[lo:lo + step])
    return h.hexdigest()


def _describe(arrays: dict[str, np.ndarray], known: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The shape, type and content hash of every array (`known` = hashes the caller has computed already)."""
    known = known or {}
    return {name: {"shape": list(np.shape(a)), "dtype": np.asarray(a).dtype.str, "hash": known.get(name) or content_hash(a)} for name, a in arrays.items()}


def _program_version() -> str:
    from . import __version__
    return __version__


def _stamp(meta: dict[str, Any], arrays: dict[str, np.ndarray], known: dict[str, str] | None = None) -> dict[str, Any]:
    return {**meta, "schema": SCHEMA, "version": _program_version(), "files": _describe(arrays, known), "run": uuid.uuid4().hex[:16]}


# ----------------------------------------------------------------------------------------------------------------------------------------- registry
def register(kind: str, digest: str, folder: Path) -> None:
    key = f"{kind}:{digest}"
    with _LOCK:
        lst = _REG.setdefault(key, [])
        s = str(folder)
        if s in lst:
            lst.remove(s)
        lst.append(s)
        del lst[:-MAX_PER_KEY]
        while len(_REG) > MAX_KEYS:
            _REG.pop(next(iter(_REG)))
    _touch(digest)


def _touch(digest: str) -> None:
    """`digest` is in use: its entries are not pruned for a while."""
    with _LOCK:
        if digest in _RECENT:
            _RECENT.remove(digest)
        _RECENT.append(digest)
        del _RECENT[:-RECENT_KEYS]


INVALID: list[str] = []                      # the entries that failed their check (diagnostics, tests)


def entries(kind: str, digest: str) -> list[Path]:
    """The folders of (kind, digest) that still exist AND pass their check (`verified`), the newest first. A damaged one is forgotten."""
    with _LOCK:
        lst = list(_REG.get(f"{kind}:{digest}", []))
    out = []
    for s in reversed(lst):
        p = Path(s)
        if not (p / "meta.json").is_file():
            continue
        if verified(p):
            out.append(p)
            _touch(digest)
        else:
            forget(kind, digest, p)
    return out


def any_entries(kind: str) -> bool:
    """Is anything of this kind registered at all (a cheap test before the digest of a large mesh is computed)."""
    with _LOCK:
        return any(k.startswith(kind + ":") and v for k, v in _REG.items())


def forget(kind: str, digest: str, folder: Path) -> None:
    with _LOCK:
        lst = _REG.get(f"{kind}:{digest}", [])
        if str(folder) in lst:
            lst.remove(str(folder))
        _CHECKED.pop(str(folder), None)


def has_room(folder: Path, need_bytes: int) -> bool:
    """True when the volume of `folder` keeps `MIN_FREE_GB` free after `need_bytes` more are written (the cache is an optimisation: with no room it is skipped)."""
    p = Path(folder)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(p).free - 2 * int(need_bytes) > MIN_FREE_GB * 2 ** 30
    except OSError:
        return False


# ----------------------------------------------------------------------------------------------------------------------------------------- files
LAST_ERROR = ""                              # the last write that failed (diagnostics: the cache is an optimisation, a failed write is not an error of the run)


def _rename(src: Path, dst: Path) -> None:
    """Renames the finished temporary folder; Windows refuses it for a moment while a scanner or an indexer looks at the new files – a few tries. An existing target is a
    `FileExistsError` at once on every platform (the caller decides what to do with it)."""
    import time
    for attempt in range(8):
        try:
            src.rename(dst)
            return
        except FileExistsError:
            raise
        except OSError as e:
            if e.errno in (errno.ENOTEMPTY, errno.EEXIST):                 # POSIX: a rename onto a non-empty folder is ENOTEMPTY, not FileExistsError
                raise FileExistsError(str(dst)) from None
            if not isinstance(e, PermissionError):
                raise
            if dst.exists():
                raise FileExistsError(str(dst)) from None
            if attempt == 7:
                raise
            time.sleep(0.1 * (attempt + 1))


def meta_of(folder: Path) -> dict[str, Any] | None:
    """The meta of an entry when it is of the current schema AND program version (an old entry – another layout, another build – is not read)."""
    try:
        m = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or m.get("schema") != SCHEMA or m.get("version") != _program_version():
        return None
    return m


def _files_ok(folder: Path, m: dict[str, Any]) -> bool:
    """Every file listed in the meta is there with its shape and type (cheap: the headers only)."""
    files = m.get("files")
    if not isinstance(files, dict):
        return False
    for name, d in files.items():
        try:
            a = np.load(Path(folder) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return False
        if list(a.shape) != list(d.get("shape", [])) or a.dtype.str != d.get("dtype"):
            return False
        del a
    return True


def _folder_stamp(folder: Path, m: dict[str, Any]) -> tuple | None:
    """Size and time of the meta and of every array file of an entry (None when one is missing)."""
    try:
        return tuple((name, int(s.st_size), int(s.st_mtime_ns)) for name in ["meta"] + sorted(m.get("files") or {})
                     for s in [(Path(folder) / ("meta.json" if name == "meta" else f"{name}.npy")).stat()])
    except OSError:
        return None


def _mark_checked(folder: Path) -> None:
    """The entry was written (or found to be the same) by this very call: its content is known, no need to read it again."""
    m = meta_of(folder)
    stamp = _folder_stamp(folder, m) if m is not None else None
    if stamp is not None:
        with _LOCK:
            _CHECKED[str(folder)] = stamp


def verified(folder: Path, full: bool = True) -> bool:
    """Is the entry intact: current schema and version, every file listed there present with its shape and type and – `full` – the content hash of every array agrees. The full
    check is made once per entry and process, again when the size or the time of any of its files changed since. A sub-entry added later (`add_files`) is its own entry in this
    sense."""
    folder = Path(folder)
    m = meta_of(folder)
    if m is None:
        return False
    stamp = _folder_stamp(folder, m)
    if stamp is None:
        return False
    with _LOCK:
        if _CHECKED.get(str(folder)) == stamp:
            return True
    if not _files_ok(folder, m):
        INVALID.append(str(folder))
        return False
    if full:
        for name, d in (m.get("files") or {}).items():
            try:
                a = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                ok = content_hash(a) == d.get("hash")
                del a
            except (OSError, ValueError):
                ok = False
            if not ok:
                INVALID.append(f"{folder}:{name}")
                return False
    if full:                                                  # (a header check alone proves nothing about the content: a later full check must still read it)
        with _LOCK:
            _CHECKED[str(folder)] = stamp
    return True


def _write_folder(tmp: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any], known: dict[str, str] | None = None) -> dict[str, Any]:
    tmp.mkdir()
    full = _stamp(meta, arrays, known)
    for name, a in arrays.items():
        np.save(tmp / f"{name}.npy", np.ascontiguousarray(a), allow_pickle=False)
    (tmp / "meta.json").write_text(json.dumps(full, default=str), encoding="utf-8")
    return full


def _same_content(folder: Path, new_meta: dict[str, Any]) -> bool:
    """The existing entry is intact AND holds exactly the arrays described by `new_meta` (same names, shapes, types, content hashes)."""
    if not verified(folder):
        return False
    old = (meta_of(folder) or {}).get("files") or {}
    return {k: (v.get("shape"), v.get("dtype"), v.get("hash")) for k, v in old.items()} == {k: (v.get("shape"), v.get("dtype"), v.get("hash")) for k, v in new_meta["files"].items()}


def _drop(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)


def _publish(tmp: Path, final: Path, new_meta: dict[str, Any]) -> Path:
    """Puts the finished temporary folder at `final`. Where an entry exists already (an earlier run in this work folder, a concurrent writer): the same content – it stays, the new
    folder is dropped; a damaged / partial / old one – it is replaced; another content that is still valid (it may be mapped by a reader) – the new one gets a name of its own
    next to it. Returns the folder that now holds the content."""
    try:
        _rename(tmp, final)
        return final
    except FileExistsError:
        pass
    if _same_content(final, new_meta):
        _drop(tmp)
        return final
    if not verified(final):                                  # damaged, partial, old schema / version: replace it (a rename of the old folder aside, then the new one in)
        aside = final.with_name(f".old_{uuid.uuid4().hex[:12]}")
        try:
            final.rename(aside)
        except OSError:
            aside = None
        if aside is not None:
            try:
                _rename(tmp, final)
                _drop(aside)
                return final
            except OSError:
                _drop(aside)
    other = final.with_name(f"{final.name}_{uuid.uuid4().hex[:8]}")
    _rename(tmp, other)
    return other


def write_entry(root: Path, kind: str, digest: str, tag: str, arrays: dict[str, np.ndarray], meta: dict[str, Any], known: dict[str, str] | None = None) -> Path | None:
    """Writes the entry atomically (a temporary folder, then a rename) under `root/state_cache/` and registers it. None when it could not be written (no room, an
    error): the result is then simply not cached. An existing entry of the same name is validated (see the module text); with no room for a new one, an existing entry is
    returned only when it holds exactly these arrays. known = content hashes of arrays the caller computed already (`content_hash`)."""
    base = Path(root) / "state_cache"
    final = base / f"{kind}_{digest[:20]}_{tag}"
    need = sum(int(np.asarray(a).nbytes) for a in arrays.values())
    try:
        if not has_room(base, need):
            existing = final if (final / "meta.json").is_file() else None
            if existing is None or not _same_content(existing, {"files": _describe(arrays, known)}):          # (an entry of the same name that holds anything else is of no use)
                return None
            register(kind, digest, existing)
            return existing
        base.mkdir(parents=True, exist_ok=True)
        tmp = base / f".tmp_{uuid.uuid4().hex[:12]}"
        try:
            full = _write_folder(tmp, arrays, meta, known)
            final = _publish(tmp, final, full)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    except OSError as e:
        global LAST_ERROR
        LAST_ERROR = repr(e)
        return None
    _mark_checked(final)                                      # (what was just written is not read again by the first lookup: that took seconds for a large array)
    register(kind, digest, final)
    prune(base, digest)
    return final


def add_files(folder: Path, sub: str, arrays: dict[str, np.ndarray], meta: dict[str, Any], known: dict[str, str] | None = None) -> Path | None:
    """Adds a sub-entry `folder/sub/` (atomically) to an existing entry; an existing one is validated like in `write_entry` (kept when intact and the same, replaced when damaged);
    None when it could not be written."""
    final = Path(folder) / sub
    need = sum(int(np.asarray(a).nbytes) for a in arrays.values())
    try:
        if not has_room(folder, need):
            return final if (final / "meta.json").is_file() and _same_content(final, {"files": _describe(arrays, known)}) else None
        tmp = Path(folder) / f".tmp_{uuid.uuid4().hex[:12]}"
        try:
            full = _write_folder(tmp, arrays, meta, known)
            final = _publish(tmp, final, full)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    except OSError as e:
        global LAST_ERROR
        LAST_ERROR = repr(e)
        return None
    _mark_checked(final)
    return final


def prune(base: Path, keep_digest: str) -> None:
    """Deletes the oldest entries of other keys from the state-cache folder of a work folder: the entries of `keep_digest`, those of the `RECENT_KEYS` digests written or found
    lately in this process (the chain of analyses that is running) and the `KEEP_OTHER` newest of the rest stay (what a reader still maps stays too: a folder Windows will not
    delete is left)."""
    try:
        dirs = [d for d in Path(base).iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return
    with _LOCK:
        marks = [f"_{x[:20]}_" for x in [keep_digest, *_RECENT]]           # the key just written and those of the analyses in progress
    def age(d: Path) -> float:
        try:
            return d.stat().st_mtime
        except OSError:
            return 0.0
    others = sorted((d for d in dirs if not any(m in d.name for m in marks)), key=age, reverse=True)
    for d in others[KEEP_OTHER:]:
        _drop(d)
    for d in Path(base).glob(".old_*"):
        _drop(d)


def array(folder: Path, name: str, mmap: bool = True) -> np.ndarray:
    """One array of an entry, read-only memory-mapped (`allow_pickle=False`)."""
    return np.load(Path(folder) / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)


def content_of(folder: Path, name: str) -> str | None:
    """The content hash of an array of an entry as the meta recorded it (what a result computed from that array can be tied to)."""
    m = meta_of(folder) or {}
    return ((m.get("files") or {}).get(name) or {}).get("hash")

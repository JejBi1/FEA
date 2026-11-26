"""The direct modal analysis (`modal_run.run_modes`) in the state cache (`fea_core.state_cache`): the modes, the factor's static solutions and – added later by the callers
that need them – the modal stresses and reactions are stored in the work folder of the run that computed them and loaded memory-mapped by the next one.

An entry holds the lowest `n_found` modes of one window [f_min, f_max] that were asked for as `requested` modes: it answers every request of the same window for `modes` ≤
`n_found` modes (the first `modes` of a larger analysis equal a smaller analysis to the tolerance of the eigenvalue solver), and every request for more modes when the window
held fewer than were requested (`n_found` < `requested`: the analysis found them all). The static solutions are stored per COLUMN of the right-hand side: the three unit
accelerations along x, y, z and one column per other load case; a case that is one global gravity load is a combination of the unit columns (`fea_core.gravity_states`).
"""
from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import gravity_states as GS
from .. import state_cache as SC

KIND = "modes"
UNIT_COLUMNS = ["g0", "g1", "g2"]


class _Slot:
    """The lock of one study digest and the number of threads that hold or wait for it (the slot is dropped with the last of them)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.refs = 0


_COMPUTE: dict[str, _Slot] = {}
_COMPUTE_GUARD = threading.Lock()


@contextlib.contextmanager
def compute_lock(digest: str, cancel: Any = None, log: Any = None, error: type[Exception] | None = None):
    """A lock per study digest: two analyses of the same model (the natural frequencies and the dynamics in side jobs) run one after the other, so the second finds the first stored.
    A thread that has to wait polls the `cancel` event (the wait ends with `error`, default `direct.DirectError`, "Analysis canceled by user") and says so in `log`. The lock is
    re-entrant for one thread; the table of locks holds only the digests somebody holds or waits for."""
    with _COMPUTE_GUARD:
        slot = _COMPUTE.setdefault(digest, _Slot())
        slot.refs += 1
    got = False
    try:
        if not slot.lock.acquire(blocking=False):
            if log is not None:
                log("Another analysis of this model is running - waiting for it (the result is shared)")
            while not slot.lock.acquire(timeout=0.2):
                if cancel is not None and cancel.is_set():
                    if error is None:
                        from . import DirectError as error
                    raise error("Analysis canceled by user")
        got = True
        yield
    finally:
        if got:
            slot.lock.release()
        with _COMPUTE_GUARD:
            slot.refs -= 1
            if slot.refs <= 0 and _COMPUTE.get(digest) is slot:
                del _COMPUTE[digest]


def plan_columns(cases: list[Any]) -> tuple[list[str], list[Any], list[np.ndarray]]:
    """(column names, the load case behind each EXTRA column, the coefficients over the columns for every requested case). The columns are the three unit accelerations and one
    column per distinct non-gravity case."""
    names = list(UNIT_COLUMNS)
    extra: list[Any] = []
    coeffs: list[np.ndarray] = []
    index: dict[str, int] = {}
    for lc in cases:
        v = GS.pure_gravity(lc)
        if v is not None:
            c = np.zeros(3)
            c[:] = v
            coeffs.append(c)
            continue
        sig = "c:" + SC.case_signature(lc)
        if sig not in index:
            index[sig] = len(names)
            names.append(sig)
            extra.append(lc)
        coeffs.append(np.eye(len(UNIT_COLUMNS) + len(index))[index[sig]])
    n = len(names)
    return names, extra, [np.pad(c, (0, n - len(c))) for c in coeffs]


def _tag(f_min: float, f_max: float | None, requested: int, columns: list[str]) -> str:
    import hashlib
    h = hashlib.sha256(f"{float(f_min)!r}|{f_max!r}|{requested}|{'|'.join(columns)}".encode("ascii")).hexdigest()
    return h[:12]


def lookup(digest: str, n: int, extra_nodes: int | None, modes: int, f_min: float, f_max: float | None, columns: list[str], skipped: bool = False) -> Path | None:
    """The newest intact entry of the digest that answers a request of `modes` modes in the window and holds every column asked for; None = compute. skipped = only an entry whose
    unit accelerations were found impossible to solve (`statics_skipped` in its meta: it would be asked for them again and again)."""
    need = set(columns)
    for folder in SC.entries(KIND, digest):
        m = SC.meta_of(folder)
        if m is None or m.get("n") != n or float(m.get("f_min", 0.0)) != float(f_min) or m.get("f_max") != (None if f_max is None else float(f_max)):
            continue
        if not (int(m["n_found"]) >= int(modes) or int(m["n_found"]) < int(m["requested"])):
            continue
        if not need <= set(m["columns"]):
            continue
        if skipped and not m.get("statics_skipped"):
            continue
        if extra_nodes is not None and int(m.get("extra_nodes", -1)) != int(extra_nodes):
            continue
        return folder
    return None


def build(folder: Path, modes: int, columns: list[str], coeffs: list[np.ndarray], t0: float, source: str = ""):
    """`DirectModes` of the first `modes` modes of the entry (the shapes stay memory-mapped), the static solutions of the requested cases from the stored columns."""
    from .modal_run import DirectModes
    m = SC.meta_of(folder) or {}
    n = int(m["n"])
    k = min(int(modes), int(m["n_found"]))
    lam = np.array(SC.array(folder, "lam")[:k])
    gamma = np.array(SC.array(folder, "gamma")[:k])
    rigid = np.array(SC.array(folder, "rigid")[:k])
    total_free = np.array(SC.array(folder, "total_free"))
    shapes = SC.array(folder, "shapes")[:k]
    stored = list(m["columns"])
    cols = SC.array(folder, "cols") if stored else None
    pos = [stored.index(c) for c in columns]
    coeffs_all: list[np.ndarray] = []
    for cf in coeffs:                                    # the coefficients over the columns of the ENTRY (it may hold more columns than this request)
        ca = np.zeros(len(stored))
        for j, c in zip(pos, cf):
            ca[j] = c
        coeffs_all.append(ca)
    st_full: list[np.ndarray] = []
    for ca in coeffs_all:
        v = np.zeros(0 if cols is None else cols.shape[1])
        for j in np.nonzero(ca)[0]:
            v += float(ca[j]) * np.asarray(cols[j] if cols is not None else 0.0, float)
        st_full.append(v)
    statics = [v[:3 * n].reshape(n, 3) for v in st_full] if coeffs else []
    info = dict(m.get("info") or {})
    eig = dict(info.get("eigen") or {})
    eig["modes"] = k
    info["eigen"] = eig
    info["times"] = {"total": time.time() - t0, "from_cache": True}
    info["cache"] = {"source": source or "state cache", "modes_stored": int(m["n_found"]), "modes_used": int(k)}
    dm = DirectModes(lam, shapes, gamma, total_free, float(m.get("mass_bodies", 0.0)), float(m.get("mass_added", 0.0)), rigid, statics, st_full, info, None)
    dm.cache, dm.columns_all, dm.coeffs_all, dm.cols = folder, stored, coeffs_all, cols
    dm.content = SC.content_of(folder, "shapes")
    return dm


def store(root: Path, digest: str, dm: Any, n: int, extra_nodes: int, requested: int, f_min: float, f_max: float | None, columns: list[str], cols: np.ndarray | None,
          skipped: str | None = None) -> Path | None:
    """Stores a freshly computed analysis (`dm` holds ALL the modes found); returns its folder (None = not stored). `dm.content` becomes the content hash of the shapes computed here
    – what a recovery stored beside them is tied to – and the folder is returned only when the entry holds exactly those shapes (an entry of the same name that holds other modes,
    e.g. of another sign, is never taken for this analysis)."""
    arrays = {"lam": dm.lam, "gamma": dm.gamma, "rigid": dm.rigid, "total_free": dm.total_free, "shapes": dm.shapes}
    if cols is not None:
        arrays["cols"] = cols
    meta = {"n": int(n), "extra_nodes": int(extra_nodes), "requested": int(requested), "n_found": int(len(dm.lam)), "f_min": float(f_min),
            "f_max": None if f_max is None else float(f_max), "columns": list(columns), "mass_bodies": float(dm.mass_bodies), "mass_added": float(dm.mass_added),
            "info": dm.info, "statics_skipped": skipped}
    h = SC.content_hash(dm.shapes)
    folder = SC.write_entry(root, KIND, digest, _tag(f_min, f_max, requested, columns), arrays, meta, known={"shapes": h})
    if folder is not None and SC.content_of(folder, "shapes") != h:
        folder = None
    dm.content = h if folder is not None else None
    return folder


def take(dm: Any, k: int) -> Any:
    """The first `k` modes of `dm` (the statics stay)."""
    if k >= len(dm.lam):
        return dm
    import copy
    out = copy.copy(dm)
    out.lam, out.shapes, out.gamma, out.rigid = dm.lam[:k], dm.shapes[:k], dm.gamma[:k], dm.rigid[:k]
    info = dict(dm.info)
    eig = dict(info.get("eigen") or {})
    eig["modes"] = k
    info["eigen"] = eig
    info["shared_analysis"] = {"modes_computed": int(len(dm.lam)), "modes_used": int(k)}
    out.info = info
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------- recovery
def _rec_dirs(folder: Path) -> list[Path]:
    return sorted((p for p in Path(folder).glob("rec_*") if (p / "meta.json").is_file()), key=lambda p: p.name)


def load_recovery(dm: Any, k: int, want_stress: bool) -> dict[str, Any] | None:
    """The stored modal reactions (and stresses) of the first `k` modes and of every column, memory-mapped; None when the entry has none that cover `k` modes. A recovery belongs
    to the modes it was computed from: it is used only when the content hash of those modes (recorded when it was stored) is that of the modes of `dm`, and when it is intact."""
    if dm.cache is None or getattr(dm, "content", None) is None:
        return None
    for sub in _rec_dirs(dm.cache):
        m = SC.meta_of(sub)
        if m is None or int(m["k"]) < k or list(m["columns"]) != list(dm.columns_all) or m.get("modes_content") != dm.content:
            continue
        if want_stress and not ((m.get("files") or {}).get("stress") and (m.get("files") or {}).get("col_S")):
            continue
        if not SC.verified(sub):
            continue
        out = {"rf": SC.array(sub, "rf")[:k], "cn": np.array(SC.array(sub, "cn")), "col_RF": SC.array(sub, "col_RF"), "stress": None, "col_S": None}
        if (m.get("files") or {}).get("stress"):
            out["stress"], out["col_S"] = SC.array(sub, "stress")[:k], SC.array(sub, "col_S")
        return out
    return None


def store_recovery(dm: Any, k: int, stress: np.ndarray | None, rf: np.ndarray, cn: np.ndarray, col_S: np.ndarray | None, col_RF: np.ndarray) -> None:
    """Stores the modal reactions (and stresses) next to the modes they were computed from. An entry WITH stresses always holds the stresses of the columns too (an empty array for
    a model without columns: a free or only partly held body has none) – a half entry is never written. Only a complete recovery of the modes is stored: never a basis without
    reactions (the caller does not call it)."""
    if dm.cache is None or getattr(dm, "content", None) is None:
        return
    arrays = {"rf": rf, "cn": np.asarray(cn, np.int64), "col_RF": col_RF}
    with_stress = stress is not None
    if with_stress:
        arrays["stress"] = stress
        arrays["col_S"] = col_S if col_S is not None else np.zeros((0,) + tuple(np.shape(stress)[1:]), np.float32)
        if col_S is None and len(dm.columns_all):
            return                                       # the columns exist but their stresses were not recovered: nothing consistent to store
    SC.add_files(dm.cache, f"rec_{int(k):04d}_{int(with_stress)}", arrays, {"k": int(k), "columns": list(dm.columns_all), "modes_content": dm.content})

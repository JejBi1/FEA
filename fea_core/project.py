"""Project file .fea = zip: model.step + study.json (+ results.npz, results_mesh.npz)."""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .results import sidecar_names as _result_sidecars
from .study import Study, StudyFormatError

STEP_NAME = "model.step"
META_NAME = "model.json"          # metadata from CAD for model.step (cad_import.sidecar_path) – the face table of the add-in
RESULTS_NAME = "results.npz"
RESULT_FILES = (RESULTS_NAME, *_result_sidecars(RESULTS_NAME))   # the files Results.save writes (results_2d, _shell, _dyn, _sub optional)
MOTION_FILE = "results_motion.npz"                                     # result of a mechanism (fea_core.motion)
SURROGATE_FILE = "surrogate.npz"                                       # surrogate model of a parametric study (fea_core.surrogate, no pickle)
PROJECT_FILES = (STEP_NAME, META_NAME, "study.json", *RESULT_FILES, MOTION_FILE, SURROGATE_FILE)   # everything the project may contain


MAX_UNPACKED = 16 << 30          # largest unpacked project (16 GB) – protection against a "zip bomb"
MAX_RATIO = 400                  # compression ratio of a member above 100 MB (STEP ~10, uncompressed .npz 1)
MAX_RATIO_MID = 1000             # compression ratio of a member of 10–100 MB


def _members(z: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Archive members that are unpacked: only the known project files in the root (nothing else from a foreign .fea reaches
    the working folder), with a check of the size after unpacking.
    """
    out, total = [], 0
    for info in z.infolist():
        if info.is_dir() or info.filename not in PROJECT_FILES or info.filename == SURROGATE_FILE:
            continue                                  # the surrogate is extracted separately: it can never stop a project from opening
        total += info.file_size
        ratio = info.file_size / max(info.compress_size, 1)
        if (info.file_size > 100 << 20 and ratio > MAX_RATIO) or (info.file_size > 10 << 20 and ratio > MAX_RATIO_MID) \
                or total > MAX_UNPACKED:
            raise StudyFormatError(f"The project file is damaged or suspicious: {info.filename} would unpack to {info.file_size / 2 ** 30:.1f} GB")
        out.append(info)
    return out


SURROGATE_MAX_BYTES = 16 << 20   # a surrogate file of the largest model is well below 4 MB; a bigger member is not read
DAMAGED_MARK = 0                 # a surrogate member that could not be unpacked is left as an EMPTY file: load_surrogate_file reports it


def _extract_surrogate(z: zipfile.ZipFile, workdir: Path) -> None:
    """Unpacks the surrogate member on its own: whatever is wrong with it (a damaged or truncated stream, a failed CRC, a bad local header,
    a declared size beyond the limit, a disk error) leaves an empty `surrogate.npz` in `workdir` – `load_surrogate_file` then drops the
    model with a warning – and never an exception: the study, the geometry and the results of the project open regardless."""
    try:
        info = z.getinfo(SURROGATE_FILE)
    except KeyError:
        return
    target = workdir / SURROGATE_FILE
    try:
        if info.is_dir() or info.file_size > SURROGATE_MAX_BYTES or info.compress_size > SURROGATE_MAX_BYTES:
            raise ValueError("size")
        with z.open(info) as src, open(target, "wb") as dst:
            left = SURROGATE_MAX_BYTES + 1
            while chunk := src.read(min(1 << 20, left)):
                left -= len(chunk)
                if left <= 0:
                    raise ValueError("size")
                dst.write(chunk)
    except Exception:  # noqa: BLE001 – zlib.error, BadZipFile (CRC), EOFError, NotImplementedError (unknown compression), RuntimeError, OSError …
        try:
            target.write_bytes(b"")
        except OSError:
            pass


def _sidecar(step_path) -> Path | None:
    """Valid metadata from CAD next to the STEP (not a study JSON or a foreign JSON), otherwise None."""
    from .cad_import import load_sidecar, sidecar_path
    try:
        return sidecar_path(step_path) if load_sidecar(step_path) is not None else None
    except OSError:
        return None


def save_project(path: str | Path, study: Study, step_path: str | Path, results=None, motion=None, surrogate=None,
                 warnings: list[str] | None = None) -> None:
    """STEP and study.json compressed (DEFLATE), the results .npz without a second compression (ZIP_STORED): float64 hardly shrinks
    with deflate and the compression cost ~1 s (197 k DOF, 3 states). motion = MotionResult, surrogate = SurrogateModel (optional; it is
    saved with the fingerprint of the study it was fitted for, so a study that has changed since is still recognised after reopening –
    the reason is appended to `warnings`).
    """
    if surrogate is not None and warnings is not None:
        why = surrogate.check_study(study, step_path or None)
        if why:
            warnings.append("The surrogate model in this project belongs to an earlier version of the study: " + why)
    path = Path(path)
    tmp = Path(tempfile.mkdtemp(prefix="fea_save_"))
    try:
        st = Study.from_dict(study.to_dict())
        no_geo = not str(step_path or "").strip()     # only a frame system (without STEP geometry)
        st.geometry = "" if no_geo else STEP_NAME
        st.save(tmp / "study.json")
        files = ([] if no_geo else [(Path(step_path), STEP_NAME, zipfile.ZIP_DEFLATED)]) \
            + [(tmp / "study.json", "study.json", zipfile.ZIP_DEFLATED)]
        meta = _sidecar(step_path) if not no_geo else None
        if meta is not None:                          # metadata from CAD (face table for the update from CAD)
            files.append((meta, META_NAME, zipfile.ZIP_DEFLATED))
        if results is not None:
            results.save(tmp / "results.npz")
            files += [(tmp / name, name, zipfile.ZIP_STORED) for name in RESULT_FILES if (tmp / name).exists()]
        if motion is not None:
            motion.save(tmp / MOTION_FILE)
            files.append((tmp / MOTION_FILE, MOTION_FILE, zipfile.ZIP_STORED))
        if surrogate is not None:
            surrogate.save(tmp / SURROGATE_FILE)
            files.append((tmp / SURROGATE_FILE, SURROGATE_FILE, zipfile.ZIP_DEFLATED))
        part = path.with_suffix(path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
                for src, name, how in files:
                    z.write(src, name, compress_type=how)
            part.replace(path)  # atomically – an unfinished file never overwrites a valid one
        except BaseException:
            part.unlink(missing_ok=True)              # no half-written .fea.tmp next to the project
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_project(path: str | Path, workdir: str | Path) -> tuple[Study, Path | None, object]:
    """Unpacks the project into workdir. Returns (study, path to the STEP, results or None).

    Project files that remained in workdir from earlier are deleted first – otherwise a project without results would take over
    the old results.npz of another project.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_FILES:
        (workdir / name).unlink(missing_ok=True)
    with zipfile.ZipFile(path) as z:
        for info in _members(z):
            with z.open(info) as src, open(workdir / info.filename, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        _extract_surrogate(z, workdir)
    study = Study.load(workdir / "study.json")
    step = workdir / STEP_NAME
    if str(study.geometry or "").strip() or step.exists():
        study.geometry = str(step)
    else:                                            # only a frame system
        step = None
    results = None
    if (workdir / "results.npz").exists():
        from .results import Results
        results = Results.load(workdir / "results.npz")
    return study, step, results


def load_study_json(path: str | Path) -> tuple[Study, Path | None]:
    """Study from a JSON file (`python -m fea_core run study.json`, MCP fea_study_save) → (study, STEP or None
    = only a frame system). A relative path to the geometry is taken relative to the folder of the file; Study.geometry then contains
    the absolute path. Shared by the GUI (opening .json) and MCP (fea_study_open).
    """
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(d, dict) or "name" not in d or "geometry" not in d:
        if isinstance(d, dict) and "parts" in d:           # <name>.json next to the STEP (add-in / CAD macro, REST)
            raise StudyFormatError(f"File {path.name} is not an FEA Studio study – it is CAD metadata, loaded by opening the STEP file with the same name")
        raise StudyFormatError(f"File {path.name} is not an FEA Studio study")
    st = Study.from_dict(d)
    if not str(st.geometry or "").strip():                 # only a frame system
        return st, None
    from .pathcheck import is_unc
    if is_unc(st.geometry):                                # a foreign study must not reach a network server
        raise StudyFormatError(f"Study geometry on a network (UNC) path is not allowed: {st.geometry}")
    geo = st.geometry_path(path.parent).resolve()
    if not geo.is_file():
        raise FileNotFoundError(f"Study geometry not found: {geo}")
    st.geometry = str(geo)
    return st, geo


def load_surrogate_file(workdir: str | Path, warnings: list[str] | None = None):
    """Surrogate model from an unpacked project (load_project) – SurrogateModel or None. A damaged, crafted or unreadable surrogate.npz
    NEVER stops a project from opening: whatever goes wrong while reading it (a bad zip, a truncated or corrupt member, nesting, memory,
    a malformed header …) drops the surrogate; the reason is appended to `warnings` (shown to the user by the caller)."""
    p = Path(workdir) / SURROGATE_FILE
    if not p.exists():
        return None
    try:
        if p.stat().st_size == DAMAGED_MARK:
            raise ValueError("the surrogate member of the project file is damaged (its data cannot be unpacked)")
        from .surrogate.fit import load_surrogate
        return load_surrogate(p)
    except Exception as e:  # noqa: BLE001 – the surrogate comes from a foreign .fea: BadZipFile, EOFError, zlib.error, RecursionError, MemoryError … all end here
        if warnings is not None:
            warnings.append(f"The surrogate model in the project could not be read and was dropped ({type(e).__name__}: {str(e)[:200]}); "
                            "the project opens without it – run the parametric study and fit it again.")
        return None


def load_motion(workdir: str | Path):
    """Result of a mechanism from an unpacked project (load_project) – MotionResult or None."""
    p = Path(workdir) / MOTION_FILE
    if not p.exists():
        return None
    from .motion.results import MotionResult
    return MotionResult.load(p)

"""Update of a FEA Studio installation – version and list of the program files (manifest). Only the standard library.

The installation remembers (`.venv/fea_modules.json` → `version`, `.venv/fea_install_manifest.json`) the version and the program files
it installed. A package carries the manifest of its version (`installer/manifest.json`, created by `make_package.py`); the manifests
of older releases without their own record are in `installer/manifests/<version>.json`.

An obsolete file (it was in the old manifest, not in the new one) is deleted only when:
  - it lies in the program folder and is not in a protected part (`.venv`, `tools`, `scratch`, `dist`, `.git`, user data…),
  - it is not a project `*.fea` (nor an example – the user may have saved work into it),
  - its contents have not changed from the released one (the SHA-256 digest matches) – a modified file stays and is listed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from setup_lang import T

MANIFEST_NAME = "fea_install_manifest.json"         # in .venv – the manifest of the INSTALLED version
PACKAGE_MANIFEST = Path("installer") / "manifest.json"
HISTORY_DIR = Path("installer") / "manifests"
LEGACY_VERSION = "1.0.0"                             # installation without a version record (release 1.0.0 and older)

PROTECTED_PREFIX = (".venv/", "tools/", "scratch/", "dist/", ".git/", ".claude/", "verification/out", "user/",
                    "data_user/", "projects/", "projekty/")
PROTECTED_SUFFIX = (".fea",)
# data that the package does not carry, but the user has them (PragTic – L2, downloaded by pragtic_fetch): the update does not delete them
PROTECTED_FILES = ("fea_core/fatigue/data/pragtic_vmat.json",)
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


# ------------------------------------------------------------------ verze
def parse_version(v: str) -> tuple:
    """"1.0.1" → (1, 0, 1); the suffix ("1.1.0rc1") is ignored, missing parts = 0."""
    parts = [int(m) for m in re.findall(r"\d+", (v or "").split("+")[0])[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


# The builds of the branch next-1.5 were called 1.5.0 until the release got its number 1.4.2; the state of an installation made by such a build (also the
# test installations on Linux and Windows) still says 1.5.0. It is the SAME version as 1.4.2 – never a newer one (no refused downgrade) and no step to run
# again (`migrations.pending` takes the steps with old < step <= new) – as long as the program that reads it is older than 1.5.0.
# BINDING: the next feature release after 1.4.2 is 1.5.1, NEVER 1.5.0 – a real 1.5.0 would be read as 1.4.2 by every installation that still carries the state
# "1.5.0" (installer/README.md "Version numbers"; tests/test_update.py::test_the_program_is_never_version_1_5_0 fails when `__version__` becomes a pre-release name).
PRERELEASE_NAMES = {"1.5.0": "1.4.2"}


def state_version(v: str, program: str) -> str:
    """The version of an installation state as an update reads it: a pre-release name (`PRERELEASE_NAMES`) counts as the release it became while the
    program (`program`, `fea_core.__version__`) is older than that name; any other version, and every version once the program reaches the name, is unchanged."""
    v = (v or "").strip()
    new = PRERELEASE_NAMES.get(v)
    if new and parse_version(program) < parse_version(v):
        return new
    return v


def program_version(root: Path) -> str:
    """Version of the program in the folder (without import – fea_core/__init__.py is only read). The single reader of
    `fea_core.__version__` for the installer and make_package.py."""
    m = re.search(r'__version__ = "(.+?)"', (root / "fea_core" / "__init__.py").read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


def make_build_id(root: Path | None = None, when: float | None = None, ref: str = "HEAD") -> str:
    """Build id of a package: UTC time + short git hash of the packed commit `ref` ("20261109T201500Z-553e1e86"; without git
    only the time). Two packages of the same version with different build ids are different builds (the installer then updates,
    not repairs). `ref` = the packed ref (make_package --ref) – HEAD would name another commit than the one in the package."""
    import subprocess
    import time
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(when))
    try:
        r = subprocess.run(["git", "rev-parse", "--short=8", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=str(root or Path.cwd()), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
        short = r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        short = ""
    return f"{stamp}-{short}" if re.fullmatch(r"[0-9a-f]{4,40}", short) else stamp


# ------------------------------------------------------------------ version locks (installer/locks/*.txt)
def norm_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> dict[str, tuple[str, list[str]]]:
    """Lock / requirements text (with or without digests, continuation lines `\\`) → {normalised name: (requirement, [sha256:…])}.
    Shared by make_locks.py (writing) and fea_setup.sync_packages (installing with --require-hashes)."""
    out: dict[str, tuple[str, list[str]]] = {}
    logical, cur = [], ""
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("--hash") else raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            if cur:
                logical.append(cur)
                cur = ""
            continue
        cont = line.endswith("\\")
        cur += " " + line.rstrip("\\").strip()
        if not cont:
            logical.append(cur)
            cur = ""
    if cur:
        logical.append(cur)
    for item in logical:
        toks = item.split()
        hashes = [t.split("=", 1)[1] for t in toks if t.startswith("--hash=")]
        req = " ".join(t for t in toks if not t.startswith("--hash=")).strip()
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        if m:
            out[norm_name(m.group(1))] = (req, hashes)
    return out


# ------------------------------------------------------------------ digests and manifest
def sha256_file(path: Path) -> str:
    """Full SHA-256 (hex) of a file, read in 1 MB blocks – the single implementation for fea_setup.py and make_package.py."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash(path: Path) -> str:
    """Short digest of the manifests (SHA-256, 16 hex characters)."""
    return sha256_file(path)[:16]


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def manifest_of_dir(root: Path, version: str, skip_prefix: tuple = ("tools/",), build: str | None = None) -> dict:
    """Manifest of an unpacked package: {"version", ["build",] "files": {relative path: digest}}."""
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not SKIP_DIRS & set(p.parts):
            rel = p.relative_to(root).as_posix()
            if not rel.startswith(skip_prefix) and rel != PACKAGE_MANIFEST.as_posix():
                files[rel] = file_hash(p)
    out = {"version": version, "files": files}
    if build:
        out["build"] = build
    return out


def manifest_build(root: Path) -> str | None:
    """Build id of the package in the folder (installer/manifest.json), None for a package without it (older or from git)."""
    m = read_manifest(root / PACKAGE_MANIFEST)
    b = (m or {}).get("build")
    return b if isinstance(b, str) and b else None


def manifest_of_zip(path: Path, version: str, skip_prefix: tuple = ("tools/",)) -> dict:
    """Manifest of a released zip package (FEAStudio-<version>/…) – for installer/manifests/<version>.json."""
    import zipfile
    files = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("/") or "/" not in n:
                continue
            rel = n.split("/", 1)[1]
            if rel.startswith(skip_prefix) or rel == PACKAGE_MANIFEST.as_posix() or SKIP_DIRS & set(rel.split("/")):
                continue
            files[rel] = bytes_hash(z.read(n))
    return {"version": version, "files": files}


def manifest_of_tar(path: Path, version: str, skip_prefix: tuple = ("tools/",)) -> dict:
    """Manifest of a released tar.gz package (FEAStudio-<version>/…), by the rules of manifest_of_zip."""
    import tarfile
    files = {}
    with tarfile.open(path) as t:
        for m in t:
            if not m.isfile() or "/" not in m.name:
                continue
            rel = m.name.split("/", 1)[1]
            if rel.startswith(skip_prefix) or rel == PACKAGE_MANIFEST.as_posix() or SKIP_DIRS & set(rel.split("/")):
                continue
            files[rel] = bytes_hash(t.extractfile(m).read())
    return {"version": version, "files": files}


def manifest_of_release(zip_path: Path, tar_path: Path, version: str) -> dict:
    """installer/manifests/<version>.json: every program file of the Windows zip AND of the Linux tar.gz (each carries the sources of its own CalculiX in
    licenses/sources/), so that an update on either system can remove what its release had. A file of both packages must be the same in both."""
    win, linux = manifest_of_zip(zip_path, version)["files"], manifest_of_tar(tar_path, version)["files"]
    differ = sorted(k for k in set(win) & set(linux) if win[k] != linux[k])
    if differ:
        raise ValueError(f"{version}: files that differ between the zip and the tar.gz: {', '.join(differ[:10])}")
    return {"version": version, "files": {**linux, **win}}


def read_manifest(path: Path) -> dict | None:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or not isinstance(m.get("files"), dict):
        return None
    return m


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")


def history_manifests(root: Path) -> dict[str, dict]:
    """Manifests of older releases bundled with the installer (version → manifest)."""
    out = {}
    d = root / HISTORY_DIR
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        m = read_manifest(p)
        if m:
            out[m.get("version") or p.stem] = m
    return out


def previous_manifest(root: Path, venv: Path, old_version: str) -> dict | None:
    """Manifest of the installed version: saved at the installation, otherwise the manifest of the release from installer/manifests."""
    m = read_manifest(venv / MANIFEST_NAME)
    if m:
        return m
    return history_manifests(root).get(old_version)


# ------------------------------------------------------------------ deleting obsolete files
def is_protected(rel: str) -> bool:
    if not rel or rel.startswith(("/", "\\")) or ":" in rel or ".." in Path(rel).parts:
        return True
    low = rel.replace("\\", "/").lower()
    return low.startswith(PROTECTED_PREFIX) or low.endswith(PROTECTED_SUFFIX) or low in PROTECTED_FILES


def obsolete_files(old: dict, new: dict) -> dict[str, str]:
    """Files of the old manifest that the new one does not have (path → digest of the released form)."""
    newf = {k.lower() for k in new.get("files", {})}         # Windows: the letter case in the path does not matter
    return {k: v for k, v in old.get("files", {}).items() if k.lower() not in newf}


def remove_obsolete(root: Path, old_files: dict[str, str], say=print,
                    reported: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Deletes the obsolete program files (see the module). Returns (deleted, kept modified). `reported` = the set of kept files
    already listed (shared between the calls of one update – a modified file is listed only once).
    """
    removed, kept = [], []
    root = root.resolve()
    dirs = set()
    for rel, digest in sorted(old_files.items()):
        if is_protected(rel):
            continue
        p = (root / rel)
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue                                           # a link outside the program folder
        if not p.is_file():
            continue
        if digest and file_hash(p) != digest:
            kept.append(rel)
            continue
        p.unlink()
        removed.append(rel)
        dirs.add(p.parent)
        cache = p.parent / "__pycache__"                       # bytecode of a deleted module
        if p.suffix == ".py" and cache.is_dir():
            for c in cache.glob(p.stem + ".*.pyc"):
                c.unlink(missing_ok=True)
    for d in sorted(dirs, key=lambda x: len(x.parts), reverse=True):  # empty folders after the deleted files
        while d != root and d.is_dir():
            rest = [x for x in d.iterdir() if not (x.name == "__pycache__" and x.is_dir() and not any(x.iterdir()))]
            if rest:
                break
            for x in d.iterdir():
                x.rmdir()
            d.rmdir()
            d = d.parent
    if removed:
        say(T(f"zastaralé soubory programu smazány ({len(removed)}): ", f"obsolete program files deleted ({len(removed)}): ")
            + ", ".join(removed[:8]) + (" …" if len(removed) > 8 else ""))
    new_kept = [k for k in kept if reported is None or k not in reported]
    if reported is not None:
        reported.update(kept)
    if new_kept:
        say(T(f"upravené zastaralé soubory ponechány ({len(new_kept)}): ", f"modified obsolete files kept ({len(new_kept)}): ")
            + ", ".join(new_kept[:8]) + (" …" if len(new_kept) > 8 else ""))
    return removed, kept


def orphan_bytecode(root: Path, dirs=("fea_core", "fea_gui", "installer", "examples", "verification", "tests", "adapters")) -> list[Path]:
    """`__pycache__/x.cpython-311.pyc` without the source `x.py` (a module deleted in the new version)."""
    out = []
    for top in dirs:
        base = root / top
        if not base.is_dir():
            continue
        for pyc in base.rglob("__pycache__/*.pyc"):
            src = pyc.parent.parent / (pyc.name.split(".")[0] + ".py")
            if not src.exists():
                out.append(pyc)
    return out

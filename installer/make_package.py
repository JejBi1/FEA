#!/usr/bin/env python3
"""Builds the distribution packages of FEA Studio for Windows and Linux (run on the development machine in the venv of the project):

    python installer/make_package.py [--ref HEAD | --worktree] [--allow-dirty] [--check-only] [--out dist] [--no-docs] [--no-examples] [--examples-no-results] [--no-linux-ccx] [--with-ccx-pastix]
                                     [--no-windows-ccx] [--no-runtime-sources] [--iscc path] [--no-setup] [--no-run] [--offline]

Output in `dist/`:
    FEAStudio-<version>.zip                 Windows (and universal): program files + installer, Python 3.11, uv.exe, CalculiX ccx_spooles.exe (prebuilt)
    FEAStudio-<version>.tar.gz              Linux: the program files with execute permissions and the prebuilt CalculiX ccx_spooles + the shared libraries
                                          it needs (REQUIRED: installer/linux/prebuilt/…tar.gz built at release by installer/linux/build_ccx.sh – the
                                          installer never compiles; a missing archive stops the build, `--no-linux-ccx` is the opt-out).
                                          ccx_pastix (PaStiX 6.4.0, LGPL-3.0, linked statically into the GPL-2.0-only CalculiX)
                                          is NOT in a package until the licence question L4 is answered (docs/projekt/F13_plan_speed_memory_licences_native.md):
                                          only `--with-ccx-pastix` takes it.
    FEAStudio-<version>-setup.exe           Windows installer (Inno Setup 6: ISCC from tools/innosetup – installer/get_innosetup.py,
                                          from PATH or --iscc); choice of modules, update over an old installation, bundled uv.exe
    FEAStudio-<version>-offline-setup.exe   (--offline) setup.exe plus all Python packages of the standard profile
                                          (tools/offline/wheels, digests from PyPI) – installation without internet
    FEAStudio-<version>-linux-x86_64.run    Linux self-extracting installer (sh header + .tar.gz): --target, --update
    SHA256SUMS.txt

Each package carries `installer/manifest.json` = the program files of this version with digests (the update uses it to delete files
that the new version no longer has – installer/upgrade.py) and the folder `licenses/` with the texts of the third-party licences (installer/collect_licenses.py),
`licenses/SOURCES.md` (sources of the shipped GPL / LGPL binaries) and `licenses/sources/` with the sources of everything their CalculiX is built from or shipped
with: CalculiX, SPOOLES, ARPACK-ng, LAPACK, rust-src and the crates of faer in both; the Windows packages the MSYS2 source packages of the GCC / mingw-w64 run-time
libraries linked into ccx_spooles.exe, the Linux packages the GCC archive of the pinned conda toolchain with the conda-forge recipe and patches (libgfortran, libgcc,
libgomp, libquadmath; installer/linux/ccx_gccsrc.lock) – libquadmath is LGPL; `--no-runtime-sources` leaves the run-time and Rust sources out of a test package; the
offline setup also Gmsh. The manifest of the prebuilt Windows CalculiX must match the executable and the build scripts of the checkout.
The private cross-check against Abaqus (verification/abaqus_*, its report and dashboard files) is never packaged (PRIVATE_*).

The sources always travel with the binaries (there is no written offer). A build whose licenses/sources/ lacks the sources of binaries it carries
(--no-runtime-sources, --with-ccx-pastix) is a development build: every output file and the folder inside the archives get the suffix `-dev`, its
licenses/SOURCES.md starts with NOT FOR DISTRIBUTION and the build log says so (`distribution_blockers`).
No package carries MYSTRAN (the user downloads it): every package writer refuses a stage with a MYSTRAN program in it (`refuse_mystran`).

Without the example results the package is small (~10 MB): Python, libraries, CalculiX etc. are downloaded by the installer (`Install FEA Studio.bat` / `install.sh`)
according to the selected modules (fea_core/modules.json). Content = the committed state of `--ref` (default HEAD); a working tree
with uncommitted changes is refused unless `--allow-dirty`; `--worktree` packs the working tree incl. new (not ignored) files.
Before packing, everything staged is scanned for personal identifiers (anonymization: the local GitHub-export denylist from
.git/info/github-denylist.txt via scratch/github_export/github_export.py if present + absolute home paths, also inside zip members);
a finding stops the build. `--check-only` = only this check (temporary folder, no packages). Every package gets a build id
(UTC time + git hash) in installer/manifest.json – the installer updates over the same version with another build. The examples `examples/*.fea` go into the package WITH results (zip ~135 MB; `--examples-no-results` = only the input, ~10 MB),
the PDF manuals are generated if Qt WebEngine is available (docs module).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import upgrade  # noqa: E402  (installer/upgrade.py – manifest of the program files)
import collect_licenses  # noqa: E402  (installer/collect_licenses.py – third-party licences into licenses/)

EXCLUDE_PREFIX = ("scratch/", ".claude/", "dist/", "tools/", ".venv/", "verification/out", "docs/projekt/", ".basedpyright/",
                  "installer/linux/prebuilt/", "adapters/fusion/FEAStudio/__pycache__")
EXCLUDE_SUFFIX = (".pyc", ".fea", ".orig", ".rej")
EXCLUDE_FILES = {"installer/manifest.json", "install.log",      # the manifest is created when packing
                 "CLAUDE.md", "CLAUDE.local.md",                # internal development notes (do not distribute)
                 "fea_core/fatigue/data/pragtic_vmat.json",     # PragTic: do not distribute until the author agrees (L2)
                 "docs/Přehled změn textů v1.3.md"}              # internal draft of the help texts of 1.3 (an update removes it)
PRAGTIC = "fea_core/fatigue/data/pragtic_vmat.json"
# The cross-check against Abaqus is PRIVATE (user decision 11. 11. 2026): its scripts, results, report and the generators /
# intermediate files of its HTML dashboard are not distributed (the same paths are excluded from the GitHub export,
# .git/info/github-export-exclude.txt). Nothing that is shipped imports them (tests/test_fix_lic_private.py).
PRIVATE_PREFIX = ("verification/abaqus_", "verification/out_abaqus/")
PRIVATE_FILES = {
    "tests/test_abaqus_all.py", "docs/Srovnani_Abaqus.html",
    "verification/build_docs_abaqus_report.py", "verification/make_abaqus_drawings_full.py",
    # generators and intermediate files of the Abaqus comparison dashboard (read verification/out_abaqus)
    "verification/check_dev.py", "verification/drawings_source.js", "verification/escaped_html.txt",
    "verification/generate_drawings_js.py", "verification/generate_interactive_html.py", "verification/inline_compact.html",
    "verification/inline_component.html", "verification/inline_slim.html", "verification/make_inline.py",
    "verification/make_inline_slim.py", "verification/patch_build_docs.py", "verification/payload.json",
    "verification/prepare_dashboard.py", "verification/preview_args.json", "verification/render_payload.json",
    "verification/slim_preview_args.json", "verification/slim_render_payload.json", "verification/update_report_solvers.py",
}
EXEC ={"install.sh", "fea-studio", "installer/fea_setup.py", "installer/linux/build_ccx.sh", "installer/make_package.py",
        "installer/make_locks.py"}
CRLF = (".bat", ".cmd", ".ps1")         # like .gitattributes (eol=crlf); a .ps1 from the working tree is often LF
LF = (".sh", ".lock")             # .lock = the conda lock files of installer/linux (read by bash)


def version() -> str:
    return upgrade.program_version(ROOT)            # the single reader of fea_core.__version__ (installer/upgrade.py)


def file_list(ref: str | None) -> list[str]:
    # -z: paths without quotes and escape sequences (otherwise git writes diacritics as "P\305\231…" and the file was left out)
    if ref:
        cmd = ["git", "ls-tree", "-r", "-z", "--name-only", ref]
    else:
        cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=True)
    names = sorted({n for n in out.stdout.decode("utf-8").split("\0") if n})
    return [n for n in names if packaged(n) and (ROOT / n).is_file()]


def packaged(name: str) -> bool:
    """Does the file (path relative to the root, with /) belong to the release packages? Internal notes, tools, PragTic and the
    private Abaqus comparison do not.
    """
    return not (name.startswith(EXCLUDE_PREFIX + PRIVATE_PREFIX) or name.endswith(EXCLUDE_SUFFIX)
                or name in EXCLUDE_FILES or name in PRIVATE_FILES)


def dirty_files() -> list[str]:
    """Changed / new (not ignored) files of the working tree that would end up in a package – `git status`; changes in the paths
    the package leaves out (scratch/, tools/, CLAUDE.md…) do not count."""
    out = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=normal"], cwd=ROOT, capture_output=True,
                         check=True).stdout.decode("utf-8", "replace")
    names, items = [], out.split("\0")
    i = 0
    while i < len(items):
        e = items[i]
        i += 1
        if len(e) < 4:
            continue
        code, name = e[:2], e[3:]
        if code[0] in "RC":                     # rename / copy: the original path follows as a separate item
            i += 1
        if not name.startswith(EXCLUDE_PREFIX) and name not in EXCLUDE_FILES and not name.endswith(EXCLUDE_SUFFIX):
            names.append(name)
    return sorted(set(names))


# ------------------------------------------------------------------ anonymization scan of the package contents
def git_common_root() -> Path:
    """Main checkout of the repository (for a git worktree the folder that holds the shared .git)."""
    r = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    common = Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else ROOT / ".git"
    return common.parent if common.name == ".git" else ROOT


def load_export_scanner():
    """The scanner of the GitHub export (scratch/github_export/github_export.py – local only, not in git or packages) with its
    denylist from .git/info/github-denylist.txt + identity of the machine. None when the script is not available."""
    import importlib.util
    for base in dict.fromkeys((ROOT, git_common_root())):
        script = base / "scratch" / "github_export" / "github_export.py"
        if script.is_file():
            spec = importlib.util.spec_from_file_location("_fea_github_export", script)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:  # noqa: BLE001  (a broken local tool must not hide the rest of the check)
                print(f"  ! {script.name} cannot be loaded ({e}) – the denylist scan uses only the denylist file")
                return None
            return mod
    return None


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def simple_denylist() -> list[re.Pattern]:
    """Fallback without github_export.py: lines of <git common dir>/info/github-denylist.txt (`re:` = regex), case and diacritics
    folded. Empty list = no local denylist (the scan is then skipped with a warning)."""
    path = git_common_root() / ".git" / "info" / "github-denylist.txt"
    pats = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pats.append(re.compile(line[3:], re.I) if line.startswith("re:")
                        else re.compile(r"[\s_\-.]?".join(re.escape(w) for w in re.split(r"[\s_\-.]+", _fold(line)) if w)))
    return pats


def _payloads(data: bytes, where: str, depth: int = 0):
    """The file and, for a zip (.fea, .docx, .npz), its members (recursively 2 levels) – compressed members would hide paths."""
    yield where, data
    if data[:4] == b"PK\x03\x04" and depth < 2:
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return
        for info in z.infolist():
            yield f"{where}!{info.filename}<name>", info.filename.encode("utf-8")
            if info.file_size <= 200 * 1024 * 1024:
                yield from _payloads(z.read(info), f"{where}!{info.filename}", depth + 1)


# The synthetic home paths of the scanner tests: per file the EXACT strings that PRIVATE_PATH matches there (a fake archive built in a build folder, paths
# in error messages, the fixtures of the scanner tests). Nothing else is waived in these files – a real profile folder, another user name, a longer name that
# merely starts like a fixture still stops the build: `scrub_synthetic_paths` replaces only these exact matches by a neutral token before BOTH scanners (the
# home-path scan and the personal-name / generic-pattern denylist of the GitHub export) look at the file. (Raw strings: the source text is the match.)
_FIXTURES: dict[str, set[str]] = {
    "tests/test_ccx_packaging.py": {"/home/builder/"},
    "tests/test_install_desktop.py": {"/home/me/"},          # installation folders with a space in the tests of the menu entry
    "tests/test_direct_scrub.py": {r"C:\\Users\\Bob", "C:/Users/Bob", "/home/bob/"},
    "tests/test_modules.py": {"/home/builder/"},
    "tests/test_release_infra.py": {"/home/someone/", "C:/Users/someone", r"C:\Users\someone", r"c:\users\someone", "D:/USERS/someone",
                                    r"C:\\Users\\someone", "/Users/someone/"},
    "tests/test_security.py": {"/home/u/"},
    # redaction tests of the space checks and reports (1.5.1): made-up paths that the code must strip from messages and documents
    "tests/test_space_checks_rereview.py": {"/home/user/", r"C:\\Users\\me"},
    "tests/test_space_checks_review.py": {"/home/user/", r"C:\\Users\\someone"},
    "tests/test_space_checks_wire.py": {r"C:\\Users\\x"},
    "tests/test_space_report.py": {"/home/jsmith/", r"C:\Users\someone"},
    "tests/test_space_report_fixes.py": {"C:/Users/Kowalski"},
}
SYNTHETIC_HOME_PATHS: dict[str, frozenset[str]] = {rel: frozenset(s) for rel, s in _FIXTURES.items()}
SYNTHETIC_HOME_PATHS["installer/make_package.py"] = frozenset().union(*_FIXTURES.values())          # this file names all of them above
SYNTHETIC_TOKEN = b"<synthetic path>"


def scrub_synthetic_paths(rel: str, data: bytes) -> bytes:
    """`data` of the staged file `rel` with the exact synthetic fixtures of SYNTHETIC_HOME_PATHS replaced by a token; any other match of PRIVATE_PATH stays.
    Files that are not listed are returned unchanged."""
    allowed = SYNTHETIC_HOME_PATHS.get(rel)
    if not allowed:
        return data
    return PRIVATE_PATH.sub(lambda m: SYNTHETIC_TOKEN if m.group(0).decode("latin-1") in allowed else m.group(0), data)


def scan_private_paths(stage: Path, files: list[str]) -> list[tuple[str, str]]:
    """PRIVATE_PATH (absolute paths from home folders) in the staged files incl. zip members (the synthetic fixtures of SYNTHETIC_HOME_PATHS excepted)."""
    hits = []
    for rel in files:
        p = stage / rel
        if not p.is_file():
            continue
        for where, data in _payloads(p.read_bytes(), rel):
            if where == rel:
                data = scrub_synthetic_paths(rel, data)
            m = PRIVATE_PATH.search(data)
            if m:
                hits.append((where, m.group(0).decode("latin-1")[:12] + "…"))
    return hits


_SCANNER: tuple | None = None

# The public, anonymous author identity of the project (the repository's git user, CLAUDE.md "no company, no identity"). The
# export scanner adds the git identity of the machine to its denylist; when that identity IS this public one, its parts ("FEA Studio",
# "Studio", "noreply", "example.com", "example") would match the program's own name and every example path. Only these exact
# public terms are dropped – any other identity, name or denylist line stays.
PUBLIC_IDENTITY = ("FEA Studio", "Studio", "FEA", "noreply@example.com", "noreply", "example.com", "example")


def scanner_patterns(gx) -> list:
    """The denylist patterns of the export scanner without those that only stand for the public project identity."""
    pats = gx.build_denylist()
    fold = getattr(gx, "fold", lambda s: s.lower())
    public = [fold(s) for s in PUBLIC_IDENTITY]
    return [p for p in pats if not any(p.fullmatch(s) for s in public)]


def _scan_one(stage: str, rel: str) -> list[str]:
    """One file through the GitHub-export scanner (worker process; the scanner and the denylist are loaded once per process)."""
    global _SCANNER
    if _SCANNER is None:
        gx = load_export_scanner()
        _SCANNER = (gx, scanner_patterns(gx), gx.load_allow())
    gx, pats, allow = _SCANNER
    if rel.lower().endswith(".pdf"):
        # the export scanner needs > 10 min per manual PDF (thousands of streams × all text views); the PDFs are printed from the
        # help pages that are scanned in full → here the raw file and every decompressed stream once, as lower-case latin-1
        p = Path(stage) / rel
        data = p.read_bytes()
        out = []
        for k, chunk in enumerate([data, *gx.pdf_streams(data)]):
            text = chunk.decode("latin-1").lower()
            for pat in pats:
                m = pat.search(text)
                if m:
                    out.append(f"{rel}!<pdf stream {k}>: denylist [{gx.mask(m.group(0))}]")
        return out
    return [f"{w}: denylist [{gx.mask(t)}]" for w, t, _ in gx.scan_tree(Path(stage), [rel], pats, allow)]


def anonymization_check(stage: Path) -> list[str]:
    """Anonymization scan of everything staged for the package (CLAUDE.md, release 1.0.1): the GitHub-export denylist (if available
    locally) and PRIVATE_PATH. Returns the findings (texts with the place; the denied term itself is shortened)."""
    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    found = [f"{w}: absolute home path {t}" for w, t in scan_private_paths(stage, files)]
    gx = load_export_scanner()
    if gx is not None:
        npat = len(scanner_patterns(gx))
        # the export scanner decodes many views of every file (~5–10 s for a 2 MB .docx / manual) → files in parallel processes,
        # the largest first
        order = sorted(files, key=lambda r: -(stage / r).stat().st_size)
        import concurrent.futures as cf
        import os
        try:                                        # FEA_BUILD_JOBS limits the processes like it limits the compilations of build_ccx.sh
            cap = max(1, int(os.environ.get("FEA_BUILD_JOBS", "12")))
        except ValueError:
            cap = 12
        workers = max(1, min(len(order), (os.cpu_count() or 2) - 1, cap))
        with tempfile.TemporaryDirectory() as tmp:
            scrubbed = Path(tmp)                    # the scanner tests, their synthetic fixtures replaced by the token (the staged files stay as they are)
            roots = []
            for rel in order:
                if rel in SYNTHETIC_HOME_PATHS and (stage / rel).is_file():
                    (scrubbed / rel).parent.mkdir(parents=True, exist_ok=True)
                    (scrubbed / rel).write_bytes(scrub_synthetic_paths(rel, (stage / rel).read_bytes()))
                    roots.append(str(scrubbed))
                else:
                    roots.append(str(stage))
            with cf.ProcessPoolExecutor(workers) as ex:
                for hits in ex.map(_scan_one, roots, order, chunksize=4):
                    found += hits
        print(f"  anonymization: {len(files)} files, {npat} denylist patterns (github_export, {workers} processes), "
              f"{len(found)} findings")
        return found
    pats = simple_denylist()
    if not pats:
        print("  ! anonymization: no local denylist (.git/info/github-denylist.txt) – only the home-path check ran")
        return found
    for rel in files:
        for where, data in _payloads(scrub_synthetic_paths(rel, (stage / rel).read_bytes()), rel):
            text = _fold(data.decode("utf-8", "ignore"))
            for pat in pats:
                if pat.search(text):
                    found.append(f"{where}: denylist [{pat.pattern[:4]}…]")
    print(f"  anonymization: {len(files)} files, {len(pats)} denylist patterns (denylist file), {len(found)} findings")
    return found


def stage_files(files: list[str], stage: Path, ref: str | None) -> None:
    for n in files:
        dst = stage / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        if ref:
            data = subprocess.run(["git", "show", f"{ref}:{n}"], cwd=ROOT, capture_output=True, check=True).stdout
            dst.write_bytes(data)
        else:
            shutil.copy2(ROOT / n, dst)


def normalize_eol(stage: Path) -> None:
    for p in stage.rglob("*"):
        rel = p.relative_to(stage).as_posix()
        if p.is_file() and (p.suffix.lower() in CRLF + LF or rel in EXEC):     # executable: LF (shebang on Linux)
            raw = p.read_bytes().replace(b"\r\n", b"\n")
            if p.suffix.lower() in CRLF:
                raw = raw.replace(b"\n", b"\r\n")
            p.write_bytes(raw)


def has_results(fea: Path) -> bool:
    try:
        with zipfile.ZipFile(fea) as z:
            return any(n.startswith("results") for n in z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def make_examples(stage: Path, with_results: bool = True) -> None:
    """Examples `examples/*.fea`: with results (default; ~85 MB packed – the example opens finished at once) are taken from the working
    tree where `examples.create --solve` created them; an example without results in it = a warning (the guide computes it).
    Without results (`--examples-no-results`): they are created again without results (~0.2 MB).
    """
    ex = stage / "examples"
    if with_results:
        for f in sorted((ROOT / "examples").glob("example*.fea")):
            shutil.copy2(f, ex / f.name)
            if not has_results(f) and f.stat().st_size > 100_000:
                print(f"  ! {f.name}: has no results in the source tree (python -m examples.create --solve --only N)")
    else:
        for cmd in (["-m", "examples.create", "--out", str(ex)], ["-m", "examples.frames", "--out", str(ex)]):
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ! examples: {' '.join(cmd)} failed: {r.stderr[-300:]}")
    fea = sorted(ex.glob("*.fea"))
    print(f"  examples: {len(fea)} × .fea, {sum(has_results(f) for f in fea)} with results ({sum(f.stat().st_size for f in fea) / 1e6:.1f} MB)")


def make_docs(stage: Path) -> None:
    try:
        import PySide6.QtWebEngineCore  # noqa: F401
    except ImportError:
        print("  ! the manual PDFs will not be created (PySide6 Addons / Qt WebEngine missing – docs module)")
        return
    r = subprocess.run([sys.executable, "-m", "fea_gui.help_pdf", "-o", str(stage / "docs")], cwd=ROOT, capture_output=True,
                       text=True)
    pdfs = list((stage / "docs").glob("*.pdf"))
    print(f"  manual PDFs: {len(pdfs)} files" + ("" if r.returncode == 0 else f" (error: {r.stderr[-200:]})"))


def make_licenses(stage: Path) -> None:
    """Texts of the third-party licences into `licenses/` (from the dist-info in .venv, EXTERNAL.md, README.md). All packages (zip,
    tar.gz, .run, setup.exe) carry them as part of the program; missing texts are warnings of the output.
    """
    try:
        warnings = collect_licenses.collect(stage / "licenses", ROOT / ".venv")
    except FileNotFoundError as e:
        raise SystemExit(f"licences: {e} (prepare the development .venv first)")
    for w in warnings:
        print(f"  ! licence: {w}")
    print(f"  licenses/: {sum(1 for _ in (stage / 'licenses' / 'python').iterdir())} packages with texts, {len(warnings)} warnings")


# a MYSTRAN program or its archive: anything under tools/mystran/, and a file named mystran / mystran*.exe|dll|so|bin|zip|7z|tar.gz anywhere. The program's
# own Python files (fea_core/mystran.py, verification/mystran_check.py, fea_core/beams/mystran_frame.py) are not matched.
MYSTRAN_BINARY = re.compile(r"(^|/)tools/mystran/|(^|/)mystran([-_ .][^/]*)?\.(exe|dll|so|bin|zip|7z|tgz|tar\.[a-z0-9]+)$|(^|/)mystran$", re.IGNORECASE)


def mystran_files(names) -> list[str]:
    """The names (relative paths with /) that are a MYSTRAN program or an archive of it."""
    return [n for n in names if MYSTRAN_BINARY.search(n.replace("\\", "/"))]


def refuse_mystran(stage: Path) -> None:
    """No package may carry MYSTRAN (its static program holds libraries – libquadmath (LGPL), SuperLU, METIS, GKlib, the GCC run-time, BLAS / LAPACK –
    whose sources and licence texts the packages do not have; the user downloads it). Called by every package writer: a staged MYSTRAN stops the build."""
    found = mystran_files(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    if found:
        raise SystemExit("MYSTRAN must not be in a package (it is not distributed – the user downloads it): " + ", ".join(sorted(found)[:10]))


def distribution_blockers(no_runtime_sources: bool, with_ccx_pastix: bool, no_windows_ccx: bool, no_linux_ccx: bool) -> list[str]:
    """Why the build must not be distributed: the options that leave out sources of binaries the packages carry (empty = a release build).
    A platform without its CalculiX (--no-windows-ccx / --no-linux-ccx) carries no binary that needs the left-out sources."""
    reasons = []
    platforms = [name for name, left_out in (("Windows", no_windows_ccx), ("Linux", no_linux_ccx)) if not left_out]
    if no_runtime_sources and platforms:
        reasons.append(f"--no-runtime-sources: licenses/sources/ of the {' and '.join(platforms)} packages lacks the sources of the GCC (and mingw-w64) "
                       "run-time libraries that are linked into or shipped with their CalculiX (libquadmath is LGPL) and of the Rust part of the solver")
    if with_ccx_pastix and not no_linux_ccx:
        reasons.append("--with-ccx-pastix: the Linux packages carry ccx_pastix without the sources and licence texts of PaStiX (LGPL-3.0, linked "
                       "statically into the GPL-2.0-only CalculiX – the licence question L4 is open), SCOTCH, hwloc and OpenBLAS")
    return reasons


def release_name(version: str, dev: bool) -> str:
    """The stem of the output files and the folder inside the archives; a build that must not be distributed cannot be mistaken for a release."""
    return f"FEAStudio-{version}{DEV_SUFFIX if dev else ''}"


DEV_SUFFIX = "-dev"


def add_all(files_root: Path, add) -> None:
    for p in sorted(files_root.rglob("*")):
        if p.is_file():
            add(p, p.relative_to(files_root).as_posix())


def write_zip(stage: Path, prefix: str, out: Path, skip: tuple = ()) -> None:
    refuse_mystran(stage)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add(p: Path, rel: str) -> None:
            if not rel.startswith(skip):
                z.write(p, f"{prefix}/{rel}")
        add_all(stage, add)


def neutral(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """A tar member without the building user: owner 0:0 without user and group names (the package and the bundled CalculiX)."""
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def write_tar(stage: Path, prefix: str, out: Path) -> None:
    refuse_mystran(stage)
    with tarfile.open(out, "w:gz", compresslevel=9) as t:
        def add(p: Path, rel: str) -> None:
            ti = neutral(t.gettarinfo(str(p), f"{prefix}/{rel}"))
            ti.mode = 0o755 if rel in EXEC else 0o644
            with open(p, "rb") as f:
                t.addfile(ti, f)
        add_all(stage, add)


LINUX_CCX = ROOT / "installer" / "linux" / "prebuilt" / "ccx_2.23_linux_x86_64.tar.gz"
# absolute paths from home folders (build on the development machine); /home/conda/ = the conda-forge build farm
# (static libgfortran, libgomp) – foreign, does not identify us
# Windows: either slash (also the doubled backslashes of JSON / C strings), any case of the drive letter and of "Users",
# with or without a separator after the user name
PRIVATE_PATH = re.compile(rb"/home/(?!conda/)[A-Za-z0-9_.-]+/|/mnt/[a-z]/"
                          rb"|(?i:[/\\]+users[/\\]+)[^/\\\x00]+[/\\]"
                          rb"|(?i:\b[a-z]:[/\\]+users[/\\]+)[A-Za-z0-9_.-]+")


# soname of the C library and its parts: on every Linux target (glibc >= 2.35, build_ccx.sh checks the symbols) – not shipped
SYSTEM_LIBS = frozenset({"libc.so.6", "libm.so.6", "libmvec.so.1", "libdl.so.2", "libpthread.so.0", "librt.so.1", "libutil.so.1",
                         "libresolv.so.2", "libanl.so.1", "ld-linux-x86-64.so.2"})


def elf_needed(data: bytes) -> list[str]:
    """DT_NEEDED names (the shared libraries) of an ELF64 little-endian file. Pure Python and bounded by the file size; ValueError for anything
    that is not a well-formed x86-64 / aarch64 ELF64 with a dynamic section."""
    if data[:4] != b"\x7fELF" or len(data) < 64 or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian file")
    try:
        phoff, = struct.unpack_from("<Q", data, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
        loads, dyn = [], None
        for i in range(phnum):
            p_type, _flags, off, vaddr, _paddr, filesz, _memsz, _align = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize)
            if p_type == 1:
                loads.append((vaddr, off, filesz))
            elif p_type == 2:
                dyn = (off, filesz)

        def at(vaddr: int) -> int:                       # virtual address -> file offset
            for va, off, size in loads:
                if va <= vaddr < va + size:
                    return off + vaddr - va
            raise ValueError("address outside the loadable segments")
        if dyn is None:
            return []
        entries = []
        for off in range(dyn[0], dyn[0] + dyn[1], 16):
            tag, val = struct.unpack_from("<qQ", data, off)
            if tag == 0:
                break
            entries.append((tag, val))
        strtab = next((v for tag, v in entries if tag == 5), None)           # DT_STRTAB
        needed = [v for tag, v in entries if tag == 1]                        # DT_NEEDED
        if needed and strtab is None:
            raise ValueError("DT_NEEDED without a string table")
        base = at(strtab) if needed else 0
        names = []
        for v in needed:
            s = base + v
            names.append(data[s:data.index(b"\0", s)].decode("ascii"))
        return names
    except (struct.error, StopIteration, UnicodeDecodeError) as e:
        raise ValueError(f"damaged ELF file ({e})") from None


def linux_ccx_plan(files: dict[str, bytes], with_pastix: bool = False) -> list[str]:
    """Members of the prebuilt Linux CalculiX archive that go into the package: the binaries of the mode (`ccx_spooles`; with `with_pastix`
    also `ccx_pastix`) and the shared libraries in `lib/` that they need, followed through DT_NEEDED (glibc is not shipped). An error when a
    binary of the mode is missing or needs a library that the archive does not carry (an archive built by an older build_ccx.sh, or by a build
    that failed half-way). `files` = {member name: bytes} of regular files.
    """
    binaries = ["ccx_spooles"] + (["ccx_pastix"] if with_pastix else [])
    for b in binaries:
        if b not in files:
            hint = "rebuild with installer/linux/build_ccx.sh (FEA_CCX_PASTIX=1, the default there) and pack the archive again" if b == "ccx_pastix" \
                else "rebuild with installer/linux/build_ccx.sh"
            raise SystemExit(f"{b} is missing in the archive of the prebuilt CalculiX – {hint}, or use --no-linux-ccx")
    keep, queue = set(binaries), list(binaries)
    while queue:
        name = queue.pop()
        try:
            needed = elf_needed(files[name])
        except ValueError as e:
            raise SystemExit(f"{name} in the archive of the prebuilt CalculiX: {e}") from None
        for lib in needed:
            if lib in SYSTEM_LIBS:
                continue
            member = f"lib/{lib}"
            if member not in files:
                raise SystemExit(f"{name} needs {lib}, which the archive does not carry in lib/ – an archive of an older build_ccx.sh? "
                                 "rebuild with installer/linux/build_ccx.sh, or use --no-linux-ccx")
            if member not in keep:
                keep.add(member)
                queue.append(member)
    return sorted(keep)


def linux_ccx(src: Path, dst: Path, with_pastix: bool = False) -> None:
    """Prebuilt CalculiX for Linux (installer/linux/build_ccx.sh … ARCHIVE) into the package – only what `linux_ccx_plan` selects for the
    mode (default: ccx_spooles and its libraries; with_pastix: also ccx_pastix and its libraries – the licence question L4 is open, a package
    made so must not be published). A binary with a path from a home folder = an error (build again: build_ccx.sh maps it to /build); an
    archive with other members or with an owner other than 0:0 (packed by hand) is repacked – contents, permissions and times unchanged,
    gzip without name and time.
    """
    with tarfile.open(src, "r:gz") as t:
        everything = t.getmembers()
        files = {m.name.lstrip("./"): t.extractfile(m).read() for m in everything if m.isfile()}
        wanted = set(linux_ccx_plan(files, with_pastix))
        has_lib = any(n.startswith("lib/") for n in wanted)
        members = [m for m in everything if m.name.lstrip("./") in wanted or (m.isdir() and m.name.lstrip("./") == "lib" and has_lib)]
        dropped = [m.name for m in everything if m not in members]
        bad = [m.name for m in members if m.isfile() and PRIVATE_PATH.search(files[m.name.lstrip("./")])]
        if bad:
            raise SystemExit(f"{src.name}: {', '.join(bad)} contains an absolute path from the home folder – rebuild with "
                             "installer/linux/build_ccx.sh OUTPUT WORKDIR ARCHIVE, or package with --no-linux-ccx")
        owned = [m for m in members if m.uid or m.gid or m.uname or m.gname]
        if with_pastix:
            print("  ! --with-ccx-pastix: ccx_pastix (PaStiX LGPL-3.0 statically linked into the GPL-2.0-only CalculiX) goes into the package although the "
                  "licence question L4 is open and the texts / sources of PaStiX, SCOTCH, hwloc and OpenBLAS are not in licenses/ – do NOT publish this package")
        if not owned and not dropped:
            shutil.copy2(src, dst)
            return
        if dropped:
            print(f"  {src.name}: not distributed → left out: {', '.join(dropped)}")
        if owned:
            print(f"  {src.name}: {len(owned)} members owned by the builder → repacked with owner 0:0")
        with open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz, \
                tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as out:
            for m in members:
                out.addfile(neutral(m), t.extractfile(m) if m.isfile() else None)


def require_linux_ccx(path: Path = None, with_pastix: bool = False) -> Path:
    """The prebuilt Linux CalculiX is MANDATORY for a Linux package: the installer never compiles (F13 step 3), so a package without it
    would leave a Linux user with no solver. Checked BEFORE anything is staged (a missing / unusable archive stops the build at once, not
    after ten minutes of work and not on the user's machine). Returns the archive; `--no-linux-ccx` is the explicit opt-out."""
    path = LINUX_CCX if path is None else path
    if not path.is_file():
        shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        raise SystemExit(
            f"The prebuilt CalculiX for Linux is missing: {shown}.\n"
            "The Linux installer does not compile anything, so a Linux package needs it. Build it at release with\n"
            "    bash installer/linux/build_ccx.sh OUTPUT_DIR WORKDIR installer/linux/prebuilt/ccx_2.23_linux_x86_64.tar.gz\n"
            "(WSL2 / a Linux machine), or build only the Windows side with --no-linux-ccx (the Linux package is then not complete).")
    with tempfile.TemporaryDirectory(prefix="fea_lccx_") as td:
        linux_ccx(path, Path(td) / path.name, with_pastix)   # the same validation the packaging does: ccx_spooles present, no home paths
    return path


sha = upgrade.sha256_file                           # the single SHA-256 implementation (installer/upgrade.py)


def write_sums(path: Path, files: list[Path]) -> Path:
    """SHA256SUMS.txt in the sha256sum format (digest, 2 spaces, name) with LF line ends – `sha256sum -c` on Linux would
    look for the file "name\\\\r" with CRLF (writing text on Windows would otherwise give CRLF).
    """
    path.write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), encoding="utf-8", newline="\n")
    return path


# the switches of `fea_setup.py install / update` without a value (the .run passes them on; tests/test_install_lows.py keeps the list equal
# to the real ones) – anything else is refused BEFORE the package is extracted over the target
RUN_SETUP_FLAGS = ("--quiet", "--yes", "-y", "--exact", "--force", "--no-slim", "--no-shortcuts", "--no-mcp", "--no-fusion", "--no-check",
                   "--quick-check", "--keep-downloads", "--no-hashes", "--msmpi")

RUN_HEADER = r'''#!/bin/sh
# FEA Studio @VER@ - samorozbalovaci instalator / self-extracting installer for Linux x86_64 (glibc 2.35+).
#   sh FEAStudio-@VER@-linux-x86_64.run [--target FOLDER] [--update | --install] [--extract-only] [options of installer/fea_setup.py]
# Default target: XDG_DATA_HOME/fea-studio (usually .local/share/fea-studio in the home folder). An installation in the target is updated
# (fea_setup.py update: migrations, obsolete files, modules as last time), otherwise it is installed.
# Further options go to installer/fea_setup.py, e.g. --profile standard --quiet --no-shortcuts --lang en, or names of modules.
# Navrat ke starsi verzi jen s --force. / A downgrade only with --force.
set -e
VERSION="@VER@"
PAYLOAD_SHA256="@SHA@"
target="${XDG_DATA_HOME:-${HOME}/.local/share}/fea-studio"
mode=""
extract_only=0
# the language of the messages: --lang, otherwise FEA_LANG / the locale
lang="${FEA_LANG:-${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}}"
prev=""
for a in "$@"; do
    case "$a" in --lang=*) lang="${a#--lang=}" ;; esac
    if [ "$prev" = "--lang" ]; then lang="$a"; fi
    prev="$a"
done
msg() { case "$lang" in cs*|Czech*) echo "$1" ;; *) echo "$2" ;; esac; }
# the options of the .run itself are taken out wherever they stand; the others stay in "$@" for fea_setup.py
n=$#
i=0
while [ "$i" -lt "$n" ]; do
    a="$1"; shift; i=$((i + 1))
    case "$a" in
        --target)
            [ "$i" -lt "$n" ] || { msg "--target: chybi slozka" "--target: the folder is missing" >&2; exit 2; }
            target="$1"; shift; i=$((i + 1)) ;;
        --target=*) target="${a#--target=}" ;;
        --update) mode=update ;;
        --install) mode=install ;;
        --extract-only|--noexec) extract_only=1 ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        --) ;;
        *) set -- "$@" "$a" ;;
    esac
done
# refuse an unknown option or module now, before anything is extracted over an installation
want=""
for a in "$@"; do
    if [ -n "$want" ]; then
        case "$want:$a" in
            --lang:cs|--lang:en|--profile:minimal|--profile:standard|--profile:full) want=""; continue ;;
        esac
        msg "CHYBA: neplatna hodnota volby $want: $a" "ERROR: invalid value of $want: $a" >&2; exit 2
    fi
    case "$a" in
        --lang|--profile) want="$a" ;;
        --lang=cs|--lang=en|--profile=minimal|--profile=standard|--profile=full) ;;
        @FLAGS@) ;;
        -*) msg "CHYBA: neznama volba $a (napoveda: sh $0 --help)" "ERROR: unknown option $a (help: sh $0 --help)" >&2; exit 2 ;;
        @MODULES@) ;;
        *) msg "CHYBA: neznamy modul $a (moduly: @MODULE_LIST@)" "ERROR: unknown module $a (modules: @MODULE_LIST@)" >&2; exit 2 ;;
    esac
done
[ -z "$want" ] || { msg "CHYBA: chybi hodnota volby $want" "ERROR: the value of $want is missing" >&2; exit 2; }
case "$target" in /*) ;; *) target="$PWD/$target" ;; esac
force=0
for a in "$@"; do [ "$a" = "--force" ] && force=1; done
line=$(awk '/^__FEA_STUDIO_PAYLOAD__$/ { print NR + 1; exit }' "$0")
if command -v sha256sum >/dev/null 2>&1; then
    got=$(tail -n +"$line" "$0" | sha256sum | cut -d ' ' -f 1)
elif command -v shasum >/dev/null 2>&1; then
    got=$(tail -n +"$line" "$0" | shasum -a 256 | cut -d ' ' -f 1)
else
    msg "CHYBA: chybi sha256sum i shasum - nelze overit neporusenost instalatoru" "ERROR: neither sha256sum nor shasum - the installer cannot be verified" >&2; exit 1
fi
if [ "$got" != "$PAYLOAD_SHA256" ]; then msg "CHYBA: instalator je poskozeny (kontrolni soucet nesedi)" "ERROR: the installer is damaged (the checksum does not match)" >&2; exit 1; fi
state="$target/.venv/fea_modules.json"
if [ -f "$state" ]; then
    old=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$state" | head -n 1)
    [ -n "$old" ] || old="1.0.0"
    # a pre-release build of this release (the state says 1.5.0, the release is 1.4.2) is the same version, not a newer one - unless this installer is 1.5.0 or newer
    if [ "$old" = "1.5.0" ] && [ "$(printf '%s\n%s\n' "$VERSION" 1.5.0 | sort -V | head -n 1)" = "$VERSION" ] && [ "$VERSION" != "1.5.0" ]; then old="1.4.2"; fi
    newest=$(printf '%s\n%s\n' "$old" "$VERSION" | sort -V | tail -n 1)
    if [ "$old" != "$VERSION" ] && [ "$newest" = "$old" ] && [ "$force" = 0 ]; then
        msg "CHYBA: v $target je novejsi verze $old nez tento instalator ($VERSION); navrat ke starsi verzi jen s --force" "ERROR: $target has the newer version $old than this installer ($VERSION); a downgrade only with --force" >&2
        exit 2
    fi
    [ -n "$mode" ] || mode=update
fi
[ -n "$mode" ] || mode=install
mkdir -p "$target"
echo "FEA Studio $VERSION -> $target ($mode)"
tail -n +"$line" "$0" | tar -xzf - -C "$target" --strip-components=1 --no-same-owner
if [ "$extract_only" = 1 ]; then
    msg "Rozbaleno do $target; instalace: sh \"$target/install.sh\" $mode" "Extracted into $target; to install: sh \"$target/install.sh\" $mode"
    exit 0
fi
cd "$target"
exec sh ./install.sh "$mode" "$@"
__FEA_STUDIO_PAYLOAD__
'''


def run_header(ver: str, payload_sha: str) -> str:
    """The sh header of the self-extracting installer; the module names it accepts come from fea_core/modules.json (retired ones too)."""
    raw = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))
    names = sorted(set(raw["modules"]) | set(raw.get("retired", {})))
    for n in names:
        if not re.fullmatch(r"[a-z0-9_-]+", n):
            raise SystemExit(f"module name {n!r} cannot go into the shell header of the .run")
    return (RUN_HEADER.replace("@VER@", ver).replace("@SHA@", payload_sha).replace("@FLAGS@", "|".join(RUN_SETUP_FLAGS))
            .replace("@MODULES@", "|".join(names)).replace("@MODULE_LIST@", " ".join(sorted(raw["modules"]))))


def write_run(tar_path: Path, out: Path, ver: str) -> Path:
    """Self-extracting installer for Linux: sh header (ASCII, LF) + the data .tar.gz (makeself style, without dependencies)."""
    head = run_header(ver, sha(tar_path)).encode("ascii")
    with open(out, "wb") as f, open(tar_path, "rb") as t:
        f.write(head)
        shutil.copyfileobj(t, f, 1 << 20)
    out.chmod(0o755)
    return out


def bundle_python(stage: Path) -> Path | None:
    """Python 3.11 for Windows unpacked into tools/python/<PY_DIR> (zip and setup.exe): uv finds it during the installation as a managed
    Python, so nothing is downloaded or extracted through tools/python/.temp (which fails with some antiviruses).
    """
    import fea_setup
    import slim
    plat = "windows-x86_64"
    target = stage / "tools" / "python" / fea_setup.PY_DIR[plat]
    try:
        arch = fea_setup.download(fea_setup.PY_URL[plat], ROOT / "tools" / "_download" / f"cpython-{fea_setup.PY_FULL}-{plat}.tar.gz",
                                  f"Python {fea_setup.PY_FULL}", fea_setup.PY_SHA256[plat])
        tmp = stage / "tools" / "python" / "_extract"
        shutil.rmtree(tmp, ignore_errors=True)
        fea_setup.extract_archive(arch, tmp)
        for i in range(30):                      # synchronisation of the folder / antivirus briefly holds freshly extracted files
            try:
                (tmp / "python").replace(target)
                break
            except PermissionError:
                if i == 29:
                    raise
                time.sleep(1)
        shutil.rmtree(tmp, ignore_errors=True)
    except (OSError, RuntimeError, tarfile.TarError) as e:
        # a package without Python would depend on a download through uv (antiviruses) on clean Windows – better not to release it
        shutil.rmtree(stage / "tools" / "python", ignore_errors=True)
        raise SystemExit(f"Python {fea_setup.PY_FULL} could not be bundled ({e}); run again, "
                         "or use --no-python explicitly")
    slim.prune_python_home(target)
    print(f"  Python {fea_setup.PY_FULL} bundled: {slim.dir_size(target) / 1e6:.0f} MB")
    return target


WIN_CCX = ROOT / "installer" / "windows" / "prebuilt" / "ccx_2.23_windows_x86_64.zip"
WIN_CCX_MEMBERS = ("ccx_spooles.exe", "manifest.json")      # the members of the archive made by installer/windows/build_ccx.ps1 – nothing else
# DLLs that a Windows 10+ always has: the executable of CalculiX must import nothing else (no mingw / GCC run-time DLL – libgcc, libgfortran, libgomp,
# libquadmath, winpthreads – and no BLAS DLL: everything is linked statically, the BLAS / LAPACK are the reference ones)
WIN_SYSTEM_DLLS = frozenset({"kernel32.dll", "ntdll.dll", "advapi32.dll", "user32.dll", "ucrtbase.dll", "msvcrt.dll", "shell32.dll", "bcrypt.dll",
                             "ws2_32.dll", "ole32.dll", "oleaut32.dll", "gdi32.dll", "synchronization.dll"})


def pe_imports(data: bytes) -> list[str]:
    """Names of the DLLs that a PE32+ (64-bit) executable imports (the import directory), lower case. Pure Python and bounded by the size of the
    file; ValueError for anything that is not a well-formed PE32+ executable."""
    try:
        if data[:2] != b"MZ":
            raise ValueError("not a PE file")
        pe, = struct.unpack_from("<I", data, 0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("no PE signature")
        nsec, = struct.unpack_from("<H", data, pe + 6)
        optsize, = struct.unpack_from("<H", data, pe + 20)
        opt = pe + 24
        if struct.unpack_from("<H", data, opt)[0] != 0x20B:
            raise ValueError("not a 64-bit executable (PE32+)")
        import_rva, = struct.unpack_from("<I", data, opt + 120)         # data directory 1 = import table
        sections = []
        for k in range(nsec):
            name, vsize, va, rawsize, rawptr = struct.unpack_from("<8sIIII", data, opt + optsize + 40 * k)
            sections.append((va, max(vsize, rawsize), rawptr))

        def at(rva: int) -> int:
            for va, size, raw in sections:
                if va <= rva < va + size:
                    return raw + rva - va
            raise ValueError("address outside the sections")
        names = []
        if import_rva:
            off = at(import_rva)
            for _ in range(512):
                oft, _t, _f, name_rva, _ft = struct.unpack_from("<IIIII", data, off)
                if name_rva == 0:
                    break
                n = at(name_rva)
                names.append(data[n:data.index(b"\0", n)].decode("ascii").lower())
                off += 20
        return names
    except (struct.error, IndexError, UnicodeDecodeError) as e:
        raise ValueError(f"damaged PE file ({e})") from None


# what the manifest.json of the build must say about the checkout the package is made from: key → the file whose SHA-256 it holds
WIN_CCX_BUILD_FILES = {"build_script_sha256": "installer/windows/build_ccx.sh", "toolchain_lock_sha256": "installer/windows/ccx_toolchain.lock",
                       # the faer solver (F17 stage C): the Rust toolchain lock, the crates, the glue (a dotted key = a key of the "faer" object of the manifest)
                       "faer.rust_lock_sha256": "installer/windows/ccx_rust.lock", "faer.cargo_lock_sha256": "installer/faer_ccx/Cargo.lock",
                       "faer.faer_c_sha256": "installer/faer_ccx/ccx/faer.c", "faer.faer_h_sha256": "installer/faer_ccx/ccx/faer.h"}


def checkout_bytes(rel: str, ref: str | None = None) -> bytes | None:
    """Content of a file of the checkout the package is made from: the working tree (ref None) or a git ref. None = it does not exist."""
    if ref is None:
        f = ROOT / rel
        return f.read_bytes() if f.is_file() else None
    r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _dotted(meta: dict, key: str):
    """meta["a"]["b"] for the key "a.b" (None when a level is missing)."""
    for part in key.split("."):
        meta = meta.get(part) if isinstance(meta, dict) else None
    return meta


def windows_ccx(src: Path, dst: Path, ref: str | None = None) -> None:
    """Prebuilt Windows CalculiX (installer/windows/build_ccx.ps1 → ARCHIVE) into the package: exactly the members `WIN_CCX_MEMBERS`; the
    executable must be a 64-bit PE that imports only system DLLs (everything else is linked statically) and must not carry an absolute path from a
    home folder (the build maps the work folder to /build). Its manifest must describe THIS executable (binary_sha256) and the build scripts of the
    checkout the package is made from (build_script_sha256 = build_ccx.sh, toolchain_lock_sha256 = ccx_toolchain.lock): the sources and scripts that the
    package ships are then the ones the binary was built with. The archive is copied unchanged (it was made with fixed time stamps)."""
    with zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if sorted(names) != sorted(WIN_CCX_MEMBERS):
            raise SystemExit(f"{src.name}: members {sorted(names)} instead of {sorted(WIN_CCX_MEMBERS)} – rebuild with installer/windows/build_ccx.ps1")
        exe = z.read("ccx_spooles.exe")
        manifest = z.read("manifest.json")
    try:
        imports = pe_imports(exe)
    except ValueError as e:
        raise SystemExit(f"{src.name}: ccx_spooles.exe: {e}") from None
    foreign = [d for d in imports if not (d in WIN_SYSTEM_DLLS or d.startswith("api-ms-win-"))]
    if foreign:
        raise SystemExit(f"{src.name}: ccx_spooles.exe imports {', '.join(foreign)} – not system DLLs; the build must link them statically")
    bad = [n for n, b in (("ccx_spooles.exe", exe), ("manifest.json", manifest)) if PRIVATE_PATH.search(b)]
    if bad:
        raise SystemExit(f"{src.name}: {', '.join(bad)} contains an absolute path from a user folder – rebuild with installer/windows/build_ccx.ps1")
    try:
        meta = json.loads(manifest)
    except ValueError as e:
        raise SystemExit(f"{src.name}: manifest.json is not JSON ({e})") from None
    problems = []
    if not isinstance(meta, dict):
        problems.append("it is not a JSON object")
    else:
        if meta.get("binary_sha256") != hashlib.sha256(exe).hexdigest():
            problems.append("binary_sha256 is not the SHA-256 of ccx_spooles.exe in the archive")
        for key, rel in WIN_CCX_BUILD_FILES.items():
            data = checkout_bytes(rel, ref)
            if data is None:
                problems.append(f"{rel} does not exist in {ref or 'the working tree'}")
            elif _dotted(meta, key) != hashlib.sha256(data).hexdigest():
                problems.append(f"{key} differs from {rel} of {ref or 'the working tree'} – the binary was built with other build scripts than the ones the package ships")
    if problems:
        raise SystemExit(f"{src.name}: manifest.json does not match: " + "; ".join(problems) + " – rebuild with installer/windows/build_ccx.ps1")
    shutil.copy2(src, dst)


def require_windows_ccx(path: Path = None, ref: str | None = None) -> Path:
    """The prebuilt Windows CalculiX is MANDATORY for a Windows package: the installer neither downloads nor compiles it (the third-party
    ccx_static.exe is gone), so a package without it would leave the user with no solver. Checked BEFORE anything is staged; `--no-windows-ccx`
    is the explicit opt-out."""
    path = WIN_CCX if path is None else path
    if not path.is_file():
        shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        raise SystemExit(
            f"The prebuilt CalculiX for Windows is missing: {shown}.\n"
            "The Windows installer neither downloads nor compiles CalculiX, so a Windows package needs it. Build it at release with\n"
            "    powershell -ExecutionPolicy Bypass -File installer\\windows\\build_ccx.ps1\n"
            "(30-60 minutes, pinned sources and toolchain), or build without it with --no-windows-ccx (the package is then not complete).")
    with tempfile.TemporaryDirectory(prefix="fea_wccx_") as td:
        windows_ccx(path, Path(td) / path.name, ref)
    return path


def bundle_windows_tools(stage: Path, with_ccx: bool = True, ref: str | None = None) -> list[Path]:
    """uv.exe and the prebuilt CalculiX (`ccx_2.23_windows_x86_64.zip`, built by installer/windows/build_ccx.ps1) into the packages for Windows:
    the installation then downloads no executable files for the base program (antiviruses and corporate web filters block downloads of .exe
    files / of little-known servers) and unpacks CalculiX from the archive (`fea_setup.install_ext_ccx`)."""
    made = []
    uv = ROOT / "tools" / "uv" / "uv.exe"
    if uv.exists():
        (stage / "tools" / "uv").mkdir(parents=True, exist_ok=True)
        shutil.copy2(uv, stage / "tools" / "uv" / "uv.exe")
        made.append(stage / "tools" / "uv")
    else:
        print("  ! tools/uv/uv.exe is missing – an installation from the zip downloads uv")
    if with_ccx:
        require_windows_ccx(ref=ref)
        (stage / "tools" / "calculix").mkdir(parents=True, exist_ok=True)
        windows_ccx(WIN_CCX, stage / "tools" / "calculix" / WIN_CCX.name, ref)
        made.append(stage / "tools" / "calculix")
        print(f"  CalculiX bundled: {WIN_CCX.name} ({WIN_CCX.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  ! --no-windows-ccx: the Windows package has NO CalculiX (the installer neither downloads nor compiles it) – do not release it")
    return made


WIN_ARCHIVE = ROOT / "tools" / "winccx_build" / "archive"      # the local archive of the downloads of the Windows CalculiX build (installer/windows/fetch_ccx_archive.py)


def source_archive(name: str, urls: list[str], digest: str) -> Path:
    """A pinned source archive: the file of the local archive of the Windows build when its SHA-256 matches, otherwise downloaded once into
    tools/_download/sources (digest checked; the URLs of the pin one after the other). A missing archive = an error (the GPL binaries must not ship
    without their sources)."""
    import fea_setup
    local = WIN_ARCHIVE / name
    if local.is_file() and sha(local) == digest:
        return local
    errors = []
    for url in urls:
        try:
            return fea_setup.download(url, ROOT / "tools" / "_download" / "sources" / name, name, digest)
        except (OSError, RuntimeError) as e:
            errors.append(str(e))
    raise SystemExit(f"source archive {name} could not be obtained ({'; '.join(errors)}) – the GPL binaries must not ship without it")


def tar_program() -> str:
    """A tar that reads .tar.zst: the one of Windows 10+ (bsdtar), otherwise the first in PATH."""
    import os
    system = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "tar.exe"
    return str(system) if system.is_file() else (shutil.which("tar") or "tar")


def extract_recipe(archive: Path, dst: Path) -> list[Path]:
    """PKGBUILD and .SRCINFO of an MSYS2 source package (`archive`) unchanged into `dst/<archive name without .src.tar.zst>/`."""
    folder = dst / collect_licenses.recipe_folder(archive.name)
    m = re.match(r"(mingw-w64-[^-]+)-\d", archive.name)
    if not m:
        raise SystemExit(f"{archive.name}: not a mingw-w64 source package")
    top = m.group(1)
    with tempfile.TemporaryDirectory(prefix="fea_recipe_") as td:
        r = subprocess.run([tar_program(), "-xf", str(archive), "-C", td, f"{top}/PKGBUILD", f"{top}/.SRCINFO"], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"{archive.name}: the recipe could not be extracted ({r.stderr.decode(errors='replace').strip()[:300]})")
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for n in ("PKGBUILD", ".SRCINFO"):
            shutil.copyfile(Path(td) / top / n, folder / n)
            out.append(folder / n)
    return out


# the patches of the recipe that conda-forge applies to the GCC sources of a linux-64 build (recipe/parent/meta.yaml; the mingw and macOS ones are not in the package)
GCC_RECIPE_LINUX_PATCHES = 4


def extract_gcc_recipe(dst: Path) -> list[Path]:
    """The conda-forge recipe of the GCC libraries of the Linux CalculiX (info/recipe/ of the pinned conda package that installer/linux/ccx_gccsrc.lock names),
    unchanged into `dst/gcc-<version>-conda-forge-recipe/`. The package is checked against its SHA-256 of ccx_toolchain.lock; the recipe must build exactly the GCC
    archive that the package carries (its SHA-256 in parent/meta.yaml) with its Linux patches – otherwise the sources would not be those of the binaries."""
    gcc = collect_licenses.linux_gcc_source(ROOT)
    name, digest, url = collect_licenses.linux_gcc_recipe_package(ROOT)
    package = source_archive(name, [url], digest)
    folder = dst / collect_licenses.linux_gcc_recipe_folder(gcc["version"])
    with tempfile.TemporaryDirectory(prefix="fea_gccrecipe_") as td:
        with zipfile.ZipFile(package) as z:                          # a .conda file is a zip of two tar.zst: info-*.tar.zst holds the recipe
            infos = [n for n in z.namelist() if re.fullmatch(r"info-[A-Za-z0-9._+-]+\.tar\.zst", n)]
            if len(infos) != 1:
                raise SystemExit(f"{name}: {len(infos)} info-*.tar.zst members – not a conda package")
            inner = Path(z.extract(infos[0], td))
        tree = Path(td) / "x"
        tree.mkdir()
        # the whole (small) tar: GNU tar does not take a folder name as a member when the archive has no entry for the folder
        r = subprocess.run([tar_program(), "-xf", str(inner), "-C", str(tree)], capture_output=True)
        recipe = tree / "info" / "recipe"
        if r.returncode != 0 or not recipe.is_dir():
            raise SystemExit(f"{name}: info/recipe could not be extracted ({r.stderr.decode(errors='replace').strip()[:300]}) – tar must read .tar.zst "
                             "(Windows: the tar of Windows 10+; Linux: GNU tar with zstd installed)")
        meta = recipe / "parent" / "meta.yaml"
        text = meta.read_text(encoding="utf-8") if meta.is_file() else ""
        wanted = re.compile(rf'sha256:\s*{gcc["sha256"]}\s*#\s*\[gcc_version == "{re.escape(gcc["version"])}"\]')
        if not wanted.search(text):
            raise SystemExit(f"{name}: the recipe does not build gcc-{gcc['version']} from the archive with the SHA-256 of installer/linux/ccx_gccsrc.lock")
        patches = [p for p in (recipe / "parent" / "patches").glob("*.patch") if p.is_file()]
        if len(patches) != GCC_RECIPE_LINUX_PATCHES or not all(f"patches/{p.name}" in text for p in patches):
            raise SystemExit(f"{name}: the recipe has {len(patches)} patches ({GCC_RECIPE_LINUX_PATCHES} Linux patches named in parent/meta.yaml expected)")
        made = []
        for src in sorted(recipe.rglob("*")):
            if src.is_file() and not src.is_symlink():
                out = folder / src.relative_to(recipe)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, out)
                made.append(out)
    return made


def add_sources(stage: Path, kind: str, runtime: bool = True) -> list[Path]:
    """Source archives of the shipped GPL / LGPL binaries into `licenses/sources/` (GPL-2 §3(a) – the source travels with the binary;
    list and digests in collect_licenses.SOURCE_ARCHIVES and the pins of the build scripts, described in licenses/SOURCES.md). `kind` = "all" (every package:
    CalculiX for ccx_spooles.exe / ccx_spooles), "windows" (the sources of the Windows CalculiX: SPOOLES, ARPACK-ng, LAPACK, and – with `runtime` – the Rust part and
    the MSYS2 source packages of the GCC / mingw-w64 run-time libraries that are linked statically into it, libquadmath among them (LGPL-2.1): the GCC and CRT
    packages whole, of the two other mingw-w64 packages only the recipe, they are the same git clone), "linux" (the sources of the Linux CalculiX: SPOOLES, ARPACK-ng,
    LAPACK, and – with `runtime` – the Rust part and the GCC source of the pinned conda toolchain with the conda-forge recipe: libgfortran, libgcc, and the shipped
    libgomp and libquadmath) or "offline" (only the offline setup: Gmsh, whose wheel it carries).
    Taken from the local archive of the Windows build (tools/winccx_build/archive) or downloaded once into tools/_download/sources (digest checked);
    a missing archive = an error (do not release without it).
    """
    dst = stage / "licenses" / "sources"
    dst.mkdir(parents=True, exist_ok=True)
    made = []
    wanted = dict(collect_licenses.SOURCE_ARCHIVES)
    if kind == "windows":                                              # the pinned sources of installer/windows/build_ccx.sh (except the shared CalculiX)
        wanted.update({n: {"url": urls[0], "urls": urls, "sha256": d, "packages": "windows"}
                       for n, d, urls in collect_licenses.build_ccx_win_sources(ROOT) if n not in wanted})
    elif kind == "linux":                                              # installer/linux/build_ccx.sh (except the shared CalculiX) and the GCC source of the toolchain
        wanted.update({n: {"url": urls[0], "urls": urls, "sha256": d, "packages": "linux"} for n, d, urls in collect_licenses.linux_sources(ROOT)})
    for name, s in wanted.items():
        if s["packages"] != kind:
            continue
        if (collect_licenses.is_runtime_source(name) or collect_licenses.is_rust_source(name) or collect_licenses.is_linux_gcc_source(name)) and not runtime:
            continue
        arch = source_archive(name, s.get("urls") or [s["url"]], s["sha256"])
        if collect_licenses.is_recipe_only(name):
            made += extract_recipe(arch, dst)
            continue
        (dst / name).parent.mkdir(parents=True, exist_ok=True)               # the crates of the faer library are in licenses/sources/crates/
        shutil.copy2(arch, dst / name)
        made.append(dst / name)
    if kind == "linux" and runtime and collect_licenses.linux_gcc_source(ROOT):
        made += extract_gcc_recipe(dst)                                # after the GCC archive: the recipe that builds the shipped libraries from it
    if kind == "windows" and not runtime:
        print("  ! --no-runtime-sources: the Windows package has NO source of the GCC / mingw-w64 run-time libraries (libquadmath is LGPL) – do not release it")
    if kind == "linux" and not runtime:
        print("  ! --no-runtime-sources: the Linux packages have NO source of the GCC run-time libraries (libgomp, libquadmath are shipped) – do not release them")
    shown = list(dict.fromkeys(p.relative_to(dst).parts[0] for p in made))       # a recipe folder / the crates folder once
    print(f"  licenses/sources ({kind}): {', '.join(shown) or '–'} ({len(made)} files, {sum(p.stat().st_size for p in made) / 1e6:.1f} MB)")
    return made


def remove_sources(stage: Path, made: list[Path]) -> None:
    """The files that add_sources put into licenses/sources/ (and the folders that become empty: crates/, the recipe folders)."""
    top = stage / "licenses" / "sources"
    for f in made:
        f.unlink(missing_ok=True)
        for d in f.parents:
            if d == top or top not in d.parents or not d.is_dir() or any(d.iterdir()):
                break
            d.rmdir()


def slim_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def offline_requirements() -> list[str]:
    """Exact versions of the packages of the standard profile for Windows (the same procedure as fea_setup.sync_packages)."""
    import fea_setup
    keys = fea_setup.closure(fea_setup.PROFILES["standard"])
    reqs = sorted({r for k in keys for r in fea_setup.MODS[k].get("pip", [])})
    nodeps = sorted({r for k in keys for r in fea_setup.MODS[k].get("pip_nodeps", [])})
    lock = upgrade.parse_lock((ROOT / "installer" / "locks" / "windows-x86_64.txt").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="fea_off_") as td:
        rin, out, cons = Path(td) / "req.in", Path(td) / "resolved.txt", Path(td) / "constraints.txt"
        rin.write_text("\n".join(reqs) + "\n", encoding="utf-8")
        cons.write_text("\n".join(req for req, _ in lock.values()) + "\n", encoding="utf-8")     # without digests
        subprocess.run([str(fea_setup.ensure_uv()), "pip", "compile", str(rin), "-c", str(cons), "--python-platform", "x86_64-pc-windows-msvc", "--python-version", "3.11",
                        "--no-header", "--no-annotate", "--no-config", "-q", "-o", str(out)], check=True)
        lines = [l.strip() for l in out.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return sorted(set(lines + nodeps), key=str.lower)


def fetch_wheels(reqs: list[str], dest: Path) -> list[Path]:
    """Wheels for Python 3.11 / win_amd64 from PyPI with the SHA-256 digest from PyPI (JSON API) into the cache `dest`. A package
    without a wheel = an error (an installation without internet could not build it). Returns the files of this set.
    """
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise SystemExit("--offline: run make_package.py with Python 3.11 for Windows (development .venv)")
    order = {t: i for i, t in enumerate(sys_tags())}
    dest.mkdir(parents=True, exist_ok=True)
    got, missing = [], []
    for r in reqs:
        name, _, ver = r.partition("==")
        name, ver = name.split("[")[0].strip(), ver.split(";")[0].strip()
        req = urllib.request.Request(f"https://pypi.org/pypi/{name}/{ver}/json", headers={"User-Agent": "FEAStudio-make_package"})
        with urllib.request.urlopen(req, timeout=60) as f:
            files = json.load(f)["urls"]
        best = None
        for u in files:
            if u["packagetype"] != "bdist_wheel":
                continue
            rank = min((order.get(t, 10 ** 6) for t in parse_wheel_filename(u["filename"])[3]), default=10 ** 6)
            if rank < 10 ** 6 and (best is None or rank < best[0]):
                best = (rank, u)
        if best is None:
            missing.append(r)
            continue
        u = best[1]
        out = dest / u["filename"]
        if not (out.exists() and sha(out) == u["digests"]["sha256"]):
            with urllib.request.urlopen(urllib.request.Request(u["url"], headers={"User-Agent": "FEAStudio-make_package"}),
                                        timeout=120) as f, open(str(out) + ".part", "wb") as o:
                shutil.copyfileobj(f, o, 1 << 20)
            if sha(Path(str(out) + ".part")) != u["digests"]["sha256"]:
                raise SystemExit(f"{u['filename']}: SHA-256 digest does not match PyPI")
            Path(str(out) + ".part").replace(out)
        got.append(out)
    if missing:
        raise SystemExit("--offline: bez kola pro cp311 / win_amd64: " + ", ".join(missing))
    return got


def find_iscc(given: str) -> str:
    for c in (given, str(ROOT / "tools" / "innosetup" / "ISCC.exe"), shutil.which("ISCC.exe"), shutil.which("iscc")):
        if c and Path(c).exists():
            return c
    return ""


def iscc_files(output: str) -> list[str]:
    """The files that ISCC says it put into the setup (its "Compressing: <path>" lines), with /."""
    return [m.group(1).strip().replace("\\", "/") for m in re.finditer(r"^\s*Compressing:\s+(.+?)(?:\s+\([\d.]+\))?\s*$", output, re.MULTILINE)]


def write_setup_exe(stage: Path, out: Path, ver: str, iscc: str, offline: bool = False, dev: bool = False, log: Path | None = None) -> Path | None:
    """setup.exe (Inno Setup 6) from the unpacked package for Windows (Python, uv.exe and CalculiX are already in stage/tools);
    `offline` = also the Python packages in stage/tools/offline/wheels → FEAStudio-<version>-offline-setup.exe.
    `dev` = a build that must not be distributed: the file name gets `-dev` (the version inside stays). The compiler's list of the packed files is checked
    for MYSTRAN; `log` = also write the compiler's output there (it holds local paths – not a file of the release).
    """
    import fea_setup
    refuse_mystran(stage)
    args = [iscc, f"/DAppVersion={ver}", f"/DSourceDir={stage}", f"/O{out}", f"/DPyDir={fea_setup.PY_DIR['windows-x86_64']}"]
    if dev:
        args.append(f"/DDevSuffix={DEV_SUFFIX}")
    uv = ROOT / "tools" / "uv" / "uv.exe"
    if not (stage / "tools" / "uv" / "uv.exe").exists() and uv.exists():
        args.append(f"/DUvExe={uv}")
    if offline:
        args.append("/DOffline")
    r = subprocess.run(args + [str(ROOT / "installer" / "FEAStudio.iss")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (r.stdout or "") + (r.stderr or "")
    if log is not None:
        log.write_text(output, encoding="utf-8")
    exe = out / f"{release_name(ver, dev)}-{'offline-' if offline else ''}setup.exe"
    packed = iscc_files(output)
    if bad := mystran_files(packed):
        exe.unlink(missing_ok=True)
        raise SystemExit("MYSTRAN must not be in a package: the setup compiler packed " + ", ".join(bad[:10]))
    if packed:
        print(f"  {exe.name}: the setup compiler lists {len(packed)} packed files, no MYSTRAN program among them")
    if r.returncode != 0 or not exe.exists():
        print("\n".join(output.splitlines()[-15:]))
        print(f"  ! ISCC ended with code {r.returncode} – setup.exe was not created")
        return None
    return exe


def write_package_manifest(stage: Path, ver: str, build: str) -> dict:
    """installer/manifest.json of what is in `stage` NOW (program files, without tools/). Written again whenever the content changes between the
    packages: the Windows and the Linux packages carry different sources of their CalculiX in licenses/sources/ (about 200 MB each), and a file that
    no manifest lists could never be removed by a later update."""
    manifest = upgrade.manifest_of_dir(stage, ver, build=build)
    upgrade.write_manifest(stage / upgrade.PACKAGE_MANIFEST, manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="HEAD", help="git ref whose content is packed (default HEAD = the committed state)")
    ap.add_argument("--worktree", action="store_true",
                    help="pack the working tree incl. uncommitted and new (not ignored) files instead of --ref")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build although the working tree has uncommitted changes (otherwise refused)")
    ap.add_argument("--check-only", action="store_true",
                    help="only stage the files and run the anonymization scan (no packages; stage in a temporary folder)")
    ap.add_argument("--no-scan", action="store_true", help="skip the anonymization scan (NOT for released packages)")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--examples-no-results", action="store_true", help="examples without results (small package)")
    ap.add_argument("--no-linux-ccx", action="store_true",
                    help="without the prebuilt CalculiX for Linux (the Linux package is then incomplete – the installer never compiles it; "
                         "without this switch a missing prebuilt archive stops the build)")
    ap.add_argument("--no-windows-ccx", action="store_true",
                    help="without the prebuilt CalculiX for Windows (the Windows package is then incomplete – the installer neither downloads nor "
                         "compiles it; without this switch a missing prebuilt archive stops the build)")
    ap.add_argument("--no-runtime-sources", action="store_true",
                    help="without the sources of the GCC (and mingw-w64) run-time libraries of both CalculiX programs and without the Rust sources (about 200 MB "
                         "per platform) – for a test package only: GPL-3.0 / LGPL-2.1 want the source with the binary, so such a build is named …-dev and marked "
                         "NOT FOR DISTRIBUTION")
    ap.add_argument("--with-ccx-pastix", action="store_true",
                    help="also the PaStiX build of CalculiX (ccx_pastix) in the Linux package – NOT for a published package: PaStiX (LGPL-3.0) is "
                         "linked statically into the GPL-2.0-only CalculiX, the licence question L4 is open")
    ap.add_argument("--iscc", default="", help="path to ISCC.exe (default tools/innosetup, PATH)")
    ap.add_argument("--iscc-log", default="", help="write the output of the setup compiler (the list of the packed files; local paths) to this file")
    ap.add_argument("--no-setup", action="store_true", help="without setup.exe")
    ap.add_argument("--no-run", action="store_true", help="without the self-extracting .run for Linux")
    ap.add_argument("--no-python", action="store_true", help="without the bundled Python 3.11 in the zip and setup.exe (the installation downloads it)")
    ap.add_argument("--with-pragtic", action="store_true",
                    help="include the PragTic database (PRIVATE build for own use only – the data have no licence for redistribution)")
    ap.add_argument("--offline", action="store_true",
                    help="also FEAStudio-<version>-offline-setup.exe with all Python packages of the standard profile (no internet)")
    a = ap.parse_args()
    if a.with_pragtic:
        EXCLUDE_FILES.discard(PRAGTIC)
        print("  ! PRIVATE build with the PragTic database – do not distribute")
    ver = version()
    blockers = distribution_blockers(a.no_runtime_sources, a.with_ccx_pastix, a.no_windows_ccx, a.no_linux_ccx)
    dev = bool(blockers)
    prefix = release_name(ver, dev)
    ref = None if a.worktree else a.ref
    dirty = dirty_files()
    if dirty and not a.allow_dirty:
        print(f"The working tree has {len(dirty)} uncommitted changes that are not in {ref or 'the package'}: "
              + ", ".join(dirty[:10]) + (" …" if len(dirty) > 10 else ""), file=sys.stderr)
        print("Commit them, or build with --allow-dirty (--worktree packs the working tree).", file=sys.stderr)
        return 2
    if not a.check_only and not a.no_linux_ccx:
        require_linux_ccx(with_pastix=a.with_ccx_pastix)
    if not a.check_only and not a.no_windows_ccx:
        require_windows_ccx(ref=ref)
    build = upgrade.make_build_id(ROOT, ref=ref or "HEAD") +("-dirty" if a.worktree and dirty else "")
    if a.check_only:
        out = Path(tempfile.mkdtemp(prefix="fea_pkg_check_"))
    else:
        out = Path(a.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
    stage = out / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    print(f"FEA Studio {ver} (build {build}): {'checking the package content' if a.check_only else 'building packages into ' + str(out)}"
          f" from {ref or 'the working tree'}")
    for reason in blockers:
        print(f"  !!! NOT FOR DISTRIBUTION (development build, files named {prefix}…): {reason}")
    files = file_list(ref)
    stage_files(files, stage, ref)
    print(f"  {len(files)} files")
    if not a.no_examples:
        make_examples(stage, not a.examples_no_results)
    if not a.no_docs and not a.check_only:
        make_docs(stage)
    make_licenses(stage)
    if blockers:
        collect_licenses.mark_not_for_distribution(stage / "licenses", blockers)
    add_sources(stage, "all")                                           # CalculiX source (ccx_spooles.exe, ccx_spooles)
    normalize_eol(stage)
    refuse_mystran(stage)
    if not a.no_scan:
        found = anonymization_check(stage)
        if found:
            print(f"Anonymization: {len(found)} findings – the package is NOT built:", file=sys.stderr)
            for f in found[:200]:
                print(f"  {f}", file=sys.stderr)
            shutil.rmtree(out if a.check_only else stage, ignore_errors=True)
            return 3
    else:
        print("  ! anonymization scan skipped (--no-scan)")
    if a.check_only:
        shutil.rmtree(out, ignore_errors=True)
        print("  check passed (no packages built)")
        return 0
    zip_path = out / f"{prefix}.zip"
    tar_path = out / f"{prefix}.tar.gz"
    made = [zip_path, tar_path]
    if not a.no_python:
        bundle_python(stage)                                            # Python 3.11 only in the Windows packages
    win_tools = bundle_windows_tools(stage, with_ccx=not a.no_windows_ccx, ref=ref)       # uv.exe, CalculiX for Windows
    win_src = [] if a.no_windows_ccx else add_sources(stage, "windows", runtime=not a.no_runtime_sources)   # SPOOLES, ARPACK-ng, LAPACK, the GCC / mingw-w64 run-time sources
    manifest = write_package_manifest(stage, ver, build)              # the Windows packages: their sources included
    print(f"  manifest (Windows): {len(manifest['files'])} program files, build {build}")
    write_zip(stage, prefix, zip_path)
    iscc = "" if a.no_setup else find_iscc(a.iscc)
    if iscc:
        iscc_log = Path(a.iscc_log).resolve() if a.iscc_log else None
        exe = write_setup_exe(stage, out, ver, iscc, dev=dev, log=iscc_log)
        if exe:
            made.append(exe)
        if a.offline:
            wheels = stage / "tools" / "offline" / "wheels"
            reqs = offline_requirements()
            files = fetch_wheels(reqs, ROOT / "tools" / "_download" / "offline-wheels")     # cache between the builds
            wheels.mkdir(parents=True, exist_ok=True)
            for w in files:
                shutil.copy2(w, wheels / w.name)
            print(f"  offline: {len(files)} wheels, {sum(w.stat().st_size for w in files) / 1e6:.0f} MB")
            extra_src = add_sources(stage, "offline")                   # Gmsh source next to the Gmsh wheel (GPL-2+)
            write_package_manifest(stage, ver, build)
            exe = write_setup_exe(stage, out, ver, iscc, offline=True, dev=dev)
            if exe:
                made.append(exe)
            shutil.rmtree(stage / "tools" / "offline", ignore_errors=True)
            for p in extra_src:
                p.unlink(missing_ok=True)
    elif not a.no_setup:
        print("  Inno Setup (ISCC.exe) not found – setup.exe is not created (python installer/get_innosetup.py)")
    for d in win_tools:
        shutil.rmtree(d, ignore_errors=True)
    remove_sources(stage, win_src)                                      # the Linux packages have their own sources
    shutil.rmtree(stage / "tools" / "python", ignore_errors=True)
    if not any((stage / "tools").iterdir() if (stage / "tools").exists() else []):
        shutil.rmtree(stage / "tools", ignore_errors=True)
    if not a.no_linux_ccx:                                              # SPOOLES, ARPACK-ng, LAPACK, the Rust part, the GCC source and its recipe
        add_sources(stage, "linux", runtime=not a.no_runtime_sources)
    manifest = write_package_manifest(stage, ver, build)              # the Linux packages: their sources instead of the Windows ones
    print(f"  manifest (Linux): {len(manifest['files'])} program files")
    if not a.no_linux_ccx:
        pre = require_linux_ccx(with_pastix=a.with_ccx_pastix)
        (stage / "tools" / "calculix").mkdir(parents=True, exist_ok=True)
        linux_ccx(pre, stage / "tools" / "calculix" / pre.name, a.with_ccx_pastix)
    else:
        print("  ! --no-linux-ccx: the Linux package has NO CalculiX (the installer does not compile) – do not release it")
    write_tar(stage, prefix, tar_path)
    if not a.no_run:
        made.append(write_run(tar_path, out / f"{prefix}-linux-x86_64.run", ver))
    write_sums(out / "SHA256SUMS.txt", made)
    shutil.rmtree(stage, ignore_errors=True)
    for p in made:
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
    if blockers:
        print(f"  !!! NOT FOR DISTRIBUTION: development build ({'; '.join(r.split(':')[0] for r in blockers)}) – the files are named {prefix}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())

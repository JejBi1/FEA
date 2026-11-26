"""Migration steps of a FEA Studio update (run by `fea_setup.py update` / `install` over an older installation).

Each step: the version in which it was added, the phase ("pre" = before the package synchronisation, "post" = after it, "final" = after the external components
(CalculiX, hypre ...) were installed – for a step that may remove something only once its replacement is in place), a description and the function `fn(ctx)`.
The steps with a version > the previously installed one and ≤ the new one run, sorted by version and phase. The steps MUST be idempotent
(an update interrupted in the middle runs again completely). `ctx` = `Context` below (paths, modules, output, installer).

A new step = add a function and an entry to MIGRATIONS (version = the program version that brings it).
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import upgrade
from setup_lang import T


@dataclass
class Context:
    root: Path
    venv: Path
    vpy: Path
    modules: list[str]
    say: Callable[[str], None] = print
    warn: Callable[[str], None] = print
    setup: object = None                 # module fea_setup (uv, site_packages…) – may be missing in the tests
    done: list[str] = field(default_factory=list)
    kept_reported: set[str] = field(default_factory=set)   # modified obsolete files already listed (list once)


@dataclass(frozen=True)
class Migration:
    version: str
    phase: str                           # "pre" | "post" | "final"
    title: str
    fn: Callable[[Context], None] | None   # None = only a change of the module list (`modules`), nothing to run or announce
    # transformation of the module list of the PREVIOUS installation (before the requirements of the modules are added and before the list is
    # shown): (modules) → (modules, note for the user); see `plan_modules`
    modules: Callable[[list[str]], tuple[list[str], str]] | None = None
    # a step about something that exists on Windows only (the third-party ccx_static.exe): `run` neither announces nor runs it elsewhere (its idempotent twin in
    # HOUSEKEEPING still looks, silently, for leftovers of a Windows tree that was copied to another system)
    windows_only: bool = False


# ------------------------------------------------------------------ 1.0.1
def remove_old_release_files(ctx: Context) -> None:
    """Files of older releases (0.1.0, 1.0.0) that the new version does not have – also from installations without a saved manifest
    (e.g. the moved tests/gui_*.py, fea_core/contacts.py replaced by the package contacts/). Only unchanged files.
    """
    new = upgrade.read_manifest(ctx.root / upgrade.PACKAGE_MANIFEST)
    if new is None:
        ctx.say(T("balíček nemá seznam souborů – staré soubory se nemažou",
                  "the package has no file list – old files are not removed"))
        return
    old: dict[str, str] = {}
    for m in upgrade.history_manifests(ctx.root).values():
        old.update(upgrade.obsolete_files(m, new))
    upgrade.remove_obsolete(ctx.root, old, say=ctx.say, reported=ctx.kept_reported)
    orphans = upgrade.orphan_bytecode(ctx.root)
    for p in orphans:
        p.unlink(missing_ok=True)
    if orphans:
        ctx.say(T(f"osiřelý bajtkód smazán ({len(orphans)} souborů)", f"orphaned bytecode deleted ({len(orphans)} files)"))


# packages whose __init__.py the installer deletes on purpose (installer/slim.py)
_SLIMMED_TOP = {"vtkmodules"}


def broken_packages(site: Path) -> list[str]:
    """Installed packages whose `<package>/__init__.py` from their RECORD is missing – the state after uninstalling a package with
    shared files (PySide6 + PySide6_Addons from an old environment deleted PySide6/__init__.py of PySide6_Essentials).
    Returns "name==version" to reinstall.
    """
    import csv
    out = []
    for di in sorted(site.glob("*.dist-info")):
        rec = di / "RECORD"
        if not rec.is_file():
            continue
        with open(rec, encoding="utf-8", newline="") as f:
            rows = [r[0] for r in csv.reader(f) if r]
        inits = [r for r in rows if r.count("/") == 1 and r.endswith("/__init__.py") and r.split("/")[0] not in _SLIMMED_TOP]
        if any(not (site / r).exists() for r in inits):
            name, _, ver = di.name[: -len(".dist-info")].rpartition("-")
            out.append(f"{name}=={ver}")
    return out


def repair_shared_package_files(ctx: Context) -> None:
    """Repair of 1.0.0 installations after `uv pip sync` (PySide6): reinstalls the packages with missing files and checks that Qt loads."""
    if ctx.setup is None or not ctx.vpy.exists():
        return
    site = ctx.setup.site_packages()
    broken = broken_packages(site)
    if broken:
        ctx.say(T("přeinstalace balíčků s chybějícími soubory: ", "reinstalling packages with missing files: ") + ", ".join(broken))
        _reinstall(ctx, broken)
    if "gui" in ctx.modules:
        r = subprocess.run([str(ctx.vpy), "-c", "import PySide6.QtWidgets"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            pins = [p for p in (_installed_pin(site, n) for n in ("PySide6_Essentials", "shiboken6")) if p]
            ctx.say(T("Qt nejde načíst – přeinstalace ", "Qt does not load – reinstalling ") + ", ".join(pins))
            if pins:
                _reinstall(ctx, pins)
            r = subprocess.run([str(ctx.vpy), "-c", "import PySide6.QtWidgets"], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                ctx.warn(T("grafické rozhraní (Qt) stále nejde načíst – spusťte instalaci s --force: ",
                           "the graphical interface (Qt) still does not load – run the installation with --force: ")
                         + (r.stderr.strip().splitlines() or ["?"])[-1])


def _reinstall(ctx: Context, pins: list[str]) -> None:
    """Reinstall of exact pins with the SHA-256 digests of the version lock (fea_setup.install_pins, --require-hashes); a pin
    without a digest is not installed unverified – only a warning (the migration must not stop the update)."""
    install = getattr(ctx.setup, "install_pins", None)
    if install is None:                     # an older fea_setup without install_pins
        ctx.setup.uv("pip", "install", "--reinstall", "--no-deps", *pins, "--python", str(ctx.vpy))
        return
    try:
        install(pins)
    except RuntimeError as e:               # fea_setup.HashError
        ctx.warn(str(e))


def _installed_pin(site: Path, name: str) -> str | None:
    key = name.lower().replace("-", "_")
    for di in site.glob("*.dist-info"):
        n, _, ver = di.name[: -len(".dist-info")].rpartition("-")
        if n.lower().replace("-", "_") == key:
            return f"{n}=={ver}"
    return None


# ------------------------------------------------------------------ 1.4.2
def direct_solver_modules(installed: list[str]) -> tuple[list[str], str]:
    """1.4.2: the direct solver (rslab) and the own matrix assembly moved out of the module ngsolve into the module direct (which needs accel =
    numba). `installed` = the modules of the previous installation (its state file, NOT the list after the requirements were added: ngsolve now
    requires motion, so the closure would add Exudyn to an installation that never had it):
      - without ngsolve: unchanged;
      - with ngsolve and motion: ngsolve stays (the flexible bodies of the mechanisms need it), direct + accel are added;
      - with ngsolve, without motion: ngsolve was only the solver of the large models – it is replaced by direct + accel (170 MB less, no Exudyn).
    Idempotent. → (modules, the note for the user, "" when nothing changed)."""
    if "ngsolve" not in installed or "direct" in installed:
        return list(installed), ""
    keep = "motion" in installed
    out = [k for k in installed if keep or k != "ngsolve"]
    for k in ("accel", "direct"):
        if k not in out:
            out.append(k)
    if keep:
        note = T("přidán modul direct (přímý řešič rslab a vlastní sestavení matice); modul ngsolve zůstává jen pro pružná tělesa mechanismů "
                 "(nepoužíváte-li je: python -m fea_core.modules remove ngsolve ušetří 170 MB)",
                 "added the module direct (the rslab direct solver and the own matrix assembly); the module ngsolve stays only for the flexible bodies "
                 "of mechanisms (if you do not use them: python -m fea_core.modules remove ngsolve saves 170 MB)")
    else:
        note = T("modul ngsolve nahrazen modulem direct (přímý řešič rslab a vlastní sestavení matice, Numba z modulu accel); NGSolve už lineární "
                 "statika nepotřebuje (zbývají jen pružná tělesa mechanismů – modul motion není nainstalován)",
                 "the module ngsolve is replaced by the module direct (the rslab direct solver and the own matrix assembly, Numba from the module "
                 "accel); linear statics no longer needs NGSolve (only the flexible bodies of mechanisms do – the module motion is not installed)")
    return out, note


def add_direct_solver_module(ctx: Context) -> None:
    """Safety net of the step 1.4.2 for callers that did not apply `plan_modules` to the installed list (the planned list is the normal route,
    `fea_setup.cmd_install`): the modules of the update get direct (+ accel) when they still carry ngsolve without it. Idempotent."""
    new, note = direct_solver_modules(ctx.modules)
    if new != ctx.modules:
        ctx.modules[:] = new
        ctx.say(note)


def old_ngsolve_in_exact_selection(selection: list[str], installed: list[str], old_version: str) -> tuple[list[str], str]:
    """setup.exe over a setup.exe installation of an older version passes --exact, and the wizard restores the module `ngsolve` of that
    installation – there it was the solver of large models and pulled in no mechanisms. In this version `ngsolve` is only for the flexible bodies
    of mechanisms and REQUIRES `motion` (Exudyn), so the selection taken literally would add Exudyn unasked, unlike the same update from the zip
    or .run (migration 1.4.2). The same rule as there is applied to a selection that has `ngsolve` without `motion` when the installation had it
    that way too: `direct` + `accel` instead. (`motion` ticked = the user wants it: `ngsolve` stays.) → (selection, note)."""
    if upgrade.parse_version(old_version) >= upgrade.parse_version("1.4.2"):
        return list(selection), ""
    if "ngsolve" not in selection or "motion" in selection or "ngsolve" not in installed or "motion" in installed:
        return list(selection), ""
    return direct_solver_modules([k for k in selection if k != "direct"])


def final_modules(keys: list[str], old_version: str, new_version: str) -> tuple[list[str], str]:
    """The module list after everything a "pre" step would still do to it (the 1.4.2 safety net `add_direct_solver_module`): applied before the
    list is printed, so that the printed list is the list that is installed. → (modules, note)."""
    if any(m.fn is add_direct_solver_module for m in pending(old_version, new_version, "pre")):
        return direct_solver_modules(keys)
    return list(keys), ""


def requirement_note(asked: list[str], keys: list[str]) -> str:
    """`ngsolve` brings `motion` with it (it is only the flexible bodies of mechanisms since 1.4.2): said when neither the command line / the
    wizard nor the installation named `motion`."""
    if "ngsolve" in keys and "motion" in keys and "motion" not in asked:
        return T("modul ngsolve (pružná tělesa mechanismů) vyžaduje modul motion (mechanismy, Exudyn) – přidán; bez pružných těles: "
                 "python -m fea_core.modules remove ngsolve",
                 "the module ngsolve (flexible bodies of mechanisms) requires the module motion (mechanisms, Exudyn) – added; without flexible "
                 "bodies: python -m fea_core.modules remove ngsolve")
    return ""


# ------------------------------------------------------------------ 1.5.1 – new modules of the standard profile
# the standard profile of 1.4.x without its solver module (pardiso → direct): an installation that has all of it was a standard (or full) one
STANDARD_BEFORE_151 = ("gui", "plots", "accel", "beams", "report", "motion")
NEW_STANDARD_151 = ("sketch",)


def new_standard_modules(installed: list[str]) -> tuple[list[str], str]:
    """An update keeps the modules "as last time"; a standard installation then lacked the module `sketch` (DXF import into the sketcher) that a
    fresh standard installation of 1.5.1 has. It is added to an installation that had the whole standard profile; a smaller or hand-picked one
    is left alone. Idempotent. → (modules, note)."""
    if not set(STANDARD_BEFORE_151) <= set(installed):
        return list(installed), ""
    add = [k for k in NEW_STANDARD_151 if k not in installed]
    if not add:
        return list(installed), ""
    return list(installed) + add, T(f"nový modul standardní instalace přidán: {', '.join(add)} (import DXF do náčrtníku; odebrání: "
                                    f"python -m fea_core.modules remove {' '.join(add)})",
                                    f"new module of the standard installation added: {', '.join(add)} (DXF import into the sketcher; to remove it: "
                                    f"python -m fea_core.modules remove {' '.join(add)})")


# ------------------------------------------------------------------ 1.4.2 – no Intel MKL
# Python distributions of the module `pardiso` (mkl and what it pulled in), as the names of their `<name>-<version>.dist-info` folders
INTEL_DISTS = ("mkl", "intel_openmp", "tbb")


def drop_pardiso_module(installed: list[str]) -> tuple[list[str], str]:
    """1.4.2: the module `pardiso` (Intel MKL + `ccx_i4`) no longer exists – the installation state of an older version may list it. It was the FAST
    solver of the installation (the one that `auto` chose), so the module `direct` (the rslab direct solver, the same role without Intel; it needs `accel`)
    takes its place: without it an update of a Standard installation of 1.4.x ended with neither (every study went to CalculiX, one thread of SPOOLES on
    Linux). The update then installs nothing of the old module, and `remove_intel_components` clears what it left. Idempotent.
    → (modules, the note for the user, "" when nothing changed)."""
    out = [k for k in installed if k != "pardiso"]
    if len(out) == len(installed):
        return out, ""
    for k in ("accel", "direct"):
        if k not in out:
            out.append(k)
    return out, T("modul pardiso (Intel MKL, ccx_i4) byl zrušen a nahrazen modulem direct (přímý řešič rslab, Numba z modulu accel) – CalculiX počítá "
                  "řešičem SPOOLES; zbytky Intelu se odstraní",
                  "the module pardiso (Intel MKL, ccx_i4) was removed and replaced by the module direct (the rslab direct solver, Numba from the module "
                  "accel) – CalculiX computes with SPOOLES; the Intel leftovers are removed")


def exact_retired_modules(selection: list[str], installed: list[str], named: list[str]) -> tuple[list[str], str]:
    """1.4.2, `install --exact` (setup.exe over a setup.exe installation): `plan_modules` does not run there – the selection is exactly what the user
    chose – and the retired `pardiso` is no component of the new wizard, so a selection never carries it. A custom selection of 1.4.x (core + gui +
    pardiso) would end without any fast solver and without a word. Therefore, when the installation state (`installed`: the raw module list of the
    previous installation, retired names included) or the command line (`named`) still names `pardiso`, its replacement – `direct` + `accel`, the same
    as `drop_pardiso_module` – is added to the selection and the note says so. Applies once: the state written by this run no longer names the module,
    so a later run of setup.exe where the user unticked `direct` stays as chosen. A wizard that unticks `direct` in this one run is overridden (the
    retired module meant "the fast solver"; `python -m fea_core.modules remove direct` removes it afterwards).
    → (selection with the replacement, the note for the user, "" when nothing changed)."""
    if "pardiso" not in installed and "pardiso" not in named:
        return list(selection), ""
    out, note = drop_pardiso_module([k for k in selection if k != "pardiso"] + ["pardiso"])
    return out, (note if out != list(selection) else "")


def _site_packages(ctx: Context) -> Path | None:
    if ctx.setup is not None and ctx.vpy.exists():
        try:
            return Path(ctx.setup.site_packages())
        except Exception:  # noqa: BLE001 – a broken environment: look at the usual places below
            pass
    for pat in ("Lib/site-packages", "lib/python3*/site-packages"):
        found = sorted(ctx.venv.glob(pat))
        if found:
            return found[0]
    return None


def uninstall_dist(site: Path, venv: Path, dist: str) -> tuple[int, int]:
    """Removes an installed Python distribution by its RECORD (what `pip uninstall` does – without needing pip / uv): every file the RECORD lists
    that lies inside the environment, then the `.dist-info` folder, then the folders left empty. A file that cannot be deleted (a running program
    holds the DLL) keeps the `.dist-info`, so that the next run finishes the job. → (files removed, files / folders that could not be removed);
    (0, 0) = not installed."""
    import csv
    venv_root = venv.resolve()
    removed = failed = 0
    for di in sorted(site.glob(f"{dist}-*.dist-info")):
        if di.name[: -len(".dist-info")].rpartition("-")[0].lower().replace("-", "_") != dist:     # "mkl-*" must not match "mkl_service-*"
            continue
        rec = di / "RECORD"
        dirs: set[Path] = set()
        rows: list[list[str]] = []
        if rec.is_file():
            with open(rec, encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))                            # read completely, closed before anything is deleted (Windows)
        for row in rows:
            if not row:
                continue
            p = (site / row[0]).resolve()
            if venv_root not in p.parents or di.resolve() in p.parents:       # outside the environment (a RECORD is data); the dist-info goes as a whole
                continue
            if p.is_file() or p.is_symlink():
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    failed += 1
            dirs.add(p.parent)
        if not failed:
            shutil.rmtree(di, ignore_errors=True)
            if di.exists():                                           # something inside is held: the next run finishes it
                failed += 1
        for d in sorted(dirs, key=lambda x: len(x.parts), reverse=True):      # deepest first
            while d != venv_root and venv_root in d.parents and d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except OSError:                                       # in use / not allowed: leave the folder, go on with the other distributions
                    break
                d = d.parent
    return removed, failed


def is_link(p: Path) -> bool:
    """A symbolic link or a Windows junction: the folder really lives somewhere else (a git worktree whose `.venv` and `tools` point to the main
    checkout, an environment moved to another disk). Only the reparse tags that redirect count – a placeholder of a cloud-sync folder
    (OneDrive and other sync clients: a reparse tag of its own) is an ordinary folder."""
    try:
        if p.is_symlink():
            return True
        tag = getattr(os.lstat(p), "st_reparse_tag", 0)
    except OSError:
        return False
    return tag in (getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1), getattr(stat, "IO_REPARSE_TAG_SYMLINK", -1))


def _first_link(base: Path, *parts: str) -> Path | None:
    """The first of base/parts[0], base/parts[0]/parts[1] … that is a link (`is_link`), None = real folders all the way. `base` itself is not
    checked: an installation folder that is itself a link (a user's choice) is the installation."""
    p = base
    for part in parts:
        p = p / part
        if is_link(p):
            return p
    return None


# the CalculiX build with MKL of the earlier versions: the installer wrote exactly this file (Windows: extracted from the dhondt.de archive,
# Linux: part of the prebuilt archive) into tools/calculix/<archive folder>/ – never anything else with this prefix
CCX_I4_NAMES = ("ccx_i4.exe", "ccx_i4")


def ccx_i4_files(calc: Path) -> tuple[list[Path], list[Path]]:
    """(files, folders that are links and were not entered) – the files `CCX_I4_NAMES` in `calc` and its direct subfolders: the places
    `fea_core.config.find_ccx` searches and the installer used."""
    found: list[Path] = []
    skipped: list[Path] = []
    try:
        subs = sorted(p for p in calc.iterdir() if p.is_dir())
    except OSError:
        return found, skipped
    for folder in [calc, *subs]:
        if folder != calc and is_link(folder):
            skipped.append(folder)
            continue
        for n in CCX_I4_NAMES:
            p = folder / n
            if p.is_file() or p.is_symlink():
                found.append(p)
    return found, skipped


def _redirect_note(what: str, link: Path) -> str:
    return T(f"{what}: {link} je odkaz (junction / symbolický odkaz) na jinou složku – nic se tudy nemaže (git worktree nebo přesunuté prostředí); "
             "spusťte aktualizaci v instalaci, které složka patří",
             f"{what}: {link} is a link (junction / symbolic link) to another folder – nothing is deleted through it (a git worktree or a moved "
             "environment); run the update in the installation that owns the folder")


def remove_intel_components(ctx: Context) -> None:
    """1.4.2: nothing of Intel stays on the computer: the CalculiX build with MKL (`ccx_i4.exe` / `ccx_i4` in tools/calculix) and the Python packages
    mkl, intel-openmp and tbb that the module `pardiso` installed. Idempotent and cheap when nothing is left (a few directory listings): it also
    runs on every update / repair over an existing installation (`HOUSEKEEPING`), so a file that a running program held is removed by the next run.
    Runs in the final phase, after the package synchronisation and the new CalculiX: an update that fails (no internet) removes nothing of the
    previous installation. (The new version lock has none of the packages either, so the synchronisation itself removes them as surplus; this
    step does not depend on that.)
    A link is never followed: in a git worktree `.venv` and `tools` are junctions to the main checkout, and the cleanup would delete the files of
    ANOTHER installation – such a folder is skipped with a note."""
    n = 0
    calc = ctx.root / "tools" / "calculix"
    link = _first_link(ctx.root, "tools", "calculix")
    if link is not None:
        ctx.say(_redirect_note("tools", link))
    elif calc.is_dir():
        files, skipped = ccx_i4_files(calc)
        for s in skipped:
            ctx.say(_redirect_note("tools", s))
        for p in files:
            try:
                p.unlink()
                n += 1
            except OSError as e:
                ctx.warn(T(f"{p} nejde smazat: {e}", f"cannot delete {p}: {e}"))
    site = _site_packages(ctx)
    pkgs = held = 0
    try:
        inside = site.relative_to(ctx.venv).parts if site is not None else None
    except ValueError:                                                 # site-packages outside the environment (a broken setup): not touched
        inside = None
    if inside is not None:
        link = ctx.venv if is_link(ctx.venv) else _first_link(ctx.venv, *inside)
        if link is not None:
            ctx.say(_redirect_note(".venv", link))
        else:
            for dist in INTEL_DISTS:
                removed, failed = uninstall_dist(site, ctx.venv, dist)
                pkgs += bool(removed) and not failed
                held += failed
    if n or pkgs:
        ctx.say(T(f"Intel odstraněn: {n} souborů ccx_i4, {pkgs} balíčků (mkl, intel-openmp, tbb)",
                  f"Intel removed: {n} ccx_i4 files, {pkgs} packages (mkl, intel-openmp, tbb)"))
    if held:
        ctx.warn(T(f"{held} souborů Intelu používá běžící program – zavřete FEA Studio a MCP klienty a spusťte instalaci nebo aktualizaci znovu "
                   "(stačí stejná verze, dokončí se úklid)",
                   f"{held} Intel files are in use by a running program – close FEA Studio and the MCP clients and run the installation or the update "
                   "again (the same version is enough, it finishes the cleanup)"))
    leftover = os.environ.get("FEA_CCX", "")
    if leftover and Path(leftover).name.lower().startswith("ccx_i4"):           # the variable of an installation that used the MKL build
        ctx.warn(T(f"proměnná prostředí FEA_CCX ukazuje na {leftover} – CalculiX s Intel MKL už neexistuje a program ji přeskakuje; proměnnou odstraňte "
                   "(nebo ji nasměrujte na jiný program ccx)",
                   f"the environment variable FEA_CCX points at {leftover} – CalculiX with Intel MKL no longer exists and the program skips it; remove "
                   "the variable (or point it at another ccx program)"))


# The third-party CalculiX of dhondt.de (calculix_2.23_4win.zip, SHA-256 87fc4e17…) that the installers of the versions before 1.4.2 put into
# tools/calculix/calculix_2.23_4win/: the installer extracted ccx_static.exe and README_Install, a development tree may carry the whole zip.
# `ccx_static.exe` statically contains TAUCS (AMD ordering with an educational-use NOTICE, GENMMD without a licence text), PaStiX (LGPL-3.0 in a
# GPL-2.0-only program) and Scotch (CeCILL-C) – it is replaced by the program's own build (installer/windows/build_ccx.ps1).
# Only the files of that zip are removed, and only when a file is byte for byte the one of the zip (SHA-256 below): a file that a developer replaced,
# or put there on purpose, is kept with a note; so is the file that the variable FEA_CCX names (the developer's override for comparisons). Nothing outside
# tools/calculix/calculix_2.23_4win/ is touched; a folder that still holds other files stays. The old build is removed only when the program's own
# ccx_spooles.exe is in place and matches its manifest (the final phase, after the external components were installed): an update that failed to unpack
# the new solver must not leave the user without one.
OLD_CCX_DIR = "calculix_2.23_4win"
OLD_CCX_ZIP = {
    "ccx_static.exe": "e18360eb32a5cd48f91a212c57069212cd8e27d2f798a599a78106bd0b3ecc1f",
    "ccx_dynamic.exe": "dc894b9cbb04d30ed3c19cac533f31c0596fa0604514a26314a9e7967f0e6669",
    "ccx_dynamic_i8.exe": "a675885aa3fc794bc74157d208451c929175b5f81d5db223353078a7700be437",
    "ccx_i4.exe": "ae1c27ee8d8c5ee04119525548b9f39dc2c79bb9fcabf5a01d089097b74f37f1",
    "cgx_GLUT.exe": "a9fc7bf1991401aaf1d097fe20c58f7a6af12b7fbbef94fe2657e66e2a5bcd80",
    "cgx_STATIC.exe": "b26fa5f98cf67e31e68882ff9f12348719f0ef24d4260824227d75ed1b3b6242",
    "glut64.dll": "d36dd021fa7c43e85e13be0098e7d82caebc22adef52e6f75f14906395a64a5a",
    "README_Install": "76e052b220e7d6bc9689591a77db8149e1c09bfa42fe1eccb8cd029bb2b6b8fe",
    "README_Install~": "6261a0c187f94ac795bc2c8bbce2a5651d3b3e7b5e1f73aab8fb0010991948b8",
    "tetgen.exe": "bf3116713a00544696a4651bc375133e373c6a8885774f283a753bdaf07233ee",
    "cgxCadTools/cad2fbd.exe": "ae93d37e8ff634a6002bf3eda10a6ebc4971437676f8d36b4fdf35a4c2c93ff6",
    "cgxCadTools/fbd2cad.exe": "ef0fedd342199921e6c46e2fc856532e7ea70c8243fb80331fe590bfaaeba295",
}
OWN_CCX_DIR = "ccx_2.23_windows"


def old_ccx_files(calc: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """(files of the old zip found – byte for byte, its folders, links that were not entered, files with the name of one of the zip but other content)
    under `calc` = tools/calculix. A link is never entered; a file is hashed only when it exists."""
    files: list[Path] = []
    skipped: list[Path] = []
    changed: list[Path] = []
    folders: list[Path] = []
    top = calc / OLD_CCX_DIR
    if top.is_dir():
        if is_link(top):
            skipped.append(top)
        else:
            folders += [top / "cgxCadTools", top]
            for rel, digest in OLD_CCX_ZIP.items():
                p = top / Path(rel)
                if p.parent != top and is_link(p.parent):
                    continue
                if p.is_file() and not p.is_symlink():
                    try:
                        same = upgrade.sha256_file(p) == digest
                    except OSError:
                        same = False
                    (files if same else changed).append(p)
    return files, folders, skipped, changed


def own_ccx_ready(root: Path) -> bool:
    """The program's own Windows CalculiX is in place: tools/calculix/ccx_2.23_windows/ccx_spooles.exe exists and, when its manifest.json carries a digest, has it."""
    d = root / "tools" / "calculix" / OWN_CCX_DIR
    exe = d / "ccx_spooles.exe"
    if not exe.is_file():
        return False
    try:
        want = json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("binary_sha256")
    except (OSError, ValueError, AttributeError):
        want = None
    return not isinstance(want, str) or upgrade.sha256_file(exe) == want


def remove_thirdparty_ccx(ctx: Context) -> None:
    """1.4.2: the third-party ccx_static.exe and the rest of the old CalculiX zip leave tools/calculix (the program's own CalculiX is unpacked by the
    installer next to it: ccx_2.23_windows/ccx_spooles.exe; see the comment above for what is removed and what is kept). Idempotent and cheap when nothing is
    left (a few directory listings); it runs in the final phase of every update / repair (`HOUSEKEEPING`), so a file that a running calculation held is
    removed by the next run – a locked file is tried three times first (an antivirus scan). A link is never followed (a git worktree whose `tools` is a
    junction to the main checkout: that installation owns the files)."""
    import time
    calc = ctx.root / "tools" / "calculix"
    link = _first_link(ctx.root, "tools", "calculix")
    if link is not None:
        ctx.say(_redirect_note("tools", link))
        return
    if not calc.is_dir():
        return
    files, folders, skipped, changed = old_ccx_files(calc)
    for s in skipped:
        ctx.say(_redirect_note("tools", s))
    for c in changed:
        ctx.warn(T(f"{c} není soubor původního balíčku CalculiX (jiný obsah) – ponechán", f"{c} is not the file of the original CalculiX package (other content) – kept"))
    leftover = os.environ.get("FEA_CCX", "")
    named = None
    if leftover:
        try:
            named = Path(leftover).resolve()
        except OSError:
            named = None
    keep = [f for f in files if named is not None and f.resolve() == named]
    files = [f for f in files if f not in keep]
    for f in keep:                                       # the developer's override: FEA_CCX names it
        ctx.warn(T(f"{f} zůstává: proměnná prostředí FEA_CCX na něj ukazuje (CalculiX třetí strany nahrazen vlastním sestavením ccx_spooles.exe – "
                   "proměnnou odstraňte a soubor smažte ručně, až ho nebudete potřebovat)",
                   f"{f} is kept: the environment variable FEA_CCX names it (the third-party CalculiX was replaced by the program's own build "
                   "ccx_spooles.exe – remove the variable and delete the file yourself when you no longer need it)"))
    if files and not own_ccx_ready(ctx.root):
        ctx.warn(T("starý CalculiX třetí strany zůstává: vlastní ccx_spooles.exe ještě není na místě (rozbalení z balíčku selhalo?) – smazání se zopakuje "
                   "při další instalaci nebo aktualizaci",
                   "the old third-party CalculiX stays: the program's own ccx_spooles.exe is not in place yet (did unpacking it from the package fail?) – the "
                   "removal is repeated by the next installation or update"))
        return
    n = held = 0
    for f in files:
        for attempt in range(3):
            try:
                f.unlink()
                n += 1
                break
            except OSError as e:
                if attempt == 2:
                    held += 1
                    ctx.warn(T(f"{f} nejde smazat: {e}", f"cannot delete {f}: {e}"))
                else:
                    time.sleep(0.5)
    for d in folders:                                  # deepest first; a folder that still holds a file stays
        try:
            if d.is_dir() and not is_link(d) and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    if n:
        ctx.say(T(f"CalculiX třetí strany (ccx_static.exe) odstraněn: {n} souborů – program používá vlastní sestavení CalculiX (ccx_spooles.exe)",
                  f"third-party CalculiX (ccx_static.exe) removed: {n} files – the program uses its own CalculiX build (ccx_spooles.exe)"))
    if held:
        ctx.warn(T(f"{held} souborů starého CalculiX používá běžící program – zavřete FEA Studio a spusťte instalaci nebo aktualizaci znovu "
                   "(stačí stejná verze, dokončí se úklid)",
                   f"{held} files of the old CalculiX are in use by a running program – close FEA Studio and run the installation or the update again "
                   "(the same version is enough, it finishes the cleanup)"))


# ------------------------------------------------------------------ 1.5.1
MYSTRAN_URL = "https://www.mystran.com"


def remove_bundled_mystran(ctx: Context) -> None:
    """1.5.1: the MYSTRAN program is no longer shipped (owner decision 26. 11. 2026: the upstream static build carries libraries – libquadmath
    LGPL-2.1, SuperLU, METIS – whose sources and licence texts the packages did not carry). Releases before 1.5.1 put it into tools/mystran, so the
    update removes that copy: tools/mystran/mystran.exe (Windows) or tools/mystran/mystran (Linux), and the folder when it is then empty. Runs only
    for an update from a version older than 1.5.1 (a version step): a MYSTRAN the user downloads after this update is never touched. A link is
    never followed (a git worktree whose `tools` points to the main checkout: that installation owns the file)."""
    link = _first_link(ctx.root, "tools", "mystran")
    if link is not None:
        ctx.say(_redirect_note("tools", link))
        return
    folder = ctx.root / "tools" / "mystran"
    if not folder.is_dir():
        return
    removed = 0
    for name in ("mystran.exe", "mystran"):
        f = folder / name
        if not f.is_file() or is_link(f):
            continue
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            ctx.warn(T(f"{f} nejde smazat: {e}", f"cannot delete {f}: {e}"))
    try:
        if removed and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass
    if removed:
        ctx.say(T("MYSTRAN odstraněn: program už není součástí balíčku (licence jeho přibalených knihoven). Pro křížové kontroly MYSTRAN si ho "
                  f"stáhněte sami z {MYSTRAN_URL} a uložte do tools/mystran",
                  "MYSTRAN removed: the program is no longer part of the package (licence of its bundled libraries). For the MYSTRAN cross-checks "
                  f"download it yourself from {MYSTRAN_URL} and save it in tools/mystran"))


MIGRATIONS: list[Migration] = [
    Migration("1.0.1", "pre", T("smazání souborů starších vydání a osiřelého bajtkódu",
                               "removal of files from older releases and orphaned bytecode"), remove_old_release_files),
    Migration("1.0.1", "post", T("oprava souborů sdílených balíčky (PySide6) a kontrola načtení Qt",
                                "repair of files shared by packages (PySide6) and Qt load check"), repair_shared_package_files),
    Migration("1.4.2", "pre", T("přímý řešič přesunut z modulu ngsolve do modulu direct",
                               "the direct solver moved from the module ngsolve to the module direct"), add_direct_solver_module,
              modules=direct_solver_modules),
    Migration("1.4.2", "final", T("Intel MKL odstraněn (ccx_i4, balíčky mkl, intel-openmp, tbb, modul pardiso)",
                                  "Intel MKL removed (ccx_i4, packages mkl, intel-openmp, tbb, module pardiso)"), remove_intel_components,
              modules=drop_pardiso_module),
    Migration("1.4.2", "final", T("CalculiX třetí strany (ccx_static.exe) nahrazen vlastním sestavením (ccx_spooles.exe)",
                                 "third-party CalculiX (ccx_static.exe) replaced by the own build (ccx_spooles.exe)"), remove_thirdparty_ccx,
              windows_only=True),
    Migration("1.5.1", "pre", T("nové moduly standardní instalace", "new modules of the standard installation"), None,
              modules=new_standard_modules),
    Migration("1.5.1", "final", T("MYSTRAN už není součástí balíčku (odstraněn z tools/mystran)",
                                  "MYSTRAN is no longer part of the package (removed from tools/mystran)"), remove_bundled_mystran),
]


# Idempotent cleanups that run on EVERY update / repair / install over an existing installation, whatever the previous version (after the steps of
# MIGRATIONS, not twice in one run): the state file records the new version even when a step could only be done partly (a DLL held by a running
# program), and `pending` selects steps by the version – without this the promised "run the update again" would find nothing to do. Each one must be
# cheap when there is nothing to clean (directory listings only) and report only what it did or could not do.
HOUSEKEEPING: list[Migration] = [
    Migration("1.4.2", "final", T("zbytky Intel MKL", "Intel MKL leftovers"), remove_intel_components),
    Migration("1.4.2", "final", T("zbytky CalculiX třetí strany", "third-party CalculiX leftovers"), remove_thirdparty_ccx),
]


def pending(old_version: str, new_version: str, phase: str | None = None,
            migrations: list[Migration] | None = None) -> list[Migration]:
    """Steps for the transition old → new (old < step version ≤ new), sorted by version."""
    lo, hi = upgrade.parse_version(old_version), upgrade.parse_version(new_version)
    ms = MIGRATIONS if migrations is None else migrations
    sel = [m for m in ms if lo < upgrade.parse_version(m.version) <= hi and (phase is None or m.phase == phase)]
    return sorted(sel, key=lambda m: (upgrade.parse_version(m.version), m.phase != "pre"))


def plan_modules(installed: list[str], old_version: str, new_version: str,
                 migrations: list[Migration] | None = None) -> tuple[list[str], list[str]]:
    """The module list of the previous installation carried through the steps of the transition old → new that change modules (`Migration.modules`).
    Applied BEFORE the requirements of the modules are added and before the list and its size are shown, so the user sees the final list.
    → (modules, notes for the user)."""
    mods, notes = list(installed), []
    for m in pending(old_version, new_version, migrations=migrations):
        if m.modules is not None:
            mods, note = m.modules(mods)
            if note:
                notes.append(note)
    return mods, notes


def run(ctx: Context, old_version: str, new_version: str, phase: str,
        migrations: list[Migration] | None = None) -> list[str]:
    """Runs the steps of the phase; an error of a step is reported (ctx.warn), the next steps continue. Returns the descriptions of the steps performed."""
    done = []
    ran = set()
    for m in pending(old_version, new_version, phase, migrations):
        if m.fn is None:                     # only the module list (`plan_modules`)
            continue
        if m.windows_only and not sys.platform.startswith("win"):
            continue
        ctx.say(T(f"migrace {m.version}: {m.title}", f"migration {m.version}: {m.title}"))
        ran.add(m.fn)
        try:
            m.fn(ctx)
            done.append(f"{m.version}: {m.title}")
        except Exception as ex:  # noqa: BLE001
            ctx.warn(T(f"migrace {m.version} ({m.title}) selhala: {ex}", f"migration {m.version} ({m.title}) failed: {ex}"))
    if migrations is None:                   # a caller with its own list of steps (tests) gets exactly those
        for m in HOUSEKEEPING:
            if m.phase != phase or m.fn in ran:
                continue
            try:
                m.fn(ctx)                    # silent when there is nothing to do; not listed in `done`
            except Exception as ex:  # noqa: BLE001
                ctx.warn(T(f"úklid ({m.title}) selhal: {ex}", f"cleanup ({m.title}) failed: {ex}"))
    ctx.done.extend(done)
    return done

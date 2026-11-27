"""Runs the PUBLIC verification set and builds the public page (docs/verification/).

    python -m verification.run_public [--quick] [--only V1 V4 ...] [--no-mystran] [--no-report] [--merge]

  1. the published cases of verification.catalogue (all of them, or `--only` a few) are run with verification.run_all
     (`--quick` leaves out the very long ones of verification.tiers.VERY_SLOW – used by the weekly CI run; the case KK4 of
     EXTREME takes hours and is run only when it is named with `--only KK4`);
  2. the cases the machine cannot run (a module or tool that is not installed, a solver feature that the platform lacks, e.g. the
     mortar contact of K2 on Windows) are not run and are listed with the reason – they are never counted as passed;
  3. the cross-check against MYSTRAN of the cases of catalogue.MYSTRAN_CROSS_CHECK (verification.mystran_check, coarser mesh);
  4. verification/out/run_meta.json (program, solver and platform of the run), then verification.public_report writes
     docs/verification/{README.md, results.csv, index.html} from out/report.json and out_mystran/report.json.

The report is built from what was really run: `--merge` keeps the earlier results of the cases that are not run now, `--no-report`
only runs. Exit code 0 = every case that was run is within tolerance.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from verification import catalogue, tiers  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
META = OUT / "run_meta.json"
OUT_MYSTRAN = HERE / "out_mystran"


def unavailable(entry: catalogue.Entry) -> str:
    """Why this machine cannot run the case ('' = it can). Looks at the modules / capabilities of `Entry.needs`."""
    from fea_core import config, modules
    for need in entry.needs:
        if need == "mortar":
            if not config.mortar_supported():
                return "mortar contact is substituted by surface-to-surface (the mortar matrix of CalculiX 2.23 depends on rounding; FEA_MORTAR=1 runs it)"
        elif need == "mystran":
            from fea_core import mystran
            if not mystran.installed():
                return mystran.NOT_RUN
            try:
                import pyNastran  # noqa: F401
            except ImportError:
                return "pyNastran is not installed (module verify)"
        elif not modules.available(need):
            return f"optional module '{need}' is not installed"
    return ""


# Cases that take hours: with the shipped CalculiX (SPOOLES solves on one thread) the impact KK4 (*DYNAMIC + contact, 160 increments of a
# 1.6 M-line model) was still in increment 38 of 160 after 48 minutes, ~4 h in all. They are run only on request (`--only`).
EXTREME = frozenset({"KK4"})


def select(only: list[str] | None, quick: bool) -> tuple[list[str], dict[str, str]]:
    """(keys to run, {key: reason} of the published cases that are not run)."""
    keys: list[str] = []
    skipped: dict[str, str] = {}
    for key in catalogue.public_keys():
        if only and key not in only:
            continue
        if key in EXTREME and not only:
            skipped[key] = f"takes hours (about 4 h with the shipped CalculiX): python -m verification.run_public --only {key}"
            continue
        if quick and key in tiers.VERY_SLOW:
            skipped[key] = "very long run (left out of the quick run)"
            continue
        why = unavailable(catalogue.get(key))
        if why:
            skipped[key] = why
        else:
            keys.append(key)
    keys.sort(key=lambda k: k in tiers.VERY_SLOW)          # the very long ones last (stable: the publication order is kept)
    return keys, skipped


def platform_text() -> str:
    """Operating system and processor, e.g. "Windows 10.0.26300 (AMD64)" (platform.release() says 10 also on Windows 11); no host name."""
    system = platform.system()
    return f"{system} {platform.version() if system == 'Windows' else platform.release()} ({platform.machine()})"


def solver_info() -> dict:
    """Identification of the solvers used (no paths)."""
    from fea_core import config
    info: dict = {"ccx": "", "ccx_build": "", "mystran": ""}
    ccx = config.find_ccx()
    try:
        out = subprocess.run([str(ccx), "-v"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=30, creationflags=config.NO_WINDOW).stdout
        ver = next((ln.strip() for ln in out.splitlines() if "Version" in ln), "")
        info["ccx"] = "CalculiX " + ver.replace("This is Version", "").strip() if ver else "CalculiX"
    except (OSError, subprocess.SubprocessError):
        info["ccx"] = "CalculiX (version not available)"
    info["ccx_build"] = ccx.name
    try:
        import hashlib
        info["ccx_sha256"] = hashlib.sha256(ccx.read_bytes()).hexdigest()[:16]
    except OSError:
        info["ccx_sha256"] = ""
    try:
        from fea_core import mystran
        if mystran.MYSTRAN_EXE.exists():
            out = subprocess.run([str(mystran.MYSTRAN_EXE)], input="\n", capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=30, creationflags=config.NO_WINDOW).stdout
            first = next((ln.strip() for ln in out.splitlines() if "MYSTRAN Version" in ln), "")
            info["mystran"] = " ".join(first.split()[:3]) if first else "MYSTRAN"
    except (OSError, subprocess.SubprocessError, ImportError):
        pass
    return info


def write_meta(skipped: dict[str, str], keys: list[str], seconds: float, quick: bool, merge: bool = False,
               mystran_ran: bool = False) -> dict:
    """run_meta.json of this run; with `merge` the keys, the time and the date + program version of every case of an earlier run
    (verification.run_all --merge) are kept. `mystran_ran` = the cross-check against MYSTRAN ran in this run (or in a merged earlier one)."""
    from fea_core import __version__, mystran
    old: dict = {}
    if merge:
        try:
            old = json.loads(META.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
    order = {k: i for i, k in enumerate(catalogue.public_keys())}
    ran = sorted(set(old.get("keys_run", [])) | set(keys), key=lambda k: order.get(k, 1 << 20))
    skipped = {k: v for k, v in {**(old.get("skipped") or {}), **skipped}.items() if k not in ran}
    seconds += float(old.get("duration_s", 0.0))
    now_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    case_runs = dict(old.get("case_runs") or {})
    for k in old.get("keys_run", []):            # an earlier meta without the per-case record: its own date and version hold for its cases
        case_runs.setdefault(k, {"date_utc": old.get("date_utc", ""), "program_version": old.get("program_version", "")})
    for k in keys:
        case_runs[k] = {"date_utc": now_date, "program_version": __version__}
    case_runs = {k: v for k, v in case_runs.items() if k in ran}
    quick = quick or str(old.get("profile", "")).startswith("quick")
    meta = {
        "program_version": __version__,
        "date_utc": now_date,
        "platform": platform_text(),
        "python": platform.python_version(),
        "cpus": __import__("os").cpu_count(),
        "profile": "quick (very long cases left out)" if quick else "full",
        "keys_run": ran,
        "case_runs": case_runs,
        "mystran_ran": bool(mystran_ran or old.get("mystran_ran")),
        "mystran_missing": not mystran.installed(),           # the page then says "not run", it does not call a case outside tolerance
        "skipped": skipped,
        "duration_s": round(seconds, 1),
        **solver_info(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", metavar="KEY", help="run only these published cases")
    ap.add_argument("--quick", action="store_true", help="leave out the very long cases (verification.tiers.VERY_SLOW)")
    ap.add_argument("--no-mystran", action="store_true", help="skip the cross-check against MYSTRAN")
    ap.add_argument("--no-report", action="store_true", help="only run, do not write docs/verification/")
    ap.add_argument("--merge", action="store_true", help="keep the earlier results of the cases that are not run now")
    ap.add_argument("--list", action="store_true", help="list the published cases and what would be run")
    a = ap.parse_args(argv)
    unknown = [k for k in (a.only or []) if k not in catalogue.ENTRIES or not catalogue.get(k).public]
    if unknown:
        ap.error("not a published case: " + ", ".join(unknown))
    keys, skipped = select(a.only, a.quick)
    if a.list:
        print(f"run ({len(keys)}): {' '.join(keys)}")
        for k, why in skipped.items():
            print(f"not run: {k} - {why}")
        return 0
    t0 = time.time()
    from verification import run_all
    ok = run_all.run(keys, merge=a.merge) if keys else True
    cross = [k for k in catalogue.MYSTRAN_CROSS_CHECK if k in keys]
    from fea_core import mystran
    mystran_ran = bool(cross) and not a.no_mystran and mystran.installed()
    if cross and not a.no_mystran and not mystran.installed():
        print(f"\ncross-check against MYSTRAN of {', '.join(cross)}: {mystran.not_run_text()}")
    if mystran_ran:
        from verification import mystran_check
        ok = mystran_check.run(cross, 2.0, max(1, (__import__("os").cpu_count() or 2) // 2)) and ok
    meta = write_meta(skipped, keys, time.time() - t0, a.quick, a.merge, mystran_ran)
    print(f"\nrun time {meta['duration_s']:.0f} s; not run: {len(skipped)}")
    if not a.no_report:
        from verification import public_report
        # the stored MYSTRAN comparison is read only when MYSTRAN ran (now, or in the earlier run that --merge keeps)
        rc = public_report.main([] if meta["mystran_ran"] else ["--no-mystran"])
        return 0 if ok and rc == 0 else (rc or 1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

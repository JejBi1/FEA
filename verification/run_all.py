"""Runs the verification cases and prints a table.  python -m verification.run_all [V1 V2 ...] [--backend NAME] [--merge]

`out/report.json` is rewritten after every case (a long run that is stopped keeps what it has); `--merge` keeps the entries of the cases
that are not run now (the public verification set, verification.run_public, is built from several runs).
"""
from __future__ import annotations

import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from pathlib import Path

from fea_core import mystran
from fea_core.solve import run_study
from verification import cases as C
from verification import catalogue

OUT = Path(__file__).resolve().parent / "out"


def needs_mystran(key: str) -> bool:
    """A case that itself calls the separately downloaded MYSTRAN (F9M, F3M, F11M …: `needs` of the catalogue)."""
    entry = catalogue.ENTRIES.get(key)
    return entry is not None and "mystran" in entry.needs


def _load_report() -> dict:
    try:
        data = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _verdict(all_ok: bool, selected: int, skipped: int) -> str:
    """The last line of the table. Skipped cases (MYSTRAN not installed) are not passed cases: when every selected case was skipped, nothing was
    checked and the line must not claim that the checks passed. The exit code stays 0 – a missing optional program is not a failure."""
    if not all_ok:
        return "některé kontroly NEPROŠLY"
    if skipped >= selected:
        return "nic nebylo zkontrolováno – žádná z vybraných úloh se nespustila"
    if skipped:
        return "kontroly spuštěných úloh prošly; některé úlohy se nespustily"
    return "všechny kontroly prošly"


def run(keys: list[str], backend: str = "ccx", merge: bool = False) -> bool:
    all_ok = True
    OUT.mkdir(parents=True, exist_ok=True)
    report = _load_report() if merge else {}
    not_run: list[str] = []
    for key in keys:
        case = C.CASES[key]
        if needs_mystran(key) and not mystran.installed():        # neither passed nor failed; nothing is written for the case
            print(f"\n=== {key}: {case.title}\n  {mystran.not_run_text()}")
            not_run.append(key)
            continue
        wd = OUT / (key if backend == "ccx" else f"{key}_{backend}")
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n=== {key}: {case.title}")
        try:
            study, parts = case.build(wd)
            study.solver.backend = backend
            study.save(wd / "study.json")
            if case.run is not None:                   # own calculation (mechanisms – run_motion)
                res = case.run(study, wd / "run", parts)
            else:
                res = run_study(study, wd / "run")
            checks = case.evaluate(res, study)
        except Exception as e:  # continue with the other cases
            import traceback
            traceback.print_exc()
            print(f"  CHYBA: {e}")
            all_ok = False
            report[key] = {"error": str(e)}
            _write_report(report)
            continue
        if key == "V1":
            C._V1_REF["tip"] = -C._tip_deflection(res)
        dt = time.time() - t0
        mesh = getattr(res, "mesh", None)
        if mesh is not None and getattr(mesh, "report", None):
            print(f"  síť: {mesh.report['nodes']} uzlů, {mesh.report['elements']} prvků, "
                  f"min SJ {mesh.report['min_scaled_jacobian']:.3f}, čas {dt:.1f} s")
        else:
            print(f"  čas {dt:.1f} s")
        for w in res.warnings:
            print(f"  varování: {w}")
        for c in checks:
            flag = "OK  " if c.ok else "CHYBA"
            print(f"  [{flag}] {c.name:<52} {c.value:12.6g} {c.unit:<4} ref {c.ref:12.6g}  "
                  f"odch. {100 * c.error:6.3f} % (limit {100 * c.tol:g} %)")
            all_ok &= c.ok
        rep = mesh.report if mesh is not None and getattr(mesh, "report", None) else {}
        report[key] = {
            "title": case.title, "time_s": dt, "nodes": rep.get("nodes"),
            "elements": rep.get("elements"), "mesh_size_mm": _mesh_size(study), "warnings": res.warnings,
            "checks": [{"name": c.name, "value": float(c.value), "ref": float(c.ref), "error": float(c.error),
                        "tol": c.tol, "unit": c.unit, "ok": bool(c.ok)} for c in checks],
        }
        _write_report(report)
    _write_report(report)
    print("\nVÝSLEDEK:", _verdict(all_ok, len(keys), len(not_run)))
    if not_run:
        print(f"{mystran.NOT_RUN}: {', '.join(not_run)}")
    return all_ok


def _write_report(report: dict) -> None:
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _mesh_size(study) -> float | None:
    """Global element size of the study's mesh [mm] (None where the study has none, e.g. a mechanism)."""
    size = getattr(getattr(study, "mesh", None), "size", None)
    try:
        return float(size) if size else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    args = sys.argv[1:]
    merge = "--merge" in args
    if merge:
        args.remove("--merge")
    backend = "ccx"
    if "--backend" in args:
        i = args.index("--backend")
        backend = args[i + 1]
        del args[i:i + 2]
    keys = args or list(C.CASES)
    sys.exit(0 if run(keys, backend, merge) else 1)

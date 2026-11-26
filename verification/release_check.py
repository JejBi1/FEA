"""Release check of FEA Studio: the checks before a release in a fixed order, with a summary in verification/out/release_check.*

    python -m verification.release_check [--skip pytest i18n help gui cases] [--only STAGE …] [--package]
                                         [--cases V1 T1 …] [--gui-lang cs en] [--gui-timeout 900] [--fail-fast]

Stages (each a separate process; output in verification/out/release_check/<stage>.log):
  pytest   pytest -m "not slow and not very_slow" (the fast tests)
  i18n     python -m fea_core.i18n.extract --todo …: no English text without a Czech translation (exact, templates)
  help     the Czech help pages translated from the current English ones (fea_gui.help_pdf --stale = empty)
  gui      the GUI smoke tests in Czech and English (tests/gui/run_all.py)
  cases    verification.run_all with the fixed list verification.tiers.RELEASE_CASES (no very_slow case)
  package  (only with --package) installer/make_package.py --check-only: the anonymization scan of the package content
Results: release_check.json (machine), release_check.txt and release_check.html (short summary). Exit code 0 = all run
stages passed, 1 = a stage failed, 2 = wrong arguments. By default the run continues after a failed stage (the summary shows all);
`--fail-fast` stops at the first failure.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification" / "out"
LOGS = OUT / "release_check"
STAGES = ("pytest", "i18n", "help", "gui", "cases", "package")
DEFAULT_SKIP = {"package"}


def _env() -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run(name: str, cmd: list[str], timeout: float | None = None) -> tuple[int | None, str, float]:
    """Command → (exit code | None on time-out, output, seconds); output also in LOGS/<name>.log."""
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=_env(), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        code, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        code = None
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += f"\n*** time-out after {timeout:.0f} s"
    (LOGS / f"{name}.log").write_text(f"$ {' '.join(cmd)}\n\n{out}", encoding="utf-8")
    return code, out, time.time() - t0


def _tail(text: str, n: int = 8) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


# ------------------------------------------------------------------ stages: each returns (ok, short detail)
def stage_pytest(a) -> tuple[bool, str]:
    code, out, _ = _run("pytest", [sys.executable, "-m", "pytest", "-q", "-m", "not slow and not very_slow", "-p", "no:cacheprovider"],
                        a.pytest_timeout)
    return code == 0, _tail(out, 3)


def stage_i18n(a) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="fea_rc_") as td:
        todo = Path(td) / "todo.json"
        code, out, _ = _run("i18n", [sys.executable, "-m", "fea_core.i18n.extract", "--todo", str(todo), "--where"], 600)
        if code != 0 or not todo.is_file():
            return False, _tail(out)
        data = json.loads(todo.read_text(encoding="utf-8"))
    missing = {k: v for kind in ("exact", "templates") for k, v in data.get(kind, {}).items()}
    if missing:
        first = "; ".join(f"{k!r} ({w})" if w else repr(k) for k, w in list(missing.items())[:5])
        return False, f"{len(missing)} texts without a Czech translation: {first}"
    return True, "catalogue complete"


def stage_help(a) -> tuple[bool, str]:
    code, out, _ = _run("help", [sys.executable, "-c", "import json; from fea_gui.help_pdf import stale_translations as s; "
                                 "print('STALE=' + json.dumps(s()))"], 300)
    line = next((l for l in out.splitlines() if l.startswith("STALE=")), None)
    if code != 0 or line is None:
        return False, _tail(out)
    stale = json.loads(line[6:])
    return not stale, ("Czech help up to date" if not stale else f"{len(stale)} outdated Czech pages: {', '.join(stale[:8])}")


def stage_gui(a) -> tuple[bool, str]:
    summary = LOGS / "gui_summary.json"
    cmd = [sys.executable, str(ROOT / "tests" / "gui" / "run_all.py"), "--lang", *a.gui_lang, "--timeout", str(a.gui_timeout),
           "--json", str(summary)]
    if a.gui_only:
        cmd += ["--only", *a.gui_only]
    code, out, _ = _run("gui", cmd, None)
    try:
        s = json.loads(summary.read_text(encoding="utf-8"))
        bad = [f"{r['script']} [{r['lang']}] {r['status']}" for r in s["results"] if r["status"] != "pass"]
        return code == 0, f"{s['passed']}/{s['total']} passed" + (": " + "; ".join(bad[:6]) if bad else "")
    except (OSError, ValueError, KeyError):
        return False, _tail(out)


def stage_cases(a) -> tuple[bool, str]:
    from verification import tiers
    keys = a.cases or list(tiers.RELEASE_CASES)
    slow = [k for k in keys if k in tiers.VERY_SLOW]
    if slow and not a.cases:
        return False, f"RELEASE_CASES contains very_slow cases: {', '.join(slow)}"
    code, out, _ = _run("cases", [sys.executable, "-m", "verification.run_all", *keys], a.cases_timeout)
    rep = OUT / "report.json"
    failed = []
    try:
        data = json.loads(rep.read_text(encoding="utf-8"))
        for k in keys:
            r = data.get(k)
            if r is None:
                failed.append(f"{k}: not run")
            elif "error" in r:
                failed.append(f"{k}: {r['error'][:80]}")
            elif not all(c.get("ok") for c in r.get("checks", [])):
                failed.append(f"{k}: " + ", ".join(c["name"] for c in r["checks"] if not c.get("ok"))[:100])
    except (OSError, ValueError):
        failed.append("report.json unreadable")
    ok = code == 0 and not failed
    return ok, (f"{len(keys)} cases passed" if ok else f"{len(failed)} of {len(keys)} failed: " + "; ".join(failed[:6]))


def stage_package(a) -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "installer" / "make_package.py"), "--check-only", "--allow-dirty"]
    if a.package_no_examples:
        cmd.append("--no-examples")
    code, out, _ = _run("package", cmd, 7200)
    return code == 0, _tail(out, 4)


FUNCS = {"pytest": stage_pytest, "i18n": stage_i18n, "help": stage_help, "gui": stage_gui, "cases": stage_cases,
         "package": stage_package}


# ------------------------------------------------------------------ summary
def write_summary(rows: list[dict], t_total: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = all(r["status"] in ("pass", "skipped") for r in rows)
    meta = {"ok": ok, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(t_total, 1), "stages": rows}
    try:
        sys.path.insert(0, str(ROOT / "installer"))
        import upgrade
        meta["version"] = upgrade.program_version(ROOT)
        meta["build"] = upgrade.make_build_id(ROOT)
    except Exception:  # noqa: BLE001
        pass
    (OUT / "release_check.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [f"FEA Studio {meta.get('version', '?')} release check {meta['time']} – {'PASSED' if ok else 'FAILED'} "
             f"({t_total / 60:.1f} min)"]
    for r in rows:
        lines.append(f"  {r['status'].upper():8s} {r['stage']:8s} {r['seconds']:7.0f} s  {r['detail'].splitlines()[0] if r['detail'] else ''}")
    text = "\n".join(lines) + "\n"
    (OUT / "release_check.txt").write_text(text, encoding="utf-8")
    color = {"pass": "#1a7f37", "fail": "#cf222e", "skipped": "#6e7781"}
    body = "".join(
        f"<tr><td>{html.escape(r['stage'])}</td><td style='color:{color.get(r['status'], '#000')};font-weight:600'>"
        f"{r['status']}</td><td>{r['seconds']:.0f} s</td><td><pre>{html.escape(r['detail'])}</pre></td>"
        f"<td>{html.escape(r.get('log', ''))}</td></tr>" for r in rows)
    page = (f"<!doctype html><meta charset='utf-8'><title>Release check</title>"
            f"<style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#1f2328}}td,th{{border:1px solid #d0d7de;"
            f"padding:4px 8px;vertical-align:top}}table{{border-collapse:collapse}}pre{{margin:0;white-space:pre-wrap}}</style>"
            f"<h1>FEA Studio {html.escape(str(meta.get('version', '')))} – release check: "
            f"<span style='color:{color['pass' if ok else 'fail']}'>{'passed' if ok else 'FAILED'}</span></h1>"
            f"<p>{meta['time']}, {t_total / 60:.1f} min, build {html.escape(str(meta.get('build', '')))}</p>"
            f"<table><tr><th>Stage</th><th>Result</th><th>Time</th><th>Detail</th><th>Log</th></tr>{body}</table>")
    (OUT / "release_check.html").write_text(page, encoding="utf-8")
    print("\n" + text + f"Summary: {OUT / 'release_check.html'}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip", nargs="+", default=[], choices=STAGES, help="stages to leave out")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    ap.add_argument("--package", action="store_true", help="also the package anonymization stage (make_package --check-only)")
    ap.add_argument("--package-no-examples", action="store_true", help="package stage without the example projects (faster)")
    ap.add_argument("--cases", nargs="+", help="verification cases instead of verification.tiers.RELEASE_CASES")
    ap.add_argument("--gui-lang", nargs="+", default=["cs", "en"], choices=["cs", "en"])
    ap.add_argument("--gui-only", nargs="+", help="only GUI smoke tests whose name contains one of these")
    ap.add_argument("--gui-timeout", type=float, default=900, help="time limit per GUI script and language [s]")
    ap.add_argument("--pytest-timeout", type=float, default=7200, help="time limit of the whole pytest stage [s]")
    ap.add_argument("--cases-timeout", type=float, default=7200, help="time limit of the cases stage [s]")
    ap.add_argument("--fail-fast", action="store_true", help="stop at the first failed stage")
    a = ap.parse_args(argv)
    if a.cases:
        from verification import cases as C
        unknown = [k for k in a.cases if k not in C.CASES]
        if unknown:
            print(f"unknown verification cases: {', '.join(unknown)}", file=sys.stderr)
            return 2
    run = set(a.only) if a.only else set(STAGES) - DEFAULT_SKIP | ({"package"} if a.package else set())
    run -= set(a.skip)
    rows, t_all, failed = [], time.time(), False
    for st in STAGES:
        if st not in run or (failed and a.fail_fast):
            rows.append({"stage": st, "status": "skipped", "seconds": 0.0, "detail": "", "log": ""})
            continue
        print(f"== {st} …", flush=True)
        t0 = time.time()
        try:
            ok, detail = FUNCS[st](a)
        except Exception as e:  # noqa: BLE001  (one broken stage must not hide the others)
            ok, detail = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        failed |= not ok
        rows.append({"stage": st, "status": "pass" if ok else "fail", "seconds": round(dt, 1), "detail": detail,
                     "log": str(LOGS / f"{st}.log") if (LOGS / f"{st}.log").exists() else ""})
        print(f"   {'pass' if ok else 'FAIL'} ({dt:.0f} s) {detail.splitlines()[0] if detail else ''}", flush=True)
    write_summary(rows, time.time() - t_all)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

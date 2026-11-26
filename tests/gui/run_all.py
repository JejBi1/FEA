"""Runs every GUI smoke test (tests/gui/gui_smoke.py, gui_*_smoke.py) through tests/gui/hidden_run.py (windows off the screen, nothing pops up)
in Czech and English, each in its own process with a time limit, and prints a pass / fail summary.

    python tests/gui/run_all.py [--lang cs en] [--only NAME …] [--skip NAME …] [--timeout 900] [--out DIR] [--json FILE] [--list]

NAME = part of the script name (`--only tour popout` → gui_tour_smoke.py, gui_popout_smoke.py). OUT_DIR of a script =
scratch/gui_runs/<timestamp>/<lang>/<name>/ (snapshots), its output in `<OUT_DIR>/output.log`. Projects: the scripts use their
defaults (examples/*.fea relative to the project root = the working folder of the run). Exit code 0 = all passed.
Do not run while the program GUI is open with an MCP session you need: the scripts open their own windows (off the screen).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUI = ROOT / "tests" / "gui"
HIDDEN = GUI / "hidden_run.py"


def scripts(only: list[str] | None = None, skip: list[str] | None = None) -> list[Path]:
    out = sorted(GUI.glob("gui_*smoke.py"))          # incl. gui_smoke.py (the whole workflow)
    if only:
        out = [p for p in out if any(o.lower() in p.stem.lower() for o in only)]
    if skip:
        out = [p for p in out if not any(s.lower() in p.stem.lower() for s in skip)]
    return out


def short(p: Path) -> str:
    return p.stem.removeprefix("gui_").removesuffix("_smoke")


def run_one(script: Path, lang: str, out_dir: Path, timeout: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FEA_LANG=lang, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
    env.pop("QT_QPA_PLATFORM", None)            # offscreen crashes (the OCCT 3D window needs a native window)
    env.setdefault("PROBE_NO_IPC", "1")
    log = out_dir / "output.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        try:
            p = subprocess.run([sys.executable, str(HIDDEN), str(script), str(out_dir)], cwd=ROOT, env=env, stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            code, status = p.returncode, ("pass" if p.returncode == 0 else "fail")
        except subprocess.TimeoutExpired:
            code, status = None, "timeout"
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
    return {"script": script.name, "lang": lang, "status": status, "code": code, "seconds": round(time.time() - t0, 1),
            "out_dir": str(out_dir), "tail": tail}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", nargs="+", default=["cs", "en"], choices=["cs", "en"])
    ap.add_argument("--only", nargs="+", help="only scripts whose name contains one of these")
    ap.add_argument("--skip", nargs="+", help="leave out scripts whose name contains one of these")
    ap.add_argument("--timeout", type=float, default=900, help="time limit per script and language [s] (default 900)")
    ap.add_argument("--out", help="output folder (default scratch/gui_runs/<timestamp>)")
    ap.add_argument("--json", help="write the summary as JSON here")
    ap.add_argument("--list", action="store_true", help="only list the scripts")
    a = ap.parse_args(argv)
    todo = scripts(a.only, a.skip)
    if a.list:
        for p in todo:
            print(p.name)
        return 0
    if not todo:
        print("no GUI smoke test matches", file=sys.stderr)
        return 2
    base = Path(a.out) if a.out else ROOT / "scratch" / "gui_runs" / time.strftime("%Y%m%d-%H%M%S")
    results = []
    for lang in a.lang:
        for s in todo:
            print(f"[{lang}] {s.name} …", end=" ", flush=True)
            r = run_one(s, lang, base / lang / short(s), a.timeout)
            print(f"{r['status']} ({r['seconds']:.0f} s)", flush=True)
            if r["status"] != "pass":
                for line in r["tail"]:
                    print(f"      {line}")
            results.append(r)
    bad = [r for r in results if r["status"] != "pass"]
    print(f"\nGUI smoke tests: {len(results) - len(bad)}/{len(results)} passed ({', '.join(a.lang)}), output in {base}")
    for r in bad:
        print(f"  {r['status'].upper():8s} [{r['lang']}] {r['script']}  → {r['out_dir']}{os.sep}output.log")
    summary = {"out": str(base), "passed": len(results) - len(bad), "total": len(results), "results": results}
    (base / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

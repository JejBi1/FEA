"""Developer tool: recreates the version locks installer/locks/<platform>.txt (the whole set of packages of all modules).

    python installer/make_locks.py [--uv path/to/uv] [--upgrade] [--no-hashes]

A lock = the exact versions of all packages including dependencies for Python 3.11 (uv pip compile, works also for another platform
than the one it runs on) WITH the SHA-256 digests of their files on PyPI (`--generate-hashes`). The installer uses it as constraints
when installing any subset of modules and installs with `uv pip sync --require-hashes` (fea_setup.sync_packages), so every
user gets the same verified versions and files. The packages installed without dependencies (`pip_nodeps`) are resolved with
`--no-deps` and get their digests the same way. Needs access to PyPI.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from upgrade import parse_lock  # noqa: E402  (installer/upgrade.py – shared with fea_setup.py)
PLATFORMS = {"windows-x86_64": "x86_64-pc-windows-msvc", "linux-x86_64": "x86_64-manylinux_2_35"}
HEAD = ("# Version lock for {name} (Python 3.11) - created by installer/make_locks.py from fea_core/modules.json.\n"
        "# Every package with the SHA-256 digests of its files on PyPI (uv pip sync --require-hashes). The packages of\n"
        "# pip_nodeps (sectionproperties, pyNastran, ...) are installed without dependencies.\n")


def format_lock(entries: dict[str, tuple[str, list[str]]]) -> str:
    lines = []
    for _, (req, hashes) in sorted(entries.items(), key=lambda kv: kv[1][0].lower()):
        if hashes:
            lines.append(req + " \\\n" + " \\\n".join(f"    --hash={h}" for h in hashes))
        else:
            lines.append(req)
    return "\n".join(lines) + "\n"


def compile_reqs(uv: str, reqs: list[str], plat: str, constraints: Path | None, hashes: bool, no_deps: bool, td: Path,
                 tag: str) -> dict[str, tuple[str, list[str]]]:
    rin, out = td / f"{tag}.in", td / f"{tag}.txt"
    rin.write_text("\n".join(reqs) + "\n", encoding="utf-8")
    cmd = [uv, "pip", "compile", str(rin), "--python-version", "3.11", "--python-platform", plat, "--no-header",
           "--no-annotate", "--no-config", "-o", str(out), "-q"]
    if hashes:
        cmd.append("--generate-hashes")
    if no_deps:
        cmd.append("--no-deps")
    if constraints is not None:
        cmd += ["-c", str(constraints)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return parse_lock(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv", default=str(ROOT / "tools" / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")))
    ap.add_argument("--upgrade", action="store_true", help="re-resolve the versions (otherwise the current lock is kept)")
    ap.add_argument("--no-hashes", action="store_true", help="without the SHA-256 digests (the installer then cannot require them)")
    ap.add_argument("--only", choices=list(PLATFORMS), action="append", help="only this platform (repeatable)")
    args = ap.parse_args()
    mods = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))["modules"]
    reqs = sorted({r for m in mods.values() for r in m.get("pip", [])})
    nodeps = sorted({r for m in mods.values() for r in m.get("pip_nodeps", [])})
    (ROOT / "installer" / "locks").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tdir:
        td = Path(tdir)
        for name, plat in PLATFORMS.items():
            if args.only and name not in args.only:
                continue
            out = ROOT / "installer" / "locks" / f"{name}.txt"
            cons = None
            if not args.upgrade and out.exists():             # the existing verified versions are kept (constraints without digests)
                old = parse_lock(out.read_text(encoding="utf-8"))
                cons = td / f"{name}.constraints.txt"
                cons.write_text("\n".join(req for req, _ in old.values()) + "\n", encoding="utf-8")
            entries = compile_reqs(args.uv, reqs, plat, cons, not args.no_hashes, False, td, name)
            nd = compile_reqs(args.uv, nodeps, plat, None, not args.no_hashes, True, td, name + ".nodeps")
            for k, v in nd.items():
                if k in entries and entries[k][0].split(";")[0].strip() != v[0].split(";")[0].strip():
                    raise SystemExit(f"{name}: {v[0]} (pip_nodeps) differs from the resolved {entries[k][0]}")
                entries.setdefault(k, v)
            out.write_text(HEAD.format(name=name) + format_lock(entries), encoding="utf-8", newline="\n")
            hashed = sum(1 for _, h in entries.values() if h)
            print(f"{name}: {len(entries)} packages, {hashed} with digests")
    return 0


if __name__ == "__main__":
    sys.exit(main())

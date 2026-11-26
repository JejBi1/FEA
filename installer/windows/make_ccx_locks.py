#!/usr/bin/env python3
"""Regenerates installer/windows/ccx_toolchain.lock - the exact MSYS2 packages (URL, SHA-256 from the repository databases) of the toolchain that
build_ccx.ps1 / build_ccx.sh unpack to build CalculiX for Windows.

The package lists are read from build_ccx.sh (MSYS_SPEC = tools of the MSYS2 environment, UCRT_SPEC = compilers and libraries of the UCRT64
environment). build_ccx.sh does not resolve anything: it downloads exactly the packages of the lock, checks every SHA-256 and unpacks them into the
MSYS2 base system (pacman is not used - no keyring, no administrator rights). A new version of a package = a new lock = a reviewed change.

    python installer/windows/make_ccx_locks.py [--base-dir PATH_TO_msys64]

--base-dir = an unpacked msys2-base tarball (var/lib/pacman/local lists the packages it already contains - they are not locked again); without it
the base is taken as the minimal set that the script needs (packages are added to the lock if the database says so). Needs `tar` with zstd support
(Windows 10+ tar.exe, bsdtar) to read the databases.

Lock file format: `#` lines, then `<kind> <sha256> <URL>` per file (kind: base | msys | ucrt64), in the order the packages are unpacked.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "build_ccx.sh"
LOCK = HERE / "ccx_toolchain.lock"
REPO = {"msys": "https://repo.msys2.org/msys/x86_64/", "ucrt64": "https://repo.msys2.org/mingw/ucrt64/"}
DB = {"msys": REPO["msys"] + "msys.db", "ucrt64": REPO["ucrt64"] + "ucrt64.db"}
BASE_URL = "https://repo.msys2.org/distrib/x86_64/"
PREFIX = "mingw-w64-ucrt-x86_64-"


def read_var(name: str, text: str | None = None) -> str:
    """The value of `NAME="..."` in build_ccx.sh."""
    text = SCRIPT.read_text(encoding="utf-8") if text is None else text
    m = re.search(rf'^{name}="([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit(f"{SCRIPT.name}: {name} not found")
    return m.group(1)


def parse_lock(path: Path) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """({header key: value}, [(kind, sha256, url)]) of a lock file. ValueError for a malformed file."""
    head: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r"# (msys_spec|ucrt_spec|base): (.*)$", line)
        if m:
            head[m.group(1)] = m.group(2)
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or parts[0] not in ("base", "msys", "ucrt64") or not re.fullmatch(r"[0-9a-f]{64}", parts[1]) \
                or not parts[2].startswith("https://repo.msys2.org/"):
            raise ValueError(f"{path.name}:{n}: expected '<base|msys|ucrt64> <sha256> <https://repo.msys2.org/... URL>'")
        rows.append((parts[0], parts[1], parts[2]))
    return head, rows


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fea-studio-lock"}), timeout=120) as r:
        return r.read()


def read_db(kind: str, tmp: Path) -> dict[str, dict]:
    """The package entries of a pacman sync database: {name: {...}} (the newest version of every name)."""
    f = tmp / f"{kind}.db"
    f.write_bytes(fetch(DB[kind]))
    out = tmp / f"db_{kind}"
    out.mkdir()
    subprocess.run(["tar", "-xf", str(f), "-C", str(out)], check=True)
    pk: dict[str, dict] = {}
    for d in sorted(out.iterdir()):
        text = (d / "desc").read_text(encoding="utf-8")
        fields = {m.group(1): m.group(2).strip().splitlines() for m in re.finditer(r"%(\w+)%\n(.*?)(?:\n\n|\Z)", text, re.S)}
        e = {"name": fields["NAME"][0], "version": fields["VERSION"][0], "file": fields["FILENAME"][0], "sha256": fields["SHA256SUM"][0],
             "depends": fields.get("DEPENDS", []), "provides": fields.get("PROVIDES", []), "kind": kind}
        pk[e["name"]] = e
    return pk


def installed(base_dir: Path | None) -> dict[str, str]:
    """{name: version} of the packages of the unpacked msys2-base tarball."""
    res: dict[str, str] = {}
    if base_dir is None:
        return res
    for d in (base_dir / "var" / "lib" / "pacman" / "local").glob("*"):
        m = re.fullmatch(r"(.+)-([^-]+-\d+)", d.name)
        if m:
            res[m.group(1)] = m.group(2)
    return res


def closure(wanted: list[str], pk: dict[str, dict], have: dict[str, str], other: dict[str, dict] | None = None) -> list[dict]:
    """The packages (not yet in `have`) that `wanted` needs, dependencies first (depth-first order of the dependency lists)."""
    provides: dict[str, str] = {}
    for e in pk.values():
        provides.setdefault(e["name"], e["name"])
        for p in e["provides"]:
            provides.setdefault(re.split(r"[<>=]", p)[0], e["name"])
    have_provided = set(have)
    order: list[dict] = []
    seen: set[str] = set()

    def visit(dep: str, parent: str) -> None:
        name = re.split(r"[<>=]", dep)[0]
        if name in have_provided or name in seen:
            return
        real = provides.get(name)
        if real is None:
            if other is not None and name in other:               # an msys package needed by a mingw one is the caller's business
                return
            raise SystemExit(f"{parent}: dependency '{dep}' is in no database")
        if real in have_provided or real in seen:
            return
        seen.add(real)
        seen.add(name)
        for d in pk[real]["depends"]:
            visit(d, real)
        order.append(pk[real])

    for w in wanted:
        visit(w, "spec")
    return order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-dir", type=Path, help="unpacked msys2-base (the msys64 folder)")
    ap.add_argument("--base", help="file name of the msys2-base tarball (default: the one in the current lock)")
    a = ap.parse_args()
    text = SCRIPT.read_text(encoding="utf-8")
    msys_spec, ucrt_spec = read_var("MSYS_SPEC", text), read_var("UCRT_SPEC", text)
    have = installed(a.base_dir)
    base_row = None
    if LOCK.is_file():
        _, rows = parse_lock(LOCK)
        base_row = next((r for r in rows if r[0] == "base"), None)
    if a.base:
        digest = hashlib.sha256(fetch(BASE_URL + a.base)).hexdigest()
        base_row = ("base", digest, BASE_URL + a.base)
    if base_row is None:
        raise SystemExit("no base tarball: pass --base msys2-base-x86_64-YYYYMMDD.tar.xz")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        msys = read_db("msys", tmp)
        ucrt = read_db("ucrt64", tmp)
        m_pk = closure(msys_spec.split(), msys, have)
        u_pk = closure([PREFIX + x for x in ucrt_spec.split()], ucrt, have, other=msys)
        m_have = dict(have)
        for e in m_pk:
            m_have[e["name"]] = e["version"]
        # msys packages that mingw ones depend on (none today) are flagged instead of guessed
        rows = [base_row] + [("msys", e["sha256"], REPO["msys"] + e["file"]) for e in m_pk] \
            + [("ucrt64", e["sha256"], REPO["ucrt64"] + e["file"]) for e in u_pk]
    lines = [f"# MSYS2 toolchain of installer/windows/build_ccx.sh - generated by installer/windows/make_ccx_locks.py ({datetime.date.today().isoformat()}); "
             "do not edit by hand",
             f"# msys_spec: {msys_spec}", f"# ucrt_spec: {ucrt_spec}", "# <base|msys|ucrt64> <sha256> <url>"]
    lines += [f"{k} {s} {u}" for k, s, u in rows]
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{LOCK.name}: {len(m_pk)} msys + {len(u_pk)} ucrt64 packages (msys_spec = {msys_spec}; ucrt_spec = {ucrt_spec})")
    return 0


if __name__ == "__main__":
    shutil.which("tar") or sys.exit("tar (bsdtar with zstd) not found")
    sys.exit(main())

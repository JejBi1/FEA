#!/usr/bin/env python3
"""Writes the licence list and the licence texts of the Rust crates behind libfea_faer.a (F17 stage D; standard library only, `cargo` optional).

    python installer/faer_ccx/make_crate_licences.py [--archive DIR] [--check]

Reads installer/faer_ccx/Cargo.lock; every crate (name, version, SHA-256 of the .crate file) is taken from DIR/crates/<name>-<version>.crate (DIR = the archive
that installer/windows/build_ccx.sh fills, default tools/winccx_build/archive; a missing file is downloaded from static.crates.io) and checked against the digest
of the lock. From the crate itself it takes the licence expression of its Cargo.toml and every licence file in its root.

Output (committed; the packages carry it as licenses/external/faer/):
    installer/faer_ccx/licenses/CRATES.md               one row per crate: version, SPDX expression of the crate, the licence FEA Studio uses it under, whether it is linked
    installer/faer_ccx/licenses/crates/<crate>/<files>   the licence files of the crates that are linked into the program (and of the build-time ones: they are small)

The licence FEA Studio uses a crate under: CalculiX is GPL-2.0-only, so a crate that is dual licensed with the Apache-2.0 (which GPL-2.0-only cannot take) is used
under its MIT / BSD / Zlib alternative; a crate that offers nothing but Apache-2.0 would stop the script. `--check` only verifies that the files are up to date.
If `cargo` is on the PATH the crates that the library really links (normal dependencies for the Windows and Linux GNU targets) are marked; otherwise all are
marked "?".
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "licenses"
TARGETS = ("x86_64-pc-windows-gnu", "x86_64-unknown-linux-gnu")
LICENCE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|UNLICENSE|COPYRIGHT)", re.IGNORECASE)
# order of preference when a crate offers several licences; Apache-2.0 alone is not usable inside GPL-2.0-only CalculiX
PREFERRED = ("MIT", "BSD-2-Clause", "BSD-3-Clause", "Zlib", "Unlicense", "ISC")


MIT_BODY = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def crates() -> list[tuple[str, str, str]]:
    lock = tomllib.loads((HERE / "Cargo.lock").read_text(encoding="utf-8"))
    return sorted((p["name"], p["version"], p["checksum"]) for p in lock["package"] if p.get("source", "").startswith("registry+") and "checksum" in p)


def fetch(name: str, ver: str, digest: str, archive: Path) -> bytes:
    f = archive / "crates" / f"{name}-{ver}.crate"
    data = f.read_bytes() if f.is_file() else b""
    if hashlib.sha256(data).hexdigest() != digest:
        req = urllib.request.Request(f"https://static.crates.io/crates/{name}/{name}-{ver}.crate", headers={"User-Agent": "feastudio-licence-list"})
        with urllib.request.urlopen(req, timeout=60) as r:       # noqa: S310 - a fixed https URL
            data = r.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"{name} {ver}: SHA-256 of the download does not match Cargo.lock")
    return data


def linked_crates() -> set[str] | None:
    """Names of the crates that are normal (not build-time) dependencies for the GNU targets, or None without cargo."""
    names: set[str] = set()
    for target in TARGETS:
        try:
            out = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked", "--filter-platform", target],
                                 cwd=HERE, capture_output=True, text=True, encoding="utf-8", timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode:
            return None
        meta = json.loads(out.stdout)
        by_id = {n["id"]: n for n in meta["resolve"]["nodes"]}
        pk = {p["id"]: p for p in meta["packages"]}
        seen: set[str] = set()

        def is_macro(i: str) -> bool:
            return "proc-macro" in {k for tt in pk[i]["targets"] for k in tt["kind"]}

        def walk(i: str) -> None:
            for d in by_id[i]["deps"]:
                if not any(k["kind"] is None for k in d["dep_kinds"]):          # build / dev only
                    continue
                if d["pkg"] in seen:
                    continue
                seen.add(d["pkg"])
                if not is_macro(d["pkg"]):          # a proc-macro crate runs in the compiler: neither it nor what only it uses is linked
                    walk(d["pkg"])
        walk(meta["resolve"]["root"])
        names.update(pk[i]["name"] for i in seen if not is_macro(i))
    return names


def choose_one(expr: str) -> str:
    """One term without AND: the alternative that is used from `A OR B OR C` (or `A/B`)."""
    alts = [a.strip(" ()") for a in re.split(r"\s+OR\s+|/", expr)]
    for p in PREFERRED:
        if p in alts:
            return p
    if all(a.startswith("Apache-2.0") for a in alts):
        raise SystemExit(f"a crate offers only {expr}: not usable inside GPL-2.0-only CalculiX")
    return " OR ".join(alts)


def choose(expr: str) -> str:
    """The licence a crate is used under, from its SPDX expression: every term of an AND applies, of an OR the first that GPL-2.0-only can take."""
    return " AND ".join(choose_one(term.strip(" ()")) for term in re.split(r"\s+AND\s+", expr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ROOT / "tools" / "winccx_build" / "archive"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    archive = Path(a.archive)
    linked = linked_crates()
    rows, files = [], {}
    for name, ver, digest in crates():
        data = fetch(name, ver, digest, archive)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            cargo = tomllib.loads(tf.extractfile(f"{name}-{ver}/Cargo.toml").read().decode("utf-8"))
            expr = cargo.get("package", {}).get("license") or ("see " + cargo.get("package", {}).get("license-file", "LICENSE file"))
            used = choose(expr) if cargo.get("package", {}).get("license") else expr
            texts = {}
            for m in tf.getmembers():
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                if m.isfile() and "/" not in rel and LICENCE_FILE.match(rel):
                    texts[rel] = tf.extractfile(m).read()
            has_mit_text = any(b"Permission is hereby granted, free of charge" in c for c in texts.values())
            if not has_mit_text and used.split(" AND ")[0] == "MIT":
                # a crate that declares MIT but ships no licence file (the repository has one, the package does not): the standard text with the authors of Cargo.toml,
                # said so in the file
                authors = ", ".join(a.split("<")[0].strip() for a in cargo.get("package", {}).get("authors", [])) or "the authors of the crate"
                head = (f"The crate {name} {ver} declares the licence {expr} in its Cargo.toml but its package contains no MIT licence text. This is the standard text of the MIT",
                        "licence with the authors of the crate (made by installer/faer_ccx/make_crate_licences.py).", "",
                        "MIT License", "", f"Copyright (c) {authors}", "", "")
                texts["LICENSE-MIT.generated.txt"] = ("\n".join(head) + MIT_BODY).encode("utf-8")
        files[f"{name}-{ver}"] = texts
        flag = "?" if linked is None else ("yes" if name in linked else "no (build time / other platform)")
        rows.append((name, ver, expr, used, flag, digest))
    md = ["# Rust crates of libfea_faer.a (the faer solver inside ccx_spooles)", "",
          "Generated by `installer/faer_ccx/make_crate_licences.py` from `installer/faer_ccx/Cargo.lock` (the SHA-256 digest is that of the `.crate` file on crates.io).",
          "The library is a C interface (`installer/faer_ccx/src/lib.rs`, MIT) over faer 0.24.4 (MIT). CalculiX is GPL-2.0-only, which cannot take code that is only",
          "Apache-2.0 licensed: every crate below is either MIT / BSD / Zlib licensed or offers one of them next to the Apache-2.0, and FEA Studio uses it under that",
          "licence (column *used under*). The Rust standard library (MIT OR Apache-2.0; `compiler_builtins` also contains code under Apache-2.0 with the LLVM exception) is",
          "linked in too: see `../rust-std/` (installer/license_texts/rust-std).", "",
          "**faer 0.24.4 contains code derived from other projects**, whose licence files are in the crate and therefore in `crates/faer-0.24.4/`: the ports of AMD and COLAMD",
          "of SuiteSparse (BSD-3-Clause, Timothy A. Davis et al.; `COPYING.SUITE_SPARSE.AMD.BSD`, `COPYING.SUITE_SPARSE.COLAMD.BSD`) and of parts of LAPACK (BSD-3-Clause,",
          "`COPYING.LAPACK.BSD`) are inside the program (the ordering of the LU factorization and the minimum degree ordering that FEA Studio does not use by default);",
          "the algorithms ported from Eigen (MPL-2.0, `COPYING.EIGEN.MPL2`: the divide and conquer SVD) are in the dense decomposition modules that the program does not call and",
          "the linker removes. The crate itself ships no MIT licence file: the file `LICENSE-MIT.generated.txt` is the standard text with the author of the crate.", "",
          "| Crate | Version | Licence of the crate | Used under | Linked into the program | SHA-256 of the .crate |", "|---|---|---|---|---|---|"]
    for name, ver, expr, used, flag, digest in rows:
        md.append(f"| `{name}` | {ver} | {expr} | {used} | {flag} | `{digest}` |")
    md.append("")
    text = "\n".join(md) + "\n"
    if a.check:
        ok = (OUT / "CRATES.md").is_file() and (OUT / "CRATES.md").read_text(encoding="utf-8") == text
        for cr, texts in files.items():
            for fn, content in texts.items():
                p = OUT / "crates" / cr / fn
                ok = ok and p.is_file() and p.read_bytes() == content
        print("up to date" if ok else "OUT OF DATE")
        return 0 if ok else 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CRATES.md").write_text(text, encoding="utf-8", newline="\n")
    for cr, texts in files.items():
        d = OUT / "crates" / cr
        d.mkdir(parents=True, exist_ok=True)
        for fn, content in texts.items():
            (d / fn).write_bytes(content)
    print(f"{len(rows)} crates; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

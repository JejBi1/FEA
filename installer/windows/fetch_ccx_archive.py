#!/usr/bin/env python3
"""Fills and checks the local archive of everything that the Windows CalculiX build downloads (standard library only).

    python installer/windows/fetch_ccx_archive.py [--dir tools/winccx_build/archive] [--check]

The archive (default `tools/winccx_build/archive`, gitignored) holds, with the SHA-256 that is pinned in the repository:
  - the MSYS2 base tarball and the packages of the toolchain (`ccx_toolchain.lock`), the Rust toolchain of the faer library (`ccx_rust.lock`),
  - every crate of installer/faer_ccx/Cargo.lock (folder `crates/`) and the source of the Rust standard library (rust-src),
  - the sources of CalculiX, SPOOLES, ARPACK-ng and reference LAPACK, and the MSYS2 source packages of the run-time libraries that are linked into
    the program (GCC with libgcc / libgfortran / libgomp / libquadmath, mingw-w64 CRT, winpthreads, headers) – the `get_mirrors` lines of `build_ccx.sh`.
`build_ccx.ps1` / `build_ccx.sh` look into it before the network (a file with a wrong digest is ignored and downloaded again), so a build of the release
does not depend on repo.msys2.org, GitHub or netlib being reachable, and `make_package.py` ships the source packages from it. `SHA256SUMS` in the folder is
written from the pins (`sha256sum -c SHA256SUMS` checks it).

--check only verifies (exit code 1 when a file is missing or its digest differs); without it the missing files are downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "installer"))
import collect_licenses  # noqa: E402  (installer/collect_licenses.py: the pins of build_ccx.sh)


def _load_by_path(name: str, path: Path):
    """A module of this folder by its file and under a name of its own: installer/linux has a make_ccx_locks.py as well, so `import make_ccx_locks` finds
    whichever of the two a process imported first (the full test run did: the Linux reader was taken for the Windows lock)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


make_ccx_locks = _load_by_path("make_ccx_locks_windows", HERE / "make_ccx_locks.py")      # the lock reader (installer/windows/make_ccx_locks.py)

DEFAULT_DIR = ROOT / "tools" / "winccx_build" / "archive"
CHUNK = 1 << 20


def plan(root: Path = ROOT) -> list[tuple[str, str, list[str]]]:
    """[(file name, SHA-256, [URLs])] of everything the build needs: the toolchain of the lock, then the pinned sources of build_ccx.sh."""
    _, rows = make_ccx_locks.parse_lock(root / "installer" / "windows" / "ccx_toolchain.lock")
    out = [(url.rsplit("/", 1)[1], digest, [url]) for _kind, digest, url in rows]
    rust_lock = root / "installer" / "windows" / "ccx_rust.lock"             # the Rust toolchain of the faer library (the rust-src row is in the pins below)
    if rust_lock.is_file():
        for line in rust_lock.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] == "rust":
                out.append((parts[2].rsplit("/", 1)[1], parts[1], [parts[2]]))
    out += collect_licenses.build_ccx_win_sources(root)               # the sources of build_ccx.sh, rust-src and every crate of installer/faer_ccx/Cargo.lock
    names = [n for n, _, _ in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"the pins name a file twice: {', '.join(sorted(dup))}")
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify(folder: Path, items: list[tuple[str, str, list[str]]]) -> tuple[list[str], list[str], list[str]]:
    """(good, missing, wrong) file names of the archive against the pins."""
    good, missing, wrong = [], [], []
    for name, digest, _urls in items:
        p = folder / name
        if not p.is_file():
            missing.append(name)
        elif sha256_of(p) == digest:
            good.append(name)
        else:
            wrong.append(name)
    return good, missing, wrong


def download(urls: list[str], digest: str, dest: Path, opener=None) -> None:
    """Downloads `dest` from the first URL whose content has the SHA-256 `digest` (written to a .part file, moved when it is right)."""
    opener = opener or urllib.request.urlopen
    part = dest.with_name(dest.name + ".part")
    last = "no URL"
    for url in urls:
        print(f"  downloading {url}", flush=True)
        try:
            h = hashlib.sha256()
            req = urllib.request.Request(url, headers={"User-Agent": "fea-studio-archive"})
            with opener(req, timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(CHUNK):
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() == digest:
                part.replace(dest)
                return
            last = f"SHA-256 {h.hexdigest()} instead of {digest}"
        except (urllib.error.URLError, OSError) as e:
            last = str(e)
        print(f"    failed ({last}) - trying the next source", flush=True)
    part.unlink(missing_ok=True)
    raise SystemExit(f"ERROR: could not get {dest.name}: {last}")


def write_sums(folder: Path, items: list[tuple[str, str, list[str]]]) -> Path:
    """`SHA256SUMS` of the pinned files that are present (the format of sha256sum)."""
    lines = [f"{digest}  {name}" for name, digest, _ in sorted(items) if (folder / name).is_file()]
    path = folder / "SHA256SUMS"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="the archive folder (default tools/winccx_build/archive)")
    ap.add_argument("--check", action="store_true", help="only verify the archive against the pins, download nothing")
    a = ap.parse_args(argv)
    items = plan()
    a.dir.mkdir(parents=True, exist_ok=True)
    good, missing, wrong = verify(a.dir, items)
    if not a.check:
        by_name = {n: (d, u) for n, d, u in items}
        for name in wrong + missing:
            if name in wrong:
                print(f"  {name}: the digest does not match the pin - downloading again")
                (a.dir / name).unlink()
            digest, urls = by_name[name]
            (a.dir / name).parent.mkdir(parents=True, exist_ok=True)           # crates/ is a folder of the archive
            download(urls, digest, a.dir / name)
        good, missing, wrong = verify(a.dir, items)
    write_sums(a.dir, items)
    size = sum((a.dir / n).stat().st_size for n in good)
    print(f"archive {a.dir}: {len(good)} of {len(items)} files verified, {size / 1e6:.0f} MB"
          + (f"; missing: {', '.join(missing)}" if missing else "") + (f"; wrong digest: {', '.join(wrong)}" if wrong else ""))
    return 0 if not missing and not wrong else 1


if __name__ == "__main__":
    sys.exit(main())

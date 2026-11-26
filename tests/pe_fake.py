"""Synthetic 64-bit Windows executables and the prebuilt-CalculiX archive of installer/windows/build_ccx.ps1 for the packaging tests (no compiler needed).

`build_pe(imports)` = a minimal PE32+ file whose import directory names the given DLLs (what `make_package.pe_imports` reads);
`build_archive(path, ...)` = the zip that the Windows build writes: `ccx_spooles.exe` + `manifest.json`.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path

# the DLLs that the real build imports (checked with objdump -p on the binary of installer/windows/build_ccx.sh)
SYSTEM_IMPORTS = ["KERNEL32.dll", "api-ms-win-crt-stdio-l1-1-0.dll", "api-ms-win-crt-heap-l1-1-0.dll"]


def build_pe(imports: list[str], tail: bytes = b"") -> bytes:
    """A PE32+ (x64) file: DOS header, PE signature, COFF header, optional header with the import data directory, one section `.idata`."""
    sec_rva, sec_raw = 0x1000, 0x400
    n = len(imports)
    desc_size = 20 * (n + 1)
    names_off = desc_size                                       # the names follow the descriptors inside the section
    blob = bytearray(desc_size)
    pos = names_off
    name_rvas = []
    for dll in imports:
        name_rvas.append(sec_rva + pos)
        raw = dll.encode("ascii") + b"\0"
        blob += raw
        pos += len(raw)
    for i, rva in enumerate(name_rvas):                         # OriginalFirstThunk, TimeDateStamp, ForwarderChain, Name, FirstThunk
        struct.pack_into("<IIIII", blob, 20 * i, 0, 0, 0, rva, 0)
    blob += tail
    opt_size = 0xF0
    dos = bytearray(0x80)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, opt_size, 0x22)
    opt = bytearray(opt_size)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<II", opt, 120, sec_rva, desc_size)       # data directory 1 = the import table
    sec = struct.pack("<8sIIIIIIHHI", b".idata", len(blob), sec_rva, len(blob), sec_raw, 0, 0, 0, 0, 0xC0000040)
    head = bytes(dos) + b"PE\0\0" + coff + bytes(opt) + sec
    return head.ljust(sec_raw, b"\0") + bytes(blob)


ROOT = Path(__file__).resolve().parents[1]


def manifest(exe: bytes | None = None, root: Path = ROOT, **extra) -> bytes:
    """manifest.json of the build: with `exe` it carries the digests that make_package.windows_ccx checks (the executable, build_ccx.sh and ccx_toolchain.lock of the
    checkout `root`); `extra` overrides or adds keys (a wrong digest for a test of a mismatch)."""
    d = {"name": "ccx_spooles.exe", "calculix": "2.23", "platform": "windows-x86_64", "sources": []}
    if exe is not None:
        d["binary_sha256"] = hashlib.sha256(exe).hexdigest()
        d["build_script_sha256"] = hashlib.sha256((root / "installer" / "windows" / "build_ccx.sh").read_bytes()).hexdigest()
        d["toolchain_lock_sha256"] = hashlib.sha256((root / "installer" / "windows" / "ccx_toolchain.lock").read_bytes()).hexdigest()
        d["faer"] = {k: hashlib.sha256((root / rel).read_bytes()).hexdigest() for k, rel in (
            ("rust_lock_sha256", "installer/windows/ccx_rust.lock"), ("cargo_lock_sha256", "installer/faer_ccx/Cargo.lock"),
            ("faer_c_sha256", "installer/faer_ccx/ccx/faer.c"), ("faer_h_sha256", "installer/faer_ccx/ccx/faer.h"))}
    d.update(extra)
    return json.dumps(d).encode("utf-8")


def build_archive(path: Path, imports: list[str] | None = None, exe_tail: bytes = b"", members: dict[str, bytes] | None = None,
                  **manifest_extra) -> Path:
    """The archive of build_ccx.ps1 (`members` replaces the content: for the tests of a damaged archive; `manifest_extra` goes into the manifest, which describes the
    executable and the build scripts of this checkout)."""
    exe = build_pe(SYSTEM_IMPORTS if imports is None else imports, exe_tail)
    files = {"ccx_spooles.exe": exe, "manifest.json": manifest(exe, **manifest_extra)}
    files.update(members or {})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if data is not None:
                z.writestr(zipfile.ZipInfo(name, (2026, 11, 18, 0, 0, 0)), data)
    return path

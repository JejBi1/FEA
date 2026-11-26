"""Synthetic ELF64 files for the tests of the Linux CalculiX packaging (installer/make_package.py: DT_NEEDED of the binaries)."""
from __future__ import annotations

import io
import struct
import tarfile
from pathlib import Path


def fake_elf(needed: list[str] | tuple[str, ...] = (), tail: bytes = b"") -> bytes:
    """Smallest ELF64 little-endian file with one PT_LOAD over the whole file and a PT_DYNAMIC with DT_NEEDED entries; `tail` = payload bytes."""
    strtab, offs = b"\0", []
    for n in needed:
        offs.append(len(strtab))
        strtab += n.encode() + b"\0"
    base, ph_off = 0x1000, 64
    dyn_off = ph_off + 2 * 56
    n_dyn = len(offs) + 2
    str_off = dyn_off + 16 * n_dyn
    total = str_off + len(strtab) + len(tail)
    dyn = b"".join(struct.pack("<qQ", 1, o) for o in offs) + struct.pack("<qQ", 5, base + str_off) + struct.pack("<qQ", 0, 0)
    head = bytearray(64)
    head[:6] = b"\x7fELF\x02\x01"
    head[6] = 1
    struct.pack_into("<Q", head, 0x20, ph_off)
    struct.pack_into("<HH", head, 0x36, 56, 2)
    ph = struct.pack("<IIQQQQQQ", 1, 5, 0, base, base, total, total, 0x1000)
    ph += struct.pack("<IIQQQQQQ", 2, 6, dyn_off, base + dyn_off, base + dyn_off, len(dyn), len(dyn), 8)
    return bytes(head) + ph + dyn + strtab + tail


# what the real archive of installer/linux/build_ccx.sh looks like (names of the libraries as ldd prints them)
SPOOLES_LIBS = {"lib/libgomp.so.1": ["libc.so.6", "libpthread.so.0"], "lib/libquadmath.so.0": ["libc.so.6", "libm.so.6"]}
PASTIX_LIBS = {"lib/libopenblas.so.0": ["libgomp.so.1", "libm.so.6"], "lib/libscotch.so.7.0": ["libz.so.1", "libm.so.6"],
               "lib/libscotcherr.so.7.0": ["libc.so.6"], "lib/libhwloc.so.15": ["libm.so.6"], "lib/libz.so.1": ["libc.so.6"]}


def build_archive(path: Path, pastix: bool = True, extra: dict[str, bytes] | None = None, tail: bytes = b"/build/x",
                  skip: tuple[str, ...] = ()) -> Path:
    """tar.gz like the output of build_ccx.sh: ccx_spooles, [ccx_pastix] and lib/. `extra` = further members, `skip` = left out."""
    members: dict[str, bytes] = {"ccx_spooles": fake_elf(["libgomp.so.1", "libquadmath.so.0", "libm.so.6", "libmvec.so.1", "libc.so.6"], tail)}   # the real one needs libmvec (glibc)
    if pastix:
        members["ccx_pastix"] = fake_elf(["libopenblas.so.0", "libscotch.so.7.0", "libscotcherr.so.7.0", "libhwloc.so.15", "libgomp.so.1",
                                          "libquadmath.so.0", "libz.so.1", "libc.so.6"], tail)
    for name, need in {**SPOOLES_LIBS, **(PASTIX_LIBS if pastix else {})}.items():
        members[name] = fake_elf(need)
    members.update(extra or {})
    with tarfile.open(path, "w:gz") as t:
        for name in [n for n in members if n not in skip]:
            data = members[name]
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), 0o755
            t.addfile(ti, io.BytesIO(data))
    return path

"""Packaging of the Linux CalculiX (F13 step 2, review): the package carries ccx_spooles + the libraries it needs and NOTHING of PaStiX until the licence
question L4 is answered (--with-ccx-pastix is the explicit opt-in), an archive of an older build is refused, the build script builds ccx_spooles first and
pins everything it downloads (conda packages through the lock files), the installer accepts a package with ccx_spooles only. Fast – synthetic ELF files.
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from elf_fake import PASTIX_LIBS, SPOOLES_LIBS, build_archive, fake_elf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))
sys.path.insert(0, str(ROOT / "installer" / "linux"))

mp = importlib.import_module("make_package")
locks = importlib.import_module("make_ccx_locks")
SCRIPT = (ROOT / "installer" / "linux" / "build_ccx.sh").read_text(encoding="utf-8")


PASTIX_LIBS_NAMES = ["libopenblas.so.0", "libscotch.so.7.0", "libscotcherr.so.7.0", "libhwloc.so.15", "libgomp.so.1", "libz.so.1"]


def names(path: Path) -> list[str]:
    with tarfile.open(path) as t:
        return [m.name for m in t.getmembers()]


# ================================================================= the members of the package

def test_elf_needed_reads_dt_needed():
    assert mp.elf_needed(fake_elf(["libgomp.so.1", "libc.so.6", "libquadmath.so.0"])) == ["libgomp.so.1", "libc.so.6", "libquadmath.so.0"]
    assert mp.elf_needed(fake_elf([])) == []
    for bad in (b"", b"MZ" + b"\0" * 100, b"\x7fELF\x01\x01" + b"\0" * 100, fake_elf(["a.so"])[:90]):       # not ELF64 / truncated
        with pytest.raises(ValueError):
            mp.elf_needed(bad)


def test_default_package_has_ccx_spooles_and_its_libraries_only(tmp_path):
    """ccx_pastix (PaStiX LGPL-3.0 linked statically into the GPL-2.0-only CalculiX, L4) and the libraries only it needs stay out."""
    build_archive(tmp_path / "pre.tar.gz")                                           # ccx_spooles, ccx_pastix, all libraries
    mp.linux_ccx(tmp_path / "pre.tar.gz", tmp_path / "out.tar.gz")
    got = names(tmp_path / "out.tar.gz")
    assert got == ["ccx_spooles", "lib/libgomp.so.1", "lib/libquadmath.so.0"]
    assert "ccx_pastix" not in got and not any(re.search(r"openblas|scotch|hwloc|libz", n) for n in got)


def test_libraries_are_followed_through_dt_needed(tmp_path):
    """A library that a needed library needs belongs in the package too; glibc does not."""
    build_archive(tmp_path / "pre.tar.gz", pastix=False,
                  extra={"lib/libgomp.so.1": fake_elf(["libhelper.so.2", "libc.so.6"]), "lib/libhelper.so.2": fake_elf(["libc.so.6"]),
                         "lib/libunused.so.3": fake_elf([])})
    mp.linux_ccx(tmp_path / "pre.tar.gz", tmp_path / "out.tar.gz")
    got = names(tmp_path / "out.tar.gz")
    assert "lib/libhelper.so.2" in got and "lib/libunused.so.3" not in got and "lib/libc.so.6" not in got


def test_pastix_only_with_the_explicit_option(tmp_path, capsys):
    build_archive(tmp_path / "pre.tar.gz")
    mp.linux_ccx(tmp_path / "pre.tar.gz", tmp_path / "out.tar.gz", with_pastix=True)
    got = names(tmp_path / "out.tar.gz")
    assert "ccx_pastix" in got and "ccx_spooles" in got
    assert set(PASTIX_LIBS) <= set(got) and set(SPOOLES_LIBS) <= set(got)
    out = capsys.readouterr().out
    assert "do NOT publish" in out and "L4" in out                                  # the opt-in says why it is not for a release


def test_an_archive_of_an_older_build_is_refused(tmp_path):
    """Validated per mode: the binary and EVERY library it needs must be in the archive (the earlier code took any archive with ccx_spooles)."""
    build_archive(tmp_path / "nopx.tar.gz", pastix=False)                            # built without PaStiX (failed PaStiX part / older script)
    mp.linux_ccx(tmp_path / "nopx.tar.gz", tmp_path / "ok.tar.gz")                   # fine for the default mode
    with pytest.raises(SystemExit, match="ccx_pastix is missing"):
        mp.linux_ccx(tmp_path / "nopx.tar.gz", tmp_path / "x.tar.gz", with_pastix=True)
    build_archive(tmp_path / "nolib.tar.gz", skip=("lib/libquadmath.so.0",))          # a library of the binary is missing
    with pytest.raises(SystemExit, match="libquadmath.so.0, which the archive does not carry"):
        mp.linux_ccx(tmp_path / "nolib.tar.gz", tmp_path / "y.tar.gz")
    build_archive(tmp_path / "nolib2.tar.gz", skip=("lib/libhwloc.so.15",))
    mp.linux_ccx(tmp_path / "nolib2.tar.gz", tmp_path / "z.tar.gz")                  # the default mode does not need it …
    with pytest.raises(SystemExit, match="libhwloc.so.15"):
        mp.linux_ccx(tmp_path / "nolib2.tar.gz", tmp_path / "z2.tar.gz", with_pastix=True)         # … the PaStiX mode does
    build_archive(tmp_path / "nospooles.tar.gz", skip=("ccx_spooles",))
    with pytest.raises(SystemExit, match="ccx_spooles is missing"):
        mp.linux_ccx(tmp_path / "nospooles.tar.gz", tmp_path / "w.tar.gz")
    build_archive(tmp_path / "notelf.tar.gz", extra={"ccx_spooles": b"not an elf file"})
    with pytest.raises(SystemExit, match="not an ELF64"):
        mp.linux_ccx(tmp_path / "notelf.tar.gz", tmp_path / "v.tar.gz")


def test_a_home_folder_path_in_a_shipped_binary_is_an_error(tmp_path):
    build_archive(tmp_path / "bad.tar.gz", tail=b"/home/builder/fea/ccx_build/arpack-src/UTIL/dvout.f")
    with pytest.raises(SystemExit, match="home folder"):
        mp.linux_ccx(tmp_path / "bad.tar.gz", tmp_path / "out.tar.gz")
    # … but not in a member that is left out (ccx_pastix is not shipped by default)
    build_archive(tmp_path / "px.tar.gz", extra={"ccx_pastix": fake_elf(list(PASTIX_LIBS_NAMES), b"/home/builder/x/")})
    mp.linux_ccx(tmp_path / "px.tar.gz", tmp_path / "ok.tar.gz")


def test_make_package_option_text():
    text = (ROOT / "installer" / "make_package.py").read_text(encoding="utf-8")
    assert "--with-ccx-pastix" in text and "a.with_ccx_pastix" in text
    assert "is NOT in a package until the licence question L4" in text


# ================================================================= the build script

def test_ccx_spooles_is_built_before_the_optional_parts():
    """A failed PaStiX download / build must still leave the finished ccx_spooles: it is built and shipped first, the others run in `optional`."""
    i_spooles = SCRIPT.index('build_one ccx_spooles "-DUSE_MT=1 -DFEA_FAER=1')           # F17 A: ccx_spooles is compiled with USE_MT (multi-threaded factorization)
    i_ship = SCRIPT.index('ship "$OUT/ccx_spooles"')
    i_pastix = SCRIPT.index('optional "ccx_pastix (PaStiX)" pastix_part')
    assert i_spooles < i_ship < i_pastix
    assert "pastix.tar.gz" not in SCRIPT[:SCRIPT.index("pastix_part() {")]          # the PaStiX download is inside the optional part
    assert SCRIPT.count("build_one ccx_pastix") == 1 and SCRIPT.index("build_one ccx_pastix") > SCRIPT.index("pastix_part() {")
    assert "FEA_CCX_PASTIX" in SCRIPT and "FEA_BUILD_JOBS" in SCRIPT
    assert "-j\"$(nproc)\"" not in SCRIPT                                          # the number of jobs is configurable
    body = SCRIPT[SCRIPT.index("optional() {"):SCRIPT.index("# ---- ccx_pastix")]
    assert "set +e" in body and "set -euo pipefail" in body and "set -e\n" in body
    assert not re.search(r"^\s*if\s+optional|optional .*\|\|", SCRIPT, re.MULTILINE)   # `set -e` is ignored in the subshell inside an if / ||


def test_script_pins_what_it_downloads():
    """Sources by SHA-256 (get_mirrors), conda-forge packages through the lock files – nothing is resolved by micromamba at build time."""
    assert not re.search(r"micromamba\s+create[^\n]*\$\{?(TC_SPEC|DEPS_SPEC)", SCRIPT)
    assert "create_locked_env" in SCRIPT and "--file \"$list\"" in SCRIPT and "ccx_toolchain.lock" in SCRIPT and "ccx_deps.lock" in SCRIPT
    assert 'get_mirrors "$f" "$sum" "$url"' in SCRIPT                               # every package of the lock is checked against its SHA-256
    for spec in ("TC_SPEC", "DEPS_SPEC"):
        assert re.search(rf'^{spec}="[^"]+"', SCRIPT, re.MULTILINE), spec
    assert b"\r" not in (ROOT / "installer" / "linux" / "build_ccx.sh").read_bytes()


@pytest.mark.parametrize("name", ["TC_SPEC", "DEPS_SPEC"])
def test_lock_files_match_the_specification_of_the_script(name):
    path = locks.LOCKS[name]
    spec, rows = locks.parse_lock(path)
    assert spec == locks.read_spec(name), f"{path.name} is stale – run python installer/linux/make_ccx_locks.py"
    assert len(rows) > 20 and len({u for _, _, u in rows}) == len(rows)
    assert all(u.startswith("https://conda.anaconda.org/conda-forge/") for _, _, u in rows)
    assert b"\r" not in path.read_bytes()
    pk = " ".join(u.rsplit("/", 1)[1] for _, _, u in rows)
    wanted = ("libgomp-", "libgcc-", "gcc_impl_linux-64", "cmake-") if name == "TC_SPEC" else ("libopenblas-", "libscotch-", "libhwloc-", "libzlib-")
    for w in wanted:
        assert w in pk, w
    assert "sysroot_linux-64-2.28" in pk or name == "DEPS_SPEC"                       # the glibc 2.28 sysroot (binaries run from glibc 2.35)


def test_lock_parser_rejects_malformed_lines(tmp_path):
    p = tmp_path / "x.lock"
    ok = "a" * 64 + " " + "b" * 32 + " https://conda.anaconda.org/conda-forge/noarch/x-1.0-0.conda\n"
    p.write_text("# spec: x\n" + ok, encoding="utf-8")
    assert locks.parse_lock(p) == ("x", [("a" * 64, "b" * 32, "https://conda.anaconda.org/conda-forge/noarch/x-1.0-0.conda")])
    for bad in (ok.replace("https://conda.anaconda.org/conda-forge", "http://evil.example"), "short abc https://conda.anaconda.org/conda-forge/x.conda\n",
                ok.replace(".conda", ".zip")):
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            locks.parse_lock(p)


def test_build_script_syntax():
    """bash -n (when bash is there) and the PaStiX part is a function that `optional` can call."""
    import shutil
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    try:                                      # through stdin: the bash of Windows (WSL launcher) does not understand a Windows path
        r = subprocess.run([bash, "-n"], input=SCRIPT.encode("utf-8"), capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("bash does not start")
    if r.returncode != 0 and b"Windows Subsystem" in r.stderr + r.stdout:
        pytest.skip("bash is only the WSL launcher without a distribution")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-500:]
    assert "pastix_part() {" in SCRIPT


# ================================================================= the installer on Linux

def test_installer_takes_ccx_spooles_from_the_package(tmp_path, monkeypatch):
    """The Linux package carries ccx_spooles only (no ccx_pastix): ext ccx_static must be satisfied by it, unpacking it, without a build."""
    import fea_setup
    tools = tmp_path / "tools"
    (tools / "calculix").mkdir(parents=True)
    build_archive(tmp_path / "pre.tar.gz", pastix=False)
    mp.linux_ccx(tmp_path / "pre.tar.gz", tools / "calculix" / "ccx_2.23_linux_x86_64.tar.gz")
    monkeypatch.setattr(fea_setup, "IS_WIN", False)
    monkeypatch.setattr(fea_setup, "TOOLS", tools)
    built, said = [], []
    monkeypatch.setattr(fea_setup, "run", lambda cmd, **kw: built.append(cmd))       # the installer starts no process for ccx (it never compiles)
    monkeypatch.setattr(fea_setup, "ok", said.append)
    fea_setup.install_ext_ccx("ccx_static")
    d = tools / "calculix" / "ccx_2.23_linux"
    assert (d / "ccx_spooles").is_file() and (d / "lib" / "libgomp.so.1").is_file() and not built
    assert said and "ccx_spooles" in said[-1]
    said.clear()
    fea_setup.install_ext_ccx("ccx_static")                                          # a second run: already there
    assert said and "ccx_spooles" in said[0] and not built


def test_find_ccx_prefers_a_local_pastix_build_to_spooles(tmp_path, monkeypatch):
    from fea_core import config
    d = tmp_path / "calculix" / "ccx_2.23_linux"
    d.mkdir(parents=True)
    (d / f"ccx_spooles{config.EXE}").write_text("x")
    monkeypatch.setattr(config, "CCX_TOOLS", tmp_path / "calculix")
    monkeypatch.delenv("FEA_CCX", raising=False)
    assert config.find_ccx().name == f"ccx_spooles{config.EXE}"
    (d / f"ccx_pastix{config.EXE}").write_text("x")
    assert config.find_ccx().name == f"ccx_pastix{config.EXE}"

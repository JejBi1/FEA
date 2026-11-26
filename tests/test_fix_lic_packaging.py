"""Licence fixes of the packages: only the named CalculiX binaries, CalculiX / Gmsh sources with pinned digests, SOURCES.md, pyproject."""
from __future__ import annotations

import importlib
import re
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

from elf_fake import build_archive

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

mp = importlib.import_module("make_package")
cl = importlib.import_module("collect_licenses")


def test_linux_ccx_ships_only_the_named_binaries_and_their_libraries(tmp_path):
    build_archive(tmp_path / "pre.tar.gz", pastix=False)
    mp.linux_ccx(tmp_path / "pre.tar.gz", tmp_path / "out.tar.gz")
    with tarfile.open(tmp_path / "out.tar.gz") as t:
        assert [m.name for m in t.getmembers()] == ["ccx_spooles", "lib/libgomp.so.1", "lib/libquadmath.so.0"]
    build_archive(tmp_path / "no_spooles.tar.gz", pastix=False, skip=("ccx_spooles",))
    with pytest.raises(SystemExit, match="ccx_spooles is missing"):
        mp.linux_ccx(tmp_path / "no_spooles.tar.gz", tmp_path / "out2.tar.gz")
    build_archive(tmp_path / "with_px.tar.gz")
    mp.linux_ccx(tmp_path / "with_px.tar.gz", tmp_path / "out3.tar.gz", with_pastix=True)
    with tarfile.open(tmp_path / "out3.tar.gz") as t:
        assert "ccx_pastix" in [m.name for m in t.getmembers()]


def test_windows_packages_take_only_the_own_ccx():
    assert mp.WIN_CCX_MEMBERS == ("ccx_spooles.exe", "manifest.json")        # the archive of installer/windows/build_ccx.ps1 (tests/test_win_ccx.py)
    assert not hasattr(mp, "WIN_CCX_ZIP")                                      # the third-party zip of dhondt.de is not used any more


def test_build_script_archive_members():
    text = (ROOT / "installer" / "linux" / "build_ccx.sh").read_bytes()
    assert b"\r\n" not in text                                     # LF (shebang on Linux)
    line = [l for l in text.decode("utf-8").splitlines() if l.strip().startswith("members=()")]
    assert line and "ccx_spooles" in line[0] and "ccx_pastix" in line[0]


def test_source_digests_match_build_script():
    pinned = {n: (d, urls) for n, d, urls in cl.build_ccx_sources(ROOT)}
    assert {"ccx.src.tar.bz2", "spooles.tgz", "lapack.tar.gz", "arpack.tar.gz", "spooles.corr.tar.bz2"} <= set(pinned)
    assert not any(n.startswith("micromamba") for n in pinned)
    ccx = cl.SOURCE_ARCHIVES["ccx_2.23.src.tar.bz2"]
    assert pinned["ccx.src.tar.bz2"] == (ccx["sha256"], [ccx["url"]])
    assert len(pinned["spooles.tgz"][1]) == 2                        # netlib + Debian mirror
    for s in cl.SOURCE_ARCHIVES.values():
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"]) and s["url"].startswith("https://")
        assert s["packages"] in ("all", "offline")


def test_sources_md_and_not_shipped_notes(tmp_path):
    root = tmp_path / "proj"
    (root / "installer" / "locks").mkdir(parents=True)
    (root / "installer" / "linux").mkdir(parents=True)
    for name in ("build_ccx.sh", "ccx_gccsrc.lock", "ccx_toolchain.lock"):
        (root / "installer" / "linux" / name).write_bytes((ROOT / "installer" / "linux" / name).read_bytes())
    (root / "installer" / "windows").mkdir(parents=True)
    (root / "installer" / "windows" / "build_ccx.sh").write_bytes((ROOT / "installer" / "windows" / "build_ccx.sh").read_bytes())
    for plat in ("windows-x86_64", "linux-x86_64"):
        (root / "installer" / "locks" / f"{plat}.txt").write_text("pypdfium2==5.14.0\npyside6-addons==6.11.2\n", encoding="utf-8")
    (tmp_path / "venv" / "Lib" / "site-packages").mkdir(parents=True)
    out = tmp_path / "licenses"
    warnings = cl.collect(out, tmp_path / "venv", root=root)
    assert not warnings, warnings                                     # docs-module packages are not shipped → a note only
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "not in any package" in readme and "SOURCES.md" in readme
    src = (out / "SOURCES.md").read_text(encoding="utf-8")
    for s in cl.SOURCE_ARCHIVES.values():
        assert s["sha256"] in src and s["url"] in src
    for _, digest, _urls in cl.build_ccx_sources(root):
        assert digest in src
    for name, digest, _urls in cl.linux_sources(root):                       # the Linux packages carry every one of them (tests/test_linux_sources.py)
        assert digest in src and f"sources/{name}" in src, name
    for name, digest, _urls in cl.build_ccx_win_sources(root):             # the Windows CalculiX: every pinned source archive and the runtime sources
        assert digest in src and name in src
    assert any(cl.is_runtime_source(n) for n, _, _ in cl.build_ccx_win_sources(root))      # the GCC / mingw-w64 run-time sources are among the pins
    assert "ccx_static" not in src and "TAUCS" not in src and "PaStiX" not in src          # the third-party build is gone from the text
    assert "written offer" not in src.lower()                                   # the sources travel with the binaries (owner decision of 26. 11. 2026)


def test_add_sources_uses_the_pinned_archive(tmp_path, monkeypatch):
    import fea_setup
    got = []

    def fake_download(url, dest, what="", sha256=None):
        got.append((url, sha256))
        p = tmp_path / "dl" / Path(dest).name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"source")
        return p
    monkeypatch.setattr(fea_setup, "download", fake_download)
    monkeypatch.setattr(mp, "WIN_ARCHIVE", tmp_path / "no_archive")           # not the local archive of the Windows build
    made = mp.add_sources(tmp_path / "stage", "all")
    assert [p.name for p in made] == ["ccx_2.23.src.tar.bz2"]
    assert (tmp_path / "stage" / "licenses" / "sources" / "ccx_2.23.src.tar.bz2").is_file()
    assert got == [(cl.SOURCE_ARCHIVES["ccx_2.23.src.tar.bz2"]["url"], cl.SOURCE_ARCHIVES["ccx_2.23.src.tar.bz2"]["sha256"])]
    made = mp.add_sources(tmp_path / "stage2", "offline")
    assert [p.name for p in made] == ["gmsh-4.15.2-source.tgz"]


def test_pyproject_licence_files_and_package_data():
    d = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    files = d["project"]["license-files"]
    assert "LICENSE-EXCEPTIONS.md" in files and all((ROOT / f).is_file() for f in files)
    assert d["project"]["description"].isascii() or "–" in d["project"]["description"]
    assert not re.search(r"[ěščřžýáíéůú]", (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for pattern in d["tool"]["setuptools"]["package-data"]["fea_gui"]:
        assert list((ROOT / "fea_gui").glob(pattern)) or pattern.endswith("*.css"), pattern
    assert any(p.startswith("help/cs/") for p in d["tool"]["setuptools"]["package-data"]["fea_gui"])


def test_inno_script_has_no_stale_version():
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    block = iss[iss.index("#ifndef AppVersion"):]
    block = block[:block.index("#endif")]
    assert "#define AppVersion" not in block and "#error" in block

"""The file list of a package (installer/manifest.json) names every program file of THAT package: make_package used to write it before it added
the sources of the Windows CalculiX, so 129 files (about 190 MB) of the zip and setup.exe were in no manifest and no later update could remove
them. The whole `main()` runs on a tiny staged tree; every step that needs the network, tools or a venv is replaced."""
from __future__ import annotations

import importlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

mp = importlib.import_module("make_package")
upgrade = importlib.import_module("upgrade")

WIN_SOURCES = ("licenses/sources/spooles.2.2.tgz", "licenses/sources/crates/faer-0.24.4.crate", "licenses/sources/mingw-w64-headers-14/PKGBUILD")
# the crate is in both platforms (the same file): it is removed with the Windows sources and added again with the Linux ones
LINUX_SOURCES = ("licenses/sources/spooles.tgz", "licenses/sources/gcc-16.2.0.tar.gz", "licenses/sources/gcc-16.2.0-conda-forge-recipe/parent/meta.yaml",
                 "licenses/sources/crates/faer-0.24.4.crate")
ONLY_WIN, ONLY_LINUX = set(WIN_SOURCES) - set(LINUX_SOURCES), set(LINUX_SOURCES) - set(WIN_SOURCES)


def _put(root: Path, rel: str, data: bytes = b"x") -> Path:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)
    return root / rel


def _build(tmp_path, monkeypatch, *args):
    def add_sources(stage, kind, runtime=True):
        if kind == "all":
            return [_put(stage, "licenses/sources/ccx_2.23.src.tar.bz2")]
        names = {"windows": WIN_SOURCES, "linux": LINUX_SOURCES, "offline": ("licenses/sources/gmsh-4.15.2-source.tgz",)}[kind]
        return [_put(stage, n, n.encode()) for n in names]

    monkeypatch.setattr(mp, "file_list", lambda ref: ["fea_core/__init__.py", "README.md"])
    monkeypatch.setattr(mp, "stage_files", lambda files, stage, ref: [_put(stage, f, f.encode()) for f in files])
    monkeypatch.setattr(mp, "dirty_files", lambda: [])
    for name in ("require_linux_ccx", "require_windows_ccx"):
        monkeypatch.setattr(mp, name, lambda *a, **k: tmp_path / "ccx_2.23_linux_x86_64.tar.gz")
    monkeypatch.setattr(mp, "linux_ccx", lambda src, dst, with_pastix=False: _put(dst.parent, dst.name))
    for name in ("make_examples", "make_docs", "make_licenses"):
        monkeypatch.setattr(mp, name, lambda *a, **k: None)
    monkeypatch.setattr(mp, "normalize_eol", lambda stage: None)
    monkeypatch.setattr(mp, "anonymization_check", lambda stage: [])
    monkeypatch.setattr(mp, "add_sources", add_sources)
    monkeypatch.setattr(mp, "bundle_python", lambda stage: _put(stage, "tools/python/python.exe"))
    monkeypatch.setattr(mp, "bundle_windows_tools", lambda stage, **k: [_put(stage, "tools/uv/uv.exe").parent])
    monkeypatch.setattr(mp.upgrade, "make_build_id", lambda *a, **k: "20261126T000000Z-test")
    monkeypatch.setattr(sys, "argv", ["make_package.py", "--out", str(tmp_path / "dist"), "--no-setup", "--no-run", *args])
    assert mp.main() == 0
    return tmp_path / "dist", mp.version()


@pytest.fixture
def built(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch)


@pytest.fixture
def built_without_linux_ccx(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, "--no-linux-ccx")


def _files(names, read) -> tuple[dict, set]:
    rel = Path(upgrade.PACKAGE_MANIFEST).as_posix()
    manifest = json.loads(read(rel))
    return manifest["files"], {n for n in names if not n.startswith("tools/") and n != rel}


def test_each_package_lists_exactly_its_own_program_files(built):
    out, ver = built
    prefix = f"FEAStudio-{ver}/"
    with zipfile.ZipFile(out / f"FEAStudio-{ver}.zip") as z:
        listed, there = _files([n[len(prefix):] for n in z.namelist()], lambda n: z.read(prefix + n))
    assert set(listed) == there and set(WIN_SOURCES) <= set(listed) and not ONLY_LINUX & set(listed)
    with tarfile.open(out / f"FEAStudio-{ver}.tar.gz") as t:
        members = {m.name[len(prefix):]: m for m in t.getmembers() if m.isfile()}
        listed_l, there_l = _files(list(members), lambda n: t.extractfile(members[n]).read())
    assert set(listed_l) == there_l and set(LINUX_SOURCES) <= set(listed_l) and not ONLY_WIN & set(listed_l)
    assert "licenses/sources/ccx_2.23.src.tar.bz2" in listed_l
    # the history list of the release is made from the zip AND the tar.gz: an update on either system can remove all files of its release
    history = upgrade.manifest_of_release(out / f"FEAStudio-{ver}.zip", out / f"FEAStudio-{ver}.tar.gz", ver)["files"]
    assert history == {**listed_l, **listed}


def test_without_the_linux_ccx_the_linux_package_has_no_sources_of_it(built_without_linux_ccx):
    out, ver = built_without_linux_ccx
    with tarfile.open(out / f"FEAStudio-{ver}.tar.gz") as t:
        names = {m.name.split("/", 1)[1] for m in t.getmembers() if m.isfile()}
    assert not {n for n in names if n.startswith("licenses/sources/") and n != "licenses/sources/ccx_2.23.src.tar.bz2"}


def test_two_packages_with_different_content_of_one_file_have_no_release_list(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("FEAStudio-9/README.md", "one")
    with tarfile.open(tmp_path / "a.tar.gz", "w:gz") as t:
        (tmp_path / "README.md").write_text("two", encoding="utf-8")
        t.add(tmp_path / "README.md", "FEAStudio-9/README.md")
    with pytest.raises(ValueError, match="README.md"):
        upgrade.manifest_of_release(tmp_path / "a.zip", tmp_path / "a.tar.gz", "9")


def test_an_update_removes_a_platform_source_that_a_later_version_no_longer_ships(tmp_path, built):
    out, ver = built
    old = upgrade.manifest_of_release(out / f"FEAStudio-{ver}.zip", out / f"FEAStudio-{ver}.tar.gz", ver)
    gone = ("spooles.2.2.tgz", "gcc-16.2.0.tar.gz")
    new = {"version": "9.9.9", "files": {k: v for k, v in old["files"].items() if not k.endswith(gone)}}
    assert sorted(upgrade.obsolete_files(old, new)) == ["licenses/sources/gcc-16.2.0.tar.gz", "licenses/sources/spooles.2.2.tgz"]

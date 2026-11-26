"""The Linux packages carry the sources of everything in their CalculiX (GPL-2.0 §3(a) / GPL-3.0 §6, owner decision of 26. 11. 2026): the pins of
installer/linux/build_ccx.sh, the GCC archive of the pinned conda toolchain (installer/linux/ccx_gccsrc.lock) and the conda-forge recipe of the shipped
GCC libraries. No network: the downloads are replaced; the recipe is extracted from a made-up conda package with the real tar."""
from __future__ import annotations

import importlib
import io
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

mp = importlib.import_module("make_package")
cl = importlib.import_module("collect_licenses")
fs = importlib.import_module("fea_setup")

TOOLCHAIN = ROOT / "installer" / "linux" / "ccx_toolchain.lock"
# the shared libraries that the Linux archive of CalculiX may carry in lib/: both are GCC run-time libraries, their source is the GCC archive
LIBS_WITH_SOURCE = {"lib/libgomp.so.1", "lib/libquadmath.so.0"}
GCC_PACKAGES = ("libgcc", "libgomp", "libgfortran5", "libgcc-devel_linux-64", "gcc_impl_linux-64", "gfortran_impl_linux-64")


def _rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


# ------------------------------------------------------------------ the pins
def test_the_gcc_source_pin_belongs_to_the_pinned_toolchain():
    gcc = cl.linux_gcc_source(ROOT)
    assert gcc["version"] and gcc["name"] == f"gcc-{gcc['version']}.tar.gz" and cl.is_linux_gcc_source(gcc["name"])
    assert all(u.endswith(f"/gcc-{gcc['version']}/{gcc['name']}") for u in gcc["urls"]) and gcc["urls"][0].startswith("https://ftp.gnu.org/")
    assert re.fullmatch(r"[0-9a-f]{40}", gcc["commit"])
    found = {}
    for _sha, _md5, url in _rows(TOOLCHAIN):
        m = re.fullmatch(r"(.+)-([^-]+)-([^-]+)\.(?:conda|tar\.bz2)", url.rsplit("/", 1)[1])
        if m and m[1] in GCC_PACKAGES:
            found[m[1]] = m[2]
    assert set(found) == set(GCC_PACKAGES) and set(found.values()) == {gcc["version"]}, \
        "the toolchain has another GCC than installer/linux/ccx_gccsrc.lock names: take the source line of the recipe of the new packages"
    name, digest, url = cl.linux_gcc_recipe_package(ROOT)
    assert name.startswith(f"libgomp-{gcc['version']}-") and [digest, url] in [[r[0], r[2]] for r in _rows(TOOLCHAIN)]


def test_a_malformed_gcc_lock_stops_and_a_tree_without_it_has_none(tmp_path):
    assert cl.linux_gcc_source(tmp_path) is None
    lock = tmp_path / "installer" / "linux" / "ccx_gccsrc.lock"
    lock.parent.mkdir(parents=True)
    good = (ROOT / "installer" / "linux" / "ccx_gccsrc.lock").read_text(encoding="utf-8")
    row = next(line for line in good.splitlines() if line.startswith("gccsrc "))
    for broken in (good.replace("# gcc_version:", "# version:"), good.replace(row, ""), good + row + "\n",
                   good.replace(row, "gccsrc abc https://ftp.gnu.org/x.tar.gz"), good.replace(row, row.replace("https://", "http://"))):
        lock.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="ccx_gccsrc.lock"):
            cl.linux_gcc_source(tmp_path)


def test_the_linux_sources_are_every_pin_of_the_build_and_gcc():
    items = cl.linux_sources(ROOT)
    names = [n for n, _, _ in items]
    assert len(names) == len(set(names)) and all(re.fullmatch(r"[0-9a-f]{64}", d) and u for _, d, u in items)
    assert {"spooles.tgz", "spooles.corr.tar.bz2", "lapack.tar.gz", "arpack.tar.gz", cl.linux_gcc_source(ROOT)["name"]} <= set(names)
    assert sum(n.startswith("rust-src-") for n in names) == 1
    cargo = tomllib.loads((ROOT / "installer" / "faer_ccx" / "Cargo.lock").read_text(encoding="utf-8"))
    crates = {f"crates/{p['name']}-{p['version']}.crate" for p in cargo["package"] if str(p.get("source", "")).startswith("registry+")}
    assert len(crates) == 118 and {n for n in names if n.startswith("crates/")} == crates
    shared = cl.SOURCE_ARCHIVES["ccx_2.23.src.tar.bz2"]["sha256"]                    # CalculiX itself: in licenses/sources/ of every package already
    assert shared not in {d for _, d, _ in items} and shared in {d for _, d, _ in cl.build_ccx_sources(ROOT)}
    assert not any("pastix" in n or n.startswith("micromamba") for n in names)
    # every pinned download of build_ccx.sh is covered, nothing is dropped
    script = {(d, n) for n, d, _ in cl.build_ccx_sources(ROOT) if d != shared}
    assert script <= {(d, n) for n, d, _ in items}


def test_every_library_of_the_prebuilt_linux_ccx_has_its_source(tmp_path):
    """A new shared library in lib/ of the archive would be shipped without its source: extend the sources (and this list) first."""
    archive = ROOT / "installer" / "linux" / "prebuilt" / "ccx_2.23_linux_x86_64.tar.gz"
    if not archive.is_file():
        pytest.skip("no prebuilt Linux CalculiX in this checkout")
    mp.linux_ccx(archive, tmp_path / "out.tar.gz")
    with tarfile.open(tmp_path / "out.tar.gz") as t:
        files = {m.name.lstrip("./") for m in t.getmembers() if m.isfile()}
    assert files - {"ccx_spooles"} <= LIBS_WITH_SOURCE, files


# ------------------------------------------------------------------ add_sources
def _recipe_tar(meta: str, patches: list[str]) -> bytes:
    """A made-up info-*.tar.zst: an uncompressed tar (tar finds the compression by the content, so the test needs no zstd)."""
    buf = io.BytesIO()
    files = {"info/about.json": "{}", "info/recipe/parent/meta.yaml": meta, "info/recipe/parent/build.sh": "#!/bin/bash\n",
             "info/recipe/conda_build_config.yaml": "gcc_version: x\n"}
    files.update({f"info/recipe/parent/patches/{p}": "--- a\n+++ b\n" for p in patches})
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


PATCHES = [f"000{k}-fix.patch" for k in range(1, 5)]


def _meta(digest: str, version: str, patches: list[str]) -> str:
    return (f'source:\n  - url: x\n    sha256: {digest}   # [gcc_version == "{version}"]\n'
            f'    sha256: {"0" * 64}   # [gcc_version == "1.0.0"]\n    patches:\n' + "".join(f"      - patches/{p}\n" for p in patches))


def _fake_downloads(tmp_path, monkeypatch, meta: str | None = None, patches: list[str] = PATCHES, fail: tuple = ()):
    gcc = cl.linux_gcc_source(ROOT)
    meta = _meta(gcc["sha256"], gcc["version"], patches) if meta is None else meta
    got = []

    def fake_download(url, dest, what="", sha256=None):
        got.append((url, sha256))
        if url in fail:
            raise RuntimeError("no network")
        p = tmp_path / "dl" / Path(dest).name
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.name.endswith(".conda"):
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("metadata.json", "{}")
                z.writestr("info-libgomp-1.tar.zst", _recipe_tar(meta, patches))
                z.writestr("pkg-libgomp-1.tar.zst", b"binaries")
        else:
            p.write_bytes(b"source")
        return p
    monkeypatch.setattr(fs, "download", fake_download)
    monkeypatch.setattr(mp, "WIN_ARCHIVE", tmp_path / "no_archive")
    return got


def test_add_sources_linux_takes_every_pin_and_the_recipe(tmp_path, monkeypatch):
    got = _fake_downloads(tmp_path, monkeypatch)
    stage = tmp_path / "stage"
    made = mp.add_sources(stage, "linux")
    src = stage / "licenses" / "sources"
    rel = {p.relative_to(src).as_posix() for p in made}
    assert rel == {p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()}
    assert {n for n, _, _ in cl.linux_sources(ROOT)} <= rel and "ccx_2.23.src.tar.bz2" not in rel
    folder = cl.linux_gcc_recipe_folder(cl.linux_gcc_source(ROOT)["version"])
    recipe = {r for r in rel if r.startswith(folder + "/")}
    assert {f"{folder}/parent/meta.yaml", f"{folder}/parent/build.sh", *(f"{folder}/parent/patches/{p}" for p in PATCHES)} <= recipe
    assert not any("about.json" in r or r.endswith((".conda", ".tar.zst")) for r in rel)            # the recipe only, not the binary package
    assert {d for _, d in got} == {d for _, d, _ in cl.linux_sources(ROOT)} | {cl.linux_gcc_recipe_package(ROOT)[1]}   # every download with its digest

    # a package for tests without the run-time sources says so and has neither GCC, its recipe nor the Rust part
    made2 = {p.name for p in mp.add_sources(tmp_path / "stage2", "linux", runtime=False)}
    assert made2 == {"spooles.tgz", "spooles.corr.tar.bz2", "lapack.tar.gz", "arpack.tar.gz"}

    mp.remove_sources(stage, made)
    assert src.is_dir() and not any(src.iterdir())


@pytest.mark.parametrize("meta, patches, message", [
    ("sha256: " + "1" * 64 + '   # [gcc_version == "VERSION"]\n', PATCHES, "does not build gcc-"),
    (None, PATCHES[:3], "3 patches"),
])
def test_a_recipe_of_other_binaries_stops_the_package(tmp_path, monkeypatch, meta, patches, message):
    version = cl.linux_gcc_source(ROOT)["version"]
    _fake_downloads(tmp_path, monkeypatch, meta=meta and meta.replace("VERSION", version), patches=patches)
    with pytest.raises(SystemExit, match=message):
        mp.add_sources(tmp_path / "stage", "linux")


def test_the_mirror_of_a_pin_is_used_and_no_source_is_an_error(tmp_path, monkeypatch):
    gcc = cl.linux_gcc_source(ROOT)
    got = _fake_downloads(tmp_path, monkeypatch, fail=(gcc["urls"][0],))
    assert mp.source_archive(gcc["name"], gcc["urls"], gcc["sha256"]).read_bytes() == b"source"
    assert [u for u, _ in got] == gcc["urls"][:2]
    _fake_downloads(tmp_path, monkeypatch, fail=tuple(gcc["urls"]))
    with pytest.raises(SystemExit, match="must not ship without"):
        mp.source_archive(gcc["name"], gcc["urls"], gcc["sha256"])


def test_the_recipe_of_the_cached_pinned_package_builds_the_pinned_gcc(tmp_path, monkeypatch):
    """With the real pinned conda package in the download cache of the packaging (after a package build): its recipe passes the checks."""
    name, digest, _url = cl.linux_gcc_recipe_package(ROOT)
    cached = ROOT / "tools" / "_download" / "sources" / name
    if not cached.is_file():
        pytest.skip(f"{name} is not in tools/_download/sources (no package built here)")
    monkeypatch.setattr(fs, "download", lambda url, dest, what="", sha256=None: cached if fs.sha256_file(cached) == sha256 else pytest.fail("digest"))
    made = mp.extract_gcc_recipe(tmp_path)
    assert any(p.name == "meta.yaml" for p in made) and sum(p.suffix == ".patch" for p in made) == mp.GCC_RECIPE_LINUX_PATCHES
    assert fs.sha256_file(cached) == digest


# ------------------------------------------------------------------ the generated documents
def test_sources_md_lists_the_linux_sources_and_needs_no_written_offer_for_them(tmp_path):
    cl.write_sources(tmp_path, ROOT, "2026-11-26")
    text = (tmp_path / "SOURCES.md").read_text(encoding="utf-8")
    linux = text[text.index("### The Linux program"):text.index("## LGPL libraries carried as Python wheels")]
    assert "The Linux packages (`.tar.gz`, `.run`) carry all of them in `sources/`" in linux and "written offer" not in linux.lower()
    for name, digest, _urls in cl.linux_sources(ROOT):
        assert digest in text and f"`sources/{name}`" in text, name
    gcc = cl.linux_gcc_source(ROOT)
    pkg, pkg_digest, pkg_url = cl.linux_gcc_recipe_package(ROOT)
    assert f"`sources/{gcc['name']}`" in linux and gcc["sha256"] in linux and gcc["commit"] in linux
    assert f"`sources/{cl.linux_gcc_recipe_folder(gcc['version'])}/`" in linux and pkg in linux and pkg_digest in linux and pkg_url in linux
    ccx = cl.SOURCE_ARCHIVES["ccx_2.23.src.tar.bz2"]
    assert "`sources/ccx_2.23.src.tar.bz2`" in linux and ccx["sha256"] in linux
    assert "written offer" not in text.lower()
    third = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    assert "ccx_gccsrc.lock" in third and "there is no written offer" in " ".join(third.split())

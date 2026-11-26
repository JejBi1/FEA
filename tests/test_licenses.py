"""Collection of the third-party licences (installer/collect_licenses.py, connected in make_package) – fast, without network and without Gmsh/CalculiX."""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

import collect_licenses as cl  # noqa: E402

VENV = ROOT / ".venv"
HAS_VENV = (VENV / "Lib" / "site-packages").is_dir() or any(VENV.glob("lib/python3*/site-packages"))
# the checks of the real environment expect every locked package (the development venv / the profile full), not a CI minimal one
FULL_VENV = HAS_VENV and any(any(sp.glob("ngsolve-*.dist-info")) for sp in [VENV / "Lib" / "site-packages", *VENV.glob("lib/python3*/site-packages")])


def _fake_project(root: Path, win: str, linux: str) -> None:
    (root / "installer" / "locks").mkdir(parents=True)
    (root / "installer" / "locks" / "windows-x86_64.txt").write_text(win, encoding="utf-8")
    (root / "installer" / "locks" / "linux-x86_64.txt").write_text(linux, encoding="utf-8")


def _dist(site: Path, name: str, version: str, meta: str, files: dict[str, str] | None = None) -> None:
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{meta}", encoding="utf-8")
    for rel, text in (files or {}).items():
        (info / rel).parent.mkdir(parents=True, exist_ok=True)
        (info / rel).write_text(text, encoding="utf-8")


def test_synthetic_venv_collects_and_warns(tmp_path):
    root = tmp_path / "proj"
    _fake_project(root, "fakepkg==1.0\nnolic==2.0\n", "fakepkg==1.0\nlinuxonly==3.0\n")
    venv = tmp_path / "venv"
    site = venv / "Lib" / "site-packages"
    _dist(site, "fakepkg", "1.0", "License-Expression: MIT\nLicense-File: licenses/LICENSE.txt\n",
          {"licenses/LICENSE.txt": "MIT text\n"})
    _dist(site, "nolic", "2.0", "License: BSD\n")
    out = tmp_path / "licenses"
    warnings = cl.collect(out, venv, root=root)
    assert any("nolic" in w and "has no licence file" in w for w in warnings), warnings
    assert not any("fakepkg" in w for w in warnings)
    assert (out / "python" / "fakepkg-1.0" / "licenses" / "LICENSE.txt").read_text(encoding="utf-8") == "MIT text\n"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "| Package | Version | Platform | Licence | Texts / note |" in readme
    assert "| fakepkg | 1.0 | Windows + Linux | MIT |" in readme
    assert "| linuxonly | 3.0 | Linux | unknown – not installed |" in readme
    assert "licence file MISSING" in readme
    assert "CalculiX" in (out / "EXTERNAL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("installed", [True, False])
def test_qt_for_python_ships_the_lgpl_it_is_used_under(tmp_path, installed):
    """The PySide6 / shiboken6 wheels carry only LicenseRef-Qt-Commercial.txt: the LGPL-3.0 (and the GPL-3.0 it builds on) are added from
    installer/license_texts, and the table says which licence FEA Studio uses them under – also for the Addons, which no package ships."""
    root = tmp_path / "proj"
    lock = "pyside6-essentials==6.11.2\npyside6-addons==6.11.2\nshiboken6==6.11.2\n"
    _fake_project(root, lock, lock)
    shutil.copytree(TEXTS, root / "installer" / "license_texts")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    if installed:
        for name in ("PySide6_Essentials", "shiboken6"):
            _dist(site, name, "6.11.2", "License-Expression: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only\n",
                  {"licenses/LicenseRef-Qt-Commercial.txt": "commercial\n"})
    out = tmp_path / "licenses"
    cl.collect(out, tmp_path / "venv", root=root)
    lgpl = (TEXTS / "_gnu" / "lgpl-3.0.txt").read_bytes()
    assert b"GNU LESSER GENERAL PUBLIC LICENSE" in lgpl and b"Version 3, 29 June 2007" in lgpl
    names = ["pyside6-addons-6.11.2"] + (["pyside6-essentials-6.11.2", "shiboken6-6.11.2"] if installed else [])
    for folder in names:
        found = list((out / "python").glob(folder.replace("pyside6-essentials", "[Pp]y[Ss]ide6[-_][Ee]ssentials")))
        assert found, (folder, [p.name for p in (out / "python").iterdir()])
        assert (found[0] / "lgpl-3.0.txt").read_bytes() == lgpl
        assert (found[0] / "gpl-3.0.txt").read_bytes() == (TEXTS / "_gnu" / "gpl-3.0.txt").read_bytes()
    if installed:
        readme = (out / "README.md").read_text(encoding="utf-8")
        for line in readme.splitlines():
            if line.lower().startswith(("| pyside6-essentials |", "| shiboken6 |")):
                assert "LGPL-3.0 (used under it" in line, line
    third = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    assert "shiboken6" in third and "used under the LGPL-3.0" in third


def test_the_licence_documents_name_what_is_really_in_the_package(tmp_path):
    """File and folder names that the generated documents refer to exist in the generated tree (review of 1.5.1: LICENSE-THIRD-PARTY of rust-std,
    licenses/external/lapack/, the crates/ paths of faer's CRATES.md, "two patches")."""
    out = tmp_path / "licenses"
    out.mkdir()
    cl.copy_external(out, ROOT)
    ext = {e["slug"] if e.get("slug") else cl.ext_slug(e): e for e in cl.EXTERNAL}
    assert "COPYRIGHT-library.html" in ext["rust-std"]["licence"] and (out / "external" / "rust-std" / "COPYRIGHT-library.html").is_file()
    crates = (out / "external" / "faer" / "CRATES.md").read_text(encoding="utf-8")
    for ref in re.findall(r"`(crates/[A-Za-z0-9_.\-]+/)`", crates):
        assert (out / "external" / "faer" / ref).is_dir(), ref
    assert (out / "external" / "faer" / "crates" / "faer-0.24.4" / "COPYING.LAPACK.BSD").is_file()
    third = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    for folder in re.findall(r"`licenses/external/([A-Za-z0-9_./-]+?)/`", third):
        assert (out / "external" / folder).is_dir(), folder
    patches = sorted(p.name for p in (ROOT / "installer" / "windows" / "patches").glob("*.patch"))
    cl.write_sources(out, ROOT, "today")
    text = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert len(patches) == 4 and all(p in text and p in ext["calculix"]["source"] for p in patches), patches
    assert "two patches" not in text + ext["calculix"]["source"]
    assert "installer/linux/ccx_rust.lock" in text and "tools/calculix/ccx_2.23_linux/ccx_spooles" in text
    assert "ZDROJ" not in "".join(p.name for p in (out / "external" / "calculix").iterdir())


def test_site_packages_layouts(tmp_path):
    (tmp_path / "win" / "Lib" / "site-packages").mkdir(parents=True)
    assert cl.site_packages(tmp_path / "win")[1] == "windows-x86_64"
    (tmp_path / "lin" / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
    assert cl.site_packages(tmp_path / "lin")[1] == "linux-x86_64"


def test_license_files_from_dist_info_root(tmp_path):
    root = tmp_path / "proj"
    _fake_project(root, "rootlic==1.0\n", "rootlic==1.0\n")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    _dist(site, "rootlic", "1.0", "License: MIT\n", {"COPYING": "copying\n", "README.txt": "nic\n"})
    out = tmp_path / "licenses"
    assert cl.collect(out, tmp_path / "venv", root=root) == []
    assert (out / "python" / "rootlic-1.0" / "COPYING").is_file()
    assert not (out / "python" / "rootlic-1.0" / "README.txt").exists()


def test_missing_lock_is_error(tmp_path):
    (tmp_path / "installer" / "locks").mkdir(parents=True)
    (tmp_path / "installer" / "locks" / "windows-x86_64.txt").write_text("a==1\n", encoding="utf-8")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        cl.collect(tmp_path / "out", tmp_path / "venv", root=tmp_path)


def test_main_exit_code_stays_zero_with_warnings(tmp_path):
    site = tmp_path / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    assert cl.main(["--out", str(tmp_path / "out"), "--venv", str(tmp_path / "venv")]) == 0
    assert cl.main(["--out", str(tmp_path / "out2"), "--venv", str(tmp_path / "chybi")]) == 1


@pytest.mark.skipif(not FULL_VENV, reason="needs the full development .venv")
def test_real_venv_covers_all_locks(tmp_path):
    out = tmp_path / "licenses"
    warnings = cl.collect(out, VENV)
    assert all(isinstance(w, str) for w in warnings)
    pkgs = cl.read_locks(ROOT)
    readme = (out / "README.md").read_text(encoding="utf-8")
    rows = [l for l in readme.splitlines() if l.startswith("| ") and not l.startswith("| Package ")]
    for entry in pkgs.values():
        assert f"| {entry['name']} |" in readme, f"{entry['name']} chybí v README"
    assert len(rows) >= len(pkgs)
    # installed packages with a licence file have the folder python/<name>-<version>
    folders = {p.name for p in (out / "python").iterdir()}
    assert len(folders) >= 50
    ext = (out / "EXTERNAL.md").read_text(encoding="utf-8")
    assert "https://www.dhondt.de/ccx_2.23.src.tar.bz2" in ext
    assert "https://gmsh.info/src/" in ext


@pytest.mark.skipif(not HAS_VENV, reason="chybí vývojové .venv")
def test_make_package_stage_gets_licenses(tmp_path):
    import make_package  # noqa: E402  (installer/make_package.py)

    make_package.make_licenses(tmp_path)
    assert (tmp_path / "licenses" / "README.md").is_file()
    assert (tmp_path / "licenses" / "EXTERNAL.md").is_file()
    assert any((tmp_path / "licenses" / "python").iterdir())


# --- licence texts downloaded from the official sources (installer/license_texts/, SOURCES.md) ---

TEXTS = ROOT / "installer" / "license_texts"
# Exceptions to the rule "every component of EXTERNAL has a text in external/<slug>/" (only with a justification). Today none: even MS-MPI has
# the MIT text from the sources (a redistributable runtime with its own terms, the user installs it).
EXTERNAL_WITHOUT_TEXT: dict[str, str] = {}


def test_every_external_component_has_a_text(tmp_path):
    copied = cl.copy_external(tmp_path, ROOT)
    for e in cl.EXTERNAL:
        if e["name"] in EXTERNAL_WITHOUT_TEXT:
            continue
        folder = tmp_path / "external" / cl.ext_slug(e)
        assert copied[e["name"]], f"{e['name']}: v external/ není žádný text licence"
        assert all((folder / n).is_file() and (folder / n).stat().st_size > 0 for n in copied[e["name"]])
    slugs = [cl.ext_slug(e) for e in cl.EXTERNAL]
    assert len(slugs) == len(set(slugs)), "názvy složek external/ se opakují"


def test_external_patterns_match_files():
    # the pattern in EXTERNAL["texts"] and in PYTHON_EXTRA must find something (a typo in the pattern would silently leave the text out)
    for e in cl.EXTERNAL:
        for pattern in e.get("texts", []):
            assert cl.license_text_files(ROOT, [pattern]), f"{e['name']}: vzor {pattern} nic nenašel"
    for key, patterns in cl.PYTHON_EXTRA.items():
        for pattern in patterns:
            assert cl.license_text_files(ROOT, [pattern]), f"{key}: vzor {pattern} nic nenašel"


def test_python_extra_texts_for_wheels_without_license(tmp_path):
    root = tmp_path / "proj"
    _fake_project(root, "gmsh==4.15.2\ncadquery-ocp==8.0.1.1.0\nngsolve-openblas==0.3.33\nnolic==2.0\n",
                  "gmsh==4.15.2\ncadquery-ocp==8.0.1.1.0\nngsolve-openblas==0.3.33\n")
    shutil.copytree(TEXTS, root / "installer" / "license_texts")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    _dist(site, "gmsh", "4.15.2", "License: GPLv2+\n")
    _dist(site, "cadquery_ocp", "8.0.1.1.0", "License: Apache-2.0\n")
    _dist(site, "ngsolve_openblas", "0.3.33", "License: BSD-3-Clause\n")
    _dist(site, "nolic", "2.0", "License: BSD\n")
    out = tmp_path / "licenses"
    warnings = cl.collect(out, tmp_path / "venv", root=root)
    assert [w for w in warnings if "has no licence file" in w] and all("nolic" in w for w in warnings if "has no licence file" in w)
    assert {p.name for p in (out / "python" / "gmsh-4.15.2").iterdir()} >= {"LICENSE.txt", "gpl-2.0.txt"}
    assert {p.name for p in (out / "python" / "cadquery-ocp-8.0.1.1.0").iterdir()} >= {
        "LICENSE", "LICENSE_LGPL_21.txt", "OCCT_LGPL_EXCEPTION.txt"}
    assert (out / "python" / "ngsolve-openblas-0.3.33" / "LICENSE").is_file()


def test_a_locked_package_missing_in_the_build_venv_keeps_its_texts(tmp_path):
    """The venv of the build machine may not have a package that the installation brings (1.4.2: rslab comes from tools/solver_lab in the development venv):
    its texts of installer/license_texts are used (no warning, a README row), a package of the verify module is only noted, any other missing one warns."""
    root = tmp_path / "proj"
    _fake_project(root, "rslab==1.14.0\npytest-timeout==2.4.0\nlostpkg==1.0\n", "rslab==1.14.0\n")
    shutil.copytree(TEXTS, root / "installer" / "license_texts")
    (tmp_path / "venv" / "Lib" / "site-packages").mkdir(parents=True)
    out = tmp_path / "licenses"
    warnings = cl.collect(out, tmp_path / "venv", root=root)
    assert {p.name for p in (out / "python" / "rslab-1.14.0").iterdir()} == {"LICENSE", "NOTICE"}
    assert not any("rslab" in w or "pytest-timeout" in w for w in warnings), warnings
    assert any("lostpkg" in w for w in warnings), warnings
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "| rslab | 1.14.0 | Windows + Linux | MIT |" in readme and "texts from installer/license_texts" in readme
    assert "| pytest-timeout | 2.4.0 | Windows | unknown – not installed |" in readme


@pytest.mark.skipif(not FULL_VENV, reason="needs the full development .venv")
def test_real_venv_has_texts_for_known_gaps(tmp_path):
    out = tmp_path / "licenses"
    warnings = cl.collect(out, VENV)
    folders = {p.name: p for p in (out / "python").iterdir()}
    for prefix in ("gmsh-", "cadquery-ocp-", "cadquery-ocp-proxy-", "ngsolve-openblas-"):
        hit = [n for n in folders if n.startswith(prefix) and n[len(prefix)].isdigit()]
        assert hit, f"{prefix}: chybí složka python/{prefix}<verze>"
        assert any(f.is_file() for f in folders[hit[0]].iterdir())
    for w in warnings:
        assert not any(x in w for x in ("gmsh", "cadquery-ocp", "ngsolve-openblas")), w
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "## Data" in readme and "pragtic_vmat.json" in readme and "until the author agrees" in readme
    assert "## Data" in (out / "EXTERNAL.md").read_text(encoding="utf-8")


def test_sources_md_lists_every_text_with_correct_hash():
    sources = (TEXTS / "SOURCES.md").read_text(encoding="utf-8")
    files = [p for p in TEXTS.rglob("*") if p.is_file() and p.name != "SOURCES.md"]
    assert len(files) >= 25
    table_rows = [l for l in sources.splitlines() if l.startswith("| ") and l.split("|")[2].strip().startswith("`")]
    assert len(table_rows) == len(files), "SOURCES.md popisuje jiný počet souborů, než je ve složce"
    for f in files:
        rel = f.relative_to(TEXTS).as_posix()
        rows = [l for l in sources.splitlines() if l.startswith("|") and l.split("|")[2].strip() == f"`{rel}`"]
        assert len(rows) == 1, f"{rel}: v SOURCES.md má být právě jeden řádek"
        assert f"`{hashlib.sha256(f.read_bytes()).hexdigest()}`" in rows[0], f"{rel}: SHA-256 v SOURCES.md nesedí"
        assert re.search(r"\| [0-9]{1,2}\. [0-9]{1,2}\. 20[0-9]{2} \|", rows[0]), f"{rel}: chybí datum stažení"      # d. m. yyyy


def test_license_texts_are_real_texts():
    # downloaded correctly: no HTML page with an error, no empty file
    for f in TEXTS.rglob("*"):
        if f.is_file() and f.name != "SOURCES.md" and f.suffix != ".html":
            head = f.read_bytes()[:2000].lower()
            assert b"<!doctype" not in head and b"<html" not in head, f"{f} vypadá jako HTML stránka"
        if f.is_file():
            assert f.stat().st_size > 200, f"{f} je podezřele malý"



# ------------------------------------------------------------------ the own Windows CalculiX (F13 step 5d): what it contains and what it must not
def test_own_ccx_components_have_texts_and_attribution(tmp_path):
    """Everything the own `ccx_spooles` (Windows and Linux) is made of has a licence text in the package; SPOOLES carries the acknowledgement that its authors
    ask for (the statement is stored verbatim next to the SPOOLES html page)."""
    slugs = {cl.ext_slug(e): e for e in cl.EXTERNAL}
    for slug in ("calculix", "spooles", "arpack-ng", "lapack-blas", "libgomp-libquadmath", "libbacktrace", "mingw-w64"):
        assert slug in slugs, slug
    assert b"public domain" in (TEXTS / "spooles" / "NOTICE-ReferenceManual.txt").read_bytes()
    assert b"acknowledgement" in (TEXTS / "spooles" / "NOTICE-ReferenceManual.txt").read_bytes()
    assert b"public domain" in (TEXTS / "spooles" / "spooles.2.2.html").read_bytes()
    iohb = (TEXTS / "spooles" / "NOTICE-iohb.c.txt").read_bytes()               # the NIST reader (iohb.c) is linked in: its notice goes with the documentation
    assert b"National Institute of Standards and Technology" in iohb and b"supporting documentation" in iohb
    assert b"Redistribution and use in source and binary forms" in (TEXTS / "arpack-ng" / "COPYING").read_bytes()
    lapack = (TEXTS / "lapack" / "LICENSE-3.12.0").read_bytes()
    assert b"Redistribution and use in source and binary forms" in lapack and b"University of Tennessee" in lapack
    bt = (TEXTS / "libbacktrace" / "LICENSE").read_bytes()
    assert b"Free Software Foundation" in bt and b"Redistribution and use in source and binary forms" in bt
    assert b"GCC RUNTIME LIBRARY EXCEPTION" in (TEXTS / "gcc-runtime" / "COPYING.RUNTIME").read_bytes().upper()
    assert "Used by permission" not in " ".join(e["use"] for e in cl.EXTERNAL)           # nothing that asks for it is shipped any more
    texts = cl.copy_external(tmp_path, ROOT)
    assert "spooles.2.2.html" in texts["SPOOLES"] and "NOTICE-ReferenceManual.txt" in texts["SPOOLES"] and "NOTICE-iohb.c.txt" in texts["SPOOLES"]


def test_the_removed_components_are_gone_from_the_package_texts():
    """TAUCS (with AMD, COLAMD, GENMMD), PaStiX, Scotch, hwloc and libltdl were inside the third-party ccx_static.exe and are not inside the own build:
    no component row, no text, no row of installer/license_texts/SOURCES.md, no notice in the user documentation – the build script checks the binary
    (`strings` for their symbols) and tests/test_win_ccx.py checks the real binary when it was built."""
    slugs = {cl.ext_slug(e) for e in cl.EXTERNAL}
    for slug in ("pastix", "hwloc", "scotch", "taucs", "amd", "colamd", "genmmd", "libltdl", "openblas-ccx"):
        assert slug not in slugs, slug
    for folder in ("pastix", "hwloc", "scotch", "taucs", "amd", "colamd", "genmmd"):
        assert not (TEXTS / folder).exists(), folder
    sources = (TEXTS / "SOURCES.md").read_text(encoding="utf-8")
    for word in ("TAUCS", "PaStiX", "Scotch", "hwloc", "GENMMD", "COLAMD", "`amd/", "libltdl"):
        rows = [l for l in sources.splitlines() if l.startswith("|") and word in l]
        assert not rows, (word, rows[:1])
    for rel in ("THIRD_PARTY_LICENSES.md", "fea_gui/help/index.html", "fea_gui/help/cs/index.html"):
        text = " ".join((ROOT / rel).read_text(encoding="utf-8").split())
        assert "Used by permission" not in text and "Copyright (c) 2001, 2002, 2003 by Sivan Toledo" not in text, rel
        assert "SPOOLES" in text and "acknowledgement in derivative codes" in text, rel                       # the acknowledgement that SPOOLES asks for
        assert "LAPACK" in text and "ARPACK" in text and "libbacktrace" in text and "winpthreads" in text, rel
        assert "ccx_spooles" in text, rel
    about = (ROOT / "fea_gui" / "about.py").read_text(encoding="utf-8")
    assert '"THIRD_PARTY_LICENSES.md"' in about                                                                 # the About dialog shows the notices

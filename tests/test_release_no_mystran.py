"""Owner decisions of 26. 11. 2026 for the release 1.5.1:
1. no package carries MYSTRAN (the packaging refuses a stage with it; the program works without it and the cross-checks say "not run");
2. there is no written offer: a build that leaves out sources of binaries it carries is a development build – files named …-dev, licenses/SOURCES.md
   says NOT FOR DISTRIBUTION and so does the build log."""
from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

mp = importlib.import_module("make_package")
cl = importlib.import_module("collect_licenses")

from fea_core import modules, mystran, verification_summary  # noqa: E402
from tests.test_package_manifests import _build, _put  # noqa: E402


# ------------------------------------------------------------------ 1. no MYSTRAN in a package
@pytest.mark.parametrize("name", ["tools/mystran/mystran.exe", "tools/mystran/mystran", "tools/mystran/README.txt", "Tools/MYSTRAN/MYSTRAN.EXE",
                                  "bin/mystran.exe", "mystran", "extra/MYSTRAN-19.0.zip", "x/mystran_19.dll", "mystran.tar.gz"])
def test_a_mystran_program_or_its_archive_is_recognised(name):
    assert mp.mystran_files([name]) == [name]


def test_the_programs_own_files_are_not_taken_for_mystran():
    own = [p.relative_to(ROOT).as_posix() for top in ("fea_core", "verification", "tests", "installer") for p in (ROOT / top).rglob("*mystran*")
           if p.is_file() and "__pycache__" not in p.parts and not p.relative_to(ROOT).as_posix().startswith("verification/out")]
    assert "fea_core/mystran.py" in own and "fea_core/beams/mystran_frame.py" in own and "verification/mystran_check.py" in own
    assert mp.mystran_files(own) == []


@pytest.mark.parametrize("writer", ["zip", "tar", "setup"])
def test_every_package_writer_refuses_a_staged_mystran(tmp_path, monkeypatch, writer):
    stage = tmp_path / "stage"
    _put(stage, "README.md")
    _put(stage, "tools/mystran/mystran.exe")
    monkeypatch.setattr(mp.subprocess, "run", lambda *a, **k: pytest.fail("the compiler must not be started"))
    calls = {"zip": lambda: mp.write_zip(stage, "p", tmp_path / "a.zip"), "tar": lambda: mp.write_tar(stage, "p", tmp_path / "a.tar.gz"),
             "setup": lambda: mp.write_setup_exe(stage, tmp_path, "1.5.1", "ISCC.exe")}
    with pytest.raises(SystemExit, match="MYSTRAN must not be in a package.*tools/mystran/mystran.exe"):
        calls[writer]()
    assert not (tmp_path / "a.zip").exists() and not (tmp_path / "a.tar.gz").exists()


def test_the_packaging_no_longer_copies_mystran(tmp_path, monkeypatch):
    """A real tools/mystran/mystran.exe next to the project (the developer's copy) never reaches a package."""
    out, ver = _build(tmp_path, monkeypatch)
    with zipfile.ZipFile(out / f"FEAStudio-{ver}.zip") as z:
        assert mp.mystran_files(z.namelist()) == [] and not [n for n in z.namelist() if "/tools/mystran" in n]
    with tarfile.open(out / f"FEAStudio-{ver}.tar.gz") as t:
        assert mp.mystran_files(t.getnames()) == []
    assert '"tools" / "mystran"' not in (ROOT / "installer" / "make_package.py").read_text(encoding="utf-8")


def _fake_iscc(monkeypatch, out: Path, packed: list[str], seen: list):
    def run(args, **kw):
        seen.append(args)
        suffix = "-dev" if "/DDevSuffix=-dev" in args else ""
        (out / f"FEAStudio-1.5.1{suffix}-setup.exe").write_bytes(b"MZ")
        return SimpleNamespace(returncode=0, stdout="".join(f"   Compressing: {p}\n" for p in packed), stderr="")
    monkeypatch.setattr(mp.subprocess, "run", run)


def test_the_setup_compilers_file_list_is_checked_and_logged(tmp_path, monkeypatch):
    _put(tmp_path / "stage", "README.md")
    seen: list = []
    _fake_iscc(monkeypatch, tmp_path, [r"C:\w\stage\README.md", r"C:\w\stage\python3.dll   (3.11.17150.1013)"], seen)
    log = tmp_path / "iscc.log"
    exe = mp.write_setup_exe(tmp_path / "stage", tmp_path, "1.5.1", "ISCC.exe", log=log)
    assert exe is not None and exe.name == "FEAStudio-1.5.1-setup.exe" and "Compressing: C:" in log.read_text(encoding="utf-8")
    assert "/Q" not in seen[0] and not [a for a in seen[0] if a.startswith("/DDevSuffix")]
    assert mp.iscc_files(log.read_text(encoding="utf-8")) == ["C:/w/stage/README.md", "C:/w/stage/python3.dll"]
    _fake_iscc(monkeypatch, tmp_path, [r"C:\w\stage\tools\mystran\mystran.exe"], seen)
    with pytest.raises(SystemExit, match="the setup compiler packed"):
        mp.write_setup_exe(tmp_path / "stage", tmp_path, "1.5.1", "ISCC.exe")
    assert not (tmp_path / "FEAStudio-1.5.1-setup.exe").exists()


def test_mystran_is_not_a_licensed_component_of_the_packages():
    assert not [e for e in cl.EXTERNAL if "mystran" in (e["name"] + e["slug"]).lower()]
    assert not (ROOT / "installer" / "license_texts" / "mystran").exists()
    assert "| MYSTRAN |" not in (ROOT / "installer" / "license_texts" / "SOURCES.md").read_text(encoding="utf-8")
    third = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    assert "## Not distributed: MYSTRAN" in third and "No package of\nFEA Studio carries it" in third.replace("\r\n", "\n")


def test_the_verify_module_does_not_need_mystran(tmp_path, monkeypatch):
    monkeypatch.setattr(modules.config, "TOOLS_DIR", tmp_path)                    # no tools/mystran
    monkeypatch.setattr(modules, "_has_import", lambda name: True)
    assert modules.available("verify")


def test_the_installer_says_how_to_get_mystran_and_does_not_warn(tmp_path, monkeypatch, capsys):
    import fea_setup
    monkeypatch.setattr(fea_setup, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(fea_setup, "PROBLEMS", [])
    monkeypatch.setattr(fea_setup, "_log", lambda text: None)
    fea_setup.install_ext_mystran()
    text = capsys.readouterr().out
    assert "https://www.mystran.com" in text and str(tmp_path / "tools" / "mystran") in text and fea_setup.PROBLEMS == []


def test_removing_the_verify_module_keeps_the_users_own_copy(tmp_path, monkeypatch):
    import fea_setup
    monkeypatch.setattr(fea_setup, "TOOLS", tmp_path / "tools")
    own = _put(tmp_path, "tools/mystran/mystran.exe", b"downloaded by the user")
    fea_setup.remove_ext("verify")
    assert own.read_bytes() == b"downloaded by the user"


def test_run_all_reports_a_mystran_case_as_not_run(tmp_path, monkeypatch, capsys):
    pytest.importorskip("OCP")
    from verification import run_all
    monkeypatch.setattr(mystran, "MYSTRAN_EXE", tmp_path / "none" / "mystran.exe")
    monkeypatch.setattr(run_all, "OUT", tmp_path)
    monkeypatch.setattr(run_all, "run_study", lambda *a, **k: pytest.fail("nothing may be computed"))
    assert [k for k in ("F3", "F3M", "F9M", "F11M", "V5") if run_all.needs_mystran(k)] == ["F3M", "F9M", "F11M"]
    assert run_all.run(["F9M", "F3M", "F11M"]) is True                            # not a failure, not a result
    out = capsys.readouterr().out
    assert out.count("not run – MYSTRAN not installed") == 4 and "not run – MYSTRAN not installed: F9M, F3M, F11M" in out
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == {}
    verdict = [ln for ln in out.splitlines() if ln.startswith("VÝSLEDEK:")]       # nothing ran: the verdict must not say that the checks passed
    assert verdict == ["VÝSLEDEK: nic nebylo zkontrolováno – žádná z vybraných úloh se nespustila"] and "prošly" not in out


def test_run_all_verdict_never_calls_a_skipped_case_passed():
    pytest.importorskip("OCP")
    from verification import run_all
    assert run_all._verdict(True, 3, 0) == "všechny kontroly prošly"
    assert run_all._verdict(True, 3, 3).startswith("nic nebylo zkontrolováno")
    assert "všechny" not in run_all._verdict(True, 3, 1) and "nespustily" in run_all._verdict(True, 3, 1)
    for skipped in (0, 1):                                                         # a failed case wins over any skipped one
        assert run_all._verdict(False, 3, skipped) == "některé kontroly NEPROŠLY"


def test_run_public_without_mystran_skips_its_cases_and_runs_no_cross_check(tmp_path, monkeypatch, capsys):
    pytest.importorskip("OCP")
    from verification import mystran_check, public_report, run_all, run_public
    monkeypatch.setattr(mystran, "MYSTRAN_EXE", tmp_path / "none" / "mystran.exe")
    keys, skipped = run_public.select(["V1", "F9M", "F3M"], quick=False)
    assert keys == ["V1"] and skipped == {"F9M": mystran.NOT_RUN, "F3M": mystran.NOT_RUN}
    monkeypatch.setattr(run_public, "OUT", tmp_path)
    monkeypatch.setattr(run_public, "META", tmp_path / "run_meta.json")
    monkeypatch.setattr(run_public, "solver_info", lambda: {"ccx": "CalculiX", "mystran": ""})
    monkeypatch.setattr(run_all, "run", lambda keys, merge=False: True)
    monkeypatch.setattr(mystran_check, "run", lambda *a, **k: pytest.fail("the cross-check must not start"))
    seen: list = []
    monkeypatch.setattr(public_report, "main", lambda argv=None: seen.append(argv) or 0)
    assert run_public.main(["--only", "V1"]) == 0 and seen == [["--no-mystran"]]
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["mystran_missing"] is True and meta["mystran_ran"] is False
    assert "not run – MYSTRAN not installed" in capsys.readouterr().out


def test_the_page_of_a_run_without_mystran(tmp_path):
    from tests.test_verification_public import _fake_report, _meta
    from verification import catalogue, public_report
    rep = _fake_report()
    rep["V5"] = {"title": "w", "time_s": 1.0, "nodes": 10, "elements": 5, "mesh_size_mm": 25.0, "warnings": [],
                 "checks": [{"name": "reakce Fz", "value": 1.0, "ref": 1.0, "error": 0.0, "tol": 0.001, "unit": "N", "ok": True}]}
    meta = {**_meta(), "mystran": "", "mystran_missing": True, "skipped": {"F9M": mystran.NOT_RUN}}
    m = public_report.build(rep, {}, meta)
    by = {c.entry.key: c for c in m.cases}
    assert by["V1"].status == "incomplete" and by["V1"].note == public_report.CROSS_NOT_RUN          # never a plain pass
    assert by["V2"].status == "incomplete" and by["V3"].status == "fail"                             # V3 stopped with an error
    assert catalogue.get("V5").ref_type == "mystran" and by["V5"].status == "not run" and by["V5"].note == public_report.NOT_RUN_MYSTRAN
    assert by["F9M"].status == "not run" and by["F9M"].note == mystran.NOT_RUN
    cnt = m.counts()
    assert cnt["incomplete"] == 2 and cnt["fail"] == 1 and cnt["pass"] == 0
    md, page = public_report.readme_text(m), public_report.html_text(m)
    assert "2 more are within tolerance but without their MYSTRAN cross-check" in md and "✔ (no MYSTRAN cross-check)" in md
    assert "within tolerance, no MYSTRAN cross-check" in page
    for text in (md, page):
        assert "FEA Studio does not ship it" in text
    p = tmp_path / "results.csv"
    p.write_text(public_report.csv_text(m), encoding="utf-8")
    assert {r["case_status"] for r in csv.DictReader(p.read_text(encoding="utf-8").splitlines())} == {"incomplete", "fail", "not run"}
    s = verification_summary.read_summary(p)
    assert s is not None and s.passed == 2 and s.failed == 1
    # a failed check stays outside tolerance, and with the program installed a missing comparison is still not a pass
    assert next(c for c in public_report.build(_fake_report(ok=False), {}, meta).cases if c.entry.key == "V2").status == "fail"
    assert next(c for c in public_report.build(_fake_report(), {}, _meta()).cases if c.entry.key == "V1").status == "fail"


def test_the_committed_page_says_that_mystran_is_a_separate_download():
    from verification import public_report
    for name in ("README.md", "index.html"):
        text = (ROOT / "docs" / "verification" / name).read_text(encoding="utf-8")
        assert "FEA Studio does not ship it" in text and "Cross-check against MYSTRAN" in text, name     # the rows of the run stay
    assert public_report.MYSTRAN_NOTE in (ROOT / "docs" / "verification" / "README.md").read_text(encoding="utf-8")


def test_mystran_check_without_the_program_says_not_run(tmp_path):
    pytest.importorskip("OCP")
    code = ("import sys, runpy; from pathlib import Path; from fea_core import mystran; mystran.MYSTRAN_EXE = Path(sys.argv[1]); "
            "sys.argv = ['mystran_check', 'V1']; runpy.run_module('verification.mystran_check', run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path / "mystran.exe")], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}, timeout=300)
    assert r.returncode == 0 and "not run – MYSTRAN not installed" in r.stdout, r.stdout + r.stderr


# ------------------------------------------------------------------ 2. the sources travel with the binaries; no written offer
def test_there_is_no_written_offer_in_the_release_texts(tmp_path):
    cl.write_sources(tmp_path, ROOT, "2026-11-26")
    text = (tmp_path / "SOURCES.md").read_text(encoding="utf-8")
    third = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    assert "written offer" not in text.lower() and "issue tracker" not in text.lower() and cl.NOT_FOR_DISTRIBUTION not in text
    assert "there is no written offer" in third and "written offer only" not in " ".join(third.split()) and "a written offer" not in third


@pytest.mark.parametrize("flags, platforms", [((True, False, False, False), "Windows and Linux"), ((True, False, True, False), "of the Linux packages"),
                                              ((True, False, False, True), "of the Windows packages")])
def test_leaving_out_runtime_sources_blocks_distribution(flags, platforms):
    reasons = mp.distribution_blockers(*flags)
    assert len(reasons) == 1 and "--no-runtime-sources" in reasons[0] and platforms in reasons[0]


def test_what_blocks_distribution_and_what_does_not():
    assert mp.distribution_blockers(False, False, False, False) == []
    assert mp.distribution_blockers(False, False, True, True) == []                    # no CalculiX, nothing left out
    assert mp.distribution_blockers(True, False, True, True) == []                     # no binary that would need the sources
    assert mp.distribution_blockers(False, True, False, True) == []                    # ccx_pastix only goes into a Linux package that has a CalculiX
    assert "--with-ccx-pastix" in mp.distribution_blockers(False, True, False, False)[0]
    assert len(mp.distribution_blockers(True, True, False, False)) == 2
    assert mp.release_name("1.5.1", False) == "FEAStudio-1.5.1" and mp.release_name("1.5.1", True) == "FEAStudio-1.5.1-dev"


def _licences(stage):
    _put(stage, "licenses/SOURCES.md", b"# Sources of the GPL / LGPL components in this package\n\ntext of a release\n")


def _build_with_licences(tmp_path, monkeypatch, *args):
    real_setattr = monkeypatch.setattr

    def setattr_(target, name, value, *a, **k):        # the harness replaces make_licenses by a no-op; here it writes SOURCES.md
        real_setattr(target, name, (lambda stage: _licences(stage)) if name == "make_licenses" else value, *a, **k)

    monkeypatch.setattr = setattr_
    try:
        return _build(tmp_path, monkeypatch, *args)
    finally:
        monkeypatch.setattr = real_setattr


def _sources_md(out: Path, stem: str) -> tuple[str, str]:
    with zipfile.ZipFile(out / f"{stem}.zip") as z:
        a = z.read(f"{stem}/licenses/SOURCES.md").decode("utf-8")
    with tarfile.open(out / f"{stem}.tar.gz") as t:
        b = t.extractfile(f"{stem}/licenses/SOURCES.md").read().decode("utf-8")
    return a, b


def test_a_build_without_the_sources_is_marked_and_named_dev(tmp_path, monkeypatch, capsys):
    out, ver = _build_with_licences(tmp_path, monkeypatch, "--no-runtime-sources")
    stem = f"FEAStudio-{ver}-dev"
    assert {p.name for p in out.iterdir() if p.is_file()} == {"SHA256SUMS.txt", f"{stem}.tar.gz", f"{stem}.zip"}
    for text in _sources_md(out, stem):                                      # both platforms: the notice is the first thing of the document
        assert text.splitlines()[2] == "## NOT FOR DISTRIBUTION" and "--no-runtime-sources" in text and "text of a release" in text
    assert stem + ".zip" in (out / "SHA256SUMS.txt").read_text(encoding="utf-8")
    log = capsys.readouterr().out
    assert log.count("NOT FOR DISTRIBUTION") >= 2 and f"{stem}.zip" in log          # at the start and at the end of the build log


def test_a_release_build_keeps_its_names_and_has_no_notice(tmp_path, monkeypatch, capsys):
    out, ver = _build_with_licences(tmp_path, monkeypatch)
    assert {p.name for p in out.iterdir() if p.is_file()} == {"SHA256SUMS.txt", f"FEAStudio-{ver}.tar.gz", f"FEAStudio-{ver}.zip"}
    for text in _sources_md(out, f"FEAStudio-{ver}"):
        assert "NOT FOR DISTRIBUTION" not in text
    assert "NOT FOR DISTRIBUTION" not in capsys.readouterr().out


def test_the_run_installer_and_setup_exe_of_a_dev_build_are_named_dev(tmp_path, monkeypatch):
    seen: list = []
    _put(tmp_path / "stage", "README.md")
    _fake_iscc(monkeypatch, tmp_path, [], seen)
    exe = mp.write_setup_exe(tmp_path / "stage", tmp_path, "1.5.1", "ISCC.exe", dev=True)
    assert exe is not None and exe.name == "FEAStudio-1.5.1-dev-setup.exe" and "/DDevSuffix=-dev" in seen[0] and "/DAppVersion=1.5.1" in seen[0]
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    assert iss.count("AppVersion + DevSuffix + ") == 2 and "#ifndef DevSuffix" in iss
    source = (ROOT / "installer" / "make_package.py").read_text(encoding="utf-8")
    assert 'out / f"{prefix}-linux-x86_64.run"' in source and "prefix = release_name(ver, dev)" in source

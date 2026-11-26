"""Guard of the public verification page (F15-S2): docs/verification/ is built from verification.catalogue and must

  - contain nothing private (the cross-check against the commercial solver, its case keys X1–X5, local paths),
  - cover every published case and cite an independent reference for every row,
  - be English and match the stored snapshot of the run (no failing case),
  - stay exportable: the export of the public repository leaves out verification/abaqus_* but takes docs/verification/.

The page itself is produced by `python -m verification.run_public` (a real run); these tests only read the committed files.
"""
from __future__ import annotations

import csv
import fnmatch
import importlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fea_core import verification_summary
from verification import catalogue, catalogue_labels, public_report, run_public

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "verification"
FILES = ("README.md", "results.csv", "index.html")

_PRIVATE_NAME = re.compile(r"abaqus", re.I)               # the private cross-check: its name and its files (abaqus_*)
_PRIVATE_KEYS = re.compile(r"(?<![A-Za-z0-9_])X[1-5](?![A-Za-z0-9_])")      # its cases X1–X5 (case-sensitive: "x1" may occur in text)
LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\[A-Za-z]|/ho" r"me/|/Us" r"ers/|/mn" r"t/[a-z]/|App" r"Data")   # written in pieces: the packaging scanner must not see a path here
CZECH = re.compile(r"[ěščřžůďťňýĚŠČŘŽŮĎŤŇÝ]")        # letters of Czech that no author name (Lamé, Hetényi ...) has


class _Private:
    """`PRIVATE.search(text)` = the first private word of the text, or None."""

    @staticmethod
    def search(text: str):
        return _PRIVATE_NAME.search(text) or _PRIVATE_KEYS.search(text)


PRIVATE = _Private()


def _page(name: str) -> str:
    p = PAGE / name
    if not p.is_file():
        pytest.fail(f"docs/verification/{name} is missing: python -m verification.run_public")
    return p.read_text(encoding="utf-8")


def _rows() -> list[dict]:
    with open(PAGE / "results.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------------------- the catalogue
def test_catalogue_table_is_consistent():
    assert catalogue.problems() == []


def test_every_case_has_a_catalogue_entry():
    pytest.importorskip("OCP")
    from verification import cases
    keys = set(cases.CASES)
    assert keys - set(catalogue.ENTRIES) == set(), "a verification case without a catalogue entry: classify it in verification/catalogue.py"
    # SK6 is registered only when the optional module sketch (ezdxf) is installed
    optional = set() if "SK6" in keys else {"SK6"}
    assert set(catalogue.ENTRIES) - keys - optional == set(), "a catalogue entry of a case that no longer exists"


def test_published_cases_have_a_citation_and_no_private_words():
    assert len(catalogue.public_keys()) >= 80
    for e in catalogue.ENTRIES.values():
        text = " ".join((e.title, e.source, *e.hide))
        assert not PRIVATE.search(text), e.key
        if e.public:
            assert e.ref_type in catalogue.PUBLIC_TYPES and len(e.source.strip()) >= 10, e.key
            assert not CZECH.search(e.title + e.source), f"{e.key}: the title and the citation are English"
        else:
            assert e.ref_type == "internal" and e.why_not.strip(), e.key


def test_a_cross_check_case_is_a_published_case():
    for k in catalogue.MYSTRAN_CROSS_CHECK:
        assert catalogue.get(k).public, k


# ---------------------------------------------------------------------------------------------------- the generated page
@pytest.mark.parametrize("name", FILES)
def test_page_has_nothing_private_and_no_local_path(name):
    text = _page(name)
    assert not PRIVATE.search(text), PRIVATE.search(text).group(0)
    assert not LOCAL_PATH.search(text), LOCAL_PATH.search(text).group(0)


def test_page_covers_every_published_case_and_cites_every_row():
    rows = _rows()
    assert {r["case"] for r in rows} == set(catalogue.public_keys())
    for r in rows:
        assert r["reference_type"] in catalogue.PUBLIC_TYPES, r["case"]
        assert len(r["source"].strip()) >= 10, f"{r['case']}: a published row without a citation"
        assert r["case_status"] in ("pass", "fail", "not run"), r["case"]
        if r["case_status"] != "not run":
            assert r["quantity"] and r["check_status"] in ("ok", "FAIL"), r
    cited = {r["case"]: r["source"] for r in rows}
    for k in catalogue.public_keys():
        assert cited[k], k


def test_page_is_english():
    for r in _rows():
        for col in ("title", "source", "quantity", "unit", "note"):
            assert not CZECH.search(r[col]), f"{r['case']}: Czech text in {col}: {r[col]}"
    assert not CZECH.search(_page("index.html")) and not CZECH.search(_page("README.md"))


def test_page_matches_a_run_without_failures():
    s = verification_summary.read_summary(PAGE / "results.csv")
    assert s is not None and s.passed > 0
    assert s.failed == 0, "the committed snapshot has a case outside its tolerance"
    assert s.cases == len(catalogue.public_keys())
    assert s.version and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.date)
    for r in _rows():
        if r["case_status"] == "pass":
            assert r["check_status"] != "FAIL", r["case"]


def test_page_is_self_contained_and_themed():
    html = _page("index.html")
    assert html.startswith("<!doctype html>")
    assert "prefers-color-scheme:dark" in html and "name=\"viewport\"" in html
    assert not re.search(r"(?:src|href)=[\"']https?://", html) and "<script" not in html


# ---------------------------------------------------------------------------------------------------- the export and the package
def _export_excluded(path: str, patterns: list[str]) -> bool:
    """The rule of scratch/github_export/github_export.py (export_list)."""
    return any(fnmatch.fnmatch(path, p) or path.startswith(p.rstrip("*")) for p in patterns)


def _git_info() -> Path:
    """`.git/info` of the repository, also from a worktree (where `.git` is a file and the folder is the common git dir)."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    base = (ROOT / out) if out else ROOT / ".git"
    return base / "info"


def test_export_takes_the_page_and_leaves_the_private_cross_check_out():
    excl = _git_info() / "github-export-exclude.txt"
    if not excl.is_file():
        pytest.skip("local export configuration missing (only in the development tree)")
    patterns = [ln.strip() for ln in excl.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip() and not ln.startswith("#")]
    for p in ("docs/verification/README.md", "docs/verification/results.csv", "docs/verification/index.html",
              "verification/catalogue.py", "verification/public_report.py", "verification/run_public.py",
              ".github/workflows/verification.yml"):
        assert not _export_excluded(p, patterns), f"{p} would be left out of the public repository"
    for p in ("verification/abaqus_check.py", "verification/abaqus_extra.py", "verification/out_abaqus/report.json",
              "tests/test_abaqus_all.py", "docs/Srovnani_Abaqus.html"):
        assert _export_excluded(p, patterns), f"{p} must stay private"


def test_package_carries_the_page_but_not_the_private_cross_check():
    sys.path.insert(0, str(ROOT / "installer"))
    try:
        mp = importlib.import_module("make_package")
    finally:
        sys.path.remove(str(ROOT / "installer"))
    for p in ("docs/verification/results.csv", "docs/verification/index.html", "docs/verification/README.md",
              "verification/catalogue.py", "fea_core/verification_summary.py"):
        assert mp.packaged(p), p
    assert not mp.packaged("verification/abaqus_check.py")


# ---------------------------------------------------------------------------------------------------- the report builder
def _fake_report(ok: bool = True) -> dict:
    return {"V1": {"title": "x", "time_s": 3.0, "nodes": 11000, "elements": 7000, "mesh_size_mm": 2.5, "warnings": [],
                   "checks": [{"name": "průhyb konce (Timoshenko)", "value": 0.76, "ref": 0.7634, "error": 0.004, "tol": 0.03,
                               "unit": "mm", "ok": True},
                              {"name": "reakce Fz", "value": 100.0, "ref": 100.0, "error": 0.0, "tol": 0.001, "unit": "N", "ok": True}]},
            "V2": {"title": "y", "time_s": 1.0, "nodes": 900, "elements": 500, "mesh_size_mm": 3.0, "warnings": [],
                   "checks": [{"name": "σx (střed)", "value": -100.0, "ref": -100.0, "error": 0.0, "tol": 0.001, "unit": "MPa", "ok": True},
                              {"name": "reakce Fx", "value": 9000.0, "ref": 10000.0, "error": 0.1, "tol": 0.001, "unit": "N", "ok": ok}]},
            "V3": {"error": "boom at C:" + chr(92) + "Us" + "ers" + chr(92) + "someone" + chr(92) + "proj" + chr(92) + "x.py line 3 and " + "/ho" + "me/someone/y"}}


def _meta() -> dict:
    return {"program_version": "9.9.9", "date_utc": "2026-01-02 03:04", "ccx": "CalculiX 2.23", "mystran": "MYSTRAN 19.0.0",
            "platform": "TestOS 1 (x86_64)", "python": "3.11.9", "cpus": 8, "profile": "full", "duration_s": 120.0,
            "skipped": {"K2": "needs mortar"}}


def _mystran_ok() -> dict:
    """A passing MYSTRAN comparison for every case that the page promises it for."""
    row = {"case": "F", "u_max_ccx": 1.0, "u_max_mystran": 1.0, "du_rel": 1e-9, "dr_rel": 1e-9, "nodes": 100, "mesh_size_mm": 5.0}
    return {k: [dict(row)] for k in catalogue.MYSTRAN_CROSS_CHECK}


def test_report_builder_statuses_and_hidden_checks():
    m = public_report.build(_fake_report(ok=False), _mystran_ok(), _meta())
    by = {c.entry.key: c for c in m.cases}
    assert by["V1"].status == "pass" and by["V2"].status == "fail"
    assert by["V3"].status == "fail" and public_report.WITHHELD in by["V3"].note and "someone" not in by["V3"].note
    assert by["K2"].status == "not run" and by["K2"].note == "needs mortar"
    assert by["V4"].status == "not run"                                # a published case without a result is never "pass"
    cnt = m.counts()
    assert cnt["pass"] == 1 and cnt["fail"] == 2 and cnt["cases"] == len(catalogue.public_keys())


def test_a_failing_hidden_check_fails_the_case():
    rep = _fake_report()
    rep["V6"] = {"title": "z", "time_s": 1.0, "nodes": 10, "elements": 5, "mesh_size_mm": 2.0, "warnings": [],
                 "checks": [{"name": "průhyb konce (Timoshenko)", "value": 0.76, "ref": 0.7634, "error": 0.004, "tol": 0.03,
                             "unit": "mm", "ok": True},
                            {"name": "průhyb proti V1 (jiná síť)", "value": 1.0, "ref": 0.76, "error": 0.3, "tol": 0.005, "unit": "mm",
                             "ok": False}]}
    c = next(c for c in public_report.build(rep, _mystran_ok(), _meta()).cases if c.entry.key == "V6")
    assert c.status == "fail" and c.n_hidden == 1 and len(c.rows) == 1


def test_a_failed_mystran_comparison_is_not_dropped():
    rep = _fake_report()
    row = {"case": "F", "u_max_ccx": 1.0, "u_max_mystran": 1.0, "du_rel": 5e-3, "dr_rel": 1e-9, "nodes": 100, "mesh_size_mm": 5.0}
    assert next(c for c in public_report.build(rep, {"V1": [row]}, _meta()).cases if c.entry.key == "V1").status == "fail"
    assert next(c for c in public_report.build(rep, {"V1": {"error": "no op2"}}, _meta()).cases if c.entry.key == "V1").status == "fail"
    assert next(c for c in public_report.build(rep, _mystran_ok(), _meta()).cases if c.entry.key == "V1").status == "pass"


def test_generated_text_is_escaped_and_clean():
    m = public_report.build(_fake_report(), {}, _meta())
    m.cases[0].rows[0] = public_report.Row("=HYPERLINK(1)", "mm", 1.0, 1.0, 0.0, 0.01, True, False, "analytic", "=src")
    m.cases[0].entry = replace(m.cases[0].entry, title="<b>title</b>", source="a | b <i>src</i>")
    csv_out = public_report.csv_text(m)
    assert "'=HYPERLINK(1)" in csv_out and "'=src" in csv_out
    assert not any(line.startswith("=") for line in csv_out.splitlines())
    page = public_report.html_text(m)
    assert "<b>title</b>" not in page and "&lt;b&gt;title&lt;/b&gt;" in page and "<i>src</i>" not in page
    md = public_report.readme_text(m)
    assert r"a \| b" in md
    assert "someone" not in md + page + csv_out
    assert not PRIVATE.search(md + page + csv_out)


def test_csv_rows_have_one_field_per_column():
    m = public_report.build(_fake_report(ok=False), {"V1": [{"case": "F", "u_max_ccx": 1.0, "u_max_mystran": 1.0, "du_rel": 1e-9,
                                                              "dr_rel": 1e-9, "nodes": 10, "mesh_size_mm": 5.0}]}, _meta())
    rows = list(csv.DictReader(public_report.csv_text(m).splitlines()))
    assert rows and all(None not in r and None not in r.values() for r in rows)          # no row longer or shorter than the header
    assert {r["case_status"] for r in rows} == {"pass", "fail", "not run"}
    nr = next(r for r in rows if r["case"] == "K2")
    assert nr["case_status"] == "not run" and nr["note"] == "needs mortar" and nr["nodes"] == "" and nr["quantity"] == ""
    assert next(r for r in rows if r["case"] == "V1" and r["reference_type"] == "mystran")["case_status"] == "pass"


def test_page_text_is_deterministic():
    a, b = public_report.build(_fake_report(), {}, _meta()), public_report.build(_fake_report(), {}, _meta())
    assert public_report.html_text(a) == public_report.html_text(b) and public_report.csv_text(a) == public_report.csv_text(b)


def test_paths_with_spaces_are_never_published():
    """Review I8: a path with spaces cannot be cut out of a message - a message with a path separator is withheld whole."""
    bad = ["failed to open C:" + chr(92) + "Us" + "ers" + chr(92) + "Jane Doe" + chr(92) + "My Projects" + chr(92) + "x.step at line 3",
           "cannot read /ho" + "me/jane doe/my projects/x.step",
           "cannot read " + chr(92) * 2 + "server" + chr(92) + "share name" + chr(92) + "x.step"]
    for text in bad:
        assert public_report._clean(text) == public_report.WITHHELD, text
        assert "Jane" not in public_report._clean(text)
    assert public_report._clean("the  solver\n stopped: singular matrix") == "the solver stopped: singular matrix"
    rep = _fake_report()
    rep["V4"] = {"error": bad[0]}
    c = next(c for c in public_report.build(rep, {}, _meta()).cases if c.entry.key == "V4")
    assert "Jane" not in c.note and "Doe" not in c.note and public_report.WITHHELD in c.note


def test_headline_counts_only_the_listed_checks():
    """Review I5: the hidden consistency checks are not part of the number on the page."""
    rep = _fake_report()
    rep["V6"] = {"title": "z", "time_s": 1.0, "nodes": 10, "elements": 5, "mesh_size_mm": 2.0, "warnings": [],
                 "checks": [{"name": "průhyb konce (Timoshenko)", "value": 0.76, "ref": 0.7634, "error": 0.004, "tol": 0.03,
                             "unit": "mm", "ok": True},
                            {"name": "průhyb proti V1 (jiná síť)", "value": 1.0, "ref": 0.76, "error": 0.3, "tol": 0.005, "unit": "mm",
                             "ok": True}]}
    m = public_report.build(rep, _mystran_ok(), _meta())
    assert m.counts()["checks"] == sum(len(c.rows) for c in m.cases) == 5            # V1: 2, V2: 2, V6: 1 (the hidden one is not counted)
    assert "5 checks listed" in public_report.readme_text(m) and ">5</b>checks listed" in public_report.html_text(m)


def test_bookkeeping_rows_are_hidden_and_the_weak_cases_are_not_published():
    """Review I5 / I7: equilibrium rows are not shown; S1 and F4 have no independent reference; V5 is a MYSTRAN case."""
    shown = {"V10": ["(a) tuhá, bod v těžišti: reakce Fz v bodě", "(a') distribuční, těžiště: reakce Fz v bodě"],
             "V17": ["plocha *TIE = průřez 20×10", "reakce Fz"], "K1": ["normálová síla páru = p·A"], "K1c": ["normálová síla páru = p·A"],
             "F9": ["statika: tlak = reakce"], "P1": ["reakce Fx (obě tyče)"],
             "Sh1": ["reakce Fz", "reakce My k ref. bodu (rameno síly)", "tíha: reakce Fz = ρ·t·A·g"],
             "PS1": ["PS1 mass of the beam (ρ·V)"], "M2": ["celková hmota ρV (volné těleso – žádné vázané uzly)"],
             "Y15": ["hmotnost modelu m + M"], "IM1": ["IM1 reakce Fz na vetknuti (p*A)", "IM1 reakcni moment My k c0 (v ose zatizeni = 0)"],
             "IM2": ["IM2 reaction Fz = p·A"], "T3": ["zdroj (integrace)"], "F10": ["4 prvků: průhyb konce = 1 prvek"],
             "V5": ["reakce Fz"]}
    for key, names in shown.items():
        for n in names:
            assert not catalogue.get(key).shows(n), f"{key}: {n}"
    assert catalogue.get("V10").shows("(c) distribuční kloub: reakce Fz v bodě = qL/2")        # a closed-form value of a determinate beam
    assert catalogue.get("V1").shows("reakce Fz")
    assert not catalogue.get("S1").public and not catalogue.get("F4").public
    assert "S1" not in catalogue.public_keys() and "F4" not in catalogue.public_keys()
    assert catalogue.get("V5").ref_type == "mystran" and catalogue.get("V5").source_kind == "solver" and "V5" in catalogue.MYSTRAN_CROSS_CHECK


def test_source_kinds_are_told_apart():
    """Review I6: a cited work, a stated first-principles relation, an independent solver - and a guard for each."""
    assert catalogue.get("V1").source_kind == "work" and catalogue.get("V4").source_kind == "work"
    assert catalogue.get("V2").source_kind == "relation" and catalogue.get("T4").source_kind == "relation"
    assert catalogue.get("V5").source_kind == "solver"
    assert all(e.source_kind for e in catalogue.ENTRIES.values() if e.public)
    mk = lambda ref, src: catalogue.Entry("ZZ", "t", "statics", ref, src, True)             # noqa: E731
    assert mk("analytic", "Uniform pressure: p = F/A").source_kind == "relation"
    assert mk("analytic", "see the manual").source_kind == ""
    assert mk("literature", "S. P. Timoshenko: Strength of Materials").source_kind == "work"
    saved = catalogue.ENTRIES
    try:
        catalogue.ENTRIES = {"A": mk("handbook", "just a formula: p = F/A"), "B": mk("analytic", "see the manual and the notes")}
        probs = catalogue.problems()
    finally:
        catalogue.ENTRIES = saved
    assert any("must cite a work" in p for p in probs) and any("names no work" in p for p in probs)
    assert "relation stated, no work cited" in public_report.readme_text(public_report.build(_fake_report(), _mystran_ok(), _meta()))


def test_a_cross_check_case_without_a_mystran_entry_is_not_a_clean_pass():
    """Review I9: the page promises the MYSTRAN comparison for these cases; a run without it does not pass silently."""
    c = next(c for c in public_report.build(_fake_report(), {}, _meta()).cases if c.entry.key == "V1")
    assert c.status == "fail" and "without its cross-check against MYSTRAN" in c.note
    c = next(c for c in public_report.build(_fake_report(), _mystran_ok(), _meta()).cases if c.entry.key == "V1")
    assert c.status == "pass" and not c.note
    other = _fake_report()
    other["K1"] = {**_fake_report()["V1"]}                                             # not a cross-check case: no MYSTRAN entry needed
    assert next(c for c in public_report.build(other, _mystran_ok(), _meta()).cases if c.entry.key == "K1").status == "pass"


def test_every_case_carries_the_date_and_version_of_its_own_run():
    """Review I9: a merged run holds cases of several runs; the csv says which run made which case."""
    meta = {**_meta(), "case_runs": {"V1": {"date_utc": "2026-02-03 01:00", "program_version": "1.0.0"},
                                     "V2": {"date_utc": "2026-05-06 09:00", "program_version": "1.1.0"}}}
    m = public_report.build(_fake_report(), _mystran_ok(), meta)
    rows = list(csv.DictReader(public_report.csv_text(m).splitlines()))
    v1 = {(r["run_date"], r["program_version"]) for r in rows if r["case"] == "V1" and r["reference_type"] != "mystran"}
    v2 = {(r["run_date"], r["program_version"]) for r in rows if r["case"] == "V2"}
    assert v1 == {("2026-02-03", "1.0.0")} and v2 == {("2026-05-06", "1.1.0")}
    md = public_report.readme_text(m)
    assert "2026-02-03 to 2026-05-06" in md and "1.0.0, 1.1.0" in md
    one = public_report.readme_text(public_report.build(_fake_report(), _mystran_ok(), _meta()))
    assert "2026-01-02 03:04" in one and "several runs" not in one


def test_the_summary_line_names_the_oldest_run(tmp_path):
    good = tmp_path / "ok.csv"
    good.write_text("case,case_status,run_date,program_version\nV1,pass,2026-05-06,1.1.0\nV2,pass,2026-02-03,1.0.0\n", encoding="utf-8")
    s = verification_summary.read_summary(good)
    assert s is not None and (s.date, s.version) == ("2026-02-03", "1.0.0")


def test_merged_run_keeps_the_runs_of_the_cases_it_does_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_public, "OUT", tmp_path)
    monkeypatch.setattr(run_public, "META", tmp_path / "run_meta.json")
    monkeypatch.setattr(run_public, "solver_info", lambda: {"ccx": "CalculiX", "mystran": ""})
    keys = catalogue.public_keys()[:3]
    first = run_public.write_meta({}, keys[:2], 10.0, False, False, mystran_ran=True)
    assert set(first["case_runs"]) == set(keys[:2]) and first["mystran_ran"] is True
    # an earlier meta without the per-case record: its own date and version hold for its cases
    old = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    old.pop("case_runs")
    old["date_utc"], old["program_version"] = "2026-01-01 00:00", "0.9.0"
    (tmp_path / "run_meta.json").write_text(json.dumps(old), encoding="utf-8")
    second = run_public.write_meta({}, keys[2:3], 5.0, False, True, mystran_ran=False)
    assert {k: v["program_version"] for k, v in second["case_runs"].items() if k in keys[:2]} == {keys[0]: "0.9.0", keys[1]: "0.9.0"}
    assert second["case_runs"][keys[2]]["program_version"] != "0.9.0" and second["mystran_ran"] is True   # kept from the merged run
    third = run_public.write_meta({}, keys[:1], 5.0, False, False, mystran_ran=False)               # not merged: a fresh meta
    assert set(third["case_runs"]) == set(keys[:1]) and third["mystran_ran"] is False


def test_run_public_returns_the_exit_code_of_the_report(monkeypatch):
    pytest.importorskip("OCP")
    from verification import run_all
    called = []
    monkeypatch.setattr(run_all, "run", lambda keys, merge=False: True)
    monkeypatch.setattr(run_public, "write_meta", lambda *a, **k: {"duration_s": 0.0, "mystran_ran": False})
    monkeypatch.setattr(public_report, "main", lambda argv=None: (called.append(argv) or 1))
    assert run_public.main(["--only", "V1", "--no-mystran"]) == 1                      # the page has a failing case: not a success
    assert called == [["--no-mystran"]]                                                # the stored MYSTRAN comparison is not read
    monkeypatch.setattr(run_public, "write_meta", lambda *a, **k: {"duration_s": 0.0, "mystran_ran": True})
    monkeypatch.setattr(public_report, "main", lambda argv=None: (called.append(argv) or 0))
    assert run_public.main(["--only", "V1", "--no-mystran"]) == 0 and called[-1] == []


def test_summary_reader(tmp_path):
    assert verification_summary.read_summary(tmp_path / "missing.csv") is None
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    assert verification_summary.read_summary(bad) is None
    good = tmp_path / "ok.csv"
    head = "case,case_status,run_date,program_version\n"
    good.write_text(head + "V1,pass,2026-01-02,9.9.9\nV1,pass,2026-01-02,9.9.9\nV2,pass,2026-01-02,9.9.9\nK2,not run,,\n",
                    encoding="utf-8")
    s = verification_summary.read_summary(good)
    assert s is not None and (s.cases, s.passed, s.failed, s.not_run) == (3, 2, 0, 1) and s.all_ok
    assert (s.date, s.version) == ("2026-01-02", "9.9.9")
    good.write_text(head + "V1,pass,,\nV2,fail,,\n", encoding="utf-8")
    s2 = verification_summary.read_summary(good)
    assert s2 is not None and s2.failed == 1 and not s2.all_ok


# ---------------------------------------------------------------------------------------------------- run_public and the labels
def test_run_public_selection():
    keys, skipped = run_public.select(["V1", "V2"], quick=False)
    assert keys == ["V1", "V2"] and skipped == {}
    keys, skipped = run_public.select(None, quick=True)
    from verification import tiers
    assert not set(keys) & tiers.VERY_SLOW
    assert {"KK4", "KK2"} <= set(skipped) or not ({"KK4", "KK2"} & set(catalogue.public_keys()))
    assert set(keys) | set(skipped) == set(catalogue.public_keys())


def test_the_multi_hour_case_runs_only_on_request():
    keys, skipped = run_public.select(None, quick=False)
    assert "KK4" not in keys and "hours" in skipped["KK4"]
    keys, skipped = run_public.select(["KK4"], quick=False)
    assert keys == ["KK4"] and not skipped


def test_unavailable_capability_is_reported_not_run():
    e = catalogue.Entry("ZZ", "t", "statics", "analytic", "a citation", True, needs=("no_such_module",))
    assert "no_such_module" in run_public.unavailable(e)


def test_labels_translate_and_keep_unknown_names():
    assert catalogue_labels.is_english("σx at x = 100 mm") and not catalogue_labels.is_english("průhyb konce")
    assert catalogue_labels.label("V1", "reakce Fz") == "reaction Fz"
    assert catalogue_labels.label("V1", "never seen") == "never seen"


def test_hide_patterns_match_something_in_the_last_run():
    """A hide pattern that matches no check of the last run is stale (the check was renamed) – only where a run exists."""
    rep = ROOT / "verification" / "out" / "report.json"
    if not rep.is_file():
        pytest.skip("no stored run (verification/out is not in git)")
    data = json.loads(rep.read_text(encoding="utf-8"))
    for key, e in catalogue.ENTRIES.items():
        names = [c["name"] for c in data.get(key, {}).get("checks", [])]
        if not names:
            continue
        for pat in e.hide:
            assert any(re.search(pat, n) for n in names), f"{key}: the hide pattern {pat!r} matches no check"

"""F14 S6: the Software Validation Dossier (`verification.dossier`): public independent references only, deterministic, and it refuses the private cross-check."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("docx")

from tests.test_verification_public import _fake_report, _meta, _mystran_ok  # noqa: E402
from verification import catalogue, dossier  # noqa: E402


def _text(path) -> str:
    from docx import Document
    d = Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def _write_inputs(tmp_path, report=None, name="report.json"):
    rp, mp, ep = tmp_path / name, tmp_path / "mystran.json", tmp_path / "run_meta.json"
    rp.write_text(json.dumps(report or _fake_report()), encoding="utf-8")
    mp.write_text(json.dumps(_mystran_ok()), encoding="utf-8")
    ep.write_text(json.dumps(_meta()), encoding="utf-8")
    return ["--report", str(rp), "--mystran", str(mp), "--meta", str(ep)]


def test_dossier_chapters_and_public_references(tmp_path):
    args = _write_inputs(tmp_path)
    assert dossier.main([*args, "--out", str(tmp_path / "d.docx"), "--markdown", str(tmp_path / "d.md")]) == 0
    text = _text(tmp_path / "d.docx")
    heads = ["1 Purpose and scope", "2 Method: the kinds of reference", "3 Summary by domain", "4 Results by case", "5 Cross-check against MYSTRAN",
             "6 Internal cross-checks (not independent)", "7 Cases not run and limitations", "8 Traceability"]
    pos = [text.index(h) for h in heads]
    assert pos == sorted(pos)
    assert "V1" in text and catalogue.get("V1").source[:30] in text and "CalculiX against MYSTRAN" in text and "FEA Studio 9.9.9" in text
    md = (tmp_path / "d.md").read_text(encoding="utf-8")
    assert md.startswith("# Software Validation Dossier") and "| Domain | Cases |" in md
    assert not dossier.private_word(text + md)
    assert "NOT counted as validation" in text                                  # the internal checks are never presented as independent


def test_dossier_is_deterministic_and_does_not_use_the_time_of_writing(tmp_path):
    args = _write_inputs(tmp_path)
    for n in ("a", "b"):
        assert dossier.main([*args, "--out", str(tmp_path / f"{n}.docx"), "--markdown", str(tmp_path / f"{n}.md")]) == 0
    assert (tmp_path / "a.docx").read_bytes() == (tmp_path / "b.docx").read_bytes()
    assert (tmp_path / "a.md").read_bytes() == (tmp_path / "b.md").read_bytes()
    assert "2026-01-02" in (tmp_path / "a.md").read_text(encoding="utf-8")        # the date of the run, not of today


def test_dossier_refuses_private_results_and_private_text(tmp_path, capsys):
    rep = _fake_report()
    rep["X1"] = {"title": "private", "checks": [{"name": "q", "value": 1.0, "ref": 1.0, "error": 0.0, "tol": 0.1, "unit": "", "ok": True}]}
    with pytest.raises(dossier.PrivateContentError):
        dossier.build_dossier(rep, {}, _meta())                                  # a case id of the private cross-check
    args = _write_inputs(tmp_path, report=rep)
    assert dossier.main([*args, "--out", str(tmp_path / "x.docx")]) == 3 and not (tmp_path / "x.docx").exists()
    assert "REFUSED" in capsys.readouterr().out
    named = tmp_path / "abaqus_run"
    named.mkdir()
    args = _write_inputs(named)
    assert dossier.main([*args, "--out", str(tmp_path / "y.docx")]) == 3 and not (tmp_path / "y.docx").exists()
    args = _write_inputs(tmp_path, name="abaqus_report.json")
    assert dossier.main([*args, "--out", str(tmp_path / "z.docx")]) == 3
    meta = _meta()
    meta["profile"] = "compared with Abaqus"                                      # private text that arrives through the run metadata
    with pytest.raises(dossier.PrivateContentError):
        dossier.build_dossier(_fake_report(), {}, meta)
    meta = _meta()
    meta["platform"] = "D:" + chr(92) + "work" + chr(92) + "x"                    # a local path in the metadata
    with pytest.raises(dossier.PrivateContentError):
        dossier.build_dossier(_fake_report(), {}, meta)


def test_the_guard_words():
    assert dossier.private_word("see ABAQUS results") and dossier.private_word("case X3 failed") and dossier.private_word("x3 is fine") is None
    assert dossier.private_word("file abaqus_all.py") and dossier.private_word("/ho" "me/someone/x") and dossier.private_word("ordinary text") is None


def test_internal_cross_checks_are_listed_but_not_counted_as_validation(tmp_path):
    key = catalogue.internal_keys()[0]
    rep = _fake_report()
    rep[key] = {"title": "i", "checks": [{"name": "q", "value": 1.0, "ref": 1.0, "error": 0.0, "tol": 0.1, "unit": "", "ok": True}]}
    bl = dossier.build_dossier(rep, _mystran_ok(), _meta())
    internal = next(b for b in bl if b[0] == "table" and b[3] == "Internal cross-checks of the run")
    assert [r[0] for r in internal[2]] == [key]
    assert f"{key} -" not in dossier._all_text([b for b in bl if b[0] == "p" and "Reference (" in str(b[1])])        # never in the results by case


# ---- review fixes: private markers anywhere, cleaning of the Markdown, the wording of a case without its MYSTRAN comparison, no date from the file time
_CHK = {"name": "q", "value": 1.0, "ref": 1.0, "error": 0.0, "tol": 0.1, "unit": "", "ok": True}


@pytest.mark.parametrize("where, mutate", [
    ("field name", lambda r, y, m: r["V1"].__setitem__("time_abaqus_s", 12.0)),
    ("nested key", lambda r, y, m: r["V1"]["checks"][0].__setitem__("ABQ", 1)),
    ("check name, abbreviation", lambda r, y, m: r["V1"]["checks"][0].__setitem__("name", "ABQ S11 (CAE 2023)")),
    ("check name, vendor", lambda r, y, m: r["V1"]["checks"][0].__setitem__("name", "Simulia Standard S11")),
    ("check name, product", lambda r, y, m: r["V1"]["checks"][0].__setitem__("name", "abaqus s11")),
    ("title field", lambda r, y, m: r["V1"].__setitem__("title", "Abaqus cross-check")),
    ("error text", lambda r, y, m: r.__setitem__("V2", {"error": "ABAQUS job failed"})),
    ("case id in a value", lambda r, y, m: r["V1"].__setitem__("title", "see case X3")),
    ("metadata", lambda r, y, m: m.__setitem__("profile", "compared with Abaqus")),
    ("metadata, nested", lambda r, y, m: m.__setitem__("skipped", {"V3": "licence of Abaqus missing"})),
    ("mystran record", lambda r, y, m: y.__setitem__("V1", [{"case": "F", "note": "Simulia"}])),
])
def test_private_markers_are_found_in_every_key_and_value(where, mutate):
    rep, mys, meta = _fake_report(), _mystran_ok(), _meta()
    mutate(rep, mys, meta)
    with pytest.raises(dossier.PrivateContentError):
        dossier.build_dossier(rep, mys, meta)


def test_ordinary_text_is_not_taken_for_a_private_marker():
    rep = _fake_report()
    rep["V1"]["checks"][0]["name"] = "sensor x1 and the cabin (ABQUE)"                 # lowercase x1 and a longer word are text
    dossier.build_dossier(rep, _mystran_ok(), _meta())


def test_markdown_and_word_lose_the_names_of_the_machine(tmp_path, monkeypatch):
    from fea_core.report import space_sar
    monkeypatch.setattr(space_sar, "_identities", lambda: ["LABPC-77", "kowalski"])
    meta = _meta()
    meta["platform"] = "TestOS 1 on LABPC-77 (user kowalski)"
    rp, mp, ep = tmp_path / "report.json", tmp_path / "mystran.json", tmp_path / "run_meta.json"
    rp.write_text(json.dumps(_fake_report()), encoding="utf-8")
    mp.write_text(json.dumps(_mystran_ok()), encoding="utf-8")
    ep.write_text(json.dumps(meta), encoding="utf-8")
    assert dossier.main(["--report", str(rp), "--mystran", str(mp), "--meta", str(ep), "--out", str(tmp_path / "d.docx"), "--markdown", str(tmp_path / "d.md")]) == 0
    md = (tmp_path / "d.md").read_text(encoding="utf-8")
    from tests.test_space_report import _all_xml_text
    for text in (md, _text(tmp_path / "d.docx"), _all_xml_text(tmp_path / "d.docx")):
        assert "LABPC-77" not in text and "kowalski" not in text
    assert "TestOS 1" in md


def test_a_case_without_its_mystran_comparison_is_not_outside_tolerance(tmp_path):
    args = _write_inputs(tmp_path)
    assert dossier.main([*args, "--no-mystran", "--out", str(tmp_path / "n.docx"), "--markdown", str(tmp_path / "n.md")]) == 0
    text = _text(tmp_path / "n.docx")
    line = next(ln for ln in text.splitlines() if ln.startswith("V1 - "))                 # V1: both checks pass, the page promises its MYSTRAN comparison, `--no-mystran` removed it
    assert "OUTSIDE TOLERANCE" not in line and "within tolerance; the comparison with MYSTRAN is incomplete" in line
    assert "within tolerance but without a complete comparison with MYSTRAN" in text
    md = (tmp_path / "n.md").read_text(encoding="utf-8")
    assert "V1 - " in md and "OUTSIDE TOLERANCE" not in next(ln for ln in md.splitlines() if ln.startswith("V1 - "))
    bl = dossier.build_dossier(_fake_report(ok=False), _mystran_ok(), _meta())                 # a check that fails is outside tolerance
    lines = {str(b[1]).split(" - ")[0]: str(b[1]) for b in bl if b[0] == "p" and " - " in str(b[1])[:8]}
    assert "OUTSIDE TOLERANCE" in lines["V2"] and "NOT PASSED" in lines["V3"] and "OUTSIDE TOLERANCE" not in lines["V3"]      # V3 stopped with an error
    bad = _mystran_ok()
    bad["V1"] = [{**bad["V1"][0], "du_rel": 0.5}]                                                  # a MYSTRAN comparison that disagrees: said, not "outside" for the checks
    bl = dossier.build_dossier(_fake_report(), bad, _meta())
    line = next(str(b[1]) for b in bl if b[0] == "p" and str(b[1]).startswith("V1 - "))
    assert "outside tolerance, see chapter 5" in line and "OUTSIDE TOLERANCE" not in line


def test_without_run_metadata_there_is_no_date_and_the_output_does_not_depend_on_the_file_time(tmp_path):
    import os
    args = _write_inputs(tmp_path)
    ep = tmp_path / "run_meta.json"
    ep.unlink()
    for n, t in (("a", 1_000_000_000), ("b", 1_700_000_000)):
        os.utime(tmp_path / "report.json", (t, t))
        assert dossier.main([*args, "--out", str(tmp_path / f"{n}.docx"), "--markdown", str(tmp_path / f"{n}.md")]) == 0
    assert (tmp_path / "a.docx").read_bytes() == (tmp_path / "b.docx").read_bytes() and (tmp_path / "a.md").read_bytes() == (tmp_path / "b.md").read_bytes()
    text = _text(tmp_path / "a.docx")
    assert "date unknown" in text and "2001-09" not in text and "2023-11" not in text

"""F14 S6, fixes after the independent review of the Structural Analysis Report (ECSS): chapter 7 never PASS by default, the footer and the file properties are clean, the user's
own text is not altered, several environment levels, the configuration status, the study hash, the scan of the open items and `scrub_paths`.

Chapter 7 is tested here against stand-ins of the S4 modules (the real ones are in `test_space_report.test_real_s4_module_gives_a_status_to_the_stored_results`, which runs when the
build has stage S4), so the rules hold on a build without S4 as well."""
from __future__ import annotations

import contextlib
import copy
import getpass
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("docx")

from fea_core.report import space_sar as SAR  # noqa: E402
from tests.test_space_report import PATH_LIKE, _all_xml_text, _fake_s4, _sar, _sess, _stub_chapters  # noqa: E402

NOTE = "stored result, not recomputed - tamper-evident only (staleness), not a signature"
SECTIONS = ("interface", "notch", "bolts", "pcb", "rails")


class _Shown:
    """The part of `s4_verify.Shown` that the SAR reads."""

    def __init__(self, verdict: str, fresh: bool, note: str = "") -> None:
        self.verdict, self.fresh, self.note, self.text, self.label = verdict, fresh, note, "", verdict


def _fake_store(monkeypatch, data, shown=None):
    store = types.ModuleType("fea_core.space.s4_store")
    store.SECTIONS, store.get = SECTIONS, lambda study: data
    verify = types.ModuleType("fea_core.space.s4_verify")
    verify.NOT_RECOMPUTED_NOTE = NOTE
    verify.judging = lambda study: contextlib.nullcontext()

    def default(study, section, rec):
        ok = rec.get("verdict") == "PASS" and rec.get("basis") == "now"
        return _Shown("PASS" if ok else "UNCONFIRMED_REQUIREMENTS", ok, NOTE if section in ("notch", "interface") else "")
    verify.shown = shown or default
    monkeypatch.setitem(sys.modules, "fea_core.space.s4_store", store)
    monkeypatch.setitem(sys.modules, "fea_core.space.s4_verify", verify)


GOOD = {"verdict": "PASS", "basis": "now"}


# ---------------------------------------------------------------------------------------------------------------- C1: chapter 7
@pytest.mark.parametrize("data", [
    {}, {"junk": {"B1": GOOD}}, {"bolts": []}, {"bolts": "PASS"}, {"interface": "PASS"}, {"bolts": {}}, {k: {} for k in SECTIONS}, {"bolts": {"B1": "PASS"}},
    {"bolts": {"B1": GOOD}, "junk": 1}, {"bolts": {"B1": GOOD, "B2": "x"}}, {"bolts": {"B1": GOOD}, "pcb": {}}], ids=lambda d: str(d)[:40])
def test_chapter_7_is_unverifiable_when_nothing_could_be_judged_or_part_is_unreadable(monkeypatch, data):
    _fake_store(monkeypatch, data)
    status, items, _notes = SAR._s4_status(SimpleNamespace())
    assert status == "UNVERIFIABLE" and items


def test_chapter_7_pass_needs_a_judged_record_and_every_judged_record_fresh(monkeypatch):
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD}, "pcb": {"C1": GOOD}, "rails": GOOD})
    assert SAR._s4_status(SimpleNamespace())[0] == "PASS"
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD, "B2": {"verdict": "PASS"}}})
    status, items, _ = SAR._s4_status(SimpleNamespace())
    assert status == "UNCONFIRMED_REQUIREMENTS" and any(i.startswith("B2") for i in items)
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD}}, shown=lambda study, section, rec: _Shown("OUTDATED", False))
    assert SAR._s4_status(SimpleNamespace())[0] == "OUTDATED"
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD}}, shown=lambda study, section, rec: _Shown("FAIL", True))
    assert SAR._s4_status(SimpleNamespace())[0] == "FAIL"


def test_chapter_7_a_record_that_cannot_be_judged_is_unverifiable_not_a_crash(monkeypatch):
    def boom(study, section, rec):
        raise RuntimeError("damaged record")
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD}}, shown=boom)
    status, items, _ = SAR._s4_status(SimpleNamespace())
    assert status == "UNVERIFIABLE" and "damaged record" in items[0]


def test_chapter_7_prints_the_note_of_s4_for_records_that_are_not_recomputed(tmp_path, monkeypatch):
    _fake_s4(monkeypatch, has=True, warnings=())
    _fake_store(monkeypatch, {"notch": {"N1": GOOD}, "bolts": {"B1": GOOD}})
    r, text, *_ = _sar(tmp_path)
    c = {c["key"]: c for c in r["chapters"]}["interfaces"]
    assert c["status"] == "PASS" and c["notes"] == [NOTE] and "tamper-evident only" in text and text.count("not a signature") >= 2        # in the chapter and in the summary
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD}})                                                # nothing that is not recomputed: no such note
    r, text, *_ = _sar(tmp_path, "b.docx")
    assert {c["key"]: c for c in r["chapters"]}["interfaces"]["notes"] == [] and "tamper-evident" not in text


def test_chapter_7_pass_is_withdrawn_by_a_warning_of_the_s4_writer(tmp_path, monkeypatch):
    _fake_store(monkeypatch, {"bolts": {"B1": GOOD}})
    _fake_s4(monkeypatch, has=True, warnings=("Space chapter, bolts: boom",))
    r, *_ = _sar(tmp_path)
    c = {c["key"]: c for c in r["chapters"]}["interfaces"]
    assert c["status"] == "UNVERIFIABLE" and any("could not be written" in i for i in c["open_items"])


# ---------------------------------------------------------------------------------------------------------------- I1: the footer and every part of the file
def test_footer_and_properties_are_cleaned_and_the_gate_reads_every_xml_part(tmp_path):
    bs = chr(92)
    title = "Report D:" + bs + "Projects" + bs + "Kowalski" + bs + "sat"
    r, text, *_ = _sar(tmp_path, title=title, subject="see " + bs + bs + "nas" + bs + "Kowalski" + bs + "a b.docx", author="A. Author")
    xml = _all_xml_text(r["path"])
    assert "footer" in xml.lower() and "Kowalski" not in xml and "Projects" not in xml and not PATH_LIKE.search(xml)
    from docx import Document
    foot = Document(str(r["path"])).sections[0].footer.paragraphs[0].text
    assert "FEA Studio" in foot and "<path>" in foot and "Kowalski" not in foot                      # the footer exists, carries the title and no path


# ---------------------------------------------------------------------------------------------------------------- I2: what the user typed
def test_user_entered_fields_keep_a_name_that_is_also_the_login(tmp_path, monkeypatch):
    me = getpass.getuser()
    monkeypatch.setattr(SAR, "_identities", lambda: [me, "LABPC-01"])
    sess, st, res = _sess()
    st.space.requirement_set.confirmed_by = f"{me} Novak"                                              # the provenance of the requirements is typed text too
    r, text, *_ = _sar(tmp_path, session=(sess, st, res), author=f"{me} Novak", company=f"{me} Space s.r.o.")
    xml = _all_xml_text(r["path"])
    assert text.count(f"{me} Novak") >= 2 and f"{me} Space s.r.o." in text and "<redacted>" not in text and f"{me} Novak" in xml


def test_a_login_that_is_an_ordinary_word_does_not_damage_the_prose(tmp_path, monkeypatch):
    monkeypatch.setattr(SAR, "_identities", lambda: ["model", "report", "check"])
    r, text, *_ = _sar(tmp_path, author="model Novak")
    assert "<redacted>" not in text and "2 Model and mass properties" in text and "model Novak" in text and "Structural Analysis Report" in text


def test_program_generated_text_still_loses_identity_and_path(monkeypatch):
    monkeypatch.setattr(SAR, "_identities", lambda: ["jsmith", "LABPC-01"])
    s = SAR.clean_text(r"saved to D:\work\jsmith\x.fea by jsmith on LABPC-01")
    assert "jsmith" not in s and "LABPC-01" not in s and "x.fea" in s                                    # the file name stays, the folder and the names go
    assert SAR.clean_paths("jsmith wrote D:" + chr(92) + "w" + chr(92) + "jsmith" + chr(92) + "x.fea") == "jsmith wrote <path>/x.fea"      # a typed name stays, the path goes


def test_scrub_paths_keeps_no_part_of_a_folder_only_path():
    from fea_core.space._checks_common import scrub_paths
    bs = chr(92)
    assert scrub_paths("D:" + bs + "data" + bs + "Kowalski" + bs) == "<path>" and scrub_paths("C:/Users/Kowalski/") == "<path>"
    assert scrub_paths("see C:/Users/Kowalski/x.step, ok") == "see <path>/x.step, ok" and scrub_paths("a/b") == "a/b"


# ---------------------------------------------------------------------------------------------------------------- I4: several environment levels
def test_a_level_without_a_current_3axis_result_is_a_partial_scope(tmp_path):
    import numpy as np

    from fea_core.space import environments as EV
    from fea_core.space import loads as LD
    from verification import space_load_cases as SL
    sess, st, res = _sess()
    q = SL.three_axis_level()
    a = copy.deepcopy(q)
    a.name = "acceptance"
    es = EV.EnvironmentSet(levels=[q, a], launch_axis="z")
    EV.store_environments(st.space, es)
    mesh = res.mesh
    n = len(mesh.nodes)
    f = LD.AxisField(np.full(n, 10.0, np.float32), np.zeros(n, np.float32), None)
    per = {"sine": {0: f, 1: f, 2: f}}
    r3 = LD.ThreeAxisResult(q.name, ["sine"], [0, 1, 2], per, {"sine": LD.envelope_of(per["sine"], "sine")}, {}, np.array([100.0]), np.zeros((1, 3)), None, True, True,
                            env_digest=es.content_digest(), level_kinds=["sine"], damping={"kind": "modal", "zeta": 0.02, "source": "test", "from_environment": True})
    r3.run_digest = LD.compute_run_digest(st, mesh)
    LD.remember(sess, r3)
    r, text, *_ = _sar(tmp_path, session=(sess, st, res))
    c = {c["key"]: c for c in r["chapters"]}["loads"]
    assert any("only the level 'qualification' has a 3-axis result" in i and "acceptance" in i for i in c["open_items"])
    assert c["status"] in ("PARTIAL_SCOPE", "OUTDATED", "NOT_RUN", "FAIL") and "level 'qualification' only" in text


# ---------------------------------------------------------------------------------------------------------------- configuration, hash, scan
def test_configuration_chapter_has_a_status(tmp_path):
    r, text, *_ = _sar(tmp_path)
    c = {c["key"]: c for c in r["chapters"]}["configuration"]
    assert c["status"] == "UNVERIFIABLE" and "geometry" in c["basis"] and "not checked against a confirmation" in text      # the synthetic study has no geometry file
    assert r["verdict"] == "NOT_SIGNED_OFF" and any(i.startswith("1 Configuration") for i in r["open_items"])


def test_study_hash_ignores_the_time_stamp_of_stored_s4_results_and_the_folder_style(tmp_path):
    sess, st, res = _sess()
    st.space.extra["s4"] = {"bolts": {"B1": {"verdict": "PASS", "stamp": "2026-01-01T10:00:00", "x": 1}}, "interface": {"stamp": "2026-01-01T10:00:00"}}
    a, *_ = _sar(tmp_path, "a.docx", session=(sess, st, res))
    st.space.extra["s4"]["bolts"]["B1"]["stamp"] = "2030-12-31T23:59:59"
    st.space.extra["s4"]["interface"]["stamp"] = "2030-12-31T23:59:59"
    b, *_ = _sar(tmp_path, "b.docx", session=(sess, st, res))
    assert a["traceability"]["study_sha256"] == b["traceability"]["study_sha256"]
    st.space.extra["s4"]["bolts"]["B1"]["x"] = 2                                                          # a real change is another hash
    c, *_ = _sar(tmp_path, "c.docx", session=(sess, st, res))
    assert c["traceability"]["study_sha256"] != a["traceability"]["study_sha256"]
    assert SAR._file_name("C:" + chr(92) + "a" + chr(92) + "m.step") == SAR._file_name("D:/b/c/m.step") == SAR._file_name("/x/y/m.step") == "m.step"
    assert SAR._file_name("") == "" and SAR._file_name(None) == ""


def test_open_items_do_not_come_from_the_prose_of_a_chapter(tmp_path, monkeypatch):
    def chapter(D, sess, study, res, ch, sec, o):
        D.p("Introduction: a value that is UNCONFIRMED or OUTDATED is labelled so; it is to be confirmed by the user.")           # prose: not a finding
        D.p("Status line: the limit is UNCONFIRMED here", bold=True)                                                             # a status line: is a finding
        ch.status, ch.basis = "PASS", "stub"
        return []
    _stub_chapters(monkeypatch, {"configuration": "INFO", "interfaces": "PASS", "correlation": "NOT_INCLUDED"})
    monkeypatch.setattr(SAR, "CHAPTERS", [(k, n, chapter if k == "loads" else f) for k, n, f in SAR.CHAPTERS])
    r, *_ = _sar(tmp_path)
    assert any("Status line: the limit is UNCONFIRMED here" in i for i in r["open_items"])
    assert not any("Introduction" in i for i in r["open_items"])


# ---------------------------------------------------------------------------------------------------------------- the benchmark (fast parts)
def test_benchmark_help_text_and_the_illustrative_set_name_no_standard(capsys):
    pytest.importorskip("OCP")
    import re

    from fea_core.study import Study
    from verification import cubesat_benchmark as CB
    doc = CB.__doc__ or ""
    assert "--engine" not in doc and "when the build has it" not in doc                      # no option and no step that the benchmark does not have
    with pytest.raises(SystemExit):
        CB.main(["--help"])
    helptext = capsys.readouterr().out
    assert all(flag in helptext for flag in re.findall(r"--[a-z]+(?:-[a-z]+)*", doc.split("The model")[0]))
    rs = CB.requirement_set(Study(name="x", geometry="x.step"))
    assert "ECSS" not in rs.standard and "ILLUSTRATIVE" in rs.standard and rs.name.startswith("ILLUSTRATIVE")

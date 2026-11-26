"""F14 S1: report chapter "Space structures: margins of safety" (fea_core.report.space) on a real python-docx document."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.report import space as RS  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
from fea_core.space import margins as MG  # noqa: E402
from fea_core.space.model import SpaceSettings  # noqa: E402
from verification import space_cases as SC  # noqa: E402


def _sess(confirmed: bool = True, spike: float = 3.0):
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("protoflight", confirmed=confirmed))
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA, spike=spike)])
    return SimpleNamespace(study=st, results=res), st, res


def _text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def test_has_space():
    sess, st, res = _sess()
    assert RS.has_space(st, res)
    st.space = None
    assert not RS.has_space(st, res)
    assert not RS.has_space(SimpleNamespace(space=SpaceSettings()), None)


def test_chapter_for_confirmed_inputs():
    sess, st, res = _sess()
    D = _Doc()
    warns = RS.space_section(D, sess)
    text = _text(D)
    assert warns == []
    assert "Verdict: PASS" in text and "NOT JUSTIFIED" in text                  # PASS, but the singularity still needs its note
    for needle in ("Requirement set - factors entered by the user", "Singularity justification table", "Lowest margins of safety",
                   "test clause", "test engineer", "ECSS-E-ST-32-10C", "protoflight", "TestAlloy", "test data sheet"):
        assert needle in text, needle
    s = MG.compute_margins(st, res)
    assert f"{s.worst.mos:.3g}".replace("-", "−") in text
    MG.add_justification(st.space, "Block", s.singularities[0].location, "accepted for the test", author="tester", date="2026-01-02")
    D2 = _Doc()
    RS.space_section(D2, sess)
    t2 = _text(D2)
    assert "accepted for the test" in t2 and "NOT JUSTIFIED" not in t2


def test_unconfirmed_chapter_says_so():
    sess, st, res = _sess(confirmed=False)
    D = _Doc()
    RS.space_section(D, sess)
    text = _text(D)
    assert "UNCONFIRMED_REQUIREMENTS" in text and "not an ECSS verification result" in text and "Verdict: PASS" not in text
    assert "NO" in text                                                          # "Confirmed by the user: NO"


def test_chapter_without_requirement_set_is_indicative():
    sess, st, res = _sess()
    st.space.requirement_set = None
    D = _Doc()
    RS.space_section(D, sess)
    assert "No requirement set is defined" in _text(D) and "UNCONFIRMED_REQUIREMENTS" in _text(D)


def test_chapter_has_no_local_paths_or_usernames():
    import os
    sess, st, res = _sess()
    D = _Doc()
    RS.space_section(D, sess)
    text = _text(D)
    for forbidden in (os.path.expanduser("~"), "C:\\", "/home/", "Abaqus"):
        assert forbidden not in text


def test_pass_with_open_singularities_is_not_shown_as_a_sign_off():
    sess, st, res = _sess()
    D = _Doc()
    RS.space_section(D, sess)
    text = _text(D)
    assert "PASS – not ready for sign-off (open singularities: 1)" in text and "not a sign-off" in text
    s = MG.compute_margins(st, res)
    MG.add_justification(st.space, "Block", s.singularities[0].location, "accepted for the test", author="tester", date="2026-01-02")
    D2 = _Doc()
    RS.space_section(D2, sess)
    assert "not ready for sign-off" not in _text(D2) and "Verdict: PASS" in _text(D2)


# ====================================================================================================================== F14 S6: Structural Analysis Report (ECSS)
import getpass  # noqa: E402
import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

from fea_core.report import space_sar as SAR  # noqa: E402
from fea_core.report.word import ReportOptions  # noqa: E402

STAMP = "2026-01-01 00:00"
FORBIDDEN = re.compile(r"(?i:abaqus)|(?<![A-Za-z0-9_])X[1-5](?![A-Za-z0-9_])")        # the name in any case, the case ids X1-X5 only as written (a sensor axis "x1" is text)
PATH_LIKE = re.compile(r"[A-Za-z]:[\\/]|\\\\[A-Za-z]|/ho" r"me/|/Us" r"ers/|App" r"Data")


def _docx_text(path) -> str:
    from docx import Document
    d = Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def _all_xml_text(path) -> str:
    """The text of EVERY XML part of the .docx (body, footer, header, properties, styles): the privacy gate must see what Word shows in the footer and in the file properties too."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        text = "\n".join(z.read(n).decode("utf-8", "replace") for n in sorted(z.namelist()) if n.endswith((".xml", ".rels")))
    return re.sub(r"https?://[^\s\"'<>]+", "", text)                                       # the namespace addresses of the format are not content


def _no_timestamp(text: str) -> str:
    return chr(10).join(line for line in text.splitlines() if not line.startswith("Generated (UTC)") and not re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", line))


def _sar(tmp_path, name="sar.docx", generated=STAMP, **kw):
    sess, st, res = kw.pop("session", None) or _sess()
    r = SAR.build_sar(sess, tmp_path / name, ReportOptions(ecss_sar=True, **kw), generated=generated)
    return r, _docx_text(r["path"]), sess, st


def test_sar_chapters_in_the_ecss_order_with_traceability(tmp_path):
    r, text, _sess_, st = _sar(tmp_path)
    heads = ["1 Configuration and traceability", "2 Model and mass properties", "3 Confirmed requirement set", "4 Model quality checks", "5 Load cases and envelopes",
             "6 Margins of safety and singularity justifications", "7 Interfaces, notching, bolts and PCB", "8 Test correlation", "9 Assumptions, limitations and open items"]
    pos = [text.index(h) for h in heads]
    assert pos == sorted(pos)
    assert "Sign-off summary" in text and text.index("Sign-off summary") < pos[0]
    tr = r["traceability"]
    assert re.fullmatch(r"[0-9a-f]{64}", tr["study_sha256"]) and re.fullmatch(r"[0-9a-f]{64}", tr["setup_digest"])
    assert tr["fea_studio"] and "Study content SHA-256" in text and "Setup digest SHA-256" in text and "Requirement set SHA-256" in text
    assert "ILLUSTR" not in text and "TestAlloy" in text and "test clause" in text                    # provenance of the requirement set is printed
    assert [c["key"] for c in r["chapters"]][:2] == ["configuration", "model"]


def test_sar_is_not_a_sign_off_without_checks_loads_and_justified_singularities(tmp_path):
    r, text, *_ = _sar(tmp_path)
    by = {c["key"]: c["status"] for c in r["chapters"]}
    assert r["verdict"] == "NOT_SIGNED_OFF" and "Overall verdict: NOT_SIGNED_OFF" in text and "Overall verdict: PASS" not in text
    assert by["checks"] == "NOT_RUN" and by["loads"] == "NOT_RUN" and by["margins"] == "UNCONFIRMED_REQUIREMENTS" and by["requirements"] == "PASS"
    assert by["interfaces"] in ("NOT_IN_BUILD", "NOT_RUN") and by["correlation"] == "NOT_INCLUDED"
    assert any("singularity" in i for i in r["open_items"])


def test_sar_is_deterministic_apart_from_the_timestamp(tmp_path):
    a, ta, *_ = _sar(tmp_path, "a.docx", STAMP)
    b, tb, *_ = _sar(tmp_path, "b.docx", STAMP)
    assert (tmp_path / "a.docx").read_bytes() == (tmp_path / "b.docx").read_bytes()                  # same stamp: the same bytes (members are frozen)
    c, tc, *_ = _sar(tmp_path, "c.docx", "2030-05-05 12:34")
    assert _no_timestamp(ta) == _no_timestamp(tc) and "2030-05-05 12:34" in tc and a["open_items"] == c["open_items"]


def test_sar_has_no_private_string_no_path_and_no_user_name(tmp_path):
    sess, st, res = _sess()
    st.name = r"C:\Users\someone\Documents\sat\model one"                                            # a study named after a local path must not leak it
    r, text, *_ = _sar(tmp_path, session=(sess, st, res), author="A. Author")
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text)
    assert not PATH_LIKE.search(text), PATH_LIKE.search(text)
    assert "someone" not in text and "A. Author" in text
    xml = _all_xml_text(r["path"])                                                                   # the footer and the file properties are scanned as well
    assert "someone" not in xml and not PATH_LIKE.search(xml) and not FORBIDDEN.search(xml)
    for who in {getpass.getuser(), __import__("platform").node()}:
        if len(who) >= 3 and who.lower() not in SAR._COMMON_NAMES:
            assert who.lower() not in text.lower()


def test_clean_text_removes_paths_user_and_host_names(monkeypatch):
    monkeypatch.setattr(SAR, "_identities", lambda: ["jsmith", "LABPC-01"])
    s = SAR.clean_text(r"saved to D:\work\jsmith\x.fea by jsmith on LABPC-01 (see /home/jsmith/p/q)")
    assert "jsmith" not in s and "LABPC-01" not in s and "D:" not in s and "/home" not in s
    assert "the user" in SAR.clean_text("the user")                                                  # an ordinary word is never redacted


# ---- the verdict: PASS only when every included chapter is current and PASS
def _ch(key, status, included=True):
    return SAR.Chapter(key, key, status if included else "NOT_INCLUDED")


ALL = ("model", "requirements", "checks", "loads", "margins")


def test_overall_verdict_rules():
    ok = [_ch(k, "PASS") for k in ALL] + [_ch("interfaces", "PASS"), _ch("correlation", "NOT_INCLUDED"), _ch("configuration", "INFO")]
    assert SAR.overall_verdict(ok)[0] == "PASS"
    no_s4 = [_ch("interfaces", "NOT_IN_BUILD") if c.key == "interfaces" else c for c in ok]                 # a chapter that this build cannot write is not assessed: no sign-off
    v, blockers = SAR.overall_verdict(no_s4)
    assert v == "NOT_SIGNED_OFF" and [b.key for b in blockers] == ["interfaces"]
    assert SAR.overall_verdict([_ch(k, "INFO" if k == "model" else "PASS") for k in ALL])[0] == "PASS"          # a model chapter without a budget is INFO
    for bad in ("OUTDATED", "UNCONFIRMED_REQUIREMENTS", "PARTIAL_SCOPE", "UNVERIFIABLE", "WARN", "NOT_RUN"):
        for k in ALL + ("interfaces", "correlation"):
            chs = [_ch(x, "PASS") for x in ALL] + [_ch("correlation", "PASS"), _ch("interfaces", "PASS")]
            chs = [_ch(c.key, bad) if c.key == k else c for c in chs]
            v, blockers = SAR.overall_verdict(chs)
            assert v == "NOT_SIGNED_OFF" and [b.key for b in blockers] == [k], (bad, k)
    chs = [_ch(k, "PASS") for k in ALL]
    chs[2] = _ch("checks", "FAIL")
    assert SAR.overall_verdict(chs)[0] == "FAIL"
    assert SAR.overall_verdict([_ch(k, "PASS") for k in ALL if k != "loads"])[0] == "NOT_SIGNED_OFF"          # a mandatory chapter that is missing
    assert SAR.overall_verdict([])[0] == "NOT_SIGNED_OFF"
    assert SAR.worst(["PASS", "UNCONFIRMED_REQUIREMENTS", "OUTDATED"]) == "OUTDATED" and SAR.worst([]) == "PASS"


def _stub_chapters(monkeypatch, statuses: dict[str, str]):
    """Every chapter replaced by one that writes a line and takes the given status (the document machinery is real)."""
    def make(key, name):
        def fn(D, sess, study, res, ch, sec, o):
            D.p(f"stub of {name}")
            ch.status, ch.basis = statuses.get(key, "PASS"), f"stub {key}"
            if ch.status not in ("PASS", "INFO", "NOT_INCLUDED", "NOT_IN_BUILD"):
                ch.items.append(f"{key} is {ch.status}")
            return []
        return fn
    monkeypatch.setattr(SAR, "CHAPTERS", [(k, n, make(k, n)) for k, n, _f in SAR.CHAPTERS])


@pytest.mark.parametrize("stale", ["OUTDATED", "UNCONFIRMED_REQUIREMENTS", "PARTIAL_SCOPE", "UNVERIFIABLE", "WARN", "NOT_RUN"])
@pytest.mark.parametrize("key", ["checks", "loads", "margins", "interfaces", "correlation"])
def test_sar_verdict_is_never_pass_with_a_stale_or_unconfirmed_chapter(tmp_path, monkeypatch, key, stale):
    _stub_chapters(monkeypatch, {key: stale, "configuration": "INFO"})
    r, text, *_ = _sar(tmp_path)
    assert r["verdict"] == "NOT_SIGNED_OFF" and "Overall verdict: PASS" not in text and f"{key} is {stale}" in text
    assert any(stale in i for i in r["open_items"])                                                  # collected into the open items of chapter 9


def test_sar_pass_only_when_everything_is_pass(tmp_path, monkeypatch):
    _stub_chapters(monkeypatch, {"configuration": "INFO", "interfaces": "PASS", "correlation": "NOT_INCLUDED"})
    r, text, *_ = _sar(tmp_path)
    assert r["verdict"] == "PASS" and "Overall verdict: PASS" in text and "Scope: not covered by this report" in text


def test_sar_without_stage_s4_is_not_a_sign_off(tmp_path, monkeypatch):
    _stub_chapters(monkeypatch, {"configuration": "INFO", "interfaces": "NOT_IN_BUILD", "correlation": "NOT_INCLUDED"})
    r, text, *_ = _sar(tmp_path)
    assert r["verdict"] == "NOT_SIGNED_OFF" and "Overall verdict: PASS" not in text and r["blocking"] == ["Interfaces, notching, bolts and PCB"]
    monkeypatch.undo()                                                                                  # the real chapter on a build without the S4 modules
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == SAR.S4_MODULE else real(name, *a, **k))
    r, text, *_ = _sar(tmp_path, "real.docx")
    assert "interfaces/bolts not assessed in this build" in text and {c["key"]: c["basis"] for c in r["chapters"]}["interfaces"] == "interfaces/bolts not assessed in this build"


def test_the_margins_chapter_is_outdated_when_the_kept_3axis_result_is_stale(tmp_path, monkeypatch):
    from fea_core.space import loads as LD
    real = LD.sign_off_margins

    def stale(study, results, owner, mesh=None, max_places=None):
        s = real(study, results, owner, mesh, max_places)
        s.reasons.append(f"{LD.STALE_3AXIS} [the study changed]")
        s.verdict = MG.UNCONFIRMED
        return s
    monkeypatch.setattr(LD, "sign_off_margins", stale)
    r, text, *_ = _sar(tmp_path)
    assert {c["key"]: c["status"] for c in r["chapters"]}["margins"] == "OUTDATED"
    assert "out of date" in text


def test_the_confirmation_of_the_requirement_set_is_what_makes_chapter_3_pass(tmp_path):
    r, *_ = _sar(tmp_path, session=_sess(confirmed=False))
    by = {c["key"]: c["status"] for c in r["chapters"]}
    assert by["requirements"] == "UNCONFIRMED_REQUIREMENTS" and r["verdict"] == "NOT_SIGNED_OFF"


# ---- the hook of stage S4: the module may or may not be part of the build
def _fake_s4(monkeypatch, has: bool, warnings=("s4 warning",)):
    mod = types.ModuleType(SAR.S4_MODULE)
    mod.__spec__ = importlib.machinery.ModuleSpec(SAR.S4_MODULE, None)
    mod.has_s4 = lambda study: has
    mod.s4_sections = lambda D, study, sec: (D.p("S4 section written by the module"), list(warnings))[1]
    monkeypatch.setitem(sys.modules, SAR.S4_MODULE, mod)


def test_s4_hook_without_the_module_says_so(tmp_path, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == SAR.S4_MODULE else real(name, *a, **k))
    r, text, *_ = _sar(tmp_path)
    assert {c["key"]: c["status"] for c in r["chapters"]}["interfaces"] == "NOT_IN_BUILD" and "not part of this build" in text


def test_s4_hook_with_the_module(tmp_path, monkeypatch):
    _fake_s4(monkeypatch, has=False)
    r, text, *_ = _sar(tmp_path)
    assert {c["key"]: c["status"] for c in r["chapters"]}["interfaces"] == "NOT_RUN" and "not assessed" in text
    _fake_s4(monkeypatch, has=True)
    monkeypatch.setattr(SAR, "_s4_status", lambda study: ("OUTDATED", ["bolts/B1: outdated - recompute"], []))
    r, text, *_ = _sar(tmp_path)
    by = {c["key"]: c["status"] for c in r["chapters"]}
    assert by["interfaces"] == "OUTDATED" and "S4 section written by the module" in text and r["verdict"] == "NOT_SIGNED_OFF" and "s4 warning" in r["warnings"]
    _fake_s4(monkeypatch, has=True, warnings=())
    monkeypatch.setattr(SAR, "_s4_status", lambda study: ("PASS", [], []))
    r, *_ = _sar(tmp_path)
    assert {c["key"]: c["status"] for c in r["chapters"]}["interfaces"] == "PASS"
    _fake_s4(monkeypatch, has=True, warnings=("Space chapter, bolts: boom",))                          # a section that could not be written was not shown: never a PASS
    r, *_ = _sar(tmp_path)
    c = {c["key"]: c for c in r["chapters"]}["interfaces"]
    assert c["status"] == "UNVERIFIABLE" and any("could not be written" in i for i in c["open_items"])


def test_a_chapter_that_fails_to_write_blocks_the_sign_off_and_the_others_are_written(tmp_path, monkeypatch):
    _stub_chapters(monkeypatch, {"configuration": "INFO", "interfaces": "NOT_IN_BUILD", "correlation": "NOT_INCLUDED"})

    def boom(D, sess, study, res, ch, sec, o):
        raise RuntimeError(r"failed at C:\secret\folder\x.dat")
    monkeypatch.setattr(SAR, "CHAPTERS", [(k, n, boom if k == "loads" else f) for k, n, f in SAR.CHAPTERS])
    r, text, *_ = _sar(tmp_path)
    assert r["verdict"] == "NOT_SIGNED_OFF" and "could not be written" in text and "secret" not in text.replace("<path>/x.dat", "") and "stub of Model" in text


# ---- regenerated from a saved project
def test_sar_regenerated_from_a_saved_project_is_deterministic_and_not_a_pass(tmp_path):
    from fea_core.project import load_project, save_project
    from fea_core.report import session_from
    sess, st, res = _sess()
    save_project(tmp_path / "p.fea", st, "", res)
    texts, digests = [], []
    for k in (1, 2):
        study, step, results = load_project(tmp_path / "p.fea", tmp_path / f"wd{k}")
        s2 = session_from(study, step, results, workdir=tmp_path / f"wd{k}")
        r = SAR.build_sar(s2, tmp_path / f"r{k}.docx", ReportOptions(ecss_sar=True), generated=STAMP)
        texts.append(_docx_text(r["path"]))
        digests.append((r["traceability"]["study_sha256"], r["traceability"]["setup_digest"], r["verdict"]))
    assert (tmp_path / "r1.docx").read_bytes() == (tmp_path / "r2.docx").read_bytes() and texts[0] == texts[1]
    assert digests[0] == digests[1] and digests[0][2] == "NOT_SIGNED_OFF"
    assert re.fullmatch(r"[0-9a-f]{64}", digests[0][0])
    for t in texts:
        assert not FORBIDDEN.search(t) and not PATH_LIKE.search(t) and "wd1" not in t and "wd2" not in t and str(tmp_path) not in t
    for k in (1, 2):
        xml = _all_xml_text(tmp_path / f"r{k}.docx")
        assert not FORBIDDEN.search(xml) and not PATH_LIKE.search(xml) and str(tmp_path) not in xml
    by = {c["key"]: c["status"] for c in r["chapters"]}
    assert by["checks"] == "NOT_RUN" and by["loads"] == "NOT_RUN"                       # not saved in the project: said plainly, never a stale PASS
    assert any("not run" in i or "were not run" in i or "NOT_RUN" in i for i in r["open_items"])


def test_editing_a_confirmed_limit_withdraws_the_pass_of_the_requirement_chapter(tmp_path):
    sess, st, res = _sess()
    r, *_ = _sar(tmp_path, "a.docx", session=(sess, st, res))
    assert {c["key"]: c["status"] for c in r["chapters"]}["requirements"] == "PASS"
    f = next(f for f in st.space.requirement_set.factors if f.value is not None)
    f.value = f.value * 1.5                                                             # an edit after the confirmation
    r, text, *_ = _sar(tmp_path, "b.docx", session=(sess, st, res))
    assert {c["key"]: c["status"] for c in r["chapters"]}["requirements"] == "UNCONFIRMED_REQUIREMENTS" and r["verdict"] == "NOT_SIGNED_OFF"


@pytest.mark.skipif(importlib.util.find_spec("fea_core.report.space_s4") is None, reason="stage S4 is not part of this build")
def test_real_s4_module_gives_a_status_to_the_stored_results(tmp_path):
    """With the real S4 module in the build: a stored result without a basis is never a PASS (the hook reads the status from `s4_verify`), and a store that holds nothing that
    could be judged (an unknown section, a section of the wrong shape, an empty bucket, strings instead of records) is UNVERIFIABLE - never a PASS by default."""
    from fea_core.space import s4_store
    sess, st, res = _sess()
    r, *_ = _sar(tmp_path, "none.docx", session=(sess, st, res))
    assert {c["key"]: c["status"] for c in r["chapters"]}["interfaces"] == "NOT_RUN"                # nothing stored: said, not skipped
    s4_store.record(st, "bolts", {"verdict": "PASS", "verdict_text": "PASS"}, key="B1")
    r, text, *_ = _sar(tmp_path, "some.docx", session=(sess, st, res))
    by = {c["key"]: c["status"] for c in r["chapters"]}
    assert by["interfaces"] != "PASS" and r["verdict"] == "NOT_SIGNED_OFF"
    forged = {"unknown section": {"junk": 1}, "section is a list": {"bolts": []}, "section is a string": {"interface": "PASS"}, "empty bucket": {"bolts": {}},
              "empty buckets": {k: {} for k in s4_store.SECTIONS}, "bucket of strings": {"bolts": {"B1": "PASS"}}, "bare PASS without a basis": {"interface": {"verdict": "PASS"}}}
    for tag, data in forged.items():
        sess, st, res = _sess()
        st.space.extra["s4"] = data
        r, text, *_ = _sar(tmp_path, "forged.docx", session=(sess, st, res))
        c = {c["key"]: c for c in r["chapters"]}["interfaces"]
        assert c["status"] != "PASS" and r["verdict"] == "NOT_SIGNED_OFF", tag
        if tag != "bare PASS without a basis":
            assert c["status"] == "UNVERIFIABLE", tag
    for section, rec in (("interface", {"verdict": "PASS", "verdict_text": "PASS"}), ("notch", {"N1": {"verdict": "PASS", "verdict_text": "PASS"}})):      # records of the present basis
        sess, st, res = _sess()
        st.space.extra["s4"] = {section: rec}
        for r_ in ([rec] if section == "interface" else rec.values()):
            r_["basis"] = s4_store.basis_of(st)
        r, text, *_ = _sar(tmp_path, "present.docx", session=(sess, st, res))
        c = {c["key"]: c for c in r["chapters"]}["interfaces"]
        assert c["status"] == "PASS" and "tamper-evident only" in " ".join(c["notes"]) and "not a signature" in text, section          # PASS, with S4's own caveat next to it
        assert r["verdict"] == "NOT_SIGNED_OFF"                                                                              # the other chapters still block


def test_study_hash_does_not_depend_on_the_folder_of_the_geometry(tmp_path):
    sess, st, res = _sess()
    st.geometry = "C:" + chr(92) + "first" + chr(92) + "place" + chr(92) + "model.step"
    r1, *_ = _sar(tmp_path, "a.docx", session=(sess, st, res))
    st.geometry = "D:" + chr(92) + "zzfolder" + chr(92) + "model.step"
    r2, t2, *_ = _sar(tmp_path, "b.docx", session=(sess, st, res))
    assert r1["traceability"]["study_sha256"] == r2["traceability"]["study_sha256"] and "zzfolder" not in t2
    st.geometry = "model_other.step"                                                         # another file name is another study
    r3, *_ = _sar(tmp_path, "c.docx", session=(sess, st, res))
    assert r3["traceability"]["study_sha256"] != r1["traceability"]["study_sha256"]

"""1.5.1 optimisation of the SAR report: the same document, computed once instead of twice (open items, mass properties, S4 verdicts, correlation status, identities)."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (the cases first)
from fea_core.report import space_sar as SAR  # noqa: E402
from fea_core.report.word import ReportOptions  # noqa: E402
from fea_core.space import bolts_ecss as BE  # noqa: E402
from fea_core.space import s4_store, s4_verify  # noqa: E402
from tests.test_space_report import STAMP, _sess  # noqa: E402
from tests.test_space_correlation import EXAMPLE, SENSORS, _frf_pair, _results, _study_with_limits, _test_data, fe  # noqa: E402, F401  (fe is a fixture)
from tests.test_space_s4_bolts import _bolt, _lim  # noqa: E402

MARKERS = SAR._TABLE_MARKERS


def _legacy_counts(tab) -> dict[str, int]:
    """The loop that the report used before: the text of the cell is read once per marker and per grid cell."""
    flagged: dict[str, int] = {}
    for row in tab.rows:
        for cell in row.cells:
            for m in MARKERS:
                if m in cell.text:
                    flagged[m] = flagged.get(m, 0) + 1
                    break
    return flagged


def _table(doc, rows: int, cols: int, texts: dict[tuple[int, int], str]):
    t = doc.add_table(rows=rows, cols=cols)
    for (r, c), s in texts.items():
        t.cell(r, c).text = s
    return t


def test_marker_counts_equal_the_cell_loop_with_merged_cells():
    from docx import Document
    d = Document()
    plain = _table(d, 3, 3, {(0, 0): "ok", (1, 1): "OUTDATED", (2, 2): "x UNCONFIRMED y", (2, 0): "NOT confirmed"})
    hmerge = _table(d, 3, 4, {(0, 0): "PARTIAL_SCOPE", (1, 1): "fine", (2, 3): "outdated"})
    hmerge.cell(0, 0).merge(hmerge.cell(0, 2))                                   # a flagged cell that spans three grid columns: counted three times
    hmerge.cell(1, 1).merge(hmerge.cell(1, 2))
    vmerge = _table(d, 4, 2, {(0, 0): "UNVERIFIABLE", (0, 1): "NOT JUSTIFIED", (3, 1): "OUTDATED"})
    vmerge.cell(0, 0).merge(vmerge.cell(2, 0))                                   # a flagged cell that spans three rows
    both = _table(d, 4, 4, {(1, 1): "UNCONFIRMED and OUTDATED"})
    both.cell(1, 1).merge(both.cell(2, 2))
    multi = _table(d, 2, 2, {})
    multi.cell(0, 0).paragraphs[0].add_run("first")                              # a marker split over two paragraphs of a cell is not a marker
    multi.cell(0, 0).add_paragraph().add_run("OUTD")
    multi.cell(0, 0).add_paragraph().add_run("ATED")
    for t in (plain, hmerge, vmerge, both, multi):
        assert SAR._marker_counts(t, MARKERS) == _legacy_counts(t)
    assert SAR._marker_counts(hmerge, MARKERS)["PARTIAL_SCOPE"] == 3 and SAR._marker_counts(vmerge, MARKERS)["UNVERIFIABLE"] == 3
    assert SAR._marker_counts(multi, MARKERS) == {}


def test_marker_counts_are_much_faster_than_the_cell_loop():
    from docx import Document
    d = Document()
    t = _table(d, 300, 8, {(r, r % 8): "OUTDATED" for r in range(0, 300, 7)})
    t0 = time.perf_counter()
    old = _legacy_counts(t)
    t1 = time.perf_counter()
    new = SAR._marker_counts(t, MARKERS)
    t2 = time.perf_counter()
    assert new == old
    assert (t2 - t1) < (t1 - t0) / 3.0, (t1 - t0, t2 - t1)


def test_identities_are_looked_up_once_and_follow_a_change(monkeypatch):
    calls = []
    monkeypatch.setattr(SAR.getpass, "getuser", lambda: calls.append(1) or "jsmith")
    monkeypatch.setattr(SAR.platform, "node", lambda: "LABPC-01")
    monkeypatch.setattr(SAR.socket, "gethostname", lambda: "LABPC-01")
    SAR._ID_CACHE.clear()
    out = [SAR.redact_identities(f"saved by jsmith on LABPC-01 #{i} (jsmithson stays)") for i in range(300)]
    assert len(calls) == 1
    assert out[3] == "saved by <redacted> on <redacted> #3 (jsmithson stays)"
    monkeypatch.setattr(SAR.getpass, "getuser", lambda: "other_user")                  # another lookup function: asked again
    assert "other_user" in SAR._identities() and "jsmith" not in SAR._identities()
    assert SAR.redact_identities("other_user here") == "<redacted> here"
    monkeypatch.setenv("COMPUTERNAME", "NEWHOST-9")                                    # another environment: asked again
    assert "NEWHOST-9" in SAR._identities()
    SAR._ID_CACHE.clear()


def test_redaction_equals_the_old_sequential_substitution(monkeypatch):
    import re
    names = ["LABPC-01", "jsmith", "red"]
    monkeypatch.setattr(SAR, "_identities", lambda: names)
    for s in ["jsmith on LABPC-01", "labpc-01/JSMITH", "x redacted red y", "<redacted> jsmith", "no name", ""]:
        old = s
        for n in names:
            old = re.sub(r"(?<![\w])" + re.escape(n) + r"(?![\w])", "<redacted>", old, flags=re.I)
        assert SAR.redact_identities(s) == old, s


def _s4_sess():
    sess, st, res = _sess()
    s4_store.record(st, "bolts", BE.assess_bolt(_bolt(), _lim()).to_dict(), "B1")
    return sess, st, res


def test_the_report_judges_a_stored_s4_record_once(tmp_path, monkeypatch):
    sess, st, res = _s4_sess()
    n = []
    real = s4_verify._shown
    monkeypatch.setattr(s4_verify, "_shown", lambda study, section, rec: n.append(section) or real(study, section, rec))
    SAR.build_sar(sess, tmp_path / "a.docx", ReportOptions(ecss_sar=True), generated=STAMP)
    assert n.count("bolts") == 1, n                                                    # printed by the section and summarised by the chapter: one judgement


def test_judging_is_one_block_and_a_changed_record_is_judged_again():
    sess, st, res = _s4_sess()
    rec = s4_store.get(st)["bolts"]["B1"]
    with s4_verify.judging(st):
        a = s4_verify.shown(st, "bolts", rec)
        with s4_verify.judging(st):                                                    # the same study: the same block, the memo stays
            assert s4_verify.shown(st, "bolts", rec) is a
        assert s4_verify.shown(st, "bolts", rec) is a
        assert s4_verify.shown(st, "bolts", dict(rec)) is not a                        # another record object: judged on its own
    assert s4_verify.shown(st, "bolts", rec) is not a                                  # outside a block nothing is remembered
    assert s4_verify._CTX.get() is None


def test_overlapping_judging_blocks_of_two_threads_do_not_see_each_other():
    """The context of `judging` is per thread: two reports at once restore their own state, and nothing stale (a PASS of a study that changed since) is left behind."""
    import threading
    sess, st, res = _s4_sess()
    rec = s4_store.get(st)["bolts"]["B1"]
    from fea_core.study import Study
    other = Study("other", "")
    in1_started, in2_started, in2_left = threading.Event(), threading.Event(), threading.Event()
    seen: dict = {}

    def first():
        with s4_verify.judging(st):
            seen["first_before"] = s4_verify._CTX.get()["study"] is st
            in1_started.set()
            in2_started.wait(5.0)
            in2_left.wait(5.0)                                                         # the other block has started AND ended inside ours
            seen["first_after"] = s4_verify._CTX.get()["study"] is st
        seen["first_outside"] = s4_verify._CTX.get()

    def second():
        in1_started.wait(5.0)
        with s4_verify.judging(other):
            seen["second_inside"] = s4_verify._CTX.get()["study"] is other
            in2_started.set()
        in2_left.set()
        seen["second_outside"] = s4_verify._CTX.get()

    ts = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10.0)
    assert seen == {"first_before": True, "first_after": True, "first_outside": None, "second_inside": True, "second_outside": None}
    assert s4_verify._CTX.get() is None                                                # nothing leaked into the main thread
    # the other order of leaving: the block that started first ends first (what restored the WRONG context for good with a global dictionary)
    order: list = []
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def ta():
        with s4_verify.judging(st):
            a_in.set()
            b_in.wait(5.0)
        a_out.set()

    def tb():
        a_in.wait(5.0)
        with s4_verify.judging(other):
            b_in.set()
            a_out.wait(5.0)                                                            # A has already left while B is still inside
            order.append(s4_verify._CTX.get()["study"] is other)
        order.append(s4_verify._CTX.get() is None)

    ts = [threading.Thread(target=ta), threading.Thread(target=tb)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10.0)
    assert order == [True, True]
    assert s4_verify._CTX.get() is None
    # nothing is remembered after the blocks: a record judged now is judged on the present study, so a change of the study shows (a leaked context would keep the old basis)
    assert s4_verify.mismatch(st, rec) == []
    next(iter(st.materials.values())).E *= 2.0
    assert s4_verify.mismatch(st, rec)


def test_the_report_computes_the_mass_properties_once(tmp_path, monkeypatch):
    from fea_core.space import mass_props
    sess, st, res = _sess()
    n = []
    real = mass_props.mass_properties
    monkeypatch.setattr(mass_props, "mass_properties", lambda *a, **k: n.append(1) or real(*a, **k))
    SAR.build_sar(sess, tmp_path / "m.docx", ReportOptions(ecss_sar=True), generated=STAMP)
    assert len(n) == 1


def test_sar_documents_are_deterministic_so_the_optimisation_can_be_compared_byte_for_byte(tmp_path):
    sess, st, res = _s4_sess()
    for rec in s4_store.get(st)["bolts"].values():
        rec["stamp"] = "2026-01-01T00:00:00"                                           # the stored stamp is a wall-clock time: part of the study digest
    a = SAR.build_sar(sess, tmp_path / "a.docx", ReportOptions(ecss_sar=True), generated=STAMP)["path"]
    b = SAR.build_sar(sess, tmp_path / "b.docx", ReportOptions(ecss_sar=True), generated=STAMP)["path"]
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_the_report_verifies_the_stored_correlation_once(fe, tmp_path, monkeypatch):  # noqa: F811
    from fea_core.space import corr_run as CR
    from fea_core.space import correlation as C
    from verification import space_cases as SC
    st = _study_with_limits(EXAMPLE)
    td = _test_data(fe)
    f, h = _frf_pair(fe)
    td.functions = [C.TestFunction("100:y", "100", "y", f, h, "frf", "g/N")]
    CR.run_correlation(st, _results(fe), td, sensors=list(SENSORS), predicted={"100:y": C.Predicted(f, h * 1.1, "frf", "", "acceleration", "", "force")})
    mesh = SC.box_tet_mesh()
    st.materials.update(SC.block_study().materials)
    SC.confirm_test_allowables(st)
    sess = SimpleNamespace(study=st, results=SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA)]))
    n = []
    real = CR.verify_stored
    monkeypatch.setattr(CR, "verify_stored", lambda *a, **k: n.append(1) or real(*a, **k))
    r = SAR.build_sar(sess, tmp_path / "c.docx", ReportOptions(ecss_sar=True), generated=STAMP)
    assert {c["key"]: c["status"] for c in r["chapters"]}["correlation"] == "OUTDATED"
    assert len(n) == 1


def test_identity_redactor_equals_the_per_string_substitution(monkeypatch):
    import re
    names = ["LABPC-01", "jsmith", "Zoë-PC"]
    monkeypatch.setattr(SAR, "_identities", lambda: names)
    red = SAR.identity_redactor()
    texts = ["jsmith on LABPC-01", "JSMITH", "ſmith nothing", "zoë-pc and ZOË-PC", "ascii only text", "", "jsmithson", "x_jsmith_y", "a jsmith, b LabPc-01."]
    for s in texts:
        old = s
        for n in names:
            old = re.sub(r"(?<![\w])" + re.escape(n) + r"(?![\w])", "<redacted>", old, flags=re.I)
        assert red(s) == old == SAR.redact_identities(s), s
    assert red(12) == "12"

"""A stored correlation result and the indicative SEREP COC / AOC (owner decision 25. 11. 2026): the result computed with the real rule is stored UNCONFIRMED and is never shown as PASS; a record of an
earlier build that holds PASS / FAIL COC / AOC rows of the SEREP route is not accepted as current."""
from __future__ import annotations

from fea_core.space import corr_orth as O
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from test_space_corr_core_fixes import _fresh
from test_space_correlation import _orthogonality_mass, _results, fe  # noqa: F401  (fixtures: the synthetic model, the unit-mass SEREP gram)


def test_a_result_with_the_real_serep_rule_is_stored_unconfirmed_and_never_current_as_pass(fe, monkeypatch):
    monkeypatch.setattr(C, "orthogonality", O.orthogonality)                                  # the real rule back (the autouse fixture lifts it for the pipeline tests)
    st, res = _fresh(fe)
    assert res.verdict == C.VERDICT_UNCONFIRMED and res.modal.orth.indicative
    state = CR.state_of(st)["result"]
    assert state["provenance"]["orth_mass"]["indicative"] is True
    assert {r["status"] for r in state["rows"] if r["metric"].startswith(("COC", "AOC"))} == {C.UNCONFIRMED}
    v = CR.verify_stored(st, _results(fe))
    assert v["verdict"] == C.VERDICT_UNCONFIRMED and v["current"] and v["reasons"] == []


def test_a_stored_pass_of_the_serep_route_is_never_current_whatever_the_record_says(fe):
    st, res = _fresh(fe)                                                                       # the pipeline reaches PASS through a TAM (the fixture)
    assert res.verdict == C.PASS and res.modal.orth.source == "tam"
    assert CR.verify_stored(st, _results(fe))["verdict"] == "PASS"
    om = CR.state_of(st)["result"]["provenance"]["orth_mass"]
    om["source"] = "serep"                                                                     # a record of an earlier build: the SEREP source with PASS rows ...
    om.pop("indicative", None)                                                                 # ... and no indicative key
    v = CR.verify_stored(st, _results(fe))
    assert v["verdict"] != "PASS" and any("SEREP route is stored as PASS / FAIL" in r for r in v["reasons"])
    om["indicative"] = False                                                                   # ... or one that declares itself not indicative: no exemption (the program never writes it)
    v = CR.verify_stored(st, _results(fe))
    assert v["verdict"] != "PASS" and not v["current"] and any("SEREP route is stored as PASS / FAIL" in r for r in v["reasons"])
    assert any("declares itself not indicative" in r for r in v["reasons"])


def test_a_serep_record_that_declares_itself_not_indicative_is_rejected_even_without_pass_rows(fe, monkeypatch):
    monkeypatch.setattr(C, "orthogonality", O.orthogonality)                                  # the real rule: UNCONFIRMED rows
    st, res = _fresh(fe)
    assert res.verdict == C.VERDICT_UNCONFIRMED
    assert CR.verify_stored(st, _results(fe))["current"]
    CR.state_of(st)["result"]["provenance"]["orth_mass"]["indicative"] = False
    v = CR.verify_stored(st, _results(fe))
    assert not v["current"] and any("declares itself not indicative" in r for r in v["reasons"])

"""F14 S1: requirement set, material allowables and margins of safety (fea_core.space).

The factor values used here are ARBITRARY TEST NUMBERS entered by the tests (verification/space_cases.py); the program ships none.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

from fea_core.eigen import ModeSet
from fea_core.space import margins as MG
from fea_core.space import requirements as RQ
from fea_core.space.model import (FACTOR_KEYS, PART_CLASSES, STRUCTURE_CLASSES, VERIFICATION_APPROACHES, CaseComponent, FactorEntry, Justification,
                                  LimitEntry, MarginCase, MarginRule, MaterialAllowable, RequirementError, RequirementSet, SpaceSettings)
from fea_core.study import LoadCase, Study
import verification.cases  # noqa: F401  (registers the cases; space_cases imports from it, so cases comes first)
from verification import space_cases as SC

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "fea_core" / "space"
LOAD_STEP = "Load"


# ------------------------------------------------------------------ helpers
def _setup(approach: str = "protoflight", **kw):
    """(study, results) of the synthetic block with a confirmed requirement set and confirmed allowables."""
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set(approach, **kw))
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, SC.TEST_SIGMA)], buckling=True)
    return st, res


def _yield_mos(summ: MG.MarginSummary, case: str = LOAD_STEP) -> float:
    return next(r.mos for r in summ.rows if r.check == "yield" and r.case == case and r.mos is not None)


def _expected_yield(approach: str) -> float:
    return SC.TEST_FY / (SC.design_factor("metallic", "yield", approach) * SC.TEST_SIGMA) - 1.0


# ------------------------------------------------------------------ template and the "no hardcoded factor" guard
def test_template_is_empty():
    rs = RQ.ecss_32_10c_template()
    assert len(rs.factors) == len(STRUCTURE_CLASSES) * len(FACTOR_KEYS) * len(VERIFICATION_APPROACHES)
    assert all(f.value is None for f in rs.factors)
    assert all(f.clause == "<clause>" and f.document and not f.revision for f in rs.factors)       # the revision is entered by the user
    assert all(not r.factors and not r.stress_measure and not r.thermal_handling for r in rs.rules)
    assert not rs.is_confirmed() and rs.verification_approach == ""
    with pytest.raises(RequirementError) as e:                    # nothing filled → cannot be confirmed
        RQ.confirm(rs, "someone")
    assert any("approach" in m for m in e.value.issues)
    assert RequirementSet.from_dict(json.loads(json.dumps(rs.to_dict()))) == rs


def test_dataclass_defaults_carry_no_number():
    for obj in (FactorEntry("FOSY", "metallic"), LimitEntry("X"), MaterialAllowable(), MarginRule("yield", "metallic")):
        d = obj.to_dict()
        assert all(v in (None, "", [], False) for k, v in d.items() if k not in ("key", "structure_class", "check", "approach")), d


_FACTOR_NAMES = {k.lower() for k in FACTOR_KEYS}
S1_MODULES = {"model.py", "requirements.py", "margins.py"}


def _numeric(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def test_no_hardcoded_factor_anywhere_in_space_package():
    """Static guard: no factor name is bound to a number, no dict / keyword / default gives a factor a value, in ANY module of fea_core/space."""
    offences: list[str] = []
    for f in sorted(SPACE_DIR.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _numeric(node.value):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.lower() in _FACTOR_NAMES:
                        offences.append(f"{f.name}:{node.lineno} {t.id} = number")
            elif isinstance(node, ast.AnnAssign) and node.value is not None and _numeric(node.value):
                name = node.target.id if isinstance(node.target, ast.Name) else ""
                # fields called value / factor(s) belong to the requirement-set modules of this stage; other modules only must not bind factor names
                if name.lower() in _FACTOR_NAMES or (name in ("value", "factor", "factors") and f.name in S1_MODULES):
                    offences.append(f"{f.name}:{node.lineno} field {name} has a numeric default")
            elif isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.lower() in _FACTOR_NAMES and _numeric(v):
                        offences.append(f"{f.name}:{node.lineno} dict entry {k.value}")
            elif isinstance(node, ast.keyword) and node.arg and node.arg.lower() in _FACTOR_NAMES and _numeric(node.value):
                offences.append(f"{f.name}:{node.value.lineno} keyword {node.arg}")
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\b(FOSY|FOSU|KQ|KP|KM|KLD|KMP)\b\s*[:=]\s*[0-9]", text):
            offences.append(f"{f.name}: text '{m.group(0)}'")
    assert not offences, offences


def test_template_json_has_no_value():
    text = RQ.to_json(RQ.ecss_32_10c_template())
    assert not re.search(r'"value":\s*[-0-9]', text)


# ------------------------------------------------------------------ requirement set: validation, confirmation, import
def test_confirm_needs_complete_provenance_and_withdraws_on_edit():
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    assert not rs.is_confirmed()
    RQ.confirm(rs, "Test Engineer", "2026-02-03")
    assert rs.is_confirmed() and rs.confirmed_by == "Test Engineer" and rs.date == "2026-02-03"
    # editing a value withdraws the confirmation (digest), and the flag alone does not confirm
    row = rs.entry("FOSY", "metallic")
    assert row is not None and row.value is not None
    row.value = row.value + 0.25
    assert rs.confirmed and not rs.is_confirmed()
    # a rule edit as well
    RQ.confirm(rs, "Test Engineer")
    assert rs.is_confirmed()
    rs.rules[0].note = "changed"
    assert not rs.is_confirmed()


def test_confirm_refuses_missing_things():
    rs = SC.make_requirement_set("protoflight", confirmed=False)
    row = rs.entry("KQ", "metallic")
    assert row is not None
    row.value = None                                              # a factor of a rule is not entered
    with pytest.raises(RequirementError, match="KQ"):
        RQ.confirm(rs, "x")
    row.value = 1.5
    row.clause = RQ.CLAUSE_PLACEHOLDER                            # clause placeholder not replaced
    with pytest.raises(RequirementError, match="clause"):
        RQ.confirm(rs, "x")
    row.clause = "c"
    with pytest.raises(RequirementError, match="name of the person"):
        RQ.confirm(rs, "  ")
    rs.verification_approach = ""
    with pytest.raises(RequirementError, match="approach"):
        RQ.confirm(rs, "x")


def test_validation_errors():
    rs = RQ.ecss_32_10c_template()
    rs.factors[0].value = -1.0
    rs.factors[1].value = float("nan")
    rs.factors.append(FactorEntry("FOSY", "metallic", "protoflight", 1.5))        # duplicate of a template row
    rs.rules[0].factors = ["NOPE"]
    rs.rules[0].stress_measure = "tresca_x"
    rs.verification_approach = "magic"
    msgs = " | ".join(str(i) for i in RQ.validate(rs) if i.level == "error")
    for needle in ("positive", "duplicate", "NOPE", "stress measure", "approach"):
        assert needle in msgs, (needle, msgs)


def test_json_round_trip_and_import_withdraws_confirmation():
    rs = SC.make_requirement_set("analysis_only")
    assert rs.is_confirmed()
    back = RQ.from_json(RQ.to_json(rs))
    assert not back.confirmed and not back.is_confirmed()                  # an imported set is confirmed again by its new user
    back.verification_approach = rs.verification_approach
    assert back.content_digest() == rs.content_digest()
    RQ.confirm(back, "second user")
    assert back.is_confirmed()
    for bad in ("[]", "{", '{"factors": [1]}', '{"factors": [{"value": "abc"}]}', "x" * (RQ.MAX_JSON_BYTES + 1)):
        with pytest.raises(RequirementError):
            RQ.from_json(bad)


def test_factor_csv_round_trip_and_formula_guard():
    rs = SC.make_requirement_set("protoflight")
    rs.factors[0].note = "=cmd|' /C calc'!A0"
    text = RQ.factors_to_csv(rs)
    assert "'=cmd" in text and "\n=cmd" not in text
    rows = RQ.factors_from_csv(text)
    assert len(rows) == len(rs.factors)
    assert [r.value for r in rows] == [f.value for f in rs.factors]
    with pytest.raises(RequirementError):
        RQ.factors_from_csv("key,structure_class,value\nFOSY,metallic,abc\n")


# ------------------------------------------------------------------ allowables
def test_allowable_lifecycle():
    st = SC.block_study()
    mat = st.materials[st.default_material or ""]
    assert RQ.allowable_of(mat) is None
    with pytest.raises(RequirementError):
        RQ.confirm_allowable(mat, "x")
    RQ.set_allowable(mat, MaterialAllowable(basis="A", Fy=250.0, source="handbook"))
    assert not RQ.allowable_of(mat).is_confirmed()
    RQ.confirm_allowable(mat, "Test Engineer", "2026-01-01")
    rec = RQ.allowable_of(mat)
    assert rec.is_confirmed() and rec.is_verified()
    rec.Fy = 260.0                                                  # edit → confirmation withdrawn
    RQ.set_allowable(mat, rec)
    assert not RQ.allowable_of(mat).is_confirmed()
    RQ.confirm_allowable(mat, "Test Engineer")
    rec = RQ.allowable_of(mat)
    rec.basis = "typical"
    RQ.set_allowable(mat, rec)
    RQ.confirm_allowable(mat, "Test Engineer")
    assert RQ.allowable_of(mat).is_confirmed() and not RQ.allowable_of(mat).is_verified()      # a typical basis is never "verified"
    with pytest.raises(RequirementError, match="positive"):
        bad = RQ.allowable_of(mat)
        bad.Fu = -3.0
        RQ.set_allowable(mat, bad)
        RQ.confirm_allowable(mat, "x")


# ------------------------------------------------------------------ margins: analytical reference matrix
@pytest.mark.parametrize("approach", VERIFICATION_APPROACHES)
@pytest.mark.parametrize("cls", STRUCTURE_CLASSES)
def test_margin_matches_hand_calculation(cls, approach):
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set(approach), default_class=cls if cls in PART_CLASSES else "metallic")
    RQ.confirm(st.space.requirement_set, "test engineer", "2026-01-01", scope=st.space)           # confirmed for these classes
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, SC.TEST_SIGMA)], buckling=True)
    summ = MG.compute_margins(st, res)
    if cls == "composite":                                          # no stress-based rule: never assessed, never PASS
        assert summ.verdict == MG.UNCONFIRMED
        assert all(r.mos is None for r in summ.rows if r.check in ("yield", "ultimate"))
        return
    if cls == "buckling":
        row = next(r for r in summ.rows if r.check == "buckling")
        assert row.mos == pytest.approx(SC.TEST_LAMBDA / SC.design_factor("buckling", "buckling", approach) - 1.0, rel=1e-12)
        return
    for check, allow in (("yield", SC.TEST_FY), ("ultimate", SC.TEST_FU)):
        row = next(r for r in summ.rows if r.check == check and r.case == LOAD_STEP)
        j = SC.design_factor(cls, check, approach)
        assert row.design_factor == pytest.approx(j, rel=1e-14)
        assert row.mos == pytest.approx(allow / (j * SC.TEST_SIGMA) - 1.0, rel=1e-12)
        assert row.status == "pass" and row.allowable == allow
    assert summ.verdict == MG.PASS if cls == "metallic" else summ.verdict in (MG.PASS, MG.UNCONFIRMED)


def test_all_confirmed_gives_pass_and_negative_margin_gives_fail():
    st, res = _setup()
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.PASS and not s.reasons and s.confirmed and s.approach == "protoflight"
    assert s.worst is not None and s.worst.mos > 0
    st2, res2 = _setup()
    SC.confirm_test_allowables(st2, fy=100.0, fu=110.0)             # allowables below the applied stress → negative margins
    s2 = MG.compute_margins(st2, res2)
    assert s2.verdict == MG.FAIL and s2.worst is not None and s2.worst.mos < 0


# ------------------------------------------------------------------ negative tests: never PASS without confirmed inputs
def test_unconfirmed_requirement_set_is_never_pass():
    st, res = _setup()
    st.space.requirement_set.withdraw_confirmation()
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and not s.signoff_ready
    assert any("not confirmed" in r for r in s.reasons)
    assert _yield_mos(s) == pytest.approx(_expected_yield("protoflight"), rel=1e-12)       # the numbers are still there
    assert all(r.status == "unconfirmed" for r in s.rows)


def test_edit_after_confirmation_is_never_pass():
    st, res = _setup()
    row = st.space.requirement_set.entry("FOSY", "metallic")
    row.value = row.value * 0.5
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and not s.confirmed


def test_negative_margin_with_unconfirmed_inputs_is_not_fail():
    st, res = _setup()
    SC.confirm_test_allowables(st, fy=100.0, fu=110.0)
    st.space.requirement_set.withdraw_confirmation()
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED                               # not FAIL, not PASS: the inputs are not confirmed
    assert any(r.negative for r in s.rows)


def test_unconfirmed_or_unverified_allowable_is_never_pass():
    st, res = _setup()
    mat = st.materials[st.default_material or ""]
    rec = RQ.allowable_of(mat)
    rec.Fy += 1.0                                                    # edited after the confirmation
    RQ.set_allowable(mat, rec)
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("allowables" in r for r in s.reasons)
    st2, res2 = _setup()
    SC.confirm_test_allowables(st2, basis="typical")
    s2 = MG.compute_margins(st2, res2)
    assert s2.verdict == MG.UNCONFIRMED and any("basis" in r for r in s2.reasons)


def test_material_without_allowables_uses_re_rm_but_never_passes():
    st, res = _setup()
    st.materials[st.default_material or ""].allowables = None
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED
    row = next(r for r in s.rows if r.check == "yield")
    assert row.allowable == 250.0 and any("indicative" in x for x in row.reasons)       # Re of the material, flagged
    mat = st.materials[st.default_material or ""]
    mat.Re = mat.Rm = None
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and all(r.mos is None for r in s.rows if r.check == "yield")


def test_no_requirement_set_is_indicative_only():
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, 100.0)])
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and not s.confirmed
    assert _yield_mos(s) == pytest.approx(250.0 / 100.0 - 1.0, rel=1e-12)                # Re / sigma - 1 (factor 1: indicative)


def test_incomplete_rule_is_never_pass():
    st, res = _setup()
    st.space.requirement_set.rule("yield", "metallic").factors = []        # the rule lost its factors after the confirmation
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED


def test_nonlinear_study_is_never_pass():
    st, res = _setup()
    st.analysis.kind = "nonlinear"
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("nonlinear" in r for r in s.reasons)


def test_orthotropic_material_is_flagged():
    st, res = _setup()
    st.materials[st.default_material or ""].ortho = {"E1": 1.0}
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("orthotropic" in r for r in s.reasons)


# ------------------------------------------------------------------ singularities
def test_singularity_is_separated_and_needs_a_justification():
    mesh = SC.box_tet_mesh()
    st, _ = _setup()
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, SC.TEST_SIGMA, spike=3.0)])
    s = MG.compute_margins(st, res)
    y = next(r for r in s.rows if r.check == "yield")
    j = SC.design_factor("metallic", "yield", "protoflight")
    assert y.mos == pytest.approx(SC.TEST_FY / (j * SC.TEST_SIGMA) - 1.0, rel=1e-12)             # nominal stress governs
    assert y.mos_raw == pytest.approx(SC.TEST_FY / (j * 3.0 * SC.TEST_SIGMA) - 1.0, rel=1e-12)    # the spike only in the raw margin
    assert y.mos_raw < 0 < y.mos and y.classification != "singularity"
    assert s.verdict == MG.PASS                                       # the margin outside singularities is positive ...
    assert s.singularities and not s.signoff_ready and any("singularity not justified" in x for x in s.open_items)   # ... but no sign-off yet
    sg = s.singularities[0]
    assert sg.part == "Block" and sg.cases == [LOAD_STEP] and set(sg.checks) == {"yield", "ultimate"} and "support" in sg.cause
    MG.add_justification(st.space, sg.part, sg.location, "Rigid clamp of the test; the real attachment is a bolted flange.", author="tester",
                         date="2026-01-02")
    s2 = MG.compute_margins(st, res)
    assert s2.signoff_ready and s2.singularities[0].justified
    # a note far away from the place does not match
    st.space.justifications[0].xyz = [500.0, 500.0, 500.0]
    assert not MG.compute_margins(st, res).signoff_ready
    with pytest.raises(ValueError):
        MG.add_justification(st.space, "Block", [0, 0, 0], "  ")


def test_all_places_singular_gives_no_margin():
    mesh = SC.box_tet_mesh(nx=2, ny=2, nz=2)                         # so small that the whole block is within the support's cluster radius
    st = SC.block_study()
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("protoflight"))
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, SC.TEST_SIGMA, spike=3.0)])
    s = MG.compute_margins(st, res)
    y = next(r for r in s.rows if r.check == "yield")
    assert s.verdict == MG.UNCONFIRMED and y.mos is None and y.mos_raw is not None and y.classification == "singularity"
    assert any("singularities" in x for x in y.reasons)


# ------------------------------------------------------------------ combined mechanical + thermal, combinations, cases
def _two_state_study(handling: str, thermal_factors: list[str] | None = None, load_extra=None):
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    st.load_cases = [LoadCase("Mech", list(load_extra or [])), LoadCase("Therm", [])]
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("analysis_only", thermal=handling, thermal_factors=thermal_factors),
                             cases=[MarginCase("Both", [CaseComponent("Mech", 1.0, "mechanical"), CaseComponent("Therm", 1.0, "thermal")])])
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Mech", 100.0), SC.uniform_state(mesh, "Therm", 20.0)])
    return st, res


@pytest.mark.parametrize("handling", ["same_as_mechanical", "none", "separate"])
def test_thermal_handling(handling):
    st, res = _two_state_study(handling, ["KMP"] if handling == "separate" else None)
    s = MG.compute_margins(st, res)
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    jt = {"same_as_mechanical": jm, "none": 1.0, "separate": SC.test_factor("KMP", "metallic", "analysis_only")}[handling]
    row = next(r for r in s.rows if r.case == "Both" and r.check == "yield")
    assert row.mos == pytest.approx(SC.TEST_FY / (jm * 100.0 + jt * 20.0) - 1.0, rel=1e-12)
    assert row.thermal_factor == pytest.approx(jt, rel=1e-14)


def test_relieving_thermal_load_counts_with_a_factor_of_one():
    """ECSS-E-ST-32-10C Rev.2 Corr.1 4.3.2.1b / c: the factors apply to additive thermal loads; a relieving load takes 1,0 or less (never more credit than J x relief)."""
    st, res = _two_state_study("same_as_mechanical")
    st.space.cases.append(MarginCase("Relief", [CaseComponent("Mech", 1.0, "mechanical"), CaseComponent("Therm", -1.0, "thermal")]))
    s = MG.compute_margins(st, res)
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    both = next(r for r in s.rows if r.case == "Both" and r.check == "yield")
    rel = next(r for r in s.rows if r.case == "Relief" and r.check == "yield")
    assert both.mos == pytest.approx(SC.TEST_FY / (jm * 120.0) - 1.0, rel=1e-12)                 # additive: J x (100 + 20)
    assert rel.mos == pytest.approx(SC.TEST_FY / (jm * 100.0 - 20.0) - 1.0, rel=1e-12)           # relieving: J x 100 - 1,0 x 20 (not J x 80)
    assert rel.mos < SC.TEST_FY / (jm * 80.0) - 1.0
    # "none": the thermal load carries 1,0 anyway, a relieving thermal load is the main combination
    st2, res2 = _two_state_study("none")
    st2.space.cases.append(MarginCase("Relief", [CaseComponent("Mech", 1.0, "mechanical"), CaseComponent("Therm", -1.0, "thermal")]))
    row = next(r for r in MG.compute_margins(st2, res2).rows if r.case == "Relief" and r.check == "yield")
    assert row.mos == pytest.approx(SC.TEST_FY / (jm * 100.0 - 20.0) - 1.0, rel=1e-12)


@pytest.mark.parametrize("handling, sm, sth, expect", [
    ("none", -10.0, 20.0, "1,0 sm + 1,0 sth"),
    ("none", 100.0, -20.0, "J_m sm + 1,0 sth"),
    ("separate", -10.0, 20.0, "1,0 sm + J_t sth"),
    ("separate", 100.0, -20.0, "J_m sm + 1,0 sth"),
    ("separate", 100.0, 20.0, "J_m sm + J_t sth"),
    ("none", 100.0, 20.0, "J_m sm + 1,0 sth"),
])
def test_relieving_rule_for_the_other_thermal_handlings(handling, sm, sth, expect):
    """ECSS-E-ST-32-10C 4.3.2.1 c holds for every thermal handling: the envelope over J_m S_m + J_t S_th, J_m S_m + 1,0 S_th and 1,0 S_m + J_t S_th (probe values of the review)."""
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    jt = {"none": 1.0, "separate": SC.test_factor("KMP", "metallic", "analysis_only")}[handling]
    st = SC.block_study()
    mesh = SC.box_tet_mesh()
    st.load_cases = [LoadCase("Mech", []), LoadCase("Therm", [])]
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("analysis_only", thermal=handling, thermal_factors=["KMP"] if handling == "separate" else None),
                             cases=[MarginCase("Both", [CaseComponent("Mech", 1.0, "mechanical"), CaseComponent("Therm", 1.0, "thermal")])])
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Mech", sm), SC.uniform_state(mesh, "Therm", sth)])
    row = next(r for r in MG.compute_margins(st, res).rows if r.case == "Both" and r.check == "yield")
    combos = {"J_m sm + J_t sth": abs(jm * sm + jt * sth), "J_m sm + 1,0 sth": abs(jm * sm + sth) if jt > 1.0 else abs(jm * sm + jt * sth),
              "1,0 sm + J_t sth": abs(sm + jt * sth), "1,0 sm + 1,0 sth": abs(sm + sth)}
    env = max(abs(jm * sm + jt * sth), abs(jm * sm + sth) if jt > 1.0 else 0.0, abs(sm + jt * sth) if jm > 1.0 else 0.0)
    assert row.design_stress == pytest.approx(env, rel=1e-12)
    assert row.design_stress == pytest.approx(combos[expect] if expect in combos else env, rel=1e-12)
    assert row.mos == pytest.approx(SC.TEST_FY / env - 1.0, rel=1e-12)


def test_relieving_rule_probe_values():
    """The numbers of the review: J_m = 1,336, separate J_t = 1,070."""
    st, res = _signed_two_state_study(-10.0, 20.0)
    for f in st.space.requirement_set.factors:
        if f.approach == "analysis_only" and f.structure_class == "metallic" and f.key in ("FOSY", "KQ"):
            f.value = 1.336 if f.key == "FOSY" else 1.0
    st.space.requirement_set.rule("yield", "metallic").thermal_handling = "none"
    row = next(r for r in MG.compute_margins(st, res).rows if r.case == "Both" and r.check == "yield")
    assert row.design_stress == pytest.approx(10.0, abs=1e-9)


def _signed_two_state_study(sig_mech: float, sig_therm: float):
    mesh = SC.box_tet_mesh()
    st = SC.block_study()
    st.load_cases = [LoadCase("Mech", []), LoadCase("Therm", [])]
    SC.confirm_test_allowables(st)
    st.space = SpaceSettings(requirement_set=SC.make_requirement_set("analysis_only", thermal="same_as_mechanical"),
                             cases=[MarginCase("Both", [CaseComponent("Mech", 1.0, "mechanical"), CaseComponent("Therm", 1.0, "thermal")])])
    return st, SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Mech", sig_mech), SC.uniform_state(mesh, "Therm", sig_therm)])


def test_relieving_mechanical_load_against_a_dominant_thermal_one():
    """ECSS-E-ST-32-10C 4.3.2.1 c, third combination: sigma_m (factor 1,0, relieving) + J sigma_th. Probe sigma_m = -10, sigma_th = +20: with J = 1,336 the stress is 16,72."""
    st, res = _signed_two_state_study(-10.0, 20.0)
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    row = next(r for r in MG.compute_margins(st, res).rows if r.case == "Both" and r.check == "yield")
    assert row.mos == pytest.approx(SC.TEST_FY / (-10.0 + jm * 20.0) - 1.0, rel=1e-12)           # sigma_m + J sigma_th, not |J (sigma_m + sigma_th)| or |J sigma_m + sigma_th|
    assert row.mos < SC.TEST_FY / (jm * 10.0) - 1.0 and row.mos < SC.TEST_FY / abs(jm * -10.0 + 20.0) - 1.0
    # the probe of the review: the factors of the yield rule set to J = 1,336 -> the design stress is 16,72 MPa
    for f in st.space.requirement_set.factors:
        if f.approach == "analysis_only" and f.structure_class == "metallic" and f.key in ("FOSY", "KQ"):
            f.value = 1.336 if f.key == "FOSY" else 1.0
    probe = next(r for r in MG.compute_margins(st, res).rows if r.case == "Both" and r.check == "yield")
    assert probe.design_stress == pytest.approx(16.72, abs=1e-9)
    # a purely additive case is not changed by the extra combination
    st2, res2 = _signed_two_state_study(100.0, 20.0)
    add = next(r for r in MG.compute_margins(st2, res2).rows if r.case == "Both" and r.check == "yield")
    assert add.mos == pytest.approx(SC.TEST_FY / (jm * 120.0) - 1.0, rel=1e-12)


def test_thermal_handling_undecided_is_not_pass():
    st, res = _two_state_study("same_as_mechanical")
    st.space.requirement_set.rule("yield", "metallic").thermal_handling = ""
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED


def test_state_with_mechanical_and_thermal_loads_cannot_be_separated():
    from fea_core.study import Force, ThermalLoad
    st, res = _two_state_study("separate", ["KMP"], load_extra=[Force("Support", [1.0, 0.0, 0.0]), ThermalLoad(value=50.0)])
    s = MG.compute_margins(st, res)
    mech = next(r for r in s.rows if r.case == "Mech" and r.check == "yield")
    assert mech.status == "unconfirmed" and any("cannot be factored separately" in x for x in mech.reasons)
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    jt = SC.test_factor("KMP", "metallic", "analysis_only")
    assert mech.mos == pytest.approx(SC.TEST_FY / (max(jm, jt) * 100.0) - 1.0, rel=1e-12)       # the larger factor (conservative)


def test_combination_cases_use_the_factors_of_the_combination():
    from fea_core.combinations import Combination
    st, res = _two_state_study("same_as_mechanical")
    st.combinations = [Combination("K1", factors={"Mech": 1.5, "Therm": 0.5}), Combination("Env", envelope=True, envelope_combinations=["K1"])]
    s = MG.compute_margins(st, res)
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    row = next(r for r in s.rows if r.case == "K1" and r.check == "yield")
    assert row.mos == pytest.approx(SC.TEST_FY / (jm * (1.5 * 100.0 + 0.5 * 20.0)) - 1.0, rel=1e-12)
    assert not any(r.case == "Env" for r in s.rows)                       # the members of an envelope are cases of their own


def test_preload_state_is_only_part_of_combinations():
    st, res = _setup()
    res.steps.append(SC.uniform_state(res.mesh, "Bolt preload", 10.0))
    s = MG.compute_margins(st, res)
    assert "Bolt preload" not in s.cases


def test_scalar_envelope_source():
    st, res = _setup()
    n = len(res.mesh.nodes)
    s = MG.compute_margins(st, res, extra_sources=(MG.ScalarSource("Random 3-sigma", np.full(n, 60.0)),))
    row = next(r for r in s.rows if r.case == "Random 3-sigma" and r.check == "yield")
    assert row.mos == pytest.approx(SC.TEST_FY / (SC.design_factor("metallic", "yield", "protoflight") * 60.0) - 1.0, rel=1e-12)


def test_buckling_margin_and_class():
    st, res = _setup()
    s = MG.compute_margins(st, res)
    row = next(r for r in s.rows if r.check == "buckling")
    assert row.mos == pytest.approx(SC.TEST_LAMBDA / SC.design_factor("buckling", "buckling", "protoflight") - 1.0, rel=1e-12)
    res.modes = [ModeSet("modal", "Modes", [100.0], np.zeros((1, 1, 3)))]
    assert not any(r.check == "buckling" for r in MG.compute_margins(st, res).rows)


# ------------------------------------------------------------------ persistence in Study
def test_study_json_round_trip():
    st, _ = _setup()
    MG.add_justification(st.space, "Block", [0.0, 0.0, 0.0], "accepted", cause="support edge", author="tester", date="2026-01-02")
    st.space.part_classes = {"Block": "joints_inserts"}
    st.space.cases = [MarginCase("Both", [CaseComponent("Mech", 1.0, "mechanical")])]
    st.space.extra = {"future_stage": {"k": 1}}
    d = json.loads(json.dumps(st.to_dict()))
    assert "space" in d and "space" in d["features"]
    back = Study.from_dict(d)
    assert back.space is not None and back.space.requirement_set == st.space.requirement_set
    assert back.space.requirement_set.is_confirmed() and back.space.justifications == st.space.justifications
    assert back.space.part_classes == {"Block": "joints_inserts"} and back.space.cases == st.space.cases
    assert back.space.extra == {"future_stage": {"k": 1}} and back.load_warnings == []
    mat = back.materials[back.default_material or ""]
    assert RQ.allowable_of(mat).is_verified()
    assert json.dumps(back.to_dict(), sort_keys=True) == json.dumps(st.to_dict(), sort_keys=True)


def test_old_files_are_unchanged():
    st = SC.block_study()
    d = st.to_dict()
    assert "space" not in d and "space" not in d["features"]
    assert all("allowables" not in m for m in d["materials"].values())
    assert Study.from_dict(d).space is None


def test_broken_space_block_does_not_stop_opening():
    st = SC.block_study()
    d = st.to_dict()
    d["space"] = {"requirement_set": {"factors": [{"key": "FOSY", "structure_class": "metallic", "value": "abc"}]}}
    back = Study.from_dict(d)
    assert back.space is None and any("space-structures" in w for w in back.load_warnings)


# ------------------------------------------------------------------ the hotspot classifier is shared (no MCP session)
def test_hotspots_module_is_independent_of_mcp():
    import subprocess
    import sys
    code = ("import sys; import fea_core.space.margins, fea_core.hotspots; "
            "bad = [m for m in sys.modules if m.startswith('fea_core.mcp')]; print('MCP', ' '.join(bad))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip() == "MCP", r.stdout


def test_justification_round_trip():
    j = Justification("P", [1.0, 2.0, 3.0], "n", 4.0, "c", "a", "2026-01-01")
    assert Justification.from_dict(j.to_dict()) == j


# ------------------------------------------------------------------ the registered verification cases
@pytest.mark.parametrize("key", ["S1A", "S1B", "S1C"])
def test_verification_cases_pass(key, tmp_path):
    case = verification.cases.CASES[key]
    study, parts = case.build(tmp_path / key)
    out = case.run(study, tmp_path / key / "run", parts)
    checks = case.evaluate(out, study)
    assert checks and all(c.ok for c in checks), [(c.name, c.value, c.ref) for c in checks if not c.ok]


@pytest.mark.slow
def test_solved_bar_matches_hand_calculation(tmp_path):
    """S1D: the real solver (ccx or the direct path), margins outside the clamp = hand value of the uniform tension."""
    from fea_core.solve import run_study
    case = verification.cases.CASES["S1D"]
    study, parts = case.build(tmp_path)
    res = run_study(study, tmp_path / "run")
    checks = case.evaluate(res, study)
    assert all(c.ok for c in checks), [(c.name, c.value, c.ref) for c in checks if not c.ok]


# ------------------------------------------------------------------ review fixes: a missing / undecidable input is never PASS
def _reconfirmed(st):
    """The requirement set after an edit, confirmed again (so only the missing rule can be the reason of the verdict)."""
    RQ.confirm(st.space.requirement_set, "test engineer", "2026-01-01", scope=st.space)


@pytest.mark.parametrize("missing", ["yield", "ultimate"])
def test_missing_primary_rule_is_never_pass(missing):
    st, res = _setup()
    rs = st.space.requirement_set
    rs.rules = [r for r in rs.rules if not (r.check == missing and r.structure_class == "metallic")]
    _reconfirmed(st)
    assert rs.is_confirmed()
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and not s.signoff_ready
    row = next(r for r in s.rows if r.check == missing)
    assert row.status == "unconfirmed" and row.mos is None and any(f"no '{missing}' rule" in x for x in row.reasons)


def test_buckling_modes_without_a_buckling_rule_are_never_pass():
    st, res = _setup()
    rs = st.space.requirement_set
    rs.rules = [r for r in rs.rules if r.check != "buckling"]
    _reconfirmed(st)
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("no 'buckling' rule" in x for x in s.reasons)


def test_incomplete_buckling_rule_is_never_pass():
    st, res = _setup()
    st.space.requirement_set.rule("buckling", "buckling").factors = []
    _reconfirmed(st)
    s = MG.compute_margins(st, res)
    row = next(r for r in s.rows if r.check == "buckling")
    assert s.verdict == MG.UNCONFIRMED and row.status == "unconfirmed"


def test_not_applicable_is_an_explicit_rule_state_with_provenance():
    st, res = _setup()
    st.space.default_class = "composite"
    rs = st.space.requirement_set
    for chk in ("yield", "ultimate"):                    # the composite rules do not exist: not assessed
        assert rs.rule(chk, "composite") is None
    assert MG.compute_margins(st, res).verdict == MG.UNCONFIRMED
    for chk in ("yield", "ultimate"):
        rs.rules.append(MarginRule(chk, "composite", not_applicable=True, document="doc", revision="r", clause="c", note="no composite part"))
    _reconfirmed(st)
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.PASS and any("declared not applicable" in w for w in s.warnings) and not [r for r in s.rows if r.check in ("yield", "ultimate")]
    # without provenance the set cannot be confirmed and a flag in an unconfirmed set does not count
    rs.rules[-1].clause = ""
    with pytest.raises(RequirementError, match="not applicable"):
        RQ.confirm(rs, "x")
    assert not rs.is_confirmed()
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("not applicable" in x for r in s.rows for x in r.reasons)
    back = RequirementSet.from_dict(rs.to_dict())
    assert back.rules[-1].not_applicable and back.rules[-2].not_applicable


def test_not_applicable_does_not_change_the_digest_of_other_rules():
    assert "not_applicable" not in MarginRule("yield", "metallic").to_dict()


def test_user_case_with_an_absent_state_is_a_reason():
    st, res = _setup()
    st.space.cases = [MarginCase("Ghost", [CaseComponent("Load", 1.0, "mechanical"), CaseComponent("NoSuchState", 1.0, "thermal")])]
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("NoSuchState" in x for x in s.reasons) and not any(r.case == "Ghost" for r in s.rows)
    st.space.cases = [MarginCase(LOAD_STEP, [CaseComponent("Load", 1.0, "mechanical")])]          # name clash: skipped -> reason
    assert any("name of another case" in x for x in MG.compute_margins(st, res).reasons)
    st.space.cases = [MarginCase("Empty", [])]
    assert MG.compute_margins(st, res).verdict == MG.UNCONFIRMED


def test_combination_with_unknown_states_is_a_reason():
    from fea_core.combinations import Combination
    st, res = _setup()
    st.combinations = [Combination("K9", factors={"Load": 1.0, "Wind": 1.5})]
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("K9" in x and "Wind" in x for x in s.reasons)


def test_scalar_source_that_cannot_be_evaluated_is_a_reason():
    st, res = _setup()
    n = len(res.mesh.nodes)
    s = MG.compute_margins(st, res, extra_sources=(MG.ScalarSource("Short", np.full(n - 1, 60.0)),))
    assert s.verdict == MG.UNCONFIRMED and any("Short" in x for x in s.reasons)
    st.space.requirement_set.rule("yield", "metallic").stress_measure = "max_principal"
    _reconfirmed(st)
    s = MG.compute_margins(st, res, extra_sources=(MG.ScalarSource("Env", np.full(n, 60.0)),))
    assert s.verdict == MG.UNCONFIRMED
    assert any(r.case == "Env" and r.check == "yield" and r.status == "unconfirmed" and any("equivalent (von Mises)" in x for x in r.reasons)
               for r in s.rows)
    s = MG.compute_margins(st, res, extra_sources=(MG.ScalarSource("Odd", np.full(n, 60.0), kind="whatever"),))
    assert any("Odd" in x for x in s.reasons)


def test_non_finite_stress_is_never_pass():
    st, res = _setup()
    res.steps[0].S[3, 0] = float("nan")
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and not s.signoff_ready
    row = next(r for r in s.rows if r.check == "yield")
    assert row.status == "unconfirmed" and row.mos is None and any("not finite" in x for x in row.reasons)
    res.steps[0].S[3, 0] = float("inf")
    assert MG.compute_margins(st, res).verdict == MG.UNCONFIRMED


def test_unknown_kind_of_state_is_never_factored_as_mechanical():
    st, res = _two_state_study("separate", ["KMP"])
    st.load_cases = [LoadCase("Mech", [])]                                 # the load case "Therm" does not exist: its kind is unknown
    s = MG.compute_margins(st, res)
    row = next(r for r in s.rows if r.case == "Therm" and r.check == "yield")
    assert row.status == "unconfirmed" and any("larger factor" in x for x in row.reasons)
    jm = SC.design_factor("metallic", "yield", "analysis_only")
    jt = SC.test_factor("KMP", "metallic", "analysis_only")
    assert row.mos == pytest.approx(SC.TEST_FY / (max(jm, jt) * 20.0) - 1.0, rel=1e-12)
    st.space.cases = [MarginCase("Odd", [CaseComponent("Mech", 1.0, "nonsense")])]
    row = next(r for r in MG.compute_margins(st, res).rows if r.case == "Odd" and r.check == "yield")
    assert row.status == "unconfirmed" and any("kind of the loads" in x for x in row.reasons)


def test_parts_that_are_not_assessed_are_listed():
    from fea_core.study import BoltSpec
    st, res = _setup()
    assert MG.compute_margins(st, res).verdict == MG.PASS
    st.shell.enabled = True
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and any("shell" in x for x in s.reasons)
    st.shell.enabled = False
    st.frame.add_node("a", [0, 0, 0])
    from fea_core.beams.model import FrameMember
    st.frame.members["m"] = FrameMember.__new__(FrameMember)
    assert any("beam" in x for x in MG.compute_margins(st, res).reasons)
    st.frame.members.clear()
    st.bolts.append(BoltSpec.__new__(BoltSpec))
    st.bolts[0].__dict__.update({"name": "B1", "enabled": True})
    assert any("bolts are not assessed (B1" in x for x in MG.compute_margins(st, res).reasons)


def test_confirmation_channel_is_stored_and_shown():
    st, res = _setup()
    rs = st.space.requirement_set
    assert rs.confirmed_via == ""                                       # the API default: not claimed to be the GUI
    RQ.confirm(rs, "Test Engineer", via="mcp")
    mat = st.materials[st.default_material or ""]
    RQ.confirm_allowable(mat, "Test Engineer", via="mcp")
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.PASS and s.confirmed_via == "mcp" and s.allowables_via == {mat.name: "mcp"}
    assert sum("through the MCP interface" in w for w in s.warnings) == 2
    d = s.to_dict()
    assert d["requirement_set"]["confirmed_via"] == "mcp" and d["allowables_confirmed_via"] == {mat.name: "mcp"}
    assert RequirementSet.from_dict(rs.to_dict()).confirmed_via == "mcp" and RQ.allowable_of(mat).confirmed_via == "mcp"
    rs.withdraw_confirmation()
    assert rs.confirmed_via == ""
    RQ.confirm(rs, "Test Engineer", via="gui")
    assert MG.compute_margins(st, res).confirmed_via == "gui"


def test_verdict_text_and_open_items_say_when_a_pass_is_not_a_sign_off():
    mesh = SC.box_tet_mesh()
    st, _ = _setup()
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, LOAD_STEP, SC.TEST_SIGMA, spike=3.0)])
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.PASS and not s.signoff_ready
    assert s.verdict_text == "PASS – not ready for sign-off (open singularities: 1)" and s.to_dict()["verdict_text"] == s.verdict_text
    MG.add_justification(st.space, "Block", s.singularities[0].location, "accepted", author="t")
    s = MG.compute_margins(st, res)
    assert s.signoff_ready and s.verdict_text == "PASS" and not s.open_items
    # a FAIL is an open item of its own: the list is never empty under FAIL
    st2, res2 = _setup()
    SC.confirm_test_allowables(st2, fy=100.0, fu=110.0)
    f = MG.compute_margins(st2, res2)
    assert f.verdict == MG.FAIL and not f.signoff_ready and f.verdict_text == "FAIL" and any("negative" in x for x in f.open_items)


def test_changing_the_structure_classes_withdraws_the_confirmation():
    st, res = _setup()
    assert MG.compute_margins(st, res).verdict == MG.PASS
    st.space.part_classes = {"Block": "joints_inserts"}                   # other rules apply to the part now
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.UNCONFIRMED and not s.confirmed and not st.space.requirement_confirmed()
    assert any("structure classes" in r for r in s.reasons) and st.space.requirement_set.is_confirmed()       # the content itself is untouched
    st.space.part_classes = {}
    assert MG.compute_margins(st, res).verdict == MG.PASS                 # back to what was confirmed
    st.space.default_class = "joints_inserts"
    assert MG.compute_margins(st, res).verdict == MG.UNCONFIRMED
    RQ.confirm(st.space.requirement_set, "test engineer", scope=st.space)  # confirmed again for the new classes
    s = MG.compute_margins(st, res)
    assert s.verdict == MG.PASS and all(r.structure_class == "joints_inserts" for r in s.rows if r.check == "yield")
    back = SpaceSettings.from_dict(json.loads(json.dumps(st.space.to_dict())))
    assert back.requirement_confirmed()
    back.part_classes = {"Block": "metallic"}
    assert not back.requirement_confirmed()
    rs = RQ.from_json(RQ.to_json(st.space.requirement_set))               # an imported set is never confirmed
    assert not rs.is_confirmed() and rs.scope_digest == ""


def test_the_template_names_no_edition_of_the_standard():
    rs = RQ.ecss_32_10c_template()
    assert RQ.TEMPLATE_REVISION == "" and not rs.revision and all(not f.revision for f in rs.factors) and all(not r.revision for r in rs.rules)
    assert not any(r.provenance_ok for r in rs.rules) and not any(f.provenance_ok for f in rs.factors)

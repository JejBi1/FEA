"""F14 S1: the lazy MCP group "space" (fea_space_requirement_set, fea_space_allowables_set, fea_space_margins)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

import verification.cases  # noqa: E402, F401  (registers the cases; space_cases imports from it, so cases comes first)
from fea_core.mcp import groups  # noqa: E402
from fea_core.mcp.server import build_server  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402
from verification import space_cases as SC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _factor_rows(approach: str, cls: str = "metallic") -> list[dict]:
    return [{"key": k, "structure_class": cls, "approach": approach, "value": SC.test_factor(k, cls, approach), "document": "ECSS-E-ST-32-10C",
             "revision": "Rev.2 Corr.1", "clause": "test clause"} for k in SC.BASE_FACTORS]


def _rules(thermal: str = "same_as_mechanical") -> list[dict]:
    return [{"check": c, "structure_class": "metallic", "allowable": a, "stress_measure": "von_mises", "factors": SC.TEST_RULES[("metallic", c)],
             "thermal_handling": thermal, "revision": "Rev.2 Corr.1", "clause": "test clause"} for c, a in (("yield", "Fy"), ("ultimate", "Fu"))]


def _session(tmp_path) -> Session:
    sess = Session(tmp_path / "mcp")
    mesh = SC.box_tet_mesh()
    sess.study = SC.block_study()
    sess.results = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA, spike=3.0)])
    return sess


def _run(sess: Session, steps):
    """steps = [(tool, args, expect_error)] → list of parsed answers (dict) / error texts."""
    out: list = []

    async def main():
        srv = build_server(sess, groups_spec="space")
        async with Client(srv, mode="legacy") as c:
            for name, args, expect_err in steps:
                r = await c.call_tool(name, args)
                text = r.content[0].text
                assert bool(r.is_error) == expect_err, (name, text)
                out.append(text if expect_err else json.loads(text))
    anyio.run(main)
    return out


def test_group_is_lazy_and_listed(tmp_path):
    async def main():
        srv = build_server(Session(tmp_path / "b"), groups_spec="core")
        async with Client(srv, mode="legacy") as c:
            assert not any(t.name.startswith("fea_space_") for t in (await c.list_tools()).tools)
            r = json.loads((await c.call_tool("fea_tools_enable", {"groups": "space"})).content[0].text)
            assert {"fea_space_requirement_set", "fea_space_allowables_set", "fea_space_margins"} <= set(r["added"])
    anyio.run(main)
    assert "space" in groups.GROUPS
    code = ("import sys; from fea_core.mcp.server import build_server; build_server(groups_spec='core'); "
            "print('LOADED', 'fea_core.mcp.tools_space' in sys.modules, 'fea_core.space.margins' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip().endswith("LOADED False False"), r.stdout


def test_workflow(tmp_path):
    sess = _session(tmp_path)
    ap = "protoflight"
    steps = [
        ("fea_space_requirement_set", {"action": "get"}, False),                                           # 0: nothing yet
        ("fea_space_requirement_set", {"action": "template"}, False),                                      # 1: the EMPTY template
        ("fea_space_requirement_set", {"action": "template"}, True),                                       # 2: not over an existing one
        ("fea_space_margins", {}, False),                                                                  # 3: no confirmed set
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, True),       # 4: empty template cannot be confirmed
        ("fea_space_requirement_set", {"action": "set", "data": {"name": "test", "verification_approach": ap, "factors": _factor_rows(ap),
                                                                   "rules": _rules()}}, False),            # 5
        ("fea_space_requirement_set", {"action": "confirm"}, True),                                        # 6: a name is required
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),      # 7
        ("fea_space_allowables_set", {"material": "TestAlloy", "basis": "B", "Fy": SC.TEST_FY, "Fu": SC.TEST_FU, "source": "test sheet"}, False),   # 8
        ("fea_space_margins", {}, False),                                                                  # 9: allowables not confirmed yet
        ("fea_space_allowables_set", {"material": "TestAlloy", "confirmed_by": "Test Engineer"}, False),   # 10
        ("fea_space_margins", {"top": 5}, False),                                                          # 11: PASS (singularity separated)
        ("fea_space_margins", {"justify": [{"part": "Block", "xyz": [0, 0, 0], "note": "accepted: rigid clamp of the test"}],
                               "author": "Test Engineer"}, False),                                         # 12: sign-off ready
        ("fea_space_requirement_set", {"action": "set", "data": {"factors": [dict(_factor_rows(ap)[0], value=1.5)]}}, False),   # 13: edit
        ("fea_space_margins", {}, False),                                                                  # 14: confirmation withdrawn
        ("fea_space_allowables_set", {"material": "Nope", "Fy": 1.0}, True),                               # 15
        ("fea_space_margins", {"part_classes": {"Block": "buckling"}}, True),                              # 16: not a part class
    ]
    o = _run(sess, steps)
    assert o[0]["requirement_set"] is None
    rs = o[1]["requirement_set"]
    assert rs["factor_rows"] == {"total": 84, "filled": 0} and not rs["is_confirmed"] and "empty template" in o[1]["notes"][0]
    assert o[3]["verdict"] == "UNCONFIRMED_REQUIREMENTS" and not o[3]["signoff_ready"]
    assert "approach" in o[4] or "rule" in o[4]
    assert o[5]["requirement_set"]["rules_status"][0]["complete"] is True and o[5]["requirement_set"]["factor_rows"]["filled"] == 7
    assert "confirmed_by" in o[6]
    assert o[7]["requirement_set"]["is_confirmed"] is True and o[7]["requirement_set"]["confirmed_by"] == "Test Engineer"
    assert o[8]["allowables"]["confirmed"] is False and o[8]["is_verified"] is False
    assert o[9]["verdict"] == "UNCONFIRMED_REQUIREMENTS" and any("allowables" in r for r in o[9]["reasons"])
    assert o[10]["is_confirmed"] and o[10]["is_verified"]
    assert o[11]["verdict"] == "PASS" and not o[11]["signoff_ready"] and o[11]["singularities"] and len(o[11]["lowest"]) <= 5
    j = SC.design_factor("metallic", "yield", ap)
    yld = [r for r in o[11]["lowest"] if r["check"] == "yield"][0]
    assert yld["mos"] == pytest.approx(SC.TEST_FY / (j * SC.TEST_SIGMA) - 1.0, rel=1e-4)             # the answer is rounded to 5 digits
    assert o[12]["signoff_ready"] is True and o[12]["singularities"][0]["justified"] is True
    assert any("withdrawn" in n for n in o[13]["notes"]) and o[13]["requirement_set"]["is_confirmed"] is False
    assert o[14]["verdict"] == "UNCONFIRMED_REQUIREMENTS"
    assert "Unknown structure class" in o[16]
    # the settings are in the study (saved with the project)
    assert sess.study.space is not None and sess.study.space.justifications[0].author == "Test Engineer"


def test_import_export_and_bounds(tmp_path):
    sess = _session(tmp_path)
    ap = "analysis_only"
    exp = tmp_path / "rs.json"
    csv_path = tmp_path / "factors.csv"
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    o = _run(sess, [
        ("fea_space_requirement_set", {"action": "template"}, False),
        ("fea_space_requirement_set", {"action": "set", "data": {"verification_approach": ap, "factors": _factor_rows(ap), "rules": _rules()}}, False),
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "A"}, False),
        ("fea_space_requirement_set", {"action": "export", "path": str(exp)}, False),
        ("fea_space_requirement_set", {"action": "export", "path": str(csv_path)}, False),
        ("fea_space_requirement_set", {"action": "import", "path": str(exp)}, False),                       # an import is NOT confirmed
        ("fea_space_requirement_set", {"action": "import", "path": str(bad)}, True),
        ("fea_space_requirement_set", {"action": "import", "path": str(tmp_path / "missing.json")}, True),
        ("fea_space_requirement_set", {"action": "import", "path": str(csv_path)}, False),
        ("fea_space_requirement_set", {"action": "nonsense"}, True),
        ("fea_space_requirement_set", {"action": "set", "data": {"bogus": 1}}, True),
        ("fea_space_requirement_set", {"action": "set", "data": {"factors": [{"key": "FOSY", "value": "abc"}]}}, True),
    ])
    assert exp.is_file() and csv_path.read_text(encoding="utf-8").startswith("key,structure_class")
    assert o[5]["requirement_set"]["is_confirmed"] is False and o[5]["requirement_set"]["confirmed"] is False
    assert o[8]["requirement_set"]["factor_rows"]["filled"] == 7
    assert "Unknown keys" in o[10]
    from fea_core.space.requirements import from_json
    assert from_json(exp.read_text(encoding="utf-8")).factors[0].key in ("FOSY", "FOSU", "KQ", "KP", "KM", "KLD", "KMP")


def test_an_agent_cannot_set_and_confirm_in_one_call(tmp_path):
    sess = _session(tmp_path)
    ap = "protoflight"
    data = {"name": "t", "verification_approach": ap, "factors": _factor_rows(ap), "rules": _rules()}
    o = _run(sess, [
        ("fea_space_requirement_set", {"action": "template"}, False),
        ("fea_space_requirement_set", {"action": "set", "data": data, "confirmed_by": "Agent"}, True),         # 1: values + confirmation
        ("fea_space_requirement_set", {"action": "confirm", "data": data, "confirmed_by": "Agent"}, True),     # 2
        ("fea_space_requirement_set", {"action": "get", "confirmed_by": "Agent"}, True),                       # 3
        ("fea_space_requirement_set", {"action": "set", "data": data}, False),                                 # 4
        ("fea_space_allowables_set", {"material": "TestAlloy", "basis": "B", "Fy": 276.0, "Fu": 310.0, "source": "s",
                                      "confirmed_by": "Agent"}, True),                                         # 5
        ("fea_space_allowables_set", {"material": "TestAlloy", "clear": True, "confirmed_by": "Agent"}, True),  # 6
        ("fea_space_allowables_set", {"material": "TestAlloy", "basis": "B", "Fy": 276.0, "Fu": 310.0, "source": "s"}, False),   # 7
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),         # 8
        ("fea_space_allowables_set", {"material": "TestAlloy", "confirmed_by": "Test Engineer"}, False),       # 9
        ("fea_space_margins", {}, False),                                                                       # 10
    ])
    for i in (1, 2, 3, 5, 6):
        assert "confirmed_by" in o[i] and "call of its own" in o[i], (i, o[i])
    assert not sess.study.space.requirement_set.confirmed or sess.study.space.requirement_set.confirmed_via == "mcp"
    assert o[8]["requirement_set"]["confirmed_via"] == "mcp"
    assert o[9]["allowables"]["confirmed_via"] == "mcp"
    assert o[10]["requirement_set"]["confirmed_via"] == "mcp"


def test_changing_part_classes_after_the_confirmation_gives_unconfirmed(tmp_path):
    sess = _session(tmp_path)
    ap = "protoflight"
    data = {"name": "t", "verification_approach": ap, "factors": _factor_rows(ap), "rules": _rules()}
    o = _run(sess, [
        ("fea_space_requirement_set", {"action": "template"}, False),
        ("fea_space_requirement_set", {"action": "set", "data": data}, False),
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),
        ("fea_space_allowables_set", {"material": "TestAlloy", "basis": "B", "Fy": 276.0, "Fu": 310.0, "source": "s"}, False),
        ("fea_space_allowables_set", {"material": "TestAlloy", "confirmed_by": "Test Engineer"}, False),
        ("fea_space_margins", {}, False),                                                          # 5: PASS
        ("fea_space_margins", {"default_class": "joints_inserts"}, False),                          # 6: the class changed
        ("fea_space_requirement_set", {"action": "get"}, False),                                   # 7
    ])
    assert o[5]["verdict"] == "PASS" and o[5]["requirement_set"]["confirmed"] is True
    assert o[6]["verdict"] == "UNCONFIRMED_REQUIREMENTS" and o[6]["requirement_set"]["confirmed"] is False
    assert o[7]["requirement_set"]["is_confirmed"] is False


def test_a_rule_can_be_declared_not_applicable_only_with_provenance(tmp_path):
    sess = _session(tmp_path)
    ap = "protoflight"
    data = {"name": "t", "verification_approach": ap, "factors": _factor_rows(ap), "rules": _rules()}
    na = [{"check": c, "structure_class": "composite", "not_applicable": True} for c in ("yield", "ultimate")]
    o = _run(sess, [
        ("fea_space_requirement_set", {"action": "template"}, False),
        ("fea_space_requirement_set", {"action": "set", "data": data}, False),
        ("fea_space_requirement_set", {"action": "set", "data": {"rules": na}}, False),                       # 2: no provenance yet
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, True),          # 3: refused
        ("fea_space_requirement_set", {"action": "set", "data": {"rules": [dict(r, document="d", revision="r", clause="c") for r in na]}}, False),
        ("fea_space_requirement_set", {"action": "confirm", "confirmed_by": "Test Engineer"}, False),         # 5
    ])
    assert "not applicable" in o[3]
    assert o[5]["requirement_set"]["is_confirmed"] is True
    assert all(r["not_applicable"] for r in o[5]["requirement_set"]["rules"] if r["structure_class"] == "composite")

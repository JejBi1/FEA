"""F14 S6: the reference small-satellite benchmark (`verification.cubesat_benchmark`) end to end: statics, the 3-axis launch analysis, the model checks, a synthetic correlation, the
direct (rslab) modal path against CalculiX, and the Structural Analysis Report regenerated from the saved project. Slow (minutes: meshing and solver runs)."""
from __future__ import annotations

import re

import pytest

pytest.importorskip("docx")

pytestmark = pytest.mark.slow

FORBIDDEN = re.compile(r"(?i:abaqus)|(?<![A-Za-z0-9_])X[1-5](?![A-Za-z0-9_])")        # the name in any case, the case ids X1-X5 only as written (a sensor axis "x1" is text)
PATH_LIKE = re.compile(r"[A-Za-z]:[\\/]|\\\\[A-Za-z]|/ho" r"me/|/Us" r"ers/|App" r"Data")


def _text(path) -> str:
    from docx import Document
    d = Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def test_the_benchmark_runs_end_to_end_and_the_report_is_honest(tmp_path):
    import verification.cases  # noqa: F401 - the registry first
    from fea_core import config
    from verification import cubesat_benchmark as CB
    if not config.find_ccx().exists():
        pytest.skip("CalculiX is not installed")
    try:
        import rslab  # noqa: F401
    except ImportError:
        pytest.skip("rslab (the direct solver) is not installed")
    facts = CB.run_benchmark(tmp_path, threads=4, log=lambda s: None)
    assert facts["mesh"]["nodes"] > 10000
    x = facts["modal_crosscheck"]
    assert x["modes_compared"] >= 10 and x["max_relative_difference"] < 1e-3, x             # the direct path against CalculiX on the same mesh (measured 4e-6)
    rep = facts["report"]
    assert rep["verdict"] in ("NOT_SIGNED_OFF", "FAIL") and rep["verdict"] != "PASS"      # the convergence check is left out and the singular places are not justified
    text = _text(tmp_path / rep["file"])
    assert not FORBIDDEN.search(text) and not PATH_LIKE.search(text) and "ILLUSTRATIVE" in text
    assert "Setup digest SHA-256" in text and "rslab (direct solver)" in text

    # the report regenerated from the saved project: the same text, never a PASS, what is not saved is said to be not run
    from fea_core.project import load_project
    from fea_core.report import ReportOptions, session_from
    from fea_core.report.space_sar import build_sar
    out = []
    for k in (1, 2):
        study, step, results = load_project(tmp_path / "benchmark.fea", tmp_path / f"re{k}")
        sess = session_from(study, step, results, workdir=tmp_path / f"re{k}")
        r = build_sar(sess, tmp_path / f"again{k}.docx", ReportOptions(ecss_sar=True), generated="2026-01-01 00:00")
        out.append((r, _text(r["path"])))
    assert out[0][1] == out[1][1] and out[0][0]["traceability"] == out[1][0]["traceability"]
    r = out[0][0]
    by = {c["key"]: c["status"] for c in r["chapters"]}
    assert r["verdict"] != "PASS" and by["checks"] == "NOT_RUN" and by["loads"] in ("NOT_RUN", "OUTDATED")
    assert r["traceability"]["geometry_sha256"] and not FORBIDDEN.search(out[0][1]) and not PATH_LIKE.search(out[0][1])

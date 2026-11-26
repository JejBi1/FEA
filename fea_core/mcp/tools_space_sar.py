"""MCP group "space": the Structural Analysis Report (ECSS) as a Word file (F14 stage S6, `fea_core.report.space_sar`).

The report is written from the state of the session (the solved results, the kept 3-axis result, the kept model quality checks, the stored correlation and S4 results of the
study). It never confirms or enters anything: a chapter that is not confirmed, not current or not run is reported as such and keeps the overall verdict from being PASS.
"""
from __future__ import annotations

import time

from .common import out
from .session import FeaError, Session, user_path


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_space_sar_report(output_path: str | None = None, title: str | None = None, author: str = "", company: str = "", subject: str = "", language: str = "en") -> str:
        """Word (.docx) Structural Analysis Report (ECSS): traceability (SHA-256 of study, geometry, setup; versions; mesh), model and mass, requirement set, model checks, loads and envelopes, margins, interfaces / bolts / PCB (when built), correlation, open items. Verdict PASS only when every included chapter is current and PASS, else NOT_SIGNED_OFF. No local path or user name is written. Returns path, verdict, chapter statuses."""
        from ..report import ReportOptions
        from ..report.space_sar import build_sar
        if language not in ("cs", "en"):
            raise FeaError("language must be 'cs' or 'en'")
        with sess.use(op="space_sar"):
            sess.require_study()
            p = user_path(output_path) if output_path else (sess.workdir or sess.base_dir) / f"SAR_{time.strftime('%Y%m%d_%H%M%S')}.docx"
            r = build_sar(sess, p, ReportOptions(title=title, author=author, company=company, subject=subject, language=language, ecss_sar=True), on_log=sess.add_log)
            r["with_results"] = sess.results is not None
            return out(r)

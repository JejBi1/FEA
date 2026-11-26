"""The limits of a requirement set as a short read-only HTML summary for the confirmation in the Space dialog (F14 stage S5, fix N5): every `corr_*` limit with its value, unit and source, and
EVERY criterion that is declared "not applicable" (limits and margin rules) with the interface it was entered through - so that a flag that an agent set through MCP is visible to the user
at the moment they confirm the set. All text of the requirement set (it may come from an imported file or an agent) is escaped."""
from __future__ import annotations

from html import escape
from typing import Any

AMBER = "#b36b00"
_VIA = {"mcp": "entered through the MCP interface (an agent) - check it", "gui": "entered in the GUI", "import": "from an imported file - check it"}


def _plain(x: Any, n: int = 60) -> str:
    return escape("".join(ch if ch.isprintable() else " " for ch in str(x if x is not None else "")).strip()[:n])


def limits_html(rs: Any, space: Any = None) -> str:
    """HTML of the limits and of the criteria declared not applicable of the requirement set `rs` ("" when there is no set / nothing to show)."""
    if rs is None:
        return ""
    from fea_core.space.correlation import LIMIT_INFO
    limits = [x for x in rs.limits if x.key in LIMIT_INFO or x.not_applicable]
    rules_na = [r for r in rs.rules if getattr(r, "not_applicable", False)]
    if not limits and not rules_na:
        return ""
    rows: list[str] = []
    for x in limits:
        na = bool(x.not_applicable)
        val = "<b>NOT APPLICABLE</b>" if na else ("not entered" if x.value is None else _plain(f"{x.value:g}"))
        src = " ".join(t for t in (_plain(x.document), _plain(x.revision), _plain(x.clause)) if t) or "no source"
        note = _VIA.get(str(getattr(x, "set_via", "") or ""), "interface unknown") if na else ""
        style = f" style='color:{AMBER}'" if na else ""
        rows.append(f"<tr{style}><td>{_plain(x.key)}</td><td>{val}</td><td>{_plain(x.unit, 12)}</td><td>{src}</td><td>{escape(note)}</td></tr>")
    for r in rules_na:
        src = " ".join(t for t in (_plain(r.document), _plain(r.revision), _plain(r.clause)) if t) or "no source"
        rows.append(f"<tr style='color:{AMBER}'><td>margin rule {_plain(r.check)} / {_plain(r.structure_class)}</td><td><b>NOT APPLICABLE</b></td><td></td><td>{src}</td><td></td></tr>")
    n_na = sum(1 for x in limits if x.not_applicable) + len(rules_na)
    head = "<b>Limits of the set (read-only)</b>" + (f" - <span style='color:{AMBER}'><b>{n_na} criterion/criteria declared NOT APPLICABLE</b> (not judged)</span>" if n_na else "")
    return head + "<table cellspacing='4'><tr><th align='left'>Limit</th><th align='left'>Value</th><th align='left'>Unit</th><th align='left'>Source</th><th align='left'>Note</th></tr>" + "".join(rows) + "</table>"

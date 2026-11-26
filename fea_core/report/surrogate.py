"""Report section "Surrogate model" (fea_core.surrogate, F15-S4): parameters and ranges of the analysed points, the validation on held-out
points, the guaranteed level and the cross-validated coverage of the band, whether the study changed since the fit, and the limitations
(fixed text)."""
from __future__ import annotations

from ..surrogate import LIMITATIONS
from .param import param_label
from .word import num

MAX_VALIDATION_ROWS = 20


def has_surrogate(sess) -> bool:
    sm = getattr(sess, "surrogate", None)
    return sm is not None and bool(getattr(sm, "members", None))


def _pct(v) -> str:
    return "–" if v is None else f"{100.0 * float(v):.0f} %"


def surrogate_section(D, sess) -> None:
    sm = sess.surrogate
    stale = sm.check_study(getattr(sess, "study", None), getattr(sess, "step_path", None))
    d = sm.describe()
    lvl = d["guaranteed_level"]
    failed = f" {d['n_failed']} further analyses failed and are not used." if d["n_failed"] else ""
    from ..surrogate.fit import holdout_minimum
    hold = holdout_minimum(sm.level, sm.min_holdout)                      # the rule of THIS fit (9 at 90 %, fewer only when the caller asked for it)
    designs = sorted({m.get("design", "") for m in d["members"]} - {"", "space-filling"})
    D.p(f"A surrogate model ({'Gaussian process' if sm.backend == 'gp' else 'radial-basis interpolation'}) was fitted to the {d['n_points']} "
        f"converged design points of the parametric study.{failed} It answers the key results for parameter values inside the analysed "
        f"region (the range and the convex hull of the converged points, no holes) in milliseconds, with a {lvl:.0%} band. "
        f"{int(round(100 * sm.validation_fraction))} % of the points (at least {hold}, chosen at random with the seed {sm.seed}) were held out: "
        "they are never used for the fit, they calibrate the band (split-conformal prediction), so the band does not depend on the "
        f"assumptions of the regression. The coverage of {lvl:.0%} is an average over the analysed region; near its edges it can be lower.")
    if "lines" in designs:
        D.p("The design points vary one parameter at a time: the model answers only on those lines; the interaction of the parameters was never analysed.",
            italic=True, size=9)
    if "levels" in designs:
        D.p("At least one parameter was analysed at fewer than 6 levels (a grid): the model answers at those levels only, not between them.",
            italic=True, size=9)
    if lvl < sm.level:
        D.p(f"Too few held-out points for the requested {sm.level:.0%}: the band is guaranteed for {lvl:.0%}.", italic=True, size=9)
    if stale:
        D.p("WARNING – " + stale, italic=True, size=9)
    rows = []
    for p in d["parameters"]:
        rows.append([param_label(p["key"]), ("continuous" if p["kind"] == "continuous" else "categorical"),
                     f"{num(p['lo'], 5)} … {num(p['hi'], 5)}" if p["kind"] == "continuous" else ", ".join(p["categories"])])
    D.table(["Parameter", "Kind", "Range / values"], rows, caption="Surrogate model – parameters", font=9)
    from ..sweep import kpi_columns
    label = dict(kpi_columns())
    mrows = []
    for m in d["members"]:
        mrows.append([", ".join(m["category"]) or "–", label.get(m["kpi"], m["kpi"]), f"{m['n_train']} / {m['n_validation']}", num(m["r2"], 4),
                      _pct(m["median_rel_err"]), num(m["q"], 3), _pct(m["guaranteed"]), _pct(m["coverage_cv"]), "yes" if m["reliable"] else "NO"])
    D.table(["Category", "Result (KPI)", "Fit / held-out", "R² (held-out)", "Median error", "Band factor q", "Band level (guaranteed)",
             "Coverage (cross-validated)", "Reliable"], mrows, caption="Surrogate model – validation on the held-out points", font=8)
    for m in d["members"]:
        if m["reasons"]:
            D.p(f"{label.get(m['kpi'], m['kpi'])}: " + "; ".join(m["reasons"]) + ".", italic=True, size=9)
    first = d["members"][0]
    tab = sm.validation_table(first["kpi"], tuple(first["category"]))
    if tab:
        keys = [p["key"] for p in d["parameters"]]
        vrows = [[*(num(r["params"].get(k), 4) if isinstance(r["params"].get(k), (int, float)) else str(r["params"].get(k)) for k in keys),
                  num(r["analysed"], 4), num(r["predicted"], 4), f"{num(r['lower'], 4)} … {num(r['upper'], 4)}", "yes" if r["inside"] else "NO"]
                 for r in tab[:MAX_VALIDATION_ROWS]]
        inside = sum(r["inside"] for r in tab)
        D.table([*(param_label(k) for k in keys), "Analysed", "Predicted", f"{lvl:.0%} band", "Inside"], vrows,
                caption=f"Surrogate model – held-out points, {label.get(first['kpi'], first['kpi'])} ({inside} of {len(tab)} inside the band)", font=8)
    D.p(LIMITATIONS, italic=True, size=9)
    for w in d["warnings"][:5]:
        D.p(w, italic=True, size=9)

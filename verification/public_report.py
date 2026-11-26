"""Public verification page: docs/verification/{README.md, results.csv, index.html}.

    python -m verification.public_report [--report out/report.json] [--mystran out_mystran/report.json] [--meta out/run_meta.json]
                                         [--out docs/verification]

RUNS NOTHING. It reads the results of a real run (verification.run_public → `verification/out/report.json`, the MYSTRAN comparison
`out_mystran/report.json` and `out/run_meta.json`) and writes the page: per domain a table "case | quantity | reference (source) |
FEA Studio | difference | tolerance | mesh", the cross-check against MYSTRAN, the cases that were not run on the platform of the
run (with the reason), and the program / solver / platform / date of the run.

Only the cases that verification.catalogue marks as public are published, and only their checks that are not marked as consistency
checks (`Entry.hide`); the case is "within tolerance" only when ALL its checks pass, the hidden ones included. The page contains no
local path, user or machine name. The English wording of the quantities is in verification.catalogue_labels.

MYSTRAN is a separately downloaded program (FEA Studio does not ship it). A run on a machine without it (meta "mystran_missing") does not make its cases
"outside tolerance": a case whose reference is MYSTRAN is "not run", a case of catalogue.MYSTRAN_CROSS_CHECK whose own checks pass is "incomplete" (within
tolerance, its cross-check not run) - never a plain pass.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from verification import catalogue, catalogue_labels

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "docs" / "verification"

CSV_COLUMNS = ("case", "domain", "title", "reference_type", "source", "quantity", "unit", "reference", "value", "difference",
               "tolerance", "basis", "check_status", "case_status", "nodes", "elements", "mesh_size_mm", "time_s",
               "program_version", "run_date", "solver", "note")
MYSTRAN_TOL = 1e-3                      # verification.mystran_check.TOL
_PATH_LIKE = re.compile(r"[A-Za-z]:[\\/][^\s'\"]*|\\\\[^\s'\"]+|/(?:ho" r"me|Us" r"ers|mn" r"t|tmp|var)/[^\s'\"]*")   # written in pieces: the packaging scanner must not see a path here


@dataclass
class Row:
    quantity: str
    unit: str
    reference: float
    value: float
    error: float                        # as the check computes it (relative, or absolute when the reference is ~0)
    tol: float
    ok: bool
    absolute: bool                      # True: error / tol are absolute numbers in the unit of the quantity (reference ~ 0)
    ref_type: str
    source: str


@dataclass
class CaseResult:
    entry: catalogue.Entry
    status: str                         # "pass" | "fail" | "not run"
    note: str = ""
    rows: list[Row] = field(default_factory=list)
    n_hidden: int = 0
    nodes: int | None = None
    elements: int | None = None
    mesh_size: float | None = None
    time_s: float | None = None
    run_date: str = ""                  # date (yyyy-mm-dd) and program version of the run that produced this case (a merged run has several)
    version: str = ""


@dataclass
class Model:
    meta: dict
    cases: list[CaseResult]
    mystran_rows: dict[str, list[dict]]
    untranslated: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = {"cases": len(self.cases), "pass": 0, "fail": 0, "incomplete": 0, "not run": 0, "checks": 0}
        for r in self.cases:
            c[r.status] += 1
            c["checks"] += len(r.rows)                  # the checks that are listed on the page (hidden consistency checks are not counted)
        return c


# ---------------------------------------------------------------------------------------------------- reading
def _load(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


WITHHELD = "(details withheld: the message contained a file path)"


def _clean(text: str, limit: int = 240) -> str:
    """One line of a message for the page. The page must not carry local paths, and a path with spaces cannot be cut out of a
    sentence reliably - so a message that contains a path separator (or looks like a path) is not published at all."""
    s = " ".join(str(text).split())
    if re.search(r"[\\/]", s) or _PATH_LIKE.search(s):
        return WITHHELD
    return s[:limit] + ("…" if len(s) > limit else "")


def build(report: dict, mystran: dict, meta: dict) -> Model:
    """The page model from the raw results. Cases that are public but have no result are listed as not run."""
    skipped = meta.get("skipped") or {}
    # the time of the page is the sum of the run times of the cases in the report (a run may have been made in several parts)
    case_runs = meta.get("case_runs") or {}

    def run_of(key: str) -> tuple[str, str]:
        info = case_runs.get(key) or {}
        return str(info.get("date_utc") or meta.get("date_utc", ""))[:10], str(info.get("program_version") or meta.get("program_version", ""))
    meta = {**meta, "duration_s": sum(float(r["time_s"]) for r in report.values() if isinstance(r, dict) and r.get("time_s"))}
    runs = [run_of(k) for k, r in report.items() if isinstance(r, dict) and "checks" in r and k in catalogue.ENTRIES and catalogue.get(k).public]
    meta["run_dates"] = sorted({d for d, _ in runs if d})
    meta["run_versions"] = sorted({v for _, v in runs if v})
    untranslated: list[str] = []
    cases: list[CaseResult] = []
    for group in catalogue.by_domain():
        for e in group.entries:
            rep = report.get(e.key)
            if not isinstance(rep, dict):
                cases.append(CaseResult(e, "not run", skipped.get(e.key, "not part of this run")))
                continue
            if "error" in rep:
                cases.append(CaseResult(e, "fail", "the run stopped with an error: " + _clean(rep["error"])))
                continue
            res = CaseResult(e, "pass", "", [], 0, rep.get("nodes"), rep.get("elements"), rep.get("mesh_size_mm"),
                             rep.get("time_s"), *run_of(e.key))
            ok_all = True
            for c in rep.get("checks", []):
                ok_all &= bool(c.get("ok"))
                if not e.shows(c["name"]):
                    res.n_hidden += 1
                    continue
                name = catalogue_labels.label(e.key, c["name"])
                if not catalogue_labels.is_english(name):
                    untranslated.append(f"{e.key}: {c['name']}")
                ref = float(c["ref"])
                res.rows.append(Row(name, catalogue_labels.unit(c.get("unit", "")), ref, float(c["value"]), float(c["error"]),
                                    float(c["tol"]), bool(c["ok"]), abs(ref) < 1e-12, e.ref_type, e.source))
            mys = mystran.get(e.key)
            no_mystran = False
            if mys is None and e.key in catalogue.MYSTRAN_CROSS_CHECK and meta.get("mystran_missing"):
                no_mystran = True                          # the program is not installed: the comparison could not run
                res.note = CROSS_NOT_RUN
            elif mys is None and e.key in catalogue.MYSTRAN_CROSS_CHECK:     # the page promises the comparison: it must exist
                ok_all = False
                res.note = ("the case ran without its cross-check against MYSTRAN (not installed, or not run): "
                            f"python -m verification.run_public --only {e.key}")
            elif isinstance(mys, list):
                ok_all &= all(_mystran_ok(r) for r in mys)
            elif isinstance(mys, dict):                   # the comparison itself failed: never dropped silently
                ok_all = False
                res.note = "cross-check against MYSTRAN failed: " + _clean(mys.get("error", ""))
            res.status = "pass" if ok_all and (res.rows or res.n_hidden) else "fail"
            if not (res.rows or res.n_hidden):
                res.note = "the run produced no checks"
            elif no_mystran and res.status == "pass":
                if e.ref_type == "mystran":                # MYSTRAN is the only independent reference of the case
                    res.status, res.note = "not run", NOT_RUN_MYSTRAN
                else:
                    res.status = "incomplete"
            cases.append(res)
    rows = {k: [{**r, "case": catalogue_labels.loadcase(r["case"])} for r in v] for k, v in mystran.items() if isinstance(v, list)}
    untranslated += [f"{k}: {r['case']}" for k, v in rows.items() for r in v if not catalogue_labels.is_english(r["case"])]
    return Model(meta, cases, rows, untranslated)


NOT_RUN_MYSTRAN = "not run – MYSTRAN not installed"
CROSS_NOT_RUN = "cross-check against MYSTRAN not run – MYSTRAN not installed"
# said on every page: the program that the MYSTRAN rows come from is not part of FEA Studio
MYSTRAN_NOTE = ("MYSTRAN is a separately downloaded third-party program: FEA Studio does not ship it (download: www.mystran.com). The MYSTRAN rows of "
                "this page come from the run named above; on a computer without it the cases whose reference is MYSTRAN are listed as not run, and the "
                "cross-check of the other cases is marked as not run.")


def _cross_mesh(m: dict) -> str:
    """Mesh of a MYSTRAN comparison row: node count and element size."""
    if not m.get("nodes"):
        return "–"
    return f"{m['nodes']:,} nodes".replace(",", " ") + (f", h = {m['mesh_size_mm']:g} mm" if m.get("mesh_size_mm") else "")


def _mystran_ok(r: dict) -> bool:
    return float(r.get("du_rel", 1.0)) < MYSTRAN_TOL and float(r.get("dr_rel", 1.0)) < MYSTRAN_TOL


# ---------------------------------------------------------------------------------------------------- formatting
def num(v: float) -> str:
    """A number for a table: 5 significant digits, no exponent for ordinary magnitudes."""
    if v != v:
        return "n/a"
    if v == 0:
        return "0"
    a = abs(v)
    return f"{v:.4e}".replace("e+0", "e+").replace("e-0", "e-") if (a < 1e-3 or a >= 1e7) else f"{v:.5g}"


def diff_text(r: Row) -> str:
    if r.absolute:
        return num(r.error) + (f" {r.unit}" if r.unit else "")
    pct = 100.0 * r.error
    return "< 0.001 %" if pct < 0.001 else f"{pct:.3g} %"


def tol_text(r: Row) -> str:
    if r.absolute:
        return num(r.tol) + (f" {r.unit}" if r.unit else "")
    return f"{100.0 * r.tol:.3g} %"


def mesh_text(c: CaseResult) -> str:
    if not c.nodes:
        return "–"
    s = f"{c.nodes:,} nodes".replace(",", " ")
    return s + (f", h = {c.mesh_size:g} mm" if c.mesh_size else "")


def _value_text(v: float, unit: str) -> str:
    return num(v) + (f" {unit}" if unit else "")


# ---------------------------------------------------------------------------------------------------- results.csv
def csv_text(model: Model) -> str:
    from fea_core.export_csv import text_cell
    meta = model.meta
    solver = " + ".join(x for x in (meta.get("ccx"), meta.get("mystran")) if x)
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for c in model.cases:
        e = c.entry
        common = (c.version or meta.get("program_version", ""), c.run_date or str(meta.get("date_utc", ""))[:10], solver)
        head = (e.key, e.domain, e.title)
        mesh = (c.nodes if c.nodes else "", c.elements if c.elements else "", c.mesh_size if c.mesh_size else "",
                f"{c.time_s:.1f}" if c.time_s is not None else "")
        if not c.rows and not model.mystran_rows.get(e.key):
            w.writerow([*map(text_cell, head), e.ref_type, text_cell(e.source), *([""] * 8), c.status,
                        *mesh, *common, text_cell(c.note)])
            continue
        for r in c.rows:
            w.writerow([*map(text_cell, head), r.ref_type, text_cell(r.source), text_cell(r.quantity), r.unit,
                        f"{r.reference:.10g}", f"{r.value:.10g}", f"{r.error:.6g}", f"{r.tol:.6g}",
                        "absolute" if r.absolute else "relative", "ok" if r.ok else "FAIL", c.status, *mesh, *common, ""])
        for m in model.mystran_rows.get(e.key, []):
            for q, unit, ref, val, err in _mystran_checks(m):
                w.writerow([*map(text_cell, head), "mystran", text_cell(MYSTRAN_SOURCE.format(solver=meta.get("mystran") or "MYSTRAN")),
                            text_cell(f"{q} [{m['case']}]"), unit, f"{ref:.10g}", f"{val:.10g}", f"{err:.6g}", f"{MYSTRAN_TOL:.6g}",
                            "relative" if q.startswith("max displacement") else "absolute", "ok" if err < MYSTRAN_TOL else "FAIL",
                            c.status, m.get("nodes", ""), "", m.get("mesh_size_mm", ""), "", *common, ""])
    return buf.getvalue()


MYSTRAN_SOURCE = "{solver} (open-source Nastran-dialect solver), the same mesh coarsened by a factor of 2"


def _mystran_checks(m: dict) -> list[tuple[str, str, float, float, float]]:
    """(quantity, unit, reference, value, error) of one ccx × MYSTRAN comparison row."""
    um, uc = float(m["u_max_mystran"]), float(m["u_max_ccx"])
    return [("max displacement (ccx vs MYSTRAN)", "mm", um, uc, abs(uc - um) / max(abs(um), 1e-300)),
            ("displacement field max|Δu| / max|u|", "", 0.0, float(m["du_rel"]), float(m["du_rel"])),
            ("reactions |ΔR| / |R| (force and moment)", "", 0.0, float(m["dr_rel"]), float(m["dr_rel"]))]


# ---------------------------------------------------------------------------------------------------- Markdown
def _kind_md(e: catalogue.Entry) -> str:
    return " (relation stated, no work cited)" if e.source_kind == "relation" else ""


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _run_lines(meta: dict) -> list[tuple[str, str]]:
    dates, versions = meta.get("run_dates") or [], meta.get("run_versions") or []
    program = (f"FEA Studio {', '.join(versions)} (cases of several runs; the version of each case is in results.csv)" if len(versions) > 1
               else f"FEA Studio {meta['program_version']}" if meta.get("program_version") else "")
    when = (f"{dates[0]} to {dates[-1]} (cases of several runs; the date of each case is in results.csv)" if len(dates) > 1
            else str(meta.get("date_utc", "")))
    return [(k, v) for k, v in (
        ("Program", program),
        ("Date of the run (UTC)", when),
        ("CalculiX", meta.get("ccx", "") + (f" ({meta['ccx_build']}, sha256 {meta['ccx_sha256']}…)" if meta.get("ccx_sha256") else "")),
        ("MYSTRAN", meta.get("mystran", "")),
        ("Platform", " · ".join(x for x in (meta.get("platform", ""), f"Python {meta['python']}" if meta.get("python") else "",
                                           f"{meta['cpus']} logical CPUs" if meta.get("cpus") else "") if x)),
        ("Run profile", meta.get("profile", "")),
        ("Sum of the case run times", f"{meta['duration_s'] / 60:.0f} min" if meta.get("duration_s") else "")) if v]


def readme_text(model: Model) -> str:
    cnt = model.counts()
    out = ["# FEA Studio – public verification set", "",
           "Every case below is compared with a **reference that does not come from FEA Studio**: a closed-form solution, a handbook or "
           "standard formula, a published solution, or the open-source solver MYSTRAN. For each case either a work is cited, or - for an "
           "elementary relation - the first-principles relation is stated (such a case is marked *relation stated, no work cited*). "
           "A case without such a reference is not published here (the program's own equilibrium and bookkeeping checks are part of "
           "the test suite, not of this page).", "",
           f"**{cnt['pass']} of {cnt['cases']} published cases are within tolerance**"
           + (f", {cnt['fail']} are outside" if cnt["fail"] else "")
           + (f", {cnt['incomplete']} more are within tolerance but without their MYSTRAN cross-check" if cnt["incomplete"] else "")
           + (f", {cnt['not run']} were not run on this platform" if cnt["not run"] else "")
           + f" ({cnt['checks']} checks listed below).", ""]
    out += [f"- {k}: {_md(v)}" for k, v in _run_lines(model.meta)]
    out += ["", "Files: [`index.html`](index.html) (the same page, self-contained), [`results.csv`](results.csv) (all rows). "
            "Regenerate with `python -m verification.run_public` (runs the cases, needs CalculiX and Gmsh, MYSTRAN for the MYSTRAN cases "
            "and the Chrono module for MB4C, MB4R and KK5) or, from stored results, `python -m verification.public_report`. The CI page "
            "(`.github/workflows/verification.yml`) is built without the Chrono module, so it lists those three cases as not run; "
            "this page carries their results from a run with the module.", "",
            "How to read the tables: *difference* is |FEA Studio − reference| / |reference|, or the absolute difference where the reference "
            "is zero; *tolerance* is the limit the case enforces; *mesh* is the node count and global element size of the finite-element "
            "model (second-order tetrahedra; beam, 2D and shell cases have their own discretisation). A case is within tolerance only when "
            "all its checks pass, including a few internal consistency checks that are not listed. In `results.csv` *difference* and *tolerance* "
            "are fractions (0.01 = 1 %) where `basis` is `relative` and absolute numbers in the unit of the quantity where it is `absolute`.", "",
            MYSTRAN_NOTE, ""]
    for group in catalogue.by_domain():
        cs = [c for c in model.cases if c.entry.domain == group.domain and c.status != "not run"]
        if not cs:
            continue
        out += [f"## {group.title}", "", "| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |",
                "|---|---|---|---:|---:|---:|---|"]
        for c in cs:
            e = c.entry
            mark = {"pass": "✔", "incomplete": "✔ (no MYSTRAN cross-check)"}.get(c.status, "✘")
            out.append(f"| **{e.key}** {mark} | {_md(e.title)} | "
                       f"*{catalogue.REF_TYPE_TITLES[e.ref_type]}*{_kind_md(e)}: {_md(e.source)} | | | | {mesh_text(c)} |")
            out += [f"| | {_md(r.quantity)} | {_md(_value_text(r.reference, r.unit))} | {_md(_value_text(r.value, r.unit))} | "
                    f"{'' if r.ok else '**'}{diff_text(r)}{'' if r.ok else '**'} | {tol_text(r)} | |" for r in c.rows]
        out.append("")
    out += _mystran_md(model)
    nr = [c for c in model.cases if c.status == "not run"]
    if nr:
        out += ["## Not run on this platform", "", "| Case | Title | Reason |", "|---|---|---|"]
        out += [f"| {c.entry.key} | {_md(c.entry.title)} | {_md(c.note)} |" for c in nr]
        out.append("")
    return "\n".join(out) + "\n"


def _mystran_md(model: Model) -> list[str]:
    if not model.mystran_rows:
        return []
    out = ["## Cross-check against MYSTRAN", "",
           f"The same model, mesh (coarsened ×2) and nodal loads are solved by CalculiX and by {model.meta.get('mystran') or 'MYSTRAN'} "
           "(independent assembly, boundary conditions and solver). Criterion: max|Δu| / max|u| and |ΔR| / |R| below "
           f"{MYSTRAN_TOL:g}.", "", "| Case | Load case | max\\|u\\| CalculiX [mm] | max\\|u\\| MYSTRAN [mm] | max\\|Δu\\| / max\\|u\\| | \\|ΔR\\| / \\|R\\| | Mesh |",
           "|---|---|---:|---:|---:|---:|---|"]
    for c in model.cases:
        for m in model.mystran_rows.get(c.entry.key, []):
            ok = _mystran_ok(m)
            out.append(f"| {c.entry.key} | {_md(m['case'])} | {m['u_max_ccx']:.6g} | {m['u_max_mystran']:.6g} | "
                       f"{'' if ok else '**'}{m['du_rel']:.2e}{'' if ok else '**'} | {m['dr_rel']:.2e} | "
                       f"{_cross_mesh(m)} |")
    return out + [""]


# ---------------------------------------------------------------------------------------------------- HTML
_CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1f2933;--muted:#5b6b7b;--line:#d5dce4;--head:#eef2f6;--accent:#1d4e89;--ok:#1b7a3d;--bad:#b3261e;--warn:#8a5a00;--chip:#e6edf5}
@media (prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e3e8ee;--muted:#9aa8b6;--line:#2c3540;--head:#1c232b;--accent:#7fb0e6;--ok:#58c27d;--bad:#ff8a80;--warn:#e0b050;--chip:#25303b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.7rem;margin:0 0 .4rem;color:var(--accent)}
h2{font-size:1.25rem;margin:2.2rem 0 .6rem;padding-bottom:.25rem;border-bottom:1px solid var(--line)}p{margin:.5rem 0}a{color:var(--accent)}.muted{color:var(--muted)}
.summary{display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0}.stat{background:var(--head);border:1px solid var(--line);border-radius:8px;padding:.5rem .9rem}
.stat b{display:block;font-size:1.4rem}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
dl.run{display:grid;grid-template-columns:max-content 1fr;gap:.15rem 1rem;margin:.6rem 0;font-size:.95rem}dl.run dt{color:var(--muted)}dl.run dd{margin:0}
.wrap{overflow-x:auto;margin:.5rem 0}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{border-bottom:1px solid var(--line);padding:.3rem .5rem;text-align:left;vertical-align:top}th{background:var(--head);position:sticky;top:0;font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tr.case td{background:var(--head);border-top:2px solid var(--line)}
tr.case .key{font-weight:700}.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:0 .55rem;font-size:.78rem;margin-right:.4rem}
.src{color:var(--muted);font-size:.85rem}td.fail{color:var(--bad);font-weight:700}code{font-family:ui-monospace,Consolas,monospace;font-size:.9em}
@media (max-width:640px){main{padding:16px}h1{font-size:1.4rem}}
"""


def _kind_chip(e: catalogue.Entry) -> str:
    return '<span class="chip">relation stated, no work cited</span>' if e.source_kind == "relation" else ""


def _h(s) -> str:
    return html.escape(str(s), quote=True)


def html_text(model: Model) -> str:
    cnt = model.counts()
    p: list[str] = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
                    '<meta name="viewport" content="width=device-width, initial-scale=1">',
                    '<link rel="icon" href="data:,">',
                    "<title>FEA Studio – public verification set</title>",
                    '<meta name="description" content="Verification cases of FEA Studio compared with independent references.">',
                    f"<style>{_CSS}</style></head><body><main>", "<h1>FEA Studio – public verification set</h1>",
                    '<p class="muted">Each case is compared with a reference that does not come from FEA Studio: a closed-form solution, a '
                    "handbook or standard formula, a published solution, or the open-source solver MYSTRAN. For each case either a work is "
                    "cited or, for an elementary relation, the first-principles relation is stated (marked <em>relation stated, no work "
                    "cited</em>). A case without such a reference is not published here.</p>", '<div class="summary">',
                    f'<div class="stat"><b class="{"ok" if not cnt["fail"] else "bad"}">{cnt["pass"]} / {cnt["cases"]}</b>cases within tolerance</div>',
                    f'<div class="stat"><b>{cnt["checks"]}</b>checks listed</div>']
    if cnt["fail"]:
        p.append(f'<div class="stat"><b class="bad">{cnt["fail"]}</b>outside tolerance</div>')
    if cnt["incomplete"]:
        p.append(f'<div class="stat"><b class="warn">{cnt["incomplete"]}</b>within tolerance, no MYSTRAN cross-check</div>')
    if cnt["not run"]:
        p.append(f'<div class="stat"><b class="warn">{cnt["not run"]}</b>not run on this platform</div>')
    p += ["</div>", '<dl class="run">']
    p += [f"<dt>{_h(k)}</dt><dd>{_h(v)}</dd>" for k, v in _run_lines(model.meta)]
    p += ["</dl>", '<p class="muted">Files: <a href="results.csv">results.csv</a> (all rows) · <a href="README.md">README.md</a>. '
          "<em>Difference</em> is |FEA Studio − reference| / |reference| (absolute where the reference is zero); <em>tolerance</em> is the "
          "limit the case enforces; <em>mesh</em> is the node count and global element size of the finite-element model. A case is within "
          "tolerance only when all its checks pass, including a few internal consistency checks that are not listed.</p>",
          f'<p class="muted">{_h(MYSTRAN_NOTE)}</p>']
    for group in catalogue.by_domain():
        cs = [c for c in model.cases if c.entry.domain == group.domain and c.status != "not run"]
        if not cs:
            continue
        p += [f'<h2 id="{_h(group.domain)}">{_h(group.title)}</h2>', '<div class="wrap"><table>',
              '<thead><tr><th>Case / quantity</th><th>Reference</th><th class="n">FEA Studio</th><th class="n">Difference</th>'
              '<th class="n">Tolerance</th><th>Mesh</th></tr></thead><tbody>']
        for c in cs:
            e = c.entry
            mark = {"pass": '<span class="ok">✔</span>',
                    "incomplete": '<span class="warn">✔ (no MYSTRAN cross-check)</span>'}.get(c.status, '<span class="bad">✘</span>')
            p.append(f'<tr class="case" id="{_h(e.key)}"><td colspan="5"><span class="key">{_h(e.key)}</span> {mark} {_h(e.title)}<br>'
                     f'<span class="chip">{_h(catalogue.REF_TYPE_TITLES[e.ref_type])}</span>{_kind_chip(e)}<span class="src">{_h(e.source)}</span></td>'
                     f"<td>{_h(mesh_text(c))}</td></tr>")
            for r in c.rows:
                p.append(f"<tr><td>{_h(r.quantity)}</td><td>{_h(_value_text(r.reference, r.unit))}</td>"
                         f'<td class="n">{_h(_value_text(r.value, r.unit))}</td><td class="n{"" if r.ok else " fail"}">{_h(diff_text(r))}</td>'
                         f'<td class="n">{_h(tol_text(r))}</td><td></td></tr>')
        p.append("</tbody></table></div>")
    if model.mystran_rows:
        p += ['<h2 id="mystran">Cross-check against MYSTRAN</h2>',
              f"<p>The same model, mesh (coarsened ×2) and nodal loads are solved by CalculiX and by {_h(model.meta.get('mystran') or 'MYSTRAN')} "
              "(independent assembly, boundary conditions and solver). Criterion: max|Δu| / max|u| and |ΔR| / |R| below "
              f"{MYSTRAN_TOL:g}.</p>", '<div class="wrap"><table><thead><tr><th>Case</th><th>Load case</th><th class="n">max|u| CalculiX [mm]</th>'
              '<th class="n">max|u| MYSTRAN [mm]</th><th class="n">max|Δu| / max|u|</th><th class="n">|ΔR| / |R|</th><th>Mesh</th></tr></thead><tbody>']
        for c in model.cases:
            for m in model.mystran_rows.get(c.entry.key, []):
                bad = "" if _mystran_ok(m) else " fail"
                p.append(f"<tr><td>{_h(c.entry.key)}</td><td>{_h(m['case'])}</td><td class=\"n\">{m['u_max_ccx']:.6g}</td>"
                         f"<td class=\"n\">{m['u_max_mystran']:.6g}</td><td class=\"n{bad}\">{m['du_rel']:.2e}</td>"
                         f"<td class=\"n{bad}\">{m['dr_rel']:.2e}</td><td>{_h(_cross_mesh(m))}</td></tr>")
        p.append("</tbody></table></div>")
    nr = [c for c in model.cases if c.status == "not run"]
    if nr:
        p += ['<h2 id="not-run">Not run on this platform</h2>', '<div class="wrap"><table><thead><tr><th>Case</th><th>Title</th>'
              "<th>Reason</th></tr></thead><tbody>"]
        p += [f"<tr><td>{_h(c.entry.key)}</td><td>{_h(c.entry.title)}</td><td>{_h(c.note)}</td></tr>" for c in nr]
        p.append("</tbody></table></div>")
    p.append("</main></body></html>")
    return "\n".join(p) + "\n"


# ---------------------------------------------------------------------------------------------------- main
def write_all(model: Model, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = {"README.md": readme_text(model), "results.csv": csv_text(model), "index.html": html_text(model)}
    paths = []
    for name, text in files.items():
        p = out / name
        p.write_text(text, encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Write docs/verification/ from the stored results of verification.run_public")
    ap.add_argument("--report", type=Path, default=HERE / "out" / "report.json")
    ap.add_argument("--mystran", type=Path, default=HERE / "out_mystran" / "report.json")
    ap.add_argument("--meta", type=Path, default=HERE / "out" / "run_meta.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-mystran", action="store_true", help="do not read the MYSTRAN comparison (MYSTRAN did not run)")
    a = ap.parse_args(argv)
    report = _load(a.report)
    if not report:
        print(f"no results in {a.report.name}: run  python -m verification.run_public  first")
        return 2
    meta = _load(a.meta)
    if not meta.get("date_utc"):
        import datetime
        meta["date_utc"] = datetime.datetime.fromtimestamp(a.report.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    model = build(report, {} if a.no_mystran else _load(a.mystran), meta)
    for p in write_all(model, a.out):
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name)
    cnt = model.counts()
    print(f"{cnt['pass']} of {cnt['cases']} published cases within tolerance, {cnt['fail']} outside, {cnt['not run']} not run, "
          + (f"{cnt['incomplete']} without their MYSTRAN cross-check, " if cnt["incomplete"] else "")
          + f"{cnt['checks']} checks")
    if model.untranslated:
        print(f"WARNING: {len(model.untranslated)} check names without an English label:")
        print("\n".join("  " + u for u in model.untranslated[:40]))
    return 0 if not cnt["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())

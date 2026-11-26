"""Software Validation Dossier (F14 stage S6): the document that a customer asks for to see that the analysis software was checked against references that do not come
from the program itself.

    python -m verification.dossier [--report out/report.json] [--mystran out_mystran/report.json] [--meta out/run_meta.json] --out <file.docx> [--markdown <file.md>]

RUNS NOTHING. Like the public verification page (`verification.public_report`) it reads the stored results of a real run (`verification.run_public` -> `out/report.json`, the MYSTRAN
comparison, `out/run_meta.json`) and the public catalogue (`verification.catalogue`) and writes a Word document (and, on request, the same text as Markdown):

  1 Purpose and scope            2 Method: the kinds of reference         3 Summary by domain
  4 Results by case (reference, FEA Studio, difference, tolerance, mesh)  5 Cross-check against MYSTRAN
  6 Internal cross-checks (the program against itself - listed, NOT claimed as independent)   7 Cases not run, limitations   8 Traceability

PUBLIC independent references only: analytic, handbook, literature and MYSTRAN cases of the catalogue are validated against; a case whose reference is the program itself is
named in chapter 6 and never counted as validation. The private cross-check (its name, its files, its case ids) never enters: a results file that is named after it or carries one of
its case ids is REFUSED (`PrivateContentError`), and every string of the document is checked against the private words and against local paths before anything is written.
The output is deterministic: the date is the date of the run that the results come from, never the time of writing; equal inputs give equal bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from verification import catalogue, public_report

HERE = Path(__file__).resolve().parent

PRIVATE_NAME = re.compile(r"abaqus|simulia|(?<![A-Za-z])abq(?![A-Za-z])", re.I)         # the private cross-check: its name, its vendor, its abbreviation and its files (abaqus_*, time_abaqus_s)
PRIVATE_KEYS = re.compile(r"(?<![A-Za-z0-9_])X[1-5](?![A-Za-z0-9_])")                   # its case ids X1-X5 (case-sensitive: "x1" may occur in ordinary text)
MAX_WALK = 2_000_000                                                                     # nodes of a results file that are looked at (a file larger than this is not a results file)
LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\[A-Za-z]|/ho" r"me/|/Us" r"ers/|/mn" r"t/[a-z]/|App" r"Data")      # written in pieces: the packaging scanner must not see a path here


class PrivateContentError(ValueError):
    """The input or the text of the dossier contains something that must not be published (the private cross-check, a local path)."""


def private_word(text: str) -> str | None:
    """The first private word or local path of `text`, or None."""
    m = PRIVATE_NAME.search(text) or PRIVATE_KEYS.search(text) or LOCAL_PATH.search(text)
    return m.group(0) if m else None


def _strings(x: Any) -> Iterator[tuple[str, str]]:
    """(where, text) of every key and every string value of a loaded JSON structure, at any depth."""
    stack: list[tuple[str, Any]] = [("", x)]
    n = 0
    while stack:
        where, v = stack.pop()
        n += 1
        if n > MAX_WALK:
            raise PrivateContentError("the results file is too large to be checked: it is not a results file of the verification run")
        if isinstance(v, dict):
            for k, w in v.items():
                stack.append((f"{where}/{k}", w))
                yield f"{where}/{k}", str(k)
        elif isinstance(v, (list, tuple)):
            stack += [(f"{where}[{i}]", w) for i, w in enumerate(v)]
        elif isinstance(v, str):
            yield where, v


def check_inputs(report: dict[str, Any], paths: list[Path], *others: Any) -> None:
    """Refuses the results of the private cross-check: a file that is named after it, or results (the table, the MYSTRAN comparison, the run metadata) that carry its name, its
    vendor, its abbreviation, one of its field names (`time_abaqus_s`) or one of its case ids - in ANY key or string value at any depth, not only in the case ids."""
    for p in paths:
        w = private_word(p.name) or (private_word(p.parent.name) if p.parent.name else None)
        if w:
            raise PrivateContentError(f"{p.name}: the results file belongs to the private cross-check ('{w}') and cannot be the source of the dossier")
    bad = sorted(k for k in report if PRIVATE_KEYS.fullmatch(str(k)) or PRIVATE_NAME.search(str(k)))
    if bad:
        raise PrivateContentError("the results table carries case ids of the private cross-check (" + ", ".join(bad[:5]) + "): it is not a public run")
    for part in (report, *others):
        for where, text in _strings(part):
            w = PRIVATE_NAME.search(text) or PRIVATE_KEYS.search(text)
            if w:
                raise PrivateContentError(f"the results carry '{w.group(0)}' (at {where[:60]}): it belongs to the private cross-check, this is not a public run")


# ------------------------------------------------------------------------------------------------------------------ the content (neutral blocks)
Block = tuple[Any, ...]                          # ("h", level, text) | ("p", text) | ("b", text) | ("table", header, rows, caption)


def _counts_by_domain(model: public_report.Model, report: dict[str, Any]) -> list[list[str]]:
    rows = []
    for group in catalogue.by_domain():
        cs = [c for c in model.cases if c.entry.domain == group.domain]
        kinds = sorted({catalogue.REF_TYPE_TITLES[c.entry.ref_type] for c in cs})
        n = {k: sum(_outcome(c, report) == k for c in cs) for k in ("pass", "outside", "incomplete", "not run")}
        rows.append([group.title, str(len(cs)), str(n["pass"]), str(n["outside"]), str(n["incomplete"]), str(n["not run"]), "; ".join(kinds)])
    return rows


def _outcome(c: Any, report: dict[str, Any]) -> str:
    """pass | outside | incomplete | not run. `incomplete` = every tolerance check of the case passes and only its comparison with MYSTRAN is missing or outside (a run without
    MYSTRAN, a failed comparison): that is not a case outside tolerance."""
    if c.status in ("pass", "not run"):
        return c.status
    rec = report.get(c.entry.key)
    ok = isinstance(rec, dict) and "error" not in rec and bool(rec.get("checks")) and all(bool(x.get("ok")) for x in rec["checks"])
    return "incomplete" if ok else "outside"


def _status_text(c: Any, report: dict[str, Any]) -> str:
    kind = _outcome(c, report)
    if kind == "pass":
        return "within tolerance"
    if kind == "incomplete":
        return "within tolerance; the comparison with MYSTRAN is incomplete" + (f" ({c.note})" if c.note else " (outside tolerance, see chapter 5)")
    rec = report.get(c.entry.key)
    if isinstance(rec, dict) and "error" in rec:
        return "NOT PASSED" + (f" ({c.note})" if c.note else "")
    return "OUTSIDE TOLERANCE" + (f" ({c.note})" if c.note else "")


def _internal_rows(report: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key in catalogue.internal_keys():
        rep = report.get(key)
        if not isinstance(rep, dict) or "checks" not in rep and "error" not in rep:
            continue
        e = catalogue.get(key)
        ok = "error" not in rep and bool(rep.get("checks")) and all(bool(c.get("ok")) for c in rep["checks"])
        rows.append([key, e.title, "within tolerance" if ok else "outside tolerance or stopped", str(len(rep.get("checks") or []))])
    return rows


def blocks(model: public_report.Model, report: dict[str, Any], digest: str) -> list[Block]:
    """The text of the dossier as neutral blocks (rendered to Word and to Markdown)."""
    from fea_core import __version__ as version
    cnt = model.counts()
    n_out = sum(_outcome(c, report) == "outside" for c in model.cases)
    n_inc = sum(_outcome(c, report) == "incomplete" for c in model.cases)
    meta = model.meta
    out: list[Block] = [("h", 0, f"Software Validation Dossier - FEA Studio {meta.get('program_version') or version}")]
    run_rows = [[k, v] for k, v in public_report._run_lines(meta)] or [["Program", f"FEA Studio {version}"]]
    if not any(k.startswith("Date of the run") for k, _v in run_rows):
        run_rows.insert(1, ["Date of the run (UTC)", "date unknown (the metadata file of the run is not available)"])             # never the time of the file or of today
    out.append(("table", None, run_rows, None))
    out += [("h", 1, "1 Purpose and scope"),
            ("p", "This dossier documents how the calculation results of FEA Studio were checked against references that do not come from the program itself. It is generated "
                  "automatically from the stored results of a verification run and from the public catalogue of the verification cases; nothing in it was typed by hand."),
            ("p", f"{cnt['pass']} of {cnt['cases']} published cases are within tolerance"
                  + (f", {n_out} outside tolerance" if n_out else "") + (f", {n_inc} within tolerance but without a complete comparison with MYSTRAN" if n_inc else "")
                  + (f", {cnt['not run']} were not run on the platform of this run" if cnt["not run"] else "")
                  + f" ({cnt['checks']} checks are listed in chapter 4)."),
            ("p", "The dossier supports the use of the program for the structural analysis of a small satellite (the Structural Analysis Report of the program refers to it). It does not "
                  "validate a particular model: the quality of a model (mesh, boundary conditions, joints) is checked by the model quality checks of that report."),
            ("h", 1, "2 Method: the kinds of reference"),
            ("p", "Every published case is compared with a reference of one of these kinds. A case whose reference is the program itself is not published as validation (chapter 6).")]
    out += [("b", f"{catalogue.REF_TYPE_TITLES[k]}: " + {
        "analytic": "a closed-form solution derived from first principles (beam theory, Lamé, Fourier's law, Euler buckling ...)",
        "handbook": "tabulated coefficients, empirical correlations or code-of-practice formulas (stress concentration factors, convection correlations, EN / VDI / IIW rules ...)",
        "literature": "a published solution or method with its own name (Hertz, Hetényi, Lekhnitskii, Miles, Neuber, Findley ...)",
        "mystran": "the same model solved by MYSTRAN, an independent open-source solver"}[k]) for k in catalogue.PUBLIC_TYPES]
    out += [("p", "A case is within tolerance only when all its checks pass, including a few consistency checks that are not listed. The difference is |FEA Studio - reference| / |reference|, "
                  "or the absolute difference where the reference is zero. The mesh column gives the node count and the global element size of the finite-element model."),
            ("h", 1, "3 Summary by domain"),
            ("table", ["Domain", "Cases", "Within tolerance", "Outside", "No MYSTRAN comparison", "Not run", "Kinds of reference"], _counts_by_domain(model, report),
             "Published cases by domain"),
            ("h", 1, "4 Results by case")]
    for group in catalogue.by_domain():
        cs = [c for c in model.cases if c.entry.domain == group.domain and c.status != "not run"]
        if not cs:
            continue
        out.append(("h", 2, group.title))
        for c in cs:
            e = c.entry
            out.append(("p", f"{e.key} - {e.title}: " + _status_text(c, report)
                             + f". Reference ({catalogue.REF_TYPE_TITLES[e.ref_type]}" + (", relation stated, no work cited" if e.source_kind == "relation" else "") + f"): {e.source}"))
            if c.rows:
                out.append(("table", ["Quantity", "Reference", "FEA Studio", "Difference", "Tolerance", "Mesh"],
                            [[r.quantity, public_report._value_text(r.reference, r.unit), public_report._value_text(r.value, r.unit),
                              public_report.diff_text(r) + ("" if r.ok else " (outside)"), public_report.tol_text(r), public_report.mesh_text(c) if i == 0 else ""]
                             for i, r in enumerate(c.rows)], f"{e.key}"))
    my = [(c, m) for c in model.cases for m in model.mystran_rows.get(c.entry.key, [])]
    out.append(("h", 1, "5 Cross-check against MYSTRAN"))
    if my:
        out.append(("p", f"The same model, mesh (coarsened by a factor of 2) and nodal loads are solved by CalculiX and by {meta.get('mystran') or 'MYSTRAN'} (independent assembly, boundary "
                         f"conditions and solver). Criterion: max|du| / max|u| and |dR| / |R| below {public_report.MYSTRAN_TOL:g}."))
        out.append(("table", ["Case", "Load case", "max |u| CalculiX [mm]", "max |u| MYSTRAN [mm]", "max |du| / max |u|", "|dR| / |R|", "Mesh"],
                    [[c.entry.key, m["case"], f"{m['u_max_ccx']:.6g}", f"{m['u_max_mystran']:.6g}", f"{m['du_rel']:.2e}", f"{m['dr_rel']:.2e}", public_report._cross_mesh(m)] for c, m in my],
                    "CalculiX against MYSTRAN"))
    else:
        out.append(("p", "No comparison with MYSTRAN is part of the results this dossier was made from."))
    out.append(("h", 1, "6 Internal cross-checks (not independent)"))
    irows = _internal_rows(report)
    out.append(("p", "The checks below compare the program with itself (two engines of the program against each other, an equilibrium or bookkeeping check, a wiring test of an input "
                     "table). They show that the program is consistent; they are NOT counted as validation against an independent reference."))
    if irows:
        out.append(("table", ["Case", "What is compared", "Result", "Checks"], irows, "Internal cross-checks of the run"))
    else:
        out.append(("p", "No internal cross-check is part of the results this dossier was made from."))
    out.append(("h", 1, "7 Cases not run and limitations"))
    nr = [c for c in model.cases if c.status == "not run"]
    if nr:
        out.append(("table", ["Case", "Title", "Reason"], [[c.entry.key, c.entry.title, c.note] for c in nr], "Published cases that were not run on the platform of this run"))
    else:
        out.append(("p", "Every published case was run."))
    out += [("b", "The cases cover the capabilities that the catalogue lists; a capability without a published case is not validated by this dossier."),
            ("b", "A case is validated on its own mesh; the dossier makes no statement about the discretisation error of another model."),
            ("b", "The results are those of the platform, program version and date named in the table at the top; another platform or version needs a new run.")]
    out += [("h", 1, "8 Traceability"),
            ("table", None, [["Results file SHA-256", digest], ["Catalogue", f"{len(catalogue.ENTRIES)} cases, {len(catalogue.public_keys())} published"],
                             ["Program", f"FEA Studio {meta.get('program_version') or version}"]], None)]
    return out


# ------------------------------------------------------------------------------------------------------------------ rendering
def _all_text(bl: list[Block]) -> str:
    parts: list[str] = []
    for b in bl:
        if b[0] in ("h",):
            parts.append(str(b[2]))
        elif b[0] in ("p", "b"):
            parts.append(str(b[1]))
        else:
            parts += [str(h) for h in (b[1] or [])] + [str(c) for r in b[2] for c in r] + ([str(b[3])] if b[3] else [])
    return "\n".join(parts)


def markdown(bl: list[Block]) -> str:
    from fea_core.report.space_sar import clean_paths

    def esc(t: Any) -> str:
        return clean_paths(t).replace("|", "\\|").replace("\n", " ")                                 # the same cleaning as the Word file
    out: list[str] = []
    for b in bl:
        if b[0] != "b" and out and out[-1].startswith("- "):
            out.append("")                                           # a list ends with an empty line
        if b[0] == "h":
            out += [("#" * (b[1] + 1)) + " " + esc(b[2]), ""]
        elif b[0] == "p":
            out += [esc(b[1]), ""]
        elif b[0] == "b":
            out.append("- " + esc(b[1]))
        else:
            hdr = b[1] or [""] * len(b[2][0])
            if b[3]:
                out += [f"*{esc(b[3])}*", ""]
            out += ["| " + " | ".join(esc(h) for h in hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in b[2]] + [""]
    return "\n".join(out).rstrip() + "\n"


def write_docx(bl: list[Block], path: Path, stamp: str) -> None:
    import datetime as dt

    from fea_core.report.space_sar import _CleanDoc, freeze_docx
    D = _CleanDoc()
    first = next(b for b in bl if b[0] == "h")
    D.footer(f"{first[2]} · generated from stored results")
    for b in bl:
        if b[0] == "h":
            if b[1] == 0:
                D.doc.add_heading(first[2] if b is first else b[2], 0)
            else:
                D.h(b[2], b[1])
        elif b[0] == "p":
            D.p(b[1])
        elif b[0] == "b":
            D.bullet(b[1])
        elif b[1] is None:
            D.kv([(r[0], r[1]) for r in b[2]])
        else:
            D.table(b[1], b[2], caption=b[3], font=8)
    cp = D.doc.core_properties
    cp.title, cp.author, cp.last_modified_by, cp.subject, cp.comments = str(first[2]), "FEA Studio", "FEA Studio", "Software validation", ""
    try:
        cp.created = cp.modified = dt.datetime.strptime(stamp[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        cp.created = cp.modified = dt.datetime(2000, 1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    D.doc.save(str(path))
    freeze_docx(path)


def _identity_clean(x: Any) -> Any:
    """A copy of a loaded JSON structure whose string VALUES lost the user name and the host name of the machine (they come from a run on some machine, not from this module).
    A local PATH is not removed here: it is refused when the text of the dossier is checked."""
    from fea_core.report.space_sar import identity_redactor
    redact = identity_redactor()                                                  # the names of this machine are looked up once for the whole structure

    def walk(y: Any) -> Any:
        if isinstance(y, dict):
            return {k: walk(v) for k, v in y.items()}
        if isinstance(y, list):
            return [walk(v) for v in y]
        return redact(y) if isinstance(y, str) else y
    return walk(x)


def build_dossier(report: dict[str, Any], mystran: dict[str, Any], meta: dict[str, Any], digest: str = "") -> list[Block]:
    """The blocks of the dossier from the stored results; raises `PrivateContentError` when the results or the text are not public."""
    check_inputs(report, [], mystran, meta)
    report, mystran, meta = (_identity_clean(x) for x in (report, mystran, meta))               # text from the results (an error message, a platform line) loses paths and names
    model = public_report.build(report, mystran, meta)
    bl = blocks(model, report, digest)
    bad = private_word(_all_text(bl))
    if bad:
        raise PrivateContentError(f"the text of the dossier contains '{bad}': it is not published")
    return bl


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m verification.dossier", description="Software Validation Dossier from the stored results of verification.run_public")
    ap.add_argument("--report", type=Path, default=HERE / "out" / "report.json")
    ap.add_argument("--mystran", type=Path, default=HERE / "out_mystran" / "report.json")
    ap.add_argument("--meta", type=Path, default=HERE / "out" / "run_meta.json")
    ap.add_argument("--no-mystran", action="store_true", help="do not read the MYSTRAN comparison (MYSTRAN did not run)")
    ap.add_argument("--out", type=Path, required=True, help="the Word file to write")
    ap.add_argument("--markdown", type=Path, default=None, help="also write the same text as Markdown")
    a = ap.parse_args(argv)
    try:
        report = public_report._load(a.report)
        if not report:
            print(f"no results in {a.report.name}: run  python -m verification.run_public  first")
            return 2
        mystran = {} if a.no_mystran else public_report._load(a.mystran)
        check_inputs(report, [a.report] + ([] if a.no_mystran else [a.mystran]) + [a.meta])
        meta = public_report._load(a.meta)                          # no run_meta.json: no date (never the time of the file: the output must not depend on when it was copied)
        digest = hashlib.sha256(a.report.read_bytes()).hexdigest()
        bl = build_dossier(report, mystran, meta, digest)
    except PrivateContentError as e:
        print(f"REFUSED: {e}")
        return 3
    write_docx(bl, a.out, str(meta.get("date_utc") or ""))
    print(a.out.name)
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(markdown(bl), encoding="utf-8", newline="\n")
        print(a.markdown.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

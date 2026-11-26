"""Word report "Structural Analysis Report (ECSS)" (F14 stage S6): the document that a structural verification of a small satellite hands to the customer.

It is NOT a second implementation of the analyses: every chapter is written by the section functions of the earlier stages (`space.py` S1 / S3, `space_loads.py` S2,
`space_corr.py` S5, `space_s4.py` S4 when the build has it); this module only orders them in the ECSS order, adds the configuration and traceability block, and does the
bookkeeping of the sign-off:

    1 Configuration and traceability   2 Model and mass properties       3 Confirmed requirement set       4 Model quality checks (S3)
    5 Load cases and envelopes (S2)    6 Margins of safety and singularity justifications (S1)             7 Interfaces, notching, bolts and PCB (S4)
    8 Test correlation (S5, optional)  9 Assumptions, limitations and open items

Sign-off rule. Every chapter gets a status. The overall verdict is PASS only when every INCLUDED chapter is current and PASS (the chapter "Model and mass properties" may be
INFO when no budget is entered: there is nothing to verify). Anything that is OUTDATED, UNCONFIRMED_REQUIREMENTS, PARTIAL_SCOPE, UNVERIFIABLE, WARN, NOT_RUN or FAIL blocks it and
is listed in the sign-off summary and in chapter 9. A result that a chapter reports as out of date is never turned into PASS here (the chapter functions never print a
stored verdict on its own word - they compare it with the present study first). Chapters that are optional and absent (no stored correlation; no S4 module in the build) are
named in the scope line of the summary, not silently dropped. The program contains no standard value: every factor, limit and load level printed was entered by the user
with its source; this module has none either.

Privacy: nothing that identifies the person, the machine or a folder is written - every string goes through `clean_text` (paths, the user name and the host name of the
machine are replaced); the geometry is identified by its SHA-256, not by a file name.
"""
from __future__ import annotations

import functools
import getpass
import hashlib
import importlib
import importlib.util
import os
import platform
import re
import socket
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import __version__ as FEA_VERSION
from .. import i18n
from ..i18n import tr as _tr
from . import space as SP
from . import space_loads as SL
from .word import ReportOptions, _Doc, num

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
UNCONFIRMED, OUTDATED, PARTIAL, UNVERIFIABLE, NOT_RUN = "UNCONFIRMED_REQUIREMENTS", "OUTDATED", "PARTIAL_SCOPE", "UNVERIFIABLE", "NOT_RUN"
INFO, NOT_INCLUDED, NOT_IN_BUILD = "INFO", "NOT_INCLUDED", "NOT_IN_BUILD"
NOT_SIGNED_OFF = "NOT_SIGNED_OFF"
_ORDER = (FAIL, OUTDATED, NOT_RUN, PARTIAL, UNVERIFIABLE, UNCONFIRMED, WARN)          # which status wins when a chapter has several (the most serious first)
_MARKERS = re.compile(r"UNCONFIRMED|OUTDATED|[Oo]utdated|PARTIAL_SCOPE|UNVERIFIABLE|[Tt]o be confirmed|[Tt]o be checked|NOT [Cc]onfirmed|not confirmed|NOT JUSTIFIED|NOT usable")
_TABLE_MARKERS = ("UNCONFIRMED", "OUTDATED", "outdated", "PARTIAL_SCOPE", "UNVERIFIABLE", "NOT confirmed", "NOT JUSTIFIED")
MANDATORY = ("model", "requirements", "checks", "loads", "margins")                     # a report without one of these is never a sign-off
MAX_OPEN_ITEMS = 80
_HIGHEST_PLACES = re.compile(r"the \d+ highest places are all singularities")
NOT_IN_BUILD_REASON = "interfaces/bolts not assessed in this build"
SINGULARITY_OPEN = "singularity not justified"                # the beginning of an open item of the margins (`MarginSummary.open_items`)
S4_MODULE = "fea_core.report.space_s4"


# ------------------------------------------------------------------------------------------------------------------ privacy
_COMMON_NAMES = {"user", "users", "admin", "administrator", "root", "test", "default", "guest", "owner", "student", "python", "fea"}


_ID_ENV = ("USERNAME", "USER", "LOGNAME", "LNAME", "COMPUTERNAME", "HOSTNAME")
_ID_CACHE: dict[tuple, list[str]] = {}


def _identities() -> list[str]:
    """The user name and the host names of this machine, longest first. They are looked up once (a lookup of the host name takes milliseconds and a text-by-text redaction asks
    for them thousands of times); the answer is looked up again when the environment or one of the lookup functions is not the same as before."""
    key = (getpass.getuser, platform.node, socket.gethostname, *(os.environ.get(k, "") for k in _ID_ENV))
    hit = _ID_CACHE.get(key)
    if hit is None:
        _ID_CACHE.clear()
        hit = _ID_CACHE[key] = _identities_now()
    return list(hit)


def _identities_now() -> list[str]:
    names: set[str] = set()
    for fn in (getpass.getuser, platform.node, socket.gethostname):
        try:
            names.add(str(fn()))
        except Exception:  # noqa: BLE001 - a name that cannot be found cannot leak
            pass
    names |= {os.environ.get(k, "") for k in ("USERNAME", "USER", "COMPUTERNAME", "HOSTNAME")}
    return sorted((n for n in names if len(n) >= 3 and n.lower() not in _COMMON_NAMES), key=len, reverse=True)


@functools.lru_cache(maxsize=64)
def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])", re.I)


def clean_paths(text: Any) -> str:
    """The text without a local path. This is all that is done to text the user entered (author, company, names, provenance of the requirements): a name that happens to be the login
    name of the machine, or an ordinary word, is theirs and stays."""
    from ..space._checks_common import scrub_paths
    return scrub_paths(str(text))


def clean_text(text: Any) -> str:
    """The text without a local path, the user name or the machine name: for text that the PROGRAM generated (an exception message, a file name, the environment), where such a
    name may have come in by accident. Never applied to a field that the user typed - and never to the prose of the report, where a login that is an ordinary word would damage it."""
    return redact_identities(clean_paths(text))


def redact_identities(text: Any) -> str:
    """The text without the user name and the host name of this machine (nothing else is changed)."""
    return identity_redactor()(text)


def identity_redactor() -> Callable[[Any], str]:
    """`redact_identities` with the names looked up once: for the redaction of many strings in a row (the lookup is the cost, not the substitution)."""
    names = _identities()
    patterns = [(n.lower() if n.isascii() else None, _name_pattern(n)) for n in names]

    def redact(text: Any) -> str:
        s = str(text)
        for low, p in patterns:                                                # one after the other, longest name first: the same result as a substitution per name
            if low is not None and s.isascii() and low not in s.lower():       # an ASCII name cannot match case-insensitively in an ASCII text that does not contain it
                continue
            s = p.sub("<redacted>", s)
        return s
    return redact


class _CleanDoc(_Doc):
    """The Word document of `word._Doc` with every string cleaned of local paths (`clean_paths`) - the footer included. Program-generated text that may carry an identity goes
    through `clean_text` where it is made (the exception messages of the chapters)."""

    def h(self, text: str, level: int = 1):
        return super().h(clean_paths(text), level)

    def p(self, text: str = "", *args: Any, **kw: Any):
        return super().p(clean_paths(text), *args, **kw)

    def bullet(self, text: str):
        return super().bullet(clean_paths(text))

    def footer(self, text: str):
        return super().footer(clean_paths(text))

    def table(self, header: list[str] | None, rows: list[list], widths: Any = None, caption: str | None = None, status_col: int | None = None, font: float = 9):
        header = [clean_paths(h) for h in header] if header else header
        rows = [[clean_paths(c) if isinstance(c, str) else c for c in r] for r in rows]
        return super().table(header, rows, widths, clean_paths(caption) if caption else caption, status_col, font)


# ------------------------------------------------------------------------------------------------------------------ chapters and the verdict
@dataclass
class Chapter:
    key: str
    title: str
    status: str = NOT_RUN
    basis: str = ""                                          # one line: why the chapter has this status
    items: list[str] = field(default_factory=list)           # what is open in it (every item blocks the sign-off unless the chapter is PASS)
    number: int = 0
    span: tuple[int, int, int, int] = (0, 0, 0, 0)           # paragraphs and tables of the document that belong to it (start, end, start, end)
    facts: dict[str, str] = field(default_factory=dict)      # the identification facts of the configuration chapter (digests, versions)
    notes: list[str] = field(default_factory=list)           # what the reader must know about how far the status is checked (printed beside it)

    @property
    def included(self) -> bool:
        return self.status not in (NOT_INCLUDED, NOT_IN_BUILD)

    @property
    def passed(self) -> bool:
        return self.status in (PASS, INFO)

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "key": self.key, "title": self.title, "status": self.status, "basis": self.basis, "open_items": list(self.items), "notes": list(self.notes)}


def worst(statuses: list[str]) -> str:
    """The most serious of several statuses (PASS when there is none)."""
    for s in _ORDER:
        if s in statuses:
            return s
    return PASS


def overall_verdict(chapters: list[Chapter]) -> tuple[str, list[Chapter]]:
    """(PASS | FAIL | NOT_SIGNED_OFF, the chapters that block). PASS only when every included chapter is PASS (or INFO) and every mandatory chapter is there; a chapter that is
    NOT_IN_BUILD (stage S4 is not part of the build) blocks too: the interfaces and bolts were not assessed. NOT_INCLUDED (an optional chapter, no stored correlation) does not."""
    inc = [c for c in chapters if c.included]
    blockers = [c for c in inc if not c.passed] + [c for c in chapters if c.status == NOT_IN_BUILD]       # a chapter this build cannot write is not assessed: no sign-off
    have = {c.key for c in inc}
    missing = [k for k in MANDATORY if k not in have]
    if any(c.status == FAIL for c in inc):
        return FAIL, blockers
    if blockers or missing or not inc:
        return NOT_SIGNED_OFF, blockers
    return PASS, []


# ------------------------------------------------------------------------------------------------------------------ traceability
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str:
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


_CCX_VERSION: dict[str, str] = {}


def ccx_version() -> str:
    """The version line of the CalculiX that is found (`ccx -v`, a few milliseconds); cached."""
    from .. import config
    exe = config.find_ccx()
    key = str(exe)
    if key not in _CCX_VERSION:
        ver = "not found"
        if exe.exists():
            try:
                r = subprocess.run([key, "-v"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, creationflags=getattr(config, "NO_WINDOW", 0))
                m = re.search(r"[Vv]ersion\s+([0-9][0-9.]*)", r.stdout + r.stderr)
                ver = m.group(1) if m else "version not reported"
            except (OSError, subprocess.SubprocessError):
                ver = "version not determined"
        _CCX_VERSION[key] = ver
    return _CCX_VERSION[key]


def _file_name(path: Any) -> str:
    """The last part of a path written on either platform (a study saved on Windows and read on Linux keeps its backslashes)."""
    from pathlib import PurePosixPath, PureWindowsPath
    return PurePosixPath(PureWindowsPath(str(path or "")).as_posix()).name


def _without_keys(x: Any, keys: frozenset[str]) -> Any:
    if isinstance(x, dict):
        return {k: _without_keys(v, keys) for k, v in x.items() if k not in keys}
    if isinstance(x, list):
        return [_without_keys(v, keys) for v in x]
    return x


VOLATILE_KEYS = frozenset({"stamp"})                             # the time at which a stored S4 result was computed: it is not part of what the study IS


def _stable_content(study: Any, geo: str | None) -> dict[str, Any]:
    """The content of the study that its hash covers: no folder of the geometry file and no time stamp of a stored result (the same study has the same hash when it is saved,
    moved or its stored results are recomputed with the same inputs)."""
    content = study.to_dict()
    content["geometry"] = geo or _file_name(content.get("geometry"))
    sp = content.get("space")
    if isinstance(sp, dict) and "s4" in sp:                                  # `SpaceSettings.extra` is written at the top level of the space dictionary
        sp["s4"] = _without_keys(sp["s4"], VOLATILE_KEYS)
    return content


def traceability(sess: Any, study: Any, res: Any, mesh: Any) -> tuple[list[tuple[str, str]], dict[str, str], list[str]]:
    """(rows of the table, the plain dictionary of the same facts for the caller, warnings). No local path, user or machine name."""
    from ..mcp import setup_proof as sp
    from ..space import environments as EV
    from .word import _program_row
    warnings: list[str] = []
    facts: dict[str, str] = {}
    facts["study"] = str(study.name)
    step = getattr(sess, "step_path", None)
    geo = sp.geometry_file_hash(step) if step else None
    content = _stable_content(study, geo)
    facts["study_sha256"] = _sha256_text(sp.canonical_json(content))
    facts["geometry_sha256"] = geo or ""
    try:
        facts["setup_digest"] = sp.session_digest(sess) if hasattr(sess, "require_study") else sp.setup_digest(study, geo)
        facts["setup_digest_basis"] = "full"
    except Exception as e:  # noqa: BLE001 - without the geometry the faces of the selections cannot be resolved: the digest of the study with the geometry hash is used
        facts["setup_digest"] = sp.setup_digest(study, geo)
        facts["setup_digest_basis"] = "study and geometry hash"
        warnings.append(f"Setup digest: the faces of the selections could not be resolved ({clean_text(e)}); the digest covers the study and the geometry hash only")
    rs = getattr(getattr(study, "space", None), "requirement_set", None)
    es = EV.environments_of(getattr(study, "space", None)) if getattr(study, "space", None) is not None else None
    facts["requirement_set_digest"] = rs.content_digest() if rs is not None else ""
    facts["environments_digest"] = es.content_digest() if es is not None else ""
    facts["fea_studio"] = FEA_VERSION
    facts["calculix"], facts["rslab"], facts["gmsh"] = ccx_version(), _package_version("rslab"), _package_version("gmsh")
    rows: list[tuple[str, str]] = [("Study", facts["study"]), ("Study content SHA-256", facts["study_sha256"]), ("Setup digest SHA-256 (not checked against a confirmation)", facts["setup_digest"]),
                                   ("Geometry SHA-256", facts["geometry_sha256"] or "– (no geometry file: a frame-only study, or the file is not available)"),
                                   ("Requirement set SHA-256", facts["requirement_set_digest"] or "–"), ("Launch environments SHA-256", facts["environments_digest"] or "–"),
                                   ("FEA Studio", FEA_VERSION), ("CalculiX", facts["calculix"]), ("rslab (direct solver)", facts["rslab"]), ("Gmsh", facts["gmsh"]),
                                   ("Software used for the results", _program_row(study, res)), ("Operating system", platform.system() or "–")]
    engines = sorted({str(getattr(m, "engine", "")) for m in (getattr(res, "modes", None) or []) if getattr(m, "engine", "")})
    if engines:
        rows.append(("Engine of the natural modes", ", ".join(engines)))
    if mesh is not None and len(getattr(mesh, "nodes", [])):
        rep = getattr(mesh, "report", None) or {}
        sj = rep.get("min_scaled_jacobian")
        ms = study.mesh
        facts["mesh"] = f"{len(mesh.nodes)} nodes, {len(mesh.tets)} elements"
        rows += [("Mesh", f"{len(mesh.nodes)} nodes, {len(mesh.tets)} elements ({'tet10' if ms.order == 2 else 'tet4'}), {len(mesh.part_names)} parts, {3 * len(mesh.nodes)} degrees of freedom"),
                 ("Element size settings", f"global {num(ms.size)} mm" + (f", {len(ms.part_sizes)} part size(s)" if ms.part_sizes else "") + (f", {len(ms.local)} local refinement(s)" if ms.local else "")),
                 ("Minimum scaled Jacobian", num(sj, 3) if sj is not None else "–")]
    else:
        facts["mesh"] = ""
        rows.append(("Mesh", "no mesh is available in this report"))
    return rows, facts, warnings


# ------------------------------------------------------------------------------------------------------------------ the chapters
def _used_materials(study: Any, mesh: Any) -> list[tuple[str, str]]:
    names = list(getattr(mesh, "part_names", None) or study.part_materials.keys())
    return [(p, study.part_materials.get(p) or study.default_material or "") for p in names]


def _configuration(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    mesh = SL._mesh_of(sess)
    rows, facts, warnings = traceability(sess, study, res, mesh)
    D.p("The digests identify exactly the inputs of this report: any change of a value, a face, a material, a limit, the mesh settings or the geometry gives another digest. "
        "No file name, folder, user or machine name is written into this report.")
    D.kv(rows)
    ch.facts = facts
    if not facts.get("geometry_sha256"):
        ch.status, ch.basis = UNVERIFIABLE, "the geometry is not identified: there is no geometry file hash"
        ch.items.append("the geometry is not identified (no geometry file or no hash of it): the report cannot be tied to the geometry it was made from")
    elif facts.get("setup_digest_basis") != "full":
        ch.status, ch.basis = WARN, "the setup digest covers the study and the geometry hash only"
        ch.items.append("the setup digest covers the study and the geometry hash only (the faces of the selections could not be resolved)")
    else:
        ch.status, ch.basis = INFO, "identification of the study, the software and the mesh"
    return warnings


def _model(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    mesh = SL._mesh_of(sess)
    if mesh is None or not len(getattr(mesh, "nodes", [])):
        D.p("There is no mesh in this report (the study was not solved, or the results are not available): the model and its mass properties cannot be reported.", italic=True)
        ch.status, ch.basis = NOT_RUN, "no mesh"
        ch.items.append("no mesh: solve the study before the report")
        return []
    D.p("The structure as it was modelled: the parts with their materials and structure classes, the masses that were added to the model, the supports and the mass "
        "properties of the whole model (mass, centre of mass and inertia as the mesh and the added masses give them).")
    sp = getattr(study, "space", None)
    D.table(["Part", "Material", "Structure class"], [[p, m or "–", sp.class_of(p) if sp is not None else "–"] for p, m in _used_materials(study, mesh)], caption="Parts of the model", font=8,
            widths=(0.4, 0.4, 0.2))
    if study.masses:
        D.table(["Added mass", "Kind", "Value", "Coupling / selection"],
                [[getattr(m, "name", "") or "–", m.kind.replace("_", " "), (f"{num(m.mass * 1000.0)} kg" if m.kind == "point_mass" else f"{num(m.mass_per_area)} kg/m2"),
                  (f"{m.coupling}, " if m.kind == "point_mass" else "") + m.selection] for m in study.masses], caption="Masses added to the model (components, population of a board)", font=8)
    from .word import constraint_text
    sup = [[c.selection if hasattr(c, "selection") else "–", constraint_text(c)] for c in study.constraints]
    if sup:
        D.table(["Selection", "Support"], sup, caption="Supports", font=8, widths=(0.3, 0.7))
    warnings: list[str] = []
    try:
        mass: dict[str, Any] = {}
        warnings += SL.mass_tables(D, study, mesh, mass)                              # the mass properties and the budget are computed once, by the tables
        rep = mass["budget"]
    except Exception as e:  # noqa: BLE001 - a chapter that cannot be written is NOT_RUN, never a silent PASS
        D.p(f"The mass properties could not be computed: {clean_text(e)}", bold=True)
        ch.status, ch.basis = NOT_RUN, "the mass properties could not be computed"
        ch.items.append(f"the mass properties could not be computed ({clean_text(e)})")
        return warnings + [f"Mass properties: {clean_text(e)}"]
    judged = [i for i in rep.items if i.status != "SKIPPED" and i.limit is not None]            # a quantity without a limit is only reported
    if not judged:
        ch.status, ch.basis = INFO, "reported; no mass or centre-of-mass budget is entered in the requirement set"
    else:
        ch.status = worst([FAIL if i.status == "FAIL" else UNCONFIRMED for i in judged if i.status != "PASS"])
        ch.basis = "mass and centre of mass against the budgets of the requirement set"
        ch.items += [f"{i.name}: {i.status}" for i in judged if i.status != "PASS"]
    return warnings


def _requirements(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    space = getattr(study, "space", None)
    rs = space.requirement_set if space is not None else None
    D.p("The requirement set is the only source of every factor, limit and rule of this report: each entry carries the document, revision and clause it was taken from, and the "
        "name of the person who confirmed the set. The program ships no standard value.")
    if space is None or rs is None:
        D.p("No requirement set is defined for this study: nothing in this report can be a verification result.", bold=True)
        ch.status, ch.basis = NOT_RUN, "no requirement set"
        ch.items.append("no requirement set is defined")
        return []
    SP.requirement_tables(D, space, rs)
    SP.allowable_table(D, study)
    problems: list[str] = []
    if not space.requirement_confirmed():
        problems.append("the requirement set is not confirmed by the user (or it was edited after the confirmation)")
    from ..space.requirements import allowable_of
    mesh = SL._mesh_of(sess)
    for mat in sorted({m for _p, m in _used_materials(study, mesh) if m in study.materials}):
        rec = allowable_of(study.materials[mat])
        if rec is None:
            problems.append(f"material '{mat}': no allowables are entered")
        elif not rec.is_verified():
            problems.append(f"material '{mat}': the allowables are not confirmed and verified (A or B basis, source named)")
    ch.items += problems
    ch.status = UNCONFIRMED if problems else PASS
    ch.basis = "requirement set and the allowables of the used materials confirmed" if not problems else "; ".join(problems[:2])
    return []


def _checks(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    from ..space import model_checks as MC
    run = getattr(sess, "model_checks", None)
    if run is None or not SP.has_checks(sess, study):
        if run is not None and getattr(run, "results", None):
            D.p("The model quality checks kept in this session belong to another study: they are not shown.", bold=True)
            ch.status, ch.basis = OUTDATED, "the kept checks belong to another study"
        else:
            D.p("The seven model quality checks were not run in this session (a run of the checks is not saved in the project file): run them before the report.", bold=True)
            ch.status, ch.basis = NOT_RUN, "the model quality checks were not run"
        ch.items.append(ch.basis)
        return []
    warnings = SP.checks_section(D, sess, sec)
    stale = run.digest != MC.study_digest(study)
    verdict = MC.overall_verdict(run.results)
    ch.status = OUTDATED if stale else {PASS: PASS, "WARN": WARN, "FAIL": FAIL, "INCOMPLETE": PARTIAL, "UNCONFIRMED_REQUIREMENTS": UNCONFIRMED}.get(verdict, UNCONFIRMED)
    ch.basis = ("the study changed after the run of the checks" if stale else f"verdict of the checks: {verdict}")
    if ch.status != PASS:
        ch.items += [f"check {r.number} {r.title}: {r.status}" for r in run.results if r.status != PASS] or [ch.basis]
    return warnings


def _loads(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    from ..space import environments as EV
    from ..space import loads as LD
    if not SL.has_loads(study, sess):
        D.p("No launch environment, quasi-static case or 3-axis result is defined for this study.", italic=True)
        ch.status, ch.basis = NOT_RUN, "no launch loads are defined"
        ch.items.append("no launch loads are defined")
        return []
    D.p("The environments, quasi-static factors and budgets below were entered by the user with their sources; the program contains no launcher or standard value. A result "
        "that rests on an unconfirmed or out-of-date input is labelled so and is not a verification result.")
    mesh = SL._mesh_of(sess)
    warnings: list[str] = []
    found: list[tuple[str, str]] = []                       # (status, what)
    es = EV.environments_of(study.space)
    if es is None:
        found.append((NOT_RUN, "no launch environments are entered"))
    else:
        SL.environment_tables(D, es)
        if not es.is_confirmed():
            found.append((UNCONFIRMED, "the launch environments are not confirmed (or were edited after the confirmation)"))
    st = LD.qsl_status(study)
    SL.qsl_table(D, study)
    if st["names"]:
        if not st["current_with_requirement_set"]:
            found += [(OUTDATED, "quasi-static cases: " + p) for p in (st["problems"][:4] or ["not current with the requirement set"])]
        elif not st["usable_for_verification"]:
            found.append((UNCONFIRMED, "the quasi-static cases rest on an unconfirmed requirement set or environment"))
    elif es is not None:
        found.append((NOT_RUN, "no quasi-static launch cases are generated"))
    res3 = LD.recall(sess, study, mesh)
    kept = LD.kept_result(sess)
    try:
        rep = SL.frequency_table(D, study, sess, res3)
        for it in (rep.items if rep is not None else []):
            if it.status not in ("PASS", "SKIPPED"):
                found.append((FAIL if it.status == "FAIL" else UNCONFIRMED, f"{it.name}: {it.status}"))
        if rep is None:
            found.append((NOT_RUN, "no modal data: the natural frequencies are not available"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Frequency check: {clean_text(e)}")
        found.append((NOT_RUN, f"the frequency check could not be written ({clean_text(e)})"))
    needs3 = es is not None and any(lv.kinds() for lv in es.levels)
    if res3 is not None:
        SL.envelope_tables(D, study, res3, mesh)
        others = [lv.name for lv in (es.levels if es is not None else []) if lv.kinds() and lv.name != res3.level]
        if others:
            D.p(f"The 3-axis result is that of the level '{res3.level}' only; the level(s) {', '.join(others)} also have dynamic tables and have no current result.", bold=True, size=9)
            found.append((PARTIAL, f"only the level '{res3.level}' has a 3-axis result; the level(s) {', '.join(others)} have dynamic tables without a current result"))
        found += [(UNCONFIRMED, b) for b in res3.blockers]
        if not res3.complete:
            found.append((PARTIAL, f"the dynamic analysis is a partial run ({res3.label})"))
        if not res3.confirmed_now(study, mesh):
            found.append((UNCONFIRMED, "the environment or the requirement set of the 3-axis result is not confirmed now"))
    elif kept is not None:
        why = kept.stale_reason(study, mesh) or "no mesh to compare with"
        D.p(f"A 3-axis result kept in this session is OUT OF DATE ({clean_text(why)}) and is not shown: run the 3-axis analysis again.", bold=True)
        found.append((OUTDATED, f"the kept 3-axis result is out of date ({why})"))
    elif needs3:
        D.p("The 3-axis dynamic analysis (sine, random, shock envelopes) was not run in this session (its result is not saved in the project file): run it before the report.", bold=True)
        found.append((NOT_RUN, "the 3-axis dynamic analysis was not run"))
    ch.items += [w for _s, w in found]
    ch.status = worst([s for s, _w in found])
    ch.basis = ((f"environments, quasi-static cases, natural frequencies and the 3-axis envelopes are confirmed and current (level '{res3.level}')" if res3 is not None
                 else "environments, quasi-static cases, natural frequencies and the 3-axis envelopes are confirmed and current") if not found else next(w for s, w in found if s == ch.status))
    return warnings


def _margins(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    from ..space import margins as MG
    from ..space.loads import STALE_3AXIS, sign_off_margins
    if not SP.has_space(study, res):
        D.p("There are no solved results with space settings: the margins of safety cannot be computed.", bold=True)
        ch.status, ch.basis = NOT_RUN, "no solved results"
        ch.items.append("no solved results (or no space settings)")
        return []
    summ = sign_off_margins(study, res, sess, SL._mesh_of(sess))
    D.p("Margin of safety MoS = allowable / (J · σ) − 1, where J is the product of the design factors of the margin rule and σ the equivalent stress of the unfactored load at "
        "the governing place; the governing place is the highest one that is not classified as a singularity. Margins that rest on the 3-axis envelopes are computed from the "
        "envelopes of chapter 5 only while those are current.")
    D.p(f"Verdict of the margins: {summ.verdict_text}", bold=True, size=11, color=(0x1B, 0x7F, 0x3B) if summ.signoff_ready else (0xC0, 0x39, 0x2B))
    SP.margin_tables(D, summ)
    SP.singularity_table(D, summ)
    if summ.open_items:
        D.p("Open items before sign-off:", bold=True)
        for it in summ.open_items[:20]:
            D.bullet(it)
    stale = any(STALE_3AXIS in r for r in summ.reasons)
    if stale:
        ch.status = OUTDATED
    elif summ.verdict == MG.FAIL:
        ch.status = FAIL
    elif summ.verdict == MG.PASS and summ.signoff_ready:
        ch.status = PASS
    else:
        ch.status = UNCONFIRMED
    ch.basis = {PASS: "all margins non-negative, inputs confirmed, every singular place justified", OUTDATED: "a kept 3-axis result is out of date",
                FAIL: "a margin with confirmed inputs is negative"}.get(ch.status, f"verdict of the margins: {summ.verdict_text}")
    ch.items += summ.open_items
    return list(summ.warnings)


S4_SINGLE = ("interface", "rails")                          # sections that hold ONE record; the others (notch, bolts, pcb) hold named records


def _s4_status(study: Any) -> tuple[str, list[str], list[str]]:
    """(status, items, notes) of the stored S4 results, judged by the S4 module itself (`s4_verify.shown`: a stored verdict counts only while its basis holds).
    PASS only when at least one record was judged AND every judged record is a fresh PASS and nothing in the store is unreadable (an unknown section, a section of the wrong
    shape, an empty bucket, a record that is not a dictionary): such a store is UNVERIFIABLE, never a PASS by default. `notes` = what the reader must know about how far the
    records are checked (S4's own note for the records that cannot be recomputed)."""
    s4_store = importlib.import_module("fea_core.space.s4_store")             # the modules of stage S4 (not part of every build: found by name)
    s4_verify = importlib.import_module("fea_core.space.s4_verify")
    data = s4_store.get(study)
    known = tuple(getattr(s4_store, "SECTIONS", ()))
    found: list[tuple[str, str]] = []
    notes: list[str] = []
    judged = 0
    with s4_verify.judging(study):
        for section, rec in data.items():
            if section not in known:
                found.append((UNVERIFIABLE, f"{clean_text(str(section)[:40])}: unknown section of the stored S4 results - it cannot be judged"))
                continue
            if section in S4_SINGLE:
                recs = [(section, rec)]
            elif isinstance(rec, dict) and rec:
                recs = list(rec.items())
            else:
                found.append((UNVERIFIABLE, f"{section}: the stored section is empty or not in the expected form - it cannot be judged"))
                continue
            for name, r in recs:
                if not isinstance(r, dict):
                    found.append((UNVERIFIABLE, f"{section}/{clean_text(str(name)[:40])}: the stored record is not a record - it cannot be judged"))
                    continue
                try:
                    v = s4_verify.shown(study, section, r)
                except Exception as e:  # noqa: BLE001 - a record that cannot be judged is never a PASS
                    found.append((UNVERIFIABLE, f"{section}/{clean_text(str(name)[:40])}: the record could not be judged ({type(e).__name__}: {clean_text(e)})"))
                    continue
                judged += 1
                note = str(getattr(v, "note", "") or "")
                if note and section not in getattr(s4_verify, "RECOMPUTED", ("bolts", "pcb", "rails")):          # notching and interface: the digests judge them, nothing is recomputed
                    notes.append(note)
                if not (v.verdict == "PASS" and v.fresh):
                    found.append(({"FAIL": FAIL, "OUTDATED": OUTDATED}.get(v.verdict, UNCONFIRMED), f"{name}: {v.label if hasattr(v, 'label') else v.verdict}"))
    if judged == 0 and not found:
        found.append((UNVERIFIABLE, "no stored S4 record could be judged"))
    return worst([s for s, _w in found]), [w for _s, w in found], list(dict.fromkeys(notes))


def _interfaces(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    """THE hook of stage S4: the chapter is written by `fea_core.report.space_s4` when the build has that module, else it says so."""
    if importlib.util.find_spec(S4_MODULE) is None:
        D.p("Interfaces, notching, bolted joints and PCB checks (ECSS-E-HB-32-23A): not part of this build of the program - they were not assessed in this report.", italic=True)
        ch.status, ch.basis = NOT_IN_BUILD, NOT_IN_BUILD_REASON
        return []
    mod = importlib.import_module(S4_MODULE)
    if not mod.has_s4(study):
        D.p("No result of the interface, notching, bolt, PCB or rail checks is stored in the study: they were not assessed in this report.", bold=True)
        ch.status, ch.basis = NOT_RUN, "the checks of this chapter were not run"
        ch.items.append(ch.basis)
        return []
    with importlib.import_module("fea_core.space.s4_verify").judging(study):          # the records are judged once: the sections print the verdicts, the chapter summarises them
        warnings = [clean_text(w) for w in mod.s4_sections(D, study, sec)]
        ch.status, found, notes = _s4_status(study)
    if warnings:                                                                 # a section that could not be written was not shown: never a PASS
        ch.status = worst([ch.status, UNVERIFIABLE])
        found = found + [f"a section of this chapter could not be written ({w})" for w in warnings[:3]]
    ch.basis = ("stored results of the interface, notching, bolt, PCB and rail checks, each judged against the present study" if ch.status == PASS else (found[0] if found else ""))
    ch.notes = notes                                                             # S4's own caveat for the records it cannot recompute, printed with the chapter status
    for note in notes:
        D.p(note, italic=True, size=9)
    ch.items += found
    return warnings


def _correlation(D: _CleanDoc, sess: Any, study: Any, res: Any, ch: Chapter, sec: str, o: ReportOptions) -> list[str]:
    from ..space import corr_run as CR
    from . import space_corr as SCR
    if not SCR.has_correlation(study):
        D.p("No vibration test correlation is stored in the study: this chapter is optional and was not part of this analysis.", italic=True)
        ch.status, ch.basis = NOT_INCLUDED, "no stored correlation (optional chapter)"
        return []
    status = CR.verify_stored(study, res)
    warnings = SCR.correlation_section(D, study, sec, res, status)                  # the stored result is verified once
    v = str(status["verdict"]) if status is not None else ""
    ch.status = ({"PASS": PASS, "FAIL": FAIL, "PARTIAL_SCOPE": PARTIAL, "OUTDATED": OUTDATED, "UNVERIFIABLE": UNVERIFIABLE}.get(v, UNCONFIRMED) if status is not None else UNVERIFIABLE)
    if ch.status == PASS and status is not None and not status["current"]:                      # a PASS that could not be verified now is never a PASS here
        ch.status = UNVERIFIABLE
    ch.basis = f"verdict of the correlation: {v or 'none'}"
    if ch.status != PASS and status is not None:
        ch.items += [str(r)[:300] for r in status["reasons"] + status["unverified"]] or [ch.basis]
    return warnings


CHAPTERS: list[tuple[str, str, Callable[..., list[str]]]] = [
    ("configuration", "Configuration and traceability", _configuration), ("model", "Model and mass properties", _model), ("requirements", "Confirmed requirement set", _requirements),
    ("checks", "Model quality checks", _checks), ("loads", "Load cases and envelopes", _loads), ("margins", "Margins of safety and singularity justifications", _margins),
    ("interfaces", "Interfaces, notching, bolts and PCB", _interfaces), ("correlation", "Test correlation", _correlation)]


# ------------------------------------------------------------------------------------------------------------------ open items, assumptions
ASSUMPTIONS = [
    "The analysis is linear elastic with small displacements; the margins of safety use the unfactored stress multiplied by the design factor J of the requirement set. A nonlinear "
    "study is never a PASS.",
    "Joints and contacts are modelled as set up in the study. A linear model does not capture preload loss, gapping or slip: they are covered only where chapter 7 reports them.",
    "Singular places (a re-entrant corner, a point load, a rigid connection) are excluded from the margins and need a justification by the user before sign-off.",
    "Mesh convergence is verified only where the model quality checks of chapter 4 include it; a coarse mesh gives indicative stresses.",
    "The program contains no standard value. Every factor, load level, limit and allowable in this report was entered by the user with its source; where it says 'to be confirmed' the user "
    "must confirm it against the source document before the report is used as a verification.",
    "A stored result (a correlation, a check, an interface assessment) is printed only while its provenance still holds; a result that is out of date is reported as such, never as PASS.",
    "The verdict of this report is the bookkeeping of the program over the chapters that are included. It does not replace the review and the approval by the responsible engineer.",
]


def _is_status_line(par: Any) -> bool:
    """A paragraph that states a status or a reason: a bullet, or a paragraph written in bold (every chapter writes its verdict and its warnings so); plain and italic text is prose."""
    if par.style is not None and str(par.style.name).startswith("List"):
        return True
    return any(r.bold for r in par.runs if r.text.strip())


def _marker_counts(tab: Any, markers: tuple[str, ...]) -> dict[str, int]:
    """How many layout-grid cells of the table carry each marker (the first marker found in a cell counts): the numbers `for row in tab.rows: for cell in row.cells` gives - a
    merged cell counts once per grid column it spans, a vertically merged one once per row - but read from the XML, where the text of a cell is read once, not once per marker
    and per grid cell (python-docx builds a cell, its paragraphs and its runs every time)."""
    flagged: dict[str, int] = {}
    hit: dict[Any, str | None] = {}                                           # the marker of a w:tc (a merged cell is read once)
    for tr in tab._tbl.tr_lst:
        for tc in tr.tc_lst:
            root = tc
            while root.vMerge == "continue":                                  # the text of a vertically merged cell is that of the cell that starts the merge
                root = root._tc_above
            if root not in hit:
                text = "\n".join(p.text for p in root.p_lst)                   # = _Cell.text
                hit[root] = next((m for m in markers if m in text), None)
            m = hit[root]
            if m is not None:
                flagged[m] = flagged.get(m, 0) + root.grid_span
    return flagged


def _scan_open_items(D: _CleanDoc, chapters: list[Chapter]) -> tuple[list[str], list[str]]:
    """(text items, table counts): the status lines (bullets and bold paragraphs, not the prose) and the table cells of the chapters that carry a flag (UNCONFIRMED, OUTDATED,
    PARTIAL_SCOPE, 'to be confirmed' ...)."""
    paragraphs, tables = D.doc.paragraphs, D.doc.tables
    seen: set[str] = set()
    texts: list[str] = []
    counts: list[str] = []
    for c in chapters:
        p0, p1, t0, t1 = c.span
        for par in paragraphs[p0:p1]:
            if not _is_status_line(par):                                  # the introduction and the explanations of a chapter are prose, not a finding
                continue
            t = " ".join(par.text.split())
            if t and _MARKERS.search(t) and t not in seen and not t.startswith("Table "):
                seen.add(t)
                texts.append(f"{c.number} {c.title}: {t[:300]}")
        flagged: dict[str, int] = {}
        for tab in tables[t0:t1]:
            for m, n in _marker_counts(tab, _TABLE_MARKERS).items():
                flagged[m] = flagged.get(m, 0) + n
        if flagged:
            counts.append(f"{c.number} {c.title}: " + ", ".join(f"{n} table cell(s) marked {m}" for m, n in sorted(flagged.items())))
    return texts, counts


def _open_items_chapter(D: _CleanDoc, chapters: list[Chapter], warnings: list[str], sec: str) -> list[str]:
    D.p("Everything that is not confirmed, not current, not complete or not yet assessed, collected from every chapter above, and the assumptions and limitations of the analysis.")
    D.h(f"{sec}.1 Open items", 2)
    structured = []
    for c in chapters:
        if not c.included or c.passed:
            continue
        items = c.items or [c.basis]
        walked = [i for i in items if _HIGHEST_PLACES.match(i)]
        loose = [i for i in items if not i.startswith(SINGULARITY_OPEN) and i not in walked]
        many = [i for i in items if i.startswith(SINGULARITY_OPEN)]
        if walked:                                                        # "the N highest places are all singularities" once per case: one line
            loose.append(f"for {len(walked)} case(s) and check(s) the highest places are all singularities - the margin outside singularities is undetermined there")
        if many:                                                          # one line for the many singular places (the table of chapter 6 lists every one)
            loose.append(f"{len(many)} singular place(s) are not justified by the user (listed in the table of chapter 6); first: " + many[0][len(SINGULARITY_OPEN):].strip(" :"))
        structured += [f"{c.number} {c.title} - {c.status}: {i}" for i in loose]
    texts, counts = _scan_open_items(D, chapters)
    items = list(dict.fromkeys(structured + counts + texts))
    if items:
        for it in items[:MAX_OPEN_ITEMS]:
            D.bullet(it)
        if len(items) > MAX_OPEN_ITEMS:
            D.p(f"{len(items) - MAX_OPEN_ITEMS} more open items are not printed.", italic=True, size=9)
    else:
        D.p("No open item was found in the included chapters.", italic=True)
    gone = [c for c in chapters if not c.included]
    for c in gone:
        D.bullet(f"{c.number} {c.title}: {c.basis} - not assessed in this report.")
    D.h(f"{sec}.2 Assumptions and limitations", 2)
    for a in ASSUMPTIONS:
        D.bullet(a)
    if warnings:
        D.h(f"{sec}.3 Notes of the program while writing the report", 2)
        for w in list(dict.fromkeys(warnings))[:30]:
            D.bullet(w)
    return items


# ------------------------------------------------------------------------------------------------------------------ the document
def _summary(D: _CleanDoc, chapters: list[Chapter], verdict: str, blockers: list[Chapter]) -> None:
    D.h("Sign-off summary", 1)
    color = {PASS: (0x1B, 0x7F, 0x3B), FAIL: (0xC0, 0x39, 0x2B)}.get(verdict, (0xB3, 0x6B, 0x00))
    D.p(f"Overall verdict: {verdict}", bold=True, size=13, color=color)
    if verdict == PASS:
        D.p("Every included chapter is current and PASS. See the scope line below for what is not covered by this report.", italic=True)
    elif verdict == FAIL:
        D.p("At least one chapter reports a failed check or a negative margin of safety.", italic=True)
    else:
        D.p("This report is NOT a sign-off: " + "; ".join(f"{c.title} ({c.status}" + (f": {c.basis}" if c.status == NOT_IN_BUILD else "") + ")" for c in blockers[:8])
            + (" ..." if len(blockers) > 8 else "") + ". "
            "Nothing that is stale, unconfirmed, incomplete or not yet assessed is turned into PASS.", italic=True)
    D.p("Status of the chapters", italic=True, size=9)                                    # no table number: the summary is written last but stands first
    D.table(["No.", "Chapter", "Status", "Basis"], [[str(c.number), c.title, c.status, "; ".join([_tr(c.basis or "–")] + [_tr(n) for n in c.notes])] for c in chapters], status_col=2, font=9, widths=(0.06, 0.34, 0.2, 0.4))
    gone = [c for c in chapters if not c.included]
    D.p("Scope: " + ("not covered by this report - " + "; ".join(f"{c.title} ({c.basis})" for c in gone) + "." if gone else "every chapter of the ECSS structure is included."), size=9)


def _move_before(D: _CleanDoc, anchor: Any, first: int) -> None:
    """Moves the body elements written after index `first` (the sign-off summary written last) in front of the paragraph `anchor`."""
    body = D.doc.element.body
    for el in list(body)[first:-1]:                                     # the last child is the section properties
        anchor._p.addprevious(el)                                       # noqa: SLF001 - python-docx has no public "insert before"


def freeze_docx(path: Path, when: tuple[int, int, int, int, int, int] = (2000, 1, 1, 0, 0, 0)) -> None:
    """Rewrites the .docx with a fixed date of every member (python-docx stamps the time of the save into the zip), so that equal content gives equal bytes."""
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=when)
            zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, info.external_attr
            zout.writestr(zi, zin.read(info.filename))
    tmp.replace(path)


def build_sar(sess: Any, path: str | Path, options: ReportOptions | None = None, on_log: Callable[[str], None] | None = None, generated: str | None = None) -> dict[str, Any]:
    """Writes the Structural Analysis Report (ECSS) of the session and returns {"path", "verdict", "chapters", "open_items", "traceability", "warnings", "tables"}.
    `generated` (default: the present UTC time) is the only text that differs between two runs on the same inputs (the rest is deterministic)."""
    o = options or ReportOptions()
    log = on_log or (lambda s: None)
    study = sess.require_study() if hasattr(sess, "require_study") else sess.study
    path = Path(path)
    if path.suffix.lower() != ".docx":
        path = path.with_suffix(".docx")
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = generated or time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    with i18n.using(o.language or None):
        return _build(sess, study, path, o, log, stamp)


def _build(sess: Any, study: Any, path: Path, o: ReportOptions, log: Callable[[str], None], stamp: str) -> dict[str, Any]:
    res = getattr(sess, "results", None)
    D = _CleanDoc()
    title = o.title or f"Structural Analysis Report (ECSS) – {study.name}"
    D.footer(f"{title} · FEA Studio {FEA_VERSION}")
    D.doc.add_heading(_tr(clean_paths(title)), 0)
    rs = getattr(getattr(study, "space", None), "requirement_set", None)
    meta = [("Document", "Structural Analysis Report (ECSS)"), ("Study", study.name), ("Verification approach", (rs.verification_approach if rs is not None else "") or "not set"),
            ("Requirement set", f"{rs.name or '–'} ({rs.standard or '–'}, revision {rs.revision or '–'})" if rs is not None else "none"), ("Generated (UTC)", stamp),
            ("FEA Studio", FEA_VERSION)]
    if o.author:
        meta.insert(2, ("Prepared by", o.author))
    if o.company:
        meta.insert(3, ("Company", o.company))
    D.kv(meta)
    if o.subject:
        D.p(o.subject)
    anchor = D.p("")
    chapters: list[Chapter] = []
    warnings: list[str] = []
    facts: dict[str, str] = {}
    for n, (key, name, fn) in enumerate(CHAPTERS, 1):
        ch = Chapter(key, name, number=n)
        sec = str(n)
        D.h(f"{sec} {name}", 1)
        log(f"SAR: {name}")
        p0, t0 = len(D.doc.paragraphs), len(D.doc.tables)
        try:
            warnings += fn(D, sess, study, res, ch, sec, o)
        except Exception as e:  # noqa: BLE001 - a chapter that fails is NOT_RUN and blocks the sign-off; the others are still written
            text = f"{type(e).__name__}: {clean_text(e)}"
            D.p(f"This chapter could not be written ({text}).", bold=True)
            ch.status, ch.basis = NOT_RUN, "the chapter could not be written"
            ch.items.append(f"the chapter could not be written ({text})")
            warnings.append(f"{name}: {text}")
        ch.span = (p0, len(D.doc.paragraphs), t0, len(D.doc.tables))
        facts = facts or ch.facts
        chapters.append(ch)
    n = len(chapters) + 1
    log("SAR: assumptions, limitations and open items")
    D.h(f"{n} Assumptions, limitations and open items", 1)
    p0, t0 = len(D.doc.paragraphs), len(D.doc.tables)
    items = _open_items_chapter(D, chapters, warnings, str(n))
    last = Chapter("open_items", "Assumptions, limitations and open items", INFO, "collected from the chapters above", number=n, span=(p0, len(D.doc.paragraphs), t0, len(D.doc.tables)))
    verdict, blockers = overall_verdict(chapters)
    first = len(list(D.doc.element.body)) - 1
    _summary(D, chapters + [last], verdict, blockers)
    _move_before(D, anchor, first)
    cp = D.doc.core_properties
    cp.title, cp.subject, cp.author, cp.last_modified_by, cp.comments = clean_paths(title), clean_paths(study.name), clean_paths(o.author) or "FEA Studio", "FEA Studio", ""
    import datetime as dt
    try:
        when = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M")
    except ValueError:
        when = dt.datetime(2000, 1, 1)
    cp.created = cp.modified = when
    D.doc.save(str(path))
    freeze_docx(path)
    log(f"Structural Analysis Report saved: {path.name}")
    return {"path": str(path), "verdict": verdict, "blocking": [c.title for c in blockers], "chapters": [c.to_dict() for c in chapters], "open_items": items,
            "traceability": facts, "warnings": list(dict.fromkeys(warnings)), "figures": D.fig, "tables": D.tab, "images": []}


__all__ = ["build_sar", "overall_verdict", "Chapter", "clean_text", "freeze_docx", "ccx_version", "traceability"]

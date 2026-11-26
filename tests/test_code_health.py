"""Structural guardrails of the code base (F16 / M3): keep the GUI from regrowing into a monolith, keep the shelf wiring and the
compatibility surface of `fea_gui.main` intact, forbid unexplained type-checker suppressions and keep the repository tidy.

Everything is static (ast / text / `git ls-files`), nothing is run, and no Qt is needed, except the one test that really imports
`fea_gui.main` (it skips where PySide6 is not installed).

Plan: docs/projekt/F16_plan_1_4_4.md, section M3. The type checker itself is `basedpyright` (pyproject.toml, CI job "typecheck").
"""
from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUI = ROOT / "fea_gui"
MAIN = GUI / "main.py"
MAIN_DEFS = GUI / "main_defs.py"

MAIN_MAX_LOC = 700                  # fea_gui/main.py: slim coordinator after the split (P0.1)
MIXIN_MAX_LOC = 1600                # every fea_gui/main_*.py
MODULE_MAX_LOC = 2500               # any other module of fea_core / fea_gui: a new feature gets its own module, not a bigger one
CLAUDE_MAX_LINES = 250              # CLAUDE.md stays a short entry point, history lives in docs/projekt/CLAUDE_archive.md

# the names that panels, dialogs and tests import from fea_gui.main (the compatibility surface of the split, F16 / M1)
REQUIRED_MAIN_NAMES = {
    "MainWindow", "main", "APP_NAME", "STAGES", "STAGE", "PHASES", "PHASE_OF", "OPTIONAL_STAGES", "FRAME_ONLY_OFF", "QUANTITIES",
    "THERMAL_QUANTITIES", "FORCE", "ROTATION", "THERMAL_LOAD", "CAD_SOURCE", "FRAME_MEMBER", "FRAME_NODE", "GREEN", "GREY", "ORANGE",
    "RED", "NearestPoints", "dot_icon", "msg", "phase_label",
}

# (step, target) pairs of SHELF entries that are known to do nothing and are tolerated for now; empty since P0.1 repaired the two
# leftovers of the split of fea_gui/main.py ("stage:Kontakty" -> "stage:Contacts", "@Vektory σ..." -> "@Stress vectors...").
KNOWN_SHELF_LEFTOVERS: set[tuple[str, str]] = set()

BARE_IGNORE = re.compile(r"#\s*(?:type|pyright|basedpyright)\s*:\s*ignore(?!\s*\[)")
SCANNED_FOR_IGNORE = ("fea_core", "fea_gui", "tests", "examples", "adapters", "installer", "verification")


def _loc(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _py_files(*folders: str) -> list[Path]:
    out: list[Path] = []
    for f in folders:
        out += [p for p in (ROOT / f).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def _git_files(*args: str) -> list[str] | None:
    """`git ls-files` of the checkout, or None where there is no repository / no git (a release package, a source archive)."""
    if not (ROOT / ".git").exists():
        return None
    try:
        r = subprocess.run(["git", "ls-files", "-z", *args], cwd=ROOT, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [x.decode("utf-8", "replace") for x in r.stdout.split(b"\0") if x]


# --------------------------------------------------------------------------------------------------- file sizes

def test_main_py_stays_slim():
    assert MAIN_DEFS.exists(), "fea_gui/main_defs.py (constants and helpers of the main window) is missing"
    n = _loc(MAIN)
    assert n <= MAIN_MAX_LOC, f"fea_gui/main.py has {n} lines (limit {MAIN_MAX_LOC}): put the new code into its own module / mixin"


def test_main_mixins_not_oversized():
    big = {p.name: _loc(p) for p in sorted(GUI.glob("main_*.py")) if _loc(p) > MIXIN_MAX_LOC}
    assert not big, f"fea_gui/main_*.py over {MIXIN_MAX_LOC} lines: {big}"


def test_no_new_giant_modules():
    big = {str(p.relative_to(ROOT)).replace("\\", "/"): _loc(p) for p in _py_files("fea_core", "fea_gui")
           if p != MAIN and not p.name.startswith("main_") and _loc(p) > MODULE_MAX_LOC}
    assert not big, f"modules over {MODULE_MAX_LOC} lines (split them, do not grow them): {big}"


# --------------------------------------------------------------------------------------------------- compatibility surface

def _main_importers() -> dict[str, set[str]]:
    """Names that other modules of the GUI, the GUI smoke tests and the tests take from `fea_gui.main`: name -> files."""
    found: dict[str, set[str]] = {}
    for path in _py_files("fea_gui", "tests"):
        if path == MAIN or path.name == Path(__file__).name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))   # utf-8-sig: some tests start with a BOM; a parse error must fail, not hide a file
        aliases: set[str] = set()                       # `import fea_gui.main as gm` / `from fea_gui import main as gm`
        in_gui = GUI in path.parents
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if (in_gui and node.level == 1 and node.module == "main") or (node.level == 0 and node.module == "fea_gui.main"):
                    for a in node.names:
                        if a.name != "*":
                            found.setdefault(a.name, set()).add(path.name)
                elif (node.level == 0 and node.module == "fea_gui") or (in_gui and node.level == 1 and node.module is None):
                    aliases.update(a.asname or a.name for a in node.names if a.name == "main")
            elif isinstance(node, ast.Import):
                aliases.update(a.asname for a in node.names if a.name == "fea_gui.main" and a.asname)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases and isinstance(node.ctx, ast.Load):
                found.setdefault(node.attr, set()).add(path.name)
    return found


def _module_level_names(path: Path, seen: set[Path] | None = None) -> set[str]:
    """Names a module defines or imports at its top level (a star import is followed to the sibling module)."""
    seen = seen or set()
    if path in seen or not path.exists():
        return set()
    seen.add(path)
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        nodes = [node]
        if isinstance(node, (ast.Try, ast.If)):                # try: import ... except ImportError / if TYPE_CHECKING
            nodes += list(ast.walk(node))
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Assign):
                names.update(t.id for t in n.targets if isinstance(t, ast.Name))
                for t in n.targets:
                    if isinstance(t, ast.Tuple):
                        names.update(e.id for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name == "*":
                        if n.level == 1 and n.module:
                            names |= _module_level_names(path.parent / (n.module.replace(".", "/") + ".py"), seen)
                    else:
                        names.add(a.asname or a.name)
            elif isinstance(n, ast.Import):
                names.update((a.asname or a.name).split(".")[0] for a in n.names)
    return names


def test_main_compatibility_surface_static():
    """Every name imported from fea_gui.main (by the panels and by the smoke tests) is still defined / re-exported there."""
    have = _module_level_names(MAIN)
    imported = _main_importers()
    missing = {n: sorted(f) for n, f in imported.items() if n not in have}
    assert not missing, f"fea_gui.main no longer provides: {missing}"
    lost = sorted(REQUIRED_MAIN_NAMES - have)
    assert not lost, f"fea_gui.main must keep exporting {lost}"


def test_main_compatibility_surface_import():
    pytest.importorskip("PySide6")
    import fea_gui.main as gm
    lost = sorted(n for n in REQUIRED_MAIN_NAMES | set(_main_importers()) if not hasattr(gm, n))
    assert not lost, f"import fea_gui.main lacks {lost}"


# --------------------------------------------------------------------------------------------------- tool shelf

def _const_strings() -> set[str]:
    out: set[str] = set()
    for p in _py_files("fea_gui"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.add(n.value)
    return out


def _attribute_names() -> set[str]:
    """Everything that can be a window / panel attribute: methods, classes, `self.x = ...` stores, class-body assignments."""
    out: set[str] = set()
    for p in _py_files("fea_gui"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
                out.add(n.attr)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
    return out


def _module_assign(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    return None


def _defs_tree() -> ast.Module:
    """The module with STAGES / SHELF: main_defs.py (the constants of the main window, split from main.py in P0.1)."""
    return ast.parse(MAIN_DEFS.read_text(encoding="utf-8"))


def _shelf() -> tuple[dict, list[str]]:
    tree = _defs_tree()
    consts: dict[str, object] = {}
    for node in tree.body:                                  # module-level `NAME = "literal"` constants used inside SHELF
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    shelf_node = _module_assign(tree, "SHELF")
    assert shelf_node is not None, "SHELF not found"
    shelf = eval(compile(ast.Expression(shelf_node), "<SHELF>", "eval"), {"__builtins__": {}}, consts)
    stages = consts["STAGES"]
    assert isinstance(stages, list)
    return shelf, stages


def _icon_names() -> set[str]:
    node = _module_assign(ast.parse((GUI / "theme.py").read_text(encoding="utf-8")), "ICONS")
    assert isinstance(node, ast.Dict)
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _shelf_problems() -> list[tuple[str, str, str]]:
    """(step, target, what is wrong) for every SHELF entry that cannot work (MainWindow._shelf_run ignores such a target silently)."""
    shelf, stages = _shelf()
    attrs, texts, icons = _attribute_names(), _const_strings(), _icon_names()
    bad: list[tuple[str, str, str]] = []

    def check_path(step: str, target: str, path: str):
        for part in path.split("."):
            if part not in attrs:
                bad.append((step, target, f"no attribute or method '{part}' anywhere in fea_gui"))
                return

    for step, entries in shelf.items():
        if step not in stages:
            bad.append((step, "", "SHELF step is not in STAGES"))
        for e in entries:
            if e[0] == "#":
                continue
            if len(e) != 4:
                bad.append((step, repr(e), "entry is not (label, icon, target, description)"))
                continue
            label, icon, target, _tip = e
            if icon not in icons:
                bad.append((step, str(target), f"icon '{icon}' is not in theme.ICONS (button '{label}')"))
            if isinstance(target, tuple):                   # (combo attribute, item text)
                check_path(step, str(target), target[0])
                if target[1] not in texts:
                    bad.append((step, str(target), f"item text '{target[1]}' is not a string literal of fea_gui"))
            elif target.startswith("@"):
                if target[1:] not in texts:
                    bad.append((step, target, "no button text like this in fea_gui"))
            elif ":" in target:
                kind, path = target.split(":", 1)
                if kind == "stage":
                    if path not in stages:
                        bad.append((step, target, "not a step of STAGES"))
                elif kind in ("focus", "popup", "raise", "menu"):
                    check_path(step, target, path)
                else:
                    bad.append((step, target, f"unknown target kind '{kind}'"))
            else:
                check_path(step, target, target)
    return bad


def test_shelf_entries_resolve():
    """Every SHELF entry points to an existing step, icon, attribute / method or button text - except the entries listed in
    KNOWN_SHELF_LEFTOVERS (none at the moment)."""
    bad = [b for b in _shelf_problems() if (b[0], b[1]) not in KNOWN_SHELF_LEFTOVERS]
    assert not bad, "SHELF entries that do nothing:\n" + "\n".join(f"  {s}: {t} - {w}" for s, t, w in bad)


def test_shelf_has_no_known_leftovers():
    """No SHELF entry at all is a dead button (the exemption list above is empty)."""
    bad = _shelf_problems()
    assert not bad, f"SHELF: {bad}"


def test_known_shelf_leftovers_are_current():
    """The exemption list must not outlive the bugs: an entry that is not in SHELF any more is removed from KNOWN_SHELF_LEFTOVERS."""
    present = {(b[0], b[1]) for b in _shelf_problems()}
    stale = sorted(KNOWN_SHELF_LEFTOVERS - present)
    assert not stale, f"KNOWN_SHELF_LEFTOVERS lists entries that are fine now - delete them: {stale}"


# --------------------------------------------------------------------------------------------------- type checker

def test_no_bare_type_ignore():
    """Every suppression of the type checker names the rule: `# type: ignore[attr-defined]  # reason`, `# pyright: ignore[rule]`."""
    bad: list[str] = []
    for p in _py_files(*SCANNED_FOR_IGNORE):
        if p == Path(__file__):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if BARE_IGNORE.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, "bare type-ignore comments (add the rule code in brackets and a short reason): " + ", ".join(bad)


def test_type_checker_is_configured():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["basedpyright"]
    assert cfg["typeCheckingMode"] == "basic" and cfg["pythonVersion"] == "3.11"
    assert {"fea_core", "fea_gui"} <= set(cfg["include"])
    if (ROOT / ".git").exists():                        # release packages leave .basedpyright/ out (make_package EXCLUDE_PREFIX)
        assert (ROOT / ".basedpyright" / "baseline.json").is_file(), "the baseline of the type checker is missing"
    wf = ROOT / ".github" / "workflows" / "tests.yml"
    if wf.exists():
        assert re.search(r"^  typecheck:\s*$", wf.read_text(encoding="utf-8"), re.M), "CI job 'typecheck' is missing"


# --------------------------------------------------------------------------------------------------- typed host of the mixins

HOST = GUI / "main_host.py"


def _mixin_files() -> list[Path]:
    """The modules whose classes `MainWindow` is put together from: chrome.py, ipc_bridge.py and every fea_gui/main_*.py mixin."""
    skip = {"main_defs.py", "main_host.py"}
    return [GUI / "chrome.py", GUI / "ipc_bridge.py"] + [p for p in sorted(GUI.glob("main_*.py")) if p.name not in skip]


def _components() -> list[ast.ClassDef]:
    out: list[ast.ClassDef] = []
    for p in [MAIN] + _mixin_files():
        out += [n for n in ast.parse(p.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and (n.name == "MainWindow" or n.name.endswith("Mixin"))]
    return out


def test_mixins_derive_from_the_typed_host():
    """The mixins see the window and each other through `MainWindowHost` (type checking only): the base is `_Base`, which is the host
    for the checker and `object` at run time. A catch-all `__getattr__` (what it replaced) turns every self.<attr> into Any."""
    bad: list[str] = []
    for p in _mixin_files():
        text = p.read_text(encoding="utf-8")
        mixins = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name.endswith("Mixin")]
        if not mixins:
            bad.append(f"{p.name}: no mixin class")
        for c in mixins:
            if [ast.unparse(b) for b in c.bases] != ["_Base"]:
                bad.append(f"{p.name}: class {c.name} must derive from `_Base` only")
        if "from .main_host import MainWindowHost as _Base" not in text or not re.search(r"^else:\n    _Base = object$", text, re.M):
            bad.append(f"{p.name}: `if TYPE_CHECKING: ... MainWindowHost as _Base / else: _Base = object` is missing")
        if "def __getattr__" in text:
            bad.append(f"{p.name}: a `__getattr__` hides every missing member from the type checker - declare it in main_host.py")
        if "reportIncompatibleMethodOverride=error" not in text:
            bad.append(f"{p.name}: `# pyright: reportIncompatibleMethodOverride=error` (drift of the host declarations) is missing")
    assert not bad, "; ".join(bad)


def test_host_module_is_imported_for_type_checking_only():
    """fea_gui/main_host.py costs nothing at start-up only while nothing imports it at run time (tests/test_ram_idle.py)."""
    bad: list[str] = []
    for p in _py_files("fea_gui", "tests"):
        if p in (HOST, Path(__file__)):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        guarded = {id(n) for g in ast.walk(tree) if isinstance(g, ast.If) and ast.unparse(g.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
                   for st in g.body for n in ast.walk(st)}
        for n in ast.walk(tree):
            names = ([n.module or ""] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names] if isinstance(n, ast.Import) else [])
            if any(x.split(".")[-1] == "main_host" for x in names) and id(n) not in guarded:
                bad.append(f"{p.relative_to(ROOT)}:{n.lineno}")
    assert not bad, f"main_host is imported at run time (only `if TYPE_CHECKING:` is allowed): {bad}"


def test_host_declares_only_what_the_window_has():
    """Every member of `MainWindowHost` is still defined by `MainWindow` or one of the mixins (a method or class attribute, or a
    `self.x = ...` assignment): a renamed or removed member must not live on as a stale declaration that the checker trusts."""
    host = next(n for n in ast.parse(HOST.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "MainWindowHost")
    declared: set[str] = set()
    for st in host.body:
        if isinstance(st, ast.FunctionDef):
            declared.add(st.name)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            declared.add(st.target.id)
        elif isinstance(st, ast.Assign):
            declared.update(t.id for t in st.targets if isinstance(t, ast.Name))
    assert len(declared) > 200, "the host declarations were not found"
    provided: set[str] = set()
    for c in _components():
        for st in c.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                provided.add(st.name)
            elif isinstance(st, ast.Assign):
                provided.update(t.id for t in st.targets if isinstance(t, ast.Name))
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                provided.add(st.target.id)
        provided.update(n.attr for n in ast.walk(c) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                        and isinstance(n.value, ast.Name) and n.value.id == "self")
    stale = sorted(declared - provided)
    assert not stale, f"declared in fea_gui/main_host.py but defined nowhere in the window: {stale}"


# --------------------------------------------------------------------------------------------------- repository hygiene

def test_stepview_is_gone():
    """The read-only copy of a standalone viewer was deleted in P0.3 (git history keeps it)."""
    assert not (ROOT / "stepview").exists() or not any((ROOT / "stepview").rglob("*.py"))
    tracked = _git_files("stepview")
    assert not tracked, f"tracked files under stepview/: {tracked[:3]}"


def test_czech_spec_copy_is_not_tracked():
    """The Czech root copy of the specification holds identifiers and never enters git; docs/projekt/FEA_studio_zadani.txt governs."""
    tracked = _git_files("FEA_studio_zadani.txt")
    if tracked is None:
        pytest.skip("not a git checkout")
    assert tracked == [], "FEA_studio_zadani.txt in the repository root must not be tracked"


def test_readme_plasticity_is_in_the_gui():
    """Plasticity has been in the GUI and in the report since 1.4.1 (it was 'Python API / MCP only' before)."""
    en = (ROOT / "README.md").read_text(encoding="utf-8")
    cs = (ROOT / "README_cs.md").read_text(encoding="utf-8")
    assert "Python API / MCP only" not in en and "jen Python API / MCP" not in cs


def test_claude_md_is_a_short_entry_point():
    claude, archive = ROOT / "CLAUDE.md", ROOT / "docs" / "projekt" / "CLAUDE_archive.md"
    if not claude.exists() or not archive.parent.exists():
        pytest.skip("CLAUDE.md / docs/projekt are not part of a release package or of the public export")
    if not archive.exists():
        pytest.xfail("CLAUDE.md has not been split into the short entry point + docs/projekt/CLAUDE_archive.md yet")
    n = _loc(claude)
    assert n <= CLAUDE_MAX_LINES, f"CLAUDE.md has {n} lines (limit {CLAUDE_MAX_LINES}): archive finished milestones in docs/projekt/CLAUDE_archive.md"


def test_view_calls_resolve_to_feaview_methods():
    """Every `self.view.<name>(...)` call of the main window names an attribute of FeaView (MCP fea_gui_highlight once called a missing `select_faces`)."""
    known: set[str] = set()
    for fname, cname in (("view.py", "FeaView"), ("viewer3d.py", "OccView")):     # FeaView(OccView)
        tree = ast.parse((GUI / fname).read_text(encoding="utf-8-sig"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cname)
        known |= {n.name for n in ast.walk(cls) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        known |= {t.attr for n in ast.walk(cls) if isinstance(n, (ast.Assign, ast.AnnAssign))
                  for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Attribute)}
        known |= {t.id for n in cls.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    missing = []
    for path in [*GUI.glob("main*.py"), GUI / "chrome.py", GUI / "ipc_bridge.py"]:   # the main window, where self.view is the FeaView
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            f = n.func if isinstance(n, ast.Call) else None
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and f.value.attr == "view"
                    and isinstance(f.value.value, ast.Name) and f.value.value.id == "self"
                    and f.attr not in known and not f.attr.startswith("__")):
                missing.append(f"{path.name}:{n.lineno} view.{f.attr}")
    qt_base = {"update", "setFocus", "repaint", "width", "height", "grab", "setCursor", "unsetCursor", "isVisible", "rect",
               "mapFromGlobal", "mapToGlobal", "devicePixelRatioF", "setMinimumSize", "show", "hide", "setParent", "winId"}
    assert not [m for m in missing if m.rsplit(".", 1)[1] not in qt_base], missing

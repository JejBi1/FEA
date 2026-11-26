"""No silently swallowed exceptions in the GUI.

An `except Exception:` / bare `except:` whose body only passes, continues or returns a constant hides real errors (the
user sees nothing, the developer has no trace). Such handlers must record the error – `widgets.report_soft` /
`widgets.soft_fail` (Messages dock + log), `win.warn`, `log_msg` – or catch a narrower exception type.

Intentional cosmetic cases are listed in ALLOWED as "file:function" with a reason.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "fea_gui"

# file:function → reason (each one checked: failure only affects cosmetics or a best-effort cleanup)
ALLOWED: dict[str, str] = {
    "__main__.py:_app_id": "Windows AppUserModelID – without it only the taskbar icon differs",
    "__main__.py:_splash": "application icon at start-up – cosmetic, must not stop the start",
    "__main__.py:_diagnostics": "installation of the crash log at start-up – must not stop the start",
    "about.py:write_crash_log": "the crash handler must never raise (None = the log could not be written)",
    "about.py:_show_box": "message box of the crash handler – best effort",
    "about.py:install_diagnostics": "log folder / faulthandler setup – without them the program still runs (best effort)",
    "about.py:hook": "forwarding to the previous excepthook inside the crash handler – must never raise",
    "about.py:thook": "stderr copy of the thread crash handler – must never raise",
    "session_files.py:_create_time": "creation time of a process for the autosave check – None = process unknown / gone",
    "setup_confirm.py:clear_numbers": "removal of the number labels of the setup review from the 3D window while the dialog closes – best effort (the window may be closing)",
    "beam_calc.py:_section_edit": "completer of section names – typing still works without it",
    "frame_panel.py:member_axes": "axes of a member for the 3D preview – the preview simply has no axes",
    "main_defs.py:run": "import prewarm thread – the real import error shows up when the feature is used",
    "main.py:showEvent": "3D mouse raw-input registration – no device / driver is the normal case",
    "main.py:nativeEvent": "3D mouse message on every native event – reporting would flood, no user impact",
    "main.py:closeEvent": "best-effort IPC shutdown while the window closes",
    "spacemouse.py:__init__": "3D mouse focus tracking – optional",
    "view.py:_show_scale": "font / colour of the colour scale – appearance only",
    "viewer3d.py:world_per_pixel": "scale of the view during resize / before the first paint – 0 = not known yet",
    "voxel_dialog.py:box_corners": "rotation of the preview box – falls back to an axis-aligned box",
}


def _broad(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(isinstance(n, ast.Name) and n.id in ("Exception", "BaseException") for n in names)


def _silent(body: list[ast.stmt]) -> bool:
    for st in body:
        if isinstance(st, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):      # docstring / Ellipsis
            continue
        if isinstance(st, ast.Return) and (st.value is None or isinstance(st.value, ast.Constant)):
            continue
        return False
    return True


def _scan():
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
        stack: list[str] = []

        class V(ast.NodeVisitor):
            def _fn(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()
            visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _fn

            def visit_ExceptHandler(self, node):
                if _broad(node) and _silent(node.body):
                    fn = next((s for s in reversed(stack)), "<module>")
                    found.append((f"{path.relative_to(ROOT).as_posix()}:{fn}", node.lineno))
                self.generic_visit(node)
        V().visit(tree)
    return found


def test_no_silent_broad_except():
    bad = [f"fea_gui/{k} (line {n})" for k, n in _scan() if k not in ALLOWED]
    assert not bad, ("Silently swallowed exceptions – record them (widgets.report_soft / soft_fail, win.warn) or catch a "
                     "narrower type, or add an allowlist entry with a reason:\n  " + "\n  ".join(bad))


def test_allowlist_entries_still_exist():
    present = {k for k, _ in _scan()}
    stale = [k for k in ALLOWED if k not in present]
    assert not stale, f"Allowlist entries without a matching handler (remove them): {stale}"

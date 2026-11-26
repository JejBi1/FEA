"""Start-up order of `python -m fea_gui`: the splash window is shown BEFORE the heavy libraries (numpy, OpenCascade, the core modules) are imported.

Measured (offscreen, real launch order, min of 8 runs): importing numpy before the crash-log set-up (which imports PySide6) saves ~7 MB of resident memory (the shiboken
feature hook keeps the source of every module imported after PySide6) but the splash appears 0.35 s later, and the whole list of core modules 2 s later, with the window
ready no sooner. The order is therefore a decision, and this test keeps it: whoever puts a heavy import in front of the splash has to change the test and show the numbers.
The test runs the top of `fea_gui/__main__.py` itself (everything before `from .main import main`), so it follows the real launch, not a re-implementation of it."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("PySide6")

HEAVY = ("numpy", "OCP", "scipy", "numba", "matplotlib", "mcp", "pandas", "psutil", "fea_core.solve", "fea_core.geometry", "fea_core.study", "fea_gui.main")

# runs what `python -m fea_gui` does up to the splash (the module source cut at the import of the window), then reports what is imported and whether the splash exists
CODE = r"""
import sys
src = open(sys.argv[1], encoding="utf-8").read()
cut = src.index("from .main import main")
g = {"__name__": "fea_gui.__main__", "__package__": "fea_gui", "__file__": sys.argv[1]}
exec(compile(src[:cut], sys.argv[1], "exec"), g)
heavy = [m for m in sys.argv[2:] if any(k == m or k.startswith(m + ".") for k in sys.modules)]
print("SPLASH", g.get("splash") is not None)
print("HEAVY", " ".join(heavy))
"""


def test_the_splash_comes_before_the_heavy_libraries():
    env = dict(os.environ, FEA_MCP_NO_GUI="1", QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8", FEA_UI_REMEMBER="0")
    r = subprocess.run([sys.executable, "-c", CODE, str(ROOT / "fea_gui" / "__main__.py"), *HEAVY], cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    assert "SPLASH True" in r.stdout, r.stdout + r.stderr[-400:]
    heavy = r.stdout.split("HEAVY")[-1].strip()
    assert heavy == "", f"imported before the splash is shown (see the docstring: measured trade-off): {heavy}"


def test_the_window_module_is_still_imported_only_after_the_splash():
    """The launch file must import the window (`from .main import main`) AFTER it has created the splash, and nothing heavy at the top of the file."""
    src = (ROOT / "fea_gui" / "__main__.py").read_text(encoding="utf-8")
    assert src.index("_splash()") < src.index("from .main import main")
    import ast
    top = ast.parse(src[:src.index("from .main import main")])
    mods = [a.name for n in top.body if isinstance(n, ast.Import) for a in n.names]          # MODULE-level imports only: an import inside a function
    mods += [n.module or "" for n in top.body if isinstance(n, ast.ImportFrom)]               # (the message of a missing GUI module) runs only when called
    for name in ("numpy", "OCP", "scipy", "fea_core"):
        assert not any(m == name or (m.startswith(name + ".") and m != "fea_core.config") for m in mods), name

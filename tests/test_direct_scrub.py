"""The reason of a solver fallback is saved into the results (warnings, backend_fallback) – it must not carry the folders of the machine."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.pathcheck import scrub_paths


@pytest.mark.parametrize("text, expected", [
    ("[Errno 2] No such file or directory: 'C:\\Users\\Bob\\fea\\x.npy'", "[Errno 2] No such file or directory: '<path>'"),
    ("thread panicked at C:/Users/Bob/.cargo/src/lib.rs:12:3", "thread panicked at <path>"),
    ("open /home/bob/fea/x.npy failed", "open <path> failed"),
    ("at \\\\server\\share\\f.npy", "at <path>"),
    ("ratio 3/4, a/b, time 12:30/45, t/mm3", "ratio 3/4, a/b, time 12:30/45, t/mm3"),     # no false positives in ordinary text
    ("The direct solver failed (code 0xC0000409): memory allocation of 123 bytes failed", None),
])
def test_scrub_paths(text, expected):
    out = scrub_paths(text)
    assert out == (text if expected is None else expected)


def test_fallback_reason_saved_without_a_path(monkeypatch, tmp_path):
    """auto: the direct solver fails with a message that names a file → the saved warning / reason has no path."""
    from fea_core import solve, solve_large
    from fea_core.study import Study

    class _Mesh:
        nodes = np.zeros((10, 3))
        part_names = ["P"]

    def boom(*a, **k):
        raise solve_large.LargeSolverError("[Errno 2] No such file or directory: 'C:\\Users\\Bob\\ngs\\direct\\x.npy'\nmore")
    monkeypatch.setattr(solve_large, "run_ngs", boom)
    st = Study("t", "m.step")
    st.solver.backend = "auto"
    warnings: list[str] = []
    logs: list[str] = []
    monkeypatch.setattr(solve, "_ccx_job", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    with pytest.raises(RuntimeError, match="stop here"):
        solve._run_mechanics(st, _Mesh(), tmp_path, warnings, logs.append, None, None, [], None, None,
                             decision=("ngs_direct", "r"))
    assert warnings and all("Bob" not in w and "C:" not in w for w in warnings) and "<path>" in warnings[0]
    assert all("Bob" not in ln for ln in logs)

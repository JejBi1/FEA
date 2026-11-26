"""First use of the numba kernels is announced (fea_core.jitnotice): nothing is precompiled at installation, so the compile on the
first call must not look like a hang – a cache hit and a short compile stay silent, a failed compile is never reported as compiled.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeTimer:
    """Stands in for threading.Timer: records the timers and fires them by hand."""
    made: list["FakeTimer"] = []

    def __init__(self, interval, fn, args=()):
        self.interval, self.fn, self.args = interval, fn, args
        self.cancelled = False
        self.daemon = False

    def start(self):
        FakeTimer.made.append(self)

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.fn(*self.args)


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def jn(monkeypatch):
    """The module with a clean state, one recording sink, manual timers and a manual clock."""
    from fea_core import jitnotice
    monkeypatch.setattr(jitnotice, "_burst_end", float("-inf"))
    monkeypatch.setattr(jitnotice, "_burst_areas", set())
    monkeypatch.setattr(jitnotice, "_sinks", [])
    FakeTimer.made = []
    monkeypatch.setattr(jitnotice.threading, "Timer", FakeTimer)
    clock = Clock()
    monkeypatch.setattr(jitnotice.time, "monotonic", clock)
    got: list[str] = []
    jitnotice.add_sink(got.append)
    jitnotice._local.depth = 0
    jitnotice._local.current = None
    yield jitnotice, got, clock
    jitnotice._local.depth = 0
    jitnotice._local.current = None


def test_area_names():
    from fea_core import jitnotice
    assert jitnotice.area_of("fea_core.fatigue.kernel") == "fatigue analysis"
    assert jitnotice.area_of("fea_core.printing._dist") == "3D printing"
    assert jitnotice.area_of("fea_core.mcp.raster") == "result pictures"
    assert jitnotice.area_of("fea_core.report.images") == "result pictures"
    assert jitnotice.area_of("fea_core.dynamics.psd") == "dynamics"
    assert jitnotice.area_of("fea_core.fatigue_other") == ""             # a prefix match needs the package boundary
    assert jitnotice.area_of("somewhere.else") == ""


def test_a_short_compile_says_nothing(jn):
    """A kernel that compiles in a second or two: no start line (the timer is cancelled), no "Compiled" line."""
    jitnotice, got, clock = jn
    jitnotice._on_start("fea_core.mcp.raster", "raster_triangles")
    assert len(FakeTimer.made) == 1 and FakeTimer.made[0].interval == jitnotice.NOTICE_AFTER_S and FakeTimer.made[0].daemon
    clock.t += 1.2
    jitnotice._on_end("fea_core.mcp.raster", "raster_triangles")
    FakeTimer.made[0].fire()                                                 # a timer that fires anyway after the end does nothing
    assert got == [] and FakeTimer.made[0].cancelled


def test_a_long_compile_is_announced_when_it_has_lasted_a_while_and_reported_at_the_end(jn):
    jitnotice, got, clock = jn
    jitnotice._on_start("fea_core.fatigue.critical_plane", "findley")
    assert got == []                                                         # nothing yet: it may be short
    clock.t += jitnotice.NOTICE_AFTER_S
    FakeTimer.made[0].fire()                                                 # the compile is still running
    assert len(got) == 1 and got[0].startswith("First use of fatigue analysis")
    assert "minute" not in got[0]                                            # no promise of a long wait
    clock.t += 11.4
    jitnotice._on_end("fea_core.fatigue.critical_plane", "findley")
    assert got[-1] == "Compiled fea_core.fatigue.critical_plane.findley in 14 s (cached for the next run)"


def test_nested_kernels_have_one_timer_and_one_message(jn):
    jitnotice, got, clock = jn
    jitnotice._on_start("fea_core.fatigue.kernel", "rainflow")
    jitnotice._on_start("fea_core.fatigue.kernel", "principal3")             # compiled for the sake of the first one
    jitnotice._on_end("fea_core.fatigue.kernel", "principal3")
    assert len(FakeTimer.made) == 1
    clock.t += 4
    FakeTimer.made[0].fire()
    jitnotice._on_end("fea_core.fatigue.kernel", "rainflow")
    assert len([g for g in got if g.startswith("First use")]) == 1


def _compile(jitnotice, clock, module, name, seconds=4.0):
    """One compilation that lasts `seconds` (the timer of its start fires when it is long enough)."""
    n = len(FakeTimer.made)
    jitnotice._on_start(module, name)
    clock.t += seconds
    if seconds >= jitnotice.NOTICE_AFTER_S:
        FakeTimer.made[n].fire()
    jitnotice._on_end(module, name)


def test_areas_that_alternate_in_one_burst_are_announced_once_each(jn):
    """3D-printing fatigue calls the fatigue kernels: fatigue → printing → fatigue → printing must not print four start lines."""
    jitnotice, got, clock = jn
    for module, name in (("fea_core.fatigue.kernel", "a"), ("fea_core.printing._dist", "b"), ("fea_core.fatigue.kernel", "c"),
                         ("fea_core.printing.fatigue", "d")):
        _compile(jitnotice, clock, module, name)
        clock.t += 1.0                                                       # a pause shorter than BURST_GAP_S
    starts = [g.split(":")[0] for g in got if g.startswith("First use")]
    assert starts == ["First use of fatigue analysis", "First use of 3D printing"]
    assert len([g for g in got if g.startswith("Compiled")]) == 4            # every long compile still reports its time


def test_the_assembly_kernels_are_announced_by_their_own_process_only(jn):
    """The matrix assembly runs in `python -m fea_core.assembly.worker`, which prints ">> Compiling the assembly kernel …" itself. A long
    compile of an assembly kernel in a process that has this listener must not add a second message ("Compiling the numerical kernels")."""
    jitnotice, got, clock = jn
    assert jitnotice.is_quiet("fea_core.assembly.kernels") and jitnotice.is_quiet("fea_core.assembly")
    assert not jitnotice.is_quiet("fea_core.assembly_other")
    n = len(FakeTimer.made)
    jitnotice._on_start("fea_core.assembly.kernels", "stiffness")
    assert len(FakeTimer.made) == n                                          # no timer at all
    clock.t += 30
    jitnotice._on_end("fea_core.assembly.kernels", "stiffness")
    assert got == []                                                         # no start line, no "Compiled …" line
    _compile(jitnotice, clock, "fea_core.fatigue.kernel", "rainflow")        # and the bookkeeping is intact for the next kernel
    assert [g.split(":")[0] for g in got if g.startswith("First use")] == ["First use of fatigue analysis"]


def test_a_new_burst_announces_again(jn):
    jitnotice, got, clock = jn
    _compile(jitnotice, clock, "fea_core.mcp.raster", "raster_triangles")
    clock.t += jitnotice.BURST_GAP_S + 1                                     # the same area, but a long time later
    _compile(jitnotice, clock, "fea_core.mcp.raster", "raster_segments")
    assert len([g for g in got if g.startswith("First use of result pictures")]) == 2
    clock.t += jitnotice.BURST_GAP_S + 1
    _compile(jitnotice, clock, "somewhere.else", "k")
    assert [g for g in got if g.startswith("Compiling the numerical kernels")]


def test_a_failed_compile_is_not_reported_as_compiled(jn):
    jitnotice, got, clock = jn
    jitnotice._on_start("fea_core.fatigue.kernel", "broken")
    clock.t += 9
    FakeTimer.made[0].fire()
    jitnotice._on_end("fea_core.fatigue.kernel", "broken", failed=True)
    assert [g for g in got if g.startswith("Compiled")] == []
    assert len(got) == 1 and got[0].startswith("First use")                  # the wait was announced, the failure is the caller's error


def test_a_failing_sink_does_not_stop_the_work(jn):
    jitnotice, got, _clock = jn

    def bad(_text):
        raise RuntimeError("the window is gone")
    jitnotice.add_sink(bad)
    jitnotice.say("hello")
    assert got == ["hello"]
    jitnotice.remove_sink(bad)


def test_without_sink_the_message_goes_to_stderr(monkeypatch, capsys):
    from fea_core import jitnotice
    monkeypatch.setattr(jitnotice, "_sinks", [])
    jitnotice.say("hello")
    assert "hello" in capsys.readouterr().err
    monkeypatch.setattr(sys, "stderr", None)                                  # pythonw: no console at all
    jitnotice.say("silent")


_KERNEL = textwrap.dedent('''
    from numba import njit

    @njit(cache=True)
    def inner(x):
        s = 0.0
        for v in x:
            s += v * v
        return s

    @njit(cache=True)
    def outer(x):
        return inner(x) + 1.0

    @njit(cache=True)
    def broken(x):
        return undefined_name(x)
''')

_RUN = textwrap.dedent('''
    import sys
    import numpy as np
    from fea_core.modules import need
    need("numba", "accel")                      # the same call every module with kernels makes
    from fea_core import jitnotice
    jitnotice.NOTICE_AFTER_S = 0.0              # a tiny kernel: announce at once (in the program only after a few seconds)
    sys.path.insert(0, sys.argv[1])
    import kmod
    if sys.argv[2] == "fail":
        try:
            kmod.broken(np.arange(3.0))
        except Exception as e:
            print("error:", type(e).__name__)
    else:
        print(kmod.outer(np.arange(5.0)))
''')


def _go(tmp_path, mode, env):
    p = subprocess.run([sys.executable, str(tmp_path / "run.py"), str(tmp_path), mode], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout, p.stderr


def test_real_numba_compile_is_announced_once_and_the_cache_is_silent(tmp_path):
    pytest.importorskip("numba")
    (tmp_path / "kmod.py").write_text(_KERNEL, encoding="utf-8")
    (tmp_path / "run.py").write_text(_RUN, encoding="utf-8")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8",
               FEA_MCP_NO_GUI="1")
    out, first = _go(tmp_path, "ok", env)
    assert out.strip() == "31.0"
    assert first.count("Compiling the numerical kernels") == 1, first        # one line for the whole burst (nested inner kernel too)
    assert "Compiled kmod.outer in" in first, first                           # a successful compile IS reported (numba's is_failed is inverted)
    out, second = _go(tmp_path, "ok", env)
    assert out.strip() == "31.0"
    assert "compiling" not in second.lower(), second                          # cache hit: nothing to wait for


def test_real_numba_failed_compile_is_not_reported_as_compiled(tmp_path):
    pytest.importorskip("numba")
    (tmp_path / "kmod.py").write_text(_KERNEL, encoding="utf-8")
    (tmp_path / "run.py").write_text(_RUN, encoding="utf-8")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8",
               FEA_MCP_NO_GUI="1")
    out, err = _go(tmp_path, "fail", env)
    assert out.startswith("error:"), out                                      # the error reaches the caller
    assert "Compiled" not in err, err


def test_full_warmup_precompiles_the_assembly_kernel_and_quick_does_not(monkeypatch):
    """`python -m fea_core.warmup` (developer tool) fills the numba cache of every area, the matrix assembly of the direct solver included;
    `--quick` (what the installer runs) only checks V1 and compiles nothing."""
    import verification.run_all as run_all
    from fea_core import warmup
    calls: list[str] = []
    monkeypatch.setattr(run_all, "run", lambda keys: calls.append("run " + " ".join(keys)) or True)
    monkeypatch.setattr(warmup, "_assembly_kernel", lambda: calls.append("assembly"))
    monkeypatch.setattr(warmup, "_dynamics_kernels", lambda: calls.append("dynamics"))
    monkeypatch.setattr(warmup, "_report_v1", lambda root: calls.append("report"))
    assert warmup.main(["--quick"]) == 0
    assert calls == ["run V1"]
    calls.clear()
    assert warmup.main([]) == 0
    assert calls == ["run V1 U3 P5", "assembly", "dynamics", "report"]

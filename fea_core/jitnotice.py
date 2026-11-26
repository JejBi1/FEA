"""Visible first use of the numba kernels (nothing is precompiled at installation).

The numba cache (`@njit(cache=True)`, files `*.nbi` / `*.nbc` in the `__pycache__` of the package) is specific to the CPU model of
the machine, so a release build cannot carry it, and the installer no longer runs the 1–2 minute warm-up (`python -m fea_core.warmup`
stays available for developers). The kernels are compiled at their first call on this computer – fatigue (rainflow, critical plane),
3D printing (shell distances, printing fatigue), result pictures (rasteriser of MCP and the Word report) and dynamics – and the cache
makes every later call instant. Without a message such a call (up to ~1 min for the critical plane) looks like a hang.

`install()` registers one listener for the numba event `numba:compile`. numba triggers it only when a kernel is REALLY compiled (a hit
in the disk cache does not), so the message appears exactly where the wait happens and never after the first run. It is called by
`modules.need("numba", ...)`, i.e. before any kernel can be compiled, and it is cheap and idempotent.

A compilation is announced only when it has already lasted `NOTICE_AFTER_S` seconds (a timer thread; a kernel that compiles in a second
or two needs no message and says nothing – the message therefore never promises a long wait for a short one). One start line per area
(fatigue analysis, 3D printing, …) and "burst" of compilations: kernels of two areas that call each other (3D-printing fatigue uses the
fatigue kernels) do not repeat their lines. A compilation that FAILS is never reported as compiled – the error reaches the caller.

The text goes to the sinks (`add_sink`): the GUI writes it into its log, the default is one line to stderr (if there is one – a GUI
without a console has none) and into the `fea_core.jit` logger. Headless MCP / REST / CLI runs have only that stderr line (the MCP client
sees no notice, only the delay; the server instructions say so). Sinks are called from the thread that compiles or from the timer
thread and must be thread safe.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable

log = logging.getLogger("fea_core.jit")

# a compilation is announced (start line) and reported when it ends only if it lasts at least this long [s]
NOTICE_AFTER_S = 3.0
# compilations separated by less than this belong to one burst – one start line per area in a burst
BURST_GAP_S = 5.0

# module prefix → what the user is doing (nominative, translated through the catalogue); the first matching prefix wins
AREAS = (("fea_core.fatigue", "fatigue analysis"), ("fea_core.printing", "3D printing"), ("fea_core.mcp", "result pictures"),
         ("fea_core.report", "result pictures"), ("fea_core.dynamics", "dynamics"))
DEFAULT_AREA = ""
# kernels of these packages are compiled in a process of their own that announces itself: the matrix assembly of the linear path runs in
# `python -m fea_core.assembly.worker` (it imports numba directly, never through `modules.need`, so this listener is not installed there) and says
# ">> Compiling the assembly kernel …" (shown in the log and the progress bar). If such a kernel is ever compiled in a process that has the
# listener (a script, a test, FEA_ASSEMBLY in-process), it stays silent here instead of repeating the worker's message under another name.
QUIET = ("fea_core.assembly",)

_sinks: list[Callable[[str], None]] = []
_lock = threading.Lock()              # the list of sinks
_state = threading.Lock()             # the burst bookkeeping (timer threads and compiling threads)
_installed = False
_burst_end = float("-inf")            # monotonic time when the last compilation ended
_burst_areas: set[str] = set()        # areas whose start line was already written in the current burst


class _Local(threading.local):
    """Per thread: the depth of nested compilations and the outermost one (`_Compile`)."""
    depth = 0
    current: _Compile | None = None


_local = _Local()


class _Compile:
    """One outermost compilation: its start, area and whether the start line was written / it has ended."""

    def __init__(self, area: str, t0: float, quiet: bool = False):
        self.area, self.t0, self.quiet = area, t0, quiet
        self.announced = False
        self.done = False
        self.timer: threading.Timer | None = None


def add_sink(fn: Callable[[str], None]) -> None:
    """Adds a receiver of the messages (called from any thread)."""
    with _lock:
        if fn not in _sinks:
            _sinks.append(fn)


def remove_sink(fn: Callable[[str], None]) -> None:
    with _lock:
        if fn in _sinks:
            _sinks.remove(fn)


def say(text: str) -> None:
    """Delivers a message to the sinks; without a sink to stderr (when it exists) – a failing sink never breaks the calculation."""
    log.info(text)
    with _lock:
        sinks = list(_sinks)
    if not sinks:
        err = sys.stderr
        if err is not None:
            try:
                print(text, file=err, flush=True)
            except (OSError, ValueError):
                pass
        return
    for fn in sinks:
        try:
            fn(text)
        except Exception:  # noqa: BLE001 – a message must never stop the numerical work
            log.debug("jit notice sink failed", exc_info=True)


def area_of(module: str) -> str:
    for prefix, name in AREAS:
        if module == prefix or module.startswith(prefix + "."):
            return name
    return DEFAULT_AREA


def is_quiet(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in QUIET)


def _announce(cur: _Compile) -> None:
    """The compilation has lasted NOTICE_AFTER_S: write the start line (once per area in the burst). Runs in the timer thread."""
    with _state:
        if cur.done or cur.announced:
            return
        cur.announced = True
        fresh = cur.area not in _burst_areas
        _burst_areas.add(cur.area)
    if not fresh:
        return
    if cur.area:
        say(f"First use of {cur.area}: compiling the numerical kernels (once on this computer, then cached) – this may take a while…")
    else:
        say("Compiling the numerical kernels (first use on this computer, then cached) – this may take a while…")


def _on_start(module: str, name: str) -> None:
    d = _local.depth
    _local.depth = d + 1
    if d:                                           # a kernel compiled for the sake of another kernel
        return
    now = time.monotonic()
    cur = _Compile(area_of(module), now, is_quiet(module))
    _local.current = cur
    if cur.quiet:                                   # announced by the process that compiles it (QUIET) – no timer, no burst bookkeeping
        return
    with _state:
        if now - _burst_end > BURST_GAP_S:          # a new burst: every area may announce itself again
            _burst_areas.clear()
    if NOTICE_AFTER_S <= 0:
        _announce(cur)
        return
    cur.timer = threading.Timer(NOTICE_AFTER_S, _announce, args=(cur,))
    cur.timer.daemon = True
    cur.timer.start()


def _on_end(module: str, name: str, failed: bool = False) -> None:
    global _burst_end
    d = _local.depth - 1
    _local.depth = max(d, 0)
    if d > 0:
        return
    now = time.monotonic()
    cur = _local.current
    _local.current = None
    if cur is not None and cur.quiet:
        return
    took = now - cur.t0 if cur is not None else 0.0
    with _state:
        _burst_end = now
        if cur is not None:
            cur.done = True
    if cur is not None and cur.timer is not None:
        cur.timer.cancel()
    if failed:                                      # the error reaches the caller; never claim a result that does not exist
        return
    if took >= NOTICE_AFTER_S:
        say(f"Compiled {module}.{name} in {took:.0f} s (cached for the next run)")


def install() -> bool:
    """Registers the listener of the numba compile events (once). False if this numba has no event interface."""
    global _installed
    with _lock:
        if _installed:
            return True
        try:
            from numba.core import event
        except ImportError:                          # numba older than 0.53 – no notice, everything else works
            return False

        def ident(ev) -> tuple[str, str]:
            py = getattr(ev.data["dispatcher"], "py_func", None)
            return getattr(py, "__module__", "") or "", getattr(py, "__name__", "?")

        class _Notice(event.Listener):
            """A notice must never break a compilation: whatever goes wrong here is only logged."""

            def on_start(self, ev):
                try:
                    _on_start(*ident(ev))
                except Exception:  # noqa: BLE001
                    log.debug("jit notice (start) failed", exc_info=True)

            def on_end(self, ev):
                try:
                    # not `ev.is_failed`: in numba 0.68 that property is inverted (True WITHOUT an exception)
                    _on_end(*ident(ev), failed=getattr(ev, "_exc_details", None) is not None)
                except Exception:  # noqa: BLE001
                    log.debug("jit notice (end) failed", exc_info=True)

        event.register("numba:compile", _Notice())
        _installed = True
        return True

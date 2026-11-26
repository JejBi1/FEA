"""F14 S4 review regressions of the shared runner `s4_run`: the interface report catches the interface errors and an ASD-only call needs no base-acceleration table, the stored
notch result records what it was computed with, the breakpoint truncation is reported."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.dynamics.model import Excitation
from fea_core.space import s4_run
from fea_core.study import Study
from tests.s4_helpers import confirmed_space, sdof_interface

ASD = [[20.0, 0.01], [2000.0, 0.01]]
SINE = [[10.0, 1.0], [3000.0, 1.0]]


def _study(harmonic_without_table: bool = False) -> Study:
    st = Study("review", "none.step")
    st.dynamics.damping.zeta = 0.02
    if harmonic_without_table:
        st.dynamics.harmonic.enabled = True
        st.dynamics.harmonic.excitation = [Excitation(kind="base_acc", direction=[1, 0, 0], table=[])]
    return st


def test_an_asd_only_call_needs_no_base_acceleration_table():
    st = _study(harmonic_without_table=True)
    mi = sdof_interface()
    d = s4_run.interface_report(st, mi, "x", None, ASD, 0.02)
    assert "random" in d["directions"]["x"] and "sine" not in d["directions"]["x"]
    with pytest.raises(s4_run.S4Error):                         # the harmonic settings promise a sine environment but have no table, and nothing else was asked for
        s4_run.interface_report(st, mi, "x", None, None, 0.02)
    d2 = s4_run.interface_report(st, mi, "x", SINE, None, 0.02)
    assert "sine" in d2["directions"]["x"] and "random" not in d2["directions"]["x"]
    nothing = s4_run.interface_report(_study(), mi, "x", None, None, 0.02)
    assert "hint" in nothing


def test_interface_errors_are_clean_s4_errors():
    mi = sdof_interface()
    st = _study()
    with pytest.raises(s4_run.S4Error):
        s4_run.interface_report(st, mi, "x", [[100.0, 1.0]], None, 0.02)            # one frequency: no range for the grid -> InterfaceError inside
    with pytest.raises(s4_run.S4Error):
        s4_run.interface_report(st, mi, "x", SINE, None, 0.02, group="nope")
    with pytest.raises(s4_run.S4Error):
        s4_run.interface_report(st, mi, "x", SINE, None, 1.5)                        # zeta outside 0..1


def test_the_stored_notch_result_records_how_it_was_made_and_the_truncation():
    mi = sdof_interface()
    st = Study("review", "none.step")
    st.space = confirmed_space({"notch_force_limit_N": 1.0, "notch_min_width_oct": 0.1, "notch_floor_ratio": 0.01, "notch_design_factor": 1.0})
    d, res = s4_run.notch_report(st, "sine", "x", "force", limit=0.4, profile=[[50.0, 1.0], [2000.0, 1.0]], zeta=0.02, interface=lambda: mi, group=None)
    for k in ("direction", "damping", "profile", "group", "modes", "kind", "limit_kind", "engine", "grid_level", "grid_points", "table_tolerance", "table_ratio", "design_factor_applied"):
        assert k in d, k
    assert d["direction"] == "x" and d["modes"] == 1 and d["group"] == "all" and d["zeta"] == [0.02] and "given" in d["damping"]
    assert d["breakpoints_total"] == len(res.breakpoints) and d["breakpoints_truncated"] == (len(d["breakpoints"]) < len(res.breakpoints))
    # a profile with many features: more breakpoints than the answer lists -> the answer says so
    mi2 = sdof_interface()
    old = s4_run.MAX_BREAKPOINTS
    s4_run.MAX_BREAKPOINTS = 3
    try:
        d2, r2 = s4_run.notch_report(st, "sine", "x", "force", limit=0.4, profile=[[50.0, 1.0], [2000.0, 1.0]], zeta=0.02, interface=lambda: mi2)
    finally:
        s4_run.MAX_BREAKPOINTS = old
    assert len(r2.breakpoints) > 3 and d2["breakpoints_truncated"] and len(d2["breakpoints"]) == 3 and "export_path" in d2["breakpoints_note"]
    # the stored result is JSON-able with all of it
    from fea_core.space import s4_store
    s4_store.record(st, "notch", d, "sine x force")
    import json
    json.dumps(st.space.to_dict(), allow_nan=False)
    assert np.isfinite(d["table_ratio"])

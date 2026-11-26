"""F14 S4: the stored results (`Study.space.extra['s4']`) – JSON round trip with the study, the report subsections on a real python-docx document, and the rule that
a non-PASS verdict is printed as such with the formula-chain note of the bolts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("docx")

import verification.cases  # noqa: E402, F401  (first the cases)
from fea_core.dynamics import interface as IF  # noqa: E402
from fea_core.report import space as RS  # noqa: E402
from fea_core.report import space_s4 as S4R  # noqa: E402
from fea_core.report.word import _Doc  # noqa: E402
from fea_core.space import bolts_ecss as BE  # noqa: E402
from fea_core.space import notching as NT  # noqa: E402
from fea_core.space import pcb as PB  # noqa: E402
from fea_core.space import s4_store  # noqa: E402
from fea_core.space.model import SpaceSettings  # noqa: E402
from fea_core.space.s4_limits import S4Limits  # noqa: E402
from tests.s4_helpers import M_SDOF, Z_SDOF, G, sdof_interface  # noqa: E402
from tests.test_space_s4_bolts import _bolt, _lim  # noqa: E402
from verification import space_cases as SC  # noqa: E402


def _text(D) -> str:
    parts = [p.text for p in D.doc.paragraphs]
    for t in D.doc.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def _study_with_s4():
    st = SC.block_study()
    st.space = SpaceSettings()
    mi = sdof_interface()
    f = IF.sine_grid(50.0, 2000.0, mi.freq, np.array([Z_SDOF]))
    sp = IF.sine_interface(mi, "x", 1.0, Z_SDOF, f=f)
    peak = float(sp.F_peak.max())
    res = NT.make_notch(sp.f, sp.a_g, NT.interface_response(sp), NT.NotchSpec("sine", "force", limit=0.4 * peak, design_factor=1.0, min_width_oct=0.1, floor_ratio=0.1), S4Limits(),
                        mi.freq)
    d = res.summary()
    d["justification"] = [dict(zip(NT.JUSTIFICATION_COLUMNS, r)) for r in NT.justification_rows(res)]
    s4_store.record(st, "notch", d, "sine x force")
    s4_store.record(st, "interface", {"total_mass_kg": 1000 * M_SDOF, "ref_point_mm": [0, 0, 0], "modes": 1, "engine": "test", "damping": "zeta given",
                                      "static_mass_check": mi.info.get("static_mass_check") or {"relative_error": [0.0, 0.0, 0.0]}, "modal_table": mi.modal_table(),
                                      "directions": {"x": {"sine": sp.summary(), "random": IF.random_interface(mi, "x", [[20.0, 0.01], [2000.0, 0.01]], Z_SDOF).summary()}},
                                      "warnings": []})
    br = BE.assess_bolt(_bolt(), _lim())
    s4_store.record(st, "bolts", br.to_dict(), "B1")
    pr = PB.check_pcb(PB.PcbSpec(name="U1", B_mm=127.0, h_mm=1.575, L_mm=25.4, C=1.0, r=1.0, z_rms_mm=0.05, f_n_Hz=100.0, duration_s=3600.0, const=0.00022, const_unit="in",
                                 amplitude_factor=1.0, design_factor=1.25, fatigue_exponent=6.4, ref_cycles=2e7, damage_limit=1.0))
    s4_store.record(st, "pcb", pr.to_dict(), "U1")
    s4_store.record(st, "rails", {"verdict": "UNCONFIRMED_REQUIREMENTS", "verdict_text": "x", "inputs": [], "warnings": ["a wall would have to PULL"],
                                  "supports": [{"support": "Rail R1 +y", "part": "R1", "kind": "wall", "state": "Launch", "reaction_N": 400.0, "pulls": False, "limit_N": 800.0,
                                                "mos": 1.0}], "deflection": [{"part": "R1", "max_lateral_deflection_mm": 0.25, "state": "Launch", "limit_mm": 0.5, "mos": 1.0}]})
    return st, sp, res


def test_store_round_trip_with_the_study():
    from fea_core.space.model import SpaceSettings as SS
    st, _, _ = _study_with_s4()
    back = SS.from_dict(st.space.to_dict())
    assert set(s4_store.get(SimpleNamespace(space=back))) == {"notch", "interface", "bolts", "pcb", "rails"}
    s4_store.clear(st, "rails")
    assert "rails" not in s4_store.get(st) and "bolts" in s4_store.get(st)
    with pytest.raises(ValueError):
        s4_store.record(st, "nope", {})
    # infinities (MoS of a bolt without load) survive as strings, not as invalid JSON
    s4_store.record(st, "bolts", {"margins": [{"mos": float("inf")}]}, "B9")
    import json
    json.dumps(st.space.to_dict(), allow_nan=False)


def test_subsections_of_the_report():
    st, sp, res = _study_with_s4()
    assert S4R.has_s4(st)
    D = _Doc()
    warns = S4R.s4_sections(D, st, "9")
    text = _text(D)
    assert warns == []
    for needle in ("Interface forces and moments", "Primary notching", "Bolted joints", "Printed circuit boards", "Deployer rails", "ECSS-E-HB-32-23A", "Notch justification table",
                   "Modal base-reaction vectors", "Margins of safety of bolt B1", "PCB displacement margin", "Reaction of the rail supports", "gapping", "pullout"):
        assert needle in text, needle
    assert "UNCONFIRMED_REQUIREMENTS" in text                              # the notch limit was entered directly: never printed as PASS
    assert "Verdict: PASS" not in text
    assert f"{res.notches[0].reduction_db:.3g}".replace("-", "−") in text or f"{res.notches[0].reduction_db:.4g}".replace("-", "−") in text
    assert sp.F_peak.max() > M_SDOF * G


def test_space_chapter_calls_the_subsections():
    st, _, _ = _study_with_s4()
    mesh = SC.box_tet_mesh()
    SC.confirm_test_allowables(st)
    st.space.requirement_set = SC.make_requirement_set("protoflight", confirmed=True)
    res = SC.synthetic_results(mesh, [SC.uniform_state(mesh, "Load", SC.TEST_SIGMA)])
    D = _Doc()
    RS.space_section(D, SimpleNamespace(study=st, results=res), "7")
    text = _text(D)
    assert "Verdict: " in text and "Interface forces and moments" in text and "7.1" in text

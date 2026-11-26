"""F14 S4: the MCP tools of the group "space" (interface forces, notch, ECSS bolts, PCB check, rails) – lazy group, the answers, and the rule that numbers passed by a
caller are used but never confirmed (no PASS from a tool call alone)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

import verification.cases  # noqa: E402, F401  (first the cases)
from fea_core import config  # noqa: E402
from fea_core.geometry import Part  # noqa: E402
from fea_core.mcp.server import build_server  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.solve import prepare_mesh  # noqa: E402
from fea_core.space.model import MaterialAllowable  # noqa: E402
from fea_core.space.requirements import confirm_allowable, set_allowable  # noqa: E402
from fea_core.study import Fixed, LoadCase, MeshSettings  # noqa: E402
from tests.s4_helpers import confirmed_space  # noqa: E402
from verification.cases import _study, box  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S4_TOOLS = {"fea_space_interface_forces", "fea_space_notch", "fea_space_bolts_ecss", "fea_space_pcb_check", "fea_space_rails"}
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")


def _run(sess: Session, steps):
    out: list = []

    async def main():
        srv = build_server(sess, groups_spec="space")
        async with Client(srv, mode="legacy") as c:
            for name, args, expect_err in steps:
                r = await c.call_tool(name, args)
                text = r.content[0].text
                assert bool(r.is_error) == expect_err, (name, text)
                out.append(text if expect_err else json.loads(text))
    anyio.run(main)
    return out


def test_group_is_lazy_and_lists_the_s4_tools(tmp_path):
    async def main():
        srv = build_server(Session(tmp_path / "b"), groups_spec="core")
        async with Client(srv, mode="legacy") as c:
            assert not S4_TOOLS & {t.name for t in (await c.list_tools()).tools}
            r = json.loads((await c.call_tool("fea_tools_enable", {"groups": "space"})).content[0].text)
            assert S4_TOOLS <= set(r["added"])
    anyio.run(main)
    code = ("import sys; from fea_core.mcp.server import build_server; build_server(groups_spec='core'); "
            "print('LOADED', 'fea_core.mcp.tools_space_s4' in sys.modules, 'fea_core.space.notching' in sys.modules, 'fea_core.dynamics.interface' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip().endswith("LOADED False False False"), r.stdout


def _session(tmp_path, with_mesh: bool = True) -> Session:
    wd = tmp_path / "model"
    wd.mkdir()
    st, parts = _study("s4 mcp", wd / "model.step", [Part("Konzola", box(0, 0, 0, 80.0, 10.0, 10.0))], MeshSettings(size=4.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("F", []))
    st.dynamics.enabled = True
    st.dynamics.modes = 10
    st.dynamics.damping.zeta = 0.02
    st.solver.backend = "auto"
    st.solver.modal_engine = "ccx"
    sess = Session(tmp_path / "mcp")
    sess.study = st
    sess._parts = parts
    if with_mesh:
        sess.mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
        sess.mesh_sig = sess.mesh_signature()
    return sess


@needs_ccx
def test_interface_forces_and_notch(tmp_path):
    sess = _session(tmp_path)
    sine = [[10.0, 1.0], [3000.0, 1.0]]
    steps = [
        ("fea_space_interface_forces", {"directions": "y", "sine": sine, "asd": [[20.0, 0.01], [2000.0, 0.01]]}, False),                  # 0
        ("fea_space_interface_forces", {"directions": "y", "sine": sine}, False),                                                         # 1: from the cache
        ("fea_space_interface_forces", {"directions": "q"}, True),                                                                        # 2
        ("fea_space_interface_forces", {"directions": "y", "sine": sine, "group": "nope"}, True),                                         # 3
    ]
    r = _run(sess, steps)
    a = r[0]
    assert a["static_mass_check"]["relative_error"][1] < 5e-3 and not a["from_cache"] and r[1]["from_cache"]
    assert a["modal_table"] and a["directions"]["y"]["sine"]["F_peak_max_N"] > a["total_mass_kg"] * 9.81 and "random" in a["directions"]["y"]
    peak = a["directions"]["y"]["sine"]["F_peak_max_N"]
    # notch: a limit entered through the call is used but not confirmed
    lim_n = 0.5 * peak
    r2 = _run(sess, [("fea_space_notch", {"direction": "y", "kind": "sine", "limit_kind": "force", "limit": lim_n, "design_factor": 1.0, "min_width_oct": 0.1,
                                          "floor_ratio": 0.1, "profile": sine, "export_path": str(tmp_path / "out" / "notched.csv")}, False)])[0]
    assert r2["verdict"] == "UNCONFIRMED_REQUIREMENTS" and r2["notches"] and r2["notched_peak"] == pytest.approx(lim_n, rel=1e-4)
    assert all(Path(p).exists() for p in r2["exported"]) and Path(r2["exported"][0]).read_text(encoding="utf-8").splitlines()[1].startswith("f_Hz,notched_g")
    assert "notch_min_width_oct" in r2["limits_needed"]
    # the same with the requirement set that carries the limits and is confirmed by the user (set directly: a tool call cannot confirm)
    sess.study.space = confirmed_space({"notch_force_limit_N": lim_n, "notch_min_width_oct": 0.1, "notch_floor_ratio": 0.1, "notch_design_factor": 1.0})
    r3 = _run(sess, [("fea_space_notch", {"direction": "y", "profile": sine}, False)])[0]
    assert r3["verdict"] == "PASS" and r3["limits_needed"] == []
    # the tolerance of the check of the exported table and the achieved ratio are part of the answer
    assert r3["table_tolerance"] == 1e-3 and 0.0 < r3["table_ratio"] <= 1.0 + r3["table_tolerance"] and r3["grid_level"] >= 1 and r3["grid_points"] > 100
    assert r2["table_tolerance"] == 1e-3 and r2["table_ratio"] is not None
    # the random notch against a PSD limit
    r4 = _run(sess, [("fea_space_notch", {"direction": "y", "kind": "random", "limit_kind": "force_psd", "limit": 1e-3, "design_factor": 1.0, "profile": [[20.0, 0.01], [2000.0, 0.01]]},
                      False), ("fea_space_notch", {"direction": "y", "kind": "random", "limit_kind": "force", "limit": 1.0, "profile": [[20.0, 0.01], [2000.0, 0.01]]}, True),
                     ("fea_space_notch", {"direction": "y", "limit_kind": "probe_acc", "limit": 5.0, "profile": sine}, True)])
    assert r4[0]["verdict"] == "UNCONFIRMED_REQUIREMENTS"


def test_bolts_tool(tmp_path):
    sess = _session(tmp_path, with_mesh=False)
    st = sess.study
    from fea_core.study import Material
    st.add_material(Material("Ti6Al4V", E=110000.0, nu=0.34, rho=4.4e-9, Re=880.0, Rm=950.0))
    st.add_material(Material("Al7075", E=71000.0, nu=0.33, rho=2.8e-9, Re=480.0, Rm=540.0))
    for name, fy, fu, fsu in (("Ti6Al4V", 880.0, 950.0, 570.0), ("Al7075", 480.0, 540.0, 320.0)):
        set_allowable(st.materials[name], MaterialAllowable(basis="B", Fy=fy, Fu=fu, Fsy=0.6 * fy, Fsu=fsu, source="test data sheet"))
        confirm_allowable(st.materials[name], "test engineer")
    bolt = {"name": "B1", "d_mm": 6.0, "pitch_mm": 1.0, "A_s_mm2": 20.1, "k_b": 3.5e5, "k_c": 4.0e5, "F_nom": 6000.0, "embedding_um": 6.0,
            "layers": [{"t_mm": 12.0, "alpha": 23.4e-6}], "alpha_bolt": 8.6e-6, "T_min": -40.0, "T_max": 80.0, "F_ax": 3000.0, "F_sh": 800.0, "engaged_mm": 9.0, "d_w_mm": 10.0,
            "d_h_mm": 6.6, "t_bearing_mm": 4.0, "scatter": 0.1, "relaxation": 0.05, "mu_min": 0.2,
            "factors": {"gapping": 1.2, "slip": 1.25, "tensile_yield": 1.1, "tensile": 1.4, "shear": 1.4, "combined": 1.4, "pullout": 1.5, "bearing": 1.3}}
    r = _run(sess, [("fea_space_bolts_ecss", {"bolts": [bolt], "bolt_material": "Ti6Al4V", "part_material": "Al7075"}, False),
                    ("fea_space_bolts_ecss", {"bolts": [{**bolt, "joint": "nope"}]}, True),
                    ("fea_space_bolts_ecss", {"bolts": [bolt], "bolt_material": "Nope"}, True)])
    a = r[0]["bolts"][0]
    assert "ECSS-E-HB-32-23A" in r[0]["formula_note"] and r[0]["overall"] == "UNCONFIRMED_REQUIREMENTS" and a["verdict"] == "UNCONFIRMED_REQUIREMENTS"
    assert {m["check"] for m in a["margins"]} >= {"gapping", "slip", "tensile_yield", "tensile_ultimate", "shear", "combined", "pullout", "bearing_head", "bearing_hole"}
    assert a["dF_th_N"]["T_max"] > 0 > a["dF_th_N"]["T_min"] and r[0]["limits_needed"]
    # a confirmed requirement set with the same coefficients: the allowables of the materials are verified → a verdict
    from tests.test_space_s4_bolts import COEF
    st.space = confirmed_space(COEF)
    for b in (bolt,):
        for k in ("scatter", "relaxation", "mu_min", "factors"):
            b.pop(k, None)
    r5 = _run(sess, [("fea_space_bolts_ecss", {"bolts": [bolt], "bolt_material": "Ti6Al4V", "part_material": "Al7075"}, False)])[0]
    assert r5["bolts"][0]["verdict"] in ("PASS", "FAIL") and r5["limits_needed"] == []


def test_pcb_tool(tmp_path):
    sess = _session(tmp_path, with_mesh=False)
    comp = {"name": "U1", "B_mm": 127.0, "h_mm": 1.575, "L_mm": 25.4, "r": 1.0, "z_rms_mm": 0.05, "r_source": "test drawing", "z_source": "test random analysis", "f_n_Hz": 100.0, "duration_h": 1.0}
    r = _run(sess, [("fea_space_pcb_check", {"components": [comp]}, False), ("fea_space_pcb_check", {"components": [{"name": "bad", "nonsense": 1}]}, True)])
    c = r[0]["components"][0]
    assert c["verdict"] == "UNCONFIRMED_REQUIREMENTS" and c["Z_allow_mm"] is None and "pcb_steinberg_const" in r[0]["limits_needed"]
    sess.study.space = confirmed_space({"pcb_steinberg_const": 0.00022, "pcb_amplitude_factor": 1.0, "pcb_design_factor": 1.25, "pcb_fatigue_exponent": 6.4,
                                        "pcb_ref_cycles": 2e7, "pcb_damage_limit": 1.0, "pcb_n_sigma": 3.0, "pcb_component_const": 1.0,
                                        "pcb_position_factor_min": 0.25}, units={"pcb_steinberg_const": "in"})
    c2 = _run(sess, [("fea_space_pcb_check", {"components": [comp]}, False)])[0]["components"][0]
    assert c2["verdict"] in ("PASS", "FAIL") and c2["Z_allow_mm"] == pytest.approx(0.00022 * 5.0 / 0.062 * 25.4, rel=1e-3)
    assert c2["damage_three_band"]["damage"] > 0 and c2["damage_narrow_band"] > 0
    # the same component with a typed C / n_sigma: used, never confirmed
    c3 = _run(sess, [("fea_space_pcb_check", {"components": [{**comp, "C": 1.0}]}, False)])[0]["components"][0]
    assert c3["verdict"] == "UNCONFIRMED_REQUIREMENTS"


def test_rails_tool(tmp_path):
    sess = _session(tmp_path, with_mesh=False)
    rails = {"R1": (42, 42), "R2": (42, -50), "R3": (-50, 42), "R4": (-50, -50)}
    sess._parts = [Part(n, box(0, y0, z0, 100, 8, 8)) for n, (y0, z0) in rails.items()]
    sess.study.selections.clear()
    sess.study.constraints.clear()
    r = _run(sess, [("fea_space_rails", {"action": "apply", "rail_parts": list(rails), "axial": "max"}, False),
                    ("fea_space_rails", {"action": "apply", "rail_parts": ["nope"]}, True),
                    ("fea_space_rails", {"action": "check"}, True),
                    ("fea_space_rails", {"action": "dance"}, True)])
    assert len(r[0]["supports"]) == 12 and len(sess.study.constraints) == 12
    nodes = np.array([[0, 42, 42], [100, 50, 50]], float)
    mesh = SimpleNamespace(part_names=["R1"], tets=np.array([[0, 1, 0, 1]]), tet_part=np.array([0]), nodes=nodes)
    U = np.zeros((2, 3))
    U[1, 1] = 0.25
    sess.results = SimpleNamespace(steps=[SimpleNamespace(name="Launch", reactions={"Rail R1 +y": np.array([0, -400.0, 0])}, U=U)], mesh=mesh)
    c = _run(sess, [("fea_space_rails", {"action": "check", "rail_parts": ["R1"], "reaction_limit_N": 800.0, "lateral_deflection_limit_mm": 0.5}, False)])[0]
    assert c["verdict"] == "UNCONFIRMED_REQUIREMENTS" and c["supports"][0]["mos"] == pytest.approx(1.0) and c["deflection"][0]["mos"] == pytest.approx(1.0)

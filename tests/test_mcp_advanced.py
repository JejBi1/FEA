"""MCP tools of tools_advanced (ε-N, load case categories, sweep) and the thermal / contact additions
(cavity radiation, contact conductance)."""
from __future__ import annotations

import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from fea_core.mcp.server import build_server  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402


@pytest.fixture(autouse=True)
def _no_gui(monkeypatch):
    monkeypatch.setattr("fea_core.ipc.is_gui_running", lambda *a, **k: False)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from verification.cases import box, cyl, solid
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("adv_geo")
    shape = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 5), cyl((30, 20, -1), (0, 0, 1), 5, 7)).Shape())
    write_step([Part("Deska", shape)], d / "plate.step")
    write_step([Part("A", box(0, 0, 0, 20, 20, 10)), Part("B", box(0, 0, 10, 20, 20, 10))], d / "blocks.step")
    return d


def run(body, base):
    async def main():
        srv = build_server(Session(base))
        async with Client(srv, mode="legacy") as c:
            async def call(tool, ok=True, **args):
                r = await c.call_tool(tool, args)
                text = "\n".join(getattr(x, "text", "") or "" for x in r.content)
                if ok:
                    assert not r.is_error, f"{tool}: {text}"
                    return json.loads(text)
                assert r.is_error, f"{tool} should fail: {text}"
                return text
            await body(call)
    anyio.run(main)


def test_thermal_cavity_and_conductance_tools(tmp_path, plate):
    async def body(call):
        await call("fea_study_new", step_path=str(plate / "plate.step"), workdir=str(tmp_path / "wd"),
                   material="Ocel S235")
        d = await call("fea_thermal_add_cavity", case="Ohrev", selection={"radius": 5}, emissivity=0.8, cavity="Otvor",
                       closed=True)
        assert d["cavity"]["name"] == "Otvor" and d["cavity"]["closed"] is True and "warning" in d
        assert d["bc"]["label"].startswith("cavity radiation")
        # radiation to ambient on another face coexists; a second face group joins the same cavity
        await call("fea_thermal_add_radiation", case="Ohrev", selection={"z": 5}, emissivity=0.5)
        d2 = await call("fea_thermal_add_cavity", case="Ohrev", selection={"z": 0}, emissivity=0.3, cavity="Otvor")
        assert len(d2["cavity"]["faces"]) == 2 and "warning" not in d2
        await call("fea_thermal_add_cavity", ok=False, case="Ohrev", selection={"x": 0}, emissivity=1.5)
        lst = await call("fea_thermal_list")
        bcs = lst["cases"][0]["bcs"]
        assert sum(1 for b in bcs if b["kind"] == "cavity_radiation") == 2
        name = next(b["name"] for b in bcs if b["kind"] == "cavity_radiation")
        await call("fea_thermal_remove", case="Ohrev", name=name, kind="cavity_radiation")
        s = await call("fea_thermal_suggest_conductance", material_a="steel", material_b="aluminium", pressure_MPa=5)
        assert s["conductance_W_m2K"] > 0 and s["resistance_m2K_W"] == pytest.approx(1 / s["conductance_W_m2K"], rel=1e-4)
        await call("fea_thermal_suggest_conductance", ok=False, material_a="a", material_b="b", pressure_MPa=-1)
    run(body, tmp_path / "mcp")


def test_contact_conductance_argument(tmp_path, plate):
    async def body(call):
        await call("fea_study_new", step_path=str(plate / "blocks.step"), workdir=str(tmp_path / "wd"),
                   material="Ocel S235")
        d = await call("fea_contact_set", pair="AB", a="A", b="B", type="frictionless", conductance_W_m2K=2500.0)
        assert d["thermal_conductance_W_m2K"] == 2500.0
        d = await call("fea_contact_set", pair="AB", conductance_W_m2K="auto", contact_pressure_MPa=10)
        assert d["thermal_conductance_W_m2K"] > 2000 and "estimated" in d["thermal_conductance_note"]
        d = await call("fea_contact_set", pair="AB", reset=["conductance"])
        assert "thermal_conductance_W_m2K" not in d
        await call("fea_contact_set", ok=False, pair="AB", conductance_W_m2K="nonsense")
        await call("fea_contact_set", ok=False, pair="AB", conductance_W_m2K=-5.0)
        await call("fea_contact_set", ok=False, pair="AB", conductance_W_m2K=100.0, contact_pressure_MPa=2)
    run(body, tmp_path / "mcp")


def test_categories_and_sweep_definition(tmp_path, plate):
    async def body(call):
        await call("fea_study_new", step_path=str(plate / "plate.step"), workdir=str(tmp_path / "wd"),
                   material="Ocel S235")
        await call("fea_constraint_add_fixed", name="levy", selection={"x": 0})
        await call("fea_load_add_gravity", case_name="Tiha")
        await call("fea_load_add_force", case_name="Uzitne", name="pravy", selection={"x": 60}, force=[100, 0, 0])
        d = await call("fea_loadcase_set_category", name="Uzitne", category="C")
        assert d["category"] == "cat_C"
        rows = {r["case"]: r for r in d["load_cases"]}
        assert rows["Tiha"]["source"] == "guessed from the name" and rows["Uzitne"]["psi_0"] == 0.7
        await call("fea_loadcase_set_category", ok=False, name="Uzitne", category="xyz")
        await call("fea_loadcase_set_category", ok=False, name="neni", category="B")
        g = await call("fea_combination_generate_from_study", formula="6.10")
        assert g["warnings"] and g["categories"]["Tiha"]["category"] == "permanent"
        await call("fea_loadcase_set_category", name="Tiha", category="permanent")
        g = await call("fea_combination_generate_from_study", formula="6.10")
        assert not g["warnings"] and g["combinations"][0]["factors"] == {"Tiha": 1.35, "Uzitne": 1.5}
        lst = await call("fea_combination_list")
        assert lst["combinations"][0]["case_categories"]["Tiha"]["source"] == "explicit"
        await call("fea_loadcase_set_category", name="Tiha", category=None)
        pytest.importorskip("docx")
        rep = tmp_path / "rep.docx"
        await call("fea_report_word", output_path=str(rep), images=False)
        import docx
        cells = [c.text for t in docx.Document(str(rep)).tables for r in t.rows for c in r.cells]
        assert "GUESSED from the name – check" in cells and "set in the study" in cells
        # sweep definition (no calculation)
        await call("fea_sweep_run", ok=False)                       # nothing defined yet
        d = await call("fea_sweep_define", load_factors=[0.5, 1.0, 2.0])
        assert d["points"] == 3 and d["linear_shortcut"] is True
        d = await call("fea_sweep_define", load_factors=[1.0, 2.0], materials={"Deska": ["Ocel S235", "Ocel S355"]},
                       mesh_sizes=[8.0], max_points=3)
        assert d["points"] == 3 and d["points_full"] == 4 and "warning" in d and d["linear_shortcut"] is False
        info = await call("fea_sweep_define")
        mats = [q for q in info["spec"]["params"] if q["kind"] == "material"]
        assert info["defined"] and [(q["target"], q["values"]) for q in mats] == [("Deska", ["Ocel S235", "Ocel S355"])]
        await call("fea_sweep_define", ok=False, materials={"Neni": ["Ocel S235"]})
        await call("fea_sweep_define", ok=False, materials={"Deska": ["Neexistujici material XYZ"]})
        await call("fea_sweep_define", ok=False, mesh_sizes=[-1.0])
        assert (await call("fea_sweep_define", clear=True))["cleared"]
        # tools that need results fail understandably
        await call("fea_submodel_run", ok=False)             # needs a solved 3D result
        await call("fea_fatigue_strain_life", ok=False, history_curve=[0, 1, 0])
        await call("fea_fatigue_strain_life", ok=False, notch_rule="xyz")
    run(body, tmp_path / "mcp")


@pytest.mark.slow
def test_solve_strain_life_and_sweep(tmp_path, plate):
    """Plate with a hole in tension: ε-N, linear-shortcut sweep (one solve) via a job."""
    async def body(call):
        await call("fea_study_new", step_path=str(plate / "plate.step"), workdir=str(tmp_path / "wd"),
                   material="Ocel S235")
        await call("fea_constraint_add_fixed", name="levy", selection={"x": 0})
        await call("fea_load_add_force", case_name="Tah", name="pravy", selection={"x": 60}, force=[5000, 0, 0])
        await call("fea_mesh_generate", global_size=4.0, timeout_s=300)
        await call("fea_solve_run", timeout_s=600)
        # (the submodel itself is covered by tests/test_submodel_run.py and verification SM1 / SM2)
        en = await call("fea_fatigue_strain_life", history_curve={"load_case": "Tah", "kind": "sine", "amplitude": 1.0,
                                                                   "mean": 0.0, "cycles": 1},
                        notch_rule="glinka", mean_correction="swt", top_n=3)
        assert en["worst"] and en["en_parameters"] and en["min_life_blocks"] > 0
        await call("fea_sweep_define", load_factors=[1.0, 2.0])
        j = await call("fea_sweep_run")
        jid = j["job_id"]
        for _ in range(600):
            st = await call("fea_job_status", job_id=jid)
            if st["status"] not in ("starting", "running"):
                break
            await anyio.sleep(0.5)
        assert st["status"] == "done", st
        r = await call("fea_job_result", job_id=jid)
        p0, p1 = r["points"]
        assert r["linear_shortcut"] is True
        assert p1["max_stress"] == pytest.approx(2 * p0["max_stress"], rel=1e-4)
        assert p1["min_safety"] == pytest.approx(p0["min_safety"] / 2, rel=1e-4)
    run(body, tmp_path / "mcp")


def test_cavity_tools_with_radiation_cavity_conditions(tmp_path, plate):
    """Radiation(cavity=True) conditions (files of the next-1.5 era) are cavity conditions for the tools as they are for the GUI
    and the solver: listed as cavity_radiation, synchronised (closed / ambient), replaced on the same faces, removable with
    kind 'cavity' / 'cavity_radiation'; at most MAX_CAVITIES cavities per state."""
    from fea_core.thermal.model import MAX_CAVITIES, CavityRadiation, Radiation, is_cavity
    sess = Session(tmp_path / "mcp")

    def legacy(bc):
        return Radiation(bc.selection, emissivity=bc.emissivity, ambient=bc.ambient, cavity=True, cavity_name=bc.cavity_name,
                         closed=bc.closed)

    async def main():
        srv = build_server(sess)
        async with Client(srv, mode="legacy") as c:
            async def call(tool, ok=True, **args):
                r = await c.call_tool(tool, args)
                text = "\n".join(getattr(x, "text", "") or "" for x in r.content)
                assert r.is_error is (not ok), f"{tool}: {text}"
                return json.loads(text) if ok else text
            await call("fea_study_new", step_path=str(plate / "plate.step"), workdir=str(tmp_path / "wd"), material="Ocel S235")
            await call("fea_thermal_add_cavity", case="Ohrev", selection={"radius": 5}, emissivity=0.8, cavity="Otvor")
            await call("fea_thermal_add_cavity", case="Ohrev", selection={"z": 0}, emissivity=0.3, cavity="Otvor")
            case = sess.study.thermal.cases[0]
            case.loads = [legacy(b) if isinstance(b, CavityRadiation) else b for b in case.loads]
            assert [b.kind for b in case.loads] == ["radiation", "radiation"] and all(is_cavity(b) for b in case.loads)
            lst = await call("fea_thermal_list")
            bcs = lst["cases"][0]["bcs"]
            assert [b["kind"] for b in bcs] == ["cavity_radiation"] * 2 and all(b["label"].startswith("cavity radiation") for b in bcs)
            assert all("cavity Otvor" in b["value_text"] for b in bcs)
            # a third face of the cavity, closed: closed / ambient are synchronised on the older conditions too
            d = await call("fea_thermal_add_cavity", case="Ohrev", selection={"z": 5}, emissivity=0.5, cavity="Otvor", closed=True,
                           ambient_C=30.0)
            assert len(d["cavity"]["faces"]) == 3 and "note" in d
            assert all(b.closed and b.ambient == 30.0 for b in case.loads) and len(case.loads) == 3
            # the same faces again: replaced (not duplicated next to the older Radiation(cavity=True))
            old_name = bcs[0]["name"]
            d = await call("fea_thermal_add_cavity", case="Ohrev", selection=old_name, emissivity=0.6, cavity="Otvor", closed=True,
                           ambient_C=30.0)
            assert d["replaced"] is True and len(case.loads) == 3
            assert [b.kind for b in case.loads].count("cavity_radiation") == 2 and [b.kind for b in case.loads].count("radiation") == 1
            # the aliases remove the cavity conditions (new and older ones)
            await call("fea_thermal_remove", case="Ohrev", name=bcs[1]["name"], kind="cavity")
            assert len(case.loads) == 2
            await call("fea_thermal_remove", case="Ohrev", name=old_name, kind="cavity_radiation")
            assert len(case.loads) == 1
            await call("fea_thermal_remove", ok=False, case="Ohrev", name=old_name, kind="cavity")
            # the limit of cavities per state is enforced when the condition is added (not only by the readiness check)
            case.loads = [CavityRadiation(f"s{i}", cavity_name=f"C{i}") for i in range(MAX_CAVITIES)]
            text = await call("fea_thermal_add_cavity", ok=False, case="Ohrev", selection={"z": 0}, emissivity=0.5, cavity="One more")
            assert str(MAX_CAVITIES) in text
            await call("fea_thermal_add_cavity", case="Ohrev", selection={"z": 0}, emissivity=0.5, cavity="C3")   # an existing one
    anyio.run(main)

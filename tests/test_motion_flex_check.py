"""A flexible body of a mechanism needs NGSolve (module ngsolve, which the Standard profile does not carry since release 1.4.2): the check tells it
at once with the install hint – in the model check shared by the GUI and MCP, before the mesh and the process of the calculation – instead of
"ModuleNotFoundError: No module named 'ngsolve'" from the worker. No Exudyn, Gmsh or NGSolve is needed here.
"""
from __future__ import annotations

import json

import pytest

from fea_core import modules
from fea_core.motion.model import Body, MotionJoint, MotionRun
from fea_core.motion.run import MotionError, check_model, run_motion
from fea_core.study import Material, Study


@pytest.fixture
def study():
    st = Study("flex", "")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9), default=True)
    m = st.mechanism
    m.bodies = [Body("frame", ground=True), Body("lever", parts=["Lever"], flexible=True, modes=6)]
    m.joints = [MotionJoint("pin", "revolute", "frame", "lever", [0, 0, 0], [0, 0, 1])]
    m.run = MotionRun("dynamic", t_end=0.1, outputs=10)
    return st


def _without_ngsolve(monkeypatch):
    real = modules.available
    monkeypatch.setattr(modules, "available", lambda key: False if key == "ngsolve" else real(key))


def _with_ngsolve(monkeypatch):
    real = modules.available
    monkeypatch.setattr(modules, "available", lambda key: True if key == "ngsolve" else real(key))


def test_check_model_names_the_missing_module_and_how_to_install_it(study, monkeypatch):
    _without_ngsolve(monkeypatch)
    errs, _ = check_model(study)
    miss = [e for e in errs if "ngsolve" in e]
    assert len(miss) == 1
    assert "'lever'" in miss[0] and "python -m fea_core.modules install ngsolve" in miss[0] and "Tools → Program modules" in miss[0]
    assert "ModuleNotFoundError" not in miss[0]


def test_check_model_is_silent_when_ngsolve_is_installed_or_not_needed(study, monkeypatch):
    _with_ngsolve(monkeypatch)
    assert not [e for e in check_model(study)[0] if "ngsolve" in e]
    _without_ngsolve(monkeypatch)
    study.mechanism.bodies[1].flexible = False                       # a rigid mechanism does not need NGSolve
    assert not [e for e in check_model(study)[0] if "ngsolve" in e]
    study.mechanism.bodies[1].flexible = True
    assert not [e for e in check_model(study, run=False)[0] if "ngsolve" in e]      # the mobility check builds no FEM matrices
    study.mechanism.run.solver = "chrono"                            # Chrono has no flexible bodies at all – its own error, not a second one
    assert not [e for e in check_model(study)[0] if "ngsolve" in e]


def test_run_stops_before_the_mesh_and_the_worker(study, monkeypatch, tmp_path):
    """The calculation refuses at the check: the geometry is not loaded, the body is not meshed and no process starts."""
    _without_ngsolve(monkeypatch)
    import fea_core.motion.run as run_mod

    def boom(*a, **k):
        raise AssertionError("the calculation went past the check")
    monkeypatch.setattr(run_mod, "_load_parts", boom)
    monkeypatch.setattr(run_mod, "prepare_job", boom)
    monkeypatch.setattr(run_mod, "_run_worker", boom)
    with pytest.raises(MotionError, match="install ngsolve"):
        run_motion(study, tmp_path)


def test_worker_has_the_same_message_as_a_safety_net(monkeypatch):
    """A job that reached the worker without the check (an older main process): `need` raises ModuleMissing with the install hint, which the worker
    passes on as it is (not 'ModuleNotFoundError: No module named ...')."""
    pytest.importorskip("exudyn")
    real_import = modules.importlib.import_module

    def no_ngsolve(name, *a, **k):
        if name == "ngsolve":
            raise ModuleNotFoundError("No module named 'ngsolve'", name="ngsolve")
        return real_import(name, *a, **k)
    monkeypatch.setattr(modules.importlib, "import_module", no_ngsolve)
    from fea_core.motion.exudyn_adapter import Model
    with pytest.raises(modules.ModuleMissing, match="install ngsolve"):
        Model._flex_fem(object.__new__(Model), {"name": "lever"}, None)


def test_the_worker_passes_a_missing_module_message_unchanged():
    """result.json gets the plain message for ModuleMissing (the class name is in the whitelist of the worker)."""
    from pathlib import Path
    src = (Path(modules.__file__).parent / "motion" / "worker.py").read_text(encoding="utf-8")
    assert '"ModuleMissing"' in src
    assert json.dumps(str(modules.ModuleMissing("ngsolve")))            # the message is serialisable

"""Review fixes of the state cache (`fea_core.state_cache`, `gravity_states`, `direct/modal_cache`, `dynamics/modal`): what the keys cover, what an entry has to prove before it is used,
what a solver fallback does to the unit states, and the one-run rule of the eigenvalue analysis.

Each test pins one finding of the two reviews of branch opt2-151 (1 HIGH … 12 LOW)."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verification import cases as _C  # noqa: F401  (first the cases: cyclic import of the others)
from fea_core import config, gravity_states as GS, state_cache as SC
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.solve import prepare_mesh, run_study
from fea_core.study import Displacement, Fixed, Gravity, LoadCase, MeshSettings, RemoteLoad
from verification.cases import _study, box

HAS_DIRECT = importlib.util.find_spec("rslab") is not None and importlib.util.find_spec("numba") is not None
needs_direct = pytest.mark.skipif(not HAS_DIRECT, reason="numba + rslab (modules accel, direct) are needed")
needs_ccx = pytest.mark.skipif(not config.CCX_EXE.exists(), reason="CalculiX is needed")
G = 9810.0


@pytest.fixture(scope="module", autouse=True)
def _rslab_on_the_path_of_the_worker():
    mp = pytest.MonkeyPatch()
    spec = importlib.util.find_spec("rslab")
    if spec is not None and spec.origin:
        here = str(Path(spec.origin).parent.parent)
        mp.setenv("PYTHONPATH", os.pathsep.join([here, *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]]))
    yield
    mp.undo()


def _beam(wd: Path, dims=(60, 20, 10), size=5.0):
    st, parts = _study("beam", wd / "model.step", [Part("Nosnik", box(0, 0, 0, *dims))], MeshSettings(size=size))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("end", find_faces(parts, x=dims[0]))
    st.add_selection("bottom", find_faces(parts, z=0))
    st.constraints.append(Fixed("vetknuti"))
    return st, parts


def _gravity(st, n: int = 4) -> None:
    for i, v in enumerate([[G, 0, 0], [-G, 0, 0], [0, G, 0], [0, 0, -G], [2 * G, 0, G], [0, -G, G]][:n]):
        st.load_cases.append(LoadCase(f"G{i}", [Gravity(vector=v)]))


@pytest.fixture(scope="module")
def beam(tmp_path_factory):
    wd = tmp_path_factory.mktemp("state_cache_fixes")
    st, parts = _beam(wd)
    st.load_cases.append(LoadCase("meshing", [Gravity(vector=[0.0, 0.0, -G])]))          # (a study needs a case to be meshed; the tests set their own)
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.load_cases = []
    return st, mesh, wd


def _run(st, mesh, wd, tag, cache=True):
    os.environ["FEA_STATE_CACHE"] = "1" if cache else "0"
    try:
        return run_study(st, wd / tag, study_dir=wd, on_log=lambda s: None, mesh=mesh)
    finally:
        os.environ.pop("FEA_STATE_CACHE", None)


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))


# ============================================================================================================================ 1: the unit states of the model that was solved
@needs_ccx
def test_unit_states_solved_without_the_rigid_load_of_another_case_are_not_taken_by_a_deck_that_has_it(beam, tmp_path):
    """A (the dynamics first): the unit statics of a deck of the unit cases alone, then a run whose deck also holds a rigid remote load – the stored states are of another model."""
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "ccx"
    st.load_cases = [LoadCase("Gz", [Gravity(vector=[0, 0, -G])]), LoadCase("R", [RemoteLoad("end", [80.0, 10.0, 5.0], force=[0.0, 0.0, -100.0], coupling="rigid")])]
    units = [GS.unit_case(a, 1.0) for a in range(3)]
    SC.clear()
    assert GS.static_states(st, mesh, units, tmp_path / "dyn", lambda s: None) is not None
    ref = _run(st, mesh, tmp_path, "a_ref", cache=False)
    SC.clear()
    GS.static_states(st, mesh, units, tmp_path / "dyn2", lambda s: None)
    got = _run(st, mesh, tmp_path, "a_got", cache=True)
    assert _rel(got.steps[0].U, ref.steps[0].U) < 1e-9 and _rel(got.steps[0].S, ref.steps[0].S) < 1e-9          # the case Gz is the solved one, not a combination of foreign states
    # the keys differ: the model with a rigid body is another model
    assert SC.solved_digest(st, mesh, [GS.unit_case(0, G)]) != SC.solved_digest(st, mesh, [GS.unit_case(0, G), st.load_cases[1]])


@needs_ccx
def test_unit_states_solved_with_the_rigid_load_of_another_case_are_not_taken_by_the_unit_statics(beam, tmp_path):
    """B (the run first): four gravity cases + a rigid remote load store unit states of the deck WITH the rigid body; the unit statics (units alone) must not use them."""
    from fea_core.dynamics.modal import static_solutions
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "ccx"
    _gravity(st, 4)
    st.load_cases.append(LoadCase("R", [RemoteLoad("end", [80.0, 10.0, 5.0], force=[0.0, 0.0, -100.0], coupling="rigid")]))
    SC.clear()
    r = _run(st, mesh, tmp_path, "b_run", cache=True)
    assert r.info["gravity_superposition"]["solved"] == ["x", "y", "z"]
    units = [GS.unit_case(a, 1.0) for a in range(3)]
    new = GS.static_states(st, mesh, units, tmp_path / "b_dyn", lambda s: None)
    (tmp_path / "b_old").mkdir()
    old = static_solutions(st, mesh, units, tmp_path / "b_old", lambda s: None)
    for a, b in zip(new, old):
        assert _rel(a["U"], b["U"]) < 1e-5                                    # the 6 digits of the .frd, not the 2–26 % of a model with a rigid body in it
    assert (tmp_path / "b_dyn" / "grav_units.inp").exists()                  # it was SOLVED: the stored states of the other model were not used


def test_the_digest_of_a_solved_deck_counts_the_rigid_loads_of_its_cases_only(beam):
    st, mesh, _ = beam
    rigid = LoadCase("R", [RemoteLoad("end", [80.0, 10.0, 5.0], force=[0.0, 0.0, -100.0], coupling="rigid")])
    soft = LoadCase("S", [RemoteLoad("end", [80.0, 10.0, 5.0], force=[0.0, 0.0, -100.0], coupling="distributing")])
    unit = [GS.unit_case(0, G)]
    assert SC.solved_digest(st, mesh, unit + [soft]) == SC.solved_digest(st, mesh, unit)
    assert SC.solved_digest(st, mesh, unit + [rigid]) != SC.solved_digest(st, mesh, unit)


# ============================================================================================================================ 2 + 3: the eigenvalue analysis, free / held bodies, windows
def _modal_calls(monkeypatch):
    from fea_core import direct
    calls: list[dict] = []
    real = direct.run_eigen_prepared

    def counting(job, opts, *a, **k):
        calls.append(dict(opts))
        return real(job, opts, *a, **k)
    monkeypatch.setattr(direct, "run_eigen_prepared", counting)
    return calls


@needs_direct
def test_a_free_and_a_partly_held_body_can_be_analysed_twice_in_one_work_folder(beam, tmp_path):
    from fea_core.dynamics.modal import run_basis
    st0, mesh, _ = beam
    for name, cons in (("free", []), ("one direction", [Displacement("bottom", uz=0.0)])):
        st = copy.deepcopy(st0)
        st.constraints = cons
        st.solver.modal_engine = "direct"
        st.dynamics.enabled, st.dynamics.modes = True, 8
        SC.clear()
        wd = tmp_path / name.replace(" ", "_")
        first = run_basis(st, mesh, wd, lambda s: None)
        for rec in Path(wd).glob("state_cache/modes_*/rec_*"):               # the meta never promises a file that is not there
            meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
            assert all((rec / f"{f}.npy").is_file() for f in meta["files"]), (name, rec)
        second = run_basis(st, mesh, wd, lambda s: None)                      # the same work folder again: the stored recovery is read
        assert np.array_equal(first.freq, second.freq) and np.array_equal(first.stress, second.stress), name
        assert second.engine_info.get("state_cache")


@needs_direct
def test_a_window_above_zero_asks_for_no_unit_columns_and_runs_once(beam, tmp_path, monkeypatch):
    from fea_core.direct import modal_run
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.modal_engine = "direct"
    SC.clear()
    calls = _modal_calls(monkeypatch)
    base = modal_run.run_modes(st, mesh, tmp_path / "m0", lambda s: None, modes=6)
    f_lo = float(0.5 * (base.freq[1] + base.freq[2]))
    calls.clear()
    t = time.time()
    win = modal_run.run_modes(st, mesh, tmp_path / "m1", lambda s: None, modes=4, f_min=f_lo)
    assert len(calls) == 1 and calls[0]["statics"] is False                   # ONE eigenvalue run, no columns beside a window above zero
    assert np.allclose(win.freq, base.freq[2:6], rtol=1e-7) and time.time() - t < 60
    calls.clear()
    modal_run.run_modes(st, mesh, tmp_path / "m2", lambda s: None, modes=2, f_min=f_lo)         # the same window again: stored
    assert calls == []


@needs_direct
def test_a_partly_held_body_loses_the_unit_columns_in_the_same_run_and_the_entry_serves_the_next_call(beam, tmp_path, monkeypatch):
    from fea_core.direct import modal_run
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.constraints = [Displacement("bottom", uz=0.0)]                           # held in one direction: K is singular, the static solution of the unit accelerations is not possible
    st.solver.modal_engine = "direct"
    SC.clear()
    calls = _modal_calls(monkeypatch)
    a = modal_run.run_modes(st, mesh, tmp_path / "p1", lambda s: None, modes=8)
    assert len(calls) == 1 and calls[0]["statics"] is True and calls[0]["statics_optional"] is True        # tried beside the modes, in the one run
    assert a.info["eigen"].get("statics_skipped") and not a.columns_all
    b = modal_run.run_modes(st, mesh, tmp_path / "p2", lambda s: None, modes=8)                 # asked for the columns again: the entry says it has none, no second analysis
    assert len(calls) == 1 and b.info["cache"]["source"] == "stored analysis"


@needs_direct
def test_the_static_solutions_of_explicit_cases_are_not_optional_and_a_memory_error_is_never_repeated(beam, tmp_path, monkeypatch):
    from fea_core import direct
    from fea_core.direct import modal_run
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.modal_engine = "direct"
    SC.clear()
    calls = _modal_calls(monkeypatch)
    modal_run.run_modes(st, mesh, tmp_path / "e1", lambda s: None, modes=4, static_cases=[GS.unit_case(2, 1.0)])
    assert calls[-1]["statics_optional"] is False
    n = len(calls)

    def memory_error(*a, **k):
        calls.append({})
        raise direct.DirectMemoryError("out of memory")
    monkeypatch.setattr(direct, "run_eigen_prepared", memory_error)
    SC.clear()
    with pytest.raises(direct.DirectMemoryError):
        modal_run.run_modes(st, mesh, tmp_path / "e2", lambda s: None, modes=4)
    assert len(calls) == n + 1                                                 # once


# ============================================================================================================================ 4: what the keys cover
def test_the_digests_cover_the_mesh_content_and_the_solver_that_makes_the_numbers(beam, monkeypatch):
    st, mesh, _ = beam
    base = SC.study_digest(st, mesh)
    assert SC.study_digest(st, mesh) == base

    def with_mesh(**kw):
        m = copy.copy(mesh)
        for k, v in kw.items():
            setattr(m, k, v)
        return m
    ties = {**mesh.report, "ties": [{"slave": "a", "master": "b", "n": 3}]}
    edges = {"e": np.array([[0, 1, 2]])}
    points = {"p": np.array([3])}
    cell = {"print_shell": np.ones(len(mesh.tets))}
    alias = np.arange(len(mesh.nodes))[::-1].copy()
    for what, m in (("tie table", with_mesh(report=ties)), ("edge selection", with_mesh(edges=edges)), ("point selection", with_mesh(points=points)),
                    ("per-element data", with_mesh(cell_data=cell)), ("node aliases", with_mesh(node_alias=alias)), ("node data", with_mesh(node_data={"t": np.zeros(len(mesh.nodes))}))):
        assert SC.study_digest(st, m) != base, what
    for var, value in (("FEA_FAER", "0"), ("FEA_DIRECT_ENGINE", "rslab"), ("FEA_DIRECT_PRECISION", "float64"), ("FEA_MODAL_MIN_WORK", "123"), ("FEA_ASSEMBLY", "ngsolve"),
                       ("FEA_CCX", "x.exe")):
        monkeypatch.setenv(var, value)
        assert SC.study_digest(st, mesh) != base, var
        monkeypatch.undo()
        assert SC.study_digest(st, mesh) == base, var
    monkeypatch.setattr(SC, "SCHEMA", SC.SCHEMA + 1)
    assert SC.study_digest(st, mesh) != base                                    # the layout of the cache
    monkeypatch.undo()
    real_id = SC.solver_identity
    for field, value in (("ccx", ["other.exe", 1, 2]), ("faer", False), ("program", "0.0.0"), ("direct_engine", "other")):
        monkeypatch.setattr(SC, "solver_identity", lambda field=field, value=value: {**real_id(), field: value})
        assert SC.study_digest(st, mesh) != base, field                         # the executable (path, size, time), the faer switch, the program version, the engine
        monkeypatch.undo()


def test_the_mesh_convergence_key_is_the_same_helper(beam):
    from fea_core.space._checks_conv import mesh_key
    _, mesh, _ = beam
    assert mesh_key(mesh) == SC.mesh_digest(mesh)


# ============================================================================================================================ 5 + 7: what an entry has to prove
def _entry(tmp_path, digest="d" * 40, tag="t", values=None):
    arrays = {"a": np.arange(12.0).reshape(4, 3) if values is None else values}
    return SC.write_entry(tmp_path, "test", digest, tag, arrays, {"k": 1}), arrays


def test_an_existing_entry_is_validated_not_trusted(tmp_path):
    SC.clear()
    f1, arrays = _entry(tmp_path)
    assert f1 is not None and SC.verified(f1)
    f2, _ = _entry(tmp_path)                                                    # the same content: the first stays
    assert f2 == f1
    # another content under the same key (a rerun that found other modes): both valid – the new one gets a name of its own, the old one may be mapped by a reader
    f3, _ = _entry(tmp_path, values=np.arange(12.0).reshape(4, 3) * 2)
    assert f3 != f1 and SC.verified(f1) and SC.verified(f3)
    assert np.array_equal(SC.array(f3, "a"), np.arange(12.0).reshape(4, 3) * 2)
    # a damaged one (content changed behind our back) is replaced by the same key
    SC.clear()
    f4, _ = _entry(tmp_path, tag="dmg")
    np.save(f4 / "a.npy", np.zeros((4, 3)))
    assert not SC.verified(f4)
    f5, _ = _entry(tmp_path, tag="dmg")
    assert f5 == f4 and SC.verified(f5) and np.array_equal(SC.array(f5, "a"), np.arange(12.0).reshape(4, 3))
    # a partial one (a file missing) is replaced too
    SC.clear()
    f6, _ = _entry(tmp_path, tag="part")
    (f6 / "a.npy").unlink()
    f7, _ = _entry(tmp_path, tag="part")
    assert f7 == f6 and SC.verified(f7)
    # an entry of an old schema / program version is not read and is replaced
    SC.clear()
    f8, _ = _entry(tmp_path, tag="old")
    meta = json.loads((f8 / "meta.json").read_text(encoding="utf-8"))
    for key, value in (("schema", SC.SCHEMA - 1), ("version", "0.0.1")):
        (f8 / "meta.json").write_text(json.dumps({**meta, key: value}), encoding="utf-8")
        assert SC.meta_of(f8) is None and not SC.verified(f8)
        f9, _ = _entry(tmp_path, tag="old")
        assert f9 == f8 and SC.verified(f9)


def test_a_concurrent_writer_that_got_there_first_is_validated_and_registered(tmp_path, monkeypatch):
    SC.clear()
    real = SC._rename
    first, _ = _entry(tmp_path, digest="c" * 40, tag="race")
    SC.clear()                                                                  # (another process: nothing registered here)

    def exists(src, dst):
        raise FileExistsError(str(dst))
    monkeypatch.setattr(SC, "_rename", exists)
    f, _ = _entry(tmp_path, digest="c" * 40, tag="race")
    monkeypatch.setattr(SC, "_rename", real)
    assert f == first and SC.entries("test", "c" * 40) == [first]
    assert not list((tmp_path / "state_cache").glob(".tmp_*"))


def test_a_damaged_array_is_detected_when_the_entry_is_looked_up_and_dropped(tmp_path):
    SC.clear()
    f, _ = _entry(tmp_path, digest="b" * 40)
    assert SC.entries("test", "b" * 40) == [f]
    time.sleep(0.02)
    np.save(f / "a.npy", np.arange(12.0).reshape(4, 3) + 1.0)                     # the same shape and type, other numbers
    assert SC.entries("test", "b" * 40) == []


def test_old_entries_of_other_keys_are_pruned_per_work_folder(tmp_path, monkeypatch):
    SC.clear()
    monkeypatch.setattr(SC, "KEEP_OTHER", 2)
    monkeypatch.setattr(SC, "RECENT_KEYS", 1)                                    # (only the key just written is in use here)
    made = []
    for i in range(5):
        f, _ = _entry(tmp_path, digest=f"{i:x}" * 40)
        made.append(f)
        os.utime(f, (time.time() + i, time.time() + i))
    left = [p.name for p in (tmp_path / "state_cache").iterdir()]
    assert made[-1].name in left and len(left) == 3                              # the key just written and the two newest of the others


@needs_direct
def test_a_recovery_is_never_read_for_modes_it_was_not_computed_from(beam, tmp_path):
    """Fable finding 4: a fresh eigenvalue run in a work folder that holds the stored recovery of an earlier run (modes of other signs) must not get that recovery."""
    from fea_core.dynamics.modal import run_basis
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.modal_engine = "direct"
    st.dynamics.enabled, st.dynamics.modes = True, 6
    SC.clear()
    wd = tmp_path / "w"
    b1 = run_basis(st, mesh, wd, lambda s: None)
    rec = next(Path(wd).glob("state_cache/modes_*/rec_*"))
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    assert meta["modes_content"]
    # forge a recovery of other modes: the stored stress doubled, the binding to the modes of the entry kept as it was – the content hash of the file now disagrees
    st_file = rec / "stress.npy"
    np.save(st_file, 2.0 * np.load(st_file))
    SC.clear()
    b2 = run_basis(st, mesh, wd, lambda s: None)
    assert np.array_equal(b1.stress, b2.stress)                                  # (the damaged recovery was not served: the basis was recovered again)


@needs_ccx
def test_every_reuse_is_logged_with_the_name_of_the_entry(beam, tmp_path):
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "ccx"
    _gravity(st, 4)
    SC.clear()
    log: list[str] = []
    run_study(st, tmp_path / "l1", study_dir=tmp_path, on_log=lambda s: None, mesh=mesh)
    run_study(st, tmp_path / "l2", study_dir=tmp_path, on_log=log.append, mesh=mesh)
    assert any("stored unit states used" in s and "gravunits_" in s for s in log), log


# ============================================================================================================================ 6: a solver fallback
@needs_ccx
def test_a_direct_attempt_that_fails_leaves_a_ccx_family_state_and_no_second_solve(beam, tmp_path, monkeypatch):
    from fea_core import solve, solve_large
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "auto"
    _gravity(st, 6)
    calls: list[str] = []
    real_ccx = solve._ccx_job

    def failing(study, *a, **k):
        calls.append(f"direct({len(study.load_cases)})")
        raise RuntimeError("injected: the direct solver does not fit into the memory")

    def ccx(study, *a, **k):
        calls.append(f"ccx({len(study.load_cases)})")
        return real_ccx(study, *a, **k)
    monkeypatch.setattr(solve_large, "run_ngs", failing)
    monkeypatch.setattr(solve, "_ccx_job", ccx)
    monkeypatch.setattr(solve, "decide_backend", lambda study, mesh_, log_=None: ("ngs_direct", "forced by the test"))
    SC.clear()
    on = _run(st, mesh, tmp_path, "fb_on", cache=True)
    n_on = list(calls)
    assert on.info["backend"] == "ccx" and on.info["gravity_superposition"]["family"] == "ccx"
    for name in on.info["gravity_superposition"]["solved"]:
        assert GS._entry(SC.solved_digest(st, mesh, []), "xyz".index(name), "ccx") is not None
        assert GS._entry(SC.solved_digest(st, mesh, []), "xyz".index(name), "own") is None          # never stored under the own family
    calls.clear()
    SC.clear()
    off = _run(st, mesh, tmp_path, "fb_off", cache=False)
    assert len(n_on) <= len(calls)                                               # the cache never costs a solver call: fail + one ccx job each
    for a, b in zip(on.steps, off.steps):
        assert _rel(a.U, b.U) < 1e-5


def test_two_solvers_are_never_mixed_in_one_superposition():
    pl = GS.Plan("d", "own", [], [0, 1], {0: Path("x")}, [1])
    res = SimpleNamespace(info={"backend": "ccx"})
    with pytest.raises(ValueError):
        GS.settle_family(pl, res)
    pl2 = GS.Plan("d", "own", [], [0, 1], {}, [0, 1])
    GS.settle_family(pl2, res)
    assert pl2.family == "ccx"


# ============================================================================================================================ 8: the ccx basis is looked at only where ccx is the engine
@needs_ccx
def test_a_stored_ccx_basis_is_not_looked_at_when_the_decision_is_the_direct_analysis(beam, tmp_path, monkeypatch):
    from fea_core.direct import modal_run
    from fea_core.dynamics import modal as DM
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.dynamics.enabled, st.dynamics.modes = True, 4
    looked = []
    monkeypatch.setattr(DM, "_ccx_basis_lookup", lambda key, log=None: looked.append(key) or None)
    monkeypatch.setattr(modal_run, "decide_full", lambda *a, **k: ("direct", "forced", True))
    monkeypatch.setattr(DM, "_arrays_direct", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    with pytest.raises(RuntimeError):
        DM.run_basis(st, mesh, tmp_path / "d", lambda s: None)
    assert looked == []
    monkeypatch.setattr(modal_run, "decide_full", lambda *a, **k: ("ccx", "forced", False))
    monkeypatch.setattr(DM, "_arrays_ccx", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    with pytest.raises(RuntimeError):
        DM.run_basis(st, mesh, tmp_path / "c", lambda s: None)
    assert len(looked) == 1


# ============================================================================================================================ 9: nothing stays mapped
def test_a_basis_that_is_not_asked_mapped_holds_no_memory_map(tmp_path):
    from fea_core.dynamics.modal import _resident
    SC.clear()
    f, _ = _entry(tmp_path, digest="a" * 40, tag="map", values=np.arange(30.0).reshape(10, 3))
    mapped = SC.array(f, "a")[:4]
    own = _resident(mapped, False)
    assert not isinstance(own, np.memmap) and own.base is None or not isinstance(own.base, np.memmap)
    assert _resident(mapped, True) is mapped
    plain = np.zeros(3)
    assert _resident(plain, False) is plain
    del mapped, own


# ============================================================================================================================ 10: the lock
def test_the_compute_lock_is_cancellable_and_leaves_no_table_entry():
    from fea_core.direct import DirectError, modal_cache as MCH
    cancel = threading.Event()
    holder_in = threading.Event()
    release = threading.Event()

    def hold():
        with MCH.compute_lock("dg"):
            holder_in.set()
            release.wait(10)
    t = threading.Thread(target=hold)
    t.start()
    assert holder_in.wait(5)
    seen: list[str] = []
    threading.Timer(0.5, cancel.set).start()
    t0 = time.time()
    with pytest.raises(DirectError, match="canceled by user"):
        with MCH.compute_lock("dg", cancel, seen.append):
            pass
    assert time.time() - t0 < 5 and seen                                         # the wait ended with the cancel and said it was waiting
    release.set()
    t.join(5)
    assert MCH._COMPUTE == {}                                                    # nothing kept for a digest nobody holds
    with MCH.compute_lock("dg"):                                                 # re-entrant for one thread
        with MCH.compute_lock("dg"):
            pass
    assert MCH._COMPUTE == {}


# ============================================================================================================================ 11: the notes of the missing STR estimate
def test_the_str_notes_do_not_send_a_ccx_user_to_ccx_for_a_superposed_state():
    from fea_core.mcp.diagnostics import no_error_estimate_note
    sup = SimpleNamespace(info={"backend": "ccx", "gravity_superposition": {"cases": ["G"]}})
    note = no_error_estimate_note(sup)
    assert "combinations of unit states" in note and "backend='ccx'" not in note
    assert "backend='ccx'" in no_error_estimate_note(SimpleNamespace(info={"backend": "ngs_direct"}))
    from fea_core import convergence
    src = Path(convergence.__file__).read_text(encoding="utf-8")
    assert "combinations of unit states" in src


# ============================================================================================================================ 12: the lows
@needs_ccx
def test_the_reference_point_displacements_and_the_duplicate_case_names_survive_the_superposition(beam, tmp_path):
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "ccx"
    _gravity(st, 4)
    st.load_cases[1].name = st.load_cases[0].name                               # two cases with one name
    st.load_cases.append(LoadCase("R", [RemoteLoad("end", [80.0, 10.0, 5.0], force=[0.0, 0.0, -100.0], coupling="distributing")]))
    SC.clear()
    sup = _run(st, mesh, tmp_path, "r_sup", cache=True)
    ref = _run(st, mesh, tmp_path, "r_ref", cache=False)
    assert sup.info.get("gravity_superposition") and len(sup.steps) == len(ref.steps)
    for a, b in zip(sup.steps, ref.steps):
        assert _rel(a.U, b.U) < 1e-5 and set(a.ref_displacements) == set(b.ref_displacements), a.name
        for k, v in b.ref_displacements.items():
            assert np.abs(a.ref_displacements[k] - v).max() <= 1e-5 * max(np.abs(v).max(), 1e-12) + 1e-14, (a.name, k)
    assert sup.steps[0].U is not sup.steps[1].U and not np.array_equal(sup.steps[0].U, sup.steps[1].U)       # the two cases of one name are two states (G0 = +x, G0 = −x)


def test_a_case_named_like_a_unit_state_is_never_planned_and_the_other_refusals(beam):
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    _gravity(st, 4)
    SC.clear()
    assert GS.plan(st, mesh, "ccx") is not None
    st.load_cases.append(LoadCase("_unit_gravity_x", [Gravity(vector=[0, G, 0])]))
    assert GS.plan(st, mesh, "ccx") is None
    two = copy.deepcopy(st0)
    two.twod.enabled = True
    assert GS.superposable(two, mesh) is not None
    shell = copy.deepcopy(st0)
    shell.shell.enabled = True
    assert GS.superposable(shell, mesh) is not None


def test_the_superposition_note_survives_save_and_load():
    from fea_core.results import INFO_KEYS
    assert "gravity_superposition" in INFO_KEYS


@needs_ccx
def test_the_section_stub_of_a_unit_state_is_shared_by_every_combined_state(beam, tmp_path):
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "ccx"
    _gravity(st, 4)
    SC.clear()
    r = _run(st, mesh, tmp_path, "stub", cache=True)
    stubs = [{id(stub) for stub, _ in s._combo} for s in r.steps if hasattr(s, "_combo")]
    assert len(stubs) == 4 and all(len(x) >= 1 for x in stubs)
    assert {i for x in stubs for i in x} == set.union(*stubs) and len(set.union(*stubs)) <= 3       # at most one stub per axis, not one per state


def test_the_state_cache_off_is_the_old_path(beam, monkeypatch):
    st, mesh, _ = beam
    monkeypatch.setenv("FEA_STATE_CACHE", "0")
    assert not SC.enabled() and GS.plan(st, mesh, "ccx") is None and GS.static_states(st, mesh, [GS.unit_case(0)], Path("."), lambda s: None) is None

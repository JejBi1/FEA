"""Second round of review fixes of the state cache (branch opt2-151): what two analyses of one model share, what an entry of the same name has to prove when there is no room to
write a new one, what a basis without reactions does not compute, the lock of the compute table, hashing and pruning, and the notes of the missing STR estimate.

Each test pins one finding of the second review (1 HIGH … 8 LOW)."""
from __future__ import annotations

import copy
import importlib.util
import os
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
from fea_core.study import Fixed, Gravity, LoadCase, MeshSettings, RemoteLoad
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


@pytest.fixture(scope="module")
def beam(tmp_path_factory):
    wd = tmp_path_factory.mktemp("state_cache_fixes2")
    dims = (60, 20, 10)
    st, parts = _study("beam", wd / "model.step", [Part("Nosnik", box(0, 0, 0, *dims))], MeshSettings(size=5.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("end", find_faces(parts, x=dims[0]))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("meshing", [Gravity(vector=[0.0, 0.0, -G])]))
    mesh, _ = prepare_mesh(st, wd, wd, lambda s: None, parts=parts)
    st.load_cases = []
    return st, mesh, wd


def _gravity(st, n: int = 4) -> None:
    for i, v in enumerate([[G, 0, 0], [-G, 0, 0], [0, G, 0], [0, 0, -G], [2 * G, 0, G], [0, -G, G]][:n]):
        st.load_cases.append(LoadCase(f"G{i}", [Gravity(vector=v)]))


def _run(st, mesh, wd, tag, cache=True, log=None):
    os.environ["FEA_STATE_CACHE"] = "1" if cache else "0"
    try:
        return run_study(st, wd / tag, study_dir=wd, on_log=log or (lambda s: None), mesh=mesh)
    finally:
        os.environ.pop("FEA_STATE_CACHE", None)


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))


# ============================================================================================================================ 1: the plan and the unit statics share their states
def test_the_bolt_preload_switch_is_not_part_of_the_key_without_bolts(beam):
    st0, mesh, _ = beam
    a, b = copy.deepcopy(st0), copy.deepcopy(st0)
    a.joints.preload_case, b.joints.preload_case = True, False
    assert SC.study_digest(a, mesh) == SC.study_digest(b, mesh)
    for st in (a, b):                                                            # the model of the plan (no other cases) and the model of the unit statics are one
        assert SC.solved_digest(st, mesh, []) == SC.study_digest(GS.unit_model(st, [0, 1, 2]), mesh)
    assert GS.unit_model(a, [0]).joints.preload_case is True                     # (nothing of the study is changed for the deck)
    c, d = copy.deepcopy(st0), copy.deepcopy(st0)
    c.active_bolts = d.active_bolts = lambda: [object()]                          # with an active bolt the switch changes the model: another key
    c.joints.preload_case, d.joints.preload_case = True, False
    assert SC.study_digest(c, mesh) != SC.study_digest(d, mesh)


@needs_ccx
def test_the_run_and_the_unit_statics_share_their_unit_states_in_both_orders(beam, tmp_path):
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.backend = "ccx"
    assert st.joints.preload_case is True                                         # (the default, the one that made the keys differ)
    _gravity(st, 4)
    units = [GS.unit_case(a, 1.0) for a in range(3)]
    # the run first, then the dynamics: no CalculiX job for the unit statics
    SC.clear()
    r = _run(st, mesh, tmp_path, "a_run")
    assert r.info["gravity_superposition"]["solved"] == ["x", "y", "z"]
    seen: list[str] = []
    ss = GS.static_states(st, mesh, units, tmp_path / "a_dyn", seen.append)
    assert ss is not None and any("stored unit states used" in s for s in seen)
    assert not (tmp_path / "a_dyn" / "grav_units.inp").exists()
    # the dynamics first, then the run: the run solves nothing
    SC.clear()
    GS.static_states(st, mesh, units, tmp_path / "b_dyn", lambda s: None)
    assert (tmp_path / "b_dyn" / "grav_units.inp").exists()
    r2 = _run(st, mesh, tmp_path, "b_run")
    sup = r2.info["gravity_superposition"]
    assert sup["stored"] == ["x", "y", "z"] and sup["solved"] == []
    ref = _run(st, mesh, tmp_path, "b_ref", cache=False)
    for a, b in zip(r2.steps, ref.steps):
        assert _rel(a.U, b.U) < 1e-5 and _rel(a.S, b.S) < 1e-4
    # a rigid load of another case is another model: nothing is shared then (the existing tests of test_state_cache_fixes pin it)
    rigid = copy.deepcopy(st)
    rigid.load_cases.append(LoadCase("R", [RemoteLoad("end", [60.0, 10.0, 5.0], force=[0.0, 0.0, -100.0], coupling="rigid")]))
    assert SC.solved_digest(rigid, mesh, rigid.load_cases[4:]) != SC.study_digest(GS.unit_model(rigid, [0, 1, 2]), mesh)


# ============================================================================================================================ 2: the reference points of a deck are part of what a unit state can serve
def test_a_unit_state_serves_only_a_deck_with_the_same_reference_points(beam, tmp_path):
    st0, mesh, _ = beam
    none = GS.reference_signature([])
    one = GS.reference_signature([LoadCase("R", [RemoteLoad("end", [60.0, 10.0, 5.0], force=[0.0, 0.0, -1.0], coupling="distributing")])])
    moved = GS.reference_signature([LoadCase("R", [RemoteLoad("end", [60.0, 10.0, 6.0], force=[0.0, 0.0, -1.0], coupling="distributing")])])
    other_force = GS.reference_signature([LoadCase("R", [RemoteLoad("end", [60.0, 10.0, 5.0], force=[0.0, 5.0, -9.0], coupling="distributing")])])
    assert none == "" and one and one != moved and one == other_force
    SC.clear()
    n = len(mesh.nodes)
    z = np.zeros((n, 3))
    digest = "e" * 64
    GS.save_unit(tmp_path, digest, 0, "ccx", z, np.zeros((n, 6)), z, None, {}, {n + 1: np.ones(6)}, deck="study", ref_sig=one)
    GS.save_unit(tmp_path, digest, 1, "ccx", z, np.zeros((n, 6)), z, None, {}, None, deck="study", ref_sig=none)
    assert GS._entry(digest, 0, "ccx", "study", one) is not None
    assert GS._entry(digest, 0, "ccx", "study", moved) is None                    # other reference point
    assert GS._entry(digest, 0, "ccx", None, none) is None                        # a study without remote loads does not take the state that carries a reference point
    assert GS._entry(digest, 1, "ccx", None, none) is not None
    assert GS._entry(digest, 1, "ccx", "study", one) is None
    assert GS._entry(digest, 0, "ccx") is not None                                # (the unit statics of the dynamics take any: they need U, S, RF only)


@needs_ccx
def test_a_distributing_remote_load_gets_its_reference_displacements_from_the_cache_or_a_solve(beam, tmp_path):
    st0, mesh, _ = beam
    a = copy.deepcopy(st0)
    a.solver.backend = "ccx"
    _gravity(a, 4)
    b = copy.deepcopy(a)
    b.load_cases.append(LoadCase("R", [RemoteLoad("end", [60.0, 10.0, 5.0], force=[0.0, 0.0, -50.0], coupling="distributing")]))
    c = copy.deepcopy(b)
    c.load_cases[-1] = LoadCase("R", [RemoteLoad("end", [60.0, 10.0, 7.0], force=[0.0, 0.0, -50.0], coupling="distributing")])
    ref_b, ref_c = _run(b, mesh, tmp_path, "ref_b", cache=False), _run(c, mesh, tmp_path, "ref_c", cache=False)
    assert ref_b.steps[0].ref_displacements
    # A (no remote load) then B: the unit states of A have no reference points – B solves its own (all three), its gravity cases carry the reference displacements
    SC.clear()
    _run(a, mesh, tmp_path, "run_a")
    got_b = _run(b, mesh, tmp_path, "got_b")
    assert got_b.info["gravity_superposition"]["solved"] == ["x", "y", "z"]
    assert got_b.steps[0].ref_displacements and set(got_b.steps[0].ref_displacements) == set(ref_b.steps[0].ref_displacements)
    for k, v in ref_b.steps[0].ref_displacements.items():
        assert _rel(got_b.steps[0].ref_displacements[k], v) < 1e-4
    # B then B again: stored, with the reference displacements
    again = _run(b, mesh, tmp_path, "again_b")
    assert again.info["gravity_superposition"]["stored"] == ["x", "y", "z"]
    assert _rel(again.steps[1].ref_displacements[next(iter(ref_b.steps[1].ref_displacements))], ref_b.steps[1].ref_displacements[next(iter(ref_b.steps[1].ref_displacements))]) < 1e-4
    # B then C (the reference point moved): not shared – C solves again and its reference displacements are its own
    got_c = _run(c, mesh, tmp_path, "got_c")
    assert got_c.info["gravity_superposition"]["solved"] == ["x", "y", "z"]
    for k, v in ref_c.steps[0].ref_displacements.items():
        assert _rel(got_c.steps[0].ref_displacements[k], v) < 1e-4


# ============================================================================================================================ 3: a basis without reactions recovers nothing
@needs_direct
def test_a_basis_without_reactions_builds_no_element_matrices_and_recovers_nothing(beam, tmp_path, monkeypatch):
    from fea_core import recovery
    from fea_core.dynamics.modal import run_basis
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.modal_engine = "direct"
    st.dynamics.enabled, st.dynamics.modes = True, 4
    calls: list[str] = []
    real_rb, real_es = recovery.recover_batch, recovery.element_stiffness

    def rb(*a, **k):
        calls.append("recover_batch")
        return real_rb(*a, **k)

    def es(*a, **k):
        calls.append("element_stiffness")
        return real_es(*a, **k)
    monkeypatch.setattr(recovery, "recover_batch", rb)
    monkeypatch.setattr(recovery, "element_stiffness", es)
    SC.clear()
    lite = run_basis(st, mesh, tmp_path / "lite", lambda s: None, modes=4, with_stress=False, with_reactions=False, job="pcb_modal")
    assert calls == [] and lite.K == 4 and not np.any(lite.rf)
    assert not list((tmp_path / "lite").glob("state_cache/modes_*/rec_*"))        # no recovery entry (it would lack the reactions)
    # the same with static cases: their static states need the recovered unit columns (one recovery of the columns, none of the modes)
    SC.clear()
    calls.clear()
    full = run_basis(st, mesh, tmp_path / "stat", lambda s: None, modes=4, with_stress=False, with_reactions=False, job="pcb_modal2", static_cases=[GS.unit_case(2, 1.0)])
    assert calls.count("recover_batch") == 1 and full.K == 4
    # a full basis still recovers its modes
    SC.clear()
    calls.clear()
    run_basis(st, mesh, tmp_path / "full", lambda s: None, modes=4)
    assert "recover_batch" in calls and "element_stiffness" in calls


# ============================================================================================================================ 4: the compute lock is released on every way out
@needs_ccx
def test_the_compute_lock_is_released_when_the_lookup_of_the_stored_basis_raises(beam, tmp_path, monkeypatch):
    from fea_core.direct import modal_cache as MCH
    from fea_core.dynamics import modal
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.modal_engine = "ccx"
    st.dynamics.enabled, st.dynamics.modes = True, 3

    def boom(key, log=None):
        raise OSError("disk read error")
    monkeypatch.setattr(modal, "_ccx_basis_lookup", boom)
    SC.clear()
    retained = []
    for _ in range(2):                                                            # (twice: the second would wait for ever for a lock that leaked)
        try:
            modal.run_basis(st, mesh, tmp_path / "lock", lambda s: None, modes=3)
        except OSError as e:
            retained.append(e)                                                    # the exception is kept: its traceback keeps the frames (and the context managers) alive
    assert len(retained) == 2
    assert MCH._COMPUTE == {}
    with MCH.compute_lock(SC.study_digest(st, mesh)):
        pass


# ============================================================================================================================ 5: an entry of the same name is no substitute for the one that could not be written
def _arrays(scale=1.0):
    return {"a": np.arange(12.0).reshape(4, 3) * scale, "b": np.ones(5) * scale}


def test_with_no_room_an_existing_entry_is_returned_only_when_it_holds_the_same_arrays(tmp_path, monkeypatch):
    SC.clear()
    f = SC.write_entry(tmp_path, "test", "f" * 40, "t", _arrays(), {"k": 1})
    assert f is not None
    monkeypatch.setattr(SC, "has_room", lambda folder, need: False)
    SC.clear()
    assert SC.write_entry(tmp_path, "test", "f" * 40, "t", _arrays(), {"k": 1}) == f             # the same content: the stored one serves
    assert SC.entries("test", "f" * 40) == [f]
    SC.clear()
    assert SC.write_entry(tmp_path, "test", "f" * 40, "t", _arrays(-1.0), {"k": 1}) is None       # another content: nothing
    assert SC.entries("test", "f" * 40) == []                                                 # ...and nothing registered
    assert SC.add_files(f, "rec", _arrays(), {"k": 1}) is None                                   # nothing there to take
    monkeypatch.setattr(SC, "has_room", lambda folder, need: True)
    sub = SC.add_files(f, "rec", _arrays(), {"k": 1})
    assert sub is not None
    monkeypatch.setattr(SC, "has_room", lambda folder, need: False)
    assert SC.add_files(f, "rec", _arrays(), {"k": 1}) == sub
    assert SC.add_files(f, "rec", _arrays(2.0), {"k": 1}) is None


@needs_direct
def test_with_no_room_the_modes_of_a_new_analysis_never_get_the_recovery_of_an_old_one(beam, tmp_path, monkeypatch):
    """Fable's n_noroom: the same model analysed again in the work folder of an earlier run, the eigenvector of mode 1 with the other sign, no room for a new entry."""
    from fea_core import direct, recovery
    from fea_core.dynamics.modal import run_basis
    st0, mesh, _ = beam
    st = copy.deepcopy(st0)
    st.solver.modal_engine = "direct"
    st.dynamics.enabled, st.dynamics.modes = True, 4
    SC.clear()
    run_basis(st, mesh, tmp_path, lambda s: None)
    SC.clear()                                                                    # a new process, the same work folder
    real = direct.run_eigen_prepared

    def flipped(*a, **k):
        lam, X, S, info = real(*a, **k)
        X = np.array(X)
        X[:, 0] *= -1.0
        return lam, X, S, info
    monkeypatch.setattr(direct, "run_eigen_prepared", flipped)
    monkeypatch.setattr(SC, "has_room", lambda folder, need: False)
    b = run_basis(st, mesh, tmp_path, lambda s: None)
    S, _rf = recovery.recover_batch(st, mesh, np.asarray(b.shapes[:1], float))
    s = np.asarray(b.stress[0], float)
    assert np.abs(s - S[0]).max() / np.abs(s).max() < 1e-5                         # the stresses are those of THESE shapes, not the old ones (sign flipped: 2.0)


# ============================================================================================================================ 6: hashing
def test_a_written_entry_is_not_read_again_by_its_first_lookup_and_a_header_check_proves_nothing(tmp_path, monkeypatch):
    SC.clear()
    f = SC.write_entry(tmp_path, "test", "1" * 40, "t", _arrays(), {"k": 1})
    assert f is not None and str(f) in SC._CHECKED                                  # stamped right after it was published
    monkeypatch.setattr(SC, "content_hash", lambda a: pytest.fail("the content of a freshly written entry was hashed again"))
    assert SC.entries("test", "1" * 40) == [f]
    monkeypatch.undo()
    SC.clear()
    assert SC.verified(f, full=False) and str(f) not in SC._CHECKED                 # a header check leaves no stamp ...
    assert SC.verified(f) and str(f) in SC._CHECKED                                 # ... only the full one
    np.save(f / "a.npy", np.arange(12.0).reshape(4, 3) + 1.0)
    SC.clear()
    assert SC.verified(f, full=False) and not SC.verified(f)                        # (the header is intact; the content is not)


def test_the_hash_of_a_very_large_array_samples_its_head_tail_and_a_stride_of_blocks(monkeypatch):
    monkeypatch.setattr(SC, "FULL_HASH_BYTES", 1 << 10)
    monkeypatch.setattr(SC, "SAMPLE_EDGE", 256)
    monkeypatch.setattr(SC, "SAMPLE_BLOCKS", 8)
    a = np.zeros(64 * 1024 // 8)                                                    # 64 KiB: sampled (above the 1 KiB limit), blocks of 1 MiB do not fit: one stride step
    h0 = SC.content_hash(a)
    for idx in (0, a.size - 1):
        b = a.copy()
        b[idx] = 1.0
        assert SC.content_hash(b) != h0                                             # head and tail are covered
    assert SC.content_hash(a.reshape(-1, 2)) != h0 and SC.content_hash(a.astype(np.float32)) != h0   # shape and type always are
    assert SC.content_hash(a.copy()) == h0
    monkeypatch.undo()
    monkeypatch.setattr(SC, "FULL_HASH_BYTES", 1 << 30)
    small = np.arange(1000.0)
    for idx in (0, 500, 999):                                                       # an array up to the limit is hashed completely
        b = small.copy()
        b[idx] += 1.0
        assert SC.content_hash(b) != SC.content_hash(small)


# ============================================================================================================================ 7: the entries of the analyses in progress are not pruned
def test_the_entries_of_the_digests_in_use_are_not_pruned(tmp_path, monkeypatch):
    SC.clear()
    monkeypatch.setattr(SC, "KEEP_OTHER", 0)
    monkeypatch.setattr(SC, "RECENT_KEYS", 2)
    digests = [f"{i:x}" * 40 for i in range(4)]
    made = []
    for i in range(2):
        f = SC.write_entry(tmp_path, "test", digests[i], "t", _arrays(float(i + 1)), {"k": 1})
        made.append(f)
        os.utime(f, (time.time() + i, time.time() + i))
    assert SC.entries("test", digests[0]) == [made[0]]                              # the oldest is looked up: it is in use again
    f2 = SC.write_entry(tmp_path, "test", digests[2], "t", _arrays(3.0), {"k": 1})
    assert made[0].exists() and f2 is not None and not made[1].exists()             # the unused one went, the one in use stayed
    SC.clear()
    f3 = SC.write_entry(tmp_path, "test", digests[3], "t", _arrays(4.0), {"k": 1})
    assert f3 is not None and not made[0].exists() and not f2.exists()              # (with nothing in use, only the key just written stays)


# ============================================================================================================================ 8: the notes of the missing STR estimate
def test_the_str_note_names_the_superposition_only_for_the_superposed_gravity_cases_of_a_ccx_run():
    from fea_core.mcp.diagnostics import no_error_estimate_note
    info = {"backend": "ccx", "gravity_superposition": {"cases": ["G1", "G2"], "family": "ccx"}}
    assert GS.str_missing_by_superposition(info, "G1") and not GS.str_missing_by_superposition(info, "Other")
    assert GS.str_missing_by_superposition(info) and not GS.str_missing_by_superposition({"backend": "ccx"}, "G1")
    assert not GS.str_missing_by_superposition({**info, "gravity_superposition": {**info["gravity_superposition"], "family": "own"}}, "G1")
    assert "combinations of unit states" in no_error_estimate_note(SimpleNamespace(info=info), "G1")
    other = no_error_estimate_note(SimpleNamespace(info=info), "Other")             # a load case that is not superposed has its reason: only a CalculiX solve gives the estimate
    assert "combinations of unit states" not in other and "backend='ccx'" in other
    assert "combinations of unit states" not in no_error_estimate_note(SimpleNamespace(info={"backend": "own", "gravity_superposition": {"cases": ["G1"], "family": "own"}}), "G1")


def test_the_convergence_note_and_the_mesh_fingerprint_texts():
    from fea_core.space.loads import mesh_fingerprint
    assert "tie table" in (mesh_fingerprint.__doc__ or "") and "edge" in (mesh_fingerprint.__doc__ or "")

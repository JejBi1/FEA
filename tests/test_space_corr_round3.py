"""Round-3 review of the COC / AOC (F14-S5, 25. 11. 2026): the mass-weighted residual of round 2 is gone, a TAM printed with fewer digits is still a singular PSD matrix, an unusable model
fingerprint holds a TAM, the test data changing under a running correlation void it. (The SEREP basis of every resolvable solved mode, which this file tested in the first version, was
removed by the owner decision of 25. 11. 2026: the SEREP COC is indicative only, see test_space_corr_round4.py.)"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fea_core.eigen import ModeSet
from fea_core.space import corr_orth as O
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from test_space_corr_coc_round2 import _one_tet
from verification.space_cases import block_study
from verification.space_corr_cases import _generalised_chain


def test_no_mass_weighted_residual_remains():
    phi, m = _generalised_chain([1.0] * 8)
    dofs = [(f"N{i}", "x") for i in range(8)]
    g = phi.T @ m @ phi
    o = O.orthogonality(phi[:, :4], phi[:, :4], dofs, O.OrthMass("serep", gram=g[:4, :4]))
    d = o.to_dict()
    assert "residual_weighted" not in d and "residual_weighted_max" not in d["info"] and not hasattr(o, "residual_weighted") and not hasattr(O.OrthMass(), "weights")
    assert not hasattr(C, "sensor_tributary_masses")



# ------------------------------------------------------------------ 4: a TAM printed with fewer digits
def _sereps_csv(fmt, n=12, m=4):
    phi, mass = _generalised_chain([1.0] * n)
    fe = phi[:, :m]
    pin = np.linalg.pinv(fe)
    ms = pin.T @ (fe.T @ mass @ fe) @ pin
    dofs = [(f"N{i}", "x") for i in range(n)]
    txt = "# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in dofs) + "\n" + "\n".join(f"{dofs[i][0]}:{dofs[i][1]}," + ",".join(fmt % float(v) for v in ms[i]) for i in range(n)) + "\n"
    return txt, phi, mass, dofs, fe


@pytest.mark.parametrize("fmt", ["%.7e", "%.6e", "%.5e"])
def test_a_serep_matrix_printed_with_few_digits_loads_as_a_singular_psd_matrix_with_the_true_rank(fmt):
    txt, phi, mass, dofs, fe = _sereps_csv(fmt)
    tam = O.parse_mass_csv(txt, "serep.csv")
    assert tam.rank == 4 and tam.summary()["rank"] == 4
    # a test mode with 0.75 of its norm in the null space is held, a good one is not - at every printing precision
    q, _ = np.linalg.qr(fe)
    r = np.random.default_rng(2).standard_normal(12)
    r -= q @ (q.T @ r)
    r /= np.linalg.norm(r)
    t = fe.copy()
    t[:, 0] = fe[:, 0] + 0.75 * np.linalg.norm(fe[:, 0]) * r
    o = O.orthogonality(t, fe, dofs, O.OrthMass("tam", tam=tam))
    assert o.residual[0] == pytest.approx(0.75 ** 2 / (1 + 0.75 ** 2), abs=2e-3) and 0 in o.mode_hold and 1 not in o.mode_hold and o.info["rank"] == 4
    good = O.orthogonality(fe, fe, dofs, O.OrthMass("tam", tam=tam))
    assert not good.mode_hold and np.allclose(np.diag(good.coc), 1.0, atol=2e-3)
    assert np.all(np.isfinite(good.coc))


def test_a_matrix_with_a_real_negative_eigenvalue_is_still_refused():
    dofs = [("A", "x"), ("B", "x")]
    txt = "# unit: kg\n,A:x,B:x\nA:x,1,0\nB:x,0,-0.01\n"
    with pytest.raises(Exception, match="positive semi-definite"):
        O.parse_mass_csv(txt)
    ok = O.parse_mass_csv("# unit: kg\n,A:x,B:x\nA:x,1,0\nB:x,0,-1e-6\n")                       # rounding level (1e-6 of the largest): the null space
    assert ok.rank == 1 and dofs


# ------------------------------------------------------------------ 5: the model fingerprint, the memoryview
def test_a_tam_whose_model_cannot_be_fingerprinted_holds_the_rows():
    st = block_study()
    tam = O.MassMatrix([("A", "x"), ("A", "y")], np.eye(2), "m.csv")
    CR.remember_tam(st, tam, None)
    opt = C.CorrelationOptions(orth_mass="tam")
    om = CR.build_orth_mass(st, None, None, opt, tam)
    assert om.kind == "tam" and "could not be fingerprinted" in om.model_hold
    assert O.orthogonality(np.eye(2), np.eye(2), tam.dofs, om).hold == om.model_hold
    mesh = _one_tet()
    good = CR.build_orth_mass(st, mesh, None, opt, tam)
    assert good.model_hold == ""                                                                   # a mesh at hand: bound at its first use, no hold


def test_the_model_fingerprint_is_unchanged_by_the_buffer_hashing():
    mesh, st = _one_tet(), block_study()
    h = hashlib.sha256(O.mass_model_fingerprint(st, mesh).encode("ascii"))
    for name in ("nodes", "tets"):
        a = np.ascontiguousarray(getattr(mesh, name))
        h.update(repr((name, a.shape, str(a.dtype))).encode("ascii"))
        h.update(a.tobytes())
    assert O.model_fingerprint(st, mesh) == h.hexdigest()
    fe = hashlib.sha256()
    ms = ModeSet("modal", "m", [10.0], np.zeros((1, 10, 3), np.float32))
    fe.update(__import__("json").dumps([round(float(f), 9) for f in ms.values]).encode("ascii"))
    fe.update(repr(tuple(np.shape(ms.shapes))).encode("ascii"))
    for name in ("nodes", "tets"):
        a = np.ascontiguousarray(getattr(mesh, name))
        fe.update(repr((name, a.shape, str(a.dtype))).encode("ascii"))
        fe.update(a.tobytes())
    assert C.fe_fingerprint(ms, mesh) == fe.hexdigest()


# ------------------------------------------------------------------ 2: the test data changing under a running correlation
def test_a_change_of_the_test_data_during_the_correlation_voids_the_result(monkeypatch):
    mesh = _one_tet()
    rng = np.random.default_rng(5)
    shapes = rng.standard_normal((2, 10, 3)).astype(np.float32)
    ms = ModeSet("modal", "m", [10.0, 20.0], shapes)
    sensors = [C.Sensor("P", point=[0.4, 0.5, 0.6]), C.Sensor("Q", point=[0.3, 0.8, 1.0])]
    td = C.fe_test_data(ms, mesh, sensors, damping=0.01)
    assert C.correlate(td, ms, mesh, sensors, limits=None).modal is not None                      # undisturbed
    real = C.fe_shapes_at_sensors

    def disturbing(*a, **k):
        td.set_rotation("rows" if td.rotation == "columns" else "columns")                       # the user flips the convention while the worker runs
        return real(*a, **k)
    monkeypatch.setattr(C, "fe_shapes_at_sensors", disturbing)
    with pytest.raises(ValueError, match="changed while the correlation was running"):
        C.correlate(td, ms, mesh, sensors, limits=None)

"""Round-2 review fixes of the COC / AOC (F14-S5, 25. 11. 2026): a singular (rank-deficient) TAM and a TAM with one FE mode, the Euclidean out-of-span residual that does not bound the
mass-weighted error (the residual in the norm of the tributary masses of the sensors), the lumped nodal mass, a TAM bound to the model (densities, added masses, mesh), the axes statement of
a TAM (who / via), the placeholders of the Czech catalogue."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.space import corr_orth as O
from fea_core.space import corr_run as CR
from fea_core.space import correlation as C
from test_space_corr_coc_review import _status
import test_space_correlation as T
from test_space_correlation import _orthogonality_mass, fe  # noqa: F401  (fixtures)
from test_space_corr_orthogonality import _limits
from verification.space_cases import block_study
from verification.space_corr_cases import _generalised_chain


def _rows(test, fe, dofs, mass):
    orth = O.orthogonality(test, fe, dofs, mass)
    nt, nf = test.shape[1], fe.shape[1]
    modal = C.modal_correlation(np.arange(1, nt + 1) * 10.0, test, np.arange(1, nf + 1) * 10.0, fe, dofs, orth=orth)
    return orth, modal, C.evaluate(modal, [], _limits())


def _outside(f, ratio, seed=3):
    q, _ = np.linalg.qr(f)
    r = np.random.default_rng(seed).standard_normal(f.shape[0])
    r -= q @ (q.T @ r)
    r /= np.linalg.norm(r)
    t = f.copy()
    t[:, 0] = f[:, 0] + ratio * np.linalg.norm(f[:, 0]) * r
    return t


def _csv(dofs, m):
    return "# unit: kg\n," + ",".join(f"{a}:{b}" for a, b in dofs) + "\n" + "\n".join(f"{dofs[i][0]}:{dofs[i][1]}," + ",".join(repr(float(v)) for v in m[i]) for i in range(len(dofs))) + "\n"


# ------------------------------------------------------------------ 1: a singular TAM, a TAM with one FE mode
@pytest.mark.parametrize("ratio, held", [(0.0, False), (0.1, False), (0.75, True), (3.0, True)])
def test_a_rank_deficient_tam_holds_the_rows_of_a_test_mode_in_its_null_space(ratio, held):
    phi, m = _generalised_chain([1.0] * 12)
    dofs = [(f"N{i}", "x") for i in range(12)]
    fe = phi[:, :4]
    pin = np.linalg.pinv(fe)
    tam = O.parse_mass_csv(_csv(dofs, pin.T @ (fe.T @ m @ fe) @ pin), "serep_tam.csv")          # what another program exports as a SEREP matrix: rank 4 of 12
    assert tam.rank == 4 and tam.summary()["rank"] == 4
    t = _outside(fe, ratio)
    orth, _modal, rows = _rows(t, fe, dofs, O.OrthMass("tam", tam=tam))
    want = ratio ** 2 / (1 + ratio ** 2)
    assert orth.residual[0] == pytest.approx(want, abs=1e-9) and orth.info["rank"] == 4 and orth.info["residual_kind"] == "null-space"
    assert (0 in orth.mode_hold) is held and 1 not in orth.mode_hold                           # only the mode that has a part in the null space
    assert any("singular (rank 4 of 12" in w for w in orth.warnings)
    assert set(_status(rows, "COC diagonal")) == ({C.UNCONFIRMED} if held else {C.PASS})
    full = abs(t[:, 0] @ m @ fe[:, 0]) / np.sqrt((t[:, 0] @ m @ t[:, 0]) * (fe[:, 0] @ m @ fe[:, 0]))
    if held:
        assert orth.coc[0, 0] > 0.999 > full                                                     # the TAM is blind to the part outside; the full mass matrix is not
        assert "null space" in orth.mode_hold[0] and "rank 4 of 12" in orth.mode_hold[0]


def test_a_tam_blind_to_a_compared_fe_mode_holds_every_row():
    dofs = [(f"N{i}", "x") for i in range(8)]
    phi, _m = _generalised_chain([1.0] * 8)
    fe = phi[:, :3]
    tam = O.MassMatrix(dofs, np.outer(phi[:, 5], phi[:, 5]) + np.outer(phi[:, 6], phi[:, 6]), "blind.csv")        # sees the modes 6 and 7 only
    orth = O.orthogonality(fe + 0.0, fe, dofs, O.OrthMass("tam", tam=tam))
    assert "null space of the imported mass matrix" in orth.hold and "FE mode(s) 1, 2, 3" in orth.hold


def test_a_full_rank_tam_is_not_a_singular_one():
    dofs = [(f"N{i}", "x") for i in range(8)]
    phi, _m = _generalised_chain([1.0] * 8)
    orth = O.orthogonality(phi[:, :3], phi[:, :3], dofs, O.OrthMass("tam", tam=O.parse_mass_csv(_csv(dofs, 2.0 * np.eye(8)))))
    assert orth.hold == "" and orth.residual is None and orth.info["rank"] == 8 and not any("singular" in w for w in orth.warnings)


def test_one_fe_mode_through_a_tam_cannot_be_checked_and_is_held():
    dofs = [(f"N{i}", "x") for i in range(8)]
    phi, _m = _generalised_chain([1.0] * 8)
    t = np.abs(np.random.default_rng(5).standard_normal((8, 1))) + 0.2
    rank1 = O.MassMatrix(dofs, np.ones((8, 8)), "rank1.csv")
    orth, _modal, rows = _rows(t, phi[:, :1], dofs, O.OrthMass("tam", tam=rank1))
    assert orth.coc[0, 0] == pytest.approx(1.0) and "only one FE mode" in orth.hold                  # COC 1 for a random shape through a rank-1 matrix
    assert set(_status(rows, "COC diagonal")) == {C.UNCONFIRMED}
    good = O.orthogonality(phi[:, :1] + 0.0, phi[:, :1], dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, 2.0 * np.eye(8))))
    assert "only one FE mode" in good.hold and any("only one FE mode" in w for w in good.warnings)  # even a fitting full matrix: the fit is not verified
    assert O.orthogonality(phi[:, :2], phi[:, :2], dofs, O.OrthMass("tam", tam=O.MassMatrix(dofs, 2.0 * np.eye(8)))).hold == ""


# ------------------------------------------------------------------ (the mass-weighted residual of round 2 was dropped in round 3: tests/test_space_corr_round3.py)
def _one_tet():
    x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([x, [(x[p] + x[q]) / 2 for p, q in mids]])
    mesh = Mesh(nodes, np.arange(10)[None, :], np.zeros(1, int), ["Block"])
    mesh.surfaces["Face"] = np.array([[0, 1, 2, 4, 5, 6]])
    return mesh


# ------------------------------------------------------------------ 5: the TAM is bound to the model, not only to name and geometry
def test_a_tam_imported_for_one_state_of_the_model_holds_the_rows_for_another():
    from fea_core.dynamics.model import PointMass
    mesh = _one_tet()
    st = block_study()
    dofs = [("A", "x"), ("A", "y")]
    tam = O.MassMatrix(dofs, np.eye(2), "m.csv")
    CR.remember_tam(st, tam, mesh)
    assert tam.model_key and tam.model_key == C.model_fingerprint(st, mesh)
    opt = C.CorrelationOptions(orth_mass="tam")
    assert CR.build_orth_mass(st, mesh, None, opt, tam).model_hold == ""
    st.masses = [PointMass("Face", 5e-6)]                                                 # a point mass added after the matrix was imported
    om = CR.build_orth_mass(st, mesh, None, opt, tam)
    assert om.kind == "tam" and "another state of the model" in om.model_hold
    st.masses = []
    next(iter(st.materials.values())).rho *= 2.0                                          # a density edited
    assert CR.build_orth_mass(st, mesh, None, opt, tam).model_hold
    other = Mesh(mesh.nodes * 1.01, mesh.tets, mesh.tet_part, mesh.part_names)
    st2 = block_study()
    assert CR.build_orth_mass(st2, other, None, opt, tam).model_hold                      # another mesh
    orth = O.orthogonality(np.eye(2), np.eye(2), dofs, O.OrthMass("tam", tam=tam, model_hold="the model differs"))
    assert orth.hold == "the model differs" and "the model differs" in orth.warnings


def test_a_tam_imported_without_a_mesh_is_bound_at_its_first_use():
    mesh = _one_tet()
    st = block_study()
    tam = O.MassMatrix([("A", "x"), ("A", "y")], np.eye(2), "m.csv")
    CR.remember_tam(st, tam)
    assert tam.model_key == ""
    assert CR.build_orth_mass(st, mesh, None, C.CorrelationOptions(orth_mass="tam"), tam).model_hold == "" and tam.model_key == C.model_fingerprint(st, mesh)


# ------------------------------------------------------------------ 7: who stated the axes of the TAM
def test_the_statement_about_the_axes_of_the_tam_records_who_made_it():
    o = C.CorrelationOptions(orth_mass="tam", tam_global_axes=True, tam_global_axes_via="mcp")
    back = C.CorrelationOptions.from_dict(o.to_dict())
    assert back.tam_global_axes and back.tam_global_axes_via == "mcp" and "MCP interface (an agent): check it yourself" in back.describe()
    gui = C.CorrelationOptions(orth_mass="tam", tam_global_axes=True, tam_global_axes_via="gui")
    assert "through the GUI" in gui.describe()
    off = C.CorrelationOptions(orth_mass="tam", tam_global_axes=False, tam_global_axes_via="gui")
    assert off.to_dict()["tam_global_axes_via"] == "" and "global axes" not in off.describe()


def test_the_mcp_statement_is_recorded_as_mcp(fe, tmp_path):
    pytest.importorskip("mcp")
    from test_space_corr_core_fixes import _mcp_session
    from test_space_correlation import _mcp
    sess, f = _mcp_session(tmp_path, fe)
    sensors = [s.to_dict() for s in T.SENSORS]
    _mcp(sess, [("fea_space_test_import", {"path": str(f), "sensors": sensors}, False), ("fea_space_correlate", {"tam_global_axes": True}, False)])
    stored = CR.state_of(sess.study)["options"]
    assert stored["tam_global_axes"] is True and stored["tam_global_axes_via"] == "mcp"
    _mcp(sess, [("fea_space_correlate", {}, False)])
    assert CR.state_of(sess.study)["options"]["tam_global_axes_via"] == ""

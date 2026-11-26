"""Modal basis of the dynamics from ccx *FREQUENCY (stage F8.1): shapes φ_j normalised by the mass, modal stresses σ_j, modal
reactions, participation factors Γ_j and effective masses.

Verified by a trial (scratch/f8_dyn/exp_norm.py, 15. 10. 2026, cantilever 10×10×100 C3D10):
  - the displacements in the .frd of the step *FREQUENCY are normalised by the mass: φᵀMφ = 1.000000 (M = consistent mass by own
    integration, dynamics.mass.mass_times); Γ_j from the .dat = φ_jᵀ·M·r to all digits;
  - *EL FILE S in the step *FREQUENCY writes the modal stresses (the same normalisation), *NODE FILE RF = K·φ_j in all nodes
    (Σ RF on the constraints = −ω_j²·Σ M·φ_j in the free nodes to 1e-5);
  - the nodes of the point mass (*MASS) are not written to the .frd → their motion from the kinematics of the face (mass.node_motion).
The sign of the shape is the sign of ccx (without turning as eigen._shapes) → Γ_j from the .dat and φ_j are consistent.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class DynamicsError(RuntimeError):
    pass


@dataclass
class ModalBasis:
    freq: np.ndarray                      # (K,) Hz
    shapes: np.ndarray                    # (K, N, 3) normalised by the mass (φᵀMφ = 1)
    stress: np.ndarray                    # (K, N, 6) float32 modal stresses
    gamma: np.ndarray                     # (K, 6) participation factors of ccx (x y z rx ry rz)
    eff_mass: np.ndarray                  # (K, 6) [t]
    total_mass: np.ndarray                # (6,) total mass of ccx [t] (without the mass of the constrained nodes)
    rf_sum: np.ndarray                    # (K, 3) Σ of the modal reactions on the constraints (K·φ_j)
    rf_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # constraint nodes
    rf: np.ndarray | None = None          # (K, n_c, 3) modal reactions in the nodes of the constraints
    mass_motion: list = field(default_factory=list)      # [(K, k, 3) motion of the points of the point masses]
    model_mass: float = 0.0               # mass of the model [t] (bodies + added mass)
    checks: dict = field(default_factory=dict)
    prestress: str | None = None
    prestress_nlgeom: bool = False
    rf_from_inertia: bool = False         # preload: the resultant of the reactions from the equilibrium of the inertia forces (ccx FORC without K_sigma·φ)
    engine: str = "ccx"                   # which engine computed the modes: ccx | direct (own assembly + rslab + ARPACK)
    engine_info: dict = field(default_factory=dict)      # the reason of the choice, the protocol of the direct analysis
    statics: list | None = None           # the direct path: the static solutions of the auxiliary cases requested by `static_cases` (from the same factor)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(self.freq, float)

    @property
    def K(self) -> int:
        return len(self.freq)


def _read_blocks(frd_path: Path, n: int, last_step_only: bool):
    from ..ccx.frd import read_frd
    frd = read_frd(frd_path)
    steps = frd.steps()
    step = steps[-1] if last_step_only and steps else None
    rank = None
    if step is not None:
        # preload: the NUMSTP blocks of the step *FREQUENCY continue after the increments of the static step (modes 2, 3, 4, 5 instead of
        # 1…4) → without renumbering the shapes shifted by one (φ₁ = 0, normalisation φᵀMφ = [0, 1, 1, 1], Γ wrong);
        # found with the verification case Y9) → the order of the mode in the step
        rank = {m: i + 1 for i, m in enumerate(sorted({b.mode for b in frd.blocks if b.step == step}))}
    out = {}
    for b in frd.blocks:
        if step is not None and b.step != step:
            continue
        if b.name not in ("DISP", "STRESS", "FORC"):
            continue
        msk = b.node_ids <= n
        nc = 3 if b.name != "STRESS" else 6
        a = np.zeros((n, nc))
        a[b.node_ids[msk] - 1] = b.values[msk, :nc]
        out[(b.name, rank[b.mode] if rank is not None else b.mode)] = a
    return out


def constrained_nodes(study, mesh, own: bool = False) -> np.ndarray:
    """Nodes of the mesh held by the constraints (reactions RF): clamping, displacements, cylindrical and distributing remote constraint. The elastic support
    does not belong here – its nodes are free, the reaction = −k·φ (spring_reaction_sum). own = the direct path: the face of a rigid remote support holds its
    reaction too (ccx measures it at the reference node of its rigid body, not at the nodes of the face).
    """
    from ..solve import constraint_masks
    msk = np.zeros(len(mesh.nodes), bool)
    for v in constraint_masks(study, mesh, own=own).values():
        msk |= v.any(axis=1)
    return np.nonzero(msk)[0]


def spring_reaction_sum(study, mesh, shapes: np.ndarray) -> np.ndarray:
    """(K, 3) sum of the reactions of the elastic support −k·φ_j over the modes."""
    from ..springs import supports, reactions
    out = np.zeros((len(shapes), 3))
    if not supports(study):
        return out
    for j, phi in enumerate(shapes):
        out[j] = sum(r.sum(axis=0) for r in reactions(study, mesh, phi).values())
    return out


def _arrays_ccx(study, mesh, workdir: Path, log, on_line, cancel, k: int, fm, prestress_case, storage, probe_nodes, job: str) -> dict:
    """The modal basis from a ccx *FREQUENCY job: {"freq", "shapes", "stress", "rfs" (K, N, 3), "participation", "eff_mass", "total"}."""
    from ..ccx.inp import write_inp
    from ..ccx.runner import run_ccx, CcxCrash, CcxError
    from ..config import CCX_EXE
    from ..eigen import is_free_body, read_dat, rigid_count, _sub_study
    from ..study import LoadCase
    line = f"{k}, 0." + (f", {float(fm):.12g}" if fm else "")
    if prestress_case:
        lc = next((c for c in study.load_cases if c.name == prestress_case), None)
        if lc is None:
            raise DynamicsError(f"Prestress: load case '{prestress_case}' does not exist")
        st = _sub_study(study, [lc])
        log(f"Natural frequencies after prestress by case '{lc.name}' (CalculiX, {k} modes)")
    else:
        st = _sub_study(study, [LoadCase("Natural frequencies", [])])
        log(f"Modal basis of the dynamics (CalculiX, {k} modes)")
    temps = None
    if prestress_case and study.has_thermal_loads() and study.thermal_loads(lc):
        temps = None                                   # thermal preload not yet (the thermal results are not available)
        log("Prestress: the thermal load of the case is not used in the prestressed modal analysis")
        st.load_cases = [LoadCase(lc.name, [x for x in lc.loads if x.kind != "thermal"])]
    try:
        write_inp(st, mesh, workdir / f"{job}.inp", temperatures=temps, procedure="modal", eigen_line=line,
                  modal_output=True, storage=storage, prestress=bool(prestress_case), probe_nodes=probe_nodes)
    except ValueError as e:
        raise DynamicsError(str(e)) from None
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    r = run_ccx(workdir, job, exe=exe, on_line=on_line, cancel=cancel)
    if not r.finished or r.errors:
        raise (CcxCrash if r.crashed else CcxError)("CalculiX (dynamics natural frequencies) ended with an error:\n  "
                                                  + "\n  ".join(r.errors or [f"code {r.returncode}"]) + f"\nlog: {r.log_path}")
    d = read_dat(workdir / f"{job}.dat", "modal")
    if not d["freq"]:
        raise DynamicsError("Dynamics: ccx returned no mode (try a different frequency range)")
    freq_ccx = np.array(d["freq"], float)
    freq_ccx[:rigid_count(freq_ccx, is_free_body(study))] = 0.0       # the rounding noise of the rigid-body modes of a free body: 0 Hz, as on the direct path (the same rule)
    n = len(mesh.nodes)
    blk = _read_blocks(workdir / f"{job}.frd", n, last_step_only=bool(prestress_case))
    kk = len(d["freq"])
    shapes = np.zeros((kk, n, 3))
    stress = np.zeros((kk, n, 6), np.float32)
    rfs = np.zeros((kk, n, 3))
    present = d.get("modes") or []
    ranks = sorted({r for (_, r) in blk})
    # ccx numbers the blocks of the modes consecutively from 1 and leaves the modes with a negative eigenvalue out of the eigenvalue table: the r-th block belongs to the
    # r-th mode of the table (see eigen.read_dat) – without it the shapes of a free body were shifted by the number of the omitted modes
    consecutive = ranks == list(range(1, len(ranks) + 1))
    mode_of = {r: present[r - 1] for r in ranks} if present and len(present) == len(ranks) and consecutive else {r: r for r in ranks}
    for (name, r), a in blk.items():
        j = mode_of.get(r, r) - 1
        if not 0 <= j < kk:
            continue
        if name == "DISP":
            shapes[j] = a
        elif name == "STRESS":
            stress[j] = a
        elif name == "FORC":
            rfs[j] = a
    return {"freq": freq_ccx, "shapes": shapes, "stress": stress, "rfs": rfs, "participation": d["participation"], "eff_mass": d["eff_mass"],
            "total": d["total"]}


def modal_window_share(study, fm) -> int | None:
    """The mode count of the natural-frequency analysis of the study (`study.modal`) when it asks for the same window as this basis (from 0 Hz, the same upper limit): the direct
    analysis then computes the larger of the two counts ONCE and each takes its first modes (`modal_run.run_modes`, `state_cache`)."""
    ms = study.modal
    if not ms.enabled or float(ms.f_min or 0.0) != 0.0:
        return None
    if (float(ms.f_max) if ms.f_max else None) != (float(fm) if fm else None):
        return None
    return int(ms.modes)


def _resident(a: np.ndarray, mapped: bool) -> np.ndarray:
    """`a` itself; where it is (a slice of) a read-only map of a stored file and the caller does not want a map, a resident copy – nothing the results keep then holds a file of the
    work folder open (Windows refuses to delete a folder with a mapped file)."""
    if mapped:
        return a
    b: Any = a
    while b is not None:
        if isinstance(b, np.memmap) or type(b).__name__ == "mmap":
            return np.array(a)
        b = getattr(b, "base", None)
    return a


def _arrays_direct(study, mesh, workdir: Path, log, on_line, cancel, k: int, fm, job: str, static_cases, with_stress: bool = True, mapped: bool = False,
                   with_reactions: bool = True) -> dict:
    """The modal basis of the direct path: the same arrays as `_arrays_ccx` (the stresses and the nodal forces of the modes by `recovery.recover_batch`, the static
    solutions of the auxiliary cases from the same factor). The analysis comes from the state cache when an earlier one of the same study and mesh is stored there, and so do
    the modal reactions / stresses and those of the unit-acceleration columns. with_stress = False (the interface forces need only the reactions) leaves the stresses out.
    with_reactions = False: the modes are not recovered at all (no stresses, zero reactions, whatever with_stress says; see `run_basis`) and NO recovery entry is stored (it
    would lack the reactions); the static solutions of `static_cases` are still recovered.
    mapped = the stored stresses stay memory-mapped (read-only) instead of being copied into the memory – for a basis that only a running analysis reads."""
    from ..direct import modal_cache as MCH
    from ..direct import modal_run
    from ..recovery import recover_batch
    dm = modal_run.run_modes(study, mesh, workdir, log, cancel, on_line, modes=k, f_max=float(fm) if fm else None, static_cases=static_cases, name=job,
                             share_modes=modal_window_share(study, fm))
    n = len(mesh.nodes)
    freq = dm.freq
    ok = freq > 0
    cn = constrained_nodes(study, mesh, own=True)
    with_stress = bool(with_stress and with_reactions)    # (no recovery of the modes = no modal stresses either)
    ncol = len(dm.columns_all) if dm.cols is not None else 0
    need_cols = bool(ncol and (with_reactions or dm.statics))      # (the unit-acceleration columns serve the static states of the cases and the modal reactions – a basis without either recovers nothing)
    rec = MCH.load_recovery(dm, len(freq), with_stress) if (with_reactions or need_cols) else None
    if rec is not None:
        log(f"Modal stresses and reactions of {int(ok.sum())} modes: from the stored basis ({Path(dm.cache).name})")
        stress = (rec["stress"] if mapped else np.array(rec["stress"])) if with_stress else np.zeros((len(freq), 0, 6), np.float32)
        rf = np.array(rec["rf"]) if with_reactions else np.zeros((len(freq), len(cn), 3))
        col_S, col_RF = rec["col_S"], rec["col_RF"]
    else:
        C = field = None
        if with_reactions or need_cols:                     # the element matrices are needed only to recover something
            from ..recovery import element_stiffness
            C = element_stiffness(study, mesh) if not _has_ortho(study, mesh) else None
            if C is None:
                from ..ortho import ElementField
                field = ElementField(study, mesh)
        stress = np.zeros((len(freq), n if with_stress else 0, 6), np.float32)
        rf = np.zeros((len(freq), len(cn), 3))
        idx = np.nonzero(ok)[0] if with_reactions else np.zeros(0, int)
        if len(idx):
            log(f"Modal stresses and reactions of {len(idx)} modes")
        step = 8
        for lo in range(0, len(idx), step):
            sel = idx[lo:lo + step]
            S, RF = recover_batch(study, mesh, dm.shapes[sel], C, field, stress=with_stress)
            assert RF is not None
            if with_stress and S is not None:
                stress[sel] = S.astype(np.float32)
            rf[sel] = RF[:, cn]
        col_S: Any = None
        col_RF: Any = np.zeros((0, len(cn), 3))
        if need_cols and dm.cols is not None:             # the unit accelerations and the other columns: recovered once, the static states of the cases are combinations of them
            colU = np.asarray(dm.cols[:, :3 * n], np.float64).reshape(ncol, n, 3)
            S, RF = recover_batch(study, mesh, colU, C, field, stress=with_stress)
            assert RF is not None
            col_S = S.astype(np.float32) if (with_stress and S is not None) else None
            col_RF = RF[:, cn]
        if with_reactions:
            MCH.store_recovery(dm, len(freq), stress if with_stress else None, rf, cn, col_S, col_RF)
    statics = None
    if dm.statics:
        statics = []
        for i, cf in enumerate(dm.coeffs_all):
            S_i = np.zeros((n, 6) if with_stress else (0, 6))
            RF_i = np.zeros((len(cn), 3))
            for j in np.nonzero(cf)[0]:
                if with_stress and col_S is not None:
                    S_i += float(cf[j]) * np.asarray(col_S[j], float)
                RF_i += float(cf[j]) * np.asarray(col_RF[j], float)
            statics.append({"U": dm.statics[i], "S": S_i.astype(np.float32), "RF": RF_i})
    info = modal_run.protocol(dm)
    return {"freq": freq, "shapes": _resident(dm.shapes, mapped), "stress": stress, "rfs": rf, "rfs_at_constraints": True, "cn": cn, "participation": [list(g) for g in dm.gamma],
            "eff_mass": [list(g) for g in dm.eff_mass], "total": [float(v) for v in dm.total_free], "statics": statics, "engine": "direct", "info": info}


def cross_check_basis(study, mesh, basis: ModalBasis, workdir: Path, log, on_line=None, cancel=None) -> dict:
    """The same modes in CalculiX on the same mesh against the modal basis of the direct path: frequencies, effective masses (participation factors) and the MAC table –
    `eigen.compare_modes` (repeated and clustered modes as a subspace; the modes the two analyses do not both have at a nonzero frequency are left out)."""
    from ..eigen import compare_modes
    k = int(study.dynamics.modes)
    arr = _arrays_ccx(study, mesh, Path(workdir), log, on_line, cancel, k, study.dynamics.f_max, None, False, None, "dyn_modal_check")
    ok = arr["freq"] > 0                                  # the rigid-body modes of a free body are 0 Hz on this side too (`_arrays_ccx`), the basis dropped them: the same rule
    part = np.asarray(arr["participation"], float)[ok]
    out = compare_modes(basis.freq, basis.shapes, basis.gamma, arr["freq"][ok], arr["shapes"][ok], part, total=arr["total"] or None)
    out["engine"] = "ccx"
    out["modes_direct"], out["modes_ccx"] = int(basis.K), int(ok.sum())
    return out


def _has_ortho(study, mesh) -> bool:
    from ..ortho import has_ortho
    return has_ortho(study, mesh)


def _ccx_basis_key(study, mesh, k: int, fm) -> tuple[str, str]:
    """(the study digest, the tag) under which a CalculiX modal basis of this study, mesh, mode count and upper limit is stored."""
    from .. import state_cache as SC
    return SC.study_digest(study, mesh), f"{int(k)}_{float(fm or 0.0)!r}"


def _ccx_basis_lookup(key: tuple[str, str], log=None) -> dict | None:
    """The arrays of the stored CalculiX modal basis of `key` (intact entries only; the arrays are copied: nothing stays mapped), or None."""
    from .. import state_cache as SC
    digest, tag = key
    for folder in SC.entries("ccxbasis", digest):
        m = SC.meta_of(folder)
        if m is None or m.get("tag") != tag:
            continue
        arr: dict[str, Any] = {nm: np.array(SC.array(folder, nm)) for nm in ("freq", "shapes", "stress", "rfs")}
        arr.update(participation=m["participation"], eff_mass=m["eff_mass"], total=m["total"], engine="ccx", from_cache=True,
                   info={**(m.get("info") or {}), "stored_analysis": {"source": "state cache"}})
        if log is not None:
            log(f"Modal basis (CalculiX): the stored analysis of this model is used ({folder.name})")
        return arr
    return None


def _ccx_basis_store(workdir: Path, key: tuple, arr: dict) -> None:
    from .. import state_cache as SC
    digest, tag = key
    SC.write_entry(Path(workdir), "ccxbasis", digest, tag.replace(".", "p"),
                   {nm: arr[nm] for nm in ("freq", "shapes", "stress", "rfs")},
                   {"tag": tag, "participation": [list(map(float, g)) for g in arr["participation"]], "eff_mass": [list(map(float, g)) for g in arr["eff_mass"]],
                    "total": [float(v) for v in (arr["total"] or [])], "info": arr.get("info") or {}})


def run_basis(study, mesh, workdir: Path, log, on_line=None, cancel=None, modes: int | None = None,
              f_max: float | None = None, prestress_case: str | None = None, storage: bool = False,
              probe_nodes=None, job: str = "dyn_modal", static_cases: list | None = None, with_stress: bool = True, mapped: bool = False,
              with_reactions: bool = True) -> ModalBasis:
    """The modal basis of the dynamics on the mesh of the study: the direct path (own assembly + rslab + ARPACK, `fea_core.direct.modal_run`) where it can do the study,
    otherwise CalculiX *FREQUENCY (a job on the mesh of the study: constraints, added mass) → ModalBasis. prestress_case = state, after which it is computed with the
    stress stiffness (*STEP, PERTURBATION) – CalculiX only. static_cases = auxiliary static cases whose solutions the direct path takes from the SAME factor
    (`ModalBasis.statics`; None / the ccx path: the caller computes them with `static_solutions`).
    with_stress = False: the modal STRESSES are left out (they cost about half of the modal recovery; the interface forces need only the reactions) – `ModalBasis.stress` is
    then (K, 0, 6). with_reactions = False = a basis for callers that read the frequencies, shapes, participation factors and effective masses only (the effective-mass
    model check, the board displacement of the PCB check): the modes are not recovered at all – no stresses (`stress` is (K, 0, 6) whatever with_stress says), `rf` is zero and
    `checks["mode_equilibrium_checked"]` is False; ONLY a caller that does not read them may ask for it, and such a basis never writes a recovery entry to the state cache.
    mapped = True: the arrays of a stored basis stay memory-mapped (for the 3-axis analysis and the interface forces, which keep the basis only while they run; the results of
    the dynamics copy them).
    The basis of the same study and mesh is computed once (`fea_core.state_cache`): a later call – the 3-axis analysis after the run, the interface forces – loads it."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ds = study.dynamics
    k = int(modes or ds.modes)
    fm = f_max if f_max is not None else ds.f_max
    arr = None
    reason = ""
    takeover = ""
    plain = not (prestress_case or storage or probe_nodes is not None)      # the reference runs of the cross-check and the prestressed basis stay CalculiX (and are not stored)
    ccx_key = None
    from .. import state_cache
    import contextlib
    engine, direct_possible = "ccx", False
    if plain:
        from ..direct import DirectError, modal_run
        from ..solve_large import MeshQualityError
        engine, reason, direct_possible = modal_run.decide_full(study, mesh, dynamics=True, modes=k)
    with contextlib.ExitStack() as stack:                 # (the lock is released on EVERY way out, the lookup of the stored basis included)
        if plain and engine == "ccx" and state_cache.enabled():
            # a stored CalculiX basis is looked at only where CalculiX is the engine of the decision (a direct analysis chosen now is never replaced by one that was stored), and one
            # analysis of a model runs at a time: the one that waited finds it stored
            from ..ccx.runner import CcxError
            from ..direct import modal_cache as MCH
            ccx_key = _ccx_basis_key(study, mesh, k, fm)
            stack.enter_context(MCH.compute_lock(ccx_key[0], cancel, log, CcxError))
            arr = _ccx_basis_lookup(ccx_key, log)
        if plain and arr is None:
            if engine == "ccx" and direct_possible:           # chosen by the size only: the direct analysis takes over when CalculiX DIES (not for an error it reports about the model)
                from ..ccx.runner import CcxCrash
                try:
                    arr = _arrays_ccx(study, mesh, workdir, log, on_line, cancel, k, fm, prestress_case, storage, probe_nodes, job)
                    arr["engine"], arr["info"] = "ccx", {"reason": reason}
                except CcxCrash as e:                      # CalculiX stopped without a result: the direct analysis can do this study
                    if (cancel is not None and cancel.is_set()) or "canceled by user" in str(e):
                        raise
                    from ..pathcheck import scrub_paths
                    ccx_failed = scrub_paths((str(e).splitlines() or [type(e).__name__])[0])[:200]
                    log(f"CalculiX failed ({ccx_failed}) – computing the modal basis with the direct analysis")
                    engine, reason = "direct", f"CalculiX failed ({ccx_failed})"
                    takeover = f"Dynamics: CalculiX stopped without a result ({ccx_failed}) – the modal basis was computed with the direct analysis instead"
            if engine == "direct":
                try:
                    arr = _arrays_direct(study, mesh, workdir, log, on_line, cancel, k, fm, job, static_cases, with_stress, mapped, with_reactions)
                    arr["info"] = {**arr.get("info", {}), "reason": reason, **({"takeover": takeover} if takeover else {})}
                except MeshQualityError as e:
                    raise DynamicsError(str(e)) from None
                except (DirectError, modal_run.ModalUnsupported) as e:
                    if (cancel is not None and cancel.is_set()) or "canceled by user" in str(e):
                        raise
                    if modal_run.engine_setting(study) == "direct" or reason.startswith("CalculiX failed"):
                        raise DynamicsError(f"{reason}; the direct analysis failed too: {e}" if reason.startswith("CalculiX failed") else str(e)) from None
                    from ..pathcheck import scrub_paths
                    why = scrub_paths((str(e).splitlines() or [type(e).__name__])[0])[:200]
                    log(f"The direct eigenvalue analysis failed ({why}) – computing the modal basis in CalculiX")
                    reason = f"the direct eigenvalue analysis failed ({why})"
                    takeover = f"Dynamics: the direct eigenvalue analysis failed ({why}) – the modal basis was computed with CalculiX instead"
        if arr is None:
            arr = _arrays_ccx(study, mesh, workdir, log, on_line, cancel, k, fm, prestress_case, storage, probe_nodes, job)
            arr["engine"], arr["info"] = "ccx", {"reason": reason, **({"takeover": takeover} if takeover else {})}
        if plain and ccx_key is not None and arr["engine"] == "ccx" and not arr.get("from_cache"):
            _ccx_basis_store(workdir, ccx_key, arr)
    freq, shapes, stress, rfs = arr["freq"], arr["shapes"], arr["stress"], arr["rfs"]
    if (not with_stress or not with_reactions) and stress.shape[1]:        # (the ccx path reads them from the .frd anyway: they are not kept)
        stress = np.zeros((len(freq), 0, 6), np.float32)
    if not with_reactions and np.any(rfs):
        rfs = np.zeros(np.shape(rfs))
    ok = freq > 0
    if not ok.all():
        log(f"Dynamics: {int((~ok).sum())} modes with zero frequency (rigid-body motion of a free body) – skipped")
    cn = arr["cn"] if "cn" in arr else constrained_nodes(study, mesh)
    is_nlgeom = bool(getattr(study.dynamics.prestress, "nlgeom", False)) if prestress_case else False
    sub = (lambda a: a) if ok.all() else (lambda a: a[ok])            # the arrays of the modes are large: no copy when every mode is kept
    rf_cn = sub(rfs) if arr.get("rfs_at_constraints") else sub(rfs)[:, cn]
    basis = ModalBasis(sub(freq), sub(shapes), sub(stress), np.asarray(arr["participation"], float)[ok],
                       np.asarray(arr["eff_mass"], float)[ok], np.asarray(arr["total"] or [0.0] * 6, float),
                       rf_cn.sum(axis=1), cn, rf_cn.astype(np.float32), prestress=prestress_case,
                       prestress_nlgeom=is_nlgeom, engine=arr["engine"], engine_info=arr.get("info", {}))
    basis.statics = arr.get("statics")
    from .mass import mass_points, node_motion, total_mass, mass_items
    if prestress_case:
        # the ccx .frd FORC in *STEP, PERTURBATION contains only K_0·phi without the geometric stiffness K_sigma·phi.
        # The RESULTANT of the reactions K_total·phi is given by the global equilibrium (exactly):
        # sum_{cn} (K_total·phi)_i = inert_i = -omega^2 sum_{free} (M·phi)_i → the reactions are rescaled to it.
        # The distribution among the nodes / supports remains according to K_0·phi (APPROXIMATE) and the check of the equilibrium of the modal reactions
        # loses its sense by it (it holds by construction) – in checks it is marked as not performed.
        Mphi = modal_mass_products(study, mesh, basis)
        free = np.ones(len(mesh.nodes), bool)
        free[cn] = False
        inert = -(basis.omega ** 2)[:, None] * Mphi[:, free].sum(axis=1)
        for mot, mu in point_mass_terms(study, mesh, basis):
            inert += -(basis.omega ** 2)[:, None] * np.einsum("kpi,p->ki", mot, mu)
        rfs_sum = basis.rf.sum(axis=1)
        rf_scaled = np.array(basis.rf, dtype=np.float32, copy=True)
        for k_idx in range(basis.K):
            for i in range(3):
                s_ccx = rfs_sum[k_idx, i]
                s_exact = inert[k_idx, i]
                if abs(s_ccx) > 1e-12:
                    rf_scaled[k_idx, :, i] = rf_scaled[k_idx, :, i] * (s_exact / s_ccx)
                elif abs(s_exact) > 1e-12 and len(cn) > 0:
                    rf_scaled[k_idx, :, i] = s_exact / len(cn)
        basis.rf = rf_scaled
        basis.rf_sum = basis.rf.sum(axis=1)
        basis.rf_from_inertia = True
    basis.rf_sum = basis.rf_sum + spring_reaction_sum(study, mesh, basis.shapes)   # elastic support = reaction −k·φ
    basis.mass_motion = [np.stack([node_motion(mesh, info, basis.shapes[j]) for j in range(basis.K)])
                         if basis.K else np.zeros((0, len(info["nodes"]), 3)) for info in mass_points(study, mesh)]
    added = 0.0
    for it in mass_items(study):
        if it.kind == "point_mass":
            added += float(it.mass)
        else:
            from .mass import surface_mass_nodes
            added += float(surface_mass_nodes(mesh, it)[1].sum())
    basis.model_mass = total_mass(study, mesh) + added
    basis.checks = check_basis(study, mesh, basis)
    if not with_reactions:
        basis.checks.update(mode_equilibrium=None, mode_equilibrium_checked=False)         # (the reactions were not computed: nothing to balance, no number to report)
    c = basis.checks
    if c.get("gamma_error", 0.0) > 1e-3 and c.get("mass_norm_error", 1.0) <= 5e-3 and _has_rigid_body(study):
        # Γ from ccx is wrong for a *RIGID BODY with a free ROT NODE: ccx adds M·φ also in DOF 1–3 of the ROT node (rotation) as
        # displacements – a point mass off the face / with rotary inertia gave Σ Γ² = 1.03 up to 14× the mass of the model
        # (verified 15. 10. 2026, scratch/f8_examples/pm_gamma.py; own integration Σ Γ² → 0.993). The normalisation φᵀMφ
        # by own integration agrees → the mass is complete and Γ = φᵀMr by own integration is correct → it is used
        # (the normalisation of the high modes of a coarse mesh differs by up to ~4e-3 by the integration rule, not by a missing mass).
        gam = own_gamma(study, mesh, basis)
        basis.gamma = np.array(basis.gamma, float, copy=True)
        basis.eff_mass = np.array(basis.eff_mass, float, copy=True)
        basis.gamma[:, :3] = gam
        basis.eff_mass[:, :3] = gam ** 2                 # φᵀMφ = 1
        basis.checks = check_basis(study, mesh, basis)
        if not with_reactions:
            basis.checks.update(mode_equilibrium=None, mode_equilibrium_checked=False)
        basis.checks.update(gamma_source="own", gamma_error_ccx=c["gamma_error"])
    return basis


def _has_rigid_body(study) -> bool:
    """The model has *RIGID BODY (point mass with a rigid constraint, rigid remote load / constraint)."""
    if any(getattr(m, "kind", "") == "point_mass" and m.coupling == "rigid" for m in study.masses):
        return True
    items = list(study.constraints) + [ld for lc in study.load_cases for ld in lc.loads]
    return any(getattr(x, "coupling", None) == "rigid" for x in items)


def own_gamma(study, mesh, basis: ModalBasis) -> np.ndarray:
    """Γ_j = φ_jᵀ·M·r (K, 3) by own integration (bodies, surface and point mass)."""
    gam = modal_mass_products(study, mesh, basis).sum(axis=1)
    for mot, mu in point_mass_terms(study, mesh, basis):
        gam += np.einsum("kpi,p->ki", mot, mu)
    return gam


def modal_mass_products(study, mesh, basis: ModalBasis) -> np.ndarray:
    """M·φ_j (K, N, 3) by own integration (consistent mass of the bodies + surface mass in the nodes)."""
    from .mass import mass_times, mass_items, surface_mass_nodes
    Mphi = mass_times(study, mesh, basis.shapes)
    for it in mass_items(study):
        if it.kind == "surface_mass":
            idx, mm = surface_mass_nodes(mesh, it)
            Mphi[:, idx] += mm[None, :, None] * basis.shapes[:, idx]
    return Mphi


def point_mass_terms(study, mesh, basis: ModalBasis) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(motion of the points (K, k, 3), masses of the points (k,))] of the point masses."""
    from .mass import mass_points
    return [(mot, np.array([mu for _, mu in info["nodes"]])) for mot, info in
            zip(basis.mass_motion, mass_points(study, mesh))]


def check_basis(study, mesh, basis: ModalBasis) -> dict:
    """Checks of the basis: normalisation φᵀMφ = 1 and Γ_j = φᵀMr by own integration against ccx (catches a wrong reading / the mass),
    the equilibrium of the mode Σ RF(constraints) + ω²·Σ M·φ(free) = 0, coverage of the effective mass by the directions.
    """
    if basis.K == 0:
        return {}
    Mphi = modal_mass_products(study, mesh, basis)
    gen = np.einsum("kni,kni->k", basis.shapes, Mphi)
    gam = Mphi.sum(axis=1)                              # (K, 3) φᵀ M r
    free = np.ones(len(mesh.nodes), bool)
    free[basis.rf_nodes] = False
    inert = -(basis.omega ** 2)[:, None] * Mphi[:, free].sum(axis=1)
    for mot, mu in point_mass_terms(study, mesh, basis):
        gen += np.einsum("kpi,kpi,p->k", mot, mot, mu)
        gam += np.einsum("kpi,p->ki", mot, mu)
        inert += -(basis.omega ** 2)[:, None] * np.einsum("kpi,p->ki", mot, mu)
    # only modes with a non-negligible effective mass (Γ² > 1e-4·m): for the others Γ is the noise of the 6 digits of the .frd
    m_tot = max(float(basis.model_mass), 1e-30)
    sig = (basis.gamma[:, :3] ** 2).max(axis=1) > 1e-4 * m_tot
    scale = np.maximum(np.abs(basis.gamma[:, :3]).max(), 1e-30)
    gam_err = float(np.abs(gam - basis.gamma[:, :3])[sig].max() / scale) if sig.any() else 0.0
    # scale = size of the individual reactions (the sum nearly cancels for torsional modes)
    rnorm = np.linalg.norm(basis.rf, axis=2).sum(axis=1) if basis.rf is not None and basis.rf.size else 0.0
    rscale = np.maximum(np.maximum(np.abs(inert).max(axis=1), rnorm), 1e-30)
    # Σ K·φ over all nodes = 0 (rigid displacements) → Σ RF(constraints) = −Σ K·φ(free) = −ω²·Σ M·φ(free) = inert
    has_supp = len(basis.rf_nodes) or np.abs(basis.rf_sum).max() > 0
    eq_err = float((np.abs(basis.rf_sum - inert).max(axis=1) / rscale)[sig].max()) if has_supp and sig.any() else 0.0
    tot = basis.total_mass[:3] if basis.total_mass is not None and len(basis.total_mass) else np.zeros(3)
    m_model = basis.model_mass
    cover = (basis.eff_mass[:, :3].sum(axis=0) / m_model).tolist() if m_model > 0 else [0.0] * 3
    if getattr(basis, "rf_from_inertia", False):      # the resultant of the reactions is taken from the equilibrium → the check would always hold
        eq_err = 0.0
    return {"mass_norm_error": float(np.abs(gen - 1.0).max()), "gamma_error": gam_err, "mode_equilibrium": eq_err,
            "mode_equilibrium_checked": not getattr(basis, "rf_from_inertia", False),
            "mass_fraction_xyz": [float(c) for c in cover], "model_mass_t": m_model,
            "ccx_total_mass_t": [float(v) for v in tot]}


def basis_warnings(basis: ModalBasis, dirs: set[int], f_top: float | None) -> list[str]:
    """Warnings on the modal basis: Σ m_eff < 90 % in the excited direction (EN 1998-1, 4.3.3.3.1), modes up to 1.5× the highest
    excitation frequency, checks of the normalisation.
    """
    out = []
    c = basis.checks
    if c.get("mass_norm_error", 0) > 1e-3 or c.get("gamma_error", 0) > 1e-3:
        out.append(f"Dynamics: the modal basis does not match the own mass integration (normalization {c['mass_norm_error']:.2g}, Γ {c['gamma_error']:.2g}) – verify the results")
    if (c.get("mode_equilibrium") or 0) > 1e-3:                       # (None = not computed: a basis without the modal reactions)
        out.append(f"Dynamics: modal reaction equilibrium {c['mode_equilibrium']:.2g} > 1e-3")
    if getattr(basis, "rf_from_inertia", False):
        out.append("Prestressed dynamics: the resultant support reaction is exact (from equilibrium of inertia forces), its split between several supports is approximate (CalculiX does not output the stress-stiffness part)")
    if getattr(basis, "prestress_nlgeom", False):
        out.append("Dynamics: prestress with NLGEOM – CalculiX in *FREQUENCY computes the stress stiffness K_sigma but evaluates the mode shapes on the reference (undeformed) geometry; the change of geometry by the preload is not included in the frequencies.")
    names = "xyz"
    for d in sorted(dirs):
        frac = c.get("mass_fraction_xyz", [1, 1, 1])[d]
        if frac < 0.9:
            out.append(f"Dynamics: effective modal mass in direction {names[d]} only {100 * frac:.1f} % (< 90 %) – add modes (the missing mass is added in the static correction / for shock)")
    if f_top and basis.K and basis.freq.max() < 1.5 * f_top:
        out.append(f"Dynamics: the highest basis mode {basis.freq.max():.4g} Hz < 1.5 × the highest excitation frequency {f_top:.4g} Hz – add modes")
    return out


# ------------------------------------------------------------------ damping

def zetas(spec, freq: np.ndarray) -> np.ndarray:
    """Relative damping of the modes ζ_j according to DampingSpec (structural → ζ = η/2 as a substitute in time)."""
    f = np.asarray(freq, float)
    if spec.kind == "rayleigh" and spec.f1 and spec.f2:
        w1, w2 = 2 * np.pi * spec.f1, 2 * np.pi * spec.f2
        a = 2 * spec.zeta * w1 * w2 / (w1 + w2)
        b = 2 * spec.zeta / (w1 + w2)
        w = 2 * np.pi * f
        return a / (2 * np.maximum(w, 1e-30)) + b * w / 2
    if spec.kind == "structural":
        return np.full_like(f, 0.5 * (spec.eta if spec.eta else 2 * spec.zeta))
    if spec.table:
        t = np.asarray(spec.table, float)
        return np.interp(f, t[:, 0], t[:, 1])
    return np.full_like(f, float(spec.zeta))


def static_solutions(study, mesh, cases: list, workdir: Path, log, on_line=None, cancel=None) -> list[dict]:
    """Static solution of the auxiliary states (the residual static correction: the state of the excitation by a force, the unit acceleration
    of the base) on the same mesh: [{"U", "S", "RF_sum"}]. CalculiX, one *STEP per state: the states are independent linear statics, so with several of them the steps run as
    concurrent jobs (`ccx.parallel`, as the load cases of `run_study`: the results are equal to ~1e-9); a share of the machine or `FEA_CCX_PARALLEL=0` keeps the one job. Measured against the one job (3 states, 2.5 k nodes): U ≤ 2.4e-10, S ≤ 5e-12, RF equal – relative to the maximum; the test asserts 1e-8.
    """
    from ..ccx.inp import write_inp
    from ..ccx.runner import run_ccx, CcxError
    from ..ccx.frd import read_frd
    from ..config import CCX_EXE
    workdir = Path(workdir)
    st = copy.copy(study)
    st.load_cases = list(cases)
    st.joints = copy.copy(study.joints)
    st.joints.preload_case = False
    job = "dyn_static"
    write_inp(st, mesh, workdir / f"{job}.inp")
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    from ..ccx import parallel
    workers, threads = parallel.plan(len(cases), 3 * len(mesh.nodes)) if parallel.enabled() else (1, 0)
    if workers > 1:
        log(f"Static solutions of {len(cases)} states (CalculiX, {workers} jobs × {threads} threads)")
        prun = parallel.run_steps(workdir, job, 3 * len(mesh.nodes), exe=exe, on_line=on_line, cancel=cancel, workers=workers, threads=threads)
        frd = parallel.read_frd_steps(workdir, prun.jobs)
        assert frd is not None                            # (the jobs are never empty: `run_steps` ran every state)
    else:
        r = run_ccx(workdir, job, exe=exe, on_line=on_line, cancel=cancel)
        if not r.finished or r.errors:
            raise CcxError("CalculiX (dynamics static correction) ended with an error:\n  "
                           + "\n  ".join(r.errors or [f"code {r.returncode}"]))
        frd = read_frd(workdir / f"{job}.frd")
    n = len(mesh.nodes)
    cn = constrained_nodes(study, mesh)
    out = []
    for s in frd.steps():
        res = {}
        for name, nc in (("DISP", 3), ("STRESS", 6), ("FORC", 3)):
            b = frd.get(name, s)
            a = np.zeros((n, nc))
            msk = b.node_ids <= n
            a[b.node_ids[msk] - 1] = b.values[msk, :nc]
            res[name] = a
        out.append({"U": res["DISP"], "S": res["STRESS"].astype(np.float32), "RF": res["FORC"][cn]})
    return out

"""Natural frequencies (ccx *FREQUENCY) and linear buckling (ccx *BUCKLE) on the same mesh as the statics.

- Natural frequencies: the constraints of the study (without them a free body – 6 zero rigid-body modes), mass from the
  densities of the materials (U1 bolts as steel). Output: frequencies [Hz], shapes normalised to max |u| = 1, participation
  factors and effective masses per direction (x, y, z, rx, ry, rz) from the ccx .dat, total mass.
- Buckling: the load of one load state (also thermal); factor λ = multiple of the load at loss of stability
  (linear – only stress stiffness, no imperfections and plasticity; the real capacity is usually lower). Negative λ = loss of
  stability under the reversed load.
Verified by trial (scratch/modal/exp1.py, 14. 10. 2026): cantilever V1 f₁ = 209.45 Hz (Euler–Bernoulli 208.9),
buckling by a compressive load at the end λ·1000 N = 21.65 kN (Euler β = 2: 21.59 kN). Shapes in the .frd: a block per mode, mode number
in NUMSTP of the 100C header (`ResultBlock.mode`); for *BUCKLE the first block is the static state (value 0).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ModalSettings:
    enabled: bool = False
    modes: int = 10                      # number of natural frequencies
    f_min: float = 0.0                   # lower bound of the search [Hz]
    f_max: float | None = None           # upper bound [Hz] (None = no bound)
    verify_ccx: bool = False             # the direct path: in addition the same modes in CalculiX (frequencies, effective masses, MAC in ModeSet.check)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if not d["verify_ccx"]:                  # written only when used (a file without it is what an earlier version wrote)
            del d["verify_ccx"]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "ModalSettings":
        from .study import _known
        return _known(cls, d)


@dataclass
class BucklingSettings:
    enabled: bool = False
    case: str | None = None              # load state (None = first)
    modes: int = 4

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BucklingSettings":
        from .study import _known
        return _known(cls, d)


DOF_LABELS = ("x", "y", "z", "rx", "ry", "rz")
RIGID_JUMP = 1e3                     # the rigid-body modes of a free body from CalculiX: the leading modes this many times below the first mode above them (see `rigid_count`)


def is_free_body(study) -> bool:
    """The model has no support at all (a free body: 6 rigid-body modes at 0 Hz): the same rule on the direct path (`modal_run.run_modes`) and in the ccx paths."""
    return not study.constraints and not (study.frame.active and study.frame.supports)


def rigid_count(freq, free: bool) -> int:
    """The number of rigid-body modes at the start of the ASCENDING list of frequencies [Hz] of a FREE body from CalculiX (0 for a supported one: it has none). Its eigenvalue table
    leaves the modes with a negative eigenvalue out (they come as 0.0, see `read_dat`) and lists those of the rounding noise that came out positive (1e-3 Hz against hundreds): the
    rigid modes are the leading ones that lie `RIGID_JUMP` times below the first mode above them (noise floor 1e-6 of the highest listed frequency). The direct path calls the same
    modes rigid by the strain energy of the shape (`direct.eigen`, they are returned at 0 Hz) – both engines say "0 Hz" and neither puts them into a window above zero."""
    f = np.asarray(freq, float)
    if not free or f.size == 0:
        return 0
    floor = 1e-6 * float(f.max())
    prev = 0.0
    for j, v in enumerate(f):
        if v > RIGID_JUMP * max(prev, floor):
            return j
        prev = max(prev, float(v))
    return 0


def takeover_warnings(modes) -> list[str]:
    """The warnings of the mode sets that one engine took over from the other (a crash of CalculiX, a failed direct analysis): they say which engine computed the modes."""
    out = []
    for m in modes or []:
        note = (getattr(m, "engine_info", None) or {}).get("takeover")
        if note and note not in out:
            out.append(note)
    return out


def _renumber(by: dict, present: list[int] | None) -> dict:
    """{rank r: value} → {mode number present[r − 1]: value}: ccx numbers the participation factors, the effective masses and the mode shapes consecutively from 1 while its
    eigenvalue table leaves the modes with a negative eigenvalue out. Applied only when the ranks are exactly 1…len(present) – anything else is left as it is (never an IndexError)."""
    if present and len(by) == len(present) and sorted(by) != sorted(present) and sorted(by) == list(range(1, len(present) + 1)):
        return {present[r - 1]: by[r] for r in sorted(by)}
    return by


@dataclass
class ModeSet:
    """Eigenvalue result: kind "modal" (values = frequencies [Hz]) or "buckling" (values = factors λ)."""
    kind: str
    name: str
    values: list[float]
    shapes: np.ndarray                    # (k, N, 3) float32, max |u| = 1
    case: str | None = None               # buckling: load state
    participation: list[list[float]] = field(default_factory=list)   # (k, 6)
    eff_mass: list[list[float]] = field(default_factory=list)        # (k, 6) [t], rotation [t·mm²]
    total_mass: list[float] = field(default_factory=list)            # (6,) total effective mass [t]
    frame_shapes: np.ndarray | None = None                           # (k, n_FE_members, 3) shapes of the members (max = 1)
    engine: str = "ccx"                                              # which engine computed the modes: ccx (CalculiX) | direct (own assembly + rslab + ARPACK)
    engine_info: dict = field(default_factory=dict)                 # the protocol of the direct path (fea_core.direct.modal_run.protocol), the reason for the choice
    check: dict = field(default_factory=dict)                       # the cross-check against the other engine (`cross_check`): frequencies, effective masses, MAC

    def summary(self) -> dict:
        d = {"kind": self.kind, "name": self.name, "case": self.case}
        if self.kind == "modal":
            d["engine"] = self.engine
            d["frequencies_Hz"] = [float(v) for v in self.values]
            if getattr(self, "nodal_diameters", None) is not None:      # cyclic symmetry
                d["nodal_diameters"] = list(self.nodal_diameters)
                d["multiplicity"] = list(getattr(self, "multiplicity", []))
            if self.eff_mass and self.total_mass:
                tot = np.asarray(self.total_mass, float)
                frac = np.asarray(self.eff_mass, float)[:, :3] / np.where(tot[:3] > 0, tot[:3], 1.0)
                d["effective_mass_fraction_xyz"] = np.round(frac, 4).tolist()
                d["cumulative_mass_fraction_xyz"] = np.round(frac.sum(axis=0), 4).tolist()
                d["total_mass_kg"] = float(tot[0] * 1000.0)
        elif self.kind == "nlbuckle":                # nonlinear buckling (fea_core.imperfect): λ of the stored increments
            d["frame_load_factors"] = [float(v) for v in self.values]
        else:
            d["buckling_factors"] = [float(v) for v in self.values]
        return d


class EigenError(RuntimeError):
    pass


def check(study, part_names) -> tuple[list[str], list[str]]:
    """Check of the settings of natural frequencies and buckling."""
    errors, warnings = [], []
    excl = study.excluded_parts()
    if study.modal.enabled:
        if int(study.modal.modes) < 1:
            errors.append("Natural frequencies: the number of modes must be ≥ 1")
        if study.modal.f_max is not None and study.modal.f_max <= study.modal.f_min:
            errors.append("Natural frequencies: the upper limit must be greater than the lower limit")
        for p in part_names:
            if p in excl:
                continue
            try:
                if not study.material_of(p).rho > 0:
                    errors.append(f"Natural frequencies: the material of part '{p}' has no density ρ")
            except KeyError:
                pass
        if not study.constraints and not (study.frame.active and study.frame.supports):
            warnings.append("Natural frequencies without supports: free body – the first 6 frequencies ≈ 0 Hz (rigid-body motions)")
    if study.buckling.enabled:
        if not study.load_cases:
            errors.append("Buckling: the study has no load case (buckling is computed from the load case loading)")
        elif study.buckling.case is not None and study.buckling.case not in [lc.name for lc in study.load_cases]:
            errors.append(f"Buckling: load case '{study.buckling.case}' does not exist")
        if int(study.buckling.modes) < 1:
            errors.append("Buckling: the number of modes must be ≥ 1")
        if not study.constraints and not (study.frame.active and study.frame.supports):
            errors.append("Buckling: the study has no supports")
        warnings.append("Buckling is linear (no imperfections or plasticity) – the actual resistance is often lower than λ·load; for design use a code procedure (e.g. EN 1993-1-1, 6.3) or a nonlinear analysis")
    return errors, warnings


def _sub_study(study, cases):
    """Shallow copy of the study with the given load states and without the bolt preload state."""
    st = copy.copy(study)
    st.load_cases = list(cases)
    st.joints = copy.copy(study.joints)
    st.joints.preload_case = False
    return st


_NUM = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"


def _table(text: str, title: str, ncol: int) -> tuple[dict[int, list[float]], list[float] | None]:
    """Rows of the .dat table after the heading: {mode number: ncol numbers} and the TOTAL row (| None). ccx leaves out modes with a
    negative eigenvalue (rigid motions of a free body numerically −1e-4) – therefore by mode number.
    """
    i = text.find(title)
    if i < 0:
        return {}, None
    rows, total = {}, None
    started = False
    for line in text[i + len(title):].splitlines():
        parts = line.split()
        if not parts:
            if started:
                break
            continue
        if parts[0].upper() == "TOTAL" and len(parts) >= ncol + 1:
            total = [float(v.replace("D", "E")) for v in parts[1:ncol + 1]]
            break
        if re.fullmatch(r"\d+", parts[0]) and len(parts) >= ncol + 1:
            try:
                rows[int(parts[0])] = [float(v.replace("D", "E")) for v in parts[1:ncol + 1]]
                started = True
            except ValueError:
                break
        elif started:
            break
    return rows, total


def read_dat(path: Path, kind: str) -> dict:
    """Eigenvalues from the .dat by modes 1…k (a missing mode = negative eigenvalue of a rigid motion → 0 Hz)."""
    text = Path(path).read_text(encoding="latin-1")
    if kind == "modal":
        eig, _ = _table(text, "E I G E N V A L U E   O U T P U T", 4)
        part, _ = _table(text, "P A R T I C I P A T I O N   F A C T O R S", 6)
        mass, _ = _table(text, "E F F E C T I V E   M O D A L   M A S S", 6)
        # ccx leaves the modes with a negative eigenvalue out of the eigenvalue table (rows 3…14 when the first two are left out) but NUMBERS the participation factors,
        # the effective masses and the mode shapes of the .frd consecutively from 1 (verified 18. 11. 2026: a free beam, 14 modes, the first two negative): the rows
        # are paired with the eigenvalue rows by ORDER (an earlier reading paired them by the number and shifted the shapes of a free body by the number of the omitted modes)
        present = sorted(eig)
        part, mass = _renumber(part, present), _renumber(mass, present)
        k = max([*eig, *part, *mass], default=0)
        i = text.find("T O T A L   E F F E C T I V E   M A S S")
        tot = []
        if i >= 0:
            for line in text[i:].splitlines()[1:]:
                nums = re.findall(_NUM, line)
                if len(nums) >= 6 and "COMPONENT" not in line:
                    tot = [float(v.replace("D", "E")) for v in nums[:6]]
                    break
        zero = [0.0] * 6
        return {"freq": [eig[j][2] if j in eig else 0.0 for j in range(1, k + 1)],
                "participation": [part.get(j, zero) for j in range(1, k + 1)],
                "eff_mass": [mass.get(j, zero) for j in range(1, k + 1)], "total": tot, "modes": present}
    fac, _ = _table(text, "B U C K L I N G   F A C T O R   O U T P U T", 1)
    return {"factors": [fac[j][0] for j in sorted(fac)]}


def _shapes(frd_path: Path, n_nodes: int, k: int, skip_first: bool, present: list[int] | None = None) -> np.ndarray:
    """Mode shapes 1…k (k, N, 3) normalised to max |u| = 1; the mode is taken from NUMSTP of the block. ccx leaves out the modes with a negative eigenvalue (their
    shape stays zero) and numbers the blocks of the others consecutively: `present` = the numbers of the modes of the eigenvalue table (`read_dat`) – the r-th block
    belongs to present[r − 1]. *BUCKLE: block 1 = static state, mode j = block j + 1.
    """
    from .ccx.frd import read_frd
    frd = read_frd(frd_path)
    shift = 1 if skip_first else 0
    by_mode = {b.mode - shift: b for b in frd.blocks if b.name == "DISP"}
    if not skip_first:
        by_mode = _renumber(by_mode, present)
    out = np.zeros((k, n_nodes, 3), np.float32)
    for j in range(k):
        b = by_mode.get(j + 1)
        if b is None:
            continue
        msk = b.node_ids <= n_nodes
        u = np.zeros((n_nodes, 3))
        u[b.node_ids[msk] - 1] = b.values[msk, :3]
        m = np.abs(u).max()
        if m > 0:
            u /= m
            if u.ravel()[np.abs(u).argmax()] < 0:      # sign: the largest component positive
                u = -u
        out[j] = u
    return out


def _frd_max(frd_path: Path, n_nodes: int, k: int, present: list[int] | None = None) -> list[float]:
    """Max |u| of the mode shapes 1…k in the .frd (not normalised – for the common scale with the members)."""
    from .ccx.frd import read_frd
    frd = read_frd(frd_path)
    by = _renumber({b.mode: b for b in frd.blocks if b.name == "DISP"}, present)
    return [float(np.abs(by[j + 1].values[:, :3]).max()) if (j + 1) in by else 0.0 for j in range(k)]


def modeset_from_direct(dm, n: int, name: str = "Natural frequencies") -> ModeSet:
    """`ModeSet` of the direct path (`fea_core.direct.modal_run.DirectModes`): the frequencies [Hz] (a rigid-body mode of a free body: 0), the shapes normalised to
    max |u| = 1 (the largest component positive), the participation factors Γ = φᵀMR of the mass-normalised modes and their effective masses Γ². Differences to
    CalculiX: ccx leaves out the modes whose eigenvalue is negative (rigid-body modes of a free body, noise) – here all of them are present, at 0 Hz, with a real shape."""
    from .direct import modal_run
    shapes = np.zeros(dm.shapes.shape, np.float32)               # one mode at a time: no second float64 copy of all the shapes (the main program keeps its memory for the solver)
    for j in range(len(shapes)):
        u = np.asarray(dm.shapes[j], dtype=np.float64)
        m = np.abs(u).max()
        if m > 0:
            sign = -1.0 if u.ravel()[np.abs(u).argmax()] < 0 else 1.0
            shapes[j] = u * (sign / m)
    ms = ModeSet("modal", name, [float(v) for v in dm.freq], shapes, participation=[list(map(float, g)) for g in dm.gamma],
                 eff_mass=[list(map(float, g)) for g in dm.eff_mass], total_mass=[float(v) for v in dm.total_free], engine="direct",
                 engine_info=modal_run.protocol(dm))
    return ms


def mac_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Modal assurance criterion MAC[i, j] = (aᵢ·bⱼ)² / ((aᵢ·aᵢ)(bⱼ·bⱼ)) of the shapes A (m, N, 3), B (k, N, 3) (nodal displacements; the sign does not matter)."""
    a = np.asarray(A, np.float64).reshape(len(A), -1)
    b = np.asarray(B, np.float64).reshape(len(B), -1)
    g = a @ b.T
    den = np.einsum("ij,ij->i", a, a)[:, None] * np.einsum("ij,ij->i", b, b)[None, :]
    return np.divide(g ** 2, den, out=np.zeros_like(g), where=den > 0)          # a mode that ccx left out has a zero shape: MAC 0


def compare_modes(f_a, A, g_a, f_b, B, g_b, rigid_hz: float | None = None, cluster: float = 1e-3, table: int = 12, components: int = 3, total=None) -> dict:
    """The cross-check of two analyses of the same model (a = the direct path, b = CalculiX): frequencies [Hz], shapes (m, N, 3), participation factors (m, 6).
    The rigid-body modes (f ≤ `rigid_hz`, default 1e-6 of the highest frequency: the 0.0 of the modes ccx leaves out and the rounding noise of the ones it keeps, 1e-3 Hz) are
    removed from BOTH lists by the same rule, and the elastic modes of the two analyses are then paired in ORDER (the lowest elastic mode with the lowest, …) – an earlier
    version paired by the index in the full lists, so the six rigid modes of a free body that only one side listed shifted every pair by six. A mode that one side lacks
    shows up as a frequency mismatch of the pairs behind it (`n_a`, `n_b` give the counts). Repeated and clustered frequencies (within `cluster` relative) are compared as a
    SUBSPACE: the share of the shape of a that lies in the span of the shapes of b of the same cluster (1 = the same modes), the effective masses by the cluster sum. The
    effective masses are compared in the first `components` participation factors (3 = the translations x, y, z: the rotations are about the origin of the coordinates and ccx
    gives wrong participation factors for a model with a rigid body that has a free rotation node – see `dynamics.modal`). `total` = the total effective mass of b (6,): the scale
    of "significant" (a component above 1 % of it) – without it the peak over the listed modes of b, which for a WINDOW above the first modes is much smaller than the mass.
    → {"n", "freq_rel", "eff_mass_rel", "mac" (diagonal), "subspace", "mac_table" (the first `table` modes), …}; "modes" = the 1-based positions of the compared modes in `f_a`."""
    f_a, f_b = np.asarray(f_a, float), np.asarray(f_b, float)
    top = float(max(f_a.max() if len(f_a) else 0.0, f_b.max() if len(f_b) else 0.0, 1e-30))
    rigid = rigid_hz if rigid_hz is not None else 1e-6 * top
    ia = [i for i in range(len(f_a)) if f_a[i] > rigid]
    ib = [j for j in range(len(f_b)) if f_b[j] > rigid]
    m = min(len(ia), len(ib))
    if m == 0:
        return {"n": 0, "message": "no elastic modes to compare", "n_a": len(ia), "n_b": len(ib)}
    ia, ib = ia[:m], ib[:m]
    fa, fb = f_a[ia], f_b[ib]
    Aa, Bb = np.asarray(A)[ia], np.asarray(B)[ib]
    ea = np.asarray(g_a, float)[ia][:, :components] ** 2 if components else np.zeros((m, 1))
    eb = np.asarray(g_b, float)[ib][:, :components] ** 2 if components else np.zeros((m, 1))
    mac = mac_matrix(Aa, Bb)
    out = {"n": m, "modes": [i + 1 for i in ia], "freq_direct": [float(v) for v in fa], "freq_ccx": [float(v) for v in fb], "n_a": len(ia), "n_b": len(ib)}
    out["freq_rel"] = [float(abs(fa[k] - fb[k]) / fb[k]) for k in range(m)]
    sub, em_rel, em_abs = [], [], []
    eb_all = np.asarray(g_b, float)[:, :components] ** 2 if components else eb
    tot = np.maximum(eb_all.max(axis=0), 1e-300)           # the peak over ALL the modes of b (the rigid-body modes of a free body carry the whole mass): the scale of "significant"
    if total is not None and components:
        tot = np.maximum(tot, np.asarray(total, float)[:components])
    for k in range(m):
        cl_b = [j for j in range(m) if abs(fb[j] - fb[k]) <= cluster * fb[k]]
        cl_a = [j for j in range(m) if abs(fa[j] - fa[k]) <= cluster * fa[k]]
        Qb, _ = np.linalg.qr(np.asarray(Bb[cl_b], np.float64).reshape(len(cl_b), -1).T)
        v = np.asarray(Aa[k], np.float64).reshape(-1)
        sub.append(float(np.linalg.norm(Qb.T @ v) ** 2 / (v @ v)))
        ca, cb = ea[cl_a].sum(axis=0), eb[cl_b].sum(axis=0)
        sig = cb > 1e-2 * tot
        em_rel.append(float(np.max(np.abs(ca - cb)[sig] / cb[sig])) if sig.any() else 0.0)
        em_abs.append(float(np.max(np.abs(ca - cb) / tot)))
    out.update({"mac": [float(mac[k, k]) for k in range(m)], "subspace": sub, "eff_mass_rel": em_rel, "eff_mass_vs_peak": em_abs})
    out["max_freq_rel"] = max(out["freq_rel"])
    out["min_mac"] = min(out["mac"])
    out["min_subspace"] = min(sub)
    out["max_eff_mass_rel"] = max(em_rel)
    out["max_eff_mass_vs_peak"] = max(em_abs)
    k = min(table, m)
    out["mac_table"] = [[round(float(v), 4) for v in row[:k]] for row in mac[:k]]
    return out


def _modal_direct(study, mesh, workdir: Path, log, cancel, on_line) -> ModeSet:
    """Study.modal on the direct path: the own assembly + rslab + ARPACK (`fea_core.direct.modal_run`)."""
    from .direct import modal_run
    ms = study.modal
    ds = study.dynamics
    # the dynamics of the same study asks for the modal basis in the same window (from 0 Hz, the same upper limit): ONE analysis with the larger mode count serves both
    # (the first modes of a larger analysis equal a smaller one to the tolerance of the eigenvalue solver; `state_cache`)
    share = int(ds.modes) if (ds.enabled and not ds.prestress.enabled and not float(ms.f_min or 0.0) and (float(ms.f_max) if ms.f_max else None) == (float(ds.f_max) if ds.f_max else None)) else None
    dm = modal_run.run_modes(study, mesh, workdir, log, cancel, on_line, modes=int(ms.modes), f_min=float(ms.f_min or 0.0),
                             f_max=float(ms.f_max) if ms.f_max else None, name="modal_direct", share_modes=share)
    return modeset_from_direct(dm, len(mesh.nodes))


def run_eigen(study, mesh, workdir: Path, log, on_line=None, cancel=None, thermal=None, kinds: tuple = ("modal", "buckling")) -> list[ModeSet]:
    """Natural frequencies and buckling according to Study.modal / Study.buckling: the natural frequencies on the direct path (`fea_core.direct.modal_run`) where it
    can do the study, otherwise in a ccx job; buckling always in ccx. kinds = which of the two to compute (the side jobs of `side_jobs` run only the ccx ones)."""
    from .ccx.inp import write_inp
    from .ccx.runner import run_ccx, CcxCrash, CcxError
    from .config import CCX_EXE
    from .study import LoadCase
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    n = len(mesh.nodes)

    def run(job: str, cancel_, on_line_) -> None:
        r = run_ccx(workdir, job, exe=exe, on_line=on_line_, cancel=cancel_)
        if not r.finished or r.errors:
            raise (CcxCrash if r.crashed else CcxError)(f"CalculiX ({job}) ended with an error:\n  " + "\n  ".join(r.errors or [f"code {r.returncode}"])
                                                     + f"\nlog: {r.log_path}")

    def cyclic_modal(cancel_, on_line_) -> ModeSet:
        from .cyclic.modal import run_modal as run_cyclic_modal   # frequencies by nodal diameters (ccx cyclic symmetry)
        return run_cyclic_modal(study, mesh, workdir, log, lambda job: run(job, cancel_, on_line_), _sub_study)

    def modal(cancel_, on_line_) -> ModeSet:
        from .direct import modal_run
        engine, reason, direct_possible = modal_run.decide_full(study, mesh)
        ccx_failed = ""
        takeover = ""
        if engine == "ccx" and direct_possible:           # chosen by the size only: the direct analysis takes over when CalculiX DIES (not for an error it reports about the model)
            try:
                return ccx_modal(cancel_, on_line_, reason)
            except CcxCrash as e:
                if cancel_ is not None and cancel_.is_set():
                    raise
                from .pathcheck import scrub_paths
                ccx_failed = scrub_paths((str(e).splitlines() or [type(e).__name__])[0])[:200]
                log(f"CalculiX failed ({ccx_failed}) – computing the natural frequencies with the direct analysis")
                engine, reason = "direct", f"CalculiX failed ({ccx_failed})"
                takeover = f"Natural frequencies: CalculiX stopped without a result ({ccx_failed}) – they were computed with the direct analysis instead"
        if engine == "direct":
            try:
                mset = _modal_direct(study, mesh, workdir, log, cancel_, on_line_)
            except Exception as e:  # noqa: BLE001 – a failed direct analysis continues in CalculiX under auto (never a silently missing mode); a user's choice reports it
                from .direct import DirectError
                from .solve_large import MeshQualityError
                if isinstance(e, MeshQualityError):
                    raise EigenError(str(e)) from None
                if not isinstance(e, (DirectError, modal_run.ModalUnsupported)) or (cancel_ is not None and cancel_.is_set()) or "canceled by user" in str(e):
                    raise
                if modal_run.engine_setting(study) == "direct":
                    raise EigenError(str(e)) from None
                from .pathcheck import scrub_paths
                why = scrub_paths((str(e).splitlines() or [type(e).__name__])[0])[:200]
                if ccx_failed:
                    raise EigenError(f"Natural frequencies: CalculiX failed ({ccx_failed}) and so did the direct analysis ({why})") from None
                log(f"The direct eigenvalue analysis failed ({why}) – computing in CalculiX")
                reason = f"the direct eigenvalue analysis failed ({why})"
                takeover = f"Natural frequencies: the direct eigenvalue analysis failed ({why}) – they were computed with CalculiX instead"
            else:
                mset.engine_info["reason"] = reason
                if takeover:
                    mset.engine_info["takeover"] = takeover
                if getattr(study.modal, "verify_ccx", False):
                    ref = ccx_modal(cancel_, on_line_, "cross-check of the direct analysis")
                    mset.check = compare_modes(mset.values, mset.shapes, mset.participation, ref.values, ref.shapes, ref.participation, total=ref.total_mass)
                    mset.check["engine"] = "ccx"
                return mset
        mset = ccx_modal(cancel_, on_line_, reason)
        if takeover:
            mset.engine_info["takeover"] = takeover
        return mset

    def ccx_modal(cancel_, on_line_, reason: str = "") -> ModeSet:
        """The natural frequencies in ccx with the SAME meaning of the window as the direct path: the lowest `modes` modes inside [f_min, f_max]. (ccx itself computes the
        lowest `modes` modes of the model and lists only those of them that lie in the window; the rows of the others stay as phantom 0 Hz modes with zero shapes – clamped beam,
        4 modes, f_min 4401 Hz gave [0, 0, 0, 5243] while the window holds 5243, 6441, 9579, 10827.) ccx is asked for the lowest k modes WITHOUT a window and the window is applied
        here; k grows (the modes below the window + `modes`; doubling while nothing is inside yet) until the window holds `modes` modes or its upper end / the limit of the modes
        ccx can extract (`cap`) is reached. The rigid-body modes of a free body (`rigid_count`) are 0 Hz and never inside a window above zero."""
        ms = study.modal
        want = int(ms.modes)
        f_min = float(ms.f_min or 0.0)
        f_max = float(ms.f_max) if ms.f_max else None
        free = is_free_body(study)
        cap = max(want, n)                                 # more modes than nodes is no modal analysis (and ARPACK needs room: ncv <= the number of equations)
        st = _sub_study(study, [LoadCase("Natural frequencies", [])])
        k = want + (6 if free and f_min > 0.0 else 0)       # a free body: its six rigid-body modes come first – the list must reach the first elastic mode, otherwise the rigid
        runs = 0                                             # ones cannot be told from tiny elastic ones (`rigid_count` looks for the jump above them)
        while True:
            runs += 1
            log(f"Natural frequencies (CalculiX, {k} modes)")
            try:
                imap = write_inp(st, mesh, workdir / "modal.inp", procedure="modal", eigen_line=f"{k}, 0")
            except ValueError as e:
                raise EigenError(str(e)) from None
            run("modal", cancel_, on_line_)
            d = read_dat(workdir / "modal.dat", "modal")
            if not d["freq"]:
                raise EigenError("Natural frequencies: ccx returned no mode (try a different frequency range)")
            freq = np.array(d["freq"], float)
            n_rigid = rigid_count(freq, free)
            freq[:n_rigid] = 0.0                           # rounding noise of the rigid-body modes: 0 Hz, as on the direct path
            above_zero = np.arange(len(freq)) >= (n_rigid if f_min > 0.0 else 0)
            inside = above_zero & (freq >= f_min)
            if f_max is not None:
                inside &= freq <= f_max
            sel = np.nonzero(inside)[0]
            exhausted = (f_max is not None and float(freq.max()) > f_max) or k >= cap
            if len(sel) >= want or exhausted:
                break
            below = int((freq < f_min).sum())
            k = min(cap, max(below + want, k + 1) if len(sel) else max(2 * k, k + want))
        if not len(sel):
            raise EigenError("Natural frequencies: no natural frequency in the requested range")
        sel = sel[:want]
        k_all = len(freq)
        shapes_all = _shapes(workdir / "modal.frd", n, k_all, skip_first=False, present=d.get("modes"))
        shapes = shapes_all[sel]
        info = {"reason": reason}
        if f_min > 0.0 or f_max is not None or runs > 1:
            info.update({"modes_computed": int(k_all), "ccx_runs": runs})
        if len(sel) < want:
            info["window_incomplete"] = f"only {len(sel)} of the {want} requested modes lie in the window"
        mset = ModeSet("modal", "Natural frequencies", [float(freq[j]) for j in sel], shapes,
                       participation=[d["participation"][j] for j in sel], eff_mass=[d["eff_mass"][j] for j in sel], total_mass=d["total"], engine="ccx",
                       engine_info=info)
        if imap.frame:                                # members: shapes of U1 nodes from the .dat (*NODE PRINT NSET=NFRAME)
            from .beams.fem import read_displacements
            nf = len(imap.frame["fe"].xyz)
            fsh = read_displacements(workdir / "modal.dat", imap.frame["base"], nf)
            fs = np.zeros((len(sel), nf, 3), np.float32)
            for q, j in enumerate(sel):
                if j < len(fsh):
                    fs[q] = fsh[j][:, :3]
            # common scale with the bodies: the shape of the bodies is normalised to max = 1 → the same factor
            frd_max = _frd_max(workdir / "modal.frd", n, k_all, present=d.get("modes"))
            for q, j in enumerate(sel):
                m_ = max(frd_max[j], float(np.abs(fs[q]).max()))
                if m_ > 0:
                    fs[q] /= m_
                    shapes[q] *= frd_max[j] / m_         # the shape of the bodies was normalised by its maximum only
            mset.frame_shapes = fs
        return mset

    def buckle(cancel_, on_line_) -> ModeSet:
        bs = study.buckling
        names = [lc.name for lc in study.load_cases]
        k = names.index(bs.case) if bs.case in names else 0
        lc = study.load_cases[k]
        log(f"Buckling (CalculiX, case '{lc.name}', {int(bs.modes)} modes)")
        temps = None
        if study.has_thermal_loads() and study.thermal_loads(lc):
            from .thermal.field import node_temperatures
            temps = [node_temperatures(study, mesh, lc, {t.name: t for t in (thermal or [])})]
        st = _sub_study(study, [lc])
        try:
            write_inp(st, mesh, workdir / "buckle.inp", temperatures=temps, procedure="buckle",
                      eigen_line=f"{int(bs.modes)}")
        except ValueError as e:
            raise EigenError(str(e)) from None
        run("buckle", cancel_, on_line_)
        d = read_dat(workdir / "buckle.dat", "buckle")
        if not d["factors"]:
            raise EigenError("Buckling: ccx returned no factor")
        shapes = _shapes(workdir / "buckle.frd", n, len(d["factors"]), skip_first=True)
        return ModeSet("buckling", f"Buckling – {lc.name}", d["factors"][:len(shapes)], shapes, case=lc.name)

    # the two jobs are independent → side by side when the machine has room for both (ccx.concurrent)
    cyclic = getattr(study, "cyclic", None) is not None and study.cyclic.enabled
    tasks = (([cyclic_modal] if cyclic else [modal]) if study.modal.enabled and "modal" in kinds else []) \
        + ([buckle] if study.buckling.enabled and study.load_cases and "buckling" in kinds else [])
    from .ccx.concurrent import run_tasks
    out = run_tasks(tasks, 3 * n, cancel, on_line)
    return out


# ----------------------------------------------------------------- storage (Results.save / load)

def save_arrays(modes: list[ModeSet], arrays: dict) -> list[dict]:
    meta = []
    for j, m in enumerate(modes):
        arrays[f"MODE_{j}"] = m.shapes
        if m.frame_shapes is not None:
            arrays[f"MODEF_{j}"] = m.frame_shapes
        own = getattr(m, "frame_shapes_own", None)      # buckling of members (MYSTRAN) on its own subdivision – display in the GUI
        if own is not None:
            arrays[f"MODEFO_{j}"] = np.asarray(own, np.float32)
        meta.append({"kind": m.kind, "name": m.name, "values": list(map(float, m.values)), "case": m.case,
                     "participation": m.participation, "eff_mass": m.eff_mass, "total_mass": m.total_mass,
                     **({"frame_xyz": getattr(m, "frame_xyz", None)} if own is not None else {}),
                     **({"engine": m.engine, "engine_info": m.engine_info} if getattr(m, "engine", "ccx") != "ccx" or getattr(m, "engine_info", None) else {}),
                     **({"check": m.check} if getattr(m, "check", None) else {})})
        if getattr(m, "nodal_diameters", None) is not None:      # cyclic symmetry (fea_core.cyclic.modal)
            arrays[f"MODEI_{j}"] = m.shapes_imag
            meta[-1].update(nodal_diameters=list(m.nodal_diameters), multiplicity=list(m.multiplicity),
                            sectors=int(m.sectors))
    return meta


def load_arrays(meta: list[dict], d: dict) -> list[ModeSet]:
    out = []
    for j, m in enumerate(meta or []):
        ms = ModeSet(m["kind"], m["name"], m["values"], d[f"MODE_{j}"], m.get("case"), m.get("participation") or [],
                     m.get("eff_mass") or [], m.get("total_mass") or [], d.get(f"MODEF_{j}"), m.get("engine") or "ccx", m.get("engine_info") or {},
                     m.get("check") or {})
        if f"MODEFO_{j}" in d and m.get("frame_xyz") is not None:
            ms.frame_shapes_own = d[f"MODEFO_{j}"]
            ms.frame_xyz = m["frame_xyz"]
        if m.get("nodal_diameters") is not None and f"MODEI_{j}" in d:
            ms.nodal_diameters = [int(v) for v in m["nodal_diameters"]]
            ms.multiplicity = [int(v) for v in m.get("multiplicity") or [1] * len(ms.nodal_diameters)]
            ms.sectors = int(m.get("sectors") or 0)
            ms.shapes_imag = d[f"MODEI_{j}"]
        out.append(ms)
    return out

"""The stage-S4 calculations as functions of a study (the same code serves the MCP tools, the GUI panel and the report): interface forces, notching, ECSS bolts, PCB check.

Every function takes the study (and a mesh / working folder where a modal analysis is needed) and returns a JSON-able dictionary; the calling interface only wraps
errors (`S4Error`) and stores the summary (`s4_store.record`). Limits and factors always come from `S4Limits.from_requirement_set(study.space)` – the numbers a caller
passes are used but never confirmed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


class S4Error(ValueError):
    """An S4 calculation cannot be done for the given input (a message for the user)."""


NOTCH_LIMIT_KEYS = {"force": "notch_force_limit_N", "moment": "notch_moment_limit_Nmm", "probe_acc": "notch_probe_acc_limit_g", "stress": "notch_stress_limit_MPa",
                    "force_psd": "notch_force_psd_limit_N2_Hz"}
BOLT_LIMIT_KEYS = ["bolt_tightening_scatter", "bolt_relaxation_loss", "bolt_slip_mu_min", "bolt_factor_gapping", "bolt_factor_slip", "bolt_factor_tensile_yield",
                   "bolt_factor_tensile", "bolt_factor_shear", "bolt_factor_combined", "bolt_factor_pullout", "bolt_factor_bearing"]
PCB_LIMIT_KEYS = ["pcb_steinberg_const", "pcb_amplitude_factor", "pcb_design_factor", "pcb_n_sigma", "pcb_component_const", "pcb_position_factor_min",
                  "pcb_fatigue_exponent", "pcb_ref_cycles", "pcb_damage_limit"]
SINE_ROWS = 25
MAX_BREAKPOINTS = 80
GRID_LEVELS = ((32, 41, 1.03), (64, 81, 1.015), (128, 161, 1.0075), (256, 321, 1.004))     # (points per half-power band of a resonance for a sine profile, points per peak and
                                                                                           # tail ratio of the random grid), tried in turn (see `notch_report`)
MAX_GRID_POINTS = 150_000                  # the refinement stops before a grid larger than this (the table re-check is up to 10x finer, chunked)
G_MM = 9810.0


def limits_of(st: Any):
    from .s4_limits import S4Limits
    return S4Limits.from_requirement_set(getattr(st, "space", None))


def missing_limits(lim: Any, keys: list[str]) -> list[str]:
    """Limit keys that are not usable (no value, or the requirement set / the entry is not confirmed)."""
    return [k for k in keys if k not in lim.entries or not lim.entries[k].usable]


def study_digest(st: Any, modes: int | None) -> str:
    """What the modal interface data depend on: the study without its space settings and dynamics settings, plus the mode count and the upper frequency."""
    d = st.to_dict()
    for k in ("space", "dynamics", "features"):                     # "features" lists the blocks present (it changes when the space settings are created)
        d.pop(k, None)
    ds = st.dynamics
    text = json.dumps([d, modes or ds.modes, ds.f_max], sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def profile_from_study(st: Any, kind: str, given: list | None) -> list:
    """The sine amplitude table [[f Hz, g]] or the ASD table [[f Hz, g2/Hz]] from the argument or from the dynamics settings of the study."""
    if given:
        try:
            return [[float(a), float(b)] for a, b in given]
        except (TypeError, ValueError):
            raise S4Error("A profile is a table [[f Hz, value], ...]") from None
    ds = st.dynamics
    if kind == "random":
        if ds.random.asd:
            return [[float(a), float(b)] for a, b in ds.random.asd]
    else:
        for e in ds.harmonic.excitation:
            if e.kind == "base_acc" and e.table:
                k = float(e.scale) if e.unit == "g" else float(e.scale) / 9810.0
                return [[float(a), float(b) * k] for a, b in e.table]
    raise S4Error(f"Give the {'ASD table [[f Hz, g2/Hz], ...]' if kind == 'random' else 'sine amplitude table [[f Hz, g], ...]'} (the study has none in its dynamics settings)")


def damping_of(st: Any, mi: Any, zeta: float | None) -> tuple[np.ndarray, str]:
    from ..dynamics.modal import zetas
    if zeta is not None:
        if not 0 < float(zeta) < 1:
            raise S4Error("zeta must be between 0 and 1")
        return np.full(mi.K, float(zeta)), "zeta given"
    sp = st.dynamics.damping
    if sp.kind == "structural":                      # the interface spectra are those of VISCOUS modal damping: structural damping enters as the equivalent ζ = η/2 (as in `modal.zetas`)
        return zetas(sp, mi.freq), f"damping of the study (structural, eta = {sp.eta or 2 * sp.zeta:g}): the equivalent viscous zeta = eta/2 is used"
    return zetas(sp, mi.freq), f"damping of the study ({sp.kind})"


def directions_list(d: str) -> list[str]:
    items = [x.strip().lower() for x in str(d).replace(";", ",").split(",") if x.strip()]
    if not items or any(x not in ("x", "y", "z") for x in items):
        raise S4Error("directions: x, y, z or a comma list")
    return list(dict.fromkeys(items))


def decimate(f: np.ndarray, cols: dict[str, np.ndarray], n: int = SINE_ROWS) -> list[dict]:
    """About n rows of a spectrum: log-spaced samples plus the maximum of the first column."""
    if len(f) <= n:
        idx = np.arange(len(f))
    else:
        want = np.geomspace(f[0], f[-1], n)
        idx = np.unique(np.concatenate([np.searchsorted(f, want).clip(0, len(f) - 1), [int(np.argmax(next(iter(cols.values()))))]]))
    return [{"f_Hz": float(f[i]), **{k: float(v[i]) for k, v in cols.items()}} for i in idx]


# ------------------------------------------------------------------ interface forces
def get_interface(st: Any, mesh: Any, workdir: Callable[[], Path], cache: dict, modes: int | None = None, ref_point: list[float] | None = None,
                  log: Callable[[str], None] | None = None, cancel: Any = None) -> tuple[Any, bool]:
    """The `ModalInterface` of the study (cached in `cache` until the study changes). `workdir` is called only when the modal analysis has to run."""
    from ..dynamics.interface import InterfaceError, run_interface
    key = (study_digest(st, modes), tuple(ref_point) if ref_point else None)
    hit = cache.get("interface")
    if hit is not None and hit[0] == key:
        return hit[1], True
    if mesh is None:
        raise S4Error("No valid mesh for the current study - mesh it (or solve it) first")
    try:
        mi = run_interface(st, mesh, workdir(), modes=modes or st.dynamics.modes, f_max=st.dynamics.f_max, ref_point=ref_point, log=log, cancel=cancel)
    except InterfaceError as e:
        raise S4Error(str(e)) from None
    cache["interface"] = (key, mi)
    return mi, False


def interface_report(st: Any, mi: Any, directions: str = "x,y,z", sine: list | None = None, asd: list | None = None, zeta: float | None = None, group: str | None = None,
                     cached: bool = False) -> dict:
    """The answer of the interface-force calculation: static check, modal table, per direction the sine peaks and the random RMS."""
    from ..dynamics import interface as IF
    if group is not None and group not in mi.groups:
        raise S4Error(f"Unknown group '{group}' (groups: {', '.join(mi.groups)})")
    z, z_src = damping_of(st, mi, zeta)
    d: dict[str, Any] = {"total_mass_kg": 1000 * mi.total_mass, "ref_point_mm": mi.ref_point.tolist(), "modes": mi.K, "engine": mi.info.get("engine"), "from_cache": cached,
                         "groups": {k: g.n_nodes for k, g in mi.groups.items()}, "static_mass_check": mi.info.get("static_mass_check"), "damping": z_src,
                         "warnings": list(mi.warnings)}
    rows = sorted(mi.modal_table(group), key=lambda r: -float(np.linalg.norm(r["F_N"])))[:12]
    d["modal_table"] = [{"mode": r["mode"], "f_Hz": r["f_Hz"], "gamma": r["gamma"], "F_N": r["F_N"], "M_Nmm": r["M_Nmm"]} for r in sorted(rows, key=lambda r: r["mode"])]
    # an environment that was asked for explicitly must exist; one that is only a default of the study settings is skipped when it has no table
    # (an asd-only call needs no base-acceleration table and the other way round)
    sine_t = rand_t = None
    if sine:
        sine_t = profile_from_study(st, "sine", sine)
    elif st.dynamics.harmonic.enabled and st.dynamics.harmonic.excitation:
        try:
            sine_t = profile_from_study(st, "sine", None)
        except S4Error:
            if not (asd or st.dynamics.random.asd):
                raise
    if asd:
        rand_t = profile_from_study(st, "random", asd)
    elif st.dynamics.random.asd:
        rand_t = profile_from_study(st, "random", None)
    try:
        for ax in directions_list(directions):
            item: dict[str, Any] = {}
            if sine_t:
                sp = IF.sine_interface(mi, ax, sine_t, z, group=group)
                item["sine"] = {**sp.summary(), "spectrum": decimate(sp.f, {"F_peak_N": sp.F_peak, "M_peak_Nmm": sp.M_peak})}
            if rand_t:
                item["random"] = IF.random_interface(mi, ax, rand_t, z, group=group).summary()
            d.setdefault("directions", {})[ax] = item
    except (IF.InterfaceError, ArithmeticError) as e:
        raise S4Error(str(e)) from None
    if not sine_t and not rand_t:
        d["hint"] = "give sine or asd (or set the harmonic / random dynamics of the study) to get the spectra"
    return d


# ------------------------------------------------------------------ notching
def notch_report(st: Any, kind: str, direction: str, limit_kind: str, *, limit: float | None = None, limit_table: list | None = None, profile: list | None = None,
                 design_factor: float | None = None, min_width_oct: float | None = None, floor_ratio: float | None = None, group: str | None = None,
                 probe: str | None = None, zeta: float | None = None, interface: Callable[[], Any] | None = None, dynamics: Any = None,
                 breakpoint_tol_db: float = 0.1) -> tuple[dict, Any]:
    """Notched profile + justification: (the answer dictionary, the `NotchResult`). `interface()` returns the `ModalInterface` (force / moment limits);
    `dynamics` = the `DynamicsResults` of a harmonic base-acceleration run (probe_acc / stress limits)."""
    from ..dynamics import interface as IF
    from ..dynamics.psd import loglog_interp
    from . import notching as NT
    if kind not in NT.KINDS or limit_kind not in NT.LIMIT_KINDS:
        raise S4Error(f"kind: {', '.join(NT.KINDS)}; limit_kind: {', '.join(NT.LIMIT_KINDS)}")
    lim = limits_of(st)
    spec = NT.NotchSpec(kind, limit_kind, limit, limit_table, design_factor, min_width_oct, floor_ratio, breakpoint_tol_db=float(breakpoint_tol_db))
    prof = profile_from_study(st, kind, profile)
    modes_hz = None
    notes: list[str] = []
    res = None
    ctx: dict[str, Any] = {"kind": kind, "limit_kind": limit_kind, "profile": prof}
    try:
        if limit_kind in ("probe_acc", "stress"):
            if kind != "sine":
                raise S4Error("probe_acc / stress limits apply to a sine profile")
            base_dir = NT.check_base_only(getattr(getattr(st.dynamics, "harmonic", None), "excitation", None))
            f, t, notes, probe_name = NT.harmonic_response_ex(dynamics, limit_kind, probe)
            level = loglog_interp(prof, f)
            keep = level > 0
            if not keep.any():
                raise S4Error("The profile does not overlap the frequencies of the harmonic result")
            f, level, resp = f[keep], level[keep], level[keep] * t[keep]
            spec.scope = f"at probe '{probe_name}'"
            ctx.update(probe=probe_name, direction=base_dir, damping="damping of the harmonic result")
        else:
            if interface is None:
                raise S4Error("The modal interface data are needed")
            mi = interface()
            z, z_src = damping_of(st, mi, zeta)
            modes_hz = mi.freq
            from ..dynamics import psd as PSD

            def table_sine(ff: np.ndarray, aa: np.ndarray) -> np.ndarray:
                TF, TM = IF.interface_transfer(mi, ff, z, direction, group)
                return IF.peak_magnitude((TM if limit_kind == "moment" else TF) * (aa * G_MM)[:, None])

            def table_random(ff: np.ndarray, aa: np.ndarray) -> np.ndarray:
                TF, _TM = IF.interface_transfer(mi, ff, z, direction, group)
                return (TF.real ** 2 + TF.imag ** 2).sum(axis=1) * PSD.asd_to_mm(aa)

            # the grid of the analysis is refined automatically until the exported breakpoint table keeps the limit between the grid points too: the points scale with the
            # half-power band of every resonance (a fixed number of points per band, one of them at f_n), so the refinement converges for any damping
            level_no = 0
            capped = False
            for level_no, (per_band, per_peak, ratio) in enumerate(GRID_LEVELS, 1):
                if kind == "sine":
                    sp = IF.sine_interface(mi, direction, prof, z, group=group, per_band=per_band)
                    f, level, resp = sp.f, sp.a_g, NT.interface_response(sp, "moment" if limit_kind == "moment" else "force")
                else:
                    ri = IF.random_interface(mi, direction, prof, z, group=group, per_peak=per_peak, ratio=ratio)
                    f, level, resp = ri.f, ri.asd_g2, NT.interface_psd_response(ri)
                if res is not None and len(f) > MAX_GRID_POINTS:
                    level_no -= 1
                    capped = True
                    break
                res = NT.make_notch(f, level, resp, spec, lim, modes_hz, notes, table_check=table_sine if kind == "sine" else table_random)
                if res.table_ratio is None or res.table_ratio <= 1.0 + NT.TABLE_TOL:
                    break
            n_grid = len(res.f) if res is not None else 0
            if res is not None and res.table_ratio is not None and res.table_ratio > 1.0 + NT.TABLE_TOL and res.verdict == NT.FAIL and "exported breakpoint table" in res.verdict_text:
                res.verdict_text += (f" - the grid was refined automatically {max(level_no - 1, 0)} time(s) up to {n_grid} frequencies and the table still exceeds the limit by "
                                     f"{100.0 * (res.table_ratio - 1.0):.2f} %" + (" (the grid size limit was reached)" if capped else "") +
                                     ": widen the minimum notch width of the requirement set or lower the profile level around the resonance, then repeat")
            ctx.update(grid_level=level_no, grid_levels=len(GRID_LEVELS), grid_points=n_grid, grid_per_half_power_band=int(GRID_LEVELS[max(level_no, 1) - 1][0]),
                       direction=str(direction), damping=z_src, zeta=[float(v) for v in np.unique(np.round(z, 12))[:6]], group=group or "all", modes=int(mi.K),
                       engine=mi.info.get("engine"))
        if res is None:
            res = NT.make_notch(f, level, resp, spec, lim, modes_hz, notes, require_table_check=True)
    except MemoryError:
        raise S4Error("Not enough memory for the frequency response of the notching (too many modes or too wide a frequency range): "
                      "reduce the number of modes or the frequency range of the profile and repeat") from None
    except (NT.NotchError, IF.InterfaceError, ArithmeticError, TypeError, ValueError) as e:        # (S4Error is a ValueError: its message is kept)
        raise S4Error(str(e)) from None
    d = res.summary()
    d.update({k: v for k, v in ctx.items() if k != "profile"})
    d["profile"] = [[float(a), float(b)] for a, b in ctx["profile"]]
    d["justification"] = [dict(zip(NT.JUSTIFICATION_COLUMNS, r)) for r in NT.justification_rows(res)]
    shown = res.breakpoints[:MAX_BREAKPOINTS]
    d["breakpoints"] = [[round(a, 4), b] for a, b in shown]
    d["breakpoints_total"] = len(res.breakpoints)
    d["breakpoints_truncated"] = len(res.breakpoints) > len(shown)
    if d["breakpoints_truncated"]:
        d["breakpoints_note"] = f"only the first {len(shown)} of {len(res.breakpoints)} breakpoints are listed here; the full table is in the CSV export (export_path)"
    d["limits_needed"] = missing_limits(lim, ["notch_min_width_oct", "notch_floor_ratio", "notch_design_factor", NOTCH_LIMIT_KEYS[limit_kind]])
    return d, res


# ------------------------------------------------------------------ bolts
def material_allowables(st: Any, bolt_material: str | None, part_material: str | None, bearing_key: str = "Fu") -> dict:
    """Allowables of the bolt and of the clamped part from the study materials (`fea_space_allowables_set` records); verified = confirmed with basis A or B."""
    from . import bolts_ecss as BE
    from .requirements import allowable_of
    res: dict = {}
    if bearing_key not in ("Fu", "Fy"):
        raise S4Error("bearing_allowable must be Fu or Fy")
    for name, role in ((bolt_material, "bolt"), (part_material, "part")):
        if not name:
            continue
        mat = st.materials.get(name)
        if mat is None:
            raise S4Error(f"Material '{name}' is not in the study (materials: {', '.join(st.materials) or 'none'})")
        rec = allowable_of(mat)
        if rec is None:
            raise S4Error(f"Material '{name}' has no allowables entered (fea_space_allowables_set / the Material allowables tab)")
        ver = rec.is_verified()
        src = f"{name}: {rec.source}" + ("" if ver else " (not verified)")
        pairs = (("bolt_Fty", rec.Fy), ("bolt_Ftu", rec.Fu), ("bolt_Fsy", rec.Fsy), ("bolt_Fsu", rec.Fsu)) if role == "bolt" else \
            (("part_Fsu", rec.Fsu), ("part_Fbr", getattr(rec, bearing_key)), ("hole_Fbr", getattr(rec, bearing_key)), ("part_Fy", rec.Fy))
        for key, v in pairs:
            if v is not None:
                res[key] = BE.Allowable(float(v), ver, src)
    return res


def bolt_from_dict(st: Any, raw: dict, allow: dict):
    """`EcssBolt` from a plain dictionary (`joint` = the name of a bolt of the study: stiffness, stress area, preload from the VDI 2230 evaluation)."""
    from . import bolts_ecss as BE
    c = dict(raw)
    if "combined_exponent" in c:
        raise S4Error("this bolt record was stored with an older bolt model (the field combined_exponent no longer exists: the interaction of tension and shear follows "
                      "ECSS-E-HB-32-23A Rev.1 and the shear plane `shear_plane`) - enter the bolt again and re-run the calculation")
    layers = [BE.Layer(float(x["t_mm"]), float(x["alpha"]), str(x.get("name", ""))) for x in c.pop("layers", [])]
    joint = c.pop("joint", None)
    extra_allow = c.pop("allowables", None) or {}
    if joint:
        from ..joints.fem import bolt_basics
        spec = next((b for b in st.bolts if b.name == joint), None)
        if spec is None:
            raise S4Error(f"Unknown bolt '{joint}' in the study (bolts: {', '.join(b.name for b in st.bolts) or 'none'})")
        b = BE.from_vdi_basics(bolt_basics(st, spec), str(c.pop("name", joint)))
        for k, v in c.items():
            if not hasattr(b, k):
                raise S4Error(f"unknown bolt field '{k}' (a record stored with an older bolt model? - enter the bolt again and re-run the calculation)")
            setattr(b, k, v)
    else:
        c.setdefault("name", "bolt")
        try:
            b = BE.EcssBolt(**c)
        except TypeError as e:
            raise S4Error(f"unknown or misspelled bolt field ({e}); a record stored with an older bolt model must be entered again and the calculation re-run") from None
    if layers:
        b.layers = layers
    b.allow = {**allow, **{k: BE.Allowable(float(v), False, "entered directly (not confirmed)") for k, v in extra_allow.items()}}
    return b


def bolts_report(st: Any, bolts: list[dict], bolt_material: str | None = None, part_material: str | None = None, bearing_allowable: str = "Fu") -> dict:
    from . import bolts_ecss as BE
    from .s4_limits import OBSOLETE_LIMITS
    lim = limits_of(st)
    allow = material_allowables(st, bolt_material, part_material, bearing_allowable)
    res = []
    if isinstance(bolts, dict):
        bolts = [bolts]
    if not isinstance(bolts, (list, tuple)):
        raise S4Error("bolts must be a list of dictionaries (one per bolt)")
    for raw in bolts:
        if not isinstance(raw, dict):
            raise S4Error(f"bolt: every item must be a dictionary of the bolt inputs (got {type(raw).__name__})")
        try:
            r = BE.assess_bolt(bolt_from_dict(st, raw, allow), lim).to_dict()
            r["warnings"] = list(r["warnings"]) + [f"requirement-set row {k}: {OBSOLETE_LIMITS[k]}" for k in lim.obsolete]
            # what was assessed, so that a stored verdict can be recomputed against the present requirement set (`s4_verify`)
            r["spec"] = {"raw": raw, "bolt_material": bolt_material, "part_material": part_material, "bearing_allowable": bearing_allowable}
            res.append(r)
        except (BE.BoltsError, KeyError, TypeError, ValueError, ArithmeticError) as e:
            raise S4Error(f"bolt '{raw.get('name', '?')}': {e}") from None
    return {"formula_note": BE.FORMULA_NOTE, "deviations": list(BE.DEVIATIONS), "bolts": res, "limits_needed": missing_limits(lim, BOLT_LIMIT_KEYS),
            "overall": ("FAIL" if any(r["verdict"] == "FAIL" for r in res) else "UNCONFIRMED_REQUIREMENTS" if any(r["verdict"] != "PASS" for r in res) else "PASS")}


# ------------------------------------------------------------------ PCB
def pcb_report(st: Any, components: list[dict], *, mesh: Any = None, workdir: Callable[[], Path] | None = None, direction: str = "z", asd: list | None = None,
               zeta: float | None = None, modes: int | None = None, log: Callable[[str], None] | None = None) -> dict:
    """Steinberg check of the components; `geometry` {centre, supports | support_selection, normal} takes the RMS relative displacement from the modal basis."""
    from . import pcb as PB
    lim = limits_of(st)
    rows = []
    if isinstance(components, dict):
        components = [components]
    if not isinstance(components, (list, tuple)):
        raise S4Error("components must be a list of dictionaries (one per component)")
    basis = None                                                  # the modal basis is the same for every component (the study, the mesh and the mode count do not change): once
    for raw in components:
        if not isinstance(raw, dict):
            raise S4Error(f"component: every item must be a dictionary of the component inputs (got {type(raw).__name__})")
        c = dict(raw)
        try:
            geo = c.pop("geometry", None)
            if "duration_h" in c:
                c["duration_s"] = 3600.0 * float(c.pop("duration_h"))
            extra: dict = {}
            if geo is not None:
                if mesh is None or workdir is None:
                    raise S4Error("No valid mesh for the current study - mesh it first (the RMS displacement is taken from its modal basis)")
                if zeta is not None and not 0 < float(zeta) < 1:
                    raise S4Error("zeta must be between 0 and 1")
                if basis is None:
                    basis = PB.pcb_basis(st, mesh, workdir(), modes=modes or st.dynamics.modes, log=log)
                info = PB.z_rms_from_study(st, mesh, "", centre=geo["centre"], supports=geo.get("support_selection") or geo.get("supports"), normal=geo["normal"],
                                           direction=direction, asd=profile_from_study(st, "random", asd), zeta=None if zeta is None else float(zeta),
                                           modes=modes or st.dynamics.modes, log=log, basis=basis)           # zeta None = the damping of the study (per mode, `damping_of`)
                c["z_model_mm"] = info["z_rms_mm"]                    # the model-derived value: a typed z_rms_mm that differs from it is never confirmed
                if c.get("z_rms_mm") is None:
                    c["z_rms_mm"] = info["z_rms_mm"]
                    c["z_source"] = f"modal analysis of the study ({info['modes']} modes, {info['damping']}, base direction {direction})"
                if c.get("f_n_Hz") is None:
                    c["f_n_Hz"] = info["dominant_f_Hz"]
                extra = {"z_rms_from_model": info}
            r = PB.check_pcb(PB.PcbSpec(**c), lim).to_dict()
            r["spec"] = dict(c)                                   # the typed / model-derived inputs: a stored verdict can be recomputed (`s4_verify`)
            r.update(extra)
            rows.append(r)
        except (PB.PcbError, KeyError, TypeError, ValueError, ArithmeticError) as e:
            if isinstance(e, S4Error):
                raise
            raise S4Error(f"component '{raw.get('name', '?')}': {e}") from None
    return {"components": rows, "limits_needed": missing_limits(lim, PCB_LIMIT_KEYS), "unit_note": "the unit of pcb_steinberg_const must be written with the value (in | mm)"}

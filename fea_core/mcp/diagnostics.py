"""Evaluation of the results for the LLM: fields to display, summary, places of the maxima with the classification of the singularities.

Classification of the maximum (heuristics over the mesh, with a justification):
  - singularity: an edge of a clamped / constrained face, a sharp inner (concave) edge or a corner without a fillet,
    a small (almost point) loaded face, connection of a bolt (rigid constraint of the beam);
  - probably influenced by the boundary condition: inside a face of a constraint / at a constraint, an edge of a loaded face;
  - real concentration: fillet, hole, smooth surface → check by the convergence of the mesh.
A steep gradient (node ≫ surroundings at the distance of ~1 element) is stated as a sign of the dependence on the mesh.

Heat (fea_core.thermal): fields temperature / heat_flux of the thermal states (Results.thermal), summary with the fluxes of the boundary
conditions and the heat balance, the hottest places. The topology of the mesh (KD tree, surface, selections, node → elements) is computed
once per object Results (MeshCache) – the repeated calls of the summary, of the places of the maxima and of the render share it.
"""
from __future__ import annotations

import math

import numpy as np

from ..gravity_states import str_missing_by_superposition
from ..hotspots import MeshCache, SurfaceInfo, classify_hotspots, mesh_cache  # noqa: F401 – the classifier moved to fea_core.hotspots; re-exported
from ..hotspots import is_sliding as _is_sliding, mean_edge as _mean_edge, pick as _pick, tri_keys as _tri_keys  # noqa: F401
from .session import FeaError, Session

# key: (description, unit)
FIELDS = {
    "mises": ("von Mises", "MPa"), "s1": ("1st principal stress σ1", "MPa"), "s2": ("2nd principal stress σ2", "MPa"),
    "s3": ("3rd principal stress σ3", "MPa"), "tresca": ("Tresca (σ1 − σ3)", "MPa"),
    "sxx": ("σx", "MPa"), "syy": ("σy", "MPa"), "szz": ("σz", "MPa"), "sxy": ("τxy", "MPa"),
    "syz": ("τyz", "MPa"), "szx": ("τzx", "MPa"),
    "displacement": ("Total displacement |u|", "mm"), "ux": ("Displacement ux", "mm"), "uy": ("Displacement uy", "mm"),
    "uz": ("Displacement uz", "mm"),
    "safety": ("Safety factor to Re (Re/σ_vM, max 10)", "–"),
    "print_safety": ("3D printing – safety factor (max 10)", "–"), "print_index": ("3D printing – failure index", "–"),
    "fatigue_damage": ("Fatigue – damage D (log10)", "log10 D"), "fatigue_life": ("Fatigue – life (log10 blocks)", "log10 blocks"),
    "fatigue_safety": ("Fatigue – safety factor (max 10)", "–"), "fatigue_amplitude": ("Fatigue – equivalent stress amplitude", "MPa"),
    "temperature": ("Temperature", "°C"), "heat_flux": ("Heat flux |q|", "W/m²"),
    "hfx": ("Heat flux q_x", "W/m²"), "hfy": ("Heat flux q_y", "W/m²"), "hfz": ("Heat flux q_z", "W/m²"),
    "stress_error": ("ccx stress error estimate (STR)", "%"),
    "topo_density": ("Topology – density (1 = material, 0 = removed)", "–"),
    "peeq": ("Equivalent plastic strain PEEQ", "–"),
}
ALIASES = {"von_mises": "mises", "vm": "mises", "seqv": "mises", "u": "displacement", "umag": "displacement",
           "disp": "displacement", "usum": "displacement", "sf": "safety", "s_1": "s1", "s_3": "s3",
           "t": "temperature", "temp": "temperature", "temperature": "temperature", "nt": "temperature",
           "flux": "heat_flux", "hfl": "heat_flux", "tok": "heat_flux", "q": "heat_flux", "heatflux": "heat_flux",
           "error": "stress_error", "str": "stress_error", "err": "stress_error", "teplota": "temperature", "chyba": "stress_error"}
REVERSED = {"safety", "print_safety", "fatigue_safety", "fatigue_life"}
SIGNED = {"s1", "s2", "s3", "sxx", "syy", "szz", "sxy", "syz", "szx", "ux", "uy", "uz", "hfx", "hfy", "hfz"}
THERMAL_KEYS = {"temperature", "heat_flux", "hfx", "hfy", "hfz"}
FLUX_KEYS = THERMAL_KEYS - {"temperature"}

# thermal boundary conditions (fea_core.thermal.model.THERMAL_KINDS) – descriptions for the LLM, pictures and the report
THERMAL_LABELS = {"fixed_temperature": "prescribed temperature", "heat_flux": "heat flux", "heat_flow": "heat power",
                  "convection": "heat transfer (convection)", "radiation": "radiation to ambient",
                  "heat_generation": "internal heat generation", "cavity": "cavity radiation (surface-to-surface)",
                  "cavity_radiation": "cavity radiation (surface-to-surface)", "contact_conductance": "contact conductance"}


def thermal_kind(bc) -> str:
    """Kind of a thermal boundary condition for labels and colours: "cavity" for cavity radiation (Radiation with
    cavity=True or CavityRadiation), otherwise bc.kind."""
    from ..thermal.model import is_cavity
    return "cavity" if is_cavity(bc) else bc.kind


def thermal_label(bc) -> str:
    return THERMAL_LABELS.get(thermal_kind(bc), bc.kind)


def no_solid_hint(res) -> str:
    """Message for the tools of the 3D results when the result has no 3D mesh (bars, 2D section, shells)."""
    if getattr(res, "twod", None) is not None:
        return ("The result is a 2D section analysis (no 3D mesh) – fea_twod_results, fea_twod_probe, fea_twod_line, fea_twod_linearize, fea_twod_render")
    if getattr(res, "shell", None) is not None:
        return "The result is a shell model (no 3D mesh) – fea_shell_results, fea_shell_probe, fea_shell_render"
    return "The study has no solid bodies (beam model only) – member results: fea_frame_results, image fea_frame_render"


def field_key(field: str) -> str:
    k = ALIASES.get(field.strip().lower(), field.strip().lower())
    if k not in FIELDS:
        raise FeaError(f"Unknown quantity '{field}'. Available: {', '.join(FIELDS)}")
    return k


def bc_value_text(bc) -> str:
    """Value of a thermal boundary condition with units (decimal point – for the LLM and the pictures)."""
    k = bc.kind
    if k == "fixed_temperature":
        return f"T = {bc.value:g} °C"
    if k == "heat_flux":
        return f"q = {bc.value:g} W/m²"
    if k == "heat_flow":
        return f"Q = {bc.value:g} W"
    if k == "convection":
        return f"h = {bc.h:g} W/(m²·K), ambient {bc.ambient:g} °C"
    if k == "radiation" and getattr(bc, "cavity", False):
        k = "cavity_radiation"
    if k == "radiation":
        return f"ε = {bc.emissivity:g}, ambient {bc.ambient:g} °C"
    if k == "cavity_radiation":
        name = getattr(bc, "cavity_name", None) or "CAV"
        return (f"ε = {bc.emissivity:g}, cavity {name}, closed" if bc.closed
                else f"ε = {bc.emissivity:g}, cavity {name}, open to ambient {bc.ambient:g} °C")
    if k == "contact_conductance":
        return f"h_c = {bc.conductance:g} W/(m²·K) between {bc.selection} and {bc.master}"
    if k == "heat_generation":
        return f"{bc.value:g} W in the volume of part {bc.part}"
    return k


def thermal_load_text(l) -> str:
    """Description of the mechanical thermal load (study.ThermalLoad)."""
    if l.case:
        return f"temperature field from thermal state '{l.case}'" + ("" if l.time is None else f" at time {l.time:g} s")
    parts = ", ".join(l.parts) if l.parts else "of all parts"
    return f"uniform temperature {l.value:g} °C ({('parts ' + parts if l.parts else parts)})"


def field_values(sess: Session, step: int, key: str) -> np.ndarray:
    res = sess.require_results()
    st = res.steps[step]
    if key == "mises":
        return st.mises
    if key in ("s1", "s2", "s3", "tresca"):
        p = st.principal
        return p[:, 0] - p[:, 2] if key == "tresca" else p[:, int(key[1]) - 1]
    comp = {"sxx": 0, "syy": 1, "szz": 2, "sxy": 3, "syz": 4, "szx": 5}
    if key in comp:
        return st.S[:, comp[key]]
    if key == "displacement":
        return st.umag
    if key in ("ux", "uy", "uz"):
        return st.U[:, "xyz".index(key[1])]
    if key == "safety":
        re = sess.node_re()
        if not np.isfinite(re).any():
            raise FeaError("No material has Re – safety factor cannot be computed")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.minimum(np.where(np.isfinite(re), re / np.maximum(st.mises, 1e-12), 10.0), 10.0)
    if key in ("print_safety", "print_index"):
        v = st.fields.get(key)
        if v is None:
            raise FeaError("The 3D printing quantity is not in the results (no printed part)")
        return np.minimum(np.nan_to_num(v, nan=10.0, posinf=10.0), 10.0) if key == "print_safety" else np.nan_to_num(v)
    if key.startswith("fatigue_"):
        fr = sess.fatigue
        if fr is None:
            raise FeaError("Fatigue has not been computed – run fea_fatigue_run")
        if key == "fatigue_damage":
            return np.log10(np.maximum(fr.total_damage, 1e-12))
        if key == "fatigue_life":
            return np.log10(np.minimum(np.maximum(fr.life_blocks, 1.0), 1e12))
        if key == "fatigue_safety":
            return np.minimum(np.nan_to_num(fr.safety, nan=10.0, posinf=10.0), 10.0)
        return fr.amplitude
    if key == "temperature":
        T = st.fields.get("T")
        if T is None:
            raise FeaError(f"Case '{st.name}' has no thermal load – temperature is in the thermal case result (parameter thermal_case)")
        return T
    if key in FLUX_KEYS:
        raise FeaError("Heat flux is available only in thermal case results – specify thermal_case")
    if key == "stress_error":
        v = st.fields.get("stress_error")
        if v is None:
            if str_missing_by_superposition(res.info, st.name):
                raise FeaError("The stress error estimate (STR) is not available for gravity cases made as combinations of unit states (the estimate is not linear in the load)")
            raise FeaError("The stress error estimate (STR) is available only in CalculiX results (ccx backend) – not in the direct / iterative solver results")
        return v
    if key == "topo_density":
        v = st.fields.get("topo_density")
        if v is None:
            raise FeaError("The result has no topology optimization – run fea_topo_opt_run")
        return v
    if key == "peeq":
        v = st.fields.get("peeq")
        if v is None:
            raise FeaError("The result has no plastic strain PEEQ – no material of the model has plasticity")
        return np.maximum(np.asarray(v, float), 0.0)
    raise FeaError(f"Quantity '{key}' is not available")


def thermal_values(sess: Session, k: int, key: str, time_s: float | None = None) -> np.ndarray:
    """Field of the thermal state k: temperature (for a transient state at the time time_s) or heat flux (the end of the state)."""
    res = sess.require_results()
    tr = res.thermal[k]
    if key == "temperature":
        return tr.at_time(time_s)
    if key not in FLUX_KEYS:
        raise FeaError(f"Quantity '{key}' belongs to a mechanical case – a thermal case has only temperature, heat_flux, hfx, hfy, hfz")
    if time_s is not None and tr.transient and tr.times and \
            abs(float(time_s) - tr.times[-1]) > 1e-9 * max(1.0, abs(tr.times[-1])):
        raise FeaError(f"Heat flux of transient case '{tr.name}' is stored only at the end of the case (t = {tr.times[-1]:g} s) – omit time_s")
    if key == "heat_flux":
        return tr.flux_mag
    return tr.flux[:, "xyz".index(key[2])]


def resolve_source(sess: Session, key: str, step=0, thermal_case=None) -> tuple[str, int]:
    """Where to take the field: ("thermal", index of the thermal state) or ("step", index of the mechanical state).

    thermal_case given → thermal state; heat flux → the first thermal state; temperature → the mechanical state with a thermal load (step), otherwise the first thermal state; other quantities → the mechanical state step.
    """
    res = sess.require_results()
    if not len(res.mesh.nodes):
        raise FeaError(no_solid_hint(res))
    if thermal_case is not None:
        return "thermal", sess.thermal_index(thermal_case)
    if key in FLUX_KEYS:
        return "thermal", sess.thermal_index(0)
    if key == "temperature":
        if res.steps:
            i = sess.step_index(step)
            if "T" in res.steps[i].fields:
                return "step", i
        if res.thermal:
            return "thermal", 0
        raise FeaError("Results contain no temperature (no thermal case or thermal load in the case)")
    return "step", sess.step_index(step)


# ===================================================================== places of the maxima



def hotspots(sess: Session, step: int, top_n: int = 5, field: str = "mises", tags: bool = False) -> dict:
    """Places of the maxima of the field with the classification (see `fea_core.hotspots`). tags=True adds to the places "tags" = short labels of
    the kind of the singularity / influence of the boundary condition for the GUI (e.g. "clamping edge", "sharp inner edge", "point
    load"); the output of MCP is without them.
    """
    res = sess.require_results()
    st_res = res.steps[step]
    key = field_key(field)
    vals = field_values(sess, step, key)
    if key in SIGNED:
        vals = np.abs(vals)
    out_list, h, r_cluster = classify_hotspots(sess.study, res, st_res, vals, top_n, tags)
    d = {"field": key, "unit": FIELDS[key][1], "step": st_res.name, "mean_element_edge_mm": h,
         "cluster_radius_mm": r_cluster, "hotspots": out_list}
    if st_res.fields.get("stress_error") is None and not tags:
        d["stress_error_note"] = no_error_estimate_note(res, st_res.name)
    return d


def no_error_estimate_note(res, step_name: str | None = None) -> str:
    """Why the ccx stress error estimate (stress_error_pct) is missing in a result (of the step `step_name`; None = any gravity case of the result is meant)."""
    if str_missing_by_superposition(getattr(res, "info", None), step_name):
        return ("No ccx stress error estimate (stress_error_pct) for the gravity cases of this result: they are combinations of unit states and the estimate is not linear in the load, "
                "so it cannot be combined. A case that is not a pure gravity case carries it where the solver is CalculiX.")
    return ("No ccx stress error estimate (stress_error_pct) in this result: only linear CalculiX results carry it (solver backend "
            f"'ccx'; this result: {res.info.get('backend') or '?'}). Solve with backend='ccx' to get it.")


def thermal_hotspots(sess: Session, k: int, top_n: int = 5, field: str = "temperature",
                     time_s: float | None = None) -> dict:
    """The hottest places (temperature) or the places of the highest heat flux of the thermal state k (separate places),
    with the boundary conditions at the place; for the flux a sharp inner edge = singularity. In addition the coldest place.
    """
    res = sess.require_results()
    mesh = res.mesh
    tr = res.thermal[k]
    key = field_key(field)
    vals = thermal_values(sess, k, key, time_s)
    rank_vals = np.abs(vals) if key in SIGNED else vals
    T = tr.at_time(time_s)
    q = tr.flux_mag
    c = mesh_cache(res)
    surf = c.surface
    h = c.mean_edge
    diag = float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)))
    r_cluster = max(3.0 * h, 0.02 * diag)
    case = next((cs for cs in sess.study.thermal.cases if cs.name == tr.name), None) if sess.study else None
    bcs = list(case.loads) if case is not None else []
    re_idx, re_tree = c.reentrant
    out_list = []
    for rank, n in enumerate(_pick(mesh.nodes, rank_vals, top_n, r_cluster), 1):
        x = mesh.nodes[n]
        reasons, cls = [], None
        parts = c.node_parts(n)
        for bc in bcs:
            sel = getattr(bc, "selection", None)
            if sel is not None:
                if sel in mesh.surfaces and len(mesh.surfaces[sel]) and c.selection(sel)[0][n]:
                    reasons.append(f"on face '{sel}': {thermal_label(bc)} {bc_value_text(bc)}")
            elif bc.kind == "heat_generation" and bc.part in parts:
                reasons.append(f"in part '{bc.part}' with heat generation {bc.value:g} W")
        if len(parts) > 1:
            reasons.append(f"at the interface of parts {', '.join(parts)} (perfect thermal contact)")
        if not surf.surface_nodes[n]:
            reasons.append("inside the material")
        if key in FLUX_KEYS:
            dre = re_tree.query(x)[0] if re_tree is not None else math.inf
            if dre <= h:
                reasons.append("at a sharp inner (concave) edge/corner – heat flux is theoretically infinite there, the value depends on the mesh")
                cls = "singularity"
            else:
                cls = "probably the true flux maximum"
        row = {"rank": rank, "node": n + 1, "xyz_mm": x, "value": float(vals[n]), "T_C": float(T[n]),
               "heat_flux_W_m2": float(q[n]), "parts": parts, "on_surface": bool(surf.surface_nodes[n]),
               "reasons": reasons}
        if cls:
            row["classification"] = cls
        out_list.append(row)
    d = {"field": key, "unit": FIELDS[key][1], "thermal_case": tr.name, "transient": tr.transient,
         "mean_element_edge_mm": h, "cluster_radius_mm": r_cluster, "hotspots": out_list}
    if tr.transient:
        d["time_s"] = float(tr.times[-1]) if time_s is None and tr.times else time_s
    if key == "temperature":
        i = int(np.argmin(T))
        d["coldest"] = {"node": i + 1, "xyz_mm": mesh.nodes[i], "T_C": float(T[i]), "parts": c.node_parts(i)}
    return d


# ===================================================================== souhrn

def frame_step_summary(st) -> dict:
    """Frame system in the state: extremes by bars (N, V, M, T, σ), reactions of the nodes, max. displacement (beams.results)."""
    from ..beams.results import frame_summary
    fs = frame_summary(st.frame)
    sig = [(v.get("sigma_eq_max_MPa", 0.0), k) for k, v in fs["members"].items()]
    mom = [(max(v.get("My", {}).get("max_abs", 0.0), v.get("Mz", {}).get("max_abs", 0.0)), k)
           for k, v in fs["members"].items()]
    if sig:
        fs["max_sigma_eq_MPa"] = {"value": max(sig)[0], "member": max(sig)[1]}
    if mom:
        fs["max_abs_moment_Nmm"] = {"value": max(mom)[0], "member": max(mom)[1]}
    return fs


def step_summary(sess: Session, i: int, with_hotspots: bool = True) -> dict:
    res = sess.require_results()
    st = res.steps[i]
    m = res.mesh
    if not len(m.nodes):                                  # frame system only
        d = {"step": st.name, "index": i, "frame": frame_step_summary(st)}
        eq = st.equilibrium or {}
        if eq:
            d["equilibrium"] = {"applied_load_N": eq.get("applied_exact"), "reaction_total_N": eq.get("reaction_total"),
                                "force_residual_N": eq.get("force_residual"), "imbalance_relative": eq.get("imbalance"),
                                "ok": (eq.get("imbalance") or 0) <= 1e-3}
        d["max_displacement_mm"] = {"value": d["frame"].get("max_node_displacement_mm", 0.0)}
        return d
    c = mesh_cache(res)
    study = sess.study
    vm = st.mises
    um = st.umag
    pr = st.principal
    iu, iv = int(np.argmax(um)), int(np.argmax(vm))
    i1, i3 = int(np.argmax(pr[:, 0])), int(np.argmin(pr[:, 2]))
    d = {
        "step": st.name, "index": i,
        "max_displacement_mm": {"value": float(um[iu]), "at_mm": m.nodes[iu], "vector_mm": st.U[iu]},
        "displacement_range_mm": {c_: [float(st.U[:, k].min()), float(st.U[:, k].max())] for k, c_ in enumerate(("ux", "uy", "uz"))},
        "max_von_mises_MPa": {"value": float(vm[iv]), "at_mm": m.nodes[iv], "part": c.node_part(iv)},
        "max_principal_s1_MPa": {"value": float(pr[i1, 0]), "at_mm": m.nodes[i1]},
        "min_principal_s3_MPa": {"value": float(pr[i3, 2]), "at_mm": m.nodes[i3]},
        "reactions": {k: {"force_N": v, "magnitude_N": float(np.linalg.norm(v)),
                          "moment_Nmm": st.reaction_moments.get(k)} for k, v in st.reactions.items()},
    }
    if st.moment_point is not None:
        d["reaction_moment_point_mm"] = st.moment_point
    if st.remote:                                         # remote constraints: reaction ABOUT THE POINT of the constraint, displacement / rotation of the point
        d["remote_supports"] = {k: {"point_mm": r["point"], "coupling": r["coupling"],
                                    "fixed_dofs": [n for n, on in zip(("ux", "uy", "uz", "rx", "ry", "rz"), r["dofs"]) if on],
                                    "force_N": r["force"], "moment_about_point_Nmm": r["moment"],
                                    "point_displacement_mm": r["u"], "point_rotation_rad": r["rot"]}
                                for k, r in st.remote.items()}
    eq = st.equilibrium or {}
    if eq:
        d["equilibrium"] = {"applied_load_N": eq.get("applied_exact"), "reaction_total_N": eq.get("reaction_total"),
                            "force_residual_N": eq.get("force_residual"), "imbalance_relative": eq.get("imbalance"),
                            "ok": (eq.get("imbalance") or 0) <= 1e-3}
    T = st.fields.get("T")
    if T is not None:
        lc = next((x for x in study.load_cases if x.name == st.name), None)
        src = [thermal_load_text(l) for l in lc.loads if l.kind == "thermal"] if lc is not None else []
        d["temperature_C"] = {"min": float(T.min()), "max": float(T.max()),
                              "reference": study.thermal.reference_temperature, "source": "; ".join(src) or None}
    # safety against Re by parts
    parts = []
    re_min = math.inf
    _, pmax, cnt = c.part_extrema(vm)
    for pi, pn in enumerate(m.part_names):
        if not cnt[pi]:
            continue
        vmax = float(pmax[pi])
        try:
            mat = study.material_of(pn)
        except KeyError:
            mat = None
        row = {"part": pn, "max_mises_MPa": vmax, "material": mat.name if mat else None}
        if mat is not None and getattr(mat, "is_ortho", False):
            row["note"] = "orthotropic material – safety factor to Re (von Mises) is not valid, evaluate stress in the material axes"
        elif mat is not None and mat.Re:
            row["Re_MPa"] = mat.Re
            row["safety_Re"] = mat.Re / max(vmax, 1e-12)
            re_min = min(re_min, row["safety_Re"])
        parts.append(row)
    d["parts"] = parts
    if math.isfinite(re_min):
        d["min_safety_Re"] = re_min
    try:
        from ..assessment import assessment_summary
        ass = assessment_summary(study, m, st)
        if ass:
            d["assessment"] = ass
    except Exception as e:  # noqa: BLE001 – the summary must not fall because of the assessment, but the error is shown
        d["assessment_error"] = str(e)
    if with_hotspots:
        try:
            hs = hotspots(sess, i, 5)["hotspots"]
            d["hotspots_top"] = [{"xyz_mm": h["xyz_mm"], "mises_MPa": h["mises_MPa"], "classification": h["classification"],
                                  **({"stress_error_pct": h["stress_error_pct"]} if "stress_error_pct" in h else {})}
                                 for h in hs]
            errs = [h["stress_error_pct"] for h in hs if "stress_error_pct" in h]
            if not errs and hs:
                _res = sess.require_results()
                d["stress_error_note"] = no_error_estimate_note(_res, _res.steps[i].name if 0 <= i < len(_res.steps) else None)
            if errs:
                d["max_stress_error_hotspots_pct"] = max(errs)
                d["stress_error_note"] = ("ccx stress error estimate (STR, principal stress gradient in neighboring elements) at the maxima – informative; > 10–20 % = refine the mesh there")
            ns = [h for h in hs if h["classification"] != "singularity"]
            if hs and hs[0]["classification"] != "probably real concentration":
                d["note"] = (f"The von Mises maximum is probably a singularity/edge effect ({hs[0]['classification']}) – see fea_results_hotspots.")
                if ns:
                    d["max_mises_without_singularities_MPa"] = ns[0]["mises_MPa"]
        except Exception as e:  # noqa: BLE001 – the summary must not fall because of the heuristic
            d["hotspots_error"] = str(e)
    if st.printing:
        d["printing"] = st.printing
    if getattr(st, "frame", None):
        d["frame"] = frame_step_summary(st)
    return d


def thermal_summary(sess: Session, k: int) -> dict:
    """Summary of the thermal state: T min/max (place, part), max. heat flux, temperatures by parts, fluxes of the boundary
    conditions [W] (+ = heat into the body) and the heat balance; for a transient state the frames in time.
    """
    res = sess.require_results()
    tr = res.thermal[k]
    m = res.mesh
    c = mesh_cache(res)
    T, q = tr.T, tr.flux_mag
    imax, imin, iq = int(np.argmax(T)), int(np.argmin(T)), int(np.argmax(q))
    d = {"thermal_case": tr.name, "index": k, "transient": tr.transient,
         "T_max_C": {"value": float(T[imax]), "at_mm": m.nodes[imax], "part": c.node_part(imax)},
         "T_min_C": {"value": float(T[imin]), "at_mm": m.nodes[imin], "part": c.node_part(imin)},
         "max_heat_flux_W_m2": {"value": float(q[iq]), "at_mm": m.nodes[iq], "part": c.node_part(iq)}}
    pmin, pmax, cnt = c.part_extrema(T)
    d["parts"] = [{"part": pn, "T_min_C": float(pmin[pi]), "T_max_C": float(pmax[pi])}
                  for pi, pn in enumerate(m.part_names) if cnt[pi]]
    flows = [{"kind": h["kind"], "label": THERMAL_LABELS.get(h["kind"], h["kind"]), "name": h["name"],
              "Q_W": float(h["Q_W"])} for h in tr.heat]
    d["heat_flows_W"] = flows
    d["heat_sign"] = "Q_W > 0 = heat into the body, < 0 = out of the body"
    bal = dict(tr.balance or {})
    if "imbalance" in bal:
        from .. import config
        bal["ok"] = bal["imbalance"] <= config.EQUILIBRIUM_TOL
    d["balance"] = bal
    if tr.transient:
        d["times_s"] = list(tr.times)
        if tr.frames is not None:
            d["frames"] = [{"t_s": float(t), "T_min_C": float(f.min()), "T_max_C": float(f.max())}
                           for t, f in zip(tr.times, tr.frames)]
        d["note"] = "Temperature and flux are valid at the end of the case; temperature at another time is given by time_s (probe, image)."
    return d


def results_summary(sess: Session, step=None, thermal_case=None) -> dict:
    """Summary of the results: the mechanical states (step = index/name, None = all) and the thermal states (thermal_case,
    None = all). A study with heat only has steps = [].
    """
    res = sess.require_results()
    if not res.steps:
        idx = []
    else:
        idx = range(len(res.steps)) if step is None else [sess.step_index(step)]
    d = {"study": sess.study.name, "backend": res.info.get("backend"),
         "mesh": {"nodes": len(res.mesh.nodes), "elements": len(res.mesh.tets), "dof": 3 * len(res.mesh.nodes)},
         "units": "mm, N, MPa" + (", °C, W, W/m²" if res.thermal else ""),
         "steps": [step_summary(sess, i) for i in idx], "warnings": res.warnings}
    if res.info.get("solver"):                     # why this solver, precision, memory, time per phase (fea_core.solver_protocol)
        from ..solver_protocol import brief
        d["solver"] = brief(res.info["solver"])
    if res.info.get("ccx_solvers"):                # the equation solver of every CalculiX job that ran (static, heat, modal …) – next to the protocol above
        d["ccx_solvers"] = dict(res.info["ccx_solvers"])
    if res.thermal:
        tidx = range(len(res.thermal)) if thermal_case is None else [sess.thermal_index(thermal_case)]
        d["thermal"] = [thermal_summary(sess, k) for k in tidx]
    if getattr(res, "modes", None):
        d["modes"] = [modes_summary(sess, j, brief=True) for j in range(len(res.modes))]
    if getattr(res, "twod", None) is not None:
        t = res.twod.summary()
        d["twod"] = {"kind": t["kind_label"], "nodes": t["nodes"], "elements": t["elements"],
                     "steps": [{k: s[k] for k in ("name", "max_mises", "max_disp", "reactions", "imbalance")}
                               for s in t["steps"]], "detail": "fea_twod_results"}
        d["mesh"]["note"] = "2D mesh (tri6) – see twod"
    if getattr(res, "shell", None) is not None:
        t = res.shell.summary()
        d["shell"] = {"elements": t["elements"], "nodes": t["nodes"], "parts": t["parts"],
                      "steps": [{k: s[k] for k in ("name", "u_max", "mises_max", "membrane_max", "bending_max",
                                                   "imbalance")} for s in t["steps"]], "detail": "fea_shell_results"}
        d["mesh"]["note"] = "S6 shell mesh – see shell"
    return d


def modes_summary(sess: Session, j: int, brief: bool = False) -> dict:
    """Natural frequencies (frequencies, shares of the effective mass by the directions x, y, z and cumulatively) or buckling
    (factors λ, critical load = λ × the load of the state) of the set Results.modes[j]. brief = only numbers (summary).
    """
    res = sess.require_results()
    ms = res.modes[j]
    d = {"name": ms.name, "kind": ms.kind, "index": j}
    vals = [float(v) for v in ms.values]
    if ms.kind == "modal":
        d["engine"] = getattr(ms, "engine", "ccx")
        if getattr(ms, "check", None) and ms.check.get("n"):
            c = ms.check
            d["check_vs_ccx"] = {"modes_compared": c["n"], "max_frequency_difference": c["max_freq_rel"], "min_mac_of_a_mode_with_its_cluster": c["min_subspace"],
                                 "max_effective_mass_difference": c["max_eff_mass_rel"]}
        d["frequencies_Hz"] = vals
        tot = np.asarray(ms.total_mass or [], float)
        em = np.asarray(ms.eff_mass or [], float)
        frac = None
        if em.size and tot.size >= 3:
            frac = em[:, :3] / np.where(tot[:3] > 0, tot[:3], 1.0)
            d["cumulative_mass_fraction_xyz"] = frac.sum(axis=0)
            d["total_mass_kg"] = float(tot[0] * 1000.0)
        fmax = max([abs(v) for v in vals], default=0.0)
        rigid = [i + 1 for i, v in enumerate(vals) if abs(v) <= max(1e-3 * fmax, 1e-6)]
        if rigid:
            d["rigid_body_modes"] = rigid
        if brief:
            return d
        rows = []
        cum = np.zeros(3)
        for i, f in enumerate(vals):
            r = {"mode": i + 1, "frequency_Hz": f, "period_s": (1.0 / f) if f > 0 else None}
            if frac is not None and i < len(frac):
                cum = cum + frac[i]
                r["eff_mass_fraction_xyz"] = frac[i]
                r["cumulative_xyz"] = cum.copy()
            if ms.participation and i < len(ms.participation):
                r["participation_xyz"] = list(ms.participation[i][:3])
            rows.append(r)
        d["modes"] = rows
        notes = ["Mode shapes are normalized to max |u| = 1 (displacement in a mode has no physical magnitude) – image fea_results_render_mode(mode=…)",
                 "Effective mass fraction = relative to the total effective mass in ccx (mass of free nodes; constrained nodes are not counted)"]
        if frac is not None and np.min(frac.sum(axis=0)) < 0.9:
            notes.append("Sum of effective mass fractions in some direction < 90 % – for dynamic response (seismic, spectrum) increase the number of modes (fea_modal_set modes)")
        if rigid:
            notes.append(f"Modes {', '.join(map(str, rigid))} have ≈ 0 Hz – rigid-body motions (free body / insufficient supports)")
        ei = getattr(ms, "engine_info", None) or {}
        if getattr(ms, "engine", "ccx") == "direct":
            d["engine_info"] = {k: ei.get(k) for k in ("library", "eigensolver", "lanczos_solves", "factorizations", "max_residual", "residual", "sturm_checks", "bounds", "window_raised", "retries", "takeover", "times", "memory") if k in ei}
            notes.append("Computed by the direct path (own assembly, rslab, ARPACK): the number of modes below the highest one is proved by the Sturm count; rigid-body modes of a free body are returned (0 Hz) with their shapes")
        if ei.get("reason"):
            d["engine_reason"] = ei["reason"]
        if ei.get("takeover"):
            d["engine_takeover"] = ei["takeover"]
        d["notes"] = notes
        return d
    d["case"] = ms.case
    if ms.kind == "nlbuckle":                         # nonlinear buckling (fea_core.imperfect): frames, not eigenvalues
        r = ((res.info or {}).get("nlbuckling") or {}).get("result") or {}
        d.update(frame_load_factors=vals, ultimate_load_factor=r.get("ultimate_factor"), status=r.get("status"),
                 note="frame 1 = imperfection (mm), further frames = displacement of the increments (mm); details fea_results_nlbuckling")
        return d
    d["buckling_factors"] = vals
    st = next((s for s in res.steps if s.name == ms.case), None)
    applied = None
    if st is not None and st.equilibrium:
        applied = st.equilibrium.get("applied_exact")
    if applied is not None:
        F = float(np.linalg.norm(applied))
        d["applied_load_N"] = applied
        if not brief:
            d["critical_load_N"] = [v * F for v in vals]
    if brief:
        return d
    lc = next((x for x in sess.study.load_cases if x.name == ms.case), None) if sess.study else None
    if lc is not None:
        from ..loads_text import load_text
        d["case_loads"] = [load_text(l, sess.study) for l in lc.loads]
    notes = ["λ = load case multiplier at loss of stability: critical load = λ × case load (all loads in the case are scaled together)",
             "LINEAR BUCKLING without imperfections, plasticity and nonlinearities – actual capacity is usually considerably lower (by tens of % for slender members and shells); for design use the code procedure (EN 1993-1-1, 6.3 – fea_ec3_member_check) or a nonlinear analysis with imperfections"]
    if any(v < 0 for v in vals):
        notes.append("Negative λ = loss of stability under REVERSED load (e.g. tension ↔ compression)")
    if vals and 0 < min(abs(v) for v in vals) < 1:
        notes.append("|λ| < 1 – loss of stability already at the applied load!")
    d["notes"] = notes
    return d


def summary_text(sess: Session) -> str:
    """Text summary of the last results (source fea://results/summary)."""
    if sess.study is None:
        return "No study is open."
    if sess.results is None:
        return f"Study '{sess.study.name}': results are not available (fea_solve_run)."
    s = results_summary(sess)
    lines = [f"Study '{s['study']}', solver {s['backend']}, mesh {s['mesh']['nodes']} nodes / {s['mesh']['elements']} elements ({s['mesh']['dof']} DOF). Units {s['units']}."]
    for th in s.get("thermal", []):
        kind = "transient" if th["transient"] else "steady-state"
        lines.append(f"\nThermal case '{th['thermal_case']}' ({kind}):")
        lines.append(f"  T max {th['T_max_C']['value']:.4g} °C at {np.round(th['T_max_C']['at_mm'], 3).tolist()} (part {th['T_max_C']['part']}), T min {th['T_min_C']['value']:.4g} °C")
        lines.append(f"  max. heat flux {th['max_heat_flux_W_m2']['value']:.4g} W/m²")
        for f in th["heat_flows_W"]:
            lines.append(f"  heat '{f['name']}' ({f['label']}): {f['Q_W']:+.4g} W")
        bal = th["balance"]
        if "imbalance" in bal:
            lines.append(f"  balance: input {bal.get('in_W', 0):.4g} W, output {bal.get('out_W', 0):.4g} W, imbalance {bal['imbalance']:.2e}")
        elif bal.get("note"):
            lines.append("  " + bal["note"])
    for ms in s.get("modes", []):
        if ms["kind"] == "modal":
            fr = ", ".join(f"{v:.4g}" for v in ms["frequencies_Hz"][:10])
            lines.append(f"\nNatural frequencies [Hz]: {fr}" + (" …" if len(ms["frequencies_Hz"]) > 10 else ""))
        elif ms["kind"] == "nlbuckle":
            lines.append(f"\nBuckling (nonlinear, state '{ms['case']}'): ultimate λ_u = {ms.get('ultimate_load_factor')}"
                         f" ({ms.get('status')})")
        else:
            lines.append(f"\nBuckling (linear, state '{ms['case']}'): λ = "
                         + ", ".join(f"{v:.4g}" for v in ms["buckling_factors"]))
    for st in s["steps"]:
        lines.append(f"\nState '{st['step']}':")
        fr = st.get("frame")
        if fr:
            lines.append(f"  members: max. node displacement {fr.get('max_node_displacement_mm', 0):.4g} mm"
                         + (f", max. σ_eq {fr['max_sigma_eq_MPa']['value']:.4g} MPa (member {fr['max_sigma_eq_MPa']['member']})"
                            if fr.get("max_sigma_eq_MPa") else ""))
            for k, r in fr.get("reactions", {}).items():
                lines.append(f"  node reaction '{k}': F = {np.round(np.asarray(r[:3], float), 4).tolist()} N, "
                             f"M = {np.round(np.asarray(r[3:], float), 1).tolist()} N·mm")
        if "max_von_mises_MPa" not in st:
            if "equilibrium" in st:
                lines.append(f"  equilibrium: imbalance {st['equilibrium']['imbalance_relative']:.2e}")
            continue
        lines.append(f"  max. displacement {st['max_displacement_mm']['value']:.4g} mm at {np.round(st['max_displacement_mm']['at_mm'], 3).tolist()}")
        lines.append(f"  max. von Mises {st['max_von_mises_MPa']['value']:.4g} MPa at {np.round(st['max_von_mises_MPa']['at_mm'], 3).tolist()} (part {st['max_von_mises_MPa']['part']})")
        lines.append(f"  σ1 max {st['max_principal_s1_MPa']['value']:.4g} MPa, σ3 min {st['min_principal_s3_MPa']['value']:.4g} MPa")
        if "temperature_C" in st:
            t = st["temperature_C"]
            lines.append(f"  temperature {t['min']:.4g} … {t['max']:.4g} °C (reference {t['reference']:g} °C)"
                         + (f": {t['source']}" if t.get("source") else ""))
        for k, r in st["reactions"].items():
            f = np.round(np.asarray(r["force_N"], float), 4).tolist()
            lines.append(f"  reaction '{k}': {f} N (|R| = {r['magnitude_N']:.4g} N)")
        if "equilibrium" in st:
            lines.append(f"  equilibrium: imbalance {st['equilibrium']['imbalance_relative']:.2e}")
        if "min_safety_Re" in st:
            lines.append(f"  min. safety factor to Re {st['min_safety_Re']:.3g}")
        if st.get("note"):
            lines.append("  " + st["note"])
    for w in s["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)

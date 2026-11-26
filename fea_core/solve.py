"""The whole chain without GUI: STEP → Gmsh → .inp → ccx → results."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .ccx.frd import read_frd, dat_pattern, header_lines, line_end, odd_line_breaks
from .ccx.inp import write_inp
from .ccx.runner import run_ccx, CcxError
from .loads import load_resultant, nodal_loads, nodal_cache
from .mesh import generate_mesh, Mesh
from .results import Results, StepResult
from .study import Study, Fixed, Displacement, LoadCase, CylindricalSupport, BearingLoad, RemoteSupport


class StudyError(RuntimeError):
    pass


def check_study(study: Study, part_names: list[str]) -> tuple[list[str], list[str]]:
    """Readiness check. Returns (errors, warnings)."""
    errors, warnings = [], []
    thermal_only = bool(study.thermal.cases) and not study.load_cases
    modal_only = (study.modal.enabled or study.dynamics.enabled) and not study.load_cases
    seq = study.analysis.nonlinear and not study.analysis.branches and bool(study.analysis.sequence)
    if not study.load_cases and not seq and not study.thermal.cases and not study.modal.enabled \
            and not study.dynamics.enabled:
        errors.append("The study has no load case or thermal case (nor natural frequency analysis)")
    frame_held = study.frame.active and bool(study.frame.supports)
    if not study.constraints and not thermal_only and not modal_only and not frame_held:
        errors.append("The study has no support")
    excl = study.excluded_parts()
    for imp in study.imprints:                       # Study.imprints = [Imprint] (Study.from_dict converts the JSON)
        if imp.part and imp.part not in part_names:
            errors.append(f"Imprint '{imp.name}': part '{imp.part}' is not in the geometry")
    if getattr(study.mesh, "part_sizes", None):
        for p in study.mesh.part_sizes:
            if p not in part_names and p not in excl:
                warnings.append(f"Mesh size for part '{p}': part is not in the geometry")
    for p in part_names:
        if p in excl:
            continue
        try:
            study.material_of(p)
        except KeyError:
            errors.append(f"Part '{p}' has no material assigned")
    e2, w2 = check_joints(study, part_names)
    errors += e2
    warnings += w2
    e3, w3 = check_printing(study, [p for p in part_names if p not in excl])
    errors += e3
    warnings += w3
    e4, w4 = check_cylindrical(study)
    errors += e4
    warnings += w4
    e6, w6 = check_remote(study)
    errors += e6
    warnings += w6
    e7, w7 = check_loads(study, part_names)
    errors += e7
    warnings += w7
    from .eigen import check as check_eigen
    e8, w8 = check_eigen(study, part_names)
    errors += e8
    warnings += w8
    from .imperfect.run import check as check_nlbuckling       # nonlinear buckling with imperfection (F9 C2)
    e8, w8 = check_nlbuckling(study, part_names)
    errors += e8
    warnings += w8
    from .beams.run import check as check_frame
    e11, w11 = check_frame(study)
    errors += e11
    warnings += w11
    from .springs import check as check_springs
    e9, w9 = check_springs(study)
    errors += e9
    warnings += w9
    from .ortho import check as check_ortho
    e10, w10 = check_ortho(study, part_names)
    errors += e10
    warnings += w10
    from .dynamics.mass import check as check_mass
    e12, w12 = check_mass(study, part_names)
    errors += e12
    warnings += w12
    from .dynamics.check import check as check_dynamics
    e13, w13 = check_dynamics(study, part_names)
    errors += e13
    warnings += w13
    from .contacts.check import check as check_contacts
    e14, w14 = check_contacts(study, part_names)
    errors += e14
    warnings += w14
    from .plasticity import check as check_plasticity
    e15, w15 = check_plasticity(study, [p for p in part_names if p not in excl])
    errors += e15
    warnings += w15
    from .cyclic import check as check_cyclic
    e16, w16 = check_cyclic(study, part_names)
    errors += e16
    warnings += w16
    from .solvers import check_backend, solver_warnings      # manually chosen solver: installed? can it do the study? (before meshing)
    errors += check_backend(study)
    warnings += solver_warnings(study)               # an explicit equation solver (PaStiX) where SPOOLES is the better one
    warnings += list(getattr(study, "load_warnings", []))
    from .thermal.check import check as check_thermal
    e5, w5, used = check_thermal(study, part_names)
    errors += e5
    warnings += w5
    used |= {c.selection for c in study.constraints}
    used |= {c.selection for c in study.frame.connections}     # connection of beams to a body
    used |= {m.selection for m in getattr(study, "masses", [])}  # added mass
    for lc in study.load_cases:
        if not lc.loads:
            warnings.append(f"Load case '{lc.name}' is empty")
        for l in lc.loads:
            if hasattr(l, "selection"):
                used.add(l.selection)
    imprint_names = {imp.name for imp in study.imprints}
    for s in used:
        if s in imprint_names:
            continue
        if s not in study.selections:
            errors.append(f"Selection '{s}' does not exist")
        else:
            sel = study.selections[s]
            for f in [*sel.faces, *sel.edges, *sel.points]:
                if f.part not in part_names:
                    errors.append(f"Selection '{s}': part '{f.part}' is not in the geometry")
                elif f.part in excl:
                    errors.append(f"Selection '{s}': part '{f.part}' is " + ("excluded from analysis" if f.part in study.ignored_parts
                                                                        else "replaced by a bolt") + " (not meshed)")
    return errors, warnings


def check_loads(study: Study, part_names: list[str]) -> tuple[list[str], list[str]]:
    """Kind of selection (faces / edges / points) by the constraint and the load, coordinate systems, variable pressure, rotation."""
    from . import csys
    from .load_kinds import load_handler
    from .study import ENTITY_LABEL
    errors, warnings = [], []
    for cs in study.coordinate_systems.values():
        errors += csys.check(cs)

    def entity(name):
        sel = study.selections.get(name)
        return sel.entity if sel is not None else "faces"
    for c in study.constraints:
        e = entity(c.selection)
        if e in ("edges", "points", "mixed"):
            errors.append(f"Support '{c.selection}': selection contains {ENTITY_LABEL[e]} – supports are currently faces only")
    for lc in study.load_cases:
        for l in lc.loads:
            h = load_handler(l)                    # per kind: fea_core.load_kinds
            name = getattr(l, "selection", None)
            what = f"Case '{lc.name}', load '{name or l.kind}'"
            cname = getattr(l, "csys", None)
            if cname:
                cs = study.coordinate_systems.get(cname)
                if cs is None:
                    errors.append(f"{what}: coordinate system '{cname}' does not exist")
                elif cs.kind == "cylindrical" and (h is None or h.csys != "cylindrical"):
                    errors.append(f"{what}: cylindrical coordinate system is only available for force (other loads use Cartesian)")
            if name is not None:
                e = entity(name)
                if h is not None and h.entity_check is not None:
                    er, wr = h.entity_check(what, e)
                    errors += er
                    warnings += wr
                elif e in ("edges", "points", "mixed") and (h is None or e not in h.entities):
                    errors.append(f"{what}: selection contains {ENTITY_LABEL[e]} – this load type can only be applied to faces")
            if h is not None and h.check is not None:
                er, wr = h.check(study, lc, l, part_names, what)
                errors += er
                warnings += wr
    return errors, warnings


def check_cylindrical(study: Study) -> tuple[list[str], list[str]]:
    """Cylindrical support and bearing load: the selection must be a cylindrical face (the axis is found from the mesh)."""
    errors, warnings = [], []

    def cyl_only(sel: str, what: str):
        s = study.selections.get(sel)
        if s is not None and any(f.type != "cylinder" for f in s.faces):
            errors.append(f"{what} '{sel}': selected faces must be cylindrical (hole / pin)")
    for c in study.constraints:
        if isinstance(c, CylindricalSupport):
            cyl_only(c.selection, "Cylindrical support")
            if not (c.radial or c.tangential or c.axial):
                warnings.append(f"Cylindrical support '{c.selection}' has no constrained component")
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, BearingLoad):
                cyl_only(l.selection, "Bearing load")
    return errors, warnings


def check_remote(study: Study) -> tuple[list[str], list[str]]:
    """Remote constraint: a point (3 coordinates), 6 components, kind of constraint."""
    errors, warnings = [], []
    for c in study.constraints:
        if not isinstance(c, RemoteSupport):
            continue
        what = f"Remote support '{c.selection}'"
        try:
            ok = len(c.point) == 3 and all(np.isfinite(float(v)) for v in c.point)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            errors.append(f"{what}: point must have 3 coordinates [mm]")
        if len(c.dofs) != 6:
            errors.append(f"{what}: components must be 6 values (ux uy uz rx ry rz)")
        elif not any(c.dofs):
            warnings.append(f"{what} has no constrained component")
        if c.coupling not in ("rigid", "distributing"):
            errors.append(f"{what}: unknown coupling type '{c.coupling}' (rigid | distributing)")
    return errors, warnings


def check_joints(study: Study, part_names: list[str]) -> tuple[list[str], list[str]]:
    """Check of bolts and welds (the parts exist, the tables know the size and class)."""
    from .joints import tables
    errors, warnings = [], []
    names = set(part_names)
    excl = study.excluded_parts()
    for b in study.active_bolts():
        n_err = len(errors)
        for pn in {b.head_part, b.nut_part, *[pl["part"] for pl in b.plates]}:
            if pn not in names:
                errors.append(f"{b.name}: part '{pn}' is not in the geometry")
            elif pn in excl:
                errors.append(f"{b.name}: clamped part '{pn}' is also a bolt component")
        for pn in b.cad_parts:
            if pn not in names:
                warnings.append(f"{b.name}: bolt part '{pn}' is not in the geometry")
        try:
            tables.thread(b.size, b.pitch)
        except KeyError as e:
            errors.append(f"{b.name}: {e}")
        try:
            tables.strength_class(b.strength, b.size)
        except KeyError as e:
            errors.append(f"{b.name}: {e}")
        if b.connection == "nut" and b.nut_dw <= 0:
            errors.append(f"{b.name}: nut bearing diameter is missing")
        if len(errors) == n_err:             # the settlement F_Z against the given preload (VDI 2230): the bolt must not end in compression
            try:
                from .joints.fem import bolt_basics
                from .joints.vdi2230 import consumed_note
                note = consumed_note(b.name, bolt_basics(study, b))
            except Exception:                # noqa: BLE001 – a missing material / part is reported elsewhere
                note = ""
            if note:
                warnings.append(note)
    for w in study.active_welds():
        if w.part not in names:
            errors.append(f"{w.name}: part '{w.part}' is not in the geometry")
        elif w.part in excl:
            errors.append(f"{w.name}: part '{w.part}' is a bolt component")
    seen = {}
    for w in study.active_welds():
        if w.part in seen:
            errors.append(f"Part '{w.part}' is marked as a weld twice ({seen[w.part]}, {w.name})")
        seen[w.part] = w.name
    return errors, warnings


def check_printing(study: Study, part_names: list[str]) -> tuple[list[str], list[str]]:
    """Check of the 3D printing settings (printed parts = parts with a material with E_z)."""
    from .printing.infill import PATTERNS
    from .printing.region import print_parts
    errors, warnings = [], []
    printed = print_parts(study, part_names)
    for p in printed:
        sp = study.printing.spec(p)
        if sp.enabled and sp.pattern not in PATTERNS:
            errors.append(f"3D printing '{p}': unknown infill type '{sp.pattern}'")
        if sp.enabled and not (0.0 <= sp.infill <= 1.0):
            errors.append(f"3D printing '{p}': infill density must be 0–100 %")
        if min(sp.layer_height, sp.line_width) <= 0 or np.linalg.norm(sp.build_dir) == 0:
            errors.append(f"3D printing '{p}': invalid layer height, line width, or build direction")
    for p in study.printing.parts:
        if p in part_names and p not in printed:
            warnings.append(f"Part '{p}' has print settings, but its material is not a printed material (E_z missing) – treated as isotropic")
    for p in part_names:            # plastic "for printing" without a printing model (e.g. "PLA print XY") – the printing results will not be available
        if p in printed or p in study.printing.parts:
            continue
        try:
            mname = study.material_of(p).name
        except KeyError:
            continue
        if re.search(r"\b(PLA|PETG|ABS|ASA|TPU|PEEK|PEI|PA\d*|PC)\b|tisk", mname, re.IGNORECASE):
            warnings.append(f"Part '{p}' has material '{mname}' without print properties – it is computed as a solid isotropic material and there will be no 3D print results (Tsai-Wu/Hill safety, perimeters, infill). For the print model, choose a printed material in the 3D print step (e.g. “Tisk PLA Bambu Basic”).")
    from .printing.check import check_settings
    e, w = check_settings(study, printed)
    errors += e
    warnings += w
    from .printing.creep import check as check_creep
    e, w = check_creep(study, printed)
    errors += e
    warnings += w
    from .printing.infill import FULL
    if any(study.printing.spec(p).enabled and study.printing.spec(p).infill < FULL for p in printed):
        warnings.append("3D printing: infill properties are indicative – calibrate by testing for design (ISO 527).")
    return errors, warnings


def constraint_masks(study: Study, mesh: Mesh, own: bool = False) -> dict[str, np.ndarray]:
    """Selection of a constraint -> (N, 3) bool: which DOF of the nodes the constraint holds.
    own = the own assembly: the nodes of the face of a rigid remote support hold the reaction (ccx measures it at the reference node of its rigid body)."""
    out: dict[str, np.ndarray] = {}
    for c in study.constraints:
        msk = out.setdefault(c.selection, np.zeros((len(mesh.nodes), 3), bool))
        idx = mesh.surface_nodes(c.selection)
        if isinstance(c, Fixed):
            msk[idx, :] = True
        elif isinstance(c, Displacement):
            for d, v in enumerate((c.ux, c.uy, c.uz)):
                if v is not None:
                    msk[idx, d] = True
        elif isinstance(c, CylindricalSupport) and (c.radial or c.tangential or c.axial):
            msk[idx, :] = True          # reactions in the local directions – projection see make_step
        elif isinstance(c, RemoteSupport) and c.coupling == "rigid" and own:
            msk[idx, :] = True          # own assembly: reaction = Σ (RF − F) over the nodes of the rigid face
        elif isinstance(c, RemoteSupport) and c.coupling != "rigid" and any(c.dofs):
            # distributing: reaction = RF − F in the nodes of the constraint equations (nodes with weight); a rigid coupling the nodes of the mesh
            # does not hold them – its reaction is RF REF/ROT of the node (make_step, remote=)
            from .remote import rbe3
            msk[rbe3(mesh, c.selection, c.point)["idx"], :] = True
    return out


def equilibrium(x: np.ndarray, F: np.ndarray, R: np.ndarray, Fr: np.ndarray, Mr: np.ndarray,
                ref_force: float = 0.0, Rp: np.ndarray | None = None, Mp: np.ndarray | None = None) -> dict:
    """Balance of forces and moments (x relative to the reference point).

    ref_force = scale for states without external load (bolt preload – internal forces).
    Rp, Mp = reactions outside the nodes of the mesh (rigid remote constraint: RF REF/ROT of the node measured by ccx) – force and moment
    about the reference point; they belong to the reactions and to the check.
    The result also records `force_scale` / `moment_scale` (the divisors of the relative imbalance), so that the model checks recompute the balance on the same scale.
    """
    Rp = np.zeros(3) if Rp is None else np.asarray(Rp, float)
    Mp = np.zeros(3) if Mp is None else np.asarray(Mp, float)
    fres = F.sum(axis=0) + R.sum(axis=0) + Fr + Rp
    mres = np.cross(x, F).sum(axis=0) + np.cross(x, R).sum(axis=0) + Mr + Mp
    size = float(np.linalg.norm(x.max(axis=0) - x.min(axis=0))) if len(x) else 1.0
    f_scale = max(np.linalg.norm(F, axis=1).sum(), np.linalg.norm(R, axis=1).sum() + np.linalg.norm(Rp),
                  np.linalg.norm(Fr), ref_force, 1e-30)
    m_scale = max(np.linalg.norm(np.cross(x, F), axis=1).sum(),
                  np.linalg.norm(np.cross(x, R), axis=1).sum() + np.linalg.norm(Mp),
                  np.linalg.norm(Mr), ref_force * size, 1e-30)
    imb = max(np.linalg.norm(fres) / f_scale, np.linalg.norm(mres) / m_scale)
    return {"force_residual": fres.tolist(), "moment_residual": mres.tolist(),
            "reaction_total": (R.sum(axis=0) + Rp).tolist(),
            "reaction_moment_total": (np.cross(x, R).sum(axis=0) + Mp).tolist(),   # to the reference point
            "force_scale": float(f_scale), "moment_scale": float(m_scale),         # what the residuals were divided by (the model checks recompute the balance on the same scale)
            "imbalance": float(imb)}


def _nodal(frd, name: str, step: int, n: int, ncomp: int) -> np.ndarray:
    out = np.zeros((n, ncomp))
    b = frd.get(name, step)
    mask = b.node_ids <= n
    out[b.node_ids[mask] - 1] = b.values[mask, :ncomp]
    return out


def _read_printed(dat: Path, setname: str) -> list[dict[int, np.ndarray]]:
    """Values from *NODE PRINT (not sums) for a set, by steps.

    The headings are searched in the whole file at once (not by a regular expression on every line – the .dat with joints has
    hundreds of thousands of *EL PRINT lines); the lines of the block are read as before.
    """
    buf = Path(dat).read_bytes()
    n = len(buf)
    heads = [h for h in header_lines(buf, dat_pattern("for set {} and time", setname))
             if b"total" not in buf[h:line_end(buf, h)]]
    starts = set(heads)
    out = []
    for h in heads:
        cur: dict[int, np.ndarray] = {}
        out.append(cur)
        pos = line_end(buf, h) + 1
        if odd_line_breaks(buf[h:pos]):
            return _read_printed_lines(dat, setname)
        while pos < n and pos not in starts:             # another header = a new block
            e = line_end(buf, pos)
            if odd_line_breaks(buf[pos:e + 1]):
                return _read_printed_lines(dat, setname)
            parts = buf[pos:e].decode("latin-1").split()
            if not parts:
                if cur:
                    break
            else:
                try:
                    cur[int(parts[0])] = np.array([float(v) for v in parts[1:]])
                except ValueError:
                    break
            pos = e + 1
    return out


def _read_printed_lines(dat: Path, setname: str) -> list[dict[int, np.ndarray]]:
    """The original line-by-line reading (backup for unusual line breaks, reference of the tests)."""
    out, cur = [], None
    for line in dat.read_text(encoding="latin-1").splitlines():
        if re.search(rf"for set {setname} and time", line) and "total" not in line:
            cur = {}
            out.append(cur)
            continue
        if cur is not None:
            parts = line.split()
            if not parts:
                if cur:
                    cur = None
                continue
            try:
                cur[int(parts[0])] = np.array([float(v) for v in parts[1:]])
            except ValueError:
                cur = None
    return out


def mesh_study(study: Study, parts, workdir: Path, cancel=None) -> tuple[Mesh, list[str]]:
    """Mesh of the study: without bolt parts, with imprints of the bearing faces and refinement at the joints (also for the GUI)."""
    warnings: list[str] = []
    sels = {k: [f.to_dict() for f in s.faces] for k, s in study.selections.items() if s.faces or s.entity == "empty"}
    edges = {k: [e.to_dict() for e in s.edges] for k, s in study.selections.items() if s.edges}
    points = {k: [p.to_dict() for p in s.points] for k, s in study.selections.items() if s.points}
    excl = study.excluded_parts()                 # left-out parts (also a part replaced by a beam) are not meshed even without bolts
    mparts, imprints, vsizes = [p for p in parts if p.name not in excl], [], []
    for imp in study.imprints:
        imp_dict = imp.to_mesh_dict()
        imprints.append(imp_dict)
        if imp_dict["name"] not in sels:
            sels[imp_dict["name"]] = []
    if getattr(study.mesh, "part_sizes", None):
        for pn, ps in study.mesh.part_sizes.items():
            if pn in [p.name for p in mparts]:
                vsizes.append({"part": pn, "size": float(ps)})
    if study.active_bolts() or study.active_welds():
        from .joints.fem import mesh_inputs, finish_mesh
        mparts, extra, imp_b, vs_b, jw = mesh_inputs(study, parts)
        imprints, vsizes = imprints + imp_b, vsizes + vs_b       # the imprints and the sizes per part of the study remain
        sels.update(extra)
        warnings += jw
    msettings = asdict(study.mesh)
    if getattr(study, "cyclic", None) is not None and study.cyclic.enabled:   # periodic mesh of the cut faces
        c = study.cyclic
        msettings["periodic"] = {"left": c.left, "right": c.right, "axis_point": list(c.axis_point),
                                 "axis_dir": list(c.axis_dir), "sectors": int(c.sectors)}
    if study.active_contacts():                   # parts of the contact pairs are meshed separately (without *TIE between them)
        from .contacts.detect import contact_parts
        cp = contact_parts(study)
        msettings["separate"] = sorted(set(msettings.get("separate") or []) | {p for p in cp
                                                                                 if p in {q.name for q in mparts}})
    mesh = generate_mesh(mparts, sels, msettings, Path(workdir), imprints=imprints, volume_sizes=vsizes,
                         cancel=cancel, edges=edges, points=points)
    rep = mesh.report
    for u in rep.get("unmatched", []):
        what = {"edge": "edge not found", "point": "point is not on the surface"}.get(u.get("kind"), "face not found")
        warnings.append(f"Selection '{u['selection']}': {what} (part '{u['selector']['part']}')")
    warnings.extend(rep.get("warnings", []))
    if any(u for u in rep.get("unmatched", [])):
        raise StudyError("Some selected faces, edges or points were not found:\n  " + "\n  ".join(warnings))
    if study.active_welds():
        from .joints.fem import open_weld_roots
        warnings += open_weld_roots(study, mesh)
    if study.active_bolts():
        errs, jw = finish_mesh(study, mesh)
        warnings += jw
        if errs:
            raise StudyError("Bolts cannot be connected to the mesh:\n  " + "\n  ".join(errs))
    if study.mesh.separate or (study.active_contacts() and study.contact_split() == "separate"):
        from .ties import detect as detect_ties   # separately meshed parts → contact faces *TIE
        warnings += detect_ties(study, mesh)
    if study.active_contacts():                   # contact faces of the pairs (fea_core.contacts)
        from .contacts.detect import mesh_pairs, split_interface, pair_parts
        if study.contact_split() == "conformal_split":
            for c in study.active_contacts():
                pa, pb = pair_parts(study, c)
                for a in pa:
                    for b in pb:
                        if a in mesh.part_names and b in mesh.part_names and a != b:
                            split_interface(mesh, a, b)
        warnings += mesh_pairs(study, mesh)
        from .contacts.check import mesh_errors
        errs = mesh_errors(study, mesh)
        if errs:
            raise StudyError("Contacts:\n  " + "\n  ".join(errs))
    from .printing.region import ensure as ensure_print
    warnings += ensure_print(study, mesh)
    from .thermal.check import floating_parts as thermal_floating
    for case in study.thermal.cases:
        errs = thermal_floating(study, mesh, case)
        if errs:
            raise StudyError(f"Thermal state '{case.name}': parts without heat removal (there is no path through them to a prescribed temperature, convection or radiation – the steady state has no solution):\n  " + "\n  ".join(errs))
    errs = floating_parts(study, mesh) if study.load_cases else []
    if errs:
        raise StudyError("Some parts are not connected to a support (free bodies – the problem would be singular):\n  "
                         + "\n  ".join(errs) + "\nConnect them (contact, bolt, weld) or leave them out of the calculation (GUI: step Joints → Parts excluded from the calculation; script: Study.ignored_parts).")
    return mesh, warnings


def floating_parts(study: Study, mesh: Mesh) -> list[str]:
    """Parts of the mesh that are not connected through common nodes, bolts and constraints to any constraint (support)."""
    from .mesh import part_union
    n_p = len(mesh.part_names)
    if n_p == 0:
        return []
    owner, find, union = part_union(mesh)        # common nodes (conforming mesh)
    # bolts connect the parts of their constraint faces
    from .joints.fem import coupling_keys
    for i, b in enumerate(study.active_bolts(), 1):
        ps = set()
        for key, _ in coupling_keys(b, i, mesh):
            ps.update(np.unique(owner[mesh.surface_nodes(key)]).tolist())
        ps.discard(-1)
        ps = sorted(ps)
        for q in ps[1:]:
            union(ps[0], q)
    from .ties import tied_part_pairs
    for a, b in tied_part_pairs(mesh):           # parts connected by *TIE
        union(a, b)
    supported = set()
    for r in mesh.report.get("contacts", []):    # contact connects the parts (one-sided – the convergence check separately)
        ps = set(np.unique(owner[np.unique(mesh.surfaces[r["slave"]])]).tolist()) - {-1}
        if r.get("master"):
            ps |= set(np.unique(owner[np.unique(mesh.surfaces[r["master"]])]).tolist()) - {-1}
            ps = sorted(ps)
            for q in ps[1:]:
                union(ps[0], q)
    for r in mesh.report.get("contacts", []):    # support only in compression to the ground = support
        if not r.get("master"):
            for q in np.unique(owner[np.unique(mesh.surfaces[r["slave"]])]):
                if q >= 0:
                    supported.add(int(q))
    supported = {find(q) for q in supported}
    for c in study.constraints:
        if isinstance(c, RemoteSupport) and not any(c.dofs):
            continue                      # a remote constraint without a constrained component does not support (the part at the remote point does)
        if c.selection in mesh.surfaces and len(mesh.surfaces[c.selection]):
            for q in np.unique(owner[mesh.surface_nodes(c.selection)]):
                if q >= 0:
                    supported.add(find(int(q)))
    if not supported:
        return []
    return [mesh.part_names[pi] for pi in range(n_p) if find(pi) not in supported]


def prepare_mesh(study: Study, workdir: Path, study_dir, log, mesh: Mesh | None = None,
                 parts=None, cancel=None) -> tuple[Mesh, list[str]]:
    """Geometry → readiness check → mesh (shared by both solvers).

    mesh = a finished mesh of the same study (the MCP server keeps it between calls) → not meshed again;
    parts = geometry already loaded (OCP).
    """
    geo = study.geometry_path(study_dir)
    if parts is None:
        log(f"Loading geometry {geo.name}")
        from .geometry import legacy_step_names, read_step   # OCP (~1 s) only when the geometry is really loaded
        parts = read_step(geo, legacy_names=legacy_step_names(study))
    errors, warnings = check_study(study, [p.name for p in parts])
    if errors:
        raise StudyError("Study is not ready:\n  " + "\n  ".join(errors))
    if mesh is not None:
        log("Mesh reused from previous meshing")
        old = mesh.report.get("print_warnings", [])
        # the warnings of the study check remain; only those not yet in them are added from the mesh (print variants
        # store the warnings of check_study in study_warnings too)
        warnings = warnings + [w for w in mesh.report.get("study_warnings", []) if w not in old and w not in warnings]
        min_sj = mesh.report.get("min_scaled_jacobian")
        if min_sj is not None and min_sj < 0.2:
            warnings.append(
                f"Low mesh element quality: minimum scaled Jacobian minSJ = {min_sj:.3f} (< 0.2). We recommend refining the mesh or modifying the geometry.")
        return mesh, warnings + _ensure_print(study, mesh, log)
    log("Meshing in progress (Gmsh)")
    mesh, mw = mesh_study(study, parts, workdir / "mesh", cancel)
    warnings += mw
    rep = mesh.report
    log(f"Mesh: {rep['nodes']} nodes, {rep['elements']} elements {rep['element_type']}, min. scaled Jacobian {rep['min_scaled_jacobian']:.3f}")
    min_sj = rep.get("min_scaled_jacobian")
    if min_sj is not None and min_sj < 0.2:
        warnings.append(
            f"Low mesh element quality: minimum scaled Jacobian minSJ = {min_sj:.3f} (< 0.2). We recommend refining the mesh or modifying the geometry.")
    return mesh, warnings


def _ensure_print(study: Study, mesh: Mesh, log) -> list[str]:
    """Assignment of printing to a taken-over mesh (the printing settings are not in the signature of the mesh)."""
    from .printing.region import ensure, print_signature
    if mesh.report.get("print_sig") != print_signature(study, mesh.part_names):
        log("3D printing: assigning shell and infill to the mesh")
    return ensure(study, mesh)


def choose_backend(study: Study, mesh: Mesh, log=None) -> str:
    """The solver of the mechanical states: see `choose_backend_reason` (this returns only the name)."""
    return choose_backend_reason(study, mesh, log)[0]


NONLINEAR_REASON = "nonlinear analysis (contact, plasticity, large displacements) – CalculiX only"


def decide_backend(study: Study, mesh: Mesh, log=None) -> tuple[str, str]:
    """(backend, reason) of the mechanical states of ONE `run_study`: the nonlinear analysis is always ccx, otherwise `choose_backend_reason`.
    The choice depends on the free memory of the moment, so `run_study` makes it once and passes it to everything that needs it (the side
    jobs beside the load cases, `_run_mechanics`) – asking again could give another answer and the parts would disagree.
    """
    if study.is_nonlinear(mesh=mesh):
        return "ccx", NONLINEAR_REASON
    return choose_backend_reason(study, mesh, log)


def choose_backend_reason(study: Study, mesh: Mesh, log=None) -> tuple[str, str]:
    """(backend, reason) – `Results.info["solver"]["reason"]` shows the reason.
    auto: what the own assembly cannot do (contact / nonlinear analysis, beam structures, cyclic symmetry, orthotropy with 3D printing, constraint kinds of
    `solve_large.unsupported`; bolts and welds are on the direct path since 1.4.2; only the natural frequencies of a model with bolts go to ccx) or refuses for this
    mesh (`linear_constraints.check`: faces that share nodes between constraint equations and fixed faces, a dependent *TIE master, a dense distributing support
    above the limit) → ccx WITH THE REASON; otherwise the direct solver (own assembly + rslab, ONE factorization for all load states – also for a single one; the
    numbers are in the F13 plan) when its memory estimate fits into the free RAM, then the iterative solver (own assembly + hypre, AMG), otherwise ccx.
    log = a large model without a usable direct / iterative solver → a line in the log saying what is missing.
    """
    from .solvers import canonical
    b = canonical(study.solver.backend)
    if b != "auto":
        return b, "chosen by the user"
    import psutil
    from . import direct
    from .hypre import available
    from .solve_large import unsupported
    why = unsupported(study)
    if why:
        return "ccx", "CalculiX only: " + ", ".join(why)
    hypre_why = unsupported(study, "hypre")          # what only the direct solver can do (constraints that couple nodes: remote supports / rigid loads / *TIE / rigid point mass)
    ndof = 3 * len(mesh.nodes)
    free = psutil.virtual_memory().available / 2**30
    ready = _direct_ready()
    if not ready and ndof < config.LARGE_DOF:
        from .solvers import direct_missing
        miss = direct_missing()
        return "ccx", f"the direct solver is not installed – missing {', '.join(miss)}" if miss else "the direct solver is not installed"
    from .linear_constraints import check as constraints_check
    refused = constraints_check(study, mesh)         # the constraints that the own assembly would refuse for this mesh: CalculiX from the start, with the reason
    if refused:
        return "ccx", "CalculiX only: " + "; ".join(refused)
    if ready and direct.fits(ndof, free):
        return "ngs_direct", (f"linear study without CalculiX-only features – one factorization for all load states "
                              f"(memory estimate {direct.estimate_gb(ndof):.1f} GB, free {free:.1f} GB)")
    if hypre_why:
        return "ccx", ("the direct solver does not fit into the free memory and the iterative solver cannot do " + ", ".join(hypre_why)
                       if ready else "CalculiX only: " + ", ".join(hypre_why))
    if available():
        if ready:
            return "ngs_hypre", (f"the direct solver would need about {direct.estimate_gb(ndof):.1f} GB, free {free:.1f} GB "
                                 "– iterative solver (memory grows linearly)")
        return "ngs_hypre", f"large model ({ndof} DOF)"
    if log is not None:
        from .solvers import hypre_missing
        dof = f"{ndof:,}".replace(",", " ")
        log(f"Large model ({dof} DOF): the iterative solver (hypre) would use less memory, but {', '.join(hypre_missing())} is missing – solving in CalculiX (Tools → Program modules…)")
    return "ccx", "the direct solver does not fit into the free memory and hypre is not installed"


def _solver_shown(requested: str, actual: str) -> str:
    """The solver as results and reports name it: FAER for SPOOLES in a ccx that has the faer solver (`config.solver_label`), except when SPOOLES was asked for by name (the
    deck then carries `** FEA_SOLVER=SPOOLES`, the runner keeps faer off, `ccx.runner.FAER_OFF_COMMENT`)."""
    return actual if str(requested).strip().upper() == "SPOOLES" else config.solver_label(actual)


def ccx_solver_note(study: Study, warnings: list[str], log, procedure: str = 'static') -> str:
    """Linear solver with which CalculiX actually computes (config.solver_name – into Results.info["ccx_solver"]).
    The automatic choice (the default) → one line in the log. A requested unavailable one (PaStiX in the Linux ccx) → one line in the log and a
    warning of the results with the name of the solver used (earlier it was silently replaced and the report showed the requested one).
    """
    req = str(study.solver.solver or config.AUTO_SOLVER)
    actual = config.solver_name(req, procedure)
    if req.upper() == config.AUTO_SOLVER:         # default choice: only into the log (otherwise a warning in every result)
        log(f"CalculiX equation solver: {_solver_shown(req, actual)} (automatic choice)")
    elif actual.upper() != req.upper():
        from .solvers import linear_solver_note
        text = (f"Equation solver {req} is not available in CalculiX ({linear_solver_note(req)}) – CalculiX uses the solver {_solver_shown(req, actual)}")
        if text not in warnings:
            warnings.append(text)
            log(text)
    return _solver_shown(req, actual)


def ccx_procedures(study: Study, mech_in_ccx: bool, mesh: Mesh | None = None) -> list[str]:
    """The ccx procedures (labels of `config.solver_name`) that a `run_study` of this study writes, in the order of the run: the mechanical states
    (only when ccx computes them – not with the NGSolve backends), heat, natural frequencies, buckling, dynamics (modal basis + static correction)."""
    procs: list[str] = []
    has_mech = bool(study.load_cases) or (study.analysis.nonlinear and not study.analysis.branches and bool(study.analysis.sequence))
    if mech_in_ccx and has_mech:
        from .ccx.inp import _nonlinear_procedure
        procs.append(_nonlinear_procedure(study) if study.is_nonlinear(mesh=mesh) else "static")
    if study.thermal.cases:
        procs.append("heat")
    if study.modal.enabled:
        procs.append("modal")
    if study.buckling.enabled and study.load_cases:
        procs.append("buckle")
    if study.dynamics.enabled:
        procs += ["modal", "static"]
    if study.nlbuckling.enabled:
        procs.append("static")
    return list(dict.fromkeys(procs))


def solver_label(solvers: dict[str, str]) -> str:
    """One solver → its name; several → "PASTIX (static/heat), SPOOLES (modal)" (Results.info["ccx_solver"])."""
    names = list(dict.fromkeys(solvers.values()))
    if len(names) <= 1:
        return names[0] if names else ""
    return ", ".join(f"{s} ({'/'.join(p for p, v in solvers.items() if v == s)})" for s in names)


def record_ccx_solvers(study: Study, res: Results, procedures: list[str]) -> None:
    """The solver of every ccx procedure that ran → Results.info["ccx_solvers"] ({procedure: solver}) and the readable
    Results.info["ccx_solver"]. Without it the one name of the mechanical job would also stand for the natural-frequency, heat and shell jobs,
    which use another solver (config.SPOOLES_PROCEDURES)."""
    if not procedures:
        return
    req = str(study.solver.solver or config.AUTO_SOLVER)
    solvers = {p: _solver_shown(req, config.solver_name(req, p)) for p in procedures}     # the deck says SPOOLES, a ccx with the faer solver computes with FAER
    res.info["ccx_solvers"] = solvers
    res.info["ccx_solver"] = solver_label(solvers)


def attach_solver_notes(res: Results, workdir: str | Path, since: float) -> None:
    """The ccx runs of this study folder (since the monotonic time `since`) in which the internal check of the SPOOLES factors fired (`fea_core.ccx.runner.SpoolesCheck`: a
    multi-threaded factorization was repeated, or the matrix itself fails the check): a warning for the user in `res.warnings` and the data in `res.info["solver_check"]`.
    Without a firing check nothing is added (the normal case). Also the name FAER of the solver (`record_ccx_solvers`) becomes SPOOLES when the ccx runs of this study did
    not factorize a single system with faer (`ccx.runner.take_faer_usage`)."""
    from .ccx.runner import take_faer_usage, take_solver_events
    used = take_faer_usage(workdir, since)
    solvers = res.info.get("ccx_solvers")
    if used and not any(used) and isinstance(solvers, dict) and "FAER" in solvers.values():      # a ccx with the faer solver, but SPOOLES factorized every system (unsymmetric, small, ...)
        solvers = {p: ("SPOOLES" if s == "FAER" else s) for p, s in solvers.items()}
        res.info["ccx_solvers"] = solvers
        res.info["ccx_solver"] = solver_label(solvers)
    events = take_solver_events(workdir, since)
    if not events:
        return
    for ev in events:
        if ev["note"] not in res.warnings:
            res.warnings.append(ev["note"])
    res.info["solver_check"] = [{k: v for k, v in ev.items() if k != "note"} for ev in events]


def _direct_ready() -> bool:
    """The direct solver and the assembly (numba; NGSolve with FEA_ASSEMBLY=ngsolve) are available (all are imported only in separate processes –
    here only the search)."""
    from .solvers import direct_missing
    return not direct_missing()


def read_remote(dat: Path, inpmap) -> list[dict]:
    """Measured reactions and displacements of the rigid remote constraints by steps from *NODE PRINT (NSET NRSUP, U and RF):
    [{selection: {"force": RF REF, "moment": RF ROT (about the point), "u": U REF, "rot": U ROT [rad]}}].
    The .frd does not write nodes outside the elements → only the .dat (verified 13. 10. 2026, scratch/remote/exp1.py).
    """
    if not inpmap.remote:
        return []
    buf = Path(dat).read_bytes()
    heads = [h for h in header_lines(buf, dat_pattern("for set {} and time", "NRSUP"))
             if b"total" not in buf[h:line_end(buf, h)]]
    blocks = _read_printed(dat, "NRSUP")
    steps, cur = [], None
    for h, blk in zip(heads, blocks):
        head = buf[h:line_end(buf, h)].decode("latin-1").strip().lower()
        kind = "U" if head.startswith("displacements") else "RF" if head.startswith("forces") else None
        if kind is None:
            continue
        if cur is None or kind in cur:                    # new step (each step writes U and RF once)
            cur = {}
            steps.append(cur)
        cur[kind] = blk
    out = []
    for blk in steps:
        d = {}
        for sel, info in inpmap.remote.items():
            def val(kind, node):
                v = blk.get(kind, {}).get(node)
                if v is None:
                    raise CcxError(f"{kind} of node {node} of remote support '{sel}' is missing in .dat")
                return np.asarray(v[:3], float)
            d[sel] = {"force": val("RF", info["ref"]), "moment": val("RF", info["rot"]),
                      "u": val("U", info["ref"]), "rot": val("U", info["rot"])}
        out.append(d)
    return out


def remote_reactions(study: Study, mesh: Mesh, U, F, parts_R: dict, c0, remote: dict | None,
                     reac: dict, rmom: dict, deformed: bool = False, foundation: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reactions of the remote constraints (fills reac, rmom about c0). Rigid: the measured RF REF/ROT of the node (force, moment about the point)
    MINUS the external load directly on the nodes of the rigid face – ccx moves into RF REF/ROT only the forces of the elements (K·u) of the dependent
    nodes, not their *CLOAD/*DLOAD (verified 13. 10. 2026: pressure on the edge of the end face → the imbalance = exactly this load),
    i.e. reaction = K·u − F as for the other constraints. It lies outside the nodes of the mesh, so it also returns the sum (Rp, Mp
    about c0) for the equilibrium check. Distributing: Σ(RF − F) in the nodes of the constraint equations (already in R and parts_R), the moment
    about the point measured in the same way; the displacement / rotation of the point from RBE3.
    parts_R = the constraint forces at the nodes WITHOUT the elastic foundation: `StepResult.remote` is the reaction of the support at its point.
    foundation = {selection: (N, 3) forces of the elastic support on the same face}: not part of the support at the point, but of the reaction of
    the selection (reac, rmom) – both the ccx path (measured RF REF/ROT) and the own path (Σ(RF − F)) report the same sum.
    """
    Rp, Mp, info = np.zeros(3), np.zeros(3), {}
    sups = [c for c in study.constraints if isinstance(c, RemoteSupport)]
    if not sups:
        return Rp, Mp, info
    from .remote import point_motion
    x = mesh.nodes
    for c in sups:
        p = np.asarray(c.point, float)
        if c.coupling == "rigid":
            meas = (remote or {}).get(c.selection)
            if meas is None:
                raise CcxError(f"Measured reaction of rigid remote support '{c.selection}' is missing (RF of REF/ROT node in .dat)")
            if meas.get("own"):                    # own assembly: the nodes of the face are in R already (RF − F), the point moves with the extra nodes
                r = parts_R.get(c.selection, np.zeros_like(x))
                fp = r.sum(axis=0)
                mp = np.cross(x - p, r).sum(axis=0)
                motion = np.concatenate([meas["u"], meas["rot"]])
            else:
                idx = mesh.surface_nodes(c.selection)
                fp = np.asarray(meas["force"], float) - F[idx].sum(axis=0)
                mp = np.asarray(meas["moment"], float) - np.cross(x[idx] - p, F[idx]).sum(axis=0)
                motion = np.concatenate([meas["u"], meas["rot"]])
                Rp = Rp + fp
                pe = p + np.asarray(meas["u"], float) if deformed else p    # NLGEOM: the force acts at the displaced point
                Mp = Mp + mp + np.cross(pe - c0, fp)
        else:
            r = parts_R.get(c.selection, np.zeros_like(x))
            fp = r.sum(axis=0)
            mp = np.cross(x - p, r).sum(axis=0)
            motion = point_motion(mesh, c.selection, p, U)
        reac[c.selection] = fp
        rmom[c.selection] = mp + np.cross(p - c0, fp)
        fnd = (foundation or {}).get(c.selection)
        if fnd is not None:                    # an elastic support on the face of a remote support: the reaction of the selection is the sum of both
            reac[c.selection] = fp + fnd.sum(axis=0)
            rmom[c.selection] = rmom[c.selection] + np.cross(x - c0, fnd).sum(axis=0)
        info[c.selection] = {"point": p.tolist(), "coupling": c.coupling, "dofs": [bool(v) for v in c.dofs],
                             "force": fp.tolist(), "moment": mp.tolist(),
                             "u": np.asarray(motion[:3]).tolist(), "rot": np.asarray(motion[3:]).tolist()}
    return Rp, Mp, info


def make_step(study: Study, mesh: Mesh, lc, U, S, RF, masks, refd=None, ref_force: float = 0.0,
              remote: dict | None = None, frame: tuple | None = None, extra: dict | None = None,
              x_eq: np.ndarray | None = None, check: bool = True, nodal: tuple | None = None,
              f_applied: np.ndarray | None = None) -> StepResult:
    """Result of a state: reaction = K·u − external load in the constrained DOF, equilibrium check.

    remote = measured reactions of the rigid remote constraints {selection: {"force", "moment" (about the point), "u", "rot"}}
    (RF and U of the REF/ROT node from the .dat, see read_remote). frame = (fe, base, rec) of the frame system – its loads and the reactions
    of the supports belong to the equilibrium check (forces between the members and the body are internal).
    extra = reactions outside the nodes of the mesh {name: (force (3,), points (g, 3), forces (g, 3))} (support only in compression – GAPUNI);
    x_eq = positions of the nodes for the moment equilibrium (NLGEOM: deformed x + u, the forces act in the deformed position).
    nodal = (F, Fr, Mr) of `loads.nodal_loads(study, mesh, lc)` when the caller needs them too (joint forces) – not recomputed.
    f_applied = the nodal forces that the solver really applied when they are more than `nodal_loads` (the own assembly adds the distributing remote loads and
    the weight of the point masses to the nodes of their faces) – only the force balance of the nodes of a *TIE uses it.
    """
    n = len(mesh.nodes)
    RF_solver = RF                                   # the solver's own K·u (`RF` below is rebound): a unit-gravity state keeps it for the superposition (`gravity_states`)
    F, Fr, Mr = nodal if nodal is not None else nodal_loads(study, mesh, lc)
    allc = np.zeros((n, 3), bool)
    for msk in masks.values():
        allc |= msk
    cyc = None
    if getattr(study, "cyclic", None) is not None and study.cyclic.enabled:
        from .cyclic import post_step          # per-sector reactions, forces across the cut faces, stresses at the cut
        RF, S, cyc = post_step(study, mesh, RF, F, S, allc)
    if mesh.report.get("ties"):
        # *TIE: the force of a slave node (K·u − F, which is the force of the constraint) passes through the weights of the tie into the master face. At a master node
        # that is also constrained the support holds it: R = RF − F + Σ w·(RF − F)_slave (= [Tᵀ(K·u − F)] at the constrained DOF). Without it the force that flows
        # through the tie into a support is missing in the reaction (and, with a large share, the check below reports an imbalance of the whole model).
        from .ties import flow_to_masters
        RF = RF + flow_to_masters(mesh, RF - (F if f_applied is None else f_applied))
    from .springs import supports, reactions as spring_reactions
    spring_r = spring_reactions(study, mesh, U) if supports(study) else {}
    # RF = K·u of the elements WITHOUT the force of the elastic support (measured 18. 11. 2026 on ccx: at a spring node RF = the internal force of the elements
    # = −K_spring·u) → at a node that also holds a constraint, RF − F = constraint force + reaction of the foundation: the foundation is subtracted here and added
    # once below (otherwise the shared nodes of a cylindrical support and an elastic support counted it twice: imbalance 0.15 %)
    R = np.where(allc, RF - F - (sum(spring_r.values()) if spring_r else 0.0), 0.0)
    # cylindrical support: reaction = projection of RF − F onto the constrained directions (radial / tangential / axial) at the nodes;
    # a selection with clamping / displacement is computed in global components
    plain = {c.selection for c in study.constraints if not isinstance(c, CylindricalSupport)}
    proj = {}
    if any(isinstance(c, CylindricalSupport) for c in study.constraints):
        from .cylinder import projectors
        proj = {k: v for k, v in projectors(study, mesh).items() if k not in plain}

    def part(name, msk):
        if name in proj:
            idx, P = proj[name]
            out = np.zeros_like(R)
            out[idx] = np.einsum("nij,nj->ni", P, R[idx])
            return out
        return np.where(msk, R, 0.0)
    support_R = {name: part(name, msk) for name, msk in masks.items()}       # the constraint forces (the remote supports are measured from these)
    R_con = R                                    # the constraint force of every constrained node, once (the supports below take it per support)
    parts_R = dict(support_R)
    if spring_r:                                 # elastic support: −k·u
        for name, r in spring_r.items():
            parts_R[name] = parts_R.get(name, np.zeros_like(R)) + r
            R = R + r
    reac = {name: r.sum(axis=0) for name, r in parts_R.items()}
    x = mesh.nodes
    c0 = 0.5 * (x.min(axis=0) + x.max(axis=0))
    rmom = {name: np.cross(x - c0, r).sum(axis=0) for name, r in parts_R.items()}
    # what a (node, DOF) held by k supports counts more than once in the per-support reactions: Σ per support − the force at the node once (the model checks judge the per-support
    # reactions as Σ − overlap; without the record the overlap of supports that share nodes cannot be told from a reaction that was changed)
    ov_f = sum((r.sum(axis=0) for r in support_R.values()), np.zeros(3)) - R_con.sum(axis=0)
    ov_m = sum((np.cross(x - c0, r).sum(axis=0) for r in support_R.values()), np.zeros(3)) - np.cross(x - c0, R_con).sum(axis=0)
    Rp, Mp, rinfo = remote_reactions(study, mesh, U, F, support_R, c0, remote, reac, rmom, deformed=x_eq is not None, foundation=spring_r)
    for name, (fsum, Xg, Fg) in (extra or {}).items():     # support only in compression: force on the part (−RF of the side of the body)
        reac[name] = np.asarray(fsum, float)
        mom = np.cross(np.asarray(Xg, float) - c0, np.asarray(Fg, float)).sum(axis=0)
        rmom[name] = mom
        Rp, Mp = Rp + reac[name], Mp + mom
    if cyc is not None:                                    # forces across the cut faces of the sector (neighbouring sectors)
        Rp, Mp = Rp + np.asarray(cyc["cut_force"]), Mp + np.asarray(cyc["cut_moment"])
    if x_eq is not None:                                   # NLGEOM: moments about the deformed position
        x = np.asarray(x_eq, float)
    if frame is not None:
        from .beams.run import frame_equilibrium
        fe_, _, rec = frame
        Fa, Ma, Rf, Rm = frame_equilibrium(study, fe_, lc, rec, c0)
        Fr = Fr + Fa
        Mr = Mr + Ma + np.cross(c0, Fa)               # Mr is about the origin of the coordinates
        Rp, Mp = Rp + Rf, Mp + Rm
    st = StepResult(lc.name, U, S, reac, refd or {})
    if lc.name.startswith("_unit_gravity_"):
        setattr(st, "_unit_arrays", (RF_solver, f_applied))
    st.reaction_moments = rmom
    st.moment_point = c0
    st.remote = rinfo
    st.equilibrium = equilibrium(x - c0, F, R, Fr, Mr - np.cross(c0, Fr), ref_force, Rp, Mp)
    st.equilibrium["reaction_overlap"], st.equilibrium["reaction_moment_overlap"] = ov_f.tolist(), ov_m.tolist()      # Σ per-support reactions − total (0 without shared nodes and DOF)
    if x_eq is not None:
        st.equilibrium["deformed"] = True
    st.equilibrium["applied_exact"] = load_resultant(study, mesh, lc).tolist()
    if cyc is not None:
        st.equilibrium["cyclic"] = cyc
    if check and st.equilibrium["imbalance"] > config.EQUILIBRIUM_TOL:
        e = st.equilibrium
        raise CcxError(
            f"Case '{lc.name}': reactions are not in equilibrium with the loads (forces {np.round(e['force_residual'], 5).tolist()} N, moments {np.round(e['moment_residual'], 3).tolist()} N·mm, imbalance {100 * e['imbalance']:.3g} %). The result is invalid – try a different solver.")
    return st


def study_state_digest(study: Study) -> str:
    """SHA-256 of everything a `run_study` result depends on in the study (all of it but the space-structures settings, which the solvers never read, and the "features" list that
    changes when they are created): a result that carries it (`Results.info["study_digest"]`, not saved; taken when `run_study` is ENTERED) is the result of exactly this study
    state – the mesh convergence check takes its first level from it."""
    import hashlib
    import json
    d = study.to_dict()
    for k in ("space", "features"):
        d.pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_study(study: Study, workdir: str | Path, study_dir: str | Path | None = None,
              on_log: Callable[[str], None] | None = None,
              on_solver_line: Callable[[str], None] | None = None,
              cancel: threading.Event | None = None, mesh: Mesh | None = None, parts=None,
              thermal: list | None = None,
              on_progress: Callable[[float, float, str], None] | None = None) -> Results:
    """mesh = a finished mesh of the same study → not meshed again. With parts (the geometry already loaded by OCP, MCP server)
    the study is checked; without parts it is taken over without a check (long-term creep states of 3D printing).
    thermal = finished results of the thermal states (ThermalResult) → not computed again.
    on_progress(value 0–1, end of phase 0–1, text) = progress for the GUI (fea_core.progress).
    """
    from .beams.run import frame_only, run_frame
    t_solver_notes = time.monotonic()                # the check events of the ccx runs of THIS call (attach_solver_notes)
    if frame_only(study):                            # frame system only (without STEP geometry)
        return run_frame(study, workdir, on_log, on_solver_line, cancel)
    if study.twod.enabled:                           # 2D calculation by a section (axisymmetric / plane problem)
        return run_twod(study, workdir, study_dir, parts, on_log, cancel)
    if study.shell.enabled:                          # shell model of plates and sheets (S6)
        return run_shellmodel(study, workdir, study_dir, parts, on_log, cancel)
    digest = study_state_digest(study)               # of the study as it was GIVEN (a study edited while the run goes on is not the one the result belongs to)
    from .progress import Progress, TRANSIENT_WEIGHT, WEIGHTS
    first, rest = _thermal_split(study) if thermal is None else ([], [])
    transient = {c.name for c in study.thermal.cases if c.transient}
    phases = ["check"] + (["mesh"] if mesh is None or parts is not None else [])
    phases += ["thermal"] if first else []
    has_mech = bool(study.load_cases) or (study.analysis.nonlinear and not study.analysis.branches
                                          and bool(study.analysis.sequence))
    phases += ["mech"] if has_mech else []
    phases += ["eigen"] if study.modal.enabled or study.buckling.enabled else []
    phases += ["dynamics"] if study.dynamics.enabled else []
    phases += ["post"]
    if study.load_cases and (rest or study.printing.creep.enabled):
        phases.append("background")                  # thermal states / creep run in parallel with the mechanics (with FEA_THREADS: the creep follows them in order, in this phase too)
    weights = {"thermal": TRANSIENT_WEIGHT if transient & set(first) else WEIGHTS["thermal"],
               "background": TRANSIENT_WEIGHT if transient & set(rest) else WEIGHTS["background"]}
    prog = Progress(on_progress, phases, weights)
    prog.phase("check", "Study check")
    from .solvers import check_backend
    errors = check_backend(study)                    # also without the study check (a taken-over mesh) – earlier than the meshing
    if errors:
        raise StudyError("Study is not ready:\n  " + "\n  ".join(errors))
    raw_log = on_log or (lambda s: None)
    log = prog.log(raw_log)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if mesh is not None and parts is None:
        warnings = _ensure_print(study, mesh, log)
    else:
        mesh, warnings = prepare_mesh(study, workdir, study_dir, log, mesh, parts, cancel)
    from .printing.creep import start_creep
    thermal_bg = None
    if thermal is None:
        thermal, thermal_bg = _start_thermal(study, mesh, workdir, warnings, log, on_solver_line, cancel, prog)
    if not has_mech:                                 # heat conduction / natural frequencies only
        res = Results(mesh, [], warnings, {"backend": "ccx"}, thermal)
        if getattr(study, "cyclic", None) is not None and study.cyclic.enabled:
            from .cyclic import summary as cyclic_summary
            res.info["cyclic"] = cyclic_summary(study, mesh)
        _add_eigen(study, mesh, res, workdir, log, on_solver_line, cancel, prog)
        _add_dynamics(study, mesh, res, workdir, log, on_solver_line, cancel, prog)
        procs = ccx_procedures(study, False, mesh)               # heat conduction / natural frequencies / dynamics only
        if procs:
            ccx_solver_note(study, res.warnings, log, procs[0])  # one note when the requested solver is replaced
            record_ccx_solvers(study, res, procs)
        attach_solver_notes(res, workdir, t_solver_notes)
        prog.phase("post", "Saving results")
        res.info["timings"] = prog.timings()         # durations of the phases of the calculation [s] (GUI card Properties, report)
        res.info["study_digest"] = digest
        res.save(workdir / "results.npz")
        log("Done")
        prog.finish()
        return res
    creep_bg = None
    side = None
    try:
        # background creep with the original log – its messages (own run_study) must not move the progress of the main calculation
        creep_bg = start_creep(study, mesh, thermal, workdir, raw_log, on_solver_line, cancel)
        from .side_jobs import SideJobs              # eigen / dynamics beside the load cases when the machine has room
        decision = decide_backend(study, mesh, raw_log)      # the solver of the load cases, ONCE: the side jobs and the mechanics share it
        side = SideJobs(study, mesh, workdir, raw_log, on_solver_line, cancel, thermal, prog, backend=decision[0])
        with side.mech_budget(), nodal_cache():      # nodal loads of a state integrated once (.inp, reactions, equilibrium)
            res = _run_mechanics(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, creep_bg, prog, decision, raw_log)
        if side.active:
            side.join(res)
            if side.rest_modal:                      # the natural frequencies of the direct path follow the mechanics (one direct run at a time)
                _add_eigen(study, mesh, res, workdir, log, on_solver_line, cancel, prog, kinds=("modal",))
            if side.rest_dyn:
                _add_dynamics(study, mesh, res, workdir, log, on_solver_line, cancel, prog)
            _add_nlbuckling(study, mesh, res, workdir, log, on_solver_line, cancel)
        else:
            _add_eigen(study, mesh, res, workdir, log, on_solver_line, cancel, prog)
            _add_dynamics(study, mesh, res, workdir, log, on_solver_line, cancel, prog)
        if thermal_bg is not None:                   # transient / unused thermal states computed in parallel
            prog.phase("background", "Finishing thermal cases (concurrent)")
            res.thermal = _merge_thermal(study, thermal, thermal_bg.result(), warnings)
    except BaseException:                            # main job / a background job failed or cancelled → stop the others
        for bg in (thermal_bg, creep_bg, side):
            if bg is not None:
                bg.abort(log)
        raise
    procs = ccx_procedures(study, res.info.get("backend") == "ccx", mesh)       # everything that ccx computed in this run
    if procs:
        if "ccx_solver" not in res.info:                 # no ccx mechanics (NGSolve backends) → the note of the replaced solver is still due
            ccx_solver_note(study, res.warnings, log, procs[0])
        record_ccx_solvers(study, res, procs)
    attach_solver_notes(res, workdir, t_solver_notes)
    prog.phase("background", "Saving results", 0.9)
    res.info["timings"] = prog.timings()
    res.info["study_digest"] = digest
    res.save(workdir / "results.npz")
    log("Done")
    prog.finish()
    return res


def run_twod(study, workdir, study_dir=None, parts=None, on_log=None, cancel=None) -> Results:
    """2D calculation (fea_core.twod.run_2d) wrapped into Results: empty 3D mesh, Results.twod = Results2D,
    info["twod"] = summary. Saves results.npz + results_2d.npz.
    """
    from .twod.run import run_2d
    from .twod.model import TwoDError
    from .beams.run import empty_mesh
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        r2 = run_2d(study, workdir, parts=parts, study_dir=study_dir, on_log=log, cancel=cancel)
    except TwoDError as e:
        raise StudyError(str(e)) from None
    res = Results(empty_mesh(), [], list(r2.warnings), {"backend": "ccx", "twod": r2.summary()}, twod=r2)
    attach_solver_notes(res, workdir, t0)
    res.save(workdir / "results.npz")
    log("Done")
    return res


def run_shellmodel(study, workdir, study_dir=None, parts=None, on_log=None, cancel=None) -> Results:
    """Shell model (fea_core.shell.run_shell) wrapped into Results: empty 3D mesh, Results.shell."""
    from .shell.run import run_shell
    from .shell.model import ShellError
    from .beams.run import empty_mesh
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        rs = run_shell(study, workdir / "shell", parts=parts, study_dir=study_dir, on_log=log, cancel=cancel)
    except ShellError as e:
        raise StudyError(str(e)) from None
    res = Results(empty_mesh(), [], list(rs.warnings), {"backend": "ccx", "shell": rs.summary()}, shell=rs)
    attach_solver_notes(res, workdir, t0)
    res.save(workdir / "results.npz")
    log("Done")
    return res


def _add_eigen(study, mesh, res, workdir, log, on_solver_line, cancel, prog=None, kinds: tuple | None = None) -> None:
    """Natural frequencies and buckling (fea_core.eigen): the natural frequencies on the direct path or in a ccx job, buckling in ccx, on the same mesh.
    kinds = only these ("modal" / "buckling": what the side jobs did not compute; their results are put in front of / after the others and the nonlinear buckling
    is left to the caller)."""
    if not (study.modal.enabled or study.buckling.enabled or study.nlbuckling.enabled):
        return
    from .eigen import run_eigen, takeover_warnings, EigenError
    lines = on_solver_line
    if prog is not None:
        from .progress import CcxLines
        prog.phase("eigen", "Natural frequencies / buckling")
        lines = prog.lines(on_solver_line, "eigen", CcxLines(int(study.modal.enabled) + int(study.buckling.enabled)))
    try:
        if study.modal.enabled or study.buckling.enabled:
            modes = run_eigen(study, mesh, Path(workdir) / "eigen", log, lines, cancel, res.thermal, kinds=kinds or ("modal", "buckling"))
            res.modes = modes + list(res.modes or []) if kinds else modes
            for note in takeover_warnings(modes):  # an engine that took the analysis over from the other one (a crash of ccx, a failed direct analysis) is a warning, not only a log line
                if note not in res.warnings:
                    res.warnings.append(note)
    except EigenError as e:
        raise StudyError(str(e)) from None
    if kinds is None:
        _add_nlbuckling(study, mesh, res, workdir, log, on_solver_line, cancel)


def _add_nlbuckling(study, mesh, res, workdir, log, on_solver_line, cancel) -> None:
    """Nonlinear buckling with imperfection (fea_core.imperfect) after the linear modes – also when those ran beside the
    load cases (side_jobs), because it reuses the buckling modes of res.modes."""
    if not study.nlbuckling.enabled:
        return
    from .imperfect.model import ImperfectionError
    from .imperfect.run import attach as attach_nlbuckling       # nonlinear buckling with imperfection (F9 C2)
    try:
        attach_nlbuckling(study, mesh, res, Path(workdir) / "nlbuckle", log, on_solver_line, cancel)
    except ImperfectionError as e:
        raise StudyError(str(e)) from None


def _add_dynamics(study, mesh, res, workdir, log, on_solver_line, cancel, prog=None) -> None:
    """Structural dynamics (fea_core.dynamics): modal basis + response by modal superposition on the same mesh."""
    if not study.dynamics.enabled:
        return
    from .dynamics.run import run_dynamics
    from .dynamics.modal import DynamicsError
    lines = on_solver_line
    if prog is not None:
        from .progress import CcxLines
        prog.phase("dynamics", "Dynamics (modal superposition)")
        lines = prog.lines(on_solver_line, "dynamics", CcxLines(2, "Dynamics"))
    try:
        res.dynamics = run_dynamics(study, mesh, Path(workdir) / "dynamics", log, lines, cancel)
    except DynamicsError as e:
        raise StudyError(str(e)) from None
    for w in res.dynamics.warnings:
        if w not in res.warnings:
            res.warnings.append(w)


def _start_thermal(study, mesh, workdir, warnings, log, on_solver_line, cancel, prog=None):
    """Thermal states from which the mechanics takes the temperature right away; the others (typically transient – ccx assembles the matrix
    in every iteration, minutes) in a thread in parallel with the mechanics. Returns (finished states, background job | None).
    """
    if not study.thermal.cases:
        return [], None
    from .thermal.run import run_thermal
    first, rest = _thermal_split(study)
    lines, bg_lines = on_solver_line, on_solver_line
    if prog is not None:                             # progress of the thermal states from the ccx output (time of the transient state)
        from .progress import ThermalLines
        dur = {c.name: (c.time if c.transient else None) for c in study.thermal.cases}
        if first:
            prog.phase("thermal", "Heat conduction")
            lines = prog.lines(on_solver_line, "thermal", ThermalLines([(n, dur[n]) for n in first]))
        if rest:
            bg_lines = prog.lines(on_solver_line, "background", ThermalLines([(n, dur[n]) for n in rest],
                                                                             "Thermal cases (concurrent)"), True)
    thermal = run_thermal(study, mesh, workdir / "thermal", warnings, log, lines, cancel,
                          cases=first) if first else []
    bg = None
    if rest:
        bg = Background(lambda c, w: run_thermal(study, mesh, workdir / "thermal", w, log, bg_lines, c,
                                                 cases=rest), cancel, "thermal cases")
    return thermal, bg


def _thermal_split(study) -> tuple[list[str], list[str]]:
    """(thermal states from which the mechanics takes the temperature – computed at once, the others – in the background).
    With FEA_THREADS set by the user (`config.threads_explicit`) nothing runs in the background: the cap of the threads is a budget of ALL concurrent ccx work, and the background
    states would run beside the mechanics (and its side jobs) with the full number of threads – every state is computed before the mechanics, in order."""
    names = [c.name for c in study.thermal.cases]
    if config.threads_explicit():
        return names, []
    need = {t.case for lc in study.load_cases for t in study.thermal_loads(lc) if t.case}
    first = [n for n in names if n in need] if study.load_cases else names
    return first, [n for n in names if n not in first]


def _creep_phase(study, prog, creep_bg=None) -> None:
    """Progress phase of the long-term creep states: "background" (concurrent: `start_creep` started them beside the mechanics; with FEA_THREADS they follow the mechanics in order – then
    their time is booked here as well, not under the phase the main calculation was in)."""
    if prog is not None and study.printing.creep.enabled:
        prog.phase("background", "3D printing: creep (concurrent)" if creep_bg is not None else "3D printing: creep")


def _merge_thermal(study, first: list, rest_result: tuple, warnings: list[str]) -> list:
    rest, w = rest_result
    for x in w:
        if x not in warnings:
            warnings.append(x)
    order = {c.name: k for k, c in enumerate(study.thermal.cases)}
    return sorted(list(first) + list(rest), key=lambda t: order.get(t.name, 1 << 30))


class Background:
    """Calculation in a thread in parallel with the main job (thermal states, creep). fn(cancel, warnings) → result.
    The error is passed on only in result(); abort() stops the running ccx (the main job failed or was cancelled).
    """

    def __init__(self, fn, cancel: threading.Event | None, label: str):
        self.label = label
        self.stop = threading.Event()
        self.warnings: list[str] = []
        self._out = None
        self._err: BaseException | None = None
        self._user = cancel

        def watch():
            while not self.stop.is_set() and self._thread.is_alive():
                if cancel is not None and cancel.wait(0.2):
                    self.stop.set()
                    return
                if cancel is None:
                    self.stop.wait(0.5)

        def run():
            try:
                self._out = fn(self.stop, self.warnings)
            except BaseException as e:  # noqa: BLE001 – passed on in result()
                self._err = e
        self._thread = threading.Thread(target=run, daemon=True, name=f"fea-bg-{label}")
        self._thread.start()
        threading.Thread(target=watch, daemon=True).start()

    def result(self):
        self._thread.join()
        if self._err is not None:
            raise self._err
        if self._user is not None and self._user.is_set():
            raise CcxError("Analysis canceled by user")
        return self._out, self.warnings

    def abort(self, log=None, timeout: float = 30.0) -> bool:
        """Stops the job and waits for it (at most timeout s). False = still running (logged)."""
        self.stop.set()
        self._thread.join(timeout=timeout)
        alive = self._thread.is_alive()
        if alive and log is not None:
            log(f"Background job '{self.label}' did not stop within {timeout:g} s – it ends on its own")
        return not alive


def _run_mechanics(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, creep_bg,
                   prog=None, decision: tuple[str, str] | None = None, raw_log=None) -> Results:
    """Mechanical states (ccx, or the own assembly + the direct / iterative solver) + creep of printing; without saving (run_study).
    decision = (backend, reason) made by `run_study` (`decide_backend`) – the same one that the side jobs were planned with.

    raw_log = the log without the progress marks (for the creep states computed in order, see `add_creep_steps`).

    Gravity cases of a LINEAR study (the 14 signed QSL cases of a launch analysis, the unit-gravity statics of the model checks) are linear combinations of three unit states
    (`fea_core.gravity_states`): those are solved once per study and mesh (or taken from an earlier run in the state cache), every gravity case is made from them and goes through
    `make_step` – reactions and the equilibrium check are those of the combined state. Anything that breaks the superposition (contact, joints, thermal states, a prescribed
    displacement, …) or saves nothing solves every case as before; `FEA_STATE_CACHE=0` switches it off."""
    from . import gravity_states as GS
    pl = None
    if study.load_cases and not study.is_nonlinear(mesh=mesh):
        decision = decision if decision is not None else choose_backend_reason(study, mesh, log)
        pl = GS.plan(study, mesh, decision[0])
    if pl is None:
        return _run_mechanics_solve(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, creep_bg, prog, decision, raw_log)
    sub = GS.reduced_study(pl, study)
    if sub.load_cases:
        res = _run_mechanics_solve(sub, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, creep_bg, prog, decision, raw_log)
    else:
        if prog is not None:
            prog.phase("mech", "Gravity cases (stored unit states)")
        res = Results(mesh, [], warnings, {}, thermal)
    try:
        GS.expand(pl, study, mesh, res, Path(workdir), log)
    except Exception as e:  # noqa: BLE001 – the superposition is an optimisation: whatever goes wrong in it, every case is solved as it always was
        if cancel is not None and cancel.is_set():
            raise
        log(f"Gravity cases: the superposition of the unit states failed ({(str(e).splitlines() or [type(e).__name__])[0][:160]}) - solving every case")
        return _run_mechanics_solve(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, creep_bg, prog, decision, raw_log)
    return res


def _run_mechanics_solve(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, creep_bg,
                         prog=None, decision: tuple[str, str] | None = None, raw_log=None) -> Results:
    """The solution of the mechanical states of `study` as they are (`_run_mechanics` decides which states these are)."""
    from .printing.creep import add_creep_steps
    if study.is_nonlinear(mesh=mesh):                # contact / NLGEOM / history / plasticity → nonlinear in ccx
        from .contacts.run import run_nonlinear
        if prog is not None:
            prog.phase("mech", "Nonlinear analysis (CalculiX)")
        from .ccx.inp import _nonlinear_procedure
        ccx_solver = ccx_solver_note(study, warnings, log, _nonlinear_procedure(study))      # mortar: always SPOOLES; the rest: PaStiX only where this ccx has it, otherwise SPOOLES
        res = run_nonlinear(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, prog)
        res.info["ccx_solver"] = ccx_solver
        res.info.setdefault("solver", {"backend": "ccx", "engine": f"CalculiX {ccx_solver}", "ndof": 3 * len(mesh.nodes),
                                       "reason": NONLINEAR_REASON})
        if prog is not None:
            prog.phase("post", "Processing results")
        return res
    backend, reason = decision if decision is not None else choose_backend_reason(study, mesh, log)
    fallback = None
    while backend in ("ngs_hypre", "ngs_direct"):
        from .solve_large import run_ngs
        try:
            res = run_ngs(study, mesh, workdir, warnings, log, cancel, thermal, backend[4:])
        except Exception as e:  # noqa: BLE001
            # auto chose the direct / iterative path (faster, one factorization for all states). On failure it continues: the direct solver that does not
            # fit into the memory → the iterative hypre (if installed), anything else (conditioning, a missing library) → ccx.
            # A manual choice of the backend passes the error on.
            from .solve_large import DirectUnavailable, MeshQualityError
            if (study.solver.backend != "auto" or backend != "ngs_direct" or (cancel is not None and cancel.is_set())
                    or isinstance(e, MeshQualityError)):      # a mesh problem is the mesh's, CalculiX would reject the same elements
                raise
            from .pathcheck import scrub_paths
            reason_txt = scrub_paths((str(e).splitlines() or [type(e).__name__])[0])[:200]      # saved into the results: no folders of this machine
            label = {"ngs_direct": "Direct solver", "ngs_hypre": "Iterative solver (hypre)"}[backend]
            nxt = "ccx"
            if isinstance(e, DirectUnavailable):
                from .hypre import available as hypre_ok
                from .solve_large import unsupported as ngs_unsupported
                nxt = "ngs_hypre" if hypre_ok() and not ngs_unsupported(study, "hypre") else "ccx"
            msg = (f"{label} failed ({reason_txt}) – solving " + ("with the iterative solver (hypre)" if nxt == "ngs_hypre" else "in CalculiX"))
            log(msg)
            if msg not in warnings:                  # also in Results / the report, not only in the log
                warnings.append(msg)
            fallback = {"from": backend, "reason": reason_txt, "to": nxt}
            reason = f"{label} failed ({reason_txt})"
            backend = nxt
            continue
        res.thermal = thermal
        res.info["solver"]["reason"] = reason
        if fallback is not None:
            res.info["backend_fallback"] = fallback
            res.info["solver"]["fallback"] = fallback
        if prog is not None:
            prog.phase("post", "Processing results")
        _creep_phase(study, prog)
        add_creep_steps(study, mesh, res, workdir, log, on_solver_line, cancel, pending=creep_bg, raw_log=raw_log)
        return res
    job = _ccx_job(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, prog)
    if prog is not None:
        prog.phase("post", "Processing results")
    steps, eps_ip = _ccx_step_results(study, mesh, workdir, log, job)
    info = {"inp_map": job.inpmap, "log": str(job.runs[0].log_path), "backend": "ccx", "ccx_solver": job.ccx_solver}
    info["solver"] = {"backend": "ccx", "engine": f"CalculiX {job.ccx_solver}", "reason": reason, "ndof": 3 * len(mesh.nodes),
                      "nnz": None, "threads": job.threads}
    if fallback is not None:
        info["backend_fallback"] = fallback
        info["solver"]["fallback"] = fallback
    if getattr(study, "cyclic", None) is not None and study.cyclic.enabled:
        from .cyclic import summary as cyclic_summary
        info["cyclic"] = cyclic_summary(study, mesh)
    if job.inpmap.frame:
        from .beams.run import fe_info
        info["frame"] = fe_info(job.inpmap.frame["fe"], job.inpmap.frame["base"])
    if len(job.runs) > 1:
        info["ccx_parallel"] = {"jobs": len(job.runs), "workers": job.workers, "threads": job.threads}
    geom = job.geom
    if geom is not None:
        from .joints.evaluate import joints_info
        info["joints"] = joints_info(geom, job.inpmap.preload_step)
    print_results(study, mesh, steps, info, warnings, log, eps_ip=eps_ip)
    res = Results(mesh, steps, warnings, info, thermal)
    _creep_phase(study, prog, creep_bg)
    add_creep_steps(study, mesh, res, workdir, log, on_solver_line, cancel, pending=creep_bg, raw_log=raw_log)
    return res


class _CcxJob:
    """A finished linear ccx job of the mechanical states (`_ccx_job`): input map, runs, .dat files, .frd, …"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _ccx_job(study, mesh, workdir, warnings, log, on_solver_line, cancel, thermal, prog=None) -> _CcxJob:
    """Writes the .inp of the mechanical states and runs ccx (one job, or the steps as parallel jobs)."""
    name = "job"
    geom, joint_elems = None, None
    if study.active_bolts() or study.active_welds():
        from .joints.evaluate import joint_geometry, required_elements
        geom = joint_geometry(study, mesh)
        warnings.extend(geom["warnings"])
        joint_elems = required_elements(geom)
    temps = None
    if study.has_thermal_loads():
        from .thermal.field import node_temperatures
        tmap = {t.name: t for t in thermal}
        try:
            temps = [node_temperatures(study, mesh, lc, tmap) for lc in study.load_cases]
        except KeyError as e:
            raise StudyError(str(e)) from None
    try:
        inpmap = write_inp(study, mesh, workdir / f"{name}.inp", joint_elems, temps)
    except ValueError as e:              # cylindrical / remote constraint (not a cylinder, shared nodes with another constraint) etc.
        raise StudyError(str(e)) from None
    exe = Path(study.solver.exe) if study.solver.exe else config.CCX_EXE
    from .ccx import parallel
    n_steps = len(study.load_cases) + (1 if inpmap.preload_step else 0)
    workers, threads = parallel.plan(n_steps, 3 * len(mesh.nodes)) if parallel.enabled() else (1, 0)
    ccx_solver = ccx_solver_note(study, warnings, log)
    lines = on_solver_line
    if prog is not None:                             # progress by steps from the ccx output (STEP, factorisation, Job finished)
        from .progress import CcxLines
        lines = prog.lines(on_solver_line, "mech", CcxLines(n_steps))
    if workers > 1:
        # the steps are independent → separate jobs in parallel (ccx factorises in every *STEP, the factorisation with 8 threads
        # scales badly; results equal to 1e-8, see ccx/parallel.py)
        log(f"Solving (CalculiX, {n_steps} cases concurrently: {workers} jobs × {threads} threads)")
        prun = parallel.run_steps(workdir, name, 3 * len(mesh.nodes), exe=exe, on_line=lines, cancel=cancel,
                                  workers=workers, threads=threads)
        runs, dats = prun.runs, [workdir / f"{j}.dat" for j in prun.jobs]
        frd = parallel.read_frd_steps(workdir, prun.jobs)
    else:
        log("Solving (CalculiX)")
        run = run_ccx(workdir, name, exe=exe, on_line=lines, cancel=cancel)
        if not run.finished or run.errors:
            raise CcxError("CalculiX ended with an error:\n  " + "\n  ".join(run.errors or [f"return code {run.returncode}"])
                           + f"\nlog: {run.log_path}")
        runs, dats = [run], [workdir / f"{name}.dat"]
        frd = read_frd(workdir / f"{name}.frd")
    from .ccx.runner import relevant_warnings
    for r in runs:
        warnings.extend(w for w in relevant_warnings(r.warnings) if w not in warnings)
    return _CcxJob(name=name, inpmap=inpmap, runs=runs, dats=dats, frd=frd, exe=exe, geom=geom, temps=temps,
                   workers=workers, threads=threads, ccx_solver=ccx_solver)


def _per_step(what: str, items: list, n: int) -> list:
    """Values read from the .dat by steps: when the input map says they are written, every step must have them."""
    if len(items) < n:
        raise CcxError(f"CalculiX output is incomplete: {what} found for {len(items)} of {n} steps in the .dat")
    return items


def _ccx_step_results(study, mesh, workdir, log, job: _CcxJob) -> tuple[list, list]:
    """StepResult of every ccx step (reactions, equilibrium, joint forces, …) + strains of 3D printing at the integration points."""
    inpmap, dats, frd, geom, temps = job.inpmap, job.dats, job.frd, job.geom, job.temps
    n = len(mesh.nodes)
    steps = []
    ccx_steps = frd.steps()
    cases = list(study.load_cases)
    preload_force = 0.0
    if inpmap.preload_step:
        from .joints.fem import bolt_models, PRELOAD_CASE
        cases.append(LoadCase(PRELOAD_CASE, []))
        preload_force = sum(b.preload for b in bolt_models(study, mesh))
    if len(ccx_steps) != len(cases):
        raise CcxError(f"Number of steps in the results ({len(ccx_steps)}) does not match the study")
    nc = len(cases)
    refs = _per_step("displacements of the remote points (NREF)",
                     [x for d in dats for x in _read_printed(d, "NREF")], nc) if inpmap.ref_nodes else None
    rsup = _per_step("reactions of the rigid remote supports (NRSUP)",
                     [x for d in dats for x in read_remote(d, inpmap)], nc) if inpmap.remote else None
    sig_ip = None
    if inpmap.el_print:
        from .joints.forces import read_el_print
        log("Joint forces")
        sig_ip = _per_step(f"stresses at the integration points ({inpmap.el_print})",
                           [x for d in dats for x in read_el_print(d, inpmap.el_print)], nc)
    sig_sec = None
    if getattr(inpmap, "el_print_section", None):
        from .joints.forces import read_el_print
        sig_sec = _per_step(f"stresses at the integration points ({inpmap.el_print_section})",
                            [x for d in dats for x in read_el_print(d, inpmap.el_print_section)], nc)
    eps_ip = []
    if getattr(inpmap, "el_print_strain", None):
        from .joints.forces import read_el_print_strain
        log("Strains for 3D printing")
        eps_ip = _per_step(f"strains at the integration points ({inpmap.el_print_strain})",
                           [x for d in dats for x in read_el_print_strain(d, inpmap.el_print_strain)], nc)
    masks = constraint_masks(study, mesh)
    frame_U = None
    if inpmap.frame:                                 # frames: displacements and rotations from the .dat + the U1 matrix for the internal forces
        from .beams.fem import read_displacements, u1_matrices
        fe_, base_ = inpmap.frame["fe"], inpmap.frame["base"]
        frame_U = _per_step("displacements of the frame nodes (NFRAME)",
                            [u for d in dats for u in read_displacements(d, base_, len(fe_.xyz))], nc)
        u1_matrices(fe_, workdir / "u1k", job.exe)
    for k, (lc, cs) in enumerate(zip(cases, ccx_steps)):
        U, S, RF = (_nodal(frd, name, cs, n, ncomp) for name, ncomp in (("DISP", 3), ("STRESS", 6), ("FORC", 3)))
        is_pre = k >= len(study.load_cases)
        T = temps[k] if temps is not None and not is_pre else None
        ref = preload_force if is_pre else 0.0
        if T is not None:
            from .thermal.field import force_scale
            ref = max(ref, force_scale(study, mesh, T))
        fr_ = None
        if frame_U is not None:
            from .beams.fem import recover as recover_frame
            fr_ = (inpmap.frame["fe"], inpmap.frame["base"], recover_frame(study, inpmap.frame["fe"], lc, frame_U[k]))
        nodal = nodal_loads(study, mesh, lc) if geom is not None else None    # joint forces need F too
        st = make_step(study, mesh, lc, U, S, RF, masks, refs[k] if refs is not None else {}, ref,
                       rsup[k] if rsup is not None else None, fr_, nodal=nodal)
        st._workdir = Path(workdir)
        st._job = job.name
        st._step_index = k
        if sig_sec is not None:
            st._sig_ip = sig_sec[k]
        if fr_ is not None:
            st.frame = fr_[2]
        if T is not None:
            st.fields["T"] = T
        try:                                         # estimate of the stress error of ccx (STR %, gradient of the principal stress)
            st.fields["stress_error"] = _nodal(frd, "ERROR", cs, n, 1)[:, 0]
        except KeyError:
            pass
        if geom is not None:
            from .joints.evaluate import step_forces, to_json
            st.joints = to_json(step_forces(study, mesh, geom, RF, nodal[0], sig_ip[k] if sig_ip is not None else None))
        steps.append(st)
    return steps, eps_ip


def print_results(study: Study, mesh: Mesh, steps: list, info: dict, warnings: list[str], log,
                  eps_ip: list[dict[int, np.ndarray]] | None = None) -> None:
    """3D printing: safety by Tsai-Wu/Hill into StepResult.fields + summary of the parts into info["printing"]."""
    from .printing import shell_of, CREEP_WARNING
    if shell_of(mesh) is None:
        return
    from .printing.failure import evaluate, part_summary
    log("3D printing: failure criterion")
    for k, st in enumerate(steps):
        st_eps = eps_ip[k] if eps_ip and k < len(eps_ip) else None
        ev = evaluate(study, mesh, st.U, st.fields.get("T"), eps_ip=st_eps)
        st.fields["print_safety"] = ev["safety"]
        st.fields["print_index"] = ev["index"]
        st.printing = {"criterion": ev["criterion"], "parts": ev["parts"]}
    info["printing"] = {"parts": part_summary(study, mesh)}
    if CREEP_WARNING not in warnings:
        warnings.append(CREEP_WARNING)

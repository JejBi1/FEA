"""Manufacturing constraints of the BESO (symmetry, extrusion, overhang) – the glue between the groups, the support rule and the optimiser.

Order inside one iteration (`decide`): filtered sensitivity → average over the decision groups (symmetry orbits / extrusion columns) →
threshold by bisection on the groups → overhang repair (remove the unsupported, refill with the best supported voids) → densities of the
elements. The constraints are pre/post-processing of the density field around ccx; they are not a solver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .groups import (DecisionGroups, Plane, element_sizes, extrusion_labels, mirror_mismatch, planes_from_spec,
                     symmetry_labels, unit_vector)
from .overhang import OverhangRepair, RepairStats, surface_report
from .sections import prism_report

if TYPE_CHECKING:
    from ..mesh import Mesh
    from ..study import Study
    from .spec import TopologyOptSpec

Threshold = Callable[..., "tuple[float, np.ndarray]"]


def selection_nodes(mesh: "Mesh", name: str) -> np.ndarray:
    """Mesh nodes of a face / edge / point selection (unique, sorted)."""
    chunks = []
    for attr in ("surfaces", "edges", "points"):
        group = getattr(mesh, attr, None)
        if group and name in group:
            chunks.append(np.asarray(group[name]).ravel())
    return np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.int64)


def resolve_build_dir(spec_dir, study: "Study | None", mesh: "Mesh") -> np.ndarray:
    """Build direction of the overhang rule: a vector, or 'from_print' = the build direction of a part of the mesh from the 3D printing settings."""
    if not isinstance(spec_dir, str):
        return unit_vector(spec_dir)
    printing = getattr(study, "printing", None)
    dirs = []
    for name in mesh.part_names:
        sp = printing.parts.get(name) if printing is not None else None
        if sp is not None:
            dirs.append(unit_vector(sp.build_dir))
    if not dirs:
        raise ValueError("overhang.build_dir 'from_print': no part of the model has 3D printing settings (set the build direction in the "
                         "3D printing step or enter the vector)")
    if any(float(np.linalg.norm(d - dirs[0])) > 1e-6 for d in dirs):
        raise ValueError("overhang.build_dir 'from_print': the parts have different build directions – enter the vector")
    return dirs[0]


class ManufacturingConstraints:
    """Constraints of one optimisation on one mesh; `decide` replaces the threshold step of the optimiser."""

    def __init__(self, mesh: "Mesh", spec: "TopologyOptSpec", centroids: np.ndarray, volumes: np.ndarray, non_design: np.ndarray,
                 study: "Study | None" = None) -> None:
        self.spec = spec
        self.centroids = np.asarray(centroids, dtype=float)
        self.volumes = np.asarray(volumes, dtype=float)
        self.non_design = np.asarray(non_design, dtype=bool)
        self.warnings: list[str] = []
        self.sym_info: dict | None = None
        self.ext_info: dict | None = None
        self.overhang_info: dict | None = None
        self.last_stats = RepairStats()
        self.pruned_elements = 0                       # removed by the final strict check (`prune`)
        self.pruned_volume = 0.0
        self.repair: OverhangRepair | None = None
        self.build_dir: np.ndarray | None = None
        nodes = np.asarray(mesh.nodes, dtype=float)
        centre = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0)) if len(nodes) else np.zeros(3)
        self.planes: list[Plane] = planes_from_spec(spec.symmetry, centre)
        eps = 1e-9 * max(float(np.ptp(self.centroids, axis=0).max()) if len(self.centroids) else 0.0, 1.0)
        for k, pl in enumerate(self.planes, start=1):          # a plane without elements on its negative side has no mirror images to tie
            if len(self.centroids) and not bool(np.any(pl.distance(self.centroids) < -eps)):
                self.warnings.append(f"Symmetry plane {k} has no elements on its negative side (it lies outside or on the boundary of the design "
                                     "domain): it has no effect")

        labels = None
        if spec.extrusion is not None:
            labels, self.ext_info = extrusion_labels(self.centroids, self.volumes, spec.extrusion["direction"], self.planes,
                                                     spec.extrusion.get("bin_size"))
            if self.planes:
                self.ext_info["note"] = "mirror planes are applied to the columns; planes perpendicular to the columns hold for a prismatic domain"
        elif self.planes:
            labels, self.sym_info = symmetry_labels(self.centroids, self.volumes, self.planes)
            if self.sym_info["unmatched"]:
                self.warnings.append(
                    f"Symmetry: {self.sym_info['unmatched']} of {len(self.centroids)} elements have no mirror image within "
                    f"{self.sym_info['tolerance_in_h']} x the element size (the mesh or the design domain is not symmetric there) – "
                    "they are decided on their own")
        self.groups = DecisionGroups(labels, self.volumes, self.non_design) if labels is not None else None
        if self.groups is not None and self.groups.n == len(self.centroids):
            self.groups = None                                 # every group is a single element: nothing to average

        if spec.overhang is not None:
            ov = spec.overhang
            self.build_dir = resolve_build_dir(ov["build_dir"], study, mesh)
            base = ov.get("base", "min")
            base_nodes = None
            if base != "min":
                names = selection_nodes(mesh, base)
                if len(names) == 0:
                    raise ValueError(f"overhang.base: the selection '{base}' has no nodes in the mesh (it must be a face, edge or point selection of the model)")
                base_nodes = np.zeros(len(nodes), dtype=bool)
                base_nodes[names] = True
            self.repair = OverhangRepair(nodes, mesh.tets, self.volumes, self.build_dir, ov["angle_deg"], base_nodes, self.non_design,
                                         None if self.groups is None else self.groups.labels)
            base_el = int(((self.repair.inc @ self.repair.base.astype(np.int32)) > 0).sum())
            self.overhang_info = {"build_dir": [float(x) for x in self.build_dir], "angle_deg": float(ov["angle_deg"]), "base": base,
                                  "base_elements": base_el, "approximate": True}
            if base_el == 0:
                raise ValueError("overhang: no element touches the build plate (check the build direction and the base)")

        # arrays at the level of decision units (groups, or elements without groups)
        if self.groups is not None:
            self.unit_volumes, self.unit_non_design = self.groups.volumes, self.groups.non_design
        else:
            self.unit_volumes, self.unit_non_design = self.volumes, self.non_design

    # ------------------------------------------------------------------ iteration
    @property
    def names(self) -> list[str]:
        return [n for n, on in (("symmetry", bool(self.planes)), ("extrusion", self.spec.extrusion is not None),
                                ("overhang", self.spec.overhang is not None)) if on]

    def decide(self, filtered: np.ndarray, target_volume: float, min_density: float, threshold: Threshold,
               current_rho: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """Densities of the elements (1 / min_density) for the target volume: group average → threshold → overhang repair.
        current_rho = the densities the analysis ran with (the full design when omitted); the supporters of what is solid now are ranked up."""
        sens = self.groups.mean(filtered) if self.groups is not None else np.asarray(filtered, dtype=float)
        if self.repair is not None:        # the supporters of valuable units must not be cheaper than what they carry
            if current_rho is None:
                now = np.ones(len(self.unit_volumes), dtype=bool)
            else:
                solid_now = np.asarray(current_rho) > 0.5
                now = (self.groups.mean(solid_now.astype(float)) > 0.5) if self.groups is not None else solid_now
            sens = self.repair.support_priority(sens, now)
        th, rho_u = threshold(sens, volumes=self.unit_volumes, target_volume=target_volume,
                              non_design_mask=self.unit_non_design, min_density=min_density)
        solid = np.asarray(rho_u) > 0.5
        self.last_stats = RepairStats()
        if self.repair is not None:
            solid, self.last_stats = self.repair.repair(solid, sens, self.unit_volumes, target_volume, self.unit_non_design,
                                                        None if self.groups is None else self.groups.counts)
        solid_el = self.groups.expand(solid) if self.groups is not None else solid
        return th, np.where(solid_el, 1.0, float(min_density))

    def prune(self, rho: np.ndarray) -> tuple[np.ndarray, int]:
        """Final strict check of the overhang rule (support through material): densities without the units that fail it, number of elements removed.
        Without the overhang constraint nothing changes."""
        if self.repair is None:
            return np.asarray(rho, dtype=float), 0
        solid = np.asarray(rho) > 0.5
        sol_u = self.groups.mean(solid.astype(float)) > 0.5 if self.groups is not None else solid
        out_u, stats = self.repair.prune(sol_u, self.unit_non_design, None if self.groups is None else self.groups.counts)
        out_el = self.groups.expand(out_u) if self.groups is not None else out_u
        self.pruned_elements = int(stats.removed)
        self.pruned_volume = float(self.volumes[solid & ~out_el].sum())
        return np.where(out_el | self.non_design, 1.0, float(self.spec.min_density)), int(stats.removed)

    def spread(self, rho: np.ndarray) -> float:
        """Largest difference of the densities inside one decision group (0 = the constraint holds exactly)."""
        return self.groups.spread(rho) if self.groups is not None else 0.0

    # ------------------------------------------------------------------ result
    def summary(self, mesh: "Mesh", rho: np.ndarray, reference_rho: np.ndarray | None = None, stats: RepairStats | None = None) -> dict:
        """Constraint block of the result: statistics, the exact checks and the independent measurement on the exported surface;
        reference_rho = densities of the unconstrained run for the comparison; stats = overhang repair statistics of the step that produced the
        KEPT design (the last step of the run when omitted)."""
        stats = self.last_stats if stats is None else stats
        from .export import topo_surface
        solid = np.asarray(rho) > 0.5
        ref_solid = None if reference_rho is None else np.asarray(reference_rho) > 0.5
        out: dict = {"active": self.names, "warnings": list(self.warnings)}
        if self.planes:
            sym = dict(self.sym_info or {})
            sym["planes"] = [p.to_dict() for p in self.planes]
            sym.update(mirror_mismatch(solid, self.centroids, self.volumes, self.planes))
            sym["max_group_spread"] = self.spread(rho)
            if ref_solid is not None:
                sym["reference_mismatch_volume_fraction"] = mirror_mismatch(ref_solid, self.centroids, self.volumes, self.planes)["mismatch_volume_fraction"]
            out["symmetry"] = sym
        if self.ext_info is not None:
            ext = dict(self.ext_info)
            ext["max_column_spread"] = self.spread(rho)
            if reference_rho is not None:
                ext["reference_column_spread"] = self.spread(reference_rho)
            verts, faces = topo_surface(mesh, np.where(solid, 1.0, 0.0), 0.5)          # independent check on the exported surface
            ext["prism"] = prism_report(verts, faces, ext["direction"])
            if ref_solid is not None:
                rv, rf = topo_surface(mesh, np.where(ref_solid, 1.0, 0.0), 0.5)
                ext["reference_prism"] = prism_report(rv, rf, ext["direction"])
            out["extrusion"] = ext
        if self.overhang_info is not None and self.build_dir is not None:
            ov = dict(self.overhang_info)
            ov["removed_last_iteration"] = int(stats.removed)          # the step that produced the kept design
            ov["added_last_iteration"] = int(stats.added)
            ov["deficit_left_mm3"] = float(stats.deficit_left)
            ov["cascade_passes_last_iteration"] = int(stats.cascade)
            vmin = float(np.min(self.unit_volumes)) if len(self.unit_volumes) else 0.0
            ov["volume_deficit"] = bool(stats.deficit_left > 0.5 * vmin)         # more than half of the smallest decision unit is missing
            if ov["volume_deficit"]:
                out["warnings"].append(f"Overhang: the volume taken away from unsupported elements could not be refilled with supported ones, "
                                       f"{stats.deficit_left:.4g} mm³ ({100 * stats.deficit_left / max(float(self.volumes.sum()), 1e-300):.1f} % of the domain) "
                                       "is missing from the target volume")
            fixed = self.non_design
            if self.repair is not None:        # non-design elements are anchors of the rule; how many of them hang in the air (strict chain to the plate)?
                sol_u = self.groups.mean(solid.astype(float)) > 0.5 if self.groups is not None else solid
                ov["unsupported_non_design_elements"] = int((fixed & ~self.repair.supported_elements(sol_u, strict=True)).sum())
            kw = {"base_height": float((np.asarray(mesh.nodes) @ self.build_dir).min()),
                  "base_tol": 0.2 * float(np.mean(element_sizes(self.volumes)))}
            verts, faces = topo_surface(mesh, np.where(solid, 1.0, 0.0), 0.5)
            ov["surface"] = surface_report(verts, faces, self.build_dir, self.overhang_info["angle_deg"], **kw)
            if ref_solid is not None:
                rv, rf = topo_surface(mesh, np.where(ref_solid, 1.0, 0.0), 0.5)
                ov["reference_surface"] = surface_report(rv, rf, self.build_dir, self.overhang_info["angle_deg"], **kw)
            out["overhang"] = ov
        return out

"""View refresh of the main window: status line, redraw of the model, loads and selection sets in 3D, the study tree and the part
tables with their context menus, step colours and part visibility.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QInputDialog, QTreeWidgetItem, QTableWidgetItem, QMenu

from fea_core import config
from fea_core.geometry import bbox
from fea_core.study import (Fixed, Force, Pressure, Gravity, RemoteLoad, CylindricalSupport, BearingLoad, ThermalLoad,
                            RemoteSupport, Rotation, ElasticSupport)
from fea_core.i18n import decimal, tr

from . import dyn_results as dynr, submodel_dialog as smd
from .joints_panel import util_color
from .eigen_gui import mode_label
from .thermal_panel import describe_bc, describe_case
from .widgets import C_HEX, sci

from .main_defs import (APP_NAME, STAGES, STAGE, GREEN, ORANGE, RED, GREY, C_CONSTRAINT, C_FORCE, C_PRESSURE, C_REFINE,
                        FRAME_ONLY_OFF, C_EDGE, C_ELASTIC, fmt, dot_icon, _batched, msg_via_main as msg)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class RefreshMixin(_Base):
    """View refresh of the main window: status line, redraw of the model, loads and selection sets in 3D, the study tree and
    the part tables with their context menus, step colours and part visibility."""

    def update_model_status(self):
        """Status bar on the right: equilibrium (worst state), number of elements, calculation time, units."""
        if not hasattr(self, "st_model"):
            return
        out, state = [], ""
        fs = getattr(self, "flow_state", None)          # readiness check (refresh_flow)
        if self.study is not None and fs is not None:
            errs, warns = fs[1], fs[2]
            out.append(f"× {tr('errors')}: {len(errs)}" if errs else
                       (f"! {tr('warning')}: {len(warns)}" if warns else "√ " + tr("Readiness check")))
        res = self.results
        imb = [s.equilibrium["imbalance"] for s in (res.steps if res is not None else [])
               if s.equilibrium and "imbalance" in s.equilibrium]
        if imb:
            worst = max(imb)
            ok = worst <= config.EQUILIBRIUM_TOL
            out.append(("√ " if ok else "× ") + "Equilibrium " + sci(worst))
            state = "ok" if ok else "fail"
        mesh = res.mesh if res is not None and getattr(res, "mesh", None) is not None else getattr(self, "mesh", None)
        n_el = len(mesh.tets) if mesh is not None and getattr(mesh, "tets", None) is not None else 0
        if n_el:
            out.append(f"{fmt(n_el)} elements")
        dt = getattr(res, "_gui_solve_s", None) if res is not None else None
        if dt is not None:
            out.append(decimal(f"{dt:.1f} s"))
        out.append("mm · N · MPa")
        self.st_model.setText(" · ".join(out))
        self.st_model.setProperty("state", state)
        self.st_model.style().unpolish(self.st_model)
        self.st_model.style().polish(self.st_model)

    def update_status(self):
        n = len(self.view.selected_faces())
        out = []
        if n:
            area = sum(s.area for s in self.selected_selectors())
            out.append(f"Faces selected: {n}   A = {fmt(area)} mm²")
        ne = len(self.view.selected_edges())
        if ne:
            length = sum(self.edge_prop(pi, ei)["length"] for pi, ei in self.view.selected_edges())
            out.append(f"edges: {ne}   L = {fmt(length)} mm")
        npt = len(self.view.selected_points())
        if npt:
            out.append(f"points: {npt}")
        self.st_sel.setText("   |   ".join(out))
        if hasattr(self, "ld_target") and STAGES[getattr(self, "stage", 0)] == "Loads":
            self.on_ld_type(0)

    # ================================================================== restoring the view
    @_batched
    def refresh_all(self):
        self._filling = True
        try:
            check_result = self.check() if self.study is not None else None
            self.refresh_tree(check_result)
            self.refresh_flow(check_result)
            if self.study is None:
                return
            self.apply_ghost()
            self.refresh_model_info()
            self.refresh_materials()
            for attr in ("print_panel", "frame_panel"):
                if self._built(attr) is not None:
                    self._built(attr).refresh()
            self.joints_panel.refresh()
            self.twod_box.refresh()
            self.shell_box.refresh()
            self.cyclic_box.refresh()
            self.thermal_panel.refresh()
            for attr in ("contact_panel", "dynamics_panel", "motion_panel"):
                if self._built(attr) is not None:
                    self._built(attr).refresh()
            self.refresh_thermal_load_form()
            self.con_list.clear()
            for c in self.study.constraints:
                self.con_list.addItem(self.describe_constraint(c))
            cur = self.lc_combo.currentIndex()
            self.lc_combo.blockSignals(True)
            self.lc_combo.clear()
            self.lc_combo.addItems([lc.name for lc in self.study.load_cases])
            self.lc_combo.setCurrentIndex(min(max(cur, 0), len(self.study.load_cases) - 1))
            self.lc_combo.blockSignals(False)
            self.refresh_loads(refresh_view=False)
            ms = self.study.mesh
            self.ms_size.setValue(ms.size)
            self.ms_min.setValue(ms.min_size or 0.0)
            self.ms_curv.setValue(ms.curvature_points)
            self.ref_list.clear()
            for r in ms.local:
                self.ref_list.addItem(self.describe_refinement(r))
            if self.mesh is not None:
                r = self.mesh.report
                ties = r.get("ties") or []
                self.mesh_info.setText(f"{fmt(r.get('nodes', len(self.mesh.nodes)))} nodes, {fmt(r.get('elements', len(self.mesh.tets)))} C3D10 elements, min. scaled Jacobian {fmt(r.get('min_scaled_jacobian'), 3)}"
                                       + (f"; *TIE connections: {len(ties)} (Joints step)" if ties else ""))
            else:
                self.mesh_info.setText("The mesh has not been generated yet (it is also generated during the analysis).")
            self._show_stored_solver()
            self.backend_combo.blockSignals(True)
            bi = self.backend_combo.findData(self.study.solver.backend)
            if bi < 0:                                  # unknown solver (file from another version) → automatically
                self.log_msg(f"Unknown solver “{self.study.solver.backend}” in the study – set to Automatic")
                self.study.solver.backend = "auto"
                bi = max(0, self.backend_combo.findData("auto"))
                check_result = None                     # the study changed: refresh_ready checks it again
            self.backend_combo.setCurrentIndex(bi)
            self.backend_combo.blockSignals(False)
            self.modal_engine_combo.blockSignals(True)
            self.modal_engine_combo.setCurrentIndex(max(0, self.modal_engine_combo.findData(getattr(self.study.solver, "modal_engine", "auto"))))
            self.modal_engine_combo.blockSignals(False)
            self._backend_badge()                       # signals are blocked → mark the Solver section manually
            self.eigen_box.refresh()
            cur = self.r_step.currentData()
            self.r_step.blockSignals(True)
            self.r_step.clear()
            if self.results is not None:            # mechanical states first (indices as results.steps), then heat
                for i, s in enumerate(self.results.steps):
                    self.r_step.addItem(s.name, f"step:{i}")
                if hasattr(self.study, "combinations") and self.study.combinations:
                    for j, comb in enumerate(self.study.combinations):
                        prefix = "Envelope" if comb.envelope else "Combination"
                        self.r_step.addItem(f"{prefix}: {comb.name}", f"combination:{j}")
                for j, t in enumerate(self.results.thermal):
                    self.r_step.addItem(f"Thermal: {t.name}", f"thermal:{j}")
                if self.results.twod is not None:          # 2D calculation by a section
                    for j, s2 in enumerate(self.results.twod.steps):
                        self.r_step.addItem(f"2D: {s2['name']}", f"twod:{j}")
                if self.results.shell is not None:         # shell model
                    for j, s3 in enumerate(self.results.shell.steps):
                        self.r_step.addItem(f"Shells: {s3.name}", f"shell:{j}")
                for txt, data in smd.r_step_items(self.results):   # submodels ("sub:k")
                    self.r_step.addItem(txt, data)
                for j, ms in enumerate(self.results.modes):  # natural and buckling mode shapes (eigen.ModeSet)
                    for k in range(len(ms.values)):
                        self.r_step.addItem(mode_label(ms, k), f"mode:{j}:{k}")
                for txt, data in dynr.items(self.results):   # dynamika (F8): pole „dyn:druh:k“
                    self.r_step.addItem(txt, data)
                k = self.r_step.findData(cur) if cur is not None else -1
                self.r_step.setCurrentIndex(max(k, 0))
            self.r_step.blockSignals(False)
            if hasattr(self, "r_part"):
                cur_part = self.r_part.currentText()
                self.r_part.blockSignals(True)
                self.r_part.clear()
                self.r_part.addItem("All parts", None)
                if self.results is not None and self.results.mesh is not None:
                    for pname in self.results.mesh.part_names:
                        self.r_part.addItem(pname, pname)
                idx = self.r_part.findText(cur_part)
                self.r_part.setCurrentIndex(max(idx, 0))
                self.r_part.blockSignals(False)
            if self.results is not None:
                self._sync_result_controls()
            self.modes_table.fill(self.results)
            self.res_tabs.setTabVisible(2, bool(self.results is not None and self.results.modes))
            self.res_tabs.setTabVisible(3, bool(self.results is not None and any(s.frame for s in self.results.steps)))
            has_dyn = self.results is not None and getattr(self.results, "dynamics", None) is not None
            self.res_tabs.setTabVisible(self.TAB_DYN, has_dyn)
            if has_dyn:
                self.dyn_box.fill(self.results)
            self.res_tabs.setTabVisible(self.TAB_CONTACT, bool(self.results is not None and any(
                s.contacts or s.partial or s.curves for s in self.results.steps)))
            self.refresh_sets()
            self.refresh_ready(check_result)
            self.update_status()
        finally:
            self._filling = False
            if hasattr(self, "table_dock"):
                try:
                    if STAGES[getattr(self, "stage", 0)] == "Results" or self.results is None:
                        self.res_overview.refresh()      # Assessment card (state summary only when visible)
                    self.table_dock.refresh()            # table of the current step in the bottom dock
                    self.update_model_status()
                    self.apply_part_colors()
                    self.update_view_info()
                    self.props_card.schedule()
                except Exception as e:  # noqa: BLE001 – the table is only an overview, must not block the window
                    self.warn("tables", f"The table in the bottom dock could not be refreshed: {e}")

    def refresh_loads(self, refresh_view: bool = True):
        self.load_list.clear()
        lc = self.current_lc()
        if lc:
            for l in lc.loads:
                self.load_list.addItem(self.describe_load(l))
        if refresh_view:
            self.refresh_sets()

    def refresh_sets(self):
        """Colour sets of the constraint and load faces of the current state in 3D (in the Joints step bolts and welds)."""
        if self.study is None:
            return
        v = self.view
        v.clear_sets()
        stage = STAGES[self.stage]
        if stage not in ("Joints", "Contacts", "Constraints", "Motion", "Heat", "Loads", "Dynamics", "Mesh", "Solve"):
            v.show_sets(False)
            return
        if stage != "Results" or self.results is None:              # frame system in all steps with a model
            self.frame_panel.show_3d(self.current_lc() if stage == "Loads" else None)
        if stage == "Solve":
            self.twod_box.draw_preview()
        if STAGES[self.stage] == "Joints":
            self.joints_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Heat":          # boundary conditions of the selected thermal state
            self.thermal_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Contacts":       # contact pair faces (colour by type)
            self.contact_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Motion":          # joints (axes), spring elements, mechanism loads
            self.motion_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Dynamics":       # added mass and output probes
            self.dynamics_panel.show_3d()
            v.show_sets(True)
            return
        L = 0.12 * v.model_size()
        for c in self.study.constraints:
            v.set_face_set(f"c:{c.selection}", self.sel_faces.get(c.selection, []), C_CONSTRAINT,
                           label=c.selection)
            if isinstance(c, CylindricalSupport):
                segs = []                             # axis of each selection hole (extended beyond the face)
                for p0, d, r, t0, t1, _ in self.cylinders_of(self.sel_faces.get(c.selection, [])):
                    m = 0.25 * (t1 - t0) + r
                    segs.append((p0 + (t0 - m) * d, p0 + (t1 + m) * d))
                if segs:
                    v.add_lines(f"c:{c.selection}", segs, C_CONSTRAINT, 2.5)
            elif isinstance(c, RemoteSupport):        # point (cross) + lines to the faces
                pt = np.asarray(c.point, float)
                k = 0.04 * L
                seg = [(pt - k * e, pt + k * e) for e in np.eye(3)]
                for pi, fi in self.sel_faces.get(c.selection, [])[:12]:
                    p = self.face_prop(pi, fi)
                    seg.append((pt, np.asarray(self.arrow_point(p), float)))
                v.add_lines(f"c:{c.selection}", seg, C_CONSTRAINT, 2.0, label=f"point {c.selection}")
            elif isinstance(c, ElasticSupport):       # springs from the face outwards (normal; "all directions" + cross)
                seg = []
                for pi, fi in self.sel_faces.get(c.selection, [])[:12]:
                    p = self.face_prop(pi, fi)
                    seg += self.spring_segments(self.arrow_point(p), p["normal"], 0.35 * L, c.direction == "all")
                v.add_lines(f"c:{c.selection}", seg, C_ELASTIC, 2.0)
        if stage == "Constraints":                    # cyclic symmetry: cut faces and axis
            from .cyclic_gui import show_3d as show_cyclic
            show_cyclic(self)
        lc = self.current_lc()
        if lc:
            from fea_core import csys as csys_mod
            from fea_core.loads import pressure_value

            def gvec(l, at):
                """Global direction of a force at a point (local / cylindrical system)."""
                try:
                    return csys_mod.field_global(self.study, getattr(l, "csys", None), l.vector,
                                                 np.asarray(at, float)[None, :])[0]
                except ValueError:
                    return np.asarray(l.vector, float)
            for l in lc.loads:
                sel = getattr(l, "selection", None)
                if isinstance(l, Rotation):              # rotation axis across the model
                    a = np.asarray(l.axis, float) / max(np.linalg.norm(l.axis), 1e-30)
                    p0 = np.asarray(l.point, float)
                    bb = np.array([bbox(p.shape) for p in self.parts])
                    c0 = 0.5 * (bb[:, :3].min(axis=0) + bb[:, 3:].max(axis=0))
                    t0 = float((c0 - p0) @ a)
                    half = 0.6 * v.model_size()
                    v.add_lines("l:rotation", [(p0 + (t0 - half) * a, p0 + (t0 + half) * a)], C_EDGE, 2.5,
                                label=self.describe_load(l))
                    v.add_arrows("l:rotation", [(p0 + (t0 + half) * a, a, 0.4 * L)], C_EDGE, double=True)
                    continue
                if not sel:
                    continue
                entity = self.study.selections[sel].entity if sel in self.study.selections else "faces"
                key = f"l:{sel}"
                if entity == "edges":                      # force on an edge: arrows along the edges
                    pts = []
                    for pi, ei in self.sel_edges.get(sel, []):
                        e = self.edge_prop(pi, ei)
                        pts += [e["start"], e["point"], e["end"]]
                    if pts:
                        v.add_lines(key, [(pts[i], pts[i + 1]) for i in range(0, len(pts), 3)]
                                    + [(pts[i + 1], pts[i + 2]) for i in range(0, len(pts), 3)], C_EDGE, 4.0,
                                    label=self.describe_load(l))
                        if np.linalg.norm(l.vector) > 0:
                            v.add_arrows(key, [(q, gvec(l, q), 0.8 * L) for q in pts[:24]], C_FORCE)
                    continue
                if entity == "points":                     # force to a point
                    pts = [q for _, q in self.sel_points.get(sel, [])]
                    if pts and np.linalg.norm(l.vector) > 0:
                        v.add_arrows(key, [(q, gvec(l, q), L) for q in pts], C_FORCE)
                    if pts:                                # cross at the point (with a label at the first one)
                        k = 0.04 * L
                        v.add_lines(key, [(np.add(q, -k * e), np.add(q, k * e)) for q in pts for e in np.eye(3)],
                                    C_FORCE, 2.0, label=self.describe_load(l))
                    continue
                faces = self.sel_faces.get(sel, [])
                col = C_PRESSURE if isinstance(l, Pressure) else C_FORCE
                v.set_face_set(key, faces, col, label=self.describe_load(l))
                props = [self.face_prop(pi, fi) for pi, fi in faces[:8]]
                if isinstance(l, Force) and np.linalg.norm(l.vector) > 0:
                    arr = []
                    cs = self.study.coordinate_systems.get(l.csys) if l.csys else None
                    if cs is not None and cs.kind == "cylindrical":    # arrows at points far from the axis
                        for pi, fi in faces[:4]:
                            for q, nq in self.face_points_off_axis(pi, fi, cs, 6):
                                d = gvec(l, q)
                                arr.append((v.place_on_face(q, d, nq, 0.7 * L), d, 0.7 * L))
                    else:
                        for p in props:
                            q = self.arrow_point(p)
                            d = gvec(l, q)
                            arr.append((v.place_on_face(q, d, p["normal"], L), d, L))
                    v.add_arrows(key, arr, col)
                elif isinstance(l, Pressure) and l.variable:
                    arr = []
                    for p in props:
                        q = self.arrow_point(p)
                        pv = float(pressure_value(l, np.asarray(q, float)[None, :])[0])
                        if pv != 0:
                            s_ = np.sign(pv)
                            arr.append((v.place_on_face(q, -s_ * np.asarray(p["normal"]), p["normal"], L),
                                        -s_ * np.asarray(p["normal"]), L))
                    v.add_arrows(key, arr, col)
                elif isinstance(l, Pressure) and l.value != 0:
                    s = np.sign(l.value)       # positive pressure points into the material
                    v.add_arrows(key, [(v.place_on_face(self.arrow_point(p), -s * p["normal"], p["normal"], L),
                                        -s * p["normal"], L) for p in props], col)
                elif isinstance(l, BearingLoad) and np.linalg.norm(l.vector) > 0:
                    v.add_arrows(key, self.bearing_arrows(faces, l.vector, L), col)
                elif isinstance(l, RemoteLoad):
                    if np.linalg.norm(l.force) > 0:
                        v.add_arrows(key, [(l.point, l.force, 1.3 * L)], col)
                    if np.linalg.norm(l.moment) > 0:
                        v.add_arrows(key, [(l.point, l.moment, 1.3 * L)], col, double=True)
        if STAGES[self.stage] == "Loads" and lc is not None:      # local systems used in the state
            used = {getattr(l, "csys", None) for l in lc.loads} - {None}
            for name in used:
                c = self.study.coordinate_systems.get(name)
                if c is None:
                    continue
                from fea_core.csys import matrix
                R = matrix(c)
                o = np.asarray(c.origin, float)
                k = 0.5 * L
                v.add_lines(f"cs:{name}", [(o, o + k * R[i]) for i in range(3)], (90, 90, 90), 2.0,
                            label=f"{name} ({'r, θ, z' if c.kind == 'cylindrical' else 'x, y, z'})")
        if STAGES[self.stage] == "Mesh":
            for k, r in enumerate(self.study.mesh.local):
                if r.get("selection"):
                    v.set_face_set(f"r:{r['selection']}", self.sel_faces.get(r["selection"], []), C_REFINE)
                elif r.get("center") is not None:          # sphere: three circles
                    c, R = np.asarray(r["center"], float), float(r.get("radius", 0.0))
                    t = np.linspace(0, 2 * np.pi, 49)
                    seg = []
                    for a, b in ((0, 1), (1, 2), (0, 2)):
                        P = np.tile(c, (len(t), 1))
                        P[:, a] += R * np.cos(t)
                        P[:, b] += R * np.sin(t)
                        seg += [(P[i], P[i + 1]) for i in range(len(t) - 1)]
                    v.add_lines(f"r:ball{k}", seg, C_REFINE, 2.0, label=f"refinement {fmt(float(r['size']))} mm")
        v.show_sets(STAGES[self.stage] in ("Constraints", "Loads", "Solve") or
                    (STAGES[self.stage] == "Mesh" and self.mesh is None))

    @staticmethod
    def spring_segments(base, normal, L, all_dirs=False) -> list[tuple]:
        """Spring marker: zigzag from a point of the face in the direction of the outward normal and a "ground" dash at the end."""
        n = np.asarray(normal, float)
        n = n / max(np.linalg.norm(n), 1e-30)
        e = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e /= np.linalg.norm(e)
        b = np.asarray(base, float)
        pts = [b, b + 0.2 * L * n]
        for k in range(6):
            pts.append(b + (0.2 + 0.1 * (k + 0.5)) * L * n + (0.12 if k % 2 == 0 else -0.12) * L * e)
        pts += [b + 0.8 * L * n, b + L * n]
        seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        tip = b + L * n
        seg.append((tip - 0.2 * L * e, tip + 0.2 * L * e))
        if all_dirs:                                  # springs also in the plane of the face
            e2 = np.cross(n, e)
            seg += [(b - 0.25 * L * e, b + 0.25 * L * e), (b - 0.25 * L * e2, b + 0.25 * L * e2)]
        return seg

    def face_points_off_axis(self, pi, fi, cs, n: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
        """Face points (triangulation nodes) far from the axis of a cylindrical system, spread out (the furthest points);
        the face normal at them roughly from the triangulation. For force arrows in a cylindrical system (there is no direction on the axis).
        """
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        from fea_core.csys import matrix
        face = self.view.part_faces[pi][fi]
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            p = self.face_prop(pi, fi)
            return [(np.asarray(p["point"], float), np.asarray(p["normal"], float))]
        trsf = loc.Transformation()
        X = np.array([[q.X(), q.Y(), q.Z()] for q in (tri.Node(i).Transformed(trsf)
                                                       for i in range(1, tri.NbNodes() + 1))])
        ez = matrix(cs)[2]
        d = X - np.asarray(cs.origin, float)
        r = np.linalg.norm(d - np.outer(d @ ez, ez), axis=1)
        cand = np.nonzero(r >= 0.6 * r.max())[0]
        pick = [int(cand[np.argmax(r[cand])])]
        dmin = np.linalg.norm(X[cand] - X[pick[0]], axis=1)
        for _ in range(min(n, len(cand)) - 1):
            pick.append(int(cand[np.argmax(dmin)]))
            dmin = np.minimum(dmin, np.linalg.norm(X[cand] - X[pick[-1]], axis=1))
        nrm = np.asarray(self.face_prop(pi, fi)["normal"], float)
        out = []
        for k in pick:                       # normal: planar face → same; cylinder → radial (by type)
            nq = nrm
            if self.face_prop(pi, fi)["type"] != "plane":
                rv = d[k] - (d[k] @ ez) * ez
                nq = rv / max(np.linalg.norm(rv), 1e-30) * (1 if rv @ (X[k] - np.asarray(cs.origin)) >= 0 else -1)
            out.append((X[k], nq))
        return out

    def cylinder_of(self, faces):
        """(point on the axis, direction, radius, t_min, t_max, hole?) of the first group of coaxial cylindrical CAD faces of the selection
        (see cylinders_of); None if the selection has no cylinder.
        """
        groups = self.cylinders_of(faces)
        return groups[0] if groups else None

    def cylinders_of(self, faces) -> list[tuple]:
        """Groups of coaxial cylindrical faces of the selection with the same radius (like fea_core.cylinder.selection_axes – a selection
        with several non-coaxial holes has the axis of each hole): [(point on the axis, direction, radius, t_min, t_max, hole?)], t = position
        along the axis across the faces of the group (from the bounding boxes).
        """
        from fea_core.geometry import cylinder_axis
        groups: list[list] = []                  # [axis (p0, d, r), props of the first face, [t…]]
        for pi, fi in faces:
            ax = cylinder_axis(self.view.part_faces[pi][fi])
            if ax is None:
                continue
            p, dd, rr = np.asarray(ax[0], float), np.asarray(ax[1], float), float(ax[2])
            tol = 1e-3 * max(rr, 1.0)
            g = None
            for cand in groups:
                p0, d, r = cand[0]
                q = p - p0
                if abs(abs(float(dd @ d)) - 1.0) < 1e-6 and np.linalg.norm(q - (q @ d) * d) < tol \
                        and abs(rr - r) < tol:
                    g = cand
                    break
            if g is None:
                g = [(p, dd, rr), self.face_prop(pi, fi), []]
                groups.append(g)
            bb = np.asarray(self.face_prop(pi, fi)["bbox"])
            corners = np.array([[bb[i], bb[j], bb[k]] for i in (0, 3) for j in (1, 4) for k in (2, 5)])
            g[2].append((corners - g[0][0]) @ g[0][1])
        out = []
        for (p0, d, r), props, ts in groups:
            ts = np.concatenate(ts)
            q = np.asarray(props["point"], float) - p0
            er = q - (q @ d) * d
            hole = float(np.asarray(props["normal"]) @ er) < 0
            out.append((p0, d, r, float(ts.min()), float(ts.max()), hole))
        return out

    def bearing_arrows(self, faces, F, L):
        """Bearing force arrows: pressure proportional to cos θ on the facing half of the hole (for a pin on the opposite half)."""
        cyl = self.cylinder_of(faces)
        if cyl is None:
            return []
        p0, d, r, t0, t1, hole = cyl
        F = np.asarray(F, float)
        Fa = (F @ d) * d
        Fr = F - Fa
        c = p0 + 0.5 * (t0 + t1) * d
        out = []
        Lb = min(L, 0.9 * r) if hole else L
        s = 1.0 if hole else -1.0
        if np.linalg.norm(Fr) > 1e-12 * np.linalg.norm(F):
            fh = Fr / np.linalg.norm(Fr)
            e2 = np.cross(d, fh)
            for th in np.radians(np.linspace(-72, 72, 7)):
                w = np.cos(th) * fh + np.sin(th) * e2
                out.append((c + s * r * w, w, Lb * max(np.cos(th), 0.25)))
        else:
            fh = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            fh /= np.linalg.norm(fh)
        if np.linalg.norm(Fa) > 1e-9 * np.linalg.norm(F):
            out.append((c + s * r * fh, Fa, Lb))
        return out

    @staticmethod
    def describe_constraint(c):
        if isinstance(c, Fixed):
            return f"{c.selection}"
        if isinstance(c, CylindricalSupport):
            comps = [n for n, on in (("radially", c.radial), ("tangentially", c.tangential), ("axially", c.axial)) if on]
            return f"{c.selection}: cylindrical – {', '.join(comps) or 'nothing constrained'}"
        if isinstance(c, RemoteSupport):
            from fea_core.remote import dofs_text
            return (f"{c.selection}: remote {('rigid' if c.coupling == 'rigid' else 'distributing')} – {dofs_text(c.dofs)} at the point (" + "; ".join(fmt(float(x)) for x in c.point) + ")")
        if isinstance(c, ElasticSupport):
            from fea_core.loads_text import constraint_text
            return f"{c.selection}: {constraint_text(c, num=fmt)}"
        comps = ", ".join(f"u{a} = {fmt(v)}" for a, v in zip("xyz", (c.ux, c.uy, c.uz)) if v is not None)
        return f"{c.selection}: {comps} mm"

    @staticmethod
    def describe_load(l):
        from fea_core.loads_text import pressure_text, rotation_text, vec_text

        def vec(v):
            return vec_text(v, fmt)
        cs = f" [{l.csys}]" if getattr(l, "csys", None) else ""
        if isinstance(l, Force):
            return f"{l.selection}: F = {vec(l.vector)} N{cs}"
        if isinstance(l, Pressure) and l.variable:
            return f"{l.selection}: " + pressure_text(l, fmt).split(" (level")[0].split(" ·")[0] + \
                (f", level {vec(l.origin)}" if l.clip else "")
        if isinstance(l, Pressure):
            return f"{l.selection}: p = {fmt(l.value)} MPa"
        if isinstance(l, BearingLoad):
            return f"{l.selection}: bearing F = {vec(l.vector)} N{cs}"
        if isinstance(l, Gravity):
            return f"Gravity: g = {vec(l.vector)} mm/s²{cs}"
        if isinstance(l, Rotation):
            return "Rotation: " + rotation_text(l, fmt)
        if isinstance(l, ThermalLoad):
            if l.case:
                return f"Temperature from thermal state '{l.case}'" + (f" at time {fmt(l.time)} s" if l.time is not None else "")
            return f"Temperature {fmt(l.value)} °C – " + (", ".join(l.parts) if l.parts else "all parts")
        k = getattr(l, "kind", "")
        if k == "frame_node":
            return (f"Node {l.node}: F = {vec(l.force)} N" + (f", M = {vec(l.moment)} N·mm" if any(l.moment) else ""))
        if k == "frame_member":
            q = f"q = {vec(l.q)}" + (f" → {vec(l.q_end)}" if l.q_end is not None else "")
            return f"Member {l.member}: {q} N/mm" + (" (local axes)" if l.axes == "local" else "")
        if k == "frame_temperature":
            g = (f", z gradient {fmt(l.grad_z)} K" if l.grad_z else "") + (f", y gradient {fmt(l.grad_y)} K" if l.grad_y else "")
            return f"Member temperature ({(', '.join(l.members) if l.members else 'all members')}): ΔT = {fmt(l.dT)} K{g}"
        if isinstance(l, RemoteLoad):
            parts = []
            if any(l.force):
                parts.append(f"F = {vec(l.force)} N")
            if any(l.moment):
                parts.append(f"M = {vec(l.moment)} N·mm")
            return f"{l.selection}: {', '.join(parts)} at point {vec(l.point)}{cs}"
        return str(l)

    def refresh_model_info(self):
        if hasattr(self, "btn_edit_sketch"):
            self.btn_edit_sketch.setVisible(self.study is not None and self.study.sketch is not None)
        if self.study is None:
            if hasattr(self, "model_table"):
                self.model_table.setRowCount(0)
                self.model_table.setVisible(False)
                self.btn_toggle_ignore_part.setVisible(False)
                self.btn_toggle_hide_part.setVisible(False)
                self.btn_show_all_parts.setVisible(False)
            self.model_info.setVisible(True)
            self.model_info.setMaximumHeight(16777215)
            self.model_info.setHtml("Open a STEP model (.step, .stp) or a .fea project.")
            return
        if self.frame_only():
            if hasattr(self, "model_table"):
                self.model_table.setRowCount(0)
                self.model_table.setVisible(False)
                self.btn_toggle_ignore_part.setVisible(False)
                self.btn_toggle_hide_part.setVisible(False)
                self.btn_show_all_parts.setVisible(False)
            self.model_info.setVisible(True)
            self.model_info.setMaximumHeight(16777215)
            fr = self.study.frame
            self.model_info.setHtml(f"<b>Frame without CAD</b><br>Nodes: {len(fr.nodes)}, members: {len(fr.members)}, sections: {len(fr.sections)}<br><br>Edit the frame in the <b>Members</b> step, the member material in the Materials step (study default material).<br><br>Units: mm, N, MPa, t, s")
            return
        if hasattr(self, "model_table"):
            self.model_table.setVisible(True)
            self.btn_toggle_ignore_part.setVisible(True)
            self.btn_toggle_hide_part.setVisible(True)
            self.btn_show_all_parts.setVisible(True)
            self.model_info.setMaximumHeight(130)

            name = html.escape(self.step_path.name) if self.step_path else ""
            ign = set(self.study.ignored_parts)
            hid = set(self.study.hidden_parts)
            excl = self.study.excluded_parts()

            self.model_table.blockSignals(True)
            self.model_table.setRowCount(len(self.parts))
            for r, p in enumerate(self.parts):
                try:
                    m = self.study.material_of(p.name)
                except KeyError:
                    m = None
                is_ign = p.name in ign
                is_bolt = p.name in excl and not is_ign

                for c, on in ((0, p.name not in hid), (1, not is_ign)):
                    chk = QTableWidgetItem()
                    chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                    chk.setCheckState(Qt.Checked if on else Qt.Unchecked)
                    chk.setData(Qt.UserRole, p.name)
                    self.model_table.setItem(r, c, chk)

                it_name = QTableWidgetItem(p.name)
                if is_ign:
                    it_name.setForeground(QColor(140, 140, 140))
                    font = it_name.font()
                    font.setStrikeOut(True)
                    it_name.setFont(font)
                self.model_table.setItem(r, 2, it_name)

                if is_ign:
                    st_txt = "Excluded from analysis"
                    it_st = QTableWidgetItem(st_txt)
                    it_st.setForeground(QColor(ORANGE))
                elif is_bolt:
                    st_txt = "Bolt (beam)"
                    it_st = QTableWidgetItem(st_txt)
                    it_st.setForeground(QColor(GREEN))
                else:
                    st_txt = "Active"
                    it_st = QTableWidgetItem(st_txt)
                    it_st.setForeground(QColor(GREEN))
                if p.name in hid:
                    it_st.setText(st_txt + " (hidden)")
                self.model_table.setItem(r, 3, it_st)

                vol_cm3 = p.volume() / 1000.0
                it_vol = QTableWidgetItem(f"{fmt(vol_cm3, 4)}")
                it_vol.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if is_ign:
                    it_vol.setForeground(QColor(140, 140, 140))
                self.model_table.setItem(r, 4, it_vol)

                it_mat = QTableWidgetItem(m.name if m else "-")
                if is_ign:
                    it_mat.setForeground(QColor(140, 140, 140))
                self.model_table.setItem(r, 5, it_mat)

            self.model_table.blockSignals(False)
            self.model_table.resizeColumnsToContents()
            self.model_table.horizontalHeader().setStretchLastSection(True)

            active_cnt = sum(1 for p in self.parts if p.name not in excl)
            ign_cnt = len(ign)
            summary = [f"<b>{name}</b><br>Total bodies: {len(self.parts)} (active for analysis: {active_cnt}"]
            if ign_cnt:
                summary[0] += f", excluded: {ign_cnt}"
            if hid:
                summary[0] += f", hidden in 3D: {len(hid & {p.name for p in self.parts})}"
            summary[0] += ")<br>Units: mm, N, MPa, t, s"
            if getattr(self, "geometry_warnings", None):
                summary.append("<br><br><b>STEP geometry check warnings:</b><br>"
                               + "<br>".join(html.escape(w) for w in self.geometry_warnings[:10]))
            self.model_info.setHtml("".join(summary))

    def refresh_tree(self, check_result: tuple[list[str], list[str]] | None = None):
        self.tree.clear()
        if self.study is None:
            return
        st = self.study
        root = QTreeWidgetItem(self.tree, [st.name, ""])
        root.setExpanded(True)

        def node(parent, text, info="", color=None, data=None):
            it = QTreeWidgetItem(parent, [text, info])
            if color:
                it.setIcon(0, dot_icon(color))
            it.setData(0, Qt.UserRole, data)
            return it

        errors, warnings = check_result if check_result is not None else self.check()
        excl = st.excluded_parts()
        parts = node(root, "Parts", str(len(self.parts)), RED if self.material_errors(errors) else GREEN,
                     ("stage", STAGE["Materials"]))
        ignored = set(st.ignored_parts)
        for p in self.parts:
            try:
                mname = st.material_of(p.name).name
            except KeyError:
                mname = None
            info = ("excluded from analysis" if p.name in ignored else "bolt (beam)" if p.name in excl
                    else mname or "no material")
            node(parts, p.name, info, GREY if p.name in ignored else (RED if mname is None and p.name not in excl else None),
                 ("part", p.name))
        from fea_core.printing.region import print_parts
        printed = print_parts(st, [p.name for p in self.parts if p.name not in excl])
        if printed:
            pit = node(root, "3D printing", f"{len(printed)} parts", GREEN, ("stage", STAGE["3D printing"]))
            for name in printed:
                sp = st.printing.spec(name)
                node(pit, name, f"infill {fmt(100 * sp.infill, 3)} %" if sp.enabled else "solid", None,
                     ("stage", STAGE["3D printing"]))
        if st.bolts or st.welds:
            jits = node(root, "Joints", f"{len(st.bolts)} bolts, {len(st.welds)} welds", self.joints_color(),
                        ("stage", STAGE["Joints"]))
            chk = {c.name: c for c in self.joints_panel.assessment.all()} if self.joints_panel.assessment else {}
            for i, b in enumerate(st.bolts):
                c = chk.get(b.name)
                node(jits, f"{b.name}: {b.designation}", f"utilization {fmt(c.utilization, 3)}" if c else "",
                     self.util_color(c) if c else None, ("bolt", i))
            for i, w in enumerate(st.welds):
                c = chk.get(w.name)
                node(jits, f"{w.name}: {w.part}", f"utilization {fmt(c.utilization, 3)}" if c else "",
                     self.util_color(c) if c else None, ("weld", i))
        if st.contacts or st.analysis.nonlinear:      # contact pairs (F8)
            from fea_core.contacts.model import CONTACT_LABELS
            cit = node(root, "Contacts", f"{len(st.contacts)} pairs", self._panel_color("contact_panel", errors, warnings),
                       ("stage", STAGE["Contacts"]))
            for i, c in enumerate(st.contacts):
                node(cit, c.name, CONTACT_LABELS.get(c.kind, c.kind) + ("" if c.enabled else " (off)"), None,
                     ("contact", i))
        mech = st.mechanism
        if mech.bodies:                               # mechanismus (F8, krok Pohyb)
            mit = node(root, "Motion", f"{len(mech.bodies)} bodies, {len(mech.joints)} joints",
                       self._panel_color("motion_panel"), ("stage", STAGE["Motion"]))
            for j in mech.joints:
                from fea_core.motion.model import JOINT_LABELS
                node(mit, j.name, JOINT_LABELS.get(j.kind, j.kind), None, ("stage", STAGE["Motion"]))
        thermal_only = self.no_static()
        fr = st.frame
        if fr.nodes or fr.members:
            fit = node(root, "Members", f"{len(fr.members)} members", self._panel_color("frame_panel", errors), ("stage", STAGE["Members"]))
            for mname, m in list(fr.members.items())[:200]:
                node(fit, mname, f"{m.section}", None, ("frame_member", mname))
        cons = node(root, "Constraints", str(len(st.constraints)),
                    GREEN if st.constraints else (GREY if thermal_only or fr.supports else RED),
                    ("stage", STAGE["Constraints"]))
        for i, c in enumerate(st.constraints):
            node(cons, self.describe_constraint(c), "", C_HEX(C_CONSTRAINT), ("constraint", i))
        tit = node(root, "Heat", f"{len(st.thermal.cases)} st." if st.thermal.cases else "–",
                   self.thermal_color(errors, warnings), ("stage", STAGE["Heat"]))
        for ci, c in enumerate(st.thermal.cases):
            pre = f"Thermal case '{c.name}'"
            col = (RED if any(e.startswith(pre) for e in errors) else
                   ORANGE if any(w.startswith(pre) for w in warnings) else GREEN)
            cit = node(tit, c.name, describe_case(c), col, ("tcase", ci))
            for i, bc in enumerate(c.loads):
                node(cit, describe_bc(bc), "", None, ("tbc", (ci, i)))
        lcs = node(root, "Load cases", str(len(st.load_cases)), self.loads_color(errors),
                   ("stage", STAGE["Loads"]))
        for li, lc in enumerate(st.load_cases):
            bad = any(e.startswith(f"State '{lc.name}'") for e in errors)
            lit = node(lcs, lc.name, f"{len(lc.loads)}×", RED if bad else GREEN if lc.loads else
                       (GREY if st.thermal.cases or st.modal.enabled else ORANGE), ("lc", li))
            for i, l in enumerate(lc.loads):
                node(lit, self.describe_load(l), "", None, ("load", (li, i)))
        if st.dynamics.enabled or st.masses:          # dynamics and added mass (F8)
            ds = st.dynamics
            kinds = [t for on, t in ((ds.harmonic.enabled, "harmonic"), (ds.transient.enabled, "transient"),
                                     (ds.random.enabled, "random"), (ds.shock.enabled, "shock")) if on]
            dit = node(root, "Dynamics", (f"{len(kinds)} responses" if len(kinds) > 1 else kinds[0]) if ds.enabled and kinds
                       else ("modes" if ds.enabled else "–"), self._panel_color("dynamics_panel", errors, warnings),
                       ("stage", STAGE["Dynamics"]))
            dit.setToolTip(1, ", ".join(kinds))
            for m in st.masses:
                node(dit, f"Mass: {m.name or m.selection}", f"{fmt(1000 * m.mass)} kg" if m.kind == "point_mass"
                     else f"{fmt(m.mass_per_area)} kg/m²", None, ("stage", STAGE["Dynamics"]))
        mesh_txt = f"{fmt(len(self.mesh.tets))} elements" if self.mesh is not None else "not generated"
        node(root, "Mesh", mesh_txt, GREEN if self.mesh is not None else ORANGE, ("stage", STAGE["Mesh"]))
        res_txt = "computed" if self.results is not None else "not computed"
        rit = node(root, "Results", res_txt, GREEN if self.results is not None else GREY,
                   ("stage", STAGE["Results"]))
        if self.results is not None:
            for i, s in enumerate(self.results.steps):
                if len(s.S):
                    info = f"σmax {fmt(float(s.mises.max()))} MPa"
                else:
                    mm = (s.frame or {}).get("members") or {}
                    info = (f"σ_eq {fmt(max(float(np.max(d['sigma_eq'])) for d in mm.values()))} MPa" if mm else "")
                node(rit, s.name, info, None, ("result", f"step:{i}"))
            if self.results.twod is not None:
                for i, s2 in enumerate(self.results.twod.steps):
                    node(rit, f"2D: {s2['name']}", f"σmax {fmt(float(self.results.twod.mises(i).max()))} MPa", None,
                         ("result", f"twod:{i}"))
            if self.results.shell is not None:
                for i, s3 in enumerate(self.results.shell.summary()["steps"]):
                    node(rit, f"Shells: {s3['name']}", f"σmax {fmt(s3['mises_max'])} MPa", None,
                         ("result", f"shell:{i}"))
            for j, t in enumerate(self.results.thermal):
                node(rit, f"Thermal: {t.name}", f"Tmax {fmt(float(t.T.max()))} °C", None, ("result", f"thermal:{j}"))
            for j, ms in enumerate(self.results.modes):          # natural frequencies, buckling
                info = (f"f₁ {fmt(float(ms.values[0]))} Hz" if ms.kind == "modal" else
                        f"λ₁ {fmt(float(ms.values[0]))}") if len(ms.values) else "–"
                mit = node(rit, ms.name, info, None, ("result", f"mode:{j}:0"))
                for k in range(len(ms.values)):
                    node(mit, mode_label(ms, k), "", None, ("result", f"mode:{j}:{k}"))
            dyn_items = [x for x in dynr.items(self.results) if not x[1].startswith("dyn:mode")]
            if dyn_items:                                          # dynamika (F8)
                dres = self.results.dynamics
                dit = node(rit, "Dynamics", f"f₁ {fmt(float(dres.freq[0]))} Hz" if dres.K else "", None,
                           ("result", dyn_items[0][1]))
                for txt, data in dyn_items:
                    node(dit, txt.replace("Dynamics: ", ""), "", None, ("result", data))
        fr = self.fat_panel.result
        fat_txt = (f"D max {fmt(float(fr.total_damage.max()), 3)}" if fr is not None else
                   ("set" if st.fatigue and st.fatigue.history.channels else "–"))
        node(root, "Fatigue", fat_txt, self.fatigue_color(), ("stage", STAGE["Fatigue"]))
        self.tree.expandAll()

    @staticmethod
    def util_color(check):
        return util_color(check.utilization)

    @staticmethod
    def material_errors(errors) -> list[str]:
        """Readiness errors that are fixed in the Materials step (assignment, properties for heat)."""
        return [e for e in errors if "material" in e or e.startswith("Material")]

    def thermal_only(self) -> bool:
        """A study with heat conduction only (no non-empty mechanical state) – constraints and loads are not needed."""
        st = self.study
        return st is not None and bool(st.thermal.cases) and not any(lc.loads for lc in st.load_cases)

    def thermal_color(self, errors=None, warnings=None):
        """Heat step: grey = no thermal state, red = error, orange = warning, green = ready."""
        st = self.study
        if st is None or not st.thermal.cases:
            return GREY
        if errors is None:
            errors, warnings = self.check()
        if any(e.startswith("Thermal case") or "thermal conductivity" in e or "heat capacity" in e for e in errors):
            return RED
        used = {getattr(bc, "selection", None) for c in st.thermal.cases for bc in c.loads}
        if any(e.startswith("Selection '") and e.split("'")[1] in used for e in errors):
            return RED
        return ORANGE if any(w.startswith("Thermal case") for w in warnings or []) else GREEN

    def loads_color(self, errors):
        st = self.study
        if any(e.startswith("State '") for e in errors):
            return RED
        if self.no_static():
            return GREY
        if not any(lc.loads for lc in st.load_cases):
            return RED
        # empty state: orange (with thermal states it is skipped in the calculation)
        return GREEN if all(lc.loads for lc in st.load_cases) or st.thermal.cases or st.modal.enabled or \
            st.dynamics.enabled else ORANGE

    def joints_color(self):
        """Joints step: grey = none, green = entered / satisfied, orange = close, red = not satisfied."""
        st = self.study
        if st is None or not (st.bolts or st.welds):
            return GREY
        a = getattr(self, "joints_panel", None) and self.joints_panel.assessment
        if a:
            us = [c.utilization for c in a.all() if math.isfinite(c.utilization)]
            return util_color(max(us)) if us else ORANGE
        errors, _ = self.check()
        return RED if any(e.startswith(("Bolt", "Weld")) or "bolt" in e for e in errors) else GREEN

    def fatigue_color(self):
        st = self.study
        fr = getattr(self, "fat_panel", None) and self.fat_panel.result
        if fr is not None:
            return GREEN if fr.total_damage.max() < 1 else RED
        if st is not None and st.fatigue and st.fatigue.history.channels:
            return ORANGE
        return GREY

    def refresh_flow(self, check_result: tuple[list[str], list[str]] | None = None):
        st = self.study
        if st is None:
            cols = {name: GREY for name in STAGES}
            errors, warnings = [], []
        else:
            errors, warnings = check_result if check_result is not None else self.check()
            cols = {
                "Model": GREEN,
                "Materials": RED if self.material_errors(errors) else GREEN,
                "3D printing": RED if any(e.startswith("3D printing") for e in errors) else self._panel_color("print_panel"),
                "Joints": self.joints_color(),
                "Contacts": self._panel_color("contact_panel", errors, warnings),
                "Dynamics": self._panel_color("dynamics_panel", errors, warnings),
                "Motion": self._panel_color("motion_panel"),
                "Members": self._panel_color("frame_panel", errors),
                "Constraints": GREEN if st.constraints else (GREY if self.no_static() or st.frame.supports else RED),
                "Heat": self.thermal_color(errors, warnings),
                "Loads": self.loads_color(errors),
                "Mesh": GREEN if self.mesh is not None else ORANGE,
                "Solve": GREEN if self.results is not None else (ORANGE if not errors else RED),
                "Results": GREEN if self.results is not None else GREY,
                "Fatigue": self.fatigue_color(),
            }
        fo = self.frame_only()
        self._flow_cols = cols
        for name, a in zip(STAGES, self.stage_actions):
            a.setEnabled(not (fo and name in FRAME_ONLY_OFF))       # frame system without CAD: steps hidden and disabled
        self._apply_flow_visibility()
        self.flow_state = (cols, errors if st is not None else [], warnings if st is not None else [])
        self.flowChanged.emit()

    def on_tree_click(self, item, col):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, k = data
        if kind == "stage":
            self.set_stage(k)
        elif kind == "part":
            pi = next((i for i, p in enumerate(self.parts) if p.name == k), None)
            if pi is not None:
                self.view.set_selected_parts([pi])
                if STAGES[getattr(self, "stage", 0)] == "Model":
                    if hasattr(self, "model_table"):
                        self.model_table.clearSelection()
                        if pi < self.model_table.rowCount():
                            self.model_table.selectRow(pi)
                elif STAGES[getattr(self, "stage", 0)] == "Materials":
                    for r in range(self.mat_parts.count()):
                        it = self.mat_parts.item(r)
                        if it.text() == k:
                            self.mat_parts.setCurrentItem(it)
                            break
        elif kind in ("bolt", "weld"):
            self.joints_panel.select_joint(kind, k)
        elif kind == "constraint":
            self.set_stage(STAGE["Constraints"])
            self.con_list.setCurrentRow(k)
        elif kind == "contact":
            self.set_stage(STAGE["Contacts"])
            self.contact_panel.table.selectRow(k)
        elif kind == "frame_member":
            self.set_stage(STAGE["Members"])
            self.frame_panel.select("member", k)
        elif kind == "lc":
            self.set_stage(STAGE["Loads"])
            self.lc_combo.setCurrentIndex(k)
        elif kind == "load":
            self.set_stage(STAGE["Loads"])
            self.lc_combo.setCurrentIndex(k[0])
            self.load_list.setCurrentRow(k[1])
        elif kind == "tcase":
            self.set_stage(STAGE["Heat"])
            self.thermal_panel.case_combo.setCurrentIndex(k)
        elif kind == "tbc":
            self.set_stage(STAGE["Heat"])
            self.thermal_panel.case_combo.setCurrentIndex(k[0])
            self.thermal_panel.bc_list.setCurrentRow(k[1])
        elif kind == "result":
            self.set_stage(STAGE["Results"])
            self.r_step.setCurrentIndex(max(self.r_step.findData(k), 0))

    # ================================================================== part management and exclusion from the calculation
    def apply_ghost(self):
        """Excluded parts translucent in 3D (in all steps with a model)."""
        st = self.study
        v = self.view
        if not hasattr(v, "set_ghost"):
            return
        ign = set(st.ignored_parts) if st else set()
        v.set_ghost([i for i, p in enumerate(self.parts or []) if p.name in ign])
        if hasattr(v, "set_hidden"):
            hid = set(st.hidden_parts) if st else set()
            v.set_hidden([i for i, p in enumerate(self.parts or []) if p.name in hid])

    def set_ignored(self, names: list[str], on: bool):
        """Excludes or includes parts in the calculation (ignored_parts)."""
        st = self.study
        if st is None:
            return
        cur = list(st.ignored_parts)
        new = (cur + [n for n in names if n not in cur]) if on else [n for n in cur if n not in names]
        if new == cur:
            return
        st.ignored_parts = new
        self.apply_ghost()
        self.log_msg(("Excluded from analysis: " if on else "Restored to analysis: ") + ", ".join(names))
        self.changed(invalidate_mesh=True)
        if hasattr(self, "joints_panel"):
            self.joints_panel.refresh_ignored()

    def set_hidden(self, names: list[str], on: bool):
        """Hides or shows parts in the 3D view (Study.hidden_parts) – results and mesh are not discarded."""
        st = self.study
        if st is None:
            return
        cur = list(st.hidden_parts)
        new = (cur + [n for n in names if n not in cur]) if on else [n for n in cur if n not in names]
        if new == cur:
            return
        st.hidden_parts = new
        self.apply_ghost()
        self.mark_changed("none")                 # display only – mesh and results stay
        self.refresh_model_info()
        self.refresh_tree()

    def isolate_selected(self):
        """Isolate command: only the parts selected in 3D (or in the parts table)."""
        names = [self.parts[i].name for i in self.view.selected_parts() if 0 <= i < len(self.parts)]
        if not names and hasattr(self, "model_table"):
            rows = sorted({ix.row() for ix in self.model_table.selectedIndexes()})
            names = [self.parts[r].name for r in rows if r < len(self.parts)]
        if names:
            self.isolate_parts(names)

    def show_geometry_check(self):
        """Check command: Model step and the Geometry check tab of the bottom dock."""
        self.set_stage(STAGE["Model"])
        if hasattr(self, "table_dock"):
            self.table_dock.show_tab("geometry")
            self.dock_tables.raise_()

    def isolate_parts(self, names: list[str]):
        """Shows only the given parts (hides the others)."""
        if self.study is None or not names:
            return
        keep = set(names)
        self.study.hidden_parts = []
        self.set_hidden([p.name for p in self.parts if p.name not in keep], True)
        self.apply_ghost()
        self.refresh_model_info()

    def parts_by_pattern(self, pattern: str) -> list[str]:
        """Part names by pattern (* and ?, case-insensitive; without wildcards = contains the text)."""
        import fnmatch
        pat = pattern.strip().lower()
        if not pat:
            return []
        if not any(ch in pat for ch in "*?["):
            pat = f"*{pat}*"
        return [p.name for p in self.parts if fnmatch.fnmatchcase(p.name.lower(), pat)]

    def _by_pattern_dialog(self, what: str):
        """In bulk by name: what = ignore | include | hide | show."""
        titles = {"ignore": "Exclude from the analysis the bodies whose name contains / matches the pattern:",
                  "include": "Include in the analysis the bodies whose name contains / matches the pattern:",
                  "hide": "Hide in 3D the bodies whose name contains / matches the pattern:",
                  "show": "Show in 3D the bodies whose name contains / matches the pattern:"}
        text, ok = QInputDialog.getText(self, APP_NAME, titles[what] + "\n(e.g. bolt, M6*, *magnet*)")
        if not ok:
            return
        names = self.parts_by_pattern(text)
        if not names:
            msg(self, APP_NAME, f"No body matches the pattern “{text}”.")
            return
        if what in ("ignore", "include"):
            self.set_ignored(names, what == "ignore")
        else:
            self.set_hidden(names, what == "hide")

    def _on_model_table_check(self, item):
        """A tick in the parts table: column 0 = show, 1 = count (applies to all selected rows)."""
        if getattr(self, "_filling", False) or item.column() not in (0, 1):
            return
        name = item.data(Qt.UserRole)
        if not name:
            return
        on = item.checkState() == Qt.Checked
        rows = {i.row() for i in self.model_table.selectionModel().selectedRows()}
        names = [name]
        if item.row() in rows and len(rows) > 1:          # a tick in the selection = the whole selection
            names = [self.parts[r].name for r in sorted(rows) if r < len(self.parts)]
        if item.column() == 0:
            self.set_hidden(names, not on)
        else:
            self.set_ignored(names, not on)

    def _toggle_selected_model_parts_hidden(self):
        sel_rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        names = [self.parts[r].name for r in sel_rows if r < len(self.parts)]
        if not names:
            msg(self, APP_NAME, "Select a body in the table or in the 3D window.")
            return
        hid = set(self.study.hidden_parts) if self.study else set()
        self.set_hidden(names, not all(n in hid for n in names))

    def _on_tree_menu(self, pos):
        """Context menu of the tree (right click): exclude / include the body from the calculation."""
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, k = data
        if kind == "part":
            pname = k
            pi = next((i for i, p in enumerate(self.parts) if p.name == pname), None)
            ign = set(self.study.ignored_parts) if self.study else set()
            menu = QMenu(self)
            if pname in ign:
                act = menu.addAction(f"Include part '{pname}' in the analysis")
                act.triggered.connect(lambda: self.set_ignored([pname], False))
            else:
                act = menu.addAction(f"Exclude part '{pname}' from the analysis")
                act.triggered.connect(lambda: self.set_ignored([pname], True))
            hid = set(self.study.hidden_parts) if self.study else set()
            if pname in hid:
                act = menu.addAction(f"Show part '{pname}' in 3D")
                act.triggered.connect(lambda: self.set_hidden([pname], False))
            else:
                act = menu.addAction(f"Hide part '{pname}' in 3D")
                act.triggered.connect(lambda: self.set_hidden([pname], True))
            act = menu.addAction("Show only this part")
            act.triggered.connect(lambda: self.isolate_parts([pname]))
            if pi is not None:
                menu.addSeparator()
                zoom_act = menu.addAction("Zoom in 3D view\tF")
                zoom_act.triggered.connect(lambda: (self.view.set_selected_parts([pi]), self.view.zoom_selection()))
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_model_table_selection(self):
        rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        if STAGES[getattr(self, "stage", 0)] == "Model":
            self.view.set_selected_parts(rows)
            names = [self.parts[r].name for r in rows if r < len(self.parts)]
            self.st_info.setText(("Selected body: " + ", ".join(names)) if names else "")

    def _on_model_table_menu(self, pos):
        sel_rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        item = self.model_table.itemAt(pos)
        if item is not None and item.row() not in sel_rows:
            self.model_table.selectRow(item.row())
            sel_rows = [item.row()]
        if not sel_rows:
            return
        names = [self.parts[r].name for r in sel_rows if r < len(self.parts)]
        if not names:
            return

        ign = set(self.study.ignored_parts) if self.study else set()
        all_ignored = all(n in ign for n in names)

        menu = QMenu(self)
        if len(names) == 1:
            name = names[0]
            if all_ignored:
                act = menu.addAction(f"Include body '{name}' in the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, False))
            else:
                act = menu.addAction(f"Exclude body '{name}' from the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, True))
        else:
            if all_ignored:
                act = menu.addAction(f"Include selected bodies ({len(names)}) in the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, False))
            else:
                act = menu.addAction(f"Exclude selected bodies ({len(names)}) from the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, True))

        hid = set(self.study.hidden_parts) if self.study else set()
        if all(n in hid for n in names):
            act = menu.addAction("Zobrazit ve 3D" + (f" ({len(names)})" if len(names) > 1 else ""))
            act.triggered.connect(lambda: self.set_hidden(names, False))
        else:
            act = menu.addAction("Hide in 3D" + (f" ({len(names)})" if len(names) > 1 else ""))
            act.triggered.connect(lambda: self.set_hidden(names, True))
        act = menu.addAction("Show only the selected bodies")
        act.triggered.connect(lambda: self.isolate_parts(names))
        menu.addSeparator()
        act = menu.addAction("Show all")
        act.triggered.connect(lambda: self.set_hidden([p.name for p in self.parts], False))
        act = menu.addAction("Include all in the analysis")
        act.triggered.connect(lambda: self.set_ignored([p.name for p in self.parts], False))
        sub = menu.addMenu("By name")
        for what, label in (("ignore", "Exclude from analysis…"), ("include", "Include in analysis…"),
                            ("hide", "Hide in 3D…"), ("show", "Show in 3D…")):
            a = sub.addAction(label)
            a.triggered.connect(lambda _=False, w=what: self._by_pattern_dialog(w))
        menu.addSeparator()
        zoom_act = menu.addAction("Zoom in 3D view\tF")
        zoom_act.triggered.connect(self.view.zoom_selection)
        menu.exec(self.model_table.viewport().mapToGlobal(pos))

    def _toggle_selected_model_parts_ignore(self):
        sel_rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        names = [self.parts[r].name for r in sel_rows if r < len(self.parts)]
        if not names:
            msg(self, APP_NAME, "Select a body in the table or in the 3D window.")
            return
        ign = set(self.study.ignored_parts) if self.study else set()
        all_ignored = all(n in ign for n in names)
        self.set_ignored(names, not all_ignored)

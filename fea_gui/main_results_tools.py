"""Result tools of the main window: launchers of the post-processing dialogs, surface parts, the result summary, the probe, the
hotspot table and the deformation animation.
Mixed into `main.MainWindow` (attribute and method names unchanged: the panels, the MCP bridge and the GUI tests use them as before).
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidgetItem

from fea_core.i18n import keep_names, tr

from . import submodel_dialog as smd
from .thermal_panel import KIND_SHORT
from .widgets import report_soft

from .main_defs import (APP_NAME, STAGES, STAGE, GREEN, ORANGE, GREY, TOPO_SHAPE_Q, HS_STRESS, HS_THERMAL, HS_TOP, fmt, _batched,
                        msg_via_main as msg)


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class ResultsToolsMixin(_Base):
    """Result tools of the main window: launchers of the post-processing dialogs, surface parts, the result summary, the probe,
    the hotspot table and the deformation animation."""

    # ------------------------------------------------------------------ result dialogs (F7)
    def open_path_dialog(self):
        if self.results is None:
            msg(self, APP_NAME, "Run the analysis first (Solve step).")
            return
        if not self.results.steps and not self.results.thermal and self.results.twod is None:
            msg(self, APP_NAME, "The results contain only mode shapes – a line plot needs a load case or thermal case.")
            return
        from .path_dialog import PathDialog
        if len(self.results.mesh.nodes) == 0 and self.results.twod is None:
            msg(self, APP_NAME, "Results along a line are for body and 2D section results – for members, internal force diagrams are shown directly in 3D (quantities Members – N, V, M) and in the Members tab.")
            return
        if self._path_dlg is None or self._path_dlg.win is not self:
            self._path_dlg = PathDialog(self)
        else:
            self._path_dlg.refresh()
        self._path_dlg.show()
        self._path_dlg.raise_()
        if STAGES[self.stage] != "Results":
            self.set_stage(STAGE["Results"])
        self._path_dlg.draw_line()

    def open_section_force_dialog(self):
        if not self.results or not self.results.steps:
            msg(self, APP_NAME, "First solve the study.")
            return
        from .section_dialog import SectionForceDialog
        dlg = SectionForceDialog(self)
        dlg.exec()

    def open_plasticity_dialog(self):
        if not self.results or not self.results.steps:
            msg(self, APP_NAME, "First solve the study.")
            return
        from .plasticity_dialog import PlasticityDialog
        PlasticityDialog(self).exec()

    def open_criteria_selection_dialog(self):
        if not getattr(self, "parts", None):
            msg(self, APP_NAME, "First load the STEP geometry.")
            return
        from .selection_dialog import WorksheetDialog
        dlg = WorksheetDialog(self)
        dlg.exec()

    def refresh_r_csys(self):
        if not hasattr(self, "r_csys"):
            return
        cur = self.r_csys.currentData()
        self.r_csys.blockSignals(True)
        self.r_csys.clear()
        self.r_csys.addItem("Global Cartesian", None)
        if self.study and self.study.coordinate_systems:
            for k, cs in self.study.coordinate_systems.items():
                self.r_csys.addItem(f"{k} ({cs.kind})", k)
        k = self.r_csys.findData(cur)
        if k >= 0:
            self.r_csys.setCurrentIndex(k)
        self.r_csys.blockSignals(False)

    def open_convergence_dialog(self):
        if self.study is None:
            return
        if self.frame_only() or self.study.twod.enabled or self.study.shell.enabled:
            msg(self, APP_NAME, "Mesh convergence is for 3D bodies – for a frame, increase the number of elements per member (step Members → Assessment); for a 2D section and shells, reduce the element size and compare the results.")
            return
        from .convergence_dialog import ConvergenceDialog
        if self._conv_dlg is None:
            self._conv_dlg = ConvergenceDialog(self)
        else:
            self._conv_dlg.refresh()
        self._conv_dlg.show()
        self._conv_dlg.raise_()

    def open_voxel_dialog(self):
        """Voxel print detail (fea_core.printing.voxel) – only results with print safety."""
        res = self.results
        if res is None or not any("print_safety" in s.fields for s in res.steps):
            msg(self, APP_NAME, "The voxel detail needs a 3D printing result (a printed part with a print safety factor) – set up printing in the 3D printing step and run the analysis.")
            return
        from .voxel_dialog import VoxelDialog
        if self._voxel_dlg is None or self._voxel_dlg.win is not self:
            self._voxel_dlg = VoxelDialog(self)
        else:
            self._voxel_dlg.refresh()
        self._voxel_dlg.show()
        self._voxel_dlg.raise_()

    def surface_parts_from_step(self) -> list:
        """Plate parts (faces without volume) from the STEP model – fea_core.shell.geometry.read_step_surfaces."""
        if self.step_path is None or not Path(self.step_path).exists():
            return []
        from fea_core.shell.geometry import read_step_surfaces
        try:
            return read_step_surfaces(self.step_path)
        except Exception as e:  # noqa: BLE001
            self.warn("geometry check", f"Cannot load surface parts from STEP: {e}", "Model")
            return []

    @_batched
    def load_surface_parts(self) -> int:
        """Adds plate parts from the STEP among the parts of the model (display, edge / face selection for shells). Returns the count."""
        names = {p.name for p in self.parts}
        new = [p for p in self.surface_parts_from_step() if p.name not in names]
        if not new:
            return 0
        self.parts = list(self.parts) + new
        keep_names([p.name for p in self.parts] + ([self.step_path.stem] if self.step_path else []))
        self.surface_names |= {p.name for p in new}
        self.face_cache, self.edge_cache = {}, {}
        self.view.set_parts(self.parts)
        self.resolve_all()
        self.log_msg(f"Surface parts from STEP: {', '.join((p.name for p in new))} (shell model only)")
        self.refresh_all()
        return len(new)

    def solid_parts(self) -> list:
        """Parts for the 3D mesh (without plate parts from the STEP)."""
        return [p for p in self.parts if p.name not in self.surface_names]

    def register_aux(self, t):
        """Dialog thread (convergence, print refinement) – busy() and closing the window wait for it."""
        self._aux_threads.append(t)

    def unregister_aux(self, t):
        if t in self._aux_threads:
            self._aux_threads.remove(t)

    def interpolate(self, vals, p) -> float:
        """Nodal field at point p (undeformed mesh) – shape functions of the element that contains the point."""
        from fea_core.results import tet10_shape
        res = self.results
        e, xi = res.locate(p)
        conn = res.mesh.tets[e]
        N = tet10_shape(xi) if len(conn) == 10 else np.array([1 - xi.sum(), *xi])
        return float(N @ np.asarray(vals)[conn])

    def fill_thermal_info(self, th, frame):
        res = self.results
        m = res.mesh
        T = th.T if frame is None else th.frames[frame]
        i_max, i_min = int(np.argmax(T)), int(np.argmin(T))

        def at(i):
            return ", ".join(fmt(float(c), 5) for c in m.nodes[i])
        rows = [f"<b>{escape(str(th.name))}</b> – " + (f"transient case, time {fmt(float(th.times[frame if frame is not None else -1]))} s"
                                           if th.transient and th.times else "steady state"),
                f"Max. temperature: <b>{fmt(float(T[i_max]))} °C</b> at point ({at(i_max)}) mm",
                f"Min. temperature: <b>{fmt(float(T[i_min]))} °C</b> at point ({at(i_min)}) mm"]
        if frame is None:
            q = th.flux_mag
            rows.append(f"Max. heat flux |q|: <b>{fmt(float(q.max()))} W/m²</b>")
        if self.study is not None:
            rows.append(f"Reference temperature: {fmt(self.study.thermal.reference_temperature)} °C")
        if th.heat:
            rows.append("<br><b>Heat of boundary conditions</b> [W] (+ = into the body"
                        + (", end of the state" if th.transient else "") + "):")
            rows.append("<table cellspacing=3>")
            for h in th.heat:
                rows.append(f"<tr><td>{escape(str(h['name']))}</td><td>{escape(str(KIND_SHORT.get(h['kind'], h['kind'])))}</td>"
                            f"<td align=right><b>{fmt(float(h['Q_W']))}</b></td></tr>")
            rows.append("</table>")
        bal = th.balance or {}
        if "imbalance" in bal:
            rows.append(f"Heat balance: input {fmt(bal['in_W'])} W, output {fmt(bal['out_W'])} W, residual {fmt(bal['residual_W'], 3)} W ({fmt(100 * bal['imbalance'], 2)} %)")
        elif bal.get("note"):
            rows.append(f"Heat balance: {escape(str(bal['note']))}")
        if th.transient and th.times:
            rows.append(f"Snapshots: {len(th.times)} ({fmt(float(th.times[0]))} … {fmt(float(th.times[-1]))} s); flux only at the end of the case.")
        warn = [escape(str(w)) for w in res.warnings]
        if warn:
            rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(warn) + "</span>")
        self.res_info.setHtml(self._join_rows(rows))

    @staticmethod
    def _join_rows(rows):
        """HTML lines separated by <br>; a table (<table>…</table>) without a break between rows."""
        out, in_tab = [], False
        for r in rows:
            if r.startswith("<table"):
                in_tab = True
                out.append(r)
            elif r == "</table>":
                in_tab = False
                out.append(r)
            elif in_tab:
                out.append(r)
            else:
                out.append(r + "<br>")
        return "".join(out)

    def fill_result_info(self, st, scale, imax, vals):
        res = self.results
        m = res.mesh
        vm = st.mises
        i_vm = int(np.argmax(vm))
        rows = [f"<b>{escape(str(st.name))}</b>"]
        if st.partial:                              # non-convergence – last converted increment (F8, 4.9)
            rows.append("<span style='color:#d23b3b'><b>PARTIAL RESULT</b> – the calculation did not converge, "
                        "the last converted increment is shown (advice in the Messages dock)</span>")
        if st.contacts:
            rows.append("Contacts: " + "; ".join(f"{escape(str(n))} Fn {fmt(c.get('Fn_N'))} N, p_max {fmt(c.get('p_max_MPa'))} MPa"
                                                 for n, c in list(st.contacts.items())[:4])
                        + " (Contacts tab)")
        rows += [f"Max. displacement: <b>{fmt(float(st.umag.max()))} mm</b>",
                f"Max. von Mises: <b>{fmt(float(vm.max()))} MPa</b> at point ({', '.join((fmt(float(c), 5) for c in m.nodes[i_vm]))}) mm"]
        # classification of the maximum (fea_core.mcp.diagnostics – same as MCP and the report)
        cur = self.current_result()
        hs = self.result_hotspots("step", cur[1], "mises") if cur and cur[0] == "step" else None
        top = hs["hotspots"] if hs else []
        sing = bool(top) and top[0].get("classification") == "singularity"
        if top and top[0].get("classification") != "probably real concentration":
            rows[-1] += f" – <span style='color:#b36b00'>{escape(self.hotspot_label(top[0]))}</span>"
        if sing:
            ns = next((h for h in top if h.get("classification") != "singularity"), None)
            rows.append(f"Max. von Mises excluding singularities: <b>{fmt(ns['mises_MPa'])} MPa</b> at point {self._pt(ns['xyz_mm'], 5)} mm ({escape(self.hotspot_label(ns))})" if ns else
                        f"Max. von Mises excluding singularities: all {len(top)} highest locations are singularities – evaluate away from them (≥ 2–3 elements), in a section, or by nominal stress.")
        re = self.node_re()
        if np.isfinite(re).any():
            sf = float(np.min(re / np.maximum(vm, 1e-9)))
            rows.append(f"Min. safety factor to Re: <b>{fmt(sf, 3)}</b>")
        topo = (res.info or {}).get("topology")
        if topo:
            c0, c1 = topo.get("initial_compliance_Nmm"), topo.get("final_compliance_Nmm")
            comp = f", compliance full design → result: {fmt(c0, 4)} → {fmt(c1, 4)} N·mm (×{fmt(c1 / c0, 3)})" \
                if c0 and c1 else ""
            rows.append(f"Topology optimization (BESO): {topo.get('iterations', 0)} iterations, volume {fmt(100 * float(topo.get('final_volume_fraction', 0)), 3)} % of the original{comp} (quantities Topology – density / final shape)")
        T = st.fields.get("T")
        if T is not None:
            rows.append(f"Thermal load: T {fmt(float(np.min(T)))} … {fmt(float(np.max(T)))} °C (stress-free at {fmt(self.study.thermal.reference_temperature)} °C)")
        if st.printing:
            rows.append(f"<br><b>3D printing</b> – criterion {escape(str(st.printing.get('criterion', '')))} (print axes):")
            for pname, d in st.printing.get("parts", {}).items():
                sl = ", ".join(fmt(float(v), 3) for v in d["stress_local"])
                rows.append(f"{escape(str(pname))}: min. safety factor <b>{fmt(d['min_safety'], 3)}</b> – {escape(str(d['region']))} at point ({', '.join((fmt(float(c), 4) for c in d['at']))}) mm; σ11 σ22 σ33 τ12 τ13 τ23 = ({sl}) MPa")
            rows.append("")
        rows.append(f"Deformation scale: {fmt(scale, 3)}×")
        rows.append("<br><b>Reactions</b> [N]:")
        for k, r in st.reactions.items():
            rows.append(f"{escape(str(k))}: Fx {fmt(float(r[0]))}, Fy {fmt(float(r[1]))}, Fz {fmt(float(r[2]))}")
            rem = st.remote.get(k)
            if rem:                                   # remote constraint: moment about the constraint point
                mm = rem["moment"]
                rows.append(f"&nbsp;&nbsp;moment about point ({'; '.join((fmt(float(x)) for x in rem['point']))}) mm [N·mm]: Mx {fmt(float(mm[0]))}, My {fmt(float(mm[1]))}, Mz {fmt(float(mm[2]))}")
        eq = st.equilibrium
        if eq and "imbalance" in eq:              # a combination carries only the total reaction (no equilibrium check)
            rows.append(f"Equilibrium check: imbalance {fmt(100 * eq['imbalance'], 2)} %")
        # warning for singularities
        warn = ["Stress maxima at sharp internal corners, point loads and on support edges are singularities – do not evaluate by the nodal maximum; refine the mesh and monitor convergence."]
        if sing:
            warn.insert(0, f"<b>The maximum von Mises stress is a singularity ({escape(', '.join(top[0].get('tags') or []) or 'see table')}) – see the “Highest stresses” tab.</b>")
        elif hs is None:                          # classification failed → at least the test of the constraint face
            cons_nodes = set()
            for c in self.study.constraints:
                if c.selection in m.surfaces:
                    cons_nodes.update(np.unique(m.surfaces[c.selection]).tolist())
            if i_vm in cons_nodes:
                warn.insert(0, "<b>The maximum von Mises stress lies on a support face – probably a support singularity.</b>")
        for w in res.warnings:
            warn.append(escape(str(w)))
        rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(warn) + "</span>")
        self.res_info.setHtml("<br>".join(rows))

    def on_probe(self, xyz):
        hook = self._pick_hook                     # point picked by the dialog (progress along a line, convergence)
        if hook is not None and self.results is not None:
            p = self.undeformed(xyz)
            self.view.mark(xyz, "(" + "; ".join(fmt(float(c), 5) for c in p) + ")", (20, 120, 220), key="pick")
            hook(p)
            return
        if getattr(self, "_probe", None) is None or self.results is None:
            return
        tree, ids, vals, unit, name = self._probe
        exact = getattr(self, "_probe_exact", None)
        if exact is not None:                  # thermal state: interpolation in the element (Results.probe)
            try:
                v = exact(np.asarray(xyz, float))
            except Exception as e:  # noqa: BLE001 – outside the mesh → nearest node below
                report_soft("results", self, "Probe interpolation (nearest node used)", e)
                v = None
            if v is not None and np.isfinite(v):
                txt = f"{fmt(v)} {unit}"
                self.view.mark(xyz, txt, (20, 20, 20))
                self.st_info.setText(f"Probe: {name} = {txt} at point ({', '.join((fmt(float(c), 5) for c in xyz))}) mm")
                return
        d, j = tree.query(xyz)
        n = int(ids[j])
        txt = f"{fmt(float(vals[j]))} {unit}"
        self.view.mark(tree.data[j], txt, (20, 20, 20))
        labels = getattr(self, "_probe_labels", None)
        if labels is not None and j < len(labels) and labels[j][0]:
            self.st_info.setText(f"Probe: {name} = {txt} – member {labels[j][0]}, s = {fmt(float(labels[j][1]), 5)} mm")
            return
        if n >= len(self.results.mesh.nodes):
            n = -1
        if n >= 0:
            x = self.results.mesh.nodes[n]
            self.st_info.setText(f"Probe: {name} = {txt} at node {n + 1} "
                                 f"({', '.join(fmt(float(c), 5) for c in x)}) mm")
        else:
            self.st_info.setText(f"Probe: {name} = {txt} in section ({', '.join((fmt(float(c), 5) for c in tree.data[j]))}) mm")

    # ================================================================== maximum places (singularities)
    def diag_session(self):
        """MCP diagnostics session over the GUI state: maximum places are classified by the same functions as in MCP
        (fea_results_hotspots) and in the report – fea_core.mcp.diagnostics.hotspots / thermal_hotspots.
        """
        from fea_core.mcp.session import Session
        if self._diag_sess is None:
            self._diag_sess = Session(self._workdir / "diag")
        s = self._diag_sess
        s.study, s.results, s._parts = self.study, self.results, self.parts
        s.mesh = self.results.mesh if self.results is not None else None
        s.fatigue = self.fat_panel.result
        return s

    def result_hotspots(self, kind, i, key, time_s=None):
        """Maximum places (cache on the results object): kind "step" → hotspots with labels, "thermal" →
        thermal_hotspots. A heuristic error → None (written to the log), the results are shown anyway.
        """
        res = self.results
        if res is None:
            return None
        cache = getattr(res, "_gui_hotspots", None)
        if cache is None:
            cache = res._gui_hotspots = {}
        k = (kind, i, key, time_s)
        if k not in cache:
            from fea_core.mcp import diagnostics as dg
            try:
                sess = self.diag_session()
                cache[k] = (dg.thermal_hotspots(sess, i, HS_TOP, key, time_s) if kind == "thermal"
                            else dg.hotspots(sess, i, HS_TOP, key, tags=True))
            except Exception as e:  # noqa: BLE001 – the heuristic must not block the display of results
                self.warn("results", f"Could not evaluate the maximum locations: {e}", "Results")
                cache[k] = None
        return cache[k]

    @staticmethod
    def hotspot_label(h) -> str:
        """Short classification of a place: "singularity – clamped edge", "boundary condition – …", "real place";
        for temperature (no classification) a boundary condition at the place.
        """
        cls = h.get("classification")
        tags = h.get("tags") or []
        if cls == "singularity":
            return "singularity – " + ", ".join(tags) if tags else "singularity"
        if cls == "affected by boundary condition":
            return "boundary condition – " + ", ".join(tags) if tags else cls
        if cls:
            return "actual location" if cls.startswith("probably real") else cls
        return h["reasons"][0] if h.get("reasons") else "–"

    def clear_hotspots(self):
        self._hs_rows = []
        self.hs_table.blockSignals(True)
        self.hs_table.setRowCount(0)
        self.hs_table.blockSignals(False)
        self.hs_note.setText("")
        self.hs_title.setText("<b>Highest stresses</b>")
        self.res_tabs.setTabText(1, "Highest stresses")
        self.res_tabs.setTabToolTip(1, "")
        self.legend.set_note("")

    def fill_hotspots(self, kind, i, name, unit, vals, X, frame=None):
        """Table of maximum places of the displayed quantity (stress quantities; other mechanical → von Mises; heat →
        hottest places / heat flux). Returns (note for the legend, point of a singular maximum | None, cluster radius);
        the note and the point only if the table belongs to the displayed quantity.
        """
        self.clear_hotspots()
        if kind == "thermal":
            key = HS_THERMAL.get(name, "temperature")
            th = self.results.thermal[i]
            t = float(th.times[frame]) if frame is not None and key == "temperature" else None
            direct = True
            title = "Hottest locations" if key == "temperature" else f"Highest values – {name[:1].lower()}{name[1:]}"
        else:
            key = HS_STRESS.get(name, "mises")
            t = None
            direct = name in HS_STRESS
            title = "Highest stress – " + ("von Mises" if key == "mises" else name.replace("Stress ", "")
                                           + (" (absolute)" if key != "s1" else ""))
        self.hs_title.setText(f"<b>{title}</b>")
        tab_name = title.split(" – ")[0]
        self.res_tabs.setTabText(1, tab_name)
        d = self.result_hotspots(kind, i, key, t)
        if not d:
            return "", None, 0.0
        hs = d["hotspots"]
        u = unit if direct else d["unit"]
        tab = self.hs_table
        tab.blockSignals(True)
        tab.setRowCount(len(hs))
        tab.horizontalHeaderItem(1).setText(f"value [{u}]")
        for r, h in enumerate(hs):
            n = h["node"] - 1
            v = float(vals[n]) if direct else float(h["value"])
            cls = h.get("classification", "")
            tip = "\n".join([x for x in [cls, *h.get("reasons", []), h.get("recommendation", "")] if x])
            for c, txt in enumerate((str(h["rank"]), fmt(v), ", ".join(h["parts"]), self._pt(h["xyz_mm"], 4),
                                     self.hotspot_label(h))):
                it = QTableWidgetItem(txt)
                it.setToolTip(tip)
                if c == 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tab.setItem(r, c, it)
            col = {"singularity": ORANGE, "affected by boundary condition": GREY}.get(cls, GREEN if cls else None)
            if col:
                tab.item(r, 4).setForeground(QColor(col))
            self._hs_rows.append((h, v, u))
        tab.blockSignals(False)
        self._hs_X = X
        self._hs_name = name if direct else "von Mises"
        note, sing_xyz = "", None
        if hs and hs[0].get("classification") == "singularity":
            ns = next((k for k, row in enumerate(self._hs_rows) if row[0].get("classification") != "singularity"), None)
            # two sentences are translated separately (the catalogue has each as a separate template)
            note = tr(f"Max. {fmt(self._hs_rows[0][1])} {u} is a singularity "
                      f"({', '.join(hs[0].get('tags') or []) or 'see table'}) – do not judge by the node value.")
            note += tr(f" Max. outside singularities: {fmt(self._hs_rows[ns][1])} {u} (location #{hs[ns]['rank']})."
                       if ns is not None else f" All {len(hs)} highest locations are singularities.")
            sing_xyz = np.asarray(hs[0]["xyz_mm"], float)
        nsing = sum(1 for h in hs if h.get("classification") == "singularity")
        if nsing:
            self.res_tabs.setTabText(1, tr(tab_name) + f" ! {nsing}")   # translate the name separately
            self.res_tabs.setTabToolTip(1, f"Singularities among the {len(hs)} highest locations: {nsing}")
        info = []
        if note:
            info.append(f"<span style='color:#b36b00'>! {escape(note)}</span>")
        elif any(h.get("classification") for h in hs):
            info.append(f"Singularities among the {len(hs)} highest locations: {nsing}.")
        if d.get("coldest"):
            cd = d["coldest"]
            info.append(f"Coldest location: {fmt(cd['T_C'])} °C at point {self._pt(cd['xyz_mm'], 4)} mm ({escape(', '.join(cd['parts']))}).")
        info.append("<i>Clicking a row marks the location in 3D; classification reasons in the tooltip.</i>")
        self.hs_note.setText("<br>".join(info))
        if not direct:
            return "", None, 0.0
        return note, sing_xyz, float(d.get("cluster_radius_mm", 0.0))

    def on_hotspot_row(self, r):
        """Row of the maximum-places table: marker in 3D (deformed position), value as a probe, reasons of the classification."""
        import html
        if not (0 <= r < len(self._hs_rows)) or self.results is None:
            return
        h, v, u = self._hs_rows[r]
        n = h["node"] - 1
        X = getattr(self, "_hs_X", None)
        pos = X[n] if X is not None and n < len(X) else h["xyz_mm"]
        sing = h.get("classification") == "singularity"
        self.view.mark(pos, f"#{h['rank']}: {fmt(v)} {u}", (200, 100, 0) if sing else (20, 20, 20), key="hotspot",
                       below=True)
        label = self.hotspot_label(h)
        self.st_info.setText(f"Probe: {self._hs_name} = {fmt(v)} {u} at node {n + 1} {self._pt(h['xyz_mm'], 5)} mm – location #{h['rank']}: {label}")
        rows = [f"<b>#{h['rank']}: {html.escape(label)}</b>"] + [f"• {html.escape(x)}" for x in h.get("reasons", [])]
        if h.get("recommendation"):
            rows.append(f"<i>{html.escape(h['recommendation'])}</i>")
        self.hs_note.setText("<br>".join(rows))

    def toggle_animation(self):
        """Deformation animation: LEVELS deformation levels per cycle; the first visit of a level is a whole `show_result`,
        its result objects are then kept by the view (FeaView.anim_store) and later visits only swap the displayed objects
        (anim_show). Any other redraw of the result drops the cached frames (show_result → anim_reset).
        The timer runs frame by frame: the next is scheduled only after the previous one is drawn, with a period of at
        least 33 ms and an idle time of at least the frame time, so the main thread is not at 100 % for a large model.
        The phase advances by the real time (for slow frames the animation only gets coarser, it does not run slower).
        """
        import time
        if not hasattr(self, "_anim_timer"):
            from PySide6.QtCore import QTimer
            PERIOD = 1.6                 # s, duration of one cycle of the deflection enlargement
            MIN_MS = 33                  # shortest period of the frames (~30 fps)
            LEVELS = 40                  # deformation levels per cycle (21 distinct frames, each drawn once)
            self._anim_timer = QTimer(self)
            self._anim_timer.setSingleShot(True)
            self._anim_timer.setInterval(MIN_MS)
            self._anim_phase = 0.0
            self._anim_last = 0.0

            def stop(redraw=True):
                self._anim_running = False
                self._anim_timer.stop()
                self.view.anim_end()
                self._anim_state = {}
                self._anim_factor = 1.0
                self.btn_anim.setText("Animate...")
                if redraw:
                    self.show_result()

            def on_tick():
                if self.results is None or STAGES[self.stage] != "Results":
                    stop(redraw=False)           # nothing to animate outside results
                    return
                t0 = time.perf_counter()
                dt = min(t0 - self._anim_last, PERIOD / 8)       # a long delay (modal window) does not move the phase
                self._anim_last = t0
                step = 2 * math.pi * dt / PERIOD
                warm = len(self._anim_state) < LEVELS // 2 + 1
                if warm:                             # first cycle: the frames are built level by level (none skipped)
                    step = min(step, 2 * math.pi / LEVELS)
                self._anim_phase = (self._anim_phase + step) % (2 * math.pi)
                k = int(round(self._anim_phase / (2 * math.pi) * LEVELS)) % LEVELS
                level = min(k, LEVELS - k)                       # the factor is symmetric in the phase
                self._anim_factor = 0.5 * (1.0 - math.cos(2 * math.pi * level / LEVELS))
                self._anim_frame = True              # show_result keeps the summary, hotspot table and contact box
                try:
                    with self.view.batch():          # cached frame: only the displayed objects are swapped
                        if self.view.anim_show(level):
                            self._probe, self._hs_X, self._r_view, self._probe_exact = self._anim_state[level]
                        else:
                            self.show_result()
                            self.view.anim_store(level)
                            self._anim_state[level] = (self._probe, getattr(self, "_hs_X", None), self._r_view,
                                                       self._probe_exact)
                except Exception:
                    stop(redraw=False)           # otherwise the error would repeat 20× per second
                    raise
                finally:
                    self._anim_frame = False
                frame = time.perf_counter() - t0
                if self._anim_running:
                    # period ≥ MIN_MS; idle time ≥ the frame (cached frames) / ≥ half of it (first cycle)
                    self._anim_timer.start(max(MIN_MS - int(1000 * frame), int((500 if warm else 1000) * frame)))

            self._anim_timer.timeout.connect(on_tick)
            self._anim_stop = stop
        if getattr(self, "_anim_running", False):
            self._anim_stop()
        else:
            self._anim_running = True
            self._anim_phase = 0.0
            self._anim_last = time.perf_counter()
            self.view.anim_begin()
            self._anim_state = {}
            self.btn_anim.setText("Stop animation")
            self._anim_timer.start()

    def open_isosurface_dialog(self):
        if self.results is None:
            return
        from .isosurface_dialog import IsosurfaceDialog
        cur = self.current_result()
        si = cur[1] if cur and cur[0] == "step" else 0
        dlg = IsosurfaceDialog(self, self.results, si)
        dlg.exec()

    def open_vectors_dialog(self):
        if self.results is None:
            return
        from .vectors_dialog import VectorsDialog
        cur = self.current_result()
        si = cur[1] if cur and cur[0] == "step" else 0
        step = label = None
        if cur and cur[0] == "combination":         # displayed combination (not state 1); the envelope has no vectors
            from fea_core.results import StepResult
            comb = self.study.combinations[cur[1]]
            try:
                r = self._combination(self.results, cur[1])
            except (ValueError, KeyError) as e:
                msg(self, APP_NAME, f"The combination cannot be computed: {e}")
                return
            if isinstance(r, StepResult):
                step, label = r, f"K{cur[1] + 1:02d}: {comb.name}"
        dlg = VectorsDialog(self, self.results, si, step=step, label=label)
        dlg.exec()

    def open_submodel_dialog(self):
        if self.results is None or not self.results.steps or len(self.results.mesh.nodes) == 0:
            msg(self, APP_NAME, "The submodel needs a solved load case of a 3D model (Solve step).")
            return
        dlg = getattr(self, "_submodel_dlg", None)
        if dlg is None:
            dlg = self._submodel_dlg = smd.SubmodelDialog(self)
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.draw_region()

    def open_topo_dialog(self):
        if self.study is None:
            return
        mesh = self.results.mesh if self.results else getattr(self, "mesh", None)
        if mesh is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "First create the model mesh or run the analysis.")
            return
        from .topo_dialog import TopologyOptDialog
        dlg = TopologyOptDialog(self, self.study, mesh, self.results)
        dlg.exec()

    def topo_shape_tris(self, res):
        """Surface of the resulting shape of the topology optimisation (cache on the results object, valid for the same summary)."""
        from fea_core.topo.results import shape_triangles
        info = (res.info or {}).get("topology")
        cache = getattr(res, "_topo_tris", None)
        if cache is not None and cache[0] is info:
            return cache[1]
        tris = shape_triangles(res)
        res._topo_tris = (info, tris)
        return tris

    def topo_attached(self):
        """The optimisation was attached to the results: mark the project as changed WITHOUT discarding the results and show the shape."""
        self.dirty = True
        self.update_title()
        if self.results is None:
            return
        self.results._topo_tris = None
        self.select_stage_results()
        names = [n for n, _ in self.result_qlist()]
        if TOPO_SHAPE_Q in names:
            self.r_qty.blockSignals(True)
            self.r_qty.setCurrentIndex(names.index(TOPO_SHAPE_Q))
            self.r_qty.blockSignals(False)
        self.show_result()

    def select_stage_results(self):
        """Switches to the Results step if it is not active."""
        if STAGES[self.stage] != "Results":
            self.set_stage(STAGE["Results"])

    def open_export_dialog(self):
        if not self.results or not self.results.steps:
            msg(self, APP_NAME, "First solve the study.")
            return
        from .export_dialog import ExportResultsDialog
        dlg = ExportResultsDialog(self)
        dlg.exec()

    def open_imprint_dialog(self):
        if not self.study:
            return
        from .imprint_dialog import ImprintDialog
        dlg = ImprintDialog(self)
        dlg.exec()

    def open_combinations_dialog(self):
        if not self.study:
            return
        from .combinations_dialog import CombinationsDialog
        dlg = CombinationsDialog(self)
        dlg.exec()

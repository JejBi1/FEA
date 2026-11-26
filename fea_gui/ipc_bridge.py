"""Dynamic IPC bridge (LLM / MCP) of the main window: `execute_mcp_action` called by the IPC server in the GUI thread,
read-only / long-tool classification, state copy session ↔ window and the take-over of finished MCP jobs. Mixed into
`main.MainWindow`.

Imported by `fea_gui.main` for `MainWindow`; the constants below come from `fea_gui.main_defs`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from fea_core.study import Study

from .workers import kill_mesh_workers
from .main_defs import STAGE, STAGES


# the declarations of MainWindowHost (type checking only) must match what the mixins implement: a changed signature or
# return type of a member declared there is reported
# pyright: reportIncompatibleMethodOverride=error
if TYPE_CHECKING:
    from .main_host import MainWindowHost as _Base     # type checking only: what the window and the sibling mixins provide
else:
    _Base = object


class McpBridgeMixin(_Base):
    """MCP tools run in the window through the IPC server (part of MainWindow)."""

    def _mcp_job_active(self) -> bool:
        """An asynchronous MCP job (fea_job_start) of the window session is running – it works on the shared study."""
        sess = getattr(self, "_mcp_session", None)
        jobs = sess.__dict__.get("_fea_jobs") if sess is not None else None
        return bool(jobs) and any(getattr(j, "active", False) for j in jobs.values())

    def _get_mcp_tools(self):
        if self._mcp_tools_cache is None:
            from fea_core.rest.registry import collect_tools
            self._mcp_tools_cache = collect_tools(self._mcp_session)
        return self._mcp_tools_cache

    # tools allowed while the window is busy (they do not touch the study)
    MCP_WHILE_BUSY = frozenset({"fea_gui_status", "fea_gui_toast", "fea_job_status", "fea_job_cancel"})

    # long tools: run in a worker thread through run_busy (the GUI stays painted, Cancel cancels the call)
    MCP_LONG_TOOLS = frozenset({
        "fea_solve_run", "fea_mesh_generate", "fea_mesh_convergence", "fea_motion_run", "fea_fatigue_run",
        "fea_print_variants_run", "fea_print_sweep", "fea_print_rank_build_direction", "fea_print_refine",
        "fea_print_voxel_run", "fea_topo_opt_run", "fea_report_word", "fea_results_export",
        "fea_submodel_run", "fea_param_study_run", "fea_study_new", "fea_study_open", "fea_surrogate_fit",
        "fea_setup_review",           # picture (numba kernels may compile for a minute) on the shadow shapes; the dialog opens in the GUI thread
        "fea_space_model_checks"})    # seven checks with their own solver runs

    @classmethod
    def mcp_long(cls, tool_name: str, args: dict | None = None) -> bool:
        """Long tool: runs in a worker thread through run_busy. fea_sweep_run is long only when it waits for the whole sweep
        (background=False); background=True returns a job at once and the job thread works on a copy of the study."""
        return tool_name in cls.MCP_LONG_TOOLS or (tool_name == "fea_sweep_run" and not (args or {}).get("background", True))

    _MCP_READ_ONLY = None

    @classmethod
    def mcp_read_only(cls, tool_name: str) -> bool:
        """Tool that only reads the state (lists, status, results, renders, calculators, report / export files): the
        window state is not synchronised back from the session and the study is not marked as changed."""
        if cls._MCP_READ_ONLY is None:
            import re
            cls._MCP_READ_ONLY = re.compile(
                r"_(list|status|info|get|summary|hotspots|probe|history|line|linearize|modes|search|presets|check|"
                r"check_dof|plot|results|estimate|groups)$|_render(_\w+)?$|^fea_calc_|^fea_surrogate_|^fea_geometry_(find_\w+|"
                r"get_face_info|list_parts)$|^fea_(beam_analytic|curved_beam|ec3_member_check|check_readiness|"
                r"iterations|report_word|results_export|study_save|convergence_history|results_section_force|setup_review|"
                r"results_nlbuckling|thermal_suggest_conductance|tool_help|tools_groups|space_sar_report)$")
        return bool(cls._MCP_READ_ONLY.search(tool_name)) and tool_name != "fea_job_result"

    def _mcp_sync_surrogate(self, sess):
        """Parametric study and surrogate of the session → window (a fit / a sweep / opening a project through MCP); changed = the project is
        marked as changed."""
        with sess.use(wait=2.0):
            ps, sm = sess.param_study, sess.surrogate
        changed = False
        if ps is not getattr(self, "param_study", None):
            self.param_study = ps
        if sm is not getattr(self, "surrogate_model", None):
            self.surrogate_model = sm
            changed = sm is not None
        if changed:
            self.dirty = True
            self.update_title()

    def _mcp_sync_back(self, before: str | None):
        """State of the MCP session → window after a tool: a new model (other study / STEP) is loaded like a file;
        otherwise study / mesh / results are taken over; dirty + undo snapshot only when something changed."""
        sess = self._mcp_session
        with sess.use(wait=5.0):
            study, step, mesh, results = sess.study, sess.step_path, sess.mesh, sess.results
            parts, study_path = sess._parts, getattr(sess, "study_path", None)
            checks = getattr(sess, "model_checks", None)
        if study is None:
            return
        if checks is not None and checks is not getattr(self, "model_checks", None):
            self.model_checks = checks                 # the run of the model quality checks (fea_space_model_checks) for the report and the dialog
            self._notify_space_checks()
        if (self.study is None or (step != self.step_path and step is not None)
                or (step is None and self.step_path is not None and study is not self.study)):      # a new study without geometry (beams, an unbuilt sketch)
            self._load_model(step, study, results, parts=parts)
            sp = Path(study_path) if study_path else None
            self.project_path = sp if sp is not None and sp.suffix.lower() == ".fea" else None
            self.dirty = False
            self.reset_undo()
            self.update_title()
            return
        after = json.dumps(study.to_dict(), sort_keys=True, default=str)
        study_changed = study is not self.study or after != before
        mesh_changed, results_changed = mesh is not self.mesh, results is not self.results
        self.study = study
        if results is not None and results.mesh is not None:
            mesh = results.mesh
        self.mesh, self.results = mesh, results
        if mesh_changed and mesh is not None:
            self._mesh_sig = self.study_signature(self.MESH_KEYS)
        if results_changed:
            self.fat_panel.result = None
            self.joints_panel.assessment = None
            if results is not None:
                self.messages.set_source("results", [("warning", w) for w in results.warnings], unique=True)
        if study_changed:
            self.resolve_all()
            self.push_undo()
        if study_changed or mesh_changed or results_changed:
            self.dirty = True
            self.update_title()

    def _watch_mcp_job(self, before: str | None):
        """A started background job (fea_job_start, fea_sweep_run(background=True)): the job works on the shared session in its own thread; when it ends, its state is taken over into
        the window (also when the client never calls fea_job_result)."""
        t = getattr(self, "_mcp_job_timer", None)
        if t is None:
            t = self._mcp_job_timer = QTimer(self)
            t.setInterval(1000)

            def tick():
                if self._mcp_job_active():
                    return
                t.stop()
                if self._mcp_job_stale():
                    # the user changed calculation inputs while the job ran on its own copy: keep the window's study,
                    # drop the job's results (they belong to the old inputs)
                    self._mcp_warned_for = None
                    self.warn("LLM", "The LLM job finished, but the study was changed meanwhile – its result was not "
                                     "taken over (run the calculation again).")
                    return
                try:
                    self._mcp_sync_back(self._mcp_job_before)
                except Exception as e:  # noqa: BLE001 – a busy session: the client takes the result with fea_job_result
                    self.log_msg(f"LLM job finished – the result could not be taken over yet: {e}")
                    return
                self.refresh_all()
                if self.results is not None:
                    self.set_stage(STAGE["Results"])
                    self.show_result()
            t.timeout.connect(tick)
        self._mcp_job_before = before
        t.start()

    def execute_mcp_action(self, tool_name: str, args: dict):
        """Runs an MCP tool or an interactive GUI command for the IPC server (called in the GUI thread, returns the result).

        - while the window is busy (mesh, analysis, report, dialog threads, an MCP job) only status / job-control tools run;
        - long tools (MCP_LONG_TOOLS) run in a worker thread (run_busy: the window stays painted, Cancel = call cancel);
          tools that work on the shapes shown in 3D (MCP_GUI_THREAD_TOOLS) and short ones run in the GUI thread;
        - read-only tools (mcp_read_only) do not touch the window state; the others copy the state into the session
          (under its lock), and back afterwards: dirty + undo snapshot only when the study really changed.
        """
        if tool_name == "fea_gui_status":
            return {
                "gui_running": True,
                "busy": self.busy(),
                "study": self.study.name if self.study is not None else None,
                "stage": STAGES[self.stage] if hasattr(self, "stage") else None,
                "parts": len(self.parts) if self.parts else 0,
                "mesh_ready": self.mesh is not None,
                "results_ready": self.results is not None,
            }
        if self.busy() and tool_name not in self.MCP_WHILE_BUSY:
            raise RuntimeError("FEA Studio is busy (meshing, analysis, report or another long operation is running in the "
                               "window) – try again when it has finished (fea_gui_status shows 'busy').")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"LLM: {tool_name} ...", 0)
        try:
            # 1. Interactive GUI tools
            if tool_name == "fea_cad_link_info":       # CAD add-in: what is open and which tokens to translate
                return self.cad_link_info()
            if tool_name in ("fea_cad_update", "fea_cad_open"):    # CAD add-in: new geometry / new study
                p = Path(str(args.get("step_path", "")))
                from fea_core.pathcheck import is_unc
                if is_unc(str(p)) or p.suffix.lower() not in (".step", ".stp") or not p.is_file():
                    raise ValueError(f"Invalid STEP file: {p}")
                if tool_name == "fea_cad_open" or self.study is None or self.step_path is None:
                    QTimer.singleShot(0, lambda: self.open_file(str(p)))     # only after the add-in answers
                    return {"accepted": True, "mode": "new_study"}
                solve = bool(args.get("solve", False))
                if not self._same_cad_document(p):        # a different CAD document than the open project → new study
                    QTimer.singleShot(0, lambda: self.open_file(str(p)))
                    return {"accepted": True, "mode": "new_study", "reason": "different CAD document"}
                QTimer.singleShot(0, lambda: self.update_geometry(p, "cad", solve))
                return {"accepted": True, "mode": "update", "study": self.study.name}
            if tool_name == "fea_gui_toast":
                msg_txt = str(args.get("message", ""))
                timeout_ms = int(args.get("timeout_ms", 4000))
                self.statusBar().showMessage(msg_txt, timeout_ms)
                return {"gui_running": True, "toast": msg_txt}
            if tool_name == "fea_gui_stage_switch":
                stage = str(args.get("stage", ""))
                if stage in STAGE:
                    self.set_stage(STAGE[stage])
                    return {"gui_running": True, "stage": stage}
                raise ValueError(f"Unknown step '{stage}'. Allowed: {', '.join(STAGES)}")
            if tool_name == "fea_gui_camera_focus":
                sel = args.get("selection")
                part = args.get("part")
                if part and part != "*":
                    pi = next((i for i, p in enumerate(self.parts) if p.name == part), None)
                    if pi is not None:
                        from fea_core.geometry import bbox
                        bb = bbox(self.parts[pi].shape)
                        from OCP.Bnd import Bnd_Box  # pyright: ignore[reportAttributeAccessIssue]  # OCP stubs lack it, present at run time (view.py)
                        from OCP.gp import gp_Pnt  # pyright: ignore[reportAttributeAccessIssue]  # OCP stubs lack it, present at run time
                        b = Bnd_Box()
                        b.Add(gp_Pnt(float(bb[0]), float(bb[1]), float(bb[2])))
                        b.Add(gp_Pnt(float(bb[3]), float(bb[4]), float(bb[5])))
                        self.view.fit_box(b)
                    else:
                        self.view.fit_all()
                elif sel and self.study and sel in self.study.selections:
                    faces = self.sel_faces.get(sel, [])
                    if faces:
                        self.view.set_selected_faces(faces)
                        self.view.zoom_selection()
                    else:
                        self.view.fit_all()
                else:
                    self.view.fit_all()
                self.view.update()
                return {"gui_running": True, "focused": True, "target": sel or part or "all"}
            if tool_name == "fea_gui_highlight":
                entities = args.get("entities") or []
                faces_to_show = []
                rejected = []
                for ent in entities:
                    if self.study and ent in self.study.selections:
                        faces_to_show.extend(self.sel_faces.get(ent, []))
                    elif "#" in ent:
                        pname, sidx = ent.rsplit("#", 1)
                        try:
                            f_idx = int(sidx)
                            p_idx = next((i for i, p in enumerate(self.parts) if p.name == pname), None)
                            if p_idx is not None and p_idx < len(self.view.part_faces) and 0 <= f_idx < len(self.view.part_faces[p_idx]):
                                faces_to_show.append((p_idx, f_idx))
                            else:
                                rejected.append(ent)
                        except ValueError:
                            rejected.append(ent)
                    else:
                        rejected.append(ent)
                self.view.set_selected_faces(faces_to_show)
                self.view.update()
                return {"gui_running": True, "highlighted": [e for e in entities if e not in rejected], "rejected": rejected}
            if tool_name == "fea_gui_show_mode":
                mode_num = int(args.get("mode_number", 1))
                if self.results is not None and getattr(self.results, "modes", None):
                    idx = -1
                    target_suffix = f":{mode_num - 1}"
                    for i in range(self.r_step.count()):
                        data = str(self.r_step.itemData(i) or "")
                        if data.startswith("mode:") and data.endswith(target_suffix):
                            idx = i
                            break
                    if idx >= 0:
                        self.r_step.setCurrentIndex(idx)
                        self.set_stage(STAGE["Results"])
                        self.show_result()
                        return {"gui_running": True, "mode_shown": mode_num}
                return {"gui_running": True, "mode_shown": mode_num, "note": "Results with mode shapes are not available"}

            # 2. Common MCP tools (fea_core.mcp)
            sess = self._mcp_session
            tools = self._get_mcp_tools()
            if tool_name not in tools:
                raise ValueError(f"Unknown tool: '{tool_name}'")
            tool = tools[tool_name]
            v_args = tool.validate(args)
            read_only = self.mcp_read_only(tool_name)
            job_tool = tool_name in ("fea_job_status", "fea_job_cancel", "fea_job_result")
            if tool_name in ("fea_study_new", "fea_study_open") and not self.confirm_discard():
                raise RuntimeError("The user kept the current study (unsaved changes) – nothing was opened.")
            before = json.dumps(self.study.to_dict(), sort_keys=True, default=str) if self.study is not None else None
            if not job_tool:                       # the session of a running / finished job holds its own state
                with sess.use(wait=2.0):
                    # a background job gets its own copy: edits in the window must not change its inputs mid-run
                    starts_job = tool_name == "fea_job_start" or (tool_name == "fea_sweep_run" and v_args.get("background", True))
                    sess.study = (Study.from_dict(self.study.to_dict()) if starts_job and self.study is not None
                                  else self.study)
                    sess._parts = self.parts
                    sess.step_path = self.step_path
                    sess.mesh = self.mesh
                    # the mesh of the window is valid for the study of the window (the same check as the window's own
                    # calculation) → tools that reuse a valid mesh (solve, sweep) do not mesh again
                    sess.mesh_sig = sess.mesh_signature() if self.study is not None and self.valid_mesh() is not None else None
                    sess.results = self.results
                    sess.workdir = self._workdir
                    # the parametric study and its surrogate belong to the window (saved in the .fea, shown in the dialog): the session
                    # of the bridge carries them for the tool and the sync below takes them back (a new study / sweep clears the surrogate)
                    sess.param_study = getattr(self, "param_study", None)
                    sess.surrogate = getattr(self, "surrogate_model", None)

            if self.mcp_long(tool_name, v_args):
                new_model = tool_name in ("fea_study_new", "fea_study_open")
                from fea_core.mcp.common import set_call_cancel

                def job(th):
                    set_call_cancel(th.cancel)       # Cancel in the progress window = cancellation of the call
                    try:
                        if not new_model and self.step_path is not None:
                            sess._parts = self.shadow_parts()   # OCP work (mesh, report images) never on displayed shapes
                        return tool.fn(**v_args)
                    finally:
                        set_call_cancel(None)
                result = self.run_busy(job, f"LLM: {tool_name}…", cancellable=True, on_cancel=kill_mesh_workers)
            else:
                result = tool.fn(**v_args)

            # a tool that started a background job (fea_job_start, fea_sweep_run(background=True), …): the job holds the
            # session lock – taking the state over now would wait and fail; it is taken over when the job ends. busy()
            # refuses tools while a job runs, so an active job here was started by this very call.
            if tool_name == "fea_job_start" or (not job_tool and self._mcp_job_active()):
                self._watch_mcp_job(before)
            elif tool_name in ("fea_job_status", "fea_job_cancel") or read_only:
                if not job_tool and not self._mcp_job_active():
                    self._mcp_sync_surrogate(sess)           # fea_surrogate_fit counts as read-only for the study, but it changes the surrogate
                self.statusBar().showMessage(f"LLM: {tool_name} done", 4000)
                return result
            else:
                self._mcp_sync_back(before)
                self._mcp_sync_surrogate(sess)               # after the sync: a project opened through MCP brings its own surrogate

            self.refresh_all()
            if self.results is not None and tool_name in ("fea_solve_run", "fea_job_result"):
                self.set_stage(STAGE["Results"])
                self.show_result()
            elif self.mesh is not None and tool_name == "fea_mesh_generate":
                self.set_stage(STAGE["Mesh"])
                self.show_mesh()
            elif "constraint" in tool_name:
                self.set_stage(STAGE["Constraints"])
            elif "load" in tool_name:
                self.set_stage(STAGE["Loads"])
            elif "material" in tool_name:
                self.set_stage(STAGE["Materials"])

            self.view.update()
            self.statusBar().showMessage(f"LLM: {tool_name} done", 4000)
            return result
        finally:
            QApplication.restoreOverrideCursor()

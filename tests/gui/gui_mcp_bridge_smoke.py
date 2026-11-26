"""MCP ↔ GUI bridge and state management of the main window (review fixes, 9. 11. 2026).

  python tests/gui/hidden_run.py tests/gui/gui_mcp_bridge_smoke.py OUT_DIR

1 read-only tools do not mark the study dirty and write no undo snapshot; 2 a mutating tool = dirty + one undo
snapshot; 3 while the window is busy only status tools run; 4 long tools (mesh, solve) run in a worker – the GUI
event loop keeps running (a timer ticks during the call); 5 undo of an assessment change (required safety) keeps the
results; 6 a change during a running analysis asks once and "No" reverts it; 7 mark_changed levels; 8 buffered
solver lines; 9 soft_fail records the error in the Messages dock. Dialogs are replaced (msg, ask).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_mcp_bridge_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "ERROR ") + text)
    if not cond:
        FAIL.append(text)


def main():
    import fea_gui.main as gm
    from fea_core.geometry import Part, write_step
    from verification.cases import box

    import os
    os.environ["FEA_NO_IPC"] = "1"                       # other MCP clients of this user must not reach the test window
    gm.msg = lambda *a, **k: 0
    app = QApplication(sys.argv)
    w = gm.MainWindow()
    w.resize(1300, 850)
    w.show()
    asked: list[str] = []
    answer = {"v": True}

    def fake_ask(text):
        asked.append(str(text))
        return answer["v"]
    w.ask = fake_ask
    step = OUT / "beam.step"
    write_step([Part("Beam", box(0, 0, 0, 100, 10, 10))], step)

    def call(tool, **args):
        r = w.execute_mcp_action(tool, args)
        return json.loads(r) if isinstance(r, str) and r[:1] in "{[" else r

    def s_open():
        w.open_file(str(step))
        check(w.study is not None and len(w.parts) == 1 and not w.dirty, "model opened (worker read), not dirty")

    def s_read_only():
        n = len(w._undo)
        call("fea_study_status")
        call("fea_materials_list")
        call("fea_geometry_find_faces", part="Beam", x=0)
        check(not w.dirty and len(w._undo) == n, "read-only MCP tools: not dirty, no undo snapshot")
        check(gm.MainWindow.mcp_read_only("fea_results_render_image") and not gm.MainWindow.mcp_read_only("fea_solve_run")
              and not gm.MainWindow.mcp_read_only("fea_job_result"), "read-only classification")

    def s_mutating():
        n = len(w._undo)
        call("fea_loadcase_create", name="Second")
        check(w.dirty and len(w._undo) == n + 1 and any(lc.name == "Second" for lc in w.study.load_cases),
              "mutating MCP tool: dirty + one undo snapshot, study in the window updated")
        call("fea_constraint_add_fixed", name="Fix", selection={"part": "Beam", "x": 0})
        lc = w.study.load_cases[0].name
        call("fea_load_add_force", case_name=lc, name="Tip", selection={"part": "Beam", "x": 100}, force=[0, 0, -100])
        check(len(w.study.constraints) == 1 and len(w.study.load_cases[0].loads) == 1, "constraint + force from MCP")

    def s_busy():
        w._busy_task = object()
        st = call("fea_gui_status")
        try:
            call("fea_study_status")
            refused = False
        except RuntimeError as e:
            refused = "busy" in str(e)
        w._busy_task = None
        check(st.get("busy") is True and refused, "busy window: status answers, other tools refused")

    def s_long():
        ticks = {"n": 0}
        t = QTimer()
        t.setInterval(20)
        t.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
        t.start()
        call("fea_mesh_generate", global_size=5)
        call("fea_solve_run")
        t.stop()
        check(w.mesh is not None and w.results is not None and w.results.steps, "mesh + solve through MCP in the worker")
        check(ticks["n"] > 3, f"GUI event loop ran during the long tools (timer ticks {ticks['n']})")
        check(w._mesh_sig == w.study_signature(w.MESH_KEYS), "mesh from MCP adopted as the valid mesh")

    def s_sweep_job():
        """fea_sweep_run(background=True) starts a job like fea_job_start: the call returns at once (it used to wait 5 s for
        the session lock held by the job and fail with BusyError) and the window takes the state over when the job ends."""
        import time
        import fea_core.sweep as sweep_mod
        check(gm.MainWindow.mcp_long("fea_sweep_run", {"background": False}) and not gm.MainWindow.mcp_long("fea_sweep_run", {})
              and gm.MainWindow.mcp_long("fea_solve_run", {}), "fea_sweep_run is a long tool only when it waits (background=False)")
        orig_run = sweep_mod.run_parametric_study
        seen = []

        def spy(study, spec, wd, **kw):
            seen.append({"mesh": kw.get("mesh") is not None, "parts": kw.get("parts")})
            return orig_run(study, spec, wd, **kw)
        sweep_mod.run_parametric_study = spy
        call("fea_sweep_define", load_factors=[1.0, 2.0])
        # a waiting sweep runs in the worker (the GUI stays painted) and takes the valid mesh of the window
        r0 = call("fea_sweep_run", background=False)
        check(isinstance(r0, dict) and len(r0.get("points", [])) == 2 and seen and seen[-1]["mesh"] and seen[-1]["parts"] is None,
              f"sweep with background=False through the bridge: 2 points, the valid mesh of the window reused ({seen[-1:]})")
        check(w._mcp_session.valid_mesh() is not None, "the session of the bridge knows that the mesh of the window is valid")
        seen.clear()
        t0 = time.time()
        r = call("fea_sweep_run")
        dt = time.time() - t0
        timer = getattr(w, "_mcp_job_timer", None)
        check(isinstance(r, dict) and r.get("job_id") and dt < 4.0 and timer is not None and timer.isActive(),
              f"background sweep through the bridge: job started, returned in {dt:.1f} s, job watched")
        end = time.time() + 180
        while (w._mcp_job_active() or timer.isActive()) and time.time() < end:
            app.processEvents()
            time.sleep(0.05)
        st = call("fea_job_status")
        check(not w.busy() and st.get("status") == "done", f"sweep job finished ({st.get('status')}), window not busy")
        sweep_mod.run_parametric_study = orig_run
        check(seen and seen[-1]["mesh"], "the background sweep job took the valid mesh of the window")

    def s_assessment_undo():
        res = w.results
        w.study.assessment = dict(w.study.assessment or {}, required_safety=2.5)
        w.mark_changed("none")
        check(w.results is res, "assessment change keeps the results")
        w.undo()
        check(w.results is res and (w.study.assessment or {}).get("required_safety") != 2.5,
              "undo of the assessment change keeps the results")
        w.redo()
        check(w.results is res, "redo of the assessment change keeps the results")

    def s_levels():
        res, mesh = w.results, w.mesh
        w.mark_changed("assess")
        check(w.results is res and w.mesh is mesh and w.joints_panel.assessment is None, "mark_changed('assess')")
        w.mark_changed("results")
        check(w.results is None and w.mesh is mesh, "mark_changed('results') discards the results only")
        w.mark_changed("mesh")
        check(w.mesh is None, "mark_changed('mesh') discards the mesh")
        with w.batch_changes():
            w.changed()
            w.changed()
            pending = w._refresh_pending
        check(pending and not w._refresh_pending, "batch_changes: one refresh at the end")

    def s_change_while_busy():
        w._thread = gm.SolveThread(gm.Study.from_dict(w.study.to_dict()), OUT / "never")   # not started
        w._thread_sig = w.study_signature()
        w._busy_warned = False
        n_lc = len(w.study.load_cases)
        answer["v"] = False
        asked.clear()
        w.study.load_cases.append(gm.LoadCase("During"))
        ok = w.changed()
        check(len(asked) == 1 and ok is False and len(w.study.load_cases) == n_lc,
              "change during an analysis: asked, 'No' reverts the change")
        answer["v"] = True
        w.study.load_cases.append(gm.LoadCase("During 2"))
        w.changed()
        check(len(asked) == 2 and len(w.study.load_cases) == n_lc + 1, "asked again after 'No'; 'Yes' keeps the change")
        w.study.load_cases.append(gm.LoadCase("During 3"))
        w.changed()
        check(len(asked) == 2, "after 'Yes' not asked again in the same run")
        w._thread = None
        w._busy_warned = False

    def s_solver_lines():
        before = w.log.blockCount()
        for i in range(500):
            w.append_solver_line(f"line {i}")
        mid = w.log.blockCount()
        w.flush_solver_lines()
        check(mid == before and w.log.blockCount() >= before + 500 and w.log.maximumBlockCount() > 0,
              "solver lines buffered, flushed in one batch")

    def s_soft_fail():
        from fea_gui.widgets import soft_fail
        with soft_fail("tables", w, "Test helper") as sf:
            raise ValueError("boom")
        msgs = w.messages.messages(source="tables")
        check(sf.failed and any("boom" in m.get("text", "") for m in msgs), "soft_fail → warning in the Messages dock")

    steps = [s_open, s_read_only, s_mutating, s_busy, s_long, s_sweep_job, s_assessment_undo, s_levels, s_change_while_busy,
             s_solver_lines, s_soft_fail]

    def run_next():
        if not steps:
            w.dirty = False
            app.quit()
            return
        f = steps.pop(0)
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            FAIL.append(f.__name__)
        QTimer.singleShot(100, run_next)

    QTimer.singleShot(400, run_next)
    app.exec()
    print("RESULT:", "OK" if not FAIL else f"{len(FAIL)} errors: " + "; ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

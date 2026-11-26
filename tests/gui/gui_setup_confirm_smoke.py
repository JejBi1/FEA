"""Setup confirmation before an AI calculation in the main window (F15-S1).

  python tests/gui/hidden_run.py tests/gui/gui_setup_confirm_smoke.py OUT_DIR        (FEA_LANG=cs | en)

1 the session hook and the setting (Tools → "Ask before an AI calculation", on by default); 2 fea_setup_review through the bridge opens
the non-modal dialog (numbered picture, table with the colours of the picture) and returns text; 3 the calculation is refused until the
user clicks Confirm – the digest passed by the assistant is not enough, nor is a decision of the agent kind; 4 a click on a row marks
its faces in the 3D window, the numbers are drawn there and removed when the dialog closes; 5 Confirm → the calculation runs and the record
(user, gui) is in the results; 6 a setup changed after the review is NOT confirmed by a click on the old dialog; 7 Reject is reported to the
assistant; 8 the setting off = no question; 9 the window's own calculations never ask; 10 the Czech / English texts.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_setup_confirm_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "ERROR ") + text)
    if not cond:
        FAIL.append(text)


def main():
    import fea_gui.main as gm
    from fea_core import i18n
    from fea_core.geometry import Part, write_step
    from fea_core.mcp import setup_proof
    from fea_gui import setup_confirm as sc
    from verification.cases import box

    os.environ["FEA_NO_IPC"] = "1"                       # other MCP clients of this user must not reach the test window
    gm.msg = lambda *a, **k: 0
    app = QApplication(sys.argv)
    w = gm.MainWindow()
    w.resize(1300, 850)
    w.show()
    asked: list[str] = []
    w.ask = lambda text: (asked.append(str(text)) or True)
    sess = w._mcp_session
    step = OUT / "beam.step"
    write_step([Part("Beam", box(0, 0, 0, 60, 10, 10))], step)
    stored = w.settings.value(sc.SETTING, None)
    w.settings.remove(sc.SETTING)                         # the test starts from "never chosen" (= on); the choice is restored at the end

    def call(tool, **args):
        r = w.execute_mcp_action(tool, args)
        return json.loads(r) if isinstance(r, str) and r[:1] in "{[" else r

    def refused(tool, **args) -> str:
        try:
            w.execute_mcp_action(tool, args)
        except Exception as e:  # noqa: BLE001 – the message is what the assistant gets
            return str(e)
        return ""

    state: dict = {}

    def s_open():
        w.open_file(str(step))
        check(w.study is not None and len(w.parts) == 1, "model opened")
        call("fea_materials_assign", part="*", material_name="Ocel S235")
        call("fea_constraint_add_fixed", name="Fixed end", selection={"part": "Beam", "x": 0})
        call("fea_load_add_force", case_name="Pull", name="Tip", selection={"part": "Beam", "x": 60}, force=[0, 0, -100])
        call("fea_load_add_gravity", case_name="Pull")
        pull = next(lc for lc in w.study.load_cases if lc.name == "Pull")
        check(len(w.study.constraints) == 1 and len(pull.loads) == 2, "support and two loads from the assistant")
        call("fea_mesh_generate", global_size=5)             # the mesh size is part of the setup: mesh BEFORE the review
        check(w.mesh is not None, "mesh generated before the review")

    def s_hook_and_setting():
        host = getattr(w, "_setup_confirm", None)
        check(host is not None and sess.confirm_ui is host, "the session of the window has the confirmation hook")
        check(sess.require_confirmation is True, "ask before an AI calculation is ON by default in the window")
        a = getattr(w, "act_confirm_ai", None)
        check(a is not None and a.isCheckable() and a.isChecked() and any(x is a for x in w.tools_button.menu().actions()),
              "Tools menu: checkable action 'Ask before an AI calculation', checked")
        a.setChecked(False)
        check(sess.require_confirmation is False and not sc.ask_setting(w.settings), "unchecked: the session does not ask; the choice is stored")
        d = call("fea_setup_review", output_path=str(OUT / "off.png"))
        check(d["confirmation"]["required"] is False and w._setup_confirm.dlg is None, "not required: the review opens no dialog")
        a.setChecked(True)
        check(sess.require_confirmation is True and sc.ask_setting(w.settings), "checked again: on, stored")

    def s_review_opens_dialog():
        check(gm.MainWindow.mcp_long("fea_setup_review") and gm.MainWindow.mcp_read_only("fea_setup_review"),
              "the review is a long (worker) and read-only tool of the bridge")
        r = w.execute_mcp_action("fea_setup_review", {"output_path": str(OUT / "review.png")})
        app.processEvents()                                              # the dialog is opened by the GUI thread (queued from the worker)
        check(isinstance(r, str), "with a window the review is text only (the picture is in the dialog)")
        d = json.loads(r)
        state["d"] = d
        dlg = w._setup_confirm.dlg
        check(dlg is not None and dlg.isVisible() and not dlg.isModal(), "the non-modal dialog is open")
        check(Path(d["picture_shown_to_user"].split("file: ")[1].rstrip(")")).is_file() and dlg.picture.pixmap() is not None
              and not dlg.picture.pixmap().isNull(), "the numbered picture is in the dialog")
        rows = dlg.payload["rows"]
        check(dlg.table.rowCount() == 3 and [r["no"] for r in rows] == [1, 2, 3], f"table: {dlg.table.rowCount()} rows numbered 1..3")
        check(all(r.get("color") for r in rows) and dlg.table.item(0, 0).background().color().isValid(), "the number cells carry the colours of the picture")
        check(d["digest"] == dlg.payload["digest"] == setup_proof.session_digest(sess), "digest = the digest of the window's setup")
        check(dlg.btn_confirm.isEnabled() and dlg.btn_reject.isEnabled(), "waiting for the decision: Confirm and Reject enabled")
        check(sorted(w._setup_confirm._keys) == ["setupconf:1", "setupconf:2"], f"numbers drawn in the 3D window: {w._setup_confirm._keys}")
        check(dlg.grab().save(str(OUT / "dialog.png")), "snapshot of the dialog saved")

    def s_row_click():
        dlg = w._setup_confirm.dlg
        dlg.table.selectRow(1)                                         # Tip (force)
        app.processEvents()
        check(len(w.view.selected_faces()) == 1, f"a click on a row marks its faces in the 3D window ({w.view.selected_faces()})")
        dlg.table.selectRow(0)
        app.processEvents()
        faces0 = w.view.selected_faces()
        check(len(faces0) == 1 and faces0 != [], "another row marks another face")

    def s_refused_without_the_click():
        d = state["d"]
        err = refused("fea_solve_run", confirmed=d["digest_short"])
        check("has not confirmed" in err, f"calculation refused without the click, although the assistant passed the digest: {err[:90]}")
        check(w.results is None, "no results")
        err = refused("fea_job_start", kind="solve", confirmed=d["digest_short"])
        check("has not confirmed" in err, "fea_job_start refused too")
        setup_proof.record_decision(sess, d["digest"], "confirmed", "agent", "agent")
        check("has not confirmed" in refused("fea_solve_run", confirmed=d["digest_short"]), "a decision of the agent kind does not count in the window")
        sess.confirmation = None

    def s_confirm_click():
        dlg = w._setup_confirm.dlg
        dlg.btn_confirm.click()
        rec = sess.confirmation
        check(rec and rec["decision"] == "confirmed" and rec["by"] == "user" and rec["channel"] == "gui", f"click on Confirm: {rec}")
        check(not dlg.btn_confirm.isEnabled() and dlg.btn_reject.isEnabled(), "Confirm disabled after the click")
        d = state["d"]
        r = call("fea_solve_run", confirmed=d["digest_short"])
        check(isinstance(r, dict) and r.get("ok") and r.get("setup_confirmed_by") == "user", f"the calculation runs after the click: {str(r)[:100]}")
        info = (w.results.info if w.results is not None else {}).get("setup_confirmation")
        check(info and info["confirmed_by"] == "user" and info["channel"] == "gui" and info["digest"] == d["digest"],
              f"the record is in the results: {info}")

    def s_changed_after_review():
        # a new review of the (unchanged) setup, then the assistant changes a load, then the user clicks on the OLD dialog
        d = call("fea_setup_review", output_path=str(OUT / "review2.png"))
        app.processEvents()
        dlg = w._setup_confirm.dlg
        check(dlg.btn_reject.isEnabled() and not dlg.btn_confirm.isEnabled(), "a review of an already confirmed setup shows 'confirmed'")
        sess.confirmation = None
        dlg.load(dlg.payload)
        check(dlg.btn_confirm.isEnabled(), "decision cleared: Confirm enabled again")
        call("fea_load_add_force", case_name="Pull", name="Tip", selection="Tip", force=[0, 0, -200])
        dlg.btn_confirm.click()
        check(dlg.stale and not dlg.btn_confirm.isEnabled() and sess.confirmation is None,
              "the setup changed after the review: the click does NOT confirm, the dialog says so")
        err = refused("fea_solve_run", confirmed=d["digest_short"])
        check("changed since the review" in err or "has not confirmed" in err, f"calculation refused: {err[:80]}")

    def s_reject():
        d = call("fea_setup_review", output_path=str(OUT / "review3.png"))
        app.processEvents()
        dlg = w._setup_confirm.dlg
        check(not dlg.stale and dlg.btn_confirm.isEnabled(), "a new review of the changed setup: a fresh dialog state")
        dlg.btn_reject.click()
        check(sess.confirmation and sess.confirmation["decision"] == "rejected" and sess.confirmation["by"] == "user", "click on Reject recorded")
        err = refused("fea_solve_run", confirmed=d["digest_short"])
        check("REJECTED" in err, f"the assistant is told: {err[:70]}")
        dlg.close()
        app.processEvents()
        check(w._setup_confirm._keys == [] and not any(k.startswith(sc.KEY_PREFIX) for k in w.view._sets), "closing the dialog removes the numbers from the 3D window")
        check(w._setup_confirm.dlg is None, "the dialog is gone (a new review creates a new one)")

    def s_setting_off():
        w.act_confirm_ai.setChecked(False)
        from fea_core.mcp import tools_setup
        check(tools_setup.gate(sess, None) is None, "setting off: the gate lets the calculation through without asking")
        w.act_confirm_ai.setChecked(True)

    def s_own_runs_never_ask():
        src = (Path(sc.__file__).parent / "main_mesh_solve.py").read_text(encoding="utf-8")
        check("tools_setup" not in src and "solve_core" not in src and "setup_proof" not in src,
              "the window's own calculation (Run) does not go through the gate")

    def s_texts():
        keys = ["Confirm", "Reject", "Show in 3D", "Ask before an AI calculation", "Confirm the setup before the calculation"]
        tr = {k: i18n.tr(k) for k in keys}
        if i18n.language() == "cs":
            check(all(tr[k] != k for k in keys), f"Czech texts of the dialog: {tr}")
            check(tr["Confirm"] == "Potvrdit" and tr["Reject"] == "Odmítnout", "Czech buttons")
        else:
            check(all(tr[k] == k for k in keys), "English texts are the originals")

    steps = [s_open, s_hook_and_setting, s_review_opens_dialog, s_row_click, s_refused_without_the_click, s_confirm_click,
             s_changed_after_review, s_reject, s_setting_off, s_own_runs_never_ask, s_texts]

    def run_next():
        if not steps:
            w.dirty = False
            if stored is None:
                w.settings.remove(sc.SETTING)
            else:
                w.settings.setValue(sc.SETTING, stored)
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

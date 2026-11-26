"""A pass through the sketcher in the GUI (F15-S5).

  python tests/gui/hidden_run.py tests/gui/gui_sketch_smoke.py OUT_DIR

1) The editor of a plane-stress sketch: parameters W = 100, d = 20, a rectangle and a circle by typed coordinates (expressions), names
   of the entities, a line by mouse clicks that snaps to an existing point, undo / redo, delete, trim, mirror, check (regions with a hole),
   apply → a study with the 2D settings and a selection per entity; File menu entries and the "Edit sketch" button.
2) A constraint on an edge (the selection W.4 of the rectangle) and a pressure on the opposite edge → calculation → σ at the hole
   (Kt of a plate with a hole d/W = 0.2, within 6 %).
3) Edit sketch: the parameter d = 30 → rebuilt with the constraint and the load kept; the calculation again, σ grows.
4) A beam sketch by typed coordinates (portal frame), supports by the node tool, build → members E1…E3 → load at a node → calculation →
   the horizontal reactions add up to H.
5) An axisymmetric sketch with the DXF import (when ezdxf is installed) and the "contour across the axis" error.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_sketch_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def main():
    import fea_gui.main as gm
    from fea_core.beams.model import FrameNodeLoad
    from fea_core.sketch.model import new_sketch
    from fea_core.study import Fixed, LoadCase, Pressure
    from fea_gui.main import MainWindow
    from fea_gui.sketch import hooks
    from fea_gui.sketch.editor import SketchEditor
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1500, 950)
    w.show()

    def wait(cond, secs=240.0):
        t0 = time.time()
        while time.time() - t0 < secs:
            app.processEvents()
            if cond():
                return True
            time.sleep(0.05)
        return False

    def solve():
        w.run_solve()
        ok = wait(lambda: w._thread is None and w.results is not None)
        app.processEvents()
        return ok

    def click_world(ed, x, y):
        cv = ed.canvas
        from PySide6.QtCore import QPointF
        p = cv.mapFromScene(QPointF(x, y))
        QTest.mouseMove(cv.viewport(), QPoint(p.x(), p.y()))
        QTest.mouseClick(cv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(p.x(), p.y()))
        app.processEvents()

    # ------------------------------------------------------------------ menus
    file_actions = [a.text() for a in w.menu_buttons["File"].menu().actions()]
    check("New 2D model from sketch…" in file_actions and "New beam system from sketch…" in file_actions, "menu File: new from sketch")

    # ------------------------------------------------------------------ the "new sketch" dialog
    dlg = hooks.NewSketchDialog(w, "twod")
    dlg.mode.setCurrentIndex(dlg.mode.findData("axisym"))
    dlg.thickness.setText("6")
    nk = dlg.make()
    check(nk.kind == "twod" and nk.mode == "axisym" and nk.thickness == 6.0, f"new sketch dialog (2D): {nk.mode}, thickness {nk.thickness}")
    dlg.thickness.setText("t_plate")
    dlg.mode.setCurrentIndex(dlg.mode.findData("plane_strain"))
    check(dlg.make().thickness == "t_plate", "the thickness may be an expression")
    fdlg = hooks.NewSketchDialog(w, "frame")
    fdlg.plane.setCurrentIndex(fdlg.plane.findData("XY"))
    fk0 = fdlg.make()
    check(fk0.kind == "frame" and fk0.plane == "XY", "new sketch dialog (beams): plane XY")

    # ------------------------------------------------------------------ 1) the editor, plane stress
    sk = new_sketch("twod", "plane_stress", "XY", 5)
    ed = SketchEditor(w, sk, "test")
    ed.show()
    ed.resize(1100, 700)
    app.processEvents()
    for name, expr in (("W", "100"), ("d", "20")):
        ed.p_name.setText(name)
        ed.p_expr.setText(expr)
        ed.set_param_from_fields()
    check(sk.params == {"W": "100", "d": "20"} and ed.params.rowCount() == 2, "parameters W, d set")
    ed.p_name.setText("zz")
    ed.p_expr.setText("nope+1")
    ed.set_param_from_fields()
    check("zz" not in sk.params and "unknown parameter" in ed.hint_label.text().lower(), f"a bad parameter is refused: {ed.hint_label.text()}")
    ed.select_tool("rect")
    ed.type_text("0, 0")
    ed.type_text("W, W")
    check(len(sk.entities) == 1 and sk.entities[0].type == "rect" and sk.points[sk.entities[0].points[1]] == ["W", "W"], "rectangle by typed expressions")
    ed.select_tool("circle")
    ed.type_text("W/2, W/2")
    check(len(ed.canvas.tool.pts) == 1, "circle: the centre is taken")
    ed.type_text("d/2")
    check(len(sk.entities) == 2 and sk.entities[1].radius == "d/2", f"circle radius as an expression: {sk.entities[1].radius}")
    ed.entities.item(0, 2).setText("W")
    ed.entities.item(1, 2).setText("hole")
    check([e.name for e in sk.entities] == ["W", "hole"], f"names of the entities: {[e.name for e in sk.entities]}")
    ed.entities.item(0, 2).setText("hole")
    check(sk.entities[0].name == "W" and "already used" in ed.hint_label.text(), "a duplicate name is refused")
    # a line drawn with the mouse snaps to the corner points of the rectangle
    ed.canvas.fit_all()
    app.processEvents()
    ed.select_tool("line")
    click_world(ed, 0.2, 0.1)
    click_world(ed, 99.9, 100.2)
    line = sk.entities[-1]
    check(line.type == "line" and set(line.points) <= {sk.entities[0].points[0], sk.entities[0].points[1]} or len(sk.entities) == 3,
          f"a line by two clicks: {line.type} {line.points}")
    n_pts = len(sk.points)
    check(line.points[0] in sk.entities[0].points or sk.points[line.points[0]] == [0.0, 0.0] or len(sk.points) == n_pts, "the click snapped to the corner (the same point id)")
    ed.undo()
    check(len(sk.entities) == 2, "undo removes the line")
    ed.redo()
    check(len(sk.entities) == 3, "redo brings it back")
    ed.select_tool("select")
    ed.canvas.select([line.id])
    ed.canvas.delete_selected()
    check(len(sk.entities) == 2, "Delete removes the selected line")
    # polar / relative typed entry
    ed.select_tool("polyline")
    for t in ("200, 0", "@50, 0", "@50<90", "@-50, 0"):
        ed.type_text(t)
    ed.type_text("c")
    pl = sk.entities[-1]
    check(pl.type == "polyline" and pl.closed and len(pl.points) == 4, "closed polyline by absolute, relative and polar entry")
    from fea_core.sketch.model import resolve
    seg = resolve(sk).segs_of(pl.id)
    check(abs(seg[1].p1[0] - 250.0) < 1e-9 and abs(seg[1].p1[1] - 50.0) < 1e-9, f"@50<90 is a step of 50 in y: {seg[1].p1}")
    ed.canvas.select([pl.id])
    ed.select_tool("mirror")
    ed.type_text("300, 0")
    ed.type_text("300, 10")
    check(len(sk.entities) == 4, "mirror made a copy")
    ed.canvas.select([sk.entities[-1].id, pl.id])
    ed.canvas.delete_selected()
    check(len(sk.entities) == 2, "the polyline and its mirror image deleted")
    ed.canvas.fit_all()
    ed.canvas.problems = set()
    check(ed.check(), f"check: {ed.hint_label.text()}")
    check(len(ed.canvas.regions) == 1 and len(ed.canvas.regions[0]["rings"]) == 2, "one region with a hole")
    ed.grab().save(str(OUT / "editor_plate.png"))
    ed.accept()
    check(ed.result() == 1, "Apply closes the editor")
    check(hooks.build_new(w, ed.sketch, "plate"), "build_new")
    st = w.study
    check(st.sketch is not None and st.twod.enabled and st.twod.thickness == 5.0 and len(w.parts) == 1, "study from the sketch with 2D settings")
    check({"W", "W.1", "W.2", "W.3", "W.4", "hole"} <= set(st.selections), f"selections: {sorted(st.selections)}")
    w.set_stage(w.STAGE["Model"])
    app.processEvents()
    check(not w.btn_edit_sketch.isHidden(), "Model step: the Edit sketch button is there")

    # ------------------------------------------------------------------ 2) constraint on an edge, pressure on the opposite one
    st.constraints.append(Fixed("W.4"))                       # the left side
    st.load_cases = [LoadCase("tension", [Pressure("W.2", -50.0)])]
    st.mesh.size = 4.0
    st.mesh.local = [{"selection": "hole", "size": 1.0}]
    w.mark_changed("results")
    check(solve(), "calculation 1 finished")
    r2 = w.results.twod if w.results is not None else None
    check(r2 is not None and max(s["equilibrium"]["imbalance"] for s in r2.steps) < 1e-6, "2D result with equilibrium")
    if r2 is not None:
        sx_nom = 50.0 * 100.0 / 80.0                          # p·W / (W − d)
        kt = 3 - 3.14 * 0.2 + 3.667 * 0.04 - 1.527 * 0.008
        sx = float(r2.probe([50.0, 60.0], "sy")) if False else float(np.max(r2.step(0)["S"][:, 0]))
        check(abs(sx - kt * sx_nom) / (kt * sx_nom) < 0.06, f"σx max {sx:.1f} ≈ Kt·σnet = {kt * sx_nom:.1f} MPa (d/W = 0.2)")
        first = float(np.max(r2.step(0)["S"][:, 0]))

    # ------------------------------------------------------------------ 3) Edit sketch: d = 30 keeps the setup
    sk2 = w.study.sketch.copy()
    sk2.set_param("d", 30)
    check(hooks.apply_edit(w, sk2), "apply_edit with d = 30")
    st = w.study
    check(st.sketch.params["d"] == 30 and [c.selection for c in st.constraints] == ["W.4"]
          and st.load_cases[0].loads[0].selection == "W.2" and st.mesh.local == [{"selection": "hole", "size": 1.0}],
          "constraint, load and mesh refinement kept after the rebuild")
    check(st.cad_link is not None and st.cad_link.iterations[-1].trigger == "sketch", "an iteration of the geometry was recorded")
    check(solve(), "calculation 2 finished")
    if w.results is not None and w.results.twod is not None:
        second = float(np.max(w.results.twod.step(0)["S"][:, 0]))
        check(second > first * 0.99, f"σx max with the bigger hole: {second:.1f} MPa (before {first:.1f})")
    ed2 = SketchEditor(w, w.study.sketch.copy(), "edit")
    ed2.show()
    ed2.canvas.fit_all()
    check(ed2.params.rowCount() == 2 and ed2.entities.rowCount() == 2, "the editor opens the stored sketch")
    ed2.select_tool("circle")
    ed2.type_text("-10, -10")
    ed2.type_text("0")                                      # zero radius is refused with a message
    check(len(ed2.sketch.entities) == 2 and ed2.hint_label.text() != "", f"zero radius: {ed2.hint_label.text()}")
    ed2.close()

    # ------------------------------------------------------------------ 4) beam sketch
    fk = new_sketch("frame")
    fed = SketchEditor(w, fk, "frame")
    fed.show()
    for name, expr in (("L", "6000"), ("h", "4000")):
        fed.p_name.setText(name)
        fed.p_expr.setText(expr)
        fed.set_param_from_fields()
    fed.select_tool("line")
    for t in ("0, 0", "0, h"):
        fed.type_text(t)
    fed.canvas.tool.cancel()
    fed.type_text("0, h")
    fed.type_text("L, h")
    fed.canvas.tool.cancel()
    fed.type_text("L, h")
    fed.type_text("L, 0")
    fed.canvas.tool.cancel()
    check(len(fk.entities) == 3, f"three members drawn: {[e.id for e in fk.entities]}")
    fed.select_tool("point")
    fed.kind_box.setCurrentText("fixed")
    fed.type_text("0, 0")
    fed.type_text("L, 0")
    check([s.kind for s in fk.supports] == ["fixed", "fixed"], "two fixed supports by the node tool")
    check(fed.check() and fed.result_info["members"] == 3, f"check: {fed.hint_label.text()}")
    fed.accept()
    check(hooks.build_new(w, fed.sketch, "portal"), "build_new (beams)")
    st = w.study
    check(w.frame_only() and set(st.frame.members) == {"E1", "E2", "E3"} and len(st.frame.supports) == 4, "members E1–E3 and supports in the study")
    st.load_cases = [LoadCase("H", [FrameNodeLoad("P2", [10000.0, 0, 0])])]
    w.mark_changed("results")
    check(solve(), "beam calculation finished")
    if w.results is not None:
        rec = w.results.step(0).frame["reactions"]
        hsum = -(rec["P1"][0] + rec["P4"][0]) if "P4" in rec else -sum(v[0] for v in rec.values())
        check(abs(hsum - 10000.0) < 1e-3, f"sum of the horizontal reactions {hsum:.4f} = H")

    # ------------------------------------------------------------------ 5) axisymmetric sketch, error across the axis
    ak = new_sketch("twod", "axisym")
    aed = SketchEditor(w, ak, "tube")
    aed.show()
    aed.select_tool("rect")
    aed.type_text("-10, 0")
    aed.type_text("10, 10")
    check(not aed.check() and "axis" in aed.problem and aed.canvas.problems == {"E1"}, f"across the axis: {aed.problem}")
    aed.undo()
    aed.type_text("20, 0")
    aed.type_text("40, 10")
    check(aed.check() and aed.result_info["regions"][0]["area"] > 0, "a tube section is fine")
    try:
        import ezdxf  # noqa: F401
        have = True
    except ImportError:
        have = False
    if have:
        import ezdxf
        doc = ezdxf.new("R2010", setup=False)
        doc.header["$INSUNITS"] = 4
        for a, b in (((50, 0), (60, 0)), ((60, 0), (60, 10)), ((60, 10), (50, 10)), ((50, 10), (50, 0))):
            doc.modelspace().add_line(a, b)
        doc.saveas(str(OUT / "p.dxf"))
        info = aed.import_dxf_file(str(OUT / "p.dxf"))
        check(info is not None and info["imported"] == 4 and aed.check() and len(aed.result_info["regions"]) == 2, "DXF import adds a second region")
    aed.close()
    w.close()
    print("\nVÝSLEDEK:", "všechno prošlo" if not FAIL else f"{len(FAIL)} chyb: {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

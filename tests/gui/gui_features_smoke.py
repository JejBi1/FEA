"""A pass through the GUI of the core features added to the GUI in 1.5: EN 1990 load case categories (combinations
dialog + Tables dock), the h_c field of a contact pair, ε-N fatigue with own parameters, and that post-processing edits keep
the results. (Plasticity, cavity radiation, submodel and the convergence monitor have their own smoke tests:
gui_plasticity / gui_thermal_extras / gui_submodel / gui_extras.)

  python tests/gui/hidden_run.py tests/gui/gui_features_smoke.py OUT_DIR

Model: steel cantilever 100×20×10, clamped at x = 0, force −1000 N in z on the end x = 100, element 5 mm.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_features_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from fea_core.geometry import Part, write_step
    from verification.cases import box
    write_step([Part("Beam", box(0, 0, 0, 100, 20, 10))], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow, FORCE
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")      # without modal windows
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)   # closing the window: do not save
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    step = OUT / "beam.step"
    make_step(step)
    log = []
    state = {}

    def shot(name, extra=None):
        for _ in range(3):
            app.processEvents()
        w.view.redraw()
        app.processEvents()
        from PySide6.QtGui import QImage, QPainter
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        v = QImage(str(tmp))
        p = QPainter(img)
        p.drawImage(w.view.mapTo(w, w.view.rect().topLeft()), v.scaled(w.view.size()))
        p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))
        if extra is not None:
            extra.grab().save(str(OUT / f"{name}_dialog.png"))
        log.append(name)

    def faces_where(pred):
        from fea_core.geometry import face_props
        return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                if pred(face_props(f)["bbox"])]

    def plane(ax, v):
        return faces_where(lambda b: abs(b[ax] - v) < 1e-6 and abs(b[ax + 3] - v) < 1e-6)

    # ------------------------------------------------------------------ steps
    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.ms_size.setValue(5.0)
        check(w.study.mesh.size == 5.0, "element size 5 mm")

    def s_bc():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(plane(0, 0.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(plane(0, 100.0), emit=True)
        w.ld_type.setCurrentText(FORCE)
        for s_, v in zip(w.ld_vec, (0.0, 0.0, -1000.0)):
            s_.setValue(v)
        w.add_load()
        from fea_core.study import LoadCase
        lc = w.study.load_cases[0]
        w.study.load_cases.append(LoadCase("Snow", [copy.deepcopy(lc.loads[0])]))
        w.changed()
        check(len(w.study.load_cases) == 2, "two load cases")

    def s_categories():
        from fea_gui.combinations_dialog import CombinationsDialog
        st = w.study
        dlg = CombinationsDialog(w)
        dlg.show()
        check(dlg.table_cats.rowCount() == 2, "category table: one row per load case")
        src = dlg.table_cats.item(1, 3).text()
        check(src.startswith("!"), f"'Snow' guessed from the name: {src!r}")
        cb = dlg.cat_combos[st.load_cases[0].name]
        cb.setCurrentIndex(cb.findData("permanent"))
        check(st.load_cases[0].category == "permanent", f"category set: {st.load_cases[0].category}")
        check(not dlg.table_cats.item(0, 3).text().startswith("!"), "source = set")
        shot("2_categories", dlg)
        # Tables dock: category column of the load case table
        td = w.table_dock
        td.show_tab("cases")
        td.refresh()
        t = td.tables["cases"]
        cb2 = t.cellWidget(1, 4)
        check(cb2 is not None and cb2.currentData() == "", "dock: Snow has no explicit category")
        if cb2 is not None:
            cb2.setCurrentIndex(cb2.findData("snow_cz_low"))
        check(st.load_cases[1].category == "snow_cz_low", f"dock sets the category: {st.load_cases[1].category}")
        dlg._category_info()
        # the classification dialog (main) is pre-filled from the stored categories; "OK" without a change
        import fea_gui.combinations_dialog as cd
        seen = {}
        orig_init = cd.ClassificationDialog.__init__

        def spy(self, parent, rows):
            seen["rows"] = rows
            orig_init(self, parent, rows)
        cd.ClassificationDialog.__init__ = spy
        cd.ClassificationDialog.exec = lambda self: True
        dlg._generate_en1990()
        rows = {r["name"]: r for r in seen["rows"]}
        check(rows[st.load_cases[0].name]["action_type"] == "permanent" and rows[st.load_cases[1].name]["category"] == "snow_cz_low"
              and rows[st.load_cases[1].name]["recognized"], f"dialog pre-filled from the stored categories: {rows}")
        check(st.load_cases[0].category == "permanent" and st.load_cases[1].category == "snow_cz_low",
              "confirmed categories stay stored in the load cases")
        check(bool(st.combinations), f"combinations generated ({len(st.combinations)})")
        check(any(abs(c.factors.get(st.load_cases[0].name, 0) - 1.35) < 1e-9 for c in st.combinations if not c.envelope),
              "permanent case gets γ_G = 1.35 in a combination")
        dlg.close()

    def s_conductance():
        cp = w.contact_panel
        cp.p_a.setEditText("Beam")
        cp.p_b.setEditText("Beam")
        cp.p_kind.setCurrentIndex(max(0, cp.p_kind.findData("frictionless")))   # h_c has no meaning for a bonded pair
        cp.p_hc.setValue(2500.0)
        pc = cp.pair_from_form("t")
        check(pc is not None and pc.conductance == 2500.0, f"pair conductance {pc and pc.conductance}")
        cp.p_hc.setValue(0.0)
        check(cp.pair_from_form("t").conductance is None, "0 = perfect thermal contact (None)")

    def s_solve():
        w.set_stage(w.STAGE["Solve"])
        errors, _ = w.check()
        check(not errors, f"study ready ({errors})")
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None and len(res.steps) == 2, "results of two load cases")
        if res is None:
            return
        w.set_stage(w.STAGE["Results"])
        app.processEvents()

    def s_postproc_keeps_results():
        # EN 1990 category / combinations are post-processing: undo and a "running" solve keep the results
        from fea_gui import load_categories as lcat
        from fea_gui.workers import SolveThread
        res = w.results
        lc = w.study.load_cases[1]
        lcat.set_category(w, lc, "cat_B")
        check(w.results is res, "category change keeps the results")
        w.undo()
        check(w.results is res and w.study.load_cases[1].category == "snow_cz_low", "undo of a category keeps the results")
        w.redo()
        check(w.results is res, "redo of a category keeps the results")
        fake = SolveThread(w.study_for_solve(), w._workdir / "fake", w, mesh=None, parts=None)
        w._thread, w._thread_sig = fake, w.study_signature()
        asked = []
        ask = w.ask
        w.ask = lambda text: asked.append(text) or True
        lcat.set_category(w, w.study.load_cases[1], "cat_C")
        check(not asked and w.study_signature() == w._thread_sig, "category change during a solve: no question, result valid")
        # a calculation input changes and the run ends while the question is open: "No" reverts the change and the
        # result that arrived meanwhile stays (it is judged after the answer)
        lc0 = w.study.load_cases[0]
        old_val = list(lc0.loads[0].vector)

        def finish_then_no(text):
            asked.append(text)
            w.on_solve_done(res)                  # done arrives during the question …
            w._thread = None                      # … and the thread has finished
            return False
        w.ask = finish_then_no
        lc0.loads[0].vector = [0.0, 0.0, -2000.0]
        w.mark_changed("results")
        check(len(asked) == 1 and list(w.study.load_cases[0].loads[0].vector) == old_val,
              "change during a run: question asked, 'No' reverted it")
        check(w.results is res, "result of the run that ended during the question kept")
        w.ask = ask
        fake.deleteLater()

    def s_fatigue_settings():
        fp = w.fat_panel
        w.set_stage(w.STAGE["Fatigue"])
        fp.method.setCurrentIndex(fp.method.findData("en"))
        fs = w.study.fatigue
        check(fs.method == "en" and fs.mean_correction in ("none", "morrow", "swt"), f"ε-N method: {fs.method} {fs.mean_correction}")
        check(fp.en_box.isVisible(), "ε-N section visible")
        # method switching: the mean-stress combo offers only the corrections of the method, ε-N has no critical plane / Marin factors
        means = lambda: [fp.mean.itemData(i) for i in range(fp.mean.count())]  # noqa: E731
        cp_item = fp.eq.model().item(fp.eq.findData("critical_plane"))
        marin = (fp.surface, fp.rel, fp.size_d, fp.k_misc, fp.sn_own, fp.sn_db)
        check(means() == ["none", "morrow", "swt"], f"ε-N offers none / Morrow / SWT: {means()}")
        check(not cp_item.isEnabled() and not any(w_.isEnabled() for w_ in marin), "ε-N: no critical plane, no Marin factors / S-N curve")
        fp.mean.setCurrentIndex(fp.mean.findData("swt"))
        fp.method.setCurrentIndex(fp.method.findData("sn"))
        check(means() == ["none", "goodman", "gerber"] and fp.mean.currentData() == "goodman" and fs.mean_correction == "goodman",
              f"S-N after ε-N: Goodman / Gerber only, Morrow not kept ({means()}, {fp.mean.currentData()}, {fs.mean_correction})")
        check(cp_item.isEnabled() and all(w_.isEnabled() for w_ in marin) and not fp.en_box.isVisible(),
              "S-N: critical plane, Marin factors and S-N curve available again, ε-N section hidden")
        fp.mean.setCurrentIndex(fp.mean.findData("gerber"))
        check(fs.mean_correction == "gerber", "Gerber chosen for S-N")
        fp.method.setCurrentIndex(fp.method.findData("en"))
        check(fs.method == "en" and fs.mean_correction in ("none", "morrow", "swt") and fp.en_box.isVisible(), "back to ε-N")
        fp.refresh()                                  # reload from the study: the combo follows the stored method
        check(means() == ["none", "morrow", "swt"] and fp.mean.currentData() == fs.en_mean_correction, "reload keeps the ε-N choice")
        fp.mean.setCurrentIndex(fp.mean.findData("swt"))
        check(fs.en_mean_correction == "swt", "SWT mean correction")
        fp.en_box.notch.setCurrentIndex(fp.en_box.notch.findData("glinka"))
        check(fs.en_notch_rule == "glinka", "Glinka notch rule")
        mat = fp.sn_mat.currentText()
        fp.en_box.fill_estimate()
        p = fs.en_params.get(mat)
        check(p is not None and p["b"] < 0 and p["c"] < 0 and p["K_prime"] > 0, f"ε-N parameters of '{mat}': {p}")
        fp.ch_lc.setCurrentIndex(0)
        fp.ch_kind.setCurrentIndex(0)
        fp.s_amp.setText("1")
        fp.s_mean.setText("0")
        fp.add_channel()
        check(len(fs.history.channels) == 1, "history channel")
        fp.compute()
        state["fat"] = True

    def s_fatigue_done():
        fp = w.fat_panel
        fr = fp.result
        check(fr is not None and "ε-N" in " ".join(fr.method_lines()), f"ε-N result ({fr and fr.method_lines()[:1]})")
        if fr is not None:
            check(fp.worst.rowCount() > 0 and fp.legend.isVisible(), "map and worst-place table")
            shot("6_strain_life")

    steps = [s_open, s_bc, s_categories, s_conductance, s_solve, s_results, s_postproc_keeps_results,
             s_fatigue_settings, s_fatigue_done]

    def run_next():
        if w._thread is not None or w.fat_panel._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snapshots:", ", ".join(log))
            print("RESULT:", "OK" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
            app.quit()
            return
        f = steps.pop(0)
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            FAIL.append(f.__name__)
            app.quit()
            return
        QTimer.singleShot(500, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

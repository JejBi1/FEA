"""Public verification set in the GUI: the line in the About dialog (from docs/verification/results.csv) and the help page.

  python tests/gui/hidden_run.py tests/gui/gui_verification_smoke.py OUT_DIR

1) About: the line "N public verification cases, all within tolerance" is there, English and Czech, with the date of the run.
2) Help: the page "verification" is in the contents, opens in both languages and its link `fea:verification_page` reaches the action
   of the main window (here a recorder; the real action opens docs/verification/index.html in the browser).
3) The action `open_verification_page` of the tour opens the page of the program folder (the desktop services are replaced by a recorder).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtGui import QDesktopServices  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("[OK]   " if ok else "[FAIL] ") + what)
    if not ok:
        FAILS.append(what)


def labels(dlg) -> list[str]:
    """Texts as shown: the GUI translation keeps the English original in the getter and the shown text in this property."""
    out = []
    for lb in dlg.findChildren(QLabel):
        rec = lb.property("_i18n_text")
        out.append(rec[1] if rec else lb.text())
    return out


def main() -> int:
    app = QApplication(sys.argv)
    from fea_core import i18n, verification_summary
    from fea_gui import about, help as helpmod, tour
    from fea_gui import i18n as gui_i18n
    gui_i18n.install(app, "cs")                  # wraps the Qt widgets once; the language is then switched with i18n.set_language

    summ = verification_summary.read_summary(verification_summary.default_path(ROOT))
    check(summ is not None and summ.passed > 0, "the bundled results.csv is read")
    for lang in ("en", "cs"):
        i18n.set_language(lang)
        dlg = about.AboutDialog()
        dlg.show()                               # the GUI translation of constructor texts runs when the widget is polished
        app.processEvents()
        texts = labels(dlg)
        want = ("public verification cases" if lang == "en" else "veřejných ověřovacích úloh")
        line = [t for t in texts if want in t]
        check(bool(line), f"[{lang}] About shows the verification line: {line[:1]}")
        check(any((summ.date if summ else "?") in t for t in texts), f"[{lang}] About shows the date of the run")
        dlg.close()
    i18n.set_language("en")

    check("verification" in dict(helpmod.PAGES), "the help contents list the page 'verification'")
    for lang in ("en", "cs"):
        i18n.set_language(lang)
        html = helpmod.page_html("verification")
        check("fea:verification_page" in html and "<h1>" in html, f"[{lang}] the page text is there and carries the link")
        check(len(helpmod.search("MYSTRAN")) > 0 and "verification" in [s for s, _ in helpmod.search("MYSTRAN")],
              f"[{lang}] help search finds the page by 'MYSTRAN'")
    i18n.set_language("en")
    got: list[str] = []
    win = helpmod.HelpWindow(None, on_action=got.append)
    win.show_page("verification")
    win._link(QUrl("fea:verification_page"))
    check(got == ["verification_page"], "the link reaches the action handler")
    win.close()

    opened: list[str] = []
    QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toLocalFile()) or True)
    QMessageBox.information = staticmethod(lambda *a, **k: opened.append("message") or 0)
    tour.open_verification_page(None)
    check(len(opened) == 1 and opened[0].replace("\\", "/").endswith("docs/verification/index.html"),
          f"open_verification_page opens docs/verification/index.html ({opened})")
    del app
    print("RESULT:", "all checks passed" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

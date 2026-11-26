"""The main window shows the first-use notice of the numba kernels (fea_core.jitnotice) in its log and status line, also when the
message comes from a worker thread, and stops listening when it is closed.

python tests/gui/hidden_run.py tests/gui/gui_jit_smoke.py OUT_DIR
"""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
app = QApplication(sys.argv)
from fea_core import jitnotice  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402

fails = []


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


w = MainWindow()
w.show()
app.processEvents()
check(w._jit_sink in jitnotice._sinks, "the window registered a sink")

text = "First use of fatigue analysis: compiling the numerical kernels (once on this computer, then cached) – this may take a while…"
t = threading.Thread(target=jitnotice.say, args=(text,))        # the thread that compiles is not the GUI thread
t.start()
t.join()
app.processEvents()
log = w.log.toPlainText()
want = "překlad numerických jader" if os.environ.get("FEA_LANG") == "cs" else "compiling the numerical kernels"     # translated in the Czech GUI
check(want in log, f"the message is in the log ({want}): " + log[-130:].replace("\n", " | "))
check(bool(w.statusBar().currentMessage()), "the message is in the status line")

w._closing = True
w.confirm_discard = lambda: True
w.close()
app.processEvents()
check(not any(getattr(s, "__self__", None) is w for s in jitnotice._sinks), "the sink is removed when the window closes")
sys.exit(1 if fails else 0)

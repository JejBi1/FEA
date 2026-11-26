"""Text from files in the S1 / S2 GUI pieces is shown as plain text (review fix I4): the table tooltips of the setup confirmation and the
labels made of the verification table in the About dialog (Qt offscreen, labels only – no OCCT window)."""
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_cell_tooltip_is_escaped_rich_text():
    from fea_gui.setup_confirm import cell_tip
    tip = cell_tip("<img src='file:///x.png'><b>bold</b> & more")
    assert "<img" not in tip and "<b>" not in tip and "&lt;img" in tip and "&amp;" in tip
    assert tip.startswith("<p ") and tip.endswith("</p>")            # always rich text, so the escaped text is shown as it is


def test_about_verification_labels_are_plain_text(app, monkeypatch):
    from fea_core import verification_summary
    from fea_gui import about
    evil = "<a href='http://example.invalid'>click</a>"
    summ = SimpleNamespace(failed=0, not_run=0, passed=7, cases=7, version=evil, date=evil)
    monkeypatch.setattr(verification_summary, "read_summary", lambda path: summ)
    host = QWidget()
    lay = QVBoxLayout(host)
    about.AboutDialog._add_verification_line(lay)
    labels = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert len(labels) == 2 and all(isinstance(lb, QLabel) for lb in labels)
    assert all(lb.textFormat() == Qt.PlainText for lb in labels)
    assert evil in labels[1].text()

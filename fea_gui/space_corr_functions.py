"""Function setup table of the test correlation dashboard (F14 stage S5, `fea_gui.space_corr_panel`): for every measured function of the test data the user states what the
measurement is relative to (absolute / relative to the base), how it is normalised (none / per base acceleration / per force), and - when the file has none - its unit and the
probe / DOF of the model curve. The table is the only way the dashboard changes how a curve is compared; it never enters or confirms a limit.

All text that comes from a test file (function names, units) is shown as plain text.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QTableWidgetItem, QVBoxLayout, QWidget

from fea_core.space import corr_run as CR
from fea_core.space import correlation as C

from .widgets import table

COLUMNS = ["Function", "Kind", "Unit in the file", "Compared as", "Frame", "Normalisation", "Unit (override)", "Probe", "DOF", "Reference force [N]", "Needs"]
_FRAME_ITEMS = [("", "not chosen"), ("absolute", "absolute"), ("relative", "relative to the base")]
_PER_ITEMS = [("", "from the unit"), ("none", "none"), ("base_acceleration", "per base acceleration"), ("force", "per force")]
MAX_ROWS = 500
C_FRAME, C_PER, C_UNIT, C_PROBE, C_DOF, C_FORCE = 4, 5, 6, 7, 8, 9


def _item(text: Any, editable: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem("" if text is None else str(text))
    flags = it.flags()
    it.setFlags(flags | Qt.ItemFlag.ItemIsEditable if editable else flags & ~Qt.ItemFlag.ItemIsEditable)
    return it


def _combo(items: list[tuple[str, str]], value: str) -> QComboBox:
    c = QComboBox()
    for data, label in items:
        c.addItem(label, data)
    c.setCurrentIndex(max(0, c.findData(value)))
    return c


class FunctionSetupTable(QWidget):
    """The table of the measured functions with the setup fields; `fill(test, setups)`, `values()` -> {function name: setup dict}, `describe()` refreshes the derived columns."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.test: C.TestData | None = None
        self.names: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = table(COLUMNS, placeholder="Import test data with measured functions (UFF 58 / CSV)")
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed | QAbstractItemView.EditTrigger.AnyKeyPressed)
        lay.addWidget(self.table, 1)

    def fill(self, test: C.TestData | None, setups: dict[str, C.FunctionSetup]) -> None:
        t = self.table
        t.setRowCount(0)
        self.test = test
        self.names = []
        if test is None:
            return
        desc = {d["name"]: d for d in CR.describe_functions(test, setups)}
        for fn in test.functions[:MAX_ROWS]:
            d = desc.get(fn.name, {})
            su = setups.get(fn.name) or C.FunctionSetup()
            r = t.rowCount()
            t.insertRow(r)
            self.names.append(fn.name)
            compared_as = "information only" if fn.kind not in ("frf", "psd") else (f"{d.get('quantity') or '?'}" + (f" per {d['per'].replace('_', ' ')}" if d.get("per") not in (None, "", "none") else "")
                                                                                   + (" PSD" if d.get("psd") else ""))
            t.setItem(r, 0, _item(fn.name))
            t.setItem(r, 1, _item(fn.kind))
            t.setItem(r, 2, _item(fn.unit))
            t.setItem(r, 3, _item(compared_as))
            t.setCellWidget(r, C_FRAME, _combo(_FRAME_ITEMS, su.frame))
            t.setCellWidget(r, C_PER, _combo(_PER_ITEMS, su.per))
            t.setItem(r, C_UNIT, _item(su.unit, True))
            t.setItem(r, C_PROBE, _item(su.probe, True))
            t.setItem(r, C_DOF, _item(su.dof, True))
            t.setItem(r, C_FORCE, _item("" if su.reference_force_n is None else f"{su.reference_force_n:g}", True))
            needs = "; ".join(d.get("needs") or []) or (d.get("note") or "")
            it = _item(needs)
            it.setToolTip(needs)
            t.setItem(r, 10, it)
            if fn.kind not in ("frf", "psd"):
                for c in (C_FRAME, C_PER):
                    w = t.cellWidget(r, c)
                    if w is not None:
                        w.setEnabled(False)

    def values(self) -> dict[str, dict[str, Any]]:
        """{function name: setup fields} of the rows (empty fields = taken from the file); raises ValueError for a reference force that is not a number."""
        out: dict[str, dict[str, Any]] = {}
        t = self.table
        for r, name in enumerate(self.names):
            def text(c: int, r: int = r) -> str:
                it = t.item(r, c)
                return it.text().strip() if it is not None else ""
            frame = t.cellWidget(r, C_FRAME)
            per = t.cellWidget(r, C_PER)
            force = text(C_FORCE).replace(",", ".")
            try:
                rf = float(force) if force else None
            except ValueError:
                raise ValueError(f"function {C.clean_label(name, 40)}: the reference force must be a number") from None
            out[name] = {"frame": str(frame.currentData()) if isinstance(frame, QComboBox) else "", "per": str(per.currentData()) if isinstance(per, QComboBox) else "",
                         "unit": text(C_UNIT), "probe": text(C_PROBE), "dof": text(C_DOF), "reference_force_n": rf}
        return out

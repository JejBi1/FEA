"""EN 1990 category of a load case (LoadCase.category) in the GUI: one combo box shared by the combinations dialog and
the load-case table of the Tables dock. The categories come from fea_core.combinations (data/load_factors.json)."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from fea_core.i18n import tr

# short English labels of the keys of combinations.load_categories() (the data descriptions are Czech national-annex text)
LABELS = {
    "permanent": "Permanent G",
    "cat_A": "A – domestic, residential",
    "cat_B": "B – office areas",
    "cat_C": "C – congregation areas",
    "cat_D": "D – shopping areas",
    "cat_E": "E – storage areas",
    "cat_F": "F – traffic, vehicle up to 30 kN",
    "cat_G": "G – traffic, 30 to 160 kN",
    "cat_H": "H – roofs",
    "snow_cz_low": "Snow, altitude up to 1000 m",
    "snow_cz_high": "Snow, altitude above 1000 m",
    "wind": "Wind",
    "temperature": "Temperature (non-fire)",
}
AUTO = ""                                  # item data of "guess from the name"


def categories() -> dict[str, dict]:
    from fea_core.combinations import load_categories
    return load_categories()


def label(key: str | None) -> str:
    if not key:
        return "–"
    return LABELS.get(key, key)


def psi_text(info: dict) -> str:
    if "psi_0" not in info:
        return "γ_G"
    from fea_core.i18n import decimal
    return " · ".join(f"ψ{k} {decimal(format(info[f'psi_{k}'], 'g'))}" for k in (0, 1, 2))


def guessed(name: str) -> str:
    from fea_core.combinations import guess_category
    return guess_category(name)


def category_combo(lc, on_change) -> QComboBox:
    """Combo box of the category of load case `lc`; on_change(lc, key | None) after a user change."""
    cb = QComboBox()
    cats = categories()
    cb.addItem(tr("Guess from name") + f" ({tr(label(guessed(lc.name)))})", AUTO)
    cb.setItemData(0, tr("Not set – the combination generator guesses it from the name and warns"), 3)
    for k, info in cats.items():
        cb.addItem(tr(label(k)), k)
        cb.setItemData(cb.count() - 1, f"{tr(label(k))}: {psi_text(info)}", 3)   # Qt.ToolTipRole
    cb.setProperty("_i18n_done", True)
    cur = lc.category or AUTO
    i = cb.findData(cur)
    if i < 0:                              # stored value outside the table (older file): shown, not replaced
        cb.addItem(str(cur), cur)
        i = cb.count() - 1
    cb.setCurrentIndex(i)
    cb.setToolTip(tr("EN 1990 category of the load case (combination factors γ and ψ)"))

    def changed(_i):
        key = cb.currentData() or None
        if key != (lc.category or None):
            on_change(lc, key)
    cb.currentIndexChanged.connect(changed)
    return cb


def set_category(win, lc, key: str | None) -> None:
    """Writes the category (normalised) and marks the study changed – combinations only, the results stay."""
    from fea_core.combinations import normalize_category
    lc.category = normalize_category(key) if key else None
    win.mark_changed("none", refresh=False)

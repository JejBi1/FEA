"""Strain-life (ε-N) settings of the Fatigue step (FatigueSettings.method = "en", fea_core.fatigue.strain_life):
notch rule (Neuber / Glinka), mean stress correction none / Morrow / SWT (the mean-stress combo of the panel) and the
parameters of the cyclic Ramberg–Osgood and Coffin–Manson–Basquin curves per material (FatigueSettings.en_params; empty =
Uniform Material Law estimate from Rm, fea_core.fatigue.analysis.en_for_material)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from fea_core.fatigue.settings import NOTCH_RULE_LABELS
from fea_core.i18n import tr

from .widgets import Section, dspin, help_tip

EN_MEANS = ("none", "morrow", "swt")
# key, label, tip, (default, lo, hi, decimals, step)
PARAMS = (("E", "E [MPa]", "Modulus of elasticity of the cyclic curve", (210000.0, 1.0, 1e7, 0, 1000.0)),
          ("sigma_f_prime", "σ'f [MPa]", "Fatigue strength coefficient σ'f", (900.0, 1.0, 1e5, 1, 10.0)),
          ("b", "b", "Fatigue strength exponent b (negative, −0.05 to −0.15)", (-0.087, -2.0, -1e-4, 4, 0.005)),
          ("eps_f_prime", "ε'f", "Fatigue ductility coefficient ε'f (0.1 to 1)", (0.59, 1e-4, 10.0, 4, 0.01)),
          ("c", "c", "Fatigue ductility exponent c (negative, −0.5 to −0.7)", (-0.58, -2.0, -1e-4, 4, 0.01)),
          ("K_prime", "K' [MPa]", "Cyclic strength coefficient K'", (900.0, 1.0, 1e6, 1, 10.0)),
          ("n_prime", "n'", "Cyclic strain-hardening exponent n' (0.1 to 0.25)", (0.15, 1e-4, 2.0, 4, 0.01)))
HELP = ("<b>Strain-life (ε-N)</b><p>For low-cycle fatigue (local plasticity at notches). The elastic FEM stress history at a "
        "node is converted to the local elastic-plastic strain by the <i>notch rule</i> (Neuber σ·ε = σ_el²/E, or Glinka "
        "equal strain energy) on the cyclic Ramberg–Osgood curve ε = σ/E + (σ/K')^(1/n'); rainflow counts the strain "
        "cycles and the Coffin–Manson–Basquin curve ε_a = σ'f/E·(2N)^b + ε'f·(2N)^c gives the life, with the mean stress "
        "correction Morrow (σ'f − σm) or Smith–Watson–Topper (σ_max·ε_a). Miner sums the damage.</p><p>Without own "
        "parameters they are estimated from Rm by the Uniform Material Law (Bäumel–Seeger) – INDICATIVE; enter measured "
        "values (e.g. from PragTic, Manson–Coffin) when available. Not available for 3D-printed materials and with the "
        "critical plane.</p>")


class StrainLifeBox(Section):
    """ε-N section of the fatigue panel; the panel calls load(fs, material) and reads values() / own()."""

    def __init__(self, panel):
        self.panel = panel
        inner = QWidget()
        super().__init__("Strain-life (ε-N) parameters", inner, key="fatigue.strain_life",
                         tip="Notch rule and ε-N parameters of the material", help=HELP)
        f = QFormLayout(inner)
        self.notch = QComboBox()
        for k, lbl in NOTCH_RULE_LABELS.items():
            self.notch.addItem(lbl, k)
        help_tip(self.notch, "Notch rule: elastic FEM stress → local elastic-plastic strain",
                 "Neuber (σ·ε = σ_el²/E) – conservative; Glinka (equal strain energy density) – less conservative.")
        f.addRow("Notch rule", self.notch)
        self.own = QCheckBox("own parameters (otherwise estimate from Rm)")
        help_tip(self.own, "Own ε-N parameters of the material instead of the UML estimate",
                 "On: the values below are used for the material selected at “S-N for material”. Off: Uniform Material "
                 "Law estimate from Rm and E (Bäumel–Seeger 1990), shown below for information.")
        f.addRow(self.own)
        self.spins = {}
        for key, lbl, tip, (val, lo, hi, dec, step) in PARAMS:
            w = dspin(val, lo, hi, dec, step)
            w.setToolTip(tip)
            self.spins[key] = w
            f.addRow(lbl, w)
        h = QHBoxLayout()
        self.btn_est = QPushButton("Fill estimate from Rm")
        self.btn_est.setToolTip("Uniform Material Law estimate (Bäumel–Seeger) from Rm and E of the material")
        self.btn_est.clicked.connect(self.fill_estimate)
        h.addWidget(self.btn_est)
        h.addStretch(1)
        f.addRow(h)
        self.info = QLabel()
        self.info.setTextFormat(Qt.PlainText)      # the source text comes from the study file – never rich text
        self.info.setWordWrap(True)
        self.info.setProperty("muted", True)
        self.info.setToolTip("Source of the ε-N parameters used for the material")
        f.addRow(self.info)
        self.notch.currentIndexChanged.connect(self._changed)
        self.own.toggled.connect(self._own_toggled)
        for w in self.spins.values():
            w.editingFinished.connect(self._changed)
        self._loading = False
        self._material = None

    # ------------------------------------------------------------------ data
    def _mat(self):
        st = self.panel.win.study
        name = self._material
        return st.materials.get(name) if st is not None and name else None

    def estimate(self):
        from fea_core.fatigue.analysis import en_for_material
        from fea_core.fatigue.settings import FatigueSettings
        m = self._mat()
        if m is None:
            return None
        return en_for_material(m, FatigueSettings())           # no given params → UML estimate

    def load(self, fs, material: str | None):
        """Form from FatigueSettings for the material (None = no material)."""
        self._loading = True
        try:
            self._material = material
            self.notch.setCurrentIndex(max(0, self.notch.findData(fs.en_notch_rule)))
            given = fs.en_params.get(material) if material else None
            self.own.setChecked(bool(given))
            vals = given
            src = tr("own parameters") + (f" ({given.get('source')})" if given and given.get("source") else "")
            if not given:
                try:
                    p = self.estimate()
                    vals = p.to_dict() if p is not None else None
                    src = (tr("estimate") + f": {p.source}") if p is not None else ""
                except ValueError as e:
                    vals, src = None, "! " + str(e)
            for k, w in self.spins.items():
                if vals and vals.get(k) is not None:
                    w.setValue(float(vals[k]))
                w.setEnabled(bool(given))
            self.info.setText(src)
        finally:
            self._loading = False

    def values(self) -> dict:
        d = {k: float(w.value()) for k, w in self.spins.items()}
        d["source"] = "entered in the GUI"
        return d

    # ------------------------------------------------------------------ edits
    def _own_toggled(self, on):
        for w in self.spins.values():
            w.setEnabled(on)
        self._changed()

    def fill_estimate(self):
        try:
            p = self.estimate()
        except ValueError as e:
            self.info.setText("! " + str(e))
            return
        if p is None:
            return
        for k, w in self.spins.items():
            w.setValue(float(getattr(p, k)))
        self.own.setChecked(True)
        self._changed()

    def _changed(self, *_):
        if self._loading or self.panel._filling or self.panel.win.study is None:
            return
        fs = self.panel.settings()
        new_rule = self.notch.currentData()
        name = self._material
        new_params = dict(fs.en_params)
        if name:
            if self.own.isChecked():
                new_params[name] = self.values()
            else:
                new_params.pop(name, None)
        if new_rule == fs.en_notch_rule and new_params == fs.en_params:
            return
        fs.en_notch_rule = new_rule
        fs.en_params = new_params
        self.panel.changed()

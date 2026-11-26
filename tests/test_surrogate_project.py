"""The surrogate model travels with the project file .fea (surrogate.npz, no pickle) and the report section lists it (F15-S4)."""
from __future__ import annotations

import zipfile
import zlib

import numpy as np
import pytest

from fea_core.project import PROJECT_FILES, SURROGATE_FILE, load_project, load_surrogate_file, save_project
from fea_core.study import Study
from fea_core.surrogate import fit_surrogate, lhs_design


def _model(backend="rbf"):
    X = lhs_design([(10.0, 60.0), (0.0, 40.0)], 40, seed=0)
    pts = [{"params": {"a": float(a), "b": float(b)}, "kpi": {"max_stress": float(50 + a + 0.5 * b)}} for a, b in X]
    return fit_surrogate(pts, backend=backend, seed=2)


def test_project_roundtrip_keeps_the_surrogate(tmp_path):
    sm = _model()
    fea = tmp_path / "p.fea"
    save_project(fea, Study(name="s", geometry=""), "", None, surrogate=sm)
    with zipfile.ZipFile(fea) as z:
        assert SURROGATE_FILE in z.namelist() and SURROGATE_FILE in PROJECT_FILES
    wd = tmp_path / "wd"
    load_project(fea, wd)
    back = load_surrogate_file(wd)
    q = {"a": 30.0, "b": 12.0}
    assert back is not None and back.predict(q)["kpis"]["max_stress"]["value"] == sm.predict(q)["kpis"]["max_stress"]["value"]


def test_project_without_surrogate_and_with_a_damaged_one(tmp_path):
    fea = tmp_path / "p.fea"
    save_project(fea, Study(name="s", geometry=""), "", None)
    wd = tmp_path / "wd"
    load_project(fea, wd)
    assert load_surrogate_file(wd) is None
    (wd / SURROGATE_FILE).write_bytes(b"not an npz")
    assert load_surrogate_file(wd) is None                       # unreadable: the project still opens


def test_restore_without_scikit_learn_is_exact(tmp_path, monkeypatch):
    """A model saved with the Gaussian process opens where scikit-learn is missing: the prediction is plain numpy (scikit-learn only
    optimises the hyperparameters), so the restored model answers exactly like the fitted one."""
    import pytest
    from fea_core.surrogate import model as M
    if not M.backends_available()["gp"]:
        pytest.skip("scikit-learn is not installed")
    sm = _model("gp")
    f = tmp_path / "s.npz"
    sm.save(f)
    monkeypatch.setattr(M, "backends_available", lambda: {"gp": False, "rbf": True})
    from fea_core.surrogate import load_surrogate
    back = load_surrogate(f)
    assert back.backend == "gp"
    assert back.members[()]["max_stress"].cal_ids == sm.members[()]["max_stress"].cal_ids          # the same hold-out set
    r, r0 = back.predict({"a": 30.0, "b": 12.0})["kpis"]["max_stress"], sm.predict({"a": 30.0, "b": 12.0})["kpis"]["max_stress"]
    assert r == r0 and abs(r["value"] - (50 + 30 + 6)) < 1.0


def test_rbf_file_loads_without_scikit_learn(tmp_path):
    sm = _model("rbf")
    f = tmp_path / "s.npz"
    sm.save(f)
    from fea_core.surrogate import load_surrogate
    assert load_surrogate(f).backend == "rbf"
    assert np.isfinite(load_surrogate(f).predict({"a": 20.0, "b": 5.0})["kpis"]["max_stress"]["value"])


class _FakeDoc:
    def __init__(self):
        self.tables, self.paras = [], []

    def p(self, text="", **kw):
        self.paras.append(text)

    def table(self, header, rows, **kw):
        self.tables.append((header, rows, kw.get("caption")))


def test_report_section_lists_parameters_validation_and_limitations():
    from types import SimpleNamespace
    from fea_core.report.surrogate import has_surrogate, surrogate_section
    from fea_core.surrogate import LIMITATIONS
    sm = _model()
    sess = SimpleNamespace(surrogate=sm)
    assert has_surrogate(sess) and not has_surrogate(SimpleNamespace(surrogate=None)) and not has_surrogate(SimpleNamespace())
    D = _FakeDoc()
    surrogate_section(D, sess)
    assert [t[2].split(" – ")[1].split(",")[0] for t in D.tables] == ["parameters", "validation on the held-out points", "held-out points"]
    assert D.tables[0][1][0][1] == "continuous" and len(D.tables[1][1]) == 1
    assert LIMITATIONS in D.paras and any("split-conformal" in t for t in D.paras)
    held = D.tables[2]
    assert len(held[1]) == min(20, len(sm.validation_table())) and all(len(r) == len(held[0]) for r in held[1])


def test_a_saved_gaussian_process_loads_and_predicts_without_importing_scikit_learn(tmp_path):
    """RAM: scikit-learn only OPTIMISES the hyperparameters (it adds ~25 MB to the process once, at the first fit). A model that is restored
    from a file or only asked for predictions keeps plain numpy arrays and never imports it."""
    import subprocess
    import sys
    from pathlib import Path

    import pytest
    from fea_core.surrogate.model import backends_available
    if not backends_available()["gp"]:
        pytest.skip("scikit-learn is not installed")
    f = tmp_path / "gp.npz"
    _model("gp").save(f)
    code = ("import sys\nfrom fea_core.surrogate import load_surrogate\nm = load_surrogate(sys.argv[1])\n"
            "r = m.predict({'a': 30.0, 'b': 12.0})['kpis']['max_stress']['value']\nassert 'sklearn' not in sys.modules, 'scikit-learn was imported'\nprint(r)")
    out = subprocess.run([sys.executable, "-c", code, str(f)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert abs(float(out.stdout.strip()) - _model("gp").predict({"a": 30.0, "b": 12.0})["kpis"]["max_stress"]["value"]) < 1e-9


# ------------------------------------------------------------------ N3: damage INSIDE the surrogate member of the .fea
def _flip(raw: bytes, fea, offset: int, n: int) -> bytes:
    b = bytearray(raw)
    b[offset:offset + n] = bytes((x ^ 0xFF) for x in b[offset:offset + n])
    return bytes(b)


def _member_span(fea):
    with zipfile.ZipFile(fea) as z:
        info = z.getinfo(SURROGATE_FILE)
        start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        return start, info.compress_size


def _restored(tmp_path, stored: bool):
    """A real .fea with the surrogate member (deflated, or stored so that only the CRC can notice a damaged byte)."""
    fea = tmp_path / "p.fea"
    save_project(fea, Study(name="kept", geometry=""), "", None, surrogate=_model())
    if stored:
        out = tmp_path / "stored.fea"
        with zipfile.ZipFile(fea) as zin, zipfile.ZipFile(out, "w") as zout:
            for info in zin.infolist():
                zout.writestr(info.filename, zin.read(info), compress_type=zipfile.ZIP_STORED if info.filename == SURROGATE_FILE else zipfile.ZIP_DEFLATED)
        return out
    return fea


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("where", ["middle", "start", "end"])
def test_flipped_bytes_in_the_surrogate_member_drop_only_the_surrogate(tmp_path, stored, where):
    fea = _restored(tmp_path, stored)
    start, size = _member_span(fea)
    off = {"middle": start + size // 2, "start": start + 24, "end": start + size - 4}[where]
    bad = tmp_path / "bad.fea"
    bad.write_bytes(_flip(fea.read_bytes(), fea, off, 4))
    with zipfile.ZipFile(bad) as z:                                      # the damage is real: the zip itself says the member is bad
        try:
            assert z.testzip() == SURROGATE_FILE
        except (zlib.error, zipfile.BadZipFile):
            pass
    wd = tmp_path / "wd"
    study, step, results = load_project(bad, wd)                          # opens: the study and the other members are intact
    assert study.name == "kept" and results is None
    warns: list[str] = []
    assert load_surrogate_file(wd, warns) is None
    assert warns and "surrogate" in warns[0] and "dropped" in warns[0], warns


def test_a_surrogate_member_that_is_too_large_or_lies_about_its_size_never_stops_the_project(tmp_path):
    from fea_core import project
    fea = tmp_path / "big.fea"
    with zipfile.ZipFile(fea, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("study.json", Study(name="kept", geometry="").to_json() if hasattr(Study, "to_json") else __import__("json").dumps(Study(name="kept", geometry="").to_dict()))
        z.writestr(SURROGATE_FILE, b"\0" * (project.SURROGATE_MAX_BYTES + 10))          # a zip bomb in the surrogate member only
    study, _, _ = load_project(fea, tmp_path / "wd")
    assert study.name == "kept"
    warns: list[str] = []
    assert load_surrogate_file(tmp_path / "wd", warns) is None and warns


def test_a_study_member_that_is_damaged_still_stops_the_project(tmp_path):
    fea = _restored(tmp_path, stored=True)
    with zipfile.ZipFile(fea) as z:
        info = z.getinfo("study.json")
        off = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + 3
    bad = tmp_path / "bad.fea"
    bad.write_bytes(_flip(fea.read_bytes(), fea, off, 2))
    with pytest.raises(Exception):                                         # noqa: B017 – BadZipFile / a study that does not parse: the project is not opened
        load_project(bad, tmp_path / "wd")


def test_a_surrogate_file_above_the_size_cap_is_refused(tmp_path):
    from fea_core.surrogate import SurrogateError, load_surrogate
    from fea_core.surrogate.store import MAX_FILE_BYTES
    f = tmp_path / "big.npz"
    with open(f, "wb") as fh:
        fh.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(SurrogateError, match="too large"):
        load_surrogate(f)
    assert MAX_FILE_BYTES <= 16 << 20

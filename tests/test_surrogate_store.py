"""The surrogate.npz of a foreign .fea is untrusted input (F15-S4 review fixes I4, I5): bounded reading, known fields with type and range checks,
the band recomputed from the stored split (a file cannot forge `q`, `reliable` or a narrow band), a damaged file never stops a project
from opening. The crafted files are those of the independent review (`.scratch/review/edge.py`, `corrupt.py`)."""
from __future__ import annotations

import io
import json
import time
import tracemalloc
import zipfile
import zlib

import numpy as np
import pytest

from fea_core.project import SURROGATE_FILE, load_project, load_surrogate_file, save_project
from fea_core.study import Study
from fea_core.surrogate import SurrogateError, fit_surrogate, lhs_design, load_surrogate
from fea_core.surrogate import store as ST
from fea_core.surrogate.fit import SURROGATE_KPIS


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    d = tmp_path_factory.mktemp("good")
    X = lhs_design([(0, 1), (0, 1)], 40, seed=1)
    y = 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2
    sm = fit_surrogate([{"params": {"a": float(x[0]), "b": float(x[1])}, "max_stress": float(v), "converged": True} for x, v in zip(X, y)], backend="rbf", seed=3)
    f = d / "good.npz"
    sm.save(f)
    with np.load(f, allow_pickle=False) as z:
        meta, P, K = json.loads(str(z["meta"])), z["P"], z["K"]
    return {"model": sm, "file": f, "meta": meta, "P": P, "K": K, "dir": d}


def craft(good, name, edit=None, P=None, K=None, extra=None, meta=None):
    m = json.loads(json.dumps(good["meta"])) if meta is None else meta
    if edit:
        edit(m)
    p = good["dir"] / f"{name}.npz"
    arrays = {"meta": np.array(m if isinstance(m, str) else json.dumps(m)), "P": good["P"] if P is None else P, "K": good["K"] if K is None else K}
    arrays.update(extra or {})
    with open(p, "wb") as f:
        np.savez(f, **arrays)
    return p


def refused(good, name, **kw):
    p = craft(good, name, **kw)
    with pytest.raises(SurrogateError):
        load_surrogate(p)


def M0(m):                           # the one member of the file
    return m["members"][0]


def test_roundtrip_is_exact_and_carries_fingerprint_sweep_failed_and_adaptive(tmp_path):
    X = lhs_design([(0, 1), (0, 1)], 40, seed=1)
    y = 100 + 50 * X[:, 0] + 20 * X[:, 1] ** 2
    ok = np.arange(40) != 7
    points = [{"params": {"a": float(x[0]), "b": float(x[1])}, "max_stress": float(v), "converged": bool(ok[i])} for i, (x, v) in enumerate(zip(X, y))]
    points[3]["adaptive"] = True
    fp = "ab" * 32
    sm = fit_surrogate(points, backend="rbf", seed=3, fingerprint=fp, sweep={"mode": "lhs", "seed": 3, "params": [{"kind": "mesh_size", "bounds": [0, 1]}]})
    f = tmp_path / "s.npz"
    sm.save(f)
    back = load_surrogate(f)
    q = {"a": 0.4, "b": 0.6}
    assert back.predict(q)["kpis"] == sm.predict(q)["kpis"] and back.describe()["members"] == sm.describe()["members"]
    assert back.fingerprint == fp and back.sweep == sm.sweep and len(back.failed) == 1 and back.rows[3].get("adaptive")
    assert 3 in back.members[()]["max_stress"].train_ids and 3 not in back.members[()]["max_stress"].cal_ids
    assert any("failed" in w for w in back.warnings)


def test_file_cannot_forge_the_calibration_or_the_verdict(good):
    """The stored calibration, q, rmse (σ), transform and standardisation are never trusted: the model recomputes them from the split."""
    ref = good["model"].predict({"a": 0.4, "b": 0.6})["kpis"]["max_stress"]

    def forge(m):
        M0(m)["calibration"] = {"q": 0.0, "reliable": True, "reasons": [], "coverage_cv": 1.0, "to_dict": "x", "__class__": "y"}
        M0(m).update(q=0.0, reliable=True, transform="log", ymean=1e300, ystd=1e-300)
        M0(m)["hyper"]["rmse"] = 1e-6                                                # σ is only a scale: the recomputed q compensates it
    p = craft(good, "forged", edit=forge)
    r = load_surrogate(p).predict({"a": 0.4, "b": 0.6})["kpis"]["max_stress"]
    assert r["reliable"] == ref["reliable"]
    for k in ("value", "lower", "upper"):
        assert r[k] == pytest.approx(ref[k], rel=1e-6)
    assert load_surrogate(p).members[()]["max_stress"].cal.q > 1.0                   # not the forged 0


@pytest.mark.parametrize("name, edit", [
    ("format", lambda m: m.update(format=99)),
    ("level_hi", lambda m: m.update(level=1.5)),
    ("level_str", lambda m: m.update(level="x")),
    ("level_nan", lambda m: m.update(level=float("nan"))),
    ("seed_float", lambda m: m.update(seed=1e30)),
    ("seed_neg", lambda m: m.update(seed=-1)),
    ("fraction_str", lambda m: m.update(validation_fraction="x")),
    ("fraction_big", lambda m: m.update(validation_fraction=0.9)),
    ("backend", lambda m: m.update(backend="evil")),
    ("backend_auto", lambda m: m.update(backend="auto")),
    ("fingerprint", lambda m: m.update(fingerprint="not hex")),
    ("kpis_empty", lambda m: m.update(kpis=[])),
    ("kpis_unknown", lambda m: m.update(kpis=["evil"])),
    ("kpis_dup", lambda m: m.update(kpis=["max_stress", "max_stress"])),
    ("kpis_none", lambda m: m.update(kpis=None)),
    ("members_empty", lambda m: m.update(members=[])),
    ("members_300", lambda m: m.update(members=m["members"] * 300)),
    ("members_dup", lambda m: m.update(members=m["members"] * 2)),
    ("member_str", lambda m: m.update(members=["x"])),
    ("params_missing", lambda m: m.pop("params")),
    ("params_dup_key", lambda m: m["params"][1].update(key="a")),
    ("param_kind", lambda m: m["params"][0].update(kind="evil")),
    ("param_key_null", lambda m: m["params"][0].update(key=None)),
    ("param_key_long", lambda m: m["params"][0].update(key="k" * 5000)),
    ("categorical_no_categories", lambda m: m["params"][0].update(kind="categorical", categories=[])),
    ("kpi_unknown", lambda m: M0(m).update(kpi="evil")),
    ("kpi_null", lambda m: M0(m).update(kpi=None)),
    ("group_wrong", lambda m: M0(m).update(group=["steel"])),
    ("train_range", lambda m: M0(m).update(train=[0, 1, 99999])),
    ("train_negative", lambda m: M0(m).update(train=[-1] + M0(m)["train"][1:])),
    ("train_dup", lambda m: M0(m).update(train=M0(m)["train"] + M0(m)["train"][:3])),
    ("train_repeated_1e5", lambda m: M0(m).update(train=[0, 1, 2, 3] * 25000)),
    ("train_strings", lambda m: M0(m).update(train=["a", "b"])),
    ("train_floats", lambda m: M0(m).update(train=[0.5, 1.5])),
    ("train_empty", lambda m: M0(m).update(train=[])),
    ("train_short", lambda m: M0(m).update(train=M0(m)["train"][:4])),
    ("cal_empty", lambda m: M0(m).update(cal=[])),
    ("cal_in_train", lambda m: M0(m).update(train=M0(m)["train"] + M0(m)["cal"])),
    ("cal_is_train", lambda m: M0(m).update(cal=M0(m)["train"][:12])),
    ("hyper_missing", lambda m: M0(m).update(hyper={})),
    ("hyper_str", lambda m: M0(m).update(hyper="x")),
    ("hyper_list", lambda m: M0(m).update(hyper=[1, 2])),
    ("hyper_rmse_neg", lambda m: M0(m)["hyper"].update(rmse=-1.0)),
    ("hyper_rmse_zero", lambda m: M0(m)["hyper"].update(rmse=0.0)),
    ("hyper_rmse_huge", lambda m: M0(m)["hyper"].update(rmse=1e30)),
    ("hyper_rmse_nan", lambda m: M0(m)["hyper"].update(rmse=float("nan"))),
    ("hyper_rmse_str", lambda m: M0(m)["hyper"].update(rmse="x")),
    ("hyper_smoothing_neg", lambda m: M0(m)["hyper"].update(smoothing=-1e30)),
    ("hyper_smoothing_nan", lambda m: M0(m)["hyper"].update(smoothing=float("nan"))),
    ("hyper_smoothing_big", lambda m: M0(m)["hyper"].update(smoothing=100.0)),
    ("hyper_bool", lambda m: M0(m)["hyper"].update(smoothing=True)),
    ("warnings_not_list", lambda m: m.update(warnings="x")),
    ("sweep_list", lambda m: m.update(sweep=[1])),
    ("failed_str", lambda m: m.update(failed="x")),
    ("failed_item", lambda m: m.update(failed=[1])),
    ("failed_huge", lambda m: m.update(failed=[{"a": 1.0}] * 5000)),
    ("adaptive_range", lambda m: m.update(adaptive=[99999])),
    ("adaptive_in_cal", lambda m: m.update(adaptive=M0(m)["cal"][:1])),
    ("skipped_item", lambda m: m.update(skipped=[1])),
])
def test_crafted_header_is_refused(good, name, edit):
    refused(good, name, edit=edit)


def test_gaussian_process_hyperparameters_are_range_checked_but_need_no_scikit_learn(good):
    ok = {"amplitude": 1.0, "length_scales": [0.5, 0.5], "noise": 1e-4}
    p = craft(good, "gp_ok", edit=lambda m: (m.update(backend="gp"), M0(m).update(hyper=ok)))
    sm = load_surrogate(p)                                 # the prediction is numpy: restoring a Gaussian process needs no scikit-learn
    assert sm.backend == "gp" and np.isfinite(sm.predict({"a": 0.5, "b": 0.5})["kpis"]["max_stress"]["value"])
    for bad in ({**ok, "amplitude": 1e9}, {**ok, "amplitude": -1.0}, {**ok, "noise": 5.0}, {**ok, "noise": 0.0}, {**ok, "length_scales": [0.5]},
                {**ok, "length_scales": [0.5, 1e6]}, {**ok, "length_scales": [0.5, float("nan")]}, {**ok, "length_scales": "x"}, {"amplitude": 1.0}):
        refused(good, "gp_bad", edit=lambda m, h=bad: (m.update(backend="gp"), M0(m).update(hyper=h)))


@pytest.mark.parametrize("name, P_edit", [
    ("nan", lambda P: np.full_like(P, np.nan)),
    ("inf", lambda P: np.full_like(P, np.inf)),
    ("equal", lambda P: np.zeros_like(P)),
    ("complex", lambda P: P.astype(complex)),
    ("string", lambda P: P.astype("U8")),
    ("float32", lambda P: P.astype(np.float32)),
    ("int", lambda P: (P * 100).astype(np.int64)),
    ("1d", lambda P: P.ravel()),
    ("3d", lambda P: P[:, :, None]),
    ("zero_rows", lambda P: P[:0]),
    ("too_wide", lambda P: np.zeros((P.shape[0], 9))),
    ("fortran", lambda P: np.asfortranarray(P)),
])
def test_crafted_parameter_array_is_refused(good, name, P_edit):
    P = P_edit(good["P"])
    refused(good, "P_" + name, P=P, K=good["K"][:0] if name == "zero_rows" else None)


def test_crafted_result_arrays_are_refused(good):
    K = good["K"]
    refused(good, "K_inf", K=np.where(np.arange(K.size).reshape(K.shape) == 3, np.inf, K))
    refused(good, "K_shape", K=K[:, :3])
    refused(good, "K_rows", K=K[:10])
    refused(good, "K_object", K=np.array([{"a": 1}] * K.size, dtype=object).reshape(K.shape))          # a pickle member
    refused(good, "K_all_nan", K=np.full_like(K, np.nan))                                              # no value of any KPI


def test_negative_results_load_with_the_identity_transform(good):
    p = craft(good, "K_neg", K=-np.abs(good["K"]))
    assert load_surrogate(p).members[()]["max_stress"].transform == "id"


def test_meta_that_is_not_a_bounded_json_object_is_refused(good):
    refused(good, "meta_text", meta="{{{")
    refused(good, "meta_list", meta="[1]")
    refused(good, "meta_nan", meta=json.dumps(good["meta"]).replace('"level": 0.9', '"level": NaN'))
    refused(good, "meta_deep", meta="[" * 200000 + "]" * 200000)                                      # no RecursionError, a SurrogateError
    refused(good, "meta_deep_8", meta='{"a":' * 12 + "1" + "}" * 12)
    refused(good, "meta_huge", meta=json.dumps({"x": "y" * (ST.MAX_META_CHARS + 10)}))
    p = craft(good, "meta_deep_sweep", edit=lambda m: m.update(sweep={"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": 1}}}}}}}}}}))
    with pytest.raises(SurrogateError):
        load_surrogate(p)


def test_the_work_is_bounded(good):
    """2000 rows and seven models of 2000 points each would take the cubic cost far beyond the budget: refused before anything is fitted."""
    n = 2000
    r = np.random.default_rng(0)
    P = r.random((n, 2))
    K = r.random((n, len(SURROGATE_KPIS))) + 1.0
    hyper = good["meta"]["members"][0]["hyper"]

    def many(m):
        m["members"] = [{"group": [], "kpi": k, "train": list(range(1600)), "cal": list(range(1600, n)), "hyper": hyper} for k in SURROGATE_KPIS]
        m["kpis"] = list(SURROGATE_KPIS)
    t0 = time.time()
    refused(good, "cost", edit=many, P=P, K=K)
    assert time.time() - t0 < 10.0
    refused(good, "rows", P=np.zeros((ST.MAX_ROWS + 1, 2)), K=np.zeros((ST.MAX_ROWS + 1, len(SURROGATE_KPIS))))


def test_zip_bomb_is_refused_without_reading_it(good):
    """A member that declares far more data than the file may hold: the header and the declared size are checked, nothing is unpacked."""
    p = good["dir"] / "bomb.npz"
    rows = 3_000_000                                           # 192 MB of zeros, a few hundred KB compressed
    hdr = io.BytesIO()
    np.lib.format.write_array_header_1_0(hdr, {"descr": "<f8", "fortran_order": False, "shape": (rows, 8)})
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        with zf.open("meta.npy", "w") as f:
            np.lib.format.write_array(f, np.array(json.dumps(good["meta"])))
        with zf.open("K.npy", "w") as f:
            np.lib.format.write_array(f, good["K"])
        with zf.open("P.npy", "w", force_zip64=True) as f:
            f.write(hdr.getvalue())
            chunk = bytes(8 << 20)
            for _ in range(rows * 8 * 8 // len(chunk)):
                f.write(chunk)
    assert p.stat().st_size < 2_000_000
    tracemalloc.start()
    t0 = time.time()
    with pytest.raises(SurrogateError):
        load_surrogate(p)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 20e6 and time.time() - t0 < 5.0, (peak, time.time() - t0)


def test_foreign_files_and_members_are_refused(good, tmp_path):
    (tmp_path / "nozip.npz").write_bytes(b"not a zip at all")
    with pytest.raises(SurrogateError):
        load_surrogate(tmp_path / "nozip.npz")
    np.save(tmp_path / "plain.npy", good["P"])
    with pytest.raises(SurrogateError):
        load_surrogate(tmp_path / "plain.npy")
    with pytest.raises(SurrogateError):
        load_surrogate(tmp_path / "missing.npz")
    refused(good, "extra_member", extra={"evil": np.zeros(3)})
    p = good["dir"] / "pickled.npz"                             # an object array: np.load(allow_pickle=False) would refuse, the header check does earlier
    with open(p, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(good["meta"])), P=np.array([{"a": 1}], dtype=object), K=good["K"])
    with pytest.raises(SurrogateError):
        load_surrogate(p)
    with zipfile.ZipFile(good["file"]) as zf:                    # duplicate member names
        raw = {n: zf.read(n) for n in zf.namelist()}
    d = tmp_path / "dup.npz"
    with zipfile.ZipFile(d, "w") as zf:
        for n, b in raw.items():
            zf.writestr(n, b)
        with pytest.warns(UserWarning):                          # zipfile itself warns about the duplicate name
            zf.writestr("P.npy", raw["P.npy"])
    with pytest.raises(SurrogateError):
        load_surrogate(d)
    big = tmp_path / "big.npz"
    big.write_bytes(b"\0" * (ST.MAX_FILE_BYTES + 1))
    with pytest.raises(SurrogateError, match="too large"):
        load_surrogate(big)


def test_parameter_names_stay_plain_text(good):
    """A parameter key with markup is a string like any other: it is carried, never interpreted (the GUI labels are plain text)."""
    p = craft(good, "html", edit=lambda m: m["params"][0].update(key="sketch:<img src=x><b>W</b>"))
    # the rows are keyed by the names of the model, so the renamed key is unknown to the table: the check is that nothing is evaluated
    sm = load_surrogate(p)
    assert sm.params[0].key == "sketch:<img src=x><b>W</b>" and sm.describe()["parameters"][0]["key"] == "sketch:<img src=x><b>W</b>"


# ------------------------------------------------------------------ I5: a damaged file never stops the project from opening
def _damaged_variants(raw: bytes) -> dict[str, bytes]:
    out = {"truncated_half": raw[: len(raw) // 2], "truncated_tail": raw[:-30], "empty": b"", "zeros": bytes(len(raw))}
    for off in (40, 200, len(raw) // 3, len(raw) // 2, len(raw) - 200):
        b = bytearray(raw)
        b[off:off + 8] = b"\xff" * 8
        out[f"flip_{off}"] = bytes(b)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf, zipfile.ZipFile(io.BytesIO(raw)) as src:      # a deflated member, then damaged
        for n in src.namelist():
            zf.writestr(n, src.read(n))
    b = bytearray(buf.getvalue())
    b[60:70] = b"\x00" * 10
    out["deflate_damaged"] = bytes(b)
    return out


def test_a_damaged_surrogate_file_never_stops_the_project_from_opening(good, tmp_path):
    raw = good["file"].read_bytes()
    for name, data in _damaged_variants(raw).items():
        d = tmp_path / name
        d.mkdir()
        (d / SURROGATE_FILE).write_bytes(data)
        warns: list[str] = []
        sm = load_surrogate_file(d, warns)                           # no exception of any kind
        if name.startswith("flip_"):                                 # 8 damaged bytes may hit a value that is still a legitimate number
            assert sm is None or np.isfinite(sm.predict({"a": 0.5, "b": 0.5})["kpis"]["max_stress"]["value"]), name
            continue
        assert sm is None and len(warns) == 1 and "dropped" in warns[0], name
        with pytest.raises(SurrogateError):
            load_surrogate(d / SURROGATE_FILE)                       # and the loader itself speaks SurrogateError only


@pytest.mark.parametrize("exc", [zipfile.BadZipFile("x"), EOFError("x"), zlib.error("x"), RecursionError("x"), MemoryError(), OSError("x"),
                                 KeyError("x"), ValueError("x"), OverflowError("x"), np.linalg.LinAlgError("x"), AttributeError("x"), TypeError("x"),
                                 IndexError("x"), RuntimeError("x"), UnicodeDecodeError("utf-8", b"", 0, 1, "x")])
def test_any_exception_from_the_reader_drops_the_surrogate(good, tmp_path, monkeypatch, exc):
    d = tmp_path / "any"
    d.mkdir()
    (d / SURROGATE_FILE).write_bytes(good["file"].read_bytes())
    import fea_core.surrogate.store as store

    def boom(path):
        raise exc
    monkeypatch.setattr(store, "_read_model", boom)
    warns: list[str] = []
    assert load_surrogate_file(d, warns) is None and len(warns) == 1                    # read_model maps it, load_surrogate_file catches the rest
    with pytest.raises(SurrogateError):
        load_surrogate(d / SURROGATE_FILE)


def test_project_with_a_damaged_surrogate_opens_and_says_so(good, tmp_path):
    fea = tmp_path / "p.fea"
    save_project(fea, Study(name="s", geometry=""), "", None, surrogate=good["model"])
    raw = fea.read_bytes()
    # replace the surrogate member by garbage: rewrite the zip
    out = tmp_path / "bad.fea"
    with zipfile.ZipFile(fea) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for n in src.namelist():
            dst.writestr(n, b"garbage" * 50 if n == SURROGATE_FILE else src.read(n))
    wd = tmp_path / "wd"
    study, step, res = load_project(out, wd)
    warns: list[str] = []
    assert study.name == "s" and load_surrogate_file(wd, warns) is None and warns and "project opens without it" in warns[0]
    assert raw                                                                      # (the intact project is unchanged)

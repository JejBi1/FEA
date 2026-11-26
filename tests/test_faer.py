"""The faer solver inside the program's CalculiX (F17 stage C): how the program recognises it, what the build scripts pin, what the patch and the crate promise.
Fast: no compiler, no network (the slow part - a real ccx with faer - is the gate of docs/projekt/F17_plan_ccx_equation_solver.md)."""
from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

from fea_core import config

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "installer" / "windows"
LIN = ROOT / "installer" / "linux"
FAER = ROOT / "installer" / "faer_ccx"
PATCH = "ccx_2.23_faer.patch"


# ================================================================= how the program recognises the solver
def test_the_faer_marker_is_read_from_the_executable_not_from_a_name(tmp_path):
    plain = tmp_path / "ccx_spooles.exe"
    plain.write_bytes(b"MZ" + b"\0" * 100 + b"FEA_SPOOLES_FACTOR_CHECK=2")
    marked = tmp_path / "ccx_faer_candidate.exe"
    marked.write_bytes(b"MZ" + b"\0" * 100 + config.FAER_MARKER + b" (faer 0.24.4)")
    assert not config.ccx_carries(plain, config.FAER_MARKER)
    assert config.ccx_carries(marked, config.FAER_MARKER)
    assert not config.ccx_carries(tmp_path / "missing.exe", config.FAER_MARKER) and not config.ccx_carries(None, config.FAER_MARKER)
    empty = tmp_path / "empty.exe"
    empty.write_bytes(b"")
    assert not config.ccx_carries(empty, config.FAER_MARKER)                    # an empty file cannot be mapped: no marker, no error
    plain.write_bytes(plain.read_bytes() + config.FAER_MARKER)                  # a replaced binary is looked at again (the cache is keyed by time and size)
    assert config.ccx_carries(plain, config.FAER_MARKER)


def test_faer_can_be_switched_off_and_the_label_follows_the_binary(tmp_path, monkeypatch):
    marked = tmp_path / "ccx.exe"
    marked.write_bytes(b"MZ" + config.FAER_MARKER)
    plain = tmp_path / "ccx_plain.exe"
    plain.write_bytes(b"MZ")
    monkeypatch.delenv("FEA_FAER", raising=False)
    assert config.faer_in_ccx(marked) and not config.faer_in_ccx(plain)
    assert config.solver_label("SPOOLES", marked) == "FAER" and config.solver_label("SPOOLES", plain) == "SPOOLES"
    assert config.solver_label("PASTIX", marked) == "PASTIX"                    # only the SPOOLES slot is taken over by faer
    monkeypatch.setenv("FEA_FAER", "0")
    assert not config.faer_in_ccx(marked) and config.solver_label("SPOOLES", marked) == "SPOOLES"
    monkeypatch.setenv("FEA_FAER", "1")
    assert config.faer_in_ccx(marked)


def test_test_switches_of_faer_do_not_reach_ccx_unless_dev(monkeypatch):
    for k in ("FEA_FAER_CHECK_TOL", "FEA_FAER_METHOD", "FEA_FAER_ORDER", "FEA_FAER_VERBOSE", "FEA_FAER_MIN_EQ", "FEA_FAER_UNSYM", "FEA_FAER_INJECT", "FEA_FAER_COND_MAX"):
        assert k in config.DEV_ONLY_CCX_VARIABLES
        monkeypatch.setenv(k, "0")
    monkeypatch.setenv("FEA_FAER", "0")
    monkeypatch.setenv("FEA_FAER_SOLVE_THREADS", "2")
    monkeypatch.delenv("FEA_DEV", raising=False)
    env = config.ccx_env()
    assert not any(k in env for k in config.DEV_ONLY_CCX_VARIABLES)
    assert env["FEA_FAER"] == "0" and env["FEA_FAER_SOLVE_THREADS"] == "2"       # the user's switch and the thread count of the solves do reach it
    monkeypatch.setenv("FEA_DEV", "1")
    assert config.ccx_env()["FEA_FAER_CHECK_TOL"] == "0"


# ================================================================= the build: pins, patch, crate
def _lock_rows(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def test_the_rust_toolchain_and_every_crate_are_pinned():
    rows = _lock_rows(WIN / "ccx_rust.lock")
    assert {r[0] for r in rows} == {"rust", "rustsrc"} and len(rows) == 5
    for kind, digest, url in rows:
        assert re.fullmatch(r"[0-9a-f]{64}", digest) and url.startswith("https://static.rust-lang.org/dist/") and url.endswith(".tar.xz")
    assert any("rustc-1.100.0-x86_64-pc-windows-gnu" in r[2] for r in rows)
    assert re.search(r"^# rust_commit: [0-9a-f]{40}$", (WIN / "ccx_rust.lock").read_text(encoding="utf-8"), re.MULTILINE)
    cargo = tomllib.loads((FAER / "Cargo.toml").read_text(encoding="utf-8"))
    assert cargo["dependencies"]["faer"]["version"] == "=0.24.4" and cargo["lib"]["crate-type"] == ["staticlib"]
    assert cargo["package"]["license"] == "MIT" and cargo["profile"]["release"]["panic"] == "unwind"      # a panic must be caught at the C boundary
    lock = tomllib.loads((FAER / "Cargo.lock").read_text(encoding="utf-8"))
    external = [p for p in lock["package"] if p.get("source")]
    assert len(external) >= 40
    for p in external:
        assert p["source"] == "registry+https://github.com/rust-lang/crates.io-index", p["name"]       # nothing from git or another registry
        assert re.fullmatch(r"[0-9a-f]{64}", p["checksum"]), p["name"]
    assert any(p["name"] == "faer" and p["version"] == "0.24.4" for p in external)


def test_every_crate_has_a_licence_that_gpl_2_0_only_calculix_can_take():
    """CalculiX is GPL-2.0-only: no crate may be Apache-2.0 only; the list is generated (installer/faer_ccx/make_crate_licences.py) and must name every crate of the lock."""
    lock = tomllib.loads((FAER / "Cargo.lock").read_text(encoding="utf-8"))
    names = {(p["name"], p["version"]) for p in lock["package"] if p.get("source")}
    text = (FAER / "licenses" / "CRATES.md").read_text(encoding="utf-8")
    rows = {(m.group(1), m.group(2)): m.group(3) for m in re.finditer(r"^\| `([^`]+)` \| (\S+) \| ([^|]+) \| ([^|]+) \|", text, re.MULTILINE)}
    assert names == set(rows), (names ^ set(rows))
    for (name, ver), expr in rows.items():
        assert re.search(r"\b(MIT|BSD-2-Clause|BSD-3-Clause|Zlib|Unlicense|ISC)\b", expr), (name, expr)
        crate_dir = FAER / "licenses" / "crates" / f"{name}-{ver}"
        assert crate_dir.is_dir() and any(crate_dir.iterdir()), f"licence files of {name} {ver}"


def test_the_faer_patch_is_the_same_for_both_builds_and_applies_after_the_serial_solve_patch():
    win = (WIN / "patches" / PATCH).read_bytes()
    lin = (LIN / "patches" / PATCH).read_bytes()
    assert b"\r" not in win and b"\r" not in lin and win == lin                    # one file for both builds
    text = win.decode("latin-1")
    body = text[text.index("--- a/CalculiX"):]
    assert body.count("--- a/CalculiX/ccx_2.23/src/") == 2 and "spooles.c" in body and "ccx_2.23.c" in body
    removed = [ln for ln in body.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert all(ln.strip() in ("-", "-        ") or ln[1:].strip() == "" for ln in removed), removed      # only a whitespace-only line goes: the hunks add, they change no solver code
    assert "#ifdef FEA_FAER" in body and "faerc_try_factor(" in body and "faerc_solve(b,neq);" in body and "faerc_cleanup();" in body and "faerc_marker();" in body
    assert body.count("FEA Studio change, 23. 11. 2026") >= 4                      # GPL-2.0 section 2a: the dated notices
    for rel in (f"installer/windows/patches/{PATCH}", f"installer/linux/patches/{PATCH}", "installer/faer_ccx/ccx/faer.c", "installer/faer_ccx/ccx/faer.h",
                "installer/faer_ccx/Cargo.lock", "installer/faer_ccx/src/lib.rs", "installer/faer_ccx/include/fea_faer.h"):
        assert (ROOT / rel).is_file(), rel
    ga = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "installer/windows/patches/*.patch -text" in ga and "installer/linux/patches/*.patch -text" in ga
    # the real thing when the pinned archive and patch(1) are there: the patches apply in the order of the build to a pristine CalculiX
    patch = shutil.which("patch")
    tarball = next((p for p in (ROOT / "tools" / "winccx_build" / "archive" / "ccx_2.23.src.tar.bz2",) if p.is_file()), None)
    if patch and tarball:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(tarball, "r:bz2") as tf:
                tf.extractall(td, filter="data")
            for name in ("ccx_2.23_ucrt_build.patch", "ccx_2.23_spooles_serial_solve.patch", PATCH):
                r = subprocess.run([patch, "-p1", "-i", str(WIN / "patches" / name)], cwd=td, capture_output=True, text=True, encoding="utf-8", errors="replace")
                assert r.returncode == 0, (name, r.stdout, r.stderr)


def test_the_windows_build_script_builds_faer_from_the_pinned_inputs_and_checks_the_result():
    sh = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    assert "ccx_rust.lock" in sh and 'cargo.exe" build --release --locked --offline' in sh and "--remap-path-prefix" in sh
    assert "https://static.crates.io/crates/" in sh and "faer_crates.txt" in sh                         # every crate with the digest of Cargo.lock
    assert "-DFEA_FAER=1" in sh and "SCCXC += faer.c" in sh and "libfea_faer.a" in sh
    assert re.search(r'CCXSTAMP="\$\(stamp [^\n]*ccx_2\.23_faer\.patch', sh)                            # a change of the patch builds ccx again
    assert "FEA_FAER=1" in sh and sh.count("FEA_FAER=1") >= 3                                           # in the file AND printed by `ccx -v`, checked by the build
    assert '"faer": {' in sh and "ccx_2.23_faer.patch" in sh[sh.index('"patches": ['):]                  # the manifest says so
    assert sh.index("ccx_2.23_spooles_serial_solve.patch\" >> ") < sh.index("ccx_2.23_faer.patch\" >> ")      # the faer hunks need the context of the serial solve patch
    ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "installer/faer_ccx/Cargo.lock" in ci and "ccx_rust.lock" in ci                              # a change of the library builds the cached binaries again


def test_the_glue_never_gives_a_silent_wrong_answer():
    """faer.c: every factorization is checked against the matrix as CalculiX handed it over, and every failure leaves the system to SPOOLES with a line in the log."""
    c = (FAER / "ccx" / "faer.c").read_text(encoding="utf-8")
    assert "faer_check(" in c and "FEA_FAER_CHECK_TOL" in c and "return 1.e-12;" in c and "is solved with SPOOLES" in c
    assert c.count("return 0;") >= 8                                                  # every way out that is not a success hands the system to SPOOLES
    assert "FEA_FAER=1" in c and "e=fabs(y[i]-b0[i])/(rs[i]+fabs(b0[i]));" in c      # the row-wise backward error of the SPOOLES check
    rs = (FAER / "src" / "lib.rs").read_text(encoding="utf-8")
    assert rs.count("catch_unwind") >= 4 and "FEA_FAER_PANIC" in rs                  # no panic crosses the C boundary
    h = (FAER / "include" / "fea_faer.h").read_text(encoding="utf-8")
    for fn in re.findall(r'#\[no_mangle\]\s*pub (?:unsafe )?extern "C" fn (\w+)', rs):
        assert fn in h, fn                                                            # the header declares everything the crate exports


def test_a_system_that_faer_declined_is_one_factorisation_for_the_progress_bar():
    """Review of 24. 11. 2026 (C-10): a declined system prints the factorisation line of faer AND the one of SPOOLES; the progress parser (progress.CcxLines, FACTOR_MARKS) takes
    them for one factorisation (the second line gives no new report), while two systems in a row, or a faer success followed by a SPOOLES system, are still two."""
    from fea_core.progress import CcxLines
    faer = " Factoring the system of equations using the symmetric faer solver"
    spooles = " Factoring the system of equations using the symmetric spooles solver"
    declined = " FEA faer: the factorization failed (zero pivot (singular matrix)) - the system of equations is solved with SPOOLES."
    ok_line = " faer: Cholesky factorization 0.12 s, relative residual of the check 1.0e-16"

    def feed(lines, n_steps=1):
        c = CcxLines(n_steps)
        return [c(l) for l in lines], c
    out, c = feed([" STEP 1", faer, " Using up to 4 cpu(s) for faer.", declined, spooles, " Using up to 4 cpu(s) for spooles.", " stress calculation", "Job finished"])
    assert out[1] is not None and out[4] is None and out[6] is not None and out[7][0] == 1.0, out          # the second line of the declined system reports nothing
    assert c.factored == {1} and not c._faer_offered
    out, _ = feed([" STEP 1", faer, ok_line, " some other line of ccx", spooles])                         # faer succeeded; the next system is factorized by SPOOLES: a new report
    assert out[4] is not None
    out, _ = feed([" STEP 1", spooles, spooles])                                                           # two SPOOLES systems are two factorisations
    assert out[1] is not None and out[2] is not None
    out, _ = feed([" STEP 1", faer, declined, spooles, faer, declined, spooles])                          # two declined systems
    assert out[3] is None and out[4] is not None and out[6] is None
    out, _ = feed(["[step 1] " + faer, "[step 1] " + declined, "[step 1] " + spooles, "[step 2] " + spooles], 2)    # parallel jobs: keyed by the job
    assert out[0] is not None and out[2] is None and out[3] is not None


def _faer_pure_helpers() -> str:
    c = (FAER / "ccx" / "faer.c").read_text(encoding="utf-8")
    return c[c.index("/* HARNESS-BEGIN"):c.index("/* HARNESS-END */")]


def test_faer_c_checks_the_column_counts_before_it_hashes_the_pattern_and_remembers_a_decline():
    """Review of 24. 11. 2026, C-3 / C-7 / C-8 (faer.c is C, compiled only inside ccx: the text is pinned here, the pure helpers are run in the next test)."""
    c = (FAER / "ccx" / "faer.c").read_text(encoding="utf-8")
    assert c.index("check_counts((int)fmt,n,nzs[0],icol,irow,NULL,&why)") < c.index("key=hash_ints(key,icol,(size_t)n)") < c.index("key=hash_ints(key,irow,(size_t)*nzs)")   # C-7: the sizes first
    assert "(G.nzs==nzs[0])" in c and "G.nzs=nzs[0];" in c                          # C-7: nzs is stored and compared when the analysis is reused
    assert "(nnz>=0x7fffff00LL)" in c and "0xffffff00" not in c                    # C-8
    assert "declined_skip(&G.cond_declined[sig!=0.],&G.cond_skips[sig!=0.])" in c and "G.cond_declined[sig!=0.]=1;" in c       # C-3: remembered per pattern and per shift
    assert c.count("G.cond_declined[0]=G.cond_declined[1]=0;") == 1 and c.index("static void free_pattern") < c.index("G.cond_declined[0]=G.cond_declined[1]=0;")   # a new pattern forgets it
    assert c.index("declined_skip(&G.cond_declined") < c.index("Factoring the system of equations using the %s faer solver")     # skipped before the banner of faer


def test_the_pure_helpers_of_faer_c_count_columns_and_skip_a_declined_pattern(tmp_path):
    """check_counts and declined_skip of faer.c compiled alone with tests/faer_guard_harness.c (skipped without a C compiler)."""
    from tests.test_spooles_check import _c_compiler
    cc, env = _c_compiler()
    if cc is None:
        pytest.skip("no C compiler")
    (tmp_path / "faer_pure.inc").write_text(_faer_pure_helpers(), encoding="utf-8")
    exe = tmp_path / "faer_guard.exe"
    r = subprocess.run([cc, "-O1", "-Wall", "-I", str(tmp_path), "-o", str(exe), str(ROOT / "tests" / "faer_guard_harness.c")], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([str(exe)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0 and "ALL CHECKS OK" in r.stdout and "FAIL" not in r.stdout, r.stdout


def test_every_export_of_the_crate_catches_a_panic():
    """Review of 24. 11. 2026 (C-5): no panic may cross the C boundary - EVERY exported function (also the setters, the version and the statistics) runs its body under
    catch_unwind (directly or through guard())."""
    rs = (FAER / "src" / "lib.rs").read_text(encoding="utf-8")
    names = re.findall(r'#\[no_mangle\]\s*pub (?:unsafe )?extern "C" fn (\w+)', rs)
    assert len(names) >= 10, names
    for name in names:
        start = rs.index(f"fn {name}(")
        depth, i = 0, rs.index("{", rs.index(")", start))
        j = i
        while True:
            depth += (rs[j] == "{") - (rs[j] == "}")
            j += 1
            if depth == 0:
                break
        body = rs[i:j]
        assert "catch_unwind(" in body or "guard(" in body, f"{name} is exported without a guard against a panic"


@pytest.mark.skipif(not config.faer_in_ccx(), reason="the ccx found by the program does not carry the faer solver")
def test_the_ccx_of_this_machine_with_faer_solves_and_says_so(tmp_path):
    """Only where a faer ccx is installed (FEA_CCX=... in the gate): a small cantilever through the CLI path of the program, solved by faer (the log says so)."""
    import numpy as np  # noqa: F401 - the module imports numpy anyway
    from verification import cases as C
    from fea_core.solve import run_study
    case = C.CASES["V2"]
    study, _parts = case.build(tmp_path)
    res = run_study(study, tmp_path / "run")
    assert res.info["ccx_solver"] == "FAER"
    assert all(c.ok for c in case.evaluate(res, study))


# ================================================================= what the user is told
def test_the_runner_turns_the_faer_fallback_lines_into_one_note_with_a_czech_translation():
    from fea_core import i18n
    from fea_core.ccx.runner import SpoolesCheck, spooles_check_note
    chk = SpoolesCheck()
    lines = [" *WARNING in faer: the factors fail the check of their result (relative residual of a test solve 5.3e-06, limit 1.0e-12) - the system of equations is solved with SPOOLES.",
             " FEA faer: the factorization failed (zero pivot (singular matrix)) - the system of equations is solved with SPOOLES.",
             " faer: the Cholesky factors fail the check (relative residual 1e-8), factorized again as LDL^T.",       # the ladder inside faer: not a fallback, no note
             " faer: Cholesky factorization 2.34 s, relative residual of the check 5.6e-16"]
    taken = [chk.feed(ln.strip()) for ln in lines]
    assert taken == [True, True, False, False] and chk.faer == 2 and chk.any()
    note = spooles_check_note("job", chk)
    assert "faer" in note and "SPOOLES solved them instead" in note and "2 system(s)" in note
    with i18n.using("cs"):
        assert i18n.tr(note) != note and "faer" in i18n.tr(note) and "CalculiX" in i18n.tr(note)
    assert not SpoolesCheck().any() and spooles_check_note("job", SpoolesCheck()) == ""


def test_the_solver_label_reaches_the_results_info(tmp_path, monkeypatch):
    """Results.info["ccx_solvers"] names FAER when the program's ccx has the faer solver (the deck still says SPOOLES)."""
    from fea_core import solve
    from fea_core.results import Results
    marked = tmp_path / "ccx.exe"
    marked.write_bytes(b"MZ" + config.FAER_MARKER)
    monkeypatch.setattr(config, "CCX_EXE", marked)
    monkeypatch.delenv("FEA_FAER", raising=False)

    class Study:                                                              # the part of a study that record_ccx_solvers reads
        class solver:
            solver = config.AUTO_SOLVER
    res = Results.__new__(Results)
    res.info = {}
    solve.record_ccx_solvers(Study, res, ["static", "modal"])
    assert res.info["ccx_solvers"] == {"static": "FAER", "modal": "FAER"} and res.info["ccx_solver"] == "FAER"
    monkeypatch.setenv("FEA_FAER", "0")
    res.info = {}
    solve.record_ccx_solvers(Study, res, ["static"])
    assert res.info["ccx_solvers"] == {"static": "SPOOLES"}
    assert "SOLVER=SPOOLES" in (ROOT / "fea_core" / "ccx" / "inp.py").read_text(encoding="utf-8") or config.solver_name("SPOOLES") == "SPOOLES"    # the deck keyword is unchanged


def _fake_run(tmp_path, monkeypatch, text, job="job"):
    """run_ccx of a fake ccx.exe that carries the faer marker and prints `text`."""
    from fea_core.ccx import runner
    from tests.test_spooles_check import _Proc
    exe = tmp_path / "ccx.exe"
    exe.write_bytes(b"MZ" + config.FAER_MARKER + runner.FACTOR_CHECK_MARKER + runner.ORDERED_MARKER)
    (tmp_path / f"{job}.inp").write_text("""*STEP
*STATIC
*END STEP
""")
    monkeypatch.setattr(config, "CCX_EXE", exe)
    monkeypatch.delenv("FEA_FAER", raising=False)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda cmd, **kw: _Proc(text))
    monkeypatch.setattr(runner.dllguard, "check_workdir", lambda *a, **k: None)
    return runner.run_ccx(tmp_path, job, exe=exe, threads=4)


def test_the_name_faer_stays_only_when_faer_factorized_a_system(tmp_path, monkeypatch):
    """A ccx with the faer solver that handed every system to SPOOLES (unsymmetric friction contact, small models) must not be reported as FAER."""
    import time
    from fea_core import solve
    from fea_core.results import Results

    class Study:
        class solver:
            solver = config.AUTO_SOLVER
    faer_line = """ Factoring the system of equations using the symmetric faer solver
 Using up to 4 cpu(s) for faer.

"""
    spooles_line = """ Factoring the system of equations using the spooles solver

"""
    fallback = """ FEA faer: the factorization failed (zero pivot (singular matrix)) - the system of equations is solved with SPOOLES.

Job finished
"""
    done = "Job finished\n"
    singular = " faer: the system of equations is numerically singular (its condition number is at least 7.8e+16, limit 1.0e+12) - SPOOLES solves it as before.\n\n"
    for text, label, folder in ((spooles_line * 3 + done, "SPOOLES", "a"),                                    # SPOOLES all along (unsymmetric / small)
                                (spooles_line + faer_line + done, "FAER", "b"),                               # faer made one
                                (faer_line + singular + done, "SPOOLES", "f"),                                # faer declined a numerically singular system: SPOOLES solved it
                                (faer_line + singular + faer_line + done, "FAER", "g"),                       # ... and faer did the next one
                                (faer_line + fallback, "SPOOLES", "c"),                                       # faer tried, SPOOLES took over: no faer factors
                                (done, "SPOOLES", "d")):                                                      # no factorization line at all
        wd = tmp_path / folder
        wd.mkdir()
        t0 = time.monotonic()
        _fake_run(wd, monkeypatch, text)
        res = Results.__new__(Results)
        res.info, res.warnings = {}, []
        solve.record_ccx_solvers(Study, res, ["static", "modal"])
        assert res.info["ccx_solvers"] == {"static": "FAER", "modal": "FAER"}                                   # the label before the run is read
        solve.attach_solver_notes(res, wd, t0)
        assert res.info["ccx_solver"] == label and set(res.info["ccx_solvers"].values()) == {label}, (folder, res.info)
    # a study whose ccx runs left no record (the NGSolve backend, hypre) keeps the label it has
    res = Results.__new__(Results)
    res.info, res.warnings = {}, []
    solve.record_ccx_solvers(Study, res, ["static"])
    solve.attach_solver_notes(res, tmp_path / "nowhere", 0.0)
    assert res.info["ccx_solver"] == "FAER"
    # the usage is taken once and belongs to the folder and the time of the run
    t0 = time.monotonic()
    wd = tmp_path / "e"
    wd.mkdir()
    _fake_run(wd, monkeypatch, faer_line * 2 + done)
    from fea_core.ccx import runner
    assert runner.take_faer_usage(wd, t0 + 100) == [] and runner.take_faer_usage(wd, t0) == [2] and runner.take_faer_usage(wd, t0) == []


def test_a_structure_that_faer_does_not_take_is_not_a_failure_and_the_faer_note_claims_nothing(tmp_path, monkeypatch):
    """Review of 24. 11. 2026 (C-4): "FEA faer: <structure> - the system of equations is solved with SPOOLES" (the column counts do not add up, row pointers that are not the expected
    ones, too large for 32 bit indices) means "not applicable": no user warning, and the system is not counted as one that faer factorized. A real failure (the factorization failed,
    no memory) still raises the note, and the note no longer says that the results are correct."""
    import time
    from fea_core.ccx import runner
    from fea_core.ccx.runner import SpoolesCheck, spooles_check_note
    faer_line = """ Factoring the system of equations using the symmetric faer solver
 Using up to 4 cpu(s) for faer.

"""
    done = "Job finished\n"
    structural = [" FEA faer: the column counts do not add up to nzs - the system of equations is solved with SPOOLES.\n\n",
                  " FEA faer: unexpected row pointers (input format 4) - the system of equations is solved with SPOOLES.\n\n",
                  " FEA faer: the matrix does not fit 32 bit indices - the system of equations is solved with SPOOLES.\n\n"]
    failure = " FEA faer: the factorization failed (zero pivot (singular matrix)) - the system of equations is solved with SPOOLES.\n\n"
    nomem = " FEA faer: no memory for the values - the system of equations is solved with SPOOLES.\n\n"
    for line in structural:
        chk = SpoolesCheck()
        assert chk.feed(line.strip()) and chk.faer == 0 and chk.faer_structural == 1 and not chk.any() and spooles_check_note("j", chk) == "", line
    for line in (failure, nomem):
        chk = SpoolesCheck()
        assert chk.feed(line.strip()) and chk.faer == 1 and chk.faer_structural == 0 and chk.any(), line
        note = spooles_check_note("j", chk)
        assert "SPOOLES solved them instead" in note and "correct" not in note, note
    for k, line in enumerate(structural):
        wd = tmp_path / f"s{k}"
        wd.mkdir()
        t0 = time.monotonic()
        run = _fake_run(wd, monkeypatch, faer_line + line + done)
        assert run.finished and run.warnings == [], (line, run.warnings)                        # no user warning
        assert runner.take_solver_events(wd, t0) == []                                           # nothing registered for the results
        assert runner.take_faer_usage(wd, t0) == [0]                                             # SPOOLES factorized the system: faer did not
    wd = tmp_path / "mixed"
    wd.mkdir()
    t0 = time.monotonic()
    run = _fake_run(wd, monkeypatch, faer_line * 3 + structural[0] + failure + done)           # three offered: one not applicable, one failed, one factorized by faer
    assert len(run.warnings) == 1 and "1 system(s)" in run.warnings[0] and "correct" not in run.warnings[0], run.warnings
    assert runner.take_faer_usage(wd, t0) == [1]


def test_an_explicit_choice_of_spooles_keeps_faer_off_and_is_named_so(tmp_path, monkeypatch):
    from fea_core import solve
    from fea_core.ccx import runner
    from fea_core.results import Results
    deck = tmp_path / "job.inp"
    deck.write_text("** FEA Studio - x\n** FEA_SOLVER=SPOOLES\n*HEADING\nx\n", encoding="ascii")
    other = tmp_path / "other.inp"
    other.write_text("** FEA Studio - x\n*HEADING\nx\n", encoding="ascii")
    assert runner._names(deck, runner.FAER_OFF_COMMENT) == [True] and runner._names(other, runner.FAER_OFF_COMMENT) == [False]
    assert '"** FEA_SOLVER=SPOOLES' in (ROOT / "fea_core" / "ccx" / "inp.py").read_text(encoding="utf-8")        # write_inp writes the comment for the explicit choice
    marked = tmp_path / "ccx.exe"
    marked.write_bytes(b"MZ" + config.FAER_MARKER)
    monkeypatch.setattr(config, "CCX_EXE", marked)
    monkeypatch.delenv("FEA_FAER", raising=False)

    class Study:
        class solver:
            solver = "SPOOLES"
    res = Results.__new__(Results)
    res.info = {}
    solve.record_ccx_solvers(Study, res, ["static"])
    assert res.info["ccx_solvers"] == {"static": "SPOOLES"}                    # asked for by name: it is SPOOLES, not FAER


def test_the_linux_build_script_builds_faer_from_the_same_pins():
    lin = (LIN / "build_ccx.sh").read_text(encoding="utf-8")
    assert "ccx_rust.lock" in lin and '"$RUSTROOT/bin/cargo" build --release --locked --offline' in lin and "--remap-path-prefix" in lin
    assert "https://static.crates.io/crates/" in lin and "-DFEA_FAER=1" in lin and "SCCXC += faer.c" in lin and "FEA_FAER=1" in lin
    assert 'build_one ccx_spooles "-DUSE_MT=1 -DFEA_FAER=1' in lin and "ccx_pastix" in lin
    rows = _lock_rows(LIN / "ccx_rust.lock")
    assert {r[0] for r in rows} == {"rust", "rustsrc"} and len(rows) == 4
    assert all(re.fullmatch(r"[0-9a-f]{64}", r[1]) and "x86_64-unknown-linux-gnu" in r[2] for r in rows if r[0] == "rust")
    win = _lock_rows(WIN / "ccx_rust.lock")
    assert [r for r in rows if r[0] == "rustsrc"] == [r for r in win if r[0] == "rustsrc"]                 # one source of the standard library for both
    commit = lambda p: re.search(r"^# rust_commit: ([0-9a-f]{40})$", p.read_text(encoding="utf-8"), re.MULTILINE).group(1)       # noqa: E731
    assert commit(LIN / "ccx_rust.lock") == commit(WIN / "ccx_rust.lock")                                 # the same Rust on both platforms

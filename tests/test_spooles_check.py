"""The internal check of the SPOOLES factors of the own ccx build (installer/*/patches/ccx_2.23_spooles_serial_solve.patch): what the program makes of the
messages of ccx (a note for the user in the results, not the raw lines), that the test switch of the limit does not reach the process of ccx from a user's
environment, and (slow) the built binary with the limit forced tiny: the retries are reported and the eigenvalues stay correct."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from fea_core import config
from fea_core.ccx import runner
from fea_core.ccx.runner import SERIAL_SOLVE_MARKER, SpoolesCheck, spooles_check_note

# what a run prints when the multi-threaded factorization fails its check twice and the repeat on one thread passes
LOSS = """\
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 3.000e-05, limit 1.0e-12).
                      The system of equations is factorized again with 4 threads.

 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 4.100e-05, limit 1.0e-12).
                      The system of equations is factorized again with ONE thread.

                      The factorization on ONE thread passes the check (relative residual 2.0e-17): threaded loss confirmed.

Job finished
"""
REPEAT_OK = """\
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 3.000e-05, limit 1.0e-12).
                      The system of equations is factorized again with 4 threads.

                      The repeated multi-threaded factorization passes the check (relative residual 2.0e-17): threaded loss confirmed.
Job finished
"""
MATRIX = """\
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 2.000e-09, limit 1.0e-12).
                      The system of equations is factorized again with 4 threads.

 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 2.000e-09, limit 1.0e-12).
                      The system of equations is factorized again with ONE thread.

 *WARNING in spooles: the factorization on ONE thread fails the check as well (relative residual of a test solve 2.000e-09, limit 1.0e-12):
                      the matrix, not the threads, is the cause (nearly singular or very badly scaled system of equations). The factors are used
                      as they are and the check is switched off for the rest of this run.

Job finished
"""
BOOKKEEPING = """\
 *WARNING in spooles: the bookkeeping of the aggregates of the multi-threaded factorization is violated (1 violations; aggregates sent 232, taken 232, assembled 231)
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 3.000e-17, limit 1.0e-12; the bookkeeping of the hand-over between the threads is violated).
                      The system of equations is factorized again with 4 threads.

                      The repeated multi-threaded factorization passes the check (relative residual 2.0e-17): threaded loss confirmed.
Job finished
"""
# a repeat whose own test solve could not be done (no memory) is used without a verdict: the program must not call it checked / correct
REPEAT_UNCHECKED = """\
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 3.000e-05, limit 1.0e-12).
                      The system of equations is factorized again with 4 threads.

 *WARNING in spooles: the factors of the repeated factorization could not be checked (not enough memory for the test solve).
                      The repeated multi-threaded factorization is used without a verdict: a loss in the threads is NOT confirmed.
Job finished
"""
SERIAL_UNCHECKED = """\
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 3.000e-05, limit 1.0e-12).
                      The system of equations is factorized again with 4 threads.

 *WARNING in spooles: the multi-threaded factorization failed the check of its result (relative residual of a test solve 4.000e-05, limit 1.0e-12).
                      The system of equations is factorized again with ONE thread.

 *WARNING in spooles: the factors of the repeated factorization could not be checked (not enough memory for the test solve).
                      The factorization on ONE thread is used without a verdict: a loss in the threads is NOT confirmed.
Job finished
"""
# the first test solve could not be done but the bookkeeping of SPOOLES was violated: repeated, and THIS repeat was checked and passed
UNCHECKED_THEN_CONFIRMED = """\
 *WARNING in spooles: the factors could not be checked (not enough memory for the test solve).
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (the bookkeeping of the hand-over between the threads is violated; the test solve of the factors could not be done).
                      The system of equations is factorized again with 4 threads.

                      The repeated multi-threaded factorization passes the check (relative residual 2.0e-17): threaded loss confirmed.
Job finished
"""
# the bookkeeping violated while the test solve is off (the flag of an earlier matrix, or the test limit 0): the repeat is NOT confirmed by anything
BOOKKEEPING_CHECK_OFF = """\
 *WARNING in spooles: the multi-threaded factorization failed the check of its result (the bookkeeping of the hand-over between the threads is violated; the test solve of the factors is switched off).
                      The system of equations is factorized again with 4 threads.

Job finished
"""
HARMLESS = " *WARNING: no elastic constants were assigned to some elements\nJob finished\n"


class _Out:
    def __init__(self, text):
        self._lines = text.splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        pass


class _Proc:
    """A fake ccx process that prints `text`."""
    def __init__(self, text):
        self.args = ["ccx"]
        self.returncode = 0
        self.stdout = _Out(text)


    def poll(self):
        return 0

    def wait(self):
        return 0


def _run(tmp_path, monkeypatch, text, job="eigen_modal"):
    exe = tmp_path / "ccx"
    exe.write_bytes(b"x" + runner.FACTOR_CHECK_MARKER + runner.ORDERED_MARKER)
    (tmp_path / f"{job}.inp").write_text("*STEP\n*FREQUENCY\n*END STEP\n")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda cmd, **kw: _Proc(text))
    monkeypatch.setattr(runner.dllguard, "check_workdir", lambda *a, **k: None)
    return runner.run_ccx(tmp_path, job, exe=exe, threads=4)


def test_the_messages_of_the_check_are_parsed_into_one_state():
    for text, want in ((LOSS, dict(failed=2, confirmed=True, matrix=False, bookkeeping=False, unchecked=False)),
                       (REPEAT_OK, dict(failed=1, confirmed=True, matrix=False, bookkeeping=False, unchecked=False)),
                       (MATRIX, dict(failed=2, confirmed=False, matrix=True, bookkeeping=False, unchecked=False)),
                       (BOOKKEEPING, dict(failed=1, confirmed=True, matrix=False, bookkeeping=True, unchecked=False)),
                       (REPEAT_UNCHECKED, dict(failed=1, confirmed=False, matrix=False, bookkeeping=False, unchecked=True)),
                       (SERIAL_UNCHECKED, dict(failed=2, confirmed=False, matrix=False, bookkeeping=False, unchecked=True)),
                       (UNCHECKED_THEN_CONFIRMED, dict(failed=1, confirmed=True, matrix=False, bookkeeping=True, unchecked=True)),
                       (BOOKKEEPING_CHECK_OFF, dict(failed=1, confirmed=False, matrix=False, bookkeeping=True, unchecked=False)),
                       (HARMLESS, dict(failed=0, confirmed=False, matrix=False, bookkeeping=False, unchecked=False))):
        chk = SpoolesCheck()
        for line in text.splitlines():
            chk.feed(line.strip())
        assert {k: getattr(chk, k) for k in want} == want, text
    chk = SpoolesCheck()
    assert not chk.feed("*WARNING in CalculiX: something else") and not chk.any()           # other warnings are not the check's
    assert chk.feed("*WARNING in spooles: the factors could not be checked (not enough memory for the test solve).") and chk.unchecked and chk.any()
    note = spooles_check_note("job", chk)                                                          # a check that could not be done is also said (the results were not verified)
    assert "could not be done" in note and "not verified" in note and "failed" not in note


def test_a_repeat_is_called_correct_only_when_its_own_check_passed():
    """Review of 23. 11. 2026 (b): "the final factors passed the check, the results are correct" needs a confirming line of the LAST factorization. A repeat whose test solve could
    not be done (no memory), and a repeat that nothing confirmed (the test solve is off, a truncated log) are reported as "not verified"."""
    def note(text):
        chk = SpoolesCheck()
        for line in text.splitlines():
            chk.feed(line.strip())
        return spooles_check_note("eigen_modal", chk)
    for text in (LOSS, REPEAT_OK, BOOKKEEPING):
        assert "the results are correct" in note(text) and "not verified" not in note(text), text
    n = note(UNCHECKED_THEN_CONFIRMED)           # review of 24. 11. 2026 (A-3): `unchecked` was set at some time of the run, so the last repeat's pass does not make it "correct"
    assert "results are correct" not in n and "not fully verified" in n and "could not be done" in n and "was repeated" in n, n
    for text in (REPEAT_UNCHECKED, SERIAL_UNCHECKED):
        n = note(text)
        assert "was repeated, but the check of the final factors could not be done" in n and "not enough memory" in n and "not verified" in n and "results are correct" not in n, text
    n = note(BOOKKEEPING_CHECK_OFF)
    assert "the check did not confirm the final factors" in n and "not verified" in n and "results are correct" not in n and "could not be done" not in n
    assert "also on one thread" in note(MATRIX) and "results are correct" not in note(MATRIX)
    # a later passing repeat confirms again; an unchecked line after a confirmation takes it back (the last factorization decides)
    chk = SpoolesCheck()
    for line in (REPEAT_UNCHECKED + LOSS).splitlines():
        chk.feed(line.strip())
    n = spooles_check_note("j", chk)
    assert chk.confirmed and chk.unchecked and "results are correct" not in n and "not fully verified" in n, n          # A-3: `unchecked` stays set
    chk = SpoolesCheck()
    for line in (REPEAT_OK + REPEAT_UNCHECKED).splitlines():
        chk.feed(line.strip())
    assert not chk.confirmed and "not verified" in spooles_check_note("j", chk)


@pytest.mark.parametrize("text,key", [(LOSS, "repeated"), (REPEAT_OK, "repeated"), (BOOKKEEPING, "repeated"), (MATRIX, "matrix"), (REPEAT_UNCHECKED, "unverified"),
                                      (SERIAL_UNCHECKED, "unverified"), (BOOKKEEPING_CHECK_OFF, "unverified"), (UNCHECKED_THEN_CONFIRMED, "partly")])
def test_a_firing_check_becomes_one_note_for_the_user_not_the_raw_lines(tmp_path, monkeypatch, text, key):
    run = _run(tmp_path, monkeypatch, text)
    assert run.finished and not run.errors
    assert len(run.warnings) == 1 and "*WARNING" not in run.warnings[0] and "(eigen_modal)" in run.warnings[0]
    if key == "matrix":
        assert "also on one thread" in run.warnings[0] and "may be inaccurate" in run.warnings[0]
    elif key == "unverified":
        assert "was repeated" in run.warnings[0] and "not verified" in run.warnings[0] and "results are correct" not in run.warnings[0]
    elif key == "partly":
        assert "was repeated" in run.warnings[0] and "not fully verified" in run.warnings[0] and "results are correct" not in run.warnings[0]
    else:
        assert "was repeated" in run.warnings[0] and "results are correct" in run.warnings[0]
    assert "failed the check" in (tmp_path / "eigen_modal.log").read_text(encoding="utf-8")        # the log keeps every raw line
    events = runner.take_solver_events(tmp_path)
    assert len(events) == 1 and events[0]["job"] == "eigen_modal" and events[0]["note"] == run.warnings[0]
    assert runner.take_solver_events(tmp_path) == []                                              # taken once


def test_a_run_without_a_firing_check_has_no_note_and_keeps_its_other_warnings(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch, HARMLESS)
    assert run.warnings == ["*WARNING: no elastic constants were assigned to some elements"]
    assert runner.take_solver_events(tmp_path) == []


def test_the_events_belong_to_the_folder_and_the_time_of_the_study_run(tmp_path, monkeypatch):
    import time
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    t0 = time.monotonic()
    _run(tmp_path / "a", monkeypatch, LOSS, job="job_a")
    _run(tmp_path / "b", monkeypatch, MATRIX, job="job_b")
    assert runner.take_solver_events(tmp_path / "a", t0 + 100) == []                              # an event of an earlier run is not this run's
    assert [e["job"] for e in runner.take_solver_events(tmp_path / "a", t0)] == ["job_a"]
    assert runner.take_solver_events(tmp_path / "a", t0) == []
    assert [e["job"] for e in runner.take_solver_events(tmp_path, t0)] == ["job_b"]               # the parent folder collects what is below it
    assert runner.take_solver_events(tmp_path / "nowhere", t0) == []


def test_the_results_of_a_study_run_carry_the_note_and_the_data(tmp_path, monkeypatch):
    from fea_core import solve
    import time
    t0 = time.monotonic()
    run = _run(tmp_path, monkeypatch, LOSS)
    res = NS(warnings=["an earlier warning"], info={})
    solve.attach_solver_notes(res, tmp_path, t0)
    assert res.warnings == ["an earlier warning", run.warnings[0]] and res.info["solver_check"][0]["confirmed"] is True and "note" not in res.info["solver_check"][0]
    again = NS(warnings=[], info={})
    solve.attach_solver_notes(again, tmp_path, t0)                                                # nothing left, nothing added
    assert again.warnings == [] and "solver_check" not in again.info


def test_the_note_texts_have_a_czech_translation():
    from fea_core import i18n
    chk = SpoolesCheck(failed=2, confirmed=True)
    texts = [spooles_check_note("eigen_modal", chk), spooles_check_note("eigen_modal", SpoolesCheck(failed=2, matrix=True)),
             spooles_check_note("eigen_modal", SpoolesCheck(unchecked=True)), spooles_check_note("eigen_modal", SpoolesCheck(failed=1, unchecked=True)),
             spooles_check_note("eigen_modal", SpoolesCheck(failed=1, bookkeeping=True)),
             spooles_check_note("eigen_modal", SpoolesCheck(failed=1, confirmed=True, unchecked=True)),      # A-3: confirmed, but a check of the run could not be done
             spooles_check_note("eigen_modal", SpoolesCheck(faer=1))]                                         # C-4: the faer note without a claim of correctness
    assert all(texts) and len(set(texts)) == 7
    with i18n.using("cs"):
        for t in texts:
            assert i18n.tr(t) != t and "CalculiX" in i18n.tr(t), t


# ================================================================= the test switch of the limit
def test_the_limit_of_the_check_does_not_reach_ccx_from_a_users_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FEA_SPOOLES_CHECK_TOL", "0")
    monkeypatch.delenv("FEA_DEV", raising=False)
    assert "FEA_SPOOLES_CHECK_TOL" not in config.ccx_env() and not config.dev_mode()
    monkeypatch.setenv("FEA_DEV", "0")
    assert "FEA_SPOOLES_CHECK_TOL" not in config.ccx_env()
    monkeypatch.setenv("FEA_DEV", "1")                                                            # a developer's switch
    assert config.dev_mode() and config.ccx_env()["FEA_SPOOLES_CHECK_TOL"] == "0"
    assert "FEA_SPOOLES_CHECK_TOL" in config.DEV_ONLY_CCX_VARIABLES
    monkeypatch.delenv("FEA_DEV")
    seen = {}

    def popen(cmd, **kw):
        seen["env"] = kw["env"]
        return _Proc("Job finished\n")
    exe = tmp_path / "ccx.exe"
    exe.write_bytes(b"x" + runner.FACTOR_CHECK_MARKER + runner.ORDERED_MARKER)
    (tmp_path / "j.inp").write_text("*STEP\n*END STEP\n")
    monkeypatch.setenv("FEA_SPOOLES_CHECK_TOL", "0")
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.dllguard, "check_workdir", lambda *a, **k: None)
    runner.run_ccx(tmp_path, "j", exe=exe, threads=2)
    assert "FEA_SPOOLES_CHECK_TOL" not in seen["env"]
    monkeypatch.setenv("FEA_DEV", "1")
    runner.run_ccx(tmp_path, "j", exe=exe, threads=2)
    assert seen["env"]["FEA_SPOOLES_CHECK_TOL"] == "0"


# ================================================================= the verdict function of the patch, compiled alone
ROOT = Path(__file__).resolve().parents[1]
PATCH_NAME = "ccx_2.23_spooles_serial_solve.patch"
JUDGE_HARNESS = Path(__file__).with_name("spooles_judge_harness.c")


def _c_compiler() -> tuple[str | None, dict]:
    env = dict(os.environ)
    for name in ("gcc", "cc", "clang"):
        exe = shutil.which(name)
        if exe:
            return exe, env
    tc = ROOT / "tools" / "winccx_build" / "msys64" / "ucrt64" / "bin"                          # the pinned toolchain of installer/windows/build_ccx.ps1 (a developer machine)
    if (tc / "gcc.exe").is_file():
        env["PATH"] = str(tc) + os.pathsep + env.get("PATH", "")
        return str(tc / "gcc.exe"), env
    return None, env


def _judge_source(patch: Path) -> str:
    """The text of `fea_judge` as the patch adds it to spooles.c (the added lines of the hunks, without the plus sign)."""
    text = patch.read_text(encoding="utf-8")
    added = "\n".join(l[1:] for l in text[text.index("--- a/CalculiX"):].splitlines() if l.startswith("+") and not l.startswith("+++"))
    return added[added.index("static int fea_judge("):added.index("/* link proof of the ordered assembly")]


def _threaded_conditions(patch: Path) -> list[str]:
    """The conditions of spooles_factor and spooles_factor_rad that choose between the factorization on the threads and the one on ONE thread (the added lines of the patch)."""
    text = patch.read_text(encoding="utf-8")
    added = "\n".join(l[1:] for l in text[text.index("--- a/CalculiX"):].splitlines() if l.startswith("+") and not l.startswith("+++"))
    return re.findall(r"if \((\(num_cpus > 1\)&&\(fea_attempt < 2\).*?)\) \{", added)


def _build_harness(tmp_path: Path, patch: Path, cond: str | None = None) -> tuple[Path, dict]:
    """Compiles tests/spooles_judge_harness.c with the `fea_judge` text of the patch and the threads condition of the patch (or `cond`, a mutant); skips without a C compiler."""
    cc, env = _c_compiler()
    if cc is None:
        pytest.skip("no C compiler")
    conds = _threaded_conditions(patch)
    assert len(conds) == 2 and conds[0] == conds[1], conds                       # spooles_factor and spooles_factor_rad decide alike
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "judge_body.inc").write_text(_judge_source(patch), encoding="utf-8")
    (tmp_path / "threaded_cond.inc").write_text("#define FEA_THREADED_COND " + (cond or conds[0]) + "\n", encoding="utf-8")
    exe = tmp_path / ("judge_test.exe" if os.name == "nt" else "judge_test")
    r = subprocess.run([cc, "-O1", "-I", str(tmp_path), "-o", str(exe), str(JUDGE_HARNESS), "-lm"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return exe, env


# scenario of the harness -> what the note for the user must be (the output of the C function goes through SpoolesCheck and spooles_check_note)
JUDGE_NOTES = {
    "clean": "none",
    "bad + static off-switch (earlier matrix)": "unverified",
    "bad twice + test limit 0": "unverified",
    "test solve fails, threaded repeat passes": "correct",
    "fails twice, one thread passes": "correct",
    "the matrix is the cause": "matrix",
    "fails, repeat unchecked": "unverified_unchecked",
    "fails twice, one thread unchecked": "unverified_unchecked",
    "first unchecked": "first_unchecked",
    "bad + first unchecked, repeat passes": "partly",          # A-3: the first test solve could not be done, so the run is not called correct although the repeat passed
    "bad twice with good test solves": "correct",
    "later system after the matrix verdict": "matrix",
}


def _kind_of(note: str) -> str:
    if not note:
        return "none"
    if "also on one thread" in note:
        return "matrix"
    if "results are correct" in note:
        return "correct"
    if "not fully verified" in note:
        return "partly"
    if "rare error" in note:
        return "unverified_unchecked" if "could not be done" in note else "unverified"
    return "first_unchecked" if "could not be done" in note else "?"


@pytest.mark.parametrize("patch", [ROOT / "installer" / "windows" / "patches" / PATCH_NAME, ROOT / "installer" / "linux" / "patches" / PATCH_NAME], ids=["windows", "linux"])
def test_the_verdict_function_of_the_patch_judges_the_bookkeeping_first_and_never_confirms_an_unchecked_repeat(tmp_path, patch):
    """`fea_judge` of the patch (the very text that is compiled into ccx) is compiled with stubs for the test solve and run through scenarios: (a) a violation of the bookkeeping
    of SPOOLES (error -2) is judged BEFORE the off-switches (the flag of an earlier matrix, the test limit 0) - the factors are repeated, never accepted; (b) a repeat whose test
    solve could not be done (eta < 0) is "used without a verdict", never "confirmed", and the note for the user does not call it correct. Skipped without a C compiler."""
    exe, env = _build_harness(tmp_path, patch)
    r = subprocess.run([str(exe)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0 and "ALL SCENARIOS AS EXPECTED" in r.stdout and "FAIL" not in r.stdout, r.stdout     # attempts, test solves, freed factors, the off flag of every scenario
    blocks = {}
    for part in r.stdout.split("---- ")[1:]:
        title, _, rest = part.partition("\n")
        blocks[title.strip()] = rest
    assert set(blocks) == set(JUDGE_NOTES)
    for title, kind in JUDGE_NOTES.items():
        chk = SpoolesCheck()
        for line in blocks[title].splitlines():
            if not line.startswith("[") and "ALL SCENARIOS" not in line:
                chk.feed(line.strip())
        assert _kind_of(spooles_check_note("eigen_modal", chk)) == kind, (title, blocks[title])
        if kind in ("unverified", "unverified_unchecked", "first_unchecked"):
            assert "confirmed" not in blocks[title].replace("NOT confirmed", ""), (title, blocks[title])           # no line that claims a confirmation
    assert "results are correct" not in spooles_check_note("j", SpoolesCheck(failed=1, bookkeeping=True))        # a claim needs the confirming line


@pytest.mark.parametrize("patch", [ROOT / "installer" / "windows" / "patches" / PATCH_NAME, ROOT / "installer" / "linux" / "patches" / PATCH_NAME], ids=["windows", "linux"])
def test_after_the_matrix_verdict_the_later_factorizations_run_on_one_thread(tmp_path, patch):
    """Review of 24. 11. 2026 (A-1): the verdict "the matrix" switches the check off (fea_check_off, static). A factorization on the threads is safe only while it is checked, so the
    conditions of spooles_factor and spooles_factor_rad carry `(!fea_check_off)`: the next system of the process runs on ONE thread. The harness runs two systems in one process
    with the very condition of the patch; a mutant with the old condition (threads whenever fea_attempt < 2) must make the scenario fail."""
    conds = _threaded_conditions(patch)
    assert len(conds) == 2 and all("(!fea_check_off)" in c for c in conds), conds
    exe, env = _build_harness(tmp_path / "real", patch)
    r = subprocess.run([str(exe)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0 and "[system 2: attempts=1 threaded factorizations=0 test solves=0 off_after=1]" in r.stdout, r.stdout
    mutant, env = _build_harness(tmp_path / "mutant", patch, cond="(num_cpus > 1)&&(fea_attempt < 2)")
    r = subprocess.run([str(mutant)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode != 0 and "SOME SCENARIOS FAILED" in r.stdout and "the later system must run once, on ONE thread" in r.stdout, r.stdout


@pytest.mark.parametrize("sub", ["windows", "linux"])
def test_the_check_marker_of_the_patch_is_the_one_the_program_wants(sub):
    """Review of 24. 11. 2026 (A-5): the value of the check marker is 2 (the first fea_judge printed 1); the printf of the patch, the build script and the program agree."""
    patch = ROOT / "installer" / sub / "patches" / PATCH_NAME
    text = patch.read_text(encoding="utf-8")
    added = [l for l in text[text.index("--- a/CalculiX"):].splitlines() if l.startswith("+") and "FEA_SPOOLES_FACTOR_CHECK=" in l]
    assert len(added) == 1 and runner.FACTOR_CHECK_MARKER.decode() in added[0] and runner.FACTOR_CHECK_MARKER_V1.decode() not in added[0], added
    assert runner.FACTOR_CHECK_MARKER == b"FEA_SPOOLES_FACTOR_CHECK=2" and runner.FACTOR_CHECK_MARKER not in runner.FACTOR_CHECK_MARKER_V1
    script = (ROOT / "installer" / sub / "build_ccx.sh").read_text(encoding="utf-8")
    assert runner.FACTOR_CHECK_MARKER.decode() in script and runner.FACTOR_CHECK_MARKER_V1.decode() not in script


# ================================================================= a binary with only the first marker
def test_a_binary_with_only_the_serial_solve_marker_gets_one_thread(tmp_path, monkeypatch):
    """The first candidate of F17 stage A (serial solve, factorization on threads, no check) can lose an update silently: its marker is NOT enough for threads."""
    monkeypatch.delenv("FEA_SPOOLES_THREADS", raising=False)
    only_serial = tmp_path / "ccx_serial_only.exe"
    only_serial.write_bytes(b"MZ" + b"This is Version 2.23\n" + SERIAL_SOLVE_MARKER + b" (multi-threaded SPOOLES factorization, serial triangular solve)\n")
    (tmp_path / "a.inp").write_text("*STEP\n*STATIC, SOLVER=SPOOLES\n*END STEP\n")
    assert not runner.spooles_threads_safe(only_serial) and runner.one_equation_thread(tmp_path / "a.inp", only_serial)
    seen = {}

    def popen(cmd, **kw):
        seen["env"] = kw["env"]
        return _Proc("Job finished\n")
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.dllguard, "check_workdir", lambda *a, **k: None)
    runner.run_ccx(tmp_path, "a", exe=only_serial, threads=6)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "1" and seen["env"]["OMP_NUM_THREADS"] == "6"
    both = tmp_path / "ccx_both.exe"
    both.write_bytes(only_serial.read_bytes() + runner.FACTOR_CHECK_MARKER + b"\n" + runner.ORDERED_MARKER + b"\n")
    runner.run_ccx(tmp_path, "a", exe=both, threads=6)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "6"
    # A-5: a binary with the first fea_judge (marker 1: the systems after a verdict "the matrix" ran unchecked on the threads) is not recognised, although it has the ordered SPOOLES
    first_judge = tmp_path / "ccx_first_judge.exe"
    first_judge.write_bytes(only_serial.read_bytes() + runner.FACTOR_CHECK_MARKER_V1 + b"\n" + runner.ORDERED_MARKER + b"\n")
    assert not runner.spooles_threads_safe(first_judge) and runner.one_equation_thread(tmp_path / "a.inp", first_judge)
    runner.run_ccx(tmp_path, "a", exe=first_judge, threads=6)
    assert seen["env"]["CCX_NPROC_EQUATION_SOLVER"] == "1"


# ================================================================= the built binary (slow)
def _brick_modal_deck(nx=24, ny=5, nz=5) -> str:
    """A cantilever of C3D8 bricks, six natural frequencies (SPOOLES): about 3 000 equations - enough fronts for four threads."""
    nid = {}
    lines = ["*NODE"]
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nid[i, j, k] = len(nid) + 1
                lines.append(f"{nid[i, j, k]}, {float(i)}, {float(j)}, {float(k)}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                e += 1
                n = [nid[i, j, k], nid[i + 1, j, k], nid[i + 1, j + 1, k], nid[i, j + 1, k],
                     nid[i, j, k + 1], nid[i + 1, j, k + 1], nid[i + 1, j + 1, k + 1], nid[i, j + 1, k + 1]]
                lines.append(f"{e}, " + ", ".join(map(str, n)))
    fixed = [nid[0, j, k] for k in range(nz + 1) for j in range(ny + 1)]
    lines += ["*NSET, NSET=FIX"] + [", ".join(map(str, fixed[i:i + 8])) for i in range(0, len(fixed), 8)]
    lines += ["*MATERIAL, NAME=M1", "*ELASTIC", "210000., 0.3", "*DENSITY", "7.85E-9", "*SOLID SECTION, ELSET=EALL, MATERIAL=M1",
              "*BOUNDARY", "FIX, 1, 3", "*STEP", "*FREQUENCY, SOLVER=SPOOLES", "6", "*NODE FILE", "U", "*END STEP"]
    return "\n".join(lines) + "\n"


def _eigenvalues(dat: Path) -> list[float]:
    text = dat.read_text(errors="replace")
    m = re.search(r"E I G E N V A L U E   O U T P U T(.*?)(?=P A R T I C I P A T I O N|\Z)", text, re.S)
    assert m, "no eigenvalue table in the .dat"
    out = []
    for line in m.group(1).splitlines():
        f = line.split()
        if len(f) >= 4 and f[0].isdigit() and re.fullmatch(r"[-+]?\d\.\d+E[-+]\d+", f[1]):
            out.append(float(f[1]))
    return out


def _marked_exe() -> Path | None:
    cand = os.environ.get("FEA_CCX")
    exe = Path(cand) if cand else config.CCX_EXE
    try:
        return exe if exe.is_file() and runner.spooles_threads_safe(exe) and runner.ORDERED_MARKER in exe.read_bytes() else None
    except OSError:
        return None


@pytest.mark.slow
def test_the_built_ccx_with_the_limit_forced_tiny_repeats_reports_and_stays_correct(tmp_path, monkeypatch):
    """Every check fails when the limit is 1e-30: the multi-threaded factorization is repeated with the threads, then on ONE thread, which fails the check as well
    (the verdict "the matrix, not the threads"); the eigenvalues are those of the unchecked run, and the program reports it in a note. Needs a ccx with the
    markers (FEA_CCX = the binary of installer/windows/build_ccx.ps1 or installer/linux/build_ccx.sh)."""
    exe = _marked_exe()
    if exe is None:
        pytest.skip("no ccx with the markers of the checked SPOOLES (set FEA_CCX to the built binary)")
    monkeypatch.delenv("FEA_SPOOLES_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "job.inp").write_text(_brick_modal_deck())
    (tmp_path / "forced").mkdir()
    (tmp_path / "forced" / "job.inp").write_text(_brick_modal_deck())
    monkeypatch.setenv("FEA_DEV", "1")
    monkeypatch.setenv("FEA_SPOOLES_CHECK_TOL", "0")                                              # the check off: the reference
    ref = runner.run_ccx(tmp_path / "ref", "job", exe=exe, threads=4)
    assert ref.finished and not ref.errors and not ref.warnings
    monkeypatch.setenv("FEA_SPOOLES_CHECK_TOL", "1e-30")                                          # every check fails
    run = runner.run_ccx(tmp_path / "forced", "job", exe=exe, threads=4)
    assert run.finished and not run.errors
    log = (tmp_path / "forced" / "job.log").read_text(encoding="utf-8", errors="replace")
    assert log.count("failed the check of its result") == 2 and "factorized again with 4 threads" in log and "factorized again with ONE thread" in log
    assert "the matrix, not the threads" in log and "check is switched off for the rest of this run" in log
    assert len(run.warnings) == 1 and "also on one thread" in run.warnings[0]
    a, b = _eigenvalues(tmp_path / "ref" / "job.dat"), _eigenvalues(tmp_path / "forced" / "job.dat")
    assert len(a) == len(b) == 6 and all(abs(x - y) <= 1e-9 * abs(x) for x, y in zip(a, b)), (a, b)
    monkeypatch.delenv("FEA_DEV")                                                                 # a user's environment: the switch is removed, the check stays on
    third = runner.run_ccx(tmp_path / "ref", "job", exe=exe, threads=4)
    assert third.finished and not third.warnings and "failed the check" not in (tmp_path / "ref" / "job.log").read_text(encoding="utf-8", errors="replace")


@pytest.mark.slow
def test_the_built_ccx_prints_all_markers_and_a_one_thread_run_needs_no_check(tmp_path, monkeypatch):
    exe = _marked_exe()
    if exe is None:
        pytest.skip("no ccx with the markers of the checked SPOOLES (set FEA_CCX to the built binary)")
    out = subprocess.run([str(exe), "-v"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    for marker in (runner.SERIAL_SOLVE_MARKER, runner.FACTOR_CHECK_MARKER, runner.ORDERED_MARKER):
        assert marker.decode() in out and marker in exe.read_bytes()

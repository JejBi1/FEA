"""Release / test infrastructure: hashed version locks, the platform check of the installer, the package anonymization scan and
dirty-tree detection, the verification tiers and the release check (without network, uv, Gmsh and CalculiX)."""
from __future__ import annotations

import importlib
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

upgrade = importlib.import_module("upgrade")


# ------------------------------------------------------------------ locks with digests
HASHED = """# comment
numpy==1.26.4 \\
    --hash=sha256:aaa \\
    --hash=sha256:bbb
PySide6_Essentials==6.11.2 \\
    --hash=sha256:ccc
plain==1.0
"""


def test_parse_lock_hashed_and_plain():
    lock = upgrade.parse_lock(HASHED)
    assert lock["numpy"] == ("numpy==1.26.4", ["sha256:aaa", "sha256:bbb"])
    assert lock["pyside6-essentials"] == ("PySide6_Essentials==6.11.2", ["sha256:ccc"])
    assert lock["plain"] == ("plain==1.0", [])


@pytest.mark.parametrize("plat", ["windows-x86_64", "linux-x86_64"])
def test_locks_have_digests_for_every_package(plat):
    lock = upgrade.parse_lock((ROOT / "installer" / "locks" / f"{plat}.txt").read_text(encoding="utf-8"))
    missing = [req for req, h in lock.values() if not h]
    assert not missing, f"{plat}: packages without SHA-256 digests: {missing} (python installer/make_locks.py)"
    assert all(h.startswith("sha256:") and len(h) == 71 for _, hs in lock.values() for h in hs)


def test_hashed_requirements_all_or_nothing():
    """--no-hashes (strict=False): the old all-or-nothing behaviour."""
    fs = importlib.import_module("fea_setup")
    lock = upgrade.parse_lock(HASHED)
    text, hashed = fs.hashed_requirements(["numpy==1.26.4", "PySide6_Essentials==6.11.2"], lock)
    assert hashed and text.splitlines() == ["numpy==1.26.4 --hash=sha256:aaa --hash=sha256:bbb",
                                            "PySide6_Essentials==6.11.2 --hash=sha256:ccc"]
    text, hashed = fs.hashed_requirements(["numpy==1.26.4", "plain==1.0"], lock, strict=False)    # one without digests → none
    assert not hashed and "--hash" not in text and "plain==1.0" in text
    text, hashed = fs.hashed_requirements(["numpy==1.26.5"], lock, strict=False)                   # another version than locked
    assert not hashed and text.strip() == "numpy==1.26.5"


@pytest.mark.parametrize("extra", [["evilpkg==0.1"], ["plain==1.0"]])
def test_hashed_requirements_strict_refuses_unverified(extra):
    """Default (the installer without --no-hashes): one package outside the lock / without digests / in another version
    stops the installation instead of switching the verification off for all packages."""
    fs = importlib.import_module("fea_setup")
    lock = upgrade.parse_lock(HASHED)
    with pytest.raises(fs.HashError) as e:
        fs.hashed_requirements(["numpy==1.26.4"] + extra, lock)
    assert extra[0] in str(e.value) and "--no-hashes" in str(e.value) and "numpy" not in str(e.value)
    with pytest.raises(fs.HashError):
        fs.hashed_requirements(["numpy==1.26.5"], lock)


def test_sync_packages_compiles_without_user_config_and_requires_hashes(tmp_path, monkeypatch):
    """uv pip compile ignores uv.toml (--no-config, as make_locks.py); the sync always runs with --require-hashes."""
    fs = importlib.import_module("fea_setup")
    lock_dir = tmp_path / "installer" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / f"{fs.PLAT}.txt").write_text(HASHED, encoding="utf-8")
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    monkeypatch.setattr(fs, "MODS", {"core": {"pip": ["numpy"]}})
    monkeypatch.setattr(fs, "offline_wheels", lambda: False)
    monkeypatch.setattr(fs, "installed_records", lambda: {})
    monkeypatch.setattr(fs, "info", lambda *_: None)
    calls = []

    def uv_pip(*a, **k):
        calls.append(a)
        if a[0] == "compile":
            Path(a[a.index("-o") + 1]).write_text("numpy==1.26.4\n", encoding="utf-8")
        else:
            calls.append(Path(a[1]).read_text(encoding="utf-8"))
    monkeypatch.setattr(fs, "uv_pip", uv_pip)
    fs.sync_packages(["core"])
    assert "--no-config" in calls[0]
    assert calls[1][0] == "sync" and "--require-hashes" in calls[1]
    assert calls[2].strip() == "numpy==1.26.4 --hash=sha256:aaa --hash=sha256:bbb"


def test_repair_shared_files_reads_hashed_resolved(tmp_path, monkeypatch):
    fs = importlib.import_module("fea_setup")
    resolved = tmp_path / "resolved.txt"
    resolved.write_text("pyside6-essentials==6.11.2 --hash=sha256:ccc\n", encoding="utf-8")
    before = {"pyside6_addons": {"pyside6/__init__.py"}, "pyside6_essentials": {"pyside6/__init__.py", "pyside6/qtcore.pyd"}}
    after = {"pyside6_essentials": {"pyside6/__init__.py", "pyside6/qtcore.pyd"}}
    monkeypatch.setattr(fs, "installed_records", lambda: after)
    monkeypatch.setattr(fs, "site_packages", lambda: tmp_path)
    calls = []

    def uv_pip(*a, **k):
        calls.append(a)
        calls.append(Path(a[-1]).read_text(encoding="utf-8"))      # the pins file is deleted after the call
    monkeypatch.setattr(fs, "uv_pip", uv_pip)
    monkeypatch.setattr(fs, "info", lambda *_: None)
    fs.repair_shared_files(before, resolved)
    # the digests of the resolved file are carried over (previously stripped = reinstall without verification)
    assert calls[0][:5] == ("install", "--reinstall", "--no-deps", "--require-hashes", "-r")
    assert calls[1].strip() == "pyside6-essentials==6.11.2 --hash=sha256:ccc"


def test_restore_without_digest_refused_unless_no_hashes(tmp_path, monkeypatch):
    """A pin from the installed version without a digest in the lock: an error; with --no-hashes the old plain reinstall."""
    fs = importlib.import_module("fea_setup")
    monkeypatch.setattr(fs, "locked_hashes", lambda: upgrade.parse_lock(HASHED))
    calls = []
    monkeypatch.setattr(fs, "uv_pip", lambda *a, **k: calls.append(a))
    with pytest.raises(fs.HashError):
        fs.install_pins(["PySide6_Essentials==6.10.0"])
    assert calls == []
    fs.install_pins(["PySide6_Essentials==6.11.2"])                     # digest from the lock
    assert "--require-hashes" in calls[-1]
    monkeypatch.setattr(fs, "REQUIRE_HASHES", False)
    fs.install_pins(["PySide6_Essentials==6.10.0"])
    assert calls[-1] == ("install", "--reinstall", "--no-deps", "PySide6_Essentials==6.10.0")


def test_no_hashes_switch(monkeypatch):
    fs = importlib.import_module("fea_setup")
    seen = {}
    monkeypatch.setattr(fs, "check_platform", lambda: None)
    monkeypatch.setattr(fs, "cmd_remove", lambda a: seen.setdefault("strict", fs.REQUIRE_HASHES) and 0)
    monkeypatch.setattr(fs, "LOG", None)
    try:
        fs.main(["remove", "beams", "--no-hashes"])
        assert seen["strict"] is False
    finally:
        fs.REQUIRE_HASHES = True


# ------------------------------------------------------------------ installer: platform
def test_check_platform_unsupported(monkeypatch, capsys):
    fs = importlib.import_module("fea_setup")
    monkeypatch.setattr(fs, "PLAT", "windows-arm64")
    monkeypatch.setattr(fs, "LOG", None)
    with pytest.raises(SystemExit) as e:
        fs.check_platform()
    assert e.value.code == 3
    assert "windows-arm64" in capsys.readouterr().err
    monkeypatch.setattr(fs, "PLAT", "windows-x86_64" if fs.IS_WIN else "linux-x86_64")
    fs.check_platform()                                                       # the supported one passes


def test_main_checks_platform_before_install(monkeypatch):
    fs = importlib.import_module("fea_setup")
    monkeypatch.setattr(fs, "PLAT", "linux-aarch64")
    monkeypatch.setattr(fs, "LOG", None)
    monkeypatch.setattr(fs, "cmd_install", lambda args, update=False: pytest.fail("install must not start"))
    with pytest.raises(SystemExit) as e:
        fs.main(["install", "core", "--no-shortcuts", "--quiet"])
    assert e.value.code == 3


def test_iss_requires_app_version():
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    block = iss.split("#ifndef AppVersion", 1)[1].split("#endif", 1)[0]
    assert "#error" in block and "#define AppVersion" not in block


# ------------------------------------------------------------------ make_package: dirty tree, anonymization
def test_dirty_files_parses_porcelain(monkeypatch):
    mp = importlib.import_module("make_package")
    out = b"".join(x + b"\0" for x in (b" M fea_core/a.py", b"?? new.py", b"R  new_name.py", b"old_name.py",
                                        b" M scratch/x.py", b" M CLAUDE.md", b"?? examples/ukazka.fea"))
    monkeypatch.setattr(mp.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out, returncode=0))
    assert mp.dirty_files() == ["fea_core/a.py", "new.py", "new_name.py"]


def test_scan_private_paths_also_in_zip_members(tmp_path):
    mp = importlib.import_module("make_package")
    (tmp_path / "ok.txt").write_text("see /home/conda/feedstock and C:/Program Files/x", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("log: /home/someone/fea/x.step", encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("study.json", json.dumps({"geometry": "C:/Users/someone/Desktop/part.step"}))
    (tmp_path / "proj.fea").write_bytes(buf.getvalue())
    hits = mp.scan_private_paths(tmp_path, ["ok.txt", "bad.txt", "proj.fea"])
    where = sorted(w for w, _ in hits)
    assert where == ["bad.txt", "proj.fea!study.json"]


@pytest.mark.parametrize("text,bad", [
    (rb"C:\Users\someone\proj\x.step", True), (rb"c:\users\someone", True), (rb"D:/USERS/someone/x", True),
    (rb'{"g": "C:\\Users\\someone\\x.step"}', True), (b"/Users/someone/x", True), (b"/home/someone/x", True),
    (b"/home/conda/feedstock/x", False), (rb"C:\Program Files\x", False), (b"see users guide", False),
])
def test_private_path_windows_forms(text, bad):
    """Backslash paths (Windows, also doubled in JSON), any case of the drive / Users – previously only the forward-slash form of a drive path matched."""
    mp = importlib.import_module("make_package")
    assert bool(mp.PRIVATE_PATH.search(text)) is bad, text


def test_anonymization_check_with_denylist_file(tmp_path, monkeypatch):
    """Fallback scanner (no github_export.py): denylist lines folded (case, diacritics), also inside zip members."""
    mp = importlib.import_module("make_package")
    repo = tmp_path / "repo"
    (repo / ".git" / "info").mkdir(parents=True)
    (repo / ".git" / "info" / "github-denylist.txt").write_text("# x\nJméno Příjmení\nre:secret\\s*corp\n", encoding="utf-8")
    monkeypatch.setattr(mp, "git_common_root", lambda: repo)
    monkeypatch.setattr(mp, "load_export_scanner", lambda: None)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.py").write_text("# author: JMENO prijmeni\n", encoding="utf-8")
    (stage / "b.md").write_text("nothing here", encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.xml", "Secret Corp")
    (stage / "c.docx").write_bytes(buf.getvalue())
    found = mp.anonymization_check(stage)
    assert any(f.startswith("a.py") for f in found)
    assert any(f.startswith("c.docx!doc.xml") for f in found)
    assert not any(f.startswith("b.md") for f in found)
    assert all("Jméno" not in f and "Secret" not in f for f in found)   # the denied term itself is not printed in full


def test_anonymization_check_without_any_denylist(tmp_path, monkeypatch, capsys):
    mp = importlib.import_module("make_package")
    monkeypatch.setattr(mp, "git_common_root", lambda: tmp_path)
    monkeypatch.setattr(mp, "load_export_scanner", lambda: None)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    assert mp.anonymization_check(tmp_path) == []
    assert "no local denylist" in capsys.readouterr().out


def test_make_package_refuses_dirty_tree(monkeypatch, capsys):
    mp = importlib.import_module("make_package")
    monkeypatch.setattr(mp, "dirty_files", lambda: ["fea_core/a.py"])
    monkeypatch.setattr(mp, "file_list", lambda ref: pytest.fail("nothing must be staged"))
    monkeypatch.setattr(sys, "argv", ["make_package.py", "--check-only"])
    assert mp.main() == 2
    assert "--allow-dirty" in capsys.readouterr().err


def test_make_package_shares_helpers_with_upgrade():
    mp = importlib.import_module("make_package")
    fs = importlib.import_module("fea_setup")
    assert mp.sha is upgrade.sha256_file and fs.sha256_file is upgrade.sha256_file
    assert mp.version() == upgrade.program_version(ROOT)
    from fea_core import __version__
    assert mp.version() == __version__


# ------------------------------------------------------------------ verification tiers and the release check
def test_tiers_cover_known_cases():
    from verification import cases as C
    from verification import tiers
    assert tiers.VERY_SLOW <= set(C.CASES) and tiers.NEEDS_CHRONO <= tiers.VERY_SLOW
    assert set(tiers.RELEASE_CASES) <= set(C.CASES)
    assert not set(tiers.RELEASE_CASES) & tiers.VERY_SLOW
    assert tiers.tier("KK4") == "very_slow" and tiers.tier("V1") == "slow"


def test_release_check_summary_and_skip(tmp_path, monkeypatch):
    rc = importlib.import_module("verification.release_check")
    monkeypatch.setattr(rc, "OUT", tmp_path)
    monkeypatch.setattr(rc, "LOGS", tmp_path / "logs")
    ran = []
    monkeypatch.setattr(rc, "FUNCS", {s: (lambda a, _s=s: (ran.append(_s) or (_s != "help", f"{_s} detail")))
                                      for s in rc.STAGES})
    code = rc.main(["--skip", "gui", "cases"])
    assert code == 1 and ran == ["pytest", "i18n", "help"]                   # package only with --package
    data = json.loads((tmp_path / "release_check.json").read_text(encoding="utf-8"))
    status = {r["stage"]: r["status"] for r in data["stages"]}
    assert status == {"pytest": "pass", "i18n": "pass", "help": "fail", "gui": "skipped", "cases": "skipped",
                      "package": "skipped"}
    assert not data["ok"] and "FAILED" in (tmp_path / "release_check.txt").read_text(encoding="utf-8")
    assert "<table>" in (tmp_path / "release_check.html").read_text(encoding="utf-8")
    ran.clear()
    assert rc.main(["--only", "pytest", "--package"]) == 0 and ran == ["pytest"]
    assert rc.main(["--cases", "NOPE"]) == 2


def test_gui_run_all_lists_smoke_tests():
    sys.path.insert(0, str(ROOT / "tests" / "gui"))
    ra = importlib.import_module("run_all")
    names = [p.name for p in ra.scripts()]
    assert "gui_smoke.py" in names and all(n.startswith("gui_") and n.endswith("smoke.py") for n in names)
    assert "gui_jit_smoke.py" in names                                  # first-use notice of the numba kernels (release_check "gui" stage)
    on_disk = {p.name for p in (ROOT / "tests" / "gui").glob("gui_*.py")}
    assert on_disk == set(names), f"GUI test scripts that run_all does not run (name them gui_*_smoke.py): {sorted(on_disk - set(names))}"
    assert [p.name for p in ra.scripts(only=["popout"])] == ["gui_popout_smoke.py"]
    assert "gui_popout_smoke.py" not in [p.name for p in ra.scripts(skip=["popout"])]
    assert ra.short(ROOT / "tests" / "gui" / "gui_table_dock_smoke.py") == "table_dock"


# ------------------------------------------------------------------ the synthetic home paths of the scanner tests: exact strings per file, nothing blanket
# Everything below builds its paths at run time (join of the parts): this file lists its own fixtures in make_package.SYNTHETIC_HOME_PATHS, a new literal
# home path here would have to be listed there too.
def _path(*parts, sep="/"):
    return sep.join(parts)


def test_the_listed_fixtures_are_exactly_what_the_scanner_test_files_contain():
    """The lists of SYNTHETIC_HOME_PATHS cover the repository files as they are (no finding left) and have no dead entry (a fixture removed from a test
    must leave the list)."""
    mp = importlib.import_module("make_package")
    listed = sorted(mp.SYNTHETIC_HOME_PATHS)
    assert mp.scan_private_paths(ROOT, listed) == []
    for rel in listed:
        if rel == "installer/make_package.py":                                  # names all of them: checked below as the union
            continue
        found = {m.group(0).decode("latin-1") for m in mp.PRIVATE_PATH.finditer((ROOT / rel).read_bytes())}
        assert found == set(mp.SYNTHETIC_HOME_PATHS[rel]), rel
    assert set(mp.SYNTHETIC_HOME_PATHS["installer/make_package.py"]) == set().union(*(v for k, v in mp.SYNTHETIC_HOME_PATHS.items() if k != "installer/make_package.py"))


@pytest.mark.parametrize("rel,fixture", [
    ("tests/test_security.py", _path("", "home", "u", "x.fea")),
    ("tests/test_modules.py", _path("", "home", "builder", "fea", "x.f")),
    ("tests/test_direct_scrub.py", _path("C:", "Users", "Bob", "fea", "x.npy")),
    ("tests/test_direct_scrub.py", _path("C:", "Users", "Bob", "fea", sep="\\" * 2)),
])
def test_the_exact_fixture_of_a_listed_file_passes_but_a_real_path_next_to_it_stops_the_build(tmp_path, rel, fixture):
    mp = importlib.import_module("make_package")
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(f"x = {fixture}\n", encoding="utf-8")
    assert mp.scan_private_paths(tmp_path, [rel]) == []
    real = _path("C:", "Users", "Somebodyreal", "Desktop", "part.step")
    (tmp_path / rel).write_text(f"x = {fixture}\ny = {real}\n", encoding="utf-8")
    assert [w for w, _ in mp.scan_private_paths(tmp_path, [rel])] == [rel]


@pytest.mark.parametrize("rel,path", [
    ("tests/test_security.py", _path("", "home", "u2", "x.fea")),                # a longer name that only starts like the fixture
    ("tests/test_security.py", _path("", "home", "someone", "x.fea")),          # another synthetic name – not listed for THIS file
    ("tests/test_direct_scrub.py", _path("C:", "Users", "Bobby", "x")),
    ("tests/test_direct_scrub.py", _path("", "home", "bob2", "x")),
    ("tests/test_modules.py", _path("", "mnt", "c", "x")),
])
def test_only_the_exact_strings_are_waived(tmp_path, rel, path):
    mp = importlib.import_module("make_package")
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(f"x = {path}\n", encoding="utf-8")
    assert [w for w, _ in mp.scan_private_paths(tmp_path, [rel])] == [rel]


def test_a_fixture_is_not_waived_in_a_file_that_is_not_listed(tmp_path):
    mp = importlib.import_module("make_package")
    (tmp_path / "fea_core").mkdir()
    fixture = _path("", "home", "u", "x.fea")
    (tmp_path / "fea_core" / "solve.py").write_text(f"x = {fixture}\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_modules.py").write_text(f"x = {fixture}\n", encoding="utf-8")                # listed, but for another fixture
    assert sorted(w for w, _ in mp.scan_private_paths(tmp_path, ["fea_core/solve.py", "tests/test_modules.py"])) == ["fea_core/solve.py", "tests/test_modules.py"]


class _FakeExport:
    """The GitHub-export scanner in miniature: lower-cased text × regular expressions (a personal name and the generic Windows-profile pattern of the
    real denylist); the same interface as scratch/github_export/github_export.py that make_package uses."""
    PATS = [re.compile(r"c:[\\/]+users[\\/]+(?!x[\\/]|<|%|\{|\$)"), re.compile("somebodyreal")]
    mask = staticmethod(lambda s: s[:4] + "…")

    @staticmethod
    def build_denylist():
        return _FakeExport.PATS

    @staticmethod
    def load_allow():
        return []

    @staticmethod
    def scan_tree(base, files, pats, allow):
        out = []
        for rel in files:
            text = (base / rel).read_bytes().decode("utf-8", "ignore").lower()
            for p in pats:
                out += [(rel, m.group(0), "") for m in p.finditer(text)]
        return out


def _scanner_test_file(stage, text):
    (stage / "tests").mkdir(parents=True, exist_ok=True)
    (stage / "tests" / "test_release_infra.py").write_text(text, encoding="utf-8")


def test_the_export_scanner_sees_the_listed_file_without_its_fixtures_only(tmp_path, monkeypatch):
    """The generic home-path patterns of the denylist also match the fixtures: the file reaches the scanner with exactly those strings replaced – a personal
    name or a real profile folder in it is a finding (the staged file itself is not changed)."""
    import concurrent.futures
    mp = importlib.import_module("make_package")
    monkeypatch.setattr(mp, "load_export_scanner", lambda: _FakeExport)
    monkeypatch.setattr(mp, "_SCANNER", (_FakeExport, _FakeExport.PATS, []))
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)      # the fake scanner lives in this process
    stage = tmp_path / "stage"
    fixtures = f"a = {_path('C:', 'Users', 'someone', 'proj')}\nb = {_path('', 'home', 'someone', 'fea')}\n"
    _scanner_test_file(stage, fixtures)
    assert mp.anonymization_check(stage) == []
    assert (stage / "tests" / "test_release_infra.py").read_text(encoding="utf-8") == fixtures
    real = f"c = {_path('C:', 'Users', 'Somebodyreal', 'proj')}\n"
    _scanner_test_file(stage, fixtures + real)
    found = mp.anonymization_check(stage)
    assert any("absolute home path" in f for f in found) and any("denylist" in f for f in found)
    _scanner_test_file(stage, fixtures + "d = 'somebodyreal'\n")                       # a personal name next to the fixtures: the denylist still sees it
    assert any("denylist" in f for f in mp.anonymization_check(stage))


def test_the_fallback_scanner_sees_the_listed_file_without_its_fixtures_only(tmp_path, monkeypatch):
    mp = importlib.import_module("make_package")
    repo = tmp_path / "repo"
    (repo / ".git" / "info").mkdir(parents=True)
    (repo / ".git" / "info" / "github-denylist.txt").write_text("re:c:[\\\\/]+users[\\\\/]+\nsomebodyreal\n", encoding="utf-8")
    monkeypatch.setattr(mp, "git_common_root", lambda: repo)
    monkeypatch.setattr(mp, "load_export_scanner", lambda: None)
    stage = tmp_path / "stage"
    fixtures = f"a = {_path('C:', 'Users', 'someone', 'proj')}\n"
    _scanner_test_file(stage, fixtures)
    assert mp.anonymization_check(stage) == []
    _scanner_test_file(stage, fixtures + f"c = {_path('C:', 'Users', 'Somebodyreal', 'proj')}\n")
    found = mp.anonymization_check(stage)
    assert any("absolute home path" in f for f in found) and any("denylist" in f for f in found)

"""pytest setup shared by all tests.

The MCP server exposes only the `core` tool group by default (fea_core/mcp/groups.py); the tests use every tool, so they start
the server with all groups. Tests of the default / of the groups pass `groups_spec=` explicitly.

Time limits (only when pytest-timeout is installed – module verify): a test without its own `@pytest.mark.timeout` gets
120 s (fast), 600 s (`slow`) or 7200 s (`very_slow`); overrides FEA_TEST_TIMEOUT_FAST / _SLOW / _VERY_SLOW (seconds, 0 = off).
An explicit `--timeout` on the command line wins over these defaults.

Installations with fewer program modules (CI with the `minimal` profile, fea_core/modules.json): a test that fails only because
an OPTIONAL module of the program is not installed (fea_core.modules.ModuleMissing, or an import of a package of such a module)
is reported as skipped. Real import errors (a package of an installed module, a typo) still fail.
"""
import os

import pytest

os.environ.setdefault("FEA_MCP_GROUPS", "all")
os.environ.setdefault("FEA_I18N_CACHE", "0")      # persisted start-up translations (fea_core.i18n) off – tests see the catalogue

TIMEOUTS = {"fast": 120, "slow": 600, "very_slow": 7200}


def _limit(tier: str) -> float:
    v = os.environ.get(f"FEA_TEST_TIMEOUT_{tier.upper()}", "")
    try:
        return float(v) if v.strip() else float(TIMEOUTS[tier])
    except ValueError:
        return float(TIMEOUTS[tier])


def pytest_collection_modifyitems(config, items):
    if not config.pluginmanager.hasplugin("timeout"):
        return
    if config.getoption("timeout", None):                   # `pytest --timeout N` = one limit for all
        return
    for item in items:
        if item.get_closest_marker("timeout") is not None:
            continue
        tier = ("very_slow" if item.get_closest_marker("very_slow") else
                "slow" if item.get_closest_marker("slow") else "fast")
        limit = _limit(tier)
        if limit > 0:
            item.add_marker(pytest.mark.timeout(limit))


os.environ.setdefault("FEA_MCP_NO_GUI", "1")      # MCP tools never run in an FEA Studio window open on this machine (fea_core.ipc)
os.environ.setdefault("FEA_MODAL_MIN_WORK", "0")      # natural frequencies: the tests run the direct path on tiny models (production: CalculiX below `modal_run.min_work`)


@pytest.fixture(autouse=True)
def _private_user_dirs_outside_windows(tmp_path_factory, monkeypatch):
    """Outside Windows the user settings / data live in XDG folders (fea_core.config.user_config_dir); the tests isolate
    themselves by APPDATA / LOCALAPPDATA (Windows) → each test gets fresh XDG folders, the real ~/.config stays untouched."""
    if os.name != "nt":
        d = tmp_path_factory.mktemp("xdg")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(d / "config"))
        monkeypatch.setenv("XDG_DATA_HOME", str(d / "data"))


_OPTIONAL: dict[str, str] | None = None             # import name of a package → optional module that is NOT installed


def _optional_packages() -> dict[str, str]:
    global _OPTIONAL
    if _OPTIONAL is None:
        _OPTIONAL = {}
        try:
            from fea_core import modules
            for key, m in modules.all_modules().items():
                if m.required or modules.available(key):
                    continue
                names = [i.split(".")[0] for i in m.imports]
                names += [p.split("==")[0].split("[")[0].strip().lower().replace("-", "_") for p in m.pip + m.pip_nodeps]
                for n in names:
                    _OPTIONAL.setdefault(n.lower(), key)
        except Exception:  # noqa: BLE001 – without the table nothing is converted
            pass
    return _OPTIONAL


def missing_optional(exc: BaseException | None) -> str | None:
    """Key of the optional program module whose absence caused `exc` (None = a real error)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "ModuleMissing" and getattr(exc, "key", None):
            from fea_core import modules
            return exc.key if not modules.available(exc.key) else None
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            key = _optional_packages().get(exc.name.split(".")[0].lower())
            if key:
                return key
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        key = missing_optional(call.excinfo.value)
        if key:
            rep.outcome = "skipped"
            rep.longrepr = (str(item.path), (item.location[1] or 0) + 1,
                            f"Skipped: optional program module '{key}' is not installed")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """A test module that imports an optional package at its top (numba, PySide6) is SKIPPED on an installation without that module (profile minimal), as a test
    that fails for the same reason is (above), instead of breaking the collection."""
    outcome = yield
    rep = outcome.get_result()
    if not rep.failed:
        return
    import re
    text = str(rep.longrepr)
    key = None
    m = re.search(r"ModuleMissing: .*?\((\w+)\)\. Install it", text)
    if m:
        from fea_core import modules
        key = m.group(1) if not modules.available(m.group(1)) else None
    else:
        m = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", text)
        if m:
            key = _optional_packages().get(m.group(1).split(".")[0].lower())
    if key:
        rep.outcome = "skipped"
        rep.longrepr = (str(collector.path), 0, f"Skipped: optional program module '{key}' is not installed")


@pytest.fixture(autouse=True)
def _fresh_state_cache(monkeypatch):
    """The state cache (fea_core.state_cache: the modal basis / unit states of a study) lives in the process: a test must not find the entries of an earlier one.
    Its free-space rule (MIN_FREE_GB of the work volume) is lifted: a CI runner has less than 10 GB free and the cache would silently store nothing."""
    from fea_core import state_cache
    monkeypatch.setattr(state_cache, "MIN_FREE_GB", 0.0)
    state_cache.clear()
    yield
    state_cache.clear()

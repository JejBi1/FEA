"""installer: the OpenMP runtime for numba on a Linux system without libgomp1 (fea_setup.omp_runtime_local).

The logic of the installer step is tested here (mocked probes and a real subprocess with stub modules); that the real dynamic loader resolves the copy
through the RPATH of the interpreter is tested in tests/test_install_omp_linux.py (Linux only)."""
import importlib
import sys
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))


@pytest.fixture
def env(monkeypatch, tmp_path):
    fs = importlib.import_module("fea_setup")
    home = tmp_path / "tools" / "python" / "cpython-x"
    (home / "lib").mkdir(parents=True)
    ccx_lib = tmp_path / "tools" / "calculix" / "ccx_2.23_linux" / "lib"
    ccx_lib.mkdir(parents=True)
    (ccx_lib / "libgomp.so.1").write_bytes(b"gomp")
    messages = []
    monkeypatch.setattr(fs, "IS_WIN", False)
    monkeypatch.setattr(fs, "VPY", Path(sys.executable))
    monkeypatch.setattr(fs, "VENV", tmp_path / ".venv")
    monkeypatch.setattr(fs, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(fs, "python_home", lambda: home)
    monkeypatch.setattr(fs, "warn", lambda m: messages.append(("warn", m)))
    monkeypatch.setattr(fs, "ok", lambda m: messages.append(("ok", m)))
    copied = home / "lib" / fs.OMP_NEEDED
    return types.SimpleNamespace(fs=fs, mp=monkeypatch, messages=messages, copied=copied, ccx_lib=ccx_lib / "libgomp.so.1", tmp=tmp_path)


def kinds(env):
    return [k for k, _ in env.messages]


def test_the_runtime_of_the_ccx_package_is_copied_next_to_python(env):
    env.mp.setattr(env.fs, "omp_probe", lambda: (True, "") if env.copied.exists() else (False, env.fs.OMP_RUNTIME))        # the layer loads once the file is there
    env.fs.omp_runtime_local(["core", "accel"])
    assert env.copied.read_bytes() == b"gomp" and env.copied.name == "libgomp.so.1.0.0"
    assert kinds(env) == ["ok"]


def test_nothing_is_copied_when_the_system_has_libgomp(env):
    env.mp.setattr(env.fs, "omp_probe", lambda: (True, ""))
    env.fs.omp_runtime_local(["core", "accel"])
    assert not env.copied.exists() and env.messages == []


def test_a_copy_that_does_not_help_is_removed_and_reported(env):
    env.mp.setattr(env.fs, "omp_probe", lambda: (False, env.fs.OMP_RUNTIME))
    env.fs.omp_runtime_local(["core", "accel"])
    assert not env.copied.exists()
    assert kinds(env) == ["warn"] and "libgomp1" in env.messages[0][1]


def test_without_a_runtime_to_copy_the_install_says_what_to_install(env):
    env.ccx_lib.unlink()
    env.mp.setattr(env.fs, "omp_probe", lambda: (False, env.fs.OMP_RUNTIME))
    env.fs.omp_runtime_local(["core", "accel"])
    assert kinds(env) == ["warn"] and "libgomp1" in env.messages[0][1]


def test_a_broken_numba_is_not_reported_as_a_missing_runtime(env):
    """numba does not import for another reason (a damaged package, numpy ABI): no copy, and NOT the "install libgomp1" hint – it would send the user the wrong way."""
    reason = "ModuleNotFoundError: No module named 'numba'"
    env.mp.setattr(env.fs, "omp_probe", lambda: (False, reason))
    env.fs.omp_runtime_local(["core", "accel"])
    assert not env.copied.exists()
    assert kinds(env) == ["warn"]
    text = env.messages[0][1]
    assert reason in text and "libgomp1" not in text and "apt install" not in text and "--force" in text


def test_a_copy_that_fixes_the_runtime_stays_even_when_numba_is_broken_otherwise(env):
    answers = iter([(False, env.fs.OMP_RUNTIME), (False, "ImportError: numpy.core.multiarray failed to import")])
    env.mp.setattr(env.fs, "omp_probe", lambda: next(answers))
    env.fs.omp_runtime_local(["core", "accel"])
    assert env.copied.exists()                                           # the loader found it (the error is no longer about libgomp)
    assert kinds(env) == ["warn"] and "multiarray" in env.messages[0][1] and "libgomp1" not in env.messages[0][1]


@pytest.mark.parametrize("keys,win", [(["core"], False), (["core", "accel"], True)])
def test_no_numba_or_windows_nothing_happens(env, keys, win):
    env.mp.setattr(env.fs, "IS_WIN", win)
    env.mp.setattr(env.fs, "omp_probe", lambda: pytest.fail("not asked"))
    env.fs.omp_runtime_local(keys)
    assert env.messages == []


@pytest.mark.parametrize("stderr,expected", [
    # the loader cannot open the OpenMP runtime: that IS the missing libgomp
    ("Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nImportError: libgomp.so.1.0.0: cannot open shared object file: No such file or directory\n", "runtime"),
    ("ImportError: libgomp.so.1.0.0: file too short\n", "runtime"),
    ("OSError: /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0: cannot read file data: Is a directory\n", "runtime"),
    # everything else is something else
    ("ModuleNotFoundError: No module named 'numba'\n", "ModuleNotFoundError: No module named 'numba'"),
    ("ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory\n", "ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory"),
    ("ImportError: numpy.core.multiarray failed to import\n", "ImportError: numpy.core.multiarray failed to import"),
    # libgomp named in an earlier line (a warning, a chained traceback) but the final exception is another one
    ("libgomp: warning\nTraceback (most recent call last):\nValueError: numpy.dtype size changed\n", "ValueError: numpy.dtype size changed"),
    ("", "no output"),
])
def test_the_failure_of_the_import_is_classified_by_its_exception_line(env, stderr, expected):
    assert env.fs.omp_failure(stderr) == expected


def _stub_numba(tmp_path, omppool_source):
    """A package `numba` with the module numba.np.ufunc.omppool whose import runs `omppool_source` – the real subprocess of omp_probe imports it."""
    pkg = tmp_path / "stubs" / "numba" / "np" / "ufunc"
    pkg.mkdir(parents=True)
    for d in (pkg, pkg.parent, pkg.parent.parent):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "omppool.py").write_text(textwrap.dedent(omppool_source), encoding="utf-8")
    return tmp_path / "stubs"


def test_the_real_probe_separates_a_missing_runtime_from_a_broken_numba(env):
    """A real subprocess (the interpreter of the test, a stub numba): the reason of the probe comes from what the import really printed."""
    env.mp.setenv("PYTHONPATH", str(_stub_numba(env.tmp, 'raise ImportError("libgomp.so.1.0.0: cannot open shared object file: No such file or directory")')))
    assert env.fs.omp_probe() == (False, env.fs.OMP_RUNTIME)
    env.mp.setenv("PYTHONPATH", str(_stub_numba(env.tmp / "b", 'raise ImportError("numpy.core.multiarray failed to import")')))
    assert env.fs.omp_probe() == (False, "ImportError: numpy.core.multiarray failed to import")
    env.mp.setenv("PYTHONPATH", str(_stub_numba(env.tmp / "c", "pass")))
    assert env.fs.omp_probe() == (True, "")


def test_the_install_step_end_to_end_with_real_probes(env):
    """omp_runtime_local with REAL probe subprocesses: the stub fails with the libgomp error until the file is next to the "Python", then loads."""
    stub = f'''
        import os
        if not os.path.exists({str(env.copied)!r}):
            raise ImportError("libgomp.so.1.0.0: cannot open shared object file: No such file or directory")
    '''
    env.mp.setenv("PYTHONPATH", str(_stub_numba(env.tmp, stub)))
    env.fs.omp_runtime_local(["core", "accel"])
    assert env.copied.read_bytes() == b"gomp" and kinds(env) == ["ok"]
    env.messages.clear()
    env.fs.omp_runtime_local(["core", "accel"])                          # a second run (an update): it loads now, nothing to do
    assert env.messages == []

"""Linux only, no root: the REAL dynamic loader and the libgomp that `fea_setup.omp_runtime_local` copies next to the base Python.

The unit tests (tests/test_install_omp.py) mock the probe. Here nothing is mocked: a user + mount namespace (`unshare -rm`, an unprivileged call) hides the
libgomp of the system with bind mounts of /dev/null, the environment is empty (no LD_LIBRARY_PATH), and a copy of the installer's interpreter
(python-build-standalone, RPATH `$ORIGIN/../lib`) in a temporary prefix is the "Python" whose lib/ folder receives the runtime. Proved:
  1. without libgomp numba's OpenMP layer does not import and the installer classifies it as a missing runtime;
  2. after the installer step the very file `<prefix>/lib/libgomp.so.1.0.0` is what the loader maps for numba (via the RPATH of the interpreter, the
     only place that has it) and numba's threading layer is `omp`;
  3. with the system libgomp visible again the copy is still the one numba maps (RPATH goes first) – and what the loader does with a second runtime that
     is loaded first / afterwards (the facts behind the note in installer/README.md).
Skipped where user namespaces are not allowed (some containers, AppArmor-restricted CI runners), where the interpreter is not the installer's Python, or
where numba is not installed. The same scenario for a real installation: scratch/lin/omp_rpath_check.sh (WSL)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the dynamic loader of Linux (ld.so)")

SYSTEM_DIRS = ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/lib64", "/lib")

MASK_SH = """#!/bin/sh
# hides every libgomp of the system library folders from this mount namespace (a bind mount of /dev/null over the real file), then runs the command
for d in /lib /lib64 /usr/lib /usr/lib64 /usr/local/lib; do
    [ -d "$d" ] && find "$d" -maxdepth 2 -name 'libgomp.so*' 2>/dev/null
done | while read -r f; do
    r=$(readlink -f "$f"); [ -f "$r" ] && echo "$r"
done | sort -u | while read -r r; do
    mount --bind /dev/null "$r" || exit 1
done
exec "$@"
"""

INNER_PY = '''
import json, os, subprocess, sys
from pathlib import Path
import fea_setup as fs

os.environ["PYTHONPATH"] = os.environ["OMP_PYTHONPATH"]       # fea_setup clears PYTHONPATH at import (hygiene); the children below need numba from it
lab = Path(os.environ["OMP_LAB"])
fs.IS_WIN = False
fs.VPY = Path(sys.executable)
fs.VENV = lab / ".venv"
fs.TOOLS = lab / "tools"
msgs = []
fs.warn = lambda m: msgs.append(["warn", m])
fs.ok = lambda m: msgs.append(["ok", m])

MAPS = ("import numba.np.ufunc.omppool, os\\n"
        "print(chr(10).join(sorted({os.path.realpath(l.split()[-1]) for l in open('/proc/self/maps') if 'libgomp' in l})))")
ORDER_SYSTEM_LAST = ("import numba.np.ufunc.omppool, ctypes, os\\n"
                     "ctypes.CDLL('libgomp.so.1')\\n"
                     "print(chr(10).join(sorted({os.path.realpath(l.split()[-1]) for l in open('/proc/self/maps') if 'libgomp' in l})))")
ORDER_SYSTEM_FIRST = ("import ctypes, os\\n"
                      "ctypes.CDLL('libgomp.so.1')\\n"
                      "import numba.np.ufunc.omppool\\n"
                      "print(chr(10).join(sorted({os.path.realpath(l.split()[-1]) for l in open('/proc/self/maps') if 'libgomp' in l})))")
KERNEL = ("import numpy as np, numba\\nfrom numba import njit, prange\\n"
          "@njit(parallel=True)\\ndef k(a):\\n    s = 0.0\\n    for i in prange(a.shape[0]):\\n        s += a[i] * a[i]\\n    return s\\n"
          "k(np.random.rand(200000)); print(numba.threading_layer())")


def child(code, **env):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **env})
    return r


def mapped(code):
    r = child(code)
    return [ln for ln in r.stdout.split("\\n") if ln.startswith("/")], r.returncode, (r.stderr.strip().splitlines() or [""])[-1]


out = {}
out["python_home"] = str(fs.python_home())
out["probe_before"] = list(fs.omp_probe())
out["maps_before"] = list(mapped(MAPS))
fs.omp_runtime_local(["core", "accel"])
out["messages"] = msgs
out["probe_after"] = list(fs.omp_probe())
out["maps_after"] = list(mapped(MAPS))
dbg = child("import numba.np.ufunc.omppool", LD_DEBUG="libs").stderr.splitlines()
at = [i for i, ln in enumerate(dbg) if "find library=libgomp" in ln]
out["loader_trace"] = [ln.strip() for ln in dbg[at[0]: at[0] + 6]] if at else []
out["layer"] = child(KERNEL, NUMBA_THREADING_LAYER="omp").stdout.strip().splitlines()[-1:] or ["?"]       # asked for explicitly: an installation of an earlier version still has numba's other layer
if os.environ.get("OMP_SYSTEM_VISIBLE") == "1":
    out["system_last"] = list(mapped(ORDER_SYSTEM_LAST))
    out["system_first"] = list(mapped(ORDER_SYSTEM_FIRST))
print("RESULT " + json.dumps(out))
'''


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, **kw)


def _system_libgomp() -> Path | None:
    for d in SYSTEM_DIRS:
        p = Path(d) / "libgomp.so.1"
        if p.exists():
            return p.resolve()
    return None


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    if not shutil.which("unshare") or not shutil.which("readelf") or not shutil.which("sh"):
        pytest.skip("unshare / readelf / sh are needed")
    spec = importlib.util.find_spec("numba")
    omppool = sorted((Path(spec.origin).parent / "np" / "ufunc").glob("omppool*.so")) if spec and spec.origin else []
    if not omppool:
        pytest.skip("numba (module accel) with its OpenMP layer is not installed")
    base_exe = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    tags = _run(["readelf", "-d", str(base_exe)]).stdout
    if "$ORIGIN/../lib" not in tags or not (base_exe.parent.parent / "lib").is_dir():
        pytest.skip("the interpreter is not the Python of the installer (python-build-standalone with RPATH $ORIGIN/../lib)")
    source = _system_libgomp()
    if source is None:                                  # a bare machine: any real libgomp will do as "the runtime of the CalculiX package"
        wheel = sorted(Path(spec.origin).parent.parent.glob("*.libs/libgomp*.so*"))
        source = wheel[0] if wheel else None
    if source is None:
        pytest.skip("no libgomp on this machine to use as the runtime to copy")
    root = tmp_path_factory.mktemp("omp")
    prefix = root / "py"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    shutil.copy2(base_exe, prefix / "bin" / base_exe.name)                  # a COPY: $ORIGIN of the loader is the real location of the executable
    for entry in (base_exe.parent.parent / "lib").iterdir():
        if not entry.name.startswith("libgomp"):
            (prefix / "lib" / entry.name).symlink_to(entry)
    ccx_lib = root / "tools" / "calculix" / "ccx_2.23_linux" / "lib"
    ccx_lib.mkdir(parents=True)
    shutil.copyfile(source, ccx_lib / "libgomp.so.1")
    (root / "mask.sh").write_text(MASK_SH, encoding="utf-8")
    (root / "inner.py").write_text(INNER_PY, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root), "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(Path(spec.origin).parent.parent), str(ROOT / "installer")]),
           "NUMBA_CACHE_DIR": str(root / "nbcache"), "NUMBA_NUM_THREADS": "2", "OMP_LAB": str(root)}
    env["OMP_PYTHONPATH"] = env["PYTHONPATH"]
    probe = _run(["unshare", "-rm", "sh", str(root / "mask.sh"), "true"], env=env)
    if probe.returncode != 0:
        pytest.skip("no unprivileged user + mount namespace here: " + (probe.stderr.strip().splitlines() or ["?"])[-1])
    return {"root": root, "prefix": prefix, "env": env, "py": prefix / "bin" / base_exe.name, "system": source}


def _scenario(lab, masked: bool) -> dict:
    env = dict(lab["env"], OMP_SYSTEM_VISIBLE="0" if masked else "1")
    cmd = [str(lab["py"]), str(lab["root"] / "inner.py")]
    if masked:
        cmd = ["unshare", "-rm", "sh", str(lab["root"] / "mask.sh"), *cmd]
    r = _run(cmd, env=env)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"no result (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    print(("masked" if masked else "system visible"), lines[-1])
    return json.loads(lines[-1][len("RESULT "):])


@pytest.fixture(scope="module")
def masked(lab):
    return _scenario(lab, True)


def test_without_the_system_libgomp_numba_does_not_load_and_the_installer_sees_a_missing_runtime(masked):
    assert masked["probe_before"] == [False, "runtime"]
    assert masked["maps_before"][1] != 0                                  # the import really failed in the child ...
    assert "libgomp" in masked["maps_before"][2]                          # ... on the OpenMP runtime, not on anything else


def test_after_the_installer_step_numba_maps_the_copy_next_to_python_through_the_rpath(lab, masked):
    copy = (lab["prefix"] / "lib" / "libgomp.so.1.0.0")
    assert copy.is_file() and copy.read_bytes() == lab["system"].read_bytes()
    assert masked["messages"] and [k for k, _ in masked["messages"]] == ["ok"], masked["messages"]
    assert masked["probe_after"] == [True, ""]
    files, code, _ = masked["maps_after"]
    assert code == 0 and files == [str(copy.resolve())], files            # exactly this file, no other libgomp in the process
    trace = " ".join(masked["loader_trace"])
    assert "libgomp.so.1.0.0" in trace and str(lab["prefix"] / "bin" / ".." / "lib") in trace.replace("//", "/"), masked["loader_trace"]
    assert "RPATH" in trace or "RUNPATH" in trace, masked["loader_trace"]   # the loader says by which tag of which file it searched there
    assert masked["layer"] == ["omp"]


def test_the_copy_wins_while_the_system_has_libgomp_and_the_order_of_loading_decides_about_a_second_runtime(lab, masked):
    """Facts behind the note in installer/README.md (the system libgomp1 installed LATER, the copy still in place)."""
    system = _system_libgomp()
    if system is None:
        pytest.skip("the system has no libgomp: the second-runtime scenario needs one")
    r = _scenario(lab, False)
    copy = str((lab["prefix"] / "lib" / "libgomp.so.1.0.0").resolve())
    system = str(system)
    assert r["probe_before"] == [True, ""]                                # the copy from the first scenario is there; nothing is missing now
    assert r["maps_after"][0] == [copy]                                   # numba alone: the RPATH comes before the system folders
    last, first = r["system_last"][0], r["system_first"][0]
    assert last == [copy], last                                           # the system library requested afterwards by its SONAME is the already loaded copy
    assert sorted(first) == sorted({copy, system}), first                 # loaded FIRST under another name: two OpenMP runtimes in one process

"""installer/slim.py: the libraries OCP really loads are read from /proc/self/maps – also when the installation folder has a space in its name."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

MAPS = (
    "7f1c00000000-7f1c00021000 r--p 00000000 08:10 1234   /opt/my programs/fea studio/.venv/lib/site-packages/vtkmodules/libvtkCommonCore.so\n"
    "7f1c00021000-7f1c00300000 r-xp 00021000 08:10 1234   /opt/my programs/fea studio/.venv/lib/site-packages/vtkmodules/libvtkCommonCore.so\n"
    "7f1c10000000-7f1c10001000 rw-p 00000000 00:00 0 \n"
    "7f1c20000000-7f1c20001000 rw-p 00000000 00:00 0                          [heap]\n"
    "7f1c30000000-7f1c30001000 r--p 00000000 08:10 77     /tmp/gone lib.so (deleted)\n"
    "7f1c40000000-7f1c40001000 r--p 00000000 08:10 78     /usr/lib/x86_64-linux-gnu/libc.so.6\n"
)


def test_maps_paths_keep_spaces_of_the_path():
    import slim
    got = slim.maps_paths(MAPS.splitlines(keepends=True))
    assert got == {"/opt/my programs/fea studio/.venv/lib/site-packages/vtkmodules/libvtkCommonCore.so", "/tmp/gone lib.so",
                   "/usr/lib/x86_64-linux-gnu/libc.so.6"}


def test_the_probe_that_runs_in_the_venv_contains_the_same_function():
    """The probe is a script for the python of the venv: the function is pasted into it, and the script compiles."""
    import inspect

    import slim
    probe = slim._VTK_PROBE.replace("@MAPS@", inspect.getsource(slim.maps_paths))
    assert "def maps_paths" in probe and "@MAPS@" not in probe
    compile(probe, "probe", "exec")
    ns: dict = {}
    exec(probe.split("import OCP")[0], ns)                       # the part before the OCP import: imports + the helper
    assert ns["maps_paths"](MAPS.splitlines(keepends=True)) == slim.maps_paths(MAPS.splitlines(keepends=True))


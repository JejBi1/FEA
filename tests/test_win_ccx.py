"""The own Windows CalculiX (F13 step 5d, installer/windows/build_ccx.ps1): the build script and its pins, the package validation, the installer, the
program's choice of the binary, the removal of the third-party ccx_static.exe from existing installations. Fast, no compiler and no network."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from pe_fake import SYSTEM_IMPORTS, build_archive, build_pe, manifest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))
WIN = ROOT / "installer" / "windows"
sys.path.insert(0, str(WIN))
# synthetic paths of a user folder, assembled here so that no absolute home path stands in this file (the anonymization scan of make_package.py)
USER_PATH_WIN = b"C:" + b"\\" + b"Users" + b"\\somebody\\build\\x.c"
USER_PATH_HOME = "/" + "home" + "/somebody/work/"

mp = importlib.import_module("make_package")
cl = importlib.import_module("collect_licenses")
fs = importlib.import_module("fea_setup")
migrations = importlib.import_module("migrations")
fa = importlib.import_module("fetch_ccx_archive")


# ================================================================= the build script and its pins
def test_the_build_script_pins_everything_and_names_no_removed_component():
    sh = (WIN / "build_ccx.sh").read_bytes()
    assert b"\r\n" not in sh                                                   # LF (the script runs in the MSYS2 bash)
    text = sh.decode("utf-8")
    pins = re.findall(r'^get_mirrors\s+"\$DL/([^"]+)"\s+([0-9a-f]{64})', text, re.MULTILINE)
    assert {"ccx_2.23.src.tar.bz2", "spooles.2.2.tgz", "arpack-ng-3.9.1.tar.gz"} <= {n for n, _ in pins}
    assert len({d for _, d in pins}) == len(pins)
    assert "set -Eeuo pipefail" in text and "trap 'echo \"ERROR: build_ccx.sh stopped at line" in text
    # comments and the audit list of the finished binary aside (it names the SYMBOLS that must not be in the binary), the script builds none of the removed
    # components and names none of their sources
    code = re.sub(r"(?m)^\s*#.*$", "", text)
    code = re.sub(r"(?s)for pat in 'taucs_'.*?; do", "", code)
    code = re.sub(r"(?m)^.*not_included.*$", "", code)                           # the manifest says what is NOT inside
    low = code.lower()
    for forbidden in ("taucs", "pastix", "scotch", "metis", "hwloc", "mkl", "libltdl"):
        assert forbidden not in low, forbidden


def test_the_toolchain_lock_matches_the_script_and_every_row_is_pinned():
    lock = (WIN / "ccx_toolchain.lock").read_bytes()
    assert b"\r\n" not in lock
    head, rows = {}, []
    for line in lock.decode("utf-8").splitlines():
        m = re.match(r"# (msys_spec|ucrt_spec): (.*)$", line)
        if m:
            head[m.group(1)] = m.group(2)
        elif line.strip() and not line.startswith("#"):
            kind, digest, url = line.split()
            assert kind in ("base", "msys", "ucrt64") and re.fullmatch(r"[0-9a-f]{64}", digest) and url.startswith("https://repo.msys2.org/")
            rows.append(kind)
    sh = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    assert head["msys_spec"] == re.search(r'^MSYS_SPEC="([^"]+)"', sh, re.MULTILINE).group(1)
    assert head["ucrt_spec"] == re.search(r'^UCRT_SPEC="([^"]+)"', sh, re.MULTILINE).group(1)
    assert rows.count("base") == 1 and rows[0] == "base" and "ucrt64" in rows
    # the packages the binary is made of: the compiler, the Fortran compiler, the GCC runtime
    urls = lock.decode("utf-8")
    for name in ("mingw-w64-ucrt-x86_64-gcc-", "mingw-w64-ucrt-x86_64-gcc-fortran-", "mingw-w64-ucrt-x86_64-libgomp-", "mingw-w64-ucrt-x86_64-binutils-"):
        assert name in urls, name


def test_the_build_fix_patch_removes_only_a_dead_declaration():
    """The UCRT build fix is not a change of solver code (the one change of the solver logic is ccx_2.23_spooles_serial_solve.patch, tested below)."""
    patch = (WIN / "patches" / "ccx_2.23_ucrt_build.patch").read_text(encoding="utf-8")
    removed = [l for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")]
    added = [l for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert removed == ["-_set_output_format(_TWO_DIGIT_EXPONENT);"] * 2 and not added          # a dead declaration, two files, nothing else
    assert (ROOT / ".gitattributes").read_text(encoding="utf-8").count("installer/windows/patches/*.patch -text") == 1


def test_the_serial_solve_and_factor_check_patch_changes_one_call_adds_the_check_and_prints_the_marker_in_both_builds():
    """installer/*/patches/ccx_2.23_spooles_serial_solve.patch (same file for Windows and Linux): the multi-threaded triangular solve of SPOOLES (a data race)
    is replaced by the serial one in the ONE place where ccx calls it, every multi-threaded factorization is CHECKED by a test solve and repeated when the check fails (a thread
    can lose an update silently), and `ccx -v` prints the marker that fea_core.ccx.runner.spooles_threads_safe looks for.
    Both build scripts apply it (before the compilation), build SPOOLES MT, compile with USE_MT and check the marker; the Windows stamp and manifest know it."""
    from fea_core.ccx.runner import FACTOR_CHECK_MARKER, ORDERED_MARKER
    name = "ccx_2.23_spooles_serial_solve.patch"
    win = (WIN / "patches" / name).read_bytes()
    lin = (ROOT / "installer" / "linux" / "patches" / name).read_bytes()
    assert b"\r" not in win and b"\r" not in lin
    assert win[win.index(b"--- a/CalculiX"):] == lin[lin.index(b"--- a/CalculiX"):]   # the hunks are the same for patch(1) on both (the header text may differ: date, wording)
    assert b"Dates of the change: " in win and b"21. 11. 2026" in win[:win.index(b"--- a/CalculiX")]      # GPL-2.0 section 2a: when the files were changed
    for rel in ("installer/windows/patches/" + name, "installer/linux/patches/" + name, "installer/windows/patches/ccx_2.23_ucrt_build.patch"):
        assert mp.packaged(rel), rel                                          # the patches travel with every package (the source offer of the modified CalculiX)
    text = win.decode("utf-8")
    body = text[text.index("--- a/CalculiX"):]
    removed = [l[1:].strip() for l in body.splitlines() if l.startswith("-") and not l.startswith("---")]
    added = [l for l in body.splitlines() if l.startswith("+") and not l.startswith("+++")]
    # removed: the comment line of the step, the three lines of the parallel call and the condition of the two factorization calls (now also "and fewer than two failed
    # attempts") – nothing else of the solver logic
    assert "FrontMtx_MT_solve(pfi->frontmtx, mtxX, mtxB, pfi->mtxmanager," in removed and "pfi->solvemap, pfi->cpus, DEBUG_LVL," in removed
    assert removed.count("if (num_cpus > 1) {") == 2 and len(removed) == 6 and not any("factor" in l.lower() for l in removed)
    assert text.count("goto fea_retry;") == 1 and text.count("goto fea_retry_rad;") == 1                       # both factorizations repeat
    assert text.count("fea_judge(&pfi") == 2 and text.count("fea_judge(&pfj") == 2 and text.count("fea_check_factor(p,") == 1   # the threaded factorization AND the repeat on one thread are judged
    assert "FEA_SPOOLES_CHECK_TOL" in text and "fea_free_factor" in text and "fea_matvec" in text
    for fmt in ("*inputformat==0", "*inputformat==1", "*inputformat==2", "*inputformat==3", "*inputformat==4"):
        assert fmt in text[text.index("static void fea_matvec"):text.index("static double fea_check_factor")], fmt      # the check knows every input format of spooles_factor
    # F2: the ROW-WISE backward error with a right-hand side bounded away from zero and the limit 1e-12 (the norm-wise error of the first version scaled with the stiffest row)
    assert "return 1.e-12;" in text and "e=fabs(y[i]-b0[i])/(rs[i]+fabs(b0[i]));" in text and "d=0.25+0.75*" in text
    assert "rs[R]+=fabs(v*x[C])" in text and "an*xn" not in text and "1.e-9" not in text
    # F4: the repeat on one thread is checked: a pass confirms the loss in the threads, a failure is the matrix (and ends the retries of the process, a static flag); no memory is reported
    assert "threaded loss confirmed" in text and "the matrix, not the threads" in text and "fea_check_off=1;" in text and "static int fea_check_off=0;" in text
    assert "could not be checked (not enough memory" in text
    # F3: an error of the bookkeeping of the threads in SPOOLES (error -2) is a failed check
    assert "if (error == -2) fea_agg_violation = 1;" in text and "int bad=fea_agg_violation,off,unchecked=0;" in text
    # review of 23. 11. 2026: the violation is judged BEFORE the off-switches (a flag of an earlier matrix, the test limit 0), and a repeat whose test solve could not be done
    # is never called confirmed (the behaviour is run by tests/test_spooles_check.py with the function compiled alone)
    judge = text[text.index("static int fea_judge("):text.index("/* link proof of the ordered assembly")]
    assert judge.index("if((off)&&(!bad)) return 0;") < judge.index("fea_check_factor(p,") and "if((fea_check_off)||(tol<=0.)) return 0;" not in judge
    assert judge.count("threaded loss confirmed") == 2 and judge.count("without a verdict") == 2 and "FEA Studio change, 23. 11. 2026" in body
    # F7: the marker of `ccx -v` is only printed by a program that is linked with the ordered SPOOLES (the link proof takes the address of FrontMtx_updateOrdered)
    assert "int fea_spooles_ordered(void)" in text and "=FrontMtx_updateOrdered;" in text and "if(fea_spooles_ordered()) printf(\"FEA_SPOOLES_ORDERED_ASSEMBLY=1" in text
    # F9: GPL-2.0 section 2a - every block that the hunks add carries a dated notice
    assert body.count("FEA Studio change, 21. 11. 2026") >= 9
    assert "FEA_SPOOLES_SERIAL_SOLVE=1" in text
    assert any("FrontMtx_solve(pfi->frontmtx, mtxX, mtxB, pfi->mtxmanager," in l for l in added)
    assert not any("FrontMtx_MT_solve(" in l and not l.lstrip("+").strip().startswith("*") for l in added)      # the new text only mentions it in a comment
    assert FACTOR_CHECK_MARKER.decode() in text and "#ifdef USE_MT" in text    # the marker exists only in a build that can use threads
    assert text.count("--- a/CalculiX/ccx_2.23/src/") == 2
    for attr in (".gitattributes",):
        ga = (ROOT / attr).read_text(encoding="utf-8")
        assert "installer/linux/patches/*.patch -text" in ga and "installer/windows/patches/*.patch -text" in ga
    win_sh = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    lin_sh = (ROOT / "installer" / "linux" / "build_ccx.sh").read_text(encoding="utf-8")
    for sh in (win_sh, lin_sh):
        assert name in sh and FACTOR_CHECK_MARKER.decode() in sh and "-DUSE_MT=1" in sh and "MT_factorMT.o" in sh
        assert ORDERED_MARKER.decode() in sh and sh.count("FEA_SPOOLES_ORDERED_ASSEMBLY=1") >= 3             # in the file AND printed by `ccx -v`, checked by the build
    assert re.search(r'CCXSTAMP="\$\(stamp [^\n]*ccx_2\.23_spooles_serial_solve\.patch', win_sh)         # a change of the patch builds ccx again
    assert '"spooles_serial_solve": true' in win_sh and '"patches": [' in win_sh                           # the manifest says so
    assert "installer/linux/patches/*.patch" in (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")


def test_the_spooles_ordered_assembly_patch_is_the_same_for_both_builds_and_is_applied_before_spooles_is_compiled():
    """installer/*/patches/spooles_2.2_ordered_assembly.patch (public-domain SPOOLES, changed): the order of the sums of the multi-threaded factorization does not depend on
    the arrival times (an ordered insertion of the aggregates, ordered updates; the serial FrontMtx_update is untouched). Both build scripts apply it after the correction of
    util.c and before the compilation; the Windows stamp, manifest and the source offer know it."""
    name = "spooles_2.2_ordered_assembly.patch"
    win = (WIN / "patches" / name).read_bytes()
    lin = (ROOT / "installer" / "linux" / "patches" / name).read_bytes()
    assert b"\r" not in win and b"\r" not in lin and win == lin
    for rel in ("installer/windows/patches/" + name, "installer/linux/patches/" + name):
        assert mp.packaged(rel), rel
    text = win.decode("utf-8")
    assert b"dates of the change: 21. 11. 2026" in win  and text[text.index("--- a/"):].count("FEA Studio change, 21. 11. 2026") >= 15             # GPL-2.0 section 2a style notices in the changed code
    files = re.findall(r"^--- a/(\S+)$", text, re.M)
    assert files == ["Chv/Chv.h", "Chv/src/basics.c", "ChvList/ChvList.h", "ChvList/src/basics.c", "ChvList/src/util.c", "FrontMtx/FrontMtx.h",
                     "FrontMtx/src/factorUtil.c", "FrontMtx/src/update.c", "MT/src/factorMT.c"]                                    # nothing else of SPOOLES is touched
    added = "\n".join(l[1:] for l in text.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = [l[1:].strip() for l in text.splitlines() if l.startswith("-") and not l.startswith("---")]
    assert "FrontMtx_updateOrdered" in added and "sender" in added and "sorted" in added and "owners != NULL" in added          # the threaded path only
    assert "FrontMtx_update(frontmtx, frontJ, heads," in added and "else {" in added                                           # the serial update is still called (else branch)
    assert any("ChvList_isListNonempty(aggList, J) == 1" in l for l in removed)                                                  # the early partial assembly goes
    # F3: every list of a threaded factorization is locked (no flags[] any more), the reads of head and count are done under the lock, and the hand-over is counted
    assert added.count("flags = NULL ;") == 2 and any("flags = CVinit(nfront+1, 'N') ;" in l for l in removed) and any("flags = CVinit(nfront, 'N') ;" in l for l in removed)
    assert "#define FEA_LOCKED(chvlist, ilist)" in added and added.count("Lock_lock(chvlist->lock) ;") >= 5
    assert any("(chvlist->counts != NULL && chvlist->counts[ilist] == 0) ) {" in l for l in removed)                              # the unlocked read of a finished list goes
    assert "nadded" in added and "ntaken" in added and "nassembled" in added and "ChvList_noteAssembled(aggList, nassembled) ;" in added
    assert "ChvList_closeCheck(aggList, -1, 1)" in added and "*perror = -2 ;" in added and text.index("ChvList_closeCheck(aggList, -1, 1)") < text.index("ChvList_free(aggList) ;")
    win_sh = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    lin_sh = (ROOT / "installer" / "linux" / "build_ccx.sh").read_text(encoding="utf-8")
    for sh in (win_sh, lin_sh):
        assert name in sh and "patch -p1" in sh
    assert re.search(r'SPSTAMP="\$\(stamp [^\n]*spooles_2\.2_ordered_assembly\.patch', win_sh)      # a change of the patch builds SPOOLES again
    assert win_sh.index("spooles_2.2_ordered_assembly.patch\" > \"$WORK/spooles_patch.log") < win_sh.index("make -s lib")              # applied before the compilation
    assert lin_sh.index("spooles_2.2_ordered_assembly.patch") < lin_sh.index("$MAKE -s lib")
    assert '"spooles_ordered_assembly": true' in win_sh and name in win_sh[win_sh.index('"patches": ['):]
    assert name in (ROOT / "installer" / "license_texts" / "SOURCES.md").read_text(encoding="utf-8")
    patch = shutil.which("patch")
    tgz = ROOT / "tools" / "winccx_build" / "archive" / "spooles.2.2.tgz"
    if patch and tgz.exists():                                                                                                     # the real thing when the pinned archive is here
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as td, tarfile.open(tgz) as tf:
            for m in files:
                tf.extract(m, td)
            r = subprocess.run([patch, "-p1", "-i", str(WIN / "patches" / name)], cwd=td, capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert r.returncode == 0, r.stdout + r.stderr


def test_the_linux_build_applies_the_patches_in_the_right_order_to_the_right_trees():
    """installer/linux/build_ccx.sh: the CalculiX patch (hunks made against the pristine tree) is applied to the unpacked CalculiX before anything is compiled, the SPOOLES
    patch after the correction of util.c and before SPOOLES is compiled, the PaStiX patch only in the optional PaStiX build, after the main binary; the marker checks of
    ccx_spooles come after its build. The Windows script applies the build fix first and the solver patch after it (the hunk of ccx_2.23.c then has an offset of one line)."""
    lin = (ROOT / "installer" / "linux" / "build_ccx.sh").read_text(encoding="utf-8")
    ccx_patch, sp_patch = "ccx_${CCX_VER}_spooles_serial_solve.patch", "spooles_2.2_ordered_assembly.patch"
    pastix_patch = "ccx_${CCX_VER}_pastix_upstream.patch"
    order = [lin.index('tar -xjf ../ccx.src.tar.bz2'), lin.index(ccx_patch), lin.index("cp CalculiX/ccx_2.23/SPOOLES.2.2/I2Ohash/src/util.c"), lin.index(sp_patch),
             lin.index("$MAKE -s lib"), lin.index('build_one ccx_spooles "-DUSE_MT=1 -DFEA_FAER=1'), lin.index("grep -aqF 'FEA_SPOOLES_ORDERED_ASSEMBLY=1'"), lin.index(pastix_patch)]
    assert order == sorted(order) and len(set(order)) == len(order), order
    assert "cd SPOOLES.2.2 && patch -p1" in lin and lin.count("patch -p1 -i") == 4                  # the SPOOLES patch inside the SPOOLES folder, four patches in all (serial solve, faer, SPOOLES, PaStiX)
    assert lin.index("ccx_${CCX_VER}_faer.patch") > lin.index(ccx_patch) and lin.index("ccx_${CCX_VER}_faer.patch") < lin.index("$MAKE -s lib")   # the faer hunks need the context of the serial solve patch
    win = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    worder = [win.index('patch -p1 -i "$SCRIPT_DIR/patches/ccx_2.23_ucrt_build.patch"'), win.index('patch -p1 -i "$SCRIPT_DIR/patches/ccx_2.23_spooles_serial_solve.patch"'),
              win.index('make -f Makefile.fea')]
    assert worder == sorted(worder)
    assert win.index('patch -p1 -i "$SCRIPT_DIR/patches/spooles_2.2_ordered_assembly.patch"') < win.index("make -s lib")


def test_the_ps1_launcher_is_a_thin_wrapper():
    ps1 = (WIN / "build_ccx.ps1").read_text(encoding="utf-8")
    assert "build_ccx.sh" in ps1 and "ccx_toolchain.lock" in ps1 and "subst" in ps1
    assert "ccx_2.23_windows_x86_64.zip" in ps1 and "installer\\windows\\prebuilt" in ps1.replace("Join-Path $here \"prebuilt", "installer\\windows\\prebuilt") or "prebuilt" in ps1
    assert "installer/windows/prebuilt/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    # the local archive of the downloads, the offline proof and the relinking hooks of LGPL-2.1 section 6
    for word in ("-Downloads", "-Offline", "FEA_ARCHIVE", "FEA_OFFLINE", "-LibQuadmathSource", "FEA_LIBQUADMATH_SOURCE", "-LibQuadmath", "FEA_LIBQUADMATH_DIR"):
        assert word in ps1, word
    assert "tools/winccx_build/" in (ROOT / ".gitignore").read_text(encoding="utf-8") or "/tools/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_the_build_script_uses_the_archive_stamps_its_caches_and_derives_its_paths():
    sh = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    code = re.sub(r"(?m)^\s*#.*$", "", sh)
    assert 'DL="$ARCHIVE"' in sh and 'f="$ARCHIVE/$(basename "$url")"' in sh and "FEA_OFFLINE" in sh          # the archive before the network, also for the toolchain
    for stamp in ('BLASSTAMP=', 'ARPSTAMP=', 'SPSTAMP=', 'CCXSTAMP='):                                      # cached libraries are keyed to the lock and the flags
        assert stamp in sh, stamp
    assert re.search(r'BLASSTAMP="\$\(stamp "\$STAMP" "\$FFL"', sh) and re.search(r'SPSTAMP="\$\(stamp "\$STAMP" "\$CF_SPOOLES"', sh)
    assert re.search(r'CCXSTAMP="\$\(stamp "\$STAMP" "\$SPSTAMP"', sh)
    # the audit of the finished binary derives the folders of this build; none of them is written into the script
    assert "'W:/'" not in code and "/w/work" not in code and "msys64" not in code
    assert 'WM="$(M "$WORK")"' in sh and 'paths=(' in sh
    assert "-Wl,-Map=" in sh and "audit_static_libs" in sh and " libquadmath.a " in sh        # the link map is the list of the static libraries; libquadmath is known and LGPL
    assert "FEA_LIBQUADMATH_SOURCE" in sh and "FEA_LIBQUADMATH_DIR" in sh and "libquadmath_build" in sh


def test_ci_builds_the_windows_ccx_on_a_cache_miss():
    path = ROOT / ".github" / "workflows" / "tests.yml"
    if not path.is_file():
        pytest.skip("no CI workflow in this tree")
    yml = path.read_text(encoding="utf-8")
    assert "installer/windows/prebuilt/ccx_2.23_windows_x86_64.zip" in yml and "id: ccx-win-cache" in yml
    assert "steps.ccx-win-cache.outputs.cache-hit != 'true'" in yml and "build_ccx.ps1" in yml
    assert yml.index("build_ccx.ps1") < yml.index("installer/fea_setup.py install")             # built / restored BEFORE the install
    assert "calculix_2.23_4win" not in yml and "dhondt.de/calculix" not in yml           # the third-party zip is not downloaded or cached any more


# ================================================================= the package: validation of the prebuilt archive
def test_pe_imports_reads_the_dll_names():
    assert mp.pe_imports(build_pe(SYSTEM_IMPORTS)) == [d.lower() for d in SYSTEM_IMPORTS]
    assert mp.pe_imports(build_pe([])) == []
    for bad in (b"", b"MZ", b"not an executable at all" * 10, build_pe(SYSTEM_IMPORTS)[:200]):
        with pytest.raises(ValueError):
            mp.pe_imports(bad)


def test_the_real_ccx_spooles_imports_only_system_dlls_if_it_was_built():
    z = WIN / "prebuilt" / "ccx_2.23_windows_x86_64.zip"
    if not z.is_file():
        pytest.skip("the Windows CalculiX has not been built here (installer/windows/build_ccx.ps1)")
    with zipfile.ZipFile(z) as f:
        exe = f.read("ccx_spooles.exe")
        meta = json.loads(f.read("manifest.json"))
    imports = mp.pe_imports(exe)
    assert imports and all(d in mp.WIN_SYSTEM_DLLS or d.startswith("api-ms-win-") for d in imports), imports
    assert meta["calculix"] == "2.23" and meta["sources"] and meta["binary_sha256"]
    low = exe.lower()
    # the symbols of the removed libraries (CalculiX itself names TAUCS and PaStiX in its messages "the ... library is not linked") - faer carries Rust ports of AMD and COLAMD (paths of their sources name colamd), so the C functions are looked for
    for name in (b"taucs_", b"oocsp", b"colamd_recommended", b"colamd_set_defaults", b"symamd", b"genmmd", b"amdbar", b"amdexa", b"pastix_", b"pastix version", b"scotch_", b"hwloc_", b"metis_",
                 b"lt_dlopen", b"openblas", b"mkl_"):
        assert name not in low, name
    assert not mp.PRIVATE_PATH.search(exe)


def test_windows_ccx_accepts_the_archive_of_the_build(tmp_path):
    src = build_archive(tmp_path / "pre.zip")
    mp.windows_ccx(src, tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as z:
        assert sorted(z.namelist()) == sorted(mp.WIN_CCX_MEMBERS)
    assert mp.WIN_CCX_MEMBERS == ("ccx_spooles.exe", "manifest.json")


@pytest.mark.parametrize("what,archive,match", [
    ("a foreign DLL", dict(imports=["KERNEL32.dll", "libgfortran-5.dll"]), "libgfortran-5.dll"),
    ("a mingw runtime DLL", dict(imports=["KERNEL32.dll", "libwinpthread-1.dll"]), "libwinpthread"),
    ("a path of a user folder in the binary", dict(exe_tail=USER_PATH_WIN), "absolute path"),
    ("a path of a home folder in the manifest", dict(members={"manifest.json": manifest(note=USER_PATH_HOME)}), "absolute path"),
    ("an extra member", dict(members={"README.txt": b"x"}), "members"),
    ("a missing manifest", dict(members={"manifest.json": None}), "members"),
    ("a manifest that is no JSON", dict(members={"manifest.json": b"{not json"}), "JSON"),
    ("an executable that is no PE", dict(members={"ccx_spooles.exe": b"\x7fELF" + b"\0" * 100}), "PE"),
    ("the digest of another executable", dict(binary_sha256="0" * 64), "binary_sha256"),
    ("other build scripts than the checkout", dict(build_script_sha256="1" * 64), "build_script_sha256"),
    ("another toolchain lock than the checkout", dict(toolchain_lock_sha256="2" * 64), "toolchain_lock_sha256"),
])
def test_windows_ccx_refuses_a_bad_archive(tmp_path, what, archive, match):
    src = build_archive(tmp_path / "pre.zip", **archive)
    with pytest.raises(SystemExit, match=match):
        mp.windows_ccx(src, tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_a_windows_package_without_the_prebuilt_ccx_is_an_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        mp.require_windows_ccx(tmp_path / "missing.zip")
    msg = str(e.value)
    assert "build_ccx.ps1" in msg and "--no-windows-ccx" in msg and "neither downloads nor compiles" in msg
    ok = build_archive(tmp_path / "ok.zip")
    assert mp.require_windows_ccx(ok) == ok


def test_bundle_windows_tools_puts_the_archive_into_the_package(tmp_path, monkeypatch):
    pre = build_archive(tmp_path / "pre.zip")
    monkeypatch.setattr(mp, "WIN_CCX", pre)
    stage = tmp_path / "stage"
    made = mp.bundle_windows_tools(stage)
    assert stage / "tools" / "calculix" in made
    assert (stage / "tools" / "calculix" / "pre.zip").is_file()
    stage2 = tmp_path / "stage2"
    made2 = mp.bundle_windows_tools(stage2, with_ccx=False)                    # --no-windows-ccx
    assert stage2 / "tools" / "calculix" not in made2 and not (stage2 / "tools" / "calculix").exists()


def test_windows_sources_come_from_the_pins_of_the_build_script():
    got = {n: (d, urls) for n, d, urls in cl.build_ccx_win_sources(ROOT)}
    assert {"ccx_2.23.src.tar.bz2", "spooles.2.2.tgz", "arpack-ng-3.9.1.tar.gz"} <= set(got)
    ccx = cl.SOURCE_ARCHIVES["ccx_2.23.src.tar.bz2"]
    assert got["ccx_2.23.src.tar.bz2"] == (ccx["sha256"], [ccx["url"]])        # the same file and digest as the shared entry
    assert not any(n.startswith("msys2") for n in got)                        # the toolchain is pinned in the lock, it is not a source of the binary
    assert sum(("OpenBLAS" in n) + n.startswith("lapack-") for n in got) == 1  # only the BLAS / LAPACK that is built in
    # the source packages of the run-time libraries that are linked in (libquadmath is LGPL): pinned with the digests of the MSYS2 repository
    runtime = {n for n in got if cl.is_runtime_source(n)}
    assert len(runtime) == 4 and any(n.startswith("mingw-w64-gcc-16.2.0-7") for n in runtime)
    assert all(got[n][1][0].startswith("https://repo.msys2.org/mingw/sources/") for n in runtime)
    assert {n for n in runtime if cl.is_recipe_only(n)} == {n for n in runtime if n.startswith(("mingw-w64-headers-", "mingw-w64-winpthreads-"))}


def _fake_sources(tmp_path, monkeypatch, archive_dir=None):
    """add_sources without the network and without tar: the downloads are small files, the recipes are two small files in a folder named like the package."""
    got = []

    def fake_download(url, dest, what="", sha256=None):
        got.append((Path(dest).name, sha256))
        p = tmp_path / "dl" / Path(dest).name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"source")
        return p
    monkeypatch.setattr(fs, "download", fake_download)
    monkeypatch.setattr(mp, "WIN_ARCHIVE", archive_dir or tmp_path / "no_archive")

    def fake_recipe(archive, dst):
        folder = dst / cl.recipe_folder(archive.name)
        folder.mkdir(parents=True, exist_ok=True)
        for n in ("PKGBUILD", ".SRCINFO"):
            (folder / n).write_text("recipe", encoding="utf-8")
        return [folder / "PKGBUILD", folder / ".SRCINFO"]
    monkeypatch.setattr(mp, "extract_recipe", fake_recipe)
    return got


def test_add_sources_windows_downloads_the_pinned_archives(tmp_path, monkeypatch):
    got = _fake_sources(tmp_path, monkeypatch)
    made = mp.add_sources(tmp_path / "stage", "windows")
    names = {p.name for p in made}
    assert "spooles.2.2.tgz" in names and "arpack-ng-3.9.1.tar.gz" in names
    assert "ccx_2.23.src.tar.bz2" not in names                                 # that one is in "all" (every package)
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for _, d in got)


def test_add_sources_windows_ships_the_runtime_sources_gcc_and_crt_whole_the_rest_as_recipes(tmp_path, monkeypatch):
    _fake_sources(tmp_path, monkeypatch)
    stage = tmp_path / "stage"
    made = mp.add_sources(stage, "windows")
    src = stage / "licenses" / "sources"
    whole = sorted(p.name for p in src.iterdir() if p.is_file())
    assert any(n.startswith("mingw-w64-gcc-16.2.0-7.src") for n in whole) and any(n.startswith("mingw-w64-crt-") for n in whole)     # libquadmath's source is in the GCC package
    assert not any(n.startswith(("mingw-w64-headers-", "mingw-w64-winpthreads-")) for n in whole)                                        # 57 MB each, the same git commit as the CRT
    assert any(n.startswith("rust-src-") for n in whole) and len(list((src / "crates").glob("*.crate"))) >= 40                      # the Rust sources of the faer library travel with the binary
    folders = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name != "crates")      # crates/ = the Rust crates of the faer library (F17 stage C)
    assert len(folders) == 2 and all((src / f / "PKGBUILD").is_file() and (src / f / ".SRCINFO").is_file() for f in folders)
    assert {p.parent.name for p in made if p.name == "PKGBUILD"} == set(folders)
    # a package for tests without them says so
    made2 = mp.add_sources(tmp_path / "stage2", "windows", runtime=False)
    assert made2 and not any(cl.is_runtime_source(p.name) or p.name in ("PKGBUILD", ".SRCINFO") or p.suffix == ".crate" or p.name.startswith("rust-src-") for p in made2)


def test_source_archives_come_from_the_local_archive_when_the_digest_matches(tmp_path, monkeypatch):
    got = _fake_sources(tmp_path, monkeypatch, archive_dir=tmp_path / "archive")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "good.tgz").write_bytes(b"good")
    (tmp_path / "archive" / "bad.tgz").write_bytes(b"corrupt")
    digest = hashlib.sha256(b"good").hexdigest()
    assert mp.source_archive("good.tgz", ["https://example.invalid/good.tgz"], digest) == tmp_path / "archive" / "good.tgz"
    assert not got                                                             # no download
    assert mp.source_archive("bad.tgz", ["https://example.invalid/bad.tgz"], digest).parent == tmp_path / "dl"      # a wrong file of the archive is not used
    assert got == [("bad.tgz", digest)]


def test_the_recipe_is_extracted_unchanged_from_a_source_package(tmp_path):
    tar = mp.tar_program()
    if not Path(tar).is_file() and not shutil.which(tar):
        pytest.skip("no tar")
    pkg = tmp_path / "pkg" / "mingw-w64-headers"
    pkg.mkdir(parents=True)
    (pkg / "PKGBUILD").write_text("pkgname=x\n", encoding="utf-8", newline="\n")
    (pkg / ".SRCINFO").write_text("pkgbase = x\n", encoding="utf-8", newline="\n")
    (pkg / "big.bin").write_bytes(b"x" * 1000)
    archive = tmp_path / "mingw-w64-headers-1.0.0-1.src.tar.zst"
    r = subprocess.run([tar, "--zstd", "-cf", str(archive), "-C", str(tmp_path / "pkg"), "mingw-w64-headers"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("this tar cannot write zstd")
    out = mp.extract_recipe(archive, tmp_path / "dst")
    assert sorted(p.name for p in out) == [".SRCINFO", "PKGBUILD"]
    assert (tmp_path / "dst" / "mingw-w64-headers-1.0.0-1" / "PKGBUILD").read_bytes() == b"pkgname=x\n"
    assert not (tmp_path / "dst" / "mingw-w64-headers-1.0.0-1" / "big.bin").exists()


# ================================================================= the installer unpacks, never downloads or builds
@pytest.fixture
def win(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "IS_WIN", True)
    monkeypatch.setattr(fs, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    (tmp_path / "tools" / "calculix").mkdir(parents=True)
    monkeypatch.setattr(fs, "run", lambda *a, **k: pytest.fail("the installer must not run anything for CalculiX"))
    monkeypatch.setattr(fs, "download", lambda *a, **k: pytest.fail("the installer must not download CalculiX"))
    monkeypatch.setattr(fs, "ok", lambda *_: None)
    monkeypatch.setattr(fs, "info", lambda *_: None)
    return tmp_path


def test_windows_ccx_comes_from_the_package(win):
    build_archive(win / "tools" / "calculix" / "ccx_2.23_windows_x86_64.zip")
    fs.install_ext_ccx("ccx_static")
    exe = win / "tools" / "calculix" / "ccx_2.23_windows" / "ccx_spooles.exe"
    assert exe.is_file() and (exe.parent / "manifest.json").is_file()
    fs.install_ext_ccx("ccx_static")                                           # repeated install: nothing to do


def test_windows_ccx_comes_from_the_checkout_prebuilt_folder(win):
    folder = win / "installer" / "windows" / "prebuilt"
    folder.mkdir(parents=True)
    build_archive(folder / "ccx_2.23_windows_x86_64.zip")
    assert fs.prebuilt_ccx_win_candidates()[-1] == folder / "ccx_2.23_windows_x86_64.zip"
    fs.install_ext_ccx("ccx_static")
    assert (win / "tools" / "calculix" / "ccx_2.23_windows" / "ccx_spooles.exe").is_file()


def test_windows_without_the_prebuilt_ccx_is_an_error_not_a_download(win):
    with pytest.raises(RuntimeError) as e:
        fs.install_ext_ccx("ccx_static")
    msg = str(e.value)
    assert "ccx_2.23_windows_x86_64.zip" in msg and "FEA_CCX" in msg and "build_ccx.ps1" in msg
    assert "nestahuje" in msg or "neither downloaded" in msg


def test_the_installer_no_longer_knows_the_third_party_zip():
    src = (ROOT / "installer" / "fea_setup.py").read_text(encoding="utf-8")
    assert "calculix_2.23_4win" not in src and "CCX_WIN_URL" not in src and "dhondt.de/calculix" not in src
    assert fs.CCX_WIN_EXE == "ccx_spooles.exe"


def test_solver_available_on_windows_wants_the_own_binary(win, monkeypatch):
    monkeypatch.delenv("FEA_CCX", raising=False)
    monkeypatch.setenv("PATH", str(win / "empty"))
    (win / "tools" / "calculix" / "calculix_2.23_4win").mkdir()
    (win / "tools" / "calculix" / "calculix_2.23_4win" / "ccx_static.exe").write_text("x")
    assert not fs.solver_available()                                           # the third-party build is not a solver of this version
    (win / "tools" / "calculix" / "ccx_2.23_windows").mkdir()
    (win / "tools" / "calculix" / "ccx_2.23_windows" / "ccx_spooles.exe").write_text("x")
    assert fs.solver_available()


# ================================================================= the program's choice of the binary
def test_find_ccx_prefers_the_own_build_and_no_longer_selects_ccx_static(tmp_path, monkeypatch):
    from fea_core import config
    d = tmp_path / "calculix" / "ccx_2.23_windows"
    old = tmp_path / "calculix" / "calculix_2.23_4win"
    d.mkdir(parents=True)
    old.mkdir()
    monkeypatch.setattr(config, "CCX_TOOLS", tmp_path / "calculix")
    monkeypatch.delenv("FEA_CCX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    (old / f"ccx_static{config.EXE}").write_text("x")
    assert config.find_ccx().name == f"ccx_spooles{config.EXE}" and not config.find_ccx().exists()     # the missing default, not ccx_static
    (d / f"ccx_spooles{config.EXE}").write_text("x")
    assert config.find_ccx() == d / f"ccx_spooles{config.EXE}"
    monkeypatch.setenv("FEA_CCX", str(old / f"ccx_static{config.EXE}"))                                   # a developer's override still works
    assert config.find_ccx() == old / f"ccx_static{config.EXE}"


def test_solvers_follow_the_binary(monkeypatch):
    from fea_core import config
    for exe, pastix in (("ccx_spooles.exe", False), ("ccx_spooles", False), ("ccx_pastix", True), ("ccx_2.23", False), ("ccx_static.exe", True)):
        monkeypatch.setattr(config, "CCX_EXE", Path("x") / exe)
        assert config.ccx_solvers() == {"PASTIX": pastix, "SPOOLES": True}, exe
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles.exe"))
    assert config.solver_name("PASTIX", "static") == "SPOOLES"                 # a stored PASTIX → what the binary offers
    assert config.solver_name(config.AUTO_SOLVER, "static") == "SPOOLES"
    assert config.solver_name("SPOOLES", "modal") == "SPOOLES"


def test_an_old_study_with_the_pastix_solver_loads_and_the_run_says_so(monkeypatch):
    from fea_core import config
    from fea_core.solve import ccx_solver_note
    from fea_core.study import Study
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles.exe"))
    st = Study("old", "model.step")
    st.solver.solver = "PASTIX"
    warnings, log = [], []
    assert ccx_solver_note(st, warnings, log.append, "static") == "SPOOLES"
    assert warnings and "PASTIX" in warnings[0] and "SPOOLES" in warnings[0]


# ================================================================= the old third-party CalculiX leaves existing installations
# the content of the fake files of the old zip is "x": the migration deletes a file only when it is byte for byte the one of the zip (SHA-256)
X_SHA = hashlib.sha256(b"x").hexdigest()
REAL_OLD_ZIP = dict(migrations.OLD_CCX_ZIP)                                    # the table of the module before the fixture below replaces it


@pytest.fixture(autouse=True)
def fake_zip_digests(monkeypatch):
    monkeypatch.setattr(migrations, "OLD_CCX_ZIP", {k: X_SHA for k in migrations.OLD_CCX_ZIP})
    monkeypatch.delenv("FEA_CCX", raising=False)


def _old_install(tmp_path: Path, full_zip: bool = False, own: bool = True):
    root = tmp_path / "FEA Studio"
    venv = root / ".venv"
    venv.mkdir(parents=True)
    calc = root / "tools" / "calculix"
    old = calc / "calculix_2.23_4win"
    old.mkdir(parents=True)
    names = ["ccx_static.exe", "README_Install"]
    if full_zip:
        names += ["ccx_dynamic.exe", "ccx_dynamic_i8.exe", "cgx_GLUT.exe", "cgx_STATIC.exe", "glut64.dll", "tetgen.exe", "README_Install~"]
    for n in names:
        (old / n).write_text("x")
    if full_zip:
        (old / "cgxCadTools").mkdir()
        (old / "cgxCadTools" / "cad2fbd.exe").write_text("x")
    if own:                                                                    # the program's own build is in place (unpacked by the installer)
        (calc / "ccx_2.23_windows").mkdir()
        (calc / "ccx_2.23_windows" / "ccx_spooles.exe").write_text("new")
    return root, venv, calc


def _ctx(root: Path, venv: Path, said: list, warned: list):
    return migrations.Context(root=root, venv=venv, vpy=venv / "Scripts" / "python.exe", modules=["core"], say=said.append, warn=warned.append)


@pytest.mark.parametrize("full", [False, True])
def test_the_migration_removes_ccx_static_and_the_rest_of_the_zip_and_is_idempotent(tmp_path, full):
    root, venv, calc = _old_install(tmp_path, full)
    own = calc / "ccx_2.23_windows"
    said, warned = [], []
    migrations.remove_thirdparty_ccx(_ctx(root, venv, said, warned))
    assert not warned and said and "ccx_static" in said[0]
    assert sorted(p.name for p in calc.iterdir()) == ["ccx_2.23_windows"]       # the old folder is gone, the own build stays
    assert (own / "ccx_spooles.exe").read_text() == "new"
    said.clear()
    migrations.remove_thirdparty_ccx(_ctx(root, venv, said, warned))             # a second run finds nothing
    assert said == [] and warned == []


def test_the_old_build_stays_until_the_own_one_is_in_place(tmp_path):
    """The removal runs after the external components were installed and must not leave the user without a solver when that failed."""
    root, venv, calc = _old_install(tmp_path, own=False)
    said, warned = [], []
    migrations.remove_thirdparty_ccx(_ctx(root, venv, said, warned))
    assert (calc / "calculix_2.23_4win" / "ccx_static.exe").is_file() and any("ccx_spooles.exe" in w for w in warned)
    own = calc / "ccx_2.23_windows"
    own.mkdir()
    (own / "ccx_spooles.exe").write_text("truncated")
    (own / "manifest.json").write_text(json.dumps({"binary_sha256": hashlib.sha256(b"the whole file").hexdigest()}))       # the file does not match its manifest
    warned.clear()
    migrations.remove_thirdparty_ccx(_ctx(root, venv, said, warned))
    assert (calc / "calculix_2.23_4win" / "ccx_static.exe").is_file() and warned
    (own / "ccx_spooles.exe").write_text("the whole file")
    warned.clear()
    migrations.remove_thirdparty_ccx(_ctx(root, venv, said, warned))
    assert not (calc / "calculix_2.23_4win").exists() and not warned


def test_the_migration_keeps_foreign_files_and_their_folder(tmp_path):
    root, venv, calc = _old_install(tmp_path, True)
    (calc / "calculix_2.23_4win" / "my_notes.txt").write_text("mine")
    doc = calc / "doc" / "CalculiX" / "ccx_2.23" / "doc" / "ccx"                # a manual that a developer put there: not a file of the zip, never touched
    doc.mkdir(parents=True)
    (doc / "ccx.html").write_text("<html>")
    (doc / "img1.png").write_bytes(b"png")
    (calc / "doc" / "CalculiX" / "extra.docx").write_bytes(b"mine")
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], []))
    assert not (calc / "calculix_2.23_4win" / "ccx_static.exe").exists()
    assert (calc / "calculix_2.23_4win" / "my_notes.txt").read_text() == "mine"  # an unknown file stays, so does its folder
    assert (calc / "doc" / "CalculiX" / "extra.docx").is_file() and (doc / "ccx.html").is_file() and (doc / "img1.png").is_file()


def test_a_file_that_is_not_the_one_of_the_zip_is_kept(tmp_path):
    """A ccx_static.exe that a developer built or replaced there (another content than the original zip) is not ours to delete."""
    root, venv, calc = _old_install(tmp_path)
    mine = calc / "calculix_2.23_4win" / "ccx_static.exe"
    mine.write_text("my own build")
    warned = []
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], warned))
    assert mine.read_text() == "my own build" and any("ccx_static.exe" in w for w in warned)
    assert not (calc / "calculix_2.23_4win" / "README_Install").exists()          # the README of the zip, byte for byte, is gone


def test_a_file_named_by_fea_ccx_is_kept(tmp_path, monkeypatch):
    root, venv, calc = _old_install(tmp_path)
    mine = calc / "calculix_2.23_4win" / "ccx_static.exe"
    monkeypatch.setenv("FEA_CCX", str(mine))                                      # the developer's override for comparisons
    warned = []
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], warned))
    assert mine.is_file() and any("FEA_CCX" in w and "ccx_static.exe" in w for w in warned)
    assert not (calc / "calculix_2.23_4win" / "README_Install").exists()
    monkeypatch.delenv("FEA_CCX")
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], []))                    # without the variable the next run removes it
    assert not (calc / "calculix_2.23_4win").exists()


def test_a_ccx_static_exe_outside_the_old_install_folder_is_never_touched(tmp_path):
    """The cleanup runs on every update: it deletes files of the old zip in tools/calculix/calculix_2.23_4win and nothing named like them anywhere else."""
    root, venv, calc = _old_install(tmp_path)
    elsewhere = [calc / "ccx_static.exe", calc / "my_build" / "ccx_static.exe", calc / "calculix_2.23_4win_copy" / "ccx_static.exe", root / "ccx_static.exe"]
    for p in elsewhere:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("mine")
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], []))
    assert not (calc / "calculix_2.23_4win").exists()
    assert all(p.read_text() == "mine" for p in elsewhere)


def test_the_migration_never_goes_through_a_link(tmp_path):
    root, venv, calc = _old_install(tmp_path)
    other = tmp_path / "other checkout"
    shutil_calc = other / "tools" / "calculix" / "calculix_2.23_4win"
    shutil_calc.mkdir(parents=True)
    (shutil_calc / "ccx_static.exe").write_text("theirs")
    tools = root / "tools"
    import shutil
    shutil.rmtree(tools)
    try:
        tools.symlink_to(other / "tools", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no permission to create symbolic links")
    said = []
    migrations.remove_thirdparty_ccx(_ctx(root, venv, said, []))
    assert (shutil_calc / "ccx_static.exe").read_text() == "theirs"            # the other installation owns the files
    assert said and "link" in said[0]


def test_a_locked_file_is_reported_and_the_next_run_finishes(tmp_path, monkeypatch):
    root, venv, calc = _old_install(tmp_path)
    real_unlink = Path.unlink

    def locked(self, *a, **k):
        if self.name == "ccx_static.exe":
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", locked)
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    warned = []
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], warned))
    assert len(warned) == 2 and "ccx_static.exe" in warned[0]                    # the file, and the advice to close the program
    assert (calc / "calculix_2.23_4win" / "ccx_static.exe").is_file() and not (calc / "calculix_2.23_4win" / "README_Install").exists()
    monkeypatch.setattr(Path, "unlink", real_unlink)                             # the calculation was closed
    warned.clear()
    migrations.remove_thirdparty_ccx(_ctx(root, venv, [], warned))
    assert not warned and not (calc / "calculix_2.23_4win").exists()


def test_the_migration_is_a_final_step_registered_for_1_4_2_and_runs_on_every_update():
    """final = after the external components (the own CalculiX) were installed; the old build is never deleted before the new one is unpacked."""
    assert any(m.fn is migrations.remove_thirdparty_ccx for m in migrations.pending("1.4.1", "1.4.2", "final"))
    assert not any(m.fn is migrations.remove_thirdparty_ccx for m in migrations.pending("1.4.1", "1.4.2", "pre"))
    assert not any(m.fn is migrations.remove_thirdparty_ccx for m in migrations.pending("1.4.2", "1.4.2"))
    assert any(m.fn is migrations.remove_thirdparty_ccx and m.phase == "final" for m in migrations.HOUSEKEEPING)     # a file held by a program is removed by the next run
    src = (ROOT / "installer" / "fea_setup.py").read_text(encoding="utf-8")
    body = src[src.index("def cmd_install("):]
    assert body.index("EXT[e]()") < body.index('migrations.run(ctx, old_ver, new_ver, "final")') < body.index("if not solver_available():")


def test_the_third_party_ccx_step_is_announced_on_windows_only(tmp_path, monkeypatch):
    """The step "third-party CalculiX (ccx_static.exe) replaced by the own build" is about a file that exists on Windows only: the Linux update does not print its
    title (or run it as a migration); the silent cleanup of HOUSEKEEPING still looks for leftovers on every platform."""
    root, venv, calc = _old_install(tmp_path)
    for plat, shown in (("win32", True), ("linux", False), ("darwin", False)):
        monkeypatch.setattr(migrations.sys, "platform", plat)
        said, warned = [], []
        done = migrations.run(_ctx(root, venv, said, warned), "1.4.1", "1.4.2", "final", [m for m in migrations.MIGRATIONS if m.fn is migrations.remove_thirdparty_ccx])
        assert any("ccx_static.exe) replaced by the own build" in t or "nahrazen vlastním sestavením" in t for t in said) is shown, (plat, said)
        assert bool(done) is shown
        assert migrations.remove_thirdparty_ccx in [m.fn for m in migrations.pending("1.4.1", "1.4.2", "final")]       # the step stays in the list for every platform
    assert all(not m.windows_only for m in migrations.HOUSEKEEPING)


def test_the_old_zip_list_is_the_digest_list_of_the_zip_itself():
    """The names and digests of OLD_CCX_ZIP are those of calculix_2.23_4win.zip (SHA-256 87fc4e17…) – checked when the zip is in the development cache."""
    zpath = ROOT / "tools" / "_download" / "calculix_2.23_4win.zip"
    if not zpath.is_file():
        pytest.skip("the original zip is not in tools/_download")
    with zipfile.ZipFile(zpath) as z:
        got = {i.filename.split("/", 1)[1]: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.is_dir()}
    assert REAL_OLD_ZIP == got


def test_a_leftover_variable_naming_a_deleted_ccx_static_is_ignored_with_a_note(tmp_path, monkeypatch):
    from fea_core import config
    gone = tmp_path / "calculix_2.23_4win" / "ccx_static.exe"
    own = tmp_path / "calculix" / "ccx_2.23_windows"
    own.mkdir(parents=True)
    (own / f"ccx_spooles{config.EXE}").write_text("x")
    monkeypatch.setattr(config, "CCX_TOOLS", tmp_path / "calculix")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("FEA_CCX", str(gone))
    assert config.retired_ccx(gone) and config.find_ccx() == own / f"ccx_spooles{config.EXE}"       # the file is gone: the variable is skipped
    note = config.ccx_env_note()
    assert "ccx_static" in note and "ignored" in note and "ccx_spooles.exe" in note
    gone.parent.mkdir()
    gone.write_text("x")                                                                          # a developer's copy that still exists runs
    assert not config.retired_ccx(gone) and config.find_ccx() == gone and config.ccx_env_note() == ""


# ================================================================= an installed ccx_spooles.exe is verified against the package and replaced
def _installed(win, content: bytes):
    d = win / "tools" / "calculix" / "ccx_2.23_windows"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ccx_spooles.exe").write_bytes(content)
    return d


def _package_archive(win):
    arch = win / "tools" / "calculix" / "ccx_2.23_windows_x86_64.zip"
    build_archive(arch)
    with zipfile.ZipFile(arch) as z:
        return arch, z.read("ccx_spooles.exe")


def test_an_installed_ccx_that_matches_the_package_is_left_alone(win):
    arch, exe = _package_archive(win)
    d = _installed(win, exe)
    before = (d / "ccx_spooles.exe").stat().st_mtime_ns
    fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_spooles.exe").read_bytes() == exe and (d / "ccx_spooles.exe").stat().st_mtime_ns == before
    assert not (win / "tools" / "calculix" / ".ccx_win_new").exists()


@pytest.mark.parametrize("what,content", [("a truncated file", b"MZ truncated"), ("another build", b"MZ another build of the solver" * 20), ("an empty file", b"")])
def test_an_installed_ccx_that_differs_from_the_package_is_replaced(win, what, content):
    arch, exe = _package_archive(win)
    d = _installed(win, content)
    fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_spooles.exe").read_bytes() == exe and (d / "manifest.json").is_file()
    assert not (win / "tools" / "calculix" / ".ccx_win_new").exists()


def test_a_damaged_archive_does_not_replace_a_working_ccx(win):
    arch, exe = _package_archive(win)
    d = _installed(win, b"the old but complete build")
    with zipfile.ZipFile(arch) as z:                                            # the executable of the archive does not match the digest of its manifest
        meta = z.read("manifest.json")
    arch.unlink()
    build_archive(arch, members={"ccx_spooles.exe": exe + b"damaged"}, binary_sha256=hashlib.sha256(exe).hexdigest())
    with pytest.raises(RuntimeError, match="does not match the digest of its manifest|neodpovídá kontrolnímu součtu"):
        fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_spooles.exe").read_bytes() == b"the old but complete build"
    assert not (win / "tools" / "calculix" / ".ccx_win_new").exists()


def test_a_running_ccx_that_cannot_be_replaced_is_reported_and_stays(win, monkeypatch):
    arch, exe = _package_archive(win)
    d = _installed(win, b"old build in use")
    real_replace = fs.os.replace

    def locked(src, dst, *a, **k):
        if Path(dst).name == "ccx_spooles.exe":
            raise PermissionError("in use")
        return real_replace(src, dst, *a, **k)
    monkeypatch.setattr(fs.os, "replace", locked)
    with pytest.raises(RuntimeError, match="cannot be replaced|nejde nahradit"):
        fs.install_ext_ccx("ccx_static")
    assert (d / "ccx_spooles.exe").read_bytes() == b"old build in use" and not (win / "tools" / "calculix" / ".ccx_win_new").exists()


def test_without_an_archive_an_installed_ccx_is_checked_against_its_own_manifest(win):
    exe = b"the whole solver"
    d = _installed(win, exe)
    (d / "manifest.json").write_text(json.dumps({"binary_sha256": hashlib.sha256(exe).hexdigest()}), encoding="utf-8")
    fs.install_ext_ccx("ccx_static")                                           # fine: nothing to compare with but its own manifest, and that matches
    (d / "ccx_spooles.exe").write_bytes(exe[:5])                                # truncated, and there is no archive to replace it from
    with pytest.raises(RuntimeError, match="does not match the digest of its manifest|neodpovídá kontrolnímu součtu"):
        fs.install_ext_ccx("ccx_static")


# ================================================================= LGPL-2.1: libquadmath is inside the Windows binary
def _bash():
    for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if Path(cand).is_file():
            return cand
    b = shutil.which("bash")
    return b if b and "system32" not in b.lower() else None


MAP_HEAD = "Archive member included to satisfy reference by file (symbol)\n\n"
MAP_TAIL = "\nDiscarded input sections\n\n .text 0x0 0x0 x.o\n"
MAP_KNOWN = (
    "ccx_2.23.a(main.o)\n                              ccx_2.23.o (main)\n"
    "W:/work/SPOOLES.2.2/spooles.a(Graph.o)\n                              ccx_2.23.a(spooles.o) (Graph_init1)\n"
    "C:/x/msys64/ucrt64/bin/../lib/gcc/x86_64-w64-mingw32/16.2.0/../../../../lib\\libgfortran.a(write.o)\n                              ccx_2.23.a(a.o) (_gfortran_st_write)\n"
    "C:/x/msys64/ucrt64/bin/../lib/gcc/x86_64-w64-mingw32/16.2.0/../../../../lib\\libquadmath.a(printf_fp.o)\n"
    "                              C:/x/lib\\libgfortran.a(write.o) (quadmath_snprintf)\n")


def _audit(tmp_path, body: str, texts: bool = True):
    """Runs audit_static_libs of build_ccx.sh in bash on a synthetic link map → (return code, output, members)."""
    bash = _bash()
    if not bash:
        pytest.skip("no bash")
    sh = (WIN / "build_ccx.sh").read_text(encoding="utf-8")
    snippet = sh[sh.index("STATIC_LIBS="):sh.index('audit_static_libs "$MAP"')]
    script_dir = tmp_path / "installer" / "windows"
    gcc = tmp_path / "installer" / "license_texts" / "gcc-runtime"
    script_dir.mkdir(parents=True)
    gcc.mkdir(parents=True)
    if texts:
        for n in ("libquadmath_COPYING.LIB", "NOTICE-libquadmath.txt"):
            (gcc / n).write_text("text", encoding="utf-8")
    mp_ = tmp_path / "ccx.map"
    mp_.write_text(body, encoding="utf-8")
    script = f'SCRIPT_DIR="{script_dir.as_posix()}"\n{snippet}\naudit_static_libs "{mp_.as_posix()}" "{(tmp_path / "members.txt").as_posix()}" "{(tmp_path / "counts.txt").as_posix()}"\n'
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    members = (tmp_path / "members.txt").read_text(encoding="utf-8") if (tmp_path / "members.txt").is_file() else ""
    return r.returncode, r.stdout + r.stderr, members


def test_the_static_library_audit_finds_libquadmath_by_the_link_map(tmp_path):
    rc, out, members = _audit(tmp_path, MAP_HEAD + MAP_KNOWN + MAP_TAIL)
    assert rc == 0, out
    assert "libquadmath.a printf_fp.o" in members and "libgfortran.a write.o" in members and "spooles.a Graph.o" in members
    assert "1 libquadmath.a" in (tmp_path / "counts.txt").read_text(encoding="utf-8")


def test_the_static_library_audit_stops_at_a_library_nobody_has_looked_at(tmp_path):
    """A change of the toolchain that pulls in another library (libstdc++, libssp ...) must not go unnoticed."""
    rc, out, _ = _audit(tmp_path, MAP_HEAD + MAP_KNOWN + "C:/x/lib\\libstdc++.a(new_op.o)\n                              ccx_2.23.a(b.o) (_Znwy)\n" + MAP_TAIL)
    assert rc == 1 and "libstdc++.a" in out and "not in STATIC_LIBS" in out


def test_the_static_library_audit_wants_the_licence_texts_of_libquadmath(tmp_path):
    rc, out, _ = _audit(tmp_path, MAP_HEAD + MAP_KNOWN + MAP_TAIL, texts=False)
    assert rc == 1 and "libquadmath" in out and "NOTICE-libquadmath.txt" in out
    rc, out, _ = _audit(tmp_path / "second", MAP_HEAD + "C:/x/lib\\libgcc.a(a.o)\n                              ccx_2.23.o (x)\n" + MAP_TAIL, texts=False)
    assert rc == 0, out                                                           # without libquadmath in the binary the texts are not demanded by this audit


def test_the_static_library_audit_stops_when_the_map_is_empty_or_unreadable(tmp_path):
    rc, out, _ = _audit(tmp_path, "nothing useful in here\n")
    assert rc == 1 and "no archive members" in out


def test_the_licence_texts_of_libquadmath_are_shipped_and_say_it_is_inside_the_windows_binary():
    texts = ROOT / "installer" / "license_texts" / "gcc-runtime"
    notice = (texts / "NOTICE-libquadmath.txt").read_text(encoding="utf-8")
    assert "GNU Lesser General Public" in (texts / "libquadmath_COPYING.LIB").read_text(encoding="utf-8")
    assert notice.count("Files: ") == 19 and "Free Software Foundation" in notice and "Sun Microsystems" in notice
    e = next(x for x in cl.EXTERNAL if x["slug"] == "libgomp-libquadmath")
    assert "not inside" not in e["use"] and "AND libquadmath" in e["use"] and "LGPL-2.1" in e["licence"] and "Linux package only" not in e["licence"]
    assert "NOTICE-libquadmath.txt" in e["licence"] and e["texts"] == ["gcc-runtime/*"]
    for rel in ("THIRD_PARTY_LICENSES.md", "docs/projekt/licences_detail.md", "fea_gui/help/index.html", "fea_gui/help/cs/index.html", "installer/license_texts/SOURCES.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "libquadmath is NOT inside" not in text and "libquadmath on Linux)" not in text and "jen v balíčku pro Linux" not in text and "only in the Linux package" not in text, rel
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cl.write_sources(Path(td), ROOT, "2026-11-19")
        md = (Path(td) / "SOURCES.md").read_text(encoding="utf-8")
    assert "libquadmath is inside `ccx_spooles.exe`" in md and "Relinking with a modified libquadmath" in md and "-LibQuadmathSource" in md
    assert "sources/mingw-w64-gcc-16.2.0-7.src.tar.zst" in md and "recipe" in md and "e3d18511f9759d455cae20c34703eea4a0723cf3" in md


# ================================================================= the local archive of the downloads
def test_the_archive_plan_covers_the_lock_and_every_pinned_source():
    items = fa.plan(ROOT)
    names = [n for n, _, _ in items]
    assert len(names) == len(set(names)) and all(re.fullmatch(r"[0-9a-f]{64}", d) for _, d, _ in items)
    lock = [l.split() for l in (WIN / "ccx_toolchain.lock").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    assert {u.rsplit("/", 1)[1] for _, _, u in lock} <= set(names)                       # the base tarball and all the packages
    assert {"ccx_2.23.src.tar.bz2", "spooles.2.2.tgz", "arpack-ng-3.9.1.tar.gz", "lapack-3.12.0.tar.gz", "ccx_2.23.SPOOLEScorrection.tar.bz2"} <= set(names)   # ARPACK and LAPACK tag tarballs too
    assert any(n.startswith("mingw-w64-gcc-16.2.0-7.src") for n in names) and any(n.startswith("mingw-w64-crt-") for n in names)


class _Resp:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def read(self, n):
        chunk, self.pos = self.data[self.pos:self.pos + n], self.pos + n
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_archive_is_verified_filled_and_never_trusts_a_wrong_file(tmp_path, monkeypatch):
    good, other = b"good content", b"other content"
    items = [("a.bin", hashlib.sha256(good).hexdigest(), ["https://one.invalid/a", "https://two.invalid/a"]),
             ("b.bin", hashlib.sha256(good).hexdigest(), ["https://one.invalid/b"]),
             ("c.bin", hashlib.sha256(good).hexdigest(), ["https://one.invalid/c"])]
    (tmp_path / "b.bin").write_bytes(other)                                              # wrong digest
    (tmp_path / "c.bin").write_bytes(good)                                               # right
    assert fa.verify(tmp_path, items) == (["c.bin"], ["a.bin"], ["b.bin"])
    served = {"https://one.invalid/a": other, "https://two.invalid/a": good, "https://one.invalid/b": good}

    def opener(req, timeout=0):
        return _Resp(served[req.full_url])
    monkeypatch.setattr(fa, "plan", lambda root=ROOT: items)
    monkeypatch.setattr(fa.urllib.request, "urlopen", opener)
    assert fa.main(["--dir", str(tmp_path), "--check"]) == 1                              # --check downloads nothing
    assert (tmp_path / "b.bin").read_bytes() == other
    assert fa.main(["--dir", str(tmp_path)]) == 0
    assert all((tmp_path / n).read_bytes() == good for n in ("a.bin", "b.bin", "c.bin"))   # the second source for a, b replaced
    sums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sums == [f"{hashlib.sha256(good).hexdigest()}  {n}" for n in ("a.bin", "b.bin", "c.bin")]
    assert fa.main(["--dir", str(tmp_path), "--check"]) == 0
    assert not list(tmp_path.glob("*.part"))


def test_a_source_that_never_matches_stops_the_download(tmp_path, monkeypatch):
    items = [("a.bin", hashlib.sha256(b"right").hexdigest(), ["https://one.invalid/a"])]
    monkeypatch.setattr(fa, "plan", lambda root=ROOT: items)
    monkeypatch.setattr(fa.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"wrong"))
    with pytest.raises(SystemExit, match="could not get a.bin"):
        fa.main(["--dir", str(tmp_path)])
    assert not (tmp_path / "a.bin").exists() and not list(tmp_path.glob("*.part"))


# ================================================================= SPOOLES needs a deck without empty rows (PaStiX tolerated them)
def test_the_auxiliary_node_of_a_normal_elastic_support_is_fixed_in_its_empty_dofs(tmp_path):
    """The auxiliary node of a normal elastic support has stiffness in DOF 1 only (SPRING1 + the equation u_g = n·u); its DOF 2 and 3 are empty rows. PaStiX
    solved the matrix anyway, SPOOLES (the only solver of the own CalculiX) stops with "matrix found to be singular" – so every step fixes them."""
    from test_loads import bar_mesh, bar_study
    from fea_core.ccx.inp import write_inp
    from fea_core.study import ElasticSupport, Force, LoadCase
    m, st = bar_mesh(), bar_study()
    st.constraints = [ElasticSupport("x100", 2.0, "normal")]
    st.load_cases = [LoadCase("A", [Force("x100", [0.0, 0.0, -1.0])]), LoadCase("B", [Force("x100", [0.0, 0.0, 1.0])])]
    write_inp(st, m, tmp_path / "a.inp")
    txt = (tmp_path / "a.inp").read_text()
    blocks = [b for b in txt.split("*BOUNDARY")[1:]]
    assert len(blocks) >= 2 and all("NSPR, 2, 3" in b.split("*")[0] for b in blocks)        # in the block of every step (OP=NEW resets the earlier ones)
    assert "*NSET, NSET=NSPR" in txt
    st.constraints = [ElasticSupport("x100", 2.0, "all")]                                    # direction "all": springs on the nodes themselves, no auxiliary node
    write_inp(st, m, tmp_path / "b.inp")
    assert "NSPR" not in (tmp_path / "b.inp").read_text()

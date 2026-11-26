#!/usr/bin/env bash
# Builds CalculiX 2.23 for Windows x86_64 (ccx_spooles.exe) from the pinned sources, with a pinned MSYS2 UCRT64 toolchain that is unpacked into the work
# folder (no installation, no administrator rights; pacman is not used). Everything that is downloaded is pinned by a SHA-256 digest: the sources in
# this script, the MSYS2 packages in installer/windows/ccx_toolchain.lock (installer/windows/make_ccx_locks.py regenerates it).
#
# Normally started by installer/windows/build_ccx.ps1 inside the MSYS2 shell of the work folder; the paths are MSYS paths:
#
#   usr/bin/bash.exe build_ccx.sh OUTPUT [WORKDIR]
#
# What the binary contains (licences: see installer/license_texts/SOURCES.md and the manifest.json that is written to OUTPUT):
#   CalculiX 2.23 (GPL-2.0, with two patches: patches/ccx_2.23_ucrt_build.patch = a build fix, patches/ccx_2.23_spooles_serial_solve.patch = a change of
#   the solver logic, see below)  +  SPOOLES 2.2 (public domain; its multi-threaded objects are inside: the factorization runs on CCX_NPROC_EQUATION_SOLVER
#   threads, the triangular solve is the serial one - the parallel solve of SPOOLES is a data race and returns wrong results, which the patch removes; the
#   factors of a multi-threaded factorization are CHECKED by a test solve (a thread can silently lose an update) and repeated when the check fails;
#   the binary says so in `ccx -v`: FEA_SPOOLES_FACTOR_CHECK=2, and FEA_SPOOLES_ORDERED_ASSEMBLY=1 for the ordered SPOOLES (the link fails without it), see spooles_threads_safe in fea_core/ccx/runner.py)
#   faer 0.24.4 (MIT) with the crates of installer/faer_ccx/Cargo.lock (libfea_faer.a, built below with the pinned Rust toolchain of ccx_rust.lock; patches/ccx_2.23_faer.patch
#   + faer.c / faer.h of installer/faer_ccx/ccx, GPL-2.0 like CalculiX): the direct solver of symmetric (Cholesky / LBL^T, the nested dissection of SPOOLES as the ordering) and
#   unsymmetric (LU) matrices; every factorization is checked by a test solve, SPOOLES (above) takes over when it fails; the marker FEA_FAER=1 in `ccx -v` and in the file
#   ARPACK-ng 3.9.1 (BSD-3)  +  reference BLAS / LAPACK 3.12.0 (modified BSD), all of them statically
#   GCC runtime libgcc / libgfortran / libgomp (GPL-3.0 with the runtime library exception), mingw-w64 CRT and winpthreads (ZPL-2.1 / MIT)
#   libquadmath (LGPL-2.1 or later): 19 objects, pulled in by libgfortran (write.o needs quadmath_snprintf for the formatted output of REAL(16), which
#   the program never uses - but the linker cannot know that). LGPL-2.1 section 6 for a statically linked work: the package carries the source of the
#   library (the MSYS2 source package of GCC, licenses/sources/), the source of everything else and THIS script, so the user can modify the library and
#   link the program again: FEA_LIBQUADMATH_SOURCE (a modified source tree of GCC, built here) or FEA_LIBQUADMATH_DIR (a ready libquadmath.a) makes the link use
#   that library instead of the one of the toolchain (installer/windows/build_ccx.ps1 -LibQuadmathSource / -LibQuadmath; the manifest then says so). The
#   licence texts of the library are checked below.
# NOT inside: TAUCS (AMD, COLAMD, GENMMD), PaStiX, Scotch, METIS, hwloc, OpenBLAS (checked on the binary below). No DLL besides the system ones (the universal CRT of Windows 10+).
# The static libraries that contributed objects are read from the link map and must all be in STATIC_LIBS below (an unknown one stops the build).
# BLAS / LAPACK: the reference implementation. SPOOLES has no BLAS at all and ARPACK uses level 1 / 2 only: the same decks run as fast as with the
# OpenBLAS inside the third-party ccx_static.exe (SPOOLES against SPOOLES: 0.8-1.0 x its time), so OpenBLAS (a 25 MB source, hours of build time for
# the kernels of all CPU generations, threads of its own next to the OpenMP ones) buys nothing here.
#
# Environment:
#   FEA_BUILD_JOBS=N       parallel compilations (default 4)
#   SOURCE_DATE_EPOCH      the date written into the banner and the .frd header (default: fixed, so the binary is repeatable)
#   FEA_ARCHIVE=DIR        the local archive of everything that is downloaded (the MSYS2 packages, the sources; default: the folder "archive" next to the
#                          work folder): a file in it whose SHA-256 matches the pin is used without the network, what is downloaded is stored there
#   FEA_OFFLINE=1          never download: a file that is not in the archive stops the build
#   FEA_LIBQUADMATH_SOURCE=T  the unpacked source tree of GCC 16.2.0 with a (modified) libquadmath/: it is built with the toolchain and linked instead of the library of the
#                          toolchain (LGPL-2.1 section 6: relinking with a modified library); FEA_LIBQUADMATH_DIR=D uses a ready D/libquadmath.a the same way
#   FEA_SKIP_RUNTIME_SOURCES=1  do not fetch the MSYS2 source packages of the run-time libraries (they are not needed to build; the release package ships them)
set -Eeuo pipefail
trap 'echo "ERROR: build_ccx.sh stopped at line $LINENO (exit code $?)" >&2' ERR     # a failing command that set -e turns into an exit says where

MSYS_SPEC="make patch perl diffutils"
UCRT_SPEC="gcc gcc-fortran binutils libgomp"

OUT="${1:?usage: build_ccx.sh OUTPUT [WORKDIR]}"
WORK="${2:-/tmp/fea_ccx_build}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCX_VER=2.23
JOBS="${FEA_BUILD_JOBS:-4}"
SDE="${SOURCE_DATE_EPOCH:-1794960000}"          # 2026-11-18 00:00:00 UTC
mkdir -p "$OUT" "$WORK"
OUT="$(cd "$OUT" && pwd)"
WORK="$(cd "$WORK" && pwd)"
ARCHIVE="${FEA_ARCHIVE:-$(dirname "$WORK")/archive}"
mkdir -p "$ARCHIVE"
ARCHIVE="$(cd "$ARCHIVE" && pwd)"
DL="$ARCHIVE"                                      # the sources are looked up in (and downloaded into) the archive
M() { cygpath -m "$1"; }                           # the Windows form (W:/dir) for the native compilers; MSYS2_ARG_CONV_EXCL=* keeps arguments as they are
export PATH="/ucrt64/bin:/usr/bin" MSYSTEM=UCRT64 MSYS2_ARG_CONV_EXCL='*' LANG=C LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH="$SDE"

: > "$WORK/downloads.tsv"
get_mirrors() {  # get_mirrors FILE SHA256 URL [BACKUP_URL...] - the file of the archive if its digest matches, else the first source that passes the digest check
    local file="$1" sum="$2" url; shift 2
    case "$file" in *.src.tar.zst|*/rust-src-*.tar.xz) [ "${FEA_SKIP_RUNTIME_SOURCES:-0}" != 1 ] || return 0 ;; esac     # the source packages of the run-time libraries (and rust-src) are not needed to build
    [ "${RECORD:-1}" = 0 ] || printf '%s\t%s\t%s\n' "$(basename "$file")" "$sum" "$1" >> "$WORK/downloads.tsv"      # the manifest lists the pinned sources
    if [ -s "$file" ] && echo "$sum  $file" | sha256sum -c --status -; then return 0; fi   # also verify what was downloaded before
    [ "${FEA_OFFLINE:-0}" != 1 ] || { echo "ERROR: $(basename "$file") is not in the archive with the pinned digest and FEA_OFFLINE=1 (run installer/windows/fetch_ccx_archive.py)"; exit 1; }
    for url in "$@"; do
        echo "downloading $url"
        if curl -fsSL --retry 3 -o "$file.part" "$url" && echo "$sum  $file.part" | sha256sum -c --status -; then
            mv "$file.part" "$file"; return 0
        fi
        echo "  failed (or the digest does not match) - trying the next source"
    done
    rm -f "$file.part"; echo "ERROR: could not download $(basename "$file")"; exit 1
}

# ---- 1. the toolchain: the packages of the lock unpacked over the MSYS2 base (this script runs from that very base: its tools are in /usr/bin)
LOCK="$SCRIPT_DIR/ccx_toolchain.lock"
grep -qxF "# msys_spec: $MSYS_SPEC" "$LOCK" && grep -qxF "# ucrt_spec: $UCRT_SPEC" "$LOCK" \
    || { echo "ERROR: ccx_toolchain.lock was made for other package lists - run installer/windows/make_ccx_locks.py"; exit 1; }
STAMP="$(sha256sum "$LOCK" | cut -d' ' -f1)"
if [ "$(cat /.fea_toolchain 2>/dev/null)" != "$STAMP" ]; then
    while read -r kind sum url; do
        case "$kind" in ""|"#"*|base) continue ;; esac
        url="${url%$'\r'}"
        f="$ARCHIVE/$(basename "$url")"
        RECORD=0 get_mirrors "$f" "$sum" "$url"
        tar --zstd -xf "$f" -C / --exclude=.PKGINFO --exclude=.BUILDINFO --exclude=.MTREE --exclude=.INSTALL --exclude=.CHANGELOG
    done < "$LOCK"
    printf '%s' "$STAMP" > /.fea_toolchain
fi
echo "toolchain: $(gcc --version | head -1); $(gfortran --version | head -1); $(ld --version | head -1); $(make --version | head -1)"

# ---- 2. sources (digests verified 18. 11. 2026; a new version = a new digest, otherwise the build stops). One line per file: collect_licenses.py
# reads these lines (build_ccx_win_sources) for licenses/SOURCES.md and for the source archives that make_package.py puts into the packages.
get_mirrors "$DL/ccx_2.23.src.tar.bz2" 9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7 "https://www.dhondt.de/ccx_2.23.src.tar.bz2"
get_mirrors "$DL/ccx_2.23.SPOOLEScorrection.tar.bz2" 15528f09312dc7605c0600358f5e6de12945449f249dfcfca7417eed6c220b93 "https://www.dhondt.de/ccx_2.23.SPOOLEScorrection.tar.bz2"
get_mirrors "$DL/spooles.2.2.tgz" a84559a0e987a1e423055ef4fdf3035d55b65bbe4bf915efaa1a35bef7f8c5dd "https://www.netlib.org/linalg/spooles/spooles.2.2.tgz" "https://deb.debian.org/debian/pool/main/s/spooles/spooles_2.2.orig.tar.gz"
get_mirrors "$DL/arpack-ng-3.9.1.tar.gz" f6641deb07fa69165b7815de9008af3ea47eb39b2bb97521fbf74c97aba6e844 "https://github.com/opencollab/arpack-ng/archive/refs/tags/3.9.1.tar.gz"
get_mirrors "$DL/lapack-3.12.0.tar.gz" eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz"
# The sources of the libraries that are linked into the binary but not compiled here (GCC runtime libgcc / libgfortran / libgomp / libquadmath, mingw-w64 CRT,
# winpthreads): the MSYS2 source packages of the versions in ccx_toolchain.lock (PKGBUILD, patches, the upstream archive; GPL / LGPL: they travel with the
# binary, installer/make_package.py puts them into licenses/sources/). The mingw-w64 packages carry the same upstream commit (version 14.0.0.r598.ge3d18511f).
get_mirrors "$DL/mingw-w64-gcc-16.2.0-7.src.tar.zst" bdd1b2cd820cd67d38b42a174180bdd589b80b97af1a8b042d6c4da9176bac93 "https://repo.msys2.org/mingw/sources/mingw-w64-gcc-16.2.0-7.src.tar.zst" "https://mirror.msys2.org/mingw/sources/mingw-w64-gcc-16.2.0-7.src.tar.zst"
get_mirrors "$DL/mingw-w64-crt-14.0.0.r598.ge3d18511f-1.src.tar.zst" a6df84c851473c2b293f36a8864b52980a5d875b1c7e2fbfb01dece3f400256a "https://repo.msys2.org/mingw/sources/mingw-w64-crt-14.0.0.r598.ge3d18511f-1.src.tar.zst" "https://mirror.msys2.org/mingw/sources/mingw-w64-crt-14.0.0.r598.ge3d18511f-1.src.tar.zst"
get_mirrors "$DL/mingw-w64-winpthreads-14.0.0.r598.ge3d18511f-1.src.tar.zst" 1da3dad4f6916d0c23e8fe76c38ed79946fb32078f94047c0475229b8112a1db "https://repo.msys2.org/mingw/sources/mingw-w64-winpthreads-14.0.0.r598.ge3d18511f-1.src.tar.zst" "https://mirror.msys2.org/mingw/sources/mingw-w64-winpthreads-14.0.0.r598.ge3d18511f-1.src.tar.zst"
get_mirrors "$DL/mingw-w64-headers-14.0.0.r598.ge3d18511f-1.src.tar.zst" b0c5391bca62b78df4fef32bc87eb94c5c29a867bfd6d3e64e0ddfb9e078b31e "https://repo.msys2.org/mingw/sources/mingw-w64-headers-14.0.0.r598.ge3d18511f-1.src.tar.zst" "https://mirror.msys2.org/mingw/sources/mingw-w64-headers-14.0.0.r598.ge3d18511f-1.src.tar.zst"
# the version of every toolchain package that is linked in must be the one of these source packages (the lock and the pins above change together)
for pair in "libgcc-16.2.0-7:gcc-16.2.0-7" "libgfortran-16.2.0-7:gcc-16.2.0-7" "libgomp-16.2.0-7:gcc-16.2.0-7" "libquadmath-16.2.0-7:gcc-16.2.0-7" \
            "crt-14.0.0.r598.ge3d18511f-1:crt-14.0.0.r598.ge3d18511f-1" "winpthreads-14.0.0.r598.ge3d18511f-1:winpthreads-14.0.0.r598.ge3d18511f-1"; do
    [ "${FEA_SKIP_RUNTIME_SOURCES:-0}" != 1 ] || break
    grep -q "/mingw-w64-ucrt-x86_64-${pair%%:*}-any.pkg.tar.zst" "$LOCK" && [ -f "$DL/mingw-w64-${pair##*:}.src.tar.zst" ] \
        || { echo "ERROR: the toolchain package ${pair%%:*} of ccx_toolchain.lock does not match the source package pinned in build_ccx.sh (mingw-w64-${pair##*:})"; exit 1; }
done

cd "$WORK"
# the compilers see only relative paths of the sources (cd into the directory, compile "foo.c"), so no path of the work folder gets into the binary;
# the prefix map is the second line of defence (absolute -I paths in assert messages; debug information is not written at all)
PMAP="-ffile-prefix-map=$(M "$WORK")=/build"
CF_COMMON="-O2 -w $PMAP -fno-ident"
# gcc 14+ turns old C constructs (SPOOLES, CalculiX) into errors
PERMISSIVE="-Wno-error=int-conversion -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=return-mismatch"
FFL="-O2 -fallow-argument-mismatch -w $PMAP -fno-ident"
CF_SPOOLES="$CF_COMMON"                                # the flags of SPOOLES (C). -O3 -funroll-loops (19. 11. 2026, a machine that other programs kept busy, 4 threads, paired runs): results bit-identical on
                                                       # every captured gate deck, 3-11 % faster (clear only for the contact and shell decks) - NOT used, because it changes the verified binary for a small gain;
                                                       # a one-word change here if the owner wants it (docs/projekt/F13_plan_speed_memory_licences_native.md, "Step 5d - review follow-up")

# ---- 3. reference BLAS / LAPACK 3.12.0 (static); the timer of LAPACK is the Fortran CPU_TIME intrinsic
# The libraries made here are kept between runs (a repeat takes minutes instead of 15-25) – but only while everything they depend on is the same: the toolchain
# (the lock), the flags and the source archive. `stamp NAME PARTS...` = a short digest of those; a library whose stamp file differs is built again.
stamp() { printf '%s|' "$@" | sha256sum | cut -c1-16; }
sumof() { sha256sum "$1" | cut -c1-16; }
BLASDIR="$WORK/blas_reference"
BLASSTAMP="$(stamp "$STAMP" "$FFL" "TIMER=INT_CPU_TIME" "$(sumof "$DL/lapack-3.12.0.tar.gz")")"
if [ "$(cat "$BLASDIR/.stamp" 2>/dev/null)" != "$BLASSTAMP" ]; then
    src="$WORK/lapack-3.12.0"
    rm -rf "$src" "$BLASDIR"; mkdir -p "$BLASDIR"
    tar -xzf "$DL/lapack-3.12.0.tar.gz" -C "$WORK"
    (cd "$src" && cp make.inc.example make.inc \
        && sed -i "s|^FC = .*|FC = gfortran|;s|^CC = .*|CC = gcc|;s|^FFLAGS = .*|FFLAGS = $FFL -frecursive|;s|^FFLAGS_DRV = .*|FFLAGS_DRV = \$(FFLAGS)|;s|^FFLAGS_NOOPT = .*|FFLAGS_NOOPT = -O0 -frecursive -fallow-argument-mismatch|;s|^TIMER *=.*|TIMER = INT_CPU_TIME|;s|^LDFLAGS *=.*||" make.inc \
        && (make -C BLAS/SRC -j"$JOBS" && make -C SRC -j"$JOBS") > "$WORK/lapack_build.log" 2>&1 \
        || { tail -40 "$WORK/lapack_build.log"; exit 1; })
    cp "$src"/librefblas.a "$BLASDIR/libblas.a"; cp "$src"/liblapack.a "$BLASDIR/liblapack.a"
    printf 'reference BLAS / LAPACK 3.12.0 (-O2)' > "$BLASDIR/.desc"
    printf '%s' "$BLASSTAMP" > "$BLASDIR/.stamp"
fi
BLASLIBS="$(M "$BLASDIR")/liblapack.a $(M "$BLASDIR")/libblas.a"

# ---- 4. ARPACK-ng 3.9.1: the sources of its cmake build (SRC/*.f, dbgini.f, staini.f and the listed UTIL files); only double precision is used
ARP="$WORK/arpack-ng-3.9.1"
ARPSTAMP="$(stamp "$STAMP" "$FFL" "$(sumof "$DL/arpack-ng-3.9.1.tar.gz")")"
if [ ! -f "$WORK/libarpack.a" ] || [ "$(cat "$WORK/libarpack.stamp" 2>/dev/null)" != "$ARPSTAMP" ]; then
    rm -rf "$ARP" "$WORK/arpack_obj" "$WORK/libarpack.a"; mkdir -p "$WORK/arpack_obj"
    tar -xzf "$DL/arpack-ng-3.9.1.tar.gz" -C "$WORK"
    (
        cd "$ARP"
        files="dbgini.f staini.f $(ls SRC/*.f) UTIL/icnteq.f UTIL/icopy.f UTIL/iset.f UTIL/iswap.f UTIL/ivout.f UTIL/second_NONE.f UTIL/svout.f UTIL/smout.f UTIL/dvout.f UTIL/dmout.f UTIL/cvout.f UTIL/cmout.f UTIL/zvout.f UTIL/zmout.f"
        # the objects go next to the work folder root, named with a Windows path (the compiler is a native program)
        printf '%s\n' $files | xargs -P "$JOBS" -I{} sh -c 'gfortran '"$FFL"' -ISRC -I. -c {} -o "'"$(M "$WORK")"'/arpack_obj/$(basename {} .f).o"'
        rm -f "$WORK/libarpack.a"; cd "$WORK/arpack_obj" && ar rcs ../libarpack.a ./*.o
    ) > "$WORK/arpack_build.log" 2>&1 || { tail -30 "$WORK/arpack_build.log"; exit 1; }
    printf '%s' "$ARPSTAMP" > "$WORK/libarpack.stamp"
fi

# ---- 5. SPOOLES 2.2 (+ the correction of dhondt.de): gcc, the MT objects are part of spooles.a
SPDIR="$WORK/SPOOLES.2.2"
SPSTAMP="$(stamp "$STAMP" "$CF_SPOOLES" "$PERMISSIVE" "$(sumof "$DL/spooles.2.2.tgz")" "$(sumof "$DL/ccx_2.23.SPOOLEScorrection.tar.bz2")" "$(sumof "$SCRIPT_DIR/patches/spooles_2.2_ordered_assembly.patch")")"
if [ ! -f "$SPDIR/spooles.a" ] || [ "$(cat "$SPDIR/.fea_stamp" 2>/dev/null)" != "$SPSTAMP" ]; then
    rm -rf "$SPDIR"; mkdir -p "$SPDIR"
    tar -xzf "$DL/spooles.2.2.tgz" -C "$SPDIR"
    mkdir -p "$WORK/spcorr" && tar -xjf "$DL/ccx_2.23.SPOOLEScorrection.tar.bz2" -C "$WORK/spcorr"
    cp "$WORK/spcorr/CalculiX/ccx_$CCX_VER/SPOOLES.2.2/I2Ohash/src/util.c" "$SPDIR/I2Ohash/src/util.c"
    # reproducible (ordered) assembly of the multi-threaded factorization (installer/windows/patches/spooles_2.2_ordered_assembly.patch; SPOOLES is public domain)
    (cd "$SPDIR" && patch -p1 -i "$SCRIPT_DIR/patches/spooles_2.2_ordered_assembly.patch" > "$WORK/spooles_patch.log" 2>&1) || { cat "$WORK/spooles_patch.log"; exit 1; }
    (
        cd "$SPDIR"
        sed -i "s|^  CC = .*|  CC = gcc -fcommon $CF_SPOOLES $PERMISSIVE|;s|^  RANLIB = echo|  RANLIB = ranlib|;s|^# RANLIB = ranlib|  RANLIB = ranlib|;s|^  THREAD_LIBS = .*|  THREAD_LIBS = -lpthread|" Make.inc
        sed -i 's|drawTree.c|draw.c|g' Tree/src/makeGlobalLib
        make -s lib > "$WORK/spooles_build.log" 2>&1 || true
        [ -f spooles.a ] || { tail -30 "$WORK/spooles_build.log"; echo "SPOOLES did not compile"; exit 1; }
        # the top-level makefile has the multi-threaded objects (MT/) commented out: add them to spooles.a (ccx is compiled with -DUSE_MT=1 and takes
        # the serial code when CCX_NPROC_EQUATION_SOLVER is 1; with more threads the factorization is multi-threaded and, by the patch
        # ccx_2.23_spooles_serial_solve.patch, the triangular solve serial)
        (cd MT/src && make -s -f makeGlobalLib) >> "$WORK/spooles_build.log" 2>&1 || { tail -30 "$WORK/spooles_build.log"; echo "SPOOLES MT did not compile"; exit 1; }
        ar t spooles.a | grep -q "MT_factorMT.o" || { echo "SPOOLES: the MT objects are missing in spooles.a"; exit 1; }
    )
    printf '%s' "$SPSTAMP" > "$SPDIR/.fea_stamp"
fi

# ---- 5b. LGPL-2.1 section 6 (libquadmath is linked statically): the user may modify the library and link the program again. FEA_LIBQUADMATH_SOURCE = the
# unpacked source tree of GCC 16.2.0 (gcc-16.2.0/, from the source package that licenses/sources/ carries) with a modified libquadmath/: the parts of it that the
# library needs are copied into the work folder (the native compiler cannot read an MSYS path of another drive, and configure does not handle spaces), the
# library is built there with the toolchain of the lock and linked instead of the library of the toolchain. FEA_LIBQUADMATH_DIR = a folder with a libquadmath.a that was built some other way.
if [ -n "${FEA_LIBQUADMATH_SOURCE:-}" ]; then
    [ -f "$FEA_LIBQUADMATH_SOURCE/libquadmath/configure" ] || { echo "ERROR: FEA_LIBQUADMATH_SOURCE=$FEA_LIBQUADMATH_SOURCE is not an unpacked GCC source tree (libquadmath/configure is missing)"; exit 1; }
    LQB="$WORK/libquadmath_build"; LQS="$WORK/libquadmath_src"; rm -rf "$LQB" "$LQS"; mkdir -p "$LQB" "$LQS"
    for f in libquadmath include config config.guess config.sub install-sh ltmain.sh missing depcomp compile mkinstalldirs config.rpath ar-lib test-driver \
             libtool.m4 ltoptions.m4 ltsugar.m4 ltversion.m4 'lt~obsolete.m4' ltgcc.m4 config-ml.in multilib.am; do
        [ ! -e "$FEA_LIBQUADMATH_SOURCE/$f" ] || cp -a "$FEA_LIBQUADMATH_SOURCE/$f" "$LQS/"
    done
    echo "libquadmath: building the library from the source tree given in FEA_LIBQUADMATH_SOURCE"
    (cd "$LQB" && LD=ld NM=nm AR=ar RANLIB=ranlib AS=as DLLTOOL=dlltool OBJDUMP=objdump STRIP=strip CC=gcc \
        ../libquadmath_src/libquadmath/configure --disable-shared --enable-static > configure.log 2>&1 \
        && make -j"$JOBS" > make.log 2>&1) || { tail -30 "$LQB/configure.log" "$LQB/make.log" 2>/dev/null; echo "ERROR: libquadmath did not build"; exit 1; }
    FEA_LIBQUADMATH_DIR="$LQB/.libs"
fi

# ---- 5c. faer (F17 stage C): the Rust static library libfea_faer.a of installer/faer_ccx = a C interface over faer 0.24.4 (MIT), built with the pinned Rust toolchain
# of ccx_rust.lock from the crates of installer/faer_ccx/Cargo.lock (every crate is downloaded with the SHA-256 digest of the lock file, then `cargo build --locked
# --offline` runs against a directory source made of them - the layout of `cargo vendor`). The result is linked into ccx_spooles.exe (faer.c, patch ccx_2.23_faer.patch).
FAERDIR="$(cd "$SCRIPT_DIR/../faer_ccx" && pwd)"
RUSTLOCK="$SCRIPT_DIR/ccx_rust.lock"
RUSTSTAMP="$(sha256sum "$RUSTLOCK" | cut -d' ' -f1)"
RUSTROOT="$WORK/rust"
RUST_COMMIT="$(sed -n 's/^# rust_commit: *//p' "$RUSTLOCK" | tr -d '\r' | head -1)"
mkdir -p "$DL/crates"
while read -r kind sum url; do                       # the toolchain tarballs (checked every time; recorded: the library of the Rust standard library is linked in, rust-src is its source)
    case "$kind" in
        rust) RECORD=0 get_mirrors "$DL/$(basename "${url%$'\r'}")" "$sum" "${url%$'\r'}" ;;
        rustsrc) get_mirrors "$DL/$(basename "${url%$'\r'}")" "$sum" "${url%$'\r'}" ;;
    esac
done < "$RUSTLOCK"
if [ "$(cat "$RUSTROOT/.fea_rust" 2>/dev/null)" != "$RUSTSTAMP" ]; then      # unpack the components into one prefix (what the rust-installer script of each tarball would do)
    rm -rf "$RUSTROOT" "$WORK/rust_unpack"; mkdir -p "$RUSTROOT" "$WORK/rust_unpack"
    while read -r kind sum url; do
        [ "$kind" = rust ] || continue
        url="${url%$'\r'}"
        tar -xJf "$DL/$(basename "$url")" -C "$WORK/rust_unpack"
    done < "$RUSTLOCK"
    for d in "$WORK"/rust_unpack/*/; do
        while read -r comp; do [ -z "$comp" ] || cp -a "$d$comp/." "$RUSTROOT/"; done < "${d}components"
    done
    rm -rf "$WORK/rust_unpack"
    printf '%s' "$RUSTSTAMP" > "$RUSTROOT/.fea_rust"
fi
RUSTC_V="$("$RUSTROOT/bin/rustc.exe" -vV)"
case "$RUSTC_V" in *"commit-hash: $RUST_COMMIT"*) ;; *) echo "ERROR: rustc of ccx_rust.lock is not the pinned release (commit $RUST_COMMIT): $RUSTC_V"; exit 1 ;; esac
# the crates: name, version, SHA-256 of the .crate file from Cargo.lock (packages with a registry source)
awk '/^\[\[package\]\]/{n="";v="";s=""} /^name = /{gsub(/"/,"",$3);n=$3} /^version = /{gsub(/"/,"",$3);v=$3} /^source = /{s=$3} /^checksum = /{gsub(/"/,"",$3); if(n!=""&&s!="") print n, v, $3}' "$FAERDIR/Cargo.lock" > "$WORK/faer_crates.txt"
[ "$(wc -l < "$WORK/faer_crates.txt")" -ge 20 ] || { echo "ERROR: Cargo.lock of installer/faer_ccx lists too few crates"; exit 1; }
while read -r cn cv csum; do
    get_mirrors "$DL/crates/$cn-$cv.crate" "$csum" "https://static.crates.io/crates/$cn/$cn-$cv.crate"
done < "$WORK/faer_crates.txt"
FAERLIB="$WORK/libfea_faer.a"
FAERSTAMP="$(stamp "$STAMP" "$RUSTSTAMP" "$(sumof "$FAERDIR/Cargo.lock")" "$(sumof "$FAERDIR/Cargo.toml")" "$(sumof "$FAERDIR/src/lib.rs")")"
if [ ! -f "$FAERLIB" ] || [ "$(cat "$WORK/libfea_faer.stamp" 2>/dev/null)" != "$FAERSTAMP" ]; then
    FB="$WORK/faer_build"
    rm -rf "$FB" "$FAERLIB"
    for _try in 1 2 3 4 5 6; do          # Windows may still hold the deleted tree for a moment (virus scanner, indexer): "Access denied" when the folder is created again
        mkdir -p "$FB/vendor" "$FB/crate/.cargo" "$FB/home" "$FB/target" 2>/dev/null && break
        sleep 5
    done
    [ -d "$FB/target" ] || { echo "ERROR: cannot create $FB (access denied)"; exit 1; }
    while read -r cn cv csum; do
        tar -xzf "$DL/crates/$cn-$cv.crate" -C "$FB/vendor"
        printf '{"files":{},"package":"%s"}' "$csum" > "$FB/vendor/$cn-$cv/.cargo-checksum.json"
    done < "$WORK/faer_crates.txt"
    cp -r "$FAERDIR/Cargo.toml" "$FAERDIR/Cargo.lock" "$FAERDIR/src" "$FB/crate/"
    cat > "$FB/crate/.cargo/config.toml" <<EOF
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$(M "$FB/vendor")"

[net]
offline = true

[build]
target-dir = "$(M "$FB/target")"
EOF
    (
        cd "$FB/crate"
        # the compiler and the linker for the build scripts of the crates (host = target = x86_64-pc-windows-gnu): the unpacked toolchain and the gcc of the pinned MSYS2 toolchain;
        # the paths of this build are mapped to /build and /cargo (nothing of this machine in the library)
        env PATH="$RUSTROOT/bin:/ucrt64/bin:/usr/bin" CARGO_HOME="$(M "$FB/home")" RUSTC="$(M "$RUSTROOT/bin/rustc.exe")" \
            CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=gcc CARGO_BUILD_JOBS="$JOBS" CARGO_INCREMENTAL=0 \
            RUSTFLAGS="--remap-path-prefix=$(M "$WORK")=/build --remap-path-prefix=$(M "$FB/home")=/cargo" \
            "$RUSTROOT/bin/cargo.exe" build --release --locked --offline
    ) > "$WORK/faer_build.log" 2>&1 || { tail -40 "$WORK/faer_build.log"; echo "ERROR: the Rust library did not build"; exit 1; }
    cp "$FB/target/release/libfea_faer.a" "$FAERLIB"
    printf '%s' "$FAERSTAMP" > "$WORK/libfea_faer.stamp"
fi
FAERINC="$WORK/faer_include"                          # the C header of the library (a copy: no path of the checkout in the compile line)
rm -rf "$FAERINC"; mkdir -p "$FAERINC"; cp "$FAERDIR/include/fea_faer.h" "$FAERINC/"

# ---- 6. CalculiX: our own Makefile (the one of dhondt.de has fixed library names); the date written into the banner and the .frd header is the fixed one
CCX="$WORK/CalculiX/ccx_$CCX_VER/src"
# the objects of an earlier run are kept while the sources, the patch and the flags are the same (a re-link after a change of a library is cheap)
CCXSTAMP="$(stamp "$STAMP" "$SPSTAMP" "$CF_COMMON $PERMISSIVE" "$FFL" "$SDE" "$(sumof "$SCRIPT_DIR/patches/ccx_2.23_ucrt_build.patch")" "$(sumof "$SCRIPT_DIR/patches/ccx_2.23_spooles_serial_solve.patch")" "$(sumof "$SCRIPT_DIR/patches/ccx_2.23_faer.patch")" "$(sumof "$FAERDIR/ccx/faer.c")" "$(sumof "$FAERDIR/ccx/faer.h")" "$(sumof "$DL/ccx_2.23.src.tar.bz2")")"
if [ "$(cat "$WORK/CalculiX/.fea_stamp" 2>/dev/null)" != "$CCXSTAMP" ]; then
    rm -rf "$WORK/CalculiX"
    tar -xjf "$DL/ccx_2.23.src.tar.bz2" -C "$WORK"
    # the one build fix (a dead declaration that the UCRT headers reject; no solver code): installer/windows/patches/ccx_2.23_ucrt_build.patch
    (cd "$WORK" && patch -p1 -i "$SCRIPT_DIR/patches/ccx_2.23_ucrt_build.patch" > "$WORK/ccx_patch.log" 2>&1) || { cat "$WORK/ccx_patch.log"; exit 1; }
    # the change of the solver logic (installer/windows/patches/ccx_2.23_spooles_serial_solve.patch, GPL-2.0): the multi-threaded factorization of SPOOLES is
    # kept, its multi-threaded triangular solve (a data race) is replaced by the serial one, the factors are checked after every multi-threaded factorization,
    # and `ccx -v` prints the markers FEA_SPOOLES_SERIAL_SOLVE=1, FEA_SPOOLES_FACTOR_CHECK=2 and (the ccx links only with the ordered SPOOLES) FEA_SPOOLES_ORDERED_ASSEMBLY=1
    (cd "$WORK" && patch -p1 -i "$SCRIPT_DIR/patches/ccx_2.23_spooles_serial_solve.patch" >> "$WORK/ccx_patch.log" 2>&1) || { cat "$WORK/ccx_patch.log"; exit 1; }
    # the faer solver (installer/windows/patches/ccx_2.23_faer.patch + the new files faer.c / faer.h of installer/faer_ccx/ccx, GPL-2.0 like CalculiX): spooles_factor /
    # spooles_solve / spooles_cleanup try faer first (symmetric: Cholesky or LBL^T with the ordering of SPOOLES, unsymmetric: LU), every factorization is checked by a
    # test solve and SPOOLES takes over when it fails; `ccx -v` prints the marker FEA_FAER=1 (compiled with -DFEA_FAER=1)
    (cd "$WORK" && patch -p1 -i "$SCRIPT_DIR/patches/ccx_2.23_faer.patch" >> "$WORK/ccx_patch.log" 2>&1) || { cat "$WORK/ccx_patch.log"; exit 1; }
    cp "$FAERDIR/ccx/faer.c" "$FAERDIR/ccx/faer.h" "$CCX/"
    (
        cd "$CCX"
        DATE="$(date -u -d "@$SDE" '+%a %b %e %H:%M:%S UTC %Y')"
        sed -i "s|You are using an executable made on.*|You are using an executable made on $DATE\\\\n\");|" ccx_$CCX_VER.c ccx_${CCX_VER}step.c
        sed -i "s|COMPILETIME.*|COMPILETIME       $DATE                    \\\\n\",p1);|" frd.c
    )
    printf '%s' "$CCXSTAMP" > "$WORK/CalculiX/.fea_stamp"
fi
rm -f "$CCX/ccx_$CCX_VER.exe"
MAP="$WORK/ccx_link.map"                           # the linker's map names every library object that went into the binary and the symbol that pulled it in
rm -f "$MAP"
(
    cd "$CCX"
    # LGPL-2.1 section 6: a library of the user's own (FEA_LIBQUADMATH_DIR, or built above from FEA_LIBQUADMATH_SOURCE) is searched before the one of the
    # toolchain (every -L applies to every -l)
    LQDIR=""
    if [ -n "${FEA_LIBQUADMATH_DIR:-}" ]; then
        [ -f "$FEA_LIBQUADMATH_DIR/libquadmath.a" ] || { echo "ERROR: FEA_LIBQUADMATH_DIR=$FEA_LIBQUADMATH_DIR has no libquadmath.a"; exit 1; }
        LQDIR="-L$(M "$FEA_LIBQUADMATH_DIR")"
        echo "libquadmath: linking with the library in $FEA_LIBQUADMATH_DIR instead of the one of the toolchain"
    fi
    cat > Makefile.fea <<EOF
include Makefile.inc
SCCXC += faer.c
SCCXMAIN = ccx_${CCX_VER}.c
OCCXF = \$(SCCXF:.f=.o)
OCCXC = \$(SCCXC:.c=.o)
OCCXMAIN = \$(SCCXMAIN:.c=.o)
.c.o :
	\$(CC) \$(CFLAGS) -c \$<
.f.o :
	\$(FC) \$(FFLAGS) -c \$<
ccx_${CCX_VER}.exe: \$(OCCXMAIN) ccx_${CCX_VER}.a
	\$(FC) \$(FFLAGS) -o \$@ \$(OCCXMAIN) ccx_${CCX_VER}.a \$(LIBS) \$(LDFLAGS)
ccx_${CCX_VER}.a: \$(OCCXF) \$(OCCXC)
	\$(AR) rc \$@ \$?
EOF
    make -f Makefile.fea -j"$JOBS" CC=gcc FC=gfortran AR=ar \
        CFLAGS="$CF_COMMON -fopenmp -I$(M "$SPDIR") -I$(M "$FAERINC") -DARCH=\"Linux\" -DSPOOLES -DARPACK -DMATRIXSTORAGE -DNETWORKOUT -DUSE_MT=1 -DFEA_FAER=1 -fcommon $PERMISSIVE" \
        FFLAGS="$FFL -fopenmp -cpp" \
        LIBS="$(M "$FAERLIB") -lntdll -lsynchronization -lws2_32 -luserenv -ldbghelp -lkernel32 $(M "$SPDIR")/spooles.a $(M "$WORK")/libarpack.a $BLASLIBS -lpthread -lm" \
        LDFLAGS="-static -Wl,--no-insert-timestamp -Wl,--stack,16777216 -Wl,--strip-all -Wl,-Map=$(M "$MAP") $LQDIR" \
        ccx_${CCX_VER}.exe > "$WORK/ccx_build.log" 2>&1 || { tail -40 "$WORK/ccx_build.log"; exit 1; }
)
EXE_OUT="$OUT/ccx_spooles.exe"
cp "$CCX/ccx_$CCX_VER.exe" "$EXE_OUT"

# ---- 7. checks of the finished binary: only system DLLs, no foreign component, no path of this machine
# the marker of the checked factorization (the program allows several equation-solver threads only for a binary that carries it) – in the file AND printed by `ccx -v`
grep -aqF 'FEA_SPOOLES_FACTOR_CHECK=2' "$EXE_OUT" || { echo "ERROR: the binary does not carry the marker FEA_SPOOLES_FACTOR_CHECK=2 (the patch ccx_2.23_spooles_serial_solve.patch was not applied or USE_MT is off)"; exit 1; }
VOUT="$("$EXE_OUT" -v 2>&1 || true)"          # `ccx -v` ends with a non-zero code: no pipe (pipefail), the text is looked at afterwards
case "$VOUT" in *FEA_SPOOLES_FACTOR_CHECK=2*) ;; *) echo "ERROR: ccx_spooles.exe -v does not print the marker FEA_SPOOLES_FACTOR_CHECK=2"; exit 1 ;; esac
# the marker of the ordered SPOOLES (printed only by a ccx that is linked with the SPOOLES of spooles_2.2_ordered_assembly.patch: the call of fea_spooles_ordered takes the address of FrontMtx_updateOrdered)
case "$VOUT" in *FEA_SPOOLES_ORDERED_ASSEMBLY=1*) ;; *) echo "ERROR: ccx_spooles.exe -v does not print the marker FEA_SPOOLES_ORDERED_ASSEMBLY=1 (the ordered SPOOLES is not inside)"; exit 1 ;; esac
grep -aqF 'FEA_SPOOLES_ORDERED_ASSEMBLY=1' "$EXE_OUT" || { echo "ERROR: the binary does not carry the marker FEA_SPOOLES_ORDERED_ASSEMBLY=1"; exit 1; }
# the marker of the faer solver (faer.c is compiled in and the Rust library is linked: ccx -v prints the version of the library)
case "$VOUT" in *FEA_FAER=1*) ;; *) echo "ERROR: ccx_spooles.exe -v does not print the marker FEA_FAER=1 (the faer solver is not inside)"; exit 1 ;; esac
grep -aqF 'FEA_FAER=1' "$EXE_OUT" || { echo "ERROR: the binary does not carry the marker FEA_FAER=1"; exit 1; }
echo "DLL imports:"
objdump -p "$(M "$EXE_OUT")" | awk '/DLL Name:/ {print $3}' | sort -u > "$WORK/imports.txt"
cat "$WORK/imports.txt"
bad=0
while read -r dll; do
    low="$(printf '%s' "$dll" | tr 'A-Z' 'a-z')"
    case "$low" in
        api-ms-win-*) ;;                                                   # API sets of the universal CRT (part of Windows 10+)
        lib*|*mingw*|*gcc*|*gfortran*|*pthread*|*gomp*|*quadmath*|*openblas*|*stdc++*) echo "ERROR: $dll is not a system DLL"; bad=1 ;;
        *) [ -f "/c/Windows/System32/$dll" ] || [ -f "/c/Windows/System32/$low" ] || { echo "ERROR: $dll is not in System32"; bad=1; } ;;
    esac
done < "$WORK/imports.txt"
# components that must NOT be inside: the SYMBOLS of the libraries (CalculiX has the words in the messages "the TAUCS library is not linked" and in the
# list of solver keywords) – TAUCS with AMD / COLAMD / GENMMD, PaStiX, Scotch, METIS, hwloc, libltdl, OpenBLAS and the vendor libraries – and the paths of this machine
# (the C functions of COLAMD / AMD are listed by name: faer carries Rust ports of AMD and COLAMD of SuiteSparse, BSD-3-Clause, their licence files travel with the program - and the
# word colamd stands in the path of their source file)
for pat in 'taucs_' oocsp colamd_recommended colamd_set_defaults symamd genmmd amdbar amdexa amdtru mc47 'pastix_' 'pastix version' 'scotch_' 'hwloc_' 'metis_' lt_dlopen openblas 'mkl_' \
           'Intel(R) MKL'; do
    if grep -aqiF -- "$pat" "$EXE_OUT"; then echo "ERROR: the binary contains '$pat'"; bad=1; fi
done
# the folders of THIS build, derived from where it really ran (the drive the work folder is mapped to, the toolchain folder, the work, output and archive
# folders in the MSYS and the Windows form) - not guessed
WM="$(M "$WORK")"; DRIVE="${WM%%/*}"
paths=("$WM" "$WORK" "$(M "$OUT")" "$OUT" "$(M /)" "$(basename "$(M /)")" "$(M "$ARCHIVE")")
[ "$DRIVE" = "C:" ] || paths+=("$DRIVE/" "$DRIVE$(printf '\134')")
for pat in "${paths[@]}"; do
    if grep -aqiF -- "$pat" "$EXE_OUT"; then echo "ERROR: the binary contains a path of this build ('$pat')"; bad=1; fi
done
# the paths of this machine: a folder "users" in either slash, and the strings the launcher passes in FEA_AUDIT_STRINGS (separated by |: the name of the
# user, the real path of the work folder and of the checkout) – nothing of the sort may be in the binary
U=users
for pat in "\\$U\\" "/$U/"; do
    if grep -aqiF -- "$pat" "$EXE_OUT"; then echo "ERROR: the binary contains a user folder path ('$pat')"; bad=1; fi
done
IFS='|' read -ra private <<< "${FEA_AUDIT_STRINGS:-}"
for pat in "${private[@]}"; do
    [ -n "$pat" ] || continue
    if grep -aqiF -- "$pat" "$EXE_OUT"; then echo "ERROR: the binary contains a string of this machine (see FEA_AUDIT_STRINGS)"; bad=1; fi
done
# The static libraries: every archive member that the linker took is in the link map together with the symbol that pulled it in (e.g. "libquadmath.a(printf_fp.o)
# ... libgfortran.a(write.o) (quadmath_snprintf)"). The binary itself is stripped, so the map is the reliable list. Every library must be a known one whose licence
# is covered in licenses/ (installer/collect_licenses.py): a change of the toolchain or of the link that pulls in another one (libstdc++, libssp, an import library
# of another DLL ...) stops the build until somebody has looked at its licence.
STATIC_LIBS=" ccx_2.23.a spooles.a libarpack.a liblapack.a libblas.a libgcc.a libgcc_eh.a libgfortran.a libgomp.a libquadmath.a libmingw32.a libmingwex.a libmsvcrt.a libpthread.a libkernel32.a libfea_faer.a libntdll.a libsynchronization.a libws2_32.a libuserenv.a libdbghelp.a "
LGPL_LIBS=" libquadmath.a "                         # LGPL: the notices and the text of the licence must travel with the binary
audit_static_libs() {   # audit_static_libs MAP MEMBERS COUNTS: the library members of the link map -> MEMBERS ("lib member"), COUNTS (objects per library); 1 = a problem
    local map="$1" members="$2" counts="$3" lib f rc=0
    [ -s "$map" ] || { echo "ERROR: the linker wrote no map ($map) - the static libraries cannot be checked"; return 1; }
    awk '/^Archive member included/ {on=1; next} /^(Discarded input sections|Allocating common symbols|Memory Configuration)/ {on=0} on && /^[^ ]/' "$map" \
        | sed -E 's#^(.*[/\])?([^/\()]+\.a)\(([^)]*)\).*$#\2 \3#' | sort -u > "$members"
    [ -s "$members" ] || { echo "ERROR: no archive members found in the link map - the format of the map has changed"; return 1; }
    echo "static libraries (objects pulled in):"
    cut -d' ' -f1 "$members" | uniq -c | sed 's/^ */  /' | tee "$counts"
    for lib in $(cut -d' ' -f1 "$members" | sort -u); do
        case "$STATIC_LIBS" in *" $lib "*) ;; *) echo "ERROR: the binary contains objects of $lib, which is not in STATIC_LIBS - check its licence, add its texts to installer/collect_licenses.py and the library to the list"; rc=1 ;; esac
        case "$LGPL_LIBS" in
            *" $lib "*)
                for f in libquadmath_COPYING.LIB NOTICE-libquadmath.txt; do
                    [ -f "$SCRIPT_DIR/../license_texts/gcc-runtime/$f" ] || { echo "ERROR: $lib (LGPL) is inside the binary, but installer/license_texts/gcc-runtime/$f is missing"; rc=1; }
                done ;;
        esac
    done
    return $rc
}
audit_static_libs "$MAP" "$WORK/members.txt" "$WORK/static_libs.txt" || bad=1
[ -s "$WORK/members.txt" ] || exit 1
# the libquadmath that was really linked (the toolchain's or the one of FEA_LIBQUADMATH_DIR): its digest goes into the manifest
# (the path may contain spaces: the whole line up to "libquadmath.a(" is the path)
LQLINKED="$(awk '/^Archive member included/ {on=1; next} /^(Discarded input sections|Allocating common symbols|Memory Configuration)/ {on=0} on && /^[^ ]/ && /libquadmath\.a\(/ {print; exit}' "$MAP" \
    | sed -E 's/^(.*libquadmath\.a)\(.*$/\1/' | tr '\134' '/')"
LQSHA=""
if [ -n "$LQLINKED" ] && [ -f "$(cygpath -u "$LQLINKED")" ]; then LQSHA="$(sha256sum "$(cygpath -u "$LQLINKED")" | cut -d' ' -f1)"; fi
if grep -q '^libquadmath.a ' "$WORK/members.txt" && [ -z "$LQSHA" ]; then echo "ERROR: libquadmath is inside the binary, but the library that was linked cannot be found ('$LQLINKED')"; bad=1; fi
if [ -n "${FEA_LIBQUADMATH_DIR:-}" ]; then LQSRC="REPLACED by the user (FEA_LIBQUADMATH_DIR / FEA_LIBQUADMATH_SOURCE): not the library of the release"; else LQSRC="the package mingw-w64-ucrt-x86_64-libquadmath of ccx_toolchain.lock"; fi
[ "$bad" = 0 ] || exit 1

# ---- 8. the build manifest (what the binary was made of: pinned sources with SHA-256, the toolchain lock, the flags)
json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g;s/"/\\"/g'; }
{
    printf '{\n'
    printf '  "name": "ccx_spooles.exe",\n  "calculix": "%s",\n' "$CCX_VER"
    printf '  "platform": "windows-x86_64",\n'
    printf '  "equation_solver": "faer 0.24.4 (Cholesky / LBL^T of symmetric systems with the nested dissection of SPOOLES, LU of unsymmetric ones; every factorization checked by a test solve) with SPOOLES 2.2 as the fallback (multi-threaded factorization with a checked result and a reproducible order of the sums, serial triangular solve: patches/ccx_2.23_spooles_serial_solve.patch, patches/spooles_2.2_ordered_assembly.patch, patches/ccx_2.23_faer.patch)",
'
    printf '  "spooles_serial_solve": true,\n'
    printf '  "spooles_factor_check": true,\n  "spooles_ordered_assembly": true,\n'
    printf '  "patches": [{"file": "ccx_2.23_ucrt_build.patch", "sha256": "%s"}, {"file": "ccx_2.23_spooles_serial_solve.patch", "sha256": "%s"}, {"file": "spooles_2.2_ordered_assembly.patch", "sha256": "%s"}, {"file": "ccx_2.23_faer.patch", "sha256": "%s"}],\n' "$(sha256sum "$SCRIPT_DIR/patches/ccx_2.23_ucrt_build.patch" | cut -d' ' -f1)" "$(sha256sum "$SCRIPT_DIR/patches/ccx_2.23_spooles_serial_solve.patch" | cut -d' ' -f1)" "$(sha256sum "$SCRIPT_DIR/patches/spooles_2.2_ordered_assembly.patch" | cut -d' ' -f1)" "$(sha256sum "$SCRIPT_DIR/patches/ccx_2.23_faer.patch" | cut -d' ' -f1)"
    printf '  "faer": {"marker": "FEA_FAER=1", "library": "faer 0.24.4 (MIT) through installer/faer_ccx (the crates of Cargo.lock, listed in sources with their SHA-256)", "rust": "%s", "rust_lock_sha256": "%s", "library_sha256": "%s", "patch": {"file": "ccx_2.23_faer.patch", "sha256": "%s"}, "faer_c_sha256": "%s", "faer_h_sha256": "%s", "cargo_lock_sha256": "%s"},\n' "$(json_escape "$(printf '%s' "$RUSTC_V" | sed -n 's/^release: //p;s/^commit-hash: //p' | paste -sd' ')")" "$RUSTSTAMP" "$(sha256sum "$FAERLIB" | cut -d' ' -f1)" "$(sha256sum "$SCRIPT_DIR/patches/ccx_2.23_faer.patch" | cut -d' ' -f1)" "$(sha256sum "$FAERDIR/ccx/faer.c" | cut -d' ' -f1)" "$(sha256sum "$FAERDIR/ccx/faer.h" | cut -d' ' -f1)" "$(sha256sum "$FAERDIR/Cargo.lock" | cut -d' ' -f1)"
    printf '  "eigenvalues": "ARPACK-ng 3.9.1",\n'
    printf '  "blas_lapack": "%s",\n' "$(json_escape "$(cat "$BLASDIR/.desc")")"
    printf '  "compiler": "%s",\n' "$(json_escape "$(gcc --version | head -1)")"
    printf '  "fortran": "%s",\n' "$(json_escape "$(gfortran --version | head -1)")"
    printf '  "linker": "%s",\n' "$(json_escape "$(ld --version | head -1)")"
    printf '  "link": "static (libgcc, libgfortran, libgomp, libquadmath, winpthreads); imports only the system DLLs listed below",\n'
    printf '  "libquadmath": {"licence": "LGPL-2.1+", "objects": %s, "source": "%s", "sha256": "%s"},\n' "$(grep -c '^libquadmath.a ' "$WORK/members.txt" || true)" "$(json_escape "$LQSRC")" "$LQSHA"
    printf '  "static_libraries": {'
    awk '{ printf "%s\"%s\": %s", (NR>1 ? ", " : ""), $2, $1 }' "$WORK/static_libs.txt"
    printf '},\n'
    printf '  "source_date_epoch": %s,\n' "$SDE"
    printf '  "toolchain_lock_sha256": "%s",\n' "$STAMP"
    printf '  "build_script_sha256": "%s",\n' "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
    printf '  "binary_sha256": "%s",\n' "$(sha256sum "$EXE_OUT" | cut -d' ' -f1)"
    printf '  "dll_imports": ['
    sed 's/.*/"&"/' "$WORK/imports.txt" | paste -sd, | sed 's/,/, /g' | tr -d '\n'
    printf '],\n  "sources": [\n'
    sort -u "$WORK/downloads.tsv" | awk -F'\t' '{ printf "%s    {\"file\": \"%s\", \"sha256\": \"%s\", \"url\": \"%s\"}", (NR>1 ? ",\n" : ""), $1, $2, $3 } END { printf "\n" }'
    printf '  ],\n  "not_included": ["TAUCS (AMD, COLAMD, GENMMD)", "PaStiX", "Scotch", "METIS", "hwloc", "libltdl", "OpenBLAS"]\n}\n'
} > "$OUT/manifest.json"
echo "done: $EXE_OUT ($(du -h "$EXE_OUT" | cut -f1)); manifest: $OUT/manifest.json"

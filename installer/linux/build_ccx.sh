#!/usr/bin/env bash
# Builds CalculiX 2.23 for Linux x86_64 from the sources of dhondt.de (no administrator rights, no system packages): the compilers come from
# conda-forge (micromamba), SPOOLES 2.2 from netlib, ARPACK / LAPACK from GitHub, and – optionally – PaStiX 6.4.0 from files.inria.fr (static,
# 32-bit integers; SCOTCH, hwloc and OpenBLAS from conda-forge). Everything that is downloaded is pinned by a SHA-256 digest: the sources in this
# script, the conda-forge packages in installer/linux/ccx_toolchain.lock and ccx_deps.lock (installer/linux/make_ccx_locks.py regenerates them).
#
#   installer/linux/build_ccx.sh OUTPUT [WORKDIR] [ARCHIVE.tar.gz]
#
# OUTPUT = the folder that receives: ccx_spooles (SPOOLES only – the binary that is distributed; the factorization of SPOOLES runs on
# CCX_NPROC_EQUATION_SOLVER threads, the triangular solve is the serial one and the factors are CHECKED by a test solve (repeated when the check fails): patches/
# ccx_2.23_spooles_serial_solve.patch, GPL-2.0, a change of the solver logic of CalculiX – the parallel solve of SPOOLES is a data race; `ccx_spooles -v` prints the markers FEA_SPOOLES_FACTOR_CHECK=2 and FEA_SPOOLES_ORDERED_ASSEMBLY=1 that the
# program checks, see spooles_threads_safe in fea_core/ccx/runner.py), lib/ (libgomp, libquadmath; with PaStiX also
# OpenBLAS, SCOTCH, hwloc ... – RUNPATH $ORIGIN/lib, so the binaries run without installing anything in the system; the other numerics are linked
# statically) and ccx_pastix (SPOOLES + PaStiX).
# ccx_spooles also has the faer solver (F17 stage C): the Rust static library libfea_faer.a of installer/faer_ccx (faer 0.24.4, MIT) built with the pinned Rust toolchain of
# ccx_rust.lock from the crates of installer/faer_ccx/Cargo.lock (SHA-256 of every tarball and crate), linked with faer.c and the patch ccx_2.23_faer.patch (symmetric systems:
# Cholesky / LDL^T / LBL^T with the ordering of SPOOLES, every factorization checked, SPOOLES takes over when it fails); `ccx_spooles -v` prints FEA_FAER=1.
#
# ccx_spooles is built FIRST and is complete before the optional parts start: a failed PaStiX download or build only leaves out that
# binary, the script ends with the message and code 0 ("ccx_pastix: NOT built"). Environment:
#   FEA_CCX_PASTIX=0   skip PaStiX (default 1 = try); the installer never runs this script (it only unpacks the archive made here)
#   FEA_BUILD_JOBS=N   number of parallel compilations (default: all cores)
#
# ccx 2.23 calls PaStiX through the interface of the fork PaStiX4CalculiX (IPARM_REUSE_LU, pastixResetSteps, the 4-argument pastix_task_solve ...)
# – the fork does not build without CUDA / PaRSEC (cublas_v2.h in pastixdata.h), so pastix.c is patched to the released PaStiX 6.4.0
# (installer/linux/patches/ccx_2.23_pastix_upstream.patch: no reuse of the analysis, spm->baseval = 1 + spmBase(spm, 0)).
# ARCHIVE (optional) = the prebuilt package for distribution (installer/linux/prebuilt/ccx_2.23_linux_x86_64.tar.gz): everything that was built:
# owner 0:0 without names, fixed time (SOURCE_DATE_EPOCH), sorted
# names, gzip without name and time. installer/make_package.py takes from it only ccx_spooles and the libraries it needs (ccx_pastix only with
# --with-ccx-pastix – the licence question L4 of PaStiX inside GPL-2.0-only CalculiX is open, docs/projekt/F13_plan_speed_memory_licences_native.md).
# The path of the work folder (often in a home folder -> the user name) must not get into the binaries: gcc maps it with -ffile-prefix-map to
# /build; gfortran writes the name of the source file into I/O statements (run-time error messages) and -ffile-prefix-map does NOT apply there
# (verified with gfortran 16.2) -> LAPACK / ARPACK (cmake passes absolute paths) are compiled through the wrapper fc_rel that gives the source from
# the work folder as a relative path.
# Takes some minutes; downloads about 500 MB (sources + compilers).
set -euo pipefail

OUT="${1:?usage: build_ccx.sh OUTPUT [WORKDIR] [ARCHIVE.tar.gz]}"
WORK="${2:-$(mktemp -d /tmp/fea_ccx_build.XXXXXX)}"   # no fixed path in /tmp (another user could plant files)
ARCHIVE="${3:-}"
CCX_VER=2.23
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"
JOBS="${FEA_BUILD_JOBS:-$(nproc)}"
WITH_PASTIX="${FEA_CCX_PASTIX:-1}"
mkdir -p "$OUT" "$WORK"
OUT="$(cd "$OUT" && pwd)"
WORK="$(cd "$WORK" && pwd)"
if [ -n "$ARCHIVE" ]; then
    mkdir -p "$(dirname "$ARCHIVE")"
    ARCHIVE="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"
fi
cd "$WORK"
# the work folder (often in the user's home folder) -> a neutral /build in all strings of the binaries
PMAP="-ffile-prefix-map=$WORK=/build"

get_mirrors() {  # get_mirrors FILE SHA256 URL [BACKUP_URL...] - the first source that passes the digest check
    local file="$1" sum="$2" url; shift 2
    if [ -s "$file" ] && echo "$sum  $file" | sha256sum -c --status -; then return 0; fi   # also verify what was downloaded before
    for url in "$@"; do
        echo "downloading $url"
        if curl -fsSL --retry 3 -o "$file.part" "$url" && echo "$sum  $file.part" | sha256sum -c --status -; then
            mv "$file.part" "$file"; return 0
        fi
        echo "  failed (or the digest does not match) - trying the next source"
    done
    rm -f "$file.part"; echo "ERROR: could not download $file"; exit 1
}

# SHA-256 digests verified 17. 10. 2026 (a new version of the sources = a new digest, otherwise the build stops with an error)
get_mirrors ccx.src.tar.bz2 9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7 \
    "https://www.dhondt.de/ccx_${CCX_VER}.src.tar.bz2"
get_mirrors spooles.corr.tar.bz2 15528f09312dc7605c0600358f5e6de12945449f249dfcfca7417eed6c220b93 \
    "https://www.dhondt.de/ccx_${CCX_VER}.SPOOLEScorrection.tar.bz2"
# SPOOLES 2.2 from netlib; some filtering DNS servers (e.g. DNS4EU) block netlib -> the same file as the orig.tar.gz of Debian
get_mirrors spooles.tgz a84559a0e987a1e423055ef4fdf3035d55b65bbe4bf915efaa1a35bef7f8c5dd \
    "https://www.netlib.org/linalg/spooles/spooles.2.2.tgz" \
    "https://deb.debian.org/debian/pool/main/s/spooles/spooles_2.2.orig.tar.gz"
get_mirrors lapack.tar.gz eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b \
    "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz"
get_mirrors arpack.tar.gz f6641deb07fa69165b7815de9008af3ea47eb39b2bb97521fbf74c97aba6e844 \
    "https://github.com/opencollab/arpack-ng/archive/refs/tags/3.9.1.tar.gz"
if [ ! -x mamba/bin/micromamba ]; then
    mkdir -p mamba
    # the same pinned version and digest as MAMBA_URL / MAMBA_SHA256 in installer/fea_setup.py
    get_mirrors micromamba.tar.bz2 8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd \
        "https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64.tar.bz2"
    tar -xjf micromamba.tar.bz2 -C mamba bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$WORK/mamba/root"

create_locked_env() {  # create_locked_env PREFIX LOCKFILE SPEC - a conda environment of EXACTLY the packages of the lock file: every package is
    # downloaded with the SHA-256 of the lock, micromamba installs the explicit list (it checks the MD5 too) and resolves nothing
    local prefix="$1" lock="$2" spec="$3" stamp list sum md5 url f
    grep -qxF "# spec: $spec" "$lock" || { echo "ERROR: $(basename "$lock") was made for another specification than '$spec' - run installer/linux/make_ccx_locks.py"; exit 1; }
    stamp="$({ cat "$lock"; printf '%s' "$spec"; } | sha256sum | cut -d' ' -f1)"
    if [ -d "$prefix/conda-meta" ] && [ "$(cat "$prefix/.fea_spec" 2>/dev/null)" = "$stamp" ]; then return 0; fi
    rm -rf "$prefix"
    mkdir -p "$WORK/pkgs"
    list="$WORK/$(basename "$prefix").explicit"
    echo "@EXPLICIT" > "$list"
    while read -r sum md5 url; do
        case "$sum" in ""|"#"*) continue ;; esac
        url="${url%$'\r'}"          # a lock file with CRLF (checkout with autocrlf) would otherwise break the URL
        f="$WORK/pkgs/$(basename "$url")"
        get_mirrors "$f" "$sum" "$url"
        echo "file://${f// /%20}#$md5" >> "$list"
    done < "$lock"
    "$WORK/mamba/bin/micromamba" create -y -q -p "$prefix" --file "$list"
    printf '%s' "$stamp" > "$prefix/.fea_spec"
}

# sysroot glibc 2.28: without it conda-forge (gcc 16) brings a sysroot 2.39 and the binaries want glibc >= 2.38
# (__isoc23_strtol/sscanf, fmod@GLIBC_2.38) - the program must run from glibc 2.35 (Ubuntu 22.04, Debian 12 has 2.36)
TC_SPEC="gcc_linux-64 gfortran_linux-64 sysroot_linux-64=2.28 make cmake patchelf patch"
TC="$WORK/tc"
create_locked_env "$TC" "$SCRIPT_DIR/ccx_toolchain.lock" "$TC_SPEC"
# the PaStiX part (versions pinned: the ones measured 18. 11. 2026): OpenBLAS, SCOTCH, hwloc ...; python only for the code generator of PaStiX
DEPS_SPEC="libscotch=7.0.11=int32* libhwloc=2.15.0 libopenblas=0.3.34=pthreads* libblas=3.11.0=*openblas libcblas=3.11.0 liblapacke=3.11.0 pkg-config zlib python=3.12"
CC="$TC/bin/x86_64-conda-linux-gnu-gcc"
FC="$TC/bin/x86_64-conda-linux-gnu-gfortran"
AR="$TC/bin/x86_64-conda-linux-gnu-ar"
RANLIB="$TC/bin/x86_64-conda-linux-gnu-ranlib"
# the debug information of the statically linked libgfortran / libgcc and of the conda-forge libraries carries the paths of their build farm
# (/home/conda/feedstock_root/...) - strip it; strip BEFORE patchelf (strip after patchelf can damage an ELF file)
STRIP="$TC/bin/x86_64-conda-linux-gnu-strip"
MAKE="$TC/bin/make"
DEPS="$WORK/deps"        # the environment of the PaStiX part (created there; only the path is needed by copy_deps)
# gcc 14+ turns old C constructs (SPOOLES, CalculiX) into errors
PERMISSIVE="-Wno-error=int-conversion -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=return-mismatch"

# ---- static BLAS / LAPACK (reference) and ARPACK -> the binary does not need any shared numerical library
export PATH="$TC/bin:$PATH"
FFL="-O2 -fPIC -fallow-argument-mismatch $PMAP"
# the gfortran wrapper for cmake: a source file of the work folder relative to the current folder of the compilation (see the header)
cat > "$WORK/fc_rel" <<EOF
#!/usr/bin/env bash
args=()
for a in "\$@"; do
    case "\$a" in
        "$WORK"/*.f|"$WORK"/*.F|"$WORK"/*.f90|"$WORK"/*.F90) [ -f "\$a" ] && a="\$(realpath --relative-to=. "\$a")" ;;
    esac
    args+=("\$a")
done
exec "$FC" "\${args[@]}"
EOF
chmod +x "$WORK/fc_rel"
STAMP="$FFL; fc_rel; $(sha256sum "$SCRIPT_DIR/ccx_toolchain.lock" | cut -d' ' -f1)"
# libraries of an earlier run only with the same procedure (an older build carried the path of the work folder)
if [ ! -f static/lib/libarpack.a ] || [ "$(cat static/.flags 2>/dev/null)" != "$STAMP" ]; then
    rm -rf lapack-src arpack-src static && mkdir -p lapack-src arpack-src static
    tar -xzf lapack.tar.gz -C lapack-src --strip-components=1
    tar -xzf arpack.tar.gz -C arpack-src --strip-components=1
    cmake -S lapack-src -B lapack-src/b -G "Unix Makefiles" -DCMAKE_Fortran_COMPILER="$WORK/fc_rel" -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_MAKE_PROGRAM="$MAKE" -DCMAKE_AR="$AR" -DCMAKE_RANLIB="$RANLIB" -DCMAKE_INSTALL_PREFIX="$WORK/static" -DBUILD_SHARED_LIBS=OFF -DCBLAS=OFF \
        -DLAPACKE=OFF -DBUILD_TESTING=OFF -DCMAKE_Fortran_FLAGS="$FFL" -DCMAKE_C_FLAGS="-O2 -fPIC $PMAP" \
        -DCMAKE_INSTALL_LIBDIR=lib > "$WORK/lapack_cmake.log" 2>&1
    cmake --build lapack-src/b -j"$JOBS" >> "$WORK/lapack_cmake.log" 2>&1
    cmake --install lapack-src/b >> "$WORK/lapack_cmake.log" 2>&1
    cmake -S arpack-src -B arpack-src/b -G "Unix Makefiles" -DCMAKE_Fortran_COMPILER="$WORK/fc_rel" -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_MAKE_PROGRAM="$MAKE" -DCMAKE_AR="$AR" -DCMAKE_RANLIB="$RANLIB" -DCMAKE_INSTALL_PREFIX="$WORK/static" -DBUILD_SHARED_LIBS=OFF -DMPI=OFF -DICB=OFF \
        -DEXAMPLES=OFF -DTESTS=OFF -DCMAKE_Fortran_FLAGS="$FFL" -DCMAKE_C_FLAGS="-O2 -fPIC $PMAP" -DCMAKE_INSTALL_LIBDIR=lib \
        -DBLAS_LIBRARIES="$WORK/static/lib/libblas.a" -DLAPACK_LIBRARIES="$WORK/static/lib/liblapack.a;$WORK/static/lib/libblas.a" \
        > "$WORK/arpack_cmake.log" 2>&1
    cmake --build arpack-src/b -j"$JOBS" >> "$WORK/arpack_cmake.log" 2>&1
    cmake --install arpack-src/b >> "$WORK/arpack_cmake.log" 2>&1
    [ -f static/lib/libarpack.a ] || { tail -30 "$WORK/arpack_cmake.log"; exit 1; }
    printf '%s' "$STAMP" > static/.flags
fi
STATIC="$WORK/static/lib"

rm -rf src && mkdir src && cd src
tar -xjf ../ccx.src.tar.bz2                                  # CalculiX/ccx_2.23/src
mkdir SPOOLES.2.2 && tar -xzf ../spooles.tgz -C SPOOLES.2.2
# the change of the solver logic of CalculiX (serial triangular solve after the multi-threaded factorization, marker of `ccx -v`); the same file as
# installer/windows/patches/. Applied before anything is compiled, so every binary of this script carries it (only ccx_spooles is built with USE_MT).
patch -p1 -i "$PATCH_DIR/ccx_${CCX_VER}_spooles_serial_solve.patch" > "$WORK/spooles_serial_patch.log" 2>&1 || { cat "$WORK/spooles_serial_patch.log"; exit 1; }
# the faer solver behind the SPOOLES interface (installer/faer_ccx; the same file as installer/windows/patches/; the hunks are inert without -DFEA_FAER=1, so ccx_pastix is not changed by it)
patch -p1 -i "$PATCH_DIR/ccx_${CCX_VER}_faer.patch" > "$WORK/faer_patch.log" 2>&1 || { cat "$WORK/faer_patch.log"; exit 1; }
tar -xjf ../spooles.corr.tar.bz2                              # correction of util.c according to dhondt.de
cp CalculiX/ccx_2.23/SPOOLES.2.2/I2Ohash/src/util.c SPOOLES.2.2/I2Ohash/src/util.c
# reproducible (ordered) assembly of the multi-threaded factorization (SPOOLES is public domain; the same file as installer/windows/patches/), before SPOOLES is compiled
(cd SPOOLES.2.2 && patch -p1 -i "$PATCH_DIR/spooles_2.2_ordered_assembly.patch" > "$WORK/spooles_ordered_patch.log" 2>&1) || { cat "$WORK/spooles_ordered_patch.log"; exit 1; }

# ---- SPOOLES: gcc, a fix of makeGlobalLib (drawTree.c does not exist), without MPI
cd SPOOLES.2.2
sed -i "s|^  CC = .*|  CC = $CC -w -fcommon -O2 $PERMISSIVE $PMAP|" Make.inc
sed -i 's|drawTree.c|draw.c|g' Tree/src/makeGlobalLib
$MAKE -s lib 2>&1 | tail -3
[ -f spooles.a ] || { echo "SPOOLES did not compile"; exit 1; }
# the top-level makefile has the multi-threaded objects (MT/) commented out: add them to spooles.a (ccx_spooles is compiled with -DUSE_MT=1: with
# CCX_NPROC_EQUATION_SOLVER > 1 the factorization is multi-threaded, the triangular solve serial by the patch above)
(cd MT/src && $MAKE -s -f makeGlobalLib) > "$WORK/spooles_mt_build.log" 2>&1 || { tail -30 "$WORK/spooles_mt_build.log"; echo "SPOOLES MT did not compile"; exit 1; }
"$TC/bin/x86_64-conda-linux-gnu-ar" t spooles.a | grep -q "MT_factorMT.o" || { echo "SPOOLES: the MT objects are missing in spooles.a"; exit 1; }
cd ..

# ---- faer (F17 stage C): the Rust static library libfea_faer.a of installer/faer_ccx = a C interface over faer 0.24.4 (MIT), built with the pinned Rust toolchain of
# ccx_rust.lock (x86_64-unknown-linux-gnu) from the crates of installer/faer_ccx/Cargo.lock (every crate is downloaded with the SHA-256 digest of the lock file, then
# `cargo build --locked --offline` runs against a directory source made of them - the layout of `cargo vendor`). The result is linked into ccx_spooles only
# (faer.c, patch ccx_2.23_faer.patch); the linker is the gcc of the toolchain above (glibc 2.28 sysroot).
FAERDIR="$(cd "$SCRIPT_DIR/../faer_ccx" && pwd)"
RUSTLOCK="$SCRIPT_DIR/ccx_rust.lock"
RUSTSTAMP="$(sha256sum "$RUSTLOCK" | cut -d' ' -f1)"
RUSTROOT="$WORK/rust"
RUST_COMMIT="$(sed -n 's/^# rust_commit: *//p' "$RUSTLOCK" | tr -d '\r' | head -1)"
mkdir -p "$WORK/rust_dl" "$WORK/crates"
if [ "$(cat "$RUSTROOT/.fea_rust" 2>/dev/null)" != "$RUSTSTAMP" ]; then      # unpack the components into one prefix (what the rust-installer script of each tarball would do)
    rm -rf "$RUSTROOT" "$WORK/rust_unpack"; mkdir -p "$RUSTROOT" "$WORK/rust_unpack"
    while read -r kind sum url; do
        [ "$kind" = rust ] || continue
        url="${url%$'\r'}"
        get_mirrors "$WORK/rust_dl/$(basename "$url")" "$sum" "$url"
        tar -xJf "$WORK/rust_dl/$(basename "$url")" -C "$WORK/rust_unpack"
    done < "$RUSTLOCK"
    for d in "$WORK"/rust_unpack/*/; do
        while read -r comp; do [ -z "$comp" ] || cp -a "$d$comp/." "$RUSTROOT/"; done < "${d}components"
    done
    rm -rf "$WORK/rust_unpack"
    printf '%s' "$RUSTSTAMP" > "$RUSTROOT/.fea_rust"
fi
RUSTC_V="$("$RUSTROOT/bin/rustc" -vV)"
case "$RUSTC_V" in *"commit-hash: $RUST_COMMIT"*) ;; *) echo "ERROR: rustc of ccx_rust.lock is not the pinned release (commit $RUST_COMMIT): $RUSTC_V"; exit 1 ;; esac
# the crates: name, version, SHA-256 of the .crate file from Cargo.lock (packages with a registry source)
awk '/^\[\[package\]\]/{n="";v="";s=""} /^name = /{gsub(/"/,"",$3);n=$3} /^version = /{gsub(/"/,"",$3);v=$3} /^source = /{s=$3} /^checksum = /{gsub(/"/,"",$3); if(n!=""&&s!="") print n, v, $3}' "$FAERDIR/Cargo.lock" > "$WORK/faer_crates.txt"
[ "$(wc -l < "$WORK/faer_crates.txt")" -ge 20 ] || { echo "ERROR: Cargo.lock of installer/faer_ccx lists too few crates"; exit 1; }
while read -r cn cv csum; do
    get_mirrors "$WORK/crates/$cn-$cv.crate" "$csum" "https://static.crates.io/crates/$cn/$cn-$cv.crate"
done < "$WORK/faer_crates.txt"
FAERLIB="$WORK/libfea_faer.a"
FAERSTAMP="$({ cat "$RUSTLOCK" "$FAERDIR/Cargo.lock" "$FAERDIR/Cargo.toml" "$FAERDIR/src/lib.rs"; printf '%s' "$PMAP"; } | sha256sum | cut -d' ' -f1)"
if [ ! -f "$FAERLIB" ] || [ "$(cat "$WORK/libfea_faer.stamp" 2>/dev/null)" != "$FAERSTAMP" ]; then
    FB="$WORK/faer_build"
    rm -rf "$FB" "$FAERLIB"; mkdir -p "$FB/vendor" "$FB/crate/.cargo" "$FB/home"
    while read -r cn cv csum; do
        tar -xzf "$WORK/crates/$cn-$cv.crate" -C "$FB/vendor"
        printf '{"files":{},"package":"%s"}' "$csum" > "$FB/vendor/$cn-$cv/.cargo-checksum.json"
    done < "$WORK/faer_crates.txt"
    cp -r "$FAERDIR/Cargo.toml" "$FAERDIR/Cargo.lock" "$FAERDIR/src" "$FB/crate/"
    cat > "$FB/crate/.cargo/config.toml" <<EOF
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$FB/vendor"

[net]
offline = true

[build]
target-dir = "$FB/target"
EOF
    (
        cd "$FB/crate"
        env PATH="$RUSTROOT/bin:$TC/bin:$PATH" CARGO_HOME="$FB/home" RUSTC="$RUSTROOT/bin/rustc" \
            CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC" CARGO_BUILD_JOBS="$JOBS" CARGO_INCREMENTAL=0 \
            RUSTFLAGS="--remap-path-prefix=$WORK=/build" \
            "$RUSTROOT/bin/cargo" build --release --locked --offline
    ) > "$WORK/faer_build.log" 2>&1 || { tail -40 "$WORK/faer_build.log"; echo "ERROR: the Rust library did not build"; exit 1; }
    cp "$FB/target/release/libfea_faer.a" "$FAERLIB"
    printf '%s' "$FAERSTAMP" > "$WORK/libfea_faer.stamp"
fi
FAERINC="$WORK/faer_include"                          # the C header of the library (a copy)
rm -rf "$FAERINC"; mkdir -p "$FAERINC"; cp "$FAERDIR/include/fea_faer.h" "$FAERINC/"

# ---- CalculiX
cd CalculiX/ccx_$CCX_VER/src
cp "$FAERDIR/ccx/faer.c" "$FAERDIR/ccx/faer.h" .         # the faer solver (installer/faer_ccx/ccx, GPL-2.0 like CalculiX): compiled only with -DFEA_FAER=1 (the file is empty without it)
SP="$WORK/src/SPOOLES.2.2"
# the Makefile of dhondt.de has fixed LIBS with libarpack_INTEL.a and `ar vr`; we use our own Makefile.fea
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
ccx_${CCX_VER}: \$(OCCXMAIN) ccx_${CCX_VER}.a
	\$(CC) \$(CFLAGS) -c ccx_${CCX_VER}.c; \$(FC) -O2 -o \$@ \$(OCCXMAIN) ccx_${CCX_VER}.a \$(LIBS) -fopenmp -static-libgfortran -static-libgcc
ccx_${CCX_VER}.a: \$(OCCXF) \$(OCCXC)
	\$(AR) rc \$@ \$?
EOF
MAKEFILE="-f Makefile.fea"
build_one() {  # build_one NAME CFLAGS LIBS [LAPACK_LIBS]
    local name="$1" cflags="$2" libs="$3" lapack="${4:-$STATIC/liblapack.a $STATIC/libblas.a}"
    rm -f *.o ccx_${CCX_VER}.a ccx_${CCX_VER}
    $MAKE $MAKEFILE -j"$JOBS" CC="$CC" FC="$FC" AR="$AR" \
        CFLAGS="-w -O2 -I$SP -DARCH=\"Linux\" -DSPOOLES -DARPACK -DMATRIXSTORAGE -DNETWORKOUT -fcommon $PERMISSIVE $PMAP $cflags" \
        FFLAGS="-w -O2 -cpp -fallow-argument-mismatch $PMAP" \
        LIBS="$SP/spooles.a $STATIC/libarpack.a $lapack -lpthread -lm -lc $libs" \
        ccx_${CCX_VER} > "$WORK/build_$name.log" 2>&1 || { tail -30 "$WORK/build_$name.log"; exit 1; }
    cp ccx_${CCX_VER} "$OUT/$name"
    "$STRIP" --strip-debug "$OUT/$name"
    echo "done: $OUT/$name"
}

# ---- shared libraries from the conda environments next to the binaries (RUNPATH $ORIGIN/lib)
mkdir -p "$OUT/lib"
# conda writes the install prefix into some libraries (libhwloc: the run folder, libxml2: the catalog path) - the work folder (a home
# folder name) must not stay in the archive: same-length replacement by a harmless path "/build//...//d/..." (repeated slashes are valid)
PADPATH="/build$(printf '%*s' $(( ${#WORK} - 7 )) '' | tr ' ' '/')d"
copy_deps() {  # copy_deps FILE - copies the dependencies that live in $TC/lib and $DEPS/lib
    LD_LIBRARY_PATH="$DEPS/lib:$TC/lib" ldd "$1" | awk '/=> \//{print $3}' | { grep -F -e "$TC/" -e "$DEPS/" || true; } | while read -r so; do
        [ -e "$OUT/lib/$(basename "$so")" ] && continue
        cp -L "$so" "$OUT/lib/"
        "$STRIP" --strip-debug "$OUT/lib/$(basename "$so")"
        LC_ALL=C sed -i "s#$WORK#$PADPATH#g" "$OUT/lib/$(basename "$so")"
    done
}
ship() {  # ship BINARY - RUNPATH $ORIGIN/lib and the libraries it needs (and those they need, while dependencies keep appearing)
    local pass so
    "$TC/bin/patchelf" --set-rpath '$ORIGIN/lib' "$1"
    copy_deps "$1"
    for pass in 1 2 3; do
        for so in "$OUT"/lib/*.so*; do [ -e "$so" ] && copy_deps "$so"; done
    done
    for so in "$OUT"/lib/*.so*; do [ -e "$so" ] && { "$TC/bin/patchelf" --set-rpath '$ORIGIN' "$so" 2>/dev/null || true; }; done
    return 0
}

# ---- ccx_spooles: SPOOLES only (the distributed binary) - complete before anything optional starts
build_one ccx_spooles "-DUSE_MT=1 -DFEA_FAER=1 -I$FAERINC" "$FAERLIB -lpthread -ldl -lm -lrt -lutil -lc"
ship "$OUT/ccx_spooles"
# the marker of the checked factorization (the program allows several equation-solver threads only for a binary that carries it; spooles_threads_safe in
# fea_core/ccx/runner.py reads it from the file) – in the stripped, shipped file AND printed by `ccx -v`
grep -aqF 'FEA_SPOOLES_FACTOR_CHECK=2' "$OUT/ccx_spooles" || { echo "ERROR: the binary does not carry the marker FEA_SPOOLES_FACTOR_CHECK=2 (the patch ccx_2.23_spooles_serial_solve.patch was not applied or USE_MT is off)"; exit 1; }
VOUT="$("$OUT/ccx_spooles" -v 2>&1 || true)"          # `ccx -v` ends with a non-zero code: no pipe (pipefail), the text is looked at afterwards
case "$VOUT" in *FEA_SPOOLES_FACTOR_CHECK=2*) ;; *) echo "ERROR: ccx_spooles -v does not print the marker FEA_SPOOLES_FACTOR_CHECK=2 (the patch ccx_2.23_spooles_serial_solve.patch or USE_MT is missing)"; exit 1 ;; esac
# the marker of the ordered SPOOLES (printed only by a ccx that is linked with the SPOOLES of spooles_2.2_ordered_assembly.patch: the call of fea_spooles_ordered takes the address of FrontMtx_updateOrdered)
case "$VOUT" in *FEA_SPOOLES_ORDERED_ASSEMBLY=1*) ;; *) echo "ERROR: ccx_spooles -v does not print the marker FEA_SPOOLES_ORDERED_ASSEMBLY=1 (the ordered SPOOLES is not inside)"; exit 1 ;; esac
grep -aqF 'FEA_SPOOLES_ORDERED_ASSEMBLY=1' "$OUT/ccx_spooles" || { echo "ERROR: the binary does not carry the marker FEA_SPOOLES_ORDERED_ASSEMBLY=1"; exit 1; }
# the marker of the faer solver (faer.c is compiled in and the Rust library is linked: ccx -v prints the version of the library)
case "$VOUT" in *FEA_FAER=1*) ;; *) echo "ERROR: ccx_spooles -v does not print the marker FEA_FAER=1 (the faer solver is not inside)"; exit 1 ;; esac
grep -aqF 'FEA_FAER=1' "$OUT/ccx_spooles" || { echo "ERROR: the binary does not carry the marker FEA_FAER=1"; exit 1; }

OPT_RC=0
optional() {  # optional DESCRIPTION FUNCTION - a failed optional part must not take the finished ccx_spooles with it: the function runs in a
    # subshell with its own `set -e`; the call must be a plain command (inside `if` / `||` the shell ignores `set -e` in the subshell) and the
    # result is read from OPT_RC. The failure is reported and the script goes on.
    local what="$1" fn="$2"
    set +e
    ( set -euo pipefail; "$fn" )
    OPT_RC=$?
    set -e
    if [ "$OPT_RC" -ne 0 ]; then
        echo "WARNING: $what failed (code $OPT_RC) - it is left out; ccx_spooles is complete"
    fi
}

# ---- ccx_pastix: PaStiX 6.4.0 (LGPL-3.0, Inria), static, 32-bit integers (ccx ITG = int). BLAS / LAPACK / CBLAS / LAPACKE = OpenBLAS
# (pthreads), SCOTCH (int32) and hwloc from conda-forge (the lock file), copied next to the binary. Python only for the code generator of PaStiX.
pastix_part() {
    # digest verified 18. 11. 2026; a new version = a new digest, otherwise the build stops
    get_mirrors "$WORK/pastix.tar.gz" 891d426188eed56c1075fb34d2d80132593a1536ffc05cf333567f68a4811e55 \
        "https://files.inria.fr/pastix/releases/v6/pastix-6.4.0.tar.gz"
    create_locked_env "$DEPS" "$SCRIPT_DIR/ccx_deps.lock" "$DEPS_SPEC"
    local pxstamp="pastix-6.4.0; int32; static; $(sha256sum "$SCRIPT_DIR/ccx_deps.lock" | cut -d' ' -f1); $PMAP"
    if [ ! -f "$WORK/pastix/inst/lib/libpastix.a" ] || [ "$(cat "$WORK/pastix/.flags" 2>/dev/null)" != "$pxstamp" ]; then
        rm -rf "$WORK/pastix" && mkdir -p "$WORK/pastix/src"
        tar -xzf "$WORK/pastix.tar.gz" -C "$WORK/pastix/src" --strip-components=1
        (
            export PATH="$TC/bin:$DEPS/bin:$PATH" PKG_CONFIG_PATH="$DEPS/lib/pkgconfig"     # the toolchain cmake (a system cmake does not find hwloc)
            cmake -S "$WORK/pastix/src" -B "$WORK/pastix/build" -G "Unix Makefiles" \
                -DCMAKE_C_COMPILER="$CC" -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_MAKE_PROGRAM="$MAKE" -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$WORK/pastix/inst" -DBUILD_SHARED_LIBS=OFF -DPASTIX_INT64=OFF \
                -DPASTIX_WITH_FORTRAN=OFF -DPASTIX_WITH_MPI=OFF -DPASTIX_WITH_PARSEC=OFF -DPASTIX_WITH_STARPU=OFF -DPASTIX_WITH_CUDA=OFF \
                -DPASTIX_ORDERING_SCOTCH=ON -DPASTIX_ORDERING_METIS=OFF -DPASTIX_ORDERING_PTSCOTCH=OFF \
                -DBUILD_DOCUMENTATION=OFF -DBUILD_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                -DSCOTCH_DIR="$DEPS" -DHWLOC_DIR="$DEPS" -DCMAKE_PREFIX_PATH="$DEPS" -DBLA_VENDOR=OpenBLAS \
                -DBLAS_LIBRARIES="$DEPS/lib/libopenblas.so.0" -DLAPACK_LIBRARIES="$DEPS/lib/libopenblas.so.0" \
                -DCBLAS_LIBRARIES="$DEPS/lib/libopenblas.so.0" -DCBLAS_INCLUDE_DIRS="$DEPS/include" \
                -DLAPACKE_LIBRARIES="$DEPS/lib/libopenblas.so.0" -DLAPACKE_INCLUDE_DIRS="$DEPS/include" \
                -DCMAKE_C_FLAGS="-O2 -fPIC $PMAP" > "$WORK/pastix_cmake.log" 2>&1 \
            && cmake --build "$WORK/pastix/build" -j"$JOBS" >> "$WORK/pastix_cmake.log" 2>&1 \
            && cmake --install "$WORK/pastix/build" >> "$WORK/pastix_cmake.log" 2>&1
        ) || { tail -40 "$WORK/pastix_cmake.log"; return 1; }
        [ -f "$WORK/pastix/inst/lib/libpastix.a" ] || { tail -40 "$WORK/pastix_cmake.log"; return 1; }
        printf '%s' "$pxstamp" > "$WORK/pastix/.flags"
    fi
    local pxi="$WORK/pastix/inst"
    # pastix.c of ccx 2.23 targets the PaStiX4CalculiX fork; adapt it to the released PaStiX 6.4.0 (the file is fully inside #ifdef PASTIX)
    patch -p1 -i "$PATCH_DIR/ccx_${CCX_VER}_pastix_upstream.patch" > "$WORK/pastix_patch.log" 2>&1 || { cat "$WORK/pastix_patch.log"; return 1; }
    # OpenBLAS replaces the static reference BLAS / LAPACK here - with both on the link line the reference dgemm_ of the executable would shadow
    # OpenBLAS for the PaStiX kernels
    build_one ccx_pastix "-DPASTIX -I$pxi/include -I$DEPS/include" \
        "-Wl,--start-group $pxi/lib/libpastix.a $pxi/lib/libpastix_kernels.a $pxi/lib/libspm.a -Wl,--end-group -L$DEPS/lib -lscotch -lscotcherr -lhwloc -lz -lrt -ldl" \
        "-L$DEPS/lib -l:libopenblas.so.0"
    ship "$OUT/ccx_pastix"
}

PASTIX_STATE="skipped (FEA_CCX_PASTIX=0)"
if [ "$WITH_PASTIX" = 1 ]; then
    optional "ccx_pastix (PaStiX)" pastix_part
    if [ "$OPT_RC" -eq 0 ]; then PASTIX_STATE="built"; else PASTIX_STATE="NOT built"; fi
fi
echo "libraries:"; du -sh "$OUT/lib"; ls "$OUT" "$OUT/lib"

# ---- check: the path of the work folder must not stay in the binaries (e.g. an older compiler without -ffile-prefix-map);
# an error for the distribution archive, only a warning for a build for one's own installation
for f in "$OUT"/ccx_* "$OUT"/lib/*; do
    [ -f "$f" ] || continue
    if grep -qaF "$WORK" "$f"; then
        echo "WARNING: $(basename "$f") contains the path of the build work folder"
        [ -z "$ARCHIVE" ] || exit 1
    fi
done
# ---- check: the highest glibc version required (the program supports glibc from 2.35)
GLIBC_MAX=2.35
for b in "$OUT"/ccx_*; do
    [ -f "$b" ] || continue
    need=$("$TC/bin/x86_64-conda-linux-gnu-objdump" -T "$b" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
    echo "$(basename "$b"): requires glibc >= $need"
    if [ "$(printf '%s\n%s\n' "$need" "$GLIBC_MAX" | sort -V | tail -1)" != "$GLIBC_MAX" ]; then
        echo "WARNING: $(basename "$b") requires glibc $need > $GLIBC_MAX"
        [ -z "$ARCHIVE" ] || exit 1
    fi
done

# ---- distribution archive: without the user and group of the builder, a fixed time, the same order -> repeatable contents
if [ -n "$ARCHIVE" ]; then
    members=(); for f in ccx_pastix ccx_spooles lib; do [ -e "$OUT/$f" ] && members+=("$f"); done
    tar -C "$OUT" --format=gnu --sort=name --owner=0 --group=0 --numeric-owner \
        --mtime="@${SOURCE_DATE_EPOCH:-1792022400}" --mode='u+rwX,go+rX,go-w' -cf - "${members[@]}" | gzip -9n > "$ARCHIVE"
    echo "archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1)): ${members[*]}"
fi
echo "ccx_spooles: built; ccx_pastix: $PASTIX_STATE"

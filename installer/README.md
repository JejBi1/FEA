# Installing FEA Studio (Windows x64 and Linux x86_64)

FEA Studio installs **by modules**: the base solver core (STEP → mesh → CalculiX → results) is small; optional modules (GUI, fatigue and 3D printing, direct solver, Word reports, MCP server, large model solvers) can be added or removed at any time. Everything resides in the application directory (`.venv`, `tools`). No administrator privileges are required, and no system files are modified (except desktop shortcuts, `.fea` file association, and optional MS-MPI).

## Modules (`fea_core/modules.json`)

| Module | Contents | Disk Space* |
|---|---|---|
| `core` | FEA core, CLI, CalculiX (SPOOLES), Gmsh, OpenCASCADE; statics, heat transfer, modal, buckling, beams, 2D, shells, joints, welds | ~693 MB |
| `plots` | Charts and diagrams (matplotlib) | ~43 MB |
| `gui` | Graphical interface (PySide6 / Qt, without WebEngine) | ~80 MB |
| `accel` | Numba: fatigue (rainflow, critical plane), 3D printing, result images for MCP and reports | ~129 MB |
| `beams` | Custom beam section meshing (sectionproperties via Gmsh); EN standard profiles work without it | ~15 MB |
| `sketch` | DXF import into the sketcher (ezdxf, MIT; needs `beams` for shapely); the sketcher itself needs only `beams` | ~22 MB |
| `surrogate` | Surrogate model of a parametric study with calibrated 90 % error bands: Gaussian processes (scikit-learn, BSD-3); the model also works without it (scipy radial basis, same band calibration) | ~45 MB |
| `report` | Microsoft Word engineering reports (`.docx`) | ~11 MB |
| `motion` | Multibody dynamics (Exudyn worker process) | ~22 MB |
| `mcp` | Model Context Protocol server (180+ tools) and REST API for CAD adapters | ~42 MB |
| `direct` | Direct solver: the program's own stiffness matrix assembly (Numba, needs `accel`) + pure-Rust sparse LDLᵀ `rslab` (MIT, a pip wheel); one factorization for all load cases, also for a single one; no NGSolve | ~11 MB |
| `ngsolve` | NGSolve for the flexible bodies of mechanisms (needs `motion`); linear statics does not need it | ~170 MB |
| `hypre` | Large model iterative solver (hypre BoomerAMG + MPI): MS-MPI on Windows, MPICH on Linux | ~72 MB |
| `chrono` | Rigid/flexible contact in mechanisms (Project Chrono 10 in `tools/chrono_env`); manual install only | ~600 MB |
| `verify` | Verification benchmarks, pytest, pyNastran (the MYSTRAN cross-checks also need MYSTRAN, which you download yourself – no package carries it) | ~30 MB |
| `docs` | Developer tools: PDF manual generation (Qt WebEngine); pre-built PDFs are included in `docs/` | ~384 MB |

\* Approximate Windows footprint after slimming (`installer/slim.py`). Module dependencies resolve automatically (e.g., `hypre` → `accel` → `core`, `direct` → `accel` → `core`).

Profiles:
- **minimal**: `core` (~0.7 GB)
- **standard**: `core gui plots accel direct beams sketch report motion` (~1.1 GB; measured 1.09 GB after a clean installation, 1.49 GB with the former module `pardiso`)
- **full**: standard + `mcp ngsolve hypre surrogate` (~1.4 GB)

## Windows Installation

### Option A: Portable Archive (.zip)

1. Extract `FEAStudio-<version>.zip` to your target directory (avoid cloud-synced folders like OneDrive).
2. Run **`Install FEA Studio.bat`** and select an installation profile (minimal, standard, full, or custom).
   - Silent run: `Install FEA Studio.bat --profile standard --quiet`
   - Specific modules: `Install FEA Studio.bat gui accel`
3. Launch via the desktop shortcut or `FEA Studio.bat`. Drag-and-drop `.step` or `.fea` files directly onto the launcher or window.

### Option B: Windows Installer (setup.exe)

- Run `FEAStudio-<version>-setup.exe` to install to `%LOCALAPPDATA%\Programs\FEA Studio` without admin rights.
- Supports module selection, updates over existing installations, and clean uninstallation via Windows Settings → Apps.
- Silent flags: `FEAStudio-<version>-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /TYPE=standard /DIR="C:\FEA Studio"`
- Options: `/NOSHORTCUTS`, `/NOMCP`, `/NOFUSION`, `/NOCHECK`, `/FORCE`.

### Enterprise Networks and Offline Installations

- **No PowerShell required**: Both setup.exe and `.bat` execute via bundled Python without triggering restrictive PowerShell execution policies.
- **Corporate proxies & HTTPS inspection**: Installer uses the Windows certificate store (`UV_SYSTEM_CERTS=1`) and system proxy settings (`HTTPS_PROXY`).
- **Offline environments**: Use `FEAStudio-<version>-offline-setup.exe` (~0.7 GB), which bundles pre-cached wheels for the standard profile and installs without internet connectivity.

## Linux Installation

Supported distributions: x86_64, glibc ≥ 2.35 (Ubuntu 22.04+, Debian 12+, Fedora 36+, openSUSE 15.5+).

```bash
tar xzf FEAStudio-<version>.tar.gz && cd FEAStudio-<version>
./install.sh                      # Interactive prompt, or: ./install.sh --profile standard --quiet
./fea-studio                      # Launch GUI
```

The package carries the prebuilt CalculiX `ccx_spooles` (Windows: `ccx_spooles.exe` in `tools/calculix/ccx_2.23_windows_x86_64.zip`, built at release by `installer/windows/build_ccx.ps1` from pinned sources with a pinned MSYS2 toolchain – no download, no admin rights; Linux: built by `installer/linux/build_ccx.sh`; both also build the Rust static library of the faer solver `installer/faer_ccx` with the pinned Rust toolchain of `ccx_rust.lock` from the crates of `installer/faer_ccx/Cargo.lock`, every tarball and crate with its SHA-256, `cargo build --locked --offline`; `python installer/faer_ccx/make_crate_licences.py` regenerates the licence list of the crates); the installer only unpacks it. **Nothing is compiled at installation** (no C / Fortran compiler needed): the installer unpacks and downloads – `uv`, Python 3.11 and the Python wheels (`uv pip sync --require-hashes --no-build`: only wheels with the digests of the version lock), and, for the `hypre` module and missing system libraries, conda-forge binaries through `micromamba`. A package without the prebuilt CalculiX is refused at package build time (`make_package.py`: `--no-windows-ccx` / `--no-linux-ccx` are the explicit opt-outs); if you install from a source checkout, build the archive once (`powershell -ExecutionPolicy Bypass -File installer\windows\build_ccx.ps1`, 15-25 minutes; it keeps everything it downloads – the MSYS2 base, the 27 toolchain packages, the sources, all with their SHA-256 – in the local archive `tools/winccx_build/archive` and uses that before the network (`-Offline` proves it is complete, `python installer/windows/fetch_ccx_archive.py` fills or checks it); or the Linux script) or point `FEA_CCX` at your own `ccx` (or put it into `PATH`). An update removes the third-party `ccx_static.exe` of earlier versions (migration 1.4.2). The PaStiX build `ccx_pastix` (1.5–6× faster for statics, contact and heat) is NOT distributed until the licence question of PaStiX (LGPL-3.0) inside the GPL-2.0-only CalculiX is answered; `bash installer/linux/build_ccx.sh OUTPUT` builds it for your own use and the program prefers it when it is in `tools/calculix/ccx_2.23_linux`. Nothing of Intel (MKL) is shipped, downloaded or optional on either platform.

Nothing native is built at installation. The only compile step is a best-effort byte-compile of the program's own Python files (`compileall` of `fea_core` + `fea_gui`, 2–3 s – without it a read-only installation would pay about +1–2 s of ~7 s at every start; a failure is only a note). The numba kernels (fatigue, 3D printing, result pictures, dynamics – the numba cache is specific to the CPU model) are not precompiled: they are compiled at the first call and cached; the program says so in the log ("First use of … compiling the numerical kernels"). The first fatigue run with the critical plane can take up to a minute, every later one starts at once. `python -m fea_core.warmup` (without `--quick`) fills the cache on purpose.

### OpenMP runtime for numba (Linux)

The thread-safe `omp` layer of numba (module `accel`) needs the GNU OpenMP runtime `libgomp.so.1.0.0`. On a bare system (a container, a server image) the installer copies the runtime that the CalculiX package carries (`tools/calculix/*/lib/libgomp.so.1`) to `tools/python/<version>/lib/libgomp.so.1.0.0`. The interpreter has the RPATH `$ORIGIN/../lib`, which the dynamic loader searches first for the libraries of every extension module in every entry point (GUI, command line, MCP server, helper processes). Nothing is copied where numba already loads the layer; if numba does not load for another reason, the installer reports that reason instead of the missing runtime. `tests/test_install_omp_linux.py` proves the loading with the real loader in a mount namespace without the system libgomp (Linux, no root; skipped where user namespaces are not allowed).

If the system package `libgomp1` is installed **later**, the copy stays and is still found first. Both are GCC's `libgomp` with the SONAME `libgomp.so.1`; the loader joins them while the system library is requested afterwards by that name (measured: one runtime mapped). A process that loads the system `libgomp.so.1` **before** numba's OpenMP layer (another system library) holds two OpenMP runtimes: two thread pools, correct results, but cores can be oversubscribed (measured: both mapped). Nothing detects this at run time. After installing `libgomp1` delete `tools/python/*/lib/libgomp.so.1.0.0` to have only the system one; the installer never deletes the copy and creates it again only when numba cannot load the OpenMP layer.

## Licences and sources

Every package carries `licenses/` (texts of the third-party licences, `EXTERNAL.md`, `SOURCES.md`) and `licenses/sources/` with the sources of everything its `ccx_spooles.exe` / `ccx_spooles` is built from or shipped with: CalculiX 2.23, SPOOLES, ARPACK-ng, LAPACK, `rust-src` and the crates of faer; the Windows packages the MSYS2 source packages of the GCC / mingw-w64 run-time libraries that are linked into it (libquadmath is LGPL-2.1: `licenses/SOURCES.md` says how to relink with a modified one); the Linux packages the GCC archive of the pinned conda toolchain with the conda-forge recipe and patches (libgfortran, libgcc, `lib/libgomp.so.1`, `lib/libquadmath.so.0`; `installer/linux/ccx_gccsrc.lock` – a new toolchain needs a new line there, `make_package.py` refuses a recipe that does not name the pinned archive). `make_package.py` downloads them once into `tools/_download/sources` (SHA-256 checked); building the Linux packages needs a `tar` that reads `.tar.zst`. The offline setup also carries the Gmsh source. See `THIRD_PARTY_LICENSES.md`.

## Version numbers (binding)

**The next feature release after 1.4.2 is 1.5.1 – never 1.5.0.** The pre-release builds of the branch `next-1.5` were called 1.5.0 before the release got the number 1.4.2, and the state of an installation made by such a build (`.venv/fea_modules.json`, `"version": "1.5.0"`) is read as 1.4.2 while the program is older than 1.5.0 (`upgrade.PRERELEASE_NAMES`, `upgrade.state_version`, the `.run` header and `FEAStudio.iss` `InstalledVersion`). A real release 1.5.0 would be taken for 1.4.2 by the installations that carry that state: no update step would run for them. The tolerance stays (it protects the test installations); the number 1.5.0 is therefore never used for a release. `tests/test_update.py::test_the_program_is_never_version_1_5_0` fails if `fea_core.__version__` becomes exactly `1.5.0` (or any other pre-release name) and if `CHANGELOG.md` gets a `1.5.0` heading.

## Maintenance & Uninstallation

- **Add/remove modules**: `python -m fea_core.modules install <module>` or GUI menu Tools → Application Modules.
- **Update**: Run new setup.exe over old installation or `Install FEA Studio.bat update`.
- **Uninstall**: Run `fea_setup.py uninstall` or Windows Settings → Apps. Preserves user-created `.fea` files.

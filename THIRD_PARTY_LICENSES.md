# Third-party licenses

FEA Studio is licensed under the **GNU GPL v3 or later** (`LICENSE`), with an additional permission for
MS-MPI and CAD APIs (`LICENSE-EXCEPTIONS.md`). The components below keep their own licenses. Installation packages
contain the full license texts in the `licenses/` folder.

## What this means for you

- **You may use FEA Studio for any work, including commercial engineering work.**
- **You may share and modify it** under the GPL: keep the license, publish your changes under the GPL as well.
- **You may not make it closed-source.** Gmsh and CalculiX are GPL; a proprietary product would need other licenses.
- Your own CAD licenses apply to those programs (Fusion Personal, for example, is non-commercial).

## Components

| Component | License | Used for |
|---|---|---|
| CalculiX 2.23 | GPL-2.0 | Solver (separate program). Packages carry `ccx_spooles.exe` (Windows, `installer/windows/build_ccx.ps1`) and `ccx_spooles` (Linux, `installer/linux/build_ccx.sh`), both built at release from pinned sources, together with the CalculiX source archive |
| Inside `ccx_spooles.exe` / `ccx_spooles`: SPOOLES (with the NIST Harwell-Boeing reader), ARPACK-ng, LAPACK / BLAS (reference), GCC runtime (libgcc, libgfortran with libbacktrace, libgomp, libquadmath), mingw-w64 C runtime and winpthreads (Windows) | SPOOLES: public domain; ARPACK-ng, LAPACK / BLAS, libbacktrace: BSD-3; GCC runtime: GPL-3.0 with the GCC Runtime Library Exception (libquadmath: LGPL-2.1 or later – linked statically into the Windows `ccx_spooles.exe`, shipped as a shared library on Linux); mingw-w64: ZPL-2.1, winpthreads: MIT / BSD-3 | Equation solver, eigenvalues and run-time libraries of the solver (statically linked); texts in `licenses/external/` |
| Inside `ccx_spooles.exe` / `ccx_spooles` (F17): faer 0.24.4 and 117 other Rust crates (C interface `installer/faer_ccx`), Rust standard library 1.100.0 | MIT: faer, its C interface and every crate (a crate that is also Apache-2.0 is used under MIT, BSD-2-Clause or Zlib: CalculiX is GPL-2.0-only); faer contains ports of AMD and COLAMD of SuiteSparse and parts of LAPACK (BSD-3-Clause); the Rust standard library: MIT or Apache-2.0 (`compiler_builtins` also has code under Apache-2.0 with the LLVM exception) | Direct solver of symmetric systems (Cholesky / LDL^T / LBL^T) inside CalculiX, statically linked; texts and the table of all crates in `licenses/external/faer/` and `licenses/external/rust-std/` |
| Gmsh | GPL-2.0-or-later | Meshing (separate process); the offline setup also carries its source |
| Open CASCADE (cadquery-ocp), VTK | LGPL-2.1 with exception, BSD-3 | STEP import, 3D view |
| Qt 6 and Qt for Python (PySide6, shiboken6) | LGPL-3.0 (used under the LGPL-3.0; the wheels offer LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only, or a commercial licence); texts of the LGPL-3.0 and GPL-3.0 in `licenses/python/pyside6-essentials-*/` and `licenses/python/shiboken6-*/` | User interface (dynamically linked, replaceable libraries; installed from PyPI, in the offline setup as wheels) |
| NGSolve / Netgen | LGPL-2.1 | Matrices M and K of the flexible bodies of mechanisms (module `ngsolve`); the earlier matrix assembly of the direct and iterative solvers (debugging fallback `FEA_ASSEMBLY=ngsolve`) |
| rslab (derived from feral) | MIT | Direct sparse solver of the direct-solver backend (pure Rust, pip wheel, module `direct`) |
| hypre, mpi4py, MPICH | Apache-2.0 / MIT, BSD-3, BSD-like | Parallel solver (optional) |
| MS-MPI | Microsoft redistributable | Parallel solver on Windows (installed only with your consent) |
| numpy, scipy, numba, matplotlib | BSD / PSF-style | Computation and plots |
| sectionproperties, shapely | MIT, BSD-3 | Beam cross-sections; shapely also finds the regions of a sketch |
| ezdxf | MIT | Import of lines and arcs from DXF drawings into the sketcher (optional module `sketch`) |
| scikit-learn, joblib, threadpoolctl, narwhals, cloudpickle | BSD-3, BSD-3, BSD-3, MIT, BSD-3 | Optimiser of the hyperparameters of the Gaussian-process surrogate model of a parametric study (optional module `surrogate`; the prediction itself is numpy). Inside the wheels: vendored array-api-compat and array-api-extra (MIT), packaging (BSD-2-Clause), numpydoc, a SciPy copy (BSD-3), liac-arff (MIT) in scikit-learn and loky (BSD-3) in joblib; the Windows wheel of scikit-learn also carries `vcomp140.dll` and `msvcp140.dll` (Microsoft Visual C++ runtime, redistributable unmodified under the Microsoft terms quoted in its `COPYING`; the same runtime the setup installs for the other native modules); the Linux wheel carries `libgomp` (GPL-3.0-or-later with the GCC Runtime Library Exception) |
| Exudyn, Project Chrono | BSD-3 | Mechanisms (optional) |
| python-docx, lxml | MIT, BSD-3 | Word reports |
| mcp, fastapi, uvicorn, pydantic | MIT / BSD-3 | MCP server and REST API |
| pyNastran | BSD-3 | Verification only |
| Python, uv, micromamba | PSF, MIT / Apache-2.0, BSD-3 | Runtime and installer |

The exact versions are in `installer/locks/`.

## Not distributed: MYSTRAN (optional, downloaded by the user)

Some verification cross-checks (and the buckling of frames with the CalculiX U1 beam solver) call **MYSTRAN**, a third-party program. **No package of
FEA Studio carries it and the installer does not download it.** A user who wants these checks downloads it from https://www.mystran.com and saves it as
`tools/mystran/mystran(.exe)`; the licences of that download (MYSTRAN itself and the libraries built into its program) apply to it.

## Source code of the GPL / LGPL binaries

Every package carries `licenses/SOURCES.md` (where the complete source of each shipped GPL / LGPL binary is, with SHA-256
digests) and, in `licenses/sources/`, the sources of everything its CalculiX program is built from or shipped with – the sources travel with the
binaries, there is no written offer (a build that leaves them out is a development build: its files are named `-dev` and its `SOURCES.md` says NOT FOR
DISTRIBUTION). They are: the CalculiX 2.23 source archive (https://www.dhondt.de/ccx_2.23.src.tar.bz2), SPOOLES (with the correction of dhondt.de), ARPACK-ng
and LAPACK, the Rust standard library (`rust-src`) and the crates of the faer solver, and the GCC run-time libraries:
- the Windows packages: the MSYS2 source packages of the GCC / mingw-w64 run-time libraries that are linked into `ccx_spooles.exe`
  (libquadmath is LGPL-2.1+; `SOURCES.md` says how to relink with a modified one);
- the Linux packages: the GCC 16.2.0 archive of the pinned conda-forge toolchain with the conda-forge recipe and its patches (libgfortran and
  libgcc linked into `ccx_spooles`; `lib/libgomp.so.1` and `lib/libquadmath.so.0` shipped as replaceable shared libraries; the archive and its
  digest are pinned in `installer/linux/ccx_gccsrc.lock`).

The offline setup additionally carries the Gmsh 4.15.2 source (https://gmsh.info/src/). The build scripts of the two CalculiX programs, with the
digests of everything they download (the sources, and the MSYS2 / conda-forge compiler packages), are `installer/windows/build_ccx.sh`
(+ `ccx_toolchain.lock`, `patches/`) and `installer/linux/build_ccx.sh`. The Rust part of the solver (faer and its crates, the Rust standard
library) is built by the same scripts with the pinned Rust toolchain of `installer/*/ccx_rust.lock` from the crates of
`installer/faer_ccx/Cargo.lock` (SHA-256 of every crate and every tarball).

## Notices required by third-party licences

The CalculiX program of the Windows packages is built by the project from the sources named above; it contains the libraries below, linked statically
(the texts of their licences are in `licenses/external/`). This is not legal advice. The third-party `ccx_static.exe` of earlier versions
(it contained TAUCS with AMD / COLAMD / GENMMD, PaStiX, Scotch and hwloc) is no longer used or shipped. Notices that the licences ask to be cited:

- **SPOOLES 2.2** (equation solver): Cleve Ashcraft, Roger Grimes, Joseph Liu, Jim Patterson, Dan Pierce, Yichi Pierce, Peter Schartz, Juergen Schulze,
  Wei-Pai Tang, David Wah, Jason Wu (Boeing Phantom Works and others); development funded by DARPA and the DoD. "This release of the package is totally within
  the public domain; there are absolutely no licensing restrictions ... All we ask is an acknowledgement in derivative codes and any publications from research
  that uses this software." (`licenses/external/spooles/`). SPOOLES contains the Harwell-Boeing file reader of the National Institute of Standards and Technology
  (K. A. Remington, `iohb.c`): "Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby
  granted provided that the above copyright notice appear in all copies and that both the copyright notice and this permission notice appear in supporting
  documentation. Neither the Author nor the Institution (National Institute of Standards and Technology) make any representations about the suitability of this
  software for any purpose. This software is provided "as is" without expressed or implied warranty."
- **ARPACK-ng 3.9.1** (eigenvalues): BSD licence, Copyright (c) 1996-2008 Rice University (D. C. Sorensen, R. B. Lehoucq, C. Yang, K. Maschhoff) and the
  arpack-ng contributors (`licenses/external/arpack-ng/`).
- **LAPACK / BLAS 3.12.0** (reference implementation): modified BSD licence, Copyright (c) 1992-2023 The University of Tennessee and The University of
  Tennessee Research Foundation, 2000-2023 The University of California Berkeley, 2006-2023 The University of Colorado Denver
  (`licenses/external/lapack-blas/`).
- **GCC run-time libraries** (libgcc, libgfortran, libgomp: GPL-3.0 with the GCC Runtime Library Exception 3.1, linked statically into the Windows program)
  and **libbacktrace** (part of libgfortran): BSD-3-Clause, Copyright (C) 2012-2016 Free Software Foundation, Inc. (`licenses/external/libgomp-libquadmath/`,
  `licenses/external/libbacktrace/`).
- **libquadmath** (GCC, LGPL-2.1 or later) is **inside the Windows `ccx_spooles.exe`**, linked statically: 19 objects that libgfortran pulls in for the
  formatted output of REAL(16) (found with the link map of the build; an earlier version of this file and of the F13 plan said it was not there – wrong). The
  Linux package ships it as the shared library `lib/libquadmath.so.0` (with `lib/libgomp.so.1`; both can be replaced; their source, the GCC archive and the
  conda-forge recipe, is in `licenses/sources/` of the Linux packages). The licence text (`libquadmath_COPYING.LIB`) and the copyright notices of the 19 source files
  (`NOTICE-libquadmath.txt`) are in `licenses/external/libgomp-libquadmath/`. LGPL-2.1 section 6 for a statically linked program: the Windows packages carry the
  source of the library (the MSYS2 source package of GCC 16.2.0 in `licenses/sources/`), the source of CalculiX, SPOOLES, ARPACK-ng and LAPACK, and the build scripts,
  and the build can link the program with a modified libquadmath (`installer/windows/build_ccx.ps1 -LibQuadmathSource`, described in `licenses/SOURCES.md`).
  (Not legal advice: the reading is that the combined program stays under the GPL-2.0 of CalculiX and the LGPL asks for the notices, the licence text and the
  possibility to relink – to be confirmed.)
- **mingw-w64** C runtime (ZPL-2.1) and **winpthreads** (MIT and BSD-3, Lockless Inc.) (`licenses/external/mingw-w64/`).
- **faer 0.24.4** (MIT, Copyright (c) sarah quiñones; Rust, linked statically as `libfea_faer.a` through the C interface `installer/faer_ccx`, MIT): the supernodal Cholesky, LDL^T and
  Bunch-Kaufman factorizations and their solves that `ccx_spooles` uses before SPOOLES (the ordering is still made by SPOOLES). faer contains **ports of AMD and COLAMD of SuiteSparse**
  (BSD-3-Clause, Copyright (c) 1996-2022 Timothy A. Davis, Patrick R. Amestoy, Iain S. Duff; COLAMD: Copyright 1998-2022 Timothy A. Davis) and of parts of **LAPACK** (BSD-3-Clause,
  the licence of reference LAPACK above); the notices are `COPYING.SUITE_SPARSE.AMD.BSD`, `COPYING.SUITE_SPARSE.COLAMD.BSD`, `COPYING.LAPACK.BSD` in
  `licenses/external/faer/crates/faer-0.24.4/`. The crate also lists `COPYING.EIGEN.MPL2` (algorithms ported from Eigen, MPL-2.0: the divide and conquer SVD of its dense decompositions, which the
  program does not call and the linker removes). The other crates (rayon, crossbeam, gemm / pulp / nano-gemm kernels, bytemuck, ...) are MIT, or dual licensed with MIT;
  `licenses/external/faer/CRATES.md` lists all 118 with their licence, the licence that is used and the SHA-256 digest of each crate, and the texts of every crate are in that folder.
- **Rust standard library 1.100.0** (statically linked with the Rust code above): MIT or Apache-2.0, used under the MIT licence (CalculiX is GPL-2.0-only and cannot take code that is only
  Apache-2.0); `compiler_builtins` also contains code under Apache-2.0 with the LLVM exception (the LLVM project states that this exception makes the licence compatible with
  GPL-2.0; accepted by the owner on 25. 11. 2026). Texts: `licenses/external/rust-std/` (`COPYRIGHT-library.html` is the notice file of Rust for the standard library with the
  licences of its third-party parts); the source: `rust-src` in `licenses/sources/`.

## Data

- **PragTic fatigue database** – *Papuga, J.: PragTic – material database of fatigue properties (Fatigue Data
  Repository). Czech Technical University in Prague, Faculty of Mechanical Engineering. https://www.pragtic.com/vmat.php*
  The database has no license for redistribution, so it is **not included**. You can download it for your own use:
  `python -m fea_core.fatigue.pragtic_fetch`.
- Materials, friction coefficients, steel sections, bolt and weld tables: factual values with the source given for each
  entry in the program. Texts of standards are not included.

### Sources of the material and friction data

Part of the built-in material library and of the other tables was taken from these databases, data sheets and
publications (each material in the program shows its own source):

- **Standards** – EN 10025-2, EN 10025-6, EN ISO 683-1, -2, -17, EN 10084, EN 10088-1, -3, EN 10089,
  EN ISO 4957, EN 1561, EN 1563, EN 754-2, EN 755-2, EN 1706, EN 12163, EN 12164, EN 13601, ASTM B94, B221, B348, B637,
  F2924, AMS 5662, ISO 5832-3, EN 1993-1-1, EN 1993-1-2, EN 1993-1-8, EN 1090-2, DIN 68364.
- **Handbooks** – MIL-HDBK-17-2F (Composite Materials Handbook); ASM Handbook; USDA Forest Products Laboratory: Wood
  Handbook.
- **Books** – Daniel, I. M., Ishai, O.: Engineering Mechanics of Composite Materials. Barbero, E. J.: Finite Element
  Analysis of Composite Materials. Gibson, L. J., Ashby, M. F.: Cellular Solids (1997).
- **Manufacturer data sheets** – metal 3D printing: Renishaw (RenAM 500 material data sheets), AddUp (Ti6Al4V ELI),
  3D Systems (Certified Scalmalloy), APWORKS; plastic 3D printing: Bambu Lab, Prusa Research (Prusament), Stratasys
  (Ultem 9085), EOS (PA 2200), ThermaX (PEEK); others: SSAB (Hardox 450), Toray (M40J).
- **Papers** – Gao et al. 2025, Materials 18(15):3475, doi:10.3390/ma18153475 (Ti-6Al-4V L-PBF);
  Xie et al. 2023, J. Mater. Res. Technol. 23:3322–3336 (CuCrZr L-PBF);
  Ezeh, O. H., Susmel, L.: Int. J. Fatigue 126 (2019) 319–326 (fatigue of printed PLA);
  Song et al., Mater. Des. 123 (2017) and Khosravani et al., Sci. Rep. 12 (2022) 1016 (anisotropy of printed lines);
  Waseem et al. 2020 (creep of PLA); Stoimenov et al. 2024 (friction).
- **Friction coefficients** – Roymech (Coefficients of friction), The Engineering ToolBox (Friction coefficients),
  Bayer: Snap-Fit Joints for Plastics, Ensinger, DuPont (Delrin), EN 1090-2 and EN 1993-1-8 (slip factors).
- **Grade converter** – designations compiled from the standards (EN 10027 and national standards); Czech ČSN pairs
  checked against cadforum.cz (the table itself is not copied).

Only the numerical values are used; the values are approximate and must be checked against the data of the actual
material supplier.

## Not included

Autodesk Fusion, Inventor and SOLIDWORKS are not part of FEA Studio. The add-ins and macros for them are GPL code;
the CAD programs need their own licenses. The native reader of `.ipt`, `.iam`, `.f3d` files is independent code without
any Autodesk code or libraries; always check critical models against a STEP export.

Trademarks belong to their owners. FEA Studio is not affiliated with them. This summary is not legal advice.

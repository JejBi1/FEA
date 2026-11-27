# FEA Studio – public verification set

Every case below is compared with a **reference that does not come from FEA Studio**: a closed-form solution, a handbook or standard formula, a published solution, or the open-source solver MYSTRAN. For each case either a work is cited, or - for an elementary relation - the first-principles relation is stated (such a case is marked *relation stated, no work cited*). A case without such a reference is not published here (the program's own equilibrium and bookkeeping checks are part of the test suite, not of this page).

**115 of 117 published cases are within tolerance**, 2 were not run on this platform (479 checks listed below).

- Program: FEA Studio 1.5.1
- Date of the run (UTC): 2026-11-26 10:54
- CalculiX: CalculiX 2.23 (ccx_spooles.exe, sha256 c2532a0695675a92…)
- MYSTRAN: MYSTRAN Version 19.0.0
- Platform: Windows 10.0.26300 (AMD64) · Python 3.11.17 · 16 logical CPUs
- Run profile: full
- Sum of the case run times: 31 min

Files: [`index.html`](index.html) (the same page, self-contained), [`results.csv`](results.csv) (all rows). Regenerate with `python -m verification.run_public` (runs the cases, needs CalculiX and Gmsh, MYSTRAN for the MYSTRAN cases and the Chrono module for MB4C, MB4R and KK5) or, from stored results, `python -m verification.public_report`. The CI page (`.github/workflows/verification.yml`) is built without the Chrono module, so it lists those three cases as not run; this page carries their results from a run with the module.

How to read the tables: *difference* is |FEA Studio − reference| / |reference|, or the absolute difference where the reference is zero; *tolerance* is the limit the case enforces; *mesh* is the node count and global element size of the finite-element model (second-order tetrahedra; beam, 2D and shell cases have their own discretisation). A case is within tolerance only when all its checks pass, including a few internal consistency checks that are not listed. In `results.csv` *difference* and *tolerance* are fractions (0.01 = 1 %) where `basis` is `relative` and absolute numbers in the unit of the quantity where it is `absolute`.

MYSTRAN is a separately downloaded third-party program: FEA Studio does not ship it (download: www.mystran.com). The MYSTRAN rows of this page come from the run named above; on a computer without it the cases whose reference is MYSTRAN are listed as not run, and the cross-check of the other cases is marked as not run.

## Linear statics

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **V1** ✔ | Cantilever 200×20×10 mm, end load 100 N | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | 22 137 nodes, h = 2.5 mm |
| | tip deflection (Timoshenko) | 0.7634 mm | 0.75794 mm | 0.715 % | 3 % | |
| | σx top fibre at x = 100 mm | 30 MPa | 30.007 MPa | 0.0245 % | 5 % | |
| | σx bottom fibre at x = 100 mm (magnitude) | 30 MPa | 30.015 MPa | 0.0514 % | 5 % | |
| | reaction Fz | 100 N | 100 N | < 0.001 % | 0.1 % | |
| **V2** ✔ | Bar 10×10×100 mm under uniform compression | *closed-form solution* (relation stated, no work cited): Uniaxial stress state: σ = F/A, δ = σ·L/E (Hooke's law) | | | | 4 253 nodes, h = 3 mm |
| | σx (centre) | -100 MPa | -99.999 MPa | 0.001 % | 0.1 % | |
| | largest \|σx\| over the nodes | 100 MPa | 99.999 MPa | 0.001 % | 0.1 % | |
| | end-face displacement ux | -0.047619 mm | -0.047619 mm | < 0.001 % | 0.1 % | |
| | reaction Fx | 10000 N | 10000 N | < 0.001 % | 0.1 % | |
| **V3** ✔ | Plate with a Ø10 hole under tension (1/8 model) | *handbook / standard*: R. E. Peterson: Stress Concentration Factors (Wiley) – plate of finite width with a central circular hole, polynomial of Heywood for K_t on the net section | | | | 12 486 nodes, h = 4 mm |
| | σx max at the hole, mean through the thickness (K_t,net) | 302.35 MPa | 305.31 MPa | 0.981 % | 5 % | |
| | σx max at the hole, mid-plane | 302.35 MPa | 312.07 MPa | 3.22 % | 5 % | |
| | reaction Fx (100 MPa × 50 × 2.5 mm) | 12500 N | 12500 N | < 0.001 % | 0.1 % | |
| **V4** ✔ | Thick-walled tube under internal pressure (Lamé) | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) | | | | 5 895 nodes, h = 2.5 mm |
| | σθ at r_i | 166.67 MPa | 166.32 MPa | 0.206 % | 5 % | |
| | σθ at r_o | 66.667 MPa | 66.664 MPa | 0.00458 % | 5 % | |
| | von Mises at r_i | 233.33 MPa | 231.02 MPa | 0.993 % | 5 % | |
| | radial displacement u(r_i), mean | 0.01873 mm | 0.01873 mm | 0.00243 % | 3 % | |
| **V5** ✔ | Column 100×100×1000 mm under its own weight | *independent solver (MYSTRAN)*: MYSTRAN (open-source Nastran-dialect solver): the same mesh (coarsened by a factor of 2) and nodal loads solved with an independent assembly and solver; the displacement field and the reactions are compared – see the cross-check table | | | | 6 953 nodes, h = 25 mm |
| **V6** ✔ | Cantilever of V1 split into two bonded parts | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | 22 234 nodes, h = 2.5 mm |
| | tip deflection (Timoshenko) | 0.7634 mm | 0.75795 mm | 0.714 % | 3 % | |
| | σx top fibre at x = 100 mm | 30 MPa | 30.017 MPa | 0.057 % | 5 % | |
| | σx bottom fibre at x = 100 mm (magnitude) | 30 MPa | 30.004 MPa | 0.013 % | 5 % | |
| | reaction Fz | 100 N | 100 N | < 0.001 % | 0.1 % | |
| **V7** ✔ | Shaft Ø20×200 mm in torsion, T = 100 N·m (distributing coupling) | *closed-form solution*: Saint-Venant torsion of a circular shaft: τ = 16·T/(π·d³), φ = T·L/(G·J) (S. P. Timoshenko: Strength of Materials) | | | | 28 909 nodes, h = 2.5 mm |
| | surface shear stress τ at x = 100 mm (mean around the circumference) | 63.662 MPa | 63.894 MPa | 0.365 % | 5 % | |
| | twist of the end at x = 200 mm | 0.90321 ° | 0.9032 ° | < 0.001 % | 3 % | |
| | twist between x = 50 and 150 mm (×2) | 0.90321 ° | 0.90322 ° | 0.00115 % | 3 % | |
| | reaction moment Mx at the clamp | 1e+05 N·mm | 1e+05 N·mm | < 0.001 % | 0.1 % | |
| **V7R** ✔ | Shaft Ø20×200 mm in torsion, T = 100 N·m (rigid coupling) | *closed-form solution*: Saint-Venant torsion of a circular shaft: τ = 16·T/(π·d³), φ = T·L/(G·J) (S. P. Timoshenko: Strength of Materials) | | | | 28 909 nodes, h = 2.5 mm |
| | surface shear stress τ at x = 100 mm (mean around the circumference) | 63.662 MPa | 63.894 MPa | 0.365 % | 5 % | |
| | twist of the end at x = 200 mm | 0.90321 ° | 0.90321 ° | < 0.001 % | 3 % | |
| | twist between x = 50 and 150 mm (×2) | 0.90321 ° | 0.90322 ° | 0.00115 % | 3 % | |
| | reaction moment Mx at the clamp | 1e+05 N·mm | 1e+05 N·mm | < 0.001 % | 0.1 % | |
| **V8** ✔ | Tube with a radially fixed bore under external pressure | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) – external pressure, bore fixed radially (ε_θ = 0 at r_i), constants derived in the case | | | | 5 895 nodes, h = 2.5 mm |
| | radial displacement u_r(r_o), mean | -0.0088136 mm | -0.0088135 mm | < 0.001 % | 1 % | |
| | σr at r_i (45°, z = h/2) | -135.59 MPa | -134.68 MPa | 0.673 % | 5 % | |
| | σθ at r_i = ν·σr (ε_θ = 0) | -40.678 MPa | -40.786 MPa | 0.265 % | 5 % | |
| | σθ at r_o | -76.271 MPa | -76.271 MPa | < 0.001 % | 5 % | |
| | axial displacement of the top edge of the bore (axially free) | 0.0025182 mm | 0.0025178 mm | 0.0153 % | 1 % | |
| | constraint reaction Fx = −σr(r_i)·r_i·h | 27119 N | 27119 N | < 0.001 % | 1 % | |
| **V10** ✔ | Remote supports on a cantilever, simply supported beam with a hinge | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | 22 137 nodes, h = 2.5 mm |
| | (a) rigid, point at the centroid: reaction My at the point = −F·lever arm | 20000 N·mm | 20000 N·mm | < 0.001 % | 0.01 % | |
| | (b) rigid, point 50 mm behind the face: reaction My at the point = −F·lever arm | 25000 N·mm | 25000 N·mm | < 0.001 % | 0.01 % | |
| | (a') distributing, point at the centroid: reaction My at the point = −F·lever arm | 20000 N·mm | 20000 N·mm | < 0.001 % | 0.01 % | |
| | (b') distributing, point 50 mm behind the face: reaction My at the point = −F·lever arm | 25000 N·mm | 25000 N·mm | < 0.001 % | 0.01 % | |
| | (a) tip deflection (Timoshenko) | 0.76339 mm | 0.75794 mm | 0.714 % | 3 % | |
| | (a') tip deflection (Timoshenko) | 0.76339 mm | 0.76328 mm | 0.0141 % | 1 % | |
| | (c) distributing hinge: mid-span deflection (Timoshenko) | 1.1979 mm | 1.1954 mm | 0.209 % | 3 % | |
| | (c) distributing hinge: rotation of the point θy = q·L³/(24EI) | 0.019048 rad | 0.019033 rad | 0.0768 % | 3 % | |
| | (c) distributing hinge: reaction Fz at the point = q·L/2 | 2000 N | 2000 N | < 0.001 % | 0.01 % | |
| | (c) distributing hinge: moment at the point ≈ 0 (as 1 + \|M\| / (q·L/2·L)) | 1 | 1 | < 0.001 % | 0.001 % | |
| | (c) distributing hinge: end reaction Fz = q·L/2 | 2000 N | 2000 N | < 0.001 % | 0.01 % | |
| | (c') rigid hinge: mid-span deflection (Timoshenko) | 1.1979 mm | 1.1953 mm | 0.214 % | 3 % | |
| | (c') rigid hinge: rotation of the point θy = q·L³/(24EI) | 0.019048 rad | 0.019023 rad | 0.13 % | 3 % | |
| | (c') rigid hinge: reaction Fz at the point = q·L/2 | 2000 N | 2000 N | < 0.001 % | 0.01 % | |
| | (c') rigid hinge: moment at the point ≈ 0 (as 1 + \|M\| / (q·L/2·L)) | 1 | 1 | < 0.001 % | 0.001 % | |
| | (c') rigid hinge: end reaction Fz = q·L/2 | 2000 N | 2000 N | < 0.001 % | 0.01 % | |
| **V11** ✔ | Force on an edge and at a point (cantilever of V1) | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | 22 200 nodes, h = 2.5 mm |
| | (a) edge of the end face: tip deflection (Timoshenko) | 0.76339 mm | 0.75795 mm | 0.712 % | 3 % | |
| | (a) edge of the end face: reaction Fz | 100 N | 100 N | < 0.001 % | 0.001 % | |
| | (a) edge of the end face: reaction moment My = −F·L (about the origin) | -20000 N·mm | -20000 N·mm | < 0.001 % | 0.001 % | |
| | (c) point: tip deflection (Timoshenko) | 0.76339 mm | 0.75794 mm | 0.714 % | 3 % | |
| | (c) point: reaction Fz | 100 N | 100 N | < 0.001 % | 0.001 % | |
| | (c) point: reaction moment My = −F·L (about the origin) | -20000 N·mm | -20000 N·mm | < 0.001 % | 0.001 % | |
| | (d) two points: tip deflection (Timoshenko) | 0.76339 mm | 0.75796 mm | 0.712 % | 3 % | |
| | (d) two points: reaction Fz | 100 N | 100 N | < 0.001 % | 0.001 % | |
| | (d) two points: reaction moment My = −F·L (about the origin) | -20000 N·mm | -20000 N·mm | < 0.001 % | 0.001 % | |
| | (b) top edge: tip deflection q·L⁴/(8EI) + shear | 0.57291 mm | 0.56722 mm | 0.994 % | 3 % | |
| | (b) reaction Fz = q·L | 200 N | 200 N | < 0.001 % | 0.001 % | |
| | (b) reaction moment My = −q·L²/2 (about the origin) | -20000 N·mm | -20000 N·mm | < 0.001 % | 0.001 % | |
| | (b) reaction moment Mx = 0 (edge at y = 0), as 1 + Mx/(q·L²/2) | 1 | 1 | < 0.001 % | 0.001 % | |
| **V12** ✔ | Local and cylindrical coordinate systems (shaft of V7) | *closed-form solution*: Saint-Venant torsion: φ = T·L/(G·J) (S. P. Timoshenko: Strength of Materials); statics of the tangential load | | | | 28 909 nodes, h = 2.5 mm |
| | (a) twist between x = 50 and 150 mm (×2) = T·L/(G·J) | 0.90321 ° | 0.90329 ° | 0.00964 % | 3 % | |
| | (a) reaction moment Mx = ⅔·F_θ·R | 1e+05 N·mm | 1e+05 N·mm | < 0.001 % | 0.1 % | |
| **V13** ✔ | Variable pressure: hydrostatics with a free surface, linear pressure | *closed-form solution*: Hydrostatic load q = ρ·g·b·h on a cantilever strip: Timoshenko beam theory (S. P. Timoshenko: Strength of Materials); resultant and moment of the pressure by integration | | | | 33 731 nodes, h = 5 mm |
| | (a) full height: deflection of the top | 2.3381 mm | 2.331 mm | 0.305 % | 2 % | |
| | (a) full height: reaction Fy = −ρ·g·b·h²/2 | 98.1 N | 98.1 N | < 0.001 % | 0.01 % | |
| | (a) full height: reaction moment Mx = ρ·g·b·h³/6 (about the origin) | 32700 N·mm | 32700 N·mm | < 0.001 % | 0.01 % | |
| | (b) liquid level h = 500 mm: deflection of the top | 0.16453 mm | 0.16369 mm | 0.51 % | 2 % | |
| | (b) liquid level h = 500 mm: reaction Fy = −ρ·g·b·h²/2 | 24.525 N | 24.525 N | < 0.001 % | 0.01 % | |
| | (b) liquid level h = 500 mm: reaction moment Mx = ρ·g·b·h³/6 (about the origin) | 4087.5 N·mm | 4087.5 N·mm | < 0.001 % | 0.01 % | |
| | (c) linear pressure: reaction Fy = −∫p dA | 120 N | 120 N | < 0.001 % | 0.001 % | |
| | (c) linear pressure: reaction moment Mx = ∫p·z dA | 43333 N·mm | 43333 N·mm | < 0.001 % | 0.001 % | |
| **V14** ✔ | Rotation: spinning disc, angular acceleration of a bar, selected parts only | *closed-form solution*: Rotating disc with a bore (Lamé / Timoshenko & Goodier: Theory of Elasticity); rotating bar: d'Alembert load ρ·A·α·x, Timoshenko beam theory | | | | 25 314 nodes, h = 4 mm |
| | (a) disc: σθ(r_i) | 65.312 MPa | 65.255 MPa | 0.0876 % | 1 % | |
| | (a) disc: σθ(r_o) | 16.328 MPa | 16.327 MPa | 0.00621 % | 1 % | |
| | (a) disc: u_r(r_o) = r_o·σθ(r_o)/E | 0.0077752 mm | 0.0077751 mm | 0.00226 % | 1 % | |
| | (b) bar, angular acceleration α: tip deflection 11·q₀·L⁴/(120EI) + shear | 2.5738 mm | 2.5592 mm | 0.567 % | 2 % | |
| | (b) bar, α: reaction Fy = ρ·A·α·L²/2 | 62.8 N | 62.8 N | < 0.001 % | 0.01 % | |
| | (b) bar, α: reaction moment Mz = α·J_z (about the axis) | 4188.3 N·mm | 4188.3 N·mm | < 0.001 % | 0.01 % | |
| | (c) bar, angular velocity ω: reaction Fx = −ρ·ω²·A·L²/2 | 628 N | 628 N | < 0.001 % | 0.01 % | |
| | (c) bar, ω: elongation ρ·ω²·L³/(3E) | 0.01246 mm | 0.012427 mm | 0.269 % | 1 % | |
| **V15** ✔ | Elastic support: block on a foundation, beam on a Winkler foundation | *published solution*: M. Hetényi: Beams on Elastic Foundation – w₀ = P·λ/(2·k_b), M(x) = P/(4λ)·e^(−λx)·(cos λx − sin λx), λ = (k_b/(4EI))^¼ | | | | 33 854 nodes, h = 5 mm |
| | (a) block: bottom displacement u_z = −p/k | 0.01 mm | 0.010001 mm | 0.0103 % | 1 % | |
| | (a) block: foundation reaction = p·A | 100 N | 100 N | < 0.001 % | 0.01 % | |
| | (b) beam: deflection under the load w₀ = P·λ/(2k_b) (Hetényi) | 0.16251 mm | 0.16351 mm | 0.613 % | 3 % | |
| | (b) beam: σx bottom fibre 40 mm from the load = M/W (Hetényi) | 15.774 MPa | 15.788 MPa | 0.0888 % | 3 % | |
| | (b) beam: σx top fibre 40 mm from the load = −M/W | 15.774 MPa | 15.787 MPa | 0.0809 % | 3 % | |
| | (b) beam: foundation reaction = P | 1000 N | 1000 N | < 0.001 % | 0.01 % | |
| **V16** ✔ | Orthotropic material: rotated axes in tension, cylindrical orthotropy (ring) | *published solution*: S. G. Lekhnitskii: Theory of Elasticity of an Anisotropic Body (cylindrically orthotropic ring; transformation of the compliance to rotated axes) | | | | 76 389 nodes, h = 2.5 mm |
| | (a) bar with rotated orthotropy: ε_x | 4.5125e-4 | 4.5125e-4 | < 0.001 % | 1 % | |
| | (a) ε_y (Poisson effect in the rotated axes) | -2.8125e-4 | -2.8124e-4 | 0.00206 % | 2 % | |
| | (a) γ_xy/2 (shear from the rotation of the axes) | -2.3166e-4 | -2.3166e-4 | < 0.001 % | 2 % | |
| | (b) ring: outer radial displacement u_r (Lekhnitskii) | 0.041975 mm | 0.041975 mm | < 0.001 % | 1 % | |
| **V17** ✔ | Separately meshed part joined by *TIE (split cantilever, non-conforming mesh) | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | 30 524 nodes, h = 2.5 mm |
| | tip deflection (Timoshenko, as V1) | 0.7634 mm | 0.75798 mm | 0.71 % | 3 % | |
| | σx top fibre at x = 50 mm (part A) | 45 MPa | 45.013 MPa | 0.0287 % | 5 % | |
| | σx top fibre at x = 150 mm (part B, other mesh) | 15 MPa | 14.998 MPa | 0.0161 % | 5 % | |
| **V18** ✔ | Results along a line and stress linearization (Lamé tube, beam) | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity); membrane + bending decomposition of the stress over the wall (integrals evaluated numerically in the case) | | | | 27 972 nodes, h = 2.5 mm |
| | tube: membrane σθ = p·r_i/t (equilibrium) | 100 MPa | 99.937 MPa | 0.0631 % | 1 % | |
| | tube: bending part of σθ = 6/t²·∫σθ(t/2 − x)dx (Lamé) | 45.482 MPa | 45.32 MPa | 0.356 % | 2 % | |
| | tube: membrane σr = mean of (−p … 0) | -33.333 MPa | -33.264 MPa | 0.209 % | 2 % | |
| | cantilever at x = 200 mm: bending σx at the bottom fibre = −M/W (M = 100·100 N·mm) | 30 MPa | 30.026 MPa | 0.0874 % | 2 % | |
| | cantilever: membrane σx ≈ 0 (shown as 30 + σx) | 30 MPa | 29.99 MPa | 0.0328 % | 0.5 % | |
| | cantilever: σx along the height – bottom fibre (compression) | 30 MPa | 30.025 MPa | 0.0834 % | 3 % | |
| **CS1** ✔ | Local cylindrical coordinate system: Lamé stresses in a tube | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) | | | | 5 895 nodes, h = 2.5 mm |
| | σθ at the inner surface | 166.67 MPa | 166.68 MPa | 0.00846 % | 5 % | |
| | σθ at the outer surface | 66.667 MPa | 66.669 MPa | 0.0032 % | 5 % | |
| | σr at the inner surface (−100 MPa pressure) | -100 MPa | -98.83 MPa | 1.17 % | 7 % | |
| | σr at the outer surface (0 MPa) | 0 MPa | 0.094884 MPa | 0.094884 MPa | 0.5 MPa | |
| | radial displacement u_r at the inner surface | 0.01873 mm | 0.01873 mm | 0.00195 % | 5 % | |
| **AS1** ✔ | Stress tool: Tresca and Mohr-Coulomb safety in torsion of V7 | *closed-form solution*: Pure shear: τ = 16·T/(π·d³); safety = R_e/(√3·τ) (von Mises), R_e/(2·τ) (Tresca), Mohr-Coulomb 1/(τ/R_m + τ/R_c) | | | | 28 909 nodes, h = 2.5 mm |
| | von Mises safety factor in torsion (V7) | 2.1312 | 2.1302 | 0.0457 % | 5 % | |
| | Tresca safety factor in torsion (V7) | 1.8457 | 1.8448 | 0.0457 % | 5 % | |
| | Mohr-Coulomb safety factor, cast iron, torsion (V7) | 2.992 | 2.8261 | 5.54 % | 7 % | |
| **FB1** ✔ | Free body cut: section forces and moments (cantilever of V1, torsion of V7) | *closed-form solution* (relation stated, no work cited): Static equilibrium (applied load = reaction); strength of materials – V_z = −F, M_y = F·(L − x), T = applied torque | | | | 22 137 nodes, h = 2.5 mm |
| | cut at x = 30 mm: shear force Vz | -100 N | -100 N | < 0.001 % | 5 % | |
| | cut at x = 30 mm: bending moment My about the centroid | 17000 N·mm | 17004 N·mm | 0.022 % | 2 % | |
| | cut at x = 30 mm: axial force N (= 0) | 0 N | 2.2692e-5 N | 2.2692e-5 N | 0.5 N | |
| | cut at x = 60 mm: shear force Vz | -100 N | -100 N | < 0.001 % | 5 % | |
| | cut at x = 60 mm: bending moment My about the centroid | 14000 N·mm | 13999 N·mm | 0.00963 % | 2 % | |
| | cut at x = 60 mm: axial force N (= 0) | 0 N | 8.7294e-6 N | 8.7294e-6 N | 0.5 N | |
| | cut at x = 100 mm: shear force Vz | -100 N | -100 N | < 0.001 % | 5 % | |
| | cut at x = 100 mm: bending moment My about the centroid | 10000 N·mm | 10000 N·mm | 0.00411 % | 2 % | |
| | cut at x = 100 mm: axial force N (= 0) | 0 N | 2.5888e-5 N | 2.5888e-5 N | 0.5 N | |
| | cut at x = 140 mm: shear force Vz | -100 N | -100 N | < 0.001 % | 5 % | |
| | cut at x = 140 mm: bending moment My about the centroid | 6000 N·mm | 6002.2 N·mm | 0.0368 % | 2 % | |
| | cut at x = 140 mm: axial force N (= 0) | 0 N | 3.2547e-5 N | 3.2547e-5 N | 0.5 N | |
| | cut at x = 170 mm: shear force Vz | -100 N | -100 N | < 0.001 % | 5 % | |
| | cut at x = 170 mm: bending moment My about the centroid | 3000 N·mm | 3003.4 N·mm | 0.113 % | 2 % | |
| | cut at x = 170 mm: axial force N (= 0) | 0 N | 8.4968e-8 N | 8.4968e-8 N | 0.5 N | |
| | torsion (V7): torque Mx at the cut | 1e+05 N·mm | 1e+05 N·mm | < 0.001 % | 2 % | |
| **IM1** ✔ | Face split: circular imprint r = 10 mm under 1 MPa | *closed-form solution* (relation stated, no work cited): Geometry (area π·r²), statics (reaction = p·A), cantilever strip with shear for the deflection estimate (10 % tolerance) | | | | 11 650 nodes, h = 5 mm |
| | imprint area of the circle (π·r²) | 314.16 mm² | 310.86 mm² | 1.05 % | 3 % | |
| | tip deflection at x = 100 mm (cantilever, wide strip + shear) | -0.01725 mm | -0.01815 mm | 5.22 % | 10 % | |
| **IM2** ✔ | Polygon imprint (L shape): area, reaction and centroid of the 1 MPa pressure | *closed-form solution* (relation stated, no work cited): Polygon area and centroid by the shoelace formulas; statics (reaction = p·A, moment about the pressure centroid) | | | | 14 712 nodes, h = 5 mm |
| | imprint area = shoelace area (straight edges) | 450 mm² | 450 mm² | < 0.001 % | 0.0001 % | |
| | reaction moment Mx about c0 = −((x_c − c0) × F)x | -1125 N·mm | -1125 N·mm | < 0.001 % | 0.5 % | |
| | reaction moment My about c0 = −((x_c − c0) × F)y | -750 N·mm | -750 N·mm | < 0.001 % | 0.5 % | |
| **PS1** ✔ | Parametric study: cantilever of V1, load × Young's modulus and a linear scan | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | – |
| | max. displacement F×1, E = 210000 (Timoshenko) | 0.7634 mm | 0.75849 mm | 0.643 % | 3 % | |
| | max. displacement F×1, E = 70000 (Timoshenko) | 2.2902 mm | 2.2755 mm | 0.644 % | 3 % | |
| | max. displacement F×2, E = 210000 (Timoshenko) | 1.5268 mm | 1.517 mm | 0.643 % | 3 % | |
| | max. displacement F×2, E = 70000 (Timoshenko) | 4.5804 mm | 4.5509 mm | 0.644 % | 3 % | |
| **SM1** ✔ | Submodel: plate with a hole in tension (Peterson / Heywood K_t) | *handbook / standard*: R. E. Peterson: Stress Concentration Factors (Wiley) – plate of finite width with a central circular hole, polynomial of Heywood for K_t on the net section | | | | – |
| | coarse global model: max von Mises (h = 14 mm) vs Peterson | 31.331 MPa | 30.175 MPa | 3.69 % | 8 % | |
| | submodel max σ1 vs Peterson / Heywood K_t,net = 2.506 (h = 0.6 mm) | 31.331 MPa | 32.817 MPa | 4.74 % | 6 % | |

## Heat conduction and thermal stress

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **T1** ✔ | Steel + aluminium bar: convection, flux and power (series resistances) | *closed-form solution*: Steady 1-D conduction through series thermal resistances, Fourier's law (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 260 nodes, h = 4 mm |
| | (a) interface temperature at x = 50 mm | 75.714 °C | 75.714 °C | < 0.001 % | 0.01 % | |
| | (a) end-face temperature at x = 100 mm | 68.571 °C | 68.571 °C | < 0.001 % | 0.01 % | |
| | (a) heat at the prescribed temperature | 2.4286 W | 2.4286 W | < 0.001 % | 0.1 % | |
| | (a) heat by convection | 2.4286 W | 2.4286 W | < 0.001 % | 0.1 % | |
| | (a) heat flux q_x in the steel | 24286 W/m² | 24286 W/m² | < 0.001 % | 0.1 % | |
| | (b) end-face temperature at x = 100 mm | 32.941 °C | 32.941 °C | < 0.001 % | 0.01 % | |
| | (c) end-face temperature at x = 100 mm (power 1 W) | 32.941 °C | 32.941 °C | < 0.001 % | 0.01 % | |
| | (c) heat at the prescribed temperature | 1 W | 1 W | < 0.001 % | 0.1 % | |
| **T2** ✔ | Tube r_i 20 / r_o 40 mm: radial conduction (logarithmic profile) | *closed-form solution*: Steady radial conduction in a hollow cylinder, T(r) logarithmic (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 5 895 nodes, h = 2.5 mm |
| | T(r = 30) | 53.203 °C | 53.201 °C | 0.00406 % | 0.1 % | |
| | radial heat flux q_r(30) = k·ΔT/(r·ln(r_o/r_i)) | 1.9236e+05 W/m² | 1.9223e+05 W/m² | 0.0688 % | 1 % | |
| | heat through the inner surface | 90.647 W | 90.647 W | < 0.001 % | 0.2 % | |
| **T3** ✔ | Bar with an internal heat source of 10 W (parabolic profile) | *closed-form solution*: 1-D conduction with uniform volumetric heat generation, T_max = T₀ + q‴·L²/(8·k) (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 026 nodes, h = 4 mm |
| | temperature at mid-length x = 50 mm | 45 °C | 45 °C | < 0.001 % | 0.01 % | |
| | temperature at x = 25 mm | 38.75 °C | 38.75 °C | < 0.001 % | 0.01 % | |
| | heat through the end x = 0 | 5 W | 5 W | < 0.001 % | 0.1 % | |
| | heat through the end x = 100 mm | 5 W | 5 W | < 0.001 % | 0.1 % | |
| **T4** ✔ | Thermal stress: free and restrained expansion of a bar, ΔT = 100 K | *closed-form solution* (relation stated, no work cited): Thermal expansion: ΔL = α·ΔT·L free, σ = −E·α·ΔT fully restrained | | | | 2 026 nodes, h = 4 mm |
| | (a) free expansion ux(L) = α·ΔT·L | 0.12 mm | 0.12 mm | < 0.001 % | 0.001 % | |
| | (a) stress-free: 1 + max σ_vM/(E·α·ΔT) | 1 | 1 | < 0.001 % | 0.001 % | |
| | (b) restrained: σx = −E·α·ΔT | -252 MPa | -252 MPa | 0.00119 % | 0.01 % | |
| | (b) reaction = E·α·ΔT·A | -25200 N | -25200 N | < 0.001 % | 0.01 % | |
| **T5** ✔ | Thermal stress in a tube from the temperature field of T2 (plane strain) | *published solution*: Thermal stresses in a long hollow cylinder, plane strain (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity, thermal stress) | | | | 5 895 nodes, h = 2.5 mm |
| | σθ(r = 20) | -176.25 MPa | -175.28 MPa | 0.552 % | 3 % | |
| | σz(r = 20) | -254.48 MPa | -254.21 MPa | 0.103 % | 3 % | |
| | σθ(r = 30) | 14.649 MPa | 14.468 MPa | 1.24 % | 3 % | |
| | σr(r = 30) | -22.432 MPa | -22.586 MPa | 0.686 % | 5 % | |
| | u_r(r = 30) | 0.018693 mm | 0.018695 mm | 0.011 % | 1 % | |
| | σθ(r = 40) | 111.75 MPa | 111.92 MPa | 0.157 % | 3 % | |
| **T6** ✔ | Transient conduction in a bar after a temperature step (Fourier series) | *closed-form solution*: Fourier-series solution of the 1-D transient heat equation (e.g. H. S. Carslaw, J. C. Jaeger: Conduction of Heat in Solids) | | | | 2 026 nodes, h = 4 mm |
| | ΔT(x = 50, t = 300 s) | 67.692 K | 67.465 K | 0.336 % | 1 % | |
| | ΔT(x = 100, t = 300 s) | 54.319 K | 54.003 K | 0.581 % | 1 % | |
| | ΔT(x = 50, t = 600 s) | 88.409 K | 88.247 K | 0.183 % | 1 % | |
| | ΔT(x = 100, t = 600 s) | 83.607 K | 83.378 K | 0.274 % | 1 % | |
| **T7** ✔ | Radiation from the end face of a bar to the surroundings (nonlinear) | *closed-form solution*: Energy balance at the end face: conduction = ε·σ_SB·(T⁴ − T_amb⁴), solved for the face temperature (Stefan-Boltzmann law, F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 026 nodes, h = 4 mm |
| | end-face temperature (radiation) | 242.54 °C | 242.54 °C | < 0.001 % | 0.01 % | |
| | heat radiated | 0.28731 W | 0.28731 W | < 0.001 % | 0.2 % | |
| **TH8** ✔ | Two plates in series with a contact resistance (ΔT = q / h_c) | *handbook / standard*: Thermal contact conductance in series with the plate resistances (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 294 nodes, h = 4 mm |
| | heat through the interface Q = h_c·∫ΔT dA (own integration) | 2.6667 W | 2.6667 W | < 0.001 % | 0.1 % | |
| | interface temperature at x = 50 mm, side 1 (slave) | 73.333 °C | 73.333 °C | < 0.001 % | 0.01 % | |
| | interface temperature at x = 50 mm, side 2 (master) | 46.667 °C | 46.667 °C | < 0.001 % | 0.01 % | |
| | interface ΔT (q / h_c) | 26.667 K | 26.667 K | < 0.001 % | 0.1 % | |
| | heat entering at x = 0 | 2.6667 W | 2.6667 W | < 0.001 % | 0.1 % | |
| | heat removed at x = 100 mm | 2.6667 W | 2.6667 W | < 0.001 % | 0.1 % | |
| **TH8B** ✔ | Bonded interface with contact conductance (separately meshed plates) | *handbook / standard*: Thermal contact conductance in series with the plate resistances (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 294 nodes, h = 4 mm |
| | T at x = 50 in part 1 | 73.333 °C | 73.333 °C | < 0.001 % | 0.01 % | |
| | T at x = 50 in part 2 | 46.667 °C | 46.667 °C | < 0.001 % | 0.01 % | |
| | heat in at x = 0 | 2.6667 W | 2.6667 W | < 0.001 % | 0.1 % | |
| | heat through the interface (own integration) | 2.6667 W | 2.6667 W | < 0.001 % | 0.1 % | |
| | ΔT of the interface = q / h_c | 26.667 K | 26.667 K | < 0.001 % | 0.1 % | |
| | mechanics: u(L) = F·L/(E·A) (*TIE kept) | 0.0047619 mm | 0.0047619 mm | < 0.001 % | 0.01 % | |
| **TH9** ✔ | Radiation between two parallel surfaces (cavity radiation) | *closed-form solution*: Net radiation between large parallel plates: q = σ_SB·(T₁⁴ − T₂⁴)/(1/ε₁ + 1/ε₂ − 1) (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 527 nodes, h = 6 mm |
| | radiated heat flow Q | 0.64617 W | 0.64615 W | 0.00176 % | 0.1 % | |
| | heat removed by plate 2 | 0.64617 W | 0.64616 W | 0.00127 % | 0.1 % | |
| **TH9C** ✔ | Cavity radiation with two different emissivities | *closed-form solution*: Net radiation between large parallel plates: q = σ_SB·(T₁⁴ − T₂⁴)/(1/ε₁ + 1/ε₂ − 1) (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 2 527 nodes, h = 6 mm |
| | radiated heat Q (ε 0.8 / 0.5) | 0.43078 W | 0.43077 W | 0.00125 % | 0.1 % | |
| | cavity face 'cav_hot' gives off Q | 0.43078 W | 0.43077 W | 0.00126 % | 0.1 % | |
| | cavity face 'cav_cold' receives Q | 0.43078 W | 0.43077 W | 0.00112 % | 0.1 % | |
| **KT1** ✔ | Heat conduction across a contact (separately meshed blocks, *TIE in the thermal problem) | *closed-form solution*: Fourier's law across a perfectly conducting interface: Q = k·A·ΔT/L (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 5 396 nodes, h = 2.5 mm |
| | heat flow across the contact Q = k·A·ΔT/L | 106.6 W | 106.59 W | 0.00759 % | 0.1 % | |
| | interface temperature (50 °C) | 50 °C | 49.988 °C | 0.023 % | 0.1 % | |
| **KT1c** ✔ | Heat conduction across an open conforming gap (equation on DOF 11) | *closed-form solution*: Fourier's law across a perfectly conducting interface: Q = k·A·ΔT/L (F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer) | | | | 4 943 nodes, h = 2.5 mm |
| | heat flow across the contact Q = k·A·ΔT/L | 106.6 W | 106.6 W | < 0.001 % | 0.1 % | |
| | interface temperature (50 °C) | 50 °C | 50 °C | < 0.001 % | 0.1 % | |
| **HT1** ✔ | Heat-transfer calculator: Churchill-Chu and Dittus-Boelter / Gnielinski | *handbook / standard*: Churchill-Chu natural convection on a vertical plate, Dittus-Boelter forced convection in a pipe, air / water properties from the tables of F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer | | | | 22 137 nodes, h = 2.5 mm |
| | natural convection, vertical plate: h | 4.7 W/(m²·K) | 4.7462 W/(m²·K) | 0.983 % | 6 % | |
| | forced convection, water in a pipe: h | 5100 W/(m²·K) | 5135.5 W/(m²·K) | 0.695 % | 8 % | |

## Natural frequencies, buckling and dynamics

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **M1** ✔ | Natural frequencies of a clamped cantilever (Euler-Bernoulli, effective mass) | *published solution*: Euler-Bernoulli cantilever: f = (βL)²/(2π·L²)·√(E·I/(ρ·A)), βL = 1.875104; effective mass of the first mode (R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations) | | | | 22 137 nodes, h = 2.5 mm |
| | f₁ bending about the weak axis (E–B) | 208.88 Hz | 209.45 Hz | 0.274 % | 1 % | |
| | f₂ bending about the strong axis (E–B, L/h = 10) | 417.76 Hz | 415.8 Hz | 0.469 % | 2 % | |
| | effective mass of mode 1 / m (E–B, numerical integration of the mode shape) | 0.61308 | 0.6118 | 0.208 % | 1 % | |
| **M2** ✔ | Natural frequencies of a free beam (6 rigid-body modes) | *published solution*: Euler-Bernoulli free-free beam: βL = 4.730041 (R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations) | | | | 22 137 nodes, h = 2.5 mm |
| | 6 rigid-body modes: 1 + max(f₁…f₆) / f₇ (≈ 0) | 1 | 1 | < 0.001 % | 1 % | |
| | f₇ free beam, bending about the weak axis (E–B; Timoshenko −0.9 %) | 1329.2 Hz | 1317.7 Hz | 0.86 % | 1.5 % | |
| **B1** ✔ | Linear buckling of a cantilever column loaded at its end (Euler) | *closed-form solution*: Euler critical load of a cantilever column: P_cr = π²·E·I/(4·L²) | | | | 22 137 nodes, h = 2.5 mm |
| | λ₁·F = π²·E·I/(4·L²) (weak axis) | 21590 N | 21652 N | 0.286 % | 2 % | |
| | λ₂/λ₁ = I_strong/I_weak = 4 | 4 | 3.9683 | 0.791 % | 2 % | |
| | static state: end compression = reaction | -1000 N | -1000 N | < 0.001 % | 0.01 % | |
| **Y1** ✔ | Single-DOF oscillator under a harmonic force (\|H\|, phase) | *closed-form solution*: \|H(r)\| = 1/√((1 − r²)² + (2ζr)²), phase −90° at resonance (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | frequency of the vertical mode √(k/m)/2π | 159.15 Hz | 159.14 Hz | 0.0103 % | 0.1 % | |
| | \|u\|/u_st, resonance r = 1 | 25 | 25.005 | 0.0208 % | 0.1 % | |
| | \|u\|/u_st, r = 1 − ζ | 17.947 | 17.95 | 0.0208 % | 0.1 % | |
| | \|u\|/u_st, r = 1 + ζ | 17.416 | 17.42 | 0.0208 % | 0.1 % | |
| | phase at resonance (90°) | 90 ° | 90 ° | < 0.001 % | 0.2 % | |
| | \|u\|/u_st at 20 Hz (quasi-static) | 1.016 | 1.0162 | 0.0208 % | 0.1 % | |
| **Y1s** ✔ | Single-DOF oscillator with structural damping η | *closed-form solution*: Hysteretic (structural) damping: resonance amplification 1/η (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | peak \|u\|/u_st = 1/η (η = 0.04) | 25 | 25.005 | 0.0208 % | 0.1 % | |
| **Y2** ✔ | Cantilever under a harmonic force: Euler-Bernoulli frequency and resonance peak | *published solution*: Euler-Bernoulli cantilever, first mode (R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations); resonance peak of the first mode relative to the static deflection | | | | 6 837 nodes, h = 2.5 mm |
| | f₁ (Euler–Bernoulli; Timoshenko −0.3 %) | 835.52 Hz | 833.31 Hz | 0.264 % | 0.5 % | |
| | peak \|u\|/u_st (E–B 0.9707/(2ζ)) | 24.267 | 24.236 | 0.131 % | 1 % | |
| **Y3** ✔ | Step force on a single-DOF oscillator (dynamic load factor 2) | *closed-form solution*: Undamped step response: dynamic load factor 2 (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | step DLF, 1 DOF, ζ = 0 | 2 | 2.0004 | 0.0208 % | 0.2 % | |
| **Y3d** ✔ | Step force on a damped single-DOF oscillator | *closed-form solution*: Damped step response: DLF = 1 + exp(−π·ζ/√(1 − ζ²)) (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | step DLF, 1 DOF, ζ = 2 % (1 + e^(−πζ/√(1−ζ²))) | 1.9391 | 1.9395 | 0.0208 % | 0.5 % | |
| **Y6** ✔ | Free decay: logarithmic decrement | *closed-form solution*: Logarithmic decrement δ = ln(u_i/u_i+n)/n, ζ = δ/√(4π² + δ²) (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | ζ from the logarithmic decrement (11 periods) | 0.02 | 0.02 | < 0.001 % | 1 % | |
| **Y10** ✔ | Axis start-up: residual amplitude after a rectangular pulse | *closed-form solution*: Response of a single-DOF oscillator to a rectangular base-acceleration pulse: residual amplitude (2a/ω²)·\|sin(ω·τ/2)\| (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | residual amplitude (2a/ω²)·\|sin(ωτ/2)\| | 0.015882 mm | 0.015884 mm | 0.0137 % | 0.5 % | |
| **Y15** ✔ | Cantilever with a point mass at the tip (Rayleigh) | *published solution*: Rayleigh's method: f₁ = (1/2π)·√(3·E·I/((M + 33/140·m)·L³)) (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 6 837 nodes, h = 2.5 mm |
| | f₁ with the tip mass (Rayleigh) | 128.65 Hz | 128.65 Hz | 0.00409 % | 1 % | |
| **Y16** ✔ | Base excitation through a rigid remote coupling: first frequency | *published solution*: Euler-Bernoulli cantilever, first mode (R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations) | | | | 6 837 nodes, h = 2.5 mm |
| | f₁ with the rigid base = clamped (E–B) | 835.52 Hz | 833.31 Hz | 0.264 % | 0.5 % | |
| **Y4** ✔ | Half-sine shock: shock response spectrum of a single-DOF oscillator | *closed-form solution*: Shock response spectrum (maximax absolute acceleration) of a damped single-degree-of-freedom oscillator under a half-sine base-acceleration pulse, Duhamel integral: S_a/A = 1.6516 at f_n·τ = 0.82, ζ = 5 % (Q = 10); printed example with the same definition: T. Irvine, An Introduction to the Shock Response Spectrum, Rev. S (2012), Fig. 5 (11 ms, 50 G pulse, Q = 10: 55 G at 30 Hz, 82 G at 80 Hz, 70 G at 140 Hz). The case also runs the SRSS combination with the missing mass and the transient response; those two checks are internal consistency checks against the same spectrum and are not listed | | | | 2 148 nodes, h = 5 mm |
| | S_a(f_n)/A of a half-sine pulse, f_n·τ = 0.82, ζ = 5 % | 1.6516 | 1.6516 | < 0.001 % | 0.3 % | |
| **Y5** ✔ | Random vibration of a single-DOF oscillator (Miles' equation) | *published solution*: Miles' equation: g_rms = √(π/2·f_n·Q·ASD(f_n)) for white-noise base acceleration (S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)) | | | | 2 148 nodes, h = 5 mm |
| | g_rms of the absolute acceleration of the mass (Miles·√(1+4ζ²)) | 10.049 g | 10.009 g | 0.402 % | 0.5 % | |
| **Y9** ✔ | Prestressed modal analysis: tension 0.5·N_cr | *closed-form solution* (relation stated, no work cited): Natural frequency of a simply supported beam under axial force N (tension positive): f(N) = f(0)·√(1 + N/N_cr), N_cr = π²EI/L² | | | | 13 491 nodes, h = 2.5 mm |
| | f₁(N)/f₁(0) = √(1 + N/N_cr), N = +0.5·N_cr | 1.2247 | 1.2262 | 0.116 % | 0.5 % | |
| **Y9c** ✔ | Prestressed modal analysis: compression 0.5·N_cr | *closed-form solution* (relation stated, no work cited): Natural frequency of a simply supported beam under axial force N (tension positive): f(N) = f(0)·√(1 + N/N_cr), N_cr = π²EI/L² | | | | 13 491 nodes, h = 2.5 mm |
| | f₁(N)/f₁(0) = √(1 + N/N_cr), N = -0.5·N_cr | 0.70711 | 0.70464 | 0.349 % | 0.5 % | |
| **Y7** ✔ | Rotating blade: Campbell diagram, Southwell coefficient | *published solution*: Southwell's relation for a rotating blade ω² = ω₀² + K·Ω², K = ∫ρA·x·[∫(w')²dx]dx / ∫ρA·w²dx → 1.1933 for the first flapwise mode of a uniform cantilever with zero hub radius (L. N. Virgin: Vibration of Axially-Loaded Structures, lecture notes Part C, Duke University, 2013, slides 58-61: the definition and the tabulated frequencies ω₁ = 3.5160, 3.6816, 4.7973 at Ω = 0, 1, 3 in units of √(EI/(ρA·L⁴))); the reference of the case, K = 1.1812 at Ω/ω₀ = 0.957, is computed by an independent 1-D rotating-beam script (Hermite beam elements with the centrifugal tension as geometric stiffness, 160 elements) that reproduces the table of those lecture notes at η = 0, 1, 3 | | | | 13 319 nodes, h = 2.5 mm |
| | Southwell coefficient of the 1st flapwise bending mode (independent 1-D rotating-beam model, Ω/ω₀ = 0.957) | 1.1812 | 1.1845 | 0.274 % | 1 % | |
| | f₁ at 0 rpm (Euler-Bernoulli, 104.4 Hz) | 104.44 Hz | 104.98 Hz | 0.518 % | 1 % | |

## Contacts and friction

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **K1** ✔ | Contact patch test: separately meshed blocks | *closed-form solution* (relation stated, no work cited): Uniform pressure through a flat interface: p = F/A, u = p·h/E (+ p/K of the contact stiffness) | | | | 5 396 nodes, h = 2.5 mm |
| | contact pressure – mean (10 MPa) | 10 MPa | 10.001 MPa | 0.00904 % | 0.1 % | |
| | contact pressure – largest deviation (as 1 + \|p − 10\|/10) | 1 | 1.002 | 0.204 % | 0.5 % | |
| | top displacement u_z = 2·p·h/E + p/K | 9.6238e-4 mm | 9.6238e-4 mm | < 0.001 % | 0.1 % | |
| **K1c** ✔ | Contact patch test: conforming mesh, open gap | *closed-form solution* (relation stated, no work cited): Uniform pressure through a flat interface: p = F/A, u = p·h/E (+ p/K of the contact stiffness) | | | | 4 943 nodes, h = 2.5 mm |
| | contact pressure – mean (10 MPa) | 10 MPa | 9.9998 MPa | 0.0025 % | 0.1 % | |
| | contact pressure – largest deviation (as 1 + \|p − 10\|/10) | 1 | 1 | 0.0025 % | 0.5 % | |
| | top displacement u_z = 2·p·h/E + p/K | 9.6238e-4 mm | 9.6238e-4 mm | < 0.001 % | 0.1 % | |
| **K4** ✔ | Press fit (Lamé) | *closed-form solution*: Lamé (thick-walled cylinders); Shigley's Mechanical Engineering Design, ch. 3-16 (press and shrink fits) | | | | 13 711 nodes, h = 1.25 mm |
| | joint pressure p (weighted mean of the nodal CPRESS) | 78.75 MPa | 80.253 MPa | 1.91 % | 2.5 % | |
| | hub hoop stress σ_t at the inner surface | 131.25 MPa | 131.29 MPa | 0.0331 % | 0.5 % | |
| | hub radial displacement u_r, inner surface | 7.375 µm | 7.3666 µm | 0.114 % | 1 % | |
| | shaft radial displacement u_r | 2.625 µm | 2.6225 µm | 0.0952 % | 2 % | |
| | hub radial displacement u_r, outer surface (Lamé 2p·r_i²·r_o/(E(r_o² − r_i²))) | 5 µm | 4.9946 µm | 0.108 % | 1 % | |
| **K7** ✔ | Cantilever over a rigid cylinder with a gap | *closed-form solution*: Euler-Bernoulli cantilever with a gap-limited support: compatibility of the end deflection, R = (F·a²·(3L − a)/(6EI) − gap)/(a³/(3EI)) | | | | 14 680 nodes, h = 2.5 mm |
| | contact force (E–B, gap 0.5 mm) | 222.22 N | 221.7 N | 0.236 % | 1.5 % | |
| **K7n** ✔ | Cantilever over a rigid cylinder with a gap, NLGEOM | *closed-form solution*: Euler-Bernoulli cantilever with a gap-limited support: compatibility of the end deflection, R = (F·a²·(3L − a)/(6EI) − gap)/(a³/(3EI)) | | | | 14 680 nodes, h = 2.5 mm |
| | contact force (E–B, gap 0.5 mm) | 222.22 N | 221.74 N | 0.215 % | 1.5 % | |
| **R1** ✔ | Friction plateau under a controlled displacement | *closed-form solution* (relation stated, no work cited): Coulomb friction: T = μ·N | | | | 5 396 nodes, h = 2.5 mm |
| | friction plateau T = μ·N (300 N) | 300 N | 299.99 N | 0.0033 % | 0.1 % | |
| | friction force of the pair Σ τ·A | 300 N | 300 N | < 0.001 % | 0.5 % | |
| | normal force of the pair N | 1000 N | 1000 N | < 0.001 % | 0.1 % | |
| **G1** ✔ | Compression-only support (GAPUNI) | *closed-form solution* (relation stated, no work cited): Statics of a compression-only support: reaction = p·A, u = p·h/E (+ p/K) | | | | 2 703 nodes, h = 2.5 mm |
| | compression-only support reaction = p·A | 4000 N | 3999.7 N | 0.00648 % | 0.1 % | |
| | top displacement = p·h/E + p/K | 4.8619e-4 mm | 4.8619e-4 mm | < 0.001 % | 0.2 % | |
| **KK2** ✔ | Snap-fit hook: NLGEOM, friction, prescribed motion | *handbook / standard*: Bayer MaterialScience: Snap-Fit Joints for Plastics – A Design Guide, pp. 8–16 (hook and force equations) | | | | 13 486 nodes, h = 0.5 mm |
| | W/P = tan(α + arctan μ) | 1.0611 | 1.0599 | 0.121 % | 1 % | |
| | assembly force W (Bayer: P·(μ + tan α)/(1 − μ·tan α), P from the FE result) | 1.4403 N | 1.4386 N | 0.121 % | 1 % | |

## Bolted and welded joints

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **S2** ✔ | Lap joint M10 8.8: shear, preload, VDI 2230 and EN 1993-1-8 | *handbook / standard*: VDI 2230 Part 1 (permissible assembly preload F_M,zul); EN 1993-1-8 (Eurocode 3, design of joints): bolt shear and bearing resistance, F_v,Rd = α_v·f_ub·A_s/γ_M2, F_b,Rd = k₁·α_b·f_u·d·t/γ_M2; statics (shear transfer, weight) | | | | 11 540 nodes, h = 5 mm |
| | weight (nodal forces, beam-bolt model): reaction Fz = m·g | 6.0143 N | 6.0143 N | < 0.001 % | 0.01 % | |
| | shear transferred through the joint plane (gap + bolt) | 10000 N | 10000 N | < 0.001 % | 0.01 % | |
| | VDI 2230 F_M,zul (M10 8.8, μ_G = 0.12) | 29603 N | 29603 N | < 0.001 % | 1e-07 % | |
| | EN 1993-1-8 F_v,Rd (threaded part, α_v = 0.6) | 22268 N | 22268 N | < 0.001 % | 0.1 % | |
| | EN 1993-1-8 F_b,Rd (e1 = e2 = 20 mm from the geometry) | 43636 N | 43636 N | < 0.001 % | 0.1 % | |
| **S4** ✔ | Virtual bolt from a CAD assembly (parts not loaded): same as S2 | *handbook / standard*: VDI 2230 Part 1 (permissible assembly preload F_M,zul); EN 1993-1-8 (Eurocode 3, design of joints): bolt shear and bearing resistance, F_v,Rd = α_v·f_ub·A_s/γ_M2, F_b,Rd = k₁·α_b·f_u·d·t/γ_M2; statics (shear transfer, weight) | | | | 11 627 nodes, h = 5 mm |
| | weight (nodal forces, beam-bolt model): reaction Fz = m·g | 6.0143 N | 6.0143 N | < 0.001 % | 0.01 % | |
| | shear transferred through the joint plane (gap + bolt) | 10000 N | 10000 N | < 0.001 % | 0.01 % | |
| | VDI 2230 F_M,zul (M10 8.8, μ_G = 0.12) | 29603 N | 29603 N | < 0.001 % | 1e-07 % | |
| | EN 1993-1-8 F_v,Rd (threaded part, α_v = 0.6) | 22268 N | 22268 N | < 0.001 % | 0.1 % | |
| | EN 1993-1-8 F_b,Rd (e1 = e2 = 20 mm from the geometry) | 43636 N | 43636 N | < 0.001 % | 0.1 % | |
| **S3** ✔ | Two bolts M12: load distribution in a bolt group, bearing with pitch | *handbook / standard*: EN 1993-1-8 (Eurocode 3, design of joints): bolt shear and bearing resistance, F_v,Rd = α_v·f_ub·A_s/γ_M2, F_b,Rd = k₁·α_b·f_u·d·t/γ_M2; equal share F/n of a bolt group | | | | 17 344 nodes, h = 5 mm |
| | joint plane + bolts transfer in total (FE) | 20000 N | 20000 N | < 0.001 % | 0.01 % | |
| | bolt group: bolt 1 = F/n | 10000 N | 10000 N | < 0.001 % | 0.01 % | |
| | bolt group: bolt 2 = F/n | 10000 N | 10000 N | < 0.001 % | 0.01 % | |
| | F_b,Rd plate A, bolt 1 (e1 = 60, p1 = 40, e2 = 20 mm) | 62411 N | 62411 N | < 0.001 % | 0.1 % | |
| **W1** ✔ | T-joint with fillet welds a = 5: tension and longitudinal shear | *handbook / standard*: EN 1993-1-8 (Eurocode 3, design of joints): fillet weld throat a, longitudinal shear stress τ = F/(2·a·L); statics of the weld forces | | | | 17 719 nodes, h = 5 mm |
| | tension: sum of the segment forces of both welds | 20000 N | 20000 N | < 0.001 % | 0.01 % | |
| | longitudinal shear: sum of the segment forces | 10000 N | 10000 N | < 0.001 % | 0.01 % | |
| | longitudinal shear: mean τ∥ = F/(2·a·L) | 10 MPa | 10 MPa | < 0.001 % | 0.1 % | |
| **W2** ✔ | Weld toe hot-spot stress (IIW 0.4t / 1.0t) on a bent plate | *handbook / standard*: IIW Recommendations for Fatigue Design of Welded Joints and Components: hot-spot extrapolation 0.4t / 1.0t; beam theory σ = 6·F·(L − x)/(b·t²) at the weld toe | | | | 32 418 nodes, h = 3 mm |
| | hot-spot stress at the right weld toe (x = 110.66 mm) = beam σ | 134.01 MPa | 133.83 MPa | 0.137 % | 1 % | |
| | hot-spot stress at the left weld toe (x = 89.34 mm) = beam σ | 165.99 MPa | 165.67 MPa | 0.19 % | 1 % | |

## Fatigue

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **U1** ✔ | Fatigue: cantilever of V1, ±100 N, 1000 cycles per block, 100 blocks | *closed-form solution* (relation stated, no work cited): Bending stress σ = M/W; Palmgren-Miner linear damage D = Σ n/N with a Basquin S-N curve N = N_D·(S_D/σ_a)^k | | | | 22 137 nodes, h = 2.5 mm |
| | stress amplitude σ at mid-length (top fibre) | 30 MPa | 29.879 MPa | 0.404 % | 5 % | |
| | damage per block = Miner from the nodal amplitude | 0.0074418 | 0.0074418 | < 0.001 % | 0.0001 % | |
| | safety factor = (1/(R·D))^(1/k) | 1.0609 | 1.0609 | < 0.001 % | 0.01 % | |
| | damage against theory (σa = 30 MPa) | 0.0075938 | 0.0074418 | 2 % | 25 % | |
| **U2** ✔ | Fatigue: two unit load states Fz + Fy in phase (superposition) | *closed-form solution* (relation stated, no work cited): Bending stress σ = M/W summed over the two load states; Palmgren-Miner linear damage with a Basquin S-N curve | | | | 22 137 nodes, h = 2.5 mm |
| | amplitude at the corner x = 100 mm (30 + 15 MPa) | 45 MPa | 44.989 MPa | 0.0236 % | 5 % | |
| | damage per block = Miner from the nodal amplitude | 0.057597 | 0.057597 | < 0.001 % | 0.0001 % | |
| **U3** ✔ | Fatigue, critical plane: tension / torsion / both 90° out of phase | *published solution*: Findley critical-plane criterion with hand evaluation of the plane search (tension: σ = F/A, torsion: τ = 16·T/(π·d³), 90° out of phase: maximum over the plane angle); Palmgren-Miner damage | | | | 28 909 nodes, h = 2.5 mm |
| | (a) tension: σa at the surface = F/A | 80 MPa | 79.999 MPa | < 0.001 % | 3 % | |
| | (b) torsion: critical-plane normal-stress amplitude σa = τ (theory) | 63.662 MPa | 63.867 MPa | 0.323 % | 5 % | |
| | (b) torsion: Findley σ_eq = √3·τ (theory) | 110.27 MPa | 110.6 MPa | 0.3 % | 5 % | |
| | (c) 90° out of phase: Findley = hand calculation | 130.37 MPa | 130.7 MPa | 0.246 % | 5 % | |
| | (c) 90° out of phase: critical-plane σn = hand calculation | 81.832 MPa | 81.916 MPa | 0.102 % | 5 % | |
| | (c) damage = Miner from the amplitude (5 cycles per block) | 0.059582 | 0.059582 | < 0.001 % | 0.0001 % | |
| **U4** ✔ | Strain-life fatigue: notched bar (Neuber, Coffin-Manson) | *published solution*: Neuber's rule σ·ε = σ_el²/E, Glinka's energy-density rule, Ramberg-Osgood cyclic curve, Coffin-Manson-Basquin strain-life relation with Morrow and Smith-Watson-Topper mean-stress corrections; K_t of a plate with a hole (Peterson / Heywood) | | | | 12 486 nodes, h = 4 mm |
| | Neuber product σ·ε = σ_el²/E | 0.43531 MPa | 0.43531 MPa | < 0.001 % | 0.001 % | |
| | Glinka energy density W = W_el | 0.21765 MPa | 0.21765 MPa | < 0.001 % | 0.001 % | |
| | Neuber: ε satisfies Ramberg-Osgood | 0.0020525 | 0.0020525 | < 0.001 % | 0.0001 % | |
| | Glinka strain ε (independent bisection) | 0.0016564 | 0.0016564 | < 0.001 % | 0.001 % | |
| | Coffin-Manson life N_f (independent bisection) | 27223 cycles | 27223 cycles | < 0.001 % | 0.001 % | |
| | Morrow life N_f, σ_m = 50 MPa (independent) | 23931 cycles | 23931 cycles | < 0.001 % | 0.001 % | |
| | SWT life N_f, σ_max = 300 MPa (independent) | 12246 cycles | 12246 cycles | < 0.001 % | 0.001 % | |
| | FE elastic peak σ_el at the hole vs Peterson | 302.35 MPa | 312.07 MPa | 3.22 % | 5 % | |
| | FE nodal damage D = Coffin-Manson(ε_Neuber) | 0.038783 | 0.038783 | < 0.001 % | 0.01 % | |
| | FE safety factor (D·blocks = 1, independent bisection) | 0.76697 | 0.76697 | < 0.001 % | 0.1 % | |

## 3D-printed parts

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **P1** ✔ | Printed bars upright / flat: orthotropy and Tsai-Hill | *closed-form solution*: Hooke's law of a transversely isotropic bar (u = σ·L/E_z, u = σ·L/E_xy, Poisson contraction ν₁₃); Tsai-Hill failure criterion for stress along a material axis (safety = strength/σ) | | | | 8 557 nodes, h = 3 mm |
| | A (tension ⟂ layers): u = σ·L/E_z | 0.46296 mm | 0.46296 mm | < 0.001 % | 0.1 % | |
| | B (tension in the layer plane): u = σ·L/E_xy | 0.37313 mm | 0.37313 mm | < 0.001 % | 0.1 % | |
| | A transverse: uy = −ν₁₃·σ/E_xy·10 | 0.013433 mm | 0.013433 mm | < 0.001 % | 1 % | |
| | B vertical: uz = −ν₁₃·σ/E_xy·10 | 0.013433 mm | 0.013433 mm | < 0.001 % | 1 % | |
| | σx (global) in bar A | 10 MPa | 9.9999 MPa | < 0.001 % | 0.1 % | |
| | Tsai-Hill A: λ = R_m,z/σ | 3.5 | 3.5 | < 0.001 % | 0.2 % | |
| | Tsai-Hill B: λ = R_m/σ | 3.9 | 3.9 | < 0.001 % | 0.2 % | |

## Beams and frames

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **F1** ✔ | Beams: IPE 200 cantilever (force, torsion with J, distributed load) – exact Timoshenko, 1 element | *closed-form solution*: Timoshenko beam theory (bending + shear): closed-form member results (S. P. Timoshenko: Strength of Materials) | | | | – |
| | deflection under the force F·L³/(3EI) + F·L/(κGA) (1 element) | 0.67118 mm | 0.67118 mm | < 0.001 % | 1e-07 % | |
| | root moment M = F·L | 2e+06 N·mm | 2e+06 N·mm | < 0.001 % | 1e-08 % | |
| | shear force Vz = −F | 1000 N | 1000 N | < 0.001 % | 1e-08 % | |
| | reaction Fz | 1000 N | 1000 N | < 0.001 % | 1e-08 % | |
| | twist T·L/(G·J) (J ≠ I_p) | 0.035475 rad | 0.035475 rad | < 0.001 % | 1e-07 % | |
| | torque T | 1e+05 N·mm | 1e+05 N·mm | < 0.001 % | 1e-08 % | |
| | distributed load: deflection q·L⁴/(8EI) + q·L²/(2κGA) | 0.5078 mm | 0.5078 mm | < 0.001 % | 1e-07 % | |
| | distributed load: root moment q·L²/2 | 2e+06 N·mm | 2e+06 N·mm | < 0.001 % | 1e-08 % | |
| | distributed load: displacement from the member diagram at its end = node displacement | -0.5078 mm | -0.5078 mm | < 0.001 % | 1e-07 % | |
| **F2** ✔ | Beams: simply supported HEB 200 under a distributed load | *closed-form solution*: Timoshenko beam theory (bending + shear): closed-form member results (S. P. Timoshenko: Strength of Materials) | | | | – |
| | mid-span deflection 5·q·L⁴/(384EI) + q·L²/(8κGA) (1 element) | 2.8863 mm | 2.8863 mm | < 0.001 % | 1e-07 % | |
| | mid-span moment = q·L²/8 (tension at the bottom: My < 0) | 2.0000e+7 N·mm | 2.0000e+7 N·mm | < 0.001 % | 1e-08 % | |
| | max \|M\| of the diagram = q·L²/8 (where V = 0) | 2.0000e+7 N·mm | 2.0000e+7 N·mm | < 0.001 % | 1e-08 % | |
| | reaction A = q·L/2 | 20000 N | 20000 N | < 0.001 % | 1e-08 % | |
| | reaction B = q·L/2 | 20000 N | 20000 N | < 0.001 % | 1e-08 % | |
| **F3** ✔ | Beams: portal frame (slope-deflection) | *closed-form solution* (relation stated, no work cited): Slope-deflection method of structural analysis for a portal frame with axially rigid members: M_ij = 2EI/L·(2θ_i + θ_j − 3ψ) + fixed-end moment, joint equilibrium ΣM = 0 (hand solution of the case) | | | | – |
| | moment at the column base (slope-deflection, axially rigid members) | 1.0828e+7 N·mm | 1.0828e+7 N·mm | < 0.001 % | 0.001 % | |
| | moment at the frame corner | 9.1716e+06 N·mm | 9.1716e+06 N·mm | < 0.001 % | 0.001 % | |
| | moment at the base of the second column | 1.0828e+7 N·mm | 1.0828e+7 N·mm | < 0.001 % | 0.001 % | |
| | sum of the horizontal reactions = H | 10000 N | 10000 N | < 0.001 % | 1e-05 % | |
| **F5** ✔ | Beams: two-bar truss (axial forces) | *closed-form solution* (relation stated, no work cited): Method of joints for a two-bar truss; δ = N·L/(E·A) | | | | – |
| | axial force AC = −P/(2·sin θ) | -6009.3 N | -6009.3 N | < 0.001 % | 1e-08 % | |
| | axial force BC | -6009.3 N | -6009.3 N | < 0.001 % | 1e-08 % | |
| | bending moment in a truss member = 0 (absolute) | 0 N·mm | 2.8422e-14 N·mm | 2.8422e-14 N·mm | 1.0000e-6 N·mm | |
| | vertical displacement of the apex = −N·L/(EA·sin θ) | -0.19735 mm | -0.19735 mm | < 0.001 % | 1e-07 % | |
| **F6** ✔ | Beams: Gerber beam with a hinge | *closed-form solution* (relation stated, no work cited): Statics of a Gerber beam: the hinge transmits no moment (M = 0 at the hinge), the reactions follow from ΣF = 0, ΣM = 0 | | | | – |
| | moment at the clamp = P/2·L₁ | 1e+06 N·mm | 1e+06 N·mm | < 0.001 % | 1e-08 % | |
| | moment at the hinge = 0 (absolute) | 0 N·mm | 0 N·mm | 0 N·mm | 1.0000e-6 N·mm | |
| | reaction B = P/2 | 500 N | 500 N | < 0.001 % | 1e-08 % | |
| | reaction A = P/2 | 500 N | 500 N | < 0.001 % | 1e-08 % | |
| **F7** ✔ | Beams: natural frequencies of a simply supported beam (Timoshenko, torsion) | *published solution*: Timoshenko beam frequencies with shear deformation and rotary inertia; torsional frequency f = 1/(4L)·√(G·J/(ρ·I_p)) (R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations) | | | | – |
| | f₁ weak axis (Timoshenko, 8 elements) | 11.343 Hz | 11.344 Hz | 0.00244 % | 0.01 % | |
| | f₂ weak axis | 45.268 Hz | 45.285 Hz | 0.0384 % | 0.1 % | |
| | f₃ weak axis | 101.46 Hz | 101.65 Hz | 0.189 % | 0.5 % | |
| | f₁ strong axis (shear + rotary inertia) | 41.398 Hz | 41.404 Hz | 0.0154 % | 0.05 % | |
| | torsional frequency (A fixed, B free): 1/(4L)·√(GJ/(ρ·I_p)) | 11.598 Hz | 11.617 Hz | 0.161 % | 0.3 % | |
| **F8** ✔ | Beams + solids: a member rigidly attached to the face of a cantilever solid | *closed-form solution*: Euler-Bernoulli cantilever with shear (Timoshenko beam theory (bending + shear): closed-form member results (S. P. Timoshenko: Strength of Materials)); statics of the connection | | | | 22 137 nodes, h = 2.5 mm |
| | member tip deflection (cantilever 400 mm, E–B + shear) | 6.0982 mm | 6.0692 mm | 0.475 % | 3 % | |
| | member moment at the connection = P·200 | 20000 N·mm | 20000 N·mm | < 0.001 % | 0.1 % | |
| | reaction at the clamp of the solid Fz = P | 100 N | 99.999 N | < 0.001 % | 0.1 % | |
| | force member → solid Fz (.dat, 6 digits) | -100 N | -99.911 N | 0.0893 % | 0.5 % | |
| **F9** ✔ | Beams: linear buckling (Euler + Engesser shear correction) | *published solution*: Euler critical load P_E = π²·E·I/(β·L)² with Engesser's shear correction P_cr = P_E/(1 + P_E/(G·A_s)) | | | | – |
| | buckling of the cantilever β = 2 (Euler + Engesser, 8 elements) | 184.19 | 184.19 | < 0.001 % | 0.02 % | |
| | buckling of the pinned column β = 1 (Euler + Engesser) | 1103.7 | 1103.7 | 0.00732 % | 0.02 % | |
| **F9M** ✔ | Beams: buckling – cross-check against MYSTRAN (SOL 105) | *independent solver (MYSTRAN)*: MYSTRAN (SOL 105 eigenvalue buckling, CBAR elements), open-source Nastran dialect solver | | | | – |
| | MYSTRAN (4 CBAR elements) vs own beam solver: λ1 | 184.42 | 184.43 | 0.00306 % | 0.1 % | |
| | MYSTRAN (4 CBAR elements) vs own beam solver: λ2 | 1107.2 | 1107.8 | 0.048 % | 0.1 % | |
| **F3M** ✔ | Beams: cross-check of the frame F3 against MYSTRAN (CBAR) | *independent solver (MYSTRAN)*: MYSTRAN (SOL 101, CBAR elements), open-source Nastran dialect solver | | | | – |
| | MYSTRAN CBAR vs own beam solver: max node-displacement difference / max displacement | 0 | 2.8607e-8 | 2.8607e-8 | 1.0000e-6 | |
| **F10** ✔ | Beams: accuracy of 1 element, rolled section, rotational spring, independence of subdivision | *closed-form solution*: Timoshenko beam theory (bending + shear): closed-form member results (S. P. Timoshenko: Strength of Materials) | | | | – |
| | strong axis: w = F·L³/(3EI_y) + F·L/(G·A_sz) | 0.67118 mm | 0.67118 mm | < 0.001 % | 1e-08 % | |
| | strong axis: end rotation θy = F·L²/(2EI) (positive about y = downwards) | 4.9012e-4 rad | 4.9012e-4 rad | < 0.001 % | 1e-08 % | |
| | weak axis: v = F·L³/(3EI_z) + F·L/(G·A_sy) | 8.933 mm | 8.933 mm | < 0.001 % | 1e-08 % | |
| | self-weight: tip deflection q·L⁴/(8EI) + q·L²/(2G·A_sz) | 0.11139 mm | 0.11139 mm | < 0.001 % | 1e-08 % | |
| | 4 elements: root moment = F·L | 2e+06 N·mm | 2e+06 N·mm | < 0.001 % | 1e-08 % | |
| | 16 elements: root moment = F·L | 2e+06 N·mm | 2e+06 N·mm | < 0.001 % | 1e-08 % | |
| | section rolled by 30°: displacement uy | -3.5775 mm | -3.5775 mm | < 0.001 % | 1e-08 % | |
| | section rolled by 30°: displacement uz | -2.7366 mm | -2.7366 mm | < 0.001 % | 1e-08 % | |
| | rotational spring k_θy: deflection = F·L³/(3EI) + F·L/(GA) + θ·L | 1.4712 mm | 1.4712 mm | < 0.001 % | 1e-07 % | |
| | rotational spring k_θy: reaction moment = −k·θ = F·L | 2e+06 N·mm | 2e+06 N·mm | < 0.001 % | 1e-07 % | |
| **F11M** ✔ | Beams: cross-check of the spatial frame against MYSTRAN (CBAR) | *independent solver (MYSTRAN)*: MYSTRAN (SOL 101, CBAR elements), open-source Nastran dialect solver | | | | – |
| | MYSTRAN CBAR vs own beam solver (spatial frame): max node-displacement difference / max displacement | 0 | 3.7598e-8 | 3.7598e-8 | 1.0000e-6 | |
| **F12** ✔ | Beams: torsional and bending frequencies of a cantilever, convergence | *published solution*: Torsional frequency f = 1/(4L)·√(G·J/(ρ·I_p)); Euler-Bernoulli cantilever βL = 1.875104 (R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations) | | | | – |
| | torsion mode 1, 4 elements (O(h²)) | 801.91 Hz | 807.08 Hz | 0.644 % | 0.8 % | |
| | torsion mode 1, 8 elements (O(h²)) | 801.91 Hz | 803.2 Hz | 0.161 % | 0.22 % | |
| | torsion mode 1, 16 elements | 801.91 Hz | 802.24 Hz | 0.0402 % | 0.06 % | |
| | cantilever bending mode 1 = Euler–Bernoulli (slender member, small shear) | 57.757 Hz | 57.487 Hz | 0.467 % | 0.8 % | |
| **F13** ✔ | Beams: linear buckling – Euler (2-8 elements) and Engesser | *published solution*: Euler critical loads for pinned-pinned, clamped-free and pinned-clamped columns (λ_cr = 20.1907·EI/L²); Engesser shear correction | | | | – |
| | pinned column, E–B, 4 elements | 327.86 | 328.03 | 0.0512 % | 0.1 % | |
| | pinned column, E–B, 8 elements | 327.86 | 327.87 | 0.00328 % | 0.01 % | |
| | cantilever, E–B, 2 elements | 81.965 | 82.007 | 0.0512 % | 0.1 % | |
| | cantilever, E–B, 4 elements | 81.965 | 81.968 | 0.00328 % | 0.01 % | |
| | pinned–clamped, E–B, 8 elements (λ_cr = 20.1907·EI/L²) | 670.72 | 670.81 | 0.0136 % | 0.05 % | |
| | pinned column with shear: Engesser P_E/(1 + P_E/(G·A_s)), 8 elements | 327.13 | 327.15 | 0.00609 % | 0.02 % | |
| | cantilever with shear: Engesser P_E/(1 + P_E/(G·A_s)), 8 elements | 81.92 | 81.92 | < 0.001 % | 0.02 % | |
| **F14** ✔ | Beams: temperature – restrained expansion, gradient, free bar | *closed-form solution* (relation stated, no work cited): Restrained bar N = −E·A·α·ΔT; temperature gradient: M = −E·I·α·ΔT_g/h, simply supported deflection α·ΔT_g·L²/(8h) | | | | – |
| | restrained bar: N = −E·A·α·ΔT | -3.589e+05 N | -3.589e+05 N | < 0.001 % | 1e-10 % | |
| | restrained bar: reaction = E·A·α·ΔT | 3.589e+05 N | 3.589e+05 N | < 0.001 % | 1e-10 % | |
| | restrained bar: displacement of node B = 0 (absolute) | 0 mm | 0 mm | 0 mm | 1.0000e-12 mm | |
| | clamped beam: M_y = −E·I·α·ΔT_g/h | -7.3452e+06 N·mm | -7.3452e+06 N·mm | < 0.001 % | 1e-10 % | |
| | free bar: elongation α·ΔT·L | 1.2 mm | 1.2 mm | < 0.001 % | 1e-10 % | |
| | free bar: N = 0 (absolute) | 0 N | 0 N | 0 N | 1.0000e-8 N | |
| | simply supported beam with a gradient: mid-span deflection α·ΔT_g·L²/(8h) | 0.9 mm | 0.9 mm | < 0.001 % | 1e-07 % | |
| | simply supported beam with a gradient: M = 0 (absolute) | 0 N·mm | 9.9826e-9 N·mm | 9.9826e-9 N·mm | 1.0000e-6 N·mm | |

## 2D sections and shells

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **A1** ✔ | Tube r_i 20 / r_o 40, p = 100 MPa as an axisymmetric problem | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) | | | | – |
| | σθ at r_i (Lamé) | 166.67 MPa | 166.52 MPa | 0.0898 % | 1 % | |
| | σθ at r_o (Lamé) | 66.667 MPa | 66.65 MPa | 0.0254 % | 1 % | |
| | von Mises at r_i | 233.33 MPa | 233.07 MPa | 0.112 % | 1 % | |
| | u_r(r_i) | 0.01873 mm | 0.018729 mm | 0.00565 % | 1 % | |
| | σr at r_i = −p | 100 MPa | 99.843 MPa | 0.157 % | 1 % | |
| **A2** ✔ | Thick-walled sphere r_i 20 / r_o 40, p = 100 MPa (axisymmetric) | *closed-form solution*: Lamé solution of a thick-walled sphere under internal pressure (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) | | | | – |
| | σθ at r_i, polar angle 45° (Lamé sphere) | 71.429 MPa | 71.336 MPa | 0.13 % | 1 % | |
| | σθ at r_o, polar angle 45° | 21.429 MPa | 21.429 MPa | < 0.001 % | 1 % | |
| | σθ at r_i, equator | 71.429 MPa | 71.314 MPa | 0.161 % | 1 % | |
| | \|u\|(r_i), polar angle 45° | 0.007619 mm | 0.0076187 mm | 0.00455 % | 1 % | |
| | u_r(r_i) at the equator | 0.007619 mm | 0.0076186 mm | 0.0064 % | 1 % | |
| | meridional σ = σθ (tangential at the 45° point) | 71.429 MPa | 71.386 MPa | 0.059 % | 1 % | |
| **A3** ✔ | Plate with a Ø10 hole under tension (plane stress) | *handbook / standard*: R. E. Peterson: Stress Concentration Factors (Wiley) – plate of finite width with a central circular hole, polynomial of Heywood for K_t on the net section | | | | – |
| | σx max at the hole (K_t Heywood, d/W = 0.1) | 302.35 MPa | 305.55 MPa | 1.06 % | 3 % | |
| | σx max over the nodes = at the hole | 302.35 MPa | 305.55 MPa | 1.06 % | 3 % | |
| | reaction Fx on the symmetry plane (100 MPa × 50 × 2.5 mm) | 12500 N | 12500 N | < 0.001 % | 0.1 % | |
| **A4** ✔ | Thick-walled cylinder, p = 100 MPa (plane strain) | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) (plane strain) | | | | – |
| | σθ at r_i (Lamé) | 166.67 MPa | 166.64 MPa | 0.0173 % | 1 % | |
| | σθ at r_o (Lamé) | 66.667 MPa | 66.67 MPa | 0.0044 % | 1 % | |
| | σz at r_i = ν·(σr + σθ) | 20 MPa | 20.002 MPa | 0.0105 % | 1 % | |
| | σz at r_o = ν·(σr + σθ) | 20 MPa | 20.006 MPa | 0.0315 % | 1 % | |
| | u_r(r_i), plane strain | 0.018159 mm | 0.018159 mm | < 0.001 % | 1 % | |
| | reaction \|Fx\| on the symmetry plane = p·r_i·t | 2000 N | 2000 N | < 0.001 % | 0.1 % | |
| **A5** ✔ | Rotating disc R 100, t 4, ω = 1000 rad/s (axisymmetric) | *closed-form solution*: Stresses of a rotating solid thin disc (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) | | | | – |
| | σr = σθ at the centre (thin disc) | 32.381 MPa | 32.386 MPa | 0.0141 % | 1 % | |
| | σθ at the centre | 32.381 MPa | 32.386 MPa | 0.0141 % | 1 % | |
| | σr at r = 50 mm | 24.286 MPa | 24.288 MPa | 0.00839 % | 1 % | |
| | σθ at r = 50 mm | 27.72 MPa | 27.722 MPa | 0.0077 % | 1 % | |
| | σθ at the rim | 13.738 MPa | 13.736 MPa | 0.0102 % | 1 % | |
| | u_r at the rim | 0.0065417 mm | 0.0065426 mm | 0.0144 % | 1 % | |
| **A5P** ✔ | Rotating quarter disc R 100, t 4, ω = 1000 rad/s (plane stress) | *closed-form solution*: Stresses of a rotating solid thin disc (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity); section force ρ·ω²·t·R³/3 | | | | – |
| | σx at the centre (= σr = σθ) | 32.381 MPa | 32.384 MPa | 0.00942 % | 1 % | |
| | σθ at r = 50 mm | 27.72 MPa | 27.72 MPa | 0.00247 % | 1 % | |
| | σθ at the rim | 13.738 MPa | 13.676 MPa | 0.45 % | 1 % | |
| | u_r at the rim | 0.0065417 mm | 0.0065417 mm | < 0.001 % | 1 % | |
| | reaction \|Fx\| on the symmetry plane = ρ·ω²·t·R³/3 | 10467 N | 10467 N | < 0.001 % | 0.2 % | |
| **Sh1** ✔ | Shell: cantilever of a 200×40×5 plate, end force | *closed-form solution*: Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W (S. P. Timoshenko: Strength of Materials) | | | | – |
| | tip deflection (Timoshenko beam) | 3.0491 mm | 2.987 mm | 2.04 % | 3 % | |
| | σx top surface at x = L/2 (M/W) | 60 MPa | 59.992 MPa | 0.0139 % | 2 % | |
| | σx bottom surface at x = L/2 (magnitude) | 60 MPa | 59.992 MPa | 0.0139 % | 2 % | |
| | 1 + membrane σx / (M/W) (≈ 1) | 1 | 1 | < 0.001 % | 1 % | |
| **Sh2** ✔ | Shell: square plate simply supported on its edge, uniform pressure | *published solution*: S. Timoshenko, S. Woinowsky-Krieger: Theory of Plates and Shells – w_max = 0.00406·q·a⁴/D, M = 0.0479·q·a² | | | | – |
| | centre deflection (Timoshenko, 0.00406·q·a⁴/D) | 0.43237 mm | 0.43683 mm | 1.03 % | 3 % | |
| | σx top surface at the centre (6M/t², M = 0.0479·q·a²) | 18.394 MPa | 18.512 MPa | 0.644 % | 3 % | |
| | σy bottom surface at the centre | 18.394 MPa | 18.513 MPa | 0.649 % | 3 % | |
| | reaction Fz = q·a² | 1600 N | 1600 N | < 0.001 % | 0.1 % | |
| **Sh3** ✔ | Shell: cylinder under internal pressure (σ_θ = p·R/t) | *closed-form solution*: Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity) (thin-wall limit σ_θ = p·R/t) | | | | – |
| | σθ mid-surface (Lamé, mean of r_i and r_o ≈ p·r_i/t) | 24.51 MPa | 24.509 MPa | 0.00407 % | 1 % | |
| | σθ bending part (Lamé, (σ(r_o) − σ(r_i))/2) | -0.5 MPa | -0.49062 MPa | 1.88 % | 5 % | |
| | mid-surface u_r (Lamé) | 0.011734 mm | 0.011665 mm | 0.587 % | 1 % | |
| | σθ thin-wall p·R/t (3 % tolerance) | 25 MPa | 24.509 MPa | 1.96 % | 3 % | |

## Plasticity and nonlinear buckling

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **PL1** ✔ | Plasticity: tension of a bar with bilinear hardening (BISO) | *closed-form solution* (relation stated, no work cited): Uniaxial stress state of a bilinear elastic-plastic material: σ = R_e + E_t·(ε − R_e/E), ε_p = ε − σ/E | | | | 1 001 nodes, h = 5 mm |
| | stress SZZ in the plastic range | 325 MPa | 326.67 MPa | 0.515 % | 3 % | |
| | tensile reaction RF_Z (signed: the support pushes against the tension) | -32500 N | -32524 N | 0.073 % | 3 % | |
| | equivalent plastic strain PEEQ | 0.003375 mm/mm | 0.0034504 mm/mm | 2.23 % | 8 % | |
| **PL2** ✔ | Plasticity: bending of a cantilever into the plastic range and unloading | *closed-form solution* (relation stated, no work cited): Elastic-plastic bending of a rectangular section with bilinear hardening: moment-curvature relation M(κ) and residual stress σ_res = σ(κ) − E·κ·y after elastic unloading (standard plasticity theory, closed form evaluated in the case) | | | | 14 443 nodes, h = 1.5 mm |
| | 0.8 M_y (elastic): mid-span curvature, signed (moment–curvature relation) | 2.0000e-4 1/mm | 2.0046e-4 1/mm | 0.231 % | 3 % | |
| | 0.8 M_y (elastic): σ_xx at mid-span – RMS deviation from the closed form / R_e | 0 | 5.4172e-6 | 5.4172e-6 | 0.03 | |
| | 0.8 M_y (elastic): PEEQ = 0 | 0 | 0 | 0 | 1.0000e-6 | |
| | 1.2 M_y: mid-span curvature, signed (moment–curvature relation) | 3.2227e-4 1/mm | 3.2329e-4 1/mm | 0.317 % | 3 % | |
| | 1.2 M_y: σ_xx at mid-span – RMS deviation from the closed form / R_e | 0 | 0.017595 | 0.017595 | 0.03 | |
| | 1.2 M_y: PEEQ at the surface | 3.5772e-4 | 3.6540e-4 | 2.15 % | 8 % | |
| | 1.45 M_y: mid-span curvature, signed (moment–curvature relation) | 7.0063e-4 1/mm | 6.9676e-4 1/mm | 0.552 % | 3 % | |
| | 1.45 M_y: σ_xx at mid-span – RMS deviation from the closed form / R_e | 0 | 0.022715 | 0.022715 | 0.03 | |
| | 1.45 M_y: PEEQ at the surface | 0.0022306 | 0.0022089 | 0.97 % | 8 % | |
| | unloading: residual curvature at mid-span, signed | 3.3813e-4 1/mm | 3.3342e-4 1/mm | 1.39 % | 5 % | |
| | unloading: residual σ_xx – RMS deviation from the closed form / R_e | 0 | 0.022718 | 0.022718 | 0.03 | |
| | unloading: surface PEEQ = value reached in the loading | 0.0022306 | 0.0022089 | 0.97 % | 8 % | |
| | unloading: residual σ_xx, top fibre (compression) | -107.99 MPa | -105.21 MPa | 2.58 % | 8 % | |
| | unloading: residual σ_xx, bottom fibre (tension) | 107.99 MPa | 105.29 MPa | 2.5 % | 8 % | |
| **PL4** ✔ | Plasticity: tension-compression cycle – kinematic vs isotropic hardening | *closed-form solution* (relation stated, no work cited): Uniaxial cyclic response of a bilinear material with kinematic (Bauschinger effect: reverse yielding at σ_max − 2·R_e) and isotropic hardening (reverse yielding at −σ_max), linear hardening H = E·E_t/(E − E_t) | | | | 2 043 nodes, h = 5 mm |
| | kinematic hardening, +325 MPa: strain u/L | 0.005 | 0.0050739 | 1.48 % | 5 % | |
| | kinematic hardening, +325 MPa: PEEQ | 0.003375 | 0.0034396 | 1.91 % | 3 % | |
| | kinematic hardening, unloading 1: strain u/L | 0.003375 | 0.0034349 | 1.77 % | 5 % | |
| | kinematic hardening, unloading 1: PEEQ | 0.003375 | 0.0034396 | 1.91 % | 3 % | |
| | kinematic hardening, −325 MPa: strain u/L | -0.005 | -0.0049334 | 1.33 % | 5 % | |
| | kinematic hardening, −325 MPa: PEEQ | 0.010125 | 0.010164 | 0.38 % | 3 % | |
| | kinematic hardening, unloading 2: strain u/L | -0.003375 | -0.003322 | 1.57 % | 5 % | |
| | kinematic hardening, unloading 2: PEEQ | 0.010125 | 0.010164 | 0.38 % | 3 % | |
| | isotropic hardening, +325 MPa: strain u/L | 0.005 | 0.005074 | 1.48 % | 5 % | |
| | isotropic hardening, +325 MPa: PEEQ | 0.003375 | 0.0034397 | 1.92 % | 3 % | |
| | isotropic hardening, unloading 1: strain u/L | 0.003375 | 0.003435 | 1.78 % | 5 % | |
| | isotropic hardening, unloading 1: PEEQ | 0.003375 | 0.0034397 | 1.92 % | 3 % | |
| | isotropic hardening, −325 MPa: strain u/L | 0.00175 | 0.0018016 | 2.95 % | 5 % | |
| | isotropic hardening, −325 MPa: PEEQ | 0.003375 | 0.0034397 | 1.92 % | 3 % | |
| | isotropic hardening, unloading 2: strain u/L | 0.003375 | 0.003435 | 1.78 % | 5 % | |
| | isotropic hardening, unloading 2: PEEQ | 0.003375 | 0.0034397 | 1.92 % | 3 % | |
| **BK1** ✔ | Nonlinear buckling: elastic cantilever with a mode-shaped imperfection | *closed-form solution*: Euler load P_cr = π²·E·I/(4·L²); amplification of an imperfection 1/(1 − N/N_cr), tip deflection e₀·r/(1 − r) | | | | 22 137 nodes, h = 2.5 mm |
| | N_cr (linear buckling factor × F) = π²EI/(4L²) | 21590 N | 21652 N | 0.286 % | 2 % | |
| | tip deflection at N/N_cr = 0.5: e0·r/(1 − r) | 0.2 mm | 0.20211 mm | 1.05 % | 2 % | |
| | tip deflection at N/N_cr = 0.8: e0·r/(1 − r) | 0.8 mm | 0.80256 mm | 0.32 % | 2 % | |
| | tip deflection at N/N_cr = 0.9: e0·r/(1 − r) | 1.8 mm | 1.8193 mm | 1.07 % | 2 % | |
| **BK2** ✔ | GNIA + first yield of a pin-ended column = EN 1993-1-1 buckling curve c | *handbook / standard*: EN 1993-1-1 (Eurocode 3): flexural buckling curve c, Ayrton-Perry formulation, reduction factor χ | | | | 20 983 nodes, h = 4 mm |
| | λ̄ = √(A·f_y/N_cr) of the model (linear buckling factor) | 0.99593 | 1.001 | 0.511 % | 2 % | |
| | GNIA first yield N_y = χ·A·f_y (EN 1993-1-1 curve c, Ayrton–Perry imperfection) | 76465 N | 76259 N | 0.27 % | 3 % | |
| **CY1** ✔ | Cyclic symmetry: statics of a sector, spinning disc (Lamé) | *closed-form solution*: Rotating disc with a bore (Lamé, plane stress; S. P. Timoshenko, J. N. Goodier: Theory of Elasticity); pressure × area | | | | 6 111 nodes, h = 6 mm |
| | pressure: reaction Fz of the sector = pressure × area | 23562 N | 23562 N | 0.00115 % | 0.01 % | |
| | spinning disc: σθ(r_i) (Lamé) | 65.312 MPa | 64.867 MPa | 0.682 % | 1 % | |
| | spinning disc: σθ(r_o) (Lamé) | 16.328 MPa | 16.334 MPa | 0.0386 % | 1 % | |
| | spinning disc: u_r(r_o) = r_o·σθ(r_o)/E | 0.0077752 mm | 0.0077746 mm | 0.00765 % | 1 % | |
| | spinning disc: tangential reaction of the bore ≈ 0 (as 1 + \|R\|/(ρ·ω²·V·r)) | 1 | 1.0001 | 0.00869 % | 0.1 % | |
| **CY2** ✔ | Cyclic symmetry: natural frequencies by nodal diameter, ring | *published solution*: In-plane bending vibration of a thin ring: f = 1/(2π)·√(E·I/(ρ·A·R⁴))·n·(n² − 1)/√(n² + 1) (Love; R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations) | | | | 4 292 nodes, h = 2 mm |
| | nodal diameter 2: in-plane ring bending (Love) | 191.29 Hz | 191.3 Hz | 0.00695 % | 2 % | |
| | nodal diameter 3: in-plane ring bending (Love) | 541.05 Hz | 540.87 Hz | 0.0338 % | 2 % | |
| | nodal diameter 4: in-plane ring bending (Love) | 1037.4 Hz | 1036.4 Hz | 0.0942 % | 2 % | |

## Mechanisms (multibody dynamics)

| Case | Quantity | Reference | FEA Studio | Difference | Tolerance | Mesh |
|---|---|---|---:|---:|---:|---|
| **MB1** ✔ | Slider-crank r 20, l 80, ω 100 rad/s: kinematics and drive torque | *closed-form solution* (relation stated, no work cited): Closed-form kinematics of a slider-crank (position, velocity, acceleration of the piston); drive torque from the power balance M = F·v/ω | | | | – |
| | piston position x(60°) | 88.102 mm | 88.102 mm | < 0.001 % | 0.0001 % | |
| | piston velocity v(60°) | -1953.8 mm/s | -1953.8 mm/s | < 0.001 % | 0.01 % | |
| | piston acceleration a(60°) | -75022 mm/s² | -75022 mm/s² | < 0.001 % | 0.01 % | |
| | drive torque M = F·v/ω | -19538 N·mm | -19538 N·mm | < 0.001 % | 0.01 % | |
| **MB2** ✔ | Four-bar linkage 40/120/80/100: Freudenstein, both branches | *published solution*: F. Freudenstein: displacement equation of the planar four-bar linkage (rocker angle θ₄ from the crank angle θ₂, open and crossed branch) | | | | – |
| | θ₄(60°), open branch | 64.943 ° | 64.943 ° | < 0.001 % | 0.002 % | |
| | θ₄(60°), open branch = Freudenstein (exact) | 1.1335 rad | 1.1335 rad | < 0.001 % | 1e-06 % | |
| | max \|θ₄ − Freudenstein\| over one revolution (open branch) | 0 rad | 1.7764e-15 rad | 1.7764e-15 rad | 1.0000e-7 rad | |
| | θ₄(60°), crossed branch | 248.23 ° | 248.23 ° | < 0.001 % | 0.002 % | |
| | θ₄(60°), crossed branch = Freudenstein (exact) | 4.3324 rad | 4.3324 rad | < 0.001 % | 1e-06 % | |
| | max \|θ₄ − Freudenstein\| over one revolution (crossed branch) | 0 rad | 2.2204e-14 rad | 2.2204e-14 rad | 1.0000e-7 rad | |
| **MB3** ✔ | Pendulum L 0.5 m, θ₀ 60°: period, pin force, energy | *closed-form solution* (relation stated, no work cited): Period of a physical pendulum with the complete elliptic integral T = 4·√(L_eq/g)·K(sin²(θ₀/2)); pin force m·g·(3 − 2cos θ₀); energy conservation | | | | – |
| | period T (physical pendulum, K(sin²30°)) | 1.5224 s | 1.5224 s | < 0.001 % | 0.01 % | |
| | pin force at the bottom = m·g·(3 − 2·cos θ₀) = 2·m·g | 39.237 N | 39.237 N | < 0.001 % | 0.01 % | |
| | energy conservation (max − min)/(m·g·L) | 0 | 3.9441e-8 | 3.9441e-8 | 1.0000e-5 | |
| **MB4** ✔ | Cylinder rolling on a 30° plane (rolling constraint) | *closed-form solution* (relation stated, no work cited): Rolling cylinder on an incline: a = (2/3)·g·sin α, friction force m·g·sin α/3, minimum friction coefficient tan α/3 | | | | – |
| | rolling acceleration a = (2/3)·g·sin α (μ 0.3 ≥ tan α/3 = 0.192) | 3.27 m/s² | 3.27 m/s² | < 0.001 % | 0.01 % | |
| | rolling friction force F = m·g·sin α/3 (constraint moment / R) | 1.635 N | 1.635 N | < 0.001 % | 0.01 % | |
| | required friction coefficient = \|friction force\| / normal force = tan α/3 | 0.19245 | 0.19245 | < 0.001 % | 0.1 % | |
| **MB5** ✔ | Block sliding to rest, v₀ 1 m/s, μ 0.2 | *closed-form solution* (relation stated, no work cited): Coulomb friction deceleration a = μ·g, stopping time v₀/(μ·g), stopping distance v₀²/(2·μ·g) | | | | – |
| | deceleration a = μ·g | 1962 mm/s² | 1962 mm/s² | < 0.001 % | 0.01 % | |
| | stopping time t = v₀/(μ·g) | 0.50968 s | 0.50968 s | < 0.001 % | 0.01 % | |
| | stopping distance s = v₀²/(2·μ·g) | 254.84 mm | 254.84 mm | < 0.001 % | 0.1 % | |
| | velocity after stopping (t > 0.6 s) / v₀ (≈ 0) | 0 | 1.6468e-124 | 1.6468e-124 | 0.001 | |
| **MB6** ✔ | Stick-slip on a belt (μ_s 0.4, μ_k 0.3) | *closed-form solution* (relation stated, no work cited): Stick-slip of a mass on a moving belt: breakaway x_s = μ_s·m·g/k, oscillation about x_k = μ_k·m·g/k, period of the slip-stick cycle | | | | – |
| | breakaway displacement x_s = μ_s·m·g/k (peak friction / k) | 3.924 mm | 3.9207 mm | 0.0838 % | 0.5 % | |
| | centre of the slip oscillation x_k = μ_k·m·g/k | 2.943 mm | 2.943 mm | < 0.001 % | 0.5 % | |
| | largest displacement x_k + √((x_s − x_k)² + (v/ω)²) | 6.2539 mm | 6.2873 mm | 0.533 % | 1 % | |
| | stick-slip period T | 0.19929 s | 0.1995 s | 0.109 % | 0.5 % | |
| **MB9** ✔ | Gear pair and screw joint | *closed-form solution* (relation stated, no work cited): Gear ratio ω₂/ω₁ = z₁/z₂; screw joint travel = lead × turns, torque M = F·P/(2π) | | | | – |
| | gear ratio ω₂/ω₁ | -2.5 | -2.5 | < 0.001 % | 1e-07 % | |
| | angle of wheel 2 = ratio × angle of wheel 1 (end) | -15.708 rad | -15.708 rad | < 0.001 % | 1e-07 % | |
| | nut travel = lead × turns (2 turns) | 8 mm | 8 mm | < 0.001 % | 1e-07 % | |
| | screw torque M = F·P/(2π) | 636.62 N·mm | 636.62 N·mm | < 0.001 % | 0.0001 % | |
| **MB4C** ✔ | Cylinder sliding on a 30° plane, μ 0.1 – mesh contact in Chrono | *closed-form solution*: Coulomb friction of a solid cylinder on an incline of angle α with μ < tan α/3 (the cylinder slides while it turns): a = g·(sin α − μ·cos α), angular acceleration ε = 2·μ·g·cos α/R (I = m·R²/2). The angular acceleration is checked to 5 % only (it comes out about 3 % high; the miss is attributed to the polygonal triangulation of the cylinder and to the Euler time stepping, whose error is proportional to the step). Solved by the multibody solver Chrono with contact of the triangle meshes of the CAD parts: the case needs the optional module chrono. | | | | – |
| | sliding μ 0.1: a = g·(sin α − μ·cos α) | 4.0554 m/s² | 4.0554 m/s² | < 0.001 % | 0.1 % | |
| | sliding: angular acceleration ε = 2·μ·g·cos α/R (polygonal triangulation of the cylinder, Euler time step) | 33.983 rad/s² | 32.888 rad/s² | 3.22 % | 5 % | |
| **MB4R** ✔ | Cylinder rolling on a 30° plane, μ 0.3 – mesh contact in Chrono | *closed-form solution* (relation stated, no work cited): Rolling cylinder on an incline: a = (2/3)·g·sin α, friction force m·g·sin α/3, rolling without slip (v = ω·R) while μ ≥ tan α/3 (0.192 for 30°). Solved by the multibody solver Chrono with contact of the triangle meshes of the CAD parts: the case needs the optional module chrono. | | | | – |
| | rolling μ 0.3: a = (2/3)·g·sin α | 3.27 m/s² | 3.2518 m/s² | 0.558 % | 1 % | |
| | rolling without slip: \|v − ω·R\|/v at the end | 0 | 6.2359e-4 | 6.2359e-4 | 0.01 | |
| **KK5** ✔ | Eccentric cam from CAD in a multibody model – lift of the follower, mesh contact in Chrono | *closed-form solution* (relation stated, no work cited): Closed-form lift of a roller follower on an eccentric disc (cam radius R_c, eccentricity e, roller radius r_f, guide through the axis): the roller centre stays at the distance L = R_c + r_f from the disc centre, so its height is y = e·cos θ + √(L² − e²·sin² θ) and the lift from the lowest position s(θ) = e·cos θ + √(L² − e²·sin² θ) − (L − e). Solved by the multibody solver Chrono with contact of the triangle meshes of the CAD parts: the case needs the optional module chrono. | | | | – |
| | lift s(60°) = e·cos θ + √((R_c + r_f)² − e²·sin² θ) − (R_c + r_f − e) | 7.2649 mm | 7.2624 mm | 0.0346 % | 0.1 % | |
| | lift s(90°) = e·cos θ + √((R_c + r_f)² − e²·sin² θ) − (R_c + r_f − e) | 4.6863 mm | 4.6844 mm | 0.0393 % | 0.1 % | |

## Cross-check against MYSTRAN

The same model, mesh (coarsened ×2) and nodal loads are solved by CalculiX and by MYSTRAN Version 19.0.0 (independent assembly, boundary conditions and solver). Criterion: max|Δu| / max|u| and |ΔR| / |R| below 0.001.

| Case | Load case | max\|u\| CalculiX [mm] | max\|u\| MYSTRAN [mm] | max\|Δu\| / max\|u\| | \|ΔR\| / \|R\| | Mesh |
|---|---|---:|---:|---:|---:|---|
| V1 | force 100 N | 0.757197 | 0.757197 | 6.94e-07 | 3.20e-06 | 3 875 nodes, h = 5 mm |
| V2 | pressure 100 MPa | 0.047619 | 0.047619 | 1.04e-06 | 9.39e-08 | 821 nodes, h = 6 mm |
| V3 | tension 100 MPa | 0.0720007 | 0.0720007 | 7.17e-07 | 1.75e-07 | 4 023 nodes, h = 8 mm |
| V4 | internal pressure 100 MPa | 0.0187424 | 0.0187424 | 2.67e-06 | 9.65e-08 | 1 115 nodes, h = 5 mm |
| V5 | self-weight | 0.000182112 | 0.000182112 | 2.74e-06 | 2.13e-07 | 981 nodes, h = 50 mm |
| V6 | force 100 N | 0.757178 | 0.757178 | 7.36e-07 | 3.34e-06 | 4 038 nodes, h = 5 mm |
| V7 | torque 100 N·m | 0.157653 | 0.157558 | 6.01e-04 | 5.94e-06 | 4 704 nodes, h = 5 mm |

## Not run on this platform

| Case | Title | Reason |
|---|---|---|
| K2 | Hertz contact: sphere on a plate (mortar) | mortar contact is substituted by surface-to-surface (the mortar matrix of CalculiX 2.23 depends on rounding; FEA_MORTAR=1 runs it) |
| KK4 | Impact of a sphere on a rigid plate (*DYNAMIC + contact) | takes hours (about 4 h with the shipped CalculiX): python -m verification.run_public --only KK4 |


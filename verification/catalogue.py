"""Catalogue of the verification cases: what each case is compared with, where that reference comes from, and whether the case is
published on the public verification page (docs/verification/).

A case is PUBLIC only when its reference does not come from the program itself and the page says where it comes from: a work is
cited (author, title, the relation), or - for an elementary relation - the first-principles relation is stated in the citation.
`Entry.source_kind` tells the two apart ("work" / "relation" / "solver") and the page marks a case whose reference is a stated
relation without a cited work. The kinds of reference:

  analytic    closed-form solution derived from first principles (beam theory, Lamé, Fourier's law, Euler buckling ...)
  handbook    tabulated coefficients, empirical correlations or code-of-practice formulas (stress concentration factors, convection
              correlations, EN / VDI / IIW rules ...)
  literature  a published solution or method with its own name (Hertz, Hetényi, Lekhnitskii, Miles, Neuber, Findley ...)
  mystran     the same model solved by MYSTRAN, an independent open-source solver (Nastran dialect) – see `fea_core/mystran.py`

Everything else is `internal`: the reference is the program itself (an equilibrium or bookkeeping check, a comparison of two runs of
the same model, a wiring test of an input table, a warning or error message) or no source can be named yet. Internal cases stay in
`verification.cases.CASES` and in the test suite, but they are NOT published and a published case never needs them.

Inside a public case some checks may still be self-consistency checks (for example "the equilibrium residual of the run").
They are listed in `Entry.hide` and are not shown on the public page; the case counts as "within tolerance" only when ALL its checks pass.

The source texts name the work and the formula the case uses (author, title, the relation); they carry no page or table numbers
except where the case's own code already states them. This module imports nothing heavy (no OCP, no numpy): the
public report, the guard test and the ECSS validation dossier (F14-S6) read it without loading the solver.

    from verification import catalogue
    entry = catalogue.get("V1")          # Entry
    catalogue.public_keys()              # keys of the published cases, in publication order
    catalogue.by_domain()                # {domain: [Entry, ...]} of the published cases
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# a cited work: an author with initials ("S. P. Timoshenko"), a standard / code of practice ("EN 1993-1-8", "VDI 2230", "ASME ...")
# or the name of a published solution ("Hertz", "Lamé solution")
_WORK = re.compile(r"\b[A-Z]\.(?: ?[A-Z]\.)* ?[A-Z][\w'’-]+|\b(?:EN|ISO|DIN|VDI|BS|IEC) ?\d|\b(?:ASME|IIW|FKM|DNV|ECSS|NAFEMS|API|ANSI|AISC|Bayer)\b|"
                   r"\b(?:Hertz|Lamé|Neuber|Miles|Hetényi|Lekhnitskii|Findley|Roark|Peterson|Heywood|Rayleigh|Euler|Engesser|Fourier|"
                   r"Castigliano|Saint-Venant|Tsai|Hill|Shigley|Incropera|Blevins|Winkler|Mohr|Tresca|Boussinesq|Mindlin)\b")

# domain key -> heading on the public page (publication order)
DOMAINS: dict[str, str] = {
    "statics": "Linear statics",
    "heat": "Heat conduction and thermal stress",
    "dynamics": "Natural frequencies, buckling and dynamics",
    "contacts": "Contacts and friction",
    "joints": "Bolted and welded joints",
    "fatigue": "Fatigue",
    "printing": "3D-printed parts",
    "beams": "Beams and frames",
    "shells_2d": "2D sections and shells",
    "nonlinear": "Plasticity and nonlinear buckling",
    "multibody": "Mechanisms (multibody dynamics)",
}

PUBLIC_TYPES = ("analytic", "handbook", "literature", "mystran")
REF_TYPES = PUBLIC_TYPES + ("internal",)
REF_TYPE_TITLES = {
    "analytic": "closed-form solution", "handbook": "handbook / standard", "literature": "published solution",
    "mystran": "independent solver (MYSTRAN)", "internal": "internal check",
}

# cases that, besides their closed-form reference, are also solved by MYSTRAN on the same (coarser) mesh
# (verification.mystran_check; shown in a separate table of the public page)
MYSTRAN_CROSS_CHECK = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")

# capability tokens of `Entry.needs`: an installable module of fea_core/modules.json, or one of
NEEDS_TOKENS = ("mortar", "mystran")          # mortar = the found ccx solves *CONTACT PAIR, TYPE=MORTAR; mystran = tools/mystran


@dataclass(frozen=True)
class Entry:
    key: str                              # key in verification.cases.CASES
    title: str                            # English title of the case (the case's own title is Czech)
    domain: str                           # key of DOMAINS
    ref_type: str                         # one of REF_TYPES
    source: str                           # citation of the reference ("" for internal cases)
    public: bool                          # published on the verification page
    why_not: str = ""                     # for a non-public case: why it is not published
    hide: tuple[str, ...] = ()            # regular expressions (re.search) of check names that are consistency checks – not published
    needs: tuple[str, ...] = ()           # modules / capabilities the case needs (modules.json keys, NEEDS_TOKENS)

    @property
    def source_kind(self) -> str:
        """"work" = a published work / standard / named solution is cited; "relation" = only the first-principles relation is stated
        (it contains an equation); "solver" = an independent solver (MYSTRAN); "" = neither (a published case needs one of them)."""
        if self.ref_type == "mystran":
            return "solver" if "MYSTRAN" in self.source else ""
        if _WORK.search(self.source):
            return "work"
        return "relation" if "=" in self.source else ""

    def shows(self, check_name: str) -> bool:
        """Is this check of a public case shown on the page (not one of the hidden consistency checks)?"""
        return not any(re.search(p, check_name) for p in self.hide)


# ---------------------------------------------------------------------------------------------------- source texts
S_BEAM = ("Timoshenko beam theory: w = F·L³/(3·E·I) + F·L/(κ·G·A), κ = 5/6 for a rectangular section, σ = M/W "
          "(S. P. Timoshenko: Strength of Materials)")
S_LAME = "Lamé solution of a thick-walled cylinder (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity)"
S_HEAT = "F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine: Fundamentals of Heat and Mass Transfer"
S_BLEVINS = "R. D. Blevins: Formulas for Natural Frequency and Mode Shape; S. S. Rao: Mechanical Vibrations"
S_SDOF = "S. S. Rao: Mechanical Vibrations (single-degree-of-freedom oscillator)"
S_HERTZ = "K. L. Johnson: Contact Mechanics (1985), ch. 4 (Hertz), ch. 11 (impact)"
S_PRESSFIT = "Lamé (thick-walled cylinders); Shigley's Mechanical Engineering Design, ch. 3-16 (press and shrink fits)"
S_SNAPFIT = "Bayer MaterialScience: Snap-Fit Joints for Plastics – A Design Guide, pp. 8–16 (hook and force equations)"
S_PETERSON = ("R. E. Peterson: Stress Concentration Factors (Wiley) – plate of finite width with a central circular hole, "
              "polynomial of Heywood for K_t on the net section")
S_STATICS = "Static equilibrium (applied load = reaction); strength of materials"
S_EC3_BOLT = ("EN 1993-1-8 (Eurocode 3, design of joints): bolt shear and bearing resistance, F_v,Rd = α_v·f_ub·A_s/γ_M2, "
              "F_b,Rd = k₁·α_b·f_u·d·t/γ_M2")
S_MYSTRAN = ("MYSTRAN (open-source Nastran-dialect solver): the same mesh (coarsened by a factor of 2) and nodal loads solved with an "
             "independent assembly and solver; the displacement field and the reactions are compared – see the cross-check table")
S_FRAME = "Timoshenko beam theory (bending + shear): closed-form member results (S. P. Timoshenko: Strength of Materials)"

# ---------------------------------------------------------------------------------------------------- the table
_E: list[Entry] = []


def _pub(key: str, title: str, domain: str, ref: str, source: str, hide: tuple[str, ...] = (), needs: tuple[str, ...] = ()) -> None:
    _E.append(Entry(key, title, domain, ref, source, True, "", hide, needs))


def _int(key: str, title: str, domain: str, why: str, needs: tuple[str, ...] = ()) -> None:
    _E.append(Entry(key, title, domain, "internal", "", False, why, (), needs))


_EQ = r"rovnováha|nerovnováha|equilibrium"                    # "equilibrium residual of the run" checks (consistency, not a reference)

# ---- statics ------------------------------------------------------------------------------------------------------
_pub("V1", "Cantilever 200×20×10 mm, end load 100 N", "statics", "analytic", S_BEAM)
_pub("V2", "Bar 10×10×100 mm under uniform compression", "statics", "analytic",
     "Uniaxial stress state: σ = F/A, δ = σ·L/E (Hooke's law)")
_pub("V3", "Plate with a Ø10 hole under tension (1/8 model)", "statics", "handbook", S_PETERSON)
_pub("V4", "Thick-walled tube under internal pressure (Lamé)", "statics", "analytic", S_LAME)
_pub("V5", "Column 100×100×1000 mm under its own weight", "statics", "mystran", S_MYSTRAN,
     hide=(r"^reakce Fz$",))                              # the weight = the support reaction is equilibrium; the comparison is the MYSTRAN table
_pub("V6", "Cantilever of V1 split into two bonded parts", "statics", "analytic", S_BEAM,
     hide=(r"sdílené uzly", r"proti V1"))
_pub("V7", "Shaft Ø20×200 mm in torsion, T = 100 N·m (distributing coupling)", "statics", "analytic",
     "Saint-Venant torsion of a circular shaft: τ = 16·T/(π·d³), φ = T·L/(G·J) (S. P. Timoshenko: Strength of Materials)")
_pub("V7R", "Shaft Ø20×200 mm in torsion, T = 100 N·m (rigid coupling)", "statics", "analytic",
     "Saint-Venant torsion of a circular shaft: τ = 16·T/(π·d³), φ = T·L/(G·J) (S. P. Timoshenko: Strength of Materials)")
_pub("V8", "Tube with a radially fixed bore under external pressure", "statics", "analytic",
     S_LAME + " – external pressure, bore fixed radially (ε_θ = 0 at r_i), constants derived in the case",
     hide=(r"^vazba:", r"^tečně volná", r"osově volná\) / Fx"))
_int("V9", "Eye 100×60×10 mm with a Ø20 hole, bearing load (3 load cases)", "statics",
     "checks the load distribution against its own definition (resultant, moment, cosine pressure); no independent reference value")
_pub("V10", "Remote supports on a cantilever, simply supported beam with a hinge", "statics", "analytic", S_BEAM,
     hide=(_EQ, r"posuv \+ natočení", r"\|Fx, Fy, Mx, Mz\|", r"průhyb = vetknutí", r"^\(b\) průhyb = \(a\)",
           r"^\(b'\) průhyb = \(a'\)", r"reakce výběru", r"^\([ab]'?\) [^:]*: reakce Fz v bodě$"))
_pub("V11", "Force on an edge and at a point (cantilever of V1)", "statics", "analytic", S_BEAM,
     hide=(_EQ, r"^síť:", r"uzlové síly bodů"))
_pub("V12", "Local and cylindrical coordinate systems (shaft of V7)", "statics", "analytic",
     "Saint-Venant torsion: φ = T·L/(G·J) (S. P. Timoshenko: Strength of Materials); statics of the tangential load",
     hide=(_EQ, r"výslednice tečné síly", r"radiální síla = tlak", r"kartézský místní systém"))
_pub("V13", "Variable pressure: hydrostatics with a free surface, linear pressure", "statics", "analytic",
     "Hydrostatic load q = ρ·g·b·h on a cantilever strip: Timoshenko beam theory (S. P. Timoshenko: Strength of Materials); "
     "resultant and moment of the pressure by integration",
     hide=(_EQ, r"proměnný \(\*CLOAD\)"))
_pub("V14", "Rotation: spinning disc, angular acceleration of a bar, selected parts only", "statics", "analytic",
     "Rotating disc with a bore (Lamé / Timoshenko & Goodier: Theory of Elasticity); rotating bar: d'Alembert load ρ·A·α·x, "
     "Timoshenko beam theory",
     hide=(_EQ, r"^\(d\) všechny díly", r"jen díl Tyc"))
_pub("V15", "Elastic support: block on a foundation, beam on a Winkler foundation", "statics", "literature",
     "M. Hetényi: Beams on Elastic Foundation – w₀ = P·λ/(2·k_b), M(x) = P/(4λ)·e^(−λx)·(cos λx − sin λx), λ = (k_b/(4EI))^¼",
     hide=(r"rovnováha",))
_pub("V16", "Orthotropic material: rotated axes in tension, cylindrical orthotropy (ring)", "statics", "literature",
     "S. G. Lekhnitskii: Theory of Elasticity of an Anisotropic Body (cylindrically orthotropic ring; transformation of the "
     "compliance to rotated axes)")
_pub("V17", "Separately meshed part joined by *TIE (split cantilever, non-conforming mesh)", "statics", "analytic", S_BEAM, hide=(r"^plocha \*TIE", r"^reakce Fz$",))
_pub("V18", "Results along a line and stress linearization (Lamé tube, beam)", "statics", "analytic",
     S_LAME + "; membrane + bending decomposition of the stress over the wall (integrals evaluated numerically in the case)")
_pub("CS1", "Local cylindrical coordinate system: Lamé stresses in a tube", "statics", "analytic", S_LAME)
_pub("AS1", "Stress tool: Tresca and Mohr-Coulomb safety in torsion of V7", "statics", "analytic",
     "Pure shear: τ = 16·T/(π·d³); safety = R_e/(√3·τ) (von Mises), R_e/(2·τ) (Tresca), Mohr-Coulomb 1/(τ/R_m + τ/R_c)")
_pub("FB1", "Free body cut: section forces and moments (cantilever of V1, torsion of V7)", "statics", "analytic",
     S_STATICS + " – V_z = −F, M_y = F·(L − x), T = applied torque")
_pub("IM1", "Face split: circular imprint r = 10 mm under 1 MPa", "statics", "analytic",
     "Geometry (area π·r²), statics (reaction = p·A), cantilever strip with shear for the deflection estimate (10 % tolerance)", hide=(r"^IM1 reakce Fz", r"^IM1 reakcni moment",))
_pub("IM2", "Polygon imprint (L shape): area, reaction and centroid of the 1 MPa pressure", "statics", "analytic",
     "Polygon area and centroid by the shoelace formulas; statics (reaction = p·A, moment about the pressure centroid)", hide=(r"^IM2 reaction Fz",))
_pub("PS1", "Parametric study: cantilever of V1, load × Young's modulus and a linear scan", "statics", "analytic", S_BEAM,
     hide=(r"scaled point", r"solved once", r"all grid points converged", r"mass of the beam"))
_pub("SM1", "Submodel: plate with a hole in tension (Peterson / Heywood K_t)", "statics", "handbook", S_PETERSON,
     hide=(r"kontrola rezu",))
_int("SM2", "Submodel of a study: plate with a hole, tension + pressure + hole pressure + weight", "statics",
     "compares the submodel with a fine model of the same program (self-consistency)")
_int("LC1", "Load combinations and envelopes: superposition vs direct run, EN 1990", "statics",
     "superposition compared with a direct run of the same program, envelope checked against numpy of the same fields")
_int("TO1", "Cantilever 120×40×10 mm: BESO topology optimisation with feedback to FEA", "statics",
     "heuristic optimiser: bounds and bookkeeping only, no reference solution")

# ---- heat ---------------------------------------------------------------------------------------------------------
_HB = (r"tepelná bilance|heat balance",)
_pub("T1", "Steel + aluminium bar: convection, flux and power (series resistances)", "heat", "analytic",
     "Steady 1-D conduction through series thermal resistances, Fourier's law (" + S_HEAT + ")")
_pub("T2", "Tube r_i 20 / r_o 40 mm: radial conduction (logarithmic profile)", "heat", "analytic",
     "Steady radial conduction in a hollow cylinder, T(r) logarithmic (" + S_HEAT + ")", hide=_HB)
_pub("T3", "Bar with an internal heat source of 10 W (parabolic profile)", "heat", "analytic",
     "1-D conduction with uniform volumetric heat generation, T_max = T₀ + q‴·L²/(8·k) (" + S_HEAT + ")", hide=(r"^zdroj \(integrace\)",))
_pub("T4", "Thermal stress: free and restrained expansion of a bar, ΔT = 100 K", "heat", "analytic",
     "Thermal expansion: ΔL = α·ΔT·L free, σ = −E·α·ΔT fully restrained",
     hide=(r"teplota uložena", r"stav bez ΔT"))
_pub("T5", "Thermal stress in a tube from the temperature field of T2 (plane strain)", "heat", "literature",
     "Thermal stresses in a long hollow cylinder, plane strain (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity, thermal stress)",
     hide=(_EQ,))
_pub("T6", "Transient conduction in a bar after a temperature step (Fourier series)", "heat", "analytic",
     "Fourier-series solution of the 1-D transient heat equation (e.g. H. S. Carslaw, J. C. Jaeger: Conduction of Heat in Solids)",
     hide=(r"počet snímků", r"snímek t = 0", r"interpolace mezi", r"čas posledního snímku"))
_pub("T7", "Radiation from the end face of a bar to the surroundings (nonlinear)", "heat", "analytic",
     "Energy balance at the end face: conduction = ε·σ_SB·(T⁴ − T_amb⁴), solved for the face temperature (Stefan-Boltzmann law, " + S_HEAT + ")",
     hide=_HB)
_pub("TH8", "Two plates in series with a contact resistance (ΔT = q / h_c)", "heat", "handbook",
     "Thermal contact conductance in series with the plate resistances (" + S_HEAT + ")", hide=_HB)
_int("T8", "Two plates in series with a contact resistance (alias of TH8)", "heat", "duplicate of TH8 (same evaluation)")
_pub("TH8B", "Bonded interface with contact conductance (separately meshed plates)", "heat", "handbook",
     "Thermal contact conductance in series with the plate resistances (" + S_HEAT + ")", hide=_HB)
_pub("TH9", "Radiation between two parallel surfaces (cavity radiation)", "heat", "analytic",
     "Net radiation between large parallel plates: q = σ_SB·(T₁⁴ − T₂⁴)/(1/ε₁ + 1/ε₂ − 1) (" + S_HEAT + ")", hide=_HB)
_int("T9", "Radiation between two parallel surfaces (alias of TH9)", "heat", "duplicate of TH9 (same evaluation)")
_pub("TH9C", "Cavity radiation with two different emissivities", "heat", "analytic",
     "Net radiation between large parallel plates: q = σ_SB·(T₁⁴ − T₂⁴)/(1/ε₁ + 1/ε₂ − 1) (" + S_HEAT + ")", hide=_HB)
_pub("KT1", "Heat conduction across a contact (separately meshed blocks, *TIE in the thermal problem)", "heat", "analytic",
     "Fourier's law across a perfectly conducting interface: Q = k·A·ΔT/L (" + S_HEAT + ")")
_pub("KT1c", "Heat conduction across an open conforming gap (equation on DOF 11)", "heat", "analytic",
     "Fourier's law across a perfectly conducting interface: Q = k·A·ΔT/L (" + S_HEAT + ")")
_pub("HT1", "Heat-transfer calculator: Churchill-Chu and Dittus-Boelter / Gnielinski", "heat", "handbook",
     "Churchill-Chu natural convection on a vertical plate, Dittus-Boelter forced convection in a pipe, air / water properties from "
     "the tables of " + S_HEAT,
     hide=(r"vodorovny valec",))

# ---- dynamics, buckling -------------------------------------------------------------------------------------------
_pub("M1", "Natural frequencies of a clamped cantilever (Euler-Bernoulli, effective mass)", "dynamics", "literature",
     "Euler-Bernoulli cantilever: f = (βL)²/(2π·L²)·√(E·I/(ρ·A)), βL = 1.875104; effective mass of the first mode (" + S_BLEVINS + ")")
_pub("M2", "Natural frequencies of a free beam (6 rigid-body modes)", "dynamics", "literature",
     "Euler-Bernoulli free-free beam: βL = 4.730041 (" + S_BLEVINS + ")", hide=(r"^celková hmota",))
_pub("B1", "Linear buckling of a cantilever column loaded at its end (Euler)", "dynamics", "analytic",
     "Euler critical load of a cantilever column: P_cr = π²·E·I/(4·L²)")
_pub("Y1", "Single-DOF oscillator under a harmonic force (|H|, phase)", "dynamics", "analytic",
     "|H(r)| = 1/√((1 − r²)² + (2ζr)²), phase −90° at resonance (" + S_SDOF + ")")
_pub("Y1s", "Single-DOF oscillator with structural damping η", "dynamics", "analytic",
     "Hysteretic (structural) damping: resonance amplification 1/η (" + S_SDOF + ")")
_pub("Y2", "Cantilever under a harmonic force: Euler-Bernoulli frequency and resonance peak", "dynamics", "literature",
     "Euler-Bernoulli cantilever, first mode (" + S_BLEVINS + "); resonance peak of the first mode relative to the static deflection",
     hide=(r"Python × ccx", r"normování", r"rovnováha modálních"))
_pub("Y3", "Step force on a single-DOF oscillator (dynamic load factor 2)", "dynamics", "analytic",
     "Undamped step response: dynamic load factor 2 (" + S_SDOF + ")")
_pub("Y3d", "Step force on a damped single-DOF oscillator", "dynamics", "analytic",
     "Damped step response: DLF = 1 + exp(−π·ζ/√(1 − ζ²)) (" + S_SDOF + ")")
_int("Y3b", "Cantilever under a step force: Python × ccx *MODAL DYNAMIC", "dynamics",
     "the reference is an earlier CalculiX run (1.9456), not a published solution")
_pub("Y6", "Free decay: logarithmic decrement", "dynamics", "analytic",
     "Logarithmic decrement δ = ln(u_i/u_i+n)/n, ζ = δ/√(4π² + δ²) (" + S_SDOF + ")")
_pub("Y10", "Axis start-up: residual amplitude after a rectangular pulse", "dynamics", "analytic",
     "Response of a single-DOF oscillator to a rectangular base-acceleration pulse: residual amplitude (2a/ω²)·|sin(ω·τ/2)| (" + S_SDOF + ")")
_int("Y13", "Base displacement: Python × ccx *BASE MOTION", "dynamics",
     "compares the program's modal solver with a CalculiX run of the same model (cross-check, no published reference)")
_pub("Y15", "Cantilever with a point mass at the tip (Rayleigh)", "dynamics", "literature",
     "Rayleigh's method: f₁ = (1/2π)·√(3·E·I/((M + 33/140·m)·L³)) (" + S_SDOF + ")",
     hide=(r"normování", r"Γ vlastní integrací", r"^hmotnost modelu"))
_pub("Y16", "Base excitation through a rigid remote coupling: first frequency", "dynamics", "literature",
     "Euler-Bernoulli cantilever, first mode (" + S_BLEVINS + ")", hide=(r"špička relativního posuvu hrotu > 0",))
_int("Y16s", "Symmetry plane in the excited direction: warning", "dynamics", "tests a warning message, not a numerical result")
_pub("Y4", "Half-sine shock: shock response spectrum of a single-DOF oscillator", "dynamics", "analytic",
     "Shock response spectrum (maximax absolute acceleration) of a damped single-degree-of-freedom oscillator under a half-sine "
     "base-acceleration pulse, Duhamel integral: S_a/A = 1.6516 at f_n·τ = 0.82, ζ = 5 % (Q = 10); printed example with the same "
     "definition: T. Irvine, An Introduction to the Shock Response Spectrum, Rev. S (2012), Fig. 5 (11 ms, 50 G pulse, Q = 10: "
     "55 G at 30 Hz, 82 G at 80 Hz, 70 G at 140 Hz). The case also runs the SRSS combination with the missing mass and the "
     "transient response; those two checks are internal consistency checks against the same spectrum and are not listed",
     hide=(r"^posuv SRS", r"^přechodová odezva"))
_pub("Y5", "Random vibration of a single-DOF oscillator (Miles' equation)", "dynamics", "literature",
     "Miles' equation: g_rms = √(π/2·f_n·Q·ASD(f_n)) for white-noise base acceleration (" + S_SDOF + ")")
_int("Y14", "Sine sweep: fatigue damage dn/df", "dynamics", "the check is only 'damage ≥ 0'; no reference value")
_pub("Y9", "Prestressed modal analysis: tension 0.5·N_cr", "dynamics", "analytic",
     "Natural frequency of a simply supported beam under axial force N (tension positive): f(N) = f(0)·√(1 + N/N_cr), N_cr = π²EI/L²",
     hide=(r"normování",))
_pub("Y9c", "Prestressed modal analysis: compression 0.5·N_cr", "dynamics", "analytic",
     "Natural frequency of a simply supported beam under axial force N (tension positive): f(N) = f(0)·√(1 + N/N_cr), N_cr = π²EI/L²",
     hide=(r"normování",))
_pub("Y7", "Rotating blade: Campbell diagram, Southwell coefficient", "dynamics", "literature",
     "Southwell's relation for a rotating blade ω² = ω₀² + K·Ω², K = ∫ρA·x·[∫(w')²dx]dx / ∫ρA·w²dx → 1.1933 for the first flapwise "
     "mode of a uniform cantilever with zero hub radius (L. N. Virgin: Vibration of Axially-Loaded Structures, lecture notes Part C, "
     "Duke University, 2013, slides 58-61: the definition and the tabulated frequencies ω₁ = 3.5160, 3.6816, 4.7973 at Ω = 0, 1, 3 in "
     "units of √(EI/(ρA·L⁴))); the reference of the case, K = 1.1812 at Ω/ω₀ = 0.957, is computed by an independent 1-D "
     "rotating-beam script (Hermite beam elements with the centrifugal tension as geometric stiffness, 160 elements) that reproduces "
     "the table of those lecture notes at η = 0, 1, 3")

# ---- contacts -----------------------------------------------------------------------------------------------------
_pub("K1", "Contact patch test: separately meshed blocks", "contacts", "analytic",
     "Uniform pressure through a flat interface: p = F/A, u = p·h/E (+ p/K of the contact stiffness)", hide=(_EQ, r"^normálová síla páru",))
_pub("K1c", "Contact patch test: conforming mesh, open gap", "contacts", "analytic",
     "Uniform pressure through a flat interface: p = F/A, u = p·h/E (+ p/K of the contact stiffness)", hide=(_EQ, r"^normálová síla páru",))
_pub("K4", "Press fit (Lamé)", "contacts", "analytic", S_PRESSFIT)
_pub("K7", "Cantilever over a rigid cylinder with a gap", "contacts", "analytic",
     "Euler-Bernoulli cantilever with a gap-limited support: compatibility of the end deflection, R = (F·a²·(3L − a)/(6EI) − gap)/(a³/(3EI))",
     hide=(_EQ, r"informativně"))
_pub("K7n", "Cantilever over a rigid cylinder with a gap, NLGEOM", "contacts", "analytic",
     "Euler-Bernoulli cantilever with a gap-limited support: compatibility of the end deflection, R = (F·a²·(3L − a)/(6EI) − gap)/(a³/(3EI))",
     hide=(_EQ, r"informativně"))
_pub("R1", "Friction plateau under a controlled displacement", "contacts", "analytic",
     "Coulomb friction: T = μ·N")
_pub("G1", "Compression-only support (GAPUNI)", "contacts", "analytic",
     "Statics of a compression-only support: reaction = p·A, u = p·h/E (+ p/K)", hide=(_EQ,))
_int("K8", "Initial penetration of a frictionless pair: error before the solver", "contacts",
     "tests an error message, not a numerical result")
_pub("K2", "Hertz contact: sphere on a plate (mortar)", "contacts", "literature", S_HERTZ, needs=("mortar",))
_int("K6", "Bolted joint M10 with a contact interface (preload, sticking, slip)", "contacts",
     "the preload reference F_V is computed by the program itself; the friction check is Ft = μ·Fn of the same run")
_pub("KK4", "Impact of a sphere on a rigid plate (*DYNAMIC + contact)", "contacts", "literature",
     S_HERTZ + " – maximum contact force and contact duration of the Hertz impact")
_pub("KK2", "Snap-fit hook: NLGEOM, friction, prescribed motion", "contacts", "handbook", S_SNAPFIT,
     hide=(r"křivka síla–dráha", r"mezi nosníkem a deskou"))     # the last one is a 0/1 bracket check

# ---- joints -------------------------------------------------------------------------------------------------------
_int("S1", "Bolt M10 with a 1 mm gap (bolt only): tension and shear", "joints",
     "the reference is the applied load itself (F_SA = F, Q = F): equilibrium of the bolt, not an independent value; the one other "
     "check (elongation of the beam) is a bookkeeping comparison with the program's own sum of two compliances")
_pub("S2", "Lap joint M10 8.8: shear, preload, VDI 2230 and EN 1993-1-8", "joints", "handbook",
     "VDI 2230 Part 1 (permissible assembly preload F_M,zul); " + S_EC3_BOLT + "; statics (shear transfer, weight)",
     hide=(r"^předpětí:",))
_pub("S4", "Virtual bolt from a CAD assembly (parts not loaded): same as S2", "joints", "handbook",
     "VDI 2230 Part 1 (permissible assembly preload F_M,zul); " + S_EC3_BOLT + "; statics (shear transfer, weight)",
     hide=(r"^předpětí:",))
_pub("S3", "Two bolts M12: load distribution in a bolt group, bearing with pitch", "joints", "handbook",
     S_EC3_BOLT + "; equal share F/n of a bolt group")
_pub("W1", "T-joint with fillet welds a = 5: tension and longitudinal shear", "joints", "handbook",
     "EN 1993-1-8 (Eurocode 3, design of joints): fillet weld throat a, longitudinal shear stress τ = F/(2·a·L); statics of the weld forces",
     hide=(r"únava svaru", r"účinná výška", r"délka svaru"))
_pub("W2", "Weld toe hot-spot stress (IIW 0.4t / 1.0t) on a bent plate", "joints", "handbook",
     "IIW Recommendations for Fatigue Design of Welded Joints and Components: hot-spot extrapolation 0.4t / 1.0t; beam theory "
     "σ = 6·F·(L − x)/(b·t²) at the weld toe",
     hide=(r"tloušťka plechu", r"únava paty"))

# ---- fatigue ------------------------------------------------------------------------------------------------------
_pub("U1", "Fatigue: cantilever of V1, ±100 N, 1000 cycles per block, 100 blocks", "fatigue", "analytic",
     "Bending stress σ = M/W; Palmgren-Miner linear damage D = Σ n/N with a Basquin S-N curve N = N_D·(S_D/σ_a)^k", needs=("accel",))
_pub("U2", "Fatigue: two unit load states Fz + Fy in phase (superposition)", "fatigue", "analytic",
     "Bending stress σ = M/W summed over the two load states; Palmgren-Miner linear damage with a Basquin S-N curve",
     hide=(r"superpozice: amplituda",), needs=("accel",))
_pub("U3", "Fatigue, critical plane: tension / torsion / both 90° out of phase", "fatigue", "literature",
     "Findley critical-plane criterion with hand evaluation of the plane search (tension: σ = F/A, torsion: τ = 16·T/(π·d³), "
     "90° out of phase: maximum over the plane angle); Palmgren-Miner damage",
     hide=(r"numpy kontrola", r"σn krit\. rovina / max\. hlavní", r"Findley σekv = σa \(uzel"), needs=("accel",))
_pub("U4", "Strain-life fatigue: notched bar (Neuber, Coffin-Manson)", "fatigue", "literature",
     "Neuber's rule σ·ε = σ_el²/E, Glinka's energy-density rule, Ramberg-Osgood cyclic curve, Coffin-Manson-Basquin strain-life relation with "
     "Morrow and Smith-Watson-Topper mean-stress corrections; K_t of a plate with a hole (Peterson / Heywood)", needs=("accel",))

# ---- printing -----------------------------------------------------------------------------------------------------
_pub("P1", "Printed bars upright / flat: orthotropy and Tsai-Hill", "printing", "analytic",
     "Hooke's law of a transversely isotropic bar (u = σ·L/E_z, u = σ·L/E_xy, Poisson contraction ν₁₃); Tsai-Hill failure criterion "
     "for stress along a material axis (safety = strength/σ)", hide=(r"^reakce Fx",))
_int("P2", "Printed prism with shell and 20 % infill: tension + weight", "printing",
     "the reference is built from the program's own homogenised printing constants (rule of mixtures of the program)")
_int("P3", "Printed perimeters: along / across the line, mix with infill", "printing",
     "the reference is built from the program's own homogenised printing constants (rule of mixtures of the program)")
_int("P4", "Printing: creep – isochronous modulus and long-term strength (1000 h, 30 °C)", "printing",
     "the creep factors come from the same table the solver uses (wiring check; there is no physical reference for PLA creep)")
_int("P5", "Printing: fatigue – S-N curve of the print and Hill equivalent", "printing",
     "the S-N parameters of the print material come from the program's own table (wiring check)")
_int("P6", "Printing: thin places with solid infill, perimeter × infill sweep", "printing",
     "monotonicity and Pareto-front checks of a feature; no reference value")
_int("P7", "Printing tolerance: layer δ without material", "printing",
     "compares two bars of the same model with each other (self-consistency)")
_int("P8", "Printing: perimeters not along the axes (monoclinic groups) – ccx × own assembly", "printing",
     "compares CalculiX (ANISO) with the program's own assembly (internal cross-check)")

# ---- beams and frames ---------------------------------------------------------------------------------------------
_pub("F1", "Beams: IPE 200 cantilever (force, torsion with J, distributed load) – exact Timoshenko, 1 element", "beams", "analytic",
     S_FRAME, needs=("beams",))
_pub("F2", "Beams: simply supported HEB 200 under a distributed load", "beams", "analytic", S_FRAME, needs=("beams",))
_pub("F3", "Beams: portal frame (slope-deflection)", "beams", "analytic",
     "Slope-deflection method of structural analysis for a portal frame with axially rigid members: M_ij = 2EI/L·(2θ_i + θ_j − 3ψ) "
     "+ fixed-end moment, joint equilibrium ΣM = 0 (hand solution of the case)",
     hide=(r"zbytek rovnováhy",), needs=("beams",))
_int("F4", "Beams: quarter-circle arch (Castigliano)", "beams",
     "the reference is fea_core.beams.analytic.arc_cantilever, the program's own Gauss quadrature of Castigliano's integrals, not a "
     "closed form written in the case; published once the closed form (Castigliano, bending + axial + shear) is in the case itself",
     ("beams",))
_pub("F5", "Beams: two-bar truss (axial forces)", "beams", "analytic",
     "Method of joints for a two-bar truss; δ = N·L/(E·A)", needs=("beams",))
_pub("F6", "Beams: Gerber beam with a hinge", "beams", "analytic",
     "Statics of a Gerber beam: the hinge transmits no moment (M = 0 at the hinge), the reactions follow from ΣF = 0, ΣM = 0", needs=("beams",))
_pub("F7", "Beams: natural frequencies of a simply supported beam (Timoshenko, torsion)", "beams", "literature",
     "Timoshenko beam frequencies with shear deformation and rotary inertia; torsional frequency f = 1/(4L)·√(G·J/(ρ·I_p)) (" + S_BLEVINS + ")",
     needs=("beams",))
_pub("F8", "Beams + solids: a member rigidly attached to the face of a cantilever solid", "beams", "analytic",
     "Euler-Bernoulli cantilever with shear (" + S_FRAME + "); statics of the connection", needs=("beams",))
_pub("F9", "Beams: linear buckling (Euler + Engesser shear correction)", "beams", "literature",
     "Euler critical load P_E = π²·E·I/(β·L)² with Engesser's shear correction P_cr = P_E/(1 + P_E/(G·A_s))", needs=("beams",), hide=(r"^statika: tlak = reakce",))
_pub("F9M", "Beams: buckling – cross-check against MYSTRAN (SOL 105)", "beams", "mystran",
     "MYSTRAN (SOL 105 eigenvalue buckling, CBAR elements), open-source Nastran dialect solver", needs=("beams", "mystran"))
_pub("F3M", "Beams: cross-check of the frame F3 against MYSTRAN (CBAR)", "beams", "mystran",
     "MYSTRAN (SOL 101, CBAR elements), open-source Nastran dialect solver", needs=("beams", "mystran"))
_pub("F10", "Beams: accuracy of 1 element, rolled section, rotational spring, independence of subdivision", "beams", "analytic",
     S_FRAME, needs=("beams",), hide=(r"prvků: průhyb konce = 1 prvek",))
_int("F11", "Beams: spatial frame (hinges, truss members, spring) – node equilibrium", "beams",
     "node equilibrium of the member end forces of the program's own result (self-consistency)", needs=("beams",))
_pub("F11M", "Beams: cross-check of the spatial frame against MYSTRAN (CBAR)", "beams", "mystran",
     "MYSTRAN (SOL 101, CBAR elements), open-source Nastran dialect solver", needs=("beams", "mystran"))
_pub("F12", "Beams: torsional and bending frequencies of a cantilever, convergence", "beams", "literature",
     "Torsional frequency f = 1/(4L)·√(G·J/(ρ·I_p)); Euler-Bernoulli cantilever βL = 1.875104 (" + S_BLEVINS + ")",
     hide=(r"řád konvergence", r"8 prvků = 64 prvků"), needs=("beams",))
_pub("F13", "Beams: linear buckling – Euler (2-8 elements) and Engesser", "beams", "literature",
     "Euler critical loads for pinned-pinned, clamped-free and pinned-clamped columns (λ_cr = 20.1907·EI/L²); Engesser shear correction",
     needs=("beams",))
_pub("F14", "Beams: temperature – restrained expansion, gradient, free bar", "beams", "analytic",
     "Restrained bar N = −E·A·α·ΔT; temperature gradient: M = −E·I·α·ΔT_g/h, simply supported deflection α·ΔT_g·L²/(8h)",
     needs=("beams",))
_int("F11A", "Beams: spatial frame without the spring (variant for the cross-check)", "beams",
     "node equilibrium of the member end forces of the program's own result (self-consistency)", needs=("beams",))

# ---- 2D sections and shells ---------------------------------------------------------------------------------------
_pub("A1", "Tube r_i 20 / r_o 40, p = 100 MPa as an axisymmetric problem", "shells_2d", "analytic", S_LAME,
     hide=(r"proti 3D",))
_pub("A2", "Thick-walled sphere r_i 20 / r_o 40, p = 100 MPa (axisymmetric)", "shells_2d", "analytic",
     "Lamé solution of a thick-walled sphere under internal pressure (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity)")
_pub("A3", "Plate with a Ø10 hole under tension (plane stress)", "shells_2d", "handbook", S_PETERSON)
_pub("A4", "Thick-walled cylinder, p = 100 MPa (plane strain)", "shells_2d", "analytic", S_LAME + " (plane strain)")
_pub("A5", "Rotating disc R 100, t 4, ω = 1000 rad/s (axisymmetric)", "shells_2d", "analytic",
     "Stresses of a rotating solid thin disc (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity)")
_pub("A5P", "Rotating quarter disc R 100, t 4, ω = 1000 rad/s (plane stress)", "shells_2d", "analytic",
     "Stresses of a rotating solid thin disc (S. P. Timoshenko, J. N. Goodier: Theory of Elasticity); section force ρ·ω²·t·R³/3")
_pub("Sh1", "Shell: cantilever of a 200×40×5 plate, end force", "shells_2d", "analytic", S_BEAM,
     hide=(_EQ, r"skořepina / 3D", r"^reakce Fz$", r"^reakce My", r"^tíha: reakce Fz"))
_pub("Sh2", "Shell: square plate simply supported on its edge, uniform pressure", "shells_2d", "literature",
     "S. Timoshenko, S. Woinowsky-Krieger: Theory of Plates and Shells – w_max = 0.00406·q·a⁴/D, M = 0.0479·q·a²",
     hide=(_EQ,))
_pub("Sh3", "Shell: cylinder under internal pressure (σ_θ = p·R/t)", "shells_2d", "analytic",
     S_LAME + " (thin-wall limit σ_θ = p·R/t)", hide=(_EQ, r"poloměr střední plochy", r"rozptyl"))
_int("Sh4", "Shell: plate with a hole against a 3D model", "shells_2d", "compares the shell model with a 3D model of the same program")
_int("Sh5", "Shell: T-beam of two plates against a 3D model", "shells_2d", "compares the shell model with a 3D model of the same program")
_int("Sh6", "Shell: sheet with a 90° bend against a 3D model", "shells_2d", "compares the shell model with a 3D model of the same program")
_int("Sh7", "Shell: surface part from STEP (given thickness, offset)", "shells_2d",
     "reference deflection is a result of the program's own 3D model (2.9887 mm)")

# ---- plasticity and nonlinear buckling ----------------------------------------------------------------------------
_pub("PL1", "Plasticity: tension of a bar with bilinear hardening (BISO)", "nonlinear", "analytic",
     "Uniaxial stress state of a bilinear elastic-plastic material: σ = R_e + E_t·(ε − R_e/E), ε_p = ε − σ/E")
_pub("PL2", "Plasticity: bending of a cantilever into the plastic range and unloading", "nonlinear", "analytic",
     "Elastic-plastic bending of a rectangular section with bilinear hardening: moment-curvature relation M(κ) and residual stress "
     "σ_res = σ(κ) − E·κ·y after elastic unloading (standard plasticity theory, closed form evaluated in the case)")
_int("PL3", "Plasticity: Ramberg-Osgood and Hollomon hardening, point by point", "nonlinear",
     "the reference curve is generated by the program's own table generator (checks that the solver reproduces its input)")
_pub("PL4", "Plasticity: tension-compression cycle – kinematic vs isotropic hardening", "nonlinear", "analytic",
     "Uniaxial cyclic response of a bilinear material with kinematic (Bauschinger effect: reverse yielding at σ_max − 2·R_e) and "
     "isotropic hardening (reverse yielding at −σ_max), linear hardening H = E·E_t/(E − E_t)")
_int("PL5", "Plasticity assessment: uniaxial strain, EN 1993 / ASME VIII-2, residual stress", "nonlinear",
     "ASME VIII-2 Table 5.7 constants (m2, α_sl) were typed from memory; no public source that prints them could be read "
     "(ASME is paid; secondary sources searched 25. 11. 2026 only quote the formula) – to be confirmed against the standard")
_pub("BK1", "Nonlinear buckling: elastic cantilever with a mode-shaped imperfection", "nonlinear", "analytic",
     "Euler load P_cr = π²·E·I/(4·L²); amplification of an imperfection 1/(1 − N/N_cr), tip deflection e₀·r/(1 − r)",
     hide=(r"ramp completed", r"imperfection amplitude"))
_pub("BK2", "GNIA + first yield of a pin-ended column = EN 1993-1-1 buckling curve c", "nonlinear", "handbook",
     "EN 1993-1-1 (Eurocode 3): flexural buckling curve c, Ayrton-Perry formulation, reduction factor χ")
_int("BK3", "GMNIA of a pin-ended column vs the column deflection curve (fibre model)", "nonlinear",
     "the reference is a one-dimensional numerical fibre model written for the case; no published solution")
_pub("CY1", "Cyclic symmetry: statics of a sector, spinning disc (Lamé)", "nonlinear", "analytic",
     "Rotating disc with a bore (Lamé, plane stress; S. P. Timoshenko, J. N. Goodier: Theory of Elasticity); pressure × area",
     hide=(r"U sektor", r"σ sektor", r"max σ_vM sektor", r"rovnováha sektoru", r"360° = 6 × sektor"))
_pub("CY2", "Cyclic symmetry: natural frequencies by nodal diameter, ring", "nonlinear", "literature",
     "In-plane bending vibration of a thin ring: f = 1/(2π)·√(E·I/(ρ·A·R⁴))·n·(n² − 1)/√(n² + 1) (Love; " + S_BLEVINS + ")",
     hide=(r"počet frekvencí", r"cyklické frekvence"))

# ---- multibody ----------------------------------------------------------------------------------------------------
_pub("MB1", "Slider-crank r 20, l 80, ω 100 rad/s: kinematics and drive torque", "multibody", "analytic",
     "Closed-form kinematics of a slider-crank (position, velocity, acceleration of the piston); drive torque from the power balance M = F·v/ω",
     hide=(r"rovnováha těles", r"výkonová bilance P", r"zadání", r"čas výstupu"), needs=("motion",))
_pub("MB2", "Four-bar linkage 40/120/80/100: Freudenstein, both branches", "multibody", "literature",
     "F. Freudenstein: displacement equation of the planar four-bar linkage (rocker angle θ₄ from the crank angle θ₂, open and crossed branch)",
     hide=(r"rovnováha těles", r"výkonová bilance P"), needs=("motion",))
_pub("MB3", "Pendulum L 0.5 m, θ₀ 60°: period, pin force, energy", "multibody", "analytic",
     "Period of a physical pendulum with the complete elliptic integral T = 4·√(L_eq/g)·K(sin²(θ₀/2)); pin force m·g·(3 − 2cos θ₀); energy conservation",
     hide=(r"rovnováha těles", r"výkonová bilance P", r"proti zadání"), needs=("motion",))
_pub("MB4", "Cylinder rolling on a 30° plane (rolling constraint)", "multibody", "analytic",
     "Rolling cylinder on an incline: a = (2/3)·g·sin α, friction force m·g·sin α/3, minimum friction coefficient tan α/3",
     hide=(r"rovnováha těles", r"výkonová bilance P", r"zadání"), needs=("motion",))
_pub("MB5", "Block sliding to rest, v₀ 1 m/s, μ 0.2", "multibody", "analytic",
     "Coulomb friction deceleration a = μ·g, stopping time v₀/(μ·g), stopping distance v₀²/(2·μ·g)",
     hide=(r"rovnováha těles", r"výkonová bilance P"), needs=("motion",))
_pub("MB6", "Stick-slip on a belt (μ_s 0.4, μ_k 0.3)", "multibody", "analytic",
     "Stick-slip of a mass on a moving belt: breakaway x_s = μ_s·m·g/k, oscillation about x_k = μ_k·m·g/k, period of the slip-stick cycle",
     hide=(r"rovnováha těles", r"výkonová bilance P", r"zadání"), needs=("motion",))
_int("MB3b", "Pendulum with pin friction μ·|R|·r", "multibody",
     "the reference is a numerical solution of the equations of motion written for the case (numpy), not a published solution", ("motion",))
_int("MB7", "Crank with pin friction – power balance", "multibody", "energy bookkeeping of the program's own result (self-consistency)",
     ("motion",))
_int("MB8", "Spatial 4R mechanism – mobility and redundant constraints", "multibody",
     "kinematic bookkeeping (Grübler count, Jacobian rank, message), no numerical reference value", ("motion",))
_pub("MB9", "Gear pair and screw joint", "multibody", "analytic",
     "Gear ratio ω₂/ω₁ = z₁/z₂; screw joint travel = lead × turns, torque M = F·P/(2π)",
     hide=(r"rovnováha těles", r"výkonová bilance P", r"zdvih = 8 mm"), needs=("motion",))
_int("MB10", "Flexible arm (Craig-Bampton, FFRF) against ccx and beam theory", "multibody",
     "mixed internal cross-check (Craig-Bampton against ccx); the beam-theory part is not yet reviewed for publication", ("motion",))
_int("MB11", "Crank → FEA at the instant of the maximum reaction (inertia relief)", "multibody",
     "compares FEA with the multibody result of the same program (self-consistency)", ("motion",))
_int("MB12", "Rotating reaction in a connecting-rod eye: superposition of unit states", "multibody",
     "superposition compared with a direct run of the same program", ("motion",))
_int("MB13", "Lid on two coaxial hinges: redundancy, distribution in FEA", "multibody",
     "compares FEA with the multibody result of the same program; redundancy bookkeeping", ("motion",))
_int("MB14", "Crank against a stop: collision detection without contact", "multibody",
     "geometric collision time; not yet reviewed for publication", ("motion",))
_int("MB15", "Start-up by a torque and S-curve (D6)", "multibody",
     "the reference profile is generated by the program's own motion-law module", ("motion",))
_int("MB16", "Pendulum against a stop with restitution (D6)", "multibody",
     "the restitution coefficient is an input of the contact law (wiring check)", ("motion",))
_CHRONO = " Solved by the multibody solver Chrono with contact of the triangle meshes of the CAD parts: the case needs the optional module chrono."
_pub("MB4C", "Cylinder sliding on a 30° plane, μ 0.1 – mesh contact in Chrono", "multibody", "analytic",
     "Coulomb friction of a solid cylinder on an incline of angle α with μ < tan α/3 (the cylinder slides while it turns): "
     "a = g·(sin α − μ·cos α), angular acceleration ε = 2·μ·g·cos α/R (I = m·R²/2). The angular acceleration is checked to 5 % only "
     "(it comes out about 3 % high; the miss is attributed to the polygonal triangulation of the cylinder and to the "
     "Euler time stepping, whose error is proportional to the step)." + _CHRONO,
     hide=(r"rovnováha",), needs=("chrono",))
_pub("MB4R", "Cylinder rolling on a 30° plane, μ 0.3 – mesh contact in Chrono", "multibody", "analytic",
     "Rolling cylinder on an incline: a = (2/3)·g·sin α, friction force m·g·sin α/3, rolling without slip (v = ω·R) while μ ≥ tan α/3 "
     "(0.192 for 30°)." + _CHRONO,
     hide=(r"rovnováha",), needs=("chrono",))
_pub("KK5", "Eccentric cam from CAD in a multibody model – lift of the follower, mesh contact in Chrono", "multibody", "analytic",
     "Closed-form lift of a roller follower on an eccentric disc (cam radius R_c, eccentricity e, roller radius r_f, guide through the axis): "
     "the roller centre stays at the distance L = R_c + r_f from the disc centre, so its height is y = e·cos θ + √(L² − e²·sin² θ) and the lift "
     "from the lowest position s(θ) = e·cos θ + √(L² − e²·sin² θ) − (L − e)." + _CHRONO,
     hide=(r"rovnováha", r"^normálová síla"), needs=("chrono",))

# ---- cases of 1.5.1 that were added on other branches (BESO constraints F15-S3, sketcher F15-S5, space margins F14-S1, space model checks F14-S3):
# classified at the integration as internal (not yet reviewed for publication; a promotion needs the page of docs/verification to be regenerated)
_NYR = "not yet reviewed for publication (added after the public page was generated)"
_int("TO2", "Topology optimisation: symmetry about the mid-plane, mirror images give identical decisions", "statics",
     "bounds and bookkeeping of the program's own optimiser (compliance against a run of the same program); " + _NYR)
_int("TO3", "Topology optimisation: extrusion along z, one decision per column, prism STL", "statics",
     "geometric bookkeeping of the program's own optimiser and STL export; " + _NYR)
_int("TO4", "Topology optimisation: overhang 45 deg of a half MBB beam built upwards", "statics",
     "surface measured on the program's own STL against a run of the same program; " + _NYR)
_int("S1A", "Space margins: MoS = F/(J·σ) − 1 for every structure class and verification approach", "statics",
     "the margin definition with arbitrary test factors, checked against its own formula; " + _NYR)
_int("S1B", "Space margins: a singular clamped edge goes to the justification table", "statics",
     "bookkeeping of the margin table of the program itself; " + _NYR)
_int("S1C", "Space margins: combined mechanical and thermal load, distinct factor handling", "statics",
     "the margin definition with arbitrary test factors, checked against its own formula; " + _NYR)
_int("S1D", "Space margins: a solved bar in tension, MoS outside the clamp", "statics",
     "closed-form bar stress F/A is a public reference, but the margin factors are arbitrary test values; " + _NYR)
_int("SC1", "Space model check 1: mass and centre of mass", "statics",
     "closed-form masses and centre of mass exist; " + _NYR)
_int("SC2", "Space model check 2: free-free modes, six rigid-body modes and the first elastic mode of a beam", "dynamics",
     "closed-form free-free beam frequency exists; " + _NYR)
_int("SC3", "Space model check 3: unit enforced displacement, reaction E·A·δ/L", "statics",
     "closed-form bar stiffness exists; " + _NYR)
_int("SC4", "Space model check 4: isothermal expansion of a bar held at both ends, force E·α·ΔT·A", "heat",
     "closed-form thermal force exists; " + _NYR)
_int("SC5", "Space model check 5: effective mass of an axial rod", "dynamics",
     "closed-form modal effective mass exists; " + _NYR)
_int("SC6", "Space model check 6: mesh convergence of the first frequency of a cantilever", "dynamics",
     "closed-form Euler-Bernoulli frequency exists; " + _NYR)
_int("SC7", "Space model check 7: equilibrium, reaction = load", "statics",
     "equilibrium bookkeeping of the run itself (self-consistency)")
_int("S2A", "Space loads: mass, centre of mass and Steiner-shifted inertia of a block with a point mass and a surface mass", "statics",
     "closed-form block inertia and the parallel-axis theorem exist; " + _NYR)
_int("S2B", "Space loads: Miles' equation and the resonant sine amplitude of a 1-DOF system through the random and sine chains", "dynamics",
     "Miles' equation is a published relation, but the run uses a synthetic modal basis; " + _NYR)
_int("S2C", "Space loads: bookkeeping of the 3-axis envelope (maximum of three single-axis calls on a synthetic basis)", "dynamics",
     "compares two calls of the same program on identical synthetic numbers (bookkeeping of the envelope)")
_int("S2D", "Space loads: minimum natural frequency check and budget verdicts", "dynamics",
     "the check logic with arbitrary test limits, compared with its own definition")
_int("S2E", "Space loads: signed quasi-static load cases from the requirement-set factors", "statics",
     "the generator's bookkeeping with arbitrary test factors")
_int("S2F", "Space loads: the 3-axis run over one modal basis against three complete single-axis analyses of a cantilever", "dynamics",
     "compares two chains of the same program (own basis and statics per run, real solver), not an external reference")
_int("S5A", "Space S5: mass-weighted cross-orthogonality COC / AOC and modulus difference of the test correlation on chains with closed-form modes", "dynamics",
     "closed-form modes of a mass-spring chain and the hand formula of the orthogonality; the reduced mass by the program's own SEREP route against its own full-DOF form; " + _NYR)
_int("S4A", "Space S4: interface force of a one-DOF structure, quasi-static limit M a, resonance m a/(2 zeta), notch holds the limit exactly", "dynamics",
     "closed-form one-DOF transmissibility and the definition of the notch; the program's modal interface data against its own closed form; " + _NYR)
_int("S4B", "Space S4: PCB Steinberg check by hand (test inputs)", "statics",
     "hand arithmetic of the Steinberg formula with test inputs and the Rayleigh damage of the program's own fatigue module; " + _NYR)
_int("S4C", "Space S4: bolted joint gapping, slip and differential-CTE preload change by hand (test coefficients)", "statics",
     "hand arithmetic of standard preloaded-joint mechanics with arbitrary test coefficients; " + _NYR)
_int("S4D", "Space S4: interface reaction of a real bar, quasi-static limit M a and moment M a * lever arm", "dynamics",
     "Newton's second law (reaction = total mass x acceleration) against a CalculiX modal basis of the same mesh; " + _NYR)
_int("SK1", "Sketcher: plate with a hole drawn from typed coordinates, plane stress", "shells_2d",
     "compares the drawn model with the STEP-section route of the same program (V3 / A3); " + _NYR)
_int("SK2", "Sketcher: thick-walled tube drawn in the sketcher, axisymmetric", "shells_2d",
     "closed-form Lamé solution exists; " + _NYR)
_int("SK3", "Sketcher: portal frame drawn from typed coordinates", "beams",
     "compares with case F3 of the same program and a slope-deflection solution; " + _NYR)
_int("SK4", "Sketcher: a parameter change keeps constraints, loads and materials", "shells_2d",
     "bookkeeping of the rebuild (wiring check)")
_int("SK5", "Sketcher: error cases name the entity", "shells_2d",
     "messages of the program itself (error texts)")
_int("SK6", "Sketcher: DXF import of a plate (needs the optional module sketch)", "shells_2d",
     "area and volume of the imported contour against the drawing (import check); " + _NYR)

_int("SU1", "Surrogate model: stress of a plate with a hole over hole radius and width, coverage of the 90 % band", "shells_2d",
     "compares the surrogate with 100 new analyses of the same program; the analyses are compared with Kt of Heywood / Peterson, "
     "but the case checks the surrogate; " + _NYR)
_int("SU2", "Surrogate model: tip deflection of a cantilever strip over L, b, h and E, coverage of the 90 % band", "shells_2d",
     "compares the surrogate with 100 new analyses of the same program; the analyses are compared with Timoshenko beam theory, "
     "but the case checks the surrogate; " + _NYR)
_int("SU3", "Surrogate model: load factor with a categorical material, refusal outside the sampled range", "shells_2d",
     "behaviour of the surrogate (refusals, one model per category) against the linear-elastic scaling of the same program, band against 18 fresh analyses; " + _NYR)
_int("SU4", "Surrogate model: a grid design (12 hole radii x 3 widths), refusal between the three widths", "shells_2d",
     "behaviour of the surrogate on a design that is not space-filling (a parameter with three levels is answered at them only); fresh analyses of the same program at the "
     "three widths check the band; " + _NYR)

# ---------------------------------------------------------------------------------------------------- the API
ENTRIES: dict[str, Entry] = {e.key: e for e in _E}
if len(ENTRIES) != len(_E):                                    # pragma: no cover – a typo in the table above
    raise RuntimeError("duplicate key in the verification catalogue")


@dataclass(frozen=True)
class Group:
    domain: str
    title: str
    entries: tuple[Entry, ...] = field(default_factory=tuple)


def get(key: str) -> Entry:
    """Catalogue entry of a case key (KeyError for a case that is not catalogued)."""
    return ENTRIES[key]


def public_keys() -> list[str]:
    """Keys of the published cases in publication order (by domain, then table order)."""
    return [e.key for g in by_domain() for e in g.entries]


def internal_keys() -> list[str]:
    return [k for k, e in ENTRIES.items() if not e.public]


def by_domain(public_only: bool = True) -> list[Group]:
    """Cases grouped by domain in the order of DOMAINS; only the published ones unless `public_only` is False."""
    out = []
    for dom, title in DOMAINS.items():
        es = tuple(e for e in ENTRIES.values() if e.domain == dom and (e.public or not public_only))
        if es:
            out.append(Group(dom, title, es))
    return out


def problems() -> list[str]:
    """Self-check of the table (empty = consistent): published cases need a public reference type and a citation, internal cases a reason."""
    bad = []
    for e in ENTRIES.values():
        if e.domain not in DOMAINS:
            bad.append(f"{e.key}: unknown domain {e.domain!r}")
        if e.ref_type not in REF_TYPES:
            bad.append(f"{e.key}: unknown reference type {e.ref_type!r}")
        if e.public and (e.ref_type not in PUBLIC_TYPES or len(e.source.strip()) < 10):
            bad.append(f"{e.key}: a published case needs a public reference type and a citation")
        elif e.public and not e.source_kind:
            bad.append(f"{e.key}: the citation names no work and states no relation (an equation) - cite a work or write the relation")
        elif e.public and e.ref_type in ("literature", "handbook") and e.source_kind == "relation":
            bad.append(f"{e.key}: a {e.ref_type} reference must cite a work, not only state a relation")
        if not e.public and (e.ref_type != "internal" or not e.why_not.strip()):
            bad.append(f"{e.key}: a non-public case is internal and needs a reason")
        for p in e.hide:
            try:
                re.compile(p)
            except re.error as ex:
                bad.append(f"{e.key}: bad hide pattern {p!r}: {ex}")
    return bad

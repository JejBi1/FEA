"""English wording of the checks of the published verification cases.

The check names of `verification.cases` are Czech (a legacy of the first versions, shown in the console and in the GUI test guide), the
public page is English. `label(case, name)` gives the English label of a check name:

  1. an exact entry of LABELS (the Czech name as the case writes it → English),
  2. a family of names that differ in a number or a tag (PATTERNS, regular expressions with a function),
  3. the name without the case key the old ASCII-only cases put in front of it ("IM2 imprint area …" → "imprint area …"),
  4. otherwise the name unchanged – `is_english` then tells that the page still carries Czech and the report warns about it.

`unit` does the same for the unit strings. Both are data tables: adding a check to a published case means adding its label here
(tests/test_verification_public.py fails on Czech text in the committed page).
"""
from __future__ import annotations

import re
from collections.abc import Callable

# letters of Czech that no author name in the citations (Lamé, Hetényi ...) has
_CZECH_LETTERS = re.compile(r"[ěščřžůďťňýĚŠČŘŽŮĎŤŇÝ]")
# Czech words written without diacritics (the older cases of verification/f9_cases.py and fatigue_cases.py)
_CZECH_ASCII = re.compile(r"\b(vnitrni|vnejsi|pruhyb\w*|reakcni|tezisti|tezist\w*|vetknuti|zatizeni|napeti|bezpecnost|litina|krutu|kroutici|"
                          r"kruhu|otisku|povrch|radialni|poskozeni|soucin\w*|zivotnost|pulenim|nezavisle|spicka|elasticka|svisla|"
                          r"vodorovny|prirozena|nucena|trubce|valec|rez|volneho|siroky|pas|zbytek|rovnovaha|plocha|odpovida|"
                          r"hruby|globalni|jemny|stredni|odchylka|ztrata|zlomek)\b", re.I)
_CASE_PREFIX = re.compile(r"^[A-Z]{1,3}\d+[A-Za-z]*:? (?=\S)")

UNITS: dict[str, str] = {"-": "", "cyklu": "cycles", "N*mm": "N·mm", "N*m": "N·m", "W/(m2*K)": "W/(m²·K)", "mm3": "mm³", "mm2": "mm²"}


# load-case names of the MYSTRAN comparison (verification.mystran_check shows the names the cases give their load cases)
LOADCASES: dict[str, str] = {
    "Sila 100 N": "force 100 N", "Tlak 100 MPa": "pressure 100 MPa", "Tah 100 MPa": "tension 100 MPa",
    "Vnitřní tlak 100 MPa": "internal pressure 100 MPa", "Vlastní tíha": "self-weight", "Krouticí moment 100 N·m": "torque 100 N·m",
}


def loadcase(name: str) -> str:
    """English name of a load case of the MYSTRAN comparison (unchanged when unknown)."""
    return LOADCASES.get(name, name)


def is_english(text: str) -> bool:
    """No Czech letter and no word of the list of ASCII Czech words."""
    return not _CZECH_LETTERS.search(text) and not _CZECH_ASCII.search(text)


def unit(u: str) -> str:
    return UNITS.get(u, u)


def _combo(tags: dict[str, str], rests: dict[str, str]) -> tuple[re.Pattern[str], Callable[[re.Match[str]], str]]:
    """Names "<tag>: <rest>" of a family: both halves are translated from tables."""
    rx = re.compile("^(" + "|".join(re.escape(t) for t in tags) + "): (" + "|".join(re.escape(r) for r in rests) + ")$")
    return rx, lambda m: f"{tags[m[1]]}: {rests[m[2]]}"


def _sub(pattern: str, template: str) -> tuple[re.Pattern[str], Callable[[re.Match[str]], str]]:
    rx = re.compile(pattern)
    return rx, lambda m: m.expand(template)


# ---------------------------------------------------------------------------------------------------- exact labels
LABELS: dict[str, str] = {
    # V1, V6, V17 (cantilever)
    "průhyb konce (Timoshenko)": "tip deflection (Timoshenko)",
    "σx horní vlákno x=100": "σx top fibre at x = 100 mm",
    "σx dolní vlákno x=100": "σx bottom fibre at x = 100 mm (magnitude)",
    "reakce Fz": "reaction Fz",
    "reakce Fx": "reaction Fx",
    # V2
    "σx (střed)": "σx (centre)",
    "σx max |odchylka| v uzlech": "largest |σx| over the nodes",
    "posuv čela ux": "end-face displacement ux",
    # V3, A3, SM1 (plate with a hole)
    "σx max na otvoru, průměr po tl. (Kt,n=2,729)": "σx max at the hole, mean through the thickness (K_t,net)",
    "σx max na otvoru, střední rovina": "σx max at the hole, mid-plane",
    "reakce Fx (síla 100 MPa·50·2,5)": "reaction Fx (100 MPa × 50 × 2.5 mm)",
    # V4, V8 (tube)
    "σθ na ri": "σθ at r_i",
    "σθ na ro": "σθ at r_o",
    "von Mises na ri": "von Mises at r_i",
    "u(ri) průměr": "radial displacement u(r_i), mean",
    "u_r(ro) průměr": "radial displacement u_r(r_o), mean",
    "σr na ri (45°, z = h/2)": "σr at r_i (45°, z = h/2)",
    "σθ na ri = ν·σr (εθ = 0)": "σθ at r_i = ν·σr (ε_θ = 0)",
    "uz horní hrany díry (osově volná)": "axial displacement of the top edge of the bore (axially free)",
    "reakce vazby Fx = −σr(ri)·ri·h": "constraint reaction Fx = −σr(r_i)·r_i·h",
    # V7, V7R (torsion)
    "τ na povrchu x=100 (průměr po obvodu)": "surface shear stress τ at x = 100 mm (mean around the circumference)",
    "zkroucení konce x=200": "twist of the end at x = 200 mm",
    "zkroucení mezi x=50 a 150 (×2)": "twist between x = 50 and 150 mm (×2)",
    "reakční moment Mx ve vetknutí": "reaction moment Mx at the clamp",
    # V10
    "(a) průhyb konce (Timoshenko)": "(a) tip deflection (Timoshenko)",
    "(a') průhyb konce (Timoshenko)": "(a') tip deflection (Timoshenko)",
    # V11 (b)
    "(b) horní hrana: průhyb konce qL⁴/8EI + smyk": "(b) top edge: tip deflection q·L⁴/(8EI) + shear",
    "(b) reakce Fz = q·L": "(b) reaction Fz = q·L",
    "(b) reakční moment My = −qL²/2 (k počátku)": "(b) reaction moment My = −q·L²/2 (about the origin)",
    "(b) reakční moment Mx = 0 (k počátku; hrana y = 0) + 1": "(b) reaction moment Mx = 0 (edge at y = 0), as 1 + Mx/(q·L²/2)",
    # V12
    "(a) zkroucení mezi x = 50 a 150 (×2) = T·L/(GJ)": "(a) twist between x = 50 and 150 mm (×2) = T·L/(G·J)",
    "(a) reakční moment Mx = ⅔·F_θ·R": "(a) reaction moment Mx = ⅔·F_θ·R",
    # V13 (c)
    "(c) lineární: reakce Fy = −∫p dA": "(c) linear pressure: reaction Fy = −∫p dA",
    "(c) lineární: reakční moment Mx = ∫p·z dA": "(c) linear pressure: reaction moment Mx = ∫p·z dA",
    # V14
    "(a) kotouč: σθ(ri)": "(a) disc: σθ(r_i)",
    "(a) kotouč: σθ(ro)": "(a) disc: σθ(r_o)",
    "(a) kotouč: u_r(ro) = ro·σθ(ro)/E": "(a) disc: u_r(r_o) = r_o·σθ(r_o)/E",
    "(b) tyč α: průhyb konce 11q₀L⁴/120EI + smyk": "(b) bar, angular acceleration α: tip deflection 11·q₀·L⁴/(120EI) + shear",
    "(b) tyč α: reakce Fy = ρAαL²/2": "(b) bar, α: reaction Fy = ρ·A·α·L²/2",
    "(b) tyč α: reakční moment Mz = α·J_z (k ose)": "(b) bar, α: reaction moment Mz = α·J_z (about the axis)",
    "(c) tyč ω: reakce Fx = −ρω²AL²/2": "(c) bar, angular velocity ω: reaction Fx = −ρ·ω²·A·L²/2",
    "(c) tyč ω: prodloužení ρω²L³/(3E)": "(c) bar, ω: elongation ρ·ω²·L³/(3E)",
    # V15
    "(a) u_z dna kostky = −p/k": "(a) block: bottom displacement u_z = −p/k",
    "(a) reakce podloží kostky = p·A": "(a) block: foundation reaction = p·A",
    "(b) průhyb pod silou w₀ = Pλ/(2k_b) (Hetényi)": "(b) beam: deflection under the load w₀ = P·λ/(2k_b) (Hetényi)",
    "(b) σx dolní vlákno 40 mm od síly = M/W (Hetényi)": "(b) beam: σx bottom fibre 40 mm from the load = M/W (Hetényi)",
    "(b) σx horní vlákno 40 mm od síly = −M/W": "(b) beam: σx top fibre 40 mm from the load = −M/W",
    "(b) reakce podloží nosníku = P": "(b) beam: foundation reaction = P",
    # V16
    "(a) ε_x tyče s natočenou ortotropií": "(a) bar with rotated orthotropy: ε_x",
    "(a) ε_y (Poisson v natočených osách)": "(a) ε_y (Poisson effect in the rotated axes)",
    "(a) γ_xy/2 (smyk od natočení os)": "(a) γ_xy/2 (shear from the rotation of the axes)",
    "(b) u_r vnějšího pláště prstence (Lekhnitskii)": "(b) ring: outer radial displacement u_r (Lekhnitskii)",
    # V17
    "plocha *TIE = průřez 20×10": "*TIE area = cross-section 20×10 mm",
    "průhyb konce (Timoshenko, jako V1)": "tip deflection (Timoshenko, as V1)",
    "σx horní vlákno x = 50 (díl A)": "σx top fibre at x = 50 mm (part A)",
    "σx horní vlákno x = 150 (díl B, jiná síť)": "σx top fibre at x = 150 mm (part B, other mesh)",
    # V18
    "trubka: membrána σθ = p·ri/t (rovnováha)": "tube: membrane σθ = p·r_i/t (equilibrium)",
    "trubka: ohyb σθ = 6/t²∫σθ(t/2 − x)dx (Lamé)": "tube: bending part of σθ = 6/t²·∫σθ(t/2 − x)dx (Lamé)",
    "trubka: membrána σr = (−p)…0 průměr": "tube: membrane σr = mean of (−p … 0)",
    "konzola x = 200: ohyb σx na spodním vlákně = −M/W (M = 100·100)":
        "cantilever at x = 200 mm: bending σx at the bottom fibre = −M/W (M = 100·100 N·mm)",
    "konzola: membrána σx ≈ 0 (+30)": "cantilever: membrane σx ≈ 0 (shown as 30 + σx)",
    "konzola: σx podél výšky – spodní vlákno (tlak)": "cantilever: σx along the height – bottom fibre (compression)",
    # CS1
    "CS1 sigma_theta vnitrni povrch": "σθ at the inner surface",
    "CS1 sigma_theta vnejsi povrch": "σθ at the outer surface",
    "CS1 sigma_r vnitrni tlak (-100 MPa)": "σr at the inner surface (−100 MPa pressure)",
    "CS1 sigma_r vnejsi povrch (0 MPa)": "σr at the outer surface (0 MPa)",
    "CS1 radialni posuv u_r vnitrni": "radial displacement u_r at the inner surface",
    # AS1
    "AS1 Mises bezpecnost na krutu V7": "von Mises safety factor in torsion (V7)",
    "AS1 Tresca bezpecnost na krutu V7": "Tresca safety factor in torsion (V7)",
    "AS1 Mohr-Coulomb litina na krutu V7": "Mohr-Coulomb safety factor, cast iron, torsion (V7)",
    # FB1
    "FB1 V7 kroutici moment Mx": "torsion (V7): torque Mx at the cut",
    # IM1
    "IM1 plocha otisku kruhu (pi*r^2)": "imprint area of the circle (π·r²)",
    "IM1 reakce Fz na vetknuti (p*A)": "reaction Fz at the support (p·A)",
    "IM1 reakcni moment My k c0 (v ose zatizeni = 0)": "reaction moment My about c0 (load on the axis, = 0)",
    "IM1 pruhyb volneho konce x=100 (konzola, siroky pas + smyk)": "tip deflection at x = 100 mm (cantilever, wide strip + shear)",
    # SM1
    "SM1 hruby globalni model max von Mises (h=14mm) vs Peterson": "coarse global model: max von Mises (h = 14 mm) vs Peterson",
    # T1
    "(a) T rozhraní x = 50": "(a) interface temperature at x = 50 mm",
    "(a) T čela x = 100": "(a) end-face temperature at x = 100 mm",
    "(a) teplo předepsané teploty": "(a) heat at the prescribed temperature",
    "(a) teplo konvekcí": "(a) heat by convection",
    "(a) tok q_x v oceli": "(a) heat flux q_x in the steel",
    "(b) T čela x = 100": "(b) end-face temperature at x = 100 mm",
    "(c) T čela x = 100 (výkon 1 W)": "(c) end-face temperature at x = 100 mm (power 1 W)",
    "(c) teplo předepsané teploty": "(c) heat at the prescribed temperature",
    # T4
    "(a) volná roztažnost ux(L) = αΔT·L": "(a) free expansion ux(L) = α·ΔT·L",
    "(a) bez napětí: 1 + max σ_vM/(EαΔT)": "(a) stress-free: 1 + max σ_vM/(E·α·ΔT)",
    "(b) bráněná: σx = −EαΔT": "(b) restrained: σx = −E·α·ΔT",
    "(b) reakce = EαΔT·A": "(b) reaction = E·α·ΔT·A",
    # T2, T3, T7
    "radiální tok q_r(30) = k·ΔT/(r·ln(ro/ri))": "radial heat flux q_r(30) = k·ΔT/(r·ln(r_o/r_i))",
    "teplo vnitřní plochou": "heat through the inner surface",
    "T uprostřed x = 50": "temperature at mid-length x = 50 mm",
    "T x = 25": "temperature at x = 25 mm",
    "teplo koncem x = 0": "heat through the end x = 0",
    "teplo koncem x = 100": "heat through the end x = 100 mm",
    "zdroj (integrace)": "heat source (integrated)",
    "T čela (sálání)": "end-face temperature (radiation)",
    "teplo sáláním": "heat radiated",
    # TH8, TH8B, TH9, KT1, HT1
    "T rozhraní x = 50- (slave)": "interface temperature at x = 50 mm, side 1 (slave)",
    "T rozhraní x = 50+ (master)": "interface temperature at x = 50 mm, side 2 (master)",
    "ΔT styku (q / h_c)": "interface ΔT (q / h_c)",
    "teplo vstupující na x = 0": "heat entering at x = 0",
    "teplo odvedené na x = 100": "heat removed at x = 100 mm",
    "tepelný tok sáláním Q": "radiated heat flow Q",
    "teplo odvedené deskou 2": "heat removed by plate 2",
    "tepelný tok přes kontakt Q = k·A·ΔT/L (106,6 W)": "heat flow across the contact Q = k·A·ΔT/L",
    "teplota rozhraní (50 °C)": "interface temperature (50 °C)",
    "HT1 svisla deska prirozena konvekce h": "natural convection, vertical plate: h",
    "HT1 nucena konvekce voda v trubce h": "forced convection, water in a pipe: h",
    # M1, M2, B1
    "f₁ ohyb kolem slabé osy (E–B)": "f₁ bending about the weak axis (E–B)",
    "f₂ ohyb kolem silné osy (E–B, L/h = 10)": "f₂ bending about the strong axis (E–B, L/h = 10)",
    "efektivní hmota 1. módu / m (E–B, numerická integrace tvaru, 0,6131)": "effective mass of mode 1 / m (E–B, numerical integration of the mode shape)",
    "celková hmota ρV (volné těleso – žádné vázané uzly)": "total mass ρ·V (free body – no constrained nodes)",
    "6 módů tuhého pohybu: max f₁…₆ / f₇ (≈ 0) + 1": "6 rigid-body modes: 1 + max(f₁…f₆) / f₇ (≈ 0)",
    "f₇ volný nosník, ohyb kolem slabé osy (E–B, Timoshenko −0,9 %)": "f₇ free beam, bending about the weak axis (E–B; Timoshenko −0.9 %)",
    "λ₁·F = π²EI/(4L²) (slabá osa)": "λ₁·F = π²·E·I/(4·L²) (weak axis)",
    "λ₂/λ₁ = I_silná/I_slabá = 4": "λ₂/λ₁ = I_strong/I_weak = 4",
    "statický stav: tlak na konci = reakce": "static state: end compression = reaction",
    # Y1 ... Y16
    "f svislého módu √(k/m)/2π": "frequency of the vertical mode √(k/m)/2π",
    "|u|/u_st, rezonance r = 1": "|u|/u_st, resonance r = 1",
    "fáze v rezonanci (90°)": "phase at resonance (90°)",
    "špička |u|/u_st = 1/η (η = 0,04)": "peak |u|/u_st = 1/η (η = 0.04)",
    "f₁ (Euler–Bernoulli, Timoshenko −0,3 %)": "f₁ (Euler–Bernoulli; Timoshenko −0.3 %)",
    "špička |u|/u_st (E–B 0,9707/(2ζ))": "peak |u|/u_st (E–B 0.9707/(2ζ))",
    "DLF skoku 1 DOF, ζ = 0": "step DLF, 1 DOF, ζ = 0",
    "DLF skoku 1 DOF, ζ = 2 % (1 + e^(−πζ/√(1−ζ²)))": "step DLF, 1 DOF, ζ = 2 % (1 + e^(−πζ/√(1−ζ²)))",
    "zbytková amplituda (2a/ω²)|sin(ωτ/2)|": "residual amplitude (2a/ω²)·|sin(ωτ/2)|",
    "f₁ s hmotou na konci (Rayleigh)": "f₁ with the tip mass (Rayleigh)",
    "hmotnost modelu m + M": "model mass m + M",
    "f₁ s tuhou základnou = vetknutí (E–B)": "f₁ with the rigid base = clamped (E–B)",
    "g_rms absolutního zrychlení hmoty (Miles·√(1+4ζ²))": "g_rms of the absolute acceleration of the mass (Miles·√(1+4ζ²))",
    "S_a(f_n)/A půlsinus f_nτ = 0,82, ζ 5 % (1,6516)": "S_a(f_n)/A of a half-sine pulse, f_n·τ = 0.82, ζ = 5 %",
    "Southwellův součinitel 1. ohybu mimo rovinu otáčení (1D model rotujícího nosníku E–B, Ω/ω₀ = 0,957)":
        "Southwell coefficient of the 1st flapwise bending mode (independent 1-D rotating-beam model, Ω/ω₀ = 0.957)",
    "f₁ při 0 ot/min (E–B 104,4 Hz)": "f₁ at 0 rpm (Euler-Bernoulli, 104.4 Hz)",
    # K1, K4, K7, R1, G1, KK4, KK2
    "tlak ve styku – průměr (10 MPa)": "contact pressure – mean (10 MPa)",
    "tlak ve styku – max. odchylka": "contact pressure – largest deviation (as 1 + |p − 10|/10)",
    "normálová síla páru = p·A": "normal force of the pair = p·A",
    "u_z vrchu = 2ph/E + p/K": "top displacement u_z = 2·p·h/E + p/K",
    "tlak ve spoji p (vážený průměr uzlového CPRESS)": "joint pressure p (weighted mean of the nodal CPRESS)",
    "σ_t náboje na vnitřním povrchu": "hub hoop stress σ_t at the inner surface",
    "u_r náboje vnitřní povrch": "hub radial displacement u_r, inner surface",
    "u_r hřídele": "shaft radial displacement u_r",
    "u_r náboje vnější povrch (Lamé 2p·ri²·ro/(E(ro² − ri²)))": "hub radial displacement u_r, outer surface (Lamé 2p·r_i²·r_o/(E(r_o² − r_i²)))",
    "síla ve styku (E–B, mezera 0,5)": "contact force (E–B, gap 0.5 mm)",
    "plató tření T = μN (300 N)": "friction plateau T = μ·N (300 N)",
    "třecí síla páru Σ τ·A": "friction force of the pair Σ τ·A",
    "normálová síla páru N": "normal force of the pair N",
    "reakce podpory jen v tlaku = p·A": "compression-only support reaction = p·A",
    "posuv vrchu = ph/E + p/K": "top displacement = p·h/E + p/K",
    "max. kontaktní síla (Hertz 2310 N)": "maximum contact force (Hertz)",
    "doba rázu (Hertz 52,3 µs)": "impact duration (Hertz)",
    "W/P = tan(α + arctg μ) (1,0610)": "W/P = tan(α + arctan μ)",
    "montážní síla W (Bayer: P·(μ + tg α)/(1 − μ·tg α) s P z MKP)": "assembly force W (Bayer: P·(μ + tan α)/(1 − μ·tan α), P from the FE result)",
    # S1 ... W2
    "tah: síla ve šroubu F_SA = zatížení": "tension: bolt force F_SA = applied load",
    "tah: F_A přenesená rovinou spáry": "tension: force F_A transmitted through the joint plane",
    "smyk: příčná síla ve šroubu = zatížení": "shear: transverse bolt force = applied load",
    "tíha (uzlové síly, model s U1): reakce Fz = m·g": "weight (nodal forces, beam-bolt model): reaction Fz = m·g",
    "smyk přenesený rovinou spáry (spára + šroub)": "shear transferred through the joint plane (gap + bolt)",
    "VDI F_M,zul (M10 8.8, μ_G = 0,12)": "VDI 2230 F_M,zul (M10 8.8, μ_G = 0.12)",
    "EN 1993-1-8 F_v,Rd (závit, α_v = 0,6)": "EN 1993-1-8 F_v,Rd (threaded part, α_v = 0.6)",
    "EN 1993-1-8 F_b,Rd (e1 = e2 = 20 z geometrie)": "EN 1993-1-8 F_b,Rd (e1 = e2 = 20 mm from the geometry)",
    "spára + šrouby přenesou celkem (MKP)": "joint plane + bolts transfer in total (FE)",
    "skupina: šroub 1 = F/n": "bolt group: bolt 1 = F/n",
    "skupina: šroub 2 = F/n": "bolt group: bolt 2 = F/n",
    "F_b,Rd deska A, šroub 1 (e1 = 60, p1 = 40, e2 = 20)": "F_b,Rd plate A, bolt 1 (e1 = 60, p1 = 40, e2 = 20 mm)",
    "tah: součet sil úseků obou svarů": "tension: sum of the segment forces of both welds",
    "podélný smyk: součet sil úseků": "longitudinal shear: sum of the segment forces",
    "podélný smyk: průměrné τ∥ = F/(2aL)": "longitudinal shear: mean τ∥ = F/(2·a·L)",
    # U1 ... U4
    "amplituda σ v polovině délky (horní vlákno)": "stress amplitude σ at mid-length (top fibre)",
    "poškození za blok = Miner z amplitudy uzlu": "damage per block = Miner from the nodal amplitude",
    "bezpečnost = (1/(R·D))^(1/k)": "safety factor = (1/(R·D))^(1/k)",
    "poškození proti teorii (σa = 30 MPa)": "damage against theory (σa = 30 MPa)",
    "amplituda v rohu x=100 (30 + 15 MPa)": "amplitude at the corner x = 100 mm (30 + 15 MPa)",
    "(a) tah: σa na povrchu = F/A": "(a) tension: σa at the surface = F/A",
    "(b) krut: σn krit. rovina σa = τ (teorie)": "(b) torsion: critical-plane normal-stress amplitude σa = τ (theory)",
    "(b) krut: Findley σekv = √3·τ (teorie)": "(b) torsion: Findley σ_eq = √3·τ (theory)",
    "(c) 90° mimo fázi: Findley = ruční výpočet": "(c) 90° out of phase: Findley = hand calculation",
    "(c) 90° mimo fázi: σn krit. rovina = ruční výpočet": "(c) 90° out of phase: critical-plane σn = hand calculation",
    "(c) poškození = Miner z amplitudy (5 cyklů/blok)": "(c) damage = Miner from the amplitude (5 cycles per block)",
    "U4 Neuber soucin sig*eps = sig_el^2/E": "Neuber product σ·ε = σ_el²/E",
    "U4 Glinka energie W = Wel": "Glinka energy density W = W_el",
    "U4 Neuber: eps splnuje Ramberg-Osgood": "Neuber: ε satisfies Ramberg-Osgood",
    "U4 Glinka deformace eps (nezavisle pulenim)": "Glinka strain ε (independent bisection)",
    "U4 Coffin-Manson zivotnost Nf (nezavisle pulenim)": "Coffin-Manson life N_f (independent bisection)",
    "U4 Morrow zivotnost Nf, sigma_m = 50 MPa (nezavisle)": "Morrow life N_f, σ_m = 50 MPa (independent)",
    "U4 SWT zivotnost Nf, sigma_max = 300 MPa (nezavisle)": "SWT life N_f, σ_max = 300 MPa (independent)",
    "U4 MKP elasticka spicka sig_el na otvoru vs Peterson (Kt=2.729)": "FE elastic peak σ_el at the hole vs Peterson",
    "U4 MKP poskozeni uzlu D odpovida Coffin-Manson(eps_Neuber)": "FE nodal damage D = Coffin-Manson(ε_Neuber)",
    "U4 MKP soucinitel bezpecnosti (D*bloky = 1, nezavisle pulenim)": "FE safety factor (D·blocks = 1, independent bisection)",
    # P1
    "A (tah ⟂ vrstvám): u = σL/E_z": "A (tension ⟂ layers): u = σ·L/E_z",
    "B (tah v rovině vrstvy): u = σL/E_xy": "B (tension in the layer plane): u = σ·L/E_xy",
    "A příčně: uy = −ν13·σ/E_xy·10": "A transverse: uy = −ν₁₃·σ/E_xy·10",
    "B svisle: uz = −ν13·σ/E_xy·10": "B vertical: uz = −ν₁₃·σ/E_xy·10",
    "σx (globálně) v tyči A": "σx (global) in bar A",
    "Tsai-Hill A: λ = Rm_z/σ": "Tsai-Hill A: λ = R_m,z/σ",
    "Tsai-Hill B: λ = Rm/σ": "Tsai-Hill B: λ = R_m/σ",
    "reakce Fx (obě tyče)": "reaction Fx (both bars)",
    # F1 ... F14
    "průhyb síla FL³/3EI + FL/κGA (1 prvek)": "deflection under the force F·L³/(3EI) + F·L/(κGA) (1 element)",
    "M kořen = F·L": "root moment M = F·L",
    "posouvající síla Vz = −F": "shear force Vz = −F",
    "zkroucení T·L/(G·J) (J ≠ Ip)": "twist T·L/(G·J) (J ≠ I_p)",
    "krouticí moment T": "torque T",
    "spojité: průhyb qL⁴/8EI + qL²/2κGA": "distributed load: deflection q·L⁴/(8EI) + q·L²/(2κGA)",
    "spojité: M kořen = qL²/2": "distributed load: root moment q·L²/2",
    "spojité: posuv z obrazce na konci prutu = posuv uzlu": "distributed load: displacement from the member diagram at its end = node displacement",
    "průhyb uprostřed 5qL⁴/384EI + qL²/8κGA (1 prvek)": "mid-span deflection 5·q·L⁴/(384EI) + q·L²/(8κGA) (1 element)",
    "M uprostřed = qL²/8 (tah dole: My < 0)": "mid-span moment = q·L²/8 (tension at the bottom: My < 0)",
    "max. |M| z obrazce = qL²/8 (bod V = 0)": "max |M| of the diagram = q·L²/8 (where V = 0)",
    "reakce A = qL/2": "reaction A = q·L/2",
    "reakce B = qL/2": "reaction B = q·L/2",
    "moment v patě sloupu (sklon–průhyb, osově tuhé pruty)": "moment at the column base (slope-deflection, axially rigid members)",
    "moment v rohu rámu": "moment at the frame corner",
    "moment v patě druhého sloupu": "moment at the base of the second column",
    "součet vodorovných reakcí = H": "sum of the horizontal reactions = H",
    "osová síla AC = −P/(2 sin θ)": "axial force AC = −P/(2·sin θ)",
    "osová síla BC": "axial force BC",
    "ohybový moment v příhradovém prutu = 0 (abs.)": "bending moment in a truss member = 0 (absolute)",
    "svislý posuv vrcholu = −N·L/(EA·sin θ)": "vertical displacement of the apex = −N·L/(EA·sin θ)",
    "moment ve vetknutí = P/2·L₁": "moment at the clamp = P/2·L₁",
    "moment v kloubu = 0 (abs.)": "moment at the hinge = 0 (absolute)",
    "reakce B = P/2": "reaction B = P/2",
    "reakce A = P/2": "reaction A = P/2",
    "f₁ slabá osa (Timoshenko, 8 prvků)": "f₁ weak axis (Timoshenko, 8 elements)",
    "f₂ slabá osa": "f₂ weak axis",
    "f₃ slabá osa": "f₃ weak axis",
    "f₁ silná osa (smyk + rotační setrvačnost)": "f₁ strong axis (shear + rotary inertia)",
    "f torze (A vázané, B volné): 1/(4L)·√(GJ/ρIp)": "torsional frequency (A fixed, B free): 1/(4L)·√(GJ/(ρ·I_p))",
    "průhyb konce prutu (konzola 400, E–B + smyk)": "member tip deflection (cantilever 400 mm, E–B + shear)",
    "moment v prutu u spojení = P·200": "member moment at the connection = P·200",
    "reakce vetknutí tělesa Fz = P": "reaction at the clamp of the solid Fz = P",
    "síla prut → těleso Fz (.dat 6 číslic)": "force member → solid Fz (.dat, 6 digits)",
    "vzpěr konzoly β = 2 (Euler + Engesser, 8 prvků)": "buckling of the cantilever β = 2 (Euler + Engesser, 8 elements)",
    "vzpěr kloubového sloupu β = 1 (Euler + Engesser)": "buckling of the pinned column β = 1 (Euler + Engesser)",
    "statika: tlak = reakce": "statics: compression = reaction",
    "MYSTRAN CBAR × vlastní řešič: max. rozdíl posuvů uzlů / max. posuv": "MYSTRAN CBAR vs own beam solver: max node-displacement difference / max displacement",
    "MYSTRAN CBAR × vlastní řešič (prostorový rám): max. rozdíl posuvů uzlů / max. posuv":
        "MYSTRAN CBAR vs own beam solver (spatial frame): max node-displacement difference / max displacement",
    "silná osa: w = FL³/3EI_y + FL/(G·A_sz)": "strong axis: w = F·L³/(3EI_y) + F·L/(G·A_sz)",
    "silná osa: natočení konce θy = FL²/2EI (kladné kolem y = dolů)": "strong axis: end rotation θy = F·L²/(2EI) (positive about y = downwards)",
    "slabá osa: v = FL³/3EI_z + FL/(G·A_sy)": "weak axis: v = F·L³/(3EI_z) + F·L/(G·A_sy)",
    "tíha: průhyb konce qL⁴/8EI + qL²/2G·A_sz": "self-weight: tip deflection q·L⁴/(8EI) + q·L²/(2G·A_sz)",
    "natočený průřez 30°: posuv uy": "section rolled by 30°: displacement uy",
    "natočený průřez 30°: posuv uz": "section rolled by 30°: displacement uz",
    "pružina k_θy: průhyb = F L³/3EI + FL/GA + θ·L": "rotational spring k_θy: deflection = F·L³/(3EI) + F·L/(GA) + θ·L",
    "pružina k_θy: reakční moment = −k·θ = F·L": "rotational spring k_θy: reaction moment = −k·θ = F·L",
    "torze 1. mód, 4 prvky (O(h²))": "torsion mode 1, 4 elements (O(h²))",
    "torze 1. mód, 8 prvků (O(h²))": "torsion mode 1, 8 elements (O(h²))",
    "torze 1. mód, 16 prvků": "torsion mode 1, 16 elements",
    "ohyb 1. mód konzoly = Euler–Bernoulli (štíhlý prut, malý smyk)": "cantilever bending mode 1 = Euler–Bernoulli (slender member, small shear)",
    "kloubový sloup, E–B, 4 prvky": "pinned column, E–B, 4 elements",
    "kloubový sloup, E–B, 8 prvků": "pinned column, E–B, 8 elements",
    "konzola, E–B, 2 prvky": "cantilever, E–B, 2 elements",
    "konzola, E–B, 4 prvky": "cantilever, E–B, 4 elements",
    "kloub–vetknutí, E–B, 8 prvků (λ_kr = 20,1907·EI/L²)": "pinned–clamped, E–B, 8 elements (λ_cr = 20.1907·EI/L²)",
    "kloubový sloup s posunem: Engesser P_E/(1 + P_E/GA_s), 8 prvků": "pinned column with shear: Engesser P_E/(1 + P_E/(G·A_s)), 8 elements",
    "konzola s posunem: Engesser P_E/(1 + P_E/GA_s), 8 prvků": "cantilever with shear: Engesser P_E/(1 + P_E/(G·A_s)), 8 elements",
    "vetknutá tyč: N = −E·A·α·ΔT": "restrained bar: N = −E·A·α·ΔT",
    "vetknutá tyč: reakce = E·A·α·ΔT": "restrained bar: reaction = E·A·α·ΔT",
    "vetknutá tyč: posuv uzlu B = 0 (abs.)": "restrained bar: displacement of node B = 0 (absolute)",
    "vetknutý nosník: M_y = −E·I·α·ΔT_g/h": "clamped beam: M_y = −E·I·α·ΔT_g/h",
    "volná tyč: prodloužení α·ΔT·L": "free bar: elongation α·ΔT·L",
    "volná tyč: N = 0 (abs.)": "free bar: N = 0 (absolute)",
    "prostý nosník s gradientem: průhyb uprostřed α·ΔT_g·L²/(8h)": "simply supported beam with a gradient: mid-span deflection α·ΔT_g·L²/(8h)",
    "prostý nosník s gradientem: M = 0 (abs.)": "simply supported beam with a gradient: M = 0 (absolute)",
    # A1 ... Sh3
    "σθ na ri (Lamé)": "σθ at r_i (Lamé)",
    "σθ na ro (Lamé)": "σθ at r_o (Lamé)",
    "σr na ri = −p": "σr at r_i = −p",
    "σθ na ri, polární úhel 45° (Lamé koule)": "σθ at r_i, polar angle 45° (Lamé sphere)",
    "σθ na ro, polární úhel 45°": "σθ at r_o, polar angle 45°",
    "σθ na ri, rovník": "σθ at r_i, equator",
    "|u|(ri), polární úhel 45°": "|u|(r_i), polar angle 45°",
    "u_r(ri) na rovníku": "u_r(r_i) at the equator",
    "σ meridiánové = σθ (tečné v bodě 45°)": "meridional σ = σθ (tangential at the 45° point)",
    "σx max na otvoru (Kt Heywood, d/W = 0,1)": "σx max at the hole (K_t Heywood, d/W = 0.1)",
    "σx max v uzlech = na otvoru": "σx max over the nodes = at the hole",
    "reakce Fx sym_x (100 MPa·50·2,5)": "reaction Fx on the symmetry plane (100 MPa × 50 × 2.5 mm)",
    "σz na ri = ν(σr + σθ)": "σz at r_i = ν·(σr + σθ)",
    "σz na ro = ν(σr + σθ)": "σz at r_o = ν·(σr + σθ)",
    "u_r(ri) rovinná deformace": "u_r(r_i), plane strain",
    "reakce |Fx| sym_x = p·ri·t": "reaction |Fx| on the symmetry plane = p·r_i·t",
    "σr = σθ ve středu (tenký disk)": "σr = σθ at the centre (thin disc)",
    "σθ ve středu": "σθ at the centre",
    "σr v r = 50": "σr at r = 50 mm",
    "σθ v r = 50": "σθ at r = 50 mm",
    "σθ na obvodu": "σθ at the rim",
    "u_r na obvodu": "u_r at the rim",
    "σx ve středu (= σr = σθ)": "σx at the centre (= σr = σθ)",
    "reakce |Fx| sym_x = ρω²tR³/3": "reaction |Fx| on the symmetry plane = ρ·ω²·t·R³/3",
    "průhyb konce (Timoshenko, nosník)": "tip deflection (Timoshenko beam)",
    "σx horní plocha x = L/2 (M/W)": "σx top surface at x = L/2 (M/W)",
    "σx dolní plocha x = L/2": "σx bottom surface at x = L/2 (magnitude)",
    "1 + membránové σx / M/W (≈ 1)": "1 + membrane σx / (M/W) (≈ 1)",
    "reakce My k ref. bodu (rameno síly)": "reaction My about the reference point (lever arm of the force)",
    "tíha: reakce Fz = ρ·t·A·g": "weight: reaction Fz = ρ·t·A·g",
    "průhyb středu (Timoshenko 0,00406 q a⁴/D)": "centre deflection (Timoshenko, 0.00406·q·a⁴/D)",
    "σx horní plocha střed (6M/t², M = 0,0479 q a²)": "σx top surface at the centre (6M/t², M = 0.0479·q·a²)",
    "σy dolní plocha střed": "σy bottom surface at the centre",
    "reakce Fz = q a²": "reaction Fz = q·a²",
    "σθ střední plocha (Lamé, průměr ri/ro ≈ p·ri/t)": "σθ mid-surface (Lamé, mean of r_i and r_o ≈ p·r_i/t)",
    "σθ ohybová složka (Lamé (ro − ri)/2)": "σθ bending part (Lamé, (σ(r_o) − σ(r_i))/2)",
    "u_r střední plochy (Lamé)": "mid-surface u_r (Lamé)",
    "σθ tenkostěnně pR/t (informativně, 3 %)": "σθ thin-wall p·R/t (3 % tolerance)",
    # PL1
    "PL1: Napětí SZZ v plastické oblasti": "stress SZZ in the plastic range",
    "PL1: Tahová reakce RF_Z (se znaménkem: podpora tlačí proti tahu)": "tensile reaction RF_Z (signed: the support pushes against the tension)",
    "PL1: Plastické přetvoření PEEQ": "equivalent plastic strain PEEQ",
    "u_r(ri)": "u_r(r_i)",
    # CY1, CY2
    "Kotouč ω: σθ(ri) (Lamé)": "spinning disc: σθ(r_i) (Lamé)",
    "Kotouč ω: σθ(ro) (Lamé)": "spinning disc: σθ(r_o) (Lamé)",
    "Kotouč ω: u_r(ro) = ro·σθ(ro)/E": "spinning disc: u_r(r_o) = r_o·σθ(r_o)/E",
    "Kotouč ω: tečná reakce otvoru ≈ 0 (1 + |R|/ρω²V r)": "spinning disc: tangential reaction of the bore ≈ 0 (as 1 + |R|/(ρ·ω²·V·r))",
    "Tlak: reakce Fz sektoru = tlak × plocha": "pressure: reaction Fz of the sector = pressure × area",
    # MB1 ... MB9
    "poloha pístu x(60°)": "piston position x(60°)",
    "rychlost pístu v(60°)": "piston velocity v(60°)",
    "zrychlení pístu a(60°)": "piston acceleration a(60°)",
    "moment pohonu M = F·v/ω": "drive torque M = F·v/ω",
    "θ₄(60°) otevřená větev": "θ₄(60°), open branch",
    "θ₄(60°) překřížená větev": "θ₄(60°), crossed branch",
    "θ₄(60°) otevřená = Freudenstein přesně": "θ₄(60°), open branch = Freudenstein (exact)",
    "θ₄(60°) překřížená = Freudenstein přesně": "θ₄(60°), crossed branch = Freudenstein (exact)",
    "max |θ₄ − Freudenstein| za otáčku (otevřená)": "max |θ₄ − Freudenstein| over one revolution (open branch)",
    "max |θ₄ − Freudenstein| za otáčku (překřížená)": "max |θ₄ − Freudenstein| over one revolution (crossed branch)",
    "perioda T (fyzikální kyvadlo, K(sin²30°))": "period T (physical pendulum, K(sin²30°))",
    "síla v čepu dole = mg(3 − 2cos θ₀) = 2mg": "pin force at the bottom = m·g·(3 − 2·cos θ₀) = 2·m·g",
    "zachování energie (max − min)/E": "energy conservation (max − min)/(m·g·L)",
    "zrychlení valení a = 2/3·g·sin α (μ 0,3 ≥ tan α/3 = 0,192)": "rolling acceleration a = (2/3)·g·sin α (μ 0.3 ≥ tan α/3 = 0.192)",
    "třecí síla valení F = m·g·sin α/3 (moment vazby / R)": "rolling friction force F = m·g·sin α/3 (constraint moment / R)",
    "potřebné μ řešiče = |třecí síla (moment vazby/R)| / normálová síla kloubu svah = tan α/3 (μ 0,1 < μ_min → skluz 4,055 m/s² jen s kontaktem C1 – NEOVĚŘITELNÉ)":
        "required friction coefficient = |friction force| / normal force = tan α/3",
    "zpomalení a = μ·g": "deceleration a = μ·g",
    "doba doběhu t = v₀/(μg) = 0,5097 s": "stopping time t = v₀/(μ·g)",
    "dráha s = v₀²/(2μg) = 254,8 mm": "stopping distance s = v₀²/(2·μ·g)",
    "rychlost po zastavení (t > 0,6 s) / v₀": "velocity after stopping (t > 0.6 s) / v₀ (≈ 0)",
    "x_s = μs·m·g/k (odtržení: vrchol tření / k)": "breakaway displacement x_s = μ_s·m·g/k (peak friction / k)",
    "x_k = μk·m·g/k (střed kmitu při prokluzu)": "centre of the slip oscillation x_k = μ_k·m·g/k",
    "největší výchylka x_k + √((x_s − x_k)² + (v/ω)²)": "largest displacement x_k + √((x_s − x_k)² + (v/ω)²)",
    "perioda stick-slip T": "stick-slip period T",
    "převod ω₂/ω₁": "gear ratio ω₂/ω₁",
    "úhel kola 2 = poměr × úhel kola 1 (konec)": "angle of wheel 2 = ratio × angle of wheel 1 (end)",
    "zdvih matice = stoupání × otáčky (2 otáčky)": "nut travel = lead × turns (2 turns)",
    "moment šroubu M = F·P/(2π)": "screw torque M = F·P/(2π)",
    # MB4C, MB4R, KK5 (Chrono)
    "skluz μ 0.1: a = g(sin α − μ cos α) (zadání 4,055)": "sliding μ 0.1: a = g·(sin α − μ·cos α)",
    "skluz: úhlové zrychlení ε = 2μg cos α / R (mnohoúhelník triangulace, Euler)": "sliding: angular acceleration ε = 2·μ·g·cos α/R (polygonal triangulation of the cylinder, Euler time step)",
    "valení μ 0.3: a = 2/3·g·sin α (zadání 3,270; Euler implicitní tlumí ~h)": "rolling μ 0.3: a = (2/3)·g·sin α",
    "valení bez skluzu: |v − ω·R| / v na konci": "rolling without slip: |v − ω·R|/v at the end",
    "zdvih s(60°) = e cos θ + √((R_c + r_f)² − e² sin² θ) − (R_c + r_f − e)": "lift s(60°) = e·cos θ + √((R_c + r_f)² − e²·sin² θ) − (R_c + r_f − e)",
    "zdvih s(90°) = e cos θ + √((R_c + r_f)² − e² sin² θ) − (R_c + r_f − e)": "lift s(90°) = e·cos θ + √((R_c + r_f)² − e²·sin² θ) − (R_c + r_f − e)",
    # @@MORE-LABELS@@
}

# ---------------------------------------------------------------------------------------------------- families of names
_V10_TAGS = {"(a) tuhá, bod v těžišti": "(a) rigid, point at the centroid",
             "(b) tuhá, bod 50 mm za čelem": "(b) rigid, point 50 mm behind the face",
             "(a') distribuční, těžiště": "(a') distributing, point at the centroid",
             "(b') distribuční, 50 mm za čelem": "(b') distributing, point 50 mm behind the face",
             "(c) distribuční kloub": "(c) distributing hinge", "(c') tuhý kloub": "(c') rigid hinge"}
_V10_REST = {"reakce Fz v bodě": "reaction Fz at the point",
             "reakce My v bodě = −F·rameno": "reaction My at the point = −F·lever arm",
             "průhyb uprostřed (Timoshenko)": "mid-span deflection (Timoshenko)",
             "natočení bodu θy = qL³/(24EI)": "rotation of the point θy = q·L³/(24EI)",
             "reakce Fz v bodě = qL/2": "reaction Fz at the point = q·L/2",
             "moment v bodě ≈ 0 (1 + |M| / (qL/2·L))": "moment at the point ≈ 0 (as 1 + |M| / (q·L/2·L))",
             "reakce konce Fz = qL/2": "end reaction Fz = q·L/2"}
_V11_TAGS = {"(a) hrana čela": "(a) edge of the end face", "(c) bod": "(c) point", "(d) 2 body": "(d) two points"}
_V11_REST = {"průhyb konce (Timoshenko)": "tip deflection (Timoshenko)", "reakce Fz": "reaction Fz",
             "reakční moment My = −F·L (k počátku)": "reaction moment My = −F·L (about the origin)"}
_V13_TAGS = {"(a) plná hladina": "(a) full height", "(b) hladina h = 500": "(b) liquid level h = 500 mm"}
_V13_REST = {"průhyb vrcholu": "deflection of the top", "reakce Fy = −ρgbh²/2": "reaction Fy = −ρ·g·b·h²/2",
             "reakční moment Mx = ρgbh³/6 (k počátku)": "reaction moment Mx = ρ·g·b·h³/6 (about the origin)"}

_PL2_TAGS = {"0,8 My (pružně)": "0.8 M_y (elastic)", "1,2 My": "1.2 M_y", "1,45 My": "1.45 M_y", "odlehčení": "unloading"}
_PL2_REST = {"křivost uprostřed se znaménkem (moment–křivost)": "mid-span curvature, signed (moment–curvature relation)",
             "σ_xx uprostřed – RMS odchylka od analytiky / Re": "σ_xx at mid-span – RMS deviation from the closed form / R_e",
             "pružně – PEEQ = 0": "PEEQ = 0", "PEEQ povrch": "PEEQ at the surface",
             "zbytková křivost uprostřed se znaménkem": "residual curvature at mid-span, signed",
             "zbytkové σ_xx – RMS odchylka od analytiky / Re": "residual σ_xx – RMS deviation from the closed form / R_e",
             "PEEQ povrch = z zatěžování": "surface PEEQ = value reached in the loading",
             "zbytkové σ_xx horní vlákno (Hodge, tlak)": "residual σ_xx, top fibre (compression)",
             "zbytkové σ_xx dolní vlákno (Hodge, tah)": "residual σ_xx, bottom fibre (tension)"}
_PL4_HARD = {"kinematické": "kinematic hardening", "izotropní": "isotropic hardening"}
_PL4_STEP = {"+325 MPa": "+325 MPa", "Odlehčení 1": "unloading 1", "−325 MPa": "−325 MPa", "Odlehčení 2": "unloading 2"}
_PL4_WHAT = {"přetvoření u/L": "strain u/L", "PEEQ": "PEEQ"}

PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    _combo(_V10_TAGS, _V10_REST),
    _combo(_V11_TAGS, _V11_REST),
    _combo(_V13_TAGS, _V13_REST),
    _sub(r"^FB1 rez x=(\d+) Vz$", r"cut at x = \1 mm: shear force Vz"),
    _sub(r"^FB1 rez x=(\d+) My k tezisti$", r"cut at x = \1 mm: bending moment My about the centroid"),
    _sub(r"^FB1 rez x=(\d+) N$", r"cut at x = \1 mm: axial force N (= 0)"),
    _sub(r"^submodel max σ1 vs Peterson/Heywood Kt,net = ([\d.]+), h=([\d.]+)mm$",
         r"submodel max σ1 vs Peterson / Heywood K_t,net = \1 (h = \2 mm)"),
    _sub(r"^heat through the interface \S+ → \S+ \(own integration\)$", r"heat through the interface (own integration)"),
    _sub(r"^\|u\|/u_st při ([\d.]+) Hz \(kvazistatika\)$", r"|u|/u_st at \1 Hz (quasi-static)"),
    _sub(r"^ζ z log\. dekrementu \((\d+) period\)$", r"ζ from the logarithmic decrement (\1 periods)"),
    _sub(r"^f₁\(N\)/f₁\(0\) = √\(1 \+ N/N_kr\), N = (\S+)·N_kr$", r"f₁(N)/f₁(0) = √(1 + N/N_cr), N = \1·N_cr"),
    (re.compile(r"^hot-spot v patě (vpravo|vlevo) \(x = ([\d.]+)\) = σ nosníku$"),
     lambda m: f"hot-spot stress at the {'right' if m[1] == 'vpravo' else 'left'} weld toe (x = {m[2]} mm) = beam σ"),
    _sub(r"^MYSTRAN \(4 prvky CBAR\) × vlastní řešič: λ(\d)$", r"MYSTRAN (4 CBAR elements) vs own beam solver: λ\1"),
    _sub(r"^(Ft|Fr): posuv konce (ux|uy) \(Castigliano, 0,5°\)$", r"\1: end displacement \2 (Castigliano, 0.5° segments)"),
    _sub(r"^(Ft|Fr): posuv konce uy \(Castigliano, 2° – chyba tětiv\)$", r"\1: end displacement uy (Castigliano, 2° segments – chord error)"),
    _sub(r"^(\d+) prvků: průhyb konce = 1 prvek$", r"\1 elements: tip deflection = 1 element"),
    _sub(r"^(\d+) prvků: M kořen = F·L$", r"\1 elements: root moment = F·L"),
    _combo(_PL2_TAGS, _PL2_REST),
    _sub(r"^ND (\d): ohyb prstence v rovině \(Love\)$", r"nodal diameter \1: in-plane ring bending (Love)"),
    (re.compile("^(" + "|".join(_PL4_HARD) + ") (" + "|".join(re.escape(k) for k in _PL4_STEP) + "): (" + "|".join(_PL4_WHAT) + ")$"),
     lambda m: f"{_PL4_HARD[m[1]]}, {_PL4_STEP[m[2]]}: {_PL4_WHAT[m[3]]}"),
]


def label(case: str, name: str) -> str:
    """English label of the check `name` of the case `case` (the name unchanged when nothing is known)."""
    if name in LABELS:
        return LABELS[name]
    for rx, fn in PATTERNS:
        m = rx.match(name)
        if m:
            return fn(m)
    stripped = _CASE_PREFIX.sub("", name, count=1)
    if stripped != name:
        return LABELS.get(stripped) or next((fn(m) for rx, fn in PATTERNS if (m := rx.match(stripped))), stripped)
    return name

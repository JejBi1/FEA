"""Cross-orthogonality (COC), auto-orthogonality (AOC) and modulus difference of the test correlation (F14 stage S5, ECSS-E-ST-32-11C, 31 July 2008).

Clauses (checked against the document):
  3.2.39            AOC_rs = phi_r,j^T M phi_s,j / sqrt((phi_r,j^T M phi_r,j)(phi_s,j^T M phi_s,j)) (one set j) and COC_rs = phi_r,j^T M phi_s,k / sqrt(...) (analytical j, measured k), with the
                    mass matrix of the mathematical model (FEM or TAM) as the weighting; "0 <= COC <= 1". The formula is NOT squared (the MAC is). A mode shape has an arbitrary sign, so
                    the program reports the magnitude |COC| (and |AOC|); complex test shapes use the Hermitian form, as the MAC does here
  Table 5-2         "Orthogonality or cross-orthogonality" and the "modulus difference" are MANDATORY techniques (5.8.1.1 b), next to the MAC and the frequency difference
  5.8.1.3 a.3, b    the cross-orthogonality is a general quality check of the FEM; "either the mass matrix shall be reduced to the number of the experimental DOFs, or the experimental mode
                    shapes shall be expanded to the number of the DOFs of the FEM" - either route serves the COC
  6.2.2 b           "The reduced FEM shall consist of a mass and stiffness matrix reduced to the number of measurement DOFs" (6.2.2 a NOTE 4: the reduced model is the test analysis model, TAM);
                    the reduction is the test laboratory's (Guyan / static, IRS, dynamic reduction, SEREP, ...). 6.3.2 b: the cross orthogonality between the TAM and the reference model is
                    checked "with the mass matrix of the reduced model" (needs the TAM stiffness for the TAM modes; the program has only the mass and does not do that comparison)
  5.7.1.4           auto-orthogonality: a. the measured shapes "shall be transformed to the number of the analytical degrees of freedom"; b. "the off-diagonal elements shall be less than 10 %"
  Table 5-3         example criteria (not normative, note a): COC diagonal terms > 0,90, off-diagonal terms < 0,10. Shown as HINTS (`LIMIT_HINTS`), never used as a default

THE RULE (owner decision 25. 11. 2026): only an imported TAM can PASS or FAIL a COC / AOC row. The SEREP route is INFORMATIVE.
  Four review rounds found that a SEREP reduction that the program makes from the FE modes keeps giving a COC that is not the COC of the shape: a test shape is judged through its projection onto
  the FE modes at the sensors, so a part of the shape the basis cannot represent is invisible, and which modes form the basis (the compared window, or every mode the sensors resolve) changes
  the number (a real ccx cantilever with a heavy tip mass: a test mode with 0.6 of an FE mode outside the window gives 0.9957 through the window and 0.80 through a wider basis; a mass-weighted
  error at a point the basis hardly moves is small in the Euclidean residual and large in the COC; uniaxial sensors resolve some modes only). No guard that the program can compute settles
  it. So the program still computes and shows the SEREP COC / AOC (with the out-of-span residual and the notes below), but every COC / AOC row of the SEREP route is UNCONFIRMED (never PASS and
  never FAIL: an informative number must not make a verdict either way; a value outside the limit is shown as a warning hint) with the reason `INDICATIVE_REASON`. A PASS / FAIL needs the TAM
  mass matrix of the test laboratory. The MAC, the frequency difference and the modulus difference are not affected.

The two sources of the reduced mass at the sensor DOFs (the user picks one: `CorrelationOptions.orth_mass`):
  serep   INDICATIVE (see above). Computed by the program from the solved model. The SEREP BASIS is the compared FE modes (`fe_modes`, the frequency window; Phi, all DOFs, max-normalised as the
          .frd). Phi_a = their sensor partition (a x m, a > m sensor DOFs, interpolated in the mesh), G = Phi^T M Phi the Gram matrix under the study's own mass (consistent mass of the bodies +
          surface mass + point masses, the same mass the dynamics module uses). The SEREP mass of the sensor DOFs is M_a = (Phi_a^+)^T G Phi_a^+ (Phi_a^+ = the pseudo-inverse); no stiffness and no
          solver run are needed. With q = Phi_a^+ psi (the modal coordinates of a test shape psi, least squares): COC(psi, FE mode j) = |q^H G e_j| / sqrt((q^H G q) G_jj), AOC(psi_i, psi_k) =
          |q_i^H G q_k| / sqrt((q_i^H G q_i)(q_k^H G q_k)). The FE modes are exactly orthogonal in M_a by construction, so the value that carries information is the COC of the TEST modes - and it
          is the COC of the PROJECTION of the shape onto the compared modes, not of the shape.
          What is printed (warnings, never holds): (1) the out-of-span RESIDUAL r_i = ||psi_i - Phi_a q_i||^2 / ||psi_i||^2 of every test mode (Euclidean norm over the sensor DOFs, Hermitian for
          complex shapes; the squared sine of the angle between the shape and the span; a warning above `SEREP_MAX_RESIDUAL` = 0.05); it does not bound the mass-weighted error either. (2) At
          least TWO compared FE modes and MORE sensor DOFs than FE modes are required (with one FE mode the COC is always 1; with as many DOFs as modes every shape is exactly representable): else
          no value. (3) The condition number of the mass-normalised sensor partition Phi_a diag(G)^(-1/2): no value above `SEREP_MAX_COND` = 1e8, a warning above `SEREP_WARN_COND` = 1e2 (noise
          eps on a shape moves a modal coordinate by up to cond * eps). (4) The FE modes must be orthogonal in the mass the program uses (largest off-diagonal of the normalised G, warning above
          `SEREP_MAX_FE_ORTH` = 1e-3: the mass of the study is then not the mass the modes were solved with). Not offered: Guyan / static reduction, IRS, dynamic reduction.
  tam     a test-analysis-model mass matrix of the user (CSV, `load_mass_file`): any reduction method, the DOFs must be exactly the sensor DOFs of the correlation, in the GLOBAL axes of the FE
          model (the test shapes are rotated into them; a TAM written in the test's local axes is a different matrix - with rotated sensors (UFF 2420) the rows are UNCONFIRMED until the
          user states that the TAM is in global axes, `CorrelationOptions.tam_global_axes`). COC = |psi^H M phi| / sqrt(...) with M the matrix of the file and phi the FE mode at the sensor
          DOFs. The FE-FE orthogonality through the same matrix is reported as a check of the TAM: a largest off-diagonal term above `TAM_MAX_FE_OFFDIAG` = 0.10 (the example off-diagonal of
          Table 5-3, not normative) is a warning and makes the rows UNCONFIRMED - the matrix does not reproduce the orthogonality of the FE modes, so it does not fit this model
          (the check between TAM modes and reference-model modes is 6.3.2 b, which needs the stiffness of the TAM and which the program does not do). The generalised masses of the FE modes through
          the TAM, diag(Phi_a^T M_tam Phi_a), must equal those of the study, diag(Phi^T M Phi), up to ONE global factor (`TAM_MAX_GENMASS_DEV` = 10 %, the size of the off-diagonal tolerance: an error
          of 10 % in a diagonal term moves the normalisation of a COC by about 5 %): a heavy payload left out of the matrix or modal masses rescaled mode by mode otherwise pass the orthogonality
          check; a deviation holds every row. What stays outside every check the program can make: a matrix that reproduces both the FE orthogonality and the FE generalised masses can still be
          wrong for other shapes (a SEREP matrix padded with a small regularisation is blind to everything outside the FE modes). The mass fidelity of the TAM is the laboratory's responsibility.
          An FE mode that the TAM does not see (zero energy at the sensor DOFs) holds the rows too. Further guards. A SINGULAR matrix (rank below the number of DOFs: a reduction
          with fewer modes than DOFs, e.g. another program's SEREP matrix, is of this kind) gives the part of a shape in its null space no mass at all - the blind spot of a projection - so the
          share of every test mode in the null space is printed (the same column as the SEREP residual) and the rows of a mode above `TAM_MAX_NULL_SHARE` = 0.05 are UNCONFIRMED (a compared
          FE mode with that share holds every row). The null space is told from the mass by the NOISE FLOOR OF THE FILE (`mass_rank_band`): the printed digits give a null space eigenvalues of
          both signs, so the band is `TAM_BAND_SAFETY` (10) times the most negative eigenvalue, clamped to [`TAM_BAND_MIN` = 1e-8, `TAM_RANK_TOL` = 1e-4] times the largest eigenvalue; an
          eigenvalue within the band is null space, a negative one beyond `TAM_RANK_TOL` is no mass (the matrix is refused). A file that prints a singular matrix to 6-7 digits often has no
          negative eigenvalue at all, so the band is ALSO at least the ROUNDING BOUND OF THE PRINTED DIGITS: the reader knows how many digits every cell was printed with, takes half a unit of
          the last printed digit of each cell (`rounding_bound`) and uses the Frobenius norm of that matrix of bounds as the noise floor (Weyl: no eigenvalue of the file differs from the
          true one by more than the spectral norm of the rounding, which that norm bounds). Null-space eigenvalues cannot exceed it, so a singular matrix printed to any number of digits is read
          as singular; the price is that a genuine eigenvalue below that floor (a spread beyond ~1e5 at 6 digits) is also read as null space - the rows are then held, never passed. A full-rank
          matrix with a spread up to ~1e5 printed to 6 digits keeps reading full rank. A file whose longest number has at most three significant digits is taken as exact (hand-written). With ONE compared FE mode the check above has no off-diagonal term, so the fit of the matrix to the model cannot be verified: the rows are UNCONFIRMED
          (a rank-1 matrix would give COC = 1 to any shape). A matrix is also bound to the model it was imported for (`model_fingerprint`: densities, added masses, mesh): another model holds the rows
A test mode that has a DOF not measured (NaN) gets no COC / AOC (a reduced mass cannot be cut down to fewer DOFs), and its rows are UNCONFIRMED. When no reduced mass is available the
rows of the orthogonality checks are UNCONFIRMED (never PASS, never silently left out), unless the user declared the criterion not applicable.

Modulus difference (Table 5-2). The standard names the technique and gives no formula. The program defines it (accepted by the owner of the program, 25. 11. 2026), and says so in the report: both shapes are scaled to unit length on the
DOFs measured in the test mode, and for a paired mode MD = max_d | |psi_test,d| - |phi_FE,d| | / max_d |psi_test,d| (a fraction of the largest test amplitude; the DOF of the maximum is
named). It does not depend on the sign / phase of a shape or on the scale of either set. The limit is entered by the user like every other limit.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._corr_io import TRANSLATIONS, TestDataError, _num, _read_raw, clean_label, decode_bytes, parse_dof
from ._corr_units import parse_unit

__all__ = ["ORTH_MASS", "MassMatrix", "OrthMass", "OrthResult", "load_mass_file", "parse_mass_csv", "orthogonality", "modulus_difference", "serep_gram", "mass_model_fingerprint",
           "SEREP_MAX_COND", "SEREP_WARN_COND", "SEREP_MAX_RESIDUAL", "SEREP_MAX_FE_ORTH", "TAM_MAX_FE_OFFDIAG", "TAM_RANK_TOL", "TAM_BAND_MIN", "TAM_BAND_SAFETY", "TAM_MAX_NULL_SHARE", "TAM_MAX_GENMASS_DEV", "TAM_EXACT_DIGITS", "rounding_bound",
           "MAX_MASS_DOFS", "MAX_MASS_BYTES", "INDICATIVE_REASON", "mass_rank_band", "model_fingerprint", "gram_cached", "SEREP_GRAM_CACHE_BYTES", "SEREP_GRAM_STEP"]

ORTH_MASS = ("serep", "tam", "none")
# The reason of every COC / AOC row of the SEREP route (owner decision 25. 11. 2026, module docstring): the number is shown, the row is never PASS and never FAIL.
INDICATIVE_REASON = ("indicative (SEREP reduction from the FE modes) - a PASS needs a test-analysis-model (TAM) mass matrix from the test lab (ECSS-E-ST-32-11C 5.8.1.3 b, 6.3)")
SEREP_MAX_COND = 1e8                         # condition number (mass-normalised sensor partition) above which the SEREP mass is not even computed
SEREP_WARN_COND = 1e2                        # above it a warning (relative noise eps on a shape moves a modal coordinate by up to cond * eps)
SEREP_MAX_RESIDUAL = 0.05                    # out-of-span residual of a test mode above which a warning names it (the value is of the projection, not of the shape)
SEREP_MAX_FE_ORTH = 1e-3                     # largest off-diagonal term of the normalised Gram matrix above which the study's mass is not the mass of the modes (a warning)
TAM_MAX_FE_OFFDIAG = 0.10                    # largest FE-FE off-diagonal term through a TAM above which the TAM does not fit the model (Table 5-3 example, not normative)
TAM_RANK_TOL = 1e-4                          # upper clamp of the null-space band, and the size of a negative eigenvalue (x the largest eigenvalue) beyond which the matrix is no mass matrix
TAM_BAND_MIN = 1e-8                          # lower clamp of the null-space band (x the largest eigenvalue): a full-rank matrix with a wide mass spread has eigenvalues far above it
TAM_BAND_SAFETY = 10.0                       # the band is this many times the most negative eigenvalue (the noise of the printed digits is symmetric about zero)
TAM_MAX_NULL_SHARE = 0.05                    # share of a test mode in the null space of the TAM above which its rows are UNCONFIRMED (the tolerance of the SEREP residual)
TAM_MAX_GENMASS_DEV = 0.10                   # largest deviation (after one global factor) of the generalised mass of an FE mode through the TAM from that of the study (the size of TAM_MAX_FE_OFFDIAG)
TAM_EXACT_DIGITS = 3                         # a file whose longest number has at most this many significant digits is taken as exact (hand-written round numbers): no rounding bound
MAX_MASS_DOFS = 2000                         # DOFs of an imported mass matrix (the matrix is dense)
MAX_MASS_BYTES = 128 << 20                   # 2000 x 2000 numbers of 25 characters = 100 MB
MAX_MASS_LINE_CHARS = (MAX_MASS_DOFS + 1) * 32 + 128       # one row of the matrix: 2000 cells of up to 31 characters and a label
MAX_MASS_COMMENT_LINES = 200
_MASS_UNITS = {"t": 1e3, "tonne": 1e3, "tonnes": 1e3}      # "t" is no unit atom of `_corr_units` (ambiguous there); the mass unit only labels the file: COC does not depend on the scale of M


# ------------------------------------------------------------------ an imported test-analysis-model mass matrix
@dataclass
class MassMatrix:
    """A mass matrix at the measured DOFs (the TAM of 6.3): `dofs` = [(test node label, x|y|z)], `matrix` (n, n) symmetric positive semi-definite, `unit` the label of the file."""
    dofs: list[tuple[str, str]]
    matrix: np.ndarray
    source: str = ""
    unit: str = ""
    sha256: list[str] = field(default_factory=list)
    study_key: str = ""                     # the study the matrix was imported for (`corr_run.study_key`); a matrix is used only with that study
    model_key: str = ""                     # `model_fingerprint` of that study (densities, added masses, mesh) when the matrix was bound; a different one holds the rows
    rank: int | None = None                 # numerical rank from the spectrum read at the import (None = not known: computed when needed)
    noise: float = 0.0                      # rounding bound of the printed digits of the file (`rounding_bound`, Frobenius norm of the half-unit bounds); 0 = exact / not from a file

    def fingerprint(self) -> str:
        """SHA-256 of what was read: the DOF labels in order, the matrix, the unit label."""
        h = hashlib.sha256()
        h.update(json.dumps([[n, d] for n, d in self.dofs] + [self.unit]).encode("utf-8", "replace"))
        h.update(np.ascontiguousarray(self.matrix, dtype=np.float64).data.cast("B"))
        return h.hexdigest()

    def summary(self) -> dict[str, Any]:
        return {"source": self.source, "dofs": len(self.dofs), "nodes": len(dict.fromkeys(n for n, _ in self.dofs)), "unit": self.unit, "sha256": list(self.sha256),
                "fingerprint": self.fingerprint(), "rank": self.rank}


def mass_rank_band(ev: np.ndarray, noise: float = 0.0) -> tuple[float, int]:
    """(band, rank) of a symmetric mass matrix from its eigenvalues `ev` (ascending): the null space is told from the mass by the NOISE FLOOR OF THE FILE. The printed digits of a rank-deficient
    matrix leave null-space eigenvalues of both signs, so the band is `TAM_BAND_SAFETY` times the most negative eigenvalue, clamped to [`TAM_BAND_MIN`, `TAM_RANK_TOL`] times the largest one, and at
    least `noise` = the rounding bound of the printed digits (`rounding_bound`: Weyl - no eigenvalue moves by more than the norm of the rounding, so a null-space eigenvalue is below it even when
    the rounding happened to leave no negative one). An eigenvalue above the band is mass, the others are null space. A full-rank matrix with a wide mass spread has no negative eigenvalue and
    a small noise, so its band is the lower clamp. `ValueError` for a matrix that is no mass matrix: no positive eigenvalue, a negative one beyond `TAM_RANK_TOL` times the largest and beyond `noise`
    (genuinely indefinite, not rounding), or digits so few that no eigenvalue stands above their rounding."""
    ev = np.asarray(ev, dtype=np.float64)
    top = float(ev[-1]) if ev.size else 0.0
    if not math.isfinite(top) or top <= 0:
        raise ValueError("no positive eigenvalue")
    noise = max(float(noise), 0.0) if math.isfinite(noise) else 0.0
    neg = max(-float(ev[0]), 0.0)
    if neg > max(TAM_RANK_TOL * top, noise):
        raise ValueError("a negative eigenvalue beyond the noise floor")
    band = max(min(max(TAM_BAND_SAFETY * neg, TAM_BAND_MIN * top), TAM_RANK_TOL * top), noise)
    rank = int(np.count_nonzero(ev > band))
    if rank == 0:
        raise ValueError("the printed digits do not resolve the mass above their rounding")
    return band, rank


def _scan_digits(cells: list[str], state: list[int]) -> None:
    """Accumulates into `state` = [longest number in significant digits, decimals of a fixed-point file (-1 = not fixed-point, -2 = not seen yet)] what the cells of one row tell about the
    digits the file was printed with. A cell of a "%g" file has no trailing zeros, a fixed-point / exponent cell keeps them; "D" exponents are read."""
    p, fixed = state
    for c in cells:
        mant, _, ex = c.lower().replace("d", "e").partition("e")
        ip, dot, fp = mant.lstrip("+-").partition(".")
        if ex or not dot or not fp:
            fixed = -1
        elif fixed == -2:
            fixed = len(fp)
        elif fixed != len(fp):
            fixed = -1
        digs = (ip + fp).lstrip("0")
        if digs:
            n = len(digs) if (dot and fp) else len(digs.rstrip("0"))
            if n > p:
                p = n
    state[0], state[1] = p, fixed


def rounding_bound(mat: np.ndarray, digits: int, fixed_decimals: int = -1) -> float:
    """The noise floor of a matrix printed with `digits` significant digits (the longest number of the file): the Frobenius norm of the matrix of per-cell rounding bounds, half a unit of the
    last printed digit - 0.5 x 10^(decade of the cell - (digits - 1)) in a "%g" / "%e" file, 0.5 x 10^-`fixed_decimals` for every cell of a fixed-point file. By Weyl's inequality no eigenvalue
    of the printed matrix differs from that of the true one by more than the spectral norm of the rounding, which this norm bounds. 0 for a file taken as exact (at most `TAM_EXACT_DIGITS`
    significant digits and at most that many decimals: hand-written round numbers)."""
    if digits <= TAM_EXACT_DIGITS and fixed_decimals <= TAM_EXACT_DIGITS:
        return 0.0
    a = np.abs(np.asarray(mat, dtype=np.float64))
    if fixed_decimals >= 0:
        e = np.full(a.shape, 0.5 * 10.0 ** (-fixed_decimals))
    else:
        e = np.zeros(a.shape)
        nz = a > 0
        e[nz] = 0.5 * 10.0 ** (np.floor(np.log10(a[nz]) + 1e-9) - (digits - 1))        # the +1e-9: a cell on a decade edge takes the upper decade (a larger bound)
    e = 0.5 * (e + e.T)
    return float(np.linalg.norm(e))


def _label(cell: str, where: str) -> tuple[str, str]:
    node, sep, dof = cell.strip().rpartition(":")
    if not sep or not node.strip():
        raise TestDataError(f"{where}: the DOF label '{clean_label(cell, 30)}' must be node:dof (e.g. N1:x)")
    d, sign = parse_dof(dof)
    if sign < 0:
        raise TestDataError(f"{where}: a signed DOF ('{clean_label(cell, 30)}') is not allowed in a mass matrix")
    if d not in TRANSLATIONS:
        raise TestDataError(f"{where}: only the translations x, y, z are compared with the test, not '{clean_label(cell, 30)}'")
    return clean_label(node), d


def _mass_lines(text: str) -> list[str]:
    """The lines of a mass-matrix file under the bounds of THIS reader (a row of the matrix is much longer than a line of a test file): the number of lines, the length of a line."""
    if len(text) > MAX_MASS_BYTES * 2:
        raise TestDataError("The mass matrix file is too large")
    lines = text.splitlines()
    if len(lines) > MAX_MASS_DOFS + 1 + MAX_MASS_COMMENT_LINES + 1000:
        raise TestDataError(f"The mass matrix file has too many lines (at most {MAX_MASS_DOFS} DOFs)")
    for i, ln in enumerate(lines, 1):
        if len(ln) > MAX_MASS_LINE_CHARS:
            raise TestDataError(f"Line {i} of the mass matrix file is too long (at most {MAX_MASS_LINE_CHARS} characters)")
    return lines


def _cells(line: str, delim: str) -> list[str]:
    import csv
    try:
        return next(csv.reader([line], delimiter=delim), [])
    except csv.Error as e:
        raise TestDataError(f"The mass matrix file is malformed: {str(e)[:80]}") from None


def parse_mass_csv(text: str, source: str = "") -> MassMatrix:
    """CSV of a square mass matrix. Optional comment lines `# ...` (`# unit: kg` states the mass unit). The first row is the header: an empty first cell and the DOF labels
    `node:dof` (x, y, z); every following row starts with the same label and holds the row of the matrix. The labels are those of the test data (the node labels of the modes file).
    The matrix is in the GLOBAL axes of the FE model (see the module docstring). Refused BEFORE the body is parsed: more than 2000 DOFs in the header, a number of rows that is not the
    number of columns; afterwards a matrix that is not symmetric (relative asymmetry above 1e-6), has a non-finite number, a repeated DOF or is not positive semi-definite."""
    unit = ""
    body: list[str] = []
    for ln in _mass_lines(text):
        t = ln.strip()
        if not t:
            continue
        if t.startswith("#"):
            m = re.match(r"#\s*unit\s*[:=]\s*(\S+)", t, re.IGNORECASE)
            if m:
                unit = clean_label(m.group(1), 20)
            continue
        body.append(ln)
    if len(body) < 2:
        raise TestDataError("The mass matrix file needs a header row and at least one matrix row")
    delim = max((",", ";", "\t"), key=body[0].count)
    head = [c.strip() for c in _cells(body[0], delim)]
    n = len(head) - 1
    if n < 1 or n > MAX_MASS_DOFS:
        raise TestDataError(f"The mass matrix must have 1 to {MAX_MASS_DOFS} DOFs (the header names {max(n, 0)})")
    if len(body) - 1 != n:
        raise TestDataError(f"The mass matrix must be square: {n} DOF columns but {len(body) - 1} rows")
    dofs = [_label(c, "header") for c in head[1:]]
    if len(set(dofs)) != n:
        raise TestDataError("A DOF appears twice in the header of the mass matrix")
    mat = np.zeros((n, n))
    digits = [0, -2]                                                       # longest number (significant digits), decimals of a fixed-point file: what the file was printed with
    for i, ln in enumerate(body[1:]):
        r = _cells(ln, delim)
        if len(r) != n + 1:
            raise TestDataError(f"Row {i + 2} of the mass matrix has {len(r)} cells, expected {n + 1}")
        if _label(r[0], f"row {i + 2}") != dofs[i]:
            raise TestDataError(f"Row {i + 2} is labelled '{clean_label(r[0], 30)}' but the header column {i + 1} is '{dofs[i][0]}:{dofs[i][1]}': the rows must follow the order of the columns")
        cells = [c.strip() for c in r[1:]]
        try:
            mat[i] = np.array(cells, dtype=np.float64)                  # fast path (a 2000 x 2000 file has 4 million cells)
        except ValueError:
            for j, c in enumerate(cells):
                mat[i, j] = _num(c, "a matrix entry", f"row {i + 2}")
        _scan_digits(cells, digits)
    if not np.all(np.isfinite(mat)):
        raise TestDataError("The mass matrix holds a number that is not finite")
    scale = float(np.abs(mat).max())
    if scale <= 0:
        raise TestDataError("The mass matrix is zero")
    if float(np.abs(mat - mat.T).max()) > 1e-6 * scale:
        raise TestDataError("The mass matrix is not symmetric")
    mat = 0.5 * (mat + mat.T)
    ev = np.linalg.eigvalsh(mat)
    noise = rounding_bound(mat, digits[0], digits[1])
    try:
        rank = mass_rank_band(ev, noise)[1]
    except ValueError as e:
        if "digits" in str(e):
            raise TestDataError("The numbers of the mass matrix are printed with too few digits: no eigenvalue stands above the rounding of the printed digits - export it with more digits") from None
        raise TestDataError("The mass matrix is not positive semi-definite (a mass matrix cannot have a negative eigenvalue beyond the rounding of its digits)") from None
    if unit:
        if unit.lower() not in _MASS_UNITS:
            u = parse_unit(unit)
            if u is None or u.dims != (0, 0, 1):
                raise TestDataError(f"The unit '{unit}' of the mass matrix is not a mass unit (use kg, t, Mg ...)")
    return MassMatrix(dofs, mat, source, unit, rank=rank, noise=noise)


def load_mass_file(path: str | Path) -> MassMatrix:
    """Reads a mass-matrix CSV (bounded: 128 MB, 2000 DOFs); UNC paths are refused."""
    raw = _read_raw(path, MAX_MASS_BYTES)
    mm = parse_mass_csv(decode_bytes(raw), clean_label(Path(str(path)).name, 80))
    mm.sha256 = [hashlib.sha256(raw).hexdigest()]
    return mm


# ------------------------------------------------------------------ the source of the reduced mass
def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode("ascii")).hexdigest()


def mass_model_fingerprint(study: Any, mesh: Any) -> str:
    """SHA-256 of what the mass of the SEREP reduction is made of besides the mesh: the density of every element and the added masses (point and surface) of the study."""
    from ..dynamics.mass import mass_items
    from ..printing import element_density
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(element_density(study, mesh), dtype=np.float64).data.cast("B"))
    h.update(_digest([asdict(m) if is_dataclass(m) and not isinstance(m, type) else repr(m) for m in mass_items(study)]).encode("ascii"))
    return h.hexdigest()


def model_fingerprint(study: Any, mesh: Any) -> str:
    """SHA-256 of the model a mass matrix (TAM) belongs to: the densities of the elements, the added masses and the mesh (nodes, elements). "" when it cannot be formed (no mesh, a
    mesh that is not tet10). A matrix that was imported for one model must not be applied silently to another (a changed density or point mass changes the true mass matrix)."""
    if study is None or mesh is None:
        return ""
    try:
        h = hashlib.sha256(mass_model_fingerprint(study, mesh).encode("ascii"))
        for name in ("nodes", "tets"):
            a = np.ascontiguousarray(getattr(mesh, name))
            h.update(repr((name, a.shape, str(a.dtype))).encode("ascii"))
            h.update(a.data.cast("B"))                                  # no transient copy of the connectivity (a million-node mesh has tens of MB)
        return h.hexdigest()
    except (ValueError, KeyError, AttributeError, TypeError, IndexError):
        return ""


@dataclass
class OrthMass:
    """Where the reduced mass of the orthogonality checks comes from: `kind` serep (the program: needs `study` and `mesh`; INFORMATIVE only) | tam (an imported matrix; `study` and `mesh` serve the
    check of its generalised masses) | none (not computed, `why`)."""
    kind: str = "none"
    study: Any = None
    mesh: Any = None
    tam: MassMatrix | None = None
    gram: np.ndarray | None = None          # serep with a given Gram matrix G = Phi^T M Phi of FE modes (tests, verification cases): no study / mesh needed
    gram_modes: list[int] | None = None     # the 0-based mode indices (of the mode set) the rows and columns of `gram` belong to; None = `gram` is exactly the compared modes, in their order (m x m)
    gen_mass: np.ndarray | None = None      # tam: the generalised masses of the compared FE modes under the mass of the study, given directly (tests, verification cases); else formed from `study` / `mesh`
    model_hold: str = ""                    # tam: why the matrix does not belong to this model (its model fingerprint differs): holds every row
    why: str = "no mass for the orthogonality checks was given"

    def fingerprint(self) -> str:
        try:
            if self.kind == "serep" and self.gram is not None:
                h = hashlib.sha256(np.ascontiguousarray(self.gram, dtype=np.float64).data.cast("B"))
                h.update(json.dumps(self.gram_modes).encode("ascii"))
                return h.hexdigest()
            if self.kind == "serep" and self.study is not None and self.mesh is not None:
                return mass_model_fingerprint(self.study, self.mesh)
            if self.kind == "tam" and self.tam is not None:
                return self.tam.fingerprint()
        except (ValueError, KeyError, AttributeError, TypeError):
            return ""
        return ""

    def label(self) -> str:
        return {"serep": "SEREP reduction of the FE modes (mass of the study), indicative", "tam": f"imported mass matrix {self.tam.source if self.tam is not None else ''}".strip(),
                "none": "none"}.get(self.kind, self.kind)


SEREP_GRAM_CACHE_BYTES = 128 << 20      # the compared modes are converted to double precision ONCE and kept while they take at most this much (else they are converted block by block)
SEREP_GRAM_STEP = 8                     # modes per `mass_times` call (its element batches grow with the number of modes: more at once costs RAM, not time)


def gram_cached(n_nodes: int, n_modes: int) -> bool:
    """True when the `n_modes` compared modes at `n_nodes` nodes fit `SEREP_GRAM_CACHE_BYTES` in double precision (n_nodes x 3 x 8 bytes per mode): they are converted once."""
    return max(n_nodes, 1) * 3 * 8 * max(n_modes, 1) <= SEREP_GRAM_CACHE_BYTES


def serep_gram(study: Any, mesh: Any, modeset: Any, idx: list[int]) -> np.ndarray:
    """G = Phi^T M Phi (m, m) of the FE modes `idx` of `modeset` (the shapes as stored: max-normalised) under the mass of the study: the consistent mass of the bodies, the surface mass
    and the point masses (as `dynamics.modal.check_basis` counts them). Tet10 meshes only (`ValueError` otherwise). Cost: the modes are converted to double precision ONCE (kept while
    they take at most `SEREP_GRAM_CACHE_BYTES`; `M Phi` is formed `SEREP_GRAM_STEP` modes at a time to bound the RAM of `mass_times`), only the upper triangle is computed and mirrored."""
    from ..dynamics.mass import mass_items, mass_points, mass_times, node_motion, surface_mass_nodes
    from ..dynamics.model import SurfaceMass
    shapes = np.asarray(modeset.shapes)
    n = len(mesh.nodes)
    if shapes.ndim != 3 or shapes.shape[1] != n:
        raise ValueError("the mode shapes do not belong to this mesh")
    ids = np.asarray(idx, dtype=np.int64)
    m = len(ids)
    if m == 0 or len(set(ids.tolist())) != m or ids.min() < 0 or ids.max() >= shapes.shape[0]:
        raise ValueError("the FE mode indices are not valid")
    G = np.zeros((m, m))
    surf = [(surface_mass_nodes(mesh, it)) for it in mass_items(study) if isinstance(it, SurfaceMass)]
    pts = mass_points(study, mesh)
    flat_all = shapes[ids].astype(np.float64).reshape(m, -1) if gram_cached(n, m) else None            # the one conversion (small and medium models)
    step = SEREP_GRAM_STEP
    for j0 in range(0, m, step):
        jb = ids[j0:j0 + step]
        end = j0 + len(jb)
        V = flat_all[j0:end].reshape(len(jb), n, 3) if flat_all is not None else shapes[jb].astype(np.float64)
        MV = mass_times(study, mesh, V)
        for sidx, smass in surf:
            MV[:, sidx] += smass[None, :, None] * V[:, sidx]
        flat = MV.reshape(len(jb), -1)
        if flat_all is not None:
            G[:end, j0:end] = flat_all[:end] @ flat.T                         # the rows up to the diagonal block only: the upper triangle (the rest is the mirror)
        else:
            G[j0:end, j0:end] = V.reshape(len(jb), -1) @ flat.T
            for i0 in range(0, j0, step):
                ib = ids[i0:i0 + step]
                G[i0:i0 + len(ib), j0:end] = shapes[ib].astype(np.float64).reshape(len(ib), -1) @ flat.T
        del V, MV, flat
    if pts:
        mots: list[list[np.ndarray]] = [[] for _ in pts]
        for q, k in enumerate(ids):
            vk = flat_all[q].reshape(n, 3) if flat_all is not None else shapes[k].astype(np.float64)
            for pi, info in enumerate(pts):
                mots[pi].append(node_motion(mesh, info, vk))
        for pi, info in enumerate(pts):
            w = np.array([mu for _, mu in info["nodes"]], float)
            mot = np.stack(mots[pi])                                      # (m, points, 3)
            G += np.einsum("ipc,jpc,p->ij", mot, mot, w)                  # the point masses (small: the number of points is the number of mass items)
    G = np.triu(G) + np.triu(G, 1).T                                      # the blocks hold the upper triangle; the point-mass term is symmetric: the mirror is exact
    if not np.all(np.isfinite(G)) or np.any(np.diag(G) <= 0):
        raise ValueError("the generalised masses of the FE modes are not positive (check the densities and the added masses of the study)")
    return G


# ------------------------------------------------------------------ the checks
@dataclass
class OrthResult:
    """COC (n_test, n_fe) and AOC (n_test, n_test) of the test modes, NaN for a test mode that is not complete on the DOFs. `problems` = why they were not computed at all (arrays None).
    `residual` (n_test) = the out-of-span residual of every test mode (SEREP) or its share in the null space of the TAM, NaN for a mode without COC. `indicative` = the reason that every
    COC / AOC row is UNCONFIRMED because the value is only indicative (the SEREP route; "" for a TAM). `hold` = why EVERY COC / AOC row is UNCONFIRMED although the numbers exist (a TAM that
    does not fit, axes not confirmed); `mode_hold` = test index -> why the rows that involve that test mode are UNCONFIRMED (a TAM that gives its shape no mass)."""
    source: str = "none"
    label: str = "none"
    fingerprint: str = ""
    coc: np.ndarray | None = None
    aoc: np.ndarray | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    residual: np.ndarray | None = None
    hold: str = ""
    mode_hold: dict[int, str] = field(default_factory=dict)
    indicative: str = ""

    @property
    def computed(self) -> bool:
        return self.coc is not None and self.aoc is not None

    def hold_of(self, test_indices: Any) -> str:
        """Why the rows that involve these test modes (indices) are UNCONFIRMED although the numbers exist; "" = they may be judged."""
        parts = [self.indicative] if self.indicative else []
        parts += [self.hold] if self.hold else []
        parts += [self.mode_hold[i] for i in sorted(set(test_indices)) if i in self.mode_hold]
        return "; ".join(dict.fromkeys(parts))

    def to_dict(self, matrices: bool = True, limit: int = 10_000) -> dict[str, Any]:
        d: dict[str, Any] = {"source": self.source, "mass": self.label, "fingerprint": self.fingerprint, "computed": self.computed, "problems": list(self.problems),
                             "notes": list(self.notes), "info": dict(self.info), "hold": self.hold, "indicative": self.indicative, "mode_hold": {str(k): v for k, v in sorted(self.mode_hold.items())}}
        if self.residual is not None:
            d["residual"] = [None if not math.isfinite(float(v)) else round(float(v), 5) for v in self.residual]
        if matrices and self.coc is not None and self.coc.size <= limit:
            d["coc"] = [[None if not math.isfinite(float(v)) else round(float(v), 5) for v in row] for row in self.coc]
        if matrices and self.aoc is not None and self.aoc.size <= limit:
            d["aoc"] = [[None if not math.isfinite(float(v)) else round(float(v), 5) for v in row] for row in self.aoc]
        return d


def _normalised(num: np.ndarray, e_row: np.ndarray, e_col: np.ndarray) -> np.ndarray:
    """|num| / sqrt(e_row e_col); NaN when the denominator is not positive or the value exceeds 1 beyond rounding (impossible for a positive semi-definite mass: the inputs are inconsistent)."""
    den = np.sqrt(np.outer(np.maximum(e_row, 0.0), np.maximum(e_col, 0.0)))               # a rounding-level negative energy (a null-space direction) is "no mass", not a NaN warning
    out = np.full(num.shape, np.nan)
    ok = np.isfinite(den) & (den > 0)
    v = np.abs(num[ok]) / den[ok]
    out[ok] = np.where(v > 1.0 + 1e-6, np.nan, np.minimum(v, 1.0))
    return out


def _names(numbers: list[int], limit: int = 8) -> str:
    """'3, 5, 9 ...': the numbers of the modes a message is about (never a bare count)."""
    return ", ".join(str(x) for x in numbers[:limit]) + (" ..." if len(numbers) > limit else "")


def orthogonality(test_phi: np.ndarray, fe_phi: np.ndarray, dofs: list[tuple[str, str]], mass: OrthMass | None, modeset: Any = None, fe_idx: list[int] | None = None,
                  test_numbers: list[int] | None = None, axes_hold: str = "") -> OrthResult:
    """COC and AOC of the test modes `test_phi` (n_dof, n_test; NaN = not measured) with the FE modes `fe_phi` (n_dof, n_fe) at the same DOFs, through the reduced mass `mass`.
    `test_numbers` = the mode numbers of the test modes (for the messages), `axes_hold` = a reason that makes every row UNCONFIRMED (the TAM axes are not confirmed). `fe_idx` = the 0-based
    indices of the compared modes in the mode set (the SEREP Gram matrix is formed for them; the FE mode numbers of the messages are `fe_idx + 1`).
    The SEREP route is INFORMATIVE: `OrthResult.indicative` makes every row UNCONFIRMED (module docstring). Never raises for a missing / unusable mass: the reason is in `problems` and the
    caller makes the rows UNCONFIRMED."""
    tp = np.asarray(test_phi)
    fp = np.asarray(fe_phi, dtype=np.float64)
    if mass is None or mass.kind == "none":
        return OrthResult(problems=[(mass.why if mass is not None else "no mass for the orthogonality checks was given") or "no mass for the orthogonality checks was given"])
    res = OrthResult(source=mass.kind, label=mass.label(), fingerprint=mass.fingerprint(), indicative=INDICATIVE_REASON if mass.kind == "serep" else "")
    finite = np.isfinite(tp.real) & np.isfinite(tp.imag) if np.iscomplexobj(tp) else np.isfinite(tp)
    ok = finite.all(axis=0)
    n_t, n_f = tp.shape[1], fp.shape[1]
    nums = list(test_numbers) if test_numbers is not None and len(test_numbers) == n_t else list(range(1, n_t + 1))
    fe_nums = [int(k) + 1 for k in fe_idx] if fe_idx is not None and len(fe_idx) == n_f else list(range(1, n_f + 1))
    if not ok.any():
        res.problems.append("every test mode lacks a measured DOF: the reduced mass cannot be applied (it is defined for all sensor DOFs together)")
        return res
    if not ok.all():
        gone = [nums[i] for i in np.flatnonzero(~ok)]
        res.notes.append(f"test mode(s) {_names(gone)} have a DOF that was not measured: no COC / AOC for them (a reduced mass cannot be cut down to fewer DOFs)")
    psi = tp[:, ok]
    coc = np.full((n_t, n_f), np.nan)
    aoc = np.full((n_t, n_t), np.nan)
    try:
        if mass.kind == "serep":
            e_t, e_f, num_c, num_a, info, extra = _serep(psi, fp, mass, modeset, fe_idx)
        elif mass.kind == "tam":
            e_t, e_f, num_c, num_a, info, extra = _tam(psi, fp, dofs, mass, fe_nums, modeset, fe_idx)
        else:
            res.problems.append(f"unknown mass source '{mass.kind}'")
            return res
    except _Unavailable as e:
        res.problems.append(str(e))
        return res
    where = np.flatnonzero(ok)
    holds: list[str] = list(extra.get("holds", []))
    if axes_hold and mass.kind == "tam":
        holds.append(axes_hold)
    if mass.kind == "tam" and mass.model_hold:
        holds.append(mass.model_hold)
    res.hold = "; ".join(dict.fromkeys(holds))
    res.warnings = list(extra.get("warnings", [])) + ([axes_hold] if axes_hold and mass.kind == "tam" else []) + ([mass.model_hold] if mass.kind == "tam" and mass.model_hold else [])
    res.notes += list(extra.get("notes", []))
    resid = extra.get("residual")
    if resid is not None:
        full = np.full(n_t, np.nan)
        full[where] = resid
        res.residual = full
        for k, why in extra.get("mode_holds", {}).items():
            res.mode_hold[int(where[k])] = f"test mode {nums[where[k]]} {why}"
        if res.mode_hold:
            res.warnings.append(f"{len(res.mode_hold)} test mode(s) have more than {100 * TAM_MAX_NULL_SHARE:.0f} % of their shape where the reduced mass cannot judge it "
                                f"(test modes {_names([nums[i] for i in sorted(res.mode_hold)])}): their COC / AOC are not those of the shape, their rows are UNCONFIRMED (the MAC judges the whole shape)")
        if mass.kind == "serep":
            over = [nums[where[k]] for k in range(len(resid)) if math.isfinite(float(resid[k])) and float(resid[k]) > SEREP_MAX_RESIDUAL]
            if over:
                res.warnings.append(f"test mode(s) {_names(over)} have more than {100 * SEREP_MAX_RESIDUAL:.0f} % of their shape outside the span of the compared FE modes at the sensors: their "
                                    "indicative COC / AOC are those of the projection onto that span, not of the shape (the MAC judges the whole shape)")
    coc[where, :] = _normalised(num_c, e_t, e_f)
    aoc[np.ix_(where, where)] = _normalised(num_a, e_t, e_t)
    sub = aoc[np.ix_(where, where)]
    dn = np.isnan(np.diag(sub))                                               # a test mode whose own energy is unusable (a zero shape) ...
    pair_bad = (np.isnan(sub) & ~dn[:, None] & ~dn[None, :]).any(axis=1)      # ... or one whose cross term with another usable mode is NaN (a value above 1)
    bad_rows = [nums[i] for i in where[dn | pair_bad | np.isnan(coc[where]).any(axis=1)]]
    if bad_rows:
        res.notes.append(f"COC / AOC value(s) of test mode(s) {_names(bad_rows)} could not be computed (a zero shape or a value above 1 beyond rounding: the shape and the mass are inconsistent)")
    res.coc, res.aoc, res.info = coc, aoc, info
    return res


class _Unavailable(Exception):
    """The reduced mass cannot be made / applied (the message says why)."""


def _serep_gram_of(mass: OrthMass, modeset: Any, fe_idx: list[int] | None, m: int) -> np.ndarray:
    """The Gram matrix of the compared FE modes: the given one (`mass.gram`, in the order of the comparison, or with its mode indices named in `mass.gram_modes`) or the one computed
    from the study."""
    if mass.gram is None:
        if mass.study is None or mass.mesh is None or modeset is None or fe_idx is None:
            raise _Unavailable("the SEREP mass needs the study, its mesh and the solved FE modes")
        try:
            return serep_gram(mass.study, mass.mesh, modeset, list(fe_idx))
        except (ValueError, KeyError, AttributeError, TypeError, IndexError) as e:
            raise _Unavailable(f"the mass of the FE modes could not be computed: {str(e)[:160]}") from None
    Gg = np.asarray(mass.gram, dtype=np.float64)
    if Gg.ndim != 2 or Gg.shape[0] != Gg.shape[1]:
        raise _Unavailable("the Gram matrix of the FE modes must be square")
    if mass.gram_modes is None:
        if Gg.shape != (m, m):
            raise _Unavailable(f"the Gram matrix of the FE modes is {Gg.shape[0]} x {Gg.shape[1]} but {m} FE modes are compared: a matrix of more modes needs the mode indices of its rows (OrthMass.gram_modes)")
        return Gg
    gm = [int(k) for k in mass.gram_modes]
    if len(gm) != Gg.shape[0] or len(set(gm)) != len(gm):
        raise _Unavailable("OrthMass.gram_modes must name each row of the Gram matrix once")
    want = list(fe_idx) if fe_idx is not None else list(range(m))
    pos = {k: i for i, k in enumerate(gm)}
    miss = [k + 1 for k in want if k not in pos]
    if miss:
        raise _Unavailable(f"the Gram matrix does not cover the compared FE mode(s) {_names(miss)}")
    sel = [pos[k] for k in want]
    return Gg[np.ix_(sel, sel)]


def _serep(psi: np.ndarray, fp: np.ndarray, mass: OrthMass, modeset: Any, fe_idx: list[int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """The INDICATIVE SEREP cross-orthogonality: the COC of the projection of the test shapes onto the compared FE modes at the sensors (module docstring)."""
    a, m = fp.shape
    if m < 2:
        raise _Unavailable("the SEREP cross-orthogonality needs at least two FE modes in the comparison: with one FE mode every test shape has COC = 1 with it (the COC is that of the projection onto "
                           "the FE modes), which says nothing about the shape; the MAC judges it")
    if a <= m:
        raise _Unavailable(f"the SEREP mass needs more sensor DOFs than FE modes ({a} DOFs, {m} FE modes): with as many DOFs as modes every test shape can be written exactly with the FE modes, "
                           "so the out-of-span residual could not show that it is not an FE mode - measure more DOFs or add sensors (dropping FE modes is no remedy: the COC would then "
                           "be that of the projection onto the remaining modes)")
    G = _serep_gram_of(mass, modeset, fe_idx, m)
    gd = np.diag(G)
    if not np.all(np.isfinite(gd)) or np.any(gd <= 0):
        raise _Unavailable("the generalised masses of the FE modes are not positive")
    d = np.sqrt(gd)
    s = np.linalg.svd(fp / d[None, :], compute_uv=False)                  # the mass-normalised sensor partition (the modal coordinates the COC works with)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    if not math.isfinite(cond) or cond > SEREP_MAX_COND:
        raise _Unavailable(f"the FE modes cannot be told apart at the sensor DOFs (condition number {cond:.3g} of the {a} x {m} mass-normalised shape matrix, limit {SEREP_MAX_COND:.0e}): "
                           "the sensors do not see all the FE modes of the comparison; add sensors")
    warns: list[str] = []
    if cond > SEREP_WARN_COND:
        warns.append(f"the sensors separate the {m} compared FE modes badly (condition number {cond:.3g} of the mass-normalised shape matrix, warning above {SEREP_WARN_COND:.0e}): measurement noise "
                     "is amplified into the modal coordinates of the indicative COC / AOC")
    Q = np.linalg.pinv(fp) @ psi                                          # (m, n_ok): the test shapes in the coordinates of the compared FE modes
    GQ = G @ Q
    e_t = np.real(np.sum(np.conj(Q) * GQ, axis=0))
    nrm2 = np.real(np.sum(np.conj(psi) * psi, axis=0))
    proj = fp @ Q
    proj2 = np.real(np.sum(np.conj(proj) * proj, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.where(nrm2 > 0, np.clip(1.0 - proj2 / nrm2, 0.0, 1.0), np.nan)
    orth_err = float((np.abs(G - np.diag(gd)) / np.sqrt(np.outer(gd, gd))).max())
    info: dict[str, Any] = {"fe_modes": m, "basis_modes": m, "sensor_dofs": a, "condition": cond, "fe_mass_orthogonality_error": orth_err, "indicative": True,
                            "residual_max": float(np.nanmax(residual)) if np.isfinite(residual).any() else None, "residual_limit": SEREP_MAX_RESIDUAL, "residual_kind": "out-of-span"}
    if orth_err > SEREP_MAX_FE_ORTH:
        warns.append(f"the FE modes are not mass-orthogonal under the mass of the study (largest off-diagonal {orth_err:.3g}, warning above {SEREP_MAX_FE_ORTH:.0e}): the study's mass differs from "
                     "the one the modes were solved with (densities, added masses edited after the solution?)")
    return e_t, np.diag(G).copy(), np.conj(Q).T @ G, np.conj(Q).T @ GQ, info, {"residual": residual, "warnings": warns}


def _study_gen_mass(mass: OrthMass, modeset: Any, fe_idx: list[int] | None, m: int) -> tuple[np.ndarray | None, str]:
    """(generalised masses diag(Phi^T M Phi) of the compared FE modes under the mass of the study, why they are not available). `(None, "")` = there is no study to compare with (programmatic use
    of a bare matrix): the check is not made. A study that is given but whose mass cannot be formed returns the reason (the rows are held)."""
    if mass.gen_mass is not None:
        g = np.asarray(mass.gen_mass, dtype=np.float64).ravel()
        if g.size != m or not np.all(np.isfinite(g)) or np.any(g <= 0):
            return None, f"the given generalised masses of the FE modes are not {m} positive numbers"
        return g, ""
    if mass.study is None or mass.mesh is None:
        return None, ""
    if modeset is None or fe_idx is None:
        return None, "the FE modes needed for the generalised masses are not at hand"
    try:
        return np.diag(serep_gram(mass.study, mass.mesh, modeset, list(fe_idx))).copy(), ""
    except (ValueError, KeyError, AttributeError, TypeError, IndexError) as e:
        return None, f"the generalised masses of the FE modes under the mass of the study could not be computed ({str(e)[:120]})"


def _tam(psi: np.ndarray, fp: np.ndarray, dofs: list[tuple[str, str]], mass: OrthMass, fe_nums: list[int], modeset: Any = None, fe_idx: list[int] | None = None
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    tam = mass.tam
    if tam is None:
        raise _Unavailable("no mass matrix (TAM) was imported")
    have = {d: i for i, d in enumerate(tam.dofs)}
    miss = [f"{n}:{d}" for n, d in dofs if (n, d) not in have]
    extra = [f"{n}:{d}" for n, d in tam.dofs if (n, d) not in set(dofs)]
    if miss or extra:
        raise _Unavailable("the DOFs of the imported mass matrix are not the sensor DOFs of the correlation"
                           + (f"; missing in the matrix: {', '.join(miss[:8])}{' ...' if len(miss) > 8 else ''}" if miss else "")
                           + (f"; not sensor DOFs: {', '.join(extra[:8])}{' ...' if len(extra) > 8 else ''}" if extra else ""))
    order = [have[d] for d in dofs]
    M = tam.matrix[np.ix_(order, order)]
    n = len(dofs)
    rank = tam.rank
    null_basis: np.ndarray | None = None
    if rank is None or rank < n:                                         # a matrix the reader did not vouch for, or one that is singular: the spectrum is needed
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        try:
            band, rank = mass_rank_band(w, tam.noise)                    # the SAME rule as the reader: the noise floor of the file; refuses a negative eigenvalue beyond it
        except ValueError:
            raise _Unavailable("the imported mass matrix is not positive semi-definite (a mass matrix cannot have a negative eigenvalue beyond the rounding of its digits)") from None
        if rank < n:
            keep = w > band
            null_basis = V[:, ~keep]
            M = (V[:, keep] * w[keep]) @ V[:, keep].T                    # the mass part only: the noise of the null space is no mass
    m_fe = fp.shape[1]
    Mf = M @ fp
    Mp = M @ psi
    e_f = np.sum(fp * Mf, axis=0)
    e_t = np.real(np.sum(np.conj(psi) * Mp, axis=0))
    visible = np.isfinite(e_f) & (e_f > 0)
    invisible = [fe_nums[j] for j in range(m_fe) if not visible[j]]      # an FE mode the matrix gives no energy at the sensor DOFs (zero amplitude there, or in the null space)
    fe_aoc = _normalised(fp.T @ Mf, e_f, e_f)
    off = np.where(np.eye(m_fe, dtype=bool), np.nan, fe_aoc)             # the diagonal is MASKED, not zeroed: an FE mode without energy leaves no number, not a 0
    off_max = float(np.max(off[np.isfinite(off)])) if np.isfinite(off).any() else None
    dg = np.diag(fe_aoc)
    info = {"sensor_dofs": len(dofs), "unit": tam.unit, "fe_modes": m_fe, "fe_offdiagonal_max": off_max,
            "fe_diagonal_min": float(np.min(dg[np.isfinite(dg)])) if np.isfinite(dg).any() else None, "rank": rank, "residual_kind": "null-space"}
    holds: list[str] = []
    warns: list[str] = []
    if invisible:
        msg = (f"compared FE mode(s) {_names(invisible)} are invisible to the imported mass matrix at the sensor DOFs (zero amplitude or zero energy there): the matrix cannot judge them, "
               "so it cannot be checked against this model")
        holds.append(msg)
        warns.append(msg)
    if m_fe > 1 and ((off_max is None and not invisible) or (off_max is not None and off_max > TAM_MAX_FE_OFFDIAG)):
        msg = (f"the imported mass matrix does not reproduce the orthogonality of the FE modes (largest off-diagonal term {'not computable' if off_max is None else f'{off_max:.3g}'} through it, "
               f"limit {TAM_MAX_FE_OFFDIAG:g}, the example of ECSS-E-ST-32-11C Table 5-3): it does not fit this model (DOFs, axes, scaling or reduction)")
        holds.append(msg)
        warns.append(msg)
    if m_fe == 1:
        msg = ("only one FE mode is compared, so the imported mass matrix cannot be checked against the model (the FE-FE orthogonality through it has no off-diagonal term; a rank-1 matrix "
               "gives COC = 1 to any shape): its fit (DOFs, axes, scaling, reduction) is not verified - compare at least two FE modes")
        holds.append(msg)
        warns.append(msg)
    g_study, g_why = _study_gen_mass(mass, modeset, fe_idx, m_fe)
    info["genmass_checked"] = False
    if g_why:
        holds.append(g_why)
        warns.append(g_why)
    elif g_study is not None and int(visible.sum()) >= 2:
        ratio = e_f[visible] / g_study[visible]
        scale = float(np.median(ratio))                                  # the ONE global factor: the unit of the file (kg / t), the normalisation of the matrix
        dev = np.abs(ratio / scale - 1.0)
        worst = float(dev.max())
        info.update({"genmass_checked": True, "genmass_scale": scale, "genmass_dev_max": worst, "genmass_dev_limit": TAM_MAX_GENMASS_DEV})
        if worst > TAM_MAX_GENMASS_DEV:
            vis_nums = [fe_nums[j] for j in range(m_fe) if visible[j]]
            bad = [vis_nums[i] for i in np.flatnonzero(dev > TAM_MAX_GENMASS_DEV)]
            msg = (f"the imported mass matrix does not reproduce the generalised masses of the FE modes of this model: after one global factor ({scale:.4g}) the generalised mass of FE mode(s) "
                   f"{_names(bad)} through it differs from the one under the mass of the study by up to {100 * worst:.0f} % (limit {100 * TAM_MAX_GENMASS_DEV:.0f} %): the matrix is not "
                   "the mass of this model at the sensor DOFs (a payload left out, modal masses rescaled, another reduction or another model)")
            holds.append(msg)
            warns.append(msg)
    result: dict[str, Any] = {"holds": holds, "warnings": warns}
    if g_study is None and not g_why:
        result["notes"] = ["the generalised masses of the FE modes through the imported mass matrix were NOT compared with the study (no study and mesh, or generalised masses, were given to compare with): "
                           "the matrix is judged by the FE orthogonality alone"]
    if null_basis is not None and null_basis.shape[1]:
        nrm2 = np.real(np.sum(np.conj(psi) * psi, axis=0))
        proj = null_basis.T @ psi
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(nrm2 > 0, np.clip(np.real(np.sum(np.conj(proj) * proj, axis=0)) / nrm2, 0.0, 1.0), np.nan)
        fn2 = np.sum(fp * fp, axis=0)
        fshare = np.where(fn2 > 0, np.sum((null_basis.T @ fp) ** 2, axis=0) / fn2, np.nan)
        info["residual_max"] = float(np.nanmax(share)) if np.isfinite(share).any() else None
        info["residual_limit"] = TAM_MAX_NULL_SHARE
        msg = (f"the imported mass matrix is singular (rank {rank} of {n} DOFs: it gives mass to a subspace of the sensor DOFs only, as a reduction with fewer modes than DOFs does): the part "
               f"of a shape in its null space has no mass, so the COC / AOC of a test mode with more than {100 * TAM_MAX_NULL_SHARE:.0f} % of its shape there are not those of the shape")
        warns.append(msg)
        holds_fe = [fe_nums[j] for j in range(len(fshare)) if not math.isfinite(float(fshare[j])) or fshare[j] > TAM_MAX_NULL_SHARE]
        if holds_fe:
            holds.append(f"compared FE mode(s) {_names(holds_fe)} have more than {100 * TAM_MAX_NULL_SHARE:.0f} % of their shape in the null "
                         f"space of the imported mass matrix (rank {rank} of {n}): the matrix does not see them")
        mode_holds: dict[int, str] = {}
        for k in range(len(share)):
            s = float(share[k])
            if not math.isfinite(s) or s > TAM_MAX_NULL_SHARE:
                mode_holds[k] = (f"has {100 * s:.1f} % of its shape in the null space of the imported mass matrix (rank {rank} of {n}; limit {100 * TAM_MAX_NULL_SHARE:.0f} %): the matrix "
                                 "gives that part no mass, so its COC / AOC are not those of the shape" if math.isfinite(s) else "has no usable null-space share")
        result.update({"residual": share, "mode_holds": mode_holds})
    return e_t, e_f, np.conj(psi).T @ Mf, np.conj(psi).T @ Mp, info, result


# ------------------------------------------------------------------ modulus difference
def modulus_difference(test_phi: np.ndarray, fe_phi: np.ndarray, pairs: list[tuple[int, int]], dofs: list[tuple[str, str]]) -> list[tuple[float | None, str]]:
    """The modulus difference of every pair (test index, FE index): (value, "node:dof" of the largest difference); None when the shape has fewer than two measured DOFs or is zero.
    See the module docstring for the definition (the standard gives none)."""
    out: list[tuple[float | None, str]] = []
    for i, j in pairs:
        a = np.asarray(test_phi[:, i])
        rows = (np.isfinite(a.real) & np.isfinite(a.imag)) if np.iscomplexobj(a) else np.isfinite(a)
        if rows.sum() < 2:
            out.append((None, ""))
            continue
        t = np.abs(a[rows])
        f = np.abs(np.asarray(fe_phi[rows, j], dtype=np.float64))
        nt, nf = float(np.sqrt(np.sum(t * t))), float(np.sqrt(np.sum(f * f)))
        if nt <= 0 or nf <= 0 or float(t.max()) <= 0:
            out.append((None, ""))
            continue
        diff = np.abs(t / nt - f / nf)
        k = int(np.argmax(diff))
        peak = float((t / nt).max())
        label = [dofs[d] for d in np.flatnonzero(rows)][k]
        out.append((float(diff[k] / peak), f"{label[0]}:{label[1]}"))
    return out


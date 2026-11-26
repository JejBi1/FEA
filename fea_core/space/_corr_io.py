"""Bounded readers and writers of vibration-test data for the test correlation (F14 stage S5): CSV and the universal file format (UFF) datasets 58 and 55.

Test files come from outside, so every reader is bounded and strict:
  * the file size, the number of lines, the length of a line, the number of datasets / functions / modes / values and the number of points of a function are capped
    (`MAX_*` below); anything above is rejected, never truncated silently;
  * a truncated or malformed dataset 58 / 55 (missing record, wrong number of values, text where a number is expected, not a finite number) rejects the WHOLE file with a
    `TestDataError` that names the dataset; nothing is evaluated, nothing is pickled, no code is executed, no other file is opened;
  * network (UNC) paths are refused, labels from the file are reduced to plain printable text of at most `MAX_LABEL` characters (`MAX_NAME` for a function name).
ASCII datasets other than 55 / 58 / 15 / 164 / 2411 / 2420 are skipped by number (18 is recognised but not interpreted); binary files are refused.

What is read
  CSV, shapes           mode, frequency_hz, [damping, damping_unit], node, dof, value [, imag]   (one row per mode and measured DOF; blank value = DOF not measured)
  CSV, mode table       mode, frequency_hz [, damping, damping_unit]
  CSV, functions        function, node, dof, frequency_hz and amplitude [+ phase_deg] | real + imag | value, optional kind (frf | psd | response), unit,
                        quantity (displacement | velocity | acceleration | force), per (force | acceleration | displacement | velocity)
  UFF 55                normal modes (analysis type 2) and complex modes (analysis type 3: the complex eigenvalue is ASSUMED to be rad/s - the dataset states no unit -, the shape
                        is kept complex and the correlation uses the COMPLEX MAC, never only the real part) at nodes, 3 or 6 values per node, real or complex; the damping of a
                        mode above 0.2 needs `damping_unit` (ratio | percent), as in the CSV
  UFF 58                functions of frequency (FRF, PSD, ...) at a node and direction; even or uneven abscissa; real or complex, single or double precision; the ordinate
                        quantity and denominator (records 9 / 10) and their unit labels are carried with the function (the numerator unit of record 9 and the denominator
                        unit of record 10 are combined by `correlation.resolve_function`: g and N make g/N)
  UFF 2411 / 15         nodes: the displacement coordinate system of every node
  UFF 2420              coordinate systems (Cartesian, checked orthonormal). The file does not say whether the COLUMNS or the ROWS of the 3 x 3 part R are the local axes: the
                        default is the columns (x_global = R x_local, the convention of OpenModal, the writer of the public beam.uff; `rotation_matrix`), `rotation="rows"` reads
                        the rows. The convention is stored in `TestData.rotation`; data at nodes whose matrix is not symmetric (`rotation_ambiguous`) stay UNCONFIRMED until the
                        user confirms the convention for the file (`TestData.rotation_confirmed`, set by `corr_run.confirm_rotation`)
  UFF 164               the unit system (carried and shown; the unit labels of the functions decide)
Numbers are read with the fixed widths of the format when the whitespace split does not give the declared count (numbers written without a space between them), otherwise by
the whitespace split.

Local coordinate systems: mode shapes at a node with a displacement CS are rotated into the global system when the CS is a fully defined Cartesian one; a function at such a node
gets a global `axis` (the model curve is projected on it). A node whose CS is not fully known (cylindrical / spherical, defined only in dataset 18 - whose layout is NOT interpreted -,
not defined in the file, not orthonormal) is EXCLUDED with a warning (`TestData.unresolved`): its data are never compared as if they were global. Displacement system 0, and system 1
when the file defines no system 1 (no dataset 2420), are the global system, with a note; a node that the node dataset does not list gets a note. The raw values at rotated nodes are
kept (`cs_raw_modes`), so `TestData.set_rotation` re-reads them with the other convention.

The writers produce the same formats (used for the round-trip tests and to hand the FE modes to a test engineer); they use the fixed-width records of the format. The layout of
the records is implemented from the published UFF dataset descriptions and checked on files written by `write_uff55` / `write_uff58` and on public sample files (pyuff, OpenModal);
it was NOT checked against files of a test-system vendor of the user (see the stage report), and the CS convention is the one of the public OpenModal file, not verified on a vendor file
(hence the explicit setting and the user's confirmation).
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ._corr_units import clean_unit

# ------------------------------------------------------------------ bounds of the readers
MAX_FILE_BYTES = 32 << 20            # bytes of a test file
MAX_LINES = 1_000_000                # lines of a file
MAX_LINE_CHARS = 4096                # characters of a line
MAX_DATASETS = 20_000                # UFF datasets (of any number) in a file
MAX_FUNCTIONS = 5_000                # measured functions of a file
MAX_FUNCTION_POINTS = 1 << 20        # points of one function
MAX_TOTAL_VALUES = 8_000_000         # numbers of all the functions and shapes of a file
MAX_MODES = 1_000                    # measured modes
MAX_NODES_PER_MODE = 100_000         # measured nodes of one mode (UFF 55) / DOFs of one mode (CSV)
MAX_CS = 100_000                     # coordinate systems / node records of the datasets 2420 / 2411 / 15 of a file
MAX_LABEL = 40                       # characters of a node label
MAX_NAME = 90                        # characters of a function name
DAMPING_PLAUSIBLE_MAX = 0.2          # a damping ratio above this needs an explicit unit in a CSV file

DOF_NAMES = ("x", "y", "z", "rx", "ry", "rz")
TRANSLATIONS = ("x", "y", "z")
FUNCTION_KINDS = ("frf", "psd", "response", "info")        # frf = transfer / response function, psd = power spectral density; response / info = imported as information only
COMPARED_KINDS = ("frf", "psd")                            # the kinds that are compared with the model
_UFF_FUNCTION_TYPES = {0: "general", 1: "time response", 2: "auto spectrum", 3: "cross spectrum", 4: "frequency response function", 5: "transmissibility", 6: "coherence",
                       7: "auto correlation", 8: "cross correlation", 9: "power spectral density", 10: "energy spectral density", 12: "spectrum", 18: "mode indicator function"}
_TYPE_SHORT = {1: "time", 2: "auto spectrum", 3: "cross spectrum", 4: "FRF", 5: "transmissibility", 6: "coherence", 7: "auto correlation", 8: "cross correlation",
               9: "PSD", 10: "ESD", 12: "spectrum", 18: "MIF"}
_PSD_TYPES = {2, 9}
_FRF_TYPES = {4, 5}
_ABSCISSA_FREQUENCY = 18             # UFF specific data type "frequency" of the abscissa
_ABSCISSA_TIME = 17
_SPECIFIC_QUANTITY = {8: "displacement", 11: "velocity", 12: "acceleration", 13: "force", 9: "force"}      # UFF specific data types of an ordinate / denominator
_QUANTITY_SPECIFIC = {"displacement": 8, "velocity": 11, "acceleration": 12, "force": 13}
_FIXED_WIDTHS = (13, 20, 12, 25, 26)
_READ_DATASETS = ("55", "58", "15", "164", "2411", "2420")
_IDENTITY = np.eye(3)
ROTATIONS = ("columns", "rows")
DEFAULT_ROTATION = "columns"          # the columns of the 3 x 3 part of a dataset 2420 are the local axes (the convention of OpenModal, the writer of the public beam.uff)
DAMPING_UNITS = ("", "ratio", "percent")


class TestDataError(ValueError):
    """A test file is refused (too large, malformed, truncated, unsupported)."""
    __test__ = False                 # not a pytest class


# ------------------------------------------------------------------ data
@dataclass
class TestMode:
    """A measured mode: number, frequency [Hz], viscous damping ratio (None = not given) and the shape at the measured DOFs {(node, dof): value}."""
    __test__ = False
    number: int
    frequency_hz: float
    damping: float | None = None
    shape: dict[tuple[str, str], complex | float] = field(default_factory=dict)


@dataclass
class TestFunction:
    """A measured function of frequency at a node / DOF (`values` real or complex, same length as `freq` [Hz]).

    `quantity` (displacement | velocity | acceleration | force | "" = unknown) and `per` (the denominator of a transfer function: force | acceleration | displacement | velocity
    | "" = none / unknown) come from the records 9 / 10 of a UFF 58 or the columns of a CSV; `unit` is the unit label of the ordinate. The comparison with the model needs
    them (or the choice of the user): see `fea_core.space.correlation.resolve_function`. `axis` = the global direction of the measured component when the node has a local
    coordinate system; `excluded` = why the function cannot be compared at all (empty = it can)."""
    __test__ = False
    name: str
    node: str
    dof: str
    freq: np.ndarray
    values: np.ndarray
    kind: str = "frf"                # frf | psd | response | info
    unit: str = ""
    domain: str = "frequency"        # frequency | time | other (only functions of frequency can be correlated)
    reference_node: str = ""
    description: str = ""
    quantity: str = ""
    per: str = ""
    per_unit: str = ""
    func_type: int | None = None
    reference_dof: str = ""
    local_cs: int = 0
    axis: tuple[float, float, float] | None = None
    excluded: str = ""
    pending_sign: float = 1.0        # the sign of the response direction (-X ...) that the reader has not applied yet (`_resolve_cs`: to the values, or into `axis` of a local system)


@dataclass
class CoordSystem:
    """A coordinate system of dataset 2420: `matrix` = the 3 x 3 rotation (rows = local axes in the global system), None when it is not fully known (`reason`)."""
    number: int
    kind: str = "cartesian"
    matrix: np.ndarray | None = None
    reason: str = ""


@dataclass
class TestData:
    """Everything read from one or more test files."""
    __test__ = False
    source: str = ""
    modes: list[TestMode] = field(default_factory=list)
    functions: list[TestFunction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_datasets: int = 0
    sha256: list[str] = field(default_factory=list)                       # SHA-256 of the file(s) the data were read from
    node_cs: dict[str, int] = field(default_factory=dict)                 # node -> displacement coordinate system (datasets 2411 / 15); 0 = global
    coord_systems: dict[int, CoordSystem] = field(default_factory=dict)
    unresolved: dict[str, str] = field(default_factory=dict)              # node -> why its local coordinate system is not usable (the node is left out)
    unit_system: dict[str, Any] = field(default_factory=dict)             # dataset 164
    rotation: str = DEFAULT_ROTATION                                      # how the 3 x 3 part of a dataset 2420 is read: columns | rows (see `rotation_matrix`)
    rotation_confirmed: bool = False                                      # the user confirmed the convention for this file (only a user action sets it)
    rotation_confirmed_via: str = ""                                      # gui | mcp
    rotation_ambiguous: list[str] = field(default_factory=list)           # used nodes whose local system gives different data under the two conventions
    cs_matrices: dict[str, np.ndarray] = field(default_factory=dict)      # node -> the 3 x 3 matrix of its resolved local system (as written in the file)
    cs_raw_modes: list[tuple["TestMode", str, dict[tuple[str, str], complex | float]]] = field(default_factory=list, repr=False)       # (mode, node, local values)
    cs_raw_functions: list[tuple["TestFunction", float]] = field(default_factory=list, repr=False)                                   # (function, sign of its direction)

    @property
    def rotation_unconfirmed(self) -> bool:
        """Data at rotated sensors that depend on the convention, and the user has not confirmed it: they must not give a PASS."""
        return bool(self.rotation_ambiguous) and not self.rotation_confirmed

    def set_rotation(self, convention: str) -> None:
        """Re-reads the local sensor systems with the other convention (columns | rows); the confirmation of the old convention is dropped."""
        if convention not in ROTATIONS:
            raise TestDataError(f"The rotation convention must be {' or '.join(ROTATIONS)}")
        if convention != self.rotation:
            self.rotation = convention
            self.rotation_confirmed, self.rotation_confirmed_via = False, ""
            _apply_rotation(self)

    def fingerprint(self) -> str:
        """SHA-256 of everything that was read: the modes (numbers, frequencies, damping, shapes), the functions (identity, unit, axis, curve), the rotation convention and the
        nodes that were left out. The comparison with the model depends on all of it, not only on the bytes of the file."""
        h = hashlib.sha256()

        def put(*parts: Any) -> None:
            h.update(("\x1f".join(str(p) for p in parts) + "\x1e").encode("utf-8", "replace"))
        put("rotation", self.rotation)
        for m in self.modes:
            keys = sorted(m.shape)
            put("mode", m.number, repr(float(m.frequency_hz)), repr(m.damping), len(keys), "|".join(f"{n}:{d}" for n, d in keys))
            h.update(np.array([m.shape[k] for k in keys], dtype=np.complex128).tobytes())
        for f in self.functions:
            put("function", f.name, f.node, f.dof, f.kind, f.unit, f.domain, f.quantity, f.per, f.per_unit, f.excluded, f.axis, f.reference_node, f.reference_dof, f.func_type)
            h.update(np.ascontiguousarray(f.freq, dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(f.values, dtype=np.complex128).tobytes())
        put("unresolved", sorted(self.unresolved.items()))
        return h.hexdigest()

    def nodes(self) -> list[str]:
        """Nodes (labels) that carry a measured shape value or a function, in order of appearance."""
        seen: dict[str, None] = {}
        for m in self.modes:
            for n, _ in m.shape:
                seen.setdefault(n)
        for f in self.functions:
            if f.node:
                seen.setdefault(f.node)
        return list(seen)

    def total_values(self) -> int:
        return sum(len(m.shape) for m in self.modes) + sum(int(f.values.size) for f in self.functions)

    def summary(self) -> dict[str, Any]:
        return {"source": self.source, "sha256": list(self.sha256),
                "modes": [{"number": m.number, "frequency_Hz": m.frequency_hz, "damping": m.damping, "dofs": len(m.shape)} for m in self.modes],
                "functions": [{"name": f.name, "node": f.node, "dof": f.dof, "kind": f.kind, "domain": f.domain, "points": int(f.freq.size),
                               "f_min_Hz": float(f.freq[0]) if f.freq.size else None, "f_max_Hz": float(f.freq[-1]) if f.freq.size else None, "unit": f.unit,
                               "quantity": f.quantity, "per": f.per, "per_unit": f.per_unit, "local_cs": f.local_cs, "excluded": f.excluded}
                              for f in self.functions],
                "nodes": self.nodes(), "warnings": list(self.warnings), "skipped_datasets": self.skipped_datasets,
                "local_cs_nodes": {n: c for n, c in self.node_cs.items() if c}, "unresolved_nodes": dict(self.unresolved), "unit_system": dict(self.unit_system),
                "rotation": {"convention": self.rotation, "confirmed": self.rotation_confirmed, "confirmed_via": self.rotation_confirmed_via, "ambiguous_nodes": list(self.rotation_ambiguous),
                             "unconfirmed": self.rotation_unconfirmed}}

    def merge(self, other: "TestData") -> None:
        if other.rotation != self.rotation:
            other.set_rotation(self.rotation)                                # one convention for the merged data
        if other.rotation_ambiguous:
            self.rotation_confirmed = False                                  # new rotated data: the convention was not confirmed for them
            self.rotation_confirmed_via = ""
        self.rotation_ambiguous += [n for n in other.rotation_ambiguous if n not in self.rotation_ambiguous]
        self.cs_matrices.update(other.cs_matrices)
        self.cs_raw_modes += other.cs_raw_modes
        self.cs_raw_functions += other.cs_raw_functions
        taken = {f.name for f in self.functions}
        for f in other.functions:                                           # a function name is the key of its setup and of its model curve: it must stay unique
            f.name = _unique(f.name, taken)
        self.modes += other.modes
        self.functions += other.functions
        self.warnings += other.warnings
        self.skipped_datasets += other.skipped_datasets
        self.sha256 += [h for h in other.sha256 if h not in self.sha256]
        self.node_cs.update(other.node_cs)
        self.coord_systems.update(other.coord_systems)
        self.unresolved.update(other.unresolved)
        self.unit_system = self.unit_system or other.unit_system
        self.source = ", ".join(x for x in (self.source, other.source) if x)
        if len(self.modes) > MAX_MODES or len(self.functions) > MAX_FUNCTIONS or self.total_values() > MAX_TOTAL_VALUES:
            raise TestDataError("The test data of the files together are too large")


# ------------------------------------------------------------------ helpers
_LABEL_BAD = re.compile(r"[^\w .:+\-/#()]", re.UNICODE)


def clean_label(raw: Any, limit: int = MAX_LABEL) -> str:
    """Plain, printable text of at most `limit` characters (labels of nodes and functions go to tables and the report)."""
    s = _LABEL_BAD.sub("_", str(raw).strip())
    return s[:limit]


def parse_dof(raw: str) -> tuple[str, float]:
    """A DOF name from a file -> (dof, sign): x | y | z | rx | ry | rz, optionally with a sign (+x, -z, ux, 1..6, -3). Anything else is refused."""
    s = str(raw).strip().lower().replace(" ", "")
    sign = 1.0
    if s[:1] in "+-" and s:
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    if s[-1:] in "+-" and len(s) > 1:                 # "z-" form
        sign = -1.0 if s[-1] == "-" else sign
        s = s[:-1]
    if s in ("1", "2", "3", "4", "5", "6"):
        return DOF_NAMES[int(s) - 1], sign
    if s[:1] == "u" and s[1:] in TRANSLATIONS:
        s = s[1:]
    if s in DOF_NAMES:
        return s, sign
    raise TestDataError(f"Unknown DOF '{clean_label(raw, 20)}' (use x, y, z, rx, ry, rz or 1 to 6, optionally signed)")


def _read_raw(path: str | Path, max_bytes: int) -> bytes:
    from ..pathcheck import is_unc
    if is_unc(path):
        raise TestDataError("Network (UNC) paths are not allowed")
    p = Path(str(path).strip())
    try:
        if not p.is_file():
            raise TestDataError(f"The test file does not exist: {p.name}")
        size = p.stat().st_size
        if size > max_bytes:
            raise TestDataError(f"The test file is too large ({size >> 20} MB, at most {max_bytes >> 20} MB)")
        with p.open("rb") as fh:
            raw = fh.read(max_bytes + 1)
    except OSError as e:
        raise TestDataError(f"The test file cannot be read: {e.__class__.__name__}") from None
    if len(raw) > max_bytes:
        raise TestDataError("The test file is too large")
    return raw


def _read_text(path: str | Path, max_bytes: int) -> str:
    return decode_bytes(_read_raw(path, max_bytes))


def decode_bytes(raw: bytes) -> str:
    if b"\x00" in raw:
        raise TestDataError("The file is binary (binary UFF and other binary formats are not supported)")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _lines(text: str) -> list[str]:
    if len(text) > MAX_FILE_BYTES * 4:
        raise TestDataError("The test file is too large")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        raise TestDataError(f"The test file has too many lines (at most {MAX_LINES})")
    for i, ln in enumerate(lines, 1):
        if len(ln) > MAX_LINE_CHARS:
            raise TestDataError(f"Line {i} is too long (at most {MAX_LINE_CHARS} characters)")
    return lines


def _num(tok: str, what: str, where: str) -> float:
    try:
        v = float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise TestDataError(f"{where}: {what} is not a number") from None
    if not math.isfinite(v):
        raise TestDataError(f"{where}: {what} is not a finite number")
    return v


def _int(s: str, what: str, where: str, blank: int | None = 0) -> int:
    t = s.strip()
    if not t:
        if blank is None:
            raise TestDataError(f"{where}: {what} is missing")
        return blank
    try:
        v = int(t)
    except ValueError:
        raise TestDataError(f"{where}: {what} is not an integer") from None
    return v


def _floats(tokens: list[str], where: str) -> np.ndarray:
    try:
        a = np.array([t.replace("D", "E").replace("d", "e") for t in tokens], dtype=np.float64)
    except ValueError:
        raise TestDataError(f"{where}: a data value is not a number") from None
    if not np.all(np.isfinite(a)):
        raise TestDataError(f"{where}: a data value is not a finite number")
    return a


def _tokens(lines: list[str], cap: int | None = None) -> list[str]:
    """The whitespace-separated tokens of the lines; with `cap` the splitting stops once more than `cap` tokens exist (a file that declares a few values and carries millions
    must not be split completely before it is refused)."""
    out: list[str] = []
    for ln in lines:
        out.extend(ln.split())
        if cap is not None and len(out) > cap:
            break
    return out


def _fixed(lines: list[str], width: int) -> list[str] | None:
    """The lines cut into fields of `width` characters (None when a field is empty: the width does not fit the lines)."""
    out: list[str] = []
    for ln in lines:
        s = ln.rstrip()
        if not s.strip():
            continue
        for i in range(0, len(s), width):
            c = s[i:i + width].strip()
            if not c:
                return None
            out.append(c)
    return out


def _values(lines: list[str], need: int, where: str, what: str = "data values") -> np.ndarray:
    """`need` numbers from the lines of a record block: the whitespace split when it gives that count, else the fixed widths of the format (13, 20, ...) for numbers written
    without a space between them. Raises TestDataError when neither gives `need` finite numbers."""
    toks = _tokens(lines, need)
    first: TestDataError | None = None
    if len(toks) == need:
        try:
            return _floats(toks, where)
        except TestDataError as e:
            first = e
    chars = sum(len(ln.rstrip()) for ln in lines)
    for w in _FIXED_WIDTHS:
        if chars > need * w:                                                         # more characters than `need` fields of this width can hold: do not even cut them
            continue
        cut = _fixed(lines, w)
        if cut is not None and len(cut) == need:
            try:
                return _floats(cut, where)
            except TestDataError:
                continue
    if first is not None:
        raise first
    found = f"more than {need}" if len(toks) > need else str(len(toks))
    raise TestDataError(f"{where}: {found} {what} found, {need} expected (truncated or malformed dataset)")


class _Budget:
    """Running total of the numbers read from one file."""

    def __init__(self) -> None:
        self.values = 0

    def add(self, n: int, where: str) -> None:
        self.values += int(n)
        if self.values > MAX_TOTAL_VALUES:
            raise TestDataError(f"{where}: the file holds too many values (at most {MAX_TOTAL_VALUES})")


# ------------------------------------------------------------------ UFF
def _is_delim(ln: str) -> bool:
    """The delimiter of a UFF dataset: -1 in the columns 1 to 6 and nothing after them (a data line that merely holds a -1 further right is not a delimiter)."""
    return ln[:6].strip() == "-1" and not ln[6:].strip()


def _uff_blocks(lines: list[str]) -> list[tuple[str, int, int]]:
    """[(dataset number text, first body line, end line)] of the datasets of a UFF file; a dataset without its closing -1 is truncated -> error."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        if not _is_delim(lines[i]):
            raise TestDataError(f"Line {i + 1}: a UFF dataset must start with a line '-1' (columns 1 to 6)")
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j >= n:
            raise TestDataError("The UFF file is truncated (a dataset has no number)")
        num = lines[j].strip().split()[0] if lines[j].strip() else ""
        end = j + 1
        while end < n and not _is_delim(lines[end]):
            end += 1
        if end >= n:
            raise TestDataError(f"The UFF file is truncated (dataset {clean_label(num, 8)} is not closed)")
        out.append((num, j + 1, end))
        if len(out) > MAX_DATASETS:
            raise TestDataError(f"The UFF file has too many datasets (at most {MAX_DATASETS})")
        i = end + 1
    if not out:
        raise TestDataError("No UFF dataset found")
    return out


def _ordinate_info(rec: str, where: str) -> tuple[int, str]:
    """(specific data type, unit label) of a record 9 / 10 of dataset 58 (I10, 3I5, 1X, A20, 1X, A20)."""
    return _int(rec[0:10], "the specific data type", where), clean_unit(rec[47:67]) if len(rec) > 47 else ""


def _blank_name(s: str) -> str:
    t = clean_label(s)
    return "" if t.upper() == "NONE" else t


def _unique(name: str, taken: set[str]) -> str:
    out, k = name, 2
    while out in taken:
        suffix = f" #{k}"
        out = name[:MAX_NAME - len(suffix)] + suffix
        k += 1
    taken.add(out)
    return out


def _parse58(body: list[str], where: str, budget: _Budget, notes: list[str]) -> TestFunction:
    if len(body) < 12:
        raise TestDataError(f"{where}: the dataset is truncated (fewer than 12 records)")
    r6, r7, r8, r9, r10 = body[5], body[6], body[7], body[8], body[9]
    func_type = _int(r6[0:5], "the function type", where)
    rsp_name = _blank_name(r6[31:41])
    rsp_node = _int(r6[41:51], "the response node", where)
    rsp_dir = _int(r6[51:55], "the response direction", where)
    ref_node = _int(r6[66:76], "the reference node", where)
    ref_dir = _int(r6[76:80], "the reference direction", where)
    ord_type = _int(r7[0:10], "the ordinate data type", where, blank=None)
    npts = _int(r7[10:20], "the number of data pairs", where, blank=None)
    spacing = _int(r7[20:30], "the abscissa spacing", where, blank=None)
    if ord_type not in (2, 4, 5, 6):
        raise TestDataError(f"{where}: ordinate data type {ord_type} is not supported (2, 4, 5, 6)")
    if spacing not in (0, 1):
        raise TestDataError(f"{where}: the abscissa spacing must be 0 (uneven) or 1 (even)")
    if not 1 <= npts <= MAX_FUNCTION_POINTS:
        raise TestDataError(f"{where}: the number of points {npts} is outside 1 to {MAX_FUNCTION_POINTS}")
    complex_data = ord_type in (5, 6)
    per = (2 if complex_data else 1) + (0 if spacing == 1 else 1)
    need = per * npts
    budget.add(need, where)
    a = _values(body[11:], need, where)
    if spacing == 1:
        xmin = _num(r7[30:43].strip() or "nan", "the abscissa minimum", where)
        dx = _num(r7[43:56].strip() or "nan", "the abscissa increment", where)
        if dx <= 0:
            raise TestDataError(f"{where}: the abscissa increment must be positive")
        x = xmin + dx * np.arange(npts, dtype=np.float64)
        y = a[0::2] + 1j * a[1::2] if complex_data else a
    elif complex_data:
        x, y = a[0::3], a[1::3] + 1j * a[2::3]
    else:
        x, y = a[0::2], a[1::2]
    if npts > 1 and not np.all(np.diff(x) > 0):
        raise TestDataError(f"{where}: the abscissa is not strictly increasing")
    abs_type = _int(r8[0:10], "the abscissa data type", where)
    abs_unit = clean_unit(r8[47:67]).lower() if len(r8) > 47 else ""
    abs_label = clean_label(r8[26:46]).lower() if len(r8) > 26 else ""
    domain = "frequency" if abs_type == _ABSCISSA_FREQUENCY else "time" if abs_type == _ABSCISSA_TIME else "other"
    if domain == "other" and abs_type == 0 and (abs_unit.startswith("hz") or abs_label.startswith("freq")):
        domain = "frequency"
        notes.append(f"{where}: the abscissa has no data type but the unit {clean_unit(r8[47:67])} - taken as frequency [Hz]")
    q_type, ord_unit = _ordinate_info(r9, where)
    d_type, den_unit = _ordinate_info(r10, where)
    dof, sign = ("", 1.0)
    if rsp_dir != 0:
        try:
            dof, sign = parse_dof(str(rsp_dir))
        except TestDataError:
            raise TestDataError(f"{where}: the response direction {rsp_dir} is not supported") from None
    ref_dof, ref_sign = ("", 1.0)
    if ref_dir != 0:
        try:
            ref_dof, ref_sign = parse_dof(str(ref_dir))
        except TestDataError:
            raise TestDataError(f"{where}: the reference direction {ref_dir} is not supported") from None
    kind = "psd" if func_type in _PSD_TYPES else "frf" if func_type in _FRF_TYPES else "info"
    if kind == "psd" and np.iscomplexobj(y) and not np.any(y.imag):
        y = y.real.copy()                                                                    # a spectrum written with a zero imaginary part is a real spectrum
    if kind == "frf":
        y = y * ref_sign                                                                      # the response sign follows in `_resolve_cs` (into `axis` for a local system)
    ident = clean_label(body[0], MAX_NAME) or f"function {func_type}"
    resp = f"{rsp_node}:{dof or '-'}" if rsp_node else (rsp_name or "-")
    ref = f" ref {ref_node}:{ref_dof or '-'}" if ref_node else ""
    name = clean_label(f"{ident} - {_TYPE_SHORT.get(func_type, f'type {func_type}')} - {resp}{ref}", MAX_NAME)
    f = TestFunction(name=name, node=str(rsp_node) if rsp_node else rsp_name, dof=dof, freq=x, values=y, kind=kind, unit=ord_unit, domain=domain,
                     reference_node=str(ref_node) if ref_node else "", description=_UFF_FUNCTION_TYPES.get(func_type, f"function type {func_type}"),
                     quantity=_SPECIFIC_QUANTITY.get(q_type, ""), per=_SPECIFIC_QUANTITY.get(d_type, ""), per_unit=den_unit, func_type=func_type, reference_dof=ref_dof,
                     pending_sign=sign if kind == "frf" else 1.0)                             # a PSD / auto spectrum is never negated
    return f


def _uff_damping(raw: float, unit: str, where: str) -> float:
    """The viscous damping ratio of a dataset 55 (record 8): the same rule as the CSV reader - an explicit unit (ratio | percent) is applied; without one a value above
    DAMPING_PLAUSIBLE_MAX is refused (a percent value would silently be taken as a huge ratio)."""
    if unit == "percent":
        return raw * 0.01
    if unit == "ratio":
        return raw
    if raw > DAMPING_PLAUSIBLE_MAX:
        raise TestDataError(f"{where}: the damping {raw:g} is above {DAMPING_PLAUSIBLE_MAX} and the dataset does not say whether it is a ratio or a percent - import with the damping unit "
                            "ratio or percent")
    return raw


def _node_values(chunk: list[str], ncomp: int, where: str, node: int) -> list[float]:
    """The `ncomp` numbers of one node record of dataset 55 (the fast path for the usual whitespace-separated numbers, `_values` for the rest)."""
    toks = chunk[0].split() if len(chunk) == 1 else _tokens(chunk, ncomp)
    if len(toks) == ncomp:
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            return [float(v) for v in _values(chunk, ncomp, f"{where}, node {node}", "values")]          # D exponents, text: the careful path decides
        if not all(math.isfinite(v) for v in vals):
            raise TestDataError(f"{where}, node {node}: a value is not a finite number")
        return vals
    return [float(v) for v in _values(chunk, ncomp, f"{where}, node {node}", "values")]


def _parse55(body: list[str], where: str, budget: _Budget, serial: int, notes: list[str], damping_unit: str = "") -> TestMode | None:
    if len(body) < 8:
        raise TestDataError(f"{where}: the dataset is truncated (fewer than 8 records)")
    r6 = body[5]
    analysis = _int(r6[10:20], "the analysis type", where, blank=None)
    char = _int(r6[20:30], "the data characteristic", where, blank=None)
    dtype = _int(r6[40:50], "the data type", where, blank=None)
    nval = _int(r6[50:60], "the number of values per node", where, blank=None)
    if analysis not in (2, 3):
        return None                                                      # not a mode-shape dataset (static / frequency response / ...): skipped by the caller
    if dtype not in (2, 5) or nval not in (3, 6) or char not in (2, 3):
        raise TestDataError(f"{where}: only real or complex shapes with 3 or 6 values per node are supported")
    r7 = body[6]
    nint = _int(r7[0:10], "the number of integer values", where, blank=None)
    nreal = _int(r7[10:20], "the number of real values", where, blank=None)
    if not 0 <= nint <= 6 or not 0 <= nreal <= 12:
        raise TestDataError(f"{where}: the record of analysis data is malformed")
    ints = [_int(r7[20 + 10 * k:30 + 10 * k], "an integer of the analysis data", where) for k in range(nint)]
    nreal_lines = (nreal + 5) // 6
    reals = _values(body[7:7 + nreal_lines], nreal, where, "real values of the analysis data") if nreal else np.empty(0)
    zeta: float | None
    if analysis == 2:
        freq = float(reals[0]) if nreal >= 1 else float("nan")
        if not (math.isfinite(freq) and freq >= 0):
            raise TestDataError(f"{where}: the modal frequency is missing or negative")
        zeta = _uff_damping(float(reals[2]), damping_unit, where) if nreal >= 3 and reals[2] > 0 else None
    else:                                                                # complex eigenvalue (analysis type 3): lambda = re + i im [rad/s]
        if nreal < 2:
            raise TestDataError(f"{where}: the complex eigenvalue is missing")
        lam = complex(float(reals[0]), float(reals[1]))
        wn = abs(lam)
        freq = wn / (2.0 * math.pi)
        zeta = float(-lam.real / wn) if wn > 0 and lam.real < 0 else None
        note = ("complex modes (analysis type 3): the eigenvalue is ASSUMED to be in rad/s (f = |lambda| / 2 pi; the dataset does not state its unit - a file written in Hz gives a "
                "frequency 2 pi too low), the shape is kept complex and the complex MAC is used: check the frequencies")
        if note not in notes:
            notes.append(note)
    if zeta is not None and not 0 <= zeta < 1:
        raise TestDataError(f"{where}: the damping ratio {zeta:.3g} is outside [0, 1)")
    number = ints[1] if len(ints) >= 2 else serial
    ncomp = nval * (2 if dtype == 5 else 1)
    rest = body[7 + nreal_lines:]
    k = 0
    shape: dict[tuple[str, str], complex | float] = {}
    nnodes = 0
    while k < len(rest):
        if not rest[k].strip():
            k += 1
            continue
        nodes_tok = rest[k].split()
        if len(nodes_tok) != 1:
            raise TestDataError(f"{where}: a node record is malformed (line {k + 1} of the data)")
        node = _int(nodes_tok[0], "the node number", where, blank=None)
        nlines = (ncomp + 5) // 6
        chunk = rest[k + 1:k + 1 + nlines]
        if len(chunk) != nlines:
            raise TestDataError(f"{where}: node {node} has fewer values than the {ncomp} expected (truncated or malformed dataset)")
        flat = _node_values(chunk, ncomp, where, node)
        label = str(node)
        for i, d in enumerate(DOF_NAMES[:nval]):
            shape[(label, d)] = complex(flat[2 * i], flat[2 * i + 1]) if dtype == 5 else flat[i]
        nnodes += 1
        if nnodes > MAX_NODES_PER_MODE:
            raise TestDataError(f"{where}: too many nodes (at most {MAX_NODES_PER_MODE})")
        k += 1 + nlines
    if not shape:
        raise TestDataError(f"{where}: the dataset has no node data")
    budget.add(len(shape), where)
    return TestMode(number=int(number), frequency_hz=freq, damping=zeta, shape=shape)


def _parse_nodes_2411(body: list[str], where: str, data: TestData) -> None:
    """Dataset 2411: two records per node (4I10: label, export CS, displacement CS, colour; the coordinates)."""
    recs = [ln for ln in body if ln.strip()]
    if len(recs) % 2:
        raise TestDataError(f"{where}: the dataset is truncated (an odd number of records)")
    if len(recs) // 2 > MAX_CS * 10:
        raise TestDataError(f"{where}: too many nodes")
    for i in range(0, len(recs), 2):
        label = _int(recs[i][0:10], "the node label", where, blank=None)
        data.node_cs[str(label)] = _int(recs[i][20:30], "the displacement coordinate system", where)


def _parse_nodes_15(body: list[str], where: str, data: TestData) -> None:
    """Dataset 15: one record per node (4I10 label, definition CS, displacement CS, colour, then the coordinates)."""
    recs = [ln for ln in body if ln.strip()]
    if len(recs) > MAX_CS * 10:
        raise TestDataError(f"{where}: too many nodes")
    for r in recs:
        label = _int(r[0:10], "the node label", where, blank=None)
        data.node_cs[str(label)] = _int(r[20:30], "the displacement coordinate system", where)


def _parse_cs_2420(body: list[str], where: str, data: TestData) -> None:
    """Dataset 2420: part UID, part name, then per coordinate system: label / type / colour, name, 12 numbers (3 rows of the rotation + the origin)."""
    recs = list(body)
    while recs and not recs[-1].strip():
        recs.pop()
    if len(recs) < 2 + 6 or (len(recs) - 2) % 6:
        raise TestDataError(f"{where}: the dataset is truncated or malformed (2 header records and 6 records per coordinate system expected)")
    k = 2
    while k < len(recs):
        number = _int(recs[k][0:10], "the coordinate system label", where, blank=None)
        ctype = _int(recs[k][10:20], "the coordinate system type", where)
        vals = _values(recs[k + 2:k + 6], 12, where, "transformation values")
        if len(data.coord_systems) >= MAX_CS:
            raise TestDataError(f"{where}: too many coordinate systems")
        kind = {0: "cartesian", 1: "cylindrical", 2: "spherical"}.get(ctype, f"type {ctype}")
        cs = CoordSystem(number=number, kind=kind)
        r = vals[:9].reshape(3, 3)
        if ctype != 0:
            cs.reason = f"coordinate system {number} is {kind}: only Cartesian systems are interpreted"
        elif not (np.allclose(r @ r.T, _IDENTITY, atol=1e-6) and abs(float(np.linalg.det(r)) - 1.0) < 1e-6):
            cs.reason = f"coordinate system {number} is not a right-handed orthonormal frame"
        else:
            cs.matrix = r.copy()
        data.coord_systems[number] = cs
        k += 6


def _parse_units_164(body: list[str], where: str, data: TestData) -> None:
    recs = [ln for ln in body if ln.strip()]
    if len(recs) < 2:
        raise TestDataError(f"{where}: the dataset is truncated")
    code = _int(recs[0][0:10], "the unit system code", where)
    fac = _values(recs[1:3], 4, where, "unit factors")                     # length, force, temperature factor and the temperature offset
    data.unit_system = {"code": code, "name": clean_label(recs[0][10:30], 20), "length_factor": float(fac[0]), "force_factor": float(fac[1]), "temperature_factor": float(fac[2]),
                        "temperature_offset": float(fac[3])}


def rotation_matrix(r: np.ndarray, convention: str) -> np.ndarray:
    """A: v_global = v_local @ A for a row vector v. `columns`: the local axes are the COLUMNS of the 3 x 3 part r of a dataset 2420 (x_global = r x_local, the convention of OpenModal,
    the writer of the public beam.uff) -> A = r^T; `rows`: the local axes are its ROWS (x_global = x_local r) -> A = r. Row i of A is the local axis i in the global system."""
    return r.T if convention == "columns" else r


def _rotate_triples(shape: dict[tuple[str, str], complex | float], node: str, a: np.ndarray) -> None:
    for dofs in (("x", "y", "z"), ("rx", "ry", "rz")):
        if all((node, d) in shape for d in dofs):
            v = np.array([shape[(node, d)] for d in dofs]) @ a
            for d, c in zip(dofs, v):
                shape[(node, d)] = complex(c) if isinstance(shape[(node, d)], complex) else float(np.real(c))


def _apply_rotation(data: TestData) -> None:
    """(Re)applies the local sensor systems of `data` with `data.rotation` to the values read from the file (idempotent: it starts from the values as they were written)."""
    for m, node, raw in data.cs_raw_modes:
        for k, v in raw.items():
            m.shape[k] = v
        _rotate_triples(m.shape, node, rotation_matrix(data.cs_matrices[node], data.rotation))
    for f, sign in data.cs_raw_functions:
        ax = sign * rotation_matrix(data.cs_matrices[f.node], data.rotation)[TRANSLATIONS.index(f.dof)]       # the sign of the direction is part of the global axis of the local component
        f.axis = (float(ax[0]), float(ax[1]), float(ax[2]))


def _resolve_cs(data: TestData) -> None:
    """Records the local coordinate systems, rotates mode shapes into the global system with the chosen convention (`data.rotation`), gives a function an `axis`, and EXCLUDES what
    cannot be resolved. Which nodes depend on the convention is kept in `data.rotation_ambiguous`: their data stay UNCONFIRMED until the user confirms the convention."""
    notes: list[str] = []
    defined = data.coord_systems
    local: dict[str, int] = {}
    implicit_global: list[str] = []
    for n, c in data.node_cs.items():
        if not c:
            continue
        if c == 1 and 1 not in defined:                                       # system 1 is the global one in the usual systems and the file does not define it
            implicit_global.append(n)
            continue
        local[n] = c
    if implicit_global:
        notes.append(f"{len(implicit_global)} node(s) name the displacement system 1 and the file defines no system 1 (no dataset 2420): taken as the global system - "
                     + ", ".join(implicit_global[:8]) + (" ..." if len(implicit_global) > 8 else ""))
    if data.node_cs or data.coord_systems:
        notes.append(f"The file defines coordinate systems (datasets 15 / 18 / 2411 / 2420): {len(local)} node(s) have a local displacement system and {len(defined)} system(s) are defined. "
                     "A node whose system is not fully known is left out (cylindrical / spherical systems, and systems that only the old dataset 18 would define, are not interpreted).")
    elif data.modes:
        notes.append("The file has no node dataset (2411 / 15): the displacement directions of the modes are taken as global - confirm that no local sensor axes were used.")
    used: set[str] = {n for m in data.modes for n, _ in m.shape} | {f.node for f in data.functions if f.node}
    if data.node_cs:
        absent = sorted(n for n in used if n.isdigit() and n not in data.node_cs)
        if absent:
            notes.append(f"{len(absent)} node(s) of the data are not in the node dataset (2411 / 15): their displacement system is taken as global - " + ", ".join(absent[:8])
                         + (" ..." if len(absent) > 8 else ""))
    reason_of: dict[str, str] = {}
    rot: dict[str, np.ndarray] = {}
    for node, cs_id in local.items():
        cs = defined.get(cs_id)
        if cs is None:
            reason_of[node] = f"its displacement coordinate system {cs_id} is not defined in the file (dataset 2420)"
        elif cs.matrix is None:
            reason_of[node] = cs.reason or f"coordinate system {cs_id} is not usable"
        else:
            rot[node] = cs.matrix
    for node in sorted(used & set(reason_of)):
        data.unresolved[node] = reason_of[node]
    data.cs_matrices = {n: r for n, r in rot.items() if n in used}
    for m in data.modes:
        for node in {n for n, _ in m.shape}:
            if node in data.unresolved:
                for key in [k for k in m.shape if k[0] == node]:
                    del m.shape[key]
            elif node in rot:
                data.cs_raw_modes.append((m, node, {k: v for k, v in m.shape.items() if k[0] == node}))
    for f in data.functions:
        sign, f.pending_sign = f.pending_sign, 1.0
        if f.node in data.unresolved:
            f.excluded = f"the local coordinate system of node {f.node} is not fully known: {data.unresolved[f.node]}"
            f.values = f.values * sign
            continue
        if f.node in rot and f.dof in TRANSLATIONS:
            f.local_cs = local[f.node]
            data.cs_raw_functions.append((f, sign))
        elif sign != 1.0:
            f.values = f.values * sign
    _apply_rotation(data)
    data.rotation_ambiguous = sorted((n for n, r in data.cs_matrices.items() if not np.allclose(r, r.T, atol=1e-9)), key=lambda s: (len(s), s))
    if data.rotation_ambiguous:
        other = "rows" if data.rotation == "columns" else "columns"
        notes.append(f"{len(data.rotation_ambiguous)} node(s) have a rotated local sensor system ({', '.join(data.rotation_ambiguous[:8])}{' ...' if len(data.rotation_ambiguous) > 8 else ''}). "
                     f"UFF dataset 2420 does not say whether the COLUMNS or the ROWS of its 3 x 3 part are the local axes: the {data.rotation} are used (columns = OpenModal's reading). "
                     f"With the {other} the data at these nodes would be rotated differently. Until you confirm the convention for this file the rows that depend on these data stay UNCONFIRMED.")
    data.warnings += notes
    for n, why in data.unresolved.items():
        data.warnings.append(f"node {n} is left out: {why}")


def parse_uff(text: str, source: str = "", rotation: str = DEFAULT_ROTATION, damping_unit: str = "") -> TestData:
    """Datasets 58 (functions), 55 (normal / complex modes) and the coordinate-system datasets of an ASCII UFF text. See the module docstring for the limits.
    `rotation` (columns | rows) = how the 3 x 3 part of a dataset 2420 is read (see `rotation_matrix`); `damping_unit` (ratio | percent | "") = the unit of the damping of the modes."""
    if rotation not in ROTATIONS:
        raise TestDataError(f"The rotation convention must be {' or '.join(ROTATIONS)}")
    if damping_unit not in DAMPING_UNITS:
        raise TestDataError("The damping unit must be ratio or percent")
    lines = _lines(text)
    data = TestData(source=source, rotation=rotation)
    budget = _Budget()
    seen: Counter[str] = Counter()
    skipped: list[str] = []
    taken: set[str] = set()
    notes: list[str] = []
    for num, a, b in _uff_blocks(lines):
        if num.endswith("b") or num.endswith("B"):
            raise TestDataError("Binary UFF datasets are not supported")
        seen[num] += 1
        if num not in _READ_DATASETS:
            data.skipped_datasets += 1
            continue
        where = f"UFF dataset {num} (line {a})"
        body = lines[a:b]
        if num == "58":
            fn = _parse58(body, where, budget, notes)
            fn.name = _unique(fn.name, taken)
            data.functions.append(fn)
            if len(data.functions) > MAX_FUNCTIONS:
                raise TestDataError(f"The file has too many functions (at most {MAX_FUNCTIONS})")
        elif num == "55":
            m = _parse55(body, where, budget, len(data.modes) + 1, notes, damping_unit)
            if m is None:
                r6 = body[5] if len(body) > 5 else ""
                skipped.append(f"{where}: analysis type {_int(r6[10:20], 'the analysis type', where)} is not a normal / complex mode shape - skipped")
                continue
            data.modes.append(m)
            if len(data.modes) > MAX_MODES:
                raise TestDataError(f"The file has too many modes (at most {MAX_MODES})")
        elif num == "2411":
            _parse_nodes_2411(body, where, data)
        elif num == "15":
            _parse_nodes_15(body, where, data)
        elif num == "2420":
            _parse_cs_2420(body, where, data)
        elif num == "164":
            try:
                _parse_units_164(body, where, data)
            except TestDataError as e:                                        # the unit system is information only: a damaged record is ignored, the unit labels decide
                data.skipped_datasets += 1
                notes.append(f"{e} - the unit system is ignored")
    data.warnings += notes + skipped
    if "18" in seen:
        data.warnings.append("The file holds dataset 18 (coordinate systems, old format): its layout is not interpreted (it is not documented reliably enough) - the data at nodes whose "
                             "displacement system is defined only there are LEFT OUT, never read as global.")
        for n, c in data.node_cs.items():
            if c and c not in data.coord_systems:
                data.coord_systems[c] = CoordSystem(c, "cartesian", None, f"coordinate system {c} is defined only in dataset 18, which is not interpreted")
    if not data.modes and not data.functions:
        found = ", ".join(f"{k} x{v}" for k, v in list(seen.items())[:10])
        why = ("; ".join(skipped[:3]) if skipped else "no dataset 55 (mode shapes) or 58 (functions) was found")
        raise TestDataError(f"The UFF file holds no usable test data (datasets found: {found}). {why}")
    _resolve_cs(data)
    data.modes.sort(key=lambda m: m.number)
    return data


# ------------------------------------------------------------------ CSV
_SYN = {"frequency": "frequency_hz", "freq": "frequency_hz", "freq_hz": "frequency_hz", "f_hz": "frequency_hz", "f": "frequency_hz", "hz": "frequency_hz",
        "zeta": "damping", "damping_ratio": "damping", "direction": "dof", "channel": "function", "name": "function", "mag": "amplitude", "magnitude": "amplitude",
        "phase": "phase_deg", "re": "real", "im": "imag", "imaginary": "imag", "mode_number": "mode", "mode_id": "mode", "zeta_unit": "damping_unit",
        "denominator": "per", "units": "unit"}
_DAMPING_UNITS = {"ratio": 1.0, "fraction": 1.0, "1": 1.0, "percent": 0.01, "%": 0.01, "pct": 0.01}


def _csv_rows(text: str) -> tuple[list[str], Iterator[tuple[int, dict[str, str]]]]:
    """(header, a generator of (row number, {column: cell})): the rows are produced one at a time, never held all at once."""
    lines = _lines(text)
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        raise TestDataError("The CSV file is empty")
    delim = max((",", ";", "\t"), key=body[0].count)
    reader = csv.reader(io.StringIO("\n".join(body)), delimiter=delim)
    try:
        header = [_SYN.get(h.strip().lower(), h.strip().lower()) for h in next(reader)]
    except csv.Error as e:
        raise TestDataError(f"The CSV file is malformed: {str(e)[:80]}") from None
    if len(set(header)) != len(header) or len(header) > 64:
        raise TestDataError("The CSV header has duplicate or too many columns")

    def rows() -> Iterator[tuple[int, dict[str, str]]]:
        try:
            for i, r in enumerate(reader, 2):
                if not r:
                    continue
                if len(r) > len(header):
                    raise TestDataError(f"CSV row {i} has more cells than the header")
                yield i, dict(zip(header, [c.strip() for c in r]))
        except csv.Error as e:
            raise TestDataError(f"The CSV file is malformed: {str(e)[:80]}") from None
    return header, rows()


def _cnum(row: dict[str, str], key: str, i: int, required: bool = True) -> float | None:
    s = row.get(key, "")
    if not s:
        if required:
            raise TestDataError(f"CSV row {i}: '{key}' is missing")
        return None
    return _num(s, f"'{key}'", f"CSV row {i}")


def parse_csv(text: str, source: str = "") -> TestData:
    """CSV test data: mode shapes, a mode table (frequencies / damping) or measured functions - recognised from the header (see the module docstring)."""
    header, rows = _csv_rows(text)
    data = TestData(source=source)
    hs = set(header)
    if "frequency_hz" not in hs:
        raise TestDataError("The CSV header needs a frequency_hz column")
    has_shape = {"node", "dof"} <= hs and ("value" in hs or "real" in hs) and "mode" in hs
    has_func = {"node", "dof"} <= hs and "function" in hs and bool(hs & {"amplitude", "real", "value"})
    if has_func and not has_shape:
        _csv_functions(rows, data)
    elif has_shape:
        _csv_modes(rows, data, True)
    elif "mode" in hs:
        _csv_modes(rows, data, False)
    else:
        raise TestDataError("The CSV header is not recognised: modes need mode, frequency_hz (+ node, dof, value); functions need function, node, dof, frequency_hz, amplitude | real, imag")
    if not data.modes and not data.functions:
        raise TestDataError("The CSV file holds no data rows")
    return data


def _csv_damping(r: dict[str, str], i: int) -> float | None:
    """The damping ratio of a CSV row as a ratio: an explicit `damping_unit` (ratio | percent) is applied; without it a value above DAMPING_PLAUSIBLE_MAX is refused (a percent
    value would silently be taken as a huge ratio)."""
    z = _cnum(r, "damping", i, required=False)
    if z is None:
        return None
    unit = (r.get("damping_unit") or "").strip().lower()
    if unit:
        if unit not in _DAMPING_UNITS:
            raise TestDataError(f"CSV row {i}: damping_unit must be ratio or percent")
        z *= _DAMPING_UNITS[unit]
    elif z > DAMPING_PLAUSIBLE_MAX:
        raise TestDataError(f"CSV row {i}: the damping {z:g} is above {DAMPING_PLAUSIBLE_MAX} and has no unit - add a damping_unit column (ratio or percent)")
    if not 0 <= z < 1:
        raise TestDataError(f"CSV row {i}: a damping ratio must be in [0, 1)")
    return float(z)


def _csv_modes(rows: Iterable[tuple[int, dict[str, str]]], data: TestData, with_shape: bool) -> None:
    modes: dict[str, TestMode] = {}
    numbered: dict[int, str] = {}                                   # mode number -> the key of the file (two keys must not give one number)
    pending: list[TestMode] = []                                    # modes whose key is not a number: numbered after all the numeric ones
    budget = _Budget()
    for i, r in rows:
        key = r.get("mode", "")
        if not key:
            raise TestDataError(f"CSV row {i}: 'mode' is missing")
        f = _cnum(r, "frequency_hz", i, required=key not in modes)
        z = _csv_damping(r, i)
        m = modes.get(key)
        if m is None:
            if len(modes) >= MAX_MODES:
                raise TestDataError(f"The CSV file has too many modes (at most {MAX_MODES})")
            if f is None or f < 0:
                raise TestDataError(f"CSV row {i}: the frequency of mode {clean_label(key)} must be a number >= 0")
            m = modes[key] = TestMode(number=0, frequency_hz=float(f), damping=None if not z else z)
            if re.fullmatch(r"\d+(\.0+)?", key):
                num = int(float(key))
                if num in numbered:
                    raise TestDataError(f"CSV row {i}: modes '{clean_label(numbered[num])}' and '{clean_label(key)}' have the same number")
                numbered[num] = key
                m.number = num
            else:
                pending.append(m)
        else:
            if f is not None and abs(f - m.frequency_hz) > 1e-6 * max(abs(m.frequency_hz), 1.0):
                raise TestDataError(f"CSV row {i}: mode {clean_label(key)} has two different frequencies")
            if z is not None and z > 0:
                if m.damping is None:
                    m.damping = z
                elif abs(z - m.damping) > 1e-9 * max(z, 1.0):
                    raise TestDataError(f"CSV row {i}: mode {clean_label(key)} has two different damping ratios")
        if not with_shape:
            continue
        node = clean_label(r.get("node", ""))
        if not node:
            raise TestDataError(f"CSV row {i}: 'node' is missing")
        dof, sign = parse_dof(r.get("dof", ""))
        if not r.get("value") and not r.get("real"):
            continue                                                    # DOF not measured in this mode
        re_ = _cnum(r, "value" if r.get("value") else "real", i)
        im = _cnum(r, "imag", i, required=False)
        v: complex | float = float(re_ or 0.0) * sign if im is None else complex(float(re_ or 0.0), float(im)) * sign
        if (node, dof) in m.shape:
            raise TestDataError(f"CSV row {i}: node {node} {dof} appears twice in mode {clean_label(key)}")
        m.shape[(node, dof)] = v
        if len(m.shape) > MAX_NODES_PER_MODE:
            raise TestDataError(f"Mode {clean_label(key)} has too many DOFs (at most {MAX_NODES_PER_MODE})")
        budget.add(1, f"CSV row {i}")
    free = 1
    for m in pending:                                               # a non-numeric key gets the smallest number no numeric key uses
        while free in numbered:
            free += 1
        m.number = free
        numbered[free] = "?"
    data.modes = sorted(modes.values(), key=lambda mm: mm.number)


def _csv_functions(rows: Iterable[tuple[int, dict[str, str]]], data: TestData) -> None:
    groups: dict[str, dict[str, Any]] = {}
    budget = _Budget()
    for i, r in rows:
        name = clean_label(r.get("function", ""), MAX_NAME)
        if not name:
            raise TestDataError(f"CSV row {i}: 'function' is missing")
        g = groups.get(name)
        if g is None:
            if len(groups) >= MAX_FUNCTIONS:
                raise TestDataError(f"The CSV file has too many functions (at most {MAX_FUNCTIONS})")
            dof, sign = parse_dof(r.get("dof", ""))
            kind = (r.get("kind") or "frf").strip().lower()
            if kind not in FUNCTION_KINDS:
                raise TestDataError(f"CSV row {i}: kind must be one of {', '.join(FUNCTION_KINDS)}")
            quantity = (r.get("quantity") or "").strip().lower()
            per = (r.get("per") or "").strip().lower()
            if quantity not in ("", "displacement", "velocity", "acceleration", "force"):
                raise TestDataError(f"CSV row {i}: quantity must be displacement, velocity, acceleration or force")
            if per not in ("", "none", "force", "acceleration", "displacement", "velocity"):
                raise TestDataError(f"CSV row {i}: per must be none, force, acceleration, displacement or velocity")
            g = groups[name] = {"node": clean_label(r.get("node", "")), "dof": dof, "sign": sign if kind != "psd" else 1.0, "kind": kind, "unit": clean_unit(r.get("unit", "")),
                                "quantity": quantity, "per": "" if per == "none" else per, "f": [], "v": []}          # a PSD is never negated: it is a squared quantity
            if not g["node"]:
                raise TestDataError(f"CSV row {i}: 'node' is missing")
        f = _cnum(r, "frequency_hz", i)
        if r.get("amplitude"):
            a = _cnum(r, "amplitude", i)
            ph = _cnum(r, "phase_deg", i, required=False)
            v: complex | float = float(a or 0.0) if ph is None else complex(float(a or 0.0) * math.cos(math.radians(ph)), float(a or 0.0) * math.sin(math.radians(ph)))
        elif r.get("real"):
            v = complex(float(_cnum(r, "real", i) or 0.0), float(_cnum(r, "imag", i, required=False) or 0.0))
        else:
            v = float(_cnum(r, "value", i) or 0.0)
        g["f"].append(float(f or 0.0))
        g["v"].append(v * g["sign"])
        if len(g["f"]) > MAX_FUNCTION_POINTS:
            raise TestDataError(f"Function {name} has too many points (at most {MAX_FUNCTION_POINTS})")
        budget.add(1, f"CSV row {i}")
    for name, g in groups.items():
        x = np.asarray(g["f"], dtype=np.float64)
        if x.size > 1 and not np.all(np.diff(x) > 0):
            raise TestDataError(f"Function {name}: the frequency is not strictly increasing")
        data.functions.append(TestFunction(name=name, node=g["node"], dof=g["dof"], freq=x, values=np.asarray(g["v"]), kind=g["kind"], unit=g["unit"],
                                           quantity=g["quantity"], per=g["per"]))


def load_test_file(path: str | Path, fmt: str = "auto", max_bytes: int = MAX_FILE_BYTES, rotation: str = DEFAULT_ROTATION, damping_unit: str = "") -> TestData:
    """Reads a CSV or UFF (58 / 55) test file (`fmt` = auto | csv | uff; auto by the extension and the first line). `rotation` and `damping_unit`: see `parse_uff`."""
    raw = _read_raw(path, max_bytes)
    text = decode_bytes(raw)
    name = Path(str(path)).name
    f = fmt.strip().lower()
    if f == "auto":
        ext = Path(name).suffix.lower()
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        f = "uff" if ext in (".uff", ".unv", ".uf") or first == "-1" else "csv"
    if f == "uff":
        td = parse_uff(text, clean_label(name, 80), rotation, damping_unit)
    elif f == "csv":
        td = parse_csv(text, clean_label(name, 80))
    else:
        raise TestDataError("format must be auto, csv or uff")
    td.sha256 = [hashlib.sha256(raw).hexdigest()]
    return td


# ------------------------------------------------------------------ writers
def _e13(x: float) -> str:
    return f"{x:13.5E}"


def _e20(x: float) -> str:
    return f"{x:20.12E}"


def _chunks(vals: list[float], per: int = 6) -> list[str]:
    return ["".join(_e13(v) for v in vals[i:i + per]) for i in range(0, len(vals), per)]


def write_uff55(modes: list[TestMode], title: str = "FEA Studio modes") -> str:
    """Dataset 55 of the modes (real or complex shapes, 3 translational values per node; rotations are not written)."""
    out: list[str] = []
    for m in modes:
        nodes: dict[str, dict[str, complex | float]] = {}
        for (n, d), v in m.shape.items():
            if d in TRANSLATIONS:
                nodes.setdefault(n, {})[d] = v
        cplx = any(isinstance(v, complex) and v.imag != 0 for v in m.shape.values())
        out += ["    -1", "    55", clean_label(title) or "modes", f"mode {m.number}", "NONE", "NONE", "NONE",
                f"{1:10d}{2:10d}{2:10d}{8:10d}{5 if cplx else 2:10d}{3:10d}",
                f"{2:10d}{4:10d}{1:10d}{m.number:10d}", _chunks([m.frequency_hz, 0.0, m.damping or 0.0, 0.0])[0]]
        for n, comp in nodes.items():
            if not n.lstrip("-").isdigit():
                raise TestDataError(f"UFF node numbers must be integers (node '{n}')")
            out.append(f"{int(n):10d}")
            vals: list[float] = []
            for d in TRANSLATIONS:
                v = comp.get(d, 0.0)
                vals += [complex(v).real, complex(v).imag] if cplx else [float(complex(v).real)]
            out += _chunks(vals)
        out.append("    -1")
    return "\n".join(out) + "\n"


def _exact_even(x: np.ndarray) -> tuple[float, float] | None:
    """(first value, increment) when the grid is regular AND both are exactly what the 13.5E fields of the record hold; else None (the abscissa is then written explicitly)."""
    if x.size < 3:
        return None
    dx = float(x[1] - x[0])
    if not dx > 0 or not np.allclose(np.diff(x), dx, rtol=1e-12, atol=0.0):
        return None
    x0 = float(x[0])
    if float(f"{x0:.5E}") != x0 or float(f"{dx:.5E}") != dx or not np.array_equal(x0 + dx * np.arange(x.size), x):
        return None
    return x0, dx


def write_uff58(functions: list[TestFunction], title: str = "FEA Studio functions") -> str:
    """Dataset 58 of the functions of frequency in double precision (ordinate type 4 / 6): the name on the ID line 1, the ordinate quantity / unit and the denominator in the records
    9 / 10, an even abscissa only when its increment is exactly what the record holds (else the abscissa is written value by value)."""
    out: list[str] = []
    for k, f in enumerate(functions, 1):
        if f.node and not f.node.lstrip("-").isdigit():
            raise TestDataError(f"UFF node numbers must be integers (node '{f.node}')")
        func_type = f.func_type if f.func_type is not None else 9 if f.kind == "psd" else 4 if f.kind == "frf" else 12
        direction = (DOF_NAMES.index(f.dof) + 1) if f.dof else 0
        ref_dir = (DOF_NAMES.index(f.reference_dof) + 1) if f.reference_dof else 0
        cplx = bool(np.iscomplexobj(f.values))
        even = _exact_even(np.asarray(f.freq, float))
        x0, dx = even if even is not None else (0.0, 0.0)
        qs = _QUANTITY_SPECIFIC.get(f.quantity, 0)
        ds = _QUANTITY_SPECIFIC.get(f.per, 0)
        out += ["    -1", "    58", clean_label(f.name, 80) or "function", clean_label(title), "NONE", "NONE", "NONE",
                f"{func_type:5d}{k:10d}{0:5d}{0:10d} {'NONE':<10s}{int(f.node or 0):10d}{direction:4d} {'NONE':<10s}{int(f.reference_node or 0):10d}{ref_dir:4d}",
                f"{6 if cplx else 4:10d}{f.freq.size:10d}{1 if even is not None else 0:10d}{x0:13.5E}{dx:13.5E}{0.0:13.5E}",
                f"{18:10d}{0:5d}{0:5d}{0:5d} {'Frequency':<20s} {'Hz':<20s}",
                f"{qs:10d}{0:5d}{0:5d}{0:5d} {(f.quantity or 'Response').capitalize():<20s} {clean_unit(f.unit) or 'NONE':<20s}",
                f"{ds:10d}{0:5d}{0:5d}{0:5d} {(f.per or 'NONE').capitalize():<20s} {clean_unit(f.per_unit) or 'NONE':<20s}",
                f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20s} {'NONE':<20s}"]
        vals: list[float] = []
        for x, y in zip(f.freq, f.values):
            c = complex(y)
            row = [c.real, c.imag] if cplx else [c.real]
            vals += row if even is not None else [float(x)] + row
        out += ["".join(_e20(v) for v in vals[i:i + 4]) for i in range(0, len(vals), 4)]
        out.append("    -1")
    return "\n".join(out) + "\n"


def write_modes_csv(modes: list[TestMode]) -> str:
    """Mode shapes as CSV (round trip: `parse_csv`); numbers with full precision, text cells through `export_csv.text_cell`."""
    from ..export_csv import text_cell
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["mode", "frequency_hz", "damping", "damping_unit", "node", "dof", "value", "imag"])
    for m in modes:
        for (n, d), v in m.shape.items():
            c = complex(v)
            w.writerow([m.number, repr(m.frequency_hz), "" if m.damping is None else repr(m.damping), "" if m.damping is None else "ratio", text_cell(n), d, repr(c.real),
                        repr(c.imag) if isinstance(v, complex) else ""])
    return buf.getvalue()


def write_functions_csv(functions: list[TestFunction]) -> str:
    from ..export_csv import text_cell
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["function", "node", "dof", "kind", "unit", "quantity", "per", "frequency_hz", "real", "imag"])
    for f in functions:
        for x, y in zip(f.freq, f.values):
            c = complex(y)
            w.writerow([text_cell(f.name), text_cell(f.node), f.dof, f.kind, text_cell(f.unit), f.quantity, f.per, repr(float(x)), repr(c.real),
                        repr(c.imag) if np.iscomplexobj(f.values) else ""])
    return buf.getvalue()

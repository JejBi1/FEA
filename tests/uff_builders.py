"""Builders of UFF test files in the record layouts of public sample files (pyuff / OpenModal style), made here from the published dataset descriptions: the sample files themselves
are NOT copied into the repository (their provenance and licence cannot be checked from the files), only the layouts they exercise - numbers written without a space between them,
a complex mode dataset (analysis type 3) with a header record that holds numbers, a controller PSD without node / direction whose abscissa has no data type, coordinate-system
datasets 2411 / 15 / 2420 with a rotated sensor system, and a unit dataset 164 with four factors."""
from __future__ import annotations

import numpy as np

D = "    -1"


def e13(v: float) -> str:
    """A Fortran-like E13.5 number with a three-digit exponent (the 13 characters leave no space to the next number when v < 0)."""
    mant, exp = f"{v:.5E}".split("E")
    return f"{mant}E{int(exp):+04d}".rjust(13)


def e26(v: float) -> str:
    return f"{v:26.16e}"


def ds58(name: str = "FRF H1", ftype: int = 4, node: int = 1, direction: int = 3, ref_node: int = 0, ref_dir: int = 0, x: np.ndarray | None = None, y: np.ndarray | None = None,
         spacing: int = 0, ord_dtype: int = 12, ord_unit: str = "g", den_dtype: int = 13, den_unit: str = "N", abs_dtype: int = 18, abs_label: str = "Frequency", abs_unit: str = "Hz",
         rsp_name: str = "NONE", stuck: bool = False, ord_type: int | None = None, id2: str = "NONE") -> list[str]:
    x = np.arange(1.0, 7.0) if x is None else np.asarray(x, float)
    y = -np.arange(1.0, len(x) + 1.0) if y is None else np.asarray(y)
    cplx = np.iscomplexobj(y)
    ot = ord_type if ord_type is not None else (5 if cplx else 2)
    fmt = e13 if stuck else (lambda v: f"{v:13.5E}")
    vals: list[float] = []
    for k in range(len(x)):
        row = [float(np.real(y[k]))] + ([float(np.imag(y[k]))] if cplx else [])
        vals += row if spacing == 1 else [float(x[k])] + row
    out = [D, "    58", name, id2, "NONE", "NONE", "NONE",
           f"{ftype:5d}{1:10d}{0:5d}{0:10d} {rsp_name:<10s}{node:10d}{direction:4d} {'NONE':<10s}{ref_node:10d}{ref_dir:4d}",
           f"{ot:10d}{len(x):10d}{spacing:10d}{float(x[0]) if spacing == 1 else 0.0:13.5E}{float(x[1] - x[0]) if spacing == 1 and len(x) > 1 else 0.0:13.5E}{0.0:13.5E}",
           f"{abs_dtype:10d}{0:5d}{0:5d}{0:5d} {abs_label:<20s} {abs_unit:<20s}", f"{ord_dtype:10d}{0:5d}{0:5d}{0:5d} {'Response':<20s} {ord_unit:<20s}",
           f"{den_dtype:10d}{0:5d}{0:5d}{0:5d} {'Reference':<20s} {den_unit:<20s}", f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20s} {'NONE':<20s}"]
    out += ["".join(fmt(v) for v in vals[i:i + 6]) for i in range(0, len(vals), 6)]
    return out + [D]


def ds55(mode: int, freq: float, nodes: dict[int, list[complex | float]], analysis: int = 2, damping: float = 0.0, stuck: bool = False, eig: complex | None = None,
         head5: str = "NONE") -> list[str]:
    """Dataset 55 of one mode. `nodes` {node: [3 or 6 values]}; complex values -> data type 5."""
    vals0 = next(iter(nodes.values()))
    cplx = any(isinstance(v, complex) for vs in nodes.values() for v in vs)
    fmt = e13 if stuck else (lambda v: f"{v:13.5E}")
    nval = len(vals0)
    out = [D, "    55", "TEST", f"mode {mode}", "NONE", "NONE", head5,
           f"{1:10d}{analysis:10d}{2 if nval == 3 else 3:10d}{8:10d}{5 if cplx else 2:10d}{nval:10d}"]
    if analysis == 3:
        lam = eig if eig is not None else complex(-damping * 2 * np.pi * freq, 2 * np.pi * freq)
        out += [f"{2:10d}{6:10d}{0:10d}{mode:10d}", "".join(fmt(v) for v in (lam.real, lam.imag, 0.0, 0.0, 0.0, 0.0))]
    else:
        out += [f"{2:10d}{4:10d}{1:10d}{mode:10d}", "".join(fmt(v) for v in (freq, 0.0, damping, 0.0))]
    for n, vs in nodes.items():
        out.append(f"{n:10d}")
        flat: list[float] = []
        for v in vs:
            flat += [complex(v).real, complex(v).imag] if cplx else [float(np.real(v))]
        out += ["".join(fmt(v) for v in flat[i:i + 6]) for i in range(0, len(flat), 6)]
    return out + [D]


def ds2411(node_cs: dict[int, int]) -> list[str]:
    out = [D, "  2411"]
    for n, cs in node_cs.items():
        out += [f"{n:10d}{0:10d}{cs:10d}{8:10d}", f"{0.1 * n:25.16e}{0.0:25.16e}{0.0:25.16e}"]
    return out + [D]


def ds15(node_cs: dict[int, int]) -> list[str]:
    out = [D, "    15"]
    for n, cs in node_cs.items():
        out.append(f"{n:10d}{0:10d}{cs:10d}{8:10d}" + "".join(e13(v) for v in (0.1 * n, 0.0, 0.0)))
    return out + [D]


def ds2420(systems: dict[int, tuple[int, np.ndarray]], name_blank: bool = False) -> list[str]:
    """Dataset 2420: {label: (type, 3 x 3 matrix)}; the origin is zero."""
    out = [D, "  2420", f"{1:10d}", "None"]
    for n, (ctype, r) in systems.items():
        out += [f"{n:10d}{ctype:10d}{8:10d}", "" if name_blank else f"CS{n}"]
        rows = np.vstack([np.asarray(r, float), np.zeros((1, 3))])
        out += ["".join(e26(v) for v in row) for row in rows]
    return out + [D]


def ds164(factors: tuple[float, float, float] = (1.0, 1.0, 1.0), offset: float = 0.0) -> list[str]:
    return [D, "   164", f"{9:10d}{'User unit system':<20s}{1:10d}", "".join(f"{v:25.16E}".replace("E", "D") for v in factors), f"{offset:25.16E}".replace("E", "D"), D]


def rot_y(c: float = -0.44807361612917013, s: float = 0.89399666360055785) -> np.ndarray:
    """The rotation of the sensors of the OpenModal beam sample (rows = local axes in the global system)."""
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def join(*datasets: list[str]) -> str:
    return "\n".join(line for ds in datasets for line in ds) + "\n"

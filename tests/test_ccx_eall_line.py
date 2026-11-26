"""CalculiX reads at most 16 entries from a data line: the element set of all parts (`*ELSET, ELSET=EALL`) of an assembly of more than 16 parts must continue on the next line
(found by the CubeSat-class benchmark of F14-S6: a modal run of a 27-part model stopped with "there should not be more than 16 entries in a line")."""
from __future__ import annotations

import io

import numpy as np

from fea_core.ccx.inp import InpMap, write_mesh
from fea_core.mesh import Mesh


def _mesh(n_parts: int) -> Mesh:
    nodes, tets, part = [], [], []
    for k in range(n_parts):
        o = 3.0 * k
        base = len(nodes)
        nodes += [[o, 0, 0], [o + 1, 0, 0], [o, 1, 0], [o, 0, 1]]
        tets.append([base, base + 1, base + 2, base + 3])
        part.append(k)
    return Mesh(np.array(nodes, float), np.array(tets, np.int64), np.array(part, np.int64), [f"Part {k}" for k in range(n_parts)])


def _eall_lines(n_parts: int) -> list[str]:
    buf = io.StringIO()
    write_mesh(buf, _mesh(n_parts), InpMap())
    lines = buf.getvalue().splitlines()
    i = lines.index("*ELSET, ELSET=EALL")
    out = []
    for line in lines[i + 1:]:
        if line.startswith("*") or line.startswith("**"):
            break
        out.append(line)
    return out


def test_element_set_of_all_parts_has_at_most_16_entries_per_line():
    for n in (1, 16, 17, 27, 40):
        lines = _eall_lines(n)
        assert all(len([e for e in ln.split(",") if e.strip()]) <= 16 for ln in lines), (n, lines)
        names = [e.strip() for ln in lines for e in ln.split(",") if e.strip()]
        assert names == [f"P{k + 1}" for k in range(n)]                                     # every part, in order
        assert all(ln.endswith(",") for ln in lines[:-1]) and not lines[-1].endswith(",")      # continued after a trailing comma, the last line is closed

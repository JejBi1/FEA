"""Static check of a CalculiX deck for DOFs that nothing stiffens: `analyze(text)` -> list of problems.

The bug class (found twice): an auxiliary node that only a light element (SPRING1 / SPRING2 / SPRINGA / MASS / GAPUNI) or an *EQUATION touches. ccx makes all
three translations of such a node unknowns; the spring stiffens ONE of them, the equation makes ONE dependent, the other two are empty rows. PaStiX let them
pass, the SPOOLES of the own CalculiX stops ("matrix found to be singular" or the exit code 0xFFFFFFFF). Every such node must have its remaining DOFs held by
*BOUNDARY in the block of EVERY step (an `OP=NEW` of a later step forgets the earlier ones).

The model of the check (it is a lint, not a solver):
  - an active DOF = DOF 1–3 of every node of any element (4–6 too for U1 and shells), DOF 11 of the nodes of heat elements (DC3D*);
  - it is fine when a solid / shell / U1 / heat element stiffens it, a SPRING1 / SPRING2 acts in it, it is dependent in an *EQUATION (the first term),
    a node of a *RIGID BODY set, a REF / ROT node of a rigid body or coupling, an independent term of an *EQUATION (the dependent DOF's stiffness
    reaches it), or the *BOUNDARY of the step fixes it. SPRINGA / GAPUNI stiffen DOF 1–3 of their nodes (they act along a line only – the check is not
    exact for them, a node held only by them in a transverse direction is not reported);
  - a spring with `ORIENTATION=` acts along local axes: it stiffens all three DOFs of the triple (translations or rotations) of the named DOF.
"""
from __future__ import annotations

SOLID = {"C3D4", "C3D6", "C3D8", "C3D8I", "C3D8R", "C3D10", "C3D15", "C3D20", "C3D20R", "CPS3", "CPS4", "CPS6", "CPS8", "CPE3", "CPE4", "CPE6",
         "CPE8", "CAX3", "CAX4", "CAX6", "CAX8"}
SIX = {"U1", "S3", "S4", "S6", "S8", "B31", "B32"}                # nodes with rotations: all six DOFs are stiffened
HEAT = {"DC3D4", "DC3D6", "DC3D8", "DC3D10", "DC3D15", "DC3D20"}
LIGHT = {"SPRING1", "SPRING2", "SPRINGA", "MASS", "GAPUNI"}


def _params(line: str) -> tuple[str, dict]:
    parts = [p.strip() for p in line.strip().split(",")]
    kw = parts[0].upper()
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip().upper()] = v.strip()
        elif p:
            kv[p.upper()] = ""
    return kw, kv


def _cards(text: str):
    """(keyword, parameters, data lines) of every card; the lines of an element card are joined across the trailing commas."""
    cards = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            kw, kv = _params(line)
            cards.append([kw, kv, []])
        elif cards:
            d = cards[-1][2]
            if d and d[-1].endswith(","):
                d[-1] += " " + line
            else:
                d.append(line)
    return cards


def _ints(line: str) -> list[int]:
    return [int(float(p)) for p in line.replace(",", " ").split()]


def analyze(text: str) -> list[str]:
    """Problems of the deck (empty = none): one line per step and node: 'step 2: node 17: DOF 2, 3 have no stiffness and are not held'."""
    cards = _cards(text)
    nsets: dict[str, set[int]] = {}
    nodes: set[int] = set()
    elsets: dict[str, list[tuple[str, list[int]]]] = {}
    elems: list[tuple[str, list[int], str]] = []
    springs: dict[str, tuple[list[int], bool]] = {}              # elset → (DOFs of the first data line, oriented)
    stiff: set[tuple[int, int]] = set()
    active: set[tuple[int, int]] = set()
    dep: set[tuple[int, int]] = set()
    all_dep_nodes: set[int] = set()
    steps: list[list] = []                                       # per step: [(op_new, [boundary lines])]
    base_bc: list[str] = []
    in_step = False

    def nodeset(name: str) -> set[int]:
        return nsets.setdefault(name.upper(), set())

    for kw, kv, data in cards:
        if kw == "*NODE":
            ids = [int(d.split(",")[0]) for d in data]
            nodes.update(ids)
            if "NSET" in kv:
                nodeset(kv["NSET"]).update(ids)
        elif kw == "*NSET":
            ns = nodeset(kv.get("NSET", ""))
            for d in data:
                if "GENERATE" in kv:
                    v = _ints(d)
                    ns.update(range(v[0], v[1] + 1, v[2] if len(v) > 2 else 1))
                else:
                    for tok in [t.strip() for t in d.split(",") if t.strip()]:
                        if tok.lstrip("-").isdigit():
                            ns.add(int(tok))
                        else:
                            ns.update(nsets.get(tok.upper(), set()))
        elif kw == "*ELEMENT":
            typ = kv.get("TYPE", "").upper()
            for d in data:
                v = _ints(d)
                elems.append((typ, v[1:], kv.get("ELSET", "").upper()))
                if kv.get("ELSET"):
                    elsets.setdefault(kv["ELSET"].upper(), []).append((typ, v[1:]))
        elif kw == "*SPRING":
            if data:
                springs[kv.get("ELSET", "").upper()] = (_ints(data[0]), "ORIENTATION" in kv)
        elif kw == "*EQUATION":
            vals: list[float] = []
            for d in data:
                vals += [float(p) for p in d.replace(",", " ").split()]
            i = 0
            while i < len(vals):
                n = int(vals[i])
                terms = [(int(vals[i + 1 + 3 * k]), int(vals[i + 2 + 3 * k])) for k in range(n)]
                i += 1 + 3 * n
                dep.add(terms[0])
                for t in terms[1:]:
                    stiff.add(t)
        elif kw == "*RIGID BODY":
            all_dep_nodes |= nodeset(kv.get("NSET", ""))
            for key in ("REF NODE", "ROT NODE"):
                if key in kv:
                    n = int(kv[key])
                    stiff.update({(n, d) for d in range(1, 7)})
                    active.update({(n, d) for d in range(1, 4)})
        elif kw == "*COUPLING":
            if "REF NODE" in kv:
                n = int(kv["REF NODE"])
                stiff.update({(n, d) for d in range(1, 7)})
                active.update({(n, d) for d in range(1, 4)})
        elif kw == "*STEP":
            in_step = True
            steps.append([])
        elif kw == "*END STEP":
            in_step = False
        elif kw == "*BOUNDARY":
            blk = ("OP" in kv and kv["OP"].upper() == "NEW", list(data))
            if in_step:
                steps[-1].append(blk)
            else:
                base_bc += data

    for typ, ns, es in elems:
        if typ in SOLID:
            for n in ns:
                stiff.update({(n, d) for d in (1, 2, 3)})
                active.update({(n, d) for d in (1, 2, 3)})
        elif typ in SIX:
            for n in ns:
                stiff.update({(n, d) for d in range(1, 7)})
                active.update({(n, d) for d in range(1, 7)})
        elif typ in HEAT:
            for n in ns:
                stiff.add((n, 11))
                active.add((n, 11))
        elif typ in ("SPRING1", "SPRING2"):
            dofs, oriented = springs.get(es, ([], False))
            for k, n in enumerate(ns):
                active.update({(n, d) for d in (1, 2, 3)})
                d0 = dofs[k] if k < len(dofs) else None
                if d0 is None:
                    continue
                triple = range(1, 4) if d0 <= 3 else range(4, 7)
                for d in (triple if oriented else [d0]):
                    stiff.add((n, d))
        elif typ in ("SPRINGA", "GAPUNI"):
            for n in ns:
                stiff.update({(n, d) for d in (1, 2, 3)})
                active.update({(n, d) for d in (1, 2, 3)})
        elif typ == "MASS":
            for n in ns:
                active.update({(n, d) for d in (1, 2, 3)})

    def fixed_by(lines: list[str]) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            tgt = parts[0]
            ids = {int(tgt)} if tgt.lstrip("-").isdigit() else nsets.get(tgt.upper(), set())
            d1 = int(float(parts[1]))
            d2 = int(float(parts[2])) if len(parts) > 2 and parts[2] else d1
            out |= {(n, d) for n in ids for d in range(d1, d2 + 1)}
        return out

    problems = []
    cur = fixed_by(base_bc)
    for si, blocks in enumerate(steps or [[]], 1):
        for op_new, lines in blocks:
            cur = fixed_by(lines) if op_new else cur | fixed_by(lines)
        bad: dict[int, list[int]] = {}
        for (n, d) in sorted(active):
            if (n, d) in stiff or (n, d) in dep or n in all_dep_nodes or (n, d) in cur:
                continue
            bad.setdefault(n, []).append(d)
        for n, ds in sorted(bad.items()):
            problems.append(f"step {si}: node {n}: DOF {', '.join(map(str, ds))} have no stiffness and are not held")
    return problems

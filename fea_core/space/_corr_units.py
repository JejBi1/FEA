"""Units of the measured functions of the vibration-test correlation (F14 stage S5): a small, strict parser of unit labels and the dimensional check that stops a test curve in one
quantity (m/s2) being compared with a model curve in another (g, mm).

A label ("g", "m/s^2", "(m/s2)/N", "g2/Hz", "mm/N") is read into exponents of (metre, second, kilogram) and a factor to SI. A label that is empty, "NONE" or not understood gives
`None` - the caller then cannot confirm the comparison (the row is UNCONFIRMED), it never guesses. The standard acceleration is the one of the program (g = 9.81 m/s2; the
dynamics module uses 9810 mm/s2).

The prefixes are case-aware: "mN" is a millinewton and "MN" a meganewton, "Mg" is a megagram (1000 kg, a mass) and "mg" a milli-g (an acceleration; "mg" as a milligram makes no
sense for a response). A label that is written in capitals only ("M/S^2", "G", "HZ") is read case-insensitively where that is unambiguous. An underscore is no operator, and
"ms2" (m/s2 or ms squared?) is refused: write "m/s^2" (or "ms^2").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

G_SI = 9.81                                            # m/s2 (the program's g = 9810 mm/s2)
Dims = tuple[int, int, int]                            # exponents of (m, s, kg)

QUANTITY_DIMS: dict[str, Dims] = {"displacement": (1, 0, 0), "velocity": (1, -1, 0), "acceleration": (1, -2, 0), "force": (1, -2, 1)}
QUANTITIES = tuple(QUANTITY_DIMS)
FE_UNIT_LABEL = {"displacement": "mm", "velocity": "mm/s", "acceleration": "g"}      # the units of the model curves
_ONE: Dims = (0, 0, 0)


@dataclass(frozen=True)
class Unit:
    dims: Dims
    factor: float                                      # value in SI = value in this unit * factor

    def __mul__(self, o: "Unit") -> "Unit":
        (a, b, c), (d, e, f) = self.dims, o.dims
        return Unit((a + d, b + e, c + f), self.factor * o.factor)

    def inv(self) -> "Unit":
        a, b, c = self.dims
        return Unit((-a, -b, -c), 1.0 / self.factor)

    def power(self, n: int) -> "Unit":
        a, b, c = self.dims
        return Unit((a * n, b * n, c * n), self.factor ** n)


_ATOMS: dict[str, Unit] = {                            # case-sensitive (the prefix letters decide: mN / MN, mg / Mg)
    "1": Unit(_ONE, 1.0),
    "m": Unit((1, 0, 0), 1.0), "mm": Unit((1, 0, 0), 1e-3), "cm": Unit((1, 0, 0), 1e-2), "um": Unit((1, 0, 0), 1e-6), "in": Unit((1, 0, 0), 0.0254), "ft": Unit((1, 0, 0), 0.3048),
    "s": Unit((0, 1, 0), 1.0), "sec": Unit((0, 1, 0), 1.0), "ms": Unit((0, 1, 0), 1e-3),
    "Hz": Unit((0, -1, 0), 1.0),
    "kg": Unit((0, 0, 1), 1.0), "Mg": Unit((0, 0, 1), 1e3),
    "N": Unit((1, -2, 1), 1.0), "kN": Unit((1, -2, 1), 1e3), "MN": Unit((1, -2, 1), 1e6), "mN": Unit((1, -2, 1), 1e-3), "lbf": Unit((1, -2, 1), 4.4482216),
    "g": Unit((1, -2, 0), G_SI), "mg": Unit((1, -2, 0), 1e-3 * G_SI),
}
_FOLD: dict[str, Unit] = {}                            # the lower-case spelling of every atom whose lower case is unique (used only when the exact spelling is not an atom)
_SHARED: set[str] = set()
for _k, _u in _ATOMS.items():
    _low = _k.lower()
    if _low in _FOLD or _low in _SHARED:
        _FOLD.pop(_low, None)
        _SHARED.add(_low)
    else:
        _FOLD[_low] = _u
_TOKEN = re.compile(r"\s*(?:([A-Za-z]+)|(\d+)|(\^)|(\*)|(/)|(\()|(\)))")
_BAD = {"", "none", "-", "n/a", "?", "unknown"}


def _tokens(s: str) -> list[str] | None:
    out: list[str] = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if m is None or m.end() == pos:
            return None
        out.append(next(g for g in m.groups() if g is not None))
        pos = m.end()
    return out


def parse_unit(label: str | None) -> Unit | None:
    """The unit of a label, or None when it is empty / "NONE" / not understood (never a guess)."""
    s = (label or "").strip()
    if s.lower() in _BAD:
        return None
    s = s.replace("²", "^2").replace("³", "^3").replace("µ", "u").replace("μ", "u").replace("·", "*").replace("×", "*").replace("\\", "/")
    s = re.sub(r"\[|\{", "(", s)
    s = re.sub(r"\]|\}", ")", s)
    toks = _tokens(s)
    if not toks or len(toks) > 60:
        return None
    pos = 0

    def atom() -> Unit | None:
        nonlocal pos
        if pos >= len(toks):
            return None
        t = toks[pos]
        if t == "(":
            pos += 1
            v = expr()
            if v is None or pos >= len(toks) or toks[pos] != ")":
                return None
            pos += 1
            u: Unit | None = v
        elif t.isdigit():
            if t != "1":
                return None                                  # only the number one is a unit
            pos += 1
            u = _ATOMS["1"]
        elif t in _ATOMS or t.lower() in _FOLD:
            pos += 1
            u = _ATOMS[t] if t in _ATOMS else _FOLD[t.lower()]
            if t.lower() == "ms" and pos < len(toks) and toks[pos].isdigit():
                return None                                   # "ms2": m/s2 or ms squared? never a guess
        else:
            return None
        # exponent: ^n or a digit string right after the atom (m/s2)
        if pos < len(toks) and toks[pos] == "^":
            if pos + 1 >= len(toks) or not toks[pos + 1].isdigit():
                return None
            n = int(toks[pos + 1])
            pos += 2
            u = u.power(n)
        elif pos < len(toks) and toks[pos].isdigit() and t != "1" and t != ")":
            n = int(toks[pos])
            pos += 1
            u = u.power(n)
        return u

    def expr() -> Unit | None:
        nonlocal pos
        u = atom()
        while u is not None and pos < len(toks) and toks[pos] in ("*", "/"):
            op = toks[pos]
            pos += 1
            v = atom()
            if v is None:
                return None
            u = u * v if op == "*" else u * v.inv()
        return u

    u = expr()
    if u is None or pos != len(toks) or not (1e-12 < u.factor < 1e12):
        return None
    return u


def fe_unit(quantity: str, per: str = "none", psd: bool = False) -> Unit:
    """The unit of a model curve: quantity [g | mm | mm/s], divided by the normalisation (none | base_acceleration | force), or its PSD (squared, per Hz)."""
    if quantity not in FE_UNIT_LABEL:
        raise ValueError(f"no model unit for the quantity '{quantity}'")
    u = _ATOMS["g"] if quantity == "acceleration" else parse_unit(FE_UNIT_LABEL[quantity])
    assert u is not None
    if per == "base_acceleration":
        u = u * _ATOMS["g"].inv()
    elif per == "force":
        u = u * _ATOMS["N"].inv()
    elif per != "none":
        raise ValueError(f"unknown normalisation '{per}'")
    if psd:
        u = u.power(2) * _ATOMS["Hz"].inv()
    return u


def same_dims(a: Unit, b: Unit) -> bool:
    return a.dims == b.dims


def clean_unit(raw: object, limit: int = 40) -> str:
    """Plain text of a unit label from a file (keeps ^ * / ( ) and the superscripts, drops everything else)."""
    s = re.sub(r"[^\w .:+\-/#()^*²³·µ%]", "_", str(raw).strip(), flags=re.UNICODE)
    return s[:limit]


def unit_hint(label: str | None) -> str:
    """Why a label that parse_unit refused is refused, when there is a more helpful thing to say than "not understood"."""
    t = (label or "").strip().lower().replace(" ", "")
    if re.search(r"(^|[/*(])ms\d", t):
        return "'ms2' is ambiguous (m/s^2 or ms squared) - write m/s^2"
    if "_" in t:
        return "an underscore is not an operator - write * or /"
    return ""

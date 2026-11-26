"""Restricted evaluator of the sketch expressions (coordinates, radii, parameters).

Allowed: numbers, the operators + - * / % **, unary + -, parentheses, the names of the sketch parameters, the constant `pi` and the
functions sin cos tan asin acos atan atan2 sqrt abs min max hypot rad (degrees → radians) deg (radians → degrees). The angles of
the trigonometric functions are in radians (`sin(30*pi/180)` = `sin(rad(30))`). No attribute access, no subscripts, no calls of
anything else, the exponent of `**` is at most 8 in magnitude, the text at most MAX_LEN characters. The expression is walked
as a syntax tree (it is never passed to eval).
"""
from __future__ import annotations

import ast
import math
import re

MAX_LEN = 400
MAX_NODES = 250
MAX_ABS = 1e12                                  # larger values are certainly an input error (a coordinate in mm)
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


class SketchError(ValueError):
    """Error of the sketch with a clear text; `entities` = the ids of the entities (or points) the message is about."""

    def __init__(self, message: str, entities: list[str] | tuple[str, ...] = ()):
        super().__init__(message)
        self.entities = list(entities)


FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
         "atan2": math.atan2, "sqrt": math.sqrt, "abs": abs, "min": min, "max": max, "hypot": math.hypot,
         "rad": math.radians, "deg": math.degrees}
CONSTS = {"pi": math.pi}
RESERVED = set(FUNCS) | set(CONSTS)


def check_name(name: str) -> str:
    """Name of a parameter: a letter, then letters, digits and underscores (at most 32), not a function or a constant."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SketchError(f"Invalid parameter name {name!r}: use a letter followed by letters, digits or underscores (at most 32 characters)")
    if name in RESERVED:
        raise SketchError(f"'{name}' is a built-in function or constant – choose another parameter name")
    return name


def _parse(text: str) -> ast.AST:
    if not isinstance(text, str):
        raise SketchError(f"Expression {text!r} must be a number or a text")
    s = text.strip().replace(",", ".") if re.fullmatch(r"\s*[-+]?\d+,\d+\s*", text or "") else (text or "").strip()
    if not s:
        raise SketchError("Empty expression")
    if len(s) > MAX_LEN:
        raise SketchError(f"Expression is too long (at most {MAX_LEN} characters)")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise SketchError(f"Invalid expression '{s[:60]}': {e.msg}") from None
    except (ValueError, RecursionError, MemoryError, OverflowError):      # e.g. an integer literal over the digit limit, a very deep nesting
        raise SketchError(f"Invalid expression '{s[:60]}'") from None
    n = sum(1 for _ in ast.walk(tree))
    if n > MAX_NODES:
        raise SketchError("Expression is too complicated")
    return tree.body


def names_in(text) -> set[str]:
    """Parameter names used by the expression (without the constants and the functions)."""
    if isinstance(text, (int, float)):
        return set()
    out = set()
    func_nodes = set()
    tree = _parse(text)           # SketchError for everything that cannot be parsed
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_nodes.add(id(node.func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in func_nodes and node.id not in CONSTS:
            out.add(node.id)
    return out


def _ev(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SketchError("Only numbers are allowed in an expression")
        try:
            return float(v)
        except OverflowError:                                          # an integer literal beyond the float range
            raise SketchError("Number too large in an expression") from None
    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        if node.id in CONSTS:
            return CONSTS[node.id]
        raise SketchError(f"Unknown parameter '{node.id}'")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left, env), _ev(node.right, env)
        op = node.op
        try:
            if isinstance(op, ast.Add):
                r = a + b
            elif isinstance(op, ast.Sub):
                r = a - b
            elif isinstance(op, ast.Mult):
                r = a * b
            elif isinstance(op, ast.Div):
                r = a / b
            elif isinstance(op, ast.Mod):
                r = math.fmod(a, b)                                    # fmod(x, 0) raises ValueError, not ZeroDivisionError
            elif isinstance(op, ast.Pow):
                if abs(b) > 8:
                    raise SketchError("The exponent of ** may be at most 8 in magnitude")
                r = a ** b
                if isinstance(r, complex):
                    raise SketchError("Result of ** is not a real number")
            else:
                raise SketchError("Operator is not allowed in an expression")
        except ZeroDivisionError:
            raise SketchError("Division by zero in an expression") from None
        except OverflowError:
            raise SketchError("Number too large in an expression") from None
        except ArithmeticError:
            raise SketchError("Invalid arithmetic in an expression") from None
        except ValueError:                                             # 5 % 0, fmod(inf, x) …
            raise SketchError("Invalid operands in an expression (modulo by zero?)") from None
        return r
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords or len(node.args) > 3:
            raise SketchError("Only the functions " + ", ".join(sorted(FUNCS)) + " are allowed in an expression")
        args = [_ev(a, env) for a in node.args]
        try:
            return float(FUNCS[node.func.id](*args))
        except (ValueError, TypeError, ArithmeticError):
            raise SketchError(f"Invalid argument of {node.func.id}()") from None
    raise SketchError("Only numbers, parameters, + - * / % **, parentheses and the functions "
                      + ", ".join(sorted(FUNCS)) + " are allowed in an expression")


def evaluate(text, env: dict[str, float] | None = None) -> float:
    """Value of an expression (number or text) with the parameters `env`; always finite and below MAX_ABS in magnitude."""
    if isinstance(text, bool):
        raise SketchError("Expression must be a number or a text")
    try:
        if isinstance(text, (int, float)):
            v = float(text)
        else:
            v = _ev(_parse(text), env or {})
    except SketchError:
        raise
    except (ArithmeticError, ValueError, TypeError, RecursionError, MemoryError):    # nothing but a SketchError leaves the evaluator
        raise SketchError(f"Value of '{str(text)[:60]}' is not a usable number") from None
    if not math.isfinite(v) or abs(v) > MAX_ABS:
        raise SketchError(f"Value of '{text}' is not a usable number")
    return v


def eval_params(params: dict) -> dict[str, float]:
    """Values of all parameters (expressions may use other parameters); a cycle or an unknown name = SketchError."""
    for k in params:
        check_name(k)
    deps = {k: names_in(v) for k, v in params.items()}
    for k, d in deps.items():
        for n in d:
            if n not in params:
                raise SketchError(f"Parameter '{k}': unknown parameter '{n}'")
    out: dict[str, float] = {}
    state: dict[str, int] = {}

    def visit(k: str, chain: tuple[str, ...]):
        if state.get(k) == 2:
            return
        if state.get(k) == 1:
            raise SketchError("Parameters refer to each other in a cycle: " + " → ".join(chain + (k,)))
        state[k] = 1
        for n in sorted(deps[k]):
            visit(n, chain + (k,))
        try:
            out[k] = evaluate(params[k], out)
        except SketchError as e:
            raise SketchError(f"Parameter '{k}': {e}") from None
        state[k] = 2
    for k in params:
        visit(k, ())
    return out

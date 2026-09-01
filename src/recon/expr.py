"""Safe evaluation of user-supplied numeric expressions.

The analysis tools let users type formulas (``(A - B) / A``, ``FFT(A)``,
``y / max(y)``) that are evaluated against numpy arrays. Handing such a string
straight to :func:`eval` is unsafe even with ``__builtins__`` cleared, because
Python's attribute machinery still reaches the interpreter internals::

    ().__class__.__bases__[0].__subclasses__()   # -> arbitrary code

This module evaluates expressions behind an AST whitelist instead: the string is
parsed, every node is checked against a permitted set, and only then is it
compiled and run in a namespace that contains nothing but the caller's variables
and a curated set of numeric helpers.

The whitelist is deliberately permissive about *math* (arithmetic, comparisons,
slicing, calls into ``np.*``) and strict about everything else. It is a
robustness boundary for a desktop tool handling scientific data, not a hardened
sandbox for hostile input.
"""

# Copyright (C) 2023 Dennis Lönard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import ast
import logging
from typing import Any, Callable, Mapping, Optional

import numpy as np

log = logging.getLogger(__name__)

# Reject ``a ** b`` when ``b`` is a literal above this, so a stray keystroke
# cannot freeze the GUI on an astronomically large integer power.
MAX_LITERAL_EXPONENT = 1000

# Modules whose attributes may be reached (``np.sqrt``, ``np.fft.fft``).
SAFE_MODULES: dict[str, Any] = {
    "np": np,
}

# Submodules that may be traversed one level deeper (``np.fft.fft``). Anything
# not listed here is unreachable, which is what keeps ``np.lib.npyio.load`` and
# ``np.ctypeslib`` out of range.
ALLOWED_SUBMODULES: frozenset[str] = frozenset({"fft", "linalg", "random", "ma", "polynomial"})

# Numpy entry points that read/write files or import Python objects. ``np.load``
# with ``allow_pickle=True`` is arbitrary code execution, so these are denied
# even though they are plain attributes of an allowed module.
DENIED_MODULE_ATTRS: frozenset[str] = frozenset({
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "fromregex", "memmap", "DataSource",
    "ctypeslib", "testing", "distutils", "f2py", "lib", "core", "info", "source",
})

# Bare functions available to every expression.
SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "sign": np.sign,
    "clip": np.clip,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "cumsum": np.cumsum,
    "diff": np.diff,
    "gradient": np.gradient,
    "interp": np.interp,
    "real": np.real,
    "imag": np.imag,
    "angle": np.angle,
    "conj": np.conj,
    "len": len,
}

# Constants available to every expression.
SAFE_CONSTANTS: dict[str, Any] = {
    "pi": np.pi,
    "e": np.e,
    "inf": np.inf,
    "nan": np.nan,
}

# AST node types the whitelist accepts. Everything else is rejected, so new
# Python syntax is denied by default rather than silently permitted.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.IfExp,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.MatMult, ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.UAdd, ast.USub, ast.Invert, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


class ExpressionError(ValueError):
    """Raised when an expression is malformed, unsafe, or fails to evaluate."""


def _describe(node: ast.AST) -> str:
    """Human-readable label for a rejected node."""
    return type(node).__name__


def _attribute_chain(node: ast.Attribute) -> Optional[list[str]]:
    """Return the dotted components of an attribute chain rooted at a plain name.

    ``np.fft.fft`` -> ``["np", "fft", "fft"]``;  ``(a + b).real`` -> ``None``.
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return parts


def _check_attribute(node: ast.Attribute) -> None:
    """Reject attribute access that is not a call into an allowed module."""
    # Dunder traversal is the standard sandbox escape; block every
    # underscore-prefixed attribute outright.
    if node.attr.startswith("_"):
        raise ExpressionError(f"Attribute '{node.attr}' is not allowed.")

    chain = _attribute_chain(node)
    if chain is None:
        raise ExpressionError("Attribute access is only allowed on modules such as 'np'.")

    root = chain[0]
    if root not in SAFE_MODULES:
        raise ExpressionError(f"Attribute access on '{root}' is not allowed.")

    if len(chain) == 2:
        if chain[1] in DENIED_MODULE_ATTRS:
            raise ExpressionError(f"'{root}.{chain[1]}' is not allowed.")
    elif len(chain) == 3:
        if chain[1] not in ALLOWED_SUBMODULES:
            raise ExpressionError(f"'{root}.{chain[1]}' is not allowed.")
    else:
        raise ExpressionError(f"Attribute chain '{'.'.join(chain)}' is too deep.")


def validate(expression: str, allowed_names: set[str]) -> ast.Expression:
    """Parse ``expression`` and reject anything outside the whitelist.

    :param expression: the formula text.
    :param allowed_names: variable/function names the expression may reference.
    :raises ExpressionError: if the text is not a single safe expression.
    :return: the parsed AST, ready to compile.
    """
    text = (expression or "").strip()
    if not text:
        raise ExpressionError("Expression is empty.")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Syntax error: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"{_describe(node)} is not allowed in an expression.")

        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ExpressionError(f"Name '{node.id}' is not allowed.")
            if node.id not in allowed_names:
                raise ExpressionError(f"Unknown name '{node.id}'.")

        elif isinstance(node, ast.Attribute):
            _check_attribute(node)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = node.right
            if isinstance(exponent, ast.Constant) and isinstance(exponent.value, (int, float)):
                if abs(exponent.value) > MAX_LITERAL_EXPONENT:
                    raise ExpressionError(
                        f"Exponent {exponent.value} exceeds the limit of {MAX_LITERAL_EXPONENT}."
                    )

    return tree


def build_namespace(
    variables: Mapping[str, Any],
    extra_functions: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> dict[str, Any]:
    """Assemble the evaluation namespace: helpers first, caller variables last."""
    namespace: dict[str, Any] = {"__builtins__": {}}
    namespace.update(SAFE_CONSTANTS)
    namespace.update(SAFE_FUNCTIONS)
    namespace.update(SAFE_MODULES)
    if extra_functions:
        namespace.update(extra_functions)
    namespace.update(variables)
    return namespace


def evaluate(
    expression: str,
    variables: Mapping[str, Any],
    extra_functions: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> Any:
    """Validate and evaluate ``expression`` against ``variables``.

    :param expression: the formula text, e.g. ``"(A - B) / A"``.
    :param variables: user data bound by name, e.g. ``{"A": arr_a, "B": arr_b}``.
    :param extra_functions: caller-supplied callables, e.g. ``{"FFT": fft_magnitude}``.
    :raises ExpressionError: on unsafe syntax or any failure during evaluation.
    :return: whatever the expression produced (usually an ndarray or scalar).
    """
    namespace = build_namespace(variables, extra_functions)
    allowed = {name for name in namespace if name != "__builtins__"}
    tree = validate(expression, allowed)

    try:
        code = compile(tree, filename="<expression>", mode="eval")
        return eval(code, namespace)  # noqa: S307 - namespace is whitelisted above
    except ExpressionError:
        raise
    except Exception as exc:
        log.debug("Expression '%s' failed to evaluate: %s", expression, exc)
        raise ExpressionError(str(exc)) from exc


def curve_variables(y: np.ndarray, x: np.ndarray, energy: float = 0.0) -> dict[str, Any]:
    """Bind the per-curve variables a Curve Transform expression may use.

    ``y`` / ``x`` are the curve's samples, ``i`` the sample index, ``n`` the point
    count and ``E`` the row's photon energy in eV.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    return {
        "y": y_arr,
        "x": x_arr,
        "i": np.arange(y_arr.size, dtype=np.float64),
        "n": float(y_arr.size),
        "E": float(energy),
    }


def evaluate_series(
    expression: str,
    y: np.ndarray,
    x: np.ndarray,
    energy: float = 0.0,
    length: Optional[int] = None,
) -> np.ndarray:
    """Evaluate a per-curve expression and coerce the result to a 1-D series.

    A scalar result is broadcast to ``length`` points, so ``mean(y)`` yields a flat
    line rather than failing. Any other shape is an error: a curve must stay a curve.

    :param expression: the formula, e.g. ``"y / max(y)"``.
    :param y: the curve's Y samples.
    :param x: the curve's X samples (index values when no X dataset is set).
    :param energy: the row's photon energy in eV, bound as ``E``.
    :param length: expected output length; defaults to ``len(y)``.
    :raises ExpressionError: on unsafe syntax, evaluation failure, or wrong shape.
    """
    variables = curve_variables(y, x, energy)
    expected = int(len(variables["y"]) if length is None else length)

    result: np.ndarray = np.asarray(evaluate(expression, variables), dtype=np.float64)

    if result.ndim == 0:
        return np.full(expected, float(result))
    if result.ndim != 1:
        raise ExpressionError(f"Result must be a 1-D series, got {result.ndim}-D.")
    if result.size != expected:
        raise ExpressionError(f"Result has {result.size} points, expected {expected}.")
    return result


def available_names(
    variables: Mapping[str, Any],
    extra_functions: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> list[str]:
    """Sorted list of names an expression may use; for help text and tooltips."""
    namespace = build_namespace(variables, extra_functions)
    return sorted(name for name in namespace if name != "__builtins__")

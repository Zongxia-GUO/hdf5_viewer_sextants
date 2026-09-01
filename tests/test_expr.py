"""Tests for the whitelisted expression evaluator (``src.recon.expr``)."""

import numpy as np
import pytest

from src.recon import expr


# ---------------------------------------------------------------------------
# Normal evaluation
# ---------------------------------------------------------------------------

def test_arithmetic_on_arrays():
    a = np.arange(5, dtype=float)
    b = np.ones(5)
    result = expr.evaluate("(A - B) / 2", {"A": a, "B": b})
    np.testing.assert_allclose(result, (a - b) / 2)


def test_bare_helper_functions():
    y = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(expr.evaluate("sqrt(y)", {"y": y}), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(expr.evaluate("y / max(y)", {"y": y}), y / 9.0)


def test_numpy_module_attribute_access():
    a = np.array([-1.0, 2.0])
    np.testing.assert_allclose(expr.evaluate("np.abs(A)", {"A": a}), [1.0, 2.0])


def test_nested_numpy_attribute_chain():
    a = np.arange(4, dtype=float)
    np.testing.assert_allclose(
        expr.evaluate("np.abs(np.fft.fft(A))", {"A": a}),
        np.abs(np.fft.fft(a)),
    )


def test_extra_functions_are_callable():
    a = np.arange(4, dtype=float)
    result = expr.evaluate("FFT(A)", {"A": a}, {"FFT": lambda d: np.abs(np.fft.fft(d))})
    np.testing.assert_allclose(result, np.abs(np.fft.fft(a)))


def test_constants_and_slicing():
    y = np.arange(10, dtype=float)
    np.testing.assert_allclose(expr.evaluate("y - mean(y[:5])", {"y": y}), y - 2.0)
    assert expr.evaluate("pi", {}) == pytest.approx(np.pi)


def test_comparison_and_where():
    y = np.array([1.0, 5.0, 3.0])
    np.testing.assert_allclose(expr.evaluate("where(y > 2, y, 0)", {"y": y}), [0.0, 5.0, 3.0])


def test_variables_shadow_helpers():
    # A caller variable must win over a same-named helper.
    assert expr.evaluate("e", {"e": 42.0}) == 42.0


# ---------------------------------------------------------------------------
# Rejection: sandbox escapes
# ---------------------------------------------------------------------------

def test_dunder_attribute_is_rejected():
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("A.__class__", {"A": np.zeros(3)})


def test_classic_subclasses_escape_is_rejected():
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("().__class__.__bases__[0].__subclasses__()", {})


def test_attribute_access_on_data_is_rejected():
    # Even a harmless-looking attribute: only modules may be traversed.
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("A.shape", {"A": np.zeros(3)})


def test_attribute_on_expression_result_is_rejected():
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("(A + 1).real", {"A": np.zeros(3)})


def test_dunder_name_is_rejected():
    with pytest.raises(expr.ExpressionError, match="not allowed"):
        expr.evaluate("__import__('os')", {})


@pytest.mark.parametrize(
    "bad",
    [
        "np.load('evil.npy')",           # pickle-based code execution
        "np.savetxt('out.txt', A)",      # writes to disk
        "np.fromfile('secret')",         # reads from disk
        "np.ctypeslib.as_array(A)",      # pointer machinery
        "np.lib.npyio.load('evil.npy')",  # denied submodule
        "np.testing.suppress_warnings()",
        "np.fft.helper.fftshift(A)",     # chain too deep
    ],
)
def test_dangerous_numpy_entry_points_are_rejected(bad):
    with pytest.raises(expr.ExpressionError):
        expr.evaluate(bad, {"A": np.zeros(4)})


def test_allowed_numpy_submodules_still_work():
    a = np.arange(4, dtype=float)
    np.testing.assert_allclose(expr.evaluate("np.fft.fftshift(A)", {"A": a}), np.fft.fftshift(a))
    assert expr.evaluate("np.linalg.norm(A)", {"A": a}) == pytest.approx(np.linalg.norm(a))


def test_unknown_name_is_rejected():
    with pytest.raises(expr.ExpressionError, match="Unknown name 'os'"):
        expr.evaluate("os", {})


def test_builtin_open_is_not_reachable():
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("open('secret.txt')", {})


def test_undefined_variable_is_rejected_before_evaluation():
    with pytest.raises(expr.ExpressionError, match="Unknown name 'B'"):
        expr.evaluate("A + B", {"A": np.zeros(3)})


# ---------------------------------------------------------------------------
# Rejection: syntax outside the whitelist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        "lambda: 1",
        "[i for i in range(3)]",
        "(x := 2)",
        "f'{A}'",
        "{'k': 1}",
        "{1, 2}",
        "A if A else A.__doc__",
    ],
)
def test_disallowed_syntax(bad):
    with pytest.raises(expr.ExpressionError):
        expr.evaluate(bad, {"A": np.zeros(3)})


def test_statements_are_syntax_errors():
    with pytest.raises(expr.ExpressionError, match="Syntax error"):
        expr.evaluate("import os", {})


def test_empty_expression_is_rejected():
    with pytest.raises(expr.ExpressionError, match="empty"):
        expr.evaluate("   ", {})


def test_huge_literal_exponent_is_rejected():
    with pytest.raises(expr.ExpressionError, match="Exponent"):
        expr.evaluate("2 ** 100000", {})


def test_reasonable_exponent_is_allowed():
    assert expr.evaluate("2 ** 10", {}) == 1024


# ---------------------------------------------------------------------------
# Failures during evaluation
# ---------------------------------------------------------------------------

def test_runtime_error_becomes_expression_error():
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("sqrt(A, B, 1, 2, 3)", {"A": np.zeros(3), "B": np.zeros(3)})


def test_shape_mismatch_becomes_expression_error():
    with pytest.raises(expr.ExpressionError):
        expr.evaluate("A + B", {"A": np.zeros(3), "B": np.zeros(4)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_available_names_lists_variables_and_helpers():
    names = expr.available_names({"A": np.zeros(3)}, {"FFT": np.abs})
    assert "A" in names and "FFT" in names and "sqrt" in names and "np" in names
    assert "__builtins__" not in names

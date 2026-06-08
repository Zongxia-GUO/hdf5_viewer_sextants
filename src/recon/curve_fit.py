"""Pure curve-fitting backend shared by the XRMS Analyze tool.

GUI-free model library + polynomial background subtraction + initial-parameter
auto-guessing + a thin ``run_fit`` wrapper around ``scipy.optimize.curve_fit``.
The XRMS Analyze tool (radial / angular / time profiles) and the frame-by-frame
analysis worker all fit through these.
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

from __future__ import annotations

import numpy as np


# ── Model functions (x is the generic abscissa: r, theta or frame index) ──── #

def gaussian(x, A, x0, sigma, C):
    """Single Gaussian peak."""
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + C


def double_gaussian(x, A1, x1, sigma1, A2, x2, sigma2, C):
    """Sum of two Gaussian peaks."""
    return (A1 * np.exp(-(x - x1) ** 2 / (2 * sigma1 ** 2))
            + A2 * np.exp(-(x - x2) ** 2 / (2 * sigma2 ** 2))
            + C)


def exponential(x, A, tau, C):
    """Single exponential decay."""
    return A * np.exp(-x / tau) + C


def stretched_exp(x, A, tau, beta, C):
    """Stretched (Kohlrausch) exponential."""
    return A * np.exp(-(x / tau) ** beta) + C


def double_exp(x, A1, tau1, A2, tau2, C):
    """Sum of two exponential decays."""
    return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + C


def poly2(x, a, b, c):
    """Quadratic polynomial."""
    return a * x ** 2 + b * x + c


def poly3(x, a, b, c, d):
    """Cubic polynomial."""
    return a * x ** 3 + b * x ** 2 + c * x + d


def linear(x, a, b):
    """Straight line."""
    return a * x + b


def power_law(x, A, n, C):
    """Power law."""
    return A * x ** n + C


def lorentzian(x, A, x0, gamma, C):
    """Single Lorentzian peak."""
    return A * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2) + C


def pseudo_voigt(x, A, x0, w, eta, C):
    """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian sharing width w."""
    g = np.exp(-(x - x0) ** 2 / (2 * w ** 2))
    lo = w ** 2 / ((x - x0) ** 2 + w ** 2)
    return A * (eta * lo + (1 - eta) * g) + C


# ── Model registry ─────────────────────────────────────────────────────────── #
# Param labels use a neutral "x0" so the same registry serves r / theta / time.

FIT_MODELS: dict = {
    "Gaussian  A·exp(-(x-x₀)²/2σ²) + C": {
        "func": gaussian,
        "params": [("A", 1.0), ("x₀", 0.0), ("σ", 10.0), ("C", 0.0)],
    },
    "Double Gaussian  A₁·G(x₁,σ₁) + A₂·G(x₂,σ₂) + C": {
        "func": double_gaussian,
        "params": [("A₁", 1.0), ("x₁", 0.0), ("σ₁", 10.0),
                   ("A₂", 0.5), ("x₂", 20.0), ("σ₂", 10.0), ("C", 0.0)],
    },
    "Exponential  A·exp(-x/τ) + C": {
        "func": exponential,
        "params": [("A", 1.0), ("τ", 50.0), ("C", 0.0)],
    },
    "Stretched Exp  A·exp(-(x/τ)^β) + C": {
        "func": stretched_exp,
        "params": [("A", 1.0), ("τ", 50.0), ("β", 1.0), ("C", 0.0)],
    },
    "Double Exp  A₁·exp(-x/τ₁) + A₂·exp(-x/τ₂) + C": {
        "func": double_exp,
        "params": [("A₁", 1.0), ("τ₁", 20.0), ("A₂", 0.5), ("τ₂", 80.0), ("C", 0.0)],
    },
    "Polynomial 2°  a·x² + b·x + c": {
        "func": poly2,
        "params": [("a", 0.0), ("b", 0.0), ("c", 1.0)],
    },
    "Polynomial 3°  a·x³ + b·x² + c·x + d": {
        "func": poly3,
        "params": [("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 1.0)],
    },
    "Linear  a·x + b": {
        "func": linear,
        "params": [("a", 0.0), ("b", 1.0)],
    },
    "Power Law  A·x^n + C": {
        "func": power_law,
        "params": [("A", 1.0), ("n", 1.0), ("C", 0.0)],
    },
}


# ── Peak-only models (for the Peak Fit step on radial profiles) ─────────────── #

PEAK_MODELS: dict = {
    "Gaussian  A·exp(-(x-x₀)²/2σ²) + C": {
        "func": gaussian,
        "params": [("A", 1.0), ("x₀", 0.0), ("σ", 10.0), ("C", 0.0)],
    },
    "Lorentzian  A·γ²/((x-x₀)²+γ²) + C": {
        "func": lorentzian,
        "params": [("A", 1.0), ("x₀", 0.0), ("γ", 10.0), ("C", 0.0)],
    },
    "Pseudo-Voigt  η·L + (1-η)·G": {
        "func": pseudo_voigt,
        "params": [("A", 1.0), ("x₀", 0.0), ("w", 10.0), ("η", 0.5), ("C", 0.0)],
    },
    "Double Gaussian  A₁·G(x₁,σ₁) + A₂·G(x₂,σ₂) + C": {
        "func": double_gaussian,
        "params": [("A₁", 1.0), ("x₁", 0.0), ("σ₁", 10.0),
                   ("A₂", 0.5), ("x₂", 20.0), ("σ₂", 10.0), ("C", 0.0)],
    },
}


# ── Sin-power models (for angular / azimuthal profiles; x is theta in degrees) ─ #

def _sin_power(n: int):
    """Build A·sin(theta + phi)**n + C with theta in degrees."""
    def _f(x, A, phi, C):
        return A * np.sin(np.deg2rad(x) + phi) ** n + C
    return _f


SIN_MODELS: dict = {
    "sin¹  A·sin(θ+φ) + C": {"func": _sin_power(1), "params": [("A", 1.0), ("φ", 0.0), ("C", 0.0)]},
    "sin²  A·sin²(θ+φ) + C": {"func": _sin_power(2), "params": [("A", 1.0), ("φ", 0.0), ("C", 0.0)]},
    "sin³  A·sin³(θ+φ) + C": {"func": _sin_power(3), "params": [("A", 1.0), ("φ", 0.0), ("C", 0.0)]},
    "sin⁴  A·sin⁴(θ+φ) + C": {"func": _sin_power(4), "params": [("A", 1.0), ("φ", 0.0), ("C", 0.0)]},
}


def compute_poly_background(
    x: np.ndarray,
    y: np.ndarray,
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    degree: int = 1,
) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """Fit a polynomial background on two baseline regions and subtract it.

    Regions [x1, x2] and [x3, x4] define the baseline. A polynomial of the given
    degree is fitted there (np.polyfit) and subtracted from the whole profile.

    Returns (y_corrected, background_curve, coeffs) where coeffs is highest-degree
    first, or None when there are too few baseline points to fit.
    """
    mask = ((x >= x1) & (x <= x2)) | ((x >= x3) & (x <= x4))
    mask &= np.isfinite(y)
    if mask.sum() < degree + 1:
        return y.copy(), np.zeros_like(y), None
    try:
        coeffs = np.polyfit(x[mask], y[mask], degree)
    except Exception:
        return y.copy(), np.zeros_like(y), None
    background = np.polyval(coeffs, x)
    return y - background, background, tuple(float(c) for c in coeffs)


def auto_guess(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    lo: float | None = None,
    hi: float | None = None,
    models: dict | None = None,
) -> list[float] | None:
    """Estimate initial parameters for ``model_name`` from the data.

    ``models`` selects the registry (defaults to FIT_MODELS; pass SIN_MODELS for
    angular fits). When lo/hi are given the estimate uses only points in range.
    Returns a list aligned with the model's params, or None.
    """
    models = models or FIT_MODELS
    if x is None or y is None or len(x) == 0:
        return None
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()

    if lo is not None and hi is not None:
        mask = (x >= lo) & (x <= hi) & np.isfinite(y)
        if mask.sum() > 2:
            x, y = x[mask], y[mask]
    else:
        mask = np.isfinite(y)
        if mask.any():
            x, y = x[mask], y[mask]

    if len(y) == 0:
        return None

    amp = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    offset = float(np.nanmin(y))
    span = float(x[-1] - x[0]) if len(x) > 1 else 50.0
    tau = max(span / 3.0, 1.0)
    x0 = float(x[int(np.nanargmax(y))])
    sigma = max(span / 6.0, 1.0)
    slope = float((y[-1] - y[0]) / span) if span > 0 else 0.0
    ymid = float(np.nanmean(y))

    lookup: dict = {
        "A": amp, "A₁": amp * 0.6, "A₂": amp * 0.4,
        "τ": tau, "τ₁": tau * 0.4, "τ₂": tau * 1.5,
        "β": 1.0, "φ": 0.0, "η": 0.5,
        "x₀": x0, "x₁": x0, "x₂": x0 + sigma, "r₀": x0, "t₀": x0, "θ₀": x0,
        "σ": sigma, "σ₁": sigma, "σ₂": sigma, "γ": sigma, "w": sigma,
        "C": offset,
        "a": slope, "b": ymid, "c": offset, "d": offset,
        "n": 1.0,
    }
    return [lookup.get(pname, 1.0) for pname, _ in models[model_name]["params"]]


def run_fit(
    func,
    x: np.ndarray,
    y: np.ndarray,
    p0: list[float],
    lo: float,
    hi: float,
    maxfev: int = 50000,
) -> dict:
    """Fit ``func`` to (x, y) restricted to [lo, hi], dropping non-finite points.

    Returns a dict. On success: ok=True with popt, perr, r2, rmse, x_fit, y_fit,
    y_pred, residuals. On failure: ok=False with an ``error`` string.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 3:
        return {"ok": False, "error": "Too few valid points in the range (need ≥ 3)."}

    try:
        popt, pcov = curve_fit(func, x_fit, y_fit, p0=list(p0), maxfev=maxfev)
        perr = np.sqrt(np.diag(np.abs(pcov)))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    y_pred = func(x_fit, *popt)
    residuals = y_fit - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "ok": True,
        "popt": popt, "perr": perr, "r2": r2, "rmse": rmse,
        "x_fit": x_fit, "y_fit": y_fit, "y_pred": y_pred, "residuals": residuals,
    }

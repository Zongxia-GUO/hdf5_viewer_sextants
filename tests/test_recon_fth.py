"""Unit tests for the GUI-free FTH reconstruction kernels (src.recon.fth)."""

import numpy as np
import pytest

from src.recon.fth import (
    binary_filter,
    bs_step,
    differential_filter_kernel,
    estimate_balance_ratio,
    fth_transform,
    line_gaussian_filter,
)


def _grids(n=16):
    rows = np.arange(n, dtype=float)
    cols = np.arange(n, dtype=float)
    xmat, ymat = np.meshgrid(rows, cols, indexing="ij")
    return xmat, ymat, n // 2, n // 2


# ---------------------------------------------------------------------------
# bs_step
# ---------------------------------------------------------------------------

def test_bs_step_range_and_midpoint():
    x = np.linspace(-50, 50, 101)
    out = bs_step(3.0, x)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert bs_step(3.0, np.array([0.0]))[0] == pytest.approx(0.5)
    assert out[0] < 0.01   # deep inside
    assert out[-1] > 0.99  # far outside


def test_bs_step_monotonic():
    x = np.linspace(-20, 20, 200)
    out = bs_step(2.0, x)
    assert np.all(np.diff(out) >= 0)


# ---------------------------------------------------------------------------
# line_gaussian_filter
# ---------------------------------------------------------------------------

def test_line_gaussian_filter_range_and_notch():
    filt = line_gaussian_filter(32, 32, 30.0, sigma=2.0, shift=0.0)
    assert filt.shape == (32, 32)
    assert np.all(filt >= 0.0) and np.all(filt <= 1.0)
    # On the notch line the value dips toward 0; somewhere it should be near 0.
    assert filt.min() < 0.05


# ---------------------------------------------------------------------------
# binary_filter
# ---------------------------------------------------------------------------

def test_binary_filter_is_binary_and_zeros_line():
    xmat, ymat, x0, y0 = _grids(16)
    out = binary_filter(xmat, ymat, x0, y0, 0.0, width=2)
    assert set(np.unique(out)).issubset({0.0, 1.0})
    # phi=0 -> perp distance is |dr*sin(0) - dc*cos(0)| = |dc|; pixels with |col-y0|<=2 zeroed.
    assert out[x0, y0] == 0.0
    assert out[x0, 0] == 1.0  # far column kept


# ---------------------------------------------------------------------------
# differential_filter_kernel
# ---------------------------------------------------------------------------

def test_differential_filter_kernel_is_imaginary_and_zero_at_center():
    xmat, ymat, x0, y0 = _grids(16)
    k = differential_filter_kernel(xmat, ymat, x0, y0, 45.0)
    assert np.iscomplexobj(k)
    assert np.allclose(k.real, 0.0)          # purely imaginary ramp
    assert k[x0, y0] == pytest.approx(0.0)   # zero at the chosen center


# ---------------------------------------------------------------------------
# fth_transform
# ---------------------------------------------------------------------------

def test_fth_transform_matches_manual_formula():
    rng = np.random.default_rng(0)
    xmat, ymat, x0, y0 = _grids(16)
    holo = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    out = fth_transform(holo, xmat, ymat, x0, y0, 16, 16)
    phase_corr = np.exp(2j * np.pi * (xmat * x0 / 16 + ymat * y0 / 16))
    expected = np.fft.fftshift(np.fft.fft2(holo)) * phase_corr
    assert np.allclose(out, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# estimate_balance_ratio
# ---------------------------------------------------------------------------

def test_estimate_balance_ratio_recovers_known_factor():
    rng = np.random.default_rng(1)
    b = np.abs(rng.standard_normal((40, 40))) + 1.0
    a = 2.5 * b
    assert estimate_balance_ratio(a, b) == pytest.approx(2.5, rel=1e-3)


def test_estimate_balance_ratio_no_valid_pixels_returns_one():
    a = np.zeros((8, 8))
    b = np.zeros((8, 8))
    assert estimate_balance_ratio(a, b) == 1.0

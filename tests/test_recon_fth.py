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
    photon_wavelength,
    propagate_hologram,
    propagation_kernel,
    quantize_propagation_distance,
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


# ---------------------------------------------------------------------------
# propagation / focus
# ---------------------------------------------------------------------------


def test_photon_wavelength_for_1239_ev_is_one_nanometre():
    assert photon_wavelength(1239.841984) == pytest.approx(1e-9, rel=2e-7)


def test_propagation_zero_distance_is_an_exact_complex_copy():
    holo = np.arange(24, dtype=float).reshape(4, 6)
    out = propagate_hologram(holo, 0.0, 0.18, 20e-6, 779.5)
    assert np.iscomplexobj(out)
    assert np.array_equal(out, holo)
    assert out is not holo


def test_propagation_changes_phase_not_magnitude_and_is_reversible():
    rng = np.random.default_rng(5)
    holo = rng.normal(size=(24, 40)) + 1j * rng.normal(size=(24, 40))
    forward = propagate_hologram(
        holo, 2.3e-6, 0.18, 20e-6, 779.5, quantize_wavelength=False
    )
    backward = propagate_hologram(
        forward, -2.3e-6, 0.18, 20e-6, 779.5, quantize_wavelength=False
    )
    assert forward.shape == holo.shape
    assert np.allclose(np.abs(forward), np.abs(holo), atol=1e-12)
    assert np.allclose(backward, holo, atol=1e-11)


def test_propagation_distance_quantization_is_symmetric():
    wavelength = photon_wavelength(779.5)
    positive = quantize_propagation_distance(2.3e-6, wavelength)
    negative = quantize_propagation_distance(-2.3e-6, wavelength)
    assert positive / wavelength == pytest.approx(round(2.3e-6 / wavelength))
    assert negative == pytest.approx(-positive)


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(detector_distance_m=0.0, pixel_size_m=20e-6, energy_ev=779.5),
        dict(detector_distance_m=0.18, pixel_size_m=0.0, energy_ev=779.5),
        dict(detector_distance_m=0.18, pixel_size_m=20e-6, energy_ev=0.0),
    ],
)
def test_propagation_rejects_invalid_physical_parameters(kwargs):
    with pytest.raises(ValueError):
        propagation_kernel((16, 16), 1e-6, **kwargs)


def test_propagation_rejects_non_2d_hologram():
    with pytest.raises(ValueError, match='two-dimensional'):
        propagate_hologram(np.zeros((2, 3, 4)), 1e-6, 0.18, 20e-6, 779.5)

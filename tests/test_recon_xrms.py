"""Tests for the XRMS Analyze pure backend (profiles + curve_fit)."""

import numpy as np
import pytest

from src.recon import curve_fit as cf
from src.recon import profiles as pr


# ── Profiles ───────────────────────────────────────────────────────────────── #

def _disk(n=101, cx=50, cy=50):
    yy, xx = np.indices((n, n), dtype=np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return r  # intensity == radius, so I(r) should be ~ r


def test_radial_profile_recovers_radius():
    data = _disk()
    r_bins, intensity = pr.radial_profile(data, cx=50, cy=50, r_min=0, r_max=40)
    assert len(r_bins) > 0
    # Within the ring, mean intensity at radius bin ~ equals the radius.
    finite = np.isfinite(intensity)
    sel = finite & (r_bins <= 40) & (r_bins >= 5)
    assert np.allclose(intensity[sel], r_bins[sel], atol=1.0)


def test_radial_profile_empty_when_ring_off_image():
    data = _disk()
    r_bins, intensity = pr.radial_profile(data, cx=50, cy=50, r_min=500, r_max=600)
    assert r_bins.size == 0 and intensity.size == 0


def test_angular_profile_uniform_ring_is_flat():
    # Constant image -> angular profile is constant where defined.
    data = np.ones((101, 101), dtype=np.float64)
    centers, avg = pr.angular_profile(data, cx=50, cy=50, r_min=5, r_max=40, n_bins=72)
    assert centers.size == 72
    finite = np.isfinite(avg)
    assert finite.any()
    assert np.allclose(avg[finite], 1.0)


# ── Curve fit ──────────────────────────────────────────────────────────────── #

def test_fit_models_registry_consistent():
    for name, spec in cf.FIT_MODELS.items():
        assert callable(spec["func"])
        assert len(spec["params"]) >= 2


def test_run_fit_recovers_gaussian():
    x = np.linspace(-50, 50, 400)
    true = (10.0, 5.0, 8.0, 2.0)
    y = cf.gaussian(x, *true)
    res = cf.run_fit(cf.gaussian, x, y, p0=[8, 0, 5, 0], lo=-50, hi=50)
    assert res["ok"]
    assert res["r2"] > 0.999
    assert np.allclose(res["popt"], true, atol=1e-3)


def test_run_fit_too_few_points():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    res = cf.run_fit(cf.linear, x, y, p0=[1, 0], lo=0, hi=1)
    assert not res["ok"]
    assert "few" in res["error"].lower()


def test_auto_guess_returns_param_count():
    x = np.linspace(0, 100, 200)
    y = cf.gaussian(x, 5, 50, 10, 1)
    for name, spec in cf.FIT_MODELS.items():
        guess = cf.auto_guess(name, x, y, lo=0, hi=100)
        assert guess is not None
        assert len(guess) == len(spec["params"])


def test_compute_poly_background_flat():
    x = np.linspace(0, 100, 101)
    y = 3.0 + 0.0 * x  # flat baseline
    y_corr, bg, coeffs = cf.compute_poly_background(x, y, 0, 20, 80, 100, degree=1)
    assert coeffs is not None
    assert np.allclose(bg, 3.0, atol=1e-6)
    assert np.allclose(y_corr, 0.0, atol=1e-6)


def test_sin_models_recover_power():
    theta = np.linspace(0, 360, 400)
    for name, spec in cf.SIN_MODELS.items():
        true = (3.0, 0.4, 1.5)
        y = spec["func"](theta, *true)
        g = cf.auto_guess(name, theta, y, models=cf.SIN_MODELS)
        assert g is not None and len(g) == 3
        res = cf.run_fit(spec["func"], theta, y, p0=g, lo=0, hi=360)
        assert res["ok"] and res["r2"] > 0.99, (name, res.get("error"), res.get("r2"))


def test_sin2_azimuthal_shape():
    # sin^2 azimuthal anisotropy is non-negative and peaks at 90 deg (phi=0).
    theta = np.linspace(0, 360, 361)
    y = cf.SIN_MODELS["sin²  A·sin²(θ+φ) + C"]["func"](theta, 1.0, 0.0, 0.0)
    assert abs(y[90] - 1.0) < 1e-6 and abs(y[0]) < 1e-6


def test_compute_poly_background_insufficient_points():
    x = np.linspace(0, 10, 11)
    y = np.full_like(x, np.nan)
    y_corr, bg, coeffs = cf.compute_poly_background(x, y, 0, 1, 2, 3, degree=2)
    assert coeffs is None

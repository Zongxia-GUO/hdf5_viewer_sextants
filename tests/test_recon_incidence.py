"""Tests for the incidence-correction resampling core."""

import numpy as np

from src.recon import incidence as inc


def test_incidence_factor():
    assert inc.incidence_factor(90.0) == 1.0      # sin90 = 1
    assert abs(inc.incidence_factor(30.0) - 2.0) < 1e-9   # 1/sin30 = 2
    assert inc.incidence_factor(0.0) == 1.0       # no correction
    assert inc.incidence_factor(-5.0) == 1.0
    # near-zero angle is clamped, never explodes
    assert inc.incidence_factor(0.001) == inc.MAX_FACTOR


def test_resample_no_correction_returns_input():
    stack = np.random.rand(3, 8, 8).astype("float32")
    out, (sx, sy) = inc.resample_incidence(stack, 0.0, 0.0)
    assert (sx, sy) == (1.0, 1.0)
    assert out.shape == stack.shape
    assert np.allclose(out, stack)


def test_resample_stretches_x_only():
    stack = np.ones((2, 10, 10), dtype="float32")
    out, (sx, sy) = inc.resample_incidence(stack, 30.0, 90.0)  # sx=2, sy=1
    assert sy == 1.0 and abs(sx - 2.0) < 1e-9
    assert out.shape == (2, 10, 20)  # cols doubled, rows unchanged, n unchanged


def test_resample_2d_frame():
    frame = np.ones((10, 10), dtype="float32")
    out, (sx, sy) = inc.resample_incidence(frame, 90.0, 30.0)  # sx=1, sy=2
    assert out.ndim == 2 and out.shape == (20, 10)


def test_resample_preserves_intensity_scale():
    # A flat frame stays flat after stretching.
    stack = np.full((1, 12, 12), 5.0, dtype="float32")
    out, _ = inc.resample_incidence(stack, 30.0, 45.0)
    assert np.allclose(out, 5.0, atol=1e-4)

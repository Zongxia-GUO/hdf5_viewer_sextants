"""Two things the FTH tool got wrong on a 2048x2048 detector.

A bad pixel used to destroy the whole reconstruction while the tool reported
success, and tuning the focus rebuilt a 125 ms array that never changed.
"""

import time

import numpy as np
import pytest

from src.gui.fth_reconstruction_tool import FTHReconstructionTool, _clean_non_finite
from src.recon.fth import fth_phase_correction, fth_transform


def _pair(n=128, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2) + 1.0
    mag = 40 * np.exp(-((xx - 0.6 * n) ** 2 + (yy - 0.35 * n) ** 2) / (2 * 5.0 ** 2))
    charge = 20000.0 / r ** 2 + rng.rand(n, n)
    return charge + mag, charge - mag


@pytest.fixture
def tool(qapp):
    t = FTHReconstructionTool(opened_files=(), dataset_full_keys_2d=[])
    yield t
    t.deleteLater()


def _reconstruct(tool, cl, cr):
    tool._on_load_finished(cl, cr, None)
    assert tool._apply_filters_only() is True
    assert tool._compute_fth_only() is True
    return np.asarray(tool._FTH_S1)


# ── One bad pixel used to take the whole picture with it ──────────────── #

def test_a_single_nan_no_longer_destroys_the_reconstruction(tool):
    """An FFT spreads every input across every output, so one NaN is not one
    bad pixel in the result — measured before the fix, a single NaN among
    65536 left 0 of 65536 output pixels finite."""
    cl, cr = _pair()
    cl[7, 9] = np.nan

    out = _reconstruct(tool, cl, cr)

    assert np.isfinite(out).all(), f"{(~np.isfinite(out)).sum()} of {out.size} lost"


def test_a_single_inf_no_longer_destroys_the_reconstruction(tool):
    cl, cr = _pair()
    cl[7, 9] = np.inf

    out = _reconstruct(tool, cl, cr)

    assert np.isfinite(out).all()


def test_the_repair_is_reported_rather_than_silent(tool):
    """Quietly repairing someone's data is how a detector fault goes unnoticed
    for a whole beamtime."""
    cl, cr = _pair()
    cl[1, 2] = np.nan
    cl[3, 4] = np.inf
    cr[5, 6] = -np.inf

    tool._on_load_finished(cl, cr, None)

    assert "3 non-finite pixel(s) set to 0" in tool._status_label.text()


def test_clean_data_is_not_touched_and_says_nothing(tool):
    cl, cr = _pair()

    tool._on_load_finished(cl.copy(), cr.copy(), None)

    assert "non-finite" not in tool._status_label.text()
    np.testing.assert_array_equal(tool._CL, cl)


def test_the_dark_frame_is_cleaned_too(tool):
    cl, cr = _pair()
    dark = np.zeros_like(cl)
    dark[2, 2] = np.nan

    tool._on_load_finished(cl, cr, dark)

    assert np.isfinite(np.asarray(tool._dark)).all()
    assert "1 non-finite" in tool._status_label.text()


def test_the_cleaner_counts_every_array_and_keeps_the_rest(qapp):
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    b = np.array([[np.inf, 2.0], [-np.inf, 4.0]])

    cl, cr, dark, bad = _clean_non_finite(a, b, None)

    assert bad == 3
    assert dark is None
    np.testing.assert_array_equal(cl, [[1.0, 0.0], [3.0, 4.0]])
    np.testing.assert_array_equal(cr, [[0.0, 2.0], [0.0, 4.0]])


# ── The centering ramp is built once, not per focus step ──────────────── #

def test_the_cached_ramp_is_the_one_the_maths_wants():
    n = 64
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    holo = np.random.RandomState(0).rand(n, n)

    cached = fth_transform(holo, x, y, 11.0, 13.0, n, n,
                           fth_phase_correction(x, y, 11.0, 13.0, n, n))
    computed = fth_transform(holo, x, y, 11.0, 13.0, n, n)

    np.testing.assert_array_equal(cached, computed)


def test_the_transform_still_matches_the_definition():
    """scipy's FFT replaced numpy's for speed; it has to be the same maths."""
    n = 64
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    holo = np.random.RandomState(0).rand(n, n) * 1e6

    got = fth_transform(holo, x, y, 11.0, 13.0, n, n)
    want = np.fft.fftshift(np.fft.fft2(holo)) * np.exp(
        2j * np.pi * (x * 11.0 / n + y * 13.0 / n))

    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_the_tool_builds_the_ramp_with_the_geometry(tool):
    """It has to be rebuilt when the centre moves, or it silently belongs to
    the previous crop."""
    cl, cr = _pair()
    tool._on_load_finished(cl, cr, None)
    first = tool._fth_phase
    assert first is not None
    assert first.shape == (tool._Nx, tool._Ny)

    tool._t1_xmid.setValue(int(tool._t1_xmid.value()) - 6)
    tool._compute_centered_hologram()

    assert tool._fth_phase.shape == (tool._Nx, tool._Ny)
    assert tool._fth_phase is not first


def test_the_ramp_is_not_rebuilt_while_focusing(tool, monkeypatch):
    """The whole point: 125 ms at 2048x2048, on every step of the slider."""
    cl, cr = _pair()
    tool._on_load_finished(cl, cr, None)
    tool._apply_filters_only()

    # Watch the function itself, not the tool's alias for it: fth_transform
    # falls back to building the ramp internally when it is handed None, and
    # that call goes through src.recon.fth, not through the alias.
    import src.recon.fth as recon_fth
    builds = []
    real = recon_fth.fth_phase_correction
    monkeypatch.setattr(recon_fth, "fth_phase_correction",
                        lambda *a: (builds.append(True), real(*a))[1])

    for _ in range(5):
        tool._compute_fth_only()

    assert builds == [], f"the ramp was rebuilt {len(builds)} times"


def test_the_cached_ramp_is_dropped_when_the_tool_closes(tool):
    cl, cr = _pair()
    tool._on_load_finished(cl, cr, None)
    assert tool._fth_phase is not None

    tool.reset_results()

    assert tool._fth_phase is None


def test_the_transform_is_faster_than_the_version_it_replaced():
    """Guards the reason for the change, not a wall-clock number: the cached
    ramp plus scipy measured 3.3x on a 2048x2048 detector."""
    n = 512
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    holo = np.random.RandomState(0).rand(n, n) * 1e6
    phase = fth_phase_correction(x, y, 11.0, 13.0, n, n)

    def old():
        ramp = np.exp(2j * np.pi * (x * 11.0 / n + y * 13.0 / n))
        return np.fft.fftshift(np.fft.fft2(holo)) * ramp

    def new():
        return fth_transform(holo, x, y, 11.0, 13.0, n, n, phase)

    def best(fn):
        fn()
        return min(_timed(fn) for _ in range(3))

    assert best(new) < best(old)


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start

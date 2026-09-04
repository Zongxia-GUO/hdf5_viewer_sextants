"""Combining the frames of a stack, checked against a known truth.

The stack is built from a truth image plus Poisson counting noise, so every
method can be scored on how far it lands from the answer rather than on whether
it matches the implementation. Cosmic rays are then added — isolated very
bright pixels in scattered frames, which is what actually goes wrong on a
detector — and scored again.
"""

import numpy as np
import pytest

from src.recon.stack_combine import (
    DEFAULT_KAPPA,
    combine_clipped_mean,
    combine_mean,
    combine_median,
    combine_sum,
)

N, H, W = 60, 96, 96


def _truth(level=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    radius = np.hypot(yy - H / 2, xx - W / 2) + 1.0
    return level * (400.0 / radius
                    + 60.0 * np.exp(-((xx - 60) ** 2 + (yy - 30) ** 2) / (2 * 5.0 ** 2)))


def _stack(truth, seed=0):
    return np.random.RandomState(seed).poisson(
        np.repeat(truth[None], N, axis=0)).astype(np.float64)


def _with_rays(stack, count=200, seed=1):
    rng = np.random.RandomState(seed)
    dirty = stack.copy()
    fi, yi, xi = (rng.randint(0, k, count) for k in (N, H, W))
    dirty[fi, yi, xi] += rng.uniform(3000, 20000, count)
    return dirty


def _rms(got, truth):
    return float(np.sqrt(((got - truth) ** 2).mean()))


# ── The plain ones ────────────────────────────────────────────────────── #

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_method_reduces_the_axis_it_is_given(axis):
    stack = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)

    assert combine_mean(stack, axis).shape == stack.mean(axis=axis).shape
    np.testing.assert_array_equal(combine_mean(stack, axis), stack.mean(axis=axis))
    np.testing.assert_array_equal(combine_sum(stack, axis), stack.sum(axis=axis))
    np.testing.assert_array_equal(combine_median(stack, axis), np.median(stack, axis=axis))
    assert combine_clipped_mean(stack, axis).shape == stack.mean(axis=axis).shape


def test_the_sum_is_the_mean_times_the_frame_count():
    stack = _stack(_truth())

    np.testing.assert_allclose(combine_sum(stack, 0), combine_mean(stack, 0) * N)


# ── What each one is for ──────────────────────────────────────────────── #

def test_a_mean_is_the_best_answer_until_something_goes_wrong():
    truth = _truth()
    clean = _stack(truth)

    assert _rms(combine_mean(clean, 0), truth) < _rms(combine_median(clean, 0), truth)


def test_a_single_cosmic_ray_ruins_a_mean():
    """This is the whole reason the other methods are offered."""
    truth = _truth()
    dirty = _with_rays(_stack(truth))

    assert _rms(combine_mean(dirty, 0), truth) > 10 * _rms(combine_mean(_stack(truth), 0), truth)


@pytest.mark.parametrize("method", [combine_median, combine_clipped_mean])
def test_the_robust_methods_do_not_notice_the_cosmic_rays(method):
    truth = _truth()
    clean = _stack(truth)
    dirty = _with_rays(clean)

    assert _rms(method(dirty, 0), truth) == pytest.approx(_rms(method(clean, 0), truth), rel=0.05)


def test_the_clipped_mean_beats_the_median_on_clean_data():
    """A median throws away statistics to buy protection it may not need; the
    clipped mean only pays for the samples it actually rejects."""
    truth = _truth()
    clean = _stack(truth)

    assert (_rms(combine_clipped_mean(clean, 0), truth)
            < _rms(combine_median(clean, 0), truth))


def test_the_clipped_mean_costs_almost_nothing_against_a_plain_mean():
    truth = _truth()
    clean = _stack(truth)

    plain = _rms(combine_mean(clean, 0), truth)
    clipped = _rms(combine_clipped_mean(clean, 0), truth)

    assert clipped < plain * 1.02, (clipped, plain)


# ── The case a per-pixel scale alone gets wrong ───────────────────────── #

def test_low_count_data_is_still_protected():
    """Where the per-pixel MAD collapses.

    In a faint background most frames of a pixel read the same small integer,
    so the median absolute deviation of that pixel is exactly zero — and a
    threshold of zero times anything rejects nothing, letting the cosmic ray
    through into the average. Measured with a per-pixel scale alone: rms 3.7053
    against 0.0896 once the scale is floored by the spread of the whole stack.
    """
    truth = np.full((H, W), 0.8)
    dirty = _with_rays(_stack(truth))

    clipped = _rms(combine_clipped_mean(dirty, 0), truth)
    plain = _rms(combine_mean(dirty, 0), truth)

    assert plain > 1.0, "the mean should be wrecked here"
    assert clipped < 0.2, f"the clipped mean let the rays through: rms {clipped}"


def test_the_per_pixel_scale_still_governs_where_it_is_meaningful():
    """The floor must not become the whole rule.

    Poisson noise grows with the signal, and this truth spans 5.8 to 400
    counts, so one threshold for the whole pattern is far too tight on the
    bright pixels and rejects real counts there. Measured on this stack: a
    per-pixel scale floored by the global one rejects 0.005% of the samples
    and lands within 1.0001 of a plain mean, where a single global threshold
    rejects 0.382% — 76 times more — and lands at 1.0575.
    """
    truth = _truth()
    clean = _stack(truth)

    cost = _rms(combine_clipped_mean(clean, 0), truth) / _rms(combine_mean(clean, 0), truth)

    assert cost < 1.02, f"rejecting too much of the bright signal: {cost:.4f}"


# ── Degenerate stacks ─────────────────────────────────────────────────── #

def test_frames_that_are_all_identical_survive_intact():
    """No spread means nothing can be an outlier, and no division by zero."""
    stack = np.full((5, 4, 4), 7.0)

    np.testing.assert_allclose(combine_clipped_mean(stack, 0), 7.0)


def test_a_lone_spike_in_an_otherwise_flat_stack_is_rejected():
    stack = np.zeros((5, 32, 32))
    stack[0, 0, 0] = 1e9

    assert combine_clipped_mean(stack, 0)[0, 0] == pytest.approx(0.0)


def test_nothing_comes_back_as_nan():
    stack = np.zeros((4, 8, 8))
    stack[1] = 1e6

    assert np.all(np.isfinite(combine_clipped_mean(stack, 0)))


# ── The threshold ─────────────────────────────────────────────────────── #

def test_the_default_threshold_is_five():
    assert DEFAULT_KAPPA == 5.0


def test_a_tighter_threshold_rejects_more_and_costs_accuracy():
    """Measured on the reference stack: k=3 throws away 1.576% of the samples
    where only 0.006% are real hits, and the result is worse for it."""
    truth = _truth()
    clean = _stack(truth)

    tight = _rms(combine_clipped_mean(clean, 0, kappa=2.0), truth)
    loose = _rms(combine_clipped_mean(clean, 0, kappa=DEFAULT_KAPPA), truth)

    assert tight > loose


def test_a_very_loose_threshold_becomes_a_plain_mean():
    truth = _truth()
    clean = _stack(truth)

    np.testing.assert_allclose(
        combine_clipped_mean(clean, 0, kappa=1e6), combine_mean(clean, 0))

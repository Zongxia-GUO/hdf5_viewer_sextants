"""The despike filter has to remove glitches without eating real features.

Both halves of that matter equally here: a filter that misses spikes is useless,
but a filter that quietly flattens a hysteresis switch or an absorption edge is
worse than useless, because the data still looks plausible afterwards.
"""

from __future__ import annotations

import numpy as np

from src.recon.despike import (
    DIRECTION_BOTH,
    DIRECTION_DOWN,
    DIRECTION_UP,
    METHOD_DIFFERENCE,
    METHOD_HAMPEL,
    SPACE_LINEAR,
    SPACE_LOG,
    despike,
    robust_scale,
    use_log_space,
)


def _noise(size: int, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(size) * sigma


def _flagged(result) -> list[int]:
    return np.flatnonzero(result.spikes).tolist()


# ---------------------------------------------------------------------------
# Finding spikes
# ---------------------------------------------------------------------------

def test_isolated_spikes_are_found():
    y = 20 * np.sin(np.linspace(0, 6, 500)) + _noise(500)
    y[[100, 250, 400]] += 60

    assert _flagged(despike(y)) == [100, 250, 400]


def test_a_downward_dropout_is_a_spike_too():
    """A beam dropout reads low; it is the same defect upside down."""
    y = np.full(300, 50.0) + _noise(300)
    y[123] = 0.0

    assert _flagged(despike(y)) == [123]


def test_clean_noise_is_left_alone():
    """The threshold is in sigma, so ordinary scatter must stay under it."""
    y = 10 + _noise(2000, sigma=0.1, seed=3)

    assert despike(y).count == 0


def test_a_spike_is_replaced_by_the_local_level():
    """Replacing with the local median keeps integrals and transforms usable,
    which NaN would not."""
    y = np.linspace(0, 10, 401)
    expected = y.copy()
    y[200] = 999.0

    result = despike(y)

    assert abs(result.values[200] - expected[200]) < 0.05
    np.testing.assert_allclose(np.delete(result.values, 200), np.delete(expected, 200))


# ---------------------------------------------------------------------------
# Not eating real features — the reason the default is the difference method
# ---------------------------------------------------------------------------

def test_a_real_step_edge_survives():
    """A hysteresis switch is a large jump that stays jumped. The step out of a
    spike is what tells the two apart, and a step edge has no step out."""
    y = np.concatenate([np.zeros(250), np.full(250, 40.0)]) + _noise(500, 0.05)

    assert despike(y).count == 0


def test_spikes_are_found_on_a_scan_that_also_has_a_real_edge():
    y = np.concatenate([np.zeros(250), np.full(250, 40.0)]) + _noise(500, 0.05)
    y[100] += 30
    y[400] -= 30

    result = despike(y)

    assert _flagged(result) == [100, 400]
    assert abs(result.values[300] - 40.0) < 0.3, "the new level is intact"
    assert abs(result.values[200] - 0.0) < 0.3, "the old level is intact"


def test_a_narrow_peak_several_samples_wide_is_not_a_spike():
    """Real peaks are resolved by the scan; a glitch is one sample."""
    y = np.full(400, 5.0) + _noise(400, 0.02)
    peak = np.exp(-0.5 * ((np.arange(400) - 200) / 3.0) ** 2) * 40
    y = y + peak

    assert despike(y).count == 0


def test_a_glitch_riding_on_a_peak_is_still_found():
    """Protecting peaks must not amount to a blind spot on top of them."""
    y = np.full(400, 5.0) + _noise(400, 0.02)
    y = y + np.exp(-0.5 * ((np.arange(400) - 200) / 3.0) ** 2) * 40
    y[196] += 60

    assert _flagged(despike(y)) == [196]


def test_noise_growing_along_the_scan_does_not_become_spikes():
    """Counting statistics make the scatter at the bright end many times the
    scatter at the dark end; judging both against one number flagged the bright
    end wholesale."""
    n = 2000
    y = np.linspace(0, 100, n) + np.random.RandomState(5).randn(n) * np.linspace(0.1, 1.0, n)

    assert despike(y).count == 0


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def test_direction_can_be_limited_to_one_sign():
    """Detector glitches are often one-sided, and saying so halves the risk of
    catching a real feature."""
    y = np.full(300, 5.0) + _noise(300, 0.05)
    y[50] += 20
    y[200] -= 20

    assert _flagged(despike(y, direction=DIRECTION_BOTH)) == [50, 200]
    assert _flagged(despike(y, direction=DIRECTION_UP)) == [50]
    assert _flagged(despike(y, direction=DIRECTION_DOWN)) == [200]


# ---------------------------------------------------------------------------
# Reporting, and the mark-only mode
# ---------------------------------------------------------------------------

def test_marking_reports_the_spikes_without_changing_the_values():
    y = np.full(200, 3.0) + _noise(200, 0.02)
    y[77] = 90.0

    result = despike(y, replace=False)

    assert result.count == 1
    assert result.values[77] == 90.0


def test_the_summary_says_what_was_done():
    """It goes in the status line and in the exported file's header, so a
    despiked export is never mistaken for raw data."""
    y = np.full(200, 3.0) + _noise(200, 0.02)
    y[77] = 90.0

    line = despike(y).summary(METHOD_DIFFERENCE, 7, 3.5)

    assert "1 point" in line
    assert "3.5" in line


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

def test_too_short_to_judge():
    for size in (0, 1, 2):
        assert despike(np.ones(size)).count == 0


def test_a_constant_series_has_no_outliers():
    """With no spread there is no scale, and everything would be infinitely far
    from everything else."""
    assert despike(np.full(100, 7.0)).count == 0


def test_a_perfectly_smooth_curve_still_gives_up_its_spike():
    """A median filter reproduces a smooth curve exactly, so the residual MAD
    collapses to zero — the case that used to make the Hampel test blind."""
    y = 20 * np.sin(np.linspace(0, 8, 500))
    y[321] += 50

    assert 321 in _flagged(despike(y, method=METHOD_HAMPEL))


def test_nan_is_missing_data_not_a_spike():
    y = np.full(100, 4.0) + _noise(100, 0.02)
    y[40] = np.nan

    result = despike(y)

    assert not result.spikes[40]


def test_robust_scale_of_pure_noise_recovers_its_sigma():
    """The threshold is quoted in sigma, so this constant is what makes the
    number on the dialog mean anything."""
    noise = np.random.RandomState(7).randn(20000) * 2.5

    assert abs(robust_scale(noise) - 2.5) < 0.1


# ---------------------------------------------------------------------------
# Data spanning decades — reflectivity, where an absolute scale cannot work
# ---------------------------------------------------------------------------

def _reflectivity(n: int = 1200, seed: int = 1) -> np.ndarray:
    """A reflectivity curve: six decades, Kiessig fringes, Poisson counts."""
    rs = np.random.RandomState(seed)
    theta = np.linspace(0.05, 5.0, n)
    theta_c = 0.22
    fresnel = np.where(theta <= theta_c, 1.0, (theta_c / theta) ** 4)
    fringes = 1 + 0.55 * np.cos(2 * np.pi * theta / 0.09)
    i0 = 2e7
    return rs.poisson(np.clip(i0 * fresnel * fringes + 0.4, 0, None)).astype(float) / i0


def test_a_glitch_in_the_dim_tail_is_found_as_readily_as_one_on_the_plateau():
    """The whole point of the log scale. Against an absolute noise estimate the
    tail glitch measured 2.25 sigma — invisible — while the identical glitch on
    the plateau measured 25000, because one number cannot describe both ends."""
    y = _reflectivity()
    y[120] *= 6.0        # bright plateau, R ~ 4e-2
    y[1000] *= 6.0       # far tail,       R ~ 4e-6

    flagged = _flagged(despike(y))

    assert flagged == [120, 1000]


def test_kiessig_fringes_are_not_glitches():
    """A reflectivity curve is nothing but sharp features; eating them would
    be worse than leaving every spike in place."""
    assert despike(_reflectivity()).count == 0


def test_the_tail_glitch_is_replaced_into_its_neighbourhood():
    """Detected but replaced with a plateau-sized number would be no better."""
    y = _reflectivity()
    neighbourhood = np.median(y[995:1006])
    y[1000] *= 6.0

    result = despike(y)

    assert result.spikes[1000]
    assert 0.2 * neighbourhood < result.values[1000] < 5 * neighbourhood


def test_auto_picks_the_log_scale_for_a_wide_dynamic_range():
    assert use_log_space(_reflectivity()) is True


def test_auto_stays_linear_for_ordinary_single_scale_data():
    """Most scans are a field sweep or an energy scan around one level, and a
    log scale there would only make the threshold harder to reason about."""
    assert use_log_space(np.full(500, 20.0) + _noise(500, 0.5)) is False


def test_auto_stays_linear_for_data_that_crosses_zero():
    """A difference signal has no logarithm."""
    assert use_log_space(np.linspace(-5.0, 5.0, 400)) is False


def test_a_few_non_positive_points_do_not_veto_the_log_scale():
    """A background subtraction that overshoots in the last few points is
    exactly the case that needs a log scale most."""
    y = _reflectivity()
    y[-5:] = -1e-8

    assert use_log_space(y) is True


def test_non_positive_points_are_left_alone_rather_than_invented():
    """A dead channel is missing data, not a glitch; replacing it would put a
    number where the measurement failed."""
    y = _reflectivity()
    y[600] = 0.0
    y[601] = -2e-7

    result = despike(y, space=SPACE_LOG)

    assert not result.spikes[600] and not result.spikes[601]
    assert result.values[600] == 0.0
    assert result.values[601] == -2e-7


def test_a_gap_does_not_blind_the_test_around_it():
    """A NaN met by a median filter spreads across a whole window, which used
    to silently switch the filter off for its neighbours."""
    y = _reflectivity()
    y[700] = np.nan
    y[703] *= 6.0

    assert 703 in _flagged(despike(y))


def test_the_scale_can_be_forced_either_way():
    y = _reflectivity()
    y[1000] *= 6.0

    assert despike(y, space=SPACE_LOG).spikes[1000]
    assert not despike(y, space=SPACE_LINEAR).spikes[1000], "this is the old behaviour"


def test_the_result_reports_which_scale_was_used():
    """It goes in the export header: the threshold means a ratio in one case
    and an absolute amount in the other."""
    assert despike(_reflectivity()).log_space is True
    assert despike(np.full(200, 5.0) + _noise(200)).log_space is False


def test_untouched_points_survive_the_log_round_trip_exactly():
    """A logarithm and back is not the identity in floating point, so only the
    spikes may come from that path."""
    y = _reflectivity()
    y[1000] *= 6.0

    result = despike(y)

    kept = ~result.spikes
    np.testing.assert_array_equal(result.values[kept], y[kept])


# ---------------------------------------------------------------------------
# The second method
# ---------------------------------------------------------------------------

def test_hampel_finds_a_spike_at_the_very_first_sample():
    """The difference test needs a step on both sides, so it cannot reach the
    ends; this is what the alternative method is for."""
    y = np.full(200, 6.0) + _noise(200, 0.02)
    y[0] = 60.0

    assert 0 in _flagged(despike(y, method=METHOD_HAMPEL))
    assert 0 not in _flagged(despike(y, method=METHOD_DIFFERENCE))

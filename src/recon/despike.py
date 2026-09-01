"""Finding and removing single-sample glitches in a 1-D scan.

A spike is one bad reading — a beam dropout, a detector glitch, a cosmic ray —
not noise and not a real feature. The whole difficulty is telling it apart from
a genuinely sharp change, because a hysteresis switch or an absorption edge is
also a large jump between neighbouring points.

Two methods, both robust in the same way: the scale of the data is measured
with the **median absolute deviation** rather than the standard deviation. A
standard deviation is inflated by the very spike being looked for, so a large
glitch hides itself; the MAD is not.

``MAD_TO_SIGMA`` makes the MAD a consistent estimator of the standard deviation
for normal data, which is what lets the threshold be read as "so many sigma".
Without it the number would mean nothing in particular.

No Qt here: this is the arithmetic, and it is tested on its own.
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

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

# Scales a MAD to a standard deviation for normal data, so a threshold in
# "sigma" means what it says.
MAD_TO_SIGMA = 1.4826
# The same job for a mean absolute deviation, used only where the MAD collapses
# to zero. Less resistant to outliers, which is why it is the fallback.
MEAN_AD_TO_SIGMA = 1.253314

# How the two methods are named in the interface.
METHOD_DIFFERENCE = "Difference (sharp features safe)"
METHOD_HAMPEL = "Hampel (values)"
METHODS = (METHOD_DIFFERENCE, METHOD_HAMPEL)

DIRECTION_BOTH = "Both"
DIRECTION_UP = "Up only"
DIRECTION_DOWN = "Down only"
DIRECTIONS = (DIRECTION_BOTH, DIRECTION_UP, DIRECTION_DOWN)

# Whether the arithmetic happens on the values or on their logarithm.
SPACE_AUTO = "Auto"
SPACE_LINEAR = "Linear"
SPACE_LOG = "Log"
SPACES = (SPACE_AUTO, SPACE_LINEAR, SPACE_LOG)

# Auto switches to log at this span, in decades. Two is well above anything a
# single-scale measurement produces and well below a reflectivity curve's six.
AUTO_LOG_DECADES = 2.0
# ...provided the data is mostly positive. A reflectivity tail that dips below
# zero in a few background-subtracted points is still a log-scale measurement,
# and it is exactly the case that needs log the most.
AUTO_LOG_POSITIVE_SHARE = 0.9

DEFAULT_WINDOW = 7
DEFAULT_THRESHOLD = 3.5


@dataclass(frozen=True)
class DespikeResult:
    """What a despike pass did: the values to use, and what it changed."""

    values: np.ndarray
    spikes: np.ndarray          # boolean mask, True where a spike was found
    log_space: bool = False     # whether it worked on the logarithm

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.spikes))

    def summary(self, method: str, window: int, threshold: float) -> str:
        """One line for the status bar and for the exported file's header."""
        return (
            f"Despike: {method}, {'log' if self.log_space else 'linear'} scale, "
            f"window={window}, threshold={threshold:g} sigma, "
            f"{self.count} point(s) replaced"
        )


def _odd(window: int) -> int:
    """Windows are centred on the sample, so they have to be odd and >= 3."""
    size = max(3, int(window))
    return size if size % 2 else size + 1


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median in a centred window, truncated at the two ends.

    Truncation, as MATLAB's ``hampel`` does it, rather than scipy's padding.
    Padding repeats the end sample to fill the window, so a spike sitting on
    the first sample gets repeated too and becomes the majority of its own
    window — the median then equals the spike and it hides perfectly.
    """
    size = _odd(window)
    out = median_filter(values, size=size, mode="nearest")

    half = size // 2
    n = values.size
    if n <= 1:
        return out
    for i in range(min(half, n)):
        out[i] = np.median(values[: i + half + 1])
        out[n - 1 - i] = np.median(values[max(0, n - 1 - i - half):])
    return out


def robust_scale(residual: np.ndarray) -> float:
    """Noise level of a residual, as a standard deviation, from the whole series.

    Measured globally on purpose. A MAD taken inside a seven-sample window is
    itself built from seven numbers, so it scatters by tens of percent, and
    wherever it lands low the score explodes — that alone flagged several
    percent of ordinary noise as spikes. Over a whole scan the estimate is
    steady, and the handful of real spikes cannot shift a median.
    """
    finite = np.asarray(residual, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0

    spread = np.abs(finite - np.median(finite))
    scale = float(MAD_TO_SIGMA * np.median(spread))
    if scale > 0:
        return scale

    # A zero MAD means over half the residuals are identical — a quantised
    # counter, or a curve so smooth the median filter reproduces it exactly.
    # The MAD has nothing left to measure, but the few points that do deviate
    # are still real, so fall back to a mean that they can move. Without this
    # the cleanest data in the set was the data where nothing was ever found.
    return float(MEAN_AD_TO_SIGMA * np.mean(spread))


def use_log_space(values: np.ndarray, space: str = SPACE_AUTO) -> bool:
    """Whether to work on the logarithm of the data.

    Over a reflectivity curve's six decades an absolute noise scale is set
    entirely by the bright end, and a glitch out in the tail — a factor of six
    above its neighbours — comes to barely two sigma against it and is never
    seen. Taking the logarithm turns those multiplicative glitches into
    additive ones, so the same threshold means the same *ratio* everywhere.
    """
    if space == SPACE_LOG:
        return True
    if space == SPACE_LINEAR:
        return False

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False

    positive = finite[finite > 0]
    if positive.size < AUTO_LOG_POSITIVE_SHARE * finite.size or positive.size < 2:
        return False
    span = np.log10(positive.max()) - np.log10(positive.min())
    return bool(span >= AUTO_LOG_DECADES)


def _bridge(values: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Fill gaps by drawing a straight line across them.

    Gaps are samples the arithmetic cannot use — a NaN, or a value at or below
    zero when working in log. They cannot simply be left in place: a median
    filter that meets one spreads it across a whole window, blinding the test
    on every neighbour. Bridging keeps the neighbours testable, and the gaps
    themselves are excluded from the result afterwards.
    """
    if not np.any(gaps):
        return values
    good = ~gaps
    if not np.any(good):
        return np.zeros_like(values)
    index = np.arange(values.size, dtype=float)
    return np.interp(index, index[good], values[good])


def _limit_direction(spikes: np.ndarray, deviation: np.ndarray, direction: str) -> np.ndarray:
    """Keep only spikes that jump the way the user asked about.

    A detector glitch is usually one-sided — hot pixels read high, a dropout
    reads low — and saying so halves the chance of eating a real feature.
    """
    if direction == DIRECTION_UP:
        return spikes & (deviation > 0)
    if direction == DIRECTION_DOWN:
        return spikes & (deviation < 0)
    return spikes


def _replace(values: np.ndarray, spikes: np.ndarray, window: int) -> np.ndarray:
    """Put the local median in place of each spike.

    The median is taken from the data with the spikes bridged over, so two
    adjacent glitches cannot repair each other. Bridging rather than filling
    with some global stand-in: on a steeply falling curve a filler taken from
    the far end of the data sits nowhere near the window and shifts its
    ranking, which pulled every replacement one position down the slope.
    """
    if not np.any(spikes):
        return values.copy()

    cleaned = values.astype(float).copy()
    cleaned[spikes] = rolling_median(_bridge(cleaned, spikes), window)[spikes]
    return cleaned


def find_spikes_difference(
    values: np.ndarray,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    direction: str = DIRECTION_BOTH,
) -> np.ndarray:
    """Flag single-sample glitches by the pair of steps around them.

    Works on the first difference, which is what separates a glitch from a
    sharp edge: a real edge is a move to a new level that stays there, while a
    spike is one sample out and straight back, so its large step is immediately
    undone by an equally large one the other way.

    The first and last samples have no step on one side, so this test cannot
    reach them; the Hampel test can.
    """
    y = np.asarray(values, dtype=float).ravel()
    spikes = np.zeros(y.size, dtype=bool)
    if y.size < 3:
        return spikes

    steps = np.diff(y, prepend=y[0])
    steps[0] = 0.0
    scale = robust_scale(steps[1:])
    if scale <= 0:
        # A perfectly regular series: no noise to compare a step against, so
        # nothing can be called an outlier without inventing a scale.
        return spikes

    # How much the steps vary locally, as well as across the whole scan. The
    # top of a well-resolved peak also has a step up followed by a step down,
    # and on quiet data those steps are large next to the noise — the apex of
    # every clean peak was being called a spike. What separates them is that a
    # peak's steps are at their *smallest* at the apex and much larger on the
    # flanks either side, while a glitch's steps tower over their neighbours.
    # Taking the larger of the two scales can only make the test stricter, so
    # it costs nothing where the data is flat.
    # The typical step *size* nearby, not its deviation from a local median:
    # a median filter reproduces a smooth run of steps exactly, so measuring
    # the deviation gives zero along every flank and protects nothing.
    # Scaled the same way as the global estimate so the two are comparable —
    # for pure noise the two agree, which is why flat data behaves as before.
    local = MAD_TO_SIGMA * rolling_median(np.abs(steps), window)
    limit = float(threshold) * np.maximum(scale, local)

    big = np.abs(steps - np.median(steps[1:])) > limit
    # The signature of a one-sample glitch: a large step onto the sample and a
    # large step straight back off it. A real edge has only the step onto the
    # new level, and nothing coming back — which is exactly why it survives.
    spikes[1:-1] = big[1:-1] & big[2:] & (np.sign(steps[1:-1]) != np.sign(steps[2:]))

    return _limit_direction(spikes, y - rolling_median(y, window), direction)


def find_spikes_hampel(
    values: np.ndarray,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    direction: str = DIRECTION_BOTH,
) -> np.ndarray:
    """Flag samples that sit far from the median of their neighbourhood.

    The classic Hampel test, with the scale taken over the whole scan rather
    than inside the window. It reaches the two end samples and catches a spike
    sitting on a slope, but on noisy data it is the more eager of the two: a
    fluctuation that happens to be large is judged on its size alone, with
    nothing asking whether the series came back.
    """
    y = np.asarray(values, dtype=float).ravel()
    if y.size < 3:
        return np.zeros(y.size, dtype=bool)

    deviation = y - rolling_median(y, window)
    scale = robust_scale(deviation)
    if scale <= 0:
        return np.zeros(y.size, dtype=bool)

    spikes = np.abs(deviation) > float(threshold) * scale
    return _limit_direction(spikes, deviation, direction)


def despike(
    values: np.ndarray,
    method: str = METHOD_DIFFERENCE,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    direction: str = DIRECTION_BOTH,
    replace: bool = True,
    space: str = SPACE_AUTO,
) -> DespikeResult:
    """Find spikes and, unless asked only to mark them, replace them.

    ``replace=False`` reports what was found and changes nothing — for deciding
    whether a threshold is sensible before letting it rewrite anything.

    ``space`` decides whether the arithmetic runs on the values or on their
    logarithm; see :func:`use_log_space` for why that matters over a wide
    dynamic range.
    """
    y = np.asarray(values, dtype=float).ravel()
    log_space = use_log_space(y, space)

    # Samples the arithmetic cannot use. In log that includes anything at or
    # below zero — a dead channel, or a background subtraction that overshot.
    # They are bridged over for the detection and then put back untouched: a
    # missing reading is not a glitch, and inventing a value for one would be
    # a worse lie than the glitch was.
    gaps = ~np.isfinite(y)
    if log_space:
        gaps = gaps | (y <= 0)
    work = np.log10(np.where(gaps, 1.0, y)) if log_space else np.where(gaps, 0.0, y)
    work = _bridge(work, gaps)

    finder = find_spikes_hampel if method == METHOD_HAMPEL else find_spikes_difference
    spikes = finder(work, window=window, threshold=threshold, direction=direction)
    spikes &= ~gaps

    if not replace:
        return DespikeResult(values=y.copy(), spikes=spikes, log_space=log_space)

    repaired = _replace(work, spikes, window)
    if log_space:
        repaired = np.power(10.0, repaired)
    # Only the spikes change. Everything else is the number that was loaded,
    # not a value that has been through a logarithm and back.
    cleaned = np.where(spikes, repaired, y)
    return DespikeResult(values=cleaned, spikes=spikes, log_space=log_space)


def despike_table(values: np.ndarray, **options) -> DespikeResult:
    """Despike a single curve or a table of them, one column at a time.

    Columns are separate measurements that happen to share an X axis, so each
    gets its own noise estimate; pooling them would let a loud channel raise
    the threshold on a quiet one.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim <= 1:
        return despike(data, **options)

    cleaned = np.empty_like(data)
    spikes = np.zeros(data.shape, dtype=bool)
    log_space = False
    for column in range(data.shape[1]):
        result = despike(data[:, column], **options)
        cleaned[:, column] = result.values
        spikes[:, column] = result.spikes
        log_space |= result.log_space
    return DespikeResult(values=cleaned, spikes=spikes, log_space=log_space)

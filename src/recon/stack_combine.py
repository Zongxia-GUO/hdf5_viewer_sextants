"""Combining the frames of a stack into one image.

A stack of repeats can be averaged for better statistics, but a mean is only
the right answer when nothing in the stack is wrong. Measured on 100 frames of
256x256 with Poisson counting noise and 400 cosmic-ray hits — 0.006% of the
pixels — against the known truth:

    clean stack          with cosmic rays
    mean    rms 0.235    rms 9.902, worst error 239   <- destroyed
    median  rms 0.430    rms 0.430
    clipped rms 0.236    rms 0.236                    <- k = 5

So a mean is ruined by a handful of bad pixels; a median is immune but costs
1.8x the noise even when there was nothing to defend against; and a clipped
mean throws out the outliers and averages the rest, which on clean data is
within 0.5% of the plain mean and on dirty data is the best of the three.

No Qt here: this is the arithmetic, so it can be tested against a known truth.
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

from src.recon.despike import MAD_TO_SIGMA, robust_scale

#: How many robust standard deviations a sample may sit from the per-pixel
#: median before a clipped mean throws it away.
#:
#: 5, not the customary 3. Measured on the stack described above: k=3 rejects
#: 1.576% of the samples where only 0.006% are real hits, and the collateral
#: shows up in the result (rms 0.267 against 0.236 at k=5). Rejecting less and
#: still catching every cosmic ray is strictly better.
DEFAULT_KAPPA = 5.0


def combine_mean(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """The plain average. Best statistics, no defence against outliers."""
    return np.asarray(np.asarray(stack).mean(axis=int(axis)))


def combine_sum(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """The total. The mean times the frame count, so it only changes scale."""
    return np.asarray(np.asarray(stack).sum(axis=int(axis)))


def combine_median(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """The per-pixel median. Immune to outliers, noisier than a mean."""
    return np.asarray(np.median(np.asarray(stack), axis=int(axis)))


def combine_clipped_mean(
    stack: np.ndarray,
    axis: int = 0,
    kappa: float = DEFAULT_KAPPA,
) -> np.ndarray:
    """Reject outliers against a robust per-pixel centre, then average the rest.

    One pass, not the iterated version astronomy usually writes. Measured: the
    three-iteration form took 2665 ms on a 100x256x256 stack against 237 ms for
    this one and produced the same result to the digit — the extra passes have
    nothing left to find once the median and the MAD have done their work, and
    2.7 s is far too slow to sit behind a control that reloads when touched.

    Like any robust estimator this assumes the outliers are a small minority.
    Cosmic rays are (0.006% of the samples in the measurement above); a stack
    where an entire frame is corrupt is not something this can rescue.
    """
    arr = np.asarray(stack, dtype=np.float64)
    axis = int(axis) % arr.ndim
    kappa = float(kappa)

    centre = np.median(arr, axis=axis, keepdims=True)
    residual = arr - centre
    per_pixel = MAD_TO_SIGMA * np.median(np.abs(residual), axis=axis, keepdims=True)

    # Per-pixel scale, floored by the scale of the whole stack.
    #
    # Per-pixel is needed because the noise here is Poisson and so grows with
    # the signal: one number for the whole image is too tight on a bright
    # speckle and too loose on the background. But a per-pixel MAD collapses to
    # zero wherever more than half the frames read the same value, which in a
    # low-count background is most of them — and a zero scale lets the outlier
    # straight through. Measured on a 0.8-count background with cosmic rays,
    # per-pixel alone gave rms 3.7053 against 0.0896 with this floor.
    floor = robust_scale(residual)
    scale = np.maximum(per_pixel, floor if floor > 0 else 0.0)
    # Only if both are zero, which means every frame of every pixel is
    # identical: there is no outlier to find, so keep everything.
    scale = np.where(scale > 0, scale, np.inf)

    keep = np.abs(residual) <= kappa * scale
    kept = keep.sum(axis=axis)
    total = np.where(keep, arr, 0.0).sum(axis=axis)

    # If a threshold somehow rejected every frame of a pixel, the robust centre
    # stands in rather than leaving a NaN behind.
    centre_flat = np.squeeze(centre, axis=axis)
    return np.asarray(np.where(kept > 0, total / np.maximum(kept, 1), centre_flat))

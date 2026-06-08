"""Incidence-angle geometric correction by resampling.

Grazing-incidence geometry compresses the recorded pattern by sin(theta) along an
axis; the correction stretches it back by 1/sin(theta). Unlike a pure display
transform, this resamples the data so downstream ROI analysis sees the corrected
image directly.
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

# Cap the stretch factor so a near-zero angle can't explode the array size.
MAX_FACTOR = 10.0


def incidence_factor(theta_deg: float) -> float:
    """Return the stretch factor 1/sin(theta), clamped to [1, MAX_FACTOR].

    theta_deg <= 0 (or >= 180) means "no correction" and returns 1.0.
    """
    try:
        theta = float(theta_deg)
    except (TypeError, ValueError):
        return 1.0
    if theta <= 0.0 or theta >= 180.0:
        return 1.0
    s = np.sin(np.deg2rad(theta))
    if s <= 0:
        return 1.0
    return float(min(1.0 / s, MAX_FACTOR))


def resample_incidence(
    stack: np.ndarray,
    theta_x: float,
    theta_y: float,
    order: int = 1,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Stretch every frame of a stack by 1/sin(theta) along col (x) and row (y).

    Parameters
    ----------
    stack : (n, H, W) array (a 2-D frame is also accepted and treated as n=1).
    theta_x, theta_y : incidence angles in degrees for the column / row axes.
    order : spline order for ``scipy.ndimage.zoom`` (1 = bilinear).

    Returns
    -------
    (resampled_stack, (sx, sy)) where sx/sy are the applied factors. When both
    factors are 1.0 the input is returned unchanged (as float32).
    """
    arr = np.asarray(stack, dtype=np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[np.newaxis, ...]

    sx = incidence_factor(theta_x)
    sy = incidence_factor(theta_y)
    if sx == 1.0 and sy == 1.0:
        return (arr[0] if squeeze else arr), (sx, sy)

    from scipy.ndimage import zoom
    out = zoom(arr, (1.0, sy, sx), order=order).astype(np.float32)
    return (out[0] if squeeze else out), (sx, sy)

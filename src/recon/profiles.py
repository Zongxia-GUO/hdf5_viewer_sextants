"""Pure radial / angular profile extraction for the XRMS Analyze tool.

GUI-free azimuthal (I(r)) and radial-ring (I(theta)) profile computation,
shared by the XRMS Analyze tool and the frame-by-frame analysis panels.
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


def polar_grids(h: int, w: int, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, angles_deg) pixel grids for a centre, for caching by the caller."""
    y_idx, x_idx = np.indices((h, w))
    dx = x_idx.astype(np.float64) - cx
    dy = y_idx.astype(np.float64) - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    return r, angles


def radial_profile(
    data: np.ndarray,
    cx: float,
    cy: float,
    r_min: int = 0,
    r_max: int = 0,
    angle_min: int = 0,
    angle_max: int = 180,
    symmetric: bool = True,
    r: np.ndarray | None = None,
    angles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average I(r) within an arc sector (optionally its 180-deg mirror).

    symmetric=True also includes the mirror wedge (angle+180). Radial range is
    r_min..r_max (r_max == 0 means no outer limit). (cx, cy) are pixel coords.
    Precomputed ``r``/``angles`` grids (from ``polar_grids``) skip recomputation.

    Returns (r_bins, mean_intensity) as 1-D float64 arrays.
    """
    if r is None or angles is None:
        r, angles = polar_grids(data.shape[0], data.shape[1], cx, cy)

    a0 = float(angle_min) % 360.0
    a1 = float(angle_max) % 360.0

    def _arc_mask(a_start, a_end):
        if a_start <= a_end:
            return (angles >= a_start) & (angles <= a_end)
        return (angles >= a_start) | (angles <= a_end)

    angle_mask = _arc_mask(a0, a1)
    if symmetric:
        angle_mask = angle_mask | _arc_mask((a0 + 180.0) % 360.0, (a1 + 180.0) % 360.0)

    r_mask = r >= float(r_min)
    if r_max > 0:
        r_mask &= r <= float(r_max)

    # Exclude non-finite pixels (e.g. beam-stop-masked pixels set to NaN).
    combined = angle_mask & r_mask & np.isfinite(data)
    if not combined.any():
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    r_int = np.round(r).astype(np.int64)
    flat_r = r_int[combined].ravel()
    flat_data = data[combined].ravel().astype(np.float64)
    n_bins = int(flat_r.max()) + 1
    radial_sum = np.bincount(flat_r, weights=flat_data, minlength=n_bins)
    radial_count = np.bincount(flat_r, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_intensity = np.where(radial_count > 0, radial_sum / radial_count, np.nan)
    return np.arange(n_bins, dtype=np.float64), mean_intensity


def angular_profile(
    data: np.ndarray,
    cx: float,
    cy: float,
    r_min: int,
    r_max: int,
    n_bins: int = 360,
    r: np.ndarray | None = None,
    angles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity I(theta) over 0..360 deg for pixels inside the radial ring.

    Precomputed ``r``/``angles`` grids skip recomputation. Returns
    (angle_centers_deg, mean_intensity); empty arrays when the ring is empty.
    """
    if r is None or angles is None:
        r, angles = polar_grids(data.shape[0], data.shape[1], cx, cy)

    # Radial mask — exclude the center pixel and non-finite (masked) pixels.
    r_mask = r >= max(float(r_min), 0.5)
    if r_max > 0:
        r_mask &= r <= float(r_max)
    r_mask &= np.isfinite(data)

    if not r_mask.any():
        return np.array([]), np.array([])

    valid_angles = angles[r_mask].ravel()
    valid_data = data[r_mask].ravel().astype(np.float64)

    bins = np.linspace(0.0, 360.0, n_bins + 1)
    sums, _ = np.histogram(valid_angles, bins=bins, weights=valid_data)
    counts, _ = np.histogram(valid_angles, bins=bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(counts > 0, sums / counts.astype(np.float64), np.nan)

    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers, avg

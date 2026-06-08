"""Pure (GUI-free) FTH / HERALDO reconstruction math.

Numerical kernels for Fourier-Transform Holography, extracted from
``src.gui.fth_reconstruction_tool`` so they can be unit-tested without
constructing any Qt widgets. The GUI tool reads its widget state, calls these
functions, and renders the results.
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

import logging

import numpy as np

log = logging.getLogger(__name__)


def bs_step(sigma: float, x: np.ndarray) -> np.ndarray:
    """Smooth step function for beamstop masking.

    Returns values in [0, 1]:  ~0 for x << 0 (inside BS),
    0.5 at x = 0 (edge), ~1 for x >> 0 (outside BS).
    ``sigma`` controls the transition width in pixels.
    """
    return 0.5 * (1.0 + np.tanh(x / (sigma + 1e-12)))


def line_gaussian_filter(Nx: int, Ny: int, phi_deg: float,
                         sigma: float, shift: float) -> np.ndarray:
    """Gaussian line-notch filter computed analytically (no rotated interpolation).

    This avoids angle-dependent width artifacts from resampling. Returned values
    are in [0, 1], with ~0 on the notch line and ~1 far away.
    """
    phi = np.deg2rad(phi_deg)
    rows, cols = np.meshgrid(np.arange(Nx, dtype=float), np.arange(Ny, dtype=float), indexing="ij")
    drow = rows - (Nx - 1) / 2.0
    dcol = cols - (Ny - 1) / 2.0

    # Signed perpendicular distance to the line in (col=x, row=y) view coordinates.
    # shift moves the notch along this perpendicular axis.
    perp = drow * np.cos(phi) - dcol * np.sin(phi) + shift
    return 1.0 - np.exp(-(perp ** 2) / (2.0 * sigma ** 2 + 1e-12))


def binary_filter(xmat: np.ndarray, ymat: np.ndarray, x0: float, y0: float,
                  phi_deg: float, width: int) -> np.ndarray:
    """Vectorised binary notch filter: zeros pixels within ``width`` of the
    line through (x0, y0) at angle ``phi_deg`` (perpendicular distance test)."""
    phi_rad = np.deg2rad(phi_deg)
    dr = xmat - x0
    dc = ymat - y0
    perp = np.abs(dr * np.sin(phi_rad) - dc * np.cos(phi_rad))
    return (perp > width).astype(float)


def differential_filter_kernel(xmat: np.ndarray, ymat: np.ndarray,
                               x0: float, y0: float, phi_deg: float) -> np.ndarray:
    """HERALDO differential (linear-ramp) filter kernel for a slit at ``phi_deg``."""
    phi = np.deg2rad(phi_deg)
    return 1j * (-(xmat - x0) * np.sin(phi) + (ymat - y0) * np.cos(phi))


def fth_transform(holo: np.ndarray, xmat: np.ndarray, ymat: np.ndarray,
                  x0: float, y0: float, nx: int, ny: int) -> np.ndarray:
    """FTH reconstruction: centered FFT of the hologram with phase correction."""
    phase_corr = np.exp(2j * np.pi * (xmat * x0 / nx + ymat * y0 / ny))
    return np.fft.fftshift(np.fft.fft2(holo)) * phase_corr


def estimate_balance_ratio(src_l: np.ndarray, src_r: np.ndarray) -> float:
    """Estimate scalar ratio r such that src_l ≈ r*src_r (L1 objective)."""
    from scipy.optimize import minimize_scalar

    m = np.isfinite(src_l) & np.isfinite(src_r) & (src_l > 0) & (src_r > 0)
    if not np.any(m):
        return 1.0
    a = src_l[m].astype(np.float64, copy=False)
    b = src_r[m].astype(np.float64, copy=False)
    step = max(1, a.size // 200_000)
    a = a[::step]
    b = b[::step]

    def obj(r: float) -> float:
        return float(np.sum(np.abs(a - r * b)))

    try:
        res = minimize_scalar(obj, bounds=(0.1, 10.0), method="bounded")
        if res.success and np.isfinite(res.x):
            return float(res.x)
    except Exception as exc:
        log.debug("Auto-balance ratio estimation failed; fallback to 1.0: %s", exc)
    return 1.0

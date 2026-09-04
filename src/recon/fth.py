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
import scipy.fft as sp_fft
from scipy.constants import c, e, h

log = logging.getLogger(__name__)


def _positive_finite(value: float, name: str) -> float:
    '''Return value as float, rejecting invalid physical parameters.'''
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be finite and greater than zero')
    return value


def photon_wavelength(energy_ev: float) -> float:
    '''Return the photon wavelength in metres for an energy in electronvolts.'''
    energy_ev = _positive_finite(energy_ev, 'photon energy')
    return h * c / (energy_ev * e)


def quantize_propagation_distance(distance_m: float, wavelength_m: float) -> float:
    '''Round a propagation distance to the nearest whole wavelength.'''
    distance_m = float(distance_m)
    if not np.isfinite(distance_m):
        raise ValueError('propagation distance must be finite')
    wavelength_m = _positive_finite(wavelength_m, 'wavelength')
    return float(np.round(distance_m / wavelength_m) * wavelength_m)


def propagation_kernel(
    shape: tuple[int, int],
    distance_m: float,
    detector_distance_m: float,
    pixel_size_m: float,
    energy_ev: float,
    *,
    quantize_wavelength: bool = True,
) -> np.ndarray:
    '''Return the detector-plane angular-spectrum propagation kernel.

    Detector distance and detector pixel size are in metres and photon energy
    is in electronvolts. The coordinates match the centered detector arrays
    consumed by :func:`fth_transform`.
    '''
    if len(shape) != 2 or any(int(size) <= 0 for size in shape):
        raise ValueError('shape must contain two positive dimensions')
    distance_m = float(distance_m)
    if not np.isfinite(distance_m):
        raise ValueError('propagation distance must be finite')
    detector_distance_m = _positive_finite(detector_distance_m, 'detector distance')
    pixel_size_m = _positive_finite(pixel_size_m, 'detector pixel size')
    wavelength_m = photon_wavelength(energy_ev)
    if quantize_wavelength:
        distance_m = quantize_propagation_distance(distance_m, wavelength_m)
    if distance_m == 0.0:
        return np.ones((int(shape[0]), int(shape[1])), dtype=np.complex128)

    nrows, ncols = int(shape[0]), int(shape[1])
    rows = np.arange(nrows, dtype=np.float64) - nrows / 2.0
    cols = np.arange(ncols, dtype=np.float64) - ncols / 2.0
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    radial_pixel2 = rr * rr + cc * cc
    radicand = 1.0 - (pixel_size_m / detector_distance_m) ** 2 * radial_pixel2
    if float(np.min(radicand)) < -1e-12:
        raise ValueError(
            'detector geometry produces evanescent samples; check pixel size '
            'and detector distance'
        )
    np.maximum(radicand, 0.0, out=radicand)
    phase = (2.0 * np.pi * distance_m / wavelength_m) * np.sqrt(radicand)
    return np.exp(1j * phase)


def propagate_hologram(
    hologram: np.ndarray,
    distance_m: float,
    detector_distance_m: float,
    pixel_size_m: float,
    energy_ev: float,
    *,
    quantize_wavelength: bool = True,
) -> np.ndarray:
    '''Apply free-space propagation to a centered two-dimensional hologram.'''
    hologram = np.asarray(hologram)
    if hologram.ndim != 2:
        raise ValueError('hologram must be a two-dimensional array')
    if not np.isfinite(float(distance_m)):
        raise ValueError('propagation distance must be finite')
    if float(distance_m) == 0.0:
        dtype = np.result_type(hologram.dtype, np.complex128)
        return hologram.astype(dtype, copy=True)
    kernel = propagation_kernel(
        hologram.shape,
        distance_m,
        detector_distance_m,
        pixel_size_m,
        energy_ev,
        quantize_wavelength=quantize_wavelength,
    )
    return hologram * kernel


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


def fth_phase_correction(xmat: np.ndarray, ymat: np.ndarray,
                         x0: float, y0: float, nx: int, ny: int) -> np.ndarray:
    """The centering phase ramp :func:`fth_transform` multiplies by.

    Separated so it can be built once and reused. It depends only on the
    geometry — the centred array and where its origin sits — none of which
    changes while a propagation distance is being tuned, yet building it costs
    125 ms on a 2048x2048 detector, which was being paid on every step of the
    focus slider.
    """
    return np.exp(2j * np.pi * (xmat * x0 / nx + ymat * y0 / ny))


def fth_transform(holo: np.ndarray, xmat: np.ndarray, ymat: np.ndarray,
                  x0: float, y0: float, nx: int, ny: int,
                  phase_correction: np.ndarray | None = None) -> np.ndarray:
    """FTH reconstruction: centered FFT of the hologram with phase correction.

    ``phase_correction`` is the array :func:`fth_phase_correction` returns; pass
    it to avoid rebuilding it. It is computed here when omitted, so a caller
    that does not care can ignore it.

    scipy's FFT rather than numpy's: measured on 2048x2048, 80 ms against
    204 ms for the identical computation, differing only in float64 round-off
    (1.2e-10). Single precision was measured too — another 13% — and rejected,
    since the whole gain here comes without touching the arithmetic.
    """
    if phase_correction is None:
        phase_correction = fth_phase_correction(xmat, ymat, x0, y0, nx, ny)
    return sp_fft.fftshift(sp_fft.fft2(holo, workers=-1)) * phase_correction


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

"""Pure (GUI-free) CDI math helpers.

These are the standalone numerical helpers used across the CDI reconstruction
worker threads in ``src.gui.cdi_reconstruction_tool``, extracted so they can be
unit-tested without constructing any Qt widgets. The iteration engines
themselves still live in the worker classes (they carry per-worker state and
Qt signals); only these pure functions are shared here.
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

from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter


def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D inverse FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def fourier_error_masked(
    F: np.ndarray,
    target: np.ndarray,
    floating_mask: Optional[np.ndarray] = None,
) -> float:
    """NRMSE of Fourier amplitudes, optionally skipping floating (bad) pixels."""
    valid = np.ones(target.shape, dtype=bool)
    if floating_mask is not None:
        valid = ~floating_mask.astype(bool)
    if not np.any(valid):
        return 0.0
    t = np.abs(target[valid])
    num = np.sum((np.abs(F[valid]) - t) ** 2)
    den = np.sum(t ** 2) + 1e-12
    return float(np.sqrt(num / den))


def tutorial_error_db(guess: np.ndarray, target: np.ndarray) -> float:
    """Reconstruction error in dB, mirroring the reference tutorial metric."""
    num = np.sum(np.abs(target - guess) ** 2)
    den = np.sum(np.abs(target) ** 2) + 1e-12
    return float(10.0 * np.log10((num / den) + 1e-30))


def arctan_beta_schedule(nit: int, beta_zero: float) -> np.ndarray:
    """Smooth arctan beta schedule used by HIO/RAAR."""
    step = np.arange(nit, dtype=float)
    return beta_zero + (
        0.5 - np.arctan((step - min(nit / 2.0, 700.0)) / (0.15 * nit)) / np.pi
    ) * (0.98 - beta_zero)


def preprocess_tutorial_intensity(arr: np.ndarray) -> np.ndarray:
    """Clean a measured intensity: subtract a 5th-percentile floor, clip negatives/NaNs."""
    out = np.asarray(arr, dtype=np.float64).copy()
    valid = out[np.isfinite(out) & (out != 0)]
    if valid.size:
        out -= float(np.percentile(valid, 5))
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out


def shrinkwrap_support(
    psi: np.ndarray,
    support_orig: np.ndarray,
    sigma: float,
    threshold: float,
) -> np.ndarray:
    """Shrinkwrap: threshold a Gaussian-blurred amplitude, masked by the original support."""
    blurred = gaussian_filter(np.abs(psi), sigma)
    peak = blurred.max()
    if peak == 0:
        return support_orig.copy()
    return (blurred > threshold * peak).astype(float) * support_orig


def run_single_target_cdi(
    amp: np.ndarray,
    support_orig: np.ndarray,
    algo_steps,
    beta: float,
    restarts: int,
    *,
    sw_enabled: bool = False,
    sw_sigma: float = 2.0,
    sw_threshold: float = 0.2,
    sw_every: int = 1,
    floating: Optional[np.ndarray] = None,
    init_obj: Optional[np.ndarray] = None,
    feat=(),
    emit_every: int = 10,
    on_progress: Optional[Callable[[int, int, float, np.ndarray], None]] = None,
    on_restart_done: Optional[Callable[[int, float], None]] = None,
    is_stopped: Callable[[], bool] = lambda: False,
):
    """Single-target CDI phase retrieval (ER / HIO / RAAR, optional shrinkwrap).

    Pure/GUI-free core of ``_CDIReconWorker.run``. Qt signals are replaced by the
    optional ``on_progress`` / ``on_restart_done`` callbacks; cooperative stopping
    is driven by the ``is_stopped`` predicate.

    Args mirror the worker's already-preprocessed state: ``amp`` is non-negative
    Fourier amplitude, ``support_orig`` the real-space support, ``feat`` an
    iterable of ``(bool_mask, complex_values)`` feature constraints.

    Returns ``(best_obj, best_errs)``.
    """
    best_obj = None
    best_err = np.inf
    best_errs: list[float] = []

    for r_idx in range(restarts):
        if is_stopped():
            break

        psi = init_obj.copy() if init_obj is not None \
            else ifft2c(amp * np.exp(1j * 2.0 * np.pi * np.random.rand(*amp.shape)))
        support = support_orig.copy()
        errs: list[float] = []
        global_iter = 0

        for algo, n_iters in algo_steps:
            for t in range(n_iters):
                if is_stopped():
                    break

                F = fft2c(psi)
                F_proj = amp * np.exp(1j * np.angle(F))
                if floating is not None:
                    # Bad pixels retain their current phase (float freely)
                    F_proj = np.where(floating, F, F_proj)
                psi_pF = ifft2c(F_proj)

                if algo == "er":
                    psi = psi_pF * support
                elif algo == "hio":
                    psi = np.where(support > 0, psi_pF, psi - beta * psi_pF)
                elif algo == "raar":
                    pS_pF = psi_pF * support
                    psi = beta * (2.0 * pS_pF - psi_pF) + (1.0 - beta) * psi_pF
                elif algo == "er_shrinkwrap":
                    psi = psi_pF * support
                    if sw_enabled and t > 0 and t % sw_every == 0:
                        support = shrinkwrap_support(psi, support_orig, sw_sigma, sw_threshold)
                elif algo == "hio_shrinkwrap":
                    psi = np.where(support > 0, psi_pF, psi - beta * psi_pF)
                    if sw_enabled and t > 0 and t % sw_every == 0:
                        support = shrinkwrap_support(psi, support_orig, sw_sigma, sw_threshold)
                elif algo == "raar_shrinkwrap":
                    pS_pF = psi_pF * support
                    psi = beta * (2.0 * pS_pF - psi_pF) + (1.0 - beta) * psi_pF
                    if sw_enabled and t > 0 and t % sw_every == 0:
                        support = shrinkwrap_support(psi, support_orig, sw_sigma, sw_threshold)
                else:
                    raise ValueError(f"Unknown algorithm: {algo!r}")

                # Region C — feature constraints (e.g. Pt pillar positions):
                # lock to reference values, overriding whatever the algorithm did.
                for feat_mask, feat_vals in feat:
                    psi[feat_mask] = feat_vals[feat_mask]

                err = fourier_error_masked(fft2c(psi), amp, floating)
                errs.append(err)
                global_iter += 1

                if global_iter % emit_every == 0 and on_progress is not None:
                    on_progress(r_idx, global_iter, err, psi.copy())

            if is_stopped():
                break

        if errs:
            if on_restart_done is not None:
                on_restart_done(r_idx, errs[-1])
            if errs[-1] < best_err:
                best_err = errs[-1]
                best_obj = psi.copy()
                best_errs = list(errs)

    # Align global phase so mean phase inside support ≈ 0
    if best_obj is not None:
        mask = support_orig > 0
        if not mask.any():
            mask = np.abs(best_obj) > 0
        best_obj *= np.exp(-1j * np.angle(np.sum(best_obj[mask])))

    return best_obj, best_errs


def match_amplitude_scale(
    psi: np.ndarray,
    amp: np.ndarray,
    floating: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Linearly rescale ``psi`` so its amplitude best fits the data amplitude ``amp``."""
    valid = (~floating) if floating is not None else np.ones(amp.shape, dtype=bool)
    x = amp[valid].ravel()
    y = np.abs(psi)[valid].ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) > 2 and np.std(x[finite]) > 1e-12:
        try:
            slope, intercept = np.polyfit(x[finite], y[finite], 1)
            if np.isfinite(slope) and abs(slope) > 1e-12:
                return (psi - intercept) / slope
        except Exception:
            pass
    return psi


def run_multi_target_cdi(
    targets,
    beta: float,
    restarts: int,
    *,
    sw_enabled: bool = False,
    sw_sigma: float = 2.0,
    sw_threshold: float = 0.2,
    sw_every: int = 1,
    floating: Optional[np.ndarray] = None,
    init_obj: Optional[np.ndarray] = None,
    emit_every: int = 10,
    scale_initial_guess: bool = False,
    scale_inherited_phase: bool = False,
    on_progress: Optional[Callable[[int, int, float, np.ndarray], None]] = None,
    on_restart_done: Optional[Callable[[int, float], None]] = None,
    is_stopped: Callable[[], bool] = lambda: False,
):
    """Multi-target CDI pipeline with phase inheritance between targets.

    Pure/GUI-free core of ``_CDIMultiTargetWorker.run``. ``targets`` is the
    already-normalized list of dicts (kind/operation/data_source/amp/support/
    steps/feat/beta_schedule/bg_sub). Qt signals are replaced by the optional
    ``on_progress`` / ``on_restart_done`` callbacks; ``is_stopped`` drives the stop.

    Returns ``(best_obj, best_errs, best_sources)``.
    """
    best_obj: Optional[np.ndarray] = None
    best_err = np.inf
    best_errs: list[float] = []
    best_sources: dict[str, np.ndarray] = {}

    for r_idx in range(restarts):
        if is_stopped():
            break

        psi: Optional[np.ndarray] = init_obj.copy() if init_obj is not None else None
        restart_errs: list[float] = []
        global_iter = 0
        prev_amp_sq_sum: Optional[float] = None   # for inherited-phase scaling
        is_first_target = True

        pending_intervals: list[str] = []
        restart_sources: dict[str, np.ndarray] = {}

        for tgt in targets:
            if is_stopped():
                break
            if tgt['kind'] == 'interval':
                pending_intervals.append(tgt['operation'])
                continue

            support_orig = tgt['support']
            feat = tgt['feat']
            algo_steps = tgt['steps']
            beta_schedule = tgt['beta_schedule']
            data_source = tgt.get('data_source')

            # Optional background subtraction before √I
            raw_amp = tgt['amp']
            if tgt['bg_sub']:
                intensity = preprocess_tutorial_intensity(raw_amp ** 2)
                amp = np.sqrt(intensity)
            else:
                amp = raw_amp

            was_cold = (psi is None)

            if was_cold:
                psi = ifft2c(amp * np.exp(
                    1j * 2.0 * np.pi * np.random.rand(*amp.shape)
                ))
            for operation in pending_intervals:
                if operation == "reset support":
                    phase = np.angle(ifft2c(support_orig))
                    psi = support_orig.astype(np.float64) * np.exp(1j * phase)
                elif operation == "random phase":
                    psi = ifft2c(amp * np.exp(
                        1j * 2.0 * np.pi * np.random.rand(*amp.shape)
                    ))
                elif operation == "normalize" and prev_amp_sq_sum is not None:
                    curr_sum = float(np.sum(amp ** 2))
                    if prev_amp_sq_sum > 1e-12:
                        psi = psi * np.sqrt(curr_sum / prev_amp_sq_sum)
                elif operation == "match data amplitude":
                    psi = match_amplitude_scale(psi, amp, floating)
            pending_intervals.clear()

            # Scale initial object amplitude to match first target's data amplitude
            if (is_first_target and not was_cold
                    and scale_initial_guess
                    and init_obj is not None):
                psi = match_amplitude_scale(psi, amp, floating)

            prev_amp_sq_sum = float(np.sum(amp ** 2))
            is_first_target = False

            # Determine β schedule for this target
            # "arctan_cold": arctan only if this target started cold, else constant
            n_total_tgt = sum(n for _, n in algo_steps)
            use_arctan = False
            if beta_schedule == "arctan":
                use_arctan = True
            elif beta_schedule == "arctan_cold":
                use_arctan = was_cold
            if use_arctan and n_total_tgt > 0:
                beta_arr = arctan_beta_schedule(n_total_tgt, beta_zero=0.5)
            else:
                beta_arr = None  # use constant beta

            support = support_orig.copy()
            tgt_iter = 0  # iteration index within this target (for beta_arr)

            for algo, n_iters in algo_steps:
                if is_stopped():
                    break
                for t in range(n_iters):
                    if is_stopped():
                        break

                    cur_beta = (float(beta_arr[min(tgt_iter, len(beta_arr) - 1)])
                                if beta_arr is not None else beta)

                    F = fft2c(psi)
                    F_proj = amp * np.exp(1j * np.angle(F))
                    if floating is not None:
                        F_proj = np.where(floating, F, F_proj)
                    psi_pF = ifft2c(F_proj)

                    if algo == "er":
                        psi = psi_pF * support
                    elif algo == "hio":
                        psi = np.where(support > 0, psi_pF,
                                       psi - cur_beta * psi_pF)
                    elif algo == "raar":
                        pS_pF = psi_pF * support
                        psi = (cur_beta * (2.0 * pS_pF - psi_pF)
                               + (1.0 - cur_beta) * psi_pF)
                    elif algo == "er_shrinkwrap":
                        psi = psi_pF * support
                        if sw_enabled and t > 0 and t % sw_every == 0:
                            support = shrinkwrap_support(psi, support_orig, sw_sigma, sw_threshold)
                    elif algo == "hio_shrinkwrap":
                        psi = np.where(support > 0, psi_pF,
                                       psi - cur_beta * psi_pF)
                        if sw_enabled and t > 0 and t % sw_every == 0:
                            support = shrinkwrap_support(psi, support_orig, sw_sigma, sw_threshold)
                    elif algo == "raar_shrinkwrap":
                        pS_pF = psi_pF * support
                        psi = (cur_beta * (2.0 * pS_pF - psi_pF)
                               + (1.0 - cur_beta) * psi_pF)
                        if sw_enabled and t > 0 and t % sw_every == 0:
                            support = shrinkwrap_support(psi, support_orig, sw_sigma, sw_threshold)
                    else:
                        raise ValueError(f"Unknown algorithm: {algo!r}")

                    for feat_mask, feat_vals in feat:
                        psi[feat_mask] = feat_vals[feat_mask]

                    err = fourier_error_masked(fft2c(psi), amp, floating)
                    restart_errs.append(err)
                    global_iter += 1
                    tgt_iter += 1

                    if global_iter % emit_every == 0 and on_progress is not None:
                        on_progress(r_idx, global_iter, err, psi.copy())

            if psi is not None and data_source in ("CL", "CR"):
                restart_sources[data_source] = fft2c(psi).copy()

        final_err = restart_errs[-1] if restart_errs else np.inf
        if on_restart_done is not None:
            on_restart_done(r_idx, final_err)

        if final_err < best_err and psi is not None:
            best_err = final_err
            best_obj = psi.copy()
            best_errs = list(restart_errs)
            best_sources = {
                source: field.copy()
                for source, field in restart_sources.items()
            }

        # Each restart begins fresh from initial_obj (not inherited)
        psi = None

    if best_obj is not None:
        mask = np.abs(best_obj) > np.percentile(np.abs(best_obj), 75)
        if not np.any(mask):
            mask = np.abs(best_obj) > 0
        best_obj *= np.exp(-1j * np.angle(np.sum(best_obj[mask])))

    return best_obj, best_errs, best_sources


def _mine_phase_retrieve(
    diffract: np.ndarray,
    support: np.ndarray,
    phase: np.ndarray,
    bsmask: np.ndarray,
    nit: int,
    beta_zero: float,
    beta_mode: str,
    average_img: int,
    stage: str,
    done_offset: int,
    total: int,
    plot_every: int,
    *,
    on_progress: Optional[Callable[[str, int, int, float, np.ndarray], None]] = None,
    is_stopped: Callable[[], bool] = lambda: False,
) -> tuple:
    """Tutorial "mine" CDI phase-retrieval stage (returns ``(phase, errs)``)."""
    beta = (
        arctan_beta_schedule(nit, beta_zero)
        if beta_mode == "arctan" else beta_zero * np.ones(nit)
    )
    bsmask_f = np.fft.fftshift(bsmask.astype(float))
    guess = (1.0 - bsmask) * diffract * np.exp(1j * np.angle(phase)) + phase * bsmask
    guess_cp = np.fft.fftshift(guess)
    mask_cp = np.fft.fftshift(support)
    diffract_cp = np.fft.fftshift(diffract)
    eps = 1e-12

    prev = np.fft.fft2(
        (1.0 - bsmask_f) * diffract_cp * np.exp(1j * np.angle(guess_cp))
        + guess_cp * bsmask_f
    )

    errs: list[float] = []
    requested_keep = max(1, min(int(average_img), max(1, nit)))
    bytes_per_guess = int(np.prod(diffract.shape)) * np.dtype(np.complex64).itemsize
    max_average_bytes = 128 * 1024 * 1024
    memory_capped_keep = max(1, max_average_bytes // max(1, bytes_per_guess))
    best_keep = max(1, min(requested_keep, memory_capped_keep))
    best_guess = np.zeros((best_keep,) + diffract.shape, dtype=np.complex64)
    best_error = np.full(best_keep, np.inf, dtype=np.float64)

    for s in range(nit):
        if is_stopped():
            break
        inv = np.fft.fft2(
            guess_cp * ((1.0 - bsmask_f) * diffract_cp / (np.abs(guess_cp) + eps) + bsmask_f)
        )
        # Tutorial "mine" update.
        inv += beta[s] * (prev - 2.0 * inv) * (1.0 - mask_cp)
        prev = np.copy(inv)
        guess_cp = np.fft.ifft2(inv)

        err = tutorial_error_db(
            np.abs(guess_cp) * (1.0 - bsmask_f),
            diffract_cp * (1.0 - bsmask_f),
        )
        if s <= 2 or s % plot_every == 0 or s >= nit - best_keep * 2:
            errs.append(err)
            if on_progress is not None:
                on_progress(stage, done_offset + s + 1, total, err, np.fft.ifftshift(inv))

        if s >= max(2, nit - best_keep * 2):
            worst = int(np.argmax(best_error))
            if err < best_error[worst]:
                best_error[worst] = err
                best_guess[worst] = guess_cp.astype(np.complex64, copy=False)

    finite = np.isfinite(best_error)
    if np.any(finite):
        guess_cp = np.mean(best_guess[finite], axis=0)

    guess_cp = (
        (1.0 - bsmask_f) * diffract_cp * np.exp(1j * np.angle(guess_cp))
        + guess_cp * bsmask_f
    )
    return np.fft.ifftshift(guess_cp), errs


def run_sequential_cdi(
    cl_raw: np.ndarray,
    cr_raw: np.ndarray,
    pixel_mask: np.ndarray,
    support: np.ndarray,
    pipeline,
    *,
    on_progress: Optional[Callable[[str, int, int, float, np.ndarray], None]] = None,
    is_stopped: Callable[[], bool] = lambda: False,
):
    """Sequential ("Classic" CL->CR tutorial/mine) CDI pipeline.

    Pure/GUI-free core of ``_CDISequentialReconWorker.run``. Returns
    ``(retrieved_cl, retrieved_cr, diff, recon, errs_cl, errs_cr, stopped)``.
    """
    if cl_raw.shape != cr_raw.shape or cl_raw.shape != support.shape:
        raise ValueError(
            f"Shape mismatch: CL {cl_raw.shape}, CR {cr_raw.shape}, "
            f"support {support.shape}"
        )

    raw_intensity_by_source = {"CL": cl_raw, "CR": cr_raw}

    def _target_intensity(source: str, target: dict) -> np.ndarray:
        raw = raw_intensity_by_source[source]
        if bool(target.get('bg_sub', False)):
            return preprocess_tutorial_intensity(raw)
        out = np.asarray(raw, dtype=np.float64).copy()
        out[~np.isfinite(out)] = 0.0
        out[out < 0] = 0.0
        return out

    def _target_bsmask(intensity: np.ndarray) -> np.ndarray:
        bsmask = pixel_mask.copy()
        bsmask[intensity <= 2] = True
        return bsmask

    start = ifft2c(support)
    first_source = next(
        (
            target.get('data_source', 'CL')
            for target in pipeline
            if target.get('kind', 'target') == 'target'
            if target.get('data_source', 'CL') in raw_intensity_by_source
        ),
        "CL",
    )
    first_target = next(
        (
            target
            for target in pipeline
            if target.get('kind', 'target') == 'target'
            and target.get('data_source', 'CL') == first_source
        ),
        {'bg_sub': False},
    )
    first_intensity = _target_intensity(first_source, first_target)
    valid = ~pixel_mask
    x = np.sqrt(np.maximum(first_intensity, 0.0))[valid].ravel()
    y = np.abs(start)[valid].ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) > 2 and np.std(x[finite]) > 1e-12:
        try:
            slope, intercept = np.polyfit(x[finite], y[finite], 1)
            if np.isfinite(slope) and abs(slope) > 1e-12:
                start = (start - intercept) / slope
        except Exception:
            pass

    total = sum(
        int(n)
        for target in pipeline
        if target.get('kind', 'target') == 'target'
        for algo, n in target.get('steps', [])
        if algo == "mine"
    )
    if total <= 0:
        raise ValueError("Mine pipeline has no iterations.")
    done = 0
    phase = start
    previous_sum: Optional[float] = None
    retrieved_by_source: dict[str, np.ndarray] = {}
    errors_by_source: dict[str, list[float]] = {"CL": [], "CR": []}
    pending_intervals: list[str] = []

    for target_idx, target in enumerate(pipeline, start=1):
        if target.get('kind', 'target') == 'interval':
            pending_intervals.append(target.get('operation', 'none'))
            continue
        source = target.get('data_source', 'CL')
        if source not in raw_intensity_by_source:
            raise ValueError(
                "Mine pipeline supports CL and CR targets only; "
                f"got {source!r}."
            )
        intensity = _target_intensity(source, target)
        for operation in pending_intervals:
            if operation == "reset support":
                phase = start.copy()
            elif operation == "random phase":
                shape = intensity.shape
                phase = np.exp(1j * 2.0 * np.pi * np.random.rand(*shape))
            elif operation == "normalize" and previous_sum is not None:
                curr_sum = float(np.sum(intensity))
                phase = phase * np.sqrt((curr_sum + 1e-12) / (previous_sum + 1e-12))
            elif operation == "match data amplitude":
                diffract = np.sqrt(np.maximum(intensity, 0.0))
                valid = ~pixel_mask
                x = diffract[valid].ravel()
                y = np.abs(phase)[valid].ravel()
                finite = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(finite) > 2 and np.std(x[finite]) > 1e-12:
                    try:
                        slope, intercept = np.polyfit(x[finite], y[finite], 1)
                        if np.isfinite(slope) and abs(slope) > 1e-12:
                            phase = (phase - intercept) / slope
                    except Exception:
                        pass
        pending_intervals.clear()
        previous_sum = float(np.sum(intensity))

        for algo, n_iters in target.get('steps', []):
            if algo != "mine":
                raise ValueError(
                    "Mine pipeline cannot mix tutorial/mine steps with "
                    f"{algo!r}. Use the standard CDI pipeline for ER/HIO/RAAR."
                )
            beta_key = target.get('beta_schedule', "arctan (cold-start only)")
            beta_mode = "arctan" if (
                beta_key == "arctan"
                or (beta_key == "arctan (cold-start only)" and done == 0)
            ) else "const"
            plot_every = 25 if int(n_iters) >= 100 else 5
            stage = f"{source} target {target_idx}: {int(n_iters)} mine/{beta_mode}"
            phase, errs = _mine_phase_retrieve(
                diffract=np.sqrt(np.maximum(intensity, 0.0)),
                support=support,
                phase=phase,
                bsmask=_target_bsmask(intensity),
                nit=int(n_iters),
                beta_zero=0.5,
                beta_mode=beta_mode,
                average_img=30,
                stage=stage,
                done_offset=done,
                total=total,
                plot_every=plot_every,
                on_progress=on_progress,
                is_stopped=is_stopped,
            )
            done += int(n_iters)
            errors_by_source[source].extend(errs)
            retrieved_by_source[source] = phase
            if is_stopped():
                return None, None, None, None, [], [], True

    retrieved_cl = retrieved_by_source.get("CL")
    retrieved_cr = retrieved_by_source.get("CR")
    if retrieved_cl is not None and retrieved_cr is not None:
        diff = retrieved_cl - retrieved_cr
        recon = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(diff)))
    else:
        diff = None
        recon = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(phase)))

    return (
        retrieved_cl,
        retrieved_cr,
        diff,
        recon,
        errors_by_source["CL"],
        errors_by_source["CR"],
        False,
    )

"""Unit tests for the GUI-free CDI math helpers (src.recon.cdi)."""

import numpy as np
import pytest

from src.recon.cdi import (
    arctan_beta_schedule,
    fft2c,
    fourier_error_masked,
    ifft2c,
    match_amplitude_scale,
    preprocess_tutorial_intensity,
    run_multi_target_cdi,
    run_sequential_cdi,
    run_single_target_cdi,
    shrinkwrap_support,
    tutorial_error_db,
)


def test_fft_ifft_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    assert np.allclose(ifft2c(fft2c(x)), x, atol=1e-10)


def test_fft2c_is_centered():
    # A centered delta transforms to a constant-magnitude spectrum.
    x = np.zeros((8, 8), dtype=complex)
    x[4, 4] = 1.0
    F = fft2c(x)
    assert np.allclose(np.abs(F), np.abs(F).flat[0], atol=1e-10)


def test_fourier_error_zero_when_identical():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    assert fourier_error_masked(f, f) == pytest.approx(0.0, abs=1e-12)


def test_fourier_error_all_masked_returns_zero():
    f = np.ones((4, 4), dtype=complex)
    target = 2.0 * np.ones((4, 4), dtype=complex)
    floating = np.ones((4, 4), dtype=bool)  # every pixel masked out
    assert fourier_error_masked(f, target, floating) == 0.0


def test_fourier_error_known_value():
    # |F|=1 everywhere, |target|=2 everywhere -> NRMSE = sqrt(sum(1)/sum(4)) = 0.5
    f = np.ones((4, 4), dtype=complex)
    target = 2.0 * np.ones((4, 4), dtype=complex)
    assert fourier_error_masked(f, target) == pytest.approx(0.5, rel=1e-6)


def test_fourier_error_floating_mask_excludes_pixels():
    # Make the only-correct pixel the masked one; error should reflect the rest.
    f = np.ones((2, 2), dtype=complex)
    target = np.array([[1.0, 3.0], [3.0, 3.0]], dtype=complex)
    floating = np.array([[True, False], [False, False]])  # mask out the matching pixel
    # valid pixels: |F|=1, |t|=3 -> sqrt(3*4 / (3*9)) = sqrt(12/27)
    assert fourier_error_masked(f, target, floating) == pytest.approx(np.sqrt(12 / 27), rel=1e-6)


def test_tutorial_error_db_identical_is_very_negative():
    a = np.ones((4, 4), dtype=complex)
    assert tutorial_error_db(a, a) < -100.0


def test_arctan_beta_schedule_shape_and_range():
    nit = 50
    beta = arctan_beta_schedule(nit, 0.9)
    assert beta.shape == (nit,)
    assert np.all(beta >= 0.9 - 1e-9)
    assert np.all(beta <= 0.98 + 1e-9)


def test_preprocess_clips_negatives_and_nans():
    arr = np.array([[np.nan, -5.0], [10.0, 100.0]])
    out = preprocess_tutorial_intensity(arr)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert out.dtype == np.float64


def test_preprocess_subtracts_percentile_floor():
    arr = np.arange(1, 101, dtype=float).reshape(10, 10)
    out = preprocess_tutorial_intensity(arr)
    # 5th percentile floor subtracted, so the minimum positive value drops.
    assert out.min() == 0.0
    assert out.max() < arr.max()


# ---------------------------------------------------------------------------
# shrinkwrap_support
# ---------------------------------------------------------------------------

def test_shrinkwrap_masked_by_original_support():
    psi = np.zeros((16, 16), dtype=complex)
    psi[6:10, 6:10] = 5.0
    support0 = np.ones((16, 16), dtype=float)
    out = shrinkwrap_support(psi, support0, sigma=1.0, threshold=0.2)
    assert out.sum() < support0.sum()  # tightened
    # Never exceeds the original support.
    assert np.all(out <= support0)


def test_shrinkwrap_zero_peak_returns_original():
    psi = np.zeros((8, 8), dtype=complex)
    support0 = np.ones((8, 8), dtype=float)
    out = shrinkwrap_support(psi, support0, sigma=1.0, threshold=0.2)
    assert np.array_equal(out, support0)


# ---------------------------------------------------------------------------
# run_single_target_cdi (pure engine extracted from _CDIReconWorker)
# ---------------------------------------------------------------------------

def _compact_problem(n=24, seed=123):
    rng = np.random.default_rng(seed)
    support = np.zeros((n, n), dtype=float)
    support[n // 4:3 * n // 4, n // 4:3 * n // 4] = 1.0
    obj_true = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * support
    amp = np.abs(fft2c(obj_true))
    return amp, support, obj_true


def test_engine_shapes_and_error_length():
    amp, support, _ = _compact_problem()
    np.random.seed(0)
    obj, errs = run_single_target_cdi(
        amp, support, [("hio", 5), ("er", 5)], beta=0.9, restarts=2,
    )
    assert obj.shape == amp.shape
    # restarts re-run independently; best_errs holds one restart's full curve.
    assert len(errs) == 10


def test_engine_recovers_from_true_object_seed():
    # Seeding ER with the true object keeps error ~0.
    amp, support, obj_true = _compact_problem()
    obj, errs = run_single_target_cdi(
        amp, support, [("er", 3)], beta=0.9, restarts=1, init_obj=obj_true,
    )
    assert errs[-1] == pytest.approx(0.0, abs=1e-9)


def test_engine_deterministic_with_global_seed():
    amp, support, _ = _compact_problem()
    np.random.seed(42)
    a_obj, a_errs = run_single_target_cdi(amp, support, [("hio", 6)], 0.9, 2)
    np.random.seed(42)
    b_obj, b_errs = run_single_target_cdi(amp, support, [("hio", 6)], 0.9, 2)
    assert np.allclose(a_obj, b_obj)
    assert a_errs == b_errs


def test_engine_invokes_callbacks_and_respects_stop():
    amp, support, _ = _compact_problem()
    progress, restarts_done = [], []
    np.random.seed(1)
    run_single_target_cdi(
        amp, support, [("hio", 10)], 0.9, restarts=2, emit_every=5,
        on_progress=lambda r, g, e, p: progress.append((r, g)),
        on_restart_done=lambda r, e: restarts_done.append(r),
    )
    assert len(progress) == 4  # 2 restarts * (10 iters / emit_every 5)
    assert restarts_done == [0, 1]

    # is_stopped short-circuits before any iteration runs.
    np.random.seed(1)
    obj, errs = run_single_target_cdi(
        amp, support, [("hio", 10)], 0.9, restarts=2, is_stopped=lambda: True,
    )
    assert obj is None and errs == []


def test_engine_unknown_algorithm_raises():
    amp, support, _ = _compact_problem()
    np.random.seed(0)
    with pytest.raises(ValueError):
        run_single_target_cdi(amp, support, [("bogus", 1)], 0.9, 1)


def test_engine_feature_constraints_locked():
    amp, support, _ = _compact_problem()
    feat_mask = np.zeros((24, 24), dtype=bool)
    feat_mask[12, 12] = True
    feat_vals = np.zeros((24, 24), dtype=complex)
    feat_vals[12, 12] = 3.0 + 4.0j
    np.random.seed(0)
    obj, _ = run_single_target_cdi(
        amp, support, [("er", 4)], 0.9, restarts=1, feat=[(feat_mask, feat_vals)],
    )
    # Feature pixel is locked after each projection; the final object is
    # globally phase-aligned, so compare magnitude (rotation-invariant).
    assert abs(obj[12, 12]) == pytest.approx(5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# match_amplitude_scale
# ---------------------------------------------------------------------------

def test_match_amplitude_scale_recovers_linear_factor():
    rng = np.random.default_rng(3)
    amp = np.abs(rng.standard_normal((10, 10))) + 0.5
    psi = (3.0 * amp).astype(complex)  # |psi| = 3*amp exactly
    out = match_amplitude_scale(psi, amp)
    assert np.allclose(np.abs(out), amp, atol=1e-9)


def test_match_amplitude_scale_degenerate_returns_input():
    amp = np.ones((4, 4))  # zero std -> no fit
    psi = np.full((4, 4), 2.0, dtype=complex)
    out = match_amplitude_scale(psi, amp)
    assert np.array_equal(out, psi)


# ---------------------------------------------------------------------------
# run_multi_target_cdi (pure engine extracted from _CDIMultiTargetWorker)
# ---------------------------------------------------------------------------

def _norm_target(amp, support, steps, *, data_source=None, feat=(),
                 beta_schedule="constant", bg_sub=False):
    return {"kind": "target", "operation": "none", "data_source": data_source,
            "amp": np.abs(amp).astype(np.float64),
            "support": np.asarray(support, dtype=np.float64),
            "steps": list(steps),
            "feat": [(m.astype(bool), v.astype(np.complex128)) for m, v in feat],
            "beta_schedule": beta_schedule, "bg_sub": bg_sub}


def test_multi_target_phase_inheritance_and_sources():
    amp, support, _ = _compact_problem(n=20)
    targets = [
        _norm_target(amp, support, [("hio", 4)], data_source="CL"),
        _norm_target(amp, support, [("er", 3)], data_source="CR"),
    ]
    np.random.seed(0)
    obj, errs, sources = run_multi_target_cdi(targets, beta=0.9, restarts=1)
    assert obj.shape == amp.shape
    assert len(errs) == 7
    assert set(sources) == {"CL", "CR"}


def test_multi_target_deterministic_with_seed():
    amp, support, _ = _compact_problem(n=20)
    targets = [_norm_target(amp, support, [("hio", 5)], data_source="CL")]
    np.random.seed(11)
    a_obj, a_errs, _ = run_multi_target_cdi(targets, 0.9, 2)
    np.random.seed(11)
    b_obj, b_errs, _ = run_multi_target_cdi(targets, 0.9, 2)
    assert np.allclose(a_obj, b_obj)
    assert a_errs == b_errs


def test_multi_target_stop_yields_no_result():
    amp, support, _ = _compact_problem(n=20)
    targets = [_norm_target(amp, support, [("hio", 5)], data_source="CL")]
    obj, errs, sources = run_multi_target_cdi(
        targets, 0.9, 2, is_stopped=lambda: True,
    )
    assert obj is None and errs == [] and sources == {}


# ---------------------------------------------------------------------------
# run_sequential_cdi (pure engine extracted from _CDISequentialReconWorker)
# ---------------------------------------------------------------------------

def _seq_setup(n=18):
    rng = np.random.default_rng(99)
    support = np.zeros((n, n), dtype=float)
    support[4:14, 4:14] = 1.0
    cl = np.abs(rng.standard_normal((n, n))) * 50 + 1.0
    cr = np.abs(rng.standard_normal((n, n))) * 60 + 1.0
    pixel_mask = np.zeros((n, n), dtype=bool)
    return cl, cr, pixel_mask, support


def test_sequential_cl_cr_produces_diff_and_recon():
    cl, cr, mask, support = _seq_setup()
    pipeline = [
        {"kind": "target", "data_source": "CL", "steps": [("mine", 10)]},
        {"kind": "target", "data_source": "CR", "steps": [("mine", 8)]},
    ]
    np.random.seed(0)
    rcl, rcr, diff, recon, ecl, ecr, stopped = run_sequential_cdi(
        cl, cr, mask, support, pipeline)
    assert not stopped
    assert rcl is not None and rcr is not None
    assert np.allclose(diff, rcl - rcr)
    assert recon.shape == support.shape
    assert len(ecl) > 0 and len(ecr) > 0


def test_sequential_shape_mismatch_raises():
    cl, cr, mask, support = _seq_setup()
    with pytest.raises(ValueError):
        run_sequential_cdi(cl, cr[:10, :10], mask, support,
                           [{"kind": "target", "data_source": "CL", "steps": [("mine", 2)]}])


def test_sequential_no_iterations_raises():
    cl, cr, mask, support = _seq_setup()
    with pytest.raises(ValueError):
        run_sequential_cdi(cl, cr, mask, support, [{"kind": "interval", "operation": "normalize"}])


def test_sequential_rejects_non_mine_step():
    cl, cr, mask, support = _seq_setup()
    with pytest.raises(ValueError):
        run_sequential_cdi(cl, cr, mask, support,
                           [{"kind": "target", "data_source": "CL", "steps": [("hio", 5)]}])


def test_sequential_stop_flag_propagates():
    cl, cr, mask, support = _seq_setup()
    pipeline = [{"kind": "target", "data_source": "CL", "steps": [("mine", 10)]}]
    np.random.seed(0)
    rcl, rcr, diff, recon, ecl, ecr, stopped = run_sequential_cdi(
        cl, cr, mask, support, pipeline, is_stopped=lambda: True)
    assert stopped is True

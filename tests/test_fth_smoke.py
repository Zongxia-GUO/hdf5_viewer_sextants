from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from src.gui._shared import get_colormap as _get_colormap
from src.gui.fth_reconstruction_tool import FTH_COLORMAPS, FTHReconstructionTool


def _write_h5(path: Path, dataset: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=data)


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    from PyQt6.QtWidgets import QApplication

    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _prepare_tool_with_arrays(qapp, n: int = 128) -> FTHReconstructionTool:
    tool = FTHReconstructionTool(opened_files=())
    arr_l = np.full((n, n), 10.0, dtype=np.float64)
    arr_r = np.full((n, n), 8.0, dtype=np.float64)
    tool._CL = arr_l
    tool._CR = arr_r
    tool._t1_xmid.setValue(n // 2)
    tool._t1_ymid.setValue(n // 2)
    tool._compute_centered_hologram()
    return tool


def _select_reconstruction_tab(tool):
    """What switching to the last page does, without a real click."""
    tool._tabs.setCurrentIndex(2)
    tool._on_tab_changed(2)


def test_reconstruction_inherits_the_phase_and_amplitude_just_tuned(qapp):
    """They are the same two quantities on both pages; arriving with them back
    at their defaults threw away the tuning done on the hologram."""
    tool = _prepare_tool_with_arrays(qapp)
    tool._phase_scale_slider.setValue(120)
    tool._rs_scale_slider.setValue(250)

    _select_reconstruction_tab(tool)

    assert tool._t4_ph_slider.value() == 120
    assert tool._t4_rs_slider.value() == 250


def test_a_reconstruction_slider_the_user_moved_is_not_overwritten(qapp):
    """Inheriting again would undo that work on every visit to the page."""
    tool = _prepare_tool_with_arrays(qapp)
    tool._phase_scale_slider.setValue(120)
    _select_reconstruction_tab(tool)

    tool._t4_ph_slider.setValue(-200)
    tool._mark_t4_sliders_touched()      # what a drag emits

    tool._tabs.setCurrentIndex(1)
    _select_reconstruction_tab(tool)

    assert tool._t4_ph_slider.value() == -200


def test_inheriting_does_not_count_as_the_user_taking_over(qapp):
    tool = _prepare_tool_with_arrays(qapp)
    tool._phase_scale_slider.setValue(90)

    _select_reconstruction_tab(tool)

    assert tool._t4_sliders_touched is False
    tool._phase_scale_slider.setValue(-90)
    _select_reconstruction_tab(tool)
    assert tool._t4_ph_slider.value() == -90, "still following the previous page"


def test_run_locked_lands_on_the_reconstruction_page(qapp):
    """Run Locked is one click from a new file to a reconstruction, so it must
    not stop on the alignment page it has just finished setting up."""
    tool = _prepare_tool_with_arrays(qapp)
    tool._lock_current_params()
    tool._tabs.setCurrentIndex(0)

    tool._apply_locked_params_to_current_data()

    index = tool._tabs.currentIndex()
    assert tool._tabs.tabText(index) == "Reconstruction"


def test_the_other_pages_are_still_reachable(qapp):
    """Switching pages is a convenience, not a lock."""
    tool = _prepare_tool_with_arrays(qapp)
    tool._lock_current_params()
    tool._apply_locked_params_to_current_data()

    tool._tabs.setCurrentIndex(0)
    assert tool._tabs.tabText(tool._tabs.currentIndex()) == "Alignment"


def test_smoke_load_cl_cr_in_background(qapp, tmp_path: Path):
    cl_path = tmp_path / "cl.h5"
    cr_path = tmp_path / "cr.h5"
    _write_h5(cl_path, "img", np.ones((128, 128), dtype=np.float64) * 11.0)
    _write_h5(cr_path, "img", np.ones((128, 128), dtype=np.float64) * 9.0)

    tool = FTHReconstructionTool(opened_files=())
    tool._t1_xmid.setValue(64)
    tool._t1_ymid.setValue(64)
    tool.add_dataset_to_combo(f"{cl_path}::img", "CL")
    tool.add_dataset_to_combo(f"{cr_path}::img", "CR")

    tool._load_data()
    ok = _wait_until(lambda: tool._CL is not None and tool._CR is not None and tool._CL_c is not None, timeout_s=5.0)
    assert ok, "Background load did not complete in time"
    assert tool._CL.shape == (128, 128)
    assert tool._CR.shape == (128, 128)
    assert tool._CL_c is not None and tool._CR_c is not None
    tool.close()


def test_smoke_load_single_dataset_in_background(qapp, tmp_path: Path):
    h5_path = tmp_path / "single.h5"
    data = np.ones((128, 128), dtype=np.float64) * 11.0
    _write_h5(h5_path, "img", data)

    tool = FTHReconstructionTool(opened_files=())
    tool.add_dataset_to_combo(f"{h5_path}::img", "CL")

    tool._load_data()
    ok = _wait_until(lambda: tool._CL is not None and tool._CR is not None and tool._CL_c is not None, timeout_s=5.0)
    assert ok, "Single dataset load did not complete in time"
    np.testing.assert_allclose(tool._CL, data)
    np.testing.assert_allclose(tool._CR, np.zeros_like(data))
    assert tool._single_dataset_mode is True
    assert tool._CL_c is not None and tool._CR_c is not None
    tool.close()


def test_loaded_image_center_defaults_to_data_geometry(qapp):
    tool = FTHReconstructionTool(opened_files=())
    tool._CL = np.ones((1228, 1228), dtype=np.float64)
    tool._CR = np.ones((1228, 1228), dtype=np.float64)

    tool._initialize_center_controls_for_loaded_shape(tool._CL.shape)
    tool._compute_centered_hologram()

    assert tool._t1_xmid.value() == 614
    assert tool._t1_ymid.value() == 614
    assert tool._CL_c is not None
    assert tool._CL_c.shape == (1226, 1226)
    tool.close()


def test_slit_none_path_has_no_secondary_slit_filter(qapp):
    tool = _prepare_tool_with_arrays(qapp)
    tool._chk_balance.setChecked(False)
    tool._slit_combo.setCurrentText("None")
    tool._filter_combo.setCurrentText("Gaussian")

    assert tool._apply_filters_only() is True
    assert tool._Holo2_S1 is not None and tool._Holo2_S2 is not None

    diff = tool._CL_c - tool._CR_c
    assert np.allclose(tool._Holo2_S1, diff)
    assert np.allclose(tool._Holo2_S2, diff)
    tool.close()


def test_slit_mask_can_apply_each_direction_independently(qapp):
    tool = _prepare_tool_with_arrays(qapp)
    tool._phi1_spin.setValue(0.0)
    tool._phi2_spin.setValue(90.0)
    tool._slit_mask_width.setValue(0.0)
    tool._slit_mask_sigma.setValue(1.0)

    tool._g_slit_mask.setChecked(True)
    tool._slit_mask_phi1_chk.setChecked(True)
    tool._slit_mask_phi2_chk.setChecked(False)
    tool._apply_slit_mask()
    phi1_only = tool._slit_mask.copy()

    tool._slit_mask_phi1_chk.setChecked(False)
    tool._slit_mask_phi2_chk.setChecked(True)
    tool._apply_slit_mask()
    phi2_only = tool._slit_mask.copy()

    center = tool._X0
    assert phi1_only[center, 10] < 0.01
    assert phi1_only[10, center] > 0.99
    assert phi2_only[10, center] < 0.01
    assert phi2_only[center, 10] > 0.99
    tool.close()


def test_roi_count_1_to_4_enablement(qapp):
    tool = _prepare_tool_with_arrays(qapp)

    tool._on_roi_count_changed(1)
    assert tool._roi_count == 1
    assert tool._btn_roi2.isEnabled() is False
    assert tool._btn_roi3.isEnabled() is False
    assert tool._btn_roi4.isEnabled() is False

    tool._on_roi_count_changed(4)
    assert tool._roi_count == 4
    assert tool._btn_roi2.isEnabled() is True
    assert tool._btn_roi3.isEnabled() is True
    assert tool._btn_roi4.isEnabled() is True
    tool.close()


def test_lock_and_apply_params_roundtrip(qapp):
    tool = _prepare_tool_with_arrays(qapp)
    tool._phi1_spin.setValue(13.0)
    tool._phi2_spin.setValue(77.0)
    tool._slit_mask_width.setValue(4.0)
    tool._slit_mask_sigma.setValue(22.0)
    tool._roi_count_spin.setValue(3)
    tool._roi_size_slider.setValue(170)
    tool._slit_combo.setCurrentText("Slit 2")
    tool._filter_combo.setCurrentText("Binary")

    tool._lock_current_params()
    assert tool._locked_params is not None

    # Mutate values away from locked state.
    tool._phi1_spin.setValue(1.0)
    tool._phi2_spin.setValue(2.0)
    tool._slit_mask_width.setValue(1.0)
    tool._slit_mask_sigma.setValue(5.0)
    tool._roi_count_spin.setValue(1)
    tool._roi_size_slider.setValue(120)
    tool._slit_combo.setCurrentText("None")
    tool._filter_combo.setCurrentText("None")

    tool._apply_locked_params_to_current_data()
    assert abs(tool._phi1_spin.value() - 13.0) < 1e-9
    assert abs(tool._phi2_spin.value() - 77.0) < 1e-9
    assert abs(tool._slit_mask_width.value() - 4.0) < 1e-9
    assert abs(tool._slit_mask_sigma.value() - 22.0) < 1e-9
    assert tool._roi_count == 3
    assert tool._roi_size == 170
    assert tool._slit_combo.currentText() == "Slit 2"
    assert tool._filter_combo.currentText() == "Binary"
    tool.close()


def test_gray_colormap_is_available_for_fth(qapp):
    assert "gray" in FTH_COLORMAPS
    cmap = _get_colormap("gray")
    assert cmap is not None


# ---------------------------------------------------------------------------
# The auto phase fit was removed
# ---------------------------------------------------------------------------
#
# It minimised the imaginary energy of a window at the ROI centre, and it did
# that correctly — the residual came out at 0.000000. But rotating by pi flips
# the sign and leaves imaginary energy unchanged, so the criterion cannot tell
# phi from phi + pi and 0.5*atan2 could only ever return a value in
# (-pi/2, pi/2]. Measured over eight known phases, three came back with the
# contrast inverted and nothing on screen said so. For magnetic domains, where
# the plus and minus areas are about equal, no data-driven rule breaks that tie
# — so it could not be fixed, only removed. The manual phase rotation control
# does the same job with the result visible.

def test_the_auto_phase_fit_is_gone(qapp):
    tool = _prepare_tool_with_arrays(qapp)
    try:
        for name in ("_chk_phase_fit", "_t4_phase_fit_win", "_t4_phase_fit_label",
                     "_last_roi_phase_fit", "_estimate_roi_phase_rotation",
                     "_on_phase_fit_toggled"):
            assert not hasattr(tool, name), f"{name} came back"
    finally:
        tool.deleteLater()


def test_the_manual_phase_control_still_covers_a_whole_turn(qapp):
    """It is what the auto fit was standing in for, so it has to reach every
    phase — the removal is only defensible while this does."""
    tool = _prepare_tool_with_arrays(qapp)
    try:
        lo = np.pi * tool._t4_ph_slider.minimum() / 100.0
        hi = np.pi * tool._t4_ph_slider.maximum() / 100.0

        assert hi - lo >= 2 * np.pi
        assert lo < 0 < hi
    finally:
        tool.deleteLater()


def test_the_other_page_four_corrections_survived(qapp):
    """The fit was removed from the middle of the Corrections group."""
    tool = _prepare_tool_with_arrays(qapp)
    try:
        for name in ("_chk_inv_contrast", "_chk_inv_realimag",
                     "_chk_gauss_filter", "_t4_gauss_sigma"):
            assert hasattr(tool, name), f"{name} was removed by accident"
    finally:
        tool.deleteLater()


def test_page_four_still_builds_a_roi_and_displays_it(qapp):
    """_compute_roi held the fit call; the rest of it has to still run."""
    tool = _prepare_tool_with_arrays(qapp)
    try:
        yy, xx = np.mgrid[0:200, 0:200]
        obj = np.sign(np.sin(xx / 5.0) * np.cos(yy / 7.0)).astype(float)
        tool._FTH_S1 = obj * np.exp(1j * 0.9)
        tool._Holo_S1 = np.abs(tool._FTH_S1)
        tool._roi_centers[1][1] = (100, 100)

        roi = tool._compute_roi(1, 1)
        tool._update_t4_display()

        assert roi is not None
        assert roi.dtype.kind == "c"
        assert np.all(np.isfinite(roi))
    finally:
        tool.deleteLater()


def test_focus_is_opt_in_and_zero_distance_keeps_the_old_result(qapp):
    tool = _prepare_tool_with_arrays(qapp, n=64)
    try:
        rng = np.random.default_rng(17)
        holo = rng.normal(size=(tool._Nx, tool._Ny))
        tool._Holo2_S1 = holo.copy()
        tool._Holo2_S2 = holo.copy()

        assert tool._focus_group.isChecked() is False
        assert tool._compute_fth_only()
        old_result = tool._FTH_S1.copy()

        tool._focus_group.setChecked(True)
        tool._focus_update_timer.stop()
        assert tool._compute_fth_only(reset_display_scale=False)
        assert np.array_equal(tool._FTH_S1, old_result)
    finally:
        tool.deleteLater()


def test_focus_reconstructs_from_hologram_without_resetting_display(qapp):
    tool = _prepare_tool_with_arrays(qapp, n=64)
    try:
        rng = np.random.default_rng(19)
        holo = rng.normal(size=(tool._Nx, tool._Ny))
        tool._Holo2_S1 = holo.copy()
        tool._Holo2_S2 = (2.0 * holo).copy()
        assert tool._compute_fth_only()
        old_result = tool._FTH_S1.copy()

        tool._phase_scale_slider.setValue(73)
        tool._rs_scale_slider.setValue(165)
        tool._t4_ph_slider.setValue(-81)
        tool._t4_rs_slider.setValue(142)
        tool._t4_sliders_touched = True

        tool._focus_group.setChecked(True)
        tool._focus_distance.setValue(2.3)
        tool._focus_update_timer.stop()
        assert tool._compute_fth_only(reset_display_scale=False)

        assert not np.allclose(tool._FTH_S1, old_result)
        assert tool._phase_scale_slider.value() == 73
        assert tool._rs_scale_slider.value() == 165
        assert tool._t4_ph_slider.value() == -81
        assert tool._t4_rs_slider.value() == 142
        assert tool._t4_sliders_touched is True
    finally:
        tool.deleteLater()


def test_focus_parameters_are_part_of_lock_roundtrip(qapp):
    tool = _prepare_tool_with_arrays(qapp)
    try:
        tool._focus_group.setChecked(True)
        tool._focus_distance.setValue(-3.25)
        tool._focus_energy.setValue(852.7)
        tool._focus_detector_distance.setValue(245.0)
        tool._focus_pixel_size.setValue(13.5)
        tool._focus_quantize.setChecked(False)
        tool._focus_update_timer.stop()
        tool._lock_current_params()

        tool._focus_group.setChecked(False)
        tool._focus_distance.setValue(0.0)
        tool._focus_energy.setValue(500.0)
        tool._focus_detector_distance.setValue(100.0)
        tool._focus_pixel_size.setValue(20.0)
        tool._focus_quantize.setChecked(True)
        tool._focus_update_timer.stop()

        tool._apply_locked_params_to_current_data()
        assert tool._focus_group.isChecked() is True
        assert tool._focus_distance.value() == -3.25
        assert tool._focus_energy.value() == 852.7
        assert tool._focus_detector_distance.value() == 245.0
        assert tool._focus_pixel_size.value() == 13.5
        assert tool._focus_quantize.isChecked() is False
    finally:
        tool.deleteLater()

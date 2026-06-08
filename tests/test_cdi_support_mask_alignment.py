from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QGroupBox

from src.gui.cdi_reconstruction_tool import CDIReconstructionTool


def _prepare_cdi_with_raw_fth(qapp, shape=(1228, 1228)) -> CDIReconstructionTool:
    tool = CDIReconstructionTool(opened_files=())
    fth = tool._fth_tool
    fth._CL = np.ones(shape, dtype=np.float64)
    fth._CR = np.ones(shape, dtype=np.float64)
    fth._initialize_center_controls_for_loaded_shape(shape)
    fth._compute_centered_hologram()
    return tool


def test_cdi_sync_accepts_single_fth_dataset(qapp):
    tool = CDIReconstructionTool(opened_files=())
    fth = tool._fth_tool
    data = np.full((128, 128), 9.0, dtype=np.float64)
    fth._CL = data
    fth._CR = np.zeros_like(data)
    fth._single_dataset_mode = True
    fth._initialize_center_controls_for_loaded_shape(data.shape)
    fth._compute_centered_hologram()

    assert tool._sync_from_fth_tool() is True
    assert tool._single_dataset_mode is True
    assert tool._amp_cl is not None
    assert tool._amp_cr is not None
    np.testing.assert_allclose(tool._resolve_amplitude_for_source("CL+CR"), tool._amp_cl)
    np.testing.assert_allclose(tool._resolve_amplitude_for_source("CL-CR"), tool._amp_cl)
    assert tool._resolve_amplitude_for_source("CR") is None
    tool.close()


def test_loaded_support_mask_is_cropped_like_default_fth_center(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    raw = np.arange(1228 * 1228, dtype=np.float32).reshape(1228, 1228)

    aligned = tool._align_support_mask_to_current_fth(raw)

    assert aligned.shape == tool._fth_tool._CL_c.shape == (1226, 1226)
    assert aligned[0, 0] == raw[1, 1]
    tool.close()


def test_loaded_support_mask_follows_changed_fth_center(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    fth = tool._fth_tool
    fth._t1_xmid.setValue(600)
    fth._t1_ymid.setValue(610)
    fth._compute_centered_hologram()
    raw = np.arange(1228 * 1228, dtype=np.float32).reshape(1228, 1228)

    aligned = tool._align_support_mask_to_current_fth(raw)

    assert aligned.shape == fth._CL_c.shape == (1200, 1200)
    assert aligned[0, 0] == raw[0, 10]
    tool.close()


def test_loaded_support_mask_is_shown_as_transparent_overlay(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[100:200, 120:220] = 1.0

    tool._support_source_mask = raw
    tool._refresh_support_mask_overlay()

    overlay = tool._supp_mask_img.image
    assert tool._supp_mask_img.isVisible()
    assert overlay.shape == (1226, 1226, 4)
    assert overlay.dtype == np.uint8
    assert int(overlay[..., 3].max()) == 95
    assert int(overlay[..., 3].min()) == 0
    tool.close()


def test_rect_support_mask_uses_roi_rotation(qapp):
    tool = CDIReconstructionTool(opened_files=())
    fth = tool._fth_tool
    fth._CL = np.ones((128, 128), dtype=np.float64)
    fth._CR = np.ones((128, 128), dtype=np.float64)
    fth._t1_xmid.setValue(64)
    fth._t1_ymid.setValue(64)
    fth._compute_centered_hologram()

    tool._add_rect(y=64, x=64, w=40, h=10)
    entry = tool._entries[-1]
    entry.set_values(64, 64, 40, 10, 45)
    mask = tool._compute_support_mask()

    assert round(entry.roi.angle(), 1) == 45.0
    assert round(entry.angle, 1) == 45.0
    assert mask[64, 64] == 1
    assert mask[64, 85] == 0
    tool.close()


def test_bad_pixel_mask_is_cropped_like_fth_center(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[1, 1] = 1.0
    raw[10, 20] = 1.0

    tool._bad_pixel_mask_source = raw
    aligned = tool._current_bad_pixel_mask()

    assert aligned.shape == (1226, 1226)
    assert int(aligned.sum()) == 2
    assert aligned[0, 0]
    tool.close()


def test_bad_pixel_mask_follows_changed_holography_center(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    fth = tool._fth_tool
    fth._t1_xmid.setValue(600)
    fth._t1_ymid.setValue(610)
    fth._compute_centered_hologram()
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[0, 10] = 1.0

    tool._bad_pixel_mask_source = raw
    aligned = tool._current_bad_pixel_mask()

    assert aligned.shape == fth._CL_c.shape == (1200, 1200)
    assert int(aligned.sum()) == 1
    assert aligned[0, 0]
    tool.close()


def test_bad_pixel_mask_controls_and_overlay_are_on_alignment_page(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    groups = tool._tabs.widget(0).findChildren(QGroupBox)
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[1, 1] = 1.0
    raw[20, 30] = 1.0

    tool._bad_pixel_mask_source = raw
    tool._refresh_bad_pixel_mask_overlay()

    overlay = tool._bad_pixel_mask_img.image
    assert any(group.title() == "Bad Pixel Mask Editor" for group in groups)
    assert tool._bad_pixel_mask_img.isVisible()
    assert overlay.shape == (1226, 1226, 4)
    assert overlay.dtype == np.uint8
    assert int(overlay[..., 3].max()) == 120
    assert int(overlay[..., 3].min()) == 0
    assert int(overlay[..., 0].max()) == 255
    tool.close()


def test_bad_pixel_mask_overlay_matches_image_after_apply_center(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    fth = tool._fth_tool
    fth._t1_xmid.setValue(600)
    fth._t1_ymid.setValue(610)
    fth._on_center_changed()
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[0, 10] = 1.0

    tool._bad_pixel_mask_source = raw
    tool._refresh_bad_pixel_mask_overlay()

    overlay = tool._bad_pixel_mask_img.image
    assert overlay.shape[:2] == fth._t1_value_data.shape == (1200, 1200)
    assert tool._bad_pixel_mask_img.pos() == fth._t1_main_img.pos()
    assert tool._bad_pixel_mask_img.transform() == fth._t1_main_img.transform()
    assert bool(tool._bad_pixel_mask[0, 0])
    tool.close()


def test_bad_pixel_mask_overlay_ignores_stale_cdi_shape_after_apply_center(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    fth = tool._fth_tool
    tool._amp_meas = np.zeros((1226, 1226), dtype=np.float32)
    fth._t1_xmid.setValue(600)
    fth._t1_ymid.setValue(610)
    fth._on_center_changed()
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[0, 10] = 1.0

    tool._bad_pixel_mask_source = raw
    tool._refresh_bad_pixel_mask_overlay()

    assert tool._bad_pixel_mask is not None
    assert tool._bad_pixel_mask.shape == fth._t1_value_data.shape == (1200, 1200)
    assert tool._bad_pixel_mask_img.image.shape[:2] == (1200, 1200)
    assert bool(tool._bad_pixel_mask[0, 0])
    tool.close()


def test_bad_pixel_mask_is_merged_into_cdi_pixel_mask(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[1, 1] = 1.0
    tool._bad_pixel_mask_source = raw

    ok = tool._sync_from_fth_tool()

    assert ok
    assert tool._pixel_mask is not None
    assert tool._pixel_mask.shape == (1226, 1226)
    assert int(tool._pixel_mask.sum()) == 1
    assert tool._pixel_mask[0, 0]
    tool.close()


def test_bad_pixel_mask_shift_moves_overlay_and_floating_mask(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp)
    raw = np.zeros((1228, 1228), dtype=np.float32)
    raw[1, 1] = 1.0

    tool._bad_pixel_mask_source = raw
    tool._bad_pixel_shift_x_spin.setValue(2)
    tool._bad_pixel_shift_y_spin.setValue(3)
    tool._refresh_bad_pixel_mask_overlay()
    ok = tool._sync_from_fth_tool()

    assert ok
    assert tool._bad_pixel_mask is not None
    assert not tool._bad_pixel_mask[0, 0]
    assert tool._bad_pixel_mask[3, 2]
    assert tool._bad_pixel_mask_img.image[3, 2, 3] == 120
    assert tool._pixel_mask is not None
    assert tool._pixel_mask[3, 2]
    tool.close()


def test_support_mask_shift_moves_group_mask(qapp):
    tool = _prepare_cdi_with_raw_fth(qapp, shape=(128, 128))
    target_shape = tool._fth_tool._CL_c.shape
    mask = np.zeros(target_shape, dtype=np.float32)
    mask[10, 12] = 1.0

    tool._add_mask_group()
    group = tool._mask_groups[0]
    group["source_masks"].append(mask)
    group["shift_x"] = 3
    group["shift_y"] = 2

    shifted = tool._compute_group_mask(0)

    assert shifted is not None
    assert shifted[12, 15] == 1.0
    assert shifted[10, 12] == 0.0
    tool.close()


def test_result_view_combo_can_show_real_part(qapp):
    tool = CDIReconstructionTool(opened_files=())
    result = np.array([[1 + 2j, -3 + 4j]], dtype=np.complex128)

    tool._result_obj = result
    tool._res_view_combo.setCurrentText("Real part")
    tool._refresh_results_display()

    np.testing.assert_allclose(tool._res_amp_img.image, np.real(result))
    tool.close()

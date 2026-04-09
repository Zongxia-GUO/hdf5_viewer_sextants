from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from src.gui.fth_reconstruction_tool import FTH_COLORMAPS, FTHReconstructionTool, _get_colormap


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


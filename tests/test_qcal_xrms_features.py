"""Tests for the XRMS-style features added to the Q calibration tool:
CL/CR/BG operation combine, and Ring/Sector/Circle ROIs with profiles.
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np

from src.gui.q_calibration_tool import QCalibrationTool


def _make_file(**datasets) -> str:
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "qcal.h5")
    with h5py.File(fp, "w") as f:
        for name, arr in datasets.items():
            f.create_dataset(name, data=np.asarray(arr, "float32"))
    return fp


def _tool(qapp, fp, **slots) -> QCalibrationTool:
    keys = [f"{fp}::/{n}" for n in slots.values()]
    t = QCalibrationTool(opened_files=(fp,), dataset_full_keys_2d=keys)
    for slot, name in slots.items():
        t._combos[slot].add_full_key(f"{fp}::/{name}", select=True)
    return t


def _setop(t, code):
    t._combo_operation.setCurrentIndex(t._combo_operation.findData(code))


def test_operation_difference(qapp):
    fp = _make_file(cl=np.full((20, 20), 10.0), cr=np.full((20, 20), 6.0),
                    bg=np.full((20, 20), 1.0))
    t = _tool(qapp, fp, CL="cl", CR="cr", BG="bg")
    _setop(t, "diff")
    t._load_data()
    assert abs(float(np.nanmean(t._data)) - 4.0) < 1e-4  # (10-1)-(6-1)


def test_operation_cl_bg_averages_stack(qapp):
    fp = _make_file(cl3d=np.stack([np.full((20, 20), 8.0)] * 4), bg=np.full((20, 20), 1.0))
    t = _tool(qapp, fp, CL="cl3d", BG="bg")
    _setop(t, "cl_bg")
    t._load_data()
    assert t._data.shape == (20, 20)
    assert abs(float(np.nanmean(t._data)) - 7.0) < 1e-4


def test_ring_radial_profile_recovers_radius(qapp):
    yy, xx = np.indices((101, 101))
    rad = np.sqrt((xx - 50) ** 2 + (yy - 50) ** 2)
    fp = _make_file(cl=rad)
    t = _tool(qapp, fp, CL="cl")
    _setop(t, "cl_bg")
    t._load_data()
    assert t._center_pixel() == (50.0, 50.0)
    t._add_roi("ring")
    roi = t._selected_roi()
    roi["r_inner"], roi["r_outer"] = 5, 40
    t._sync_param_sliders_from_roi(roi)
    t._compute_current_profiles()
    x, y = t._profile_curve.getData()
    sel = (x >= 10) & (x <= 35)
    assert sel.any() and np.allclose(x[sel], y[sel], atol=1.5)


def test_sector_has_drag_handles_and_azimuthal(qapp):
    fp = _make_file(cl=np.ones((101, 101)))
    t = _tool(qapp, fp, CL="cl")
    _setop(t, "cl_bg")
    t._load_data()
    t._add_roi("sector")
    roi = t._selected_roi()
    roi["r_inner"], roi["r_outer"], roi["a_min"], roi["a_max"] = 10, 40, 0, 90
    t._sync_param_sliders_from_roi(roi)
    assert set(t._roi_handles) == {"r_inner", "r_outer", "a_min", "a_max"}
    t._combo_roi_mode.setCurrentIndex(1)  # azimuthal
    t._compute_current_profiles()
    x, _ = t._profile_curve.getData()
    assert len(x) > 0


def test_circle_roi_creates_interactive_item(qapp):
    fp = _make_file(cl=np.ones((60, 60)))
    t = _tool(qapp, fp, CL="cl")
    _setop(t, "cl_bg")
    t._load_data()
    t._add_roi("circle")
    roi = t._selected_roi()
    assert roi["type"] == "circle" and roi["item"] is not None
    t._compute_current_profiles()
    x, _ = t._profile_curve.getData()
    assert len(x) > 0


def test_incidence_resamples_so_roi_uses_corrected_image(qapp):
    # X-axis incidence at 30 deg stretches columns by 1/sin30 = 2.
    fp = _make_file(cl=np.ones((100, 100)))
    t = _tool(qapp, fp, CL="cl")
    _setop(t, "cl_bg")
    t._load_data()
    assert t._data is t._raw_data and t._data.shape == (100, 100)
    t._spin_inc_deg.setValue(30.0)
    t._combo_inc_axis.setCurrentText("X")
    t._apply_incident_cali()
    # working image is resampled; raw kept; ROIs read the corrected array.
    assert t._data.shape == (100, 200)
    assert t._raw_data.shape == (100, 100)
    assert t._incident_applied is True
    assert t._center_pixel() == (100.0, 50.0)  # center col scaled x2
    # q-math must not re-apply incidence (it's baked into the resampled data).
    p = t._collect_q_params()
    assert p["use_incidence"] is False and p["incidence_applied_in_display"] is False
    # re-applying resamples from raw (not cumulative)
    t._apply_incident_cali()
    assert t._data.shape == (100, 200)


def test_azimuthal_phase_shift(qapp):
    yy, xx = np.indices((101, 101))
    ang = np.degrees(np.arctan2(yy - 50, xx - 50)) % 360
    fp = _make_file(cl=ang)
    t = _tool(qapp, fp, CL="cl")
    _setop(t, "cl_bg")
    t._load_data()
    t._add_roi("sector")
    roi = t._selected_roi()
    roi.update(r_inner=10, r_outer=45, a_min=0, a_max=360, mode="azimuthal", phase=0)
    t._sync_param_sliders_from_roi(roi)
    t._combo_roi_mode.setCurrentIndex(1)  # azimuthal -> phase control visible
    assert t._cap_phase.isVisibleTo(t._roi_params) is True
    # roll helper is bin-count independent and direction-correct
    ay = np.arange(360, dtype=float)
    assert np.allclose(t._apply_phase_shift(ay, 30), np.roll(ay, 30))
    assert np.allclose(t._apply_phase_shift(ay, -20), np.roll(ay, -20))
    assert np.allclose(t._apply_phase_shift(np.arange(180.0), 30), np.roll(np.arange(180.0), 15))
    # changing phase shifts the plotted curve
    t._compute_current_profiles()
    _, y0 = t._profile_curve.getData()
    t._spin_phase.setValue(45)
    t._compute_current_profiles()
    _, y1 = t._profile_curve.getData()
    assert not np.array_equal(y0, y1)
    # radial mode hides the phase control
    t._combo_roi_mode.setCurrentIndex(0)
    assert t._cap_phase.isVisibleTo(t._roi_params) is False


def test_new_data_clears_rois(qapp):
    fp = _make_file(cl=np.ones((40, 40)))
    t = _tool(qapp, fp, CL="cl")
    _setop(t, "cl_bg")
    t._load_data()
    t._add_roi("ring")
    assert len(t._rois) == 1
    t._load_data()  # reload -> new geometry
    assert t._rois == []

"""Tests for the Time Resolved XRMS data-input combine pipeline.

Delay and norm are read per-frame from each stack's own data files (CL and CR
each carry their own delays), so CL↔CR are aligned by matching delay value and
the result is sorted by delay. CL/CR images come from real HDF5 files here;
backgrounds are injected directly (they need no delay).
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np

from src.gui.xrms_analyze_tool import XRMSAnalyzeTool

H = W = 8


def _tool(qapp) -> XRMSAnalyzeTool:
    return XRMSAnalyzeTool(opened_files=())


def _setop(tool, code: str) -> None:
    tool._combo_operation.setCurrentIndex(tool._combo_operation.findData(code))


def _means(tool) -> list[float]:
    return [float(tool._combined[i].mean()) for i in range(tool._combined.shape[0])]


def _make(name, values, delays, norms=None) -> str:
    """Write an HDF5 file with /img (a frame per value), /delay and optional /norm."""
    fp = os.path.join(tempfile.mkdtemp(), f"{name}.h5")
    with h5py.File(fp, "w") as f:
        f.create_dataset("img", data=np.stack(
            [np.full((H, W), float(v)) for v in values]).astype("float32"))
        f.create_dataset("delay", data=np.asarray(delays, float))
        if norms is not None:
            f.create_dataset("norm", data=np.asarray(norms, float))
    return fp


def test_sort_by_delay_value(qapp):
    # delays out of order; frame value = 10 + delay
    fp = _make("cl", values=[12, 10, 11], delays=[2, 0, 1])
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{fp}::/img"])
    t._slots["CL_BG"] = np.full((H, W), 1.0, "float32")
    t._delay_addr.setText("/delay")
    _setop(t, "cl")
    t._recompute_stack()
    assert np.allclose(t._combined_delay, [0, 1, 2])
    assert np.allclose(_means(t), [9, 10, 11])  # (10 + d) - bg, sorted by delay


def test_value_match_with_reversed_cr_order(qapp):
    # CR frames in a different delay order from CL -> must pair by value, then sort.
    clf = _make("cl", values=[10, 11, 12], delays=[0, 1, 2])
    crf = _make("cr", values=[7, 5, 6], delays=[2, 0, 1])  # value 5+delay
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{clf}::/img"])
    t._slot_combo["CR"].set_entries([f"{crf}::/img"])
    t._delay_addr.setText("/delay")
    _setop(t, "diff")
    t._recompute_stack()
    assert np.allclose(t._combined_delay, [0, 1, 2])
    # per delay d: cl=10+d, cr=5+d -> diff = 5
    assert np.allclose(_means(t), 5.0, atol=1e-4)


def test_diff_with_background(qapp):
    clf = _make("cl", values=[10, 11, 12], delays=[0, 1, 2])
    crf = _make("cr", values=[5, 6, 7], delays=[0, 1, 2])
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{clf}::/img"])
    t._slot_combo["CR"].set_entries([f"{crf}::/img"])
    t._slots["CL_BG"] = np.full((H, W), 1.0, "float32")
    t._slots["CR_BG"] = np.full((H, W), 0.5, "float32")
    t._delay_addr.setText("/delay")
    _setop(t, "diff")
    t._recompute_stack()
    # (10 + d - 1) - (5 + d - 0.5) = 4.5 for every delay
    assert np.allclose(_means(t), 4.5, atol=1e-4)


def test_intersection_on_differing_counts(qapp):
    clf = _make("cl", values=[10, 11, 12], delays=[0, 1, 2])
    crf = _make("cr", values=[5, 6], delays=[0, 1])  # missing delay 2
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{clf}::/img"])
    t._slot_combo["CR"].set_entries([f"{crf}::/img"])
    t._delay_addr.setText("/delay")
    _setop(t, "diff")
    t._recompute_stack()
    assert np.allclose(t._combined_delay, [0, 1])  # only common delays survive


def test_norm_divide_per_frame(qapp):
    fp = _make("cl", values=[10, 11, 12], delays=[0, 1, 2], norms=[2, 4, 5])
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{fp}::/img"])
    t._slots["CL_BG"] = np.full((H, W), 1.0, "float32")
    t._delay_addr.setText("/delay")
    t._norm_addr.setText("/norm")
    t._chk_norm.setChecked(True)
    _setop(t, "cl")
    t._recompute_stack()
    # (10+d-1)/norm -> 9/2, 10/4, 11/5
    assert np.allclose(_means(t), [9 / 2, 10 / 4, 11 / 5], atol=1e-4)


def test_no_delay_address_falls_back_to_index(qapp):
    fp = _make("cl", values=[10, 11, 12], delays=[0, 1, 2])
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{fp}::/img"])
    _setop(t, "cl")
    t._recompute_stack()  # no delay address -> frame index order, kept as-is
    assert np.allclose(t._combined_delay, [0, 1, 2])
    assert np.allclose(_means(t), [10, 11, 12])


def test_asymmetry(qapp):
    clf = _make("cl", values=[10, 11, 12], delays=[0, 1, 2])
    crf = _make("cr", values=[5, 6, 7], delays=[0, 1, 2])
    t = _tool(qapp)
    t._slot_combo["CL"].set_entries([f"{clf}::/img"])
    t._slot_combo["CR"].set_entries([f"{crf}::/img"])
    t._slots["CL_BG"] = np.full((H, W), 1.0, "float32")
    t._slots["CR_BG"] = np.full((H, W), 0.5, "float32")
    t._delay_addr.setText("/delay")
    _setop(t, "asym")
    t._recompute_stack()
    # cl=9+d, cr=4.5+d -> (cl-cr)/(cl+cr) = 4.5 / (13.5 + 2d)
    expected = [4.5 / (13.5 + 2 * d) for d in (0, 1, 2)]
    assert np.allclose(_means(t), expected, atol=1e-4)


def test_azimuthal_phase_shift_rolls_angular_profile(qapp):
    t = _tool(qapp)
    yy, xx = np.indices((101, 101))
    ang = (np.degrees(np.arctan2(yy - 50, xx - 50)) % 360).astype("float32")
    t._slots["CL"] = ang[None]  # single frame, delay defaults to index [0]
    _setop(t, "cl")
    t._recompute_stack()
    t._add_roi("sector")
    roi = t._selected_roi()
    roi.update(r_inner=10, r_outer=45, a_min=0, a_max=360, mode="angular", phase=0)
    t._combo_roi_mode.setCurrentIndex(1)
    t._update_profile_plots()
    y0 = t._angular_y.copy()
    roi["phase"] = 40
    t._update_profile_plots()
    assert np.allclose(np.roll(y0, 40), t._angular_y, equal_nan=True)
    # phase control visibility follows azimuthal mode
    t._on_roi_combo_changed()
    assert t._cap_phase.isVisibleTo(t._roi_params) is True
    roi["mode"] = "radial"
    t._combo_roi_mode.setCurrentIndex(0)
    assert t._cap_phase.isVisibleTo(t._roi_params) is False

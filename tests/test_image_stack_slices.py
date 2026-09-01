"""A stack of frames keeps the axis and the frame that were chosen.

The viewer lets both be picked — a combo for the axis, a slider for the frame —
and two things downstream ignored that:

* the export dialog previewed frame 0 of axis 0 and wrote the whole stack the
  same way, so pictures came out cut the wrong way;
* the Q button re-read the dataset from the file and averaged **every** frame
  together, so the pattern analysed belonged to no single measurement.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.gui.batch_export import (
    BatchExportDialog,
    export_image_dataset,
    slice_axis_from,
    slice_count,
    take_slice,
)

# Distinct along every axis, so a frame cut the wrong way is obvious.
STACK = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)


# ---------------------------------------------------------------------------
# Slicing along a chosen axis
# ---------------------------------------------------------------------------

def test_a_frame_is_taken_along_the_chosen_axis():
    np.testing.assert_array_equal(take_slice(STACK, 0, 1), STACK[1])
    np.testing.assert_array_equal(take_slice(STACK, 1, 2), STACK[:, 2])
    np.testing.assert_array_equal(take_slice(STACK, 2, 3), STACK[:, :, 3])


def test_the_frame_shape_follows_the_axis():
    assert take_slice(STACK, 0, 0).shape == (4, 5)
    assert take_slice(STACK, 1, 0).shape == (3, 5)
    assert take_slice(STACK, 2, 0).shape == (3, 4)


def test_the_count_follows_the_axis():
    assert slice_count(STACK, 0) == 3
    assert slice_count(STACK, 1) == 4
    assert slice_count(STACK, 2) == 5


def test_an_index_past_the_end_is_clamped():
    np.testing.assert_array_equal(take_slice(STACK, 0, 99), STACK[-1])
    np.testing.assert_array_equal(take_slice(STACK, 0, -5), STACK[0])


def test_a_plain_image_is_returned_whole():
    frame = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(take_slice(frame, 0, 2), frame)
    assert slice_count(frame, 0) == 1


@pytest.mark.parametrize("settings, axis", [
    ({}, 0),
    ({"slice_axis": 2}, 2),
    ({"slice_axis": None}, 0),
    ({"slice_axis": "1"}, 1),
    ({"slice_axis": "nonsense"}, 0),
])
def test_the_axis_setting_is_read_forgivingly(settings, axis):
    assert slice_axis_from(settings) == axis


# ---------------------------------------------------------------------------
# Writing every frame
# ---------------------------------------------------------------------------

def _write(tmp_path, axis):
    settings = {
        "output_dir": str(tmp_path),
        "colormap": "viridis",
        "invert": False,
        "scale": "Linear",
        "levels": None,
        "contrast": "Per frame",
        "save_tiff": False,
        "image_format": "PNG",
        "image_rotation": "0°",
        "slice_axis": axis,
    }
    export_image_dataset(pathlib.Path("scanx_0340.h5"), "entry/data", STACK, settings)
    return sorted(p.name for p in tmp_path.glob("*.png"))


def test_every_frame_of_the_first_axis_is_written(tmp_path):
    assert len(_write(tmp_path, 0)) == 3


def test_every_frame_of_a_chosen_axis_is_written(tmp_path):
    """Along axis 2 there are five frames, not three."""
    assert len(_write(tmp_path, 2)) == 5


def test_the_written_frames_are_cut_the_way_the_preview_showed(tmp_path):
    from PIL import Image

    _write(tmp_path, 1)
    first = np.asarray(Image.open(tmp_path / "scanx_0340_entry_data_slice0000.png"))

    # Axis 1 gives 3x5 frames; axis 0 would have given 4x5.
    assert first.shape[:2] == take_slice(STACK, 1, 0).shape


def test_a_single_frame_stack_is_written_once(tmp_path):
    settings = {
        "output_dir": str(tmp_path), "colormap": "viridis", "invert": False,
        "scale": "Linear", "levels": None, "contrast": "Per frame",
        "save_tiff": False, "image_format": "PNG", "image_rotation": "0°",
    }
    export_image_dataset(
        pathlib.Path("scanx_0340.h5"), "entry/data", STACK[:1], settings
    )

    assert len(list(tmp_path.glob("*.png"))) == 1


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def _dialog(qapp, tmp_path, data, **kwargs):
    return BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0340"],
        dataset_path="entry/data",
        sample_data=data,
        data_kind="image",
        preview_x_loader=lambda *a, **k: None,
        **kwargs,
    )


def test_a_stack_gets_axis_and_slice_controls(qapp, tmp_path):
    dialog = _dialog(qapp, tmp_path, STACK)
    try:
        assert dialog._is_stack is True
        assert dialog.cb_slice_axis.count() == 3
        assert dialog.sl_slice.maximum() == 2
    finally:
        dialog.deleteLater()


def test_a_plain_image_gets_none(qapp, tmp_path):
    """A single frame has no axis to choose."""
    dialog = _dialog(qapp, tmp_path, np.arange(20.0).reshape(4, 5))
    try:
        assert dialog._is_stack is False
        assert dialog.slice_axis() == 0
        assert dialog.settings()["slice_axis"] == 0
    finally:
        dialog.deleteLater()


def test_the_dialog_opens_where_the_viewer_left_off(qapp, tmp_path):
    """Otherwise it starts again at frame 0 of axis 0, which is not what was
    on screen when Export was pressed."""
    dialog = _dialog(qapp, tmp_path, STACK, slice_axis=2, slice_index=3)
    try:
        assert dialog.slice_axis() == 2
        assert dialog.sl_slice.value() == 3
        assert dialog.settings()["slice_axis"] == 2
    finally:
        dialog.deleteLater()


def test_choosing_an_axis_resets_the_slider_to_that_axis_length(qapp, tmp_path):
    dialog = _dialog(qapp, tmp_path, STACK)
    try:
        dialog.cb_slice_axis.setCurrentIndex(2)

        assert dialog.sl_slice.maximum() == 4
        assert dialog.sl_slice.value() == 0
        assert dialog.lbl_slice.text() == "1 / 5"
    finally:
        dialog.deleteLater()


def test_the_preview_follows_the_slider(qapp, tmp_path):
    dialog = _dialog(qapp, tmp_path, STACK)
    try:
        dialog.sl_slice.setValue(2)

        np.testing.assert_array_equal(dialog._preview_image_frame(), STACK[2])
    finally:
        dialog.deleteLater()


def test_the_preview_follows_the_axis(qapp, tmp_path):
    dialog = _dialog(qapp, tmp_path, STACK)
    try:
        dialog.cb_slice_axis.setCurrentIndex(1)
        dialog.sl_slice.setValue(3)

        np.testing.assert_array_equal(dialog._preview_image_frame(), STACK[:, 3])
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# The Q button: the pattern tool gets the frame that is on screen
# ---------------------------------------------------------------------------

@pytest.fixture
def viewer_window(qapp, tmp_path):
    """A main window showing a stack, with the q tool's loaders recorded."""
    import h5py

    from src.gui.main_window import MainWindow

    path = tmp_path / "scanx_0340.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/data", data=STACK)

    win = MainWindow()
    win._open_file(path)
    win._show_data(STACK, "Array2D", source_dataset_key=f"{path}::entry/data")
    yield win
    win.close()


def _record_q_calls(window, monkeypatch):
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        type(window), "open_q_tool_for_array",
        lambda self, arr, source_label="": calls.update(array=np.asarray(arr), label=source_label) or True,
    )
    monkeypatch.setattr(
        type(window), "_open_q_tool_for_key",
        lambda self, key: calls.update(key=key),
    )
    return calls


def test_the_q_button_sends_the_displayed_frame(viewer_window, monkeypatch):
    """It used to send the dataset key, and the q tool then averaged every
    frame in the file — a pattern belonging to no single measurement."""
    image_view = viewer_window._current_image_view_2d()
    assert image_view is not None and image_view.data_3d is not None
    image_view.slider_slice.setValue(2)

    calls = _record_q_calls(viewer_window, monkeypatch)
    viewer_window._handle_q_request_from_viewer("scanx_0340.h5::entry/data")

    assert "key" not in calls, "the key path averages the stack"
    np.testing.assert_array_equal(calls["array"], STACK[2])


def test_the_q_button_follows_the_chosen_axis(viewer_window, monkeypatch):
    image_view = viewer_window._current_image_view_2d()
    image_view.combo_slice_axis.setCurrentIndex(1)
    image_view.slider_slice.setValue(3)

    calls = _record_q_calls(viewer_window, monkeypatch)
    viewer_window._handle_q_request_from_viewer("scanx_0340.h5::entry/data")

    np.testing.assert_array_equal(calls["array"], STACK[:, 3])


def test_the_label_says_which_frame_it_is(viewer_window, monkeypatch):
    image_view = viewer_window._current_image_view_2d()
    image_view.slider_slice.setValue(1)

    calls = _record_q_calls(viewer_window, monkeypatch)
    viewer_window._handle_q_request_from_viewer("scanx_0340.h5::entry/data")

    assert "axis 0" in calls["label"] and "slice 1" in calls["label"]


def test_a_plain_image_still_goes_by_key(qapp, tmp_path, monkeypatch):
    """Nothing to disambiguate there, and the key carries the provenance."""
    from src.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win._show_data(np.arange(20.0).reshape(4, 5), "Array2D", source_dataset_key="f.h5::img")
        calls = _record_q_calls(win, monkeypatch)

        win._handle_q_request_from_viewer("f.h5::img")

        assert calls.get("key") == "f.h5::img"
        assert "array" not in calls
    finally:
        win.close()


def test_the_slice_count_is_shown(qapp, tmp_path):
    dialog = _dialog(qapp, tmp_path, STACK)
    try:
        assert dialog.lbl_slice.text() == "1 / 3"
        dialog.sl_slice.setValue(2)
        assert dialog.lbl_slice.text() == "3 / 3"
    finally:
        dialog.deleteLater()

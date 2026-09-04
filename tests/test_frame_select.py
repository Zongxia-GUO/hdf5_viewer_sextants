"""Picking a frame out of a stack, the same way in every tool.

A 3-D dataset is a stack of frames. The Q calibration tool learned to ask which
axis counts them and which one to use; the FTH and CDI tools could not take a
stack at all — CDI said so, FTH crashed with "too many values to unpack
(expected 2, got 3)", which tells the user nothing about dimensions.

The question is asked by one widget now, and answered by one set of functions,
so the three windows cannot drift apart the way the axis naming already had:
the batch export said ``0 (128)`` where the Q tool said ``0``.
"""

import pathlib

import h5py
import numpy as np
import pytest

from src.gui.frame_select import FrameSelector
from src.lib_h5.stacks import (
    MEAN_OF_FRAMES,
    axis_label,
    describe_reduction,
    read_frame,
    reduce_stack,
)

STACK = np.arange(7 * 5 * 4, dtype=np.float64).reshape(7, 5, 4)
FLAT = np.arange(5 * 4, dtype=np.float64).reshape(5, 4)


@pytest.fixture
def h5(tmp_path):
    path = tmp_path / "Scan_ECL_050.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("entry/data")
        g.create_dataset("stack", data=STACK)
        g.create_dataset("flat", data=FLAT)
    return path


# ── The pure part ─────────────────────────────────────────────────────── #

def test_the_mean_is_the_mean_and_says_so():
    frame, note = reduce_stack(STACK, 0, MEAN_OF_FRAMES, "CL")

    np.testing.assert_array_equal(frame, STACK.mean(axis=0))
    assert note == "CL: mean of 7 frames (axis 0)"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_one_frame_is_taken_along_the_chosen_axis(axis):
    frame, note = reduce_stack(STACK, axis, 2)

    np.testing.assert_array_equal(frame, np.take(STACK, 2, axis=axis))
    assert note == f"frame 2 of {STACK.shape[axis]} (axis {axis})"


def test_a_2d_array_is_returned_untouched_and_unremarked():
    """Callers hand everything through this without checking first."""
    frame, note = reduce_stack(FLAT, 0, 3, "CL")

    assert frame is FLAT
    assert note == ""


def test_an_index_past_the_end_is_clamped_not_wrapped():
    """Wrapping would quietly hand back a different frame than the one asked."""
    frame, note = reduce_stack(STACK, 0, 999)

    np.testing.assert_array_equal(frame, STACK[6])
    assert "frame 6 of 7" in note


def test_the_note_can_be_built_from_a_shape_alone():
    """The lazy reader never holds the stack, so it describes from the shape."""
    for axis in (0, 1, 2):
        for index in (MEAN_OF_FRAMES, 2):
            assert (describe_reduction(STACK.shape, axis, index, "CL")
                    == reduce_stack(STACK, axis, index, "CL")[1])


def test_the_axis_is_named_with_its_length():
    assert axis_label(0, 128) == "0 (128)"


# ── The lazy read ─────────────────────────────────────────────────────── #

@pytest.mark.parametrize("axis,index", [(0, 3), (1, 2), (2, 1)])
def test_reading_one_frame_from_the_file_matches_slicing_in_memory(h5, axis, index):
    """52x faster on a real stack, so it has to give the identical answer."""
    with h5py.File(h5, "r") as f:
        got = read_frame(f["entry/data/stack"], axis, index)

    np.testing.assert_array_equal(got, np.take(STACK, index, axis=axis))


def test_reading_the_mean_from_the_file_matches_the_mean_in_memory(h5):
    with h5py.File(h5, "r") as f:
        got = read_frame(f["entry/data/stack"], 0, MEAN_OF_FRAMES)

    np.testing.assert_array_equal(got, STACK.mean(axis=0))


def test_reading_a_2d_dataset_returns_all_of_it(h5):
    with h5py.File(h5, "r") as f:
        got = read_frame(f["entry/data/flat"], 0, 3)

    np.testing.assert_array_equal(got, FLAT)


def test_the_lazy_read_does_not_pull_the_whole_stack(h5, monkeypatch):
    """The point of it: one frame is read, not the stack it came from."""
    read_sizes = []
    real = h5py.Dataset.__getitem__

    def spy(self, key):
        out = real(self, key)
        read_sizes.append(np.asarray(out).size)
        return out

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    with h5py.File(h5, "r") as f:
        read_frame(f["entry/data/stack"], 0, 3)

    assert read_sizes == [STACK[0].size], read_sizes


# ── The widget ────────────────────────────────────────────────────────── #

def test_the_selector_hides_itself_for_2d_data(qapp):
    sel = FrameSelector()
    try:
        sel.set_shape(FLAT.shape)

        assert sel.isHidden()
        # And still answers, so a caller never has to ask whether it is showing.
        assert sel.selection() == (0, MEAN_OF_FRAMES)
    finally:
        sel.deleteLater()


def test_the_selector_offers_every_axis_and_every_frame(qapp):
    sel = FrameSelector()
    try:
        sel.set_shape(STACK.shape)

        assert [sel.combo_axis.itemText(i) for i in range(sel.combo_axis.count())] == \
            ["0 (7)", "1 (5)", "2 (4)"]
        assert sel.combo_frame.itemText(0) == "Mean"
        assert sel.combo_frame.count() == 1 + 7
        assert sel.selection() == (0, MEAN_OF_FRAMES)
    finally:
        sel.deleteLater()


def test_changing_the_axis_rebuilds_the_frame_list(qapp):
    sel = FrameSelector()
    try:
        sel.set_shape(STACK.shape)
        sel.combo_axis.setCurrentIndex(2)

        assert sel.combo_frame.count() == 1 + 4
        assert sel.selection()[0] == 2
    finally:
        sel.deleteLater()


def test_the_selector_reports_a_change_so_the_tool_can_reload(qapp):
    sel = FrameSelector()
    try:
        sel.set_shape(STACK.shape)
        seen = []
        sel.changed.connect(lambda: seen.append(True))

        sel.combo_frame.setCurrentIndex(3)
        sel.combo_axis.setCurrentIndex(1)

        assert len(seen) == 2
    finally:
        sel.deleteLater()


def test_the_selector_reduces_with_its_own_selection(qapp):
    sel = FrameSelector()
    try:
        sel.set_shape(STACK.shape)
        sel.combo_frame.setCurrentIndex(1 + 4)

        frame, note = sel.reduce(STACK, "CL")

        np.testing.assert_array_equal(frame, STACK[4])
        assert note == "CL: frame 4 of 7 (axis 0)"
    finally:
        sel.deleteLater()


# ── The shared module is where the helpers live ───────────────────────── #

def test_the_batch_export_still_exposes_the_moved_helpers():
    """They moved so the reconstruction tools would not have to import the
    batch export to slice a stack; its own call sites are unchanged."""
    from src.gui import batch_export

    assert batch_export.take_slice is not None
    assert batch_export.axis_label(0, 8) == "0 (8)"


def test_nothing_defines_its_own_axis_naming_any_more():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "gui"
    q_tool = (src / "q_calibration_tool.py").read_text(encoding="utf-8")

    assert "addItem(str(axis), axis)" not in q_tool
    assert "FrameSelector" in q_tool

"""The FTH and CDI tools take a stack of frames.

Before this, feeding either one a 3-D dataset failed. CDI said so —
"CL must be 2D after squeeze" — while FTH had no check at all and crashed
somewhere downstream with "too many values to unpack (expected 2, got 3)",
which says nothing about dimensions to whoever is holding a 400-frame scan.

CDI's data input is the FTH tool's Alignment tab, which it takes over
wholesale, so both windows are served by the one selector added to FTH.
"""

import h5py
import numpy as np
import pytest

from src.gui.fth_reconstruction_tool import FTHReconstructionTool, _FTHWorker
from src.lib_h5.stacks import MEAN_OF_FRAMES

RNG = np.random.RandomState(0)
STACK = RNG.rand(7, 32, 32).astype(np.float32)
FLAT = RNG.rand(32, 32).astype(np.float32)
STACK_KEY = "entry/data/stack"
FLAT_KEY = "entry/data/flat"


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "Scan_ECL_050.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("entry/data")
        g.create_dataset("stack", data=STACK)
        g.create_dataset("flat", data=FLAT)
    return path


@pytest.fixture
def fth(qapp, scan):
    tool = FTHReconstructionTool(opened_files=(scan,), dataset_full_keys_2d=[])
    yield tool
    tool.deleteLater()


# ── The reader ────────────────────────────────────────────────────────── #

def test_the_reader_averages_a_stack_and_says_it_did(scan):
    arr, note = _FTHWorker._read_one(str(scan), STACK_KEY, 0, MEAN_OF_FRAMES, "CL")

    np.testing.assert_allclose(arr, STACK.mean(axis=0))
    assert note == "CL: mean of 7 frames (axis 0)"


@pytest.mark.parametrize("axis,index", [(0, 3), (1, 5), (2, 9)])
def test_the_reader_takes_the_frame_it_was_asked_for(scan, axis, index):
    arr, note = _FTHWorker._read_one(str(scan), STACK_KEY, axis, index, "CL")

    np.testing.assert_allclose(arr, np.take(STACK, index, axis=axis))
    assert f"frame {index}" in note and f"axis {axis}" in note


def test_a_2d_dataset_is_read_exactly_as_before(scan):
    """The whole point of the reduction being a no-op below 3-D: nothing that
    worked before may change."""
    arr, note = _FTHWorker._read_one(str(scan), FLAT_KEY, 0, MEAN_OF_FRAMES, "CL")

    np.testing.assert_array_equal(arr, FLAT.astype(np.float64))
    assert note == ""


# ── Reduce, then sum ──────────────────────────────────────────────────── #

def test_each_dataset_is_reduced_before_they_are_summed(scan):
    """The other order sums the stacks and picks a frame out of the total,
    which is a different quantity."""
    entry = (str(scan), STACK_KEY)
    worker = _FTHWorker([entry, entry], [(str(scan), FLAT_KEY)], None, 0, 3)
    got = {}
    worker.finished.connect(lambda cl, cr, dark: got.update(cl=cl, cr=cr))

    worker.run()

    np.testing.assert_allclose(got["cl"], STACK[3].astype(np.float64) * 2)
    assert worker.notes == ["CL[0]: frame 3 of 7 (axis 0)",
                            "CL[1]: frame 3 of 7 (axis 0)"]


def test_the_dark_frame_is_reduced_the_same_way(scan):
    entry = (str(scan), FLAT_KEY)
    worker = _FTHWorker([entry], [entry], (str(scan), STACK_KEY), 0, 2)
    got = {}
    worker.finished.connect(lambda cl, cr, dark: got.update(dark=dark))

    worker.run()

    np.testing.assert_allclose(got["dark"], STACK[2].astype(np.float64))
    assert "Dark: frame 2 of 7 (axis 0)" in worker.notes


def test_a_2d_load_reports_no_frame_note(scan):
    entry = (str(scan), FLAT_KEY)
    worker = _FTHWorker([entry], [entry], None, 0, MEAN_OF_FRAMES)

    worker.run()

    assert worker.notes == []


# ── The FTH window ────────────────────────────────────────────────────── #

def test_the_frame_controls_appear_only_for_a_stack(fth, scan):
    assert fth._frames_group.isVisibleTo(fth) is False

    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)
    assert fth._frames_group.isVisibleTo(fth) is True

    fth._cl_combo.add_full_key(f"{scan}::{FLAT_KEY}", select=True)
    assert fth._frames_group.isVisibleTo(fth) is False


def test_the_choice_is_offered_before_loading(fth, scan):
    """It is part of deciding what to load. A control that only appears after
    the data is in has already reduced the stack without being asked."""
    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert fth._CL is None                      # nothing loaded yet
    assert fth._stack_shape() == STACK.shape
    assert fth._frames.combo_frame.count() == 1 + 7


def test_the_axes_are_named_the_way_every_other_selector_names_them(fth, scan):
    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert [fth._frames.combo_axis.itemText(i)
            for i in range(fth._frames.combo_axis.count())] == ["0 (7)", "1 (32)", "2 (32)"]


def test_the_mean_is_the_default(fth, scan):
    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert fth._frames.selection() == (0, MEAN_OF_FRAMES)


def test_the_dark_slot_can_bring_up_the_controls_on_its_own(fth, scan):
    fth._dark_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert fth._stack_shape() == STACK.shape


# ── The CDI window, which takes the same tab ──────────────────────────── #

def test_the_cdi_window_gets_the_same_frame_controls(qapp, scan):
    """CDI has no loader of its own — it takes over FTH's Alignment tab, so the
    selector added there is the one it uses."""
    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    cdi = CDIReconstructionTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        inner = cdi._fth_tool
        assert inner._frames_group.window() is cdi
        assert inner._frames_group.isVisibleTo(cdi) is False

        inner._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

        assert inner._frames_group.isVisibleTo(cdi) is True
        assert inner._frames.combo_frame.count() == 1 + 7
    finally:
        cdi.deleteLater()


def test_a_cdi_mask_may_not_be_a_stack(qapp, scan):
    """A mask marks detector pixels, so a stack of them is not something the
    tool can average or pick from without guessing. Both loaders already say
    so; this pins it, since the frame selector deliberately does not apply."""
    import inspect

    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    for name in ("_load_bad_pixel_mask_npy", "_load_support_mask_npy"):
        source = inspect.getsource(getattr(CDIReconstructionTool, name))
        assert "ndim != 2" in source, name
        assert "must be 2D" in source, name


def test_a_singleton_axis_is_not_a_stack(qapp, tmp_path):
    """``(1, H, W)`` and ``(H, W, 1)`` are how a single frame gets stored, and
    squeezing them is what the tool always did. Treating them as stacks would
    put a selector with one entry in front of the user and add a note about a
    choice they never had — so the test is "more than two real axes", not
    "ndim >= 3"."""
    base = np.random.RandomState(1).rand(16, 16).astype(np.float32)
    path = tmp_path / "shapes.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("flat", data=base)
        f.create_dataset("lead1", data=base[None, :, :])
        f.create_dataset("trail1", data=base[:, :, None])

    for key in ("flat", "lead1", "trail1"):
        arr, note = _FTHWorker._read_one(str(path), key, 0, MEAN_OF_FRAMES, "CL")

        np.testing.assert_allclose(arr, base, err_msg=key)
        assert note == "", key

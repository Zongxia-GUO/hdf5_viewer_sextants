"""The FTH and CDI tools take a stack of frames.

Before this, feeding either one a 3-D dataset failed. CDI said so —
"CL must be 2D after squeeze" — while FTH had no check at all and crashed
somewhere downstream with "too many values to unpack (expected 2, got 3)",
which says nothing about dimensions to whoever is holding a 400-frame scan.

CDI's data input is the FTH tool's Alignment tab, which it takes over
wholesale. CL and CR share one selector; Dark has an independent selector.
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
    arr, note = _FTHWorker._read_one(str(scan), STACK_KEY, 0, MEAN_OF_FRAMES, label="CL")

    np.testing.assert_allclose(arr, STACK.mean(axis=0))
    assert note == "CL: mean of 7 frames (axis 0)"


def test_chunked_read_progress_does_not_change_the_result(scan, monkeypatch):
    monkeypatch.setattr(_FTHWorker, "_READ_CHUNK_BYTES", 1024)
    fractions = []

    arr, note = _FTHWorker._read_one(
        str(scan),
        STACK_KEY,
        0,
        MEAN_OF_FRAMES,
        label="CL",
        progress_callback=fractions.append,
    )

    np.testing.assert_allclose(arr, STACK.mean(axis=0))
    assert note == "CL: mean of 7 frames (axis 0)"
    assert len(fractions) > 1
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


@pytest.mark.parametrize("axis,index", [(0, 3), (1, 5), (2, 9)])
def test_the_reader_takes_the_frame_it_was_asked_for(scan, axis, index):
    arr, note = _FTHWorker._read_one(str(scan), STACK_KEY, axis, index, label="CL")

    np.testing.assert_allclose(arr, np.take(STACK, index, axis=axis))
    assert f"frame {index}" in note and f"axis {axis}" in note


def test_a_2d_dataset_is_read_exactly_as_before(scan):
    """The whole point of the reduction being a no-op below 3-D: nothing that
    worked before may change."""
    arr, note = _FTHWorker._read_one(str(scan), FLAT_KEY, 0, MEAN_OF_FRAMES, label="CL")

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


def test_dark_and_signal_frames_are_selected_independently(scan):
    stack_entry = (str(scan), STACK_KEY)
    flat_entry = (str(scan), FLAT_KEY)
    worker = _FTHWorker(
        [stack_entry],
        [flat_entry],
        stack_entry,
        0,
        1,
        5.0,
        0,
        4,
        5.0,
    )
    got = {}
    worker.finished.connect(
        lambda cl, cr, dark: got.update(cl=cl, cr=cr, dark=dark)
    )

    worker.run()

    np.testing.assert_allclose(got["cl"], STACK[1].astype(np.float64))
    np.testing.assert_allclose(got["dark"], STACK[4].astype(np.float64))
    assert "CL: frame 1 of 7 (axis 0)" in worker.notes
    assert "Dark: frame 4 of 7 (axis 0)" in worker.notes


def test_a_2d_load_reports_no_frame_note(scan):
    entry = (str(scan), FLAT_KEY)
    worker = _FTHWorker([entry], [entry], None, 0, MEAN_OF_FRAMES)

    worker.run()

    assert worker.notes == []


def test_worker_reports_monotonic_byte_weighted_progress(scan, monkeypatch):
    monkeypatch.setattr(_FTHWorker, "_READ_CHUNK_BYTES", 1024)
    stack_entry = (str(scan), STACK_KEY)
    flat_entry = (str(scan), FLAT_KEY)
    worker = _FTHWorker([stack_entry], [flat_entry], None, 0, MEAN_OF_FRAMES)
    events = []
    worker.progress.connect(lambda percent, stage: events.append((percent, stage)))

    worker.run()

    percentages = [percent for percent, _stage in events]
    assert percentages == sorted(percentages)
    assert events[0] == (0, "Preparing load")
    assert events[-1] == (100, "Finalizing load")
    assert any(stage == "Processing CL" for _percent, stage in events)
    assert any(stage == "Loading CR" for _percent, stage in events)


# ── The FTH window ────────────────────────────────────────────────────── #

def test_the_frame_controls_appear_only_for_a_stack(fth, scan):
    assert fth._frames_group.isVisibleTo(fth) is False
    assert fth._dark_frames_group.isVisibleTo(fth) is False

    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)
    assert fth._frames_group.isVisibleTo(fth) is True
    assert fth._dark_frames_group.isVisibleTo(fth) is False

    fth._cl_combo.add_full_key(f"{scan}::{FLAT_KEY}", select=True)
    assert fth._frames_group.isVisibleTo(fth) is False


def test_the_choice_is_offered_before_loading(fth, scan):
    """It is part of deciding what to load. A control that only appears after
    the data is in has already reduced the stack without being asked."""
    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert fth._CL is None                      # nothing loaded yet
    assert fth._stack_shape() == STACK.shape
    assert fth._frames.spin_frame.maximum() == 6


def test_the_axes_are_named_the_way_every_other_selector_names_them(fth, scan):
    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert [fth._frames.combo_axis.itemText(i)
            for i in range(fth._frames.combo_axis.count())] == ["0 (7)", "1 (32)", "2 (32)"]


def test_the_first_frame_is_the_default(fth, scan):
    """Not the mean. Computing something out of every frame changes the data in
    a way the user has to be able to see they asked for."""
    fth._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert fth._frames.combo_method.currentText() == "Single frame"
    assert fth._frames.selection() == (0, 0)


@pytest.mark.parametrize("method,expected", [
    ("Mean", lambda: STACK.mean(axis=0)),
    ("Sum", lambda: STACK.sum(axis=0)),
    ("Median", lambda: np.median(STACK, axis=0)),
])
def test_each_combination_reaches_the_reader(scan, method, expected):
    """The method travels to the worker as the frame index, so it has to come
    out the other end as the arithmetic it names."""
    from src.gui.frame_select import METHOD_CHOICES

    sentinel = dict((name, value) for name, value in METHOD_CHOICES)[method]
    arr, note = _FTHWorker._read_one(str(scan), STACK_KEY, 0, sentinel, label="CL")

    np.testing.assert_allclose(arr, expected(), rtol=1e-6)
    assert method.lower() in note


def test_the_clipped_mean_rejects_what_a_mean_would_average_in(scan, tmp_path):
    """The reason this method exists: one cosmic ray ruins a mean."""
    from src.lib_h5.stacks import CLIPPED_MEAN_OF_FRAMES

    clean = np.random.RandomState(2).poisson(50.0, (40, 16, 16)).astype(np.float32)
    dirty = clean.copy()
    dirty[7, 8, 8] += 50000.0
    path = tmp_path / "rays.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("stack", data=dirty)

    mean, _ = _FTHWorker._read_one(str(path), "stack", 0, MEAN_OF_FRAMES)
    clipped, note = _FTHWorker._read_one(str(path), "stack", 0, CLIPPED_MEAN_OF_FRAMES)
    truth = clean.mean(axis=0)[8, 8]

    assert mean[8, 8] > truth + 1000, "the mean should be wrecked"
    assert abs(clipped[8, 8] - truth) < 1.0, "the clipped mean should not be"
    assert "k=5" in note


def test_the_dark_slot_can_bring_up_the_controls_on_its_own(fth, scan):
    fth._dark_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

    assert fth._stack_shape() == STACK.shape
    assert fth._signal_stack_shape() is None
    assert fth._dark_stack_shape() == STACK.shape
    assert fth._frames_group.isVisibleTo(fth) is False
    assert fth._dark_frames_group.isVisibleTo(fth) is True
    assert fth._dark_frames.spin_frame.maximum() == 6


def test_fth_sidebars_use_page_specific_widths(fth):
    first_splitter = fth._tabs.widget(0).layout().itemAt(0).widget()
    third_splitter = fth._tabs.widget(2).layout().itemAt(0).widget()

    assert first_splitter.widget(0).minimumWidth() >= 440
    assert third_splitter.widget(0).minimumWidth() == 360

    focus_form = fth._focus_group.layout()
    slider_row, slider_role = focus_form.getWidgetPosition(fth._focus_distance_slider)
    value_row, value_role = focus_form.getWidgetPosition(fth._focus_distance)
    assert value_row == slider_row + 1
    assert slider_role == value_role


def test_signal_frames_are_between_signal_and_dark_paths(fth):
    splitter = fth._tabs.widget(0).layout().itemAt(0).widget()
    controls_layout = splitter.widget(0).widget().layout()
    titles = [
        controls_layout.itemAt(index).widget().title()
        for index in range(controls_layout.count())
        if controls_layout.itemAt(index).widget() is not None
        and hasattr(controls_layout.itemAt(index).widget(), "title")
    ]

    assert titles[:5] == [
        "CL Dataset  (circular left polarisation)",
        "CR Dataset  (optional for single-file mode)",
        "CL / CR Frames",
        "Dark Scan  (optional)",
        "Dark Frames",
    ]


# ── The CDI window, which takes the same tab ──────────────────────────── #

def test_the_cdi_window_gets_the_same_frame_controls(qapp, scan):
    """CDI has no loader of its own — it takes over FTH's Alignment tab, so the
    selector added there is the one it uses."""
    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    cdi = CDIReconstructionTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        inner = cdi._fth_tool
        assert inner._frames_group.window() is cdi
        assert inner._dark_frames_group.window() is cdi
        assert inner._frames_group.isVisibleTo(cdi) is False

        inner._cl_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

        assert inner._frames_group.isVisibleTo(cdi) is True
        assert inner._frames.spin_frame.maximum() == 6

        inner._dark_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

        assert inner._dark_frames_group.isVisibleTo(cdi) is True
        assert inner._dark_frames.spin_frame.maximum() == 6

        first_splitter = cdi._tabs.widget(0).layout().itemAt(0).widget()
        third_splitter = cdi._tabs.widget(2).layout().itemAt(0).widget()
        assert first_splitter.widget(0).minimumWidth() >= 440
        assert third_splitter.widget(0).minimumWidth() == 400
        assert third_splitter.widget(0).maximumWidth() == 410
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
        arr, note = _FTHWorker._read_one(str(path), key, 0, MEAN_OF_FRAMES, label="CL")

        np.testing.assert_allclose(arr, base, err_msg=key)
        assert note == "", key


# ── The chunked reader must be invisible in the result ────────────────── #
#
# Reading a dataset in slabs is a change to *how* the bytes arrive, so the one
# thing it must never do is change what they are. These check that across the
# dtypes a detector actually writes, every axis, and a chunk budget small
# enough to force many slabs.

CHUNK_CASES = {
    "f32": np.random.RandomState(1).rand(9, 40, 30).astype(np.float32),
    "f64": np.random.RandomState(2).rand(5, 20, 20).astype(np.float64),
    "u16": (np.random.RandomState(3).rand(7, 16, 24) * 60000).astype(np.uint16),
    "flat": np.random.RandomState(4).rand(32, 32).astype(np.float32),
    "lead1": np.random.RandomState(5).rand(1, 24, 24).astype(np.float32),
}


@pytest.fixture
def chunky(tmp_path):
    path = tmp_path / "chunky.h5"
    with h5py.File(path, "w") as f:
        for key, value in CHUNK_CASES.items():
            f.create_dataset(key, data=value)
    return path


@pytest.mark.parametrize("key", list(CHUNK_CASES))
def test_a_chunked_read_returns_the_whole_dataset_unchanged(chunky, key):
    want = CHUNK_CASES[key]
    with h5py.File(chunky, "r") as f:
        for axis in range(want.ndim):
            got = _FTHWorker._read_dataset_chunked(f[key], axis, None, None)

            np.testing.assert_array_equal(got, want, err_msg=f"{key} axis {axis}")
            assert got.dtype == want.dtype, f"{key} axis {axis}"


@pytest.mark.parametrize("key", ["f32", "f64", "u16"])
def test_a_chunked_frame_read_matches_taking_that_frame(chunky, key):
    want = CHUNK_CASES[key]
    with h5py.File(chunky, "r") as f:
        for axis in range(want.ndim):
            for index in (0, want.shape[axis] // 2, want.shape[axis] - 1):
                got = _FTHWorker._read_selected_frame_chunked(
                    f[key], axis, index, None, None)

                np.testing.assert_array_equal(
                    got, np.take(want, index, axis=axis),
                    err_msg=f"{key} axis {axis} index {index}")


def test_a_frame_index_past_the_end_is_clamped(chunky):
    """Wrapping would hand back a different frame than the one asked for."""
    with h5py.File(chunky, "r") as f:
        got = _FTHWorker._read_selected_frame_chunked(f["f32"], 0, 999, None, None)

    np.testing.assert_array_equal(got, CHUNK_CASES["f32"][-1])


def test_many_small_slabs_give_the_same_answer_as_one(chunky, monkeypatch):
    """The slab size is a memory budget, not part of the arithmetic."""
    monkeypatch.setattr(_FTHWorker, "_READ_CHUNK_BYTES", 64)
    with h5py.File(chunky, "r") as f:
        whole = _FTHWorker._read_dataset_chunked(f["f32"], 0, None, None)
        frame = _FTHWorker._read_selected_frame_chunked(f["f32"], 1, 3, None, None)

    np.testing.assert_array_equal(whole, CHUNK_CASES["f32"])
    np.testing.assert_array_equal(frame, np.take(CHUNK_CASES["f32"], 3, axis=1))


def test_a_read_can_be_interrupted_part_way(chunky, monkeypatch):
    """Reading in slabs is what makes a long load abandonable; without the
    check it would only be a memory bound."""
    monkeypatch.setattr(_FTHWorker, "_READ_CHUNK_BYTES", 64)
    checks = []

    def stop_after_two():
        checks.append(True)
        return len(checks) > 2

    with h5py.File(chunky, "r") as f:
        with pytest.raises(InterruptedError):
            _FTHWorker._read_dataset_chunked(f["f32"], 0, None, stop_after_two)

    assert len(checks) == 3, "it should stop at the first check that says so"


def test_the_dark_scan_has_its_own_frame_choice(qapp, scan):
    """A dark is often a single exposure where the data is a stack, so it
    cannot be forced to share the data's frame selector."""
    tool = FTHReconstructionTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        assert tool._dark_frames is not tool._frames
        assert tool._dark_frames_group.isVisibleTo(tool) is False

        tool._dark_combo.add_full_key(f"{scan}::{STACK_KEY}", select=True)

        assert tool._dark_frames_group.isVisibleTo(tool) is True
    finally:
        tool.deleteLater()

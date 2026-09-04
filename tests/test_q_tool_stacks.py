"""The pattern tool says what it did to a stack of frames.

Reducing a stack used to happen in silence: a 400-frame stack went in, the
status line read "Loaded (200x46)", and nothing said what the pattern on screen
was made of. The two ways into the tool also disagreed, one averaging and the
other taking frame zero, so the same stack gave two different patterns.

Both are now the same one control, and the default is a single frame rather
than the mean — computing something out of 400 frames is a large thing to do
to someone's data, and it should be asked for. The tests that exercise the mean
therefore select it, the way a user would.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.gui.q_calibration_tool import QCalibrationTool

# Each frame is constant and equal to its index, so the mean (199.5) and the
# first frame (0.0) cannot be confused.
STACK = np.stack([np.full((20, 12), i, dtype=np.float32) for i in range(400)])


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "Scan_ECL_5p0uJIR_050.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/signal/img", data=STACK)
        f.create_dataset("entry/signal/flat", data=np.ones((20, 12), dtype=np.float32))
    return path


@pytest.fixture
def tool(qapp, scan):
    widget = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    yield widget
    widget.close()
    widget.deleteLater()


# ---------------------------------------------------------------------------
# Reducing a stack
# ---------------------------------------------------------------------------

def _choose(tool, text):
    """Pick a combination method by its name in the frame box."""
    index = tool._frames.combo_method.findText(text)
    assert index >= 0, text
    tool._frames.combo_method.setCurrentIndex(index)


def test_a_stack_loaded_from_a_file_uses_the_first_frame_by_default(tool, scan):
    """The default computes nothing: frame 0, exactly as recorded."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    assert tool._data.shape == (20, 12)
    assert float(tool._data[0, 0]) == pytest.approx(0.0)
    assert "frame 0 of 400" in tool._status.toolTip()


def test_a_stack_loaded_from_a_file_is_averaged(tool, scan):
    _choose(tool, "Mean")
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    assert tool._data.shape == (20, 12)
    assert float(tool._data[0, 0]) == pytest.approx(199.5)


def test_the_status_line_says_the_frames_were_averaged(tool, scan):
    """Otherwise the only clue is a shape that no longer matches the dataset."""
    _choose(tool, "Mean")
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    assert "mean of 400 frames" in tool._status.toolTip()


def test_an_array_handed_over_is_reduced_the_same_way(tool):
    """It used to take frame zero here and the mean from a file, so the same
    stack gave two different patterns depending on how it arrived."""
    _choose(tool, "Mean")
    tool.load_array_data(STACK, source_label="from viewer")

    assert float(tool._data[0, 0]) == pytest.approx(199.5)
    assert "mean of 400 frames" in tool._status.toolTip()


def test_a_plain_image_says_nothing_extra(tool, scan):
    tool.load_dataset_full_key(f"{scan}::entry/signal/flat", auto_load=True, slot="CL")

    assert tool._data.shape == (20, 12)
    assert "mean of" not in tool._status.toolTip()


def test_frames_are_counted_along_the_first_axis(tool):
    """The old rule reshaped on the *last* two axes, which turned a stack
    stored as (H, W, N) — a layout the 2-D viewer supports — into nonsense."""
    stack = np.zeros((4, 6, 10), dtype=np.float32)
    stack[2] = 1.0
    tool._frames.set_shape(stack.shape)
    _choose(tool, "Mean")

    reduced = tool._flatten_stack(stack, "CL")

    assert reduced.shape == (6, 10), "4 frames of 6x10, not 4x6 frames of 10"
    assert float(reduced[0, 0]) == pytest.approx(0.25)


def test_the_note_is_cleared_between_loads(tool, scan):
    """A note left over from a stack would claim a plain image was averaged."""
    _choose(tool, "Mean")
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")
    assert "mean of" in tool._status.toolTip()

    tool.load_array_data(np.ones((8, 8), dtype=np.float32), source_label="plain")

    assert "mean of" not in tool._status.toolTip()


def test_a_two_dimensional_array_is_untouched(tool):
    frame = np.arange(24.0, dtype=np.float32).reshape(4, 6)

    np.testing.assert_array_equal(tool._flatten_stack(frame, "CL"), frame)


# ---------------------------------------------------------------------------
# Opening the tool on whatever the tree had selected
# ---------------------------------------------------------------------------

def test_a_group_is_not_loaded_and_does_not_raise_a_dialog(tool, scan):
    """The key comes from the tree selection, which is often a group. The tool
    used to try it anyway and open with "Dataset not found" about a path the
    user can plainly see."""
    ok = tool.load_dataset_full_key(f"{scan}::entry", auto_load=True, slot="CL")

    assert ok is False
    assert tool._data is None
    assert "is a group" in tool._status.toolTip()


def test_the_selection_is_still_filled_in(tool, scan):
    """It is a useful starting point even when it cannot be loaded; only the
    automatic load is held back."""
    tool.load_dataset_full_key(f"{scan}::entry", auto_load=True, slot="CL")

    assert tool._cl_combo.currentText() != ""


def test_a_one_dimensional_dataset_is_refused_with_a_reason(tool, scan, tmp_path):
    import h5py as _h5py

    path = tmp_path / "curve.h5"
    with _h5py.File(path, "w") as f:
        f.create_dataset("curve", data=np.arange(50.0))

    ok = tool.load_dataset_full_key(f"{path}::curve", auto_load=True, slot="CL")

    assert ok is False
    assert "needs an image" in tool._status.toolTip()


def test_a_missing_path_says_so(tool, scan):
    ok = tool.load_dataset_full_key(f"{scan}::no/such/thing", auto_load=True, slot="CL")

    assert ok is False
    assert "No such dataset" in tool._status.toolTip()


def test_a_real_image_still_loads(tool, scan):
    ok = tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    assert ok is True
    assert tool._data is not None


def test_a_group_reached_through_load_data_says_what_it_is(tool, scan):
    """Choosing a group by hand in the combo goes down a different path."""
    tool._cl_combo.add_full_key(f"{scan}::entry", select=True)

    with pytest.raises(KeyError, match="is a group"):
        tool._read_slot_2d("CL")


# ---------------------------------------------------------------------------
# The status line must not resize the window
# ---------------------------------------------------------------------------

def test_loading_a_stack_does_not_widen_the_settings_panel(qapp, scan):
    """A QLabel asks for the full width of its text, and these messages are
    unbounded — one note per loaded stack, or a saved path. Left free, the first
    "Loaded: … mean of 400 frames …" doubled the panel and squeezed the image
    beside it to a sliver."""
    from PyQt6.QtWidgets import QApplication, QSplitter

    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    tool.show()
    QApplication.processEvents()
    try:
        splitter = tool.findChild(QSplitter)
        before = splitter.sizes()[0]

        tool._cl_combo.add_full_key(f"{scan}::entry/signal/img", select=True)
        tool._cr_combo.add_full_key(f"{scan}::entry/signal/img", select=True)
        index = tool._frames.combo_method.findText("Mean")
        tool._frames.combo_method.setCurrentIndex(index)
        tool._load_data()
        QApplication.processEvents()

        assert "mean of 400 frames" in tool._status.toolTip(), "the message really is long"
        assert splitter.sizes()[0] == before
    finally:
        tool.close()
        tool.deleteLater()


def test_the_settings_panel_is_sized_by_the_splitter_not_its_contents(qapp, scan):
    """Whatever grows inside — a long dataset path, a status message, the frame
    selectors appearing — the panel keeps the width the splitter gave it."""
    from PyQt6.QtWidgets import QApplication, QSizePolicy, QSplitter

    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    tool.resize(1320, 820)
    tool.show()
    QApplication.processEvents()
    try:
        splitter = tool.findChild(QSplitter)
        panel = splitter.widget(0)

        assert panel.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
        assert panel.width() < panel.minimumSizeHint().width(), (
            "the panel is narrower than its contents ask for, which is the point"
        )
    finally:
        tool.close()
        tool.deleteLater()


def test_a_long_message_is_elided_but_kept_in_full_on_hover(qapp, scan):
    from src.gui.q_calibration_tool import STATUS_MAX_WIDTH, QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        message = "Saved image: " + "C:/a/very/deep/folder/" * 6 + "pattern.png"
        tool._set_status(message)

        shown = tool._status.text()
        assert shown != message, "not elided"
        assert tool._status.toolTip() == message, "the whole message is still reachable"
        width = tool._status.fontMetrics().horizontalAdvance(shown)
        assert width <= STATUS_MAX_WIDTH
    finally:
        tool.deleteLater()


def test_a_short_message_is_left_alone(qapp, scan):
    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        tool._set_status("Switched to pixel axes.")

        assert tool._status.text() == "Switched to pixel axes."
    finally:
        tool.deleteLater()


# ---------------------------------------------------------------------------
# Choosing the frame instead of taking whatever it gives you
# ---------------------------------------------------------------------------

def _settle(tool, timeout_ms=2000):
    """Wait for the selector's coalescing timer to fire.

    Editing the frame or k schedules a reload rather than doing one per click,
    because a clipped mean of a big stack takes about a second and the spin box
    arrows fire once per press.
    """
    from PyQt6.QtCore import QDeadlineTimer
    from PyQt6.QtWidgets import QApplication

    deadline = QDeadlineTimer(timeout_ms)
    while tool._frames._settle.isActive() and not deadline.hasExpired():
        QApplication.processEvents()
    QApplication.processEvents()


def test_a_plain_image_offers_no_frame_controls(tool, scan):
    tool.load_dataset_full_key(f"{scan}::entry/signal/flat", auto_load=True, slot="CL")

    assert tool._frames.isVisibleTo(tool) is False


def test_a_stack_brings_up_the_frame_controls(tool, scan):
    """Before loading: the choice has to be there while choosing what to load."""
    tool._cl_combo.add_full_key(f"{scan}::entry/signal/img", select=True)

    assert tool._frames.isVisibleTo(tool) is True
    # Each axis carries its length: the number is what the code slices by, and
    # the length is what tells the axes apart. Same naming as the batch export
    # and the 2-D viewer, which is why this moved to a shared widget.
    assert [tool._frames.combo_axis.itemText(i)
            for i in range(tool._frames.combo_axis.count())] == ["0 (400)", "1 (20)", "2 (12)"]


def test_the_methods_offered_are_the_same_everywhere(tool, scan):
    """A list of every frame would be 401 entries on this scan, which is no way
    to reach frame 287. The box names a method; the box beside it takes the
    frame number, or k, or nothing."""
    tool._cl_combo.add_full_key(f"{scan}::entry/signal/img", select=True)

    offered = [tool._frames.combo_method.itemText(i)
               for i in range(tool._frames.combo_method.count())]

    assert offered == ["Single frame", "Mean", "Sum", "Median", "Clipped mean"]


def test_the_first_frame_is_the_default(tool, scan):
    """Computing something out of 400 frames should be asked for, not assumed."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    assert float(tool._data[0, 0]) == pytest.approx(0.0)
    assert "frame 0 of 400" in tool._status.toolTip()


def test_one_frame_can_be_picked(tool, scan):
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    tool._frames.spin_frame.setValue(3)
    _settle(tool)

    assert float(tool._data[0, 0]) == pytest.approx(3.0)
    assert "frame 3 of 400" in tool._status.toolTip()


@pytest.mark.parametrize("method,expected", [
    ("Mean", 199.5),
    ("Sum", 199.5 * 400),
    ("Median", 199.5),
    ("Clipped mean", 199.5),
])
def test_each_combination_method_reaches_the_data(tool, scan, method, expected):
    """Every frame here is constant and equal to its index, so each method has
    a value that only it (or an equal one) can produce."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    _choose(tool, method)
    _settle(tool)

    assert float(tool._data[0, 0]) == pytest.approx(expected, rel=1e-6)
    assert method.lower() in tool._status.toolTip()


def test_the_rejection_threshold_only_shows_for_the_clipped_mean(tool, scan):
    tool._cl_combo.add_full_key(f"{scan}::entry/signal/img", select=True)

    _choose(tool, "Single frame")
    assert tool._frames.spin_frame.isVisibleTo(tool) is True
    assert tool._frames.spin_kappa.isVisibleTo(tool) is False

    _choose(tool, "Clipped mean")
    assert tool._frames.spin_frame.isVisibleTo(tool) is False
    assert tool._frames.spin_kappa.isVisibleTo(tool) is True

    _choose(tool, "Mean")
    assert tool._frames.spin_frame.isVisibleTo(tool) is False
    assert tool._frames.spin_kappa.isVisibleTo(tool) is False


def test_the_threshold_is_reported_with_the_clipped_mean(tool, scan):
    """k changes the answer, so a note that leaves it out is incomplete."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    _choose(tool, "Clipped mean")
    tool._frames.spin_kappa.setValue(3.0)
    _settle(tool)

    assert "k=3" in tool._status.toolTip()


def test_changing_the_frame_reloads_without_pressing_load(tool, scan):
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")
    before = float(tool._data[0, 0])

    tool._frames.spin_frame.setValue(7)
    _settle(tool)

    assert float(tool._data[0, 0]) != before


def test_a_burst_of_edits_reloads_once(tool, scan):
    """The arrows fire once per press and a clipped mean of a real stack takes
    about a second; reloading on each would make the control feel stuck."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")
    reloads = []
    tool._frames.changed.connect(lambda: reloads.append(True))

    for value in (1, 2, 3, 4, 5):
        tool._frames.spin_frame.setValue(value)
    _settle(tool)

    assert reloads == [True]
    assert float(tool._data[0, 0]) == pytest.approx(5.0)


def test_choosing_an_axis_relists_the_frames(tool, scan):
    """A different axis, a different number of frames, and a different shape."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    tool._frames.combo_axis.setCurrentIndex(2)
    _settle(tool)

    assert tool._frames.spin_frame.maximum() == 11
    assert tool._data.shape == (400, 20)


def test_the_axis_used_is_reported(tool, scan):
    tool.load_dataset_full_key(f"{scan}::entry/signal/img", auto_load=True, slot="CL")

    tool._frames.combo_axis.setCurrentIndex(1)
    _settle(tool)

    assert "axis 1" in tool._status.toolTip()


# ---------------------------------------------------------------------------
# The ROI profile shares the image's units
# ---------------------------------------------------------------------------

@pytest.fixture
def calibrated(tool, scan):
    """A tool with a ring ROI on a loaded image, ready to calibrate."""
    tool.load_dataset_full_key(f"{scan}::entry/signal/flat", auto_load=True, slot="CL")
    tool._center_col.setValue(6)
    tool._center_row.setValue(10)
    tool._add_roi("ring")
    tool._spin_energy.setValue(700.0)
    tool._spin_pixel.setValue(75.0)
    tool._spin_dist.setValue(500.0)
    return tool


def _profile_x(tool):
    x, _y = tool._profile_curve.getData()
    return np.asarray(x if x is not None else [])


def test_the_radial_profile_starts_in_pixels(calibrated):
    calibrated._compute_current_profiles()

    assert calibrated._profile_plot.getAxis("bottom").labelText == "r (px)"


def test_applying_the_calibration_moves_the_profile_to_q(calibrated):
    """The image axes switched to q while the profile below them stayed in
    pixels, so the two halves of one window were in different units."""
    # The profile is normally drawn by a debounce timer; ask for it directly.
    calibrated._compute_current_profiles()
    before = _profile_x(calibrated)
    assert before.size > 0

    calibrated._apply_calibration()
    after = _profile_x(calibrated)

    assert calibrated._profile_plot.getAxis("bottom").labelText == "q (1/A)"
    assert after.size == before.size
    assert after.max() < before.max(), "q values are far smaller than pixel radii"
    assert np.all(np.diff(after) > 0), "still increasing with radius"


def test_switching_back_to_pixels_takes_the_profile_with_it(calibrated):
    calibrated._apply_calibration()

    calibrated._disable_q()

    assert calibrated._profile_plot.getAxis("bottom").labelText == "r (px)"


def test_an_azimuthal_profile_stays_in_degrees(calibrated):
    """It is already an angle; there is nothing for a q calibration to do."""
    roi = calibrated._selected_roi()
    roi["mode"] = "azimuthal"

    calibrated._apply_calibration()

    assert calibrated._profile_plot.getAxis("bottom").labelText == "θ (deg)"


def test_an_incomplete_calibration_leaves_the_profile_readable(calibrated):
    """Zero photon energy cannot give a q; showing nothing would be worse than
    showing the radius."""
    calibrated._spin_energy.setValue(0.0)

    calibrated._apply_calibration()
    calibrated._compute_current_profiles()

    assert calibrated._profile_plot.getAxis("bottom").labelText == "r (px)"
    assert _profile_x(calibrated).size > 0


# ---------------------------------------------------------------------------
# The calculator's Q button
# ---------------------------------------------------------------------------

def test_the_calculator_sends_the_displayed_frame_of_a_stack(qapp, scan, monkeypatch):
    """The guard used to be ndim == 2, so pressing Q on a stack did nothing at
    all — no tool, no message."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    sent: dict = {}

    class _Host:
        def open_q_tool_for_array(self, arr, source_label=""):
            sent["array"] = np.asarray(arr)
            sent["label"] = source_label
            return True

    host = _Host()
    calc = DataCalculatorEnhanced((scan,), None)
    try:
        calc.result_data = STACK[:5]
        calc._last_operation_expr = "A - B"
        calc._update_result_display()
        widget = calc._result_image_widget()
        assert widget is not None
        widget.slider_slice.setValue(3)

        monkeypatch.setattr(type(calc), "parent", lambda self: host)
        calc._on_q_request_from_result_viewer(None)

        assert sent["array"].shape == (20, 12)
        assert float(sent["array"][0, 0]) == pytest.approx(3.0), "the frame on screen"
        assert "slice 3" in sent["label"]
    finally:
        calc.deleteLater()


def test_the_calculator_still_sends_a_plain_result(qapp, scan, monkeypatch):
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    sent: dict = {}

    class _Host:
        def open_q_tool_for_array(self, arr, source_label=""):
            sent["array"] = np.asarray(arr)
            return True

    calc = DataCalculatorEnhanced((scan,), None)
    try:
        calc.result_data = np.arange(240.0).reshape(20, 12)
        calc._update_result_display()

        monkeypatch.setattr(type(calc), "parent", lambda self: _Host())
        calc._on_q_request_from_result_viewer(None)

        assert sent["array"].shape == (20, 12)
    finally:
        calc.deleteLater()

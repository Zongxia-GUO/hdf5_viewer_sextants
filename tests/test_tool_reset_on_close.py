"""Closing a tool window throws its results away.

Both tools are kept alive after being closed so that reopening them is instant.
That is worth having, but it made closing look like a reset it was not: the tool
came back showing the previous run, which reads as a result for whatever dataset
is selected now — the most expensive kind of wrong, because it looks right.

The rule is only about *results*. The A/B dataset selections in the calculator
are the question, not the answer, and they survive.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtGui import QCloseEvent

from src.gui.data_calculator_enhanced import DataCalculatorEnhanced
from src.gui.xrms_analyze_tool import XRMSAnalyzeTool


def _close(widget) -> None:
    """Close the window the way the title bar's X does."""
    widget.closeEvent(QCloseEvent())


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

@pytest.fixture
def calculator(qapp):
    tool = DataCalculatorEnhanced(tuple())
    yield tool
    tool.deleteLater()


def test_a_calculated_result_does_not_survive_closing(calculator):
    calculator.result_data = np.arange(10.0)
    calculator.data_a = np.arange(10.0)
    calculator.data_b = np.ones(10)
    calculator._last_operation_expr = "A - B"

    _close(calculator)

    assert calculator.result_data is None
    assert calculator.data_a is None
    assert calculator.data_b is None
    assert calculator._last_operation_expr == ""


def test_the_result_viewer_is_emptied_too(calculator):
    """Clearing the array but leaving the curve drawn would be worse than not
    clearing at all."""
    calculator.result_data = np.arange(10.0)
    calculator._update_result_display()

    _close(calculator)

    assert calculator.result_widget.current_widget.y_data.size == 0


def test_the_export_buttons_go_dead_with_the_result(calculator):
    calculator.result_data = np.arange(10.0)
    for button in (calculator.btn_export, calculator.btn_plot,
                   calculator.btn_transfer_to_comparison):
        button.setEnabled(True)

    _close(calculator)

    assert not calculator.btn_export.isEnabled()
    assert not calculator.btn_plot.isEnabled()
    assert not calculator.btn_transfer_to_comparison.isEnabled()


def test_the_tool_still_works_after_being_closed_and_reopened(calculator):
    """It is the same object next time, so the reset must leave it usable."""
    calculator.result_data = np.arange(10.0)
    _close(calculator)

    calculator.result_data = np.arange(5.0)
    calculator._update_result_display()

    assert calculator.result_widget.current_widget.y_data.size == 5


def test_the_plot_and_the_export_follow_the_result_viewer_s_despike(calculator):
    """The viewer's Despike button is where a glitch in a calculated curve gets
    removed. Plotting or exporting the raw result afterwards would put the spike
    back in both — the worst outcome, because the filter would look like it had
    worked."""
    from src.gui.despike_dialog import DEFAULT_SETTINGS

    result = np.full(300, 4.0) + np.random.RandomState(0).randn(300) * 0.05
    result[150] += 40.0
    calculator.result_data = result
    calculator._last_operation_expr = "A - B"
    calculator._update_result_display()

    viewer = calculator._result_plot_widget()
    viewer.despike_settings = dict(DEFAULT_SETTINGS)
    viewer.apply_despike()

    headers, columns = calculator._result_export_columns(None)
    plotted = calculator.plot_series()

    # The result is always the last column, and is now named after the
    # expression rather than the fixed word "Result".
    assert headers[-1] == "A - B"
    assert max(columns[-1]) < 10.0
    assert max(plotted[-1].y) < 10.0
    assert "Despike" in calculator._despike_comment()[0]


def test_an_unfiltered_result_is_exported_untouched(calculator):
    result = np.arange(10.0)
    calculator.result_data = result
    calculator._update_result_display()

    _headers, columns = calculator._result_export_columns(None)

    np.testing.assert_array_equal(columns[-1], result)
    assert calculator._despike_comment() == []


# ---------------------------------------------------------------------------
# XRMS analysis
# ---------------------------------------------------------------------------

@pytest.fixture
def xrms(qapp):
    tool = XRMSAnalyzeTool(tuple(), [])
    yield tool
    tool.deleteLater()


def _load_a_stack(tool, frames: int = 4, size: int = 16) -> None:
    stack = np.random.RandomState(0).rand(frames, size, size).astype(np.float32)
    tool._set_new_stack(stack, "test")


def test_the_loaded_stack_does_not_survive_closing(xrms):
    """It is also the largest thing the application holds — whole image stacks
    stayed in memory for as long as it ran."""
    _load_a_stack(xrms)
    assert xrms._data is not None

    _close(xrms)

    assert xrms._data is None
    assert xrms._combined is None
    assert xrms._n_frames == 0
    assert all(value is None for value in xrms._slots.values())


def test_computed_profiles_do_not_survive_closing(xrms):
    _load_a_stack(xrms)
    xrms._radial_x = np.arange(5.0)
    xrms._radial_y = np.arange(5.0)
    xrms._time_x = np.arange(4.0)
    xrms._time_y = np.arange(4.0)

    _close(xrms)

    assert xrms._radial_x is None and xrms._radial_y is None
    assert xrms._time_x is None and xrms._time_y is None


def test_the_batch_fit_table_does_not_survive_closing(xrms):
    """Page 4 draws its summary straight from this list."""
    _load_a_stack(xrms)
    xrms._b3_results = [{"frame": 0, "R2": 0.99}, {"frame": 1, "R2": 0.98}]

    _close(xrms)

    assert xrms._b3_results == []


def test_rois_and_the_beam_stop_do_not_survive_closing(xrms):
    """They are drawn in pixel coordinates, so they mean nothing over a
    different image."""
    _load_a_stack(xrms)
    xrms._add_roi("ring")
    assert xrms._rois

    _close(xrms)

    assert xrms._rois == []
    assert xrms._beamstop is None
    assert xrms._mask_shapes == []


def test_the_frame_navigator_goes_back_to_empty(xrms):
    _load_a_stack(xrms, frames=4)
    assert xrms._sl_frame.isEnabled()

    _close(xrms)

    assert not xrms._sl_frame.isEnabled()
    assert xrms._sl_frame.maximum() == 0
    assert xrms._lbl_frame_info.text() == "0 / 0"


def test_a_stack_can_be_loaded_again_after_closing(xrms):
    """The reset must leave a working tool, not a broken one."""
    _load_a_stack(xrms, frames=4)
    _close(xrms)

    _load_a_stack(xrms, frames=6, size=8)

    assert xrms._n_frames == 6
    assert xrms._data.shape == (6, 8, 8)


def test_the_fit_range_lines_come_back(xrms):
    """Clearing the plots takes them with it; they are furniture, not results,
    and losing them would leave the fit page unusable."""
    _load_a_stack(xrms)

    _close(xrms)

    for line in xrms._range_lines:
        assert line.getViewBox() is not None, "line is not on any plot"

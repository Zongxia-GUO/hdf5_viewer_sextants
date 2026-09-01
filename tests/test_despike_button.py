"""The Despike button in the 1-D viewer toolbar.

It follows the same rules as X to q next to it: the parameters are asked for
when it is switched on, cancelling leaves it looking off because nothing was
applied, and clicking it again puts the raw values back.

The property that matters most is the last one — the filter is a view of the
data, never an edit of it, so ``y_data`` must come through untouched no matter
what the button has been doing.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog

from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced


@pytest.fixture
def widget(qapp):
    view = PlotWidget1DEnhanced()
    yield view
    view.close()
    view.deleteLater()


def _spiky(n: int = 400) -> np.ndarray:
    """Flat data with exactly two glitches, one up and one down."""
    y = np.full(n, 10.0) + np.random.RandomState(0).randn(n) * 0.05
    y[n // 4] += 40.0
    y[3 * n // 4] -= 40.0
    return y


def _settings(**overrides) -> dict:
    from src.gui.despike_dialog import DEFAULT_SETTINGS

    return {**DEFAULT_SETTINGS, **overrides}


# ---------------------------------------------------------------------------
# Switching it on and off
# ---------------------------------------------------------------------------

def test_the_button_is_dead_until_there_is_data(widget):
    assert widget.btn_despike.isEnabled() is False

    widget.set_data(_spiky())

    assert widget.btn_despike.isEnabled() is True


def test_applying_marks_the_button_and_reports_the_count(widget):
    widget.set_data(_spiky())
    widget.despike_settings = _settings()

    widget.apply_despike()

    assert widget.btn_despike.isChecked() is True
    assert not widget.label_despike.isHidden()
    assert "2 / 400" in widget.label_despike.text()


def test_switching_it_off_restores_the_raw_curve(widget):
    y = _spiky()
    widget.set_data(y)
    widget.despike_settings = _settings()
    widget.apply_despike()
    assert widget.display_y[100] < 20.0, "the spike is gone while it is on"

    widget._on_despike_clicked(False)

    assert widget.btn_despike.isChecked() is False
    assert widget.label_despike.isHidden()
    np.testing.assert_array_equal(widget.display_y, y)


def test_cancelling_the_dialog_leaves_the_button_off(widget, monkeypatch):
    """Otherwise the button says the data is filtered when it is not."""
    widget.set_data(_spiky())
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Rejected.value)

    widget.btn_despike.setChecked(True)     # what the click does before the handler
    widget._on_despike_clicked(True)

    assert widget.btn_despike.isChecked() is False
    assert widget._despiked_y is None


def test_accepting_the_dialog_applies_it(widget, monkeypatch):
    widget.set_data(_spiky())
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Accepted.value)

    widget._on_despike_clicked(True)

    assert widget.btn_despike.isChecked() is True
    assert widget._despiked_y is not None


def test_clicking_with_no_data_does_nothing(widget):
    widget.btn_despike.setChecked(True)

    widget._on_despike_clicked(True)

    assert widget.btn_despike.isChecked() is False


# ---------------------------------------------------------------------------
# The source data is never touched
# ---------------------------------------------------------------------------

def test_the_loaded_data_is_never_modified(widget):
    y = _spiky()
    original = y.copy()
    widget.set_data(y)
    widget.despike_settings = _settings()

    widget.apply_despike()

    np.testing.assert_array_equal(widget.y_data, original)
    np.testing.assert_array_equal(y, original)


def test_loading_another_dataset_refilters_rather_than_reusing_the_old_mask(widget):
    widget.set_data(_spiky())
    widget.despike_settings = _settings()
    widget.apply_despike()

    widget.set_data(_spiky(200))

    assert widget.btn_despike.isChecked() is True
    assert widget._spike_mask is not None
    assert widget._spike_mask.shape == (200,), "the mask belongs to the new curve"


def test_data_that_no_longer_needs_filtering_shows_a_count_of_zero(widget):
    widget.set_data(_spiky())
    widget.despike_settings = _settings()
    widget.apply_despike()

    widget.set_data(np.full(300, 5.0) + np.random.RandomState(1).randn(300) * 0.05)

    assert "0 / 300" in widget.label_despike.text()


# ---------------------------------------------------------------------------
# What is shown is what is exported
# ---------------------------------------------------------------------------

def test_the_plot_window_gets_the_filtered_curve(widget):
    widget.set_data(_spiky())
    widget.despike_settings = _settings()
    widget.apply_despike()

    series = widget.current_series()

    assert max(series[0].y) < 20.0


def test_an_export_says_it_was_filtered(widget):
    """A despiked file that does not admit it is a trap for whoever opens it
    next, which may well be the same person a year later."""
    widget.set_data(_spiky())
    widget.despike_settings = _settings()
    widget.apply_despike()

    comment = widget._despike_comment()

    assert len(comment) == 1
    assert "Despike" in comment[0]
    assert "2 point(s) replaced" in comment[0]
    assert "linear scale" in comment[0]


def test_a_reflectivity_export_records_that_it_used_the_log_scale(widget):
    """The threshold means a ratio in that case and an absolute amount in the
    other, so the number alone does not describe what was done."""
    decades = 10.0 ** np.linspace(0, -6, 600)
    decades[400] *= 6.0
    widget.set_data(decades)
    widget.despike_settings = _settings()

    widget.apply_despike()

    assert widget._spike_mask[400]
    assert "log scale" in widget._despike_comment()[0]


def test_no_comment_when_the_filter_is_off(widget):
    widget.set_data(_spiky())

    assert widget._despike_comment() == []


def test_mark_only_reports_flagged_and_leaves_the_values(widget):
    y = _spiky()
    widget.set_data(y)
    widget.despike_settings = _settings(replace=False)

    widget.apply_despike()

    np.testing.assert_array_equal(widget.display_y, y)
    assert "flagged only" in widget._despike_comment()[0]
    assert int(np.count_nonzero(widget._spike_mask)) == 2


# ---------------------------------------------------------------------------
# The parameter box
# ---------------------------------------------------------------------------

def test_the_prompt_says_what_the_settings_would_catch(qapp):
    """A threshold in sigma means nothing on its own — only against the data
    in front of you — so the count is shown before anything is applied."""
    from src.gui.despike_dialog import DespikeDialog

    # A modest glitch, so that raising the threshold can actually clear it —
    # the spikes in _spiky() stand hundreds of sigma out of the noise.
    y = np.full(400, 10.0) + np.random.RandomState(0).randn(400)
    y[100] += 20.0

    prompt = DespikeDialog(y)
    try:
        assert "1 of 400" in prompt.lbl_preview.text()

        prompt.spn_threshold.setValue(100.0)
        assert "0 of 400" in prompt.lbl_preview.text()
    finally:
        prompt.deleteLater()


def test_the_prompt_opens_on_the_settings_last_used(qapp):
    from src.gui.despike_dialog import DespikeDialog

    prompt = DespikeDialog(_spiky(), settings=_settings(threshold=9.0, window=11))
    try:
        assert prompt.settings()["threshold"] == 9.0
        assert prompt.settings()["window"] == 11
    finally:
        prompt.deleteLater()


def test_the_prompt_says_which_scale_auto_chose(qapp):
    """Auto is the default, and its answer changes what the threshold means —
    so it must not be a decision the user cannot see."""
    from src.gui.despike_dialog import DespikeDialog

    decades = 10.0 ** np.linspace(0, -6, 600)
    prompt = DespikeDialog(decades)
    try:
        assert "Auto chose the log scale" in prompt.lbl_preview.text()

        prompt.cmb_space.setCurrentText("Linear")
        assert "Auto chose" not in prompt.lbl_preview.text()
    finally:
        prompt.deleteLater()


def test_the_prompt_warns_when_it_would_reshape_the_data(qapp):
    from src.gui.despike_dialog import DespikeDialog

    prompt = DespikeDialog(_spiky(), settings=_settings(threshold=0.5))
    try:
        assert "raise the threshold" in prompt.lbl_preview.text()
    finally:
        prompt.deleteLater()


# ---------------------------------------------------------------------------
# Multi-column data
# ---------------------------------------------------------------------------

def test_each_column_is_filtered_on_its_own(widget):
    table = np.column_stack([_spiky(), np.full(400, 3.0)])
    table[50, 1] = 90.0
    widget.set_data(table)
    widget.despike_settings = _settings()

    widget.apply_despike()

    assert widget._spike_mask.shape == table.shape
    assert widget._spike_mask[50, 1]
    assert widget._spike_mask[100, 0]
    assert not widget._spike_mask[50, 0]

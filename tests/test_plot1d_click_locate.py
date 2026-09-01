"""Tests for click-to-locate in the 1-D plot widget.

Regression: the search used ``argmin(|x - x_click|)`` to pick a centre index and
then scanned only +-20 indices around it. ``np.argmin`` returns the index of a
NaN when one is present, so a single missing X sample pinned the centre for every
click and froze the reachable range to a fixed index window.
"""

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import Qt

from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced


class _Click:
    def __init__(self, scene_pos, button=Qt.MouseButton.LeftButton):
        self._p = scene_pos
        self._b = button

    def button(self):
        return self._b

    def scenePos(self):
        return self._p


def _widget(qapp, y, x=None):
    w = PlotWidget1DEnhanced()
    w.resize(1000, 650)
    w.show()
    w.set_data(y, x)
    return w


def _click_at(widget, x_value, y_value):
    vb = widget.plot_widget.plotItem.vb
    widget.selected_point = None
    widget._on_mouse_clicked(_Click(vb.mapViewToScene(pg.Point(float(x_value), float(y_value)))))
    return widget.selected_point


@pytest.fixture
def sweep():
    """A field sweep with a missing sample in the middle, as in scan 33."""
    x = np.linspace(-7200.0, 7200.0, 73)
    y = x * 0.87
    x = x.copy()
    x[36] = np.nan  # the zero-field setpoint was not recorded
    return x, y


# ---------------------------------------------------------------------------
# The reported bug
# ---------------------------------------------------------------------------

def test_nan_in_x_does_not_pin_the_search(qapp, sweep):
    """A NaN anywhere in X must not make far-out points unselectable."""
    x, y = sweep
    w = _widget(qapp, y, x)

    assert _click_at(w, x[0], y[0])[0] == pytest.approx(x[0])
    assert _click_at(w, x[-1], y[-1])[0] == pytest.approx(x[-1])


def test_every_finite_sample_is_reachable_with_a_nan_in_x(qapp, sweep):
    x, y = sweep
    w = _widget(qapp, y, x)

    misses = [
        float(x[i])
        for i in np.flatnonzero(np.isfinite(x))
        if _click_at(w, x[i], y[i])[0] != pytest.approx(float(x[i]))
    ]
    assert misses == []


def test_the_nan_sample_itself_is_never_selected(qapp, sweep):
    x, y = sweep
    w = _widget(qapp, y, x)

    located = _click_at(w, 0.0, 0.0)
    assert located is not None
    assert np.isfinite(located[0])


# ---------------------------------------------------------------------------
# Assumptions the old index-window search made
# ---------------------------------------------------------------------------

def test_non_monotonic_x_is_handled(qapp):
    """A hysteresis loop revisits X values; nearest-in-X is not nearest-in-index."""
    up = np.linspace(-1000.0, 1000.0, 101)
    x = np.concatenate([up, up[::-1]])
    y = np.concatenate([up * 0.5 - 200.0, up[::-1] * 0.5 + 200.0])
    w = _widget(qapp, y, x)

    # Two samples share x = 0; the click must resolve to the branch it is on.
    lower = _click_at(w, 0.0, -200.0)
    upper = _click_at(w, 0.0, 200.0)
    assert lower[1] == pytest.approx(-200.0)
    assert upper[1] == pytest.approx(200.0)


def test_distance_is_measured_in_pixels_not_data_units(qapp):
    """With a huge Y range and a tiny X range, X must still discriminate."""
    x = np.linspace(0.0, 1.0, 201)
    y = np.linspace(0.0, 1e6, 201)
    w = _widget(qapp, y, x)

    located = _click_at(w, x[50], y[50])
    assert located[0] == pytest.approx(x[50])


def test_far_index_is_reachable_beyond_the_old_window(qapp):
    """The old code scanned only +-20 indices around its centre."""
    x = np.linspace(0.0, 500.0, 501)
    y = np.sin(x / 50.0)
    w = _widget(qapp, y, x)

    assert _click_at(w, x[480], y[480])[0] == pytest.approx(x[480])


# ---------------------------------------------------------------------------
# Other modes
# ---------------------------------------------------------------------------

def test_log_y_skips_non_positive_samples(qapp):
    y = np.array([-5.0, 1.0, 10.0, 100.0, 1000.0])
    w = _widget(qapp, y)
    w.chk_log_y.setChecked(True)

    located = _click_at(w, 0.0, np.log10(1.0))
    assert located is not None and located[1] > 0


def test_multi_curve_selects_the_nearest_curve(qapp):
    y = np.column_stack([np.zeros(50), np.full(50, 1000.0)])
    w = _widget(qapp, y)

    low = _click_at(w, 25, 20)
    high = _click_at(w, 25, 980)
    assert low[1] == pytest.approx(0.0) and low[2] == 0
    assert high[1] == pytest.approx(1000.0) and high[2] == 1


def test_single_curve_reports_no_curve_index(qapp):
    w = _widget(qapp, np.arange(20, dtype=float))
    assert _click_at(w, 10, 10)[2] is None


def test_right_click_clears_the_selection(qapp):
    w = _widget(qapp, np.arange(20, dtype=float))
    assert _click_at(w, 10, 10) is not None

    vb = w.plot_widget.plotItem.vb
    w._on_mouse_clicked(
        _Click(vb.mapViewToScene(pg.Point(10.0, 10.0)), button=Qt.MouseButton.RightButton)
    )
    assert w.selected_point is None

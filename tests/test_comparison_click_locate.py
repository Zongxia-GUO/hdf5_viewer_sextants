"""Tests for click-to-locate in the Data Comparison plot.

Regression: points drawn outside the current view rectangle used to be dropped
from the candidate set, so zooming the Y axis made the curve's peaks
unselectable even though their X position was still on screen.
"""

import numpy as np
import pyqtgraph as pg
import pytest

from src.gui.data_comparison import DataComparisonTool


class _Click:
    """Stand-in for the pyqtgraph mouse-click event."""

    def __init__(self, scene_pos):
        self._p = scene_pos

    def scenePos(self):
        return self._p


@pytest.fixture
def sweep(qapp):
    """A field sweep: setpoint on X (+-7000), measured value on Y (+-6000)."""
    tool = DataComparisonTool(tuple())
    x = np.linspace(-7000.0, 7000.0, 141)
    y = x * 0.86
    tool.add_dataset_from_array("scan::measured", y)
    tool.resize(900, 600)
    tool.show()
    tool._on_x_data_selected(x, "scan::setpoint")
    return tool, x, y


def _click_at(tool, x_value, y_value):
    """Click the plot at a data coordinate and return the located point."""
    vb = tool.plot_widget.plotItem.vb
    tool._on_plot_clicked(_Click(vb.mapViewToScene(pg.Point(float(x_value), float(y_value)))))
    return tool.selected_point


def test_point_inside_the_view_is_located(sweep):
    tool, x, y = sweep
    assert _click_at(tool, x[70], y[70]) == pytest.approx((x[70], y[70]))


def test_point_outside_a_zoomed_y_axis_is_still_located(sweep):
    """The reported bug: Y zoomed to +-4000 hid every |x| beyond ~4600."""
    tool, x, y = sweep
    tool.plot_widget.plotItem.vb.setRange(yRange=(-4000, 4000), padding=0)

    far = len(x) - 3  # x ~ +6900, y ~ +5900 -> above the visible band
    located = _click_at(tool, x[far], y[far])
    assert located == pytest.approx((x[far], y[far]))


def test_every_sample_stays_reachable_when_y_is_zoomed(sweep):
    tool, x, y = sweep
    tool.plot_widget.plotItem.vb.setRange(yRange=(-4000, 4000), padding=0)

    misses = [i for i in range(0, len(x), 7) if _click_at(tool, x[i], y[i]) != pytest.approx((x[i], y[i]))]
    assert misses == []


def test_point_outside_a_zoomed_x_axis_is_still_located(sweep):
    tool, x, y = sweep
    tool.plot_widget.plotItem.vb.setRange(xRange=(-1000, 1000), padding=0)

    far = 2
    assert _click_at(tool, x[far], y[far]) == pytest.approx((x[far], y[far]))


# ---------------------------------------------------------------------------
# Invalid samples
# ---------------------------------------------------------------------------

def test_nan_samples_are_never_selected(qapp):
    """Real scans carry NaN where a point was skipped; those are not clickable."""
    tool = DataComparisonTool(tuple())
    x = np.array([-200.0, np.nan, 200.0, 400.0])
    y = np.array([-2.0, 0.0, 2.0, 4.0])
    tool.add_dataset_from_array("scan::measured", y)
    tool.resize(900, 600)
    tool.show()
    tool._on_x_data_selected(x, "scan::setpoint")

    # Click right where the NaN sample would sit if it were plotted.
    located = _click_at(tool, 0.0, 0.0)
    assert located is not None
    assert np.isfinite(located[0])
    assert located[0] in (-200.0, 200.0)


def test_log_y_excludes_non_positive_samples(qapp):
    tool = DataComparisonTool(tuple())
    y = np.array([-5.0, 1.0, 10.0, 100.0])
    tool.add_dataset_from_array("scan::measured", y)
    tool.resize(900, 600)
    tool.show()
    tool.chk_log_y.setChecked(True)

    located = _click_at(tool, 0.0, np.log10(1.0))
    assert located is not None
    assert located[1] > 0


# ---------------------------------------------------------------------------
# Several curves
# ---------------------------------------------------------------------------

def test_nearest_curve_wins(qapp):
    tool = DataComparisonTool(tuple())
    tool.add_dataset_from_array("low", np.zeros(50))
    tool.add_dataset_from_array("high", np.full(50, 1000.0))
    tool.resize(900, 600)
    tool.show()

    assert _click_at(tool, 25, 20)[1] == pytest.approx(0.0)
    assert _click_at(tool, 25, 980)[1] == pytest.approx(1000.0)


def test_curve_transform_is_honoured_by_the_click(qapp):
    """The click must search the transformed curve, not the raw samples."""
    tool = DataComparisonTool(tuple())
    tool.add_dataset_from_array("scan::measured", np.arange(50, dtype=float))
    tool.resize(900, 600)
    tool.show()
    tool.datasets[0].y_expr = "y * 10"
    tool._update_plot()

    assert _click_at(tool, 30, 300)[1] == pytest.approx(300.0)

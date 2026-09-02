"""One spelling per quantity, one palette per overlay.

The same q axis used to be written three ways in one window ("qx (m1/A)" from
pyqtgraph's SI prefix, "q (1/A) (x0.001)" from its scale suffix, "q (A^-1)" by
hand), and the same region drawn over an image was red in the 2-D viewer and
blue in the scattering tool. These tests pin both.
"""

import pathlib

import pyqtgraph as pg

from src.gui._shared import (
    AXIS_ANGLE_DEG,
    AXIS_Q,
    AXIS_RADIUS_PX,
    ROI_IDLE_RGB,
    ROI_SELECTED_RGB,
    roi_pen,
    set_axis_label,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "gui"


def _read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


# ── The label helper ──────────────────────────────────────────────────── #

def test_the_unit_is_written_out_and_the_ticks_are_left_alone(qapp):
    plot = pg.PlotWidget()
    try:
        set_axis_label(plot, "bottom", "qx", "1/A")
        axis = plot.plotItem.getAxis("bottom")

        assert "1/A" in axis.labelString()
        # No "m" prefix on the unit and no "(x0.001)" suffix on the label.
        assert "m1/A" not in axis.labelString()
        assert axis.autoSIPrefix is False
        # The numbers as they are, which is what the axis is being read for.
        assert axis.tickStrings([0.0005, 0.001], axis.scale, 0.0005) == ["0.0005", "0.0010"]
    finally:
        plot.close()
        plot.deleteLater()


def test_a_label_that_already_carries_its_unit_is_not_given_a_second_one(qapp):
    plot = pg.PlotWidget()
    try:
        set_axis_label(plot, "bottom", AXIS_Q)
        assert plot.plotItem.getAxis("bottom").labelString().count("1/A") == 1
    finally:
        plot.close()
        plot.deleteLater()


def test_a_plot_item_works_as_well_as_a_plot_widget(qapp):
    win = pg.GraphicsLayoutWidget()
    try:
        item = win.addPlot()
        set_axis_label(item, "left", "Intensity")
        assert "Intensity" in item.getAxis("left").labelString()
    finally:
        win.close()
        win.deleteLater()


# ── One spelling per quantity ─────────────────────────────────────────── #

def test_the_radial_and_azimuthal_axes_are_spelled_once():
    """The scattering tool and the Q tool name the same profile the same way."""
    xrms = _read("xrms_analyze_tool.py")
    assert "Radius (pixels)" not in xrms
    assert "Angle (°)" not in xrms
    assert "Angle (deg)" not in xrms
    assert AXIS_RADIUS_PX in _read("q_calibration_tool.py")
    assert AXIS_ANGLE_DEG in _read("q_calibration_tool.py")


def test_q_is_spelled_once():
    """The comparison window wrote A^-1 where every other window wrote 1/A."""
    assert 'setLabel("bottom", "q (A^-1)")' not in _read("data_comparison.py")
    for name in ("data_comparison.py", "plot_widget_1d_enhanced.py"):
        assert "AXIS_Q" in _read(name)


def test_no_window_lets_pyqtgraph_prefix_the_q_unit():
    """`units="1/A"` is what produced "qx (m1/A)"; set_axis_label replaced it."""
    for name in ("image_view_2d_enhanced.py", "q_calibration_tool.py",
                 "xrms_analyze_tool.py", "plot_widget_1d_enhanced.py",
                 "data_comparison.py"):
        assert 'units="1/A"' not in _read(name)
        assert "units='1/A'" not in _read(name)


# ── One palette per overlay ───────────────────────────────────────────── #

def test_the_selected_and_idle_pens_differ_in_colour_and_width():
    assert roi_pen(True).color().getRgb()[:3] == ROI_SELECTED_RGB
    assert roi_pen(False).color().getRgb()[:3] == ROI_IDLE_RGB
    assert roi_pen(True).widthF() > roi_pen(False).widthF()
    assert roi_pen(False, width=1.2).widthF() == 1.2


def test_no_region_over_an_image_is_drawn_in_red():
    """The line, the rectangle and the sector outline all went through 'r'."""
    view = _read("image_view_2d_enhanced.py")
    assert "mkPen('r', width=2)\n                )" not in view
    assert "pen = pg.mkPen('r', width=1.5)" not in view
    assert view.count("roi_pen(") >= 3


def test_both_tools_take_their_overlay_pen_from_the_same_place():
    for name in ("image_view_2d_enhanced.py", "q_calibration_tool.py"):
        text = _read(name)
        assert "roi_pen" in text
        # The literals now live only in _shared.
        assert "(0, 255, 255) if selected" not in text
        assert "(90, 160, 200)" not in text


def test_the_line_and_the_rectangle_are_created_with_the_shared_pen(qapp):
    """Drawn, not just spelled: the pens that reach pyqtgraph are the blue ones."""
    rect = pg.RectROI([0, 0], [10, 10], pen=roi_pen(selected=True))
    line = pg.LineSegmentROI([[0, 0], [10, 0]], pen=roi_pen(selected=True))
    for item in (rect, line):
        assert item.pen.color().getRgb()[:3] == ROI_SELECTED_RGB

"""One spelling per quantity, one palette per overlay.

The same q axis used to be written three ways in one window ("qx (m1/A)" from
pyqtgraph's SI prefix, "q (1/A) (x0.001)" from its scale suffix, "q (A^-1)" by
hand), and the same region drawn over an image was red in the 2-D viewer and
blue in the scattering tool. These tests pin both.
"""

import pathlib

import pyqtgraph as pg
import pytest

from src.gui._shared import (
    AXIS_ANGLE_DEG,
    AXIS_Q,
    AXIS_RADIUS_PX,
    PROFILE_CURVE_RGB,
    ROI_IDLE_RGB,
    ROI_SELECTED_RGB,
    profile_pen,
    roi_pen,
    set_axis_label,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "gui"


def _read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


# ── The label helper ──────────────────────────────────────────────────── #

def _ticks(plot, axis_name, values, spacing):
    """The tick text as pyqtgraph actually draws it.

    Reading ``axis.scale`` alone is not enough: the number a tick shows is
    scaled by ``autoSIPrefixScale * scale``, and it was the first factor that
    went wrong. A test that passes only ``axis.scale`` cannot see the bug.
    """
    axis = plot.plotItem.getAxis(axis_name) if hasattr(plot, "plotItem") else plot.getAxis(axis_name)
    return axis.tickStrings(list(values), axis.autoSIPrefixScale * axis.scale, spacing)


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
        assert _ticks(plot, "bottom", [0.0005, 0.001], 0.0005) == ["0.0005", "0.0010"]
    finally:
        plot.close()
        plot.deleteLater()


def test_an_axis_that_once_had_a_prefix_stops_multiplying_its_ticks(qapp):
    """The 2-D image read 0.018 while the profile below it read 18.

    Turning the feature off is not enough: pyqtgraph recomputes
    autoSIPrefixScale on the way out and then keeps it, so the label loses its
    "m" and the ticks go on being multiplied by 1000 — which looks correct and
    is not.
    """
    plot = pg.PlotWidget()
    try:
        plot.setLabel("bottom", "qx", units="1/A")     # what the old code did
        plot.plot([0.0, 0.018, 0.035], [1, 2, 3])
        plot.plotItem.vb.autoRange()
        axis = plot.plotItem.getAxis("bottom")
        assert _ticks(plot, "bottom", [0.018], 0.005) == ["18"]      # the bug

        set_axis_label(plot, "bottom", "qx", "1/A")

        assert _ticks(plot, "bottom", [0.018], 0.005) == ["0.018"]
        assert axis.autoSIPrefixScale == 1.0
        assert axis.labelUnits == ""
        assert "m1/A" not in axis.labelString()

        # And it stays put when the view is ranged again.
        plot.plotItem.vb.setXRange(0.0, 0.04)
        assert _ticks(plot, "bottom", [0.018], 0.005) == ["0.018"]
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


def test_every_profile_is_drawn_as_a_plain_line(qapp):
    """The line and rectangle profiles put a marker on every sample, which on a
    long profile merges into a band twice the width of the line; the ring and
    sector profiles were already a plain line."""
    view = _read("image_view_2d_enhanced.py")
    assert "symbolBrush='c'" not in view
    assert "pg.mkPen(color='c', width=2)" not in view
    assert view.count("profile_pen()") == 3
    assert "profile_pen()" in _read("q_calibration_tool.py")


def test_the_profile_pen_is_the_one_the_ring_profile_already_used(qapp):
    assert profile_pen().color().getRgb()[:3] == PROFILE_CURVE_RGB
    assert profile_pen().widthF() == 2.0


def test_the_line_and_the_rectangle_are_created_with_the_shared_pen(qapp):
    """Drawn, not just spelled: the pens that reach pyqtgraph are the blue ones."""
    rect = pg.RectROI([0, 0], [10, 10], pen=roi_pen(selected=True))
    line = pg.LineSegmentROI([[0, 0], [10, 0]], pen=roi_pen(selected=True))
    for item in (rect, line):
        assert item.pen.color().getRgb()[:3] == ROI_SELECTED_RGB


# ── End to end, through the widget the bug was seen in ────────────────── #

# A soft x-ray geometry like the beamline's: 700 eV, 13.5 um pixels, 300 mm.
CALIBRATION = {
    "energy_ev": 700.0,
    "pixel_um": 13.5,
    "distance_mm": 300.0,
    "center_x": 60.0,
    "center_y": 40.0,
}


def test_a_line_profile_reads_the_same_q_as_the_image_above_it(qapp):
    """The reported bug: the image said 0.018 and the profile said 18."""
    import numpy as np

    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    view = ImageView2DEnhanced()
    try:
        view.set_data(np.random.RandomState(0).rand(80, 120).astype(np.float32))
        view.set_q_calibration(CALIBRATION)
        assert view.apply_q_axes_calibration(CALIBRATION) is True

        view._on_roi_type_changed("Line")
        handles = view.current_roi.getHandles()
        view.current_roi.movePoint(handles[0], pg.Point(60.0, 40.0))
        view.current_roi.movePoint(handles[1], pg.Point(95.0, 40.0))
        view._update_roi_statistics()

        curve = view.roi_plot_widget.plotItem.listDataItems()[0]
        xs = np.asarray(curve.getData()[0], dtype=float)
        far_end = view._q_components_at_pixel_float(95.0, 40.0, CALIBRATION)

        # The profile's last sample is the q of the pixel the line ends on —
        # not that q times a thousand.
        assert xs[-1] == pytest.approx(abs(far_end[0]), rel=1e-6)
        assert _ticks(view.roi_plot_widget, "bottom", [xs[-1]], xs[1] - xs[0])[0] != "18"

        axis = view.roi_plot_widget.getAxis("bottom")
        assert axis.autoSIPrefixScale == 1.0
        assert "m1/A" not in axis.labelString()

        # And no marker on every sample.
        assert curve.opts.get("symbol") in (None, "")
    finally:
        view.close()
        view.deleteLater()

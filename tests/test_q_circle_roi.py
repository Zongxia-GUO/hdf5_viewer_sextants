"""The circle ROI is a spot tool, and its axis has to say so.

A circle ROI is dragged over a diffraction spot and its profile is measured
outward from *that* spot, not from the beam. Two things followed from nobody
having written that down:

* the radius was converted to q using the beam centre, so the axis was |q|
  along a ray out of the beam and started at zero wherever the spot sat — a
  spot at |q| = 0.00102 spanning 0.00080..0.00129 was drawn on an axis running
  0..0.00032, labelled "q";
* the profile ran out to the ellipse's *long* semi-axis, so every bin past the
  short one averaged over fewer and fewer directions.
"""

import numpy as np
import pytest

from src.gui._shared import AXIS_DELTA_Q, AXIS_Q, AXIS_RADIUS_PX
from src.gui.q_calibration_tool import QCalibrationTool

# A soft x-ray geometry like the beamline's, with the beam centre at (100, 100).
CAL = dict(energy_ev=700.0, pixel_um=13.5, distance_mm=300.0,
           center_x=100.0, center_y=100.0)

# Short camera length, where q per pixel varies enough for the origin to matter.
WIDE = dict(energy_ev=700.0, pixel_um=30.0, distance_mm=10.0,
            center_x=100.0, center_y=100.0)


@pytest.fixture
def tool(qapp):
    t = QCalibrationTool(opened_files=(), dataset_full_keys_2d=[])
    yield t
    t.deleteLater()


def _load(tool, cal=None, image=None):
    img = np.ones((200, 200)) if image is None else image
    tool._raw_data = img
    tool._data = img
    tool._img.set_data(img)
    if cal is not None:
        tool._img.set_q_calibration(cal)
        tool._img.apply_q_axes_calibration(cal)
        tool._center_col.setValue(int(cal["center_x"]))
        tool._center_row.setValue(int(cal["center_y"]))
    return img


def _circle(tool, col, row, rx, ry=None):
    tool._add_roi("circle")
    roi = tool._rois[-1]
    roi["cx"], roi["cy"] = float(col), float(row)
    roi["rx"], roi["ry"] = float(rx), float(ry if ry is not None else rx)
    return roi


def _q(tool, cal, col, row):
    return np.asarray(tool._img._q_components_at_pixel_float(col, row, cal)[:2])


# ── The axis is a separation from the spot, not an absolute q ─────────── #

@pytest.mark.parametrize("cal,spot", [(CAL, (150.0, 60.0)), (WIDE, (170.0, 30.0))])
def test_the_circle_axis_is_the_q_distance_from_its_own_centre(tool, cal, spot):
    _load(tool, cal)
    roi = _circle(tool, spot[0], spot[1], 20.0)
    radii = np.arange(0.0, 21.0)

    axis, label = tool._radial_axis(roi, radii)

    direction = np.asarray(spot) - np.asarray([cal["center_x"], cal["center_y"]])
    direction = direction / np.hypot(*direction)
    origin = _q(tool, cal, *spot)
    expected = np.asarray([
        np.hypot(*(_q(tool, cal, spot[0] + r * direction[0],
                      spot[1] + r * direction[1]) - origin))
        for r in radii
    ])

    assert label == AXIS_DELTA_Q
    assert np.array_equal(axis, expected)


def test_the_axis_starts_at_the_spot_so_its_first_point_is_zero(tool):
    _load(tool, CAL)
    roi = _circle(tool, 150.0, 60.0, 20.0)

    axis, _ = tool._radial_axis(roi, np.arange(0.0, 21.0))

    assert axis[0] == 0.0
    assert np.all(np.diff(axis) > 0), "a separation must grow with the radius"


def test_the_old_beam_centre_conversion_is_gone(tool):
    """It answered a different question: |q| along a ray out of the beam."""
    _load(tool, WIDE)
    spot = (170.0, 30.0)
    roi = _circle(tool, spot[0], spot[1], 20.0)
    radii = np.arange(0.0, 21.0)

    axis, _ = tool._radial_axis(roi, radii)
    old = np.asarray([np.hypot(*_q(tool, WIDE, WIDE["center_x"] + r, WIDE["center_y"]))
                      for r in radii])

    assert not np.allclose(axis, old, rtol=1e-3)


def test_a_spot_sitting_on_the_beam_centre_still_works(tool):
    """The beam-to-spot direction is undefined there; any direction will do."""
    _load(tool, CAL)
    roi = _circle(tool, CAL["center_x"], CAL["center_y"], 15.0)

    axis, label = tool._radial_axis(roi, np.arange(0.0, 16.0))

    assert label == AXIS_DELTA_Q
    assert np.all(np.isfinite(axis))
    assert axis[0] == 0.0


def test_without_a_calibration_the_circle_axis_stays_in_pixels(tool):
    _load(tool)
    roi = _circle(tool, 150.0, 60.0, 20.0)
    radii = np.arange(0.0, 21.0)

    axis, label = tool._radial_axis(roi, radii)

    assert label == AXIS_RADIUS_PX
    assert np.array_equal(axis, radii)


def test_a_ring_still_reports_absolute_q(tool):
    """Only the circle moved origin; a ring IS measured from the beam."""
    _load(tool, CAL)
    tool._add_roi("ring")
    roi = tool._rois[-1]

    axis, label = tool._radial_axis(roi, np.arange(0.0, 21.0))

    # A ring samples along its own mid-angle, which for the default 0..180 is
    # straight up rather than along +x.
    mid = np.deg2rad((float(roi["a_min"]) + float(roi["a_max"])) / 2.0)
    dx, dy = float(np.cos(mid)), float(np.sin(mid))
    expected = np.asarray([
        np.hypot(*_q(tool, CAL, CAL["center_x"] + r * dx, CAL["center_y"] + r * dy))
        for r in np.arange(0.0, 21.0)
    ])

    assert label == AXIS_Q
    assert np.array_equal(axis, expected)


def test_the_exported_header_carries_the_new_axis_name(tool):
    """The export reads the axis label, so it was wrong the same way."""
    _load(tool, CAL)
    _circle(tool, 150.0, 60.0, 20.0)
    tool._compute_current_profiles()

    headers, _ = tool._profile_columns()

    assert headers[0] == AXIS_DELTA_Q


# ── The profile stops where the ring is still whole ───────────────────── #

@pytest.mark.parametrize("rx,ry", [(30.0, 30.0), (40.0, 12.0), (12.0, 40.0)])
def test_the_profile_stops_at_the_short_semi_axis(tool, rx, ry):
    image = _load(tool)
    roi = _circle(tool, 100.0, 100.0, rx, ry)

    radii, values, _, _ = tool._roi_profiles(roi, image)
    finite = np.isfinite(values)

    assert radii[finite].max() == pytest.approx(min(rx, ry), abs=1.0)


def test_every_bin_averages_over_the_whole_circle(tool):
    """Past the short semi-axis a bin only sees the directions still inside the
    ellipse. On data that varies with angle the old profile read up to 2.0
    where the true full-ring mean is 1.0, and it rose smoothly enough to look
    like a real feature."""
    yy, xx = np.mgrid[0:200, 0:200]
    angle = np.arctan2(yy - 100.0, xx - 100.0)
    image = _load(tool, image=1.0 + np.cos(2 * angle))
    roi = _circle(tool, 100.0, 100.0, 40.0, 12.0)

    _, values, _, _ = tool._roi_profiles(roi, image)
    finite = np.isfinite(values)

    # Skip r = 0: a single pixel is not an average over anything.
    assert np.allclose(values[finite][1:], 1.0, atol=1e-9)


def test_a_thin_ellipse_still_produces_a_profile(tool):
    """floor() of a small semi-axis must not round the profile away."""
    image = _load(tool)
    roi = _circle(tool, 100.0, 100.0, 30.0, 1.4)

    radii, values, _, _ = tool._roi_profiles(roi, image)

    assert len(radii) >= 1
    assert np.any(np.isfinite(values))

"""The 2D viewer's q-space geometry.

This is the block the Q Calibration tool, the XRMS tool and the batch export's
incidence correction all read through, and it was uncovered. The numbers here
are checked against the closed form q = 4*pi*sin(theta/2)/lambda rather than
against the implementation, so a refactor that changes the maths gets caught.
"""

import numpy as np
import pytest

from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

# A transmission geometry with round numbers: 1 A wavelength, 100 um pixels.
ENERGY_EV = 12398.4193      # lambda = 1 A exactly
PIXEL_UM = 100.0
DISTANCE_MM = 1000.0
CENTER = (32.0, 32.0)

BASE = {
    "energy_ev": ENERGY_EV,
    "pixel_um": PIXEL_UM,
    "distance_mm": DISTANCE_MM,
    "center_x": CENTER[0],
    "center_y": CENTER[1],
}


@pytest.fixture
def view(qapp):
    v = ImageView2DEnhanced()
    v.set_data(np.zeros((64, 64)))
    return v


def analytic_q(offset_px: float) -> float:
    """|q| for a detector point ``offset_px`` from the beam centre."""
    lambda_a = 12398.4193 / ENERGY_EV
    metres = offset_px * PIXEL_UM * 1e-6
    two_theta = np.arctan2(metres, DISTANCE_MM * 1e-3)
    return 4.0 * np.pi * np.sin(two_theta / 2.0) / lambda_a


# ---------------------------------------------------------------------------
# Refusing to guess
# ---------------------------------------------------------------------------

def test_no_parameters_means_no_q(view):
    assert view._q_components_at_pixel_float(10.0, 10.0, None) is None
    assert view._q_components_at_pixel_float(10.0, 10.0, {}) is None


@pytest.mark.parametrize("missing", ["energy_ev", "pixel_um", "distance_mm"])
def test_a_missing_or_zero_geometry_term_yields_nothing(view, missing):
    """Better no readout than one computed from a zero the user never set."""
    params = dict(BASE, **{missing: 0.0})
    assert view._q_components_at_pixel_float(10.0, 10.0, params) is None


def test_a_negative_energy_is_refused(view):
    assert view._q_components_at_pixel_float(10.0, 10.0, dict(BASE, energy_ev=-1.0)) is None


# ---------------------------------------------------------------------------
# The maths
# ---------------------------------------------------------------------------

def test_q_is_zero_at_the_beam_centre(view):
    qx, qy, qz = view._q_components_at_pixel_float(*CENTER, BASE)

    assert (qx, qy, qz) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize("offset", [1.0, 5.0, 20.0, 31.0])
def test_the_magnitude_matches_the_scattering_formula(view, offset):
    view.set_q_calibration(BASE)

    got = view._q_value_at_pixel(int(CENTER[0] + offset), int(CENTER[1]))

    assert got == pytest.approx(analytic_q(offset), rel=1e-9)


def test_the_two_axes_are_symmetric(view):
    view.set_q_calibration(BASE)
    along_x = view._q_value_at_pixel(int(CENTER[0] + 10), int(CENTER[1]))
    along_y = view._q_value_at_pixel(int(CENTER[0]), int(CENTER[1] + 10))

    assert along_x == pytest.approx(along_y, rel=1e-12)


def test_the_sign_follows_the_side_of_the_centre(view):
    left = view._q_components_at_pixel_float(CENTER[0] - 10, CENTER[1], BASE)
    right = view._q_components_at_pixel_float(CENTER[0] + 10, CENTER[1], BASE)

    assert left[0] < 0 < right[0]
    assert left[0] == pytest.approx(-right[0])


def test_q_grows_with_distance_from_the_centre(view):
    view.set_q_calibration(BASE)
    values = [view._q_value_at_pixel(int(CENTER[0] + d), int(CENTER[1])) for d in (1, 5, 10, 20)]

    assert values == sorted(values)


def test_a_shorter_wavelength_gives_a_larger_q(view):
    """q scales with 1/lambda, so doubling the energy roughly doubles it."""
    soft = view._q_value_at_pixel  # bound for readability
    view.set_q_calibration(BASE)
    low = soft(int(CENTER[0] + 10), int(CENTER[1]))
    view.set_q_calibration(dict(BASE, energy_ev=2 * ENERGY_EV))
    high = soft(int(CENTER[0] + 10), int(CENTER[1]))

    assert high == pytest.approx(2 * low, rel=1e-3)


# ---------------------------------------------------------------------------
# Incidence correction, and the double-correction trap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "angle, expected",
    [(90.0, 1.0), (30.0, 2.0), (0.0, 1.0), (180.0, 1.0), (-5.0, 1.0)],
)
def test_the_incidence_factor_is_one_over_sine_and_safe_outside_range(angle, expected):
    assert ImageView2DEnhanced._incidence_factor_from_deg(angle) == pytest.approx(expected)


def test_a_grazing_incidence_stretches_the_pixel_offsets(view):
    """At 30 degrees the in-plane axis is foreshortened by sin(theta) = 1/2."""
    plain = view._q_components_at_pixel_float(CENTER[0] + 10, CENTER[1], BASE)
    tilted = view._q_components_at_pixel_float(
        CENTER[0] + 10, CENTER[1],
        dict(BASE, use_incidence=True, incidence_deg=30.0, incidence_axis="X"),
    )

    assert abs(tilted[0]) > abs(plain[0]), "the same pixel maps further out in q"


def test_the_correction_applies_to_the_chosen_axis_only(view):
    params = dict(BASE, use_incidence=True, incidence_deg=30.0, incidence_axis="Y")

    along_x = view._q_components_at_pixel_float(CENTER[0] + 10, CENTER[1], params)
    along_y = view._q_components_at_pixel_float(CENTER[0], CENTER[1] + 10, params)

    assert abs(along_y[1]) > abs(along_x[0])


def test_an_incidence_already_in_the_image_is_not_applied_twice(view):
    """The display can resample the image itself; then q must not stretch again."""
    stretched = view._q_components_at_pixel_float(
        CENTER[0] + 10, CENTER[1],
        dict(BASE, use_incidence=True, incidence_deg=30.0, incidence_axis="X"),
    )
    already = view._q_components_at_pixel_float(
        CENTER[0] + 10, CENTER[1],
        dict(BASE, use_incidence=True, incidence_deg=30.0, incidence_axis="X",
             incidence_applied_in_display=True),
    )

    assert abs(already[0]) < abs(stretched[0])


def test_applying_the_display_correction_records_that_it_was_applied(view):
    view.apply_incidence_display_correction(25.0, axis="Y")

    p = view._q_calibration
    assert p["use_incidence"] is True
    assert p["incidence_deg"] == 25.0
    assert p["incidence_axis"] == "Y"
    assert p["incidence_applied_in_display"] is True, "the flag that prevents double-counting"


def test_an_unknown_incidence_axis_falls_back_to_x(view):
    view.apply_incidence_display_correction(25.0, axis="nonsense")
    assert view._q_calibration["incidence_axis"] == "X"


# ---------------------------------------------------------------------------
# Relabelling the axes in q
# ---------------------------------------------------------------------------

def test_axis_calibration_needs_data(qapp):
    empty = ImageView2DEnhanced()
    assert empty.apply_q_axes_calibration(BASE) is False


def test_axis_calibration_refuses_an_unusable_geometry(view):
    assert view.apply_q_axes_calibration(dict(BASE, distance_mm=0.0)) is False


def test_a_good_geometry_relabels_both_axes_in_q(view):
    assert view.apply_q_axes_calibration(BASE) is True

    assert view.plot_widget.getAxis("bottom").labelText == "qx (1/A)"
    assert view.plot_widget.getAxis("left").labelText == "qy (1/A)"


def test_calibration_leaves_the_image_geometry_alone(view):
    """It maps the tick text; resampling the pixels would be a different feature."""
    before = view.data.copy()

    view.apply_q_axes_calibration(BASE)

    np.testing.assert_array_equal(view.data, before)


def test_clearing_the_calibration_turns_the_readout_off(view):
    view.set_q_calibration(BASE)
    assert view._q_value_at_pixel(40, 32) is not None

    view.set_q_calibration(None)

    assert view._q_calibration is None
    assert view._q_value_at_pixel(40, 32) is None


def test_an_empty_dict_counts_as_no_calibration(view):
    view.set_q_calibration({})
    assert view._q_calibration is None

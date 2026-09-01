"""The data viewer's "X to q" button.

It used to be a checkbox with a permanent energy box beside it in the toolbar.
Now it is one button next to "Set X" that asks for the photon energy at the only
moment the number matters — when the conversion is switched on.
"""

import math

import numpy as np
import pytest
from PyQt6.QtWidgets import QInputDialog

from src.gui.plot_widget_1d_enhanced import (
    DEFAULT_PHOTON_ENERGY_EV,
    PlotWidget1DEnhanced,
)

ANGLES = np.array([1.0, 2.0, 3.0])


@pytest.fixture
def asked(monkeypatch):
    """Answer the energy dialog, and record that it was opened."""
    calls: dict = {"count": 0, "value": 12000.0, "accept": True, "initial": []}

    def fake(_parent, _title, _label, value, *_a, **_k):
        calls["count"] += 1
        calls["initial"].append(value)
        return calls["value"], calls["accept"]

    monkeypatch.setattr(QInputDialog, "getDouble", staticmethod(fake))
    return calls


@pytest.fixture
def viewer(qapp):
    w = PlotWidget1DEnhanced()
    w.set_data(np.array([5.0, 6.0, 7.0]), ANGLES)
    return w


def q_of(angle_deg: float, energy_ev: float) -> float:
    wavelength = 12398 / energy_ev
    return (4 * math.pi / wavelength) * math.sin(math.radians(angle_deg))


# ---------------------------------------------------------------------------
# The control itself
# ---------------------------------------------------------------------------

def test_it_is_a_button_beside_set_x_not_a_checkbox_and_a_field(qapp, tmp_path):
    w = PlotWidget1DEnhanced(opened_files=(tmp_path / "f.h5",))

    assert w.btn_convert_to_q.text() == "X to q"
    assert w.btn_convert_to_q.isCheckable()
    assert not hasattr(w, "input_energy"), "the toolbar no longer parks an energy box"
    assert not hasattr(w, "chk_convert_to_q")


def test_it_is_disabled_until_there_is_an_x_axis(qapp):
    w = PlotWidget1DEnhanced()
    assert not w.btn_convert_to_q.isEnabled()

    w.set_data(np.array([5.0, 6.0, 7.0]), ANGLES)
    assert w.btn_convert_to_q.isEnabled()


# ---------------------------------------------------------------------------
# Switching it on
# ---------------------------------------------------------------------------

def test_switching_on_asks_for_the_energy_and_converts(viewer, asked):
    viewer.btn_convert_to_q.click()

    assert asked["count"] == 1
    assert viewer.photon_energy_ev == 12000.0
    np.testing.assert_allclose(viewer.x_data, [q_of(a, 12000.0) for a in ANGLES])
    np.testing.assert_allclose(viewer.x_data_original, ANGLES)


def test_the_dialog_starts_on_the_default_then_on_the_last_value(viewer, asked):
    viewer.btn_convert_to_q.click()
    viewer.btn_convert_to_q.click()   # off
    viewer.btn_convert_to_q.click()   # on again

    assert asked["initial"] == [DEFAULT_PHOTON_ENERGY_EV, 12000.0]


def test_cancelling_leaves_the_axis_and_the_button_alone(viewer, asked):
    asked["accept"] = False

    viewer.btn_convert_to_q.click()

    assert not viewer.btn_convert_to_q.isChecked(), "nothing converted, so nothing shown as on"
    np.testing.assert_allclose(viewer.x_data, ANGLES)
    assert viewer.photon_energy_ev == DEFAULT_PHOTON_ENERGY_EV


def test_the_axis_is_relabelled_while_converted(viewer, asked):
    viewer.btn_convert_to_q.click()
    assert viewer.plot_widget.getAxis("bottom").labelText == "q (1/A)"


# ---------------------------------------------------------------------------
# Switching it off
# ---------------------------------------------------------------------------

def test_switching_off_restores_the_angles_without_asking_again(viewer, asked):
    viewer.btn_convert_to_q.click()
    viewer.btn_convert_to_q.click()

    assert asked["count"] == 1, "turning it off is not a question"
    np.testing.assert_allclose(viewer.x_data, ANGLES)


def test_resetting_the_button_in_code_never_opens_a_dialog(viewer, asked):
    """set_data clears it when the X goes away; that must stay silent."""
    viewer.btn_convert_to_q.click()
    asked["count"] = 0

    viewer.set_data(np.arange(50, dtype=float))

    assert asked["count"] == 0
    assert not viewer.btn_convert_to_q.isChecked()
    assert not viewer.btn_convert_to_q.isEnabled()


def test_clicking_with_no_x_axis_does_not_ask(qapp, asked):
    w = PlotWidget1DEnhanced()
    w.set_data(np.array([5.0, 6.0, 7.0]))
    w.btn_convert_to_q.setEnabled(True)   # force the case

    w.btn_convert_to_q.click()

    assert asked["count"] == 0
    assert not w.btn_convert_to_q.isChecked()


# ---------------------------------------------------------------------------
# The maths is unchanged
# ---------------------------------------------------------------------------

def test_the_conversion_uses_the_energy_that_was_entered(viewer, asked):
    asked["value"] = 700.0
    viewer.btn_convert_to_q.click()

    np.testing.assert_allclose(viewer.x_data, [q_of(a, 700.0) for a in ANGLES])


def test_a_reconversion_starts_from_the_angles_not_from_q(viewer, asked):
    """Converting twice must not square the transform."""
    viewer.btn_convert_to_q.click()
    viewer.photon_energy_ev = 700.0
    viewer.apply_q_conversion()

    np.testing.assert_allclose(viewer.x_data, [q_of(a, 700.0) for a in ANGLES])

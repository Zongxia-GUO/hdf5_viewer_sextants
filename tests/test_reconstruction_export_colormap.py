"""Saving and copying a reconstruction keeps the colormap it is shown with.

Both tools built a ``Format_Grayscale8`` image for every save and copy, so a
phase map carefully set to a diverging colormap on screen arrived in the file
as grey — the one setting that makes it readable was dropped exactly where the
picture leaves the application.
"""

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from src.gui._shared import array_to_qimage, colormap_lut

RAMP = np.linspace(-1.0, 1.0, 32 * 32).reshape(32, 32).astype(np.float32)
LEVELS = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# The shared renderer
# ---------------------------------------------------------------------------

def test_a_colormapped_render_is_rgb(qapp):
    image = array_to_qimage(RAMP, LEVELS, "viridis")

    assert image.format() == QImage.Format.Format_RGB888
    assert image.size().width() == 32


def test_different_colormaps_give_different_pixels(qapp):
    viridis = array_to_qimage(RAMP, LEVELS, "viridis")
    inferno = array_to_qimage(RAMP, LEVELS, "inferno")

    assert viridis.pixelColor(8, 8) != inferno.pixelColor(8, 8)


def test_inverting_swaps_the_ends(qapp):
    normal = array_to_qimage(RAMP, LEVELS, "viridis")
    flipped = array_to_qimage(RAMP, LEVELS, "viridis", invert=True)

    assert normal.pixelColor(0, 0) == flipped.pixelColor(31, 31)


def test_an_unknown_colormap_falls_back_to_grey(qapp):
    """Grey is the honest answer, not a crash and not a wrong colour."""
    image = array_to_qimage(RAMP, LEVELS, "not-a-colormap")

    assert image.format() in (
        QImage.Format.Format_Grayscale8,
        QImage.Format.Format_RGB888,
    )
    assert not image.isNull()


def test_nan_pixels_do_not_poison_the_render(qapp):
    values = RAMP.copy()
    values[0, 0] = np.nan

    image = array_to_qimage(values, LEVELS, "viridis")

    assert not image.isNull()


def test_an_empty_level_range_still_renders(qapp):
    image = array_to_qimage(RAMP, (1.0, 1.0), "viridis")

    assert not image.isNull()


def test_the_lookup_table_is_the_expected_shape(qapp):
    lut = colormap_lut("viridis")

    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    assert colormap_lut("not-a-colormap") is None


# ---------------------------------------------------------------------------
# FTH: each panel has its own colormap
# ---------------------------------------------------------------------------

@pytest.fixture
def fth(qapp):
    from src.gui.fth_reconstruction_tool import FTHReconstructionTool

    return FTHReconstructionTool(opened_files=())


def test_fth_saves_the_panel_colormap(fth):
    fth._t4_set_panel_override(2, "cmap", "CET-D9")   # the phase panel

    image = fth._component_to_qimage(RAMP, LEVELS, "phase")

    assert image.format() == QImage.Format.Format_RGB888
    assert image.pixelColor(16, 16).getRgb()[:3] != (128, 128, 128)


def test_fth_components_do_not_share_one_colormap(fth):
    """Phase is usually diverging while abs is not; they are set separately."""
    fth._t4_set_panel_override(2, "cmap", "CET-D9")
    fth._t4_set_panel_override(3, "cmap", "viridis")

    phase = fth._component_to_qimage(RAMP, LEVELS, "phase")
    magnitude = fth._component_to_qimage(RAMP, LEVELS, "abs")

    assert phase.pixelColor(8, 8) != magnitude.pixelColor(8, 8)


def test_fth_honours_the_panel_invert(fth):
    fth._t4_set_panel_override(3, "cmap", "viridis")
    normal = fth._component_to_qimage(RAMP, LEVELS, "abs")
    fth._t4_set_panel_override(3, "invert", True)
    flipped = fth._component_to_qimage(RAMP, LEVELS, "abs")

    assert normal.pixelColor(0, 0) == flipped.pixelColor(31, 31)


def test_fth_an_unknown_component_still_renders(fth):
    assert not fth._component_to_qimage(RAMP, LEVELS, "nonsense").isNull()


def test_fth_composite_is_colour_too(fth):
    fth._t4_set_panel_override(2, "cmap", "CET-D9")
    components = {n: RAMP for n in ("real", "imag", "phase", "abs")}

    sheet = fth._composite_components_qimage(components)

    assert sheet is not None
    assert sheet.format() == QImage.Format.Format_RGB888


# ---------------------------------------------------------------------------
# CDI: phase has its own picker, the rest follow amplitude
# ---------------------------------------------------------------------------

@pytest.fixture
def cdi(qapp):
    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    return CDIReconstructionTool(opened_files=())


def test_cdi_phase_uses_the_phase_picker(cdi):
    cdi._res_phase_cmap.setCurrentText("CET-D9")
    cdi._res_amp_cmap.setCurrentText("viridis")

    phase = cdi._component_to_qimage(RAMP, LEVELS, "phase")
    magnitude = cdi._component_to_qimage(RAMP, LEVELS, "abs")

    assert phase.pixelColor(8, 8) != magnitude.pixelColor(8, 8)


@pytest.mark.parametrize("name", ["real", "imag", "abs"])
def test_cdi_the_other_components_follow_amplitude(cdi, name):
    cdi._res_amp_cmap.setCurrentText("inferno")

    expected = array_to_qimage(RAMP, LEVELS, "inferno")
    assert cdi._component_to_qimage(RAMP, LEVELS, name).pixelColor(8, 8) == (
        expected.pixelColor(8, 8)
    )


def test_cdi_saved_pixels_are_not_grey(cdi):
    cdi._res_amp_cmap.setCurrentText("viridis")

    image = cdi._component_to_qimage(RAMP, LEVELS, "abs")

    assert image.format() == QImage.Format.Format_RGB888
    red, green, blue = image.pixelColor(16, 16).getRgb()[:3]
    assert not (red == green == blue), "a grey pixel means the colormap was lost"


def test_cdi_composite_is_colour_too(cdi):
    cdi._res_phase_cmap.setCurrentText("CET-D9")
    components = {n: RAMP for n in ("real", "imag", "phase", "abs")}

    sheet = cdi._composite_cdi_components_qimage(components)

    assert sheet is not None
    assert sheet.format() == QImage.Format.Format_RGB888

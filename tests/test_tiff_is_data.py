"""A TIFF is the data; PNG and JPEG are the picture.

The four save paths each grew their own TIFF writer and disagreed: the batch
rescaled floats to 16 bits, the viewer and the reconstruction tools wrote the
coloured picture, the Q tool wrote true floats. A TIFF that has been colour-
mapped, rotated or rescaled cannot be measured, which is the only reason to
save one — so every path now writes the values as recorded, and the display
settings live in the picture formats.
"""

import pathlib

import numpy as np
import pytest
from PIL import Image

from src.lib_h5.data_exporter import DataExporter

FRAME = (np.arange(32 * 64, dtype=np.float64).reshape(32, 64) * 0.001) - 5.0


def read(path):
    with Image.open(path) as img:
        return img.mode, img.size, np.asarray(img)


# ---------------------------------------------------------------------------
# The shared writer
# ---------------------------------------------------------------------------

def test_float_values_survive_exactly(tmp_path):
    """The old writer normalised to 16 bits, so the numbers in the file were
    no longer the numbers that were measured."""
    target = tmp_path / "raw.tif"

    assert DataExporter.export_raw_tiff(FRAME, target)

    mode, _size, values = read(target)
    assert mode == "F"
    np.testing.assert_allclose(values, FRAME.astype(np.float32))


def test_negative_values_are_kept(tmp_path):
    """A difference image is mostly negative; clipping it would be silent loss."""
    target = tmp_path / "diff.tif"
    DataExporter.export_raw_tiff(FRAME, target)

    _mode, _size, values = read(target)
    assert values.min() < 0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32])
def test_integer_data_keeps_its_type(tmp_path, dtype):
    counts = (np.arange(64, dtype=dtype)).reshape(8, 8)
    target = tmp_path / f"{np.dtype(dtype).name}.tif"

    assert DataExporter.export_raw_tiff(counts, target)

    _mode, _size, values = read(target)
    np.testing.assert_array_equal(values, counts)


def test_a_nan_is_written_not_replaced(tmp_path):
    values = FRAME.copy()
    values[0, 0] = np.nan
    target = tmp_path / "nan.tif"

    DataExporter.export_raw_tiff(values, target)

    _mode, _size, read_back = read(target)
    assert np.isnan(read_back[0, 0])


def test_non_2d_data_is_refused(tmp_path):
    assert DataExporter.export_raw_tiff(np.zeros((2, 3, 4)), tmp_path / "x.tif") is False


# ---------------------------------------------------------------------------
# The batch export
# ---------------------------------------------------------------------------

def batch_export(tmp_path, **settings):
    from src.gui.batch_export import export_image_dataset

    export_image_dataset(
        pathlib.Path("scanx_0050.h5"),
        "scan/data_10",
        FRAME,
        {"output_dir": tmp_path, "colormap": "viridis", **settings},
    )
    return tmp_path


def test_the_batch_tiff_is_the_data(qapp, tmp_path):
    batch_export(tmp_path, save_tiff=True)

    _mode, _size, values = read(next(tmp_path.glob("*.tif")))
    np.testing.assert_allclose(values, FRAME.astype(np.float32))


def test_the_batch_tiff_ignores_the_rotation(qapp, tmp_path):
    """Rotation is a way of looking at the frame, not a change to it."""
    batch_export(tmp_path, save_tiff=True, image_rotation=90)

    _mode, _size, values = read(next(tmp_path.glob("*.tif")))
    assert values.shape == FRAME.shape
    np.testing.assert_allclose(values, FRAME.astype(np.float32))


def test_the_batch_picture_does_apply_the_rotation(qapp, tmp_path):
    batch_export(tmp_path, save_tiff=True, image_rotation=90)

    _mode, size, _values = read(next(tmp_path.glob("*.png")))
    assert size == (FRAME.shape[0], FRAME.shape[1]), "the picture is turned"


def test_the_batch_picture_is_coloured(qapp, tmp_path):
    batch_export(tmp_path, save_tiff=True)

    mode, _size, pixels = read(next(tmp_path.glob("*.png")))
    assert mode == "RGB"
    assert len(set(pixels[10, 10])) > 1, "a grey pixel means the colormap was lost"


def test_no_tiff_unless_it_was_asked_for(qapp, tmp_path):
    batch_export(tmp_path)

    assert list(tmp_path.glob("*.tif")) == []
    assert len(list(tmp_path.glob("*.png"))) == 1


# ---------------------------------------------------------------------------
# The data viewer
# ---------------------------------------------------------------------------

def test_the_viewer_tiff_is_the_data_not_the_picture(qapp, tmp_path):
    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    view = ImageView2DEnhanced()
    view.set_data(FRAME)
    target = tmp_path / "view.tif"

    assert view.export_colormapped_image(target)

    mode, _size, values = read(target)
    assert mode == "F", "not an RGB picture"
    np.testing.assert_allclose(values, FRAME.astype(np.float32), rtol=1e-6)


def test_the_viewer_png_is_still_the_picture(qapp, tmp_path):
    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    view = ImageView2DEnhanced()
    view.set_data(FRAME)
    target = tmp_path / "view.png"

    view.export_colormapped_image(target)

    mode, _size, pixels = read(target)
    assert mode == "RGB"
    assert len(set(pixels[10, 10])) > 1


# ---------------------------------------------------------------------------
# The reconstruction tools
# ---------------------------------------------------------------------------

COMPONENT = np.linspace(-1.0, 1.0, 32 * 32).reshape(32, 32).astype(np.float32)


def test_fth_writes_values_to_a_tiff(qapp, tmp_path):
    from src.gui.fth_reconstruction_tool import FTHReconstructionTool

    tool = FTHReconstructionTool(opened_files=())
    tool._t4_set_panel_override(2, "cmap", "CET-D9")
    target = tmp_path / "phase.tif"

    tool._write_component(COMPONENT, "phase", target)

    mode, _size, values = read(target)
    assert mode == "F"
    np.testing.assert_allclose(values, COMPONENT)


def test_fth_writes_a_picture_to_a_png(qapp, tmp_path):
    from src.gui.fth_reconstruction_tool import FTHReconstructionTool

    tool = FTHReconstructionTool(opened_files=())
    tool._t4_set_panel_override(2, "cmap", "CET-D9")
    target = tmp_path / "phase.png"

    tool._write_component(COMPONENT, "phase", target)

    mode, _size, pixels = read(target)
    assert mode == "RGB"
    assert len(set(pixels[16, 16])) > 1


def test_cdi_writes_values_to_a_tiff(qapp, tmp_path):
    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    tool = CDIReconstructionTool(opened_files=())
    target = tmp_path / "abs.tif"

    tool._write_cdi_component(COMPONENT, "abs", target)

    mode, _size, values = read(target)
    assert mode == "F"
    np.testing.assert_allclose(values, COMPONENT)


def test_cdi_writes_a_picture_to_a_png(qapp, tmp_path):
    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    tool = CDIReconstructionTool(opened_files=())
    tool._res_amp_cmap.setCurrentText("viridis")
    target = tmp_path / "abs.png"

    tool._write_cdi_component(COMPONENT, "abs", target)

    mode, _size, pixels = read(target)
    assert mode == "RGB"
    assert len(set(pixels[16, 16])) > 1


# ---------------------------------------------------------------------------
# The save dialogs offer the formats separately
# ---------------------------------------------------------------------------

def test_every_image_dialog_lists_the_formats_separately():
    """A single combined filter leaves the user typing the extension, and Qt
    saves by extension — a typo there fails silently."""
    gui = pathlib.Path(__file__).resolve().parent.parent / "src" / "gui"
    offenders = []
    for path in sorted(gui.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue          # the shared module names the anti-pattern
            if "*.png *.jpg" in line:
                offenders.append(f"{path.name}:{number}")

    assert offenders == [], "use IMAGE_SAVE_FILTERS from _shared"


@pytest.mark.parametrize(
    "chosen, expected",
    [
        ("PNG Image (*.png)", ".png"),
        ("JPEG Image (*.jpg *.jpeg)", ".jpg"),
        ("TIFF Image, raw values (*.tif *.tiff)", ".tif"),
        ("", ".png"),
    ],
)
def test_the_chosen_filter_supplies_the_extension(chosen, expected):
    from src.gui._shared import extension_for_filter

    assert extension_for_filter(chosen) == expected


def test_the_tiff_entry_says_it_is_data():
    from src.gui._shared import IMAGE_SAVE_FILTERS

    assert "raw values" in IMAGE_SAVE_FILTERS
    assert IMAGE_SAVE_FILTERS.count(";;") == 2, "three separate entries"

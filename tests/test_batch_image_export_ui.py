"""The 2D batch export dialog's settings row.

The incidence-angle correction used to hide behind a toolbar button that asked
two questions in a row and then showed nothing of what it had been told. It is
now a value in the settings, which matters because whatever the preview is
showing is exactly what the export writes.
"""

import pathlib

import numpy as np
import pytest
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QLineEdit

from src.gui.batch_export import BatchExportDialog


@pytest.fixture
def dialog(qapp):
    d = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0049"],
        dataset_path="scan_0049/scan_data/data_10",
        sample_data=np.random.RandomState(0).rand(64, 128),
        data_kind="image",
        preview_x_loader=lambda *a, **k: None,
    )
    d.close()
    return d


# ---------------------------------------------------------------------------
# The angle control
# ---------------------------------------------------------------------------

def test_the_toolbar_no_longer_carries_an_angle_button(dialog):
    assert not dialog.preview_image.btn_q_calibration.isVisibleTo(dialog.preview_image)


def test_the_angle_row_says_what_the_correction_is(dialog):
    box = dialog.chk_save_tiff.parentWidget()
    texts = [w.text() for w in box.findChildren(QLabel)]

    assert "Incident angle correction:" in texts
    assert "Angle:" not in texts


def test_the_angle_is_a_value_in_the_settings(dialog):
    assert isinstance(dialog.spin_angle, QDoubleSpinBox)
    assert isinstance(dialog.cb_angle_axis, QComboBox)
    assert [dialog.cb_angle_axis.itemText(i) for i in range(dialog.cb_angle_axis.count())] == ["X", "Y"]


def test_no_correction_is_applied_until_asked(dialog):
    """A batch must not silently reshape frames nobody asked it to."""
    assert dialog.spin_angle.value() == 0.0
    assert dialog.spin_angle.specialValueText() == "Off"
    assert dialog.settings()["incidence"] is None


def test_typing_an_angle_reaches_the_preview_and_so_the_export(dialog):
    dialog.spin_angle.setValue(30.0)

    incidence = dialog.settings()["incidence"]
    assert incidence["use_incidence"] is True
    assert incidence["incidence_deg"] == 30.0
    assert incidence["incidence_axis"] == "X"


def test_the_axis_choice_reaches_the_preview(dialog):
    dialog.spin_angle.setValue(30.0)
    dialog.cb_angle_axis.setCurrentText("Y")

    assert dialog.settings()["incidence"]["incidence_axis"] == "Y"


def test_returning_to_zero_turns_the_correction_off_again(dialog):
    dialog.spin_angle.setValue(30.0)
    assert dialog.settings()["incidence"] is not None

    dialog.spin_angle.setValue(0.0)

    assert dialog.settings()["incidence"] is None


def test_the_angle_stays_inside_a_meaningful_range(dialog):
    """1/sin(theta) blows up at 0 and 180; the spin box must not offer them."""
    dialog.spin_angle.setValue(500.0)
    assert dialog.spin_angle.value() <= 179.99

    dialog.spin_angle.setValue(-5.0)
    assert dialog.spin_angle.value() >= 0.0


def test_the_correction_is_marked_as_already_in_the_image(dialog):
    """The preview resamples, so the q readout must not stretch a second time."""
    dialog.spin_angle.setValue(30.0)

    assert dialog.settings()["incidence"]["incidence_applied_in_display"] is True


# ---------------------------------------------------------------------------
# The settings rows
# ---------------------------------------------------------------------------

def grid_rows(dialog):
    """Map each row of the settings box to the widgets on it."""
    box = dialog.chk_save_tiff.parentWidget().layout()
    rows: dict[int, list] = {}

    def collect(layout, out):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                out.append(item.widget())
            elif item.layout() is not None:
                collect(item.layout(), out)

    for index in range(box.count()):
        item = box.itemAt(index)
        out = rows.setdefault(index, [])
        if item.widget() is not None:
            out.append(item.widget())
        elif item.layout() is not None:
            collect(item.layout(), out)
    return rows


def row_of(dialog, widget):
    for row, widgets in grid_rows(dialog).items():
        if widget in widgets:
            return row
    raise AssertionError(f"{widget} is not in the settings box")


def label_row(dialog, text):
    for row, widgets in grid_rows(dialog).items():
        if any(isinstance(w, QLabel) and w.text() == text for w in widgets):
            return row
    raise AssertionError(f"no row carries {text!r}")


def test_range_comes_before_folder(dialog):
    assert label_row(dialog, "Range:") < label_row(dialog, "Folder:")


def test_the_layout_matches_the_1d_dialog(dialog):
    """The two dialogs do the same job and must not read differently.

    Destination first — where the files go, then which files they are — and the
    image treatment after it.
    """
    assert label_row(dialog, "Range:") == 0
    assert label_row(dialog, "Folder:") == 1
    assert row_of(dialog, dialog.cb_image_format) == 2
    assert row_of(dialog, dialog.spin_angle) == 3


def test_the_folder_field_takes_the_width_like_the_1d_dialog(dialog):
    """A path is the one field here that can be arbitrarily long."""
    row = dialog.chk_save_tiff.parentWidget().layout().itemAt(1).layout()
    stretches = [row.stretch(i) for i in range(row.count())]

    assert stretches[row.indexOf(dialog.le_output_dir)] == 1
    assert sum(stretches) == 1, "only the path stretches"


def test_the_tiff_box_shares_the_format_row(dialog):
    """Both answer "what files do I get", so they belong together."""
    assert row_of(dialog, dialog.chk_save_tiff) == row_of(dialog, dialog.cb_image_format)


def test_the_format_row_reads_label_then_field_then_box(dialog):
    on_row = grid_rows(dialog)[row_of(dialog, dialog.cb_image_format)]

    assert on_row.index(dialog.cb_image_format) < on_row.index(dialog.chk_save_tiff)


def test_the_angle_field_is_wide_enough_for_its_own_text(dialog):
    """A clipped suffix reads as a typo in the number."""
    dialog.spin_angle.setValue(179.99)
    metrics = dialog.spin_angle.fontMetrics()
    text_width = metrics.horizontalAdvance(dialog.spin_angle.text())

    assert dialog.spin_angle.minimumWidth() > text_width


def test_the_folder_still_drives_the_export(dialog, tmp_path):
    dialog.le_output_dir.setText(str(tmp_path))
    assert dialog.settings()["output_dir"] == tmp_path


def test_the_tiff_box_still_drives_the_export(dialog):
    assert dialog.settings()["save_tiff"] is False
    dialog.chk_save_tiff.setChecked(True)
    assert dialog.settings()["save_tiff"] is True


def test_an_image_batch_has_no_x_controls_on_screen(dialog):
    """They exist only to keep settings() uniform; they must not be shown."""
    box = dialog.chk_save_tiff.parentWidget()
    shown = box.findChildren(QLineEdit)

    assert dialog.le_output_dir in shown
    assert dialog.le_x_path not in shown


def test_the_preview_hides_the_tools_a_batch_cannot_use(dialog):
    view = dialog.preview_image
    for hidden in (view.btn_copy_image, view.btn_save_image,
                   view.btn_roi_line, view.btn_roi_rect, view.btn_ruler):
        assert not hidden.isVisibleTo(view)


def test_the_colormap_choice_still_reaches_the_export(dialog):
    dialog.preview_image.combo_colormap.setCurrentText("inferno")
    assert dialog.settings()["colormap"] == "inferno"


def test_only_checkboxes_the_dialog_owns_are_in_the_settings_box(dialog):
    box = dialog.chk_save_tiff.parentWidget()
    boxes = [c.text() for c in box.findChildren(QCheckBox)]

    assert boxes == ["Save raw TIFF"], "the label says what kind of TIFF it is"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_rotation_sits_beside_the_axis_choice(dialog):
    row = row_of(dialog, dialog.cb_angle_axis)
    on_row = grid_rows(dialog)[row]

    assert on_row.index(dialog.cb_angle_axis) < on_row.index(dialog.cb_image_rotation)


def test_only_quarter_turns_are_offered(dialog):
    """Anything else would resample the frame."""
    labels = [dialog.cb_image_rotation.itemText(i) for i in range(dialog.cb_image_rotation.count())]

    assert labels == ["0°", "90°", "180°", "270°"]
    assert dialog.image_rotation() == 0


def test_choosing_a_rotation_reaches_the_settings(dialog):
    dialog.cb_image_rotation.setCurrentText("90°")

    assert dialog.image_rotation() == 90
    assert dialog.settings()["image_rotation"] == 90


def test_the_preview_turns_with_the_choice(dialog):
    """Preview and file must agree, so both rotate the same source frame."""
    before = dialog.preview_image.data.shape

    dialog.cb_image_rotation.setCurrentText("90°")

    assert dialog.preview_image.data.shape == (before[1], before[0])


def test_the_rotation_is_clockwise_as_the_label_reads(dialog):
    from src.gui.batch_export import rotate_frame

    frame = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Top-left goes to the top-right corner on a clockwise quarter turn.
    assert rotate_frame(frame, 90)[0, 1] == 1.0
    assert rotate_frame(frame, 90)[0, 0] == 3.0


@pytest.mark.parametrize("degrees", [0, 90, 180, 270, 360, 450, -90])
def test_any_multiple_of_ninety_is_handled(degrees):
    from src.gui.batch_export import rotate_frame

    out = rotate_frame(np.zeros((4, 6)), degrees)

    assert out.shape in {(4, 6), (6, 4)}


def test_a_nonsense_rotation_setting_means_none():
    from src.gui.batch_export import rotation_from

    assert rotation_from({}) == 0
    assert rotation_from({"image_rotation": "sideways"}) == 0
    assert rotation_from({"image_rotation": "90°"}) == 90


def test_the_written_file_is_rotated_too(tmp_path):
    from PIL import Image

    from src.gui.batch_export import export_image_dataset

    export_image_dataset(
        pathlib.Path("scanx_0050.h5"),
        "scan_0050/scan_data/data_10",
        np.random.RandomState(0).rand(32, 64),
        {"output_dir": tmp_path, "colormap": "viridis", "image_rotation": 90},
    )

    written = list(tmp_path.glob("*.png"))[0]
    with Image.open(written) as img:
        assert img.size == (32, 64), "a 64-wide frame becomes 32 wide"


def test_without_a_rotation_the_file_keeps_its_shape(tmp_path):
    from PIL import Image

    from src.gui.batch_export import export_image_dataset

    export_image_dataset(
        pathlib.Path("scanx_0050.h5"),
        "scan_0050/scan_data/data_10",
        np.zeros((32, 64)),
        {"output_dir": tmp_path, "colormap": "viridis"},
    )

    with Image.open(list(tmp_path.glob("*.png"))[0]) as img:
        assert img.size == (64, 32)


# ---------------------------------------------------------------------------
# The image format
# ---------------------------------------------------------------------------

def test_png_is_the_default_format(dialog):
    """A detector frame is read for its values; JPEG smears them."""
    assert dialog.image_format().label == "PNG"
    assert dialog.settings()["image_format"] == "PNG"


def test_both_formats_are_offered(dialog):
    labels = [dialog.cb_image_format.itemText(i) for i in range(dialog.cb_image_format.count())]
    assert labels == ["PNG", "JPEG"]


def test_choosing_jpeg_reaches_the_settings(dialog):
    dialog.cb_image_format.setCurrentText("JPEG")

    assert dialog.settings()["image_format"] == "JPEG"
    assert dialog.image_format().suffix == ".jpg"


@pytest.mark.parametrize(
    "chosen, suffix, pil_format",
    [("PNG", ".png", "PNG"), ("JPEG", ".jpg", "JPEG")],
)
def test_the_chosen_format_is_what_lands_on_disk(tmp_path, chosen, suffix, pil_format):
    from PIL import Image

    from src.gui.batch_export import export_image_dataset

    export_image_dataset(
        pathlib.Path("scanx_0050.h5"),
        "scan_0050/scan_data/data_10",
        np.random.RandomState(0).rand(32, 48),
        {"output_dir": tmp_path, "image_format": chosen, "colormap": "viridis"},
    )

    written = list(tmp_path.glob(f"*{suffix}"))
    assert len(written) == 1
    with Image.open(written[0]) as img:
        assert img.format == pil_format


def test_settings_without_a_format_still_write_png(tmp_path):
    """Older callers and the quick paths do not set one."""
    from src.gui.batch_export import export_image_dataset

    export_image_dataset(
        pathlib.Path("scanx_0050.h5"),
        "scan_0050/scan_data/data_10",
        np.zeros((8, 8)),
        {"output_dir": tmp_path, "colormap": "viridis"},
    )

    assert len(list(tmp_path.glob("*.png"))) == 1


def test_the_tiff_is_written_beside_whichever_format_was_chosen(tmp_path):
    from src.gui.batch_export import export_image_dataset

    export_image_dataset(
        pathlib.Path("scanx_0050.h5"),
        "scan_0050/scan_data/data_10",
        np.zeros((8, 8)),
        {"output_dir": tmp_path, "image_format": "JPEG", "colormap": "viridis", "save_tiff": True},
    )

    assert len(list(tmp_path.glob("*.jpg"))) == 1
    assert len(list(tmp_path.glob("*.tif"))) == 1, "the raw values are unaffected by the choice"

"""Batch export: the settings dialog and the unit of work it exports.

Lifted out of ``main_window`` — the window only needs to collect a selection and
hand it over, and keeping the dialog here lets it be built and driven in tests
without a MainWindow.
"""

# Copyright (C) 2023 Dennis Lönard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import os
import pathlib
import re
from typing import Any, NamedTuple

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.export_naming import last_save_directory, short_series_label, x_column_header
from src.gui.table_model import CopyableTableView, DataTable, TableModel
from src.gui.x_target import register_x_target
from src.img.img_path import img_path
from src.lib_h5.data_exporter import DataExporter
# Re-exported: these moved to src.lib_h5.stacks so the reconstruction tools can
# slice a stack the same way without importing the batch export to do it. Call
# sites here and in the viewer are unchanged.
from src.lib_h5.stacks import (          # noqa: F401
    axis_label,
    slice_axis_from,
    slice_count,
    take_slice,
)
from src.lib_h5.table_format import (
    DEFAULT_TABLE_FORMAT_KEY,
    TableFormat,
    format_labels,
    get_table_format,
)
from src.lib_h5.table_writer import columns_from_2d, write_table


def compact_combo(combo: QComboBox, min_chars: int = 16) -> QComboBox:
    """Stop a combo from widening its dialog to fit the longest entry.

    Dataset keys and dialect labels are long; by default a QComboBox asks for
    enough width to show all of the widest one, which is what made the export
    dialogs so wide. The popup still shows the full text.
    """
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    return combo


def elided_label(text: str, max_width: int = 200) -> QLabel:
    """An informational label that never widens the dialog.

    Dataset paths and scan lists are unbounded, and a plain QLabel makes the
    window as wide as its longest line. This clamps the width and moves the full
    text into the tooltip.
    """
    label = QLabel()
    label.setMaximumWidth(max_width)
    label.setToolTip(text)
    label.setText(label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, max_width))
    return label


# The Plot page's twin of the Export page's Single/Combined choice. Overlaying
# is the point of a batch plot, so it is the default.
PLOT_MODE_COMBINED = "Figure combined"
PLOT_MODE_PER_SCAN = "Figure per scan"
PLOT_OUTPUT_MODES = (PLOT_MODE_COMBINED, PLOT_MODE_PER_SCAN)

# Image formats for a written figure. PNG first: a plot is line art, which is
# what JPG's block transform handles worst.
PLOT_FORMATS = ("PNG", "JPG")
DEFAULT_PLOT_FORMAT = PLOT_FORMATS[0]


# Caps for the on-screen table only: it has to show the shape of the result,
# not carry it. Every path that produces output passes None instead.
PREVIEW_MAX_TARGETS = 5
PREVIEW_MAX_ROWS = 200


class ImageFormat(NamedTuple):
    """A written image format: what the user picks, and what PIL is told."""

    label: str
    name: str
    suffix: str


# For the colormapped 2D export. PNG is lossless and the default: a detector
# frame is being read for its values, and JPEG's block transform would smear
# the very features the colormap is there to show. JPEG is offered anyway for
# the case where a small file matters more.
IMAGE_FORMATS = (
    ImageFormat("PNG", "PNG", ".png"),
    ImageFormat("JPEG", "JPEG", ".jpg"),
)
DEFAULT_IMAGE_FORMAT = IMAGE_FORMATS[0]


# Quarter turns, because anything else would resample the frame. Clockwise, the
# direction the label reads as.
IMAGE_ROTATIONS = ("0°", "90°", "180°", "270°")
DEFAULT_IMAGE_ROTATION = IMAGE_ROTATIONS[0]


def rotation_from(export_settings: dict[str, Any]) -> int:
    """The chosen clockwise rotation in degrees, or 0 when there is none."""
    text = str(export_settings.get("image_rotation", "") or "").strip().rstrip("°")
    try:
        degrees = int(float(text))
    except ValueError:
        return 0
    return degrees % 360 // 90 * 90


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Turn ``frame`` clockwise by a multiple of 90 degrees.

    Applied to the source frame in both the preview and the writer, so the two
    cannot disagree — and applied *before* the incidence correction, which then
    means the same axis on screen as it does in the file.
    """
    quarter_turns = int(degrees) % 360 // 90
    if quarter_turns == 0:
        return frame
    # np.rot90 turns counter-clockwise, so negate to read as the label does.
    return np.rot90(frame, k=-quarter_turns)


def image_format_from(export_settings: dict[str, Any]) -> ImageFormat:
    """Read the chosen image format, tolerating settings dicts without one."""
    wanted = str(export_settings.get("image_format", "") or "").upper()
    for fmt in IMAGE_FORMATS:
        if fmt.label == wanted:
            return fmt
    return DEFAULT_IMAGE_FORMAT


class BatchTarget(NamedTuple):
    """One dataset to export: a scan file crossed with one Y dataset path."""

    file_path: pathlib.Path
    scan_num: str
    ds_path: str
    label: str


class BatchProgress:
    """A small progress window for a batch running on the GUI thread.

    The writers are synchronous — h5py reads and matplotlib renders, neither of
    which moves off this thread without a much larger change — so the event loop
    has to be pumped between items for the bar to move and Cancel to be heard.

    Used as a context manager so the window closes even when a writer raises.
    """

    def __init__(self, parent, total: int, title: str, label: str) -> None:
        self._dialog = QProgressDialog(label, "Cancel", 0, max(0, total), parent)
        self._dialog.setWindowTitle(title)
        # Shown straight away: an export the user asked to watch should not
        # depend on how long Qt guesses it will take.
        self._dialog.setMinimumDuration(0)
        # Blocks its own window only, so the rest of the application stays live.
        self._dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        self._done = 0

    def __enter__(self) -> "BatchProgress":
        self._dialog.setValue(0)
        QApplication.processEvents()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._dialog.close()
        self._dialog.deleteLater()
        QApplication.processEvents()

    @property
    def cancelled(self) -> bool:
        return bool(self._dialog.wasCanceled())

    def advance(self, label: str = "") -> bool:
        """Count one item done. Returns False once the user has cancelled."""
        self._done += 1
        if label:
            self._dialog.setLabelText(label)
        self._dialog.setValue(self._done)
        QApplication.processEvents()
        return not self.cancelled


class BatchExportDialog(QDialog):
    """Settings dialog for single-file and batch dataset export.

    Stays open when the export runs: a batch is usually one of several, and
    closing the settings on OK meant re-entering the whole range to write the
    next one. Progress goes to a :class:`BatchProgress` window instead, and
    closing this dialog is left to the user.
    """

    #: Emitted when the user asks for the export or the figure to be written.
    #: Replaces ``accepted``, which carried "and close" along with it.
    export_requested = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        default_dir: pathlib.Path,
        scan_numbers: list[str],
        dataset_path: str,
        sample_data: np.ndarray,
        data_kind: str,
        preview_x_loader,
        preview_curve_loader=None,
        default_x_path: str = "",
        warning: str = "",
        warning_detail: str = "",
        slice_axis: int = 0,
        slice_index: int = 0,
    ) -> None:
        super().__init__(parent)
        self._default_x_path = default_x_path
        self._scan_numbers = list(scan_numbers)
        self.setWindowTitle("Batch Export / Plot")
        self.setWindowIcon(QIcon(str(pathlib.Path(img_path(), "sextants.ico"))))
        # Left as a plain QDialog, which is Qt.WindowType.Dialog: owned by the
        # window it was opened from, kept above it, and going behind other
        # programs along with it. That ownership is what keeps this dialog in
        # front while you go on clicking in the tree — the one thing being
        # non-modal is for. Opened with no parent it owned nothing, so the
        # first click into the tree buried it. StaysOnTopHint would also keep
        # it visible, by pinning it over every other program on the desktop,
        # which is the behaviour that was removed here once already.
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        # Square: the preview sits above the controls, so extra width buys nothing.
        self.resize(680, 680)
        self._sample_data = np.asarray(sample_data)
        self._data_kind = data_kind
        self._preview_x_loader = preview_x_loader
        self._preview_curve_loader = preview_curve_loader
        # Where the viewer left the stack, so the dialog opens on the frame that
        # was on screen rather than starting again at the first one.
        self._initial_slice_axis = max(0, int(slice_axis))
        self._initial_slice_index = max(0, int(slice_index))

        root_layout = QVBoxLayout(self)
        if data_kind == "curve":
            self._init_curve_export_ui(
                root_layout, default_dir, scan_numbers, dataset_path,
                warning=warning, warning_detail=warning_detail,
            )
            return
        if data_kind == "image":
            self._init_image_export_ui(root_layout, default_dir, scan_numbers, dataset_path)
            return

        body_layout = QHBoxLayout()
        root_layout.addLayout(body_layout, stretch=1)
        left_panel = QWidget()
        left_panel.setMaximumWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        body_layout.addWidget(left_panel)

        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)
        out_row = QHBoxLayout()
        self.le_output_dir = QLineEdit(str(default_dir))
        btn_choose_dir = QPushButton("...")
        btn_choose_dir.setFixedWidth(32)
        btn_choose_dir.clicked.connect(self._choose_output_dir)
        out_row.addWidget(self.le_output_dir, stretch=1)
        out_row.addWidget(btn_choose_dir)
        out_form.addRow("Folder:", out_row)
        out_form.addRow("Dataset:", QLabel(dataset_path))
        out_form.addRow("Scans:", QLabel(", ".join(scan_numbers[:8]) + (" ..." if len(scan_numbers) > 8 else "")))
        out_form.addRow("Detected:", QLabel("1D / small table" if data_kind == "curve" else "2D image"))
        left_layout.addWidget(out_group)

        self.x_group = QGroupBox("1D / Small Table Export")
        x_form = QFormLayout(self.x_group)
        self.cb_x_mode = QComboBox()
        self.cb_x_mode.addItems([
            "Index",
            "Each file uses own X dataset",
            "Use one shared X dataset",
        ])
        x_form.addRow("X values:", self.cb_x_mode)
        self.le_x_path = QLineEdit(self._default_x_path)
        self.le_x_path.setPlaceholderText("Optional X dataset path")
        x_form.addRow("X path:", self.le_x_path)
        self.le_shared_x_scan = QLineEdit(scan_numbers[0] if scan_numbers else "")
        self.le_shared_x_scan.setPlaceholderText("Scan number used for shared X")
        x_form.addRow("Shared scan:", self.le_shared_x_scan)
        self.cb_x_mode.currentTextChanged.connect(self._refresh_curve_preview)
        self.le_x_path.textChanged.connect(self._refresh_curve_preview)
        self.le_shared_x_scan.textChanged.connect(self._refresh_curve_preview)
        left_layout.addWidget(self.x_group)

        self.img_group = QGroupBox("2D Image Export")
        img_form = QFormLayout(self.img_group)
        self.cb_colormap = QComboBox()
        self.cb_colormap.addItems([
            "viridis",
            "inferno",
            "cividis",
            "turbo",
            "CET-L9",
            "CET-L1",
            "CET-L4",
            "CET-R4",
            "CET-D1",
            "CET-D9",
        ])
        img_form.addRow("Colormap:", self.cb_colormap)
        self.cb_contrast = QComboBox()
        self.cb_contrast.addItems(["Auto histogram", "Full range"])
        img_form.addRow("Contrast:", self.cb_contrast)
        self.chk_save_tiff = QCheckBox("Also save TIFF")
        img_form.addRow("", self.chk_save_tiff)
        self.cb_colormap.currentTextChanged.connect(self._refresh_image_preview)
        self.cb_contrast.currentTextChanged.connect(self._refresh_image_preview)
        left_layout.addWidget(self.img_group)
        left_layout.addStretch()

        self.preview_stack = QStackedWidget()
        body_layout.addWidget(self.preview_stack, stretch=1)

        self.preview_table = CopyableTableView()
        self.preview_stack.addWidget(self.preview_table)

        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
        self.preview_image = ImageView2DEnhanced(self)
        self.preview_image.btn_copy_image.hide()
        self.preview_image.btn_save_image.hide()
        self.preview_image.btn_q_calibration.setText("Angle")
        self.preview_image.btn_q_calibration.setFixedWidth(46)
        self.preview_image.btn_q_calibration.setToolTip("Apply incidence angle correction to preview")
        try:
            self.preview_image.btn_q_calibration.clicked.disconnect()
        except TypeError:
            pass
        self.preview_image.btn_q_calibration.clicked.connect(self._apply_preview_angle_correction)
        self.preview_stack.addWidget(self.preview_image)

        self.x_group.setVisible(data_kind == "curve")
        self.img_group.setVisible(data_kind == "image")
        self.preview_stack.setCurrentWidget(self.preview_table if data_kind == "curve" else self.preview_image)
        if data_kind == "curve":
            self._refresh_curve_preview()
        else:
            self._refresh_image_preview()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.export_requested.emit)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

    def _init_image_export_ui(
        self,
        root_layout: QVBoxLayout,
        default_dir: pathlib.Path,
        scan_numbers: list[str],
        dataset_path: str,
    ) -> None:
        """Build the 2D export dialog: image preview above, export controls below."""
        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

        self.preview_image = ImageView2DEnhanced(self)
        self.preview_image.btn_copy_image.hide()
        self.preview_image.btn_save_image.hide()
        self.preview_image.btn_roi_line.hide()
        self.preview_image.btn_roi_rect.hide()
        self.preview_image.btn_ruler.hide()
        if hasattr(self.preview_image, "label_roi"):
            self.preview_image.label_roi.hide()
        # The angle correction lives in the settings below as a value you can
        # see, not behind a toolbar button that asked two questions in a row and
        # then showed nothing of what it had been told.
        self.preview_image.btn_q_calibration.hide()
        root_layout.addWidget(self.preview_image, stretch=1)

        # Same shape as the 1D dialog's Batch box: full-width rows, the field
        # that can be long taking the slack, and the paired settings sharing a
        # row. Two dialogs that do the same job should not be read differently.
        bottom = QGroupBox("Export")
        bottom_layout = QVBoxLayout(bottom)

        row_info = QHBoxLayout()
        row_info.addWidget(QLabel("Range:"))
        row_info.addWidget(elided_label(", ".join(scan_numbers) or "-"))
        row_info.addSpacing(10)
        row_info.addWidget(QLabel("Dataset:"))
        row_info.addWidget(elided_label(dataset_path))
        row_info.addStretch()
        bottom_layout.addLayout(row_info)

        self._build_slice_controls(bottom_layout)

        row_output = QHBoxLayout()
        self.le_output_dir = QLineEdit(str(default_dir))
        self.le_output_dir.setToolTip("Where the exported images are written")
        btn_choose_dir = QPushButton("...")
        btn_choose_dir.setFixedWidth(32)
        btn_choose_dir.clicked.connect(self._choose_output_dir)
        row_output.addWidget(QLabel("Folder:"))
        row_output.addWidget(self.le_output_dir, stretch=1)
        row_output.addWidget(btn_choose_dir)
        bottom_layout.addLayout(row_output)

        # Where the files go and which files they are, then how the picture is
        # made: the destination first, the image treatment after it.
        self.cb_image_format = QComboBox()
        self.cb_image_format.addItems([f.label for f in IMAGE_FORMATS])
        self.cb_image_format.setToolTip(
            "PNG keeps every value the colormap encodes; JPEG is smaller but "
            "smears fine detector features"
        )
        self.chk_save_tiff = QCheckBox("Save raw TIFF")
        self.chk_save_tiff.setToolTip(
            "Also write the values as recorded — no colormap, no rotation, no rescaling"
        )
        row_format = QHBoxLayout()
        row_format.addWidget(QLabel("Format:"))
        row_format.addWidget(compact_combo(self.cb_image_format), stretch=1)
        row_format.addSpacing(10)
        row_format.addWidget(self.chk_save_tiff)
        bottom_layout.addLayout(row_format)

        self._build_angle_controls()
        row_angle = QHBoxLayout()
        row_angle.addWidget(QLabel("Incident angle correction:"))
        row_angle.addWidget(self.spin_angle)
        row_angle.addSpacing(10)
        row_angle.addWidget(QLabel("Axis:"))
        row_angle.addWidget(self.cb_angle_axis, stretch=1)
        row_angle.addSpacing(10)
        row_angle.addWidget(QLabel("Rotate:"))
        row_angle.addWidget(self.cb_image_rotation, stretch=1)
        bottom_layout.addLayout(row_angle)

        root_layout.addWidget(bottom)

        # Outside the box: the box describes the export, the buttons act on it.
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.export_requested.emit)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

        # These controls are not used by image export but keep settings() simple.
        self.cb_x_mode = QComboBox()
        self.cb_x_mode.addItem("Index")
        self.le_x_path = QLineEdit()
        self.le_shared_x_scan = QLineEdit()

        self._refresh_image_preview()

    def _init_curve_export_ui(
        self,
        root_layout: QVBoxLayout,
        default_dir: pathlib.Path,
        scan_numbers: list[str],
        dataset_path: str,
        warning: str = "",
        warning_detail: str = "",
    ) -> None:
        """Build the 1D dialog: an Export page and a Plot page over one selection.

        Each page is only its preview — the table, or the figure. Everything
        that says where the batch goes lives in one ``Batch`` box below them:
        the scans, the X axis, the folder, and an Output/Format pair that
        follows whichever page is in front. Both pages therefore answer the same
        question in the same place, and only the destination differs.
        """
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_export_page(), "Export")
        self.tabs.addTab(self._build_plot_page(), "Plot")
        root_layout.addWidget(self.tabs, stretch=1)

        bottom = QGroupBox("Batch")
        bottom_layout = QVBoxLayout(bottom)

        # What the batch *is* comes before where it goes.
        row_info = QHBoxLayout()
        row_info.addWidget(QLabel("Range:"))
        row_info.addWidget(elided_label(", ".join(scan_numbers) or "-"))
        row_info.addSpacing(10)
        row_info.addWidget(QLabel("Dataset:"))
        row_info.addWidget(elided_label(dataset_path))
        row_info.addStretch()
        bottom_layout.addLayout(row_info)

        # A scan number matching several files is allowed — they keep their own
        # names, so nothing is overwritten — but the batch then holds twice what
        # was asked for. Said here rather than in a message box on the way in:
        # this dialog's own Export button is already the confirmation, and a
        # box that appears on every export stops being read.
        self.lbl_warning = QLabel(warning)
        self.lbl_warning.setStyleSheet("color: #b26a00;")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setToolTip(warning_detail or warning)
        self.lbl_warning.setVisible(bool(warning))
        bottom_layout.addWidget(self.lbl_warning)

        row_output = QHBoxLayout()
        self.le_output_dir = QLineEdit(str(default_dir))
        btn_choose_dir = QPushButton("...")
        btn_choose_dir.setFixedWidth(32)
        btn_choose_dir.clicked.connect(self._choose_output_dir)
        row_output.addWidget(QLabel("Folder:"))
        row_output.addWidget(self.le_output_dir, stretch=1)
        row_output.addWidget(btn_choose_dir)
        bottom_layout.addLayout(row_output)

        row_x = QHBoxLayout()
        self.chk_export_x = QCheckBox("Export X")
        self.chk_export_x.setChecked(True)
        # Pre-filled from the tree's "Set as X" so the same path does not have to
        # be dragged in again for every export.
        self.le_x_path = QLineEdit(self._default_x_path)
        self.le_x_path.setPlaceholderText("Drag or type X dataset path")
        self.le_x_path.setAcceptDrops(True)
        self.le_x_path.dragEnterEvent = self._x_path_drag_enter
        self.le_x_path.dropEvent = self._x_path_drop
        self.chk_share_x = QCheckBox("Share X")
        self.le_shared_x_scan = QLineEdit(scan_numbers[0] if scan_numbers else "")
        self.le_shared_x_scan.setPlaceholderText("Shared scan")
        row_x.addWidget(self.chk_export_x)
        row_x.addWidget(self.le_x_path, stretch=1)
        row_x.addWidget(self.chk_share_x)
        row_x.addWidget(self.le_shared_x_scan)
        bottom_layout.addLayout(row_x)

        # One Output/Format row, swapped with the page: the two destinations ask
        # the same two questions and answering them in one place keeps the
        # dialog from growing a second, near-identical control block.
        self.output_stack = QStackedWidget()
        self.output_stack.addWidget(self._build_export_output_row())
        self.output_stack.addWidget(self._build_plot_output_row())
        bottom_layout.addWidget(self.output_stack)

        root_layout.addWidget(bottom)

        # Outside the box: the box describes the batch, the buttons act on it.
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # Copy has no counterpart on the Export page — a table goes to a file,
        # not to the clipboard — so it is added here and shown with the figure.
        self.btn_copy_figure = self.buttons.addButton(
            "Copy", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.btn_copy_figure.setToolTip("Copy the figure to the clipboard as an image")
        self.btn_copy_figure.setAutoDefault(False)
        self.btn_copy_figure.clicked.connect(self.plot_panel.copy_figure)
        self.buttons.accepted.connect(self.export_requested.emit)
        self.buttons.rejected.connect(self.reject)
        root_layout.addWidget(self.buttons)

        # The button and the Output/Format row say what the page in front will
        # do, so OK is never a guess between a folder of tables and one of
        # figures.
        self.tabs.currentChanged.connect(self._sync_to_page)
        self._sync_to_page()

        # These controls are not used by curve export but keep settings() simple.
        self.chk_save_tiff = QCheckBox()
        self.cb_x_mode = QComboBox()
        self.cb_x_mode.addItem("Index")
        self.cb_colormap = QComboBox()
        self.cb_colormap.addItem("viridis")
        self.cb_contrast = QComboBox()
        self.cb_contrast.addItem("Auto histogram")

        # Only the curve dialog has a usable X field, so only it offers itself
        # to the tree's "Set X".
        register_x_target(self)

        self.chk_export_x.stateChanged.connect(self._refresh_curve_preview)
        self.chk_share_x.stateChanged.connect(self._refresh_curve_preview)
        self.cb_curve_output_mode.currentTextChanged.connect(self._refresh_curve_preview)
        self.le_x_path.textChanged.connect(self._refresh_curve_preview)
        self.le_shared_x_scan.textChanged.connect(self._refresh_curve_preview)
        self._refresh_curve_preview()

    def _build_export_page(self) -> QWidget:
        """The table that will be written. Where it goes lives in the Batch box."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_table = CopyableTableView()
        layout.addWidget(self.preview_table, stretch=1)
        return page

    def _build_export_output_row(self) -> QWidget:
        """How the tables are written: one file each or one combined, and which dialect."""
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)

        self.cb_curve_output_mode = QComboBox()
        self.cb_curve_output_mode.addItems(["Single files", "Combined file"])
        row.addWidget(QLabel("Output:"))
        row.addWidget(compact_combo(self.cb_curve_output_mode), stretch=1)
        row.addWidget(QLabel("Format:"))
        row.addWidget(self._build_table_format_combo(), stretch=1)
        return row_widget

    def _build_plot_output_row(self) -> QWidget:
        """How the figures are written: one per scan or one combined, and which image format."""
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)

        self.cb_plot_output_mode = QComboBox()
        self.cb_plot_output_mode.addItems(PLOT_OUTPUT_MODES)
        self.cb_plot_output_mode.setToolTip(
            "Figure combined: the whole range overlaid, exactly as previewed.\n"
            "Figure per scan: one image file for each scan."
        )
        self.cb_plot_format = QComboBox()
        self.cb_plot_format.addItems(PLOT_FORMATS)
        self.cb_plot_format.setToolTip(
            "PNG keeps every pixel; JPG is smaller but blurs thin lines and text."
        )
        row.addWidget(QLabel("Output:"))
        row.addWidget(compact_combo(self.cb_plot_output_mode), stretch=1)
        row.addWidget(QLabel("Format:"))
        row.addWidget(compact_combo(self.cb_plot_format), stretch=1)

        # The mode reshapes the figure itself, not just the wording below it.
        self.cb_plot_output_mode.currentTextChanged.connect(self._refresh_plot_preview)
        self.cb_plot_format.currentTextChanged.connect(self._refresh_plot_summary)
        return row_widget

    def _build_plot_page(self) -> QWidget:
        """The figure itself, drawn from the same table as the export.

        The real panel rather than a button that opens one: the point of the
        second page is to see the batch before committing to it, and a preview
        you have to launch is not a preview.
        """
        from src.gui.plot_dialog import PlotPanel

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # The X below the tabs is the authority here, so the panel neither shows
        # its own X field nor reaches for the remembered one behind it.
        self.plot_panel = PlotPanel(page, allow_x_drop=False, adopt_remembered_x=False)
        layout.addWidget(self.plot_panel, stretch=1)

        self.lbl_plot_summary = QLabel()
        self.lbl_plot_summary.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_plot_summary)
        return page

    def _sync_to_page(self) -> None:
        """Point the buttons and the Output/Format row at the page in front."""
        plotting = self.action() == "plot"
        button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            # Named for what it produces: a folder of tables, or a figure file.
            button.setText("Save Figure" if plotting else "Export")
        if hasattr(self, "btn_copy_figure"):
            self.btn_copy_figure.setVisible(plotting)
        if hasattr(self, "output_stack"):
            self.output_stack.setCurrentIndex(1 if plotting else 0)

    def action(self) -> str:
        """What OK will do: ``"export"`` or ``"plot"``, following the open page."""
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return "export"
        return "plot" if tabs.tabText(tabs.currentIndex()) == "Plot" else "export"

    def _refresh_plot_page(self, headers: list[str], columns: list) -> None:
        """Redraw the embedded figure from the very table the export previews."""
        if not hasattr(self, "plot_panel"):
            return
        from src.gui.plot_series import series_from_table

        self.plot_panel.set_series(series_from_table(headers, columns))
        self._refresh_plot_summary()

    def _refresh_plot_summary(self) -> None:
        """Say how many files OK will write, since only one figure is previewed."""
        if not hasattr(self, "lbl_plot_summary"):
            return
        count = len(self.plot_panel.all_series())
        suffix = self.plot_format().lower()
        if self.plot_output_mode() == PLOT_MODE_PER_SCAN:
            figures = max(len(self._scan_numbers), 1)
            self.lbl_plot_summary.setText(
                f"Writes {figures} .{suffix} file(s), one per scan. "
                "Above is the first of them."
            )
        else:
            self.lbl_plot_summary.setText(
                f"{count} curve(s). Writes one .{suffix} file of exactly this figure."
            )

    def plot_output_mode(self) -> str:
        """``"Figure combined"`` or ``"Figure per scan"``."""
        combo = getattr(self, "cb_plot_output_mode", None)
        return combo.currentText() if combo is not None else PLOT_MODE_COMBINED

    def plot_format(self) -> str:
        """``"PNG"`` or ``"JPG"``."""
        combo = getattr(self, "cb_plot_format", None)
        return combo.currentText() if combo is not None else DEFAULT_PLOT_FORMAT

    def image_format(self) -> ImageFormat:
        """The format the colormapped 2D export writes."""
        combo = getattr(self, "cb_image_format", None)
        if combo is None:
            return DEFAULT_IMAGE_FORMAT
        return image_format_from({"image_format": combo.currentText()})

    def image_rotation(self) -> int:
        """Clockwise rotation applied to the written image, in degrees."""
        combo = getattr(self, "cb_image_rotation", None)
        if combo is None:
            return 0
        return rotation_from({"image_rotation": combo.currentText()})

    def _build_table_format_combo(self) -> QComboBox:
        """Build the tabular-dialect selector, restoring the last used choice."""
        self.cb_table_format = QComboBox()
        self.cb_table_format.addItems(format_labels())
        self.cb_table_format.setToolTip(
            "Excel splits columns by the machine's regional list separator, not by the file.\n\n"
            "TXT   tab — safe in every locale\n"
            "CSV   en/US Excel, pandas, numpy\n"
            "CSV2  fr/de/es/it Excel"
        )
        stored = QSettings().value("export/table_format", DEFAULT_TABLE_FORMAT_KEY)
        self.cb_table_format.setCurrentText(get_table_format(str(stored)).label)
        return compact_combo(self.cb_table_format)

    def table_format(self) -> TableFormat:
        """The tabular dialect chosen in this dialog."""
        combo = getattr(self, "cb_table_format", None)
        if combo is None:
            return get_table_format(DEFAULT_TABLE_FORMAT_KEY)
        return get_table_format(combo.currentText())

    def set_x_dataset(self, key: str) -> bool:
        """Take an X dataset handed over by the tree's ``Set X``.

        The dialog addresses a path inside each matched scan file, so only the
        dataset half of a ``file::path`` key is kept.
        """
        self.le_x_path.setText(key.split("::", 1)[-1].strip("/"))
        self.chk_export_x.setChecked(True)
        return True

    def _x_path_drag_enter(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData().hasText():
            event.acceptProposedAction()

    def _x_path_drop(self, event: QDropEvent | None) -> None:
        if event is None or not event.mimeData().hasText():
            return
        text = event.mimeData().text().strip()
        if "::" in text:
            _file_path, text = text.split("::", 1)
        self.le_x_path.setText(text.strip("/"))
        event.acceptProposedAction()

    def _choose_output_dir(self) -> None:
        start = self.le_output_dir.text().strip() or str(last_save_directory())
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", start)
        if folder:
            self.le_output_dir.setText(folder)

    def settings(self) -> dict[str, Any]:
        return {
            "output_dir": pathlib.Path(self.le_output_dir.text().strip() or pathlib.Path.home()),
            "export_x": self.chk_export_x.isChecked() if self._data_kind == "curve" else False,
            "share_x": self.chk_share_x.isChecked() if self._data_kind == "curve" else False,
            "curve_output_mode": (
                self.cb_curve_output_mode.currentText() if self._data_kind == "curve" else "Single files"
            ),
            "x_mode": self.cb_x_mode.currentText(),
            "x_path": self.le_x_path.text().strip().strip("/"),
            "shared_x_scan": self.le_shared_x_scan.text().strip(),
            "colormap": (
                self.preview_image.combo_colormap.currentText() if self._data_kind == "image"
                else self.cb_colormap.currentText()
            ),
            "invert": self.preview_image.chk_invert.isChecked() if self._data_kind == "image" else False,
            "scale": self.preview_image.combo_scale.currentText() if self._data_kind == "image" else "Linear",
            "levels": self.preview_image.histogram.getLevels() if self._data_kind == "image" else None,
            "incidence": getattr(self.preview_image, "_q_calibration", None) if self._data_kind == "image" else None,
            "contrast": "Preview levels" if self._data_kind == "image" else self.cb_contrast.currentText(),
            "save_tiff": self.chk_save_tiff.isChecked(),
            "image_format": self.image_format().label,
            "image_rotation": self.image_rotation(),
            "slice_axis": self.slice_axis(),
            "table_format": self.table_format(),
            "plot_output_mode": self.plot_output_mode(),
            "plot_format": self.plot_format(),
        }

    @staticmethod
    def _curve_columns(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data).squeeze()
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        return arr.reshape(arr.shape[0], -1)

    def _build_preview(
        self,
        settings: dict[str, Any],
        *,
        max_targets: int | None = PREVIEW_MAX_TARGETS,
        max_rows: int | None = PREVIEW_MAX_ROWS,
    ) -> tuple[np.ndarray, list[str]]:
        """The table ``settings`` would produce, from the loader or the sample."""
        if callable(self._preview_curve_loader):
            preview, headers = self._preview_curve_loader(
                settings, max_targets=max_targets, max_rows=max_rows
            )
            return np.asarray(preview), list(headers)

        y_columns = self._curve_columns(self._sample_data)
        x_data = self._preview_x_loader(settings, y_columns.shape[0])
        n = y_columns.shape[0] if max_rows is None else min(max_rows, y_columns.shape[0])
        if x_data is None:
            return y_columns[:n], [f"Y_{i + 1}" for i in range(y_columns.shape[1])]
        return (
            np.column_stack([x_data[:n], y_columns[:n]]),
            ["X"] + [f"Y_{i + 1}" for i in range(y_columns.shape[1])],
        )

    def plot_settings(self) -> dict[str, Any]:
        """Settings as the Plot page means them.

        The page's own Output decides whether the figure holds the whole range
        or a single scan; reading the Export page's choice would make one page's
        control silently reshape the other's preview.
        """
        settings = self.settings()
        settings["curve_output_mode"] = (
            "Single files" if self.plot_output_mode() == PLOT_MODE_PER_SCAN else "Combined file"
        )
        return settings

    def _refresh_curve_preview(self) -> None:
        if self._data_kind != "curve":
            return
        try:
            preview, headers = self._build_preview(self.settings())
            self.preview_table.setModel(DataTable(preview, column_names=headers))
        except Exception as exc:
            self.preview_table.setModel(TableModel(header=["Error"]))
            logging.warning("Failed to refresh batch export table preview: %s", exc)
        self._refresh_plot_preview()

    def _refresh_plot_preview(self) -> None:
        """Redraw the figure from the table the Plot page's own Output implies."""
        if self._data_kind != "curve" or not hasattr(self, "plot_panel"):
            return
        try:
            # Uncapped: this figure is the specification for the written one,
            # so it has to hold every scan at full length.
            preview, headers = self._build_preview(
                self.plot_settings(), max_targets=None, max_rows=None
            )
        except Exception as exc:
            logging.warning("Failed to refresh the batch plot preview: %s", exc)
            return
        self._refresh_plot_page(headers, columns_from_2d(np.asarray(preview)))

    def _build_slice_controls(self, parent_layout: QVBoxLayout) -> None:
        """Axis and slice for a stack of frames, mirroring the viewer's own pair.

        Without them the dialog previewed frame 0 of axis 0 and wrote the whole
        stack the same way, so an axis chosen in the viewer was silently
        dropped and the pictures came out sliced the wrong way.
        """
        shape = np.asarray(self._sample_data).shape
        self._is_stack = len(shape) >= 3 and shape[0] > 1
        if not self._is_stack:
            # Not added at all rather than added and hidden: a single frame has
            # no axis to choose, and an empty row still takes up a position that
            # the rows below are read by.
            return

        self.row_slice = QHBoxLayout()
        self.cb_slice_axis = QComboBox()
        self.cb_slice_axis.setToolTip("Which array axis the frames are counted along")
        self.sl_slice = QSlider(Qt.Orientation.Horizontal)
        self.sl_slice.setMinimum(0)
        self.sl_slice.setToolTip("Which frame the preview shows; every frame is exported")
        self.lbl_slice = QLabel("- / -")

        self.row_slice.addWidget(QLabel("Axis:"))
        self.row_slice.addWidget(self.cb_slice_axis)
        self.row_slice.addSpacing(10)
        self.row_slice.addWidget(QLabel("Preview slice:"))
        self.row_slice.addWidget(self.sl_slice, stretch=1)
        self.row_slice.addWidget(self.lbl_slice)
        parent_layout.addLayout(self.row_slice)

        for axis in range(len(shape)):
            self.cb_slice_axis.addItem(axis_label(axis, shape[axis]), axis)
        self.cb_slice_axis.setCurrentIndex(min(self._initial_slice_axis, len(shape) - 1))
        self._reset_slice_slider()
        self.sl_slice.setValue(min(self._initial_slice_index, self.sl_slice.maximum()))

        self.cb_slice_axis.currentIndexChanged.connect(self._on_slice_axis_changed)
        self.sl_slice.valueChanged.connect(self._on_slice_changed)
        self._update_slice_label()

    def _reset_slice_slider(self) -> None:
        shape = np.asarray(self._sample_data).shape
        axis = self.slice_axis()
        count = shape[axis] if axis < len(shape) else 1
        self.sl_slice.blockSignals(True)
        self.sl_slice.setMaximum(max(0, count - 1))
        self.sl_slice.setValue(0)
        self.sl_slice.blockSignals(False)

    def _update_slice_label(self) -> None:
        self.lbl_slice.setText(f"{self.sl_slice.value() + 1} / {self.sl_slice.maximum() + 1}")

    def _on_slice_axis_changed(self, _index: int) -> None:
        self._reset_slice_slider()
        self._update_slice_label()
        self._refresh_image_preview()

    def _on_slice_changed(self, _value: int) -> None:
        self._update_slice_label()
        self._refresh_image_preview()

    def slice_axis(self) -> int:
        """The axis the frames are counted along."""
        if not getattr(self, "_is_stack", False):
            return 0
        axis = self.cb_slice_axis.currentData()
        return int(axis) if axis is not None else 0

    def _preview_image_frame(self) -> np.ndarray:
        arr = np.asarray(self._sample_data)
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        elif arr.ndim >= 3:
            index = self.sl_slice.value() if getattr(self, "_is_stack", False) else 0
            arr = take_slice(arr, self.slice_axis(), index)
        return np.squeeze(arr)

    def _refresh_image_preview(self) -> None:
        if self._data_kind != "image":
            return
        try:
            frame = rotate_frame(self._preview_image_frame(), self.image_rotation())
            self.preview_image.set_data(frame)
            self.preview_image._auto_contrast()
            # set_data rebuilds the display, so the angle has to go back on.
            self._apply_preview_angle_correction()
        except Exception as exc:
            logging.warning("Failed to refresh batch export image preview: %s", exc)

    def _build_angle_controls(self) -> None:
        """Incidence-angle correction, shown as its own value.

        A grazing-incidence frame is foreshortened along one axis; the
        correction stretches it back by 1/sin(theta). Zero means "off", which is
        why the spin box carries a special text there rather than needing a
        separate enable box.
        """
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(0.0, 179.99)
        self.spin_angle.setDecimals(2)
        self.spin_angle.setSingleStep(1.0)
        self.spin_angle.setSuffix("°")
        self.spin_angle.setSpecialValueText("Off")
        self.spin_angle.setValue(0.0)
        # Wide enough for "179.99" plus the degree sign and the arrows: a
        # clipped suffix reads as a typo in the number.
        self.spin_angle.setMinimumWidth(110)
        self.spin_angle.setMaximumWidth(140)
        self.spin_angle.setToolTip(
            "Incidence angle for the geometry correction; 0 leaves the image as recorded"
        )

        self.cb_angle_axis = QComboBox()
        self.cb_angle_axis.addItems(["X", "Y"])
        self.cb_angle_axis.setToolTip("Which axis the incidence foreshortens")
        compact_combo(self.cb_angle_axis, 4)

        self.cb_image_rotation = QComboBox()
        self.cb_image_rotation.addItems(IMAGE_ROTATIONS)
        self.cb_image_rotation.setToolTip(
            "Turn the written image clockwise. Quarter turns only, so no pixel is resampled."
        )
        compact_combo(self.cb_image_rotation, 6)

        self.spin_angle.valueChanged.connect(self._apply_preview_angle_correction)
        self.cb_angle_axis.currentTextChanged.connect(self._apply_preview_angle_correction)
        # Rotation changes the frame itself, so the preview is rebuilt from it.
        self.cb_image_rotation.currentTextChanged.connect(self._refresh_image_preview)

    def _apply_preview_angle_correction(self) -> None:
        """Push the angle controls into the preview, which the export then reads."""
        angle = float(self.spin_angle.value())
        if angle <= 0.0:
            self.preview_image.set_q_calibration(None)
            self.preview_image._apply_display_transform()
            self.preview_image._auto_fit_view()
            return
        self.preview_image.apply_incidence_display_correction(
            angle, self.cb_angle_axis.currentText()
        )


# What separates the parts of a scan filename: Scan_ECL_5p0uJIR_050.
_STEM_SEPARATORS = re.compile(r"[_\-.]+")


def parse_keywords(text: str) -> list[str]:
    """Split the keyword field on whitespace.

    Whitespace rather than commas, although the scan-number field beside it
    uses commas: there the commas mean *any of these*, here they would mean
    *all of these*. One punctuation mark with opposite meanings in two adjacent
    boxes is a trap. Whitespace is also what every search box does.

    Quoting is deliberately not supported — a scan filename does not contain
    spaces, and a keyword that needs one can be matched by a fragment either
    side of it.
    """
    return [word.lower() for word in str(text or "").split() if word]


def stem_matches_keywords(stem: str, keywords: list[str]) -> bool:
    """Whether every keyword appears somewhere in the filename stem.

    Substring, not prefix: the interesting part is often in the middle —
    ``ECL`` in ``Scan_ECL_5p0uJIR_050`` — and it is not always a whole token
    either, so anchoring it would rule out ``ScanECL_050``.
    """
    lowered = str(stem or "").lower()
    return all(keyword in lowered for keyword in keywords)


def scan_number_in_stem(stem: str, scan_numbers: list[str]) -> str | None:
    """The first requested scan number that the stem carries, or ``None``.

    A whole token, unlike the keywords beside it, because a number matched as a
    substring over-matches ruinously: ``050`` would also pick up ``..._1050``,
    and ``0`` would match every file in a set whose names contain ``5p0uJIR``
    or ``0340``.

    Compared numerically when both sides are digits, so ``47`` finds ``_047``
    and the zero padding does not have to be remembered.
    """
    tokens = [token for token in _STEM_SEPARATORS.split(str(stem or "")) if token]
    digits = {token: int(token) for token in tokens if token.isdigit()}
    for wanted in scan_numbers:
        if wanted in tokens:
            return wanted
        if wanted.isdigit() and int(wanted) in digits.values():
            return wanted
    return None


def batch_number_ambiguity(
    matching_files: list[tuple[pathlib.Path, str]],
) -> dict[str, list[pathlib.Path]]:
    """Scan numbers that matched more than one file, and which files.

    Happens as soon as the keyword is loose enough to span two families —
    ``Scan_`` covers ``Scan_ECL_5p0uJIR_047`` and ``Scan_ECL_10p0uJIR_047``
    alike. It is allowed: the two are written under their own names, so nothing
    is overwritten. It is reported because otherwise the batch quietly holds
    twice what was asked for.
    """
    by_number: dict[str, list[pathlib.Path]] = {}
    for path, scan_num in matching_files:
        by_number.setdefault(scan_num, []).append(path)
    return {number: paths for number, paths in by_number.items() if len(paths) > 1}


def describe_number_ambiguity(ambiguity: dict[str, list[pathlib.Path]]) -> str:
    """The ambiguity written out for a message box, or ``""`` when there is none."""
    if not ambiguity:
        return ""
    lines = []
    for number in sorted(ambiguity):
        lines.append(f"{number}:")
        lines.extend(f"    {path.name}" for path in sorted(ambiguity[number]))
    return (
        f"{len(ambiguity)} scan number(s) match more than one file:\n"
        + "\n".join(lines)
        + "\n\nThey are written under their own names, so nothing is overwritten."
        "\nNarrow the keywords to pick one family."
    )


def summarise_number_ambiguity(ambiguity: dict[str, list[pathlib.Path]]) -> str:
    """One line for the export dialog, or ``""`` when there is none."""
    if not ambiguity:
        return ""
    total = sum(len(paths) for paths in ambiguity.values())
    numbers = ", ".join(sorted(ambiguity))
    return f"⚠ {numbers} match more than one file ({total} files) — narrow the keywords to split them"


def scan_stem_parts(stem: str) -> tuple[str, str] | None:
    """Split a filename stem into its prefix and its trailing scan number.

    ``scanx_0340`` becomes ``("scanx_", "0340")``. ``None`` when the name ends
    in something other than a number, which means it is not a scan file.
    """
    match = re.fullmatch(r"(.*?)(\d+)", str(stem or ""))
    return (match.group(1), match.group(2)) if match else None


def common_keyword(stems: list[str]) -> str:
    """A keyword that matches all of these filenames and nothing narrower.

    The shared opening of the names, cut back to the last separator: two files
    of different families share ``Scan_EC`` only because ``ECL`` and ``ECR``
    happen to start alike, and half a word is not a keyword. Cut there it comes
    out as ``Scan_``, which is the part that really is common.

    A single file keeps its whole leading part, which is the right answer for a
    drop of one: it names that family exactly.
    """
    kept = [str(stem) for stem in stems if str(stem)]
    if not kept:
        return ""

    shared = os.path.commonprefix(kept)
    if len(kept) == 1:
        # Nothing to compare against: drop the trailing scan number, which is
        # the other field's business.
        parts = scan_stem_parts(shared)
        return parts[0] if parts else shared

    while shared and shared[-1] not in "_-.":
        shared = shared[:-1]
    return shared


def compress_scan_numbers(numbers: list[str]) -> str:
    """Write a set of scan numbers the short way: ``0340-0342,0350``.

    Consecutive numbers of the same width become a range, so a drop of twenty
    files reads as one span rather than twenty tokens.
    """
    kept = sorted({str(n).strip() for n in numbers if str(n).strip()}, key=lambda n: (int(n), n))
    if not kept:
        return ""

    parts: list[str] = []
    run_start = run_end = kept[0]
    for number in kept[1:]:
        contiguous = int(number) == int(run_end) + 1 and len(number) == len(run_end)
        if contiguous:
            run_end = number
            continue
        parts.append(run_start if run_start == run_end else f"{run_start}-{run_end}")
        run_start = run_end = number
    parts.append(run_start if run_start == run_end else f"{run_start}-{run_end}")
    return ",".join(parts)


def batch_folder_conflict(matching_files: list[tuple[pathlib.Path, str]]) -> str:
    """Say which folders a batch would span, or ``""`` when it stays in one.

    A batch belongs to one measurement directory. Allowed to span several, two
    scans with the same number in different folders both matched, and since an
    export is named after the file's stem the second silently overwrote the
    first — four datasets, two files on disk, no warning. Within one folder
    filenames are unique, so the whole class of collisions cannot arise.
    """
    folders = sorted({str(path.parent) for path, _scan in matching_files})
    if len(folders) <= 1:
        return ""
    listed = "\n".join(f"  {folder}" for folder in folders)
    return (
        f"The scan numbers match files in {len(folders)} folders:\n{listed}\n\n"
        "A batch has to stay in one folder — otherwise two scans with the same "
        "number would be written to the same file name.\n"
        "Close the files you do not want, or narrow the scan numbers."
    )


def batch_target_labels(batch_paths: list[str]) -> dict[str, str]:
    """Map each Y path to a short column tag, unique within the export.

    With a single dataset the tag is empty (columns stay ``scan_Y_1``, as
    before). With several, the leaf name is used, falling back to the full
    path when two datasets share a leaf.
    """
    if len(batch_paths) <= 1:
        return {path: "" for path in batch_paths}

    leaves = [path.rstrip("/").split("/")[-1] or path for path in batch_paths]
    if len(set(leaves)) == len(leaves):
        return dict(zip(batch_paths, leaves))
    return {path: path.strip("/").replace("/", "_") for path in batch_paths}


def build_batch_targets(
    matching_files: list[tuple[pathlib.Path, str]],
    batch_paths: list[str],
) -> list[BatchTarget]:
    """Expand scans x Y paths into the flat list of datasets to export.

    Scan-major order: every Y dataset of scan 0080, then those of 0081, so a
    combined table groups the columns by scan.
    """
    tags = batch_target_labels(batch_paths)
    targets: list[BatchTarget] = []
    for file_path, scan_num in matching_files:
        for batch_path in batch_paths:
            tag = tags.get(batch_path, "")
            targets.append(
                BatchTarget(
                    file_path=file_path,
                    scan_num=scan_num,
                    ds_path=adjust_batch_path_for_scan(batch_path, scan_num),
                    label=f"{file_path.stem}_{tag}" if tag else file_path.stem,
                )
            )
    return targets


def adjust_batch_path_for_scan(batch_path: str, scan_num: str) -> str:
    """Replace the first scan-looking number in a dataset path."""
    adjusted_path = str(batch_path).strip().strip("/")
    matches = re.findall(r"\d{4}", adjusted_path)
    if matches:
        adjusted_path = adjusted_path.replace(matches[0], scan_num, 1)
    return adjusted_path


def safe_export_name(file_path: pathlib.Path, dataset_path: str, suffix: str = "") -> str:
    """Build a filesystem-safe export stem."""
    ds = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_path).strip("/"))
    ds = ds.strip("_") or "dataset"
    suffix = f"_{suffix}" if suffix else ""
    return f"{file_path.stem}_{ds}{suffix}"


def table_format_from(export_settings: dict[str, Any]) -> TableFormat:
    """Read the chosen tabular dialect, tolerating settings dicts without one."""
    fmt = export_settings.get("table_format")
    if isinstance(fmt, TableFormat):
        return fmt
    return get_table_format(str(fmt) if fmt else DEFAULT_TABLE_FORMAT_KEY)


def is_curve_export_data(data: np.ndarray) -> bool:
    """Return true for 1D data or 2D data with up to 10 columns."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return True
    return arr.ndim == 2 and arr.shape[1] <= 10


def curve_export_columns(data: np.ndarray) -> np.ndarray:
    """Normalize curve export data to rows x columns."""
    arr = np.asarray(data).squeeze()
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] <= 10:
        return arr
    raise ValueError(f"Not a curve/table export shape: {arr.shape}")


def build_curve_preview_table(
    targets: list[BatchTarget],
    matching_files: list[tuple[pathlib.Path, str]],
    export_settings: dict[str, Any],
    *,
    max_targets: int | None = PREVIEW_MAX_TARGETS,
    max_rows: int | None = PREVIEW_MAX_ROWS,
) -> tuple[np.ndarray, list[str]]:
    """Build the 1D table for a batch, from the same settings the export uses.

    The caps exist for the on-screen table, which only has to show the shape of
    the result. Anything that *produces* output — the batch plot writes real
    figures from this — must pass ``None`` for both, or it silently ships the
    first few scans at the first 200 points.
    """
    if not targets:
        return np.empty((0, 0)), []

    combined = str(export_settings.get("curve_output_mode")) == "Combined file"
    # Single-file mode writes one file per target, so only the first is shown.
    if combined:
        preview_targets = targets if max_targets is None else targets[:max_targets]
    else:
        preview_targets = targets[:1]
    series: list[tuple[BatchTarget, np.ndarray, np.ndarray | None]] = []

    for target in preview_targets:
        with h5py.File(target.file_path, "r") as f:
            if target.ds_path not in f:
                raise KeyError(f"Dataset path not found: {target.file_path.name}::{target.ds_path}")
            y_columns = curve_export_columns(np.asarray(f[target.ds_path][()]))
        x_data = read_batch_x_data(
            export_settings=export_settings,
            current_file=target.file_path,
            current_scan=target.scan_num,
            matching_files=matching_files,
            expected_len=y_columns.shape[0],
        )
        series.append((target, y_columns, x_data))

    if not combined:
        target, y_columns, x_data = series[0]
        n = y_columns.shape[0] if max_rows is None else min(max_rows, y_columns.shape[0])
        y_headers = batch_y_headers(target.file_path, target.ds_path, y_columns.shape[1])
        if x_data is None:
            return y_columns[:n], y_headers
        return (
            np.column_stack([x_data[:n], y_columns[:n]]),
            [batch_x_header(export_settings, target.file_path)] + y_headers,
        )

    longest = max(y.shape[0] for _target, y, _x in series)
    max_len = longest if max_rows is None else min(max_rows, longest)
    headers: list[str] = []
    columns: list[np.ndarray] = []
    shared_x_written = False
    for target, y_columns, x_data in series:
        if x_data is not None:
            if bool(export_settings.get("share_x", False)):
                if not shared_x_written:
                    col = np.full(max_len, np.nan, dtype=float)
                    col[: min(max_len, len(x_data))] = x_data[:max_len]
                    columns.append(col)
                    headers.append(batch_x_header(export_settings, None))
                    shared_x_written = True
            else:
                col = np.full(max_len, np.nan, dtype=float)
                col[: min(max_len, len(x_data))] = x_data[:max_len]
                columns.append(col)
                headers.append(batch_x_header(export_settings, target.file_path))
        y_headers = batch_y_headers(target.file_path, target.ds_path, y_columns.shape[1])
        for col_idx in range(y_columns.shape[1]):
            col = np.full(max_len, np.nan, dtype=float)
            n = min(max_len, y_columns.shape[0])
            col[:n] = y_columns[:n, col_idx]
            columns.append(col)
            headers.append(y_headers[col_idx])

    return np.column_stack(columns) if columns else np.empty((0, 0)), headers


def batch_y_headers(
    file_path: pathlib.Path,
    ds_path: str,
    n_columns: int,
) -> list[str]:
    """Name the Y columns of one scan, the way every other export names them.

    ``scanx_0340_data_03``, with ``_col1`` appended only when the dataset has
    more than one column. Single-file exports headed these ``Y_1``, ``Y_2``,
    which says nothing at all once the file is open somewhere else — not which
    scan it came from, and not which dataset.
    """
    key = f"{file_path}::{ds_path}"
    if n_columns <= 1:
        return [short_series_label(key, fallback="Y")]
    return [
        short_series_label(f"{key} [Col {index}]", fallback=f"Y_{index + 1}")
        for index in range(n_columns)
    ]


def batch_x_header(export_settings: dict[str, Any], file_path: pathlib.Path | None) -> str:
    """Name an X column after the X dataset, not after the Y beside it.

    A shared X is deliberately left without a scan number — it is one column
    standing for every scan in the table, so ``actuator_1_1_X`` rather than the
    name of whichever scan happened to supply it.
    """
    x_path = str(export_settings.get("x_path", "") or "").strip()
    if not x_path:
        return "X"
    return x_column_header(f"{file_path}::{x_path}" if file_path is not None else f"::{x_path}")


def read_batch_x_data(
    *,
    export_settings: dict[str, Any],
    current_file: pathlib.Path,
    current_scan: str,
    matching_files: list[tuple[pathlib.Path, str]],
    expected_len: int,
) -> np.ndarray | None:
    """Read optional X data according to the batch export settings."""
    if not bool(export_settings.get("export_x", False)):
        return None
    x_path_template = str(export_settings.get("x_path", "")).strip().strip("/")
    if not x_path_template:
        return None

    if bool(export_settings.get("share_x", False)):
        shared_scan = str(export_settings.get("shared_x_scan", "")).strip() or current_scan
        x_scan = shared_scan
        x_file = next((fp for fp, sn in matching_files if sn == shared_scan), None)
        if x_file is None:
            raise ValueError(f"Shared X scan '{shared_scan}' is not opened")
    else:
        x_file = current_file
        x_scan = current_scan

    x_path = adjust_batch_path_for_scan(x_path_template, x_scan)
    with h5py.File(x_file, "r") as f:
        if x_path not in f:
            raise KeyError(f"X dataset path not found: {x_file.name}::{x_path}")
        x_data = np.asarray(f[x_path][()]).squeeze()
    if x_data.ndim != 1:
        raise ValueError(f"X dataset must be 1D, got shape {x_data.shape}")
    if len(x_data) != expected_len:
        raise ValueError(f"X length {len(x_data)} does not match data length {expected_len}")
    return x_data


def export_curve_dataset(
    file_path: pathlib.Path,
    scan_num: str,
    dataset_path: str,
    data: np.ndarray,
    export_settings: dict[str, Any],
    matching_files: list[tuple[pathlib.Path, str]],
) -> None:
    """Export 1D or small-column 2D data as CSV."""
    arr = np.asarray(data).squeeze()
    if arr.ndim == 1:
        y_columns = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] <= 10:
        y_columns = arr
    else:
        raise ValueError(f"Not a curve/table export shape: {arr.shape}")

    x_data = read_batch_x_data(
        export_settings=export_settings,
        current_file=file_path,
        current_scan=scan_num,
        matching_files=matching_files,
        expected_len=y_columns.shape[0],
    )

    fmt = table_format_from(export_settings)
    output_dir = pathlib.Path(export_settings["output_dir"])
    out_path = output_dir / f"{safe_export_name(file_path, dataset_path)}{fmt.suffix}"

    headers = (
        ([batch_x_header(export_settings, file_path)] if x_data is not None else [])
        + batch_y_headers(file_path, dataset_path, y_columns.shape[1])
    )
    columns = ([x_data] if x_data is not None else []) + columns_from_2d(y_columns)
    write_table(out_path, headers, columns, fmt)


def export_curve_combined_table(
    targets: list[BatchTarget],
    matching_files: list[tuple[pathlib.Path, str]],
    export_settings: dict[str, Any],
) -> tuple[int, list[str]]:
    """Export every scan x Y dataset into one wide table."""
    fmt = table_format_from(export_settings)
    output_dir = pathlib.Path(export_settings["output_dir"])
    first = targets[0]
    out_path = output_dir / f"{safe_export_name(first.file_path, first.ds_path, 'batch')}{fmt.suffix}"
    series: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    fail_details: list[str] = []

    for target in targets:
        try:
            with h5py.File(target.file_path, "r") as f:
                if target.ds_path not in f:
                    raise KeyError(f"Dataset path not found: {target.ds_path}")
                arr = np.asarray(f[target.ds_path][()]).squeeze()
            if arr.ndim == 1:
                y_columns = arr.reshape(-1, 1)
            elif arr.ndim == 2 and arr.shape[1] <= 10:
                y_columns = arr
            else:
                raise ValueError(f"Not a curve/table export shape: {arr.shape}")
            x_data = read_batch_x_data(
                export_settings=export_settings,
                current_file=target.file_path,
                current_scan=target.scan_num,
                matching_files=matching_files,
                expected_len=y_columns.shape[0],
            )
            series.append((target, y_columns, x_data))
        except Exception as exc:
            fail_details.append(f"{target.file_path.name}::{target.ds_path}: {exc}")

    if not series:
        return 0, fail_details

    share_x = bool(export_settings.get("share_x", False))
    headers: list[str] = []
    columns: list[Any] = []
    shared_x_written = False
    for target, y_columns, x_data in series:
        if x_data is not None and not (share_x and shared_x_written):
            headers.append(batch_x_header(export_settings, None if share_x else target.file_path))
            columns.append(x_data)
            shared_x_written = True
        for col_idx, header in enumerate(
            batch_y_headers(target.file_path, target.ds_path, y_columns.shape[1])
        ):
            headers.append(header)
            columns.append(y_columns[:, col_idx])

    write_table(out_path, headers, columns, fmt)
    return len(series), fail_details


def resolve_batch_colormap(name: str) -> pg.ColorMap | None:
    try:
        return pg.colormap.get(name)
    except Exception:
        return None


def render_batch_colormapped_rgb(data: np.ndarray, export_settings: dict[str, Any]) -> np.ndarray:
    """Render 2D data as RGB using pyqtgraph colormaps."""
    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    display_data = np.asarray(data, dtype=np.float64)
    display_data = np.nan_to_num(display_data, nan=0.0, posinf=0.0, neginf=0.0)

    scale_mode = str(export_settings.get("scale", "Linear"))
    if scale_mode == "Log":
        positive_mask = display_data > 0
        min_positive = float(np.min(display_data[positive_mask])) if np.any(positive_mask) else 1e-10
        display_data = np.log10(np.where(positive_mask, display_data, min_positive))
    elif scale_mode == "SymLog":
        display_data = np.sign(display_data) * np.log10(1 + np.abs(display_data))
    elif scale_mode == "Square root":
        display_data = np.sqrt(np.clip(display_data, 0, None))

    saved_levels = export_settings.get("levels")
    if saved_levels is not None:
        levels = (float(saved_levels[0]), float(saved_levels[1]))
    elif "Auto" in str(export_settings.get("contrast", "")):
        levels = ImageView2DEnhanced._robust_auto_levels(display_data)
    else:
        levels = (float(display_data.min()), float(display_data.max())) if display_data.size else (0.0, 1.0)
    level_min, level_max = levels if levels is not None else (0.0, 1.0)
    if not np.isfinite(level_min) or not np.isfinite(level_max) or level_max <= level_min:
        level_min, level_max = 0.0, 1.0

    normalized = np.clip((display_data - level_min) / (level_max - level_min), 0.0, 1.0)
    cmap = resolve_batch_colormap(str(export_settings.get("colormap", "viridis")))
    if cmap is None:
        cmap = pg.colormap.get("viridis")
    if bool(export_settings.get("invert", False)):
        try:
            cmap = cmap.reverse() or cmap
        except Exception:
            pass
    lut = np.asarray(cmap.getLookupTable(0.0, 1.0, 256))
    if lut.dtype.kind == "f":
        lut = np.clip(lut, 0.0, 255.0)
    lut = lut.astype(np.uint8)
    indices = np.clip(np.rint(normalized * 255), 0, 255).astype(np.uint8)
    return lut[indices, :3]


def export_image_dataset(
    file_path: pathlib.Path,
    dataset_path: str,
    data: np.ndarray,
    export_settings: dict[str, Any],
) -> None:
    """Export 2D data as colormapped PNG, optionally also raw TIFF."""
    from PIL import Image
    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    arr = np.asarray(data)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)

    if arr.ndim == 2:
        frames = [(arr, "")]
    elif arr.ndim >= 3:
        # Along the axis the dialog is previewing, not always the first: the
        # viewer lets that be chosen, and frames cut the wrong way are not the
        # pictures that were on screen.
        axis = slice_axis_from(export_settings)
        frames = [
            (take_slice(arr, axis, i), f"slice{i:04d}")
            for i in range(slice_count(arr, axis))
        ]
    else:
        raise ValueError(f"Not an image export shape: {arr.shape}")

    output_dir = pathlib.Path(export_settings["output_dir"])
    rotation = rotation_from(export_settings)
    for frame, suffix in frames:
        if frame.ndim > 2:
            frame = np.squeeze(frame)
        if frame.ndim != 2:
            raise ValueError(f"Cannot export image frame with shape {frame.shape}")
        stem = safe_export_name(file_path, dataset_path, suffix)
        if export_settings.get("save_tiff"):
            # The values as recorded: no rotation, no colormap, no rescaling.
            DataExporter.export_raw_tiff(frame, output_dir / f"{stem}.tif")

        # The picture, on the other hand, is what is on screen — rotated first,
        # exactly as the preview does it.
        frame = rotate_frame(frame, rotation)
        rgb = render_batch_colormapped_rgb(frame, export_settings)
        img = Image.fromarray(rgb, mode="RGB")
        incidence = export_settings.get("incidence") or {}
        if bool(incidence.get("use_incidence", False)) and bool(
            incidence.get("incidence_applied_in_display", False)
        ):
            fac = ImageView2DEnhanced._incidence_factor_from_deg(float(incidence.get("incidence_deg", 0.0)))
            axis = str(incidence.get("incidence_axis", "X")).upper()
            if fac > 1.0:
                w, h = img.size
                if axis == "Y":
                    img = img.resize((w, max(1, int(round(h * fac)))), Image.Resampling.BICUBIC)
                else:
                    img = img.resize((max(1, int(round(w * fac))), h), Image.Resampling.BICUBIC)
        fmt = image_format_from(export_settings)
        if fmt.name == "JPEG":
            img.save(output_dir / f"{stem}{fmt.suffix}", "JPEG", quality=95)
        else:
            img.save(output_dir / f"{stem}{fmt.suffix}", "PNG")

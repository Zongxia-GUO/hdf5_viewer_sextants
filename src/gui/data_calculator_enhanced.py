"""Enhanced Data Calculator Dialog with drag-and-drop support."""

# Copyright (C) 2023 Dennis Leonard
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
import pathlib
import re
from typing import Any

import h5py
import numpy as np
from PyQt6.QtCore import QSettings, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from src.gui.batch_export import compact_combo, elided_label
from src.gui.dataset_path_combo import DatasetPathCombo
from src.gui.export_naming import (
    pair_label,
    remember_save_directory,
    short_series_label,
    suggested_save_path,
    x_column_header,
)
from src.gui.table_model import CopyableTableView, DataTable
from src.gui.x_target import (
    read_x_dataset,
    register_x_target,
    remembered_x_dataset,
    x_scope_of,
)
from src.img.img_path import img_path
from src.lib_h5.table_format import (
    DEFAULT_TABLE_FORMAT_KEY,
    TableFormat,
    dialog_filter,
    format_labels,
    get_table_format,
)
from src.lib_h5.table_writer import write_table
from src.recon.expr import ExpressionError, evaluate as _evaluate_expression

# The second output button reads differently for a curve and for an image:
# a 2-D result cannot be plotted as a curve, but it can be put on the clipboard.
PLOT_BUTTON_TEXT = "Plot"
PLOT_BUTTON_TOOLTIP = "Plot: the operands and the result, as a table and a figure"
COPY_BUTTON_TEXT = "Copy"
COPY_BUTTON_TOOLTIP = "Copy: put the result image, with its colormap, on the clipboard"


# The value a selector carries for "do not pick anything out".
SELECT_ALL = -1

# Width of the controls side of the splitter. It was widened to 440 to fit a
# 3-D operand's two selectors; once those stopped carrying their own labels in
# every entry the pair fits in half the room, so the result viewer gets the
# space back.
CONTROLS_PANEL_WIDTH = 300

# The two selectors hold values, not sentences — "0 (400)" and "12" — so they
# are sized for those plus the dropdown arrow, and capped so that a layout with
# room to spare cannot inflate them into two mostly-empty boxes.
AXIS_COMBO_MIN_WIDTH = 55
AXIS_COMBO_MAX_WIDTH = 75
PART_COMBO_MIN_WIDTH = 70
PART_COMBO_MAX_WIDTH = 90


def _shapes_broadcast(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> bool:
    """Whether numpy can pair these two shapes up element by element."""
    try:
        np.broadcast_shapes(tuple(shape_a), tuple(shape_b))
    except ValueError:
        return False
    return True


def operand_from_selection(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Cut one operand down to what the selectors ask for.

    Two shapes, one idea — take a part out, or keep the whole thing:

    * 2-D with a column chosen  ->  that column
    * 3-D with a frame chosen   ->  that frame, counted along ``axis``
    * 3-D with no frame chosen  ->  the whole stack, reoriented so the chosen
      axis comes first

    Reorienting rather than remembering the axis keeps the result an ordinary
    frame-major stack, so the viewer, the export and the pattern tool all read
    it the same way without being told anything.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr[:, index] if index >= 0 else arr
    if arr.ndim < 3:
        return arr

    axis = int(axis) % arr.ndim
    if index >= 0:
        return np.asarray(np.take(arr, min(index, arr.shape[axis] - 1), axis=axis))
    return np.moveaxis(arr, axis, 0) if axis else arr


class DragDropComboBox(DatasetPathCombo):
    """Calculator dataset selector (shared behavior with FTH path combo)."""

    def __init__(self, placeholder: str, parent: Any = None) -> None:
        super().__init__(placeholder=placeholder, parent=parent)


class ResultExportDialog(QDialog):
    """Full-export settings for a calculator result.

    Shows the columns that will be written — the two operands and the result —
    and lets an X axis be attached, the same way the batch export dialog does.
    """

    settings_changed = pyqtSignal()

    def __init__(
        self,
        parent: Any,
        *,
        opened_files: tuple[pathlib.Path, ...],
        dataset_full_keys_1d: list[str],
        preferred_x_key: str | None,
        expression: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Calculation Result")
        self.setWindowIcon(QIcon(str(pathlib.Path(img_path(), "sextants.ico"))))
        # A plain Window, shown non-modally, so the tree stays draggable.
        self.setWindowFlag(Qt.WindowType.Window)
        # Square: the preview sits above the controls, so extra width buys nothing.
        self.resize(560, 560)
        self._opened_files = opened_files

        root = QVBoxLayout(self)

        self.preview_table = CopyableTableView()
        root.addWidget(self.preview_table, stretch=1)

        box = QGroupBox("Export")
        form = QVBoxLayout(box)

        # The result viewer's own X wins; failing that, the one chosen in the
        # tree *for this tool*, so the same dataset does not have to be found
        # twice — and so an X set in some other tool never turns up here.
        preferred_x_key = preferred_x_key or remembered_x_dataset(x_scope_of(parent))

        row_x = QHBoxLayout()
        self.chk_export_x = QCheckBox("Export X")
        self.chk_export_x.setChecked(bool(preferred_x_key))
        row_x.addWidget(self.chk_export_x)
        self.combo_x = compact_combo(DragDropComboBox("Drag or type the X dataset"), 20)
        self.combo_x.populate_from_full_keys(list(dataset_full_keys_1d), opened_files=opened_files)
        if preferred_x_key:
            self.combo_x.add_full_key(preferred_x_key, select=True)
        row_x.addWidget(self.combo_x, stretch=1)
        form.addLayout(row_x)

        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("Format:"))
        self.combo_format = compact_combo(QComboBox())
        self.combo_format.addItems(format_labels())
        stored = QSettings().value("export/table_format", DEFAULT_TABLE_FORMAT_KEY)
        self.combo_format.setCurrentText(get_table_format(str(stored)).label)
        row_fmt.addWidget(self.combo_format, stretch=1)
        form.addLayout(row_fmt)

        row_expr = QHBoxLayout()
        row_expr.addWidget(QLabel("Expression:"))
        row_expr.addWidget(elided_label(expression or "-"))
        row_expr.addStretch()
        form.addLayout(row_expr)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

        root.addWidget(box)

        self.chk_export_x.stateChanged.connect(self._emit_refresh)
        self.combo_x.currentTextChanged.connect(self._emit_refresh)
        register_x_target(self)

    def _emit_refresh(self, *_args: Any) -> None:
        self.settings_changed.emit()

    def set_x_dataset(self, key: str) -> bool:
        """Take an X dataset handed over by the tree's ``Set X``."""
        if "::" not in key:
            return False
        self.combo_x.add_full_key(key, select=True)
        self.chk_export_x.setChecked(True)
        return True

    def x_key(self) -> str | None:
        """Full ``file::dataset`` key of the chosen X, or None when X is off."""
        if not self.chk_export_x.isChecked():
            return None
        entry = self.combo_x.get_entry(opened_files=self._opened_files)
        if entry is None:
            return None
        return f"{entry[0]}::{entry[1]}"

    def table_format(self) -> TableFormat:
        """The tabular dialect chosen for this export."""
        return get_table_format(self.combo_format.currentText())

    def show_preview(self, headers: list[str], columns: list[Any]) -> None:
        """Fill the preview with the columns that would be written."""
        rows = min(200, max((len(col) for col in columns), default=0))
        table = np.full((rows, len(columns)), np.nan, dtype=float)
        for idx, col in enumerate(columns):
            take = min(rows, len(col))
            table[:take, idx] = np.asarray(col, dtype=float)[:take]
        self.preview_table.setModel(DataTable(table, headers))


class DataCalculatorEnhanced(QDialog):
    """Enhanced dialog for performing calculations on two datasets with drag-drop support."""
    def __init__(
        self,
        opened_files: tuple[pathlib.Path, ...],
        parent: Any = None,
        dataset_full_keys_1d: list[str] | None = None,
    ) -> None:
        """
        Initialize Enhanced Data Calculator Dialog.

        :param opened_files: Tuple of currently opened HDF5 files
        :param parent: Parent widget
        """
        super().__init__(parent)
        self.opened_files = opened_files
        self.dataset_full_keys_1d = dataset_full_keys_1d or []
        # An X chosen here belongs to this tool. Its own export and plot windows
        # inherit it through the parent chain; nothing outside the calculator
        # does. See src.gui.x_target.x_scope_of.
        self.x_scope = "calculator"
        self.result_data: np.ndarray | None = None
        self.result_widget: QWidget | None = None  # Current result display widget
        self.data_a: np.ndarray | None = None  # Original dataset A
        self.data_b: np.ndarray | None = None  # Original dataset B, optional for single-dataset operations
        self._is_populating_combos = False
        self._last_keys_sig: tuple[str, ...] = tuple()
        self._last_operation_expr: str = ""

        self.setWindowTitle("Data Calculator")
        self.setModal(False)  # Non-modal to allow dragging from main window

        # Set window flags to ensure proper layering behavior
        # Use Window flag instead of WindowStaysOnTopHint to prevent staying on top of all apps
        self.setWindowFlags(Qt.WindowType.Window)

        # Set initial size (not minimum) to allow later compression
        self.resize(1200, 650)

        self._init_ui()
        # Offer this window to the tree's "Set X", as the comparison tool does.
        # Without it, closing a Plot window opened from here sent the next
        # Set X past the calculator and into the main window's viewer.
        register_x_target(self)

    def _init_ui(self) -> None:
        """Initialize the user interface with left-right split layout."""
        main_layout = QVBoxLayout()

        # Add instruction label at top
        instruction = QLabel(
            "<b>Data Calculator</b> - Perform calculations on two datasets "
            "Drag and drop datasets from the tree view into the dropdown boxes!"
        )
        instruction.setWordWrap(False)
        instruction.setFixedHeight(35)
        instruction.setStyleSheet("background-color: #e3f2fd; padding: 5px; border-radius: 5px;")
        main_layout.addWidget(instruction)

        # Create horizontal splitter for left (controls) and right (result) panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Set size policy to allow splitter to expand with window
        from PyQt6.QtWidgets import QSizePolicy
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ===== LEFT PANEL - Controls =====
        left_panel = QWidget()
        # Set size policy for left panel to allow free resizing
        left_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Dataset Selection Group
        selection_group = QGroupBox("Select Datasets")
        selection_layout = QFormLayout()

        # Dataset A selection - with drag-drop support
        self.combo_dataset_a = DragDropComboBox("-- no dataset A --")
        self.combo_dataset_a.currentTextChanged.connect(self._on_dataset_a_changed)
        self.combo_dataset_a.lineEdit().returnPressed.connect(
            lambda: self._on_dataset_line_entered(self.combo_dataset_a, "A")
        )
        selection_layout.addRow("Dataset A:", self.combo_dataset_a)

        # Dataset A info
        self.label_info_a = QLabel("Shape: -, Type: -")
        self.label_info_a.setStyleSheet("color: gray; font-size: 9pt;")
        selection_layout.addRow("", self.label_info_a)

        # Dataset A: which part of it to use.
        self.lbl_part_a, self.combo_axis_a, self.spin_col_a = self._build_part_row(
            selection_layout, "A"
        )

        # Dataset B selection - with drag-drop support
        self.combo_dataset_b = DragDropComboBox("-- no dataset B --")
        self.combo_dataset_b.currentTextChanged.connect(self._on_dataset_b_changed)
        self.combo_dataset_b.lineEdit().returnPressed.connect(
            lambda: self._on_dataset_line_entered(self.combo_dataset_b, "B")
        )
        selection_layout.addRow("Dataset B:", self.combo_dataset_b)

        # Dataset B info
        self.label_info_b = QLabel("Shape: -, Type: -")
        self.label_info_b.setStyleSheet("color: gray; font-size: 9pt;")
        selection_layout.addRow("", self.label_info_b)

        # Dataset B: which part of it to use.
        self.lbl_part_b, self.combo_axis_b, self.spin_col_b = self._build_part_row(
            selection_layout, "B"
        )

        selection_group.setLayout(selection_layout)
        left_layout.addWidget(selection_group)

        # Operation Selection Group
        operation_group = QGroupBox("Operations")
        operation_layout = QVBoxLayout()

        # Quick operation buttons - Row 1
        btn_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.btn_add = QPushButton("A + B")
        self.btn_add.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_add.clicked.connect(lambda: self._perform_operation("A + B"))
        row1.addWidget(self.btn_add)

        self.btn_subtract = QPushButton("A - B")
        self.btn_subtract.setAutoDefault(False)
        self.btn_subtract.clicked.connect(lambda: self._perform_operation("A - B"))
        row1.addWidget(self.btn_subtract)
        btn_layout.addLayout(row1)

        # Row 2
        row2 = QHBoxLayout()
        self.btn_multiply = QPushButton("A * B")
        self.btn_multiply.setAutoDefault(False)
        self.btn_multiply.clicked.connect(lambda: self._perform_operation("A * B"))
        row2.addWidget(self.btn_multiply)

        self.btn_divide = QPushButton("A / B")
        self.btn_divide.setAutoDefault(False)
        self.btn_divide.clicked.connect(lambda: self._perform_operation("A / B"))
        row2.addWidget(self.btn_divide)
        btn_layout.addLayout(row2)

        # Row 3 - More operations
        row3 = QHBoxLayout()
        self.btn_avg = QPushButton("(A+B)/2")
        self.btn_avg.setAutoDefault(False)
        self.btn_avg.clicked.connect(lambda: self._perform_operation("(A + B) / 2"))
        row3.addWidget(self.btn_avg)

        self.btn_abs_diff = QPushButton("|A-B|")
        self.btn_abs_diff.setAutoDefault(False)
        self.btn_abs_diff.clicked.connect(lambda: self._perform_operation("np.abs(A - B)"))
        row3.addWidget(self.btn_abs_diff)
        btn_layout.addLayout(row3)

        # Row 4
        row4 = QHBoxLayout()
        self.btn_diff_ratio = QPushButton("(A-B)/(A+B)")
        self.btn_diff_ratio.setAutoDefault(False)
        self.btn_diff_ratio.clicked.connect(lambda: self._perform_operation("(A - B) / (A + B)"))
        row4.addWidget(self.btn_diff_ratio)

        self.btn_fft_a = QPushButton("FFT(A)")
        self.btn_fft_a.setAutoDefault(False)
        self.btn_fft_a.setToolTip("Calculate centered FFT magnitude of Dataset A")
        self.btn_fft_a.clicked.connect(lambda: self._perform_operation("FFT(A)"))
        row4.addWidget(self.btn_fft_a)
        btn_layout.addLayout(row4)

        operation_layout.addLayout(btn_layout)

        # Custom expression
        custom_layout = QVBoxLayout()
        custom_layout.addWidget(QLabel("Custom Expression:"))
        self.edit_custom = QLineEdit()
        self.edit_custom.setPlaceholderText("e.g., (A - B) / A, A * 2, FFT(A)")
        # Note: Enter key removed to avoid conflict with X->q energy input
        custom_layout.addWidget(self.edit_custom)

        self.btn_custom = QPushButton("Calculate")
        self.btn_custom.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_custom.clicked.connect(self._perform_custom_operation)
        custom_layout.addWidget(self.btn_custom)
        operation_layout.addLayout(custom_layout)

        # Help text
        help_label = QLabel(
            "<i>Available: A, optional B, +, -, *, /, **, FFT(A), "
            "sqrt() abs() log() exp() mean() max() gradient() ..., np.* math functions, "
            "pi, e. Slicing (A[:10]) works; anything else is rejected.</i>"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 8pt;")
        operation_layout.addWidget(help_label)

        operation_group.setLayout(operation_layout)
        left_layout.addWidget(operation_group)

        # Transfer result to comparison tool
        self.btn_transfer_to_comparison = QPushButton("Transfer Result to Comparison")
        self.btn_transfer_to_comparison.setAutoDefault(False)
        self.btn_transfer_to_comparison.setEnabled(False)
        self.btn_transfer_to_comparison.clicked.connect(self._transfer_result_to_comparison)
        left_layout.addWidget(self.btn_transfer_to_comparison)

        # The two things you do with a finished result, side by side: write it
        # out, or draw it. Quick save/copy live as icons in the result viewer
        # itself, like everywhere else in the app.
        output_row = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_export.clicked.connect(self._export_result)
        self.btn_export.setEnabled(False)
        output_row.addWidget(self.btn_export)

        # One button with two jobs, because a result has two shapes and only one
        # of them can be plotted: a curve gets Plot, an image gets Copy. The
        # dispatch happens on click rather than by reconnecting, so there is
        # never a moment when the label and the wiring disagree.
        self.btn_plot = QPushButton(PLOT_BUTTON_TEXT)
        self.btn_plot.setAutoDefault(False)
        self.btn_plot.setToolTip(PLOT_BUTTON_TOOLTIP)
        self.btn_plot.clicked.connect(self._on_plot_or_copy)
        self.btn_plot.setEnabled(False)
        output_row.addWidget(self.btn_plot)
        left_layout.addLayout(output_row)

        # Add stretch to push everything to the top
        left_layout.addStretch()

        left_panel.setLayout(left_layout)
        splitter.addWidget(left_panel)

        # ===== RIGHT PANEL - Result Display =====
        right_panel = QWidget()
        # Set size policy for right panel to allow free resizing
        right_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_label = QLabel("<b>Calculation Result</b>")
        right_layout.addWidget(result_label)

        # Result display. The real viewer is built straight away, empty, so its
        # toolbar — Copy / Save / Plot, Set X, the axis switches — is there from
        # the moment the calculator opens rather than appearing after the first
        # calculation.
        self.result_container = QWidget()
        self.result_container_layout = QVBoxLayout()
        self.result_container_layout.setContentsMargins(0, 0, 0, 0)
        self._create_result_viewer()

        self.result_container.setLayout(self.result_container_layout)
        right_layout.addWidget(self.result_container)

        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)

        # The result viewer takes the rest; it has a toolbar to fit.
        splitter.setSizes([CONTROLS_PANEL_WIDTH, 1200 - CONTROLS_PANEL_WIDTH])

        main_layout.addWidget(splitter)

        self.setLayout(main_layout)

        # Populate datasets asynchronously to avoid blocking on network files
        QTimer.singleShot(0, self._start_dataset_population)

    def keyPressEvent(self, event) -> None:
        """Ensure Enter in dataset path fields only validates paths, never triggers calculation."""
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            fw = self.focusWidget()
            le_a = self.combo_dataset_a.lineEdit()
            le_b = self.combo_dataset_b.lineEdit()
            if fw is le_a:
                self._on_dataset_line_entered(self.combo_dataset_a, "A")
                event.accept()
                return
            if fw is le_b:
                self._on_dataset_line_entered(self.combo_dataset_b, "B")
                event.accept()
                return
        super().keyPressEvent(event)

    def _populate_datasets_from_keys(self, full_keys: list[str]) -> None:
        """Populate combo boxes from pre-collected full keys."""
        prev_a = self.combo_dataset_a.get_entry(opened_files=self.opened_files)
        prev_b = self.combo_dataset_b.get_entry(opened_files=self.opened_files)
        self._is_populating_combos = True
        # Reset combo boxes with placeholder (like FTH tool behavior)
        self.combo_dataset_a.clear_datasets()
        self.combo_dataset_b.clear_datasets()
        self.combo_dataset_a.blockSignals(True)
        self.combo_dataset_b.blockSignals(True)

        # Add to combo boxes in linear time (avoid add_full_key O(n^2) checks here)
        for full_key in full_keys:
            short = DatasetPathCombo.short_display_from_full_key(full_key)
            self.combo_dataset_a.addItem(short, userData=full_key)
            self.combo_dataset_b.addItem(short, userData=full_key)
        key_set = set(full_keys)
        if prev_a is not None:
            k = f"{prev_a[0]}::{prev_a[1]}"
            if k in key_set:
                self.combo_dataset_a.add_full_key(k, select=True)
        if prev_b is not None:
            k = f"{prev_b[0]}::{prev_b[1]}"
            if k in key_set:
                self.combo_dataset_b.add_full_key(k, select=True)
        self.combo_dataset_a.blockSignals(False)
        self.combo_dataset_b.blockSignals(False)
        self._is_populating_combos = False
        self.label_info_a.setText("Shape: -, Type: -")
        self.label_info_b.setText("Shape: -, Type: -")
        self.spin_col_a.clear(); self.spin_col_a.addItem("N/A", -1); self.spin_col_a.setEnabled(False)
        self.spin_col_b.clear(); self.spin_col_b.addItem("N/A", -1); self.spin_col_b.setEnabled(False)

        # Do not warn during index warming batches; validation is handled on calculate.

    def _start_dataset_population(self) -> None:
        """Populate dataset list from shared index only (no local file scan)."""
        if self.dataset_full_keys_1d:
            self._populate_datasets_from_keys(self.dataset_full_keys_1d)
            self.combo_dataset_a.setEnabled(True)
            self.combo_dataset_b.setEnabled(True)
            return

        # Shared index may still be warming; keep controls usable for manual typing.
        self.combo_dataset_a.setEnabled(True)
        self.combo_dataset_b.setEnabled(True)
        self.label_info_a.setText("Waiting for shared index...")
        self.label_info_b.setText("Waiting for shared index...")

    def refresh_dataset_keys(
        self,
        full_keys_1d: list[str],
        opened_files: tuple[pathlib.Path, ...] | None = None,
    ) -> None:
        """Refresh dataset combo candidates from shared index."""
        if opened_files is not None:
            self.opened_files = tuple(opened_files)
        new_sig = tuple(full_keys_1d)
        if new_sig == self._last_keys_sig:
            return
        self._last_keys_sig = new_sig
        self.dataset_full_keys_1d = list(full_keys_1d)
        self._populate_datasets_from_keys(self.dataset_full_keys_1d)
        self.combo_dataset_a.setEnabled(True)
        self.combo_dataset_b.setEnabled(True)
        if self.result_widget is not None:
            try:
                from src.gui.unified_data_viewer import UnifiedDataViewer

                if isinstance(self.result_widget, UnifiedDataViewer):
                    self.result_widget.refresh_dataset_keys(
                        self.dataset_full_keys_1d,
                        opened_files=self.opened_files,
                    )
            except Exception:
                pass

    def _resolve_dataset_from_combo(self, combo: DatasetPathCombo) -> tuple[pathlib.Path, str] | None:
        """Resolve combo selection from currentData or editable text."""
        entry = combo.get_entry(opened_files=self.opened_files)
        if entry is None:
            return None
        return pathlib.Path(entry[0]), entry[1]

    def _normalize_full_key(self, full_key: str) -> str | None:
        """Normalize 'file_token::dataset' to absolute-file full key if possible."""
        txt = (full_key or "").strip()
        if "::" not in txt:
            return None
        file_token, ds_path = txt.split("::", 1)
        file_token = file_token.strip()
        ds_path = ds_path.strip()
        if not file_token or not ds_path:
            return None
        for fp in self.opened_files:
            fp_str = str(fp)
            if fp_str == file_token or pathlib.Path(fp_str).name == file_token:
                return f"{fp_str}::{ds_path}"
        return None

    def _build_part_row(self, form: QFormLayout, side: str):
        """The row that says which part of an operand to use.

        One row, two shapes: a 2-D dataset offers its columns, a 3-D one offers
        an axis and a frame along it. They are the same question — take a piece
        out, or use the whole thing — so they share a row rather than each
        getting one that is empty most of the time.
        """
        row = QHBoxLayout()
        axis_label_widget = QLabel("Axis:")
        axis_label_widget.setVisible(False)
        row.addWidget(axis_label_widget)

        axis_combo = QComboBox()
        axis_combo.setToolTip("Which array axis the frames are counted along")
        # Sized from what it holds, and never below "Axis 0 (1200)" plus the
        # arrow: a combo's hint is otherwise settled the first time it is shown,
        # which here is before it has any items, and the entries then come out
        # elided to "Axis 0 (1...".
        axis_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        axis_combo.setMinimumWidth(AXIS_COMBO_MIN_WIDTH)
        axis_combo.setMaximumWidth(AXIS_COMBO_MAX_WIDTH)
        axis_combo.setVisible(False)
        axis_combo.currentIndexChanged.connect(
            lambda _i, s=side: self._on_axis_changed(s)
        )
        row.addWidget(axis_combo)

        label = QLabel("Column:")
        row.addWidget(label)

        part_combo = QComboBox()
        part_combo.setEnabled(False)
        # Sized from its entries, like the axis box beside it. The old 150px
        # minimum was set for "All columns"; the entries are values now.
        part_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        part_combo.setMinimumWidth(PART_COMBO_MIN_WIDTH)
        part_combo.setMaximumWidth(PART_COMBO_MAX_WIDTH)
        part_combo.setToolTip("Select column for multi-column datasets")
        row.addWidget(part_combo)
        row.addStretch()
        form.addRow("", row)
        # Kept together so _configure_part_row can show or hide the axis pair.
        label.axis_label = axis_label_widget
        return label, axis_combo, part_combo

    def _configure_part_row(self, side: str, shape: tuple[int, ...]) -> None:
        """Point one operand's row at the shape actually loaded."""
        label = self.lbl_part_a if side == "A" else self.lbl_part_b
        axis_combo = self.combo_axis_a if side == "A" else self.combo_axis_b
        part_combo = self.spin_col_a if side == "A" else self.spin_col_b

        previous = part_combo.currentData()
        part_combo.clear()

        # The entries carry the value only. Repeating "Slice" or "Column" in
        # every one of them, next to a label that already says it, is what made
        # these boxes twice as wide as anything they hold.
        if len(shape) >= 3:
            label.setText("Slice:")
            label.axis_label.setVisible(True)
            axis_combo.setVisible(True)
            axis = self._rebuild_axis_combo(axis_combo, shape)
            part_combo.addItem("All", SELECT_ALL)
            for index in range(shape[axis]):
                part_combo.addItem(str(index), index)
            part_combo.setEnabled(True)
            part_combo.setToolTip(
                "One frame of the stack, or all of them.\n"
                "A single frame against a whole stack subtracts that frame "
                "from every one of them."
            )
        elif len(shape) == 2:
            label.setText("Column:")
            label.axis_label.setVisible(False)
            axis_combo.setVisible(False)
            part_combo.addItem("All", SELECT_ALL)
            for index in range(shape[1]):
                part_combo.addItem(str(index), index)
            part_combo.setEnabled(True)
            part_combo.setToolTip("Select column for multi-column datasets")
        else:
            label.setText("Column:")
            label.axis_label.setVisible(False)
            axis_combo.setVisible(False)
            part_combo.addItem("N/A", SELECT_ALL)
            part_combo.setEnabled(False)
            return

        if previous is not None:
            for index in range(part_combo.count()):
                if part_combo.itemData(index) == previous:
                    part_combo.setCurrentIndex(index)
                    break

    @staticmethod
    def _rebuild_axis_combo(axis_combo: QComboBox, shape: tuple[int, ...]) -> int:
        """Fill an axis selector for ``shape`` and return the axis in force.

        Bare numbers here, where the viewer and the export dialog write
        ``0 (400)``: the shape is printed on the line directly above this row,
        so carrying each axis's length in the entries repeats what is already
        on screen and doubles the width of the box.
        """
        previous = axis_combo.currentData()
        axis_combo.blockSignals(True)
        axis_combo.clear()
        for axis in range(len(shape)):
            axis_combo.addItem(str(axis), axis)
        if previous is not None and 0 <= int(previous) < len(shape):
            axis_combo.setCurrentIndex(int(previous))
        axis_combo.blockSignals(False)
        return int(axis_combo.currentData() or 0)

    def _on_axis_changed(self, side: str) -> None:
        """Re-list the frames when the axis changes: a different axis, a
        different number of them."""
        if self._is_populating_combos:
            return
        combo = self.combo_dataset_a if side == "A" else self.combo_dataset_b
        shape = self._dataset_shape(combo)
        if shape is None or len(shape) < 3:
            return
        axis_combo = self.combo_axis_a if side == "A" else self.combo_axis_b
        part_combo = self.spin_col_a if side == "A" else self.spin_col_b
        axis = int(axis_combo.currentData() or 0)

        part_combo.blockSignals(True)
        part_combo.clear()
        part_combo.addItem("All", SELECT_ALL)
        for index in range(shape[axis]):
            part_combo.addItem(str(index), index)
        part_combo.blockSignals(False)

    def _dataset_shape(self, combo: QComboBox) -> tuple[int, ...] | None:
        """The shape of the dataset a combo points at, or None."""
        resolved = self._resolve_dataset_from_combo(combo)
        if not resolved:
            return None
        file_path, dataset_path = resolved
        try:
            with h5py.File(file_path, "r") as f:
                return tuple(f[dataset_path].shape)
        except Exception as exc:
            logging.warning("Could not read the shape of %s: %s", dataset_path, exc)
            return None

    def _operand_selection(self, side: str) -> tuple[int, int]:
        """``(axis, index)`` for one operand; index -1 means "the whole thing"."""
        axis_combo = self.combo_axis_a if side == "A" else self.combo_axis_b
        part_combo = self.spin_col_a if side == "A" else self.spin_col_b
        axis = axis_combo.currentData()
        index = part_combo.currentData()
        return (int(axis) if axis is not None else 0,
                int(index) if index is not None else SELECT_ALL)

    def _on_dataset_line_entered(self, combo: QComboBox, side: str) -> None:
        """Handle Enter in editable dataset combobox."""
        resolved = self._resolve_dataset_from_combo(combo)
        if resolved is None:
            QMessageBox.warning(
                self,
                "Dataset Not Found",
                "Cannot resolve typed dataset.\nUse format: filename::path/to/dataset"
            )
            return
        if side == "A":
            self._on_dataset_a_changed()
        else:
            self._on_dataset_b_changed()

    def _on_dataset_changed(self, side: str) -> None:
        """Re-describe one operand after its dataset selection changed.

        A and B were two near-identical copies of this; they are one method now,
        which is what let the 3-D case be added in a single place.
        """
        if self._is_populating_combos:
            return
        combo = self.combo_dataset_a if side == "A" else self.combo_dataset_b
        info = self.label_info_a if side == "A" else self.label_info_b
        part_combo = self.spin_col_a if side == "A" else self.spin_col_b

        resolved = self._resolve_dataset_from_combo(combo)
        if not resolved:
            info.setText("Shape: -, Type: -")
            self._configure_part_row(side, ())
            return

        file_path, dataset_path = resolved
        try:
            with h5py.File(file_path, "r") as f:
                dataset = f[dataset_path]
                shape = tuple(dataset.shape)
                dtype = dataset.dtype
        except Exception as exc:
            logging.error("Could not describe %s: %s", dataset_path, exc)
            info.setText(f"Error: {exc}")
            part_combo.clear()
            part_combo.setEnabled(False)
            return

        note = ""
        if len(shape) >= 3:
            note = f" | <b>{shape[0]} slices</b>"
        elif len(shape) == 2:
            note = f" | <b>{shape[1]} columns</b>"
        info.setText(f"Shape: {shape}, Type: {dtype}{note}")
        self._configure_part_row(side, shape)

    def _on_dataset_a_changed(self) -> None:
        """Update dataset A info label and part selector when selection changes."""
        self._on_dataset_changed("A")

    def _on_dataset_b_changed(self) -> None:
        """Update dataset B info label and part selector when selection changes."""
        self._on_dataset_changed("B")

    @staticmethod
    def _fft_magnitude(data: np.ndarray) -> np.ndarray:
        """Return centered FFT magnitude for 1D/2D/ND numeric data."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 0:
            return np.abs(np.fft.fft(np.atleast_1d(arr)))
        if arr.ndim == 1:
            return np.abs(np.fft.fftshift(np.fft.fft(arr)))
        return np.abs(np.fft.fftshift(np.fft.fftn(arr)))

    @staticmethod
    def _expression_uses_b(expression: str) -> bool:
        """Return True when expression references the Dataset B variable."""
        return re.search(r"\bB\b", expression or "") is not None

    def _load_datasets(self, *, require_b: bool = False) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Load selected datasets with column selection support. Dataset B is optional."""
        resolved_a = self._resolve_dataset_from_combo(self.combo_dataset_a)
        resolved_b = self._resolve_dataset_from_combo(self.combo_dataset_b)
        if not resolved_a:
            QMessageBox.warning(self, "No Selection", "Please select Dataset A.")
            return None
        if require_b and not resolved_b:
            QMessageBox.warning(self, "No Selection", "This expression requires Dataset B.")
            return None

        try:
            # Load Dataset A
            file_path_a, dataset_path_a = resolved_a
            from src.lib_h5.file_validator import is_hdf5_file
            if is_hdf5_file(file_path_a):
                with h5py.File(file_path_a, "r") as f:
                    data_a = np.array(f[dataset_path_a])
            else:
                from src.gui.main_window import load_regular_data_file
                data_a = load_regular_data_file(file_path_a)

            # Cut A down to the column, or the frame, that was asked for.
            axis_a, index_a = self._operand_selection("A")
            data_a = operand_from_selection(data_a, axis_a, index_a)
            logging.info("Dataset A: axis=%s index=%s -> %s", axis_a, index_a, data_a.shape)

            data_b = None
            if resolved_b:
                # Load Dataset B
                file_path_b, dataset_path_b = resolved_b
                if is_hdf5_file(file_path_b):
                    with h5py.File(file_path_b, "r") as f:
                        data_b = np.array(f[dataset_path_b])
                else:
                    from src.gui.main_window import load_regular_data_file
                    data_b = load_regular_data_file(file_path_b)

                axis_b, index_b = self._operand_selection("B")
                data_b = operand_from_selection(data_b, axis_b, index_b)
                logging.info("Dataset B: axis=%s index=%s -> %s", axis_b, index_b, data_b.shape)

            return data_a, data_b

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", f"Failed to load datasets:\n{e}")
            return None

    def _perform_operation(self, expression: str) -> None:
        """
        Perform a predefined operation and display result in right panel.

        :param expression: Expression string (e.g., "A + B")
        """
        require_b = self._expression_uses_b(expression)
        datasets = self._load_datasets(require_b=require_b)
        if datasets is None:
            return

        data_a, data_b = datasets

        # Ask only about shapes numpy genuinely cannot pair up. Different
        # shapes that broadcast are not a mismatch — one frame against a whole
        # stack is a dark-frame subtraction, and asking "attempt anyway?" every
        # time would make the most ordinary use of a stack read like a mistake.
        if data_b is not None and not _shapes_broadcast(data_a.shape, data_b.shape):
            reply = QMessageBox.question(
                self,
                "Shape Mismatch",
                f"Dataset A shape: {data_a.shape}\n"
                f"Dataset B shape: {data_b.shape}\n\n"
                f"These shapes cannot be combined. Attempt the operation anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        try:
            self._last_operation_expr = expression
            # Save original data for export
            self.data_a = data_a
            self.data_b = data_b

            # Bind user data by name. Convert to float64 to handle negative
            # values from subtraction.
            variables: dict[str, Any] = {"A": data_a.astype(np.float64)}
            if data_b is not None:
                variables["B"] = data_b.astype(np.float64)

            # Evaluate through the whitelisted evaluator (src.recon.expr), which
            # rejects attribute/name escapes before compiling the expression.
            self.result_data = np.asarray(
                _evaluate_expression(expression, variables, {"FFT": self._fft_magnitude})
            )

            # Automatically squeeze out dimensions of size 1
            self.result_data = np.squeeze(self.result_data)

            # Update result display in right panel
            self._update_result_display()

            # Enable export button
            self.btn_export.setEnabled(True)
            self.btn_plot.setEnabled(True)
            can_transfer = False
            if self.result_data is not None:
                ndim = int(getattr(self.result_data, "ndim", 0))
                if ndim == 1:
                    can_transfer = True
                elif ndim == 2:
                    try:
                        can_transfer = int(self.result_data.shape[1]) < 100
                    except Exception:
                        can_transfer = False
            self.btn_transfer_to_comparison.setEnabled(can_transfer)

            logging.info(f"Calculation successful: {expression}")

        except ExpressionError as e:
            QMessageBox.critical(
                self,
                "Invalid Expression",
                f"Cannot evaluate:\n  {expression}\n\n{e}\n\n"
                "Allowed: A, optional B, arithmetic, FFT(), sqrt()/log()/abs()..., np.*",
            )
            logging.warning(f"Rejected expression '{expression}': {e}")

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to perform calculation:\n{e}")
            logging.error(f"Calculation error: {e}")

    def _create_result_viewer(self) -> None:
        """Build the result viewer, empty, and put it in the container."""
        from src.gui.unified_data_viewer import UnifiedDataViewer

        for i in reversed(range(self.result_container_layout.count())):
            widget = self.result_container_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        viewer = UnifiedDataViewer(
            parent=self,
            opened_files=self.opened_files,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
        )
        viewer.q_calibration_requested.connect(self._on_q_request_from_result_viewer)
        # An empty curve: it draws nothing but brings the toolbar with it.
        viewer.set_data(np.zeros(0), source_dataset_key="Result::calculation_result")
        self.result_container_layout.addWidget(viewer)
        self.result_widget = viewer

    def reset_results(self) -> None:
        """Throw away the calculated result and the arrays it was made from.

        The window is kept alive between uses so that reopening is instant, but
        that made closing it look like a reset it was not: it came back showing
        the previous result, which reads as a result for whatever is selected
        now. The A/B selections are left alone — they are the question, not the
        answer, and both are re-read from the files on the next calculation.
        """
        self.result_data = None
        self.data_a = None
        self.data_b = None
        self._last_operation_expr = ""

        from src.gui.unified_data_viewer import UnifiedDataViewer

        if isinstance(self.result_widget, UnifiedDataViewer):
            self.result_widget.set_data(np.zeros(0), source_dataset_key="Result::calculation_result")
        else:
            self._create_result_viewer()

        # Nothing to export, plot or transfer until something is calculated.
        for button in (self.btn_export, self.btn_plot, self.btn_transfer_to_comparison):
            button.setEnabled(False)
        self._update_output_button()

    def closeEvent(self, event) -> None:
        """Closing the window means starting over next time."""
        self.reset_results()
        super().closeEvent(event)

    def _update_result_display(self) -> None:
        """Show the calculation result in the viewer.

        The viewer is reused rather than rebuilt, so the toolbar does not blink
        out of existence between calculations.
        """
        from src.gui.unified_data_viewer import UnifiedDataViewer

        if not isinstance(self.result_widget, UnifiedDataViewer):
            self._create_result_viewer()

        viewer = self.result_widget
        assert isinstance(viewer, UnifiedDataViewer)
        viewer.set_data(self.result_data, source_dataset_key="Result::calculation_result")

        # The axis was chosen per operand and the result reoriented to match, so
        # the viewer's own axis selector would be a second control for a
        # settled question. The slider stays: browsing the frames is still
        # worth doing.
        shown = viewer.get_current_widget()
        if hasattr(shown, "set_axis_choice_visible"):
            shown.set_axis_choice_visible(False)
        # The one place a result reaches the viewer, so the one place the button
        # has to be told what kind of result it is now offering to act on.
        self._update_output_button()

    def _on_q_request_from_result_viewer(self, _source_dataset_key: object) -> None:
        """Send what the result viewer is showing to the pattern tool.

        For a stack that means the frame on screen. The guard used to be
        ``ndim == 2``, so pressing Q on a stack did nothing at all — no tool,
        no message — and now that a calculation can produce a stack that is a
        button that appears to be broken.
        """
        mw = self.parent()
        try:
            if mw is None or self.result_data is None or not hasattr(mw, "open_q_tool_for_array"):
                return

            data = np.asarray(self.result_data)
            label = self._result_label()
            if data.ndim > 2:
                widget = self._result_image_widget()
                index = int(getattr(widget, "current_slice_index", 0)) if widget else 0
                axis = int(getattr(widget, "current_slice_axis", 0)) if widget else 0
                data = operand_from_selection(data, axis, min(index, data.shape[axis % data.ndim] - 1))
                label = f"{label} [axis {axis}, slice {index}]"
            if data.ndim != 2:
                return

            mw.open_q_tool_for_array(data, source_label=label)
        except Exception as e:
            logging.error(f"Failed to open Q calibration tool from calculator: {e}")

    def _perform_custom_operation(self) -> None:
        """Perform a custom operation from user input."""
        expression = self.edit_custom.text().strip()
        if not expression:
            QMessageBox.warning(self, "No Expression", "Please enter a custom expression.")
            return

        self._perform_operation(expression)

    def _default_transfer_label(self) -> str:
        """Build transfer label: head_numA_numB."""
        entry_a = self.combo_dataset_a.get_entry(opened_files=self.opened_files)
        entry_b = self.combo_dataset_b.get_entry(opened_files=self.opened_files)
        if entry_a and entry_b:
            head_a, n1 = self._scan_head_and_number(str(entry_a[0]))
            head_b, n2 = self._scan_head_and_number(str(entry_b[0]))
            head = head_a if head_a else (head_b if head_b else "scan")
            return f"{head}_{n1}_{n2}"
        return "calc_result"

    def _transfer_result_to_comparison(self) -> None:
        """Send current result to Data Comparison without disrupting existing datasets."""
        if self.result_data is None:
            QMessageBox.warning(self, "No Result", "Please calculate a result first.")
            return
        arr = np.asarray(self.result_data)
        if arr.ndim == 2 and int(arr.shape[1]) >= 100:
            QMessageBox.warning(
                self,
                "Comparison Limit",
                f"Result has {arr.shape[1]} columns (>=100).\n"
                "Transfer supports up to 99 columns.",
            )
            return
        if arr.ndim not in (1, 2):
            QMessageBox.warning(
                self,
                "Unsupported Result",
                f"Transfer supports 1D/2D only.\nCurrent ndim: {arr.ndim}",
            )
            return
        mw = self.parent()
        if mw is None or not hasattr(mw, "transfer_calculator_result_to_comparison"):
            QMessageBox.warning(self, "Transfer Failed", "Main window bridge unavailable.")
            return
        label = self._default_transfer_label()
        try:
            payload = arr
            if arr.ndim == 2 and 2 <= int(arr.shape[1]) <= 10:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Question)
                msg.setWindowTitle("Transfer 2D Result")
                msg.setText(f"Result has {arr.shape[1]} columns.")
                msg.setInformativeText("Choose transfer mode:")
                msg.addButton("Transfer all columns", QMessageBox.ButtonRole.AcceptRole)
                btn_one = msg.addButton("Transfer one column", QMessageBox.ButtonRole.ActionRole)
                btn_cancel = msg.addButton(QMessageBox.StandardButton.Cancel)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn_cancel:
                    return
                if clicked == btn_one:
                    col, ok = QInputDialog.getInt(
                        self,
                        "Select Column",
                        f"Column index (0 to {arr.shape[1] - 1}):",
                        0,
                        0,
                        int(arr.shape[1] - 1),
                        1,
                    )
                    if not ok:
                        return
                    payload = arr[:, int(col)]
                else:
                    payload = arr

            ok = bool(mw.transfer_calculator_result_to_comparison(label, payload))
            if not ok:
                QMessageBox.warning(self, "Transfer Failed", "Could not transfer result to comparison.")
        except Exception as e:
            QMessageBox.critical(self, "Transfer Failed", f"Failed to transfer result:\n{e}")

    @staticmethod
    def _scan_head_and_number(file_path: str) -> tuple[str, str]:
        """Extract (head, scan_number) from file stem, e.g. scanx_0035 -> (scanx, 0035)."""
        stem = pathlib.Path(file_path).stem
        m = re.match(r"^(.*?)(\d+)$", stem)
        if m:
            head = m.group(1).rstrip("_- ")
            num = m.group(2)
            return (head or "scan"), num
        return stem.rstrip("_- ") or "scan", "0000"

    @staticmethod
    def _suffix_from_expression(expr: str) -> str:
        """Map operation expression to concise filename suffix."""
        key = " ".join((expr or "").split()).lower()
        mapping = {
            "a + b": "sum",
            "a - b": "diff",
            "(a + b) / 2": "mean",
            "np.abs(a - b)": "absdiff",
            "(a - b) / (a + b)": "asy",
            "a * b": "mul",
            "a / b": "div",
            "fft(a)": "fft",
        }
        return mapping.get(key, "calcu")

    def _operand_key(self, combo) -> str | None:
        """The ``file::dataset`` an operand combo currently points at."""
        entry = combo.get_entry(opened_files=self.opened_files)
        return f"{entry[0]}::{entry[1]}" if entry else None

    def _operand_label(self, combo, col_combo, fallback: str) -> str:
        """Name an operand the way every other curve in the application is named.

        ``scanx_0083_data_04_col2`` — scan, dataset, and the column when one was
        picked out. ``Data_A`` said only which side of the expression it was on,
        which is no help at all once the figure leaves the calculator.
        """
        key = self._operand_key(combo)
        if not key:
            return fallback
        column = col_combo.currentData()
        if column is not None and column >= 0:
            # The suffix short_series_label reads a column out of.
            key = f"{key} [Col {column}]"
        return short_series_label(key, fallback=fallback)

    def _result_label(self) -> str:
        """Name the result after what was done, not after the word "result".

        With the operands carrying their scan and dataset names, the one thing
        the third curve still has to say is which operation it came from.
        """
        expression = " ".join((self._last_operation_expr or "").split())
        return expression or "Result"

    def _default_export_base_name(self) -> str:
        """Name the result after the two operands and what was done to them.

        Whatever distinguishes A from B is what goes in the name — two scans,
        two datasets of one scan, or two columns of one dataset. See
        :func:`src.gui.export_naming.pair_label`.
        """
        base = pair_label(
            self._operand_key(self.combo_dataset_a),
            self._operand_key(self.combo_dataset_b),
        )
        suffix = self._suffix_from_expression(self._last_operation_expr)
        return f"{base}_{suffix}"

    def _result_plot_widget(self):
        """The 1-D curve widget inside the result viewer, or None."""
        from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
        from src.gui.unified_data_viewer import UnifiedDataViewer

        widget = self.result_widget
        if isinstance(widget, UnifiedDataViewer):
            widget = widget.get_current_widget()
        return widget if isinstance(widget, PlotWidget1DEnhanced) else None

    def _result_image_widget(self):
        """The 2-D image widget inside the result viewer, or None."""
        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
        from src.gui.unified_data_viewer import UnifiedDataViewer

        widget = self.result_widget
        if isinstance(widget, UnifiedDataViewer):
            widget = widget.get_current_widget()
        return widget if isinstance(widget, ImageView2DEnhanced) else None

    def result_is_image(self) -> bool:
        """Whether the result is being *shown* as an image rather than a curve.

        Asked of the viewer, not of the array's shape: a 2-D result with a few
        columns is drawn as several curves, and going by ``ndim`` would offer
        Copy for something the user is looking at as a plot.
        """
        return self._result_image_widget() is not None

    def _update_output_button(self) -> None:
        """Point the second output button at whatever this result can do.

        Plotting an image would flatten it into one very long curve, which is
        not a picture of anything; copying it is the useful thing there.
        """
        image = self.result_is_image()
        self.btn_plot.setText(COPY_BUTTON_TEXT if image else PLOT_BUTTON_TEXT)
        self.btn_plot.setToolTip(COPY_BUTTON_TOOLTIP if image else PLOT_BUTTON_TOOLTIP)

    def _on_plot_or_copy(self) -> None:
        """Whichever of the two the button is currently offering."""
        if self.result_is_image():
            self.copy_result_image()
        else:
            self.open_plot()

    def copy_result_image(self) -> None:
        """Put the result image on the clipboard, colormap and all."""
        widget = self._result_image_widget()
        if widget is None:
            QMessageBox.information(self, "No Image", "Calculate a 2-D result first.")
            return
        widget.quick_copy()

    def _displayed_result(self) -> np.ndarray | None:
        """The result as the viewer is showing it, despiked if its filter is on.

        The viewer's Despike button is where a glitch in a calculated curve gets
        removed, so the plot and the export have to follow it. Reading
        ``result_data`` directly would put the spike back in both, which is the
        worst outcome: the filter would look like it had worked.
        """
        widget = self._result_plot_widget()
        displayed = getattr(widget, "display_y", None) if widget is not None else None
        if displayed is None or self.result_data is None:
            return self.result_data
        # A shape mismatch means the viewer is showing something else entirely.
        if np.shape(displayed) != np.shape(self.result_data):
            return self.result_data
        return displayed

    def _despike_comment(self) -> list[str]:
        """The line the export carries when the result was filtered."""
        widget = self._result_plot_widget()
        if widget is None:
            return []
        return widget._despike_comment()

    def _viewer_x_dataset_key(self) -> str | None:
        """The X dataset the result viewer is currently using, if any."""
        widget = self._result_plot_widget()
        return getattr(widget, "x_dataset_path", None) if widget is not None else None

    def can_take_x(self) -> bool:
        """Whether there is a result for an X axis to go against.

        Being the window in front is a claim on ``Set X``. With nothing
        calculated the claim has to be dropped, or the choice is swallowed here
        and never reaches the viewer in the main window — which is what closing
        the calculator, and so clearing its result, started causing.
        """
        widget = self._result_plot_widget()
        y_data = getattr(widget, "y_data", None) if widget is not None else None
        return y_data is not None and bool(len(y_data))

    def set_x_dataset(self, key: str) -> bool:
        """Take an X dataset handed over by the tree's ``Set X``.

        Applied to the result viewer, which is where the export and the plot
        both read their X from — the same place the viewer's own ``Set X``
        button writes it.
        """
        widget = self._result_plot_widget()
        if widget is None or getattr(widget, "y_data", None) is None or not len(widget.y_data):
            QMessageBox.information(self, "No Result", "Run a calculation first.")
            return False

        try:
            x_data = read_x_dataset(key)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Use As X", str(exc))
            return False
        except Exception as exc:
            logging.warning("Could not read %s as X: %s", key, exc)
            QMessageBox.warning(self, "Cannot Use As X", f"Could not read the dataset:\n{exc}")
            return False

        if len(x_data) != len(widget.y_data):
            QMessageBox.warning(
                self,
                "Length Mismatch",
                f"The X dataset is {len(x_data)} long and the result is "
                f"{len(widget.y_data)}, so nothing was changed.",
            )
            return False

        widget._on_x_data_selected(x_data, key)
        return True

    def _result_export_columns(self, x_key: str | None) -> tuple[list[str], list[Any]]:
        """Build the columns a full export would write: optional X, A, B, Result."""
        headers: list[str] = []
        columns: list[Any] = []

        if x_key:
            x_data = self._load_full_key_1d(x_key)
            if x_data is not None:
                headers.append(x_column_header(x_key))
                columns.append(x_data)

        if self.data_a is not None:
            headers.append(self._operand_label(self.combo_dataset_a, self.spin_col_a, "Data_A"))
            columns.append(np.ravel(self.data_a))
        if self.data_b is not None:
            headers.append(self._operand_label(self.combo_dataset_b, self.spin_col_b, "Data_B"))
            columns.append(np.ravel(self.data_b))

        # The result goes last, and everything downstream relies on that rather
        # than on matching its name — which is no longer a fixed word.
        headers.append(self._result_label())
        columns.append(np.ravel(self._displayed_result()))
        return headers, columns

    def plot_series(self) -> list:
        """The operands and the result as plot Series, against the chosen X.

        Same source as the export columns, so the figure cannot disagree with
        the file.
        """
        from src.gui.plot_series import series_from_table

        if self.result_data is None:
            return []
        headers, columns = self._result_export_columns(self._viewer_x_dataset_key())
        return series_from_table(headers, columns)

    def plot_axes_series(self) -> tuple[list, list]:
        """The curves split by axis: ``(operands, result)``.

        A calculator result rarely shares a scale with its operands — a ratio
        against raw counts, an FFT against a sweep — so putting them on one axis
        flattens whichever is smaller into the baseline. The operands go on the
        left, the result on its own axis to the right.
        """
        series = self.plot_series()
        if not series:
            return ([], [])
        # By position, not by name: the result curve is now labelled with the
        # expression, so there is no fixed word left to match on.
        operands, result = series[:-1], series[-1:]
        # With no operands to compare against, a second axis buys nothing.
        return (operands, result) if operands else (result, [])

    def open_plot(self) -> None:
        """Open a Plot window on the result: the Export button's twin."""
        from src.gui.plot_dialog import open_plot_dialog

        operands, result = self.plot_axes_series()
        if not operands and not result:
            QMessageBox.information(self, "No Result", "Run a calculation first.")
            return
        open_plot_dialog(
            self,
            operands,
            right_series=result,
            title="Calculator",
            # The right axis carries the result, so it is labelled with the same
            # expression its curve is.
            right_label=self._result_label(),
        )

    def _load_full_key_1d(self, full_key: str) -> np.ndarray | None:
        """Read a 1-D dataset given a ``file::dataset`` key."""
        try:
            file_part, ds_path = full_key.split("::", 1)
            with h5py.File(file_part, "r") as f:
                if ds_path not in f:
                    return None
                arr = np.asarray(f[ds_path][()]).squeeze()
            return arr if arr.ndim == 1 else None
        except Exception as exc:
            logging.warning("Could not read X dataset %s: %s", full_key, exc)
            return None

    def _export_result_full(self) -> None:
        """Open the full-export dialog for a 1-D result.

        Shown non-modally so a dataset can still be dragged out of the main
        window's tree into the dialog's X field.
        """
        dialog = ResultExportDialog(
            self,
            opened_files=self.opened_files,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
            preferred_x_key=self._viewer_x_dataset_key(),
            expression=self._last_operation_expr,
        )

        def refresh() -> None:
            headers, columns = self._result_export_columns(dialog.x_key())
            dialog.show_preview(headers, columns)

        dialog.settings_changed.connect(refresh)
        dialog.accepted.connect(lambda d=dialog: self._finish_result_export(d))
        dialog.rejected.connect(dialog.deleteLater)
        refresh()

        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._export_dialog = dialog

    def _finish_result_export(self, dialog: "ResultExportDialog") -> None:
        """Write the file once the non-modal settings dialog is accepted."""
        fmt = dialog.table_format()
        headers, columns = self._result_export_columns(dialog.x_key())
        dialog.deleteLater()

        from PyQt6.QtWidgets import QFileDialog

        file_path, _selected = QFileDialog.getSaveFileName(
            self,
            "Export Calculation Result",
            suggested_save_path(self._default_export_base_name(), extension=fmt.suffix),
            dialog_filter(fmt),
        )
        if not file_path:
            return

        export_path = pathlib.Path(file_path)
        if not export_path.suffix:
            export_path = export_path.with_suffix(fmt.suffix)
        remember_save_directory(export_path)
        QSettings().setValue("export/table_format", fmt.key)

        try:
            write_table(export_path, headers, columns, fmt, comments=self._despike_comment())
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export result:\n{exc}")
            logging.error("Calculator export failed: %s", exc)
            return

        QMessageBox.information(
            self,
            "Export Successful",
            f"Result exported to:\n{export_path}\n\nColumns: {', '.join(headers)}",
        )
        logging.info("Exported calculator result to %s", export_path)

    def _export_result(self) -> None:
        """Export the result to a file."""
        if self.result_data is None:
            return

        try:
            from PyQt6.QtWidgets import QFileDialog
            from src.lib_h5.data_exporter import DataExporter
            from src.lib_h5.dataset_types import H5DatasetType

            # Log shape for debugging
            logging.info(f"Exporting result - Shape: {self.result_data.shape}, Dims: {len(self.result_data.shape)}")

            # Determine data type based on dimensions and size
            ndim = len(self.result_data.shape)

            # 1D results get the full dialog: operands + result, with an optional X.
            if ndim == 1:
                self._export_result_full()
                return

            if ndim == 2:
                # 2D array - decide between table and image
                if self.result_data.size < 10000:
                    # Small 2D array - can be table or image
                    data_type = H5DatasetType.Array2D
                else:
                    # Large 2D array - prefer image export
                    data_type = H5DatasetType.Array2D
            else:
                # 3D or higher - try to export as Array2D (will use first slice or flatten)
                logging.warning(f"Exporting {ndim}D array - will attempt Array2D export")
                data_type = H5DatasetType.Array2D

            # Get file filter
            file_filter = DataExporter.get_export_filter(data_type)
            default_ext = DataExporter.get_default_extension(data_type)

            # Show save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Calculation Result",
                suggested_save_path(self._default_export_base_name(), extension=default_ext),
                file_filter,
            )

            if not file_path:
                return
            remember_save_directory(file_path)

            # Export
            success = DataExporter.export_data(self.result_data, pathlib.Path(file_path), data_type)

            if success:
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Result exported to:\n{file_path}\n\n"
                    f"Shape: {self.result_data.shape}, Type: {self.result_data.dtype}"
                )
                logging.info(f"Successfully exported to: {file_path}")
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export result to:\n{file_path}"
                )

        except Exception as e:
            error_msg = (
                f"Failed to export result:\n{e}\n\n"
                f"Shape: {self.result_data.shape}, Type: {self.result_data.dtype}"
            )
            QMessageBox.critical(self, "Export Error", error_msg)
            logging.error(f"Export error: {e}, shape: {self.result_data.shape}")

    def add_to_dataset_a(self, dataset_path: str) -> None:
        """
        Add dataset to Calculator A from context menu.

        :param dataset_path: Full path to dataset (format: filename::dataset_path)
        """
        logging.info(f"add_to_dataset_a: START, dataset_path={dataset_path}")
        if not self.combo_dataset_a.isEnabled():
            QMessageBox.information(
                self,
                "Still Loading",
                "Dataset list is still loading. Please try again in a moment.",
            )
            return

        if self.combo_dataset_a._try_select_by_text(dataset_path):
            idx = self.combo_dataset_a.currentIndex()
            logging.info(f"Added to Calculator A: {self.combo_dataset_a.itemText(idx)}")
            self._on_dataset_a_changed()
            logging.info("add_to_dataset_a: END")
            return

        normalized = self._normalize_full_key(dataset_path)
        if normalized is not None:
            self.combo_dataset_a.add_full_key(normalized, select=True)
            self._on_dataset_a_changed()
            logging.info(f"Added to Calculator A via fallback insert: {normalized}")
            logging.info("add_to_dataset_a: END")
            return

        logging.warning(f"Dataset not found in combo box A: {dataset_path}")
        logging.info("add_to_dataset_a: END (not found)")
        QMessageBox.warning(
            self,
            "Dataset Not Found",
            f"Could not find dataset in the list:\n{dataset_path}\n\n"
            "Make sure the file is opened in the main window."
        )

    def add_to_dataset_b(self, dataset_path: str) -> None:
        """
        Add dataset to Calculator B from context menu.

        :param dataset_path: Full path to dataset (format: filename::dataset_path)
        """
        if not self.combo_dataset_b.isEnabled():
            QMessageBox.information(
                self,
                "Still Loading",
                "Dataset list is still loading. Please try again in a moment.",
            )
            return

        if self.combo_dataset_b._try_select_by_text(dataset_path):
            idx = self.combo_dataset_b.currentIndex()
            logging.info(f"Added to Calculator B: {self.combo_dataset_b.itemText(idx)}")
            self._on_dataset_b_changed()
            return

        normalized = self._normalize_full_key(dataset_path)
        if normalized is not None:
            self.combo_dataset_b.add_full_key(normalized, select=True)
            self._on_dataset_b_changed()
            logging.info(f"Added to Calculator B via fallback insert: {normalized}")
            return

        logging.warning(f"Dataset not found in combo box B: {dataset_path}")
        QMessageBox.warning(
            self,
            "Dataset Not Found",
            f"Could not find dataset in the list:\n{dataset_path}\n\n"
            "Make sure the file is opened in the main window."
        )

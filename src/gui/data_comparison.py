"""Data Comparison Tool for comparing multiple 1D datasets."""

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
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.gui._shared import AXIS_Q, quick_icon_button, set_axis_label
from src.gui.export_naming import (
    pair_label,
    remember_save_directory,
    short_series_label,
    suggested_save_path,
    x_column_header,
)
from src.gui.batch_export import compact_combo
from src.gui.table_model import CopyableTableView, DataTable
from src.gui.plot_context_menu import attach_plot_menu
from src.gui.x_target import (
    read_x_dataset,
    register_x_target,
    remembered_x_dataset,
    x_scope_of,
)
from src.lib_h5.table_format import (
    DEFAULT_TABLE_FORMAT_KEY,
    TableFormat,
    dialog_filter,
    format_from_filter,
    format_labels,
    get_table_format,
    save_dialog_filter,
)
from src.lib_h5.table_writer import write_table
from src.recon.expr import ExpressionError, evaluate_series

# Table column layout. The order matches the toggle buttons above the table.
COL_NAME = 0
COL_POINTS = 1
COL_ENERGY = 2
COL_FY = 3
COL_XAXIS = 4
COL_FX = 5
COLUMN_COUNT = 6

# Columns holding a Curve Transform expression.
EXPR_COLUMNS = (COL_FY, COL_FX)

FX_PLACEHOLDER = "x"
FY_PLACEHOLDER = "y"

EXPR_HELP = (
    "Variables: y (this curve), x (its X axis), i (index), n (points), E (eV).\n"
    "Functions: sqrt log log10 exp abs sin cos gradient cumsum diff mean max min "
    "std sum clip where interp ..., plus np.*\n"
    "Examples:  y/max(y)   y-mean(y[:20])   log10(y)   gradient(y, x)   x*2+1\n"
    "Empty means no transform. f(X) is applied first, so f(Y) sees the new x."
)


class ComparisonExportDialog(QDialog):
    """Full-export settings for the comparison plot.

    Shows the exact columns that will be written. The X control mirrors the batch
    export dialog: an ``Export X`` switch plus a field a dataset can be dragged
    into. Left empty, the X shown in the table is used as-is.

    The dialog is used non-modally so the tree stays reachable for that drag —
    a modal dialog blocks input to the main window, which is exactly what stopped
    dragging from working.
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Comparison Data")
        self.setWindowFlag(Qt.WindowType.Window)
        # Square: the preview sits above the controls, so extra width buys nothing.
        self.resize(560, 560)

        root = QVBoxLayout(self)

        self.preview_table = CopyableTableView()
        root.addWidget(self.preview_table, stretch=1)

        box = QGroupBox("Export")
        form = QVBoxLayout(box)

        row_x = QHBoxLayout()
        self.chk_export_x = QCheckBox("Export X")
        self.chk_export_x.setChecked(True)
        row_x.addWidget(self.chk_export_x)
        # Starts from the X already chosen in the tree *for this tool*, so the
        # same dataset does not have to be found twice — while an X set in some
        # other tool, over a different set of files, never leaks in here.
        self.le_x_path = QLineEdit(remembered_x_dataset(x_scope_of(parent)))
        self.le_x_path.setPlaceholderText("Drag or type X dataset (blank = X as displayed)")
        self.le_x_path.setAcceptDrops(True)
        self.le_x_path.dragEnterEvent = self._x_path_drag_enter
        self.le_x_path.dropEvent = self._x_path_drop
        row_x.addWidget(self.le_x_path, stretch=1)
        form.addLayout(row_x)

        row = QHBoxLayout()
        row.addWidget(QLabel("Format:"))
        self.combo_format = compact_combo(QComboBox())
        self.combo_format.addItems(format_labels())
        stored = QSettings().value("export/table_format", DEFAULT_TABLE_FORMAT_KEY)
        self.combo_format.setCurrentText(get_table_format(str(stored)).label)
        row.addWidget(self.combo_format, stretch=1)
        form.addLayout(row)

        self.chk_comments = QCheckBox("Include metadata header")
        self.chk_comments.setChecked(True)
        self.chk_comments.setToolTip("Write the '#' lines describing energy and f(X)/f(Y) per curve")
        form.addWidget(self.chk_comments)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)
        root.addWidget(box)

        self.chk_export_x.stateChanged.connect(lambda _s: self.settings_changed.emit())
        self.le_x_path.textChanged.connect(lambda _t: self.settings_changed.emit())
        register_x_target(self)

    def set_x_dataset(self, key: str) -> bool:
        """Take an X dataset handed over by the tree's ``Set X``."""
        self.le_x_path.setText(key)
        self.chk_export_x.setChecked(True)
        return True

    def _x_path_drag_enter(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData().hasText():
            event.acceptProposedAction()

    def _x_path_drop(self, event: QDropEvent | None) -> None:
        """Accept a dataset dragged from the tree, keeping the full file::path key."""
        if event is None or not event.mimeData().hasText():
            return
        text = event.mimeData().text().strip().splitlines()[0].strip()
        if text:
            self.le_x_path.setText(text)
            self.chk_export_x.setChecked(True)
        event.acceptProposedAction()

    def export_x(self) -> bool:
        """Whether any X column is written at all."""
        return self.chk_export_x.isChecked()

    def x_key(self) -> str | None:
        """Explicit X dataset key, or None to use the X shown in the table."""
        if not self.chk_export_x.isChecked():
            return None
        return self.le_x_path.text().strip() or None

    def table_format(self) -> TableFormat:
        """Chosen tabular dialect."""
        return get_table_format(self.combo_format.currentText())

    def include_comments(self) -> bool:
        """Whether the '#' metadata lines are written."""
        return self.chk_comments.isChecked()

    def show_preview(self, headers: list[str], columns: list[Any]) -> None:
        """Fill the preview with the columns that would be written."""
        rows = min(200, max((len(col) for col in columns), default=0))
        table = np.full((rows, len(columns)), np.nan, dtype=float)
        for idx, col in enumerate(columns):
            take = min(rows, len(col))
            table[:take, idx] = np.asarray(col, dtype=float)[:take]
        self.preview_table.setModel(DataTable(table, headers))


@dataclass
class CurveEntry:
    """One row of the comparison table: a curve plus its per-row transform state."""

    name: str
    data: np.ndarray
    energy: float = 0.0
    y_expr: str = ""
    x_expr: str = ""
    x_data: np.ndarray | None = None
    x_path: str | None = None


class DatasetTableWidget(QTableWidget):
    """Custom TableWidget that accepts drag and drop of datasets with offset support."""

    def __init__(self, parent=None):
        """Initialize the table widget."""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.comparison_tool = None  # Will be set by DataComparisonTool

        # Setup table
        self.setColumnCount(COLUMN_COUNT)
        self.setHorizontalHeaderLabels(["Dataset", "Points", "E(eV)", "f(Y)", "X Axis", "f(X)"])

        # Configure columns
        header = self.horizontalHeader()
        if header:
            # Disable automatic stretching of last section to prevent resize jumps
            header.setStretchLastSection(False)

            # Use Interactive mode to allow user to resize columns
            # Avoid ResizeToContents which causes jumps during splitter drag
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

            # Set default widths for all columns
            self.setColumnWidth(COL_NAME, 200)
            self.setColumnWidth(COL_POINTS, 70)
            self.setColumnWidth(COL_ENERGY, 70)
            self.setColumnWidth(COL_FY, 140)
            self.setColumnWidth(COL_XAXIS, 130)
            self.setColumnWidth(COL_FX, 140)

        # Enable horizontal scrollbar when needed
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Set size policy - Ignored for horizontal to allow free resizing, Expanding for vertical
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        # Set size adjust policy - don't adjust widget size to contents
        self.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustIgnored)

        # Enable editing only for offset column
        self.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        """Handle drag enter events."""
        if event is None:
            return
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent | None) -> None:
        """Handle drag move events."""
        if event is None:
            return
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Handle drop events."""
        if event is None:
            return
        if not event.mimeData().hasText():
            return

        # Get dropped text (dataset path)
        dropped_text = event.mimeData().text().strip()
        logging.info(f"Dropped dataset: '{dropped_text}'")

        # Tell the comparison tool to add this dataset
        if self.comparison_tool:
            self.comparison_tool.add_dataset_from_path(dropped_text)
            event.acceptProposedAction()


class DataComparisonTool(QDialog):
    """Dialog for comparing multiple 1D datasets."""
    _ROLE_INPUT_ROW = Qt.ItemDataRole.UserRole + 11

    def __init__(
        self,
        opened_files: tuple[pathlib.Path, ...],
        parent: Any = None,
        dataset_full_keys_1d: list[str] | None = None,
    ) -> None:
        """
        Initialize the Data Comparison Tool.

        Args:
            opened_files: Tuple of currently opened HDF5 file paths
            parent: Parent widget
        """
        super().__init__(parent)

        # Set window flags to ensure proper layering behavior
        # This prevents the dialog from staying on top of all applications
        self.setWindowFlags(Qt.WindowType.Window)

        # An X chosen here belongs to this tool; its export dialog inherits it
        # through the parent chain, and nothing outside does. See x_scope_of.
        self.x_scope = "comparison"

        self.opened_files = opened_files
        self.dataset_full_keys_1d = dataset_full_keys_1d or []
        self._opened_by_name = {p.name: p for p in self.opened_files}
        self._opened_by_full = {str(p): p for p in self.opened_files}
        self._shared_by_file_name: dict[str, set[str]] = {}
        self._shared_by_file_full: dict[str, set[str]] = {}
        for key in self.dataset_full_keys_1d:
            if "::" not in key:
                continue
            file_part, ds_path = key.split("::", 1)
            file_name = pathlib.Path(file_part).name
            self._shared_by_file_name.setdefault(file_name, set()).add(ds_path)
            self._shared_by_file_full.setdefault(file_part, set()).add(ds_path)
        self.datasets: list[CurveEntry] = []
        self._x_selection_target_row: int | None = None  # None = apply to all rows
        self.selected_point = None  # (x, y) of selected point
        self.selected_marker = None  # Circle marker for selected point
        self.line_width = 3  # Default line width in pixels
        self._defer_plot_update = False

        # Color palette for different datasets
        self.colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring green
            (255, 0, 128),    # Deep pink
        ]

        self._init_ui()
        self._populate_available_datasets()
        register_x_target(self)

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Data Comparison")
        # Set initial size (not minimum) to allow later compression
        self.resize(1200, 650)

        # Main layout
        main_layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            "<b>Data Comparison Tool</b> - Compare multiple 1D datasets\n"
            "Drag and drop 1D datasets from the tree view to the list below"
        )
        info_label.setStyleSheet("background-color: #e3f2fd; padding: 5px; border-radius: 5px;")
        main_layout.addWidget(info_label)

        # Splitter for left (list) and right (plot) panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Set size policy to allow splitter to expand with window
        from PyQt6.QtWidgets import QSizePolicy
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Left panel - Dataset list
        left_panel = QWidget()
        # Set size policy for left panel to allow free resizing
        from PyQt6.QtWidgets import QSizePolicy
        left_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout()
        # Zero margins inside the splitter panels, as in the calculator; the
        # dialog's own margin is the only inset.
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Header with label and column toggle buttons
        header_layout = QHBoxLayout()

        # Column toggle buttons
        self.btn_toggle_energy = QPushButton("E(eV)")
        self.btn_toggle_energy.setCheckable(True)
        self.btn_toggle_energy.setChecked(False)  # Hide by default
        self.btn_toggle_energy.setMaximumWidth(60)
        self.btn_toggle_energy.setToolTip("Toggle Energy column")
        self.btn_toggle_energy.clicked.connect(lambda: self._toggle_column(COL_ENERGY, self.btn_toggle_energy))
        header_layout.addWidget(self.btn_toggle_energy)

        self.btn_toggle_fy = QPushButton("f(Y)")
        self.btn_toggle_fy.setCheckable(True)
        self.btn_toggle_fy.setChecked(False)  # Hide by default
        self.btn_toggle_fy.setMaximumWidth(60)
        self.btn_toggle_fy.setToolTip("Toggle f(Y) column — per-curve Y formula\n\n" + EXPR_HELP)
        self.btn_toggle_fy.clicked.connect(lambda: self._toggle_column(COL_FY, self.btn_toggle_fy))
        header_layout.addWidget(self.btn_toggle_fy)

        self.btn_toggle_xaxis = QPushButton("X Axis")
        self.btn_toggle_xaxis.setCheckable(True)
        self.btn_toggle_xaxis.setChecked(False)  # Hide by default
        self.btn_toggle_xaxis.setMaximumWidth(65)
        self.btn_toggle_xaxis.setToolTip("Toggle X Axis column (per-row custom X)")
        self.btn_toggle_xaxis.clicked.connect(lambda: self._toggle_column(COL_XAXIS, self.btn_toggle_xaxis))
        header_layout.addWidget(self.btn_toggle_xaxis)

        self.btn_toggle_fx = QPushButton("f(X)")
        self.btn_toggle_fx.setCheckable(True)
        self.btn_toggle_fx.setChecked(False)  # Hide by default
        self.btn_toggle_fx.setMaximumWidth(60)
        self.btn_toggle_fx.setToolTip("Toggle f(X) column — per-curve X formula\n\n" + EXPR_HELP)
        self.btn_toggle_fx.clicked.connect(lambda: self._toggle_column(COL_FX, self.btn_toggle_fx))
        header_layout.addWidget(self.btn_toggle_fx)

        header_layout.addStretch()

        left_layout.addLayout(header_layout)

        self.dataset_table = DatasetTableWidget()
        self.dataset_table.comparison_tool = self
        # Connect cell change signal to update plot when offset is edited
        self.dataset_table.cellChanged.connect(self._on_cell_changed)
        left_layout.addWidget(self.dataset_table)

        # Initialize column visibility based on button states
        self.dataset_table.setColumnHidden(COL_ENERGY, not self.btn_toggle_energy.isChecked())
        self.dataset_table.setColumnHidden(COL_FY, not self.btn_toggle_fy.isChecked())
        self.dataset_table.setColumnHidden(COL_XAXIS, not self.btn_toggle_xaxis.isChecked())
        self.dataset_table.setColumnHidden(COL_FX, not self.btn_toggle_fx.isChecked())

        # Connect double-click (X Axis column → set X for row) and right-click context menu
        self.dataset_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.dataset_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dataset_table.customContextMenuRequested.connect(self._on_table_context_menu)

        # Buttons for managing list (row 1)
        button_row1_layout = QHBoxLayout()

        self.btn_add_row = QPushButton("Add Row")
        self.btn_add_row.setAutoDefault(False)
        self.btn_add_row.setToolTip("Add an editable row and type/paste: file::/dataset/path")
        self.btn_add_row.clicked.connect(self._add_input_row)
        button_row1_layout.addWidget(self.btn_add_row)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_remove.clicked.connect(self._remove_selected)
        button_row1_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_clear.clicked.connect(self._clear_all)
        button_row1_layout.addWidget(self.btn_clear)

        left_layout.addLayout(button_row1_layout)

        # Full export (settings dialog). Quick save/copy of the figure are icons
        # in the plot toolbar on the right, as in every other viewer.
        button_row2_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_export.setToolTip("Full export: choose the column layout, dialect and header")
        self.btn_export.clicked.connect(self._export_full)
        button_row2_layout.addWidget(self.btn_export)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.setAutoDefault(False)
        self.btn_plot.setToolTip("Plot: the exported table and a matplotlib figure of it")
        self.btn_plot.clicked.connect(self.open_plot)
        button_row2_layout.addWidget(self.btn_plot)

        left_layout.addLayout(button_row2_layout)

        left_panel.setLayout(left_layout)
        splitter.addWidget(left_panel)

        # Right panel - Plot view
        right_panel = QWidget()
        # Set size policy for right panel to allow free resizing
        right_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        plot_label = QLabel("<b>Comparison Plot</b>")
        right_layout.addWidget(plot_label)

        # Axis scale controls
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)

        # Quick actions live here, matching the icon pair in every other viewer.
        self.btn_copy_image = quick_icon_button("copy.ico", "Quick copy: put the plot on the clipboard")
        self.btn_copy_image.clicked.connect(self._copy_plot_image)
        control_layout.addWidget(self.btn_copy_image)

        self.btn_save_data = quick_icon_button(
            "save.ico", "Quick export: save the plotted curves as displayed"
        )
        self.btn_save_data.clicked.connect(self.quick_export)
        control_layout.addWidget(self.btn_save_data)

        self.btn_open_plot = quick_icon_button(
            "plot.ico", "Plot: draw the compared curves in a Plot window"
        )
        self.btn_open_plot.clicked.connect(self.open_plot)
        control_layout.addWidget(self.btn_open_plot)

        control_layout.addSpacing(10)

        scale_label = QLabel("Axis Scale:")
        control_layout.addWidget(scale_label)

        self.chk_log_x = QCheckBox("Log X")
        self.chk_log_x.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_x)

        self.chk_log_y = QCheckBox("Log Y")
        self.chk_log_y.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_y)

        control_layout.addSpacing(20)

        # Custom X data control. Named and behaving like the data viewer's and
        # the tree's "Set X"; the per-row nuance lives in the tooltip.
        self.btn_select_x = QPushButton("Set X")
        self.btn_select_x.setAutoDefault(False)
        self.btn_select_x.setMinimumWidth(0)
        self.btn_select_x.setMaximumWidth(65)
        self.btn_select_x.setToolTip("Set the X axis for every row of matching length")
        self.btn_select_x.clicked.connect(self._select_custom_x)
        control_layout.addWidget(self.btn_select_x)

        control_layout.addSpacing(20)

        # Line width control
        from PyQt6.QtWidgets import QSpinBox
        linewidth_label = QLabel("Line Width:")
        control_layout.addWidget(linewidth_label)

        self.spinbox_linewidth = QSpinBox()
        self.spinbox_linewidth.setMinimum(1)
        self.spinbox_linewidth.setMaximum(10)
        self.spinbox_linewidth.setValue(3)  # Default line width
        self.spinbox_linewidth.setSuffix(" px")
        self.spinbox_linewidth.valueChanged.connect(self._on_linewidth_changed)
        control_layout.addWidget(self.spinbox_linewidth)

        control_layout.addSpacing(20)

        # Q conversion for scattering experiments
        self.chk_convert_to_q = QCheckBox("X->q")
        self.chk_convert_to_q.setToolTip("Convert X-axis angle to momentum transfer q using energy from table")
        self.chk_convert_to_q.setEnabled(False)  # Disabled until X data is loaded
        self.chk_convert_to_q.stateChanged.connect(self._on_q_conversion_changed)
        control_layout.addWidget(self.chk_convert_to_q)

        control_layout.addSpacing(20)

        # Coordinates display label
        self.label_coords = QLabel("X: - | Y: -")
        self.label_coords.setStyleSheet("color: gray; font-size: 9pt;")
        # Set max width to prevent excessive expansion, but allow shrinking
        self.label_coords.setMaximumWidth(200)
        self.label_coords.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        control_layout.addWidget(self.label_coords)

        control_layout.addStretch()

        right_layout.addLayout(control_layout)

        # Import pyqtgraph for plotting
        import pyqtgraph as pg

        self.plot_widget = pg.PlotWidget()
        # Set size policy to allow plot to expand and fill available space
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_widget.setLabel("bottom", "Index")
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.addLegend(offset=(-10, 10))  # Position in top-right corner
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Apply dark theme
        self.plot_widget.setBackground('k')  # Black background

        # Set axis colors to white for dark theme
        axis_pen = pg.mkPen(color='w', width=1)
        for axis in ['left', 'bottom', 'right', 'top']:
            self.plot_widget.getAxis(axis).setPen(axis_pen)
            self.plot_widget.getAxis(axis).setTextPen(axis_pen)

        # Right-click reaches the same Export and Plot as the buttons below.
        attach_plot_menu(
            self.plot_widget,
            on_export=self.quick_export,
            on_plot=self.open_plot,
        )

        # Connect mouse click event for data point selection
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        right_layout.addWidget(self.plot_widget)

        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)

        # Set initial splitter sizes to ensure toolbar is fully visible
        # Left: 300px for dataset list, Right: 850px for plot with toolbar
        splitter.setSizes([300, 850])

        main_layout.addWidget(splitter)

        self.setLayout(main_layout)

    def _populate_available_datasets(self) -> None:
        """Populate available datasets from opened files."""
        # This is called on initialization to prepare dataset info
        # Actual datasets are added via drag-drop
        if self.dataset_full_keys_1d:
            logging.info(
                "Data Comparison initialized with shared 1D index: %d datasets from %d files",
                len(self.dataset_full_keys_1d),
                len(self.opened_files),
            )
        else:
            logging.info(f"Data Comparison Tool initialized with {len(self.opened_files)} files")

    def refresh_dataset_keys(
        self,
        full_keys_1d: list[str],
        opened_files: tuple[pathlib.Path, ...] | None = None,
    ) -> None:
        """Refresh shared dataset index used by path resolution and X-data picker."""
        if opened_files is not None:
            self.opened_files = tuple(opened_files)
        self.dataset_full_keys_1d = list(full_keys_1d)
        self._opened_by_name = {p.name: p for p in self.opened_files}
        self._opened_by_full = {str(p): p for p in self.opened_files}
        self._shared_by_file_name.clear()
        self._shared_by_file_full.clear()
        for key in self.dataset_full_keys_1d:
            if "::" not in key:
                continue
            file_part, ds_path = key.split("::", 1)
            file_name = pathlib.Path(file_part).name
            self._shared_by_file_name.setdefault(file_name, set()).add(ds_path)
            self._shared_by_file_full.setdefault(file_part, set()).add(ds_path)

    @staticmethod
    def _parse_dataset_path_with_optional_col(dataset_path: str) -> tuple[str, str, int | None] | None:
        """Parse 'file::/path[/...] [Col N]' into (file_token, ds_path, forced_col)."""
        if "::" not in dataset_path:
            return None
        filename, h5_path = dataset_path.split("::", 1)
        forced_col = None
        col_match = re.search(r"\s*\[Col\s+(\d+)\]\s*$", h5_path)
        if col_match:
            forced_col = int(col_match.group(1))
            h5_path = h5_path[: col_match.start()].strip()
        return filename, h5_path, forced_col

    def _ingest_loaded_dataset(
        self,
        filename_token: str,
        h5_path: str,
        data: np.ndarray,
        forced_col: int | None = None,
    ) -> None:
        """Ingest already-loaded dataset data into table/list (no file I/O)."""
        if data.ndim == 1:
            dataset_name = f"{filename_token}::{h5_path}"
            self._add_dataset_to_table(
                dataset_name, data, energy=0.0, update_plot=False
            )
            logging.info(f"Added 1D dataset: {dataset_name}, shape: {data.shape}")
        elif data.ndim == 2:
            from PyQt6.QtWidgets import QInputDialog

            num_cols = int(data.shape[1])
            if num_cols >= 100:
                QMessageBox.warning(
                    self,
                    "Comparison Limit",
                    f"Dataset has {num_cols} columns (>=100).\n"
                    "Data Comparison supports fewer than 100 columns for 2D datasets."
                )
                return

            if forced_col is not None:
                if forced_col < 0 or forced_col >= num_cols:
                    QMessageBox.warning(
                        self,
                        "Invalid Column",
                        f"Column {forced_col} out of range for shape {data.shape}."
                    )
                    return
                col_data = data[:, forced_col]
                dataset_name = f"{filename_token}::{h5_path} [Col {forced_col}]"
                self._add_dataset_to_table(
                    dataset_name, col_data, energy=0.0, update_plot=False
                )
                logging.info(
                    f"Added forced column {forced_col} from 2D dataset: {dataset_name}, shape: {col_data.shape}"
                )
                if not self._defer_plot_update:
                    self._update_plot()
                return

            items = [f"All columns ({num_cols} curves)"]
            for i in range(num_cols):
                items.append(f"Column {i}")

            item, ok = QInputDialog.getItem(
                self,
                "Select Column",
                f"Dataset: {h5_path}\n"
                f"Shape: {data.shape} ({num_cols} columns)\n\n"
                f"Select which column(s) to add:",
                items,
                0,
                False,
            )
            if not ok:
                return

            if item.startswith("All columns"):
                for col_idx in range(num_cols):
                    col_data = data[:, col_idx]
                    dataset_name = f"{filename_token}::{h5_path} [Col {col_idx}]"
                    self._add_dataset_to_table(
                        dataset_name, col_data, energy=0.0, update_plot=False
                    )
                    logging.info(
                        f"Added column {col_idx} from 2D dataset: {dataset_name}, shape: {col_data.shape}"
                    )
            else:
                column = int(item.split()[-1])
                col_data = data[:, column]
                dataset_name = f"{filename_token}::{h5_path} [Col {column}]"
                self._add_dataset_to_table(
                    dataset_name, col_data, energy=0.0, update_plot=False
                )
                logging.info(
                    f"Added column {column} from 2D dataset: {dataset_name}, shape: {col_data.shape}"
                )
        else:
            QMessageBox.warning(
                self,
                "Unsupported Dataset",
                f"Cannot compare {data.ndim}D datasets.\n\n"
                f"Dataset '{h5_path}' has shape: {data.shape}\n\n"
                "This tool supports 1D and 2D datasets only.\n"
                "For 2D datasets, you can select specific columns."
            )
            return

        if not self._defer_plot_update:
            self._update_plot()

    def add_dataset_from_loaded_path(self, dataset_path: str, data: np.ndarray) -> None:
        """Add already-loaded dataset payload to comparison list (avoids re-reading file)."""
        parsed = self._parse_dataset_path_with_optional_col(dataset_path)
        if parsed is None:
            QMessageBox.warning(
                self,
                "Invalid Path",
                f"Cannot parse dataset path:\n{dataset_path}\n\n"
                "Expected format: filename.ext::path/to/dataset"
            )
            return
        filename, h5_path, forced_col = parsed
        self._ingest_loaded_dataset(filename, h5_path, np.asarray(data), forced_col=forced_col)

    def add_dataset_from_path(self, dataset_path: str) -> None:
        """
        Add a dataset to the comparison list from a path string.

        Args:
            dataset_path: Path in format "filename.ext::path/to/dataset"
        """
        try:
            # Parse the path
            parsed = self._parse_dataset_path_with_optional_col(dataset_path)
            if parsed is None:
                QMessageBox.warning(
                    self,
                    "Invalid Path",
                    f"Cannot parse dataset path:\n{dataset_path}\n\n"
                    "Expected format: filename.ext::path/to/dataset"
                )
                return

            filename, h5_path, forced_col = parsed

            # Resolve file path from full path token first, then from short filename.
            file_path = self._opened_by_full.get(filename)
            if file_path is None:
                file_path = self._opened_by_name.get(pathlib.Path(filename).name)

            if file_path is None:
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"File not found in opened files:\n{filename}\n\n"
                    "Please ensure the file is open in the main window."
                )
                return

            # Resolve compact dataset names via shared index first (fast path).
            shared_paths = self._shared_by_file_full.get(str(file_path))
            if shared_paths is None:
                shared_paths = self._shared_by_file_name.get(file_path.name)
            if shared_paths is None:
                shared_paths = set()

            if h5_path not in shared_paths:
                target_leaf = h5_path.strip().strip("/")
                candidates = [
                    p for p in shared_paths
                    if p.strip("/") == target_leaf or p.strip("/").split("/")[-1] == target_leaf
                ]
                if len(candidates) == 1:
                    h5_path = candidates[0]

            from src.lib_h5.file_validator import is_hdf5_file
            if is_hdf5_file(file_path):
                # Load the dataset
                with h5py.File(file_path, "r") as h5file:
                    if h5_path not in h5file:
                        QMessageBox.warning(
                            self,
                            "Dataset Not Found",
                            f"Dataset not found in file:\n{h5_path}"
                        )
                        return

                    dataset = h5file[h5_path]
                    data = np.asarray(dataset[:])
            else:
                from src.gui.main_window import load_regular_data_file
                data = load_regular_data_file(file_path)
            self._ingest_loaded_dataset(filename, h5_path, data, forced_col=forced_col)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Dataset",
                f"Failed to load dataset:\n{dataset_path}\n\nError: {e}"
            )
            logging.error(f"Error loading dataset {dataset_path}: {e}")

    def add_dataset_from_array(
        self,
        name: str,
        data: np.ndarray,
        energy: float = 0.0,
        y_expr: str = "",
        x_expr: str = "",
    ) -> None:
        """Add an in-memory 1D/2D result array into comparison list."""
        try:
            arr = np.asarray(data)
            if arr.ndim == 1:
                self._add_dataset_to_table(
                    name,
                    arr,
                    energy=energy,
                    y_expr=y_expr,
                    x_expr=x_expr,
                    update_plot=False,
                )
                if not self._defer_plot_update:
                    self._update_plot()
                return

            if arr.ndim == 2:
                if int(arr.shape[1]) >= 100:
                    QMessageBox.warning(
                        self,
                        "Comparison Limit",
                        f"Result has {arr.shape[1]} columns (>=100).\n"
                        "Only up to 99 columns are allowed.",
                    )
                    return
                for col_idx in range(arr.shape[1]):
                    self._add_dataset_to_table(
                        f"{name} [Col {col_idx}]",
                        arr[:, col_idx],
                        energy=energy,
                        y_expr=y_expr,
                        x_expr=x_expr,
                        update_plot=False,
                    )
                if not self._defer_plot_update:
                    self._update_plot()
                return

            QMessageBox.warning(
                self,
                "Unsupported Result",
                f"Cannot transfer {arr.ndim}D result to comparison.\n"
                "Only 1D/2D results are supported.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Transfer Failed", f"Failed to add result:\n{e}")

    def _add_dataset_to_table(
        self,
        name: str,
        data: np.ndarray,
        energy: float = 0.0,
        y_expr: str = "",
        x_expr: str = "",
        update_plot: bool = True,
        x_data: np.ndarray | None = None,
        x_path: str | None = None,
    ) -> None:
        """Add a dataset to the table and internal list."""
        self.datasets.append(
            CurveEntry(
                name=name,
                data=data,
                energy=energy,
                y_expr=y_expr,
                x_expr=x_expr,
                x_data=x_data,
                x_path=x_path,
            )
        )

        row = self.dataset_table.rowCount()
        prev_block = self.dataset_table.blockSignals(True)
        self.dataset_table.insertRow(row)

        # Dataset name
        display_name = self._compact_dataset_name(name)
        name_item = QTableWidgetItem(display_name)
        name_item.setToolTip(name)
        name_item.setData(Qt.ItemDataRole.UserRole, name)
        self.dataset_table.setItem(row, COL_NAME, name_item)

        # Number of points (read-only)
        points_item = QTableWidgetItem(str(len(data)))
        points_item.setFlags(points_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.dataset_table.setItem(row, COL_POINTS, points_item)

        # Energy in eV (editable)
        self.dataset_table.setItem(row, COL_ENERGY, QTableWidgetItem(str(energy)))

        # Curve Transform expressions (editable); empty means no transform.
        self.dataset_table.setItem(row, COL_FY, self._make_expr_item(y_expr, FY_PLACEHOLDER))
        self.dataset_table.setItem(row, COL_FX, self._make_expr_item(x_expr, FX_PLACEHOLDER))

        # X Axis (read-only, set via double-click or right-click)
        x_label = self._short_key_label(x_path) if x_path else "—"
        x_item = QTableWidgetItem(x_label)
        x_item.setFlags(x_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if x_path:
            x_item.setToolTip(x_path)
        self.dataset_table.setItem(row, COL_XAXIS, x_item)

        self.dataset_table.blockSignals(prev_block)

        # Update plot
        if update_plot and not self._defer_plot_update:
            self._update_plot()

    def _compact_dataset_name(self, full_name: str) -> str:
        """Return a compact dataset label for table display."""
        if "::" not in full_name:
            return pathlib.Path(full_name).name

        file_part, dataset_part = full_name.split("::", 1)
        file_name = pathlib.Path(file_part).name

        # Keep optional column suffix, e.g. [Col 3], to avoid ambiguity.
        match = re.search(r"\s*(\[Col\s+\d+\])\s*$", dataset_part)
        col_suffix = match.group(1) if match else ""
        dataset_core = dataset_part[: match.start()].strip() if match else dataset_part.strip()

        # Show only the leaf dataset name, e.g. /a/b/data_21 -> data_21
        dataset_leaf = dataset_core.rstrip("/").split("/")[-1] if dataset_core else dataset_core
        if not dataset_leaf:
            dataset_leaf = dataset_core

        compact = f"{file_name}::{dataset_leaf}" if dataset_leaf else file_name
        if match:
            compact = f"{compact} {col_suffix}"
        return compact

    # ------------------------------------------------------------------
    # Curve Transform (per-row f(X) / f(Y) expressions)
    # ------------------------------------------------------------------

    def _make_expr_item(self, expression: str, placeholder: str) -> QTableWidgetItem:
        """Build an editable expression cell showing a hint when it is empty."""
        item = QTableWidgetItem(expression)
        self._style_expr_item(item, placeholder, error=None)
        return item

    @staticmethod
    def _style_expr_item(item: QTableWidgetItem, placeholder: str, error: str | None) -> None:
        """Color an expression cell: red on error, grey hint when empty."""
        from PyQt6.QtGui import QBrush, QColor

        if error:
            item.setBackground(QBrush(QColor("#ffe0e0")))
            item.setForeground(QBrush(QColor("#b00000")))
            item.setToolTip(f"{error}\n\n{EXPR_HELP}")
            return

        item.setBackground(QBrush(QColor("#ffffff")))
        if item.text().strip():
            item.setForeground(QBrush(QColor("#111111")))
            item.setToolTip(EXPR_HELP)
        else:
            item.setForeground(QBrush(QColor("#111111")))
            item.setToolTip(f"Empty = no transform (result is '{placeholder}').\n\n{EXPR_HELP}")

    def _mark_expr_error(self, row: int, column: int, error: str | None) -> None:
        """Flag or clear the error state of an expression cell without re-plotting."""
        item = self.dataset_table.item(row, column)
        if item is None:
            return
        placeholder = FY_PLACEHOLDER if column == COL_FY else FX_PLACEHOLDER
        prev_block = self.dataset_table.blockSignals(True)
        self._style_expr_item(item, placeholder, error)
        self.dataset_table.blockSignals(prev_block)

    def _base_x_values(self, entry: CurveEntry) -> np.ndarray:
        """X samples before f(X): the row's X dataset (q-converted if enabled), else the index."""
        if entry.x_data is not None and len(entry.x_data) == len(entry.data):
            if self.chk_convert_to_q.isChecked():
                return np.array([self._convert_angle_to_q(a, entry.energy) for a in entry.x_data], dtype=float)
            return np.asarray(entry.x_data, dtype=float)
        return np.arange(len(entry.data), dtype=float)

    def _transform_entry(self, entry: CurveEntry) -> tuple[np.ndarray, np.ndarray, str | None, str | None]:
        """Apply the row's Curve Transform.

        f(X) runs first so that f(Y) can refer to the transformed ``x``
        (``gradient(y, x)`` then means what the user sees on the axis).

        :return: ``(x_values, y_values, x_error, y_error)``. On error the offending
            axis falls back to its untransformed values so the curve still plots.
        """
        x_values = self._base_x_values(entry)
        y_values = np.asarray(entry.data, dtype=float)
        x_error: str | None = None
        y_error: str | None = None

        if entry.x_expr.strip():
            try:
                x_values = evaluate_series(entry.x_expr, y_values, x_values, entry.energy)
            except ExpressionError as exc:
                x_error = str(exc)
                logging.warning("f(X) failed for '%s': %s", entry.name, exc)

        if entry.y_expr.strip():
            try:
                y_values = evaluate_series(entry.y_expr, y_values, x_values, entry.energy)
            except ExpressionError as exc:
                y_error = str(exc)
                logging.warning("f(Y) failed for '%s': %s", entry.name, exc)

        return x_values, y_values, x_error, y_error

    def _transform_row(self, row: int, entry: CurveEntry) -> tuple[np.ndarray, np.ndarray]:
        """Transform one row and reflect any expression error in its cells."""
        x_values, y_values, x_error, y_error = self._transform_entry(entry)
        self._mark_expr_error(row, COL_FX, x_error)
        self._mark_expr_error(row, COL_FY, y_error)
        return x_values, y_values

    def _toggle_column(self, column_index: int, button: QPushButton) -> None:
        """
        Toggle visibility of a table column.

        Args:
            column_index: Index of the column to toggle
            button: The toggle button that triggered this action
        """
        is_visible = button.isChecked()
        self.dataset_table.setColumnHidden(column_index, not is_visible)
        logging.info(f"Column {column_index} visibility: {is_visible}")

        # If toggling E(eV) column, also update X->q checkbox availability
        if column_index == COL_ENERGY:
            self._update_q_conversion_availability()

    def _update_q_conversion_availability(self) -> None:
        """Update X->q checkbox: enabled when at least one row has x_data AND E(eV) column is visible."""
        has_x_data = any(entry.x_data is not None for entry in self.datasets)
        energy_column_visible = self.btn_toggle_energy.isChecked()
        should_enable = has_x_data and energy_column_visible
        self.chk_convert_to_q.setEnabled(should_enable)
        if not should_enable and self.chk_convert_to_q.isChecked():
            self.chk_convert_to_q.setChecked(False)
        logging.info(f"X->q availability: has_x={has_x_data}, E(eV)={energy_column_visible}, enabled={should_enable}")

    def _on_cell_changed(self, row: int, column: int) -> None:
        """
        Handle cell changes in the table (energy and Curve Transform expressions).

        Args:
            row: Row index
            column: Column index
        """
        if column == COL_NAME:
            edited_item = self.dataset_table.item(row, 0)
            if edited_item is None:
                return
            is_input_row = bool(edited_item.data(self._ROLE_INPUT_ROW))
            if not is_input_row:
                return
            raw_text = edited_item.text().strip()
            if not raw_text:
                return

            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if not lines:
                return

            # Remove input row first; then append parsed datasets as normal rows.
            self.dataset_table.blockSignals(True)
            self.dataset_table.removeRow(row)
            self.dataset_table.blockSignals(False)
            self._defer_plot_update = True
            try:
                for line in lines:
                    self.add_dataset_from_path(line)
            finally:
                self._defer_plot_update = False
            self._update_plot()
            return

        if column not in (COL_ENERGY, COL_FY, COL_FX):
            return

        if row < 0 or row >= len(self.datasets):
            return

        changed_item = self.dataset_table.item(row, column)
        if changed_item is None:
            return

        entry = self.datasets[row]

        if column in EXPR_COLUMNS:
            # Expressions are free text: store as typed and let _update_plot
            # report a bad formula by coloring the cell. Popping a dialog here
            # would fire on every partially-typed edit.
            expression = changed_item.text().strip()
            if column == COL_FY:
                entry.y_expr = expression
                logging.info("f(Y) for '%s': %r", entry.name, expression)
            else:
                entry.x_expr = expression
                logging.info("f(X) for '%s': %r", entry.name, expression)
            self._update_plot()
            return

        # Energy: numeric, revert on garbage.
        try:
            new_energy = float(changed_item.text())
        except ValueError:
            prev_block = self.dataset_table.blockSignals(True)
            changed_item.setText(str(entry.energy))
            self.dataset_table.blockSignals(prev_block)
            QMessageBox.warning(self, "Invalid Value", "Invalid value. Please enter a valid number.")
            return

        logging.info(f"Updated Energy for '{entry.name}': {entry.energy} -> {new_energy} eV")
        entry.energy = new_energy
        self._update_plot()

    def _add_input_row(self) -> None:
        """Insert an editable input row for manual/paste dataset path import."""
        row = self.dataset_table.rowCount()
        self.dataset_table.blockSignals(True)
        self.dataset_table.insertRow(row)

        path_item = QTableWidgetItem("")
        path_item.setToolTip("Type or paste: file::/dataset/path , then press Enter")
        path_item.setData(self._ROLE_INPUT_ROW, True)
        self.dataset_table.setItem(row, COL_NAME, path_item)

        for col in range(1, COLUMN_COUNT):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.dataset_table.setItem(row, col, item)

        self.dataset_table.blockSignals(False)
        self.dataset_table.setCurrentCell(row, 0)
        self.dataset_table.editItem(path_item)

    def _remove_selected(self) -> None:
        """Remove selected dataset rows from the table."""
        selection_model = self.dataset_table.selectionModel()
        if selection_model is None:
            return

        # Some table selection modes return only selectedIndexes() (cell selection),
        # so fall back to index aggregation if selectedRows() is empty.
        selected_rows_set = {idx.row() for idx in selection_model.selectedRows()}
        if not selected_rows_set:
            selected_rows_set = {idx.row() for idx in selection_model.selectedIndexes()}
        if not selected_rows_set:
            current_row = self.dataset_table.currentRow()
            if current_row >= 0:
                selected_rows_set = {current_row}

        selected_rows = sorted(selected_rows_set, reverse=True)
        if not selected_rows:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select one or more datasets to remove."
            )
            return

        # Block signals to prevent cellChanged from firing
        self.dataset_table.blockSignals(True)

        # Remove from table and backing list from bottom to top
        for row in selected_rows:
            self.dataset_table.removeRow(row)
            if 0 <= row < len(self.datasets):
                removed = self.datasets.pop(row)
                logging.info(f"Removed dataset: {removed.name}")

        # Unblock signals
        self.dataset_table.blockSignals(False)

        # Update plot
        self._update_plot()

    def _clear_all(self) -> None:
        """Clear all datasets from the table."""
        if not self.datasets:
            return

        reply = QMessageBox.question(
            self,
            "Clear All",
            "Are you sure you want to remove all datasets?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Block signals
            self.dataset_table.blockSignals(True)

            # Clear table
            self.dataset_table.setRowCount(0)

            # Unblock signals
            self.dataset_table.blockSignals(False)

            # Clear datasets
            self.datasets.clear()
            self._update_plot()
            logging.info("Cleared all datasets")

    def _clear_all_silent(self) -> None:
        """Clear all datasets without confirmation (used on window close)."""
        self.dataset_table.blockSignals(True)
        self.dataset_table.setRowCount(0)
        self.dataset_table.blockSignals(False)
        self.datasets.clear()
        self.chk_convert_to_q.setChecked(False)
        self.chk_convert_to_q.setEnabled(False)
        self._update_plot()
        logging.info("Cleared all datasets on close")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Clear comparison list when closing the dialog."""
        try:
            self._clear_all_silent()
        finally:
            super().closeEvent(event)

    def _update_plot(self) -> None:
        """Update the plot with current datasets, applying each row's Curve Transform."""
        # Clear existing plot
        self.plot_widget.clear()

        # Reset selected marker and coordinates
        self.selected_marker = None
        self.selected_point = None
        self.label_coords.setText("X: - | Y: -")
        self.label_coords.setStyleSheet("color: gray; font-size: 9pt;")

        if not self.datasets:
            self.plot_widget.addLegend(offset=(-10, 10))
            return

        # Set X-axis label: q if conversion on, else first row with x_path, else Index
        if self.chk_convert_to_q.isChecked():
            set_axis_label(self.plot_widget, "bottom", AXIS_Q)
        else:
            first_x_path = next((e.x_path for e in self.datasets if e.x_path is not None), None)
            if first_x_path:
                set_axis_label(self.plot_widget, "bottom",
                               self._short_key_label(first_x_path) or "Custom X")
            else:
                set_axis_label(self.plot_widget, "bottom", "Index")

        import pyqtgraph as pg
        # The legend names the curves exactly as the export headers do.
        labels = self._unique_series_labels()
        for i, entry in enumerate(self.datasets):
            color = self.colors[i % len(self.colors)]
            pen = pg.mkPen(color=color, width=self.line_width)

            x_values, y_values = self._transform_row(i, entry)

            self.plot_widget.plot(
                x_values, y_values,
                pen=pen,
                name=self._format_name_with_transforms(entry, labels[i] if i < len(labels) else None),
            )

        logging.info(f"Updated plot with {len(self.datasets)} datasets")

    def _format_name_with_transforms(self, entry: CurveEntry, label: str | None = None) -> str:
        """Format dataset name with its energy and Curve Transform expressions."""
        name = label or self._legend_short_label(entry.name)
        transforms = []

        # Show energy if q conversion is enabled
        if self.chk_convert_to_q.isChecked():
            transforms.append(f"{entry.energy:.0f}eV")

        if entry.x_expr.strip():
            transforms.append(f"X={entry.x_expr.strip()}")

        if entry.y_expr.strip():
            transforms.append(f"Y={entry.y_expr.strip()}")

        if transforms:
            return f"{name} ({', '.join(transforms)})"
        else:
            return name

    @staticmethod
    def _short_key_label(full_key: str | None) -> str:
        """Format '<file>::<dataset>' to short 'file::dataset_leaf'."""
        if not full_key:
            return ""
        try:
            if "::" in full_key:
                file_part, ds_part = full_key.rsplit("::", 1)
                file_name = pathlib.Path(file_part).name
                ds_leaf = ds_part.strip().rstrip("/").split("/")[-1]
                if ds_leaf:
                    return f"{file_name}::{ds_leaf}"
                return file_name
            return pathlib.Path(full_key).name
        except Exception:
            return str(full_key)

    def _legend_short_label(self, full_name: str) -> str:
        """Name a curve the way the export headers do — see :mod:`export_naming`."""
        return short_series_label(full_name)

    def _unique_series_labels(self) -> list[str]:
        """Short labels for every row, with collisions numbered apart.

        Shortening can make two rows agree — the same dataset added twice, or
        two files whose stems match. A repeated column header would make the
        exported table ambiguous, so the later ones get a counter.
        """
        labels: list[str] = []
        seen: dict[str, int] = {}
        for entry in self.datasets:
            label = self._legend_short_label(entry.name)
            seen[label] = seen.get(label, 0) + 1
            labels.append(label if seen[label] == 1 else f"{label}_{seen[label]}")
        return labels

    def _update_axis_scale(self) -> None:
        """Update axis scale (linear/log) based on checkboxes."""
        log_x = self.chk_log_x.isChecked()
        log_y = self.chk_log_y.isChecked()

        try:
            self.plot_widget.setLogMode(x=log_x, y=log_y)

            # Refresh marker if a point is selected
            if self.selected_point is not None and self.selected_marker is not None:
                closest_x, closest_y = self.selected_point

                # Remove old marker
                self.plot_widget.removeItem(self.selected_marker)

                # Recreate marker with correct coordinates for new scale
                marker_x = np.log10(closest_x) if (log_x and closest_x > 0) else closest_x
                marker_y = np.log10(closest_y) if (log_y and closest_y > 0) else closest_y

                import pyqtgraph as pg
                marker_width = self.line_width
                marker_size = (8 + marker_width * 2) * 0.7
                self.selected_marker = pg.ScatterPlotItem(
                    [marker_x], [marker_y],
                    size=marker_size,
                    pen=pg.mkPen('orange', width=marker_width),
                    brush=pg.mkBrush('orange'),
                    symbol='o'
                )
                self.plot_widget.addItem(self.selected_marker)

                logging.debug(f"Refreshed marker at ({closest_x:.6g}, {closest_y:.6g}) for new axis scale")

        except Exception as e:
            logging.error(f"Failed to set log mode: {e}")
            QMessageBox.warning(
                self,
                "Log Scale Error",
                f"Failed to set log scale:\n{e}\n\n"
                "Note: Log scale requires positive values in all datasets."
            )
            # Reset checkboxes
            self.chk_log_x.setChecked(False)
            self.chk_log_y.setChecked(False)

    def _convert_angle_to_q(self, angle_deg: float, energy_ev: float) -> float:
        """
        Convert angle (in degrees) to momentum transfer q (in A^-1).

        Formula: q = (4*pi/lambda) * sin(theta)


        Args:
            angle_deg: Angle in degrees
            energy_ev: Photon energy in eV

        Returns:
            Momentum transfer q in A^-1
        """
        try:
            # E(eV) = 12398 / lambda(A)  =>  lambda(A) = 12398 / E(eV)
            wavelength = 12398 / energy_ev

            import math
            angle_rad = math.radians(angle_deg)
            q = (4 * math.pi / wavelength) * math.sin(angle_rad)
            return q
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _on_q_conversion_changed(self, state: int) -> None:
        """Handle X->q conversion checkbox state change."""
        if state and not any(entry.x_data is not None for entry in self.datasets):
            QMessageBox.information(
                self,
                "No X Axis",
                "Please set X axis data for at least one row first.\n\n"
                "Double-click the X Axis column or use 'Set All'.",
            )
            self.chk_convert_to_q.setChecked(False)
            return
        logging.info("X->q conversion: %s", "ON" if state else "OFF")
        self._update_plot()

    def _on_linewidth_changed(self, value: int) -> None:
        """Handle line width change."""
        self.line_width = value
        self._update_plot()
        logging.info(f"Changed line width to: {value}px")

    def _select_custom_x(self) -> None:
        """Open dialog to apply X data to all matching rows."""
        self._x_selection_target_row = None
        self._open_x_selection_dialog()

    def set_x_dataset(self, key: str) -> bool:
        """Take an X dataset handed over by the tree's ``Set X``.

        Same outcome as the ``Set X`` button, minus the picker: every row of
        matching length is re-based, and the rest keep the X they had.
        """
        if not self.datasets:
            QMessageBox.information(self, "No Data", "Add datasets to the comparison list first.")
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

        if not any(len(entry.data) == len(x_data) for entry in self.datasets):
            QMessageBox.warning(
                self,
                "Length Mismatch",
                f"The X dataset is {len(x_data)} long and no row here has that many "
                "points, so nothing was changed.",
            )
            return False

        self._x_selection_target_row = None
        self._on_x_data_selected(x_data, key)
        return True

    def _open_x_selection_dialog(self) -> None:
        from src.gui.plot_widget_1d_enhanced import XDataSelectionDialog
        dialog = XDataSelectionDialog(
            self.opened_files,
            self,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
        )
        dialog.data_selected.connect(self._on_x_data_selected)
        dialog.show()

    def _set_x_axis_for_row(self, row: int) -> None:
        """Open X selection dialog for a single table row."""
        self._x_selection_target_row = row
        self._open_x_selection_dialog()

    def _clear_x_axis_for_row(self, row: int) -> None:
        """Clear per-row X axis assignment."""
        if 0 <= row < len(self.datasets):
            entry = self.datasets[row]
            entry.x_data = None
            entry.x_path = None
            x_item = self.dataset_table.item(row, COL_XAXIS)
            if x_item is not None:
                x_item.setText("—")
                x_item.setToolTip("")
            self._update_q_conversion_availability()
            self._update_plot()

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Double-click on the X Axis column opens X axis selection for that row."""
        if col == COL_XAXIS and 0 <= row < len(self.datasets):
            self._set_x_axis_for_row(row)

    def _apply_expression_to_all(self, column: int, expression: str) -> None:
        """Copy one row's expression into every row."""
        prev_block = self.dataset_table.blockSignals(True)
        for row_idx, entry in enumerate(self.datasets):
            if column == COL_FY:
                entry.y_expr = expression
            else:
                entry.x_expr = expression
            item = self.dataset_table.item(row_idx, column)
            if item is not None:
                item.setText(expression)
        self.dataset_table.blockSignals(prev_block)
        logging.info("Applied %s to all rows: %r", "f(Y)" if column == COL_FY else "f(X)", expression)
        self._update_plot()

    def _clear_expression_for_row(self, row: int, column: int) -> None:
        """Reset a single row's expression to empty (no transform)."""
        entry = self.datasets[row]
        if column == COL_FY:
            entry.y_expr = ""
        else:
            entry.x_expr = ""
        item = self.dataset_table.item(row, column)
        if item is not None:
            prev_block = self.dataset_table.blockSignals(True)
            item.setText("")
            self.dataset_table.blockSignals(prev_block)
        self._update_plot()

    def _on_table_context_menu(self, pos) -> None:
        """Right-click menu: X Axis column sets/clears X, expression columns apply to all rows."""
        index = self.dataset_table.indexAt(pos)
        if not index.isValid():
            return
        row, col = index.row(), index.column()
        if row < 0 or row >= len(self.datasets):
            return

        menu = QMenu(self)
        if col == COL_XAXIS:
            act_set = QAction("Set X Axis...", self)
            act_set.triggered.connect(lambda: self._set_x_axis_for_row(row))
            menu.addAction(act_set)
            act_clear = QAction("Clear X Axis", self)
            act_clear.setEnabled(self.datasets[row].x_data is not None)
            act_clear.triggered.connect(lambda: self._clear_x_axis_for_row(row))
            menu.addAction(act_clear)
        elif col in EXPR_COLUMNS:
            label = "f(Y)" if col == COL_FY else "f(X)"
            entry = self.datasets[row]
            expression = (entry.y_expr if col == COL_FY else entry.x_expr).strip()
            act_all = QAction(f"Apply this {label} to all rows", self)
            act_all.setEnabled(bool(expression))
            act_all.triggered.connect(lambda: self._apply_expression_to_all(col, expression))
            menu.addAction(act_all)
            act_clear_expr = QAction(f"Clear {label}", self)
            act_clear_expr.setEnabled(bool(expression))
            act_clear_expr.triggered.connect(lambda: self._clear_expression_for_row(row, col))
            menu.addAction(act_clear_expr)
            act_clear_all = QAction(f"Clear {label} in all rows", self)
            act_clear_all.triggered.connect(lambda: self._apply_expression_to_all(col, ""))
            menu.addAction(act_clear_all)
        else:
            return

        vp = self.dataset_table.viewport()
        if vp is not None:
            menu.popup(vp.mapToGlobal(pos))

    def _on_x_data_selected(self, x_data: np.ndarray, x_path: str) -> None:
        """Handle X data selection: apply to one row or all matching rows."""
        target_row = self._x_selection_target_row
        self._x_selection_target_row = None

        def _update_row(row_idx: int) -> None:
            entry = self.datasets[row_idx]
            entry.x_data = x_data
            entry.x_path = x_path
            x_item = self.dataset_table.item(row_idx, COL_XAXIS)
            if x_item is not None:
                x_item.setText(self._short_key_label(x_path) or "Custom X")
                x_item.setToolTip(x_path)

        self.dataset_table.blockSignals(True)
        if target_row is not None:
            if 0 <= target_row < len(self.datasets):
                _update_row(target_row)
        else:
            skipped = 0
            for row_idx, entry in enumerate(self.datasets):
                if len(entry.data) == len(x_data):
                    _update_row(row_idx)
                else:
                    skipped += 1
            if skipped:
                logging.info("Set X for %d rows, skipped %d (length mismatch)", len(self.datasets) - skipped, skipped)
        self.dataset_table.blockSignals(False)

        self._update_q_conversion_availability()
        logging.info("Set X data: %s", x_path)
        self._update_plot()

    def _display_space(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a curve to the coordinates the view actually draws in.

        :return: ``(display_x, display_y, valid)``. ``valid`` excludes NaN/inf
            samples, and in log mode also the non-positive ones.
        """
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)

        if self.chk_log_x.isChecked():
            valid_x = x_arr > 0
            display_x = np.zeros_like(x_arr)
            np.log10(x_arr, out=display_x, where=valid_x)
        else:
            valid_x = np.isfinite(x_arr)
            display_x = x_arr

        if self.chk_log_y.isChecked():
            valid_y = y_arr > 0
            display_y = np.zeros_like(y_arr)
            np.log10(y_arr, out=display_y, where=valid_y)
        else:
            valid_y = np.isfinite(y_arr)
            display_y = y_arr

        return display_x, display_y, valid_x & valid_y

    def _closest_point_to_click(self, click_pos) -> tuple[float, float, str] | None:
        """Find the sample nearest to a click, measured in screen pixels.

        Every finite sample is a candidate, including ones drawn outside the
        current view: with the Y axis zoomed in, a curve's peaks leave the
        viewport while their X position stays perfectly clickable, and filtering
        by the view rectangle made those points unselectable.

        The view-to-scene mapping is affine, so the distance is computed
        vectorised via ``viewPixelSize`` instead of mapping each point through Qt.
        """
        vb = self.plot_widget.plotItem.vb
        view_pt = vb.mapSceneToView(click_pos)
        click_x, click_y = float(view_pt.x()), float(view_pt.y())

        pixel_w, pixel_h = (float(v) for v in vb.viewPixelSize())
        if not pixel_w or not pixel_h:
            return None

        best: tuple[float, float, str] | None = None
        best_distance = float("inf")

        for entry in self.datasets:
            # Same transform the plot used; errors already surfaced by _update_plot.
            x_values, y_values, _x_err, _y_err = self._transform_entry(entry)
            display_x, display_y, valid = self._display_space(x_values, y_values)
            if not valid.any():
                continue

            dx = (display_x - click_x) / pixel_w
            dy = (display_y - click_y) / pixel_h
            distance = np.where(valid, dx * dx + dy * dy, np.inf)

            idx = int(np.argmin(distance))
            if distance[idx] < best_distance:
                best_distance = float(distance[idx])
                best = (float(x_values[idx]), float(y_values[idx]), entry.name)

        return best

    def _on_plot_clicked(self, event) -> None:
        """Handle mouse click on plot to select data point."""
        if not self.datasets:
            return

        closest = self._closest_point_to_click(event.scenePos())
        if closest is not None:
            closest_x, closest_y, closest_dataset_name = closest
            # Store selected point
            self.selected_point = (closest_x, closest_y)

            # Remove old marker if exists
            if self.selected_marker is not None:
                self.plot_widget.removeItem(self.selected_marker)

            # Create circle marker at selected point
            # In log mode, use log space coordinates
            marker_x = np.log10(closest_x) if (self.chk_log_x.isChecked() and closest_x > 0) else closest_x
            marker_y = np.log10(closest_y) if (self.chk_log_y.isChecked() and closest_y > 0) else closest_y

            import pyqtgraph as pg
            # Size scales with line width (70% of original size, like PlotWidget1DEnhanced)
            marker_width = self.line_width
            marker_size = (8 + marker_width * 2) * 0.7
            self.selected_marker = pg.ScatterPlotItem(
                [marker_x], [marker_y],
                size=marker_size,
                pen=pg.mkPen('orange', width=marker_width),  # Orange border
                brush=pg.mkBrush('orange'),  # Orange fill
                symbol='o'  # Circle symbol
            )
            self.plot_widget.addItem(self.selected_marker)

            # Update label - only show coordinates
            label_text = f"X: {closest_x:.6g} | Y: {closest_y:.6g}"
            self.label_coords.setText(label_text)
            self.label_coords.setStyleSheet("color: blue; font-size: 9pt; font-weight: bold;")

            logging.debug(f"Selected point: ({closest_x:.6g}, {closest_y:.6g}) from {closest_dataset_name}")

    def _is_shared_xq_compatible(self) -> tuple[bool, str]:
        """Check whether all datasets can share a single X/q column in export."""
        if not self.datasets:
            return True, ""

        first = self.datasets[0]

        for entry in self.datasets:
            name = entry.name
            if entry.x_expr.strip() != first.x_expr.strip():
                return False, f"f(X) differs ({name}: '{entry.x_expr}', first: '{first.x_expr}')."
            if self.chk_convert_to_q.isChecked() and abs(entry.energy - first.energy) > 1e-12:
                return False, f"Energy differs in q mode ({name}: {entry.energy:g} eV, first: {first.energy:g} eV)."
            if entry.x_path != first.x_path:
                return False, f"X Axis dataset differs ({name})."
            if entry.x_data is not None and first.x_data is not None and len(entry.x_data) != len(first.x_data):
                return False, f"X Axis length differs ({name})."
            if (entry.x_data is None) != (first.x_data is None):
                return False, f"Mixed X Axis assignment ({name})."
        return True, ""

    def _build_export_series(self, entry: CurveEntry) -> tuple[np.ndarray, np.ndarray]:
        """Build transformed x/y series exactly matching current plot logic."""
        x_values, y_values, _x_err, _y_err = self._transform_entry(entry)
        return x_values, y_values

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

    def _default_export_base_name(self) -> str:
        """Name the comparison after the range it spans: first and last row.

        Whatever distinguishes them is what appears — two scans, two datasets
        or two columns. See :func:`src.gui.export_naming.pair_label`.
        """
        if not self.datasets:
            return "comparison_compar"

        first = self.datasets[0].name
        last = self.datasets[-1].name if len(self.datasets) > 1 else None
        return f"{pair_label(first, last)}_compar"

    def _capture_plot_pixmap(self):
        """Capture comparison plot area as pixmap."""
        try:
            return self.plot_widget.grab()
        except Exception as e:
            logging.error(f"Failed to capture comparison plot: {e}")
            return None

    def _copy_plot_image(self) -> None:
        """Copy comparison plot screenshot to clipboard."""
        pixmap = self._capture_plot_pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "No Image", "No plot image available to copy.")
            return
        QApplication.clipboard().setPixmap(pixmap)
        logging.info("Copied comparison plot image to clipboard")

    def _load_x_dataset(self, x_key: str) -> np.ndarray | None:
        """Read a 1-D X dataset from a ``file::dataset`` key."""
        try:
            if "::" not in x_key:
                return None
            file_part, ds_path = x_key.split("::", 1)
            resolved = pathlib.Path(file_part.strip())
            if not resolved.exists():
                match = next((p for p in self.opened_files if p.name == resolved.name), None)
                if match is None:
                    return None
                resolved = match
            with h5py.File(resolved, "r") as f:
                if ds_path.strip() not in f:
                    return None
                arr = np.asarray(f[ds_path.strip()][()]).squeeze()
            return arr if arr.ndim == 1 else None
        except Exception as exc:
            logging.warning("Could not read X dataset %s: %s", x_key, exc)
            return None

    def build_export_table(
        self,
        export_x: bool = True,
        x_key: str | None = None,
    ) -> tuple[list[str], list[Any], list[str]]:
        """Build the table an export would write, as displayed.

        :param export_x: when false, only Y columns are written.
        :param x_key: an explicit ``file::dataset`` X to use for every curve; when
            None the X shown in the table is used, shared across curves when they
            agree on one and per-curve otherwise.
        :return: ``(headers, columns, comments)``.
        """
        transformed = []
        for entry in self.datasets:
            x_values, y_values = self._build_export_series(entry)
            transformed.append((entry, x_values, y_values))

        override_x = self._load_x_dataset(x_key) if (export_x and x_key) else None
        has_custom_x = any(entry.x_data is not None for entry in self.datasets)
        compatible, reason = self._is_shared_xq_compatible()
        # An explicit X is by definition shared by every curve.
        use_shared_mode = True if override_x is not None else compatible

        comments = [
            f"# Export mode: {'Aligned table' if use_shared_mode else 'Per-dataset columns'}",
            f"# q conversion: {'ON' if self.chk_convert_to_q.isChecked() else 'OFF'}",
        ]
        if export_x and x_key:
            state = "not readable, using the displayed X" if override_x is None else "applied to all curves"
            comments.append(f"# X dataset: {x_key} ({state})")
        if not use_shared_mode and reason:
            comments.append(f"# auto-switch reason: {reason}")
        comments.extend(
            f"# {entry.name}: E={entry.energy:g} eV, "
            f"fX={entry.x_expr.strip() or 'x'}, fY={entry.y_expr.strip() or 'y'}"
            for entry, _x, _y in transformed
        )

        headers: list[str] = []
        columns: list[Any] = []
        # The same short name the legend shows, so a figure and the table
        # behind it call one curve by one name.
        labels = self._unique_series_labels()
        as_q = self.chk_convert_to_q.isChecked()
        if use_shared_mode:
            # One shared X/q column, then one Y column per curve. The X is named
            # after the dataset it came from, not after any of the Y curves.
            if export_x and (override_x is not None or has_custom_x):
                source = x_key if override_x is not None else next(
                    (e.x_path for e in self.datasets if e.x_path), None
                )
                headers.append(x_column_header(source, as_q))
                columns.append(override_x if override_x is not None else transformed[0][1])
            for (entry, _x_values, y_values), label in zip(transformed, labels):
                headers.append(f"{label}_Y")
                columns.append(y_values)
        else:
            # The curves do not share an axis, so each gets its own X/Y pair.
            for (entry, x_values, y_values), label in zip(transformed, labels):
                if export_x:
                    # Falls back to the Y's name only when this row has no X
                    # dataset of its own — then the column really is "that
                    # curve's axis" and nothing better can be said about it.
                    headers.append(
                        x_column_header(entry.x_path, as_q) if entry.x_path
                        else f"{label}_{'q' if as_q else 'X'}"
                    )
                    columns.append(x_values)
                headers.append(f"{label}_Y")
                columns.append(y_values)

        return headers, columns, comments

    def _ask_export_path(
        self,
        fmt: TableFormat,
        ask_dialect: bool,
    ) -> tuple[pathlib.Path, TableFormat] | None:
        """Ask where to write, remembering the folder and dialect.

        :param ask_dialect: quick export lets the dialect be picked here; the
            full export already chose it in its settings dialog.
        :return: ``(path, dialect)`` or ``None`` when cancelled.
        """
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Data",
            suggested_save_path(self._default_export_base_name(), extension=fmt.suffix),
            save_dialog_filter(fmt) if ask_dialect else dialog_filter(fmt),
        )
        if not file_path:
            return None

        chosen = format_from_filter(selected_filter) if ask_dialect else fmt
        export_path = pathlib.Path(file_path)
        if not export_path.suffix:
            export_path = export_path.with_suffix(chosen.suffix)

        remember_save_directory(export_path)
        QSettings().setValue("export/table_format", chosen.key)
        return export_path, chosen

    def quick_export(self) -> None:
        """Quick export: write the plotted curves straight to a chosen file."""
        if not self.datasets:
            QMessageBox.information(
                self,
                "No Data",
                "No datasets to export. Please add datasets to the comparison list first."
            )
            return

        settings_fmt = get_table_format(str(QSettings().value("export/table_format", DEFAULT_TABLE_FORMAT_KEY)))
        asked = self._ask_export_path(settings_fmt, ask_dialect=True)
        if asked is None:
            return
        export_path, fmt = asked

        headers, columns, comments = self.build_export_table()
        self._write_export(export_path, headers, columns, comments, fmt)

    def plot_series(self) -> list:
        """The compared curves as plot Series.

        Built from the export table so the figure and the file always show the
        same numbers, formulas and X choice included.
        """
        from src.gui.plot_series import series_from_table

        headers, columns, _comments = self.build_export_table()
        return series_from_table(headers, columns)

    def open_plot(self) -> None:
        """Open a Plot window on the compared curves.

        The log switches follow the tool's own, so the figure opens looking like
        the plot it was asked for.
        """
        from src.gui.plot_dialog import open_plot_dialog

        if not self.datasets:
            QMessageBox.information(self, "No Data", "Add datasets to the comparison list first.")
            return

        dialog = open_plot_dialog(self, self.plot_series(), title="Comparison")
        if dialog is None:
            return
        if self.chk_log_x.isChecked() and dialog.chk_log_x.isEnabled():
            dialog.chk_log_x.setChecked(True)
        if self.chk_log_y.isChecked() and dialog.chk_log_y.isEnabled():
            dialog.chk_log_y.setChecked(True)

    def _export_full(self) -> None:
        """Full export: settings dialog first, then write.

        Shown non-modally so a dataset can still be dragged out of the main
        window's tree into the dialog's X field.
        """
        if not self.datasets:
            QMessageBox.information(
                self,
                "No Data",
                "No datasets to export. Please add datasets to the comparison list first."
            )
            return

        dialog = ComparisonExportDialog(self)

        def refresh() -> None:
            headers, columns, _comments = self.build_export_table(dialog.export_x(), dialog.x_key())
            dialog.show_preview(headers, columns)

        dialog.settings_changed.connect(refresh)
        dialog.accepted.connect(lambda d=dialog: self._finish_full_export(d))
        dialog.rejected.connect(dialog.deleteLater)
        refresh()

        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._export_dialog = dialog

    def _finish_full_export(self, dialog: "ComparisonExportDialog") -> None:
        """Write the file once the non-modal settings dialog is accepted."""
        asked = self._ask_export_path(dialog.table_format(), ask_dialect=False)
        if asked is None:
            dialog.deleteLater()
            return
        export_path, fmt = asked

        headers, columns, comments = self.build_export_table(dialog.export_x(), dialog.x_key())
        include_comments = dialog.include_comments()
        dialog.deleteLater()

        self._write_export(
            export_path,
            headers,
            columns,
            comments if include_comments else [],
            fmt,
        )

    def _write_export(
        self,
        export_path: pathlib.Path,
        headers: list[str],
        columns: list[Any],
        comments: list[str],
        fmt: TableFormat,
    ) -> None:
        """Write the built table and report the outcome."""
        file_path = str(export_path)
        try:
            write_table(export_path, headers, columns, fmt, comments=comments)

            logging.info(f"Exported comparison data to: {file_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Comparison data exported successfully to:\n{file_path}"
            )

        except Exception as e:
            logging.error(f"Failed to export comparison data: {e}")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export data:\n{str(e)}"
            )

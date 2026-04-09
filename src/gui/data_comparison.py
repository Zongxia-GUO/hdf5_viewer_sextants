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

import csv
import logging
import pathlib
import re
from typing import Any

import h5py
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class DatasetTableWidget(QTableWidget):
    """Custom TableWidget that accepts drag and drop of datasets with offset support."""

    def __init__(self, parent=None):
        """Initialize the table widget."""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.comparison_tool = None  # Will be set by DataComparisonTool

        # Setup table
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["Dataset", "Points", "E(eV)", "Offset X", "Offset Y", "Scale Y"])

        # Configure columns
        header = self.horizontalHeader()
        if header:
            # Disable automatic stretching of last section to prevent resize jumps
            header.setStretchLastSection(False)

            # Use Interactive mode to allow user to resize columns
            # Avoid ResizeToContents which causes jumps during splitter drag
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

            # Set default widths for all columns
            self.setColumnWidth(0, 200)  # Dataset name
            self.setColumnWidth(1, 70)   # Points
            self.setColumnWidth(2, 70)   # E(eV)
            self.setColumnWidth(3, 80)   # Offset X
            self.setColumnWidth(4, 80)   # Offset Y
            self.setColumnWidth(5, 80)   # Scale Y

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
        self.datasets = []  # List of (name, data, energy, offset_x, offset_y, scale_y) tuples
        self.x_data = None  # Custom X-axis data (shared by all datasets)
        self.x_data_original = None  # Original X data before q conversion
        self.x_dataset_path = None  # Path to X dataset
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

        # Header with label and column toggle buttons
        header_layout = QHBoxLayout()

        list_label = QLabel("<b>Dataset List</b>")
        header_layout.addWidget(list_label)

        header_layout.addSpacing(10)

        # Column toggle buttons
        self.btn_toggle_energy = QPushButton("E(eV)")
        self.btn_toggle_energy.setCheckable(True)
        self.btn_toggle_energy.setChecked(False)  # Hide by default
        self.btn_toggle_energy.setMaximumWidth(60)
        self.btn_toggle_energy.setToolTip("Toggle Energy column")
        self.btn_toggle_energy.clicked.connect(lambda: self._toggle_column(2, self.btn_toggle_energy))
        header_layout.addWidget(self.btn_toggle_energy)

        self.btn_toggle_offset_x = QPushButton("Offset X")
        self.btn_toggle_offset_x.setCheckable(True)
        self.btn_toggle_offset_x.setChecked(False)  # Hide by default
        self.btn_toggle_offset_x.setMaximumWidth(70)
        self.btn_toggle_offset_x.setToolTip("Toggle Offset X column")
        self.btn_toggle_offset_x.clicked.connect(lambda: self._toggle_column(3, self.btn_toggle_offset_x))
        header_layout.addWidget(self.btn_toggle_offset_x)

        self.btn_toggle_offset_y = QPushButton("Offset Y")
        self.btn_toggle_offset_y.setCheckable(True)
        self.btn_toggle_offset_y.setChecked(False)  # Hide by default
        self.btn_toggle_offset_y.setMaximumWidth(70)
        self.btn_toggle_offset_y.setToolTip("Toggle Offset Y column")
        self.btn_toggle_offset_y.clicked.connect(lambda: self._toggle_column(4, self.btn_toggle_offset_y))
        header_layout.addWidget(self.btn_toggle_offset_y)

        self.btn_toggle_scale_y = QPushButton("Scale Y")
        self.btn_toggle_scale_y.setCheckable(True)
        self.btn_toggle_scale_y.setChecked(False)  # Hide by default
        self.btn_toggle_scale_y.setMaximumWidth(70)
        self.btn_toggle_scale_y.setToolTip("Toggle Scale Y column")
        self.btn_toggle_scale_y.clicked.connect(lambda: self._toggle_column(5, self.btn_toggle_scale_y))
        header_layout.addWidget(self.btn_toggle_scale_y)

        header_layout.addStretch()

        left_layout.addLayout(header_layout)

        self.dataset_table = DatasetTableWidget()
        self.dataset_table.comparison_tool = self
        # Connect cell change signal to update plot when offset is edited
        self.dataset_table.cellChanged.connect(self._on_cell_changed)
        left_layout.addWidget(self.dataset_table)

        # Initialize column visibility based on button states
        self.dataset_table.setColumnHidden(2, not self.btn_toggle_energy.isChecked())  # E(eV)
        self.dataset_table.setColumnHidden(3, not self.btn_toggle_offset_x.isChecked())  # Offset X
        self.dataset_table.setColumnHidden(4, not self.btn_toggle_offset_y.isChecked())  # Offset Y
        self.dataset_table.setColumnHidden(5, not self.btn_toggle_scale_y.isChecked())  # Scale Y

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

        # Export/figure actions (row 2)
        button_row2_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export...")
        self.btn_export.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_export.clicked.connect(self._export_to_csv)
        button_row2_layout.addWidget(self.btn_export)

        self.btn_save_image = QPushButton("Save Image...")
        self.btn_save_image.setAutoDefault(False)
        self.btn_save_image.clicked.connect(self._save_plot_image)
        button_row2_layout.addWidget(self.btn_save_image)

        self.btn_copy_image = QPushButton("Copy Image")
        self.btn_copy_image.setAutoDefault(False)
        self.btn_copy_image.clicked.connect(self._copy_plot_image)
        button_row2_layout.addWidget(self.btn_copy_image)

        left_layout.addLayout(button_row2_layout)

        left_panel.setLayout(left_layout)
        splitter.addWidget(left_panel)

        # Right panel - Plot view
        right_panel = QWidget()
        # Set size policy for right panel to allow free resizing
        right_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout()

        plot_label = QLabel("<b>Comparison Plot</b>")
        right_layout.addWidget(plot_label)

        # Axis scale controls
        control_layout = QHBoxLayout()

        scale_label = QLabel("Axis Scale:")
        control_layout.addWidget(scale_label)

        self.chk_log_x = QCheckBox("Log X")
        self.chk_log_x.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_x)

        self.chk_log_y = QCheckBox("Log Y")
        self.chk_log_y.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_y)

        control_layout.addSpacing(20)

        # Custom X data control
        x_label = QLabel("X Axis:")
        control_layout.addWidget(x_label)

        self.btn_select_x = QPushButton("Select")
        self.btn_select_x.setAutoDefault(False)  # Prevent Enter key from triggering this button
        # Set maximum width to allow toolbar to compress when window is resized
        self.btn_select_x.setMinimumWidth(0)  # Allow shrinking
        self.btn_select_x.setMaximumWidth(60)
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

        # Disable right-click menu for consistent UI
        self.plot_widget.plotItem.vb.setMenuEnabled(False)

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
                dataset_name, data, energy=0.0, offset_x=0.0, offset_y=0.0, scale_y=1.0, update_plot=False
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
                    dataset_name, col_data, energy=0.0, offset_x=0.0, offset_y=0.0, scale_y=1.0, update_plot=False
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
                        dataset_name, col_data, energy=0.0, offset_x=0.0, offset_y=0.0, scale_y=1.0, update_plot=False
                    )
                    logging.info(
                        f"Added column {col_idx} from 2D dataset: {dataset_name}, shape: {col_data.shape}"
                    )
            else:
                column = int(item.split()[-1])
                col_data = data[:, column]
                dataset_name = f"{filename_token}::{h5_path} [Col {column}]"
                self._add_dataset_to_table(
                    dataset_name, col_data, energy=0.0, offset_x=0.0, offset_y=0.0, scale_y=1.0, update_plot=False
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
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale_y: float = 1.0,
    ) -> None:
        """Add an in-memory 1D/2D result array into comparison list."""
        try:
            arr = np.asarray(data)
            if arr.ndim == 1:
                self._add_dataset_to_table(
                    name,
                    arr,
                    energy=energy,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    scale_y=scale_y,
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
                        offset_x=offset_x,
                        offset_y=offset_y,
                        scale_y=scale_y,
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
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale_y: float = 1.0,
        update_plot: bool = True,
    ) -> None:
        """
        Add a dataset to the table and internal list.

        Args:
            name: Dataset name
            data: Dataset array
            energy: Photon energy in eV (default 0.0)
            offset_x: X-axis offset (default 0.0)
            offset_y: Y-axis offset (default 0.0)
            scale_y: Y-axis scale multiplier (default 1.0)
        """
        # Add to internal datasets list
        self.datasets.append((name, data, energy, offset_x, offset_y, scale_y))

        # Add to table (block table signals to avoid redundant cellChanged-triggered updates)
        row = self.dataset_table.rowCount()
        prev_block = self.dataset_table.blockSignals(True)
        self.dataset_table.insertRow(row)

        # Column 0: Dataset name (editable display text for quick copy/edit workflows)
        display_name = self._compact_dataset_name(name)
        name_item = QTableWidgetItem(display_name)
        name_item.setToolTip(name)
        name_item.setData(Qt.ItemDataRole.UserRole, name)
        self.dataset_table.setItem(row, 0, name_item)

        # Column 1: Number of points (read-only)
        points_item = QTableWidgetItem(str(len(data)))
        points_item.setFlags(points_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
        self.dataset_table.setItem(row, 1, points_item)

        # Column 2: Energy in eV (editable)
        energy_item = QTableWidgetItem(str(energy))
        self.dataset_table.setItem(row, 2, energy_item)

        # Column 3: Offset X (editable)
        offset_x_item = QTableWidgetItem(str(offset_x))
        self.dataset_table.setItem(row, 3, offset_x_item)

        # Column 4: Offset Y (editable)
        offset_y_item = QTableWidgetItem(str(offset_y))
        self.dataset_table.setItem(row, 4, offset_y_item)

        # Column 5: Scale Y (editable)
        scale_y_item = QTableWidgetItem(str(scale_y))
        self.dataset_table.setItem(row, 5, scale_y_item)
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
        if column_index == 2:  # E(eV) column
            self._update_q_conversion_availability()

    def _update_q_conversion_availability(self) -> None:
        """Update X->q checkbox availability based on X data and E(eV) column visibility."""
        # X->q conversion requires both custom X data AND visible E(eV) column
        has_x_data = self.x_data is not None
        energy_column_visible = self.btn_toggle_energy.isChecked()

        # Enable only if both conditions are met
        should_enable = has_x_data and energy_column_visible
        self.chk_convert_to_q.setEnabled(should_enable)

        # If disabling while checked, uncheck it
        if not should_enable and self.chk_convert_to_q.isChecked():
            self.chk_convert_to_q.setChecked(False)

        logging.info(f"X->q availability: X data={has_x_data}, E(eV) visible={energy_column_visible}, enabled={should_enable}")

    def _on_cell_changed(self, row: int, column: int) -> None:
        """
        Handle cell changes in the table (for energy, offset and scale editing).

        Args:
            row: Row index
            column: Column index
        """
        if column == 0:
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

        if column not in (2, 3, 4, 5):  # Only handle Energy (col 2), Offset X (col 3), Offset Y (col 4), and Scale Y (col 5)
            return

        if row < 0 or row >= len(self.datasets):
            return

        # Get the changed item
        changed_item = self.dataset_table.item(row, column)
        if changed_item is None:
            return

        try:
            new_value = float(changed_item.text())

            # Update internal dataset list
            name, data, old_energy, old_offset_x, old_offset_y, old_scale_y = self.datasets[row]

            if column == 2:  # Energy changed
                self.datasets[row] = (name, data, new_value, old_offset_x, old_offset_y, old_scale_y)
                logging.info(f"Updated Energy for '{name}': {old_energy} -> {new_value} eV")
            elif column == 3:  # Offset X changed
                self.datasets[row] = (name, data, old_energy, new_value, old_offset_y, old_scale_y)
                logging.info(f"Updated Offset X for '{name}': {old_offset_x} -> {new_value}")
            elif column == 4:  # Offset Y changed
                self.datasets[row] = (name, data, old_energy, old_offset_x, new_value, old_scale_y)
                logging.info(f"Updated Offset Y for '{name}': {old_offset_y} -> {new_value}")
            elif column == 5:  # Scale Y changed
                self.datasets[row] = (name, data, old_energy, old_offset_x, old_offset_y, new_value)
                logging.info(f"Updated Scale Y for '{name}': {old_scale_y} -> {new_value}")

            # Update plot
            self._update_plot()

        except ValueError:
            # Invalid number, reset to previous value
            if row < len(self.datasets):
                _, _, old_energy, old_offset_x, old_offset_y, old_scale_y = self.datasets[row]
                if column == 2:
                    old_value = old_energy
                elif column == 3:
                    old_value = old_offset_x
                elif column == 4:
                    old_value = old_offset_y
                else:  # column == 5
                    old_value = old_scale_y
                changed_item.setText(str(old_value))
            QMessageBox.warning(
                self,
                "Invalid Value",
                f"Invalid value. Please enter a valid number."
            )

    def _add_input_row(self) -> None:
        """Insert an editable input row for manual/paste dataset path import."""
        row = self.dataset_table.rowCount()
        self.dataset_table.blockSignals(True)
        self.dataset_table.insertRow(row)

        path_item = QTableWidgetItem("")
        path_item.setToolTip("Type or paste: file::/dataset/path , then press Enter")
        path_item.setData(self._ROLE_INPUT_ROW, True)
        self.dataset_table.setItem(row, 0, path_item)

        for col in (1, 2, 3, 4, 5):
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
                logging.info(f"Removed dataset: {removed[0]}")

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
        # Reset custom X state on close.
        self.x_data = None
        self.x_data_original = None
        self.x_dataset_path = None
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
        """Update the plot with current datasets, applying offsets and scale."""
        # Clear existing plot
        self.plot_widget.clear()

        # Reset selected marker and coordinates
        self.selected_marker = None
        self.selected_point = None
        self.label_coords.setText("X: - | Y: -")
        self.label_coords.setStyleSheet("color: gray; font-size: 9pt;")

        if not self.datasets:
            # Add legend back if cleared
            self.plot_widget.addLegend(offset=(-10, 10))  # Position in top-right corner
            return

        # Keep custom X only when all datasets share the same Y length.
        if self.x_data is not None:
            x_len = len(self.x_data)
            if any(len(data) != x_len for _, data, _, _, _, _ in self.datasets):
                logging.info(
                    "Resetting custom X in comparison due to size mismatch: len(X)=%s",
                    x_len,
                )
                self.x_data = None
                self.x_data_original = None
                self.x_dataset_path = None
                self.chk_convert_to_q.setChecked(False)
                self.chk_convert_to_q.setEnabled(False)

        # Determine X data to use and X-axis label
        if self.x_data is not None:
            base_x = self.x_data
            # Set X-axis label based on q conversion state
            if self.chk_convert_to_q.isChecked():
                self.plot_widget.setLabel("bottom", "q (A^-1)")
            else:
                self.plot_widget.setLabel("bottom", self._short_key_label(self.x_dataset_path) or "Custom X")
        else:
            base_x = None  # Will use indices
            self.plot_widget.setLabel("bottom", "Index")

        # Plot each dataset with offsets and scale
        import pyqtgraph as pg
        for i, (name, data, energy, offset_x, offset_y, scale_y) in enumerate(self.datasets):
            color = self.colors[i % len(self.colors)]
            # Create pen with line width
            pen = pg.mkPen(color=color, width=self.line_width)

            # Apply Y scale and offset to data
            # Order: first scale (multiply), then offset (add)
            data_transformed = (data * scale_y) + offset_y

            # Create plot with or without custom X, applying X offset and q conversion
            if base_x is not None and len(base_x) == len(data):
                # Apply q conversion if enabled
                if self.chk_convert_to_q.isChecked():
                    # Convert each angle to q using this dataset's energy
                    x_converted = np.array([self._convert_angle_to_q(angle, energy) for angle in base_x])
                    # Apply X offset to q data
                    x_with_offset = x_converted + offset_x
                else:
                    # Apply X offset to custom X data (angles)
                    x_with_offset = base_x + offset_x

                self.plot_widget.plot(
                    x_with_offset, data_transformed,
                    pen=pen,
                    name=self._format_name_with_transforms(name, energy, offset_x, offset_y, scale_y)
                )
            else:
                # Use default indices with X offset
                x_indices = np.arange(len(data)) + offset_x
                self.plot_widget.plot(
                    x_indices, data_transformed,
                    pen=pen,
                    name=self._format_name_with_transforms(name, energy, offset_x, offset_y, scale_y)
                )

        logging.info(f"Updated plot with {len(self.datasets)} datasets")

    def _format_name_with_transforms(self, name: str, energy: float, offset_x: float, offset_y: float, scale_y: float) -> str:
        """Format dataset name with transformation information."""
        name = self._legend_short_label(name)
        transforms = []

        # Show energy if q conversion is enabled
        if self.chk_convert_to_q.isChecked():
            transforms.append(f"{energy:.0f}eV")

        if offset_x != 0:
            transforms.append(f"X: {offset_x:+g}")

        if scale_y != 1.0:
            transforms.append(f"Y*{scale_y:g}")

        if offset_y != 0:
            transforms.append(f"Y: {offset_y:+g}")

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
        """Format legend label as file::dataset::colN when applicable."""
        if "::" not in full_name:
            return pathlib.Path(full_name).name
        file_part, dataset_part = full_name.split("::", 1)
        file_name = pathlib.Path(file_part).name

        # Preserve optional column suffix, e.g. [Col 3]
        match = re.search(r"\s*\[Col\s+(\d+)\]\s*$", dataset_part)
        col_suffix = f"::col{match.group(1)}" if match else ""
        dataset_core = dataset_part[:match.start()].strip() if match else dataset_part.strip()
        dataset_leaf = dataset_core.rstrip("/").split("/")[-1] if dataset_core else dataset_core

        base = f"{file_name}::{dataset_leaf}" if dataset_leaf else file_name
        return f"{base}{col_suffix}"

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
        if self.x_data is None:
            QMessageBox.information(
                self,
                "No X Axis",
                "Please select custom X axis data first using 'Select' button.\n\n"
                "The X data should contain angle values in degrees."
            )
            self.chk_convert_to_q.setChecked(False)
            return

        if state:  # Checked - convert X axis to q
            # Save original X data
            if self.x_data_original is None:
                self.x_data_original = self.x_data.copy()

            logging.info("Converting X-axis from angle to q for comparison")
        else:  # Unchecked - restore original X axis
            if self.x_data_original is not None:
                self.x_data = self.x_data_original.copy()
                logging.info("Restored original X-axis (angle) for comparison")

        # Update the plot
        self._update_plot()

    def _on_linewidth_changed(self, value: int) -> None:
        """Handle line width change."""
        self.line_width = value
        self._update_plot()
        logging.info(f"Changed line width to: {value}px")

    def _select_custom_x(self) -> None:
        """Open dialog to select custom X data."""
        from src.gui.plot_widget_1d_enhanced import XDataSelectionDialog

        dialog = XDataSelectionDialog(
            self.opened_files,
            self,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
        )
        dialog.data_selected.connect(self._on_x_data_selected)
        dialog.show()

    def _on_x_data_selected(self, x_data: np.ndarray, x_path: str) -> None:
        """Handle X data selection from dialog."""
        # Check if X data length matches any of the datasets
        if self.datasets:
            # Find the most common dataset length
            lengths = [len(data) for _, data, _, _, _, _ in self.datasets]
            most_common_length = max(set(lengths), key=lengths.count)

            if len(x_data) != most_common_length:
                QMessageBox.warning(
                    self,
                    "Data Size Mismatch",
                    f"X data length ({len(x_data)}) does not match the most common Y data length ({most_common_length}).\n\n"
                    "Some datasets may not display correctly."
                )

        self.x_data = x_data
        self.x_dataset_path = x_path

        # Update X->q conversion checkbox availability (requires both X data and E(eV) column)
        self._update_q_conversion_availability()

        # If X->q conversion is currently enabled, update the original data reference
        if self.chk_convert_to_q.isChecked():
            self.x_data_original = x_data.copy()
            logging.info(f"Set custom X data with q conversion enabled: {x_path}")
        else:
            logging.info(f"Set custom X data: {x_path}")

        self._update_plot()

    def _on_plot_clicked(self, event) -> None:
        """Handle mouse click on plot to select data point."""
        if not self.datasets:
            return

        import pyqtgraph as pg

        # Get mouse position in scene (pixel) coordinates
        click_pos = event.scenePos()
        vb = self.plot_widget.plotItem.vb
        (x0, x1), (y0, y1) = vb.viewRange()
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)

        # Find closest point across all datasets using pixel distance
        min_distance = float('inf')
        closest_x = None
        closest_y = None
        closest_dataset_name = None

        for name, data, energy, offset_x, offset_y, scale_y in self.datasets:
            # Determine X values for this dataset
            if self.x_data is not None and len(self.x_data) == len(data):
                # Apply q conversion if enabled
                if self.chk_convert_to_q.isChecked():
                    # Convert each angle to q using this dataset's energy
                    x_converted = np.array([self._convert_angle_to_q(angle, energy) for angle in self.x_data])
                    x_values = x_converted + offset_x
                else:
                    x_values = self.x_data + offset_x
            else:
                x_values = np.arange(len(data)) + offset_x

            # Apply transformations
            y_values = data * scale_y + offset_y

            # Build display-space arrays (same coordinates used by the view)
            if self.chk_log_x.isChecked():
                valid_x = x_values > 0
                display_x_all = np.zeros_like(x_values, dtype=float)
                display_x_all[valid_x] = np.log10(x_values[valid_x])
            else:
                valid_x = np.ones_like(x_values, dtype=bool)
                display_x_all = x_values.astype(float, copy=False)

            if self.chk_log_y.isChecked():
                valid_y = y_values > 0
                display_y_all = np.zeros_like(y_values, dtype=float)
                display_y_all[valid_y] = np.log10(y_values[valid_y])
            else:
                valid_y = np.ones_like(y_values, dtype=bool)
                display_y_all = y_values.astype(float, copy=False)

            valid_mask = valid_x & valid_y
            visible_mask = (
                (display_x_all >= x_min) & (display_x_all <= x_max) &
                (display_y_all >= y_min) & (display_y_all <= y_max)
            )

            # Performance: only scan points currently visible in the viewport.
            candidate_idx = np.flatnonzero(valid_mask & visible_mask)
            if candidate_idx.size == 0:
                candidate_idx = np.flatnonzero(valid_mask)

            # Find closest point in this dataset using pixel distance
            for i in candidate_idx:
                x_val = x_values[i]
                y_val = y_values[i]
                display_x = display_x_all[i]
                display_y = display_y_all[i]

                # Map data coordinates to scene coordinates
                point_in_view = self.plot_widget.plotItem.vb.mapViewToScene(
                    pg.Point(display_x, display_y)
                )

                # Calculate pixel distance
                dx_pixels = point_in_view.x() - click_pos.x()
                dy_pixels = point_in_view.y() - click_pos.y()
                distance = dx_pixels**2 + dy_pixels**2

                if distance < min_distance:
                    min_distance = distance
                    closest_x = x_val
                    closest_y = y_val
                    closest_dataset_name = name

        if closest_x is not None and closest_y is not None:
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

        first_energy = self.datasets[0][2]
        first_offset_x = self.datasets[0][3]
        x_len = len(self.x_data) if self.x_data is not None else None

        for name, data, energy, offset_x, _offset_y, _scale_y in self.datasets:
            if abs(offset_x - first_offset_x) > 1e-12:
                return False, (
                    f"Offset X differs ({name}: {offset_x:g}, first: {first_offset_x:g})."
                )
            if self.chk_convert_to_q.isChecked() and abs(energy - first_energy) > 1e-12:
                return False, (
                    f"Energy differs in q mode ({name}: {energy:g} eV, first: {first_energy:g} eV)."
                )
            if x_len is not None and len(data) != x_len:
                return False, (
                    f"Y length differs from shared X length ({name}: len(Y)={len(data)}, len(X)={x_len})."
                )
        return True, ""

    def _build_export_series(
        self, data: np.ndarray, energy: float, offset_x: float, offset_y: float, scale_y: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build transformed x/y series exactly matching current plot logic."""
        if self.x_data is not None and len(self.x_data) == len(data):
            if self.chk_convert_to_q.isChecked():
                x_values = np.array([self._convert_angle_to_q(angle, energy) for angle in self.x_data]) + offset_x
            else:
                x_values = self.x_data + offset_x
        else:
            x_values = np.arange(len(data)) + offset_x

        y_values = data * scale_y + offset_y
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
        """Build default comparison export name: head_firstNum_lastNum_compar."""
        if not self.datasets:
            return "scan_0000_0000_compar"

        def _file_part(name: str) -> str:
            return name.split("::", 1)[0].strip() if "::" in name else name.strip()

        name1 = self.datasets[0][0]
        name2 = self.datasets[-1][0] if len(self.datasets) > 1 else self.datasets[0][0]
        head1, n1 = self._scan_head_and_number(_file_part(name1))
        head2, n2 = self._scan_head_and_number(_file_part(name2))
        head = head1 if head1 else (head2 if head2 else "scan")
        return f"{head}_{n1}_{n2}_compar"

    def _capture_plot_pixmap(self):
        """Capture comparison plot area as pixmap."""
        try:
            return self.plot_widget.grab()
        except Exception as e:
            logging.error(f"Failed to capture comparison plot: {e}")
            return None

    def _save_plot_image(self) -> None:
        """Save comparison plot screenshot."""
        pixmap = self._capture_plot_pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "No Image", "No plot image available to save.")
            return

        from PyQt6.QtCore import QSettings
        settings = QSettings()
        last_dir = settings.value("paths/last_export_directory", pathlib.Path.home())
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Plot Image",
            str(pathlib.Path(last_dir) / "comparison_plot.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)",
        )
        if not file_path:
            return

        settings.setValue("paths/last_export_directory", pathlib.Path(file_path).parent)

        fmt = "PNG"
        if "JPEG" in selected_filter or file_path.lower().endswith((".jpg", ".jpeg")):
            fmt = "JPG"
        elif "BMP" in selected_filter or file_path.lower().endswith(".bmp"):
            fmt = "BMP"

        ok = pixmap.save(file_path, fmt, 95 if fmt == "JPG" else -1)
        if not ok:
            QMessageBox.critical(self, "Save Failed", f"Failed to save image:\n{file_path}")

    def _copy_plot_image(self) -> None:
        """Copy comparison plot screenshot to clipboard."""
        pixmap = self._capture_plot_pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "No Image", "No plot image available to copy.")
            return
        QApplication.clipboard().setPixmap(pixmap)
        logging.info("Copied comparison plot image to clipboard")

    def _export_to_csv(self) -> None:
        """Export comparison data with automatic X-column strategy."""
        if not self.datasets:
            QMessageBox.information(
                self,
                "No Data",
                "No datasets to export. Please add datasets to the comparison list first."
            )
            return

        # Open file save dialog
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QSettings

        settings = QSettings()
        last_dir = settings.value("paths/last_export_directory", pathlib.Path.home())

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Data",
            str(pathlib.Path(last_dir) / f"{self._default_export_base_name()}.csv"),
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        # Save directory for next time
        settings.setValue("paths/last_export_directory", pathlib.Path(file_path).parent)

        try:
            # Determine delimiter based on file extension
            file_ext = pathlib.Path(file_path).suffix.lower()
            delimiter = "\t" if file_ext == ".txt" else ","

            # Determine the maximum length of all datasets
            max_length = max(len(data) for _, data, _, _, _, _ in self.datasets)
            has_custom_x = self.x_data is not None
            compatible, reason = self._is_shared_xq_compatible()
            use_shared_mode = compatible

            # Build transformed data series
            transformed = []
            for name, data, energy, offset_x, offset_y, scale_y in self.datasets:
                x_values, y_values = self._build_export_series(data, energy, offset_x, offset_y, scale_y)
                transformed.append((name, x_values, y_values, energy, offset_x, offset_y, scale_y))

            # Write data file with UTF-8 BOM for better Excel compatibility
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=delimiter)

                # Metadata rows
                writer.writerow([f"# Export mode: {'Aligned table' if use_shared_mode else 'Per-dataset columns'}"])
                writer.writerow([f"# q conversion: {'ON' if self.chk_convert_to_q.isChecked() else 'OFF'}"])
                if not use_shared_mode and reason:
                    writer.writerow([f"# auto-switch reason: {reason}"])
                for name, _x_values, _y_values, energy, offset_x, offset_y, scale_y in transformed:
                    writer.writerow([f"# {name}: E={energy:g} eV, OffsetX={offset_x:g}, OffsetY={offset_y:g}, ScaleY={scale_y:g}"])

                if use_shared_mode:
                    # Shared X/q columns + per-dataset Y columns
                    header = []
                    shared_x = None
                    if has_custom_x:
                        first_name, first_x, _first_y, _e, _ox, _oy, _sy = transformed[0]
                        shared_x = first_x
                        header.append("q" if self.chk_convert_to_q.isChecked() else "X")
                    for name, _x_values, _y_values, _e, _ox, _oy, _sy in transformed:
                        header.append(f"{name}_Y")
                    writer.writerow(header)

                    for row_idx in range(max_length):
                        row = []
                        if shared_x is not None:
                            row.append(f"{shared_x[row_idx]:.10g}" if row_idx < len(shared_x) else "")
                        for _name, _x_values, y_values, _e, _ox, _oy, _sy in transformed:
                            row.append(f"{y_values[row_idx]:.10g}" if row_idx < len(y_values) else "")
                        writer.writerow(row)
                else:
                    # Per-dataset X/q and Y columns
                    header = []
                    for name, _x_values, _y_values, _e, _ox, _oy, _sy in transformed:
                        x_label = f"{name}_q" if self.chk_convert_to_q.isChecked() else f"{name}_X"
                        header.extend([x_label, f"{name}_Y"])
                    writer.writerow(header)

                    for row_idx in range(max_length):
                        row = []
                        for _name, x_values, y_values, _e, _ox, _oy, _sy in transformed:
                            row.append(f"{x_values[row_idx]:.10g}" if row_idx < len(x_values) else "")
                            row.append(f"{y_values[row_idx]:.10g}" if row_idx < len(y_values) else "")
                        writer.writerow(row)

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





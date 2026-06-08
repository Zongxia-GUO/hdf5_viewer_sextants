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
import pyqtgraph as pg
from PyQt6.QtCore import QBuffer, QByteArray, QMimeData, QTimer, Qt
from PyQt6.QtGui import QClipboard, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
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
from src.gui.dataset_path_combo import DatasetPathCombo


class DragDropComboBox(DatasetPathCombo):
    """Calculator dataset selector (shared behavior with FTH path combo)."""

    def __init__(self, placeholder: str, parent: Any = None) -> None:
        super().__init__(placeholder=placeholder, parent=parent)


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

        # Dataset A column selection
        col_a_layout = QHBoxLayout()
        col_a_layout.addWidget(QLabel("Column:"))
        self.spin_col_a = QComboBox()
        self.spin_col_a.setEnabled(False)
        self.spin_col_a.setMinimumWidth(150)
        self.spin_col_a.setToolTip("Select column for multi-column datasets")
        col_a_layout.addWidget(self.spin_col_a)
        col_a_layout.addStretch()
        selection_layout.addRow("", col_a_layout)

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

        # Dataset B column selection
        col_b_layout = QHBoxLayout()
        col_b_layout.addWidget(QLabel("Column:"))
        self.spin_col_b = QComboBox()
        self.spin_col_b.setEnabled(False)
        self.spin_col_b.setMinimumWidth(150)
        self.spin_col_b.setToolTip("Select column for multi-column datasets")
        col_b_layout.addWidget(self.spin_col_b)
        col_b_layout.addStretch()
        selection_layout.addRow("", col_b_layout)

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
        help_label = QLabel("<i>Available: A, optional B, +, -, *, /, **, FFT(A), np.sqrt(), np.abs(), np.log(), etc.</i>")
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

        # Export button
        self.btn_export = QPushButton("Export Result...")
        self.btn_export.setAutoDefault(False)  # Prevent Enter key from triggering this button
        self.btn_export.clicked.connect(self._export_result)
        self.btn_export.setEnabled(False)
        left_layout.addWidget(self.btn_export)

        # Image export/copy helpers for the result display panel
        image_ops_layout = QHBoxLayout()
        self.btn_save_image = QPushButton("Save Image...")
        self.btn_save_image.setAutoDefault(False)
        self.btn_save_image.setEnabled(False)
        self.btn_save_image.clicked.connect(self._save_result_image)
        image_ops_layout.addWidget(self.btn_save_image)

        self.btn_copy_image = QPushButton("Copy Image")
        self.btn_copy_image.setAutoDefault(False)
        self.btn_copy_image.setEnabled(False)
        self.btn_copy_image.clicked.connect(self._copy_result_image)
        image_ops_layout.addWidget(self.btn_copy_image)
        left_layout.addLayout(image_ops_layout)

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

        # Placeholder for result display
        self.result_container = QWidget()
        self.result_container_layout = QVBoxLayout()
        self.result_container_layout.setContentsMargins(0, 0, 0, 0)

        # Initial empty plot widget (like main window at startup)
        initial_plot = pg.PlotWidget()
        initial_plot.setLabel("bottom", "Index")
        initial_plot.setLabel("left", "Value")
        initial_plot.showGrid(x=True, y=True, alpha=0.3)
        initial_plot.setBackground('k')  # Dark theme

        # Set axis colors to white for dark theme
        axis_pen = pg.mkPen(color='w', width=1)
        for axis in ['left', 'bottom', 'right', 'top']:
            initial_plot.getAxis(axis).setPen(axis_pen)
            initial_plot.getAxis(axis).setTextPen(axis_pen)

        initial_plot.plotItem.vb.setMenuEnabled(False)
        self.result_container_layout.addWidget(initial_plot)

        self.result_container.setLayout(self.result_container_layout)
        right_layout.addWidget(self.result_container)

        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)

        # Set initial splitter sizes to ensure result area toolbar is fully visible
        # Left: 300px for controls, Right: 850px for result with toolbar
        splitter.setSizes([300, 850])

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

    def _on_dataset_a_changed(self) -> None:
        """Update dataset A info label and column selector when selection changes."""
        if self._is_populating_combos:
            return
        logging.info("_on_dataset_a_changed: START")
        resolved = self._resolve_dataset_from_combo(self.combo_dataset_a)
        if resolved:
            file_path, dataset_path = resolved
            logging.info(f"_on_dataset_a_changed: file_path={file_path}, dataset_path={dataset_path}")
            try:
                logging.info(f"_on_dataset_a_changed: Opening file {file_path}")
                with h5py.File(file_path, "r") as f:
                    logging.info(f"_on_dataset_a_changed: File opened, accessing dataset")
                    dataset = f[dataset_path]
                    logging.info(f"_on_dataset_a_changed: Got dataset, shape={dataset.shape}")
                    shape = dataset.shape
                    self.label_info_a.setText(f"Shape: {shape}, Type: {dataset.dtype}")

                    # Store current column selection
                    current_col = self.spin_col_a.currentData()
                    logging.info(f"_on_dataset_a_changed: Current col={current_col}")

                    # Update column selector for A
                    logging.info(f"_on_dataset_a_changed: Clearing column selector")
                    self.spin_col_a.clear()
                    if len(shape) == 2:
                        # Multi-column dataset - enable column selection
                        num_cols = shape[1]
                        self.spin_col_a.addItem("All columns", -1)
                        for i in range(num_cols):
                            self.spin_col_a.addItem(f"Column {i}", i)
                        self.spin_col_a.setEnabled(True)
                        self.label_info_a.setText(
                            f"Shape: {shape}, Type: {dataset.dtype} | <b>{num_cols} columns</b>"
                        )

                        # Restore previous column selection if it was valid
                        if current_col is not None and current_col >= -1 and current_col < num_cols:
                            for i in range(self.spin_col_a.count()):
                                if self.spin_col_a.itemData(i) == current_col:
                                    self.spin_col_a.setCurrentIndex(i)
                                    break
                    else:
                        # 1D dataset - disable column selection
                        self.spin_col_a.addItem("N/A", -1)
                        self.spin_col_a.setEnabled(False)

            except Exception as e:
                logging.error(f"_on_dataset_a_changed: Error: {e}")
                self.label_info_a.setText(f"Error: {e}")
                self.spin_col_a.clear()
                self.spin_col_a.setEnabled(False)
        else:
            self.label_info_a.setText("Shape: -, Type: -")
            self.spin_col_a.clear()
            self.spin_col_a.addItem("N/A", -1)
            self.spin_col_a.setEnabled(False)

        logging.info("_on_dataset_a_changed: END")

    def _on_dataset_b_changed(self) -> None:
        """Update dataset B info label and column selector when selection changes."""
        if self._is_populating_combos:
            return
        resolved = self._resolve_dataset_from_combo(self.combo_dataset_b)
        if resolved:
            file_path, dataset_path = resolved
            try:
                with h5py.File(file_path, "r") as f:
                    dataset = f[dataset_path]
                    shape = dataset.shape
                    self.label_info_b.setText(f"Shape: {shape}, Type: {dataset.dtype}")

                    # Store current column selection
                    current_col = self.spin_col_b.currentData()

                    # Update column selector for B
                    self.spin_col_b.clear()
                    if len(shape) == 2:
                        # Multi-column dataset - enable column selection
                        num_cols = shape[1]
                        self.spin_col_b.addItem("All columns", -1)
                        for i in range(num_cols):
                            self.spin_col_b.addItem(f"Column {i}", i)
                        self.spin_col_b.setEnabled(True)
                        self.label_info_b.setText(
                            f"Shape: {shape}, Type: {dataset.dtype} | <b>{num_cols} columns</b>"
                        )

                        # Restore previous column selection if it was valid
                        if current_col is not None and current_col >= -1 and current_col < num_cols:
                            for i in range(self.spin_col_b.count()):
                                if self.spin_col_b.itemData(i) == current_col:
                                    self.spin_col_b.setCurrentIndex(i)
                                    break
                    else:
                        # 1D dataset - disable column selection
                        self.spin_col_b.addItem("N/A", -1)
                        self.spin_col_b.setEnabled(False)

            except Exception as e:
                self.label_info_b.setText(f"Error: {e}")
                self.spin_col_b.clear()
                self.spin_col_b.setEnabled(False)
        else:
            self.label_info_b.setText("Shape: -, Type: -")
            self.spin_col_b.clear()
            self.spin_col_b.addItem("N/A", -1)
            self.spin_col_b.setEnabled(False)

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

            # Apply column selection for A
            col_a = self.spin_col_a.currentData()
            if col_a is not None and col_a >= 0 and data_a.ndim == 2:
                # Select specific column
                data_a = data_a[:, col_a]
                logging.info(f"Dataset A: Selected column {col_a}, new shape: {data_a.shape}")

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

                # Apply column selection for B
                col_b = self.spin_col_b.currentData()
                if col_b is not None and col_b >= 0 and data_b.ndim == 2:
                    # Select specific column
                    data_b = data_b[:, col_b]
                    logging.info(f"Dataset B: Selected column {col_b}, new shape: {data_b.shape}")

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

        # Check shape compatibility
        if data_b is not None and data_a.shape != data_b.shape:
            reply = QMessageBox.question(
                self,
                "Shape Mismatch",
                f"Dataset A shape: {data_a.shape}\n"
                f"Dataset B shape: {data_b.shape}\n\n"
                f"Shapes don't match. Attempt operation anyway?\n"
                f"(Broadcasting may occur)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        try:
            self._last_operation_expr = expression
            # Save original data for export
            self.data_a = data_a
            self.data_b = data_b

            # Create namespace for evaluation
            # Convert to float64 to handle negative values from subtraction
            namespace = {
                "__builtins__": {},
                "A": data_a.astype(np.float64),
                "np": np,
                "FFT": self._fft_magnitude,
            }
            if data_b is not None:
                namespace["B"] = data_b.astype(np.float64)

            # Evaluate expression
            self.result_data = eval(expression, namespace)
            self.result_data = np.asarray(self.result_data)

            # Automatically squeeze out dimensions of size 1
            original_shape = self.result_data.shape
            self.result_data = np.squeeze(self.result_data)
            squeezed = original_shape != self.result_data.shape

            # Update result display in right panel
            self._update_result_display()

            # Enable export button
            self.btn_export.setEnabled(True)
            self.btn_save_image.setEnabled(True)
            self.btn_copy_image.setEnabled(True)
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

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to perform calculation:\n{e}")
            logging.error(f"Calculation error: {e}")

    def _update_result_display(self) -> None:
        """Update the result display panel with calculation result."""
        # Clear current result widget
        for i in reversed(range(self.result_container_layout.count())):
            widget = self.result_container_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Use unified data viewer for automatic widget selection
        from src.gui.unified_data_viewer import UnifiedDataViewer

        viewer = UnifiedDataViewer(
            parent=self,
            opened_files=self.opened_files,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
        )
        viewer.q_calibration_requested.connect(self._on_q_request_from_result_viewer)
        viewer.set_data(self.result_data, source_dataset_key="Result::calculation_result")
        self.result_container_layout.addWidget(viewer)
        self.result_widget = viewer

    def _on_q_request_from_result_viewer(self, _source_dataset_key: object) -> None:
        """Handle Q button from calculator result view and open Q calibration tool."""
        mw = self.parent()
        try:
            if (
                mw is not None
                and self.result_data is not None
                and getattr(self.result_data, "ndim", 0) == 2
                and hasattr(mw, "open_q_tool_for_array")
            ):
                mw.open_q_tool_for_array(np.asarray(self.result_data), source_label="calculator_result")
            # Non-2D results do not expose Q button in UI; ignore silently.
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
                btn_all = msg.addButton("Transfer all columns", QMessageBox.ButtonRole.AcceptRole)
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

    def _result_capture_pixmap(self) -> QPixmap | None:
        """Capture result image based on active viewer type."""
        if self.result_widget is None:
            return None
        try:
            from src.gui.unified_data_viewer import UnifiedDataViewer
            from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
            from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

            target = self.result_widget
            if isinstance(self.result_widget, UnifiedDataViewer):
                current = self.result_widget.get_current_widget()
                if current is not None:
                    target = current

            # 2D image view: capture the rendered image area (includes histogram/colorbar)
            if isinstance(target, ImageView2DEnhanced):
                return target.graphics_layout.grab()

            # 1D/curve view: capture the graph display only
            if isinstance(target, PlotWidget1DEnhanced):
                return target.plot_widget.grab()

            # Fallback for other widget types
            return target.grab()
        except Exception as e:
            logging.error(f"Failed to capture result pixmap: {e}")
            return None

    def _current_result_image_view(self):
        """Return active 2D result image viewer, if the result is image-like."""
        if self.result_widget is None:
            return None
        try:
            from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
            from src.gui.unified_data_viewer import UnifiedDataViewer

            target = self.result_widget
            if isinstance(target, UnifiedDataViewer):
                target = target.get_current_widget()
            if isinstance(target, ImageView2DEnhanced):
                return target
        except Exception:
            return None
        return None

    def _save_result_image(self) -> None:
        """Save current result view as an image."""
        image_view = self._current_result_image_view()
        if image_view is not None:
            image_view.save_colormapped_image_dialog(self._default_export_base_name())
            return

        pixmap = self._result_capture_pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "No Image", "No result image available to save.")
            return

        from src.gui.image_view_2d_enhanced import IMAGE_SAVE_FILTER, ImageView2DEnhanced
        from PyQt6.QtCore import QSettings

        settings = QSettings()
        saved_dir = settings.value("paths/last_export_directory", defaultValue=str(pathlib.Path.home()))
        default_dir = pathlib.Path(str(saved_dir)) if saved_dir else pathlib.Path.home()
        if not default_dir.exists():
            default_dir = pathlib.Path.home()
        base = self._default_export_base_name()
        default_name = str(default_dir / f"{base}.png")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Result Image",
            default_name,
            IMAGE_SAVE_FILTER,
        )
        if not file_path:
            return

        export_path = pathlib.Path(file_path)
        if not export_path.suffix:
            export_path = export_path.with_suffix(ImageView2DEnhanced.image_extension_from_filter(selected_filter))
            file_path = str(export_path)

        settings.setValue("paths/last_export_directory", str(export_path.parent))
        fmt = ImageView2DEnhanced.image_format_from_path(file_path, selected_filter)
        ok = pixmap.save(file_path, fmt, 95 if fmt == "JPEG" else -1)
        if ok:
            logging.info(f"Saved result image to: {file_path}")
        else:
            QMessageBox.critical(self, "Save Failed", f"Failed to save image:\n{file_path}")

    def _copy_result_image(self) -> None:
        """Copy current result image to system clipboard."""
        image_view = self._current_result_image_view()
        if image_view is not None:
            image_view.copy_colormapped_image_to_clipboard()
            return

        pixmap = self._result_capture_pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "No Image", "No result image available to copy.")
            return

        clipboard: QClipboard = QApplication.clipboard()
        clipboard.setPixmap(pixmap)

        # Also attach JPEG bytes so image-focused apps can paste as JPEG when supported.
        try:
            jpeg_bytes = QByteArray()
            buf = QBuffer(jpeg_bytes)
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            pixmap.toImage().save(buf, "JPEG", 95)
            buf.close()

            mime = QMimeData()
            mime.setImageData(pixmap.toImage())
            if not jpeg_bytes.isEmpty():
                mime.setData("image/jpeg", jpeg_bytes)
            clipboard.setMimeData(mime)
        except Exception as e:
            logging.debug(f"Failed to add JPEG mime payload: {e}")

        logging.info("Copied result image to clipboard")

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

    def _default_export_base_name(self) -> str:
        """Build default filename: head_numA_numB_suffix."""
        entry_a = self.combo_dataset_a.get_entry(opened_files=self.opened_files)
        entry_b = self.combo_dataset_b.get_entry(opened_files=self.opened_files)
        if entry_a and entry_b:
            head_a, n1 = self._scan_head_and_number(str(entry_a[0]))
            head_b, n2 = self._scan_head_and_number(str(entry_b[0]))
            head = head_a if head_a else (head_b if head_b else "scan")
        elif entry_a:
            head, n1 = self._scan_head_and_number(str(entry_a[0]))
            n2 = "single"
        else:
            head, n1, n2 = "scan", "0000", "0000"
        suffix = self._suffix_from_expression(self._last_operation_expr)
        return f"{head}_{n1}_{n2}_{suffix}"


    def _export_result(self) -> None:
        """Export the result to a file."""
        if self.result_data is None:
            return

        try:
            from PyQt6.QtWidgets import QFileDialog
            from src.lib_h5.data_exporter import DataExporter
            from src.lib_h5.dataset_types import H5DatasetType
            from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

            # Log shape for debugging
            logging.info(f"Exporting result - Shape: {self.result_data.shape}, Dims: {len(self.result_data.shape)}")

            # Determine data type based on dimensions and size
            ndim = len(self.result_data.shape)

            # For 1D data with original datasets, export all together
            if ndim == 1 and self.data_a is not None and self.data_b is not None:
                # Check if result viewer has custom X data and q conversion
                from src.gui.unified_data_viewer import UnifiedDataViewer

                has_custom_x = False
                has_q_conversion = False
                x_data = None
                q_data = None
                plot_widget = None

                # Get the actual plot widget from UnifiedDataViewer
                if isinstance(self.result_widget, UnifiedDataViewer):
                    current_widget = self.result_widget.get_current_widget()
                    if isinstance(current_widget, PlotWidget1DEnhanced):
                        plot_widget = current_widget
                        has_custom_x = plot_widget.x_data is not None
                        has_q_conversion = plot_widget.chk_convert_to_q.isChecked()
                        if has_custom_x:
                            if has_q_conversion and plot_widget.x_data_original is not None:
                                x_data = plot_widget.x_data_original
                                q_data = plot_widget.x_data
                            else:
                                x_data = plot_widget.x_data

                self._export_1d_with_datasets(x_data, q_data, has_custom_x, has_q_conversion)
                return

            if ndim == 1:
                # 1D array - export as CSV or text
                data_type = H5DatasetType.Array1D
            elif ndim == 2:
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
                f"{self._default_export_base_name()}{default_ext}",
                file_filter,
            )

            if not file_path:
                return

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
            error_msg = f"Failed to export result:\n{e}\n\nShape: {self.result_data.shape}, Type: {self.result_data.dtype}"
            QMessageBox.critical(self, "Export Error", error_msg)
            logging.error(f"Export error: {e}, shape: {self.result_data.shape}")

    def _export_1d_with_datasets(self, x_data: np.ndarray | None, q_data: np.ndarray | None,
                                  has_custom_x: bool, has_q_conversion: bool) -> None:
        """Export 1D data with original datasets A and B, and result (with or without X and q axes)."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            from PyQt6.QtCore import QSettings
            import csv

            settings = QSettings()
            last_dir = settings.value("paths/last_export_directory", pathlib.Path.home())

            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export Calculation Result",
                str(pathlib.Path(last_dir) / f"{self._default_export_base_name()}.csv"),
                "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
            )

            if not file_path:
                return  # User cancelled

            # Save directory for next time
            settings.setValue("paths/last_export_directory", pathlib.Path(file_path).parent)

            # Determine delimiter based on file extension
            file_ext = pathlib.Path(file_path).suffix.lower()
            delimiter = "\t" if file_ext == ".txt" else ","

            # Determine the length to iterate
            max_len = max(len(self.data_a), len(self.data_b), len(self.result_data))

            # Write data file with UTF-8 BOM for better Excel compatibility
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=delimiter)

                # Write header based on whether we have X data and q conversion
                header = []
                if has_custom_x and x_data is not None:
                    header.append("X")
                    max_len = max(max_len, len(x_data))
                    if has_q_conversion and q_data is not None:
                        header.append("q")
                        max_len = max(max_len, len(q_data))
                else:
                    header.append("Index")

                header.extend(["Data_A", "Data_B", "Result"])
                writer.writerow(header)

                # Write data rows
                for i in range(max_len):
                    row = []

                    # First column: X, q, or Index
                    if has_custom_x and x_data is not None:
                        # With X data
                        row.append(f"{x_data[i]:.10g}" if i < len(x_data) else "")
                        # Add q column if conversion is enabled
                        if has_q_conversion and q_data is not None:
                            row.append(f"{q_data[i]:.10g}" if i < len(q_data) else "")
                    else:
                        # Without X data, use index
                        row.append(str(i))

                    # Data columns
                    row.extend([
                        f"{self.data_a[i]:.10g}" if i < len(self.data_a) else "",
                        f"{self.data_b[i]:.10g}" if i < len(self.data_b) else "",
                        f"{self.result_data[i]:.10g}" if i < len(self.result_data) else ""
                    ])

                    writer.writerow(row)

            logging.info(f"Exported calculation result to: {file_path}")

            # Build columns info string
            if has_custom_x:
                if has_q_conversion:
                    columns_info = "X, q, Data_A, Data_B, Result"
                else:
                    columns_info = "X, Data_A, Data_B, Result"
            else:
                columns_info = "Index, Data_A, Data_B, Result"

            QMessageBox.information(
                self,
                "Export Successful",
                f"Calculation data exported successfully to:\n{file_path}\n\n"
                f"Columns: {columns_info}"
            )

        except Exception as e:
            logging.error(f"Failed to export calculation with X-axis: {e}")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export data:\n{str(e)}"
            )

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
        logging.info(f"add_to_dataset_a: END (not found)")
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




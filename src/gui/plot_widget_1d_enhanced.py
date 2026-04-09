"""Enhanced 1D Plot Widget with axis controls and custom X/Y data support."""

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

import csv
import logging
import pathlib
from typing import Any, Optional

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QBuffer, QByteArray, QMimeData, QSize
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.img.img_path import img_path


class PlotWidget1DEnhanced(QWidget):
    """Enhanced 1D plot widget with axis controls and custom X/Y data."""

    def __init__(self, parent=None, opened_files=None, dataset_full_keys_1d: list[str] | None = None):
        """
        Initialize the enhanced plot widget.

        Args:
            parent: Parent widget
            opened_files: Tuple of opened HDF5 file paths (for custom X data selection)
        """
        super().__init__(parent)
        self.opened_files = opened_files or tuple()
        self.dataset_full_keys_1d = list(dataset_full_keys_1d or [])
        self.y_data = None
        self.x_data = None
        self.x_data_original = None  # Store original X data before q conversion
        self.x_dataset_path = None  # Full path to custom X dataset
        self.y_source_dataset_key = None  # Full key for Y data source (file::dataset)
        self.selected_point = None  # (x, y, curve_idx) of selected point (curve_idx is None for 1D data)
        self.selected_marker = None  # Circle marker for selected point

        self._init_ui()

    def refresh_dataset_keys(
        self,
        full_keys_1d: list[str],
        opened_files: tuple[pathlib.Path, ...] | None = None,
    ) -> None:
        """Refresh shared dataset keys used by Select X Data dialog."""
        self.dataset_full_keys_1d = list(full_keys_1d or [])
        if opened_files is not None:
            self.opened_files = tuple(opened_files)

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Set size policy to allow shrinking horizontally (ignore minimum size hints)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Control panel at top
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)  # Compact spacing between widgets

        self.btn_copy_plot = QPushButton()
        self.btn_copy_plot.setIcon(QIcon(str(pathlib.Path(img_path(), "copy.ico"))))
        self.btn_copy_plot.setIconSize(QSize(16, 16))
        self.btn_copy_plot.setFixedSize(28, 24)
        self.btn_copy_plot.setToolTip("Copy current plot screenshot")
        self.btn_copy_plot.clicked.connect(self.copy_plot_screenshot_to_clipboard)
        control_layout.addWidget(self.btn_copy_plot)

        self.btn_save_plot = QPushButton()
        self.btn_save_plot.setIcon(QIcon(str(pathlib.Path(img_path(), "save.ico"))))
        self.btn_save_plot.setIconSize(QSize(16, 16))
        self.btn_save_plot.setFixedSize(28, 24)
        self.btn_save_plot.setToolTip("Save/export current dataset")
        self.btn_save_plot.clicked.connect(self._request_main_export)
        control_layout.addWidget(self.btn_save_plot)

        control_layout.addSpacing(10)

        # Axis scale controls
        scale_label = QLabel("Axis Scale:")
        control_layout.addWidget(scale_label)

        self.chk_log_x = QCheckBox("Log X")
        self.chk_log_x.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_x)

        self.chk_log_y = QCheckBox("Log Y")
        self.chk_log_y.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_y)

        control_layout.addSpacing(10)

        # Custom X data control
        if self.opened_files:
            x_label = QLabel("X Axis:")
            control_layout.addWidget(x_label)

            self.btn_select_x = QPushButton("Select")
            self.btn_select_x.setAutoDefault(False)  # Prevent Enter key from triggering this button
            # Set maximum width to allow toolbar to compress when window is resized
            self.btn_select_x.setMinimumWidth(0)  # Allow shrinking
            self.btn_select_x.setMaximumWidth(60)
            self.btn_select_x.clicked.connect(self._select_custom_x)
            control_layout.addWidget(self.btn_select_x)

            control_layout.addSpacing(10)

        # Line width control
        linewidth_label = QLabel("Line Width:")
        control_layout.addWidget(linewidth_label)

        self.spinbox_linewidth = QSpinBox()
        self.spinbox_linewidth.setMinimum(1)
        self.spinbox_linewidth.setMaximum(10)
        self.spinbox_linewidth.setValue(3)  # Default line width
        self.spinbox_linewidth.setSuffix(" px")
        self.spinbox_linewidth.setToolTip("Line width (press Enter to apply)")
        self.spinbox_linewidth.editingFinished.connect(self._update_plot)  # Only trigger on Enter or focus loss
        control_layout.addWidget(self.spinbox_linewidth)

        control_layout.addSpacing(10)

        # Q conversion for scattering experiments
        self.chk_convert_to_q = QCheckBox("X to q")
        self.chk_convert_to_q.setToolTip("Convert X-axis angle to momentum transfer q")
        self.chk_convert_to_q.setEnabled(False)  # Disabled until X data is loaded
        self.chk_convert_to_q.stateChanged.connect(self._on_q_conversion_changed)
        control_layout.addWidget(self.chk_convert_to_q)

        # Photon energy input for q conversion
        energy_label = QLabel("E(eV):")
        control_layout.addWidget(energy_label)

        self.input_energy = QLineEdit()
        self.input_energy.setMaximumWidth(45)  # Width for ~4 digits
        self.input_energy.setToolTip("Photon energy in eV (e.g., Cu Ka = 8050 eV)")
        self.input_energy.returnPressed.connect(self._on_energy_changed)
        control_layout.addWidget(self.input_energy)

        control_layout.addSpacing(10)

        # Coordinates display label
        self.label_coords = QLabel("X: - | Y: -")
        self.label_coords.setStyleSheet("color: gray; font-size: 9pt;")
        # Set max width to prevent excessive expansion, but allow shrinking
        self.label_coords.setMaximumWidth(200)
        self.label_coords.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        control_layout.addWidget(self.label_coords)

        # Add stretch at the end to push everything to the left
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        # Set size policy to allow plot to expand and fill available space
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_widget.setLabel("bottom", "Index")
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Apply default dark theme
        self.plot_widget.setBackground('k')  # Black background

        # Set axis colors to white for dark theme
        axis_pen = pg.mkPen(color='w', width=1)
        for axis in ['left', 'bottom', 'right', 'top']:
            self.plot_widget.getAxis(axis).setPen(axis_pen)
            self.plot_widget.getAxis(axis).setTextPen(axis_pen)

        # Disable right-click menu for consistent UI
        self.plot_widget.plotItem.vb.setMenuEnabled(False)

        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def set_data(self, y_data: np.ndarray, x_data: Optional[np.ndarray] = None) -> None:
        """
        Set the data to plot.

        Args:
            y_data: Y-axis data (1D array or 2D array with multiple columns)
            x_data: Optional X-axis data (1D array). If None, uses indices.
        """
        self.y_data = y_data
        y_len = len(y_data)

        if x_data is not None:
            # Explicit X provided by caller: use it if compatible; otherwise reset.
            if len(x_data) == y_len:
                self.x_data = np.asarray(x_data)
                if self.chk_convert_to_q.isChecked():
                    self.x_data_original = np.asarray(x_data).copy()
                self.chk_convert_to_q.setEnabled(True)
            else:
                logging.warning(
                    "Ignoring provided X data due to length mismatch: len(X)=%s, len(Y)=%s",
                    len(x_data),
                    y_len,
                )
                self.x_data = None
                self.x_data_original = None
                self.x_dataset_path = None
                self.chk_convert_to_q.setEnabled(False)
                self.chk_convert_to_q.setChecked(False)
        else:
            # No explicit X passed: preserve currently selected X when dimensions match.
            if self.x_data is not None and len(self.x_data) == y_len:
                self.chk_convert_to_q.setEnabled(True)
            else:
                if self.x_data is not None:
                    logging.info(
                        "Resetting custom X due to Y size mismatch: len(X)=%s, len(Y)=%s",
                        len(self.x_data),
                        y_len,
                    )
                self.x_data = None
                self.x_data_original = None
                self.x_dataset_path = None
                self.chk_convert_to_q.setEnabled(False)
                self.chk_convert_to_q.setChecked(False)

        self._update_plot()

    def copy_plot_screenshot_to_clipboard(self) -> None:
        """Copy the current plot widget as a screenshot."""
        try:
            app = QApplication.instance()
            if app is None:
                return
            pm = self.plot_widget.grab()
            if pm is None or pm.isNull():
                return

            jpeg_bytes = QByteArray()
            buf = QBuffer(jpeg_bytes)
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            pm.toImage().save(buf, "JPEG", 95)
            buf.close()

            mime = QMimeData()
            mime.setImageData(pm.toImage())
            if not jpeg_bytes.isEmpty():
                mime.setData("image/jpeg", jpeg_bytes)
            app.clipboard().setMimeData(mime)
            logging.info("Copied plot screenshot to clipboard")
        except Exception as exc:
            logging.error("Failed to copy plot screenshot: %s", exc)

    def _request_main_export(self) -> None:
        """Ask the main window to run the existing Export action."""
        win = self.window()
        handler = getattr(win, "_handle_action_export_current", None)
        if callable(handler):
            handler()
        else:
            logging.warning("Main export handler not available from plot viewer")

    def set_source_dataset_key(self, full_key: str | None) -> None:
        """Set source full key for Y data (used by legend labels)."""
        self.y_source_dataset_key = full_key

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

    def _update_plot(self) -> None:
        """Update the plot with current data and settings."""
        # Clear selection when plot is updated
        self._clear_selection()

        self.plot_widget.clear()

        if self.y_data is None:
            return

        # Determine X data and label
        if self.x_data is not None:
            x = self.x_data
            # Set X-axis label based on q conversion state
            if self.chk_convert_to_q.isChecked():
                self.plot_widget.setLabel("bottom", "q (1/A)")
            else:
                short_x = self._short_key_label(self.x_dataset_path)
                self.plot_widget.setLabel("bottom", short_x or "Custom X")
        else:
            x = np.arange(len(self.y_data))
            self.plot_widget.setLabel("bottom", "Index")

        # Plot - handle both 1D and 2D (multi-column) data
        line_width = self.spinbox_linewidth.value()
        try:
            if self.y_data.ndim == 1:
                # Single curve
                pen = pg.mkPen(color='b', width=line_width)
                self.plot_widget.plot(x, self.y_data, pen=pen)
            elif self.y_data.ndim == 2:
                # Multiple curves (one per column)
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                num_cols = self.y_data.shape[1]

                # Add legend if multiple curves (positioned in top-right corner)
                if num_cols > 1:
                    self.plot_widget.addLegend(offset=(-10, 10))

                for col_idx in range(num_cols):
                    color = colors[col_idx % len(colors)]
                    pen = pg.mkPen(color=color, width=line_width)
                    base = self._short_key_label(self.y_source_dataset_key) or "Result"
                    self.plot_widget.plot(
                        x, self.y_data[:, col_idx],
                        pen=pen,
                        name=f"{base}::col{col_idx}"
                    )
        except Exception as e:
            logging.error(f"Failed to plot data: {e}")

    def _update_axis_scale(self) -> None:
        """Update axis scale (linear/log) based on checkboxes."""
        # Clear selection when changing axis scale to avoid incorrect marker position
        self._clear_selection()

        log_x = self.chk_log_x.isChecked()
        log_y = self.chk_log_y.isChecked()

        try:
            self.plot_widget.setLogMode(x=log_x, y=log_y)
        except Exception as e:
            logging.error(f"Failed to set log mode: {e}")
            QMessageBox.warning(
                self,
                "Log Scale Error",
                f"Failed to set log scale:\n{e}\n\n"
                "Note: Log scale requires positive values."
            )
            # Reset checkboxes
            self.chk_log_x.setChecked(False)
            self.chk_log_y.setChecked(False)

    def _convert_angle_to_q(self, angle_deg: float) -> float:
        """
        Convert angle (in degrees) to momentum transfer q (in 1/A).

        Formula: q = (4*pi/lambda) * sin(theta)
        Where: lambda(A) = 12398 / E(eV)

        Args:
            angle_deg: Angle in degrees

        Returns:
            Momentum transfer q in 1/A
        """
        try:
            energy_text = self.input_energy.text().strip()
            if not energy_text:
                energy_ev = 8050  # Default Cu Ka energy
            else:
                energy_ev = float(energy_text)

            # Convert energy (eV) to wavelength (A)
            # E(eV) = 12398 / lambda(A)  =>  lambda(A) = 12398 / E(eV)
            wavelength = 12398 / energy_ev

            import math
            angle_rad = math.radians(angle_deg)
            q = (4 * math.pi / wavelength) * math.sin(angle_rad)
            return q
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _on_q_conversion_changed(self, state: int) -> None:
        """Handle X to q conversion checkbox state change."""
        if self.x_data is None:
            return

        if state:  # Checked - convert X axis to q
            # Check if energy value is provided
            energy_text = self.input_energy.text().strip()
            if not energy_text:
                # No energy provided, wait for user input
                logging.info("X to q conversion enabled, waiting for energy input")
                return

            # Save original X data
            if self.x_data_original is None:
                self.x_data_original = self.x_data.copy()

            # Convert all X values to q
            self.x_data = np.array([self._convert_angle_to_q(x) for x in self.x_data_original])
            logging.info("Converted X-axis from angle to q")
            # Update the plot
            self._update_plot()
        else:  # Unchecked - restore original X axis
            if self.x_data_original is not None:
                self.x_data = self.x_data_original.copy()
                logging.info("Restored original X-axis (angle)")
                # Update the plot
                self._update_plot()

    def _on_energy_changed(self) -> None:
        """Handle energy input change (Enter pressed)."""
        # Only apply conversion if checkbox is checked and X data exists
        if not self.chk_convert_to_q.isChecked() or self.x_data is None:
            return

        energy_text = self.input_energy.text().strip()
        if not energy_text:
            logging.warning("No energy value provided")
            return

        # Ensure original X data is saved
        if self.x_data_original is None:
            self.x_data_original = self.x_data.copy()

        # Convert all X values to q with new energy
        self.x_data = np.array([self._convert_angle_to_q(x) for x in self.x_data_original])
        logging.info(f"Updated X to q conversion with energy: {energy_text} eV")
        # Update the plot
        self._update_plot()

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate distance between two points, considering log scale.

        Args:
            x1, y1: First point coordinates
            x2, y2: Second point coordinates

        Returns:
            Distance in the appropriate space (linear or log)
        """
        log_x = self.chk_log_x.isChecked()
        log_y = self.chk_log_y.isChecked()

        # Convert to log space if needed
        if log_x:
            # For log scale, use log10 of positive values
            if x1 > 0 and x2 > 0:
                x1 = np.log10(x1)
                x2 = np.log10(x2)
            else:
                # If values are not positive, can't use log - return large distance
                return float('inf')

        if log_y:
            if y1 > 0 and y2 > 0:
                y1 = np.log10(y1)
                y2 = np.log10(y2)
            else:
                # If values are not positive, can't use log - return large distance
                return float('inf')

        # Calculate Euclidean distance
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def _on_mouse_moved(self, pos) -> None:
        """Handle mouse movement to display coordinates."""
        # If a point is selected (locked), don't update the coordinates
        if self.selected_point is not None:
            return

        if self.y_data is None:
            return

        # Get mouse position in view coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_mouse = mouse_point.x()

        # Determine X data array
        if self.x_data is not None:
            x = self.x_data
        else:
            x = np.arange(len(self.y_data))

        # Find the closest data point
        try:
            # Handle 1D and 2D data
            if self.y_data.ndim == 1:
                # For 1D data, find closest X index
                idx = np.argmin(np.abs(x - x_mouse))
                x_val = x[idx]
                y_val = self.y_data[idx]

                # Display with optional q conversion
                if self.chk_convert_to_q.isChecked():
                    q_value = self._convert_angle_to_q(x_val)
                    self.label_coords.setText(f"X: {x_val:.3f} | q: {q_value:.4f} 1/A | Y: {y_val:.3g}")
                else:
                    self.label_coords.setText(f"X: {x_val:.3f} | Y: {y_val:.3g}")
            elif self.y_data.ndim == 2:
                # For 2D data (multiple curves), find closest point across all curves
                y_mouse = mouse_point.y()
                min_distance = float('inf')
                closest_x = x_mouse
                closest_y = y_mouse
                closest_curve = 0

                # Search through all curves
                num_cols = self.y_data.shape[1]
                for col_idx in range(num_cols):
                    # Find closest X index in this curve
                    idx = np.argmin(np.abs(x - x_mouse))
                    x_val = x[idx]
                    y_val = self.y_data[idx, col_idx]

                    # Calculate distance considering log scale
                    distance = self._calculate_distance(x_val, y_val, x_mouse, y_mouse)

                    if distance < min_distance:
                        min_distance = distance
                        closest_x = x_val
                        closest_y = y_val
                        closest_curve = col_idx

                # Display with optional q conversion
                if self.chk_convert_to_q.isChecked():
                    q_value = self._convert_angle_to_q(closest_x)
                    self.label_coords.setText(f"Curve {closest_curve + 1} | X: {closest_x:.3f} | q: {q_value:.4f} 1/A | Y: {closest_y:.3g}")
                else:
                    self.label_coords.setText(f"Curve {closest_curve + 1} | X: {closest_x:.3f} | Y: {closest_y:.3g}")
        except Exception as e:
            # Fallback to mouse position
            self.label_coords.setText(f"X: {x_mouse:.3f} | Y: {mouse_point.y():.3g}")

    def _on_mouse_clicked(self, event) -> None:
        """Handle mouse click to select and lock a data point."""
        logging.info("Mouse clicked event triggered")

        if self.y_data is None:
            return

        # Right click: clear selection
        if event.button() == Qt.MouseButton.RightButton:
            self._clear_selection()
            return

        # Only handle left click
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Get click position in view coordinates
        try:
            scene_pos = event.scenePos()
            view_pos = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos)
            x_click = view_pos.x()
            y_click = view_pos.y()


            # In log mode, mapSceneToView returns log space coordinates
            # Convert back to linear space for distance calculation
            log_x = self.chk_log_x.isChecked()
            log_y = self.chk_log_y.isChecked()

            if log_x:
                x_click = 10 ** x_click  # Convert from log to linear
            if log_y:
                y_click = 10 ** y_click  # Convert from log to linear (e.g., 10^-8.84 = 1.45e-9)

        except Exception as e:
            logging.error(f"Error getting click position: {e}")
            return

        # Determine X data array
        if self.x_data is not None:
            x = self.x_data
        else:
            x = np.arange(len(self.y_data))

        # Find the closest data point to the click
        try:
            if self.y_data.ndim == 1:
                # Single curve - use efficient search
                # Step 1: Find closest X index
                if self.chk_log_x.isChecked() and np.all(x > 0):
                    # For log X, compare in log space
                    x_distances = np.abs(np.log10(x) - np.log10(x_click))
                else:
                    x_distances = np.abs(x - x_click)

                center_idx = np.argmin(x_distances)

                # Step 2: Check nearby points for best match
                # Use larger window for log scale since points can be visually far apart
                window = 50 if self.chk_log_y.isChecked() else 20
                start_idx = max(0, center_idx - window)
                end_idx = min(len(x), center_idx + window + 1)


                min_distance = float('inf')
                best_idx = center_idx

                # First try window search
                for idx in range(start_idx, end_idx):
                    x_val = x[idx]
                    y_val = self.y_data[idx]

                    # Calculate distance considering log scale
                    distance = self._calculate_distance(x_val, y_val, x_click, y_click)

                    if distance < min_distance:
                        min_distance = distance
                        best_idx = idx

                # If window search failed (all distances were inf), do global search
                if min_distance == float('inf'):
                    for idx in range(len(x)):
                        x_val = x[idx]
                        y_val = self.y_data[idx]

                        distance = self._calculate_distance(x_val, y_val, x_click, y_click)

                        if distance < min_distance:
                            min_distance = distance
                            best_idx = idx

                x_val = x[best_idx]
                y_val = self.y_data[best_idx]
                curve_idx = None


                # Store selected point (x, y, curve_idx)
                self.selected_point = (x_val, y_val, curve_idx)

                # Update label with optional q conversion
                if self.chk_convert_to_q.isChecked():
                    q_value = self._convert_angle_to_q(x_val)
                    label_text = f"X: {x_val:.3f} | q: {q_value:.4f} 1/A | Y: {y_val:.3g}"
                else:
                    label_text = f"X: {x_val:.3f} | Y: {y_val:.3g}"

            elif self.y_data.ndim == 2:
                # Multiple curves - efficient search
                # Step 1: Find closest X index (same for all curves)
                if self.chk_log_x.isChecked() and np.all(x > 0):
                    x_distances = np.abs(np.log10(x) - np.log10(x_click))
                else:
                    x_distances = np.abs(x - x_click)

                center_idx = np.argmin(x_distances)

                # Step 2: Check nearby points in all curves
                # Use larger window for log scale since points can be visually far apart
                window = 50 if self.chk_log_y.isChecked() else 20
                start_idx = max(0, center_idx - window)
                end_idx = min(len(x), center_idx + window + 1)


                min_distance = float('inf')
                best_x = x_click
                best_y = y_click
                best_idx = center_idx
                best_curve = 0

                num_cols = self.y_data.shape[1]

                # First try window search
                for col_idx in range(num_cols):
                    for idx in range(start_idx, end_idx):
                        x_val = x[idx]
                        y_val = self.y_data[idx, col_idx]

                        # Calculate distance considering log scale
                        distance = self._calculate_distance(x_val, y_val, x_click, y_click)

                        if distance < min_distance:
                            min_distance = distance
                            best_x = x_val
                            best_y = y_val
                            best_idx = idx
                            best_curve = col_idx

                # If window search failed (all distances were inf), do global search
                if min_distance == float('inf'):
                    for col_idx in range(num_cols):
                        for idx in range(len(x)):
                            x_val = x[idx]
                            y_val = self.y_data[idx, col_idx]

                            distance = self._calculate_distance(x_val, y_val, x_click, y_click)

                            if distance < min_distance:
                                min_distance = distance
                                best_x = x_val
                                best_y = y_val
                                best_idx = idx
                                best_curve = col_idx


                # Store selected point (x, y, curve_idx)
                self.selected_point = (best_x, best_y, best_curve)

                # Update label with curve number and optional q conversion
                if self.chk_convert_to_q.isChecked():
                    q_value = self._convert_angle_to_q(best_x)
                    label_text = f"Curve {best_curve + 1} | X: {best_x:.3f} | q: {q_value:.4f} 1/A | Y: {best_y:.3g}"
                else:
                    label_text = f"Curve {best_curve + 1} | X: {best_x:.3f} | Y: {best_y:.3g}"

                x_val, y_val = best_x, best_y

            else:
                logging.warning("Unsupported data dimensions for point selection: %s", self.y_data.ndim)
                return

            # Check if a valid point was found
            if min_distance == float('inf'):
                logging.warning(f"Could not find valid data point in log mode. Data may have non-positive values.")
                return

            # Remove old marker if exists
            if self.selected_marker is not None:
                self.plot_widget.removeItem(self.selected_marker)

            # Create circle marker at selected point
            # In log mode, use log space coordinates
            marker_x = np.log10(x_val) if (self.chk_log_x.isChecked() and x_val > 0) else x_val
            marker_y = np.log10(y_val) if (self.chk_log_y.isChecked() and y_val > 0) else y_val


            # Use same line width as plot curves for the marker border
            marker_width = self.spinbox_linewidth.value()
            # Size scales with line width (70% of original size)
            marker_size = (8 + marker_width * 2) * 0.7

            self.selected_marker = pg.ScatterPlotItem(
                [marker_x], [marker_y],
                size=marker_size,
                pen=pg.mkPen('orange', width=marker_width),  # Orange border
                brush=pg.mkBrush('orange'),  # Orange fill
                symbol='o'  # Circle symbol
            )
            self.plot_widget.addItem(self.selected_marker)

            # Update label
            self.label_coords.setText(label_text)
            self.label_coords.setStyleSheet("color: blue; font-size: 9pt; font-weight: bold;")
            logging.info(f"Selected point: {label_text}")

        except Exception as e:
            logging.error(f"Error selecting point: {e}")

    def _clear_selection(self) -> None:
        """Clear the selected point and marker."""
        self.selected_point = None

        # Remove marker
        if self.selected_marker is not None:
            self.plot_widget.removeItem(self.selected_marker)
            self.selected_marker = None

        # Reset label style
        self.label_coords.setText("X: - | Y: -")
        self.label_coords.setStyleSheet("color: gray; font-size: 9pt;")

    def _select_custom_x(self) -> None:
        """Open dialog to select custom X data."""
        dialog = XDataSelectionDialog(
            self.opened_files,
            self,
            self.y_data,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
        )
        dialog.data_selected.connect(self._on_x_data_selected)
        dialog.show()

    def _on_x_data_selected(self, x_data: np.ndarray, x_path: str) -> None:
        """Handle X data selection from dialog."""
        if len(x_data) != len(self.y_data):
            QMessageBox.warning(
                self,
                "Data Size Mismatch",
                f"X data length ({len(x_data)}) does not match Y data length ({len(self.y_data)}).\n\n"
                "X and Y data must have the same length."
            )
            return

        # Enable X to q conversion checkbox now that X data is available
        self.chk_convert_to_q.setEnabled(True)

        # Check if X to q conversion is currently enabled
        if self.chk_convert_to_q.isChecked():
            # Store the new custom data as original
            self.x_data_original = x_data.copy()
            # Apply q conversion to get display data
            self.x_data = np.array([self._convert_angle_to_q(x) for x in x_data])
            logging.info(f"Set custom X data and applied q conversion: {x_path}")
        else:
            # No conversion, just set the data
            self.x_data = x_data
            logging.info(f"Set custom X data: {x_path}")

        self.x_dataset_path = x_path
        self._update_plot()

    def _export_to_csv(self) -> None:
        """Export plot data to CSV or TXT file with X and Y columns."""
        if self.y_data is None:
            QMessageBox.information(
                self,
                "No Data",
                "No data to export. Please load data first."
            )
            return

        # Open file save dialog
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import QFileDialog

        settings = QSettings()
        last_dir = settings.value("paths/last_export_directory", pathlib.Path.home())

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot Data",
            str(pathlib.Path(last_dir) / "plot_data.csv"),
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )

        if not file_path:
            return  # User cancelled

        # Save directory for next time
        settings.setValue("paths/last_export_directory", pathlib.Path(file_path).parent)

        try:
            import csv

            # Determine delimiter based on file extension
            file_ext = pathlib.Path(file_path).suffix.lower()
            delimiter = "\t" if file_ext == ".txt" else ","

            # Determine X values - use original if q conversion is enabled
            x_original = None
            q_values = None

            if self.chk_convert_to_q.isChecked() and self.x_data_original is not None:
                # When q conversion is enabled, export both original X and converted q
                x_original = self.x_data_original
                q_values = self.x_data
            elif self.x_data is not None:
                # No conversion, just use x_data
                x_original = self.x_data
            else:
                # No custom X data, use indices
                x_original = np.arange(len(self.y_data))

            # Write data file with UTF-8 BOM for better Excel compatibility
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=delimiter)

                # Handle 1D and 2D data differently
                if self.y_data.ndim == 1:
                    # Single column Y data
                    # Write header
                    if q_values is not None:
                        # Export both X and q columns
                        writer.writerow(["X", "q", "Y"])
                    else:
                        writer.writerow(["X", "Y"])

                    # Write data
                    if q_values is not None:
                        for x_val, q_val, y_val in zip(x_original, q_values, self.y_data):
                            writer.writerow([f"{x_val:.10g}", f"{q_val:.10g}", f"{y_val:.10g}"])
                    else:
                        for x_val, y_val in zip(x_original, self.y_data):
                            writer.writerow([f"{x_val:.10g}", f"{y_val:.10g}"])

                elif self.y_data.ndim == 2:
                    # Multiple columns Y data
                    num_cols = self.y_data.shape[1]

                    # Write header
                    if q_values is not None:
                        # Export both X and q columns
                        header = ["X", "q"]
                    else:
                        header = ["X"]

                    for col_idx in range(num_cols):
                        header.append(f"Y_Column_{col_idx + 1}")
                    writer.writerow(header)

                    # Write data
                    for row_idx in range(len(self.y_data)):
                        if q_values is not None:
                            row = [f"{x_original[row_idx]:.10g}", f"{q_values[row_idx]:.10g}"]
                        else:
                            row = [f"{x_original[row_idx]:.10g}"]

                        for col_idx in range(num_cols):
                            y_val = self.y_data[row_idx, col_idx]
                            row.append(f"{y_val:.10g}")
                        writer.writerow(row)

            logging.info(f"Exported plot data to: {file_path}")
            QMessageBox.information(
                self,
                "Export Successful",
                f"Plot data exported successfully to:\n{file_path}"
            )

        except Exception as e:
            logging.error(f"Failed to export plot data: {e}")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export data:\n{str(e)}"
            )


class XDataSelectionDialog(QDialog):
    """Dialog for selecting custom X-axis data."""

    # Signal emitted when data is selected (data, path)
    data_selected = pyqtSignal(np.ndarray, str)

    def __init__(
        self,
        opened_files: tuple[pathlib.Path, ...],
        parent=None,
        y_data=None,
        dataset_full_keys_1d: list[str] | None = None,
    ):
        """
        Initialize the dialog.

        Args:
            opened_files: Tuple of opened HDF5 file paths
            parent: Parent widget
            y_data: Y-axis data for length validation (optional)
        """
        super().__init__(parent)
        self.opened_files = opened_files
        self.y_data = y_data
        self.dataset_full_keys_1d = dataset_full_keys_1d or []
        self.selected_data = None
        self.selected_path = None
        self.dragged_path = None  # Store path from drag-drop

        # Make dialog non-modal but stay on top
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self._init_ui()
        self._populate_datasets()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Select X-Axis Data")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            "<b>Select X-Axis Data</b><br>"
            "The selected dataset must have the same length as the Y-axis data.<br>"
            "You can either <b>drag and drop</b> a dataset or select from the list below."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Drag and drop area
        self.drop_label = QLabel("Drag and Drop Dataset Here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setMinimumHeight(80)
        self.drop_label.setStyleSheet(
            "QLabel {"
            "   border: 2px dashed #999;"
            "   border-radius: 5px;"
            "   background-color: #f0f0f0;"
            "   color: #666;"
            "   font-size: 14px;"
            "}"
        )
        self.drop_label.setAcceptDrops(True)
        self.drop_label.dragEnterEvent = self._drag_enter_event
        self.drop_label.dropEvent = self._drop_event
        layout.addWidget(self.drop_label)

        # Or separator
        or_label = QLabel("- OR -")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label.setStyleSheet("color: #999; margin: 10px;")
        layout.addWidget(or_label)

        # Dataset selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select from list:"))

        self.combo_datasets = QComboBox()
        self.combo_datasets.setMinimumWidth(350)
        selector_layout.addWidget(self.combo_datasets)

        layout.addLayout(selector_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        button_layout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close)
        button_layout.addWidget(self.btn_cancel)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_ok_clicked(self) -> None:
        """Handle OK button click."""
        x_data, x_path = self.get_selected_data()
        if x_data is not None and x_path is not None:
            # Emit signal with the selected data
            self.data_selected.emit(x_data, x_path)
            self.close()
        else:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a dataset or drag and drop one."
            )

    def _drag_enter_event(self, event: QDragEnterEvent | None) -> None:
        """Handle drag enter events."""
        if event is None:
            return
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet(
                "QLabel {"
                "   border: 2px dashed #4CAF50;"
                "   border-radius: 5px;"
                "   background-color: #e8f5e9;"
                "   color: #2E7D32;"
                "   font-size: 14px;"
                "}"
            )

    def _drop_event(self, event: QDropEvent | None) -> None:
        """Handle drop events."""
        # Reset style
        self.drop_label.setStyleSheet(
            "QLabel {"
            "   border: 2px dashed #999;"
            "   border-radius: 5px;"
            "   background-color: #f0f0f0;"
            "   color: #666;"
            "   font-size: 14px;"
            "}"
        )

        if event is None:
            return
        if not event.mimeData().hasText():
            return

        # Get dropped text (dataset path)
        dropped_text = event.mimeData().text().strip()
        logging.info(f"Dropped dataset: '{dropped_text}'")

        # Validate the path format
        if "::" not in dropped_text:
            QMessageBox.warning(
                self,
                "Invalid Format",
                f"Invalid dataset path format:\n{dropped_text}\n\n"
                "Expected format: filename.ext::path/to/dataset"
            )
            return

        # Store the dragged path
        self.dragged_path = dropped_text

        # Update drop label to show selected dataset
        self.drop_label.setText(f"Selected: {dropped_text}")
        self.drop_label.setStyleSheet(
            "QLabel {"
            "   border: 2px solid #4CAF50;"
            "   border-radius: 5px;"
            "   background-color: #e8f5e9;"
            "   color: #2E7D32;"
            "   font-size: 12px;"
            "   padding: 10px;"
            "}"
        )

        event.acceptProposedAction()

    def _populate_datasets(self) -> None:
        """Populate the dataset selector from shared 1D index only."""
        if self.dataset_full_keys_1d:
            for full_key in self.dataset_full_keys_1d:
                if "::" not in full_key:
                    continue
                filename, item_path = full_key.split("::", 1)
                file_name = pathlib.Path(filename).name
                display_text = f"{file_name}::{item_path}"
                # Keep full key in userData to avoid ambiguity for duplicate file names.
                self.combo_datasets.addItem(display_text, full_key)
            return
        logging.info("XDataSelectionDialog: shared index is empty, waiting for index warm-up")

    def _add_datasets_recursive(self, group: h5py.Group, filename: str, path: str) -> None:
        """
        Recursively add datasets to the combo box.

        Args:
            group: HDF5 group
            filename: File name
            path: Current path in the file
        """
        for key in group.keys():
            item = group[key]
            item_path = f"{path}/{key}" if path else key

            if isinstance(item, h5py.Dataset):
                # Only add 1D datasets
                if item.ndim == 1:
                    display_text = f"{filename}::{item_path} ({len(item)} points)"
                    data_path = f"{filename}::{item_path}"
                    self.combo_datasets.addItem(display_text, data_path)
            elif isinstance(item, h5py.Group):
                self._add_datasets_recursive(item, filename, item_path)

    def get_selected_data(self) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get the selected data.

        Returns:
            Tuple of (data array, dataset path) or (None, None) if selection failed
        """
        # Priority 1: Use dragged path if available
        if self.dragged_path:
            data_path = self.dragged_path
        # Priority 2: Use combo box selection
        elif self.combo_datasets.currentIndex() >= 0:
            data_path = self.combo_datasets.currentData()
        else:
            return None, None

        if not data_path or "::" not in data_path:
            return None, None

        # Parse path
        parts = data_path.split("::", 1)
        file_token = parts[0]
        h5_path = parts[1]

        # Find file
        file_path = None
        for opened_file in self.opened_files:
            if str(opened_file) == file_token or opened_file.name == file_token or str(opened_file).endswith(file_token):
                file_path = opened_file
                break

        if file_path is None:
            QMessageBox.warning(
                self,
                "File Not Found",
                f"File not found: {file_token}"
            )
            return None, None

        # Load data
        try:
            with h5py.File(file_path, "r") as h5file:
                if h5_path not in h5file:
                    QMessageBox.warning(
                        self,
                        "Dataset Not Found",
                        f"Dataset not found: {h5_path}"
                    )
                    return None, None

                data = h5file[h5_path][:]

                if data.ndim != 1:
                    QMessageBox.warning(
                        self,
                        "Invalid Dataset",
                        f"Dataset is not 1D: {data.shape}"
                    )
                    return None, None

                return data, data_path

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Data",
                f"Failed to load dataset:\n{e}"
            )
            return None, None





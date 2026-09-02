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

import logging
import pathlib
from typing import Any, Optional

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QBuffer, QByteArray, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gui._shared import (
    AXIS_Q,
    TOOLBAR_ITEM_GAP,
    labelled,
    quick_icon_button,
    set_axis_label,
    toolbar_group_gap,
)
from src.gui.plot_context_menu import attach_plot_menu
from src.gui.export_naming import (
    export_stem,
    remember_save_directory,
    short_series_label,
    suggested_save_path,
    x_column_header,
)
from src.lib_h5.table_format import (
    DEFAULT_TABLE_FORMAT_KEY,
    format_from_filter,
    get_table_format,
    save_dialog_filter,
)
from src.lib_h5.table_writer import write_table

# Photon energy for the angle -> q conversion. The default is Cu Ka; the bounds
# only exist to keep the input dialog from accepting a value that would divide
# the wavelength by zero or overflow it.
DEFAULT_PHOTON_ENERGY_EV = 8050.0
MIN_PHOTON_ENERGY_EV = 1.0
MAX_PHOTON_ENERGY_EV = 1_000_000.0

# A toggle that is on has to look on. Qt's own checked state is a slight change
# of shade that is easy to miss, and this one changes the numbers. The green
# matches the folder-monitor button in the main window; the ink is dark because
# the white that button uses is barely readable on it.
ACTIVE_BUTTON_STYLE = "QPushButton:checked { background-color: #9AEBA3; color: #14311a; font-weight: bold; }"


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
        # Asked for when "X to q" is switched on, and kept for the next time.
        self.photon_energy_ev = DEFAULT_PHOTON_ENERGY_EV
        # Despiking. The settings are remembered between uses, but the result is
        # kept apart from y_data: the raw values stay exactly as loaded, so the
        # filter is undone by putting these back to None and nothing else.
        self.despike_settings: dict | None = None
        self._despiked_y: np.ndarray | None = None
        self._spike_mask: np.ndarray | None = None
        self._despike_log_space = False

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

        # Control panel at top. Gaps come from _shared so the toolbar reads as
        # groups: label-to-control tightest, then within a tool, then between
        # tools. See toolbar_group_gap for why the spacer is not the full width.
        control_layout = QHBoxLayout()
        control_layout.setSpacing(TOOLBAR_ITEM_GAP)

        self.btn_copy_plot = quick_icon_button("copy.ico", "Quick copy: put the plot on the clipboard")
        self.btn_copy_plot.clicked.connect(self.quick_copy)
        control_layout.addWidget(self.btn_copy_plot)

        self.btn_save_plot = quick_icon_button("save.ico", "Quick export: save the X/Y data as displayed")
        self.btn_save_plot.clicked.connect(self.quick_export)
        control_layout.addWidget(self.btn_save_plot)

        self.btn_plot = quick_icon_button("plot.ico", "Plot: draw the displayed curve in a Plot window")
        self.btn_plot.clicked.connect(self.open_plot)
        control_layout.addWidget(self.btn_plot)

        toolbar_group_gap(control_layout)

        # Axis scale controls
        self.chk_log_x = QCheckBox("Log X")
        self.chk_log_x.stateChanged.connect(self._update_axis_scale)
        control_layout.addLayout(labelled("Axis Scale:", self.chk_log_x))

        self.chk_log_y = QCheckBox("Log Y")
        self.chk_log_y.stateChanged.connect(self._update_axis_scale)
        control_layout.addWidget(self.chk_log_y)

        toolbar_group_gap(control_layout)

        # Custom X data control: one self-explanatory button, matching the tree's
        # "Set X" rather than a label plus a bare "Select".
        if self.opened_files:
            self.btn_select_x = QPushButton("Set X")
            self.btn_select_x.setAutoDefault(False)  # Prevent Enter key from triggering this button
            self.btn_select_x.setToolTip("Choose the dataset used as the X axis")
            # Set maximum width to allow toolbar to compress when window is resized
            self.btn_select_x.setMinimumWidth(0)  # Allow shrinking
            self.btn_select_x.setMaximumWidth(60)
            self.btn_select_x.clicked.connect(self._select_custom_x)
            control_layout.addWidget(self.btn_select_x)

        # Q conversion for scattering experiments. One button next to Set X:
        # it asks for the photon energy when switched on, which is the only
        # moment that number matters, instead of parking a box in the toolbar.
        self.btn_convert_to_q = QPushButton("X to q")
        self.btn_convert_to_q.setCheckable(True)
        self.btn_convert_to_q.setAutoDefault(False)
        self.btn_convert_to_q.setMinimumWidth(0)
        self.btn_convert_to_q.setMaximumWidth(60)
        self.btn_convert_to_q.setToolTip(
            "Convert the X axis from angle to momentum transfer q; asks for the photon energy"
        )
        self.btn_convert_to_q.setEnabled(False)  # Disabled until X data is loaded
        self.btn_convert_to_q.clicked.connect(self._on_q_conversion_clicked)
        control_layout.addWidget(self.btn_convert_to_q)

        toolbar_group_gap(control_layout)

        # Despiking, on the same pattern as X to q: the button asks for its
        # parameters when switched on, and switching it off puts the raw values
        # back. It only ever affects what is drawn and exported, never the file.
        self.btn_despike = QPushButton("Despike")
        self.btn_despike.setCheckable(True)
        self.btn_despike.setAutoDefault(False)
        self.btn_despike.setMinimumWidth(0)
        self.btn_despike.setMaximumWidth(70)
        self.btn_despike.setToolTip(
            "Remove single-point glitches from the displayed curve; asks for the settings"
        )
        self.btn_despike.setStyleSheet(ACTIVE_BUTTON_STYLE)
        self.btn_despike.setEnabled(False)  # Disabled until data is loaded
        self.btn_despike.clicked.connect(self._on_despike_clicked)
        control_layout.addWidget(self.btn_despike)

        # A filter that silently changed the numbers would be the worst version
        # of this feature, so the count is on screen whenever it is on.
        self.label_despike = QLabel("")
        self.label_despike.setStyleSheet("color: #2e7d32; font-size: 9pt;")
        self.label_despike.setVisible(False)
        control_layout.addWidget(self.label_despike)

        toolbar_group_gap(control_layout)

        # Line width control
        self.spinbox_linewidth = QSpinBox()
        self.spinbox_linewidth.setMinimum(1)
        self.spinbox_linewidth.setMaximum(10)
        self.spinbox_linewidth.setValue(3)  # Default line width
        self.spinbox_linewidth.setSuffix(" px")
        self.spinbox_linewidth.setToolTip("Line width (press Enter to apply)")
        self.spinbox_linewidth.editingFinished.connect(self._update_plot)  # Only trigger on Enter or focus loss
        control_layout.addLayout(labelled("Line Width:", self.spinbox_linewidth))

        toolbar_group_gap(control_layout)

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

        # Right-click reaches the same two things as the toolbar icons, for the
        # people who look for them there. It acts on the curve that was clicked,
        # which in an embedded viewer — the calculator's result, say — is that
        # curve and not the window's wider idea of an export.
        attach_plot_menu(
            self.plot_widget,
            on_export=self.quick_export,
            on_plot=self.open_plot,
        )

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

        # The filter stays on across a change of dataset, the way X to q does,
        # but its result belongs to the old array — so recompute it rather than
        # letting a stale mask sit over the new curve.
        self.btn_despike.setEnabled(y_data is not None)
        self._despiked_y = None
        self._spike_mask = None
        despike_was_on = self.btn_despike.isChecked()

        if x_data is not None:
            # Explicit X provided by caller: use it if compatible; otherwise reset.
            if len(x_data) == y_len:
                self.x_data = np.asarray(x_data)
                if self.btn_convert_to_q.isChecked():
                    self.x_data_original = np.asarray(x_data).copy()
                self.btn_convert_to_q.setEnabled(True)
            else:
                logging.warning(
                    "Ignoring provided X data due to length mismatch: len(X)=%s, len(Y)=%s",
                    len(x_data),
                    y_len,
                )
                self.x_data = None
                self.x_data_original = None
                self.x_dataset_path = None
                self.btn_convert_to_q.setEnabled(False)
                self.btn_convert_to_q.setChecked(False)
        else:
            # No explicit X passed: preserve currently selected X when dimensions match.
            if self.x_data is not None and len(self.x_data) == y_len:
                self.btn_convert_to_q.setEnabled(True)
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
                self.btn_convert_to_q.setEnabled(False)
                self.btn_convert_to_q.setChecked(False)

        if despike_was_on and self.despike_settings:
            self.apply_despike()      # updates the plot itself
        else:
            self.clear_despike()
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

    def set_source_dataset_key(self, full_key: str | None) -> None:
        """Set source full key for Y data (used by legend labels and export names)."""
        self.y_source_dataset_key = full_key

    def quick_export(self) -> None:
        """Quick export: save the X/Y data as currently displayed."""
        self._export_to_csv()

    def quick_copy(self) -> None:
        """Quick copy: put a screenshot of the current plot on the clipboard."""
        self.copy_plot_screenshot_to_clipboard()

    def current_series(self) -> list:
        """The displayed curves as plot Series, one per column of Y.

        This is what the quick plot draws: the X and Y on screen, including a q
        conversion if one is active, and nothing that is not being shown.
        """
        from src.gui.plot_series import series_from_columns

        if self.y_data is None:
            return []
        label = self._short_key_label(self.y_source_dataset_key) or "Data"
        return series_from_columns(label, self.display_y, self.x_data)

    def open_plot(self) -> None:
        """Open a Plot window on the curves currently displayed."""
        from src.gui.plot_dialog import open_plot_dialog

        series = self.current_series()
        if not series:
            return

        dialog = open_plot_dialog(
            self,
            series,
            title=self._short_key_label(self.y_source_dataset_key) or "Data",
        )
        if dialog is None:
            return

        # Carry the viewer's own axis state over; the checkbox stays disabled if
        # the data cannot take a log axis, so this only ever relaxes to linear.
        dialog.le_x_label.setText(self.plot_widget.getAxis("bottom").labelText or "X")
        if self.chk_log_x.isChecked() and dialog.chk_log_x.isEnabled():
            dialog.chk_log_x.setChecked(True)
        if self.chk_log_y.isChecked() and dialog.chk_log_y.isEnabled():
            dialog.chk_log_y.setChecked(True)

    @staticmethod
    def _short_key_label(full_key: str | None) -> str:
        """Name a dataset the way every other label in the app does.

        See :mod:`src.gui.export_naming` — one convention for legends, axis
        labels and column headers, so the same curve reads the same everywhere.
        """
        if not full_key:
            return ""
        return short_series_label(full_key, fallback=str(full_key))

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
            if self.btn_convert_to_q.isChecked():
                set_axis_label(self.plot_widget, "bottom", AXIS_Q)
            else:
                short_x = self._short_key_label(self.x_dataset_path)
                set_axis_label(self.plot_widget, "bottom", short_x or "Custom X")
        else:
            x = np.arange(len(self.y_data))
            set_axis_label(self.plot_widget, "bottom", "Index")

        # Plot - handle both 1D and 2D (multi-column) data
        line_width = self.spinbox_linewidth.value()
        y_display = self.display_y
        try:
            if y_display.ndim == 1:
                # Single curve
                pen = pg.mkPen(color='b', width=line_width)
                self.plot_widget.plot(x, y_display, pen=pen)
            elif y_display.ndim == 2:
                # Multiple curves (one per column)
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                num_cols = y_display.shape[1]

                # Add legend if multiple curves (positioned in top-right corner)
                if num_cols > 1:
                    self.plot_widget.addLegend(offset=(-10, 10))

                for col_idx in range(num_cols):
                    color = colors[col_idx % len(colors)]
                    pen = pg.mkPen(color=color, width=line_width)
                    base = self._short_key_label(self.y_source_dataset_key) or "Result"
                    self.plot_widget.plot(
                        x, y_display[:, col_idx],
                        pen=pen,
                        name=f"{base}::col{col_idx}"
                    )
            self._mark_spikes(x)
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
            # Convert energy (eV) to wavelength (A)
            # E(eV) = 12398 / lambda(A)  =>  lambda(A) = 12398 / E(eV)
            wavelength = 12398 / self.photon_energy_ev

            import math
            angle_rad = math.radians(angle_deg)
            q = (4 * math.pi / wavelength) * math.sin(angle_rad)
            return q
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _on_q_conversion_clicked(self, turning_on: bool) -> None:
        """Switch the X axis between angle and q, asking for the energy first.

        Connected to ``clicked`` rather than ``toggled`` so that resetting the
        button in code — when the X data goes away, say — cannot pop a dialog.
        """
        if self.x_data is None:
            self.btn_convert_to_q.setChecked(False)
            return

        if not turning_on:
            if self.x_data_original is not None:
                self.x_data = self.x_data_original.copy()
                logging.info("Restored original X-axis (angle)")
                self._update_plot()
            return

        energy, accepted = QInputDialog.getDouble(
            self,
            "X to q",
            "Photon energy (eV):",
            self.photon_energy_ev,
            MIN_PHOTON_ENERGY_EV,
            MAX_PHOTON_ENERGY_EV,
            1,
        )
        if not accepted:
            # Nothing was converted, so the button must not look as if it was.
            self.btn_convert_to_q.setChecked(False)
            return

        self.photon_energy_ev = energy
        self.apply_q_conversion()

    # -- Despiking ---------------------------------------------------------

    @property
    def display_y(self) -> Optional[np.ndarray]:
        """The Y values in use: despiked when the filter is on, raw otherwise.

        Everything that shows or writes a number goes through here, so the plot,
        the readout, the quick export and the Plot window can never disagree
        about which version of the data they are looking at.
        """
        if self._despiked_y is not None:
            return self._despiked_y
        return self.y_data

    def _on_despike_clicked(self, turning_on: bool) -> None:
        """Switch the filter on via its parameter box, or off again.

        On ``clicked`` rather than ``toggled``, like X to q, so that clearing
        the button in code — when the data changes, say — cannot pop a dialog.
        """
        if self.y_data is None:
            self.btn_despike.setChecked(False)
            return

        if not turning_on:
            self.clear_despike()
            return

        from src.gui.despike_dialog import DespikeDialog

        # Blocking, like the energy prompt X to q puts up: it is one question
        # asked at the moment of switching on, not a window to work alongside.
        prompt = DespikeDialog(self.y_data, self, self.despike_settings)
        if prompt.exec() != QDialog.DialogCode.Accepted:
            # Nothing was filtered, so the button must not look as if it was.
            self.btn_despike.setChecked(False)
            return

        self.despike_settings = prompt.settings()
        self.apply_despike()

    def apply_despike(self) -> None:
        """Recompute the filtered curve from the current settings."""
        if self.y_data is None or not self.despike_settings:
            return

        from src.recon.despike import despike_table

        try:
            result = despike_table(self.y_data, **self.despike_settings)
        except Exception as exc:
            logging.error("Despike failed: %s", exc)
            QMessageBox.warning(self, "Despike Failed", f"Could not despike the data:\n{exc}")
            self.clear_despike()
            return

        self._despiked_y = result.values.reshape(np.shape(self.y_data))
        self._spike_mask = result.spikes.reshape(np.shape(self.y_data))
        self._despike_log_space = result.log_space
        self.btn_despike.setChecked(True)

        total = int(np.size(self.y_data))
        self.label_despike.setText(f"{result.count} / {total} despiked")
        self.label_despike.setToolTip(
            result.summary(
                self.despike_settings["method"],
                self.despike_settings["window"],
                self.despike_settings["threshold"],
            )
        )
        self.label_despike.setVisible(True)
        logging.info("Despiked %s of %s points", result.count, total)
        self._update_plot()

    def clear_despike(self) -> None:
        """Put the raw values back. The settings are kept for the next time."""
        had_filter = self._despiked_y is not None
        self._despiked_y = None
        self._spike_mask = None
        self.btn_despike.setChecked(False)
        self.label_despike.setVisible(False)
        self.label_despike.setText("")
        if had_filter:
            self._update_plot()

    def _despike_comment(self) -> list[str]:
        """The line an export carries so it is never mistaken for raw data."""
        if self._despiked_y is None or not self.despike_settings:
            return []
        count = int(np.count_nonzero(self._spike_mask))
        return [
            f"Despike: {self.despike_settings['method']}, "
            f"{'log' if self._despike_log_space else 'linear'} scale, "
            f"window={self.despike_settings['window']}, "
            f"threshold={self.despike_settings['threshold']:g} sigma, "
            f"direction={self.despike_settings['direction']}, "
            f"{count} point(s) "
            + ("replaced" if self.despike_settings["replace"] else "flagged only")
        ]

    def _mark_spikes(self, x: np.ndarray) -> None:
        """Ring the points the filter touched, so the change is visible.

        A filter whose only evidence is a smoother line is one nobody can
        check; these say exactly which samples were involved.
        """
        if self._spike_mask is None or not np.any(self._spike_mask):
            return

        mask = self._spike_mask
        y = np.asarray(self.display_y, dtype=float)
        columns = [(mask, y)] if y.ndim == 1 else [
            (mask[:, col], y[:, col]) for col in range(y.shape[1])
        ]

        marker_x: list[float] = []
        marker_y: list[float] = []
        for column_mask, column_y in columns:
            marker_x.extend(np.asarray(x, dtype=float)[column_mask].tolist())
            marker_y.extend(column_y[column_mask].tolist())
        if not marker_x:
            return

        # The plot draws in log space when a log axis is on, so the markers
        # have to be placed there too or they land somewhere else entirely.
        px = np.asarray(marker_x, dtype=float)
        py = np.asarray(marker_y, dtype=float)
        if self.chk_log_x.isChecked():
            px = np.where(px > 0, np.log10(np.where(px > 0, px, 1.0)), np.nan)
        if self.chk_log_y.isChecked():
            py = np.where(py > 0, np.log10(np.where(py > 0, py, 1.0)), np.nan)

        self.plot_widget.addItem(
            pg.ScatterPlotItem(
                px, py,
                size=11,
                pen=pg.mkPen("#ff4081", width=2),
                brush=None,
                symbol="o",
            )
        )

    def apply_q_conversion(self) -> None:
        """Recompute the q axis from the angles and the current energy."""
        if self.x_data is None:
            return
        if self.x_data_original is None:
            self.x_data_original = self.x_data.copy()

        self.x_data = np.array([self._convert_angle_to_q(x) for x in self.x_data_original])
        logging.info("Converted X-axis to q at %s eV", self.photon_energy_ev)
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

    def _display_axis(self, values: np.ndarray, log_scale: bool) -> tuple[np.ndarray, np.ndarray]:
        """Map one axis into the coordinates the view draws in.

        :return: ``(display, valid)``; ``valid`` drops NaN/inf, and in log mode
            also the non-positive samples that cannot be plotted.
        """
        arr = np.asarray(values, dtype=float)
        if log_scale:
            valid = arr > 0
            return np.log10(np.where(valid, arr, 1.0)), valid
        return arr, np.isfinite(arr)

    def _closest_sample_to_click(self, scene_pos) -> tuple[float, float, int | None] | None:
        """Find the sample nearest to a click, measured in screen pixels.

        Every finite sample is a candidate. An earlier version located the click
        with ``argmin(|x - x_click|)`` and then searched +-20 indices around it,
        which broke in two ways: ``argmin`` returns the index of a NaN when the X
        dataset has one, pinning the window to that spot forever, and the index
        window assumes X is sorted, which is false for a swept loop.

        The view-to-scene mapping is affine, so distances are computed vectorised
        via ``viewPixelSize`` rather than mapping every point through Qt. Pixels
        are also the right metric: a Euclidean distance in data units is
        dominated by whichever axis happens to have the larger numbers.
        """
        if self.y_data is None:
            return None

        vb = self.plot_widget.plotItem.vb
        view_pt = vb.mapSceneToView(scene_pos)
        click_x, click_y = float(view_pt.x()), float(view_pt.y())
        pixel_w, pixel_h = (float(v) for v in vb.viewPixelSize())
        if not pixel_w or not pixel_h:
            return None

        y_data = np.asarray(self.display_y, dtype=float)
        n_rows = y_data.shape[0]
        if self.x_data is not None and len(self.x_data) == n_rows:
            x_values = np.asarray(self.x_data, dtype=float)
        else:
            x_values = np.arange(n_rows, dtype=float)

        display_x, valid_x = self._display_axis(x_values, self.chk_log_x.isChecked())
        dx = (display_x - click_x) / pixel_w

        columns = [(None, y_data)] if y_data.ndim == 1 else [
            (col, y_data[:, col]) for col in range(y_data.shape[1])
        ]

        best: tuple[float, float, int | None] | None = None
        best_distance = float("inf")
        for curve_idx, y_column in columns:
            display_y, valid_y = self._display_axis(y_column, self.chk_log_y.isChecked())
            valid = valid_x & valid_y
            if not valid.any():
                continue

            dy = (display_y - click_y) / pixel_h
            distance = np.where(valid, dx * dx + dy * dy, np.inf)
            idx = int(np.argmin(distance))
            if distance[idx] < best_distance:
                best_distance = float(distance[idx])
                best = (float(x_values[idx]), float(y_column[idx]), curve_idx)

        return best

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

        # The curve under the cursor, which is the despiked one when the filter
        # is on; reading out a value that is not the one drawn would be a lie.
        y_display = self.display_y

        # Determine X data array
        if self.x_data is not None:
            x = self.x_data
        else:
            x = np.arange(len(y_display))

        # Find the closest data point
        try:
            # Handle 1D and 2D data
            if y_display.ndim == 1:
                # For 1D data, find closest X index
                idx = np.argmin(np.abs(x - x_mouse))
                x_val = x[idx]
                y_val = y_display[idx]

                # Display with optional q conversion
                if self.btn_convert_to_q.isChecked():
                    q_value = self._convert_angle_to_q(x_val)
                    self.label_coords.setText(f"X: {x_val:.3f} | q: {q_value:.4f} 1/A | Y: {y_val:.3g}")
                else:
                    self.label_coords.setText(f"X: {x_val:.3f} | Y: {y_val:.3g}")
            elif y_display.ndim == 2:
                # For 2D data (multiple curves), find closest point across all curves
                y_mouse = mouse_point.y()
                min_distance = float('inf')
                closest_x = x_mouse
                closest_y = y_mouse
                closest_curve = 0

                # Search through all curves
                num_cols = y_display.shape[1]
                for col_idx in range(num_cols):
                    # Find closest X index in this curve
                    idx = np.argmin(np.abs(x - x_mouse))
                    x_val = x[idx]
                    y_val = y_display[idx, col_idx]

                    # Calculate distance considering log scale
                    distance = self._calculate_distance(x_val, y_val, x_mouse, y_mouse)

                    if distance < min_distance:
                        min_distance = distance
                        closest_x = x_val
                        closest_y = y_val
                        closest_curve = col_idx

                # Display with optional q conversion
                if self.btn_convert_to_q.isChecked():
                    q_value = self._convert_angle_to_q(closest_x)
                    self.label_coords.setText(
                        f"Curve {closest_curve + 1} | X: {closest_x:.3f} | "
                        f"q: {q_value:.4f} 1/A | Y: {closest_y:.3g}"
                    )
                else:
                    self.label_coords.setText(f"Curve {closest_curve + 1} | X: {closest_x:.3f} | Y: {closest_y:.3g}")
        except Exception as exc:
            # Fallback to mouse position. Debug level on purpose: this runs on
            # every mouse move, so anything louder would flood the log.
            logging.debug("Hover readout failed, falling back to cursor position: %s", exc)
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

        if self.y_data.ndim not in (1, 2):
            logging.warning("Unsupported data dimensions for point selection: %s", self.y_data.ndim)
            return

        # Find the closest data point to the click
        try:
            closest = self._closest_sample_to_click(event.scenePos())
            if closest is None:
                logging.warning("Could not find a valid data point to select.")
                return

            x_val, y_val, curve_idx = closest

            # Store selected point (x, y, curve_idx)
            self.selected_point = (x_val, y_val, curve_idx)

            prefix = "" if curve_idx is None else f"Curve {curve_idx + 1} | "
            if self.btn_convert_to_q.isChecked():
                q_value = self._convert_angle_to_q(x_val)
                label_text = f"{prefix}X: {x_val:.3f} | q: {q_value:.4f} 1/A | Y: {y_val:.3g}"
            else:
                label_text = f"{prefix}X: {x_val:.3f} | Y: {y_val:.3g}"

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
        self.btn_convert_to_q.setEnabled(True)

        # Check if X to q conversion is currently enabled
        if self.btn_convert_to_q.isChecked():
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
        """Quick export: write the curve exactly as displayed.

        With no X dataset set, the plot is drawn against the sample index and no
        X column is written — an index column carries no information the row
        order does not already give, and it just gets in the way downstream.
        """
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
        # A faithful copy of one dataset, so the name is just that dataset —
        # no tag needed to say where in the application it came from.
        stem = export_stem(self.y_source_dataset_key, "plot_data")
        preferred = get_table_format(str(settings.value("export/table_format", DEFAULT_TABLE_FORMAT_KEY)))
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot Data",
            suggested_save_path(stem, extension=preferred.suffix),
            save_dialog_filter(preferred),
        )

        if not file_path:
            return  # User cancelled

        fmt = format_from_filter(selected_filter)
        export_path = pathlib.Path(file_path)
        if not export_path.suffix:
            export_path = export_path.with_suffix(fmt.suffix)

        remember_save_directory(export_path)
        settings.setValue("export/table_format", fmt.key)

        try:
            # Quick export is what is on screen: the X/Y actually plotted, plus
            # the q column when the conversion is active. Log axes are a
            # rendering mode, so the values written stay linear.
            headers: list[str] = []
            columns: list[Any] = []
            # Named after the X dataset itself, so a column can still be traced
            # back to where it came from once the file leaves the application.
            if self.btn_convert_to_q.isChecked() and self.x_data_original is not None:
                headers.extend([
                    x_column_header(self.x_dataset_path),
                    x_column_header(self.x_dataset_path, as_q=True),
                ])
                columns.extend([self.x_data_original, self.x_data])
            elif self.x_data is not None:
                headers.append(x_column_header(self.x_dataset_path))
                columns.append(self.x_data)

            # Despiked values when the filter is on — what is exported is what
            # is on screen — with a comment line saying so, because a file that
            # has been filtered and does not admit it is a trap.
            y_export = self.display_y
            if y_export.ndim == 1:
                headers.append("Y")
                columns.append(y_export)
            else:
                for col_idx in range(y_export.shape[1]):
                    headers.append(f"Y_Column_{col_idx + 1}")
                    columns.append(y_export[:, col_idx])

            write_table(export_path, headers, columns, fmt, comments=self._despike_comment())
            file_path = str(export_path)

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

        # Non-modal so a dataset can still be dragged out of the main window's
        # tree, and an ordinary window so it goes behind the application when
        # you switch away rather than sitting above every other program.
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Window, True)

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
            if (str(opened_file) == file_token or opened_file.name == file_token
                    or str(opened_file).endswith(file_token)):
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

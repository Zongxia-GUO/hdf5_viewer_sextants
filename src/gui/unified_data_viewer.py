"""Unified data viewer that automatically selects appropriate display widget."""

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
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class UnifiedDataViewer(QWidget):
    """
    Unified data viewer that automatically selects appropriate display widget.

    This widget automatically chooses the most suitable display widget based on
    the data type and dimensions.

    Supported display types:
    - String data ->QTextBrowser
    - Table data ->QTableView with DataTable model
    - 1D arrays ->PlotWidget1DEnhanced (line plot with controls)
    - 2D arrays (2-5 columns) ->PlotWidget1DEnhanced (multi-curve plot)
    - 2D arrays (large) ->ImageView2DEnhanced (heatmap with colormap controls)
    - 3D+ arrays ->ImageView2DEnhanced (image with slice navigation)

    Features:
    - Automatic widget selection based on data type and dimensions
    - Support for explicit data type hints
    - Multi-curve plotting for 2D data with few columns
    - Image display with colormap and scale controls for 2D/3D data
    - Unified interface across all display types
    """

    def __init__(
        self,
        parent: Any = None,
        opened_files: tuple = None,
        dataset_full_keys_1d: list[str] | None = None,
    ) -> None:
        """
        Initialize the unified data viewer.

        Args:
            parent: Parent widget
            opened_files: Tuple of opened HDF5 file paths (for custom X data selection)
        """
        super().__init__(parent)
        self.opened_files = opened_files or tuple()
        self.dataset_full_keys_1d = list(dataset_full_keys_1d or [])
        self.current_widget: QWidget | None = None
        self.source_dataset_key: str | None = None

        # Set size policy to allow flexible resizing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

    def set_data(
        self,
        data: np.ndarray,
        data_type: str | None = None,
        source_dataset_key: str | None = None,
    ) -> None:
        """
        Set data and automatically choose appropriate display widget.

        This method analyzes the data and creates the most suitable viewer:
        - String data ->Text browser
        - Table data ->Table view
        - 1D data ->Line plot (PlotWidget1DEnhanced)
        - 2D data with 2-5 columns ->Multi-curve plot (PlotWidget1DEnhanced)
        - 2D data (large) ->Image/heatmap (ImageView2DEnhanced)
        - 3D+ data ->Image with slice navigation (ImageView2DEnhanced)

        Args:
            data: NumPy array to display
            data_type: Optional data type hint ("String", "Table", "Array1D", "Array2D", "ImageRGB")
        """
        try:
            if source_dataset_key is not None:
                self.source_dataset_key = source_dataset_key

            # Deferred imports to avoid circular dependencies
            from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
            from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

            # If explicit data type is provided, use it
            if data_type == "String":
                logging.info("UnifiedDataViewer: Displaying as string")
                self._create_string_widget(data)
                return
            elif data_type == "Table":
                logging.info("UnifiedDataViewer: Displaying as table")
                self._create_table_widget(data)
                return

            # Otherwise, determine widget type based on data dimensions
            ndim = len(data.shape)

            # 0-d scalar: always show as plain text regardless of data_type hint
            if ndim == 0:
                logging.info("UnifiedDataViewer: Displaying 0-d scalar as string")
                self._create_string_widget(data)
                return

            # Handle 2D data that should be flattened
            if data_type == "Array1D" and ndim == 2 and min(data.shape) == 1:
                data = data.ravel()
                ndim = 1

            if ndim == 1 or data_type == "Array1D":
                # 1D data - line plot with enhanced controls
                logging.info("UnifiedDataViewer: Displaying as 1D line plot")
                if isinstance(self.current_widget, PlotWidget1DEnhanced):
                    self.current_widget.set_source_dataset_key(self.source_dataset_key)
                    self.current_widget.set_data(data)
                    return
                self._clear_current_widget()
                self._create_plot_widget(data)

            elif ndim == 2 and data.shape[1] <= 5 and data.shape[0] > 1 and data_type != "Array2D":
                # 2D data with few columns (2-5) - multi-curve plot
                logging.info(f"UnifiedDataViewer: Displaying as multi-curve plot ({data.shape[1]} curves)")
                if isinstance(self.current_widget, PlotWidget1DEnhanced):
                    self.current_widget.set_source_dataset_key(self.source_dataset_key)
                    self.current_widget.set_data(data)
                    return
                self._clear_current_widget()
                self._create_plot_widget(data)

            elif ndim >= 2 or data_type == "Array2D":
                # 2D/3D+ data - image viewer (slice slider appears automatically for 3D)
                logging.info(f"UnifiedDataViewer: Displaying as image (ndim={ndim})")
                if isinstance(self.current_widget, ImageView2DEnhanced):
                    self.current_widget.set_data(data)
                    return
                self._clear_current_widget()
                self._create_image_widget(data)

        except Exception as e:
            logging.error(f"UnifiedDataViewer: Failed to create display widget: {e}")
            self._create_error_widget(data, e)

    def _clear_current_widget(self) -> None:
        """Clear the current display widget."""
        if self.current_widget is not None:
            self.layout.removeWidget(self.current_widget)
            self.current_widget.deleteLater()
            self.current_widget = None

    def _swap_widget(self, new_widget: QWidget) -> None:
        """Add new_widget to layout first, then remove old widget.

        This ordering prevents the layout from collapsing to zero height
        between the removal and insertion, which would cause a visible
        'shrink then expand' flicker on every dataset switch.
        """
        self.layout.addWidget(new_widget)
        old = self.current_widget
        self.current_widget = new_widget
        if old is not None:
            self.layout.removeWidget(old)
            old.deleteLater()

    def _create_plot_widget(self, data: np.ndarray) -> None:
        """Create a 1D plot widget."""
        from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

        plot_widget = PlotWidget1DEnhanced(
            parent=self,
            opened_files=self.opened_files,
            dataset_full_keys_1d=self.dataset_full_keys_1d,
        )
        plot_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        plot_widget.set_source_dataset_key(self.source_dataset_key)
        plot_widget.set_data(data)
        self._swap_widget(plot_widget)

    def refresh_dataset_keys(self, full_keys_1d: list[str], opened_files: tuple | None = None) -> None:
        """Refresh shared 1D dataset index for X-data selection dialogs."""
        self.dataset_full_keys_1d = list(full_keys_1d or [])
        if opened_files is not None:
            self.opened_files = tuple(opened_files)
        if self.current_widget is not None:
            try:
                from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

                if isinstance(self.current_widget, PlotWidget1DEnhanced):
                    self.current_widget.dataset_full_keys_1d = list(self.dataset_full_keys_1d)
                    self.current_widget.opened_files = tuple(self.opened_files)
            except Exception:
                pass

    def _create_image_widget(self, data: np.ndarray) -> None:
        """Create a 2D/3D image widget."""
        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

        image_view = ImageView2DEnhanced(parent=self)
        image_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        image_view.q_calibration_requested.connect(self._on_image_q_requested)
        self._swap_widget(image_view)
        image_view.set_data(data)

    def _on_image_q_requested(self) -> None:
        """Forward image-viewer q request with current source dataset key."""
        self.q_calibration_requested.emit(self.source_dataset_key)

    def _create_string_widget(self, data: np.ndarray) -> None:
        """Create a text widget for string data."""

        if data.ndim == 0:
            label = data.item()
            if isinstance(label, bytes):
                label = label.decode()
            label = str(label)
        else:
            label = str(data)

        text_widget = QTextBrowser()
        text_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        text_widget.setText(label)
        self._swap_widget(text_widget)

    def _create_table_widget(self, data: np.ndarray) -> None:
        """Create a table widget for structured data, with a slice slider for 3D+ data."""
        from src.gui.table_model import CopyableTableView, DataTable

        # Squeeze out size-1 dimensions so (1, 2048, 2048) ->(2048, 2048)
        if data.dtype.names is None:
            data = np.squeeze(data)
            if data.ndim == 0:
                data = data.reshape(1, 1)
            elif data.ndim == 1:
                data = data.reshape(-1, 1)

        # Container holds optional slice bar + table
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(4)

        table_view = CopyableTableView()
        table_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        # Multi-slice table: data is 3D+ with more than one slice
        if data.ndim > 2 and data.shape[0] > 1:
            n_slices = data.shape[0]

            # -- slice controls --
            ctrl = QHBoxLayout()
            ctrl.setContentsMargins(4, 2, 4, 2)
            lbl_prefix = QLabel("Slice:")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(n_slices - 1)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.NoTicks)
            lbl_index = QLabel(f"1 / {n_slices}")
            lbl_index.setMinimumWidth(60)
            ctrl.addWidget(lbl_prefix)
            ctrl.addWidget(slider, stretch=1)
            ctrl.addWidget(lbl_index)
            c_layout.addLayout(ctrl)

            # Initial model (slice 0)
            table_view.setModel(DataTable(data[0]))

            def _on_slice(idx: int) -> None:
                lbl_index.setText(f"{idx + 1} / {n_slices}")
                table_view.setModel(DataTable(data[idx]))

            slider.valueChanged.connect(_on_slice)

        else:
            # Plain 2D table -no slider needed
            display = data[0] if data.ndim > 2 else data
            table_view.setModel(DataTable(display))

        c_layout.addWidget(table_view)
        self._swap_widget(container)

    def _create_error_widget(self, data: np.ndarray, error: Exception) -> None:
        """Create an error display widget as fallback."""
        fallback_label = QLabel(
            f"<b>Error displaying data:</b><br>{str(error)}<br><br>"
            f"Data shape: {data.shape}<br>"
            f"Data dtype: {data.dtype}"
        )
        fallback_label.setWordWrap(True)
        fallback_label.setStyleSheet("color: red; padding: 10px;")
        self._swap_widget(fallback_label)

    def get_current_widget(self) -> QWidget | None:
        """
        Get the current display widget.

        Returns:
            The current PlotWidget1DEnhanced, ImageView2DEnhanced, or error widget
        """
        return self.current_widget

    def clear(self) -> None:
        """Clear all displayed data and widgets."""
        self._clear_current_widget()



    q_calibration_requested = pyqtSignal(object)

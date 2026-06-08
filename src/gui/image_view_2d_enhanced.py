"""Enhanced 2D Image Viewer with colormap and scale controls."""

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
import types
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QEvent, QObject, pyqtSignal, QRectF, QBuffer, QByteArray, QMimeData, QSize, QSettings
from PyQt6.QtGui import QIcon, QImage, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.img.img_path import img_path

IMAGE_SAVE_FILTER = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff)"


class SceneEventFilter(QObject):
    """Event filter for capturing mouse press/release events on the scene."""

    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget

    def eventFilter(self, obj, event):
        """Filter events to capture mouse press and release."""
        if event.type() == QEvent.Type.GraphicsSceneMousePress:
            self.parent_widget._on_mouse_pressed(event)
            return False  # Allow event to propagate
        elif event.type() == QEvent.Type.GraphicsSceneMouseRelease:
            self.parent_widget._on_mouse_released(event)
            return False  # Allow event to propagate

        return super().eventFilter(obj, event)


class ImageView2DEnhanced(QWidget):
    """Enhanced ImageView with colormap and scale mode controls."""
    q_calibration_requested = pyqtSignal()
    # Emitted by TargetItem handle drag: (r_min, r_max, angle_min, angle_max)
    sigRadialParamsChanged = pyqtSignal(int, int, int, int)
    # Emitted when the radial center is moved (click or programmatic): (cx, cy)
    sigRadialCenterChanged = pyqtSignal(float, float)

    def __init__(self, parent: Any = None) -> None:
        """Initialize the enhanced 2D image viewer."""
        super().__init__(parent)

        self.data = None  # Store original data (2D slice)
        self.data_3d = None  # Store original 3D data if applicable
        self.current_slice_index = 0  # Current slice index for 3D data
        self.current_slice_axis = 0  # Current axis used for 3D slicing
        self._lazy_shape: tuple[int, ...] | None = None
        self.current_transform = "linear"  # Current scale mode
        self.global_min = None  # Global min for locked levels
        self.global_max = None  # Global max for locked levels
        self.locked_levels = None  # Locked color levels (min, max)
        self.current_roi = None  # Current ROI item
        self.roi_type = None  # Current ROI type
        self.ruler_roi = None  # Independent ruler ROI
        self.ruler_text = None  # Distance label for ruler
        self._axis_linear_map_active = False
        self._axis_linear_x_scale = 1.0
        self._axis_linear_y_scale = 1.0
        self._axis_linear_x_offset = 0.0
        self._axis_linear_y_offset = 0.0
        self._axis_linear_unit = "px"
        self._axis_label_x = "X (pixels)"
        self._axis_label_y = "Y (pixels)"

        # Sector ROI parameters
        self.sector_center = None  # (x, y) center point
        self.sector_angle_start = 0  # Start angle in degrees
        self.sector_angle_end = 90  # End angle in degrees
        self.sector_radius_inner = 10  # Inner radius
        self.sector_radius_outer = 50  # Outer radius
        self.sector_plot_item = None  # PlotDataItem for sector outline
        self.sector_center_marker = None  # ScatterPlotItem for center point

        # Sector control points (handles) - using ScatterPlotItem for fixed pixel size
        self.sector_handle_outer_start = None  # Outer radius - start angle
        self.sector_handle_outer_end = None    # Outer radius - end angle
        self.sector_handle_inner_start = None  # Inner radius - start angle
        self.sector_handle_inner_end = None    # Inner radius - end angle

        # Track which handle is being dragged
        self.dragging_handle = None
        self.drag_start_pos = None

        # Radial profile tool
        self._radial_mode_active = False
        self._radial_center: tuple[float, float] | None = None
        self._radial_center_marker = None  # ScatterPlotItem shown on the image
        self._radial_r_min: int = 0
        self._radial_r_max: int = 0          # 0 means "use maximum possible"
        self._radial_angle_min: int = 0      # degrees, -180–180
        self._radial_angle_max: int = 180    # degrees, -180–180
        self._radial_arc1 = None             # PlotDataItem — first arc sector boundary
        self._radial_arc2 = None             # PlotDataItem — symmetric arc sector boundary
        self._radial_handle_outer_start = None  # TargetItem corner handles
        self._radial_handle_outer_end   = None
        self._radial_handle_inner_start = None
        self._radial_handle_inner_end   = None
        self._radial_active_drag_attr: str | None = None  # attr name of handle being dragged

        # Event filter for mouse press/release events
        self.scene_event_filter = None

        # Lazy slice loading (for large 3D datasets over network)
        self._slice_loader = None   # callable(idx) -> np.ndarray
        self._slice_cache: dict = {}  # idx -> float32 array (caches last few slices)
        self._q_calibration: dict | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Set size policy to allow shrinking horizontally (ignore minimum size hints)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Control panel
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)  # Reduce spacing between widgets

        self.btn_copy_image = QPushButton()
        self.btn_copy_image.setIcon(QIcon(str(pathlib.Path(img_path(), "copy.ico"))))
        self.btn_copy_image.setIconSize(QSize(16, 16))
        self.btn_copy_image.setFixedSize(28, 24)
        self.btn_copy_image.setToolTip("Copy current image data with the active colormap")
        self.btn_copy_image.clicked.connect(self.copy_colormapped_image_to_clipboard)
        control_layout.addWidget(self.btn_copy_image)

        self.btn_save_image = QPushButton()
        self.btn_save_image.setIcon(QIcon(str(pathlib.Path(img_path(), "save.ico"))))
        self.btn_save_image.setIconSize(QSize(16, 16))
        self.btn_save_image.setFixedSize(28, 24)
        self.btn_save_image.setToolTip("Save/export current dataset")
        self.btn_save_image.clicked.connect(self._request_main_export)
        control_layout.addWidget(self.btn_save_image)

        self.btn_q_calibration = QPushButton("Q")
        self.btn_q_calibration.setFixedSize(24, 24)
        self.btn_q_calibration.setToolTip("Open Q Calibration Tool")
        self.btn_q_calibration.clicked.connect(self.q_calibration_requested.emit)
        control_layout.addWidget(self.btn_q_calibration)

        control_layout.addSpacing(10)

        # Colormap selection
        colormap_label = QLabel("Colormap:")
        control_layout.addWidget(colormap_label)

        self.combo_colormap = QComboBox()
        # Only use colormaps that are guaranteed to exist in pyqtgraph
        self.combo_colormap.addItems([
            "viridis",
            "inferno",
            "cividis",
            "turbo",
            "CET-L9",      # Grayscale
            "CET-L1",      # Thermal
            "CET-L4",      # Rainbow
            "CET-R4",      # Red-blue diverging
            "CET-D1",      # Blue-white-red diverging
            "CET-D9",      # Blue-red diverging
        ])
        self.combo_colormap.setCurrentText("viridis")
        self.combo_colormap.setMinimumWidth(0)  # Allow shrinking
        self.combo_colormap.setMaximumWidth(100)  # Reduced from 120
        self.combo_colormap.currentTextChanged.connect(self._update_colormap)
        control_layout.addWidget(self.combo_colormap)

        # Invert colormap checkbox
        self.chk_invert = QCheckBox("Invert")
        self.chk_invert.stateChanged.connect(self._update_colormap)
        control_layout.addWidget(self.chk_invert)

        control_layout.addSpacing(10)  # Reduced from 20

        # Scale mode selection (Linear/Log)
        scale_label = QLabel("Scale:")
        control_layout.addWidget(scale_label)

        self.combo_scale = QComboBox()
        self.combo_scale.addItems([
            "Linear",
            "Log",
            "SymLog",
            "Square root",
        ])
        self.combo_scale.setMinimumWidth(0)  # Allow shrinking
        self.combo_scale.setMaximumWidth(100)  # Reduced from 120
        self.combo_scale.currentTextChanged.connect(self._update_scale)
        control_layout.addWidget(self.combo_scale)

        self.btn_auto_contrast = QPushButton("Auto")
        self.btn_auto_contrast.setFixedSize(42, 24)
        self.btn_auto_contrast.setToolTip("Auto contrast from histogram percentiles")
        self.btn_auto_contrast.clicked.connect(self._auto_contrast)
        control_layout.addWidget(self.btn_auto_contrast)

        control_layout.addSpacing(10)  # Reduced from 20

        # Show axes checkbox
        self.chk_show_axes = QCheckBox("Show Axes")
        self.chk_show_axes.setToolTip("Display coordinate axes on all four sides")
        self.chk_show_axes.setChecked(False)
        self.chk_show_axes.stateChanged.connect(self._on_show_axes_changed)
        control_layout.addWidget(self.chk_show_axes)

        control_layout.addSpacing(10)  # Reduced from 20

        # ROI controls - CHANGED TO BUTTONS
        self.label_roi = QLabel("ROI:")
        control_layout.addWidget(self.label_roi)

        # Line ROI button
        self.btn_roi_line = QPushButton("━")
        self.btn_roi_line.setFixedSize(24, 24)  # Reduced from 26x26
        self.btn_roi_line.setCheckable(True)
        self.btn_roi_line.setToolTip("Line ROI")
        self.btn_roi_line.setStyleSheet("font-weight: bold; font-size: 9pt; padding: 0px;")
        self.btn_roi_line.clicked.connect(lambda: self._on_roi_button_clicked("Line"))
        control_layout.addWidget(self.btn_roi_line)

        # Rectangle ROI button
        self.btn_roi_rect = QPushButton("■")
        self.btn_roi_rect.setFixedSize(24, 24)  # Reduced from 26x26
        self.btn_roi_rect.setCheckable(True)
        self.btn_roi_rect.setToolTip("Rectangle ROI")
        self.btn_roi_rect.setStyleSheet("font-weight: bold; font-size: 9pt; padding: 0px;")
        self.btn_roi_rect.clicked.connect(lambda: self._on_roi_button_clicked("Rectangle"))
        control_layout.addWidget(self.btn_roi_rect)

        # Ruler button (independent from ROI profile tools)
        self.btn_ruler = QPushButton("R")
        self.btn_ruler.setFixedSize(24, 24)
        self.btn_ruler.setCheckable(True)
        self.btn_ruler.setToolTip("Ruler")
        self.btn_ruler.setStyleSheet("font-weight: bold; font-size: 9pt; padding: 0px;")
        self.btn_ruler.clicked.connect(self._on_ruler_clicked)
        control_layout.addWidget(self.btn_ruler)

        control_layout.addSpacing(10)  # Reduced from 20

        # Coordinates display label
        self.label_coords = QLabel("X: - | Y: - | Value: -")
        self.label_coords.setStyleSheet("color: gray; font-size: 8pt;")  # Reduced font size
        # Set max width to prevent excessive expansion, but allow shrinking
        self.label_coords.setMaximumWidth(250)
        self.label_coords.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        control_layout.addWidget(self.label_coords)

        # Add stretch at the end to keep controls left-aligned when resizing
        control_layout.addStretch()

        layout.addLayout(control_layout)

        # Slice navigator for 3D data (initially hidden)
        self.slice_layout = QHBoxLayout()

        self.label_slice_axis = QLabel("Axis:")
        self.slice_layout.addWidget(self.label_slice_axis)

        self.combo_slice_axis = QComboBox()
        self.combo_slice_axis.setToolTip("Choose which 3D array axis the slice slider moves through")
        self.combo_slice_axis.currentIndexChanged.connect(self._on_slice_axis_changed)
        self.slice_layout.addWidget(self.combo_slice_axis)

        self.label_slice = QLabel("Slice:")
        self.slice_layout.addWidget(self.label_slice)

        self.slider_slice = QSlider(Qt.Orientation.Horizontal)
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(0)
        self.slider_slice.setValue(0)
        self.slider_slice.valueChanged.connect(self._on_slice_changed)
        self.slice_layout.addWidget(self.slider_slice)

        self.label_slice_info = QLabel("0 / 0")
        # Remove width constraint to allow flexible resizing
        self.slice_layout.addWidget(self.label_slice_info)

        self.slice_layout.addSpacing(20)

        # Lock levels checkbox for consistent contrast across slices
        self.chk_lock_levels = QCheckBox("Lock Levels")
        self.chk_lock_levels.setToolTip("Lock current color levels when browsing slices")
        self.chk_lock_levels.setChecked(False)
        self.chk_lock_levels.stateChanged.connect(self._on_lock_levels_changed)
        self.slice_layout.addWidget(self.chk_lock_levels)

        layout.addLayout(self.slice_layout)

        # Hide slice controls initially (only show for 3D data)
        self.label_slice_axis.hide()
        self.combo_slice_axis.hide()
        self.label_slice.hide()
        self.slider_slice.hide()
        self.label_slice_info.hide()
        self.chk_lock_levels.hide()

        # Create a GraphicsLayoutWidget to hold plot and histogram
        self.graphics_layout = pg.GraphicsLayoutWidget()
        # Set size policy to allow graphics layout to expand and fill available space
        self.graphics_layout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.graphics_layout.setBackground('k')  # Black background

        # Create a PlotItem for the image using addPlot
        self.plot_item = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_item.setMenuEnabled(False)  # Disable right-click menu

        # Get ViewBox and configure it
        self.view_box = self.plot_item.getViewBox()
        self.view_box.setAspectLocked(True, ratio=1.0)  # Square pixels by default
        # Keep image row 0 at top (same orientation as FTH panels / detector convention).
        self.view_box.invertY(True)

        # Create ImageItem for displaying the image
        # Force local row-major axis order so ROI geometry/profiles do not depend on
        # external/global pyqtgraph config state.
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_item.addItem(self.image_item)

        # Hide axes by default (will show when checkbox is checked)
        for axis_name in ['left', 'bottom', 'right', 'top']:
            self.plot_item.hideAxis(axis_name)

        # Create histogram widget for color levels control
        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.graphics_layout.addItem(self.histogram, row=0, col=1)

        # Connect mouse movement to update coordinates
        self.graphics_layout.scene().sigMouseMoved.connect(self._on_mouse_moved)
        # sigMouseClicked fires only on genuine clicks (no drag) — used by radial tool
        self.graphics_layout.scene().sigMouseClicked.connect(self._on_scene_clicked)

        # Store references for compatibility
        self.image_view = type('ImageViewCompat', (), {
            'view': self.view_box,
            'imageItem': self.image_item,
            'scene': self.graphics_layout.scene(),
        })()

        # Create plot_widget reference for axes control (points to plot_item)
        self.plot_widget = self.plot_item
        self._axis_tick_original: dict[str, Any] = {
            n: self.plot_widget.getAxis(n).tickStrings
            for n in ("left", "bottom", "right", "top")
        }

        # Install event filter for mouse press/release
        self.scene_event_filter = SceneEventFilter(self)
        self.graphics_layout.scene().installEventFilter(self.scene_event_filter)

        # ROI statistics plot widget (initially hidden)
        self.roi_plot_widget = pg.PlotWidget()
        self.roi_plot_widget.setMinimumHeight(100)  # Minimum height when visible
        self.roi_plot_widget.setBackground('k')  # Black background (dark theme)
        self.roi_plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Set axis colors to white for dark theme
        axis_pen = pg.mkPen(color='w', width=1)
        for axis in ['left', 'bottom', 'right', 'top']:
            self.roi_plot_widget.getAxis(axis).setPen(axis_pen)
            self.roi_plot_widget.getAxis(axis).setTextPen(axis_pen)

        self.roi_plot_widget.hide()

        # Create a vertical splitter for image and ROI plot
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.graphics_layout)
        self.splitter.addWidget(self.roi_plot_widget)
        # Set initial sizes: 80% for image, 20% for ROI plot
        self.splitter.setSizes([800, 200])
        # Allow collapsing the ROI plot when hidden
        self.splitter.setCollapsible(1, True)

        layout.addWidget(self.splitter)

        self.setLayout(layout)

        # Apply default colormap (viridis)
        self._update_colormap()

    @staticmethod
    def _to_float32(data: np.ndarray) -> np.ndarray:
        """Convert data to float32 for faster processing and lower memory usage."""
        if data.dtype == np.float32:
            return data
        return data.astype(np.float32, copy=False)

    def set_data(self, data: np.ndarray) -> None:
        """
        Set the data to display.

        Args:
            data: 2D or 3D numpy array to display
        """
        # Remove existing ROI when loading new data
        if self.current_roi is not None:
            self.image_view.view.removeItem(self.current_roi)
            self.current_roi = None
            self.roi_type = None
            self.btn_roi_line.setChecked(False)
            self.btn_roi_rect.setChecked(False)
            self.roi_plot_widget.hide()

        # Remove sector outline if it exists
        if self.sector_plot_item is not None:
            self.image_view.view.removeItem(self.sector_plot_item)
            self.sector_plot_item = None

        # Remove sector control points if they exist
        self._remove_sector_control_points()

        # Remove radial overlays when loading new data
        self._remove_radial_overlays()
        self._radial_center = None
        if self._radial_mode_active:
            self.roi_plot_widget.hide()

        # Check if data is genuinely 3D after removing singleton dimensions.
        # This lets stacks stored as (H, W, N), (N, H, W), or (1, H, W, N)
        # choose the correct slice axis instead of always assuming axis 0.
        display_source = np.squeeze(data) if data.ndim >= 3 else data
        if display_source.ndim >= 3:
            # True 3D data with multiple slices - enable slice mode
            # Performance: convert to float32 to halve memory usage
            self.data_3d = self._to_float32(display_source)
            self.current_slice_index = 0
            self.current_slice_axis = 0

            # Calculate global min/max for locked levels
            self.global_min = float(np.nanmin(display_source)) if display_source.size > 0 else 0
            self.global_max = float(np.nanmax(display_source)) if display_source.size > 0 else 0

            self._configure_slice_axis_combo(self.data_3d.shape)
            self._reset_slice_slider(self.data_3d.shape[self.current_slice_axis])
            self._show_slice_controls()

            # Extract first slice as current data (no copy needed, data_3d owns the memory)
            self.data = self._extract_full_slice(0)

        else:
            # 2D data or 3D data with singleton first dimension
            # Squeeze any singleton dimensions if present
            if data.ndim >= 3:
                data = display_source
                logging.info(f"Squeezed singleton dimension: shape now {data.shape}")
            self.data_3d = None
            self._lazy_shape = None
            # Performance: convert to float32
            self.data = self._to_float32(data)
            self.global_min = None
            self.global_max = None
            self.locked_levels = None

            self._hide_slice_controls()
            self.chk_lock_levels.setChecked(False)

        # Reset lazy loader when a full dataset is supplied
        self._slice_loader = None
        self._slice_cache = {}

        # Apply current transform and display
        self._update_display()

    def set_data_lazy(
        self,
        first_slice: np.ndarray,
        total_slices: int,
        loader,
        full_shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Display a large 3D dataset without loading it all into memory.

        Only the first slice is kept in RAM. Remaining slices are fetched
        on demand via *loader(idx)* when the user moves the slice slider.
        This avoids transferring hundreds of MB over the network upfront.

        Args:
            first_slice: 2-D array for slice 0 (already downloaded by worker)
            total_slices: total number of slices in the full 3-D dataset
            loader: callable(axis, idx) or callable(idx) that reads one slice from the server
            full_shape: Full source dataset shape. When supplied, the axis selector can
                browse X/Y/Z instead of being limited to axis 0.
        """
        # Remove existing ROI / sector overlays
        if self.current_roi is not None:
            self.image_view.view.removeItem(self.current_roi)
            self.current_roi = None
            self.roi_type = None
            self.btn_roi_line.setChecked(False)
            self.btn_roi_rect.setChecked(False)
            self.roi_plot_widget.hide()
        if self.sector_plot_item is not None:
            self.image_view.view.removeItem(self.sector_plot_item)
            self.sector_plot_item = None
        self._remove_sector_control_points()

        # Lazy state
        self._slice_loader = loader
        self._slice_cache = {(0, 0): self._to_float32(first_slice)}  # pre-cache axis 0, slice 0
        self.data_3d = None                     # not fully loaded
        self._lazy_shape = tuple(full_shape) if full_shape is not None else (int(total_slices),) + tuple(first_slice.shape)
        self.global_min = None
        self.global_max = None
        self.locked_levels = None
        self.current_slice_index = 0
        self.current_slice_axis = 0
        self.data = self._slice_cache[(0, 0)]

        self._configure_slice_axis_combo(self._lazy_shape)
        self._reset_slice_slider(self._lazy_shape[self.current_slice_axis])

        self._show_slice_controls()
        self.chk_lock_levels.setChecked(False)

        self._update_display()

    def _configure_slice_axis_combo(self, shape: tuple[int, ...]) -> None:
        """Populate the X/Y/Z slice-axis selector for a 3D dataset shape."""
        axis_names = ["X", "Y", "Z"]
        max_axes = min(3, len(shape))
        previous_axis = min(self.current_slice_axis, max_axes - 1)

        self.combo_slice_axis.blockSignals(True)
        self.combo_slice_axis.clear()
        for axis in range(max_axes):
            self.combo_slice_axis.addItem(f"{axis_names[axis]} ({shape[axis]})", axis)
        self.combo_slice_axis.setCurrentIndex(previous_axis)
        self.combo_slice_axis.blockSignals(False)
        self.current_slice_axis = previous_axis

    def _show_slice_controls(self) -> None:
        """Show controls used for 3D slice browsing."""
        self.label_slice_axis.show()
        self.combo_slice_axis.show()
        self.label_slice.show()
        self.slider_slice.show()
        self.label_slice_info.show()
        self.chk_lock_levels.show()

    def _hide_slice_controls(self) -> None:
        """Hide controls used for 3D slice browsing."""
        self.label_slice_axis.hide()
        self.combo_slice_axis.hide()
        self.label_slice.hide()
        self.slider_slice.hide()
        self.label_slice_info.hide()
        self.chk_lock_levels.hide()

    def _reset_slice_slider(self, num_slices: int) -> None:
        """Reset the slice slider for the selected axis."""
        num_slices = max(1, int(num_slices))
        self.slider_slice.blockSignals(True)
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(num_slices - 1)
        self.slider_slice.setValue(0)
        self.slider_slice.blockSignals(False)
        self.current_slice_index = 0
        self.label_slice_info.setText(f"1 / {num_slices}")

    def _extract_full_slice(self, index: int) -> np.ndarray:
        """Return one 2D/3D slice from a fully loaded ndarray."""
        if self.data_3d is None:
            raise ValueError("No full 3D data is loaded")
        return np.take(self.data_3d, int(index), axis=int(self.current_slice_axis))

    def _load_lazy_slice(self, axis: int, index: int) -> np.ndarray:
        """Load one slice from the lazy loader, accepting old and new call signatures."""
        if self._slice_loader is None:
            raise ValueError("No lazy slice loader is configured")
        try:
            return self._slice_loader(int(axis), int(index))
        except TypeError:
            if int(axis) != 0:
                raise
            return self._slice_loader(int(index))

    def _update_display(self) -> None:
        """Update the display with current transform."""
        if self.data is None:
            return

        # Apply transform
        transformed_data = self._transform_data(self.data)

        # Display
        try:
            # Set image data (row-major order configured in pyqtgraph, no transpose needed)
            self.image_item.setImage(transformed_data, autoLevels=False)

            # Set color levels
            if self.chk_lock_levels.isChecked() and self.locked_levels is not None:
                # Use locked levels
                self.histogram.setLevels(self.locked_levels[0], self.locked_levels[1])
                logging.debug(f"Applied locked levels: [{self.locked_levels[0]:.3g}, {self.locked_levels[1]:.3g}]")
            else:
                # _transform_data guarantees no NaN/inf; use faster min/max (single pass each)
                data_min = float(transformed_data.min())
                data_max = float(transformed_data.max())
                # If data spans both negative and positive values, use symmetric levels
                # around zero to keep diverging contrast balanced.
                if data_min < 0 < data_max:
                    abs_max = max(abs(data_min), abs(data_max))
                    self.histogram.setLevels(-abs_max, abs_max)
                else:
                    self.histogram.setLevels(data_min, data_max)

            # Reapply display geometry (incidence scale centred on display_origin).
            self.view_box.setAspectLocked(True, ratio=1.0)
            self._apply_display_transform()
            self._auto_fit_view()

        except Exception as e:
            logging.error(f"Failed to display image: {e}")

    @staticmethod
    def _robust_auto_levels(display_data: np.ndarray) -> tuple[float, float] | None:
        """Return contrast levels from histogram-like percentiles."""
        finite = np.asarray(display_data)[np.isfinite(display_data)]
        if finite.size == 0:
            return None

        data_min = float(finite.min())
        data_max = float(finite.max())
        if data_max <= data_min:
            eps = abs(data_min) * 1e-6 or 1.0
            return data_min - eps, data_max + eps

        low, high = np.percentile(finite, [0.5, 99.5])
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = data_min, data_max

        # Difference images should keep zero centered, with balanced positive
        # and negative clipping bounds.
        if data_min < 0 < data_max:
            bound = max(abs(low), abs(high))
            if not np.isfinite(bound) or bound <= 0:
                bound = max(abs(data_min), abs(data_max))
            return -float(bound), float(bound)

        # Positive-only images use the actual histogram lower edge so the low
        # tail is not clipped, while the upper edge remains percentile-based to
        # avoid a few hot pixels flattening the contrast.
        if data_min >= 0:
            upper = high if high > 0 else data_max
            return float(data_min), float(upper)

        # Negative-only images use the actual upper edge for the same reason.
        if data_max <= 0:
            lower = low if low < 0 else data_min
            return float(lower), float(data_max)

        return low, high

    def _auto_contrast(self) -> None:
        """Set color levels from the current image histogram distribution."""
        if self.data is None:
            return

        try:
            display_data = self._transform_data(self.data)
            levels = self._robust_auto_levels(display_data)
            if levels is None:
                return

            level_min, level_max = levels
            self.histogram.setLevels(level_min, level_max)
            if self.chk_lock_levels.isChecked():
                self.locked_levels = (level_min, level_max)

            logging.info("Auto contrast levels: [%.6g, %.6g]", level_min, level_max)
        except Exception as exc:
            logging.error("Failed to apply auto contrast: %s", exc)

    def _transform_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply scale transformation to data.

        Args:
            data: Original data array

        Returns:
            Transformed data array
        """
        mode = self.combo_scale.currentText()

        if mode == "Linear":
            return data

        elif mode == "Log":
            # Log scale: log10(data) - use np.where to avoid copy
            positive_mask = data > 0
            min_positive = float(np.min(data[positive_mask])) if np.any(positive_mask) else 1e-10
            return np.log10(np.where(positive_mask, data, min_positive))

        elif mode == "SymLog":
            # Symmetric log: sign(data) * log10(1 + abs(data))
            return np.sign(data) * np.log10(1 + np.abs(data))

        elif mode == "Square root":
            # Square root scale - use np.clip to avoid copy
            return np.sqrt(np.clip(data, 0, None))

        else:
            return data

    def export_colormapped_image(self, file_path: str | pathlib.Path, data: np.ndarray | None = None) -> bool:
        """Export the current 2D image using the active colormap and levels."""
        try:
            from PIL import Image

            rgb = self.render_colormapped_rgb(data=data)
            if rgb is None:
                return False

            export_path = pathlib.Path(file_path)
            img = Image.fromarray(rgb, mode="RGB")
            suffix = export_path.suffix.lower()
            if suffix in [".jpg", ".jpeg"]:
                img.save(export_path, "JPEG", quality=95)
            elif suffix in [".tif", ".tiff"]:
                img.save(export_path, "TIFF")
            else:
                img.save(export_path, "PNG")
            logging.info(
                "Successfully exported colormapped image to %s (colormap=%s, inverted=%s)",
                export_path,
                self.combo_colormap.currentText(),
                self.chk_invert.isChecked(),
            )
            return True
        except Exception as exc:
            logging.error("Failed to export colormapped image: %s", exc)
            return False

    @staticmethod
    def image_extension_from_filter(selected_filter: str) -> str:
        """Infer image extension from a save-dialog filter."""
        text = (selected_filter or "").lower()
        if "*.jpg" in text or "*.jpeg" in text:
            return ".jpg"
        if "*.tif" in text or "*.tiff" in text:
            return ".tif"
        return ".png"

    @staticmethod
    def image_format_from_path(file_path: str, selected_filter: str = "") -> str:
        """Return the Qt/PIL image format name for a path/filter pair."""
        suffix = pathlib.Path(file_path).suffix.lower()
        if suffix in [".jpg", ".jpeg"] or "jpeg" in (selected_filter or "").lower():
            return "JPEG"
        if suffix in [".tif", ".tiff"] or "tiff" in (selected_filter or "").lower():
            return "TIFF"
        return "PNG"

    def save_colormapped_image_dialog(self, default_base_name: str = "image") -> bool:
        """Save current colormapped image through the shared image save dialog."""
        rgb = self.render_colormapped_rgb()
        if rgb is None:
            QMessageBox.warning(self, "No Image", "No image data available to save.")
            return False

        settings = QSettings()
        saved_dir = settings.value("paths/last_export_directory", defaultValue=str(pathlib.Path.home()))
        default_dir = pathlib.Path(str(saved_dir)) if saved_dir else pathlib.Path.home()
        if not default_dir.exists():
            default_dir = pathlib.Path.home()

        stem = pathlib.Path(str(default_base_name or "image")).stem or "image"
        default_path = str(default_dir / f"{stem}.png")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            default_path,
            IMAGE_SAVE_FILTER,
        )
        if not file_path:
            return False

        export_path = pathlib.Path(file_path)
        if not export_path.suffix:
            export_path = export_path.with_suffix(self.image_extension_from_filter(selected_filter))

        settings.setValue("paths/last_export_directory", str(export_path.parent))
        ok = self.export_colormapped_image(export_path)
        if not ok:
            QMessageBox.critical(self, "Save Failed", f"Failed to save image:\n{export_path}")
        return ok

    def render_colormapped_rgb(self, data: np.ndarray | None = None) -> np.ndarray | None:
        """Render data to an RGB uint8 array using current scale, levels, and colormap."""
        source_data = self.data if data is None else data
        if source_data is None:
            return None

        display_data = np.asarray(self._transform_data(np.asarray(source_data)), dtype=np.float64)
        display_data = np.nan_to_num(display_data, nan=0.0, posinf=0.0, neginf=0.0)

        levels = self.histogram.getLevels()
        if levels is None:
            level_min = float(display_data.min()) if display_data.size else 0.0
            level_max = float(display_data.max()) if display_data.size else 1.0
        else:
            level_min, level_max = float(levels[0]), float(levels[1])

        if not np.isfinite(level_min) or not np.isfinite(level_max) or level_max <= level_min:
            level_min = float(display_data.min()) if display_data.size else 0.0
            level_max = float(display_data.max()) if display_data.size else 1.0
        if level_max <= level_min:
            normalized = np.zeros_like(display_data, dtype=np.float64)
        else:
            normalized = np.clip((display_data - level_min) / (level_max - level_min), 0.0, 1.0)

        cmap = self._resolve_colormap(self.combo_colormap.currentText())
        if cmap is None:
            return None
        cmap = self._maybe_invert_colormap(cmap, self.chk_invert.isChecked())
        lut = np.asarray(cmap.getLookupTable(0.0, 1.0, 256))
        if lut.dtype.kind == "f":
            lut = np.clip(lut, 0.0, 255.0)
        lut = lut.astype(np.uint8)
        if lut.ndim != 2 or lut.shape[1] < 3:
            logging.error("Invalid colormap lookup table shape: %s", getattr(lut, "shape", None))
            return None

        indices = np.clip(np.rint(normalized * 255), 0, 255).astype(np.uint8)
        return lut[indices, :3]

    def copy_colormapped_image_to_clipboard(self) -> None:
        """Copy current image data rendered with the active colormap."""
        rgb = self.render_colormapped_rgb()
        if rgb is None:
            return
        try:
            app = QApplication.instance()
            if app is None:
                return
            rgb = np.ascontiguousarray(rgb)
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

            jpeg_bytes = QByteArray()
            buf = QBuffer(jpeg_bytes)
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            qimg.save(buf, "JPEG", 95)
            buf.close()

            mime = QMimeData()
            mime.setImageData(qimg)
            if not jpeg_bytes.isEmpty():
                mime.setData("image/jpeg", jpeg_bytes)
            app.clipboard().setMimeData(mime)
            logging.info("Copied colormapped image data to clipboard")
        except Exception as exc:
            logging.error("Failed to copy colormapped image: %s", exc)

    def _request_main_export(self) -> None:
        """Ask the main window to run the existing Export action."""
        win = self.window()
        handler = getattr(win, "_handle_action_export_current", None)
        if callable(handler):
            handler()
        else:
            logging.warning("Main export handler not available from image viewer")

    def _effective_display_scales(self) -> tuple[float, float]:
        """Return effective display scales, including incident correction if active."""
        actual_scale_x = 1.0
        actual_scale_y = 1.0

        p = self._q_calibration or {}
        if bool(p.get("use_incidence", False)) and bool(p.get("incidence_applied_in_display", False)):
            if "incidence_deg_x" in p or "incidence_deg_y" in p:
                # Dual-axis mode: independent X and Y corrections
                if "incidence_deg_x" in p:
                    actual_scale_x *= self._incidence_factor_from_deg(float(p["incidence_deg_x"]))
                if "incidence_deg_y" in p:
                    actual_scale_y *= self._incidence_factor_from_deg(float(p["incidence_deg_y"]))
            else:
                # Legacy single-axis mode (used by Q-Cal tool)
                fac = self._incidence_factor_from_deg(float(p.get("incidence_deg", 0.0)))
                axis_u = str(p.get("incidence_axis", "X")).strip().upper()
                if axis_u == "Y":
                    actual_scale_y *= fac
                else:
                    actual_scale_x *= fac

        return float(actual_scale_x), float(actual_scale_y)

    @staticmethod
    def _incidence_factor_from_deg(theta_deg: float) -> float:
        """Return incidence correction factor 1/sin(theta); 1.0 for invalid input."""
        try:
            theta = float(theta_deg)
            if theta <= 0.0 or theta >= 180.0:
                return 1.0
            s = np.sin(np.deg2rad(theta))
            if s <= 0:
                return 1.0
            return float(1.0 / s)
        except Exception:
            return 1.0

    # ------------------------------------------------------------------ #
    # Pixel ↔ view conversion (accounts for scale + display origin)      #
    # ------------------------------------------------------------------ #

    def _pixel_to_view(self, px_col: float, px_row: float) -> tuple[float, float]:
        """Convert pixel (col, row) to view (vx, vy).

        With row-major ImageItem: view_x = col direction, view_y = row direction.
        sx scales the col/x axis, sy scales the row/y axis.
        """
        sx, sy = self._effective_display_scales()
        p = self._q_calibration or {}
        orig = p.get("display_origin")
        if orig is not None:
            orig_col, orig_row = float(orig[0]), float(orig[1])
            vx = (px_col - orig_col) * sx + orig_col
            vy = (px_row - orig_row) * sy + orig_row
        else:
            vx = px_col * sx
            vy = px_row * sy
        return float(vx), float(vy)

    def _view_to_pixel(self, vx: float, vy: float) -> tuple[float, float]:
        """Convert view (vx, vy) to pixel (col, row). Inverse of _pixel_to_view."""
        sx, sy = self._effective_display_scales()
        p = self._q_calibration or {}
        orig = p.get("display_origin")
        if orig is not None:
            orig_col, orig_row = float(orig[0]), float(orig[1])
            px_col = (vx - orig_col) / sx + orig_col if sx > 0 else vx
            px_row = (vy - orig_row) / sy + orig_row if sy > 0 else vy
        else:
            px_col = vx / sx if sx > 0 else vx
            px_row = vy / sy if sy > 0 else vy
        return float(px_col), float(px_row)

    def _apply_display_transform(self) -> None:
        """Apply the geometric transform (incidence scale centred on display_origin)."""
        sx, sy = self._effective_display_scales()
        self.image_item.resetTransform()
        if sx != 1.0 or sy != 1.0:
            p = self._q_calibration or {}
            orig = p.get("display_origin")
            tr = QTransform()
            if orig is not None:
                orig_col, orig_row = float(orig[0]), float(orig[1])
                # Scale centred on origin: T(orig_col, orig_row) · S(sx,sy) · T(-orig_col, -orig_row)
                tr.translate(orig_col, orig_row)
                tr.scale(sx, sy)
                tr.translate(-orig_col, -orig_row)
            else:
                tr.scale(sx, sy)
            self.image_item.setTransform(tr)

    # ------------------------------------------------------------------ #

    def apply_incidence_display_correction(self, theta_deg: float, axis: str = "X") -> None:
        """Apply incidence-angle geometry correction to image display transform."""
        if self._q_calibration is None:
            self._q_calibration = {}
        self._q_calibration["use_incidence"] = True
        self._q_calibration["incidence_deg"] = float(theta_deg)
        self._q_calibration["incidence_axis"] = "Y" if str(axis or "X").strip().upper() == "Y" else "X"
        self._q_calibration["incidence_applied_in_display"] = True

        self.view_box.setAspectLocked(True, ratio=1.0)
        self._apply_display_transform()
        self._auto_fit_view()

    def configure_q_tool_mode(self) -> None:
        """Hide controls not needed in q-calibration tool."""
        for w in (
            self.btn_q_calibration,
        ):
            w.setVisible(False)

    def _set_axes_top_right(self, x_label: str, y_label: str) -> None:
        """Set labels/style for all four axes; visibility follows Show Axes."""
        self._axis_label_x = x_label
        self._axis_label_y = y_label
        axis_pen = pg.mkPen(color="w", width=1)
        for axis_name in ("left", "bottom", "top", "right"):
            axis = self.plot_widget.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
            axis.setStyle(showValues=True)
        self.plot_widget.setLabel("bottom", x_label)
        self.plot_widget.setLabel("left", y_label)
        self.plot_widget.setLabel("top", x_label)
        self.plot_widget.setLabel("right", y_label)
        self._apply_show_axes_state()
        self.view_box.setDefaultPadding(0.0)
        self.view_box.autoRange(padding=0.0)

    def _apply_show_axes_state(self) -> None:
        """Apply current Show Axes checkbox state without changing axis mapping."""
        if self.chk_show_axes.isChecked():
            for axis_name in ("left", "bottom", "right", "top"):
                self.plot_widget.showAxis(axis_name)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        else:
            for axis_name in ("left", "bottom", "right", "top"):
                self.plot_widget.hideAxis(axis_name)
            self.plot_widget.showGrid(x=False, y=False)

    def _reset_axis_tick_mapping(self) -> None:
        """Restore default pyqtgraph tick formatters on all axes."""
        for n in ("left", "bottom", "right", "top"):
            axis = self.plot_widget.getAxis(n)
            orig = self._axis_tick_original.get(n)
            if orig is not None:
                axis.tickStrings = orig
        self._axis_linear_map_active = False
        self._axis_linear_x_scale = 1.0
        self._axis_linear_y_scale = 1.0
        self._axis_linear_x_offset = 0.0
        self._axis_linear_y_offset = 0.0
        self._axis_linear_unit = "px"
        self._update_ruler_label()

    def _set_axis_linear_mapping(
        self,
        x_scale: float,
        x_offset: float,
        y_scale: float,
        y_offset: float,
    ) -> None:
        """Map displayed axis values by linear transforms without changing image geometry."""
        def _nice_step(step: float) -> float:
            step = abs(float(step))
            if not np.isfinite(step) or step <= 0:
                return 1.0
            exp = np.floor(np.log10(step))
            base = step / (10.0 ** exp)
            if base < 1.5:
                nice = 1.0
            elif base < 3.5:
                nice = 2.0
            elif base < 7.5:
                nice = 5.0
            else:
                nice = 10.0
            return float(nice * (10.0 ** exp))

        def _make_tick(axis_name: str):
            is_x = axis_name in ("bottom", "top")
            sc = x_scale if is_x else y_scale
            off = x_offset if is_x else y_offset

            def _tick_strings(_self, values, _scale, _spacing):
                q_spacing = _nice_step(float(_spacing) * float(sc))
                if q_spacing <= 0 or not np.isfinite(q_spacing):
                    q_spacing = 1.0
                # decimal digits from nice step (e.g. 0.002 -> 3 decimals)
                decimals = max(0, int(np.ceil(-np.log10(q_spacing)))) if q_spacing < 1 else 0
                decimals = min(decimals, 8)
                out = []
                for v in values:
                    mv = float(v) * float(sc) + float(off)
                    if not np.isfinite(mv):
                        out.append("")
                        continue
                    snapped = np.round(mv / q_spacing) * q_spacing
                    out.append(f"{snapped:.{decimals}f}")
                return out

            return _tick_strings

        for n in ("left", "bottom", "right", "top"):
            axis = self.plot_widget.getAxis(n)
            axis.tickStrings = types.MethodType(_make_tick(n), axis)
        self._axis_linear_map_active = True
        self._axis_linear_x_scale = float(x_scale)
        self._axis_linear_y_scale = float(y_scale)
        self._axis_linear_x_offset = float(x_offset)
        self._axis_linear_y_offset = float(y_offset)
        self._axis_linear_unit = "1/A"
        self._update_ruler_label()

    def set_pixel_axes_top_right(self) -> None:
        """Use pixel coordinates and show top/right axes."""
        if self.data is None:
            return
        self._reset_axis_tick_mapping()
        h, w = int(self.data.shape[0]), int(self.data.shape[1])
        self.image_item.resetTransform()
        rect = QRectF(0.0, 0.0, float(max(1, w)), float(max(1, h)))
        self.image_item.setRect(rect)
        self._set_axes_top_right("X (pixels)", "Y (pixels)")
        self.view_box.setRange(xRange=(rect.left(), rect.right()), yRange=(rect.top(), rect.bottom()), padding=0.0)
        self._update_ruler_label()

    def set_pixel_axes_labels_only(self) -> None:
        """Switch axis mapping back to pixel labels without altering image geometry."""
        self._reset_axis_tick_mapping()
        self._set_axes_top_right("X (pixels)", "Y (pixels)")
        self._update_ruler_label()

    def apply_q_axes_calibration(self, params: dict[str, Any]) -> bool:
        """Map axis labels to q-space without changing image geometry."""
        if self.data is None:
            return False
        try:
            cx = float(params.get("center_x", 0.0))
            cy = float(params.get("center_y", 0.0))
            q0 = self._q_components_at_pixel_float(cx, cy, params)
            qx1 = self._q_components_at_pixel_float(cx + 1.0, cy, params)
            qy1 = self._q_components_at_pixel_float(cx, cy + 1.0, params)
            if q0 is None or qx1 is None or qy1 is None:
                return False

            # Local linearization only for axis label mapping (display aid).
            sx = float(qx1[0] - q0[0])  # d(qx)/dx near center
            sy = float(qy1[1] - q0[1])  # d(qy)/dy near center
            if not np.isfinite(sx) or not np.isfinite(sy) or abs(sx) <= 0 or abs(sy) <= 0:
                return False

            # Keep image geometry unchanged; only remap axis value text.
            self._set_axis_linear_mapping(
                x_scale=float(sx),
                x_offset=float(-cx * sx),
                y_scale=float(sy),
                y_offset=float(-cy * sy),
            )
            self._set_axes_top_right("qx (1/A)", "qy (1/A)")
            self._update_ruler_label()
            return True
        except Exception:
            return False

    def _on_ruler_clicked(self, checked: bool) -> None:
        """Toggle ruler overlay."""
        if checked:
            self._create_ruler()
        else:
            self._remove_ruler()

    def _create_ruler(self) -> None:
        """Create a draggable line ruler with distance label."""
        if self.ruler_roi is not None:
            self._update_ruler_label()
            return
        if self.data is None:
            self.btn_ruler.blockSignals(True)
            self.btn_ruler.setChecked(False)
            self.btn_ruler.blockSignals(False)
            return
        h, w = self.data.shape[:2]
        x0 = float(w) * 0.35
        x1 = float(w) * 0.65
        y = float(h) * 0.5
        self.ruler_roi = pg.LineSegmentROI([[x0, y], [x1, y]], pen=pg.mkPen("y", width=2))
        self.plot_item.addItem(self.ruler_roi)
        self.ruler_text = pg.TextItem("", color="y", anchor=(0.5, 1.2))
        self.plot_item.addItem(self.ruler_text)
        self.ruler_roi.sigRegionChanged.connect(self._update_ruler_label)
        self._update_ruler_label()

    def _remove_ruler(self) -> None:
        """Remove ruler ROI and its label."""
        if self.ruler_roi is not None:
            try:
                self.ruler_roi.sigRegionChanged.disconnect(self._update_ruler_label)
            except Exception:
                pass
            self.plot_item.removeItem(self.ruler_roi)
            self.ruler_roi = None
        if self.ruler_text is not None:
            self.plot_item.removeItem(self.ruler_text)
            self.ruler_text = None

    def _update_ruler_label(self) -> None:
        """Update ruler distance text for pixel or q mapping."""
        if self.ruler_roi is None or self.ruler_text is None:
            return
        try:
            p0, p1 = self.ruler_roi.getSceneHandlePositions()
            if len(p0) == 2 and len(p1) == 2:
                # getSceneHandlePositions returns tuples: (handle, QPointF)
                q0 = p0[1]
                q1 = p1[1]
                v0 = self.view_box.mapSceneToView(q0)
                v1 = self.view_box.mapSceneToView(q1)
                dx = float(v1.x() - v0.x())
                dy = float(v1.y() - v0.y())
            else:
                return
            dist = self._distance_between_points_in_current_unit(
                float(v0.x()), float(v0.y()), float(v1.x()), float(v1.y())
            )
            if self._axis_linear_map_active:
                txt = f"{dist:.5g} {self._axis_linear_unit}"
            else:
                txt = f"{dist:.2f} px"
            mx = (float(v0.x()) + float(v1.x())) * 0.5
            my = (float(v0.y()) + float(v1.y())) * 0.5
            self.ruler_text.setText(txt)
            self.ruler_text.setPos(mx, my)
        except Exception:
            pass

    def _distance_in_current_unit(self, dx: float, dy: float) -> float:
        """Distance in current display unit from component deltas."""
        if self._axis_linear_map_active:
            dx_u = float(dx) * float(self._axis_linear_x_scale)
            dy_u = float(dy) * float(self._axis_linear_y_scale)
            return float(np.hypot(dx_u, dy_u))
        return float(np.hypot(float(dx), float(dy)))

    def _distance_between_points_in_current_unit(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """Distance between two points in pixel or strict q coordinates."""
        if self._axis_linear_map_active and self._q_calibration:
            q0 = self._q_components_at_pixel_float(x0, y0, self._q_calibration)
            q1 = self._q_components_at_pixel_float(x1, y1, self._q_calibration)
            if q0 is not None and q1 is not None:
                # Use in-plane q distance for ruler/line in q display mode.
                return float(np.hypot(q1[0] - q0[0], q1[1] - q0[1]))
        return self._distance_in_current_unit(float(x1) - float(x0), float(y1) - float(y0))

    def _roi_distance_scale_for_direction(self, direction: str = "x") -> float:
        """Return distance scale for ROI x-axis in current mapping mode."""
        if not self._axis_linear_map_active:
            return 1.0
        d = str(direction).lower()
        if d.startswith("y") or d.startswith("v"):
            return float(abs(self._axis_linear_y_scale))
        if d.startswith("r"):  # radial/unknown: isotropic approximation
            return float(0.5 * (abs(self._axis_linear_x_scale) + abs(self._axis_linear_y_scale)))
        return float(abs(self._axis_linear_x_scale))

    def _roi_distance_unit(self) -> str:
        """Return ROI x-axis unit label."""
        return "1/A" if self._axis_linear_map_active else "pixels"

    def _auto_fit_view(self) -> None:
        """Auto-fit the current image according to the axes display mode."""
        if self.chk_show_axes.isChecked():
            self.view_box.autoRange(padding=0)
        else:
            self.view_box.autoRange()

    def _update_colormap(self) -> None:
        """Update the colormap based on selection."""
        colormap_name = self.combo_colormap.currentText()
        invert = self.chk_invert.isChecked()

        try:
            cmap = self._resolve_colormap(colormap_name)
            if cmap is None:
                logging.error("Even fallback colormap 'viridis' is None!")
                return
            cmap = self._maybe_invert_colormap(cmap, invert)

            if cmap is not None:
                self.histogram.gradient.setColorMap(cmap)
                logging.debug(f"Applied colormap: {colormap_name}, inverted: {invert}")
            else:
                logging.error(f"Colormap is None after processing, cannot apply")

        except Exception as e:
            logging.error(f"Failed to set colormap '{colormap_name}': {e}")

    def _resolve_colormap(self, colormap_name: str) -> Optional[pg.ColorMap]:
        candidates = [colormap_name]

        for name in candidates:
            try:
                cmap = pg.colormap.get(name, skipCache=True)
                if cmap is not None:
                    return cmap
            except Exception as exc:
                logging.debug(f"Native colormap lookup failed for '{name}': {exc}")

        for name in candidates:
            try:
                cmap = pg.colormap.get(name, source="matplotlib", skipCache=True)
                if cmap is not None:
                    return cmap
            except Exception as exc:
                logging.debug(f"Matplotlib colormap lookup failed for '{name}': {exc}")

        logging.warning(f"Colormap '{colormap_name}' not found, using 'viridis' as fallback")
        try:
            cmap = pg.colormap.get("viridis", skipCache=True)
            if cmap is not None:
                return cmap
        except Exception as native_exc:
            logging.debug(f"Native fallback colormap lookup failed: {native_exc}")

        try:
            return pg.colormap.get("viridis", source="matplotlib", skipCache=True)
        except Exception as mpl_exc:
            logging.error(f"Matplotlib fallback colormap lookup failed: {mpl_exc}")
            return None

    @staticmethod
    def _maybe_invert_colormap(cmap: pg.ColorMap, invert: bool) -> pg.ColorMap:
        if not invert:
            return cmap
        try:
            reversed_cmap = cmap.reversed()
            if reversed_cmap is not None:
                logging.debug("Successfully reversed colormap using reversed()")
                return reversed_cmap
            logging.debug("reversed() returned None, trying reverse()")
            result = cmap.reverse()
            return result if result is not None else cmap
        except AttributeError as attr_err:
            logging.debug(f"reversed() not available ({attr_err}), trying reverse()")
            try:
                result = cmap.reverse()
                return result if result is not None else cmap
            except Exception as rev_err:
                logging.warning(f"Could not reverse colormap: {rev_err}")
                return cmap
        except Exception as e:
            logging.warning(f"Error during colormap reversal: {e}")
            return cmap

    def _update_scale(self) -> None:
        """Update the scale transformation."""
        self.current_transform = self.combo_scale.currentText()
        self._update_display()

        logging.info(f"Changed 2D scale mode to: {self.current_transform}")

    def _on_show_axes_changed(self, state: int) -> None:
        """Handle show axes checkbox state change."""
        if state:  # Checked - show axes and grid
            # Configure axes styling for dark theme
            axis_pen = pg.mkPen(color='w', width=1)
            for axis_name in ['left', 'bottom', 'right', 'top']:
                axis = self.plot_widget.getAxis(axis_name)
                axis.setPen(axis_pen)
                axis.setTextPen(axis_pen)
                axis.setStyle(showValues=True)

            self.plot_widget.setLabel('left', self._axis_label_y)
            self.plot_widget.setLabel('bottom', self._axis_label_x)
            self.plot_widget.setLabel('right', self._axis_label_y)
            self.plot_widget.setLabel('top', self._axis_label_x)
            self._apply_show_axes_state()

            # Tightly fit axes to image - disable auto-padding
            self.view_box.setDefaultPadding(0.0)

            # Apply tight fit immediately
            self.view_box.autoRange(padding=0)

            logging.info("Enabled coordinate axes (all sides) with grid (tight fit)")
        else:  # Unchecked - hide axes and grid
            self._apply_show_axes_state()

            # Re-enable auto-padding
            self.view_box.setDefaultPadding(0.02)

            # Restore normal auto range
            self.view_box.autoRange()

            logging.info("Disabled coordinate axes and grid")

    def _on_lock_levels_changed(self, state: int) -> None:
        """Handle lock levels checkbox state change."""
        if state:  # Checked - lock current levels
            # Get current levels from the histogram
            levels = self.histogram.getLevels()
            if levels is not None:
                self.locked_levels = levels
                logging.info(f"Locked levels at: [{levels[0]:.3g}, {levels[1]:.3g}]")
        else:  # Unchecked - unlock levels
            self.locked_levels = None
            logging.info("Unlocked levels")
            # Refresh display with auto levels
            self._update_display()

    def _on_slice_changed(self, value: int) -> None:
        """Handle slice slider value change for 3D data (full or lazy)."""
        self.current_slice_index = value
        axis = int(self.current_slice_axis)

        if self.data_3d is not None:
            # Fully loaded in memory
            self.data = self._extract_full_slice(value)
            num_slices = self.data_3d.shape[axis]

        elif self._slice_loader is not None:
            # Lazy mode: serve from local cache or fetch from server
            cache_key = (axis, int(value))
            if cache_key in self._slice_cache:
                self.data = self._slice_cache[cache_key]
            else:
                try:
                    self.data = self._to_float32(self._load_lazy_slice(axis, value))
                    # Keep up to 8 slices cached to avoid repeated network reads
                    if len(self._slice_cache) >= 8:
                        del self._slice_cache[next(iter(self._slice_cache))]
                    self._slice_cache[cache_key] = self.data
                except Exception as e:
                    logging.error(f"Failed to load axis {axis} slice {value}: {e}")
                    return
            num_slices = self._lazy_shape[axis] if self._lazy_shape else self.slider_slice.maximum() + 1

        else:
            return

        self.label_slice_info.setText(f"{value + 1} / {num_slices}")
        self._update_display()

        if self.current_roi is not None:
            self._update_roi_statistics()

        logging.debug(f"Displaying slice {value + 1} of {num_slices}")

    def _on_slice_axis_changed(self, _index: int) -> None:
        """Handle X/Y/Z slice axis changes for 3D data."""
        axis_data = self.combo_slice_axis.currentData()
        if axis_data is None:
            return

        self.current_slice_axis = int(axis_data)
        shape = self.data_3d.shape if self.data_3d is not None else self._lazy_shape
        if not shape or self.current_slice_axis >= len(shape):
            return

        self._reset_slice_slider(shape[self.current_slice_axis])
        try:
            if self.data_3d is not None:
                self.data = self._extract_full_slice(0)
            elif self._slice_loader is not None:
                cache_key = (self.current_slice_axis, 0)
                if cache_key not in self._slice_cache:
                    self._slice_cache[cache_key] = self._to_float32(
                        self._load_lazy_slice(self.current_slice_axis, 0)
                    )
                self.data = self._slice_cache[cache_key]
            self._update_display()
            if self.current_roi is not None:
                self._update_roi_statistics()
        except Exception as e:
            logging.error(f"Failed to switch slice axis to {self.current_slice_axis}: {e}")

    def _on_roi_button_clicked(self, roi_type: str) -> None:
        """Handle ROI button click."""
        # Switching to a standard ROI deactivates radial mode
        if self._radial_mode_active:
            self._deactivate_radial_mode()

        # Get the button that was clicked
        if roi_type == "Line":
            clicked_button = self.btn_roi_line
            other_button = self.btn_roi_rect
        else:  # Rectangle
            clicked_button = self.btn_roi_rect
            other_button = self.btn_roi_line

        # If button is now checked, activate this ROI type
        if clicked_button.isChecked():
            # Uncheck the other button
            other_button.setChecked(False)
            # Activate this ROI type
            self._on_roi_type_changed(roi_type)
        else:
            # Button was unchecked, remove ROI
            self._on_roi_type_changed("None")

    def _on_roi_type_changed(self, roi_type: str) -> None:
        """Handle ROI type selection change."""
        # Remove existing ROI if any
        if self.current_roi is not None:
            self.image_view.view.removeItem(self.current_roi)
            self.current_roi = None

        # Remove sector outline if it exists
        if self.sector_plot_item is not None:
            self.image_view.view.removeItem(self.sector_plot_item)
            self.sector_plot_item = None

        # Remove sector control points if they exist
        self._remove_sector_control_points()

        # Hide statistics when ROI is removed
        if roi_type == "None":
            self.roi_plot_widget.hide()
            self.roi_type = None
            # Uncheck all ROI buttons
            self.btn_roi_line.setChecked(False)
            self.btn_roi_rect.setChecked(False)
            logging.info("ROI removed")
            return

        # Create new ROI based on type
        if self.data is None:
            logging.warning("No image data loaded, cannot create ROI")
            # Uncheck all ROI buttons
            self.btn_roi_line.setChecked(False)
            self.btn_roi_rect.setChecked(False)
            return

        # Get image dimensions (display uses row-major orientation)
        height, width = self.data.shape[0], self.data.shape[1]

        # Create ROI in the center of the image
        center_x, center_y = width // 2, height // 2

        try:
            if roi_type == "Rectangle":
                # Create rectangular ROI
                roi_size = min(width, height) // 4
                self.current_roi = pg.RectROI(
                    [center_x - roi_size // 2, center_y - roi_size // 2],
                    [roi_size, roi_size],
                    pen=pg.mkPen('r', width=2)
                )
                # Add rotation handle (at top-right corner)
                self.current_roi.addRotateHandle([1, 0], [0.5, 0.5])
                self.roi_type = "Rectangle"

            elif roi_type == "Line":
                # Create line ROI
                line_length = min(width, height) // 3
                self.current_roi = pg.LineSegmentROI(
                    [[center_x - line_length // 2, center_y],
                     [center_x + line_length // 2, center_y]],
                    pen=pg.mkPen('r', width=2)
                )
                self.roi_type = "Line"

            if self.current_roi is not None:
                # Add ROI to image view
                self.image_view.view.addItem(self.current_roi)

                # Connect ROI change signal to update statistics
                self.current_roi.sigRegionChanged.connect(self._update_roi_statistics)

                # Show plot widget
                self.roi_plot_widget.show()

                # Initial statistics update
                self._update_roi_statistics()

                logging.info(f"Created {roi_type} ROI")

        except Exception as e:
            logging.error(f"Failed to create ROI: {e}")
            # Uncheck all ROI buttons on error
            self.btn_roi_line.setChecked(False)
            self.btn_roi_rect.setChecked(False)

    def _update_roi_statistics(self) -> None:
        """Update ROI statistics plot display."""
        # For Sector ROI, we can draw the outline even without data
        # For other ROI types, we need both ROI and data
        if self.roi_type != "Sector":
            if self.current_roi is None or self.data is None:
                return

        try:
            # Clear previous plot
            self.roi_plot_widget.clear()

            # Handle Sector ROI separately (it doesn't use getArrayRegion)
            if self.roi_type == "Sector":
                # For sector ROI, calculate radial profile
                # Get center position from sector_center (updated by dragging center marker)
                logging.info(f"Sector ROI statistics update: center={self.sector_center}")

                if self.sector_center is None:
                    logging.warning("Sector center is None, cannot update ROI statistics")
                    return

                center_x, center_y = self.sector_center

                # Always draw sector outline on image (regardless of data availability)
                logging.info(f"About to draw sector outline at ({center_x:.1f}, {center_y:.1f})")
                self._draw_sector_outline(center_x, center_y)

                # Calculate radial profile if data is available
                if self.data is not None:
                    radial_profile = self._calculate_radial_profile(
                        self.data,
                        center_x,
                        center_y,
                        self.sector_radius_inner,
                        self.sector_radius_outer,
                        self.sector_angle_start,
                        self.sector_angle_end
                    )

                    if radial_profile is not None and len(radial_profile) > 0:
                        radii, intensities = radial_profile
                        radii = np.asarray(radii, dtype=np.float64)
                        radii = radii * self._roi_distance_scale_for_direction("radial")

                        # Plot the radial profile
                        pen = pg.mkPen(color='c', width=2)
                        self.roi_plot_widget.plot(radii, intensities, pen=pen, symbol='o', symbolSize=4, symbolBrush='c')

                        # Set labels
                        self.roi_plot_widget.setLabel('left', 'Pixel Intensity', units='')
                        self.roi_plot_widget.setLabel('bottom', 'Distance', units=self._roi_distance_unit())

                        # Set title with statistics
                        title = f"Radial Profile | Angle: {self.sector_angle_start:.0f} deg-{self.sector_angle_end:.0f} deg | " \
                                f"R: {self.sector_radius_inner:.0f}-{self.sector_radius_outer:.0f}px"
                        self.roi_plot_widget.setTitle(title)
                    else:
                        self.roi_plot_widget.setTitle("No data in sector region")
                else:
                    self.roi_plot_widget.setTitle("No image data loaded")

                return  # Done processing Sector ROI

            # For non-sector ROI types, sample directly from the displayed data orientation.
            roi_data = self.current_roi.getArrayRegion(
                self.data,
                self.image_view.imageItem
            )

            if roi_data.size == 0:
                self.roi_plot_widget.setTitle("ROI contains no data")
                return

            # Calculate statistics for title
            mean_val = np.nanmean(roi_data)
            std_val = np.nanstd(roi_data)
            min_val = np.nanmin(roi_data)
            max_val = np.nanmax(roi_data)

            if self.roi_type == "Line":
                # For line ROI, show intensity profile along the line
                # Sample directly along the line in image coordinates to avoid any
                # axis-order ambiguity from getArrayRegion.
                profile = None
                try:
                    pts = self._line_roi_endpoints_image_coords(self.current_roi)
                    if pts is not None:
                        x0, y0, x1, y1 = pts
                        profile = self._sample_line_profile(self.data, x0, y0, x1, y1)
                except Exception:
                    profile = None

                if profile is None:
                    # Fallback to ROI-extracted data if direct sampling fails.
                    arr = np.asarray(roi_data, dtype=np.float64)
                    if arr.ndim == 1:
                        profile = arr
                    else:
                        avg_axis = int(np.argmin(arr.shape))
                        profile = np.nanmean(arr, axis=avg_axis)
                        if np.ndim(profile) > 1:
                            profile = np.asarray(profile).ravel()

                # Create x-axis as actual axis coordinates along the line
                # (pixel: X/Y, q-mode: qx/qy), instead of 0..Length.
                pts_view = self._line_roi_endpoints_view_coords(self.current_roi)
                if pts_view is not None:
                    vx0, vy0, vx1, vy1 = pts_view
                    n = max(2, int(len(profile)))
                    xs = np.linspace(float(vx0), float(vx1), n, dtype=np.float64)
                    ys = np.linspace(float(vy0), float(vy1), n, dtype=np.float64)

                    x_label_text = "X"
                    x_label_unit = "pixels"
                    x_values = None

                    if self._axis_linear_map_active:
                        # Prefer strict q coordinate sampling when available.
                        q_vals = []
                        for xx, yy in zip(xs, ys):
                            q_vals.append(self._q_components_at_pixel_float(float(xx), float(yy), self._q_calibration))

                        if q_vals and all(v is not None for v in q_vals):
                            qx = np.asarray([float(v[0]) for v in q_vals], dtype=np.float64)
                            qy = np.asarray([float(v[1]) for v in q_vals], dtype=np.float64)
                            if abs(float(qx[-1] - qx[0])) >= abs(float(qy[-1] - qy[0])):
                                x_values = qx
                                x_label_text = "qx"
                            else:
                                x_values = qy
                                x_label_text = "qy"
                            x_label_unit = "1/A"
                        else:
                            # Fallback to linear axis mapping if strict q sampling fails.
                            mx = xs * float(self._axis_linear_x_scale) + float(self._axis_linear_x_offset)
                            my = ys * float(self._axis_linear_y_scale) + float(self._axis_linear_y_offset)
                            if abs(float(mx[-1] - mx[0])) >= abs(float(my[-1] - my[0])):
                                x_values = mx
                                x_label_text = "qx"
                            else:
                                x_values = my
                                x_label_text = "qy"
                            x_label_unit = "1/A"
                    else:
                        # Pixel mode: pick dominant displayed axis.
                        if abs(float(xs[-1] - xs[0])) >= abs(float(ys[-1] - ys[0])):
                            x_values = xs
                            x_label_text = "X"
                        else:
                            x_values = ys
                            x_label_text = "Y"
                        x_label_unit = "pixels"
                else:
                    x_values = np.arange(len(profile))
                    x_label_text = "Index"
                    x_label_unit = ""

                # Plot the profile (cyan color for dark theme)
                pen = pg.mkPen(color='c', width=2)
                self.roi_plot_widget.plot(x_values, profile, pen=pen, symbol='o', symbolSize=4, symbolBrush='c')

                # Set labels
                self.roi_plot_widget.setLabel('left', 'Pixel Intensity', units='')
                self.roi_plot_widget.setLabel('bottom', x_label_text, units=x_label_unit)

                # Calculate line length with same ruler-consistent basis.
                if pts_view is not None:
                    vx0, vy0, vx1, vy1 = pts_view
                    profile_length = self._distance_between_points_in_current_unit(vx0, vy0, vx1, vy1)
                else:
                    profile_length = float(np.sqrt(np.sum(np.diff(self.current_roi.getState()['points'], axis=0)**2)))
                title = f"Line Profile | Mean: {mean_val:.3g} | Std: {std_val:.3g} | Length: {profile_length:.3g} {self._roi_distance_unit()}"
                self.roi_plot_widget.setTitle(title)

            elif self.roi_type == "Rectangle":
                # For rectangle ROI, extract a 1D profile along the longer on-screen ROI side.
                # Infer roi_data axis mapping from ROI size to avoid 90-degree orientation mismatch.
                if roi_data.ndim == 2:
                    roi_size = self.current_roi.size()
                    roi_w = max(1.0, float(abs(roi_size.x())))
                    roi_h = max(1.0, float(abs(roi_size.y())))
                    want_horizontal = roi_w >= roi_h

                    d0, d1 = roi_data.shape[0], roi_data.shape[1]
                    # axis0_is_x=True means roi_data.shape[0] corresponds to ROI width (X).
                    score_axis0_x = abs(d0 - roi_w) + abs(d1 - roi_h)
                    score_axis0_y = abs(d0 - roi_h) + abs(d1 - roi_w)
                    axis0_is_x = score_axis0_x <= score_axis0_y

                    if want_horizontal:
                        profile_direction = "horizontal"
                        profile = (
                            np.nanmean(roi_data, axis=1) if axis0_is_x
                            else np.nanmean(roi_data, axis=0)
                        )
                    else:
                        profile_direction = "vertical"
                        profile = (
                            np.nanmean(roi_data, axis=0) if axis0_is_x
                            else np.nanmean(roi_data, axis=1)
                        )
                else:
                    profile = roi_data.flatten()
                    profile_direction = "unknown"

                # Create x-axis (distance) for selected profile direction
                n = max(2, int(len(profile)))
                if roi_data.ndim == 2:
                    length_px = float(roi_w if profile_direction == "horizontal" else roi_h)
                else:
                    length_px = float(max(1, len(profile) - 1))
                scale_dir = self._roi_distance_scale_for_direction(
                    "x" if profile_direction == "horizontal" else ("y" if profile_direction == "vertical" else "radial")
                )
                x_values = np.linspace(0.0, length_px * scale_dir, n)

                # Plot the profile (cyan color for dark theme)
                pen = pg.mkPen(color='c', width=2)
                self.roi_plot_widget.plot(x_values, profile, pen=pen, symbol='o', symbolSize=4, symbolBrush='c')

                # Set labels
                self.roi_plot_widget.setLabel('left', 'Pixel Intensity', units='')
                self.roi_plot_widget.setLabel('bottom', 'Distance', units=self._roi_distance_unit())

                # Set title with statistics
                title = f"Rectangle Profile ({profile_direction}) | Mean: {mean_val:.3g} | Std: {std_val:.3g} | Range: [{min_val:.3g}, {max_val:.3g}]"
                self.roi_plot_widget.setTitle(title)

            logging.debug(f"Updated ROI plot: mean={mean_val:.4g}")

        except Exception as e:
            logging.error(f"Failed to update ROI plot: {e}")
            self.roi_plot_widget.setTitle(f"Error: {e}")

    def _line_roi_endpoints_image_coords(self, roi) -> tuple[float, float, float, float] | None:
        """Return (x0, y0, x1, y1) for a LineSegmentROI in image-item coordinates."""
        if roi is None:
            return None
        try:
            scene_handles = roi.getSceneHandlePositions()
            if scene_handles is None or len(scene_handles) < 2:
                return None
            p0_scene = scene_handles[0][1]
            p1_scene = scene_handles[1][1]
            p0_img = self.image_item.mapFromScene(p0_scene)
            p1_img = self.image_item.mapFromScene(p1_scene)
            return (
                float(p0_img.x()),
                float(p0_img.y()),
                float(p1_img.x()),
                float(p1_img.y()),
            )
        except Exception:
            return None

    def _line_roi_endpoints_view_coords(self, roi) -> tuple[float, float, float, float] | None:
        """Return (x0, y0, x1, y1) for a LineSegmentROI in view coordinates."""
        if roi is None:
            return None
        try:
            scene_handles = roi.getSceneHandlePositions()
            if scene_handles is None or len(scene_handles) < 2:
                return None
            p0_scene = scene_handles[0][1]
            p1_scene = scene_handles[1][1]
            p0_view = self.view_box.mapSceneToView(p0_scene)
            p1_view = self.view_box.mapSceneToView(p1_scene)
            return (
                float(p0_view.x()),
                float(p0_view.y()),
                float(p1_view.x()),
                float(p1_view.y()),
            )
        except Exception:
            return None

    @staticmethod
    def _sample_line_profile(data: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Sample intensity along a line segment using bilinear interpolation."""
        if data is None or data.ndim < 2:
            return np.array([], dtype=np.float64)
        h, w = data.shape[:2]
        length = max(2, int(round(float(np.hypot(x1 - x0, y1 - y0)))))
        xs = np.linspace(x0, x1, length, dtype=np.float64)
        ys = np.linspace(y0, y1, length, dtype=np.float64)

        # Clamp sample points to valid image bounds.
        xs = np.clip(xs, 0.0, max(0.0, w - 1.0))
        ys = np.clip(ys, 0.0, max(0.0, h - 1.0))

        x_floor = np.floor(xs).astype(np.int64)
        y_floor = np.floor(ys).astype(np.int64)
        x_ceil = np.clip(x_floor + 1, 0, w - 1)
        y_ceil = np.clip(y_floor + 1, 0, h - 1)

        wx = xs - x_floor
        wy = ys - y_floor

        v00 = data[y_floor, x_floor].astype(np.float64)
        v10 = data[y_floor, x_ceil].astype(np.float64)
        v01 = data[y_ceil, x_floor].astype(np.float64)
        v11 = data[y_ceil, x_ceil].astype(np.float64)

        # Bilinear interpolation.
        return (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v10
            + (1.0 - wx) * wy * v01
            + wx * wy * v11
        )

    def _on_mouse_moved(self, pos) -> None:
        """Handle mouse movement to display coordinates and value."""
        if self.data is None:
            return

        # Convert scene position to image-local pixel coordinates for data sampling.
        img_point = self.image_item.mapFromScene(pos)
        x, y = img_point.x(), img_point.y()
        # Also keep view coordinates for metric readouts (ruler/ROI/q labels).
        view_point = self.view_box.mapSceneToView(pos)
        xv, yv = float(view_point.x()), float(view_point.y())

        # Handle sector control point dragging if active
        if self.dragging_handle is not None and self.roi_type == "Sector":
            self._update_dragged_handle_position(self.dragging_handle, x, y)

        # Display coordinates in image pixel space
        x_int, y_int = int(np.floor(x)), int(np.floor(y))

        # Check if position is within image bounds
        # Display uses row-major orientation (no transpose in view/data mapping).
        if 0 <= x_int < self.data.shape[1] and 0 <= y_int < self.data.shape[0]:
            # Get value at this pixel directly from [row, col] = [y, x]
            value = self.data[y_int, x_int]

            # Update label
            if self._axis_linear_map_active:
                qcomp = self._q_components_at_pixel_float(xv, yv, self._q_calibration)
                if qcomp is not None:
                    qx, qy, qz = qcomp
                    qabs = float(np.sqrt(qx * qx + qy * qy + qz * qz))
                    self.label_coords.setText(
                        f"qx: {qx:.5g} | qy: {qy:.5g} | Value: {value:.3g} | q: {qabs:.5g} 1/A"
                    )
                else:
                    qx = float(x_int) * self._axis_linear_x_scale + self._axis_linear_x_offset
                    qy = float(y_int) * self._axis_linear_y_scale + self._axis_linear_y_offset
                    self.label_coords.setText(
                        f"qx: {qx:.5g} | qy: {qy:.5g} | Value: {value:.3g}"
                    )
            else:
                self.label_coords.setText(f"X: {x_int} | Y: {y_int} | Value: {value:.3g}")
            self.label_coords.setStyleSheet("color: black; font-size: 9pt;")
        else:
            # Mouse is outside image bounds
            if self._axis_linear_map_active:
                self.label_coords.setText("qx: - | qy: - | Value: -")
            else:
                self.label_coords.setText("X: - | Y: - | Value: -")
            self.label_coords.setStyleSheet("color: gray; font-size: 9pt;")

    def set_q_calibration(self, params: dict | None) -> None:
        """Set q-calibration parameters for cursor readout. None disables q display."""
        self._q_calibration = params if params else None

    def _q_value_at_pixel(self, x_px: int, y_px: int) -> float | None:
        """Compute |q| at pixel coordinate using strict per-pixel vector geometry."""
        comp = self._q_components_at_pixel_float(float(x_px), float(y_px), self._q_calibration)
        if comp is None:
            return None
        qx, qy, qz = comp
        return float(np.sqrt(qx * qx + qy * qy + qz * qz))

    def _q_components_at_pixel_float(
        self,
        x_px: float,
        y_px: float,
        params: dict | None,
    ) -> tuple[float, float, float] | None:
        """Strict per-pixel q components (qx, qy, qz) in 1/A, relative to beam center."""
        p = params
        if not p:
            return None
        try:
            e_ev = float(p.get("energy_ev", 0.0))
            px_um = float(p.get("pixel_um", 0.0))
            dist_mm = float(p.get("distance_mm", 0.0))
            cx = float(p.get("center_x", 0.0))
            cy = float(p.get("center_y", 0.0))
            if e_ev <= 0 or px_um <= 0 or dist_mm <= 0:
                return None

            ai_deg = float(p.get("incidence_deg", 0.0)) if bool(p.get("use_incidence", False)) else 0.0
            ai = float(np.deg2rad(ai_deg))
            inc_axis = str(p.get("incidence_axis", "X")).upper()
            incidence_in_display = bool(p.get("incidence_applied_in_display", False))

            # Coordinates in detector plane, in meters.
            dx = float(x_px) - cx
            dy = float(y_px) - cy
            if bool(p.get("use_incidence", False)) and not incidence_in_display and 0.0 < ai_deg < 180.0:
                fac = self._incidence_factor_from_deg(ai_deg)
                if inc_axis == "Y":
                    dy *= fac
                else:
                    dx *= fac

            xm = dx * px_um * 1e-6
            ym = dy * px_um * 1e-6
            zm = dist_mm * 1e-3

            # Wavevector magnitude in 1/A.
            lambda_a = 12398.4193 / e_ev
            k = 2.0 * np.pi / lambda_a

            # Outgoing ray direction.
            r = np.array([xm, ym, zm], dtype=np.float64)
            rn = np.linalg.norm(r)
            if not np.isfinite(rn) or rn <= 0:
                return None
            sf = r / rn
            kf = k * sf

            # Incident direction (reflection geometry): grazing incidence in XZ or YZ plane.
            if inc_axis == "Y":
                si = np.array([0.0, np.cos(ai), -np.sin(ai)], dtype=np.float64)
            else:
                si = np.array([np.cos(ai), 0.0, -np.sin(ai)], dtype=np.float64)
            ki = k * si

            q = kf - ki

            # Make beam center the zero reference in q-space.
            rc = np.array([0.0, 0.0, zm], dtype=np.float64)
            sfc = rc / np.linalg.norm(rc)
            kfc = k * sfc
            q0 = kfc - ki
            qr = q - q0

            return float(qr[0]), float(qr[1]), float(qr[2])
        except Exception:
            return None

    def _on_mouse_pressed(self, event) -> None:
        """Handle mouse press to start dragging a control point (press-and-hold)."""
        if self.roi_type != "Sector":
            return

        # Only handle left mouse button
        try:
            from PyQt6.QtCore import Qt
            if event.button() != Qt.MouseButton.LeftButton:
                return
        except Exception:
            # Fallback for different Qt versions
            if event.button() != 1:  # 1 is left button
                return

        # Get mouse position in view coordinates
        mouse_point = self.image_view.view.mapSceneToView(event.scenePos())
        mouse_x, mouse_y = mouse_point.x(), mouse_point.y()

        # Check if we pressed near any handle (within 15 pixels in data coordinates)
        click_threshold = 15

        # Check center marker first
        if self.sector_center is not None:
            center_x, center_y = self.sector_center
            distance = np.sqrt((mouse_x - center_x)**2 + (mouse_y - center_y)**2)
            if distance < click_threshold:
                self.dragging_handle = 'center'
                self.drag_start_pos = self.sector_center
                # Disable image panning while dragging control point
                self.image_view.view.setMouseEnabled(x=False, y=False)
                logging.info("Started dragging center")
                return

        # Check other handles
        handle_map = {
            'outer_start': self.sector_handle_outer_start,
            'outer_end': self.sector_handle_outer_end,
            'inner_start': self.sector_handle_inner_start,
            'inner_end': self.sector_handle_inner_end
        }

        for handle_name, handle in handle_map.items():
            if handle is not None:
                try:
                    data = handle.getData()
                    if data and len(data) == 2 and len(data[0]) > 0 and len(data[1]) > 0:
                        handle_x, handle_y = data[0][0], data[1][0]
                        distance = np.sqrt((mouse_x - handle_x)**2 + (mouse_y - handle_y)**2)
                        if distance < click_threshold:
                            self.dragging_handle = handle_name
                            self.drag_start_pos = (handle_x, handle_y)
                            # Disable image panning while dragging control point
                            self.image_view.view.setMouseEnabled(x=False, y=False)
                            logging.info(f"Started dragging {handle_name}")
                            return
                except Exception as exc:
                    logging.debug(f"Failed to inspect ROI handle '{handle_name}': {exc}")

    def _on_mouse_released(self, event) -> None:
        """Handle mouse release to stop dragging (release after press-and-hold)."""
        if self.roi_type != "Sector":
            return

        # Stop dragging when mouse is released
        if self.dragging_handle is not None:
            logging.info(f"Stopped dragging {self.dragging_handle}")
            self.dragging_handle = None
            self.drag_start_pos = None
            # Re-enable image panning after dragging control point
            self.image_view.view.setMouseEnabled(x=True, y=True)

    def _update_dragged_handle_position(self, handle_name: str, new_x: float, new_y: float) -> None:
        """Update the position of a dragged handle and recalculate sector parameters."""
        if self.sector_center is None:
            return

        center_x, center_y = self.sector_center

        # Calculate new sector parameters based on which handle is being dragged
        if handle_name == 'center':
            # Update center position
            self.sector_center = (new_x, new_y)

            # Update center marker position using ScatterPlotItem
            if self.sector_center_marker is not None:
                self.sector_center_marker.setData([new_x], [new_y])

            # Update all control points to follow the new center
            self._update_sector_control_point_positions()

            # Redraw sector outline
            self._draw_sector_outline(new_x, new_y)

            # Update statistics plot
            self._update_roi_statistics()
            return

        elif handle_name == 'outer_start':
            # Update outer radius start angle point
            dx = new_x - center_x
            dy = new_y - center_y
            radius = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))

            self.sector_angle_start = angle
            self.sector_radius_outer = radius

            # Update the scatter plot item position
            if self.sector_handle_outer_start is not None:
                self.sector_handle_outer_start.setData([new_x], [new_y])

        elif handle_name == 'outer_end':
            # Update outer radius end angle point
            dx = new_x - center_x
            dy = new_y - center_y
            radius = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))

            self.sector_angle_end = angle
            self.sector_radius_outer = radius

            # Update the scatter plot item position
            if self.sector_handle_outer_end is not None:
                self.sector_handle_outer_end.setData([new_x], [new_y])

        elif handle_name == 'inner_start':
            # Update inner radius start angle point
            dx = new_x - center_x
            dy = new_y - center_y
            radius = np.sqrt(dx**2 + dy**2)

            self.sector_radius_inner = radius

            # Keep angle locked to start angle from outer handle
            angle_rad = np.radians(self.sector_angle_start)
            actual_x = center_x + radius * np.cos(angle_rad)
            actual_y = center_y + radius * np.sin(angle_rad)

            # Update the scatter plot item position
            if self.sector_handle_inner_start is not None:
                self.sector_handle_inner_start.setData([actual_x], [actual_y])

        elif handle_name == 'inner_end':
            # Update inner radius end angle point
            dx = new_x - center_x
            dy = new_y - center_y
            radius = np.sqrt(dx**2 + dy**2)

            self.sector_radius_inner = radius

            # Keep angle locked to end angle from outer handle
            angle_rad = np.radians(self.sector_angle_end)
            actual_x = center_x + radius * np.cos(angle_rad)
            actual_y = center_y + radius * np.sin(angle_rad)

            # Update the scatter plot item position
            if self.sector_handle_inner_end is not None:
                self.sector_handle_inner_end.setData([actual_x], [actual_y])

        # Ensure inner radius is less than outer radius
        if self.sector_radius_inner >= self.sector_radius_outer:
            self.sector_radius_inner = self.sector_radius_outer * 0.8

        # Update all control points to be on the correct positions
        self._update_sector_control_point_positions()

        # Redraw sector outline
        self._draw_sector_outline(center_x, center_y)

        # Update statistics plot
        self._update_roi_statistics()

    # ------------------------------------------------------------------ #
    # Radial profile tool                                                  #
    # ------------------------------------------------------------------ #

    def _remove_radial_overlays(self) -> None:
        """Remove center marker, arc overlays and corner handles from the view."""
        for attr in (
            "_radial_center_marker", "_radial_arc1", "_radial_arc2",
            "_radial_handle_outer_start", "_radial_handle_outer_end",
            "_radial_handle_inner_start", "_radial_handle_inner_end",
        ):
            item = getattr(self, attr, None)
            if item is not None:
                self.view_box.removeItem(item)
                setattr(self, attr, None)

    @staticmethod
    def _arc_boundary_xy(
        cx: float, cy: float,
        r_min: int, r_max: int,
        a_start_deg: float, a_end_deg: float,
        r_fallback: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closed boundary of one arc sector as (x, y) VIEW-space arrays.

        (cx, cy) and the radii are all in view coordinates so the figure
        always appears as a circle regardless of incidence-correction scaling.
        """
        n = max(3, int(abs(a_end_deg - a_start_deg)) * 2)
        theta = np.linspace(np.radians(a_start_deg), np.radians(a_end_deg), n)
        r_o = r_max if r_max > 0 else r_fallback
        x_out = cx + r_o * np.cos(theta)
        y_out = cy + r_o * np.sin(theta)
        if r_min > 0:
            x_in = cx + r_min * np.cos(theta[::-1])
            y_in = cy + r_min * np.sin(theta[::-1])
        else:
            x_in = np.array([cx])
            y_in = np.array([cy])
        x = np.concatenate([x_out, x_in, [x_out[0]]])
        y = np.concatenate([y_out, y_in, [y_out[0]]])
        return x, y

    def _draw_radial_overlay(
        self, cx: float, cy: float,
        r_min: int, r_max: int,
        angle_min: int, angle_max: int,
    ) -> None:
        """Draw two symmetric arc sector boundaries in VIEW space (always circular)."""
        pen = pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine)
        if self.data is not None:
            h, w = self.data.shape[:2]
            r_fb = float(int(np.ceil(max(
                np.hypot(cx, cy), np.hypot(w - cx, cy),
                np.hypot(cx, h - cy), np.hypot(w - cx, h - cy),
            ))))
        else:
            r_fb = float(max(r_min + 10, 100))

        for i, offset in enumerate([0, 180]):
            x, y = self._arc_boundary_xy(
                cx, cy, r_min, r_max,
                angle_min + offset, angle_max + offset, r_fb,
            )
            attr = f"_radial_arc{i + 1}"
            item = getattr(self, attr)
            if item is None:
                item = pg.PlotDataItem(x, y, pen=pen)
                item.setZValue(9)
                self.view_box.addItem(item)
                setattr(self, attr, item)
            else:
                item.setData(x, y)

        self._place_radial_handles(cx, cy, r_min, r_max, angle_min, angle_max, r_fb)

    def _place_radial_handles(
        self, cx: float, cy: float,
        r_min: int, r_max: int,
        angle_min: int, angle_max: int,
        r_fallback: float,
    ) -> None:
        """Create or update the 4 draggable TargetItem corner handles (view-space positions)."""
        r_o = float(r_max) if r_max > 0 else r_fallback
        r_i = float(r_min)
        a0 = np.radians(angle_min)
        a1 = np.radians(angle_max)
        positions = {
            "_radial_handle_outer_start": (cx + r_o * np.cos(a0), cy + r_o * np.sin(a0)),
            "_radial_handle_outer_end":   (cx + r_o * np.cos(a1), cy + r_o * np.sin(a1)),
            "_radial_handle_inner_start": (cx + r_i * np.cos(a0), cy + r_i * np.sin(a0)),
            "_radial_handle_inner_end":   (cx + r_i * np.cos(a1), cy + r_i * np.sin(a1)),
        }
        pen        = pg.mkPen("w", width=1)
        brush      = pg.mkBrush("w")
        hover_pen  = pg.mkPen("y", width=2)
        hover_brush = pg.mkBrush("y")
        for attr, (hx, hy) in positions.items():
            if attr == self._radial_active_drag_attr:
                continue  # Don't reposition the handle currently being dragged
            handle = getattr(self, attr)
            if handle is None:
                handle = pg.TargetItem(
                    pos=(float(hx), float(hy)),
                    movable=True,
                    symbol="s",
                    size=10,
                    pen=pen,
                    brush=brush,
                    hoverPen=hover_pen,
                    hoverBrush=hover_brush,
                )
                handle.setZValue(12)
                self.view_box.addItem(handle)
                handle_name = attr.replace("_radial_handle_", "")
                handle.sigPositionChanged.connect(
                    lambda item, name=handle_name, a=attr: self._on_radial_handle_moved(item, name, a)
                )
                setattr(self, attr, handle)
            else:
                handle.blockSignals(True)
                handle.setPos(float(hx), float(hy))
                handle.blockSignals(False)

    def _on_radial_handle_moved(self, item, handle_name: str, attr: str) -> None:
        """Called by TargetItem.sigPositionChanged when a corner handle is dragged."""
        if self._radial_center is None:
            return
        pos = item.pos()
        self._radial_active_drag_attr = attr
        self._update_radial_handle_drag(float(pos.x()), float(pos.y()), handle_name)
        self._radial_active_drag_attr = None

    def _update_radial_handle_drag(self, vx: float, vy: float, handle_name: str) -> None:
        """Recompute r/angle from dragged handle position and update everything."""
        if self._radial_center is None:
            return
        cx, cy = self._radial_center
        # Tout en coordonnées vue : le handle est déjà en vue.
        dx, dy = vx - cx, vy - cy
        new_r     = max(0, int(round(float(np.hypot(dx, dy)))))
        new_angle = int(round(float(np.degrees(np.arctan2(dy, dx)))))
        new_angle = max(-180, min(180, new_angle))

        r_min = self._radial_r_min
        r_max = self._radial_r_max
        a_min = self._radial_angle_min
        a_max = self._radial_angle_max

        handle = handle_name
        if handle == "outer_start":
            r_max, a_min = new_r, new_angle
        elif handle == "outer_end":
            r_max, a_max = new_r, new_angle
        elif handle == "inner_start":
            r_min, a_min = new_r, new_angle
        elif handle == "inner_end":
            r_min, a_max = new_r, new_angle

        # Keep r_min < r_max when both non-zero
        if r_max > 0 and r_min >= r_max:
            if "inner" in handle:
                r_min = max(0, r_max - 1)
            else:
                r_max = r_min + 1

        self._radial_r_min = r_min
        self._radial_r_max = r_max
        self._radial_angle_min = a_min
        self._radial_angle_max = a_max

        self._draw_radial_overlay(cx, cy, r_min, r_max, a_min, a_max)
        self.sigRadialParamsChanged.emit(r_min, r_max, a_min, a_max)

    def _on_scene_clicked(self, event) -> None:
        """Place or move the radial-profile center when the image is clicked."""
        if not self._radial_mode_active:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        view_pt = self.view_box.mapSceneToView(event.scenePos())
        # Stocker directement en coordonnées vue : la figure reste ronde.
        self._set_radial_center(view_pt.x(), view_pt.y())
        event.accept()

    def activate_radial_profile(self) -> None:
        """Enable center-click mode (radial overlays are managed by the dialog)."""
        if self.roi_type is not None:
            self._on_roi_type_changed("None")
        self._radial_mode_active = True

    def deactivate_radial_profile(self) -> None:
        """Disable center-click mode and remove overlays."""
        self._deactivate_radial_mode()

    def _set_radial_center(self, cx: float, cy: float) -> None:
        """Place or move the radial-profile center marker.

        (cx, cy) are VIEW-space coordinates: the figure always looks like a
        circle regardless of incidence-correction scaling.
        """
        self._radial_center = (cx, cy)
        if self._radial_center_marker is None:
            self._radial_center_marker = pg.ScatterPlotItem(
                [cx], [cy],
                symbol="+",
                size=20,
                pen=pg.mkPen("r", width=2),
                brush=pg.mkBrush(None),
            )
            self._radial_center_marker.setZValue(10)
            self.view_box.addItem(self._radial_center_marker)
        else:
            self._radial_center_marker.setData([cx], [cy])

        self._draw_radial_overlay(
            cx, cy,
            self._radial_r_min, self._radial_r_max,
            self._radial_angle_min, self._radial_angle_max,
        )
        self.sigRadialCenterChanged.emit(cx, cy)

    def _deactivate_radial_mode(self) -> None:
        """Turn off center-click mode and clean up overlays."""
        self._radial_mode_active = False
        self._radial_center = None
        self._remove_radial_overlays()

    @staticmethod
    def _compute_full_radial_profile(
        data: np.ndarray,
        cx: float,
        cy: float,
        r_min: int = 0,
        r_max: int = 0,
        angle_min: int = 0,
        angle_max: int = 180,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Azimuthal average within two symmetric arc sectors.

        Arc 1: angle_min → angle_max
        Arc 2: (angle_min+180°) → (angle_max+180°)
        Radial range: r_min to r_max (0 = no outer limit).

        Returns (r_bins, mean_intensity) as 1-D float64 arrays.
        """
        h, w = data.shape[:2]
        y_idx, x_idx = np.indices((h, w))
        dx = x_idx.astype(np.float64) - cx
        dy = y_idx.astype(np.float64) - cy
        r = np.sqrt(dx ** 2 + dy ** 2)

        # Angular mask — two symmetric arcs.
        # Normalize everything to 0–360 so comparisons work regardless of sign.
        angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        a0 = float(angle_min) % 360.0
        a1 = float(angle_max) % 360.0
        a0s = (a0 + 180.0) % 360.0
        a1s = (a1 + 180.0) % 360.0

        def _arc_mask(a_start, a_end):
            if a_start <= a_end:
                return (angles >= a_start) & (angles <= a_end)
            # Wrap-around: arc crosses the 0°/360° boundary
            return (angles >= a_start) | (angles <= a_end)

        angle_mask = _arc_mask(a0, a1) | _arc_mask(a0s, a1s)

        # Radial mask
        r_mask = r >= float(r_min)
        if r_max > 0:
            r_mask &= r <= float(r_max)

        combined = angle_mask & r_mask
        if not combined.any():
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        r_int = np.round(r).astype(np.int64)
        flat_r = r_int[combined].ravel()
        flat_data = data[combined].ravel().astype(np.float64)
        n_bins = int(flat_r.max()) + 1
        radial_sum = np.bincount(flat_r, weights=flat_data, minlength=n_bins)
        radial_count = np.bincount(flat_r, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_intensity = np.where(
                radial_count > 0, radial_sum / radial_count, np.nan
            )
        return np.arange(n_bins, dtype=np.float64), mean_intensity

    def _calculate_radial_profile(
        self,
        data: np.ndarray,
        center_x: float,
        center_y: float,
        r_inner: float,
        r_outer: float,
        angle_start: float,
        angle_end: float,
        num_bins: int = 50
    ) -> tuple:
        """
        Calculate radial profile within a sector region.

        Args:
            data: 2D image data
            center_x: X coordinate of sector center
            center_y: Y coordinate of sector center
            r_inner: Inner radius
            r_outer: Outer radius
            angle_start: Start angle in degrees (0 deg is right, counter-clockwise)
            angle_end: End angle in degrees
            num_bins: Number of radial bins

        Returns:
            Tuple of (radii, intensities) arrays for plotting
        """
        height, width = data.shape

        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:height, :width]

        # Calculate distance and angle from center for each pixel
        dx = x_grid - center_x
        dy = y_grid - center_y
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.degrees(np.arctan2(dy, dx))  # -180 to 180 degrees

        # Normalize angles to 0-360 range
        angles = (angles + 360) % 360
        angle_start_norm = angle_start % 360
        angle_end_norm = angle_end % 360

        # Create mask for sector region
        if angle_end_norm > angle_start_norm:
            angle_mask = (angles >= angle_start_norm) & (angles <= angle_end_norm)
        else:
            # Handle wrap-around case (e.g., 350 deg to 10 deg)
            angle_mask = (angles >= angle_start_norm) | (angles <= angle_end_norm)

        radius_mask = (distances >= r_inner) & (distances <= r_outer)
        sector_mask = angle_mask & radius_mask

        # Extract data within sector
        sector_distances = distances[sector_mask]
        sector_intensities = data[sector_mask]

        if len(sector_distances) == 0:
            return None

        # Bin the data radially
        radial_bins = np.linspace(r_inner, r_outer, num_bins + 1)
        bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

        bin_intensities = []
        for i in range(len(radial_bins) - 1):
            r_min, r_max = radial_bins[i], radial_bins[i + 1]
            mask = (sector_distances >= r_min) & (sector_distances < r_max)
            if np.any(mask):
                bin_intensities.append(np.nanmean(sector_intensities[mask]))
            else:
                bin_intensities.append(np.nan)

        return bin_centers, np.array(bin_intensities)

    def _draw_sector_outline(self, center_x: float, center_y: float) -> None:
        """
        Draw sector outline on the image.

        Args:
            center_x: X coordinate of sector center
            center_y: Y coordinate of sector center
        """
        # Remove previous sector outline if it exists
        if self.sector_plot_item is not None:
            self.image_view.view.removeItem(self.sector_plot_item)

        # Generate sector outline points
        angle_start_rad = np.radians(self.sector_angle_start)
        angle_end_rad = np.radians(self.sector_angle_end)

        # Number of points for smooth arcs
        num_points = 50

        # Generate angles for the arcs
        angles = np.linspace(angle_start_rad, angle_end_rad, num_points)

        # Outer arc
        outer_x = center_x + self.sector_radius_outer * np.cos(angles)
        outer_y = center_y + self.sector_radius_outer * np.sin(angles)

        # Inner arc (reversed for continuous outline)
        inner_x = center_x + self.sector_radius_inner * np.cos(angles[::-1])
        inner_y = center_y + self.sector_radius_inner * np.sin(angles[::-1])

        # Combine to form closed outline
        outline_x = np.concatenate([outer_x, inner_x, [outer_x[0]]])
        outline_y = np.concatenate([outer_y, inner_y, [outer_y[0]]])

        # Create plot item for sector outline (red thin solid line, 1.5 pixels)
        pen = pg.mkPen('r', width=1.5)
        self.sector_plot_item = pg.PlotDataItem(
            outline_x, outline_y,
            pen=pen
        )
        self.sector_plot_item.setZValue(0)  # Lower Z-value, behind control points
        self.image_view.view.addItem(self.sector_plot_item)

        logging.info(f"Drew sector outline at center ({center_x:.1f}, {center_y:.1f}), "
                     f"angles {self.sector_angle_start:.0f} deg-{self.sector_angle_end:.0f} deg, "
                     f"radii {self.sector_radius_inner:.0f}-{self.sector_radius_outer:.0f}")

    def _create_sector_control_points(self) -> None:
        """Create interactive control points for sector ROI parameters using ScatterPlotItem."""
        # Remove existing control points if any
        self._remove_sector_control_points()

        # Get center position
        center_x, center_y = self.sector_center

        # Calculate positions for control points
        # Outer radius - start angle
        angle_start_rad = np.radians(self.sector_angle_start)
        x_outer_start = center_x + self.sector_radius_outer * np.cos(angle_start_rad)
        y_outer_start = center_y + self.sector_radius_outer * np.sin(angle_start_rad)

        # Outer radius - end angle
        angle_end_rad = np.radians(self.sector_angle_end)
        x_outer_end = center_x + self.sector_radius_outer * np.cos(angle_end_rad)
        y_outer_end = center_y + self.sector_radius_outer * np.sin(angle_end_rad)

        # Inner radius - start angle
        x_inner_start = center_x + self.sector_radius_inner * np.cos(angle_start_rad)
        y_inner_start = center_y + self.sector_radius_inner * np.sin(angle_start_rad)

        # Inner radius - end angle
        x_inner_end = center_x + self.sector_radius_inner * np.cos(angle_end_rad)
        y_inner_end = center_y + self.sector_radius_inner * np.sin(angle_end_rad)

        # Control point size (fixed pixel size) - reduced to half size
        handle_size = 8

        # Create center marker using ScatterPlotItem (red, size 10 - half of original 20)
        self.sector_center_marker = pg.ScatterPlotItem(
            [center_x], [center_y],
            size=10,
            pen=pg.mkPen('r', width=2),
            brush=pg.mkBrush('r'),
            symbol='o',
            pxMode=True,  # Fixed pixel size, doesn't scale with data
            hoverable=True,
            tip=None
        )
        self.sector_center_marker.tag = 'center'
        self.sector_center_marker.setZValue(10)  # Higher Z-value, above outline

        # Create control points as ScatterPlotItem (fixed pixel size, no handles)
        # Yellow for outer radius points
        self.sector_handle_outer_start = pg.ScatterPlotItem(
            [x_outer_start], [y_outer_start],
            size=handle_size,
            pen=pg.mkPen('y', width=2),
            brush=pg.mkBrush('y'),
            symbol='o',
            pxMode=True,  # Fixed pixel size
            hoverable=True,
            tip=None
        )
        self.sector_handle_outer_start.tag = 'outer_start'
        self.sector_handle_outer_start.setZValue(10)  # Higher Z-value, above outline

        self.sector_handle_outer_end = pg.ScatterPlotItem(
            [x_outer_end], [y_outer_end],
            size=handle_size,
            pen=pg.mkPen('y', width=2),
            brush=pg.mkBrush('y'),
            symbol='o',
            pxMode=True,
            hoverable=True,
            tip=None
        )
        self.sector_handle_outer_end.tag = 'outer_end'
        self.sector_handle_outer_end.setZValue(10)  # Higher Z-value, above outline

        # Green for inner radius points
        self.sector_handle_inner_start = pg.ScatterPlotItem(
            [x_inner_start], [y_inner_start],
            size=handle_size,
            pen=pg.mkPen('g', width=2),
            brush=pg.mkBrush('g'),
            symbol='o',
            pxMode=True,
            hoverable=True,
            tip=None
        )
        self.sector_handle_inner_start.tag = 'inner_start'
        self.sector_handle_inner_start.setZValue(10)  # Higher Z-value, above outline

        self.sector_handle_inner_end = pg.ScatterPlotItem(
            [x_inner_end], [y_inner_end],
            size=handle_size,
            pen=pg.mkPen('g', width=2),
            brush=pg.mkBrush('g'),
            symbol='o',
            pxMode=True,
            hoverable=True,
            tip=None
        )
        self.sector_handle_inner_end.tag = 'inner_end'
        self.sector_handle_inner_end.setZValue(10)  # Higher Z-value, above outline

        # Add control points to the view (center marker first, then the 4 handles)
        self.image_view.view.addItem(self.sector_center_marker)
        self.image_view.view.addItem(self.sector_handle_outer_start)
        self.image_view.view.addItem(self.sector_handle_outer_end)
        self.image_view.view.addItem(self.sector_handle_inner_start)
        self.image_view.view.addItem(self.sector_handle_inner_end)

        # Install event filter on the scene to capture mouse press/release events
        # Remove existing event filter if any
        if self.scene_event_filter is not None:
            self.image_view.scene.removeEventFilter(self.scene_event_filter)

        # Create and install new event filter
        self.scene_event_filter = SceneEventFilter(self)
        self.image_view.scene.installEventFilter(self.scene_event_filter)

    def _remove_sector_control_points(self) -> None:
        """Remove all sector control points from the view."""
        # Remove center marker
        if self.sector_center_marker is not None:
            self.image_view.view.removeItem(self.sector_center_marker)
            self.sector_center_marker = None

        # Remove control points
        if self.sector_handle_outer_start is not None:
            self.image_view.view.removeItem(self.sector_handle_outer_start)
            self.sector_handle_outer_start = None

        if self.sector_handle_outer_end is not None:
            self.image_view.view.removeItem(self.sector_handle_outer_end)
            self.sector_handle_outer_end = None

        if self.sector_handle_inner_start is not None:
            self.image_view.view.removeItem(self.sector_handle_inner_start)
            self.sector_handle_inner_start = None

        if self.sector_handle_inner_end is not None:
            self.image_view.view.removeItem(self.sector_handle_inner_end)
            self.sector_handle_inner_end = None

        # Remove event filter if it exists
        if self.scene_event_filter is not None:
            self.image_view.scene.removeEventFilter(self.scene_event_filter)
            self.scene_event_filter = None

        # Reset dragging state
        self.dragging_handle = None
        self.drag_start_pos = None


    def _update_sector_control_point_positions(self) -> None:
        """Update positions of control points to match current sector parameters."""
        if self.sector_center is None:
            return

        center_x, center_y = self.sector_center

        # Calculate positions for control points
        angle_start_rad = np.radians(self.sector_angle_start)
        angle_end_rad = np.radians(self.sector_angle_end)

        # Outer radius - start angle
        x_outer_start = center_x + self.sector_radius_outer * np.cos(angle_start_rad)
        y_outer_start = center_y + self.sector_radius_outer * np.sin(angle_start_rad)

        # Outer radius - end angle
        x_outer_end = center_x + self.sector_radius_outer * np.cos(angle_end_rad)
        y_outer_end = center_y + self.sector_radius_outer * np.sin(angle_end_rad)

        # Inner radius - start angle
        x_inner_start = center_x + self.sector_radius_inner * np.cos(angle_start_rad)
        y_inner_start = center_y + self.sector_radius_inner * np.sin(angle_start_rad)

        # Inner radius - end angle
        x_inner_end = center_x + self.sector_radius_inner * np.cos(angle_end_rad)
        y_inner_end = center_y + self.sector_radius_inner * np.sin(angle_end_rad)

        # Update control point positions using setData (for ScatterPlotItem)
        # Update center marker first
        if self.sector_center_marker is not None:
            self.sector_center_marker.setData([center_x], [center_y])

        if self.sector_handle_outer_start is not None:
            self.sector_handle_outer_start.setData([x_outer_start], [y_outer_start])

        if self.sector_handle_outer_end is not None:
            self.sector_handle_outer_end.setData([x_outer_end], [y_outer_end])

        if self.sector_handle_inner_start is not None:
            self.sector_handle_inner_start.setData([x_inner_start], [y_inner_start])

        if self.sector_handle_inner_end is not None:
            self.sector_handle_inner_end.setData([x_inner_end], [y_inner_end])




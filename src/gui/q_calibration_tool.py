"""Standalone q-calibration tool with FTH-style layout."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtCore import QBuffer, QByteArray, QMimeData, QSettings
from PyQt6.QtGui import QClipboard, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.gui.dataset_path_combo import DatasetPathCombo
from src.gui.export_naming import (
    remember_save_directory,
    short_series_label,
    suggested_save_path,
)
from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
from src.gui._shared import profile_pen, roi_pen, set_axis_label
from src.gui.plot_context_menu import attach_plot_menu
from src.lib_h5.data_exporter import DataExporter
from src.lib_h5.table_format import (
    DEFAULT_TABLE_FORMAT_KEY,
    format_from_filter,
    get_table_format,
    save_dialog_filter,
)
from src.lib_h5.table_writer import write_table
from src.recon import profiles as pr


# How wide the status line may get. It sits in the settings panel, and the panel
# is sized by the controls above it — not by whatever the last message happened
# to say.
STATUS_MAX_WIDTH = 420


class QCalibrationTool(QDialog):
    """Q-calibration workspace: left controls + right 2D image."""

    def __init__(
        self,
        opened_files,
        dataset_full_keys_2d: list[str] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._opened_files = opened_files
        self._dataset_full_keys_2d = dataset_full_keys_2d or []
        self._data: np.ndarray | None = None       # working image (incidence-corrected)
        self._raw_data: np.ndarray | None = None    # as-loaded, before incidence resampling
        self._pending_center_pick = False
        self._incident_applied = False
        # What was done to reduce a stack of frames to one pattern, for the
        # status line to report. See _flatten_stack.
        self._stack_notes: list[str] = []

        # ROI subsystem (Ring / Sector / Circle + radial / azimuthal profiles).
        self._N_ANGLE_BINS = 360
        self._rois: list[dict] = []
        self._roi_counter = {"ring": 0, "sector": 0, "circle": 0}
        self._roi_overlay_items: list = []
        self._roi_handles: dict = {}
        self._grid_cache = None
        self._polar_cache = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._compute_current_profiles)

        self.setWindowTitle("Scattering Pattern Analyze")
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1320, 820)

        self._build_ui()
        self._populate_dataset_combo()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left panel
        left = QWidget()
        # Ignored horizontally, as the calculator's controls panel is: the
        # splitter's own sizes then decide the width, and nothing inside — a
        # long dataset path, a status message, the frame selectors appearing —
        # can push it wider on its own. Without this the panel is whatever its
        # widest child happens to want, which changes as data is loaded.
        left.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        left_lay = QVBoxLayout(left)
        # Zero inside the splitter panels; the dialog's own margin is the only
        # inset, matching the calculator and comparison windows.
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)

        g_ds = QGroupBox("Input Datasets  (CL / CR / BG)")
        fl_ds = QFormLayout(g_ds)
        self._cl_combo = DatasetPathCombo("-- no CL dataset --")
        self._cr_combo = DatasetPathCombo("-- no CR dataset --")
        self._bg_combo = DatasetPathCombo("-- no BG (optional) --")
        for combo in (self._cl_combo, self._cr_combo, self._bg_combo):
            combo.lineEdit().returnPressed.connect(self._load_data)
            # A stack in any slot brings up the frame controls, before loading:
            # the choice has to be available *while* choosing what to load.
            combo.currentTextChanged.connect(self._refresh_frame_controls)
        self._combos = {"CL": self._cl_combo, "CR": self._cr_combo, "BG": self._bg_combo}
        fl_ds.addRow("CL:", self._cl_combo)
        fl_ds.addRow("CR:", self._cr_combo)
        fl_ds.addRow("BG:", self._bg_combo)

        self._combo_operation = QComboBox()
        self._combo_operation.addItem("CL − BG", "cl_bg")
        self._combo_operation.addItem("CR − BG", "cr_bg")
        self._combo_operation.addItem("Sum  (CL−BG)+(CR−BG)", "sum")
        self._combo_operation.addItem("Difference  (CL−BG)−(CR−BG)", "diff")
        self._combo_operation.addItem("Asymmetry  diff / sum", "asym")
        self._combo_operation.setToolTip(
            "How CL / CR / BG combine into the 2-D image to calibrate.\n"
            "BG defaults to zero when not loaded."
        )
        fl_ds.addRow("Operation:", self._combo_operation)

        # Which frame of a stack to calibrate. One control for all three slots:
        # they are required to have the same shape anyway, and three copies of
        # the same question would be three chances to answer it inconsistently.
        self._row_frames = QHBoxLayout()
        self._lbl_frames = QLabel("Axis:")
        self._combo_frame_axis = QComboBox()
        self._combo_frame_axis.setToolTip("Which array axis the frames are counted along")
        self._combo_frame_axis.setMinimumWidth(55)
        self._combo_frame_axis.currentIndexChanged.connect(self._on_frame_axis_changed)
        self._combo_frame = QComboBox()
        self._combo_frame.setMinimumWidth(90)
        self._combo_frame.setToolTip(
            "Which frame to calibrate, or the mean of them all.\n"
            "The mean is the default: more frames, better statistics."
        )
        self._combo_frame.currentIndexChanged.connect(self._on_frame_changed)
        self._row_frames.addWidget(self._lbl_frames)
        self._row_frames.addWidget(self._combo_frame_axis)
        self._row_frames.addWidget(QLabel("Frame:"))
        self._row_frames.addWidget(self._combo_frame)
        self._row_frames.addStretch()
        self._frames_label = QLabel("Frames:")
        fl_ds.addRow(self._frames_label, self._row_frames)
        self._show_frame_controls(False)

        self._btn_load = QPushButton("Load Data")
        self._btn_load.clicked.connect(self._load_data)
        fl_ds.addRow("", self._btn_load)
        left_lay.addWidget(g_ds)

        g_inc = QGroupBox("Incident Correction")
        fl_inc = QFormLayout(g_inc)
        self._spin_inc_deg = QDoubleSpinBox()
        self._spin_inc_deg.setRange(0.0, 89.9999)
        self._spin_inc_deg.setDecimals(4)
        self._spin_inc_deg.setValue(0.0)
        self._spin_inc_deg.setSuffix(" deg")
        self._spin_inc_deg.lineEdit().returnPressed.connect(self._apply_incident_cali)
        self._combo_inc_axis = QComboBox()
        self._combo_inc_axis.addItem("X", userData="X")
        self._combo_inc_axis.addItem("Y", userData="Y")
        self._combo_inc_axis.setCurrentText("X")
        row_inc = QHBoxLayout()
        row_inc.addWidget(self._spin_inc_deg)
        row_inc.addWidget(self._combo_inc_axis)
        fl_inc.addRow("Incident angle:", row_inc)
        self._btn_apply_inc = QPushButton("Resize image")
        self._btn_apply_inc.clicked.connect(self._apply_incident_cali)
        fl_inc.addRow("", self._btn_apply_inc)
        left_lay.addWidget(g_inc)

        g_center = QGroupBox("Center")
        fl_center = QFormLayout(g_center)
        self._btn_pick_center = QPushButton("Click to Set Center")
        self._btn_pick_center.setCheckable(True)
        self._btn_pick_center.toggled.connect(self._on_pick_center_toggled)
        self._btn_apply_center = QPushButton("Apply Center")
        self._btn_apply_center.clicked.connect(self._on_apply_center)
        row_center_btn = QHBoxLayout()
        row_center_btn.addWidget(self._btn_pick_center)
        row_center_btn.addWidget(self._btn_apply_center)
        fl_center.addRow(row_center_btn)

        self._center_row = QSpinBox()
        self._center_row.setRange(0, 99999)
        self._center_col = QSpinBox()
        self._center_col.setRange(0, 99999)
        self._center_row.valueChanged.connect(self._on_center_changed)
        self._center_col.valueChanged.connect(self._on_center_changed)
        fl_center.addRow("Center row:", self._center_row)
        fl_center.addRow("Center col:", self._center_col)
        left_lay.addWidget(g_center)

        g_geo = QGroupBox("Q Geometry")
        fl_geo = QFormLayout(g_geo)
        self._spin_energy = QDoubleSpinBox()
        self._spin_energy.setRange(0.0, 50000.0)
        self._spin_energy.setDecimals(4)
        self._spin_energy.setValue(779.0)
        self._spin_energy.setSuffix(" eV")
        fl_geo.addRow("Energy:", self._spin_energy)

        self._spin_pixel = QDoubleSpinBox()
        self._spin_pixel.setRange(0.001, 10000.0)
        self._spin_pixel.setDecimals(4)
        self._spin_pixel.setValue(13.5)
        self._spin_pixel.setSuffix(" um")
        fl_geo.addRow("Pixel size:", self._spin_pixel)

        self._spin_dist = QDoubleSpinBox()
        self._spin_dist.setRange(0.001, 100000.0)
        self._spin_dist.setDecimals(4)
        self._spin_dist.setValue(260.0)
        self._spin_dist.setSuffix(" mm")
        fl_geo.addRow("Distance:", self._spin_dist)

        # The axis switch belongs to the geometry it is computed from.
        row_action = QHBoxLayout()
        self._btn_apply = QPushButton("to q")
        self._btn_apply.setToolTip("Convert the image axes to q using the geometry above")
        self._btn_apply.clicked.connect(self._apply_calibration)
        row_action.addWidget(self._btn_apply)
        self._btn_disable = QPushButton("to pixel")
        self._btn_disable.setToolTip("Switch the image axes back to pixels")
        self._btn_disable.clicked.connect(self._disable_q)
        row_action.addWidget(self._btn_disable)
        fl_geo.addRow(row_action)

        left_lay.addWidget(g_geo)

        left_lay.addWidget(self._build_roi_group())

        for btn in (
            self._btn_load,
            self._btn_apply_inc,
            self._btn_pick_center,
            self._btn_apply_center,
            self._btn_apply,
            self._btn_disable,
        ):
            btn.setAutoDefault(False)
            btn.setDefault(False)

        self._status = QLabel("Ready")
        self._status.setStyleSheet("color: #444;")
        # A label asks for the full width of its text, and the messages here are
        # unbounded — a saved path, or one note per loaded stack. Left free, the
        # first "Loaded: … mean of 400 frames …" doubled the panel and squeezed
        # the image beside it to a sliver. Capped and elided instead, with the
        # whole message on hover.
        self._status.setMaximumWidth(STATUS_MAX_WIDTH)
        self._status.setMinimumWidth(0)
        left_lay.addWidget(self._status)
        left_lay.addStretch(1)

        # Right panel: image on top, ROI profile plot below (vertical splitter).
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)
        right_split = QSplitter(Qt.Orientation.Vertical)
        self._right_split = right_split
        right_lay.addWidget(right_split)

        self._img = ImageView2DEnhanced(parent=self)
        self._img.configure_q_tool_mode()
        self._img.chk_show_axes.setChecked(True)
        self._configure_image_toolbar_actions()
        right_split.addWidget(self._img)

        self._profile_plot = pg.PlotWidget()
        self._profile_plot.setBackground("k")
        self._profile_plot.showGrid(x=True, y=True, alpha=0.3)
        for axis in ("left", "bottom"):
            self._profile_plot.getAxis(axis).setPen(pg.mkPen("w"))
            self._profile_plot.getAxis(axis).setTextPen(pg.mkPen("w"))
        set_axis_label(self._profile_plot, "bottom", "r (px)")
        set_axis_label(self._profile_plot, "left", "Mean intensity")
        self._profile_curve = self._profile_plot.plot([], [], pen=profile_pen())
        # The only way to get this curve out of the tool; it has no buttons of
        # its own, and pyqtgraph's own Export bypasses the naming and dialect
        # rules every other export in the application follows.
        attach_plot_menu(
            self._profile_plot,
            on_export=self._export_profile,
            on_plot=self._plot_profile_window,
        )
        right_split.addWidget(self._profile_plot)
        right_split.setSizes([620, 200])
        # Hidden until the user starts a ROI (shown by _update_profile_visibility).
        self._profile_plot.setVisible(False)

        # Center overlays (same row/col convention as FTH).
        self._center_cross = pg.ScatterPlotItem(
            x=[],
            y=[],
            symbol="+",
            size=14,
            pen=pg.mkPen("y", width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self._center_circle = pg.ScatterPlotItem(
            x=[],
            y=[],
            symbol="o",
            size=20,
            pen=pg.mkPen("y", width=1.5),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self._img.plot_item.addItem(self._center_cross)
        self._img.plot_item.addItem(self._center_circle)
        self._img.image_view.scene.sigMouseClicked.connect(self._on_image_clicked)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([360, 960])

    def _configure_image_toolbar_actions(self) -> None:
        """Use the image toolbar buttons for q-calibration copy/save actions."""
        for btn, handler, tooltip in (
            (self._img.btn_copy_image, self._copy_image, "Copy current Q calibration image with the active colormap"),
            (self._img.btn_save_image, self._save_image, "Save current Q calibration image"),
        ):
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(handler)
            btn.setToolTip(tooltip)
            btn.setAutoDefault(False)
            btn.setDefault(False)

    def _set_status(self, text: str, error: bool = False) -> None:
        """Show a status message without letting it resize the panel.

        Elided from the middle: these messages carry the useful part at both
        ends — what happened at the front, the file name or the frame count at
        the back — and cutting from the right would drop the half that says how
        it turned out.
        """
        message = str(text or "")
        metrics = self._status.fontMetrics()
        self._status.setText(
            metrics.elidedText(message, Qt.TextElideMode.ElideMiddle, STATUS_MAX_WIDTH)
        )
        self._status.setToolTip(message)
        self._status.setStyleSheet("color: #b00020;" if error else "color: #444;")

    def _populate_dataset_combo(self) -> None:
        for combo in self._combos.values():
            combo.populate_from_full_keys(
                self._dataset_full_keys_2d,
                opened_files=self._opened_files,
            )

    def refresh_dataset_keys(self, keys_2d: list[str]) -> None:
        """Refresh available dataset list from shared index."""
        self._dataset_full_keys_2d = list(keys_2d)
        self._populate_dataset_combo()

    def set_opened_files(self, opened_files) -> None:
        """Sync latest opened-files tuple from main window."""
        self._opened_files = opened_files

    def load_dataset_full_key(self, full_key: str, auto_load: bool = True, slot: str = "CL") -> bool:
        """Select a dataset by full key into a slot (default CL) and optionally load.

        The key usually comes from whatever is selected in the tree, which is
        often a *group* — opening the tool then greeted the user with "Dataset
        not found" about a path that plainly exists. The selection is still
        filled in either way, because it is a useful starting point; only the
        automatic load is held back, and the reason goes on the status line
        rather than into a message box the user did not ask for.
        """
        if not full_key or "::" not in full_key:
            return False
        combo = self._combos.get(slot, self._cl_combo)
        combo.add_full_key(full_key, select=True)

        if auto_load:
            problem = self._why_not_an_image(full_key)
            if problem:
                self._set_status(problem, error=True)
                return False
            self._load_data()
        return True

    def _why_not_an_image(self, full_key: str) -> str:
        """Say why a key cannot be loaded as a pattern, or "" when it can."""
        try:
            file_part, ds_path = str(full_key).split("::", 1)
            ds_path = ds_path.strip("/")
            with h5py.File(file_part, "r") as h5:
                if ds_path not in h5:
                    return f"No such dataset: {ds_path}"
                obj = h5[ds_path]
                if not isinstance(obj, h5py.Dataset):
                    return f"{ds_path} is a group — pick a dataset inside it."
                if len(obj.shape) < 2:
                    return f"{ds_path} is {len(obj.shape)}-D; this tool needs an image."
        except Exception as exc:
            return f"Could not read {full_key}: {exc}"
        return ""

    # ── Frame selection for a stack ───────────────────────────────────── #

    #: The frame selector's value for "average them all".
    MEAN_OF_FRAMES = -1

    def _show_frame_controls(self, visible: bool) -> None:
        self._frames_label.setVisible(visible)
        self._lbl_frames.setVisible(visible)
        self._combo_frame_axis.setVisible(visible)
        self._combo_frame.setVisible(visible)
        for index in range(self._row_frames.count()):
            item = self._row_frames.itemAt(index)
            widget = item.widget() if item is not None else None
            if isinstance(widget, QLabel):
                widget.setVisible(visible)

    def _stack_shape(self) -> tuple[int, ...] | None:
        """The shape of the first selected slot, if it is a stack of frames."""
        for slot in ("CL", "CR", "BG"):
            entry = self._combos[slot].get_entry(opened_files=self._opened_files)
            if entry is None:
                continue
            fp, ds = entry
            try:
                with h5py.File(fp, "r") as h5:
                    if ds in h5 and isinstance(h5[ds], h5py.Dataset):
                        shape = tuple(h5[ds].shape)
                        return shape if len(shape) >= 3 else None
            except Exception as exc:
                logging.debug("Could not read the shape of %s: %s", ds, exc)
        return None

    def _refresh_frame_controls(self, *_args) -> None:
        """Show the axis and frame selectors when a slot holds a stack."""
        shape = self._stack_shape()
        self._show_frame_controls(shape is not None)
        if shape is None:
            return

        previous_axis = self._combo_frame_axis.currentData()
        self._combo_frame_axis.blockSignals(True)
        self._combo_frame_axis.clear()
        for axis in range(len(shape)):
            self._combo_frame_axis.addItem(str(axis), axis)
        if previous_axis is not None and 0 <= int(previous_axis) < len(shape):
            self._combo_frame_axis.setCurrentIndex(int(previous_axis))
        self._combo_frame_axis.blockSignals(False)
        self._rebuild_frame_combo(shape)

    def _rebuild_frame_combo(self, shape: tuple[int, ...]) -> None:
        axis = int(self._combo_frame_axis.currentData() or 0)
        previous = self._combo_frame.currentData()
        self._combo_frame.blockSignals(True)
        self._combo_frame.clear()
        self._combo_frame.addItem("Mean", self.MEAN_OF_FRAMES)
        for index in range(shape[axis % len(shape)]):
            self._combo_frame.addItem(str(index), index)
        if previous is not None:
            found = self._combo_frame.findData(previous)
            self._combo_frame.setCurrentIndex(max(0, found))
        self._combo_frame.blockSignals(False)

    def _on_frame_axis_changed(self, _index: int) -> None:
        shape = self._stack_shape()
        if shape is not None:
            self._rebuild_frame_combo(shape)
            self._reload_if_loaded()

    def _on_frame_changed(self, _index: int) -> None:
        self._reload_if_loaded()

    def _reload_if_loaded(self) -> None:
        """Re-read the slots so the change is visible without pressing Load."""
        if self._data is not None:
            self._load_data()

    def _frame_selection(self) -> tuple[int, int]:
        """``(axis, index)``; index :data:`MEAN_OF_FRAMES` means average them."""
        if not self._combo_frame.isVisibleTo(self):
            return 0, self.MEAN_OF_FRAMES
        axis = self._combo_frame_axis.currentData()
        index = self._combo_frame.currentData()
        return (int(axis) if axis is not None else 0,
                int(index) if index is not None else self.MEAN_OF_FRAMES)

    def _read_slot_2d(self, slot: str) -> np.ndarray | None:
        """Read a slot's dataset as a 2-D image, averaging a stack of frames."""
        entry = self._combos[slot].get_entry(opened_files=self._opened_files)
        if entry is None:
            return None
        fp, ds = entry
        with h5py.File(fp, "r") as h5:
            if ds not in h5:
                raise KeyError(f"{slot}: no such dataset: {ds}")
            # A group exists but is not readable as an image; saying "not found"
            # about a path the user can see in the tree sends them looking for
            # the wrong problem.
            if not isinstance(h5[ds], h5py.Dataset):
                raise KeyError(f"{slot}: {ds} is a group — pick a dataset inside it.")
            arr = np.asarray(h5[ds][()], dtype=np.float32)
        arr = self._flatten_stack(arr, slot)
        if arr.ndim != 2:
            raise ValueError(f"{slot}: expected a 2-D image but got shape {arr.shape}.")
        return arr

    def _flatten_stack(self, arr: np.ndarray, slot: str) -> np.ndarray:
        """Reduce a stack of frames to the one pattern to calibrate.

        Either the mean of the frames — a reasonable default for a scattering
        pattern, more frames meaning better statistics — or a single frame the
        user picked. Whichever it was is recorded, because it used to happen in
        silence: a 400-frame stack arrived, the status line said
        "Loaded (200x46)", and nothing said that what was on screen was the
        mean of all 400.

        The axis is the one chosen beside the slots, defaulting to the first,
        the way frames are counted everywhere else in the application. The
        previous rule reshaped on the last two axes, which quietly produced
        nonsense for a stack stored as ``(H, W, N)`` — a layout the 2-D viewer
        explicitly supports.
        """
        if arr.ndim <= 2:
            return arr

        axis, index = self._frame_selection()
        axis %= arr.ndim
        frames = int(arr.shape[axis])
        if index >= 0:
            index = min(index, frames - 1)
            self._stack_notes.append(f"{slot}: frame {index} of {frames} (axis {axis})")
            return np.asarray(np.take(arr, index, axis=axis))

        self._stack_notes.append(f"{slot}: mean of {frames} frames (axis {axis})")
        return np.asarray(arr.mean(axis=axis))

    # Operation spec: (required slots, slots used in formula).
    _OP_REQUIRED = {
        "cl_bg": ["CL"], "cr_bg": ["CR"],
        "diff": ["CL", "CR"], "sum": ["CL", "CR"], "asym": ["CL", "CR"],
    }
    _OP_LABELS = {
        "cl_bg": "CL − BG", "cr_bg": "CR − BG",
        "diff": "Difference (CL−BG)−(CR−BG)", "sum": "Sum (CL−BG)+(CR−BG)",
        "asym": "Asymmetry diff/sum",
    }

    def _load_data(self) -> None:
        op = self._combo_operation.currentData()
        self._stack_notes = []
        try:
            for slot in self._OP_REQUIRED[op]:
                if self._combos[slot].get_entry(opened_files=self._opened_files) is None:
                    QMessageBox.warning(self, "Scattering Pattern Analyze", f"Select a valid {slot} dataset.")
                    return
            cl = self._read_slot_2d("CL")
            cr = self._read_slot_2d("CR")
            bg = self._read_slot_2d("BG")
        except Exception as exc:
            logging.exception("Q calibration load failed")
            QMessageBox.critical(self, "Scattering Pattern Analyze", f"Failed to load dataset:\n{exc}")
            return

        shapes = [a.shape for a in (cl, cr, bg) if a is not None]
        if shapes and any(s != shapes[0] for s in shapes):
            QMessageBox.warning(self, "Scattering Pattern Analyze", f"CL / CR / BG shapes differ: {shapes}.")
            return
        if bg is None and shapes:
            bg = np.zeros(shapes[0], dtype=np.float32)

        if op == "cl_bg":
            data = cl - bg
        elif op == "cr_bg":
            data = cr - bg
        elif op == "sum":
            data = (cl - bg) + (cr - bg)
        elif op == "diff":
            data = (cl - bg) - (cr - bg)
        else:  # asym
            d, s = (cl - bg) - (cr - bg), (cl - bg) + (cr - bg)
            with np.errstate(invalid="ignore", divide="ignore"):
                data = np.where(np.abs(s) > 1e-12, d / s, np.nan)

        self._set_loaded_array(np.asarray(data, dtype=np.float32), self._OP_LABELS[op])

    def load_array_data(self, arr: np.ndarray, source_label: str = "calculation_result") -> bool:
        """Load 2D data from memory (no file dataset required)."""
        try:
            data = np.asarray(arr)
            self._stack_notes = []
            if data.ndim > 2:
                # The same reduction the file path uses. It used to take frame
                # zero here and the mean there, so the identical stack gave two
                # different patterns depending on how it had been handed over.
                data = self._flatten_stack(data, "Array")
            if data.ndim != 2:
                QMessageBox.warning(
                    self,
                    "Scattering Pattern Analyze",
                    f"Only 2D arrays are supported.\nCurrent shape: {data.shape}",
                )
                return False
            self._set_loaded_array(np.asarray(data), str(source_label))
            return True
        except Exception as exc:
            logging.exception("Q calibration memory load failed")
            QMessageBox.critical(self, "Scattering Pattern Analyze", f"Failed to load result array:\n{exc}")
            return False

    def _set_loaded_array(self, arr2d: np.ndarray, source_label: str) -> None:
        """Apply loaded 2D array to the viewer and reset calibration state."""
        self._raw_data = np.asarray(arr2d)
        self._data = self._raw_data
        self._img.set_data(self._data)
        self._img.set_pixel_axes_top_right()
        rows, cols = int(self._data.shape[0]), int(self._data.shape[1])
        row_prev_block = self._center_row.blockSignals(True)
        col_prev_block = self._center_col.blockSignals(True)
        try:
            self._center_row.setValue(rows // 2)
            self._center_col.setValue(cols // 2)
        finally:
            self._center_row.blockSignals(row_prev_block)
            self._center_col.blockSignals(col_prev_block)
        # New geometry: drop existing ROIs and resize the radius sliders.
        self._clear_all_rois()
        r_max = int(np.ceil(np.hypot(rows, cols)))
        self._sl_ri.setMaximum(r_max)
        self._sl_ro.setMaximum(r_max)
        self._update_center_overlay()
        self._redraw_roi_overlays()
        self._incident_applied = False
        # Any frame averaging is named here rather than left for the user to
        # infer from a shape that no longer matches the dataset they picked.
        notes = "  |  " + ", ".join(self._stack_notes) if self._stack_notes else ""
        self._set_status(f"Loaded: {source_label} ({rows}x{cols}){notes}")

    def _on_pick_center_toggled(self, checked: bool) -> None:
        self._pending_center_pick = checked
        self._btn_pick_center.setText(
            ">>> Click on image to set center <<<" if checked else "Click to Set Center"
        )

    def _on_apply_center(self) -> None:
        if self._btn_pick_center.isChecked():
            self._btn_pick_center.setChecked(False)
        self._update_center_overlay()
        self._set_status(
            f"Center applied: row={self._center_row.value()}, col={self._center_col.value()}"
        )

    def _on_image_clicked(self, event) -> None:
        if not self._pending_center_pick or self._data is None:
            return
        try:
            pos = event.scenePos()
            vb = self._img.plot_item.getViewBox()
            if vb is None or not vb.sceneBoundingRect().contains(pos):
                return

            # Use display coordinates (after transform) so center follows
            # the same coordinate space as incident-corrected display.
            pt = vb.mapSceneToView(pos)
            col = int(round(float(pt.x())))
            row = int(round(float(pt.y())))
            if row < 0 or col < 0:
                return
            self._center_row.setValue(row)
            self._center_col.setValue(col)
            self._set_status(
                f"Center pending: row={row}, col={col}. Click 'Apply Center' to confirm."
            )
        except Exception as exc:
            logging.debug("q calibration click handling failed: %s", exc)

    def _update_center_overlay(self) -> None:
        if self._data is None:
            self._center_cross.setData([], [])
            self._center_circle.setData([], [])
            return
        row = float(self._center_row.value())
        col = float(self._center_col.value())
        self._center_cross.setData([col], [row])
        self._center_circle.setData([col], [row])

    def _on_center_changed(self) -> None:
        """Center is the ROI origin: redraw the center marker and ROI overlays."""
        self._update_center_overlay()
        self._redraw_roi_overlays()
        self._position_roi_handles()
        self._schedule_refresh()

    # ================================================================== #
    # ROI subsystem (Ring / Sector / Circle + radial / azimuthal)          #
    # ================================================================== #

    def _build_roi_group(self) -> QGroupBox:
        g = QGroupBox("ROI  (radial / azimuthal profile)")
        lay = QVBoxLayout(g)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        row_add = QHBoxLayout()
        for kind in ("ring", "sector", "circle"):
            b = QPushButton(f"+ {kind.capitalize()}")
            b.setAutoDefault(False)
            b.setDefault(False)
            b.clicked.connect(lambda _=False, k=kind: self._add_roi(k))
            row_add.addWidget(b)
        lay.addLayout(row_add)

        row_sel = QHBoxLayout()
        self._roi_combo = QComboBox()
        self._roi_combo.currentIndexChanged.connect(self._on_roi_combo_changed)
        row_sel.addWidget(self._roi_combo, 1)
        self._btn_roi_remove = QPushButton("Remove")
        self._btn_roi_remove.setAutoDefault(False)
        self._btn_roi_remove.setDefault(False)
        self._btn_roi_remove.clicked.connect(self._remove_selected_roi)
        row_sel.addWidget(self._btn_roi_remove)
        lay.addLayout(row_sel)

        self._roi_params = QWidget()
        fp = QFormLayout(self._roi_params)
        fp.setContentsMargins(0, 2, 0, 0)
        fp.setSpacing(4)

        def _slider_row(rng):
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(*rng)
            lbl = QLabel("0")
            lbl.setFixedWidth(56)
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(sl, 1)
            h.addWidget(lbl)
            return sl, lbl, w

        self._sl_ri, self._lbl_ri, self._row_ri = _slider_row((0, 5000))
        self._sl_ro, self._lbl_ro, self._row_ro = _slider_row((0, 5000))
        self._sl_amin, self._lbl_amin, row_amin = _slider_row((-180, 180))
        self._sl_amax, self._lbl_amax, row_amax = _slider_row((-180, 360))
        for sl in (self._sl_ri, self._sl_ro, self._sl_amin, self._sl_amax):
            sl.valueChanged.connect(self._on_roi_param_changed)

        self._cap_inner = QLabel("Inner r:")
        self._cap_outer = QLabel("Outer r:")
        fp.addRow(self._cap_inner, self._row_ri)
        fp.addRow(self._cap_outer, self._row_ro)

        self._roi_angles = QWidget()
        fa = QFormLayout(self._roi_angles)
        fa.setContentsMargins(0, 0, 0, 0)
        fa.setSpacing(4)
        fa.addRow("Angle min:", row_amin)
        fa.addRow("Angle max:", row_amax)
        fp.addRow(self._roi_angles)

        self._cap_mode = QLabel("Profile:")
        self._combo_roi_mode = QComboBox()
        self._combo_roi_mode.addItem("Radial  I(r)", "radial")
        self._combo_roi_mode.addItem("Azimuthal  I(θ)", "azimuthal")
        self._combo_roi_mode.currentIndexChanged.connect(self._on_roi_param_changed)
        fp.addRow(self._cap_mode, self._combo_roi_mode)

        # Phase shift: rolls the azimuthal profile along θ (only shown for I(θ)).
        self._cap_phase = QLabel("Phase shift:")
        self._spin_phase = QSpinBox()
        self._spin_phase.setRange(-180, 180)
        self._spin_phase.setSingleStep(5)
        self._spin_phase.setSuffix("°")
        self._spin_phase.setToolTip("Offset the azimuthal profile along the angle axis.")
        self._spin_phase.valueChanged.connect(self._on_roi_param_changed)
        fp.addRow(self._cap_phase, self._spin_phase)

        self._roi_circle_hint = QLabel("Drag the ellipse handles to size the spot.")
        self._roi_circle_hint.setStyleSheet("color:#666; font-style:italic;")
        self._roi_circle_hint.setWordWrap(True)
        fp.addRow(self._roi_circle_hint)

        self._roi_params.setVisible(False)
        lay.addWidget(self._roi_params)
        return g

    def _center_pixel(self) -> tuple[float, float]:
        """ROI center in pixel (col=x, row=y)."""
        return float(self._center_col.value()), float(self._center_row.value())

    # ── add / remove / select ─────────────────────────────────────────── #

    def _add_roi(self, kind: str) -> None:
        if self._data is None:
            QMessageBox.warning(self, "Scattering Pattern Analyze", "Load data before adding a ROI.")
            return
        h, w = int(self._data.shape[0]), int(self._data.shape[1])
        self._roi_counter[kind] += 1
        name = f"{kind.capitalize()} {self._roi_counter[kind]}"
        if kind == "circle":
            radius = max(3, int(0.10 * min(h, w)))
            roi = {"type": "circle", "name": name,
                   "cx": w / 2.0, "cy": h / 2.0, "rx": float(radius), "ry": float(radius),
                   "mode": "radial", "item": None}
            self._rois.append(roi)
            self._create_circle_item(roi)
        else:
            ri = max(1, int(0.12 * min(h, w)))
            ro = max(ri + 1, int(0.30 * min(h, w)))
            self._rois.append({"type": kind, "name": name,
                               "r_inner": ri, "r_outer": ro,
                               "a_min": 0, "a_max": (180 if kind == "ring" else 60),
                               "mode": "radial", "phase": 0})
        self._roi_combo.addItem(name)
        self._roi_combo.setCurrentIndex(self._roi_combo.count() - 1)

    def _remove_selected_roi(self) -> None:
        i = self._roi_combo.currentIndex()
        if not (0 <= i < len(self._rois)):
            return
        self._remove_circle_item(self._rois[i])
        self._rois.pop(i)
        self._roi_combo.blockSignals(True)
        self._roi_combo.removeItem(i)
        self._roi_combo.blockSignals(False)
        self._on_roi_combo_changed(self._roi_combo.currentIndex())

    def _selected_roi(self) -> dict | None:
        i = self._roi_combo.currentIndex()
        return self._rois[i] if 0 <= i < len(self._rois) else None

    def _clear_all_rois(self) -> None:
        for roi in self._rois:
            self._remove_circle_item(roi)
        self._clear_roi_handles()
        self._rois = []
        self._roi_combo.blockSignals(True)
        self._roi_combo.clear()
        self._roi_combo.blockSignals(False)
        self._roi_params.setVisible(False)
        self._profile_curve.setData([], [])
        self._update_profile_visibility()

    def _on_roi_combo_changed(self, _idx: int = -1) -> None:
        roi = self._selected_roi()
        if roi is None:
            self._roi_params.setVisible(False)
            self._rebuild_roi_handles()
            self._refresh_after_geometry()
            return
        self._roi_params.setVisible(True)
        is_sector = roi["type"] == "sector"
        is_circle = roi["type"] == "circle"
        self._cap_inner.setVisible(not is_circle)
        self._row_ri.setVisible(not is_circle)
        self._roi_angles.setVisible(is_sector)
        self._cap_mode.setVisible(not is_circle)
        self._combo_roi_mode.setVisible(not is_circle)
        self._roi_circle_hint.setVisible(is_circle)
        self._cap_outer.setText("Radius:" if is_circle else "Outer r:")

        widgets = (self._sl_ri, self._sl_ro, self._sl_amin, self._sl_amax,
                   self._combo_roi_mode, self._spin_phase)
        for wdg in widgets:
            wdg.blockSignals(True)
        if is_circle:
            self._sl_ro.setValue(int(round((roi["rx"] + roi["ry"]) / 2.0)))
        else:
            self._sl_ri.setValue(int(roi["r_inner"]))
            self._sl_ro.setValue(int(roi["r_outer"]))
            self._sl_amin.setValue(int(roi["a_min"]))
            self._sl_amax.setValue(int(roi["a_max"]))
            self._combo_roi_mode.setCurrentIndex(0 if roi["mode"] == "radial" else 1)
            self._spin_phase.setValue(int(roi.get("phase", 0)))
        for wdg in widgets:
            wdg.blockSignals(False)
        self._update_param_labels()
        self._update_phase_visibility(roi)
        self._rebuild_roi_handles()
        self._refresh_after_geometry()

    def _update_phase_visibility(self, roi: dict | None) -> None:
        """Phase shift applies only to the azimuthal profile of a ring/sector."""
        show = (roi is not None and roi.get("type") in ("ring", "sector")
                and roi.get("mode") == "azimuthal")
        self._cap_phase.setVisible(show)
        self._spin_phase.setVisible(show)

    def _on_roi_param_changed(self, *_a) -> None:
        roi = self._selected_roi()
        if roi is None:
            return
        if roi["type"] == "circle":
            roi["rx"] = roi["ry"] = float(max(1, self._sl_ro.value()))
            self._update_param_labels()
            self._sync_circle_item(roi)
            self._refresh_after_geometry()
            return
        ri, ro = self._sl_ri.value(), self._sl_ro.value()
        if ro > 0 and ri >= ro:
            self._sl_ri.blockSignals(True)
            self._sl_ri.setValue(max(0, ro - 1))
            self._sl_ri.blockSignals(False)
            ri = self._sl_ri.value()
        roi.update(r_inner=ri, r_outer=ro,
                   a_min=self._sl_amin.value(), a_max=self._sl_amax.value(),
                   mode=self._combo_roi_mode.currentData(),
                   phase=int(self._spin_phase.value()))
        self._update_param_labels()
        self._update_phase_visibility(roi)
        self._refresh_after_geometry()

    def _update_param_labels(self) -> None:
        self._lbl_ri.setText(f"{self._sl_ri.value()} px")
        ro = self._sl_ro.value()
        self._lbl_ro.setText(f"{ro} px" if ro > 0 else "0 (max)")
        self._lbl_amin.setText(f"{self._sl_amin.value()}°")
        self._lbl_amax.setText(f"{self._sl_amax.value()}°")

    def _refresh_after_geometry(self) -> None:
        self._redraw_roi_overlays()
        self._position_roi_handles()
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        self._refresh_timer.start()

    # ── Circle ROI interactive item ───────────────────────────────────── #

    def _create_circle_item(self, roi: dict) -> None:
        cx, cy, rx, ry = roi["cx"], roi["cy"], roi["rx"], roi["ry"]
        item = pg.EllipseROI([cx - rx, cy - ry], [2 * rx, 2 * ry],
                             pen=roi_pen(selected=False), movable=True)
        item.setZValue(16)
        item.sigRegionChanged.connect(lambda _it, ro=roi: self._on_circle_roi_changed(ro))
        self._img.view_box.addItem(item)
        roi["item"] = item

    def _sync_circle_item(self, roi: dict) -> None:
        item = roi.get("item")
        if item is None:
            return
        cx, cy = roi["cx"], roi["cy"]
        rx, ry = max(1.0, roi["rx"]), max(1.0, roi["ry"])
        item.blockSignals(True)
        item.setPos([cx - rx, cy - ry])
        item.setSize([2 * rx, 2 * ry])
        item.blockSignals(False)

    def _on_circle_roi_changed(self, roi: dict) -> None:
        item = roi.get("item")
        if item is None:
            return
        pos, size = item.pos(), item.size()
        rx, ry = float(size.x()) / 2.0, float(size.y()) / 2.0
        roi["cx"], roi["cy"] = float(pos.x()) + rx, float(pos.y()) + ry
        roi["rx"], roi["ry"] = max(1.0, rx), max(1.0, ry)
        if self._selected_roi() is roi:
            self._sl_ro.blockSignals(True)
            self._sl_ro.setValue(int(round((roi["rx"] + roi["ry"]) / 2.0)))
            self._sl_ro.blockSignals(False)
            self._lbl_ro.setText(f"{int(round(roi['rx']))}×{int(round(roi['ry']))} px")
        self._schedule_refresh()

    def _remove_circle_item(self, roi: dict) -> None:
        item = roi.get("item")
        if item is not None:
            try:
                self._img.view_box.removeItem(item)
            except Exception:
                pass
            roi["item"] = None

    # ── ROI overlays ──────────────────────────────────────────────────── #

    @staticmethod
    def _arc_pixel_xy(cx, cy, r, a0=0.0, a1=360.0):
        th = np.deg2rad(np.linspace(a0, a1, 181))
        return cx + r * np.cos(th), cy + r * np.sin(th)

    def _add_curve(self, vx, vy, pen) -> None:
        item = pg.PlotCurveItem(vx, vy, pen=pen)
        item.setZValue(15)
        self._img.view_box.addItem(item)
        self._roi_overlay_items.append(item)

    def _update_profile_visibility(self) -> None:
        """Show the profile window only once at least one ROI exists.

        A hidden QSplitter child collapses to 0 px, so on show we restore a
        sensible split or the plot would appear empty.
        """
        show = len(self._rois) > 0
        self._profile_plot.setVisible(show)
        if show:
            sizes = self._right_split.sizes()
            if len(sizes) == 2 and sizes[1] < 40:
                total = sum(sizes) or 800
                self._right_split.setSizes([int(total * 0.72), int(total * 0.28)])

    def _redraw_roi_overlays(self) -> None:
        self._update_profile_visibility()
        for item in self._roi_overlay_items:
            try:
                self._img.view_box.removeItem(item)
            except Exception:
                pass
        self._roi_overlay_items = []
        if self._data is None:
            return
        cx, cy = self._center_pixel()
        sel = self._roi_combo.currentIndex()
        for i, roi in enumerate(self._rois):
            selected = (i == sel)
            if roi["type"] == "circle":
                self._sync_circle_item(roi)
                item = roi.get("item")
                if item is not None:
                    item.setPen(roi_pen(selected))
            else:
                self._draw_one_roi(roi, cx, cy, selected)

    def _draw_one_roi(self, roi: dict, cx: float, cy: float, selected: bool) -> None:
        pen = roi_pen(selected, width=2.0 if selected else 1.2)
        ri, ro = float(roi["r_inner"]), float(roi["r_outer"])
        if roi["type"] == "ring":
            for r in (ri, ro):
                if r > 0:
                    vx, vy = self._arc_pixel_xy(cx, cy, r)
                    self._add_curve(vx, vy, pen)
        elif roi["type"] == "sector":
            a0 = float(roi["a_min"]) % 360.0
            a1 = float(roi["a_max"]) % 360.0
            if a1 <= a0:
                a1 += 360.0
            for r in (ri, ro):
                if r > 0:
                    vx, vy = self._arc_pixel_xy(cx, cy, r, a0, a1)
                    self._add_curve(vx, vy, pen)
            for ang in (a0, a1):
                th = np.deg2rad(ang)
                self._add_curve(
                    np.array([cx + ri * np.cos(th), cx + ro * np.cos(th)]),
                    np.array([cy + ri * np.sin(th), cy + ro * np.sin(th)]), pen)

    # ── Ring / Sector drag handles ────────────────────────────────────── #

    def _clear_roi_handles(self) -> None:
        for ti in self._roi_handles.values():
            try:
                self._img.view_box.removeItem(ti)
            except Exception:
                pass
        self._roi_handles = {}

    def _handle_target_pos(self, role: str, roi: dict) -> tuple[float, float]:
        cx, cy = self._center_pixel()
        ri, ro = float(roi["r_inner"]), float(roi["r_outer"])
        if roi["type"] == "ring":
            r = ri if role == "r_inner" else ro
            return cx + r, cy
        a0 = float(roi["a_min"]) % 360.0
        a1 = float(roi["a_max"]) % 360.0
        if a1 <= a0:
            a1 += 360.0
        if role == "r_inner":
            ang, rr = (a0 + a1) / 2.0, ri
        elif role == "r_outer":
            ang, rr = (a0 + a1) / 2.0, ro
        elif role == "a_min":
            ang, rr = a0, ro
        else:
            ang, rr = a1, ro
        th = np.deg2rad(ang)
        return cx + rr * np.cos(th), cy + rr * np.sin(th)

    def _rebuild_roi_handles(self) -> None:
        self._clear_roi_handles()
        roi = self._selected_roi()
        if roi is None or roi["type"] not in ("ring", "sector") or self._data is None:
            return
        roles = ["r_inner", "r_outer"]
        if roi["type"] == "sector":
            roles += ["a_min", "a_max"]
        for role in roles:
            x, y = self._handle_target_pos(role, roi)
            ti = pg.TargetItem(pos=(x, y), size=11, movable=True,
                               pen=pg.mkPen("y", width=1.5),
                               brush=pg.mkBrush(255, 255, 0, 130))
            ti.setZValue(22)
            self._img.view_box.addItem(ti)
            ti.sigPositionChanged.connect(lambda *_a, r=role: self._on_handle_dragged(r))
            self._roi_handles[role] = ti

    def _position_roi_handles(self, skip: str | None = None) -> None:
        roi = self._selected_roi()
        if roi is None:
            return
        for role, ti in self._roi_handles.items():
            if role == skip:
                continue
            x, y = self._handle_target_pos(role, roi)
            ti.blockSignals(True)
            ti.setPos((x, y))
            ti.blockSignals(False)

    def _sync_param_sliders_from_roi(self, roi: dict) -> None:
        for wdg in (self._sl_ri, self._sl_ro, self._sl_amin, self._sl_amax):
            wdg.blockSignals(True)
        self._sl_ri.setValue(int(roi["r_inner"]))
        self._sl_ro.setValue(int(roi["r_outer"]))
        self._sl_amin.setValue(int(roi["a_min"]))
        self._sl_amax.setValue(int(roi["a_max"]))
        for wdg in (self._sl_ri, self._sl_ro, self._sl_amin, self._sl_amax):
            wdg.blockSignals(False)
        self._update_param_labels()

    def _on_handle_dragged(self, role: str) -> None:
        roi = self._selected_roi()
        ti = self._roi_handles.get(role)
        if roi is None or ti is None:
            return
        cx, cy = self._center_pixel()
        p = ti.pos()
        dx, dy = float(p.x()) - cx, float(p.y()) - cy
        if role in ("r_inner", "r_outer"):
            new_r = float(np.hypot(dx, dy))
            if role == "r_inner":
                roi["r_inner"] = max(0.0, min(new_r, roi["r_outer"] - 1))
            else:
                roi["r_outer"] = max(new_r, roi["r_inner"] + 1)
        else:
            roi[role] = float(np.degrees(np.arctan2(dy, dx)))
        self._sync_param_sliders_from_roi(roi)
        self._redraw_roi_overlays()
        self._position_roi_handles(skip=role)
        self._schedule_refresh()

    # ── Cached geometry + profile extraction ──────────────────────────── #

    def _grid(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        if self._grid_cache is None or self._grid_cache[0] != (h, w):
            yy, xx = np.indices((h, w))
            self._grid_cache = ((h, w), yy.astype(np.float64), xx.astype(np.float64))
        return self._grid_cache[1], self._grid_cache[2]

    def _polar(self, h: int, w: int, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
        key = (h, w, round(float(cx), 2), round(float(cy), 2))
        if self._polar_cache is None or self._polar_cache[0] != key:
            r, ang = pr.polar_grids(h, w, cx, cy)
            self._polar_cache = (key, r, ang)
        return self._polar_cache[1], self._polar_cache[2]

    def _circle_semi_axes(self, roi: dict) -> tuple[float, float]:
        return max(1.0, float(roi.get("rx", 1.0))), max(1.0, float(roi.get("ry", 1.0)))

    def _circle_mask(self, roi: dict, h: int, w: int) -> np.ndarray:
        yy, xx = self._grid(h, w)
        rx, ry = self._circle_semi_axes(roi)
        return ((xx - roi["cx"]) / rx) ** 2 + ((yy - roi["cy"]) / ry) ** 2 <= 1.0

    def _roi_profiles(self, roi: dict, frame: np.ndarray):
        """Return (r_x, r_y, ang_x, ang_y) for a ROI on the 2-D frame."""
        h, w = frame.shape[:2]
        if roi["type"] == "circle":
            cx, cy = roi["cx"], roi["cy"]
            rx_a, ry_a = self._circle_semi_axes(roi)
            ro = int(np.ceil(max(rx_a, ry_a)))
            frame = np.where(self._circle_mask(roi, h, w), frame, np.nan)
            r, angles = self._polar(h, w, cx, cy)
            rx, ry = pr.radial_profile(frame, cx, cy, 0, ro, 0, 180, symmetric=True, r=r, angles=angles)
            ax, ay = pr.angular_profile(frame, cx, cy, 0, ro, self._N_ANGLE_BINS, r=r, angles=angles)
            return rx, ry, ax, ay
        cx, cy = self._center_pixel()
        r, angles = self._polar(h, w, cx, cy)
        ri, ro = int(roi["r_inner"]), int(roi["r_outer"])
        if roi["type"] == "ring":
            rx, ry = pr.radial_profile(frame, cx, cy, ri, ro, 0, 180, symmetric=True, r=r, angles=angles)
        else:
            rx, ry = pr.radial_profile(frame, cx, cy, ri, ro, int(roi["a_min"]), int(roi["a_max"]),
                                       symmetric=False, r=r, angles=angles)
        ax, ay = pr.angular_profile(frame, cx, cy, ri, ro, self._N_ANGLE_BINS, r=r, angles=angles)
        return rx, ry, ax, ay

    def _compute_current_profiles(self) -> None:
        """Extract the selected ROI's profile and show it in the profile plot."""
        roi = self._selected_roi()
        if roi is None or self._data is None:
            self._profile_curve.setData([], [])
            return
        try:
            rx, ry, ax, ay = self._roi_profiles(roi, np.asarray(self._data, dtype=np.float64))
        except Exception as exc:
            logging.debug("q calibration profile failed: %s", exc)
            self._profile_curve.setData([], [])
            return
        mode = roi.get("mode", "radial")
        if roi["type"] != "circle" and mode == "azimuthal":
            ay = self._apply_phase_shift(np.asarray(ay), int(roi.get("phase", 0)))
            # An azimuthal profile is already an angle; there is nothing for a
            # q calibration to convert it to.
            x, y, xlabel = ax, ay, "θ (deg)"
        else:
            x, xlabel = self._radial_axis(roi, np.asarray(rx, dtype=np.float64))
            y = ry
        finite = np.isfinite(np.asarray(y)) & np.isfinite(np.asarray(x))
        set_axis_label(self._profile_plot, "bottom", xlabel)
        self._profile_curve.setData(np.asarray(x)[finite], np.asarray(y)[finite])

    def _radial_axis(self, roi: dict, radii_px: np.ndarray) -> tuple[np.ndarray, str]:
        """The radial profile's x axis: pixel radius, or |q| once calibrated.

        The image axes switch to q when a calibration is applied, but the ROI
        profile below them went on being drawn against a pixel radius — so the
        two halves of the same window were in different units, and the profile
        could not be read off against the pattern it came from.

        Converted the way the line ROI does it: the q of a real pixel, sampled
        here along the ROI's own mid-angle. That direction matters, because an
        incidence correction makes |q| at a given radius depend on it.
        """
        params = getattr(self._img, "_q_calibration", None)
        if not getattr(self._img, "_axis_linear_map_active", False) or not params:
            return radii_px, "r (px)"

        cx, cy = self._center_pixel()
        if roi.get("type") == "circle":
            angle = 0.0
        else:
            angle = np.deg2rad((float(roi.get("a_min", 0)) + float(roi.get("a_max", 0))) / 2.0)
        dx, dy = float(np.cos(angle)), float(np.sin(angle))

        q_values = []
        for radius in radii_px:
            comp = self._img._q_components_at_pixel_float(
                cx + radius * dx, cy + radius * dy, params
            )
            q_values.append(float(np.hypot(comp[0], comp[1])) if comp else np.nan)

        q_axis = np.asarray(q_values, dtype=np.float64)
        if not np.any(np.isfinite(q_axis)):
            # An incomplete calibration is not a reason to show nothing.
            return radii_px, "r (px)"
        return q_axis, "q (1/A)"

    # ── Getting the profile out ───────────────────────────────────────── #

    def _profile_columns(self) -> tuple[list[str], list[np.ndarray]] | None:
        """The curve on screen as ``(headers, columns)``, or None if there is none.

        Headed with the axis actually in use, so a file exported in q says q and
        one exported in pixels says pixels. The profile had no way out at all
        before this: the image had Copy and Save, the curve computed from it had
        nothing.
        """
        x, y = self._profile_curve.getData()
        if x is None or y is None or len(x) == 0:
            return None
        roi = self._selected_roi()
        name = str(roi.get("name", "profile")) if roi else "profile"
        x_label = self._profile_plot.getAxis("bottom").labelText or "x"
        return [x_label, f"{name} (mean intensity)"], [np.asarray(x), np.asarray(y)]

    def _export_profile(self) -> None:
        """Write the ROI profile through the application's own export rules."""
        columns = self._profile_columns()
        if columns is None:
            self._set_status("No profile to export — add a ROI first.", error=True)
            return
        headers, data = columns

        settings = QSettings()
        fmt = get_table_format(str(settings.value("export/table_format", DEFAULT_TABLE_FORMAT_KEY)))
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export ROI Profile",
            suggested_save_path(self._default_save_name("profile"), extension=fmt.suffix),
            save_dialog_filter(fmt),
        )
        if not path:
            return

        fmt = format_from_filter(selected) or fmt
        out = pathlib.Path(path)
        if not out.suffix:
            out = out.with_suffix(fmt.suffix)
        try:
            write_table(out, headers, data, fmt)
        except Exception as exc:
            logging.error("ROI profile export failed: %s", exc)
            QMessageBox.critical(self, "Export Failed", f"Failed to export the profile:\n{exc}")
            return

        remember_save_directory(out)
        settings.setValue("export/table_format", fmt.key)
        self._set_status(f"Profile exported to {out.name}")

    def _plot_profile_window(self) -> None:
        """Open the ROI profile in a Plot window."""
        from src.gui.plot_dialog import open_plot_dialog
        from src.gui.plot_series import Series

        columns = self._profile_columns()
        if columns is None:
            self._set_status("No profile to plot — add a ROI first.", error=True)
            return
        headers, data = columns
        dialog = open_plot_dialog(
            self,
            [Series(headers[1], data[1], data[0])],
            title="Scattering Pattern Analyze",
        )
        if dialog is not None:
            dialog.le_x_label.setText(headers[0])

    def _apply_phase_shift(self, ay: np.ndarray, phase_deg: int) -> np.ndarray:
        """Roll the azimuthal profile along θ by phase_deg (bins span 0..360)."""
        if not phase_deg or ay.size == 0:
            return ay
        shift = int(round(phase_deg * ay.size / 360.0))
        return np.roll(ay, shift)

    def _collect_q_params(self) -> dict[str, float | bool | str]:
        incid = float(self._spin_inc_deg.value())
        incid_axis = str(self._combo_inc_axis.currentText() or "X").strip().upper()
        applied = bool(self._incident_applied)
        return {
            "energy_ev": float(self._spin_energy.value()),
            "pixel_um": float(self._spin_pixel.value()),
            "distance_mm": float(self._spin_dist.value()),
            "center_x": float(self._center_col.value()),
            "center_y": float(self._center_row.value()),
            # When the image has been resampled (Resize image), incidence is baked
            # into the geometry, so q-math must NOT re-apply the 1/sin factor.
            "use_incidence": (incid > 0.0) and not applied,
            "incidence_deg": 0.0 if applied else incid,
            "incidence_axis": "Y" if incid_axis == "Y" else "X",
            "incidence_applied_in_display": False,
        }

    def _apply_calibration(self) -> None:
        if self._data is None:
            self._set_status("Load a 2D dataset first.", error=True)
            return
        p = self._collect_q_params()
        self._img.set_q_calibration(p)
        if not self._img.apply_q_axes_calibration(p):
            self._set_status("q calibration applied, but failed to update q axes.", error=True)
            return
        # The ROI profile shares these units, so it is redrawn with them.
        self._compute_current_profiles()
        self._set_status("q calibration applied. Axes switched to q.")

    def _apply_incident_cali(self) -> None:
        """Resample the image by 1/sin(theta) so ROIs/profiles analyze the
        geometry-corrected pattern (not just a stretched display)."""
        if self._raw_data is None:
            self._set_status("Load a 2D dataset first.", error=True)
            return
        # Always resample from the as-loaded data so re-applying isn't cumulative.
        corrected, sx, sy = self._resample_for_incidence(self._raw_data)
        self._data = corrected
        self._incident_applied = sx != 1.0 or sy != 1.0

        # Reset any prior display transform; the data itself now carries the stretch.
        self._img.set_q_calibration(None)
        self._img.set_data(self._data)
        self._img.set_pixel_axes_top_right()

        rows, cols = int(self._data.shape[0]), int(self._data.shape[1])
        new_col = int(round(self._center_col.value() * sx))
        new_row = int(round(self._center_row.value() * sy))
        for spin, val in ((self._center_col, new_col), (self._center_row, new_row)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        # New geometry: resize sliders, drop existing ROIs, redraw.
        r_max = int(np.ceil(np.hypot(rows, cols)))
        self._sl_ri.setMaximum(r_max)
        self._sl_ro.setMaximum(r_max)
        self._clear_all_rois()
        self._update_center_overlay()
        self._redraw_roi_overlays()
        self._set_status(
            f"Incidence resized ×{max(sx, sy):.3f} on {self._combo_inc_axis.currentText()} "
            f"({self._spin_inc_deg.value():.4f} deg) → {rows}×{cols}."
        )

    def _disable_q(self) -> None:
        self._img.set_q_calibration(None)
        self._img.set_pixel_axes_labels_only()
        self._update_center_overlay()
        self._compute_current_profiles()
        self._set_status("Switched to pixel axes.")

    def _capture_display_pixmap(self):
        """Grab current rendered image area (same logic as other tools)."""
        try:
            return self._img.graphics_layout.grab()
        except Exception:
            return None

    def _copy_image(self) -> None:
        """Copy current image data rendered with the active colormap."""
        rgb = self._img.render_colormapped_rgb()
        if rgb is None:
            QMessageBox.warning(self, "Copy Image", "No image available to copy.")
            return
        clipboard: QClipboard = QApplication.clipboard()
        try:
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
            clipboard.setMimeData(mime)
        except Exception as exc:
            logging.exception("Q calibration copy image failed")
            QMessageBox.critical(self, "Copy Image", f"Failed to copy image:\n{exc}")
            return
        self._set_status("Colormapped image data copied.")

    def _resample_for_incidence(self, src: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Stretch ``src`` by 1/sin(theta) along the chosen axis.

        Returns (resampled, sx, sy) where sx/sy are the col/row scale factors.
        With theta <= 0 the input is returned unchanged (factors 1.0).
        """
        out = np.asarray(src, dtype=np.float32)
        theta = float(self._spin_inc_deg.value())
        s = float(np.sin(np.deg2rad(theta))) if theta > 0 else 0.0
        if s <= 0:
            return out, 1.0, 1.0
        fac = 1.0 / s
        axis = str(self._combo_inc_axis.currentText() or "X").strip().upper()
        sx, sy = (1.0, fac) if axis == "Y" else (fac, 1.0)
        try:
            from scipy.ndimage import zoom
            return zoom(out, (sy, sx), order=1), sx, sy
        except Exception:
            rep = max(1, int(round(fac)))
            ax = 0 if axis == "Y" else 1
            return (
                np.repeat(out, rep, axis=ax),
                (1.0 if axis == "Y" else float(rep)),
                (float(rep) if axis == "Y" else 1.0),
            )

    @staticmethod
    def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        finite = np.isfinite(a)
        if not np.any(finite):
            return np.zeros_like(a, dtype=np.uint8)
        lo = float(np.nanmin(a[finite]))
        hi = float(np.nanmax(a[finite]))
        if hi <= lo:
            return np.zeros_like(a, dtype=np.uint8)
        u8 = np.clip((a - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)
        return u8

    def _default_save_name(self, suffix: str = "qcal") -> str:
        """Scan and dataset, tagged so it is not mistaken for the raw frame."""
        txt = self._cl_combo.currentText().strip()
        stem = short_series_label(txt, fallback="qcal") if "::" in txt else "qcal"
        return suggested_save_path(stem, suffix, extension=".png")

    def _save_image(self) -> None:
        """Save current image data: PNG/JPEG with colormap, TIFF as grayscale data."""
        if self._data is None:
            QMessageBox.warning(self, "Save Image", "No image available to save.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            self._default_save_name(),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff)",
        )
        if not path:
            return
        remember_save_directory(path)
        out_path = pathlib.Path(path)
        if not out_path.suffix:
            selected_ext = DataExporter.get_extension_from_filter(selected)
            out_path = out_path.with_suffix(selected_ext or ".png")
        ext = out_path.suffix.lower()
        try:
            if ext in (".jpg", ".jpeg"):
                ok = self._img.export_colormapped_image(out_path)
            elif ext == ".png":
                ok = self._img.export_colormapped_image(out_path)
            elif ext in (".tif", ".tiff"):
                from PIL import Image

                Image.fromarray(np.asarray(self._data, dtype=np.float32), mode="F").save(out_path)
                ok = True
            else:
                raise RuntimeError(f"Unsupported image format: {ext or '(none)'}")
            if not ok:
                raise RuntimeError(f"Qt could not save image as {ext}.")
            self._set_status(f"Saved image: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Image", f"Failed to save image:\n{exc}")

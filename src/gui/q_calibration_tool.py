"""Standalone q-calibration tool with FTH-style layout."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QBuffer, QByteArray, QMimeData
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
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.gui.dataset_path_combo import DatasetPathCombo
from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
from src.lib_h5.data_exporter import DataExporter


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
        self._data: np.ndarray | None = None
        self._pending_center_pick = False
        self._incident_applied = False
        self._q_axes_active = False

        self.setWindowTitle("Q Calibration Tool")
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
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(8)

        g_ds = QGroupBox("Input Dataset")
        fl_ds = QFormLayout(g_ds)
        self._combo_dataset = DatasetPathCombo("-- no 2D dataset --")
        self._combo_dataset.lineEdit().returnPressed.connect(self._load_data)
        fl_ds.addRow("Dataset:", self._combo_dataset)
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
        self._center_row.valueChanged.connect(self._update_center_overlay)
        self._center_col.valueChanged.connect(self._update_center_overlay)
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
        left_lay.addWidget(g_geo)

        row_action = QHBoxLayout()
        self._btn_apply = QPushButton("to q")
        self._btn_apply.clicked.connect(self._apply_calibration)
        row_action.addWidget(self._btn_apply)
        self._btn_disable = QPushButton("to pixel")
        self._btn_disable.clicked.connect(self._disable_q)
        row_action.addWidget(self._btn_disable)
        left_lay.addLayout(row_action)

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
        left_lay.addWidget(self._status)
        left_lay.addStretch(1)

        # Right panel
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(4)

        self._img = ImageView2DEnhanced(parent=self)
        self._img.configure_q_tool_mode()
        self._img.chk_show_axes.setChecked(True)
        self._configure_image_toolbar_actions()
        right_lay.addWidget(self._img)

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
        self._status.setText(text)
        self._status.setStyleSheet("color: #b00020;" if error else "color: #444;")

    def _populate_dataset_combo(self) -> None:
        self._combo_dataset.populate_from_full_keys(
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

    def load_dataset_full_key(self, full_key: str, auto_load: bool = True) -> bool:
        """Select a dataset by full key and optionally load immediately."""
        if not full_key or "::" not in full_key:
            return False
        self._combo_dataset.add_full_key(full_key, select=True)
        if auto_load:
            self._load_data()
        return True

    def _on_dataset_entry_entered(self) -> None:
        entry = self._combo_dataset.get_entry(opened_files=self._opened_files)
        if entry is None:
            self._set_status("Invalid dataset path format. Use: file::dataset/path", error=True)
            return
        fp, ds = entry
        if not self._dataset_exists(fp, ds):
            self._set_status(f"Dataset not found: {fp}::{ds}", error=True)
            return
        self._set_status("Dataset path verified. Click 'Load Data' to load.")

    def _dataset_exists(self, fp: str, ds: str) -> bool:
        try:
            with h5py.File(fp, "r") as h5:
                return ds in h5 and isinstance(h5[ds], h5py.Dataset)
        except Exception:
            return False

    def _load_data(self) -> None:
        entry = self._combo_dataset.get_entry(opened_files=self._opened_files)
        if entry is None:
            QMessageBox.warning(self, "Q Calibration", "Please select a valid 2D dataset.")
            return
        fp, ds = entry
        try:
            with h5py.File(fp, "r") as h5:
                if ds not in h5 or not isinstance(h5[ds], h5py.Dataset):
                    raise KeyError(f"Dataset not found: {ds}")
                arr = np.asarray(h5[ds][()])
        except Exception as exc:
            logging.exception("Q calibration load failed")
            QMessageBox.critical(self, "Q Calibration", f"Failed to load dataset:\n{exc}")
            return

        if arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            QMessageBox.warning(
                self,
                "Q Calibration",
                f"Only 2D datasets are supported.\nCurrent shape: {arr.shape}",
            )
            return

        self._set_loaded_array(np.asarray(arr), f"{fp}::{ds}")

    def load_array_data(self, arr: np.ndarray, source_label: str = "calculation_result") -> bool:
        """Load 2D data from memory (no file dataset required)."""
        try:
            data = np.asarray(arr)
            if data.ndim > 2:
                data = data[0]
            if data.ndim != 2:
                QMessageBox.warning(
                    self,
                    "Q Calibration",
                    f"Only 2D arrays are supported.\nCurrent shape: {data.shape}",
                )
                return False
            self._set_loaded_array(np.asarray(data), str(source_label))
            return True
        except Exception as exc:
            logging.exception("Q calibration memory load failed")
            QMessageBox.critical(self, "Q Calibration", f"Failed to load result array:\n{exc}")
            return False

    def _set_loaded_array(self, arr2d: np.ndarray, source_label: str) -> None:
        """Apply loaded 2D array to the viewer and reset calibration state."""
        self._data = np.asarray(arr2d)
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
        self._update_center_overlay()
        self._incident_applied = False
        self._q_axes_active = False
        self._set_status(f"Loaded: {source_label} ({rows}x{cols})")

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

    def _collect_q_params(self) -> dict[str, float | bool | str]:
        incid = float(self._spin_inc_deg.value())
        incid_axis = str(self._combo_inc_axis.currentText() or "X").strip().upper()
        return {
            "energy_ev": float(self._spin_energy.value()),
            "pixel_um": float(self._spin_pixel.value()),
            "distance_mm": float(self._spin_dist.value()),
            "center_x": float(self._center_col.value()),
            "center_y": float(self._center_row.value()),
            "use_incidence": incid > 0.0,
            "incidence_deg": incid,
            "incidence_axis": "Y" if incid_axis == "Y" else "X",
            "incidence_applied_in_display": bool(self._incident_applied),
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
        self._q_axes_active = False
        self._set_status("q calibration applied. Axes switched to q.")

    def _apply_incident_cali(self) -> None:
        """Apply only incident-angle related q correction to current image."""
        if self._data is None:
            self._set_status("Load a 2D dataset first.", error=True)
            return
        self._q_axes_active = False
        self._incident_applied = True
        p = self._collect_q_params()
        self._img.set_q_calibration(p)
        self._img.apply_incidence_display_correction(
            float(p.get("incidence_deg", 0.0)),
            str(p.get("incidence_axis", "X")),
        )
        self._update_center_overlay()
        self._set_status(
            f"Incident angle applied: {self._spin_inc_deg.value():.4f} deg on {self._combo_inc_axis.currentText()}."
        )

    def _disable_q(self) -> None:
        self._img.set_q_calibration(None)
        self._q_axes_active = False
        self._img.set_pixel_axes_labels_only()
        self._update_center_overlay()
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

    def _incident_corrected_data(self) -> np.ndarray | None:
        """Return raw data after incident-angle resize correction."""
        if self._data is None:
            return None
        out = np.asarray(self._data, dtype=np.float32)
        theta = float(self._spin_inc_deg.value())
        if theta <= 0:
            return out
        s = float(np.sin(np.deg2rad(theta)))
        if s <= 0:
            return out
        fac = 1.0 / s
        axis = str(self._combo_inc_axis.currentText() or "X").strip().upper()
        zoom_factors = (1.0, fac) if axis == "X" else (fac, 1.0)
        try:
            from scipy.ndimage import zoom
            return zoom(out, zoom_factors, order=1)
        except Exception:
            # Fallback: nearest-like resize via repeat (keeps feature intent)
            if axis == "X":
                rep = max(1, int(round(fac)))
                return np.repeat(out, rep, axis=1)
            rep = max(1, int(round(fac)))
            return np.repeat(out, rep, axis=0)

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

    def _default_save_name(self, suffix: str = "current") -> str:
        txt = self._combo_dataset.currentText().strip()
        if "::" in txt:
            fname = txt.split("::", 1)[0].strip()
            stem = pathlib.Path(fname).stem
            return f"{stem}_{suffix}.png"
        return f"qcal_{suffix}.png"

    def _save_image(self) -> None:
        """Save current image data: PNG/JPEG with colormap, TIFF as grayscale data."""
        if self._data is None:
            QMessageBox.warning(self, "Save Image", "No image available to save.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            self._default_save_name("current"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff)",
        )
        if not path:
            return
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

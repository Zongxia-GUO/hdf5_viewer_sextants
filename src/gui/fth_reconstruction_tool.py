"""FTH/HERALDO Holographic Reconstruction Tool - MATLAB HERALDO GUI style."""

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
import re
from typing import Optional

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QBuffer, QByteArray, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QCursor, QImage, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from src.gui._shared import apply_hist_colormap as _apply_hist_colormap
from src.gui.dataset_path_combo import DatasetPathCombo
from src.recon.fth import (
    binary_filter as _binary_filter_kernel,
    bs_step as _bs_step,
    differential_filter_kernel as _differential_filter_kernel,
    estimate_balance_ratio as _estimate_balance_ratio_impl,
    fth_transform as _fth_transform,
    line_gaussian_filter as _line_gaussian_filter,
)

# Keep colormap options consistent with the main ImageView2DEnhanced toolbar.
FTH_COLORMAPS = [
    "gray",
    "viridis",
    "inferno",
    "cividis",
    "turbo",
    "CET-L9",
    "CET-L1",
    "CET-L4",
    "CET-R4",
    "CET-D1",
    "CET-D9",
]

# ---------------------------------------------------------------------------
# Dataset selector: dropdown + drag-and-drop combined
# ---------------------------------------------------------------------------

class FTHDatasetCombo(DatasetPathCombo):
    """FTH dataset selector using shared drag-drop/entry behavior."""

    def __init__(self, placeholder: str = "-- none --", parent=None) -> None:
        super().__init__(placeholder=placeholder, parent=parent)

    def populate(self, opened_files) -> None:
        """Populate with 2D+ datasets for FTH/HERALDO processing."""
        super().populate(opened_files, min_ndim=2)


# ---------------------------------------------------------------------------
# Background worker - loads and sums HDF5 datasets
# ---------------------------------------------------------------------------

class _FTHWorker(QThread):
    """Loads CL / CR datasets (possibly multiple scans) in a background thread."""

    finished = pyqtSignal(object, object, object)   # cl_array, cr_array, dark (may be None)
    error    = pyqtSignal(str)

    def __init__(
        self,
        cl_entries: list,   # [(filename, ds_path), ...]
        cr_entries: list,
        dark_entry: Optional[tuple],
    ) -> None:
        super().__init__()
        self._cl   = cl_entries
        self._cr   = cr_entries
        self._dark = dark_entry

    @staticmethod
    def _read_one(filename: str, ds_path: str) -> np.ndarray:
        from src.lib_h5.file_validator import is_hdf5_file
        from src.gui.main_window import load_regular_data_file

        if not is_hdf5_file(filename):
            return np.squeeze(np.asarray(load_regular_data_file(filename))).astype(np.float64)
        with h5py.File(filename, "r") as f:
            return np.squeeze(np.array(f[ds_path])).astype(np.float64)

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return

            single_source = None
            cl = None
            for fn, dp in self._cl:
                if self.isInterruptionRequested():
                    return
                arr = self._read_one(fn, dp)
                cl = arr if cl is None else (cl + arr)

            cr = None
            for fn, dp in self._cr:
                if self.isInterruptionRequested():
                    return
                arr = self._read_one(fn, dp)
                cr = arr if cr is None else (cr + arr)

            if cl is None and cr is None:
                raise RuntimeError("No CL/CR datasets provided.")
            if cl is None:
                cl = cr
                cr = np.zeros_like(cl, dtype=np.asarray(cl).dtype)
                single_source = "CR"
            if cr is None:
                cr = np.zeros_like(cl, dtype=np.asarray(cl).dtype)
                single_source = "CL"

            if self.isInterruptionRequested():
                return
            dark = self._read_one(*self._dark) if self._dark else None
            if self.isInterruptionRequested():
                return
            if single_source is not None:
                logging.info("FTH single-dataset load: %s used as CL, CR set to zeros.", single_source)
            self.finished.emit(cl, cr, dark)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main tool window
# ---------------------------------------------------------------------------

class FTHReconstructionTool(QMainWindow):
    """
    FTH / FTH-HERALDO holographic reconstruction tool.

    4-tab layout mirroring the MATLAB HERALDO GUI (M. Grelier 2023):
      Tab 1 -Alignment    (load CL/CR via drag-drop, center, slit angles)
      Tab 2 -BeamStop     (beamstop smoothing)
      Tab 3 -Filter & FTH (differential filter ->FFT ->FTH display + ROI selection)
      Tab 4 -Reconstruction (4-panel Real/Imag/Phase/Abs of cropped ROI + export)
    """

    def __init__(
        self,
        parent=None,
        opened_files: tuple = None,
        dataset_full_keys_2d: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("FTH Reconstruction Tool")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1400, 880)

        # -- shared data state ------------------------------------------------
        self._opened_files = opened_files or ()
        self._dataset_full_keys_2d = dataset_full_keys_2d or []

        self._CL:  Optional[np.ndarray] = None   # summed raw
        self._CR:  Optional[np.ndarray] = None
        self._single_dataset_mode: bool = False
        self._pending_single_dataset_mode: bool = False
        self._dark: Optional[np.ndarray] = None
        self._CL_c: Optional[np.ndarray] = None  # centerd, cropped
        self._CR_c: Optional[np.ndarray] = None

        self._CL_smooth: Optional[np.ndarray] = None
        self._CR_smooth: Optional[np.ndarray] = None
        self._bs_mask: Optional[np.ndarray] = None
        self._slit_mask: Optional[np.ndarray] = None

        self._Holo_S1:  Optional[np.ndarray] = None
        self._Holo_S2:  Optional[np.ndarray] = None
        self._Holo2_S1: Optional[np.ndarray] = None
        self._Holo2_S2: Optional[np.ndarray] = None

        self._FTH_S1: Optional[np.ndarray] = None   # complex
        self._FTH_S2: Optional[np.ndarray] = None
        self._balance_ratio: float = 1.0

        # hologram geometry (set after centering)
        self._X0:  int = 0    # row center of centered hologram
        self._Y0:  int = 0    # col center
        self._Nx:  int = 0    # rows
        self._Ny:  int = 0    # cols
        self._xmat: Optional[np.ndarray] = None   # row index grid (Nx x Ny)
        self._ymat: Optional[np.ndarray] = None   # col index grid

        # ROI state: roi_centers[slit][roi_idx] = (row, col), slit in {1,2}, roi in {1..4}
        self._roi_centers: dict = {
            1: {1: None, 2: None, 3: None, 4: None},
            2: {1: None, 2: None, 3: None, 4: None},
        }
        self._roi_size: int = 150
        self._roi_count: int = 2

        # display state
        self._current_slit:    int = 0  # 0=None (combined), 1=Slit1, 2=Slit2
        self._current_roi:     int = 1
        self._show_realpart:   str = "Real"
        self._rs_scale:        float = 1.0
        self._rs_scale_base:   float = 1.0   # auto-detected from FTH amplitude
        self._phase_scale:     float = 0.0  # phase rotation (rad) applied to complex FTH data
        self._cmap_name:       str   = "Jet"
        self._last_roi_phase_fit: float = 0.0
        self._t4_autoleveled_key: tuple | None = None  # (slit, roi_idx) last auto-leveled
        self._t4_rs_base:         float = 1.0          # auto-detected FT amplitude base
        self._t4_panel_display: dict = {0: {}, 1: {}, 2: {}, 3: {}}
        self._t4_disp_data: list[Optional[np.ndarray]] = [None, None, None, None]
        # Tab-1 ROI/profile & coordinate display state
        self._t1_value_data: Optional[np.ndarray] = None
        self._t1_current_roi = None
        self._t1_roi_kind: Optional[str] = None
        self._t1_profile_dialog: Optional[QDialog] = None
        self._t1_profile_plot = None

        # percentile level caches
        self._t1_levels_src_id: int = -1
        self._t1_levels_mode: str = ""
        self._t1_cached_levels: tuple = (0.0, 1.0)
        self._t3_holo_src_id: int = -1
        self._t3_holo_mode: str = ""
        self._t3_holo_cached_levels: tuple = (0.0, 1.0)
        # Tab-2 per-panel display overrides (holo/fth). Missing keys fall back to toolbar values.
        self._t3_panel_display: dict = {"holo": {}, "fth": {}}
        self._t3_holo_disp_data: Optional[np.ndarray] = None
        self._t3_fth_disp_data: Optional[np.ndarray] = None
        self._t3_profile_dialog: Optional[QDialog] = None
        self._t3_profile_plot = None
        self._t3_profile_roi_items: dict = {"holo": None, "fth": None}
        self._t3_profile_roi_kind: dict = {"holo": None, "fth": None}

        # picking mode flags
        self._picking_center:  bool = False
        self._picking_bs_center: bool = False
        self._picking_roi1:    bool = False
        self._picking_roi2:    bool = False
        self._picking_roi3:    bool = False
        self._picking_roi4:    bool = False
        self._last_xmid: Optional[int] = None
        self._last_ymid: Optional[int] = None

        # background worker
        self._worker: Optional[_FTHWorker] = None
        self._locked_params: Optional[dict] = None
        self._apply_locked_on_next_load: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Hide the menu bar (File / Help removed per user request)
        self.menuBar().setVisible(False)

        # Central widget holds everything else
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # status bar at bottom
        self._status_label = QLabel("Ready -select or drag CL/CR datasets from the HDF5 tree. Single dataset is supported.")
        self._status_label.setStyleSheet(
            "color:#aaa; font-size:11px; padding:3px 6px; "
            "border-top:1px solid #444; background:#1a1a1a;"
        )
        self._status_label.setFixedHeight(22)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab1(), "Alignment")
        self._tabs.addTab(self._build_tab3(), "Filter && FTH")
        self._tabs.addTab(self._build_tab4(), "Reconstruction")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        root.addWidget(self._tabs, stretch=1)
        root.addWidget(self._status_label)
        self.setCentralWidget(central)
        self._populate_dataset_combos()

    def _build_menu_bar(self) -> None:
        """Populate the QMainWindow menu bar (consistent with the main HDF5-viewer window)."""
        mb = self.menuBar()

        # -- File ------------------------------------------------------
        file_menu: QMenu = mb.addMenu("&File")

        act_export = QAction("&Export Results...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.setStatusTip("Switch to the Reconstruction tab and export results")
        act_export.triggered.connect(lambda: self._tabs.setCurrentIndex(2))
        file_menu.addAction(act_export)

        file_menu.addSeparator()

        act_close = QAction("&Close", self)
        act_close.setShortcut("Ctrl+W")
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        # -- Help ------------------------------------------------------
        help_menu: QMenu = mb.addMenu("&Help")

        act_about = QAction("&About...", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About FTH / HERALDO Reconstruction Tool",
            "<b>FTH / HERALDO Reconstruction Tool</b><br><br>"
            "Fourier-Transform Holography and HERALDO reconstruction.<br>"
            "Based on the MATLAB HERALDO GUI (M. Grelier, 2023).",
        )

    # -- shared UI helpers --------------------------------------------

    def _make_scroll_ctrl(self) -> tuple:
        """Return (scroll_widget, inner_layout) for the left control panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(400)
        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.setSpacing(5)
        lay.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(inner)
        return scroll, lay

    def _make_splitter(self) -> QSplitter:
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setChildrenCollapsible(False)
        return sp

    @staticmethod
    def _add_slider_row(form: QFormLayout, label: str,
                        mn: float, mx: float, val: float,
                        decimals: int = 1) -> tuple:
        """Add a (slider, entry) row to a QFormLayout; returns (slider, entry)."""
        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(int((val - mn) / (mx - mn) * 1000))
        entry  = QLineEdit(f"{val:.{decimals}f}")
        entry.setFixedWidth(54)
        row.addWidget(slider, stretch=1)
        row.addWidget(entry)
        form.addRow(label, row)
        return slider, entry

    @staticmethod
    def _add_plusminus_row(form: QFormLayout, label: str,
                            spin: QSpinBox) -> None:
        """Add spin +/-1 +/-5 buttons on the same row."""
        row = QHBoxLayout()
        for delta, text in [(-5, "-5"), (-1, "-1"), (+1, "+1"), (+5, "+5")]: 
            b = QPushButton(text)
            b.setFixedWidth(30)
            b.clicked.connect(lambda _, d=delta, s=spin: s.setValue(s.value() + d))
            row.addWidget(b)
        row.addWidget(spin)
        form.addRow(label, row)

    @staticmethod
    def _cmap_combo(default: str = "gray") -> QComboBox:
        cb = QComboBox()
        cb.addItems(FTH_COLORMAPS)
        if default in FTH_COLORMAPS:
            cb.setCurrentText(default)
        else:
            cb.setCurrentText("viridis")
        return cb

    def _make_pg_widget(self) -> pg.GraphicsLayoutWidget:
        w = pg.GraphicsLayoutWidget()
        w.setBackground("k")
        w.setMinimumSize(10, 10)   # prevent pyqtgraph from reporting a large minimum that squeezes the left panel
        return w

    @staticmethod
    def _add_plot(glw: pg.GraphicsLayoutWidget,
                  row: int, col: int, title: str = "",
                  rowspan: int = 1, colspan: int = 1) -> tuple:
        """Add a PlotItem + row-major ImageItem; returns (plot, img_item)."""
        p = glw.addPlot(row=row, col=col, title=title,
                        rowspan=rowspan, colspan=colspan)
        # Disable right-click context menus in all FTH image panels.
        p.setMenuEnabled(False)
        if hasattr(p, "vb") and hasattr(p.vb, "setMenuEnabled"):
            p.vb.setMenuEnabled(False)
        p.setAspectLocked(True)
        p.invertY(True)
        for ax in ("left", "bottom", "right", "top"):
            p.hideAxis(ax)
        img = pg.ImageItem(axisOrder="row-major")
        p.addItem(img)
        return p, img
    # ------------------------------------------------------------------
    # TAB 1 -Alignment  (merged: Alignment + BeamStop)
    # ------------------------------------------------------------------

    def _build_tab1(self) -> QWidget:
        tab = QWidget()
        splitter = self._make_splitter()
        QHBoxLayout(tab).addWidget(splitter)
        tab.layout().setContentsMargins(0, 0, 0, 0)

        # -- left controls ----------------------------------------------
        scroll, lay = self._make_scroll_ctrl()

        # CL dataset
        g_cl = QGroupBox("CL Dataset  (circular left polarisation)")
        fl_cl = QFormLayout(g_cl)
        self._cl_combo = FTHDatasetCombo("-- no CL dataset --")
        self._cl_combo.lineEdit().returnPressed.connect(self._on_dataset_entry_entered)
        fl_cl.addRow("Dataset:", self._cl_combo)
        lay.addWidget(g_cl)

        # CR dataset
        g_cr = QGroupBox("CR Dataset  (optional for single-file mode)")
        fl_cr = QFormLayout(g_cr)
        self._cr_combo = FTHDatasetCombo("-- no CR dataset --")
        self._cr_combo.lineEdit().returnPressed.connect(self._on_dataset_entry_entered)
        fl_cr.addRow("Dataset:", self._cr_combo)
        lay.addWidget(g_cr)

        # Dark scan (optional)
        g_dark = QGroupBox("Dark Scan  (optional)")
        fl_dk = QFormLayout(g_dark)
        self._dark_combo = FTHDatasetCombo("-- no dark scan --")
        self._dark_combo.lineEdit().returnPressed.connect(self._on_dataset_entry_entered)
        fl_dk.addRow("Dataset:", self._dark_combo)
        lay.addWidget(g_dark)

        # Load / Lock / Run buttons
        self._load_btn = QPushButton("Load Data")
        self._load_btn.setMinimumHeight(32)
        self._load_btn.clicked.connect(self._load_data)
        lay.addWidget(self._load_btn)
        lock_run_row = QHBoxLayout()
        self._btn_lock_params = QPushButton("Lock Params")
        self._btn_lock_params.setMinimumHeight(28)
        self._btn_lock_params.clicked.connect(self._lock_current_params)
        lock_run_row.addWidget(self._btn_lock_params)
        self._btn_load_apply_locked = QPushButton("Run Locked")
        self._btn_load_apply_locked.setMinimumHeight(30)
        self._btn_load_apply_locked.clicked.connect(self._load_data_apply_locked)
        lock_run_row.addWidget(self._btn_load_apply_locked)
        lay.addLayout(lock_run_row)

        # -- Beamstop Center ----------------------------------------------
        g_bs = QGroupBox("Beamstop Center")
        fl_bs = QFormLayout(g_bs)
        self._btn_pick_bs = QPushButton("Click to Set Beamstop center")
        self._btn_pick_bs.setCheckable(True)
        self._btn_pick_bs.toggled.connect(self._on_pick_bs_toggled)
        self._btn_apply_bs_center = QPushButton("Apply BS Center")
        self._btn_apply_bs_center.setMinimumHeight(28)
        self._btn_apply_bs_center.clicked.connect(self._on_apply_bs_center_clicked)
        bs_btn_row = QHBoxLayout()
        bs_btn_row.addWidget(self._btn_pick_bs)
        bs_btn_row.addWidget(self._btn_apply_bs_center)
        fl_bs.addRow(bs_btn_row)
        self._bs_cx = QSpinBox(); self._bs_cx.setRange(0, 9999); self._bs_cx.setValue(1024)
        self._bs_cy = QSpinBox(); self._bs_cy.setRange(0, 9999); self._bs_cy.setValue(1024)
        self._add_plusminus_row(fl_bs, "BS X (row):", self._bs_cx)
        self._add_plusminus_row(fl_bs, "BS Y (col):", self._bs_cy)
        self._bs_cx.valueChanged.connect(self._update_bs_overlay)
        self._bs_cy.valueChanged.connect(self._update_bs_overlay)
        self._bs_cx.valueChanged.connect(self._on_bs_inputs_changed)
        self._bs_cy.valueChanged.connect(self._on_bs_inputs_changed)
        lay.addWidget(g_bs)

        # -- Beamstop Mask ------------------------------------------------
        g_bs_mask = QGroupBox("Beamstop Mask")
        g_bs_mask.setCheckable(True)
        g_bs_mask.setChecked(False)
        g_bs_mask.toggled.connect(self._on_bs_mask_toggled)
        self._g_bs_mask = g_bs_mask
        fl_bsm = QFormLayout(g_bs_mask)
        self._bs_radius = QSpinBox(); self._bs_radius.setRange(0, 500); self._bs_radius.setValue(90)
        self._bs_radius.valueChanged.connect(self._update_bs_overlay)
        self._bs_radius.valueChanged.connect(self._apply_bs_correction)
        fl_bsm.addRow("Radius (px):", self._bs_radius)
        self._bs_sigma = QDoubleSpinBox()
        self._bs_sigma.setRange(0.1, 500.0); self._bs_sigma.setValue(5.0)
        self._bs_sigma.setSingleStep(1.0); self._bs_sigma.setDecimals(1); self._bs_sigma.setSuffix(" px")
        self._bs_sigma.valueChanged.connect(self._apply_bs_correction)
        fl_bsm.addRow("sigma transition (px):", self._bs_sigma)
        lay.addWidget(g_bs_mask)

        # -- Hologram Center -------------------------------------------
        g_center = QGroupBox("Hologram Center")
        fl_c = QFormLayout(g_center)
        self._btn_pick_center = QPushButton("Click to Set Center")
        self._btn_pick_center.setCheckable(True)
        self._btn_pick_center.toggled.connect(self._on_pick_center_toggled)
        self._btn_apply_center = QPushButton("Apply Center")
        self._btn_apply_center.setMinimumHeight(28)
        self._btn_apply_center.clicked.connect(self._on_apply_center_clicked)
        center_btn_row = QHBoxLayout()
        center_btn_row.addWidget(self._btn_pick_center)
        center_btn_row.addWidget(self._btn_apply_center)
        fl_c.addRow(center_btn_row)
        self._t1_xmid = QSpinBox(); self._t1_xmid.setRange(0, 9999); self._t1_xmid.setValue(1024)
        self._t1_ymid = QSpinBox(); self._t1_ymid.setRange(0, 9999); self._t1_ymid.setValue(1024)
        self._add_plusminus_row(fl_c, "Xmid (row):", self._t1_xmid)
        self._add_plusminus_row(fl_c, "Ymid (col):", self._t1_ymid)
        self._t1_xmid.valueChanged.connect(self._on_center_inputs_changed)
        self._t1_ymid.valueChanged.connect(self._on_center_inputs_changed)
        lay.addWidget(g_center)

        # -- Slit Angles ---------------------------------------------------
        g_phi = QGroupBox("Slit Angles")
        fl_phi = QFormLayout(g_phi)
        self._phi1_spin = QDoubleSpinBox()
        self._phi1_spin.setRange(-180.0, 180.0); self._phi1_spin.setValue(0.0)
        self._phi1_spin.setSingleStep(0.1); self._phi1_spin.setDecimals(1)
        self._phi1_spin.setSuffix(" deg")
        self._phi2_spin = QDoubleSpinBox()
        self._phi2_spin.setRange(-180.0, 180.0); self._phi2_spin.setValue(90.0)
        self._phi2_spin.setSingleStep(0.1); self._phi2_spin.setDecimals(1)
        self._phi2_spin.setSuffix(" deg")
        fl_phi.addRow("Phi 1 (horiz. slit):", self._phi1_spin)
        fl_phi.addRow("Phi 2 (vert.  slit):", self._phi2_spin)
        self._phi1_spin.valueChanged.connect(self._update_slit_lines)
        self._phi2_spin.valueChanged.connect(self._update_slit_lines)
        self._phi1_spin.valueChanged.connect(self._apply_slit_mask)
        self._phi2_spin.valueChanged.connect(self._apply_slit_mask)
        lay.addWidget(g_phi)

        # -- Slit Mask --------------------------------------------------------
        g_slit_mask = QGroupBox("Slit Mask")
        g_slit_mask.setCheckable(True)
        g_slit_mask.setChecked(False)
        g_slit_mask.toggled.connect(self._on_slit_mask_toggled)
        self._g_slit_mask = g_slit_mask
        fl_sm2 = QFormLayout(g_slit_mask)
        self._slit_mask_phi1_chk = QCheckBox("Apply Phi 1 mask")
        self._slit_mask_phi1_chk.setChecked(True)
        self._slit_mask_phi2_chk = QCheckBox("Apply Phi 2 mask")
        self._slit_mask_phi2_chk.setChecked(True)
        self._slit_mask_phi1_chk.toggled.connect(self._apply_slit_mask)
        self._slit_mask_phi2_chk.toggled.connect(self._apply_slit_mask)
        slit_mask_row = QHBoxLayout()
        slit_mask_row.addWidget(self._slit_mask_phi1_chk)
        slit_mask_row.addWidget(self._slit_mask_phi2_chk)
        fl_sm2.addRow("Apply:", slit_mask_row)
        self._slit_mask_width = QDoubleSpinBox()
        self._slit_mask_width.setRange(0.0, 2000.0); self._slit_mask_width.setValue(2.0)
        self._slit_mask_width.setSingleStep(5.0); self._slit_mask_width.setDecimals(1); self._slit_mask_width.setSuffix(" px")
        fl_sm2.addRow("Width (px):", self._slit_mask_width)
        self._slit_mask_sigma = QDoubleSpinBox()
        self._slit_mask_sigma.setRange(1.0, 2000.0); self._slit_mask_sigma.setValue(50.0)
        self._slit_mask_sigma.setSingleStep(10.0); self._slit_mask_sigma.setDecimals(1); self._slit_mask_sigma.setSuffix(" px")
        fl_sm2.addRow("sigma band (px):", self._slit_mask_sigma)
        self._slit_mask_width.valueChanged.connect(self._apply_slit_mask)
        self._slit_mask_sigma.valueChanged.connect(self._apply_slit_mask)
        lay.addWidget(g_slit_mask)

        splitter.addWidget(scroll)

        # -- right display --------------------------------------------------
        self._t1_glw = self._make_pg_widget()

        self._t1_main_plot, self._t1_main_img = self._add_plot(
            self._t1_glw, 0, 0, "Hologram  (log scale, centerd)"
        )
        self._t1_hist = pg.HistogramLUTItem(gradientPosition="right")
        self._t1_hist.setImageItem(self._t1_main_img)
        self._t1_glw.addItem(self._t1_hist, row=0, col=1)

        # Overlays: center marker, slit lines, beamstop circle
        self._t1_center_marker = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen("c", width=2), brush=pg.mkBrush(None), symbol="+"
        )
        self._t1_slit1_line = pg.PlotCurveItem(pen=pg.mkPen("r", width=1.5))
        self._t1_slit2_line = pg.PlotCurveItem(pen=pg.mkPen((0, 180, 255), width=1.5))
        self._t1_bs_circle  = pg.PlotCurveItem(pen=pg.mkPen("y", width=2))
        self._t1_bs_dot     = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen("y", width=2), brush=pg.mkBrush(None), symbol="+"
        )
        for _ov in (self._t1_center_marker, self._t1_slit1_line,
                    self._t1_slit2_line, self._t1_bs_circle, self._t1_bs_dot):
            self._t1_main_plot.addItem(_ov, ignoreBounds=True)

        self._t1_main_plot.scene().sigMouseClicked.connect(self._on_t1_clicked)
        self._t1_main_plot.scene().sigMouseMoved.connect(self._on_t1_mouse_moved)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)
        toolbar_layout.addWidget(QLabel("Colormap:"))
        self._t1_tb_cmap = QComboBox()
        self._t1_tb_cmap.addItems(FTH_COLORMAPS)
        self._t1_tb_cmap.setCurrentText("gray")
        self._t1_tb_cmap.currentTextChanged.connect(self._on_t1_toolbar_changed)
        toolbar_layout.addWidget(self._t1_tb_cmap)

        self._t1_tb_invert = QCheckBox("Invert")
        self._t1_tb_invert.toggled.connect(self._on_t1_toolbar_changed)
        toolbar_layout.addWidget(self._t1_tb_invert)

        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(QLabel("Scale:"))
        self._t1_tb_scale = QComboBox()
        self._t1_tb_scale.addItems(["Linear", "Log", "SymLog", "Square root"])
        self._t1_tb_scale.setCurrentText("Log")
        self._t1_tb_scale.currentTextChanged.connect(self._on_t1_toolbar_changed)
        toolbar_layout.addWidget(self._t1_tb_scale)

        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(QLabel("Scale X:"))
        self._t1_tb_scale_x = QLineEdit("")
        self._t1_tb_scale_x.setMaximumWidth(45)
        self._t1_tb_scale_x.returnPressed.connect(self._on_t1_toolbar_changed)
        toolbar_layout.addWidget(self._t1_tb_scale_x)
        toolbar_layout.addWidget(QLabel("Y:"))
        self._t1_tb_scale_y = QLineEdit("")
        self._t1_tb_scale_y.setMaximumWidth(45)
        self._t1_tb_scale_y.returnPressed.connect(self._on_t1_toolbar_changed)
        toolbar_layout.addWidget(self._t1_tb_scale_y)

        self._t1_tb_axes = QCheckBox("Show Axes")
        self._t1_tb_axes.toggled.connect(self._on_t1_toolbar_changed)
        toolbar_layout.addWidget(self._t1_tb_axes)

        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(QLabel("ROI:"))
        self._t1_btn_roi_line = QPushButton("━")
        self._t1_btn_roi_line.setFixedSize(24, 24)
        self._t1_btn_roi_line.setCheckable(True)
        self._t1_btn_roi_line.setToolTip("Line ROI")
        self._t1_btn_roi_line.clicked.connect(lambda: self._on_t1_roi_button_clicked("Line"))
        toolbar_layout.addWidget(self._t1_btn_roi_line)

        self._t1_btn_roi_rect = QPushButton("■")
        self._t1_btn_roi_rect.setFixedSize(24, 24)
        self._t1_btn_roi_rect.setCheckable(True)
        self._t1_btn_roi_rect.setToolTip("Rectangle ROI")
        self._t1_btn_roi_rect.clicked.connect(lambda: self._on_t1_roi_button_clicked("Rectangle"))
        toolbar_layout.addWidget(self._t1_btn_roi_rect)

        toolbar_layout.addSpacing(8)
        self._t1_tb_coords = QLabel("X: - | Y: - | Value: -")
        self._t1_tb_coords.setStyleSheet("color: gray; font-size: 9pt;")
        self._t1_tb_coords.setMaximumWidth(260)
        toolbar_layout.addWidget(self._t1_tb_coords)
        toolbar_layout.addStretch()

        right_layout.addLayout(toolbar_layout)
        right_layout.addWidget(self._t1_glw, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([390, 1010])    # Tab 1 needs a wider left panel than Tab 2/3
        splitter.setStretchFactor(0, 0)   # left panel: fixed width on window resize
        splitter.setStretchFactor(1, 1)   # right panel: absorbs all resize
        return tab


    # ------------------------------------------------------------------
    # TAB 3 -Filter & FTH
    # ------------------------------------------------------------------

    def _build_tab3(self) -> QWidget:
        tab = QWidget()
        splitter = self._make_splitter()
        QHBoxLayout(tab).addWidget(splitter)
        tab.layout().setContentsMargins(0, 0, 0, 0)

        scroll, lay = self._make_scroll_ctrl()

        # Slit & filter
        g_filt = QGroupBox("Differential Filter")
        fl_f = QFormLayout(g_filt)
        self._slit_combo = QComboBox(); self._slit_combo.addItems(["None", "Slit 1", "Slit 2"])
        self._slit_combo.setCurrentText("Slit 1")
        self._slit_combo.currentIndexChanged.connect(self._on_slit_changed)
        fl_f.addRow("Current slit:", self._slit_combo)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["Gaussian", "Binary", "None"])
        self._filter_combo.setCurrentText("None")
        self._filter_combo.currentTextChanged.connect(self._on_filter_type_changed)
        fl_f.addRow("Secondary filter:", self._filter_combo)

        # Gaussian params
        self._g_gaussian = QGroupBox("Gaussian filter params")
        fl_gp = QFormLayout(self._g_gaussian)
        self._filt_sigma = QDoubleSpinBox(); self._filt_sigma.setRange(0.1, 500); self._filt_sigma.setValue(5.0)
        self._filt_sigma.setSuffix(" px")
        fl_gp.addRow("sigma:", self._filt_sigma)
        self._filt_shift = QDoubleSpinBox(); self._filt_shift.setRange(-500, 500); self._filt_shift.setValue(0.0)
        self._filt_shift.setSuffix(" px")
        fl_gp.addRow("Shift:", self._filt_shift)
        fl_f.addRow(self._g_gaussian)

        # Binary params
        self._g_binary = QGroupBox("Binary filter params")
        fl_bp = QFormLayout(self._g_binary)
        self._filt_width = QSpinBox(); self._filt_width.setRange(1, 200); self._filt_width.setValue(5)
        self._filt_width.setSuffix(" px")
        fl_bp.addRow("Width:", self._filt_width)
        fl_f.addRow(self._g_binary)
        self._g_binary.setVisible(False)
        self._on_filter_type_changed(self._filter_combo.currentText())

        g_filt.setCheckable(True)
        g_filt.setChecked(False)
        g_filt.toggled.connect(self._on_diff_filter_toggled)
        self._g_diff_filter = g_filt
        lay.addWidget(g_filt)

        # Auto balance row — placed directly below the GroupBox, wrapped so it
        # can be hidden as a unit by tools that embed this tab.
        self._chk_balance = QCheckBox("Auto balance CL/CR")
        self._chk_balance.setChecked(True)
        self._balance_ratio_edit = QLineEdit("1.0000")
        self._balance_ratio_edit.setReadOnly(True)
        self._balance_ratio_edit.setFixedWidth(70)
        self._balance_widget = QWidget()
        _bal_row = QHBoxLayout(self._balance_widget)
        _bal_row.setContentsMargins(0, 0, 0, 0)
        _bal_row.addWidget(self._chk_balance)
        _bal_row.addWidget(QLabel("Ratio:"))
        _bal_row.addWidget(self._balance_ratio_edit)
        lay.addWidget(self._balance_widget)

        # Compute FTH button — applies filter then computes in one step
        self._btn_compute_fth = QPushButton("Compute FTH")
        self._btn_compute_fth.setMinimumHeight(30)
        self._btn_compute_fth.clicked.connect(self._apply_and_compute_fth)
        lay.addWidget(self._btn_compute_fth)

        # FTH display controls
        g_fth_disp = QGroupBox("FTH Display")
        fl_fd = QFormLayout(g_fth_disp)
        self._realpart_combo = QComboBox()
        self._realpart_combo.addItems(["Real", "Imag.", "Phase", "Abs."])
        self._realpart_combo.currentTextChanged.connect(self._update_t3_fth_display)
        fl_fd.addRow("Show:", self._realpart_combo)

        self._rs_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._rs_scale_slider.setRange(1, 400); self._rs_scale_slider.setValue(100)
        self._rs_scale_slider.valueChanged.connect(self._update_t3_fth_display)
        self._rs_scale_entry = QLineEdit("1.0"); self._rs_scale_entry.setFixedWidth(54)
        self._rs_scale_entry.returnPressed.connect(self._on_rs_scale_text)
        rs_row = QHBoxLayout()
        rs_row.addWidget(self._rs_scale_slider, stretch=1)
        rs_row.addWidget(self._rs_scale_entry)
        fl_fd.addRow("FT amplitude:", rs_row)

        self._phase_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._phase_scale_slider.setRange(-314, 314); self._phase_scale_slider.setValue(0)
        self._phase_scale_slider.valueChanged.connect(self._update_t3_fth_display)
        self._phase_scale_entry = QLineEdit("0.00"); self._phase_scale_entry.setFixedWidth(54)
        self._phase_scale_entry.returnPressed.connect(self._on_phase_scale_text)
        ph_row = QHBoxLayout()
        ph_row.addWidget(self._phase_scale_slider, stretch=1)
        ph_row.addWidget(self._phase_scale_entry)
        fl_fd.addRow("Phase rotation (rad):", ph_row)

        lay.addWidget(g_fth_disp)

        # ROI controls
        g_roi = QGroupBox("ROI Selection  (click on FTH panel)")
        fl_roi = QFormLayout(g_roi)

        self._btn_roi1 = QPushButton("ROI 1 -click to set center")
        self._btn_roi1.setCheckable(True)
        self._btn_roi1.toggled.connect(self._on_roi1_toggled)
        fl_roi.addRow(self._btn_roi1)
        self._roi1_x = QLineEdit("-"); self._roi1_x.setReadOnly(True); self._roi1_x.setFixedWidth(50)
        self._roi1_y = QLineEdit("-"); self._roi1_y.setReadOnly(True); self._roi1_y.setFixedWidth(50)
        roi1_xy = QHBoxLayout()
        roi1_xy.addWidget(QLabel("row:")); roi1_xy.addWidget(self._roi1_x)
        roi1_xy.addWidget(QLabel("col:")); roi1_xy.addWidget(self._roi1_y)
        fl_roi.addRow("ROI 1 pos:", roi1_xy)

        self._btn_roi2 = QPushButton("ROI 2 -click to set center")
        self._btn_roi2.setCheckable(True)
        self._btn_roi2.toggled.connect(self._on_roi2_toggled)
        fl_roi.addRow(self._btn_roi2)
        self._roi2_x = QLineEdit("-"); self._roi2_x.setReadOnly(True); self._roi2_x.setFixedWidth(50)
        self._roi2_y = QLineEdit("-"); self._roi2_y.setReadOnly(True); self._roi2_y.setFixedWidth(50)
        roi2_xy = QHBoxLayout()
        roi2_xy.addWidget(QLabel("row:")); roi2_xy.addWidget(self._roi2_x)
        roi2_xy.addWidget(QLabel("col:")); roi2_xy.addWidget(self._roi2_y)
        fl_roi.addRow("ROI 2 pos:", roi2_xy)

        self._btn_roi3 = QPushButton("ROI 3 -click to set center")
        self._btn_roi3.setCheckable(True)
        self._btn_roi3.toggled.connect(self._on_roi3_toggled)
        fl_roi.addRow(self._btn_roi3)
        self._roi3_x = QLineEdit("-"); self._roi3_x.setReadOnly(True); self._roi3_x.setFixedWidth(50)
        self._roi3_y = QLineEdit("-"); self._roi3_y.setReadOnly(True); self._roi3_y.setFixedWidth(50)
        roi3_xy = QHBoxLayout()
        roi3_xy.addWidget(QLabel("row:")); roi3_xy.addWidget(self._roi3_x)
        roi3_xy.addWidget(QLabel("col:")); roi3_xy.addWidget(self._roi3_y)
        fl_roi.addRow("ROI 3 pos:", roi3_xy)

        self._btn_roi4 = QPushButton("ROI 4 -click to set center")
        self._btn_roi4.setCheckable(True)
        self._btn_roi4.toggled.connect(self._on_roi4_toggled)
        fl_roi.addRow(self._btn_roi4)
        self._roi4_x = QLineEdit("-"); self._roi4_x.setReadOnly(True); self._roi4_x.setFixedWidth(50)
        self._roi4_y = QLineEdit("-"); self._roi4_y.setReadOnly(True); self._roi4_y.setFixedWidth(50)
        roi4_xy = QHBoxLayout()
        roi4_xy.addWidget(QLabel("row:")); roi4_xy.addWidget(self._roi4_x)
        roi4_xy.addWidget(QLabel("col:")); roi4_xy.addWidget(self._roi4_y)
        fl_roi.addRow("ROI 4 pos:", roi4_xy)

        self._roi_count_spin = QSpinBox()
        self._roi_count_spin.setRange(1, 4)
        self._roi_count_spin.setValue(self._roi_count)
        self._roi_count_spin.valueChanged.connect(self._on_roi_count_changed)
        fl_roi.addRow("ROI count:", self._roi_count_spin)

        self._roi_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._roi_size_slider.setRange(10, 400); self._roi_size_slider.setValue(150)
        self._roi_size_slider.valueChanged.connect(self._on_roi_size_changed)
        self._roi_size_entry = QLineEdit("150"); self._roi_size_entry.setFixedWidth(40)
        self._roi_size_entry.returnPressed.connect(self._on_roi_size_text)
        rs2_row = QHBoxLayout()
        rs2_row.addWidget(self._roi_size_slider, stretch=1)
        rs2_row.addWidget(self._roi_size_entry)
        fl_roi.addRow("ROI size (px):", rs2_row)

        self._chk_show_roi = QCheckBox("Show ROI rectangles")
        self._chk_show_roi.setChecked(True)
        self._chk_show_roi.toggled.connect(self._update_roi_rects)
        fl_roi.addRow(self._chk_show_roi)
        lay.addWidget(g_roi)

        splitter.addWidget(scroll)

        # -- right display (two panels) -------------------------------
        self._t3_glw = self._make_pg_widget()

        self._t3_holo_plot, self._t3_holo_img = self._add_plot(
            self._t3_glw, 0, 0, "|Filtered hologram|  (log)"
        )
        self._t3_holo_hist = pg.HistogramLUTItem(gradientPosition="right")
        self._t3_holo_hist.setImageItem(self._t3_holo_img)
        self._t3_glw.addItem(self._t3_holo_hist, row=0, col=1)

        self._t3_fth_plot, self._t3_fth_img = self._add_plot(
            self._t3_glw, 0, 2, "FTH Reconstruction"
        )
        self._t3_fth_hist = pg.HistogramLUTItem(gradientPosition="right")
        self._t3_fth_hist.setImageItem(self._t3_fth_img)
        self._t3_glw.addItem(self._t3_fth_hist, row=0, col=3)

        # ROI rectangle overlays on FTH panel
        self._roi1_rect = pg.RectROI([0, 0], [50, 50],
                                      pen=pg.mkPen("r", width=1.5), movable=False, rotatable=False, resizable=False)
        self._roi2_rect = pg.RectROI([0, 0], [50, 50],
                                      pen=pg.mkPen((255, 128, 0), width=1.5), movable=False, rotatable=False, resizable=False)
        self._roi3_rect = pg.RectROI([0, 0], [50, 50],
                                      pen=pg.mkPen((0, 220, 120), width=1.5), movable=False, rotatable=False, resizable=False)
        self._roi4_rect = pg.RectROI([0, 0], [50, 50],
                                      pen=pg.mkPen((80, 180, 255), width=1.5), movable=False, rotatable=False, resizable=False)
        self._t3_fth_plot.addItem(self._roi1_rect)
        self._t3_fth_plot.addItem(self._roi2_rect)
        self._t3_fth_plot.addItem(self._roi3_rect)
        self._t3_fth_plot.addItem(self._roi4_rect)
        self._roi1_rect.setVisible(False)
        self._roi2_rect.setVisible(False)
        self._roi3_rect.setVisible(False)
        self._roi4_rect.setVisible(False)
        self._on_roi_count_changed(self._roi_count)

        self._t3_fth_plot.scene().sigMouseClicked.connect(self._on_t3_fth_clicked)
        self._t3_glw.scene().sigMouseClicked.connect(self._on_t3_scene_clicked)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self._t3_glw, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([290, 1110])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return tab

    # ------------------------------------------------------------------
    # TAB 4 -Reconstruction
    # ------------------------------------------------------------------

    def _build_tab4(self) -> QWidget:
        tab = QWidget()
        splitter = self._make_splitter()
        QHBoxLayout(tab).addWidget(splitter)
        tab.layout().setContentsMargins(0, 0, 0, 0)

        scroll, lay = self._make_scroll_ctrl()

        # ROI selector
        g_sel = QGroupBox("ROI Selection")
        fl_sel = QFormLayout(g_sel)
        self._t4_slitroi = QComboBox()
        self._t4_slitroi.addItems(["ROI 1", "ROI 2"])
        self._t4_slitroi.currentIndexChanged.connect(self._update_t4_display)
        fl_sel.addRow("Selection:", self._t4_slitroi)
        self._refresh_t4_roi_selector()
        self._on_roi_count_changed(self._roi_count)

        self._t4_roi_size = QSpinBox(); self._t4_roi_size.setRange(10, 800); self._t4_roi_size.setValue(150)
        self._t4_roi_size.valueChanged.connect(self._update_t4_display)
        fl_sel.addRow("ROI size (px):", self._t4_roi_size)
        # Sync with t3 slider -use blockSignals to prevent infinite recursion
        def _on_t4_roi_slider(v):
            self._roi_size = v
            self._t4_roi_size.blockSignals(True)
            self._t4_roi_size.setValue(v)
            self._t4_roi_size.blockSignals(False)
            self._update_t4_display()

        def _on_t4_roi_spinbox(v):
            self._roi_size = v
            self._roi_size_slider.blockSignals(True)
            self._roi_size_slider.setValue(v)
            self._roi_size_slider.blockSignals(False)

        self._roi_size_slider.valueChanged.connect(_on_t4_roi_slider)
        self._t4_roi_size.valueChanged.connect(_on_t4_roi_spinbox)
        lay.addWidget(g_sel)

        # Amplitude
        g_amp = QGroupBox("Display Amplitude")
        fl_amp = QFormLayout(g_amp)
        self._t4_rs_slider = QSlider(Qt.Orientation.Horizontal)
        self._t4_rs_slider.setRange(1, 400); self._t4_rs_slider.setValue(100)
        self._t4_rs_slider.valueChanged.connect(self._update_t4_display)
        self._t4_rs_entry = QLineEdit("1.0"); self._t4_rs_entry.setFixedWidth(54)
        self._t4_rs_entry.returnPressed.connect(self._on_t4_rs_text)
        rs4_row = QHBoxLayout()
        rs4_row.addWidget(self._t4_rs_slider, stretch=1)
        rs4_row.addWidget(self._t4_rs_entry)
        fl_amp.addRow("FT amplitude:", rs4_row)

        self._t4_ph_slider = QSlider(Qt.Orientation.Horizontal)
        self._t4_ph_slider.setRange(-314, 314); self._t4_ph_slider.setValue(0)
        self._t4_ph_slider.valueChanged.connect(self._update_t4_display)
        self._t4_ph_entry = QLineEdit("0.00"); self._t4_ph_entry.setFixedWidth(54)
        self._t4_ph_entry.returnPressed.connect(self._on_t4_ph_text)
        ph4_row = QHBoxLayout()
        ph4_row.addWidget(self._t4_ph_slider, stretch=1)
        ph4_row.addWidget(self._t4_ph_entry)
        fl_amp.addRow("Phase rotation (rad):", ph4_row)
        lay.addWidget(g_amp)

        # Corrections
        g_corr = QGroupBox("Corrections")
        fl_corr = QFormLayout(g_corr)
        self._chk_inv_contrast  = QCheckBox("Invert contrast")
        self._chk_inv_realimag  = QCheckBox("Invert Real/Imag  (x exp(i*pi/2))")
        self._chk_gauss_filter  = QCheckBox("Gaussian filter")
        self._t4_gauss_sigma    = QDoubleSpinBox()
        self._t4_gauss_sigma.setRange(0.1, 20.0); self._t4_gauss_sigma.setValue(1.0)
        self._t4_gauss_sigma.setSuffix(" px")
        self._chk_phase_fit = QCheckBox("Auto phase fit (ROI)")
        self._t4_phase_fit_win = QSpinBox()
        self._t4_phase_fit_win.setRange(8, 512); self._t4_phase_fit_win.setValue(70)
        self._t4_phase_fit_win.setSuffix(" px")
        self._t4_phase_fit_label = QLineEdit("0.0000 rad")
        self._t4_phase_fit_label.setReadOnly(True)
        for chk in (self._chk_inv_contrast, self._chk_inv_realimag, self._chk_gauss_filter):
            chk.toggled.connect(self._update_t4_display)
            fl_corr.addRow(chk)
        self._t4_gauss_sigma.valueChanged.connect(self._update_t4_display)
        fl_corr.addRow("Gauss sigma:", self._t4_gauss_sigma)
        self._chk_phase_fit.toggled.connect(self._on_phase_fit_toggled)
        fl_corr.addRow(self._chk_phase_fit)
        self._t4_phase_fit_win.valueChanged.connect(self._update_t4_display)
        fl_corr.addRow("Fit window:", self._t4_phase_fit_win)
        fl_corr.addRow("Estimated phase:", self._t4_phase_fit_label)
        lay.addWidget(g_corr)

        # Export
        g_exp = QGroupBox("Save && Export")
        fl_exp = QFormLayout(g_exp)
        self._exp_target_combo = QComboBox()
        self._exp_target_combo.addItems(["Real", "Imag.", "Phase", "Abs."])
        fl_exp.addRow("Target:", self._exp_target_combo)

        btn_row = QHBoxLayout()
        self._btn_copy_component = QPushButton("Copy")
        self._btn_copy_component.clicked.connect(self._copy_selected_component_jpeg)
        self._btn_save_component = QPushButton("Save")
        self._btn_save_component.clicked.connect(self._save_selected_component)
        btn_row.addWidget(self._btn_copy_component)
        btn_row.addWidget(self._btn_save_component)
        fl_exp.addRow(btn_row)

        self._btn_save_all_components = QPushButton("Save All")
        self._btn_save_all_components.clicked.connect(self._save_all_components)
        fl_exp.addRow(self._btn_save_all_components)
        lay.addWidget(g_exp)

        splitter.addWidget(scroll)

        # -- right display (2x2) --------------------------------------
        self._t4_glw = self._make_pg_widget()

        self._t4_plots = []
        self._t4_imgs  = []
        self._t4_hists = []
        titles = ["Real", "Imag.", "Phase", "Abs."]
        for idx, title in enumerate(titles):
            r, c = idx // 2, idx % 2
            c_img = c * 2          # images at cols 0, 2
            p, img = self._add_plot(self._t4_glw, r, c_img, title)
            self._t4_plots.append(p)
            self._t4_imgs.append(img)
            hist = pg.HistogramLUTItem(gradientPosition="right")
            hist.setImageItem(img)
            self._t4_glw.addItem(hist, row=r, col=c_img + 1)
            self._t4_hists.append(hist)
        self._t4_glw.scene().sigMouseClicked.connect(self._on_t4_scene_clicked)

        splitter.addWidget(self._t4_glw)
        splitter.setSizes([290, 1110])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return tab

    # ==================================================================
    # Dataset combo helpers
    # ==================================================================

    def _populate_dataset_combos(self) -> None:
        """Populate CL/CR/Dark combos with all 2D+ datasets from opened HDF5 files."""
        full_keys_2d = self._dataset_full_keys_2d
        if not full_keys_2d:
            full_keys_2d = DatasetPathCombo.collect_full_keys(
                self._opened_files,
                min_ndim=2,
                min_second_dim=101,
            )
        for combo in (self._cl_combo, self._cr_combo, self._dark_combo):
            combo.populate_from_full_keys(full_keys_2d, opened_files=self._opened_files)
        if not self._opened_files:
            self._set_status("No HDF5 files open -open a file, then select or drag datasets here.")

    def refresh_dataset_keys(
        self,
        full_keys_2d: list[str],
        opened_files: tuple[pathlib.Path, ...] | None = None,
    ) -> None:
        """Refresh CL/CR/Dark dataset candidates from shared index."""
        if opened_files is not None:
            self._opened_files = tuple(opened_files)
        self._dataset_full_keys_2d = list(full_keys_2d)
        self._populate_dataset_combos()

    def add_dataset_to_combo(self, full_path: str, channel: str) -> None:
        """Add a dataset to the CL, CR, or Dark combo programmatically.

        Args:
            full_path: ``/abs/path/file.nxs::hdf5/dataset/path`` (same format as drag-and-drop)
            channel: "CL", "CR", or "Dark"
        """
        channel_map = {"CL": self._cl_combo, "CR": self._cr_combo, "Dark": self._dark_combo}
        combo = channel_map.get(channel)
        if combo is None:
            logging.warning(f"FTHReconstructionTool: unknown channel '{channel}'")
            return
        combo.add_full_key(full_path, select=True)
        self.raise_()
        self.activateWindow()

    # ==================================================================
    # TAB 1 -Data loading & display
    # ==================================================================

    def _load_data(self) -> None:
        cl_entry   = self._cl_combo.get_entry(opened_files=self._opened_files)
        cr_entry   = self._cr_combo.get_entry(opened_files=self._opened_files)
        dark_entry = self._dark_combo.get_entry(opened_files=self._opened_files)

        if not cl_entry and not cr_entry:
            self._set_status("Please select at least one CL or CR dataset.", error=True)
            return

        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(200)

        self._load_btn.setEnabled(False)
        self._pending_single_dataset_mode = not (cl_entry and cr_entry)
        self._set_status(
            "Loading single dataset in background..."
            if self._pending_single_dataset_mode
            else "Loading datasets in background..."
        )

        self._worker = _FTHWorker(
            [cl_entry] if cl_entry else [],
            [cr_entry] if cr_entry else [],
            dark_entry,
        )
        self._worker.finished.connect(self._on_load_finished)
        self._worker.error.connect(self._on_load_error)
        self._worker.start()

    def _on_dataset_entry_entered(self) -> None:
        """Pressing Enter validates dataset entries only; loading is manual via button."""
        cl_ok = self._cl_combo.get_entry(opened_files=self._opened_files) is not None
        cr_ok = self._cr_combo.get_entry(opened_files=self._opened_files) is not None
        dark_txt = self._dark_combo.currentText().strip()
        dark_ok = (not dark_txt) or dark_txt.startswith("--") or (
            self._dark_combo.get_entry(opened_files=self._opened_files) is not None
        )

        if (cl_ok or cr_ok) and dark_ok:
            self._set_status("Dataset entries validated. Click 'Load Data' to continue.")
        else:
            self._set_status(
                "Invalid dataset entry. Use format: filename::dataset_path",
                error=True,
            )

    def _load_data_apply_locked(self) -> None:
        if not self._locked_params:
            self._set_status("No locked params yet. Click 'Lock' first.", error=True)
            return
        if (
            not self._cl_combo.get_entry(opened_files=self._opened_files)
            and not self._cr_combo.get_entry(opened_files=self._opened_files)
        ):
            self._set_status("Please select at least one CL or CR dataset.", error=True)
            return
        self._apply_locked_on_next_load = True
        self._load_data()

    def _on_load_finished(self, cl: np.ndarray, cr: np.ndarray,
                           dark: Optional[np.ndarray]) -> None:
        sender = self.sender()
        if sender is not self._worker:
            return
        self._load_btn.setEnabled(True)
        if dark is not None:
            cl = np.clip(cl - dark, 0, None)
            cr = np.clip(cr - dark, 0, None)
        self._CL   = cl
        self._CR   = cr
        self._single_dataset_mode = bool(self._pending_single_dataset_mode)
        self._dark = dark
        self._initialize_center_controls_for_loaded_shape(cl.shape)
        # Reset downstream results
        self._CL_smooth = self._CR_smooth = None
        self._bs_mask = None
        self._slit_mask = None
        self._Holo_S1 = self._Holo_S2 = None
        self._Holo2_S1 = self._Holo2_S2 = None
        self._FTH_S1 = self._FTH_S2 = None
        # Force one-shot autolevel for newly loaded data.
        self._t1_levels_src_id = -1
        self._t1_levels_mode = ""
        self._compute_centered_hologram()
        self._last_xmid = self._t1_xmid.value()
        self._last_ymid = self._t1_ymid.value()
        # Initialise beamstop center to the center of the centerd image
        if self._X0 > 0:
            self._bs_cx.setValue(self._X0)
            self._bs_cy.setValue(self._Y0)
        # Re-apply active masks to the freshly loaded/centered data.
        self._apply_bs_correction()
        self._apply_slit_mask()
        self._update_t1_main_display()
        self._on_t1_toolbar_changed()
        self._update_slit_lines()
        self._update_center_marker()
        if self._apply_locked_on_next_load and self._locked_params:
            self._apply_locked_on_next_load = False
            self._apply_locked_params_to_current_data()
            self._apply_filters_only()
            self._compute_fth_only()
            self._update_t4_display()
            return
        mode_label = "single dataset" if self._single_dataset_mode else "CL/CR pair"
        self._set_status(
            f"Loaded OK ({mode_label}) - shape {cl.shape}, "
            f"centered to ({self._Nx}x{self._Ny}), X0={self._X0} Y0={self._Y0}"
        )

    def _initialize_center_controls_for_loaded_shape(self, shape: tuple[int, int]) -> None:
        """Use the loaded image geometry instead of stale detector defaults."""
        rows, cols = int(shape[0]), int(shape[1])
        center_row = max(0, rows // 2)
        center_col = max(0, cols // 2)
        for spin, value, limit in (
            (self._t1_xmid, center_row, rows),
            (self._t1_ymid, center_col, cols),
        ):
            spin.blockSignals(True)
            try:
                spin.setRange(0, max(0, limit - 1))
                spin.setValue(min(max(0, value), max(0, limit - 1)))
            finally:
                spin.blockSignals(False)

    def _on_load_error(self, msg: str) -> None:
        sender = self.sender()
        if sender is not self._worker:
            return
        self._load_btn.setEnabled(True)
        self._apply_locked_on_next_load = False
        self._set_status(f"Load error: {msg}", error=True)

    def _lock_current_params(self) -> None:
        if self._CL is None:
            self._set_status("Load one dataset first, then lock params.", error=True)
            return
        # Toggle lock: pressing Lock again clears the template.
        if self._locked_params is not None:
            self._locked_params = None
            self._btn_lock_params.setStyleSheet("")
            self._set_status("Locked params cleared.")
            return
        rows, cols = self._CL.shape
        rows = max(1, rows)
        cols = max(1, cols)
        xmid = self._t1_xmid.value()
        ymid = self._t1_ymid.value()
        x0 = self._X0 if self._X0 > 0 else 0
        y0 = self._Y0 if self._Y0 > 0 else 0

        roi_offsets = {1: {}, 2: {}}
        for slit in (1, 2):
            for roi_idx in (1, 2, 3, 4):
                ctr = self._roi_centers[slit].get(roi_idx)
                if ctr is None:
                    roi_offsets[slit][roi_idx] = None
                else:
                    roi_offsets[slit][roi_idx] = (int(ctr[0] - x0), int(ctr[1] - y0))

        self._locked_params = {
            "center_frac": (float(xmid / max(1, rows - 1)), float(ymid / max(1, cols - 1))),
            "phi1": float(self._phi1_spin.value()),
            "phi2": float(self._phi2_spin.value()),
            "slit_mask_enabled": bool(self._g_slit_mask.isChecked()),
            "slit_mask_phi1": bool(self._slit_mask_phi1_chk.isChecked()),
            "slit_mask_phi2": bool(self._slit_mask_phi2_chk.isChecked()),
            "slit_width": float(self._slit_mask_width.value()),
            "slit_sigma": float(self._slit_mask_sigma.value()),
            "bs_mask_enabled": bool(self._g_bs_mask.isChecked()),
            "bs_radius": int(self._bs_radius.value()),
            "bs_sigma": float(self._bs_sigma.value()),
            "bs_offset": (int(self._bs_cx.value() - x0), int(self._bs_cy.value() - y0)),
            "slit_mode": self._slit_combo.currentText(),
            "filter_type": self._filter_combo.currentText(),
            "auto_balance": bool(self._chk_balance.isChecked()),
            "filt_sigma": float(self._filt_sigma.value()),
            "filt_shift": float(self._filt_shift.value()),
            "filt_width": int(self._filt_width.value()),
            "roi_count": int(self._roi_count),
            "roi_size": int(self._roi_size),
            "roi_offsets": roi_offsets,
        }
        self._btn_lock_params.setStyleSheet("QPushButton { background-color: #c9302c; color: white; font-weight: 600; }")
        self._set_status("Current params locked.")

    def _apply_locked_params_to_current_data(self) -> None:
        if not self._locked_params or self._CL is None:
            return
        lp = self._locked_params
        rows, cols = self._CL.shape
        rows = max(1, rows)
        cols = max(1, cols)

        # 1) Restore center from normalized coordinates, then recenter once.
        frac_r, frac_c = lp.get("center_frac", (0.5, 0.5))
        nxmid = int(round(float(frac_r) * max(1, rows - 1)))
        nymid = int(round(float(frac_c) * max(1, cols - 1)))
        nxmid = max(0, min(rows - 1, nxmid))
        nymid = max(0, min(cols - 1, nymid))
        self._t1_xmid.setValue(nxmid)
        self._t1_ymid.setValue(nymid)
        self._on_center_changed()

        # 2) Restore mask/filter scalar params.
        self._phi1_spin.setValue(float(lp.get("phi1", self._phi1_spin.value())))
        self._phi2_spin.setValue(float(lp.get("phi2", self._phi2_spin.value())))
        self._slit_mask_width.setValue(float(lp.get("slit_width", self._slit_mask_width.value())))
        self._slit_mask_sigma.setValue(float(lp.get("slit_sigma", self._slit_mask_sigma.value())))
        self._slit_mask_phi1_chk.setChecked(bool(lp.get("slit_mask_phi1", True)))
        self._slit_mask_phi2_chk.setChecked(bool(lp.get("slit_mask_phi2", True)))
        self._g_slit_mask.setChecked(bool(lp.get("slit_mask_enabled", False)))

        self._bs_radius.setValue(int(lp.get("bs_radius", self._bs_radius.value())))
        self._bs_sigma.setValue(float(lp.get("bs_sigma", self._bs_sigma.value())))
        self._g_bs_mask.setChecked(bool(lp.get("bs_mask_enabled", False)))

        # 3) Restore BS center using offset to centered hologram center.
        x0 = self._X0 if self._X0 > 0 else 0
        y0 = self._Y0 if self._Y0 > 0 else 0
        bs_dr, bs_dc = lp.get("bs_offset", (0, 0))
        br = max(0, min(max(0, self._Nx - 1), int(round(x0 + bs_dr))))
        bc = max(0, min(max(0, self._Ny - 1), int(round(y0 + bs_dc))))
        self._bs_cx.setValue(br)
        self._bs_cy.setValue(bc)
        self._apply_bs_correction()
        self._apply_slit_mask()

        # 4) Restore filter and ROI config.
        self._slit_combo.setCurrentText(str(lp.get("slit_mode", "Slit 1")))
        self._filter_combo.setCurrentText(str(lp.get("filter_type", "None")))
        self._chk_balance.setChecked(bool(lp.get("auto_balance", True)))
        self._filt_sigma.setValue(float(lp.get("filt_sigma", self._filt_sigma.value())))
        self._filt_shift.setValue(float(lp.get("filt_shift", self._filt_shift.value())))
        self._filt_width.setValue(int(lp.get("filt_width", self._filt_width.value())))
        self._roi_count_spin.setValue(int(lp.get("roi_count", self._roi_count)))
        self._roi_size_slider.setValue(int(lp.get("roi_size", self._roi_size)))

        # 5) Restore ROI centers from centered offsets.
        roi_offsets = lp.get("roi_offsets", {})
        for slit in (1, 2):
            offs_s = roi_offsets.get(slit, {})
            for roi_idx in (1, 2, 3, 4):
                off = offs_s.get(roi_idx)
                if off is None:
                    self._roi_centers[slit][roi_idx] = None
                    continue
                rr = max(0, min(max(0, self._Nx - 1), int(round(x0 + off[0]))))
                cc = max(0, min(max(0, self._Ny - 1), int(round(y0 + off[1]))))
                self._roi_centers[slit][roi_idx] = (rr, cc)

        # Refresh ROI text boxes for active slit.
        active_slit = self._active_roi_slit()
        for idx, xed, yed in (
            (1, self._roi1_x, self._roi1_y),
            (2, self._roi2_x, self._roi2_y),
            (3, self._roi3_x, self._roi3_y),
            (4, self._roi4_x, self._roi4_y),
        ):
            ctr = self._roi_centers[active_slit].get(idx)
            if ctr is None:
                xed.setText("-")
                yed.setText("-")
            else:
                xed.setText(str(ctr[0]))
                yed.setText(str(ctr[1]))

        self._update_roi_rects()
        self._update_t3_holo_display()
        self._update_t3_fth_display()
        self._update_t4_display()
        self._set_status("Locked params applied to loaded data.")

    def _compute_centered_hologram(self) -> None:
        """Crop CL/CR to the largest centered square, compute X0/Y0/Xmat/Ymat."""
        if self._CL is None:
            return
        Xmid = self._t1_xmid.value()
        Ymid = self._t1_ymid.value()
        rows, cols = self._CL.shape
        xsize  = min(Xmid, rows - 1 - Xmid)
        ysize  = min(Ymid, cols - 1 - Ymid)
        minsize = min(xsize, ysize)
        if minsize <= 0:
            self._set_status("Center out of bounds -adjust Xmid/Ymid.", error=True)
            return
        cl = self._CL[Xmid - minsize : Xmid + minsize, Ymid - minsize : Ymid + minsize]
        cr = self._CR[Xmid - minsize : Xmid + minsize, Ymid - minsize : Ymid + minsize]
        # Make even
        if cl.shape[0] % 2 == 1:
            cl, cr = cl[:-1, :-1], cr[:-1, :-1]
        self._CL_c = cl
        self._CR_c = cr
        self._Nx, self._Ny = cl.shape
        self._X0, self._Y0 = self._Nx // 2, self._Ny // 2
        rows_idx  = np.arange(self._Nx, dtype=float)
        cols_idx  = np.arange(self._Ny, dtype=float)
        self._xmat, self._ymat = np.meshgrid(rows_idx, cols_idx, indexing="ij")

    def _update_t1_main_display(self) -> None:
        """Show the best available centerd hologram (smooth > centerd > raw)."""
        src = (self._CL_smooth if self._CL_smooth is not None
               else (self._CL_c if self._CL_c is not None
                     else self._CL))
        if src is None:
            return
        mode = self._t1_tb_scale.currentText() if hasattr(self, "_t1_tb_scale") else "Log"
        # Use raw magnitude as base; display mode decides Linear/Log/etc.
        base = np.abs(src)
        self._t1_value_data = base
        data = self._transform_for_display(base, mode)
        self._t1_main_img.setImage(data)

        # Set levels through the HistogramLUTItem so that the on-screen handles and
        # the image's actual display levels stay in sync. To avoid contrast jumping
        # during picking/mask edits, autolevel only on first display and on Scale-mode change.
        src_id = id(src)
        if self._t1_levels_src_id < 0 or mode != self._t1_levels_mode:
            vmin = float(np.percentile(data, 1))
            vmax = float(np.percentile(data, 99))
            vmax = max(vmax, vmin + 1e-6)
            self._t1_cached_levels = (vmin, vmax)
            self._t1_hist.setLevels(*self._t1_cached_levels)
        self._t1_levels_src_id = src_id
        self._t1_levels_mode = mode

        cmap_name, invert = self._read_cmap_controls(
            self._t1_tb_cmap if hasattr(self, "_t1_tb_cmap") else None,
            self._t1_tb_invert if hasattr(self, "_t1_tb_invert") else None,
            default="gray",
        )
        _apply_hist_colormap(self._t1_hist, cmap_name, invert=invert)

        tr = self._build_axis_transform(
            self._t1_tb_scale_x if hasattr(self, "_t1_tb_scale_x") else None,
            self._t1_tb_scale_y if hasattr(self, "_t1_tb_scale_y") else None,
        )
        self._set_items_transform([self._t1_main_img], tr)
        # Keep overlays (center/slit/beamstop) in the same coordinate transform
        # as the image so Auto-range reset never causes apparent drift.
        self._set_items_transform((
            self._t1_center_marker,
            self._t1_slit1_line,
            self._t1_slit2_line,
            self._t1_bs_circle,
            self._t1_bs_dot,
        ), tr)

        show_axes = self._t1_tb_axes.isChecked() if hasattr(self, "_t1_tb_axes") else False
        self._set_plots_axes_visible([self._t1_main_plot], show_axes)
        self._update_bs_overlay()
        self._update_t1_profile()
        callback = getattr(self, "_t1_display_updated_callback", None)
        if callback is not None:
            try:
                callback()
            except Exception:
                logging.debug("Alignment display update callback failed", exc_info=True)

    def _on_t1_toolbar_changed(self, *_) -> None:
        self._update_t1_main_display()

    def _on_t1_roi_button_clicked(self, roi_type: str) -> None:
        if roi_type == "Line":
            clicked = self._t1_btn_roi_line
            other = self._t1_btn_roi_rect
        else:
            clicked = self._t1_btn_roi_rect
            other = self._t1_btn_roi_line
        if clicked.isChecked():
            other.setChecked(False)
            self._on_t1_roi_type_changed(roi_type)
        else:
            self._on_t1_roi_type_changed("None")

    def _on_t1_roi_type_changed(self, roi_type: str) -> None:
        if self._t1_current_roi is not None:
            try:
                self._t1_main_plot.removeItem(self._t1_current_roi)
            except Exception:
                pass
            self._t1_current_roi = None
            self._t1_roi_kind = None
        if roi_type == "None":
            self._t1_btn_roi_line.setChecked(False)
            self._t1_btn_roi_rect.setChecked(False)
            self._update_t1_profile()
            return
        if self._t1_value_data is None:
            self._set_status("No image data loaded for Tab1 ROI.", error=True)
            self._t1_btn_roi_line.setChecked(False)
            self._t1_btn_roi_rect.setChecked(False)
            return

        h, w = self._t1_value_data.shape[:2]
        cx = w * 0.5
        cy = h * 0.5
        if roi_type == "Line":
            roi = pg.LineSegmentROI(
                [[cx - w * 0.15, cy], [cx + w * 0.15, cy]],
                pen=pg.mkPen((255, 90, 90), width=2),
            )
        else:
            side = max(20.0, min(float(h), float(w)) * 0.2)
            roi = pg.RectROI(
                [cx - side * 0.5, cy - side * 0.5],
                [side, side],
                pen=pg.mkPen((255, 200, 60), width=2),
                movable=True,
                rotatable=False,
                resizable=True,
            )
        self._t1_main_plot.addItem(roi)
        self._t1_current_roi = roi
        self._t1_roi_kind = "line" if roi_type == "Line" else "square"
        roi.sigRegionChanged.connect(self._update_t1_profile)
        self._ensure_t1_profile_window()
        self._t1_profile_dialog.show()
        self._t1_profile_dialog.raise_()
        self._t1_profile_dialog.activateWindow()
        self._update_t1_profile()

    def _ensure_t1_profile_window(self) -> None:
        if self._t1_profile_dialog is not None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Tab1 ROI Profile")
        dlg.resize(620, 340)
        lay = QVBoxLayout(dlg)
        self._t1_profile_plot = pg.PlotWidget()
        self._t1_profile_plot.setBackground("k")
        self._t1_profile_plot.showGrid(x=True, y=True, alpha=0.25)
        lay.addWidget(self._t1_profile_plot)
        self._t1_profile_dialog = dlg

    def _update_t1_profile(self) -> None:
        if self._t1_profile_plot is None:
            return
        self._t1_profile_plot.clear()
        if self._t1_current_roi is None or self._t1_value_data is None or self._t1_roi_kind is None:
            self._t1_profile_plot.setTitle("No ROI")
            return
        try:
            reg = self._t1_current_roi.getArrayRegion(self._t1_value_data, self._t1_main_img)
            if reg is None:
                return
            arr = np.asarray(reg, dtype=np.float64)
            if arr.size == 0:
                return
            if self._t1_roi_kind == "line":
                y = None
                pts = self._line_roi_endpoints_image_coords(self._t1_current_roi, self._t1_main_img)
                if pts is not None:
                    x0, y0, x1, y1 = pts
                    y = self._sample_line_profile(self._t1_value_data, x0, y0, x1, y1)
                if y is None or np.asarray(y).size == 0:
                    # Fallback path
                    y = arr.ravel() if arr.ndim == 1 else np.nanmean(arr, axis=int(np.argmin(arr.shape)))
                    if np.ndim(y) > 1:
                        y = np.asarray(y).ravel()
            else:
                y = np.nanmean(arr, axis=0) if arr.ndim > 1 else arr
            x = np.arange(y.size, dtype=np.float64)
            self._t1_profile_plot.plot(x, y, pen=pg.mkPen((100, 220, 255), width=2))
            self._t1_profile_plot.setTitle(f"Tab1 ROI profile ({self._t1_roi_kind})")
            self._t1_profile_plot.setLabel("left", "Intensity")
            self._t1_profile_plot.setLabel("bottom", "Pixel")
        except Exception as exc:
            logging.debug("Tab1 ROI profile update failed: %s", exc)

    @staticmethod
    def _line_roi_endpoints_image_coords(roi, img_item) -> Optional[tuple[float, float, float, float]]:
        """Return (x0, y0, x1, y1) for a LineSegmentROI in image-item coordinates."""
        if roi is None or img_item is None:
            return None
        try:
            hs = roi.getSceneHandlePositions()
            if hs is None or len(hs) < 2:
                return None
            p0_scene = hs[0][1]
            p1_scene = hs[1][1]
            p0_img = img_item.mapFromScene(p0_scene)
            p1_img = img_item.mapFromScene(p1_scene)
            return (
                float(p0_img.x()),
                float(p0_img.y()),
                float(p1_img.x()),
                float(p1_img.y()),
            )
        except Exception:
            return None

    @staticmethod
    def _sample_line_profile(data: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Sample intensity along a line segment using bilinear interpolation."""
        if data is None or data.ndim < 2:
            return np.array([], dtype=np.float64)
        h, w = data.shape[:2]
        n = max(2, int(round(float(np.hypot(x1 - x0, y1 - y0)))))
        xs = np.linspace(x0, x1, n, dtype=np.float64)
        ys = np.linspace(y0, y1, n, dtype=np.float64)
        xs = np.clip(xs, 0.0, max(0.0, w - 1.0))
        ys = np.clip(ys, 0.0, max(0.0, h - 1.0))

        x0i = np.floor(xs).astype(np.int64)
        y0i = np.floor(ys).astype(np.int64)
        x1i = np.clip(x0i + 1, 0, w - 1)
        y1i = np.clip(y0i + 1, 0, h - 1)
        wx = xs - x0i
        wy = ys - y0i

        v00 = data[y0i, x0i].astype(np.float64)
        v10 = data[y0i, x1i].astype(np.float64)
        v01 = data[y1i, x0i].astype(np.float64)
        v11 = data[y1i, x1i].astype(np.float64)

        return (
            (1.0 - wx) * (1.0 - wy) * v00
            + wx * (1.0 - wy) * v10
            + (1.0 - wx) * wy * v01
            + wx * wy * v11
        )

    def _on_t1_mouse_moved(self, pos) -> None:
        if self._t1_value_data is None or not hasattr(self, "_t1_tb_coords"):
            return
        vb = self._t1_main_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self._t1_tb_coords.setText("X: - | Y: - | Value: -")
            self._t1_tb_coords.setStyleSheet("color: gray; font-size: 9pt;")
            return
        try:
            pt = self._t1_main_img.mapFromScene(pos)
            x = int(round(float(pt.x())))
            y = int(round(float(pt.y())))
            if 0 <= y < self._t1_value_data.shape[0] and 0 <= x < self._t1_value_data.shape[1]:
                v = float(self._t1_value_data[y, x])
                self._t1_tb_coords.setText(f"X: {x} | Y: {y} | Value: {v:.3g}")
                self._t1_tb_coords.setStyleSheet("color: black; font-size: 9pt;")
            else:
                self._t1_tb_coords.setText("X: - | Y: - | Value: -")
                self._t1_tb_coords.setStyleSheet("color: gray; font-size: 9pt;")
        except Exception:
            self._t1_tb_coords.setText("X: - | Y: - | Value: -")
            self._t1_tb_coords.setStyleSheet("color: gray; font-size: 9pt;")

    def _update_slit_lines(self) -> None:
        """Draw slit direction lines on the centerd hologram."""
        no_data = self._CL_c is None
        if no_data:
            self._t1_slit1_line.setData([], [])
            self._t1_slit2_line.setData([], [])
            return
        xmid, ymid = self._preview_center_in_view()
        xmid = float(xmid)   # center in current centerd-image view coordinates (row)
        ymid = float(ymid)   # center in current centerd-image view coordinates (col)
        L = max(self._Nx, self._Ny) * 0.7
        for spin, line in [(self._phi1_spin, self._t1_slit1_line),
                            (self._phi2_spin, self._t1_slit2_line)]:
            phi_rad = np.deg2rad(spin.value())
            # phi=0 ->horizontal slit ->horizontal guide line (dx=L, dy=0)
            dx = np.cos(phi_rad) * L
            dy = np.sin(phi_rad) * L
            line.setData(x=[ymid - dx, ymid + dx], y=[xmid - dy, xmid + dy])

    def _preview_center_in_view(self) -> tuple[int, int]:
        """Return center marker position in current view coordinates.

        Before Apply Center, show a preview using pending Xmid/Ymid values.
        After apply (or without pending change), this equals (X0, Y0).
        """
        if self._CL_c is None or self._Nx <= 0 or self._Ny <= 0:
            return (self._X0, self._Y0)
        if self._last_xmid is None or self._last_ymid is None:
            return (self._X0, self._Y0)
        dx = self._t1_xmid.value() - self._last_xmid
        dy = self._t1_ymid.value() - self._last_ymid
        prow = int(round(self._X0 + dx))
        pcol = int(round(self._Y0 + dy))
        prow = max(0, min(self._Nx - 1, prow))
        pcol = max(0, min(self._Ny - 1, pcol))
        return (prow, pcol)

    def _update_center_marker(self) -> None:
        """Mark the hologram center on the centerd image at (X0, Y0)."""
        if self._X0 == 0:
            self._t1_center_marker.setData([], [])
            return
        prow, pcol = self._preview_center_in_view()
        self._t1_center_marker.setData(x=[float(pcol)], y=[float(prow)])

    def _on_center_inputs_changed(self, *_args) -> None:
        if self._CL is None:
            return
        xmid = self._t1_xmid.value()
        ymid = self._t1_ymid.value()
        if self._last_xmid is None or self._last_ymid is None:
            return
        if xmid != self._last_xmid or ymid != self._last_ymid:
            self._set_status(
                f"Hologram center pending: row={xmid}, col={ymid}. Click 'Apply Center' to recenter."
            )
        self._update_slit_lines()
        self._update_center_marker()

    def _on_bs_inputs_changed(self, *_args) -> None:
        if self._CL is None:
            return
        self._set_status(
            f"Beamstop center pending: row={self._bs_cx.value()}, col={self._bs_cy.value()}. Click 'Apply BS Center' to apply."
        )

    def _on_apply_bs_center_clicked(self) -> None:
        if self._btn_pick_bs.isChecked():
            self._btn_pick_bs.setChecked(False)
        self._apply_bs_correction()

    def _on_apply_center_clicked(self) -> None:
        if self._btn_pick_center.isChecked():
            self._btn_pick_center.setChecked(False)
        self._on_center_changed()

    def _on_center_changed(self) -> None:
        # Recompute the centerd hologram so +/- center nudges update display immediately.
        if self._CL is not None:
            if self._last_xmid is not None and self._last_ymid is not None:
                if self._t1_xmid.value() == self._last_xmid and self._t1_ymid.value() == self._last_ymid:
                    self._set_status("Hologram center unchanged.")
                    return
            old_xmid = self._last_xmid if self._last_xmid is not None else self._t1_xmid.value()
            old_ymid = self._last_ymid if self._last_ymid is not None else self._t1_ymid.value()
            old_x0, old_y0 = self._X0, self._Y0
            # Preserve BS physical location in original-image coordinates.
            bs_orig_row = self._bs_cx.value() + old_xmid - old_x0
            bs_orig_col = self._bs_cy.value() + old_ymid - old_y0

            self._CL_smooth = self._CR_smooth = None   # invalidate masks
            self._bs_mask = None
            self._slit_mask = None
            self._compute_centered_hologram()

            # Map preserved original BS point into the new centerd-image coordinates.
            new_xmid = self._t1_xmid.value()
            new_ymid = self._t1_ymid.value()
            bs_new_row = int(round(bs_orig_row - new_xmid + self._X0))
            bs_new_col = int(round(bs_orig_col - new_ymid + self._Y0))
            bs_new_row = max(0, min(self._Nx - 1, bs_new_row))
            bs_new_col = max(0, min(self._Ny - 1, bs_new_col))
            self._bs_cx.blockSignals(True)
            self._bs_cy.blockSignals(True)
            try:
                self._bs_cx.setValue(bs_new_row)
                self._bs_cy.setValue(bs_new_col)
            finally:
                self._bs_cx.blockSignals(False)
                self._bs_cy.blockSignals(False)

            self._apply_bs_correction()
            self._apply_slit_mask()
            self._update_t1_main_display()
            self._last_xmid = new_xmid
            self._last_ymid = new_ymid
            self._set_status(f"Hologram recentered: row={new_xmid}, col={new_ymid}")
        self._update_slit_lines()
        self._update_center_marker()

    def _on_pick_center_toggled(self, checked: bool) -> None:
        if checked:
            # Hard mutual exclusion: force BS pick mode off first.
            if self._btn_pick_bs.isChecked():
                self._btn_pick_bs.blockSignals(True)
                self._btn_pick_bs.setChecked(False)
                self._btn_pick_bs.blockSignals(False)
            self._picking_bs_center = False
            self._picking_center = True
        else:
            self._picking_center = False
        self._btn_pick_center.setText(
            ">>> Click on image to set center <<<" if checked else "Click to Set Center"
        )

    def _on_t1_clicked(self, event) -> None:
        """Handle clicks on the merged Alignment display for center or BS picking."""
        # Snapshot flags before any side-effects change them
        picking_bs  = self._picking_bs_center and self._btn_pick_bs.isChecked()
        picking_ctr = self._picking_center and self._btn_pick_center.isChecked()
        if picking_bs and picking_ctr:
            # Safety fallback: keep center picking, force BS mode off.
            self._btn_pick_bs.blockSignals(True)
            self._btn_pick_bs.setChecked(False)
            self._btn_pick_bs.blockSignals(False)
            self._picking_bs_center = False
            picking_bs = False
        if not (picking_bs or picking_ctr):
            return
        try:
            vb  = self._t1_main_plot.getViewBox()
            pos = event.scenePos()
            if not vb.sceneBoundingRect().contains(pos):
                return
            pt  = vb.mapSceneToView(pos)
            col = int(round(float(pt.x())))
            row = int(round(float(pt.y())))
            # Bounds check (centerd-image coords)
            nx = self._Nx if self._Nx > 0 else -1
            ny = self._Ny if self._Ny > 0 else -1
            if not (0 <= row < nx and 0 <= col < ny):
                return
            if picking_bs:
                self._bs_cy.setValue(col)
                self._bs_cx.setValue(row)
                self._set_status(
                    f"Beamstop center pending: row={row}, col={col}. Click 'Apply BS Center' to apply."
                )
            elif picking_ctr:
                # centerd-image coords -> original image coords
                old_xmid = self._last_xmid if self._last_xmid is not None else self._t1_xmid.value()
                old_ymid = self._last_ymid if self._last_ymid is not None else self._t1_ymid.value()
                orig_row = row + old_xmid - self._X0
                orig_col = col + old_ymid - self._Y0
                self._t1_xmid.setValue(max(0, min(9999, orig_row)))
                self._t1_ymid.setValue(max(0, min(9999, orig_col)))
                self._set_status(
                    f"Hologram center pending: row={orig_row}, col={orig_col}. Click 'Apply Center' to recenter."
                )
        except Exception as exc:
            logging.debug("t1 click: %s", exc)

    # ==================================================================
    # Masks (Beamstop + Slit)
    # ==================================================================

    def _on_bs_mask_toggled(self, checked: bool) -> None:
        self._apply_bs_correction()
        if checked:
            self._set_status("Beamstop mask enabled.")
        else:
            self._set_status("Beamstop mask disabled.")

    def _on_slit_mask_toggled(self, checked: bool) -> None:
        self._apply_slit_mask()
        if checked:
            self._set_status("Slit mask enabled.")
        else:
            self._set_status("Slit mask disabled.")

    def _apply_bs_correction(self) -> None:
        """Apply smooth-step beamstop mask and recompute displayed masked data."""
        if self._CL_c is None:
            return
        try:
            if not self._g_bs_mask.isChecked():
                self._bs_mask = None
                self._recompute_smooth_data()
                self._Holo_S1 = self._Holo_S2 = None
                self._FTH_S1  = self._FTH_S2  = None
                self._update_t1_main_display()
                return
            cx     = self._bs_cx.value()
            cy     = self._bs_cy.value()
            sig    = self._bs_sigma.value()
            radius = self._bs_radius.value()
            dist   = np.sqrt((self._xmat - cx) ** 2 + (self._ymat - cy) ** 2)
            self._bs_mask = _bs_step(sig, dist - radius)
            self._recompute_smooth_data()
            self._Holo_S1 = self._Holo_S2 = None
            self._FTH_S1  = self._FTH_S2  = None
            self._update_t1_main_display()
            self._set_status(f"Beamstop mask applied  (\u03c3={sig:.1f} px, radius={radius} px).")
        except Exception as exc:
            self._set_status(f"BS mask error: {exc}", error=True)
            logging.exception("BS mask")

    def _apply_slit_mask(self) -> None:
        """Apply slit-notch mask and recompute from centered base data (no stacking)."""
        if self._CL_c is None:
            return
        try:
            if hasattr(self, "_g_slit_mask") and not self._g_slit_mask.isChecked():
                self._slit_mask = None
                self._recompute_smooth_data()
                self._Holo_S1 = self._Holo_S2 = None
                self._FTH_S1  = self._FTH_S2  = None
                self._update_t1_main_display()
                return
            use_phi1 = bool(getattr(self, "_slit_mask_phi1_chk", None) is None
                            or self._slit_mask_phi1_chk.isChecked())
            use_phi2 = bool(getattr(self, "_slit_mask_phi2_chk", None) is None
                            or self._slit_mask_phi2_chk.isChecked())
            if not (use_phi1 or use_phi2):
                self._slit_mask = None
                self._recompute_smooth_data()
                self._Holo_S1 = self._Holo_S2 = None
                self._FTH_S1  = self._FTH_S2  = None
                self._update_t1_main_display()
                self._set_status("Slit mask cleared - no slit direction selected.")
                return
            phi1 = np.deg2rad(self._phi1_spin.value())
            phi2 = np.deg2rad(self._phi2_spin.value())
            sigma = self._slit_mask_sigma.value()
            width = self._slit_mask_width.value()
            drow = self._xmat - self._X0
            dcol = self._ymat - self._Y0
            # Signed perpendicular distance to guide line in (col, row) view coordinates:
            # n = (-sin(phi), cos(phi));  dist = n . (dcol, drow)
            perp1 = drow * np.cos(phi1) - dcol * np.sin(phi1)
            perp2 = drow * np.cos(phi2) - dcol * np.sin(phi2)

            # Width sets a full notch band around each slit line, sigma controls soft edge.
            half_w = max(width * 0.5, 0.0)
            edge1 = np.maximum(np.abs(perp1) - half_w, 0.0)
            edge2 = np.maximum(np.abs(perp2) - half_w, 0.0)
            band1 = np.exp(-(edge1 ** 2) / (2 * sigma ** 2))
            band2 = np.exp(-(edge2 ** 2) / (2 * sigma ** 2))

            bands = []
            if use_phi1:
                bands.append(band1)
            if use_phi2:
                bands.append(band2)
            combined_band = np.maximum.reduce(bands)
            self._slit_mask = np.clip(1.0 - combined_band, 0.0, 1.0)
            self._recompute_smooth_data()
            self._Holo_S1 = self._Holo_S2 = None
            self._FTH_S1  = self._FTH_S2  = None
            self._update_t1_main_display()
            active = ", ".join(
                name for name, enabled in (("Phi 1", use_phi1), ("Phi 2", use_phi2)) if enabled
            )
            self._set_status(
                f"Slit mask applied ({active}; width={width:.1f} px, \u03c3={sigma:.1f} px)."
            )
        except Exception as exc:
            self._set_status(f"Slit mask error: {exc}", error=True)
            logging.exception("Slit mask")

    def _recompute_smooth_data(self) -> None:
        """Recompute masked CL/CR from centered base data and active masks."""
        if self._CL_c is None:
            self._CL_smooth = None
            self._CR_smooth = None
            return
        mask = np.ones_like(self._CL_c, dtype=np.float64)
        if self._bs_mask is not None:
            mask *= self._bs_mask
        if self._slit_mask is not None:
            mask *= self._slit_mask
        self._CL_smooth = self._CL_c * mask
        self._CR_smooth = self._CR_c * mask

    def _update_bs_overlay(self) -> None:
        cx = self._bs_cx.value()
        cy = self._bs_cy.value()
        r  = self._bs_radius.value()
        if r > 0:
            theta = np.linspace(0, 2 * np.pi, 256)
            self._t1_bs_circle.setData(x=cy + r * np.cos(theta), y=cx + r * np.sin(theta))
            self._t1_bs_dot.setData(x=[float(cy)], y=[float(cx)])
        else:
            self._t1_bs_circle.setData([], [])
            self._t1_bs_dot.setData([], [])

    def _on_pick_bs_toggled(self, checked: bool) -> None:
        if checked:
            # Hard mutual exclusion: force center-pick mode off first.
            if self._btn_pick_center.isChecked():
                self._btn_pick_center.blockSignals(True)
                self._btn_pick_center.setChecked(False)
                self._btn_pick_center.blockSignals(False)
            self._picking_center = False
            self._picking_bs_center = True
        else:
            self._picking_bs_center = False
        self._btn_pick_bs.setText(
            ">>> Click to set BS center <<<" if checked else "Click to Set Beamstop center"
        )

    # ==================================================================
    # TAB 3 -Differential Filter & FTH
    # ==================================================================

    def _on_filter_type_changed(self, name: str) -> None:
        self._g_gaussian.setVisible(name == "Gaussian")
        self._g_binary.setVisible(name == "Binary")

    def _on_t3_toolbar_changed(self, *_) -> None:
        """Refresh both Tab-2 images so toolbar controls behave consistently."""
        self._update_t3_holo_display()
        self._update_t3_fth_display()

    def _t3_effective_display(self, panel: str) -> dict:
        ov = self._t3_panel_display.get(panel, {})
        sx_tb, sy_tb = self._read_axis_scales(
            self._t3_tb_scale_x if hasattr(self, "_t3_tb_scale_x") else None,
            self._t3_tb_scale_y if hasattr(self, "_t3_tb_scale_y") else None,
        )
        return {
            "cmap": ov.get(
                "cmap",
                "gray",
            ),
            "invert": ov.get("invert", self._t3_tb_invert.isChecked() if hasattr(self, "_t3_tb_invert") else False),
            "scale": ov.get("scale", self._t3_tb_scale.currentText() if hasattr(self, "_t3_tb_scale") else "Linear"),
            "sx": ov.get("sx", sx_tb),
            "sy": ov.get("sy", sy_tb),
            "show_axes": ov.get("show_axes", self._t3_tb_axes.isChecked() if hasattr(self, "_t3_tb_axes") else False),
        }

    def _t3_set_panel_override(self, panel: str, key: str, value) -> None:
        self._t3_panel_display.setdefault(panel, {})[key] = value
        self._update_t3_holo_display() if panel == "holo" else self._update_t3_fth_display()

    def _t3_clear_panel_overrides(self, panel: str) -> None:
        self._t3_panel_display[panel] = {}
        self._set_status(f"Tab2 {panel} panel now follows global toolbar.")
        self._update_t3_holo_display() if panel == "holo" else self._update_t3_fth_display()

    def _on_t3_scene_clicked(self, event) -> None:
        try:
            if event.button() != Qt.MouseButton.RightButton:
                return
        except Exception:
            return
        pos = event.scenePos()
        panel = None
        if self._t3_holo_plot.getViewBox().sceneBoundingRect().contains(pos):
            panel = "holo"
        elif self._t3_fth_plot.getViewBox().sceneBoundingRect().contains(pos):
            panel = "fth"
        if panel is None:
            return
        try:
            event.accept()
        except Exception:
            pass
        self._show_t3_panel_context_menu(panel)

    def _show_t3_panel_context_menu(self, panel: str) -> None:
        eff = self._t3_effective_display(panel)
        menu = QMenu(self)
        menu.setTitle(f"Tab2 {panel.capitalize()} Display")

        m_cmap = menu.addMenu("Colormap")
        for name in FTH_COLORMAPS:
            act = QAction(name, menu)
            act.setCheckable(True)
            act.setChecked(name == eff["cmap"])
            act.triggered.connect(lambda checked=False, p=panel, n=name: self._t3_set_panel_override(p, "cmap", n))
            m_cmap.addAction(act)

        act_invert = QAction("Invert", menu)
        act_invert.setCheckable(True)
        act_invert.setChecked(bool(eff["invert"]))
        act_invert.triggered.connect(lambda checked, p=panel: self._t3_set_panel_override(p, "invert", bool(checked)))
        menu.addAction(act_invert)

        m_scale = menu.addMenu("Scale")
        for mode in ("Linear", "Log", "SymLog", "Square root"):
            act = QAction(mode, menu)
            act.setCheckable(True)
            act.setChecked(mode == eff["scale"])
            act.triggered.connect(lambda checked=False, p=panel, m=mode: self._t3_set_panel_override(p, "scale", m))
            m_scale.addAction(act)

        menu.addSeparator()
        act_sx = QAction(f"Scale X... ({eff['sx']:.4g})", menu)
        act_sy = QAction(f"Scale Y... ({eff['sy']:.4g})", menu)

        def _set_scale_x():
            v, ok = QInputDialog.getDouble(self, "Set Scale X", f"{panel.capitalize()} Scale X:", float(eff["sx"]), 1e-6, 1e6, 6)
            if ok:
                self._t3_set_panel_override(panel, "sx", float(v))

        def _set_scale_y():
            v, ok = QInputDialog.getDouble(self, "Set Scale Y", f"{panel.capitalize()} Scale Y:", float(eff["sy"]), 1e-6, 1e6, 6)
            if ok:
                self._t3_set_panel_override(panel, "sy", float(v))

        act_sx.triggered.connect(_set_scale_x)
        act_sy.triggered.connect(_set_scale_y)
        menu.addAction(act_sx)
        menu.addAction(act_sy)

        act_axes = QAction("Show Axes", menu)
        act_axes.setCheckable(True)
        act_axes.setChecked(bool(eff["show_axes"]))
        act_axes.triggered.connect(lambda checked, p=panel: self._t3_set_panel_override(p, "show_axes", bool(checked)))
        menu.addAction(act_axes)

        menu.addSeparator()
        m_roi = menu.addMenu("ROI Profile")
        act_line = QAction("Line", menu)
        act_square = QAction("Square", menu)
        act_clear = QAction("Clear ROI", menu)
        act_line.triggered.connect(lambda checked=False, p=panel: self._activate_t3_profile_roi(p, "line"))
        act_square.triggered.connect(lambda checked=False, p=panel: self._activate_t3_profile_roi(p, "square"))
        act_clear.triggered.connect(lambda checked=False, p=panel: self._clear_t3_profile_roi(p))
        m_roi.addAction(act_line)
        m_roi.addAction(act_square)
        m_roi.addSeparator()
        m_roi.addAction(act_clear)

        menu.exec(QCursor.pos())

    def _ensure_t3_profile_window(self) -> None:
        if self._t3_profile_dialog is not None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Tab2 ROI Profile")
        dlg.resize(640, 360)
        lay = QVBoxLayout(dlg)
        self._t3_profile_plot = pg.PlotWidget()
        self._t3_profile_plot.setBackground("k")
        self._t3_profile_plot.showGrid(x=True, y=True, alpha=0.25)
        lay.addWidget(self._t3_profile_plot)
        dlg.finished.connect(self._on_t3_profile_window_closed)
        self._t3_profile_dialog = dlg

    def _on_t3_profile_window_closed(self, _result: int) -> None:
        # Requested behavior: closing the profile window removes profile ROIs.
        self._clear_t3_profile_roi(None)
        self._set_status("Tab2 ROI profile closed; profile ROIs removed.")

    def _clear_t3_profile_roi(self, panel: Optional[str]) -> None:
        panels = ("holo", "fth") if panel is None else (panel,)
        for p in panels:
            roi = self._t3_profile_roi_items.get(p)
            if roi is None:
                continue
            plot, _, _ = self._t3_panel_plot_img_data(p)
            try:
                plot.removeItem(roi)
            except Exception:
                pass
            self._t3_profile_roi_items[p] = None
            self._t3_profile_roi_kind[p] = None
        if self._t3_profile_plot is not None:
            self._t3_profile_plot.clear()

    def _t3_panel_plot_img_data(self, panel: str):
        if panel == "holo":
            return self._t3_holo_plot, self._t3_holo_img, self._t3_holo_disp_data
        return self._t3_fth_plot, self._t3_fth_img, self._t3_fth_disp_data

    def _activate_t3_profile_roi(self, panel: str, roi_kind: str) -> None:
        plot, img, data = self._t3_panel_plot_img_data(panel)
        if data is None:
            self._set_status(f"No displayed data for {panel} panel yet.", error=True)
            return

        old = self._t3_profile_roi_items.get(panel)
        if old is not None:
            try:
                plot.removeItem(old)
            except Exception:
                pass

        h, w = data.shape[:2]
        cx = w * 0.5
        cy = h * 0.5
        if roi_kind == "line":
            roi = pg.LineSegmentROI(
                [[cx - w * 0.15, cy], [cx + w * 0.15, cy]],
                pen=pg.mkPen((255, 80, 80), width=2),
            )
        else:
            side = max(20.0, min(float(h), float(w)) * 0.2)
            roi = pg.RectROI(
                [cx - side * 0.5, cy - side * 0.5],
                [side, side],
                pen=pg.mkPen((255, 200, 60), width=2),
                movable=True,
                rotatable=False,
                resizable=True,
            )
        plot.addItem(roi)
        self._t3_profile_roi_items[panel] = roi
        self._t3_profile_roi_kind[panel] = roi_kind
        roi.sigRegionChanged.connect(lambda: self._update_t3_profile_from_roi(panel))

        self._ensure_t3_profile_window()
        self._t3_profile_dialog.show()
        self._t3_profile_dialog.raise_()
        self._t3_profile_dialog.activateWindow()
        self._update_t3_profile_from_roi(panel)
        self._set_status(f"Tab2 {panel} ROI profile enabled ({roi_kind}).")

    def _update_t3_profile_from_roi(self, panel: str) -> None:
        if self._t3_profile_plot is None:
            return
        plot, img, data = self._t3_panel_plot_img_data(panel)
        roi = self._t3_profile_roi_items.get(panel)
        kind = self._t3_profile_roi_kind.get(panel)
        if data is None or roi is None or kind is None:
            return
        try:
            reg = roi.getArrayRegion(data, img)
            if reg is None:
                return
            arr = np.asarray(reg, dtype=np.float64)
            if arr.size == 0:
                return
            if kind == "line":
                y = None
                pts = self._line_roi_endpoints_image_coords(roi, img)
                if pts is not None:
                    x0, y0, x1, y1 = pts
                    y = self._sample_line_profile(data, x0, y0, x1, y1)
                if y is None or np.asarray(y).size == 0:
                    y = arr.ravel() if arr.ndim == 1 else np.nanmean(arr, axis=int(np.argmin(arr.shape)))
                    if np.ndim(y) > 1:
                        y = np.asarray(y).ravel()
            else:
                if arr.ndim == 1:
                    y = arr
                else:
                    y = np.nanmean(arr, axis=0)
            x = np.arange(y.size, dtype=np.float64)
            self._t3_profile_plot.clear()
            pen = pg.mkPen((80, 220, 255), width=2)
            self._t3_profile_plot.plot(x, y, pen=pen)
            self._t3_profile_plot.setTitle(f"{panel.upper()} ROI profile ({kind})")
            self._t3_profile_plot.setLabel("left", "Intensity")
            self._t3_profile_plot.setLabel("bottom", "Pixel")
        except Exception as exc:
            logging.debug("Tab2 ROI profile update failed (%s/%s): %s", panel, kind, exc)

    def _on_diff_filter_toggled(self, checked: bool) -> None:
        """Enable/disable differential slit filter via the GroupBox checkbox."""
        if checked:
            self._on_slit_changed()
        else:
            self._current_slit = 0

    def _apply_and_compute_fth(self) -> None:
        """Apply differential filter then compute FTH in one step."""
        if not self._apply_filters_only():
            return
        self._compute_fth_only()

    def _on_slit_changed(self, _=None) -> None:
        prev_slit = self._current_slit
        txt = self._slit_combo.currentText()
        if txt == "Slit 1":
            self._current_slit = 1
        elif txt == "Slit 2":
            self._current_slit = 2
        else:
            self._current_slit = 0
        self._refresh_t4_roi_selector()
        mode_changed = (prev_slit == 0) != (self._current_slit == 0)
        if mode_changed and self._xmat is not None and (
            self._Holo_S1 is not None or self._Holo2_S1 is not None
        ):
            # None-mode (normal FTH) and slit-mode (HERALDO) use different filter paths.
            # Recompute once on mode switch to avoid showing stale residual patterns.
            self._apply_filters_only()
            return
        self._update_t3_holo_display()
        self._update_t3_fth_display()
        self._update_roi_rects()

    @staticmethod
    def _estimate_balance_ratio(src_l: np.ndarray, src_r: np.ndarray) -> float:
        """Estimate scalar ratio r such that src_l ≈ r*src_r; delegates to src.recon.fth."""
        return _estimate_balance_ratio_impl(src_l, src_r)

    def _apply_filters_only(self) -> bool:
        src_l = self._CL_smooth if self._CL_smooth is not None else getattr(self, "_CL_c", None)
        src_r = self._CR_smooth if self._CR_smooth is not None else getattr(self, "_CR_c", None)
        if src_l is None or src_r is None:
            self._set_status("Load && center data first (Tab 1).", error=True)
            return False
        if self._xmat is None:
            self._set_status("Centered hologram not computed yet.", error=True)
            return False
        try:
            phi1 = self._phi1_spin.value()
            phi2 = self._phi2_spin.value()
            if self._chk_balance.isChecked():
                self._balance_ratio = self._estimate_balance_ratio(src_l, src_r)
            else:
                self._balance_ratio = 1.0
            self._balance_ratio_edit.setText(f"{self._balance_ratio:.4f}")
            diff = src_l - self._balance_ratio * src_r

            diff_filter_active = (
                getattr(self, "_g_diff_filter", None) is not None
                and self._g_diff_filter.isChecked()
                and self._current_slit != 0
            )
            if not diff_filter_active:
                # Normal-FTH path: no slit differential kernels, no slit-specific secondary filters.
                self._Holo_S1 = diff
                self._Holo_S2 = diff
                self._Holo2_S1 = diff
                self._Holo2_S2 = diff
                ftype = "None (slit disabled)"
            else:
                # Differential filter kernels (HERALDO slit path)
                df1 = _differential_filter_kernel(self._xmat, self._ymat, self._X0, self._Y0, phi1)
                df2 = _differential_filter_kernel(self._xmat, self._ymat, self._X0, self._Y0, phi2)

                self._Holo_S1 = diff * df1
                self._Holo_S2 = diff * df2

                # Secondary slit filter
                ftype = self._filter_combo.currentText()
                if ftype == "Gaussian":
                    sigma = self._filt_sigma.value()
                    shift = self._filt_shift.value()
                    try:
                        filt_for_s1 = _line_gaussian_filter(self._Nx, self._Ny, -phi2, sigma, shift)
                        filt_for_s2 = _line_gaussian_filter(self._Nx, self._Ny, -phi1, sigma, shift)
                        self._Holo2_S1 = self._Holo_S1 * filt_for_s1
                        self._Holo2_S2 = self._Holo_S2 * filt_for_s2
                    except Exception as fe:
                        logging.warning("Gaussian filter failed: %s - using unfiltered", fe)
                        self._Holo2_S1 = self._Holo_S1
                        self._Holo2_S2 = self._Holo_S2
                elif ftype == "Binary":
                    w = self._filt_width.value()
                    self._Holo2_S1 = self._Holo_S1 * self._binary_filter(phi2, w)
                    self._Holo2_S2 = self._Holo_S2 * self._binary_filter(phi1, w)
                else:
                    self._Holo2_S1 = self._Holo_S1
                    self._Holo2_S2 = self._Holo_S2

            # Invalidate previous FTH until explicit compute step.
            self._FTH_S1 = None
            self._FTH_S2 = None
            self._t4_autoleveled_key = None
            self._update_t3_holo_display()
            mode_label = "single-file" if self._single_dataset_mode else "CL/CR"
            self._set_status(
                f"Filters applied ({mode_label}) - phi1={phi1:.2f}deg, "
                f"phi2={phi2:.2f}deg, filter={ftype}, ratio={self._balance_ratio:.4f}"
            )
            return True
        except Exception as exc:
            self._set_status(f"Filter error: {exc}", error=True)
            logging.exception("FTH filtering")
            return False

    def _compute_fth_only(self) -> bool:
        if self._Holo2_S1 is None or self._Holo2_S2 is None:
            self._set_status("Apply filters first.", error=True)
            return False
        if self._xmat is None:
            self._set_status("Centered hologram not computed yet.", error=True)
            return False
        try:
            # FTH = FFT + phase correction to center
            self._FTH_S1 = _fth_transform(
                self._Holo2_S1, self._xmat, self._ymat, self._X0, self._Y0, self._Nx, self._Ny)
            self._FTH_S2 = _fth_transform(
                self._Holo2_S2, self._xmat, self._ymat, self._X0, self._Y0, self._Nx, self._Ny)

            # Auto-scale based on 95th percentile of selected FTH amplitude
            fth_ref = self._select_slit_data(self._FTH_S1, self._FTH_S2)
            self._rs_scale_base = float(np.percentile(np.abs(fth_ref), 95))
            self._rs_scale_base = max(self._rs_scale_base, 1e-9)
            self._rs_scale_slider.setValue(100)
            self._rs_scale = self._rs_scale_base
            self._t4_autoleveled_key = None

            self._update_t3_fth_display()
            self._set_status("FTH computed.")
            return True
        except Exception as exc:
            self._set_status(f"FTH compute error: {exc}", error=True)
            logging.exception("FTH computation")
            return False

    def _apply_filters_fth(self) -> None:
        """Backward-compatible helper: apply filters then compute FTH."""
        if self._apply_filters_only():
            self._compute_fth_only()

    def _binary_filter(self, phi_deg: float, width: int) -> np.ndarray:
        """Binary notch filter through (X0, Y0); delegates to src.recon.fth."""
        return _binary_filter_kernel(self._xmat, self._ymat, self._X0, self._Y0, phi_deg, width)

    def _select_slit_data(self, d1, d2):
        """Return selected slit data; when slit=None, return combined average."""
        if self._current_slit == 1:
            return d1
        if self._current_slit == 2:
            return d2
        if d1 is None and d2 is None:
            return None
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        return 0.5 * (d1 + d2)

    def _update_t3_holo_display(self) -> None:
        holo = self._select_slit_data(self._Holo2_S1, self._Holo2_S2)
        if holo is None:
            holo = self._select_slit_data(self._Holo_S1, self._Holo_S2)
        if holo is None:
            return
        eff = self._t3_effective_display("holo")
        mode = eff["scale"]
        data = self._transform_for_display(np.log1p(np.abs(holo)), mode)
        self._t3_holo_disp_data = data
        self._t3_holo_img.setImage(data)
        src_id = id(holo)
        if src_id != self._t3_holo_src_id or mode != self._t3_holo_mode:
            vmin = float(np.percentile(data, 1))
            vmax = float(np.percentile(data, 99))
            vmax = max(vmax, vmin + 1e-6)
            self._t3_holo_cached_levels = (vmin, vmax)
            self._t3_holo_src_id = src_id
            self._t3_holo_mode = mode
            self._t3_holo_hist.setLevels(*self._t3_holo_cached_levels)
        _apply_hist_colormap(self._t3_holo_hist, str(eff["cmap"]), invert=bool(eff["invert"]))
        tr = QTransform()
        tr.scale(float(eff["sx"]), float(eff["sy"]))
        self._set_items_transform([self._t3_holo_img], tr)
        self._set_plots_axes_visible([self._t3_holo_plot], bool(eff["show_axes"]))
        if self._t3_profile_roi_items.get("holo") is not None:
            self._update_t3_profile_from_roi("holo")

    def _update_t3_fth_display(self, *_) -> None:
        fth = self._select_slit_data(self._FTH_S1, self._FTH_S2)
        if fth is None:
            return

        # FT amplitude: slider multiplies the base auto-scale by 0.01x - 4x
        self._rs_scale    = self._rs_scale_base * (self._rs_scale_slider.value() / 100.0)
        # Phase rotation: slider maps [-314, 314] -> [-pi, pi] rad
        self._phase_scale = np.pi * (self._phase_scale_slider.value() / 100.0)
        # Keep entries in sync (block signals to avoid recursion)
        self._rs_scale_entry.blockSignals(True)
        self._rs_scale_entry.setText(f"{self._rs_scale:.4g}")
        self._rs_scale_entry.blockSignals(False)
        self._phase_scale_entry.blockSignals(True)
        self._phase_scale_entry.setText(f"{self._phase_scale:.4g}")
        self._phase_scale_entry.blockSignals(False)

        # Apply complex phase rotation: rotates the entire complex plane by _phase_scale
        # so that Real, Imag, Phase, and Abs all respond to the slider.
        fth_rot = fth * np.exp(1j * self._phase_scale)

        show = self._realpart_combo.currentText()
        if show == "Real":
            data   = np.real(fth_rot).astype(np.float32)
            levels = (-self._rs_scale, self._rs_scale)
        elif show == "Imag.":
            data   = np.imag(fth_rot).astype(np.float32)
            levels = (-self._rs_scale, self._rs_scale)
        elif show == "Phase":
            data   = np.angle(fth_rot).astype(np.float32)
            levels = (-np.pi, np.pi)
        else:  # Abs.
            data   = np.abs(fth_rot).astype(np.float32)
            levels = (0, self._rs_scale)

        eff = self._t3_effective_display("fth")
        mode = eff["scale"]
        data = self._transform_for_display(data, mode)
        self._t3_fth_disp_data = data
        levels = self._transform_levels(levels, mode)

        self._t3_fth_img.setImage(data)
        self._t3_fth_hist.setLevels(levels[0], levels[1])
        _apply_hist_colormap(self._t3_fth_hist, str(eff["cmap"]), invert=bool(eff["invert"]))
        tr = QTransform()
        tr.scale(float(eff["sx"]), float(eff["sy"]))
        self._set_items_transform([self._t3_fth_img], tr)
        self._set_plots_axes_visible([self._t3_fth_plot], bool(eff["show_axes"]))
        if self._t3_profile_roi_items.get("fth") is not None:
            self._update_t3_profile_from_roi("fth")
        self._update_roi_rects()
        callback = getattr(self, "_t3_fth_display_updated_callback", None)
        if callback is not None:
            try:
                callback()
            except Exception:
                logging.debug("FTH display update callback failed", exc_info=True)

    @staticmethod
    def _transform_for_display(data: np.ndarray, mode: str) -> np.ndarray:
        """Apply display-only value transform."""
        if mode == "Linear":
            return data
        if mode in ("Log", "SymLog"):
            return np.sign(data) * np.log1p(np.abs(data))
        if mode == "Square root":
            return np.sign(data) * np.sqrt(np.abs(data))
        return data

    @staticmethod
    def _transform_levels(levels: tuple[float, float], mode: str) -> tuple[float, float]:
        """Transform level limits with the same display transform."""
        arr = np.array([levels[0], levels[1]], dtype=np.float64)
        tr = FTHReconstructionTool._transform_for_display(arr, mode)
        return float(np.min(tr)), float(np.max(tr))

    @staticmethod
    def _read_axis_scales(x_edit: Optional[QLineEdit], y_edit: Optional[QLineEdit]) -> tuple[float, float]:
        """Read Scale X/Y fields; empty/invalid means 1.0."""
        def _parse(edit: Optional[QLineEdit]) -> float:
            if edit is None:
                return 1.0
            text = edit.text().strip()
            if not text:
                return 1.0
            try:
                value = float(text)
                return value if value > 0 else 1.0
            except ValueError:
                return 1.0
        return _parse(x_edit), _parse(y_edit)

    @staticmethod
    def _read_cmap_controls(
        cmap_box: Optional[QComboBox], invert_box: Optional[QCheckBox], default: str = "gray"
    ) -> tuple[str, bool]:
        cmap_name = cmap_box.currentText() if cmap_box is not None else default
        invert = invert_box.isChecked() if invert_box is not None else False
        return cmap_name, invert

    def _build_axis_transform(self, x_edit: Optional[QLineEdit], y_edit: Optional[QLineEdit]) -> QTransform:
        sx, sy = self._read_axis_scales(x_edit, y_edit)
        tr = QTransform()
        tr.scale(sx, sy)
        return tr

    @staticmethod
    def _set_items_transform(items, tr: QTransform) -> None:
        for item in items:
            item.setTransform(tr)

    @staticmethod
    def _set_plots_axes_visible(plots, show_axes: bool) -> None:
        for plot in plots:
            for ax in ("left", "bottom", "right", "top"):
                if show_axes:
                    plot.showAxis(ax)
                else:
                    plot.hideAxis(ax)

    def _on_rs_scale_text(self) -> None:
        try:
            self._rs_scale_base = float(self._rs_scale_entry.text())
            self._rs_scale_slider.setValue(100)  # reset multiplier to 1x
        except ValueError:
            self._rs_scale_entry.setText(f"{self._rs_scale:.4g}")
        self._update_t3_fth_display()

    def _on_phase_scale_text(self) -> None:
        try:
            self._phase_scale = float(self._phase_scale_entry.text())
            slider_val = int(round(self._phase_scale / np.pi * 100))
            self._phase_scale_slider.blockSignals(True)
            self._phase_scale_slider.setValue(max(-314, min(314, slider_val)))
            self._phase_scale_slider.blockSignals(False)
        except ValueError:
            self._phase_scale_entry.setText(f"{self._phase_scale:.4g}")
        self._update_t3_fth_display()

    def _on_t4_rs_text(self) -> None:
        try:
            self._t4_rs_base = float(self._t4_rs_entry.text())
            self._t4_rs_slider.blockSignals(True)
            self._t4_rs_slider.setValue(100)   # reset multiplier to 1x
            self._t4_rs_slider.blockSignals(False)
        except ValueError:
            self._t4_rs_entry.setText(f"{self._t4_rs_base:.4g}")
        self._update_t4_display()

    def _on_t4_ph_text(self) -> None:
        try:
            ph = float(self._t4_ph_entry.text())
            slider_val = int(round(ph / np.pi * 100))
            self._t4_ph_slider.blockSignals(True)
            self._t4_ph_slider.setValue(max(-314, min(314, slider_val)))
            self._t4_ph_slider.blockSignals(False)
        except ValueError:
            self._t4_ph_entry.setText(f"{self._t4_phase:.4g}")
        self._update_t4_display()

    def _on_phase_fit_toggled(self, checked: bool) -> None:
        if checked and hasattr(self, "_t4_phase_fit_win"):
            target = int(round(float(self._roi_size) * 0.9))
            target = max(self._t4_phase_fit_win.minimum(), min(self._t4_phase_fit_win.maximum(), target))
            self._t4_phase_fit_win.setValue(target)
        self._update_t4_display()

    def _on_roi_size_changed(self, v: int) -> None:
        self._roi_size = v
        self._roi_size_entry.setText(str(v))
        self._update_roi_rects()

    def _on_roi_size_text(self) -> None:
        try:
            v = int(self._roi_size_entry.text())
            self._roi_size_slider.setValue(v)
        except ValueError:
            self._roi_size_entry.setText(str(self._roi_size))

    def _on_roi_count_changed(self, v: int) -> None:
        self._roi_count = max(1, min(4, int(v)))
        for idx, btn, xed, yed in (
            (2, self._btn_roi2, self._roi2_x, self._roi2_y),
            (3, self._btn_roi3, self._roi3_x, self._roi3_y),
            (4, self._btn_roi4, self._roi4_x, self._roi4_y),
        ):
            enabled = self._roi_count >= idx
            btn.setEnabled(enabled)
            xed.setEnabled(enabled)
            yed.setEnabled(enabled)
            if not enabled and btn.isChecked():
                btn.setChecked(False)
        self._refresh_t4_roi_selector()
        if hasattr(self, "_roi1_rect") and hasattr(self, "_roi4_rect"):
            self._update_roi_rects()
        if hasattr(self, "_t4_slitroi"):
            self._update_t4_display()

    def _active_roi_slit(self) -> int:
        """Return ROI storage slit index for current mode.

        In normal-FTH mode (current_slit=None), reuse slit-1 ROI slots.
        """
        return 1 if self._current_slit == 0 else self._current_slit

    def _active_recon_slit(self) -> int:
        """Return slit index used for reconstruction display/export."""
        return self._active_roi_slit()

    def _selected_t4_roi_idx(self) -> int:
        if not hasattr(self, "_t4_slitroi"):
            return 1
        txt = self._t4_slitroi.currentText().strip()
        try:
            return int(txt.split()[-1])
        except (ValueError, IndexError):
            return 1

    def _refresh_t4_roi_selector(self) -> None:
        if not hasattr(self, "_t4_slitroi"):
            return
        current = self._selected_t4_roi_idx() if self._t4_slitroi.count() > 0 else 1
        items = [f"ROI {i}" for i in range(1, self._roi_count + 1)]
        self._t4_slitroi.blockSignals(True)
        self._t4_slitroi.clear()
        self._t4_slitroi.addItems(items)
        current = max(1, min(self._roi_count, current))
        self._t4_slitroi.setCurrentIndex(current - 1)
        self._t4_slitroi.blockSignals(False)

    def _update_roi_rects(self) -> None:
        if not hasattr(self, "_roi1_rect") or not hasattr(self, "_roi4_rect"):
            return
        show = self._chk_show_roi.isChecked()
        half = self._roi_size // 2
        active_slit = self._active_roi_slit()
        rect_map = {
            1: self._roi1_rect,
            2: self._roi2_rect,
            3: self._roi3_rect,
            4: self._roi4_rect,
        }

        for roi_idx, rect in rect_map.items():
            if roi_idx > self._roi_count:
                rect.setVisible(False)
                continue
            ctr = self._roi_centers[active_slit].get(roi_idx)
            if ctr and show:
                row, col = ctr
                rect.setPos([col - half, row - half])
                rect.setSize([self._roi_size, self._roi_size])
                rect.setVisible(True)
            else:
                rect.setVisible(False)

    def _on_roi1_toggled(self, checked: bool) -> None:
        self._set_active_roi_picker(1, checked)

    def _on_roi2_toggled(self, checked: bool) -> None:
        self._set_active_roi_picker(2, checked)

    def _on_roi3_toggled(self, checked: bool) -> None:
        self._set_active_roi_picker(3, checked)

    def _on_roi4_toggled(self, checked: bool) -> None:
        self._set_active_roi_picker(4, checked)

    def _set_active_roi_picker(self, roi_idx: int, checked: bool) -> None:
        buttons = {1: self._btn_roi1, 2: self._btn_roi2, 3: self._btn_roi3, 4: self._btn_roi4}
        states = {
            1: "_picking_roi1",
            2: "_picking_roi2",
            3: "_picking_roi3",
            4: "_picking_roi4",
        }
        if checked:
            for i, b in buttons.items():
                if i != roi_idx and b.isChecked():
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
            for i, attr in states.items():
                setattr(self, attr, i == roi_idx)
        else:
            setattr(self, states[roi_idx], False)
        for i, b in buttons.items():
            active = getattr(self, states[i], False)
            b.setText(f">>> Click FTH to set ROI {i} <<<" if active else f"ROI {i} -click to set center")

    def _on_t3_fth_clicked(self, event) -> None:
        try:
            if event.button() != Qt.MouseButton.LeftButton:
                return
        except Exception:
            return
        if not (self._picking_roi1 or self._picking_roi2 or self._picking_roi3 or self._picking_roi4):
            return
        try:
            vb  = self._t3_fth_plot.getViewBox()
            pos = event.scenePos()
            if not vb.sceneBoundingRect().contains(pos):
                return
            pt  = vb.mapSceneToView(pos)
            col = int(round(float(pt.x())))
            row = int(round(float(pt.y())))
            target_slit = self._active_roi_slit()
            active_roi = 1
            if self._picking_roi2:
                active_roi = 2
            elif self._picking_roi3:
                active_roi = 3
            elif self._picking_roi4:
                active_roi = 4
            if active_roi > self._roi_count:
                self._set_status(f"ROI {active_roi} is disabled (ROI count = {self._roi_count}).", error=True)
                return

            self._roi_centers[target_slit][active_roi] = (row, col)
            if active_roi == 1:
                self._roi1_x.setText(str(row)); self._roi1_y.setText(str(col)); self._btn_roi1.setChecked(False)
            elif active_roi == 2:
                self._roi2_x.setText(str(row)); self._roi2_y.setText(str(col)); self._btn_roi2.setChecked(False)
            elif active_roi == 3:
                self._roi3_x.setText(str(row)); self._roi3_y.setText(str(col)); self._btn_roi3.setChecked(False)
            else:
                self._roi4_x.setText(str(row)); self._roi4_y.setText(str(col)); self._btn_roi4.setChecked(False)
            mode = "None" if self._current_slit == 0 else str(self._current_slit)
            self._set_status(f"ROI {active_roi} (Slit {mode}): row={row}, col={col}")

            self._update_roi_rects()
        except Exception as exc:
            logging.debug("t3 FTH click: %s", exc)

    # ==================================================================
    # TAB 4 -Reconstruction display & export
    # ==================================================================

    def _t4_effective_display(self, panel_idx: int) -> dict:
        ov = self._t4_panel_display.get(panel_idx, {})
        return {
            "cmap": ov.get("cmap", "gray"),
            "invert": ov.get("invert", False),
            "scale": ov.get("scale", "Linear"),
            "sx": ov.get("sx", 1.0),
            "sy": ov.get("sy", 1.0),
            "show_axes": ov.get("show_axes", False),
        }

    def _t4_set_panel_override(self, panel_idx: int, key: str, value) -> None:
        self._t4_panel_display.setdefault(panel_idx, {})[key] = value
        self._update_t4_display()

    def _t4_clear_panel_overrides(self, panel_idx: int) -> None:
        self._t4_panel_display[panel_idx] = {}
        self._update_t4_display()

    def _on_t4_scene_clicked(self, event) -> None:
        try:
            if event.button() != Qt.MouseButton.RightButton:
                return
        except Exception:
            return
        pos = event.scenePos()
        panel_idx = None
        for idx, plot in enumerate(self._t4_plots):
            if plot.getViewBox().sceneBoundingRect().contains(pos):
                panel_idx = idx
                break
        if panel_idx is None:
            return
        try:
            event.accept()
        except Exception:
            pass
        self._show_t4_panel_context_menu(panel_idx)

    def _show_t4_panel_context_menu(self, panel_idx: int) -> None:
        labels = ["Real", "Imag.", "Phase", "Abs."]
        title = labels[panel_idx] if 0 <= panel_idx < len(labels) else f"Panel {panel_idx + 1}"
        eff = self._t4_effective_display(panel_idx)

        menu = QMenu(self)
        menu.setTitle(f"Tab3 {title} Display")

        m_cmap = menu.addMenu("Colormap")
        for name in FTH_COLORMAPS:
            act = QAction(name, menu)
            act.setCheckable(True)
            act.setChecked(name == eff["cmap"])
            act.triggered.connect(
                lambda checked=False, i=panel_idx, n=name: self._t4_set_panel_override(i, "cmap", n)
            )
            m_cmap.addAction(act)

        act_invert = QAction("Invert", menu)
        act_invert.setCheckable(True)
        act_invert.setChecked(bool(eff["invert"]))
        act_invert.triggered.connect(
            lambda checked, i=panel_idx: self._t4_set_panel_override(i, "invert", bool(checked))
        )
        menu.addAction(act_invert)

        m_scale = menu.addMenu("Scale")
        for mode in ("Linear", "Log", "SymLog", "Square root"):
            act = QAction(mode, menu)
            act.setCheckable(True)
            act.setChecked(mode == eff["scale"])
            act.triggered.connect(
                lambda checked=False, i=panel_idx, m=mode: self._t4_set_panel_override(i, "scale", m)
            )
            m_scale.addAction(act)

        menu.addSeparator()
        act_sx = QAction(f"Scale X... ({eff['sx']:.4g})", menu)
        act_sy = QAction(f"Scale Y... ({eff['sy']:.4g})", menu)

        def _set_scale_x():
            v, ok = QInputDialog.getDouble(
                self, "Set Scale X", f"{title} Scale X:", float(eff["sx"]), 1e-6, 1e6, 6
            )
            if ok:
                self._t4_set_panel_override(panel_idx, "sx", float(v))

        def _set_scale_y():
            v, ok = QInputDialog.getDouble(
                self, "Set Scale Y", f"{title} Scale Y:", float(eff["sy"]), 1e-6, 1e6, 6
            )
            if ok:
                self._t4_set_panel_override(panel_idx, "sy", float(v))

        act_sx.triggered.connect(_set_scale_x)
        act_sy.triggered.connect(_set_scale_y)
        menu.addAction(act_sx)
        menu.addAction(act_sy)

        act_axes = QAction("Show Axes", menu)
        act_axes.setCheckable(True)
        act_axes.setChecked(bool(eff["show_axes"]))
        act_axes.triggered.connect(
            lambda checked, i=panel_idx: self._t4_set_panel_override(i, "show_axes", bool(checked))
        )
        menu.addAction(act_axes)

        menu.addSeparator()
        act_reset = QAction("Reset This Panel", menu)
        act_reset.triggered.connect(lambda checked=False, i=panel_idx: self._t4_clear_panel_overrides(i))
        menu.addAction(act_reset)

        menu.exec(QCursor.pos())

    def _compute_roi(self, slit: int, roi_idx: int) -> Optional[np.ndarray]:
        """Return the (possibly corrected) complex ROI crop, or None."""
        fth = self._FTH_S1 if slit == 1 else self._FTH_S2
        if fth is None:
            return None
        ctr = self._roi_centers[slit].get(roi_idx)
        if ctr is None:
            return None
        row, col = ctr
        half = self._roi_size // 2
        r0, r1 = max(0, row - half), min(fth.shape[0], row + half)
        c0, c1 = max(0, col - half), min(fth.shape[1], col + half)
        if r0 >= r1 or c0 >= c1:
            return None
        roi = fth[r0:r1, c0:c1].copy()
        if self._chk_inv_contrast.isChecked():
            roi = -roi
        if self._chk_inv_realimag.isChecked():
            roi = -roi * np.exp(1j * np.pi / 2.0)
        if self._chk_gauss_filter.isChecked():
            from scipy.ndimage import gaussian_filter
            sigma = self._t4_gauss_sigma.value()
            roi = (gaussian_filter(np.real(roi), sigma)
                   + 1j * gaussian_filter(np.imag(roi), sigma))
        # Optional ROI phase-fit correction:
        # estimate a single phase that minimizes imaginary energy in a center window,
        # then rotate the whole ROI by that phase.
        phi_fit = 0.0
        if self._chk_phase_fit.isChecked():
            phi_fit = self._estimate_roi_phase_rotation(roi)
            if np.isfinite(phi_fit):
                roi = roi * np.exp(-1j * phi_fit)
            else:
                phi_fit = 0.0
        self._last_roi_phase_fit = float(phi_fit)
        self._t4_phase_fit_label.setText(f"{self._last_roi_phase_fit:.4f} rad")
        return roi

    def _estimate_roi_phase_rotation(self, roi: np.ndarray) -> float:
        """Estimate global phase rotation (rad) from a central ROI window.

        Uses a least-squares criterion that minimizes imaginary-part energy after
        rotation z' = z * exp(-1j*phi). Closed-form solution:
            phi = 0.5 * atan2(2*sum(Re*Im), sum(Re^2 - Im^2))
        """
        if roi is None or roi.size == 0:
            return 0.0
        h, w = roi.shape[:2]
        win = int(self._t4_phase_fit_win.value()) if hasattr(self, "_t4_phase_fit_win") else 70
        win = max(4, min(win, h, w))
        half = win // 2
        cy, cx = h // 2, w // 2
        r0, r1 = max(0, cy - half), min(h, cy + half)
        c0, c1 = max(0, cx - half), min(w, cx + half)
        patch = roi[r0:r1, c0:c1]
        if patch.size == 0:
            return 0.0

        re = np.real(patch).astype(np.float64, copy=False).ravel()
        im = np.imag(patch).astype(np.float64, copy=False).ravel()
        m = np.isfinite(re) & np.isfinite(im)
        if not np.any(m):
            return 0.0
        re = re[m]
        im = im[m]
        s_rr = float(np.sum(re * re))
        s_ii = float(np.sum(im * im))
        s_ri = float(np.sum(re * im))
        return 0.5 * float(np.arctan2(2.0 * s_ri, s_rr - s_ii))

    def _update_t4_display(self, *_) -> None:
        slit = self._active_recon_slit()
        roi_idx = self._selected_t4_roi_idx()
        if roi_idx > self._roi_count:
            roi_idx = 1
        roi = self._compute_roi(slit, roi_idx)
        if roi is None:
            return

        # Auto-detect amplitude scale the first time each ROI/slit combination is shown.
        # This ensures the initial display is meaningful regardless of the slider default.
        autokey = (slit, roi_idx)
        if self._t4_autoleveled_key != autokey:
            self._t4_autoleveled_key = autokey
            auto_rs = float(np.percentile(np.abs(roi), 99))
            if auto_rs > 0:
                self._t4_rs_base = auto_rs
                self._t4_rs_slider.blockSignals(True)
                self._t4_rs_slider.setValue(100)   # 1x base
                self._t4_rs_slider.blockSignals(False)

        rs = self._t4_rs_base * (self._t4_rs_slider.value() / 100.0)
        ph = np.pi * (self._t4_ph_slider.value() / 100.0)

        # Keep text entries in sync with slider positions
        self._t4_rs_entry.blockSignals(True)
        self._t4_rs_entry.setText(f"{rs:.4g}")
        self._t4_rs_entry.blockSignals(False)
        self._t4_ph_entry.blockSignals(True)
        self._t4_ph_entry.setText(f"{ph:.4g}")
        self._t4_ph_entry.blockSignals(False)
        # Apply complex phase rotation: all four panels respond to the slider
        roi_rot = roi * np.exp(1j * ph)
        components = [
            (np.real(roi_rot), (-rs,  rs),   "Real"),
            (np.imag(roi_rot), (-rs,  rs),   "Imag."),
            (np.angle(roi_rot), (-np.pi, np.pi), "Phase"),
            (np.abs(roi_rot),   (0,   rs),   "Abs."),
        ]
        for idx, ((data, levels, label), img, hist, plot) in enumerate(
            zip(components, self._t4_imgs, self._t4_hists, self._t4_plots)
        ):
            eff = self._t4_effective_display(idx)
            mode = str(eff["scale"])
            disp_data = self._transform_for_display(data.astype(np.float32), mode)
            disp_levels = self._transform_levels(levels, mode)
            self._t4_disp_data[idx] = disp_data

            img.setImage(disp_data)
            hist.setLevels(disp_levels[0], disp_levels[1])
            _apply_hist_colormap(hist, str(eff["cmap"]), invert=bool(eff["invert"]))

            tr = QTransform()
            tr.scale(float(eff["sx"]), float(eff["sy"]))
            img.setTransform(tr)
            self._set_plots_axes_visible([plot], bool(eff["show_axes"]))

    def _get_t4_export_components(self) -> Optional[dict[str, np.ndarray]]:
        """Return computed arrays for Real/Imag/Phase/Abs after all recon corrections."""
        slit = self._active_recon_slit()
        roi_idx = self._selected_t4_roi_idx()
        if roi_idx > self._roi_count:
            roi_idx = 1
        roi = self._compute_roi(slit, roi_idx)
        if roi is None:
            self._set_status("No ROI computed -set ROI center first (Tab 3).", error=True)
            return
        ph = np.pi * (self._t4_ph_slider.value() / 100.0)
        roi_rot = roi * np.exp(1j * ph)
        return {
            "real": np.real(roi_rot).astype(np.float32),
            "imag": np.imag(roi_rot).astype(np.float32),
            "phase": np.angle(roi_rot).astype(np.float32),
            "abs": np.abs(roi_rot).astype(np.float32),
        }

    def _selected_export_component_name(self) -> str:
        txt = self._exp_target_combo.currentText().strip().lower()
        if txt.startswith("real"):
            return "real"
        if txt.startswith("imag"):
            return "imag"
        if txt.startswith("phase"):
            return "phase"
        return "abs"

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

    def _current_export_name_base(self) -> str:
        """Build base name: head_num1_num2 from current CL/CR dataset selections."""
        cl_entry = self._cl_combo.get_entry(opened_files=self._opened_files)
        cr_entry = self._cr_combo.get_entry(opened_files=self._opened_files)
        if cl_entry and cr_entry:
            head1, n1 = self._scan_head_and_number(str(cl_entry[0]))
            head2, n2 = self._scan_head_and_number(str(cr_entry[0]))
            head = head1 if head1 else (head2 if head2 else "scan")
            return f"{head}_{n1}_{n2}"
        single_entry = cl_entry or cr_entry
        if single_entry:
            head, n1 = self._scan_head_and_number(str(single_entry[0]))
            return f"{head}_{n1}_single"
        return "scan_0000_0000"

    def _component_display_levels(self, name: str) -> tuple[float, float]:
        """Levels used to map data to 8-bit image (follows current FT amplitude semantics)."""
        rs = self._t4_rs_base * (self._t4_rs_slider.value() / 100.0)
        if name in ("real", "imag"):
            return -float(rs), float(rs)
        if name == "phase":
            return -float(np.pi), float(np.pi)
        return 0.0, float(rs)

    @staticmethod
    def _component_to_qimage(arr: np.ndarray, levels: tuple[float, float]) -> QImage:
        """Convert array to grayscale QImage using provided levels."""
        lo, hi = float(levels[0]), float(levels[1])
        if not np.isfinite(lo):
            lo = float(np.nanmin(arr))
        if not np.isfinite(hi):
            hi = float(np.nanmax(arr))
        if hi <= lo:
            hi = lo + 1e-12
        norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        img_u8 = (norm * 255.0).astype(np.uint8)
        h, w = img_u8.shape
        qimg = QImage(img_u8.data, w, h, img_u8.strides[0], QImage.Format.Format_Grayscale8)
        return qimg.copy()

    def _copy_selected_component_jpeg(self) -> None:
        """Copy selected reconstructed component as JPEG image to clipboard."""
        components = self._get_t4_export_components()
        if components is None:
            return
        name = self._selected_export_component_name()
        arr = components.get(name)
        if arr is None:
            self._set_status("Invalid export component selected.", error=True)
            return
        qimg = self._component_to_qimage(arr, self._component_display_levels(name))
        try:
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            ok = qimg.save(buf, "JPEG", 95)
            buf.close()
            if not ok:
                self._set_status("Failed to encode JPEG for clipboard.", error=True)
                return
            mime = QApplication.clipboard().mimeData()
            if mime is None:
                from PyQt6.QtCore import QMimeData
                mime = QMimeData()
            else:
                from PyQt6.QtCore import QMimeData
                m2 = QMimeData()
                mime = m2
            mime.setImageData(qimg)
            mime.setData("image/jpeg", ba)
            QApplication.clipboard().setMimeData(mime)
            QApplication.clipboard().setPixmap(QPixmap.fromImage(qimg))
            self._set_status(f"Copied JPEG: {name}")
        except Exception as exc:
            self._set_status(f"Copy failed: {exc}", error=True)
            logging.exception("FTH copy JPEG")

    def _save_selected_component(self) -> None:
        """Save selected reconstructed component. Format is chosen in dialog."""
        components = self._get_t4_export_components()
        if components is None:
            return
        name = self._selected_export_component_name()
        arr = components.get(name)
        if arr is None:
            self._set_status("Invalid export component selected.", error=True)
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {name}",
            f"{self._current_export_name_base()}_{name}.png",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        p = pathlib.Path(path)
        try:
            qimg = self._component_to_qimage(arr, self._component_display_levels(name))
            ok = qimg.save(str(p))
            if not ok:
                raise RuntimeError("Image save failed")
            self._set_status(f"Saved image: {p.name}")
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", error=True)
            logging.exception("FTH save selected component")

    def _save_all_components(self) -> None:
        """Save all four reconstructed components. Format is chosen in dialog."""
        components = self._get_t4_export_components()
        if components is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save all reconstruction components",
            f"{self._current_export_name_base()}.png",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        p = pathlib.Path(path)
        ext = p.suffix.lower()
        try:
            stem = p.with_suffix("")
            for n, a in components.items():
                qimg = self._component_to_qimage(a, self._component_display_levels(n))
                out_path = pathlib.Path(f"{stem}_{n}{ext}")
                ok = qimg.save(str(out_path))
                if not ok:
                    raise RuntimeError(f"Image save failed for {out_path.name}")
            self._set_status(f"Saved all images: {stem.name}_*.{ext.lstrip('.')}")
        except Exception as exc:
            self._set_status(f"Save all failed: {exc}", error=True)
            logging.exception("FTH save all components")

    # ==================================================================
    # Tab switching refresh
    # ==================================================================

    def _on_tab_changed(self, idx: int) -> None:
        """Refresh the display whenever the user switches tabs."""
        if idx == 0:   # Alignment (merged)
            self._update_t1_main_display()
            self._update_slit_lines()
            self._update_center_marker()
            self._update_bs_overlay()
        elif idx == 1:  # Filter & FTH
            self._update_t3_holo_display()
            self._update_t3_fth_display()
        elif idx == 2:  # Reconstruction
            self._update_t4_display()

    # ==================================================================
    # Utilities
    # ==================================================================

    def _set_status(self, msg: str, error: bool = False) -> None:
        color = "#e05050" if error else "#90ee90"
        self._status_label.setStyleSheet(
            f"color:{color}; font-size:11px; padding:3px 6px; "
            "border-top:1px solid #444; background:#1a1a1a;"
        )
        self._status_label.setText(msg)
        if error:
            logging.warning("FTH: %s", msg)
        else:
            logging.info("FTH: %s", msg)

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(500)
        super().closeEvent(event)





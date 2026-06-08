"""XRMS Analyze — unified X-ray resonant magnetic scattering analysis tool.

Merges the former Radial Profile and Time Resolve tools into one dialog that
shares a single image viewer and the pure ``src/recon`` backend:

  * Image & Region tab — load one stack (+ optional reference image with a
    per-frame sum/difference operation), navigate 3-D frames, pick a region
    (rectangle / circle / disk-arc), incidence correction + display origin,
    and live I(r) / I(theta) / I(t) plots.
  * Curve Fitting tab — fit the active profile (radial, angular or time) with the
    shared model library and polynomial background subtraction.
  * Frame Analysis tab — fit the chosen model on every frame (radial/angular).
"""

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

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
from src.lib_h5.file_validator import is_hdf5_file
from src.recon import curve_fit as cf
from src.recon import incidence as inc
from src.recon import profiles as pr


class _SlotCombo(QComboBox):
    """Drop target for a data slot.

    Accepts drag-drop of dataset keys ("file::path", e.g. from the HDF5 tree) and
    of file paths (URLs). Keeps an ordered list of entries; the dropdown browses
    them. Files (entries without "::") are read at the slot's shared Address.
    """

    sigEntriesChanged = pyqtSignal()

    def __init__(self, placeholder: str = "— drop dataset / files —", parent=None) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._entries: list[str] = []
        self.setAcceptDrops(True)
        self.setMinimumWidth(120)
        self.addItem(placeholder, None)

    # ---- drag & drop ---- #
    def dragEnterEvent(self, e) -> None:
        md = e.mimeData()
        if md is not None and (md.hasText() or md.hasUrls()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:
        e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        md = e.mimeData()
        added: list[str] = []
        if md.hasUrls():
            for u in md.urls():
                p = u.toLocalFile()
                if p:
                    added.append(p)
        if md.hasText():
            for line in md.text().splitlines():
                line = line.strip()
                if line and line not in added:
                    added.append(line)
        for a in added:
            if a and a not in self._entries:
                self._entries.append(a)
        self._rebuild()
        e.acceptProposedAction()
        self.sigEntriesChanged.emit()

    # ---- entries ---- #
    @staticmethod
    def _label(entry: str) -> str:
        if "::" in entry:
            fp, ds = entry.rsplit("::", 1)
            return f"{pathlib.Path(fp).name} :: {ds.strip().split('/')[-1]}"
        return pathlib.Path(entry).name

    def _rebuild(self) -> None:
        self.blockSignals(True)
        self.clear()
        if not self._entries:
            self.addItem(self._placeholder, None)
        else:
            for ent in self._entries:
                self.addItem(self._label(ent), ent)
        self.blockSignals(False)

    def entries(self) -> list[str]:
        return list(self._entries)

    def set_entries(self, entries: list[str]) -> None:
        self._entries = list(entries)
        self._rebuild()
        self.sigEntriesChanged.emit()

    def clear_entries(self) -> None:
        self._entries = []
        self._rebuild()
        self.sigEntriesChanged.emit()


class _AddrLineEdit(QLineEdit):
    """Address field that accepts a dropped dataset and keeps its path part.

    Dropping a "file::dataset" token (e.g. a leaf from the HDF5 tree) sets the
    field to just the dataset path; a bare path with no "::" is taken as-is.
    Emits ``editingFinished`` on drop so the slot reloads immediately.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e) -> None:
        md = e.mimeData()
        if md is not None and md.hasText():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:
        e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        md = e.mimeData()
        text = md.text().strip() if md is not None and md.hasText() else ""
        first = text.splitlines()[0].strip() if text else ""
        if "::" in first:
            first = first.rsplit("::", 1)[1].strip()
        if first:
            self.setText(first)
            e.acceptProposedAction()
            self.editingFinished.emit()
        else:
            e.ignore()


class XRMSAnalyzeTool(QDialog):
    """Unified XRMS analysis workspace (radial / angular / time-resolved)."""

    _N_ANGLE_BINS = 360
    _TXT_DS = "/data"   # virtual dataset name the main window assigns to text files

    def __init__(
        self,
        opened_files=(),
        dataset_full_keys_2d: list[str] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._opened_files = tuple(opened_files)
        self._dataset_full_keys_2d = dataset_full_keys_2d or []

        # Data model: 6 input slots. CL/CR/BG hold (n, H, W) image stacks; Delay/Norm
        # hold 1-D value arrays. _combined = delay-aligned/normalised/operated stack
        # (pre-incidence); _data = post-incidence stack actually analyzed.
        # Image slots have drag-drop combos; Delay/Norm are read from the same
        # CL/CR files at an address (no separate combo).
        self._slot_names = ("CL", "CL_BG", "CR", "CR_BG")
        self._slots: dict[str, np.ndarray | None] = {s: None for s in self._slot_names}
        self._combined: np.ndarray | None = None
        self._combined_delay: np.ndarray | None = None   # delay value per combined frame
        self._data: np.ndarray | None = None
        self._n_frames: int = 0
        self._norm_enabled: bool = False

        # Incidence correction (resampling)
        self._incidence_active: bool = False
        self._inc_theta_x: float = 0.0
        self._inc_theta_y: float = 0.0

        # Beam stop mask (applied after incidence; masked pixels excluded from analysis)
        self._beamstop: np.ndarray | None = None      # effective bool (H, W), True = masked
        self._brush_mask: np.ndarray | None = None    # bool (H, W) brush strokes
        self._mask_shapes: list[dict] = []            # {type: circle|square, item: pg ROI}
        self._mask_counter: dict[str, int] = {"circle": 0, "square": 0}
        self._mask_overlay = None                     # pg.ImageItem overlay
        self._brush_active: bool = False
        self._brush_size: int = 8

        # ROIs — list of dicts (see _add_roi). Ring/Sector share the global Center.
        self._rois: list[dict] = []
        self._roi_counter: dict[str, int] = {"ring": 0, "circle": 0, "sector": 0}
        self._roi_overlay_items: list = []         # pg items currently drawn on the image
        self._roi_handles: dict[str, object] = {}  # draggable handles for the selected ring/sector

        # Geometry caches (avoid re-allocating big grids each recompute -> perf).
        self._grid_cache: tuple | None = None      # ((h, w), yy, xx)
        self._polar_cache: tuple | None = None     # (key, r, angles)

        # Profiles for the current frame (shared with the Curve Fitting tab)
        self._radial_x: np.ndarray | None = None
        self._radial_y: np.ndarray | None = None
        self._angular_x: np.ndarray | None = None
        self._angular_y: np.ndarray | None = None
        self._time_x: np.ndarray | None = None
        self._time_y: np.ndarray | None = None

        # Curve fitting state
        self._fit_param_spins: list[QDoubleSpinBox] = []
        self._bg_enabled: bool = False
        self._guess_attempt: int = 0

        # Display origin / incidence
        self._display_origin: tuple[float, float] | None = None
        self._origin_pick_active: bool = False

        # Frame analysis (prevent GC)
        self._frame_analysis_inner = None

        # Page 3 (batch over frames) — per-frame fit results stored for the page-4 summary.
        self._b3_results: list[dict] = []
        self._b3_bg_xs: list[float] | None = None
        self._b3_range_xs: list[float] | None = None
        self._b3_fit_items: list = []
        self._b3_overlay_items: list = []
        # Page 4 (summary) — 4 plots, each with a parameter selector.
        self._b4_plots: list = []
        self._b4_combos: list = []

        # Debounce timer: coalesce expensive profile/time recompute during ROI drags.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(90)
        self._refresh_timer.timeout.connect(self._do_deferred_refresh)

        # Debounce mask-shape moves (rasterizing a big mask each drag event is heavy).
        self._mask_timer = QTimer(self)
        self._mask_timer.setSingleShot(True)
        self._mask_timer.setInterval(90)
        self._mask_timer.timeout.connect(self._recompute_beamstop)
        self._shapes_raster: np.ndarray | None = None  # cached union of mask shapes

        self.setWindowTitle("Time Resolved XRMS")
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1400, 900)

        self._build_ui()

    # ================================================================== #
    # UI construction                                                      #
    # ================================================================== #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs)

        self._tabs.addTab(self._build_image_region_tab(), "Image && Region")

        tab2 = QWidget()
        self._tabs.addTab(tab2, "Curve Fitting")
        self._build_curve_fit_tab(tab2)

        self._tabs.addTab(self._build_batch_tab(), "Batch")
        self._tabs.addTab(self._build_summary_tab(), "Summary")

        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _build_image_region_tab(self) -> QWidget:
        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(4, 4, 4, 4)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(h_splitter)

        # ── Left controls (scrollable) ── #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(380)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(8)

        # Datasets — 6 inputs. Drag a dataset (file::path) or multiple files onto each
        # combo; for files, the data at "Address" is read from every file.
        g_ds = QGroupBox("Datasets")
        fl_ds = QFormLayout(g_ds)
        fl_ds.setContentsMargins(6, 4, 6, 4)
        fl_ds.setSpacing(4)

        self._slot_combo: dict[str, _SlotCombo] = {}
        self._slot_addr: dict[str, QLineEdit] = {}
        self._slot_axis: dict[str, QSpinBox] = {}
        self._slot_shape: dict[str, QLabel] = {}
        slot_help = {
            "CL": "Circular-left stack", "CL_BG": "CL background (per-frame subtract)",
            "CR": "Circular-right stack", "CR_BG": "CR background (per-frame subtract)",
        }
        # Image slots: drag-drop combos sharing one address/axis (CL+CR, CL_BG+CR_BG).
        for group in [("CL", "CR"), ("CL_BG", "CR_BG")]:
            addr = _AddrLineEdit()
            addr.setPlaceholderText("address — type or drop a dataset here")
            axis = QSpinBox()
            axis.setRange(0, 5)
            axis.setPrefix("ax ")
            axis.setFixedWidth(52)
            axis.setToolTip("Frame axis to split a multi-image dataset by.")
            addr.editingFinished.connect(lambda g=group: [self._load_slot(s) for s in g])
            axis.valueChanged.connect(lambda _v, g=group: [self._load_slot(s) for s in g])
            for slot in group:
                combo = _SlotCombo()
                combo.setToolTip(f"{slot_help[slot]}.\nDrop a dataset (file::path) or one/more files here.")
                combo.sigEntriesChanged.connect(lambda s=slot: self._load_slot(s))
                lbl = QLabel("—")
                lbl.setStyleSheet("color:#666; font-style:italic; font-size:11px;")
                lbl.setFixedWidth(78)
                lbl.setToolTip("Loaded shape")
                crow = QHBoxLayout()
                crow.setContentsMargins(0, 0, 0, 0)
                crow.setSpacing(4)
                crow.addWidget(combo, 1)
                crow.addWidget(lbl)
                fl_ds.addRow(f"{slot}:", crow)
                self._slot_combo[slot] = combo
                self._slot_addr[slot] = addr
                self._slot_axis[slot] = axis
                self._slot_shape[slot] = lbl
            sub = QHBoxLayout()
            sub.addWidget(addr, 1)
            sub.addWidget(axis)
            fl_ds.addRow("addr:", sub)

        # Delay / Norm: dataset paths read from each CL/CR file (no combo). The
        # delay aligns CL↔CR by value; the result is then sorted by delay.
        self._delay_addr = QLineEdit()
        self._delay_addr.setPlaceholderText("delay dataset path in each CL/CR file")
        self._delay_addr.setToolTip("Per-frame delay read from the CL/CR data files; "
                                    "CL and CR are paired by matching delay value.")
        self._delay_addr.editingFinished.connect(self._recompute_stack)
        fl_ds.addRow("Delay addr:", self._delay_addr)

        self._norm_addr = QLineEdit()
        self._norm_addr.setPlaceholderText("norm-factor dataset path in each CL/CR file")
        self._norm_addr.setToolTip("Per-frame normalisation factor read from the CL/CR data files.")
        self._norm_addr.editingFinished.connect(self._recompute_stack)
        fl_ds.addRow("Norm addr:", self._norm_addr)

        self._chk_norm = QCheckBox("Apply normalisation (÷ norm per frame)")
        self._chk_norm.toggled.connect(self._on_norm_toggled)
        fl_ds.addRow(self._chk_norm)

        self._combo_operation = QComboBox()
        self._combo_operation.addItem("CL",                              "cl")
        self._combo_operation.addItem("CR",                              "cr")
        self._combo_operation.addItem("Difference  CL − CR",             "diff")
        self._combo_operation.addItem("Sum  CL + CR",                    "sum")
        self._combo_operation.addItem("Asymmetry  (CL−CR)/(CL+CR)",      "asym")
        self._combo_operation.setToolTip(
            "Per delay: cl=(CL−CL_BG)/Norm, cr=(CR−CR_BG)/Norm, then this operation.\n"
            "CL and CR are paired by matching delay value."
        )
        self._combo_operation.currentIndexChanged.connect(self._recompute_stack)
        fl_ds.addRow("Operation:", self._combo_operation)
        ll.addWidget(g_ds)

        # Frame navigation
        self._g_frame = QGroupBox("Frame Navigation")
        fl_fr = QFormLayout(self._g_frame)
        fl_fr.setContentsMargins(6, 4, 6, 4)
        row_fr = QHBoxLayout()
        self._sl_frame = QSlider(Qt.Orientation.Horizontal)
        self._sl_frame.setRange(0, 0)
        self._spin_frame = QSpinBox()
        self._spin_frame.setRange(0, 0)
        self._spin_frame.setFixedWidth(60)
        row_fr.addWidget(self._sl_frame)
        row_fr.addWidget(self._spin_frame)
        fl_fr.addRow("Frame:", row_fr)
        self._lbl_frame_info = QLabel("— / —")
        self._lbl_frame_info.setStyleSheet("color:#555;")
        fl_fr.addRow("", self._lbl_frame_info)
        self._sl_frame.valueChanged.connect(self._on_frame_changed)
        self._spin_frame.valueChanged.connect(self._sl_frame.setValue)
        ll.addWidget(self._g_frame)
        self._g_frame.setVisible(False)

        # ROIs — add multiple Ring / Circle / Sector; the selected one drives
        # the right-side profile and the Curve Fitting tab.
        g_roi = QGroupBox("ROIs")
        roi_v = QVBoxLayout(g_roi)
        roi_v.setContentsMargins(6, 4, 6, 4)
        roi_v.setSpacing(5)

        add_row = QHBoxLayout()
        for label, kind in (("+ Ring", "ring"), ("+ Circle", "circle"), ("+ Sector", "sector")):
            btn = QPushButton(label)
            btn.setAutoDefault(False)
            btn.clicked.connect(lambda _=False, k=kind: self._add_roi(k))
            add_row.addWidget(btn)
        roi_v.addLayout(add_row)

        sel_row = QHBoxLayout()
        self._roi_combo = QComboBox()
        self._roi_combo.setToolTip("Select a ROI to edit and to show its profile on the right.")
        self._roi_combo.currentIndexChanged.connect(self._on_roi_combo_changed)
        sel_row.addWidget(self._roi_combo, 1)
        self._btn_remove_roi = QPushButton("Remove")
        self._btn_remove_roi.setAutoDefault(False)
        self._btn_remove_roi.clicked.connect(self._remove_selected_roi)
        sel_row.addWidget(self._btn_remove_roi)
        roi_v.addLayout(sel_row)

        # Per-ROI parameter controls (shown only when a ROI is selected).
        self._roi_params = QWidget()
        rp = QFormLayout(self._roi_params)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(4)

        row_ri = QHBoxLayout()
        self._sl_ri = QSlider(Qt.Orientation.Horizontal)
        self._sl_ri.setRange(0, 1000)
        self._lbl_ri = QLabel("0 px")
        self._lbl_ri.setMinimumWidth(70)
        row_ri.addWidget(self._sl_ri)
        row_ri.addWidget(self._lbl_ri)
        self._cap_inner = QLabel("Inner r:")
        rp.addRow(self._cap_inner, row_ri)
        row_ro = QHBoxLayout()
        self._sl_ro = QSlider(Qt.Orientation.Horizontal)
        self._sl_ro.setRange(0, 1000)
        self._lbl_ro = QLabel("0 px")
        self._lbl_ro.setMinimumWidth(70)
        row_ro.addWidget(self._sl_ro)
        row_ro.addWidget(self._lbl_ro)
        self._cap_outer = QLabel("Outer r:")
        rp.addRow(self._cap_outer, row_ro)
        self._sl_ri.valueChanged.connect(self._on_roi_param_changed)
        self._sl_ro.valueChanged.connect(self._on_roi_param_changed)

        # Angle controls — only relevant for Sector.
        self._roi_angles = QWidget()
        ang_form = QFormLayout(self._roi_angles)
        ang_form.setContentsMargins(0, 0, 0, 0)
        ang_form.setSpacing(4)
        row_amin = QHBoxLayout()
        self._sl_amin = QSlider(Qt.Orientation.Horizontal)
        self._sl_amin.setRange(-180, 180)
        self._lbl_amin = QLabel("0°")
        self._lbl_amin.setMinimumWidth(45)
        row_amin.addWidget(self._sl_amin)
        row_amin.addWidget(self._lbl_amin)
        ang_form.addRow("θ min:", row_amin)
        row_amax = QHBoxLayout()
        self._sl_amax = QSlider(Qt.Orientation.Horizontal)
        self._sl_amax.setRange(-180, 180)
        self._sl_amax.setValue(180)
        self._lbl_amax = QLabel("180°")
        self._lbl_amax.setMinimumWidth(45)
        row_amax.addWidget(self._sl_amax)
        row_amax.addWidget(self._lbl_amax)
        ang_form.addRow("θ max:", row_amax)
        self._sl_amin.valueChanged.connect(self._on_roi_param_changed)
        self._sl_amax.valueChanged.connect(self._on_roi_param_changed)
        rp.addRow(self._roi_angles)

        self._combo_roi_mode = QComboBox()
        self._combo_roi_mode.addItem("Radial  I(r)", "radial")
        self._combo_roi_mode.addItem("Azimuthal  I(θ)", "angular")
        self._combo_roi_mode.setToolTip("Which profile this ROI feeds to the Curve Fitting tab.")
        self._combo_roi_mode.currentIndexChanged.connect(self._on_roi_param_changed)
        self._cap_mode = QLabel("Profile:")
        rp.addRow(self._cap_mode, self._combo_roi_mode)

        # Phase shift: rolls the azimuthal profile along θ (only shown for I(θ)).
        self._cap_phase = QLabel("Phase shift:")
        self._spin_phase = QSpinBox()
        self._spin_phase.setRange(-180, 180)
        self._spin_phase.setSingleStep(5)
        self._spin_phase.setSuffix("°")
        self._spin_phase.setToolTip("Offset the azimuthal profile along the angle axis.")
        self._spin_phase.valueChanged.connect(self._on_roi_param_changed)
        rp.addRow(self._cap_phase, self._spin_phase)

        self._roi_circle_hint = QLabel("Drag / resize the circle on the image to place the spot.")
        self._roi_circle_hint.setWordWrap(True)
        self._roi_circle_hint.setStyleSheet("color:#777; font-style:italic;")
        rp.addRow(self._roi_circle_hint)

        roi_v.addWidget(self._roi_params)
        self._roi_params.setVisible(False)
        ll.addWidget(g_roi)

        # Center — the ROI centre (beam centre). Same UX as the FTH Beamstop
        # Center: click-to-set + numeric X/Y with -5/-1/+1/+5 nudge buttons.
        g_img_center = QGroupBox("Center")
        fl_o = QFormLayout(g_img_center)
        fl_o.setContentsMargins(6, 4, 6, 4)
        self._btn_pick_origin = QPushButton("Click to Set Center")
        self._btn_pick_origin.setCheckable(True)
        self._btn_pick_origin.setAutoDefault(False)
        self._btn_pick_origin.setToolTip("Toggle on, then click the image (repeatedly) to set the centre.")
        self._btn_pick_origin.toggled.connect(self._on_origin_pick_toggled)
        btn_apply_center = QPushButton("Apply Center")
        btn_apply_center.setAutoDefault(False)
        btn_apply_center.setToolTip("Fix the centre and leave click-to-set mode.")
        btn_apply_center.clicked.connect(self._on_apply_center)
        row_center_btns = QHBoxLayout()
        row_center_btns.addWidget(self._btn_pick_origin)
        row_center_btns.addWidget(btn_apply_center)
        fl_o.addRow(row_center_btns)
        self._spin_center_x = QDoubleSpinBox()
        self._spin_center_x.setRange(0, 99999)
        self._spin_center_x.setDecimals(1)
        self._spin_center_x.setSuffix(" px")
        self._spin_center_x.valueChanged.connect(self._on_center_spin_changed)
        self._spin_center_y = QDoubleSpinBox()
        self._spin_center_y.setRange(0, 99999)
        self._spin_center_y.setDecimals(1)
        self._spin_center_y.setSuffix(" px")
        self._spin_center_y.valueChanged.connect(self._on_center_spin_changed)
        self._add_pm_row(fl_o, "X (col):", self._spin_center_x)
        self._add_pm_row(fl_o, "Y (row):", self._spin_center_y)
        btn_reset_orig = QPushButton("Reset center")
        btn_reset_orig.setAutoDefault(False)
        btn_reset_orig.clicked.connect(self._on_reset_origin)
        fl_o.addRow(btn_reset_orig)
        # Placed directly under Datasets; pushed below Incidence Correction (inserted at 1 next).
        ll.insertWidget(1, g_img_center)

        # Incidence correction
        g_inc = QGroupBox("Incidence Correction")
        fl_i = QFormLayout(g_inc)
        fl_i.setContentsMargins(6, 4, 6, 4)
        self._spin_inc_x = QDoubleSpinBox()
        self._spin_inc_x.setRange(0.0, 89.9)
        self._spin_inc_x.setDecimals(2)
        self._spin_inc_x.setSuffix("°")
        fl_i.addRow("X angle:", self._spin_inc_x)
        self._spin_inc_y = QDoubleSpinBox()
        self._spin_inc_y.setRange(0.0, 89.9)
        self._spin_inc_y.setDecimals(2)
        self._spin_inc_y.setSuffix("°")
        fl_i.addRow("Y angle:", self._spin_inc_y)
        row_ib = QHBoxLayout()
        btn_apply_inc = QPushButton("Apply")
        btn_apply_inc.setAutoDefault(False)
        btn_apply_inc.clicked.connect(self._on_apply_incidence)
        row_ib.addWidget(btn_apply_inc)
        btn_reset_inc = QPushButton("Reset")
        btn_reset_inc.setAutoDefault(False)
        btn_reset_inc.clicked.connect(self._on_reset_incidence)
        row_ib.addWidget(btn_reset_inc)
        fl_i.addRow("", row_ib)
        # Insert above Center (which is at index 1), directly under Datasets.
        ll.insertWidget(1, g_inc)

        # Beam Stop Mask — masked pixels (beam stop) are excluded from analysis.
        # Applied on the incidence-corrected image.
        g_mask = QGroupBox("Beam Stop Mask")
        ml = QVBoxLayout(g_mask)
        ml.setContentsMargins(6, 4, 6, 4)
        ml.setSpacing(4)
        mask_add_row = QHBoxLayout()
        for label, kind in (("+ Circle", "circle"), ("+ Square", "square")):
            b = QPushButton(label)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, k=kind: self._add_mask_shape(k))
            mask_add_row.addWidget(b)
        ml.addLayout(mask_add_row)
        brush_row = QHBoxLayout()
        self._btn_brush = QPushButton("Brush (paint)")
        self._btn_brush.setCheckable(True)
        self._btn_brush.setAutoDefault(False)
        self._btn_brush.setToolTip("Toggle on, then drag on the image to paint masked pixels.")
        self._btn_brush.toggled.connect(self._on_brush_toggled)
        brush_row.addWidget(self._btn_brush)
        self._spin_brush = QSpinBox()
        self._spin_brush.setRange(1, 200)
        self._spin_brush.setValue(self._brush_size)
        self._spin_brush.setSuffix(" px")
        self._spin_brush.setToolTip("Brush radius.")
        self._spin_brush.valueChanged.connect(lambda v: setattr(self, "_brush_size", int(v)))
        brush_row.addWidget(self._spin_brush)
        ml.addLayout(brush_row)
        mask_btn_row = QHBoxLayout()
        btn_clear_mask = QPushButton("Clear mask")
        btn_clear_mask.setAutoDefault(False)
        btn_clear_mask.clicked.connect(self._clear_beamstop)
        mask_btn_row.addWidget(btn_clear_mask)
        btn_apply_mask = QPushButton("Apply mask")
        btn_apply_mask.setAutoDefault(False)
        btn_apply_mask.setToolTip("Merge the shapes into the committed red mask.")
        btn_apply_mask.clicked.connect(self._on_apply_mask)
        mask_btn_row.addWidget(btn_apply_mask)
        ml.addLayout(mask_btn_row)
        # Insert directly below Incidence Correction.
        ll.insertWidget(2, g_mask)

        self._status = QLabel("Ready")
        self._status.setStyleSheet("color:#444;")
        ll.addWidget(self._status)
        ll.addStretch(1)
        scroll.setWidget(left)
        h_splitter.addWidget(scroll)

        # ── Right: pattern (top) + a single profile plot (bottom) ── #
        right_v = QSplitter(Qt.Orientation.Vertical)
        self._img = ImageView2DEnhanced(parent=self)
        self._img.graphics_layout.scene().sigMouseClicked.connect(self._on_scene_clicked_origin)
        self._img.graphics_layout.scene().sigMouseMoved.connect(self._on_brush_move)
        right_v.addWidget(self._img)

        self._plot_profile = self._make_plot("Radius (pixels)", "Intensity", "Profile")
        right_v.addWidget(self._plot_profile)
        right_v.setSizes([620, 280])
        right_v.setStretchFactor(0, 3)
        right_v.setStretchFactor(1, 1)

        h_splitter.addWidget(right_v)
        h_splitter.setSizes([320, 1080])
        return tab

    @staticmethod
    def _add_pm_row(form, label, spin) -> None:
        """Add -5/-1/+1/+5 nudge buttons + the spinbox on one row (FTH-style)."""
        row = QHBoxLayout()
        for delta, text in [(-5, "-5"), (-1, "-1"), (+1, "+1"), (+5, "+5")]:
            b = QPushButton(text)
            b.setFixedWidth(30)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, d=delta, s=spin: s.setValue(s.value() + d))
            row.addWidget(b)
        row.addWidget(spin)
        form.addRow(label, row)

    @staticmethod
    def _make_plot(x_label: str, y_label: str, title: str):
        p = pg.PlotWidget()
        p.setBackground("k")
        p.showGrid(x=True, y=True, alpha=0.3)
        for axis in ("left", "bottom"):
            p.getAxis(axis).setPen(pg.mkPen("w"))
            p.getAxis(axis).setTextPen(pg.mkPen("w"))
        p.setLabel("bottom", x_label)
        p.setLabel("left", y_label)
        p.setTitle(title, color="w")
        p.setMinimumHeight(150)
        return p

    # ── Curve Fitting tab ─────────────────────────────────────────────── #

    def _build_curve_fit_tab(self, parent: QWidget) -> None:
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(4, 4, 4, 4)
        h_split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(h_split)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(400)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)

        g_profile = QGroupBox("Profile to fit")
        fl_p = QFormLayout(g_profile)
        self._combo_fit_profile = QComboBox()
        self._combo_fit_profile.addItem("I(r) — Radial profile", "radial")
        self._combo_fit_profile.addItem("I(θ) — Azimuthal profile", "angular")
        self._combo_fit_profile.currentIndexChanged.connect(self._on_fit_profile_changed)
        fl_p.addRow("Profile:", self._combo_fit_profile)
        self._lbl_fit_info = QLabel("Frame: —")
        self._lbl_fit_info.setStyleSheet("color:#555; font-style:italic;")
        fl_p.addRow(self._lbl_fit_info)
        ll.addWidget(g_profile)

        # ── 2) Background fit (radial / time only; hidden for angular) ── #
        self._g_bg = QGroupBox("Background Fit")
        fl_bg = QFormLayout(self._g_bg)
        self._chk_bg = QCheckBox("Subtract background")
        self._chk_bg.toggled.connect(self._on_bg_toggled)
        fl_bg.addRow(self._chk_bg)
        self._combo_bg_degree = QComboBox()
        self._combo_bg_degree.addItem("Linear (deg 1)", 1)
        self._combo_bg_degree.addItem("Quadratic (deg 2)", 2)
        self._combo_bg_degree.addItem("Cubic (deg 3)", 3)
        self._combo_bg_degree.addItem("Quartic (deg 4)", 4)
        self._combo_bg_degree.currentIndexChanged.connect(lambda _=0: self._redraw_fit_curves())
        fl_bg.addRow("Model:", self._combo_bg_degree)
        bg_hint = QLabel("Range: drag the 4 green lines (two baseline regions, before/after the peak).")
        bg_hint.setWordWrap(True)
        bg_hint.setStyleSheet("color:#777; font-style:italic;")
        fl_bg.addRow(bg_hint)
        btn_fit_bg = QPushButton("Fit Background")
        btn_fit_bg.setAutoDefault(False)
        btn_fit_bg.clicked.connect(self._on_fit_background)
        fl_bg.addRow(btn_fit_bg)
        ll.addWidget(self._g_bg)

        # ── 3) Peak fit ── #
        g_peak = QGroupBox("Peak Fit")
        pk = QVBoxLayout(g_peak)
        pk.setContentsMargins(6, 4, 6, 4)
        pk.setSpacing(4)
        form_pm = QFormLayout()
        self._combo_fit_model = QComboBox()
        self._combo_fit_model.currentIndexChanged.connect(self._on_fit_model_changed)
        form_pm.addRow("Model:", self._combo_fit_model)
        pk.addLayout(form_pm)
        peak_hint = QLabel("Range: drag the 2 yellow lines.")
        peak_hint.setWordWrap(True)
        peak_hint.setStyleSheet("color:#777; font-style:italic;")
        pk.addWidget(peak_hint)
        self._g_fit_params = QGroupBox("Initial parameters")
        self._fl_fit_params = QFormLayout(self._g_fit_params)
        btn_auto = QPushButton("Auto-guess")
        btn_auto.setAutoDefault(False)
        btn_auto.clicked.connect(self._on_auto_guess)
        self._fl_fit_params.addRow("", btn_auto)
        pk.addWidget(self._g_fit_params)
        btn_fit = QPushButton("Run Peak Fit")
        btn_fit.setAutoDefault(False)
        btn_fit.clicked.connect(self._on_run_fit)
        pk.addWidget(btn_fit)
        ll.addWidget(g_peak)

        g_res = QGroupBox("Results")
        rl = QVBoxLayout(g_res)
        self._fit_results = QTextEdit()
        self._fit_results.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        self._fit_results.setFont(mono)
        self._fit_results.setMinimumHeight(120)
        self._fit_results.setMaximumHeight(260)
        rl.addWidget(self._fit_results)
        btn_copy = QPushButton("Copy results")
        btn_copy.setAutoDefault(False)
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self._fit_results.toPlainText())
        )
        rl.addWidget(btn_copy)
        ll.addWidget(g_res)

        ll.addStretch(1)
        scroll.setWidget(left)
        h_split.addWidget(scroll)

        # Right: one plot normally; a second appears below for the subtracted signal.
        self._fit_split = QSplitter(Qt.Orientation.Vertical)
        self._fit_top = self._make_plot("Radius (pixels)", "Intensity", "")
        self._fit_top.setMinimumHeight(200)
        self._fit_split.addWidget(self._fit_top)
        self._fit_bottom = self._make_plot("Radius (pixels)", "Background-subtracted", "")
        self._fit_bottom.setMinimumHeight(180)
        self._fit_split.addWidget(self._fit_bottom)
        self._fit_bottom.setVisible(False)
        h_split.addWidget(self._fit_split)
        h_split.setSizes([340, 960])

        # Draggable vertical lines: 4 background baseline bounds + 2 fit-range bounds.
        self._bg_lines = [pg.InfiniteLine(angle=90, movable=True,
                          pen=pg.mkPen((0, 200, 80), width=1.2, style=Qt.PenStyle.DashLine))
                          for _ in range(4)]
        self._range_lines = [pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("y", width=1.2))
                             for _ in range(2)]
        for ln in self._bg_lines + self._range_lines:
            ln.sigPositionChanged.connect(self._on_fit_line_moved)
        self._bg_xs: list[float] | None = None
        self._range_xs: list[float] | None = None
        self._fit_items: list = []

        self._fit_line_timer = QTimer(self)
        self._fit_line_timer.setSingleShot(True)
        self._fit_line_timer.setInterval(60)
        self._fit_line_timer.timeout.connect(self._redraw_fit_curves)

        self._on_fit_profile_changed()

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._do_deferred_refresh()
        elif idx == 2:
            self._b3_refresh()
        elif idx == 3:
            self._b4_refresh()

    # ================================================================== #
    # Batch tab (page 3): per-frame fit over the whole stack               #
    # ================================================================== #

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Upper: 3 columns — pattern / profile / background-subtracted.
        self._b3_cols = QSplitter(Qt.Orientation.Horizontal)
        self._b3_img_plot = pg.PlotWidget()
        self._b3_img_plot.setBackground("k")
        self._b3_img_plot.setAspectLocked(True)
        self._b3_img_plot.getViewBox().invertY(self._img.view_box.yInverted())
        self._b3_img_plot.setTitle("Pattern", color="w")
        self._b3_img_item = pg.ImageItem()
        self._b3_img_plot.addItem(self._b3_img_item)
        self._b3_cols.addWidget(self._b3_img_plot)
        self._b3_profile_plot = self._make_plot("Radius (pixels)", "Intensity", "Profile")
        self._b3_cols.addWidget(self._b3_profile_plot)
        self._b3_sub_plot = self._make_plot("Radius (pixels)", "Intensity", "Background-subtracted + peak")
        self._b3_cols.addWidget(self._b3_sub_plot)
        self._b3_cols.setSizes([400, 400, 400])
        v.addWidget(self._b3_cols, stretch=3)

        # Frame slider.
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Frame:"))
        self._b3_slider = QSlider(Qt.Orientation.Horizontal)
        self._b3_slider.setRange(0, 0)
        self._b3_spin = QSpinBox()
        self._b3_spin.setRange(0, 0)
        self._b3_spin.setFixedWidth(64)
        self._b3_lbl = QLabel("— / —")
        self._b3_lbl.setStyleSheet("color:#555;")
        fr.addWidget(self._b3_slider, 1)
        fr.addWidget(self._b3_spin)
        fr.addWidget(self._b3_lbl)
        self._b3_slider.valueChanged.connect(self._b3_on_frame_changed)
        self._b3_spin.valueChanged.connect(self._b3_slider.setValue)
        v.addLayout(fr)

        # Bottom: batch button (left) + per-frame Background / Peak fit settings.
        bottom = QHBoxLayout()
        left = QVBoxLayout()
        self._b3_batch_btn = QPushButton("Batch all frames")
        self._b3_batch_btn.setAutoDefault(False)
        self._b3_batch_btn.clicked.connect(self._on_batch_run)
        left.addWidget(self._b3_batch_btn)
        self._b3_progress = QProgressBar()
        self._b3_progress.setRange(0, 100)
        self._b3_progress.setValue(0)
        left.addWidget(self._b3_progress)
        self._b3_status = QLabel("Load a stack and select a ROI on the Image && Region tab.")
        self._b3_status.setWordWrap(True)
        self._b3_status.setStyleSheet("color:#888;")
        left.addWidget(self._b3_status)
        left.addStretch(1)
        bottom.addLayout(left, 1)

        self._g_b3_bg = QGroupBox("Background Fit")
        fbg = QFormLayout(self._g_b3_bg)
        self._b3_bg_degree = QComboBox()
        for lbl, d in (("Linear (1)", 1), ("Quadratic (2)", 2), ("Cubic (3)", 3), ("Quartic (4)", 4)):
            self._b3_bg_degree.addItem(lbl, d)
        self._b3_bg_degree.currentIndexChanged.connect(lambda _=0: self._b3_redraw())
        fbg.addRow("Model:", self._b3_bg_degree)
        hb = QLabel("Range: 4 green lines on the Profile column.")
        hb.setWordWrap(True)
        hb.setStyleSheet("color:#777; font-style:italic;")
        fbg.addRow(hb)
        bottom.addWidget(self._g_b3_bg)

        g_pk = QGroupBox("Peak Fit")
        fpk = QFormLayout(g_pk)
        self._b3_peak_model = QComboBox()
        self._b3_peak_model.currentIndexChanged.connect(lambda _=0: self._b3_redraw())
        fpk.addRow("Model:", self._b3_peak_model)
        hp = QLabel("Range: 2 yellow lines (fitted frame-by-frame on Batch).")
        hp.setWordWrap(True)
        hp.setStyleSheet("color:#777; font-style:italic;")
        fpk.addRow(hp)
        bottom.addWidget(g_pk)
        v.addLayout(bottom, stretch=1)

        # Draggable lines for page 3.
        self._b3_bg_lines = [pg.InfiniteLine(angle=90, movable=True,
                             pen=pg.mkPen((0, 200, 80), width=1.2, style=Qt.PenStyle.DashLine))
                             for _ in range(4)]
        self._b3_range_lines = [pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("y", width=1.2))
                                for _ in range(2)]
        for ln in self._b3_bg_lines + self._b3_range_lines:
            ln.sigPositionChanged.connect(self._b3_on_line_moved)
        return tab

    def _b3_profile_mode(self) -> str:
        # Inherit the profile type (radial / azimuthal) from the Curve Fitting page.
        return self._fit_profile_mode()

    def _b3_models(self) -> dict:
        return cf.SIN_MODELS if self._b3_profile_mode() == "angular" else cf.PEAK_MODELS

    def _b3_rebuild_models(self) -> None:
        self._b3_peak_model.blockSignals(True)
        self._b3_peak_model.clear()
        for name in self._b3_models():
            self._b3_peak_model.addItem(name)
        self._b3_peak_model.blockSignals(False)

    def _b3_add(self, plot, item) -> None:
        plot.addItem(item)
        self._b3_fit_items.append((plot, item))

    def _b3_clear_items(self) -> None:
        for plot, item in self._b3_fit_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._b3_fit_items = []

    def _b3_apply_line_positions(self) -> None:
        for lines, xs in ((self._b3_bg_lines, self._b3_bg_xs), (self._b3_range_lines, self._b3_range_xs)):
            if not xs:
                continue
            for ln, xv in zip(lines, xs):
                ln.blockSignals(True)
                ln.setValue(xv)
                ln.blockSignals(False)

    def _b3_reset_lines(self) -> None:
        x = self._b3_profile_x()
        if x is None or len(x) == 0:
            return
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        span = max(xmax - xmin, 1.0)
        self._b3_bg_xs = [xmin, xmin + 0.15 * span, xmax - 0.15 * span, xmax]
        self._b3_range_xs = [xmin, xmax]
        self._b3_apply_line_positions()

    def _b3_inherit_from_page2(self) -> None:
        """Adopt the Curve Fitting page's fit settings as the page-3 starting point."""
        # Background polynomial degree.
        j = self._b3_bg_degree.findData(int(self._combo_bg_degree.currentData()))
        if j >= 0:
            self._b3_bg_degree.blockSignals(True)
            self._b3_bg_degree.setCurrentIndex(j)
            self._b3_bg_degree.blockSignals(False)
        # Peak / sin model (combo is rebuilt for the current mode just before this).
        k = self._b3_peak_model.findText(self._combo_fit_model.currentText())
        if k >= 0:
            self._b3_peak_model.blockSignals(True)
            self._b3_peak_model.setCurrentIndex(k)
            self._b3_peak_model.blockSignals(False)
        # Background-region and fit-range line positions.
        if self._bg_xs is not None:
            self._b3_bg_xs = list(self._bg_xs)
        if self._range_xs is not None:
            self._b3_range_xs = list(self._range_xs)

    def _b3_arrange_lines(self) -> None:
        for ln in self._b3_bg_lines + self._b3_range_lines:
            for p in (self._b3_profile_plot, self._b3_sub_plot):
                try:
                    p.removeItem(ln)
                except Exception:
                    pass
        if self._b3_profile_mode() == "angular":
            for ln in self._b3_range_lines:
                self._b3_profile_plot.addItem(ln)
        else:
            for ln in self._b3_bg_lines:
                self._b3_profile_plot.addItem(ln)
            for ln in self._b3_range_lines:
                self._b3_sub_plot.addItem(ln)

    def _b3_on_line_moved(self, *_a) -> None:
        if self._b3_bg_xs is not None:
            self._b3_bg_xs = [float(ln.value()) for ln in self._b3_bg_lines]
        if self._b3_range_xs is not None:
            self._b3_range_xs = [float(ln.value()) for ln in self._b3_range_lines]
        self._b3_redraw()

    def _b3_profile_x(self):
        """The x array (r or theta) of the current frame's selected-ROI profile."""
        roi = self._selected_roi()
        if self._data is None or roi is None:
            return None
        rx, ry, ax, ay = self._roi_profiles(roi, self._analysis_frame(self._b3_slider.value()))
        return ax if self._b3_profile_mode() == "angular" else rx

    def _b3_refresh(self) -> None:
        roi = self._selected_roi()
        ok = self._data is not None and roi is not None and self._n_frames >= 1
        self._g_b3_bg.setEnabled(ok)
        self._b3_batch_btn.setEnabled(ok)
        if not ok:
            self._b3_status.setText("Load a stack and select a ROI on the Image && Region tab.")
            self._b3_clear_items()
            self._b3_img_item.clear()
            return
        n = self._n_frames
        for w in (self._b3_slider, self._b3_spin):
            w.blockSignals(True)
            w.setRange(0, max(0, n - 1))
            w.blockSignals(False)
        is_az = self._b3_profile_mode() == "angular"
        self._b3_sub_plot.setVisible(not is_az)
        self._g_b3_bg.setVisible(not is_az)
        x_label = "Angle (°)" if is_az else "Radius (pixels)"
        self._b3_profile_plot.setLabel("bottom", x_label)
        self._b3_sub_plot.setLabel("bottom", x_label)
        self._b3_rebuild_models()
        # Inherit correction/beamstop/ROI are already shared via self._data /
        # _analysis_frame / _selected_roi; here we also adopt page 2's fit settings.
        self._b3_reset_lines()              # sensible defaults for the current x extent
        self._b3_inherit_from_page2()       # override with the Curve Fitting page's settings
        self._b3_apply_line_positions()
        self._b3_arrange_lines()
        self._b3_status.setText(f"{n} frames. Inherited fit settings from Curve Fitting; "
                                f"adjust per-frame if needed, then Batch all frames.")
        self._b3_show_frame(self._b3_slider.value())

    def _b3_on_frame_changed(self, idx: int) -> None:
        self._b3_spin.blockSignals(True)
        self._b3_spin.setValue(idx)
        self._b3_spin.blockSignals(False)
        self._b3_show_frame(idx)

    def _b3_show_frame(self, idx: int) -> None:
        roi = self._selected_roi()
        if self._data is None or roi is None:
            return
        idx = max(0, min(idx, self._n_frames - 1))
        self._b3_lbl.setText(f"{idx + 1} / {self._n_frames}")
        frame = self._analysis_frame(idx)
        finite = np.isfinite(frame)
        fill = float(np.nanmin(frame)) if finite.any() else 0.0
        self._b3_img_item.setImage(np.where(finite, frame, fill), autoLevels=True)
        self._b3_draw_overlay(roi)
        self._b3_redraw()

    def _b3_draw_overlay(self, roi: dict) -> None:
        for it in self._b3_overlay_items:
            try:
                self._b3_img_plot.removeItem(it)
            except Exception:
                pass
        self._b3_overlay_items = []
        cx, cy = self._center_pixel()
        marker = pg.ScatterPlotItem([cx], [cy], symbol="+", size=16,
                                    pen=pg.mkPen("r", width=2), brush=pg.mkBrush(None))
        self._b3_img_plot.addItem(marker)
        self._b3_overlay_items.append(marker)
        pen = pg.mkPen((0, 255, 255), width=1.5)

        def add(vx, vy):
            c = pg.PlotCurveItem(vx, vy, pen=pen)
            self._b3_img_plot.addItem(c)
            self._b3_overlay_items.append(c)

        if roi["type"] == "ring":
            for r in (roi["r_inner"], roi["r_outer"]):
                if r > 0:
                    add(*self._arc_pixel_xy(cx, cy, r))
        elif roi["type"] == "sector":
            a0, a1 = float(roi["a_min"]) % 360.0, float(roi["a_max"]) % 360.0
            if a1 <= a0:
                a1 += 360.0
            for r in (roi["r_inner"], roi["r_outer"]):
                if r > 0:
                    add(*self._arc_pixel_xy(cx, cy, r, a0, a1))
            for ang in (a0, a1):
                th = np.deg2rad(ang)
                add(np.array([cx + roi["r_inner"] * np.cos(th), cx + roi["r_outer"] * np.cos(th)]),
                    np.array([cy + roi["r_inner"] * np.sin(th), cy + roi["r_outer"] * np.sin(th)]))
        else:  # circle / ellipse — its own centre
            rx, ry = self._circle_semi_axes(roi)
            th = np.linspace(0, 2 * np.pi, 181)
            add(roi["cx"] + rx * np.cos(th), roi["cy"] + ry * np.sin(th))

    def _b3_frame_profile(self, idx: int):
        """(x, y) of the selected ROI's profile for frame idx (azimuthal or radial)."""
        roi = self._selected_roi()
        rx, ry, ax, ay = self._roi_profiles(roi, self._analysis_frame(idx))
        if self._b3_profile_mode() == "angular":
            return np.asarray(ax, dtype=np.float64), np.asarray(ay, dtype=np.float64)
        return np.asarray(rx, dtype=np.float64), np.asarray(ry, dtype=np.float64)

    def _b3_redraw(self) -> None:
        self._b3_clear_items()
        roi = self._selected_roi()
        if self._data is None or roi is None:
            return
        if self._b3_bg_xs is None:
            self._b3_reset_lines()
        x, y = self._b3_frame_profile(self._b3_slider.value())
        if len(x) == 0:
            return
        is_az = self._b3_profile_mode() == "angular"
        pen_color = (255, 160, 0) if is_az else "c"
        valid = np.isfinite(y)
        self._b3_add(self._b3_profile_plot,
                     pg.PlotCurveItem(x[valid], y[valid], pen=pg.mkPen(pen_color, width=1.5)))

        if is_az:
            fit_x, fit_y, fit_plot = x, y, self._b3_profile_plot
        else:
            x1, x2, x3, x4 = sorted(self._b3_bg_xs)
            deg = int(self._b3_bg_degree.currentData())
            y_corr, bg, _ = cf.compute_poly_background(x, y, x1, x2, x3, x4, deg)
            self._b3_add(self._b3_profile_plot,
                         pg.PlotCurveItem(x, bg, pen=pg.mkPen((0, 200, 80), width=1.3, style=Qt.PenStyle.DashLine)))
            vc = np.isfinite(y_corr)
            self._b3_add(self._b3_sub_plot,
                         pg.PlotCurveItem(x[vc], y_corr[vc], pen=pg.mkPen(pen_color, width=1.5)))
            fit_x, fit_y, fit_plot = x, y_corr, self._b3_sub_plot

        reg = self._b3_models()
        model_name = self._b3_peak_model.currentText()
        if model_name in reg and self._b3_range_xs:
            lo, hi = min(self._b3_range_xs), max(self._b3_range_xs)
            p0 = cf.auto_guess(model_name, fit_x, fit_y, lo, hi, models=reg)
            if p0:
                res = cf.run_fit(reg[model_name]["func"], fit_x, fit_y, p0, lo, hi)
                if res["ok"]:
                    xf = res["x_fit"]
                    xd = np.linspace(float(xf[0]), float(xf[-1]), max(400, len(xf) * 4))
                    self._b3_add(fit_plot, pg.PlotCurveItem(
                        xd, reg[model_name]["func"](xd, *res["popt"]), pen=pg.mkPen("r", width=2)))
                    fit_plot.setTitle(f"Frame fit — R² = {res['r2']:.4f}", color="w")

    def _on_batch_run(self) -> None:
        roi = self._selected_roi()
        if self._data is None or roi is None:
            self._b3_status.setText("Load a stack and select a ROI first.")
            return
        is_az = self._b3_profile_mode() == "angular"
        reg = self._b3_models()
        model_name = self._b3_peak_model.currentText()
        if model_name not in reg:
            return
        func = reg[model_name]["func"]
        pnames = [n for n, _ in reg[model_name]["params"]]
        if self._b3_range_xs is None:
            self._b3_reset_lines()
        lo, hi = min(self._b3_range_xs), max(self._b3_range_xs)
        deg = int(self._b3_bg_degree.currentData())
        bgxs = sorted(self._b3_bg_xs) if (not is_az and self._b3_bg_xs) else None

        n = self._n_frames
        self._b3_results = []
        self._b3_progress.setRange(0, n)
        p0 = None
        for i in range(n):
            x, y = self._b3_frame_profile(i)
            row: dict = {"frame": i, "model": model_name}
            if len(x) > 0:
                # Raw integrated intensity between the two peak lines (per frame).
                in_rng = (x >= lo) & (x <= hi) & np.isfinite(y)
                row["I_sum"] = float(np.nansum(y[in_rng])) if in_rng.any() else float("nan")
                yfit = y
                if bgxs is not None:
                    yfit, _, _ = cf.compute_poly_background(x, y, *bgxs, deg)
                guess = p0 or cf.auto_guess(model_name, x, yfit, lo, hi, models=reg)
                if guess:
                    res = cf.run_fit(func, x, yfit, guess, lo, hi)
                    if res["ok"]:
                        for nm, val in zip(pnames, res["popt"]):
                            row[nm] = float(val)
                        for nm, val in zip(pnames, res["perr"]):
                            row[f"{nm}_err"] = float(val)
                        row["R2"] = float(res["r2"])
                        row["RMSE"] = float(res["rmse"])
                        p0 = list(res["popt"])   # chain to the next frame
            self._b3_results.append(row)
            self._b3_progress.setValue(i + 1)
            if i % 5 == 0:
                QApplication.processEvents()
        n_ok = sum(1 for r in self._b3_results if "R2" in r)
        self._b3_status.setText(f"Batch done: {n_ok}/{n} frames fitted. "
                                f"(Parameter-vs-frame trends will show on the Summary page.)")
        self._b4_refresh()

    # ================================================================== #
    # Summary tab (page 4): parameter-vs-frame trends                      #
    # ================================================================== #

    _B4_LABELS = {
        "I_sum": "Intensity sum (raw, in range)", "A": "Amplitude A", "x₀": "Peak center x₀",
        "σ": "Width σ", "C": "Offset C", "γ": "Width γ", "w": "Width w", "η": "Mixing η",
        "φ": "Phase φ", "R2": "R²", "RMSE": "RMSE",
    }

    def _b4_label(self, key: str) -> str:
        return self._B4_LABELS.get(key, key)

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(4, 4, 4, 4)

        # Left: data/parameter selection panel.
        side = QVBoxLayout()
        g = QGroupBox("Curves")
        fg = QFormLayout(g)
        self._b4_combos = []
        for i in range(4):
            cb = QComboBox()
            cb.currentIndexChanged.connect(lambda _=0, k=i: self._b4_draw_one(k))
            self._b4_combos.append(cb)
            fg.addRow(f"Plot {i + 1}:", cb)
        side.addWidget(g)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setAutoDefault(False)
        btn_refresh.clicked.connect(self._b4_refresh)
        side.addWidget(btn_refresh)
        btn_csv = QPushButton("Export CSV…")
        btn_csv.setAutoDefault(False)
        btn_csv.clicked.connect(self._b4_export_csv)
        side.addWidget(btn_csv)
        self._b4_status = QLabel("Run Batch on the Batch page first.")
        self._b4_status.setWordWrap(True)
        self._b4_status.setStyleSheet("color:#888;")
        side.addWidget(self._b4_status)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setMaximumWidth(280)
        root.addWidget(side_w)

        # Right: 2x2 grid of trend plots.
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        self._b4_plots = []
        for i in range(4):
            p = self._make_plot("Frame", "Value", f"Plot {i + 1}")
            self._b4_plots.append(p)
            grid.addWidget(p, i // 2, i % 2)
        root.addWidget(grid_w, 1)
        return tab

    def _b4_param_keys(self) -> list[str]:
        keys: list[str] = []
        for row in self._b3_results:
            for k in row:
                if k in ("frame", "model") or k.endswith("_err"):
                    continue
                if k not in keys:
                    keys.append(k)
        if "I_sum" in keys:                      # show the raw intensity first
            keys.remove("I_sum")
            keys.insert(0, "I_sum")
        return keys

    def _b4_refresh(self) -> None:
        keys = self._b4_param_keys()
        if not keys:
            self._b4_status.setText("No batch results — run Batch on the Batch page first.")
            for p in self._b4_plots:
                p.clear()
            return
        self._b4_status.setText(f"{len(self._b3_results)} frames. Pick a parameter per plot.")
        defaults = (["I_sum", "x₀", "A", "σ"] + keys)   # sensible starting set
        for i, cb in enumerate(self._b4_combos):
            prev = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for k in keys:
                cb.addItem(self._b4_label(k), k)
            want = prev if prev in keys else next((d for d in defaults[i:] if d in keys), keys[0])
            cb.setCurrentIndex(max(0, cb.findData(want)))
            cb.blockSignals(False)
        for i in range(4):
            self._b4_draw_one(i)

    def _b4_draw_one(self, i: int) -> None:
        if i >= len(self._b4_plots):
            return
        p = self._b4_plots[i]
        p.clear()
        key = self._b4_combos[i].currentData()
        if not self._b3_results or key is None:
            return
        frames = np.array([row["frame"] for row in self._b3_results], dtype=np.float64)
        ys = np.array([float(row[key]) if row.get(key) is not None else np.nan
                       for row in self._b3_results], dtype=np.float64)
        p.plot(frames, ys, pen=pg.mkPen("c", width=1.5), symbol="o", symbolSize=5,
               symbolBrush=pg.mkBrush((100, 200, 255)), connect="finite")
        p.setLabel("left", self._b4_label(key))
        p.setTitle(self._b4_label(key), color="w")

    def _b4_export_csv(self) -> None:
        if not self._b3_results:
            self._b4_status.setText("Nothing to export — run Batch first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export batch results", "xrms_batch.csv",
                                              "CSV files (*.csv)")
        if not path:
            return
        import csv
        keys: list[str] = []
        for row in self._b3_results:
            for k in row:
                if k not in keys:
                    keys.append(k)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for row in self._b3_results:
                    writer.writerow([row.get(k, "") for k in keys])
        except Exception as exc:
            self._b4_status.setText(f"Export failed: {exc}")
            return
        self._b4_status.setText(f"Exported {len(self._b3_results)} rows.")

    # ================================================================== #
    # Dataset combo population + loading                                   #
    # ================================================================== #

    @staticmethod
    def _key_label(full_key: str) -> str:
        try:
            fp, ds = full_key.rsplit("::", 1)
            ds = ds.strip()
            name = pathlib.Path(fp).name
            if not ds or ds in ("/data", "data"):
                return name
            parts = ds.split("/")
            short = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]
            return f"{name}  ::  {short}"
        except Exception:
            return full_key

    # ── main-window contract ──────────────────────────────────────────────── #

    def refresh_dataset_keys(self, keys_2d: list[str]) -> None:
        """Re-sync the available dataset list (main-window contract).

        Slots are drag-drop targets, so this only records the keys for reference;
        the user drags datasets/files onto the slot combos directly.
        """
        self._dataset_full_keys_2d = list(keys_2d)

    def set_opened_files(self, opened_files) -> None:
        """Sync the latest opened-files tuple from the main window."""
        self._opened_files = tuple(opened_files)

    def load_dataset_full_key(self, full_key: str, auto_load: bool = True, slot: str = "CL") -> bool:
        """Add a dataset (``file::path``) to a slot (default CL) as if dropped on it."""
        if not full_key or "::" not in full_key:
            return False
        combo = self._slot_combo.get(slot, self._slot_combo["CL"])
        ents = combo.entries()
        if full_key not in ents:
            combo.set_entries(ents + [full_key])  # sigEntriesChanged → _load_slot
        return True

    # ── per-slot loading ──────────────────────────────────────────────────── #

    @staticmethod
    def _read_nd_array(fp: str, ds: str, allow_1d: bool = False) -> np.ndarray:
        """Read an array at ``ds`` from an HDF5 file (or a whole text file)."""
        if is_hdf5_file(fp):
            import h5py
            with h5py.File(fp, "r") as h5:
                if not ds or ds not in h5 or not hasattr(h5[ds], "shape"):
                    raise KeyError(f"Dataset not found: {ds!r} in {pathlib.Path(fp).name}")
                arr = np.asarray(h5[ds][()])
        else:
            arr = np.genfromtxt(fp, comments="#", encoding=None, invalid_raise=False)
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if not allow_1d and arr.ndim < 2:
            raise ValueError(f"Expected a 2D/3D image but got shape {arr.shape}.")
        return arr

    def _load_slot(self, slot: str) -> None:
        """Read every entry of an image slot at its address/axis, then recombine.

        Builds an (n, H, W) stack: each entry's 2-D array is one frame, each 3-D
        array is split into frames along ``axis``.
        """
        combo = self._slot_combo[slot]
        entries = combo.entries()
        addr = self._slot_addr[slot].text().strip()
        axis = int(self._slot_axis[slot].value())
        lbl = self._slot_shape[slot]

        if not entries:
            self._slots[slot] = None
            lbl.setText("—")
            self._recompute_stack()
            return

        # A top-level (file) entry has no "::" and needs the address to name its
        # dataset; wait quietly for the user to type it rather than erroring.
        if not addr and any("::" not in ent for ent in entries):
            self._slots[slot] = None
            lbl.setText("set address")
            self._recompute_stack()
            return

        parts: list[np.ndarray] = []
        try:
            for ent in entries:
                if "::" in ent:
                    fp, path = ent.rsplit("::", 1)
                else:
                    fp, path = ent, addr
                parts.append(self._read_nd_array(fp.strip(), path.strip()))
        except Exception as exc:
            logging.exception("XRMS: load %s failed", slot)
            self._slots[slot] = None
            lbl.setText("load error")
            self._set_status(f"{slot}: {exc}", error=True)
            return

        frames: list[np.ndarray] = []
        for a in parts:
            a = np.asarray(a, np.float32)
            if a.ndim >= 3:
                a = np.moveaxis(a, min(axis, a.ndim - 1), 0)
                a = a.reshape(a.shape[0], a.shape[1], -1)
                frames.extend(a)
            elif a.ndim == 2:
                frames.append(a)
        if not frames:
            self._slots[slot] = None
            lbl.setText("no 2D data")
            self._set_status(f"{slot}: no 2-D frames found.", error=True)
            return
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            self._slots[slot] = None
            lbl.setText("shape mismatch")
            self._set_status(f"{slot}: frames have differing shapes {shapes}.", error=True)
            return
        stack = np.stack(frames)
        self._slots[slot] = stack
        lbl.setText(" × ".join(str(s) for s in stack.shape))
        self._recompute_stack()

    def clear_slot(self, slot: str) -> None:
        """Forget a loaded slot and recombine."""
        if slot in self._slot_combo:
            self._slot_combo[slot].clear_entries()  # sigEntriesChanged → _load_slot

    def _on_norm_toggled(self, checked: bool) -> None:
        self._norm_enabled = bool(checked)
        self._recompute_stack()

    # ── delay-aligned / normalised combine ────────────────────────────────── #

    _OP_LABELS = {
        "cl": "CL", "cr": "CR",
        "diff": "Difference CL−CR", "sum": "Sum CL+CR", "asym": "Asymmetry (CL−CR)/(CL+CR)",
    }

    def _slot_bg(self, name: str) -> np.ndarray | None:
        """Background image for a slot: the frame, or the mean of a multi-frame stack."""
        b = self._slots[name]
        if b is None:
            return None
        return b.mean(axis=0) if b.ndim == 3 else np.asarray(b)

    def _read_stack_aux(self, stack_slot: str, address: str, n: int) -> np.ndarray | None:
        """Read a per-frame 1-D value (delay/norm) for a stack from its own files.

        Reads ``address`` from each entry's file (mirroring the image entries) and
        concatenates; returns an (n,) array, or None when unavailable/mismatched.
        """
        if not address:
            return None
        entries = self._slot_combo[stack_slot].entries()
        if not entries:
            return None
        vals: list[np.ndarray] = []
        for ent in entries:
            fp = ent.rsplit("::", 1)[0] if "::" in ent else ent
            arr = self._read_nd_array(fp.strip(), address.strip(), allow_1d=True)
            vals.append(np.asarray(arr, np.float64).ravel())
        out = np.concatenate(vals) if vals else None
        if out is None or out.size != n:
            self._set_status(
                f"{stack_slot}: '{address}' gave {0 if out is None else out.size} "
                f"values for {n} frames; using frame index.", error=True)
            return None
        return out

    def _process(self, name: str, bg_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(CL or CR) − background, optionally ÷ norm. Returns (stack, frame_delays).

        Delay and norm are read per-frame from this stack's own data files, so CL
        and CR each carry their own delays for value-based alignment.
        """
        stack = self._slots[name]
        if stack is None:
            return None, None
        st = stack.astype(np.float32)
        bg = self._slot_bg(bg_name)
        if bg is not None:
            if bg.shape != st.shape[1:]:
                raise ValueError(f"{bg_name} shape {bg.shape} ≠ {name} frame {st.shape[1:]}.")
            st = st - bg[None]
        n = st.shape[0]
        delay = self._read_stack_aux(name, self._delay_addr.text().strip(), n)
        d = delay if delay is not None else np.arange(n, dtype=np.float64)
        if self._norm_enabled:
            norm = self._read_stack_aux(name, self._norm_addr.text().strip(), n)
            if norm is not None:
                nv = np.where(norm == 0, np.nan, norm)
                st = st / nv[:, None, None]
        return st, d

    @staticmethod
    def _match_by_delay(cl_d: np.ndarray, cr_d: np.ndarray) -> list[tuple[int, int, float]]:
        """Pair CL/CR frame indices sharing a delay value, ordered by delay."""
        cr_lut: dict[float, int] = {}
        for j, d in enumerate(cr_d):
            cr_lut.setdefault(float(d), j)
        pairs = [(i, cr_lut[float(d)], float(d)) for i, d in enumerate(cl_d) if float(d) in cr_lut]
        pairs.sort(key=lambda t: t[2])
        return pairs

    def _recompute_stack(self) -> None:
        """Combine the 6 slots into the analyzed stack.

        Per delay: cl = (CL − CL_BG)/Norm, cr = (CR − CR_BG)/Norm. CL and CR are
        paired by matching delay value; the chosen operation produces each frame.
        """
        op = self._combo_operation.currentData()
        try:
            cl_st, cl_d = self._process("CL", "CL_BG")
            cr_st, cr_d = self._process("CR", "CR_BG")
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return

        if op == "cl":
            if cl_st is None:
                self._set_status("Load a CL dataset.", error=True)
                return
            order = np.argsort(cl_d, kind="stable")
            stack, dl = cl_st[order], cl_d[order]
        elif op == "cr":
            if cr_st is None:
                self._set_status("Load a CR dataset.", error=True)
                return
            order = np.argsort(cr_d, kind="stable")
            stack, dl = cr_st[order], cr_d[order]
        else:
            if cl_st is None or cr_st is None:
                self._set_status("Load both CL and CR datasets.", error=True)
                return
            if cl_st.shape[1:] != cr_st.shape[1:]:
                self._set_status("CL and CR frame shapes differ.", error=True)
                return
            pairs = self._match_by_delay(cl_d, cr_d)
            if not pairs:
                self._set_status("No common delay value between CL and CR.", error=True)
                return
            cls = np.stack([cl_st[i] for i, _, _ in pairs])
            crs = np.stack([cr_st[j] for _, j, _ in pairs])
            dl = np.array([d for _, _, d in pairs], dtype=np.float64)
            if op == "diff":
                stack = cls - crs
            elif op == "sum":
                stack = cls + crs
            else:  # asym
                ssum = cls + crs
                with np.errstate(invalid="ignore", divide="ignore"):
                    stack = np.where(np.abs(ssum) > 1e-12, (cls - crs) / ssum, np.nan)

        self._combined_delay = np.asarray(dl, np.float64)
        self._set_new_stack(np.asarray(stack, dtype=np.float32), self._OP_LABELS[op])

    def _set_new_stack(self, combined: np.ndarray, label: str) -> None:
        # Keep the pre-incidence stack so incidence can be re-applied/reset later.
        self._combined = np.asarray(combined, dtype=np.float32)
        stack = self._apply_incidence(self._combined)
        same_shape = self._data is not None and self._data.shape == stack.shape
        self._data = stack
        self._n_frames = int(stack.shape[0])
        h, w = int(stack.shape[1]), int(stack.shape[2])

        max_frame = max(0, self._n_frames - 1)
        for widget in (self._sl_frame, self._spin_frame):
            widget.blockSignals(True)
            widget.setRange(0, max_frame)
            if not same_shape:
                widget.setValue(0)
            widget.setEnabled(self._n_frames > 1)
            widget.blockSignals(False)
        self._g_frame.setVisible(self._n_frames > 1)

        if not same_shape:
            # New geometry: drop existing ROIs + beam-stop mask and re-center.
            self._clear_all_rois()
            self._reset_beamstop_for_new_geometry()
            r_possible = int(np.ceil(np.hypot(w, h)))
            self._sl_ri.setMaximum(r_possible)
            self._sl_ro.setMaximum(r_possible)
            self._spin_center_x.setRange(0, float(w - 1))
            self._spin_center_y.setRange(0, float(h - 1))
            # Default the center to the image center (defined, not yet stretched).
            self._set_display_origin(w / 2.0, h / 2.0, apply=False)

        idx = self._sl_frame.value()
        self._show_frame(idx)
        self._schedule_refresh()
        self._set_status(f"{label}  ({self._n_frames} frames, {h} × {w} px)")

    # ================================================================== #
    # Frame navigation                                                     #
    # ================================================================== #

    def _on_frame_changed(self, index: int) -> None:
        self._spin_frame.blockSignals(True)
        self._spin_frame.setValue(index)
        self._spin_frame.blockSignals(False)
        self._show_frame(index)
        self._schedule_refresh()

    def _show_frame(self, index: int) -> None:
        if self._data is None:
            return
        index = max(0, min(index, self._n_frames - 1))
        self._lbl_frame_info.setText(f"{index + 1} / {self._n_frames}")
        self._img.set_data(self._data[index])
        self._redraw_roi_overlays()

    # ================================================================== #
    # ROI management                                                       #
    # ================================================================== #

    def _add_roi(self, kind: str) -> None:
        """Create a Ring / Sector / Circle ROI and select it."""
        if kind == "circle" and self._data is None:
            QMessageBox.warning(self, "Time Resolved XRMS", "Load a dataset before adding a Circle ROI.")
            return
        self._roi_counter[kind] += 1
        name = f"{kind.capitalize()} {self._roi_counter[kind]}"
        if self._data is not None:
            h, w = int(self._data.shape[1]), int(self._data.shape[2])
        else:
            h = w = 0
        if kind == "circle":
            radius = max(3, int(0.10 * min(h, w))) if min(h, w) else 10
            roi = {"type": "circle", "name": name,
                   "cx": w / 2.0, "cy": h / 2.0, "rx": float(radius), "ry": float(radius),
                   "mode": "radial", "item": None}
            self._rois.append(roi)
            self._create_circle_item(roi)
        else:
            ri = max(1, int(0.12 * min(h, w))) if min(h, w) else 20
            ro = max(ri + 1, int(0.30 * min(h, w))) if min(h, w) else 60
            self._rois.append({
                "type": kind, "name": name,
                "r_inner": ri, "r_outer": ro,
                "a_min": 0, "a_max": (180 if kind == "ring" else 60),
                "mode": "radial", "phase": 0,
            })
        self._roi_combo.addItem(name)
        self._roi_combo.setCurrentIndex(self._roi_combo.count() - 1)  # triggers select

    # ── Circle ROI interactive item ───────────────────────────────────── #

    def _create_circle_item(self, roi: dict) -> None:
        # EllipseROI (resizable into an ellipse) in pixel/data coordinates.
        cx, cy, rx, ry = roi["cx"], roi["cy"], roi["rx"], roi["ry"]
        item = pg.EllipseROI([cx - rx, cy - ry], [2 * rx, 2 * ry],
                             pen=pg.mkPen((90, 160, 200), width=1.5), movable=True)
        item.setZValue(16)
        item.sigRegionChanged.connect(lambda _it, ro=roi: self._on_circle_roi_changed(ro))
        self._img.view_box.addItem(item)
        roi["item"] = item

    def _sync_circle_item(self, roi: dict) -> None:
        """Position the EllipseROI item from the ROI's pixel geometry."""
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
        """User dragged / resized the ellipse: read its geometry back into the ROI."""
        item = roi.get("item")
        if item is None:
            return
        pos, size = item.pos(), item.size()
        rx, ry = float(size.x()) / 2.0, float(size.y()) / 2.0
        roi["cx"], roi["cy"] = float(pos.x()) + rx, float(pos.y()) + ry
        roi["rx"], roi["ry"] = max(1.0, rx), max(1.0, ry)
        if self._selected_roi() is roi:
            mean_r = int(round((roi["rx"] + roi["ry"]) / 2.0))
            self._sl_ro.blockSignals(True)
            self._sl_ro.setValue(mean_r)
            self._sl_ro.blockSignals(False)
            self._lbl_ro.setText(f"{int(round(roi['rx']))}×{int(round(roi['ry']))} px")
        # The item already follows the cursor; defer the expensive recompute.
        self._schedule_refresh()

    def _remove_circle_item(self, roi: dict) -> None:
        item = roi.get("item")
        if item is not None:
            try:
                self._img.view_box.removeItem(item)
            except Exception:
                pass
            roi["item"] = None

    def _clear_all_rois(self) -> None:
        for roi in self._rois:
            self._remove_circle_item(roi)
        self._clear_roi_handles()
        self._rois = []
        self._roi_combo.blockSignals(True)
        self._roi_combo.clear()
        self._roi_combo.blockSignals(False)
        self._roi_params.setVisible(False)
        self._redraw_roi_overlays()

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
        self._sl_ri.setVisible(not is_circle)
        self._lbl_ri.setVisible(not is_circle)
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
        self._sync_fit_profile_to_roi(roi)
        self._rebuild_roi_handles()
        self._refresh_after_geometry()

    def _update_phase_visibility(self, roi: dict | None) -> None:
        """Phase shift applies only to the azimuthal profile of a ring/sector."""
        show = (roi is not None and roi.get("type") in ("ring", "sector")
                and roi.get("mode") == "angular")
        self._cap_phase.setVisible(show)
        self._spin_phase.setVisible(show)

    def _on_roi_param_changed(self, *_a) -> None:
        roi = self._selected_roi()
        if roi is None:
            return
        if roi["type"] == "circle":
            # The slider sets a uniform radius (circularises); drag handles for ellipse.
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
        # Sector spans CCW from a_min to a_max (wrap allowed), so do NOT force
        # a_min < a_max — that would collapse a wedge crossing the +/-180 seam.
        amin, amax = self._sl_amin.value(), self._sl_amax.value()
        roi.update(r_inner=ri, r_outer=ro, a_min=amin, a_max=amax,
                   mode=self._combo_roi_mode.currentData(),
                   phase=int(self._spin_phase.value()))
        self._update_param_labels()
        self._update_phase_visibility(roi)
        self._sync_fit_profile_to_roi(roi)
        self._refresh_after_geometry()

    def _update_param_labels(self) -> None:
        self._lbl_ri.setText(f"{self._sl_ri.value()} px")
        ro = self._sl_ro.value()
        self._lbl_ro.setText(f"{ro} px" if ro > 0 else "0 px (max)")
        self._lbl_amin.setText(f"{self._sl_amin.value()}°")
        self._lbl_amax.setText(f"{self._sl_amax.value()}°")

    def _sync_fit_profile_to_roi(self, roi: dict) -> None:
        """Keep the Curve Fitting profile selector in sync with the ROI's mode."""
        idx = self._combo_fit_profile.findData(roi.get("mode", "radial"))
        if idx >= 0 and idx != self._combo_fit_profile.currentIndex():
            self._combo_fit_profile.blockSignals(True)
            self._combo_fit_profile.setCurrentIndex(idx)
            self._combo_fit_profile.blockSignals(False)
            self._on_fit_profile_changed()   # reconfigure models / bg / plots

    def _refresh_after_geometry(self) -> None:
        """Cheap live update (redraw overlays); defer the expensive recompute."""
        self._redraw_roi_overlays()
        self._position_roi_handles()
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        """Coalesce expensive profile/time recompute to fire once after interaction settles."""
        self._refresh_timer.start()

    def _do_deferred_refresh(self) -> None:
        """The heavy work: recompute the selected ROI's profile (and time only on fit tab)."""
        self._compute_current_profiles()   # -> _update_profile_plots + button/label
        if self._tabs.currentIndex() == 1:
            self._redraw_fit_curves()

    # ── ROI overlays ──────────────────────────────────────────────────── #

    def _center_pixel(self) -> tuple[float, float]:
        """Global center in pixel (col, row); defaults to image center."""
        if self._display_origin is not None:
            return self._display_origin
        if self._data is not None:
            h, w = int(self._data.shape[1]), int(self._data.shape[2])
            return (w / 2.0, h / 2.0)
        return (0.0, 0.0)

    @staticmethod
    def _arc_pixel_xy(cx, cy, r, a0=0.0, a1=360.0):
        """Pixel-space points of an arc (radius r, angles a0..a1 degrees).

        ROIs live in raw pixel/data coordinates, independent of the incidence
        display correction.
        """
        th = np.deg2rad(np.linspace(a0, a1, 181))
        return cx + r * np.cos(th), cy + r * np.sin(th)

    def _add_curve(self, vx, vy, pen) -> None:
        item = pg.PlotCurveItem(vx, vy, pen=pen)
        item.setZValue(15)
        self._img.view_box.addItem(item)
        self._roi_overlay_items.append(item)

    def _redraw_roi_overlays(self) -> None:
        for item in self._roi_overlay_items:
            try:
                self._img.view_box.removeItem(item)
            except Exception:
                pass
        self._roi_overlay_items = []
        if self._data is None:
            return
        cx, cy = self._center_pixel()
        marker = pg.ScatterPlotItem([cx], [cy], symbol="+", size=18,
                                    pen=pg.mkPen("r", width=2), brush=pg.mkBrush(None))
        marker.setZValue(20)
        self._img.view_box.addItem(marker)
        self._roi_overlay_items.append(marker)

        sel = self._roi_combo.currentIndex()
        for i, roi in enumerate(self._rois):
            selected = (i == sel)
            if roi["type"] == "circle":
                # Persistent interactive item: reposition + recolour, never recreate.
                self._sync_circle_item(roi)
                item = roi.get("item")
                if item is not None:
                    item.setPen(pg.mkPen((0, 255, 255) if selected else (90, 160, 200),
                                         width=2.0 if selected else 1.5))
            else:
                self._draw_one_roi(roi, cx, cy, selected)

    def _draw_one_roi(self, roi: dict, cx: float, cy: float, selected: bool) -> None:
        pen = pg.mkPen((0, 255, 255) if selected else (90, 160, 200),
                       width=2.0 if selected else 1.2)
        ri, ro = float(roi["r_inner"]), float(roi["r_outer"])
        if roi["type"] == "ring":
            for r in (ri, ro):
                if r > 0:
                    vx, vy = self._arc_pixel_xy(cx, cy, r)
                    self._add_curve(vx, vy, pen)
        elif roi["type"] == "sector":
            # A single wedge swept CCW from a_min to a_max (wrap-aware so it stays
            # continuous across the +/-180 seam). Add a second Sector for the other side.
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
        """Pixel (x, y) where a handle sits for the given role."""
        cx, cy = self._center_pixel()
        ri, ro = float(roi["r_inner"]), float(roi["r_outer"])
        if roi["type"] == "ring":
            r = ri if role == "r_inner" else ro
            return cx + r, cy
        # CCW wedge a_min -> a_max (wrap-aware), so the mid-angle is inside it.
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
        else:  # a_max
            ang, rr = a1, ro
        th = np.deg2rad(ang)
        return cx + rr * np.cos(th), cy + rr * np.sin(th)

    def _rebuild_roi_handles(self) -> None:
        """(Re)create drag handles for the selected ring/sector."""
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
        """Move handles to match the current ROI params (without firing their signals)."""
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
            # Set the dragged edge directly; the wedge is CCW from a_min to a_max
            # (wrap-aware), so no min/max clamp — that's what broke crossing +/-180.
            ang = float(np.degrees(np.arctan2(dy, dx)))   # -180..180
            roi[role] = ang
        self._sync_param_sliders_from_roi(roi)
        self._redraw_roi_overlays()
        self._position_roi_handles(skip=role)
        self._schedule_refresh()

    # ── Cached geometry grids (perf) ──────────────────────────────────── #

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

    @staticmethod
    def _arc_mask(angles: np.ndarray, s: float, e: float) -> np.ndarray:
        return (angles >= s) & (angles <= e) if s <= e else (angles >= s) | (angles <= e)

    def _circle_semi_axes(self, roi: dict) -> tuple[float, float]:
        rx = float(roi.get("rx", roi.get("radius", 1.0)))
        ry = float(roi.get("ry", roi.get("radius", 1.0)))
        return max(1.0, rx), max(1.0, ry)

    def _circle_mask(self, roi: dict, h: int, w: int) -> np.ndarray:
        yy, xx = self._grid(h, w)
        rx, ry = self._circle_semi_axes(roi)
        return ((xx - roi["cx"]) / rx) ** 2 + ((yy - roi["cy"]) / ry) ** 2 <= 1.0

    def _roi_mask(self, roi: dict, h: int, w: int) -> np.ndarray:
        """Boolean (H, W) mask of a ROI in pixel space."""
        if roi["type"] == "circle":
            return self._circle_mask(roi, h, w)
        cx, cy = self._center_pixel()
        r, angles = self._polar(h, w, cx, cy)
        ri, ro = float(roi["r_inner"]), float(roi["r_outer"])
        m = r >= ri
        if ro > 0:
            m = m & (r <= ro)
        if roi["type"] == "ring":
            return m
        # Sector: a single wedge (no 180-deg mirror).
        a0, a1 = float(roi["a_min"]) % 360.0, float(roi["a_max"]) % 360.0
        return m & self._arc_mask(angles, a0, a1)

    def _roi_profiles(self, roi: dict, frame: np.ndarray):
        """Return (r_x, r_y, ang_x, ang_y) for a ROI on one frame."""
        h, w = frame.shape[:2]
        if roi["type"] == "circle":
            # I(r) from the spot centre, restricted to the (elliptical) spot.
            cx, cy = roi["cx"], roi["cy"]
            rx_a, ry_a = self._circle_semi_axes(roi)
            ro = int(np.ceil(max(rx_a, ry_a)))
            frame = np.where(self._circle_mask(roi, h, w), frame, np.nan)
            r, angles = self._polar(h, w, cx, cy)
            # Full circle via the symmetric 0..180 convention (avoid 360 % 360 == 0).
            rx, ry = pr.radial_profile(frame, cx, cy, 0, ro, 0, 180, symmetric=True, r=r, angles=angles)
            ax, ay = pr.angular_profile(frame, cx, cy, 0, ro, self._N_ANGLE_BINS, r=r, angles=angles)
            return rx, ry, ax, ay
        cx, cy = self._center_pixel()
        r, angles = self._polar(h, w, cx, cy)
        ri, ro = int(roi["r_inner"]), int(roi["r_outer"])
        if roi["type"] == "ring":
            # Full circle: 0..180 plus its mirror (symmetric) covers all angles.
            rx, ry = pr.radial_profile(frame, cx, cy, ri, ro, 0, 180, symmetric=True, r=r, angles=angles)
        else:
            rx, ry = pr.radial_profile(frame, cx, cy, ri, ro, int(roi["a_min"]), int(roi["a_max"]),
                                       symmetric=False, r=r, angles=angles)
        ax, ay = pr.angular_profile(frame, cx, cy, ri, ro, self._N_ANGLE_BINS, r=r, angles=angles)
        return rx, ry, ax, ay

    # ================================================================== #
    # Display origin + incidence correction                                #
    # ================================================================== #

    def _on_origin_pick_toggled(self, checked: bool) -> None:
        self._origin_pick_active = checked
        self._btn_pick_origin.setText(
            ">>> Click on image to set center <<<" if checked else "Click to Set Center"
        )

    def _on_center_spin_changed(self) -> None:
        self._set_display_origin(self._spin_center_x.value(), self._spin_center_y.value())

    def _on_scene_clicked_origin(self, event) -> None:
        if not self._origin_pick_active:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        # ROIs/Center live in pixel/data coords; view_box coords == data coords.
        # Pick mode stays on so the centre can be clicked repeatedly; "Apply
        # Center" fixes it (FTH-style).
        view_pt = self._img.view_box.mapSceneToView(event.scenePos())
        self._set_display_origin(view_pt.x(), view_pt.y())
        event.accept()

    def _on_apply_center(self) -> None:
        """Fix the centre and leave click-to-set mode (FTH-style)."""
        if self._btn_pick_origin.isChecked():
            self._btn_pick_origin.setChecked(False)
        self._refresh_after_geometry()

    def _set_display_origin(self, px_col: float, px_row: float, *, apply: bool = True) -> None:
        """Set the ROI center (col, row); reflect it in the spinboxes and overlays.

        This is purely the ROI centre — it is independent of the incidence
        correction (which has its own pivot).
        """
        self._display_origin = (float(px_col), float(px_row))
        for spin, val in ((self._spin_center_x, px_col), (self._spin_center_y, px_row)):
            spin.blockSignals(True)
            spin.setValue(float(val))
            spin.blockSignals(False)
        if apply:
            self._refresh_after_geometry()

    def _on_reset_origin(self) -> None:
        """Reset the center back to the image center."""
        if self._data is not None:
            h, w = int(self._data.shape[1]), int(self._data.shape[2])
            self._set_display_origin(w / 2.0, h / 2.0)
        else:
            self._set_display_origin(0.0, 0.0)

    def _apply_incidence(self, combined: np.ndarray) -> np.ndarray:
        """Return the incidence-corrected (resampled) stack, or combined unchanged."""
        if not self._incidence_active:
            return combined
        out, _ = inc.resample_incidence(combined, self._inc_theta_x, self._inc_theta_y)
        return out

    def _on_apply_incidence(self) -> None:
        if self._combined is None:
            self._set_status("Load data before applying incidence correction.", error=True)
            return
        self._inc_theta_x = self._spin_inc_x.value()
        self._inc_theta_y = self._spin_inc_y.value()
        self._incidence_active = self._inc_theta_x > 0.0 or self._inc_theta_y > 0.0
        # Re-derive the analyzed stack from the combined data (resamples the data,
        # so ROIs analyze the corrected image). Changes geometry -> ROIs reset.
        self._set_new_stack(self._combined, f"Incidence X={self._inc_theta_x:.1f}°, "
                                            f"Y={self._inc_theta_y:.1f}°")

    def _on_reset_incidence(self) -> None:
        self._spin_inc_x.setValue(0.0)
        self._spin_inc_y.setValue(0.0)
        self._incidence_active = False
        self._inc_theta_x = self._inc_theta_y = 0.0
        if self._combined is not None:
            self._set_new_stack(self._combined, "Incidence correction reset")

    # ================================================================== #
    # Beam stop mask                                                       #
    # ================================================================== #

    def _add_mask_shape(self, kind: str) -> None:
        if self._data is None:
            QMessageBox.warning(self, "Time Resolved XRMS", "Load data before adding a mask shape.")
            return
        h, w = int(self._data.shape[1]), int(self._data.shape[2])
        self._mask_counter[kind] += 1
        size = max(4, int(0.12 * min(h, w)))
        cx, cy = w / 2.0, h / 2.0
        pen = pg.mkPen((255, 80, 80), width=1.5)
        if kind == "circle":
            # EllipseROI so a circle handle can deform it into an ellipse.
            item = pg.EllipseROI([cx - size / 2, cy - size / 2], [size, size], pen=pen, movable=True)
        else:
            item = pg.RectROI([cx - size / 2, cy - size / 2], [size, size], pen=pen, movable=True)
            item.addRotateHandle([1, 0], [0.5, 0.5])   # corner handle rotates about the centre
        item.setZValue(14)
        item.sigRegionChanged.connect(lambda *_: self._mask_timer.start())
        self._img.view_box.addItem(item)
        self._mask_shapes.append({"type": kind, "item": item})
        self._recompute_beamstop()

    def _ensure_mask_arrays(self) -> None:
        if self._data is None:
            return
        h, w = int(self._data.shape[1]), int(self._data.shape[2])
        if self._brush_mask is None or self._brush_mask.shape != (h, w):
            self._brush_mask = np.zeros((h, w), dtype=bool)

    def _rasterize_shapes(self, h: int, w: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=bool)
        if not self._mask_shapes:
            return m
        yy, xx = self._grid(h, w)
        for s in self._mask_shapes:
            item = s["item"]
            pos, size = item.pos(), item.size()
            if s["type"] == "circle":
                rx, ry = max(float(size.x()) / 2.0, 1e-6), max(float(size.y()) / 2.0, 1e-6)
                cx, cy = float(pos.x()) + rx, float(pos.y()) + ry
                m |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            else:
                # Rotated rectangle: map pixels into the rect's local frame.
                ox, oy = float(pos.x()), float(pos.y())
                wv, hv = float(size.x()), float(size.y())
                ang = np.deg2rad(float(item.angle()))
                ca, sa = np.cos(ang), np.sin(ang)
                dx, dy = xx - ox, yy - oy
                lx = dx * ca + dy * sa
                ly = -dx * sa + dy * ca
                m |= (lx >= 0) & (lx <= wv) & (ly >= 0) & (ly <= hv)
        return m

    def _update_effective_beamstop(self, live_overlay: bool = True) -> None:
        if self._data is None:
            self._beamstop = None
            return
        h, w = int(self._data.shape[1]), int(self._data.shape[2])
        eff = self._shapes_raster.copy() if self._shapes_raster is not None \
            else np.zeros((h, w), dtype=bool)
        if self._brush_mask is not None and self._brush_mask.shape == (h, w):
            eff |= self._brush_mask
        self._beamstop = eff if eff.any() else None
        if live_overlay:
            self._update_mask_overlay()

    def _recompute_beamstop(self) -> None:
        """Recompute the cached shape raster + effective mask, then refresh analysis."""
        if self._data is not None:
            h, w = int(self._data.shape[1]), int(self._data.shape[2])
            self._shapes_raster = self._rasterize_shapes(h, w)
        self._update_effective_beamstop(live_overlay=True)
        self._schedule_refresh()

    def _on_apply_mask(self) -> None:
        """Bake the current shapes into the committed red mask and remove their handles."""
        if self._data is None:
            return
        h, w = int(self._data.shape[1]), int(self._data.shape[2])
        self._ensure_mask_arrays()
        if self._mask_shapes:
            self._brush_mask |= self._rasterize_shapes(h, w)
            for s in self._mask_shapes:
                try:
                    self._img.view_box.removeItem(s["item"])
                except Exception:
                    pass
            self._mask_shapes = []
            self._shapes_raster = None
        self._update_effective_beamstop(live_overlay=True)
        self._schedule_refresh()

    def _update_mask_overlay(self) -> None:
        if self._data is None:
            return
        h, w = int(self._data.shape[1]), int(self._data.shape[2])
        if self._mask_overlay is None:
            self._mask_overlay = pg.ImageItem()
            self._mask_overlay.setZValue(12)
            self._img.view_box.addItem(self._mask_overlay)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if self._beamstop is not None:
            rgba[self._beamstop] = (255, 0, 0, 110)
        self._mask_overlay.setImage(rgba, autoLevels=False)

    def _on_brush_toggled(self, checked: bool) -> None:
        self._brush_active = checked
        # Disable view panning while painting so a drag paints instead of pans.
        self._img.view_box.setMouseEnabled(not checked, not checked)
        self._btn_brush.setText(">>> Painting — drag on image <<<" if checked else "Brush (paint)")

    def _on_brush_move(self, scene_pos) -> None:
        if not self._brush_active or self._data is None:
            return
        if not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton):
            return
        view_pt = self._img.view_box.mapSceneToView(scene_pos)
        self._stamp_brush(view_pt.x(), view_pt.y())

    def _stamp_brush(self, col: float, row: float) -> None:
        self._ensure_mask_arrays()
        if self._brush_mask is None:
            return
        h, w = self._brush_mask.shape
        c, rr, r = int(round(col)), int(round(row)), int(self._brush_size)
        if not (-r <= c < w + r and -r <= rr < h + r):
            return
        y0, y1 = max(0, rr - r), min(h, rr + r + 1)
        x0, x1 = max(0, c - r), min(w, c + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        self._brush_mask[y0:y1, x0:x1] |= (xx - c) ** 2 + (yy - rr) ** 2 <= r * r
        self._update_effective_beamstop(live_overlay=True)
        self._schedule_refresh()

    def _clear_beamstop(self) -> None:
        for s in self._mask_shapes:
            try:
                self._img.view_box.removeItem(s["item"])
            except Exception:
                pass
        self._mask_shapes = []
        self._shapes_raster = None
        if self._brush_mask is not None:
            self._brush_mask[:] = False
        self._beamstop = None
        self._update_mask_overlay()
        self._schedule_refresh()

    def _reset_beamstop_for_new_geometry(self) -> None:
        """Drop the mask when the image geometry changes (e.g. incidence applied)."""
        for s in self._mask_shapes:
            try:
                self._img.view_box.removeItem(s["item"])
            except Exception:
                pass
        self._mask_shapes = []
        self._shapes_raster = None
        self._brush_mask = None
        self._beamstop = None
        if self._btn_brush.isChecked():
            self._btn_brush.setChecked(False)
        self._update_mask_overlay()

    def _analysis_frame(self, idx: int) -> np.ndarray:
        """The current frame with beam-stop-masked pixels set to NaN."""
        frame = self._data[idx]
        if self._beamstop is not None:
            frame = np.where(self._beamstop, np.nan, frame)
        return frame

    def _analysis_stack(self) -> np.ndarray:
        """The whole stack with beam-stop-masked pixels set to NaN (for frame analysis)."""
        if self._beamstop is None:
            return self._data
        return np.where(self._beamstop[np.newaxis, :, :], np.nan, self._data)

    # ================================================================== #
    # Profile / time computation (driven by the selected ROI)              #
    # ================================================================== #

    @staticmethod
    def _apply_phase_shift(ay: np.ndarray, phase_deg: int) -> np.ndarray:
        """Roll the azimuthal profile along θ by phase_deg (bins span 0..360)."""
        if not phase_deg or ay.size == 0:
            return ay
        shift = int(round(phase_deg * ay.size / 360.0))
        return np.roll(ay, shift)

    def _update_profile_plots(self) -> None:
        """Compute the selected ROI's I(r)/I(θ) and plot the one matching its mode."""
        self._plot_profile.clear()
        self._radial_x = self._radial_y = None
        self._angular_x = self._angular_y = None
        roi = self._selected_roi()
        if self._data is None or roi is None:
            self._plot_profile.setTitle("Profile — add and select a ROI", color="w")
            return
        frame = self._analysis_frame(self._sl_frame.value())
        rx, ry, ax, ay = self._roi_profiles(roi, frame)
        if len(rx) > 0:
            self._radial_x = np.asarray(rx, dtype=np.float64)
            self._radial_y = np.asarray(ry, dtype=np.float64)
        if len(ax) > 0:
            self._angular_x = np.asarray(ax, dtype=np.float64)
            self._angular_y = self._apply_phase_shift(np.asarray(ay, dtype=np.float64),
                                                      int(roi.get("phase", 0)))

        mode = roi.get("mode", "radial")
        if mode == "angular" and self._angular_x is not None:
            valid = np.isfinite(self._angular_y)
            self._plot_profile.plot(self._angular_x[valid], self._angular_y[valid],
                                    pen=pg.mkPen((255, 160, 0), width=1.5))
            self._plot_profile.setLabel("bottom", "Angle (°)")
            self._plot_profile.setTitle(f"I(θ)  |  {roi['name']}", color="w")
        elif self._radial_x is not None:
            self._plot_profile.plot(self._radial_x, self._radial_y, pen=pg.mkPen("c", width=1.5))
            self._plot_profile.setLabel("bottom", "Radius (pixels)")
            self._plot_profile.setTitle(f"I(r)  |  {roi['name']}", color="w")
        else:
            self._plot_profile.setTitle(f"{roi['name']} — empty region", color="w")

    def _update_time_plot(self) -> None:
        """Compute the selected ROI's mean-intensity-per-frame series (no plot)."""
        self._time_x = self._time_y = None
        if self._data is None:
            return
        h, w = int(self._data.shape[1]), int(self._data.shape[2])
        flat2d = self._data.reshape(self._n_frames, -1)
        roi = self._selected_roi()
        if roi is None:
            sel = ~self._beamstop.ravel() if self._beamstop is not None else slice(None)
            means = np.nanmean(flat2d[:, sel], axis=1).astype(np.float64)
        else:
            mask = self._roi_mask(roi, h, w)
            if self._beamstop is not None:
                mask = mask & ~self._beamstop   # exclude beam-stop pixels
            if not mask.any():
                return
            # Vectorized over all frames at once (no per-frame Python loop).
            means = np.nanmean(flat2d[:, mask.ravel()], axis=1).astype(np.float64)
        self._time_x = np.arange(self._n_frames, dtype=np.float64)
        self._time_y = means

    def _compute_current_profiles(self) -> None:
        """Ensure the selected ROI's profiles for the current frame are up to date."""
        self._update_profile_plots()
        self._lbl_fit_info.setText(f"Frame: {self._sl_frame.value() + 1} / {self._n_frames}")

    # ================================================================== #
    # Curve fitting                                                        #
    # ================================================================== #

    def _fit_profile_mode(self) -> str:
        return self._combo_fit_profile.currentData() or "radial"

    def _fit_active_xy(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        mode = self._fit_profile_mode()
        if mode == "angular":
            return self._angular_x, self._angular_y
        if mode == "time":
            return self._time_x, self._time_y
        return self._radial_x, self._radial_y

    def _fit_unit(self) -> str:
        return {"angular": "°", "time": "frame"}.get(self._fit_profile_mode(), "px")

    def _fit_models_registry(self) -> dict:
        """Sin-power models for angular; peak-only models for radial."""
        return cf.SIN_MODELS if self._fit_profile_mode() == "angular" else cf.PEAK_MODELS

    def _rebuild_fit_model_combo(self) -> None:
        self._combo_fit_model.blockSignals(True)
        self._combo_fit_model.clear()
        for name in self._fit_models_registry():
            self._combo_fit_model.addItem(name)
        self._combo_fit_model.blockSignals(False)

    def _on_fit_profile_changed(self, _idx: int = 0) -> None:
        mode = self._fit_profile_mode()
        x_label = {"angular": "Angle (°)", "time": "Frame"}.get(mode, "Radius (pixels)")
        self._fit_top.setLabel("bottom", x_label)
        self._fit_bottom.setLabel("bottom", x_label)
        # Angular has no slope-background subtraction.
        is_angular = mode == "angular"
        self._g_bg.setVisible(not is_angular)
        if is_angular and self._chk_bg.isChecked():
            self._chk_bg.blockSignals(True)
            self._chk_bg.setChecked(False)
            self._chk_bg.blockSignals(False)
        self._bg_enabled = self._chk_bg.isChecked() and not is_angular
        self._rebuild_fit_model_combo()
        self._on_fit_model_changed(0)
        self._reset_fit_lines()
        self._arrange_fit_lines()
        self._redraw_fit_curves()

    # ── Fit-range / background draggable lines ────────────────────────── #

    def _reset_fit_lines(self) -> None:
        x, y = self._fit_active_xy()
        if x is None or len(x) == 0:
            return
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        span = max(xmax - xmin, 1.0)
        self._bg_xs = [xmin, xmin + 0.15 * span, xmax - 0.15 * span, xmax]
        self._range_xs = [xmin, xmax]
        self._apply_line_positions()

    def _apply_line_positions(self) -> None:
        for lines, xs in ((self._bg_lines, self._bg_xs), (self._range_lines, self._range_xs)):
            if not xs:
                continue
            for ln, xv in zip(lines, xs):
                ln.blockSignals(True)
                ln.setValue(xv)
                ln.blockSignals(False)

    def _arrange_fit_lines(self) -> None:
        """Place lines on the correct plots; the bottom plot shows only with BG on."""
        for ln in self._bg_lines + self._range_lines:
            for p in (self._fit_top, self._fit_bottom):
                try:
                    p.removeItem(ln)
                except Exception:
                    pass
        self._fit_bottom.setVisible(self._bg_enabled)
        active = self._fit_bottom if self._bg_enabled else self._fit_top
        if self._bg_enabled:
            for ln in self._bg_lines:
                self._fit_top.addItem(ln)
        for ln in self._range_lines:
            active.addItem(ln)

    def _on_fit_line_moved(self, *_a) -> None:
        if self._bg_xs is not None:
            self._bg_xs = [float(ln.value()) for ln in self._bg_lines]
        if self._range_xs is not None:
            self._range_xs = [float(ln.value()) for ln in self._range_lines]
        self._fit_line_timer.start()

    def _add_fit_item(self, plot, item) -> None:
        plot.addItem(item)
        self._fit_items.append((plot, item))

    def _clear_fit_items(self) -> None:
        for plot, item in self._fit_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        self._fit_items = []

    def _fit_signal(self):
        """Return (x, y_to_fit, active_plot, bg_info) for the current config."""
        x, y = self._fit_active_xy()
        if x is None or y is None or len(x) == 0:
            return None
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self._bg_xs is None:
            self._reset_fit_lines()
        if self._bg_enabled and self._bg_xs:
            x1, x2, x3, x4 = sorted(self._bg_xs)
            deg = int(self._combo_bg_degree.currentData())
            y_corr, bg, coeffs = cf.compute_poly_background(x, y, x1, x2, x3, x4, deg)
            return x, y_corr, self._fit_bottom, (bg, coeffs, deg)
        return x, y, self._fit_top, None

    def _redraw_fit_curves(self) -> None:
        """Redraw the profile / background / subtracted curves (keeps the lines)."""
        self._clear_fit_items()
        sig = self._fit_signal()
        if sig is None:
            self._fit_top.setTitle("No profile — set up a ROI in Image && Region", color="w")
            return
        x, y_used, _active, bg_info = sig
        x_raw, y_raw = self._fit_active_xy()
        y_raw = np.asarray(y_raw, dtype=np.float64)
        mode = self._fit_profile_mode()
        pen_color = {"angular": (255, 160, 0), "time": (100, 200, 255)}.get(mode, "c")
        valid = np.isfinite(y_raw)
        self._add_fit_item(self._fit_top, pg.PlotCurveItem(
            np.asarray(x_raw)[valid], y_raw[valid], pen=pg.mkPen(pen_color, width=1.5)))
        if bg_info is not None:
            bg, _coeffs, _deg = bg_info
            self._add_fit_item(self._fit_top, pg.PlotCurveItem(
                x, bg, pen=pg.mkPen((0, 200, 80), width=1.5, style=Qt.PenStyle.DashLine)))
            vc = np.isfinite(y_used)
            self._add_fit_item(self._fit_bottom, pg.PlotCurveItem(
                x[vc], y_used[vc], pen=pg.mkPen(pen_color, width=1.5)))
            self._fit_top.setTitle("Profile + background (drag green lines)", color="w")
            self._fit_bottom.setTitle("Subtracted — drag yellow lines, then Run Fit", color="w")
        else:
            self._fit_top.setTitle("Profile — drag yellow lines, then Run Fit", color="w")

    def _on_fit_model_changed(self, _idx: int = 0) -> None:
        self._guess_attempt = 0
        while self._fl_fit_params.rowCount() > 1:
            self._fl_fit_params.removeRow(1)
        self._fit_param_spins.clear()
        reg = self._fit_models_registry()
        model_name = self._combo_fit_model.currentText()
        if model_name not in reg:
            return
        x, y = self._fit_active_xy()
        guesses = cf.auto_guess(model_name, x, y, models=reg) if x is not None else None
        for i, (pname, default) in enumerate(reg[model_name]["params"]):
            spin = QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setValue(guesses[i] if guesses is not None else default)
            self._fl_fit_params.addRow(f"{pname}:", spin)
            self._fit_param_spins.append(spin)

    def _fit_range(self) -> tuple[float, float]:
        if self._range_xs:
            return min(self._range_xs), max(self._range_xs)
        x, _ = self._fit_active_xy()
        if x is not None and len(x):
            return float(np.nanmin(x)), float(np.nanmax(x))
        return 0.0, 1e12

    def _on_auto_guess(self) -> None:
        reg = self._fit_models_registry()
        model_name = self._combo_fit_model.currentText()
        if model_name not in reg:
            return
        # Guess from the (background-subtracted) signal over the fit range.
        sig = self._fit_signal()
        x, y = (sig[0], sig[1]) if sig is not None else self._fit_active_xy()
        lo, hi = self._fit_range()
        base = cf.auto_guess(model_name, x, y, lo, hi, models=reg)
        if base is None:
            return
        if self._guess_attempt == 0:
            guesses = base
        else:
            rng = np.random.default_rng()
            guesses = []
            for val in base:
                if val == 0.0:
                    guesses.append(float(rng.uniform(-0.1, 0.1)))
                else:
                    guesses.append(float(np.sign(val) * abs(val) * float(rng.lognormal(0.0, 0.6))))
        self._guess_attempt += 1
        for spin, val in zip(self._fit_param_spins, guesses):
            spin.setValue(val)

    def _on_bg_toggled(self, checked: bool) -> None:
        self._bg_enabled = checked and self._fit_profile_mode() != "angular"
        self._arrange_fit_lines()
        self._redraw_fit_curves()

    def _on_fit_background(self) -> None:
        """Fit (and show) the polynomial background from the 4 baseline lines."""
        if self._fit_profile_mode() == "angular":
            return
        if not self._chk_bg.isChecked():
            self._chk_bg.setChecked(True)   # -> _on_bg_toggled redraws everything
        else:
            self._bg_enabled = True
            self._arrange_fit_lines()
            self._redraw_fit_curves()

    def _on_run_fit(self) -> None:
        if self._fit_active_xy()[0] is None:
            self._compute_current_profiles()
        sig = self._fit_signal()
        if sig is None:
            self._fit_results.setPlainText(
                "No profile available.\nSet up a dataset and ROI in the Image && Region tab."
            )
            return
        x, y_used, active, bg_info = sig
        self._redraw_fit_curves()   # fresh base (data / bg / subtracted)

        lo, hi = self._fit_range()
        reg = self._fit_models_registry()
        model_name = self._combo_fit_model.currentText()
        if model_name not in reg:
            return
        func = reg[model_name]["func"]
        p0 = [spin.value() for spin in self._fit_param_spins]
        res = cf.run_fit(func, x, y_used, p0, lo, hi)
        if not res["ok"]:
            self._fit_results.setPlainText(
                f"Fit failed:\n{res['error']}\n\n"
                "Tips:\n• Try Auto-guess\n• Adjust the yellow range lines\n• Try a different model"
            )
            return

        unit = self._fit_unit()
        names = [n for n, _ in reg[model_name]["params"]]
        lines = [
            f"Profile: {self._fit_profile_mode()}",
            f"Model: {model_name}",
            f"Fit range: [{lo:.1f}, {hi:.1f}] {unit}  ({len(res['x_fit'])} points)",
        ]
        if bg_info is not None and bg_info[1] is not None:
            _bg, coeffs, deg = bg_info
            lines.append(f"BG poly (deg {deg}): " + ", ".join(f"{c:+.4g}" for c in coeffs))
        lines.append("")
        for pname, val, err in zip(names, res["popt"], res["perr"]):
            lines.append(f"  {pname:>4} = {val:.6g}  ±  {err:.3g}")
        lines += [f"\n  R²   = {res['r2']:.6f}", f"  RMSE = {res['rmse']:.4g}"]
        self._fit_results.setPlainText("\n".join(lines))

        # Draw the fit curve + range markers on the active plot (kept until next redraw).
        x_fit = res["x_fit"]
        x_dense = np.linspace(float(x_fit[0]), float(x_fit[-1]), max(600, len(x_fit) * 5))
        self._add_fit_item(active, pg.PlotCurveItem(
            x_dense, func(x_dense, *res["popt"]), pen=pg.mkPen("r", width=2)))
        active.setTitle(f"Fit — R² = {res['r2']:.5f}   RMSE = {res['rmse']:.4g}", color="w")

    def _on_analyze_all_frames(self) -> None:
        if self._data is None or self._n_frames < 2:
            QMessageBox.warning(self, "Frame Analysis", "Load a 3-D stack first.")
            return
        roi = self._selected_roi()
        if roi is None:
            QMessageBox.warning(self, "Frame Analysis", "Select a Ring or Sector ROI first.")
            return
        mode = self._fit_profile_mode()
        if mode == "time":
            QMessageBox.information(self, "Frame Analysis",
                                    "Frame analysis applies to the radial or angular profile, "
                                    "not the time series. Switch the profile selector to I(r) or I(θ).")
            return
        if roi["type"] == "circle":
            px_col, px_row = roi["cx"], roi["cy"]
            r_min, r_max, a_min, a_max = 0, int(np.ceil(max(roi["rx"], roi["ry"]))), 0, 180
        else:
            px_col, px_row = self._center_pixel()
            r_min, r_max = int(roi["r_inner"]), int(roi["r_outer"])
            a_min, a_max = (0, 180) if roi["type"] == "ring" else (int(roi["a_min"]), int(roi["a_max"]))
        model_name = self._combo_fit_model.currentText()
        p0 = [spin.value() for spin in self._fit_param_spins]

        from src.gui.frame_analysis_tool import FrameAnalysisTool
        dlg = FrameAnalysisTool(
            data3d=self._analysis_stack(),
            cx=px_col, cy=px_row,
            r_min=r_min, r_max=r_max,
            a_min=a_min, a_max=a_max,
            fit_models=self._fit_models_registry(),
            active_mode=mode,
            model_name=model_name,
            p0=p0,
            fit_lo=self._fit_range()[0],
            fit_hi=self._fit_range()[1],
            parent=self,
        )
        dlg.show()

    # ================================================================== #
    # Helpers                                                              #
    # ================================================================== #

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet("color: #b00020;" if error else "color: #444;")

    def closeEvent(self, event) -> None:
        try:
            self._clear_all_rois()
        except Exception:
            pass
        super().closeEvent(event)


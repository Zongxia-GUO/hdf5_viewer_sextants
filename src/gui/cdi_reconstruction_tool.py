"""CDI Phase Retrieval Reconstruction Tool.

Embeds FTH Reconstruction Tool tabs (Alignment + Filter/FTH) as the first two pages,
then adds CDI Reconstruction and Summary/Save pages.  This creates a unified XMCD-CDI
workflow: align and reconstruct FTH -> define support on FTH image -> run CDI with
the support FFT phase as the initial guess.

Tab layout:
  Tab 1 - Alignment           (FTH tool tab 0 - load CL/CR, center, beamstop)
  Tab 2 - Filter && FTH       (FTH tool tab 2 - filters, FTH reconstruction, support shapes)
  Tab 3 - CDI Reconstruction  (algorithm sequence, parameters, run/stop, live preview)
  Tab 4 - Summary && Save     (amplitude / phase / error curve + export)
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import median_filter
from PyQt6.QtCore import QByteArray, QBuffer, QMimeData, Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui._shared import (
    apply_colormap as _apply_colormap,
    set_combo_light_palette as _set_combo_light_palette,
    set_widget_light_palette as _set_widget_light_palette,
)
from src.gui.dataset_path_combo import DatasetPathCombo
from src.gui.fth_reconstruction_tool import FTHReconstructionTool
from src.recon.cdi import (
    fft2c as _fft2c,
    ifft2c as _ifft2c,
    run_multi_target_cdi as _run_multi_target_cdi,
    run_sequential_cdi as _run_sequential_cdi,
    run_single_target_cdi as _run_single_target_cdi,
)

log = logging.getLogger(__name__)

CDI_COLORMAPS = [
    "gray", "viridis", "inferno", "cividis", "turbo",
    "CET-L9", "CET-L1", "CET-L4", "CET-R4", "CET-D1", "CET-D9",
]

CDI_ALGORITHM_LABELS = {
    "mine": "Example",
    "er": "ER",
    "hio": "HIO",
    "raar": "RAAR",
    "er_shrinkwrap": "ER Shrinkwarp",
    "hio_shrinkwrap": "HIO Shrinkwarp",
    "raar_shrinkwrap": "RAAR Shrinkwarp",
}
CDI_ALGORITHMS = list(CDI_ALGORITHM_LABELS)
CDI_ALGORITHM_DISPLAY_TO_KEY = {
    label: key for key, label in CDI_ALGORITHM_LABELS.items()
}

CDI_INTERVAL_STEPS = [
    "none",
    "match data amplitude",
    "normalize",
    "reset support",
    "random phase",
]


# ---------------------------------------------------------------------------
# Dataset combo
# ---------------------------------------------------------------------------

class CDIDatasetCombo(DatasetPathCombo):
    """CDI dataset selector: accepts 2D+ datasets via drag-drop or dropdown."""

    def __init__(self, placeholder: str = "-- none --", parent=None) -> None:
        super().__init__(placeholder=placeholder, parent=parent)

    def populate(self, opened_files) -> None:
        super().populate(opened_files, min_ndim=2)


# ---------------------------------------------------------------------------
# Circle entry — groups ROI and fallback values for one support circle
# ---------------------------------------------------------------------------

class _CircleEntry:
    """Keeps a pg.CircleROI in sync with Y/X/R spinboxes (all optional)."""

    def __init__(
        self,
        roi: pg.CircleROI,
        y: float = 0.0,
        x: float = 0.0,
        r: float = 1.0,
        widget: Optional[QWidget] = None,
        y_spin: Optional[QDoubleSpinBox] = None,
        x_spin: Optional[QDoubleSpinBox] = None,
        r_spin: Optional[QDoubleSpinBox] = None,
    ) -> None:
        self.roi    = roi
        self.widget = widget
        self.y_spin = y_spin
        self.x_spin = x_spin
        self.r_spin = r_spin
        # Fallback values used when spinboxes are None
        self._y = float(y)
        self._x = float(x)
        self._r = float(r)

    @property
    def y(self) -> float:
        return self.y_spin.value() if self.y_spin is not None else self._y

    @property
    def x(self) -> float:
        return self.x_spin.value() if self.x_spin is not None else self._x

    @property
    def r(self) -> float:
        return self.r_spin.value() if self.r_spin is not None else self._r

    def read_from_roi(self) -> None:
        """Pull ROI geometry → update spinboxes and fallback values."""
        pos  = self.roi.pos()
        size = self.roi.size()
        r    = size[0] / 2.0
        cx   = pos[0] + r
        cy   = pos[1] + r
        self._y = float(cy)
        self._x = float(cx)
        self._r = float(abs(r))
        if self.y_spin is None:
            return
        for spin in (self.y_spin, self.x_spin, self.r_spin):
            spin.blockSignals(True)
        self.y_spin.setValue(self._y)
        self.x_spin.setValue(self._x)
        self.r_spin.setValue(self._r)
        for spin in (self.y_spin, self.x_spin, self.r_spin):
            spin.blockSignals(False)

    def write_to_roi(self) -> None:
        """Push current Y/X/R → update ROI geometry."""
        y, x, r = self.y, self.x, self.r
        self._y, self._x, self._r = y, x, r
        self.roi.blockSignals(True)
        self.roi.setPos((x - r, y - r))
        self.roi.setSize((2.0 * r, 2.0 * r))
        self.roi.blockSignals(False)

    def set_values(self, y: float, x: float, r: float) -> None:
        """Set Y/X/R and propagate to both spinboxes and ROI."""
        self._y, self._x, self._r = float(y), float(x), float(r)
        if self.y_spin is not None:
            for spin, val in ((self.y_spin, y), (self.x_spin, x), (self.r_spin, r)):
                spin.blockSignals(True)
                spin.setValue(float(val))
                spin.blockSignals(False)
        self.write_to_roi()


class _SupportShapeEntry:
    """Keeps one circle or rectangle ROI in sync with support parameters."""

    def __init__(
        self,
        kind: str,
        roi,
        y: float,
        x: float,
        a: float,
        b: Optional[float] = None,
        angle: float = 0.0,
    ) -> None:
        self.kind = kind
        self.roi = roi
        self._y = float(y)
        self._x = float(x)
        self._a = float(a)
        self._b = float(a if b is None else b)
        self._angle = float(angle)

    @property
    def y(self) -> float:
        return self._y

    @property
    def x(self) -> float:
        return self._x

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def r(self) -> float:
        return self._a

    @property
    def angle(self) -> float:
        return self._angle

    def read_from_roi(self) -> None:
        size = self.roi.size()
        if self.kind == "circle":
            pos = self.roi.pos()
            self._a = abs(float(size[0])) / 2.0
            self._b = self._a
            self._x = float(pos[0]) + self._a
            self._y = float(pos[1]) + self._a
            self._angle = 0.0
        else:
            self._a = max(0.1, abs(float(size[0])))
            self._b = max(0.1, abs(float(size[1])))
            center = self.roi.mapToParent(self._a / 2.0, self._b / 2.0)
            self._x = float(center.x())
            self._y = float(center.y())
            self._angle = float(self.roi.angle())

    def set_values(
        self,
        y: float,
        x: float,
        a: float,
        b: Optional[float] = None,
        angle: Optional[float] = None,
    ) -> None:
        self._y = float(y)
        self._x = float(x)
        self._a = max(0.1, float(a))
        self._b = max(0.1, float(self._a if b is None else b))
        if angle is not None:
            self._angle = float(angle)
        self.roi.blockSignals(True)
        if self.kind == "circle":
            self.roi.setPos((self._x - self._a, self._y - self._a))
            self.roi.setSize((2.0 * self._a, 2.0 * self._a))
        else:
            self.roi.setSize((self._a, self._b))
            self.roi.setPos((self._x - self._a / 2.0, self._y - self._b / 2.0))
            self.roi.setAngle(self._angle, center=[0.5, 0.5], update=False, finish=False)
            center = self.roi.mapToParent(self._a / 2.0, self._b / 2.0)
            dx = self._x - float(center.x())
            dy = self._y - float(center.y())
            pos = self.roi.pos()
            self.roi.setPos((float(pos.x()) + dx, float(pos.y()) + dy))
        self.roi.blockSignals(False)


# ---------------------------------------------------------------------------
# Curves widget  (Photoshop-style spline → intensity-to-mask-weight mapping)
# ---------------------------------------------------------------------------

class _CurvesWidget(pg.PlotWidget):
    """Interactive PCHIP-spline curves widget.

    X axis = pixel intensity.  Y axis = mask weight [0, 1].
    Pixels where the curve exceeds 0.5 are considered masked.

    Interaction:
      • Left-click on empty area  → add control point
      • Left-drag existing point  → move it  (endpoints fixed in X)
      • Right-click on point      → remove it  (endpoints are permanent)
    """

    curveChanged = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackground("#111")
        self.setMenuEnabled(False)
        vb = self.getViewBox()
        vb.setMenuEnabled(False)
        vb.setMouseEnabled(x=False, y=False)
        vb.setRange(xRange=(0.0, 1.0), yRange=(-0.05, 1.1), padding=0)
        self.hideAxis("left")
        self.getAxis("bottom").setLabel("Intensity", size="8pt")

        # Horizontal reference lines at 0, 0.5 (mask boundary), 1
        for y_val, color, style in (
            (0.0,  (180, 180, 180, 60),  Qt.PenStyle.SolidLine),
            (0.5,  (255, 200,   0, 100), Qt.PenStyle.DashLine),
            (1.0,  (180, 180, 180, 60),  Qt.PenStyle.SolidLine),
        ):
            self.addItem(
                pg.InfiniteLine(pos=y_val, angle=0,
                                pen=pg.mkPen(color, width=1, style=style)),
                ignoreBounds=True,
            )

        # Histogram bars (background, log-normalised to [0, 0.9])
        self._hist_item = pg.BarGraphItem(
            x=np.array([0.5]), height=np.array([0.0]), width=1.0,
            brush=pg.mkBrush(60, 110, 220, 90), pen=pg.mkPen(None),
        )
        self.addItem(self._hist_item)

        # Spline curve line
        self._curve_item = pg.PlotCurveItem(
            pen=pg.mkPen((220, 220, 220), width=2), antialias=True,
        )
        self.addItem(self._curve_item)

        # Control-point scatter
        self._pts_item = pg.ScatterPlotItem(
            size=11,
            pen=pg.mkPen((255, 255, 255), width=2),
            brush=pg.mkBrush(30, 30, 30),
            hoverable=True,
            hoverPen=pg.mkPen((255, 200, 0), width=2),
            hoverBrush=pg.mkBrush(80, 80, 40),
        )
        self.addItem(self._pts_item)

        self._ctrl_pts: list[list[float]] = [[0.0, 0.0], [1.0, 0.0]]
        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._dragging_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, data: np.ndarray) -> None:
        """Rebuild histogram; reset curve only when data range changes."""
        flat = data.ravel()
        pos = flat[np.isfinite(flat) & (flat > 0)]
        if pos.size < 2:
            return
        x_min = float(pos.min())
        x_max = float(pos.max())
        changed = (x_min != self._x_min or x_max != self._x_max)
        self._x_min = x_min
        self._x_max = x_max

        counts, edges = np.histogram(pos, bins=400)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        log_h = np.log10(counts.astype(float) + 1.0)
        peak = log_h.max()
        if peak > 0:
            log_h = log_h / peak * 0.9
        self._hist_item.setOpts(x=centers, height=log_h, width=widths)
        self.getViewBox().setRange(xRange=(x_min, x_max), yRange=(-0.05, 1.1), padding=0)

        if changed:
            self._ctrl_pts = [[x_min, 0.0], [x_max, 0.0]]
            self._redraw()

    def reset_curve(self) -> None:
        self._ctrl_pts = [[self._x_min, 0.0], [self._x_max, 0.0]]
        self._redraw()
        self.curveChanged.emit()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Return mask weight ∈ [0, 1] interpolated at each intensity value."""
        if len(self._ctrl_pts) < 2:
            return np.zeros(np.asarray(x).shape, dtype=float)
        xs = np.array([p[0] for p in self._ctrl_pts], dtype=float)
        ys = np.array([p[1] for p in self._ctrl_pts], dtype=float)
        xf = np.asarray(x, dtype=float)
        if len(xs) >= 3:
            from scipy.interpolate import PchipInterpolator
            out = PchipInterpolator(xs, ys, extrapolate=False)(xf)
            out = np.where(np.isfinite(out), out, 0.0)
        else:
            out = np.interp(xf, xs, ys)
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        if self._x_min >= self._x_max or len(self._ctrl_pts) < 2:
            return
        xs = np.linspace(self._x_min, self._x_max, 512)
        ys = self.evaluate(xs)
        self._curve_item.setData(xs, ys)
        self._pts_item.setData(
            x=[p[0] for p in self._ctrl_pts],
            y=[p[1] for p in self._ctrl_pts],
        )

    def _scene_to_data(self, scene_pos) -> tuple[float, float]:
        pt = self.getViewBox().mapSceneToView(scene_pos)
        return float(pt.x()), float(pt.y())

    def _nearest_ctrl_idx(self, x: float, y: float) -> Optional[int]:
        vr = self.getViewBox().viewRect()
        x_rng = max(vr.width(), 1e-12)
        y_rng = max(vr.height(), 1e-12)
        tol = 0.045
        best_d, best_i = float("inf"), None
        for i, (px, py) in enumerate(self._ctrl_pts):
            d = (((px - x) / x_rng) ** 2 + ((py - y) / y_rng) ** 2) ** 0.5
            if d < tol and d < best_d:
                best_d, best_i = d, i
        return best_i

    # ------------------------------------------------------------------
    # Mouse events  (override QGraphicsView for reliable press/move/release)
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        vb = self.getViewBox()
        sp = self.mapToScene(ev.pos())
        if not vb.sceneBoundingRect().contains(sp):
            super().mousePressEvent(ev)
            return
        x, y = self._scene_to_data(sp)
        y = max(0.0, min(1.0, y))

        if ev.button() == Qt.MouseButton.LeftButton:
            idx = self._nearest_ctrl_idx(x, y)
            if idx is not None:
                self._dragging_idx = idx
            elif self._x_min <= x <= self._x_max:
                # Insert at sorted position (never before first or after last)
                insert_at = next(
                    (i for i, p in enumerate(self._ctrl_pts) if p[0] > x),
                    len(self._ctrl_pts),
                )
                insert_at = max(1, min(insert_at, len(self._ctrl_pts) - 1))
                self._ctrl_pts.insert(insert_at, [x, y])
                self._dragging_idx = insert_at
                self._redraw()
                self.curveChanged.emit()

        elif ev.button() == Qt.MouseButton.RightButton:
            idx = self._nearest_ctrl_idx(x, y)
            if idx is not None and 0 < idx < len(self._ctrl_pts) - 1:
                self._ctrl_pts.pop(idx)
                self._dragging_idx = None
                self._redraw()
                self.curveChanged.emit()

    def mouseMoveEvent(self, ev) -> None:
        if self._dragging_idx is None:
            super().mouseMoveEvent(ev)
            return
        x, y = self._scene_to_data(self.mapToScene(ev.pos()))
        y = max(0.0, min(1.0, y))
        idx = self._dragging_idx
        pts = self._ctrl_pts
        margin = max((self._x_max - self._x_min) * 1e-6, 1e-12)
        if idx == 0:
            x = pts[0][0]
        elif idx == len(pts) - 1:
            x = pts[-1][0]
        else:
            x = max(pts[idx - 1][0] + margin, min(pts[idx + 1][0] - margin, x))
        pts[idx] = [x, y]
        self._redraw()
        self.curveChanged.emit()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging_idx = None
        super().mouseReleaseEvent(ev)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _CDILoadWorker(QThread):
    """Load CL/CR diffraction data and optional pixel mask from HDF5."""

    # (amp_cl, amp_cr, signed_magnetic_target, pixel_mask or None)
    finished = pyqtSignal(object, object, object, object)
    error    = pyqtSignal(str)

    def __init__(
        self,
        cl_entry: tuple,           # (file_path, dataset_path)
        cr_entry: tuple,           # (file_path, dataset_path)
        mask_entry: Optional[tuple],
        input_mode: str,           # "intensity" | "amplitude"
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cl   = cl_entry
        self._cr   = cr_entry
        self._mask = mask_entry
        self._mode = input_mode

    @staticmethod
    def _read(file_path: str, ds_path: str) -> np.ndarray:
        import h5py
        with h5py.File(file_path, "r") as f:
            return np.squeeze(np.array(f[ds_path])).astype(np.float64)

    def run(self) -> None:
        try:
            cl_raw = self._read(*self._cl)
            cr_raw = self._read(*self._cr)
            for label, arr in (("CL", cl_raw), ("CR", cr_raw)):
                if arr.ndim != 2:
                    raise ValueError(
                        f"{label} must be 2D after squeeze; got shape {arr.shape}"
                    )
            if cl_raw.shape != cr_raw.shape:
                raise ValueError(
                    f"CL shape {cl_raw.shape} != CR shape {cr_raw.shape}"
                )
            if self._mode == "intensity":
                diff = cl_raw - cr_raw
                signed_mag = np.sign(diff) * np.sqrt(np.abs(diff))
                cl = np.sqrt(np.maximum(cl_raw, 0.0))
                cr = np.sqrt(np.maximum(cr_raw, 0.0))
            else:
                signed_mag = cl_raw - cr_raw
                cl = cl_raw
                cr = cr_raw
            mask = None
            if self._mask is not None:
                mask = self._read(*self._mask).astype(np.float64)
                if mask.shape != cl.shape:
                    raise ValueError(
                        f"Pixel mask shape {mask.shape} != diffraction {cl.shape}"
                    )
            self.finished.emit(cl, cr, signed_mag, mask)
        except Exception as exc:
            self.error.emit(str(exc))


class _CDIReconWorker(QThread):
    """Run CDI phase retrieval in a background thread.

    Emits ``progress`` every ``emit_every`` iterations for live UI updates.
    Supports floating pixel mask (bad pixels float freely during the
    Fourier projection step).
    """

    # (restart_idx, global_iter, fourier_error, current_psi_copy)
    progress     = pyqtSignal(int, int, float, object)
    restart_done = pyqtSignal(int, float)   # (restart_idx, final_error)
    finished     = pyqtSignal(object, list) # (best_obj, error_list)
    error        = pyqtSignal(str)

    def __init__(
        self,
        amp_meas: np.ndarray,
        support_init: np.ndarray,
        algo_steps: list,          # [(algo_str, n_iters), ...]
        beta: float,
        restarts: int,
        sw_enabled: bool,
        sw_sigma: float,
        sw_threshold: float,
        sw_every: int,
        floating_mask: Optional[np.ndarray] = None,
        initial_obj: Optional[np.ndarray] = None,
        feature_constraints: Optional[list] = None,  # [(bool_mask, complex_array), ...]
        emit_every: int = 10,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._amp          = np.abs(amp_meas).copy()  # CDI requires non-negative amplitude
        self._support_init = support_init.copy()
        self._algo_steps   = list(algo_steps)
        self._beta         = float(beta)
        self._restarts     = int(restarts)
        self._sw_enabled   = bool(sw_enabled)
        self._sw_sigma     = float(sw_sigma)
        self._sw_threshold = float(sw_threshold)
        self._sw_every     = int(sw_every)
        self._floating     = None if floating_mask is None \
            else floating_mask.astype(bool).copy()
        self._init_obj     = None if initial_obj is None \
            else initial_obj.astype(np.complex128).copy()
        # Feature constraints: region C — known-value pixels (e.g. Pt pillars).
        # Each entry is (bool_mask, complex_array); these pixels are locked to their
        # reference values after every real-space projection, overriding the HIO/ER update.
        self._feat = [] if not feature_constraints else [
            (m.astype(bool), v.astype(np.complex128))
            for m, v in feature_constraints
        ]
        self._emit_every   = max(1, int(emit_every))
        self._stop         = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            best_obj, best_errs = _run_single_target_cdi(
                self._amp,
                self._support_init,
                self._algo_steps,
                self._beta,
                self._restarts,
                sw_enabled=self._sw_enabled,
                sw_sigma=self._sw_sigma,
                sw_threshold=self._sw_threshold,
                sw_every=self._sw_every,
                floating=self._floating,
                init_obj=self._init_obj,
                feat=self._feat,
                emit_every=self._emit_every,
                on_progress=lambda r, g, e, psi: self.progress.emit(r, g, e, psi),
                on_restart_done=lambda r, e: self.restart_done.emit(r, e),
                is_stopped=lambda: self._stop,
            )
            self.finished.emit(best_obj, best_errs)

        except Exception as exc:
            self.error.emit(str(exc))


class _CDIMultiTargetWorker(QThread):
    """Multi-target CDI pipeline: run targets sequentially with phase inheritance.

    Each target specifies its own Fourier amplitude, support mask, algorithm
    steps, and optional feature constraints.  The complex field ``psi`` is
    passed from one target to the next so phase information accumulates.
    Restarts re-run the entire pipeline from the initial guess.
    """

    progress     = pyqtSignal(int, int, float, object)  # (restart_idx, global_iter, error, psi)
    restart_done = pyqtSignal(int, float)               # (restart_idx, final_error)
    finished     = pyqtSignal(object, list, object)     # (best_obj, error_list, source_results)
    error        = pyqtSignal(str)

    def __init__(
        self,
        targets: list,                   # [{'amp', 'support', 'steps', 'feat', 'beta_schedule', 'bg_sub'}, ...]
        beta: float,
        restarts: int,
        sw_enabled: bool,
        sw_sigma: float,
        sw_threshold: float,
        sw_every: int,
        floating_mask:        Optional[np.ndarray] = None,
        initial_obj:          Optional[np.ndarray] = None,
        emit_every:           int  = 10,
        scale_initial_guess:  bool = False,
        scale_inherited_phase: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._targets = [
            {
                'kind':          t.get('kind', 'target'),
                'operation':     t.get('operation', 'none'),
                'data_source':   t.get('data_source'),
                'amp':           None if t.get('amp') is None else np.abs(t['amp']).astype(np.float64),
                'support':       None if t.get('support') is None else np.asarray(t['support'], dtype=np.float64),
                'steps':         list(t.get('steps', [])),
                'feat':          [(m.astype(bool), v.astype(np.complex128))
                                  for m, v in t.get('feat', [])],
                'beta_schedule': t.get('beta_schedule', 'constant'),
                'bg_sub':        bool(t.get('bg_sub', False)),
            }
            for t in targets
        ]
        self._beta                  = float(beta)
        self._restarts              = int(restarts)
        self._sw_enabled            = bool(sw_enabled)
        self._sw_sigma              = float(sw_sigma)
        self._sw_threshold          = float(sw_threshold)
        self._sw_every              = int(sw_every)
        self._floating              = None if floating_mask is None \
            else floating_mask.astype(bool).copy()
        self._init_obj              = None if initial_obj is None \
            else initial_obj.astype(np.complex128).copy()
        self._emit_every            = max(1, int(emit_every))
        self._scale_initial_guess   = bool(scale_initial_guess)
        self._scale_inherited_phase = bool(scale_inherited_phase)
        self._stop                  = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            best_obj, best_errs, best_sources = _run_multi_target_cdi(
                self._targets,
                self._beta,
                self._restarts,
                sw_enabled=self._sw_enabled,
                sw_sigma=self._sw_sigma,
                sw_threshold=self._sw_threshold,
                sw_every=self._sw_every,
                floating=self._floating,
                init_obj=self._init_obj,
                emit_every=self._emit_every,
                scale_initial_guess=self._scale_initial_guess,
                scale_inherited_phase=self._scale_inherited_phase,
                on_progress=lambda r, g, e, psi: self.progress.emit(r, g, e, psi),
                on_restart_done=lambda r, e: self.restart_done.emit(r, e),
                is_stopped=lambda: self._stop,
            )
            self.finished.emit(best_obj, best_errs, best_sources)

        except Exception as exc:
            self.error.emit(str(exc))


class _CDISequentialReconWorker(QThread):
    """Run configurable tutorial/mine CDI stages.

    This keeps the Classic CL->CR math path, but takes its stages from the
    same target/step pipeline used by the main reconstruction UI.
    """

    # (stage_label, done, total, error, current_complex)
    progress = pyqtSignal(str, int, int, float, object)
    finished = pyqtSignal(object, object, object, object, list, list)
    stopped  = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(
        self,
        cl_intensity: np.ndarray,
        cr_intensity: np.ndarray,
        pixel_mask: Optional[np.ndarray],
        support: np.ndarray,
        pipeline: Optional[list] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cl_raw = np.asarray(cl_intensity, dtype=np.float64).copy()
        self._cr_raw = np.asarray(cr_intensity, dtype=np.float64).copy()
        self._pixel_mask = (
            np.zeros(self._cl_raw.shape, dtype=bool)
            if pixel_mask is None else np.asarray(pixel_mask, dtype=bool).copy()
        )
        self._support = np.asarray(support, dtype=np.float64).copy()
        self._pipeline = list(pipeline or [
            {'data_source': 'CL', 'steps': [('mine', 700)]},
            {'data_source': 'CL', 'steps': [('mine', 50)]},
            {'data_source': 'CR', 'steps': [('mine', 50)]},
        ])
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            (retrieved_cl, retrieved_cr, diff, recon,
             errs_cl, errs_cr, stopped) = _run_sequential_cdi(
                self._cl_raw,
                self._cr_raw,
                self._pixel_mask,
                self._support,
                self._pipeline,
                on_progress=lambda s, d, t, e, c: self.progress.emit(s, d, t, e, c),
                is_stopped=lambda: self._stop,
            )
            if stopped:
                self.stopped.emit()
                return
            self.finished.emit(
                retrieved_cl, retrieved_cr, diff, recon, errs_cl, errs_cr,
            )
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main tool window
# ---------------------------------------------------------------------------

class CDIReconstructionTool(QMainWindow):
    """Unified FTH + CDI Phase Retrieval Reconstruction Tool.

    The first two tabs are taken from an embedded FTHReconstructionTool so the
    user can align data and compute FTH in the same window, then use the FTH
    image to place support shapes for CDI.
    """

    def __init__(
        self,
        parent=None,
        opened_files: tuple = (),
        dataset_full_keys_2d: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CDI Reconstruction Tool")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1400, 900)

        self._opened_files         = opened_files or ()
        self._dataset_full_keys_2d = dataset_full_keys_2d or []

        # --- data state ---
        self._amp_cl:   Optional[np.ndarray] = None
        self._amp_cr:   Optional[np.ndarray] = None
        self._int_cl:   Optional[np.ndarray] = None
        self._int_cr:   Optional[np.ndarray] = None
        self._single_dataset_mode: bool = False
        self._amp_meas: Optional[np.ndarray] = None
        self._pixel_mask:      Optional[np.ndarray] = None
        self._bad_pixel_mask_source: Optional[np.ndarray] = None
        self._bad_pixel_mask_source_meta: Optional[dict] = None
        self._bad_pixel_mask: Optional[np.ndarray] = None
        self._bad_pixel_shift_y: int = 0
        self._bad_pixel_shift_x: int = 0
        # --- mask editor state ---
        self._mask_edit_mode: str = "none"
        self._last_hist_shape: Optional[tuple] = None
        self._pan_last_scene_pos = None   # used for right-drag pan in brush/eraser mode
        self._support:         Optional[np.ndarray] = None
        self._support_source_mask: Optional[np.ndarray] = None
        self._result_obj:      Optional[np.ndarray] = None
        self._result_cl:       Optional[np.ndarray] = None
        self._result_cr:       Optional[np.ndarray] = None
        self._result_diff:     Optional[np.ndarray] = None
        self._result_errs: list[float] = []
        self._result_errs_cr: list[float] = []

        # --- mask groups: index 0 = primary support, 1+ = feature masks ---
        self._mask_groups: list[dict] = []   # [{'name': str, 'entries': list, 'source_masks': list}]
        self._entries: list[_SupportShapeEntry] = []
        self._active_group_idx: int = -1
        self._picking_center: bool = False

        # --- pipeline targets: list of dicts tracking UI cards + state ---
        self._pipeline_targets: list[dict] = []
        self._total_iters_per_restart: int = 1

        # --- live error accumulation across restarts ---
        self._live_errs: list[float] = []

        # --- workers ---
        self._load_worker:  Optional[_CDILoadWorker]           = None
        self._recon_worker: Optional[_CDIMultiTargetWorker]    = None
        self._seq_worker:   Optional[_CDISequentialReconWorker] = None

        # --- embedded FTH tool (provides Alignment + Filter/FTH tabs) ---
        self._fth_tool: Optional[FTHReconstructionTool] = None

        self._build_ui()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        self.menuBar().setVisible(False)

        central = QWidget()
        root    = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet(
            "color:#aaa; font-size:11px; padding:3px 6px; "
            "border-top:1px solid #444; background:#1a1a1a;"
        )
        self._status_label.setFixedHeight(22)

        # Instantiate FTH tool and steal its first two tabs
        self._fth_tool = FTHReconstructionTool(
            parent=self,
            opened_files=self._opened_files,
            dataset_full_keys_2d=self._dataset_full_keys_2d,
        )
        self._fth_tool._set_status = self._set_status  # route FTH status here

        fth_alignment_tab = self._take_fth_tab(0)   # "Alignment"
        fth_filter_tab    = self._take_fth_tab(0)   # "Filter & FTH" (now index 0)
        self._install_bad_pixel_mask_controls(fth_alignment_tab)
        self._install_support_circle_controls(fth_filter_tab)

        self._tabs = QTabWidget()
        self._tabs.addTab(fth_alignment_tab, "Alignment")
        self._tabs.addTab(fth_filter_tab,    "Filter && FTH / Support")
        self._tabs.addTab(self._build_tab_cdi(),     "CDI Reconstruction")
        self._tabs.addTab(self._build_tab_results(), "Summary && Save")

        root.addWidget(self._tabs, stretch=1)
        root.addWidget(self._status_label)
        self.setCentralWidget(central)
        self._populate_dataset_combos()

    def _take_fth_tab(self, index: int) -> QWidget:
        widget = self._fth_tool._tabs.widget(index)
        self._fth_tool._tabs.removeTab(index)
        widget.setParent(None)
        return widget

    def _install_bad_pixel_mask_controls(self, fth_alignment_tab: QWidget) -> None:
        """Inject bad-pixel mask editor and overlays into the Alignment page."""
        # --- permanent mask overlay (red) ---
        self._bad_pixel_mask_img = pg.ImageItem(axisOrder="row-major")
        self._bad_pixel_mask_img.setZValue(25)
        self._bad_pixel_mask_img.setVisible(False)
        self._fth_tool._t1_main_plot.addItem(self._bad_pixel_mask_img)

        # --- threshold preview overlay (orange) ---
        self._threshold_preview_img = pg.ImageItem(axisOrder="row-major")
        self._threshold_preview_img.setZValue(24)
        self._threshold_preview_img.setVisible(False)
        self._fth_tool._t1_main_plot.addItem(self._threshold_preview_img)

        self._fth_tool._t1_display_updated_callback = self._on_alignment_display_updated

        # Hide FTH-only groups not needed in CDI workflow
        self._hide_alignment_tab_extras(fth_alignment_tab)

        splitter = fth_alignment_tab.layout().itemAt(0).widget()
        lay = splitter.widget(0).widget().layout()

        # ================================================================
        g_bpm = QGroupBox("Bad Pixel Mask Editor")
        g_bpm_lay = QVBoxLayout(g_bpm)
        g_bpm_lay.setSpacing(5)

        # --- mode toolbar (Brush / Eraser — toggle on/off) ---
        mode_row = QHBoxLayout()
        self._mask_mode_brush_btn = QPushButton("Brush")
        self._mask_mode_brush_btn.setCheckable(True)
        self._mask_mode_brush_btn.setChecked(False)
        self._mask_mode_eraser_btn = QPushButton("Eraser")
        self._mask_mode_eraser_btn.setCheckable(True)
        self._mask_mode_eraser_btn.setChecked(False)
        for btn in (self._mask_mode_brush_btn, self._mask_mode_eraser_btn):
            btn.setMinimumHeight(26)
            mode_row.addWidget(btn)
        self._mask_mode_brush_btn.clicked.connect(
            lambda checked: self._set_mask_edit_mode("brush" if checked else "none")
        )
        self._mask_mode_eraser_btn.clicked.connect(
            lambda checked: self._set_mask_edit_mode("eraser" if checked else "none")
        )
        g_bpm_lay.addLayout(mode_row)

        # --- Brush / Eraser panel ---
        self._brush_panel = QWidget()
        bp_form = QFormLayout(self._brush_panel)
        bp_form.setContentsMargins(0, 4, 0, 0)
        self._brush_radius_spin = QSpinBox()
        self._brush_radius_spin.setRange(1, 200)
        self._brush_radius_spin.setValue(5)
        self._brush_radius_spin.setSuffix(" px")
        bp_form.addRow("Brush radius:", self._brush_radius_spin)
        self._brush_panel.setVisible(False)
        g_bpm_lay.addWidget(self._brush_panel)

        # --- Load / Save / Clear ---
        io_row = QHBoxLayout()
        self._load_bad_pixel_mask_btn = QPushButton("Load .npy")
        self._load_bad_pixel_mask_btn.clicked.connect(self._load_bad_pixel_mask_npy)
        self._save_bad_pixel_mask_btn = QPushButton("Save .npy")
        self._save_bad_pixel_mask_btn.clicked.connect(self._save_bad_pixel_mask_npy)
        self._clear_bad_pixel_mask_btn = QPushButton("Clear")
        self._clear_bad_pixel_mask_btn.clicked.connect(self._clear_bad_pixel_mask)
        for btn in (self._load_bad_pixel_mask_btn, self._save_bad_pixel_mask_btn, self._clear_bad_pixel_mask_btn):
            btn.setMinimumHeight(26)
            io_row.addWidget(btn)
        g_bpm_lay.addLayout(io_row)

        # --- Shift (for aligning an externally loaded mask) ---
        shift_form = QFormLayout()
        self._bad_pixel_shift_x_spin = QSpinBox()
        self._bad_pixel_shift_x_spin.setRange(-9999, 9999)
        self._bad_pixel_shift_x_spin.setSuffix(" px")
        self._bad_pixel_shift_x_spin.valueChanged.connect(self._on_bad_pixel_shift_changed)
        self._bad_pixel_shift_y_spin = QSpinBox()
        self._bad_pixel_shift_y_spin.setRange(-9999, 9999)
        self._bad_pixel_shift_y_spin.setSuffix(" px")
        self._bad_pixel_shift_y_spin.valueChanged.connect(self._on_bad_pixel_shift_changed)
        shift_form.addRow("X shift:", self._bad_pixel_shift_x_spin)
        shift_form.addRow("Y shift:", self._bad_pixel_shift_y_spin)
        g_bpm_lay.addLayout(shift_form)

        # --- Status label ---
        self._bad_pixel_mask_label = QLabel("No mask")
        self._bad_pixel_mask_label.setWordWrap(True)
        self._bad_pixel_mask_label.setStyleSheet("color:#aaa; font-size:9pt;")
        g_bpm_lay.addWidget(self._bad_pixel_mask_label)

        lay.addWidget(g_bpm)

        # Connect mouse events for brush/eraser painting
        self._fth_tool._t1_main_plot.scene().sigMouseMoved.connect(self._on_alignment_mouse_moved_paint)
        self._fth_tool._t1_main_plot.scene().sigMouseClicked.connect(self._on_alignment_mouse_clicked_paint)

        # Patch ViewBox so scroll-wheel zoom works even when mouse-drag is
        # disabled in brush/eraser mode.  The patch temporarily re-enables the
        # ViewBox mouse state for the duration of the wheel event only.
        _vb  = self._fth_tool._t1_main_plot.getViewBox()
        _cls_wheel = type(_vb).wheelEvent

        def _wheel_always_zoom(ev, axis=None, _vb=_vb, _orig=_cls_wheel):
            was = list(_vb.state['mouseEnabled'])
            if not any(was):
                _vb.state['mouseEnabled'] = [True, True]
                _orig(_vb, ev, axis)
                _vb.state['mouseEnabled'] = was
            else:
                _orig(_vb, ev, axis)

        _vb.wheelEvent = _wheel_always_zoom

    def _install_support_circle_controls(self, fth_filter_tab: QWidget) -> None:
        """Inject circle-ROI controls into the FTH Filter/FTH tab's left panel.

        Hides the FTH ROI-selection box and instead overlays a single support
        circle on the FTH real-space image.
        """
        # Hide FTH ROI selection UI (not needed for CDI workflow)
        from PyQt6.QtWidgets import QGroupBox
        for group in fth_filter_tab.findChildren(QGroupBox):
            if group.title().startswith("ROI Selection"):
                group.hide()
        for attr in ("_roi1_rect", "_roi2_rect", "_roi3_rect", "_roi4_rect"):
            rect = getattr(self._fth_tool, attr, None)
            if rect is not None:
                rect.setVisible(False)

        # Attach the support ROI overlay to the FTH real-space image panel
        self._supp_plot = self._fth_tool._t3_fth_plot
        self._supp_img  = self._fth_tool._t3_fth_img
        self._supp_mask_img = pg.ImageItem(axisOrder="row-major")
        self._supp_mask_img.setZValue(20)
        self._supp_mask_img.setVisible(False)
        self._supp_plot.addItem(self._supp_mask_img)
        self._supp_plot.scene().sigMouseClicked.connect(self._on_support_image_clicked)

        self._fth_tool._t3_fth_display_updated_callback = self._refresh_support_mask_overlay

        # Inject spinbox controls into the left scroll panel of the FTH tab
        splitter = fth_filter_tab.layout().itemAt(0).widget()
        lay = splitter.widget(0).widget().layout()

        g_circ = QGroupBox("CDI Support Shapes")
        g_circ.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        g_lay  = QVBoxLayout(g_circ)
        g_lay.setSpacing(4)
        g_lay.setContentsMargins(6, 6, 6, 6)
        g_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Mask list — each row = one named mask group
        self._mask_list = QListWidget()
        self._mask_list.setFixedHeight(135)
        self._mask_list.setToolTip(
            "Support (first) = primary CDI constraint\nFeature N = additional feature masks"
        )
        self._mask_list.currentRowChanged.connect(self._on_mask_list_row_changed)
        g_lay.addWidget(self._mask_list)

        mask_mgr_row = QHBoxLayout()
        mask_mgr_row.setSpacing(3)
        self._add_mask_btn = QPushButton("Add Mask")
        self._add_mask_btn.clicked.connect(self._add_mask_group)
        self._remove_mask_btn = QPushButton("Remove")
        self._remove_mask_btn.clicked.connect(self._remove_selected_mask)
        self._clear_masks_btn = QPushButton("Clear All")
        self._clear_masks_btn.clicked.connect(self._clear_shapes)
        mask_mgr_row.addWidget(self._add_mask_btn)
        mask_mgr_row.addWidget(self._remove_mask_btn)
        mask_mgr_row.addWidget(self._clear_masks_btn)
        g_lay.addLayout(mask_mgr_row)

        shape_row = QHBoxLayout()
        shape_row.setSpacing(3)
        self._pick_center_btn = QPushButton("Pick on FTH")
        self._pick_center_btn.setCheckable(True)
        self._pick_center_btn.toggled.connect(self._on_pick_center_toggled)
        self._add_circle_btn = QPushButton("+ Circle")
        self._add_circle_btn.clicked.connect(lambda: self._add_circle())
        self._add_rect_btn = QPushButton("+ Rect")
        self._add_rect_btn.clicked.connect(lambda: self._add_rect())
        shape_row.addWidget(self._pick_center_btn)
        shape_row.addWidget(self._add_circle_btn)
        shape_row.addWidget(self._add_rect_btn)
        g_lay.addLayout(shape_row)

        io_row = QHBoxLayout()
        io_row.setSpacing(3)
        save_mask_btn = QPushButton("Save (.npy)")
        save_mask_btn.clicked.connect(self._save_support_mask)
        load_mask_btn = QPushButton("Load (.npy)")
        load_mask_btn.clicked.connect(self._load_support_mask_npy)
        io_row.addWidget(save_mask_btn)
        io_row.addWidget(load_mask_btn)
        g_lay.addLayout(io_row)

        shift_form = QFormLayout()
        self._support_shift_x_spin = QSpinBox()
        self._support_shift_x_spin.setRange(-9999, 9999)
        self._support_shift_x_spin.setSuffix(" px")
        self._support_shift_x_spin.valueChanged.connect(self._on_support_shift_changed)
        self._support_shift_y_spin = QSpinBox()
        self._support_shift_y_spin.setRange(-9999, 9999)
        self._support_shift_y_spin.setSuffix(" px")
        self._support_shift_y_spin.valueChanged.connect(self._on_support_shift_changed)
        shift_form.addRow("X shift:", self._support_shift_x_spin)
        shift_form.addRow("Y shift:", self._support_shift_y_spin)
        g_lay.addLayout(shift_form)

        lay.addWidget(g_circ)

        # Hide differential filter section — HERALDO slit-based processing does
        # not apply to CDI reconstruction.
        fth = self._fth_tool
        for g in fth_filter_tab.findChildren(QGroupBox):
            if g.title() == "Differential Filter":
                g.setVisible(False)
                break
        fth._btn_compute_fth.setVisible(False)

        # Add CDI compute controls directly above the FTH Display GroupBox.
        lay.removeWidget(fth._balance_widget)
        fth._balance_widget.setVisible(True)
        cdi_fft_btn = QPushButton("Compute FFT")
        cdi_fft_btn.setMinimumHeight(36)
        cdi_fft_btn.clicked.connect(self._cdi_compute_fft)
        insert_idx = lay.count()  # fallback: end of layout
        for g in fth_filter_tab.findChildren(QGroupBox):
            if g.title() == "FTH Display":
                for i in range(lay.count()):
                    item = lay.itemAt(i)
                    if item is not None and item.widget() is g:
                        insert_idx = i
                        break
                break
        lay.insertWidget(insert_idx, cdi_fft_btn)
        lay.insertWidget(insert_idx + 1, fth._balance_widget)

        # Remove the filtered-hologram panel — CDI only needs the FFT result
        fth._t3_glw.removeItem(fth._t3_holo_plot)
        fth._t3_glw.removeItem(fth._t3_holo_hist)

    # ------------------------------------------------------------------
    # Shared layout helpers (mirrors FTH tool conventions)

    def _make_scroll_ctrl(self) -> tuple:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(620)
        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.setSpacing(5)
        lay.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(inner)
        return scroll, lay

    @staticmethod
    def _make_splitter() -> QSplitter:
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setChildrenCollapsible(False)
        return sp

    @staticmethod
    def _make_pg_widget() -> pg.GraphicsLayoutWidget:
        w = pg.GraphicsLayoutWidget()
        w.setBackground("k")
        w.setMinimumSize(10, 10)
        return w

    @staticmethod
    def _add_image_panel(
        glw: pg.GraphicsLayoutWidget,
        row: int, col: int, title: str = "",
        rowspan: int = 1, colspan: int = 1,
    ) -> tuple:
        p = glw.addPlot(row=row, col=col, title=title,
                        rowspan=rowspan, colspan=colspan)
        p.setMenuEnabled(False)
        p.setAspectLocked(True)
        p.invertY(True)
        for ax in ("left", "bottom", "right", "top"):
            p.hideAxis(ax)
        img = pg.ImageItem(axisOrder="row-major")
        p.addItem(img)
        return p, img

    @staticmethod
    def _cmap_combo(default: str = "gray") -> QComboBox:
        cb = QComboBox()
        cb.addItems(CDI_COLORMAPS)
        cb.setCurrentText(default if default in CDI_COLORMAPS else "gray")
        return cb

    # ==================================================================
    # Tab 3 — CDI Reconstruction
    # ==================================================================

    def _build_tab_cdi(self) -> QWidget:
        tab      = QWidget()
        main_lay = QHBoxLayout(tab)
        main_lay.setContentsMargins(0, 0, 0, 0)
        splitter = self._make_splitter()
        main_lay.addWidget(splitter)

        scroll, lay = self._make_scroll_ctrl()

        # --- Algorithm sequence (multi-target pipeline) ---
        g_algo = QGroupBox("Algorithm Sequence")
        g_algo_lay = QVBoxLayout(g_algo)
        g_algo_lay.setSpacing(4)
        g_algo_lay.setContentsMargins(6, 6, 6, 6)

        # Scrollable area that holds target cards
        pipeline_scroll = QScrollArea()
        pipeline_scroll.setWidgetResizable(True)
        pipeline_scroll.setMinimumHeight(360)
        pipeline_scroll.setMaximumHeight(900)
        pipeline_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pipeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        pipeline_inner = QWidget()
        self._pipeline_lay = QVBoxLayout(pipeline_inner)
        # Left-align so equalized fixed-width cards hug the left edge instead of
        # stretching across (or centering in) the panel.
        self._pipeline_lay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._pipeline_lay.setSpacing(4)
        self._pipeline_lay.setContentsMargins(2, 2, 4, 2)
        pipeline_scroll.setWidget(pipeline_inner)
        g_algo_lay.addWidget(pipeline_scroll)

        add_target_btn = QPushButton("+ Add Target")
        add_target_btn.setMinimumHeight(26)
        add_target_btn.clicked.connect(lambda: self._add_pipeline_target())
        add_interval_btn = QPushButton("+ Interval Step")
        add_interval_btn.setMinimumHeight(26)
        add_interval_btn.clicked.connect(lambda: self._add_interval_step())
        add_row = QHBoxLayout()
        add_row.addWidget(add_target_btn)
        add_row.addWidget(add_interval_btn)
        g_algo_lay.addLayout(add_row)

        # --- Parameters ---
        g_par = QGroupBox("Parameters")
        fl_p  = QFormLayout(g_par)
        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(0.01, 1.0)
        self._beta_spin.setSingleStep(0.05)
        self._beta_spin.setValue(0.9)
        self._beta_spin.setDecimals(2)
        fl_p.addRow("beta:", self._beta_spin)
        self._restarts_spin = QSpinBox()
        self._restarts_spin.setRange(1, 99)
        self._restarts_spin.setValue(3)
        fl_p.addRow("Restarts:", self._restarts_spin)

        # How to initialize the reconstruction (applies to target 1; later targets inherit phase)
        self._initial_guess_combo = QComboBox()
        self._initial_guess_combo.addItems([
            "Support FFT phase",
            "Random phase",
            "Previous CDI reconstruction",
        ])
        self._initial_guess_combo.setToolTip(
            "Support FFT: use the phase of fftshift(ifft2(ifftshift(support mask))).\n"
            "Random: independent random initializations (ensemble).\n"
            "Previous: re-use the last CDI result (refine).\n"
            "Targets 2+ always inherit phase from the preceding target."
        )
        fl_p.addRow("Initial guess:", self._initial_guess_combo)
        self._scale_init_chk = QCheckBox("match data amplitude")
        self._scale_init_chk.setChecked(False)
        self._scale_init_chk.setToolTip(
            "Linear-regression scale the initial guess so |ψ₀| matches the target amplitude.\n"
            "Applies only to the first cold-start target when using Support FFT phase init."
        )
        self._scale_inherited_chk = QCheckBox("normalize between targets")
        self._scale_inherited_chk.setChecked(False)
        self._scale_inherited_chk.setToolTip(
            "When adding a new target, default its transition to normalize.\n"
            "Existing targets use their own transition dropdown."
        )
        lay.addWidget(g_par)
        lay.addWidget(g_algo, stretch=1)

        # --- Shrinkwrap ---
        self._sw_group = QGroupBox("Shrinkwrap")
        self._sw_group.setCheckable(True)
        self._sw_group.setChecked(True)
        fl_sw = QFormLayout(self._sw_group)
        self._sw_sigma_spin = QDoubleSpinBox()
        self._sw_sigma_spin.setRange(0.5, 30.0)
        self._sw_sigma_spin.setValue(3.0)
        self._sw_sigma_spin.setSingleStep(0.5)
        self._sw_sigma_spin.setSuffix(" px")
        fl_sw.addRow("Blur sigma:", self._sw_sigma_spin)
        self._sw_thresh_spin = QDoubleSpinBox()
        self._sw_thresh_spin.setRange(0.01, 0.99)
        self._sw_thresh_spin.setValue(0.18)
        self._sw_thresh_spin.setSingleStep(0.01)
        self._sw_thresh_spin.setDecimals(2)
        fl_sw.addRow("Threshold:", self._sw_thresh_spin)
        self._sw_every_spin = QSpinBox()
        self._sw_every_spin.setRange(1, 500)
        self._sw_every_spin.setValue(20)
        fl_sw.addRow("Update every N iters:", self._sw_every_spin)
        lay.addWidget(self._sw_group)

        # --- Run / Stop ---
        g_run = QGroupBox("Run")
        fl_r  = QFormLayout(g_run)
        self._classic_btn = QPushButton("Load example sequence")
        self._classic_btn.setMinimumHeight(28)
        self._classic_btn.setToolTip(
            "Replace the current target list with the configurable Classic "
            "CL(700 mine/arctan) → CL-refine(50 mine/const) → CR(50 mine/const) preset."
        )
        self._classic_btn.clicked.connect(self._load_classic_pipeline_preset)
        fl_r.addRow(self._classic_btn)
        run_row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setMinimumHeight(32)
        self._start_btn.clicked.connect(self._start_reconstruction)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setMinimumHeight(32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_reconstruction)
        run_row.addWidget(self._start_btn)
        run_row.addWidget(self._stop_btn)
        fl_r.addRow(run_row)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        fl_r.addRow("Progress:", self._progress_bar)
        self._progress_label = QLabel("--")
        self._progress_label.setWordWrap(True)
        fl_r.addRow(self._progress_label)
        lay.addWidget(g_run)

        # Right — live amplitude preview + error curve
        right_glw = pg.GraphicsLayoutWidget()
        right_glw.setBackground("k")
        self._live_amp_plot, self._live_amp_img = self._add_image_panel(
            right_glw, 0, 0, "Live Amplitude"
        )
        self._live_err_plot = right_glw.addPlot(row=1, col=0, title="Fourier Error")
        self._live_err_plot.setLabel("bottom", "Iteration")
        self._live_err_plot.setLabel("left", "Error")
        self._live_err_curve = self._live_err_plot.plot(pen=pg.mkPen("r", width=1.5))
        self._live_err_plot.showGrid(x=True, y=True, alpha=0.3)

        splitter.addWidget(scroll)
        splitter.addWidget(right_glw)
        splitter.setSizes([420, 980])

        self._add_pipeline_target("CL+CR", "support", [("raar", 300), ("er", 50)])
        return tab

    # ==================================================================
    # Tab 4 — Summary & Save
    # ==================================================================

    def _build_tab_results(self) -> QWidget:
        tab      = QWidget()
        splitter = self._make_splitter()
        QHBoxLayout(tab).addWidget(splitter)
        tab.layout().setContentsMargins(0, 0, 0, 0)

        scroll, lay = self._make_scroll_ctrl()

        g_disp = QGroupBox("Display Options")
        fl_d   = QFormLayout(g_disp)
        self._res_channel_combo = QComboBox()
        self._res_channel_combo.addItems([
            "Final result",
            "CL",
            "CR",
            "CL − CR",
        ])
        self._res_channel_combo.currentTextChanged.connect(self._refresh_results_display)
        fl_d.addRow("Channel:", self._res_channel_combo)
        self._res_view_combo = QComboBox()
        self._res_view_combo.addItems([
            "Abs amplitude",
            "Real part",
            "Imaginary part",
            "Phase",
        ])
        self._res_view_combo.currentTextChanged.connect(self._refresh_results_display)
        fl_d.addRow("Result view:", self._res_view_combo)
        self._res_amp_cmap = self._cmap_combo("gray")
        fl_d.addRow("Result cmap:", self._res_amp_cmap)
        self._res_amp_cmap.currentTextChanged.connect(self._refresh_results_display)
        self._res_phase_cmap = self._cmap_combo("CET-D9")
        fl_d.addRow("Phase cmap:", self._res_phase_cmap)
        self._res_phase_cmap.currentTextChanged.connect(self._refresh_results_display)
        lay.addWidget(g_disp)

        g_exp = QGroupBox("Export")
        fl_e  = QFormLayout(g_exp)

        roi_row = QHBoxLayout()
        self._res_roi_btn = QPushButton("ROI")
        self._res_roi_btn.setCheckable(True)
        self._res_roi_btn.toggled.connect(self._on_result_roi_toggled)
        self._res_roi_reset_btn = QPushButton("Reset ROI")
        self._res_roi_reset_btn.clicked.connect(self._reset_result_roi)
        roi_row.addWidget(self._res_roi_btn)
        roi_row.addWidget(self._res_roi_reset_btn)
        fl_e.addRow("Crop:", roi_row)

        self._cdi_exp_target_combo = QComboBox()
        self._cdi_exp_target_combo.addItems(["Real", "Imag.", "Phase", "Abs."])
        fl_e.addRow("Image:", self._cdi_exp_target_combo)

        exp_row = QHBoxLayout()
        self._cdi_copy_btn = QPushButton("Copy")
        self._cdi_copy_btn.clicked.connect(self._copy_selected_cdi_component_jpeg)
        self._cdi_save_btn = QPushButton("Save")
        self._cdi_save_btn.clicked.connect(self._save_selected_cdi_component)
        exp_row.addWidget(self._cdi_copy_btn)
        exp_row.addWidget(self._cdi_save_btn)
        fl_e.addRow(exp_row)

        self._cdi_save_all_btn = QPushButton("Save all")
        self._cdi_save_all_btn.clicked.connect(self._save_all_cdi_components)
        fl_e.addRow(self._cdi_save_all_btn)
        lay.addWidget(g_exp)

        lay.addStretch()

        # 2 × 2 image grid
        self._res_glw = self._make_pg_widget()
        self._res_amp_plot,   self._res_amp_img   = self._add_image_panel(
            self._res_glw, 0, 0, "Abs amplitude"
        )
        self._res_roi = pg.RectROI([0, 0], [10, 10], pen=pg.mkPen("y", width=2))
        self._res_roi.setZValue(30)
        self._res_roi.setVisible(False)
        self._res_amp_plot.addItem(self._res_roi)
        self._res_roi_shape: Optional[tuple[int, int]] = None
        self._res_phase_plot, self._res_phase_img = self._add_image_panel(
            self._res_glw, 0, 1, "Phase [rad]"
        )
        self._res_diff_plot,  self._res_diff_img  = self._add_image_panel(
            self._res_glw, 1, 0, "Log |FFT|²  (reconstructed)"
        )
        self._res_err_plot = self._res_glw.addPlot(row=1, col=1, title="Fourier Error")
        self._res_err_plot.setLabel("bottom", "Iteration")
        self._res_err_plot.setLabel("left", "Error")
        self._res_err_curve = self._res_err_plot.plot(pen=pg.mkPen("r", width=1.5))
        self._res_err_plot.showGrid(x=True, y=True, alpha=0.3)

        splitter.addWidget(scroll)
        splitter.addWidget(self._res_glw)
        splitter.setSizes([310, 1090])
        return tab

    # ==================================================================
    # Dataset index — delegate to embedded FTH tool
    # ==================================================================

    def _populate_dataset_combos(self) -> None:
        if self._fth_tool is not None:
            self._fth_tool.refresh_dataset_keys(
                self._dataset_full_keys_2d,
                opened_files=self._opened_files,
            )

    def update_opened_files(
        self,
        opened_files,
        dataset_full_keys_2d: list[str] | None = None,
    ) -> None:
        self._opened_files         = opened_files or ()
        self._dataset_full_keys_2d = dataset_full_keys_2d or []
        self._populate_dataset_combos()

    def add_dataset_to_combo(self, full_path: str, channel: str) -> None:
        """Add a dataset to the embedded FTH CL/CR/Dark selector."""
        if self._fth_tool is None:
            return
        self._fth_tool.add_dataset_to_combo(full_path, channel)
        self.raise_()
        self.activateWindow()

    # ==================================================================
    # Support shape management
    # ==================================================================

    def _shape_hint(self) -> tuple[int, int]:
        """Best-effort image shape for mask creation."""
        if self._amp_meas is not None:
            return self._amp_meas.shape
        fth = self._fth_tool
        for attr in ("_FTH_S1", "_FTH_S2", "_CL_c", "_CR_c"):
            a = getattr(fth, attr, None)
            if a is not None:
                return a.shape
        all_entries = [e for g in self._mask_groups for e in g["entries"]]
        if all_entries:
            max_y = max(e.y + (e.r if e.kind == "circle" else e.b / 2.0) for e in all_entries)
            max_x = max(e.x + (e.r if e.kind == "circle" else e.a / 2.0) for e in all_entries)
            return int(max_y * 1.1) or 256, int(max_x * 1.1) or 256
        for group in self._mask_groups:
            source_masks = self._group_source_masks(group)
            if source_masks:
                return tuple(np.asarray(source_masks[0]).shape)
        return 256, 256

    # ------------------------------------------------------------------
    # Mask group helpers

    _MASK_GROUP_COLORS = ["y", "c", (0, 220, 80), (255, 140, 0), (180, 80, 255), "r"]

    def _group_color(self, idx: int):
        return self._MASK_GROUP_COLORS[idx % len(self._MASK_GROUP_COLORS)]

    def _ensure_active_group(self) -> dict:
        if not self._mask_groups or self._active_group_idx < 0:
            self._add_mask_group()
        return self._mask_groups[self._active_group_idx]

    @staticmethod
    def _group_source_masks(group: dict) -> list[np.ndarray]:
        """Return loaded masks for a group, including legacy single-mask state."""
        masks = []
        legacy = group.get("source_mask")
        if legacy is not None:
            masks.append(legacy)
        masks.extend(group.get("source_masks") or [])
        return masks

    def _add_mask_group(self) -> None:
        idx = len(self._mask_groups)
        name = "Support" if idx == 0 else f"Feature {idx}"
        self._mask_groups.append({
            "name": name,
            "entries": [],
            "source_mask": None,
            "source_masks": [],
            "shift_x": 0,
            "shift_y": 0,
        })
        self._active_group_idx = len(self._mask_groups) - 1
        self._refresh_mask_list()

    def _remove_selected_mask(self) -> None:
        row = self._mask_list.currentRow()
        if row < 0 or row >= len(self._mask_groups):
            return
        group = self._mask_groups.pop(row)
        for entry in group["entries"]:
            self._supp_plot.removeItem(entry.roi)
            if entry in self._entries:
                self._entries.remove(entry)
        if row == 0:
            self._support_source_mask = None
        for i, g in enumerate(self._mask_groups):
            g["name"] = "Support" if i == 0 else f"Feature {i}"
        self._active_group_idx = min(row, len(self._mask_groups) - 1)
        self._refresh_mask_list()
        self._update_mask_overlay()

    def _on_mask_list_row_changed(self, row: int) -> None:
        self._active_group_idx = row
        self._sync_support_shift_controls()

    def _refresh_mask_list(self) -> None:
        self._mask_list.blockSignals(True)
        self._mask_list.clear()
        for g in self._mask_groups:
            n = len(g["entries"])
            loaded_n = len(self._group_source_masks(g))
            parts = [f"{n} shape{'s' if n != 1 else ''}"]
            if loaded_n:
                parts.append(f"{loaded_n} loaded mask{'s' if loaded_n != 1 else ''}")
            shift_x = int(g.get("shift_x", 0) or 0)
            shift_y = int(g.get("shift_y", 0) or 0)
            if shift_x or shift_y:
                parts.append(f"shift X={shift_x}, Y={shift_y}")
            self._mask_list.addItem(f"{g['name']}  ({', '.join(parts)})")
        self._mask_list.blockSignals(False)
        if self._mask_list.count() > 0:
            row = max(0, min(self._active_group_idx, self._mask_list.count() - 1))
            self._mask_list.setCurrentRow(row)
            self._sync_support_shift_controls()
        else:
            self._sync_support_shift_controls()

    def _sync_support_shift_controls(self) -> None:
        if not hasattr(self, "_support_shift_x_spin"):
            return
        group = (
            self._mask_groups[self._active_group_idx]
            if 0 <= self._active_group_idx < len(self._mask_groups)
            else None
        )
        x = int(group.get("shift_x", 0) or 0) if group is not None else 0
        y = int(group.get("shift_y", 0) or 0) if group is not None else 0
        for spin, value in (
            (self._support_shift_x_spin, x),
            (self._support_shift_y_spin, y),
        ):
            spin.blockSignals(True)
            spin.setEnabled(group is not None)
            spin.setValue(value)
            spin.blockSignals(False)

    def _on_support_shift_changed(self, *_args) -> None:
        if not (0 <= self._active_group_idx < len(self._mask_groups)):
            return
        group = self._mask_groups[self._active_group_idx]
        group["shift_x"] = int(self._support_shift_x_spin.value())
        group["shift_y"] = int(self._support_shift_y_spin.value())
        self._refresh_mask_list()
        self._update_mask_overlay()

    # ------------------------------------------------------------------
    # Shape add / remove

    def _add_circle(
        self,
        y: Optional[float] = None,
        x: Optional[float] = None,
        r: float = 50.0,
    ) -> None:
        group = self._ensure_active_group()
        H, W = self._shape_hint()
        cy = y if y is not None else H / 2.0
        cx = x if x is not None else W / 2.0
        color = self._group_color(self._active_group_idx)
        roi = pg.CircleROI(
            pos=(cx - r, cy - r),
            size=(2.0 * r, 2.0 * r),
            pen=pg.mkPen(color, width=2),
            movable=True,
            resizable=True,
        )
        self._supp_plot.addItem(roi)
        entry = _SupportShapeEntry(kind="circle", roi=roi, y=cy, x=cx, a=r)
        group["entries"].append(entry)
        self._entries.append(entry)
        roi.sigRegionChangeFinished.connect(
            lambda _r, e=entry: (e.read_from_roi(), self._update_mask_overlay())
        )
        self._refresh_mask_list()
        self._update_mask_overlay()

    def _add_rect(
        self,
        y: Optional[float] = None,
        x: Optional[float] = None,
        w: float = 100.0,
        h: float = 50.0,
    ) -> None:
        group = self._ensure_active_group()
        H, W = self._shape_hint()
        cy = y if y is not None else H / 2.0
        cx = x if x is not None else W / 2.0
        color = self._group_color(self._active_group_idx)
        roi = pg.RectROI(
            pos=(cx - w / 2.0, cy - h / 2.0),
            size=(w, h),
            pen=pg.mkPen(color, width=2),
            movable=True,
            resizable=True,
            rotatable=True,
        )
        roi.addRotateHandle([1.0, 0.0], [0.5, 0.5])
        self._supp_plot.addItem(roi)
        entry = _SupportShapeEntry(kind="rect", roi=roi, y=cy, x=cx, a=w, b=h, angle=0.0)
        group["entries"].append(entry)
        self._entries.append(entry)
        roi.sigRegionChangeFinished.connect(
            lambda _r, e=entry: (e.read_from_roi(), self._update_mask_overlay())
        )
        self._refresh_mask_list()
        self._update_mask_overlay()

    def _clear_shapes(self) -> None:
        for group in self._mask_groups:
            for entry in group["entries"]:
                self._supp_plot.removeItem(entry.roi)
        self._mask_groups.clear()
        self._entries.clear()
        self._active_group_idx = -1
        self._support = None
        self._support_source_mask = None
        self._refresh_mask_list()
        self._refresh_support_mask_overlay()
        self._update_mask_overlay()

    def _on_pick_center_toggled(self, checked: bool) -> None:
        self._picking_center = checked
        self._pick_center_btn.setText(
            "Click FTH panel…  (ESC to cancel)" if checked else "Pick on FTH"
        )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._picking_center:
            self._pick_center_btn.setChecked(False)
        super().keyPressEvent(event)

    def _on_support_image_clicked(self, ev) -> None:
        if not self._picking_center:
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = self._supp_plot.vb.mapSceneToView(ev.scenePos())
        self._add_circle(y=pos.y(), x=pos.x())
        self._pick_center_btn.setChecked(False)

    def _compute_group_mask(self, group_idx: int) -> Optional[np.ndarray]:
        """Compute binary mask for a single mask group."""
        if group_idx >= len(self._mask_groups):
            return None
        group = self._mask_groups[group_idx]
        entries = group["entries"]
        source_masks = self._group_source_masks(group)
        if not entries and not source_masks:
            return None
        H, W = self._shape_hint()
        Y, X = np.ogrid[:H, :W]
        mask = np.zeros((H, W), dtype=np.uint8)
        for source_mask in source_masks:
            aligned = self._align_support_mask_to_current_fth(source_mask, show_error=False)
            if aligned is None and not entries:
                aligned = np.asarray(source_mask, dtype=np.float32)
            if aligned is not None:
                if tuple(aligned.shape) != (H, W):
                    H, W = aligned.shape
                    Y, X = np.ogrid[:H, :W]
                    mask = np.zeros((H, W), dtype=np.uint8)
                mask |= (np.asarray(aligned) > 0).astype(np.uint8)
        for e in entries:
            if e.kind == "circle":
                shape_mask = (X - e.x) ** 2 + (Y - e.y) ** 2 <= e.r ** 2
            else:
                theta = np.deg2rad(e.angle)
                dx = X - e.x
                dy = Y - e.y
                local_x = dx * np.cos(theta) + dy * np.sin(theta)
                local_y = -dx * np.sin(theta) + dy * np.cos(theta)
                shape_mask = (
                    (np.abs(local_x) <= e.a / 2.0)
                    & (np.abs(local_y) <= e.b / 2.0)
                )
            mask |= shape_mask.astype(np.uint8)
        shift_x = int(group.get("shift_x", 0) or 0)
        shift_y = int(group.get("shift_y", 0) or 0)
        if shift_x or shift_y:
            mask = self._shift_boolean_mask(mask > 0, shift_y, shift_x).astype(np.uint8)
        return mask.astype(np.float32)

    def _compute_support_mask(self) -> Optional[np.ndarray]:
        """Combined mask of ALL groups (used for display overlay)."""
        mask: Optional[np.ndarray] = None
        for idx in range(len(self._mask_groups)):
            group_mask = self._compute_group_mask(idx)
            if group_mask is None:
                continue
            group_arr = np.asarray(group_mask) > 0
            if mask is None or tuple(mask.shape) != tuple(group_arr.shape):
                mask = np.zeros(group_arr.shape, dtype=np.uint8)
            mask |= group_arr.astype(np.uint8)
        if mask is None:
            return None
        return mask.astype(np.float32)

    def _update_mask_overlay(self) -> None:
        mask = self._compute_support_mask()
        if mask is None:
            self._support = None
            self._refresh_support_mask_overlay()
            return
        self._support = mask
        self._support_source_mask = None
        self._refresh_support_mask_overlay()
        # Mask is stored; ROI outlines are already visible, so no filled overlay is needed.

    def _save_support_mask(self) -> None:
        mask = self._compute_support_mask()
        if mask is None and self._support is not None:
            mask = self._support
        if mask is None:
            QMessageBox.warning(self, "No Mask", "No support shapes defined.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Support Mask", "", "NumPy (*.npy)")
        if path:
            np.save(path, mask)
            self._set_status(f"Saved mask → {path}")

    def _load_support_mask_npy(self) -> None:
        row = self._mask_list.currentRow()
        if row < 0 or row >= len(self._mask_groups):
            QMessageBox.warning(
                self,
                "No Mask Selected",
                "Add a mask group and select it in the list before loading a .npy mask.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Support Mask", "", "NumPy (*.npy)")
        if not path:
            return
        try:
            mask = np.squeeze(np.load(path)).astype(np.float32)
            if mask.ndim != 2:
                raise ValueError(f"Support mask must be 2D after squeeze; got shape {mask.shape}")
            target_group = self._mask_groups[row]
            legacy = target_group.get("source_mask")
            if legacy is not None:
                target_group.setdefault("source_masks", []).append(legacy)
                target_group["source_mask"] = None
            target_group.setdefault("source_masks", []).append(mask)
            self._active_group_idx = row
            self._support_source_mask = None
            aligned = self._align_support_mask_to_current_fth(mask, show_error=False)
            support = self._compute_group_mask(0)
            if support is not None:
                self._support = support
            if aligned is not None:
                self._refresh_support_mask_overlay()
                self._refresh_mask_list()
                self._set_status(
                    f"Added mask from {path}  shape={mask.shape} -> aligned {aligned.shape}; "
                    f"listed under {target_group['name']}."
                )
            else:
                if support is None:
                    self._support = mask
                self._refresh_support_mask_overlay()
                self._refresh_mask_list()
                self._set_status(
                    f"Added mask from {path}  shape={mask.shape}; listed under "
                    f"{target_group['name']} and will align before CDI."
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _load_bad_pixel_mask_npy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Bad Pixel Mask",
            "",
            "NumPy masks (*.npz *.npy);;All files (*)",
        )
        if not path:
            return
        try:
            mask, meta = self._load_bad_pixel_mask_file(path)
            if mask.ndim != 2:
                raise ValueError(f"Bad pixel mask must be 2D after squeeze; got shape {mask.shape}")
            self._bad_pixel_mask_source = mask
            self._bad_pixel_mask_source_meta = meta
            aligned = self._align_saved_center_bad_pixel_mask(mask, meta)
            if aligned is None:
                aligned = self._align_bad_pixel_mask_to_current_fth(mask, show_error=False)
            if aligned is not None:
                self._bad_pixel_mask = aligned
                self._refresh_bad_pixel_mask_overlay()
                center_note = ""
                if meta and meta.get("center") is not None:
                    fth = self._fth_tool
                    saved_x, saved_y = meta["center"]
                    center_note = (
                        f"; saved center=({saved_x}, {saved_y}), "
                        f"current center=({int(fth._t1_xmid.value())}, {int(fth._t1_ymid.value())})"
                    )
                self._set_bad_pixel_mask_label(
                    f"Loaded {mask.shape} -> aligned {aligned.shape}; bad pixels={int(np.count_nonzero(aligned))}; "
                    f"shift X={self._bad_pixel_shift_x}, Y={self._bad_pixel_shift_y}{center_note}"
                )
                self._set_status(f"Loaded bad pixel mask: {aligned.shape}")
            else:
                self._bad_pixel_mask = None
                self._refresh_bad_pixel_mask_overlay()
                self._set_bad_pixel_mask_label(f"Loaded {mask.shape}; will align after FTH data is ready.")
                self._set_status("Loaded bad pixel mask; waiting for FTH-aligned data.")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _clear_bad_pixel_mask(self) -> None:
        self._bad_pixel_mask_source = None
        self._bad_pixel_mask_source_meta = None
        self._bad_pixel_mask = None
        self._last_hist_shape = None  # force threshold line reset on next data load
        self._refresh_bad_pixel_mask_overlay()
        if hasattr(self, "_threshold_preview_img"):
            self._threshold_preview_img.clear()
            self._threshold_preview_img.setVisible(False)
        if hasattr(self, "_thresh_preview_label"):
            self._thresh_preview_label.setText("Preview: — px selected")
        self._set_bad_pixel_mask_label("No mask")
        self._set_status("Bad pixel mask cleared.")

    def _on_bad_pixel_shift_changed(self, *_args) -> None:
        self._bad_pixel_shift_x = int(self._bad_pixel_shift_x_spin.value())
        self._bad_pixel_shift_y = int(self._bad_pixel_shift_y_spin.value())
        self._refresh_bad_pixel_mask_overlay()
        if self._bad_pixel_mask is not None:
            self._set_bad_pixel_mask_label(
                f"Active {self._bad_pixel_mask.shape}; bad pixels={int(np.count_nonzero(self._bad_pixel_mask))}; "
                f"shift X={self._bad_pixel_shift_x}, Y={self._bad_pixel_shift_y}"
            )

    def _set_bad_pixel_mask_label(self, text: str) -> None:
        if hasattr(self, "_bad_pixel_mask_label"):
            self._bad_pixel_mask_label.setText(text)

    # ------------------------------------------------------------------
    # Alignment tab simplification
    # ------------------------------------------------------------------

    def _hide_alignment_tab_extras(self, alignment_tab: QWidget) -> None:
        """Hide FTH groups and overlays not needed in the CDI workflow.

        Must receive the already-detached alignment_tab widget because after
        _take_fth_tab() the groups are no longer children of self._fth_tool.
        """
        fth = self._fth_tool
        # Hide slit mask, lock buttons — not needed for CDI
        for attr in ("_g_slit_mask", "_btn_lock_params", "_btn_load_apply_locked"):
            w = getattr(fth, attr, None)
            if w is not None:
                w.setVisible(False)
        # Hide Slit Angles group only (keep Beamstop Center visible)
        for w in alignment_tab.findChildren(QGroupBox):
            if w.title() == "Slit Angles":
                w.setVisible(False)
        # Hide slit-angle lines only; keep beamstop circle/dot visible
        for attr in ("_t1_slit1_line", "_t1_slit2_line"):
            item = getattr(fth, attr, None)
            if item is not None:
                item.setVisible(False)

    # ------------------------------------------------------------------
    # Mask editor — mode switching
    # ------------------------------------------------------------------

    def _set_mask_edit_mode(self, mode: str) -> None:
        self._mask_edit_mode = mode
        self._pan_last_scene_pos = None
        active = mode in ("brush", "eraser")
        self._mask_mode_brush_btn.setChecked(mode == "brush")
        self._mask_mode_eraser_btn.setChecked(mode == "eraser")
        self._brush_panel.setVisible(active)

        # Disable ViewBox left-drag pan while a paint mode is active so
        # left-button is free for painting; right-drag pan handled manually.
        vb = self._fth_tool._t1_main_plot.getViewBox()
        vb.setMouseEnabled(x=not active, y=not active)

        if hasattr(self, "_threshold_preview_img"):
            self._threshold_preview_img.setVisible(False)

    # ------------------------------------------------------------------
    # Alignment display callback — drives histogram + overlay refresh
    # ------------------------------------------------------------------

    def _on_alignment_display_updated(self) -> None:
        self._refresh_bad_pixel_mask_overlay()

    # ------------------------------------------------------------------
    # Threshold histogram
    # ------------------------------------------------------------------

    def _get_mask_base_data(self) -> Optional[np.ndarray]:
        """Raw intensity array shown in the Alignment view (before log transform)."""
        fth = self._fth_tool
        data = getattr(fth, "_t1_value_data", None)
        if data is not None:
            return np.asarray(data, dtype=np.float64)
        for attr in ("_CL_c", "_CL"):
            arr = getattr(fth, attr, None)
            if arr is not None:
                return np.abs(np.asarray(arr, dtype=np.float64))
        return None

    def _refresh_threshold_histogram(self) -> None:
        """Feed current alignment data into the curves widget."""
        if not hasattr(self, "_curves_widget"):
            return
        data = self._get_mask_base_data()
        if data is not None:
            self._curves_widget.initialize(data)

    def _on_curves_changed(self) -> None:
        """Called whenever the curves widget emits curveChanged."""
        if self._mask_edit_mode == "threshold":
            self._refresh_threshold_preview_overlay()

    # keep old name as alias so any external callers don't break
    _on_threshold_line_moved = _on_curves_changed  # type: ignore[assignment]

    def _compute_threshold_mask(self) -> Optional[np.ndarray]:
        """Boolean mask: pixels where curves weight > 0.5 are masked."""
        data = self._get_mask_base_data()
        if data is None or not hasattr(self, "_curves_widget"):
            return None
        return self._curves_widget.evaluate(data) > 0.5

    def _refresh_threshold_preview_overlay(self) -> None:
        """Show orange overlay for threshold-selected pixels (preview only)."""
        if not hasattr(self, "_threshold_preview_img"):
            return
        mask = self._compute_threshold_mask()
        if mask is None or not np.any(mask):
            self._threshold_preview_img.clear()
            self._threshold_preview_img.setVisible(False)
            if hasattr(self, "_thresh_preview_label"):
                self._thresh_preview_label.setText("Preview: 0 px selected")
            return
        n = int(np.count_nonzero(mask))
        if hasattr(self, "_thresh_preview_label"):
            self._thresh_preview_label.setText(f"Preview: {n:,} px selected")
        rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = 165
        rgba[..., 3] = np.where(mask, 160, 0).astype(np.uint8)
        self._threshold_preview_img.setImage(rgba, autoLevels=False)
        self._match_overlay_image_geometry(self._threshold_preview_img, self._fth_tool._t1_main_img)
        self._threshold_preview_img.setVisible(True)

    def _apply_threshold_to_mask(self, replace: bool = False) -> None:
        """Merge or replace the permanent bad pixel mask with the threshold selection."""
        mask = self._compute_threshold_mask()
        if mask is None:
            QMessageBox.warning(self, "No Data", "Load CL/CR data first.")
            return
        if replace or self._bad_pixel_mask is None:
            self._bad_pixel_mask = mask.copy()
        else:
            self._bad_pixel_mask = self._bad_pixel_mask.astype(bool) | mask
        self._bad_pixel_mask_source = None
        self._bad_pixel_mask_source_meta = None
        n = int(np.count_nonzero(self._bad_pixel_mask))
        self._set_bad_pixel_mask_label(f"Active {self._bad_pixel_mask.shape}; {n:,} bad px")
        self._refresh_bad_pixel_mask_overlay()
        verb = "Replaced" if replace else "Added to"
        self._set_status(f"{verb} bad pixel mask — {n:,} pixels now masked.")

    # ------------------------------------------------------------------
    # Brush / Eraser painting
    # ------------------------------------------------------------------

    def _ensure_mask_array(self) -> bool:
        """Initialise _bad_pixel_mask if not already set. Returns True on success."""
        fth = self._fth_tool
        shape: Optional[tuple] = None
        data = getattr(fth, "_t1_value_data", None)
        if data is not None:
            shape = data.shape
        elif getattr(fth, "_Nx", 0) > 0 and getattr(fth, "_Ny", 0) > 0:
            shape = (int(fth._Nx), int(fth._Ny))
        if shape is None:
            return False
        if self._bad_pixel_mask is None or self._bad_pixel_mask.shape != shape:
            self._bad_pixel_mask = np.zeros(shape, dtype=bool)
            self._bad_pixel_mask_source = None
        return True

    def _on_alignment_mouse_clicked_paint(self, event) -> None:
        """Single-click brush/eraser paint."""
        if self._mask_edit_mode not in ("brush", "eraser"):
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self._fth_tool._t1_main_plot.getViewBox()
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        pt = vb.mapSceneToView(pos)
        self._paint_mask_at(int(round(float(pt.y()))), int(round(float(pt.x()))),
                            add=(self._mask_edit_mode == "brush"))

    def _on_alignment_mouse_moved_paint(self, pos) -> None:
        """Left-drag: paint/erase.  Right-drag: pan the view."""
        if self._mask_edit_mode not in ("brush", "eraser"):
            self._pan_last_scene_pos = None
            return

        from PyQt6.QtWidgets import QApplication as _App
        buttons = _App.mouseButtons()
        vb = self._fth_tool._t1_main_plot.getViewBox()

        if buttons & Qt.MouseButton.RightButton:
            # Manual pan while ViewBox mouse is disabled
            if self._pan_last_scene_pos is not None:
                prev = vb.mapSceneToView(self._pan_last_scene_pos)
                curr = vb.mapSceneToView(pos)
                dx = curr.x() - prev.x()
                dy = curr.y() - prev.y()
                vb.translateBy(x=-dx, y=-dy)
            self._pan_last_scene_pos = pos
            return

        self._pan_last_scene_pos = None

        if not (buttons & Qt.MouseButton.LeftButton):
            return
        if not vb.sceneBoundingRect().contains(pos):
            return
        pt = vb.mapSceneToView(pos)
        self._paint_mask_at(int(round(float(pt.y()))), int(round(float(pt.x()))),
                            add=(self._mask_edit_mode == "brush"))

    def _paint_mask_at(self, row: int, col: int, add: bool = True) -> None:
        """Paint (add=True) or erase (add=False) a circular brush footprint."""
        if not self._ensure_mask_array():
            return
        H, W = self._bad_pixel_mask.shape
        r = self._brush_radius_spin.value()
        rr, cc = np.ogrid[-r: r + 1, -r: r + 1]
        disk = (rr ** 2 + cc ** 2) <= r ** 2
        r0, r1 = max(0, row - r), min(H, row + r + 1)
        c0, c1 = max(0, col - r), min(W, col + r + 1)
        dr0, dr1 = r0 - (row - r), r1 - (row - r)
        dc0, dc1 = c0 - (col - r), c1 - (col - r)
        if r1 > r0 and c1 > c0:
            patch = disk[dr0:dr1, dc0:dc1]
            if add:
                self._bad_pixel_mask[r0:r1, c0:c1] |= patch
            else:
                self._bad_pixel_mask[r0:r1, c0:c1] &= ~patch
        self._bad_pixel_mask_source = None
        n = int(np.count_nonzero(self._bad_pixel_mask))
        self._set_bad_pixel_mask_label(f"Active {self._bad_pixel_mask.shape}; {n:,} bad px")
        self._refresh_bad_pixel_mask_overlay()

    # ------------------------------------------------------------------
    # Save bad pixel mask
    # ------------------------------------------------------------------

    def _save_bad_pixel_mask_npy(self) -> None:
        if self._bad_pixel_mask is None:
            QMessageBox.warning(self, "No Mask", "No bad pixel mask to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Bad Pixel Mask",
            "",
            "NumPy archive with center (*.npz);;Legacy mask only (*.npy)",
        )
        if path:
            fth = self._fth_tool
            xmid = int(fth._t1_xmid.value())
            ymid = int(fth._t1_ymid.value())
            raw = getattr(fth, "_CL", None)
            raw_shape = np.array(raw.shape if raw is not None else self._bad_pixel_mask.shape, dtype=np.int64)
            mask = self._bad_pixel_mask.astype(np.uint8)
            if path.lower().endswith(".npy"):
                np.save(
                    path,
                    {
                        "mask": mask,
                        "center": np.array([xmid, ymid], dtype=np.int64),
                        "raw_shape": raw_shape,
                        "mask_shape": np.array(mask.shape, dtype=np.int64),
                    },
                    allow_pickle=True,
                )
            else:
                if not path.lower().endswith(".npz"):
                    path += ".npz"
                np.savez(
                    path,
                    mask=mask,
                    center=np.array([xmid, ymid], dtype=np.int64),
                    raw_shape=raw_shape,
                    mask_shape=np.array(mask.shape, dtype=np.int64),
                )
            self._set_status(f"Saved bad pixel mask with center ({xmid}, {ymid}) → {path}")

    @staticmethod
    def _shift_boolean_mask(mask: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim != 2 or (shift_y == 0 and shift_x == 0):
            return arr.copy()
        out = np.zeros_like(arr, dtype=bool)
        rows, cols = arr.shape
        src_y0 = max(0, -shift_y)
        src_y1 = min(rows, rows - shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(cols, cols - shift_x)
        if src_y0 >= src_y1 or src_x0 >= src_x1:
            return out
        dst_y0 = src_y0 + shift_y
        dst_y1 = src_y1 + shift_y
        dst_x0 = src_x0 + shift_x
        dst_x1 = src_x1 + shift_x
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
        return out

    @staticmethod
    def _crop_bounds_for_center(shape: tuple[int, int], xmid: int, ymid: int) -> tuple[slice, slice]:
        rows, cols = shape
        xsize = min(xmid, rows - 1 - xmid)
        ysize = min(ymid, cols - 1 - ymid)
        minsize = min(xsize, ysize)
        if minsize <= 0:
            raise ValueError("Mask cannot be centered with the stored or current center.")
        row_slice = slice(xmid - minsize, xmid + minsize)
        col_slice = slice(ymid - minsize, ymid + minsize)
        return row_slice, col_slice

    def _load_bad_pixel_mask_file(self, path: str) -> tuple[np.ndarray, Optional[dict]]:
        loaded = np.load(path, allow_pickle=True)
        meta = None
        try:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                mask = np.squeeze(np.asarray(loaded["mask"])).astype(np.float32)
                meta = {
                    "center": tuple(int(v) for v in np.asarray(loaded["center"]).ravel()[:2]),
                    "raw_shape": tuple(int(v) for v in np.asarray(loaded["raw_shape"]).ravel()[:2]),
                }
            else:
                arr = np.asarray(loaded)
                if arr.dtype == object and arr.shape == ():
                    obj = arr.item()
                    if isinstance(obj, dict) and "mask" in obj:
                        mask = np.squeeze(np.asarray(obj["mask"])).astype(np.float32)
                        center = obj.get("center")
                        raw_shape = obj.get("raw_shape")
                        if center is not None and raw_shape is not None:
                            meta = {
                                "center": tuple(int(v) for v in np.asarray(center).ravel()[:2]),
                                "raw_shape": tuple(int(v) for v in np.asarray(raw_shape).ravel()[:2]),
                            }
                    else:
                        raise ValueError("Unsupported object .npy bad pixel mask format.")
                else:
                    mask = np.squeeze(arr).astype(np.float32)
        finally:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded.close()
        return mask, meta

    def _align_saved_center_bad_pixel_mask(
        self,
        mask: np.ndarray,
        meta: Optional[dict],
    ) -> Optional[np.ndarray]:
        if not meta:
            return None
        fth = self._fth_tool
        raw_shape = tuple(meta.get("raw_shape", ()))
        saved_center = meta.get("center")
        raw = getattr(fth, "_CL", None)
        current_raw_shape = tuple(raw.shape) if raw is not None else raw_shape
        if len(raw_shape) != 2 or len(current_raw_shape) != 2 or raw_shape != current_raw_shape:
            return None
        if saved_center is None:
            return None
        saved_x, saved_y = (int(saved_center[0]), int(saved_center[1]))
        curr_x = int(fth._t1_xmid.value())
        curr_y = int(fth._t1_ymid.value())
        old_rows, old_cols = self._crop_bounds_for_center(raw_shape, saved_x, saved_y)
        canvas = np.zeros(raw_shape, dtype=np.float32)
        old_h = min(mask.shape[0], old_rows.stop - old_rows.start)
        old_w = min(mask.shape[1], old_cols.stop - old_cols.start)
        canvas[old_rows.start:old_rows.start + old_h, old_cols.start:old_cols.start + old_w] = (
            np.asarray(mask[:old_h, :old_w]) > 0
        )
        new_rows, new_cols = self._crop_bounds_for_center(raw_shape, curr_x, curr_y)
        aligned = canvas[new_rows, new_cols]
        target_shape = None
        if getattr(fth, "_t1_value_data", None) is not None:
            target_shape = tuple(fth._t1_value_data.shape)
        if target_shape is not None and tuple(aligned.shape) != target_shape:
            return None
        return self._shift_boolean_mask(aligned > 0, self._bad_pixel_shift_y, self._bad_pixel_shift_x)

    def _align_bad_pixel_mask_to_current_fth(
        self,
        mask: np.ndarray,
        show_error: bool = True,
    ) -> Optional[np.ndarray]:
        """Return bad-pixel mask in the coordinate system of the Alignment image."""
        try:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim != 2:
                raise ValueError(f"Bad pixel mask must be 2D; got shape {mask.shape}")

            fth = self._fth_tool
            target_shape = None
            if getattr(fth, "_t1_value_data", None) is not None:
                target_shape = tuple(fth._t1_value_data.shape)
            elif getattr(fth, "_CL_c", None) is not None:
                target_shape = tuple(fth._CL_c.shape)

            raw_shape = tuple(fth._CL.shape) if getattr(fth, "_CL", None) is not None else None
            if raw_shape is not None and tuple(mask.shape) == raw_shape:
                aligned = self._crop_array_like_current_fth(mask)
                if target_shape is not None and tuple(aligned.shape) != target_shape:
                    raise ValueError(
                        f"Aligned bad pixel mask shape {aligned.shape} != current Alignment image {target_shape}"
                    )
                return self._shift_boolean_mask(
                    np.asarray(aligned) > 0,
                    self._bad_pixel_shift_y,
                    self._bad_pixel_shift_x,
                )

            if target_shape is not None and tuple(mask.shape) == target_shape:
                return self._shift_boolean_mask(
                    mask > 0,
                    self._bad_pixel_shift_y,
                    self._bad_pixel_shift_x,
                )

            if target_shape is None:
                return None
            raise ValueError(
                f"Bad pixel mask shape {mask.shape} does not match raw FTH data {raw_shape} "
                f"or current Alignment image {target_shape}."
            )
        except Exception as exc:
            if show_error:
                QMessageBox.warning(self, "Bad Pixel Mask Shape Mismatch", str(exc))
                self._set_status(f"Bad pixel mask shape mismatch: {exc}")
            return None

    def _current_bad_pixel_mask(self) -> Optional[np.ndarray]:
        if self._bad_pixel_mask_source is not None:
            aligned = self._align_saved_center_bad_pixel_mask(
                self._bad_pixel_mask_source,
                self._bad_pixel_mask_source_meta,
            )
            if aligned is None:
                aligned = self._align_bad_pixel_mask_to_current_fth(self._bad_pixel_mask_source)
            if aligned is not None:
                self._bad_pixel_mask = aligned
                self._set_bad_pixel_mask_label(
                    f"Active {aligned.shape}; bad pixels={int(np.count_nonzero(aligned))}; "
                    f"shift X={self._bad_pixel_shift_x}, Y={self._bad_pixel_shift_y}"
                )
            return aligned
        if self._bad_pixel_mask is not None:
            aligned = self._align_bad_pixel_mask_to_current_fth(self._bad_pixel_mask)
            if aligned is not None:
                self._bad_pixel_mask = aligned
            return aligned
        return None

    def _refresh_bad_pixel_mask_overlay(self) -> None:
        """Draw bad pixels as a transparent overlay on the Alignment image."""
        if not hasattr(self, "_bad_pixel_mask_img"):
            return
        mask = None
        if self._bad_pixel_mask_source is not None:
            mask = self._align_saved_center_bad_pixel_mask(
                self._bad_pixel_mask_source,
                self._bad_pixel_mask_source_meta,
            )
            if mask is None:
                mask = self._align_bad_pixel_mask_to_current_fth(self._bad_pixel_mask_source, show_error=False)
            if mask is not None:
                self._bad_pixel_mask = mask
        elif self._bad_pixel_mask is not None:
            mask = self._align_bad_pixel_mask_to_current_fth(self._bad_pixel_mask, show_error=False)
            if mask is not None:
                self._bad_pixel_mask = mask

        if mask is None:
            self._bad_pixel_mask_img.clear()
            self._bad_pixel_mask_img.setVisible(False)
            return
        try:
            arr = np.asarray(mask, dtype=bool)
            if arr.ndim != 2 or not np.any(arr):
                self._bad_pixel_mask_img.clear()
                self._bad_pixel_mask_img.setVisible(False)
                return
            rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = 45
            rgba[..., 2] = 45
            rgba[..., 3] = np.where(arr, 120, 0).astype(np.uint8)
            self._bad_pixel_mask_img.setImage(rgba, autoLevels=False)
            self._match_overlay_image_geometry(self._bad_pixel_mask_img, self._fth_tool._t1_main_img)
            self._bad_pixel_mask_img.setVisible(True)
        except Exception as exc:
            log.debug("Bad pixel mask overlay refresh failed: %s", exc)
            self._bad_pixel_mask_img.setVisible(False)

    @staticmethod
    def _match_overlay_image_geometry(overlay: pg.ImageItem, base: pg.ImageItem) -> None:
        """Keep transparent mask overlays in the exact item geometry of their image."""
        overlay.setPos(base.pos())
        overlay.setTransform(base.transform())

    def _refresh_support_mask_overlay(self) -> None:
        """Draw the current support mask as a transparent overlay on the FTH panel."""
        # If CDI preview mode is active, override the FTH image with CDI real-space data
        # BEFORE drawing the mask overlay on top.
        self._apply_cdi_preview_image()
        if not hasattr(self, "_supp_mask_img"):
            return
        mask = self._support
        if any(self._group_source_masks(g) for g in self._mask_groups):
            grouped = self._compute_support_mask()
            if grouped is not None:
                mask = grouped
                self._support = grouped
        if self._support_source_mask is not None:
            aligned = self._align_support_mask_to_current_fth(self._support_source_mask, show_error=False)
            if aligned is not None:
                mask = aligned
                self._support = aligned
        if mask is None:
            self._supp_mask_img.clear()
            self._supp_mask_img.setVisible(False)
            return
        try:
            arr = np.asarray(mask, dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0:
                self._supp_mask_img.clear()
                self._supp_mask_img.setVisible(False)
                return

            visible = np.isfinite(arr) & (arr > 0)
            if not np.any(visible):
                self._supp_mask_img.clear()
                self._supp_mask_img.setVisible(False)
                return

            rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
            rgba[..., 0] = 0
            rgba[..., 1] = 220
            rgba[..., 2] = 255
            rgba[..., 3] = np.where(visible, 95, 0).astype(np.uint8)
            self._supp_mask_img.setImage(rgba, autoLevels=False)
            self._match_overlay_image_geometry(self._supp_mask_img, self._supp_img)
            self._supp_mask_img.setVisible(True)
        except Exception as exc:
            log.debug("Support mask overlay refresh failed: %s", exc)
            self._supp_mask_img.setVisible(False)

    def _on_cdi_preview_toggled(self, checked: bool) -> None:
        """Switch the support panel image between CDI real-space and FTH reconstruction."""
        self._cdi_preview_active = checked
        if checked:
            self._apply_cdi_preview_image()
        else:
            # Restore FTH reconstruction display; falls through to callback which will
            # re-draw the mask overlay on top.
            self._fth_tool._update_t3_fth_display()

    def _apply_cdi_preview_image(self) -> None:
        """If CDI preview is active, set |ifft2c(√I_CL)|^0.4 into the support image panel.

        The CDI real-space field has the SAMPLE at the image center (row=H/2, col=W/2).
        This is the correct position for drawing the CDI support — NOT the FTH
        cross-correlation peak which is displaced to the reference-hole offset.
        """
        if not getattr(self, "_cdi_preview_active", False):
            return
        if not hasattr(self, "_supp_img"):
            return
        fth = self._fth_tool
        cl = getattr(fth, "_CL_smooth", None)
        if cl is None:
            cl = getattr(fth, "_CL_c", None)
        if cl is None:
            return
        try:
            amp = np.sqrt(np.maximum(np.asarray(cl, dtype=np.float64), 0.0))
            psi = _ifft2c(amp.astype(np.complex128))
            data = np.abs(psi).astype(np.float32) ** 0.4
            self._supp_img.setImage(data, autoLevels=True)
        except Exception as exc:
            log.debug("CDI preview image failed: %s", exc)

    def _crop_array_like_current_fth(self, arr: np.ndarray) -> np.ndarray:
        """Apply the same center crop that FTH uses for CL/CR alignment."""
        fth = self._fth_tool
        xmid = int(fth._t1_xmid.value())
        ymid = int(fth._t1_ymid.value())
        rows, cols = arr.shape
        xsize = min(xmid, rows - 1 - xmid)
        ysize = min(ymid, cols - 1 - ymid)
        minsize = min(xsize, ysize)
        if minsize <= 0:
            raise ValueError("Support mask cannot be centered with the current FTH center.")
        cropped = arr[xmid - minsize:xmid + minsize, ymid - minsize:ymid + minsize]
        if cropped.shape[0] % 2 == 1:
            cropped = cropped[:-1, :-1]
        return cropped

    def _align_support_mask_to_current_fth(
        self,
        mask: np.ndarray,
        show_error: bool = True,
    ) -> Optional[np.ndarray]:
        """Return support mask in the same cropped coordinate system as CDI data."""
        try:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim != 2:
                raise ValueError(f"Support mask must be 2D; got shape {mask.shape}")
            fth = self._fth_tool
            target_shape = None
            if self._amp_meas is not None:
                target_shape = tuple(self._amp_meas.shape)
            elif getattr(fth, "_CL_c", None) is not None:
                target_shape = tuple(fth._CL_c.shape)

            raw_shape = tuple(fth._CL.shape) if getattr(fth, "_CL", None) is not None else None
            if raw_shape is not None and tuple(mask.shape) == raw_shape:
                aligned = self._crop_array_like_current_fth(mask)
                if target_shape is not None and tuple(aligned.shape) != target_shape:
                    raise ValueError(
                        f"Aligned support mask shape {aligned.shape} != current CDI data {target_shape}"
                    )
                return aligned.astype(np.float32, copy=False)

            if target_shape is not None and tuple(mask.shape) == target_shape:
                return mask.astype(np.float32, copy=True)

            if target_shape is None:
                return None
            raise ValueError(
                f"Support mask shape {mask.shape} does not match raw FTH data {raw_shape} "
                f"or current CDI data {target_shape}."
            )
        except Exception as exc:
            if show_error:
                QMessageBox.warning(self, "Support Mask Shape Mismatch", str(exc))
                self._set_status(f"Support mask shape mismatch: {exc}")
            return None

    def _current_support_for_reconstruction(self) -> "np.ndarray | bool | None":
        # Primary support = group 0 ONLY.  Feature groups (1+) are applied as
        # separate hard constraints inside the worker, not merged into support.
        if self._mask_groups and (
            self._mask_groups[0]["entries"]
            or self._group_source_masks(self._mask_groups[0])
        ):
            support = self._compute_group_mask(0)
            if support is not None:
                self._support = support
            return support
        # Fallback: if no groups defined, still show a combined mask so NPY-loaded
        # masks still work (those don't use _mask_groups at all).
        if any(g["entries"] or self._group_source_masks(g) for g in self._mask_groups):
            support = self._compute_support_mask()
            if support is not None:
                self._support = support
            return support
        if self._support_source_mask is not None:
            support = self._align_support_mask_to_current_fth(self._support_source_mask)
            if support is not None:
                self._support = support
                self._refresh_support_mask_overlay()
            return support
        if self._support is not None:
            support = self._align_support_mask_to_current_fth(self._support)
            if support is not None:
                self._support = support
                self._refresh_support_mask_overlay()
            return support
        return None

    def _collect_feature_constraints(
        self, initial_obj: np.ndarray
    ) -> list:
        """Build (mask, values) pairs for feature groups (index 1+).

        The reference values are taken from *initial_obj* (the FTH-derived or
        previous-result initial estimate).  These become the region-C hard
        constraints in the CDI worker: feature pixels are locked to these
        values after every HIO/ER/RAAR projection.
        """
        constraints = []
        for i in range(1, len(self._mask_groups)):
            grp_mask = self._compute_group_mask(i)
            if grp_mask is None:
                continue
            bool_mask = grp_mask.astype(bool)
            if not np.any(bool_mask):
                continue
            if bool_mask.shape != initial_obj.shape:
                continue
            ref_vals = initial_obj.copy()
            constraints.append((bool_mask, ref_vals))
        return constraints

    # ==================================================================
    # CDI-specific FFT preview (applies bad pixel mask before FTH pipeline)
    # ==================================================================

    def _cdi_compute_fft(self) -> None:
        """Compute FFT (FTH reconstruction) with CDI bad pixel mask applied.

        Fills bad pixel positions with the local 5×5 median in a temporary
        copy of the FTH hologram data, runs the standard FTH filter + FFT
        pipeline, then restores the original (unmodified) FTH data.
        """
        fth = self._fth_tool
        bad_mask = self._current_bad_pixel_mask()

        if bad_mask is None or not np.any(bad_mask):
            fth._apply_and_compute_fth()
            return

        def _fill(data) -> Optional[np.ndarray]:
            if data is None:
                return None
            d = np.asarray(data, dtype=np.float64)
            if d.shape != bad_mask.shape:
                return data
            filled = d.copy()
            filled[bad_mask] = median_filter(d, size=5)[bad_mask]
            return filled

        attrs = ("_CL_c", "_CR_c", "_CL_smooth", "_CR_smooth")
        orig = {a: getattr(fth, a, None) for a in attrs}
        try:
            for a, v in orig.items():
                if v is not None:
                    setattr(fth, a, _fill(v))
            fth._apply_and_compute_fth()
        finally:
            for a, v in orig.items():
                setattr(fth, a, v)

    # ==================================================================
    # FTH data sync & seed computation
    # ==================================================================

    def _sync_from_fth_tool(self) -> bool:
        """Pull aligned CL/CR data and masks from the FTH tool.

        Must be called before starting CDI to ensure we use the same
        centered/cropped hologram the FTH reconstruction uses.
        """
        fth = self._fth_tool
        cl_c = getattr(fth, "_CL_c", None)
        cr_c = getattr(fth, "_CR_c", None)
        if cl_c is None or cr_c is None:
            QMessageBox.warning(
                self, "No FTH Data",
                "Load one dataset or a CL/CR pair and complete the Alignment page first."
            )
            return False
        if np.asarray(cl_c).shape != np.asarray(cr_c).shape:
            QMessageBox.warning(
                self, "Shape Mismatch",
                f"CL shape {np.asarray(cl_c).shape} != CR shape {np.asarray(cr_c).shape}"
            )
            return False

        # Prefer masked data (_CL_smooth = _CL_c * bs_mask * slit_mask) if available;
        # falls back to raw centered hologram otherwise.
        cl_smooth = getattr(fth, "_CL_smooth", None)
        cr_smooth = getattr(fth, "_CR_smooth", None)
        src_cl = cl_smooth if cl_smooth is not None else cl_c
        src_cr = cr_smooth if cr_smooth is not None else cr_c

        cl_f = np.asarray(src_cl, dtype=np.float64)
        cr_f = np.asarray(src_cr, dtype=np.float64)
        self._single_dataset_mode = bool(getattr(fth, "_single_dataset_mode", False))
        self._int_cl = cl_f.copy()
        self._int_cr = cr_f.copy()
        self._amp_cl = np.sqrt(np.maximum(cl_f, 0.0))
        self._amp_cr = np.sqrt(np.maximum(cr_f, 0.0))

        # Build floating mask from beamstop + slit masks
        floating = np.zeros(cl_f.shape, dtype=bool)
        for attr in ("_bs_mask", "_slit_mask"):
            m = getattr(fth, attr, None)
            if m is not None:
                floating |= np.asarray(m) < 0.95
        bad_mask = self._current_bad_pixel_mask()
        if bad_mask is not None:
            if tuple(bad_mask.shape) != tuple(cl_f.shape):
                QMessageBox.warning(
                    self,
                    "Bad Pixel Mask Shape Mismatch",
                    f"Bad pixel mask shape {bad_mask.shape} != current CDI data {cl_f.shape}",
                )
                return False
            floating |= bad_mask
        self._pixel_mask = floating if np.any(floating) else None

        # Keep _amp_meas as the active measured amplitude for shape checks.
        self._amp_meas = self._amp_cl if self._single_dataset_mode else (self._amp_cl + self._amp_cr)
        return True

    def _compute_fth_seed_complex(self, show_errors: bool = False) -> "np.ndarray | bool":
        """Return complex FTH real-space image for use as CDI initial phase.

        Priority:
          1. Use already-computed _FTH_S1 / _FTH_S2 from the FTH tool.
          2. Compute a simple FTH from aligned _CL_c / _CR_c.
          3. Fail with an informative message.
        """
        fth = self._fth_tool

        # 1. Prefer already-computed FTH reconstruction
        s1 = getattr(fth, "_FTH_S1", None)
        s2 = getattr(fth, "_FTH_S2", None)
        if s1 is not None:
            active = int(getattr(fth, "_current_slit", 1))
            result = (s2 if active == 2 and s2 is not None else s1)
            self._set_status("Using existing FTH reconstruction as CDI phase seed.")
            return result.astype(np.complex128)

        # 2. Compute simple FTH from aligned hologram with proper centering phase correction.
        #    This matches the FTH tool's _compute_fth_only() formula exactly.
        cl_c = getattr(fth, "_CL_c", None)
        cr_c = getattr(fth, "_CR_c", None)
        if cl_c is not None and cr_c is not None:
            try:
                cl = np.asarray(cl_c, dtype=np.float64)
                cr = np.asarray(cr_c, dtype=np.float64)
                ratio = float(getattr(fth, "_balance_ratio", 1.0))
                holo  = cl - ratio * cr
                raw   = np.fft.fftshift(np.fft.fft2(holo))
                # Apply same centering phase correction the FTH tool uses
                xmat = getattr(fth, "_xmat", None)
                ymat = getattr(fth, "_ymat", None)
                X0   = getattr(fth, "_X0",   0)
                Y0   = getattr(fth, "_Y0",   0)
                Nx   = getattr(fth, "_Nx",   holo.shape[0])
                Ny   = getattr(fth, "_Ny",   holo.shape[1])
                if xmat is not None and ymat is not None and Nx > 0 and Ny > 0:
                    phase_corr = np.exp(
                        2j * np.pi * (xmat * X0 / Nx + ymat * Y0 / Ny)
                    )
                    raw = raw * phase_corr
                result = raw.astype(np.complex128)
                self._set_status(
                    f"Computed FTH seed from aligned CL/CR (balance={ratio:.4f})."
                )
                return result
            except Exception as exc:
                if show_errors:
                    QMessageBox.critical(self, "FTH Seed Error", str(exc))
                self._set_status(f"FTH seed error: {exc}")
                return False

        # 3. Nothing available
        if show_errors:
            QMessageBox.warning(
                self, "No FTH Data",
                "Complete the Alignment page and run Filter & FTH before starting CDI."
            )
        return False

    @staticmethod
    def _estimate_balance_ratio(cl: np.ndarray, cr: np.ndarray) -> float:
        valid = np.isfinite(cl) & np.isfinite(cr) & (cl > 0) & (cr > 0)
        if not np.any(valid):
            return 1.0
        a = cl[valid].ravel()
        b = cr[valid].ravel()
        step = max(1, a.size // 200_000)
        a, b = a[::step], b[::step]
        den = float(np.dot(b, b))
        if den <= 0:
            return 1.0
        return float(np.clip(np.dot(a, b) / den, 0.1, 10.0))

    # ==================================================================
    # CDI run / stop
    # ==================================================================

    # ==================================================================
    # Pipeline target management
    # ==================================================================

    def _clear_pipeline_targets(self) -> None:
        for entry in list(self._pipeline_targets):
            entry['card'].deleteLater()
        self._pipeline_targets.clear()

    def _load_classic_pipeline_preset(self) -> None:
        """Populate the target list with the configurable Classic CL->CR flow."""
        self._clear_pipeline_targets()
        if bool(getattr(self._fth_tool, "_single_dataset_mode", False)):
            self._add_interval_step("none", "arctan (cold-start only)", True)
            self._add_pipeline_target("CL", "support", [("mine", 700)])
            self._initial_guess_combo.setCurrentText("Support FFT phase")
            self._scale_init_chk.setChecked(False)
            self._scale_inherited_chk.setChecked(False)
            self._sw_group.setChecked(False)
            self._restarts_spin.setValue(1)
            self._set_status("Single-file mine preset loaded. Press Start to run it.")
            return
        self._add_interval_step("none", "arctan (cold-start only)", True)
        self._add_pipeline_target("CL", "support", [("mine", 700)])
        self._add_interval_step("none", "constant", True)
        self._add_pipeline_target("CL", "support", [("mine", 50)])
        self._add_interval_step("normalize", "constant", True)
        self._add_pipeline_target("CR", "support", [("mine", 50)])
        self._initial_guess_combo.setCurrentText("Support FFT phase")
        self._scale_init_chk.setChecked(False)
        self._scale_inherited_chk.setChecked(False)
        self._sw_group.setChecked(False)
        self._restarts_spin.setValue(1)
        self._set_status("Classic CL->CR preset loaded. Press Start to run it.")

    def _add_pipeline_target(
        self,
        data_source: str = "CL+CR",
        mask_mode: str = "support",
        steps: Optional[list] = None,
    ) -> None:
        if steps is None:
            steps = [("raar", 200), ("er", 30)]

        t_idx = len(self._pipeline_targets)

        card = QWidget()
        card.setObjectName("target_card")
        card.setStyleSheet(
            "QWidget#target_card {"
            "  background: #f5f5f5;"
            "  border: 1px solid #aaa;"
            "  border-radius: 4px;"
            "}"
            "QWidget#target_card QLabel {"
            "  color: #222; background: transparent;"
            "}"
            "QWidget#target_card QPushButton {"
            "  color: #222; background: #e0e0e0;"
            "  border: 1px solid #bbb; border-radius: 3px;"
            "}"
            "QWidget#target_card QPushButton:hover {"
            "  background: #d0d0d0;"
            "}"
        )
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(6, 4, 6, 4)
        card_lay.setSpacing(2)

        # Header: label + data source + mask mode + move/remove buttons
        hdr_lay = QHBoxLayout()
        hdr_lay.setSpacing(3)
        lbl = QLabel(f"Target {t_idx + 1}:")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        src_combo = QComboBox()
        src_combo.addItems(["CL+CR", "CL-CR", "CL", "CR"])
        src_combo.setCurrentText(data_source if data_source in ("CL+CR", "CL-CR", "CL", "CR") else "CL+CR")
        src_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        src_combo.setFixedWidth(82)
        src_combo.setToolTip("Fourier amplitude used for this target")
        _set_combo_light_palette(src_combo)
        mask_combo = QComboBox()
        mask_combo.addItems(["support", "feature", "support+feature"])
        mask_combo.setCurrentText(mask_mode if mask_mode in ("support", "feature", "support+feature") else "support")
        mask_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        mask_combo.setFixedWidth(124)
        mask_combo.setToolTip(
            "support: exterior=0 constraint using group 0 only.\n"
            "feature: support + lock feature groups to reference values.\n"
            "support+feature: expanded support (all groups) + feature locking."
        )
        _set_combo_light_palette(mask_combo)
        up_btn = QPushButton("↑"); up_btn.setFixedSize(22, 22)
        dn_btn = QPushButton("↓"); dn_btn.setFixedSize(22, 22)
        rm_btn = QPushButton("✕"); rm_btn.setFixedSize(22, 22)
        hdr_lay.addWidget(lbl)
        hdr_lay.addWidget(src_combo)
        hdr_lay.addWidget(mask_combo)
        hdr_lay.addStretch()
        hdr_lay.addWidget(up_btn)
        hdr_lay.addWidget(dn_btn)
        hdr_lay.addWidget(rm_btn)
        card_lay.addLayout(hdr_lay)

        # Steps sub-layout (indented)
        steps_inner = QWidget()
        steps_lay = QVBoxLayout(steps_inner)
        steps_lay.setContentsMargins(0, 0, 0, 0)
        steps_lay.setSpacing(2)
        steps_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        card_lay.addWidget(steps_inner)

        add_step_btn = QPushButton("+ Add Step")
        add_step_btn.setFixedHeight(22)
        card_lay.addWidget(add_step_btn)

        entry = {
            'card':         card,
            'lbl':          lbl,
            'kind':         'target',
            'src_combo':    src_combo,
            'mask_combo':   mask_combo,
            'steps_inner':  steps_inner,
            'steps_lay':    steps_lay,
            'step_widgets': [],
        }
        self._pipeline_targets.append(entry)

        rm_btn.clicked.connect(lambda: self._remove_pipeline_target(entry))
        up_btn.clicked.connect(lambda: self._move_pipeline_target(entry, -1))
        dn_btn.clicked.connect(lambda: self._move_pipeline_target(entry, +1))
        add_step_btn.clicked.connect(lambda: self._add_target_step(entry))

        self._pipeline_lay.addWidget(card)

        for algo, n in steps:
            self._add_target_step(entry, algo, n)

        self._renumber_targets()

    def _add_interval_step(
        self,
        operation: str = "none",
        beta_schedule: str = "arctan (cold-start only)",
        bg_sub: bool = False,
    ) -> None:
        card = QWidget()
        card.setObjectName("interval_card")
        card.setStyleSheet(
            "QWidget#interval_card {"
            "  background: #eef2f5;"
            "  border: 1px dashed #8a9ba8;"
            "  border-radius: 4px;"
            "}"
            "QWidget#interval_card QLabel {"
            "  color: #222; background: transparent;"
            "}"
            "QWidget#interval_card QPushButton {"
            "  color: #222; background: #dce5eb;"
            "  border: 1px solid #aab8c2; border-radius: 3px;"
            "}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)
        lbl = QLabel("Interval:")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        op_combo = QComboBox()
        op_combo.addItems(CDI_INTERVAL_STEPS)
        op_combo.setCurrentText(operation if operation in CDI_INTERVAL_STEPS else "none")
        op_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        op_combo.setFixedWidth(124)
        op_combo.setToolTip(
            "Processing applied before the next target.\n"
            "match data amplitude: regression-scale current result to next target data.\n"
            "normalize: scale by total target intensity.\n"
            "reset support: restart from the support seed.\n"
            "random phase: restart from random Fourier phase."
        )
        _set_combo_light_palette(op_combo)
        beta_combo = QComboBox()
        beta_combo.addItems(["arctan (cold-start only)", "arctan", "constant"])
        beta_combo.setCurrentText(
            beta_schedule
            if beta_schedule in ("arctan (cold-start only)", "arctan", "constant")
            else "arctan (cold-start only)"
        )
        beta_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        beta_combo.setFixedWidth(186)
        beta_combo.setToolTip("Beta schedule used by the next target.")
        _set_combo_light_palette(beta_combo)
        bg_chk = QCheckBox("bg-sub")
        bg_chk.setChecked(bool(bg_sub))
        bg_chk.setToolTip("Subtract 5th-percentile background before the next target.")
        up_btn = QPushButton("↑"); up_btn.setFixedSize(22, 22)
        dn_btn = QPushButton("↓"); dn_btn.setFixedSize(22, 22)
        rm_btn = QPushButton("✕"); rm_btn.setFixedSize(22, 22)
        beta_lbl = QLabel("β:")
        beta_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Row 0: label + operation + bg-sub + move/remove buttons
        row0 = QHBoxLayout()
        row0.setContentsMargins(0, 0, 0, 0)
        row0.setSpacing(3)
        row0.addWidget(lbl)
        row0.addWidget(op_combo)
        row0.addWidget(bg_chk)
        row0.addStretch()
        row0.addWidget(up_btn)
        row0.addWidget(dn_btn)
        row0.addWidget(rm_btn)
        lay.addLayout(row0)

        # Row 1: beta label + beta schedule combo (independent column widths)
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(3)
        row1.addWidget(beta_lbl)
        row1.addWidget(beta_combo)
        row1.addStretch()
        lay.addLayout(row1)

        entry = {
            'card': card,
            'lbl': lbl,
            'kind': 'interval',
            'op_combo': op_combo,
            'beta_combo': beta_combo,
            'beta_lbl': beta_lbl,
            'bg_chk': bg_chk,
        }
        self._pipeline_targets.append(entry)
        rm_btn.clicked.connect(lambda: self._remove_pipeline_target(entry))
        up_btn.clicked.connect(lambda: self._move_pipeline_target(entry, -1))
        dn_btn.clicked.connect(lambda: self._move_pipeline_target(entry, +1))
        self._pipeline_lay.addWidget(card)
        self._renumber_targets()

    def _remove_pipeline_target(self, entry: dict) -> None:
        if entry in self._pipeline_targets:
            self._pipeline_targets.remove(entry)
        entry['card'].deleteLater()
        self._renumber_targets()

    def _move_pipeline_target(self, entry: dict, direction: int) -> None:
        if entry not in self._pipeline_targets:
            return
        idx = self._pipeline_targets.index(entry)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self._pipeline_targets):
            return
        self._pipeline_targets[idx], self._pipeline_targets[new_idx] = (
            self._pipeline_targets[new_idx], self._pipeline_targets[idx]
        )
        for t in self._pipeline_targets:
            self._pipeline_lay.removeWidget(t['card'])
        for t in self._pipeline_targets:
            self._pipeline_lay.addWidget(t['card'])
        self._renumber_targets()

    def _renumber_targets(self) -> None:
        target_idx = 1
        interval_idx = 1
        # Target and interval labels are sized in independent groups so that the
        # wider "Interval N:" text never inflates the "Target N:" label (and thus
        # the target card's header/step indent) when an interval is added.
        target_labels = []
        interval_labels = []
        for t in self._pipeline_targets:
            if t.get('kind') == 'interval':
                t['lbl'].setText(f"Interval {interval_idx}:")
                interval_labels.append(t['lbl'])
                if 'beta_lbl' in t:
                    interval_labels.append(t['beta_lbl'])
                interval_idx += 1
            else:
                t['lbl'].setText(f"Target {target_idx}:")
                target_labels.append(t['lbl'])
                target_idx += 1

        def _fit(labels: list) -> int:
            if not labels:
                return 0
            w = max(lbl.fontMetrics().horizontalAdvance(lbl.text()) for lbl in labels) + 4
            for lbl in labels:
                lbl.setFixedWidth(w)
            return w

        target_label_width = _fit(target_labels)
        _fit(interval_labels)
        for t in self._pipeline_targets:
            if t.get('kind') == 'target' and 'steps_lay' in t:
                t['steps_lay'].setContentsMargins(target_label_width + 3, 0, 0, 0)

        self._equalize_card_widths()

    def _equalize_card_widths(self) -> None:
        """Give every target/interval card the same minimal width.

        Each card is pinned to a non-stretching fixed width equal to the widest
        card's natural content width, so target and interval cards line up and
        no horizontal panel space is wasted.
        """
        cards = [t['card'] for t in self._pipeline_targets]
        if not cards:
            return
        for card in cards:
            # Stop cards from expanding to fill the panel; size to their content.
            card.setSizePolicy(QSizePolicy.Policy.Fixed, card.sizePolicy().verticalPolicy())
            card.setMaximumWidth(16777215)
        common = max(card.layout().sizeHint().width() for card in cards)
        for card in cards:
            card.setFixedWidth(common)

    def _add_target_step(
        self, entry: dict, algo: str = "raar", n: int = 100
    ) -> None:
        row_w = QWidget()
        row_lay = QHBoxLayout(row_w)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(4)

        cb = QComboBox()
        cb.addItems(CDI_ALGORITHM_LABELS.values())
        cb.setCurrentText(CDI_ALGORITHM_LABELS.get(algo, CDI_ALGORITHM_LABELS["raar"]))
        cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        _set_combo_light_palette(cb)

        sp = QSpinBox()
        sp.setRange(1, 99999)
        sp.setValue(n)
        sp.setSuffix(" iter")
        sp.setFixedWidth(120)
        _set_widget_light_palette(sp)

        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(22, 22)
        rm_btn.clicked.connect(lambda: self._remove_target_step(entry, row_w))

        row_lay.addWidget(cb)
        row_lay.addStretch()
        row_lay.addWidget(sp)
        row_lay.addWidget(rm_btn)

        entry['steps_lay'].addWidget(row_w)
        entry['step_widgets'].append((cb, sp, rm_btn))

    def _remove_target_step(self, entry: dict, row_w: QWidget) -> None:
        for i, (_, __, btn) in enumerate(entry['step_widgets']):
            if btn.parent() is row_w:
                entry['steps_lay'].removeWidget(row_w)
                row_w.deleteLater()
                del entry['step_widgets'][i]
                return

    def _collect_pipeline(self) -> list:
        """Return target and interval nodes from the UI."""
        result = []
        pending_beta = "arctan (cold-start only)"
        pending_bg_sub = False
        for t in self._pipeline_targets:
            if t.get('kind') == 'interval':
                pending_beta = t['beta_combo'].currentText()
                pending_bg_sub = t['bg_chk'].isChecked()
                result.append({
                    'kind': 'interval',
                    'operation': t['op_combo'].currentText(),
                    'beta_schedule': pending_beta,
                    'bg_sub': pending_bg_sub,
                })
                continue
            steps = [
                (
                    CDI_ALGORITHM_DISPLAY_TO_KEY.get(cb.currentText(), cb.currentText()),
                    sp.value(),
                )
                for cb, sp, _ in t['step_widgets']
            ]
            result.append({
                'kind':        'target',
                'data_source': t['src_combo'].currentText(),
                'mask_mode':   t['mask_combo'].currentText(),
                'beta_schedule': pending_beta,
                'bg_sub': pending_bg_sub,
                'steps':       steps,
            })
            pending_beta = "arctan (cold-start only)"
            pending_bg_sub = False
        return result

    @staticmethod
    def _pipeline_uses_mine(pipeline: list) -> bool:
        return any(
            algo == "mine"
            for target in pipeline
            for algo, _ in target.get('steps', [])
        )

    def _start_classic_reconstruction(self) -> None:
        """Compatibility wrapper for older internal callers."""
        self._load_classic_pipeline_preset()
        self._start_reconstruction()

    def _start_mine_reconstruction(self, pipeline: list, support: np.ndarray) -> None:
        """Run a configurable tutorial/mine pipeline from the target list."""
        if self._int_cl is None or self._int_cr is None:
            QMessageBox.warning(self, "No Data", "Load and align CL/CR data first.")
            return
        for idx, target in enumerate(pipeline, start=1):
            if target.get('kind', 'target') == 'interval':
                continue
            if target['data_source'] not in ("CL", "CR"):
                QMessageBox.warning(
                    self,
                    "Unsupported Target",
                    "The mine/Classic pipeline supports CL and CR targets only. "
                    f"Target {idx} uses {target['data_source']}."
                )
                return
            if self._single_dataset_mode and target['data_source'] == "CR":
                QMessageBox.warning(
                    self,
                    "Unsupported Target",
                    "Single-file mode has no CR target. Use CL, or use the default CL+CR target with RAAR/HIO/ER."
                )
                return
            for algo, _ in target['steps']:
                if algo != "mine":
                    QMessageBox.warning(
                        self,
                        "Mixed Algorithms",
                        "Mine/Classic steps cannot be mixed with ER/HIO/RAAR in one run."
                    )
                    return

        total_iters = sum(
            n
            for target in pipeline
            if target.get('kind', 'target') == 'target'
            for _, n in target['steps']
        )
        self._total_iters_per_restart = max(1, total_iters)
        self._progress_bar.setRange(0, total_iters)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting mine pipeline...")
        self._start_btn.setEnabled(False)
        self._classic_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._live_errs = []

        self._seq_worker = _CDISequentialReconWorker(
            cl_intensity = self._int_cl,
            cr_intensity = self._int_cr,
            pixel_mask   = self._pixel_mask,
            support      = support.astype(float),
            pipeline     = pipeline,
        )
        self._seq_worker.progress.connect(self._on_seq_progress)
        self._seq_worker.finished.connect(self._on_seq_finished)
        self._seq_worker.stopped.connect(self._on_recon_stopped)
        self._seq_worker.error.connect(self._on_recon_error)
        self._seq_worker.start()
        target_sources = [
            target['data_source']
            for target in pipeline
            if target.get('kind', 'target') == 'target'
        ]
        self._set_status(
            f"Mine pipeline running - targets={target_sources}, "
            f"{total_iters} iterations..."
        )

    def _start_classic_reconstruction_old(self) -> None:
        """Run the original sequential CL→CR pipeline (exact replica of old workflow)."""
        if not self._sync_from_fth_tool():
            return
        if self._int_cl is None or self._int_cr is None:
            QMessageBox.warning(self, "No Data", "Load and align CL/CR data first.")
            return
        support = self._current_support_for_reconstruction()
        if support is None or np.sum(support) == 0:
            QMessageBox.warning(
                self, "No Support",
                "Define one or more support shapes in the Filter & FTH / Support tab first."
            )
            return

        total_iters = 700 + 50 + 50
        self._progress_bar.setRange(0, total_iters)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting Classic CL→CR…")
        self._start_btn.setEnabled(False)
        self._classic_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._live_errs = []

        self._seq_worker = _CDISequentialReconWorker(
            cl_intensity = self._int_cl,
            cr_intensity = self._int_cr,
            pixel_mask   = self._pixel_mask,
            support      = support.astype(float),
        )
        self._seq_worker.progress.connect(self._on_seq_progress)
        self._seq_worker.finished.connect(self._on_seq_finished)
        self._seq_worker.stopped.connect(self._on_recon_stopped)
        self._seq_worker.error.connect(self._on_recon_error)
        self._seq_worker.start()
        self._set_status("Classic CL→CR pipeline running — CL 700 arctan → CL-refine 50 → CR 50…")

    def _start_reconstruction(self) -> None:
        if not self._sync_from_fth_tool():
            return

        pipeline = self._collect_pipeline()
        if not pipeline:
            QMessageBox.warning(self, "No Targets",
                                "Add at least one target to the pipeline.")
            return
        target_count = sum(1 for tgt in pipeline if tgt.get('kind', 'target') == 'target')
        if target_count == 0:
            QMessageBox.warning(self, "No Targets",
                                "Add at least one target to the pipeline.")
            return
        for i, tgt in enumerate(pipeline):
            if tgt.get('kind', 'target') == 'interval':
                continue
            if not tgt['steps']:
                QMessageBox.warning(self, "No Steps",
                                    f"Target {i + 1} has no algorithm steps.")
                return

        support = self._current_support_for_reconstruction()
        if support is None or np.sum(support) == 0:
            QMessageBox.warning(
                self, "No Support",
                "Define one or more support shapes in the Filter & FTH / Support tab first."
            )
            return

        if self._pipeline_uses_mine(pipeline):
            self._start_mine_reconstruction(pipeline, support.astype(float))
            return

        initial_obj = self._resolve_initial_object(support.astype(float))
        if initial_obj is False:
            return

        # Resolve each pipeline target into concrete arrays for the worker
        resolved = []
        for tgt in pipeline:
            if tgt.get('kind', 'target') == 'interval':
                resolved.append({
                    'kind': 'interval',
                    'operation': tgt.get('operation', 'none'),
                })
                continue
            amp = self._resolve_amplitude_for_source(tgt['data_source'])
            if amp is None:
                QMessageBox.warning(self, "No CL/CR Data",
                                    "Load and align CL/CR data first.")
                return
            supp, feat = self._resolve_target_constraints(
                tgt['mask_mode'],
                initial_obj.astype(np.complex128) if initial_obj is not None else None,
            )
            if supp is None:
                QMessageBox.warning(self, "No Support",
                                    "Define support shapes first.")
                return
            # Map UI label to internal key used by the worker
            _sched_map = {
                "arctan (cold-start only)": "arctan_cold",
                "arctan":                   "arctan",
                "constant":                 "constant",
            }
            beta_sched = _sched_map.get(tgt.get('beta_schedule'), "arctan_cold")
            resolved.append({
                'kind':          'target',
                'data_source':   tgt['data_source'],
                'amp':           amp,
                'support':       supp.astype(float),
                'steps':         tgt['steps'],
                'feat':          feat,
                'beta_schedule': beta_sched,
                'bg_sub':        bool(tgt.get('bg_sub', False)),
            })

        restarts    = self._restarts_spin.value()
        iters_per   = sum(
            n
            for tgt in pipeline
            if tgt.get('kind', 'target') == 'target'
            for _, n in tgt['steps']
        )
        total_iters = iters_per * restarts
        emit_every  = max(1, iters_per // 50)
        self._total_iters_per_restart = iters_per

        self._progress_bar.setRange(0, total_iters)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting…")
        self._start_btn.setEnabled(False)
        self._classic_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._live_errs = []

        self._recon_worker = _CDIMultiTargetWorker(
            targets               = resolved,
            beta                  = self._beta_spin.value(),
            restarts              = restarts,
            sw_enabled            = self._sw_group.isChecked(),
            sw_sigma              = self._sw_sigma_spin.value(),
            sw_threshold          = self._sw_thresh_spin.value(),
            sw_every              = self._sw_every_spin.value(),
            floating_mask         = self._pixel_mask,
            initial_obj           = initial_obj,
            emit_every            = emit_every,
            scale_initial_guess   = self._scale_init_chk.isChecked(),
            scale_inherited_phase = self._scale_inherited_chk.isChecked(),
        )
        self._recon_worker.progress.connect(self._on_progress)
        self._recon_worker.restart_done.connect(self._on_restart_done)
        self._recon_worker.finished.connect(self._on_recon_finished)
        self._recon_worker.error.connect(self._on_recon_error)
        self._recon_worker.start()
        self._set_status(
            f"CDI pipeline running — {target_count} target(s), {restarts} restart(s)…"
        )

    def _resolve_initial_object(self, support: np.ndarray):
        """Return initial complex object, None (random), or False (cancelled)."""
        mode = self._initial_guess_combo.currentText()
        if mode.startswith("Random"):
            return None
        if mode.startswith("Previous"):
            if self._result_obj is None:
                QMessageBox.warning(
                    self, "No Previous Result",
                    "Run CDI once first, or choose 'Support FFT phase' or 'Random'."
                )
                return False
            ref_shape = None
            if self._amp_cl is not None:
                ref_shape = self._amp_cl.shape
            if ref_shape is not None and self._result_obj.shape != ref_shape:
                QMessageBox.warning(
                    self, "Shape Mismatch",
                    f"Previous result shape {self._result_obj.shape} != "
                    f"current data {ref_shape}."
                )
                return False
            return self._result_obj.copy()
        return self._compute_support_fft_initial_object(support)

    def _resolve_amplitude_for_source(self, data_source: str) -> Optional[np.ndarray]:
        """Return CDI Fourier amplitude for the given data source."""
        if self._amp_cl is None or self._amp_cr is None:
            return None
        if data_source == "CL":
            return self._amp_cl.copy()
        elif data_source == "CR":
            if self._single_dataset_mode:
                return None
            return self._amp_cr.copy()
        elif data_source == "CL+CR":
            if self._single_dataset_mode:
                return self._amp_cl.copy()
            return (self._amp_cl + self._amp_cr).copy()
        else:  # CL-CR
            if self._single_dataset_mode:
                return self._amp_cl.copy()
            return np.abs(self._amp_cl - self._amp_cr)

    def _resolve_target_constraints(
        self,
        mask_mode: str,
        initial_obj: Optional[np.ndarray],
    ) -> tuple:
        """Return (support_array, feature_constraints) for a given mask_mode.

        mask_mode="support": group 0 support only, no feature locking.
        mask_mode="feature": group 0 support + groups 1+ locked to reference values.
        mask_mode="support+feature": expanded support (groups 0∪1+) + feature locking.
        """
        support = self._current_support_for_reconstruction()
        if support is None:
            return None, []

        feat: list = []
        if mask_mode in ("feature", "support+feature"):
            if initial_obj is not None and len(self._mask_groups) > 1:
                feat = self._collect_feature_constraints(initial_obj)
            if mask_mode == "support+feature":
                combined = support.copy()
                for i in range(1, len(self._mask_groups)):
                    grp = self._compute_group_mask(i)
                    if grp is not None and grp.shape == combined.shape:
                        combined = np.maximum(combined, grp)
                support = combined

        return support, feat

    def _compute_support_fft_initial_object(self, support: np.ndarray):
        """Build the CDI initial object from the phase of the support-mask FFT."""
        try:
            support = np.asarray(support, dtype=np.float64)
            start = _ifft2c(support)
            phase = np.angle(start)
            init  = support.astype(np.float64) * np.exp(1j * phase)
            if np.sum(np.abs(init)) == 0:
                raise ValueError("Support is empty after applying support FFT phase.")
            self._live_amp_img.setImage(np.abs(start))
            _apply_colormap(self._live_amp_img, "gray")
            self._set_status("Using support FFT phase as CDI initial guess.")
            return init.astype(np.complex128)
        except Exception as exc:
            QMessageBox.critical(self, "Support FFT Seed Error", str(exc))
            return False

    def _stop_reconstruction(self) -> None:
        if self._recon_worker:
            self._recon_worker.request_stop()
        if self._seq_worker:
            self._seq_worker.request_stop()
        self._start_btn.setEnabled(False)
        self._classic_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._set_status("Stop requested — finishing current iteration…")

    def _on_progress(
        self, restart: int, iteration: int, err: float, psi: np.ndarray
    ) -> None:
        global_iter = restart * self._total_iters_per_restart + iteration
        self._progress_bar.setValue(min(global_iter, self._progress_bar.maximum()))
        self._progress_label.setText(
            f"Restart {restart + 1}/{self._restarts_spin.value()},  "
            f"iter {iteration},  error {err:.4e}"
        )
        self._live_amp_img.setImage(np.abs(psi))
        _apply_colormap(self._live_amp_img, "gray")
        self._live_errs.append(err)
        self._live_err_curve.setData(self._live_errs)

    def _on_seq_progress(
        self, stage: str, done: int, total: int, err: float, current: np.ndarray
    ) -> None:
        self._progress_bar.setValue(min(done, self._progress_bar.maximum()))
        self._progress_label.setText(f"{stage}, iter {done}/{total}, error {err:.3f} dB")
        self._live_amp_img.setImage(np.abs(current))
        _apply_colormap(self._live_amp_img, "gray")
        self._live_errs.append(err)
        self._live_err_curve.setData(self._live_errs)

    def _on_restart_done(self, restart: int, final_error: float) -> None:
        self._set_status(
            f"Restart {restart + 1} finished — final error {final_error:.4e}"
        )

    def _on_seq_finished(
        self,
        retrieved_cl: np.ndarray,
        retrieved_cr: np.ndarray,
        diff: np.ndarray,
        recon: np.ndarray,
        errors_cl: list,
        errors_cr: list,
    ) -> None:
        self._result_cl = retrieved_cl
        self._result_cr = retrieved_cr
        self._result_diff = diff
        self._result_obj = recon
        self._result_errs = list(errors_cl)
        self._result_errs_cr = list(errors_cr)
        self._start_btn.setEnabled(True)
        self._classic_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._seq_worker = None
        self._progress_bar.setValue(self._progress_bar.maximum())
        final_cl = f"{errors_cl[-1]:.3f} dB" if errors_cl else "n/a"
        final_cr = f"{errors_cr[-1]:.3f} dB" if errors_cr else "n/a"
        self._progress_label.setText(f"Done - CL {final_cl}, CR {final_cr}")
        self._refresh_results_display()
        self._tabs.setCurrentIndex(3)
        has_cl = retrieved_cl is not None
        has_cr = retrieved_cr is not None
        has_diff = diff is not None
        self._set_status(
            f"Classic CL→CR complete — CL {final_cl} ({has_cl}), "
            f"CR {final_cr} ({has_cr}), CL−CR ({has_diff})"
        )

    def _on_recon_stopped(self) -> None:
        self._start_btn.setEnabled(True)
        self._classic_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._recon_worker = None
        self._seq_worker = None
        self._progress_label.setText("Stopped.")
        self._set_status("CDI stopped. Ready to start again.")

    def _on_recon_finished(
        self,
        best_obj: Optional[np.ndarray],
        errors: list,
        source_results: Optional[dict] = None,
    ) -> None:
        source_results = source_results or {}
        self._result_cl = source_results.get("CL")
        self._result_cr = source_results.get("CR")
        if self._result_cl is not None and self._result_cr is not None:
            self._result_diff = self._result_cl - self._result_cr
            self._result_obj = _ifft2c(self._result_diff)
        else:
            self._result_diff = None
            self._result_obj = best_obj
        self._result_errs_cr = []
        self._result_errs = errors
        self._start_btn.setEnabled(True)
        self._classic_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._recon_worker = None
        self._progress_bar.setValue(self._progress_bar.maximum())

        if best_obj is not None:
            final = f"{errors[-1]:.4e}" if errors else "n/a"
            self._progress_label.setText(f"Done — final error {final}")
            self._refresh_results_display()
            self._tabs.setCurrentIndex(3)
            has_cl = self._result_cl is not None
            has_cr = self._result_cr is not None
            has_diff = self._result_diff is not None
            self._set_status(
                f"CDI complete — final error {final}, "
                f"CL ({has_cl}), CR ({has_cr}), CL−CR ({has_diff})"
            )
        else:
            self._on_recon_stopped()

    def _on_recon_error(self, msg: str) -> None:
        self._start_btn.setEnabled(True)
        self._classic_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._recon_worker = None
        self._seq_worker = None
        self._set_status(f"Reconstruction error: {msg}")
        QMessageBox.critical(self, "Reconstruction Error", msg)

    # ==================================================================
    # Results display & export
    # ==================================================================

    def _update_channel_combo_state(self) -> None:
        """Enable/disable CL, CR, CL−CR items based on available results."""
        if not hasattr(self, "_res_channel_combo"):
            return
        has_final = self._result_obj is not None
        has_cl = self._result_cl is not None
        has_cr = self._result_cr is not None
        model = self._res_channel_combo.model()
        # indices: 0=Final, 1=CL, 2=CR, 3=CL−CR
        for idx, enabled in [
            (0, has_final),
            (1, has_cl),
            (2, has_cr),
            (3, has_cl and has_cr),
        ]:
            item = model.item(idx)
            if item is not None:
                item.setEnabled(bool(enabled))
        # If current selection is now disabled, switch to the first available channel.
        cur = self._res_channel_combo.currentIndex()
        if cur >= 0 and model.item(cur) is not None and not model.item(cur).isEnabled():
            for idx in range(self._res_channel_combo.count()):
                item = model.item(idx)
                if item is not None and item.isEnabled():
                    self._res_channel_combo.setCurrentIndex(idx)
                    break

    def _resolve_display_field(self) -> Optional[np.ndarray]:
        """Return the complex image-space field for the currently selected channel."""
        channel = self._res_channel_combo.currentText() if hasattr(self, "_res_channel_combo") else "Final result"
        if channel == "Final result":
            return self._result_obj
        if channel == "CL":
            if self._result_cl is None:
                return None
            return _ifft2c(self._result_cl)
        if channel == "CR":
            if self._result_cr is None:
                return None
            return _ifft2c(self._result_cr)
        if channel == "CL − CR":
            if self._result_cl is None or self._result_cr is None:
                return None
            return _ifft2c(self._result_cl - self._result_cr)
        return None

    def _refresh_results_display(self) -> None:
        self._update_channel_combo_state()
        psi = self._resolve_display_field()
        if psi is None:
            return
        view_name = self._res_view_combo.currentText() if hasattr(self, "_res_view_combo") else "Abs amplitude"
        if view_name.startswith("Real"):
            result_view = np.real(psi)
            result_title = "Real part"
        elif view_name.startswith("Imag"):
            result_view = np.imag(psi)
            result_title = "Imaginary part"
        elif view_name.startswith("Phase"):
            result_view = np.angle(psi)
            result_title = "Phase"
        else:
            result_view = np.abs(psi)
            result_title = "Abs amplitude"

        channel = self._res_channel_combo.currentText() if hasattr(self, "_res_channel_combo") else "Final result"
        panel_title = f"{result_title}  [{channel}]"

        self._res_amp_plot.setTitle(panel_title)
        self._res_amp_img.setImage(result_view)
        _apply_colormap(self._res_amp_img, self._res_amp_cmap.currentText())
        self._ensure_result_roi_for_shape(result_view.shape)

        self._res_phase_img.setImage(np.angle(psi))
        _apply_colormap(self._res_phase_img, self._res_phase_cmap.currentText())

        # Bottom-left: log|FFT|² of the displayed field (or stored diff for CL−CR)
        if channel == "CL − CR" and self._result_diff is not None:
            log_diff = np.log10(np.abs(self._result_diff) ** 2 + 1.0)
        else:
            log_diff = np.log10(np.abs(_fft2c(psi)) ** 2 + 1.0)
        self._res_diff_img.setImage(log_diff)
        _apply_colormap(self._res_diff_img, "inferno")

        if self._result_errs:
            errs = list(self._result_errs)
            if self._result_errs_cr:
                errs = errs + list(self._result_errs_cr)
            self._res_err_curve.setData(errs)

    def _export_result_npy(self) -> None:
        if self._result_obj is None:
            QMessageBox.warning(self, "No Result", "Run CDI reconstruction first.")
            return
        if self._result_diff is not None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Tutorial Reconstruction", "", "NumPy archive (*.npz)"
            )
            if path:
                np.savez(
                    path,
                    reconstructed=self._result_obj,
                    retrieved_cl=self._result_cl,
                    retrieved_cr=self._result_cr,
                    retrieved_diff=self._result_diff,
                    support=self._support,
                    errors_cl=np.asarray(self._result_errs),
                    errors_cr=np.asarray(self._result_errs_cr),
                )
                self._set_status(f"Saved tutorial reconstruction -> {path}")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Reconstruction", "", "NumPy (*.npy)"
        )
        if path:
            np.save(path, self._result_obj)
            self._set_status(f"Saved reconstruction → {path}")

    def _on_result_roi_toggled(self, checked: bool) -> None:
        if not hasattr(self, "_res_roi"):
            return
        psi = self._resolve_display_field()
        if psi is not None:
            self._ensure_result_roi_for_shape(psi.shape)
        self._res_roi.setVisible(bool(checked))
        self._set_status("CDI result ROI enabled." if checked else "CDI result ROI disabled.")

    def _reset_result_roi(self) -> None:
        psi = self._resolve_display_field()
        if psi is None:
            QMessageBox.warning(self, "No Result", "Run CDI reconstruction first.")
            return
        self._res_roi_shape = None
        self._ensure_result_roi_for_shape(psi.shape)
        self._res_roi_btn.setChecked(True)
        self._res_roi.setVisible(True)
        self._set_status("Reset CDI result ROI.")

    def _ensure_result_roi_for_shape(self, shape: tuple[int, int]) -> None:
        if not hasattr(self, "_res_roi"):
            return
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            return
        if self._res_roi_shape == (h, w):
            return
        roi_w = max(1, w // 2)
        roi_h = max(1, h // 2)
        self._res_roi.setPos((max(0, (w - roi_w) // 2), max(0, (h - roi_h) // 2)))
        self._res_roi.setSize((roi_w, roi_h))
        self._res_roi_shape = (h, w)

    @staticmethod
    def _roi_slices_from_pos_size(pos, size, shape: tuple[int, int]) -> tuple[slice, slice]:
        h, w = int(shape[0]), int(shape[1])
        x0 = int(np.floor(float(pos[0])))
        y0 = int(np.floor(float(pos[1])))
        x1 = int(np.ceil(float(pos[0]) + float(size[0])))
        y1 = int(np.ceil(float(pos[1]) + float(size[1])))
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return slice(y0, y1), slice(x0, x1)

    def _current_result_roi_slices(self, shape: tuple[int, int]) -> Optional[tuple[slice, slice]]:
        if not hasattr(self, "_res_roi_btn") or not self._res_roi_btn.isChecked():
            return None
        pos = self._res_roi.pos()
        size = self._res_roi.size()
        return self._roi_slices_from_pos_size(
            (pos.x(), pos.y()),
            (size.x(), size.y()),
            shape,
        )

    def _get_cdi_export_components(self) -> Optional[dict[str, np.ndarray]]:
        psi = self._resolve_display_field()
        if psi is None:
            QMessageBox.warning(self, "No Result", "Run CDI reconstruction first.")
            return None
        arr = np.asarray(psi)
        if arr.ndim != 2:
            QMessageBox.warning(self, "Invalid Result", "CDI result must be a 2D array.")
            return None
        roi = self._current_result_roi_slices(arr.shape)
        if roi is not None:
            arr = arr[roi]
        return {
            "real": np.real(arr).astype(np.float32),
            "imag": np.imag(arr).astype(np.float32),
            "phase": np.angle(arr).astype(np.float32),
            "abs": np.abs(arr).astype(np.float32),
        }

    def _selected_cdi_export_component_name(self) -> str:
        txt = self._cdi_exp_target_combo.currentText().strip().lower()
        if txt.startswith("real"):
            return "real"
        if txt.startswith("imag"):
            return "imag"
        if txt.startswith("phase"):
            return "phase"
        return "abs"

    def _current_cdi_export_name_base(self) -> str:
        channel = "result"
        if hasattr(self, "_res_channel_combo"):
            channel = self._res_channel_combo.currentText().lower()
            channel = channel.replace("−", "-").replace(" ", "_")
        return f"cdi_{channel}"

    @staticmethod
    def _cdi_component_display_levels(arr: np.ndarray, name: str) -> tuple[float, float]:
        finite = np.asarray(arr, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        if name == "phase":
            return -float(np.pi), float(np.pi)
        if name in ("real", "imag"):
            vmax = float(np.percentile(np.abs(finite), 99.5))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            return -vmax, vmax if vmax > 0 else 1.0
        lo = 0.0
        hi = float(np.percentile(finite, 99.5))
        if not np.isfinite(hi) or hi <= lo:
            hi = float(np.max(finite)) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    @staticmethod
    def _component_to_qimage(arr: np.ndarray, levels: tuple[float, float]) -> QImage:
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

    def _copy_selected_cdi_component_jpeg(self) -> None:
        components = self._get_cdi_export_components()
        if components is None:
            return
        name = self._selected_cdi_export_component_name()
        arr = components[name]
        qimg = self._component_to_qimage(arr, self._cdi_component_display_levels(arr, name))
        try:
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            ok = qimg.save(buf, "JPEG", 95)
            buf.close()
            if not ok:
                self._set_status("Failed to encode CDI JPEG for clipboard.")
                return
            mime = QMimeData()
            mime.setImageData(qimg)
            mime.setData("image/jpeg", ba)
            QApplication.clipboard().setMimeData(mime)
            QApplication.clipboard().setPixmap(QPixmap.fromImage(qimg))
            self._set_status(f"Copied CDI JPEG: {name}")
        except Exception as exc:
            self._set_status(f"CDI copy failed: {exc}")
            logging.exception("CDI copy JPEG")

    def _save_selected_cdi_component(self) -> None:
        components = self._get_cdi_export_components()
        if components is None:
            return
        name = self._selected_cdi_export_component_name()
        arr = components[name]
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save CDI {name}",
            f"{self._current_cdi_export_name_base()}_{name}.png",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        p = pathlib.Path(path)
        try:
            qimg = self._component_to_qimage(arr, self._cdi_component_display_levels(arr, name))
            if not qimg.save(str(p)):
                raise RuntimeError("Image save failed")
            self._set_status(f"Saved CDI image: {p.name}")
        except Exception as exc:
            self._set_status(f"CDI save failed: {exc}")
            logging.exception("CDI save selected component")

    def _save_all_cdi_components(self) -> None:
        components = self._get_cdi_export_components()
        if components is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save all CDI components",
            f"{self._current_cdi_export_name_base()}.png",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        p = pathlib.Path(path)
        ext = p.suffix.lower() or ".png"
        try:
            stem = p.with_suffix("")
            for name, arr in components.items():
                qimg = self._component_to_qimage(arr, self._cdi_component_display_levels(arr, name))
                out_path = pathlib.Path(f"{stem}_{name}{ext}")
                if not qimg.save(str(out_path)):
                    raise RuntimeError(f"Image save failed: {out_path.name}")
            self._set_status(f"Saved CDI images: {stem.name}_real/imag/phase/abs{ext}")
        except Exception as exc:
            self._set_status(f"CDI save all failed: {exc}")
            logging.exception("CDI save all components")

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        log.info("CDI tool: %s", msg)

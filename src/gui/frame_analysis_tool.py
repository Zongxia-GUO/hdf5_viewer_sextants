"""Frame-by-frame profile fitting — I(r) and I(θ) parameter tracking."""

from __future__ import annotations

import csv

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.recon import profiles as _profiles


# ─────────────────────────────────────────────────────────────────── #
# Background worker                                                    #
# ─────────────────────────────────────────────────────────────────── #

class _FitWorker(QThread):
    """Fit a model to each frame's I(r) or I(θ) profile."""

    # (frame_idx, popt_list_or_nan)
    progress = pyqtSignal(int, list)
    finished = pyqtSignal()

    def __init__(
        self,
        data3d: np.ndarray,
        cx: float, cy: float,
        r_min: int, r_max: int,
        a_min: int, a_max: int,
        func,
        p0: list[float],
        fit_lo: float,
        fit_hi: float,
        mode: str,    # 'radial' | 'angular'
        chain: bool,
        n_bins: int = 360,
    ) -> None:
        super().__init__()
        self._data3d = data3d
        self._cx, self._cy = cx, cy
        self._r_min, self._r_max = r_min, r_max
        self._a_min, self._a_max = a_min, a_max
        self._func = func
        self._p0 = list(p0)
        self._fit_lo, self._fit_hi = fit_lo, fit_hi
        self._mode = mode
        self._chain = chain
        self._n_bins = n_bins
        self._stop_flag = False
        self.results: list[list[float]] = []

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        from scipy.optimize import curve_fit

        p0 = list(self._p0)
        nan_row = [float("nan")] * len(p0)

        for i in range(self._data3d.shape[0]):
            if self._stop_flag:
                break
            frame = self._data3d[i]
            try:
                if self._mode == "radial":
                    r_bins, intensity = _profiles.radial_profile(
                        frame, self._cx, self._cy,
                        self._r_min, self._r_max,
                        self._a_min, self._a_max,
                    )
                    x = np.asarray(r_bins, dtype=np.float64)
                    y = np.asarray(intensity, dtype=np.float64)
                else:
                    x, y = _profiles.angular_profile(
                        frame, self._cx, self._cy,
                        self._r_min, self._r_max,
                        n_bins=self._n_bins,
                    )
                    x = np.asarray(x, dtype=np.float64)
                    y = np.asarray(y, dtype=np.float64)

                mask = (
                    (x >= self._fit_lo) & (x <= self._fit_hi)
                    & np.isfinite(x) & np.isfinite(y)
                )
                if mask.sum() < 3:
                    raise ValueError("too few valid points in fit range")

                popt, _ = curve_fit(
                    self._func, x[mask], y[mask], p0=p0, maxfev=20000
                )
                if self._chain:
                    p0 = list(popt)
                result = [float(v) for v in popt]
            except Exception:
                result = list(nan_row)

            self.results.append(result)
            self.progress.emit(i, result)

        self.finished.emit()


# ─────────────────────────────────────────────────────────────────── #
# Per-mode panel                                                       #
# ─────────────────────────────────────────────────────────────────── #

class _ProfileFitPanel(QWidget):
    """Controls + live parameter-vs-frame plots for one profile mode."""

    def __init__(
        self,
        mode: str,               # 'radial' | 'angular'
        n_frames: int,
        data3d: np.ndarray,
        cx: float, cy: float,
        r_min: int, r_max: int,
        a_min: int, a_max: int,
        fit_models: dict,
        default_model: str = "",
        default_p0: list[float] | None = None,
        default_fit_lo: float = 0.0,
        default_fit_hi: float = 100.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._n_frames = n_frames
        self._data3d = data3d
        self._cx, self._cy = cx, cy
        self._r_min, self._r_max = r_min, r_max
        self._a_min, self._a_max = a_min, a_max
        self._fit_models = fit_models
        self._default_model = default_model
        self._default_p0: list[float] = default_p0 or []
        self._default_fit_lo = default_fit_lo
        self._default_fit_hi = default_fit_hi

        self._worker: _FitWorker | None = None
        self._results: list[list[float]] = []
        self._param_names: list[str] = []
        self._fit_param_spins: list[QDoubleSpinBox] = []
        self._param_plots: list = []
        self._param_curves: list = []

        self._build_ui()

    # ---------------------------------------------------------------- #
    # UI                                                                 #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        h_split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(h_split)

        # ── Left controls ── #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(360)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(6)

        # Model selector
        g_model = QGroupBox("Modèle")
        fl_m = QFormLayout(g_model)
        fl_m.setContentsMargins(6, 4, 6, 4)
        fl_m.setSpacing(3)
        self._combo_model = QComboBox()
        for name in self._fit_models:
            self._combo_model.addItem(name)
        if self._default_model and self._default_model in self._fit_models:
            self._combo_model.setCurrentText(self._default_model)
        self._combo_model.currentIndexChanged.connect(self._on_model_changed)
        fl_m.addRow("Modèle:", self._combo_model)
        cl.addWidget(g_model)

        # Initial params (dynamic)
        self._g_params = QGroupBox("Paramètres initiaux")
        self._fl_params = QFormLayout(self._g_params)
        self._fl_params.setContentsMargins(6, 4, 6, 4)
        self._fl_params.setSpacing(3)
        btn_auto = QPushButton("Auto-guess (frame 0)")
        btn_auto.setAutoDefault(False)
        btn_auto.setDefault(False)
        btn_auto.clicked.connect(self._on_auto_guess)
        self._fl_params.addRow("", btn_auto)
        cl.addWidget(self._g_params)

        # Fit range
        g_range = QGroupBox("Plage de fit")
        fl_r = QFormLayout(g_range)
        fl_r.setContentsMargins(6, 4, 6, 4)
        fl_r.setSpacing(3)
        row_rng = QHBoxLayout()
        self._spin_lo = QSpinBox()
        self._spin_hi = QSpinBox()
        if self._mode == "angular":
            self._spin_lo.setRange(0, 360)
            self._spin_hi.setRange(0, 360)
            unit = "θ (°):"
        else:
            self._spin_lo.setRange(0, 99999)
            self._spin_hi.setRange(0, 99999)
            unit = "r (px):"
        self._spin_lo.setValue(int(self._default_fit_lo))
        self._spin_hi.setValue(int(self._default_fit_hi))
        row_rng.addWidget(self._spin_lo)
        row_rng.addWidget(QLabel("—"))
        row_rng.addWidget(self._spin_hi)
        fl_r.addRow(unit, row_rng)
        cl.addWidget(g_range)

        # Options
        g_opt = QGroupBox("Options")
        fl_o = QFormLayout(g_opt)
        fl_o.setContentsMargins(6, 4, 6, 4)
        self._chk_chain = QCheckBox("Chaîner les fits (p₀ ← résultat précédent)")
        self._chk_chain.setChecked(True)
        fl_o.addRow(self._chk_chain)
        cl.addWidget(g_opt)

        # Run / Stop
        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Lancer l'analyse")
        self._btn_run.setAutoDefault(False)
        self._btn_run.setDefault(False)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_stop = QPushButton("Arrêter")
        self._btn_stop.setAutoDefault(False)
        self._btn_stop.setDefault(False)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_stop)
        cl.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, self._n_frames)
        self._progress.setValue(0)
        cl.addWidget(self._progress)

        self._lbl_status = QLabel("Prêt")
        self._lbl_status.setWordWrap(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        self._lbl_status.setFont(mono)
        cl.addWidget(self._lbl_status)

        self._btn_export = QPushButton("Exporter CSV")
        self._btn_export.setAutoDefault(False)
        self._btn_export.setDefault(False)
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        cl.addWidget(self._btn_export)

        cl.addStretch(1)
        scroll.setWidget(ctrl)
        h_split.addWidget(scroll)

        # ── Right: one plot per parameter ── #
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("k")
        h_split.addWidget(self._glw)
        h_split.setSizes([300, 800])

        # Build spinboxes + plots for the initially selected model
        self._on_model_changed(0)

    # ---------------------------------------------------------------- #
    # Model management                                                   #
    # ---------------------------------------------------------------- #

    def _on_model_changed(self, _idx: int = 0) -> None:
        while self._fl_params.rowCount() > 1:
            self._fl_params.removeRow(1)
        self._fit_param_spins.clear()

        model_name = self._combo_model.currentText()
        params = self._fit_models[model_name]["params"]

        use_defaults = (
            self._default_p0
            and model_name == self._default_model
            and len(self._default_p0) == len(params)
        )

        for i, (pname, default) in enumerate(params):
            spin = QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setValue(self._default_p0[i] if use_defaults else default)
            self._fl_params.addRow(f"{pname}:", spin)
            self._fit_param_spins.append(spin)

        self._param_names = [n for n, _ in params]
        self._rebuild_plots()

    def _rebuild_plots(self) -> None:
        self._glw.clear()
        self._param_plots.clear()
        self._param_curves.clear()
        self._results.clear()
        self._progress.setValue(0)
        self._btn_export.setEnabled(False)

        unit = "°" if self._mode == "angular" else "px"
        x_label = f"θ ({unit})" if self._mode == "angular" else f"r ({unit})"

        for i, pname in enumerate(self._param_names):
            plot = self._glw.addPlot(row=i, col=0)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", pname)
            if i == len(self._param_names) - 1:
                plot.setLabel("bottom", "Frame")
            else:
                plot.getAxis("bottom").setStyle(showValues=False)
            curve = plot.plot(
                [], [],
                pen=pg.mkPen((100, 200, 255), width=1.5),
                symbol="o", symbolSize=4,
                symbolBrush=pg.mkBrush((100, 200, 255)),
                connect="finite",
            )
            self._param_plots.append(plot)
            self._param_curves.append(curve)

    # ---------------------------------------------------------------- #
    # Auto-guess                                                         #
    # ---------------------------------------------------------------- #

    def _on_auto_guess(self) -> None:
        if self._data3d is None or len(self._data3d) == 0:
            return
        frame = self._data3d[0]
        try:
            if self._mode == "radial":
                r_bins, intensity = _profiles.radial_profile(
                    frame, self._cx, self._cy,
                    self._r_min, self._r_max, self._a_min, self._a_max,
                )
                x = np.asarray(r_bins, dtype=np.float64)
                y = np.asarray(intensity, dtype=np.float64)
            else:
                x, y = _profiles.angular_profile(
                    frame, self._cx, self._cy,
                    self._r_min, self._r_max,
                )
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)

            lo, hi = float(self._spin_lo.value()), float(self._spin_hi.value())
            mask = (x >= lo) & (x <= hi) & np.isfinite(y)
            if mask.sum() > 2:
                x, y = x[mask], y[mask]

            amp    = float(np.nanmax(y) - np.nanmin(y)) or 1.0
            offset = float(np.nanmin(y))
            span   = float(x[-1] - x[0]) if len(x) > 1 else 50.0
            tau    = max(span / 3.0, 1.0)
            x0     = float(x[np.nanargmax(y)])
            sigma  = max(span / 6.0, 1.0)
            slope  = float((y[-1] - y[0]) / span) if span > 0 else 0.0
            ymid   = float(np.nanmean(y))

            lookup = {
                "A": amp, "A₁": amp * 0.6, "A₂": amp * 0.4,
                "τ": tau, "τ₁": tau * 0.4, "τ₂": tau * 1.5, "β": 1.0,
                "r₀": x0, "θ₀": x0, "σ": sigma,
                "C": offset, "a": slope, "b": ymid, "c": offset, "d": offset,
                "n": 1.0,
            }
            model_name = self._combo_model.currentText()
            for spin, (pname, _) in zip(
                self._fit_param_spins, self._fit_models[model_name]["params"]
            ):
                spin.setValue(lookup.get(pname, 1.0))
        except Exception:
            pass

    # ---------------------------------------------------------------- #
    # Run / stop                                                         #
    # ---------------------------------------------------------------- #

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        model_name = self._combo_model.currentText()
        func = self._fit_models[model_name]["func"]
        p0 = [spin.value() for spin in self._fit_param_spins]
        lo = float(self._spin_lo.value())
        hi = float(self._spin_hi.value())

        self._results.clear()
        self._progress.setValue(0)
        for curve in self._param_curves:
            curve.setData([], [])
        self._btn_export.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_status.setText("Analyse en cours…")

        self._worker = _FitWorker(
            data3d=self._data3d,
            cx=self._cx, cy=self._cy,
            r_min=self._r_min, r_max=self._r_max,
            a_min=self._a_min, a_max=self._a_max,
            func=func, p0=p0,
            fit_lo=lo, fit_hi=hi,
            mode=self._mode,
            chain=self._chk_chain.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        self._btn_stop.setEnabled(False)

    def _on_progress(self, frame_idx: int, popt: list[float]) -> None:
        self._results.append(popt)
        self._progress.setValue(frame_idx + 1)
        n_done = len(self._results)
        xs = list(range(n_done))
        for i, curve in enumerate(self._param_curves):
            ys = [r[i] if i < len(r) else float("nan") for r in self._results]
            curve.setData(xs, ys, connect="finite")

        # Brief inline summary of the latest result
        model_name = self._combo_model.currentText()
        pnames = [n for n, _ in self._fit_models[model_name]["params"]]
        parts = [f"{n}={v:.4g}" for n, v in zip(pnames, popt) if not np.isnan(v)]
        self._lbl_status.setText(
            f"Frame {frame_idx + 1}/{self._n_frames}\n" + "  ".join(parts)
        )

    def _on_finished(self) -> None:
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        n_ok = sum(
            1 for r in self._results
            if r and not all(np.isnan(v) for v in r)
        )
        self._lbl_status.setText(
            f"Terminé — {n_ok}/{len(self._results)} frames fittées."
        )
        self._btn_export.setEnabled(bool(self._results))

    # ---------------------------------------------------------------- #
    # Export                                                             #
    # ---------------------------------------------------------------- #

    def _on_export(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les résultats", "",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        model_name = self._combo_model.currentText()
        pnames = [n for n, _ in self._fit_models[model_name]["params"]]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + pnames)
            for i, row in enumerate(self._results):
                writer.writerow([i] + [f"{v:.8g}" for v in row])


# ─────────────────────────────────────────────────────────────────── #
# Main dialog                                                          #
# ─────────────────────────────────────────────────────────────────── #

class FrameAnalysisTool(QDialog):
    """
    Frame-by-frame fitting of I(r) and I(θ).

    Opens from the Curve Fitting tab of the XRMS Analyze tool.
    The active profile at launch (radial or angular) is pre-filled with
    the model / parameters the user just fitted on the reference frame.
    """

    def __init__(
        self,
        data3d: np.ndarray,
        cx: float, cy: float,
        r_min: int, r_max: int,
        a_min: int, a_max: int,
        fit_models: dict,
        active_mode: str = "radial",   # which tab to show first
        model_name: str = "",
        p0: list[float] | None = None,
        fit_lo: float = 0.0,
        fit_hi: float = 100.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._data3d = data3d
        n = data3d.shape[0]

        self.setWindowTitle(f"Analyse frame par frame — {n} frames")
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1200, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Info bar
        info = QLabel(
            f"  {n} frames  |  centre ({cx:.1f}, {cy:.1f})  |  "
            f"r ∈ [{r_min}, {r_max if r_max > 0 else 'max'}]  |  "
            f"θ ∈ [{a_min}°, {a_max}°]"
        )
        info.setStyleSheet("color:#aaa; padding:4px;")
        outer.addWidget(info)

        tabs = QTabWidget()
        outer.addWidget(tabs)

        # Determine which mode gets the pre-filled params
        p0_r = p0 if active_mode == "radial"  else []
        p0_a = p0 if active_mode == "angular" else []
        mn_r = model_name if active_mode == "radial"  else next(iter(fit_models))
        mn_a = model_name if active_mode == "angular" else next(iter(fit_models))
        lo_r = fit_lo if active_mode == "radial"  else 0.0
        hi_r = fit_hi if active_mode == "radial"  else (r_max if r_max > 0 else 500.0)
        lo_a = fit_lo if active_mode == "angular" else 0.0
        hi_a = fit_hi if active_mode == "angular" else 360.0

        panel_r = _ProfileFitPanel(
            mode="radial",
            n_frames=n,
            data3d=data3d,
            cx=cx, cy=cy,
            r_min=r_min, r_max=r_max,
            a_min=a_min, a_max=a_max,
            fit_models=fit_models,
            default_model=mn_r,
            default_p0=p0_r,
            default_fit_lo=lo_r,
            default_fit_hi=hi_r,
        )
        tabs.addTab(panel_r, "I(r) — Profil radial")

        panel_a = _ProfileFitPanel(
            mode="angular",
            n_frames=n,
            data3d=data3d,
            cx=cx, cy=cy,
            r_min=r_min, r_max=r_max,
            a_min=a_min, a_max=a_max,
            fit_models=fit_models,
            default_model=mn_a,
            default_p0=p0_a,
            default_fit_lo=lo_a,
            default_fit_hi=hi_a,
        )
        tabs.addTab(panel_a, "I(θ) — Profil angulaire")

        # Show the tab that matches what the user was fitting
        tabs.setCurrentIndex(0 if active_mode == "radial" else 1)

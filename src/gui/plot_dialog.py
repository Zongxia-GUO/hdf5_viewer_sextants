"""The Plot surface: a matplotlib figure over a list of :class:`Series`.

Two things live here:

* :class:`PlotPanel` — the figure, its toolbar and every setting that shapes it.
  A plain widget, so it can be dropped into a tab as easily as into a window.
* :class:`PlotDialog` — a window around one panel, with the Data page beside it.

The split exists because the batch dialog shows a live figure on its own second
page rather than opening a window; without it that page would either duplicate
the controls or be reduced to a button that pops a dialog.

matplotlib is imported here rather than at application start; it costs a good
fraction of a second and only a plot needs it.
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

import io
import json
import logging
import pathlib
from typing import Any, Sequence

import matplotlib
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QIcon, QImage
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (  # noqa: E402  (must follow use())
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402

from src.gui.plot_palettes import (  # noqa: E402
    DEFAULT_PALETTE_KEY,
    PALETTES,
    Palette,
    get_palette,
    palette_from_label,
    style_for,
)
from src.gui.plot_series import (  # noqa: E402
    Series,
    common_axis_labels,
    positive_axis,
    table_from_series,
)
from src.gui.export_naming import (  # noqa: E402
    pair_label,
    remember_save_directory,
    suggested_save_path,
)
from src.gui.table_model import CopyableTableView, DataTable  # noqa: E402
from src.gui.x_target import (  # noqa: E402
    read_x_dataset,
    register_x_target,
    remembered_x_dataset,
    x_scope_of,
)
from src.img.img_path import img_path  # noqa: E402

log = logging.getLogger(__name__)

# Past this many curves a legend covers the data it is labelling.
MAX_LEGEND_ENTRIES = 12

FIGURE_FILTERS = "PNG image (*.png);;PDF document (*.pdf);;SVG vector (*.svg)"

# Axis labels are a few words, so the four label fields are capped rather than
# left to claim a text box's default width each. See _build_controls.
LABEL_FIELD_MAX_WIDTH = 165

# Remembered across windows so the palette is a preference, not a per-plot choice.
_SETTINGS_PALETTE = "plot/palette"
_SETTINGS_DPI = "plot/save_dpi"
# Lock: keep one look across Plot windows instead of re-deriving it per dataset.
_SETTINGS_LOCKED = "plot/locked"
_SETTINGS_LOCK_STATE = "plot/locked_state"

# Windows opened through open_plot_dialog(), kept alive until they close.
_OPEN_PLOTS: set["PlotDialog"] = set()


class PlotPanel(QWidget):
    """A matplotlib figure with the controls that shape it.

    Everything about drawing lives here — the axes, the settings box, the X drop
    target, the lock. It knows nothing about windows or tabs, so the same panel
    serves the Plot window and the batch dialog's Plot page.
    """

    #: Emitted when the curves change (an X was dropped or reset), so whatever
    #: shows the numbers alongside can follow.
    series_changed = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        series: Sequence[Series] = (),
        right_series: Sequence[Series] = (),
        right_label: str = "",
        default_dir: pathlib.Path | None = None,
        allow_x_drop: bool = True,
        adopt_remembered_x: bool = True,
    ) -> None:
        super().__init__(parent)
        self._allow_x_drop = allow_x_drop
        self._series = list(series)
        # Curves on a second Y axis. Only the calculator uses this, to put a
        # derived result beside operands it shares no scale with. Two scales are
        # easy to misread, so the right axis is coloured to its own curve.
        self._right_series = list(right_series)
        self._right_label = right_label or (
            self._right_series[0].label if len(self._right_series) == 1 else "Result"
        )
        # A matplotlib Axes once twinx() has been called, None while there is
        # only one Y axis.
        self.axes_right: Any = None
        # Kept so "Reset X" can undo a dropped X dataset.
        self._original_series = list(series)
        self._original_right_series = list(right_series)
        self._default_dir = pathlib.Path(default_dir) if default_dir else pathlib.Path.home()
        # The application-wide store, like every other preference here. A
        # bespoke organisation/application pair would also force NativeFormat,
        # which is what let the test suite write into the real registry.
        self._settings = QSettings()
        # True while the code, not the user, is moving the controls, so
        # restoring a locked state does not immediately overwrite it.
        self._restoring = False

        self.figure = Figure(figsize=(6.0, 4.5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self._build_controls())
        self.redraw()
        if adopt_remembered_x:
            self.adopt_remembered_x()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> QGroupBox:
        """Labels, palette and axis switches, two rows so nothing sets the width."""
        x_label, y_label = common_axis_labels(self._series)

        box = QGroupBox("Plot settings")
        grid = QGridLayout(box)

        self.le_title = QLineEdit()
        self.le_title.setPlaceholderText("Optional figure title")
        self.le_x_label = QLineEdit(x_label)
        self.le_y_label = QLineEdit(y_label)
        grid.addWidget(QLabel("Title:"), 0, 0)

        grid.addWidget(self.le_title, 0, 1)
        grid.addWidget(QLabel("X label:"), 0, 2)
        grid.addWidget(self.le_x_label, 0, 3)
        grid.addWidget(QLabel("Y label:"), 0, 4)
        grid.addWidget(self.le_y_label, 0, 5)

        # Only meaningful when there is a second axis, so it is hidden until
        # there is one — set_series() may bring one along later.
        self.lbl_right_label = QLabel("Right Y:")
        self.le_right_label = QLineEdit(self._right_label)
        grid.addWidget(self.lbl_right_label, 0, 6)
        grid.addWidget(self.le_right_label, 0, 7)
        self._show_right_label_field()

        # A QLineEdit asks for about seventeen characters' worth of room, and
        # four of them side by side set the whole window's minimum width — wide
        # enough that the fourth field could not appear without the window
        # growing. These are short labels; capping them costs nothing and lets
        # all four share one row.
        for field in (self.le_title, self.le_x_label, self.le_y_label, self.le_right_label):
            field.setMinimumWidth(70)
            field.setMaximumWidth(LABEL_FIELD_MAX_WIDTH)

        self.cb_palette = QComboBox()
        self.cb_palette.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.cb_palette.setMinimumContentsLength(10)
        for index, palette in enumerate(PALETTES):
            self.cb_palette.addItem(palette.label, palette.key)
            self.cb_palette.setItemData(index, palette.tooltip(), Qt.ItemDataRole.ToolTipRole)
        remembered = self._settings.value(_SETTINGS_PALETTE, DEFAULT_PALETTE_KEY, type=str)
        self.cb_palette.setCurrentIndex(max(0, self.cb_palette.findData(remembered)))
        self._update_palette_tooltip()

        self.chk_log_x = QCheckBox("Log X")
        self.chk_log_y = QCheckBox("Log Y")
        self.chk_log_y.setToolTip("Put the left Y axis on a log scale")
        # One switch used to drive both Y axes, on the reasoning that different
        # scales invite reading the two sides against each other. But the two
        # sides carry different quantities — that is why there are two — and a
        # ratio next to raw counts genuinely needs one log and one linear.
        self.chk_log_ry = QCheckBox("Log RY")
        self.chk_log_ry.setToolTip("Put the right Y axis on a log scale")
        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        # Lines and Markers pick the mark: line, scatter, or both. At least one
        # has to stay on — with neither, the figure would simply be empty.
        self.chk_lines = QCheckBox("Lines")
        self.chk_lines.setChecked(True)
        self.chk_lines.setToolTip("Draw the curves as lines")
        self.chk_markers = QCheckBox("Markers")
        self.chk_markers.setToolTip("Draw a marker at every point; with Lines off this is a scatter plot")
        self.chk_legend = QCheckBox("Legend")
        self.chk_legend.setChecked(len(self._series) > 1)

        self.chk_lock = QCheckBox("Lock")
        self.chk_lock.setToolTip(
            "Keep these settings for every plot opened afterwards, so a run of "
            "scans comes out looking the same.\n"
            "Uncheck to go back to per-plot defaults."
        )
        self.chk_lock.setChecked(bool(self._settings.value(_SETTINGS_LOCKED, False, type=bool)))

        self._gate_log_switches()

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Colors:"))
        row2.addWidget(self.cb_palette)
        row2.addSpacing(8)
        for check in (self.chk_log_x, self.chk_log_y, self.chk_log_ry, self.chk_grid,
                      self.chk_lines, self.chk_markers, self.chk_legend):
            row2.addWidget(check)
        row2.addSpacing(8)
        row2.addWidget(self.chk_lock)
        row2.addStretch()
        # The full eight columns: the label row uses all of them, so a shorter
        # span left the rows below stopping short of the Right Y field.
        grid.addLayout(row2, 1, 0, 1, 8)
        grid.addWidget(self._build_x_row(), 2, 0, 1, 8)

        # A locked panel opens on the stored settings rather than on defaults
        # derived from this particular dataset.
        if self.chk_lock.isChecked():
            self._restore_locked_state()

        for widget in (self.le_title, self.le_x_label, self.le_y_label, self.le_right_label):
            widget.textChanged.connect(self._setting_changed)
        self.cb_palette.currentIndexChanged.connect(self._palette_changed)
        for check in (self.chk_log_x, self.chk_log_y, self.chk_log_ry, self.chk_grid,
                      self.chk_lines, self.chk_markers, self.chk_legend):
            check.toggled.connect(self._setting_changed)
        self.chk_lines.toggled.connect(lambda on: self._keep_a_mark(on, self.chk_markers))
        self.chk_markers.toggled.connect(lambda on: self._keep_a_mark(on, self.chk_lines))
        self.chk_lock.toggled.connect(self._lock_toggled)
        return box

    def _show_right_label_field(self) -> None:
        wanted = bool(self._right_series)
        self.lbl_right_label.setVisible(wanted)
        self.le_right_label.setVisible(wanted)

    def _keep_a_mark(self, still_on: bool, other: QCheckBox) -> None:
        """Turning off the last mark turns the other one on.

        Refusing the click outright would leave the box looking stuck; switching
        to the other mark is what "no lines" means in practice — a scatter.
        """
        if not still_on and not other.isChecked():
            other.setChecked(True)

    def _build_x_row(self) -> QWidget:
        """The drop target for an X dataset.

        Curves often arrive plotted against the sample index, and the X that
        belongs with them is one drag away in the tree — so accept it here
        rather than sending the user back to re-open the plot.

        Hidden when the host already owns the X choice: two X fields on one
        screen would disagree the moment either was used.
        """
        self.le_x_source = QLineEdit()
        self.le_x_source.setPlaceholderText(
            "Drag a dataset here (or type file::path) to use it as X"
        )
        self.le_x_source.setAcceptDrops(True)
        self.le_x_source.dragEnterEvent = self._x_drag_enter        # type: ignore[assignment]
        self.le_x_source.dropEvent = self._x_drop                   # type: ignore[assignment]
        self.le_x_source.returnPressed.connect(
            lambda: self.apply_x_key(self.le_x_source.text())
        )

        self.btn_x_reset = QPushButton("Reset X")
        self.btn_x_reset.setAutoDefault(False)
        self.btn_x_reset.setToolTip("Go back to the X the curves arrived with")
        self.btn_x_reset.clicked.connect(self.reset_x)

        self.x_row = QWidget()
        row = QHBoxLayout(self.x_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("X data:"))
        row.addWidget(self.le_x_source, stretch=1)
        row.addWidget(self.btn_x_reset)
        self.x_row.setVisible(self._allow_x_drop)
        return self.x_row

    # ------------------------------------------------------------------
    # The curves
    # ------------------------------------------------------------------

    def all_series(self) -> list[Series]:
        """Every curve drawn, left axis first — the figure's full content."""
        return self._series + self._right_series

    def set_series(
        self,
        series: Sequence[Series],
        right_series: Sequence[Series] = (),
        *,
        keep_labels: bool = True,
    ) -> None:
        """Replace the curves, keeping the settings the user has chosen.

        Used by the batch dialog, whose selection changes under the panel while
        it is on screen. Axis labels are left alone by default so a typed label
        is not wiped out by an unrelated refresh.
        """
        self._series = list(series)
        self._right_series = list(right_series)
        self._original_series = list(series)
        self._original_right_series = list(right_series)
        self._show_right_label_field()

        if not keep_labels:
            x_label, y_label = common_axis_labels(self._series)
            self._restoring = True
            try:
                self.le_x_label.setText(x_label)
                self.le_y_label.setText(y_label)
            finally:
                self._restoring = False

        self._gate_log_switches()
        self.redraw()
        self.series_changed.emit()

    def _gate_log_switches(self) -> None:
        """A log axis on data that touches zero silently drops points.

        Each switch is judged on the curves it actually governs. Judging the
        right axis by the left one's data would disable a log scale the right
        axis could perfectly well take, and vice versa.
        """
        gates = (
            (self.chk_log_x, "x", self.all_series(), "X"),
            (self.chk_log_y, "y", self._series or self._right_series, "The left Y axis"),
            (self.chk_log_ry, "y", self._right_series, "The right Y axis"),
        )
        for check, axis, series, name in gates:
            usable = bool(series) and positive_axis(series, axis)
            if not usable and check.isChecked():
                check.setChecked(False)
            check.setEnabled(usable)
            check.setToolTip(
                "" if usable
                else f"{name} contains values <= 0, so a log scale would drop points"
            )
        # Nothing on the right means nothing to switch.
        self.chk_log_ry.setVisible(bool(self._right_series))

    # ------------------------------------------------------------------
    # X data dropped in from the tree
    # ------------------------------------------------------------------

    def _x_drag_enter(self, event: Any) -> None:
        if event is not None and event.mimeData().hasText():
            event.acceptProposedAction()

    def _x_drop(self, event: Any) -> None:
        if event is None or not event.mimeData().hasText():
            return
        event.acceptProposedAction()
        key = event.mimeData().text().strip()
        self.le_x_source.setText(key)
        self.apply_x_key(key)

    def apply_x_key(self, key: str) -> bool:
        """Draw the curves against the dataset named by ``key``.

        Only curves of the same length are re-based. A shorter or longer one is
        left on its own X rather than being silently trimmed to fit — that would
        pair up values that do not belong together.
        """
        try:
            x_data = read_x_dataset(key)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Use As X", str(exc))
            return False
        except Exception as exc:
            log.warning("Could not read %s as X: %s", key, exc)
            QMessageBox.warning(self, "Cannot Use As X", f"Could not read the dataset:\n{exc}")
            return False

        skipped: list[str] = []

        def rebase(group: list[Series]) -> list[Series]:
            out = []
            for item in group:
                if item.y.size == x_data.size:
                    out.append(Series(item.label, item.y, x_data))
                else:
                    skipped.append(item.label)
                    out.append(item)
            return out

        left = rebase(self._series)
        right = rebase(self._right_series)

        if not any(s.x is x_data for s in left + right):
            QMessageBox.warning(
                self,
                "Length Mismatch",
                f"The X dataset is {x_data.size} long and no curve here has that many "
                "points, so nothing was changed.",
            )
            return False

        self._series = left
        self._right_series = right
        self._apply_series_change(default_x_label=key.split("::")[-1].strip("/").split("/")[-1])
        if skipped:
            QMessageBox.information(
                self,
                "Some Curves Kept Their X",
                "These curves have a different length and were left as they were:\n"
                + ", ".join(skipped),
            )
        return True

    def set_x_dataset(self, key: str) -> bool:
        """Take an X dataset handed over by the tree's ``Set X``."""
        self.le_x_source.setText(key)
        return self.apply_x_key(key)

    def adopt_remembered_x(self) -> bool:
        """Start from the X already chosen in the tree, if it fits these curves.

        Quiet on purpose: the user asked for a plot, not for this, so a length
        that does not match is simply left alone rather than explained.
        """
        drawn = self.all_series()
        # The X chosen for the tool this plot belongs to. A plot opened from the
        # calculator must not start from an X chosen in the main window's tree
        # for entirely different files.
        key = remembered_x_dataset(x_scope_of(self))
        if not key or not drawn or any(item.x is not None for item in drawn):
            return False
        try:
            x_data = read_x_dataset(key)
        except Exception:
            return False
        if not all(item.y.size == x_data.size for item in drawn):
            return False

        self._series = [Series(s.label, s.y, x_data) for s in self._series]
        self._right_series = [Series(s.label, s.y, x_data) for s in self._right_series]
        self._original_series = list(self._series)
        self._original_right_series = list(self._right_series)
        self.le_x_source.setText(key)
        self._apply_series_change(
            default_x_label=key.split("::")[-1].strip("/").split("/")[-1]
        )
        return True

    def reset_x(self) -> None:
        """Put the curves back on the X they were opened with."""
        self.le_x_source.clear()
        self._series = list(self._original_series)
        self._right_series = list(self._original_right_series)
        self._apply_series_change(default_x_label=common_axis_labels(self.all_series())[0])

    def _apply_series_change(self, default_x_label: str) -> None:
        """Re-gate the log switches, relabel, and redraw after the X changed."""
        self._gate_log_switches()
        self.le_x_label.setText(default_x_label)
        self.redraw()
        self.series_changed.emit()

    # ------------------------------------------------------------------
    # Locked settings
    # ------------------------------------------------------------------

    def lock_state(self) -> dict[str, Any]:
        """What the lock carries from one plot to the next.

        Exactly the controls in the Plot settings box and nothing else — the
        view rectangle in particular is *not* locked, so each figure still
        autoscales to its own data.
        """
        return {
            "title": self.le_title.text(),
            "x_label": self.le_x_label.text(),
            "y_label": self.le_y_label.text(),
            "right_label": self.le_right_label.text(),
            "palette": self.cb_palette.currentData(),
            "log_x": self.chk_log_x.isChecked(),
            "log_y": self.chk_log_y.isChecked(),
            "log_ry": self.chk_log_ry.isChecked(),
            "grid": self.chk_grid.isChecked(),
            "lines": self.chk_lines.isChecked(),
            "markers": self.chk_markers.isChecked(),
            "legend": self.chk_legend.isChecked(),
        }

    def _stored_lock_state(self) -> dict[str, Any]:
        raw = self._settings.value(_SETTINGS_LOCK_STATE, "", type=str)
        try:
            state = json.loads(raw) if raw else {}
        except ValueError:
            log.warning("Ignoring an unreadable locked plot state.")
            return {}
        return state if isinstance(state, dict) else {}

    def _save_lock_state(self) -> None:
        if self.chk_lock.isChecked():
            self._settings.setValue(_SETTINGS_LOCK_STATE, json.dumps(self.lock_state()))

    def _restore_locked_state(self) -> None:
        """Put the stored settings into the controls, before the first draw.

        A log switch that this dataset cannot take stays off: the lock carries a
        preference, and it must not put the axes into a state that would drop
        points silently.
        """
        state = self._stored_lock_state()
        if not state:
            return

        self._restoring = True
        try:
            self.le_title.setText(str(state.get("title", "")))
            self.le_x_label.setText(str(state.get("x_label", self.le_x_label.text())))
            self.le_y_label.setText(str(state.get("y_label", self.le_y_label.text())))
            self.le_right_label.setText(str(state.get("right_label", self.le_right_label.text())))
            index = self.cb_palette.findData(state.get("palette"))
            if index >= 0:
                self.cb_palette.setCurrentIndex(index)
            for check, key in (
                (self.chk_log_x, "log_x"),
                (self.chk_log_y, "log_y"),
                (self.chk_log_ry, "log_ry"),
                (self.chk_grid, "grid"),
                (self.chk_lines, "lines"),
                (self.chk_markers, "markers"),
                (self.chk_legend, "legend"),
            ):
                if check.isEnabled():
                    check.setChecked(bool(state.get(key, check.isChecked())))
            # The mutual guard is bypassed while restoring, so re-assert it:
            # a stored state with neither mark would draw an empty figure.
            if not (self.chk_lines.isChecked() or self.chk_markers.isChecked()):
                self.chk_lines.setChecked(True)
        finally:
            self._restoring = False

    def _setting_changed(self) -> None:
        if self._restoring:
            return
        self.redraw()
        self._save_lock_state()

    def _lock_toggled(self, checked: bool) -> None:
        """Capture the current look on lock; release to per-plot defaults on unlock."""
        self._settings.setValue(_SETTINGS_LOCKED, checked)
        if checked:
            self._settings.setValue(_SETTINGS_LOCK_STATE, json.dumps(self.lock_state()))
        else:
            self._settings.remove(_SETTINGS_LOCK_STATE)
            self.redraw()

    def current_palette(self) -> Palette:
        """The palette currently selected in the picker."""
        key = self.cb_palette.currentData()
        return get_palette(key) if key else palette_from_label(self.cb_palette.currentText())

    def _palette_changed(self) -> None:
        self._settings.setValue(_SETTINGS_PALETTE, self.cb_palette.currentData())
        self._update_palette_tooltip()
        if self._restoring:
            return
        self.redraw()
        self._save_lock_state()

    def _update_palette_tooltip(self) -> None:
        self.cb_palette.setToolTip(self.current_palette().tooltip())

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def redraw(self) -> None:
        """Rebuild the axes from the current settings.

        The whole body counts as "not the user", so a locked panel does not
        record the redraw itself as a change.
        """
        self._restoring = True
        try:
            self._draw_axes()
        finally:
            self._restoring = False
        self.canvas.draw_idle()

    def _plot_onto(self, axes: Any, series: Sequence[Series], first_slot: int) -> None:
        """Draw ``series`` on ``axes``, taking palette slots from ``first_slot``.

        The slot offset is what keeps the two axes from repeating a colour: on a
        dual-axis figure the curves are told apart by hue alone, so a repeat
        would make the second axis unreadable.
        """
        palette = self.current_palette()
        draw_lines = self.chk_lines.isChecked()
        marker = "o" if self.chk_markers.isChecked() else None
        # Markers alone are a scatter, so give them room; on a line they are
        # an accent and a big dot would swamp the trace.
        marker_size = 3.5 if draw_lines else 5.0

        for index, item in enumerate(series):
            x, y = item.points()
            color, line_style = style_for(first_slot + index, palette)
            axes.plot(
                x, y,
                color=color,
                linestyle=line_style if draw_lines else "none",
                linewidth=1.6,
                marker=marker,
                markersize=marker_size,
                label=item.label,
            )

    def _draw_axes(self) -> None:
        self.axes.clear()
        if self.axes_right is not None:
            self.axes_right.remove()
            self.axes_right = None

        self._plot_onto(self.axes, self._series, 0)

        if self._right_series:
            self.axes_right = self.axes.twinx()
            self._plot_onto(self.axes_right, self._right_series, len(self._series))
            self._dress_right_axis()

        if self.chk_log_x.isChecked():
            self.axes.set_xscale("log")
        if self.chk_log_y.isChecked():
            self.axes.set_yscale("log")
        if self.chk_log_ry.isChecked() and self.axes_right is not None:
            self.axes_right.set_yscale("log")

        self.axes.set_xlabel(self.le_x_label.text())
        self.axes.set_ylabel(self.le_y_label.text())
        if self.le_title.text().strip():
            self.axes.set_title(self.le_title.text().strip())
        # Style kwargs must only go with grid(True): matplotlib treats any of them
        # as "you clearly want a grid" and turns it back on, so passing alpha
        # alongside False left the box unable to switch the grid off at all.
        if self.chk_grid.isChecked():
            self.axes.grid(True, alpha=0.3)
        else:
            self.axes.grid(False)

        # Identity must not rest on colour alone, but a legend longer than the
        # plot is worse than none — past the cap the toolbar's hover readout and
        # the data table carry it instead.
        drawn = self.all_series()
        if self.chk_legend.isChecked() and drawn:
            if len(drawn) <= MAX_LEGEND_ENTRIES:
                # One legend for both axes; twinx would otherwise give the right
                # axis its own box on top of the left one's.
                handles, labels = self.axes.get_legend_handles_labels()
                if self.axes_right is not None:
                    right_handles, right_labels = self.axes_right.get_legend_handles_labels()
                    handles += right_handles
                    labels += right_labels
                self.axes.legend(handles, labels, fontsize=8, framealpha=0.85)
            else:
                self.axes.set_title(
                    (self.axes.get_title() + " " if self.axes.get_title() else "")
                    + f"({len(drawn)} curves; see the Data page)"
                )

    def _dress_right_axis(self) -> None:
        """Label the right axis and tint it to its curve.

        Two scales on one figure are easy to misread. Colouring the axis to the
        curve it belongs to is what makes the pairing unambiguous, so it is not
        decoration — without it the second axis should not be drawn at all.
        """
        assert self.axes_right is not None
        self.axes_right.set_ylabel(self.le_right_label.text())

        color, _style = style_for(len(self._series), self.current_palette())
        self.axes_right.yaxis.label.set_color(color)
        self.axes_right.tick_params(axis="y", colors=color)
        self.axes_right.spines["right"].set_color(color)
        # A second grid over the first would double every horizontal line.
        self.axes_right.grid(False)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def copy_figure(self) -> None:
        """Put the rendered figure on the clipboard as a PNG."""
        buffer = io.BytesIO()
        try:
            self.figure.savefig(buffer, format="png", dpi=150)
        except Exception as exc:
            log.warning("Could not render the figure for the clipboard: %s", exc)
            QMessageBox.warning(self, "Copy Failed", f"Could not render the figure:\n{exc}")
            return
        image = QImage.fromData(buffer.getvalue(), "PNG")
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setImage(image)

    def figure_name(self) -> str:
        """Name the figure after the curves in it, not after the window.

        One curve gives that curve's name; several give the range they span, so
        a folder of saved figures can be told apart without opening them.
        """
        drawn = self.all_series()
        if not drawn:
            return "plot"
        if len(drawn) == 1:
            return drawn[0].label
        return pair_label(drawn[0].label, drawn[-1].label)

    def save_figure(self) -> None:
        """Write the figure to PNG, PDF or SVG; the extension follows the filter."""
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            suggested_save_path(self.figure_name(), extension=".png"),
            FIGURE_FILTERS,
        )
        if not path:
            return

        target = pathlib.Path(path)
        for label, suffix in (("PNG", ".png"), ("PDF", ".pdf"), ("SVG", ".svg")):
            if selected.startswith(label) and target.suffix.lower() != suffix:
                target = target.with_suffix(suffix)
                break

        dpi = self._settings.value(_SETTINGS_DPI, 300, type=int)
        try:
            self.figure.savefig(target, dpi=dpi)
        except Exception as exc:
            log.warning("Could not save the figure to %s: %s", target, exc)
            QMessageBox.warning(self, "Save Failed", f"Could not save the figure:\n{exc}")
            return
        self._default_dir = target.parent
        remember_save_directory(target)


class PlotDialog(QDialog):
    """A window around one :class:`PlotPanel`, with the Data page beside it."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        series: Sequence[Series],
        right_series: Sequence[Series] = (),
        title: str = "Plot",
        right_label: str = "",
        default_dir: pathlib.Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Plot - {title}")
        self.setWindowIcon(QIcon(str(pathlib.Path(img_path(), "plot.ico"))))
        # An ordinary window, like the export dialogs: a plot is something you
        # keep open beside the data, so it must be able to go behind the main
        # window and must never block it.
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        # Square, like the full export dialogs: the controls sit under the
        # figure, so extra width only stretches the axes.
        self.resize(720, 720)

        self.panel = PlotPanel(
            self,
            series=series,
            right_series=right_series,
            right_label=right_label,
            default_dir=default_dir,
        )

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_page(), "Data")
        self.tabs.addTab(self.panel, "Plot")
        self.tabs.setCurrentIndex(1)   # the figure is what was asked for
        root.addWidget(self.tabs, stretch=1)
        root.addLayout(self._build_action_row())

        self.panel.series_changed.connect(self._refresh_data_page)
        register_x_target(self)

    def __getattr__(self, name: str) -> Any:
        """Fall through to the panel.

        The dialog is a frame around one panel, so ``dialog.chk_grid`` and
        ``dialog.axes`` mean the panel's. Keeping the names here spares every
        caller from knowing where the split runs.
        """
        panel = self.__dict__.get("panel")
        if panel is not None:
            try:
                return getattr(panel, name)
            except AttributeError:
                pass
        raise AttributeError(name)

    def _build_data_page(self) -> QWidget:
        """The numbers behind the figure — the same points, not a re-read."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.data_view = CopyableTableView()
        layout.addWidget(self.data_view, stretch=1)

        self.lbl_data_info = QLabel()
        self.lbl_data_info.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_data_info)
        self._refresh_data_page()
        return page

    def _refresh_data_page(self) -> None:
        """Rebuild the Data page so it keeps matching what the figure draws."""
        drawn = self.panel.all_series()
        headers, table = table_from_series(drawn)
        self.data_view.setModel(DataTable(table, column_names=headers))
        self.lbl_data_info.setText(
            f"{len(drawn)} curve(s), {table.shape[0]} row(s). "
            "Ctrl+C copies the selection."
        )

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setToolTip("Copy the figure to the clipboard as an image")
        self.btn_copy.clicked.connect(self.panel.copy_figure)
        self.btn_save = QPushButton("Save Figure")
        self.btn_save.setToolTip("Save the figure as PNG, PDF or SVG")
        self.btn_save.clicked.connect(self.panel.save_figure)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        for button in (self.btn_copy, self.btn_save, self.btn_close):
            button.setAutoDefault(False)
            row.addWidget(button)
        return row


def open_plot_dialog(
    parent: QWidget | None,
    series: Sequence[Series],
    *,
    right_series: Sequence[Series] = (),
    title: str = "Plot",
    right_label: str = "",
    default_dir: pathlib.Path | None = None,
) -> PlotDialog | None:
    """Show a plot window, or explain why there is nothing to plot.

    ``right_series`` puts those curves on a second Y axis — for a quantity that
    shares the X but not the scale, such as a calculator result beside the
    operands it came from.

    Non-modal on purpose: a plot is something you keep open next to the data,
    and a modal one would block the tree it came from.
    """
    def drawable(group: Sequence[Series]) -> list[Series]:
        return [s for s in group if s.finite_points()[0].size]

    usable = drawable(series)
    usable_right = drawable(right_series)
    if not usable and not usable_right:
        QMessageBox.information(parent, "Nothing to Plot", "The selection has no finite data points.")
        return None
    if not usable:
        # Nothing left for the left axis: draw the survivors there instead of
        # showing a figure with an empty primary axis.
        usable, usable_right = usable_right, []

    dialog = PlotDialog(
        parent,
        series=usable,
        right_series=usable_right,
        title=title,
        right_label=right_label,
        default_dir=default_dir,
    )
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    # A parentless window would be collected the moment this returns; holding it
    # here keeps a plot opened from a tool alive until the user closes it.
    _OPEN_PLOTS.add(dialog)
    dialog.destroyed.connect(lambda *_: _OPEN_PLOTS.discard(dialog))
    dialog.show()
    dialog.raise_()
    return dialog

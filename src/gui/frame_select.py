"""The axis / frame pair every tool that takes an image needs for a stack.

A 3-D dataset is a stack of frames. Any tool that reconstructs or calibrates a
single image has to ask which axis counts them and what to do with them — and
the answer has to be visible, because reducing a stack in silence is how a mean
of 400 frames gets read as a measurement.

The Q calibration tool grew this pair first; the FTH and CDI tools could not
take a stack at all, and the batch export had grown its own axis naming. One
widget now, so the three windows ask the question the same way and slice with
the same code.

The frame box names a *method*, not a frame, and the box beside it changes to
suit: a frame number for "Single frame", the rejection threshold for "Clipped
mean", nothing for the rest. A list of every frame instead would be 401 entries
on a real 400-frame scan, which is no way to reach frame 287.
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

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from src.lib_h5.stacks import (
    CLIPPED_MEAN_OF_FRAMES,
    MEAN_OF_FRAMES,
    MEDIAN_OF_FRAMES,
    SUM_OF_FRAMES,
    axis_label,
    reduce_stack,
)
from src.recon.stack_combine import DEFAULT_KAPPA

#: The frame box's value for "one frame, the one in the box beside me".
SINGLE_FRAME = 0

#: What the frame box offers, in order. The first is the default: one frame,
#: shown as it was recorded. Every other entry computes something out of the
#: whole stack, and the default should not do that without being asked.
METHOD_CHOICES = (
    ("Single frame", SINGLE_FRAME),
    ("Mean", MEAN_OF_FRAMES),
    ("Sum", SUM_OF_FRAMES),
    ("Median", MEDIAN_OF_FRAMES),
    ("Clipped mean", CLIPPED_MEAN_OF_FRAMES),
)

#: Widths that hold the longest entry without stretching the panel.
AXIS_COMBO_WIDTH = 78
METHOD_COMBO_WIDTH = 108
PARAM_WIDTH = 66

#: How long to wait after the last edit before reloading. A clipped mean of a
#: 100x512x512 stack takes about a second, and the spin box arrows fire once
#: per click, so reloading on every one of them would make the control feel
#: stuck. Long enough to swallow a burst of clicks, short enough not to be a
#: wait of its own.
RELOAD_DELAY_MS = 350


class FrameSelector(QWidget):
    """Pick the frame axis and what to make of the frames along it.

    Hidden entirely while the selected data is 2-D: a control that can only be
    set one way is noise, and every tool that uses this has a panel where the
    room matters.
    """

    #: Emitted when the selection settles, so a tool can reload.
    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shape: tuple[int, ...] | None = None

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(RELOAD_DELAY_MS)
        self._settle.timeout.connect(self.changed.emit)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self.combo_axis = QComboBox()
        self.combo_axis.setToolTip("Which array axis the frames are counted along")
        self.combo_axis.setFixedWidth(AXIS_COMBO_WIDTH)
        self.combo_axis.currentIndexChanged.connect(self._on_axis_changed)

        self.combo_method = QComboBox()
        self.combo_method.setFixedWidth(METHOD_COMBO_WIDTH)
        self.combo_method.setToolTip(
            "One frame, or what to make of them all.\n"
            "Mean: best statistics, but one cosmic ray spoils it.\n"
            "Median: immune to those, and noisier for it.\n"
            "Clipped mean: rejects the outliers, averages the rest."
        )
        for text, value in METHOD_CHOICES:
            self.combo_method.addItem(text, value)
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)

        self.spin_frame = QSpinBox()
        self.spin_frame.setFixedWidth(PARAM_WIDTH)
        self.spin_frame.setToolTip("Which frame, counting from 0")
        self.spin_frame.valueChanged.connect(lambda _v: self._settle.start())

        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setFixedWidth(PARAM_WIDTH)
        self.spin_kappa.setRange(0.5, 20.0)
        self.spin_kappa.setSingleStep(0.5)
        self.spin_kappa.setDecimals(1)
        self.spin_kappa.setValue(DEFAULT_KAPPA)
        self.spin_kappa.setPrefix("k ")
        self.spin_kappa.setToolTip(
            "How many robust standard deviations from the median a sample may\n"
            "sit before it is thrown away. Lower rejects more, including data\n"
            "that was never wrong."
        )
        self.spin_kappa.valueChanged.connect(lambda _v: self._settle.start())

        row.addWidget(QLabel("Axis:"))
        row.addWidget(self.combo_axis)
        row.addWidget(QLabel("Frame:"))
        row.addWidget(self.combo_method)
        row.addWidget(self.spin_frame)
        row.addWidget(self.spin_kappa)
        row.addStretch()

        self._sync_parameter_box()
        self.setVisible(False)

    # ── What the tool tells it ────────────────────────────────────────── #

    def set_shape(self, shape: tuple[int, ...] | None) -> None:
        """Point the selector at a dataset shape, or at nothing.

        Anything with fewer than three axes is not a stack, so the whole
        control disappears rather than offering a choice of one.
        """
        shape = tuple(int(n) for n in shape) if shape else None
        if shape is not None and len(shape) < 3:
            shape = None
        if shape == self._shape:
            self.setVisible(shape is not None)
            return

        self._shape = shape
        self.setVisible(shape is not None)
        if shape is None:
            return

        previous_axis = self.combo_axis.currentData()
        self.combo_axis.blockSignals(True)
        self.combo_axis.clear()
        for axis in range(len(shape)):
            self.combo_axis.addItem(axis_label(axis, shape[axis]), axis)
        if previous_axis is not None and 0 <= int(previous_axis) < len(shape):
            self.combo_axis.setCurrentIndex(int(previous_axis))
        self.combo_axis.blockSignals(False)
        self._sync_frame_range()

    # ── What the tool asks it ─────────────────────────────────────────── #

    def selection(self) -> tuple[int, int]:
        """``(axis, index)``, where a negative index names a method.

        Read from the boxes whatever the shape is. The method is a choice the
        user made and it outlives any one dataset — an array handed straight to
        a tool has no shape to offer, and answering "frame 0" there would throw
        that choice away. With no shape the axis box is empty, so the answer
        falls back to axis 0, and the untouched default is frame 0 of it.
        """
        axis = self.combo_axis.currentData()
        method = self.combo_method.currentData()
        index = int(self.spin_frame.value()) if method == SINGLE_FRAME else int(method)
        return int(axis) if axis is not None else 0, index

    def kappa(self) -> float:
        """The rejection threshold, meaningful only for the clipped mean."""
        return float(self.spin_kappa.value())

    def reduce(self, stack: np.ndarray, label: str = "") -> tuple[np.ndarray, str]:
        """Apply the current selection, returning ``(frame, note)``."""
        axis, index = self.selection()
        return reduce_stack(stack, axis, index, label, self.kappa())

    # ── Internals ─────────────────────────────────────────────────────── #

    def _sync_frame_range(self) -> None:
        if self._shape is None:
            return
        axis = int(self.combo_axis.currentData() or 0) % len(self._shape)
        frames = self._shape[axis]
        self.spin_frame.blockSignals(True)
        self.spin_frame.setRange(0, max(0, frames - 1))
        self.spin_frame.setSuffix(f" / {frames - 1}")
        self.spin_frame.blockSignals(False)

    def _sync_parameter_box(self) -> None:
        """Show the one box the chosen method actually takes."""
        method = self.combo_method.currentData()
        self.spin_frame.setVisible(method == SINGLE_FRAME)
        self.spin_kappa.setVisible(method == CLIPPED_MEAN_OF_FRAMES)

    def _on_axis_changed(self, _index: int) -> None:
        self._sync_frame_range()
        self._settle.start()

    def _on_method_changed(self, _index: int) -> None:
        self._sync_parameter_box()
        self._settle.start()

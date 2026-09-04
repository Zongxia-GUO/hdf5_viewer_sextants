"""The axis / frame pair every tool that takes an image needs for a stack.

A 3-D dataset is a stack of frames. Any tool that reconstructs or calibrates a
single image has to ask which axis counts them and which one to use — and the
answer has to be visible, because reducing a stack in silence is how a mean of
400 frames gets read as a measurement.

The Q calibration tool grew this pair first; the FTH and CDI tools could not
take a stack at all, and the batch export had grown its own axis naming. One
widget now, so the three windows ask the question the same way and slice with
the same code.
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
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from src.lib_h5.stacks import MEAN_OF_FRAMES, axis_label, reduce_stack

#: Widths that hold "0 (128)" and "Mean" without stretching the panel they
#: sit in. The axis box holds a number and a length; the frame box holds a
#: number or the word Mean.
AXIS_COMBO_WIDTH = 78
FRAME_COMBO_WIDTH = 90


class FrameSelector(QWidget):
    """Pick the frame axis and the frame, or the mean of them all.

    Hidden entirely while the selected data is 2-D: a control that can only be
    set one way is noise, and every tool that uses this has a panel where the
    room matters.
    """

    #: Emitted when the axis or the frame changes, so a tool can reload.
    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shape: tuple[int, ...] | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self.combo_axis = QComboBox()
        self.combo_axis.setToolTip("Which array axis the frames are counted along")
        self.combo_axis.setFixedWidth(AXIS_COMBO_WIDTH)
        self.combo_axis.currentIndexChanged.connect(self._on_axis_changed)

        self.combo_frame = QComboBox()
        self.combo_frame.setFixedWidth(FRAME_COMBO_WIDTH)
        self.combo_frame.setToolTip(
            "Which frame to use, or the mean of them all.\n"
            "The mean is the default: more frames, better statistics."
        )
        self.combo_frame.currentIndexChanged.connect(lambda _i: self.changed.emit())

        row.addWidget(QLabel("Axis:"))
        row.addWidget(self.combo_axis)
        row.addWidget(QLabel("Frame:"))
        row.addWidget(self.combo_frame)
        row.addStretch()
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
        self._rebuild_frames()

    # ── What the tool asks it ─────────────────────────────────────────── #

    def selection(self) -> tuple[int, int]:
        """``(axis, index)``; index :data:`MEAN_OF_FRAMES` means average them.

        A hidden selector always answers "mean of axis 0", so a caller that
        loads a 2-D dataset never has to ask whether the control is showing.
        """
        if self._shape is None:
            return 0, MEAN_OF_FRAMES
        axis = self.combo_axis.currentData()
        index = self.combo_frame.currentData()
        return (int(axis) if axis is not None else 0,
                int(index) if index is not None else MEAN_OF_FRAMES)

    def reduce(self, stack: np.ndarray, label: str = "") -> tuple[np.ndarray, str]:
        """Apply the current selection, returning ``(frame, note)``."""
        axis, index = self.selection()
        return reduce_stack(stack, axis, index, label)

    # ── Internals ─────────────────────────────────────────────────────── #

    def _rebuild_frames(self) -> None:
        if self._shape is None:
            return
        axis = int(self.combo_axis.currentData() or 0) % len(self._shape)
        previous = self.combo_frame.currentData()
        self.combo_frame.blockSignals(True)
        self.combo_frame.clear()
        self.combo_frame.addItem("Mean", MEAN_OF_FRAMES)
        for index in range(self._shape[axis]):
            self.combo_frame.addItem(str(index), index)
        if previous is not None:
            found = self.combo_frame.findData(previous)
            self.combo_frame.setCurrentIndex(max(0, found))
        self.combo_frame.blockSignals(False)

    def _on_axis_changed(self, _index: int) -> None:
        self._rebuild_frames()
        self.changed.emit()

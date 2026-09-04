"""Reducing a stack of frames to the one image a tool works on.

A 3-D dataset is a stack of frames — time, energy, repeats, whatever was
scanned. Every tool that takes an image has to answer the same two questions
about one: which axis counts the frames, and which frame (or their mean). The
answers used to live wherever they were first needed, so the batch export
named its axes ``0 (128)`` while the Q tool named the same axis ``0``, and the
reconstruction tools could not take a stack at all.

This module holds the part that is not a widget, so the selector, the export
and the readers all slice the same way.
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

from typing import Any

import numpy as np

from src.recon.stack_combine import (
    DEFAULT_KAPPA,
    combine_clipped_mean,
    combine_mean,
    combine_median,
    combine_sum,
)

# How to turn many frames into one, encoded in the frame index so that the
# selection stays a plain (axis, index) pair everywhere it travels — into a
# background worker, into a note, into a test. Negative because a real frame
# index never is.
#
#: Average them all. Best statistics, no defence against outliers.
MEAN_OF_FRAMES = -1
#: Add them all. The mean times the frame count, so only the scale changes.
SUM_OF_FRAMES = -2
#: Per-pixel median. Immune to outliers, noisier than a mean.
MEDIAN_OF_FRAMES = -3
#: Reject outliers, then average the rest. See :mod:`src.recon.stack_combine`.
CLIPPED_MEAN_OF_FRAMES = -4

#: Every way of combining frames, in the order the selector offers them.
COMBINE_METHODS = (
    MEAN_OF_FRAMES,
    SUM_OF_FRAMES,
    MEDIAN_OF_FRAMES,
    CLIPPED_MEAN_OF_FRAMES,
)

#: What each one is called, on screen and in the note the status line shows.
METHOD_NAMES = {
    MEAN_OF_FRAMES: "mean",
    SUM_OF_FRAMES: "sum",
    MEDIAN_OF_FRAMES: "median",
    CLIPPED_MEAN_OF_FRAMES: "clipped mean",
}


def is_combination(index: int) -> bool:
    """True when the index names a way of combining frames, not one frame."""
    return int(index) in METHOD_NAMES


def axis_label(axis: int, length: int) -> str:
    """Name one array axis for a selector: ``0 (128)``.

    Numbered, not X/Y/Z. In a stack of frames shaped ``(100, 512, 512)`` axis 0
    is the frame — time, energy, whatever was scanned — and calling it "X"
    says it is a spatial direction, which it is not; the image's own X is the
    *last* axis, which the same scheme called "Z". The number is what the code
    slices by, and the length in brackets is what tells the axes apart in
    practice.

    The word "Axis" is not repeated here: every one of these selectors sits
    beside a label that already says it, and carrying it in each entry made the
    box twice as wide as the value it holds.
    """
    return f"{int(axis)} ({int(length)})"


def take_slice(stack: np.ndarray, axis: int, index: int) -> np.ndarray:
    """One frame out of a stack, counted along ``axis``.

    The viewer lets the axis be chosen, so everything downstream has to slice
    the same way. Taking ``stack[index]`` regardless — which is what the export
    did — silently ignores that choice and writes frames cut the wrong way.
    """
    arr = np.asarray(stack)
    if arr.ndim < 3:
        return arr
    axis = int(axis) % arr.ndim
    index = max(0, min(int(index), arr.shape[axis] - 1))
    return np.asarray(np.take(arr, index, axis=axis))


def slice_count(stack: np.ndarray, axis: int) -> int:
    """How many frames a stack holds along ``axis``."""
    arr = np.asarray(stack)
    if arr.ndim < 3:
        return 1
    return int(arr.shape[int(axis) % arr.ndim])


def slice_axis_from(export_settings: dict[str, Any]) -> int:
    """The axis the frames are counted along, defaulting to the first."""
    try:
        return int(export_settings.get("slice_axis", 0) or 0)
    except (TypeError, ValueError):
        return 0


def describe_reduction(
    shape: tuple[int, ...],
    axis: int,
    index: int,
    label: str = "",
    kappa: float = DEFAULT_KAPPA,
) -> str:
    """The note :func:`reduce_stack` would produce, from a shape alone.

    The lazy reader never holds the whole stack, so it cannot ask the array
    what was done to it; it knows the shape and the selection, which is all the
    sentence needs.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) <= 2:
        return ""
    axis = int(axis) % len(shape)
    frames = shape[axis]
    prefix = f"{label}: " if label else ""
    if is_combination(index):
        name = METHOD_NAMES[int(index)]
        extra = f", k={float(kappa):g}" if int(index) == CLIPPED_MEAN_OF_FRAMES else ""
        return f"{prefix}{name} of {frames} frames{extra} (axis {axis})"
    index = max(0, min(int(index), frames - 1))
    return f"{prefix}frame {index} of {frames} (axis {axis})"


def reduce_stack(
    stack: np.ndarray,
    axis: int,
    index: int,
    label: str = "",
    kappa: float = DEFAULT_KAPPA,
) -> tuple[np.ndarray, str]:
    """Reduce a stack to one 2-D frame, and say in words what was done.

    ``index`` of :data:`MEAN_OF_FRAMES` averages the frames; anything else
    takes that one. Returns ``(frame, note)`` — the note is not decoration.
    A stack used to be reduced in silence: 400 frames arrived, the status line
    said "Loaded (200x46)", and nothing told the user that what they were
    looking at was the mean of all 400.

    Anything already 2-D is returned untouched with an empty note, so a caller
    can hand every input through this without checking first.
    """
    arr = np.asarray(stack)
    if arr.ndim <= 2:
        return arr, ""

    axis = int(axis) % arr.ndim
    note = describe_reduction(arr.shape, axis, index, label, kappa)

    if is_combination(index):
        return combine_frames(arr, axis, index, kappa), note

    index = max(0, min(int(index), int(arr.shape[axis]) - 1))
    return np.asarray(np.take(arr, index, axis=axis)), note


def combine_frames(
    stack: np.ndarray,
    axis: int,
    index: int,
    kappa: float = DEFAULT_KAPPA,
) -> np.ndarray:
    """Apply one of the combination methods named by ``index``."""
    if int(index) == SUM_OF_FRAMES:
        return combine_sum(stack, axis)
    if int(index) == MEDIAN_OF_FRAMES:
        return combine_median(stack, axis)
    if int(index) == CLIPPED_MEAN_OF_FRAMES:
        return combine_clipped_mean(stack, axis, kappa)
    return combine_mean(stack, axis)


def read_frame(dataset, axis: int, index: int,
               kappa: float = DEFAULT_KAPPA) -> np.ndarray:
    """Read one frame straight out of an open h5py dataset.

    Reading the whole stack and then slicing costs the whole stack: measured on
    120x512x512 float32, taking frame 60 took 36.9 ms and pulled 126 MB through
    memory, where the equivalent h5py slice took 0.7 ms and pulled 1 MB — 52x,
    and the gap grows with the stack. A mean still has to read everything.

    ``index`` of :data:`MEAN_OF_FRAMES` returns the mean over ``axis``.
    """
    shape = tuple(dataset.shape)
    if len(shape) < 3:
        return np.asarray(dataset[()])

    axis = int(axis) % len(shape)
    if is_combination(index):
        # Every combination looks at all the frames, so there is nothing to
        # save here; the whole stack has to come in.
        return combine_frames(np.asarray(dataset[()]), axis, index, kappa)

    index = max(0, min(int(index), shape[axis] - 1))
    selector: list = [slice(None)] * len(shape)
    selector[axis] = index
    return np.asarray(dataset[tuple(selector)])

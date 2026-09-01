"""What a plot draws: a list of (x, y, label) curves, independent of its source.

The plot window is fed by four different places — the batch controls, the
comparison tool, the calculator and a data viewer — and none of them should have
to know about the others. They each produce ``Series`` and hand them over.

Note this is deliberately *not* the same shape as the export table. An export
writes flat columns with a possibly shared X and ragged padding; a plot needs the
X paired with each Y. Forcing one shape to serve both would distort both, so the
table builders in :mod:`src.gui.batch_export` stay as they are and this module
adapts their output.
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

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Header names an export table uses for a shared X column.
SHARED_X_HEADERS = ("X", "q")
# Suffixes marking a per-curve X column, e.g. "scanx_0080_X".
PER_CURVE_X_SUFFIXES = ("_X", "_q")


@dataclass(frozen=True)
class Series:
    """One curve: Y values, an optional X, and the name shown in the legend."""

    label: str
    y: np.ndarray
    x: np.ndarray | None = field(default=None)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """The (x, y) actually drawn.

        Without an X the curve is drawn against the sample index. Both arrays are
        trimmed to the shorter length so a mismatched pair cannot raise inside
        the plotting call.
        """
        y = np.asarray(self.y, dtype=float).ravel()
        if self.x is None:
            return np.arange(y.size, dtype=float), y
        x = np.asarray(self.x, dtype=float).ravel()
        n = min(x.size, y.size)
        return x[:n], y[:n]

    def finite_points(self) -> tuple[np.ndarray, np.ndarray]:
        """The drawn points with NaN/inf pairs dropped, for range calculations."""
        x, y = self.points()
        keep = np.isfinite(x) & np.isfinite(y)
        return x[keep], y[keep]


def series_from_table(headers: Sequence[str], columns: Sequence[Sequence[float]]) -> list[Series]:
    """Adapt an export-style flat table into curves.

    The export tables come in two shapes and this reads both:

    * one shared ``X``/``q`` column followed by several Y columns
    * repeating ``<name>_X``/``<name>_Y`` groups

    An X column governs **every** Y column that follows it, up to the next X.
    A dataset with two columns exports as ``scan_X, scan_Y_1, scan_Y_2``, so
    handing the X to the first Y alone would silently drop the second one back
    onto the sample index — which is what used to happen.

    A Y column with no X in front of it becomes an index-plotted series, which is
    what an export with "no X dataset" produces.
    """
    current_x: np.ndarray | None = None
    out: list[Series] = []

    for header, column in zip(headers, columns):
        values = np.asarray(column, dtype=float)
        if header in SHARED_X_HEADERS or any(
            header.endswith(suffix) for suffix in PER_CURVE_X_SUFFIXES
        ):
            current_x = values
            continue

        out.append(Series(label=header, y=values, x=current_x))

    return out


def series_from_columns(
    label_prefix: str,
    y_columns: np.ndarray,
    x: np.ndarray | None = None,
) -> list[Series]:
    """Adapt a rows x columns block into one series per column."""
    block = np.asarray(y_columns)
    if block.ndim == 1:
        return [Series(label=label_prefix, y=block, x=x)]

    if block.shape[1] == 1:
        return [Series(label=label_prefix, y=block[:, 0], x=x)]
    return [
        Series(label=f"{label_prefix}_{idx + 1}", y=block[:, idx], x=x)
        for idx in range(block.shape[1])
    ]


def common_axis_labels(series: Sequence[Series]) -> tuple[str, str]:
    """Sensible starting axis labels for a set of curves."""
    if not series:
        return "X", "Y"
    x_label = "X" if any(s.x is not None for s in series) else "Index"
    y_label = series[0].label if len(series) == 1 else "Value"
    return x_label, y_label


def data_bounds(series: Sequence[Series]) -> tuple[float, float, float, float] | None:
    """(x_min, x_max, y_min, y_max) over every finite point, or None if empty."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for item in series:
        x, y = item.finite_points()
        if x.size:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    return float(all_x.min()), float(all_x.max()), float(all_y.min()), float(all_y.max())


def positive_axis(series: Sequence[Series], axis: str = "y") -> bool:
    """True when every finite value on ``axis`` is > 0, i.e. a log axis is usable.

    Checked per axis: a curve of counts against a magnetic field sweeping through
    zero has a perfectly good log Y and an impossible log X, and gating both on
    the same answer would take the useful one away.
    """
    for item in series:
        x, y = item.finite_points()
        if not x.size:
            continue
        values = x if axis == "x" else y
        if values.min() <= 0:
            return False
    return True


def positive_only(series: Sequence[Series]) -> bool:
    """True when every finite point is > 0, i.e. both axes could go log."""
    return positive_axis(series, "x") and positive_axis(series, "y")


def shares_one_x(series: Sequence[Series]) -> bool:
    """True when every curve is drawn against the same X values."""
    with_x = [s for s in series if s.x is not None]
    if len(with_x) != len(series) or len(with_x) < 2:
        return len(with_x) == len(series)
    first, _ = with_x[0].points()
    for item in with_x[1:]:
        x, _y = item.points()
        if x.size != first.size or not np.allclose(x, first, equal_nan=True):
            return False
    return True


def table_from_series(series: Sequence[Series]) -> tuple[list[str], np.ndarray]:
    """The plotted points as a rectangular table, for the dialog's Data page.

    Curves of different lengths are padded with NaN so they fit one array; that
    is display only, and the export path keeps its own ragged handling.
    """
    if not series:
        return [], np.zeros((0, 0))

    columns: list[np.ndarray] = []
    headers: list[str] = []
    shared = shares_one_x(series) and series[0].x is not None

    if shared:
        headers.append("X")
        columns.append(series[0].points()[0])

    for item in series:
        x, y = item.points()
        if not shared and item.x is not None:
            headers.append(f"{item.label}_X")
            columns.append(x)
        headers.append(item.label)
        columns.append(y)

    rows = max(col.size for col in columns)
    table = np.full((rows, len(columns)), np.nan)
    for index, col in enumerate(columns):
        table[: col.size, index] = col
    return headers, table

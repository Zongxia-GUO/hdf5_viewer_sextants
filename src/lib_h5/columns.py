"""What the columns of a 2-D block are for: which one is the abscissa, which
are curves, which to leave out.

The main window's Columns panel and, later, the Data editor both need to turn
"here is an array with N columns" into "draw these against that". The rule for
the first, untouched view lives here, with no Qt, so the same answer is used
whether it is a panel populating itself or the viewer choosing a widget, and so
it can be tested against arrays rather than against the application.
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

from dataclasses import dataclass, replace
from enum import Enum


class Role(str, Enum):
    """What a column does when the block is drawn."""

    X = "X"
    Y = "Y"
    NONE = "—"


@dataclass(frozen=True)
class ColumnRole:
    """One column's part in the plot: its label, its role, whether it shows."""

    name: str
    role: Role = Role.NONE
    #: Only meaningful for a ``Y`` column; an ``X`` or ``NONE`` column never draws.
    visible: bool = False

    def with_role(self, role: Role) -> "ColumnRole":
        """The same column in a new role, with visibility that role allows.

        A column promoted to ``Y`` comes on; anything else goes dark, because a
        checked "show" box over an ``X`` column would claim something the plot
        cannot honour.
        """
        return replace(self, role=role, visible=(role is Role.Y))


@dataclass(frozen=True)
class DisplaySpec:
    """The plot a set of column roles asks for.

    ``x_index`` is ``None`` when no column is the abscissa — the row number is
    it. ``x_name`` is that column's label, carried here so a rename in the
    panel reaches the axis without a second lookup. ``y`` is the visible
    curves, in column order, each with the label to put in the legend.
    """

    x_index: int | None
    x_name: str | None
    y: tuple[tuple[int, str], ...]

    @classmethod
    def from_roles(cls, roles: list[ColumnRole]) -> "DisplaySpec":
        x = next(((i, c.name) for i, c in enumerate(roles) if c.role is Role.X), None)
        y = tuple(
            (i, c.name) for i, c in enumerate(roles) if c.role is Role.Y and c.visible
        )
        return cls(x_index=x[0] if x else None, x_name=x[1] if x else None, y=y)

    @property
    def has_curves(self) -> bool:
        return bool(self.y)


def _labels(n_cols: int, names: tuple[str, ...]) -> list[str]:
    return [
        names[i] if i < len(names) and str(names[i]).strip() else f"Col {i}"
        for i in range(n_cols)
    ]


def default_column_roles(
    n_cols: int,
    names: tuple[str, ...] = (),
    is_text: bool = False,
) -> list[ColumnRole]:
    """How a freshly opened block draws before anyone touches the panel.

    The point is that nothing changes on screen: this reproduces what the
    viewer did on its own.

    * One column has no abscissa but itself, so it is a curve against the row
      index.
    * A text file writes ``x  y`` — the first column is the abscissa. With two
      columns that is a single curve; with more, the extra columns are curves
      too but start hidden, so a wide file still opens as the table it was and
      the panel is how you bring a column into the plot.
    * Anything else (an HDF5 dataset a few columns wide) has no such
      convention, so every column is a curve against the row index, which is
      what the multi-curve view already did.
    """
    if n_cols <= 0:
        return []
    labels = _labels(n_cols, names)

    if n_cols == 1:
        return [ColumnRole(labels[0], Role.Y, True)]

    if is_text:
        first = ColumnRole(labels[0], Role.X, False)
        rest_visible = n_cols == 2
        return [first] + [
            ColumnRole(labels[i], Role.Y, rest_visible) for i in range(1, n_cols)
        ]

    return [ColumnRole(labels[i], Role.Y, True) for i in range(n_cols)]

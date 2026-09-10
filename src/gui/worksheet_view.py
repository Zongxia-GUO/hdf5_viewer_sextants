"""An Origin-style worksheet: the column controls live in the header itself.

The top of every column carries a show box and an X/Y combo, the column name
below it, and — only where the Data editor asks for it — an ``f(x)`` field
below that. Those controls are child widgets of the table's own
``QHeaderView``, positioned from the very section geometry that lays out the
data columns, so a control cell and its column can never drift apart. Below the
header is nothing but the numbers, read only.

The controls are a view over a :class:`~src.gui.table_model.ColumnRolesModel`,
which keeps the one rule a plot cannot state two ways: a single abscissa, so
choosing X for a column drops it from whoever held it.
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
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.table_model import ColumnRolesModel, CopyableTableView, DataTable
from src.lib_h5.columns import Role

#: Band heights in the header. The role band (checkbox + combo) needs a little
#: more air than the plain text bands (name, f(x)). paintSection rules a divider
#: at each band boundary so a column reads as stacked cells.
_ROLE_BAND_HEIGHT = 28
_TEXT_BAND_HEIGHT = 22

#: Wide enough for ``X``/``Y`` plus the drop-down arrow on every style.
_COMBO_WIDTH = 54

#: A column has to hold ``[x] [X v]`` at its narrowest.
_MIN_COLUMN_WIDTH = 78
_DEFAULT_COLUMN_WIDTH = 92

#: Bare strip left uncovered at each section's right edge so the header's own
#: resize handle stays grabbable through the control widgets.
_RESIZE_GRIP = 8

#: The name and f(x) fields are editable but should read as labels until then.
_FLAT_EDIT_QSS = "QLineEdit{border:none;background:transparent;padding:0 3px;}"


class _ColumnControls(QWidget):
    """The stack of controls for one column, living inside the header."""

    def __init__(
        self,
        on_role,
        on_show,
        on_rename,
        show_fx: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_role = on_role
        self._on_show = on_show
        self._on_rename = on_rename

        self.show_box = QCheckBox(self)
        self.show_box.setToolTip("Draw this column")
        self.combo = QComboBox(self)
        self.combo.addItems([Role.X.value, Role.Y.value])
        self.combo.setFixedWidth(_COMBO_WIDTH)
        self.combo.setToolTip("Abscissa or curve")

        self.name_edit = QLineEdit(self)
        self.name_edit.setStyleSheet(_FLAT_EDIT_QSS)
        self.name_edit.setToolTip("Column name — click to rename")

        # One widget per band, each a fixed height that matches where
        # paintSection rules the divider between them.
        role_band = QWidget(self)
        role_band.setFixedHeight(_ROLE_BAND_HEIGHT)
        role_row = QHBoxLayout(role_band)
        role_row.setContentsMargins(3, 0, 3, 0)
        role_row.setSpacing(3)
        role_row.addWidget(self.show_box)
        role_row.addWidget(self.combo)
        role_row.addStretch(1)
        self.name_edit.setFixedHeight(_TEXT_BAND_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(role_band)
        layout.addWidget(self.name_edit)

        self.fx_edit: QLineEdit | None = None
        if show_fx:
            self.fx_edit = QLineEdit(self)
            self.fx_edit.setStyleSheet(_FLAT_EDIT_QSS)
            self.fx_edit.setPlaceholderText("f(x)=")
            self.fx_edit.setFixedHeight(_TEXT_BAND_HEIGHT)
            layout.addWidget(self.fx_edit)
        layout.addStretch(1)

        self.show_box.toggled.connect(lambda checked: self._on_show(bool(checked)))
        self.combo.currentTextChanged.connect(self._role_picked)
        self.name_edit.editingFinished.connect(
            lambda: self._on_rename(self.name_edit.text())
        )

    def set_state(self, role: Role, name: str, visible: bool) -> None:
        """Match the widgets to a column's role without echoing signals back."""
        for widget in (self.show_box, self.combo, self.name_edit):
            widget.blockSignals(True)
        self.combo.setCurrentText(role.value)
        # The box is live for both roles: on a Y it draws the curve, on the X
        # it makes that column the abscissa (off ⇒ the row index is).
        active_role = role is not Role.NONE
        self.show_box.setEnabled(active_role)
        self.show_box.setChecked(active_role and visible)
        self.show_box.setToolTip(
            "Use as the X axis" if role is Role.X else "Draw this column"
        )
        if not self.name_edit.hasFocus():
            self.name_edit.setText(name)
        for widget in (self.show_box, self.combo, self.name_edit):
            widget.blockSignals(False)

    def _role_picked(self, text: str) -> None:
        try:
            self._on_role(Role(text))
        except ValueError:
            pass


class ColumnHeaderView(QHeaderView):
    """A horizontal header that hosts an :class:`_ColumnControls` per column.

    Every cell is placed from ``sectionViewportPosition`` / ``sectionSize`` —
    the same geometry the view uses for the data columns — so alignment is
    exact by construction and follows a resize or a scroll for free.
    """

    def __init__(self, show_fx: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._show_fx = show_fx
        self._roles: ColumnRolesModel | None = None
        self._cells: list[_ColumnControls] = []

        self.setSectionsClickable(False)
        self.setSectionsMovable(False)
        self.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.setMinimumSectionSize(_MIN_COLUMN_WIDTH)
        self.setDefaultSectionSize(_DEFAULT_COLUMN_WIDTH)
        self.setStretchLastSection(False)

        self.sectionResized.connect(lambda *_: self._reposition())
        self.sectionCountChanged.connect(lambda *_: self._reposition())
        self.geometriesChanged.connect(self._reposition)

    # -- sizing --------------------------------------------------------- #

    def _band_heights(self) -> list[int]:
        heights = [_ROLE_BAND_HEIGHT, _TEXT_BAND_HEIGHT]
        if self._show_fx:
            heights.append(_TEXT_BAND_HEIGHT)
        return heights

    def _total_height(self) -> int:
        return sum(self._band_heights())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        base = super().sizeHint()
        return QSize(base.width(), self._total_height())

    def paintSection(self, painter, rect, logical_index) -> None:  # noqa: N802
        """A flat panel ruled into stacked cells; the controls are widgets on top.

        The default would paint the model's column name, which shows through the
        gaps in the control stack as a ghost. Instead this draws the same thin
        grid the data cells below have: a divider down the right of the section,
        one along the bottom, and one at each band boundary.
        """
        palette = self.palette()
        painter.fillRect(rect, palette.color(QPalette.ColorRole.Button))
        painter.save()
        painter.setPen(palette.color(QPalette.ColorRole.Mid))
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        offset = 0
        for band_height in self._band_heights()[:-1]:
            offset += band_height
            y = rect.top() + offset
            painter.drawLine(rect.left(), y, rect.right(), y)
        painter.restore()

    # -- contents ----------------------------------------------------- #

    def set_roles(self, roles: ColumnRolesModel | None) -> None:
        self._roles = roles
        self._rebuild()

    def refresh(self) -> None:
        """Push the current roles back onto the widgets (after any edit)."""
        if self._roles is None:
            return
        entries = self._roles.roles()
        for col, cell in enumerate(self._cells):
            if col < len(entries):
                entry = entries[col]
                cell.set_state(entry.role, entry.name, entry.visible)

    def _rebuild(self) -> None:
        for cell in self._cells:
            cell.deleteLater()
        self._cells = []
        if self._roles is None:
            return
        for col in range(len(self._roles.roles())):
            cell = _ColumnControls(
                on_role=lambda role, c=col: self._route_role(c, role),
                on_show=lambda visible, c=col: self._route_show(c, visible),
                on_rename=lambda name, c=col: self._route_rename(c, name),
                show_fx=self._show_fx,
                parent=self.viewport(),
            )
            cell.show()
            self._cells.append(cell)
        self.refresh()
        self._reposition()

    # -- placement -------------------------------------------------- #

    def _reposition(self, *_args) -> None:
        height = self.viewport().height()
        for col, cell in enumerate(self._cells):
            if self.isSectionHidden(col):
                cell.setVisible(False)
                continue
            width = max(0, self.sectionSize(col) - _RESIZE_GRIP)
            cell.setGeometry(self.sectionViewportPosition(col), 0, width, height)
            cell.setVisible(True)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._reposition()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._reposition()

    # -- edits go through the roles model -------------------------- #

    def _route_role(self, col: int, role: Role) -> None:
        if self._roles is not None:
            self._roles.setData(
                self._roles.index(col, ColumnRolesModel.COL_ROLE),
                role.value,
                Qt.ItemDataRole.EditRole,
            )

    def _route_show(self, col: int, visible: bool) -> None:
        if self._roles is not None:
            state = Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
            self._roles.setData(
                self._roles.index(col, ColumnRolesModel.COL_SHOW),
                state.value,
                Qt.ItemDataRole.CheckStateRole,
            )

    def _route_rename(self, col: int, name: str) -> None:
        if self._roles is None:
            return
        self._roles.setData(
            self._roles.index(col, ColumnRolesModel.COL_NAME),
            name,
            Qt.ItemDataRole.EditRole,
        )
        # A blank or unchanged name is refused and emits nothing, which would
        # leave the field showing the rejected text — put the real name back.
        self.refresh()


class WorksheetView(CopyableTableView):
    """A read-only data grid whose header carries the per-column controls."""

    #: Re-emitted from the roles model so a panel can debounce a redraw.
    spec_changed = pyqtSignal()

    def __init__(self, show_fx: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._roles: ColumnRolesModel | None = None
        self._grid: DataTable | None = None

        self._header = ColumnHeaderView(show_fx, self)
        self.setHorizontalHeader(self._header)
        self.setEditTriggers(CopyableTableView.EditTrigger.NoEditTriggers)  # read only
        self.setHorizontalScrollMode(CopyableTableView.ScrollMode.ScrollPerPixel)
        self.horizontalScrollBar().valueChanged.connect(self._header._reposition)

    # -- population -------------------------------------------------- #

    def set_data(self, array: np.ndarray, roles: ColumnRolesModel) -> None:
        """Show ``array`` with ``roles`` driving the header controls."""
        if self._roles is not None:
            try:
                self._roles.spec_changed.disconnect(self._on_roles_changed)
            except TypeError:
                pass
        self._roles = roles

        old = self._grid
        self._grid = DataTable(np.asarray(array), [c.name for c in roles.roles()])
        self.setModel(self._grid)
        if old is not None:
            old.deleteLater()

        self._header.set_roles(roles)
        roles.spec_changed.connect(self._on_roles_changed)

    def clear(self) -> None:
        if self._roles is not None:
            try:
                self._roles.spec_changed.disconnect(self._on_roles_changed)
            except TypeError:
                pass
        self._roles = None
        self._header.set_roles(None)
        self.setModel(None)
        if self._grid is not None:
            self._grid.deleteLater()
        self._grid = None

    def header_controls(self) -> list[_ColumnControls]:
        return list(self._header._cells)

    def column_roles(self) -> ColumnRolesModel | None:
        return self._roles

    # -- keeping the two halves in step --------------------------- #

    def _on_roles_changed(self) -> None:
        # A rename retitles the grid columns; a role flip or the single-X
        # demotion moves another column's combo.
        if self._grid is not None and self._roles is not None:
            self._grid.set_column_names([c.name for c in self._roles.roles()])
        self._header.refresh()
        self.spec_changed.emit()

"""The Columns tab of the main window's side panel.

The main window is a browser: this panel does not change the numbers, it
chooses how the block already on screen is drawn. One row per column, each with
a role — the abscissa, a curve, or left out — and a show box. Editing a row
emits a :class:`DisplaySpec`; the window hands that to the data view, which
swaps between the table and a multi-curve plot accordingly.

The richer job — per-column ``f(x)``, a second Y axis, colour, several blocks
at once — belongs to the Data editor, not here.
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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from src.gui.table_model import ColumnRolesModel, CopyableTableView, DataTable
from src.lib_h5.columns import ColumnRole, DisplaySpec, Role, default_column_roles

#: A change in the roles table waits this long for the next one before the plot
#: is asked to redraw, so dragging through the role combo does not repaint on
#: every step.
REDRAW_DEBOUNCE_MS = 120

#: Past this many columns a block is a detector frame, not a worksheet — one
#: row per column would be a scroll of hundreds and every one of them a curve.
#: Those open as an image and the panel steps aside.
MAX_COLUMNS = 64


class _RoleDelegate(QStyledItemDelegate):
    """A combo of the three roles for the Role column."""

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt override
        combo = QComboBox(parent)
        combo.addItems([role.value for role in Role])
        return combo

    def setEditorData(self, editor, index):  # noqa: N802 - Qt override
        current = index.data(Qt.ItemDataRole.EditRole)
        pos = editor.findText(str(current))
        if pos >= 0:
            editor.setCurrentIndex(pos)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt override
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class ColumnViewPanel(QWidget):
    """Roles table over a read-only grid; emits a DisplaySpec on every edit."""

    display_spec_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        self._data: np.ndarray | None = None
        self._source_key: str | None = None
        #: True only while :meth:`set_data` is filling the model, so the reset
        #: it triggers is not mistaken for an edit and does not queue a redraw.
        self._loading = False
        #: Role choices are kept for the life of the session so flipping between
        #: two datasets does not throw away what you set on each.
        self._roles_by_key: dict[str, list[ColumnRole]] = {}

        self._roles_model = ColumnRolesModel(parent=self)
        self._roles_view = CopyableTableView()
        self._roles_view.setModel(self._roles_model)
        self._roles_view.setItemDelegateForColumn(
            ColumnRolesModel.COL_ROLE, _RoleDelegate(self._roles_view)
        )
        v_header = self._roles_view.verticalHeader()
        if v_header is not None:
            v_header.setVisible(False)
        roles_header = self._roles_view.horizontalHeader()
        if roles_header is not None:
            roles_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            roles_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            roles_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._roles_view.setColumnWidth(ColumnRolesModel.COL_ROLE, 64)
        self._roles_view.setColumnWidth(ColumnRolesModel.COL_SHOW, 52)
        self._roles_view.setEditTriggers(
            CopyableTableView.EditTrigger.DoubleClicked
            | CopyableTableView.EditTrigger.SelectedClicked
            | CopyableTableView.EditTrigger.EditKeyPressed
        )

        self._grid = CopyableTableView()
        self._grid.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._roles_view)
        splitter.addWidget(self._grid)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([160, 400])

        self._placeholder = QLabel("Not a column dataset.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: palette(mid); padding: 24px;")

        self._stack = QStackedWidget()
        self._stack.addWidget(splitter)          # index 0
        self._stack.addWidget(self._placeholder)  # index 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(REDRAW_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_spec)
        self._roles_model.spec_changed.connect(self._on_roles_changed)

    # -- population -------------------------------------------------------- #

    def set_data(
        self,
        data: np.ndarray | None,
        names: tuple[str, ...] = (),
        source_key: str | None = None,
        is_text: bool = False,
    ) -> None:
        """Show the columns of ``data``, or the placeholder if it has none.

        A 2-D numeric array is the only thing with columns to assign; an image,
        a stack, a structured array or text all fall through to the placeholder
        and leave the Attributes tab to do the talking.
        """
        self._source_key = source_key
        arr = np.asarray(data) if data is not None else None

        if (
            arr is None
            or arr.ndim != 2
            or not 1 <= arr.shape[1] <= MAX_COLUMNS
            or arr.dtype.names is not None
            or arr.dtype.kind not in "fiu"
        ):
            self._data = None
            self._grid.setModel(None)
            self._roles_model.set_roles([])
            self._stack.setCurrentIndex(1)
            return

        self._data = arr
        n_cols = arr.shape[1]
        roles = self._roles_by_key.get(source_key or "")
        if roles is None or len(roles) != n_cols:
            roles = default_column_roles(n_cols, names, is_text)

        grid_names = [c.name for c in roles]
        self._grid.setModel(DataTable(arr, grid_names))

        # Fill the model without the reset it triggers being taken for an edit
        # and queuing a redraw for a selection the user did not make.
        self._loading = True
        try:
            self._roles_model.set_roles(roles)
        finally:
            self._loading = False
        self._stack.setCurrentIndex(0)
        self._store_roles()

    def current_spec(self) -> DisplaySpec:
        return self._roles_model.display_spec()

    def has_columns(self) -> bool:
        return self._data is not None

    # -- reacting to edits ----------------------------------------------- #

    def _on_roles_changed(self) -> None:
        if self._loading:
            return
        self._sync_grid_headers()
        self._store_roles()
        self._debounce.start()

    def _sync_grid_headers(self) -> None:
        """Keep the grid's column titles in step with a renamed role."""
        model = self._grid.model()
        if isinstance(model, DataTable) and self._data is not None:
            model.set_column_names([c.name for c in self._roles_model.roles()])

    def _store_roles(self) -> None:
        if self._source_key is not None:
            self._roles_by_key[self._source_key] = self._roles_model.roles()

    def _emit_spec(self) -> None:
        if self._data is not None:
            self.display_spec_changed.emit(self._roles_model.display_spec())

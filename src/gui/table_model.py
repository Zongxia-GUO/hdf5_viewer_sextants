"""Children of QAbstractTableModel."""

# Copyright (C) 2023 Dennis Leonard
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

from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication, QAbstractItemView, QMenu, QTableView

from src.lib_h5.columns import DisplaySpec, Role


class CopyableTableView(QTableView):
    """QTableView with right-click and Ctrl+C copy support."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_copy_menu)

    def keyPressEvent(self, event: Any) -> None:
        """Copy selected cells on Ctrl+C."""
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection_to_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

    def _show_copy_menu(self, pos: Any) -> None:
        menu = QMenu(self)
        action = menu.addAction("Copy")
        action.setEnabled(bool(self.selectedIndexes()) or self.currentIndex().isValid())
        action.triggered.connect(self.copy_selection_to_clipboard)
        menu.exec(self.viewport().mapToGlobal(pos))

    def copy_selection_to_clipboard(self) -> None:
        """Copy the selected table cells as tab-separated text."""
        model = self.model()
        if model is None:
            return

        indexes = self.selectedIndexes()
        if not indexes and self.currentIndex().isValid():
            indexes = [self.currentIndex()]
        if not indexes:
            return

        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        values: dict[tuple[int, int], str] = {}

        for index in indexes:
            value = model.data(index, Qt.ItemDataRole.DisplayRole)
            values[(index.row(), index.column())] = "" if value is None else str(value)

        text = "\n".join(
            "\t".join(values.get((row, column), "") for column in columns)
            for row in rows
        )
        QApplication.clipboard().setText(text)


class TableModel(QAbstractTableModel):
    """Table Model that can append and remove Rows."""

    def __init__(self, header: list[str]) -> None:
        """Table Model that can append and remove Rows."""
        QAbstractTableModel.__init__(self)
        self._header = header
        self._data: list[Any] = []

    def rowCount(self, parent: None | QModelIndex = None) -> int:
        """Get Row Count."""
        return len(self._data)

    def columnCount(self, parent: None | QModelIndex = None) -> int:
        """Get Column Count."""
        return len(self._header)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Item Flags for Cell at Index."""
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

    def appendRow(self, new_data: list[Any]) -> bool:
        """Append Row."""
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._data.append(new_data)
        self.endInsertRows()
        return True

    def removeRow(self, row: int, parent: None | QModelIndex = None) -> bool:
        """Remove Row."""
        self.beginRemoveRows(QModelIndex(), row, row)
        try:
            self._data.pop(row)
        except IndexError:
            self.endRemoveRows()
            return False
        self.endRemoveRows()
        return True

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Get Data, Alignment, Colors etc. depending on Role."""
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]

    def setData(self, index: QModelIndex, value: str, role: int = Qt.ItemDataRole.EditRole) -> bool:
        """Set Data when Cell is edited."""
        if role != Qt.ItemDataRole.EditRole:
            return False

        self._data[index.row()][index.column()] = value
        self.dataChanged.emit(index, index)
        return True

    def getData(self, index: Any = None) -> Any:
        """Get Data at Index or Row."""
        if index is None:
            return self._data
        elif isinstance(index, QModelIndex):
            return self._data[index.row()][index.column()]
        elif isinstance(index, int):
            return self._data[index]

    def resetData(self) -> None:
        """Reset to Empty Table."""
        for i in range(self.rowCount()):
            self.removeRow(0)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> None | str | int:
        """Get Headers for horizontal | vertical Orientation."""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._header[section]
            if orientation == Qt.Orientation.Vertical:
                return section + 1
        return None


class DataTable(QAbstractTableModel):
    """Table Model for 2D Numpy Arrays with enhanced display features."""

    def __init__(self, data: npt.NDArray, column_names: list[str] | None = None) -> None:
        """
        Table Model for 2D Numpy Arrays.

        :param data: 2D numpy array to display
        :param column_names: Optional list of column names
        """
        QAbstractTableModel.__init__(self)

        # Ensure data is at most 2D for display
        if data.dtype.names is None and data.ndim > 2:
            # Flatten leading dimensions: (1, 2048, 2048) ->(2048, 2048)
            data = data.reshape(-1, data.shape[-1])

        self._data = data
        self._column_names = column_names

        # Auto-detect if data is structured array
        if data.dtype.names is not None:
            self._is_structured = True
            self._column_names = list(data.dtype.names)
        else:
            self._is_structured = False
            if column_names is None:
                # Generate default column names
                self._column_names = [f"Col {i}" for i in range(self.columnCount())]

    def rowCount(self, parent: None | QModelIndex = None) -> int:
        """Get Row Count."""
        return int(self._data.shape[0])

    def columnCount(self, parent: None | QModelIndex = None) -> int:
        """Get Column Count."""
        if self._is_structured:
            return len(self._column_names)
        return int(self._data.shape[1]) if len(self._data.shape) > 1 else 1

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Item Flags for Cell at Index."""
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> None | str:
        """Get Data, Alignment, Colors etc. depending on Role."""
        if role == Qt.ItemDataRole.DisplayRole:
            try:
                # Get the raw value
                if self._is_structured:
                    col_name = self._column_names[index.column()]
                    value = self._data[col_name][index.row()]
                else:
                    if len(self._data.shape) > 1:
                        value = self._data[index.row()][index.column()]
                    else:
                        value = self._data[index.row()]

                # Format the value based on its type
                return self._format_value(value)

            except (IndexError, KeyError, ValueError, TypeError) as e:
                return f"Error: {str(e)}"

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            # Right-align numeric values
            try:
                if self._is_structured:
                    col_name = self._column_names[index.column()]
                    value = self._data[col_name][index.row()]
                else:
                    value = (
                        self._data[index.row()][index.column()] if len(self._data.shape) > 1
                        else self._data[index.row()]
                    )

                if isinstance(value, (int, float, complex)) or self._is_numeric_type(value):
                    return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            except Exception:
                pass

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> None | str | int:
        """Get Headers for horizontal | vertical Orientation."""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                # Column headers
                if section < len(self._column_names):
                    return self._column_names[section]
                return f"Col {section}"
            elif orientation == Qt.Orientation.Vertical:
                # Row numbers (1-indexed)
                return str(section + 1)
        return None

    def set_column_names(self, names: list[str]) -> None:
        """Rename the columns in place; the Columns panel edits these labels."""
        if self._is_structured:
            return
        self._column_names = list(names)
        self.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, max(0, len(names) - 1)
        )

    def _format_value(self, value: Any) -> str:
        """
        Format a value for display in the table.

        :param value: The value to format
        :return: Formatted string representation
        """
        # Handle bytes
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return str(value)

        # Handle strings
        if isinstance(value, str):
            return value

        # Handle None/NaN
        if value is None:
            return "None"

        # Handle numpy scalar types
        try:
            if isinstance(value, (np.floating, float)):
                if np.isnan(value):
                    return "NaN"
                elif np.isinf(value):
                    return "Inf" if value > 0 else "-Inf"
                else:
                    # Format float with reasonable precision
                    if abs(value) < 1e-4 or abs(value) > 1e6:
                        return f"{value:.4e}"  # Scientific notation
                    else:
                        return f"{value:.6g}"  # General format

            elif isinstance(value, (np.integer, int)):
                return str(value)

            elif isinstance(value, (np.complexfloating, complex)):
                return f"{value:.4g}"

            elif isinstance(value, (np.bool_, bool)):
                return "True" if value else "False"

        except Exception:
            pass

        # Fallback: convert to string
        return str(value)

    @staticmethod
    def _is_numeric_type(value: Any) -> bool:
        """Check if a value is numeric type."""
        try:
            return isinstance(value, (int, float, complex, np.number))
        except Exception:
            return isinstance(value, (int, float, complex))


class ColumnRolesModel(QAbstractTableModel):
    """The columns of a 2-D block as editable rows: label, role, whether shown.

    One row per column of the data. The role decides what a column is for when
    the block is drawn — the abscissa, a curve, or left out — and this model
    keeps the one rule that a plot cannot express two ways: there is at most one
    ``X``, so promoting a column to it demotes whoever held it. ``spec_changed``
    fires after any edit; the panel that owns the model debounces it before
    redrawing.
    """

    COL_NAME, COL_ROLE, COL_SHOW = 0, 1, 2
    _HEADERS = ("Name", "Role", "Show")

    spec_changed = pyqtSignal()

    def __init__(self, roles: list | None = None, parent: Any = None) -> None:
        super().__init__(parent)
        self._roles: list = list(roles or [])

    # -- contents ------------------------------------------------------------ #

    def set_roles(self, roles: list) -> None:
        """Replace every row, e.g. when a new dataset is selected."""
        self.beginResetModel()
        self._roles = list(roles)
        self.endResetModel()
        self.spec_changed.emit()

    def roles(self) -> list:
        return list(self._roles)

    def display_spec(self) -> DisplaySpec:
        return DisplaySpec.from_roles(self._roles)

    # -- Qt model API ------------------------------------------------------- #

    def rowCount(self, parent: None | QModelIndex = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._roles)

    def columnCount(self, parent: None | QModelIndex = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._HEADERS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> None | str | int:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._HEADERS[section]
        return section + 1

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        col = index.column()
        if col in (self.COL_NAME, self.COL_ROLE):
            flags |= Qt.ItemFlag.ItemIsEditable
        elif col == self.COL_SHOW and self._roles[index.row()].role is Role.Y:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        entry = self._roles[index.row()]
        col = index.column()

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if col == self.COL_NAME:
                return entry.name
            if col == self.COL_ROLE:
                return entry.role.value
            return None

        if role == Qt.ItemDataRole.CheckStateRole and col == self.COL_SHOW:
            if entry.role is not Role.Y:
                return None
            return (
                Qt.CheckState.Checked if entry.visible else Qt.CheckState.Unchecked
            )

        if role == Qt.ItemDataRole.TextAlignmentRole and col != self.COL_NAME:
            return Qt.AlignmentFlag.AlignCenter

        return None

    def setData(
        self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        entry = self._roles[row]

        if col == self.COL_NAME and role == Qt.ItemDataRole.EditRole:
            name = str(value).strip()
            if not name or name == entry.name:
                return False
            self._roles[row] = replace(entry, name=name)
            self.dataChanged.emit(index, index)
            self.spec_changed.emit()
            return True

        if col == self.COL_ROLE and role == Qt.ItemDataRole.EditRole:
            try:
                new_role = Role(str(value))
            except ValueError:
                return False
            if new_role is entry.role:
                return False
            self._apply_role(row, new_role)
            return True

        if col == self.COL_SHOW and role == Qt.ItemDataRole.CheckStateRole:
            if entry.role is not Role.Y:
                return False
            visible = Qt.CheckState(value) == Qt.CheckState.Checked
            if visible == entry.visible:
                return False
            self._roles[row] = replace(entry, visible=visible)
            self.dataChanged.emit(index, index, [role])
            self.spec_changed.emit()
            return True

        return False

    # -- the single-X rule ------------------------------------------------- #

    def _apply_role(self, row: int, new_role: Role) -> None:
        touched = {row}
        if new_role is Role.X:
            for i, other in enumerate(self._roles):
                if i != row and other.role is Role.X:
                    # Demotion is a side effect, not a choice to plot it: the
                    # old abscissa becomes an available Y, not a live curve.
                    self._roles[i] = replace(other, role=Role.Y, visible=False)
                    touched.add(i)
        self._roles[row] = self._roles[row].with_role(new_role)
        lo, hi = min(touched), max(touched)
        self.dataChanged.emit(
            self.index(lo, 0), self.index(hi, self.columnCount() - 1)
        )
        self.spec_changed.emit()

"""The Origin-style worksheet: controls inside the header, read-only data below.

The point of putting the controls in the table's own ``QHeaderView`` is that a
control cell is placed from the same section geometry as its data column, so
they cannot drift. These tests cover the wiring — a control routes to the roles
model, a rename reaches the grid, choosing X for one column drops it from
another — and that the data stays read only.
"""

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from src.gui.table_model import ColumnRolesModel, DataTable
from src.gui.worksheet_view import _RESIZE_GRIP, ColumnHeaderView, WorksheetView
from src.lib_h5.columns import ColumnRole, Role, default_column_roles

ARRAY = np.c_[np.arange(4.0), np.arange(4.0) ** 2, np.arange(4.0) * 3]


def make_roles(*entries: ColumnRole) -> ColumnRolesModel:
    return ColumnRolesModel(list(entries))


@pytest.fixture
def view(qapp):
    widget = WorksheetView(show_fx=False)
    yield widget
    widget.clear()
    widget.deleteLater()


def wide_text_roles(n=3):
    return ColumnRolesModel(default_column_roles(n, (), is_text=True))


# ── the data half ───────────────────────────────────────────────────── #

def test_the_grid_is_the_numbers_read_only(view):
    view.set_data(ARRAY, wide_text_roles())

    model = view.model()
    assert isinstance(model, DataTable)
    assert model.rowCount() == 4
    assert model.columnCount() == 3
    assert model.data(model.index(1, 2)) == "3"  # arange(4)*3 second entry
    editable = Qt.ItemFlag.ItemIsEditable
    assert not (model.flags(model.index(0, 0)) & editable)


def test_the_row_numbers_start_at_one(view):
    view.set_data(ARRAY, wide_text_roles())
    model = view.model()

    assert model.headerData(0, Qt.Orientation.Vertical) == "1"
    assert model.headerData(3, Qt.Orientation.Vertical) == "4"


# ── the header controls ────────────────────────────────────────────── #

def test_one_control_stack_per_column(view):
    view.set_data(ARRAY, wide_text_roles())

    assert len(view.header_controls()) == 3


def test_a_control_reflects_its_column_role(view):
    view.set_data(
        ARRAY[:, :2],
        make_roles(ColumnRole("t", Role.X, True), ColumnRole("a", Role.Y, True)),
    )

    x_cell, y_cell = view.header_controls()
    assert x_cell.combo.currentText() == "X"
    assert x_cell.name_edit.text() == "t"
    assert x_cell.show_box.isEnabled(), "the box makes this column the abscissa"
    assert x_cell.show_box.isChecked()
    assert y_cell.combo.currentText() == "Y"
    assert y_cell.show_box.isChecked()


def test_switching_the_x_box_off_plots_against_the_row_index(view):
    roles = make_roles(ColumnRole("t", Role.X, True), ColumnRole("a", Role.Y, True))
    view.set_data(ARRAY[:, :2], roles)
    assert roles.display_spec().x_index == 0

    view.header_controls()[0].show_box.setChecked(False)

    assert roles.display_spec().x_index is None
    assert view.header_controls()[0].combo.currentText() == "X", "still marked X"


def test_the_combo_is_visible_and_a_real_dropdown(view):
    """It was shrunk to nothing once; keep it a usable control."""
    view.set_data(ARRAY, wide_text_roles())

    combo = view.header_controls()[0].combo
    assert combo.width() >= 40
    assert combo.count() == 2


def test_the_show_box_drives_the_curve(view):
    roles = wide_text_roles()
    view.set_data(ARRAY, roles)
    assert roles.display_spec().y == ()  # wide text opens with the curves hidden

    view.header_controls()[1].show_box.setChecked(True)

    assert roles.display_spec().y == ((1, "Col 1"),)


def test_the_combo_moves_x_and_the_other_control_follows(view):
    roles = make_roles(
        ColumnRole("t", Role.X, False), ColumnRole("a", Role.Y, True)
    )
    view.set_data(ARRAY[:, :2], roles)

    view.header_controls()[1].combo.setCurrentText("X")

    assert roles.roles()[1].role is Role.X
    assert roles.roles()[0].role is Role.Y, "the old X became an ordinary curve"
    assert roles.roles()[0].visible is False, "and was not silently switched on"
    assert view.header_controls()[0].combo.currentText() == "Y"
    assert view.header_controls()[0].show_box.isEnabled()


def test_renaming_in_the_header_reaches_the_grid_and_the_spec(view):
    roles = wide_text_roles()
    view.set_data(ARRAY, roles)
    seen = []
    view.spec_changed.connect(lambda: seen.append(True))

    cell = view.header_controls()[1]
    cell.name_edit.setText("photodiode")
    cell.name_edit.editingFinished.emit()

    assert roles.roles()[1].name == "photodiode"
    grid = view.model()
    assert grid.headerData(1, Qt.Orientation.Horizontal) == "photodiode"
    assert seen, "a rename moved the legend text, so it is a spec change"


# ── alignment: cells sit exactly on their sections ─────────────────── #

def test_each_control_cell_tracks_its_section(qapp):
    widget = WorksheetView(show_fx=False)
    try:
        widget.resize(400, 300)
        widget.set_data(np.zeros((20, 4)), ColumnRolesModel(
            default_column_roles(4, (), is_text=True)))
        widget.show()
        qapp.processEvents()
        header = widget.horizontalHeader()

        for col, cell in enumerate(widget.header_controls()):
            assert cell.x() == header.sectionViewportPosition(col)
            # a bare strip is left for the header's own resize handle
            assert cell.width() == header.sectionSize(col) - _RESIZE_GRIP

        # a resize keeps them in step
        header.resizeSection(1, 140)
        qapp.processEvents()
        for col, cell in enumerate(widget.header_controls()):
            assert cell.x() == header.sectionViewportPosition(col)
            # a bare strip is left for the header's own resize handle
            assert cell.width() == header.sectionSize(col) - _RESIZE_GRIP
    finally:
        widget.close()
        widget.deleteLater()


# ── lifecycle ─────────────────────────────────────────────────────── #

def test_clear_tears_everything_down(view):
    view.set_data(ARRAY, wide_text_roles())
    view.clear()

    assert view.model() is None
    assert view.header_controls() == []
    assert view.column_roles() is None


def test_reusing_the_view_for_another_block_rebuilds_the_header(view):
    view.set_data(ARRAY, wide_text_roles(3))
    view.set_data(ARRAY[:, :2], wide_text_roles(2))

    assert len(view.header_controls()) == 2
    assert view.model().columnCount() == 2


# ── the f(x) row, for the Data editor ─────────────────────────────── #

def test_the_fx_field_appears_only_when_asked_for(qapp):
    plain = WorksheetView(show_fx=False)
    editor = WorksheetView(show_fx=True)
    try:
        for v in (plain, editor):
            v.set_data(ARRAY, wide_text_roles())
        assert plain.header_controls()[0].fx_edit is None
        assert editor.header_controls()[0].fx_edit is not None
    finally:
        for v in (plain, editor):
            v.clear()
            v.deleteLater()


def test_the_header_height_grows_for_the_fx_row(qapp):
    two = ColumnHeaderView(show_fx=False)
    three = ColumnHeaderView(show_fx=True)
    assert three.sizeHint().height() > two.sizeHint().height()

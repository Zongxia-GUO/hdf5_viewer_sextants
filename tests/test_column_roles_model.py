"""The editable roles table behind the Columns panel.

One row per column of the data. The only rule the plot cannot express two ways
is that there is a single abscissa, so the model has to hold that: promoting a
column to X demotes whoever had it. Everything else is bookkeeping, but the
bookkeeping is what the panel debounces a redraw on, so it is worth pinning.
"""

import pytest
from PyQt6.QtCore import Qt

from src.gui.table_model import ColumnRolesModel
from src.lib_h5.columns import ColumnRole, Role

NAME, ROLE, SHOW = (
    ColumnRolesModel.COL_NAME,
    ColumnRolesModel.COL_ROLE,
    ColumnRolesModel.COL_SHOW,
)


@pytest.fixture
def model(qapp):
    return ColumnRolesModel(
        [
            ColumnRole("energy", Role.X, True),
            ColumnRole("i0", Role.Y, True),
            ColumnRole("i1", Role.Y, False),
        ]
    )


def role_at(model, row):
    return model.roles()[row].role


def set_role(model, row, value):
    return model.setData(model.index(row, ROLE), value, Qt.ItemDataRole.EditRole)


def set_show(model, row, checked):
    state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
    return model.setData(model.index(row, SHOW), state.value, Qt.ItemDataRole.CheckStateRole)


# ── shape ────────────────────────────────────────────────────────────── #

def test_a_row_per_column_and_three_fields(model):
    assert model.rowCount() == 3
    assert model.columnCount() == 3
    assert [
        model.headerData(c, Qt.Orientation.Horizontal) for c in range(3)
    ] == ["Name", "Role", "Show"]


# ── the single-X rule ────────────────────────────────────────────────── #

def test_promoting_a_column_to_x_demotes_the_old_one(model):
    changes = []
    model.spec_changed.connect(lambda: changes.append(model.display_spec()))

    set_role(model, 2, "X")

    assert role_at(model, 2) is Role.X
    assert role_at(model, 0) is Role.Y, "the old X became an ordinary curve"
    assert changes[-1].x_index == 2


def test_a_demoted_x_is_not_left_showing_a_stale_state(model):
    set_role(model, 1, "X")

    assert model.roles()[0].role is Role.Y
    assert model.roles()[0].visible is False, "demotion does not silently switch it on"


def test_setting_none_takes_a_column_out_of_the_plot(model):
    set_role(model, 1, "—")

    assert role_at(model, 1) is Role.NONE
    assert model.display_spec().y == ()


# ── the show box ─────────────────────────────────────────────────────── #

def test_show_is_checkable_on_x_and_y_but_not_an_unassigned_column(model):
    model.setData(model.index(2, ROLE), "—", Qt.ItemDataRole.EditRole)

    checkable = Qt.ItemFlag.ItemIsUserCheckable
    assert model.flags(model.index(0, SHOW)) & checkable   # X: is it the abscissa
    assert model.flags(model.index(1, SHOW)) & checkable   # Y: draw the curve
    assert not (model.flags(model.index(2, SHOW)) & checkable)  # NONE: nothing to show


def test_switching_the_x_column_off_drops_the_abscissa(model):
    assert model.display_spec().x_index == 0

    set_show(model, 0, False)

    assert model.display_spec().x_index is None
    assert model.roles()[0].role is Role.X, "still marked X, just not active"


def test_checking_show_adds_the_curve(model):
    assert model.display_spec().y == ((1, "i0"),)

    set_show(model, 2, True)

    assert model.display_spec().y == ((1, "i0"), (2, "i1"))


def test_the_check_state_reads_back_from_visibility(model):
    model.setData(model.index(2, ROLE), "—", Qt.ItemDataRole.EditRole)

    assert model.data(model.index(0, SHOW), Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    assert model.data(model.index(1, SHOW), Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    assert model.data(model.index(2, SHOW), Qt.ItemDataRole.CheckStateRole) is None


# ── rename ───────────────────────────────────────────────────────────── #

def test_a_rename_reaches_the_spec(model):
    model.setData(model.index(1, NAME), "  photodiode  ", Qt.ItemDataRole.EditRole)

    assert model.roles()[1].name == "photodiode"
    assert model.display_spec().y == ((1, "photodiode"),)


def test_a_blank_rename_is_refused(model):
    assert model.setData(model.index(1, NAME), "   ", Qt.ItemDataRole.EditRole) is False
    assert model.roles()[1].name == "i0"


# ── no-op edits do not churn the plot ───────────────────────────────── #

def test_setting_a_role_to_what_it_already_is_changes_nothing(model):
    fired = []
    model.spec_changed.connect(lambda: fired.append(1))

    assert set_role(model, 1, "Y") is False
    assert fired == []


def test_set_roles_replaces_every_row(model):
    model.set_roles([ColumnRole("only", Role.Y, True)])

    assert model.rowCount() == 1
    assert model.display_spec() .y == ((0, "only"),)

"""The Columns tab, and what the main window does with the spec it emits.

The main window is a browser: this tab does not edit the numbers, it chooses
how the block already on screen is drawn. So the tests are about which widget
the Data dock ends up showing, and about the tab standing aside for anything
that is not a column dataset.
"""

import time

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from src.gui.column_view_panel import MAX_COLUMNS, ColumnViewPanel
from src.gui.main_window import MainWindow
from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
from src.gui.table_model import ColumnRolesModel
from src.gui.unified_data_viewer import UnifiedDataViewer
from src.lib_h5.columns import Role

ROLE, SHOW = ColumnRolesModel.COL_ROLE, ColumnRolesModel.COL_SHOW


def spin(qapp, seconds=0.3):
    """Let the panel's redraw debounce fire."""
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()


def set_show(model, row, checked):
    state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
    model.setData(model.index(row, SHOW), state.value, Qt.ItemDataRole.CheckStateRole)


# ── the panel on its own ─────────────────────────────────────────────── #

@pytest.fixture
def panel(qapp):
    widget = ColumnViewPanel()
    yield widget
    widget.close()
    widget.deleteLater()


def test_a_2d_numeric_block_gets_a_row_per_column(panel):
    panel.set_data(np.zeros((10, 3)), ("a", "b", "c"), "k", is_text=False)

    assert panel.has_columns()
    assert [c.name for c in panel._roles_model.roles()] == ["a", "b", "c"]


@pytest.mark.parametrize("array", [
    np.zeros((8, 8, 8)),                         # a stack
    np.zeros(5),                                 # 1-D
    np.array([("x", 1)], dtype=[("s", "U1"), ("n", "i4")]),  # structured
    np.array([["a", "b"], ["c", "d"]]),          # text
    np.zeros((4, MAX_COLUMNS + 1)),              # a detector frame, not a worksheet
])
def test_anything_without_assignable_columns_shows_the_placeholder(panel, array):
    panel.set_data(array, (), "k", is_text=False)

    assert not panel.has_columns()


def test_role_choices_are_kept_per_dataset_for_the_session(panel):
    panel.set_data(np.zeros((6, 2)), ("x", "y"), "one", is_text=True)
    panel._roles_model.setData(
        panel._roles_model.index(0, ROLE), "Y", Qt.ItemDataRole.EditRole
    )
    assert panel._roles_model.roles()[0].role is Role.Y

    panel.set_data(np.zeros((6, 2)), ("p", "q"), "two", is_text=True)
    panel.set_data(np.zeros((6, 2)), ("x", "y"), "one", is_text=True)

    assert panel._roles_model.roles()[0].role is Role.Y, "came back as it was left"


def test_a_column_count_change_drops_the_remembered_roles(panel):
    panel.set_data(np.zeros((6, 2)), (), "k", is_text=True)
    panel.set_data(np.zeros((6, 3)), (), "k", is_text=True)

    assert panel._roles_model.rowCount() == 3


def test_editing_emits_a_spec_once_the_debounce_settles(panel, qapp):
    panel.set_data(np.zeros((6, 3)), ("x", "a", "b"), "k", is_text=True)
    seen = []
    panel.display_spec_changed.connect(seen.append)

    set_show(panel._roles_model, 1, True)
    set_show(panel._roles_model, 2, True)
    spin(qapp)

    assert len(seen) == 1, "the two edits coalesced into one redraw"
    assert seen[-1].y == ((1, "a"), (2, "b"))


def test_filling_the_panel_does_not_emit(panel, qapp):
    seen = []
    panel.display_spec_changed.connect(seen.append)

    panel.set_data(np.zeros((6, 2)), ("x", "y"), "k", is_text=True)
    spin(qapp)

    assert seen == [], "a selection the user did not make must not redraw"


# ── wired into the main window ──────────────────────────────────────── #

@pytest.fixture
def win(qapp):
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()


def current(win):
    viewer = win.dock_plot.widget()
    return viewer.current_widget if isinstance(viewer, UnifiedDataViewer) else viewer


def test_a_wide_text_file_opens_as_a_table_then_a_column_brings_the_plot(win, qapp):
    arr = np.c_[np.arange(5.0), np.arange(5.0) ** 2, np.arange(5.0) * 3]
    win._show_data(arr, "Array1D", source_dataset_key="C:/d/scan.txt::data")
    spin(qapp)

    assert not isinstance(current(win), PlotWidget1DEnhanced), "3 columns -> table"
    assert win.column_view_panel.has_columns()

    model = win.column_view_panel._roles_model
    set_show(model, 1, True)
    set_show(model, 2, True)
    spin(qapp)

    plot = current(win)
    assert isinstance(plot, PlotWidget1DEnhanced)
    assert plot.y_data.ndim == 2
    assert plot._curve_labels == ["Col 1", "Col 2"]


def test_turning_every_curve_off_goes_back_to_the_table(win, qapp):
    arr = np.c_[np.arange(5.0), np.arange(5.0) ** 2]
    win._show_data(arr, "Array1D", source_dataset_key="C:/d/scan.txt::data")
    spin(qapp)
    model = win.column_view_panel._roles_model  # 2-col text: col1 already shown

    set_show(model, 1, False)
    spin(qapp)

    assert not isinstance(current(win), PlotWidget1DEnhanced)


def test_choosing_an_x_column_redraws_against_it(win, qapp):
    arr = np.c_[np.arange(5.0) * 10, np.arange(5.0) ** 2, np.arange(5.0)]
    win._show_data(arr, "Array2D", source_dataset_key="C:/d/m.h5::grp/curves")
    spin(qapp)
    assert win.column_view_panel.has_columns()

    model = win.column_view_panel._roles_model
    model.setData(model.index(0, ROLE), "X", Qt.ItemDataRole.EditRole)
    spin(qapp)

    plot = current(win)
    assert isinstance(plot, PlotWidget1DEnhanced)
    np.testing.assert_array_equal(plot.x_data, arr[:, 0])
    assert plot.plot_widget.getAxis("bottom").labelText == "Col 0"


def test_a_rename_reaches_the_legend(win, qapp):
    arr = np.c_[np.arange(5.0), np.arange(5.0) ** 2, np.arange(5.0)]
    win._show_data(arr, "Array1D", source_dataset_key="C:/d/scan.txt::data")
    spin(qapp)
    model = win.column_view_panel._roles_model
    set_show(model, 1, True)
    set_show(model, 2, True)
    spin(qapp)

    model.setData(model.index(1, ColumnRolesModel.COL_NAME), "squared",
                  Qt.ItemDataRole.EditRole)
    spin(qapp)

    assert current(win)._curve_labels == ["squared", "Col 2"]


def test_an_image_dataset_leaves_the_panel_on_its_placeholder(win, qapp):
    win._show_data(np.zeros((128, 128)), "Array2D",
                   source_dataset_key="C:/d/m.h5::entry/image")
    spin(qapp)

    assert not win.column_view_panel.has_columns()

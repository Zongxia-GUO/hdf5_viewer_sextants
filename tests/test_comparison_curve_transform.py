"""Tests for the Curve Transform f(X)/f(Y) columns in the Data Comparison tool."""

import numpy as np
import pytest

from src.gui.data_comparison import (
    COL_ENERGY,
    COL_FX,
    COL_FY,
    COL_NAME,
    COL_POINTS,
    COL_XAXIS,
    COLUMN_COUNT,
    DataComparisonTool,
)


@pytest.fixture
def tool(qapp):
    """A comparison tool with two straight-line curves loaded."""
    widget = DataComparisonTool(tuple())
    widget.add_dataset_from_array("a::curve", np.arange(10, dtype=float))
    widget.add_dataset_from_array("b::curve", np.arange(10, dtype=float) * 2.0)
    return widget


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_columns_are_named_and_ordered(tool):
    table = tool.dataset_table
    assert table.columnCount() == COLUMN_COUNT
    labels = [table.horizontalHeaderItem(c).text() for c in range(COLUMN_COUNT)]
    assert labels == ["Dataset", "Points", "E(eV)", "f(Y)", "X Axis", "f(X)"]
    assert (COL_NAME, COL_POINTS, COL_ENERGY, COL_FY, COL_XAXIS, COL_FX) == (0, 1, 2, 3, 4, 5)


def test_toggle_buttons_match_new_columns(tool):
    assert tool.btn_toggle_energy.text() == "E(eV)"
    assert tool.btn_toggle_fy.text() == "f(Y)"
    assert tool.btn_toggle_xaxis.text() == "X Axis"
    assert tool.btn_toggle_fx.text() == "f(X)"
    # The old offset/scale buttons are gone.
    assert not hasattr(tool, "btn_toggle_offset_x")
    assert not hasattr(tool, "btn_toggle_offset_y")
    assert not hasattr(tool, "btn_toggle_scale_y")


def test_toggling_a_button_shows_its_column(tool):
    assert tool.dataset_table.isColumnHidden(COL_FY)
    tool.btn_toggle_fy.setChecked(True)
    tool._toggle_column(COL_FY, tool.btn_toggle_fy)
    assert not tool.dataset_table.isColumnHidden(COL_FY)


# ---------------------------------------------------------------------------
# Transform semantics
# ---------------------------------------------------------------------------

def test_no_expression_is_identity(tool):
    x, y, x_err, y_err = tool._transform_entry(tool.datasets[0])
    np.testing.assert_allclose(y, np.arange(10, dtype=float))
    np.testing.assert_allclose(x, np.arange(10, dtype=float))
    assert x_err is None and y_err is None


def test_legacy_offset_scale_is_expressible(tool):
    """The old Offset Y / Scale Y pair is just a formula now."""
    entry = tool.datasets[0]
    entry.y_expr = "y * 2 + 1"
    entry.x_expr = "x + 0.5"
    x, y, _, _ = tool._transform_entry(entry)
    np.testing.assert_allclose(y, np.arange(10, dtype=float) * 2 + 1)
    np.testing.assert_allclose(x, np.arange(10, dtype=float) + 0.5)


def test_normalization_expression(tool):
    entry = tool.datasets[1]
    entry.y_expr = "y / max(y)"
    _x, y, _, _ = tool._transform_entry(entry)
    np.testing.assert_allclose(y, np.arange(10, dtype=float) * 2 / 18.0)


def test_scalar_result_broadcasts_to_a_flat_line(tool):
    entry = tool.datasets[0]
    entry.y_expr = "mean(y)"
    _x, y, _, _ = tool._transform_entry(entry)
    np.testing.assert_allclose(y, np.full(10, 4.5))


def test_fx_runs_before_fy_so_fy_sees_new_x(tool):
    entry = tool.datasets[0]
    entry.x_expr = "x * 2"
    entry.y_expr = "gradient(y, x)"
    x, y, x_err, y_err = tool._transform_entry(entry)
    assert x_err is None and y_err is None
    np.testing.assert_allclose(x, np.arange(10, dtype=float) * 2)
    # dy/dx with x doubled is half of the untransformed slope of 1.
    np.testing.assert_allclose(y, np.full(10, 0.5))


def test_energy_is_available_as_E(tool):
    entry = tool.datasets[0]
    entry.energy = 780.0
    entry.y_expr = "y * E"
    _x, y, _, _ = tool._transform_entry(entry)
    np.testing.assert_allclose(y, np.arange(10, dtype=float) * 780.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_bad_expression_falls_back_to_identity_without_raising(tool):
    entry = tool.datasets[0]
    entry.y_expr = "y +"  # syntax error
    _x, y, _x_err, y_err = tool._transform_entry(entry)
    assert y_err is not None
    np.testing.assert_allclose(y, np.arange(10, dtype=float))


def test_wrong_length_result_is_an_error(tool):
    entry = tool.datasets[0]
    entry.y_expr = "y[:3]"
    _x, y, _x_err, y_err = tool._transform_entry(entry)
    assert y_err is not None and "expected 10" in y_err
    np.testing.assert_allclose(y, np.arange(10, dtype=float))


def test_unsafe_expression_is_refused(tool):
    entry = tool.datasets[0]
    entry.y_expr = "y.__class__"
    _x, y, _x_err, y_err = tool._transform_entry(entry)
    assert y_err is not None
    np.testing.assert_allclose(y, np.arange(10, dtype=float))


def test_bad_expression_marks_the_cell_and_plot_still_updates(tool):
    tool.datasets[0].y_expr = "nonsense("
    tool._update_plot()
    item = tool.dataset_table.item(0, COL_FY)
    assert "Syntax error" in item.toolTip()
    # The other curve is unaffected.
    _x, y, _, y_err = tool._transform_entry(tool.datasets[1])
    assert y_err is None
    np.testing.assert_allclose(y, np.arange(10, dtype=float) * 2)


def test_fixing_an_expression_clears_the_error_mark(tool):
    tool.datasets[0].y_expr = "nonsense("
    tool._update_plot()
    tool.datasets[0].y_expr = "y * 3"
    tool._update_plot()
    item = tool.dataset_table.item(0, COL_FY)
    assert "Syntax error" not in item.toolTip()


# ---------------------------------------------------------------------------
# Editing through the table
# ---------------------------------------------------------------------------

def test_editing_the_cell_updates_the_entry(tool):
    tool.dataset_table.item(0, COL_FY).setText("y * 5")
    _x, y, _, _ = tool._transform_entry(tool.datasets[0])
    assert tool.datasets[0].y_expr == "y * 5"
    np.testing.assert_allclose(y, np.arange(10, dtype=float) * 5)


def test_apply_to_all_rows(tool):
    tool._apply_expression_to_all(COL_FY, "y / max(y)")
    assert [e.y_expr for e in tool.datasets] == ["y / max(y)", "y / max(y)"]
    for entry in tool.datasets:
        _x, y, _, err = tool._transform_entry(entry)
        assert err is None
        assert y[-1] == pytest.approx(1.0)


def test_clear_expression_for_row(tool):
    tool.datasets[0].y_expr = "y * 5"
    tool._clear_expression_for_row(0, COL_FY)
    assert tool.datasets[0].y_expr == ""
    assert tool.dataset_table.item(0, COL_FY).text() == ""


def test_invalid_energy_reverts(tool, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    tool.datasets[0].energy = 700.0
    tool.dataset_table.item(0, COL_ENERGY).setText("not a number")
    assert tool.datasets[0].energy == 700.0
    assert tool.dataset_table.item(0, COL_ENERGY).text() == "700.0"


# ---------------------------------------------------------------------------
# Legend / export wiring
# ---------------------------------------------------------------------------

def test_legend_shows_expressions(tool):
    entry = tool.datasets[0]
    entry.y_expr = "y / max(y)"
    entry.x_expr = "x + 1"
    label = tool._format_name_with_transforms(entry)
    assert "X=x + 1" in label and "Y=y / max(y)" in label


def test_legend_is_plain_without_expressions(tool):
    label = tool._format_name_with_transforms(tool.datasets[0])
    assert "X=" not in label and "Y=" not in label


def test_export_series_matches_the_transform(tool):
    entry = tool.datasets[0]
    entry.y_expr = "y * 2 + 1"
    x_export, y_export = tool._build_export_series(entry)
    x_plot, y_plot, _, _ = tool._transform_entry(entry)
    np.testing.assert_allclose(x_export, x_plot)
    np.testing.assert_allclose(y_export, y_plot)


def test_shared_x_export_rejects_differing_fx(tool):
    tool.datasets[0].x_expr = "x + 1"
    compatible, reason = tool._is_shared_xq_compatible()
    assert not compatible and "f(X) differs" in reason


def test_shared_x_export_accepts_matching_fx(tool):
    tool.datasets[0].x_expr = "x + 1"
    tool.datasets[1].x_expr = "x + 1"
    compatible, _reason = tool._is_shared_xq_compatible()
    assert compatible

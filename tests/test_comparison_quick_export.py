"""Tests for the Data Comparison quick export (what is plotted, in one file)."""

import pathlib
from typing import Any

import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from src.gui.data_comparison import DataComparisonTool
from src.lib_h5.table_format import TABLE_FORMATS, format_from_filter, get_table_format


@pytest.fixture
def save_to(tmp_path, monkeypatch):
    seen: dict[str, Any] = {"names": [], "boxes": [], "pick": None}

    def fake_dialog(_parent, _title, default_path, selected_filter, *a, **k):
        seen["names"].append(pathlib.Path(default_path).name)
        chosen = seen["pick"] or selected_filter.split(";;")[0]
        name = pathlib.Path(default_path).name
        if any(chosen.startswith(fmt.label) for fmt in TABLE_FORMATS):
            name = pathlib.Path(default_path).stem + format_from_filter(chosen).suffix
        seen["path"] = tmp_path / name
        return str(seen["path"]), chosen

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, kind, staticmethod(lambda *a, _k=kind, **k: seen["boxes"].append(_k))
        )
    return seen


def _rows(path, fmt):
    text = path.read_text(encoding=fmt.encoding)
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return [ln.split(fmt.delimiter) for ln in body]


@pytest.fixture
def tool(qapp):
    widget = DataComparisonTool(tuple())
    widget.add_dataset_from_array("a", np.array([1.0, 2.0, 3.0]))
    widget.add_dataset_from_array("b", np.array([10.0, 20.0, 30.0]))
    return widget


# ---------------------------------------------------------------------------
# Column layout: one shared X, or one X per curve
# ---------------------------------------------------------------------------

def test_curves_sharing_an_axis_get_one_x_column(tool, save_to):
    x = np.array([0.5, 1.5, 2.5])
    tool._on_x_data_selected(x, "scan::setpoint")

    tool.btn_save_data.click()

    fmt = get_table_format("txt")
    rows = _rows(save_to["path"], fmt)
    # The X column is named after the X dataset, not after a Y curve.
    assert rows[0] == ["scan_setpoint_X", "a_Y", "b_Y"]
    assert rows[1] == ["0.5", "1", "10"]


def test_curves_with_different_x_get_a_pair_each(tool, save_to):
    tool._x_selection_target_row = 0
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::x1")
    tool._x_selection_target_row = 1
    tool._on_x_data_selected(np.array([9.0, 8.0, 7.0]), "scan::x2")

    tool.btn_save_data.click()

    rows = _rows(save_to["path"], get_table_format("txt"))
    # Each X column carries the name of its own X dataset.
    assert rows[0] == ["scan_x1_X", "a_Y", "scan_x2_X", "b_Y"]
    assert rows[1] == ["0.5", "1", "9", "10"]


def test_index_x_still_exports_without_an_x_column(tool, save_to):
    """No X dataset anywhere: the aligned table is Y columns only."""
    tool.btn_save_data.click()

    rows = _rows(save_to["path"], get_table_format("txt"))
    assert rows[0] == ["a_Y", "b_Y"]


# ---------------------------------------------------------------------------
# What "as displayed" means
# ---------------------------------------------------------------------------

def test_curve_transforms_are_applied(tool, save_to):
    tool.datasets[0].y_expr = "y * 100"
    tool._update_plot()

    tool.btn_save_data.click()

    rows = _rows(save_to["path"], get_table_format("txt"))
    assert rows[1][0] == "100"


def test_expressions_are_recorded_in_the_comments(tool, save_to):
    tool.datasets[0].y_expr = "y / max(y)"
    tool._update_plot()

    tool.btn_save_data.click()

    text = save_to["path"].read_text(encoding="utf-8")
    assert "fY=y / max(y)" in text


def test_a_delimiter_inside_a_comment_does_not_break_the_file(tool, save_to):
    """csv2 uses ';'; a comment containing one must stay a plain comment line."""
    save_to["pick"] = get_table_format("csv2").label
    tool.datasets[0].y_expr = "where(y > 1; y; 0)".replace(";", ",")
    tool._update_plot()

    tool.btn_save_data.click()

    fmt = get_table_format("csv2")
    lines = save_to["path"].read_text(encoding=fmt.encoding).splitlines()
    comment = next(ln for ln in lines if ln.startswith("# a:"))
    assert not comment.startswith('"')


def test_chosen_dialect_reaches_the_comparison_export(tool, save_to):
    save_to["pick"] = get_table_format("csv2").label
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::setpoint")

    tool.btn_save_data.click()

    fmt = get_table_format("csv2")
    assert save_to["path"].suffix == ".csv"
    rows = _rows(save_to["path"], fmt)
    assert rows[1] == ["0,5", "1", "10"]

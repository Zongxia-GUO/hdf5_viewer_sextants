"""Tests for the calculator's full export: operands + result, with an optional X."""

import pathlib

import h5py
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from src.gui.data_calculator_enhanced import DataCalculatorEnhanced, ResultExportDialog
from src.lib_h5.table_format import get_table_format


@pytest.fixture
def scan(tmp_path):
    """A file with two data columns and an actuator axis."""
    path = tmp_path / "scanx_0085.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data_a", data=np.array([1.0, 2.0, 3.0]))
        f.create_dataset("data_b", data=np.array([10.0, 20.0, 30.0]))
        f.create_dataset("actuator", data=np.array([0.5, 1.5, 2.5]))
        f.create_dataset("table", data=np.arange(12.0).reshape(3, 4))
    return path


@pytest.fixture
def calc(qapp, scan):
    tool = DataCalculatorEnhanced((scan,))
    tool.add_to_dataset_a(f"{scan}::data_a")
    tool.add_to_dataset_b(f"{scan}::data_b")
    tool._perform_operation("A + B")
    return tool


# ---------------------------------------------------------------------------
# Which columns get written
# ---------------------------------------------------------------------------

def test_columns_are_operands_and_result(calc):
    headers, columns = calc._result_export_columns(None)

    assert headers == ["scanx_0085_data_a", "scanx_0085_data_b", "A + B"]
    np.testing.assert_allclose(columns[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(columns[1], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(columns[2], [11.0, 22.0, 33.0])


# ---------------------------------------------------------------------------
# The second output button: Plot for a curve, Copy for an image
# ---------------------------------------------------------------------------

def test_a_curve_result_offers_plot(calc):
    assert calc.btn_plot.text() == "Plot"
    assert calc.result_is_image() is False


def test_an_image_result_offers_copy_instead(qapp, scan):
    """An image cannot be plotted as a curve — flattening it into one very long
    line is not a picture of anything — but it can be put on the clipboard."""
    tool = DataCalculatorEnhanced((scan,))
    tool.result_data = np.arange(256.0).reshape(16, 16)
    tool._update_result_display()

    assert tool.result_is_image() is True
    assert tool.btn_plot.text() == "Copy"
    assert "clipboard" in tool.btn_plot.toolTip()


def test_the_button_goes_back_to_plot_for_the_next_curve(qapp, scan):
    """The two live on one button, so the label must follow every result."""
    tool = DataCalculatorEnhanced((scan,))
    tool.result_data = np.arange(256.0).reshape(16, 16)
    tool._update_result_display()
    assert tool.btn_plot.text() == "Copy"

    tool.result_data = np.arange(5.0)
    tool._update_result_display()

    assert tool.btn_plot.text() == "Plot"


def test_the_button_copies_the_image_rather_than_plotting_it(qapp, scan, monkeypatch):
    """Dispatch happens on click, so the label and the wiring cannot disagree."""
    tool = DataCalculatorEnhanced((scan,))
    tool.result_data = np.arange(256.0).reshape(16, 16)
    tool._update_result_display()

    done: list[str] = []
    monkeypatch.setattr(tool, "open_plot", lambda: done.append("plot"))
    monkeypatch.setattr(
        type(tool._result_image_widget()), "quick_copy", lambda self: done.append("copy")
    )

    tool.btn_plot.setEnabled(True)
    tool.btn_plot.click()

    assert done == ["copy"]


def test_the_button_still_plots_a_curve(qapp, scan, monkeypatch):
    tool = DataCalculatorEnhanced((scan,))
    tool.result_data = np.arange(5.0)
    tool._update_result_display()

    done: list[str] = []
    monkeypatch.setattr(tool, "open_plot", lambda: done.append("plot"))

    tool.btn_plot.setEnabled(True)
    tool.btn_plot.click()

    assert done == ["plot"]


def test_a_reset_calculator_offers_plot_again(qapp, scan):
    """Closing clears the result, so the button cannot keep advertising Copy."""
    tool = DataCalculatorEnhanced((scan,))
    tool.result_data = np.arange(256.0).reshape(16, 16)
    tool._update_result_display()

    tool.reset_results()

    assert tool.btn_plot.text() == "Plot"


def test_a_chosen_column_is_named_in_the_operand(calc, scan, qapp):
    """Two columns of one dataset are two different curves, and the legend has
    to say which is which."""
    tool = DataCalculatorEnhanced((scan,))
    tool.add_to_dataset_a(f"{scan}::table")
    tool.spin_col_a.addItem("Column 2", 2)
    tool.spin_col_a.setCurrentIndex(tool.spin_col_a.count() - 1)
    tool.data_a = np.arange(3.0)
    tool.result_data = np.arange(3.0)
    tool._last_operation_expr = "A * 2"

    headers, _columns = tool._result_export_columns(None)

    assert headers == ["scanx_0085_table_col2", "A * 2"]


def test_the_result_is_named_after_the_operation(calc):
    """"Result" says nothing once the figure has left the calculator; the
    expression says what the curve actually is."""
    assert calc._result_label() == "A + B"

    calc._last_operation_expr = "(A - B) / (A + B)"
    assert calc._result_label() == "(A - B) / (A + B)"


def test_an_unnamed_operation_still_gets_a_label(qapp, scan):
    """A result set without going through an operation — nothing to name it
    after, so the old word has to stay available."""
    tool = DataCalculatorEnhanced((scan,))

    assert tool._result_label() == "Result"


def test_the_result_stays_on_the_right_axis_although_it_is_renamed(calc):
    """The split used to match the literal word "Result"; it now goes by
    position, and getting that wrong would put the result on the wrong axis."""
    operands, result = calc.plot_axes_series()

    assert [s.label for s in operands] == ["scanx_0085_data_a", "scanx_0085_data_b"]
    assert [s.label for s in result] == ["A + B"]


def test_an_x_dataset_is_prepended(calc, scan):
    headers, columns = calc._result_export_columns(f"{scan}::actuator")

    assert headers == ["scanx_0085_actuator_X", "scanx_0085_data_a", "scanx_0085_data_b", "A + B"]
    np.testing.assert_allclose(columns[0], [0.5, 1.5, 2.5])


def test_an_unreadable_x_is_skipped_rather_than_failing(calc, scan):
    headers, _columns = calc._result_export_columns(f"{scan}::no_such_dataset")
    assert headers == ["scanx_0085_data_a", "scanx_0085_data_b", "A + B"]


def test_single_operand_result_omits_data_b(qapp, scan):
    tool = DataCalculatorEnhanced((scan,))
    tool.add_to_dataset_a(f"{scan}::data_a")
    tool._perform_operation("A * 2")

    headers, _columns = tool._result_export_columns(None)
    assert headers == ["scanx_0085_data_a", "A * 2"]


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def test_dialog_exposes_x_and_format(qapp, scan):
    dialog = ResultExportDialog(
        None,
        opened_files=(scan,),
        dataset_full_keys_1d=[f"{scan}::actuator"],
        preferred_x_key=f"{scan}::actuator",
        expression="A + B",
    )

    assert dialog.chk_export_x.isChecked()
    assert dialog.x_key() == f"{scan}::actuator"

    dialog.chk_export_x.setChecked(False)
    assert dialog.x_key() is None

    dialog.combo_format.setCurrentText(get_table_format("csv2").label)
    assert dialog.table_format().key == "csv2"


def test_x_combo_accepts_drops(qapp, scan):
    """The X field must take a dataset dragged out of the tree."""
    dialog = ResultExportDialog(
        None,
        opened_files=(scan,),
        dataset_full_keys_1d=[],
        preferred_x_key=None,
        expression="A + B",
    )
    assert dialog.combo_x.acceptDrops()


def test_dialog_without_a_preferred_x_starts_with_x_off(qapp, scan):
    dialog = ResultExportDialog(
        None,
        opened_files=(scan,),
        dataset_full_keys_1d=[f"{scan}::actuator"],
        preferred_x_key=None,
        expression="A + B",
    )
    assert not dialog.chk_export_x.isChecked()
    assert dialog.x_key() is None


def test_preview_shows_the_columns_that_will_be_written(qapp, scan, calc):
    dialog = ResultExportDialog(
        None,
        opened_files=(scan,),
        dataset_full_keys_1d=[f"{scan}::actuator"],
        preferred_x_key=f"{scan}::actuator",
        expression="A + B",
    )
    headers, columns = calc._result_export_columns(dialog.x_key())
    dialog.show_preview(headers, columns)

    model = dialog.preview_table.model()
    assert model.columnCount() == 4
    assert model.rowCount() == 3


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_the_settings_dialog_is_not_modal(calc):
    """A modal dialog blocks the tree, which is what broke dragging an X in."""
    calc._export_result()

    dialog = calc._export_dialog
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal


def test_export_writes_every_column(calc, scan, tmp_path, monkeypatch):
    out = tmp_path / "result.txt"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, **k: None))

    calc._export_result()
    dialog = calc._export_dialog
    # Pick the X the way a user would once the dialog is up.
    dialog.combo_x.add_full_key(f"{scan}::actuator", select=True)
    dialog.chk_export_x.setChecked(True)
    dialog.accept()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == [
        "scanx_0085_actuator_X", "scanx_0085_data_a", "scanx_0085_data_b", "A + B",
    ]
    assert lines[1].split("\t") == ["0.5", "1", "10", "11"]


def test_cancelling_the_dialog_writes_nothing(calc, tmp_path, monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: called.append("x") or ("", ""))
    )

    calc._export_result()
    calc._export_dialog.reject()

    assert called == []
    assert list(pathlib.Path(tmp_path).glob("*.txt")) == []

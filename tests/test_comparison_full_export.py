"""Tests for the comparison tool's full export (settings dialog) vs quick export."""

import pathlib
from typing import Any

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import h5py

from src.gui.data_comparison import ComparisonExportDialog, DataComparisonTool
from src.lib_h5.table_format import get_table_format


@pytest.fixture
def x_file(tmp_path):
    path = tmp_path / "scanx_0001.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("motor", data=np.array([7.0, 8.0, 9.0]))
    return path


@pytest.fixture
def tool(qapp, x_file):
    widget = DataComparisonTool((x_file,))
    widget.add_dataset_from_array("a", np.array([1.0, 2.0, 3.0]))
    widget.add_dataset_from_array("b", np.array([10.0, 20.0, 30.0]))
    return widget


@pytest.fixture
def saved(tmp_path, monkeypatch):
    seen: dict[str, Any] = {"path": None, "boxes": []}

    def fake_dialog(_parent, _title, default_path, _filter, *a, **k):
        seen["path"] = tmp_path / pathlib.Path(default_path).name
        return str(seen["path"]), _filter.split(";;")[0]

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


# ---------------------------------------------------------------------------
# Layout choice
# ---------------------------------------------------------------------------

def test_displayed_x_is_used_when_no_dataset_is_given(tool):
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::setpoint")

    headers, columns, _comments = tool.build_export_table(export_x=True, x_key=None)
    assert headers == ["scan_setpoint_X", "a_Y", "b_Y"]
    np.testing.assert_allclose(columns[0], [0.5, 1.5, 2.5])


def test_an_explicit_x_dataset_replaces_the_displayed_one(tool, x_file):
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::setpoint")

    headers, columns, comments = tool.build_export_table(
        export_x=True, x_key=f"{x_file}::motor"
    )
    assert headers == ["scanx_0001_motor_X", "a_Y", "b_Y"]
    np.testing.assert_allclose(columns[0], [7.0, 8.0, 9.0])
    assert any("X dataset:" in c and "applied to all curves" in c for c in comments)


def test_an_explicit_x_forces_a_single_shared_column(tool, x_file):
    """Curves with different X normally split into pairs; an explicit X unifies them."""
    tool._x_selection_target_row = 0
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::x1")
    tool._x_selection_target_row = 1
    tool._on_x_data_selected(np.array([9.0, 8.0, 7.0]), "scan::x2")

    split, _c, _m = tool.build_export_table(export_x=True, x_key=None)
    assert split == ["scan_x1_X", "a_Y", "scan_x2_X", "b_Y"]

    unified, _c2, _m2 = tool.build_export_table(export_x=True, x_key=f"{x_file}::motor")
    assert unified == ["scanx_0001_motor_X", "a_Y", "b_Y"]


def test_export_x_off_writes_only_y_columns(tool):
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::setpoint")

    headers, _columns, _comments = tool.build_export_table(export_x=False)
    assert headers == ["a_Y", "b_Y"]


def test_export_x_off_also_drops_per_curve_x(tool):
    tool._x_selection_target_row = 0
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::x1")
    tool._x_selection_target_row = 1
    tool._on_x_data_selected(np.array([9.0, 8.0, 7.0]), "scan::x2")

    headers, _columns, _comments = tool.build_export_table(export_x=False)
    assert headers == ["a_Y", "b_Y"]


def test_an_unreadable_x_dataset_falls_back_to_the_displayed_x(tool, x_file):
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5]), "scan::setpoint")

    headers, columns, comments = tool.build_export_table(
        export_x=True, x_key=f"{x_file}::no_such_dataset"
    )
    # The requested X could not be read, so the column is the displayed one and
    # must be named after that, not after the dataset that failed.
    assert headers == ["scan_setpoint_X", "a_Y", "b_Y"]
    np.testing.assert_allclose(columns[0], [0.5, 1.5, 2.5])
    assert any("not readable" in c for c in comments)


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def test_dialog_defaults(qapp):
    dialog = ComparisonExportDialog()

    assert dialog.export_x() is True
    assert dialog.x_key() is None  # blank field = use the displayed X
    assert dialog.include_comments() is True
    assert dialog.table_format().key == "txt"


def test_dialog_reports_its_choices(qapp):
    dialog = ComparisonExportDialog()
    dialog.le_x_path.setText("f.h5::motor")
    dialog.combo_format.setCurrentText(get_table_format("csv2").label)
    dialog.chk_comments.setChecked(False)

    assert dialog.x_key() == "f.h5::motor"
    assert dialog.table_format().key == "csv2"
    assert dialog.include_comments() is False


def test_unchecking_export_x_hides_the_typed_dataset(qapp):
    dialog = ComparisonExportDialog()
    dialog.le_x_path.setText("f.h5::motor")
    dialog.chk_export_x.setChecked(False)

    assert dialog.export_x() is False
    assert dialog.x_key() is None


def test_dropping_a_dataset_fills_the_x_field(qapp):
    """The whole point of the non-modal dialog: drag from the tree."""

    class _Mime:
        def __init__(self, text):
            self._t = text

        def text(self):
            return self._t

        def hasText(self):
            return True

    class _Event:
        def __init__(self, text):
            self._m = _Mime(text)
            self.accepted = False

        def mimeData(self):
            return self._m

        def acceptProposedAction(self):
            self.accepted = True

    dialog = ComparisonExportDialog()
    dialog.chk_export_x.setChecked(False)
    event = _Event("d:/data/scanx_0001.h5::motor\nd:/data/scanx_0001.h5::other")

    dialog._x_path_drop(event)

    assert event.accepted
    # Only the first dragged dataset lands, and dropping re-enables X.
    assert dialog.le_x_path.text() == "d:/data/scanx_0001.h5::motor"
    assert dialog.export_x() is True


def test_dialog_preview_matches_the_built_table(qapp, tool):
    dialog = ComparisonExportDialog()
    headers, columns, _comments = tool.build_export_table(dialog.export_x(), dialog.x_key())
    dialog.show_preview(headers, columns)

    model = dialog.preview_table.model()
    assert model.columnCount() == len(headers)
    assert model.rowCount() == 3


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_the_settings_dialog_is_not_modal(tool):
    """A modal dialog blocks the tree, which is what broke dragging an X in."""
    tool.btn_export.click()

    dialog = tool._export_dialog
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal


def test_full_export_applies_the_dialog_settings(tool, saved, x_file):
    tool.btn_export.click()
    dialog = tool._export_dialog
    dialog.le_x_path.setText(f"{x_file}::motor")
    dialog.combo_format.setCurrentText(get_table_format("csv2").label)
    dialog.chk_comments.setChecked(False)

    dialog.accept()

    fmt = get_table_format("csv2")
    text = saved["path"].read_text(encoding=fmt.encoding)
    assert not text.startswith("#")  # metadata header switched off
    rows = _rows(saved["path"], fmt)
    assert rows[0] == ["scanx_0001_motor_X", "a_Y", "b_Y"]
    assert rows[1] == ["7", "1", "10"]


def test_cancelling_the_full_dialog_writes_nothing(tool, saved):
    tool.btn_export.click()
    tool._export_dialog.reject()

    assert saved["path"] is None


def test_quick_export_still_writes_without_a_dialog(tool, saved, monkeypatch):
    """The toolbar icon must not open the settings dialog."""
    monkeypatch.setattr(
        ComparisonExportDialog,
        "show",
        lambda self: pytest.fail("quick export must not open the settings dialog"),
    )

    tool.btn_save_data.click()

    assert saved["path"] is not None and saved["path"].exists()
    rows = _rows(saved["path"], get_table_format("txt"))
    assert rows[0] == ["a_Y", "b_Y"]


def test_export_with_no_datasets_reports_instead_of_writing(qapp, saved):
    empty = DataComparisonTool(tuple())
    empty._export_full()

    assert saved["boxes"] == ["information"]
    assert saved["path"] is None

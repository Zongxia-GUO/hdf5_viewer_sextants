"""The menu-bar Export runs the full settings dialog for the selected dataset.

It used to be a bare save-path dialog, so a single dataset could not reach the
X-axis, dialect and colormap options the batch path already had.
"""

import pathlib

import h5py
import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox, QWidget

from src.gui.batch_export import BatchExportDialog, BatchTarget
from src.gui.main_window import MainWindow


class _Host(QWidget):
    """MainWindow's export surface, without constructing the real window.

    Only the window-bound methods need borrowing now — the export logic itself
    lives in :mod:`src.gui.batch_export` and is called as plain functions.

    A real QWidget, because the dialog is now parented to whoever opens it —
    that ownership is what keeps it above the main window.
    """

    _handle_action_export_current = MainWindow._handle_action_export_current
    _read_selected_dataset = MainWindow._read_selected_dataset
    _open_export_dialog = MainWindow._open_export_dialog
    _displayed_slice_position = MainWindow._displayed_slice_position
    _current_image_view_2d = staticmethod(lambda: None)
    _x_dataset_path = MainWindow._x_dataset_path
    _scan_token_from_filename = staticmethod(MainWindow._scan_token_from_filename)
    _finish_batch_export = MainWindow._finish_batch_export

    def __init__(self, cur_file=None, cur_obj_path=""):
        super().__init__()
        self.cur_file = cur_file
        self.cur_obj_path = cur_obj_path
        self._batch_export_dialog = None
        self._x_dataset_key = None


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scanx_0033.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan_0033/scan_data")
        grp.create_dataset("data_01", data=np.array([1.0, 2.0, 3.0]))
        grp.create_dataset("actuator_1_1", data=np.array([10.0, 20.0, 30.0]))
    return path


@pytest.fixture
def boxes(monkeypatch):
    seen: list[str] = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, kind, staticmethod(lambda *a, _k=kind, **k: seen.append(_k))
        )
    return seen


# ---------------------------------------------------------------------------
# Scan token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("scanx_0033.nxs", "0033"),
        ("scanx_0033.h5", "0033"),
        ("run12_part007.h5", "007"),   # last run of digits wins
        ("no_digits.h5", ""),
    ],
)
def test_scan_token_from_filename(name, expected):
    assert MainWindow._scan_token_from_filename(name) == expected


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_no_selection_warns_and_opens_nothing(qapp, boxes):
    host = _Host(cur_file=None, cur_obj_path="")
    host._handle_action_export_current()

    assert boxes == ["warning"]
    assert host._batch_export_dialog is None


def test_selecting_a_group_warns(qapp, scan, boxes):
    host = _Host(cur_file=scan, cur_obj_path="/scan_0033/scan_data")
    host._handle_action_export_current()

    assert boxes == ["warning"]
    assert host._batch_export_dialog is None


def test_unreadable_dataset_reports_instead_of_raising(qapp, scan, boxes):
    host = _Host(cur_file=scan, cur_obj_path="/no/such/dataset")
    host._handle_action_export_current()

    assert boxes == ["critical"]
    assert host._batch_export_dialog is None


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

def test_menu_export_opens_the_full_settings_dialog(qapp, scan, boxes):
    host = _Host(cur_file=scan, cur_obj_path="/scan_0033/scan_data/data_01")
    host._handle_action_export_current()

    dialog = host._batch_export_dialog
    assert isinstance(dialog, BatchExportDialog)
    assert dialog.isVisible() and not dialog.isModal()
    assert dialog._data_kind == "curve"
    # The options a bare save dialog could not offer:
    assert hasattr(dialog, "cb_table_format")
    assert hasattr(dialog, "le_x_path")


def test_a_single_dataset_becomes_a_one_element_target(qapp, scan, monkeypatch):
    captured = {}

    def fake_open(self, targets, matching_files, scan_numbers, shown_path, sample_data=None):
        captured.update(
            targets=targets, matching=matching_files, scans=scan_numbers,
            shown=shown_path, sample=sample_data,
        )

    monkeypatch.setattr(_Host, "_open_export_dialog", fake_open)
    host = _Host(cur_file=scan, cur_obj_path="/scan_0033/scan_data/data_01")
    host._handle_action_export_current()

    assert len(captured["targets"]) == 1
    target = captured["targets"][0]
    assert isinstance(target, BatchTarget)
    assert target.file_path == scan
    assert target.scan_num == "0033"
    assert target.ds_path == "scan_0033/scan_data/data_01"
    assert captured["shown"] == "scan_0033/scan_data/data_01"
    np.testing.assert_allclose(captured["sample"], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_accepting_writes_the_dataset_with_the_chosen_settings(qapp, scan, tmp_path, boxes):
    out = tmp_path / "out"
    out.mkdir()

    host = _Host(cur_file=scan, cur_obj_path="/scan_0033/scan_data/data_01")
    host._handle_action_export_current()

    dialog = host._batch_export_dialog
    dialog.le_output_dir.setText(str(out))
    dialog.chk_export_x.setChecked(True)
    dialog.le_x_path.setText("scan_0033/scan_data/actuator_1_1")
    dialog.cb_table_format.setCurrentIndex(2)   # CSV2
    assert dialog.table_format().key == "csv2"

    # Not accept(): the dialog deliberately stays open when the export runs.
    dialog.export_requested.emit()

    written = list(out.glob("*.csv"))
    assert len(written) == 1
    lines = written[0].read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "scanx_0033_actuator_1_1_X;scanx_0033_data_01"
    assert lines[1] == "10;1"


def test_a_non_hdf5_file_is_exported_from_the_passed_sample(qapp, tmp_path, boxes, monkeypatch):
    """Regular data files cannot be re-read with h5py, so the array is passed in."""
    csv_path = tmp_path / "curve_0007.csv"
    csv_path.write_text("1\n2\n3\n", encoding="utf-8")

    captured = {}
    monkeypatch.setattr(
        _Host,
        "_open_export_dialog",
        lambda self, t, m, s, p, sample_data=None: captured.update(sample=sample_data, scans=s),
    )

    host = _Host(cur_file=csv_path, cur_obj_path="/curve")
    host._handle_action_export_current()

    assert captured["scans"] == ["0007"]
    np.testing.assert_allclose(np.ravel(captured["sample"]), [1.0, 2.0, 3.0])


def test_batch_and_menu_share_one_dialog_path():
    """Both entry points must funnel through _open_export_dialog."""
    import inspect

    batch_src = inspect.getsource(MainWindow._handle_batch_export)
    menu_src = inspect.getsource(MainWindow._handle_action_export_current)
    assert "_open_export_dialog" in batch_src
    assert "_open_export_dialog" in menu_src
    # The old bare save-path dialog is gone from the menu path.
    assert "getSaveFileName" not in menu_src


def test_export_dialog_survives_a_path_without_leading_slash(qapp, scan, boxes):
    host = _Host(cur_file=scan, cur_obj_path="scan_0033/scan_data/data_01")
    host._handle_action_export_current()

    assert isinstance(host._batch_export_dialog, BatchExportDialog)
    assert pathlib.Path(scan).exists()


# ---------------------------------------------------------------------------
# The menu's Plot action — Export's twin on the same selection
# ---------------------------------------------------------------------------

class _PlotHost(_Host):
    _handle_action_plot_current = MainWindow._handle_action_plot_current

    def _current_plot_widget_1d(self):
        return self.viewer

    def __init__(self, cur_file=None, cur_obj_path="", viewer=None):
        super().__init__(cur_file, cur_obj_path)
        self.viewer = viewer


@pytest.fixture
def opened(monkeypatch):
    """Capture what would have been plotted instead of opening a window."""
    calls: list[dict] = []

    def fake_open(parent, series, **kwargs):
        calls.append({"series": list(series), **kwargs})
        return None

    monkeypatch.setattr("src.gui.plot_dialog.open_plot_dialog", fake_open)
    return calls


def test_menu_plot_sends_the_selected_dataset_to_the_plot_window(qapp, scan, opened, boxes):
    host = _PlotHost(cur_file=scan, cur_obj_path="/scan_0033/scan_data/data_01")
    host._handle_action_plot_current()

    assert len(opened) == 1
    assert [s.label for s in opened[0]["series"]] == ["data_01"]
    np.testing.assert_allclose(opened[0]["series"][0].y, [1.0, 2.0, 3.0])


def test_menu_plot_picks_up_the_x_the_viewer_is_showing(qapp, scan, opened, boxes):
    class _Viewer:
        x_data = np.array([10.0, 20.0, 30.0])

    host = _PlotHost(scan, "/scan_0033/scan_data/data_01", viewer=_Viewer())
    host._handle_action_plot_current()

    np.testing.assert_allclose(opened[0]["series"][0].x, [10.0, 20.0, 30.0])


def test_a_mismatched_viewer_x_is_ignored_rather_than_forced(qapp, scan, opened, boxes):
    class _Viewer:
        x_data = np.arange(99.0)

    host = _PlotHost(scan, "/scan_0033/scan_data/data_01", viewer=_Viewer())
    host._handle_action_plot_current()

    assert opened[0]["series"][0].x is None


def test_an_image_is_refused_instead_of_drawn_as_thousands_of_curves(qapp, tmp_path, opened, boxes):
    path = tmp_path / "img_0001.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("frame", data=np.zeros((64, 64)))

    host = _PlotHost(path, "/frame")
    host._handle_action_plot_current()

    assert opened == []
    assert boxes == ["information"]


def test_menu_plot_refuses_an_empty_selection(qapp, opened, boxes):
    _PlotHost(cur_file=None, cur_obj_path="")._handle_action_plot_current()

    assert opened == []
    assert boxes == ["warning"]

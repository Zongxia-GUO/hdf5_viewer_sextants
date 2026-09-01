"""Tests for the tree right-click quick export.

Quick export means "what is on screen", so a tree request displays the dataset
first and the export is queued until the viewer actually holds the data — the
load runs on a worker thread.
"""

import pathlib
from typing import Any

import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog, QLabel, QMessageBox

from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
from src.gui.main_window import MainWindow
from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
from src.gui.unified_data_viewer import UnifiedDataViewer


class _Dock:
    def __init__(self, widget):
        self._w = widget

    def widget(self):
        return self._w


class _Host:
    """Stand-in for MainWindow's queued-tree-action surface."""

    _run_pending_tree_action = MainWindow._run_pending_tree_action

    def __init__(self, widget, pending=""):
        self.dock_plot = _Dock(widget)
        self._pending_tree_action = pending


@pytest.fixture
def calls(monkeypatch):
    """Record the quick actions without touching the filesystem."""
    seen: dict[str, Any] = {"export": 0, "copy": 0, "plot": 0, "boxes": []}
    for cls in (PlotWidget1DEnhanced, ImageView2DEnhanced):
        monkeypatch.setattr(cls, "quick_export", lambda self: seen.__setitem__("export", seen["export"] + 1))
        monkeypatch.setattr(cls, "quick_copy", lambda self: seen.__setitem__("copy", seen["copy"] + 1))
    monkeypatch.setattr(
        PlotWidget1DEnhanced, "open_plot", lambda self: seen.__setitem__("plot", seen["plot"] + 1)
    )
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, kind, staticmethod(lambda *a, _k=kind, **k: seen["boxes"].append(_k))
        )
    return seen


@pytest.fixture
def curve_viewer(qapp):
    viewer = UnifiedDataViewer()
    viewer.set_data(np.arange(10, dtype=float), source_dataset_key="f.h5::curve")
    return viewer


@pytest.fixture
def image_viewer(qapp):
    viewer = UnifiedDataViewer()
    viewer.set_data(np.random.RandomState(0).rand(40, 40), source_dataset_key="f.h5::img")
    return viewer


# ---------------------------------------------------------------------------
# The queued action
# ---------------------------------------------------------------------------

def test_a_queued_export_runs_the_viewer_quick_export(curve_viewer, calls):
    host = _Host(curve_viewer, pending="export")
    host._run_pending_tree_action()

    assert calls["export"] == 1 and calls["copy"] == 0 and calls["plot"] == 0


def test_a_queued_plot_opens_the_viewer_plot_window(curve_viewer, calls):
    host = _Host(curve_viewer, pending="plot")
    host._run_pending_tree_action()

    assert calls["plot"] == 1 and calls["export"] == 0


def test_image_viewer_is_reached_through_the_unified_viewer(image_viewer, calls):
    assert isinstance(image_viewer.get_current_widget(), ImageView2DEnhanced)

    _Host(image_viewer, pending="export")._run_pending_tree_action()
    assert calls["export"] == 1


def test_an_image_cannot_be_plotted(image_viewer, calls):
    """The plot window draws curves; an image has its own viewer."""
    _Host(image_viewer, pending="plot")._run_pending_tree_action()

    assert calls["plot"] == 0
    assert calls["boxes"] == ["information"]


def test_nothing_happens_without_a_queued_request(curve_viewer, calls):
    host = _Host(curve_viewer, pending="")
    host._run_pending_tree_action()

    assert calls == {"export": 0, "copy": 0, "plot": 0, "boxes": []}


def test_the_request_is_consumed_once(curve_viewer, calls):
    host = _Host(curve_viewer, pending="export")
    host._run_pending_tree_action()
    host._run_pending_tree_action()

    assert calls["export"] == 1
    assert host._pending_tree_action == ""


def test_a_bare_widget_is_supported_without_the_unified_wrapper(qapp, calls):
    plot = PlotWidget1DEnhanced()
    plot.set_data(np.arange(5, dtype=float))

    _Host(plot, pending="export")._run_pending_tree_action()
    assert calls["export"] == 1


# ---------------------------------------------------------------------------
# Cases with nothing to export
# ---------------------------------------------------------------------------

def test_a_non_plot_widget_reports_instead_of_crashing(qapp, calls):
    """Text and table datasets have no "as displayed" form to export."""
    host = _Host(QLabel("some text"), pending="export")
    host._run_pending_tree_action()

    assert calls["boxes"] == ["information"]
    assert calls["export"] == 0
    assert host._pending_tree_action == ""


def test_a_failed_load_clears_the_queued_request(qapp, calls, monkeypatch):
    """_on_load_error must not leave the request armed for the next dataset."""

    class _ErrHost:
        _on_load_error = MainWindow._on_load_error

        def __init__(self):
            self._pending_tree_action = "export"
            self._loading_timer = type("T", (), {"stop": lambda self: None})()
            self.dock_plot = type("D", (), {"setWidget": lambda self, w: None})()

    host = _ErrHost()
    host._on_load_error("boom")
    assert host._pending_tree_action == ""


# ---------------------------------------------------------------------------
# The public quick API the tree relies on
# ---------------------------------------------------------------------------

def test_both_viewers_expose_the_same_quick_api():
    for cls in (PlotWidget1DEnhanced, ImageView2DEnhanced):
        assert callable(getattr(cls, "quick_export"))
        assert callable(getattr(cls, "quick_copy"))


def test_the_tree_menu_is_export_plot_set_x_in_that_order(qapp):
    """Three text entries, no icons, in the order the user reads them."""
    import inspect
    import re

    src = inspect.getsource(MainWindow._handle_tree_menu)
    labels = re.findall(r'QAction\("(Export|Plot|Set X)", self\)', src)

    assert labels == ["Export", "Plot", "Set X"]
    assert '"save.ico"' not in src


# ---------------------------------------------------------------------------
# "Set as X" — the tree's replacement for dragging an X dataset around
# ---------------------------------------------------------------------------

class _XHost:
    _set_dataset_as_x = MainWindow._set_dataset_as_x
    _x_dataset_path = MainWindow._x_dataset_path

    def __init__(self, widget=None):
        self._x_dataset_key = None
        self._widget = widget
        self.status: list[str] = []

    def _current_plot_widget_1d(self):
        return self._widget

    def _set_status_text(self, text=""):
        self.status.append(text)


@pytest.fixture
def h5_with_x(tmp_path):
    import h5py

    path = tmp_path / "scanx_0033.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan_0033/scan_data")
        grp.create_dataset("data_01", data=np.arange(10, dtype=float))
        grp.create_dataset("actuator_1_1", data=np.linspace(-7000.0, 7000.0, 10))
        grp.create_dataset("frames", data=np.zeros((4, 4)))
    return path


def _keys(path, ds_path):
    return [str(path)] + ds_path.split("/")


def test_set_as_x_applies_to_the_displayed_curve(qapp, h5_with_x, calls):
    plot = PlotWidget1DEnhanced()
    plot.set_data(np.arange(10, dtype=float))
    host = _XHost(plot)

    host._set_dataset_as_x(_keys(h5_with_x, "scan_0033/scan_data/actuator_1_1"))

    np.testing.assert_allclose(plot.x_data, np.linspace(-7000.0, 7000.0, 10))
    assert plot.x_dataset_path.endswith("actuator_1_1")


def test_set_as_x_is_remembered_as_a_bare_dataset_path_for_export(qapp, h5_with_x, calls):
    host = _XHost()
    host._set_dataset_as_x(_keys(h5_with_x, "scan_0033/scan_data/actuator_1_1"))

    # The export dialog addresses a path inside each scan file, so no file half.
    assert host._x_dataset_path() == "scan_0033/scan_data/actuator_1_1"
    assert "::" in host._x_dataset_key


def test_no_choice_yet_means_an_empty_default(qapp):
    assert _XHost()._x_dataset_path() == ""


def test_a_length_mismatch_keeps_the_choice_but_leaves_the_curve_alone(qapp, h5_with_x, calls):
    """The next scan may well fit, so the default is not thrown away."""
    plot = PlotWidget1DEnhanced()
    plot.set_data(np.arange(99, dtype=float))
    host = _XHost(plot)

    host._set_dataset_as_x(_keys(h5_with_x, "scan_0033/scan_data/actuator_1_1"))

    assert plot.x_data is None
    assert host._x_dataset_path().endswith("actuator_1_1")
    assert "not applied" in host.status[-1]


def test_a_2d_dataset_is_refused_as_an_axis(qapp, h5_with_x, calls):
    host = _XHost()
    host._set_dataset_as_x(_keys(h5_with_x, "scan_0033/scan_data/frames"))

    assert host._x_dataset_key is None
    assert calls["boxes"] == ["warning"]


def test_an_unreadable_dataset_reports_instead_of_raising(qapp, h5_with_x, calls):
    host = _XHost()
    host._set_dataset_as_x(_keys(h5_with_x, "scan_0033/scan_data/not_there"))

    assert host._x_dataset_key is None
    assert calls["boxes"] == ["critical"]


def test_setting_x_without_a_curve_on_screen_still_records_it(qapp, h5_with_x, calls):
    host = _XHost(None)
    host._set_dataset_as_x(_keys(h5_with_x, "scan_0033/scan_data/actuator_1_1"))

    assert host._x_dataset_path().endswith("actuator_1_1")
    assert "no curve displayed" in host.status[-1]


def test_the_export_dialog_starts_with_the_chosen_x(qapp):
    """The point of the whole feature: no dragging on the next export."""
    from src.gui.batch_export import BatchExportDialog

    dialog = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0033"],
        dataset_path="/scan_0033/scan_data/data_01",
        sample_data=np.arange(10, dtype=float),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        default_x_path="scan_0033/scan_data/actuator_1_1",
    )
    dialog.close()

    assert dialog.le_x_path.text() == "scan_0033/scan_data/actuator_1_1"
    assert dialog.settings()["x_path"] == "scan_0033/scan_data/actuator_1_1"


def test_quick_export_writes_a_file_end_to_end(qapp, tmp_path, monkeypatch):
    """The queued action really reaches the writer, not just a stub."""
    names: list[str] = []

    def fake_dialog(_p, _t, default_path, selected_filter, *a, **k):
        names.append(pathlib.Path(default_path).name)
        return str(tmp_path / pathlib.Path(default_path).name), selected_filter.split(";;")[0]

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, **k: None))

    viewer = UnifiedDataViewer()
    viewer.set_data(np.arange(6, dtype=float), source_dataset_key="scanx_0033.nxs::/a/data_01")

    _Host(viewer, pending="export")._run_pending_tree_action()

    assert names == ["scanx_0033_data_01.txt"]
    written = tmp_path / "scanx_0033_data_01.txt"
    assert written.exists()
    # No X dataset was set, so the file is a bare Y column — no index column.
    assert written.read_text(encoding="utf-8").splitlines()[0] == "Y"

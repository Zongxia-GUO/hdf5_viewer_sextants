"""What a text file's columns mean once they reach the display.

Two columns are an x-y curve — the first is the abscissa, not a second curve
drawn against the row number, and the header names the axes. More columns have
no such reading, so they are shown as the table they are. One column has no
abscissa of its own, so the row index is it.

None of this may reach an HDF5 dataset of the same shape, where N narrow
columns really are N curves.
"""

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView

from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
from src.gui.unified_data_viewer import UnifiedDataViewer, _is_text_source

TWO_COLUMNS = "energy intensity\n1 10\n2 20\n3 30\n"


@pytest.fixture
def viewer(qapp):
    widget = UnifiedDataViewer()
    yield widget
    widget.close()
    widget.deleteLater()


def show(viewer, path, values):
    viewer.set_data(np.asarray(values), source_dataset_key=f"{path}::data")
    return viewer.current_widget


def bottom_label(plot):
    return plot.plot_widget.getAxis("bottom").labelText


# ── Which widget ──────────────────────────────────────────────────────── #

def test_two_columns_are_one_curve_against_the_first(qapp, viewer, tmp_path):
    path = tmp_path / "curve.txt"
    path.write_text(TWO_COLUMNS, encoding="utf-8")

    widget = show(viewer, path, [[1, 10], [2, 20], [3, 30]])

    assert isinstance(widget, PlotWidget1DEnhanced)
    assert widget.y_data.ndim == 1, "one curve, not two"
    np.testing.assert_array_equal(widget.y_data, [10, 20, 30])
    np.testing.assert_array_equal(widget.x_data, [1, 2, 3])


def test_the_header_names_the_axes(qapp, viewer, tmp_path):
    path = tmp_path / "curve.txt"
    path.write_text(TWO_COLUMNS, encoding="utf-8")

    widget = show(viewer, path, [[1, 10], [2, 20], [3, 30]])

    assert "energy" in bottom_label(widget)
    assert "intensity" in widget.plot_widget.getAxis("left").labelText


def test_no_header_still_means_x_and_y(qapp, viewer, tmp_path):
    """The columns mean the same thing whether or not the file says so."""
    path = tmp_path / "plain.txt"
    path.write_text("1 10\n2 20\n", encoding="utf-8")

    widget = show(viewer, path, [[1, 10], [2, 20]])

    np.testing.assert_array_equal(widget.x_data, [1, 2])
    np.testing.assert_array_equal(widget.y_data, [10, 20])


def test_three_columns_are_shown_as_a_table(qapp, viewer, tmp_path):
    path = tmp_path / "three.txt"
    path.write_text("energy i0 i1\n1 10 100\n2 20 200\n", encoding="utf-8")

    widget = show(viewer, path, [[1, 10, 100], [2, 20, 200]])

    assert not isinstance(widget, PlotWidget1DEnhanced)


def test_one_column_is_drawn_against_the_row_index(qapp, viewer, tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("intensity\n10\n20\n30\n", encoding="utf-8")

    widget = show(viewer, path, [10, 20, 30])

    assert isinstance(widget, PlotWidget1DEnhanced)
    assert widget.x_data is None
    assert bottom_label(widget) == "Index"


# ── And not to anything else ──────────────────────────────────────────── #

def test_an_hdf5_dataset_of_the_same_shape_is_still_curves(qapp, viewer):
    """Three narrow columns in an HDF5 file are three curves; the text rules
    must not reach them."""
    widget = show(viewer, "C:/data/scan.h5", [[1, 10, 100], [2, 20, 200]])

    assert isinstance(widget, PlotWidget1DEnhanced)
    assert widget.y_data.ndim == 2


def test_a_source_is_recognised_by_its_suffix():
    assert _is_text_source("C:/data/curve.txt::data")
    assert _is_text_source("C:/data/CURVE.CSV::data")
    assert not _is_text_source("C:/data/scan.h5::entry/data")
    assert not _is_text_source(None)


def test_switching_to_another_dataset_drops_the_names(qapp, tmp_path):
    """Otherwise the previous file's header sits over the next one's axes."""
    plot = PlotWidget1DEnhanced()
    try:
        plot.set_source_dataset_key("C:/data/curve.txt::data")
        plot.set_axis_names("energy", "intensity")
        plot.set_data(np.array([10.0, 20.0]), np.array([1.0, 2.0]))
        assert "energy" in bottom_label(plot)

        plot.set_source_dataset_key("C:/data/scan.h5::entry/data")
        plot.set_data(np.array([10.0, 20.0]), np.array([1.0, 2.0]))

        assert "energy" not in bottom_label(plot)
    finally:
        plot.close()
        plot.deleteLater()


def test_the_table_keeps_the_header_over_its_columns(qapp, viewer, tmp_path):
    """Otherwise "Col 0" sits over data that arrived with a name."""
    path = tmp_path / "three.txt"
    path.write_text("energy i0 i1\n1 10 100\n2 20 200\n", encoding="utf-8")

    show(viewer, path, [[1, 10, 100], [2, 20, 200]])
    table = viewer.current_widget.findChild(QAbstractItemView)
    model = table.model()

    labels = [model.headerData(i, Qt.Orientation.Horizontal,
                               Qt.ItemDataRole.DisplayRole) for i in range(3)]
    assert labels == ["energy", "i0", "i1"]

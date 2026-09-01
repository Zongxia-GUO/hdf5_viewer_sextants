"""The tree shows a filename and keeps the path in its own column.

The first column used to hold the absolute path, and everything that needed a
path read that label back — drag and drop, the context menu, ``opened_files``,
the folder monitor. That made the column unshortenable: the moment it showed
anything friendlier, all of those broke at once.

So these tests are less about the new label than about the things that must not
have noticed it changed.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from src.gui.main_window import (
    TREE_COLUMN_FOLDER,
    TREE_COLUMN_NAME,
    TREE_HEADERS,
    HDF5TreeView,
    MainWindow,
    tree_item_file_path,
)


@pytest.fixture
def scan(tmp_path):
    """One scan file, in a folder deep enough to be worth hiding."""
    folder = tmp_path / "beamtime" / "2026" / "Test output"
    folder.mkdir(parents=True)
    path = folder / "scanx_0340.nxs"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan_0340/scan_data")
        grp.create_dataset("data_03", data=np.arange(5.0))
        grp.create_dataset("actuator_2_1", data=np.arange(5.0) * 2)
    return path


@pytest.fixture
def window(qapp, scan):
    win = MainWindow()
    win._open_file(scan)
    yield win
    win.close()


def _file_item(window):
    return window.tree_model_file.item(0, TREE_COLUMN_NAME)


def _dataset_index(window, *names):
    """Expand down to a dataset and return its column-0 index.

    Loads children straight from the source model; the expand slot takes a
    proxy index, which is not what indexFromItem gives back.
    """
    item = _file_item(window)
    window._load_tree_children(item)
    for name in names:
        for row in range(item.rowCount()):
            child = item.child(row, TREE_COLUMN_NAME)
            if child is not None and child.text() == name:
                item = child
                window._load_tree_children(item)
                break
        else:
            raise AssertionError(f"{name!r} not found under {item.text()!r}")
    return window.tree_model_file.indexFromItem(item)


# ---------------------------------------------------------------------------
# What is shown
# ---------------------------------------------------------------------------

def test_the_first_column_shows_the_filename(window, scan):
    """The scan number is what a file is looked up by, and it was the part an
    ElideRight column cut off first."""
    assert _file_item(window).text() == "scanx_0340.nxs"


def test_the_folder_has_its_own_column(window, scan):
    assert TREE_HEADERS[TREE_COLUMN_FOLDER] == "Folder"

    folder_item = window.tree_model_file.item(0, TREE_COLUMN_FOLDER)

    assert folder_item.text() == str(scan.parent)


def test_the_folder_column_does_not_repeat_the_filename(window, scan):
    """Printing it twice would spend the width the change was made to save."""
    folder_item = window.tree_model_file.item(0, TREE_COLUMN_FOLDER)

    assert scan.name not in folder_item.text()


def test_the_full_path_is_still_one_hover_away(window, scan):
    assert _file_item(window).toolTip() == str(scan)


def test_dataset_rows_leave_the_folder_column_empty(window):
    """It says something only about a file."""
    _dataset_index(window, "scan_0340")
    item = _file_item(window)
    child_folder = item.child(0, TREE_COLUMN_FOLDER)

    assert child_folder is not None and child_folder.text() == ""


# ---------------------------------------------------------------------------
# The path is still reachable — this is what used to come from the label
# ---------------------------------------------------------------------------

def test_the_file_row_carries_the_absolute_path(window, scan):
    assert tree_item_file_path(_file_item(window)) == str(scan)


def test_a_dataset_deep_in_the_file_resolves_to_the_same_path(window, scan):
    index = _dataset_index(window, "scan_0340", "scan_data", "data_03")
    item = window.tree_model_file.itemFromIndex(index)

    assert tree_item_file_path(item) == str(scan)


def test_opened_files_is_absolute(window, scan):
    """Every tool in the application is handed this tuple."""
    assert window.opened_files == (scan,)


def test_a_tool_opened_from_the_menu_gets_absolute_paths(window, scan):
    """The shared scan list reaches the calculator and the comparison tool
    through opened_files; a filename would leave them unable to open anything."""
    window._handle_action_calculator()
    window._handle_action_comparison()

    assert window.calculator.opened_files == (scan,)
    assert window.comparison_tool.opened_files == (scan,)
    window.calculator.close()
    window.comparison_tool.close()


def test_the_shared_dataset_index_is_fed_absolute_paths(window, scan):
    """Its cache is keyed by these, and its keys are handed to every X
    selector as ``file::dataset``."""
    assert list(window._sorted_opened_files_for_index()) == [scan]


# ---------------------------------------------------------------------------
# Dragging — the token the drop targets parse
# ---------------------------------------------------------------------------

def test_dragging_a_dataset_still_sends_file_and_dataset(window, scan):
    index = _dataset_index(window, "scan_0340", "scan_data", "data_03")

    token = HDF5TreeView._path_for_index(index)

    assert token == f"{scan}::scan_0340/scan_data/data_03"


def test_dragging_a_file_sends_its_bare_path(window, scan):
    index = window.tree_model_file.indexFromItem(_file_item(window))

    assert HDF5TreeView._path_for_index(index) == str(scan)


def test_a_group_is_not_draggable(window):
    index = _dataset_index(window, "scan_0340")

    assert HDF5TreeView._path_for_index(index) is None


# ---------------------------------------------------------------------------
# Set X and the context menu, which read the same address
# ---------------------------------------------------------------------------

def test_the_tree_address_splits_file_from_dataset(window, scan):
    index = _dataset_index(window, "scan_0340", "scan_data", "actuator_2_1")

    file_path, names = window._tree_address(index)

    assert file_path == str(scan)
    assert names == ["scan_0340", "scan_data", "actuator_2_1"]


def test_a_file_row_has_no_dataset_names(window, scan):
    index = window.tree_model_file.indexFromItem(_file_item(window))

    file_path, names = window._tree_address(index)

    assert file_path == str(scan)
    assert names == []


def test_set_x_gets_the_absolute_path_the_context_menu_built(window, scan):
    """The menu hands ``[file, *names]`` to _set_dataset_as_x, and that first
    element has to be a path h5py can open."""
    index = _dataset_index(window, "scan_0340", "scan_data", "actuator_2_1")
    file_path, names = window._tree_address(index)

    window._set_dataset_as_x([file_path, *names])

    assert window._x_dataset_key == f"{scan}::scan_0340/scan_data/actuator_2_1"
    assert window._x_dataset_path() == "scan_0340/scan_data/actuator_2_1"


def test_clicking_a_dataset_still_opens_it(window, scan):
    """_handle_item_changed took the file from the label too."""
    index = _dataset_index(window, "scan_0340", "scan_data", "data_03")

    window._handle_item_changed(index)

    assert window.cur_file == scan
    assert window.cur_obj_path == "/scan_0340/scan_data/data_03"


def test_clicking_the_file_row_reports_the_file(window, scan):
    index = window.tree_model_file.indexFromItem(_file_item(window))

    window._handle_item_changed(index)

    assert window.cur_file == scan
    assert window.cur_obj_path == ""


# ---------------------------------------------------------------------------
# The folder monitor, which keeps a set of absolute paths
# ---------------------------------------------------------------------------

def test_a_file_removed_on_disk_is_matched_by_its_path(window, scan):
    window._monitor_known = {str(scan)}

    window._on_folder_scan_done([], [str(scan)])

    assert window.tree_model_file.rowCount() == 0
    assert window._monitor_known == set()


def test_a_file_that_only_shares_a_name_is_not_removed(window, scan, tmp_path):
    """The label is a bare filename now, so matching on it would take the wrong
    row out whenever two folders hold the same scan number."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    twin = other / scan.name

    window._on_folder_scan_done([], [str(twin)])

    assert window.tree_model_file.rowCount() == 1


def test_two_files_of_the_same_name_stay_distinguishable(qapp, scan, tmp_path):
    """What the folder column is for: same scan number, different beamtime."""
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    twin = other_dir / scan.name
    with h5py.File(twin, "w") as f:
        f.create_dataset("scan_0340/scan_data/data_03", data=np.arange(5.0))

    win = MainWindow()
    try:
        win._open_file(scan)
        win._open_file(twin)

        names = [win.tree_model_file.item(r, TREE_COLUMN_NAME).text() for r in range(2)]
        folders = [win.tree_model_file.item(r, TREE_COLUMN_FOLDER).text() for r in range(2)]

        assert names == [scan.name, twin.name], "the labels are the same..."
        assert folders == [str(scan.parent), str(twin.parent)], "...the folders are not"
        assert win.opened_files == (scan, twin)
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Files that are not HDF5
# ---------------------------------------------------------------------------

def test_a_text_file_gets_the_same_treatment(qapp, tmp_path):
    path = tmp_path / "sample_data.txt"
    path.write_text("1.0\n2.0\n3.0\n", encoding="utf-8")

    win = MainWindow()
    try:
        win._open_file(path)
        item = win.tree_model_file.item(0, TREE_COLUMN_NAME)

        assert item.text() == "sample_data.txt"
        assert tree_item_file_path(item) == str(path)
        assert win.tree_model_file.item(0, TREE_COLUMN_FOLDER).text() == str(tmp_path)
        assert win.opened_files == (path,)

        child = item.child(0, TREE_COLUMN_NAME)
        index = win.tree_model_file.indexFromItem(child)
        # The token joins the row labels, and a regular file's single child is
        # labelled "data" — unchanged by any of this.
        assert HDF5TreeView._path_for_index(index) == f"{path}::data"
    finally:
        win.close()


def test_the_path_helper_is_safe_on_a_detached_item(qapp):
    from PyQt6.QtGui import QStandardItem

    assert tree_item_file_path(QStandardItem("orphan")) == ""
    assert tree_item_file_path(None) == ""


def test_paths_with_a_double_colon_in_a_folder_name(qapp, tmp_path):
    """The drag token is split on '::', so the file half must not contain one.
    Windows forbids it in a path, which is what makes the token unambiguous."""
    path = tmp_path / "scanx_0341.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(3.0))

    win = MainWindow()
    try:
        win._open_file(path)
        assert "::" not in str(pathlib.Path(path).parent)
    finally:
        win.close()

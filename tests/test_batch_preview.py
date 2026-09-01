"""Tests that Enter in the batch inputs previews data instead of opening the export dialog."""

import pathlib

import pytest

from src.gui.main_window import MainWindow


class _Line:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text


class _Combo:
    def currentText(self):
        return "Auto"


class _Model:
    def __init__(self):
        self.rows = []

    def resetData(self):
        self.rows = []

    def appendRow(self, row):
        self.rows.append(row)


class _Host:
    """Minimal stand-in for MainWindow's batch-input surface."""

    _resolve_batch_selection = MainWindow._resolve_batch_selection
    _handle_batch_preview = MainWindow._handle_batch_preview
    _parse_scan_range = MainWindow._parse_scan_range
    _matching_files_for_scans = MainWindow._matching_files_for_scans
    _batch_paths_for_operations = MainWindow._batch_paths_for_operations
    _batch_keywords_text = MainWindow._batch_keywords_text
    _batch_range_text = MainWindow._batch_range_text

    def __init__(self, opened_files, prefix="scanx_", scan_range="", path=""):
        self.opened_files = tuple(opened_files)
        self.le_batch_keywords = _Line(prefix)
        self.le_scan_range = _Line(scan_range)
        self.le_batch_path = _Line(path)
        self.cb_plot_type = _Combo()
        self.table_model_dataset = _Model()
        self.cur_file = None
        self.cur_obj_path = None
        self.status_messages = []
        self.plot_requests = []

    def _batch_path_for_operations(self):
        return self.le_batch_path.text().strip()

    def _set_status_text(self, text=""):
        self.status_messages.append(text)

    def _request_plot_data(self, plot_type=""):
        self.plot_requests.append(plot_type)


class _Mime:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def hasText(self):
        return bool(self._text)


class _DropEvent:
    def __init__(self, text):
        self._mime = _Mime(text)
        self.accepted = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True


class _Field:
    """A QLineEdit stand-in that records what the drop handler writes."""

    def __init__(self):
        self.value = ""
        self.tooltip = ""

    def text(self):
        return self.value

    def setText(self, text):
        self.value = text

    def setToolTip(self, text):
        self.tooltip = text


class _DropHost:
    _batch_path_drop = MainWindow._batch_path_drop
    _batch_path_display_text = MainWindow._batch_path_display_text
    _batch_paths_for_operations = MainWindow._batch_paths_for_operations
    _batch_path_for_operations = MainWindow._batch_path_for_operations

    _batch_keywords_text = MainWindow._batch_keywords_text

    def __init__(self):
        self.le_batch_path = _Field()
        self.le_batch_keywords = _Line("scanx_")
        self._batch_path_template = None
        self._batch_path_hidden_prefix = None


# ---------------------------------------------------------------------------
# Dropping several datasets at once
# ---------------------------------------------------------------------------

def test_multi_selection_drop_keeps_every_dataset():
    """A multi-select drag arrives as one token per line; all of them must land."""
    host = _DropHost()
    event = _DropEvent(
        "d:/data/scanx_0080.h5::entry/det1\n"
        "d:/data/scanx_0080.h5::entry/det2\n"
        "d:/data/scanx_0080.h5::entry/det3"
    )
    host._batch_path_drop(event)

    assert event.accepted
    assert host._batch_paths_for_operations() == ["entry/det1", "entry/det2", "entry/det3"]
    assert "3 Y dataset(s)" in host.le_batch_path.tooltip


def test_single_drop_still_hides_the_scan_group():
    host = _DropHost()
    host._batch_path_drop(_DropEvent("d:/data/scanx_0080.h5::scanx_0080/entry/det1"))

    # The per-scan group is hidden from the narrow field but kept in the template.
    assert host.le_batch_path.value == "entry/det1"
    assert host._batch_paths_for_operations() == ["scanx_0080/entry/det1"]


def test_multi_drop_does_not_hide_the_scan_group():
    host = _DropHost()
    host._batch_path_drop(
        _DropEvent(
            "d:/data/scanx_0080.h5::scanx_0080/entry/det1\n"
            "d:/data/scanx_0080.h5::scanx_0080/entry/det2"
        )
    )

    assert host._batch_paths_for_operations() == [
        "scanx_0080/entry/det1",
        "scanx_0080/entry/det2",
    ]


def test_duplicate_drop_tokens_are_collapsed():
    host = _DropHost()
    host._batch_path_drop(_DropEvent("f.h5::entry/det1\nf.h5::entry/det1"))
    assert host._batch_paths_for_operations() == ["entry/det1"]


@pytest.fixture
def files(tmp_path: pathlib.Path):
    paths = []
    for scan in ("0080", "0081", "0082"):
        p = tmp_path / f"scanx_{scan}.h5"
        p.touch()
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# What Enter shows
# ---------------------------------------------------------------------------

def test_single_scan_previews_that_scan(files):
    host = _Host(files, scan_range="0081", path="entry/data")
    host._handle_batch_preview()

    assert host.cur_file.name == "scanx_0081.h5"
    assert host.cur_obj_path == "/entry/data"
    assert host.plot_requests == ["Auto"]


def test_range_previews_the_first_scan(files):
    host = _Host(files, scan_range="0080-0082", path="entry/data")
    host._handle_batch_preview()

    assert host.cur_file.name == "scanx_0080.h5"
    assert "first of 3 matched scans" in host.status_messages[-1]


def test_list_previews_the_first_listed_scan(files):
    """The typed order wins, even though matching_files is ordered by opened file."""
    host = _Host(files, scan_range="0082,0080", path="entry/data")
    host._handle_batch_preview()

    assert host.cur_file.name == "scanx_0082.h5"
    assert host.cur_obj_path == "/entry/data"


def test_unopened_first_scan_falls_back_to_a_real_match(files):
    host = _Host(files, scan_range="0099,0081", path="entry/data")
    host._handle_batch_preview()

    assert host.cur_file.name == "scanx_0081.h5"


def test_scan_number_in_the_path_is_rewritten_for_the_previewed_scan(files):
    host = _Host(files, scan_range="0081-0082", path="scan_0080/data")
    host._handle_batch_preview()

    assert host.cur_file.name == "scanx_0081.h5"
    assert host.cur_obj_path == "/scan_0081/data"


def test_preview_fills_the_property_table(files):
    host = _Host(files, scan_range="0080-0082", path="entry/data")
    host._handle_batch_preview()

    labels = [row[0] for row in host.table_model_dataset.rows]
    assert labels == ["Name", "File", "Scan", "Matched"]
    assert host.table_model_dataset.rows[-1][1] == "3 of 3 scans"


# ---------------------------------------------------------------------------
# Incomplete input must stay quiet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prefix, scan_range, path",
    [
        ("scanx_", "", "entry/data"),       # still typing the range
        ("scanx_", "0080", ""),             # no dataset path yet
        ("", "0080", "entry/data"),         # prefix cleared
        ("scanx_", "not-a-range", "entry/data"),
        ("scanx_", "0999", "entry/data"),   # no such file open
    ],
)
def test_incomplete_input_reports_on_the_status_bar_only(files, prefix, scan_range, path):
    host = _Host(files, prefix=prefix, scan_range=scan_range, path=path)
    host._handle_batch_preview()

    assert host.plot_requests == []
    assert host.cur_file is None
    assert len(host.status_messages) == 1
    assert "\n" not in host.status_messages[0]


def test_quiet_resolution_never_raises_without_a_qt_dialog(files):
    """quiet=True must not reach QMessageBox — the stub has no parent widget."""
    host = _Host(files, scan_range="", path="")
    assert host._resolve_batch_selection(quiet=True) is None


def test_resolution_returns_matches_when_complete(files):
    host = _Host(files, scan_range="0080,0081", path="entry/data")
    resolved = host._resolve_batch_selection(quiet=True)

    assert resolved is not None
    matching_files, batch_paths, scan_numbers = resolved
    assert [p.name for p, _s in matching_files] == ["scanx_0080.h5", "scanx_0081.h5"]
    assert batch_paths == ["entry/data"]
    assert scan_numbers == ["0080", "0081"]


def test_several_dropped_datasets_are_split(files):
    host = _Host(files, scan_range="0080", path="entry/det1; entry/det2 ; entry/det1")
    resolved = host._resolve_batch_selection(quiet=True)

    assert resolved is not None
    _matching, batch_paths, _scans = resolved
    # Order preserved, duplicates dropped.
    assert batch_paths == ["entry/det1", "entry/det2"]


def test_preview_shows_the_first_of_several_datasets(files):
    host = _Host(files, scan_range="0080", path="entry/det1; entry/det2")
    host._handle_batch_preview()

    assert host.cur_obj_path == "/entry/det1"

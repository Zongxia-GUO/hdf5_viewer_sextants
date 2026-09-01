"""Tests for the shared tabular writer and the save-dialog filter helpers."""

import numpy as np
import pytest

from src.lib_h5.table_format import (
    TABLE_FORMATS,
    format_from_filter,
    get_table_format,
    save_dialog_filter,
)
from src.lib_h5.table_writer import columns_from_2d, write_table


def _lines(path, fmt):
    return path.read_text(encoding=fmt.encoding).splitlines()


# ---------------------------------------------------------------------------
# write_table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, sep, first_row", [
    ("txt", "\t", "1.5\t10"),
    ("csv", ",", "1.5,10"),
    ("csv2", ";", "1,5;10"),
])
def test_each_dialect_writes_its_own_separator_and_decimal(tmp_path, key, sep, first_row):
    fmt = get_table_format(key)
    out = write_table(
        tmp_path / f"t{fmt.suffix}",
        ["X", "Y"],
        [np.array([1.5, 2.5]), np.array([10.0, 20.0])],
        fmt,
    )
    lines = _lines(out, fmt)
    assert lines[0] == f"X{sep}Y"
    assert lines[1] == first_row


def test_ragged_columns_are_padded(tmp_path):
    fmt = get_table_format("txt")
    out = write_table(
        tmp_path / "t.txt",
        ["A", "B"],
        [np.array([1.0, 2.0, 3.0]), np.array([9.0])],
        fmt,
    )
    rows = [line.split("\t") for line in _lines(out, fmt)]
    assert rows[1] == ["1", "9"]
    assert rows[2] == ["2", ""]
    assert rows[3] == ["3", ""]


def test_comments_are_written_verbatim_above_the_header(tmp_path):
    fmt = get_table_format("csv2")
    out = write_table(
        tmp_path / "t.csv",
        ["A"],
        [np.array([1.0])],
        fmt,
        comments=["# mode: aligned", "# note: a;b inside a comment"],
    )
    lines = _lines(out, fmt)
    assert lines[0] == "# mode: aligned"
    # Raw, not run through the csv writer, so the ';' does not get the line quoted.
    assert lines[1] == "# note: a;b inside a comment"
    assert lines[2] == "A"


def test_header_count_must_match_columns(tmp_path):
    with pytest.raises(ValueError, match="2 headers for 1 columns"):
        write_table(tmp_path / "t.txt", ["A", "B"], [np.array([1.0])], get_table_format("txt"))


def test_empty_column_set_writes_only_the_header(tmp_path):
    fmt = get_table_format("txt")
    out = write_table(tmp_path / "t.txt", [], [], fmt)
    assert _lines(out, fmt) == [""]


def test_only_csv_dialects_get_a_bom(tmp_path):
    for key, has_bom in (("txt", False), ("csv", True), ("csv2", True)):
        fmt = get_table_format(key)
        out = write_table(tmp_path / f"{key}{fmt.suffix}", ["A"], [np.array([1.0])], fmt)
        assert (out.read_bytes()[:3] == b"\xef\xbb\xbf") is has_bom


def test_columns_from_2d_splits_and_passes_1d_through():
    assert len(columns_from_2d(np.zeros((5, 3)))) == 3
    assert len(columns_from_2d(np.zeros(5))) == 1


# ---------------------------------------------------------------------------
# Save-dialog filters
# ---------------------------------------------------------------------------

def test_filter_lists_every_dialect():
    entries = save_dialog_filter().split(";;")
    assert len(entries) == len(TABLE_FORMATS)


def test_preferred_dialect_is_listed_first():
    entries = save_dialog_filter(get_table_format("csv2")).split(";;")
    assert entries[0].startswith(get_table_format("csv2").label)


def test_filter_round_trips_back_to_its_dialect():
    for fmt in TABLE_FORMATS:
        entry = save_dialog_filter(fmt).split(";;")[0]
        assert format_from_filter(entry).key == fmt.key


def test_csv_and_csv2_are_told_apart_despite_the_shared_suffix():
    """Both write .csv, so the extension alone cannot resolve the dialect."""
    csv_entry = save_dialog_filter(get_table_format("csv")).split(";;")[0]
    csv2_entry = save_dialog_filter(get_table_format("csv2")).split(";;")[0]
    assert get_table_format("csv").suffix == get_table_format("csv2").suffix
    assert format_from_filter(csv_entry).key == "csv"
    assert format_from_filter(csv2_entry).key == "csv2"


def test_unknown_filter_falls_back_to_the_default():
    assert format_from_filter("Whatever (*.dat)").key == "txt"
    assert format_from_filter(None).key == "txt"

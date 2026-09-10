"""A text file's header names its columns; it is not a row of data.

``numpy.genfromtxt`` reports a value it could not parse as NaN rather than
raising, and the loader used to accept any array that came back with something
in it — which an array of NaN is. So a header became a row of NaN, a ``#``
header destroyed the whole file, and a comma-separated ``.txt`` came back empty.
Each of those is a case below, written against a file on disk.
"""

import numpy as np
import pytest

from src.lib_h5.text_table import (
    column_names,
    header_names,
    read_text_table,
    sniff_delimiter,
    split_header,
)

NAMES = ("energy", "intensity")


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ── The three failures that started this ──────────────────────────────── #

def test_a_header_is_not_a_row_of_nan(tmp_path):
    path = write(tmp_path, "header.txt", "energy intensity\n1 10\n2 20\n3 30\n")

    table = read_text_table(path)

    assert table.values.shape == (3, 2)
    assert np.isfinite(table.values).all()
    assert table.names == NAMES


def test_a_hash_header_does_not_destroy_the_file(tmp_path):
    """``comments=None`` had turned off the handling that skips it, so every
    value came back NaN."""
    path = write(tmp_path, "hash.txt", "# energy intensity\n1 10\n2 20\n3 30\n")

    table = read_text_table(path)

    assert np.isfinite(table.values).all()
    assert table.names == NAMES


def test_a_comma_separated_txt_is_read_as_commas(tmp_path):
    """Splitting ``1.0,10.0`` on whitespace yields one field that is not a
    number. The retry-with-commas only ran on an exception, which never came."""
    path = write(tmp_path, "comma.txt", "1,10\n2,20\n3,30\n")

    table = read_text_table(path)

    assert table.values.shape == (3, 2)
    assert np.isfinite(table.values).all()


def test_a_word_and_a_number_is_not_a_data_point(tmp_path):
    """``Scan 42`` holds a number but is not a measurement; reading it as one
    invented the point ``[nan, 42.0]``."""
    path = write(tmp_path, "scan.txt", "Scan 42\nenergy intensity\n1 10\n2 20\n")

    table = read_text_table(path)

    assert table.values.shape == (2, 2)
    assert np.isfinite(table.values).all()
    assert table.names == NAMES


# ── Shapes a beamline actually writes ─────────────────────────────────── #

@pytest.mark.parametrize("name,text", [
    ("plain.txt", "1 10\n2 20\n3 30\n"),
    ("tab.txt", "energy\tintensity\n1\t10\n2\t20\n"),
    ("semi.txt", "energy;intensity\n1;10\n2;20\n"),
    ("comma_header.txt", "energy,intensity\n1,10\n2,20\n"),
    ("trailing.txt", "energy intensity\n1 10\n2 20\n\n"),
    ("crlf.txt", "energy intensity\r\n1 10\r\n2 20\r\n"),
])
def test_every_shape_parses_without_a_nan(tmp_path, name, text):
    table = read_text_table(write(tmp_path, name, text))

    assert table.values.shape[1] == 2
    assert np.isfinite(table.values).all(), name


def test_a_file_with_no_header_has_no_names(tmp_path):
    assert read_text_table(write(tmp_path, "plain.txt", "1 10\n2 20\n")).names == ()


def test_a_units_row_is_passed_over_for_the_names_above_it(tmp_path):
    """``(eV) (counts)`` describes the columns but does not name them."""
    path = write(tmp_path, "units.txt", "energy intensity\n(eV) (counts)\n1 10\n2 20\n")

    assert read_text_table(path).names == NAMES


def test_units_are_used_when_they_are_all_there_is(tmp_path):
    path = write(tmp_path, "only_units.txt", "(eV) (counts)\n1 10\n2 20\n")

    assert read_text_table(path).names == ("(eV)", "(counts)")


def test_a_csv_is_commas_whatever_its_contents_suggest(tmp_path):
    assert sniff_delimiter(["1 10"], ".csv") == ","


def test_a_header_whose_width_does_not_match_is_not_used(tmp_path):
    """A title line is not a set of names, however tempting its position."""
    path = write(tmp_path, "title.txt", "measured on tuesday\n1 10\n2 20\n")

    assert read_text_table(path).names == ()


def test_three_columns_keep_all_three_names(tmp_path):
    path = write(tmp_path, "three.txt", "energy i0 i1\n1 10 100\n2 20 200\n")

    table = read_text_table(path)

    assert table.values.shape == (2, 3)
    assert table.names == ("energy", "i0", "i1")


def test_one_column_is_a_column(tmp_path):
    table = read_text_table(write(tmp_path, "one.txt", "intensity\n10\n20\n30\n"))

    assert table.values.shape == (3,)
    assert table.columns == 1
    assert table.names == ("intensity",)


def test_a_file_of_words_comes_back_as_text_not_a_grid_of_nan(tmp_path):
    path = write(tmp_path, "notes.txt", "this file\nhas no numbers\n")

    table = read_text_table(path)

    assert table.values.dtype.kind in "US"


# ── The pieces, checked directly ──────────────────────────────────────── #

def test_split_header_stops_at_the_first_all_numeric_line():
    lines = ["Scan 42", "energy intensity", "1 10", "2 20"]

    header, first = split_header(lines, None)

    assert first == 2
    assert header == lines[:2]


def test_header_names_reads_from_the_bottom_up():
    """The line nearest the numbers is the one that describes them."""
    header = ["sample A B", "energy intensity"]

    assert header_names(header, None, 2) == NAMES


# ── The names cache ───────────────────────────────────────────────────── #

def test_names_are_remembered_and_re_read_when_the_file_changes(tmp_path):
    path = write(tmp_path, "cached.txt", "energy intensity\n1 10\n2 20\n")
    assert column_names(path) == NAMES

    import os
    stat = path.stat()
    path.write_text("wavelength counts\n1 10\n2 20\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_mtime_ns + 10**9, stat.st_mtime_ns + 10**9))

    assert column_names(path) == ("wavelength", "counts")


def test_a_missing_file_has_no_names_rather_than_an_error(tmp_path):
    assert column_names(tmp_path / "nothing.txt") == ()

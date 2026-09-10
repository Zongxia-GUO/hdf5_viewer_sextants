"""Reading a text or CSV file that may carry a header.

``numpy.genfromtxt`` does not raise when it cannot parse a value — it returns
NaN. The loader here used to treat "an array came back with something in it" as
success, which is true of an array that is entirely NaN, so every way of
failing was accepted silently. Three things went wrong because of it, all
measured:

* a header line became a row of NaN, and a line like ``Scan 42`` became the
  data point ``[nan, 42.0]`` — a number that was never measured;
* a ``#`` header destroyed the whole file, because ``comments=None`` had turned
  off the handling that would have skipped it;
* a comma-separated ``.txt`` came back as a column of NaN, and the fallback
  that would have retried with commas only ran on an exception, which never
  came.

So parsing is decided here instead: find the delimiter that yields the most
numbers, treat the leading lines that are not entirely numeric as a header, and
report failure as failure. No Qt, so it can be tested against files rather than
against the application.
"""

# Copyright (C) 2023 Dennis Lönard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import functools
import logging
import pathlib
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Delimiters to try for a ``.txt``, in the order they are preferred when two
#: of them parse equally well. Whitespace first because it is what
#: ``numpy.savetxt`` writes and what most beamline software emits.
CANDIDATE_DELIMITERS: tuple[str | None, ...] = (None, ",", "\t", ";")

#: Anything from this character to the end of a line is a comment.
COMMENT = "#"

#: Lines read to decide the delimiter and find the header. A header is a few
#: lines at most, so there is no reason to scan a large file to find it.
SNIFF_LINES = 200


@dataclass(frozen=True)
class TextTable:
    """The numbers in a text file, and the column names above them."""

    values: np.ndarray
    #: One per column, from the last header line that has the right count.
    #: Empty when the file has no header this can be read from.
    names: tuple[str, ...] = ()

    @property
    def columns(self) -> int:
        return int(self.values.shape[1]) if self.values.ndim >= 2 else 1


def _strip_comment(line: str) -> str:
    return line.split(COMMENT, 1)[0]


def _fields(line: str, delimiter: str | None) -> list[str]:
    body = _strip_comment(line)
    parts = body.split() if delimiter is None else body.split(delimiter)
    return [p.strip() for p in parts if p.strip() != ""]


def _all_numeric(fields: list[str]) -> bool:
    """Whether every field on the line is a number.

    Every field, not any: a line like ``Scan 42`` holds a number but is not a
    row of data, and reading it as one invents the measurement ``42``.
    """
    if not fields:
        return False
    for field in fields:
        try:
            float(field)
        except ValueError:
            return False
    return True


def sniff_delimiter(lines: list[str], suffix: str) -> str | None:
    """Pick the delimiter that reads the most numbers out of these lines.

    A ``.csv`` is comma-separated by definition. For a ``.txt`` the file has to
    be asked: splitting ``1.0,10.0`` on whitespace yields the single field
    ``"1.0,10.0"``, which is not a number, and the old code accepted the NaN
    that came back rather than trying the comma.
    """
    if suffix == ".csv":
        return ","

    best: str | None = None
    best_score = (0, 0)
    for delimiter in CANDIDATE_DELIMITERS:
        numeric_rows = 0
        widest = 0
        for line in lines:
            fields = _fields(line, delimiter)
            if _all_numeric(fields):
                numeric_rows += 1
                widest = max(widest, len(fields))
        score = (numeric_rows, widest)
        if score > best_score:
            best_score = score
            best = delimiter
    return best


def split_header(lines: list[str], delimiter: str | None) -> tuple[list[str], int]:
    """Return the header lines and the index of the first data line.

    A header is however many leading lines are not entirely numeric — one line
    of names, or names and units, or a scan number above them. Blank and
    comment-only lines count as header too; they carry no data either way.
    """
    for index, line in enumerate(lines):
        if _all_numeric(_fields(line, delimiter)):
            return lines[:index], index
    return list(lines), len(lines)


def _looks_like_units(fields: list[str]) -> bool:
    """Whether every field is bracketed, as a row of units usually is.

    ``(eV) (counts)`` describes the columns but does not name them, and a file
    that writes names and then units puts the units nearest the numbers — so
    the row closest to the data is the wrong one to take there.
    """
    return all(
        (f.startswith("(") and f.endswith(")")) or (f.startswith("[") and f.endswith("]"))
        for f in fields
    )


def header_names(header: list[str], delimiter: str | None, columns: int) -> tuple[str, ...]:
    """Column names from the header, or empty if it does not supply them.

    Read from the bottom up, because a file often carries a scan number or a
    title above the names and it is the line nearest the numbers that describes
    them. A comment marker is not part of a name: ``# energy intensity`` is the
    commonest way of writing a header and means the same as writing it plain.
    A row of units is passed over while any other candidate remains.
    """
    candidates = []
    for line in reversed(header):
        fields = _fields(line.lstrip().lstrip(COMMENT), delimiter)
        if len(fields) == columns and not _all_numeric(fields):
            candidates.append(tuple(fields))
    if not candidates:
        return ()
    for fields in candidates:
        if not _looks_like_units(list(fields)):
            return fields
    return candidates[0]


def read_text_table(path: str | pathlib.Path) -> TextTable:
    """Read a text or CSV file into numbers, skipping any header above them."""
    file_path = pathlib.Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    delimiter = sniff_delimiter(lines[:SNIFF_LINES], file_path.suffix.lower())
    header, first_data_line = split_header(lines, delimiter)

    if first_data_line < len(lines):
        values = np.asarray(np.genfromtxt(
            file_path,
            delimiter=delimiter,
            comments=COMMENT,
            skip_header=first_data_line,
            dtype=float,
        ))
        if values.size and np.isfinite(values).any():
            names = header_names(header, delimiter, TextTable(values).columns)
            return TextTable(values, names)

    # Nothing numeric anywhere: it is a text file in the ordinary sense, and
    # showing it as text beats showing a grid of NaN. Split it here rather than
    # asking genfromtxt again, which refuses a file whose lines are not all the
    # same width — which is exactly what free text is.
    log.debug("No numeric data found in %s; reading it as text.", file_path)
    rows = [_fields(line, delimiter) for line in lines]
    width = max((len(row) for row in rows), default=0)
    if width <= 1:
        return TextTable(np.array([line.rstrip() for line in lines], dtype=str))
    padded = [row + [""] * (width - len(row)) for row in rows]
    return TextTable(np.array(padded, dtype=str))


@functools.lru_cache(maxsize=64)
def _cached_names(path_str: str, signature: tuple[int, int]) -> tuple[str, ...]:
    del signature  # only here to make a rewritten file a different cache entry
    try:
        return read_text_table(path_str).names
    except Exception as exc:                       # pragma: no cover - defensive
        log.debug("Could not read column names from %s: %s", path_str, exc)
        return ()


def column_names(path: str | pathlib.Path) -> tuple[str, ...]:
    """The column names of a text file, remembered between reads.

    The display asks for these separately from the numbers, so that carrying
    them did not have to be threaded through every signature between the
    loader and the axis. Keyed on the file's size and modification time, so an
    edited file is read again.
    """
    file_path = pathlib.Path(path)
    try:
        stat = file_path.stat()
    except OSError:
        return ()
    return _cached_names(str(file_path), (int(stat.st_mtime_ns), int(stat.st_size)))

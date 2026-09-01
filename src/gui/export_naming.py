"""Short names for what a curve is, wherever it has to be written down.

A quick export needs a file name, a legend needs an entry, a table needs a
column header — and all three start from the same ``<file>::<dataset>`` key.
Spelled out in full that key is unusable: a legend of
``scanx_0083.nxs::scan_0083/scan_data/data_04 [Col 2]`` covers the plot it is
labelling, and the same text as a column header makes a table unreadable.

The convention is the scan file's stem, the dataset's leaf, and the column
number only when one was picked: ``scanx_0083_data_04_col2``. It is the
discriminating part — everything dropped is shared by every curve on screen.
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

import pathlib
import re

from PyQt6.QtCore import QSettings

# The comparison tool marks a chosen column on the end of a dataset name.
_COLUMN_SUFFIX = re.compile(r"\s*\[Col\s+(\d+)\]\s*$", re.IGNORECASE)

# One folder for every save in the application. Not split by kind: the user
# thinks "where did I put things last time", not "where did I put my PNGs".
_LAST_DIRECTORY = "paths/last_export_directory"


def last_save_directory() -> pathlib.Path:
    """The folder the last save went to, or home when there is none yet."""
    stored = QSettings().value(_LAST_DIRECTORY, "")
    if stored:
        folder = pathlib.Path(str(stored))
        if folder.is_dir():
            return folder
    return pathlib.Path.home()


def remember_save_directory(path: str | pathlib.Path) -> None:
    """Record where a save landed, for the next one to start from."""
    target = pathlib.Path(path)
    folder = target if target.is_dir() else target.parent
    if str(folder):
        QSettings().setValue(_LAST_DIRECTORY, str(folder))


def suggested_save_path(stem: str, suffix: str = "", extension: str = "") -> str:
    """Where a save dialog should open, and under what name.

    The name is what the data *is* — scan and dataset — plus what was done to
    it, and that second part only when something was: a faithful copy of one
    dataset needs no tag, while a calculator result or one of four
    reconstructed components would be ambiguous without one.
    """
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{stem}_{suffix}" if suffix else stem).strip("_")
    name = clean or "export"
    if extension and not name.lower().endswith(extension.lower()):
        name = f"{name}{extension}"
    return str(last_save_directory() / name)


def split_key(source_key: str | None) -> tuple[str, str, str] | None:
    """A key as ``(file stem, dataset leaf, column)``, or None when unusable."""
    key = (source_key or "").strip()
    if not key:
        return None

    column = ""
    match = _COLUMN_SUFFIX.search(key)
    if match:
        column = f"col{int(match.group(1))}"
        key = key[: match.start()].strip()

    if "::" in key:
        file_part, ds_part = key.rsplit("::", 1)
        return (
            pathlib.Path(file_part.strip()).stem,
            ds_part.strip().rstrip("/").split("/")[-1],
            column,
        )
    return pathlib.Path(key).stem, "", column


def scan_head_and_number(stem: str) -> tuple[str, str]:
    """``scanx_0083`` as ``("scanx", "0083")``; no trailing digits gives ``("", "")``."""
    match = re.search(r"^(.*?)(\d+)$", stem.strip())
    if not match:
        return "", ""
    return match.group(1).rstrip("_- "), match.group(2)


def _merge_tokens(first: str, second: str) -> str:
    """Join two names, writing out only the one token that differs.

    ``scanx_0083_data_04`` and ``scanx_0085_data_04`` give
    ``scanx_0083_0085_data_04``. Reached when the pair is already-shortened
    labels rather than ``file::dataset`` keys — the plot window names its
    figure from the curve labels it is showing.
    """
    left = first.split("_")
    right = second.split("_")
    if len(left) == len(right):
        differing = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(differing) == 1:
            at = differing[0]
            return "_".join(left[: at + 1] + [right[at]] + left[at + 1:])
    return f"{first}_{second}"


def pair_label(first: str | None, second: str | None) -> str:
    """Name two combined sources by whatever actually differs between them.

    Two scans give ``scanx_0083_0085_data_04``; one scan's two datasets give
    ``scanx_0083_data_04_data_10``; one dataset's two columns give
    ``scanx_0083_data_04_col1_col2``. Repeating the identical half says nothing
    and dropping the differing half loses the point of the file, so only the
    part that actually distinguishes them is written twice.
    """
    left = split_key(first)
    right = split_key(second)
    if left is None and right is None:
        return "export"
    if right is None or left == right:
        return short_series_label(first or second)
    if left is None:
        return short_series_label(second)

    stem_a, leaf_a, col_a = left
    stem_b, leaf_b, col_b = right

    if stem_a != stem_b:
        head_a, num_a = scan_head_and_number(stem_a)
        head_b, num_b = scan_head_and_number(stem_b)
        # Two scans of one series: the numbers are the whole difference.
        if head_a and head_a == head_b:
            parts = [head_a, num_a, num_b]
        else:
            parts = [_merge_tokens(stem_a, stem_b)]
        parts.append(leaf_a if leaf_a == leaf_b else f"{leaf_a}_{leaf_b}")
    elif leaf_a != leaf_b:
        parts = [stem_a, leaf_a, leaf_b]
    else:
        parts = [stem_a, leaf_a, col_a, col_b]

    joined = "_".join(p for p in parts if p)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", joined).strip("_") or "export"


def short_series_label(source_key: str | None, fallback: str = "data") -> str:
    """Name one curve as briefly as still tells it apart from the others.

    ``scanx_0083.nxs::scan_0083/scan_data/data_04 [Col 2]`` becomes
    ``scanx_0083_data_04_col2``. Used for legends and column headers alike, so
    a figure and the table behind it name the same curve the same way.
    """
    key = (source_key or "").strip()
    if not key:
        return fallback

    column = ""
    match = _COLUMN_SUFFIX.search(key)
    if match:
        column = f"col{int(match.group(1))}"
        key = key[: match.start()].strip()

    if "::" in key:
        file_part, ds_part = key.rsplit("::", 1)
        parts = [
            pathlib.Path(file_part.strip()).stem,
            ds_part.strip().rstrip("/").split("/")[-1],
        ]
    else:
        parts = [pathlib.Path(key).stem]

    parts.append(column)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(p for p in parts if p)).strip("_")
    return label or fallback


def x_column_header(source_key: str | None, as_q: bool = False) -> str:
    """Header for an X column, named after the X dataset itself.

    ``scanx_0033_actuator_1_1_X``. Several exports used to head the X column
    with the *Y* dataset's name plus ``_X``, which reads as "the X belonging to
    that Y" — it is really a dataset in its own right, often shared by every
    curve in the file.

    The ``_X``/``_q`` ending is load-bearing: :func:`plot_series.series_from_table`
    finds the X columns by it when reading an exported table back. Without a
    named dataset — an index axis, or the X the viewer happens to show — the
    plain ``X``/``q`` is still the honest answer.
    """
    suffix = "q" if as_q else "X"
    label = short_series_label(source_key, fallback="")
    return f"{label}_{suffix}" if label else suffix


def export_stem(source_dataset_key: str | None, fallback: str = "export") -> str:
    """Build a filesystem-safe stem from a ``<file>::<dataset>`` key.

    ``d:/data/scanx_0033.nxs::/scan_0033/scan_data/data_01`` becomes
    ``scanx_0033_data_01``. Anything unusable falls back to ``fallback``.
    """
    key = (source_dataset_key or "").strip()
    if not key:
        return fallback

    if "::" in key:
        file_part, ds_part = key.rsplit("::", 1)
        file_stem = pathlib.Path(file_part.strip()).stem
        ds_leaf = ds_part.strip().rstrip("/").split("/")[-1]
        parts = [p for p in (file_stem, ds_leaf) if p]
    else:
        parts = [pathlib.Path(key).stem]

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts)).strip("_")
    return stem or fallback

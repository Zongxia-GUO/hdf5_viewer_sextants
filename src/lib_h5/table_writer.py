"""One writer for every tabular export in the app.

Quick exports, batch exports and the comparison tool all used to carry their own
``csv.writer`` loop, each with its own hardcoded delimiter and encoding. That is
how the csv2 dialect ended up reaching only the batch path. They now all funnel
through :func:`write_table`, so a dialect added to
:mod:`src.lib_h5.table_format` reaches every export at once.
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

import csv
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.lib_h5.data_exporter import DataExporter
from src.lib_h5.table_format import TableFormat


def format_cell(value: Any, fmt: TableFormat) -> str:
    """Render one cell in the given dialect (decimal mark included)."""
    return str(DataExporter._format_csv_value(value, fmt.decimal))


def write_table(
    path: str | pathlib.Path,
    headers: Sequence[str],
    columns: Sequence[Sequence[Any]],
    fmt: TableFormat,
    comments: Sequence[str] | None = None,
) -> pathlib.Path:
    """Write columns of possibly unequal length as one delimited table.

    :param path: destination file.
    :param headers: one header per column.
    :param columns: the columns; shorter ones are padded with empty cells so a
        ragged set of curves still lines up.
    :param fmt: dialect deciding delimiter, decimal mark and encoding.
    :param comments: optional lines written verbatim above the header. They are
        emitted raw rather than through the csv writer so a delimiter inside a
        comment does not get the line quoted.
    :raises ValueError: if the header count does not match the column count.
    :return: the path written.
    """
    if len(headers) != len(columns):
        raise ValueError(f"{len(headers)} headers for {len(columns)} columns")

    out_path = pathlib.Path(path)
    n_rows = max((len(col) for col in columns), default=0)

    with open(out_path, "w", newline="", encoding=fmt.encoding) as handle:
        for comment in comments or ():
            handle.write(f"{comment}\n")

        writer = csv.writer(handle, delimiter=fmt.delimiter)
        writer.writerow(list(headers))
        for row_idx in range(n_rows):
            writer.writerow(
                [
                    format_cell(col[row_idx], fmt) if row_idx < len(col) else ""
                    for col in columns
                ]
            )

    return out_path


def columns_from_2d(data: np.ndarray) -> list[np.ndarray]:
    """Split a rows x columns array into a list of column arrays."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return [arr]
    return [arr[:, idx] for idx in range(arr.shape[1])]

"""Explicit dialects for tabular text export.

Excel does not read the delimiter from the file — it uses the machine's regional
"list separator". A comma file written here opens as a single column on a French
or German Windows, and its ``1.5`` values are not even recognised as numbers
there (those locales expect ``1,5``).

Rather than guess what machine the file will be opened on, this module follows
R's approach and makes each dialect an explicit, named choice, the way
``write.csv`` and ``write.csv2`` do:

    txt   tab       ``.``   the default: unambiguous in every locale
    csv   comma     ``.``   R's write.csv  — en/US Excel, pandas, numpy
    csv2  semicolon ``,``   R's write.csv2 — fr/de/es/it/... Excel

Keeping the pair (delimiter, decimal) together matters: a semicolon file with
``.`` decimals still lands as text in a French Excel, so the two settings are
never exposed separately.
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

from dataclasses import dataclass


@dataclass(frozen=True)
class TableFormat:
    """One tabular text dialect: what separates fields, and what marks decimals."""

    key: str
    label: str
    suffix: str
    delimiter: str
    decimal: str
    encoding: str

    def __post_init__(self) -> None:
        """Reject a dialect whose delimiter and decimal mark collide."""
        if self.delimiter == self.decimal:
            raise ValueError(
                f"Table format '{self.key}' is unusable: delimiter and decimal "
                f"mark are both {self.delimiter!r}."
            )


# The default is tab-separated text. A tab cannot collide with any locale's
# decimal mark, and double-clicking a .txt opens Excel's import wizard instead of
# silently mis-parsing the file.
TABLE_FORMATS: tuple[TableFormat, ...] = (
    TableFormat(
        key="txt",
        label="TXT (tab)",
        suffix=".txt",
        delimiter="\t",
        decimal=".",
        # No BOM: a .txt is usually read by scripts, where a leading BOM
        # becomes part of the first field.
        encoding="utf-8",
    ),
    TableFormat(
        key="csv",
        label="CSV (comma / dot)",
        suffix=".csv",
        delimiter=",",
        decimal=".",
        # BOM so Excel detects UTF-8 rather than the ANSI codepage.
        encoding="utf-8-sig",
    ),
    TableFormat(
        key="csv2",
        label="CSV2 (semicolon / comma)",
        suffix=".csv",
        delimiter=";",
        decimal=",",
        encoding="utf-8-sig",
    ),
)

DEFAULT_TABLE_FORMAT_KEY = "txt"

_BY_KEY = {fmt.key: fmt for fmt in TABLE_FORMATS}
_BY_LABEL = {fmt.label: fmt for fmt in TABLE_FORMATS}


def get_table_format(key_or_label: str | None) -> TableFormat:
    """Resolve a format by key or by the label shown in the UI.

    Falls back to the default rather than raising, so a stale value in
    ``QSettings`` can never block an export.
    """
    if key_or_label:
        found = _BY_KEY.get(key_or_label) or _BY_LABEL.get(key_or_label)
        if found is not None:
            return found
    return _BY_KEY[DEFAULT_TABLE_FORMAT_KEY]


def format_labels() -> list[str]:
    """Labels for a combo box, in declaration order (default first)."""
    return [fmt.label for fmt in TABLE_FORMATS]


def dialog_filter(fmt: TableFormat) -> str:
    """The save-dialog filter entry for one dialect."""
    return f"{fmt.label} (*{fmt.suffix})"


def save_dialog_filter(preferred: TableFormat | None = None) -> str:
    """Build the full ``;;``-joined save-dialog filter string.

    ``preferred`` is listed first so ``QFileDialog`` preselects it; pass the
    dialect the user chose last.
    """
    ordered = list(TABLE_FORMATS)
    if preferred is not None:
        ordered.sort(key=lambda f: f.key != preferred.key)
    return ";;".join(dialog_filter(fmt) for fmt in ordered)


def format_from_filter(selected_filter: str | None) -> TableFormat:
    """Resolve the dialect the user picked in the save dialog.

    Matching is on the label, not the extension: csv and csv2 both write ``.csv``,
    so the suffix alone cannot tell them apart.
    """
    text = (selected_filter or "").strip()
    for fmt in TABLE_FORMATS:
        if text.startswith(fmt.label):
            return fmt
    return get_table_format(DEFAULT_TABLE_FORMAT_KEY)

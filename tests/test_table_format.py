"""Tests for the explicit tabular export dialects (R-style write.csv / write.csv2)."""

import numpy as np
import pytest

from src.lib_h5.data_exporter import DataExporter
from src.lib_h5.table_format import (
    DEFAULT_TABLE_FORMAT_KEY,
    TABLE_FORMATS,
    TableFormat,
    format_labels,
    get_table_format,
)


def test_default_is_tab_separated_text():
    """Tab is the safe default: it cannot collide with any locale's decimal mark."""
    assert DEFAULT_TABLE_FORMAT_KEY == "txt"
    fmt = get_table_format(None)
    assert (fmt.key, fmt.suffix, fmt.delimiter, fmt.decimal) == ("txt", ".txt", "\t", ".")


def test_csv_dialect_matches_r_write_csv():
    fmt = get_table_format("csv")
    assert (fmt.suffix, fmt.delimiter, fmt.decimal) == (".csv", ",", ".")


def test_csv2_dialect_matches_r_write_csv2():
    fmt = get_table_format("csv2")
    assert (fmt.suffix, fmt.delimiter, fmt.decimal) == (".csv", ";", ",")


def test_only_csv_dialects_carry_a_bom():
    """Excel needs the BOM to detect UTF-8; a .txt read by scripts must not have one."""
    assert get_table_format("txt").encoding == "utf-8"
    assert get_table_format("csv").encoding == "utf-8-sig"
    assert get_table_format("csv2").encoding == "utf-8-sig"


def test_no_dialect_has_a_colliding_delimiter_and_decimal():
    for fmt in TABLE_FORMATS:
        assert fmt.delimiter != fmt.decimal


def test_colliding_dialect_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unusable"):
        TableFormat("bad", "bad", ".csv", ",", ",", "utf-8")


def test_lookup_by_label_and_fallback_on_garbage():
    labels = format_labels()
    assert len(labels) == len(TABLE_FORMATS)
    assert get_table_format(labels[1]).key == TABLE_FORMATS[1].key
    # A stale QSettings value must not block an export.
    assert get_table_format("no-such-format").key == DEFAULT_TABLE_FORMAT_KEY


# ---------------------------------------------------------------------------
# Decimal mark handling
# ---------------------------------------------------------------------------

def test_default_decimal_is_unchanged():
    assert DataExporter._format_csv_value(1.5) == "1.5"
    assert DataExporter._format_csv_value(np.float64(0.125)) == "0.125"


def test_comma_decimal_is_emitted_for_csv2():
    assert DataExporter._format_csv_value(1.5, ",") == "1,5"
    assert DataExporter._format_csv_value(np.float64(-0.125), ",") == "-0,125"


def test_integers_and_specials_are_unaffected_by_decimal():
    assert DataExporter._format_csv_value(7, ",") == "7"
    assert DataExporter._format_csv_value(float("nan"), ",") == "NaN"
    assert DataExporter._format_csv_value(float("inf"), ",") == "Inf"


def test_strings_are_not_rewritten():
    """A text cell containing a dot must survive the csv2 dialect intact."""
    assert DataExporter._format_csv_value("scan_0080.h5", ",") == "scan_0080.h5"

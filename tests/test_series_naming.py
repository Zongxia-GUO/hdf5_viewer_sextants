"""One convention for naming a curve, wherever the name is written down.

A legend entry, a column header and an export file name all start from the same
``<file>::<dataset>`` key. Spelled out in full it is unusable — a legend of
``scanx_0083.nxs::scan_0083/scan_data/data_04 [Col 2]`` covers the plot it is
labelling. These pin the short form and, more importantly, that every place
uses the *same* short form: a figure and the table behind it must call one
curve by one name.
"""

import numpy as np
import pytest

from src.gui.export_naming import export_stem, short_series_label

KEY = "C:/data/scanx_0083.nxs::scan_0083/scan_data/data_04"


# ---------------------------------------------------------------------------
# The convention
# ---------------------------------------------------------------------------

def test_a_key_becomes_scan_plus_dataset():
    assert short_series_label(KEY) == "scanx_0083_data_04"


def test_the_column_is_kept_when_one_was_picked():
    assert short_series_label(f"{KEY} [Col 2]") == "scanx_0083_data_04_col2"


def test_no_column_suffix_when_none_was_picked():
    assert not short_series_label(KEY).endswith("col1")


@pytest.mark.parametrize("written", ["[Col 3]", "[col 3]", "  [Col 3]  "])
def test_the_column_marker_is_read_however_it_was_written(written):
    assert short_series_label(f"{KEY} {written}") == "scanx_0083_data_04_col3"


def test_a_bare_path_still_gives_something_usable():
    assert short_series_label("C:/data/scanx_0083.nxs") == "scanx_0083"


def test_nothing_usable_falls_back():
    assert short_series_label("") == "data"
    assert short_series_label(None, fallback="curve") == "curve"


def test_the_label_is_safe_as_a_file_name_and_a_column_header():
    label = short_series_label("C:/d/scan 83 (raw).nxs::a b/c+d")

    assert " " not in label and "+" not in label and "(" not in label


def test_the_label_and_the_export_stem_agree():
    """They are the same idea; a file called one thing and a column another is
    the sort of mismatch that costs an afternoon."""
    assert export_stem(KEY) == short_series_label(KEY)


# ---------------------------------------------------------------------------
# Everywhere that names a curve uses it
# ---------------------------------------------------------------------------

def test_the_1d_viewer_labels_match_the_convention(qapp):
    from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

    assert PlotWidget1DEnhanced._short_key_label(KEY) == "scanx_0083_data_04"
    assert PlotWidget1DEnhanced._short_key_label(None) == ""


def comparison_with(qapp, names):
    from src.gui.data_comparison import CurveEntry, DataComparisonTool

    tool = DataComparisonTool(tuple())
    tool.datasets = [CurveEntry(name=name, data=np.arange(5.0)) for name in names]
    return tool


def test_comparison_headers_are_short(qapp):
    tool = comparison_with(qapp, [KEY, f"{KEY.replace('0083', '0084')} [Col 2]"])

    headers, _columns, _comments = tool.build_export_table()

    assert headers == ["scanx_0083_data_04_Y", "scanx_0084_data_04_col2_Y"]


def test_comparison_legend_uses_the_same_names_as_the_headers(qapp):
    tool = comparison_with(qapp, [KEY, KEY.replace("0083", "0084")])

    headers, _columns, _comments = tool.build_export_table()
    legend = tool._unique_series_labels()

    assert [h[: -len("_Y")] for h in headers] == legend


def test_a_repeated_curve_is_numbered_apart(qapp):
    """Shortening can make two rows agree; a repeated header would make the
    exported table ambiguous."""
    tool = comparison_with(qapp, [KEY, KEY, KEY])

    labels = tool._unique_series_labels()

    assert labels == ["scanx_0083_data_04", "scanx_0083_data_04_2", "scanx_0083_data_04_3"]
    assert len(set(labels)) == 3


def test_per_curve_x_headers_are_short_too(qapp, monkeypatch):
    """The un-shared branch writes an X column per curve; it was the longest."""
    tool = comparison_with(qapp, [KEY, KEY.replace("0083", "0084")])
    monkeypatch.setattr(tool, "_is_shared_xq_compatible", lambda: (False, "lengths differ"))
    for entry in tool.datasets:
        entry.x_data = np.arange(5.0)

    headers, _columns, _comments = tool.build_export_table()

    assert headers == [
        "scanx_0083_data_04_X", "scanx_0083_data_04_Y",
        "scanx_0084_data_04_X", "scanx_0084_data_04_Y",
    ]


# ---------------------------------------------------------------------------
# The X column is named after the X dataset, not the Y beside it
# ---------------------------------------------------------------------------

def test_an_x_column_is_named_after_its_own_dataset():
    from src.gui.export_naming import x_column_header

    assert x_column_header("scanx_0033.h5::scan/actuator_1_1") == "scanx_0033_actuator_1_1_X"


def test_a_q_column_says_so():
    from src.gui.export_naming import x_column_header

    assert x_column_header("scanx_0033.h5::scan/theta", as_q=True) == "scanx_0033_theta_q"


def test_without_a_named_dataset_the_plain_letter_is_the_honest_answer():
    from src.gui.export_naming import x_column_header

    assert x_column_header(None) == "X"
    assert x_column_header("", as_q=True) == "q"


def test_the_ending_is_what_the_plot_reads_back():
    """series_from_table finds X columns by the _X/_q ending; renaming without
    it would turn every X column into a curve."""
    from src.gui.export_naming import x_column_header
    from src.gui.plot_series import PER_CURVE_X_SUFFIXES, series_from_table

    header = x_column_header("scanx_0033.h5::scan/actuator_1_1")
    assert any(header.endswith(s) for s in PER_CURVE_X_SUFFIXES)

    series = series_from_table([header, "a_Y"], [[0.5, 1.5], [1.0, 2.0]])
    assert [s.label for s in series] == ["a_Y"], "the X column is not a curve"
    np.testing.assert_allclose(series[0].x, [0.5, 1.5])


def test_the_comparison_x_header_names_the_x_dataset(qapp):
    tool = comparison_with(qapp, [KEY])
    tool._on_x_data_selected(np.array([0.5, 1.5, 2.5, 3.5, 4.5]), "scanx_0033.h5::scan/setpoint")

    headers, _columns, _comments = tool.build_export_table()

    assert headers[0] == "scanx_0033_setpoint_X"
    assert "data_04" not in headers[0], "the X is not named after a Y curve"


def test_each_curve_keeps_its_own_x_name_when_they_differ(qapp):
    tool = comparison_with(qapp, [KEY, KEY.replace("0083", "0084")])
    tool._x_selection_target_row = 0
    tool._on_x_data_selected(np.arange(5.0), "scanx_0083.h5::scan/motor_a")
    tool._x_selection_target_row = 1
    tool._on_x_data_selected(np.arange(5.0) * 2, "scanx_0084.h5::scan/motor_b")

    headers, _columns, _comments = tool.build_export_table()

    assert headers == [
        "scanx_0083_motor_a_X", "scanx_0083_data_04_Y",
        "scanx_0084_motor_b_X", "scanx_0084_data_04_Y",
    ]


def test_a_row_without_an_x_dataset_still_says_whose_axis_it_is(qapp):
    """Nothing better can be said about it than which curve it belongs to."""
    tool = comparison_with(qapp, [KEY, KEY.replace("0083", "0084")])
    tool._x_selection_target_row = 0
    tool._on_x_data_selected(np.arange(5.0), "scanx_0083.h5::scan/motor_a")

    headers, _columns, _comments = tool.build_export_table()

    assert "scanx_0083_motor_a_X" in headers


def test_no_header_carries_a_full_path(qapp):
    tool = comparison_with(qapp, [KEY, f"{KEY} [Col 2]"])

    headers, _columns, _comments = tool.build_export_table()

    for header in headers:
        assert "::" not in header and "/" not in header
        assert len(header) < 40, header

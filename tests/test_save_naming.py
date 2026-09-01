"""One naming rule and one remembered folder for every save in the application.

The name says what the data **is** — scan and dataset — plus what was **done**
to it, and the second part only when something was. A faithful copy of one
dataset needs no tag; a calculator result or one of four reconstructed
components would be ambiguous without one. That is what keeps the rule from
becoming a suffix on everything.

The folder is one shared setting: the user thinks "where did I put things last
time", not "where did I put my PNGs".
"""

import pathlib

import numpy as np
import pytest
from PyQt6.QtCore import QSettings

from src.gui.export_naming import (
    last_save_directory,
    pair_label,
    remember_save_directory,
    suggested_save_path,
)


@pytest.fixture(autouse=True)
def clean_settings():
    QSettings().remove("paths/last_export_directory")
    yield
    QSettings().remove("paths/last_export_directory")


# ---------------------------------------------------------------------------
# The remembered folder
# ---------------------------------------------------------------------------

def test_without_a_previous_save_it_offers_home():
    assert last_save_directory() == pathlib.Path.home()


def test_a_save_is_remembered_for_the_next_one(tmp_path):
    remember_save_directory(tmp_path / "curve.txt")

    assert last_save_directory() == tmp_path


def test_a_folder_can_be_remembered_directly(tmp_path):
    remember_save_directory(tmp_path)

    assert last_save_directory() == tmp_path


def test_a_folder_that_has_gone_away_falls_back(tmp_path):
    """A removable drive should not leave the dialog opening on nothing."""
    gone = tmp_path / "usb"
    gone.mkdir()
    remember_save_directory(gone)
    gone.rmdir()

    assert last_save_directory() == pathlib.Path.home()


def test_one_folder_is_shared_by_every_kind_of_save(tmp_path):
    """Not split per file type: there is one "last place I saved something"."""
    remember_save_directory(tmp_path / "figure.png")

    assert suggested_save_path("curve", extension=".txt").startswith(str(tmp_path))
    assert suggested_save_path("frame", extension=".png").startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# The suggested name
# ---------------------------------------------------------------------------

def test_a_faithful_copy_carries_no_tag(tmp_path):
    remember_save_directory(tmp_path)

    assert suggested_save_path("scanx_0083_data_04", extension=".txt") == str(
        tmp_path / "scanx_0083_data_04.txt"
    )


def test_a_derived_result_carries_what_was_done(tmp_path):
    remember_save_directory(tmp_path)

    assert suggested_save_path("scanx_0083_0085", "diff", ".txt") == str(
        tmp_path / "scanx_0083_0085_diff.txt"
    )


def test_the_extension_is_not_doubled(tmp_path):
    remember_save_directory(tmp_path)

    assert suggested_save_path("plot.png", extension=".png").endswith("plot.png")


def test_the_name_is_always_safe_for_a_filesystem(tmp_path):
    remember_save_directory(tmp_path)

    name = pathlib.Path(suggested_save_path("scan 83 (raw)/x", "a+b", ".txt")).name

    assert " " not in name and "/" not in name and "+" not in name


def test_an_empty_name_still_produces_a_file(tmp_path):
    remember_save_directory(tmp_path)

    assert pathlib.Path(suggested_save_path("", extension=".txt")).name == "export.txt"


# ---------------------------------------------------------------------------
# Two sources are named by what differs between them
# ---------------------------------------------------------------------------

KEY = "d/scanx_0083.nxs::scan_0083/scan_data/data_04"


def test_two_scans_give_the_range():
    assert pair_label(KEY, "d/scanx_0085.nxs::scan_0085/scan_data/data_04") == (
        "scanx_0083_0085_data_04"
    )


def test_two_datasets_of_one_scan_give_both_datasets():
    assert pair_label(KEY, "d/scanx_0083.nxs::scan_0083/scan_data/data_10") == (
        "scanx_0083_data_04_data_10"
    )


def test_two_columns_of_one_dataset_give_both_columns():
    assert pair_label(f"{KEY} [Col 1]", f"{KEY} [Col 2]") == "scanx_0083_data_04_col1_col2"


def test_both_differences_are_kept_when_both_differ():
    assert pair_label(KEY, "d/scanx_0085.nxs::scan_0085/scan_data/data_10") == (
        "scanx_0083_0085_data_04_data_10"
    )


def test_one_source_is_named_on_its_own():
    assert pair_label(KEY, None) == "scanx_0083_data_04"
    assert pair_label(KEY, KEY) == "scanx_0083_data_04", "no point repeating it"


def test_unrelated_files_keep_both_stems():
    assert pair_label("d/alpha.h5::a/x", "d/beta.h5::b/y") == "alpha_beta_x_y"


# ---------------------------------------------------------------------------
# What the tools actually offer
# ---------------------------------------------------------------------------

def test_the_calculator_names_the_result_after_its_operands(qapp, tmp_path):
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc._last_operation_expr = "A - B"
    calc._operand_key = lambda combo: (
        KEY if combo is calc.combo_dataset_a else KEY.replace("0083", "0085")
    )

    assert calc._default_export_base_name() == "scanx_0083_0085_data_04_diff"


def test_the_comparison_names_the_range_it_spans(qapp):
    from src.gui.data_comparison import CurveEntry, DataComparisonTool

    tool = DataComparisonTool(tuple())
    tool.datasets = [
        CurveEntry(name=KEY, data=np.arange(3.0)),
        CurveEntry(name=KEY.replace("0083", "0085"), data=np.arange(3.0)),
    ]

    assert tool._default_export_base_name() == "scanx_0083_0085_data_04_compar"


def test_the_plot_window_names_the_figure_after_its_curves(qapp):
    from src.gui.plot_dialog import PlotPanel
    from src.gui.plot_series import Series

    one = PlotPanel(series=[Series("scanx_0083_data_04", np.arange(5.0))])
    one.close()
    assert one.figure_name() == "scanx_0083_data_04"

    many = PlotPanel(series=[
        Series("scanx_0083_data_04", np.arange(5.0)),
        Series("scanx_0084_data_04", np.arange(5.0)),
        Series("scanx_0085_data_04", np.arange(5.0)),
    ])
    many.close()
    assert many.figure_name() == "scanx_0083_0085_data_04", "first and last, not all three"


def test_an_empty_plot_still_has_a_name(qapp):
    from src.gui.plot_dialog import PlotPanel

    panel = PlotPanel(series=[])
    panel.close()

    assert panel.figure_name() == "plot"


def test_every_save_dialog_remembers_where_it_went():
    """A grep, because a new save site forgetting this is invisible until used."""
    gui = pathlib.Path(__file__).resolve().parent.parent / "src" / "gui"
    offenders = []
    for path in sorted(gui.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "getSaveFileName" not in text:
            continue
        if "remember_save_directory" not in text:
            offenders.append(path.name)

    assert offenders == [], "call remember_save_directory() with the chosen path"

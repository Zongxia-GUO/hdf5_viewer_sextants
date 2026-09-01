"""Tests for the plot Series abstraction and the categorical palettes."""

import numpy as np
import pytest

from src.gui.plot_palettes import (
    CVD_FLOOR,
    CVD_TARGET,
    DEFAULT_PALETTE_KEY,
    LINE_STYLES,
    PALETTES,
    get_palette,
    palette_from_label,
    palette_labels,
    style_for,
    styles_for,
)
from src.gui.plot_series import (
    Series,
    common_axis_labels,
    data_bounds,
    positive_only,
    series_from_columns,
    series_from_table,
)


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

def test_series_without_x_uses_the_sample_index():
    x, y = Series("a", np.array([5.0, 6.0, 7.0])).points()

    np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y, [5.0, 6.0, 7.0])


def test_series_with_x_pairs_them():
    x, y = Series("a", np.array([5.0, 6.0]), np.array([10.0, 20.0])).points()

    np.testing.assert_allclose(x, [10.0, 20.0])
    np.testing.assert_allclose(y, [5.0, 6.0])


def test_mismatched_lengths_are_trimmed_not_raised():
    """A ragged pair must not blow up inside the plotting call."""
    x, y = Series("a", np.arange(5.0), np.arange(3.0)).points()

    assert x.size == y.size == 3


def test_finite_points_drops_nan_pairs():
    s = Series("a", np.array([1.0, np.nan, 3.0]), np.array([10.0, 20.0, np.nan]))
    x, y = s.finite_points()

    np.testing.assert_allclose(x, [10.0])
    np.testing.assert_allclose(y, [1.0])


# ---------------------------------------------------------------------------
# Adapting export tables
# ---------------------------------------------------------------------------

def test_shared_x_table_becomes_curves_against_that_x():
    headers = ["X", "a_Y", "b_Y"]
    columns = [[0.5, 1.5], [1.0, 2.0], [10.0, 20.0]]

    series = series_from_table(headers, columns)

    assert [s.label for s in series] == ["a_Y", "b_Y"]
    for s in series:
        np.testing.assert_allclose(s.x, [0.5, 1.5])


def test_per_curve_x_pairs_each_with_the_column_before_it():
    headers = ["a_X", "a_Y", "b_X", "b_Y"]
    columns = [[0.5, 1.5], [1.0, 2.0], [9.0, 8.0], [10.0, 20.0]]

    series = series_from_table(headers, columns)

    assert [s.label for s in series] == ["a_Y", "b_Y"]
    np.testing.assert_allclose(series[0].x, [0.5, 1.5])
    np.testing.assert_allclose(series[1].x, [9.0, 8.0])


def test_a_per_curve_x_covers_every_y_of_its_group():
    """A two-column dataset exports as X, Y_1, Y_2 — both Y belong to that X."""
    headers = ["scan_X", "scan_Y_1", "scan_Y_2"]
    columns = [[0.5, 1.5], [1.0, 2.0], [10.0, 20.0]]

    series = series_from_table(headers, columns)

    assert [s.label for s in series] == ["scan_Y_1", "scan_Y_2"]
    for s in series:
        np.testing.assert_allclose(s.x, [0.5, 1.5])


def test_each_group_keeps_its_own_x_when_several_follow():
    headers = ["a_X", "a_Y_1", "a_Y_2", "b_X", "b_Y_1", "b_Y_2"]
    columns = [[0.5, 1.5], [1.0, 2.0], [3.0, 4.0], [9.0, 8.0], [5.0, 6.0], [7.0, 8.0]]

    series = series_from_table(headers, columns)

    assert [s.label for s in series] == ["a_Y_1", "a_Y_2", "b_Y_1", "b_Y_2"]
    for s in series[:2]:
        np.testing.assert_allclose(s.x, [0.5, 1.5])
    for s in series[2:]:
        np.testing.assert_allclose(s.x, [9.0, 8.0])


def test_a_q_column_counts_as_the_shared_x():
    series = series_from_table(["q", "a_Y"], [[0.1, 0.2], [1.0, 2.0]])
    np.testing.assert_allclose(series[0].x, [0.1, 0.2])


def test_table_without_any_x_gives_index_plotted_curves():
    series = series_from_table(["a_Y", "b_Y"], [[1.0, 2.0], [3.0, 4.0]])

    assert len(series) == 2
    assert all(s.x is None for s in series)


def test_calculator_style_table_keeps_operands_and_result():
    headers = ["X", "Data_A", "Data_B", "Result"]
    columns = [[0.5, 1.5], [1.0, 2.0], [10.0, 20.0], [11.0, 22.0]]

    series = series_from_table(headers, columns)
    assert [s.label for s in series] == ["Data_A", "Data_B", "Result"]


# ---------------------------------------------------------------------------
# Column blocks
# ---------------------------------------------------------------------------

def test_single_column_block_keeps_the_plain_label():
    series = series_from_columns("curve", np.array([[1.0], [2.0]]))
    assert [s.label for s in series] == ["curve"]


def test_multi_column_block_numbers_the_labels():
    series = series_from_columns("curve", np.zeros((3, 2)))
    assert [s.label for s in series] == ["curve_1", "curve_2"]


# ---------------------------------------------------------------------------
# Helpers the plot page needs
# ---------------------------------------------------------------------------

def test_bounds_span_every_curve():
    series = [
        Series("a", np.array([1.0, 5.0]), np.array([0.0, 10.0])),
        Series("b", np.array([-2.0, 3.0]), np.array([2.0, 20.0])),
    ]
    assert data_bounds(series) == (0.0, 20.0, -2.0, 5.0)


def test_bounds_ignore_nan_and_empty():
    assert data_bounds([]) is None
    assert data_bounds([Series("a", np.array([np.nan]))]) is None


def test_positive_only_gates_the_log_axis():
    assert positive_only([Series("a", np.array([1.0, 2.0]), np.array([1.0, 2.0]))])
    assert not positive_only([Series("a", np.array([0.0, 2.0]), np.array([1.0, 2.0]))])


def test_axis_labels_name_a_lone_curve():
    assert common_axis_labels([Series("counts", np.zeros(3))]) == ("Index", "counts")
    two = [Series("a", np.zeros(3), np.zeros(3)), Series("b", np.zeros(3), np.zeros(3))]
    assert common_axis_labels(two) == ("X", "Value")


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

DEFAULT = get_palette(DEFAULT_PALETTE_KEY)


def test_default_palette_is_set1():
    assert DEFAULT_PALETTE_KEY == "Set1"
    assert get_palette(None).key == "Set1"


def test_the_frozen_hexes_match_matplotlibs_own_values():
    """These are copies, so they must not drift from the real thing."""
    assert DEFAULT.colors[:3] == ("#e41a1c", "#377eb8", "#4daf4a")
    assert len(DEFAULT) == 9
    assert get_palette("tab10").colors[:3] == ("#1f77b4", "#ff7f0e", "#2ca02c")
    assert len(get_palette("tab10")) == 10


def test_unknown_palette_falls_back_to_the_default():
    assert get_palette("no-such-palette").key == DEFAULT_PALETTE_KEY
    assert palette_from_label(None).key == DEFAULT_PALETTE_KEY


def test_every_requested_palette_is_offered():
    keys = {p.key for p in PALETTES}
    assert keys == {
        "tab10", "tab20", "tab20b", "tab20c", "okabe_ito", "Paired", "Accent",
        "Dark2", "Set1", "Set2", "Set3", "Pastel1", "Pastel2",
    }
    assert palette_labels()[0].startswith("Set1"), "the default sits at the top of the picker"


def test_colours_are_assigned_in_fixed_slot_order():
    """Slot order is what the separation figure was measured on."""
    assert [style_for(i, DEFAULT)[0] for i in range(3)] == ["#e41a1c", "#377eb8", "#4daf4a"]


def test_an_extra_curve_reuses_a_hue_with_a_new_line_style():
    """Past the palette we add a second channel rather than invent a colour."""
    extra = style_for(len(DEFAULT), DEFAULT)

    assert extra[0] == DEFAULT.colors[0]
    assert extra[1] == LINE_STYLES[1]
    assert extra[1] != style_for(0, DEFAULT)[1]


def test_styles_for_is_unique_across_one_full_cycle():
    pairs = styles_for(len(DEFAULT) * len(LINE_STYLES), DEFAULT)
    assert len(set(pairs)) == len(pairs)


def test_styles_for_handles_zero_and_negative():
    assert styles_for(0, DEFAULT) == []
    assert styles_for(-3, DEFAULT) == []


# ---------------------------------------------------------------------------
# The measured separation figure surfaced in the picker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("palette", PALETTES)
def test_every_palette_carries_a_measured_separation(palette):
    assert palette.cvd_delta_e >= 0
    assert palette.cvd_verdict in {"good", "marginal", "poor"}
    assert f"{palette.cvd_delta_e:.1f}" in palette.tooltip()


def test_verdict_bands_match_the_thresholds():
    good = [p for p in PALETTES if p.cvd_delta_e >= CVD_TARGET]
    poor = [p for p in PALETTES if p.cvd_delta_e < CVD_FLOOR]

    assert all(p.cvd_verdict == "good" for p in good)
    assert all(p.cvd_verdict == "poor" for p in poor)


def test_the_measured_figures_are_the_ones_that_were_validated():
    """Frozen so an edit to a palette cannot silently keep a stale score."""
    measured = {p.key: p.cvd_delta_e for p in PALETTES}
    assert measured["Set1"] == 5.9
    assert measured["tab10"] == 0.7
    assert measured["okabe_ito"] == 15.8
    assert measured["Paired"] == 11.5


def test_the_default_still_reports_its_measured_separation():
    """The default is the user's call; the tooltip must not hide what it scores."""
    assert DEFAULT.cvd_delta_e < CVD_TARGET
    assert "colour-blind" in DEFAULT.tooltip()


@pytest.mark.parametrize("palette", PALETTES)
def test_every_palette_is_non_empty_and_hex(palette):
    assert palette.colors
    for color in palette.colors:
        assert color.startswith("#") and len(color) == 7


@pytest.mark.parametrize("palette", PALETTES)
def test_no_palette_repeats_a_colour(palette):
    assert len(set(palette.colors)) == len(palette.colors)

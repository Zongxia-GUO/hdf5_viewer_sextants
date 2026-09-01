"""The export UI convention: quick actions are icons, full exports say "Export".

Quick save/copy belong in the viewer (and the tree context menu); a tool's own
settings panel carries only the full-export button.
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QPushButton

from src.gui._shared import quick_icon_button
from src.gui.data_calculator_enhanced import DataCalculatorEnhanced
from src.gui.data_comparison import DataComparisonTool
from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced


def _icon_buttons(widget):
    return [b for b in widget.findChildren(QPushButton) if not b.text().strip() and not b.icon().isNull()]


def _texts(widget):
    return [b.text().strip() for b in widget.findChildren(QPushButton) if b.text().strip()]


# ---------------------------------------------------------------------------
# The shared button factory
# ---------------------------------------------------------------------------

def test_quick_icon_button_has_no_text_and_a_tooltip(qapp):
    button = quick_icon_button("copy.ico", "Quick copy: something")
    assert button.text() == ""
    assert not button.icon().isNull()
    assert button.toolTip() == "Quick copy: something"
    assert not button.autoDefault()


# ---------------------------------------------------------------------------
# Viewers carry the quick icons
# ---------------------------------------------------------------------------

def test_plot_viewer_offers_copy_save_and_plot(qapp):
    """Three quick actions: copy the picture, save the data, plot it.

    The old third button was a second export and was removed; this one opens a
    Plot window, which nothing else in the viewer does.
    """
    w = PlotWidget1DEnhanced()
    w.set_data(np.arange(10, dtype=float))

    assert len(_icon_buttons(w)) == 3
    for btn in (w.btn_copy_plot, w.btn_save_plot, w.btn_plot):
        assert btn.text() == "" and not btn.icon().isNull()
    assert not hasattr(w, "btn_save_plot_image")


def test_the_viewer_x_control_is_one_button_called_set_x(qapp, tmp_path):
    """A bare "Select" next to a label said nothing about what it selects."""
    w = PlotWidget1DEnhanced(opened_files=(tmp_path / "f.h5",))
    w.set_data(np.arange(10, dtype=float))

    assert w.btn_select_x.text() == "Set X"
    assert "X Axis:" not in _texts(w)


def _toolbar(widget):
    return widget.layout().itemAt(0).layout()


def test_the_viewer_toolbar_gaps_are_all_the_same(qapp, tmp_path):
    """Every boundary between tools must measure the same, whatever each holds."""
    from src.gui._shared import TOOLBAR_GROUP_GAP, TOOLBAR_ITEM_GAP

    viewer = PlotWidget1DEnhanced(opened_files=(tmp_path / "f.h5",))
    bar = _toolbar(viewer)
    assert bar.spacing() == TOOLBAR_ITEM_GAP

    spacers = [
        bar.itemAt(i).spacerItem()
        for i in range(bar.count())
        if bar.itemAt(i).spacerItem() is not None
    ]
    # The trailing stretch is not a group gap.
    gaps = [s.sizeHint().width() for s in spacers if s.sizeHint().width() > 0]

    assert len(gaps) >= 4, "one gap per tool boundary"
    assert len(set(gaps)) == 1, f"boundaries differ: {gaps}"
    # The layout adds its own spacing on either side of the spacer.
    assert gaps[0] + 2 * TOOLBAR_ITEM_GAP == TOOLBAR_GROUP_GAP


def test_a_label_hugs_the_control_it_names(qapp, tmp_path):
    """Tighter than the gap to the next control, so the pair reads as one tool."""
    from src.gui._shared import TOOLBAR_ITEM_GAP, TOOLBAR_LABEL_GAP

    viewer = PlotWidget1DEnhanced(opened_files=(tmp_path / "f.h5",))
    bar = _toolbar(viewer)
    pairs = [
        bar.itemAt(i).layout()
        for i in range(bar.count())
        if bar.itemAt(i).layout() is not None
    ]

    assert pairs, "labels are bound to their controls, not dropped in loose"
    assert all(p.spacing() == TOOLBAR_LABEL_GAP for p in pairs)
    assert TOOLBAR_LABEL_GAP < TOOLBAR_ITEM_GAP


def test_image_viewer_offers_copy_and_save_but_no_plot(qapp):
    """The plot window draws curves, so a 2D viewer has nothing to send it."""
    w = ImageView2DEnhanced()
    w.set_data(np.random.RandomState(0).rand(16, 16))

    for btn in (w.btn_copy_image, w.btn_save_image):
        assert btn.text() == "" and not btn.icon().isNull()
    assert not hasattr(w, "btn_plot")


# ---------------------------------------------------------------------------
# Tools: full export is a text button, quick lives in the viewer
# ---------------------------------------------------------------------------

def test_calculator_full_export_is_labelled_export(qapp):
    calc = DataCalculatorEnhanced(tuple())
    assert calc.btn_export.text() == "Export"


def test_calculator_has_no_image_buttons_in_its_settings_panel(qapp):
    """They moved into the result viewer, so the old duplicates must be gone."""
    calc = DataCalculatorEnhanced(tuple())

    assert not hasattr(calc, "btn_save_image")
    assert not hasattr(calc, "btn_copy_image")
    assert "Save Image..." not in _texts(calc)
    assert "Copy Image" not in _texts(calc)


def test_the_calculator_shows_its_viewer_toolbar_from_the_start(qapp):
    """The toolbar must not appear only after the first calculation."""
    calc = DataCalculatorEnhanced(tuple())
    assert len(_icon_buttons(calc)) == 3

    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()

    assert len(_icon_buttons(calc)) == 3, "reused, not rebuilt"


def test_the_calculator_reuses_one_result_viewer(qapp):
    calc = DataCalculatorEnhanced(tuple())
    first = calc.result_widget

    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()

    assert calc.result_widget is first


def test_comparison_full_export_is_labelled_export(qapp):
    comp = DataComparisonTool(tuple())
    assert comp.btn_export.text() == "Export"


def test_comparison_quick_actions_are_icons_now(qapp):
    """Copy the picture, save the data — the same pair as every other viewer."""
    comp = DataComparisonTool(tuple())

    for btn in (comp.btn_copy_image, comp.btn_save_data, comp.btn_open_plot):
        assert btn.text() == "" and not btn.icon().isNull()
    assert "Save Image..." not in _texts(comp)
    assert "Copy Image" not in _texts(comp)


def test_both_tools_offer_plot_from_the_viewer_and_beside_export(qapp):
    """Two ways in on purpose: the toolbar icon and the text button."""
    comp = DataComparisonTool(tuple())
    calc = DataCalculatorEnhanced(tuple())

    assert comp.btn_open_plot.text() == "" and not comp.btn_open_plot.icon().isNull()
    assert comp.btn_plot.text() == "Plot"
    assert calc.btn_plot.text() == "Plot"
    assert len(_icon_buttons(calc)) == 3, "the calculator's icon comes from its viewer"


# ---------------------------------------------------------------------------
# Plot follows the same split: a text button beside Export, an icon in the viewer
# ---------------------------------------------------------------------------

def test_full_plot_is_a_text_button_beside_export(qapp):
    comp = DataComparisonTool(tuple())
    calc = DataCalculatorEnhanced(tuple())

    assert comp.btn_plot.text() == "Plot"
    assert calc.btn_plot.text() == "Plot"


def test_the_calculator_can_only_plot_once_there_is_a_result(qapp):
    calc = DataCalculatorEnhanced(tuple())
    assert not calc.btn_plot.isEnabled()
    assert calc.plot_series() == []
    assert calc.plot_axes_series() == ([], [])


@pytest.mark.parametrize("expression", ["A - B", "(A - B) / (A + B)"])
def test_a_2d_difference_result_is_shown_diverging(qapp, expression):
    """A - B and the asymmetry both run either side of zero."""
    from src.gui.image_view_2d_enhanced import DIVERGING_COLORMAP

    calc = DataCalculatorEnhanced(tuple())
    rs = np.random.RandomState(0)
    a, b = rs.rand(48, 48) + 1, rs.rand(48, 48) + 1
    calc.result_data = (a - b) if expression == "A - B" else (a - b) / (a + b)

    calc._update_result_display()

    viewer = calc.result_widget.get_current_widget()
    assert viewer.combo_colormap.currentText() == DIVERGING_COLORMAP


def test_a_2d_sum_result_stays_sequential(qapp):
    from src.gui.image_view_2d_enhanced import SEQUENTIAL_COLORMAP

    calc = DataCalculatorEnhanced(tuple())
    rs = np.random.RandomState(0)
    calc.result_data = rs.rand(48, 48) + rs.rand(48, 48)

    calc._update_result_display()

    viewer = calc.result_widget.get_current_widget()
    assert viewer.combo_colormap.currentText() == SEQUENTIAL_COLORMAP


def test_the_calculator_puts_the_result_on_its_own_axis(qapp):
    """Operands and a derived result rarely share a scale."""
    calc = DataCalculatorEnhanced(tuple())
    calc.data_a = np.arange(10, dtype=float) * 1000.0
    calc.data_b = np.arange(10, dtype=float) * 900.0
    calc.result_data = np.full(10, 0.1)

    operands, result = calc.plot_axes_series()

    assert [s.label for s in operands] == ["Data_A", "Data_B"]
    assert [s.label for s in result] == ["Result"]


def test_a_result_with_no_operands_stays_on_one_axis(qapp):
    """A second axis with nothing to compare against is just a worse plot."""
    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(10, dtype=float)

    operands, result = calc.plot_axes_series()

    assert [s.label for s in operands] == ["Result"]
    assert result == []


def test_comparison_plots_exactly_what_it_would_export(qapp):
    """One source for both, so a figure can never disagree with the file."""
    comp = DataComparisonTool(tuple())
    comp.datasets = []

    assert comp.plot_series() == []


# ---------------------------------------------------------------------------
# No stale labels anywhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stale",
    ["Export...", "Export Result...", "Export CSV…", "Exporter CSV", "Save All", "Save all"],
)
def test_old_export_labels_are_gone_from_the_sources(stale):
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "gui"
    hits = [
        path.name
        for path in root.glob("*.py")
        if f'QPushButton("{stale}")' in path.read_text(encoding="utf-8")
    ]
    assert hits == []

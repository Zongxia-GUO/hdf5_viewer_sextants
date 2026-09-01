"""Tests for the Plot window: both shapes, the controls, and figure output."""

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.gui.plot_dialog import MAX_LEGEND_ENTRIES, PlotDialog, open_plot_dialog
from src.gui.plot_palettes import DEFAULT_PALETTE_KEY, get_palette
from src.gui.plot_series import Series


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def curves():
    x = np.linspace(1.0, 10.0, 25)
    return [
        Series("alpha", np.sin(x) + 2.0, x),
        Series("beta", np.cos(x) + 2.0, x),
    ]


def make(app, curves, **kwargs):
    dialog = PlotDialog(series=curves, **kwargs)
    dialog.close()
    return dialog


# ---------------------------------------------------------------------------
# The two shapes
# ---------------------------------------------------------------------------

def test_there_is_one_kind_of_plot_window(app, curves):
    """The quick variant is gone: every entry point gets Data and Plot."""
    dialog = make(app, curves, title="scan_0033")

    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ["Data", "Plot"]
    assert dialog.tabs.currentIndex() == 1, "opens on the plot, not the table"
    assert dialog.windowTitle() == "Plot - scan_0033"
    assert "mode" not in PlotDialog.__init__.__code__.co_varnames


def test_the_window_is_non_modal_and_not_pinned_on_top(app, curves):
    """A plot sits beside the data; it must never block or cover everything."""
    from PyQt6.QtCore import Qt

    dialog = make(app, curves)

    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal
    assert not (dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)


def test_the_data_page_shows_the_plotted_points(app, curves):
    dialog = make(app, curves)
    model = dialog.data_view.model()

    # Shared X, so one X column plus one per curve.
    assert model.columnCount() == 3
    assert model.rowCount() == curves[0].y.size


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_every_series_is_drawn_once(app, curves):
    dialog = make(app, curves)
    assert len(dialog.axes.lines) == len(curves)


def test_a_fresh_window_opens_on_the_default_palette(app, curves):
    """With nothing stored, the picker must show the declared default.

    This is also the guard for the settings leak that made the installed app
    open on whatever colormap the test suite had last selected: the suite now
    writes to a throwaway store (see conftest), so a remembered value here can
    only come from this test session.
    """
    from PyQt6.QtCore import QSettings

    QSettings().remove("plot/palette")

    dialog = make(app, curves)

    assert dialog.current_palette().key == DEFAULT_PALETTE_KEY == "Set1"
    assert dialog.cb_palette.currentIndex() == 0


def test_a_remembered_palette_is_honoured_over_the_default(app, curves):
    from PyQt6.QtCore import QSettings

    QSettings().setValue("plot/palette", "Dark2")
    try:
        assert make(app, curves).current_palette().key == "Dark2"
    finally:
        QSettings().remove("plot/palette")


def test_the_test_suite_cannot_reach_the_real_settings_store(app):
    """The leak that caused it: QSettings must be redirected to a temp file."""
    from PyQt6.QtCore import QSettings

    settings = QSettings()
    assert settings.format() == QSettings.Format.IniFormat
    assert "hdf5-viewer-test-settings-" in settings.fileName()


def test_curves_take_the_palette_in_slot_order(app, curves):
    dialog = make(app, curves)
    palette = dialog.current_palette()

    drawn = [line.get_color() for line in dialog.axes.lines]
    assert drawn == list(palette.colors[: len(curves)])


def test_changing_the_palette_repaints(app, curves):
    dialog = make(app, curves)
    dialog.cb_palette.setCurrentIndex(dialog.cb_palette.findData("okabe_ito"))

    assert dialog.current_palette().key == "okabe_ito"
    assert dialog.axes.lines[0].get_color() == get_palette("okabe_ito").colors[0]


def test_axis_labels_follow_the_fields(app, curves):
    dialog = make(app, curves)
    dialog.le_x_label.setText("Field (mT)")
    dialog.le_y_label.setText("Counts")

    assert dialog.axes.get_xlabel() == "Field (mT)"
    assert dialog.axes.get_ylabel() == "Counts"


def test_a_title_is_only_set_when_typed(app, curves):
    dialog = make(app, curves)
    assert dialog.axes.get_title() == ""

    dialog.le_title.setText("Scan 33")
    assert dialog.axes.get_title() == "Scan 33"


def test_legend_is_on_for_several_curves_and_off_for_one(app, curves):
    assert make(app, curves).chk_legend.isChecked()
    assert not make(app, curves[:1]).chk_legend.isChecked()


def test_curves_are_lines_by_default(app, curves):
    dialog = make(app, curves)

    assert dialog.chk_lines.isChecked()
    assert not dialog.chk_markers.isChecked()
    assert dialog.axes.lines[0].get_linestyle() != "None"
    assert dialog.axes.lines[0].get_marker() in (None, "None", "")


def test_turning_lines_off_leaves_a_scatter(app, curves):
    dialog = make(app, curves)
    dialog.chk_markers.setChecked(True)
    dialog.chk_lines.setChecked(False)

    line = dialog.axes.lines[0]
    assert line.get_linestyle() == "None"
    assert line.get_marker() == "o"


def test_lines_and_markers_together_give_a_marked_line(app, curves):
    dialog = make(app, curves)
    dialog.chk_markers.setChecked(True)

    line = dialog.axes.lines[0]
    assert line.get_linestyle() != "None"
    assert line.get_marker() == "o"


def test_the_last_mark_cannot_be_switched_off(app, curves):
    """With neither lines nor markers the figure would simply be empty."""
    dialog = make(app, curves)

    dialog.chk_lines.setChecked(False)
    assert dialog.chk_markers.isChecked(), "unchecking Lines switches to a scatter"

    dialog.chk_markers.setChecked(False)
    assert dialog.chk_lines.isChecked(), "and back again"

    assert dialog.axes.lines[0].get_linestyle() != "None"


def test_the_grid_box_switches_the_grid_both_ways(app, curves):
    """matplotlib turns the grid back on if style kwargs come with grid(False)."""
    dialog = make(app, curves)
    assert dialog.chk_grid.isChecked()
    assert any(line.get_visible() for line in dialog.axes.get_xgridlines())

    dialog.chk_grid.setChecked(False)
    assert not any(line.get_visible() for line in dialog.axes.get_xgridlines())
    assert not any(line.get_visible() for line in dialog.axes.get_ygridlines())

    dialog.chk_grid.setChecked(True)
    assert any(line.get_visible() for line in dialog.axes.get_xgridlines())


def test_a_huge_legend_is_replaced_by_a_note(app):
    x = np.arange(5.0)
    many = [Series(f"c{i}", x + i, x) for i in range(MAX_LEGEND_ENTRIES + 1)]
    dialog = make(app, many)

    assert dialog.axes.get_legend() is None
    assert "see the Data page" in dialog.axes.get_title()


# ---------------------------------------------------------------------------
# Log axes are gated per axis
# ---------------------------------------------------------------------------

def test_log_is_available_when_the_data_is_positive(app, curves):
    dialog = make(app, curves)
    assert dialog.chk_log_x.isEnabled()
    assert dialog.chk_log_y.isEnabled()

    dialog.chk_log_y.setChecked(True)
    assert dialog.axes.get_yscale() == "log"


def test_a_field_sweep_through_zero_keeps_log_y(app):
    """X crosses zero and Y does not, so only the X switch is taken away."""
    x = np.linspace(-7000.0, 7000.0, 50)
    dialog = make(app, [Series("counts", np.abs(x) + 1.0, x)])

    assert not dialog.chk_log_x.isEnabled()
    assert "<= 0" in dialog.chk_log_x.toolTip()
    assert dialog.chk_log_y.isEnabled()


# ---------------------------------------------------------------------------
# A second Y axis (the calculator's result beside its operands)
# ---------------------------------------------------------------------------

@pytest.fixture
def operands():
    x = np.linspace(1.0, 10.0, 25)
    return [Series("Data_A", x * 1000.0, x), Series("Data_B", x * 900.0, x)]


@pytest.fixture
def result():
    x = np.linspace(1.0, 10.0, 25)
    return [Series("Result", np.full_like(x, 0.1), x)]


def test_no_second_axis_unless_one_is_asked_for(app, curves):
    dialog = make(app, curves)
    assert dialog.axes_right is None


def test_the_right_series_go_on_their_own_axis(app, operands, result):
    dialog = make(app, operands, right_series=result)

    assert dialog.axes_right is not None
    assert [line.get_label() for line in dialog.axes.lines] == ["Data_A", "Data_B"]
    assert [line.get_label() for line in dialog.axes_right.lines] == ["Result"]


def test_the_two_axes_never_repeat_a_colour(app, operands, result):
    """On a dual-axis figure the curves are told apart by hue alone."""
    dialog = make(app, operands, right_series=result)

    colours = [line.get_color() for line in dialog.axes.lines + dialog.axes_right.lines]
    assert len(set(colours)) == len(colours)
    palette = dialog.current_palette()
    assert colours == list(palette.colors[:3])


def test_the_right_axis_is_labelled_and_tinted_to_its_curve(app, operands, result):
    """Without the tint, two scales cannot be told apart at a glance."""
    dialog = make(app, operands, right_series=result)

    assert dialog.axes_right.get_ylabel() == "Result"
    curve_colour = dialog.axes_right.lines[0].get_color()
    assert dialog.axes_right.yaxis.label.get_color() == curve_colour
    assert dialog.axes_right.spines["right"].get_edgecolor()[:3] != (0.0, 0.0, 0.0)


def test_one_legend_covers_both_axes(app, operands, result):
    dialog = make(app, operands, right_series=result)
    legend = dialog.axes.get_legend()

    assert [t.get_text() for t in legend.get_texts()] == ["Data_A", "Data_B", "Result"]
    assert dialog.axes_right.get_legend() is None


def test_the_data_page_holds_both_axes(app, operands, result):
    dialog = make(app, operands, right_series=result)

    # shared X + A + B + Result
    assert dialog.data_view.model().columnCount() == 4


def test_each_y_axis_takes_a_log_scale_on_its_own(app, operands, result):
    """The two sides carry different quantities — that is why there are two —
    so a ratio on one and raw counts on the other need separate switches."""
    dialog = make(app, operands, right_series=result)

    dialog.chk_log_y.setChecked(True)
    assert dialog.axes.get_yscale() == "log"
    assert dialog.axes_right.get_yscale() == "linear"

    dialog.chk_log_y.setChecked(False)
    dialog.chk_log_ry.setChecked(True)
    assert dialog.axes.get_yscale() == "linear"
    assert dialog.axes_right.get_yscale() == "log"


def test_zeroes_on_one_axis_do_not_veto_the_other(app, operands):
    """Judging the left switch by the right axis's data disabled a log scale
    the left axis could take perfectly well."""
    dialog = make(app, operands, right_series=[Series("Result", np.zeros(25))])

    assert dialog.chk_log_y.isEnabled()
    assert not dialog.chk_log_ry.isEnabled()


def test_the_right_log_switch_is_hidden_without_a_right_axis(app, curves):
    """A switch for an axis that is not there is just clutter."""
    dialog = make(app, curves)

    assert dialog.chk_log_ry.isHidden()


def test_all_four_label_fields_fit_on_one_row(app, operands, result):
    """Four text boxes at their natural width set the window's minimum wider
    than the screen needed to be, so the fourth could not appear without the
    window growing to meet it."""
    from src.gui.plot_dialog import LABEL_FIELD_MAX_WIDTH

    dialog = make(app, operands, right_series=result)
    fields = (dialog.le_title, dialog.le_x_label, dialog.le_y_label, dialog.le_right_label)

    for field in fields:
        # sizeHint stays at the text box's natural width; it is the cap the
        # layout honours, and so the cap that bounds the panel below.
        assert field.maximumWidth() == LABEL_FIELD_MAX_WIDTH
        assert field.sizeHint().width() > LABEL_FIELD_MAX_WIDTH

    # Four uncapped fields put this over 1200, which is what dragged the window
    # wider than the figure needed.
    assert dialog.minimumSizeHint().width() <= 1100


def test_redrawing_does_not_stack_up_right_axes(app, operands, result):
    """twinx() adds an axes each time; the old one has to go."""
    dialog = make(app, operands, right_series=result)
    before = len(dialog.figure.axes)

    dialog.chk_markers.setChecked(True)
    dialog.chk_grid.setChecked(False)

    assert len(dialog.figure.axes) == before == 2


def test_a_new_x_reaches_the_right_axis_too(app, operands, result, x_file, warned):
    dialog = make(app, operands, right_series=result)
    dialog.apply_x_key(f"{x_file}::scan_0033/scan_data/actuator_1_1")

    np.testing.assert_allclose(
        dialog.axes_right.lines[0].get_xdata(), np.linspace(-7000.0, 7000.0, 25)
    )


def test_reset_x_restores_both_axes(app, operands, result, x_file, warned):
    dialog = make(app, operands, right_series=result)
    dialog.apply_x_key(f"{x_file}::scan_0033/scan_data/actuator_1_1")
    dialog.reset_x()

    np.testing.assert_allclose(dialog.axes_right.lines[0].get_xdata(), result[0].x)


def test_a_right_series_with_no_finite_points_is_dropped(app, operands, warned):
    dialog = open_plot_dialog(None, operands, right_series=[Series("Result", np.full(25, np.nan))])

    assert dialog is not None
    assert dialog.axes_right is None
    dialog.close()


def test_an_empty_left_axis_promotes_the_right_one(app, result, warned):
    """Better than a figure whose primary axis holds nothing."""
    dialog = open_plot_dialog(None, [Series("Data_A", np.full(25, np.nan))], right_series=result)

    assert dialog is not None
    assert dialog.axes_right is None
    assert [line.get_label() for line in dialog.axes.lines] == ["Result"]
    dialog.close()


# ---------------------------------------------------------------------------
# An X dataset dragged in from the tree
# ---------------------------------------------------------------------------

@pytest.fixture
def x_file(tmp_path):
    import h5py

    path = tmp_path / "scanx_0033.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan_0033/scan_data")
        grp.create_dataset("actuator_1_1", data=np.linspace(-7000.0, 7000.0, 25))
        grp.create_dataset("short", data=np.arange(3.0))
        grp.create_dataset("frames", data=np.zeros((8, 8)))
    return path


@pytest.fixture
def warned(monkeypatch):
    seen: list[str] = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            "src.gui.plot_dialog.QMessageBox." + kind,
            staticmethod(lambda *a, _k=kind, **k: seen.append(_k)),
        )
    return seen


def key_for(path, ds):
    return f"{path}::{ds}"


def test_a_dropped_dataset_becomes_the_x_axis(app, curves, x_file, warned):
    dialog = make(app, curves)
    ok = dialog.apply_x_key(key_for(x_file, "scan_0033/scan_data/actuator_1_1"))

    assert ok
    for line in dialog.axes.lines:
        np.testing.assert_allclose(line.get_xdata(), np.linspace(-7000.0, 7000.0, 25))
    assert dialog.le_x_label.text() == "actuator_1_1"
    assert warned == []


def test_the_data_page_follows_the_new_x(app, curves, x_file, warned):
    dialog = make(app, curves)
    dialog.apply_x_key(key_for(x_file, "scan_0033/scan_data/actuator_1_1"))

    model = dialog.data_view.model()
    assert model.columnCount() == 3  # shared X + two curves
    assert model.rowCount() == 25


def test_a_new_x_re_gates_the_log_switch(app, x_file, warned):
    """The dropped X sweeps through zero, so log X has to be taken away."""
    x = np.linspace(1.0, 10.0, 25)
    dialog = make(app, [Series("a", x, x)])
    assert dialog.chk_log_x.isEnabled()
    dialog.chk_log_x.setChecked(True)

    dialog.apply_x_key(key_for(x_file, "scan_0033/scan_data/actuator_1_1"))

    assert not dialog.chk_log_x.isEnabled()
    assert not dialog.chk_log_x.isChecked()
    assert dialog.axes.get_xscale() == "linear"


def test_reset_puts_the_original_x_back(app, curves, x_file, warned):
    dialog = make(app, curves)
    original = curves[0].x.copy()

    dialog.apply_x_key(key_for(x_file, "scan_0033/scan_data/actuator_1_1"))
    dialog.reset_x()

    np.testing.assert_allclose(dialog.axes.lines[0].get_xdata(), original)
    assert dialog.le_x_source.text() == ""


def test_a_length_mismatch_changes_nothing(app, curves, x_file, warned):
    """Trimming to fit would pair values that do not belong together."""
    dialog = make(app, curves)
    ok = dialog.apply_x_key(key_for(x_file, "scan_0033/scan_data/short"))

    assert not ok
    assert warned == ["warning"]
    np.testing.assert_allclose(dialog.axes.lines[0].get_xdata(), curves[0].x)


def test_curves_of_a_different_length_keep_their_own_x(app, x_file, warned):
    x = np.linspace(1.0, 10.0, 25)
    dialog = make(app, [Series("fits", x, x), Series("other", np.arange(4.0))])

    assert dialog.apply_x_key(key_for(x_file, "scan_0033/scan_data/actuator_1_1"))
    assert warned == ["information"], "the user is told which curves were left alone"
    np.testing.assert_allclose(dialog.axes.lines[1].get_xdata(), [0, 1, 2, 3])


@pytest.mark.parametrize(
    "bad, why",
    [
        ("scan_0033/scan_data/frames", "a 2-D dataset is not an axis"),
        ("scan_0033/scan_data/not_there", "a missing dataset"),
    ],
)
def test_an_unusable_dataset_is_refused(app, curves, x_file, warned, bad, why):
    dialog = make(app, curves)
    assert not dialog.apply_x_key(key_for(x_file, bad)), why
    assert warned == ["warning"]


def test_text_without_a_file_half_is_refused(app, curves, warned):
    assert not make(app, curves).apply_x_key("just_a_name")
    assert warned == ["warning"]


def test_a_missing_file_is_refused(app, curves, warned, tmp_path):
    assert not make(app, curves).apply_x_key(key_for(tmp_path / "nope.h5", "a"))
    assert warned == ["warning"]


def test_the_x_field_accepts_drops(app, curves):
    assert make(app, curves).le_x_source.acceptDrops()


# ---------------------------------------------------------------------------
# Lock: one look across a series of scans
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_lock():
    """Never let a locked state leak between tests or out to the real app."""
    from PyQt6.QtCore import QSettings

    settings = QSettings()
    for key in ("plot/locked", "plot/locked_state"):
        settings.remove(key)
    yield
    for key in ("plot/locked", "plot/locked_state"):
        settings.remove(key)
    settings.sync()


def other_curves():
    x = np.linspace(100.0, 200.0, 15)
    return [Series("gamma", x * 2.0, x)]


def test_lock_is_off_by_default(app, curves):
    assert not make(app, curves).chk_lock.isChecked()


def test_locking_carries_the_settings_to_the_next_window(app, curves):
    first = make(app, curves)
    first.le_x_label.setText("Field (mT)")
    first.le_title.setText("Series A")
    first.chk_grid.setChecked(False)
    first.chk_markers.setChecked(True)
    first.chk_lock.setChecked(True)

    second = make(app, other_curves())

    assert second.chk_lock.isChecked()
    assert second.le_x_label.text() == "Field (mT)"
    assert second.axes.get_title() == "Series A"
    assert not second.chk_grid.isChecked()
    assert second.chk_markers.isChecked()


def test_a_setting_changed_while_locked_updates_what_is_carried(app, curves):
    first = make(app, curves)
    first.chk_lock.setChecked(True)
    first.le_y_label.setText("Counts")

    assert make(app, other_curves()).le_y_label.text() == "Counts"


def test_the_lock_carries_only_the_settings_box(app, curves):
    """The zoom is not a setting: each figure still autoscales to its own data."""
    first = make(app, curves)
    first.chk_lock.setChecked(True)

    assert set(first.lock_state()) == {
        "title", "x_label", "y_label", "right_label", "palette",
        "log_x", "log_y", "log_ry", "grid", "lines", "markers", "legend",
    }


def test_the_zoom_is_not_carried_to_the_next_window(app, curves):
    first = make(app, curves)
    first.axes.set_xlim(2.0, 4.0)
    first.chk_lock.setChecked(True)

    second = make(app, other_curves())

    assert second.axes.get_xlim() != (2.0, 4.0)


def test_a_locked_window_still_autoscales_on_redraw(app, curves):
    dialog = make(app, curves)
    dialog.chk_lock.setChecked(True)
    dialog.axes.set_xlim(2.0, 4.0)

    dialog.chk_markers.setChecked(True)

    assert dialog.axes.get_xlim() != (2.0, 4.0)


def test_unlocking_releases_the_next_window_to_its_own_defaults(app, curves):
    first = make(app, curves)
    first.le_x_label.setText("Field (mT)")
    first.chk_lock.setChecked(True)
    first.chk_lock.setChecked(False)

    second = make(app, other_curves())

    assert not second.chk_lock.isChecked()
    assert second.le_x_label.text() == "X"


def test_a_locked_log_axis_is_not_forced_onto_data_that_cannot_take_it(app, curves):
    """The lock carries a preference; it must never silently drop points."""
    first = make(app, curves)
    first.chk_log_x.setChecked(True)
    first.chk_lock.setChecked(True)

    x = np.linspace(-7000.0, 7000.0, 20)
    second = make(app, [Series("sweep", np.abs(x) + 1.0, x)])

    assert not second.chk_log_x.isChecked()
    assert second.axes.get_xscale() == "linear"


def test_a_locked_state_with_no_mark_at_all_still_draws(app, curves):
    """An old or hand-edited state must not produce a blank figure."""
    import json

    from PyQt6.QtCore import QSettings

    settings = QSettings()
    settings.setValue("plot/locked", True)
    settings.setValue("plot/locked_state", json.dumps({"lines": False, "markers": False}))

    dialog = make(app, curves)

    assert dialog.chk_lines.isChecked()
    assert dialog.axes.lines[0].get_linestyle() != "None"


def test_a_corrupt_locked_state_is_ignored_rather_than_raising(app, curves):
    from PyQt6.QtCore import QSettings

    settings = QSettings()
    settings.setValue("plot/locked", True)
    settings.setValue("plot/locked_state", "{not json")

    dialog = make(app, curves)
    assert dialog.chk_lock.isChecked()
    assert len(dialog.axes.lines) == len(curves)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def test_copy_puts_an_image_on_the_clipboard(app, curves):
    dialog = make(app, curves)
    QApplication.clipboard().clear()

    dialog.copy_figure()
    assert not QApplication.clipboard().image().isNull()


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg"])
def test_the_figure_saves_in_every_offered_format(app, curves, tmp_path, suffix):
    dialog = make(app, curves)
    target = tmp_path / f"figure{suffix}"

    dialog.figure.savefig(target)
    assert target.stat().st_size > 0


def test_save_corrects_an_extension_that_does_not_match_the_filter(app, curves, tmp_path, monkeypatch):
    dialog = make(app, curves)
    chosen = tmp_path / "figure.png"
    monkeypatch.setattr(
        "src.gui.plot_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(chosen), "SVG vector (*.svg)"),
    )

    dialog.save_figure()

    assert (tmp_path / "figure.svg").exists()
    assert not chosen.exists()


def test_cancelling_the_save_dialog_writes_nothing(app, curves, tmp_path, monkeypatch):
    dialog = make(app, curves)
    monkeypatch.setattr(
        "src.gui.plot_dialog.QFileDialog.getSaveFileName", lambda *a, **k: ("", "")
    )

    dialog.save_figure()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def test_empty_data_reports_instead_of_opening_a_blank_window(app, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "src.gui.plot_dialog.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args),
    )

    assert open_plot_dialog(None, [Series("nan", np.array([np.nan, np.nan]))]) is None
    assert shown, "the user is told why nothing opened"


def test_open_plot_dialog_returns_a_visible_window(app, curves):
    dialog = open_plot_dialog(None, curves, title="scan_0033")

    assert dialog is not None
    assert dialog.isVisible()
    assert "scan_0033" in dialog.windowTitle()
    dialog.close()

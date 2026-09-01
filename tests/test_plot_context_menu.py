"""Right-click on a plot offers the application's Export and Plot.

pyqtgraph builds that menu from three places. The view controls are worth
keeping, its own ``Export...`` is not — it writes files through its own dialog
and so ignores the naming, the dialects and the remembered folder that every
other export here follows. The Transforms/Average block has to go too: it
shadows the application's own toggles and drifts out of step with them.
"""

from __future__ import annotations

import h5py
import numpy as np
import pyqtgraph as pg
import pytest

from src.gui.plot_context_menu import EXPORT_TEXT, PLOT_TEXT, attach_plot_menu


def _scene_items(plot):
    return [action.text() for action in (plot.plotItem.scene().contextMenu or [])]


def _viewbox_items(plot):
    return [
        action.text().replace("&", "")
        for action in plot.plotItem.vb.menu.actions()
        if not action.isSeparator()
    ]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_the_view_controls_are_kept(qapp):
    """Auto-range and the axis controls have no equivalent anywhere in the
    application, so dropping them would cost something."""
    plot = pg.PlotWidget()

    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)

    assert plot.plotItem.vb.menuEnabled() is True
    assert _viewbox_items(plot) == ["View All", "X axis", "Y axis"]


def test_mouse_mode_is_dropped(qapp):
    """It only offers a rubber-band zoom — and zoom already works by wheel and
    by right-drag — while the choice is not remembered, so a new plot is back
    to the default and it would have to be made again every time."""
    plot = pg.PlotWidget()
    assert "Mouse Mode" in _viewbox_items(plot)

    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)

    assert "Mouse Mode" not in _viewbox_items(plot)


def test_dropping_mouse_mode_leaves_the_rest_working(qapp):
    plot = pg.PlotWidget()
    plot.plot([0, 1, 2], [0, 1, 4])

    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)
    plot.plotItem.vb.autoRange()      # what View All calls

    assert _viewbox_items(plot)[0] == "View All"


def test_the_transforms_block_is_removed(qapp):
    """Transforms → Log X puts the axis into log while the toolbar checkbox
    still reads linear, and Average/Downsample change the curve without
    changing what an export would write."""
    plot = pg.PlotWidget()

    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)

    assert plot.plotItem.menuEnabled() is False


def test_pyqtgraphs_export_is_replaced(qapp):
    plot = pg.PlotWidget()
    assert _scene_items(plot) == ["Export..."]

    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)

    assert _scene_items(plot) == [EXPORT_TEXT, PLOT_TEXT]


def test_the_actions_call_what_they_were_given(qapp):
    plot = pg.PlotWidget()
    called: list[str] = []

    attach_plot_menu(
        plot,
        on_export=lambda: called.append("export"),
        on_plot=lambda: called.append("plot"),
    )
    for action in plot.plotItem.scene().contextMenu:
        action.trigger()

    assert called == ["export", "plot"]


def test_attaching_twice_does_not_pile_up(qapp):
    """A window may hand its plot different actions later."""
    plot = pg.PlotWidget()

    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)
    attach_plot_menu(plot, on_export=lambda: None, on_plot=lambda: None)

    assert _scene_items(plot) == [EXPORT_TEXT, PLOT_TEXT]


def test_one_plot_is_not_changed_by_another(qapp):
    """The scene menu is per-widget, which is what makes this safe to do to
    some plots and not others."""
    attached = pg.PlotWidget()
    untouched = pg.PlotWidget()

    attach_plot_menu(attached, on_export=lambda: None, on_plot=lambda: None)

    assert _scene_items(untouched) == ["Export..."]


def test_a_missing_callback_leaves_that_item_out(qapp):
    plot = pg.PlotWidget()

    attach_plot_menu(plot, on_export=lambda: None)

    assert _scene_items(plot) == [EXPORT_TEXT]


def test_something_that_is_not_a_plot_is_refused_quietly(qapp):
    from PyQt6.QtWidgets import QWidget

    assert attach_plot_menu(QWidget(), on_export=lambda: None) is False


# ---------------------------------------------------------------------------
# The five windows
# ---------------------------------------------------------------------------

@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scanx_0340.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("curve", data=np.arange(50.0))
        f.create_dataset("img", data=np.ones((40, 40), dtype=np.float32))
    return path


def _windows(qapp, scan):
    """Every window the menu was rolled out to, and the plot it lives on."""
    from src.gui.data_comparison import DataComparisonTool
    from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
    from src.gui.q_calibration_tool import QCalibrationTool
    from src.gui.xrms_analyze_tool import XRMSAnalyzeTool

    viewer = PlotWidget1DEnhanced()
    viewer.set_data(np.arange(20.0))
    comparison = DataComparisonTool(tuple())
    scattering = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    xrms = XRMSAnalyzeTool(tuple())
    return [
        ("1-D viewer", viewer, viewer.plot_widget),
        ("comparison", comparison, comparison.plot_widget),
        ("scattering", scattering, scattering._profile_plot),
        ("time resolved", xrms, xrms._plot_profile),
    ]


def test_every_target_window_offers_the_same_two_items(qapp, scan):
    """One vocabulary, whichever plot is under the cursor."""
    for name, owner, plot in _windows(qapp, scan):
        try:
            assert _scene_items(plot) == [EXPORT_TEXT, PLOT_TEXT], name
            assert plot.plotItem.menuEnabled() is False, name
            assert plot.plotItem.vb.menuEnabled() is True, name
            assert _viewbox_items(plot) == ["View All", "X axis", "Y axis"], name
        finally:
            owner.deleteLater()


def test_the_calculator_result_gets_it_through_the_viewer(qapp, scan):
    """It embeds the 1-D viewer, so attaching there covers it."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced((scan,))
    try:
        calc.result_data = np.arange(10.0)
        calc._update_result_display()
        plot = calc._result_plot_widget().plot_widget

        assert _scene_items(plot) == [EXPORT_TEXT, PLOT_TEXT]
    finally:
        calc.deleteLater()


# ---------------------------------------------------------------------------
# The two profiles that had no way out at all
# ---------------------------------------------------------------------------

def test_the_scattering_profile_is_headed_with_the_axis_in_use(qapp, scan):
    """A file exported in q says q; one exported in pixels says pixels."""
    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        tool.load_dataset_full_key(f"{scan}::img", auto_load=True, slot="CL")
        tool._center_col.setValue(20)
        tool._center_row.setValue(20)
        tool._add_roi("ring")
        tool._compute_current_profiles()

        headers, columns = tool._profile_columns()
        assert headers[0] == "r (px)"
        assert len(columns) == 2 and len(columns[0]) == len(columns[1])

        tool._spin_energy.setValue(700.0)
        tool._spin_pixel.setValue(75.0)
        tool._spin_dist.setValue(500.0)
        tool._apply_calibration()

        assert tool._profile_columns()[0][0] == "q (1/A)"
    finally:
        tool.deleteLater()


def test_the_scattering_profile_says_so_when_there_is_none(qapp, scan):
    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        assert tool._profile_columns() is None

        tool._export_profile()      # must not raise or open a file dialog

        assert "No profile" in tool._status.text()
    finally:
        tool.deleteLater()


def test_the_xrms_profile_follows_its_own_mode(qapp):
    from src.gui.xrms_analyze_tool import XRMSAnalyzeTool

    tool = XRMSAnalyzeTool(tuple())
    try:
        tool._set_new_stack(np.random.RandomState(0).rand(4, 40, 40).astype(np.float32), "t")
        tool._add_roi("ring")
        tool._update_profile_plots()

        assert tool._profile_columns()[0][0] == "Radius (pixels)"

        tool._selected_roi()["mode"] = "angular"
        tool._update_profile_plots()

        assert tool._profile_columns()[0][0] == "Angle (deg)"
    finally:
        tool.deleteLater()


def test_the_xrms_profile_says_so_when_there_is_none(qapp):
    from src.gui.xrms_analyze_tool import XRMSAnalyzeTool

    tool = XRMSAnalyzeTool(tuple())
    try:
        assert tool._profile_columns() is None

        tool._export_profile()

        assert "No profile" in tool._status.text()
    finally:
        tool.deleteLater()


# ---------------------------------------------------------------------------
# The 2-D viewer's Line / Rect / Sector profile
# ---------------------------------------------------------------------------

@pytest.fixture
def image_view(qapp):
    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    view = ImageView2DEnhanced()
    view.set_data(np.arange(64 * 64, dtype=float).reshape(64, 64))
    view.set_source_dataset_key("d:/data/scanx_0340.h5::entry/data")
    yield view
    view.deleteLater()


def _draw_roi(view, kind):
    """Arm the button the way a click does, which is what creates the ROI.

    ``kind`` is what the button itself passes: "Line" or "Rectangle".
    """
    button = view.btn_roi_line if kind == "Line" else view.btn_roi_rect
    button.setChecked(True)
    view._on_roi_button_clicked(kind)
    assert view.roi_type is not None, f"{kind} ROI was not created"


def test_the_roi_profile_plot_has_the_menu(image_view):
    """It is inside the 2-D viewer, so every window that embeds one gets it —
    the main window, the calculator, the tools."""
    plot = image_view.roi_plot_widget

    assert _scene_items(plot) == [EXPORT_TEXT, PLOT_TEXT]
    assert plot.plotItem.menuEnabled() is False
    assert plot.plotItem.vb.menuEnabled() is True


def test_a_line_roi_profile_can_be_exported(image_view):
    _draw_roi(image_view, "Line")

    headers, columns = image_view.roi_profile_columns()

    assert headers == ["X (pixels)", "Pixel intensity"]
    assert len(columns) == 2 and len(columns[0]) == len(columns[1]) > 0


def test_a_rect_roi_profile_can_be_exported(image_view):
    _draw_roi(image_view, "Rectangle")

    headers, columns = image_view.roi_profile_columns()

    assert headers[1] == "Pixel intensity"
    assert len(columns[0]) > 0


def test_the_profile_header_follows_the_q_axes(image_view):
    """The line ROI switches its own axis to qx/qy once calibrated; what is
    written has to say the same thing."""
    _draw_roi(image_view, "Line")
    assert image_view.roi_profile_columns()[0][0] == "X (pixels)"

    image_view.set_q_calibration({
        "energy_ev": 700.0, "pixel_um": 75.0, "distance_mm": 500.0,
        "center_x": 32.0, "center_y": 32.0, "use_incidence": False,
        "incidence_deg": 0.0, "incidence_axis": "X",
        "incidence_applied_in_display": False,
    })
    image_view.apply_q_axes_calibration(image_view._q_calibration)
    image_view._update_roi_statistics()

    assert image_view.roi_profile_columns()[0][0] == "qx (1/A)"


def test_no_roi_means_nothing_to_export(image_view, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    told: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda *a, **k: told.append(a[1]) or QMessageBox.StandardButton.Ok),
    )

    assert image_view.roi_profile_columns() is None
    image_view.export_roi_profile()

    assert told == ["No Profile"], "and no file dialog was opened"


# ---------------------------------------------------------------------------
# Deliberately left alone
# ---------------------------------------------------------------------------

def test_the_rollout_stops_where_it_was_meant_to():
    """The menu belongs on windows for *browsing* a curve. FTH and CDI are made
    of intermediate steps, and the Plot window is matplotlib, where none of this
    machinery applies.
    """
    import pathlib

    gui = pathlib.Path(__file__).resolve().parent.parent / "src" / "gui"
    users = sorted(
        path.name for path in gui.glob("*.py")
        if "attach_plot_menu" in path.read_text(encoding="utf-8")
        and path.name != "plot_context_menu.py"
    )

    assert users == [
        "data_comparison.py",
        "image_view_2d_enhanced.py",
        "plot_widget_1d_enhanced.py",
        "q_calibration_tool.py",
        "xrms_analyze_tool.py",
    ]

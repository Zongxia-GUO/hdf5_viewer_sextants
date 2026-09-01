"""The tree's "Set X" reaches the X-capable window the user was last in.

Several windows have an X field — the export dialogs and the Plot window — and
the point of the register is that ``Set X`` lands in the one in front rather
than always in the main viewer.
"""

import h5py
import numpy as np
import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from src.gui.main_window import MainWindow
from src.gui.x_target import active_x_target, clear_x_targets, register_x_target


class _Target(QDialog):
    """Minimal window implementing the X-target protocol."""

    def __init__(self, name="target", accepts=True):
        super().__init__()
        self.setWindowTitle(name)
        self.received: list[str] = []
        self._accepts = accepts

    def set_x_dataset(self, key: str) -> bool:
        self.received.append(key)
        return self._accepts


@pytest.fixture(autouse=True)
def clean_register():
    clear_x_targets()
    yield
    clear_x_targets()


def activate(widget):
    """Stand in for the user clicking on a window.

    Sent through the application, not straight to ``widget.event`` — event
    filters run inside ``QCoreApplication.notify``, so a direct call would skip
    the watcher entirely and test nothing.
    """
    QApplication.sendEvent(widget, QEvent(QEvent.Type.WindowActivate))


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------

def test_nothing_is_registered_to_begin_with(qapp):
    assert active_x_target() is None


def test_a_registered_window_becomes_the_target(qapp):
    target = _Target()
    target.show()
    register_x_target(target)

    assert active_x_target() is target


def test_the_newest_window_wins(qapp):
    first, second = _Target("first"), _Target("second")
    for t in (first, second):
        t.show()
        register_x_target(t)

    assert active_x_target() is second


def test_activating_an_older_window_brings_it_back_to_the_front(qapp):
    first, second = _Target("first"), _Target("second")
    for t in (first, second):
        t.show()
        register_x_target(t)

    activate(first)
    assert active_x_target() is first


def test_a_hidden_window_is_not_a_target(qapp):
    target = _Target()
    target.show()
    register_x_target(target)
    target.hide()

    assert active_x_target() is None


def test_a_closed_window_falls_out_of_the_register(qapp):
    first, second = _Target("first"), _Target("second")
    for t in (first, second):
        t.show()
        register_x_target(t)

    second.close()
    assert active_x_target() is first


def test_a_window_without_the_method_is_refused(qapp):
    with pytest.raises(TypeError):
        register_x_target(QDialog())


# ---------------------------------------------------------------------------
# Routing from the tree
# ---------------------------------------------------------------------------

class _XHost:
    _set_dataset_as_x = MainWindow._set_dataset_as_x
    _x_dataset_path = MainWindow._x_dataset_path

    def __init__(self, widget=None):
        self._x_dataset_key = None
        self._widget = widget
        self.status: list[str] = []

    def _current_plot_widget_1d(self):
        return self._widget

    def _set_status_text(self, text=""):
        self.status.append(text)


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scanx_0033.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan_0033/scan_data")
        grp.create_dataset("actuator_1_1", data=np.linspace(-7000.0, 7000.0, 10))
    return path


@pytest.fixture
def boxes(monkeypatch):
    seen: list[str] = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, kind, staticmethod(lambda *a, _k=kind, **k: seen.append(_k))
        )
    return seen


def keys_for(path):
    return [str(path)] + "scan_0033/scan_data/actuator_1_1".split("/")


def test_set_x_goes_to_the_open_window_not_the_viewer(qapp, scan, boxes):
    from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

    viewer = PlotWidget1DEnhanced()
    viewer.set_data(np.arange(10, dtype=float))
    target = _Target("Plot - scan_0033")
    target.show()
    register_x_target(target)

    _XHost(viewer)._set_dataset_as_x(keys_for(scan))

    assert target.received == [f"{scan}::scan_0033/scan_data/actuator_1_1"]
    assert viewer.x_data is None, "the viewer is left alone while a window is in front"


def test_the_choice_is_still_remembered_for_later_dialogs(qapp, scan, boxes):
    target = _Target()
    target.show()
    register_x_target(target)

    host = _XHost()
    host._set_dataset_as_x(keys_for(scan))

    assert host._x_dataset_path() == "scan_0033/scan_data/actuator_1_1"


def test_the_viewer_is_used_when_no_such_window_is_open(qapp, scan, boxes):
    from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

    viewer = PlotWidget1DEnhanced()
    viewer.set_data(np.arange(10, dtype=float))

    _XHost(viewer)._set_dataset_as_x(keys_for(scan))

    np.testing.assert_allclose(viewer.x_data, np.linspace(-7000.0, 7000.0, 10))


def test_a_refusing_window_says_so_instead_of_falling_through(qapp, scan, boxes):
    """The window has already explained why; re-routing would be a surprise."""
    from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced

    viewer = PlotWidget1DEnhanced()
    viewer.set_data(np.arange(10, dtype=float))
    target = _Target("Plot - other", accepts=False)
    target.show()
    register_x_target(target)

    host = _XHost(viewer)
    host._set_dataset_as_x(keys_for(scan))

    assert viewer.x_data is None
    assert "did not accept" in host.status[-1]


def test_an_unreadable_dataset_never_reaches_the_window(qapp, scan, boxes):
    target = _Target()
    target.show()
    register_x_target(target)

    _XHost()._set_dataset_as_x([str(scan), "scan_0033", "scan_data", "missing"])

    assert target.received == []
    assert boxes == ["critical"]


# ---------------------------------------------------------------------------
# The real windows implement the protocol
# ---------------------------------------------------------------------------

def test_the_plot_window_takes_a_handed_over_x(qapp, scan):
    from src.gui.plot_dialog import PlotDialog
    from src.gui.plot_series import Series

    key = f"{scan}::scan_0033/scan_data/actuator_1_1"
    dialog = PlotDialog(series=[Series("a", np.arange(10, dtype=float))])
    dialog.close()

    assert dialog.set_x_dataset(key)
    assert dialog.le_x_source.text() == key
    np.testing.assert_allclose(dialog.axes.lines[0].get_xdata(), np.linspace(-7000.0, 7000.0, 10))


def test_the_batch_export_dialog_keeps_only_the_dataset_half(qapp):
    import pathlib

    from src.gui.batch_export import BatchExportDialog

    dialog = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0033"],
        dataset_path="/scan_0033/scan_data/data_01",
        sample_data=np.arange(10, dtype=float),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
    )
    dialog.close()

    assert dialog.set_x_dataset("C:/data/scanx_0033.h5::/scan_0033/scan_data/actuator_1_1")
    assert dialog.le_x_path.text() == "scan_0033/scan_data/actuator_1_1"
    assert dialog.chk_export_x.isChecked()


def test_the_comparison_export_dialog_keeps_the_whole_key(qapp):
    from src.gui.data_comparison import ComparisonExportDialog

    dialog = ComparisonExportDialog()
    dialog.close()

    key = "C:/data/scanx_0033.h5::scan_0033/scan_data/actuator_1_1"
    assert dialog.set_x_dataset(key)
    assert dialog.x_key() == key


def test_the_comparison_tool_takes_a_handed_over_x(qapp, scan):
    """Same outcome as its own Set X button: every row of matching length."""
    from src.gui.data_comparison import CurveEntry, DataComparisonTool

    comp = DataComparisonTool(tuple())
    comp.datasets = [
        CurveEntry(name="fits", data=np.arange(10, dtype=float)),
        CurveEntry(name="other", data=np.arange(4, dtype=float)),
    ]

    key = f"{scan}::scan_0033/scan_data/actuator_1_1"
    assert comp.set_x_dataset(key)

    np.testing.assert_allclose(comp.datasets[0].x_data, np.linspace(-7000.0, 7000.0, 10))
    assert comp.datasets[1].x_data is None, "a row of another length keeps its own X"


def test_the_comparison_tool_refuses_an_x_no_row_can_use(qapp, scan, boxes):
    from src.gui.data_comparison import CurveEntry, DataComparisonTool

    comp = DataComparisonTool(tuple())
    comp.datasets = [
        CurveEntry(name="only", data=np.arange(4, dtype=float)),
    ]

    assert not comp.set_x_dataset(f"{scan}::scan_0033/scan_data/actuator_1_1")
    assert boxes == ["warning"]


def test_the_comparison_tool_says_so_when_it_holds_nothing(qapp, scan, boxes):
    from src.gui.data_comparison import DataComparisonTool

    comp = DataComparisonTool(tuple())

    assert not comp.set_x_dataset(f"{scan}::scan_0033/scan_data/actuator_1_1")
    assert boxes == ["information"]


def test_the_comparison_button_is_called_set_x(qapp):
    from src.gui.data_comparison import DataComparisonTool

    assert DataComparisonTool(tuple()).btn_select_x.text() == "Set X"


def test_the_comparison_tool_offers_itself_as_a_target(qapp):
    from src.gui.data_comparison import DataComparisonTool

    comp = DataComparisonTool(tuple())
    comp.show()

    assert active_x_target() is comp
    comp.close()


def test_the_calculator_takes_a_handed_over_x(qapp, scan):
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()

    key = f"{scan}::scan_0033/scan_data/actuator_1_1"
    assert calc.set_x_dataset(key)

    widget = calc._result_plot_widget()
    np.testing.assert_allclose(widget.x_data, np.linspace(-7000.0, 7000.0, 10))
    assert calc._viewer_x_dataset_key() == key


def test_the_calculator_x_reaches_its_export_and_its_plot(qapp, scan):
    """The viewer is where both read their X from, so one write serves both."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.data_a = np.arange(10, dtype=float)
    calc.result_data = np.arange(10, dtype=float) * 2
    calc._update_result_display()

    calc.set_x_dataset(f"{scan}::scan_0033/scan_data/actuator_1_1")

    headers, _columns = calc._result_export_columns(calc._viewer_x_dataset_key())
    assert headers[0] == "scanx_0033_actuator_1_1_X"
    operands, _result = calc.plot_axes_series()
    np.testing.assert_allclose(operands[0].x, np.linspace(-7000.0, 7000.0, 10))


def test_the_calculator_refuses_before_there_is_a_result(qapp, scan, boxes):
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())

    assert not calc.set_x_dataset(f"{scan}::scan_0033/scan_data/actuator_1_1")
    assert boxes == ["information"]


def test_the_calculator_refuses_a_length_that_does_not_fit(qapp, scan, boxes):
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(4, dtype=float)
    calc._update_result_display()

    assert not calc.set_x_dataset(f"{scan}::scan_0033/scan_data/actuator_1_1")
    assert boxes == ["warning"]


def test_the_calculator_offers_itself_as_a_target(qapp):
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()
    calc.show()

    assert active_x_target() is calc
    calc.close()


def test_a_calculator_with_no_result_does_not_claim_set_x(qapp):
    """Being in front is a claim on Set X, and a window that would only refuse
    must not make it — the choice would be swallowed there and never reach the
    viewer in the main window. Closing the calculator clears its result, so this
    is the state it comes back in."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.show()

    assert calc.can_take_x() is False
    assert active_x_target() is None

    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()

    assert calc.can_take_x() is True
    assert active_x_target() is calc
    calc.close()


# ---------------------------------------------------------------------------
# One remembered X per tool, not one for the whole application
# ---------------------------------------------------------------------------

def test_a_window_belongs_to_the_tool_that_opened_it(qapp):
    """Read off the parent chain, so a dialog opened from a tool inherits that
    tool's scope without being told."""
    from PyQt6.QtWidgets import QWidget
    from src.gui.x_target import DEFAULT_X_SCOPE, x_scope_of

    outer = QWidget()
    outer.x_scope = "calculator"
    inner = QWidget(outer)
    grandchild = QWidget(inner)

    assert x_scope_of(grandchild) == "calculator"
    assert x_scope_of(QWidget()) == DEFAULT_X_SCOPE
    assert x_scope_of(None) == DEFAULT_X_SCOPE


def test_each_tool_remembers_its_own_x(qapp):
    from src.gui.x_target import remember_x_dataset, remembered_x_dataset

    remember_x_dataset("a.h5::main_axis", "main")
    remember_x_dataset("b.h5::calc_axis", "calculator")

    assert remembered_x_dataset("main") == "a.h5::main_axis"
    assert remembered_x_dataset("calculator") == "b.h5::calc_axis"
    assert remembered_x_dataset("comparison") == ""


def test_an_x_set_in_the_calculator_does_not_reach_the_batch_export(qapp, scan, boxes):
    """The reported leak. The batch export runs over a whole range of scans in
    the tree; an X picked for one calculated curve is not an answer to that."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced
    from src.gui.x_target import DEFAULT_X_SCOPE, remembered_x_dataset

    window = MainWindow()
    calc = DataCalculatorEnhanced(tuple(), window)
    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()
    calc.show()
    assert active_x_target() is calc

    window._set_dataset_as_x([str(scan), "scan_0033", "scan_data", "actuator_1_1"])

    assert remembered_x_dataset("calculator").endswith("actuator_1_1")
    assert remembered_x_dataset(DEFAULT_X_SCOPE) == ""
    assert window._x_dataset_path() == "", "the batch export starts from nothing"
    calc.close()
    window.close()


def test_an_x_set_for_the_main_viewer_still_reaches_the_batch_export(qapp, scan):
    """Isolation must not break the inheritance that was wanted."""
    from src.gui.x_target import DEFAULT_X_SCOPE, remembered_x_dataset

    window = MainWindow()
    window._set_dataset_as_x([str(scan), "scan_0033", "scan_data", "actuator_1_1"])

    assert remembered_x_dataset(DEFAULT_X_SCOPE).endswith("actuator_1_1")
    assert window._x_dataset_path() == "scan_0033/scan_data/actuator_1_1"
    window.close()


def test_a_plot_opened_from_a_tool_inherits_that_tool_s_x(qapp, scan):
    """The inheritance the user does want: inside one tool, an X set once is
    the starting point for its plot and its export."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced
    from src.gui.plot_dialog import open_plot_dialog
    from src.gui.plot_series import Series
    from src.gui.x_target import remember_x_dataset

    key = f"{scan}::scan_0033/scan_data/actuator_1_1"
    remember_x_dataset(key, "calculator")
    remember_x_dataset("", "main")

    calc = DataCalculatorEnhanced(tuple())
    plot = open_plot_dialog(calc, [Series("Result", np.arange(10, dtype=float))])

    assert plot.le_x_source.text() == key
    plot.close()
    calc.close()


def test_a_plot_opened_from_the_main_window_does_not_inherit_a_tool_s_x(qapp, scan):
    from src.gui.plot_dialog import open_plot_dialog
    from src.gui.plot_series import Series
    from src.gui.x_target import remember_x_dataset

    remember_x_dataset(f"{scan}::scan_0033/scan_data/actuator_1_1", "calculator")

    window = MainWindow()
    plot = open_plot_dialog(window, [Series("Data", np.arange(10, dtype=float))])

    assert plot.le_x_source.text() == ""
    plot.close()
    window.close()


def test_a_reopened_calculator_stops_swallowing_set_x(qapp, scan, boxes):
    """The whole round trip: close it, reopen it, and the tree's Set X has to
    reach past it instead of stopping on 'Run a calculation first'."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()
    calc.show()
    assert active_x_target() is calc

    calc.close()          # clears the result
    calc.show()           # the same object: it is kept for a fast reopen

    assert active_x_target() is None
    assert boxes == []


def test_closing_a_plot_hands_set_x_back_to_the_calculator(qapp, scan):
    """The reported bug: it went past the calculator to the main viewer."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(10, dtype=float)
    calc._update_result_display()
    calc.show()   # registered when it was built

    plot = _Target("Plot - Calculator")
    plot.show()
    register_x_target(plot)

    assert active_x_target() is plot, "the newest window wins while it is open"

    plot.close()

    assert active_x_target() is calc, "not the main window's viewer"


def test_three_layers_hand_over_in_order(qapp):
    """Calculator, then its export dialog, then a plot: last one in front wins."""
    from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

    calc = DataCalculatorEnhanced(tuple())
    calc.result_data = np.arange(10, dtype=float)   # so it can take an X at all
    calc._update_result_display()
    calc.show()

    export = _Target("Export")
    export.show()
    register_x_target(export)

    plot = _Target("Plot")
    plot.show()
    register_x_target(plot)

    assert active_x_target() is plot
    plot.close()
    assert active_x_target() is export
    export.close()
    assert active_x_target() is calc


def test_an_image_export_dialog_never_offers_itself(qapp):
    """It has no usable X field, so Set X must not land there."""
    import pathlib

    from src.gui.batch_export import BatchExportDialog

    dialog = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0033"],
        dataset_path="/scan_0033/scan_data/frames",
        sample_data=np.zeros((32, 32)),
        data_kind="image",
        preview_x_loader=lambda *a, **k: None,
    )
    dialog.show()

    assert active_x_target() is None
    dialog.close()

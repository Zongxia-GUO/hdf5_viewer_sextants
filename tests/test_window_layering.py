"""No window sits above every other program on the desktop.

Several dialogs are deliberately non-modal, so the main window's tree stays
usable while they are open. That needs `setModal(False)` and nothing more —
``WindowStaysOnTopHint`` additionally pins the window above *other
applications*, which is why switching to another program still left them in
front. The two were repeatedly confused, so the rule is checked here rather
than rediscovered.
"""

import pathlib

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from src.gui.batch_export import BatchExportDialog
from src.gui.data_calculator_enhanced import DataCalculatorEnhanced
from src.gui.data_comparison import DataComparisonTool
from src.gui.plot_dialog import PlotDialog
from src.gui.plot_series import Series
from src.gui.plot_widget_1d_enhanced import XDataSelectionDialog
from src.gui.q_calibration_tool import QCalibrationTool
from src.gui.xrms_analyze_tool import XRMSAnalyzeTool


def batch_dialog(kind, data):
    return BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0080"],
        dataset_path="/scan/scan_data/data_01",
        sample_data=data,
        data_kind=kind,
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _s, **caps: (np.zeros((4, 2)), ["X", "Y"]),
    )


@pytest.fixture(params=[
    "plot", "batch-curve", "batch-image", "x-select",
    "comparison", "calculator", "xrms", "q-calibration",
])
def window(request, qapp):
    kind = request.param
    if kind == "plot":
        return PlotDialog(series=[Series("a", np.arange(5.0))])
    if kind == "batch-curve":
        return batch_dialog("curve", np.arange(12.0))
    if kind == "batch-image":
        return batch_dialog("image", np.zeros((16, 16)))
    if kind == "x-select":
        return XDataSelectionDialog(tuple(), None, np.arange(5.0))
    if kind == "comparison":
        return DataComparisonTool(tuple())
    if kind == "xrms":
        return XRMSAnalyzeTool(tuple())
    if kind == "q-calibration":
        return QCalibrationTool(tuple())
    return DataCalculatorEnhanced(tuple())


def test_no_dialog_is_pinned_above_other_applications(window):
    assert not (window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)


def test_the_batch_export_stays_above_the_window_it_came_from(qapp):
    """It is non-modal so the tree stays usable — and then the first click into
    that tree sank it behind the main window, which defeats the point.

    Ownership, not StaysOnTopHint: a dialog is kept above its parent by the
    window manager while the pair still goes behind other programs together.
    """
    parent = DataCalculatorEnhanced(tuple())
    dialog = BatchExportDialog(
        parent,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0080"],
        dataset_path="/scan/scan_data/data_01",
        sample_data=np.arange(12.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _s, **caps: (np.zeros((4, 2)), ["X", "Y"]),
    )
    try:
        # `&` yields a plain int here, so compare against the enum's value.
        window_type = int(dialog.windowFlags() & Qt.WindowType.WindowType_Mask)

        assert dialog.parent() is parent, "ownership is what does the work"
        assert window_type == Qt.WindowType.Dialog.value
        assert not (dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        assert not dialog.isModal()
    finally:
        dialog.deleteLater()
        parent.deleteLater()


def test_the_batch_export_is_opened_with_an_owner(qapp):
    """The flags were never the problem — it was opened with no parent at all,
    so there was nothing for the window manager to keep it above."""
    import inspect

    from src.gui import main_window

    source = inspect.getsource(main_window.MainWindow._open_export_dialog)
    call = source.split("BatchExportDialog(", 1)[1]
    first_argument = next(
        line.strip().rstrip(",")
        for line in call.splitlines()[1:]
        if line.strip() and not line.strip().startswith("#")
    )

    assert first_argument == "self", "a parentless dialog sinks behind the main window"


def test_every_dialog_stays_non_modal(window):
    """Blocking the main window would take the tree away, which is the point."""
    assert not window.isModal()
    assert window.windowModality() == Qt.WindowModality.NonModal


def test_the_rule_is_not_reintroduced_in_the_sources():
    """A grep, because the flag reads as harmless at the call site."""
    gui = pathlib.Path(__file__).resolve().parent.parent / "src" / "gui"
    offenders = []
    for path in gui.glob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "WindowStaysOnTopHint" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{number}")

    assert offenders == [], "use setModal(False) alone; see this module's docstring"


def test_nothing_is_shown_with_exec(qapp):
    """``exec()`` blocks the whole application until the window closes.

    Correct for a confirmation box or a short prompt, wrong for anything the
    user is meant to work alongside — and the two are one call apart.

    Blocking is allowed only through one of the names below, so a call site has
    to say which kind of window it opened. ``dialog.exec()`` fails on purpose:
    that name says nothing, and it is what a tool window would be called.
    """
    blocking_by_design = ("msg", "box", "drag", "prompt")

    gui = pathlib.Path(__file__).resolve().parent.parent / "src" / "gui"
    offenders = []
    for path in gui.glob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or ".exec()" not in stripped:
                continue
            if any(f"{name}.exec()" in stripped for name in blocking_by_design):
                continue
            offenders.append(f"{path.name}:{number}  {stripped}")

    assert offenders == [], "show() a tool window; exec() only a confirmation"

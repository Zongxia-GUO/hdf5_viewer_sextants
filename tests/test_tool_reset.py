"""Closing a tool throws away what it was holding.

Every tool window is kept alive between uses so that reopening is instant. The
cost of that is state: a closed and reopened tool came back showing the
previous scan's image, ROIs, profile or reconstruction, which reads as data for
whatever is selected now.

The dataset selections and the parameters are deliberately NOT cleared — they
are the question, not the answer, and every tool re-reads the files when it
next computes. That line was drawn with the calculator first and the rest
follow it.
"""

import ast
import pathlib
import re

import h5py
import numpy as np
import pyqtgraph as pg
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "gui"

# Every tool that survives its own close.
TOOLS = [
    ("data_calculator_enhanced", "DataCalculatorEnhanced"),
    ("q_calibration_tool", "QCalibrationTool"),
    ("xrms_analyze_tool", "XRMSAnalyzeTool"),
    ("fth_reconstruction_tool", "FTHReconstructionTool"),
    ("cdi_reconstruction_tool", "CDIReconstructionTool"),
]

ARRAY_TOOLS = [
    ("fth_reconstruction_tool", "FTHReconstructionTool"),
    ("cdi_reconstruction_tool", "CDIReconstructionTool"),
]


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "Scan_ECL_007.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("scan_0007/scan_data")
        g.create_dataset("curve", data=np.arange(64.0))
        g.create_dataset("image", data=np.random.RandomState(0).rand(48, 160))
    return path


def _class_node(module: str, cls: str) -> ast.ClassDef:
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls)


def _calls_in(method: ast.FunctionDef) -> set:
    return {n.func.attr for n in ast.walk(method)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


# ── Every tool clears on close ────────────────────────────────────────── #

@pytest.mark.parametrize("module,cls", TOOLS)
def test_the_tool_clears_itself_when_it_closes(module, cls):
    """Not that it *can* clear, but that closing is wired to it."""
    node = _class_node(module, cls)
    methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}

    assert "reset_results" in methods, f"{cls} cannot clear itself"
    assert "closeEvent" in methods, f"{cls} never clears on close"

    close = next(m for m in node.body
                 if isinstance(m, ast.FunctionDef) and m.name == "closeEvent")
    assert "reset_results" in _calls_in(close), f"{cls}.closeEvent does not reset"


def test_the_comparison_window_clears_by_its_own_name():
    """It holds a list of curves rather than one result, so its clearing method
    is named for that; the wiring is what matters."""
    close = next(m for m in _class_node("data_comparison", "DataComparisonTool").body
                 if isinstance(m, ast.FunctionDef) and m.name == "closeEvent")

    assert "_clear_all_silent" in _calls_in(close)


# ── No array attribute can be added and silently left behind ──────────── #

def _declared_array_attrs(module: str) -> list:
    """Attributes __init__ declares as ``Optional[np.ndarray] = None``."""
    text = (SRC / f"{module}.py").read_text(encoding="utf-8")
    return re.findall(r"^        self\.(_\w+):\s*Optional\[np\.ndarray\] = None",
                      text, re.M)


@pytest.mark.parametrize("module,cls", ARRAY_TOOLS)
def test_every_array_the_tool_declares_is_on_its_clear_list(module, cls):
    """A reset that lists attributes by hand goes stale silently: the new array
    keeps the previous scan and nothing says so. This fails the day one is
    added to __init__ without being listed."""
    import importlib

    tool_cls = getattr(importlib.import_module(f"src.gui.{module}"), cls)
    declared = set(_declared_array_attrs(module))
    listed = set(tool_cls._DATA_ATTRS)

    assert declared, "the scan for array attributes found nothing — check the pattern"
    assert declared - listed == set(), f"not cleared on close: {sorted(declared - listed)}"


# ── The Q tool, which is the one that was reported ────────────────────── #

def test_the_q_tool_forgets_its_image_and_its_rois(qapp, scan):
    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        loaded = tool.load_dataset_full_key(
            f"{scan}::scan_0007/scan_data/image", auto_load=True, slot="CL")
        assert loaded is True
        tool._center_row.setValue(24)
        tool._center_col.setValue(80)
        tool._add_roi("ring")
        tool._compute_current_profiles()
        assert tool._data is not None and tool._rois

        tool.close()

        assert tool._data is None
        assert tool._raw_data is None
        assert tool._rois == []
        assert tool._roi_counter == {"ring": 0, "sector": 0, "circle": 0}
        assert tool._grid_cache is None
        assert tool._polar_cache is None
        assert tool._stack_notes == []
        drawn = tool._profile_curve.getData()[0]
        assert drawn is None or len(drawn) == 0
    finally:
        tool.deleteLater()


def test_the_q_tool_keeps_the_dataset_selection(qapp, scan):
    """The selection is the question. Clearing it would mean re-picking the
    file every time the tool is reopened."""
    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        tool.load_dataset_full_key(
            f"{scan}::scan_0007/scan_data/image", auto_load=True, slot="CL")
        chosen = tool._cl_combo.currentText()
        assert chosen

        tool.close()

        assert tool._cl_combo.currentText() == chosen
    finally:
        tool.deleteLater()


def test_the_q_tool_can_be_used_again_after_being_closed(qapp, scan):
    """A reset that leaves the tool unusable would be worse than the bug."""
    from src.gui.q_calibration_tool import QCalibrationTool

    tool = QCalibrationTool(opened_files=(scan,), dataset_full_keys_2d=[])
    try:
        key = f"{scan}::scan_0007/scan_data/image"
        tool.load_dataset_full_key(key, auto_load=True, slot="CL")
        tool.close()
        tool.show()

        assert tool.load_dataset_full_key(key, auto_load=True, slot="CL") is True
        assert tool._data is not None
    finally:
        tool.deleteLater()


# ── The generic display sweep ─────────────────────────────────────────── #

class _Holder:
    """Stands in for a tool: the sweep only looks at attribute values."""


def test_the_sweep_blanks_images_and_curves_without_removing_them(qapp):
    from src.gui._shared import clear_tool_displays

    holder = _Holder()
    plot = pg.PlotWidget()
    try:
        holder.img = pg.ImageItem(np.ones((4, 4)))
        holder.curve = plot.plot([1, 2, 3], [4, 5, 6])
        holder.not_a_display = "left alone"

        assert clear_tool_displays(holder) == 2

        assert holder.img.image is None
        assert len(holder.curve.getData()[0] or []) == 0
        assert holder.not_a_display == "left alone"
        # The items are still there to draw into next time.
        assert holder.curve in plot.plotItem.listDataItems()
    finally:
        plot.close()
        plot.deleteLater()


def test_the_sweep_survives_an_item_that_will_not_clear(qapp):
    """One stuck view must not stop the rest being cleared."""
    from src.gui._shared import clear_tool_displays

    def explode():
        raise RuntimeError("stuck")

    holder = _Holder()
    holder.broken = pg.ImageItem(np.ones((2, 2)))
    holder.broken.clear = explode
    holder.good = pg.ImageItem(np.ones((2, 2)))

    assert clear_tool_displays(holder) == 1
    assert holder.good.image is None


# ── The heavy tools, exercised for real ───────────────────────────────── #
#
# The AST tests above prove the wiring; they cannot prove the reset runs. And
# closeEvent deliberately swallows exceptions so a window is always closable,
# which means a reset that throws would fail silently and leave the data in
# place. These call reset_results() directly, where a failure is loud.

@pytest.mark.parametrize("module,cls", ARRAY_TOOLS)
def test_the_reset_actually_clears_every_array(qapp, module, cls):
    import importlib

    tool_cls = getattr(importlib.import_module(f"src.gui.{module}"), cls)
    tool = tool_cls(opened_files=(), dataset_full_keys_2d=[])
    try:
        for name in tool_cls._DATA_ATTRS:
            setattr(tool, name, np.ones((4, 4)))

        tool.reset_results()

        still_held = [n for n in tool_cls._DATA_ATTRS if getattr(tool, n) is not None]
        assert still_held == []
    finally:
        tool.deleteLater()


def test_the_cdi_reset_clears_its_error_curves_and_masks(qapp):
    from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

    tool = CDIReconstructionTool(opened_files=(), dataset_full_keys_2d=[])
    try:
        tool._result_errs = [1.0, 2.0]
        tool._result_errs_cr = [3.0]
        tool._single_dataset_mode = True
        tool._bad_pixel_shift_x = 5

        tool.reset_results()

        assert tool._result_errs == []
        assert tool._result_errs_cr == []
        assert tool._mask_groups == []
        assert tool._single_dataset_mode is False
        assert tool._bad_pixel_shift_x == 0
    finally:
        tool.deleteLater()


@pytest.mark.parametrize("module,cls", ARRAY_TOOLS)
def test_the_reset_is_safe_to_run_twice(qapp, module, cls):
    """Closing an already-clear tool must not raise."""
    import importlib

    tool_cls = getattr(importlib.import_module(f"src.gui.{module}"), cls)
    tool = tool_cls(opened_files=(), dataset_full_keys_2d=[])
    try:
        tool.reset_results()
        tool.reset_results()
    finally:
        tool.deleteLater()

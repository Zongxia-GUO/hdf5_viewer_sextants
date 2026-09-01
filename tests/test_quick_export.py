"""Tests for quick export from the data viewers.

Regression: the viewers' save buttons delegated to the main window's export
action via ``self.window()``. Inside a tool dialog there is no main window in
that parent chain, so the button silently did nothing at all.
"""

import pathlib

from typing import Any

import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from src.gui.export_naming import export_stem
from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
from src.gui.unified_data_viewer import UnifiedDataViewer
from src.lib_h5.table_format import TABLE_FORMATS, format_from_filter, get_table_format

KEY = "d:/data/scanx_0033.nxs::/scan_0033/scan_data/data_01"


@pytest.fixture
def save_to(tmp_path, monkeypatch):
    """Redirect the save dialog into tmp_path and record the pre-filled name.

    ``seen["pick"]`` may be set to a dialect label to simulate the user choosing
    a different entry in the dialog's filter dropdown.
    """
    seen: dict[str, Any] = {"names": [], "boxes": [], "filters": [], "pick": None}

    def fake_dialog(_parent, _title, default_path, selected_filter, *a, **k):
        seen["names"].append(pathlib.Path(default_path).name)
        seen["filters"].append(selected_filter)
        chosen = seen["pick"] or selected_filter.split(";;")[0]
        name = pathlib.Path(default_path).name
        if any(chosen.startswith(fmt.label) for fmt in TABLE_FORMATS):
            # A table dialect was picked; the suffix follows the dialect.
            name = pathlib.Path(default_path).stem + format_from_filter(chosen).suffix
        return str(tmp_path / name), chosen

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox,
            kind,
            staticmethod(lambda *a, _k=kind, **k: seen["boxes"].append(_k)),
        )
    return seen


# ---------------------------------------------------------------------------
# Default file names
# ---------------------------------------------------------------------------

def test_export_stem_from_a_dataset_key():
    assert export_stem(KEY) == "scanx_0033_data_01"


def test_export_stem_without_a_dataset_part():
    assert export_stem("d:/data/scanx_0033.nxs") == "scanx_0033"


def test_export_stem_falls_back_when_unknown():
    assert export_stem(None, "plot_data") == "plot_data"
    assert export_stem("", "image") == "image"


def test_export_stem_is_filesystem_safe():
    assert export_stem("a b.h5::/grp/da ta:1") == "a_b_da_ta_1"


# ---------------------------------------------------------------------------
# 1-D quick export
# ---------------------------------------------------------------------------

def test_plot_viewer_save_button_writes_a_file(qapp, tmp_path, save_to):
    """The default dialect is tab-separated .txt, shared with the batch export."""
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.arange(10, dtype=float))

    w.btn_save_plot.click()

    assert save_to["names"] == ["scanx_0033_data_01.txt"]
    written = list(tmp_path.glob("*.txt"))
    assert len(written) == 1 and written[0].stat().st_size > 0


def test_no_x_dataset_means_no_x_column(qapp, tmp_path, save_to):
    """Without an X dataset the plot uses the sample index — do not write it out."""
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.array([1.5, 2.5, 3.5]))

    w.btn_save_plot.click()

    lines = (tmp_path / "scanx_0033_data_01.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Y"
    assert lines[1:] == ["1.5", "2.5", "3.5"]


def test_multi_curve_without_x_writes_only_the_y_columns(qapp, tmp_path, save_to):
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.column_stack([np.array([1.0, 2.0]), np.array([3.0, 4.0])]))

    w.btn_save_plot.click()

    lines = (tmp_path / "scanx_0033_data_01.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["Y_Column_1", "Y_Column_2"]
    assert lines[1].split("\t") == ["1", "3"]


def test_plot_quick_export_contains_x_and_y(qapp, tmp_path, save_to):
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.array([1.5, 2.5, 3.5]), np.array([10.0, 20.0, 30.0]))

    w.btn_save_plot.click()

    rows = [
        line.split("\t")
        for line in (tmp_path / "scanx_0033_data_01.txt").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0] == ["X", "Y"]
    assert rows[1] == ["10", "1.5"]


def test_plot_quick_export_honours_the_chosen_dialect(qapp, tmp_path, save_to):
    """Picking CSV2 in the dialog gives a French-Excel file, like the batch export."""
    save_to["pick"] = get_table_format("csv2").label
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.array([1.5, 2.5]), np.array([10.0, 20.0]))

    w.btn_save_plot.click()

    text = (tmp_path / "scanx_0033_data_01.csv").read_text(encoding="utf-8-sig")
    assert text.splitlines()[1] == "10;1,5"


def test_plot_quick_export_writes_the_q_column(qapp, tmp_path, save_to):
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    w.x_data_original = np.array([10.0, 20.0])
    w.btn_convert_to_q.setChecked(True)

    w.btn_save_plot.click()

    header = (tmp_path / "scanx_0033_data_01.txt").read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == ["X", "q", "Y"]


def test_log_axis_does_not_change_the_exported_values(qapp, tmp_path, save_to):
    """Log is an axis rendering mode; the file must stay linear."""
    w = PlotWidget1DEnhanced()
    w.set_source_dataset_key(KEY)
    w.set_data(np.array([1e-9, 1e-6, 1e-3]))
    w.chk_log_y.setChecked(True)

    w.btn_save_plot.click()

    text = (tmp_path / "scanx_0033_data_01.txt").read_text(encoding="utf-8")
    assert "1e-09" in text
    assert "-9" not in text.replace("1e-09", "").replace("1e-06", "")


# ---------------------------------------------------------------------------
# 2-D quick export
# ---------------------------------------------------------------------------

def test_image_viewer_save_button_writes_a_file(qapp, tmp_path, save_to):
    w = ImageView2DEnhanced()
    w.set_source_dataset_key("d:/data/scanx_0033.nxs::/scan_0033/scan_data/image_02")
    w.set_data(np.random.RandomState(0).rand(32, 32))

    w.btn_save_image.click()

    assert save_to["names"] == ["scanx_0033_image_02.png"]
    written = list(tmp_path.glob("*.png"))
    assert len(written) == 1 and written[0].stat().st_size > 0


def test_image_quick_export_follows_the_active_colormap(qapp, tmp_path, save_to):
    """Two different colormaps must not produce the same bytes."""
    data = np.tile(np.linspace(0.0, 1.0, 32), (32, 1))

    w = ImageView2DEnhanced()
    w.set_source_dataset_key("f.h5::a")
    w.set_data(data)
    w.combo_colormap.setCurrentText("viridis")
    w.btn_save_image.click()
    first = (tmp_path / "f_a.png").read_bytes()

    w.combo_colormap.setCurrentText("inferno")
    w.btn_save_image.click()
    second = (tmp_path / "f_a.png").read_bytes()

    assert first != second


# ---------------------------------------------------------------------------
# The actual regression: viewers inside a tool dialog
# ---------------------------------------------------------------------------

def test_viewer_inside_a_plain_parent_still_exports(qapp, tmp_path, save_to):
    """No main window in the parent chain — the old delegate did nothing here."""
    from PyQt6.QtWidgets import QDialog, QVBoxLayout

    host = QDialog()
    layout = QVBoxLayout(host)
    viewer = UnifiedDataViewer(parent=host)
    layout.addWidget(viewer)
    viewer.set_data(np.arange(8, dtype=float), source_dataset_key=KEY)

    inner = viewer.get_current_widget()
    assert isinstance(inner, PlotWidget1DEnhanced)
    assert not hasattr(inner.window(), "_handle_action_export_current")

    inner.btn_save_plot.click()

    assert list(tmp_path.glob("*.txt")), "quick export wrote nothing inside a tool dialog"


def test_unified_viewer_passes_the_source_key_to_the_image_widget(qapp):
    viewer = UnifiedDataViewer()
    viewer.set_data(np.random.RandomState(0).rand(40, 40), source_dataset_key="f.h5::img")

    inner = viewer.get_current_widget()
    assert isinstance(inner, ImageView2DEnhanced)
    assert inner.source_dataset_key == "f.h5::img"

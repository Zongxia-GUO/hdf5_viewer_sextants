"""The batch-export logic must stay usable without a MainWindow.

That independence is the point of the split: the export helpers are plain
functions, and the dialog can be built on its own.
"""

import inspect
import pathlib

import numpy as np
import pytest

from src.gui import batch_export
from src.gui.batch_export import (
    BatchExportDialog,
    BatchTarget,
    adjust_batch_path_for_scan,
    batch_target_labels,
    build_batch_targets,
    curve_export_columns,
    is_curve_export_data,
    safe_export_name,
    table_format_from,
)

PUBLIC_FUNCTIONS = [
    "adjust_batch_path_for_scan",
    "batch_target_labels",
    "build_batch_targets",
    "build_curve_preview_table",
    "curve_export_columns",
    "export_curve_combined_table",
    "export_curve_dataset",
    "export_image_dataset",
    "is_curve_export_data",
    "read_batch_x_data",
    "render_batch_colormapped_rgb",
    "resolve_batch_colormap",
    "safe_export_name",
    "table_format_from",
]


def test_the_module_does_not_import_main_window():
    """A dependency back onto the window would undo the split."""
    import ast

    tree = ast.parse(pathlib.Path(batch_export.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any("main_window" in name for name in imported), sorted(imported)


def test_every_helper_is_a_plain_function_without_self():
    for name in PUBLIC_FUNCTIONS:
        func = getattr(batch_export, name)
        assert inspect.isfunction(func), f"{name} is not a module-level function"
        params = list(inspect.signature(func).parameters)
        assert "self" not in params, f"{name} still takes self"


def test_main_window_no_longer_owns_the_export_helpers():
    from src.gui.main_window import MainWindow

    for name in PUBLIC_FUNCTIONS:
        assert not hasattr(MainWindow, f"_{name}"), f"MainWindow still defines _{name}"


# ---------------------------------------------------------------------------
# The helpers work standalone
# ---------------------------------------------------------------------------

def test_target_expansion_without_any_window(tmp_path):
    files = [(tmp_path / "scanx_0080.h5", "0080"), (tmp_path / "scanx_0081.h5", "0081")]
    targets = build_batch_targets(files, ["det1", "det2"])

    assert len(targets) == 4
    assert all(isinstance(t, BatchTarget) for t in targets)
    assert [t.ds_path for t in targets] == ["det1", "det2", "det1", "det2"]


def test_path_and_name_helpers():
    assert adjust_batch_path_for_scan("scan_0080/data", "0085") == "scan_0085/data"
    assert safe_export_name(pathlib.Path("scanx_0033.h5"), "/a/b/data_01") == "scanx_0033_a_b_data_01"
    assert batch_target_labels(["only"]) == {"only": ""}


def test_shape_helpers():
    assert is_curve_export_data(np.zeros(10)) is True
    assert is_curve_export_data(np.zeros((10, 3))) is True
    assert is_curve_export_data(np.zeros((10, 400))) is False
    assert curve_export_columns(np.zeros(5)).shape == (5, 1)


def test_table_format_from_tolerates_a_missing_key():
    assert table_format_from({}).key == "txt"
    assert table_format_from({"table_format": "csv2"}).key == "csv2"


def test_shape_helper_rejects_an_unsupported_array():
    with pytest.raises(ValueError, match="Not a curve"):
        curve_export_columns(np.zeros((4, 4, 4)))


# ---------------------------------------------------------------------------
# The dialog builds on its own
# ---------------------------------------------------------------------------

def test_dialog_can_be_built_without_a_main_window(qapp, tmp_path):
    dialog = BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0080"],
        dataset_path="entry/data",
        sample_data=np.arange(20, dtype=float),
        data_kind="curve",
        preview_x_loader=lambda settings, n: None,
        preview_curve_loader=lambda settings, **caps: (
            np.arange(20, dtype=float).reshape(-1, 1), ["Y_1"]
        ),
    )

    assert dialog._data_kind == "curve"
    assert dialog.table_format().key == "txt"
    assert dialog.settings()["output_dir"] == tmp_path

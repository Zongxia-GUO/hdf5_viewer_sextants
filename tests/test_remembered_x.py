"""The X chosen in the tree carries into windows opened afterwards.

Picking an X is a decision about the measurement, not about one window, so the
export dialogs and the plot start from it rather than making the user find the
same dataset again. It is held in :mod:`src.gui.x_target` rather than reached
for through parent chains — a dialog opened from a tool has no path back to the
main window, which is the bug that keeps recurring here.
"""

import h5py
import numpy as np
import pytest

from src.gui.x_target import clear_x_targets, remember_x_dataset, remembered_x_dataset

SWEEP = np.linspace(-7000.0, 7000.0, 10)


@pytest.fixture(autouse=True)
def clean_register():
    clear_x_targets()
    yield
    clear_x_targets()


@pytest.fixture
def x_key(tmp_path):
    path = tmp_path / "scanx_0033.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan_0033/scan_data")
        grp.create_dataset("actuator_1_1", data=SWEEP)
        grp.create_dataset("short", data=np.arange(3.0))
    return f"{path}::scan_0033/scan_data/actuator_1_1"


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

def test_nothing_is_remembered_to_begin_with():
    assert remembered_x_dataset() == ""


def test_the_choice_is_kept_and_trimmed():
    remember_x_dataset("  a.h5::x  ")
    assert remembered_x_dataset() == "a.h5::x"


def test_the_tree_records_its_choice(qapp, x_key, monkeypatch):
    """_set_dataset_as_x is the only place the user makes this decision."""
    from src.gui.main_window import MainWindow

    class _Host:
        _set_dataset_as_x = MainWindow._set_dataset_as_x
        _x_dataset_path = MainWindow._x_dataset_path

        def __init__(self):
            self._x_dataset_key = None

        def _current_plot_widget_1d(self):
            return None

        def _set_status_text(self, text=""):
            pass

    file_part, ds_path = x_key.split("::")
    _Host()._set_dataset_as_x([file_part] + ds_path.split("/"))

    assert remembered_x_dataset() == x_key


# ---------------------------------------------------------------------------
# The export dialogs start from it
# ---------------------------------------------------------------------------

def test_the_comparison_export_dialog_starts_from_it(qapp, x_key):
    from src.gui.data_comparison import ComparisonExportDialog

    remember_x_dataset(x_key)
    dialog = ComparisonExportDialog()
    dialog.close()

    assert dialog.le_x_path.text() == x_key
    assert dialog.x_key() == x_key


def test_the_comparison_export_dialog_is_empty_without_one(qapp):
    from src.gui.data_comparison import ComparisonExportDialog

    dialog = ComparisonExportDialog()
    dialog.close()

    assert dialog.le_x_path.text() == ""


def test_the_calculator_export_dialog_starts_from_it(qapp, x_key):
    from src.gui.data_calculator_enhanced import ResultExportDialog

    remember_x_dataset(x_key)
    dialog = ResultExportDialog(
        None, opened_files=tuple(), dataset_full_keys_1d=[],
        preferred_x_key=None, expression="",
    )
    dialog.close()

    assert dialog.chk_export_x.isChecked()
    assert dialog.x_key() == x_key


def test_the_result_viewers_own_x_wins_over_the_remembered_one(qapp, x_key):
    """The calculator's own choice is more specific than the tree's."""
    from src.gui.data_calculator_enhanced import ResultExportDialog

    remember_x_dataset(x_key)
    own = "other.h5::other/x"
    dialog = ResultExportDialog(
        None, opened_files=tuple(), dataset_full_keys_1d=[],
        preferred_x_key=own, expression="",
    )
    dialog.close()

    assert dialog.x_key() == own


def test_the_batch_export_dialog_starts_from_it(qapp, x_key, tmp_path):
    """It addresses a path inside each scan file, so it keeps the dataset half."""
    from src.gui.batch_export import BatchExportDialog
    from src.gui.main_window import MainWindow

    class _Host:
        _x_dataset_path = MainWindow._x_dataset_path

        def __init__(self, key):
            self._x_dataset_key = key

    dialog = BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0033"],
        dataset_path="/scan_0033/scan_data/data_01",
        sample_data=np.arange(10.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        default_x_path=_Host(x_key)._x_dataset_path(),
    )
    dialog.close()

    assert dialog.le_x_path.text() == "scan_0033/scan_data/actuator_1_1"


# ---------------------------------------------------------------------------
# The plot starts from it
# ---------------------------------------------------------------------------

def plot_with(series):
    from src.gui.plot_dialog import PlotPanel

    panel = PlotPanel(series=series)
    panel.close()
    return panel


def test_an_index_plotted_curve_adopts_the_remembered_x(qapp, x_key):
    from src.gui.plot_series import Series

    remember_x_dataset(x_key)
    panel = plot_with([Series("counts", np.arange(10.0))])

    np.testing.assert_allclose(panel.axes.lines[0].get_xdata(), SWEEP)
    assert panel.le_x_source.text() == x_key
    assert panel.le_x_label.text() == "actuator_1_1"


def test_a_curve_that_already_has_an_x_is_left_alone(qapp, x_key):
    """The caller knew what X it wanted; the remembered one is only a default."""
    from src.gui.plot_series import Series

    remember_x_dataset(x_key)
    own = np.arange(10.0) * 3
    panel = plot_with([Series("counts", np.arange(10.0), own)])

    np.testing.assert_allclose(panel.axes.lines[0].get_xdata(), own)


def test_a_length_that_does_not_fit_is_passed_over_quietly(qapp, x_key, monkeypatch):
    """The user asked for a plot, not for this, so it must not interrupt."""
    from src.gui.plot_series import Series

    boxes: list[str] = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            f"src.gui.plot_dialog.QMessageBox.{kind}",
            staticmethod(lambda *a, **k: boxes.append(kind)),
        )

    remember_x_dataset(x_key)
    panel = plot_with([Series("counts", np.arange(4.0))])

    assert boxes == []
    np.testing.assert_allclose(panel.axes.lines[0].get_xdata(), np.arange(4.0))


def test_an_unreadable_remembered_x_is_passed_over_quietly(qapp, monkeypatch):
    from src.gui.plot_series import Series

    boxes: list[str] = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            f"src.gui.plot_dialog.QMessageBox.{kind}",
            staticmethod(lambda *a, **k: boxes.append(kind)),
        )

    remember_x_dataset("C:/gone/missing.h5::a/b")
    panel = plot_with([Series("counts", np.arange(4.0))])

    assert boxes == []
    assert panel.le_x_source.text() == ""


def test_every_curve_must_fit_before_any_is_rebased(qapp, x_key):
    """Half a figure on one X and half on another would be unreadable."""
    from src.gui.plot_series import Series

    remember_x_dataset(x_key)
    panel = plot_with([Series("a", np.arange(10.0)), Series("b", np.arange(4.0))])

    np.testing.assert_allclose(panel.axes.lines[0].get_xdata(), np.arange(10.0))


def test_reset_x_goes_back_to_the_adopted_one(qapp, x_key):
    """Adopting sets the baseline; Reset must not undo it into an index axis."""
    from src.gui.plot_series import Series

    remember_x_dataset(x_key)
    panel = plot_with([Series("counts", np.arange(10.0))])
    panel.reset_x()

    np.testing.assert_allclose(panel.axes.lines[0].get_xdata(), SWEEP)


def test_the_batch_plot_panel_does_not_reach_for_it(qapp, x_key, tmp_path):
    """That dialog has its own X field below the tabs; two would disagree."""
    from src.gui.batch_export import BatchExportDialog

    remember_x_dataset(x_key)
    table = np.column_stack([np.arange(10.0), np.arange(10.0)])
    dialog = BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0033"],
        dataset_path="/scan_0033/scan_data/data_01",
        sample_data=np.arange(10.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _s, **caps: (table, ["a_Y", "b_Y"]),
    )
    dialog.close()

    assert dialog.plot_panel.le_x_source.text() == ""
    np.testing.assert_allclose(dialog.plot_panel.axes.lines[0].get_xdata(), np.arange(10.0))

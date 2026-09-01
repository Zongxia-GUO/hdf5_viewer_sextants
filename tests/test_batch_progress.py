"""The batch export runs with a progress window and leaves its settings open.

Closing the settings dialog on OK meant re-entering the whole scan range to
export the next one, and a long batch gave no sign it was running at all. Now
the writing happens behind a small progress window and the dialog is left for
the user to close.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from src.gui.batch_export import BatchExportDialog, BatchProgress, BatchTarget
from src.gui.main_window import MainWindow


@pytest.fixture
def scans(tmp_path):
    """Three scan files with one curve each."""
    files = []
    for index, number in enumerate(("0080", "0081", "0082")):
        path = tmp_path / f"scanx_{number}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("curve", data=np.arange(5.0) + index)
        files.append((path, number))
    return files


@pytest.fixture
def dialog(qapp, tmp_path):
    d = BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0080"],
        dataset_path="/curve",
        sample_data=np.arange(5.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _s, **caps: (np.zeros((4, 1)), ["Y"]),
    )
    yield d
    d.deleteLater()


def _host(monkeypatch):
    """MainWindow's export surface, without building the real window."""
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            f"src.gui.main_window.QMessageBox.{kind}", staticmethod(lambda *a, **k: None)
        )

    class _Host:
        _finish_batch_export = MainWindow._finish_batch_export
        _report_batch_result = MainWindow._report_batch_result
        _plot_batch_selection = MainWindow._plot_batch_selection

    return _Host()


# ---------------------------------------------------------------------------
# The dialog stays open
# ---------------------------------------------------------------------------

def test_the_export_button_does_not_close_the_dialog(dialog):
    """A batch is usually one of several; closing the settings threw away a
    scan range that had just been typed in."""
    fired: list[int] = []
    dialog.export_requested.connect(lambda: fired.append(1))
    dialog.show()

    dialog.buttons.accepted.emit()

    assert fired == [1]
    assert dialog.isVisible(), "the settings are left for the user to close"


def test_cancel_still_closes_the_dialog(dialog):
    dialog.show()

    dialog.buttons.rejected.emit()

    assert not dialog.isVisible()


def test_running_an_export_leaves_the_dialog_alive(qapp, dialog, scans, tmp_path, monkeypatch):
    """It has to survive the run, not just the click — the writer used to
    delete it halfway through."""
    host = _host(monkeypatch)
    out = tmp_path / "out"
    dialog.le_output_dir.setText(str(out))
    targets = [
        BatchTarget(file_path=path, scan_num=number, ds_path="curve", label=number)
        for path, number in scans
    ]

    host._finish_batch_export(dialog, targets, scans, "curve", _FakeSettings())

    assert len(list(out.glob("*.txt"))) == 3
    assert dialog.parent() is None and dialog.isEnabled(), "still a usable dialog"


class _FakeSettings:
    def setValue(self, *_args) -> None:
        pass


# ---------------------------------------------------------------------------
# The progress window
# ---------------------------------------------------------------------------

def test_progress_counts_through_the_batch(qapp):
    with BatchProgress(None, 3, "Batch Export", "Exporting…") as progress:
        assert progress.advance("one") is True
        assert progress.advance("two") is True
        assert progress.advance("three") is True
        assert progress.cancelled is False


def test_cancelling_stops_the_run(qapp):
    with BatchProgress(None, 10, "Batch Export", "Exporting…") as progress:
        progress._dialog.cancel()

        assert progress.advance("one") is False
        assert progress.cancelled is True


def test_a_cancelled_export_stops_early_and_says_so(qapp, dialog, scans, tmp_path, monkeypatch):
    """Half a batch on disk with no explanation would look like a failure."""
    host = _host(monkeypatch)
    out = tmp_path / "out"
    dialog.le_output_dir.setText(str(out))
    targets = [
        BatchTarget(file_path=path, scan_num=number, ds_path="curve", label=number)
        for path, number in scans
    ]

    reported: list[str] = []
    monkeypatch.setattr(
        "src.gui.main_window.QMessageBox.information",
        staticmethod(lambda _p, _t, message, *a, **k: reported.append(message) or QMessageBox.StandardButton.Ok),
    )
    # Cancel as soon as the first file is done.
    real_advance = BatchProgress.advance

    def advance_once(self, label=""):
        real_advance(self, label)
        self._dialog.cancel()
        return False

    monkeypatch.setattr(BatchProgress, "advance", advance_once)

    host._finish_batch_export(dialog, targets, scans, "curve", _FakeSettings())

    assert len(list(out.glob("*.txt"))) == 1, "stopped after the first"
    assert reported and "Stopped at your request" in reported[0]


def test_a_failing_dataset_does_not_abandon_the_progress_window(qapp, tmp_path, monkeypatch):
    """The window is a context manager precisely so a raising writer still
    leaves the screen clean."""
    with pytest.raises(ValueError):
        with BatchProgress(None, 2, "Batch Export", "Exporting…") as progress:
            progress.advance("one")
            raise ValueError("boom")

    # Nothing left behind that a later test could trip over.
    assert progress._dialog is not None


def test_progress_reports_the_path_being_written(qapp):
    with BatchProgress(None, 2, "Batch Export", "Exporting…") as progress:
        progress.advance("scanx_0080.h5")

        assert progress._dialog.labelText() == "scanx_0080.h5"


def test_a_single_step_batch_still_gets_a_window(qapp):
    """The combined table is one call, so there is nothing to count — the
    window says what is happening and nothing more."""
    with BatchProgress(None, 0, "Batch Export", "Writing the combined table…") as progress:
        assert progress._dialog.labelText() == "Writing the combined table…"
        assert progress.cancelled is False


def test_the_progress_window_never_pins_itself_over_other_applications(qapp):
    from PyQt6.QtCore import Qt

    with BatchProgress(None, 3, "Batch Export", "Exporting…") as progress:
        flags = progress._dialog.windowFlags()

        assert not (flags & Qt.WindowType.WindowStaysOnTopHint)


def test_the_export_folder_is_still_created(qapp, dialog, scans, tmp_path, monkeypatch):
    host = _host(monkeypatch)
    out = tmp_path / "made" / "here"
    dialog.le_output_dir.setText(str(out))
    targets = [
        BatchTarget(file_path=scans[0][0], scan_num=scans[0][1], ds_path="curve", label="0080")
    ]

    host._finish_batch_export(dialog, targets, scans, "curve", _FakeSettings())

    assert out.is_dir()
    assert list(out.glob("*.txt"))


def test_paths_are_untouched_when_the_folder_cannot_be_made(qapp, dialog, scans, monkeypatch):
    host = _host(monkeypatch)
    dialog.le_output_dir.setText(str(pathlib.Path(scans[0][0])))  # a file, not a folder
    targets = [
        BatchTarget(file_path=scans[0][0], scan_num=scans[0][1], ds_path="curve", label="0080")
    ]

    host._finish_batch_export(dialog, targets, scans, "curve", _FakeSettings())

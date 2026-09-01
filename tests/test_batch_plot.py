"""The batch dialog's two pages: export the selection, or draw it.

Both pages read one table, so a figure can never list curves the export would
not write. What they share — the scans, the datasets, the X axis — sits below
the tabs; only the destination differs.
"""

import pathlib

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QGroupBox, QLabel

from src.gui.batch_export import BatchExportDialog

HEADERS = ["X", "scan_0080_data_01", "scan_0081_data_01"]
TABLE = np.column_stack([
    np.linspace(0.0, 1.0, 12),
    np.arange(12.0),
    np.arange(12.0) * 2,
])


@pytest.fixture
def dialog(qapp):
    d = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0080", "0081"],
        dataset_path="/scan/scan_data/data_01",
        sample_data=np.arange(12.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _settings, **caps: (TABLE, HEADERS),
    )
    d.close()
    return d


def tab_titles(d):
    return [d.tabs.tabText(i) for i in range(d.tabs.count())]


def ok_text(d):
    from PyQt6.QtWidgets import QDialogButtonBox

    return d.buttons.button(QDialogButtonBox.StandardButton.Ok).text()


# ---------------------------------------------------------------------------
# The two pages
# ---------------------------------------------------------------------------

def test_the_window_can_go_behind_the_application(dialog):
    """Non-modal so the tree stays usable — not pinned above every program."""
    assert not (dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal


def test_export_comes_first_and_plot_second(dialog):
    assert tab_titles(dialog) == ["Export", "Plot"]
    assert dialog.tabs.currentIndex() == 0


def test_the_open_page_decides_what_ok_does(dialog):
    assert dialog.action() == "export"
    assert ok_text(dialog) == "Export"

    dialog.tabs.setCurrentIndex(1)

    assert dialog.action() == "plot"
    assert ok_text(dialog) == "Save Figure", "OK must never be a guess between the two"


def test_the_plot_page_offers_copy_save_figure_and_cancel(dialog):
    """The same three the standalone Plot window ends on."""
    dialog.tabs.setCurrentIndex(1)
    shown = [b.text() for b in dialog.buttons.buttons() if b.isVisibleTo(dialog.buttons)]

    assert shown == ["Save Figure", "Cancel", "Copy"]


def test_copy_belongs_to_the_figure_only(dialog):
    """A table goes to a file, not to the clipboard."""
    assert not dialog.btn_copy_figure.isVisibleTo(dialog.buttons)

    dialog.tabs.setCurrentIndex(1)
    assert dialog.btn_copy_figure.isVisibleTo(dialog.buttons)

    dialog.tabs.setCurrentIndex(0)
    assert not dialog.btn_copy_figure.isVisibleTo(dialog.buttons)


def test_copy_puts_the_embedded_figure_on_the_clipboard(dialog, qapp):
    from PyQt6.QtWidgets import QApplication

    dialog.tabs.setCurrentIndex(1)
    QApplication.clipboard().clear()

    dialog.btn_copy_figure.click()

    assert not QApplication.clipboard().image().isNull()


def test_copy_does_not_close_the_dialog(dialog):
    """It is an action, not an answer: the ActionRole must not accept."""
    finished: list[int] = []
    dialog.finished.connect(finished.append)

    dialog.tabs.setCurrentIndex(1)
    dialog.btn_copy_figure.click()

    assert finished == []


def test_the_plot_page_holds_a_real_figure(dialog):
    """Not a button that opens one: the page is the preview."""
    from src.gui.plot_dialog import PlotPanel

    assert isinstance(dialog.plot_panel, PlotPanel)
    assert dialog.plot_panel.canvas is not None


def test_the_figure_draws_the_curves_the_table_holds(dialog):
    labels = [line.get_label() for line in dialog.plot_panel.axes.lines]

    assert labels == ["scan_0080_data_01", "scan_0081_data_01"], "the X column is not a curve"
    assert "2 curve(s)" in dialog.lbl_plot_summary.text()


def test_the_embedded_panel_hides_its_own_x_field(dialog):
    """Two X fields on one screen would disagree the moment either was used."""
    assert not dialog.plot_panel.x_row.isVisibleTo(dialog.plot_panel)


def test_the_curves_are_drawn_against_the_x_column(dialog):
    drawn_x = dialog.plot_panel.axes.lines[0].get_xdata()
    np.testing.assert_allclose(drawn_x, TABLE[:, 0])


def test_without_an_x_column_the_curves_fall_back_to_the_index(qapp):
    d = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0080"],
        dataset_path="/scan/scan_data/data_01",
        sample_data=np.arange(12.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _s, **caps: (TABLE[:, 1:], HEADERS[1:]),
    )
    d.close()

    np.testing.assert_allclose(d.plot_panel.axes.lines[0].get_xdata(), np.arange(12.0))


def test_the_figure_redraws_when_the_shared_x_changes(dialog):
    """The X lives below the tabs because it drives both pages."""
    calls: list[int] = []
    dialog._preview_curve_loader = lambda _s, **caps: (calls.append(1) or (TABLE, HEADERS))

    dialog.le_x_path.setText("scan/scan_data/actuator_1_1")

    assert calls, "the figure is rebuilt from the table, not left stale"
    assert len(dialog.plot_panel.axes.lines) == 2


# ---------------------------------------------------------------------------
# The shared settings really are shared
# ---------------------------------------------------------------------------

def test_the_x_controls_are_not_inside_either_page(dialog):
    """Both pages need the X, so it must not be owned by one of them."""
    export_page = dialog.tabs.widget(0)
    plot_page = dialog.tabs.widget(1)

    for page in (export_page, plot_page):
        assert dialog.le_x_path not in page.findChildren(type(dialog.le_x_path))


def test_the_destination_controls_are_all_in_the_batch_box(dialog):
    """Folder, Output and Format say where the batch goes, so they sit together."""
    pages = [dialog.tabs.widget(i) for i in range(dialog.tabs.count())]

    for page in pages:
        assert dialog.le_output_dir not in page.findChildren(type(dialog.le_output_dir))
        assert dialog.cb_curve_output_mode not in page.findChildren(QComboBox)
        assert dialog.cb_plot_output_mode not in page.findChildren(QComboBox)


def test_the_batch_box_asks_what_before_where(dialog):
    """Range names the batch; Folder is where it lands — that is the reading order."""
    box = dialog.buttons.parent().findChild(QGroupBox, options=Qt.FindChildOption.FindDirectChildrenOnly)
    labels = [w.text() for w in box.findChildren(QLabel) if w.text().endswith(":")]

    assert labels.index("Range:") < labels.index("Folder:")


def test_the_buttons_sit_outside_the_batch_box(dialog):
    """The box describes the batch; the buttons act on it."""
    box = dialog.findChild(QGroupBox)

    assert dialog.buttons not in box.findChildren(type(dialog.buttons))
    assert dialog.buttons.parent() is dialog


def test_the_output_row_follows_the_open_page(dialog):
    assert dialog.output_stack.currentIndex() == 0

    dialog.tabs.setCurrentIndex(1)
    assert dialog.output_stack.currentIndex() == 1, "the plot page gets the figure controls"

    dialog.tabs.setCurrentIndex(0)
    assert dialog.output_stack.currentIndex() == 0


# ---------------------------------------------------------------------------
# One figure, or one per scan — the Plot page's twin of Single/Combined
# ---------------------------------------------------------------------------

def test_overlaying_the_range_is_the_default(dialog):
    """A batch plot exists to compare scans; per-scan is the deliberate choice."""
    from src.gui.batch_export import PLOT_MODE_COMBINED

    assert dialog.plot_output_mode() == PLOT_MODE_COMBINED
    assert dialog.settings()["plot_output_mode"] == PLOT_MODE_COMBINED
    assert "one .png file" in dialog.lbl_plot_summary.text()


def test_png_is_the_default_image_format(dialog):
    """A plot is line art, which is what JPG's block transform handles worst."""
    from src.gui.batch_export import PLOT_FORMATS

    assert dialog.plot_format() == "PNG"
    assert [dialog.cb_plot_format.itemText(i)
            for i in range(dialog.cb_plot_format.count())] == list(PLOT_FORMATS)


def test_the_chosen_format_shows_up_in_the_summary(dialog):
    dialog.cb_plot_format.setCurrentText("JPG")

    assert dialog.settings()["plot_format"] == "JPG"
    assert ".jpg" in dialog.lbl_plot_summary.text()


def test_the_plot_preview_follows_its_own_output_not_the_export_page(qapp):
    """One page's control must not silently reshape the other's preview."""
    from src.gui.batch_export import PLOT_MODE_PER_SCAN

    seen: list[str] = []

    def loader(settings, **caps):
        seen.append(str(settings["curve_output_mode"]))
        return TABLE, HEADERS

    d = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0080", "0081"],
        dataset_path="/scan/scan_data/data_01",
        sample_data=np.arange(12.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=loader,
    )
    d.close()

    # The export page is on "Single files"; the plot page defaults to combined.
    assert d.cb_curve_output_mode.currentText() == "Single files"
    assert seen[-1] == "Combined file", "the figure shows the whole range"

    seen.clear()
    d.cb_plot_output_mode.setCurrentText(PLOT_MODE_PER_SCAN)
    assert seen == ["Single files"], "per-scan previews one scan"

    seen.clear()
    d.cb_curve_output_mode.setCurrentText("Combined file")
    assert seen[-1] == "Single files", "the plot page keeps its own choice"


def test_the_export_preview_ignores_the_plot_output(dialog):
    from src.gui.batch_export import PLOT_MODE_PER_SCAN

    dialog.cb_plot_output_mode.setCurrentText(PLOT_MODE_PER_SCAN)

    assert dialog.settings()["curve_output_mode"] == "Single files"
    assert dialog.plot_settings()["curve_output_mode"] == "Single files"

    dialog.cb_curve_output_mode.setCurrentText("Combined file")
    assert dialog.settings()["curve_output_mode"] == "Combined file"
    assert dialog.plot_settings()["curve_output_mode"] == "Single files"


def test_choosing_per_scan_says_how_many_files(dialog):
    from src.gui.batch_export import PLOT_MODE_PER_SCAN

    dialog.cb_plot_output_mode.setCurrentText(PLOT_MODE_PER_SCAN)

    assert dialog.settings()["plot_output_mode"] == PLOT_MODE_PER_SCAN
    # Two scans in the range. Only one figure can be previewed, so the page has
    # to say what OK will actually write.
    assert "Writes 2 .png file(s)" in dialog.lbl_plot_summary.text()


# ---------------------------------------------------------------------------
# Writing the figures
# ---------------------------------------------------------------------------

def _plot_host(monkeypatch, table=None):
    from src.gui.main_window import MainWindow

    monkeypatch.setattr(
        "src.gui.main_window.build_curve_preview_table",
        table or (lambda targets, *a, **k: (TABLE, HEADERS)),
    )
    monkeypatch.setattr(
        "src.gui.main_window.QMessageBox.information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        "src.gui.main_window.QMessageBox.warning", staticmethod(lambda *a, **k: None)
    )

    class _Host:
        _plot_batch_selection = MainWindow._plot_batch_selection
        _write_batch_figures = MainWindow._write_batch_figures
        _batch_targets_by_scan = staticmethod(MainWindow._batch_targets_by_scan)
        _report_batch_result = MainWindow._report_batch_result

    return _Host()


def _targets(*scans):
    from src.gui.batch_export import BatchTarget

    return [
        BatchTarget(
            file_path=pathlib.Path(f"scanx_{s}.h5"),
            scan_num=s,
            ds_path="scan/scan_data/data_01",
            label=s,
        )
        for s in scans
    ]


def test_a_combined_figure_holds_every_scan_not_just_the_first(qapp, tmp_path, monkeypatch, dialog):
    """The bug: the writer read the Export page's Output, which says one file
    per scan by default, so the combined figure came out with one curve."""
    from src.gui.batch_export import PLOT_MODE_COMBINED

    seen: list[dict] = []

    def table(targets, _files, settings, **caps):
        seen.append({"targets": len(targets), "mode": settings["curve_output_mode"], **caps})
        return TABLE, HEADERS

    host = _plot_host(monkeypatch, table=table)

    host._plot_batch_selection(
        dialog, _targets("0080", "0081", "0082"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_COMBINED,
         "plot_format": "PNG", "curve_output_mode": "Combined file"},
    )

    assert seen[0]["targets"] == 3, "all three scans reached the table"
    assert seen[0]["mode"] == "Combined file"


def test_the_written_figure_is_never_truncated(qapp, tmp_path, monkeypatch, dialog):
    """The table builder caps at 5 scans and 200 rows for the on-screen preview;
    a file built with those caps would silently lose data."""
    from src.gui.batch_export import PLOT_MODE_COMBINED

    caps_seen: list[dict] = []

    def table(_targets, _files, _settings, **caps):
        caps_seen.append(caps)
        return TABLE, HEADERS

    host = _plot_host(monkeypatch, table=table)

    host._plot_batch_selection(
        dialog, _targets("0080"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_COMBINED, "plot_format": "PNG"},
    )

    assert caps_seen[0] == {"max_targets": None, "max_rows": None}


def test_the_action_decides_which_settings_the_writer_gets(qapp, tmp_path, monkeypatch):
    """_finish_batch_export must hand the plot path the Plot page's settings."""
    from PyQt6.QtCore import QSettings

    from src.gui.batch_export import PLOT_MODE_COMBINED
    from src.gui.main_window import MainWindow

    handed: list[dict] = []

    class _Host:
        _finish_batch_export = MainWindow._finish_batch_export

        def _plot_batch_selection(self, _dialog, _targets, _files, settings):
            handed.append(settings)

    d = BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0080", "0081"],
        dataset_path="/scan/scan_data/data_01",
        sample_data=np.arange(12.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
        preview_curve_loader=lambda _s, **caps: (TABLE, HEADERS),
    )
    d.close()
    d.tabs.setCurrentIndex(1)
    assert d.cb_curve_output_mode.currentText() == "Single files", "the trap"
    assert d.plot_output_mode() == PLOT_MODE_COMBINED

    _Host()._finish_batch_export(d, [], [], "curve", QSettings())

    assert handed[0]["curve_output_mode"] == "Combined file"


def test_the_combined_mode_writes_one_file(qapp, tmp_path, monkeypatch, dialog):
    from src.gui.batch_export import PLOT_MODE_COMBINED

    host = _plot_host(monkeypatch)

    host._plot_batch_selection(
        dialog, _targets("0080", "0081", "0082"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_COMBINED, "plot_format": "PNG"},
    )

    written = list(tmp_path.glob("*.png"))
    assert len(written) == 1
    assert "0080-0082" in written[0].name, "the span is in the name"
    assert written[0].stat().st_size > 0


def test_per_scan_writes_one_file_for_each(qapp, tmp_path, monkeypatch, dialog):
    from src.gui.batch_export import PLOT_MODE_PER_SCAN

    seen_groups: list[list] = []

    def table(targets, *_a, **_k):
        seen_groups.append([t.scan_num for t in targets])
        return TABLE, HEADERS

    host = _plot_host(monkeypatch, table=table)

    host._plot_batch_selection(
        dialog, _targets("0080", "0081", "0082"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_PER_SCAN, "plot_format": "PNG"},
    )

    assert len(list(tmp_path.glob("*.png"))) == 3
    assert seen_groups == [["0080"], ["0081"], ["0082"]], "each figure gets only its own scan"


def test_several_datasets_of_one_scan_share_its_figure(qapp, tmp_path, monkeypatch, dialog):
    from src.gui.batch_export import PLOT_MODE_PER_SCAN, BatchTarget

    host = _plot_host(monkeypatch)
    targets = [
        BatchTarget(file_path=pathlib.Path("a.h5"), scan_num="0080", ds_path="a", label="a"),
        BatchTarget(file_path=pathlib.Path("a.h5"), scan_num="0080", ds_path="b", label="b"),
        BatchTarget(file_path=pathlib.Path("b.h5"), scan_num="0081", ds_path="a", label="a"),
    ]

    host._plot_batch_selection(
        dialog, targets, [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_PER_SCAN, "plot_format": "PNG"},
    )

    assert len(list(tmp_path.glob("*.png"))) == 2, "one per scan, not one per dataset"


def test_the_jpg_choice_reaches_the_file(qapp, tmp_path, monkeypatch, dialog):
    from src.gui.batch_export import PLOT_MODE_COMBINED

    host = _plot_host(monkeypatch)

    host._plot_batch_selection(
        dialog, _targets("0080"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_COMBINED, "plot_format": "JPG"},
    )

    assert len(list(tmp_path.glob("*.jpg"))) == 1
    assert list(tmp_path.glob("*.png")) == []


def test_the_written_figure_carries_the_settings_on_screen(qapp, tmp_path, monkeypatch, dialog):
    """The preview is the specification, not an approximation of it."""
    from src.gui.batch_export import PLOT_MODE_COMBINED

    dialog.plot_panel.le_title.setText("Field sweep")
    host = _plot_host(monkeypatch)

    host._plot_batch_selection(
        dialog, _targets("0080"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_COMBINED, "plot_format": "PNG"},
    )

    assert dialog.plot_panel.axes.get_title() == "Field sweep"


def test_one_unreadable_scan_does_not_stop_the_others(qapp, tmp_path, monkeypatch, dialog):
    from src.gui.batch_export import PLOT_MODE_PER_SCAN

    def table(targets, *_a, **_k):
        if targets[0].scan_num == "0081":
            raise OSError("file is locked")
        return TABLE, HEADERS

    host = _plot_host(monkeypatch, table=table)

    host._plot_batch_selection(
        dialog, _targets("0080", "0081", "0082"), [],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_PER_SCAN, "plot_format": "PNG"},
    )

    assert len(list(tmp_path.glob("*.png"))) == 2


def test_end_to_end_a_combined_figure_really_draws_every_scan(qapp, tmp_path, monkeypatch):
    """Straight through the real table builder, from files on disk."""
    import h5py

    from src.gui.batch_export import BatchTarget, PLOT_MODE_COMBINED
    from src.gui.main_window import MainWindow

    targets = []
    for index, scan in enumerate(("0080", "0081", "0082")):
        path = tmp_path / f"scanx_{scan}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("scan/data", data=np.arange(300.0) * (index + 1))
        targets.append(
            BatchTarget(file_path=path, scan_num=scan, ds_path="scan/data", label=f"s{scan}")
        )

    monkeypatch.setattr(
        "src.gui.main_window.QMessageBox.information", staticmethod(lambda *a, **k: None)
    )

    dialog = BatchExportDialog(
        None,
        default_dir=tmp_path,
        scan_numbers=["0080", "0081", "0082"],
        dataset_path="scan/data",
        sample_data=np.arange(300.0),
        data_kind="curve",
        preview_x_loader=lambda *a, **k: None,
    )
    dialog.close()

    class _Host:
        _plot_batch_selection = MainWindow._plot_batch_selection
        _write_batch_figures = MainWindow._write_batch_figures
        _batch_targets_by_scan = staticmethod(MainWindow._batch_targets_by_scan)
        _report_batch_result = MainWindow._report_batch_result

    _Host()._plot_batch_selection(
        dialog, targets, [(t.file_path, t.scan_num) for t in targets],
        {"output_dir": tmp_path, "plot_output_mode": PLOT_MODE_COMBINED,
         "plot_format": "PNG", "curve_output_mode": "Combined file"},
    )

    lines = dialog.plot_panel.axes.lines
    assert len(lines) == 3, "one curve per scan, not just the first"
    assert len(lines[0].get_ydata()) == 300, "and at full length, not the 200-row preview cap"
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_an_image_batch_has_no_plot_page(qapp):
    d = BatchExportDialog(
        None,
        default_dir=pathlib.Path.home(),
        scan_numbers=["0080"],
        dataset_path="/scan/scan_data/frames",
        sample_data=np.zeros((32, 32)),
        data_kind="image",
        preview_x_loader=lambda *a, **k: None,
    )
    d.close()

    assert d.action() == "export", "an image batch can only be exported"

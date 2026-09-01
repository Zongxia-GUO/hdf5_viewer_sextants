"""The two rows under the tree line up on both edges.

They are read as one block, so a few pixels of offset between them is visible.
Two separate causes had to be fixed: only one row zeroed its layout margins, so
the other was inset by the style's default; and every widget in the top row had
a maximum width, so that row stopped at the sum of those and left the rest of
the panel empty while the bottom row stretched to a different cap.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication, QLabel

from src.gui.main_window import MainWindow

# What Qt reports for a widget with no maximum set.
UNCAPPED = 16777215


@pytest.fixture
def window(qapp):
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    QApplication.processEvents()
    yield win
    win.close()


def _plot_as_label(window) -> QLabel:
    for label in window.findChildren(QLabel):
        if label.text() == "Plot as":
            return label
    raise AssertionError("the 'Plot as' label is gone")


def _left(window, widget) -> int:
    return widget.mapTo(window, widget.rect().topLeft()).x()


def _right(window, widget) -> int:
    return widget.mapTo(window, widget.rect().topRight()).x()


def test_both_rows_start_at_the_same_x(window):
    """The margin difference showed up here."""
    assert _left(window, window.btn_collapse_all) == _left(window, _plot_as_label(window))


def test_both_rows_end_at_the_same_x(window):
    """The width caps showed up here."""
    assert _right(window, window.btn_batch_add) == _right(window, window.cb_plot_type)


def test_the_plot_as_combo_is_wider_than_the_two_buttons(window):
    """It should reach across them, not stop short."""
    buttons = window.btn_batch_browse.width() + window.btn_batch_add.width()

    assert window.cb_plot_type.width() > buttons


def test_the_rows_stay_aligned_when_the_window_is_resized(window):
    for width in (800, 1400, 1000):
        window.resize(width, 800)
        QApplication.processEvents()

        assert _left(window, window.btn_collapse_all) == _left(window, _plot_as_label(window))
        assert _right(window, window.btn_batch_add) == _right(window, window.cb_plot_type)


def test_each_row_has_one_thing_that_stretches(window):
    """A row where everything is capped cannot reach the panel edge."""
    assert window.le_batch_path.maximumWidth() == UNCAPPED
    assert window.cb_plot_type.maximumWidth() == UNCAPPED


def test_the_buttons_do_not_stretch(window):
    """Only the fields absorb the extra width; a stretched button looks broken."""
    assert window.btn_batch_browse.maximumWidth() < UNCAPPED
    assert window.btn_batch_add.maximumWidth() < UNCAPPED
    assert window.btn_collapse_all.maximumWidth() < UNCAPPED


def test_the_selection_fields_keep_their_size(window):
    """They hold short, known strings; the path field is the one that grows."""
    assert window.le_batch_keywords.maximumWidth() < UNCAPPED
    assert window.le_scan_range.maximumWidth() < UNCAPPED


def test_the_path_field_does_not_collapse_when_narrow(window):
    window.resize(600, 800)
    QApplication.processEvents()

    assert window.le_batch_path.width() >= window.le_batch_path.minimumWidth()

"""Reopening last time's files is a question when there are many of them.

The window reopens whatever was open when it last closed, which is right for a
handful of scans and wrong for a folder of a thousand: every file is read again
to build its row and stat'd again to check the index, and on a share that is
most of a minute before anything can be done.

A small session still restores without asking. A dialog in front of three files
costs more attention than the wait it would save.
"""

import pytest
from PyQt6.QtCore import QSettings

from src.gui.main_window import MainWindow
from src.gui.session_restore import (
    ALWAYS_ASK,
    DEFAULT_PROMPT_THRESHOLD,
    NEVER_ASK,
    THRESHOLD_KEY,
    read_threshold,
    should_ask,
)

FILES = tuple(f"C:/data/scan_{i:04d}.h5" for i in range(80))


# ── When to ask ───────────────────────────────────────────────────────── #

def test_a_big_session_is_worth_a_question():
    assert should_ask(80, 50) is True
    assert should_ask(50, 50) is True


def test_a_small_one_is_not():
    assert should_ask(49, 50) is False
    assert should_ask(3, 50) is False


def test_nothing_to_restore_is_never_worth_asking():
    """A dialog offering to reopen zero files is pure noise, whatever the
    threshold says."""
    assert should_ask(0, ALWAYS_ASK) is False
    assert should_ask(0, 1) is False


def test_zero_means_ask_every_time():
    assert should_ask(1, ALWAYS_ASK) is True


def test_minus_one_means_never_ask():
    """The way back to how it behaved before this existed."""
    assert should_ask(100000, NEVER_ASK) is False


def test_the_default_threshold_is_fifty():
    assert DEFAULT_PROMPT_THRESHOLD == 50


def test_an_unset_threshold_is_the_default(qapp):
    QSettings().remove(THRESHOLD_KEY)

    assert read_threshold(QSettings()) == DEFAULT_PROMPT_THRESHOLD


def test_an_unreadable_threshold_is_the_default(qapp):
    """Something else wrote the key; a bad value must not stop the app."""
    QSettings().setValue(THRESHOLD_KEY, "not a number")

    assert read_threshold(QSettings()) == DEFAULT_PROMPT_THRESHOLD


# ── What the window does with the answer ──────────────────────────────── #

@pytest.fixture
def win(qapp, monkeypatch):
    """A window whose restore is recorded rather than performed."""
    restored = []
    monkeypatch.setattr(MainWindow, "_restore_session",
                        lambda self, files, folder: restored.append(tuple(files)))
    window = MainWindow()
    window.restored = restored
    yield window
    window.close()
    window.deleteLater()


def _answer(monkeypatch, value):
    """Stand in for the dialog, and record what it was asked about."""
    asked = []
    import src.gui.main_window as mod
    monkeypatch.setattr(mod, "ask_restore",
                        lambda parent, count: (asked.append(count), value)[1])
    return asked


def test_a_small_session_restores_without_asking(win, monkeypatch):
    asked = _answer(monkeypatch, False)
    QSettings().setValue(THRESHOLD_KEY, 50)

    win._offer_session_restore(FILES[:10])

    assert asked == []
    assert win.restored == [FILES[:10]]


def test_a_big_session_asks_and_restores_when_told_to(win, monkeypatch):
    asked = _answer(monkeypatch, True)
    QSettings().setValue(THRESHOLD_KEY, 50)

    win._offer_session_restore(FILES)

    assert asked == [len(FILES)]
    assert win.restored == [FILES]


def test_declining_leaves_the_files_closed(win, monkeypatch):
    asked = _answer(monkeypatch, False)
    QSettings().setValue(THRESHOLD_KEY, 50)

    win._offer_session_restore(FILES)

    assert asked == [len(FILES)]
    assert win.restored == []


def test_declining_says_so_where_it_can_be_read(win, monkeypatch):
    """Otherwise an empty window after choosing "start empty" is
    indistinguishable from a session that was lost."""
    _answer(monkeypatch, False)
    QSettings().setValue(THRESHOLD_KEY, 50)

    win._offer_session_restore(FILES)

    message = win._menu_status_raw_text
    assert "Started empty" in message
    assert str(len(FILES)) in message


def test_declining_does_not_delete_the_saved_session(win, monkeypatch):
    """The answer is about this launch. What is open at the next clean close is
    what gets saved, so a decline followed by a close is what clears it."""
    _answer(monkeypatch, False)
    QSettings().setValue(THRESHOLD_KEY, 50)
    QSettings().setValue("settings/last_opened_files", FILES)

    win._offer_session_restore(FILES)

    kept = tuple(QSettings().value("settings/last_opened_files", ()) or ())
    assert len(kept) == len(FILES)


def test_an_empty_session_neither_asks_nor_restores(win, monkeypatch):
    asked = _answer(monkeypatch, True)
    QSettings().setValue(THRESHOLD_KEY, ALWAYS_ASK)

    win._offer_session_restore(())

    assert asked == []
    assert win.restored == []


def test_never_ask_restores_silently_however_many(win, monkeypatch):
    asked = _answer(monkeypatch, False)
    QSettings().setValue(THRESHOLD_KEY, NEVER_ASK)

    win._offer_session_restore(FILES)

    assert asked == []
    assert win.restored == [FILES]


# ── One look at each file, not two ────────────────────────────────────── #
#
# Accepting a file and deciding which kind of row to build are the same
# question. Asking it twice was two of the three filesystem round trips a
# restored file cost, which is invisible locally and most of the wait on a
# share.

def _round_trips(monkeypatch, paths, window):
    """Count the filesystem calls adding these files makes."""
    import h5py

    calls = {"is_hdf5": 0, "open": 0}
    real_is, real_file = h5py.is_hdf5, h5py.File
    monkeypatch.setattr(h5py, "is_hdf5",
                        lambda *a, **k: (calls.__setitem__("is_hdf5", calls["is_hdf5"] + 1),
                                         real_is(*a, **k))[1])
    monkeypatch.setattr(h5py, "File",
                        lambda *a, **k: (calls.__setitem__("open", calls["open"] + 1),
                                         real_file(*a, **k))[1])
    for path in paths:
        window._open_file(path)
    return calls


@pytest.fixture
def scans(tmp_path):
    import h5py
    import numpy as np

    made = []
    for i in range(5):
        path = tmp_path / f"Scan_{i:03d}.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("entry/data", data=np.arange(4.0))
        made.append(path)
    return made


def test_adding_a_file_looks_at_it_once(qapp, scans, monkeypatch):
    window = MainWindow()
    try:
        calls = _round_trips(monkeypatch, scans, window)

        assert calls["is_hdf5"] == len(scans), "one signature read per file"
        assert calls["open"] == 0, "the full open was the redundant half"
    finally:
        window.close()
        window.deleteLater()


def test_a_file_we_do_not_read_is_not_opened_at_all(qapp, tmp_path):
    """An unsupported suffix is settled by its name; touching the disk to
    reject it is pure latency."""
    import h5py

    from src.lib_h5.file_validator import classify_data_file

    junk = tmp_path / "notes.docx"
    junk.write_text("x", encoding="utf-8")
    opened = []
    real = h5py.is_hdf5
    h5py.is_hdf5 = lambda *a, **k: (opened.append(True), real(*a, **k))[1]
    try:
        assert classify_data_file(junk) is None
    finally:
        h5py.is_hdf5 = real

    assert opened == []


def test_the_classifier_agrees_with_what_it_replaced(qapp, scans, tmp_path):
    from src.lib_h5.file_validator import (
        KIND_HDF5,
        KIND_REGULAR,
        classify_data_file,
        is_supported_data_file,
    )

    text = tmp_path / "curve.txt"
    text.write_text("1 2\n", encoding="utf-8")
    dat = tmp_path / "curve.dat"
    dat.write_text("1 2\n", encoding="utf-8")
    junk = tmp_path / "notes.docx"
    junk.write_text("x", encoding="utf-8")

    assert classify_data_file(scans[0]) == KIND_HDF5
    assert classify_data_file(text) == KIND_REGULAR
    assert classify_data_file(dat) == KIND_REGULAR
    assert classify_data_file(junk) is None
    for path in (scans[0], text, dat, junk):
        assert (classify_data_file(path) is not None) == is_supported_data_file(path), path


def test_a_regular_file_still_gets_its_single_data_row(qapp, tmp_path):
    """The kind decides the row shape, so the merged check has to keep that."""
    text = tmp_path / "curve.txt"
    text.write_text("1 2\n3 4\n", encoding="utf-8")
    window = MainWindow()
    try:
        window._open_file(text)

        assert window.tree_model_file.rowCount() == 1
        row = window.tree_model_file.item(0, 0)
        assert row.rowCount() == 1
        assert row.child(0, 0).text() == "data"
    finally:
        window.close()
        window.deleteLater()

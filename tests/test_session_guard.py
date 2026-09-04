"""Starting clean after a run that did not finish.

The session remembers which files were open and reopens them next time, which
is right after a normal close and wrong after a crash: the files are added and
indexed all over again, and if one of them is what killed the process then the
application cannot be started at all — it crashes, restores, crashes again.

A heartbeat is what tells the two apart, and it is a timestamp rather than a
flag because nothing here stops a second window being opened: a flag cannot
distinguish "the last run died" from "another window is running right now",
and would answer the second by throwing away that window's file list.
"""

import json
import time

import pytest
from PyQt6.QtCore import QSettings

from src.gui.main_window import MainWindow
from src.gui.session_guard import (
    ENDED_BADLY,
    ENDED_CLEANLY,
    HEARTBEAT_KEY,
    STALE_AFTER_S,
    STILL_RUNNING,
    SessionGuard,
    classify,
    describe_age,
)

FILES = tuple(f"C:/data/scan_{i:04d}.h5" for i in range(7))


# ── What the stamp means ──────────────────────────────────────────────── #

def test_no_stamp_means_the_last_run_finished():
    assert classify(None, 1000.0) == ENDED_CLEANLY


def test_a_fresh_stamp_means_another_window_is_up():
    assert classify(1000.0 - 1, 1000.0) == STILL_RUNNING
    assert classify(1000.0 - (STALE_AFTER_S - 1), 1000.0) == STILL_RUNNING


def test_a_stale_stamp_means_the_writer_never_came_back():
    assert classify(1000.0 - (STALE_AFTER_S + 1), 1000.0) == ENDED_BADLY
    assert classify(1000.0 - 3600, 1000.0) == ENDED_BADLY


def test_a_few_missed_beats_are_not_a_crash():
    """An application briefly too busy to run its timers is not a dead one."""
    from src.gui.session_guard import HEARTBEAT_INTERVAL_MS

    two_missed = 2 * HEARTBEAT_INTERVAL_MS / 1000.0

    assert classify(1000.0 - two_missed, 1000.0) == STILL_RUNNING


def test_a_stamp_from_the_future_decides_nothing():
    """The clock moved backwards between runs — a time zone, an NTP step. The
    cost of guessing wrong here is a discarded session, so it does not guess."""
    assert classify(1000.0 + 50, 1000.0) == STILL_RUNNING


@pytest.mark.parametrize("seconds,text", [
    (5, "5 seconds ago"),
    (89, "89 seconds ago"),
    (120, "2 minutes ago"),
    (3600, "60 minutes ago"),
    (7200, "2 hours ago"),
])
def test_the_age_is_put_in_words(seconds, text):
    assert describe_age(seconds) == text


# ── The guard itself ──────────────────────────────────────────────────── #

def test_starting_stamps_immediately(qapp):
    """A crash in the first ten seconds still has to leave evidence."""
    QSettings().remove(HEARTBEAT_KEY)
    guard = SessionGuard()

    guard.start()

    assert QSettings().value(HEARTBEAT_KEY) is not None
    guard.mark_clean_exit()


def test_closing_cleanly_removes_the_stamp(qapp):
    guard = SessionGuard()
    guard.start()

    guard.mark_clean_exit()

    assert QSettings().value(HEARTBEAT_KEY, None) is None


def test_a_guard_started_after_a_stale_stamp_reports_the_crash(qapp):
    QSettings().setValue(HEARTBEAT_KEY, time.time() - 400)

    guard = SessionGuard()

    assert guard.previous_run_crashed is True
    assert guard.age_of_last_stamp == pytest.approx(400, abs=5)


def test_an_unreadable_stamp_is_not_evidence(qapp):
    """Something else wrote the key. Unreadable is not the same as crashed."""
    QSettings().setValue(HEARTBEAT_KEY, "not a number")

    assert SessionGuard().verdict == ENDED_CLEANLY


# ── What the window does about it ─────────────────────────────────────── #

def _restored_by_a_fresh_window(qapp):
    """Open a window and report how many files it decided to reopen."""
    restored = []
    real = MainWindow._restore_session
    MainWindow._restore_session = lambda self, files, folder: restored.append(tuple(files))
    try:
        win = MainWindow()
        qapp.processEvents()
    finally:
        MainWindow._restore_session = real
    return win, (restored[0] if restored else ())


def test_a_crash_stops_the_files_being_reopened(qapp):
    QSettings().setValue("settings/last_opened_files", FILES)
    QSettings().setValue(HEARTBEAT_KEY, time.time() - 400)

    win, restored = _restored_by_a_fresh_window(qapp)
    try:
        assert restored == ()
    finally:
        win.close()
        win.deleteLater()


def test_a_crash_clears_the_list_so_it_cannot_happen_twice(qapp):
    """Otherwise a file that kills the process kills every start after it."""
    QSettings().setValue("settings/last_opened_files", FILES)
    QSettings().setValue(HEARTBEAT_KEY, time.time() - 400)

    win, _ = _restored_by_a_fresh_window(qapp)
    try:
        assert tuple(QSettings().value("settings/last_opened_files", ()) or ()) == ()
    finally:
        win.close()
        win.deleteLater()


def test_a_crash_is_reported_where_it_can_be_read(qapp):
    """Silently dropping someone's session would look like the session was
    lost, which is worse than the wait it avoids."""
    QSettings().setValue("settings/last_opened_files", FILES)
    QSettings().setValue(HEARTBEAT_KEY, time.time() - 400)

    win, _ = _restored_by_a_fresh_window(qapp)
    try:
        message = win._menu_status_raw_text

        assert "unexpectedly" in message
        assert str(len(FILES)) in message
        assert "minutes ago" in message
    finally:
        win.close()
        win.deleteLater()


def test_a_clean_close_still_reopens_everything(qapp):
    QSettings().setValue("settings/last_opened_files", FILES)
    QSettings().remove(HEARTBEAT_KEY)

    win, restored = _restored_by_a_fresh_window(qapp)
    try:
        assert restored == FILES
    finally:
        win.close()
        win.deleteLater()


def test_a_second_window_does_not_discard_the_first_ones_session(qapp):
    """The reason the evidence is a timestamp and not a flag."""
    QSettings().setValue("settings/last_opened_files", FILES)
    QSettings().setValue(HEARTBEAT_KEY, time.time())

    win, restored = _restored_by_a_fresh_window(qapp)
    try:
        assert restored == FILES
        assert win._menu_status_raw_text == ""
    finally:
        win.close()
        win.deleteLater()


def test_closing_the_window_marks_the_run_as_finished(qapp):
    win = MainWindow()
    qapp.processEvents()
    assert QSettings().value(HEARTBEAT_KEY, None) is not None

    win.close()

    assert QSettings().value(HEARTBEAT_KEY, None) is None
    win.deleteLater()


# ── The cache that is written on the way out ──────────────────────────── #

def test_the_index_cache_is_replaced_not_truncated(qapp, monkeypatch):
    """It is written during closeEvent, so a process that dies there used to
    leave half a JSON document. The loader survives that but throws the cache
    away, and the next start rescans every file for nothing."""
    win = MainWindow()
    try:
        win._dataset_per_file_index_cache = {"C:/a.h5": ((1, 2), ["C:/a.h5::x"], [])}
        win._save_disk_index_cache()
        path = win._disk_index_cache_path()
        good = path.read_text(encoding="utf-8")

        def die(*args, **kwargs):
            raise OSError("the process went away mid-write")

        monkeypatch.setattr(json, "dump", die)
        win._dataset_per_file_index_cache = {"C:/b.h5": ((3, 4), ["C:/b.h5::y"], [])}
        win._save_disk_index_cache()

        assert path.read_text(encoding="utf-8") == good, "the old cache was destroyed"
        assert not path.with_suffix(path.suffix + ".tmp").exists()
    finally:
        win.close()
        win.deleteLater()


def test_the_cache_still_round_trips(qapp):
    win = MainWindow()
    try:
        win._dataset_per_file_index_cache = {"C:/a.h5": ((1, 2), ["C:/a.h5::x"], [])}
        win._save_disk_index_cache()

        win._dataset_per_file_index_cache = {}
        win._load_disk_index_cache()

        assert list(win._dataset_per_file_index_cache) == ["C:/a.h5"]
    finally:
        win.close()
        win.deleteLater()

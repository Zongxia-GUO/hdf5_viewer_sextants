"""Importing many scans at once: what it costs and what it must not drop.

Three defects are pinned here, all found by measuring a 300-file import:

* a warm started while another was running was **dropped entirely**, so files
  opened during the first scan never entered the index and stayed invisible to
  every tool's dataset picker;
* the ``fast`` index scope walked the whole file and filtered afterwards, so
  restricting it saved 2% — the setting read as if it halved the work;
* the tree and the scanner each opened every file more times than they needed.

The timing numbers that motivated these live in the commit message; the tests
here assert the behaviour that makes them true, not the timings themselves.
"""

import pathlib

import h5py
import numpy as np
import pytest

from src.gui.main_window import MainWindow, _DatasetIndexWarmWorker
from src.lib_h5.file_validator import is_hdf5_file, looks_like_hdf5

SCAN = "scan_0041"


@pytest.fixture
def scans(tmp_path):
    """Three files shaped like real scans: a small wanted group, a big unwanted one."""
    paths = []
    for i in range(3):
        path = tmp_path / f"Scan_ECL_{i:03d}.hdf5"
        with h5py.File(path, "w") as f:
            data = f.create_group(f"{SCAN}/scan_data")
            data.create_dataset("data_01", data=np.arange(10.0))
            data.create_dataset("frames", data=np.zeros((4, 256)))
            meta = f.create_group(f"{SCAN}/SEXTANTS")
            for dev in range(6):
                g = meta.create_group(f"device_{dev}")
                for k in range(4):
                    g.create_dataset(f"param_{k}", data=np.arange(2.0))
        paths.append(path)
    return paths


# ── The scope filter now runs before the cost, not after ──────────────── #

def _lookups(path, *, scope, fast_paths):
    """Which names the scanner actually instantiates an h5py object for."""
    seen = []
    real = h5py.Group.__getitem__

    def spy(self, name):
        seen.append(name)
        return real(self, name)

    h5py.Group.__getitem__ = spy
    try:
        _DatasetIndexWarmWorker._scan_single_file_both(
            str(path), index_scope=scope, fast_group_paths=fast_paths
        )
    finally:
        h5py.Group.__getitem__ = real
    return seen


def test_fast_scope_never_opens_an_out_of_scope_dataset(scans):
    """This is the whole point: visititems() built an object for every node and
    only then discarded it, so the scope filter cost as much as no filter."""
    seen = _lookups(scans[0], scope="fast", fast_paths=("scan_data",))

    assert seen, "the scanner opened nothing at all"
    assert all("scan_data" in name for name in seen), seen
    assert not any("device_" in name for name in seen)


def test_fast_scope_opens_far_fewer_objects_than_full_scope(scans):
    fast = _lookups(scans[0], scope="fast", fast_paths=("scan_data",))
    full = _lookups(scans[0], scope="all", fast_paths=())

    assert len(fast) < len(full) / 4, (len(fast), len(full))


def test_the_keys_are_unchanged_by_the_faster_walk(scans):
    """Equivalence, not just speed: the same datasets in the same order."""
    keys_1d, keys_2d = _DatasetIndexWarmWorker._scan_single_file_both(
        str(scans[0]), index_scope="fast", fast_group_paths=("scan_data",)
    )

    assert keys_1d == [f"{scans[0]}::{SCAN}/scan_data/data_01",
                       f"{scans[0]}::{SCAN}/scan_data/frames"]
    assert keys_2d == [f"{scans[0]}::{SCAN}/scan_data/frames"]


def test_a_file_that_is_not_hdf5_still_reaches_the_regular_loader(tmp_path):
    """The scanner dropped its is_hdf5_file() probe for a single open; the
    fallback for everything else has to survive that."""
    path = tmp_path / "curve.txt"
    path.write_text("1 2\n3 4\n5 6\n", encoding="utf-8")

    keys_1d, _ = _DatasetIndexWarmWorker._scan_single_file_both(
        str(path), index_scope="all", fast_group_paths=()
    )

    assert keys_1d == [f"{path}::data"]


# ── The signature probe ───────────────────────────────────────────────── #

def test_the_signature_probe_agrees_with_the_full_open(scans, tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")
    missing = tmp_path / "gone.h5"

    for path in (scans[0], text, missing):
        assert looks_like_hdf5(path) == is_hdf5_file(path), path


def test_the_signature_probe_does_not_open_the_file(scans, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("looks_like_hdf5 opened the file")

    monkeypatch.setattr(h5py, "File", explode)

    assert looks_like_hdf5(scans[0]) is True


# ── A warm started during a warm is no longer lost ────────────────────── #

def test_files_opened_during_a_warm_are_picked_up_when_it_finishes(qapp, scans):
    """Measured before the fix: 100 of 300 files never entered the index."""
    win = MainWindow()
    try:
        for path in scans:
            win._open_file(path)

        # Finish a warm that only ever knew about the first file.
        first = scans[0]
        win._index_warm_files = (first,)
        win._index_warm_worker = None
        restarted = []
        win._prime_dataset_index_async = lambda: restarted.append(True)

        win._on_dataset_index_warm_done(
            {str(first): ((0, 0), [f"{first}::a"], [])},
            win._index_scope,
            win._fast_group_paths,
        )

        assert restarted == [True], "the files opened mid-warm were dropped"
    finally:
        win.close()
        win.deleteLater()


def test_a_warm_that_covered_everything_does_not_restart(qapp, scans):
    """The rescan must be triggered by a real gap, not run every time."""
    win = MainWindow()
    try:
        for path in scans:
            win._open_file(path)
        win._index_warm_files = tuple(win.opened_files)
        win._index_warm_worker = None
        restarted = []
        win._prime_dataset_index_async = lambda: restarted.append(True)

        win._on_dataset_index_warm_done(
            {str(p): ((0, 0), [], []) for p in scans},
            win._index_scope,
            win._fast_group_paths,
        )

        assert restarted == []
    finally:
        win.close()
        win.deleteLater()


def test_the_worker_records_the_file_set_it_was_given(qapp, scans):
    win = MainWindow()
    try:
        for path in scans:
            win._open_file(path)
        win._prime_dataset_index_async()

        assert win._index_warm_files == tuple(win.opened_files)
        if win._index_warm_worker is not None:
            win._index_warm_worker.wait(30000)
    finally:
        win.close()
        win.deleteLater()


def test_opened_files_are_plain_paths(qapp, scans):
    """The comparison in the done handler is a tuple equality, so the two sides
    have to be the same kind of object."""
    win = MainWindow()
    try:
        for path in scans:
            win._open_file(path)

        assert all(isinstance(p, pathlib.Path) for p in win.opened_files)
    finally:
        win.close()
        win.deleteLater()


# ── Open tools are not rebuilt once per scanned batch ─────────────────── #

def _count_refreshes(win):
    hits = []
    win.dataset_index_changed.connect(lambda: hits.append(True))
    return hits


def test_a_scanned_batch_does_not_rebuild_the_tools_immediately(qapp, scans):
    """Every refresh rebuilds each tool's combos from the WHOLE key list, so
    doing it per batch costs sum-of-prefixes. Measured on 300 files: 25
    rebuilds at batch 10, 745 ms of GUI-thread work."""
    win = MainWindow()
    try:
        hits = _count_refreshes(win)

        win._on_dataset_index_warm_batch(
            {str(scans[0]): ((0, 0), [f"{scans[0]}::a"], [])},
            1, 3, win._index_scope, win._fast_group_paths,
        )

        assert hits == [], "the batch rebuilt the tools synchronously"
        assert win._index_refresh_timer.isActive()
    finally:
        win.close()
        win.deleteLater()


def test_many_batches_still_schedule_only_one_refresh(qapp, scans):
    """The cost must stop scaling with how the scan happens to be batched."""
    win = MainWindow()
    try:
        hits = _count_refreshes(win)
        starts = []
        real_start = win._index_refresh_timer.start
        win._index_refresh_timer.start = lambda *a: (starts.append(True), real_start(*a))[1]

        for i in range(20):
            win._on_dataset_index_warm_batch(
                {str(scans[0]): ((0, 0), [f"{scans[0]}::a{i}"], [])},
                i + 1, 20, win._index_scope, win._fast_group_paths,
            )

        assert hits == []
        assert len(starts) == 1, f"the timer was restarted {len(starts)} times"
    finally:
        win.close()
        win.deleteLater()


def test_the_pending_refresh_is_not_pushed_back_forever(qapp, scans):
    """Restarting the timer on each batch would starve it on a fast scan and
    show nothing until the very end, which is why start() is guarded."""
    win = MainWindow()
    try:
        win._on_dataset_index_warm_batch(
            {str(scans[0]): ((0, 0), [], [])}, 1, 9, win._index_scope, win._fast_group_paths,
        )
        first = win._index_refresh_timer.remainingTime()
        for i in range(5):
            win._on_dataset_index_warm_batch(
                {str(scans[0]): ((0, 0), [], [])}, i + 2, 9,
                win._index_scope, win._fast_group_paths,
            )

        assert win._index_refresh_timer.remainingTime() <= first
    finally:
        win.close()
        win.deleteLater()


def test_the_final_refresh_is_immediate(qapp, scans):
    """Coalescing must never swallow the last one: that is the refresh that
    leaves the tools showing the complete index."""
    win = MainWindow()
    try:
        for path in scans:
            win._open_file(path)
        win._index_warm_files = tuple(win.opened_files)
        win._index_warm_worker = None
        win._index_refresh_timer.start()
        hits = _count_refreshes(win)

        win._on_dataset_index_warm_done(
            {str(p): ((0, 0), [], []) for p in scans},
            win._index_scope, win._fast_group_paths,
        )

        assert hits == [True]
        assert not win._index_refresh_timer.isActive()
    finally:
        win.close()
        win.deleteLater()


def test_the_timer_emits_the_same_signal_the_tools_listen_to(qapp):
    win = MainWindow()
    try:
        hits = _count_refreshes(win)
        win._index_refresh_timer.timeout.emit()

        assert hits == [True]
    finally:
        win.close()
        win.deleteLater()


# ── The combo label is memoised, not reimplemented ────────────────────── #

def test_the_memoised_basename_matches_pathlib_exactly():
    from src.gui.dataset_path_combo import DatasetPathCombo

    for full_key in (
        "C:/a/b/scan_001.hdf5::g/scan_data/d1",
        r"D:\x\y\f.h5::a/b/c",
        "rel.h5::d",
        "/unix/p/f.nxs::  g/d  ",
        "no_dir.h5::x/y/z/w",
    ):
        fname, ds_path = full_key.rsplit("::", 1)
        expected = (f"{pathlib.Path(fname).name}::"
                    f"{DatasetPathCombo._short_dataset_label(ds_path.strip())}")

        assert DatasetPathCombo.short_display_from_full_key(full_key) == expected, full_key


def test_the_basename_is_computed_once_per_file(qapp):
    """Every dataset repeats its file's path; pathlib was 45% of a refresh."""
    from src.gui.dataset_path_combo import DatasetPathCombo

    DatasetPathCombo._file_basename.cache_clear()
    for i in range(50):
        DatasetPathCombo.short_display_from_full_key(f"C:/d/scan.h5::g/data_{i}")

    info = DatasetPathCombo._file_basename.cache_info()
    assert info.misses == 1 and info.hits == 49

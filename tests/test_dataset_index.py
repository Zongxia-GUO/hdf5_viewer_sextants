"""The dataset index: what the pickers list, and the cache behind it.

Two things are covered here, both previously untested and both with silent
failure modes:

* the **scan rules** decide which datasets appear in every picker in the
  application — the comparison X list, the calculator's A/B, the batch path
  field, the search. A mistake here shows up as "my dataset is missing", with
  no error anywhere.
* the **disk cache** persists that index between runs, keyed on a
  (mtime, size) signature, with a format version and an LRU cap. Cache
  invalidation fails silently by definition: a stale entry lists datasets that
  are no longer in the file.
"""

import json
import os
import pathlib
import time

import h5py
import numpy as np
import pytest

from src.gui.main_window import MainWindow, _DatasetIndexWarmWorker

SCAN = "scan_0033"


@pytest.fixture
def h5(tmp_path):
    """A file shaped like a real beamline scan."""
    path = tmp_path / "scanx_0033.h5"
    with h5py.File(path, "w") as f:
        data = f.create_group(f"{SCAN}/scan_data")
        data.create_dataset("data_01", data=np.arange(10.0))
        data.create_dataset("actuator_1_1", data=np.arange(10.0))
        data.create_dataset("frames", data=np.zeros((4, 256)))     # FTH candidate
        data.create_dataset("narrow", data=np.zeros((4, 8)))       # too few columns
        meta = f.create_group(f"{SCAN}/scan_meta")
        meta.create_dataset("comment", data=np.arange(3.0))
    return path


def scan(path, *, scope="full", fast_paths=()):
    return _DatasetIndexWarmWorker._scan_single_file_both(
        str(path), index_scope=scope, fast_group_paths=tuple(fast_paths)
    )


def leaves(keys):
    return sorted(k.split("::", 1)[1].rsplit("/", 1)[-1] for k in keys)


# ---------------------------------------------------------------------------
# What the scan finds
# ---------------------------------------------------------------------------

def test_every_dataset_is_indexed_in_full_scope(h5):
    keys_1d, _ = scan(h5, scope="full")

    assert leaves(keys_1d) == ["actuator_1_1", "comment", "data_01", "frames", "narrow"]


def test_groups_are_not_datasets(h5):
    keys_1d, _ = scan(h5, scope="full")

    assert not any(k.endswith("scan_data") or k.endswith(SCAN) for k in keys_1d)


def test_the_keys_are_full_file_plus_path(h5):
    keys_1d, _ = scan(h5, scope="full")

    assert all(k.startswith(f"{h5}::") for k in keys_1d)
    assert f"{h5}::{SCAN}/scan_data/data_01" in keys_1d


def test_a_wide_2d_dataset_is_offered_to_the_fth_tools(h5):
    """The FTH picker wants image-like frames, not two-column tables."""
    _keys_1d, keys_2d = scan(h5, scope="full")

    assert leaves(keys_2d) == ["frames"], "256 columns qualifies, 8 does not"


def test_fast_scope_keeps_only_the_configured_groups(h5):
    keys_1d, _ = scan(h5, scope="fast", fast_paths=("scan_data",))

    assert leaves(keys_1d) == ["actuator_1_1", "data_01", "frames", "narrow"]
    assert "comment" not in leaves(keys_1d), "scan_meta is outside the fast scope"


def test_fast_scope_without_any_paths_keeps_everything(h5):
    """An empty filter must mean "no filter", not "nothing matches"."""
    keys_1d, _ = scan(h5, scope="fast", fast_paths=())

    assert len(keys_1d) == 5


def test_fast_paths_match_a_whole_path_segment(h5):
    """"scan_dat" must not match "scan_data", or the filter would be a substring test."""
    keys_1d, _ = scan(h5, scope="fast", fast_paths=("scan_dat",))
    assert keys_1d == []


def test_blank_entries_in_the_fast_paths_are_ignored(h5):
    keys_1d, _ = scan(h5, scope="fast", fast_paths=("", "  ", "/scan_data/"))
    assert leaves(keys_1d) == ["actuator_1_1", "data_01", "frames", "narrow"]


def test_a_plain_text_file_is_indexed_as_one_dataset(tmp_path):
    path = tmp_path / "curve.txt"
    path.write_text("1 2\n3 4\n5 6\n", encoding="utf-8")

    keys_1d, keys_2d = scan(path)

    assert keys_1d == [f"{path}::data"]
    assert keys_2d == [], "two columns is not an image"


# ---------------------------------------------------------------------------
# The signature that decides whether a rescan is needed
# ---------------------------------------------------------------------------

def test_the_signature_is_modification_time_and_size(h5):
    sig = _DatasetIndexWarmWorker._file_signature(str(h5))
    st = os.stat(h5)

    assert sig[1] == st.st_size
    assert isinstance(sig[0], int)


def test_rewriting_a_file_changes_its_signature(h5):
    before = _DatasetIndexWarmWorker._file_signature(str(h5))

    time.sleep(0.01)
    with h5py.File(h5, "a") as f:
        f[f"{SCAN}/scan_data"].create_dataset("added", data=np.arange(4.0))

    assert _DatasetIndexWarmWorker._file_signature(str(h5)) != before


def worker(files, prev=None, scope="full", fast=(), batch=1):
    return _DatasetIndexWarmWorker(
        opened_files=tuple(files),
        prev_cache=prev or {},
        index_scope=scope,
        fast_group_paths=tuple(fast),
        batch_size=batch,
    )


def test_an_unchanged_file_is_served_from_the_cache_not_rescanned(h5, monkeypatch):
    """This is the whole point of the cache: no second walk of the file."""
    sig = _DatasetIndexWarmWorker._file_signature(str(h5))
    prev = {str(h5): (sig, ["cached::key"], [])}

    rescans: list[str] = []
    monkeypatch.setattr(
        _DatasetIndexWarmWorker, "_scan_single_file_both",
        staticmethod(lambda p, **k: rescans.append(p) or ([], [])),
    )

    out = worker([h5], prev=prev)._update_cache(prev)

    assert rescans == []
    assert out[str(h5)][1] == ["cached::key"]


def test_a_changed_file_is_rescanned(h5):
    stale = {str(h5): (((0, 0)), ["stale::key"], [])}

    out = worker([h5], prev=stale)._update_cache(stale)

    assert out[str(h5)][1] != ["stale::key"]
    assert len(out[str(h5)][1]) == 5


def test_one_unreadable_file_does_not_stop_the_rest(h5, tmp_path):
    missing = tmp_path / "gone.h5"

    out = worker([missing, h5])._update_cache({})

    assert str(missing) not in out
    assert str(h5) in out


def test_progress_is_reported_in_batches(h5, tmp_path):
    """The status line updates while a folder of scans is indexed."""
    copies = []
    for i in range(4):
        target = tmp_path / f"copy_{i}.h5"
        target.write_bytes(h5.read_bytes())
        copies.append(target)

    seen: list[tuple[int, int]] = []
    w = worker(copies, batch=2)
    w.batch.connect(lambda _d, done, total, _s, _f: seen.append((done, total)))

    w._update_cache({})

    assert seen == [(2, 4), (4, 4)]


# ---------------------------------------------------------------------------
# The disk cache
# ---------------------------------------------------------------------------

class _CacheHost:
    """MainWindow's index-cache surface, without the window.

    Every method below touches only plain attributes, which is what makes this
    stand-in honest rather than a mock.
    """

    _INDEX_CACHE_VERSION = MainWindow._INDEX_CACHE_VERSION
    _load_disk_index_cache = MainWindow._load_disk_index_cache
    _save_disk_index_cache = MainWindow._save_disk_index_cache
    _prune_index_cache = MainWindow._prune_index_cache

    def __init__(self, path, scope="fast", fast_paths=("scan_data",), max_files=5000):
        self._path = pathlib.Path(path)
        self._index_scope = scope
        self._fast_group_paths = tuple(fast_paths)
        self._dataset_per_file_index_cache: dict = {}
        self._dataset_index_last_used: dict = {}
        self._index_cache_max_files = max_files
        self.opened_files: tuple = ()

    def _disk_index_cache_path(self):
        return self._path


def record(sig=(111, 222), keys=("a::b",)):
    return ((sig[0], sig[1]), list(keys), [])


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "index_cache.json"


def test_the_cache_survives_a_round_trip(cache_file):
    host = _CacheHost(cache_file)
    host._dataset_per_file_index_cache = {"/data/a.h5": record()}
    host._dataset_index_last_used = {"/data/a.h5": 999}
    host._save_disk_index_cache()

    reloaded = _CacheHost(cache_file)
    reloaded._load_disk_index_cache()

    assert reloaded._dataset_per_file_index_cache == {"/data/a.h5": ((111, 222), ["a::b"], [])}
    assert reloaded._dataset_index_last_used["/data/a.h5"] == 999


def test_a_missing_cache_file_is_simply_no_cache(cache_file):
    host = _CacheHost(cache_file)
    host._load_disk_index_cache()

    assert host._dataset_per_file_index_cache == {}


def test_a_cache_from_an_older_format_is_discarded(cache_file):
    host = _CacheHost(cache_file)
    host._dataset_per_file_index_cache = {"/data/a.h5": record()}
    host._save_disk_index_cache()

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["version"] = MainWindow._INDEX_CACHE_VERSION + 1
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = _CacheHost(cache_file)
    reloaded._load_disk_index_cache()

    assert reloaded._dataset_per_file_index_cache == {}


def test_a_cache_built_under_another_scope_is_discarded(cache_file):
    """A fast-scope index lists fewer datasets; reusing it in full scope would lie."""
    host = _CacheHost(cache_file, scope="fast")
    host._dataset_per_file_index_cache = {"/data/a.h5": record()}
    host._save_disk_index_cache()

    reloaded = _CacheHost(cache_file, scope="full")
    reloaded._load_disk_index_cache()

    assert reloaded._dataset_per_file_index_cache == {}


def test_changing_the_fast_paths_discards_the_cache(cache_file):
    host = _CacheHost(cache_file, scope="fast", fast_paths=("scan_data",))
    host._dataset_per_file_index_cache = {"/data/a.h5": record()}
    host._save_disk_index_cache()

    reloaded = _CacheHost(cache_file, scope="fast", fast_paths=("other_group",))
    reloaded._load_disk_index_cache()

    assert reloaded._dataset_per_file_index_cache == {}


@pytest.mark.parametrize("content", ["{not json", "[]", '"a string"', ""])
def test_a_damaged_cache_file_is_ignored_rather_than_fatal(cache_file, content):
    cache_file.write_text(content, encoding="utf-8")
    host = _CacheHost(cache_file)

    host._load_disk_index_cache()

    assert host._dataset_per_file_index_cache == {}


def test_a_malformed_entry_is_dropped_and_the_rest_kept(cache_file):
    cache_file.write_text(json.dumps({
        "version": MainWindow._INDEX_CACHE_VERSION,
        "scope": "fast",
        "fast_paths": ["scan_data"],
        "files": {
            "/data/good.h5": {"sig": [1, 2], "keys_1d": ["k"], "keys_2d_fth": []},
            "/data/no_sig.h5": {"keys_1d": ["k"], "keys_2d_fth": []},
            "/data/short_sig.h5": {"sig": [1], "keys_1d": ["k"], "keys_2d_fth": []},
            "/data/bad_keys.h5": {"sig": [1, 2], "keys_1d": "not a list", "keys_2d_fth": []},
            "/data/not_a_record.h5": "nonsense",
        },
    }), encoding="utf-8")
    host = _CacheHost(cache_file)

    host._load_disk_index_cache()

    assert list(host._dataset_per_file_index_cache) == ["/data/good.h5"]


def test_saving_creates_the_folder_it_needs(tmp_path):
    nested = tmp_path / "a" / "b" / "index.json"
    host = _CacheHost(nested)
    host._dataset_per_file_index_cache = {"/data/a.h5": record()}

    host._save_disk_index_cache()

    assert nested.exists()


def test_an_unwritable_destination_is_logged_not_raised(tmp_path):
    host = _CacheHost(tmp_path)   # a directory, not a file
    host._dataset_per_file_index_cache = {"/data/a.h5": record()}

    host._save_disk_index_cache()   # must not raise


# ---------------------------------------------------------------------------
# Pruning: the cap that stops the index growing without bound
# ---------------------------------------------------------------------------

def fill(host, count, first_used=0):
    for i in range(count):
        key = f"/data/f{i:04d}.h5"
        host._dataset_per_file_index_cache[key] = record(sig=(i, 1))
        host._dataset_index_last_used[key] = first_used + i


def test_a_cache_under_the_cap_is_left_alone(cache_file):
    host = _CacheHost(cache_file, max_files=100)
    fill(host, 100)

    host._prune_index_cache()

    assert len(host._dataset_per_file_index_cache) == 100


def test_the_least_recently_used_go_first(cache_file):
    host = _CacheHost(cache_file, max_files=100)
    fill(host, 130)

    host._prune_index_cache()

    remaining = set(host._dataset_per_file_index_cache)
    assert len(remaining) == 100
    assert "/data/f0000.h5" not in remaining, "the oldest went"
    assert "/data/f0129.h5" in remaining, "the newest stayed"


def test_the_last_used_table_is_pruned_with_the_cache(cache_file):
    """Otherwise it grows for ever behind the cache it describes."""
    host = _CacheHost(cache_file, max_files=100)
    fill(host, 130)

    host._prune_index_cache()

    assert set(host._dataset_index_last_used) == set(host._dataset_per_file_index_cache)


def test_the_cap_never_drops_below_a_hundred(cache_file):
    """A user typing 1 into the limit box must not disable the index."""
    host = _CacheHost(cache_file, max_files=1)
    fill(host, 150)

    host._prune_index_cache()

    assert len(host._dataset_per_file_index_cache) == 100


def test_the_cap_is_enforced_even_when_everything_is_protected(cache_file):
    """Protection is a preference, not a licence to grow without bound."""
    host = _CacheHost(cache_file, max_files=100)
    fill(host, 130)
    host.tree_model_file = object()   # what protect_opened looks for
    host.opened_files = tuple(pathlib.Path(k) for k in host._dataset_per_file_index_cache)

    host._prune_index_cache(protect_opened=True)

    assert len(host._dataset_per_file_index_cache) == 100

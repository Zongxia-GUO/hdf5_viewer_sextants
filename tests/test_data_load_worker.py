"""The background loader, and the decision to stream a stack instead of reading it.

``DataLoadWorker`` is the producer of the lazy path whose consumer
(``ImageView2DEnhanced.set_data_lazy``) is covered in ``test_image_view_2d``.
The threshold here is what keeps a few hundred megabytes from crossing the
network on one tree click, and it was untested — the failure is not an error,
just a very long wait.

The worker's ``run`` is called directly rather than through ``start()``: the
signals then arrive on this thread, and the test is about what it decides, not
about Qt's thread plumbing.
"""

import h5py
import numpy as np
import pytest

from src.gui.main_window import DataLoadWorker

THRESHOLD = 2048   # bytes, small enough to cross with a toy dataset


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    monkeypatch.setattr("src.gui.main_window._LAZY_LOAD_THRESHOLD", THRESHOLD)


def h5_with(tmp_path, name, data):
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("stack", data=data)
    return path


def run(path, obj_path="stack", plot_type=""):
    """Run the worker and collect whichever signal it emitted."""
    worker = DataLoadWorker(str(path), obj_path, plot_type)
    out: dict = {"full": [], "lazy": [], "error": []}
    worker.data_ready.connect(lambda *a: out["full"].append(a))
    worker.data_ready_lazy.connect(lambda *a: out["lazy"].append(a))
    worker.load_error.connect(lambda msg: out["error"].append(msg))
    worker.run()
    return worker, out


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------

def test_a_small_stack_is_read_whole(tmp_path):
    path = h5_with(tmp_path, "small", np.zeros((4, 8, 8), dtype=np.float32))

    _w, out = run(path)

    assert len(out["full"]) == 1 and out["lazy"] == []
    assert out["full"][0][0].shape == (4, 8, 8)


def test_a_large_stack_streams_one_slice_at_a_time(tmp_path):
    path = h5_with(tmp_path, "large", np.zeros((10, 64, 64), dtype=np.float32))

    _w, out = run(path)

    assert len(out["lazy"]) == 1 and out["full"] == []
    first_slice, shape, _type, _file, _obj = out["lazy"][0]
    assert first_slice.shape == (64, 64), "only slice 0 crossed the network"
    assert tuple(shape) == (10, 64, 64), "the viewer still learns the full shape"


def test_a_single_frame_stored_as_3d_is_never_lazy(tmp_path):
    """(1, H, W) has no second slice to fetch, so streaming buys nothing."""
    big_one = np.zeros((1, 512, 512), dtype=np.float32)
    path = h5_with(tmp_path, "one", big_one)

    _w, out = run(path)

    assert len(out["full"]) == 1 and out["lazy"] == []


def test_a_large_2d_image_is_read_whole(tmp_path):
    """Lazy loading is per slice; a single image has nothing to slice."""
    path = h5_with(tmp_path, "image", np.zeros((512, 512), dtype=np.float32))

    _w, out = run(path)

    assert len(out["full"]) == 1 and out["lazy"] == []


def test_the_threshold_is_on_bytes_not_on_the_slice_count(tmp_path):
    """Many tiny slices stay cheap to read in one go."""
    path = h5_with(tmp_path, "many", np.zeros((50, 2, 2), dtype=np.float32))

    _w, out = run(path)

    assert len(out["full"]) == 1, "50 slices, but only 800 bytes"


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------

def test_the_emitted_identity_is_the_file_and_path_asked_for(tmp_path):
    path = h5_with(tmp_path, "ident", np.arange(6.0))

    _w, out = run(path)
    _data, _type, file_path, obj_path = out["full"][0]

    assert file_path == str(path)
    assert obj_path == "stack"


def test_an_explicit_plot_type_overrides_the_guess(tmp_path):
    path = h5_with(tmp_path, "typed", np.arange(6.0))

    _w, out = run(path, plot_type="Array2D")

    assert out["full"][0][1] == "Array2D"


def test_auto_lets_the_data_choose_its_type(tmp_path):
    path = h5_with(tmp_path, "auto", np.arange(6.0))

    _w, out = run(path, plot_type="Auto")

    assert out["full"][0][1] != "Auto"


def test_a_group_comes_back_as_its_member_names(tmp_path):
    """Clicking a group should list what is inside, not fail."""
    path = tmp_path / "grouped.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("scan")
        grp.create_dataset("a", data=np.arange(3.0))
        grp.create_dataset("b", data=np.arange(3.0))

    _w, out = run(path, obj_path="scan")

    data, type_str, _file, _obj = out["full"][0]
    assert type_str == "String"
    assert sorted(data.tolist()) == ["a", "b"]


def test_a_regular_data_file_is_loaded_without_h5py(tmp_path):
    path = tmp_path / "curve.txt"
    path.write_text("1 2\n3 4\n5 6\n", encoding="utf-8")

    _w, out = run(path, obj_path="/data")

    assert len(out["full"]) == 1
    assert out["full"][0][0].shape == (3, 2)


# ---------------------------------------------------------------------------
# Failure and cancellation
# ---------------------------------------------------------------------------

def test_a_missing_dataset_reports_an_error_instead_of_crashing(tmp_path):
    path = h5_with(tmp_path, "missing", np.arange(3.0))

    _w, out = run(path, obj_path="not_there")

    assert out["error"] and out["full"] == []


def test_an_unreadable_file_reports_an_error(tmp_path):
    path = tmp_path / "broken.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"garbage" * 10)

    _w, out = run(path)

    assert out["error"] and out["full"] == []


def test_a_cancelled_load_emits_nothing(tmp_path):
    """The tree fires a new load on every click; the old one must go quiet."""
    path = h5_with(tmp_path, "cancelled", np.zeros((4, 8, 8)))
    worker = DataLoadWorker(str(path), "stack")
    out: list = []
    worker.data_ready.connect(lambda *a: out.append(a))
    worker.data_ready_lazy.connect(lambda *a: out.append(a))
    worker.load_error.connect(lambda *a: out.append(a))

    worker.cancel()
    worker.run()

    assert out == []


def test_a_cancelled_load_does_not_report_its_own_failure(tmp_path):
    """Closing a file mid-read raises; that is not something to show the user."""
    worker = DataLoadWorker(str(tmp_path / "gone.h5"), "stack")
    errors: list = []
    worker.load_error.connect(errors.append)

    worker.cancel()
    worker.run()

    assert errors == []

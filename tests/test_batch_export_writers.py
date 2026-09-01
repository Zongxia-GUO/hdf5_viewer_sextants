"""Tests that the batch export writers honour the selected tabular dialect."""

import pathlib

import h5py
import numpy as np
import pytest

from src.gui.batch_export import (
    batch_target_labels,
    build_batch_targets,
    export_curve_combined_table,
    export_curve_dataset,
)
from src.lib_h5.table_format import get_table_format


@pytest.fixture
def scans(tmp_path: pathlib.Path):
    """Two scan files, each holding two curves (det1/det2) and a shared X axis."""
    files = []
    for idx, scan in enumerate(("0080", "0081")):
        path = tmp_path / f"scanx_{scan}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("curve", data=np.array([1.5, 2.25, 3.0]) + idx)
            f.create_dataset("det2", data=np.array([10.0, 20.0, 30.0]) + idx)
            f.create_dataset("xaxis", data=np.array([0.5, 1.5, 2.5]))
        files.append((path, scan))
    return files


def _targets(scans, paths):
    return build_batch_targets(scans, paths)


def _settings(tmp_path: pathlib.Path, fmt_key: str, **extra):
    base = {
        "output_dir": tmp_path,
        "table_format": get_table_format(fmt_key),
        "export_x": False,
        "share_x": False,
        "x_path": "",
        "shared_x_scan": "",
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Single-file export
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt_key, suffix, expected_first_row, bom",
    [
        ("txt", ".txt", "1.5\n", False),
        ("csv", ".csv", "1.5\n", True),
        ("csv2", ".csv", "1,5\n", True),
    ],
)
def test_single_export_uses_the_selected_dialect(tmp_path, scans, fmt_key, suffix, expected_first_row, bom):
    file_path, scan = scans[0]
    with h5py.File(file_path, "r") as f:
        data = np.asarray(f["curve"][()])

    export_curve_dataset(
        file_path, scan, "curve", data, _settings(tmp_path, fmt_key), scans
    )

    out = tmp_path / f"scanx_{scan}_curve{suffix}"
    assert out.exists()
    raw = out.read_bytes()
    assert (raw[:3] == b"\xef\xbb\xbf") is bom
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0] == f"scanx_{scan}_curve"
    assert lines[1] == expected_first_row.strip()


def test_txt_export_is_tab_separated(tmp_path, scans):
    file_path, scan = scans[0]
    with h5py.File(file_path, "r") as f:
        data = np.asarray(f["curve"][()])

    export_curve_dataset(
        file_path,
        scan,
        "curve",
        data,
        _settings(tmp_path, "txt", export_x=True, x_path="xaxis"),
        scans,
    )

    text = (tmp_path / f"scanx_{scan}_curve.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"scanx_{scan}_xaxis_X\tscanx_{scan}_curve"
    assert text.splitlines()[1] == "0.5\t1.5"


def test_csv2_export_is_semicolon_separated_with_comma_decimals(tmp_path, scans):
    """The French-Excel case: ';' splits the columns and '1,5' parses as a number."""
    file_path, scan = scans[0]
    with h5py.File(file_path, "r") as f:
        data = np.asarray(f["curve"][()])

    export_curve_dataset(
        file_path,
        scan,
        "curve",
        data,
        _settings(tmp_path, "csv2", export_x=True, x_path="xaxis"),
        scans,
    )

    text = (tmp_path / f"scanx_{scan}_curve.csv").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == f"scanx_{scan}_xaxis_X;scanx_{scan}_curve"
    assert text.splitlines()[1] == "0,5;1,5"
    # A decimal comma must never double as the field separator.
    assert text.splitlines()[1].count(";") == 1


# ---------------------------------------------------------------------------
# Combined table
# ---------------------------------------------------------------------------

def test_combined_table_honours_the_dialect(tmp_path, scans):
    ok, failures = export_curve_combined_table(
        _targets(scans, ["curve"]), scans, _settings(tmp_path, "csv2")
    )
    assert ok == 2 and not failures

    out = tmp_path / "scanx_0080_curve_batch.csv"
    text = out.read_text(encoding="utf-8-sig")
    header, first = text.splitlines()[:2]
    assert header == "scanx_0080_curve;scanx_0081_curve"
    assert first == "1,5;2,5"


def test_combined_table_defaults_to_txt_when_unset(tmp_path, scans):
    """A settings dict without a dialect falls back to the default rather than failing."""
    settings = _settings(tmp_path, "txt")
    del settings["table_format"]

    ok, failures = export_curve_combined_table(_targets(scans, ["curve"]), scans, settings)
    assert ok == 2 and not failures
    assert (tmp_path / "scanx_0080_curve_batch.txt").exists()


# ---------------------------------------------------------------------------
# Several Y datasets
# ---------------------------------------------------------------------------

def test_targets_are_scan_major(scans):
    targets = _targets(scans, ["curve", "det2"])
    assert [(t.file_path.stem, t.ds_path) for t in targets] == [
        ("scanx_0080", "curve"),
        ("scanx_0080", "det2"),
        ("scanx_0081", "curve"),
        ("scanx_0081", "det2"),
    ]


def test_single_dataset_keeps_the_plain_label(scans):
    assert [t.label for t in _targets(scans, ["curve"])] == ["scanx_0080", "scanx_0081"]


def test_several_datasets_get_a_leaf_tag(scans):
    labels = [t.label for t in _targets(scans, ["entry/curve", "entry/det2"])]
    assert labels == [
        "scanx_0080_curve", "scanx_0080_det2",
        "scanx_0081_curve", "scanx_0081_det2",
    ]


def test_colliding_leaves_fall_back_to_the_full_path(scans):
    labels = batch_target_labels(["a/data", "b/data"])
    assert labels == {"a/data": "a_data", "b/data": "b_data"}


def test_combined_table_holds_every_scan_and_dataset(tmp_path, scans):
    ok, failures = export_curve_combined_table(
        _targets(scans, ["curve", "det2"]), scans, _settings(tmp_path, "txt")
    )
    assert ok == 4 and not failures

    text = (tmp_path / "scanx_0080_curve_batch.txt").read_text(encoding="utf-8")
    header, first = text.splitlines()[:2]
    assert header.split("\t") == [
        "scanx_0080_curve", "scanx_0080_det2",
        "scanx_0081_curve", "scanx_0081_det2",
    ]
    assert first.split("\t") == ["1.5", "10", "2.5", "11"]


def test_combined_table_with_shared_x_writes_one_x_column(tmp_path, scans):
    ok, failures = export_curve_combined_table(
        _targets(scans, ["curve", "det2"]),
        scans,
        _settings(tmp_path, "txt", export_x=True, share_x=True, x_path="xaxis", shared_x_scan="0080"),
    )
    assert ok == 4 and not failures

    text = (tmp_path / "scanx_0080_curve_batch.txt").read_text(encoding="utf-8")
    header = text.splitlines()[0].split("\t")
    assert header[0] == "xaxis_X" and header.count("xaxis_X") == 1
    assert len(header) == 5
    assert text.splitlines()[1].split("\t") == ["0.5", "1.5", "10", "2.5", "11"]


def test_one_file_per_dataset_in_single_mode(tmp_path, scans):
    """Each (scan, dataset) writes its own file; the dataset name keeps them apart."""
    for target in _targets(scans, ["curve", "det2"]):
        with h5py.File(target.file_path, "r") as f:
            data = np.asarray(f[target.ds_path][()])
        export_curve_dataset(
            target.file_path, target.scan_num, target.ds_path, data,
            _settings(tmp_path, "txt"), scans,
        )

    written = sorted(p.name for p in tmp_path.glob("*.txt"))
    assert written == [
        "scanx_0080_curve.txt", "scanx_0080_det2.txt",
        "scanx_0081_curve.txt", "scanx_0081_det2.txt",
    ]


def test_a_missing_dataset_does_not_lose_the_others(tmp_path, scans):
    ok, failures = export_curve_combined_table(
        _targets(scans, ["curve", "nope"]), scans, _settings(tmp_path, "txt")
    )
    assert ok == 2
    assert len(failures) == 2 and "nope" in failures[0]

    header = (tmp_path / "scanx_0080_curve_batch.txt").read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == ["scanx_0080_curve", "scanx_0081_curve"]

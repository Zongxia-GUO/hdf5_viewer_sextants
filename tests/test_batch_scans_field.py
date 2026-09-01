"""Choosing which files a batch covers: keywords, then scan numbers.

Real filenames are not ``scanx_0340``. They look like
``Scan_ECL_5p0uJIR_050``: the family at the front, the number at the back, and
something that varies in between. The old matcher wanted the number
immediately after the prefix, so those files matched **nothing at all**.

The two fields are matched by deliberately different rules — a keyword is a
substring, a scan number is a whole part of the name — and the tests for why
are the point of this module.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from src.gui.batch_export import (
    batch_folder_conflict,
    batch_number_ambiguity,
    common_keyword,
    compress_scan_numbers,
    describe_number_ambiguity,
    parse_keywords,
    scan_number_in_stem,
    scan_stem_parts,
    stem_matches_keywords,
    summarise_number_ambiguity,
)

REAL_STEMS = [
    "Scan_ECL_5p0uJIR_047",
    "Scan_ECL_5p0uJIR_050",
    "Scan_ECR_5p0uJIR_047",
    "Scan_ECL_10p0uJIR_047",
]


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, keywords", [
    ("Scan_", ["scan_"]),
    ("ECL 5p0uJIR", ["ecl", "5p0ujir"]),
    ("  ECL   5p0uJIR  ", ["ecl", "5p0ujir"]),
    ("", []),
    ("   ", []),
])
def test_keywords_are_split_on_whitespace(text, keywords):
    """Not commas: the field beside this one uses commas to mean *any of*,
    and here they would mean *all of*. One mark, two meanings, is a trap."""
    assert parse_keywords(text) == keywords


def test_a_keyword_matches_anywhere_in_the_name():
    """ECL sits in the middle, so a prefix test could never find it."""
    assert stem_matches_keywords("Scan_ECL_5p0uJIR_050", ["ecl"])
    assert stem_matches_keywords("Scan_ECL_5p0uJIR_050", ["5p0ujir"])
    assert not stem_matches_keywords("Scan_ECR_5p0uJIR_050", ["ecl"])


def test_a_keyword_need_not_be_a_whole_part_of_the_name():
    """So a naming scheme without separators still works."""
    assert stem_matches_keywords("ScanECL050", ["ecl"])


def test_case_is_ignored():
    assert stem_matches_keywords("Scan_ECL_5p0uJIR_050", parse_keywords("ecl"))
    assert stem_matches_keywords("scan_ecl_5p0ujir_050", parse_keywords("ECL"))


def test_every_keyword_has_to_match():
    stem = "Scan_ECL_5p0uJIR_050"

    assert stem_matches_keywords(stem, parse_keywords("ECL 5p0uJIR"))
    assert not stem_matches_keywords(stem, parse_keywords("ECL 10p0uJIR"))


def test_no_keywords_matches_everything():
    assert stem_matches_keywords("anything", [])


# ---------------------------------------------------------------------------
# Scan numbers — a whole part of the name, not a substring
# ---------------------------------------------------------------------------

def test_a_number_matches_a_whole_part_of_the_name():
    assert scan_number_in_stem("Scan_ECL_5p0uJIR_050", ["050"]) == "050"


def test_a_number_does_not_match_inside_a_longer_one():
    """As a substring, 050 would also pick up 1050 — a different scan."""
    assert scan_number_in_stem("Scan_ECL_5p0uJIR_1050", ["050"]) is None


def test_a_short_number_does_not_match_everything():
    """As a substring, '0' appears in 5p0uJIR and in every padded number, so it
    would select the whole folder."""
    for stem in REAL_STEMS + ["scanx_0340"]:
        assert scan_number_in_stem(stem, ["0"]) is None


def test_zero_padding_does_not_have_to_be_remembered():
    assert scan_number_in_stem("Scan_ECL_5p0uJIR_047", ["47"]) == "47"
    assert scan_number_in_stem("Scan_ECL_5p0uJIR_047", ["047"]) == "047"


def test_a_padded_request_does_not_match_a_different_number():
    assert scan_number_in_stem("Scan_ECL_5p0uJIR_470", ["47"]) is None


def test_the_simple_naming_still_works():
    """scanx_0340 is what this used to be written for."""
    assert scan_number_in_stem("scanx_0340", ["0340"]) == "0340"


def test_a_trailing_word_does_not_hide_the_number():
    """scanx_0340_bkg matched before this change and has to keep matching."""
    assert scan_number_in_stem("scanx_0340_bkg", ["0340"]) == "0340"


def test_none_of_the_requested_numbers():
    assert scan_number_in_stem("Scan_ECL_5p0uJIR_050", ["047", "048"]) is None


# ---------------------------------------------------------------------------
# Reading a dropped filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem, parts", [
    ("scanx_0340", ("scanx_", "0340")),
    ("Scan_ECL_5p0uJIR_050", ("Scan_ECL_5p0uJIR_", "050")),
    ("0340", ("", "0340")),
    ("no_number_here", None),
    ("", None),
])
def test_a_filename_splits_into_prefix_and_trailing_number(stem, parts):
    assert scan_stem_parts(stem) == parts


@pytest.mark.parametrize("numbers, text", [
    (["0340", "0341", "0342"], "0340-0342"),
    (["0340", "0341", "0342", "0350"], "0340-0342,0350"),
    (["0350", "0342", "0341", "0340"], "0340-0342,0350"),
    (["047", "050"], "047,050"),
    (["0340"], "0340"),
    (["0340", "0340"], "0340"),
    ([], ""),
])
def test_numbers_are_written_as_spans(numbers, text):
    assert compress_scan_numbers(numbers) == text


def test_numbers_of_different_width_do_not_merge():
    """'99' and '0100' are consecutive as integers but not as filenames."""
    assert compress_scan_numbers(["99", "0100"]) == "99,0100"


# ---------------------------------------------------------------------------
# The keyword a drop derives
# ---------------------------------------------------------------------------

def test_two_families_fall_back_to_what_they_really_share():
    """ECL and ECR share 'EC' by accident; half a word is not a keyword."""
    assert common_keyword(["Scan_ECL_5p0uJIR_050", "Scan_ECR_5p0uJIR_047"]) == "Scan_"


def test_one_family_keeps_its_own_name():
    assert common_keyword([
        "Scan_ECL_5p0uJIR_047", "Scan_ECL_5p0uJIR_050",
    ]) == "Scan_ECL_5p0uJIR_"


def test_a_single_file_drops_only_its_number():
    """The number is the other field's business."""
    assert common_keyword(["Scan_ECL_5p0uJIR_050"]) == "Scan_ECL_5p0uJIR_"


def test_nothing_dropped_gives_nothing():
    assert common_keyword([]) == ""


# ---------------------------------------------------------------------------
# One number, several files
# ---------------------------------------------------------------------------

def _matches(*names):
    out = []
    for name in names:
        path = pathlib.Path("d:/beamtime") / f"{name}.hdf5"
        out.append((path, scan_stem_parts(name)[1]))
    return out


def test_one_file_per_number_is_no_ambiguity():
    assert batch_number_ambiguity(_matches(
        "Scan_ECL_5p0uJIR_047", "Scan_ECL_5p0uJIR_050",
    )) == {}


def test_the_same_number_in_two_families_is_reported():
    """Allowed — they are written under their own names — but said out loud."""
    ambiguity = batch_number_ambiguity(_matches(
        "Scan_ECL_5p0uJIR_047", "Scan_ECL_10p0uJIR_047",
    ))

    assert set(ambiguity) == {"047"}
    assert len(ambiguity["047"]) == 2


def test_the_report_names_the_files():
    ambiguity = batch_number_ambiguity(_matches(
        "Scan_ECL_5p0uJIR_047", "Scan_ECL_10p0uJIR_047",
    ))
    detail = describe_number_ambiguity(ambiguity)

    assert "Scan_ECL_5p0uJIR_047" in detail
    assert "Scan_ECL_10p0uJIR_047" in detail
    assert "nothing is overwritten" in detail
    assert "047" in summarise_number_ambiguity(ambiguity)


def test_no_ambiguity_says_nothing():
    assert describe_number_ambiguity({}) == ""
    assert summarise_number_ambiguity({}) == ""


# ---------------------------------------------------------------------------
# One folder per batch
# ---------------------------------------------------------------------------

def test_one_folder_is_no_conflict():
    assert batch_folder_conflict([
        (pathlib.Path("d:/a/scanx_0340.h5"), "0340"),
        (pathlib.Path("d:/a/scanx_0341.h5"), "0341"),
    ]) == ""


def test_two_folders_are_refused_and_both_are_named():
    message = batch_folder_conflict([
        (pathlib.Path("d:/beamtime_A/scanx_0340.h5"), "0340"),
        (pathlib.Path("d:/beamtime_B/scanx_0340.h5"), "0340"),
    ])

    assert "beamtime_A" in message and "beamtime_B" in message
    assert "one folder" in message


# ---------------------------------------------------------------------------
# End to end, through the window
# ---------------------------------------------------------------------------

@pytest.fixture
def beamtime(tmp_path):
    """A folder of files named the way the real ones are."""
    folder = tmp_path / "beamtime"
    folder.mkdir()
    for stem in REAL_STEMS:
        with h5py.File(folder / f"{stem}.hdf5", "w") as f:
            f.create_dataset("scan_data/data_04", data=np.arange(5.0))
    return folder


@pytest.fixture
def window(qapp, beamtime):
    from src.gui.main_window import MainWindow

    win = MainWindow()
    for path in sorted(beamtime.glob("*.hdf5")):
        win._open_file(path)
    yield win
    win.close()


def test_the_real_filenames_now_match(window):
    """They matched nothing at all before: the number is not adjacent to the
    prefix, and the old rule required it to be."""
    matched = window._matching_files_for_scans("Scan_", ["047", "050"])

    assert sorted(p.stem for p, _n in matched) == sorted(REAL_STEMS)


def test_a_keyword_narrows_to_one_family(window):
    matched = window._matching_files_for_scans("ECL 5p0uJIR", ["047", "050"])

    assert sorted(p.stem for p, _n in matched) == [
        "Scan_ECL_5p0uJIR_047", "Scan_ECL_5p0uJIR_050",
    ]


def test_a_keyword_in_the_middle_works(window):
    matched = window._matching_files_for_scans("ECR", ["047"])

    assert [p.stem for p, _n in matched] == ["Scan_ECR_5p0uJIR_047"]


def test_the_batch_resolves_with_both_fields(window, beamtime):
    window.le_batch_keywords.setText("ECL 5p0uJIR")
    window.le_scan_range.setText("047,050")
    window.le_batch_path.setText("scan_data/data_04")

    resolved = window._resolve_batch_selection(quiet=True)

    assert resolved is not None
    matching, _paths, scans = resolved
    assert scans == ["047", "050"]
    assert len(matching) == 2


def test_a_loose_keyword_reports_the_ambiguity_on_the_status_bar(window):
    """Never a modal here — this runs on every Enter while still typing."""
    window.le_batch_keywords.setText("Scan_")
    window.le_scan_range.setText("047")
    window.le_batch_path.setText("scan_data/data_04")

    resolved = window._resolve_batch_selection(quiet=True)

    assert resolved is not None, "allowed, not refused"
    assert "047" in window._menu_status_raw_text
    assert "more than one file" in window._menu_status_raw_text


def test_add_to_calculator_a_and_b_splits_the_two_scans(window, beamtime, monkeypatch):
    """It narrows the numbers field to one scan at a time. It went on driving
    the merged field that the two-field revert removed, so the menu item raised
    AttributeError the moment it was used — invisible to flake8, because an
    attribute access on self is valid whatever the name."""
    added: list[tuple[str, str]] = []
    monkeypatch.setattr(
        type(window), "_batch_add_to_tool",
        lambda self, tool: added.append((tool, self.le_scan_range.text())),
    )
    window.le_batch_keywords.setText("ECL 5p0uJIR")
    window.le_scan_range.setText("047,050")
    window.le_batch_path.setText("scan_data/data_04")

    window._batch_add_to_calculator_ab()

    assert added == [("calculator_a", "047"), ("calculator_b", "050")]
    assert window.le_scan_range.text() == "047,050", "the field is put back"
    assert window.le_batch_keywords.text() == "ECL 5p0uJIR", "only the numbers move"


def test_narrowing_the_keyword_clears_the_warning(window):
    """A warning left over from the wider keyword would say this batch is
    ambiguous when narrowing it has just fixed that."""
    window.le_batch_path.setText("scan_data/data_04")
    window.le_scan_range.setText("047")
    window.le_batch_keywords.setText("Scan_")
    window._resolve_batch_selection(quiet=True)
    assert "more than one file" in window._menu_status_raw_text

    window.le_batch_keywords.setText("ECL 5p0uJIR")
    window._resolve_batch_selection(quiet=True)

    assert window._menu_status_raw_text == ""


# ---------------------------------------------------------------------------
# Dropping onto each field
# ---------------------------------------------------------------------------

class _Mime:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def hasText(self):
        return bool(self._text)


class _Drop:
    def __init__(self, text):
        self._mime = _Mime(text)
        self.accepted = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True


def _drop_of(beamtime, *stems):
    return _Drop("\n".join(str(beamtime / f"{stem}.hdf5") for stem in stems))


def test_dropping_on_the_keywords_fills_only_the_keywords(window, beamtime):
    """Each field fills only itself, so a drop cannot quietly widen a filter
    that was typed by hand."""
    window.le_scan_range.setText("0001")

    window._batch_keywords_drop(_drop_of(beamtime, "Scan_ECL_5p0uJIR_047", "Scan_ECL_5p0uJIR_050"))

    assert window.le_batch_keywords.text() == "Scan_ECL_5p0uJIR_"
    assert window.le_scan_range.text() == "0001", "untouched"


def test_dropping_on_the_numbers_fills_only_the_numbers(window, beamtime):
    window.le_batch_keywords.setText("ECL")

    window._batch_scan_range_drop(_drop_of(beamtime, "Scan_ECL_5p0uJIR_047", "Scan_ECL_5p0uJIR_050"))

    assert window.le_scan_range.text() == "047,050"
    assert window.le_batch_keywords.text() == "ECL", "untouched"


def test_dropping_two_families_widens_the_keyword_to_what_they_share(window, beamtime):
    window._batch_keywords_drop(_drop_of(beamtime, "Scan_ECL_5p0uJIR_050", "Scan_ECR_5p0uJIR_047"))

    assert window.le_batch_keywords.text() == "Scan_"


def test_dropping_a_dataset_uses_only_its_file(window, beamtime):
    path = beamtime / "Scan_ECL_5p0uJIR_050.hdf5"
    window._batch_scan_range_drop(_Drop(f"{path}::scan_data/data_04"))

    assert window.le_scan_range.text() == "050"


def test_dropping_across_folders_is_refused(window, beamtime, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warned.append(a[1]) or QMessageBox.StandardButton.Ok),
    )
    window.le_scan_range.setText("0001")

    window._batch_scan_range_drop(_Drop(
        f"{beamtime / 'Scan_ECL_5p0uJIR_047.hdf5'}\n{tmp_path / 'other' / 'Scan_ECL_5p0uJIR_050.hdf5'}"
    ))

    assert warned and "Folders" in warned[0]
    assert window.le_scan_range.text() == "0001"


def test_a_file_without_a_number_is_reported_not_guessed(window, tmp_path):
    path = tmp_path / "alignment.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(3.0))
    window.le_scan_range.setText("0001")

    window._batch_scan_range_drop(_Drop(str(path)))

    assert window.le_scan_range.text() == "0001"
    assert "scan number" in window._menu_status_raw_text


def test_an_empty_drop_changes_nothing(window):
    window.le_batch_keywords.setText("ECL")
    window.le_scan_range.setText("0340")

    window._batch_keywords_drop(_Drop(""))
    window._batch_scan_range_drop(_Drop(""))

    assert window.le_batch_keywords.text() == "ECL"
    assert window.le_scan_range.text() == "0340"

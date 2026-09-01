"""The calculator takes a part out of a stack, the way it does out of a table.

Element-wise arithmetic on whole stacks already worked — `A - B` on two
(3,4,5) arrays gave (3,4,5). What was missing was any way to say *which frame*:
a 3-D dataset left the part selector disabled, showing "N/A".

The row under each dataset asks one question in two shapes: take a column out
of a table, or a frame out of a stack, or use the whole thing.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.gui.data_calculator_enhanced import (
    SELECT_ALL,
    DataCalculatorEnhanced,
    operand_from_selection,
)

# Distinct along every axis, so a frame cut the wrong way is obvious.
STACK = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
TABLE = np.arange(12.0).reshape(4, 3)


# ---------------------------------------------------------------------------
# Cutting an operand down
# ---------------------------------------------------------------------------

def test_a_whole_stack_is_left_alone():
    np.testing.assert_array_equal(operand_from_selection(STACK, 0, SELECT_ALL), STACK)


def test_one_frame_of_the_first_axis():
    np.testing.assert_array_equal(operand_from_selection(STACK, 0, 2), STACK[2])


def test_one_frame_of_a_chosen_axis():
    np.testing.assert_array_equal(operand_from_selection(STACK, 1, 3), STACK[:, 3])
    np.testing.assert_array_equal(operand_from_selection(STACK, 2, 4), STACK[:, :, 4])


def test_a_chosen_axis_is_moved_to_the_front():
    """So the result is an ordinary frame-major stack that the viewer, the
    export and the pattern tool all read without being told anything."""
    result = operand_from_selection(STACK, 2, SELECT_ALL)

    assert result.shape == (5, 3, 4)
    np.testing.assert_array_equal(result[4], STACK[:, :, 4])


def test_the_first_axis_is_not_reordered():
    """Leaving the axis alone must change nothing at all for existing use."""
    result = operand_from_selection(STACK, 0, SELECT_ALL)

    assert result.shape == STACK.shape
    assert result is STACK or np.shares_memory(result, STACK)


def test_a_frame_index_past_the_end_is_clamped():
    np.testing.assert_array_equal(operand_from_selection(STACK, 0, 99), STACK[2])


def test_a_table_still_gives_up_a_column():
    np.testing.assert_array_equal(operand_from_selection(TABLE, 0, 1), TABLE[:, 1])
    np.testing.assert_array_equal(operand_from_selection(TABLE, 0, SELECT_ALL), TABLE)


def test_a_curve_is_returned_whole():
    curve = np.arange(6.0)
    np.testing.assert_array_equal(operand_from_selection(curve, 0, 2), curve)


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scanx_0340.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("ecl", data=STACK)
        f.create_dataset("ecr", data=STACK + 100.0)
        f.create_dataset("dark", data=np.ones((4, 5)))
        f.create_dataset("table", data=TABLE)
        f.create_dataset("curve", data=np.arange(6.0))
    return path


@pytest.fixture
def calc(qapp, scan):
    tool = DataCalculatorEnhanced((scan,))
    yield tool
    tool.deleteLater()


def test_a_stack_offers_an_axis_and_its_frames(calc, scan):
    """It used to leave this selector disabled, showing 'N/A'."""
    calc.add_to_dataset_a(f"{scan}::ecl")

    assert calc.combo_axis_a.isVisibleTo(calc) is True
    # Bare numbers: the shape is printed on the line above, so repeating each
    # axis's length here only doubles the width of the box.
    assert [calc.combo_axis_a.itemText(i) for i in range(calc.combo_axis_a.count())] == [
        "0", "1", "2",
    ]
    assert calc.spin_col_a.isEnabled()
    assert calc.spin_col_a.itemText(0) == "All"
    assert calc.spin_col_a.count() == 1 + 3
    assert calc.lbl_part_a.text() == "Slice:"
    assert calc.lbl_part_a.axis_label.isVisibleTo(calc) is True


def test_a_table_still_offers_its_columns(calc, scan):
    calc.add_to_dataset_a(f"{scan}::table")

    assert calc.combo_axis_a.isVisibleTo(calc) is False
    assert calc.lbl_part_a.axis_label.isVisibleTo(calc) is False
    assert calc.spin_col_a.itemText(0) == "All"
    assert calc.spin_col_a.count() == 1 + 3
    assert calc.lbl_part_a.text() == "Column:"


def test_a_curve_offers_nothing(calc, scan):
    calc.add_to_dataset_a(f"{scan}::curve")

    assert calc.spin_col_a.isEnabled() is False
    assert calc.spin_col_a.itemText(0) == "N/A"


def test_choosing_an_axis_relists_the_frames(calc, scan):
    """A different axis, a different number of frames."""
    calc.add_to_dataset_a(f"{scan}::ecl")

    calc.combo_axis_a.setCurrentIndex(2)

    assert calc.spin_col_a.count() == 1 + 5


def test_the_shape_note_says_how_many_slices(calc, scan):
    calc.add_to_dataset_a(f"{scan}::ecl")

    assert "3 slices" in calc.label_info_a.text()


# ---------------------------------------------------------------------------
# Calculating
# ---------------------------------------------------------------------------

def test_two_whole_stacks_still_compute_element_wise(calc, scan):
    calc.add_to_dataset_a(f"{scan}::ecr")
    calc.add_to_dataset_b(f"{scan}::ecl")

    calc._perform_operation("A - B")

    assert calc.result_data.shape == (3, 4, 5)
    np.testing.assert_allclose(calc.result_data, 100.0)


def test_one_frame_against_another(calc, scan):
    calc.add_to_dataset_a(f"{scan}::ecr")
    calc.add_to_dataset_b(f"{scan}::ecl")
    calc.spin_col_a.setCurrentIndex(1 + 2)     # frame 2
    calc.spin_col_b.setCurrentIndex(1 + 0)     # frame 0

    calc._perform_operation("A - B")

    assert calc.result_data.shape == (4, 5)
    np.testing.assert_allclose(calc.result_data, STACK[2] + 100.0 - STACK[0])


def test_one_frame_is_subtracted_from_every_frame(calc, scan):
    """A single frame against a whole stack broadcasts — which is exactly a
    background or dark-frame subtraction, with no formula to write."""
    calc.add_to_dataset_a(f"{scan}::ecl")
    calc.add_to_dataset_b(f"{scan}::ecl")
    calc.spin_col_b.setCurrentIndex(1 + 0)     # frame 0 of B

    calc._perform_operation("A - B")

    assert calc.result_data.shape == (3, 4, 5)
    np.testing.assert_array_equal(calc.result_data[0], np.zeros((4, 5)))
    np.testing.assert_array_equal(calc.result_data[1], STACK[1] - STACK[0])


def test_a_two_dimensional_operand_broadcasts_over_the_stack(calc, scan):
    calc.add_to_dataset_a(f"{scan}::ecl")
    calc.add_to_dataset_b(f"{scan}::dark")

    calc._perform_operation("A - B")

    assert calc.result_data.shape == (3, 4, 5)
    np.testing.assert_array_equal(calc.result_data, STACK - 1.0)


def test_choosing_an_axis_reorients_the_result(calc, scan):
    calc.add_to_dataset_a(f"{scan}::ecl")
    calc.combo_axis_a.setCurrentIndex(2)

    calc._perform_operation("A * 2")

    assert calc.result_data.shape == (5, 3, 4)
    np.testing.assert_array_equal(calc.result_data[4], STACK[:, :, 4] * 2)


def test_broadcasting_shapes_are_not_called_a_mismatch(qapp, scan, monkeypatch):
    """Asking "attempt anyway?" for a dark-frame subtraction would make the most
    ordinary use of a stack read like a mistake."""
    from PyQt6.QtWidgets import QMessageBox

    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: asked.append(a[1]) or QMessageBox.StandardButton.Yes),
    )
    tool = DataCalculatorEnhanced((scan,))
    try:
        tool.add_to_dataset_a(f"{scan}::ecl")
        tool.add_to_dataset_b(f"{scan}::dark")

        tool._perform_operation("A - B")

        assert asked == []
        assert tool.result_data.shape == (3, 4, 5)
    finally:
        tool.deleteLater()


def test_shapes_that_cannot_pair_up_are_still_questioned(qapp, scan, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: asked.append(a[1]) or QMessageBox.StandardButton.No),
    )
    tool = DataCalculatorEnhanced((scan,))
    try:
        tool.add_to_dataset_a(f"{scan}::table")     # (4, 3)
        tool.add_to_dataset_b(f"{scan}::curve")     # (6,)

        tool._perform_operation("A - B")

        assert asked == ["Shape Mismatch"]
    finally:
        tool.deleteLater()


def test_a_column_of_a_table_still_works(calc, scan):
    calc.add_to_dataset_a(f"{scan}::table")
    calc.spin_col_a.setCurrentIndex(1 + 1)     # column 1

    calc._perform_operation("A * 2")

    np.testing.assert_array_equal(calc.result_data, TABLE[:, 1] * 2)


# ---------------------------------------------------------------------------
# The result viewer
# ---------------------------------------------------------------------------

def test_the_result_viewer_hides_its_axis_selector(calc, scan):
    """The axis was chosen per operand and the result reoriented to match, so a
    second control for it would only re-cut a settled question."""
    calc.add_to_dataset_a(f"{scan}::ecl")
    calc._perform_operation("A * 2")

    shown = calc.result_widget.get_current_widget()

    assert shown.combo_slice_axis.isHidden()
    assert shown.slider_slice.isHidden() is False, "browsing the frames still works"


# ---------------------------------------------------------------------------
# Room for the two selectors
# ---------------------------------------------------------------------------

def test_neither_selector_is_clipped(calc, scan, qapp):
    """A combo settles its width the first time it is shown, which here is
    before it has any items — so the entries came out elided."""
    from PyQt6.QtWidgets import QApplication

    calc.show()
    calc.add_to_dataset_a(f"{scan}::ecl")
    QApplication.processEvents()
    try:
        for combo in (calc.combo_axis_a, calc.spin_col_a):
            longest = max((combo.itemText(i) for i in range(combo.count())), key=len)
            text_width = combo.fontMetrics().horizontalAdvance(longest)

            assert combo.width() >= combo.sizeHint().width(), "clipped"
            assert combo.width() > text_width, "no room left for the dropdown arrow"
    finally:
        calc.close()


def test_the_two_selectors_stay_narrow(calc, scan, qapp):
    """They hold "0" and "All", not sentences. Carrying the words in every
    entry made the pair twice as wide as anything it could show."""
    from PyQt6.QtWidgets import QApplication

    from src.gui.data_calculator_enhanced import (
        AXIS_COMBO_MAX_WIDTH,
        PART_COMBO_MAX_WIDTH,
    )

    calc.show()
    calc.add_to_dataset_a(f"{scan}::ecl")
    QApplication.processEvents()
    try:
        together = calc.combo_axis_a.width() + calc.spin_col_a.width()

        assert together <= AXIS_COMBO_MAX_WIDTH + PART_COMBO_MAX_WIDTH
        assert together < 200, "the pair used to take 330px"
    finally:
        calc.close()


def test_the_viewer_keeps_its_axis_selector_everywhere_else(qapp):
    """Only the calculator's result view gives it up."""
    from src.gui.image_view_2d_enhanced import ImageView2DEnhanced

    view = ImageView2DEnhanced()
    try:
        view.set_data(STACK)
        assert view.combo_slice_axis.isVisibleTo(view) is True
    finally:
        view.deleteLater()

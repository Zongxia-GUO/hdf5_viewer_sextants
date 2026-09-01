"""Tests for the 2D image viewer.

It is the most reused widget in the application — the main window, the batch
export preview, Q calibration, FTH/CDI and frame analysis all embed it — and was
until now the least covered, so a change here could break five places silently.

These pin the parts other code depends on: what ``set_data`` does with each
shape, how slices are served (loaded and lazy), the scale transforms, the
contrast rule, and the RGB the export and clipboard paths both go through.
"""

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from src.gui.image_view_2d_enhanced import ImageView2DEnhanced


@pytest.fixture
def view(qapp):
    return ImageView2DEnhanced()


def frame(h=8, w=6, start=0.0):
    return np.arange(start, start + h * w, dtype=float).reshape(h, w)


# ---------------------------------------------------------------------------
# set_data: shapes
# ---------------------------------------------------------------------------

def test_a_2d_array_is_shown_without_slice_controls(view):
    view.set_data(frame())

    assert view.data_3d is None
    assert view.data.shape == (8, 6)
    assert not view.slider_slice.isVisibleTo(view)


def test_data_is_held_as_float32_to_halve_the_memory(view):
    view.set_data(frame().astype(np.float64))
    assert view.data.dtype == np.float32


def test_a_3d_stack_turns_on_slice_browsing(view):
    view.set_data(np.stack([frame(start=s * 100) for s in range(5)]))

    assert view.data_3d is not None
    assert view.data.shape == (8, 6), "the first slice is displayed"
    assert view.slider_slice.maximum() == 4


def test_a_singleton_leading_axis_is_not_a_stack(view):
    """(1, H, W) is one image stored awkwardly, not a one-slice stack."""
    view.set_data(frame().reshape(1, 8, 6))

    assert view.data_3d is None
    assert view.data.shape == (8, 6)


def test_the_slice_axis_selector_offers_every_axis_with_its_length(view):
    """Numbered, not X/Y/Z: axis 0 of a stack is the frame — time, energy,
    whatever was scanned — and the image's own X is the *last* axis, which the
    old scheme called Z."""
    view.set_data(np.zeros((4, 8, 6)))

    labels = [view.combo_slice_axis.itemText(i) for i in range(view.combo_slice_axis.count())]
    assert labels == ["0 (4)", "1 (8)", "2 (6)"]


def test_a_stack_stored_as_h_w_n_can_still_be_browsed(view):
    """Stacks arrive as (N,H,W) and (H,W,N); the axis selector is the answer."""
    view.set_data(np.zeros((64, 64, 10)))

    assert view.combo_slice_axis.count() == 3
    assert view.combo_slice_axis.itemText(2) == "2 (10)"


def test_global_levels_are_measured_over_the_whole_stack(view):
    stack = np.stack([np.full((4, 4), s) for s in range(1, 6)]).astype(float)
    view.set_data(stack)

    assert view.global_min == 1.0
    assert view.global_max == 5.0


def test_loading_new_data_drops_the_previous_lock(view):
    view.set_data(np.zeros((3, 4, 4)))
    view.chk_lock_levels.setChecked(True)

    view.set_data(frame())

    assert not view.chk_lock_levels.isChecked()
    assert view.locked_levels is None


# ---------------------------------------------------------------------------
# Slice navigation
# ---------------------------------------------------------------------------

def test_moving_the_slider_shows_that_slice(view):
    view.set_data(np.stack([np.full((4, 4), s, dtype=float) for s in range(5)]))

    view.slider_slice.setValue(3)

    assert float(view.data[0, 0]) == 3.0


def test_the_slice_axis_decides_what_a_slice_is(view):
    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    view.set_data(data)

    view.current_slice_axis = 2
    np.testing.assert_allclose(view._extract_full_slice(1), data[:, :, 1])


# ---------------------------------------------------------------------------
# Lazy slices: the network path
# ---------------------------------------------------------------------------

def test_a_lazy_stack_only_holds_the_first_slice(view):
    calls: list[tuple] = []

    def loader(axis, index):
        calls.append((axis, index))
        return np.full((4, 4), index, dtype=float)

    view.set_data_lazy(np.zeros((4, 4)), total_slices=20, loader=loader)

    assert view.data_3d is None, "nothing but slice 0 was fetched"
    assert calls == []
    assert view.slider_slice.maximum() == 19


def test_a_fetched_lazy_slice_is_cached(view):
    calls: list[int] = []

    def loader(axis, index):
        calls.append(index)
        return np.full((4, 4), index, dtype=float)

    view.set_data_lazy(np.zeros((4, 4)), total_slices=20, loader=loader)
    view.slider_slice.setValue(7)
    view.slider_slice.setValue(0)
    view.slider_slice.setValue(7)

    assert calls == [7], "the second visit is served from the cache"


def test_the_lazy_cache_is_bounded(view):
    """It exists to spare the network, not to re-download the dataset into RAM."""
    view.set_data_lazy(np.zeros((4, 4)), total_slices=50,
                       loader=lambda axis, index: np.full((4, 4), index, dtype=float))

    for index in range(1, 20):
        view.slider_slice.setValue(index)

    assert len(view._slice_cache) <= 8


def test_an_old_single_argument_loader_still_works(view):
    """Older callers pass loader(idx); axis 0 must keep working."""
    view.set_data_lazy(np.zeros((4, 4)), total_slices=5,
                       loader=lambda index: np.full((4, 4), index * 10, dtype=float))

    view.slider_slice.setValue(2)

    assert float(view.data[0, 0]) == 20.0


def test_a_failing_loader_keeps_the_previous_slice_on_screen(view):
    def loader(axis, index):
        raise OSError("connection dropped")

    view.set_data_lazy(np.full((4, 4), 42.0), total_slices=5, loader=loader)
    view.slider_slice.setValue(3)

    assert float(view.data[0, 0]) == 42.0, "a dead link must not blank the viewer"


# ---------------------------------------------------------------------------
# Scale transforms
# ---------------------------------------------------------------------------

def test_linear_is_the_identity(view):
    data = frame()
    view.combo_scale.setCurrentText("Linear")

    np.testing.assert_allclose(view._transform_data(data), data)


def test_log_never_produces_nan_from_zero_or_negative_values(view):
    """Detector frames are full of zeros; log10(0) would poison the levels."""
    data = np.array([[-5.0, 0.0], [1.0, 100.0]])
    view.combo_scale.setCurrentText("Log")

    out = view._transform_data(data)

    assert np.all(np.isfinite(out))
    assert out[1, 1] == pytest.approx(2.0)


def test_symlog_keeps_the_sign_of_a_difference_image(view):
    data = np.array([[-99.0, 0.0], [9.0, 0.0]])
    view.combo_scale.setCurrentText("SymLog")

    out = view._transform_data(data)

    assert out[0, 0] < 0 and out[1, 0] > 0
    assert out[0, 1] == 0.0
    assert out[1, 0] == pytest.approx(1.0)


def test_square_root_does_not_choke_on_negatives(view):
    view.combo_scale.setCurrentText("Square root")

    out = view._transform_data(np.array([[-4.0, 9.0]]))

    assert np.all(np.isfinite(out))
    assert out[0, 1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Auto contrast
# ---------------------------------------------------------------------------

def test_a_flat_image_still_gets_a_usable_range(view):
    """Equal levels would make pyqtgraph divide by zero."""
    levels = ImageView2DEnhanced._robust_auto_levels(np.full(100, 7.0))

    assert levels is not None
    assert levels[0] < levels[1]


def test_all_nan_data_has_no_levels_to_offer(view):
    assert ImageView2DEnhanced._robust_auto_levels(np.full(10, np.nan)) is None


def test_a_negative_only_image_keeps_its_upper_edge(view):
    levels = ImageView2DEnhanced._robust_auto_levels(np.linspace(-100.0, -10.0, 1000))

    assert levels is not None
    assert levels[1] == -10.0


def test_hot_pixels_do_not_flatten_the_contrast(view):
    data = np.concatenate([np.linspace(0.0, 10.0, 999), [1e9]])

    levels = ImageView2DEnhanced._robust_auto_levels(data)

    assert levels is not None
    assert levels[1] < 1e6, "the 99.5th percentile, not the maximum"


def test_auto_contrast_records_the_levels_when_locked(view):
    view.set_data(np.stack([frame(start=s * 100) for s in range(3)]))
    view.chk_lock_levels.setChecked(True)

    view._auto_contrast()

    assert view.locked_levels is not None


# ---------------------------------------------------------------------------
# The colormap follows the kind of data
# ---------------------------------------------------------------------------

def test_magnitude_data_gets_a_sequential_map(view):
    from src.gui.image_view_2d_enhanced import SEQUENTIAL_COLORMAP

    view.set_data(np.abs(np.random.RandomState(0).randn(32, 32)))

    assert view.combo_colormap.currentText() == SEQUENTIAL_COLORMAP


def test_data_that_spans_zero_gets_a_diverging_map(view):
    """A difference image needs two hues meeting at a neutral midpoint, or the
    sign of a pixel cannot be read off the picture."""
    from src.gui.image_view_2d_enhanced import DIVERGING_COLORMAP

    view.set_data(np.random.RandomState(0).randn(32, 32))

    assert view.combo_colormap.currentText() == DIVERGING_COLORMAP


def test_the_choice_is_revisited_for_each_dataset(view):
    from src.gui.image_view_2d_enhanced import DIVERGING_COLORMAP, SEQUENTIAL_COLORMAP

    view.set_data(np.random.RandomState(0).randn(32, 32))
    assert view.combo_colormap.currentText() == DIVERGING_COLORMAP

    view.set_data(np.abs(np.random.RandomState(1).randn(32, 32)))
    assert view.combo_colormap.currentText() == SEQUENTIAL_COLORMAP


def test_a_few_negative_pixels_are_noise_not_a_second_direction(view):
    """Read-out noise below zero must not repaint an ordinary detector frame."""
    from src.gui.image_view_2d_enhanced import SEQUENTIAL_COLORMAP

    frame = np.abs(np.random.RandomState(0).randn(64, 64))
    frame.flat[:3] = -0.01

    view.set_data(frame)

    assert view.combo_colormap.currentText() == SEQUENTIAL_COLORMAP


def test_a_deliberate_choice_survives_the_next_dataset(view):
    """Once the user picks a colormap it is their decision, not ours."""
    view.set_data(np.abs(np.random.RandomState(0).randn(32, 32)))
    view.combo_colormap.setCurrentText("inferno")
    view._on_colormap_chosen(0)          # what a click emits

    view.set_data(np.random.RandomState(1).randn(32, 32))

    assert view.combo_colormap.currentText() == "inferno"


def test_all_positive_and_all_negative_are_both_sequential(view):
    from src.gui.image_view_2d_enhanced import SEQUENTIAL_COLORMAP

    view.set_data(np.full((8, 8), -5.0))

    assert view.combo_colormap.currentText() == SEQUENTIAL_COLORMAP


@pytest.mark.parametrize(
    "values, diverging",
    [
        (np.array([1.0, 2.0, 3.0]), False),
        (np.array([-1.0, -2.0]), False),
        (np.array([-1.0, 1.0]), True),
        (np.array([0.0, 0.0]), False),
        (np.array([np.nan, np.nan]), False),
        (np.array([]), False),
    ],
)
def test_the_diverging_test_itself(values, diverging):
    from src.gui.image_view_2d_enhanced import is_diverging

    assert is_diverging(values) is diverging


def test_the_rendered_image_uses_the_chosen_map(view, tmp_path):
    """The map is not just shown; it is what gets written out."""
    view.set_data(np.random.RandomState(0).randn(32, 32))
    diverging = view.render_colormapped_rgb()

    view.combo_colormap.setCurrentText("viridis")
    sequential = view.render_colormapped_rgb()

    assert not np.array_equal(diverging, sequential)


# ---------------------------------------------------------------------------
# The RGB both the export and the clipboard go through
# ---------------------------------------------------------------------------

def test_no_data_renders_nothing(view):
    assert view.render_colormapped_rgb() is None


def test_the_render_is_a_uint8_rgb_image_of_the_same_shape(view):
    view.set_data(frame())

    rgb = view.render_colormapped_rgb()

    assert rgb.shape == (8, 6, 3)
    assert rgb.dtype == np.uint8


def test_inverting_the_colormap_swaps_the_ends(view):
    view.set_data(frame())
    view.histogram.setLevels(0.0, 47.0)
    normal = view.render_colormapped_rgb()

    view.chk_invert.setChecked(True)
    inverted = view.render_colormapped_rgb()

    np.testing.assert_allclose(inverted[0, 0], normal[-1, -1], atol=2)
    np.testing.assert_allclose(inverted[-1, -1], normal[0, 0], atol=2)


def test_the_render_follows_the_histogram_levels(view):
    view.set_data(frame())

    view.histogram.setLevels(0.0, 47.0)
    full = view.render_colormapped_rgb()
    view.histogram.setLevels(40.0, 47.0)
    clipped = view.render_colormapped_rgb()

    assert not np.array_equal(full, clipped)
    # Everything below the new floor collapses onto one colour.
    assert len(np.unique(clipped[:5].reshape(-1, 3), axis=0)) == 1


def test_an_unusable_level_range_falls_back_to_the_data(view):
    """A reversed or empty range must not divide by zero."""
    view.set_data(frame())
    view.histogram.setLevels(10.0, 10.0)

    rgb = view.render_colormapped_rgb()

    assert rgb is not None and rgb.shape == (8, 6, 3)


def test_an_unknown_colormap_falls_back_instead_of_failing(view):
    view.set_data(frame())
    view.combo_colormap.addItem("not-a-colormap")
    view.combo_colormap.setCurrentText("not-a-colormap")

    assert view.render_colormapped_rgb() is not None


def test_the_render_can_be_given_data_the_viewer_is_not_showing(view):
    """The batch preview renders other frames through the same settings."""
    view.set_data(frame())

    rgb = view.render_colormapped_rgb(data=np.zeros((3, 3)))

    assert rgb.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# Writing the image out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [("out.png", "PNG"), ("out.jpg", "JPEG"), ("out.jpeg", "JPEG"),
     ("out.tif", "TIFF"), ("out.tiff", "TIFF")],
)
def test_the_extension_decides_the_file_format(view, tmp_path, name, expected):
    view.set_data(frame())
    target = tmp_path / name

    assert view.export_colormapped_image(target)
    with Image.open(target) as img:
        assert img.format == expected
        assert img.size == (6, 8)


def test_exporting_without_data_reports_failure_rather_than_raising(view, tmp_path):
    assert view.export_colormapped_image(tmp_path / "nothing.png") is False
    assert not (tmp_path / "nothing.png").exists()


def test_an_unwritable_path_is_reported_not_raised(view, tmp_path):
    view.set_data(frame())
    assert view.export_colormapped_image(tmp_path / "no_such_dir" / "x.png") is False


@pytest.mark.parametrize(
    "filter_text, expected",
    [("PNG Image (*.png)", ".png"), ("JPEG Image (*.jpg *.jpeg)", ".jpg"),
     ("TIFF Image (*.tif *.tiff)", ".tif"), ("", ".png")],
)
def test_the_dialog_filter_supplies_a_missing_extension(filter_text, expected):
    assert ImageView2DEnhanced.image_extension_from_filter(filter_text) == expected


def test_the_format_name_comes_from_the_path_first_then_the_filter():
    assert ImageView2DEnhanced.image_format_from_path("a.jpg") == "JPEG"
    assert ImageView2DEnhanced.image_format_from_path("a", "TIFF (*.tif)") == "TIFF"
    assert ImageView2DEnhanced.image_format_from_path("a") == "PNG"


def test_quick_export_names_the_file_after_the_dataset(view, tmp_path, monkeypatch):
    seen: list[str] = []

    def fake_dialog(_p, _t, default_path, *a, **k):
        seen.append(default_path)
        return str(tmp_path / "img.png"), "PNG Image (*.png)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    view.set_data(frame())
    view.set_source_dataset_key("scanx_0033.nxs::/scan/scan_data/frames")

    view.quick_export()

    assert "scanx_0033_frames" in seen[0]
    assert (tmp_path / "img.png").exists()


def test_a_cancelled_save_writes_nothing(view, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    view.set_data(frame())

    assert view.save_colormapped_image_dialog() is False
    assert list(tmp_path.iterdir()) == []


def test_saving_with_nothing_displayed_warns(view, monkeypatch):
    warned: list[int] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(1)))

    assert view.save_colormapped_image_dialog() is False
    assert warned == [1]


def test_the_written_file_carries_the_active_colormap(view, tmp_path):
    """Quick export means "as displayed", so the colormap has to be baked in."""
    view.set_data(frame())
    view.combo_colormap.setCurrentText("viridis")
    view.export_colormapped_image(tmp_path / "viridis.png")
    view.combo_colormap.setCurrentText("inferno")
    view.export_colormapped_image(tmp_path / "inferno.png")

    with Image.open(tmp_path / "viridis.png") as a, Image.open(tmp_path / "inferno.png") as b:
        assert np.asarray(a).tobytes() != np.asarray(b).tobytes()

from __future__ import annotations

import numpy as np

from src.gui.image_view_2d_enhanced import ImageView2DEnhanced


def test_auto_contrast_positive_image_uses_histogram_bounds():
    data = np.linspace(10.0, 100.0, 1000)

    levels = ImageView2DEnhanced._robust_auto_levels(data)

    assert levels is not None
    assert levels[0] == 10.0
    assert 99.0 < levels[1] < 100.0


def test_auto_contrast_diff_image_uses_symmetric_bounds():
    data = np.concatenate([np.linspace(-4.0, -1.0, 500), np.linspace(1.0, 8.0, 500)])

    levels = ImageView2DEnhanced._robust_auto_levels(data)

    assert levels is not None
    assert levels[0] < 0
    assert levels[1] > 0
    assert np.isclose(abs(levels[0]), levels[1])

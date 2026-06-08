from pathlib import Path

import numpy as np
from PIL import Image

from src.gui.main_window import load_regular_data_file
from src.lib_h5.file_validator import has_supported_extension, is_supported_data_file


def test_load_png_as_grayscale_array(tmp_path: Path):
    path = tmp_path / "img.png"
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 100
    Image.fromarray(rgb, mode="RGB").save(path)

    data = load_regular_data_file(path)

    assert data.shape == (4, 5)
    assert np.all(data > 0)
    assert has_supported_extension(path)
    assert is_supported_data_file(path)


def test_load_csv_numeric_array(tmp_path: Path):
    path = tmp_path / "curve.csv"
    np.savetxt(path, np.arange(6, dtype=float).reshape(3, 2), delimiter=",")

    data = load_regular_data_file(path)

    np.testing.assert_allclose(data, np.arange(6, dtype=float).reshape(3, 2))


def test_load_txt_numeric_array(tmp_path: Path):
    path = tmp_path / "curve.txt"
    np.savetxt(path, np.arange(4, dtype=float))

    data = load_regular_data_file(path)

    np.testing.assert_allclose(data, np.arange(4, dtype=float))

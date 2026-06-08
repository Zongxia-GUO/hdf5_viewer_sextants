from pathlib import Path

import h5py
import numpy as np

from src.gui.data_calculator_enhanced import DataCalculatorEnhanced


def _write_h5(path: Path, dataset: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=data)


def test_calculator_custom_expression_can_use_dataset_a_only(qapp, tmp_path: Path):
    h5_path = tmp_path / "scanx_0001.h5"
    data = np.arange(8, dtype=float)
    _write_h5(h5_path, "curve", data)

    tool = DataCalculatorEnhanced((h5_path,))
    tool.add_to_dataset_a(f"{h5_path}::curve")
    tool._perform_operation("A * 2")

    np.testing.assert_allclose(tool.result_data, data * 2)
    assert tool.data_b is None


def test_calculator_fft_function_uses_centered_magnitude(qapp, tmp_path: Path):
    h5_path = tmp_path / "scanx_0002.h5"
    data = np.arange(8, dtype=float)
    _write_h5(h5_path, "curve", data)

    tool = DataCalculatorEnhanced((h5_path,))
    tool.add_to_dataset_a(f"{h5_path}::curve")
    tool._perform_operation("FFT(A)")

    expected = np.abs(np.fft.fftshift(np.fft.fft(data)))
    np.testing.assert_allclose(tool.result_data, expected)

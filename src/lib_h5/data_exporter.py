"""Data export utilities for HDF5 datasets."""

# Copyright (C) 2023 Dennis Leonard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import csv
import logging
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

from src.lib_h5.dataset_types import H5DatasetType


class DataExporter:
    """Export HDF5 datasets to various file formats."""

    @staticmethod
    def export_to_csv(
        data: npt.NDArray,
        file_path: pathlib.Path,
        column_names: list[str] | None = None,
        delimiter: str = ",",
    ) -> bool:
        """
        Export data to CSV file.

        :param data: Numpy array to export
        :param file_path: Output file path
        :param column_names: Optional column names for header
        :param delimiter: CSV delimiter (default: comma)
        :return: True if successful, False otherwise
        """
        try:
            data = DataExporter._prepare_tabular_data(data)
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=delimiter)

                # Handle scalar values
                if data.ndim == 0:
                    writer.writerow([DataExporter._format_csv_value(data.item())])

                # Handle structured arrays
                elif data.dtype.names is not None:
                    # Write header with field names
                    writer.writerow(data.dtype.names)
                    # Write data rows
                    for row in data:
                        writer.writerow(row)

                # Handle 2D arrays
                elif len(data.shape) == 2:
                    # Write header if provided
                    if column_names:
                        writer.writerow(column_names)
                    # Write data
                    for row in data:
                        # Format each cell
                        formatted_row = [DataExporter._format_csv_value(val) for val in row]
                        writer.writerow(formatted_row)

                # Handle 1D arrays
                elif len(data.shape) == 1:
                    # Write header
                    if column_names and len(column_names) > 0:
                        writer.writerow([column_names[0]])
                    else:
                        writer.writerow(["Value"])
                    # Write data
                    for val in data:
                        writer.writerow([DataExporter._format_csv_value(val)])

                else:
                    logging.error(f"Cannot export {len(data.shape)}D array to CSV")
                    return False

            logging.info(f"Successfully exported data to CSV: {file_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to export CSV: {e}")
            return False

    @staticmethod
    def export_to_txt(
        data: npt.NDArray,
        file_path: pathlib.Path,
        delimiter: str = "\t",
    ) -> bool:
        """
        Export data to text file (tab-delimited).

        :param data: Numpy array to export
        :param file_path: Output file path
        :param delimiter: Delimiter (default: tab)
        :return: True if successful, False otherwise
        """
        try:
            np.savetxt(file_path, data, delimiter=delimiter, fmt="%s")
            logging.info(f"Successfully exported data to TXT: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to export TXT: {e}")
            return False

    @staticmethod
    def export_image_to_png(data: npt.NDArray, file_path: pathlib.Path) -> bool:
        """
        Export image data to PNG file.

        :param data: Image data (2D or 3D array)
        :param file_path: Output file path
        :return: True if successful, False otherwise
        """
        try:
            from PIL import Image

            data = DataExporter._prepare_image_data(data)
            # Normalize data to 0-255 range for image export
            normalized_data = DataExporter._normalize_image_data(data)

            # Convert to PIL Image
            if len(data.shape) == 2:
                # Grayscale image
                img = Image.fromarray(normalized_data.astype(np.uint8), mode="L")
            elif len(data.shape) == 3 and data.shape[2] == 3:
                # RGB image
                img = Image.fromarray(normalized_data.astype(np.uint8), mode="RGB")
            elif len(data.shape) == 3 and data.shape[2] == 4:
                # RGBA image
                img = Image.fromarray(normalized_data.astype(np.uint8), mode="RGBA")
            else:
                logging.error(f"Unsupported image shape: {data.shape}")
                return False

            img.save(file_path, "PNG")
            logging.info(f"Successfully exported image to PNG: {file_path}")
            return True

        except ImportError:
            logging.error("PIL/Pillow is required for image export. Install with: pip install Pillow")
            return False
        except Exception as e:
            logging.error(f"Failed to export PNG: {e}")
            return False

    @staticmethod
    def export_image_to_jpeg(data: npt.NDArray, file_path: pathlib.Path, quality: int = 95) -> bool:
        """
        Export image data to JPEG file.

        :param data: Image data (2D or 3D array)
        :param file_path: Output file path
        :param quality: JPEG quality (1-100)
        :return: True if successful, False otherwise
        """
        try:
            from PIL import Image

            data = DataExporter._prepare_image_data(data)
            # Normalize data
            normalized_data = DataExporter._normalize_image_data(data)

            # Convert to PIL Image
            if len(data.shape) == 2:
                # Grayscale
                img = Image.fromarray(normalized_data.astype(np.uint8), mode="L")
            elif len(data.shape) == 3 and data.shape[2] >= 3:
                # RGB (JPEG doesn't support alpha channel)
                img = Image.fromarray(normalized_data[:, :, :3].astype(np.uint8), mode="RGB")
            else:
                logging.error(f"Unsupported image shape for JPEG: {data.shape}")
                return False

            img.save(file_path, "JPEG", quality=quality)
            logging.info(f"Successfully exported image to JPEG: {file_path}")
            return True

        except ImportError:
            logging.error("PIL/Pillow is required for image export. Install with: pip install Pillow")
            return False
        except Exception as e:
            logging.error(f"Failed to export JPEG: {e}")
            return False

    @staticmethod
    def export_image_to_tiff(data: npt.NDArray, file_path: pathlib.Path) -> bool:
        """
        Export image data to TIFF file.

        :param data: Image data (2D or 3D array)
        :param file_path: Output file path
        :return: True if successful, False otherwise
        """
        try:
            from PIL import Image

            data = DataExporter._prepare_image_data(data)
            # For TIFF, we can preserve the original data type
            if len(data.shape) == 2:
                # Grayscale
                # Normalize to 16-bit for better precision
                if data.dtype in [np.float32, np.float64]:
                    finite = np.isfinite(data)
                    if np.any(finite):
                        data_min = float(np.min(data[finite]))
                        data_max = float(np.max(data[finite]))
                    else:
                        data_min = data_max = 0.0
                    if data_max > data_min:
                        scaled = np.nan_to_num(
                            (data - data_min) / (data_max - data_min),
                            nan=0.0,
                            posinf=1.0,
                            neginf=0.0,
                        )
                        normalized = (np.clip(scaled, 0.0, 1.0) * 65535).astype(np.uint16)
                    else:
                        normalized = np.zeros_like(data, dtype=np.uint16)
                    img = Image.fromarray(normalized, mode="I;16")
                else:
                    img = Image.fromarray(data)
            elif len(data.shape) == 3:
                # RGB/RGBA
                normalized_data = DataExporter._normalize_image_data(data)
                if data.shape[2] == 3:
                    img = Image.fromarray(normalized_data.astype(np.uint8), mode="RGB")
                elif data.shape[2] == 4:
                    img = Image.fromarray(normalized_data.astype(np.uint8), mode="RGBA")
                else:
                    logging.error(f"Unsupported image shape for TIFF: {data.shape}")
                    return False
            else:
                logging.error(f"Unsupported image shape for TIFF: {data.shape}")
                return False

            img.save(file_path, "TIFF")
            logging.info(f"Successfully exported image to TIFF: {file_path}")
            return True

        except ImportError:
            logging.error("PIL/Pillow is required for image export. Install with: pip install Pillow")
            return False
        except Exception as e:
            logging.error(f"Failed to export TIFF: {e}")
            return False

    @staticmethod
    def export_data(
        data: npt.NDArray,
        file_path: pathlib.Path,
        data_type: H5DatasetType,
        **kwargs: Any,
    ) -> bool:
        """
        Auto-detect and export data based on file extension and data type.

        :param data: Data to export
        :param file_path: Output file path
        :param data_type: HDF5 dataset type
        :param kwargs: Additional arguments for specific exporters
        :return: True if successful, False otherwise
        """
        extension = file_path.suffix.lower()

        # CSV/TXT exports
        if extension in [".csv", ".txt", ".tsv"]:
            delimiter = "," if extension == ".csv" else "\t"
            return DataExporter.export_to_csv(
                data,
                file_path,
                column_names=kwargs.get("column_names"),
                delimiter=delimiter,
            )

        # Image exports
        elif extension == ".png":
            return DataExporter.export_image_to_png(data, file_path)

        elif extension in [".jpg", ".jpeg"]:
            quality = kwargs.get("quality", 95)
            return DataExporter.export_image_to_jpeg(data, file_path, quality=quality)

        elif extension in [".tif", ".tiff"]:
            return DataExporter.export_image_to_tiff(data, file_path)

        else:
            logging.error(f"Unsupported file format: {extension}")
            return False

    @staticmethod
    def get_extension_from_filter(selected_filter: str) -> str:
        """
        Infer the default extension from a QFileDialog selected filter.

        :param selected_filter: The selected file filter text from QFileDialog
        :return: Extension with leading dot, or an empty string when unknown
        """
        filter_text = selected_filter.lower()
        if "*.csv" in filter_text:
            return ".csv"
        if "*.txt" in filter_text:
            return ".txt"
        if "*.tsv" in filter_text:
            return ".tsv"
        if "*.png" in filter_text:
            return ".png"
        if "*.jpg" in filter_text or "*.jpeg" in filter_text:
            return ".jpg"
        if "*.tif" in filter_text or "*.tiff" in filter_text:
            return ".tif"
        return ""

    @staticmethod
    def _prepare_image_data(data: npt.NDArray) -> npt.NDArray:
        """Normalize HDF5 image-like arrays to shapes supported by PIL."""
        if data.ndim >= 3 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        return data

    @staticmethod
    def _prepare_tabular_data(data: npt.NDArray) -> npt.NDArray:
        """Normalize HDF5 data to shapes supported by CSV/TXT export."""
        if data.dtype.names is None and data.ndim > 2:
            data = np.squeeze(data)
        return data

    @staticmethod
    def _format_csv_value(value: Any) -> str:
        """Format a value for CSV export."""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return str(value)

        if isinstance(value, str):
            return value

        if value is None:
            return ""

        try:
            if isinstance(value, (np.floating, float)):
                if np.isnan(value):
                    return "NaN"
                elif np.isinf(value):
                    return "Inf" if value > 0 else "-Inf"
                else:
                    return f"{value:.10g}"

            elif isinstance(value, (np.integer, int)):
                return str(value)

            elif isinstance(value, (np.bool_, bool)):
                return "True" if value else "False"

        except Exception:
            pass

        return str(value)

    @staticmethod
    def _normalize_image_data(data: npt.NDArray) -> npt.NDArray:
        """
        Normalize image data to 0-255 range for export.

        :param data: Image data
        :return: Normalized data
        """
        # Make a copy to avoid modifying original
        normalized = data.astype(np.float64)

        # Normalize to 0-1 range
        data_min = normalized.min()
        data_max = normalized.max()

        if data_max > data_min:
            normalized = (normalized - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(normalized)

        # Scale to 0-255
        normalized = normalized * 255

        return normalized

    @staticmethod
    def get_export_filter(data_type: H5DatasetType) -> str:
        """
        Get file filter string for export dialog based on data type.

        :param data_type: HDF5 dataset type
        :return: Filter string for file dialog
        """
        if data_type in [H5DatasetType.Table, H5DatasetType.Array1D]:
            return "CSV Files (*.csv);;Text Files (*.txt);;TSV Files (*.tsv);;All Files (*.*)"

        elif data_type in [H5DatasetType.Array2D, H5DatasetType.ImageRGB]:
            return (
                "PNG Image (*.png);;"
                "JPEG Image (*.jpg *.jpeg);;"
                "TIFF Image (*.tif *.tiff);;"
                "CSV Files (*.csv);;"
                "All Files (*.*)"
            )

        elif data_type == H5DatasetType.String:
            return "Text Files (*.txt);;All Files (*.*)"

        else:
            return "All Files (*.*)"

    @staticmethod
    def get_default_extension(data_type: H5DatasetType) -> str:
        """
        Get default file extension based on data type.

        :param data_type: HDF5 dataset type
        :return: Default extension (with dot)
        """
        if data_type in [H5DatasetType.Table, H5DatasetType.Array1D]:
            return ".csv"
        elif data_type in [H5DatasetType.Array2D, H5DatasetType.ImageRGB]:
            return ".png"
        else:
            return ".txt"




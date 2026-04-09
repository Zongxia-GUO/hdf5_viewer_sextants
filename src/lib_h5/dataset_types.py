"""Dataset type classification."""

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

from enum import Enum, auto

import numpy.typing as npt


class H5DatasetType(Enum):
    """Enum representing the type of data in a dataset."""

    Unknown = auto()
    String = auto()
    Array1D = auto()
    Array2D = auto()
    ImageRGB = auto()
    Table = auto()

    @classmethod
    def from_string(cls, plot_type: str) -> "H5DatasetType":
        """Construct type from string."""
        match plot_type:
            case "String":
                return cls.String
            case "Array1D":
                return cls.Array1D
            case "Array2D":
                return cls.Array2D
            case "Table":
                return cls.Table
            case "ImageRGB":
                return cls.Array2D  # legacy alias
            case _:
                return cls.String

    @classmethod
    def from_numpy_array(cls, array: npt.NDArray) -> "H5DatasetType":
        """
        Construct type from numpy array.

        Classification logic:
        - Structured arrays ->Table
        - 1D numeric arrays ->Array1D
        - 2D numeric arrays (< 10000 elements) ->Table
        - 2D numeric arrays (>= 10000 elements) ->Array2D (heatmap)
        - 3D numeric arrays ->ImageRGB
        - Other types ->String

        Size-1 dimensions are ignored when determining dimensionality, so
        (1, 2048, 2048) is treated the same as (2048, 2048) ->Array2D.
        """
        arr_type = str(array.dtype)
        arr_size = array.size

        # 0-d scalar (shape == ()) -display as a plain string value.
        if array.ndim == 0:
            return cls.String

        # Check if it's a structured array (has named fields)
        if array.dtype.names is not None:
            return cls.Table

        # Effective shape: drop any size-1 dimensions for type detection.
        # (1, 2048, 2048) ->(2048, 2048);  (3, 2048, 2048) ->unchanged.
        eff_shape = tuple(s for s in array.shape if s > 1) or (1,)
        eff_ndim = len(eff_shape)

        is_numeric = "int" in arr_type or "float" in arr_type

        # Single-element arrays (any shape) ->scalar, display as string
        if arr_size == 1 and is_numeric:
            return cls.String

        # 1D numeric arrays ->line plot
        if eff_ndim == 1 and is_numeric:
            return cls.Array1D

        # 2D numeric arrays ->multi-curve plot, table, or heatmap
        if eff_ndim == 2 and is_numeric:
            rows, cols = eff_shape

            # Special case: few columns (2-5) ->treat as multi-curve 1D plot
            if cols <= 5 and rows > 1:
                return cls.Array1D

            # Small arrays ->table; large arrays ->heatmap image
            if arr_size < 10000:
                return cls.Table
            else:
                return cls.Array2D

        # 3D+ arrays ->image with slice navigation (slice slider shown automatically)
        if eff_ndim >= 3 and is_numeric:
            return cls.Array2D

        # Fallback: display as string
        return cls.String




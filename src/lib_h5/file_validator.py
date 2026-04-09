"""HDF5 File validation and detection utilities."""

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

import pathlib
from typing import Union

import h5py


# Common HDF5 file extensions
HDF5_EXTENSIONS = {
    ".h5",       # Standard HDF5
    ".hdf5",     # Standard HDF5
    ".hdf",      # Older HDF format
    ".he5",      # HDF-EOS
    ".nxs",      # NeXus format (common in scientific instruments)
    ".nx5",      # NeXus HDF5
    ".cxi",      # Coherent X-ray Imaging
    ".mat",      # MATLAB v7.3+ uses HDF5
}


def is_hdf5_file(file_path: Union[str, pathlib.Path]) -> bool:
    """
    Check if a file is a valid HDF5 file by attempting to open it.

    This function tries to open the file with h5py to verify it's a valid HDF5 file,
    regardless of its extension.

    :param file_path: Path to the file to check
    :return: True if the file is a valid HDF5 file, False otherwise
    """
    try:
        with h5py.File(file_path, "r"):
            return True
    except (OSError, ValueError, IOError):
        return False


def has_hdf5_extension(file_path: Union[str, pathlib.Path]) -> bool:
    """
    Check if a file has a known HDF5 extension.

    :param file_path: Path to the file to check
    :return: True if the file has a known HDF5 extension, False otherwise
    """
    path = pathlib.Path(file_path)
    return path.suffix.lower() in HDF5_EXTENSIONS


def get_file_filter_string() -> str:
    """
    Get the file filter string for file dialogs.

    :return: Filter string in Qt file dialog format
    """
    extensions = " ".join(f"*{ext}" for ext in sorted(HDF5_EXTENSIONS))
    return f"HDF5 Files ({extensions});;All Files (*.*)"




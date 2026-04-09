"""Build executable with pyinstaller."""

# Copyright (C) 2023 Dennis Lönard
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

import sys

import PyInstaller.__main__


def build() -> None:
    """Build executable with pyinstaller."""
    build_args = [
        "main.py",
        "--name=HDF5-Viewer",
        "--noconfirm",
        "--windowed",
        "--clean",

        # Data files
        "--add-data=src/img/*:img",
        "--add-data=LICENSE:.",
        "--add-data=README.md:.",

        # Hidden imports for h5py
        "--hidden-import=h5py",
        "--hidden-import=h5py.defs",
        "--hidden-import=h5py.utils",
        "--hidden-import=h5py._proxy",

        # Hidden imports for PyQt6
        "--hidden-import=PyQt6",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        "--hidden-import=PyQt6.QtWidgets",

        # Hidden imports for pyqtgraph
        "--hidden-import=pyqtgraph",
        "--hidden-import=pyqtgraph.graphicsItems",
        "--hidden-import=pyqtgraph.imageview",

        # Other dependencies
        "--hidden-import=numpy",
        "--hidden-import=natsort",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        # scipy: required by FTH/HERALDO tool
        "--hidden-import=scipy",
        "--hidden-import=scipy.fftpack",
        "--hidden-import=scipy.ndimage",
        "--hidden-import=scipy.special",
        "--hidden-import=scipy.optimize",
        # Local shared widgets/helpers
        "--hidden-import=src.gui.dataset_path_combo",

        # Collect all package data
        "--collect-all=h5py",
        "--collect-all=pyqtgraph",
        "--collect-all=scipy",
    ]

    if sys.platform == "win32":
        build_args.append("--icon=src/img/sextants.ico")
        # On Windows, use onefile by default
        build_args.append("--onefile")
    else:
        build_args.append("--onefile")

    print("="*60)
    print("Building HDF5 Viewer executable...")
    print("="*60)
    print("\nBuild arguments:")
    for arg in build_args:
        print(f"  {arg}")
    print("\n" + "="*60 + "\n")

    PyInstaller.__main__.run(build_args)

    print("\n" + "="*60)
    print("Build completed!")
    print(f"Executable location: dist/HDF5-Viewer{'.exe' if sys.platform == 'win32' else ''}")
    print("="*60)


if __name__ == "__main__":
    build()

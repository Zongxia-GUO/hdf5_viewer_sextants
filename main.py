"""HDF5 File Viewer entry point."""

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

import logging.config
import os
import sys

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.logging_config import logging_config

# Performance: configure pyqtgraph for speed
# Note: useOpenGL is NOT enabled globally because it breaks the
# HistogramLUTItem rendering in PyInstaller-packaged builds.
# The other optimizations (row-major, no antialias, float32) still provide
# significant speedup without GPU dependency.
pg.setConfigOptions(
    antialias=False,          # Disable anti-aliasing for speed
    imageAxisOrder='row-major',  # Avoid unnecessary transposes
)

if sys.platform == "win32":
    # Set Windows Taskbar Icon
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("hdf5viewer")


def configure_light_color_scheme() -> None:
    """Ask Qt's Windows platform plugin to stay light without changing widget style."""
    if sys.platform != "win32":
        return

    platform = os.environ.get("QT_QPA_PLATFORM", "")
    platform_lower = platform.lower()
    if not platform:
        os.environ["QT_QPA_PLATFORM"] = "windows:darkmode=0"
    elif platform_lower.startswith("windows") and "darkmode=" not in platform_lower:
        os.environ["QT_QPA_PLATFORM"] = f"{platform}:darkmode=0"


def apply_light_color_scheme(app: QApplication) -> None:
    """Request the light color scheme when the current Qt version supports it."""
    set_color_scheme = getattr(app.styleHints(), "setColorScheme", None)
    if set_color_scheme is not None:
        set_color_scheme(Qt.ColorScheme.Light)


def main() -> None:
    """HDF5 File Viewer entry point."""
    logging.config.dictConfig(logging_config)
    logging.info("Starting GUI...")

    configure_light_color_scheme()
    app = QApplication(sys.argv)
    apply_light_color_scheme(app)
    app.setOrganizationName("HDF5Viewer")
    app.setApplicationName("HDF5ViewerPython")
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

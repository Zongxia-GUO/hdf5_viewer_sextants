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
import pathlib
import sys

import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.img.img_path import img_path
from src.logging_config import logging_config

# Windows taskbar identity. It MUST match the AppUserModelID on the shortcuts
# the installer creates (see windows/compile.iss): declaring an ID the shell
# cannot resolve to a shortcut is what left the first-ever launch showing a
# placeholder icon, correct only from the second run once the shell had scraped
# the window icon and cached it. Microsoft's form is Company.Product.
APP_USER_MODEL_ID = "Soleil.SEXTANTS.HDF5Viewer"

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
    # Group the taskbar button under our own identity rather than under the
    # host executable. Must happen before the first window exists.
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:   # pragma: no cover - a missing shell32 must not stop the app
        logging.warning("Could not set the Windows application id", exc_info=True)


def application_icon() -> QIcon:
    """The application icon, with its bitmaps already read from disk.

    ``QIcon(path)`` is lazy: nothing is read until some pixmap is asked for. On
    a cold start that read lands at the moment the shell wants the taskbar
    button, competing with the ~2,900 files the frozen app is already paging
    in. Forcing the sizes now moves that read off the critical moment.
    """
    icon_file = pathlib.Path(img_path(), "sextants.ico")
    if not icon_file.exists():
        logging.warning("Application icon not found: %s", icon_file)
        return QIcon()

    icon = QIcon(str(icon_file))
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.pixmap(QSize(size, size))
    return icon


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
    # Application-wide, so dialogs and the taskbar have an icon even before the
    # main window exists. Only the window carried one before.
    app.setWindowIcon(application_icon())
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Image path utils."""

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

import logging
import pathlib
import sys


def img_path() -> pathlib.Path:
    """Get path to img directory.

    Three layouts, because PyInstaller has two and the source tree is a third:

    * ``--onedir``  — the data sits beside the exe in ``_internal/img``
    * ``--onefile`` — the exe unpacks to a temp directory named by
      ``sys._MEIPASS``, and the data is at ``img`` directly under it; there is
      no ``_internal`` there at all
    * running from source — this file's own directory

    Only the first was handled, so every icon in a one-file build resolved to a
    path that does not exist. The candidates are tried in order and the first
    that is really there wins, which also survives PyInstaller changing its
    layout again.
    """
    if getattr(sys, "frozen", False):
        roots = []
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            roots.append(pathlib.Path(meipass))
        roots.append(pathlib.Path(sys.executable).parent)

        candidates = [pathlib.Path(root, *parts)
                      for root in roots
                      for parts in (("_internal", "img"), ("img",))]
        path = next((c for c in candidates if c.is_dir()), candidates[0])
    else:
        path = pathlib.Path(__file__).absolute().parent

    logging.info(f"Image path '{path}'")

    return path

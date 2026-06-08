"""Shared GUI helpers used by multiple reconstruction tools.

Centralizes colormap resolution/application and light-palette styling that was
previously duplicated (and had begun to diverge) between
``cdi_reconstruction_tool`` and ``fth_reconstruction_tool``.
"""

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

import logging
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QComboBox, QWidget

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def get_colormap(name: str) -> Optional[pg.ColorMap]:
    """Resolve colormap name across pyqtgraph/colorcet/matplotlib sources."""
    aliases = {
        "Jet": "jet",
        "Gray": "gray",
        "Hot": "hot",
        "bluered": "bwr",
    }
    query = aliases.get(name, name)

    # Try native lookup first (handles CET-* and other built-ins)
    try:
        return pg.colormap.get(query, skipCache=True)
    except Exception as exc:
        log.debug("Native colormap lookup failed for '%s': %s", query, exc)

    # Fallback to matplotlib maps (jet/gray/hot/bwr, etc.)
    try:
        return pg.colormap.get(query, source="matplotlib", skipCache=True)
    except Exception as exc:
        log.warning("Unknown colormap '%s'; keeping previous colormap (%s).", name, exc)
        return None


def _invert_colormap(cmap: pg.ColorMap) -> pg.ColorMap:
    """Return a reversed copy of a colormap."""
    lut = cmap.getLookupTable(0.0, 1.0, 256)
    lut = lut[::-1]
    return pg.ColorMap(np.linspace(0.0, 1.0, lut.shape[0]), lut)


def apply_colormap(img_item: pg.ImageItem, name: str, levels=None, invert: bool = False) -> None:
    """Apply a named colormap to a pg.ImageItem."""
    cmap = get_colormap(name)
    if cmap is not None:
        if invert:
            cmap = _invert_colormap(cmap)
        img_item.setColorMap(cmap)
    if levels is not None:
        img_item.setLevels(levels)


def apply_hist_colormap(hist: pg.HistogramLUTItem, name: str, invert: bool = False) -> None:
    """Apply a named colormap to a HistogramLUTItem gradient.

    Always routes through the histogram so that dragging the handles
    never overrides the chosen colormap.
    """
    cmap = get_colormap(name)
    if cmap is None:
        return
    if invert:
        cmap = _invert_colormap(cmap)
    hist.gradient.setColorMap(cmap)


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def set_widget_light_palette(widget: QWidget) -> None:
    """Apply a light (white base) palette to any input widget without CSS."""
    pal = widget.palette()
    for grp in (QPalette.ColorGroup.Normal, QPalette.ColorGroup.Inactive):
        pal.setColor(grp, QPalette.ColorRole.Base,   QColor("#ffffff"))
        pal.setColor(grp, QPalette.ColorRole.Text,   QColor("#111111"))
        pal.setColor(grp, QPalette.ColorRole.Button, QColor("#f0f0f0"))
        pal.setColor(grp, QPalette.ColorRole.Window, QColor("#f0f0f0"))
    widget.setPalette(pal)


def set_combo_light_palette(combo: QComboBox) -> None:
    """Apply a light palette to a combo box."""
    set_widget_light_palette(combo)

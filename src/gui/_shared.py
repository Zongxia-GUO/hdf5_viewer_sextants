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
import pathlib
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QColor, QIcon, QImage, QPalette
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from src.img.img_path import img_path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quick-action buttons
# ---------------------------------------------------------------------------

# Viewer-toolbar spacing, in one place so every gap is deliberate.
#
# Three widths, tightest first: a label sits closest to the control it names, a
# control sits a little further from its neighbours in the same tool, and one
# tool is separated from the next by the widest gap. Without this the label,
# the neighbour and the next tool were all the same distance away, so nothing
# read as a group and single-widget tools looked stranded.
TOOLBAR_LABEL_GAP = 3
TOOLBAR_ITEM_GAP = 5
TOOLBAR_GROUP_GAP = 14


def toolbar_group_gap(layout: QHBoxLayout) -> None:
    """Insert the gap that separates one tool from the next.

    Measured off :data:`TOOLBAR_ITEM_GAP` because the layout also puts its own
    spacing on either side of the spacer — so every boundary comes out at
    exactly :data:`TOOLBAR_GROUP_GAP`, whatever each group contains.
    """
    layout.addSpacing(TOOLBAR_GROUP_GAP - 2 * TOOLBAR_ITEM_GAP)


def labelled(text: str, widget: QWidget) -> QHBoxLayout:
    """A label bound to the control it names, closer than the gap between tools."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(TOOLBAR_LABEL_GAP)
    row.addWidget(QLabel(text))
    row.addWidget(widget)
    return row


def quick_icon_button(icon_name: str, tooltip: str) -> QPushButton:
    """Build a compact icon button for a quick export/copy action.

    Quick actions are icons everywhere in the app (viewer toolbars, tool panels,
    tree context menu); full exports are text buttons labelled ``Export``. This
    keeps that one visual rule in a single place.
    """
    button = QPushButton()
    button.setIcon(QIcon(str(pathlib.Path(img_path(), icon_name))))
    button.setIconSize(QSize(16, 16))
    button.setFixedSize(28, 24)
    button.setAutoDefault(False)
    button.setToolTip(tooltip)
    return button


# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------

# The two axes a radial/azimuthal profile can be plotted against, spelled once.
# The scattering tool said "Radius (pixels)" and "Angle (°)" where the Q tool
# said "r (px)" and "θ (deg)", so the same profile was named two ways depending
# on which window it was opened from.
AXIS_RADIUS_PX = "r (px)"
AXIS_ANGLE_DEG = "θ (deg)"

# The scattering vector, spelled one way. The comparison window wrote "q (A^-1)"
# and the 1-D viewer "q (1/A)" for the same conversion.
AXIS_Q = "q (1/A)"


def set_axis_label(plot_widget: object, axis: str, text: str, unit: str = "") -> None:
    """Label one axis, with its unit written out and the tick values left alone.

    pyqtgraph offers two ways to say the same thing and they read differently.
    Passing ``units="1/A"`` lets it prepend an SI prefix, so a q axis comes out
    as ``qx (m1/A)``; putting the unit in the text instead leaves the prefix
    machinery on with nothing to attach to, and it appends ``(x0.001)``. The
    2-D image, the line profile and the ring profile each picked one, so the
    same quantity was written three ways in one window.

    Here the unit goes in the text and the automatic prefix is switched off, so
    the label says ``qx (1/A)`` and the ticks read ``0.0005``, ``0.0010`` —
    the numbers as they are, which is what the axis is being read for.
    """
    item = getattr(plot_widget, "plotItem", plot_widget)
    try:
        axis_item = item.getAxis(axis)
        axis_item.enableAutoSIPrefix(False)
        # Switching the feature off does not undo a prefix already chosen.
        # pyqtgraph recomputes autoSIPrefixScale on the way out and then keeps
        # it, so an axis that had once shown "qx (m1/A)" loses the "m" from its
        # label and goes on multiplying its ticks by 1000 — the label then reads
        # 1/A over ticks of 18 where the value is 0.018. Clear the prefix, the
        # scale and the stale unit by hand, before the label is written.
        axis_item.labelUnits = ""
        axis_item.labelUnitPrefix = ""
        axis_item.autoSIPrefixScale = 1.0
    except Exception as exc:                       # pragma: no cover - defensive
        log.debug("Could not configure the %s axis: %s", axis, exc)
        return
    label = f"{text} ({unit})" if unit else text
    item.setLabel(axis, label)
    # The ticks are cached in a QPicture; without this the old numbers stay up
    # until something else happens to invalidate them.
    axis_item.picture = None
    axis_item.update()


# ---------------------------------------------------------------------------
# ROI overlay colours
# ---------------------------------------------------------------------------

# One palette for every region drawn over an image, whichever tool drew it.
# The scattering tool already used these; the line and rectangle in the 2-D
# viewer were red, so two tools marked up the same picture in two colours.
ROI_SELECTED_RGB = (0, 255, 255)
ROI_IDLE_RGB = (90, 160, 200)


def roi_pen(selected: bool = True, width: float | None = None):
    """The pen for a region drawn over an image."""
    return pg.mkPen(
        ROI_SELECTED_RGB if selected else ROI_IDLE_RGB,
        width=width if width is not None else (2.0 if selected else 1.5),
    )


# ---------------------------------------------------------------------------
# Profile curves
# ---------------------------------------------------------------------------

# The curve a 1-D profile is drawn with, whichever ROI produced it. The line
# and rectangle profiles carried a marker on every sample, which on a profile
# hundreds of points long merges into a band twice the width of the line and
# hides the shape it was drawn to show; the ring and sector profiles were a
# plain line. This is the plain line.
PROFILE_CURVE_RGB = (0, 200, 255)


def profile_pen(width: float = 2.0):
    """The pen every 1-D profile curve is drawn with."""
    return pg.mkPen(PROFILE_CURVE_RGB, width=width)


# ---------------------------------------------------------------------------
# Clearing a tool between uses
# ---------------------------------------------------------------------------

def clear_tool_displays(owner: object) -> int:
    """Blank every image and curve the tool holds. Returns how many it blanked.

    Tool windows are kept alive between uses so that reopening is instant, so
    each one needs to put itself back to a blank state when it closes. Listing
    the views by hand goes stale silently: a view added later keeps showing the
    previous scan, and nothing says so. Walking the attributes instead means a
    new view is covered the day it is added.

    Only the *contents* are cleared, never the items themselves —
    ``PlotWidget.clear()`` would remove curves the tool keeps a handle on and
    reuses, leaving it drawing into an item no longer in the scene.
    """
    cleared = 0
    for value in list(vars(owner).values()):
        try:
            if isinstance(value, pg.ImageItem):
                value.clear()
            elif isinstance(value, pg.PlotDataItem):
                value.setData([], [])
            else:
                continue
        except Exception as exc:                   # pragma: no cover - defensive
            log.debug("Could not clear a display item: %s", exc)
            continue
        cleared += 1
    return cleared


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
# Saving an image: the picture formats and the data one
# ---------------------------------------------------------------------------

# One entry per format, so the save dialog offers a real choice and fills the
# extension in. A single combined "Image Files (*.png *.jpg ...)" filter looks
# equivalent but leaves the user typing the extension, and Qt saves by
# extension — a typo there fails silently.
IMAGE_SAVE_FILTERS = (
    "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image, raw values (*.tif *.tiff)"
)

# TIFF is the data format here; the picture formats carry the colormap.
RAW_TIFF_SUFFIXES = (".tif", ".tiff")


def extension_for_filter(selected_filter: str, fallback: str = ".png") -> str:
    """The extension a chosen save-dialog filter implies."""
    text = (selected_filter or "").lower()
    if "*.jpg" in text or "*.jpeg" in text:
        return ".jpg"
    if "*.tif" in text or "*.tiff" in text:
        return ".tif"
    if "*.png" in text:
        return ".png"
    return fallback


def colormap_lut(name: str, invert: bool = False) -> Optional[np.ndarray]:
    """A 256x3 uint8 lookup table for a named colormap, or None if unknown."""
    cmap = get_colormap(name)
    if cmap is None:
        return None
    lut = np.asarray(cmap.getLookupTable(0.0, 1.0, 256))
    if lut.dtype.kind == "f":
        lut = np.clip(lut, 0.0, 255.0)
    lut = lut.astype(np.uint8)
    if lut.ndim != 2 or lut.shape[1] < 3:
        log.warning("Colormap '%s' produced an unusable lookup table.", name)
        return None
    if invert:
        lut = lut[::-1]
    return lut[:, :3]


def array_to_qimage(
    arr: np.ndarray,
    levels: tuple[float, float],
    colormap: str = "gray",
    invert: bool = False,
) -> QImage:
    """Render an array to a QImage through a colormap.

    Saving or copying a reconstruction used to produce a grey picture whatever
    was on screen, which quietly threw away the one setting that makes a phase
    map readable. Grey stays the honest fallback for an unknown colormap.
    """
    lo, hi = float(levels[0]), float(levels[1])
    if not np.isfinite(lo):
        lo = float(np.nanmin(arr))
    if not np.isfinite(hi):
        hi = float(np.nanmax(arr))
    if hi <= lo:
        hi = lo + 1e-12

    values = np.nan_to_num(arr.astype(np.float32), nan=lo, posinf=hi, neginf=lo)
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    indices = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)

    lut = colormap_lut(colormap, invert)
    if lut is None:
        height, width = indices.shape
        grey = np.ascontiguousarray(indices)
        return QImage(
            grey.data, width, height, grey.strides[0], QImage.Format.Format_Grayscale8
        ).copy()

    rgb = np.ascontiguousarray(lut[indices])
    height, width = rgb.shape[:2]
    return QImage(
        rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
    ).copy()


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

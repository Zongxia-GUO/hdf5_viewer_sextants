"""The right-click menu on a pyqtgraph plot.

pyqtgraph builds that menu from three places and merges them when the button is
pressed. Each needs different treatment, which is the whole of this module:

``ViewBox.menu``
    View All, X axis, Y axis — **kept**: auto-range and the axis controls are
    real capabilities nothing here reimplements. **Mouse Mode goes.** All it
    offers is a rubber-band zoom, and zooming already works three other ways
    (wheel, right-drag, and View All to undo); the setting is not remembered,
    so a new plot is back to the default and it would have to be chosen again
    every time; and "3 button / 1 button" names the mouse it was designed for
    rather than anything it does.

``PlotItem.ctrlMenu``
    Transforms, Downsample, Average, Alpha, Grid, Points. **Removed.** Two of
    them shadow the application's own controls and go out of step with them:
    using Transforms → Log X puts the axis into log while the toolbar checkbox
    still reads linear. Two more, Average and Downsample, change the curve that
    is drawn without changing the data that gets exported, which would break the
    rule that a figure and the file behind it agree.

``GraphicsScene.contextMenu``
    pyqtgraph's own ``Export...``. **Replaced**, because it writes files through
    its own dialog and so bypasses every convention the application has for
    them — the naming, the TXT/CSV/CSV2 dialects, the remembered folder.

Nothing here knows how to export anything. Each window passes in what it
already does for its own Export and Plot buttons, so there is one behaviour per
window rather than two that can drift apart.
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
from typing import Any, Callable

from PyQt6.QtGui import QAction

log = logging.getLogger(__name__)

# Named as the buttons elsewhere in the application are, so the menu and the
# toolbar read as the same two actions rather than two similar ones.
EXPORT_TEXT = "Export"
PLOT_TEXT = "Plot"

# The one pyqtgraph view control that is dropped along with its ctrl menu.
MOUSE_MODE_TEXT = "Mouse Mode"


def _drop_mouse_mode(view_box: Any) -> None:
    """Take the Mouse Mode submenu off a ViewBox menu.

    Matched by text because pyqtgraph keeps no handle to that submenu; missing
    it is not worth failing over, so a version that no longer has it simply
    leaves the menu as it is.
    """
    menu = getattr(view_box, "menu", None)
    if menu is None:
        return
    for action in list(menu.actions()):
        if action.text().replace("&", "") == MOUSE_MODE_TEXT:
            menu.removeAction(action)


def attach_plot_menu(
    plot_widget: Any,
    *,
    on_export: Callable[[], None] | None = None,
    on_plot: Callable[[], None] | None = None,
) -> bool:
    """Put the application's Export and Plot on a pyqtgraph plot's right-click.

    :param plot_widget: a ``pg.PlotWidget`` (or anything with ``plotItem``).
    :param on_export: called for "Export"; omitted if None.
    :param on_plot: called for "Plot"; omitted if None.
    :return: whether the menu could be attached.

    Safe to call twice on the same widget: the scene's menu is replaced, not
    appended to, so a window can hand its plot different actions later without
    collecting duplicates.
    """
    item = getattr(plot_widget, "plotItem", None)
    if item is None:
        log.debug("No plotItem to attach a context menu to: %r", plot_widget)
        return False

    try:
        # The view controls stay; the Transforms/Average block goes.
        item.setMenuEnabled(False, True)
        _drop_mouse_mode(item.vb)

        actions: list[QAction] = []
        if on_export is not None:
            export_action = QAction(EXPORT_TEXT, plot_widget)
            export_action.triggered.connect(lambda _checked=False: on_export())
            actions.append(export_action)
        if on_plot is not None:
            plot_action = QAction(PLOT_TEXT, plot_widget)
            plot_action.triggered.connect(lambda _checked=False: on_plot())
            actions.append(plot_action)

        # Parented to the widget so the actions live exactly as long as it does;
        # the scene holds only borrowed references.
        item.scene().contextMenu = actions
    except Exception as exc:
        log.warning("Could not attach the plot context menu: %s", exc)
        return False
    return True

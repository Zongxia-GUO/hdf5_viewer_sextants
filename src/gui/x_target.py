"""Windows that can be handed an X dataset from the tree's ``Set X``.

Several windows have an X field you can drag a dataset into — the export
dialogs and the Plot window. ``Set X`` should reach whichever of them the user
was last looking at, so the choice does not have to be dragged across the
screen into a window that is already open and in front.

"Last looking at" cannot be read off Qt: there is no portable stacking order,
and by the time the tree's context menu is used the main window is the active
one. So activation is *recorded* as it happens — each registered window is
watched, and the one activated most recently wins.

A window joins in by implementing ``set_x_dataset(key) -> bool`` and calling
:func:`register_x_target` once it is built. Registering is the window saying
"I can take an X right now"; a dialog that has no usable X field (an image
export, say) simply does not register.

A window whose ability to take an X comes and goes — the calculator has nothing
to put an X against until something has been calculated — also implements
``can_take_x() -> bool``. Being in front is a claim on ``Set X``, and a window
that would only refuse must not make that claim: it would swallow the choice
that the viewer behind it could have used.
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
from typing import Any, Optional

import h5py
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QEvent, QObject

from src.lib_h5.file_validator import is_hdf5_file

log = logging.getLogger(__name__)


def read_x_dataset(key: str) -> npt.NDArray[np.float64]:
    """Read a ``file::dataset`` key as the 1-D array an X axis needs.

    Shared by every X target so a dataset is accepted or refused for the same
    reasons wherever it is dropped. Raises ``ValueError`` with a message meant
    for a message box — a bad drag must not blow up the window it landed on.
    """
    text = (key or "").strip()
    if not text:
        raise ValueError("No dataset given.")
    if "::" not in text:
        raise ValueError("Drag a dataset from the tree, or type file::path.")

    file_part, ds_path = text.split("::", 1)
    path = pathlib.Path(file_part)
    if not path.exists():
        raise ValueError(f"No such file:\n{file_part}")

    if is_hdf5_file(path):
        with h5py.File(path, "r") as file:
            ds_path = ds_path.strip("/")
            if ds_path not in file:
                raise ValueError(f"No such dataset:\n{ds_path}")
            obj = file[ds_path]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError("That is a group, not a dataset.")
            values = np.asarray(obj[()]).squeeze()
    else:
        from src.gui.main_window import load_regular_data_file
        values = np.asarray(load_regular_data_file(path)).squeeze()

    if values.ndim != 1:
        raise ValueError(f"An X axis has to be 1-D; this one is {values.ndim}-D.")
    return np.asarray(values, dtype=float)


def _can_take_x(widget: Any) -> bool:
    """Whether ``widget`` would actually accept an X right now.

    Most windows always can, and say nothing. The ones that answer are the ones
    whose curve comes and goes: a calculator with no result would refuse the
    dataset *and* keep the viewer behind it from getting it, so being in front
    would make ``Set X`` do nothing at all.
    """
    ask = getattr(widget, "can_take_x", None)
    if ask is None:
        return True
    try:
        return bool(ask())
    except Exception as exc:                      # pragma: no cover - defensive
        log.warning("An X target could not say whether it is ready: %s", exc)
        return False


class XTargetWatcher(QObject):
    """Remembers which registered window was activated most recently."""

    def __init__(self) -> None:
        super().__init__()
        self._seq = 0
        # id(widget) -> (activation order, widget)
        self._targets: dict[int, tuple[int, Any]] = {}

    def register(self, widget: Any) -> None:
        """Start watching ``widget``; a fresh window counts as the newest."""
        if not hasattr(widget, "set_x_dataset"):
            raise TypeError("An X target must implement set_x_dataset(key).")

        widget.installEventFilter(self)
        self._touch(widget)
        key = id(widget)
        widget.destroyed.connect(lambda *_: self._targets.pop(key, None))

    def _touch(self, widget: Any) -> None:
        self._seq += 1
        self._targets[id(widget)] = (self._seq, widget)

    def eventFilter(self, obj: Optional[QObject], event: Optional[QEvent]) -> bool:
        if event is not None and event.type() == QEvent.Type.WindowActivate:
            if obj is not None and id(obj) in self._targets:
                self._touch(obj)
        return False

    def most_recent(self) -> Any:
        """The newest still-usable target, or None when there is none."""
        best: Any = None
        best_seq = -1
        for key, (seq, widget) in list(self._targets.items()):
            try:
                alive = widget.isVisible()
            except RuntimeError:
                # The C++ object went away without destroyed() reaching us.
                self._targets.pop(key, None)
                continue
            if alive and seq > best_seq and _can_take_x(widget):
                best, best_seq = widget, seq
        return best

    def clear(self) -> None:
        """Forget every target — for tests, so one cannot leak into the next."""
        self._targets.clear()


_WATCHER = XTargetWatcher()


def register_x_target(widget: Any) -> None:
    """Declare that ``widget`` can accept an X dataset while it is open."""
    _WATCHER.register(widget)


def active_x_target() -> Any:
    """The open X-capable window the user was last in, or None."""
    return _WATCHER.most_recent()


# The X the user last chose, per tool. Windows opened afterwards start from it
# instead of making the same choice again.
#
# Kept per tool rather than as one value, because one value leaked: an X chosen
# while the calculator was in front turned up as the default in a batch export
# started from the main window, which is a different question about different
# files. Inheriting inside a tool is the useful half — the calculator's export
# and plot should start from the X set in its own viewer — and that is what a
# scope keeps while the leak is closed.
DEFAULT_X_SCOPE = "main"
_REMEMBERED_X: dict[str, str] = {}


def x_scope_of(widget: Any) -> str:
    """Which tool a window belongs to, for the purpose of remembering an X.

    Read off the window itself or any of its parents, so a dialog opened from a
    tool inherits that tool's scope without having to be told. Anything not
    inside a tool — the main window and its own dialogs — is :data:`DEFAULT_X_SCOPE`.
    """
    node = widget
    for _ in range(20):        # a parent chain, not a search: bound it
        if node is None:
            break
        scope = getattr(node, "x_scope", None)
        if isinstance(scope, str) and scope:
            return scope
        parent = getattr(node, "parent", None)
        node = parent() if callable(parent) else None
    return DEFAULT_X_SCOPE


def remember_x_dataset(key: str, scope: str = DEFAULT_X_SCOPE) -> None:
    """Record the X the user chose, for windows of the same tool to start from."""
    _REMEMBERED_X[scope or DEFAULT_X_SCOPE] = (key or "").strip()


def remembered_x_dataset(scope: str = DEFAULT_X_SCOPE) -> str:
    """The last X chosen in ``scope`` as a ``file::dataset`` key, or ``""``."""
    return _REMEMBERED_X.get(scope or DEFAULT_X_SCOPE, "")


def clear_x_targets() -> None:
    """Drop all registrations and every remembered X (tests)."""
    _WATCHER.clear()
    _REMEMBERED_X.clear()

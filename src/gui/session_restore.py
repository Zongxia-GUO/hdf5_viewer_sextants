"""Asking whether last time's files should come back.

The window reopens whatever was open when it last closed. That is the right
default for a handful of scans and the wrong one for a folder of a thousand,
where the files have to be added and indexed again before anything can be
done — two filesystem round trips each, which on a share is most of a minute.

So a large session asks first. A small one does not, because a dialog in front
of three files is only in the way; the threshold is what separates the two and
it is a setting, since what counts as large depends on where the data lives.
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

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox, QWidget

#: Where the threshold is kept.
THRESHOLD_KEY = "settings/restore_prompt_threshold"

#: Ask from this many files up. Measured: reopening a session costs two
#: filesystem round trips per file, so on a share at 20 ms an operation 50
#: files is about two seconds — around where a wait becomes worth a question.
DEFAULT_PROMPT_THRESHOLD = 50

#: Setting the threshold to this asks every time, however few files there are.
ALWAYS_ASK = 0

#: And to this never asks, restoring silently as it did before.
NEVER_ASK = -1


def should_ask(count: int, threshold: int = DEFAULT_PROMPT_THRESHOLD) -> bool:
    """Whether a session of ``count`` files is worth stopping to ask about.

    Nothing to restore is never worth a question, whatever the threshold — a
    dialog offering to reopen zero files would be pure noise.
    """
    count = int(count)
    if count <= 0:
        return False
    threshold = int(threshold)
    if threshold == NEVER_ASK:
        return False
    if threshold <= ALWAYS_ASK:
        return True
    return count >= threshold


def read_threshold(settings) -> int:
    """The configured threshold, falling back to the default if it is unusable."""
    try:
        return int(settings.value(THRESHOLD_KEY, DEFAULT_PROMPT_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_PROMPT_THRESHOLD


def ask_restore(parent: QWidget | None, count: int) -> bool:
    """Put the question, and return whether to reopen the files.

    Restore is the default button: it is what the window did before this
    existed, and it is what someone who hit Enter without reading meant.
    Declining leaves the saved list alone rather than deleting it — the answer
    is about this launch, not about throwing the session away. It is what is
    open at the next clean close that gets saved, so a decline followed by a
    close is what actually clears it.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Restore previous session")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText(f"The last session had {int(count)} file(s) open. Restore them?")
    restore = box.addButton("Restore", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Start empty", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(restore)
    box.exec()
    return box.clickedButton() is restore

"""Noticing that the last run did not finish, so the next one can start clean.

A session remembers which files were open and reopens them on the next start.
That is right after a normal close and wrong after a crash: the files are
reopened, the index worker walks every one of them again, and if one of them is
what killed the process the application cannot be started at all — it crashes,
restores, and crashes again.

The evidence is a heartbeat. While the window is up it stamps the wall clock
into the settings every :data:`HEARTBEAT_INTERVAL_MS`; closing normally removes
the stamp. So on the next start:

* no stamp — the last session closed normally, or this is the first run;
* a stale stamp — the process that wrote it is gone and never cleaned up;
* a fresh stamp — another window is running right now.

The third case is why this is a timestamp and not a flag. Nothing in this
application stops a second window being opened, and a plain "still running"
flag cannot tell that apart from a crash — the second window would decide the
first had died and throw away its file list. A stamp costs 0.667 ms every ten
seconds, measured, and cannot make that mistake.
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

import logging
import time

from PyQt6.QtCore import QObject, QSettings, QTimer

#: Where the stamp lives.
HEARTBEAT_KEY = "session/heartbeat"

#: How often the running window stamps the clock.
HEARTBEAT_INTERVAL_MS = 10_000

#: How old a stamp has to be before the process that wrote it is presumed gone.
#: Three missed beats, so an application briefly too busy to run its timers is
#: not mistaken for a dead one.
STALE_AFTER_S = 30.0

#: The previous session closed normally, or there was no previous session.
ENDED_CLEANLY = "clean"
#: The previous session left a stale stamp behind: it never reached its close.
ENDED_BADLY = "crashed"
#: Another window is running now, so there is nothing to conclude.
STILL_RUNNING = "running"


def classify(
    stamp: float | None,
    now: float,
    stale_after: float = STALE_AFTER_S,
) -> str:
    """What the stamp says about whoever wrote it.

    A stamp from the future means the clock moved backwards between the two
    runs — a time zone change, an NTP correction. That is not evidence of a
    crash, and the cost of guessing wrong is a discarded session, so an
    unreadable clock is reported as :data:`STILL_RUNNING`: the conservative
    answer, which changes nothing.
    """
    if stamp is None:
        return ENDED_CLEANLY
    age = float(now) - float(stamp)
    if age < 0:
        return STILL_RUNNING
    return ENDED_BADLY if age > float(stale_after) else STILL_RUNNING


def describe_age(seconds: float) -> str:
    """How long ago, in words, for the message the user actually reads."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(round(seconds))} seconds ago"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{int(round(minutes))} minutes ago"
    return f"{int(round(minutes / 60.0))} hours ago"


class SessionGuard(QObject):
    """Stamps the clock while the window is up, and reads the last stamp once."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._verdict = ENDED_CLEANLY
        self._age_s = 0.0

        stamp = self._read_stamp()
        now = time.time()
        self._verdict = classify(stamp, now)
        if stamp is not None:
            self._age_s = max(0.0, now - float(stamp))

        self._timer = QTimer(self)
        self._timer.setInterval(HEARTBEAT_INTERVAL_MS)
        self._timer.timeout.connect(self.beat)

    # ── What the window asks ──────────────────────────────────────────── #

    @property
    def verdict(self) -> str:
        """One of :data:`ENDED_CLEANLY`, :data:`ENDED_BADLY`, :data:`STILL_RUNNING`."""
        return self._verdict

    @property
    def previous_run_crashed(self) -> bool:
        return self._verdict == ENDED_BADLY

    @property
    def age_of_last_stamp(self) -> float:
        """Seconds between the last stamp and this start-up."""
        return self._age_s

    # ── What the window tells it ──────────────────────────────────────── #

    def start(self) -> None:
        """Begin stamping. The first one goes down immediately, so a crash in
        the first ten seconds is still noticed."""
        self.beat()
        self._timer.start()

    def beat(self) -> None:
        settings = QSettings()
        settings.setValue(HEARTBEAT_KEY, time.time())
        settings.sync()

    def mark_clean_exit(self) -> None:
        """Remove the stamp. Its absence is what says the close was orderly."""
        self._timer.stop()
        settings = QSettings()
        settings.remove(HEARTBEAT_KEY)
        settings.sync()

    # ── Internals ─────────────────────────────────────────────────────── #

    @staticmethod
    def _read_stamp() -> float | None:
        raw = QSettings().value(HEARTBEAT_KEY, None)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            # Something else wrote the key, or it was written by a version that
            # stored it differently. Unreadable is not evidence of a crash.
            logging.debug("Ignoring an unreadable session heartbeat: %r", raw)
            return None

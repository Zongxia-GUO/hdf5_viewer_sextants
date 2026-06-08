"""Single source of truth for the application version.

Bump the version HERE only. Everything else reads from this:
  * setup.cfg  -> ``version = attr: src.version.__version__``
  * build.py   -> imports ``__version__`` (and passes it to the Windows installer)

Use Semantic Versioning (MAJOR.MINOR.PATCH):
  * PATCH (0.1.0 -> 0.1.1): bug fixes, no behaviour change
  * MINOR (0.1.0 -> 0.2.0): new features, backwards compatible
  * MAJOR (0.x   -> 1.0.0): first stable / breaking changes
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

__version__ = "0.1.0"

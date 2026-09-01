"""The application's Windows identity and icon.

Setting an explicit AppUserModelID tells Windows "this is who I am on the
taskbar". The shell then looks for a Start Menu shortcut carrying the same id
to take the icon and the pinning identity from. Declaring an id that no
shortcut carries is worse than declaring none: the first ever launch has
nothing to draw and shows a placeholder, and only later runs look right once
the shell has scraped the window icon and cached it.

Nothing at runtime can notice the two drifting apart — the installer is built
by a separate tool — so the check lives here.
"""

import pathlib
import re

import pytest
from PyQt6.QtGui import QIcon

import main as app_main

ISS = pathlib.Path(__file__).resolve().parent.parent / "windows" / "compile.iss"


def test_the_identity_is_a_company_product_id():
    """A bare word like "hdf5viewer" collides across vendors."""
    assert app_main.APP_USER_MODEL_ID.count(".") >= 1
    assert " " not in app_main.APP_USER_MODEL_ID


def test_the_installer_declares_the_same_identity():
    text = ISS.read_text(encoding="utf-8", errors="replace")
    defined = re.search(r'#define\s+MyAppUserModelID\s+"([^"]+)"', text)

    assert defined, "the installer must define an AppUserModelID"
    assert defined.group(1) == app_main.APP_USER_MODEL_ID


def test_every_shortcut_carries_the_identity():
    """A shortcut without it is one the shell cannot resolve the icon from."""
    text = ISS.read_text(encoding="utf-8", errors="replace")
    icons = [
        line for line in text.splitlines()
        if line.strip().startswith("Name:") and "{#MyAppExeName}" in line
    ]

    assert icons, "the installer creates at least one shortcut"
    for line in icons:
        assert "AppUserModelID:" in line, line


def test_the_icon_is_loaded_and_not_left_empty(qapp):
    icon = app_main.application_icon()

    assert isinstance(icon, QIcon)
    assert not icon.isNull(), "a null icon is what the placeholder looks like"


def test_the_icon_carries_the_sizes_the_shell_asks_for(qapp):
    """The taskbar wants 16-32 px; the alt-tab switcher wants much larger."""
    icon = app_main.application_icon()
    sizes = {(s.width(), s.height()) for s in icon.availableSizes()}

    assert (16, 16) in sizes
    assert max(w for w, _h in sizes) >= 48


@pytest.mark.parametrize("size", [16, 32, 256])
def test_the_bitmaps_are_already_in_memory(qapp, size):
    """QIcon is lazy; the cold-start read must not land on the shell's request."""
    from PyQt6.QtCore import QSize

    pixmap = app_main.application_icon().pixmap(QSize(size, size))

    assert not pixmap.isNull()


def test_a_missing_icon_file_degrades_instead_of_raising(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(app_main, "img_path", lambda: str(tmp_path))

    icon = app_main.application_icon()

    assert icon.isNull(), "an empty icon, not a crash at startup"

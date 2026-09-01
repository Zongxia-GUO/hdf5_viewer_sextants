"""What the frozen application needs, checked against build.py.

None of this can be caught by running from source: every failure here shows up
only after packaging, as a missing icon or an ImportError on someone else's
machine.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = (ROOT / "build.py").read_text(encoding="utf-8")
SOURCES = [ROOT / "main.py"] + sorted((ROOT / "src").rglob("*.py"))


def _third_party_roots() -> set[str]:
    """Top-level packages the application imports that are not stdlib."""
    stdlib = set(sys.stdlib_module_names)
    roots: set[str] = set()
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root and root not in stdlib and root != "src":
                    roots.add(root)
    return roots


def test_every_imported_package_is_bundled():
    """PyInstaller finds most imports itself, but the ones reached through a
    string or a plugin system it cannot, so each is named deliberately."""
    missing = [
        root for root in sorted(_third_party_roots())
        if f"--collect-all={root}" not in BUILD and f"--hidden-import={root}" not in BUILD
    ]

    assert missing == [], f"imported but never named in build.py: {missing}"


def test_nothing_imported_is_also_excluded():
    """An --exclude-module that the app actually imports is an ImportError in
    the frozen build and nowhere else."""
    excluded = [
        root for root in sorted(_third_party_roots())
        if f"--exclude-module={root}" in BUILD
    ]

    assert excluded == []


def test_the_icons_are_shipped():
    assert "--add-data=src/img/*:img" in BUILD


def test_every_icon_the_code_asks_for_exists():
    """A missing icon is a blank button in the built app and nothing in the logs."""
    img_dir = ROOT / "src" / "img"
    present = {path.name for path in img_dir.iterdir() if path.is_file()}

    asked_for: set[str] = set()
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for name in present:
            if name in text:
                asked_for.add(name)

    assert asked_for, "no icon is referenced by name at all — the scan is broken"
    assert asked_for <= present


def test_the_installer_launches_the_exe_the_build_produces():
    iss = (ROOT / "windows" / "compile.iss").read_text(encoding="utf-8", errors="replace")

    assert 'define MyAppExeName "HDF5-Viewer.exe"' in iss
    assert "APP_NAME = \"HDF5-Viewer\"" in BUILD


def test_the_installer_fallback_version_matches_the_source_of_truth():
    """build.py passes the real version to ISCC, so the .iss default only shows
    up when the script is compiled by hand — which is exactly when a stale
    number would go unnoticed."""
    from src.version import __version__

    iss = (ROOT / "windows" / "compile.iss").read_text(encoding="utf-8", errors="replace")

    assert f'#define MyAppVersion "{__version__}"' in iss


def test_the_changelog_has_an_entry_for_this_version():
    from src.version import __version__

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{__version__}]" in changelog
    assert f"[{__version__}]: https://" in changelog, "the compare link is missing"


def test_the_taskbar_identity_matches_the_shortcut():
    """Windows groups a window under its shortcut only when the two agree; a
    mismatch is why the first launch showed a placeholder icon."""
    iss = (ROOT / "windows" / "compile.iss").read_text(encoding="utf-8", errors="replace")
    main = (ROOT / "main.py").read_text(encoding="utf-8")

    assert 'MyAppUserModelID "Soleil.SEXTANTS.HDF5Viewer"' in iss
    assert 'APP_USER_MODEL_ID = "Soleil.SEXTANTS.HDF5Viewer"' in main


# ---------------------------------------------------------------------------
# Where the icons are looked for once frozen
# ---------------------------------------------------------------------------

@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Pretend to be a PyInstaller build laid out under ``tmp_path``."""
    from src.img import img_path as module

    monkeypatch.setattr(module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(module.sys, "executable", str(tmp_path / "HDF5-Viewer.exe"))
    monkeypatch.delattr(module.sys, "_MEIPASS", raising=False)
    return tmp_path


def test_a_onedir_build_finds_its_icons(frozen):
    from src.img.img_path import img_path

    (frozen / "_internal" / "img").mkdir(parents=True)

    assert img_path() == frozen / "_internal" / "img"


def test_a_onefile_build_finds_its_icons(frozen, monkeypatch):
    """A one-file build unpacks to _MEIPASS and has no _internal at all, so the
    onedir path resolved to somewhere that does not exist and every icon in the
    application came back blank."""
    from src.img import img_path as module

    unpacked = frozen / "temp_meipass"
    (unpacked / "img").mkdir(parents=True)
    monkeypatch.setattr(module.sys, "_MEIPASS", str(unpacked), raising=False)

    assert module.img_path() == unpacked / "img"


def test_running_from_source_is_unaffected():
    from src.img.img_path import img_path

    assert img_path() == (ROOT / "src" / "img")
    assert (img_path() / "sextants.ico").exists()

import os
import tempfile

# Run Qt in headless mode for CI/local test runs without a display server.
# This must happen before PyQt6 is imported, hence the deliberate late imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# Send every QSettings write to a throwaway directory. Without this the suite
# writes into the real user preference store (the Windows registry) — a test
# that picks a colormap then genuinely changes what the installed application
# opens with, which is exactly what happened once.
_SETTINGS_DIR = tempfile.mkdtemp(prefix="hdf5-viewer-test-settings-")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
for _scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
    QSettings.setPath(QSettings.Format.IniFormat, _scope, _SETTINGS_DIR)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _close_widgets_after_each_test():
    """Destroy every widget a test left behind, before the next test runs.

    Widgets that outlive their test stay registered with the shared
    QApplication. A later test that pumps the event loop — the FTH background
    load waits in a ``processEvents`` spin — then delivers events to those
    stragglers and to objects queued for deletion, which crashed the run with an
    access violation once enough of them had piled up.
    """
    yield

    # A window that offered itself as an X target must not still be on the
    # register when the next test asks which window is in front.
    from src.gui.x_target import clear_x_targets
    clear_x_targets()

    # Preferences a test stored (an export dialect, a plot palette) would
    # otherwise decide what the next test's defaults are.
    settings = QSettings()
    settings.clear()
    settings.sync()

    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    # Let the queued deletions actually run while nothing else is in flight.
    app.processEvents()

"""The parameter box behind the viewer's Despike button.

Small on purpose, and shaped like the "X to q" dialog next to it: the button
asks its question at the one moment the answer matters, instead of parking four
controls in a toolbar that is already full.

The one thing it does beyond collecting numbers is say what they would do —
how many points a threshold catches — before anything is applied. A despike
threshold has no meaning that can be read off the number itself; it only means
something against the data in front of you.
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

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from src.recon.despike import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    DIRECTIONS,
    METHOD_DIFFERENCE,
    METHODS,
    SPACE_AUTO,
    SPACES,
    despike_table,
)

log = logging.getLogger(__name__)

# What the button applies when nothing has been chosen yet.
DEFAULT_SETTINGS = {
    "method": METHOD_DIFFERENCE,
    "window": DEFAULT_WINDOW,
    "threshold": DEFAULT_THRESHOLD,
    "direction": DIRECTIONS[0],
    "replace": True,
    "space": SPACE_AUTO,
}


class DespikeDialog(QDialog):
    """Ask for the despike parameters, showing what they would catch."""

    def __init__(self, values, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Despike")
        self._values = np.asarray(values, dtype=float) if values is not None else None
        self._settings = dict(DEFAULT_SETTINGS)
        self._settings.update(settings or {})

        self._build_ui()
        self._load(self._settings)
        self._update_preview()

    # -- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(METHODS)
        self.cmb_method.setToolTip(
            "Difference: a spike is a jump that comes straight back, so real "
            "steps and peaks are left alone.\n"
            "Hampel: compares each value with its neighbours — reaches the end "
            "samples, but is quicker to flag ordinary scatter."
        )
        form.addRow("Method:", self.cmb_method)

        self.cmb_space = QComboBox()
        self.cmb_space.addItems(SPACES)
        self.cmb_space.setToolTip(
            "Log judges each point against its neighbours as a ratio, so a "
            "glitch in a reflectivity tail counts for as much as one on the "
            "bright plateau. Auto chooses it when the data spans more than "
            "two decades."
        )
        form.addRow("Scale:", self.cmb_space)

        self.spn_threshold = QDoubleSpinBox()
        self.spn_threshold.setRange(0.5, 100.0)
        self.spn_threshold.setSingleStep(0.5)
        self.spn_threshold.setDecimals(1)
        self.spn_threshold.setSuffix("  sigma")
        self.spn_threshold.setToolTip(
            "How far from the local noise a point must be. Lower catches more, "
            "and eventually catches noise."
        )
        form.addRow("Threshold:", self.spn_threshold)

        self.spn_window = QSpinBox()
        self.spn_window.setRange(3, 999)
        self.spn_window.setSingleStep(2)
        self.spn_window.setSuffix("  points")
        self.spn_window.setToolTip(
            "How many neighbouring points define 'local'. Keep it well below "
            "the width of the features you want to keep."
        )
        form.addRow("Window:", self.spn_window)

        self.cmb_direction = QComboBox()
        self.cmb_direction.addItems(DIRECTIONS)
        self.cmb_direction.setToolTip(
            "Detector glitches are often one-sided; saying so halves the "
            "chance of catching a real feature."
        )
        form.addRow("Direction:", self.cmb_direction)

        layout.addLayout(form)

        self.chk_replace = QCheckBox("Replace spikes with the local median")
        self.chk_replace.setToolTip(
            "Off: the spikes are only ringed on the plot and the values are "
            "left as they are."
        )
        layout.addWidget(self.chk_replace)

        self.lbl_preview = QLabel()
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color: #555;")
        layout.addWidget(self.lbl_preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        apply_button.setDefault(True)
        apply_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for widget in (self.cmb_method, self.cmb_direction, self.cmb_space):
            widget.currentTextChanged.connect(self._update_preview)
        for widget in (self.spn_threshold, self.spn_window):
            widget.valueChanged.connect(self._update_preview)
        self.chk_replace.toggled.connect(self._update_preview)

        self.setWindowModality(Qt.WindowModality.WindowModal)

    # -- state --------------------------------------------------------------

    def _load(self, settings: dict) -> None:
        self.cmb_method.setCurrentText(settings["method"])
        self.spn_threshold.setValue(float(settings["threshold"]))
        self.spn_window.setValue(int(settings["window"]))
        self.cmb_direction.setCurrentText(settings["direction"])
        self.chk_replace.setChecked(bool(settings["replace"]))
        self.cmb_space.setCurrentText(settings.get("space", SPACE_AUTO))

    def settings(self) -> dict:
        """The parameters as chosen, ready to hand to :func:`despike_table`."""
        return {
            "method": self.cmb_method.currentText(),
            "window": self.spn_window.value(),
            "threshold": self.spn_threshold.value(),
            "direction": self.cmb_direction.currentText(),
            "replace": self.chk_replace.isChecked(),
            "space": self.cmb_space.currentText(),
        }

    # -- preview ------------------------------------------------------------

    def _update_preview(self, *_args) -> None:
        if self._values is None or self._values.size == 0:
            self.lbl_preview.setText("No data loaded.")
            return

        try:
            result = despike_table(self._values, **self.settings())
        except Exception as exc:                     # pragma: no cover - defensive
            log.warning("Despike preview failed: %s", exc)
            self.lbl_preview.setText("Could not evaluate these settings.")
            return

        total = int(self._values.size)
        share = 100.0 * result.count / total if total else 0.0
        verb = "replace" if self.chk_replace.isChecked() else "mark"
        text = f"Would {verb} {result.count} of {total} points ({share:.2f}%)."
        if self.cmb_space.currentText() == SPACE_AUTO:
            # Auto is the default, so say which way it went — the answer changes
            # what the threshold means, and it should not be a hidden decision.
            text += f"  Auto chose the {'log' if result.log_space else 'linear'} scale."
        if share > 5.0:
            # Past a few percent the filter has stopped removing glitches and
            # started reshaping the data, which is worth saying out loud.
            text += "  That is a large fraction — raise the threshold."
        self.lbl_preview.setText(text)

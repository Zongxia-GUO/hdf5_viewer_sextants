"""Categorical palettes for multi-curve plots.

These are for *identity* — telling one curve from another — so they are
qualitative sets with a fixed slot order, never a continuous colormap sampled at
N points (adjacent samples of a continuous ramp are the classic unreadable
multi-line chart).

The hex values are matplotlib's own qualitative colormaps, read from the
installed matplotlib and frozen here so the module stays importable and testable
without it. Okabe-Ito is not a matplotlib colormap and is included by name.

``cvd_delta_e`` on each entry is measured, not estimated: the worst OKLab ΔE
(x100) between *adjacent* slots under simulated protanopia/deutanopia, against a
white figure surface. Higher is easier to tell apart; ~8 is the usual target and
below ~6 two neighbouring curves can look identical to a colour-blind reader.
The numbers are surfaced in the picker's tooltip so the choice is informed —
several popular palettes, the default included, score below the target.

Slot order is part of the measurement: the check runs on adjacent pairs, so
reordering a palette changes its score. Re-measure after any edit.
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

from dataclasses import dataclass

# Worst adjacent-pair CVD deltas below this read as "these two curves may be
# indistinguishable to a colour-blind reader" in the picker's tooltip.
CVD_TARGET = 8.0
CVD_FLOOR = 6.0


@dataclass(frozen=True)
class Palette:
    """A fixed-order set of categorical colours."""

    key: str
    label: str
    colors: tuple[str, ...]
    cvd_delta_e: float

    def __len__(self) -> int:
        return len(self.colors)

    @property
    def cvd_verdict(self) -> str:
        """One-word reading of :attr:`cvd_delta_e`."""
        if self.cvd_delta_e >= CVD_TARGET:
            return "good"
        if self.cvd_delta_e >= CVD_FLOOR:
            return "marginal"
        return "poor"

    def tooltip(self) -> str:
        """Picker tooltip: how well the colours separate, in plain terms."""
        reading = {
            "good": "adjacent curves stay distinct for colour-blind readers",
            "marginal": "adjacent curves are close; line styles help",
            "poor": "adjacent curves can look identical to colour-blind readers",
        }[self.cvd_verdict]
        return f"{len(self)} colours · colour-blind separation {self.cvd_delta_e:.1f} — {reading}"


PALETTES: tuple[Palette, ...] = (
    Palette(
        key="Set1",
        label="Set1 (9)",
        colors=(
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#ffff33", "#a65628", "#f781bf",
            "#999999",
        ),
        cvd_delta_e=5.9,
    ),
    Palette(
        key="tab10",
        label="tab10 (10)",
        colors=(
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ),
        cvd_delta_e=0.7,
    ),
    Palette(
        key="tab20",
        label="tab20 (20)",
        colors=(
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a", "#d62728", "#ff9896",
            "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
            "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
        ),
        cvd_delta_e=7.4,
    ),
    Palette(
        key="tab20b",
        label="tab20b (20)",
        colors=(
            "#393b79", "#5254a3", "#6b6ecf", "#9c9ede",
            "#637939", "#8ca252", "#b5cf6b", "#cedb9c",
            "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
            "#843c39", "#ad494a", "#d6616b", "#e7969c",
            "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
        ),
        cvd_delta_e=6.2,
    ),
    Palette(
        key="tab20c",
        label="tab20c (20)",
        colors=(
            "#3182bd", "#6baed6", "#9ecae1", "#c6dbef",
            "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2",
            "#31a354", "#74c476", "#a1d99b", "#c7e9c0",
            "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb",
            "#636363", "#969696", "#bdbdbd", "#d9d9d9",
        ),
        cvd_delta_e=5.8,
    ),
    Palette(
        key="okabe_ito",
        label="Okabe-Ito (8)",
        colors=(
            "#000000", "#e69f00", "#56b4e9", "#009e73",
            "#f0e442", "#0072b2", "#d55e00", "#cc79a7",
        ),
        cvd_delta_e=15.8,
    ),
    Palette(
        key="Paired",
        label="Paired (12)",
        colors=(
            "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
            "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
            "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
        ),
        cvd_delta_e=11.5,
    ),
    Palette(
        key="Accent",
        label="Accent (8)",
        colors=(
            "#7fc97f", "#beaed4", "#fdc086", "#ffff99",
            "#386cb0", "#f0027f", "#bf5b17", "#666666",
        ),
        cvd_delta_e=8.1,
    ),
    Palette(
        key="Dark2",
        label="Dark2 (8)",
        colors=(
            "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
            "#66a61e", "#e6ab02", "#a6761d", "#666666",
        ),
        cvd_delta_e=6.4,
    ),
    Palette(
        key="Set2",
        label="Set2 (8)",
        colors=(
            "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
            "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
        ),
        cvd_delta_e=1.5,
    ),
    Palette(
        key="Set3",
        label="Set3 (12)",
        colors=(
            "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
            "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
            "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
        ),
        cvd_delta_e=1.5,
    ),
    Palette(
        key="Pastel1",
        label="Pastel1 (9)",
        colors=(
            "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4",
            "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec",
            "#f2f2f2",
        ),
        cvd_delta_e=3.9,
    ),
    Palette(
        key="Pastel2",
        label="Pastel2 (8)",
        colors=(
            "#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4",
            "#e6f5c9", "#fff2ae", "#f1e2cc", "#cccccc",
        ),
        cvd_delta_e=1.2,
    ),
)

DEFAULT_PALETTE_KEY = "Set1"

# The secondary channel once the colours run out, and useful on its own for
# grayscale print.
LINE_STYLES: tuple[str, ...] = ("-", "--", "-.", ":")

_BY_KEY = {p.key: p for p in PALETTES}
_BY_LABEL = {p.label: p for p in PALETTES}


def get_palette(key: str | None) -> Palette:
    """Resolve a palette by key, falling back to the default."""
    return _BY_KEY.get(key or "", _BY_KEY[DEFAULT_PALETTE_KEY])


def palette_labels() -> list[str]:
    """Labels for a combo box, in declaration order (default first)."""
    return [p.label for p in PALETTES]


def palette_from_label(label: str | None) -> Palette:
    """Resolve a palette by the label shown in the UI."""
    return _BY_LABEL.get(label or "", _BY_KEY[DEFAULT_PALETTE_KEY])


def style_for(index: int, palette: Palette) -> tuple[str, str]:
    """Colour and line style for series ``index``.

    Colours are taken in fixed slot order. Past the end of the palette the hues
    repeat but the line style advances, so an extra curve is separated by a
    second channel instead of an invented colour.
    """
    color = palette.colors[index % len(palette)]
    style = LINE_STYLES[(index // len(palette)) % len(LINE_STYLES)]
    return color, style


def styles_for(count: int, palette: Palette) -> list[tuple[str, str]]:
    """Colour/line-style pairs for ``count`` series."""
    return [style_for(i, palette) for i in range(max(0, count))]

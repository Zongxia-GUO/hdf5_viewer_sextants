"""Tests for the FTH/CDI "Save & Export" box.

Target picks the component; Copy and Export both follow it. "Full" means all
four: Copy yields one 2x2 composite (the clipboard holds a single image),
Export writes one file per component.
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QPushButton

from src.gui.cdi_reconstruction_tool import CDIReconstructionTool
from src.gui.fth_reconstruction_tool import FTHReconstructionTool


@pytest.fixture
def fth(qapp):
    return FTHReconstructionTool(opened_files=())


@pytest.fixture
def cdi(qapp):
    return CDIReconstructionTool(opened_files=())


def _components(h=30, w=40):
    return {
        name: np.random.RandomState(seed).rand(h, w).astype(np.float32)
        for seed, name in enumerate(("real", "imag", "phase", "abs"))
    }


# ---------------------------------------------------------------------------
# The box itself
# ---------------------------------------------------------------------------

def test_fth_box_has_two_text_buttons(fth):
    assert fth._btn_copy_component.text() == "Copy"
    assert fth._btn_export_component.text() == "Export"
    for btn in (fth._btn_copy_component, fth._btn_export_component):
        assert btn.icon().isNull(), "these are text buttons, not icons"


def test_cdi_box_has_two_text_buttons(cdi):
    assert cdi._cdi_copy_btn.text() == "Copy"
    assert cdi._cdi_export_btn.text() == "Export"
    for btn in (cdi._cdi_copy_btn, cdi._cdi_export_btn):
        assert btn.icon().isNull(), "these are text buttons, not icons"


def test_the_old_third_button_is_gone(fth, cdi):
    assert not hasattr(fth, "_btn_save_all_components")
    assert not hasattr(cdi, "_cdi_save_all_btn")


@pytest.mark.parametrize("attr", ["_exp_target_combo"])
def test_fth_target_offers_full(fth, attr):
    combo = getattr(fth, attr)
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["Real", "Imag.", "Phase", "Abs.", "Full"]


def test_cdi_target_offers_full(cdi):
    combo = cdi._cdi_exp_target_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["Real", "Imag.", "Phase", "Abs.", "Full"]


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, expected",
    [("Real", "real"), ("Imag.", "imag"), ("Phase", "phase"), ("Abs.", "abs"), ("Full", "full")],
)
def test_fth_target_resolves(fth, label, expected):
    fth._exp_target_combo.setCurrentText(label)
    assert fth._selected_export_component_name() == expected


@pytest.mark.parametrize(
    "label, expected",
    [("Real", "real"), ("Imag.", "imag"), ("Phase", "phase"), ("Abs.", "abs"), ("Full", "full")],
)
def test_cdi_target_resolves(cdi, label, expected):
    cdi._cdi_exp_target_combo.setCurrentText(label)
    assert cdi._selected_cdi_export_component_name() == expected


# ---------------------------------------------------------------------------
# The composite Copy builds for "Full"
# ---------------------------------------------------------------------------

def test_fth_composite_is_a_2x2_tiling(fth):
    sheet = fth._composite_components_qimage(_components(h=30, w=40))

    assert sheet is not None
    # Two 40x30 tiles per side plus a 4 px gap.
    assert (sheet.width(), sheet.height()) == (84, 64)


def test_cdi_composite_is_a_2x2_tiling(cdi):
    sheet = cdi._composite_cdi_components_qimage(_components(h=30, w=40))

    assert sheet is not None
    assert (sheet.width(), sheet.height()) == (84, 64)


def test_composite_needs_every_component(fth):
    partial = _components()
    del partial["phase"]
    assert fth._composite_components_qimage(partial) is None


def test_composite_panels_are_not_identical(fth):
    """A tiling bug that drew the same tile four times would still be 2x2."""
    sheet = fth._composite_components_qimage(_components(h=30, w=40))

    top_left = sheet.copy(0, 0, 40, 30)
    bottom_right = sheet.copy(44, 34, 40, 30)
    assert top_left != bottom_right


# ---------------------------------------------------------------------------
# Export routing
# ---------------------------------------------------------------------------

def test_fth_export_routes_full_to_all_and_single_to_one(fth, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(FTHReconstructionTool, "_save_all_components", lambda self: calls.append("all"))
    monkeypatch.setattr(
        FTHReconstructionTool, "_save_selected_component", lambda self: calls.append("selected")
    )

    fth._exp_target_combo.setCurrentText("Full")
    fth._btn_export_component.click()
    fth._exp_target_combo.setCurrentText("Phase")
    fth._btn_export_component.click()

    assert calls == ["all", "selected"]


def test_cdi_export_routes_full_to_all_and_single_to_one(cdi, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(CDIReconstructionTool, "_save_all_cdi_components", lambda self: calls.append("all"))
    monkeypatch.setattr(
        CDIReconstructionTool, "_save_selected_cdi_component", lambda self: calls.append("selected")
    )

    cdi._cdi_exp_target_combo.setCurrentText("Full")
    cdi._cdi_export_btn.click()
    cdi._cdi_exp_target_combo.setCurrentText("Real")
    cdi._cdi_export_btn.click()

    assert calls == ["all", "selected"]


def test_no_stale_icon_buttons_remain_in_the_export_boxes(fth, cdi):
    """Every button in these tools carries a label now, not a bare icon."""
    for tool in (fth, cdi):
        bare_icons = [
            b for b in tool.findChildren(QPushButton)
            if not b.text().strip() and not b.icon().isNull()
        ]
        assert bare_icons == []

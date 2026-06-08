from __future__ import annotations

from src.gui.main_window import MainWindow


def test_parse_scan_range_mixed_single_numbers_and_ranges():
    parse = MainWindow._parse_scan_range

    assert parse(None, "0001,0003,0005") == ["0001", "0003", "0005"]
    assert parse(None, "0001-0005") == ["0001", "0002", "0003", "0004", "0005"]
    assert parse(None, "0001-0004,0005-0007") == [
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
    ]
    assert parse(None, "0001，0003-0004") == ["0001", "0003", "0004"]

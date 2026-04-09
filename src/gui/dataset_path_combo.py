"""Shared dataset path combo box for drag-drop and editable dataset selection."""

import logging
import pathlib
from typing import Optional

import h5py
from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QComboBox, QSizePolicy


class DatasetPathCombo(QComboBox):
    """Editable dataset selector with drag-drop replace and short display labels."""

    def __init__(self, placeholder: str = "-- none --", parent=None) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self.setAcceptDrops(True)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("Select from dropdown, or drag a dataset from the HDF5 tree")
        self.addItem(placeholder, userData=None)
        if self.lineEdit() is not None:
            self.lineEdit().installEventFilter(self)

    @staticmethod
    def _short_dataset_label(ds_path: str) -> str:
        parts = ds_path.split("/")
        return "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]

    @classmethod
    def short_display_from_full_key(cls, full_key: str) -> str:
        """Convert full key '<path>::<dataset>' to compact display text."""
        fname, ds_path = full_key.rsplit("::", 1)
        return f"{pathlib.Path(fname).name}::{cls._short_dataset_label(ds_path.strip())}"

    def _item_to_entry(self, item_data) -> Optional[tuple[str, str]]:
        """Normalize itemData (legacy tuple or full-key str) to (file, dataset)."""
        if item_data is None:
            return None
        if isinstance(item_data, tuple) and len(item_data) == 2:
            return str(item_data[0]).strip(), str(item_data[1]).strip()
        item_text = str(item_data)
        if "::" not in item_text:
            return None
        fname, ds_path = item_text.rsplit("::", 1)
        return fname.strip(), ds_path.strip()

    def add_full_key(self, full_key: str, select: bool = False) -> int:
        """Add full dataset key if missing; returns index."""
        for i in range(self.count()):
            if str(self.itemData(i)) == full_key:
                if select:
                    self.setCurrentIndex(i)
                return i
        idx = self.count()
        self.addItem(self.short_display_from_full_key(full_key), userData=full_key)
        if select:
            self.setCurrentIndex(idx)
        return idx

    def clear_datasets(self) -> None:
        """Clear combo and restore placeholder item."""
        self.blockSignals(True)
        self.clear()
        self.addItem(self._placeholder, userData=None)
        self.setCurrentIndex(0)
        self.blockSignals(False)

    def populate(self, opened_files, min_ndim: int = 1) -> None:
        """Populate combo from opened files with datasets of at least `min_ndim`."""
        saved_entry = self.get_entry(opened_files=opened_files)
        self.clear_datasets()
        self.blockSignals(True)
        for full_key in self.collect_full_keys(opened_files, min_ndim=min_ndim):
            self.add_full_key(full_key, select=False)
        self.blockSignals(False)

        if saved_entry is not None:
            saved_key = f"{saved_entry[0]}::{saved_entry[1]}"
            self.add_full_key(saved_key, select=True)

    @staticmethod
    def collect_full_keys(
        opened_files,
        min_ndim: int = 1,
        min_second_dim: int = 0,
    ) -> list[str]:
        """Collect all '<file>::<dataset>' keys from opened files once."""
        keys: list[str] = []
        for fp in opened_files:
            fp_str = str(fp)
            try:
                with h5py.File(fp_str, "r") as f:
                    def _visit(name, obj, _fp=fp_str):
                        if not isinstance(obj, h5py.Dataset):
                            return
                        shp = obj.shape
                        if len(shp) < min_ndim:
                            return
                        if min_second_dim > 0 and (len(shp) < 2 or shp[1] <= min_second_dim):
                            return
                        if True:
                            keys.append(f"{_fp}::{name}")
                    f.visititems(_visit)
            except Exception as exc:
                logging.warning("Skip unreadable dataset file '%s': %s", fp_str, exc)
        return keys

    def populate_from_full_keys(self, full_keys: list[str], opened_files=None) -> None:
        """Populate combo from pre-collected full keys (no HDF5 scan here)."""
        saved_entry = self.get_entry(opened_files=opened_files)
        self.clear_datasets()
        self.blockSignals(True)
        for full_key in full_keys:
            self.add_full_key(full_key, select=False)
        self.blockSignals(False)

        if saved_entry is not None:
            saved_key = f"{saved_entry[0]}::{saved_entry[1]}"
            self.add_full_key(saved_key, select=True)

    def _try_select_by_text(self, text: str) -> bool:
        """Try selecting an existing item from free-form text."""
        text = text.strip()
        if not text:
            return False

        # 0) Full parse: filename/path + dataset
        if "::" in text:
            file_token, ds_token = text.split("::", 1)
            file_token = file_token.strip()
            ds_token = ds_token.strip()
            for i in range(self.count()):
                entry = self._item_to_entry(self.itemData(i))
                if entry is None:
                    continue
                item_file, item_ds = entry
                if item_ds != ds_token:
                    continue
                if item_file == file_token or pathlib.Path(item_file).name == file_token:
                    self.setCurrentIndex(i)
                    return True

        # 1) Exact display text
        for i in range(self.count()):
            if text == self.itemText(i):
                self.setCurrentIndex(i)
                return True

        # 2) Loose contains
        for i in range(self.count()):
            item_txt = self.itemText(i)
            if text in item_txt or item_txt in text:
                self.setCurrentIndex(i)
                return True
        return False

    def get_entry(self, opened_files=None) -> Optional[tuple[str, str]]:
        """Return selected/typed (file_path, dataset_path), or None if invalid."""
        text = self.currentText().strip()
        current = self._item_to_entry(self.currentData())
        current_text = self.itemText(self.currentIndex()).strip() if self.currentIndex() >= 0 else ""
        current_data = str(self.currentData()).strip() if self.currentData() is not None else ""

        if current is not None and text in ("", current_text, current_data):
            return current

        # The line edit may have been manually edited while currentData still
        # points at the previous item; always try to resolve the visible text
        # before falling back to stale itemData.
        if text and self._try_select_by_text(text):
            matched = self._item_to_entry(self.currentData())
            if matched is not None:
                return matched

        if "::" not in text:
            return None
        file_token, ds_path = text.rsplit("::", 1)
        file_token = file_token.strip()
        ds_path = ds_path.strip()
        if not file_token or not ds_path:
            return None

        # Resolve by currently opened files (filename or full path).
        if opened_files is not None:
            for fp in opened_files:
                fp_str = str(fp)
                if fp_str == file_token or pathlib.Path(fp_str).name == file_token:
                    return fp_str, ds_path

        # Fallback to typed token (can be full file path not listed yet).
        return file_token, ds_path

    def eventFilter(self, obj, event):
        """Force drag-drop replacement in line edit (never append)."""
        le = self.lineEdit()
        if obj is le and event is not None:
            if event.type() == QEvent.Type.DragEnter and event.mimeData().hasText():
                event.acceptProposedAction()
                return True
            if event.type() == QEvent.Type.Drop and event.mimeData().hasText():
                text = event.mimeData().text().strip()
                if text:
                    self.setEditText(text)
                    self._try_select_by_text(text)
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasText() and "::" in event.mimeData().text():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        text = event.mimeData().text().strip()
        if "::" not in text:
            event.ignore()
            return
        self.setEditText(text)
        if self._try_select_by_text(text):
            event.acceptProposedAction()
            return
        self.add_full_key(text, select=True)
        event.acceptProposedAction()




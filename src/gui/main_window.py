"""Main Window of the GUI."""

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
import os
import pathlib
import sys
import json
import time
import re
from collections import OrderedDict, deque
from typing import Any, Generator

import h5py
import numpy as np
import pyqtgraph as pg
from natsort import natsorted
from PyQt6.QtCore import (
    QModelIndex,
    QPoint,
    QSettings,
    QSize,
    QSortFilterProxyModel,
    QStandardPaths,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from src.gui.about_page import AboutPage
from src.gui.batch_export import (
    PLOT_MODE_PER_SCAN,
    BatchExportDialog,
    BatchProgress,
    BatchTarget,
    adjust_batch_path_for_scan,
    batch_folder_conflict,
    batch_number_ambiguity,
    build_batch_targets,
    build_curve_preview_table,
    common_keyword,
    compress_scan_numbers,
    describe_number_ambiguity,
    export_curve_combined_table,
    export_curve_dataset,
    export_image_dataset,
    is_curve_export_data,
    parse_keywords,
    read_batch_x_data,
    safe_export_name,
    scan_number_in_stem,
    scan_stem_parts,
    stem_matches_keywords,
    summarise_number_ambiguity,
    table_format_from,
)
from src.gui.export_naming import (
    last_save_directory,
    remember_save_directory,
    short_series_label,
)
from src.gui.table_model import CopyableTableView, TableModel
from src.gui.x_target import (
    DEFAULT_X_SCOPE,
    active_x_target,
    remember_x_dataset,
    x_scope_of,
)
from src.img.img_path import img_path
from src.lib_h5.dataset_types import H5DatasetType
from src.lib_h5.file_size import file_size_to_str
from src.lib_h5.file_validator import (
    get_file_filter_string,
    has_supported_extension,
    is_hdf5_file,
    is_supported_data_file,
)

FTH_MIN_SECOND_DIM = 100  # FTH candidate requires shape[1] > 100

# Separator between several Y dataset paths in the batch address field.
BATCH_PATH_SEPARATOR = "; "

# What the scans field starts on, and what a bare range falls back to.
DEFAULT_SCAN_PREFIX = "scanx_"

# Gap between the controls of the two batch rows under the tree. One value, so
# the rows cannot drift apart.
BATCH_ROW_SPACING = 6

# Wider than this and a 2D dataset is an image, not a set of curves to plot.
MAX_PLOT_COLUMNS = 16


class HDF5TreeView(QTreeView):
    """Custom TreeView that sends full dataset paths when dragging."""

    def __init__(self, parent=None):
        """Initialize the custom tree view."""
        super().__init__(parent)
        self.main_window = None  # Will be set by MainWindow

    @staticmethod
    def _path_for_index(index0):
        """Build the drag token for a column-0 index, or None if not draggable.

        Datasets → "<file>::<dataset>"; a top-level file → its bare absolute path
        (the drop target's address field then names the dataset).
        """
        node_type = index0.data(_ROLE_NODE_TYPE)
        if node_type not in ("dataset", "file"):
            return None

        # The file half comes from the stored path, the dataset half from the
        # labels of the rows in between — those are the HDF5 names and are what
        # the token needs. Reading the file half from a label too is what tied
        # the token to how the first column happened to be written.
        file_path = tree_file_path(index0)
        if not file_path:
            return None
        if node_type == "file":
            return file_path

        names = []
        node = index0
        while node.isValid() and node.data(_ROLE_NODE_TYPE) != "file":
            if node.data():
                names.append(str(node.data()))
            node = node.parent()
        if not names:
            return None
        return f"{file_path}::{'/'.join(reversed(names))}"

    def startDrag(self, supportedActions):
        """Send the selected dataset/file path(s); multi-selection drags as lines."""
        if self.main_window is None:
            return

        # Collect every selected column-0 row (de-duplicated, order preserved) so
        # selecting several files and dragging them in adds them all.
        seen = set()
        tokens = []
        for idx in self.selectedIndexes():
            if idx.column() != 0:
                continue
            index0 = idx.sibling(idx.row(), 0)
            token = self._path_for_index(index0)
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
        if not tokens:
            # Fall back to the focused row (e.g. dragging without a selection).
            cur = self.currentIndex()
            if cur.isValid():
                token = self._path_for_index(cur.sibling(cur.row(), 0))
                if token:
                    tokens.append(token)
        if not tokens:
            return

        from PyQt6.QtCore import QMimeData
        from PyQt6.QtGui import QDrag

        mime_data = QMimeData()
        mime_data.setText("\n".join(tokens))

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        # Avoid repeated Qt warnings "QPixmap::scaled: Pixmap is a null pixmap"
        # during drag by always providing a non-null drag pixmap.
        pm = QPixmap(16, 16)
        pm.fill(Qt.GlobalColor.transparent)
        drag.setPixmap(pm)
        drag.exec(supportedActions)
        logging.info("Dragging %d path(s)", len(tokens))


# h5py chunk cache for network access (128 MB keeps recently read chunks in RAM)
_H5PY_CHUNK_CACHE = 128 * 1024 * 1024
# Datasets larger than this threshold use lazy per-slice loading (saves bandwidth for 3D stacks)
_LAZY_LOAD_THRESHOLD = 50 * 1024 * 1024   # 50 MB
# Number of recently loaded datasets kept in memory to avoid re-reading on re-click
_DATASET_CACHE_SIZE = 5

# Tree lazy-loading roles
_ROLE_H5_PATH = int(Qt.ItemDataRole.UserRole) + 1
_ROLE_NODE_TYPE = int(Qt.ItemDataRole.UserRole) + 2
_ROLE_CHILDREN_LOADED = int(Qt.ItemDataRole.UserRole) + 3
# The file's absolute path, held on the file row.
#
# It used to be the row's *label*, and everything that needed a path read the
# label back — drag and drop, the context menu, the folder monitor. That made
# the first column unshortenable: the moment it showed anything friendlier than
# an absolute path, all of those broke. The label is now free to be short
# because the path lives here instead.
_ROLE_FILE_PATH = int(Qt.ItemDataRole.UserRole) + 4

# Tree columns.
TREE_COLUMN_NAME = 0
TREE_COLUMN_TYPE = 1
TREE_COLUMN_SHAPE = 2
TREE_COLUMN_FOLDER = 3
TREE_HEADERS = ["Name", "Type", "Shape", "Folder"]


def tree_file_path(index) -> str:
    """The absolute file path a tree row belongs to, or ``""``.

    Walks up to the top-level row, which is the one that carries the path, so
    it answers for a dataset deep in a file as readily as for the file itself.
    """
    node = index
    while node is not None and node.isValid():
        stored = node.sibling(node.row(), TREE_COLUMN_NAME).data(_ROLE_FILE_PATH)
        if stored:
            return str(stored)
        node = node.parent()
    return ""


def tree_blank_cell() -> QStandardItem:
    """An empty, non-editable cell.

    The Folder column only says anything on a file row. Every other row still
    gets the cell, so each row has the full set of columns rather than a ragged
    end that Qt would have to guess at.
    """
    cell = QStandardItem("")
    cell.setEditable(False)
    return cell


def tree_item_file_path(item) -> str:
    """The same, for a QStandardItem rather than an index."""
    node = item
    while node is not None:
        stored = node.data(_ROLE_FILE_PATH)
        if stored:
            return str(stored)
        node = node.parent()
    return ""


_REGULAR_DATASET_PATH = "/data"


def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA arrays to a 2D luminance image for scientific display."""
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float64, copy=False)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return arr


def load_regular_data_file(file_path: str | pathlib.Path) -> np.ndarray:
    """Load supported non-HDF image/text files as a NumPy array."""
    path = pathlib.Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"}:
        from PIL import Image, ImageSequence

        with Image.open(path) as img:
            frames = [_rgb_to_gray(np.asarray(frame)) for frame in ImageSequence.Iterator(img)]
        if not frames:
            raise ValueError(f"No image frames found in {path}")
        if len(frames) == 1:
            return np.asarray(frames[0])
        first_shape = frames[0].shape
        if any(frame.shape != first_shape for frame in frames):
            raise ValueError("Multi-page image has frames with different shapes.")
        return np.stack(frames, axis=0)

    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        try:
            data = np.genfromtxt(path, delimiter=delimiter, comments=None)
            if np.asarray(data).size > 0:
                return np.asarray(data)
        except Exception:
            if suffix == ".txt":
                try:
                    data = np.genfromtxt(path, delimiter=",", comments=None)
                    if np.asarray(data).size > 0:
                        return np.asarray(data)
                except Exception:
                    pass
        return np.genfromtxt(path, delimiter=delimiter, dtype=str, comments=None)

    raise ValueError(f"Unsupported non-HDF file type: {suffix}")


def _regular_file_kind(file_path: str | pathlib.Path) -> str:
    suffix = pathlib.Path(file_path).suffix.lower()
    if suffix in {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"}:
        return "Image File"
    if suffix in {".csv", ".txt"}:
        return "Text/CSV File"
    return "Data File"


class DataLoadWorker(QThread):
    """Background thread: loads HDF5 dataset without blocking the UI."""

    # (data, data_type_str, file_path, obj_path) - full dataset loaded
    data_ready = pyqtSignal(object, str, str, str)
    # (first_slice, shape_tuple, data_type_str, file_path, obj_path) - large 3D, lazy mode
    data_ready_lazy = pyqtSignal(object, object, str, str, str)
    load_error = pyqtSignal(str)

    def __init__(self, file_path, obj_path, plot_type=""):
        super().__init__()
        self._file_path = file_path
        self._obj_path = obj_path
        self._plot_type = plot_type
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if not is_hdf5_file(self._file_path):
                data = load_regular_data_file(self._file_path)
                if self._cancelled:
                    return
                dtype = (H5DatasetType.from_string(self._plot_type)
                         if self._plot_type and self._plot_type != "Auto"
                         else H5DatasetType.from_numpy_array(np.asarray(data)))
                type_str = dtype.name if dtype != H5DatasetType.String else "String"
                self.data_ready.emit(data, type_str, str(self._file_path), self._obj_path)
                return

            with h5py.File(self._file_path, "r", rdcc_nbytes=_H5PY_CHUNK_CACHE) as f:
                if self._cancelled:
                    return
                obj = f[self._obj_path]

                if isinstance(obj, h5py.Group):
                    data = np.array([name for name in obj])
                    if not self._cancelled:
                        self.data_ready.emit(data, "String",
                                             str(self._file_path), self._obj_path)
                    return

                if not isinstance(obj, h5py.Dataset):
                    return

                shape = obj.shape
                total_bytes = obj.size * obj.dtype.itemsize
                is_large_3d = (len(shape) >= 3 and shape[0] > 1
                               and total_bytes > _LAZY_LOAD_THRESHOLD)

                if is_large_3d:
                    # Only read the first slice; remaining slices are loaded on demand
                    first_slice = np.array(obj[0])
                    if self._cancelled:
                        return
                    dtype = (H5DatasetType.from_string(self._plot_type)
                             if self._plot_type and self._plot_type != "Auto"
                             else H5DatasetType.from_numpy_array(first_slice))
                    self.data_ready_lazy.emit(first_slice, tuple(shape), dtype.name,
                                              str(self._file_path), self._obj_path)
                else:
                    data = obj[...]
                    if self._cancelled:
                        return
                    dtype = (H5DatasetType.from_string(self._plot_type)
                             if self._plot_type and self._plot_type != "Auto"
                             else H5DatasetType.from_numpy_array(data))
                    type_str = dtype.name if dtype != H5DatasetType.String else "String"
                    self.data_ready.emit(data, type_str,
                                         str(self._file_path), self._obj_path)

        except Exception as e:
            if not self._cancelled:
                self.load_error.emit(str(e))


class _FolderScanWorker(QThread):
    """Background thread: scans a directory for new/removed HDF5 files.

    Only does lightweight disk enumeration and extension checks.
    The actual tree-model updates (_open_file / removeRow) must happen on the
    main thread and are performed by the connected slot.
    """

    scan_done = pyqtSignal(list, list)   # (new_paths_sorted, removed_paths)

    def __init__(self, folder: pathlib.Path, known: frozenset, parent=None) -> None:
        super().__init__(parent)
        self._folder = folder
        self._known  = known   # immutable snapshot - safe to read from thread

    def run(self) -> None:
        try:
            disk: set = {
                str(p)
                for p in self._folder.iterdir()
                if p.is_file() and has_supported_extension(p)
            }
        except OSError:
            disk = set()
        new     = sorted(disk - self._known)
        removed = sorted(self._known - disk)
        self.scan_done.emit(new, removed)


class _DatasetIndexWarmWorker(QThread):
    """Background worker that incrementally updates shared dataset-key indices."""

    batch = pyqtSignal(object, int, int, str, object)  # (delta_cache, processed, total, scope, fast_paths)
    done = pyqtSignal(object, str, object)  # (cache, scope, fast_paths)

    def __init__(
        self,
        opened_files: tuple[pathlib.Path, ...],
        prev_cache: dict[str, tuple[tuple[int, int], list[str], list[str]]],
        index_scope: str,
        fast_group_paths: tuple[str, ...],
        batch_size: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._opened_files = tuple(opened_files)
        self._prev_cache = dict(prev_cache or {})
        self._index_scope = str(index_scope or "fast")
        self._fast_group_paths = tuple(p for p in fast_group_paths if p)
        self._batch_size = max(1, int(batch_size))

    @staticmethod
    def _file_signature(path_str: str) -> tuple[int, int]:
        st = os.stat(path_str)
        return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))), int(st.st_size)

    @staticmethod
    def _scan_single_file_both(
        path_str: str,
        *,
        index_scope: str,
        fast_group_paths: tuple[str, ...],
    ) -> tuple[list[str], list[str]]:
        """Scan one file once and produce both 1D and FTH-2D key lists."""
        keys_1d: list[str] = []
        keys_2d_fth: list[str] = []

        if not is_hdf5_file(path_str):
            arr = np.asarray(load_regular_data_file(path_str))
            full_key = f"{path_str}::data"
            if arr.ndim >= 1:
                keys_1d.append(full_key)
            if arr.ndim >= 2 and int(arr.shape[1]) > FTH_MIN_SECOND_DIM:
                keys_2d_fth.append(full_key)
            return keys_1d, keys_2d_fth

        def _in_fast_scope(ds_name: str) -> bool:
            if index_scope != "fast":
                return True
            if not fast_group_paths:
                return True
            norm = f"/{ds_name.strip('/')}/"
            for g in fast_group_paths:
                gg = str(g).strip().strip("/")
                if not gg:
                    continue
                token = f"/{gg}/"
                if token in norm:
                    return True
            return False

        with h5py.File(path_str, "r") as f:
            def _visit(name, obj, _fp=path_str):
                if not isinstance(obj, h5py.Dataset):
                    return
                if not _in_fast_scope(name):
                    return
                shp = obj.shape
                if len(shp) >= 1:
                    full_key = f"{_fp}::{name}"
                    keys_1d.append(full_key)
                    if len(shp) >= 2 and shp[1] > FTH_MIN_SECOND_DIM:
                        keys_2d_fth.append(full_key)
            f.visititems(_visit)
        return keys_1d, keys_2d_fth

    def _update_cache(
        self,
        prev_cache: dict[str, tuple[tuple[int, int], list[str], list[str]]],
    ) -> dict[str, tuple[tuple[int, int], list[str], list[str]]]:
        next_cache: dict[str, tuple[tuple[int, int], list[str], list[str]]] = {}
        delta_cache: dict[str, tuple[tuple[int, int], list[str], list[str]]] = {}
        total = len(self._opened_files)
        processed = 0
        batch_size = self._batch_size
        for fp in self._opened_files:
            fp_str = str(fp)
            try:
                sig = self._file_signature(fp_str)
                prev = prev_cache.get(fp_str)
                if prev is not None and prev[0] == sig:
                    rec = prev
                else:
                    keys_1d, keys_2d_fth = self._scan_single_file_both(
                        fp_str,
                        index_scope=self._index_scope,
                        fast_group_paths=self._fast_group_paths,
                    )
                    rec = (sig, keys_1d, keys_2d_fth)
                next_cache[fp_str] = rec
                delta_cache[fp_str] = rec
            except Exception as exc:
                logging.warning("Skip unreadable dataset file '%s': %s", fp_str, exc)
            finally:
                processed += 1
                if delta_cache and (processed % batch_size == 0 or processed == total):
                    self.batch.emit(
                        dict(delta_cache),
                        processed,
                        total,
                        self._index_scope,
                        self._fast_group_paths,
                    )
                    delta_cache.clear()
        return next_cache

    def run(self) -> None:
        next_cache = self._update_cache(self._prev_cache)
        self.done.emit(next_cache, self._index_scope, self._fast_group_paths)


class MainWindow(QMainWindow):
    """Start Main Window of the GUI."""
    dataset_index_changed = pyqtSignal()
    _INDEX_CACHE_VERSION = 1

    def __init__(self) -> None:
        """Start Main Window of the GUI."""
        super().__init__(flags=Qt.WindowType.Window)
        self.setAcceptDrops(True)

        # Variables
        self.cur_file = pathlib.Path()
        self.cur_obj_path = ""
        self.icon_dir = img_path()
        self._missing_icons_logged: set[str] = set()

        # Network performance: background loader + LRU dataset cache
        self._load_worker: DataLoadWorker | None = None
        self._dataset_cache: OrderedDict = OrderedDict()  # (file, path) -> (data, type_str)

        # Folder monitor state
        self._monitor_folder: pathlib.Path | None = None
        self._monitor_known: set = set()          # str(fpath) already opened
        self._scan_worker: _FolderScanWorker | None = None
        self._open_queue: deque[str] = deque()
        self._open_queue_total = 0
        self._open_queue_processed = 0
        self._open_queue_mode = ""
        self._open_queue_mark_known = False
        self._open_queue_removed_count = 0
        self._open_queue_batch_size = 25
        self._open_queue_timer = QTimer(self)
        self._open_queue_timer.setSingleShot(True)
        self._open_queue_timer.timeout.connect(self._process_open_queue_batch)
        self._batch_path_template: str | None = None
        self._batch_path_hidden_prefix: str | None = None
        # A tree Export or Plot request ("export" / "plot"), consumed once the
        # dataset is actually displayed.
        self._pending_tree_action: str = ""
        # Set by the tree's "Set as X": the dataset every export, plot and viewer
        # should use as its X axis until another one is chosen.
        self._x_dataset_key: str | None = None
        # Incremental per-file index cache:
        # file_path -> ((mtime_ns,size), keys_1d, keys_2d_fth)
        self._dataset_per_file_index_cache: dict[
            str, tuple[tuple[int, int], list[str], list[str]]
        ] = {}
        self._dataset_index_last_used: dict[str, int] = {}
        self._index_scope: str = "fast"  # "fast" or "full"
        self._fast_group_paths: tuple[str, ...] = ("scan_data",)
        self._index_batch_size: int = 50
        self._index_cache_max_files: int = 5000
        self._load_index_scope_settings()
        self._load_disk_index_cache()
        self._index_warm_worker: _DatasetIndexWarmWorker | None = None
        self.dataset_index_changed.connect(self._refresh_open_tools_dataset_index)

        # Appearance
        settings = QSettings()
        # Remove minimum size restriction to allow flexible window resizing
        self.setWindowTitle("HDF5 Viewer")
        self.resize(settings.value("main_window/size", defaultValue=QSize(1400, 700)))
        self.move(settings.value("main_window/position", defaultValue=QPoint(300, 150)))
        self._ensure_on_screen()
        self.setWindowIcon(self._icon_from_name("sextants.ico"))

        # Layout Right Side
        self.table_model_dataset = TableModel(header=["Attribute", "Value"])
        self.table_view_dataset = CopyableTableView()
        # Set size policy to allow flexible resizing
        from PyQt6.QtWidgets import QSizePolicy
        self.table_view_dataset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table_view_dataset.setModel(self.table_model_dataset)
        # Configure header for flexible resizing
        attr_header = self.table_view_dataset.horizontalHeader()
        if attr_header:
            attr_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            attr_header.setStretchLastSection(False)
        self.table_view_dataset.setColumnWidth(1, 300)
        self.plot_wgt_dataset = pg.PlotWidget()
        # Set size policy to allow plot to expand and fill available space
        self.plot_wgt_dataset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Disable right-click menu for consistent UI (use menu bar for export)
        self.plot_wgt_dataset.plotItem.vb.setMenuEnabled(False)

        self.dock_table = QDockWidget()
        self.dock_table.setWindowTitle("Attributes")
        # Use Ignored horizontal policy to allow shrinking below minimum size hints
        self.dock_table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.dock_table.setWidget(self.table_view_dataset)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_table)

        self.dock_plot = QDockWidget()
        self.dock_plot.setWindowTitle("Data")
        # Use Ignored horizontal policy to allow shrinking below minimum size hints
        self.dock_plot.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.dock_plot.setWidget(self.plot_wgt_dataset)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_plot)

        # Center Layout - Use custom TreeView for drag support
        self.tree_view_file = HDF5TreeView()
        # Set size policy to allow tree view to expand
        self.tree_view_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tree_view_file.main_window = self  # Set reference for path building
        self.tree_view_file.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view_file.customContextMenuRequested.connect(self._handle_tree_menu)
        self.tree_model_file = QStandardItemModel()
        self.tree_model_file.setHorizontalHeaderLabels(TREE_HEADERS)
        self.tree_model_file_proxy = QSortFilterProxyModel()
        self.tree_model_file_proxy.setRecursiveFilteringEnabled(True)

        self.tree_model_file_proxy.setSourceModel(self.tree_model_file)
        self.tree_view_file.setModel(self.tree_model_file_proxy)
        # Configure header to allow flexible column resizing
        tree_header = self.tree_view_file.header()
        if tree_header:
            tree_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            tree_header.setStretchLastSection(False)
        # Name only has to hold a filename now, not an absolute path, so it can
        # give most of that width back — the folder column costs less than Name
        # saves. Elide from the middle: the ends of a path are the informative
        # parts, and cutting from the right removed the scan number, which is
        # the one thing a file is looked up by.
        self.tree_view_file.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree_view_file.setColumnWidth(TREE_COLUMN_NAME, 200)
        self.tree_view_file.setColumnWidth(TREE_COLUMN_TYPE, 100)   # includes dtype
        self.tree_view_file.setColumnWidth(TREE_COLUMN_SHAPE, 120)
        self.tree_view_file.setColumnWidth(TREE_COLUMN_FOLDER, 160)
        self.tree_view_file.setAcceptDrops(True)
        self.tree_view_file.setDragEnabled(True)  # Enable dragging items from tree view
        self.tree_view_file.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree_view_file.clicked.connect(self._handle_item_changed)
        self.tree_view_file.expanded.connect(self._on_tree_item_expanded)

        # Collapse all button (refresh/collapse all files)
        self.btn_collapse_all = QPushButton("")  # Refresh/collapse symbol
        self.btn_collapse_all.setIcon(self._icon_from_name("ref.ico"))
        self.btn_collapse_all.setMaximumWidth(40)
        self.btn_collapse_all.setToolTip("Collapse all files in tree view")
        self.btn_collapse_all.clicked.connect(self._collapse_all_files)

        # Batch add controls: which files, then which scans within them.
        #
        # Two fields, not one, because the two are independent dimensions in a
        # real filename: Scan_ECL_5p0uJIR_050 has the family at the front, the
        # number at the back, and something that varies in between. A single
        # field can only express the pair when they are adjacent.
        self.le_batch_keywords = QLineEdit()
        self.le_batch_keywords.setText(DEFAULT_SCAN_PREFIX)
        self.le_batch_keywords.setMaximumWidth(140)
        self.le_batch_keywords.setPlaceholderText("keywords")
        self.le_batch_keywords.setAcceptDrops(True)
        self.le_batch_keywords.dragEnterEvent = self._batch_keywords_drag_enter
        self.le_batch_keywords.dropEvent = self._batch_keywords_drop
        self.le_batch_keywords.setToolTip(
            "Words the file name must contain, e.g. Scan_ or ECL.\n"
            "Several are separated by spaces and all must match: ECL 5p0uJIR.\n"
            "Case is ignored, and a word may sit anywhere in the name.\n"
            "Drag files from the tree to fill this in."
        )
        self.le_batch_keywords.returnPressed.connect(self._handle_batch_preview)

        self.le_scan_range = QLineEdit()
        self.le_scan_range.setMaximumWidth(120)
        self.le_scan_range.setPlaceholderText("0080-0085")
        self.le_scan_range.setAcceptDrops(True)
        self.le_scan_range.dragEnterEvent = self._batch_scan_range_drag_enter
        self.le_scan_range.dropEvent = self._batch_scan_range_drop
        self.le_scan_range.setToolTip(
            "Scan numbers: a range 0080-0085, or a list 0080,0085,0027.\n"
            "A number matches a whole part of the file name, so 050 never\n"
            "picks up 1050. Zero padding is ignored: 47 finds 047.\n"
            "Drag files from the tree to fill this in.\n"
            "Press Enter to preview the first matched scan."
        )
        self.le_scan_range.returnPressed.connect(self._handle_batch_preview)

        # Create drag-drop enabled path input
        self.le_batch_path = QLineEdit()
        # The one stretchy thing in its row: it holds the longest text
        # (scan_0340/scan_data/data_03) and it is what makes the row end where
        # the panel ends. It used to be capped at 200px as well, so the row
        # stopped short of the panel edge however wide the window was.
        self.le_batch_path.setMinimumWidth(140)
        self.le_batch_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.le_batch_path.setPlaceholderText("Drag or type dataset path")
        self.le_batch_path.setAcceptDrops(True)
        self.le_batch_path.dragEnterEvent = self._batch_path_drag_enter
        self.le_batch_path.dropEvent = self._batch_path_drop
        self.le_batch_path.textEdited.connect(self._sync_batch_path_template_from_visible_text)
        self.le_batch_path.returnPressed.connect(self._handle_batch_preview)
        self.le_batch_path.setToolTip(
            "Drag a dataset from the tree or type the path manually\n"
            "Press Enter to preview the first matched scan"
        )

        # Export button
        self.btn_batch_browse = QPushButton("Export")
        self.btn_batch_browse.setMaximumWidth(80)
        self.btn_batch_browse.clicked.connect(self._handle_batch_export)
        self.btn_batch_browse.setToolTip("Open export settings for this dataset over the selected scan range")

        # Batch add button with menu
        self.btn_batch_add = QPushButton("Add to")
        self.btn_batch_add.setMaximumWidth(100)
        batch_menu = QMenu(self)

        action_to_comparison = QAction("-> Comparison Tool", self)
        action_to_comparison.triggered.connect(lambda: self._batch_add_to_tool("comparison"))
        batch_menu.addAction(action_to_comparison)

        batch_menu.addSeparator()

        action_to_calc_a = QAction("-> Calculator A", self)
        action_to_calc_a.triggered.connect(lambda: self._batch_add_to_tool("calculator_a"))
        batch_menu.addAction(action_to_calc_a)

        action_to_calc_b = QAction("-> Calculator B", self)
        action_to_calc_b.triggered.connect(lambda: self._batch_add_to_tool("calculator_b"))
        batch_menu.addAction(action_to_calc_b)

        batch_menu.addSeparator()

        action_to_calc_ab = QAction("-> Calculator A & B", self)
        action_to_calc_ab.triggered.connect(self._batch_add_to_calculator_ab)
        batch_menu.addAction(action_to_calc_ab)

        batch_menu.addSeparator()

        action_to_fth_cl = QAction("-> FTH as CL", self)
        action_to_fth_cl.triggered.connect(lambda: self._batch_add_to_tool("fth_cl"))
        batch_menu.addAction(action_to_fth_cl)

        action_to_fth_cr = QAction("-> FTH as CR", self)
        action_to_fth_cr.triggered.connect(lambda: self._batch_add_to_tool("fth_cr"))
        batch_menu.addAction(action_to_fth_cr)

        action_to_fth_dark = QAction("-> FTH as Dark", self)
        action_to_fth_dark.triggered.connect(lambda: self._batch_add_to_tool("fth_dark"))
        batch_menu.addAction(action_to_fth_dark)

        self.btn_batch_add.setMenu(batch_menu)
        self.btn_batch_add.setToolTip("Batch add datasets from selected scans to comparison or calculator tool")

        # The two rows under the tree are read as a block, so they have to line
        # up on both edges. Two things stopped them:
        #
        # 1. Only one of them zeroed its margins, so one row was inset by the
        #    style's default layout margin and the other was not — the few
        #    pixels of offset visible at the left.
        # 2. Every widget in the top row had a maximum width, so the row
        #    stopped at the sum of those and left the rest of the panel empty,
        #    while the bottom row's combo stretched to a different cap.
        #
        # Now each row has one widget that stretches and no cap on it, so both
        # rows end exactly at the panel edge whatever the width.
        lyt_plot_type = QHBoxLayout()
        lyt_plot_type.setSpacing(BATCH_ROW_SPACING)
        lyt_plot_type.setContentsMargins(0, 0, 0, 0)
        lbl_plot_as = QLabel("Plot as")
        lbl_plot_as.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lyt_plot_type.addWidget(lbl_plot_as)
        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["Auto", "String", "Array1D", "Array2D", "Table"])
        self.cb_plot_type.currentTextChanged.connect(self._handle_plot_type_changed)
        # Uncapped, so it reaches the same right edge as the Add to button above.
        self.cb_plot_type.setMinimumWidth(120)
        self.cb_plot_type.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lyt_plot_type.addWidget(self.cb_plot_type)

        lyt_filter = QHBoxLayout()
        lyt_filter.setSpacing(BATCH_ROW_SPACING)
        lyt_filter.setContentsMargins(0, 0, 0, 0)
        lyt_filter.addWidget(self.btn_collapse_all)
        lyt_filter.addWidget(self.le_batch_keywords)
        lyt_filter.addWidget(self.le_scan_range)
        lyt_filter.addWidget(self.le_batch_path)
        lyt_filter.addWidget(self.btn_batch_browse)
        lyt_filter.addWidget(self.btn_batch_add)

        lyt_file_tree = QVBoxLayout()
        lyt_file_tree.addWidget(self.tree_view_file)
        lyt_file_tree.addLayout(lyt_filter)
        lyt_file_tree.addLayout(lyt_plot_type)

        wgt_total = QHBoxLayout()
        wgt_total.addLayout(lyt_file_tree)
        # wgt_total.addLayout(self.lyt_dataset)
        wgt_central = QWidget()
        # Set size policy for central widget to allow flexible resizing
        wgt_central.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        wgt_central.setLayout(wgt_total)
        self.setCentralWidget(wgt_central)

        # Debounce dataset plotting to avoid repeated reloads during rapid clicks/changes.
        self._plot_debounce_timer = QTimer(self)
        self._plot_debounce_timer.setSingleShot(True)
        self._plot_debounce_timer.setInterval(200)
        self._plot_debounce_timer.timeout.connect(self._plot_data_debounced)
        self._pending_plot_type = "Auto"
        self._loading_timer = QTimer(self)
        self._loading_timer.setSingleShot(True)
        self._loading_timer.setInterval(180)
        self._loading_timer.timeout.connect(self._show_loading_if_still_loading)

        # File Menu
        if (menu_bar := self.menuBar()) is None:
            return
        if (mbr_file := menu_bar.addMenu("&File")) is not None:
            act_file = QAction("&Open File...", self)
            act_file.setIcon(self._icon_from_name("file.svg"))
            act_file.setShortcut("Ctrl+O")
            act_file.triggered.connect(self._handle_action_open_file)
            mbr_file.addAction(act_file)
            act_open_folder = QAction("&Open Folder...", self)
            act_open_folder.setIcon(self._icon_from_name("group.svg"))
            act_open_folder.triggered.connect(self._handle_action_open_folder)
            mbr_file.addAction(act_open_folder)
            act_clear_files = QAction("&Close all Files", self)
            act_clear_files.setIcon(self._icon_from_name("file_clear.svg"))
            act_clear_files.triggered.connect(self._handle_action_clear_files)
            mbr_file.addAction(act_clear_files)
            mbr_file.addSeparator()
            act_monitor = QAction("&Monitor Folder...", self)
            act_monitor.setShortcut("Ctrl+Shift+M")
            act_monitor.setCheckable(True)
            act_monitor.setToolTip(
                "Watch a folder for new HDF5 files; refresh button updates while active"
            )
            act_monitor.triggered.connect(self._handle_action_monitor_folder)
            mbr_file.addAction(act_monitor)
            self._act_monitor = act_monitor
            mbr_file.addSeparator()
            act_quit = QAction("&Quit", self)
            act_quit.setIcon(self._icon_from_name("quit.svg"))
            act_quit.setShortcut("Ctrl+Q")
            act_quit.triggered.connect(self._handle_close)
            mbr_file.addAction(act_quit)

        # Export / Plot Menu — the two full-function outputs for the selection:
        # one writes a table, the other draws it.
        if (mbr_export := menu_bar.addMenu("&Export/Plot")) is not None:
            act_export_current = QAction("&Export", self)
            act_export_current.setShortcut("Ctrl+E")
            act_export_current.setToolTip("Full export: column layout, dialect and X axis")
            act_export_current.triggered.connect(self._handle_action_export_current)
            mbr_export.addAction(act_export_current)

            act_plot_current = QAction("&Plot", self)
            act_plot_current.setShortcut("Ctrl+Shift+P")
            act_plot_current.setToolTip("Full plot: the data table and a matplotlib figure")
            act_plot_current.triggered.connect(self._handle_action_plot_current)
            mbr_export.addAction(act_plot_current)

        # Tools Menu
        if (mbr_tools := menu_bar.addMenu("&Tools")) is not None:
            act_calculator = QAction("Data &Calculator...", self)
            act_calculator.setShortcut("Ctrl+Shift+C")
            act_calculator.triggered.connect(self._handle_action_calculator)
            mbr_tools.addAction(act_calculator)

            act_comparison = QAction("Data C&omparison...", self)
            act_comparison.setShortcut("Ctrl+Shift+O")
            act_comparison.triggered.connect(self._handle_action_comparison)
            mbr_tools.addAction(act_comparison)

            mbr_tools.addSeparator()

            act_q_cal = QAction("&Scattering Pattern Analyze...", self)
            act_q_cal.setShortcut("Ctrl+Shift+Q")
            act_q_cal.triggered.connect(self._handle_action_q_calibration)
            mbr_tools.addAction(act_q_cal)

            act_fth = QAction("&FTH Reconstruction...", self)
            act_fth.setShortcut("Ctrl+Shift+F")
            act_fth.triggered.connect(self._handle_action_fth)
            mbr_tools.addAction(act_fth)

            act_cdi = QAction("&CDI Reconstruction...", self)
            act_cdi.setShortcut("Ctrl+Shift+D")
            act_cdi.triggered.connect(self._handle_action_cdi)
            mbr_tools.addAction(act_cdi)

            mbr_tools.addSeparator()

            act_xrms = QAction("&Time Resolved XRMS...", self)
            act_xrms.setShortcut("Ctrl+Shift+X")
            act_xrms.triggered.connect(self._handle_action_xrms_analyze)
            mbr_tools.addAction(act_xrms)

        # Setting Menu
        if (mbr_setting := menu_bar.addMenu("&Setting")) is not None:
            m_scope = mbr_setting.addMenu("Index Scope")
            act_scope_fast = QAction("Fast", self)
            act_scope_fast.setCheckable(True)
            act_scope_full = QAction("Full", self)
            act_scope_full.setCheckable(True)
            scope_group = QActionGroup(self)
            scope_group.setExclusive(True)
            scope_group.addAction(act_scope_fast)
            scope_group.addAction(act_scope_full)
            act_scope_fast.setChecked(self._index_scope == "fast")
            act_scope_full.setChecked(self._index_scope == "full")
            act_scope_fast.triggered.connect(lambda _=False: self._set_index_scope("fast"))
            act_scope_full.triggered.connect(lambda _=False: self._set_index_scope("full"))
            m_scope.addAction(act_scope_fast)
            m_scope.addAction(act_scope_full)
            m_scope.addSeparator()
            act_set_fast_paths = QAction("Set Fast Paths...", self)
            act_set_fast_paths.triggered.connect(self._edit_fast_paths)
            m_scope.addAction(act_set_fast_paths)
            mbr_setting.addSeparator()
            act_set_batch_size = QAction("Set Incremental File Threshold...", self)
            act_set_batch_size.triggered.connect(self._edit_index_batch_size)
            mbr_setting.addAction(act_set_batch_size)
            act_set_cache_limit = QAction("Set Index Cache Limit...", self)
            act_set_cache_limit.triggered.connect(self._edit_index_cache_limit)
            mbr_setting.addAction(act_set_cache_limit)

        # Help Menu
        if (mbr_help := menu_bar.addMenu("&Help")) is not None:
            act_about = QAction("&About Page...", self)
            act_about.setIcon(self._icon_from_name("about.svg"))
            act_about.triggered.connect(self._handle_action_about)
            mbr_help.addAction(act_about)

        # Top-right status area: merged runtime + index state.
        status_corner = QWidget()
        self._menu_corner_widget = status_corner
        status_corner.setMinimumWidth(360)
        status_corner.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        status_corner_layout = QHBoxLayout(status_corner)
        status_corner_layout.setContentsMargins(8, 0, 14, 0)
        status_corner_layout.setSpacing(0)

        self._menu_status_label = QLabel("")
        self._menu_status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._menu_status_label.setMinimumWidth(320)
        self._menu_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._menu_status_label.setStyleSheet("color: #444;")
        status_corner_layout.addWidget(self._menu_status_label, stretch=1)

        self._menu_status_raw_text = ""
        self._menu_index_raw_text = "Index: Idle"
        self._menu_index_warming = False

        menu_bar.setCornerWidget(status_corner, Qt.Corner.TopRightCorner)
        self.statusBar().setVisible(False)

        # Open a file passed on the command line immediately (before window shows)
        if len(sys.argv) > 1:
            self._open_file(pathlib.Path(sys.argv[1]))

        # Restore previous session *after* the window is visible so startup is instant.
        _files = settings.value("settings/last_opened_files", ())
        if _files:
            QTimer.singleShot(0, lambda: self._restore_session(_files, ""))

        # Pre-import the heavy reconstruction/tool modules while the app is idle so
        # the first click on a tool doesn't pay the ~150 ms import cost. This only
        # imports modules (no widgets are created), so it is safe on the GUI thread.
        QTimer.singleShot(1500, self._prewarm_tool_modules)

    def _prewarm_tool_modules(self) -> None:
        """Import heavy tool modules ahead of first use to speed up opening them."""
        modules = (
            "src.gui.fth_reconstruction_tool",
            "src.gui.cdi_reconstruction_tool",
            "src.gui.data_calculator_enhanced",
            "src.gui.data_comparison",
            "src.gui.q_calibration_tool",
            "src.gui.xrms_analyze_tool",
        )
        import importlib
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as exc:  # never let prewarming break the app
                logging.debug("Tool module prewarm failed for %s: %s", name, exc)

    def _set_status_text(self, text: str = "") -> None:
        """Show status text in the top-right menu bar corner."""
        self._menu_status_raw_text = text or ""
        self._refresh_menu_corner_texts()

    def _set_index_status(self, text: str, warming: bool = False) -> None:
        """Show index warm state in the top-right menu bar corner."""
        self._menu_index_raw_text = text or ""
        self._menu_index_warming = bool(warming)
        self._refresh_menu_corner_texts()

    def _refresh_menu_corner_texts(self) -> None:
        """Render merged top-right status text without overlap."""
        if not hasattr(self, "_menu_status_label") or self._menu_status_label is None:
            return

        try:
            status_txt = (self._menu_status_raw_text or "").strip()
            index_txt = (self._menu_index_raw_text or "").strip()
            raw = ""
            if status_txt and index_txt:
                raw = f"{status_txt}  |  {index_txt}"
            elif index_txt:
                raw = index_txt
            else:
                raw = status_txt

            if not raw:
                self._menu_status_label.setText("")
                self._menu_status_label.setStyleSheet("color: #444;")
                return

            fm = self._menu_status_label.fontMetrics()
            corner_w = int(self._menu_corner_widget.width()) if hasattr(self, "_menu_corner_widget") else 0
            label_w = int(self._menu_status_label.width())
            basis_w = min(w for w in (corner_w, label_w) if w > 0) if (corner_w > 0 or label_w > 0) else 0
            avail = max(20, basis_w - 24)
            txt = fm.elidedText(raw, Qt.TextElideMode.ElideMiddle, avail)
            self._menu_status_label.setText(txt)
            self._menu_status_label.setToolTip(raw)
            self._menu_status_label.setStyleSheet("color: #8a6d1a;" if self._menu_index_warming else "color: #444;")
        except Exception:
            raw = " | ".join(
                t for t in (
                    (self._menu_status_raw_text or "").strip(),
                    (self._menu_index_raw_text or "").strip(),
                ) if t
            )
            self._menu_status_label.setText(raw)
            self._menu_status_label.setToolTip(raw)
            self._menu_status_label.setStyleSheet("color: #444;")

    def resizeEvent(self, event) -> None:
        """Keep top-right menu texts readable on window resize."""
        super().resizeEvent(event)
        self._refresh_menu_corner_texts()

    def _load_index_scope_settings(self) -> None:
        """Load index scope configuration from QSettings."""
        settings = QSettings()
        scope = str(settings.value("settings/index_scope", "fast")).strip().lower()
        self._index_scope = "full" if scope == "full" else "fast"
        raw_paths = str(settings.value("settings/index_fast_paths", "scan_data"))
        paths = [p.strip().strip("/") for p in raw_paths.split(",") if p.strip()]
        self._fast_group_paths = tuple(paths) if paths else ("scan_data",)
        try:
            self._index_batch_size = max(1, int(settings.value("settings/index_batch_size", 50)))
        except Exception:
            self._index_batch_size = 50
        try:
            self._index_cache_max_files = max(100, int(settings.value("settings/index_cache_max_files", 5000)))
        except Exception:
            self._index_cache_max_files = 5000

    def _save_index_scope_settings(self) -> None:
        """Persist index scope configuration into QSettings."""
        settings = QSettings()
        settings.setValue("settings/index_scope", self._index_scope)
        settings.setValue("settings/index_fast_paths", ",".join(self._fast_group_paths))
        settings.setValue("settings/index_batch_size", int(self._index_batch_size))
        settings.setValue("settings/index_cache_max_files", int(self._index_cache_max_files))

    def _prune_index_cache(self, protect_opened: bool = False) -> None:
        """Trim in-memory index cache to max entries to prevent unbounded growth."""
        max_files = max(100, int(self._index_cache_max_files))
        cache = self._dataset_per_file_index_cache
        if len(cache) <= max_files:
            return

        protected: set[str] = set()
        if protect_opened and hasattr(self, "tree_model_file"):
            protected = {str(p) for p in self.opened_files}

        # Evict least-recently-used first, prefer non-protected entries.
        candidates = sorted(
            (
                (
                    int(self._dataset_index_last_used.get(fp, int(rec[0][0]))),
                    fp,
                )
                for fp, rec in cache.items()
                if fp not in protected
            ),
            key=lambda t: t[0],
        )
        for _lru, fp in candidates:
            if len(cache) <= max_files:
                break
            cache.pop(fp, None)
            self._dataset_index_last_used.pop(fp, None)

        if len(cache) <= max_files:
            return

        # If still above cap, evict oldest protected too (hard cap enforcement).
        candidates_all = sorted(
            (
                (
                    int(self._dataset_index_last_used.get(fp, int(rec[0][0]))),
                    fp,
                )
                for fp, rec in cache.items()
            ),
            key=lambda t: t[0],
        )
        for _lru, fp in candidates_all:
            if len(cache) <= max_files:
                break
            cache.pop(fp, None)
            self._dataset_index_last_used.pop(fp, None)

    def _ensure_on_screen(self) -> None:
        """Keep a restored window geometry visible.

        A saved position from a since-disconnected monitor can land the window
        off every screen; if so, re-centre it on the primary screen.
        """
        from PyQt6.QtGui import QGuiApplication
        frame = self.frameGeometry()
        for screen in QGuiApplication.screens():
            overlap = screen.availableGeometry().intersected(frame)
            if overlap.width() >= 100 and overlap.height() >= 30:
                return  # enough of the window is on a real screen
        primary = QGuiApplication.primaryScreen()
        if primary is not None:
            avail = primary.availableGeometry()
            x = avail.x() + max(0, (avail.width() - self.width()) // 2)
            y = avail.y() + max(0, (avail.height() - self.height()) // 2)
            self.move(x, y)
        else:
            self.move(300, 150)

    @staticmethod
    def _disk_index_cache_path() -> pathlib.Path:
        """Return persistent dataset-index cache file path."""
        app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        if app_data:
            return pathlib.Path(app_data) / "hdf5_viewer_index_cache.json"
        return pathlib.Path.home() / ".hdf5_viewer_index_cache.json"

    def _load_disk_index_cache(self) -> None:
        """Load persistent per-file index cache from disk."""
        p = self._disk_index_cache_path()
        if not p.exists():
            return
        try:
            with p.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return
            if int(payload.get("version", 0)) != self._INDEX_CACHE_VERSION:
                return
            if str(payload.get("scope", "fast")).lower() != self._index_scope:
                return
            cached_paths = tuple(
                str(x).strip().strip("/") for x in payload.get("fast_paths", []) if str(x).strip()
            )
            if self._index_scope == "fast" and cached_paths and cached_paths != self._fast_group_paths:
                return
            files = payload.get("files", {})
            if not isinstance(files, dict):
                return
            loaded: dict[str, tuple[tuple[int, int], list[str], list[str]]] = {}
            for fp, rec in files.items():
                if not isinstance(fp, str) or not isinstance(rec, dict):
                    continue
                sig = rec.get("sig")
                k1 = rec.get("keys_1d", [])
                k2 = rec.get("keys_2d_fth", [])
                lu = rec.get("last_used", int(sig[0]) if isinstance(sig, (list, tuple)) and len(sig) == 2 else 0)
                if (
                    not isinstance(sig, (list, tuple))
                    or len(sig) != 2
                    or not isinstance(k1, list)
                    or not isinstance(k2, list)
                ):
                    continue
                loaded[fp] = ((int(sig[0]), int(sig[1])), [str(x) for x in k1], [str(x) for x in k2])
                try:
                    self._dataset_index_last_used[fp] = int(lu)
                except Exception:
                    self._dataset_index_last_used[fp] = int(sig[0])
            self._dataset_per_file_index_cache = loaded
            self._prune_index_cache(protect_opened=False)
            logging.info("Loaded disk index cache: %d files", len(loaded))
        except Exception as exc:
            logging.warning("Failed to load disk index cache: %s", exc)

    def _save_disk_index_cache(self) -> None:
        """Persist per-file index cache to disk."""
        p = self._disk_index_cache_path()
        try:
            self._prune_index_cache(protect_opened=True)
            p.parent.mkdir(parents=True, exist_ok=True)
            files_payload = {}
            for fp, rec in self._dataset_per_file_index_cache.items():
                sig, keys_1d, keys_2d = rec
                files_payload[fp] = {
                    "sig": [int(sig[0]), int(sig[1])],
                    "keys_1d": list(keys_1d),
                    "keys_2d_fth": list(keys_2d),
                    "last_used": int(self._dataset_index_last_used.get(fp, int(sig[0]))),
                }
            payload = {
                "version": self._INDEX_CACHE_VERSION,
                "scope": self._index_scope,
                "fast_paths": list(self._fast_group_paths),
                "files": files_payload,
            }
            with p.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=True, separators=(",", ":"))
        except Exception as exc:
            logging.warning("Failed to save disk index cache: %s", exc)

    def _set_index_scope(self, scope: str) -> None:
        """Set index scope ('fast' or 'full') and trigger rebuild."""
        new_scope = "full" if str(scope).strip().lower() == "full" else "fast"
        if new_scope == self._index_scope:
            return
        self._index_scope = new_scope
        self._save_index_scope_settings()
        self._dataset_per_file_index_cache.clear()
        self._dataset_index_last_used.clear()
        self._set_index_status("Index: Scope changed")
        self._prime_dataset_index_async()

    def _edit_fast_paths(self) -> None:
        """Edit fast-scan group paths (comma separated)."""
        current = ",".join(self._fast_group_paths) if self._fast_group_paths else "scan_data"
        text, ok = QInputDialog.getText(
            self,
            "Set Fast Paths",
            "Group path keywords (comma separated):",
            text=current,
        )
        if not ok:
            return
        paths = [p.strip().strip("/") for p in str(text).split(",") if p.strip()]
        if not paths:
            paths = ["scan_data"]
        new_paths = tuple(paths)
        if new_paths == self._fast_group_paths:
            return
        self._fast_group_paths = new_paths
        self._save_index_scope_settings()
        if self._index_scope == "fast":
            self._dataset_per_file_index_cache.clear()
            self._dataset_index_last_used.clear()
            self._set_index_status("Index: Fast paths changed")
            self._prime_dataset_index_async()

    def _edit_index_batch_size(self) -> None:
        """Edit incremental index batch size (files per UI push)."""
        val, ok = QInputDialog.getInt(
            self,
            "Set Index Batch Size",
            "Files per incremental update:",
            int(self._index_batch_size),
            1,
            10000,
            1,
        )
        if not ok:
            return
        new_val = max(1, int(val))
        if new_val == self._index_batch_size:
            return
        self._index_batch_size = new_val
        self._save_index_scope_settings()
        self._set_index_status(f"Index: Batch size = {self._index_batch_size}")

    def _edit_index_cache_limit(self) -> None:
        """Edit maximum number of files kept in disk index cache."""
        val, ok = QInputDialog.getInt(
            self,
            "Set Index Cache Limit",
            "Maximum cached files:",
            int(self._index_cache_max_files),
            100,
            200000,
            100,
        )
        if not ok:
            return
        new_val = max(100, int(val))
        if new_val == self._index_cache_max_files:
            return
        self._index_cache_max_files = new_val
        self._prune_index_cache(protect_opened=True)
        self._save_index_scope_settings()
        self._save_disk_index_cache()
        self._set_index_status(f"Index: Cache limit = {self._index_cache_max_files}")

    @staticmethod
    def iter_items(root: QStandardItem) -> Generator[Any, Any, None]:
        """Iterate recursively through all children of a QStandardItem."""

        def recurse(parent: QStandardItem) -> Generator[Any, Any, None]:
            for row in range(parent.rowCount()):
                if (child := parent.child(row, 0)) is not None:
                    yield child.text()
                    if child.hasChildren():
                        yield from recurse(child)

        if root is not None:
            yield from recurse(root)

    @property
    def selected_item(self) -> tuple[pathlib.Path, str, Any]:
        """Tuple of selected file name, object name and object type."""
        if not self.cur_obj_path:
            obj_type = h5py.File
        else:
            with h5py.File(self.cur_file, "r") as file:
                obj_type = type(file[self.cur_obj_path])

        return self.cur_file, self.cur_obj_path, obj_type

    @property
    def opened_files(self) -> tuple[pathlib.Path, ...]:
        """Currently opened files."""
        file_paths = []
        for i in range(self.tree_model_file.rowCount()):
            item = self.tree_model_file.item(i, TREE_COLUMN_NAME)
            # From the stored path, not the label: every tool in the
            # application is handed this tuple, and the label is a filename.
            if item is not None and (path := tree_item_file_path(item)):
                file_paths.append(pathlib.Path(path))
        return tuple(file_paths)

    @staticmethod
    def _file_signature(path_str: str) -> tuple[int, int]:
        st = os.stat(path_str)
        return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))), int(st.st_size)

    @staticmethod
    def _scan_number_from_filename(name: str) -> int:
        """Extract last numeric token from filename stem; return -1 if missing."""
        stem = pathlib.Path(name).stem
        import re
        m = re.search(r"(\d+)(?!.*\d)", stem)
        if m is None:
            return -1
        try:
            return int(m.group(1))
        except Exception:
            return -1

    def _sorted_opened_files_for_index(self) -> list[pathlib.Path]:
        """Sort files by parent folder, then scan number, then filename."""
        files = list(self.opened_files)
        files.sort(
            key=lambda p: (
                str(pathlib.Path(p).parent).lower(),
                self._scan_number_from_filename(pathlib.Path(p).name),
                pathlib.Path(p).name.lower(),
                str(p).lower(),
            )
        )
        return files

    def _aggregate_cached_keys(self, min_ndim: int = 1, min_second_dim: int = 0) -> list[str]:
        """Aggregate per-file cached keys with stable grouped scan ordering."""
        out: list[str] = []
        use_1d = (min_ndim, min_second_dim) == (1, 0)
        use_fth2d = (min_ndim, min_second_dim) == (2, FTH_MIN_SECOND_DIM)
        for fp in self._sorted_opened_files_for_index():
            rec = self._dataset_per_file_index_cache.get(str(fp))
            if rec is None:
                continue
            self._dataset_index_last_used[str(fp)] = time.time_ns()
            if use_1d:
                out.extend(rec[1])
            elif use_fth2d:
                out.extend(rec[2])
            else:
                logging.debug(
                    "Unsupported key filter (%s, %s) requested; returning cached subsets only.",
                    min_ndim,
                    min_second_dim,
                )
        return out

    def _get_dataset_full_keys(self, min_ndim: int = 1, min_second_dim: int = 0) -> list[str]:
        """Return keys; synchronously fills missing/changed files only (incremental)."""
        per_file = dict(self._dataset_per_file_index_cache)
        touched = False
        opened_set = {str(p) for p in self.opened_files}

        # Drop cache records for files no longer opened.
        stale = [fp for fp in per_file.keys() if fp not in opened_set]
        if stale:
            touched = True
            for fp in stale:
                per_file.pop(fp, None)
                self._dataset_index_last_used.pop(fp, None)

        # Refresh only missing/changed files.
        for fp in self.opened_files:
            fp_str = str(fp)
            try:
                sig = self._file_signature(fp_str)
            except Exception:
                continue
            rec = per_file.get(fp_str)
            if rec is not None and rec[0] == sig:
                continue
            keys_1d, keys_2d_fth = _DatasetIndexWarmWorker._scan_single_file_both(
                fp_str,
                index_scope=self._index_scope,
                fast_group_paths=self._fast_group_paths,
            )
            per_file[fp_str] = (sig, keys_1d, keys_2d_fth)
            self._dataset_index_last_used[fp_str] = time.time_ns()
            touched = True

        if touched:
            self._dataset_per_file_index_cache = per_file
        return self._aggregate_cached_keys(min_ndim=min_ndim, min_second_dim=min_second_dim)

    def _peek_dataset_full_keys(self, min_ndim: int = 1, min_second_dim: int = 0) -> list[str]:
        """Return cached keys only; do not trigger a synchronous scan."""
        return self._aggregate_cached_keys(min_ndim=min_ndim, min_second_dim=min_second_dim)

    def _prime_dataset_index_async(self) -> None:
        """Prewarm shared dataset indices on a background thread."""
        opened = self.opened_files
        if not opened:
            self._dataset_per_file_index_cache.clear()
            self._dataset_index_last_used.clear()
            self._set_index_status("Index: Idle")
            self.dataset_index_changed.emit()
            return

        if self._index_warm_worker is not None and self._index_warm_worker.isRunning():
            return

        scope_txt = f"fast:{'/'.join(self._fast_group_paths)}" if self._index_scope == "fast" else "full"
        self._set_index_status(f"Index: Warming [{scope_txt}]...", warming=True)
        self._index_warm_worker = _DatasetIndexWarmWorker(
            opened,
            prev_cache=self._dataset_per_file_index_cache,
            index_scope=self._index_scope,
            fast_group_paths=self._fast_group_paths,
            batch_size=self._index_batch_size,
            parent=self,
        )
        self._index_warm_worker.batch.connect(self._on_dataset_index_warm_batch)
        self._index_warm_worker.done.connect(self._on_dataset_index_warm_done)
        self._index_warm_worker.start()

    def _on_dataset_index_warm_batch(
        self,
        delta_cache: dict[str, tuple[tuple[int, int], list[str], list[str]]],
        processed: int,
        total: int,
        worker_scope: str,
        worker_fast_paths: tuple[str, ...],
    ) -> None:
        """Incrementally merge one index batch and refresh open tools."""
        if worker_scope != self._index_scope or tuple(worker_fast_paths) != tuple(self._fast_group_paths):
            return
        if delta_cache:
            self._dataset_per_file_index_cache.update(delta_cache)
            now = time.time_ns()
            for fp in delta_cache.keys():
                self._dataset_index_last_used[fp] = now
        scope_txt = f"fast:{'/'.join(self._fast_group_paths)}" if self._index_scope == "fast" else "full"
        self._set_index_status(f"Index: Warming [{scope_txt}] {processed}/{max(1, total)}", warming=True)
        self.dataset_index_changed.emit()

    def _on_dataset_index_warm_done(
        self,
        per_file_cache: dict[str, tuple[tuple[int, int], list[str], list[str]]],
        worker_scope: str,
        worker_fast_paths: tuple[str, ...],
    ) -> None:
        """Store warmed indices and notify open tools."""
        if worker_scope != self._index_scope or tuple(worker_fast_paths) != tuple(self._fast_group_paths):
            # Scope changed while worker was running; discard stale result and rerun.
            self._index_warm_worker = None
            self._prime_dataset_index_async()
            return
        self._dataset_per_file_index_cache = dict(per_file_cache)
        now = time.time_ns()
        for fp in self._dataset_per_file_index_cache.keys():
            self._dataset_index_last_used.setdefault(fp, now)
        self._save_disk_index_cache()
        self._index_warm_worker = None
        keys_1d = self._aggregate_cached_keys(min_ndim=1, min_second_dim=0)
        keys_2d_fth = self._aggregate_cached_keys(min_ndim=2, min_second_dim=FTH_MIN_SECOND_DIM)
        scope_txt = f"fast:{'/'.join(self._fast_group_paths)}" if self._index_scope == "fast" else "full"
        self._set_index_status(
            f"Index: Ready [{scope_txt}] ({len(keys_1d)} / {len(keys_2d_fth)})"
        )
        self.dataset_index_changed.emit()

    @pyqtSlot()
    def _refresh_open_tools_dataset_index(self) -> None:
        """Push latest shared index to already-open tools."""
        keys_1d = self._peek_dataset_full_keys(min_ndim=1)
        keys_2d_fth = self._peek_dataset_full_keys(min_ndim=2, min_second_dim=FTH_MIN_SECOND_DIM)

        if hasattr(self, "calculator") and self.calculator is not None and self.calculator.isVisible():
            self.calculator.refresh_dataset_keys(keys_1d, opened_files=self.opened_files)
        if hasattr(self, "comparison_tool") and self.comparison_tool is not None and self.comparison_tool.isVisible():
            self.comparison_tool.refresh_dataset_keys(keys_1d, opened_files=self.opened_files)
        if hasattr(self, "fth_tool") and self.fth_tool is not None and self.fth_tool.isVisible():
            self.fth_tool.refresh_dataset_keys(keys_2d_fth, opened_files=self.opened_files)
        if hasattr(self, "q_cal_tool") and self.q_cal_tool is not None and self.q_cal_tool.isVisible():
            self.q_cal_tool.set_opened_files(self.opened_files)
            self.q_cal_tool.refresh_dataset_keys(keys_2d_fth)
        if hasattr(self, "cdi_tool") and self.cdi_tool is not None and self.cdi_tool.isVisible():
            keys_2d = self._peek_dataset_full_keys(min_ndim=2)
            self.cdi_tool.update_opened_files(self.opened_files, keys_2d)

    def _restore_session(self, files, monitor_folder_str: str) -> None:
        """Restore the previous session after the main window is visible.

        Called via QTimer.singleShot(0) so the window appears instantly on
        startup regardless of how many files need to be reloaded.
        Monitor folder is never auto-restored; user must re-enable it manually.
        """
        if files:
            self._start_open_queue(list(files), mode="restore", mark_known=False)

    def _icon_from_name(self, icon_name: str) -> QIcon:
        """Load icon safely; return empty icon when file is missing."""
        icon_path = pathlib.Path(self.icon_dir, icon_name)
        if icon_path.exists():
            return QIcon(str(icon_path))
        if icon_name not in self._missing_icons_logged:
            logging.warning("Icon not found: %s", icon_path)
            self._missing_icons_logged.add(icon_name)
        return QIcon()

    def _open_file(self, file_path: pathlib.Path) -> None:
        """
        Open one File.

        :param str file_path: File Path
        """
        if not is_supported_data_file(file_path):
            logging.warning("Skipped unsupported file: '%s'", file_path)
            return

        logging.info(f"Open file '{file_path}'")
        # Lazy strategy: do not recurse file contents on add/open.
        # Labelled with the filename, which is what a scan is looked up by; the
        # absolute path goes in the role every reader uses and in the Folder
        # column, so nothing has to widen this one to read it.
        parent_name = QStandardItem(pathlib.Path(file_path).name)
        parent_name.setEditable(False)
        parent_name.setToolTip(str(file_path))
        parent_name.setData("/", _ROLE_H5_PATH)
        parent_name.setData("file", _ROLE_NODE_TYPE)
        parent_name.setData(False, _ROLE_CHILDREN_LOADED)
        parent_name.setData(str(file_path), _ROLE_FILE_PATH)

        parent_text = QStandardItem("HDF5 File")
        parent_text.setEditable(False)
        parent_text.setIcon(self._icon_from_name("file.svg"))
        # Files don't have shape
        parent_shape = QStandardItem("-")
        parent_shape.setEditable(False)

        # The containing folder, not the whole path: the filename is already in
        # the first column and printing it twice spends the width just saved.
        parent_folder = QStandardItem(str(pathlib.Path(file_path).parent))
        parent_folder.setEditable(False)
        parent_folder.setToolTip(str(file_path))

        self.tree_model_file.appendRow([parent_name, parent_text, parent_shape, parent_folder])
        if not is_hdf5_file(file_path):
            parent_text.setText(_regular_file_kind(file_path))
            parent_name.setData(True, _ROLE_CHILDREN_LOADED)
            child_name = QStandardItem("data")
            child_name.setEditable(False)
            child_name.setData(_REGULAR_DATASET_PATH, _ROLE_H5_PATH)
            child_name.setData("dataset", _ROLE_NODE_TYPE)
            child_type = QStandardItem(pathlib.Path(file_path).suffix.lower().lstrip(".").upper())
            child_type.setEditable(False)
            child_type.setIcon(self._icon_from_name("dataset.svg"))
            child_shape = QStandardItem("-")
            child_shape.setEditable(False)
            parent_name.appendRow([child_name, child_type, child_shape, tree_blank_cell()])
            return
        self._append_lazy_placeholder(parent_name)

    def _append_lazy_placeholder(self, parent_item: QStandardItem) -> None:
        """Add a dummy child so Qt shows expand arrow before real children are loaded."""
        dummy_name = QStandardItem("...")
        dummy_name.setEditable(False)
        dummy_name.setData("placeholder", _ROLE_NODE_TYPE)
        dummy_type = QStandardItem("-")
        dummy_type.setEditable(False)
        dummy_shape = QStandardItem("-")
        dummy_shape.setEditable(False)
        parent_item.appendRow([dummy_name, dummy_type, dummy_shape, tree_blank_cell()])

    def _clear_placeholders(self, parent_item: QStandardItem) -> None:
        """Remove placeholder rows under a parent item."""
        for row in range(parent_item.rowCount() - 1, -1, -1):
            child = parent_item.child(row, 0)
            if child is not None and child.data(_ROLE_NODE_TYPE) == "placeholder":
                parent_item.removeRow(row)

    def _file_path_for_item(self, item: QStandardItem) -> pathlib.Path:
        """Resolve owning file path from any item in the tree."""
        return pathlib.Path(tree_item_file_path(item))

    def _load_tree_children(self, parent_item: QStandardItem) -> None:
        """Load one level of HDF5 children for a file/group tree item."""
        if parent_item.data(_ROLE_CHILDREN_LOADED):
            return

        node_type = parent_item.data(_ROLE_NODE_TYPE)
        if node_type not in ("file", "group"):
            return

        h5_path = parent_item.data(_ROLE_H5_PATH)
        if not h5_path:
            h5_path = "/"
        file_path = self._file_path_for_item(parent_item)

        self._clear_placeholders(parent_item)
        try:
            with h5py.File(file_path, "r") as f:
                obj = f[h5_path]
                for name in natsorted(obj):
                    value = obj[name]
                    if isinstance(value, h5py.Group):
                        child_name = QStandardItem(name)
                        child_name.setEditable(False)
                        child_name.setData(
                            f"{h5_path.rstrip('/')}/{name}" if h5_path != "/" else f"/{name}",
                            _ROLE_H5_PATH,
                        )
                        child_name.setData("group", _ROLE_NODE_TYPE)
                        child_name.setData(False, _ROLE_CHILDREN_LOADED)

                        child_type = QStandardItem("Group")
                        child_type.setEditable(False)
                        child_type.setIcon(self._icon_from_name("group.svg"))

                        child_shape = QStandardItem("-")
                        child_shape.setEditable(False)
                        parent_item.appendRow([child_name, child_type, child_shape, tree_blank_cell()])
                        self._append_lazy_placeholder(child_name)
                    elif isinstance(value, h5py.Dataset):
                        child_name = QStandardItem(name)
                        child_name.setEditable(False)
                        child_name.setData(
                            f"{h5_path.rstrip('/')}/{name}" if h5_path != "/" else f"/{name}",
                            _ROLE_H5_PATH,
                        )
                        child_name.setData("dataset", _ROLE_NODE_TYPE)
                        child_name.setData(True, _ROLE_CHILDREN_LOADED)

                        child_type = QStandardItem(str(value.dtype))
                        child_type.setEditable(False)
                        child_type.setIcon(self._icon_from_name("dataset.svg"))

                        child_shape = QStandardItem(str(value.shape))
                        child_shape.setEditable(False)
                        parent_item.appendRow([child_name, child_type, child_shape, tree_blank_cell()])

            parent_item.setData(True, _ROLE_CHILDREN_LOADED)
        except Exception as err:
            logging.warning(f"Lazy load failed for '{file_path}::{h5_path}': {err}")
            # Keep one placeholder to indicate expandable node without crashing UI.
            if parent_item.rowCount() == 0:
                self._append_lazy_placeholder(parent_item)

    @pyqtSlot(QModelIndex)
    def _on_tree_item_expanded(self, proxy_index: QModelIndex) -> None:
        """Lazy-load children when a tree node is expanded."""
        if not proxy_index.isValid():
            return
        src_index = self.tree_model_file_proxy.mapToSource(proxy_index)
        item = self.tree_model_file.itemFromIndex(src_index.sibling(src_index.row(), 0))
        if item is None:
            return
        self._load_tree_children(item)

        # Removed old filter completer - now using batch add functionality

    def _hdf5_recursion(
        self,
        hdf5_object: h5py.File | h5py.Group | h5py.Dataset,
        root: QStandardItem,
        parent: QStandardItem,
    ) -> None:
        """Recursively go through hdf5 File and construct tree view model."""
        for name in natsorted(hdf5_object):
            value = hdf5_object[name]
            if isinstance(value, h5py.Group):
                child_name = QStandardItem(name)
                child_name.setEditable(False)
                child_type = QStandardItem("Group")
                child_type.setEditable(False)
                child_type.setIcon(self._icon_from_name("group.svg"))
                # Groups don't have shape
                child_shape = QStandardItem("-")
                child_shape.setEditable(False)
                parent.appendRow([child_name, child_type, child_shape, tree_blank_cell()])
                self._hdf5_recursion(value, root, child_name)
            elif isinstance(value, h5py.Dataset):
                child_name = QStandardItem(name)
                child_name.setEditable(False)
                # For datasets, Type column shows the data type
                child_type = QStandardItem(str(value.dtype))
                child_type.setEditable(False)
                child_type.setIcon(self._icon_from_name("dataset.svg"))
                # Shape column shows the shape
                child_shape = QStandardItem(str(value.shape))
                child_shape.setEditable(False)
                parent.appendRow([child_name, child_type, child_shape, tree_blank_cell()])

    @pyqtSlot()
    def _plot_data(self, plot_type: str = "") -> None:
        """Load and display the selected HDF5 dataset (non-blocking)."""
        if self.cur_file is None or not self.cur_obj_path or not os.path.exists(self.cur_file):
            return

        cache_key = (str(self.cur_file), self.cur_obj_path)

        # Serve from cache when available (avoids re-reading from server)
        if cache_key in self._dataset_cache:
            self._loading_timer.stop()
            data, data_type_str = self._dataset_cache[cache_key]
            self._dataset_cache.move_to_end(cache_key)
            logging.info(f"Cache hit: {self.cur_obj_path}")
            # Honour user-selected plot type; fall back to cached auto-detected type
            effective_type = plot_type if plot_type and plot_type != "Auto" else data_type_str
            source_key = f"{self.cur_file}::{self.cur_obj_path}" if self.cur_obj_path else None
            self._show_data(data, effective_type, source_dataset_key=source_key)
            return

        # Cancel any still-running load for a previous selection
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.cancel()
            self._load_worker.wait(300)

        # Start background load
        self._load_worker = DataLoadWorker(self.cur_file, self.cur_obj_path, plot_type)
        self._load_worker.data_ready.connect(self._on_data_ready)
        self._load_worker.data_ready_lazy.connect(self._on_data_ready_lazy)
        self._load_worker.load_error.connect(self._on_load_error)
        self._load_worker.start()
        # Delay loading placeholder to avoid flicker on fast loads.
        self._loading_timer.start()

    # ------------------------------------------------------------------
    # Background-load callbacks
    # ------------------------------------------------------------------

    @pyqtSlot(object, str, str, str)
    def _on_data_ready(self, data, data_type_str, file_path, obj_path):
        """Called from worker thread when a full dataset has been loaded."""
        # Discard stale results if the user already clicked elsewhere
        if str(self.cur_file) != file_path or self.cur_obj_path != obj_path:
            return
        self._loading_timer.stop()

        # Add to LRU cache (only for reasonably sized datasets)
        if hasattr(data, "nbytes") and data.nbytes < _LAZY_LOAD_THRESHOLD:
            cache_key = (file_path, obj_path)
            self._dataset_cache[cache_key] = (data, data_type_str)
            if len(self._dataset_cache) > _DATASET_CACHE_SIZE:
                self._dataset_cache.popitem(last=False)

        source_key = f"{file_path}::{obj_path}" if obj_path else None
        self._show_data(data, data_type_str, source_dataset_key=source_key)

    @pyqtSlot(object, object, str, str, str)
    def _on_data_ready_lazy(self, first_slice, shape, data_type_str, file_path, obj_path):
        """Called when a large 3D dataset is opened in lazy mode (only first slice loaded)."""
        if str(self.cur_file) != file_path or self.cur_obj_path != obj_path:
            return
        self._loading_timer.stop()

        # Build a closure that opens the file on demand for each requested slice
        captured_path = file_path
        captured_obj = obj_path

        def slice_loader(axis: int, idx: int) -> np.ndarray:
            with h5py.File(captured_path, "r", rdcc_nbytes=_H5PY_CHUNK_CACHE) as f:
                ds = f[captured_obj]
                selection = [slice(None)] * len(ds.shape)
                selection[int(axis)] = int(idx)
                return np.array(ds[tuple(selection)])

        from src.gui.unified_data_viewer import UnifiedDataViewer
        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
        from PyQt6.QtWidgets import QSizePolicy

        viewer = UnifiedDataViewer(
            parent=self,
            opened_files=self.opened_files,
            dataset_full_keys_1d=self._peek_dataset_full_keys(min_ndim=1),
        )
        viewer.q_calibration_requested.connect(self._handle_q_request_from_viewer)
        viewer.source_dataset_key = f"{file_path}::{obj_path}"
        image_view = ImageView2DEnhanced(parent=viewer)
        image_view.q_calibration_requested.connect(
            lambda: self._handle_q_request_from_viewer(viewer.source_dataset_key)
        )
        image_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        image_view.set_data_lazy(first_slice, shape[0], slice_loader, full_shape=tuple(shape))
        viewer.layout.addWidget(image_view)
        viewer.current_widget = image_view

        self._finalize_dock(viewer)

    @pyqtSlot(str)
    def _on_load_error(self, error_msg):
        """Show an error label when the background load fails."""
        self._loading_timer.stop()
        # Nothing got displayed, so a queued tree action has nothing to act on.
        self._pending_tree_action = ""
        from PyQt6.QtWidgets import QLabel
        label = QLabel(f"Error loading data:\n{error_msg}")
        label.setStyleSheet("color: red; padding: 10px;")
        label.setWordWrap(True)
        self.dock_plot.setWidget(label)
        logging.error(f"Failed to load dataset: {error_msg}")

    def _show_loading_if_still_loading(self) -> None:
        """Show loading placeholder only for genuinely slow loads."""
        if self._load_worker is not None and self._load_worker.isRunning():
            self._show_loading_indicator()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_loading_indicator(self):
        """Replace dock content with a lightweight loading label."""
        from PyQt6.QtWidgets import QLabel
        label = QLabel("Loading data...")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: gray; font-size: 11pt;")
        self.dock_plot.setWidget(label)

    def _show_data(self, data, data_type_str, source_dataset_key: str | None = None):
        """Create a UnifiedDataViewer for already-loaded data and dock it."""
        from src.gui.unified_data_viewer import UnifiedDataViewer
        viewer = UnifiedDataViewer(
            parent=self,
            opened_files=self.opened_files,
            dataset_full_keys_1d=self._peek_dataset_full_keys(min_ndim=1),
        )
        viewer.q_calibration_requested.connect(self._handle_q_request_from_viewer)
        try:
            viewer.set_data(data, data_type=data_type_str, source_dataset_key=source_dataset_key)
        except Exception as err:
            logging.error(f"Failed to display data: {err}")
            return
        self._finalize_dock(viewer)

    def _quick_export_tree_dataset(self, index: QModelIndex) -> None:
        """Show a right-clicked dataset, then export exactly what is shown."""
        self._run_tree_dataset_action(index, "export")

    def _plot_tree_dataset(self, index: QModelIndex) -> None:
        """Show a right-clicked dataset, then plot it."""
        self._run_tree_dataset_action(index, "plot")

    def _run_tree_dataset_action(self, index: QModelIndex, action: str) -> None:
        """Display a right-clicked dataset, then act on what is displayed.

        Both tree actions mean "what is on screen", so the dataset is shown
        first rather than acted on blind — that keeps the rule free of
        exceptions, and it is what lets the plot pick up a custom X that
        actually belongs to this curve. The load may be asynchronous, so the
        request is queued and fired by :meth:`_finalize_dock` once the viewer
        holds the data.
        """
        self._pending_tree_action = action
        self._handle_item_changed(index)
        # This is one explicit request, not rapid tree browsing: skip the debounce.
        self._plot_debounce_timer.stop()
        self._plot_data(self.cb_plot_type.currentText())

    def _x_dataset_path(self) -> str:
        """The "Set as X" choice as a bare dataset path, without its file.

        The export dialogs address a path inside each matched scan file, so the
        ``file::`` half of the remembered key is dropped there.
        """
        if not self._x_dataset_key:
            return ""
        return self._x_dataset_key.split("::", 1)[-1]

    def _set_dataset_as_x(self, parents_list: list) -> None:
        """Adopt a right-clicked dataset as the X axis.

        Saves the drag: the choice is applied to the curve on screen straight
        away and remembered as the default X for every export and plot dialog
        opened afterwards. A length mismatch only stops the immediate apply —
        the dataset stays the default, because the next scan may well fit.
        """
        if len(parents_list) < 2:
            return
        file_path = parents_list[0]
        ds_path = "/".join(parents_list[1:])

        try:
            with h5py.File(file_path, "r") as file:
                obj = file[ds_path]
                if not isinstance(obj, h5py.Dataset):
                    QMessageBox.warning(self, "Not a Dataset", "Pick a dataset, not a group.")
                    return
                x_data = np.asarray(obj[()]).squeeze()
        except Exception as exc:
            logging.error("Set as X: could not read %s::%s: %s", file_path, ds_path, exc)
            QMessageBox.critical(self, "Cannot Read Dataset", f"Could not read the dataset:\n{exc}")
            return

        if x_data.ndim != 1:
            QMessageBox.warning(
                self,
                "Not a 1-D Dataset",
                f"An X axis has to be one dimensional; this one is {x_data.ndim}-D.",
            )
            return

        key = f"{file_path}::{ds_path}"
        leaf = ds_path.rstrip("/").split("/")[-1]

        # An open Plot or export window that takes an X wins over the viewer:
        # if one is in front, that is the X the user means to set.
        target = active_x_target()

        # Remembered for the tool it actually went to, and only for that tool.
        # Remembering it globally meant an X set in the calculator came back as
        # the default in a batch export over a different set of files.
        scope = x_scope_of(target) if target is not None else DEFAULT_X_SCOPE
        remember_x_dataset(key, scope)
        if scope == DEFAULT_X_SCOPE:
            # The main window's own choice, which the batch export starts from.
            self._x_dataset_key = key

        if target is not None:
            title = target.windowTitle() or "the open window"
            if target.set_x_dataset(key):
                self._set_status_text(f"X axis set to {leaf} in {title}.")
            else:
                self._set_status_text(f"{title} did not accept {leaf} as X.")
            return

        widget = self._current_plot_widget_1d()
        if widget is None or getattr(widget, "y_data", None) is None:
            self._set_status_text(f"X axis set to {leaf} (no curve displayed yet).")
            return

        if len(x_data) != len(widget.y_data):
            self._set_status_text(
                f"X axis set to {leaf}, but it is {len(x_data)} long and the displayed "
                f"curve is {len(widget.y_data)} — not applied to this curve."
            )
            return

        widget._on_x_data_selected(x_data, key)
        self._set_status_text(f"X axis set to {leaf}.")

    def _run_pending_tree_action(self) -> None:
        """Fire a queued tree Export or Plot once the viewer is populated."""
        action, self._pending_tree_action = self._pending_tree_action, ""
        if not action:
            return

        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
        from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
        from src.gui.unified_data_viewer import UnifiedDataViewer

        widget = self.dock_plot.widget()
        if isinstance(widget, UnifiedDataViewer):
            widget = widget.get_current_widget()

        if action == "plot":
            # Only curves can be plotted; an image has its own viewer.
            if not isinstance(widget, PlotWidget1DEnhanced):
                QMessageBox.information(
                    self,
                    "Cannot Plot",
                    "This dataset is not shown as a curve, so there is nothing to plot.",
                )
                return
            widget.open_plot()
            return

        if not isinstance(widget, (PlotWidget1DEnhanced, ImageView2DEnhanced)):
            QMessageBox.information(
                self,
                "Cannot Save",
                "This dataset is not shown as a curve or an image, so there is "
                "nothing to save as displayed.\nUse Export for the full dialog.",
            )
            return

        widget.quick_export()

    def _finalize_dock(self, viewer):
        """Dock a viewer widget and ensure the window is wide enough."""
        self.dock_plot.setWidget(viewer)

        recommended_width = 1200
        recommended_dock_width = 850
        if self.width() < recommended_width:
            self.resize(recommended_width, self.height())
        self.resizeDocks([self.dock_plot], [recommended_dock_width], Qt.Orientation.Horizontal)

        # A queued tree action waits here for the data it is meant to act on.
        self._run_pending_tree_action()

    # ----- Drag & Drop ----- #
    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        """Accept Drag Events for HDF5 files to initiate Drag & Drop Events."""
        if event is None:
            return
        if (mime_data := event.mimeData()) is None:
            return

        # Parse dropped files
        files = []
        for file in mime_data.text().split("\n"):
            if len(file) == 0:
                continue
            # Remove file:// prefix
            if sys.platform == "win32":
                file_path = file[8:] if file.startswith("file:///") else file
            else:
                file_path = file.removeprefix("file:")
            files.append(file_path.strip())

        # Accept if at least one file is supported.
        has_valid_file = False
        for file_path in files:
            if file_path and pathlib.Path(file_path).exists():
                if is_supported_data_file(file_path):
                    has_valid_file = True
                    break

        if has_valid_file:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Open Files that are dropped into Window."""
        if event is None:
            return
        if (mime_data := event.mimeData()) is None:
            return
        for file in mime_data.text().split("\n"):
            if len(file) == 0:
                continue
            if sys.platform == "win32":
                file = file[8:] if file.startswith("file:///") else file
            else:
                file = file.removeprefix("file:")
            file_path = pathlib.Path(file.strip())

            if file_path.exists() and is_supported_data_file(file_path):
                self._open_file(file_path)
            else:
                logging.warning(f"Skipped unsupported file: '{file_path}'")
        event.acceptProposedAction()

    # ----- Slots ----- #
    @pyqtSlot(str)
    def _handle_plot_type_changed(self, plot_type: str) -> None:
        """Update plot when new plot type is selected."""
        self._request_plot_data(plot_type)

    @pyqtSlot(QModelIndex)
    def _handle_item_changed(self, index: None | QModelIndex) -> None:
        """Update Info of currently selected Item."""
        if index is None:
            return

        # Always use column 0 regardless of which column the user clicked.
        # Clicking on the Type column (e.g. "uint32") or Shape column would
        # otherwise place those strings into the HDF5 path, causing a KeyError.
        file_path, names = self._tree_address(index)
        if not file_path:
            return
        path = "".join("/" + name for name in names)
        self.cur_file = pathlib.Path(file_path)
        self.cur_obj_path = path

        if not names:
            self.table_model_dataset.resetData()
            self.table_model_dataset.appendRow(["Name", file_path])
            self.table_model_dataset.appendRow(["File Size", file_size_to_str(file_path)])
            return

        if not is_hdf5_file(file_path):
            self.table_model_dataset.resetData()
            self.table_model_dataset.appendRow(["Name", pathlib.Path(file_path).name])
            self.table_model_dataset.appendRow(["File", file_path])
            self.table_model_dataset.appendRow(["Type", _regular_file_kind(file_path)])
            self.table_model_dataset.appendRow(["Data", "loaded from file"])
            self._request_plot_data(self.cb_plot_type.currentText())
            return

        try:
            with h5py.File(file_path, "r") as file:
                h5_obj = file[path]

                if isinstance(h5_obj, h5py.Group):
                    self.table_model_dataset.resetData()
                    self.table_model_dataset.appendRow(["Name", str(h5_obj.name)])

                elif isinstance(h5_obj, h5py.Dataset):
                    self.table_model_dataset.resetData()
                    self.table_model_dataset.appendRow(["Name", str(h5_obj.name)])
                    self.table_model_dataset.appendRow(["Data", f"shape {h5_obj.shape} of type {h5_obj.dtype}"])

                    for attribute, value in h5_obj.attrs.items():
                        self.table_model_dataset.appendRow([attribute, str(value)])
        except Exception as e:
            logging.warning(f"_handle_item_changed: could not open '{path}': {e}")
            return

        self._request_plot_data(self.cb_plot_type.currentText())

    def _request_plot_data(self, plot_type: str = "") -> None:
        """Debounced request to load/display current selection."""
        self._pending_plot_type = plot_type or self.cb_plot_type.currentText() or "Auto"
        self._plot_debounce_timer.start()

    def _plot_data_debounced(self) -> None:
        """Execute the latest pending plot request."""
        self._plot_data(self._pending_plot_type)

    def _tree_recursion(self, item: QModelIndex, path: list[str]) -> None:
        """Get Array of all Parents."""
        if (data := item.parent().data()) is None:
            return
        path.append(data)
        self._tree_recursion(item.parent(), path)

    @staticmethod
    def _tree_address(index: QModelIndex) -> tuple[str, list[str]]:
        """Split a tree row into the file it lives in and the HDF5 names below it.

        The two halves come from different places on purpose: the file from the
        stored path, the names from the row labels, which *are* the HDF5 names.
        Taking the file from a label as well is what made the first column
        unable to show anything but an absolute path.

        :return: ``(file_path, names)``; ``names`` is empty on a file row.
        """
        col0 = index.sibling(index.row(), TREE_COLUMN_NAME)
        names: list[str] = []
        node = col0
        while node.isValid() and node.data(_ROLE_NODE_TYPE) != "file":
            if node.data():
                names.append(str(node.data()))
            node = node.parent()
        names.reverse()
        return tree_file_path(col0), names

    @pyqtSlot()
    def _batch_path_drag_enter(self, event: QDragEnterEvent | None) -> None:
        """Accept drag enter events for batch path."""
        if event is None:
            return
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def _sync_batch_path_template_from_visible_text(self, text: str) -> None:
        """Keep hidden scan prefix in the full batch template after manual edits."""
        visible_path = str(text or "").strip().strip("/")
        if not visible_path:
            self._batch_path_template = None
            return

        if ";" in visible_path:
            # Several Y datasets: take them literally, no per-scan group hiding.
            self._batch_path_hidden_prefix = None
            self._batch_path_template = visible_path
            self.le_batch_path.setToolTip(
                f"{len(self._batch_paths_for_operations())} Y dataset(s)"
            )
            return

        prefix = (self._batch_path_hidden_prefix or "").strip("/")
        if prefix and not visible_path.startswith(f"{prefix}/") and visible_path != prefix:
            self._batch_path_template = f"{prefix}/{visible_path}"
        else:
            self._batch_path_template = visible_path
        self.le_batch_path.setToolTip(
            f"Display path: {visible_path}\nFull batch template: {self._batch_path_template}"
        )

    def _batch_path_for_operations(self) -> str:
        """Return full batch path template when available, otherwise visible text."""
        return (self._batch_path_template or self.le_batch_path.text()).strip()

    def _tool_is_open(self, attr: str) -> bool:
        """True when a tool window has been created and is currently visible.

        The tools are created lazily and kept alive when closed, so "open" needs
        all three checks; this idiom was repeated at seven call sites.
        """
        tool = getattr(self, attr, None)
        return tool is not None and tool.isVisible()

    def _batch_paths_for_operations(self) -> list[str]:
        """Split the batch address field into one or more Y dataset paths.

        Several datasets can be dropped at once; they are stored separated by
        ``;``. Order is preserved and duplicates dropped, so the export column
        order matches what the user dragged in.
        """
        raw = self._batch_path_for_operations()
        seen: set[str] = set()
        paths: list[str] = []
        for chunk in re.split(r"[;\n]", raw):
            path = chunk.strip().strip("/")
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _batch_path_display_text(self, dataset_path: str, file_path: str = "") -> str:
        """Hide the leading per-scan group from a dropped dataset path."""
        parts = [p for p in str(dataset_path).strip("/").split("/") if p]
        if len(parts) <= 1:
            self._batch_path_hidden_prefix = None
            return str(dataset_path).strip("/")

        first = parts[0]
        file_stem = pathlib.Path(file_path).stem if file_path else ""
        prefix = self._batch_keywords_text()
        is_scan_group = bool(re.fullmatch(r"scan[A-Za-z_]*\d+", first))
        if (file_stem and first == file_stem) or (prefix and first.startswith(prefix)) or is_scan_group:
            self._batch_path_hidden_prefix = first
            return "/".join(parts[1:])
        self._batch_path_hidden_prefix = None
        return "/".join(parts)

    # ----- The two batch fields ----- #

    def _batch_keywords_text(self) -> str:
        return self.le_batch_keywords.text().strip()

    def _batch_range_text(self) -> str:
        return self.le_scan_range.text().strip()

    def _dropped_scan_stems(self, event: QDropEvent | None) -> list[str] | None:
        """The filename stems a drop carries, or ``None`` if it cannot be used.

        Shared by both batch fields. Each of them fills only itself — dropping
        on the keywords never touches the numbers and the other way round — so
        that a drop cannot quietly widen a filter that was typed by hand.
        """
        if event is None or not event.mimeData().hasText():
            return None
        event.acceptProposedAction()

        stems: list[str] = []
        folders: set[str] = set()
        for line in event.mimeData().text().splitlines():
            token = line.strip()
            if not token:
                continue
            # A dataset drag carries "<file>::<dataset>"; only the file matters.
            path = pathlib.Path(token.split("::", 1)[0])
            stems.append(path.stem)
            folders.add(str(path.parent))

        if not stems:
            return None
        if len(folders) > 1:
            QMessageBox.warning(
                self,
                "Files In Several Folders",
                "Those files are in different folders, and a batch has to stay "
                "in one.\nDrag the files of a single folder.",
            )
            return None
        return stems

    def _batch_keywords_drag_enter(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData().hasText():
            event.acceptProposedAction()

    def _batch_keywords_drop(self, event: QDropEvent | None) -> None:
        """Fill the keywords from files dragged out of the tree, and only those."""
        stems = self._dropped_scan_stems(event)
        if not stems:
            return

        keyword = common_keyword(stems)
        if not keyword:
            self._set_status_text("Those files have nothing in common to filter on.")
            return
        self.le_batch_keywords.setText(keyword)
        self._set_status_text(f"Keywords set from {len(stems)} file(s).")

    def _batch_scan_range_drag_enter(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData().hasText():
            event.acceptProposedAction()

    def _batch_scan_range_drop(self, event: QDropEvent | None) -> None:
        """Fill the scan numbers from dragged files, and only those."""
        stems = self._dropped_scan_stems(event)
        if not stems:
            return

        parts = [scan_stem_parts(stem) for stem in stems]
        numbers = [p[1] for p in parts if p is not None]
        if not numbers:
            self._set_status_text("Dropped files have no scan number in their name.")
            return

        self.le_scan_range.setText(compress_scan_numbers(numbers))
        skipped = len(parts) - len(numbers)
        note = f", {skipped} without a scan number" if skipped else ""
        self._set_status_text(f"Scan numbers set from {len(numbers)} file(s){note}.")

    def _batch_path_drop(self, event: QDropEvent | None) -> None:
        """Handle drop events for batch path."""
        if event is None:
            return
        if not event.mimeData().hasText():
            return

        # A multi-selection drag arrives as one "<file>::<dataset>" token per line.
        tokens = [line.strip() for line in event.mimeData().text().splitlines() if line.strip()]
        if not tokens:
            return

        file_path = ""
        dataset_paths: list[str] = []
        for token in tokens:
            if "::" in token:
                token_file, dataset_path = token.split("::", 1)
                file_path = file_path or token_file
            else:
                dataset_path = token
            dataset_path = dataset_path.strip("/")
            if dataset_path and dataset_path not in dataset_paths:
                dataset_paths.append(dataset_path)
        if not dataset_paths:
            return

        if len(dataset_paths) == 1:
            self._batch_path_template = dataset_paths[0]
            display_text = self._batch_path_display_text(self._batch_path_template, file_path=file_path)
        else:
            # Hiding the per-scan group only makes sense for a single path; with
            # several, show them in full so the user can see what will be exported.
            self._batch_path_hidden_prefix = None
            self._batch_path_template = BATCH_PATH_SEPARATOR.join(dataset_paths)
            display_text = self._batch_path_template

        self.le_batch_path.setText(display_text)
        self.le_batch_path.setToolTip(
            f"{len(dataset_paths)} Y dataset(s):\n" + "\n".join(f"  {p}" for p in dataset_paths)
        )
        event.acceptProposedAction()
        logging.info("Batch path set to %d dataset(s): %s", len(dataset_paths), self._batch_path_template)

    def _batch_add_to_tool(self, tool: str) -> None:
        """
        Batch add datasets from multiple files to comparison or calculator tool.

        :param tool: Target tool - "comparison", "calculator_a", or "calculator_b"
        """
        # Get file prefix
        file_prefix = self._batch_keywords_text()
        if not file_prefix:
            QMessageBox.warning(self, "No File Prefix", "Please enter file name prefix (e.g., scanx_ or scan_)")
            return

        # Get scan range
        scan_range_text = self._batch_range_text()
        if not scan_range_text:
            QMessageBox.warning(self, "No Scan Range", "Please enter a scan number range (e.g., 0080-0085)")
            return

        # Get batch path(s) — several datasets can be dropped at once
        batch_paths = self._batch_paths_for_operations()
        if not batch_paths:
            QMessageBox.warning(self, "No Path", "Please drag a dataset to set the batch path")
            return

        # Parse scan range
        scan_numbers = self._parse_scan_range(scan_range_text)
        if not scan_numbers:
            QMessageBox.warning(
                self, "Invalid Range",
                "Invalid scan range format.\nUse:\n- Range: 0080-0085\n- List: 0080,0085,0027"
            )
            return

        # Find matching files
        matching_files = []
        for file_path in self.opened_files:
            filename = file_path.stem  # Get filename without extension
            # Check if filename starts with prefix and contains scan number
            if filename.startswith(file_prefix):
                # Extract the part after prefix
                suffix = filename[len(file_prefix):]
                # Check if any scan number matches
                for scan_num in scan_numbers:
                    if suffix.startswith(scan_num):
                        matching_files.append((file_path, scan_num))
                        logging.info(f"Matched file: {file_path.name} (scan: {scan_num})")
                        break

        if not matching_files:
            QMessageBox.warning(
                self, "No Matches",
                f"No open files found matching:\n"
                f"Prefix: {file_prefix}\n"
                f"Scan numbers: {', '.join(scan_numbers)}\n\n"
                f"Example expected filename: {file_prefix}{scan_numbers[0]}.nxs"
            )
            return

        # Same rule as the export: a batch belongs to one measurement folder.
        if conflict := batch_folder_conflict(matching_files):
            QMessageBox.warning(self, "Files In Several Folders", conflict)
            return

        # Nothing is overwritten when a number matches twice — the files keep
        # their own names — but the batch then holds twice what was asked for,
        # so it is confirmed rather than assumed. There is no settings dialog
        # on this path to carry the warning instead.
        if not self._confirm_number_ambiguity(matching_files, "Add"):
            return

        # Check if target tool is open, auto-open if not
        if tool == "comparison":
            if not self._tool_is_open("comparison_tool"):
                logging.info("Auto-opening Comparison Tool")
                self._handle_action_comparison()
        elif tool in ("fth_cl", "fth_cr", "fth_dark"):
            if not self._tool_is_open("fth_tool"):
                logging.info("Auto-opening FTH Tool")
                self._handle_action_fth()
        else:  # calculator_a or calculator_b
            if not self._tool_is_open("calculator"):
                logging.info("Auto-opening Data Calculator")
                self._handle_action_calculator()

        # Add datasets
        added_count = 0
        failed_count = 0
        error_details = []

        targets = build_batch_targets(matching_files, batch_paths)
        logging.info(
            "Starting batch add: %d file(s) x %d dataset(s) = %d",
            len(matching_files), len(batch_paths), len(targets),
        )

        for target in targets:
            file_path, adjusted_path = target.file_path, target.ds_path
            try:
                logging.info(f"Checking path '{adjusted_path}' in {file_path.name}")

                # Open once per file for existence and (comparison) shape checks.
                loaded_data = None
                with h5py.File(file_path, "r") as f:
                    if adjusted_path not in f:
                        msg = f"Path '{adjusted_path}' not found in {file_path.name}"
                        logging.warning(msg)
                        error_details.append(msg)
                        failed_count += 1
                        continue
                    if tool == "comparison":
                        try:
                            ds_obj = f[adjusted_path]
                            shape = tuple(getattr(ds_obj, "shape", ()))
                            if len(shape) == 2 and int(shape[1]) >= 100:
                                msg = (
                                    f"Skipped {file_path.name}::{adjusted_path} "
                                    f"(columns={shape[1]} >= 100)"
                                )
                                logging.warning(msg)
                                error_details.append(msg)
                                failed_count += 1
                                continue
                            # Stage-2 optimization: read once here and pass payload directly
                            # to comparison tool to avoid reopening the same file/path.
                            loaded_data = np.asarray(ds_obj[()])
                        except Exception as e:
                            msg = f"Failed to inspect shape for {file_path.name}::{adjusted_path}: {e}"
                            logging.warning(msg)
                            error_details.append(msg)
                            failed_count += 1
                            continue

                # Always use absolute path so all tools can locate the file unambiguously,
                # and _try_select_by_text can match on the exact file_token.
                full_path = f"{str(file_path)}::{adjusted_path}"
                logging.info(f"Adding: {full_path}")

                # Add to tool
                if tool == "comparison":
                    if loaded_data is None:
                        # Fallback to path-based load if payload wasn't captured.
                        self.comparison_tool.add_dataset_from_path(full_path)
                    else:
                        self.comparison_tool.add_dataset_from_loaded_path(full_path, loaded_data)
                    logging.info(f"Added to comparison tool: {full_path}")
                elif tool == "calculator_a":
                    self.calculator.add_to_dataset_a(full_path)
                    logging.info(f"Added to calculator A: {full_path}")
                elif tool == "calculator_b":
                    self.calculator.add_to_dataset_b(full_path)
                    logging.info(f"Added to calculator B: {full_path}")
                elif tool == "fth_cl":
                    self.fth_tool.add_dataset_to_combo(full_path, "CL")
                    logging.info(f"Added to FTH CL: {full_path}")
                elif tool == "fth_cr":
                    self.fth_tool.add_dataset_to_combo(full_path, "CR")
                    logging.info(f"Added to FTH CR: {full_path}")
                else:  # fth_dark
                    self.fth_tool.add_dataset_to_combo(full_path, "Dark")
                    logging.info(f"Added to FTH Dark: {full_path}")

                added_count += 1

            except Exception as e:
                msg = f"Failed to add {file_path.name}: {e}"
                logging.error(msg)
                error_details.append(msg)
                failed_count += 1

        # Show result - only show message box if there were errors
        logging.info(f"Batch add complete: {added_count} added, {failed_count} failed")

        if failed_count > 0:
            # Only show message box for errors
            result_msg = "Batch add completed with errors:\n\n"
            result_msg += f"Successfully added: {added_count}\n"
            result_msg += f"Failed: {failed_count}\n"
            if error_details:
                result_msg += "\nErrors:\n" + "\n".join(error_details[:3])  # Show first 3 errors
            QMessageBox.warning(self, "Batch Add - Partial Success", result_msg)
        # If all succeeded, no message box (user can see the data in the tool)

    def _parse_scan_range(self, range_text: str) -> list[str]:
        """
        Parse scan range text into list of scan number strings.

        Supports:
        - Range: "0080-0085" -> ["0080", "0081", "0082", "0083", "0084", "0085"]
        - List: "0080,0085,0027" -> ["0080", "0085", "0027"]
        - Mixed ranges/list: "0001,0003,0005-0007" -> ["0001", "0003", "0005", "0006", "0007"]

        :param range_text: Scan range text
        :return: List of scan number strings
        """
        scan_numbers: list[str] = []
        seen: set[str] = set()

        try:
            normalized = str(range_text or "").replace("，", ",").strip()
            if not normalized:
                return []

            for token in [p.strip() for p in normalized.split(",") if p.strip()]:
                if "-" in token:
                    parts = [p.strip() for p in token.split("-")]
                    if len(parts) != 2 or not parts[0] or not parts[1]:
                        return []
                    start_str, end_str = parts
                    width = max(len(start_str), len(end_str))
                    start = int(start_str)
                    end = int(end_str)
                    if start > end:
                        return []
                    expanded = (str(num).zfill(width) for num in range(start, end + 1))
                else:
                    expanded = (token,)

                for scan_num in expanded:
                    if scan_num and scan_num not in seen:
                        scan_numbers.append(scan_num)
                        seen.add(scan_num)

        except ValueError as e:
            logging.error(f"Failed to parse scan range '{range_text}': {e}")
            return []

        return scan_numbers

    def _matching_files_for_scans(
        self,
        keyword_text: str,
        scan_numbers: list[str],
    ) -> list[tuple[pathlib.Path, str]]:
        """Opened files whose name carries every keyword and one of the numbers.

        The keywords match as substrings and the number as a whole part of the
        name; see :func:`stem_matches_keywords` and :func:`scan_number_in_stem`
        for why the two are matched differently. The old rule wanted the number
        immediately after the prefix, which no name of the form
        ``Scan_ECL_5p0uJIR_050`` can satisfy — those files matched nothing at all.
        """
        keywords = parse_keywords(keyword_text)
        matching_files: list[tuple[pathlib.Path, str]] = []
        for file_path in self.opened_files:
            stem = file_path.stem
            if not stem_matches_keywords(stem, keywords):
                continue
            scan_num = scan_number_in_stem(stem, scan_numbers)
            if scan_num is not None:
                matching_files.append((file_path, scan_num))
        return matching_files

    def _resolve_batch_selection(
        self,
        quiet: bool = False,
    ) -> tuple[list[tuple[pathlib.Path, str]], list[str], list[str]] | None:
        """Validate the three batch inputs and resolve them to matching files.

        :param quiet: when true (Enter-to-preview), report problems on the status
            bar instead of a modal box — the user is probably still typing.
        :return: ``(matching_files, batch_paths, scan_numbers)`` or ``None``.
        """
        def _complain(title: str, message: str) -> None:
            if quiet:
                self._set_status_text(message.splitlines()[0])
            else:
                QMessageBox.warning(self, title, message)

        file_prefix = self._batch_keywords_text()
        if not file_prefix:
            _complain("No File Prefix", "Please enter file name prefix (e.g., scanx_ or scan_)")
            return None

        scan_range_text = self._batch_range_text()
        if not scan_range_text:
            _complain("No Scan Range", "Please enter a scan number range (e.g., 0080-0085)")
            return None

        batch_paths = self._batch_paths_for_operations()
        if not batch_paths:
            _complain("No Path", "Please drag or type a dataset path before exporting.")
            return None

        scan_numbers = self._parse_scan_range(scan_range_text)
        if not scan_numbers:
            _complain(
                "Invalid Range",
                "Invalid scan range format.\nUse:\n- Range: 0080-0085\n- List: 0080,0085,0027",
            )
            return None

        matching_files = self._matching_files_for_scans(file_prefix, scan_numbers)
        if not matching_files:
            _complain(
                "No Matches",
                f"No open files found matching:\nPrefix: {file_prefix}\nScan numbers: {', '.join(scan_numbers)}",
            )
            return None

        if conflict := batch_folder_conflict(matching_files):
            _complain("Files In Several Folders", conflict)
            return None

        # A number matching several files is allowed — they are written under
        # their own names — but it is said out loud. Here that is the status
        # bar and, for the export itself, a line in the settings dialog whose
        # own button is the confirmation; never a modal, because this runs on
        # every Enter while the fields are still being typed.
        #
        # Set either way: a warning left over from the previous, wider keyword
        # would say this batch is ambiguous when narrowing it has just fixed
        # that — worse than no warning at all.
        self._set_status_text(summarise_number_ambiguity(batch_number_ambiguity(matching_files)))

        return matching_files, batch_paths, scan_numbers

    def _confirm_number_ambiguity(
        self,
        matching_files: list[tuple[pathlib.Path, str]],
        action: str,
    ) -> bool:
        """Ask before going ahead with a batch holding a number twice.

        For the paths that have no settings dialog to put a warning line in.
        """
        ambiguity = batch_number_ambiguity(matching_files)
        if not ambiguity:
            return True
        answer = QMessageBox.question(
            self,
            "Some Scan Numbers Match Several Files",
            describe_number_ambiguity(ambiguity) + f"\n\n{action} all of them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _handle_batch_preview(self) -> None:
        """Show the first matched scan in the data view (Enter in a batch input).

        Enter never opens the export dialog — that is the Export button's job.
        With a range or a list, the first matched scan is what gets shown.
        """
        resolved = self._resolve_batch_selection(quiet=True)
        if resolved is None:
            return
        matching_files, batch_paths, scan_numbers = resolved

        # Preview the first scan number the user typed. matching_files is ordered
        # by opened file, not by the typed order, so "0082,0080" must not show 0080.
        by_scan = {scan: path for path, scan in reversed(matching_files)}
        preview_scan = next((s for s in scan_numbers if s in by_scan), matching_files[0][1])
        preview_file = by_scan.get(preview_scan, matching_files[0][0])
        preview_path = "/" + adjust_batch_path_for_scan(batch_paths[0], preview_scan).strip("/")

        self.cur_file = preview_file
        self.cur_obj_path = preview_path
        self._request_plot_data(self.cb_plot_type.currentText())

        self.table_model_dataset.resetData()
        self.table_model_dataset.appendRow(["Name", preview_path])
        self.table_model_dataset.appendRow(["File", preview_file.name])
        self.table_model_dataset.appendRow(["Scan", preview_scan])
        self.table_model_dataset.appendRow(["Matched", f"{len(matching_files)} of {len(scan_numbers)} scans"])

        if len(matching_files) > 1:
            self._set_status_text(
                f"Preview: {preview_file.name}::{preview_path} "
                f"(first of {len(matching_files)} matched scans)"
            )
        else:
            self._set_status_text(f"Preview: {preview_file.name}::{preview_path}")
        logging.info("Batch preview: %s::%s", preview_file.name, preview_path)

    def _handle_batch_export(self) -> None:
        """Open export settings and export one or more matched datasets."""
        resolved = self._resolve_batch_selection(quiet=False)
        if resolved is None:
            return
        matching_files, batch_paths, scan_numbers = resolved
        targets = build_batch_targets(matching_files, batch_paths)

        shown_path = (
            batch_paths[0] if len(batch_paths) == 1 else f"{len(batch_paths)} datasets: " + ", ".join(batch_paths)
        )
        self._open_export_dialog(targets, matching_files, scan_numbers, shown_path)

    def _open_export_dialog(
        self,
        targets: list[BatchTarget],
        matching_files: list[tuple[pathlib.Path, str]],
        scan_numbers: list[str],
        shown_path: str,
        sample_data: np.ndarray | None = None,
    ) -> None:
        """Show the full-export settings dialog for one or more targets.

        Shared by the batch controls and the menu bar's ``Export``: a single
        dataset is just a one-element target list.

        :param sample_data: the array to preview; read from ``targets[0]`` when
            omitted (the menu path already holds it, and may come from a
            non-HDF5 file that cannot be re-read with h5py).
        """
        settings = QSettings()
        last_dir = last_save_directory()
        sample = targets[0]

        if sample_data is None:
            try:
                with h5py.File(sample.file_path, "r") as f:
                    if sample.ds_path not in f:
                        raise KeyError(f"Dataset path not found: {sample.ds_path}")
                    sample_data = np.asarray(f[sample.ds_path][()])
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Preview Failed",
                    f"Cannot read preview dataset:\n{sample.file_path.name}::{sample.ds_path}\n\n{exc}",
                )
                return

        data_kind = "curve" if is_curve_export_data(sample_data) else "image"

        def _preview_x_loader(export_settings: dict[str, Any], expected_len: int) -> np.ndarray | None:
            return read_batch_x_data(
                export_settings=export_settings,
                current_file=sample.file_path,
                current_scan=sample.scan_num,
                matching_files=matching_files,
                expected_len=expected_len,
            )

        def _preview_curve_loader(
            export_settings: dict[str, Any],
            **caps: Any,
        ) -> tuple[np.ndarray, list[str]]:
            return build_curve_preview_table(targets, matching_files, export_settings, **caps)

        dialog = BatchExportDialog(
            # Parented, so it stays above the main window while the tree is
            # still being used — the reason it is non-modal in the first place.
            self,
            default_dir=last_dir if last_dir.exists() else pathlib.Path.home(),
            scan_numbers=scan_numbers,
            dataset_path=shown_path,
            sample_data=sample_data,
            data_kind=data_kind,
            preview_x_loader=_preview_x_loader,
            preview_curve_loader=_preview_curve_loader if data_kind == "curve" else None,
            default_x_path=self._x_dataset_path(),
            # Carried into the dialog rather than raised on the way in: the
            # dialog's own Export button is the confirmation.
            warning=summarise_number_ambiguity(batch_number_ambiguity(matching_files)),
            warning_detail=describe_number_ambiguity(batch_number_ambiguity(matching_files)),
            # Open on the frame the viewer is showing, not back at the first
            # one of the first axis.
            **self._displayed_slice_position(),
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)
        # export_requested, not accepted: writing a batch no longer closes the
        # settings, so a second range can be exported without entering it again.
        dialog.export_requested.connect(
            lambda d=dialog, tg=targets, mf=matching_files, dk=data_kind, s=settings: self._finish_batch_export(
                d,
                tg,
                mf,
                dk,
                s,
            )
        )
        dialog.rejected.connect(dialog.deleteLater)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._batch_export_dialog = dialog

    def _plot_batch_selection(
        self,
        dialog: BatchExportDialog,
        targets: list[BatchTarget],
        matching_files: list[tuple[pathlib.Path, str]],
        export_settings: dict[str, Any],
    ) -> None:
        """Write the batch as image files instead of as tables.

        Every figure is rendered by the dialog's own panel, so the files carry
        the palette, log axes and labels that were on screen — the preview is
        the specification, not an approximation of it.
        """
        out_dir = pathlib.Path(export_settings["output_dir"])
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Plot Failed", f"Cannot create folder:\n{out_dir}\n\n{exc}")
            return

        suffix = "." + str(export_settings.get("plot_format", "PNG")).lower()
        if str(export_settings.get("plot_output_mode")) == PLOT_MODE_PER_SCAN:
            jobs = [
                (group, out_dir / f"{safe_export_name(group[0].file_path, group[0].ds_path)}{suffix}")
                for _scan, group in self._batch_targets_by_scan(targets)
            ]
        else:
            sample = targets[0]
            scans = sorted({t.scan_num for t in targets if t.scan_num})
            span = f"{scans[0]}-{scans[-1]}" if len(scans) > 1 else (scans[0] if scans else "batch")
            stem = safe_export_name(sample.file_path, sample.ds_path, suffix=f"plot_{span}")
            jobs = [(targets, out_dir / f"{stem}{suffix}")]

        written, failures, cancelled = self._write_batch_figures(
            dialog, jobs, matching_files, export_settings
        )
        self._report_batch_result(
            "Batch Plot Complete",
            f"Wrote {written} figure(s) to:\n{out_dir}",
            failures,
            cancelled,
        )

    def _write_batch_figures(
        self,
        dialog: BatchExportDialog,
        jobs: list[tuple[list[BatchTarget], pathlib.Path]],
        matching_files: list[tuple[pathlib.Path, str]],
        export_settings: dict[str, Any],
    ) -> tuple[int, list[str], bool]:
        """Render each job through the dialog's panel and save it.

        :return: ``(written, failures, cancelled)``.
        """
        from src.gui.plot_series import series_from_table
        from src.lib_h5.table_writer import columns_from_2d

        panel = dialog.plot_panel
        written = 0
        failures: list[str] = []
        cancelled = False

        with BatchProgress(dialog, len(jobs), "Batch Plot", "Rendering…") as progress:
            for group, path in jobs:
                try:
                    # Uncapped: this is the file, not a look at it.
                    preview, headers = build_curve_preview_table(
                        group, matching_files, export_settings, max_targets=None, max_rows=None
                    )
                    panel.set_series(
                        series_from_table(list(headers), columns_from_2d(np.asarray(preview)))
                    )
                    panel.figure.savefig(path, dpi=300)
                    written += 1
                except Exception as exc:
                    logging.error("Batch plot: %s failed: %s", path.name, exc)
                    failures.append(f"{path.name}: {exc}")
                if not progress.advance(path.name):
                    cancelled = True
                    break
        return written, failures, cancelled

    @staticmethod
    def _batch_targets_by_scan(targets: list[BatchTarget]) -> list[tuple[str, list[BatchTarget]]]:
        """Group targets by scan, keeping the order the range was given in."""
        grouped: dict[str, list[BatchTarget]] = {}
        for target in targets:
            grouped.setdefault(target.scan_num, []).append(target)
        return list(grouped.items())

    def _finish_batch_export(
        self,
        dialog: BatchExportDialog,
        targets: list[BatchTarget],
        matching_files: list[tuple[pathlib.Path, str]],
        data_kind: str,
        settings: QSettings,
    ) -> None:
        """Run whichever page of the settings dialog was accepted."""

        export_settings = dialog.settings()
        if dialog.action() == "plot":
            # The Plot page's own Output decides whether a figure holds the
            # whole range or one scan; reading the Export page's choice here is
            # what made a combined figure come out with only the first scan.
            self._plot_batch_selection(dialog, targets, matching_files, dialog.plot_settings())
            return

        output_dir = pathlib.Path(export_settings["output_dir"])
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", f"Cannot create export folder:\n{output_dir}\n\n{exc}")
            return
        remember_save_directory(output_dir)
        settings.setValue("export/table_format", table_format_from(export_settings).key)

        ok_count = 0
        fail_details: list[str] = []
        cancelled = False
        if data_kind == "curve" and str(export_settings.get("curve_output_mode")) == "Combined file":
            # One call writes the whole table, so there is nothing to count
            # through — the window says what is happening and nothing more.
            with BatchProgress(dialog, 0, "Batch Export", "Writing the combined table…"):
                ok_count, fail_details = export_curve_combined_table(
                    targets,
                    matching_files,
                    export_settings,
                )
        else:
            with BatchProgress(dialog, len(targets), "Batch Export", "Exporting…") as progress:
                for target in targets:
                    try:
                        with h5py.File(target.file_path, "r") as f:
                            if target.ds_path not in f:
                                raise KeyError(f"Dataset path not found: {target.ds_path}")
                            data = np.asarray(f[target.ds_path][()])

                        if is_curve_export_data(data):
                            export_curve_dataset(
                                target.file_path,
                                target.scan_num,
                                target.ds_path,
                                data,
                                export_settings,
                                matching_files,
                            )
                        elif data.ndim >= 2:
                            export_image_dataset(
                                target.file_path, target.ds_path, data, export_settings
                            )
                        else:
                            raise ValueError(f"Unsupported dataset shape: {data.shape}")
                        ok_count += 1
                    except Exception as exc:
                        msg = f"{target.file_path.name}::{target.ds_path}: {exc}"
                        logging.error("Batch export failed for %s", msg)
                        fail_details.append(msg)
                    if not progress.advance(f"{target.file_path.name}"):
                        cancelled = True
                        break

        self._report_batch_result(
            "Batch Export Complete",
            f"Exported {ok_count} dataset(s) to:\n{output_dir}",
            fail_details,
            cancelled,
        )

    def _report_batch_result(
        self,
        title: str,
        message: str,
        failures: list[str],
        cancelled: bool = False,
    ) -> None:
        """Say what a finished batch wrote, and what it did not.

        The settings dialog stays open behind this, so the count is the only
        sign the run happened at all.
        """
        if cancelled:
            message = "Stopped at your request.\n\n" + message
        if failures:
            message += f"\n\nFailed: {len(failures)}\n" + "\n".join(failures[:5])
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def _batch_add_to_calculator_ab(self) -> None:
        """
        Batch add two datasets to Calculator A and B.
        Expects exactly two scan numbers (e.g., "0072,0073" or "0072-0073").
        First scan goes to A, second goes to B.
        """
        # Get scan range
        scan_range_text = self._batch_range_text()
        if not scan_range_text:
            QMessageBox.warning(
                self, "No Scan Numbers",
                "Please enter exactly two scan numbers.\n\n"
                "Examples:\n- 0072,0073\n- 0072-0073"
            )
            return

        # Parse scan range
        scan_numbers = self._parse_scan_range(scan_range_text)

        # Validate exactly 2 scan numbers
        if len(scan_numbers) != 2:
            QMessageBox.warning(
                self, "Invalid Scan Count",
                f"Expected exactly 2 scan numbers, but got {len(scan_numbers)}.\n\n"
                "Examples:\n- 0072,0073\n- 0072-0073"
            )
            return

        logging.info(f"Calculator A & B: Adding {scan_numbers[0]} to A and {scan_numbers[1]} to B")

        # Temporarily narrow the numbers to one scan at a time, then put them
        # back. Only that field moves; the keywords say which family both scans
        # come from and are the same for each half.
        original_range = self.le_scan_range.text()

        try:
            # Add first scan to Calculator A
            self.le_scan_range.setText(scan_numbers[0])
            self._batch_add_to_tool("calculator_a")

            # Add second scan to Calculator B
            self.le_scan_range.setText(scan_numbers[1])
            self._batch_add_to_tool("calculator_b")

        finally:
            self.le_scan_range.setText(original_range)

    def _batch_browse_files(self) -> None:
        """Browse matching files in the tree view."""
        logging.info("Browse button clicked - starting _batch_browse_files")

        # Get file prefix
        file_prefix = self._batch_keywords_text()
        logging.info(f"Browse: file_prefix='{file_prefix}'")
        if not file_prefix:
            QMessageBox.warning(self, "No File Prefix", "Please enter file name prefix (e.g., scanx_ or scan_)")
            return

        # Get scan range
        scan_range_text = self._batch_range_text()
        if not scan_range_text:
            QMessageBox.warning(self, "No Scan Range", "Please enter a scan number range (e.g., 0080-0085)")
            return

        # Parse scan range
        scan_numbers = self._parse_scan_range(scan_range_text)
        if not scan_numbers:
            QMessageBox.warning(
                self, "Invalid Range",
                "Invalid scan range format.\nUse:\n- Range: 0080-0085\n- List: 0080,0085,0027"
            )
            return

        matching_files = self._matching_files_for_scans(file_prefix, scan_numbers)

        if not matching_files:
            QMessageBox.warning(
                self, "No Matches",
                f"No open files found matching:\n"
                f"Prefix: {file_prefix}\n"
                f"Scan numbers: {', '.join(scan_numbers)}"
            )
            return

        # Same rule as the export: a batch belongs to one measurement folder.
        if conflict := batch_folder_conflict(matching_files):
            QMessageBox.warning(self, "Files In Several Folders", conflict)
            return

        if not self._confirm_number_ambiguity(matching_files, "Browse"):
            return

        # Expand matching files in tree view. Only one node can be selected, so
        # with several Y datasets dropped in, browse to the first.
        batch_paths = self._batch_paths_for_operations()
        batch_path = batch_paths[0] if batch_paths else ""
        first_match_index = None

        logging.info(f"Browse: Starting with batch_path='{batch_path}', {len(matching_files)} files to process")

        root = self.tree_model_file.invisibleRootItem()
        logging.info(f"Browse: Tree has {root.rowCount()} root items")

        for row in range(root.rowCount()):
            file_item = root.child(row)
            if file_item is None:
                continue

            # From the stored path: the label is a filename now, which happens
            # to be what this compares on, but the two must not be the same
            # thing by accident.
            filename = pathlib.Path(tree_item_file_path(file_item)).name
            logging.info(f"Browse: Checking tree item (filename: '{filename}')")

            # Check if this file matches
            for file_path, scan_num in matching_files:
                logging.info(f"Browse: Comparing '{filename}' with '{file_path.name}'")
                if file_path.name == filename:
                    # Don't expand tree - just find the dataset
                    # If we have a batch path, try to find and select it
                    if batch_path and first_match_index is None:
                        logging.info(f"Browse: Attempting to find dataset in '{filename}'")
                        # Replace scan number in path
                        import re
                        adjusted_path = batch_path
                        scan_pattern = re.compile(r'\d{4}')
                        matches = scan_pattern.findall(adjusted_path)
                        if matches:
                            old_scan = matches[0]
                            adjusted_path = adjusted_path.replace(old_scan, scan_num, 1)
                            logging.info(f"Browse: Adjusted path from '{batch_path}' to '{adjusted_path}'")
                        else:
                            logging.warning(f"Browse: No scan number found in path '{batch_path}'")

                        # Find the dataset in the tree
                        logging.info(f"Browse: Looking for path '{adjusted_path}' in file '{filename}'")
                        dataset_index = self._find_dataset_in_tree(file_item, adjusted_path)
                        if dataset_index is not None:
                            first_match_index = dataset_index
                            logging.info(f"Browse: Found dataset at index: {dataset_index.data()}")
                        else:
                            logging.warning(f"Browse: Dataset not found: {adjusted_path}")

                    break

        # Display first match without touching the tree view
        if first_match_index is not None:
            proxy_index = self.tree_model_file_proxy.mapFromSource(first_match_index)
            # Only trigger display, don't select or scroll in tree view
            self._handle_item_changed(proxy_index)
            logging.info(f"Browsed to: {first_match_index.data()}")
        else:
            # No specific dataset found
            logging.info(f"Found {len(matching_files)} matching file(s), but no dataset path specified")

    def _find_dataset_in_tree(self, file_item: QStandardItem, dataset_path: str) -> QModelIndex | None:
        """Find a dataset in the tree by path."""
        # Split path into parts
        parts = dataset_path.strip('/').split('/')
        logging.info(f"_find_dataset_in_tree: Looking for path parts: {parts}")

        current_item = file_item
        for i, part in enumerate(parts):
            # Ensure current level is loaded in lazy mode
            self._load_tree_children(current_item)

            # Search for this part in children
            found = False
            logging.info(f"_find_dataset_in_tree: Searching for part '{part}' (level {i})")

            # Log available children
            available_children = [
                current_item.child(row).text()
                for row in range(current_item.rowCount())
                if current_item.child(row)
            ]
            logging.info(f"_find_dataset_in_tree: Available children: {available_children[:10]}")  # Show first 10

            for row in range(current_item.rowCount()):
                child = current_item.child(row)
                if child and child.text() == part:
                    current_item = child
                    found = True
                    logging.info(f"_find_dataset_in_tree: Found '{part}'")
                    # Don't expand tree - just find the item
                    break

            if not found:
                logging.warning(f"_find_dataset_in_tree: Part '{part}' not found at level {i}")
                return None

        # Return the index of the final item
        logging.info("_find_dataset_in_tree: Successfully found dataset")
        return self.tree_model_file.indexFromItem(current_item)

    def _get_selected_file_source_rows(self) -> set:
        """Return source-model row numbers for all selected root-level (file) items."""
        sel = self.tree_view_file.selectionModel()
        if sel is None:
            return set()
        rows: set = set()
        for proxy_idx in sel.selectedRows(0):
            if not proxy_idx.parent().isValid():   # root level = file
                src_idx = self.tree_model_file_proxy.mapToSource(proxy_idx)
                rows.add(src_idx.row())
        return rows

    @pyqtSlot(QPoint)
    def _handle_tree_menu(self, pos: QPoint) -> None:
        """Handle right-click context menu on tree view."""
        menu = QMenu(self)
        index = self.tree_view_file.indexAt(pos)

        # Always normalise to column 0 so dtype / shape text is never used as a path
        if index.isValid():
            index = index.sibling(index.row(), 0)

        # If clicking on a file (root level)
        if index.isValid() and not index.parent().isValid():
            # Collect all selected root-level rows; if the right-clicked item is
            # outside the current selection, use only that item.
            selected_rows = self._get_selected_file_source_rows()
            src_row = self.tree_model_file_proxy.mapToSource(index).row()
            if src_row not in selected_rows:
                selected_rows = {src_row}

            n = len(selected_rows)
            label = f"Close {n} selected files" if n > 1 else "Close file"
            action_close = QAction(label, self)
            menu.addAction(action_close)

            def _close_files(checked: bool = False, rows: set = selected_rows) -> None:
                for row in sorted(rows, reverse=True):
                    item = self.tree_model_file.item(row, TREE_COLUMN_NAME)
                    if item is not None:
                        # The monitor holds absolute paths, which the label is
                        # no longer one of.
                        self._monitor_known.discard(tree_item_file_path(item))
                    self.tree_model_file.removeRow(row)

            action_close.triggered.connect(_close_files)

            if (viewport := self.tree_view_file.viewport()) is not None:
                menu.popup(viewport.mapToGlobal(pos))

        # If clicking on a dataset or group
        elif index.isValid():
            # Build full path for the dataset (used by both comparison and calculator).
            # Shaped as it always was — the absolute file path followed by the
            # HDF5 names — so everything below reads the same; only where the
            # file half comes from has changed.
            _file_path, _names = self._tree_address(index)
            parents_list = [_file_path, *_names] if _file_path else []
            if not parents_list:
                return

            # Quick export of a dataset node, straight from the tree.
            if len(parents_list) > 1 and index.data(_ROLE_NODE_TYPE) == "dataset":
                col0_index = index.sibling(index.row(), 0)

                action_quick_save = QAction("Export", self)
                action_quick_save.setToolTip("Show this dataset, then save it as displayed")
                action_quick_save.triggered.connect(
                    lambda _checked=False, idx=col0_index: self._quick_export_tree_dataset(idx)
                )
                menu.addAction(action_quick_save)

                action_plot = QAction("Plot", self)
                action_plot.setToolTip("Show this dataset, then draw it in a Plot window")
                action_plot.triggered.connect(
                    lambda _checked=False, idx=col0_index: self._plot_tree_dataset(idx)
                )
                menu.addAction(action_plot)

                action_set_x = QAction("Set X", self)
                action_set_x.setToolTip(
                    "Use this dataset as the X axis: applied to the displayed curve and "
                    "pre-filled in the export dialog, instead of dragging it in each time"
                )
                action_set_x.triggered.connect(
                    lambda _checked=False, keys=list(parents_list): self._set_dataset_as_x(keys)
                )
                menu.addAction(action_set_x)
                menu.addSeparator()

            # Check if comparison tool is open
            if self._tool_is_open("comparison_tool"):
                action_add_to_comparison = QAction("Add to Comparison...", self)
                menu.addAction(action_add_to_comparison)

                # Connect to handler
                def add_to_comparison() -> None:
                    if len(parents_list) > 1:
                        # Format: /full/path/to/filename.ext::path/to/dataset
                        filename = parents_list[0]   # full absolute path
                        dataset_path = "/".join(parents_list[1:])
                        full_path = f"{filename}::{dataset_path}"

                        # Restrict comparison input: for 2D datasets, columns must be < 100
                        try:
                            with h5py.File(filename, "r") as f:
                                if dataset_path in f and isinstance(f[dataset_path], h5py.Dataset):
                                    ds = f[dataset_path]
                                    shape = tuple(getattr(ds, "shape", ()))
                                    if len(shape) == 2 and int(shape[1]) >= 100:
                                        QMessageBox.warning(
                                            self,
                                            "Comparison Limit",
                                            f"This dataset has {shape[1]} columns (>=100).\n"
                                            "It cannot be added to Data Comparison.",
                                        )
                                        return
                        except Exception as e:
                            logging.warning(f"Failed to validate comparison dataset shape: {e}")

                        # Add to comparison tool
                        self.comparison_tool.add_dataset_from_path(full_path)

                action_add_to_comparison.triggered.connect(add_to_comparison)

            # Check if calculator tool is open
            if self._tool_is_open("calculator"):
                # Add separator if comparison menu was added
                if self._tool_is_open("comparison_tool"):
                    menu.addSeparator()

                action_add_to_calc_a = QAction("Add to Calculator A", self)
                menu.addAction(action_add_to_calc_a)

                action_add_to_calc_b = QAction("Add to Calculator B", self)
                menu.addAction(action_add_to_calc_b)

                # Connect to handlers
                def add_to_calc_a() -> None:
                    if len(parents_list) > 1:
                        # Format: filename.ext::path/to/dataset
                        filename = parents_list[0]
                        dataset_path = "/".join(parents_list[1:])
                        full_path = f"{pathlib.Path(filename).name}::{dataset_path}"

                        # Add to calculator A
                        self.calculator.add_to_dataset_a(full_path)

                def add_to_calc_b() -> None:
                    if len(parents_list) > 1:
                        # Format: filename.ext::path/to/dataset
                        filename = parents_list[0]
                        dataset_path = "/".join(parents_list[1:])
                        full_path = f"{pathlib.Path(filename).name}::{dataset_path}"

                        # Add to calculator B
                        self.calculator.add_to_dataset_b(full_path)

                action_add_to_calc_a.triggered.connect(add_to_calc_a)
                action_add_to_calc_b.triggered.connect(add_to_calc_b)

            # Check if FTH tool is open
            if self._tool_is_open("fth_tool"):
                if menu.actions():
                    menu.addSeparator()

                for _ch, _label in [("CL", "CL"), ("CR", "CR"), ("Dark", "Dark")]:
                    _action = QAction(f"-> FTH as {_label}", self)

                    def _make_fth_handler(ch=_ch):
                        def _handler():
                            if len(parents_list) > 1:
                                filename = parents_list[0]
                                dataset_path = "/".join(parents_list[1:])
                                full_path = f"{filename}::{dataset_path}"
                                self.fth_tool.add_dataset_to_combo(full_path, ch)
                        return _handler

                    _action.triggered.connect(_make_fth_handler())
                    menu.addAction(_action)

            # Check if CDI tool is open
            if hasattr(self, 'cdi_tool') and self.cdi_tool is not None and self.cdi_tool.isVisible():
                if menu.actions():
                    menu.addSeparator()

                for _ch, _label in [("CL", "CL"), ("CR", "CR"), ("Dark", "Dark")]:
                    _action = QAction(f"-> CDI as {_label}", self)

                    def _make_cdi_handler(ch=_ch):
                        def _handler():
                            if len(parents_list) > 1:
                                filename = parents_list[0]
                                dataset_path = "/".join(parents_list[1:])
                                full_path = f"{filename}::{dataset_path}"
                                self.cdi_tool.add_dataset_to_combo(full_path, ch)
                        return _handler

                    _action.triggered.connect(_make_cdi_handler())
                    menu.addAction(_action)

            # Show menu if any actions were added
            if menu.actions():
                if (viewport := self.tree_view_file.viewport()) is not None:
                    menu.popup(viewport.mapToGlobal(pos))

    @pyqtSlot()
    def _handle_action_open_file(self) -> None:
        """Open HDF5 Files."""
        settings = QSettings()
        folder: pathlib.Path = pathlib.Path(
            settings.value("paths/last_opened_file_directory", defaultValue=os.path.expanduser("~"))
        )
        default_path = str(folder.absolute()) if folder.absolute().exists() else os.path.expanduser("~")
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open File",
            default_path,
            get_file_filter_string(),
        )
        if not file_paths:
            return

        settings.setValue("paths/last_opened_file_directory", pathlib.Path(file_paths[0]).parent)
        self._start_open_queue(list(file_paths), mode="files", mark_known=False)

    @pyqtSlot()
    def _handle_action_open_folder(self) -> None:
        """Open all HDF5 Files in a Folder."""
        settings = QSettings()
        folder: pathlib.Path = settings.value(
            "paths/last_opened_folder_directory",
            defaultValue=pathlib.Path(os.path.expanduser("~")),
        )
        default_path = str(folder.absolute()) if folder.absolute().exists() else os.path.expanduser("~")
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", default_path)
        if not folder_path:
            return

        settings.setValue("paths/last_opened_folder_directory", pathlib.Path(folder_path))
        # Fast path for large folders/network shares:
        # filter by extension only (no per-file h5py open), then natural sort.
        try:
            files_in_dir = [pathlib.Path(folder_path, name) for name in os.listdir(folder_path)]
        except OSError:
            files_in_dir = []
        data_files = [
            str(p)
            for p in natsorted(
                (p for p in files_in_dir if p.is_file() and has_supported_extension(p)),
                key=lambda p: p.name.lower(),
            )
        ]
        if data_files:
            self._start_open_queue(data_files, mode="folder", mark_known=False)

    @pyqtSlot()
    def _handle_action_clear_files(self) -> None:
        """Clear Tree Widget."""
        self.tree_model_file.clear()
        self.table_model_dataset.resetData()
        self._dataset_per_file_index_cache.clear()
        self._dataset_index_last_used.clear()
        self._set_index_status("Index: Idle")
        self.dataset_index_changed.emit()

    def _collapse_all_files(self) -> None:
        """Collapse all files/items in the tree view.
        In monitor mode, also scans the watched folder for new HDF5 files first."""
        if self._monitor_folder is not None:
            self._refresh_monitor_folder()
        self.tree_view_file.collapseAll()
        logging.info("Collapsed all files in tree view")

    def _handle_action_monitor_folder(self) -> None:
        """Toggle folder monitoring mode (File -> Monitor Folder...)."""
        if self._act_monitor.isChecked():
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Monitor")
            if not folder_path:
                self._act_monitor.setChecked(False)
                return
            self._monitor_folder = pathlib.Path(folder_path)
            # Files already open are treated as known so they won't be re-added
            self._monitor_known = {str(fp) for fp in self.opened_files}
            # Immediately open any new HDF5 files already in the folder
            self._refresh_monitor_folder()
            # Visual indicator on the refresh button
            self.btn_collapse_all.setToolTip(
                f"Monitoring: {self._monitor_folder.name}\n"
                "Click to scan for new files + collapse tree"
            )
            self.btn_collapse_all.setStyleSheet(
                "QPushButton { background-color: #9AEBA3; color: white; font-weight: bold; }"
            )
        else:
            # Stop monitoring
            self._monitor_folder = None
            self._monitor_known.clear()
            self.btn_collapse_all.setToolTip("Collapse all files in tree view")
            self.btn_collapse_all.setStyleSheet("")
            self._set_status_text("")

    def _refresh_monitor_folder(self) -> None:
        """Launch a background scan of the monitored folder.

        The disk enumeration + extension checks run in _FolderScanWorker
        so the UI is never frozen.  Tree-model updates happen in the connected
        slot _on_folder_scan_done() which is called on the main thread.
        """
        if self._monitor_folder is None or not self._monitor_folder.exists():
            return
        # Prevent a second scan from launching while one is already running
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        self._set_status_text(
            "Monitoring  |  scanning..."
        )
        self._scan_worker = _FolderScanWorker(
            self._monitor_folder, frozenset(self._monitor_known)
        )
        self._scan_worker.scan_done.connect(self._on_folder_scan_done)
        self._scan_worker.start()

    def _on_folder_scan_done(self, new_files: list, removed_files: list) -> None:
        """Called on the main thread when _FolderScanWorker finishes.

        Removes tree rows for deleted files, then opens new files in batches
        so the UI stays responsive even when hundreds of files are loaded.
        """
        removed_set = set(removed_files)
        removed_count = 0
        if removed_set:
            for row in range(self.tree_model_file.rowCount() - 1, -1, -1):
                item = self.tree_model_file.item(row, TREE_COLUMN_NAME)
                # The worker reports absolute paths; match on those, not labels.
                if item is not None and tree_item_file_path(item) in removed_set:
                    self.tree_model_file.removeRow(row)
                    removed_count += 1
            self._monitor_known -= removed_set

        if new_files:
            self._start_open_queue(
                list(new_files),
                mode="monitor",
                mark_known=True,
                removed_count=removed_count,
            )
            return

        parts = []
        if removed_count:
            parts.append(f"-{removed_count} removed")
        if not parts:
            parts.append("no change")
        self._set_status_text(
            "Monitoring  |  "
            f"{', '.join(parts)}  ({len(self._monitor_known)} loaded)"
        )
        if removed_count:
            self._prime_dataset_index_async()

    def _start_open_queue(
        self,
        file_paths: list[str],
        *,
        mode: str,
        mark_known: bool,
        removed_count: int = 0,
    ) -> None:
        """Start/restart batched file opening to keep UI responsive."""
        self._open_queue_timer.stop()
        self._open_queue = deque(file_paths)
        self._open_queue_total = len(file_paths)
        self._open_queue_processed = 0
        self._open_queue_mode = mode
        self._open_queue_mark_known = mark_known
        self._open_queue_removed_count = removed_count
        if self._open_queue_total > 0:
            self._open_queue_timer.start(0)

    def _process_open_queue_batch(self) -> None:
        """Process one batch of file-open work and yield back to Qt."""
        if not self._open_queue:
            self._finalize_open_queue()
            return

        n = min(self._open_queue_batch_size, len(self._open_queue))
        for _ in range(n):
            fstr = self._open_queue.popleft()
            self._open_file(pathlib.Path(fstr))
            if self._open_queue_mark_known:
                self._monitor_known.add(fstr)
            self._open_queue_processed += 1

        if self._open_queue_mode == "monitor":
            self._set_status_text(
                "Monitoring  |  "
                f"loading {self._open_queue_processed}/{self._open_queue_total}..."
            )
        else:
            self._set_status_text(
                f"Loading files: {self._open_queue_processed}/{self._open_queue_total}..."
            )

        if self._open_queue:
            self._open_queue_timer.start(0)
        else:
            self._finalize_open_queue()

    def _finalize_open_queue(self) -> None:
        """Finalize status after batched loading completes."""
        if self._open_queue_mode == "monitor":
            parts = []
            if self._open_queue_total:
                parts.append(f"+{self._open_queue_total} added")
            if self._open_queue_removed_count:
                parts.append(f"-{self._open_queue_removed_count} removed")
            if not parts:
                parts.append("no change")
            self._set_status_text(
                "Monitoring  |  "
                f"{', '.join(parts)}  ({len(self._monitor_known)} loaded)"
            )
        else:
            self._set_status_text(f"Loaded {self._open_queue_processed} file(s).")
        self._prime_dataset_index_async()

    @pyqtSlot()
    def _handle_action_about(self) -> None:
        """Open About Page."""
        self._about_page = AboutPage()

    @pyqtSlot()
    def _handle_action_calculator(self) -> None:
        """Open Data Calculator dialog."""
        # Reuse existing dialog if possible so state is preserved and reopening is instant.
        dataset_full_keys_1d = self._peek_dataset_full_keys(min_ndim=1)
        if getattr(self, "calculator", None) is not None:
            self.calculator.refresh_dataset_keys(dataset_full_keys_1d, opened_files=self.opened_files)
        else:
            from src.gui.data_calculator_enhanced import DataCalculatorEnhanced

            # Use non-modal dialog to allow dragging from main window
            self.calculator = DataCalculatorEnhanced(
                self.opened_files,
                self,
                dataset_full_keys_1d=dataset_full_keys_1d,
            )
        self.calculator.show()
        self.calculator.raise_()  # Bring to front
        self.calculator.activateWindow()  # Activate the window

    @pyqtSlot()
    def _handle_action_comparison(self) -> None:
        """Open Data Comparison dialog."""
        from src.gui.data_comparison import DataComparisonTool

        # Reuse existing dialog if possible so datasets are preserved.
        dataset_full_keys_1d = self._peek_dataset_full_keys(min_ndim=1)
        if hasattr(self, "comparison_tool") and self.comparison_tool is not None:
            self.comparison_tool.refresh_dataset_keys(dataset_full_keys_1d, opened_files=self.opened_files)
        else:
            self.comparison_tool = DataComparisonTool(
                self.opened_files,
                self,
                dataset_full_keys_1d=dataset_full_keys_1d,
            )
        self.comparison_tool.show()
        self.comparison_tool.raise_()  # Bring to front
        self.comparison_tool.activateWindow()  # Activate the window

    def transfer_calculator_result_to_comparison(self, label: str, data: np.ndarray) -> bool:
        """Append calculator result into comparison tool without clearing existing rows."""
        try:
            arr = np.asarray(data)
            if arr.ndim == 2 and int(arr.shape[1]) >= 100:
                logging.warning(
                    "Transfer to comparison rejected: 2D columns >=100 (cols=%s)",
                    arr.shape[1],
                )
                return False
            if arr.ndim not in (1, 2):
                logging.warning("Transfer to comparison rejected: only 1D/2D allowed (got %sD)", arr.ndim)
                return False
            from src.gui.data_comparison import DataComparisonTool
            if not (hasattr(self, "comparison_tool") and self.comparison_tool is not None):
                dataset_full_keys_1d = self._peek_dataset_full_keys(min_ndim=1)
                self.comparison_tool = DataComparisonTool(
                    self.opened_files,
                    self,
                    dataset_full_keys_1d=dataset_full_keys_1d,
                )
            else:
                dataset_full_keys_1d = self._peek_dataset_full_keys(min_ndim=1)
                self.comparison_tool.refresh_dataset_keys(dataset_full_keys_1d, opened_files=self.opened_files)

            self.comparison_tool.show()
            self.comparison_tool.raise_()
            self.comparison_tool.activateWindow()
            self.comparison_tool.add_dataset_from_array(str(label), arr)
            return True
        except Exception as exc:
            logging.error("Failed to transfer calculator result to comparison: %s", exc)
            return False

    @pyqtSlot()
    def _handle_action_fth(self) -> None:
        """Open FTH/HERALDO Reconstruction Tool."""
        # Reuse existing tool if possible so loaded data is preserved and reopening is instant.
        keys_2d_fth = self._peek_dataset_full_keys(min_ndim=2, min_second_dim=FTH_MIN_SECOND_DIM)
        if getattr(self, "fth_tool", None) is not None:
            self.fth_tool.refresh_dataset_keys(keys_2d_fth, opened_files=self.opened_files)
        else:
            from src.gui.fth_reconstruction_tool import FTHReconstructionTool

            self.fth_tool = FTHReconstructionTool(
                parent=self,
                opened_files=self.opened_files,
                dataset_full_keys_2d=keys_2d_fth,
            )
        self.fth_tool.show()
        self.fth_tool.raise_()
        self.fth_tool.activateWindow()

    @pyqtSlot()
    def _handle_action_cdi(self) -> None:
        """Open CDI Reconstruction Tool."""
        # Reuse existing tool if possible so loaded data is preserved and reopening is instant.
        keys_2d = self._peek_dataset_full_keys(min_ndim=2)
        if getattr(self, "cdi_tool", None) is not None:
            self.cdi_tool.update_opened_files(self.opened_files, keys_2d)
        else:
            from src.gui.cdi_reconstruction_tool import CDIReconstructionTool

            self.cdi_tool = CDIReconstructionTool(
                parent=self,
                opened_files=self.opened_files,
                dataset_full_keys_2d=keys_2d,
            )
        self.cdi_tool.show()
        self.cdi_tool.raise_()
        self.cdi_tool.activateWindow()

    @pyqtSlot()
    def _handle_action_xrms_analyze(self) -> None:
        """Open the unified XRMS Analyze tool (radial / angular / time-resolved)."""
        # Reuse existing tool if possible so loaded data is preserved and reopening is instant.
        keys_2d = self._peek_dataset_full_keys(min_ndim=2)
        if getattr(self, "xrms_tool", None) is not None:
            self.xrms_tool.set_opened_files(self.opened_files)
            self.xrms_tool.refresh_dataset_keys(keys_2d)
        else:
            from src.gui.xrms_analyze_tool import XRMSAnalyzeTool

            self.xrms_tool = XRMSAnalyzeTool(self.opened_files, keys_2d, self)
        self.xrms_tool.show()
        self.xrms_tool.raise_()
        self.xrms_tool.activateWindow()

    @pyqtSlot()
    def _handle_action_q_calibration(self) -> None:
        """Open Q calibration dialog."""
        self._open_q_tool_for_key(self._current_dataset_full_key())

    @pyqtSlot(object)
    def _handle_q_request_from_viewer(self, source_dataset_key=None) -> None:
        """Open the q tool on whatever the viewer is showing.

        For a stack of frames that has to be the *displayed slice*. Handing the
        dataset key over instead made the q tool re-read the file and average
        every frame together — so the slice chosen with the slider, and the
        axis chosen with the combo, both went missing, and the pattern it
        analysed belonged to no single measurement.
        """
        image_view = self._current_image_view_2d()
        if image_view is not None and getattr(image_view, "data_3d", None) is not None:
            slice_data = getattr(image_view, "data", None)
            if slice_data is not None and getattr(slice_data, "ndim", 0) == 2:
                self.open_q_tool_for_array(
                    np.asarray(slice_data),
                    source_label=self._displayed_slice_label(image_view, source_dataset_key),
                )
                return

        self._open_q_tool_for_key(str(source_dataset_key) if source_dataset_key else self._current_dataset_full_key())

    def _displayed_slice_position(self) -> dict[str, int]:
        """Which frame of a stack the viewer is on, for a dialog to open there."""
        image_view = self._current_image_view_2d()
        if image_view is None:
            return {"slice_axis": 0, "slice_index": 0}
        return {
            "slice_axis": int(getattr(image_view, "current_slice_axis", 0)),
            "slice_index": int(getattr(image_view, "current_slice_index", 0)),
        }

    @staticmethod
    def _displayed_slice_label(image_view, source_dataset_key) -> str:
        """Name a slice so the q tool says which frame it is looking at."""
        base = short_series_label(str(source_dataset_key), fallback="") if source_dataset_key else ""
        base = base or "image"
        axis = int(getattr(image_view, "current_slice_axis", 0))
        index = int(getattr(image_view, "current_slice_index", 0))
        return f"{base} [axis {axis}, slice {index}]"

    def _current_dataset_full_key(self) -> str | None:
        """Return currently selected dataset as '<file>::<dataset>'."""
        if not self.cur_obj_path or not self.cur_file:
            return None
        fp = str(self.cur_file)
        ds = str(self.cur_obj_path).strip()
        if not fp or not ds:
            return None
        return f"{fp}::{ds}"

    def _open_q_tool_for_key(self, full_key: str | None) -> None:
        """Open q tool and preload provided dataset key."""
        from src.gui.q_calibration_tool import QCalibrationTool

        keys_2d_fth = self._peek_dataset_full_keys(min_ndim=2, min_second_dim=FTH_MIN_SECOND_DIM)
        if getattr(self, "q_cal_tool", None) is None:
            self.q_cal_tool = QCalibrationTool(
                opened_files=self.opened_files,
                dataset_full_keys_2d=keys_2d_fth,
                parent=self,
            )
        else:
            self.q_cal_tool.set_opened_files(self.opened_files)
            self.q_cal_tool.refresh_dataset_keys(keys_2d_fth)
        if full_key:
            self.q_cal_tool.load_dataset_full_key(full_key, auto_load=True)
        self.q_cal_tool.show()
        self.q_cal_tool.raise_()
        self.q_cal_tool.activateWindow()

    def open_q_tool_for_array(self, arr: np.ndarray, source_label: str = "calculation_result") -> bool:
        """Open q tool and preload an in-memory 2D array."""
        from src.gui.q_calibration_tool import QCalibrationTool

        keys_2d_fth = self._peek_dataset_full_keys(min_ndim=2, min_second_dim=FTH_MIN_SECOND_DIM)
        if getattr(self, "q_cal_tool", None) is None:
            self.q_cal_tool = QCalibrationTool(
                opened_files=self.opened_files,
                dataset_full_keys_2d=keys_2d_fth,
                parent=self,
            )
        else:
            self.q_cal_tool.set_opened_files(self.opened_files)
            self.q_cal_tool.refresh_dataset_keys(keys_2d_fth)

        ok = bool(self.q_cal_tool.load_array_data(np.asarray(arr), source_label=source_label))
        self.q_cal_tool.show()
        self.q_cal_tool.raise_()
        self.q_cal_tool.activateWindow()
        return ok

    def _current_image_view_2d(self):
        """Return active 2D image viewer widget, or None if not active."""
        from src.gui.image_view_2d_enhanced import ImageView2DEnhanced
        from src.gui.unified_data_viewer import UnifiedDataViewer

        w = self.dock_plot.widget()
        if isinstance(w, UnifiedDataViewer):
            cur = w.get_current_widget()
            if isinstance(cur, ImageView2DEnhanced):
                return cur
        return None

    def get_current_image_shape_2d(self) -> tuple[int, int] | None:
        """Expose current 2D image shape for helper dialogs."""
        img_view = self._current_image_view_2d()
        if img_view is None or getattr(img_view, "data", None) is None:
            return None
        data = img_view.data
        if getattr(data, "ndim", 0) < 2:
            return None
        return int(data.shape[0]), int(data.shape[1])

    def apply_q_calibration_to_current(self, params: dict) -> bool:
        """Apply q-calibration params to active 2D image readout."""
        img_view = self._current_image_view_2d()
        if img_view is None:
            return False
        img_view.set_q_calibration(params)
        self._set_status_text("Q calibration applied to current image.")
        return True

    def clear_q_calibration_on_current(self) -> bool:
        """Disable q-calibration on active 2D image readout."""
        img_view = self._current_image_view_2d()
        if img_view is None:
            return False
        img_view.set_q_calibration(None)
        self._set_status_text("Q calibration disabled on current image.")
        return True

    @staticmethod
    def _scan_token_from_filename(name: str) -> str:
        """Last digit run of a file stem, keeping its zero padding ("0033")."""
        match = re.search(r"(\d+)(?!.*\d)", pathlib.Path(name).stem)
        return match.group(1) if match else ""

    def _read_selected_dataset(self, action: str) -> np.ndarray | None:
        """Read the dataset selected in the tree, reporting why if it cannot.

        Shared by the menu bar's Export and Plot: both act on the selection and
        must refuse it for the same reasons.
        """
        if not self.cur_obj_path or self.cur_file is None or not self.cur_file.exists():
            logging.warning("No dataset selected for %s", action.lower())
            QMessageBox.warning(
                self,
                "No Dataset Selected",
                f"Please select a dataset from the tree view before {action.lower()}ting.",
            )
            return None

        try:
            if is_hdf5_file(self.cur_file):
                with h5py.File(self.cur_file, "r") as file:
                    h5_obj = file[self.cur_obj_path]
                    if isinstance(h5_obj, h5py.Group):
                        QMessageBox.warning(
                            self,
                            f"Cannot {action} Group",
                            f"Please select a dataset (not a group) to {action.lower()}.",
                        )
                        return None
                    return np.asarray(h5_obj[()])
            return load_regular_data_file(self.cur_file)
        except Exception as exc:
            logging.error("%s: could not read %s::%s: %s", action, self.cur_file, self.cur_obj_path, exc)
            QMessageBox.critical(self, f"{action} Failed", f"Cannot read the selected dataset:\n{exc}")
            return None

    @pyqtSlot()
    def _handle_action_plot_current(self) -> None:
        """Full plot of the dataset selected in the tree.

        The Export action's twin: same selection, same X, one draws instead of
        writing. 2D data is refused rather than drawn as thousands of curves —
        an image belongs in the image viewer.
        """
        from src.gui.plot_dialog import open_plot_dialog
        from src.gui.plot_series import series_from_columns

        data = self._read_selected_dataset("Plot")
        if data is None:
            return

        values = np.asarray(data).squeeze()
        if values.ndim > 2 or (values.ndim == 2 and values.shape[1] > MAX_PLOT_COLUMNS):
            QMessageBox.information(
                self,
                "Not a Curve",
                f"{'x'.join(str(n) for n in values.shape)} data is an image, not a set of curves.\n"
                "Open it in the image viewer instead.",
            )
            return

        x_data = None
        widget = self._current_plot_widget_1d()
        if widget is not None and getattr(widget, "x_data", None) is not None:
            if len(widget.x_data) == values.shape[0]:
                x_data = widget.x_data

        label = self.cur_obj_path.strip("/").split("/")[-1] if self.cur_obj_path else "Data"
        open_plot_dialog(
            self,
            series_from_columns(label, values, x_data),
            title=label,
            default_dir=self.cur_file.parent if self.cur_file else None,
        )

    def _current_plot_widget_1d(self):
        """The active 1D curve viewer, or None when a 2D image is shown."""
        from src.gui.plot_widget_1d_enhanced import PlotWidget1DEnhanced
        from src.gui.unified_data_viewer import UnifiedDataViewer

        widget = self.dock_plot.widget()
        if isinstance(widget, UnifiedDataViewer):
            widget = widget.get_current_widget()
        return widget if isinstance(widget, PlotWidget1DEnhanced) else None

    @pyqtSlot()
    def _handle_action_export_current(self) -> None:
        """Full export of the dataset selected in the tree.

        Runs the same settings dialog as the batch controls, with a one-element
        target list, so a single dataset gets the same X-axis, dialect and
        colormap options instead of a bare save-path dialog.
        """
        data = self._read_selected_dataset("Export")
        if data is None:
            return

        scan = self._scan_token_from_filename(self.cur_file.name)
        ds_path = self.cur_obj_path.strip("/")
        target = BatchTarget(
            file_path=self.cur_file,
            scan_num=scan,
            ds_path=ds_path,
            label=self.cur_file.stem,
        )
        self._open_export_dialog(
            [target],
            [(self.cur_file, scan)],
            [scan] if scan else [],
            ds_path,
            sample_data=data,
        )

    @pyqtSlot()
    def _handle_close(self) -> None:
        """Close Window."""
        self.close()

    @pyqtSlot()
    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Close Window."""
        if a0 is None:
            return

        # Stop any in-progress background load before exiting
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.cancel()
            self._load_worker.wait(1000)

        settings = QSettings()
        settings.setValue("main_window/size", self.size())
        settings.setValue("main_window/position", self.pos())

        # Monitor folder is not persisted; user must re-enable it manually each session.
        settings.setValue("settings/monitor_folder", "")

        files_to_save = self.opened_files
        settings.setValue("settings/last_opened_files", files_to_save)
        settings.sync()
        self._save_disk_index_cache()

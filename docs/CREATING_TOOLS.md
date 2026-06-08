# Writing your own analysis tool

This project is built so you can drop in your own analysis/reconstruction tool
(like *Data Calculator*, *FTH*, *CDI*, *Q Calibration*) without touching the
viewer internals or breaking the shared interface. This document describes the
**tool contract**: the small set of conventions every tool follows so the main
window can open it, feed it datasets, and reuse it.

Read this once, copy the [template](#5-copy-paste-template) at the bottom, and you
have a working tool in a few minutes.

---

## 1. The big picture

```
┌─────────────────────┐   opened_files + dataset keys    ┌──────────────────┐
│  MainWindow         │ ───────────────────────────────► │  YourTool(QDialog)│
│  (src/gui/          │                                   │  src/gui/your_... │
│   main_window.py)   │ ◄─────────────────────────────── │                   │
└─────────────────────┘     (optional) results back      └────────┬─────────┘
                                                                   │ calls
                                                                   ▼
                                                          ┌──────────────────┐
                                                          │ pure math/algos  │
                                                          │ src/recon/*.py   │
                                                          └──────────────────┘
```

Three rules keep things clean:

1. **The GUI is a thin shell.** Put real numerics in `src/recon/` as plain
   functions of NumPy arrays — no Qt imports. They get unit-tested in `tests/`.
2. **Talk to data only through the shared conventions** below
   (`full_key`, `opened_files`, `DatasetPathCombo`). Never reach into the main
   window's private attributes.
3. **One tool = one non-modal `QDialog`** that the main window owns and reuses.

---

## 2. The data contract

### 2.1 `full_key` — how a dataset is named

Every dataset in the app is identified by a single string:

```
"<file_path>::<dataset_path>"
# e.g. "C:/data/scan_042.h5::entry/data/image"
```

Split it with `rsplit("::", 1)` to get `(file_path, dataset_path)`. This is the
same string that the HDF5 tree puts on the clipboard when you drag a dataset, so
drag-and-drop "just works" if you use `DatasetPathCombo` (below).

### 2.2 `opened_files` — the list of open files

The main window passes its `opened_files` (an iterable of file-path strings) into
every tool. You don't scan the filesystem yourself; you read from these paths
with `h5py`.

### 2.3 Reading a dataset

Always open read-only and copy into memory — never hold an open `h5py.File`:

```python
import h5py
import numpy as np

with h5py.File(file_path, "r") as h5:
    if dataset_path not in h5 or not isinstance(h5[dataset_path], h5py.Dataset):
        raise KeyError(dataset_path)
    arr = np.asarray(h5[dataset_path][()])
```

---

## 3. The reusable widgets

Use these instead of rolling your own — they give you drag-drop and a consistent
look for free.

### `DatasetPathCombo` — the dataset picker

`src/gui/dataset_path_combo.py`. An editable combo box that accepts a dataset
dragged from the HDF5 tree and resolves free-typed `file::dataset` text.

```python
from src.gui.dataset_path_combo import DatasetPathCombo

self._combo = DatasetPathCombo("-- no 2D dataset --")

# Fill it from a pre-collected key list (no disk scan):
self._combo.populate_from_full_keys(full_keys, opened_files=opened_files)

# Read the user's selection — returns (file_path, dataset_path) or None:
entry = self._combo.get_entry(opened_files=opened_files)
if entry is None:
    ...  # invalid / nothing selected
fp, ds = entry
```

Other handy helpers on it: `add_full_key(key, select=True)`,
`load_dataset_full_key(...)` (on tools), `clear_datasets()`.

### `ImageView2DEnhanced`

`src/gui/image_view_2d_enhanced.py` — a 2-D image view (pyqtgraph) with
histogram, ROI, colormap and export. Drop it on the right side of a tool when you
work with images (see Q Calibration and FTH).

### Shared colormap / palette helpers

`src/gui/_shared.py` — colormap get/apply (with `invert`) and light-palette
helpers, so all tools render images consistently. Import what you need rather
than duplicating colormap code.

---

## 4. The tool contract (what the main window expects)

A tool is a `QDialog`. To plug into the main window cleanly, follow this
**standard constructor and method shape**. (The existing tools predate this doc
and vary a little; for a *new* tool, use exactly the shape below — it matches the
calculator/comparison/FTH tools and the singleton wiring in §4.2.)

### 4.1 Constructor + public methods

```python
class YourTool(QDialog):
    def __init__(self, opened_files, parent=None, *, dataset_full_keys=None):
        super().__init__(parent)
        self._opened_files = opened_files
        self._keys = list(dataset_full_keys or [])
        self.setWindowTitle("Your Tool")
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.Window)   # real top-level window
        self._build_ui()
        self._populate()

    # Called by the main window every time the tool is re-opened, so the
    # dataset list and file set stay in sync with what's open. Keep it cheap.
    def refresh_dataset_keys(self, keys, opened_files=None) -> None:
        self._keys = list(keys)
        if opened_files is not None:
            self._opened_files = opened_files
        self._populate()
```

That's the whole required surface: a constructor taking
`(opened_files, parent, dataset_full_keys=...)` and a
`refresh_dataset_keys(keys, opened_files=...)` method. Everything else is yours.

> **Why non-modal + `Window` flag?** So the user can keep dragging datasets from
> the main window while the tool is open, and so the tool is a normal OS window
> (own taskbar entry, resizable) rather than a blocking pop-up.

### 4.2 Wire it into the main window

Three small, mechanical edits in `src/gui/main_window.py`. Nothing else in the
viewer needs to change.

**(a) Add the menu action** (in the `&Tools` menu block, ~line 1138):

```python
act_your = QAction("&Your Tool...", self)
act_your.setShortcut("Ctrl+Shift+Y")   # pick an UNUSED combo (see note)
act_your.triggered.connect(self._handle_action_your_tool)
mbr_tools.addAction(act_your)
```

**(b) Add the handler** — copy the **singleton-reuse pattern** used by every
tool. It keeps one instance alive so reopening is instant and preserves state:

```python
@pyqtSlot()
def _handle_action_your_tool(self) -> None:
    """Open Your Tool."""
    keys = self._peek_dataset_full_keys(min_ndim=2)   # see §4.3
    if getattr(self, "your_tool", None) is not None:
        self.your_tool.refresh_dataset_keys(keys, opened_files=self.opened_files)
    else:
        from src.gui.your_tool import YourTool          # lazy import = fast startup
        self.your_tool = YourTool(
            self.opened_files, self, dataset_full_keys=keys,
        )
    self.your_tool.show()
    self.your_tool.raise_()
    self.your_tool.activateWindow()
```

**(c) (Optional) Pre-warm the import** so the first click has no import lag — add
your module to the tuple in `_prewarm_tool_modules` (~line 1241):

```python
"src.gui.your_tool",
```

> **Shortcut note:** keep shortcuts unique. Currently taken in Tools:
> `Ctrl+Shift+C` (Calculator), `Ctrl+Shift+O` (Comparison), `Ctrl+Shift+Q` (Q Cal),
> `Ctrl+Shift+F` (FTH), `Ctrl+Shift+D` (CDI). Pick another letter.

### 4.3 Getting candidate datasets — `_peek_dataset_full_keys`

The main window keeps a **background-warmed cache** of dataset keys, so don't
scan files on the GUI thread. Ask the cache:

```python
self._peek_dataset_full_keys(min_ndim=1)                    # any ≥1-D dataset
self._peek_dataset_full_keys(min_ndim=2)                    # any image-like
self._peek_dataset_full_keys(min_ndim=2, min_second_dim=8)  # wide 2-D only
```

It returns a `list[str]` of `full_key`s and never blocks. Pass the result to your
tool's constructor / `refresh_dataset_keys`.

### 4.4 Where to put the math

If your tool does anything numerical, put the array-in/array-out core in
`src/recon/your_algo.py` as **pure functions** (no Qt, no widgets), import them
into the GUI, and add tests in `tests/test_recon_your_algo.py`. This is how CDI
and FTH are structured — it keeps the algorithm testable and lets others reuse it
headless. Run the suite with:

```
QT_QPA_PLATFORM=offscreen pytest -q
```

---

## 5. Copy-paste template

Save as `src/gui/your_tool.py`, then do the three `main_window.py` edits in §4.2.

```python
"""Your analysis tool — short description."""

from __future__ import annotations

import logging

import h5py
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.dataset_path_combo import DatasetPathCombo


class YourTool(QDialog):
    """One-line summary of what this tool does."""

    def __init__(self, opened_files, parent=None, *, dataset_full_keys=None) -> None:
        super().__init__(parent)
        self._opened_files = opened_files
        self._keys = list(dataset_full_keys or [])
        self._data: np.ndarray | None = None

        self.setWindowTitle("Your Tool")
        self.setModal(False)
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(900, 600)

        self._build_ui()
        self._populate()

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        row = QHBoxLayout()
        self._combo = DatasetPathCombo("-- no dataset --")
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._load_data)
        row.addWidget(self._combo)
        row.addWidget(btn_load)
        root.addLayout(row)

        self._status = QLabel("")
        root.addWidget(self._status)

        # TODO: add your plot / image / controls here.
        root.addWidget(QWidget())

    def _populate(self) -> None:
        self._combo.populate_from_full_keys(self._keys, opened_files=self._opened_files)

    # ---- main-window contract --------------------------------------------
    def refresh_dataset_keys(self, keys, opened_files=None) -> None:
        """Re-sync the dataset list when the tool is re-opened."""
        self._keys = list(keys)
        if opened_files is not None:
            self._opened_files = opened_files
        self._populate()

    # ---- data -------------------------------------------------------------
    def _load_data(self) -> None:
        entry = self._combo.get_entry(opened_files=self._opened_files)
        if entry is None:
            QMessageBox.warning(self, "Your Tool", "Select a valid dataset.")
            return
        fp, ds = entry
        try:
            with h5py.File(fp, "r") as h5:
                if ds not in h5 or not isinstance(h5[ds], h5py.Dataset):
                    raise KeyError(f"Dataset not found: {ds}")
                self._data = np.asarray(h5[ds][()])
        except Exception as exc:
            logging.exception("Load failed")
            QMessageBox.critical(self, "Your Tool", f"Failed to load:\n{exc}")
            return
        self._status.setText(f"Loaded {fp}::{ds}  shape={self._data.shape}")
        # TODO: call your src/recon/ function and display the result.
```

---

## 6. Checklist before you commit

- [ ] Tool lives in `src/gui/`, numerics (if any) in `src/recon/` with tests.
- [ ] Constructor is `(opened_files, parent, *, dataset_full_keys=None)`.
- [ ] Implements `refresh_dataset_keys(keys, opened_files=None)`.
- [ ] Uses `DatasetPathCombo` + `full_key`; no private main-window access.
- [ ] Reads files read-only and copies into NumPy; holds no open file handle.
- [ ] Three `main_window.py` edits done; shortcut is unique.
- [ ] `flake8 --select=F src/` is clean; `pytest -q` passes.

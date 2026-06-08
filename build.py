"""Unified build script for the HDF5 Viewer (SEXTANTS edition).

Replaces the old trio of build scripts (pyinstaller.py / build_onefile.py /
build_no_gpu.py) with a single entry point.

Examples
--------
    python build.py                 # onedir folder build (default, recommended)
    python build.py --onefile       # single self-contained .exe
    python build.py --installer     # onedir build + Windows installer (.exe setup)

Modes
-----
onedir (default)
    Produces ``dist/HDF5-Viewer/HDF5-Viewer.exe`` next to its dependencies.
    Fastest startup; this is the layout the Windows installer packages.
onefile (--onefile)
    Produces a single ``dist/HDF5-Viewer.exe``. Easiest to hand around, but
    starts slower because it unpacks to a temp dir on every launch.
installer (--installer)
    Runs an onedir build, then invokes Inno Setup (ISCC.exe) on
    ``windows/compile.iss`` to produce a Windows installer in ``dist/``.
    Requires Inno Setup 6 to be installed. Implies onedir.

Both executable builds:
  * collect h5py / pyqtgraph / scipy fully (needed for the histogram LUT and
    FTH/CDI math to work in the frozen app),
  * exclude GPU/CUDA and other heavy unused packages to keep the size sane,
  * on Windows, clean up any CUDA DLLs that dependencies dragged in.
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

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys

import PyInstaller.__main__

from src.version import __version__

APP_NAME = "HDF5-Viewer"
ENTRY_POINT = "main.py"
ICON_PATH = "src/img/sextants.ico"
ISS_SCRIPT = os.path.join("windows", "compile.iss")

# CUDA DLLs some scientific deps pull in; safe to delete from a CPU-only build.
GPU_DLL_PATTERNS = [
    "cublas*.dll", "cublasLt*.dll",
    "cudart*.dll", "cudnn*.dll",
    "cufft*.dll", "curand*.dll",
    "cusolver*.dll", "cusparse*.dll",
    "nvinfer*.dll", "nvrtc*.dll",
]


def print_header(text: str) -> None:
    """Print a banner."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def human_size(size_bytes: float) -> str:
    """Convert a byte count to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def cleanup() -> None:
    """Remove previous build artifacts so the build starts clean."""
    print_header("CLEANUP")
    for item in ("build", "dist"):
        if os.path.exists(item):
            print(f"Removing {item}/")
            shutil.rmtree(item, ignore_errors=True)
    for spec_file in pathlib.Path(".").glob("*.spec"):
        print(f"Removing {spec_file}")
        spec_file.unlink()
    print("Cleanup complete")


def build_args(onefile: bool) -> list[str]:
    """Assemble the PyInstaller argument list shared by both exe modes."""
    args = [
        ENTRY_POINT,
        f"--name={APP_NAME}",
        "--noconfirm",
        "--windowed",
        "--clean",
        "--onefile" if onefile else "--onedir",

        # Data files bundled into the app.
        "--add-data=src/img/*:img",
        "--add-data=LICENSE:.",
        "--add-data=README.md:.",

        # Hidden imports PyInstaller's static analysis misses.
        "--hidden-import=h5py",
        "--hidden-import=h5py.defs",
        "--hidden-import=h5py.utils",
        "--hidden-import=h5py._proxy",
        "--hidden-import=PyQt6",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=pyqtgraph",
        "--hidden-import=pyqtgraph.graphicsItems",
        "--hidden-import=pyqtgraph.imageview",
        "--hidden-import=numpy",
        "--hidden-import=natsort",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        # scipy: used by the FTH/HERALDO and CDI reconstruction tools.
        "--hidden-import=scipy",
        "--hidden-import=scipy.fftpack",
        "--hidden-import=scipy.ndimage",
        "--hidden-import=scipy.special",
        "--hidden-import=scipy.optimize",
        "--hidden-import=src.gui.dataset_path_combo",

        # Collect full package data. CRITICAL: dropping these breaks the
        # pyqtgraph HistogramLUTItem and parts of scipy in the frozen app.
        "--collect-all=h5py",
        "--collect-all=pyqtgraph",
        "--collect-all=scipy",

        # Exclude GPU/CUDA stacks (fixes the multi-GB bloat problem).
        "--exclude-module=cupy",
        "--exclude-module=cupyx",
        "--exclude-module=cuda",
        "--exclude-module=cudnn",
        "--exclude-module=tensorrt",
        "--exclude-module=torch",
        "--exclude-module=tensorflow",

        # Exclude heavy packages this app never imports (size only, safe).
        "--exclude-module=matplotlib",
        "--exclude-module=pandas",
        "--exclude-module=tkinter",
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        "--exclude-module=notebook",
        "--exclude-module=pytest",
        "--exclude-module=sphinx",
    ]

    if sys.platform == "win32":
        args.append(f"--icon={ICON_PATH}")

    return args


def post_build_cleanup(onefile: bool) -> None:
    """Strip CUDA DLLs from an onedir build (onefile is packed, nothing to do)."""
    if onefile or sys.platform != "win32":
        return

    dist_dir = os.path.join("dist", APP_NAME)
    if not os.path.exists(dist_dir):
        return

    print_header("POST-BUILD: scanning for CUDA DLLs")
    total = 0
    for root, _dirs, _files in os.walk(dist_dir):
        for pattern in GPU_DLL_PATTERNS:
            for match in glob.glob(os.path.join(root, pattern)):
                size = os.path.getsize(match)
                total += size
                os.remove(match)
                print(f"  Removed {os.path.basename(match)} ({human_size(size)})")
    print(f"  Reclaimed {human_size(total)}" if total else "  No CUDA DLLs found (good!)")


def report(onefile: bool) -> bool:
    """Print where the build landed and its size; return success."""
    print_header("BUILD COMPLETE")
    if onefile:
        target = pathlib.Path("dist") / f"{APP_NAME}.exe"
        if sys.platform != "win32":
            target = pathlib.Path("dist") / APP_NAME
        if target.exists():
            print(f"Single-file executable: {target.absolute()}")
            print(f"Size: {human_size(target.stat().st_size)}")
            return True
    else:
        folder = pathlib.Path("dist") / APP_NAME
        if folder.exists():
            total = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _d, files in os.walk(folder)
                for f in files
            )
            exe = "HDF5-Viewer.exe" if sys.platform == "win32" else "HDF5-Viewer"
            print(f"Folder build: {folder.absolute()}{os.sep}")
            print(f"Run: {folder.absolute() / exe}")
            print(f"Total size: {human_size(total)}")
            return True

    print("Build FAILED — see the PyInstaller output above.")
    return False


def find_iscc() -> str | None:
    """Locate the Inno Setup command-line compiler (ISCC.exe)."""
    on_path = shutil.which("ISCC") or shutil.which("iscc")
    if on_path:
        return on_path
    # Search the standard install roots for any Inno Setup version (6, 7, ...).
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]
    candidates = []
    for root in roots:
        candidates += glob.glob(os.path.join(root, "Inno Setup *", "ISCC.exe"))
    # Prefer the highest version directory if several are installed.
    candidates.sort(reverse=True)
    return next((c for c in candidates if os.path.exists(c)), None)


def build_installer() -> bool:
    """Run Inno Setup against compile.iss to produce a Windows installer."""
    print_header("BUILD INSTALLER (Inno Setup)")
    if sys.platform != "win32":
        print("Installers can only be built on Windows. Skipping.")
        return False
    if not os.path.exists(ISS_SCRIPT):
        print(f"Missing Inno Setup script: {ISS_SCRIPT}")
        return False

    iscc = find_iscc()
    if iscc is None:
        print("Inno Setup (ISCC.exe) not found.")
        print("Install it from https://jrsoftware.org/isdl.php, then re-run with --installer.")
        return False

    print(f"Using: {iscc}")
    # Pass the single-source version into the .iss (overrides its fallback default).
    result = subprocess.run([iscc, f"/DMyAppVersion={__version__}", ISS_SCRIPT])
    if result.returncode != 0:
        print(f"ISCC failed with exit code {result.returncode}.")
        return False

    produced = sorted(pathlib.Path("dist").glob("*Installer*.exe"))
    if produced:
        print(f"\nInstaller: {produced[-1].absolute()}")
    return True


def main() -> int:
    """Parse arguments and run the requested build."""
    parser = argparse.ArgumentParser(
        description="Unified build script for the HDF5 Viewer (SEXTANTS edition).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--onefile",
        action="store_true",
        help="Build a single self-contained .exe (slower startup).",
    )
    mode.add_argument(
        "--onedir",
        action="store_true",
        help="Build a folder with the .exe and its deps (default, fast startup).",
    )
    parser.add_argument(
        "--installer",
        action="store_true",
        help="Also build a Windows installer via Inno Setup (implies onedir).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip removing build/ and dist/ before building.",
    )
    args = parser.parse_args()

    # --installer needs the onedir layout that compile.iss packages.
    onefile = args.onefile and not args.installer
    if args.onefile and args.installer:
        print("Note: --installer requires onedir; ignoring --onefile.\n")

    print_header(f"HDF5 VIEWER {__version__} BUILD — mode: {'onefile' if onefile else 'onedir'}")

    try:
        if not args.no_clean:
            cleanup()

        PyInstaller.__main__.run(build_args(onefile))
        post_build_cleanup(onefile)
        if not report(onefile):
            return 1

        if args.installer:
            if not build_installer():
                return 1

        print("\nDone.")
        return 0

    except KeyboardInterrupt:
        print("\nBuild cancelled by user.")
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level build guard
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

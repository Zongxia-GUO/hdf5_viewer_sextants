"""Build single executable file (onefile mode) with optimization."""

import os
import shutil
import sys
import pathlib
import PyInstaller.__main__


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def get_size_str(size_bytes):
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def cleanup():
    """Clean up build artifacts."""
    print_header("CLEANUP")

    for item in ['build', 'dist']:
        if os.path.exists(item):
            print(f"Removing {item}/")
            shutil.rmtree(item, ignore_errors=True)

    for spec_file in pathlib.Path('.').glob('*.spec'):
        print(f"Removing {spec_file}")
        spec_file.unlink()

    for pycache in pathlib.Path('.').rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)

    print("Cleanup complete\n")


def build():
    """Build single executable file."""
    print_header("SINGLE FILE BUILD (OPTIMIZED)")

    build_args = [
        "main.py",
        "--name=HDF5-Viewer",
        "--noconfirm",
        "--windowed",
        "--clean",

        # IMPORTANT: Single file mode
        "--onefile",

        # Data files - only essential ones
        "--add-data=src/img/sextants.ico:img",
        "--add-data=src/img/about.svg:img",
        "--add-data=src/img/dataset.svg:img",
        "--add-data=src/img/file.svg:img",
        "--add-data=src/img/file_clear.svg:img",
        "--add-data=src/img/group.svg:img",
        "--add-data=src/img/quit.svg:img",
        "--add-data=src/img/export.svg:img",
        "--add-data=LICENSE:.",
        "--add-data=README.md:.",

        # Hidden imports - only necessary
        "--hidden-import=h5py",
        "--hidden-import=h5py.defs",
        "--hidden-import=h5py.utils",
        "--hidden-import=h5py._proxy",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=pyqtgraph",
        "--hidden-import=numpy",
        "--hidden-import=natsort",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        # scipy: required by FTH/HERALDO reconstruction
        "--hidden-import=scipy",
        "--hidden-import=scipy.fftpack",
        "--hidden-import=scipy.ndimage",
        "--hidden-import=scipy.special",
        "--hidden-import=scipy.optimize",
        # local shared widget
        "--hidden-import=src.gui.dataset_path_combo",

        # Selective collection (NOT --collect-all)
        "--copy-metadata=h5py",
        "--copy-metadata=pyqtgraph",
        "--copy-metadata=scipy",

        # Exclude unnecessary packages (CRITICAL for size reduction)
        "--exclude-module=matplotlib",
        "--exclude-module=pandas",
        "--exclude-module=tkinter",
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        "--exclude-module=notebook",
        "--exclude-module=sphinx",
        "--exclude-module=pytest",
        "--exclude-module=setuptools",
        "--exclude-module=pip",
        "--exclude-module=wheel",
        "--exclude-module=test",
        "--exclude-module=tests",
        "--exclude-module=unittest",
        "--exclude-module=distutils",

        # Exclude unused Qt modules
        "--exclude-module=PyQt6.QtBluetooth",
        "--exclude-module=PyQt6.QtDBus",
        "--exclude-module=PyQt6.QtDesigner",
        "--exclude-module=PyQt6.QtHelp",
        "--exclude-module=PyQt6.QtMultimedia",
        "--exclude-module=PyQt6.QtMultimediaWidgets",
        "--exclude-module=PyQt6.QtNetwork",
        "--exclude-module=PyQt6.QtNetworkAuth",
        "--exclude-module=PyQt6.QtNfc",
        "--exclude-module=PyQt6.QtOpenGL",
        "--exclude-module=PyQt6.QtOpenGLWidgets",
        "--exclude-module=PyQt6.QtPositioning",
        "--exclude-module=PyQt6.QtPrintSupport",
        "--exclude-module=PyQt6.QtQml",
        "--exclude-module=PyQt6.QtQuick",
        "--exclude-module=PyQt6.QtQuick3D",
        "--exclude-module=PyQt6.QtQuickWidgets",
        "--exclude-module=PyQt6.QtRemoteObjects",
        "--exclude-module=PyQt6.QtSensors",
        "--exclude-module=PyQt6.QtSerialPort",
        "--exclude-module=PyQt6.QtSql",
        "--exclude-module=PyQt6.QtSvg",
        "--exclude-module=PyQt6.QtSvgWidgets",
        "--exclude-module=PyQt6.QtTest",
        "--exclude-module=PyQt6.QtWebChannel",
        "--exclude-module=PyQt6.QtWebEngineCore",
        "--exclude-module=PyQt6.QtWebEngineWidgets",
        "--exclude-module=PyQt6.QtWebSockets",
        "--exclude-module=PyQt6.QtXml",
    ]

    if sys.platform == "win32":
        build_args.append("--icon=src/img/sextants.ico")

    print("Build mode: SINGLE FILE (--onefile)")
    print("\nOptimizations:")
    print("  ✓ Exclude unnecessary packages")
    print("  ✓ Exclude unused Qt modules")
    print("  ✓ Selective data collection")
    print("\nExpected size: 100-250 MB")
    print("(Single file is larger than directory mode)")
    print("\nAdvantages:")
    print("  ✓ Only ONE exe file")
    print("  ✓ Easy to distribute")
    print("  ✓ No folder structure needed")
    print("\nBuilding... This may take 3-6 minutes...\n")

    PyInstaller.__main__.run(build_args)


def show_results():
    """Display build results."""
    print_header("BUILD COMPLETE")

    exe_path = pathlib.Path('dist/HDF5-Viewer.exe')

    if exe_path.exists():
        size = exe_path.stat().st_size
        print("✓ Build successful!")
        print(f"\nLocation: {exe_path.absolute()}")
        print(f"File size: {get_size_str(size)}")

        print("\n" + "=" * 70)
        print("SINGLE EXE FILE - Ready to distribute!")
        print("=" * 70)

        print("\nWhat you get:")
        print("  dist/")
        print("  └── HDF5-Viewer.exe  (SINGLE FILE)")

        print("\nTo distribute:")
        print("  1. Copy HDF5-Viewer.exe to any computer")
        print("  2. Double-click to run")
        print("  3. That's it!")

        print("\nNo installation needed!")
        print("No folder structure needed!")
        print("Just one exe file!")

        return True
    else:
        print("✗ Build failed!")
        print("Check the output above for errors.")
        return False


def main():
    """Main build process."""
    print_header("HDF5 VIEWER - SINGLE FILE BUILD")
    print("This will create ONE executable file.")
    print("\nNote: Single file is larger than directory mode")
    print("  Expected: 100-250 MB (vs 50-150 MB for directory)")
    print("  But MUCH easier to distribute!")
    print("\nThis will take 3-6 minutes...")

    try:
        cleanup()
        build()
        success = show_results()

        if success:
            print("\n" + "=" * 70)
            print("SUCCESS! Your single-file executable is ready.")
            print("Find it at: dist\\HDF5-Viewer.exe")
            print("=" * 70)
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n\nBuild cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

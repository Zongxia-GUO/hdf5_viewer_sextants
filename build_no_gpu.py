"""Build without GPU/CUDA libraries.

Based on the original pyinstaller.py with minimal changes:
- onedir instead of onefile (faster startup)
- GPU/CUDA exclusions (fixes 2GB problem)
- Post-build cleanup of CUDA DLLs
"""

import os
import sys
import glob
import shutil
import pathlib
import PyInstaller.__main__


def cleanup():
    """Clean up build artifacts."""
    print("Cleaning up...")
    for item in ['build', 'dist']:
        if os.path.exists(item):
            shutil.rmtree(item, ignore_errors=True)
    for spec_file in pathlib.Path('.').glob('*.spec'):
        spec_file.unlink()


def post_build_cleanup():
    """Remove GPU/CUDA DLLs that may have been pulled in by dependencies."""
    print("\nPost-build: scanning for CUDA DLLs...")

    dist_dir = os.path.join("dist", "HDF5-Viewer")
    if not os.path.exists(dist_dir):
        return

    gpu_patterns = [
        "cublas*.dll", "cublasLt*.dll",
        "cudart*.dll", "cudnn*.dll",
        "cufft*.dll", "curand*.dll",
        "cusolver*.dll", "cusparse*.dll",
        "nvinfer*.dll", "nvrtc*.dll",
    ]

    total_size = 0
    for root, _dirs, _files in os.walk(dist_dir):
        for pattern in gpu_patterns:
            for match in glob.glob(os.path.join(root, pattern)):
                size = os.path.getsize(match)
                total_size += size
                os.remove(match)
                print(f"  Removed: {os.path.basename(match)} ({size / 1024 / 1024:.1f} MB)")

    if total_size > 0:
        print(f"  Saved {total_size / 1024 / 1024:.1f} MB")
    else:
        print("  No CUDA DLLs found (good!)")


def build():
    """Build executable - same as original pyinstaller.py but onedir + no GPU."""
    # ===== SAME AS ORIGINAL pyinstaller.py =====
    build_args = [
        "main.py",
        "--name=HDF5-Viewer",
        "--noconfirm",
        "--windowed",
        "--clean",

        # Data files (same as original)
        "--add-data=src/img/*:img",

        # Hidden imports (same as original)
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
        # scipy: used by FTH/HERALDO reconstruction tool
        "--hidden-import=scipy",
        "--hidden-import=scipy.fftpack",
        "--hidden-import=scipy.ndimage",
        "--hidden-import=scipy.special",
        "--hidden-import=scipy.optimize",
        "--hidden-import=src.gui.dataset_path_combo",

        # Collect all package data (same as original - CRITICAL for histogram!)
        "--collect-all=h5py",
        "--collect-all=pyqtgraph",
        "--collect-all=scipy",

        # ===== ONLY ADDITION: exclude GPU/CUDA =====
        "--exclude-module=cupy",
        "--exclude-module=cupyx",
        "--exclude-module=cuda",
        "--exclude-module=cudnn",
        "--exclude-module=tensorrt",
        "--exclude-module=torch",
        "--exclude-module=tensorflow",
    ]

    if sys.platform == "win32":
        build_args.append("--icon=src/img/sextants.ico")

    # KEY DIFFERENCE: use onedir (not onefile) for faster startup
    # Original uses --onefile which is slow because it extracts to temp dir each time

    print("Building HDF5 Viewer (no GPU, onedir)...")
    print(f"  Args: {len(build_args)} options")
    print()

    PyInstaller.__main__.run(build_args)


def main():
    """Main build process."""
    try:
        cleanup()
        build()
        post_build_cleanup()

        dist_dir = os.path.join("dist", "HDF5-Viewer")
        if os.path.exists(dist_dir):
            total = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _d, files in os.walk(dist_dir)
                for f in files
            )
            print(f"\nBuild complete! Size: {total / 1024 / 1024:.0f} MB")
            print(f"Location: {dist_dir}/")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The single source of truth for the version number is [src/version.py](src/version.py).

## [Unreleased]

## [0.1.0] - 2026-06-08

First public release of the SEXTANTS edition — an extended fork of
[loenard97/hdf5-viewer](https://github.com/loenard97/hdf5-viewer) tailored for the
Soleil SEXTANTS beamline.

### Added

- **Viewing & browsing**
  - Open or drag & drop HDF5 files; browse all groups/datasets in a tree view.
  - Smart format detection: `.h5`, `.hdf5`, `.hdf`, `.nxs`, `.nx5`, `.he5`, `.cxi`,
    `.mat` (MATLAB v7.3+), and extension-less files via content sniffing.
  - Automatic display by type/shape: text, 1-D line plots, multi-curve plots,
    2-D heatmaps, and slice-navigated 3-D images.
  - Remote-file support, dataset-name filtering, and background-indexed search.
  - Export datasets to `.csv` and other formats.
- **Analysis tools** (Tools menu):
  - Data Calculator — interactive arithmetic / FFT on datasets.
  - Data Comparison — overlay and compare multiple datasets.
  - Scattering Pattern Analyze — q-calibration / scattering analysis (X → q conversion).
  - FTH Reconstruction — Fourier-Transform Holography / HERALDO (CL/CR alignment,
    beamstop, differential & Gaussian line filters).
  - CDI Reconstruction — phase retrieval (ER / HIO / RAAR, optional shrinkwrap).
  - Time Resolved XRMS — unified X-ray resonant magnetic scattering analysis:
    region selection with incidence-angle correction, live I(r) / I(θ) / I(t) profiles,
    curve fitting with background subtraction, and frame-by-frame parameter tracking.
- GUI-independent numerical cores in `src/recon/`, covered by a 98-test suite.
- Unified packaging via `build.py` (onedir / onefile / Windows installer).
- Continuous integration workflow, `CONTRIBUTING.md`, and `docs/CREATING_TOOLS.md`.

### License

GNU General Public License v3. Based on the original
[HDF5 Viewer](https://github.com/loenard97/hdf5-viewer) by Dennis Lönard.

[Unreleased]: https://github.com/guozongxia0106-dot/hdf5_viewer_sextants/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guozongxia0106-dot/hdf5_viewer_sextants/releases/tag/v0.1.0

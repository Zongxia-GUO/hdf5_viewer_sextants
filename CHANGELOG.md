# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The single source of truth for the version number is [src/version.py](src/version.py).

## [Unreleased]

## [0.2.0] - 2026-08-20

Analysis, export and batch work. Everything that writes a file now agrees on how
it is named, which dialect it uses and where it is saved, and every curve on
screen has a way out of the window it is in.

### Added

- **Plot windows** — a matplotlib figure for any curve, with 13 qualitative
  palettes measured for colour-vision separation, per-curve line styles, dual Y
  axes, a settings Lock that carries one look across windows, and Copy / Save
  Figure. Reachable from the viewers, the tools, the tree's context menu and the
  batch dialog.
- **Despike filter** — removes single-sample glitches from a displayed curve
  without touching the file. Difference and Hampel methods, a Log/Linear/Auto
  scale so a reflectivity tail is judged by ratio rather than absolute size, and
  a preview of how many points a threshold would catch. Flagged points are
  ringed on the plot and recorded in the exported file's header.
- **Batch export / plot** — one dialog with an Export page and a Plot page over
  the same selection, single-file or combined tables, per-scan or combined
  figures, a progress window, and TIFF/PNG/JPEG for image stacks.
- **Curve transforms** in the comparison tool — per-curve `f(X)` / `f(Y)`
  expressions, replacing the old offset/scale columns.
- **3-D stack handling** — axis and frame selection in the calculator, the
  export dialog and the scattering tool; a frame chosen in the viewer reaches
  the pattern tool and the export instead of being averaged away.
- **Set X** from the tree, routed to the window in front, remembered per tool.
- **Folder column** in the file tree, so the first column can show a filename.
- **Right-click Export / Plot** on every curve-browsing plot, replacing
  pyqtgraph's own export dialog.
- ROI profiles in the 2-D viewer, the scattering tool and the time-resolved tool
  can now be exported at all — they previously had no way out.

### Changed

- **A batch belongs to one folder.** Files matching the same scan numbers in two
  folders used to be exported over each other silently; that is now refused with
  both folders named.
- **One field for the scans**, matched by keyword (substring, space-separated)
  and scan number (a whole part of the name, padding-insensitive), so names like
  `Scan_ECL_5p0uJIR_050` work — they previously matched nothing at all.
- **Export naming** is one convention everywhere: `scan number + dataset + column`
  for curves and headers, the expression for a calculated result.
- **TIFF is data, pictures are pictures** — TIFF exports carry the values as
  recorded, PNG/JPEG carry the colormap and rotation.
- CSV dialects (TXT tab / CSV / CSV2) so exports open correctly in Excel
  regardless of the machine's locale.
- Save dialogs remember the last folder used.
- Scattering-tool ROI profiles follow the q axes instead of staying in pixels.
- Tools clear their results when closed, so reopening one is a fresh start.

### Fixed

- The taskbar icon on first launch, and a slow cold start (2,903 → 2,098 files).
- Several silent data-loss paths in batch export and in the pattern tool.
- Test runs no longer write to the user's real application settings.

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

[Unreleased]: https://github.com/Zongxia-GUO/hdf5_viewer_sextants/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Zongxia-GUO/hdf5_viewer_sextants/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Zongxia-GUO/hdf5_viewer_sextants/releases/tag/v0.1.0

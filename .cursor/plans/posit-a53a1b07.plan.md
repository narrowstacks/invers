<!-- a53a1b07-960a-438c-932d-ed85d3de1b73 da872a1d-2bd2-4220-aa85-e52d78d71edb -->
# Positize: Film Negative → Positive (Rust + cxx-qt + CLI)

## Overview

A Rust workspace providing a negative-to-positive conversion pipeline usable via a cxx-qt GUI and a `clap`-based CLI. Supports camera RAW decoding, high bit-depth TIFF/PNG inputs, ROI-based film base estimation, film profiles, user presets, batch processing, and exports to Linear DNG and 16-bit TIFF. Color management via ICC (lcms2). Initial focus on accurate color negative handling and robust black-and-white.

## Crate/Workspace Layout

- `Cargo.toml` (workspace)
- `crates/positize-core/`: decoding, pipeline, profiles, presets, color mgmt, exporters
- `crates/positize-cli/`: CLI using `clap`
- `crates/positize-gui/`: cxx-qt application, QML UI
- `assets/`: sample images (small), test fixtures, ICC profiles
- `profiles/`: built-in scan/film profiles

## Key Dependencies

- Decode: `libraw` (via FFI crate) for camera RAW, `tiff`, `image`, `png`, `exif`
- Math & perf: `ndarray`, `rayon`
- Color: `lcms2`
- Serialization: `serde`, `serde_yaml`, `serde_json`
- CLI/UI: `clap`, `cxx-qt`, `qml`
- Metadata: `exif`, optional `xmp_toolkit` bindings later

## Data Models

- `FilmPreset`: film name, per-channel base offsets, color matrix (3x3), tone curve params, notes
- `ScanProfile`: capture source (DSLR/mirrorless, flatbed), typical white/black level handling, demosaic/white balance hints
- `BaseEstimation`: ROI, per-channel medians, noise stats
- `ConvertOptions`: input path(s), output format (tiff|dng), working colorspace, bit-depth policy, batch options
- Preset storage: YAML files in `~/.config/positize/presets/` and project-local overrides

## Processing Pipeline (Core)

1. Input decode

- RAW: `libraw` to linear, black-level corrected buffer (demosaiced for v1), extract color matrices and metadata
- TIFF/PNG: load 16-bit linear (or convert to linear if gamma-encoded); preserve bit depth metadata

2. Normalize and metadata

- Track black/white levels, exposure scaling; work internally in f32 linear scene-referred

3. Film base estimation

- From user ROI (median per-channel), optional auto-estimation heuristic
- Store as `BaseEstimation`; can be reused for batch

4. Base subtraction and inversion

- Subtract base; guard against underflow; invert to positive

5. Tone mapping

- Apply neutral S-curve by default; optional film-specific curve from preset

6. Color correction

- Apply 3x3 matrix from preset/profile; optional gray-balance refinement

7. Colorspace transform

- Working space (linear ProPhoto or Rec.2020) via `lcms2`; output profile embedded

8. Quantization & export

- TIFF: 16-bit integer linear
- DNG (Linear): write TIFF+DNG tags; 16-bit container initially; preserve metadata where feasible
- Bit-depth policy: match input effective bit depth when feasible; fallback to 16-bit if not supported by container/implementation

## GUI (cxx-qt + QML)

- Views
- Import panel: file(s) picker, drag-and-drop
- Viewer: zoom/pan, histogram, pixel probe
- ROI tool: rectangular selection for film base; numeric ROI readout
- Preset manager: load/save presets; choose film and scan profile
- Batch queue: list, status, apply base/preset to batch
- Export settings: format (DNG/TIFF), colorspace, bit depth policy, output dir
- UX Essentials
- Real-time preview at reduced resolution; full-res render on export
- Undo/redo for ROI and settings

## CLI (clap)

- Commands
- `convert`: convert file(s)/folder with options and presets
- `analyze-base`: compute and print/save base from ROI or heuristic
- `batch`: apply base/preset to a set of inputs
- `preset`: create/list/show/save
- Examples
- `positize convert input/IMG_0001.CR3 --preset presets/portra400.yml --roi 120,400,240,520 --export dng --out out/`
- `positize analyze-base scan.tif --roi 50,50,500,200 --save base.json`
- `positize batch input/*.tif --base-from base.json --preset profiles/flatbed.yml --export tiff16`

## Profiles & Presets

- Ship a few example `ScanProfile`s (DSLR copy, Epson flatbed)
- Provide starter `FilmPreset`s (generic color negative, generic B&W) with documentation on creating film-specific presets

## Batch Processing

- Apply a reference `BaseEstimation` and preset to a set of files
- Parallelized exports using `rayon` with configurable concurrency

## Color Management

- Default working space: linear Rec.2020
- Output ICC embedded (sRGB/Display P3/Rec.2020/ProPhoto)
- Respect input white balance and camera color matrices when present

## Export Details

- Linear DNG (demosaiced) with essential DNG tags (CFA tags omitted in linear case), `BitsPerSample` initially 16; carry `AsShotNeutral`, `BlackLevel`, `WhiteLevel` where applicable
- 16-bit linear TIFF with embedded ICC
- Bit-depth rule: match input if supported; otherwise 16-bit fallback; document behavior

## Testing & Validation

- Unit tests for base estimation, inversion, tone curves, color transforms
- Golden image tests (small crops) to validate pipeline determinism
- Round-trip tests: RAW → positive TIFF/DNG with expected histograms/statistics

## Milestones

- M1 (CLI MVP): RAW and TIFF/PNG decoding, ROI-based base estimation, invert + neutral curve, 16-bit TIFF export, basic presets, batch
- M2 (GUI MVP): cxx-qt app, preview + ROI tool, preset manager, batch UI
- M3 (Pro): Linear DNG export, ICC color mgmt, improved film profiles, performance tuning, metadata preservation

## Risks & Mitigations

- DNG writing complexity: start with linear DNG minimal tags; fall back to TIFF where needed
- RAW support variance: rely on libraw; provide helpful errors with guidance
- Performance on large scans: tiled processing and preview downscales; parallelism

### To-dos

- [ ] Create Cargo workspace with core, cli, and gui crates
- [ ] Implement decoders: libraw (RAW) and TIFF/PNG loaders
- [ ] Implement base subtract, invert, tone curve in positize-core
- [ ] Add ROI model, median-based base estimation, optional heuristic
- [ ] Define FilmPreset and ScanProfile; YAML load/save
- [ ] Build CLI: convert, analyze-base, batch, preset
- [ ] Export 16-bit linear TIFF with ICC embedding
- [ ] Integrate lcms2 for working/output colorspaces
- [ ] Add Linear DNG export with essential tags
- [ ] Create cxx-qt app with QML shell and file loader
- [ ] Implement preview renderer and ROI selection UI
- [ ] Add preset manager and batch queue to GUI
- [ ] Unit and golden tests; sample assets; E2E CLI tests
- [ ] Write README usage for CLI and GUI; preset authoring guide
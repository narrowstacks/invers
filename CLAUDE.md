# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Invers is a film negative to positive conversion tool written in Rust. It processes scanned film negatives (color and B&W) and converts them to positive images while applying film-specific corrections, tone curves, and color matrices.

The core conversion pipeline is fully functional with CPU processing. GPU acceleration is partially implemented.

## Workspace Structure

This is a Cargo workspace with three crates:

- **invers-core**: Core library containing all conversion logic, models, and utilities

  - Image decoders (TIFF, PNG, RAW via LibRaw)
  - Processing pipeline (base estimation, inversion, tone mapping, color correction)
  - Exporters (TIFF16 implemented, Linear DNG planned)
  - Preset and profile management (YAML-based)
  - Data models for film presets, scan profiles, and conversion options
  - Auto-adjustment algorithms (levels, color, exposure, white balance)
  - GPU acceleration (partial, feature-gated)

- **invers-cli**: Command-line interface

  - Built with clap (derive API)
  - Commands: convert, analyze, batch, preset (list/show/create), init
  - Debug-only commands: diagnose, test-params
  - All commands are fully implemented

- **invers-gui**: Qt-based GUI (planned)
  - Currently a scaffold showing intended architecture
  - Will use cxx-qt for Rust/Qt integration
  - Modules: app_state, preview, roi_tool, ui, viewer

## Build and Development Commands

```bash
# Build entire workspace
cargo build

# Build release version
cargo build --release --features gpu

# Build specific crate
cargo build -p invers-cli
cargo build -p invers-core
cargo build -p invers-gui

# Run CLI (after building)
./target/release/invers --help
./target/release/invers convert --help

# Check code without building
cargo check

# Run tests (when available)
cargo test

# Format code
cargo fmt

# Lint code
cargo clippy
```

## Architecture

### Processing Pipeline Flow

The conversion pipeline (crates/invers-core/src/pipeline.rs) follows these stages:

1. **Decode**: Load image from TIFF/PNG/RAW (crates/invers-core/src/decoders.rs)
2. **Base Estimation**: Calculate film base color (ROI, border-based, histogram-based, or region-based with fallback)
3. **Inversion**: Multiple modes (Linear, Logarithmic, DivideBlend, MaskAware, BlackAndWhite)
4. **Shadow Lift**: Adjustable shadow recovery (Fixed, Percentile, or None modes)
5. **Highlight Compression**: Prevent highlight clipping
6. **Auto Adjustments**: Optional auto-levels, auto-color, auto-exposure, auto white balance
7. **Tone Mapping**: Apply film-specific tone curves (S-curve and asymmetric)
8. **Color Correction**: Apply 3x3 color matrices and HSL adjustments
9. **Colorspace Transform**: Convert to output colorspace (planned for M3)
10. **Export**: Write to TIFF16 (crates/invers-core/src/exporters.rs)

### Data Models

Core data structures (crates/invers-core/src/models.rs):

- **FilmPreset**: Film-specific parameters (base_offsets, color_matrix, tone_curve)
- **ScanProfile**: Capture source characteristics (source_type, white_level, black_level, demosaic/WB hints)
- **BaseEstimation**: Film base estimation results (ROI, medians, noise_stats)
- **ConvertOptions**: Complete conversion configuration for pipeline
- **DecodedImage**: Raw decoded image data with metadata
- **ProcessedImage**: Pipeline output ready for export

### Preset Management

Presets are YAML files managed via crates/invers-core/src/presets.rs:

- Film presets: `~/invers/presets/film/` (or `profiles/film/` in repo)
- Scan profiles: stored alongside film presets
- Functions: load, save, list presets

### Colorspace Handling

Working colorspace defaulted to "linear-rec2020" throughout the pipeline. All internal processing happens in linear light (f32, 0.0-1.0 range).

## Key Implementation Notes

- All image data flows through f32 linear RGB representation
- Rayon is available for parallel processing (used in batch operations)
- Error handling uses Result<T, String> throughout
- Serde used for YAML serialization of presets/profiles
- RGB channel order: [R, G, B] in arrays and matrices

### CPU/GPU Pipeline Parity (CRITICAL)

**Any changes to the processing pipeline MUST be implemented in both CPU and GPU codepaths with 1:1 functional equivalence.**

- CPU pipeline: `crates/invers-core/src/pipeline.rs`
- GPU pipeline: `crates/invers-core/src/gpu/pipeline.rs`
- GPU shaders: `crates/invers-core/src/gpu/shaders/` (WGSL files)
  - `inversion.wgsl` - Negative inversion operations
  - `color_matrix.wgsl` - Color matrix transformations
  - `color_convert.wgsl` - Colorspace conversions
  - `tone_curve.wgsl` - Tone curve application
  - `histogram.wgsl` - GPU histogram computation
  - `utility.wgsl` - Shared utility functions

When modifying pipeline logic:

1. Update the CPU implementation in `pipeline.rs`
2. Update the corresponding GPU shader(s) in `gpu/shaders/`
3. Update the GPU pipeline orchestration in `gpu/pipeline.rs` if needed
4. Ensure both paths produce identical results (within floating-point tolerance)
5. Run tests to verify CPU/GPU parity: `cargo test --features gpu`

The GPU shaders must mirror the CPU algorithms exactly. Differences in results between CPU and GPU execution are bugs.

## Project Status

### Fully Implemented

- **Decoders**: TIFF (all bit depths), PNG (8/16-bit), RAW (via LibRaw with AHD demosaic)
- **Base Estimation**: ROI-based, border-based, histogram-based, region-based with validation
- **Inversion**: 5 modes (Linear, Logarithmic, DivideBlend, MaskAware, BlackAndWhite)
- **Tone Curves**: S-curve and asymmetric curve application
- **Color Correction**: 3x3 matrix transforms, HSL adjustments
- **Auto Adjustments**: Auto-levels, auto-color, auto-exposure, white balance (temperature/tint)
- **Shadow/Highlight**: Shadow lift (Fixed/Percentile/None), highlight compression
- **Exporters**: TIFF16 (RGB and grayscale)
- **CLI Commands**: convert, analyze, batch, preset (list/show/create), init
- **Debug Tools**: diagnose (comparison with third-party software), test-params (parameter optimization)
- **Preset Management**: YAML-based film presets and scan profiles

### Partially Implemented

- **GPU Pipeline**: Framework in place with WGPU, some stages implemented (inversion, shadow lift, highlight compression). Falls back to CPU when unavailable.

### Not Yet Implemented

- **Linear DNG Export**: Stub only
- **Colorspace Transform**: Planned for M3
- **GUI**: Architecture defined, implementation planned

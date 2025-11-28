# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Invers is a film negative to positive conversion tool written in Rust. It processes scanned film negatives (color and B&W) and converts them to positive images while applying film-specific corrections, tone curves, and color matrices.

The project is in early development with most core functionality stubbed out but the architecture defined.

## Workspace Structure

This is a Cargo workspace with three crates:

- **invers-core**: Core library containing all conversion logic, models, and utilities

  - Image decoders (TIFF, PNG, planned RAW support)
  - Processing pipeline (base estimation, inversion, tone mapping, color correction)
  - Exporters (TIFF16, Linear DNG)
  - Preset and profile management (YAML-based)
  - Data models for film presets, scan profiles, and conversion options

- **invers-cli**: Command-line interface

  - Built with clap (derive API)
  - Commands: convert, analyze-base, batch, preset (list/show/create)
  - Most commands are currently unimplemented stubs

- **invers-gui**: Qt-based GUI (planned for M2)
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
2. **Base Estimation**: Calculate film base color from ROI or auto-detection
3. **Base Subtraction & Inversion**: Subtract base, invert to positive (1.0 - negative)
4. **Tone Mapping**: Apply film-specific tone curves
5. **Color Correction**: Apply 3x3 color matrices
6. **Colorspace Transform**: Convert to output colorspace (linear-rec2020 default)
7. **Export**: Write to TIFF16 or Linear DNG (crates/invers-core/src/exporters.rs)

All pipeline functions are currently stubs returning "not yet implemented" errors.

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

- Film presets: `~/.config/invers/presets/` (or `profiles/film/` in repo)
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

## Project Status

Most functionality is currently unimplemented:

- Decoders (TIFF, PNG, RAW) - stubs only
- Pipeline functions (base estimation, inversion, tone mapping, color correction) - stubs only
- Exporters (TIFF16, DNG) - stubs only
- CLI commands (convert, analyze-base, batch, preset show/create) - stubs only
- GUI - architecture defined, M2 implementation planned

Working functionality:

- Preset listing (cmd_preset_list in CLI)
- Basic CLI argument parsing
- Data model structures complete

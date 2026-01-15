# invers-core

Core library for film negative to positive conversion.

## Modules

All major modules use directory structure with `mod.rs` and submodules.

- `pipeline/` - Processing pipeline (CPU)
  - `base_estimation/` - Base color extraction (analysis, extraction, methods)
  - `inversion/` - Negative inversion algorithms (modes, reciprocal)
  - `tone_mapping/` - Tone curves and application (curves, apply)
  - `helpers.rs`, `legacy.rs`, `research.rs`
- `gpu/` - GPU acceleration (wgpu)
  - `context/` - GPU context and pipeline initialization
  - `pipeline/` - GPU pipeline orchestration (analysis, cb, dispatch, histogram, ops)
  - `shaders/` - WGSL compute shaders
- `cb_pipeline/` - Curve-based pipeline (CB-style processing)
  - `layers.rs`, `process.rs`, `tests.rs`
- `auto_adjust/` - Auto-adjustment algorithms
  - `levels/` - Auto-levels (analysis, histogram, auto_levels)
  - `white_balance/` - White balance (auto_wb, compute, kelvin)
  - `color.rs`, `exposure.rs`
- `models/` - Data structures
  - `cb/` - CB pipeline models (color_model, enums, histogram, options, presets, tone_profile)
  - `convert_options/` - Conversion options (defaults, density, enums, impls)
  - `scan_profile/` - Scan profiles (hints, hsl, mask, profile)
- `decoders/` - Image decoders (tiff.rs, png.rs, raw.rs)
- `exporters.rs` - Image export (TIFF16)
- `color/` - Color space conversions (conversions, hsl, lab)
- `diagnostics/` - Debug output and comparison (compare, output, stats)
- `testing/` - Test utilities (grid_search, runners, scoring, types)
- `config/` - Pipeline configuration and defaults

## Pipeline Modes

Three processing pipelines (`pipeline/mod.rs:60`):

- **Legacy**: Original pipeline with multiple inversion modes
- **Research**: Density-balance-first for eliminating color crossover
- **CbStyle**: Curve-based pipeline (Negative Lab Pro-inspired)

## GPU Acceleration

GPU is feature-gated. Enable with `--features gpu`.

**Shaders in `gpu/shaders/*.wgsl` must match CPU algorithms exactly.**
Run `cargo test --features gpu` to verify CPU/GPU parity.

## Adding a Pipeline Stage

1. Add CPU implementation in `pipeline/` or `cb_pipeline/`
2. Add GPU shader in `gpu/shaders/`
3. Update GPU orchestration in `gpu/pipeline/`
4. Add tests verifying CPU/GPU parity

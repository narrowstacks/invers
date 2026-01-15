# invers-core

Core library for film negative to positive conversion.

## Modules

- `pipeline/` - Processing pipeline (CPU): base estimation, inversion, tone mapping
- `gpu/` - GPU acceleration (wgpu): shaders, buffers, context
- `cb_pipeline/` - Curve-based pipeline (CB-style processing)
- `auto_adjust/` - Auto-levels, color, exposure, white balance algorithms
- `models/` - Data structures (FilmPreset, ConvertOptions, BaseEstimation, etc.)
- `decoders.rs` - Image decoders (TIFF, PNG, delegates RAW to invers-raw)
- `exporters.rs` - Image export (TIFF16)
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
3. Update GPU orchestration in `gpu/pipeline.rs`
4. Add tests verifying CPU/GPU parity

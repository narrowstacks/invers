# Invers

Film negative to positive conversion tool in Rust. Processes scanned negatives (color/B&W) with film-specific corrections, tone curves, and color matrices.

## Rules

- You should always invoke the Context7 tools for acquiring docs on crates, shaders, and other libraries.

## Workspace

- **invers-core**: Processing pipeline, decoders (TIFF/PNG/RAW), exporters, models, auto-adjustments
- **invers-cli**: CLI with clap - commands: convert, analyze, batch, init
- **invers-raw**: LibRaw bindings for RAW file decoding

## Commands

```bash
cargo build --release --features gpu    # Build with GPU acceleration
cargo test --features gpu               # Run tests (includes CPU/GPU parity)
cargo fmt && cargo clippy               # Format and lint
./target/release/invers convert <file>  # Convert a negative
```

## Architecture

Pipeline stages (`crates/invers-core/src/pipeline/mod.rs`):

1. Decode → Base Estimation → Inversion → Shadow Lift → Highlight Compression
2. Auto Adjustments → Tone Mapping → Color Correction → Export

All processing uses f32 linear RGB (0.0-1.0 range).

## Critical: CPU/GPU Parity

**Pipeline changes MUST be implemented in both CPU and GPU codepaths.**

| Component | CPU               | GPU                    |
| --------- | ----------------- | ---------------------- |
| Pipeline  | `pipeline/mod.rs` | `gpu/pipeline/mod.rs`  |
| Shaders   | N/A               | `gpu/shaders/*.wgsl`   |

When modifying pipeline logic:

1. Update CPU implementation
2. Update corresponding GPU shader(s)
3. Run `cargo test --features gpu` to verify parity

## Key Files

- `crates/invers-core/src/pipeline/` - CPU processing pipeline (base_estimation/, inversion/, tone_mapping/)
- `crates/invers-core/src/gpu/` - GPU acceleration (context/, pipeline/, shaders/)
- `crates/invers-core/src/models/` - Data structures (cb/, convert_options/, scan_profile/)
- `crates/invers-core/src/auto_adjust/` - Auto-adjustments (levels/, white_balance/, color.rs, exposure.rs)
- `crates/invers-core/src/decoders/` - Image decoders (tiff.rs, png.rs, raw.rs)
- `crates/invers-core/src/color/` - Color space conversions (hsl.rs, lab.rs)
- `crates/invers-cli/src/commands/` - CLI command implementations
- `config/pipeline_defaults.yml` - Default pipeline configuration

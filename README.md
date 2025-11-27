# Invers

A professional-grade film negative to positive conversion tool written in Rust. Invers processes scanned film negatives (color and black & white) and converts them to high-quality positive images with film-specific corrections, tone curves, and color matrices.

## Features

- **Intelligent Base Estimation**: Automatically detects film base color from image borders or manual ROI selection
- **Multiple Inversion Modes**: Linear and logarithmic (density-based) inversion algorithms
- **Advanced Tone Curves**: Asymmetric curves with independent toe/shoulder controls, plus traditional S-curves
- **Color Correction**: 3×3 color matrices for orange mask and film dye correction
- **Auto-Adjustments**: Auto-levels, auto-color, auto-exposure with configurable parameters
- **Shadow Recovery**: Adaptive shadow lift based on image analysis
- **High-Quality Output**: 16-bit TIFF export with optional ICC profile embedding
- **Parallel Processing**: Multi-threaded pipeline using Rayon for fast batch processing
- **Film Presets**: YAML-based preset system for film-specific settings

## Installation

### Homebrew (macOS and Linux)

The easiest way to install invers is via Homebrew:

```bash
brew install narrowstacks/invers/invers
```

This installs pre-built binaries for:

- macOS (Intel and Apple Silicon)
- Linux (x86_64)

GUI is not yet packaged, as it's not finished!

### Building from Source

If you prefer to build from source or need a platform not covered by Homebrew:

#### Prerequisites

- Rust 1.70 or later
- Cargo (included with Rust)

#### Build

```bash
# Clone the repository
git clone https://github.com/narrowstacks/invers.git
cd invers

# Build release version (recommended)
cargo build --release

# The binary will be at ./target/release/invers
```

#### Other Build Commands

```bash
# Run tests
cargo test

# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

## Quick Start

### Basic Conversion

```bash
# Convert a single negative with automatic settings
invers convert negative.tif

# Convert with a specific film preset
invers convert negative.tif --preset profiles/film/generic-color-negative.yml

# Specify output location
invers convert negative.tif --out ./converted/
```

### Analyze Film Base

```bash
# Auto-detect film base
invers analyze-base negative.tif

# Analyze specific region (x,y,width,height)
invers analyze-base negative.tif --roi 100,100,500,500

# Save base estimation for reuse
invers analyze-base negative.tif --save base.yml
```

### Batch Processing

```bash
# Process multiple files
invers batch *.tif --out ./converted/

# Batch with preset and parallel threads
invers batch *.tif \
  --preset profiles/film/fuji-superia-400.yml \
  --threads 8 \
  --out ./converted/
```

### Managing Presets

```bash
# List available presets
invers preset list --dir profiles/film

# Show preset details
invers preset show generic-color-negative

# Create new preset template
invers preset create my-film --dir ./presets/
```

## CLI Reference

### `convert` - Convert Negative to Positive

```text
invers convert [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input negative image file

Options:
  -o, --out <PATH>           Output file or directory
  -p, --preset <FILE>        Film preset YAML file
  -s, --scan-profile <FILE>  Scan profile YAML file
  -r, --roi <ROI>            Base estimation ROI (x,y,w,h)
  -f, --format <FORMAT>      Output format [default: tiff16]
      --no-tone-curve        Skip tone curve application
      --no-color-matrix      Skip color matrix application
      --exposure <VALUE>     Exposure compensation (e.g., 1.2 = +0.26 EV)
      --debug                Enable debug output
  -h, --help                 Print help
```

### `analyze-base` - Analyze Film Base

```text
invers analyze-base [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input negative image file

Options:
  -r, --roi <ROI>    Base estimation ROI (x,y,w,h)
  -s, --save <FILE>  Save estimation to YAML file
  -h, --help         Print help
```

### `batch` - Batch Process Files

```text
invers batch [OPTIONS] <INPUTS>...

Arguments:
  <INPUTS>...  Input negative image files

Options:
  -o, --out <DIR>            Output directory
  -p, --preset <FILE>        Film preset YAML file
  -t, --threads <N>          Number of parallel threads
  -b, --base <FILE>          Shared base estimation file
  -h, --help                 Print help
```

### `preset` - Manage Presets

```text
invers preset <COMMAND>

Commands:
  list    List available presets
  show    Display preset details
  create  Create new preset template
```

## Processing Pipeline

The conversion pipeline processes images through these stages:

```text
Input Image (TIFF/PNG)
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Decode to f32 linear RGB (0.0-1.0) │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Base Estimation (auto or manual)   │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Invert to Positive                 │
   │  (linear or logarithmic mode)       │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Shadow Lift (adaptive/fixed)       │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Auto-Adjustments                   │
   │  • Auto-levels (histogram stretch)  │
   │  • Auto-color (neutralize casts)    │
   │  • Auto-exposure (normalize)        │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Color Matrix (3×3 correction)      │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Tone Curve (S-curve/asymmetric)    │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Export to TIFF16                   │
   └─────────────────────────────────────┘
```

## Film Presets

Film presets are YAML files that define film-specific conversion parameters.

### Preset Structure

```yaml
name: "Generic Color Negative"
base_offsets: [0.0, 0.0, 0.0]
color_matrix:
  - [1.10, -0.02, -0.08]
  - [-0.02, 1.05, -0.03]
  - [-0.08, -0.03, 1.15]
tone_curve:
  curve_type: "asymmetric" # linear, s-curve, or asymmetric
  strength: 0.4
  toe_strength: 0.4 # Shadow lift (0.0-1.0)
  shoulder_strength: 0.3 # Highlight compression (0.0-1.0)
  toe_length: 0.25 # Where toe region extends
  shoulder_start: 0.75 # Where shoulder begins
notes: "General-purpose color negative preset"
```

### Included Presets

| Preset                                  | Description                          |
| --------------------------------------- | ------------------------------------ |
| `generic-color-negative.yml`            | General-purpose color negative       |
| `generic-color-negative-asymmetric.yml` | Color negative with asymmetric curve |
| `generic-bw.yml`                        | Black & white negative               |
| `fuji-superia-400.yml`                  | Fuji Superia 400 specific            |
| `optimized-standard.yml`                | Optimized general-purpose            |

### Tone Curve Types

- **Linear**: No curve applied (pass-through)
- **S-Curve**: Symmetric contrast enhancement with adjustable strength
- **Asymmetric**: Film-like curve with independent toe and shoulder controls for natural-looking conversions

## Configuration

### Pipeline Defaults

Default pipeline settings can be configured in `pipeline_defaults.yml`:

```yaml
auto_levels_enabled: true
auto_levels_clip_percent: 0.1
auto_color_enabled: true
auto_exposure_enabled: true
auto_exposure_target: 0.18
shadow_lift_mode: "percentile"
inversion_mode: "linear"
highlight_compression: 0.95
```

### Default Locations

- **Film presets**: `~/.config/invers/presets/` or `profiles/film/`
- **Scan profiles**: Same location as film presets
- **Pipeline config**: `pipeline_defaults.yml` or `pipeline.yml`

## Project Structure

```text
invers/
├── crates/
│   ├── invers-core/     # Core conversion library
│   │   ├── src/
│   │   │   ├── lib.rs          # Library exports
│   │   │   ├── models.rs       # Data structures
│   │   │   ├── pipeline.rs     # Processing pipeline
│   │   │   ├── decoders.rs     # Image format decoders
│   │   │   ├── exporters.rs    # Output format writers
│   │   │   ├── presets.rs      # Preset management
│   │   │   ├── auto_adjust.rs  # Auto-adjustment algorithms
│   │   │   ├── config.rs       # Configuration system
│   │   │   └── diagnostics.rs  # Testing utilities
│   │   └── Cargo.toml
│   │
│   ├── invers-cli/      # Command-line interface
│   │   ├── src/
│   │   │   ├── main.rs         # CLI entry point
│   │   │   └── lib.rs          # Shared utilities
│   │   └── Cargo.toml
│   │
│   └── invers-gui/      # GUI application (in development)
│       ├── src/
│       │   └── main.rs
│       └── Cargo.toml
│
├── profiles/
│   └── film/            # Film preset files
│
├── pipeline_defaults.yml
├── Cargo.toml           # Workspace manifest
└── README.md
```

## Technical Details

### Image Processing

- **Working Colorspace**: Linear RGB (Rec. 2020 primaries)
- **Internal Precision**: 32-bit floating point (0.0-1.0 range)
- **Output Precision**: 16-bit integer (TIFF16)
- **Channel Order**: RGB

### Base Estimation Algorithm

1. Sample multiple regions (borders, corners, center)
2. Calculate median values per channel
3. Validate candidates:
   - Brightness threshold (>0.25)
   - Noise statistics
   - Orange mask characteristics (for color film)
4. Select best candidate or fall back to center region
5. Return medians and noise statistics

### Performance Optimizations

- Parallel processing for images >100k pixels
- Chunk-based processing (256-pixel chunks) for cache efficiency
- Single-pass histogram computation
- Partial sorting for median calculation (O(n) vs O(n log n))

## Supported Formats

### Input

- TIFF (8-bit, 16-bit, 32-bit)
- PNG (8-bit, 16-bit)
- RAW formats (planned: CR2, CR3, NEF, ARW, etc.)

### Output

- TIFF16 (16-bit linear)
- Linear DNG (planned)

## Roadmap

- [x] Core conversion pipeline
- [x] TIFF/PNG decoding
- [x] Base estimation
- [x] Tone curves (S-curve, asymmetric)
- [x] Color matrix correction
- [x] Auto-adjustments
- [x] TIFF16 export
- [x] CLI interface
- [x] Preset system
- [ ] RAW format support
- [ ] Linear DNG export
- [ ] GUI application
- [ ] ICC profile embedding

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with Rust and the amazing ecosystem of image processing crates
- Inspired by professional film scanning workflows and tools like Grain2Pixel and NegativeLabPro.

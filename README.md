# Invers

A professional-grade film negative to positive conversion tool written in Rust. Invers processes scanned film negatives (color and black & white) and converts them to high-quality positive images with film-specific corrections, tone curves, and color matrices.

## Features

- **Orange Mask-Aware Inversion**: Intelligent algorithm that properly accounts for the orange mask in color negative film, eliminating the common blue cast problem
- **Intelligent Base Estimation**: Automatically detects film base color from image borders with validation for orange mask characteristics
- **Multiple Inversion Modes**: Mask-aware (default), linear, logarithmic, and divide-blend algorithms
- **Advanced Tone Curves**: Asymmetric curves with independent toe/shoulder controls, plus traditional S-curves
- **Auto-Adjustments**: Auto-levels, auto-color, auto-exposure, and auto-white-balance with configurable parameters
- **Shadow Recovery**: Adaptive shadow lift based on image analysis
- **High-Quality Output**: 16-bit TIFF export in linear Rec. 2020 colorspace
- **Parallel Processing**: Multi-threaded pipeline using Rayon for fast batch processing
- **Film Presets**: YAML-based preset system for film-specific settings
- **Modular Architecture**: Clean separation of concerns with dedicated modules for pipeline stages, auto-adjustments, and configuration

## Architecture

Invers follows a modular Rust workspace architecture designed for maintainability and extensibility:

- **invers-core**: Core library with modular subsystems

  - `pipeline/` - Processing stages (base estimation, inversion, tone mapping)
  - `auto_adjust/` - Auto-correction algorithms (levels, color, exposure, white balance)
  - `cb_pipeline/` - Curves-based processing pipeline inspired by Negative Lab Pro
  - `models/` - Data structures for presets, profiles, and conversion options
  - `config/` - Configuration management with defaults and testing utilities
  - `gpu/` - Optional GPU acceleration via WGPU (feature-gated)

- **invers-cli**: Command-line interface with modular command structure

  - Each command (`convert`, `batch`, `analyze`, `preset`, `init`) in its own module
  - Debug commands available in debug builds

- **invers-gui**: Qt-based GUI (planned)

## Installation

### Homebrew (macOS and Linux)

The easiest way to install invers is via Homebrew:

```bash
brew tap narrowstacks/invers
brew install invers
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
cargo build --release --features gpu

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

# Convert with warm white balance
invers convert negative.tif --white-balance warm

# Specify output location and format
invers convert negative.tif --out ./converted/ --export tiff16
```

### Analyze Film Base

```bash
# Auto-detect film base color
invers analyze negative.tif

# Analyze specific region (x,y,width,height)
invers analyze negative.tif --roi 100,100,500,500

# Save base estimation for reuse across a roll
invers analyze negative.tif --save base.json
```

### Batch Processing

```bash
# Process multiple files (shares base estimation from first image)
invers batch *.tif --out ./converted/

# Batch with parallel threads and shared base
invers batch *.tif \
  --threads 8 \
  --out ./converted/

# Use pre-analyzed base for consistent results across a roll
invers batch *.tif --base-from base.json --out ./converted/

# Process each image independently (different rolls mixed together)
invers batch *.tif --per-image --out ./converted/
```

### Initialize Config

```bash
# Set up user config directory with default presets
invers init

# Generate shell completions
invers completions zsh > ~/.zfunc/_invers
```

## CLI Reference

### `convert` - Convert Negative to Positive

```text
invers convert [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input file or directory

Output Options:
  -o, --out <PATH>          Output directory or file path
      --export <FORMAT>     Export format: tiff16 (default) or dng

Processing Options:
  -w, --white-balance <PRESET>  White balance preset [default: auto]
                                Values: auto, none, neutral, warm, cool
      --exposure <FLOAT>        Exposure compensation (1.0 = no change, >1.0 = brighter)
      --base <R,G,B>            Manual base RGB values (use 'invers analyze' to find these)
      --bw                      Force black and white conversion mode

General Options:
      --silent              Suppress non-essential output (timing, progress)
      --cpu                 Force CPU-only processing (GPU used by default)
  -v, --verbose             Enable verbose output (config loading, processing details)
  -h, --help                Print help
```

### `analyze` - Analyze Film Base

Analyze an image to estimate film base color. Use this to find base RGB values
that can be reused across multiple frames from the same roll.

```text
invers analyze [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input file to analyze

Options:
      --roi <X,Y,W,H>             ROI for base estimation (x,y,width,height)
      --base-method <METHOD>      Base estimation method [default: regions]
                                  Values: regions, border
      --border-percent <PERCENT>  Border percentage for "border" method [default: 5.0]
      --json                      Output as JSON (machine-readable)
  -s, --save <FILE>               Save analysis to file (JSON format)
  -v, --verbose                   Show detailed analysis output
  -h, --help                      Print help
```

### `batch` - Batch Process Files

Process multiple files with shared settings. By default, assumes all images are
from the same roll and shares base estimation from the first image.

```text
invers batch [OPTIONS] [INPUTS]...

Arguments:
  [INPUTS]...  Input files or directories

Base Estimation:
      --base-from <FILE>    Base estimation file (JSON from 'analyze --save')
      --base <R,G,B>        Manual base RGB values
      --per-image           Estimate base per-image instead of sharing from first

Output Options:
      --export <FORMAT>     Export format: tiff16 (default) or dng
  -o, --out <DIR>           Output directory

Processing Options:
  -w, --white-balance <PRESET>  White balance preset [default: auto]
      --exposure <FLOAT>        Exposure compensation (1.0 = no change)
      --bw                      Force black and white conversion mode

General Options:
  -r, --recursive           Recursively search directories for images
  -j, --threads <N>         Number of parallel threads
      --silent              Suppress non-essential output
  -v, --verbose             Enable verbose output
      --cpu                 Force CPU-only processing
      --dry-run             List files that would be processed without processing
  -h, --help                Print help
```

### `init` - Initialize Configuration

Set up user configuration directory with default presets. Safe to run multiple
times - won't overwrite existing files unless --force is used.

```text
invers init [OPTIONS]

Options:
      --force    Force overwrite of existing files
  -h, --help     Print help
```

### `completions` - Generate Shell Completions

Generate shell completions for your shell of choice.

```text
invers completions <SHELL>

Arguments:
  <SHELL>  Shell to generate completions for
           Values: bash, zsh, fish, powershell

Examples:
  bash: invers completions bash > ~/.bash_completion.d/invers
  zsh:  invers completions zsh > ~/.zfunc/_invers
  fish: invers completions fish > ~/.config/fish/completions/invers.fish
```

### Global Options

```text
      --config-path    Show the path to the config file being used and exit
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
   │  • Detect film base color           │
   │  • Calculate mask profile           │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Invert to Positive                 │
   │  • MaskAware: shadow floor correct  │
   │  • Linear: (base - neg) / base      │
   │  • Log: density-based inversion     │
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
   │  (skipped for MaskAware mode)       │
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

### Orange Mask-Aware Inversion

The default `mask-aware` inversion mode compensates for the orange mask in color negative film:

1. **Base Detection**: Analyzes the film base to determine its orange characteristics
2. **Standard Inversion**: Inverts each channel: `positive = 1.0 - (negative / base)`
3. **Shadow Floor Correction**: Subtracts per-channel floor values from green and blue to remove the blue cast that would otherwise result from inverting the orange mask

The correction uses dye impurity values (how much extra light each dye layer absorbs) to calculate shadow floors:

- **Magenta layer**: Absorbs some blue light (impurity ~0.5)
- **Cyan layer**: Absorbs some green light (impurity ~0.3)

The correction strength is automatically scaled based on how "orange" the detected base is.

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
inversion_mode: "mask-aware" # Default inversion algorithm
auto_levels_enabled: true
auto_levels_clip_percent: 0.1
auto_color_enabled: true
auto_exposure_enabled: true
auto_exposure_target: 0.18
shadow_lift_mode: "percentile"
highlight_compression: 0.95
```

### Default Locations

- **Film presets**: `~/invers/presets/film/`
- **Scan profiles**: `~/invers/presets/scan/`
- **Pipeline config**: `~/invers/pipeline_defaults.yml`

## Project Structure

```text
invers/
├── crates/
│   ├── invers-core/         # Core conversion library
│   │   ├── src/
│   │   │   ├── lib.rs               # Library exports
│   │   │   ├── models/              # Data structures (modular)
│   │   │   │   ├── mod.rs           # Module exports
│   │   │   │   ├── base_estimation.rs
│   │   │   │   ├── cb.rs            # Curves-based pipeline models
│   │   │   │   ├── convert_options.rs
│   │   │   │   ├── preset.rs
│   │   │   │   └── scan_profile.rs
│   │   │   ├── pipeline/            # Processing pipeline (modular)
│   │   │   │   ├── mod.rs           # Main pipeline orchestration
│   │   │   │   ├── base_estimation.rs
│   │   │   │   ├── inversion.rs
│   │   │   │   └── tone_mapping.rs
│   │   │   ├── auto_adjust/         # Auto-adjustment algorithms (modular)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── color.rs
│   │   │   │   ├── exposure.rs
│   │   │   │   ├── levels.rs
│   │   │   │   ├── parallel.rs
│   │   │   │   └── white_balance.rs
│   │   │   ├── cb_pipeline/         # Curves-based pipeline (NLP-inspired)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── histogram.rs
│   │   │   │   ├── layers.rs
│   │   │   │   └── white_balance.rs
│   │   │   ├── config/              # Configuration system (modular)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── defaults.rs
│   │   │   │   └── testing.rs
│   │   │   ├── gpu/                 # GPU acceleration (optional)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── context.rs
│   │   │   │   ├── buffers.rs
│   │   │   │   ├── pipeline.rs
│   │   │   │   └── shaders/
│   │   │   ├── decoders.rs          # Image format decoders
│   │   │   ├── exporters.rs         # Output format writers
│   │   │   ├── presets.rs           # Preset management
│   │   │   ├── color.rs             # Color space utilities
│   │   │   ├── diagnostics.rs       # Debug utilities
│   │   │   └── testing.rs           # Test utilities
│   │   └── Cargo.toml
│   │
│   ├── invers-cli/          # Command-line interface
│   │   ├── src/
│   │   │   ├── main.rs              # CLI entry point
│   │   │   ├── lib.rs               # Shared utilities
│   │   │   └── commands/            # Command implementations (modular)
│   │   │       ├── mod.rs
│   │   │       ├── analyze.rs
│   │   │       ├── batch.rs
│   │   │       ├── convert.rs
│   │   │       ├── debug.rs         # Debug-only commands
│   │   │       ├── init.rs
│   │   │       └── preset.rs
│   │   └── Cargo.toml
│   │
│   └── invers-gui/          # GUI application (planned)
│       ├── src/
│       │   └── main.rs
│       └── Cargo.toml
│
├── config/
│   └── pipeline_defaults.yml
├── profiles/
│   └── film/                # Film preset files
│
├── Cargo.toml               # Workspace manifest
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
- [x] Orange mask-aware inversion
- [x] Tone curves (S-curve, asymmetric)
- [x] Color matrix correction
- [x] Auto-adjustments
- [x] TIFF16 export
- [x] CLI interface
- [x] Preset system
- [x] Modular architecture (core library and CLI)
- [x] Curves-based pipeline (NLP-inspired algorithms)
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
- Inspired by professional film scanning workflows and tools like Grain2Pixel.
- Evan Dorsky's [Why is Color Negative Film Orange?](https://observablehq.com/@dorskyee/understanding-color-film)

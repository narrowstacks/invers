# Reference Comparison Testing Framework

This framework uses ImageMagick CLI tools to empirically compare invers' negative-to-positive conversion results against reference images from other software (e.g., Grain2Pixel, commercial plugins, or manually corrected images).

## Prerequisites

### Required Software

1. **ImageMagick 7.x** (with HDRI enabled for accurate metrics)
   ```bash
   # macOS (Homebrew)
   brew install imagemagick

   # Ubuntu/Debian
   sudo apt-get install imagemagick

   # Verify installation
   magick --version
   ```

2. **Python 3.9+** - System Python or pyenv managed

3. **invers** (built from this repository)
   ```bash
   # From repo root
   cargo build --release
   ```

## Environment Setup

**Always use a virtual environment for this project.** Run the setup script to create and configure the venv:

```bash
cd scripts/reference_compare

# Run setup script (creates venv and installs dependencies)
./setup.sh

# Activate the virtual environment
source .venv/bin/activate
```

### Manual Setup (if setup.sh doesn't work)

```bash
cd scripts/reference_compare

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verifying Setup

```bash
# Ensure venv is activated (you should see (.venv) in your prompt)
which python  # Should show .venv/bin/python

# Test ImageMagick utilities
python -c "from utils.imagemagick import get_image_info; print('OK')"
```

### Deactivating

When done, deactivate the virtual environment:
```bash
deactivate
```

### Optional Software

- **exiftool** - For metadata comparison
  ```bash
  brew install exiftool  # macOS
  ```

## Directory Structure

```
reference_compare/
├── README.md                 # This file
├── setup.sh                  # Environment setup script
├── requirements.txt          # Python dependencies
├── compare_reference.py      # Main comparison script
├── analyze_metrics.py        # Statistical analysis & reporting
├── suggest_improvements.py   # Parameter recommendation engine
├── run_sweep.py              # Automated parameter optimization
├── .venv/                    # Virtual environment (created by setup.sh)
└── utils/
    ├── __init__.py
    └── imagemagick.py        # ImageMagick wrapper utilities
```

## Quick Start

### 1. Set Up Environment

```bash
cd scripts/reference_compare

# Run setup (creates .venv and installs dependencies)
./setup.sh

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Set Up Test Data

Create a test directory with pairs of images:
```bash
mkdir -p ~/invers-tests/test_set_1
```

Place your files:
- `negative.tif` - Original scanned negative
- `reference.tif` - Reference positive (from Grain2Pixel, commercial plugins, etc.)

### 3. Run Single Comparison

```bash
# Compare invers output against a reference
python compare_reference.py \
    ~/invers-tests/test_set_1/negative.tif \
    ~/invers-tests/test_set_1/reference.tif \
    --output-dir ~/invers-tests/results

# With specific inversion mode
python compare_reference.py \
    ~/invers-tests/test_set_1/negative.tif \
    ~/invers-tests/test_set_1/reference.tif \
    --inversion-mode mask-aware
```

### 4. Analyze Metrics

```bash
# Generate detailed analysis report
python analyze_metrics.py \
    ~/invers-tests/results/comparison_*.json \
    --report ~/invers-tests/analysis_report.md
```

### 5. Get Parameter Suggestions

```bash
# Analyze results and suggest improvements
python suggest_improvements.py \
    ~/invers-tests/results/ \
    --target-metric ssim
```

### 6. Run Parameter Sweep

```bash
# Automatically test parameter combinations
python run_sweep.py \
    ~/invers-tests/test_set_1/negative.tif \
    ~/invers-tests/test_set_1/reference.tif \
    --output-dir ~/invers-tests/sweep_results \
    --max-iterations 50
```

## Metrics Explained

### Image Quality Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **RMSE** | Root Mean Square Error - overall pixel difference | 0 (lower is better) |
| **PSNR** | Peak Signal-to-Noise Ratio - quality measure in dB | >40 dB (higher is better) |
| **SSIM** | Structural Similarity Index - perceptual similarity | 1.0 (higher is better) |
| **MAE** | Mean Absolute Error - average pixel difference | 0 (lower is better) |
| **NCC** | Normalized Cross-Correlation - pattern matching | 1.0 (higher is better) |

### Color Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Delta E (CIE2000)** | Perceptual color difference | <2 imperceptible, <5 acceptable |
| **Channel Bias** | Per-channel mean difference | 0 (closer to 0 is better) |
| **Saturation Diff** | Overall saturation difference | 0 |
| **Hue Shift** | Color hue rotation | 0 degrees |

### Tonal Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Histogram Correlation** | Tonal distribution similarity | 1.0 |
| **Mean Luminance Diff** | Brightness difference | 0 |
| **Contrast Ratio** | Dynamic range comparison | 1.0 |
| **Shadow Lift** | Black point difference | 0 |
| **Highlight Compression** | White point difference | 0 |

## Configuration

### Test Configuration File

Create `test_config.yml` for batch testing:

```yaml
# Test configuration
test_sets:
  - name: "Portra 400 - Daylight"
    negative: "portra400_daylight_neg.tif"
    reference: "portra400_daylight_ref.tif"
    notes: "Standard daylight scene"

  - name: "Portra 400 - Tungsten"
    negative: "portra400_tungsten_neg.tif"
    reference: "portra400_tungsten_ref.tif"
    notes: "Indoor tungsten lighting"

# Parameter ranges to test
sweep_parameters:
  inversion_mode: ["mask-aware", "linear", "log"]
  exposure: [0.8, 0.9, 1.0, 1.1, 1.2]
  auto_wb: [true, false]

# Target metrics (weighted)
target_weights:
  ssim: 0.4
  psnr: 0.2
  delta_e: 0.3
  histogram_correlation: 0.1
```

### Environment Variables

```bash
# Optional: Set path to invers binary
export INVERS_BIN=/path/to/invers

# Optional: Set ImageMagick options
export MAGICK_THREAD_LIMIT=4
```

## Output Files

Each comparison generates:

1. **comparison_[timestamp].json** - Raw metrics data
2. **comparison_[timestamp].png** - Visual side-by-side
3. **diff_[timestamp].png** - Difference visualization
4. **histogram_[timestamp].png** - Histogram comparison

## Interpreting Results

### Good Match (SSIM > 0.95, PSNR > 35dB)
- invers is producing results comparable to reference
- Minor tweaks may improve specific areas

### Moderate Match (SSIM 0.85-0.95, PSNR 25-35dB)
- Noticeable differences in color/exposure
- Check suggested parameters for improvements

### Poor Match (SSIM < 0.85, PSNR < 25dB)
- Significant differences detected
- May need different inversion mode or manual base values
- Review difference image to identify problem areas

## Troubleshooting

### "magick: command not found"
Ensure ImageMagick 7.x is installed and in PATH.

### Metrics show NaN or Inf
Images may have different dimensions or color spaces. Ensure both are the same format.

### Very high RMSE despite similar appearance
Check if images have different bit depths or gamma encoding. Use `--normalize` flag.

### Script runs slowly
Reduce image size with `--resize 2000` for faster testing during iteration.

## Contributing Test Sets

If you have well-corrected reference images from professional software, consider contributing them to help improve invers. Create a test case with:

1. Original negative scan (TIFF preferred)
2. Reference positive from known-good software
3. Metadata about film stock, scanner, lighting conditions
4. Any manual adjustments applied to reference

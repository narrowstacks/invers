# Film Preset Guide

This directory contains film-specific presets optimized through systematic testing against professional conversion software (Grain2Pixel).

## Quick Reference

| Preset | Film Base RGB Range | Test Score | Use When |
|--------|-------------------|------------|----------|
| **optimized-standard.yml** | 0.2 - 0.6 | 0.05 - 0.08 | Standard color negatives, default choice |
| **dark-base-negative.yml** | 0.2 - 0.5 | 0.05 - 0.08 | Dark/heavy orange mask, good exposure |
| **light-base-negative.yml** | 0.7 - 0.95 | 0.99 | Thin negatives, minimal orange mask |
| **manual-base-negative.yml** | > 0.95 (failed) | 1.60+ | Auto-detection fails, needs manual ROI |

## Choosing a Preset

### 1. Start with Optimized Standard
```bash
positize convert input.tif --preset profiles/film/optimized-standard.yml
```

The optimized standard preset works excellently for most properly exposed color negatives. It produces results within 5-10% accuracy of professional software.

### 2. Check Your Base Estimation

If results aren't satisfactory, analyze your film base:

```bash
positize analyze-base input.tif --auto
```

Look at the **Base RGB values**:

- **0.2 - 0.6**: Use `optimized-standard.yml` or `dark-base-negative.yml`
- **0.6 - 0.95**: Use `light-base-negative.yml` and consider manual tuning
- **> 0.95 or [1.0, 1.0, 1.0]**: Auto-detection failed! Use `manual-base-negative.yml`

### 3. Film Base Categories Explained

#### Dark/Standard Base (0.2 - 0.6)
**Characteristics:**
- Normal color negative orange mask
- Properly exposed film
- Good density range
- Standard C-41 processing

**Results:** Excellent (test scores 0.05-0.08)

**Presets:** `optimized-standard.yml`, `dark-base-negative.yml`

#### Light Base (0.7 - 0.95)
**Characteristics:**
- Thin or light-density negatives
- Minimal orange mask
- Possible overexposure
- Faded or aged film

**Results:** Challenging (test scores 0.99)

**Preset:** `light-base-negative.yml`

**Additional tweaking may be needed:**
- Increase `exposure_compensation` to 1.2-1.5
- Try `auto_levels_clip_percent` of 2.0-5.0
- Use `base_brightest_percent` of 15-20

#### Failed Detection (> 0.95 or white)
**Characteristics:**
- Severe overexposure
- No clear film base visible in scan
- Cropped scans missing borders
- Scanner auto-exposure issues

**Results:** Failed (test scores 1.60+)

**Preset:** `manual-base-negative.yml`

**REQUIRED:** Manual ROI specification
```bash
# First, find a good base area (borders, sprocket holes, between frames)
positize analyze-base input.tif --roi "x,y,width,height"

# Then convert with manual ROI
positize convert input.tif --preset profiles/film/manual-base-negative.yml \
  --roi "x,y,width,height"
```

## Testing Methodology

These presets are based on systematic parameter grid searches comparing our conversion pipeline against Grain2Pixel v5.5.2 beta.

**Test Images:**
- raw0359: Score 0.0778, Base [0.500, 0.306, 0.229] ✓ Excellent
- raw0003: Score 0.0535, Base [0.498, 0.271, 0.204] ✓ Excellent
- raw0002: Score 0.9910, Base [0.934, 0.928, 0.701] ⚠ Poor
- raw0017: Score 1.6046, Base [1.000, 1.000, 1.000] ✗ Failed

**Scoring Metrics:**
- Mean Absolute Error (MAE) across RGB channels
- Exposure ratio matching (target: 1.0±0.05)
- Color shift per channel (target: <0.02)
- Contrast ratio matching (target: 1.0±0.05)

**Lower scores are better.** Excellent results are < 0.10, acceptable is < 0.20.

## Key Parameters

The presets define color matrices and tone curves. For optimal results, the following processing parameters are recommended (now built into defaults):

- **Auto-levels:** Enabled (1% clip) - Critical for exposure/color matching
- **Auto-color:** Disabled - Redundant with auto-levels
- **Base sampling:** Median of top 10% brightest pixels
- **Inversion:** Linear mode
- **Shadow lift:** Percentile mode (target 0.02)
- **Tone curve strength:** 0.5-0.7

## Legacy Presets

The following presets existed before the optimization work:

- `generic-color-negative.yml` - Basic color negative template
- `generic-bw.yml` - Black and white negative template
- `fuji-superia-400.yml` - Fuji Superia 400 specific

These presets use older parameters and may not perform as well as the optimized presets above.

## Advanced Usage

### Testing Parameters on Your Images

To find optimal settings for your specific film/scanner combination:

```bash
# Quick test with default parameters
positize test-params your_negative.tif reference_conversion.tif \
  --save-output test_output.tif

# Full grid search
positize test-params your_negative.tif reference_conversion.tif \
  --grid --top 10 --output results.json
```

### Creating Custom Presets

1. Start with the closest existing preset
2. Copy it to a new name
3. Adjust the color matrix and tone curve based on your results
4. Document recommended settings in the notes section

## Troubleshooting

### Image too dark
- Increase `exposure_compensation` (try 1.2-1.5)
- Increase `auto_levels_clip_percent` (try 2.0-5.0)
- Check if base detection failed (RGB > 0.95)

### Wrong colors
- Enable `auto_color` temporarily (may help with severe casts)
- Check film base RGB values - outliers need special handling
- Verify you're using the right preset for your base type

### High test scores (> 0.20)
- Film base may be outside normal range
- Base detection may have failed
- Consider manual ROI selection
- Try the light-base or manual-base presets

## Contributing

If you develop presets for specific film stocks, please consider contributing them! Include:
- Test scores and base RGB values
- Sample images (if possible)
- Recommended parameter adjustments
- Film stock and development details

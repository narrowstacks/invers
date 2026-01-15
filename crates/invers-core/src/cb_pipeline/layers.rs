//! Layer processing functions for the CB pipeline.
//!
//! Contains all the `apply_*_layer()` functions that implement individual
//! adjustment layers: gamma, exposure, contrast, highlights, shadows, blacks,
//! whites, and toning.

use super::{apply_inverse_sigmoid, apply_sigmoid, clamp01};
use crate::models::CbWbMethod;

/// Shadow transition threshold (0.1 in CB)
pub(super) const SHADOW_THRESHOLD: f32 = 0.1;

/// Highlight transition threshold (0.8 in CB)
pub(super) const HIGHLIGHT_THRESHOLD: f32 = 0.8;

// ============================================================
// White Balance Pixel Application
// ============================================================

/// Apply white balance to a single pixel using the specified method
pub fn apply_wb_pixel(pixel: &mut [f32], offsets: &[f32; 3], gamma: &[f32; 3], method: CbWbMethod) {
    for (ch, value) in pixel.iter_mut().enumerate() {
        let v = *value;
        let offset = offsets[ch];
        let g = gamma[ch];

        *value = match method {
            CbWbMethod::LinearFixed => {
                if v > 0.0 && v < 1.0 {
                    if v > HIGHLIGHT_THRESHOLD {
                        // Blend offset toward highlights
                        let blend = 1.0 - HIGHLIGHT_THRESHOLD;
                        clamp01(
                            offset
                                + (blend * v - offset * v + offset * HIGHLIGHT_THRESHOLD) / blend,
                        )
                    } else if v < SHADOW_THRESHOLD {
                        // Scale offset in shadows
                        clamp01(v + offset * v / SHADOW_THRESHOLD)
                    } else {
                        // Full offset in midtones
                        clamp01(v + offset)
                    }
                } else {
                    clamp01(v)
                }
            }
            CbWbMethod::LinearDynamic => clamp01(v + offset),
            CbWbMethod::ShadowWeighted => clamp01(v.powf(1.0 / g)),
            CbWbMethod::HighlightWeighted => clamp01(1.0 - (1.0 - v).powf(g)),
            CbWbMethod::MidtoneWeighted => {
                clamp01((v.powf(1.0 / g) + 1.0 - (1.0 - v).powf(g)) / 2.0)
            }
        };
    }
}

// ============================================================
// Tonal Adjustment Layers
// ============================================================

/// Apply gamma layer (brightness) to a pixel
pub fn apply_gamma_layer(pixel: &mut [f32], gamma: f32) {
    for value in pixel.iter_mut() {
        *value = value.powf(gamma);
    }
}

/// Apply exposure layer
pub fn apply_exposure_layer(pixel: &mut [f32], exposure: f32) {
    for value in pixel.iter_mut() {
        let v = *value;
        *value = if exposure < 1.0 {
            1.0 - (1.0 - v).powf(1.0 / exposure)
        } else if exposure > 1.0 {
            v * (1.0 - (1.0 - 2.0_f32.powf(-exposure)) * 0.4)
        } else {
            v
        };
    }
}

/// Apply contrast layer using sigmoid
///
/// OPTIMIZATION: Pre-computes sigmoid normalization constants outside the channel loop.
/// For high-frequency calls (per-pixel), this avoids redundant tanh() computations.
pub fn apply_contrast_layer(pixel: &mut [f32], contrast: f32) {
    let midpoint = 0.5;
    let scale = 0.2;

    if contrast >= 1.0 {
        let c = 1.0 + contrast * scale;
        // Pre-compute sigmoid normalization constants
        let sig_0 = (0.5 * c * (0.0 - midpoint)).tanh();
        let sig_1 = (0.5 * c * (1.0 - midpoint)).tanh();
        let inv_range = 1.0 / (sig_1 - sig_0);

        for value in pixel.iter_mut() {
            let sig_x = (0.5 * c * (*value - midpoint)).tanh();
            *value = (sig_x - sig_0) * inv_range;
        }
    } else if contrast <= -1.0 {
        let c = 1.0 + contrast.abs() * scale * 0.5;
        for value in pixel.iter_mut() {
            *value = apply_inverse_sigmoid(c, midpoint, *value, 1.0);
        }
    }
}

/// Apply highlights adjustment
///
/// OPTIMIZATION: Pre-computes sigmoid normalization constants outside the channel loop.
pub fn apply_highlights_layer(pixel: &mut [f32], highlights: f32) {
    let midpoint = 0.75;
    let range = 0.9;
    let scale = 0.1;
    let strength = 0.5 + highlights.abs() * scale;
    let threshold = 1.0 - range;

    if highlights >= 1.0 {
        // Pre-compute sigmoid constants for this strength/midpoint/range
        // Note: sig_0 and sig_1 use 0.0 and 1.0 as inputs, NOT scaled by range
        // The range only scales the actual input value x
        let sig_0 = (0.5 * strength * (0.0 - midpoint)).tanh();
        let sig_1 = (0.5 * strength * (1.0 - midpoint)).tanh();
        let norm_factor = range / (sig_1 - sig_0);

        for value in pixel.iter_mut() {
            if *value > threshold {
                // x = inv_value / range, where inv_value = 1.0 - *value
                let x = (1.0 - *value) / range;
                let sig_x = (0.5 * strength * (x - midpoint)).tanh();
                *value = 1.0 - (sig_x - sig_0) * norm_factor;
            }
        }
    } else if highlights <= -1.0 {
        for value in pixel.iter_mut() {
            if *value > threshold {
                *value = 1.0 - apply_inverse_sigmoid(strength, midpoint, 1.0 - *value, range);
            }
        }
    }
}

/// Apply shadows adjustment
///
/// OPTIMIZATION: Pre-computes sigmoid normalization constants outside the channel loop.
pub fn apply_shadows_layer(pixel: &mut [f32], shadows: f32) {
    let midpoint = 0.75;
    let range = 0.9;
    let strength = 0.5 + shadows.abs() * 0.1;

    if shadows > 0.0 {
        for value in pixel.iter_mut() {
            if *value < range {
                *value = apply_inverse_sigmoid(strength, midpoint, *value, range);
            }
        }
    } else if shadows < 0.0 {
        // Pre-compute sigmoid constants
        // Note: sig_0 and sig_1 use 0.0 and 1.0 as inputs, NOT scaled by range
        let sig_0 = (0.5 * strength * (0.0 - midpoint)).tanh();
        let sig_1 = (0.5 * strength * (1.0 - midpoint)).tanh();
        let norm_factor = range / (sig_1 - sig_0);

        for value in pixel.iter_mut() {
            if *value < range {
                // x = *value / range (normalized input)
                let x = *value / range;
                let sig_x = (0.5 * strength * (x - midpoint)).tanh();
                *value = (sig_x - sig_0) * norm_factor;
            }
        }
    }
}

/// Apply blacks adjustment
pub fn apply_blacks_layer(pixel: &mut [f32], blacks: f32, shadow_range: f32) {
    let decay = (-shadow_range + 1.0) * 0.33 + 5.0;
    let midpoint = 0.75;
    let range = 0.5;

    if blacks >= 1.0 {
        let lift = blacks / 255.0;
        for value in pixel.iter_mut() {
            if *value < 0.9 {
                *value = lift * (-*value * decay).exp() + *value;
            }
        }
    } else if blacks <= -1.0 {
        let strength = 0.5 + blacks.abs() * 0.1;
        for value in pixel.iter_mut() {
            if *value < range {
                *value = apply_sigmoid(strength, midpoint, *value, range);
            }
        }
    }
}

/// Apply whites adjustment
pub fn apply_whites_layer(pixel: &mut [f32], whites: f32, highlight_range: f32) {
    let decay = (-highlight_range + 1.0) * 0.33 + 5.0;
    let midpoint = 0.75;
    let range = 0.5;

    if whites <= -1.0 {
        let lift = -whites / 255.0;
        for value in pixel.iter_mut() {
            if *value > 0.1 {
                *value = 1.0 - lift * (-(1.0 - *value) * decay).exp() - (1.0 - *value);
            }
        }
    } else if whites >= 1.0 {
        let strength = 0.5 + whites.abs() * 0.1;
        for value in pixel.iter_mut() {
            if *value > 1.0 - range {
                *value = 1.0 - apply_sigmoid(strength, midpoint, 1.0 - *value, range);
            }
        }
    }
}

// ============================================================
// Toning Layers
// ============================================================

/// Apply shadow toning (per-channel color shift in shadows)
pub fn apply_shadow_toning(pixel: &mut [f32], shadow_colors: [f32; 3], shadow_range: f32) {
    let range = 0.9 - (10.0 - shadow_range) * 0.0444;
    let midpoint = 0.75;
    let scale = 0.125;

    for (ch, value) in pixel.iter_mut().enumerate() {
        let color_shift = -shadow_colors[ch];
        if color_shift == 0.0 {
            continue;
        }

        let strength = 0.75 + color_shift.abs() * (1.0 + (10.0 - shadow_range) / 18.0) * scale;

        if *value < range {
            *value = if color_shift > 0.0 {
                apply_inverse_sigmoid(strength, midpoint, *value, range)
            } else {
                apply_sigmoid(strength, midpoint, *value, range)
            };
        }
    }
}

/// Apply highlight toning (per-channel color shift in highlights)
pub fn apply_highlight_toning(pixel: &mut [f32], highlight_colors: [f32; 3], highlight_range: f32) {
    let range = 0.9 - (10.0 - highlight_range) * 0.0444;
    let midpoint = 0.75;
    let scale = 0.125;

    for (ch, value) in pixel.iter_mut().enumerate() {
        let color_shift = -highlight_colors[ch];
        if color_shift == 0.0 {
            continue;
        }

        let strength = 0.75 + color_shift.abs() * (1.0 + (10.0 - highlight_range) / 18.0) * scale;

        if *value > 1.0 - range {
            *value = if color_shift > 0.0 {
                1.0 - apply_sigmoid(strength, midpoint, 1.0 - *value, range)
            } else {
                1.0 - apply_inverse_sigmoid(strength, midpoint, 1.0 - *value, range)
            };
        }
    }
}

/// Apply color gamma layer (cyan/tint/temp adjustments)
pub fn apply_color_gamma_layer(pixel: &mut [f32], color_offsets: [f32; 3]) {
    let blend_range = 0.2;

    for (ch, value) in pixel.iter_mut().enumerate() {
        let offset = (1.0 - color_offsets[ch]) / 4.0;
        let adjusted = *value - offset;

        if *value > 0.0 && *value < 1.0 {
            *value = if adjusted >= 1.0 {
                1.0
            } else if *value > 1.0 - blend_range {
                *value - offset * (1.0 - *value) / blend_range
            } else if adjusted <= 0.0 {
                0.0
            } else {
                adjusted
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_layer() {
        let mut pixel = [0.5, 0.25, 0.75];
        apply_gamma_layer(&mut pixel, 2.0);
        assert!((pixel[0] - 0.25).abs() < 0.001); // 0.5^2 = 0.25
        assert!((pixel[1] - 0.0625).abs() < 0.001); // 0.25^2 = 0.0625
        assert!((pixel[2] - 0.5625).abs() < 0.001); // 0.75^2 = 0.5625
    }

    #[test]
    fn test_contrast_layer_increase() {
        let mut pixel = [0.3, 0.5, 0.7];
        let original = pixel;
        apply_contrast_layer(&mut pixel, 5.0);
        // Increased contrast should push values away from midpoint
        assert!(pixel[0] < original[0]); // Below 0.5 should go lower
        assert!((pixel[1] - 0.5).abs() < 0.01); // 0.5 should stay near 0.5
        assert!(pixel[2] > original[2]); // Above 0.5 should go higher
    }

    /// Verify optimized highlights layer matches the original apply_sigmoid-based implementation
    #[test]
    fn test_highlights_layer_parity() {
        let midpoint = 0.75f32;
        let range = 0.9f32;
        let highlights: f32 = 5.0; // Test with positive highlights
        let strength = 0.5 + highlights.abs() * 0.1;
        let threshold = 1.0 - range;

        // Test values in the highlight range (> 0.1)
        let test_values = [0.15, 0.3, 0.5, 0.7, 0.85, 0.95];

        for &val in &test_values {
            if val > threshold {
                // Original implementation using apply_sigmoid
                let inv_value = 1.0 - val;
                let expected = 1.0 - apply_sigmoid(strength, midpoint, inv_value, range);

                // Optimized implementation
                let mut pixel = [val, val, val];
                apply_highlights_layer(&mut pixel, highlights);

                let diff = (pixel[0] - expected).abs();
                assert!(
                    diff < 1e-5,
                    "Highlights parity failed for val={}: expected={}, got={}, diff={}",
                    val,
                    expected,
                    pixel[0],
                    diff
                );
            }
        }
    }

    /// Verify optimized shadows layer matches the original apply_sigmoid-based implementation
    #[test]
    fn test_shadows_layer_parity() {
        let midpoint = 0.75f32;
        let range = 0.9f32;
        let shadows: f32 = -5.0; // Test with negative shadows (uses sigmoid)
        let strength = 0.5 + shadows.abs() * 0.1;

        // Test values in the shadow range (< 0.9)
        let test_values = [0.1, 0.3, 0.5, 0.7, 0.85];

        for &val in &test_values {
            if val < range {
                // Original implementation using apply_sigmoid
                let expected = apply_sigmoid(strength, midpoint, val, range);

                // Optimized implementation
                let mut pixel = [val, val, val];
                apply_shadows_layer(&mut pixel, shadows);

                let diff = (pixel[0] - expected).abs();
                assert!(
                    diff < 1e-5,
                    "Shadows parity failed for val={}: expected={}, got={}, diff={}",
                    val,
                    expected,
                    pixel[0],
                    diff
                );
            }
        }
    }

    /// Verify contrast layer optimization matches original
    #[test]
    fn test_contrast_layer_parity() {
        let midpoint = 0.5f32;
        let scale = 0.2f32;
        let contrast: f32 = 5.0;
        let c = 1.0 + contrast * scale;

        let test_values = [0.1, 0.3, 0.5, 0.7, 0.9];

        for &val in &test_values {
            // Original implementation
            let expected = apply_sigmoid(c, midpoint, val, 1.0);

            // Optimized implementation
            let mut pixel = [val, val, val];
            apply_contrast_layer(&mut pixel, contrast);

            let diff = (pixel[0] - expected).abs();
            assert!(
                diff < 1e-5,
                "Contrast parity failed for val={}: expected={}, got={}, diff={}",
                val,
                expected,
                pixel[0],
                diff
            );
        }
    }
}

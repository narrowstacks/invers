//! Image analysis functions for levels adjustment
//!
//! Provides functions for measuring dark/mid/light distribution and applying
//! complete Photoshop-style levels adjustments.

use super::LevelsParams;

/// Full Photoshop-style per-channel levels with gamma and output range
///
/// Provides complete control over:
/// - Per-channel input black/white points
/// - Per-channel gamma correction
/// - Per-channel output black/white points
///
/// This is a comprehensive levels adjustment that goes beyond simple auto_levels.
pub fn apply_levels_complete(
    data: &mut [f32],
    r: &LevelsParams,
    g: &LevelsParams,
    b: &LevelsParams,
) {
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] = apply_levels_single(pixel[0], r);
        pixel[1] = apply_levels_single(pixel[1], g);
        pixel[2] = apply_levels_single(pixel[2], b);
    }
}

/// Apply levels to a single channel value
///
/// Steps:
/// 1. Map input range [input_black, input_white] to [0, 1]
/// 2. Apply gamma correction
/// 3. Map [0, 1] to output range [output_black, output_white]
#[inline]
fn apply_levels_single(value: f32, params: &LevelsParams) -> f32 {
    let input_range = (params.input_white - params.input_black).max(0.0001);

    // Step 1: Normalize to input range
    let normalized = ((value - params.input_black) / input_range).clamp(0.0, 1.0);

    // Step 2: Apply gamma
    let gamma_corrected = if (params.gamma - 1.0).abs() < 0.0001 {
        normalized
    } else {
        normalized.powf(params.gamma)
    };

    // Step 3: Map to output range
    let output_range = params.output_white - params.output_black;
    (params.output_black + gamma_corrected * output_range).clamp(0.0, 1.0)
}

/// Measure dark/mid/light population percentages
///
/// Returns the percentage of pixels in dark (0-0.25), mid (0.25-0.75),
/// and light (0.75-1.0) regions.
///
/// Returns (dark_percent, mid_percent, light_percent)
pub fn measure_dark_mid_light(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut dark = 0usize;
    let mut mid = 0usize;
    let mut light = 0usize;

    for &value in data {
        let v = value.clamp(0.0, 1.0);
        if v < 0.25 {
            dark += 1;
        } else if v < 0.75 {
            mid += 1;
        } else {
            light += 1;
        }
    }

    let total = data.len() as f32;
    (
        dark as f32 / total * 100.0,
        mid as f32 / total * 100.0,
        light as f32 / total * 100.0,
    )
}

/// Measure dark/mid/light for RGB using luminance
pub fn measure_dark_mid_light_rgb(data: &[f32]) -> (f32, f32, f32) {
    let luminances: Vec<f32> = data
        .chunks_exact(3)
        .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        .collect();

    measure_dark_mid_light(&luminances)
}

//! Helper functions for the processing pipeline
//!
//! Contains utility functions for color matrix application, range enforcement,
//! statistics computation, and scan profile adjustments.

use super::{WORKING_RANGE_CEILING, WORKING_RANGE_FLOOR};
use rayon::prelude::*;

/// Apply color correction matrix
/// Performs 3x3 matrix multiplication on RGB pixels
///
/// Uses parallel processing for large images (>100k pixels)
pub fn apply_color_matrix(data: &mut [f32], matrix: &[[f32; 3]; 3], channels: u8) {
    if channels != 3 {
        return; // Only works for RGB
    }

    let num_pixels = data.len() / 3;
    const PARALLEL_THRESHOLD: usize = 100_000; // Use parallelism for >100k pixels

    if num_pixels >= PARALLEL_THRESHOLD {
        // Parallel processing for large images
        // Process in chunks of 256 pixels (768 f32s) for good cache locality
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for pixel in chunk.chunks_exact_mut(3) {
                apply_color_matrix_to_pixel(pixel, matrix);
            }
        });
    } else {
        // Sequential processing for small images
        for pixel in data.chunks_exact_mut(3) {
            apply_color_matrix_to_pixel(pixel, matrix);
        }
    }
}

/// Apply color matrix to a single pixel
///
/// Uses simple clamping to working range to preserve highlight detail
#[inline(always)]
fn apply_color_matrix_to_pixel(pixel: &mut [f32], matrix: &[[f32; 3]; 3]) {
    let r = pixel[0];
    let g = pixel[1];
    let b = pixel[2];

    // Matrix multiplication: output = matrix * input
    // Clamp to working range (no soft-clip to preserve highlight detail)
    pixel[0] = clamp_to_working_range(matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b);
    pixel[1] = clamp_to_working_range(matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b);
    pixel[2] = clamp_to_working_range(matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b);
}

/// Clamp all values into the working range to avoid clipped blacks or whites.
pub fn enforce_working_range(data: &mut [f32]) {
    for value in data.iter_mut() {
        *value = clamp_to_working_range(*value);
    }
}

/// Clamp a single value to the working range
#[inline]
pub(crate) fn clamp_to_working_range(value: f32) -> f32 {
    value.clamp(WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING)
}

/// Compute min, max, and mean statistics for debug output
pub fn compute_stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0;

    for &value in data {
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }

    let mean = sum / data.len() as f32;
    (min, max, mean)
}

/// Apply scan profile gamma and HSL in a single fused pass
///
/// OPTIMIZATION: Combines gamma correction and HSL adjustments into one iteration
/// over the image data, improving cache locality and reducing memory bandwidth.
pub(crate) fn apply_scan_profile_fused(
    data: &mut [f32],
    gamma: Option<[f32; 3]>,
    hsl_adj: Option<&crate::models::HslAdjustments>,
) {
    use crate::color::{hsl_to_rgb, rgb_to_hsl};

    // Pre-compute inverse gamma values if gamma is specified
    let inv_gamma = gamma.map(|g| [1.0 / g[0], 1.0 / g[1], 1.0 / g[2]]);
    let has_gamma = inv_gamma.is_some_and(|g| g != [1.0, 1.0, 1.0]);
    let has_hsl = hsl_adj.is_some_and(|h| h.has_adjustments());

    for pixel in data.chunks_exact_mut(3) {
        // Step 1: Apply gamma correction (if specified)
        if has_gamma {
            let ig = inv_gamma.unwrap();
            pixel[0] = pixel[0].powf(ig[0]);
            pixel[1] = pixel[1].powf(ig[1]);
            pixel[2] = pixel[2].powf(ig[2]);
        }

        // Step 2: Apply HSL adjustments (if specified)
        if has_hsl {
            let adj = hsl_adj.unwrap();
            let hsl = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);

            // Skip achromatic pixels
            if hsl.s >= 0.01 {
                // Get color range weights (inline simplified version)
                let h = ((hsl.h % 360.0) + 360.0) % 360.0;
                let (primary, secondary, blend) = get_hsl_color_weights(h);

                // Calculate weighted adjustments
                let hue_adj = adj.hue[primary] * (1.0 - blend) + adj.hue[secondary] * blend;
                let sat_adj =
                    adj.saturation[primary] * (1.0 - blend) + adj.saturation[secondary] * blend;
                let lum_adj =
                    adj.luminance[primary] * (1.0 - blend) + adj.luminance[secondary] * blend;

                // Apply adjustments
                let mut new_hsl = hsl;
                new_hsl.h = (new_hsl.h + hue_adj * 0.3) % 360.0;
                if new_hsl.h < 0.0 {
                    new_hsl.h += 360.0;
                }
                new_hsl.s = (new_hsl.s * (1.0 + sat_adj / 100.0)).clamp(0.0, 1.0);
                new_hsl.l = (new_hsl.l + lum_adj / 200.0).clamp(0.0, 1.0);

                let (new_r, new_g, new_b) = hsl_to_rgb(new_hsl);
                pixel[0] = new_r;
                pixel[1] = new_g;
                pixel[2] = new_b;
            }
        }
    }
}

/// Get HSL color range weights for 8-color adjustments
/// Returns (primary_index, secondary_index, blend_factor)
#[inline]
pub(crate) fn get_hsl_color_weights(hue: f32) -> (usize, usize, f32) {
    // Hue centers for 8 colors (non-uniform spacing like Camera Raw)
    const CENTERS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 285.0, 315.0];

    let h = ((hue % 360.0) + 360.0) % 360.0;

    // Find primary (nearest center)
    let mut min_dist = f32::MAX;
    let mut primary = 0usize;
    for (i, &center) in CENTERS.iter().enumerate() {
        let diff = (h - center).abs();
        let dist = diff.min(360.0 - diff);
        if dist < min_dist {
            min_dist = dist;
            primary = i;
        }
    }

    // Find secondary (next nearest)
    let mut second_dist = f32::MAX;
    let mut secondary = (primary + 1) % 8;
    for (i, &center) in CENTERS.iter().enumerate() {
        if i != primary {
            let diff = (h - center).abs();
            let dist = diff.min(360.0 - diff);
            if dist < second_dist {
                second_dist = dist;
                secondary = i;
            }
        }
    }

    let total_dist = min_dist + second_dist;
    let blend = if total_dist > 0.0 {
        (min_dist / total_dist).clamp(0.0, 0.5)
    } else {
        0.0
    };

    (primary, secondary, blend)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // compute_stats Tests
    // ========================================================================

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![0.0, 0.5, 1.0, 0.25, 0.75];
        let (min, max, mean) = compute_stats(&data);

        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);
        assert!((mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_empty() {
        let data: Vec<f32> = vec![];
        let (min, max, mean) = compute_stats(&data);

        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_uniform() {
        let data = vec![0.5; 100];
        let (min, max, mean) = compute_stats(&data);

        assert!((min - 0.5).abs() < 0.001);
        assert!((max - 0.5).abs() < 0.001);
        assert!((mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = vec![-0.5, 0.0, 0.5];
        let (min, max, mean) = compute_stats(&data);

        assert!((min - (-0.5)).abs() < 0.001);
        assert!((max - 0.5).abs() < 0.001);
        assert!((mean - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // enforce_working_range Tests
    // ========================================================================

    #[test]
    fn test_enforce_working_range_clamps_below() {
        let mut data = vec![-0.5, -0.1, 0.0];
        enforce_working_range(&mut data);

        for &val in &data {
            assert!(val >= WORKING_RANGE_FLOOR);
        }
    }

    #[test]
    fn test_enforce_working_range_clamps_above() {
        let mut data = vec![1.0, 1.5, 2.0];
        enforce_working_range(&mut data);

        for &val in &data {
            assert!(val <= WORKING_RANGE_CEILING);
        }
    }

    #[test]
    fn test_enforce_working_range_preserves_middle() {
        let mut data = vec![0.3, 0.5, 0.7];
        let original = data.clone();
        enforce_working_range(&mut data);

        for (i, (&orig, &new)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - new).abs() < 0.001,
                "Middle values should be preserved at index {}",
                i
            );
        }
    }

    // ========================================================================
    // apply_color_matrix Tests
    // ========================================================================

    #[test]
    fn test_apply_color_matrix_identity() {
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut data = vec![0.5, 0.3, 0.7];
        let original = data.clone();
        apply_color_matrix(&mut data, &identity, 3);

        for (i, (&orig, &new)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - new).abs() < 0.001,
                "Identity matrix should preserve values at index {}",
                i
            );
        }
    }

    #[test]
    fn test_apply_color_matrix_scale() {
        // Matrix that doubles all values
        let scale: [[f32; 3]; 3] = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];

        let mut data = vec![0.2, 0.3, 0.4];
        apply_color_matrix(&mut data, &scale, 3);

        // Values should be doubled (but clamped to working range)
        assert!(
            (data[0] - 0.4).abs() < 0.001,
            "R should be doubled: {}",
            data[0]
        );
        assert!(
            (data[1] - 0.6).abs() < 0.001,
            "G should be doubled: {}",
            data[1]
        );
        assert!(
            (data[2] - 0.8).abs() < 0.001,
            "B should be doubled: {}",
            data[2]
        );
    }

    #[test]
    fn test_apply_color_matrix_grayscale_conversion() {
        // Matrix that converts to grayscale (luminosity method)
        let grayscale: [[f32; 3]; 3] = [
            [0.2126, 0.7152, 0.0722],
            [0.2126, 0.7152, 0.0722],
            [0.2126, 0.7152, 0.0722],
        ];

        let mut data = vec![1.0, 0.5, 0.0]; // Pure red to gray
        apply_color_matrix(&mut data, &grayscale, 3);

        // All channels should be equal after grayscale conversion
        let tolerance = 0.001;
        assert!(
            (data[0] - data[1]).abs() < tolerance,
            "R and G should be equal"
        );
        assert!(
            (data[1] - data[2]).abs() < tolerance,
            "G and B should be equal"
        );
    }

    #[test]
    fn test_apply_color_matrix_wrong_channels() {
        let identity: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut data = vec![0.5, 0.3, 0.7];
        let original = data.clone();

        // Should do nothing for non-RGB channels
        apply_color_matrix(&mut data, &identity, 4);

        assert_eq!(data, original, "Should not modify data with wrong channels");
    }

    #[test]
    fn test_apply_color_matrix_clamps_output() {
        // Matrix that would produce values > 1.0
        let amplify: [[f32; 3]; 3] = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];

        let mut data = vec![0.5, 0.5, 0.5];
        apply_color_matrix(&mut data, &amplify, 3);

        // Values should be clamped to working range
        for &val in &data {
            assert!(
                val <= WORKING_RANGE_CEILING,
                "Output should be clamped: {}",
                val
            );
        }
    }

    // ========================================================================
    // get_hsl_color_weights Tests
    // ========================================================================

    #[test]
    fn test_hsl_color_weights_red() {
        // Hue 0 = Red (index 0)
        let (primary, _secondary, blend) = get_hsl_color_weights(0.0);
        assert_eq!(primary, 0, "Red hue (0) should have primary index 0");
        assert!(blend <= 0.5, "Blend should be <= 0.5");
    }

    #[test]
    fn test_hsl_color_weights_green() {
        // Hue 120 = Green (index 3)
        let (primary, _secondary, _blend) = get_hsl_color_weights(120.0);
        assert_eq!(primary, 3, "Green hue (120) should have primary index 3");
    }

    #[test]
    fn test_hsl_color_weights_blue() {
        // Hue 240 = Blue (index 5)
        let (primary, _secondary, _blend) = get_hsl_color_weights(240.0);
        assert_eq!(primary, 5, "Blue hue (240) should have primary index 5");
    }

    #[test]
    fn test_hsl_color_weights_wraps_negative() {
        // Negative hue should wrap to positive
        let (primary1, _, _) = get_hsl_color_weights(-30.0);
        let (primary2, _, _) = get_hsl_color_weights(330.0);
        assert_eq!(primary1, primary2, "Negative hue should wrap correctly");
    }

    #[test]
    fn test_hsl_color_weights_wraps_over_360() {
        // Hue > 360 should wrap
        let (primary1, _, _) = get_hsl_color_weights(390.0);
        let (primary2, _, _) = get_hsl_color_weights(30.0);
        assert_eq!(primary1, primary2, "Hue > 360 should wrap correctly");
    }

    // ========================================================================
    // Working Range Constants Tests
    // ========================================================================

    #[test]
    fn test_working_range_constants() {
        // Verify constants are properly defined at compile time
        // These are const assertions expressed as runtime checks for documentation
        const _: () = {
            assert!(WORKING_RANGE_FLOOR > 0.0);
            assert!(WORKING_RANGE_CEILING < 1.0);
        };

        // Runtime verification that clamp_to_working_range uses valid bounds
        assert_eq!(clamp_to_working_range(0.0), WORKING_RANGE_FLOOR);
        assert_eq!(clamp_to_working_range(1.0), WORKING_RANGE_CEILING);
        assert_eq!(clamp_to_working_range(0.5), 0.5);
    }
}

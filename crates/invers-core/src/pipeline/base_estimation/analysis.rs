//! Film base analysis functions
//!
//! Contains functions for analyzing and validating base pixel samples:
//! - Computing statistics from pixel samples
//! - Validating base candidates against expected characteristics
//! - Detecting B&W vs color film

use crate::config;
use crate::verbose_println;

use super::{
    BASE_VALIDATION_MAX_NOISE, BASE_VALIDATION_MIN_BRIGHTNESS, MAX_BASE_SAMPLE_FRACTION,
    MIN_BASE_SAMPLE_FRACTION,
};

/// Get the configured base sample fraction, clamped to valid range
pub(crate) fn base_sample_fraction() -> f32 {
    let defaults = &config::pipeline_config_handle().config.defaults;
    let fraction = defaults.base_brightest_percent / 100.0;
    fraction.clamp(MIN_BASE_SAMPLE_FRACTION, MAX_BASE_SAMPLE_FRACTION)
}

/// Compute statistics from ROI pixels for base estimation
///
/// Returns: (num_brightest, percentage, medians, noise_stats)
pub(crate) fn compute_base_stats(
    roi_pixels: &[[f32; 3]],
    fraction: f32,
) -> (usize, f32, [f32; 3], [f32; 3]) {
    let mut num_brightest = (roi_pixels.len() as f32 * fraction).ceil() as usize;
    num_brightest = num_brightest.max(10).min(roi_pixels.len());
    let percentage =
        ((num_brightest as f32 / roi_pixels.len() as f32) * 100.0 * 10.0).round() / 10.0;

    let medians = compute_channel_medians_from_brightest(roi_pixels, num_brightest);
    let noise_stats = compute_noise_stats(roi_pixels, &medians);

    (num_brightest, percentage, medians, noise_stats)
}

/// Detect if the image appears to be B&W based on channel similarity
pub(crate) fn is_likely_bw(medians: &[f32; 3]) -> bool {
    let [r, g, b] = *medians;
    let avg = (r + g + b) / 3.0;
    if avg <= 0.0 {
        return false;
    }
    // Check if all channels are within 15% of the average (low chroma)
    let deviation_threshold = 0.15;
    let r_dev = (r - avg).abs() / avg;
    let g_dev = (g - avg).abs() / avg;
    let b_dev = (b - avg).abs() / avg;
    r_dev < deviation_threshold && g_dev < deviation_threshold && b_dev < deviation_threshold
}

/// Validate a base candidate against expected film base characteristics
///
/// Returns: (is_valid, reason_string)
pub(crate) fn validate_base_candidate(
    medians: &[f32; 3],
    noise: &[f32; 3],
    brightness: f32,
) -> (bool, String) {
    let [r, g, b] = *medians;
    let max_noise = noise.iter().cloned().fold(0.0, f32::max);

    if brightness < BASE_VALIDATION_MIN_BRIGHTNESS {
        return (
            false,
            format!(
                "brightness {:.3} < {:.3}",
                brightness, BASE_VALIDATION_MIN_BRIGHTNESS
            ),
        );
    }

    // Adaptive noise threshold: scale based on brightness
    // Higher brightness regions tend to have more visible noise in scans
    let adaptive_noise_threshold = BASE_VALIDATION_MAX_NOISE * (1.0 + brightness * 0.5);
    if max_noise > adaptive_noise_threshold {
        return (
            false,
            format!(
                "noise {:.4} exceeds adaptive threshold {:.4}",
                max_noise, adaptive_noise_threshold
            ),
        );
    }

    if !(r.is_finite() && g.is_finite() && b.is_finite()) {
        return (false, "median contains non-finite values".to_string());
    }

    if r <= 0.0 || g <= 0.0 || b <= 0.0 {
        return (false, "median channel <= 0".to_string());
    }

    // Check if this appears to be B&W film
    let is_bw = is_likely_bw(medians);
    if is_bw {
        verbose_println!(
            "[BASE]   detected B&W film (low chroma), skipping orange mask validation"
        );
        return (true, "B&W film - all channels similar".to_string());
    }

    // Color film: validate orange mask characteristics
    let rg_ratio = r / g;
    let gb_ratio = g / b;

    if !(0.70..=2.20).contains(&rg_ratio) {
        return (
            false,
            format!("R/G ratio {:.3} outside expected range 0.70-2.20", rg_ratio),
        );
    }

    if !(1.00..=2.50).contains(&gb_ratio) {
        return (
            false,
            format!("G/B ratio {:.3} outside expected range 1.00-2.50", gb_ratio),
        );
    }

    if r < g || g < b {
        return (
            false,
            "channel ordering not orange-mask like (R >= G >= B expected)".to_string(),
        );
    }

    (true, "within expected range".to_string())
}

/// Compute per-channel medians from the brightest N pixels
/// This samples the clearest film base without image content
///
/// For color negative film, we filter for pixels that match the orange mask
/// characteristics (R > G > B with typical G/B ratio 1.3-2.0) before selecting
/// the brightest pixels. This prevents scanner artifacts or edge effects with
/// elevated blue from skewing the base estimation.
pub(crate) fn compute_channel_medians_from_brightest(
    pixels: &[[f32; 3]],
    num_pixels: usize,
) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    // First, filter for pixels that match orange mask characteristics
    // Orange mask: R > G > B, with G/B ratio typically 1.3-2.0
    let orange_mask_pixels: Vec<[f32; 3]> = pixels
        .iter()
        .filter(|p| {
            let [r, g, b] = **p;
            // Require minimum brightness in each channel
            if r < 0.3 || g < 0.2 || b < 0.1 {
                return false;
            }
            // Require orange mask channel ordering
            if !(r > g && g > b) {
                return false;
            }
            // Check G/B ratio is in typical orange mask range
            // Wider range (1.15-2.5) to include more valid film bases
            // while still excluding obvious non-film pixels
            let gb_ratio = g / b;
            (1.15..=2.5).contains(&gb_ratio)
        })
        .copied()
        .collect();

    // If we have enough orange-mask pixels, use those; otherwise fall back to all pixels
    let working_pixels = if orange_mask_pixels.len() >= num_pixels.min(100) {
        verbose_println!(
            "[BASE]   filtered {} of {} pixels as orange-mask-like",
            orange_mask_pixels.len(),
            pixels.len()
        );
        orange_mask_pixels
    } else {
        verbose_println!(
            "[BASE]   only {} orange-mask pixels found, using all {} pixels",
            orange_mask_pixels.len(),
            pixels.len()
        );
        pixels.to_vec()
    };

    // Create a vec of (brightness, pixel) tuples
    let mut brightness_pixels: Vec<(f32, [f32; 3])> = working_pixels
        .iter()
        .map(|p| {
            let brightness = p[0] + p[1] + p[2]; // Sum of RGB as brightness
            (brightness, *p)
        })
        .collect();

    // Use partial sort to find top N brightest pixels (much faster than full sort)
    let n = num_pixels.min(brightness_pixels.len());
    let threshold_idx = brightness_pixels.len().saturating_sub(n);
    brightness_pixels.select_nth_unstable_by(threshold_idx, |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    // The brightest N pixels are now in the last n positions
    let brightest_slice = &brightness_pixels[threshold_idx..];

    // Compute median for each channel from these brightest pixels
    // Pre-allocate with exact capacity for efficiency
    let mut r_values: Vec<f32> = Vec::with_capacity(n);
    let mut g_values: Vec<f32> = Vec::with_capacity(n);
    let mut b_values: Vec<f32> = Vec::with_capacity(n);

    for (_, pixel) in brightest_slice {
        r_values.push(pixel[0]);
        g_values.push(pixel[1]);
        b_values.push(pixel[2]);
    }

    [
        compute_median(&mut r_values),
        compute_median(&mut g_values),
        compute_median(&mut b_values),
    ]
}

/// Compute median of a slice using partial sorting (much faster than full sort)
pub(crate) fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let len = values.len();
    let mid = len / 2;

    if len.is_multiple_of(2) {
        // Even length: average of two middle values
        // Use select_nth_unstable to partially sort only what we need
        values.select_nth_unstable_by(mid - 1, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let lower = values[mid - 1];
        values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let upper = values[mid];
        (lower + upper) / 2.0
    } else {
        // Odd length: middle value
        values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        values[mid]
    }
}

/// Compute noise statistics (standard deviation per channel)
pub(crate) fn compute_noise_stats(pixels: &[[f32; 3]], medians: &[f32; 3]) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = pixels.len() as f32;

    // Compute variance for each channel
    let mut var_r = 0.0;
    let mut var_g = 0.0;
    let mut var_b = 0.0;

    for pixel in pixels {
        var_r += (pixel[0] - medians[0]).powi(2);
        var_g += (pixel[1] - medians[1]).powi(2);
        var_b += (pixel[2] - medians[2]).powi(2);
    }

    var_r /= n;
    var_g /= n;
    var_b /= n;

    // Return standard deviations
    [var_r.sqrt(), var_g.sqrt(), var_b.sqrt()]
}

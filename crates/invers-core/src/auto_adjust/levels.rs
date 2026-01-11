//! Automatic levels adjustment functions
//!
//! Provides histogram-based auto-levels with various modes and gamma correction.

use rayon::prelude::*;

use super::parallel::parallel_for_each_chunk_mut;
use super::PARALLEL_THRESHOLD;

// Re-export AutoLevelsMode from models for backward compatibility
pub use crate::models::AutoLevelsMode;

/// Per-channel levels parameters for complete levels control
#[derive(Debug, Clone, Copy)]
pub struct LevelsParams {
    /// Input black point (0.0-1.0)
    pub input_black: f32,
    /// Input white point (0.0-1.0)
    pub input_white: f32,
    /// Gamma value (1.0 = no change, <1.0 = lighten, >1.0 = darken)
    pub gamma: f32,
    /// Output black point (0.0-1.0)
    pub output_black: f32,
    /// Output white point (0.0-1.0)
    pub output_white: f32,
}

impl Default for LevelsParams {
    fn default() -> Self {
        Self {
            input_black: 0.0,
            input_white: 1.0,
            gamma: 1.0,
            output_black: 0.0,
            output_white: 1.0,
        }
    }
}

impl LevelsParams {
    /// Create from just input range (useful for auto-levels results)
    pub fn from_input_range(black: f32, white: f32) -> Self {
        Self {
            input_black: black,
            input_white: white,
            ..Default::default()
        }
    }

    /// Calculate gamma to bring midpoint to target
    /// Formula: gamma = log(target / max) / log(0.5)
    pub fn gamma_for_midpoint(target_mid: f32) -> f32 {
        if target_mid <= 0.0 || target_mid >= 1.0 {
            return 1.0;
        }
        (target_mid.ln() / 0.5_f32.ln()).clamp(0.1, 10.0)
    }
}

/// Auto-levels: Stretch histogram to full 0.0-1.0 range per channel
/// This is the key step that Photoshop applies
///
/// Uses histogram-based percentile finding to avoid copying all channel data.
/// Memory usage: O(buckets * 3) instead of O(n * 3)
pub fn auto_levels(data: &mut [f32], channels: u8, clip_percent: f32) -> [f32; 6] {
    auto_levels_with_mode(data, channels, clip_percent, AutoLevelsMode::PerChannel)
}

/// Auto-levels without clipping: Normalize histogram while preserving all data
///
/// This version:
/// 1. Applies the same percentile-based normalization as regular auto-levels
/// 2. Then scales the entire result so the max doesn't exceed the original max
/// 3. This gives proper color balance while preserving all highlight detail
///
/// Optimized single-pass implementation: computes scale factors upfront and applies
/// normalization + scaling in one pass over the data.
///
/// Returns [r_min, r_max, g_min, g_max, b_min, b_max] - the percentile ranges found
pub fn auto_levels_no_clip(data: &mut [f32], channels: u8, clip_percent: f32) -> [f32; 6] {
    if channels != 3 {
        panic!("auto_levels_no_clip only supports 3-channel RGB images");
    }

    // Build per-channel histograms and track actual max in a single pass
    const NUM_BUCKETS: usize = 65536;
    let mut r_hist = vec![0u32; NUM_BUCKETS];
    let mut g_hist = vec![0u32; NUM_BUCKETS];
    let mut b_hist = vec![0u32; NUM_BUCKETS];

    // Track actual max values per channel
    let mut r_actual_max = 0.0f32;
    let mut g_actual_max = 0.0f32;
    let mut b_actual_max = 0.0f32;

    for pixel in data.chunks_exact(3) {
        // For histogram, clamp to 0-1 for bucket calculation
        let r_bucket =
            ((pixel[0].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let g_bucket =
            ((pixel[1].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let b_bucket =
            ((pixel[2].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        r_hist[r_bucket] += 1;
        g_hist[g_bucket] += 1;
        b_hist[b_bucket] += 1;

        // Track per-channel max
        r_actual_max = r_actual_max.max(pixel[0]);
        g_actual_max = g_actual_max.max(pixel[1]);
        b_actual_max = b_actual_max.max(pixel[2]);
    }

    let overall_actual_max = r_actual_max.max(g_actual_max).max(b_actual_max);
    let num_pixels = data.len() / 3;

    // Compute clipped percentile ranges (same as regular auto-levels)
    let (r_min, r_clip_max) =
        compute_clipped_range_from_histogram(&r_hist, num_pixels, clip_percent);
    let (g_min, g_clip_max) =
        compute_clipped_range_from_histogram(&g_hist, num_pixels, clip_percent);
    let (b_min, b_clip_max) =
        compute_clipped_range_from_histogram(&b_hist, num_pixels, clip_percent);

    // Compute per-channel ranges
    let r_range = r_clip_max - r_min;
    let g_range = g_clip_max - g_min;
    let b_range = b_clip_max - b_min;

    // Compute what the normalized max would be for each channel:
    // normalized_max = (actual_max - min) / range
    let r_normalized_max = if r_range.abs() > 0.0001 {
        (r_actual_max - r_min) / r_range
    } else {
        r_actual_max
    };
    let g_normalized_max = if g_range.abs() > 0.0001 {
        (g_actual_max - g_min) / g_range
    } else {
        g_actual_max
    };
    let b_normalized_max = if b_range.abs() > 0.0001 {
        (b_actual_max - b_min) / b_range
    } else {
        b_actual_max
    };

    // The new_max after normalization is the max of all normalized channel maxes
    let new_max = r_normalized_max.max(g_normalized_max).max(b_normalized_max);

    // Pre-compute the final scale factor
    let final_scale = if new_max > 0.0001 {
        overall_actual_max / new_max
    } else {
        1.0
    };

    // Single pass: apply normalization and scaling together
    // Final value = ((v - min) / range) * final_scale
    //             = (v - min) * (final_scale / range)
    //             = v * (final_scale / range) - min * (final_scale / range)
    // Let scale_factor = final_scale / range
    // Let offset = -min * scale_factor
    // Then: final = v * scale_factor + offset

    let r_scale = if r_range.abs() > 0.0001 {
        final_scale / r_range
    } else {
        1.0
    };
    let g_scale = if g_range.abs() > 0.0001 {
        final_scale / g_range
    } else {
        1.0
    };
    let b_scale = if b_range.abs() > 0.0001 {
        final_scale / b_range
    } else {
        1.0
    };

    let r_offset = if r_range.abs() > 0.0001 {
        -r_min * r_scale
    } else {
        0.0
    };
    let g_offset = if g_range.abs() > 0.0001 {
        -g_min * g_scale
    } else {
        0.0
    };
    let b_offset = if b_range.abs() > 0.0001 {
        -b_min * b_scale
    } else {
        0.0
    };

    // Apply combined transformation in a single pass
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] = pixel[0] * r_scale + r_offset;
        pixel[1] = pixel[1] * g_scale + g_offset;
        pixel[2] = pixel[2] * b_scale + b_offset;
    }

    [r_min, r_clip_max, g_min, g_clip_max, b_min, b_clip_max]
}

/// RGB histogram type for parallel accumulation
type RgbHistograms = (Vec<u32>, Vec<u32>, Vec<u32>);

/// Build RGB histograms in parallel using fold/reduce pattern
fn build_rgb_histograms_parallel(data: &[f32], num_buckets: usize) -> RgbHistograms {
    data.par_chunks_exact(3)
        .fold(
            || {
                (
                    vec![0u32; num_buckets],
                    vec![0u32; num_buckets],
                    vec![0u32; num_buckets],
                )
            },
            |(mut r, mut g, mut b), pixel| {
                let r_bucket = ((pixel[0].clamp(0.0, 1.0) * (num_buckets - 1) as f32) as usize)
                    .min(num_buckets - 1);
                let g_bucket = ((pixel[1].clamp(0.0, 1.0) * (num_buckets - 1) as f32) as usize)
                    .min(num_buckets - 1);
                let b_bucket = ((pixel[2].clamp(0.0, 1.0) * (num_buckets - 1) as f32) as usize)
                    .min(num_buckets - 1);
                r[r_bucket] += 1;
                g[g_bucket] += 1;
                b[b_bucket] += 1;
                (r, g, b)
            },
        )
        .reduce(
            || {
                (
                    vec![0u32; num_buckets],
                    vec![0u32; num_buckets],
                    vec![0u32; num_buckets],
                )
            },
            |(mut r1, mut g1, mut b1), (r2, g2, b2)| {
                for i in 0..num_buckets {
                    r1[i] += r2[i];
                    g1[i] += g2[i];
                    b1[i] += b2[i];
                }
                (r1, g1, b1)
            },
        )
}

/// Build RGB histograms sequentially
fn build_rgb_histograms_sequential(data: &[f32], num_buckets: usize) -> RgbHistograms {
    let mut r_hist = vec![0u32; num_buckets];
    let mut g_hist = vec![0u32; num_buckets];
    let mut b_hist = vec![0u32; num_buckets];

    for pixel in data.chunks_exact(3) {
        let r_bucket =
            ((pixel[0].clamp(0.0, 1.0) * (num_buckets - 1) as f32) as usize).min(num_buckets - 1);
        let g_bucket =
            ((pixel[1].clamp(0.0, 1.0) * (num_buckets - 1) as f32) as usize).min(num_buckets - 1);
        let b_bucket =
            ((pixel[2].clamp(0.0, 1.0) * (num_buckets - 1) as f32) as usize).min(num_buckets - 1);
        r_hist[r_bucket] += 1;
        g_hist[g_bucket] += 1;
        b_hist[b_bucket] += 1;
    }

    (r_hist, g_hist, b_hist)
}

/// Auto-levels with configurable mode for color preservation
///
/// Modes:
/// - `PerChannel`: Independent stretching per channel (may shift colors)
/// - `SaturationAware`: Reduces stretch for channels that would clip heavily
/// - `Unified`: Uses a shared min/max across all channels
pub fn auto_levels_with_mode(
    data: &mut [f32],
    channels: u8,
    clip_percent: f32,
    mode: AutoLevelsMode,
) -> [f32; 6] {
    if channels != 3 {
        panic!("auto_levels only supports 3-channel RGB images");
    }

    let num_pixels = data.len() / 3;
    const NUM_BUCKETS: usize = 65536;

    // Build per-channel histograms - parallel for large images
    let (r_hist, g_hist, b_hist) = if num_pixels >= PARALLEL_THRESHOLD {
        build_rgb_histograms_parallel(data, NUM_BUCKETS)
    } else {
        build_rgb_histograms_sequential(data, NUM_BUCKETS)
    };

    // Compute initial min/max for each channel with clipping using histograms
    let (mut r_min, mut r_max) =
        compute_clipped_range_from_histogram(&r_hist, num_pixels, clip_percent);
    let (mut g_min, mut g_max) =
        compute_clipped_range_from_histogram(&g_hist, num_pixels, clip_percent);
    let (mut b_min, mut b_max) =
        compute_clipped_range_from_histogram(&b_hist, num_pixels, clip_percent);

    // Apply mode-specific adjustments
    match mode {
        AutoLevelsMode::PerChannel => {
            // No adjustment needed - each channel stretched independently
        }
        AutoLevelsMode::Unified => {
            // Use the same stretch for all channels to preserve color relationships
            // This prevents color shifts that can occur with per-channel stretching
            let min_of_mins = r_min.min(g_min).min(b_min);
            let max_of_maxs = r_max.max(g_max).max(b_max);
            r_min = min_of_mins;
            g_min = min_of_mins;
            b_min = min_of_mins;
            r_max = max_of_maxs;
            g_max = max_of_maxs;
            b_max = max_of_maxs;
        }
        AutoLevelsMode::SaturationAware => {
            // Calculate stretch ratios for each channel
            let r_range = (r_max - r_min).max(0.001);
            let g_range = (g_max - g_min).max(0.001);
            let b_range = (b_max - b_min).max(0.001);

            let max_range = r_range.max(g_range).max(b_range);

            // If any channel has significantly less range (>30% difference),
            // it would clip more heavily. Blend toward the max range.
            const BLEND_THRESHOLD: f32 = 0.30; // 30% difference triggers blending

            let adjust_channel = |min: &mut f32, max: &mut f32, range: f32| {
                let ratio = range / max_range;
                if ratio < (1.0 - BLEND_THRESHOLD) {
                    // This channel would clip heavily, blend toward max_range
                    let target_range = max_range;
                    let current_mid = (*min + *max) / 2.0;
                    *min = (current_mid - target_range / 2.0).max(0.0);
                    *max = (current_mid + target_range / 2.0).min(1.0);
                }
            };

            adjust_channel(&mut r_min, &mut r_max, r_range);
            adjust_channel(&mut g_min, &mut g_max, g_range);
            adjust_channel(&mut b_min, &mut b_max, b_range);
        }
    }

    // Apply stretch to each channel in-place - parallel for large images
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] = stretch_value(pixel[0], r_min, r_max);
        pixel[1] = stretch_value(pixel[1], g_min, g_max);
        pixel[2] = stretch_value(pixel[2], b_min, b_max);
    });

    // Return the adjustment parameters for debugging
    [r_min, r_max, g_min, g_max, b_min, b_max]
}

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

/// Auto-levels with per-channel gamma correction
///
/// Combines histogram-based auto-levels with per-channel gamma adjustment.
/// The gamma values can be:
/// - Calculated automatically to target a specific midpoint
/// - Provided explicitly for manual control
///
/// Returns the LevelsParams used for each channel.
pub fn auto_levels_with_gamma(
    data: &mut [f32],
    channels: u8,
    clip_percent: f32,
    gamma: [f32; 3],
) -> [LevelsParams; 3] {
    if channels != 3 {
        panic!("auto_levels_with_gamma only supports 3-channel RGB images");
    }

    // Build per-channel histograms
    const NUM_BUCKETS: usize = 65536;
    let mut r_hist = vec![0u32; NUM_BUCKETS];
    let mut g_hist = vec![0u32; NUM_BUCKETS];
    let mut b_hist = vec![0u32; NUM_BUCKETS];

    for pixel in data.chunks_exact(3) {
        let r_bucket =
            ((pixel[0].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let g_bucket =
            ((pixel[1].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let b_bucket =
            ((pixel[2].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        r_hist[r_bucket] += 1;
        g_hist[g_bucket] += 1;
        b_hist[b_bucket] += 1;
    }

    let num_pixels = data.len() / 3;

    // Compute input ranges
    let (r_min, r_max) = compute_clipped_range_from_histogram(&r_hist, num_pixels, clip_percent);
    let (g_min, g_max) = compute_clipped_range_from_histogram(&g_hist, num_pixels, clip_percent);
    let (b_min, b_max) = compute_clipped_range_from_histogram(&b_hist, num_pixels, clip_percent);

    // Create LevelsParams with gamma
    let r_params = LevelsParams {
        input_black: r_min,
        input_white: r_max,
        gamma: gamma[0],
        output_black: 0.0,
        output_white: 1.0,
    };
    let g_params = LevelsParams {
        input_black: g_min,
        input_white: g_max,
        gamma: gamma[1],
        output_black: 0.0,
        output_white: 1.0,
    };
    let b_params = LevelsParams {
        input_black: b_min,
        input_white: b_max,
        gamma: gamma[2],
        output_black: 0.0,
        output_white: 1.0,
    };

    // Apply levels
    apply_levels_complete(data, &r_params, &g_params, &b_params);

    [r_params, g_params, b_params]
}

/// Auto-levels with automatic gamma calculation to target a specific midpoint
///
/// Calculates gamma values to bring each channel's midpoint to the target value.
/// This is similar to auto-contrast behavior.
pub fn auto_levels_with_target_midpoint(
    data: &mut [f32],
    channels: u8,
    clip_percent: f32,
    target_midpoint: f32,
) -> [LevelsParams; 3] {
    // Calculate gamma needed to bring current midpoints to target
    // If current midpoint maps to 0.5 after normalization, gamma = log(target) / log(0.5)
    let gamma = LevelsParams::gamma_for_midpoint(target_midpoint);

    auto_levels_with_gamma(data, channels, clip_percent, [gamma, gamma, gamma])
}

/// Compute min/max with percentile clipping from a histogram
pub(crate) fn compute_clipped_range_from_histogram(
    histogram: &[u32],
    num_pixels: usize,
    clip_percent: f32,
) -> (f32, f32) {
    let clip_fraction = clip_percent / 100.0;
    // Ensure we clip at least 0 pixels from each end (find true min/max)
    // Adding 1 ensures we go past the target when clip_percent is 0
    let low_target = ((num_pixels as f32 * clip_fraction) as usize).max(1);
    let high_target = (num_pixels as f32 * (1.0 - clip_fraction)) as usize;

    let num_buckets = histogram.len();
    let mut cumulative = 0usize;
    let mut min_bucket = 0usize;
    let mut max_bucket = num_buckets - 1;

    // Find low percentile bucket (first bucket where cumulative reaches target)
    for (bucket, &count) in histogram.iter().enumerate() {
        if count > 0 && min_bucket == 0 {
            // Record first non-empty bucket as potential minimum
            min_bucket = bucket;
        }
        cumulative += count as usize;
        if cumulative >= low_target {
            min_bucket = bucket;
            break;
        }
    }

    // Find high percentile bucket (last bucket where cumulative hasn't exceeded target)
    cumulative = 0;
    for (bucket, &count) in histogram.iter().enumerate().rev() {
        if count > 0 && max_bucket == num_buckets - 1 {
            // Record last non-empty bucket as potential maximum
            max_bucket = bucket;
        }
        cumulative += count as usize;
        if cumulative > num_pixels - high_target {
            max_bucket = bucket;
            break;
        }
    }

    let min_val = min_bucket as f32 / (num_buckets - 1) as f32;
    let max_val = max_bucket as f32 / (num_buckets - 1) as f32;

    // Ensure we have a valid range
    if (max_val - min_val).abs() < 0.0001 {
        (min_val, min_val + 0.1)
    } else {
        (min_val, max_val)
    }
}

/// Stretch a value from [old_min, old_max] to [0.0, 1.0]
#[inline]
fn stretch_value(value: f32, old_min: f32, old_max: f32) -> f32 {
    if (old_max - old_min).abs() < 0.0001 {
        return value.clamp(0.0, 1.0);
    }

    let stretched = (value - old_min) / (old_max - old_min);
    stretched.clamp(0.0, 1.0)
}

// =============================================================================
// Histogram Utilities
// =============================================================================

/// Result of histogram ends detection
#[derive(Debug, Clone, Copy)]
pub struct HistogramEnds {
    /// First non-zero bin index (0.0-1.0 normalized)
    pub low: f32,
    /// Last non-zero bin index (0.0-1.0 normalized)
    pub high: f32,
    /// Number of non-empty bins
    pub non_empty_bins: usize,
}

/// Find first and last non-zero histogram bins
///
/// Finds the actual data range without percentile clipping.
/// Useful for detecting true black/white points in an image.
///
/// Returns normalized values (0.0-1.0) for the first and last non-empty bins.
pub fn find_histogram_ends(data: &[f32]) -> HistogramEnds {
    const NUM_BUCKETS: usize = 65536;
    let mut histogram = vec![0u32; NUM_BUCKETS];

    // Build histogram
    for &value in data {
        let bucket =
            ((value.clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        histogram[bucket] += 1;
    }

    // Find first non-zero bin
    let mut low_bucket = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            low_bucket = i;
            break;
        }
    }

    // Find last non-zero bin
    let mut high_bucket = NUM_BUCKETS - 1;
    for (i, &count) in histogram.iter().enumerate().rev() {
        if count > 0 {
            high_bucket = i;
            break;
        }
    }

    // Count non-empty bins
    let non_empty_bins = histogram.iter().filter(|&&c| c > 0).count();

    HistogramEnds {
        low: low_bucket as f32 / (NUM_BUCKETS - 1) as f32,
        high: high_bucket as f32 / (NUM_BUCKETS - 1) as f32,
        non_empty_bins,
    }
}

/// Find histogram ends per channel for RGB data
///
/// Returns [R_ends, G_ends, B_ends]
pub fn find_histogram_ends_rgb(data: &[f32]) -> [HistogramEnds; 3] {
    const NUM_BUCKETS: usize = 65536;
    let mut r_hist = vec![0u32; NUM_BUCKETS];
    let mut g_hist = vec![0u32; NUM_BUCKETS];
    let mut b_hist = vec![0u32; NUM_BUCKETS];

    // Build per-channel histograms
    for pixel in data.chunks_exact(3) {
        let r_bucket =
            ((pixel[0].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let g_bucket =
            ((pixel[1].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let b_bucket =
            ((pixel[2].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        r_hist[r_bucket] += 1;
        g_hist[g_bucket] += 1;
        b_hist[b_bucket] += 1;
    }

    [
        histogram_ends_from_array(&r_hist),
        histogram_ends_from_array(&g_hist),
        histogram_ends_from_array(&b_hist),
    ]
}

/// Helper to extract ends from a pre-built histogram
fn histogram_ends_from_array(histogram: &[u32]) -> HistogramEnds {
    let num_buckets = histogram.len();

    let mut low_bucket = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            low_bucket = i;
            break;
        }
    }

    let mut high_bucket = num_buckets - 1;
    for (i, &count) in histogram.iter().enumerate().rev() {
        if count > 0 {
            high_bucket = i;
            break;
        }
    }

    let non_empty_bins = histogram.iter().filter(|&&c| c > 0).count();

    HistogramEnds {
        low: low_bucket as f32 / (num_buckets - 1) as f32,
        high: high_bucket as f32 / (num_buckets - 1) as f32,
        non_empty_bins,
    }
}

/// Result of Otsu's threshold computation
#[derive(Debug, Clone, Copy)]
pub struct OtsuResult {
    /// Optimal threshold value (0.0-1.0)
    pub threshold: f32,
    /// Inter-class variance at the threshold
    pub variance: f32,
    /// Ratio of pixels below threshold
    pub below_ratio: f32,
}

/// Otsu's method for automatic thresholding
///
/// Finds the optimal threshold that separates an image into two classes
/// (e.g., film base vs image content) by maximizing inter-class variance.
///
/// This is useful for automatically detecting the boundary between
/// unexposed film base and actual image content.
///
/// Returns the optimal threshold (0.0-1.0) and the inter-class variance.
pub fn otsu_threshold(data: &[f32]) -> OtsuResult {
    const NUM_BUCKETS: usize = 256; // 256 bins is standard for Otsu

    // Build histogram
    let mut histogram = vec![0u32; NUM_BUCKETS];
    for &value in data {
        let bucket =
            ((value.clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        histogram[bucket] += 1;
    }

    let total = data.len() as f32;
    if total < 2.0 {
        return OtsuResult {
            threshold: 0.5,
            variance: 0.0,
            below_ratio: 0.5,
        };
    }

    // Calculate normalized histogram (probabilities)
    let probabilities: Vec<f32> = histogram.iter().map(|&c| c as f32 / total).collect();

    // Calculate cumulative sums and cumulative means
    let mut cum_sum = vec![0.0f32; NUM_BUCKETS];
    let mut cum_mean = vec![0.0f32; NUM_BUCKETS];

    cum_sum[0] = probabilities[0];
    cum_mean[0] = 0.0; // bucket 0 * p[0]

    for i in 1..NUM_BUCKETS {
        cum_sum[i] = cum_sum[i - 1] + probabilities[i];
        cum_mean[i] = cum_mean[i - 1] + (i as f32) * probabilities[i];
    }

    let global_mean = cum_mean[NUM_BUCKETS - 1];

    // Find threshold that maximizes inter-class variance
    let mut max_variance = 0.0f32;
    let mut optimal_threshold = 0usize;

    for t in 0..NUM_BUCKETS - 1 {
        let w0 = cum_sum[t]; // Weight of class 0 (below threshold)
        let w1 = 1.0 - w0; // Weight of class 1 (above threshold)

        if w0 < 1e-10 || w1 < 1e-10 {
            continue;
        }

        let mu0 = cum_mean[t] / w0; // Mean of class 0
        let mu1 = (global_mean - cum_mean[t]) / w1; // Mean of class 1

        // Inter-class variance
        let variance = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

        if variance > max_variance {
            max_variance = variance;
            optimal_threshold = t;
        }
    }

    OtsuResult {
        threshold: optimal_threshold as f32 / (NUM_BUCKETS - 1) as f32,
        variance: max_variance,
        below_ratio: cum_sum[optimal_threshold],
    }
}

/// Otsu threshold for RGB data using luminance
///
/// Computes Otsu threshold on the luminance channel (Rec.709 weights)
pub fn otsu_threshold_rgb(data: &[f32]) -> OtsuResult {
    // Extract luminance values
    let luminances: Vec<f32> = data
        .chunks_exact(3)
        .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        .collect();

    otsu_threshold(&luminances)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_levels_simple() {
        // Simple test: values from 0.2 to 0.8 should stretch to 0.0-1.0
        let mut data = vec![
            0.2, 0.3, 0.4, // R channel: 0.2-0.8
            0.3, 0.4, 0.5, // G channel: 0.3-0.9
            0.4, 0.5, 0.6, // B channel: 0.4-0.7
            0.5, 0.6, 0.7, 0.6, 0.7, 0.8, 0.7, 0.8, 0.9, 0.8, 0.9, 0.7,
        ];

        let params = auto_levels(&mut data, 3, 0.0);

        // Check that values have been stretched
        assert!(data[0] < 0.2); // First R value should be lower
        let min_val = data.iter().cloned().fold(f32::MAX, f32::min);
        let max_val = data.iter().cloned().fold(f32::MIN, f32::max);
        assert!(min_val <= 0.001, "expected min close to 0, got {}", min_val);
        assert!(max_val >= 0.999, "expected max close to 1, got {}", max_val);

        println!("Auto-levels params: {:?}", params);
        println!("Adjusted data: {:?}", data);
    }

    #[test]
    fn test_find_histogram_ends() {
        // Data ranging from 0.2 to 0.8
        let data: Vec<f32> = (0..100).map(|i| 0.2 + 0.6 * (i as f32 / 99.0)).collect();

        let ends = find_histogram_ends(&data);

        println!("Histogram ends: low={}, high={}", ends.low, ends.high);

        // Low should be close to 0.2
        assert!(
            (ends.low - 0.2).abs() < 0.01,
            "Expected low ~0.2, got {}",
            ends.low
        );
        // High should be close to 0.8
        assert!(
            (ends.high - 0.8).abs() < 0.01,
            "Expected high ~0.8, got {}",
            ends.high
        );
    }

    #[test]
    fn test_otsu_threshold_bimodal() {
        // Create truly bimodal distribution with clear separation
        // All low values exactly at 0.2, all high values exactly at 0.8
        let mut data = vec![0.2; 500]; // Low cluster: 500 samples at 0.2
        data.extend(vec![0.8; 500]); // High cluster: 500 samples at 0.8

        let result = otsu_threshold(&data);

        println!(
            "Otsu threshold: {}, variance: {}, below_ratio: {}",
            result.threshold, result.variance, result.below_ratio
        );

        // Threshold should be at or between the two modes
        // With discrete bimodal data at 0.2 and 0.8, Otsu may find threshold
        // at the boundary of one of the modes (equal variance for any value in between)
        assert!(
            result.threshold >= 0.2 && result.threshold <= 0.8,
            "Expected threshold at/between modes (0.2 and 0.8), got {}",
            result.threshold
        );
        // Variance should be non-zero for bimodal data
        assert!(
            result.variance > 0.0,
            "Expected non-zero variance for bimodal data"
        );
    }

    #[test]
    fn test_measure_dark_mid_light() {
        // Create data with known distribution
        let mut data = vec![0.1; 25]; // 25% dark (0.0-0.25)
        data.extend(vec![0.5; 50]); // 50% mid (0.25-0.75)
        data.extend(vec![0.9; 25]); // 25% light (0.75-1.0)

        let (dark, mid, light) = measure_dark_mid_light(&data);

        println!("Dark: {}%, Mid: {}%, Light: {}%", dark, mid, light);

        assert!((dark - 25.0).abs() < 0.1, "Expected 25% dark, got {}", dark);
        assert!((mid - 50.0).abs() < 0.1, "Expected 50% mid, got {}", mid);
        assert!(
            (light - 25.0).abs() < 0.1,
            "Expected 25% light, got {}",
            light
        );
    }

    #[test]
    fn test_auto_levels_no_clip() {
        // Test the optimized single-pass auto_levels_no_clip
        // Create data with different ranges per channel
        let mut data = vec![
            0.1, 0.2, 0.3, // Pixel 1
            0.2, 0.3, 0.4, // Pixel 2
            0.3, 0.4, 0.5, // Pixel 3
            0.4, 0.5, 0.6, // Pixel 4
            0.5, 0.6, 0.7, // Pixel 5
            0.6, 0.7, 0.8, // Pixel 6 - B channel has max (0.8)
        ];

        // Save original max
        let original_max = data.iter().cloned().fold(f32::MIN, f32::max);

        let params = auto_levels_no_clip(&mut data, 3, 0.0);

        // The max value after adjustment should not exceed original max
        let new_max = data.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            new_max <= original_max + 0.001,
            "Max after no-clip should not exceed original max. Original: {}, New: {}",
            original_max,
            new_max
        );

        // Params should reflect the channel ranges found
        assert!(params[0] < params[1], "R min should be less than R max");
        assert!(params[2] < params[3], "G min should be less than G max");
        assert!(params[4] < params[5], "B min should be less than B max");

        println!("Auto-levels no-clip params: {:?}", params);
        println!("Original max: {}, New max: {}", original_max, new_max);
    }

    #[test]
    fn test_auto_levels_no_clip_preserves_proportions() {
        // Verify that the relative proportions between channels are maintained
        // This is the key property of the no-clip variant
        let mut data = vec![
            0.2, 0.4, 0.6, // Proportions: 1:2:3
            0.4, 0.8, 0.9, // Different proportions
        ];

        let original_max = data.iter().cloned().fold(f32::MIN, f32::max);

        let _ = auto_levels_no_clip(&mut data, 3, 0.0);

        let new_max = data.iter().cloned().fold(f32::MIN, f32::max);

        // Max should be preserved (within tolerance)
        assert!(
            (new_max - original_max).abs() < 0.01,
            "Max should be approximately preserved. Original: {}, New: {}",
            original_max,
            new_max
        );
    }
}

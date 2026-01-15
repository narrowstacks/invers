//! Auto-levels functions for histogram-based automatic adjustment

use rayon::prelude::*;

use super::analysis::apply_levels_complete;
use super::LevelsParams;
use crate::auto_adjust::parallel::parallel_for_each_chunk_mut;
use crate::auto_adjust::PARALLEL_THRESHOLD;
use crate::models::AutoLevelsMode;

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

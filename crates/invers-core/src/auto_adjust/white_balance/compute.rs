//! Compute-only white balance functions for GPU optimization
//!
//! These functions compute white balance multipliers without applying them,
//! designed for use with GPU pipelines where the actual application happens
//! in shaders.

use super::{WbStats, SUBSAMPLED_GRAY_THRESHOLD, SUBSAMPLED_HIGHLIGHT_THRESHOLD};
use crate::auto_adjust::parallel::parallel_fold_reduce;

/// Compute white balance multipliers using Gray Pixel method (without applying)
///
/// This is the compute-only variant of `auto_white_balance` for use with GPU pipelines.
/// It analyzes the data and returns multipliers without modifying the input.
///
/// Designed for subsampled data (stride=8 means 1/64th of pixels), with adjusted thresholds.
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data (can be subsampled)
/// * `channels` - Number of channels (must be 3)
/// * `strength` - Adjustment strength (0.0 = no change, 1.0 = full correction)
///
/// # Returns
/// RGB multipliers [r_mult, g_mult, b_mult]
pub fn compute_wb_multipliers_gray_pixel(data: &[f32], channels: u8, strength: f32) -> [f32; 3] {
    if channels != 3 {
        panic!("compute_wb_multipliers_gray_pixel only supports 3-channel RGB images");
    }

    // Collect statistics - parallel for large images
    let stats = parallel_fold_reduce(
        data,
        3,
        WbStats::default,
        |mut stats, pixel| {
            stats.accumulate_pixel(pixel[0], pixel[1], pixel[2]);
            stats
        },
        WbStats::merge,
    );

    // Prefer gray pixels if we have enough, otherwise use highlights, then fallback to total
    // Thresholds are adjusted for subsampled data
    let (r_avg, g_avg, b_avg) = if stats.gray_count > SUBSAMPLED_GRAY_THRESHOLD {
        (
            (stats.gray_r_sum / stats.gray_count as f64) as f32,
            (stats.gray_g_sum / stats.gray_count as f64) as f32,
            (stats.gray_b_sum / stats.gray_count as f64) as f32,
        )
    } else if stats.highlight_count > SUBSAMPLED_HIGHLIGHT_THRESHOLD {
        (
            (stats.highlight_r_sum / stats.highlight_count as f64) as f32,
            (stats.highlight_g_sum / stats.highlight_count as f64) as f32,
            (stats.highlight_b_sum / stats.highlight_count as f64) as f32,
        )
    } else if stats.total_count > 0 {
        (
            (stats.total_r_sum / stats.total_count as f64) as f32,
            (stats.total_g_sum / stats.total_count as f64) as f32,
            (stats.total_b_sum / stats.total_count as f64) as f32,
        )
    } else {
        return [1.0, 1.0, 1.0];
    };

    // Calculate multipliers to make the reference neutral
    // We normalize to the green channel (common in photography)
    let r_mult = if r_avg > 0.0001 { g_avg / r_avg } else { 1.0 };
    let g_mult = 1.0; // Green is reference
    let b_mult = if b_avg > 0.0001 { g_avg / b_avg } else { 1.0 };

    // Apply strength
    let r_final = 1.0 + strength * (r_mult - 1.0);
    let g_final = 1.0 + strength * (g_mult - 1.0);
    let b_final = 1.0 + strength * (b_mult - 1.0);

    [r_final, g_final, b_final]
}

/// Compute white balance multipliers using Average method (without applying)
///
/// This is the compute-only variant of `auto_white_balance_avg` for use with GPU pipelines.
/// Implements Gray World Assumption: assumes the average of all pixels should be neutral gray.
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data (can be subsampled)
/// * `channels` - Number of channels (must be 3)
/// * `strength` - Adjustment strength (0.0 = no change, 1.0 = full correction)
///
/// # Returns
/// RGB multipliers [r_mult, g_mult, b_mult]
pub fn compute_wb_multipliers_avg(data: &[f32], channels: u8, strength: f32) -> [f32; 3] {
    if channels != 3 {
        panic!("compute_wb_multipliers_avg only supports 3-channel RGB images");
    }

    let num_pixels = data.len() / 3;
    if num_pixels == 0 {
        return [1.0, 1.0, 1.0];
    }

    // Calculate average for each channel (Gray World Assumption)
    let (r_sum, g_sum, b_sum) = parallel_fold_reduce(
        data,
        3,
        || (0.0f64, 0.0f64, 0.0f64),
        |acc, pixel| {
            (
                acc.0 + pixel[0] as f64,
                acc.1 + pixel[1] as f64,
                acc.2 + pixel[2] as f64,
            )
        },
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    );

    let r_avg = (r_sum / num_pixels as f64) as f32;
    let g_avg = (g_sum / num_pixels as f64) as f32;
    let b_avg = (b_sum / num_pixels as f64) as f32;

    // Calculate multipliers to make average neutral
    // Normalize to green channel (standard in photography)
    let r_mult = if r_avg > 0.0001 { g_avg / r_avg } else { 1.0 };
    let g_mult = 1.0; // Green is reference
    let b_mult = if b_avg > 0.0001 { g_avg / b_avg } else { 1.0 };

    // Apply strength
    let r_final = 1.0 + strength * (r_mult - 1.0);
    let g_final = 1.0 + strength * (g_mult - 1.0);
    let b_final = 1.0 + strength * (b_mult - 1.0);

    [r_final, g_final, b_final]
}

/// Compute white balance multipliers using Percentile method (without applying)
///
/// This is the compute-only variant of `auto_white_balance_percentile` for use with GPU pipelines.
/// Implements Robust White Patch: finds high percentile values and equalizes them.
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data (can be subsampled)
/// * `channels` - Number of channels (must be 3)
/// * `strength` - Adjustment strength (0.0 = no change, 1.0 = full correction)
/// * `percentile` - Percentile to use (e.g., 98.0 for 98th percentile)
///
/// # Returns
/// RGB multipliers [r_mult, g_mult, b_mult]
pub fn compute_wb_multipliers_percentile(
    data: &[f32],
    channels: u8,
    strength: f32,
    percentile: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("compute_wb_multipliers_percentile only supports 3-channel RGB images");
    }

    let num_pixels = data.len() / 3;
    if num_pixels == 0 {
        return [1.0, 1.0, 1.0];
    }

    // Collect channel values separately
    let mut r_vals: Vec<f32> = Vec::with_capacity(num_pixels);
    let mut g_vals: Vec<f32> = Vec::with_capacity(num_pixels);
    let mut b_vals: Vec<f32> = Vec::with_capacity(num_pixels);

    for pixel in data.chunks_exact(3) {
        r_vals.push(pixel[0]);
        g_vals.push(pixel[1]);
        b_vals.push(pixel[2]);
    }

    // Sort to find percentiles
    r_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    g_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    b_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Get percentile values
    let percentile_idx = ((num_pixels as f32 * percentile / 100.0) as usize).min(num_pixels - 1);
    let r_pct = r_vals[percentile_idx];
    let g_pct = g_vals[percentile_idx];
    let b_pct = b_vals[percentile_idx];

    // Calculate multipliers to make percentile values equal
    // Normalize to green channel
    let r_mult = if r_pct > 0.0001 { g_pct / r_pct } else { 1.0 };
    let g_mult = 1.0; // Green is reference
    let b_mult = if b_pct > 0.0001 { g_pct / b_pct } else { 1.0 };

    // Apply strength
    let r_final = 1.0 + strength * (r_mult - 1.0);
    let g_final = 1.0 + strength * (g_mult - 1.0);
    let b_final = 1.0 + strength * (b_mult - 1.0);

    [r_final, g_final, b_final]
}

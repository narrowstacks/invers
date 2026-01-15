//! Auto white balance algorithms
//!
//! Provides various automatic white balance correction methods including
//! gray pixel detection, gray world assumption, and percentile-based approaches.

use crate::auto_adjust::parallel::{parallel_fold_reduce, parallel_for_each_chunk_mut};

// =============================================================================
// White Balance Statistics
// =============================================================================

/// Statistics accumulated during white balance analysis
#[derive(Clone, Copy, Default)]
pub(crate) struct WbStats {
    pub highlight_r_sum: f64,
    pub highlight_g_sum: f64,
    pub highlight_b_sum: f64,
    pub highlight_count: usize,
    pub gray_r_sum: f64,
    pub gray_g_sum: f64,
    pub gray_b_sum: f64,
    pub gray_count: usize,
    pub total_r_sum: f64,
    pub total_g_sum: f64,
    pub total_b_sum: f64,
    pub total_count: usize,
}

impl WbStats {
    pub fn merge(mut self, other: Self) -> Self {
        self.highlight_r_sum += other.highlight_r_sum;
        self.highlight_g_sum += other.highlight_g_sum;
        self.highlight_b_sum += other.highlight_b_sum;
        self.highlight_count += other.highlight_count;
        self.gray_r_sum += other.gray_r_sum;
        self.gray_g_sum += other.gray_g_sum;
        self.gray_b_sum += other.gray_b_sum;
        self.gray_count += other.gray_count;
        self.total_r_sum += other.total_r_sum;
        self.total_g_sum += other.total_g_sum;
        self.total_b_sum += other.total_b_sum;
        self.total_count += other.total_count;
        self
    }

    pub fn accumulate_pixel(&mut self, r: f32, g: f32, b: f32) {
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Highlight pixels (bright areas)
        if lum > 0.6 {
            self.highlight_r_sum += r as f64;
            self.highlight_g_sum += g as f64;
            self.highlight_b_sum += b as f64;
            self.highlight_count += 1;
        }

        // Gray pixels (low saturation - channels are similar)
        let max_ch = r.max(g).max(b);
        let min_ch = r.min(g).min(b);
        if max_ch > 0.1 && (max_ch - min_ch) / max_ch < 0.15 {
            self.gray_r_sum += r as f64;
            self.gray_g_sum += g as f64;
            self.gray_b_sum += b as f64;
            self.gray_count += 1;
        }

        // Always accumulate totals for fallback
        self.total_r_sum += r as f64;
        self.total_g_sum += g as f64;
        self.total_b_sum += b as f64;
        self.total_count += 1;
    }
}

// =============================================================================
// Auto White Balance Functions
// =============================================================================

/// Auto white balance: Estimate and correct color temperature/tint
///
/// This function finds neutral/highlight areas and calculates multipliers
/// to make them neutral gray. Unlike auto-color (which uses midtone averages),
/// this looks at bright areas that should be white/neutral.
///
/// Returns [r_mult, g_mult, b_mult] - the multipliers applied
pub fn auto_white_balance(data: &mut [f32], channels: u8, strength: f32) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_white_balance only supports 3-channel RGB images");
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
    let (r_avg, g_avg, b_avg) = if stats.gray_count > 1000 {
        (
            (stats.gray_r_sum / stats.gray_count as f64) as f32,
            (stats.gray_g_sum / stats.gray_count as f64) as f32,
            (stats.gray_b_sum / stats.gray_count as f64) as f32,
        )
    } else if stats.highlight_count > 100 {
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

    // Apply with strength
    let r_final = 1.0 + strength * (r_mult - 1.0);
    let g_final = 1.0 + strength * (g_mult - 1.0);
    let b_final = 1.0 + strength * (b_mult - 1.0);

    // Apply multipliers - parallel for large images
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] *= r_final;
        pixel[1] *= g_final;
        pixel[2] *= b_final;
    });

    [r_final, g_final, b_final]
}

/// Auto white balance without clipping - same as auto_white_balance but scales
/// result to preserve original max value
pub fn auto_white_balance_no_clip(data: &mut [f32], channels: u8, strength: f32) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_white_balance_no_clip only supports 3-channel RGB images");
    }

    // Find original max
    let mut original_max = 0.0f32;
    for pixel in data.chunks_exact(3) {
        original_max = original_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }

    // Apply white balance
    let multipliers = auto_white_balance(data, channels, strength);

    // Find new max and scale back if needed
    let mut new_max = 0.0f32;
    for pixel in data.chunks_exact(3) {
        new_max = new_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }

    if new_max > original_max && new_max > 0.0001 {
        let scale = original_max / new_max;
        for value in data.iter_mut() {
            *value *= scale;
        }
        [
            multipliers[0] * scale,
            multipliers[1] * scale,
            multipliers[2] * scale,
        ]
    } else {
        multipliers
    }
}

/// Auto white balance using "Average" method (Gray World Assumption)
///
/// This implements curves-based "AUTO AVG" white balance:
/// - Assumes the average of all pixels should be neutral gray
/// - Uses ALL pixels, not just gray or highlight areas
/// - Simple but effective for most images
///
/// Returns [r_mult, g_mult, b_mult] - the multipliers applied
pub fn auto_white_balance_avg(data: &mut [f32], channels: u8, strength: f32) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_white_balance_avg only supports 3-channel RGB images");
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
    // This adjusts R and B to match G's average, removing color casts
    let r_mult = if r_avg > 0.0001 { g_avg / r_avg } else { 1.0 };
    let g_mult = 1.0; // Green is reference
    let b_mult = if b_avg > 0.0001 { g_avg / b_avg } else { 1.0 };

    // Apply with strength
    let r_final = 1.0 + strength * (r_mult - 1.0);
    let g_final = 1.0 + strength * (g_mult - 1.0);
    let b_final = 1.0 + strength * (b_mult - 1.0);

    // Apply multipliers
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] *= r_final;
        pixel[1] *= g_final;
        pixel[2] *= b_final;
    });

    [r_final, g_final, b_final]
}

/// Auto white balance using Percentile method (Robust White Patch)
///
/// This implements a noise-robust white patch algorithm:
/// - Find the value at a high percentile (default 98th) for each channel
/// - Scale channels so these percentile values are equal
/// - More robust than max RGB, less aggressive than gray world
///
/// This is likely closest to what commercial plugins' "AUTO AVG" actually does.
///
/// Returns [r_mult, g_mult, b_mult] - the multipliers applied
pub fn auto_white_balance_percentile(
    data: &mut [f32],
    channels: u8,
    strength: f32,
    percentile: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_white_balance_percentile only supports 3-channel RGB images");
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

    // Apply with strength
    let r_final = 1.0 + strength * (r_mult - 1.0);
    let g_final = 1.0 + strength * (g_mult - 1.0);
    let b_final = 1.0 + strength * (b_mult - 1.0);

    // Apply multipliers
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] *= r_final;
        pixel[1] *= g_final;
        pixel[2] *= b_final;
    });

    [r_final, g_final, b_final]
}

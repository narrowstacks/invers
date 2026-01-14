//! White balance adjustment functions
//!
//! Provides automatic and manual white balance correction including
//! temperature-based adjustment and various auto white balance algorithms.

use super::parallel::{parallel_fold_reduce, parallel_for_each_chunk_mut};

// =============================================================================
// Color Temperature to RGB Conversion
// =============================================================================

/// Convert color temperature in Kelvin to RGB multipliers
///
/// Based on Tanner Helland's algorithm which approximates the Planckian locus.
/// Reference: https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
///
/// # Arguments
/// * `kelvin` - Color temperature in Kelvin (1000-40000)
///
/// # Returns
/// RGB multipliers normalized to green channel = 1.0
#[allow(clippy::excessive_precision)] // Published constants from Tanner Helland algorithm
pub fn kelvin_to_rgb_multipliers(kelvin: f32) -> [f32; 3] {
    // Clamp temperature to valid range
    let temp = (kelvin / 100.0).clamp(10.0, 400.0);

    // Calculate RGB values using polynomial approximation
    let (r, g, b) = if temp <= 66.0 {
        // For temperatures <= 6600K
        let r = 255.0;
        let g = 99.4708025861 * temp.ln() - 161.1195681661;
        let b = if temp <= 19.0 {
            0.0
        } else {
            138.5177312231 * (temp - 10.0).ln() - 305.0447927307
        };
        (r, g.clamp(0.0, 255.0), b.clamp(0.0, 255.0))
    } else {
        // For temperatures > 6600K
        let r = 329.698727446 * (temp - 60.0).powf(-0.1332047592);
        let g = 288.1221695283 * (temp - 60.0).powf(-0.0755148492);
        let b = 255.0;
        (r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), b)
    };

    // Normalize to 0-1 range
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;

    // Convert to multipliers (normalize to green)
    // To correct from temperature T to neutral (D65 ~6500K), we need
    // to apply the inverse of what that temperature produces
    let g_ref = g.max(0.001);
    [g_ref / r.max(0.001), 1.0, g_ref / b.max(0.001)]
}

/// Apply white balance from temperature and tint
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data
/// * `channels` - Number of channels (must be 3)
/// * `temperature` - Color temperature in Kelvin (e.g., 5500 for daylight)
/// * `tint` - Green-magenta tint adjustment (-100 to +100, 0 = neutral)
///
/// # Returns
/// The RGB multipliers that were applied
pub fn apply_white_balance_from_temperature(
    data: &mut [f32],
    channels: u8,
    temperature: f32,
    tint: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("apply_white_balance_from_temperature only supports 3-channel RGB images");
    }

    // Get base multipliers from temperature
    let mut multipliers = kelvin_to_rgb_multipliers(temperature);

    // Apply tint adjustment (affects green-magenta axis)
    // Positive tint = more green, negative = more magenta
    let tint_factor = 1.0 + tint / 200.0; // ±0.5 adjustment range
    multipliers[1] *= tint_factor;

    // Apply multipliers to the image
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] *= multipliers[0];
        pixel[1] *= multipliers[1];
        pixel[2] *= multipliers[2];
    });

    multipliers
}

// =============================================================================
// White Balance Statistics
// =============================================================================

/// Statistics accumulated during white balance analysis
#[derive(Clone, Copy, Default)]
struct WbStats {
    highlight_r_sum: f64,
    highlight_g_sum: f64,
    highlight_b_sum: f64,
    highlight_count: usize,
    gray_r_sum: f64,
    gray_g_sum: f64,
    gray_b_sum: f64,
    gray_count: usize,
    total_r_sum: f64,
    total_g_sum: f64,
    total_b_sum: f64,
    total_count: usize,
}

impl WbStats {
    fn merge(mut self, other: Self) -> Self {
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

    fn accumulate_pixel(&mut self, r: f32, g: f32, b: f32) {
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

// =============================================================================
// Compute-Only White Balance Functions (for GPU optimization)
// =============================================================================

/// Minimum pixel counts for subsampled data (adjusted for stride=8 subsampling)
/// Original thresholds are divided by 64 (8x8 subsampling factor)
const SUBSAMPLED_GRAY_THRESHOLD: usize = 16; // was 1000
const SUBSAMPLED_HIGHLIGHT_THRESHOLD: usize = 2; // was 100

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

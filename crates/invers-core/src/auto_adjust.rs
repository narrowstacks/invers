//! Automatic adjustment functions for image processing
//!
//! Provides auto-levels, auto-color, auto white balance, and other automatic
//! corrections similar to Photoshop's automatic adjustment tools.

use std::cmp::Ordering;

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

    // Find pixels in the highlight region (top 10% brightness)
    // These are most likely to be neutral (sky, clouds, white objects)
    let mut highlight_r_sum = 0.0f64;
    let mut highlight_g_sum = 0.0f64;
    let mut highlight_b_sum = 0.0f64;
    let mut highlight_count = 0usize;

    // Also collect "gray" pixels where R≈G≈B (likely neutral)
    let mut gray_r_sum = 0.0f64;
    let mut gray_g_sum = 0.0f64;
    let mut gray_b_sum = 0.0f64;
    let mut gray_count = 0usize;

    for pixel in data.chunks_exact(3) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Highlight pixels (bright areas)
        if lum > 0.6 {
            highlight_r_sum += r as f64;
            highlight_g_sum += g as f64;
            highlight_b_sum += b as f64;
            highlight_count += 1;
        }

        // Gray pixels (low saturation - channels are similar)
        let max_ch = r.max(g).max(b);
        let min_ch = r.min(g).min(b);
        if max_ch > 0.1 && (max_ch - min_ch) / max_ch < 0.15 {
            gray_r_sum += r as f64;
            gray_g_sum += g as f64;
            gray_b_sum += b as f64;
            gray_count += 1;
        }
    }

    // Prefer gray pixels if we have enough, otherwise use highlights
    let (r_avg, g_avg, b_avg) = if gray_count > 1000 {
        (
            (gray_r_sum / gray_count as f64) as f32,
            (gray_g_sum / gray_count as f64) as f32,
            (gray_b_sum / gray_count as f64) as f32,
        )
    } else if highlight_count > 100 {
        (
            (highlight_r_sum / highlight_count as f64) as f32,
            (highlight_g_sum / highlight_count as f64) as f32,
            (highlight_b_sum / highlight_count as f64) as f32,
        )
    } else {
        // Fallback: use overall average
        let mut total_r = 0.0f64;
        let mut total_g = 0.0f64;
        let mut total_b = 0.0f64;
        let count = data.len() / 3;
        for pixel in data.chunks_exact(3) {
            total_r += pixel[0] as f64;
            total_g += pixel[1] as f64;
            total_b += pixel[2] as f64;
        }
        (
            (total_r / count as f64) as f32,
            (total_g / count as f64) as f32,
            (total_b / count as f64) as f32,
        )
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

    // Apply multipliers
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] *= r_final;
        pixel[1] *= g_final;
        pixel[2] *= b_final;
    }

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
        [multipliers[0] * scale, multipliers[1] * scale, multipliers[2] * scale]
    } else {
        multipliers
    }
}

/// Auto-levels mode for controlling color preservation
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AutoLevelsMode {
    /// Independent per-channel stretching (default, may shift colors)
    #[default]
    PerChannel,
    /// Saturation-aware: reduces stretch for channels that would clip heavily
    SaturationAware,
    /// Preserve saturation: use minimum stretch across all channels
    PreserveSaturation,
}

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
/// Returns [r_min, r_max, g_min, g_max, b_min, b_max] - the percentile ranges found
pub fn auto_levels_no_clip(data: &mut [f32], channels: u8, clip_percent: f32) -> [f32; 6] {
    if channels != 3 {
        panic!("auto_levels_no_clip only supports 3-channel RGB images");
    }

    // Build per-channel histograms in a single pass
    const NUM_BUCKETS: usize = 65536;
    let mut r_hist = vec![0u32; NUM_BUCKETS];
    let mut g_hist = vec![0u32; NUM_BUCKETS];
    let mut b_hist = vec![0u32; NUM_BUCKETS];

    // Track actual max values
    let mut overall_actual_max = 0.0f32;

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

        // Track overall max
        overall_actual_max = overall_actual_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }

    let num_pixels = data.len() / 3;

    // Compute clipped percentile ranges (same as regular auto-levels)
    let (r_min, r_clip_max) = compute_clipped_range_from_histogram(&r_hist, num_pixels, clip_percent);
    let (g_min, g_clip_max) = compute_clipped_range_from_histogram(&g_hist, num_pixels, clip_percent);
    let (b_min, b_clip_max) = compute_clipped_range_from_histogram(&b_hist, num_pixels, clip_percent);

    // First pass: apply normalization and find the new max
    let mut new_max = 0.0f32;
    for pixel in data.chunks_exact_mut(3) {
        // Normalize using clipped ranges (like regular auto-levels, but to 0-1)
        pixel[0] = if (r_clip_max - r_min).abs() > 0.0001 {
            (pixel[0] - r_min) / (r_clip_max - r_min)
        } else {
            pixel[0]
        };
        pixel[1] = if (g_clip_max - g_min).abs() > 0.0001 {
            (pixel[1] - g_min) / (g_clip_max - g_min)
        } else {
            pixel[1]
        };
        pixel[2] = if (b_clip_max - b_min).abs() > 0.0001 {
            (pixel[2] - b_min) / (b_clip_max - b_min)
        } else {
            pixel[2]
        };

        new_max = new_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }

    // Second pass: scale everything so the max equals overall_actual_max
    // This preserves all data while giving proper color balance
    if new_max > 0.0001 {
        let scale = overall_actual_max / new_max;
        for value in data.iter_mut() {
            *value *= scale;
        }
    }

    [r_min, r_clip_max, g_min, g_clip_max, b_min, b_clip_max]
}

/// Auto-levels with configurable mode for color preservation
///
/// Modes:
/// - `PerChannel`: Independent stretching per channel (may shift colors)
/// - `SaturationAware`: Reduces stretch for channels that would clip heavily
/// - `PreserveSaturation`: Uses minimum stretch across all channels
pub fn auto_levels_with_mode(
    data: &mut [f32],
    channels: u8,
    clip_percent: f32,
    mode: AutoLevelsMode,
) -> [f32; 6] {
    if channels != 3 {
        panic!("auto_levels only supports 3-channel RGB images");
    }

    // Build per-channel histograms in a single pass through the data
    const NUM_BUCKETS: usize = 65536;
    let mut r_hist = vec![0u32; NUM_BUCKETS];
    let mut g_hist = vec![0u32; NUM_BUCKETS];
    let mut b_hist = vec![0u32; NUM_BUCKETS];

    // Single pass to build all three histograms
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
            // No adjustment needed
        }
        AutoLevelsMode::PreserveSaturation => {
            // Use the minimum stretch (maximum range) across all channels
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

    // Apply stretch to each channel in-place
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] = stretch_value(pixel[0], r_min, r_max);
        pixel[1] = stretch_value(pixel[1], g_min, g_max);
        pixel[2] = stretch_value(pixel[2], b_min, b_max);
    }

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
pub fn apply_levels_complete(data: &mut [f32], r: &LevelsParams, g: &LevelsParams, b: &LevelsParams) {
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
fn compute_clipped_range_from_histogram(
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

const AUTO_COLOR_INITIAL_LOW: f32 = 0.35;
const AUTO_COLOR_INITIAL_HIGH: f32 = 0.65;
const AUTO_COLOR_EXPANSION_STEP: f32 = 0.10;
const AUTO_COLOR_MAX_EXPANSIONS: usize = 4;
const AUTO_COLOR_MIN_SAMPLES: usize = 512;

/// Auto-color without clipping: Neutralize color casts while preserving all data
///
/// This version:
/// 1. Calculates and applies the ideal color correction gains (same as regular auto-color)
/// 2. Then scales the entire result so the max doesn't exceed the original max
/// 3. This gives proper color balance while preserving all highlight detail
pub fn auto_color_no_clip(
    data: &mut [f32],
    channels: u8,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_color_no_clip only supports 3-channel RGB images");
    }

    // First pass: collect stats and find current max
    let mut low = AUTO_COLOR_INITIAL_LOW;
    let mut high = AUTO_COLOR_INITIAL_HIGH;
    let mut stats = collect_channel_stats(data, low, high);
    let mut expansions = 0;

    while stats.count < AUTO_COLOR_MIN_SAMPLES
        && expansions < AUTO_COLOR_MAX_EXPANSIONS
        && (low > 0.0 || high < 1.0)
    {
        low = (low - AUTO_COLOR_EXPANSION_STEP).max(0.0);
        high = (high + AUTO_COLOR_EXPANSION_STEP).min(1.0);
        expansions += 1;
        stats = collect_channel_stats(data, low, high);
    }

    if stats.count == 0 {
        stats = collect_channel_stats(data, 0.0, 1.0);
    }

    if stats.count == 0 {
        return [1.0, 1.0, 1.0];
    }

    // Find current overall max
    let mut overall_max = 0.0f32;
    for pixel in data.chunks_exact(3) {
        overall_max = overall_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }

    let sample_count = stats.count as f32;
    let r_avg = stats.r_sum / sample_count;
    let g_avg = stats.g_sum / sample_count;
    let b_avg = stats.b_sum / sample_count;

    let target_gray = (r_avg + g_avg + b_avg) / 3.0;

    // Calculate ideal adjustment factors (same as regular auto-color)
    let clamp_gain = |value: f32| value.clamp(min_gain, max_gain);

    let r_ideal = if r_avg > 0.0001 {
        clamp_gain(target_gray / r_avg)
    } else {
        1.0
    };
    let g_ideal = if g_avg > 0.0001 {
        clamp_gain(target_gray / g_avg)
    } else {
        1.0
    };
    let b_ideal = if b_avg > 0.0001 {
        clamp_gain(target_gray / b_avg)
    } else {
        1.0
    };

    // Calculate effective gains with strength applied
    let r_gain = 1.0 - strength + strength * r_ideal;
    let g_gain = 1.0 - strength + strength * g_ideal;
    let b_gain = 1.0 - strength + strength * b_ideal;

    // Apply color correction gains and find new max
    let mut new_max = 0.0f32;
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] *= r_gain;
        pixel[1] *= g_gain;
        pixel[2] *= b_gain;
        new_max = new_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
    }

    // Scale everything so the max equals the original overall_max
    // This preserves all data while giving proper color balance
    if new_max > overall_max && new_max > 0.0001 {
        let scale = overall_max / new_max;
        for value in data.iter_mut() {
            *value *= scale;
        }
        // Return the effective gains (original gain × scale)
        [r_gain * scale, g_gain * scale, b_gain * scale]
    } else {
        [r_gain, g_gain, b_gain]
    }
}

#[derive(Default, Clone, Copy)]
struct ChannelStats {
    count: usize,
    r_sum: f32,
    g_sum: f32,
    b_sum: f32,
}

impl ChannelStats {
    fn add(&mut self, r: f32, g: f32, b: f32) {
        self.count += 1;
        self.r_sum += r;
        self.g_sum += g;
        self.b_sum += b;
    }
}

/// Auto-color: Neutralize color casts by adjusting midtones
/// Similar to Photoshop's Auto Color command
pub fn auto_color(
    data: &mut [f32],
    channels: u8,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_color only supports 3-channel RGB images");
    }

    let mut low = AUTO_COLOR_INITIAL_LOW;
    let mut high = AUTO_COLOR_INITIAL_HIGH;
    let mut stats = collect_channel_stats(data, low, high);
    let mut expansions = 0;

    while stats.count < AUTO_COLOR_MIN_SAMPLES
        && expansions < AUTO_COLOR_MAX_EXPANSIONS
        && (low > 0.0 || high < 1.0)
    {
        low = (low - AUTO_COLOR_EXPANSION_STEP).max(0.0);
        high = (high + AUTO_COLOR_EXPANSION_STEP).min(1.0);
        expansions += 1;
        stats = collect_channel_stats(data, low, high);
    }

    if stats.count == 0 {
        // Fallback: use entire histogram
        stats = collect_channel_stats(data, 0.0, 1.0);
    }

    if stats.count == 0 {
        return [1.0, 1.0, 1.0];
    }

    let sample_count = stats.count as f32;
    let r_avg = stats.r_sum / sample_count;
    let g_avg = stats.g_sum / sample_count;
    let b_avg = stats.b_sum / sample_count;

    // Calculate the target neutral gray value (average of all channels)
    let target_gray = (r_avg + g_avg + b_avg) / 3.0;

    // Calculate adjustment factors to bring each channel to neutral
    let clamp = |value: f32| value.clamp(min_gain, max_gain);

    let r_adjustment = if r_avg > 0.0001 {
        clamp(target_gray / r_avg)
    } else {
        1.0
    };
    let g_adjustment = if g_avg > 0.0001 {
        clamp(target_gray / g_avg)
    } else {
        1.0
    };
    let b_adjustment = if b_avg > 0.0001 {
        clamp(target_gray / b_avg)
    } else {
        1.0
    };

    // Apply adjustments with configurable strength
    for pixel in data.chunks_exact_mut(3) {
        // Blend between original and adjusted based on strength
        pixel[0] = (pixel[0] * (1.0 - strength + strength * r_adjustment)).clamp(0.0, 1.0);
        pixel[1] = (pixel[1] * (1.0 - strength + strength * g_adjustment)).clamp(0.0, 1.0);
        pixel[2] = (pixel[2] * (1.0 - strength + strength * b_adjustment)).clamp(0.0, 1.0);
    }

    // Return adjustments for debugging
    [r_adjustment, g_adjustment, b_adjustment]
}

fn collect_channel_stats(data: &[f32], low: f32, high: f32) -> ChannelStats {
    if low <= 0.0 && high >= 1.0 {
        let mut stats = ChannelStats::default();
        for pixel in data.chunks_exact(3) {
            stats.add(pixel[0], pixel[1], pixel[2]);
        }
        return stats;
    }

    let mut stats = ChannelStats::default();
    for pixel in data.chunks_exact(3) {
        let brightness = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
        if brightness >= low && brightness <= high {
            stats.add(pixel[0], pixel[1], pixel[2]);
        }
    }
    stats
}

/// Adaptive shadow lift based on percentile using histogram-based approach
///
/// This implementation uses a histogram to find the percentile value,
/// avoiding a full copy of the data array. Memory usage is O(buckets)
/// instead of O(n), which is significant for large images.
pub fn adaptive_shadow_lift(data: &mut [f32], target_black: f32, percentile: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // Use histogram-based percentile finding - O(n) time, O(buckets) space
    // This avoids copying the entire data array
    let current_black = find_percentile_via_histogram(data, percentile);

    // Calculate lift amount to bring current_black to target_black
    let lift = target_black - current_black;

    if lift > 0.0 {
        // Apply uniform lift to all values in-place
        for value in data.iter_mut() {
            *value = (*value + lift).clamp(0.0, 1.0);
        }
    }

    lift
}

/// Find percentile value using a histogram-based approach
///
/// Uses 65536 buckets (16-bit precision) to approximate the percentile
/// without needing to copy or sort the entire dataset.
///
/// This is O(n) time to build histogram + O(buckets) to find percentile,
/// and O(buckets) space, which is much better than O(n) space for full copy.
fn find_percentile_via_histogram(data: &[f32], percentile: f32) -> f32 {
    const NUM_BUCKETS: usize = 65536;

    // Build histogram
    let mut histogram = vec![0u32; NUM_BUCKETS];
    for &value in data {
        let bucket =
            ((value.clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        histogram[bucket] += 1;
    }

    // Find the bucket containing the percentile
    let target_count = ((data.len() as f32 * percentile / 100.0) as usize).max(1);
    let mut cumulative = 0usize;

    for (bucket, &count) in histogram.iter().enumerate() {
        cumulative += count as usize;
        if cumulative >= target_count {
            // Return the value corresponding to this bucket
            return bucket as f32 / (NUM_BUCKETS - 1) as f32;
        }
    }

    // Fallback (should not reach here)
    0.0
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

/// Highlight compression: Compress bright highlights to prevent clipping
#[inline]
pub fn compress_highlights(data: &mut [f32], threshold: f32, compression: f32) {
    for value in data.iter_mut() {
        if *value > threshold {
            // Apply soft compression above threshold
            let excess = *value - threshold;
            let compressed = excess * compression;
            *value = (threshold + compressed).clamp(0.0, 1.0);
        }
    }
}

/// Automatic exposure normalization. Scales image so that the median luminance
/// approaches `target_median`. Returns the gain that was applied.
pub fn auto_exposure(
    data: &mut [f32],
    target_median: f32,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
) -> f32 {
    auto_exposure_impl(data, target_median, strength, min_gain, max_gain, true)
}

/// Auto exposure without clipping - limits gain to prevent exceeding current max
pub fn auto_exposure_no_clip(
    data: &mut [f32],
    target_median: f32,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
) -> f32 {
    auto_exposure_impl(data, target_median, strength, min_gain, max_gain, false)
}

fn auto_exposure_impl(
    data: &mut [f32],
    target_median: f32,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
    clip: bool,
) -> f32 {
    if data.is_empty() {
        return 1.0;
    }

    // Pre-allocate luminance buffer with exact capacity
    let num_pixels = data.len() / 3;
    let mut luminances = Vec::with_capacity(num_pixels);

    // Also find max value if not clipping
    let mut current_max = 0.0f32;

    // Collect luminance samples (Rec.709 weights)
    for pixel in data.chunks_exact(3) {
        let lum = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        luminances.push(lum);
        if !clip {
            current_max = current_max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
        }
    }

    if luminances.is_empty() {
        return 1.0;
    }

    // Use select_nth_unstable for efficient median finding
    let mid = luminances.len() / 2;
    luminances.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = luminances[mid];

    if !median.is_finite() || median <= 1e-6 {
        return 1.0;
    }

    let desired_gain = (target_median / median).clamp(min_gain, max_gain);
    let mut gain = 1.0 + strength * (desired_gain - 1.0);
    gain = gain.clamp(min_gain, max_gain);

    // In no-clip mode, limit gain so max * gain doesn't exceed current max
    if !clip && current_max > 0.0 && gain > 1.0 {
        // Don't allow gain to push any values higher
        gain = 1.0;
    }

    if !gain.is_finite() || (gain - 1.0).abs() < 1e-6 {
        return 1.0;
    }

    // Apply gain in-place
    if clip {
        for value in data.iter_mut() {
            *value = (*value * gain).clamp(0.0, 1.0);
        }
    } else {
        for value in data.iter_mut() {
            *value = *value * gain;
        }
    }

    gain
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
    fn test_auto_color() {
        // Test with reddish cast
        let mut data = vec![
            0.6, 0.4, 0.4, // Reddish
            0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, 0.4,
        ];

        let adjustments = auto_color(&mut data, 3, 1.0, 0.7, 1.3);

        println!("Color adjustments: {:?}", adjustments);
        println!("Corrected data: {:?}", data);

        // Red adjustment should be < 1.0 (reduce red)
        assert!(adjustments[0] < 1.0);
    }

    #[test]
    fn test_adaptive_shadow_lift() {
        let mut data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let lift = adaptive_shadow_lift(&mut data, 0.15, 1.0);

        println!("Shadow lift: {}", lift);
        println!("Lifted data: {:?}", data);

        // First value should now be ~0.15
        assert!((data[0] - 0.15).abs() < 0.01);
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
        let mut data = Vec::new();
        // Low cluster: 500 samples at 0.2
        for _ in 0..500 {
            data.push(0.2);
        }
        // High cluster: 500 samples at 0.8
        for _ in 0..500 {
            data.push(0.8);
        }

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
        let mut data = Vec::new();
        // 25% dark (0.0-0.25)
        for _ in 0..25 {
            data.push(0.1);
        }
        // 50% mid (0.25-0.75)
        for _ in 0..50 {
            data.push(0.5);
        }
        // 25% light (0.75-1.0)
        for _ in 0..25 {
            data.push(0.9);
        }

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
}

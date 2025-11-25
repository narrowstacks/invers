//! Automatic adjustment functions for image processing
//!
//! Provides auto-levels, auto-color, and other automatic corrections
//! similar to Photoshop's automatic adjustment tools.

use std::cmp::Ordering;

/// Auto-levels mode for controlling color preservation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoLevelsMode {
    /// Independent per-channel stretching (default, may shift colors)
    PerChannel,
    /// Saturation-aware: reduces stretch for channels that would clip heavily
    SaturationAware,
    /// Preserve saturation: use minimum stretch across all channels
    PreserveSaturation,
}

impl Default for AutoLevelsMode {
    fn default() -> Self {
        Self::PerChannel
    }
}

/// Auto-levels: Stretch histogram to full 0.0-1.0 range per channel
/// This is the key missing step that Photoshop/Grain2Pixel applies
///
/// Uses histogram-based percentile finding to avoid copying all channel data.
/// Memory usage: O(buckets * 3) instead of O(n * 3)
pub fn auto_levels(data: &mut [f32], channels: u8, clip_percent: f32) -> [f32; 6] {
    auto_levels_with_mode(data, channels, clip_percent, AutoLevelsMode::PerChannel)
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
        let r_bucket = ((pixel[0].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize)
            .min(NUM_BUCKETS - 1);
        let g_bucket = ((pixel[1].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize)
            .min(NUM_BUCKETS - 1);
        let b_bucket = ((pixel[2].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize)
            .min(NUM_BUCKETS - 1);
        r_hist[r_bucket] += 1;
        g_hist[g_bucket] += 1;
        b_hist[b_bucket] += 1;
    }

    let num_pixels = data.len() / 3;

    // Compute initial min/max for each channel with clipping using histograms
    let (mut r_min, mut r_max) = compute_clipped_range_from_histogram(&r_hist, num_pixels, clip_percent);
    let (mut g_min, mut g_max) = compute_clipped_range_from_histogram(&g_hist, num_pixels, clip_percent);
    let (mut b_min, mut b_max) = compute_clipped_range_from_histogram(&b_hist, num_pixels, clip_percent);

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
        let bucket = ((value.clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize)
            .min(NUM_BUCKETS - 1);
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
    if data.is_empty() {
        return 1.0;
    }

    // Pre-allocate luminance buffer with exact capacity
    let num_pixels = data.len() / 3;
    let mut luminances = Vec::with_capacity(num_pixels);
    
    // Collect luminance samples (Rec.709 weights)
    for pixel in data.chunks_exact(3) {
        let lum = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        luminances.push(lum);
    }

    if luminances.is_empty() {
        return 1.0;
    }

    // Use select_nth_unstable for efficient median finding
    let mid = luminances.len() / 2;
    luminances.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    });
    let median = luminances[mid];

    if !median.is_finite() || median <= 1e-6 {
        return 1.0;
    }

    let desired_gain = (target_median / median).clamp(min_gain, max_gain);
    let gain = 1.0 + strength * (desired_gain - 1.0);
    let gain = gain.clamp(min_gain, max_gain);

    if !gain.is_finite() || (gain - 1.0).abs() < 1e-6 {
        return 1.0;
    }

    // Apply gain in-place
    for value in data.iter_mut() {
        *value = (*value * gain).clamp(0.0, 1.0);
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
}

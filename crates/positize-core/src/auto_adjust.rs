//! Automatic adjustment functions for image processing
//!
//! Provides auto-levels, auto-color, and other automatic corrections
//! similar to Photoshop's automatic adjustment tools.

use std::cmp::Ordering;

/// Auto-levels: Stretch histogram to full 0.0-1.0 range per channel
/// This is the key missing step that Photoshop/Grain2Pixel applies
pub fn auto_levels(data: &mut [f32], channels: u8, clip_percent: f32) -> [f32; 6] {
    if channels != 3 {
        panic!("auto_levels only supports 3-channel RGB images");
    }

    // Separate into channels
    let mut r_values = Vec::new();
    let mut g_values = Vec::new();
    let mut b_values = Vec::new();

    for pixel in data.chunks_exact(3) {
        r_values.push(pixel[0]);
        g_values.push(pixel[1]);
        b_values.push(pixel[2]);
    }

    // Compute min/max for each channel with clipping
    let (r_min, r_max) = compute_clipped_range(&mut r_values, clip_percent);
    let (g_min, g_max) = compute_clipped_range(&mut g_values, clip_percent);
    let (b_min, b_max) = compute_clipped_range(&mut b_values, clip_percent);

    // Apply stretch to each channel
    for pixel in data.chunks_exact_mut(3) {
        pixel[0] = stretch_value(pixel[0], r_min, r_max);
        pixel[1] = stretch_value(pixel[1], g_min, g_max);
        pixel[2] = stretch_value(pixel[2], b_min, b_max);
    }

    // Return the adjustment parameters for debugging
    [r_min, r_max, g_min, g_max, b_min, b_max]
}

/// Compute min/max with percentile clipping
fn compute_clipped_range(values: &mut [f32], clip_percent: f32) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 1.0);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let clip_fraction = clip_percent / 100.0;
    let low_idx = ((values.len() as f32 * clip_fraction) as usize).min(values.len() - 1);
    let high_idx = ((values.len() as f32 * (1.0 - clip_fraction)) as usize).min(values.len() - 1);

    let min = values[low_idx];
    let max = values[high_idx];

    // Ensure we have a valid range
    if (max - min).abs() < 0.0001 {
        (min, min + 0.1)
    } else {
        (min, max)
    }
}

/// Stretch a value from [old_min, old_max] to [0.0, 1.0]
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

/// Adaptive shadow lift based on percentile
pub fn adaptive_shadow_lift(data: &mut [f32], target_black: f32, percentile: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    // Find the specified percentile value (e.g., 1st percentile)
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((sorted.len() as f32 * percentile / 100.0) as usize).min(sorted.len() - 1);
    let current_black = sorted[idx];

    // Calculate lift amount to bring current_black to target_black
    let lift = target_black - current_black;

    if lift > 0.0 {
        // Apply uniform lift to all values
        for value in data.iter_mut() {
            *value = (*value + lift).clamp(0.0, 1.0);
        }
    }

    lift
}

/// Highlight compression: Compress bright highlights to prevent clipping
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

    // Collect luminance samples (Rec.709 weights)
    let mut luminances = Vec::with_capacity(data.len() / 3);
    for pixel in data.chunks_exact(3) {
        let lum = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        luminances.push(lum);
    }

    if luminances.is_empty() {
        return 1.0;
    }

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

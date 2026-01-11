//! Automatic exposure adjustment functions
//!
//! Provides automatic exposure normalization, shadow lifting, and highlight compression.

use std::cmp::Ordering;

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
            *value *= gain;
        }
    }

    gain
}

#[cfg(test)]
mod tests {
    use super::*;

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

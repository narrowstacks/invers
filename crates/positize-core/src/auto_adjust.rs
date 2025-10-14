//! Automatic adjustment functions for image processing
//!
//! Provides auto-levels, auto-color, and other automatic corrections
//! similar to Photoshop's automatic adjustment tools.

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

/// Auto-color: Neutralize color casts by adjusting midtones
/// Similar to Photoshop's Auto Color command
pub fn auto_color(data: &mut [f32], channels: u8, strength: f32) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_color only supports 3-channel RGB images");
    }

    // Find average of midtone pixels (40-60% brightness range)
    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;
    let mut count = 0;

    for pixel in data.chunks_exact(3) {
        let brightness = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
        if brightness >= 0.4 && brightness <= 0.6 {
            r_sum += pixel[0];
            g_sum += pixel[1];
            b_sum += pixel[2];
            count += 1;
        }
    }

    if count == 0 {
        return [0.0, 0.0, 0.0]; // No adjustment
    }

    let r_avg = r_sum / count as f32;
    let g_avg = g_sum / count as f32;
    let b_avg = b_sum / count as f32;

    // Calculate the target neutral gray value (average of all channels)
    let target_gray = (r_avg + g_avg + b_avg) / 3.0;

    // Calculate adjustment factors to bring each channel to neutral
    let r_adjustment = if r_avg > 0.0001 {
        target_gray / r_avg
    } else {
        1.0
    };
    let g_adjustment = if g_avg > 0.0001 {
        target_gray / g_avg
    } else {
        1.0
    };
    let b_adjustment = if b_avg > 0.0001 {
        target_gray / b_avg
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
        assert!(data[1] > 0.3); // First G value might be higher or lower

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

        let adjustments = auto_color(&mut data, 3, 1.0);

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

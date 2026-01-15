//! Automatic color correction functions
//!
//! Provides scene-adaptive color correction that preserves scene character
//! while removing color casts.

use rayon::prelude::*;

use super::parallel::{parallel_fold_reduce, parallel_for_each_chunk_mut};
use super::PARALLEL_THRESHOLD;

/// Auto-color without clipping: Scene-adaptive color correction preserving all data
///
/// This version:
/// 1. Uses scene-adaptive targeting (preserves scene warmth/coolness)
/// 2. Limits channel divergence to prevent aggressive neutralization
/// 3. Scales the result so max doesn't exceed original max (preserves highlight detail)
pub fn auto_color_no_clip(
    data: &mut [f32],
    channels: u8,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
    max_divergence: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_color_no_clip only supports 3-channel RGB images");
    }

    let num_pixels = data.len() / 3;

    if num_pixels == 0 {
        return [1.0, 1.0, 1.0];
    }

    // First pass: compute full-image channel means and find current max
    // Using full-image means ensures consistent color correction across all luminance ranges
    let (r_sum, g_sum, b_sum, overall_max) = if num_pixels >= PARALLEL_THRESHOLD {
        data.par_chunks_exact(3)
            .fold(
                || (0.0f64, 0.0f64, 0.0f64, 0.0f32),
                |acc, pixel| {
                    let pmax = pixel[0].max(pixel[1]).max(pixel[2]);
                    (
                        acc.0 + pixel[0] as f64,
                        acc.1 + pixel[1] as f64,
                        acc.2 + pixel[2] as f64,
                        acc.3.max(pmax),
                    )
                },
            )
            .reduce(
                || (0.0f64, 0.0f64, 0.0f64, 0.0f32),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3.max(b.3)),
            )
    } else {
        let mut r = 0.0f64;
        let mut g = 0.0f64;
        let mut b = 0.0f64;
        let mut max = 0.0f32;
        for pixel in data.chunks_exact(3) {
            r += pixel[0] as f64;
            g += pixel[1] as f64;
            b += pixel[2] as f64;
            max = max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
        }
        (r, g, b, max)
    };

    let count = num_pixels as f64;
    let r_avg = (r_sum / count) as f32;
    let g_avg = (g_sum / count) as f32;
    let b_avg = (b_sum / count) as f32;

    // Scene-adaptive targeting: blend between scene-preserving and neutral
    let avg_luminance = (r_avg + g_avg + b_avg) / 3.0;

    // Target values that respect scene color temperature
    let target_r = r_avg + strength * (avg_luminance - r_avg);
    let target_g = g_avg + strength * (avg_luminance - g_avg);
    let target_b = b_avg + strength * (avg_luminance - b_avg);

    // Calculate adjustment factors with per-channel gain limits
    let clamp_gain = |value: f32| value.clamp(min_gain, max_gain);

    let r_ideal = if r_avg > 0.0001 {
        clamp_gain(target_r / r_avg)
    } else {
        1.0
    };
    let g_ideal = if g_avg > 0.0001 {
        clamp_gain(target_g / g_avg)
    } else {
        1.0
    };
    let b_ideal = if b_avg > 0.0001 {
        clamp_gain(target_b / b_avg)
    } else {
        1.0
    };

    // Limit divergence to preserve scene character
    let [r_gain, g_gain, b_gain] =
        limit_channel_divergence([r_ideal, g_ideal, b_ideal], max_divergence);

    // Apply color correction gains and find new max - parallel for large images
    let new_max = if num_pixels >= PARALLEL_THRESHOLD {
        use std::sync::atomic::{AtomicU32, Ordering};
        let atomic_max = AtomicU32::new(0);

        data.par_chunks_exact_mut(3).for_each(|pixel| {
            pixel[0] *= r_gain;
            pixel[1] *= g_gain;
            pixel[2] *= b_gain;
            let pixel_max = pixel[0].max(pixel[1]).max(pixel[2]);
            // Atomic max update using compare-exchange loop
            let mut current = atomic_max.load(Ordering::Relaxed);
            loop {
                let current_f32 = f32::from_bits(current);
                if pixel_max <= current_f32 {
                    break;
                }
                match atomic_max.compare_exchange_weak(
                    current,
                    pixel_max.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => current = x,
                }
            }
        });
        f32::from_bits(atomic_max.load(Ordering::Relaxed))
    } else {
        let mut max = 0.0f32;
        for pixel in data.chunks_exact_mut(3) {
            pixel[0] *= r_gain;
            pixel[1] *= g_gain;
            pixel[2] *= b_gain;
            max = max.max(pixel[0]).max(pixel[1]).max(pixel[2]);
        }
        max
    };

    // Scale everything so the max equals the original overall_max
    // This preserves all data while giving proper color balance
    if new_max > overall_max && new_max > 0.0001 {
        let scale = overall_max / new_max;
        if num_pixels >= PARALLEL_THRESHOLD {
            data.par_iter_mut().for_each(|value| *value *= scale);
        } else {
            for value in data.iter_mut() {
                *value *= scale;
            }
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
    r_sum: f64,
    g_sum: f64,
    b_sum: f64,
}

impl ChannelStats {
    fn add(&mut self, r: f32, g: f32, b: f32) {
        self.count += 1;
        self.r_sum += r as f64;
        self.g_sum += g as f64;
        self.b_sum += b as f64;
    }

    fn merge(mut self, other: Self) -> Self {
        self.count += other.count;
        self.r_sum += other.r_sum;
        self.g_sum += other.g_sum;
        self.b_sum += other.b_sum;
        self
    }

    #[allow(dead_code)]
    fn r_avg(&self) -> f32 {
        if self.count > 0 {
            (self.r_sum / self.count as f64) as f32
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn g_avg(&self) -> f32 {
        if self.count > 0 {
            (self.g_sum / self.count as f64) as f32
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn b_avg(&self) -> f32 {
        if self.count > 0 {
            (self.b_sum / self.count as f64) as f32
        } else {
            0.0
        }
    }
}

/// Limit how much gains can diverge from each other to preserve scene character.
///
/// This prevents aggressive color correction from removing natural scene warmth or coolness.
/// For example, a warm sunset scene should retain its warmth rather than being pushed neutral.
///
/// # Arguments
/// * `gains` - Per-channel gain values [R, G, B]
/// * `max_divergence` - Maximum allowed difference between highest and lowest gain (0.0-1.0)
///
/// # Returns
/// Adjusted gains with limited divergence, preserving relative proportions
pub fn limit_channel_divergence(gains: [f32; 3], max_divergence: f32) -> [f32; 3] {
    let min_g = gains[0].min(gains[1]).min(gains[2]);
    let max_g = gains[0].max(gains[1]).max(gains[2]);
    let current_divergence = max_g - min_g;

    if current_divergence <= max_divergence {
        return gains; // Already within limits
    }

    // Scale gains toward their mean to reduce divergence while preserving relative proportions
    let mean_gain = (gains[0] + gains[1] + gains[2]) / 3.0;
    let scale = max_divergence / current_divergence;

    [
        mean_gain + (gains[0] - mean_gain) * scale,
        mean_gain + (gains[1] - mean_gain) * scale,
        mean_gain + (gains[2] - mean_gain) * scale,
    ]
}

/// Auto-color: Scene-adaptive color correction that preserves scene character
///
/// Unlike traditional auto-color that targets pure neutral gray (R=G=B), this
/// implementation preserves natural scene color temperature while removing casts.
/// A warm sunny scene stays warm; a cool shaded scene stays cool.
///
/// # Arguments
/// * `data` - Interleaved RGB pixel data
/// * `channels` - Number of channels (must be 3)
/// * `strength` - Correction strength (0.0-1.0, higher = more neutralization)
/// * `min_gain` - Minimum per-channel multiplier
/// * `max_gain` - Maximum per-channel multiplier
/// * `max_divergence` - Maximum gain divergence (0.0-1.0, 0.15 = 15%)
///
/// # Returns
/// The applied adjustment factors [R, G, B] for debugging
pub fn auto_color(
    data: &mut [f32],
    channels: u8,
    strength: f32,
    min_gain: f32,
    max_gain: f32,
    max_divergence: f32,
) -> [f32; 3] {
    if channels != 3 {
        panic!("auto_color only supports 3-channel RGB images");
    }

    let num_pixels = data.len() / 3;

    // Compute full-image channel means
    // Using full-image means instead of midtone sampling ensures that when gains
    // are applied to all pixels, the resulting color balance matches expectations.
    // Midtone-only sampling can produce gains that don't neutralize the full image
    // when different luminance ranges have different color distributions.
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

    if num_pixels == 0 {
        return [1.0, 1.0, 1.0];
    }

    let count = num_pixels as f64;
    let r_avg = (r_sum / count) as f32;
    let g_avg = (g_sum / count) as f32;
    let b_avg = (b_sum / count) as f32;

    // Scene-adaptive targeting: blend between scene-preserving and neutral
    // Instead of forcing all channels to the same value, we preserve scene character
    let avg_luminance = (r_avg + g_avg + b_avg) / 3.0;

    // Target values that respect scene color temperature
    // At strength=1.0, we target neutral; at strength=0.0, we preserve scene completely
    let target_r = r_avg + strength * (avg_luminance - r_avg);
    let target_g = g_avg + strength * (avg_luminance - g_avg);
    let target_b = b_avg + strength * (avg_luminance - b_avg);

    // Calculate adjustment factors with per-channel gain limits
    let clamp_gain = |value: f32| value.clamp(min_gain, max_gain);

    let r_adjustment = if r_avg > 0.0001 {
        clamp_gain(target_r / r_avg)
    } else {
        1.0
    };
    let g_adjustment = if g_avg > 0.0001 {
        clamp_gain(target_g / g_avg)
    } else {
        1.0
    };
    let b_adjustment = if b_avg > 0.0001 {
        clamp_gain(target_b / b_avg)
    } else {
        1.0
    };

    // Limit divergence to preserve scene character
    let [r_adj, g_adj, b_adj] =
        limit_channel_divergence([r_adjustment, g_adjustment, b_adjustment], max_divergence);

    // Apply adjustments - parallel for large images
    parallel_for_each_chunk_mut(data, 3, |pixel| {
        pixel[0] = (pixel[0] * r_adj).clamp(0.0, 1.0);
        pixel[1] = (pixel[1] * g_adj).clamp(0.0, 1.0);
        pixel[2] = (pixel[2] * b_adj).clamp(0.0, 1.0);
    });

    // Return adjustments for debugging
    [r_adj, g_adj, b_adj]
}

#[allow(dead_code)]
fn collect_channel_stats(data: &[f32], low: f32, high: f32) -> ChannelStats {
    if low <= 0.0 && high >= 1.0 {
        // No brightness filtering needed
        parallel_fold_reduce(
            data,
            3,
            ChannelStats::default,
            |mut stats, pixel| {
                stats.add(pixel[0], pixel[1], pixel[2]);
                stats
            },
            ChannelStats::merge,
        )
    } else {
        // Filter by brightness range
        parallel_fold_reduce(
            data,
            3,
            ChannelStats::default,
            |mut stats, pixel| {
                let brightness = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
                if brightness >= low && brightness <= high {
                    stats.add(pixel[0], pixel[1], pixel[2]);
                }
                stats
            },
            ChannelStats::merge,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_color() {
        // Test with reddish cast
        let mut data = vec![
            0.6, 0.4, 0.4, // Reddish
            0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, 0.4,
        ];

        // max_divergence = 1.0 allows full correction for this test
        let adjustments = auto_color(&mut data, 3, 1.0, 0.7, 1.3, 1.0);

        println!("Color adjustments: {:?}", adjustments);
        println!("Corrected data: {:?}", data);

        // Red adjustment should be < 1.0 (reduce red)
        assert!(adjustments[0] < 1.0);
    }

    #[test]
    fn test_auto_color_divergence_limit() {
        // Test that divergence limiting preserves scene character
        let mut data = vec![
            0.6, 0.4, 0.35, // Warm scene
            0.55, 0.4, 0.35, 0.6, 0.42, 0.36, 0.58, 0.41, 0.34,
        ];
        let original = data.clone();

        // With low max_divergence (0.15), correction should be limited
        let adjustments = auto_color(&mut data, 3, 1.0, 0.7, 1.3, 0.15);

        println!("Original data: {:?}", original);
        println!("Corrected data: {:?}", data);
        println!("Adjustments: {:?}", adjustments);

        // Verify divergence is limited
        let divergence = adjustments[0].max(adjustments[1]).max(adjustments[2])
            - adjustments[0].min(adjustments[1]).min(adjustments[2]);
        assert!(
            divergence <= 0.15 + 0.001,
            "Divergence {} should be <= 0.15",
            divergence
        );
    }
}

//! Histogram analysis functions for the CB pipeline.
//!
//! Contains functions for computing per-channel histograms and finding
//! white/black clip points for negative inversion.

use crate::models::{CbChannelOrigins, CbHistogramAnalysis};

// ============================================================
// Histogram Computation
// ============================================================

/// Compute histogram for a single channel (256 bins)
pub fn compute_channel_histogram(data: &[f32], channel: usize, channels: u8) -> [u32; 256] {
    let mut histogram = [0u32; 256];
    let ch = channels as usize;

    for pixel in data.chunks(ch) {
        if channel < pixel.len() {
            // Convert 0.0-1.0 to 0-255 bin
            let bin = (pixel[channel] * 255.0).clamp(0.0, 255.0) as usize;
            histogram[bin] += 1;
        }
    }

    histogram
}

/// Find white and black points from histogram using Curves-based thresholds
///
/// CB uses cumulative percentage thresholds:
/// - white_threshold: percentage of pixels to clip at white (e.g., 0.0001 = 0.01%)
/// - black_threshold: percentage of pixels to clip at black (e.g., 0.0005 = 0.05%)
#[allow(clippy::needless_range_loop)] // Index needed for both array access and value computation
pub fn find_clip_points(
    histogram: &[u32; 256],
    total_pixels: u32,
    white_threshold: f32,
    black_threshold: f32,
) -> (f32, f32, f32) {
    let total = total_pixels as f32;

    // Find white point (scanning from bright to dark)
    let mut cumulative = 0.0;
    let mut white_point = 255.0;
    for i in (0..256).rev() {
        cumulative += histogram[i] as f32 / total;
        if cumulative > white_threshold {
            white_point = i as f32;
            break;
        }
    }

    // Find black point (scanning from dark to bright)
    cumulative = 0.0;
    let mut black_point = 0.0;
    for i in 0..256 {
        cumulative += histogram[i] as f32 / total;
        if cumulative > black_threshold {
            black_point = i as f32;
            break;
        }
    }

    // Calculate weighted mean within the clipped range
    let white_bin = white_point as usize;
    let black_bin = black_point as usize;
    let range_scale = 256.0 / (white_point - black_point).max(1.0);

    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;

    for i in black_bin..=white_bin.min(255) {
        let count = histogram[i] as f32;
        let normalized_value = ((i as f32) - black_point) * range_scale;
        weighted_sum += count * normalized_value;
        weight_total += count;
    }

    let mean_point = if weight_total > 0.0 {
        (weighted_sum / weight_total / 256.0).clamp(0.0, 1.0)
    } else {
        0.5
    };

    (white_point, black_point, mean_point)
}

/// Perform full histogram analysis on RGB image
pub fn analyze_histogram(
    data: &[f32],
    channels: u8,
    white_threshold: f32,
    black_threshold: f32,
) -> CbHistogramAnalysis {
    if white_threshold <= 0.0 && black_threshold <= 0.0 {
        return analyze_histogram_no_clip(data, channels);
    }

    let total_pixels = (data.len() / channels as usize) as u32;

    let r_hist = compute_channel_histogram(data, 0, channels);
    let g_hist = compute_channel_histogram(data, 1, channels);
    let b_hist = compute_channel_histogram(data, 2, channels);

    let (r_white, r_black, r_mean) =
        find_clip_points(&r_hist, total_pixels, white_threshold, black_threshold);
    let (g_white, g_black, g_mean) =
        find_clip_points(&g_hist, total_pixels, white_threshold, black_threshold);
    let (b_white, b_black, b_mean) =
        find_clip_points(&b_hist, total_pixels, white_threshold, black_threshold);

    CbHistogramAnalysis {
        red: CbChannelOrigins {
            white_point: r_white,
            black_point: r_black,
            mean_point: r_mean,
        },
        green: CbChannelOrigins {
            white_point: g_white,
            black_point: g_black,
            mean_point: g_mean,
        },
        blue: CbChannelOrigins {
            white_point: b_white,
            black_point: b_black,
            mean_point: b_mean,
        },
    }
}

fn analyze_histogram_no_clip(data: &[f32], channels: u8) -> CbHistogramAnalysis {
    let ch = channels as usize;
    let mut r_min = f32::INFINITY;
    let mut r_max = f32::NEG_INFINITY;
    let mut g_min = f32::INFINITY;
    let mut g_max = f32::NEG_INFINITY;
    let mut b_min = f32::INFINITY;
    let mut b_max = f32::NEG_INFINITY;
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut count = 0u32;

    for pixel in data.chunks_exact(ch) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        r_min = r_min.min(r);
        r_max = r_max.max(r);
        g_min = g_min.min(g);
        g_max = g_max.max(g);
        b_min = b_min.min(b);
        b_max = b_max.max(b);
        r_sum += r as f64;
        g_sum += g as f64;
        b_sum += b as f64;
        count += 1;
    }

    if count == 0 {
        return CbHistogramAnalysis {
            red: CbChannelOrigins {
                white_point: 255.0,
                black_point: 0.0,
                mean_point: 0.5,
            },
            green: CbChannelOrigins {
                white_point: 255.0,
                black_point: 0.0,
                mean_point: 0.5,
            },
            blue: CbChannelOrigins {
                white_point: 255.0,
                black_point: 0.0,
                mean_point: 0.5,
            },
        };
    }

    let r_avg = (r_sum / count as f64) as f32;
    let g_avg = (g_sum / count as f64) as f32;
    let b_avg = (b_sum / count as f64) as f32;

    let r_range = (r_max - r_min).abs();
    let g_range = (g_max - g_min).abs();
    let b_range = (b_max - b_min).abs();

    let r_mean = if r_range > 0.0001 {
        ((r_avg - r_min) / r_range).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let g_mean = if g_range > 0.0001 {
        ((g_avg - g_min) / g_range).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let b_mean = if b_range > 0.0001 {
        ((b_avg - b_min) / b_range).clamp(0.0, 1.0)
    } else {
        0.5
    };

    CbHistogramAnalysis {
        red: CbChannelOrigins {
            white_point: r_max * 255.0,
            black_point: r_min * 255.0,
            mean_point: r_mean,
        },
        green: CbChannelOrigins {
            white_point: g_max * 255.0,
            black_point: g_min * 255.0,
            mean_point: g_mean,
        },
        blue: CbChannelOrigins {
            white_point: b_max * 255.0,
            black_point: b_min * 255.0,
            mean_point: b_mean,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_channel_histogram() {
        // Create test data with known values
        let data = vec![
            0.0, 0.5, 1.0, // pixel 1: R=0, G=0.5, B=1
            0.0, 0.5, 1.0, // pixel 2: same
            0.25, 0.75, 0.5, // pixel 3: different
        ];

        let r_hist = compute_channel_histogram(&data, 0, 3);
        let g_hist = compute_channel_histogram(&data, 1, 3);
        let b_hist = compute_channel_histogram(&data, 2, 3);

        // Red channel: 2 pixels at 0, 1 pixel at ~64 (0.25*255)
        assert_eq!(r_hist[0], 2);
        assert_eq!(r_hist[63] + r_hist[64], 1); // 0.25 * 255 = 63.75

        // Green channel: 2 pixels at ~128, 1 at ~191
        assert!(g_hist[127] + g_hist[128] >= 2);

        // Blue channel: 2 pixels at 255, 1 at ~128
        assert_eq!(b_hist[255], 2);
    }

    #[test]
    fn test_histogram_analysis() {
        // Create test data with known distribution
        let mut data = vec![0.0; 300];
        for i in 0..100 {
            data[i * 3] = 0.2; // R
            data[i * 3 + 1] = 0.5; // G
            data[i * 3 + 2] = 0.8; // B
        }

        let analysis = analyze_histogram(&data, 3, 0.0001, 0.0005);

        // Check that analysis detected distinct values per channel
        // The white/black points should correspond to the input values (scaled to 0-255)
        assert!((analysis.red.white_point - 51.0).abs() < 2.0); // 0.2 * 255 = 51
        assert!((analysis.green.white_point - 127.5).abs() < 2.0); // 0.5 * 255 = 127.5
        assert!((analysis.blue.white_point - 204.0).abs() < 2.0); // 0.8 * 255 = 204
    }

    #[test]
    fn test_find_clip_points_uniform() {
        // Uniform histogram should have white at 255, black at 0
        let histogram = [100u32; 256];
        let total = 256 * 100;

        let (white, black, mean) = find_clip_points(&histogram, total, 0.001, 0.001);

        // With uniform distribution, white should be near 255, black near 0
        assert!(white > 250.0);
        assert!(black < 5.0);
        assert!((mean - 0.5).abs() < 0.1);
    }
}

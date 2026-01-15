//! Analysis and gain computation functions for GPU pipeline.

use crate::gpu::buffers::GpuImage;
use crate::gpu::context::{GpuContext, GpuError};
use crate::models::ConvertOptions;

/// Compute auto-levels gains from histogram data
pub fn compute_auto_levels_gains(
    hist_r: &[u32],
    hist_g: &[u32],
    hist_b: &[u32],
    clip_percent: f32,
) -> ([f32; 3], [f32; 3]) {
    fn find_percentile_bounds(hist: &[u32], clip: f32) -> (f32, f32) {
        let total: u64 = hist.iter().map(|&x| x as u64).sum();
        let clip_count = (total as f32 * clip / 100.0) as u64;

        let mut low_sum: u64 = 0;
        let mut low_idx = 0;
        for (i, &count) in hist.iter().enumerate() {
            low_sum += count as u64;
            if low_sum >= clip_count {
                low_idx = i;
                break;
            }
        }

        let mut high_sum: u64 = 0;
        let mut high_idx = hist.len() - 1;
        for (i, &count) in hist.iter().enumerate().rev() {
            high_sum += count as u64;
            if high_sum >= clip_count {
                high_idx = i;
                break;
            }
        }

        // Use (len - 1) to match CPU histogram bucket normalization
        let max_bucket = (hist.len() - 1) as f32;
        (low_idx as f32 / max_bucket, high_idx as f32 / max_bucket)
    }

    let (min_r, max_r) = find_percentile_bounds(hist_r, clip_percent);
    let (min_g, max_g) = find_percentile_bounds(hist_g, clip_percent);
    let (min_b, max_b) = find_percentile_bounds(hist_b, clip_percent);

    let gains = [
        1.0 / (max_r - min_r).max(0.001),
        1.0 / (max_g - min_g).max(0.001),
        1.0 / (max_b - min_b).max(0.001),
    ];

    let offsets = [min_r, min_g, min_b];

    (gains, offsets)
}

/// Limit how much gains can diverge from each other to preserve scene character.
/// Mirrors the CPU implementation in auto_adjust.rs.
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

/// Compute scene-adaptive auto-color gains using midtone-weighted means.
///
/// Uses a Gaussian weighting centered on midtones (luminance ~0.5) to compute
/// weighted channel averages. This ensures the color correction prioritizes
/// midtone neutrality, which is what the human eye is most sensitive to.
pub fn compute_auto_color_gains_fullimage(
    image: &GpuImage,
    _ctx: &GpuContext,
    options: &ConvertOptions,
) -> Result<[f32; 3], GpuError> {
    // Download image and compute midtone-weighted channel means
    let data = image.download()?;
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut weight_sum = 0.0f64;

    for pixel in data.chunks_exact(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        // Compute luminance
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Gaussian weight centered at 0.5 with sigma ~0.25
        // This gives high weight to midtones, low weight to shadows/highlights
        let dist = lum - 0.5;
        let weight = (-dist * dist / (2.0 * 0.25 * 0.25)).exp();

        r_sum += r * weight;
        g_sum += g * weight;
        b_sum += b * weight;
        weight_sum += weight;
    }

    if weight_sum < 0.001 {
        return Ok([1.0, 1.0, 1.0]);
    }

    let avg_r = (r_sum / weight_sum) as f32;
    let avg_g = (g_sum / weight_sum) as f32;
    let avg_b = (b_sum / weight_sum) as f32;

    eprintln!(
        "[AUTO-COLOR] Midtone-weighted avgs: R={:.4} G={:.4} B={:.4}",
        avg_r, avg_g, avg_b
    );
    eprintln!(
        "[AUTO-COLOR] Ratios: R/G={:.4} B/G={:.4}",
        avg_r / avg_g,
        avg_b / avg_g
    );

    let strength = options.auto_color_strength;
    let min_gain = options.auto_color_min_gain;
    let max_gain = options.auto_color_max_gain;

    // Asymmetric color correction:
    // - Warmth (R > G) is often natural scene character - preserve it
    // - Coolness/blue cast (B > G) is often a scanning artifact - correct it
    //
    // Instead of targeting neutral R=G=B, we use G as reference and:
    // - Only reduce R if it's significantly above G (which is rare for natural scenes)
    // - Reduce B toward G when B > G (remove blue cast)
    // - Boost R/B toward G when they're below G (remove cyan/yellow casts)

    // Calculate what gain would make each channel equal to G
    let r_to_neutral = avg_g / avg_r.max(0.001);
    let b_to_neutral = avg_g / avg_b.max(0.001);

    // For R channel: if R > G (warm), boost warmth for film look
    // If R < G (cyan cast), use full strength to correct
    let r_is_warm = avg_r > avg_g;
    let r_gain = if r_is_warm {
        // Warm scene: boost warmth for pleasing film look, scaled by strength
        // Film inversions typically lose warmth - compensate with up to ~12% boost
        // This mimics the warmer tones of professional scanner output
        // At strength=0, no change (1.0); at strength=1.0, full boost (1.12)
        1.0 + strength * 0.12
    } else {
        // Cyan cast: boost R toward G
        1.0 + strength * (r_to_neutral - 1.0)
    };

    // For B channel: if B > G (blue cast), use full strength to correct
    // If B < G (yellow cast, rare), use reduced strength
    let b_is_blue = avg_b > avg_g;
    let b_strength = if b_is_blue { strength } else { strength * 0.3 };
    let b_gain = 1.0 + b_strength * (b_to_neutral - 1.0);

    eprintln!(
        "[AUTO-COLOR] r_is_warm={} (preserving) b_is_blue={} (correcting)",
        r_is_warm, b_is_blue
    );
    eprintln!("[AUTO-COLOR] b_strength={:.3}", b_strength);

    let gains = [
        r_gain.clamp(min_gain, max_gain),
        1.0, // G is reference, no adjustment
        b_gain.clamp(min_gain, max_gain),
    ];

    eprintln!(
        "[AUTO-COLOR] raw gains: R={:.4} G={:.4} B={:.4}",
        gains[0], gains[1], gains[2]
    );

    // Limit divergence to preserve scene character
    let final_gains = limit_channel_divergence(gains, options.auto_color_max_divergence);
    eprintln!(
        "[AUTO-COLOR] final gains (after divergence limit): R={:.4} G={:.4} B={:.4}",
        final_gains[0], final_gains[1], final_gains[2]
    );

    Ok(final_gains)
}

/// Compute scene-adaptive auto-color gains from histogram.
///
/// Uses the same algorithm as the CPU implementation:
/// - Scene-adaptive targeting (preserves scene warmth/coolness)
/// - Channel divergence limiting to prevent aggressive neutralization
#[allow(dead_code)]
pub fn compute_auto_color_gains(
    hist_r: &[u32],
    hist_g: &[u32],
    hist_b: &[u32],
    options: &ConvertOptions,
) -> [f32; 3] {
    // Find average values in midtone range
    fn compute_weighted_average(hist: &[u32], low: f32, high: f32) -> f32 {
        let buckets = hist.len();
        let low_idx = (low * buckets as f32) as usize;
        let high_idx = (high * buckets as f32) as usize;

        let mut sum: f64 = 0.0;
        let mut count: u64 = 0;

        for (i, &bin_count) in hist
            .iter()
            .enumerate()
            .take(high_idx.min(buckets - 1) + 1)
            .skip(low_idx)
        {
            let value = i as f64 / buckets as f64;
            sum += value * bin_count as f64;
            count += bin_count as u64;
        }

        if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.5
        }
    }

    // Use wider range (0.1-0.9) to capture more representative sample
    // The narrow midtone range (0.35-0.65) can miss highlights/shadows that
    // have different color distributions
    let avg_r = compute_weighted_average(hist_r, 0.1, 0.9);
    let avg_g = compute_weighted_average(hist_g, 0.1, 0.9);
    let avg_b = compute_weighted_average(hist_b, 0.1, 0.9);

    eprintln!(
        "[GPU AUTO-COLOR] Wide range avgs: R={:.4} G={:.4} B={:.4}",
        avg_r, avg_g, avg_b
    );

    // Scene-adaptive targeting: blend between scene-preserving and neutral
    let avg_luminance = (avg_r + avg_g + avg_b) / 3.0;
    let strength = options.auto_color_strength;

    eprintln!(
        "[GPU AUTO-COLOR] avg_luminance={:.4} strength={:.4}",
        avg_luminance, strength
    );

    // Target values that respect scene color temperature
    // At strength=1.0, we target neutral; at strength=0.0, we preserve scene completely
    let target_r = avg_r + strength * (avg_luminance - avg_r);
    let target_g = avg_g + strength * (avg_luminance - avg_g);
    let target_b = avg_b + strength * (avg_luminance - avg_b);

    eprintln!(
        "[GPU AUTO-COLOR] targets: R={:.4} G={:.4} B={:.4}",
        target_r, target_g, target_b
    );

    let min_gain = options.auto_color_min_gain;
    let max_gain = options.auto_color_max_gain;

    // Calculate gains with per-channel limits
    let gains = [
        (target_r / avg_r.max(0.001)).clamp(min_gain, max_gain),
        (target_g / avg_g.max(0.001)).clamp(min_gain, max_gain),
        (target_b / avg_b.max(0.001)).clamp(min_gain, max_gain),
    ];

    eprintln!(
        "[GPU AUTO-COLOR] raw gains: R={:.4} G={:.4} B={:.4}",
        gains[0], gains[1], gains[2]
    );
    eprintln!(
        "[GPU AUTO-COLOR] max_divergence={:.4}",
        options.auto_color_max_divergence
    );

    // Limit divergence to preserve scene character
    limit_channel_divergence(gains, options.auto_color_max_divergence)
}

/// Compute white balance gains (requires downloading image data)
pub fn compute_wb_gains_cpu(image: &GpuImage, _ctx: &GpuContext) -> Result<[f32; 3], GpuError> {
    // Download image data for WB computation
    let data = image.download()?;

    // Find highlight/gray pixels and compute gains
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut count = 0u64;

    for pixel in data.chunks_exact(3) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Use highlight pixels (bright areas likely to be neutral)
        if lum > 0.6 {
            r_sum += r as f64;
            g_sum += g as f64;
            b_sum += b as f64;
            count += 1;
        }
    }

    if count < 100 {
        return Ok([1.0, 1.0, 1.0]); // Not enough samples
    }

    let r_avg = (r_sum / count as f64) as f32;
    let g_avg = (g_sum / count as f64) as f32;
    let b_avg = (b_sum / count as f64) as f32;

    // Normalize to green channel
    Ok([g_avg / r_avg.max(0.001), 1.0, g_avg / b_avg.max(0.001)])
}

/// Compute exposure gain (requires downloading image data for median)
///
/// Uses a highlight-aware algorithm that:
/// 1. Computes the gain needed to reach target median
/// 2. Limits the gain to preserve highlights (98th percentile stays below 0.95)
pub fn compute_exposure_gain_cpu(
    image: &GpuImage,
    _ctx: &GpuContext,
    options: &ConvertOptions,
) -> Result<f32, GpuError> {
    let data = image.download()?;

    // Collect luminance values
    let mut luminances: Vec<f32> = data
        .chunks_exact(3)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();

    if luminances.is_empty() {
        return Ok(1.0);
    }

    let n = luminances.len();

    // Find median (50th percentile)
    let mid = n / 2;
    let median = *luminances
        .select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .1;

    // Find 98th percentile for highlight preservation
    let p98_idx = (n * 98) / 100;
    let p98 = *luminances
        .select_nth_unstable_by(p98_idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .1;

    // Compute gain to reach target median
    let target = options.auto_exposure_target_median;
    let median_gain = target / median.max(0.001);

    // Compute maximum gain that keeps 98th percentile below 0.95
    // This prevents highlight clipping
    const HIGHLIGHT_CEILING: f32 = 0.95;
    let highlight_limit_gain = if p98 > 0.001 {
        HIGHLIGHT_CEILING / p98
    } else {
        options.auto_exposure_max_gain
    };

    // Use the minimum of median-based gain and highlight-preserving gain
    let gain = median_gain.min(highlight_limit_gain);

    // Clamp gain to configured limits
    Ok(gain.clamp(
        options.auto_exposure_min_gain,
        options.auto_exposure_max_gain,
    ))
}

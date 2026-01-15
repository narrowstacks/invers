//! Film base estimation methods
//!
//! Contains the main estimation functions for different strategies:
//! - Manual ROI: User-specified region
//! - Regions: Discrete border regions (top, bottom, left, right)
//! - Border: Sampling the outer border of the image
//! - Histogram: Whole-image histogram analysis

use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, BaseEstimationMethod};
use crate::verbose_println;

use super::analysis::{base_sample_fraction, compute_base_stats, validate_base_candidate};
use super::extraction::{
    estimate_base_roi_candidates, extract_border_pixels, extract_roi_pixels,
    filter_valid_base_pixels,
};
use super::BaseRoiCandidate;

/// Estimate film base from ROI, border, or heuristic regions
///
/// # Arguments
/// * `image` - The decoded image to analyze
/// * `roi` - Optional manual ROI (x, y, width, height). If provided, overrides method.
/// * `method` - Base estimation method (Regions or Border). Defaults to Regions.
/// * `border_percent` - Border percentage for Border method (1-25%). Defaults to 5.0.
pub fn estimate_base(
    image: &DecodedImage,
    roi: Option<(u32, u32, u32, u32)>,
    method: Option<BaseEstimationMethod>,
    border_percent: Option<f32>,
) -> Result<BaseEstimation, String> {
    let method = method.unwrap_or_default();
    let border_pct = border_percent.unwrap_or(5.0);

    // If ROI is provided, use manual mode regardless of method
    if let Some(rect) = roi {
        return estimate_base_from_manual_roi(image, rect);
    }

    // Otherwise, use the specified method
    match method {
        BaseEstimationMethod::Border => estimate_base_from_border(image, border_pct),
        BaseEstimationMethod::Regions => estimate_base_from_regions(image),
        BaseEstimationMethod::Histogram => estimate_base_from_histogram(image),
    }
}

/// Estimate film base from a manually specified ROI
pub fn estimate_base_from_manual_roi(
    image: &DecodedImage,
    rect: (u32, u32, u32, u32),
) -> Result<BaseEstimation, String> {
    let sample_fraction = base_sample_fraction();
    let candidate = BaseRoiCandidate::from_manual_roi(image, rect);

    let (x, y, width, height) = candidate.rect;

    if x + width > image.width || y + height > image.height || width == 0 || height == 0 {
        return Err(format!(
            "ROI out of bounds: {}x{} at ({}, {})",
            width, height, x, y
        ));
    }

    let roi_pixels = extract_roi_pixels(image, x, y, width, height);
    if roi_pixels.is_empty() {
        return Err("ROI contains no pixels".to_string());
    }

    let (num_brightest, percentage, medians, noise_stats) =
        compute_base_stats(&roi_pixels, sample_fraction);

    verbose_println!(
        "[BASE] Manual ROI | using {} px ({:.1}%) brightest",
        num_brightest,
        percentage
    );
    verbose_println!(
        "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
        medians[0],
        medians[1],
        medians[2],
        noise_stats[0],
        noise_stats[1],
        noise_stats[2]
    );

    let (valid, reason) = validate_base_candidate(&medians, &noise_stats, candidate.brightness);

    if !valid {
        verbose_println!("[BASE]   -> manual ROI has warnings: {}", reason);
        verbose_println!("[BASE]   -> using despite warnings (manual ROI)");
    } else {
        verbose_println!("[BASE]   -> accepted");
    }

    // Auto-detect mask profile from base color ratios
    let mask_profile = crate::models::MaskProfile::from_base_medians(&medians);

    Ok(BaseEstimation {
        roi: Some(rect),
        medians,
        noise_stats: Some(noise_stats),
        auto_estimated: false,
        mask_profile: Some(mask_profile),
    })
}

/// Estimate film base using discrete border regions (top, bottom, left, right)
pub fn estimate_base_from_regions(image: &DecodedImage) -> Result<BaseEstimation, String> {
    let sample_fraction = base_sample_fraction();

    let mut candidates = estimate_base_roi_candidates(image);

    if candidates.is_empty() {
        return Err("Failed to determine film base ROI".to_string());
    }

    // Sort by brightness (brightest first)
    candidates.sort_by(|a, b| {
        b.brightness
            .partial_cmp(&a.brightness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fallback: Option<BaseEstimation> = None;

    for candidate in &candidates {
        let (x, y, width, height) = candidate.rect;

        if x + width > image.width || y + height > image.height || width == 0 || height == 0 {
            verbose_println!(
                "[BASE] Skipping {} candidate: ROI out of bounds ({}x{} at {}, {})",
                candidate.label,
                width,
                height,
                x,
                y
            );
            continue;
        }

        let roi_pixels = extract_roi_pixels(image, x, y, width, height);
        if roi_pixels.is_empty() {
            verbose_println!(
                "[BASE] Skipping {} candidate: ROI is empty",
                candidate.label
            );
            continue;
        }

        // Filter out clipped/extreme pixels
        let filtered = filter_valid_base_pixels(roi_pixels);

        // Reject regions with too many clipped pixels (>50% clipped = saturated border)
        if filtered.clipped_ratio > 0.50 {
            verbose_println!(
                "[BASE] Skipping {} candidate: {:.0}% pixels clipped (saturated border)",
                candidate.label,
                filtered.clipped_ratio * 100.0
            );
            continue;
        }

        if filtered.pixels.is_empty() {
            verbose_println!(
                "[BASE] Skipping {} candidate: no valid pixels after filtering",
                candidate.label
            );
            continue;
        }

        let (num_brightest, percentage, medians, noise_stats) =
            compute_base_stats(&filtered.pixels, sample_fraction);

        verbose_println!(
            "[BASE] Candidate {:>6} | brightness={:.4} | using {} px ({:.1}%)",
            candidate.label,
            candidate.brightness,
            num_brightest,
            percentage
        );
        verbose_println!(
            "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
            medians[0],
            medians[1],
            medians[2],
            noise_stats[0],
            noise_stats[1],
            noise_stats[2]
        );
        if filtered.clipped_ratio > 0.1 {
            verbose_println!(
                "[BASE]   note: {:.0}% pixels were clipped",
                filtered.clipped_ratio * 100.0
            );
        }

        let (valid, reason) = validate_base_candidate(&medians, &noise_stats, candidate.brightness);

        // Auto-detect mask profile from base color ratios
        let mask_profile = crate::models::MaskProfile::from_base_medians(&medians);

        if valid {
            verbose_println!("[BASE]   -> accepted {} candidate", candidate.label);
            return Ok(BaseEstimation {
                roi: Some(candidate.rect),
                medians,
                noise_stats: Some(noise_stats),
                auto_estimated: true,
                mask_profile: Some(mask_profile),
            });
        } else {
            verbose_println!("[BASE]   -> rejected: {}", reason);
            if fallback.is_none() {
                fallback = Some(BaseEstimation {
                    roi: Some(candidate.rect),
                    medians,
                    noise_stats: Some(noise_stats),
                    auto_estimated: true,
                    mask_profile: Some(mask_profile),
                });
            }
        }
    }

    // Try histogram-based whole-image estimation as last resort
    verbose_println!("[BASE] All region candidates rejected; trying histogram-based estimation...");
    match estimate_base_from_histogram(image) {
        Ok(histogram_estimation) => {
            verbose_println!(
                "[BASE] Using histogram-based base: [{:.4}, {:.4}, {:.4}]",
                histogram_estimation.medians[0],
                histogram_estimation.medians[1],
                histogram_estimation.medians[2]
            );
            return Ok(histogram_estimation);
        }
        Err(e) => {
            verbose_println!("[BASE] Histogram estimation failed: {}", e);
        }
    }

    // Fall back to best rejected candidate
    if let Some(estimation) = fallback {
        if let Some(rect) = estimation.roi {
            if let Some(candidate) = candidates.iter().find(|c| c.rect == rect) {
                verbose_println!(
                    "[BASE] Final fallback to {} (brightness {:.4})",
                    candidate.label,
                    candidate.brightness
                );
            } else {
                verbose_println!("[BASE] Final fallback to brightest ROI");
            }
        }
        Ok(estimation)
    } else {
        Err("Unable to derive film base from available regions".to_string())
    }
}

/// Estimate film base by sampling the outer border of the image
///
/// This method:
/// 1. Extracts all pixels from the outer border_percent of the image
/// 2. Filters out clipped/near-white pixels (which are not valid film base)
/// 3. Finds the brightest N% of remaining pixels
/// 4. Computes per-channel medians from those brightest pixels
///
/// This is useful for full-frame scans that include the film rebate area,
/// where the unexposed film base is visible around the edges.
pub fn estimate_base_from_border(
    image: &DecodedImage,
    border_percent: f32,
) -> Result<BaseEstimation, String> {
    let sample_fraction = base_sample_fraction();

    verbose_println!(
        "[BASE] Using border method: sampling outer {:.1}% of image",
        border_percent
    );

    let border_pixels = extract_border_pixels(image, border_percent);

    if border_pixels.is_empty() {
        return Err("Border region contains no pixels".to_string());
    }

    verbose_println!("[BASE] Extracted {} border pixels", border_pixels.len());

    // Filter out clipped/extreme pixels
    let filtered = filter_valid_base_pixels(border_pixels);
    let filtered_count = filtered.pixels.len();
    verbose_println!(
        "[BASE] After filtering clipped/extreme pixels: {} remaining ({:.0}% clipped, {:.0}% dark)",
        filtered_count,
        filtered.clipped_ratio * 100.0,
        filtered.dark_ratio * 100.0
    );

    // Warn if too many pixels are clipped
    if filtered.clipped_ratio > 0.50 {
        verbose_println!(
            "[BASE] WARNING: >50% of border pixels are clipped - film base may be overexposed"
        );
    }

    if filtered.pixels.is_empty() {
        return Err(
            "No valid film base pixels found in border (all clipped or extreme)".to_string(),
        );
    }

    let (num_brightest, percentage, medians, noise_stats) =
        compute_base_stats(&filtered.pixels, sample_fraction);

    verbose_println!(
        "[BASE] Border | using {} px ({:.1}%) brightest",
        num_brightest,
        percentage
    );
    verbose_println!(
        "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
        medians[0],
        medians[1],
        medians[2],
        noise_stats[0],
        noise_stats[1],
        noise_stats[2]
    );

    // Calculate average brightness for validation
    let brightness: f32 = filtered
        .pixels
        .iter()
        .map(|p| 0.5 * p[0] + 0.4 * p[1] + 0.1 * p[2])
        .sum::<f32>()
        / filtered.pixels.len() as f32;

    let (valid, reason) = validate_base_candidate(&medians, &noise_stats, brightness);

    if valid {
        verbose_println!("[BASE]   -> border estimation accepted");
    } else {
        verbose_println!("[BASE]   -> border estimation has warnings: {}", reason);
    }

    // Auto-detect mask profile from base color ratios
    let mask_profile = crate::models::MaskProfile::from_base_medians(&medians);

    Ok(BaseEstimation {
        roi: None, // Border method doesn't use a specific ROI
        medians,
        noise_stats: Some(noise_stats),
        auto_estimated: true,
        mask_profile: Some(mask_profile),
    })
}

/// Estimate film base from whole-image histogram analysis
///
/// This method finds the mode (peak) of each channel's histogram in the upper
/// brightness range, avoiding clipped highlights. This works when border regions
/// don't contain clean film base.
pub fn estimate_base_from_histogram(image: &DecodedImage) -> Result<BaseEstimation, String> {
    const NUM_BINS: usize = 256;
    const MIN_BRIGHT: f32 = 0.30; // Only consider upper 70% of range
    const MAX_BRIGHT: f32 = 0.90; // Avoid clipped highlights

    let mut r_hist = [0u32; NUM_BINS];
    let mut g_hist = [0u32; NUM_BINS];
    let mut b_hist = [0u32; NUM_BINS];

    let pixels = image.data.chunks_exact(3);
    let mut total_valid = 0u64;

    for chunk in pixels {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];

        // Only consider pixels in the target brightness range
        let max_ch = r.max(g).max(b);
        let _min_ch = r.min(g).min(b);

        if (MIN_BRIGHT..=MAX_BRIGHT).contains(&max_ch) {
            // Bin each channel
            let r_bin = ((r * (NUM_BINS - 1) as f32) as usize).min(NUM_BINS - 1);
            let g_bin = ((g * (NUM_BINS - 1) as f32) as usize).min(NUM_BINS - 1);
            let b_bin = ((b * (NUM_BINS - 1) as f32) as usize).min(NUM_BINS - 1);

            r_hist[r_bin] += 1;
            g_hist[g_bin] += 1;
            b_hist[b_bin] += 1;
            total_valid += 1;
        }
    }

    if total_valid < 1000 {
        return Err("Insufficient valid pixels for histogram analysis".to_string());
    }

    // Find the highest peak in each channel's histogram (mode)
    // Focus on the upper portion of the histogram where film base would be
    let min_bin = (MIN_BRIGHT * (NUM_BINS - 1) as f32) as usize;
    let max_bin = (MAX_BRIGHT * (NUM_BINS - 1) as f32) as usize;

    let find_peak_in_range = |hist: &[u32; NUM_BINS]| -> f32 {
        let mut peak_bin = min_bin;
        let mut peak_count = 0u32;

        for (bin, &count) in hist.iter().enumerate().take(max_bin + 1).skip(min_bin) {
            if count > peak_count {
                peak_count = count;
                peak_bin = bin;
            }
        }

        // Convert bin back to value
        peak_bin as f32 / (NUM_BINS - 1) as f32
    };

    let r_peak = find_peak_in_range(&r_hist);
    let g_peak = find_peak_in_range(&g_hist);
    let b_peak = find_peak_in_range(&b_hist);

    // For color negative, we expect R > G > B in the base
    // If this pattern isn't present, reduce confidence
    let is_color_neg_pattern = r_peak >= g_peak * 0.9 && g_peak >= b_peak * 0.9;

    verbose_println!(
        "[BASE] Histogram peaks: R={:.4}, G={:.4}, B={:.4} ({})",
        r_peak,
        g_peak,
        b_peak,
        if is_color_neg_pattern {
            "color negative pattern"
        } else {
            "atypical pattern"
        }
    );

    // Auto-detect mask profile from base color ratios
    let medians = [r_peak, g_peak, b_peak];
    let mask_profile = crate::models::MaskProfile::from_base_medians(&medians);

    Ok(BaseEstimation {
        roi: None,
        medians,
        noise_stats: None, // Histogram method doesn't compute noise
        auto_estimated: true,
        mask_profile: Some(mask_profile),
    })
}

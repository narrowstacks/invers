//! Film base estimation algorithms
//!
//! This module contains functions for estimating the film base (orange mask)
//! from scanned color negative images. The base estimation is critical for
//! accurate negative-to-positive conversion.

use crate::config;
use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, BaseEstimationMethod};
use crate::verbose_println;

const MIN_BASE_SAMPLE_FRACTION: f32 = 0.01;
const MAX_BASE_SAMPLE_FRACTION: f32 = 0.30;
const BASE_VALIDATION_MIN_BRIGHTNESS: f32 = 0.25;
const BASE_VALIDATION_MAX_NOISE: f32 = 0.15;

/// Candidate ROI for base estimation
#[derive(Clone, Copy, Debug)]
pub struct BaseRoiCandidate {
    pub rect: (u32, u32, u32, u32),
    pub brightness: f32,
    pub label: &'static str,
}

impl BaseRoiCandidate {
    pub fn new(rect: (u32, u32, u32, u32), brightness: f32, label: &'static str) -> Self {
        Self {
            rect,
            brightness,
            label,
        }
    }

    pub fn from_manual_roi(image: &DecodedImage, rect: (u32, u32, u32, u32)) -> Self {
        let brightness = sample_region_brightness(image, rect.0, rect.1, rect.2, rect.3);
        Self {
            rect,
            brightness,
            label: "manual",
        }
    }
}

/// Filter results from base pixel filtering
struct FilteredBasePixels {
    /// Valid pixels that passed filtering
    pixels: Vec<[f32; 3]>,
    /// Ratio of pixels that were clipped (0.0-1.0)
    clipped_ratio: f32,
    /// Ratio of pixels that were too dark (0.0-1.0)
    dark_ratio: f32,
}

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

// ============================================================================
// Helper functions
// ============================================================================

fn base_sample_fraction() -> f32 {
    let defaults = &config::pipeline_config_handle().config.defaults;
    let fraction = defaults.base_brightest_percent / 100.0;
    fraction.clamp(MIN_BASE_SAMPLE_FRACTION, MAX_BASE_SAMPLE_FRACTION)
}

fn compute_base_stats(roi_pixels: &[[f32; 3]], fraction: f32) -> (usize, f32, [f32; 3], [f32; 3]) {
    let mut num_brightest = (roi_pixels.len() as f32 * fraction).ceil() as usize;
    num_brightest = num_brightest.max(10).min(roi_pixels.len());
    let percentage =
        ((num_brightest as f32 / roi_pixels.len() as f32) * 100.0 * 10.0).round() / 10.0;

    let medians = compute_channel_medians_from_brightest(roi_pixels, num_brightest);
    let noise_stats = compute_noise_stats(roi_pixels, &medians);

    (num_brightest, percentage, medians, noise_stats)
}

/// Detect if the image appears to be B&W based on channel similarity
fn is_likely_bw(medians: &[f32; 3]) -> bool {
    let [r, g, b] = *medians;
    let avg = (r + g + b) / 3.0;
    if avg <= 0.0 {
        return false;
    }
    // Check if all channels are within 15% of the average (low chroma)
    let deviation_threshold = 0.15;
    let r_dev = (r - avg).abs() / avg;
    let g_dev = (g - avg).abs() / avg;
    let b_dev = (b - avg).abs() / avg;
    r_dev < deviation_threshold && g_dev < deviation_threshold && b_dev < deviation_threshold
}

fn validate_base_candidate(
    medians: &[f32; 3],
    noise: &[f32; 3],
    brightness: f32,
) -> (bool, String) {
    let [r, g, b] = *medians;
    let max_noise = noise.iter().cloned().fold(0.0, f32::max);

    if brightness < BASE_VALIDATION_MIN_BRIGHTNESS {
        return (
            false,
            format!(
                "brightness {:.3} < {:.3}",
                brightness, BASE_VALIDATION_MIN_BRIGHTNESS
            ),
        );
    }

    // Adaptive noise threshold: scale based on brightness
    // Higher brightness regions tend to have more visible noise in scans
    let adaptive_noise_threshold = BASE_VALIDATION_MAX_NOISE * (1.0 + brightness * 0.5);
    if max_noise > adaptive_noise_threshold {
        return (
            false,
            format!(
                "noise {:.4} exceeds adaptive threshold {:.4}",
                max_noise, adaptive_noise_threshold
            ),
        );
    }

    if !(r.is_finite() && g.is_finite() && b.is_finite()) {
        return (false, "median contains non-finite values".to_string());
    }

    if r <= 0.0 || g <= 0.0 || b <= 0.0 {
        return (false, "median channel <= 0".to_string());
    }

    // Check if this appears to be B&W film
    let is_bw = is_likely_bw(medians);
    if is_bw {
        verbose_println!(
            "[BASE]   detected B&W film (low chroma), skipping orange mask validation"
        );
        return (true, "B&W film - all channels similar".to_string());
    }

    // Color film: validate orange mask characteristics
    let rg_ratio = r / g;
    let gb_ratio = g / b;

    if !(0.70..=2.20).contains(&rg_ratio) {
        return (
            false,
            format!("R/G ratio {:.3} outside expected range 0.70-2.20", rg_ratio),
        );
    }

    if !(1.00..=2.50).contains(&gb_ratio) {
        return (
            false,
            format!("G/B ratio {:.3} outside expected range 1.00-2.50", gb_ratio),
        );
    }

    if r < g || g < b {
        return (
            false,
            "channel ordering not orange-mask like (R >= G >= B expected)".to_string(),
        );
    }

    (true, "within expected range".to_string())
}

/// Filter out clipped/extreme pixels that are not valid film base
///
/// Film base should be bright but NOT clipped white. This filters:
/// - Near-white clipped pixels (all channels > 0.95)
/// - Very dark pixels (all channels < 0.05)
/// - Bright grayscale pixels without color variation (not orange mask)
fn filter_valid_base_pixels(pixels: Vec<[f32; 3]>) -> FilteredBasePixels {
    let total = pixels.len() as f32;
    let mut clipped_count = 0usize;
    let mut dark_count = 0usize;

    let valid_pixels: Vec<[f32; 3]> = pixels
        .into_iter()
        .filter(|p| {
            let [r, g, b] = *p;
            let max_val = r.max(g).max(b);
            let min_val = r.min(g).min(b);

            // Exclude near-white clipped pixels (any channel > 0.98 or all > 0.90)
            // More aggressive threshold to catch partially clipped pixels
            if max_val > 0.98 || min_val > 0.90 {
                clipped_count += 1;
                return false;
            }

            // Exclude very dark pixels (all channels < 0.05)
            if max_val < 0.05 {
                dark_count += 1;
                return false;
            }

            // For bright pixels (max > 0.7), require some color variation
            // (orange mask has R > G > B, not all equal)
            if max_val > 0.7 {
                let range = max_val - min_val;
                // Require at least 5% variation between channels for bright pixels
                if range < 0.05 {
                    return false;
                }
            }

            true
        })
        .collect();

    FilteredBasePixels {
        pixels: valid_pixels,
        clipped_ratio: if total > 0.0 {
            clipped_count as f32 / total
        } else {
            0.0
        },
        dark_ratio: if total > 0.0 {
            dark_count as f32 / total
        } else {
            0.0
        },
    }
}

/// Extract pixels from a region of interest with pre-allocation
fn extract_roi_pixels(
    image: &DecodedImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Vec<[f32; 3]> {
    // Pre-allocate with exact capacity
    let capacity = (width * height) as usize;
    let mut pixels = Vec::with_capacity(capacity);

    for row in y..(y + height) {
        let row_start = (row * image.width + x) as usize * 3;
        let row_pixels = ((x + width).min(image.width) - x) as usize;
        let row_end = row_start + row_pixels * 3;

        if row_end <= image.data.len() {
            // Process entire row at once for better cache locality
            for pixel in image.data[row_start..row_end].chunks_exact(3) {
                pixels.push([pixel[0], pixel[1], pixel[2]]);
            }
        }
    }

    pixels
}

/// Extract pixels from the outer border region of an image
///
/// The border is defined as the outer N% of the image dimensions, forming
/// a rectangular frame. A pixel is included if:
/// - x < border_width OR x >= (width - border_width), OR
/// - y < border_height OR y >= (height - border_height)
///
/// This is useful for sampling film base from full-frame scans that include
/// the rebate/sprocket area around the edges.
fn extract_border_pixels(image: &DecodedImage, border_percent: f32) -> Vec<[f32; 3]> {
    let border_percent = border_percent.clamp(1.0, 25.0);
    let border_width = ((image.width as f32) * border_percent / 100.0).ceil() as u32;
    let border_height = ((image.height as f32) * border_percent / 100.0).ceil() as u32;

    // Calculate border area for pre-allocation
    // Border area = total area - inner area
    let inner_width = image.width.saturating_sub(2 * border_width);
    let inner_height = image.height.saturating_sub(2 * border_height);
    let total_area = image.width as usize * image.height as usize;
    let inner_area = inner_width as usize * inner_height as usize;
    let border_area = total_area.saturating_sub(inner_area);

    let mut pixels = Vec::with_capacity(border_area);

    let right_edge = image.width.saturating_sub(border_width);
    let bottom_edge = image.height.saturating_sub(border_height);

    for y in 0..image.height {
        let is_vertical_border = y < border_height || y >= bottom_edge;

        for x in 0..image.width {
            let is_horizontal_border = x < border_width || x >= right_edge;

            // Include pixel if it's in the border region
            if is_vertical_border || is_horizontal_border {
                let idx = ((y * image.width + x) * 3) as usize;
                if idx + 2 < image.data.len() {
                    pixels.push([image.data[idx], image.data[idx + 1], image.data[idx + 2]]);
                }
            }
        }
    }

    pixels
}

/// Compute per-channel medians from the brightest N pixels
/// This samples the clearest film base without image content
///
/// For color negative film, we filter for pixels that match the orange mask
/// characteristics (R > G > B with typical G/B ratio 1.3-2.0) before selecting
/// the brightest pixels. This prevents scanner artifacts or edge effects with
/// elevated blue from skewing the base estimation.
fn compute_channel_medians_from_brightest(pixels: &[[f32; 3]], num_pixels: usize) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    // First, filter for pixels that match orange mask characteristics
    // Orange mask: R > G > B, with G/B ratio typically 1.3-2.0
    let orange_mask_pixels: Vec<[f32; 3]> = pixels
        .iter()
        .filter(|p| {
            let [r, g, b] = **p;
            // Require minimum brightness in each channel
            if r < 0.3 || g < 0.2 || b < 0.1 {
                return false;
            }
            // Require orange mask channel ordering
            if !(r > g && g > b) {
                return false;
            }
            // Check G/B ratio is in typical orange mask range
            // Wider range (1.15-2.5) to include more valid film bases
            // while still excluding obvious non-film pixels
            let gb_ratio = g / b;
            (1.15..=2.5).contains(&gb_ratio)
        })
        .copied()
        .collect();

    // If we have enough orange-mask pixels, use those; otherwise fall back to all pixels
    let working_pixels = if orange_mask_pixels.len() >= num_pixels.min(100) {
        verbose_println!(
            "[BASE]   filtered {} of {} pixels as orange-mask-like",
            orange_mask_pixels.len(),
            pixels.len()
        );
        orange_mask_pixels
    } else {
        verbose_println!(
            "[BASE]   only {} orange-mask pixels found, using all {} pixels",
            orange_mask_pixels.len(),
            pixels.len()
        );
        pixels.to_vec()
    };

    // Create a vec of (brightness, pixel) tuples
    let mut brightness_pixels: Vec<(f32, [f32; 3])> = working_pixels
        .iter()
        .map(|p| {
            let brightness = p[0] + p[1] + p[2]; // Sum of RGB as brightness
            (brightness, *p)
        })
        .collect();

    // Use partial sort to find top N brightest pixels (much faster than full sort)
    let n = num_pixels.min(brightness_pixels.len());
    let threshold_idx = brightness_pixels.len().saturating_sub(n);
    brightness_pixels.select_nth_unstable_by(threshold_idx, |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    // The brightest N pixels are now in the last n positions
    let brightest_slice = &brightness_pixels[threshold_idx..];

    // Compute median for each channel from these brightest pixels
    // Pre-allocate with exact capacity for efficiency
    let mut r_values: Vec<f32> = Vec::with_capacity(n);
    let mut g_values: Vec<f32> = Vec::with_capacity(n);
    let mut b_values: Vec<f32> = Vec::with_capacity(n);

    for (_, pixel) in brightest_slice {
        r_values.push(pixel[0]);
        g_values.push(pixel[1]);
        b_values.push(pixel[2]);
    }

    [
        compute_median(&mut r_values),
        compute_median(&mut g_values),
        compute_median(&mut b_values),
    ]
}

/// Compute median of a slice using partial sorting (much faster than full sort)
fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let len = values.len();
    let mid = len / 2;

    if len.is_multiple_of(2) {
        // Even length: average of two middle values
        // Use select_nth_unstable to partially sort only what we need
        values.select_nth_unstable_by(mid - 1, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let lower = values[mid - 1];
        values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let upper = values[mid];
        (lower + upper) / 2.0
    } else {
        // Odd length: middle value
        values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        values[mid]
    }
}

/// Compute noise statistics (standard deviation per channel)
fn compute_noise_stats(pixels: &[[f32; 3]], medians: &[f32; 3]) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = pixels.len() as f32;

    // Compute variance for each channel
    let mut var_r = 0.0;
    let mut var_g = 0.0;
    let mut var_b = 0.0;

    for pixel in pixels {
        var_r += (pixel[0] - medians[0]).powi(2);
        var_g += (pixel[1] - medians[1]).powi(2);
        var_b += (pixel[2] - medians[2]).powi(2);
    }

    var_r /= n;
    var_g /= n;
    var_b /= n;

    // Return standard deviations
    [var_r.sqrt(), var_g.sqrt(), var_b.sqrt()]
}

/// Estimate film base ROI candidates automatically using heuristics.
/// Order is determined later based on brightness.
fn estimate_base_roi_candidates(image: &DecodedImage) -> Vec<BaseRoiCandidate> {
    let border_width = (image.width / 20).clamp(50, 200);
    let border_height = (image.height / 20).clamp(50, 200);

    let candidates = [
        (
            sample_region_brightness(image, 0, 0, image.width, border_height),
            (
                border_width,
                0,
                image.width - 2 * border_width,
                border_height,
            ),
            "top",
        ),
        (
            sample_region_brightness(
                image,
                0,
                image.height.saturating_sub(border_height),
                image.width,
                border_height,
            ),
            (
                border_width,
                image.height.saturating_sub(border_height),
                image.width - 2 * border_width,
                border_height,
            ),
            "bottom",
        ),
        (
            sample_region_brightness(image, 0, 0, border_width, image.height),
            (
                0,
                border_height,
                border_width,
                image.height - 2 * border_height,
            ),
            "left",
        ),
        (
            sample_region_brightness(
                image,
                image.width.saturating_sub(border_width),
                0,
                border_width,
                image.height,
            ),
            (
                image.width.saturating_sub(border_width),
                border_height,
                border_width,
                image.height - 2 * border_height,
            ),
            "right",
        ),
    ];

    verbose_println!(
        "[BASE] Auto-detection brightnesses: top={:.4}, bottom={:.4}, left={:.4}, right={:.4}",
        candidates[0].0,
        candidates[1].0,
        candidates[2].0,
        candidates[3].0
    );

    // Create candidates from borders
    let mut result: Vec<BaseRoiCandidate> = candidates
        .iter()
        .map(|(brightness, rect, label)| BaseRoiCandidate::new(*rect, *brightness, label))
        .collect();

    // Add center region as a fallback candidate (for full-frame images)
    // Use a smaller center sample (10% of image) for potential sky/highlight areas
    let center_width = image.width / 10;
    let center_height = image.height / 10;
    let center_x = (image.width - center_width) / 2;
    let center_y = (image.height - center_height) / 2;

    if center_width > 20 && center_height > 20 {
        let center_brightness =
            sample_region_brightness(image, center_x, center_y, center_width, center_height);
        verbose_println!("[BASE] Center region brightness: {:.4}", center_brightness);
        result.push(BaseRoiCandidate::new(
            (center_x, center_y, center_width, center_height),
            center_brightness,
            "center",
        ));
    }

    result
}

/// Sample the average brightness of a region with color-aware weighting
/// For color negatives, film base is orange (high R+G, lower B)
/// Weight channels to detect orange mask: 0.5*R + 0.4*G + 0.1*B
/// Optimized to use single-pass accumulation
fn sample_region_brightness(image: &DecodedImage, x: u32, y: u32, width: u32, height: u32) -> f32 {
    let x_end = (x + width).min(image.width);
    let y_end = (y + height).min(image.height);

    let mut sum = 0.0;
    let mut count = 0;

    // Single pass with bounds checking
    for row in y..y_end {
        let row_start = (row * image.width + x) as usize * 3;
        let row_end = (row * image.width + x_end) as usize * 3;

        if row_end <= image.data.len() {
            // Process entire row at once for better cache locality
            for pixel in image.data[row_start..row_end].chunks_exact(3) {
                // Weight for orange mask detection (prioritizes R+G over B)
                sum += 0.5 * pixel[0] + 0.4 * pixel[1] + 0.1 * pixel[2];
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

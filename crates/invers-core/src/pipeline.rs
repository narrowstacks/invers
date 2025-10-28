//! Image processing pipeline
//!
//! Core pipeline for negative-to-positive conversion.

use crate::config;
use crate::decoders::DecodedImage;
use crate::models::{BaseEstimation, ConvertOptions};

/// Prevent values from ever hitting absolute black/white while retaining full range.
const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

const MIN_BASE_SAMPLE_FRACTION: f32 = 0.01;
const MAX_BASE_SAMPLE_FRACTION: f32 = 0.30;
const BASE_VALIDATION_MIN_BRIGHTNESS: f32 = 0.25;
const BASE_VALIDATION_MAX_NOISE: f32 = 0.15;

/// Result of the processing pipeline
pub struct ProcessedImage {
    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,

    /// Processed linear RGB data (f32)
    pub data: Vec<f32>,

    /// Number of channels
    pub channels: u8,
}

/// Execute the full processing pipeline
pub fn process_image(
    image: DecodedImage,
    options: &ConvertOptions,
) -> Result<ProcessedImage, String> {
    // Step 1: Get or compute base estimation
    let base_estimation = match &options.base_estimation {
        Some(base) => base.clone(),
        None => {
            // Compute base estimation from ROI if available, else auto-estimate
            // For now, we'll require base estimation to be provided or computed separately
            return Err(
                "Base estimation required. Use analyze-base command first or provide --roi"
                    .to_string(),
            );
        }
    };

    // Step 2: Move image data for processing (avoid an extra allocation)
    let DecodedImage {
        width,
        height,
        channels,
        mut data,
        ..
    } = image;

    if channels != 3 {
        return Err(format!("Pipeline requires 3-channel RGB, got {}", channels));
    }

    // Step 3: Base subtraction and inversion
    invert_negative(&mut data, &base_estimation, channels, options)?;

    if options.debug {
        let stats = compute_stats(&data);
        eprintln!(
            "[DEBUG] After inversion - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    // Step 3.05: Apply auto-levels (histogram stretching) - CRITICAL FOR MATCHING PHOTOSHOP
    // This is the key missing step that makes images 30% darker
    if options.enable_auto_levels {
        let params =
            crate::auto_adjust::auto_levels(&mut data, channels, options.auto_levels_clip_percent);

        if options.debug {
            eprintln!(
                "[DEBUG] After auto-levels (clip={:.1}%) - R:[{:.4}-{:.4}], G:[{:.4}-{:.4}], B:[{:.4}-{:.4}]",
                options.auto_levels_clip_percent,
                params[0], params[1], params[2], params[3], params[4], params[5]
            );
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG]   min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    } else if options.debug {
        eprintln!("[DEBUG] Auto-levels skipped");
    }

    // Step 3.1: Apply film-specific base offsets if available
    // These compensate for color negative orange mask characteristics
    if let Some(preset) = &options.film_preset {
        for pixel in data.chunks_exact_mut(3) {
            pixel[0] += preset.base_offsets[0];
            pixel[1] += preset.base_offsets[1];
            pixel[2] += preset.base_offsets[2];
        }

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!("[DEBUG] After base offsets [{:.3}, {:.3}, {:.3}] - min: {:.6}, max: {:.6}, mean: {:.6}",
                      preset.base_offsets[0], preset.base_offsets[1], preset.base_offsets[2],
                      stats.0, stats.1, stats.2);
        }
    }

    // Step 3.2: Apply auto-color (neutralize color casts)
    if options.enable_auto_color {
        let adjustments =
            crate::auto_adjust::auto_color(
                &mut data,
                channels,
                options.auto_color_strength,
                options.auto_color_min_gain,
                options.auto_color_max_gain,
            );

        if options.debug {
            eprintln!(
                "[DEBUG] After auto-color (strength={:.2}) - adjustments: R={:.4}, G={:.4}, B={:.4}",
                options.auto_color_strength,
                adjustments[0], adjustments[1], adjustments[2]
            );
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG]   min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    } else if options.debug {
        eprintln!("[DEBUG] Auto-color skipped");
    }

    // Step 3.3: Normalize exposure based on target median
    if options.enable_auto_exposure {
        let gain = crate::auto_adjust::auto_exposure(
            &mut data,
            options.auto_exposure_target_median,
            options.auto_exposure_strength,
            options.auto_exposure_min_gain,
            options.auto_exposure_max_gain,
        );

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG] Auto exposure (target={:.3}) gain={:.4} - min: {:.6}, max: {:.6}, mean: {:.6}",
                options.auto_exposure_target_median,
                gain,
                stats.0,
                stats.1,
                stats.2
            );
        }
    }

    // Step 3.4: Apply exposure compensation if requested
    if (options.exposure_compensation - 1.0).abs() > 0.001 {
        for value in data.iter_mut() {
            let scaled = *value * options.exposure_compensation;
            *value = clamp_to_working_range(scaled);
        }

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG] After exposure {:.2}x - min: {:.6}, max: {:.6}, mean: {:.6}",
                options.exposure_compensation, stats.0, stats.1, stats.2
            );
        }
    }

    // Step 4: Apply color correction matrix (unless skipped)
    // Apply color matrix BEFORE tone curve to correct film dye characteristics in linear space
    if !options.skip_color_matrix {
        if let Some(preset) = &options.film_preset {
            apply_color_matrix(&mut data, &preset.color_matrix, channels);

            if options.debug {
                let stats = compute_stats(&data);
                eprintln!(
                    "[DEBUG] After color matrix - min: {:.6}, max: {:.6}, mean: {:.6}",
                    stats.0, stats.1, stats.2
                );
            }
        }
    } else if options.debug {
        eprintln!("[DEBUG] Color matrix skipped");
    }

    // Step 5: Apply tone curve (unless skipped)
    if !options.skip_tone_curve {
        if let Some(preset) = &options.film_preset {
            apply_tone_curve(&mut data, &preset.tone_curve);
        } else {
            // Apply default neutral curve
            let default_curve = crate::models::ToneCurveParams::default();
            apply_tone_curve(&mut data, &default_curve);
        }

        if options.debug {
            let stats = compute_stats(&data);
            eprintln!(
                "[DEBUG] After tone curve - min: {:.6}, max: {:.6}, mean: {:.6}",
                stats.0, stats.1, stats.2
            );
        }
    } else if options.debug {
        eprintln!("[DEBUG] Tone curve skipped");
    }

    // Step 6: Colorspace transform (for now, keep in same space)
    // TODO: Implement colorspace transforms in M3

    // Final guard: keep values within photographic working range
    enforce_working_range(&mut data);

    // Return processed image
    Ok(ProcessedImage {
        width,
        height,
        data,
        channels,
    })
}

/// Estimate film base from ROI or heuristic
pub fn estimate_base(
    image: &DecodedImage,
    roi: Option<(u32, u32, u32, u32)>,
) -> Result<BaseEstimation, String> {
    let sample_fraction = base_sample_fraction();

    let mut candidates = if let Some(rect) = roi {
        vec![BaseRoiCandidate::from_manual_roi(image, rect)]
    } else {
        estimate_base_roi_candidates(image)
    };

    if candidates.is_empty() {
        return Err("Failed to determine film base ROI".to_string());
    }

    if roi.is_none() {
        candidates.sort_by(|a, b| b
            .brightness
            .partial_cmp(&a.brightness)
            .unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut fallback: Option<BaseEstimation> = None;

    for candidate in &candidates {
        let (x, y, width, height) = candidate.rect;

        if x + width > image.width || y + height > image.height || width == 0 || height == 0 {
            eprintln!(
                "[BASE] Skipping {} candidate: ROI out of bounds ({}x{} at {}, {})",
                candidate.label, width, height, x, y
            );
            continue;
        }

        let roi_pixels = extract_roi_pixels(image, x, y, width, height);
        if roi_pixels.is_empty() {
            eprintln!("[BASE] Skipping {} candidate: ROI is empty", candidate.label);
            continue;
        }

        let (num_brightest, percentage, medians, noise_stats) =
            compute_base_stats(&roi_pixels, sample_fraction);

        eprintln!(
            "[BASE] Candidate {:>6} | brightness={:.4} | using {} px ({:.1}%)",
            candidate.label, candidate.brightness, num_brightest, percentage
        );
        eprintln!(
            "[BASE]   medians=[{:.6}, {:.6}, {:.6}] noise=[{:.5}, {:.5}, {:.5}]",
            medians[0], medians[1], medians[2], noise_stats[0], noise_stats[1], noise_stats[2]
        );

        let (valid, reason) = validate_base_candidate(&medians, &noise_stats, candidate.brightness);

        if valid {
            eprintln!("[BASE]   -> accepted {} candidate", candidate.label);
            return Ok(BaseEstimation {
                roi: Some(candidate.rect),
                medians,
                noise_stats: Some(noise_stats),
                auto_estimated: roi.is_none(),
            });
        } else {
            eprintln!("[BASE]   -> rejected: {}", reason);
            if fallback.is_none() {
                fallback = Some(BaseEstimation {
                    roi: Some(candidate.rect),
                    medians,
                    noise_stats: Some(noise_stats),
                    auto_estimated: roi.is_none(),
                });
            }

            if candidate.manual {
                eprintln!("[BASE]   -> manual ROI provided; using despite warnings");
                return Ok(BaseEstimation {
                    roi: Some(candidate.rect),
                    medians,
                    noise_stats: Some(noise_stats),
                    auto_estimated: false,
                });
            }
        }
    }

    if let Some(estimation) = fallback {
        if let Some(rect) = estimation.roi {
            if let Some(candidate) = candidates.iter().find(|c| c.rect == rect) {
                eprintln!(
                    "[BASE] All auto candidates rejected; falling back to {} (brightness {:.4})",
                    candidate.label, candidate.brightness
                );
            } else {
                eprintln!("[BASE] All auto candidates rejected; using brightest ROI");
            }
        }
        Ok(estimation)
    } else {
        Err("Unable to derive film base from available regions".to_string())
    }
}

#[derive(Clone, Copy, Debug)]
struct BaseRoiCandidate {
    rect: (u32, u32, u32, u32),
    brightness: f32,
    label: &'static str,
    manual: bool,
}

impl BaseRoiCandidate {
    fn new(rect: (u32, u32, u32, u32), brightness: f32, label: &'static str) -> Self {
        Self {
            rect,
            brightness,
            label,
            manual: false,
        }
    }

    fn from_manual_roi(image: &DecodedImage, rect: (u32, u32, u32, u32)) -> Self {
        let brightness = sample_region_brightness(image, rect.0, rect.1, rect.2, rect.3);
        Self {
            rect,
            brightness,
            label: "manual",
            manual: true,
        }
    }
}

fn base_sample_fraction() -> f32 {
    let defaults = &config::pipeline_config_handle().config.defaults;
    let fraction = defaults.base_brightest_percent / 100.0;
    fraction.clamp(MIN_BASE_SAMPLE_FRACTION, MAX_BASE_SAMPLE_FRACTION)
}

fn compute_base_stats(
    roi_pixels: &[[f32; 3]],
    fraction: f32,
) -> (usize, f32, [f32; 3], [f32; 3]) {
    let mut num_brightest = (roi_pixels.len() as f32 * fraction).ceil() as usize;
    num_brightest = num_brightest.max(10).min(roi_pixels.len());
    let percentage =
        ((num_brightest as f32 / roi_pixels.len() as f32) * 100.0 * 10.0).round() / 10.0;

    let medians = compute_channel_medians_from_brightest(roi_pixels, num_brightest);
    let noise_stats = compute_noise_stats(roi_pixels, &medians);

    (num_brightest, percentage, medians, noise_stats)
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
            format!("brightness {:.3} < {:.3}", brightness, BASE_VALIDATION_MIN_BRIGHTNESS),
        );
    }

    if max_noise > BASE_VALIDATION_MAX_NOISE {
        return (
            false,
            format!("noise {:.4} exceeds {:.4}", max_noise, BASE_VALIDATION_MAX_NOISE),
        );
    }

    if !(r.is_finite() && g.is_finite() && b.is_finite()) {
        return (false, "median contains non-finite values".to_string());
    }

    if r <= 0.0 || g <= 0.0 || b <= 0.0 {
        return (false, "median channel <= 0".to_string());
    }

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
        return (false, "channel ordering not orange-mask like (R >= G >= B expected)".to_string());
    }

    (true, "within expected range".to_string())
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

/// Compute per-channel medians from the brightest N pixels
/// This samples the clearest film base without image content
fn compute_channel_medians_from_brightest(pixels: &[[f32; 3]], num_pixels: usize) -> [f32; 3] {
    if pixels.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    // Create a vec of (brightness, pixel) tuples
    let mut brightness_pixels: Vec<(f32, [f32; 3])> = pixels
        .iter()
        .map(|p| {
            let brightness = p[0] + p[1] + p[2]; // Sum of RGB as brightness
            (brightness, *p)
        })
        .collect();

    // Use partial sort to find top N brightest pixels (much faster than full sort)
    let n = num_pixels.min(brightness_pixels.len());
    let threshold_idx = brightness_pixels.len().saturating_sub(n);
    brightness_pixels.select_nth_unstable_by(
        threshold_idx,
        |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    );
    
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
        values.select_nth_unstable_by(mid - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lower = values[mid - 1];
        values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let upper = values[mid];
        (lower + upper) / 2.0
    } else {
        // Odd length: middle value
        values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    eprintln!(
        "[BASE] Auto-detection brightnesses: top={:.4}, bottom={:.4}, left={:.4}, right={:.4}",
        candidates[0].0, candidates[1].0, candidates[2].0, candidates[3].0
    );

    candidates
        .iter()
        .map(|(brightness, rect, label)| BaseRoiCandidate::new(*rect, *brightness, label))
        .collect()
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

/// Subtract base and invert to positive
pub fn invert_negative(
    data: &mut [f32],
    base: &BaseEstimation,
    channels: u8,
    options: &crate::models::ConvertOptions,
) -> Result<(), String> {
    if channels != 3 {
        return Err(format!("Expected 3 channels, got {}", channels));
    }

    if !data.len().is_multiple_of(3) {
        return Err(format!(
            "Data length {} is not divisible by 3 (RGB)",
            data.len()
        ));
    }

    let base_r = base.medians[0].max(0.0001);
    let base_g = base.medians[1].max(0.0001);
    let base_b = base.medians[2].max(0.0001);

    // Apply inversion based on mode
    match options.inversion_mode {
        crate::models::InversionMode::Linear => {
            // Linear inversion: positive = (base - negative) / base
            for pixel in data.chunks_exact_mut(3) {
                pixel[0] = (base_r - pixel[0]) / base_r;
                pixel[1] = (base_g - pixel[1]) / base_g;
                pixel[2] = (base_b - pixel[2]) / base_b;
            }
        }
        crate::models::InversionMode::Logarithmic => {
            // Logarithmic (density-based) inversion
            // positive = 10^(log10(base) - log10(negative))
            let log_base_r = base_r.log10();
            let log_base_g = base_g.log10();
            let log_base_b = base_b.log10();

            for pixel in data.chunks_exact_mut(3) {
                let neg_r = pixel[0].max(0.0001);
                let neg_g = pixel[1].max(0.0001);
                let neg_b = pixel[2].max(0.0001);

                pixel[0] = 10f32.powf(log_base_r - neg_r.log10()).clamp(0.0, 10.0);
                pixel[1] = 10f32.powf(log_base_g - neg_g.log10()).clamp(0.0, 10.0);
                pixel[2] = 10f32.powf(log_base_b - neg_b.log10()).clamp(0.0, 10.0);
            }
        }
    }

    // Apply shadow lift based on mode
    match options.shadow_lift_mode {
        crate::models::ShadowLiftMode::Fixed => {
            // Find minimum value to see if lift is needed
            let mut min_value = f32::MAX;
            for &value in data.iter() {
                min_value = min_value.min(value);
            }

            let shadow_lift = if min_value < 0.0 {
                -min_value + options.shadow_lift_value
            } else {
                options.shadow_lift_value
            };

            // Apply uniform lift
            for value in data.iter_mut() {
                *value += shadow_lift;
            }

            if options.debug {
                eprintln!("[DEBUG] Fixed shadow lift: {:.6}", shadow_lift);
            }
        }
        crate::models::ShadowLiftMode::Percentile => {
            // Use adaptive shadow lift based on 1st percentile
            let lift = crate::auto_adjust::adaptive_shadow_lift(
                data,
                options.shadow_lift_value,
                1.0, // 1st percentile
            );

            if options.debug {
                eprintln!("[DEBUG] Percentile shadow lift: {:.6}", lift);
            }
        }
        crate::models::ShadowLiftMode::None => {
            // No shadow lift, but clamp negatives to zero
            for value in data.iter_mut() {
                *value = value.max(0.0);
            }
        }
    }

    // Apply highlight compression if enabled
    if options.highlight_compression < 1.0 {
        crate::auto_adjust::compress_highlights(data, 0.9, options.highlight_compression);

        if options.debug {
            eprintln!(
                "[DEBUG] Highlight compression: {:.2}",
                options.highlight_compression
            );
        }
    }

    Ok(())
}

/// Apply tone curve
pub fn apply_tone_curve(data: &mut [f32], curve_params: &crate::models::ToneCurveParams) {
    match curve_params.curve_type.as_str() {
        "linear" => {
            // No transformation needed
        }
        "neutral" | "s-curve" => {
            // Apply S-curve with configurable strength
            apply_s_curve(data, curve_params.strength);
        }
        _ => {
            // Unknown curve type, apply neutral S-curve
            apply_s_curve(data, curve_params.strength);
        }
    }
}

/// Apply S-curve tone mapping for natural film-like contrast
/// Strength: 0.0 = no curve (linear), 1.0 = maximum curve
fn apply_s_curve(data: &mut [f32], strength: f32) {
    // Clamp strength to valid range
    let strength = strength.clamp(0.0, 1.0);

    if strength < 0.01 {
        // Effectively linear, no transformation needed
        return;
    }

    // S-curve using a smoothstep-like function with adjustable strength
    // Formula: smoothstep with variable steepness
    for value in data.iter_mut() {
        *value = apply_s_curve_point(*value, strength);
    }
}

/// Apply S-curve transformation to a single value
/// Uses a modified smoothstep function with adjustable contrast
fn apply_s_curve_point(x: f32, strength: f32) -> f32 {
    // Clamp input to valid range
    let x = x.clamp(0.0, 1.0);

    // Blend between linear and S-curve based on strength
    // S-curve uses a smoothstep-like function: 3x^2 - 2x^3
    // For more contrast, we can use higher-order polynomials

    // Calculate S-curve value using smoothstep
    let s_value = if x < 0.5 {
        // Shadow region: lift shadows slightly
        let t = x * 2.0;
        let smooth = t * t * (3.0 - 2.0 * t);
        smooth * 0.5
    } else {
        // Highlight region: compress highlights slightly
        let t = (x - 0.5) * 2.0;
        let smooth = t * t * (3.0 - 2.0 * t);
        0.5 + smooth * 0.5
    };

    // Apply contrast adjustment based on strength
    // Stronger strength = more pronounced S-curve
    let contrast_factor = 1.0 + strength * 0.5;
    let adjusted = (s_value - 0.5) * contrast_factor + 0.5;

    // Blend between original linear value and S-curve
    let result = x * (1.0 - strength) + adjusted * strength;

    clamp_to_working_range(result)
}

/// Apply color correction matrix
/// Performs 3x3 matrix multiplication on RGB pixels
pub fn apply_color_matrix(data: &mut [f32], matrix: &[[f32; 3]; 3], channels: u8) {
    if channels != 3 {
        return; // Only works for RGB
    }

    // Process each RGB pixel
    for pixel in data.chunks_exact_mut(3) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        // Matrix multiplication: output = matrix * input
        // [R'] = [m00 m01 m02] [R]
        // [G']   [m10 m11 m12] [G]
        // [B']   [m20 m21 m22] [B]

        pixel[0] = clamp_to_working_range(matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b);
        pixel[1] = clamp_to_working_range(matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b);
        pixel[2] = clamp_to_working_range(matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b);
    }
}

/// Clamp all values into the working range to avoid clipped blacks or whites.
fn enforce_working_range(data: &mut [f32]) {
    for value in data.iter_mut() {
        *value = clamp_to_working_range(*value);
    }
}

#[inline]
fn clamp_to_working_range(value: f32) -> f32 {
    value.clamp(WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING)
}

/// Compute min, max, and mean statistics for debug output
fn compute_stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0;

    for &value in data {
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }

    let mean = sum / data.len() as f32;
    (min, max, mean)
}
